import logging, warnings
from collections import defaultdict
from tqdm import tqdm

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import Compose, DoubleToFloat, StepCounter, TransformedEnv, ParallelEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

from ..environments import EnvironmentTorch
from .agent import Agent


def _create_env_instance(env_cls, headless, max_episodes):
    """Factory function for creating environment instances in worker processes.
    Must be at module level to be picklable."""
    return env_cls(headless=headless, max_episodes=max_episodes)


class PPOTrainer:
    def __init__(self,
        env: EnvironmentTorch,
        num_envs: int,
        n_states: int,
        n_actions: int,
        lr: float,
        max_grad_norm: float,
        frames_per_batch: int,
        total_frames: int,
        sub_batch_size: int,
        num_epochs: int,
        clip_epsilon: float,
        gamma: float,
        lmbda: float,
        entropy_eps: float,
        weight_dir: str,
        weight_file: str,
        prefix: str
    ):
        from functools import partial
        
        # Store the env class and initialization parameters
        env_cls = type(env)
        
        # Extract initialization parameters from the environment instance
        # These need to be picklable (no MuJoCo objects)
        headless = True  # Always use headless for parallel workers
        max_episodes = env.max_episodes if hasattr(env, 'max_episodes') else 10000
        
        # Create a picklable factory function using partial
        make_env = partial(_create_env_instance, env_cls, headless, max_episodes)
        
        # Build the vectorized environment using TorchRL's ParallelEnv
        self.base_env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=make_env,
        )

        self.env = TransformedEnv(
            self.base_env,
            Compose(
                DoubleToFloat(),
                StepCounter(),
            ),
        )
        self.device = self.env.device

        self.max_grad_norm = max_grad_norm

        self.total_frames = total_frames
        self.frames_per_batch = frames_per_batch
        self.sub_batch_size = sub_batch_size
        self.num_epochs = num_epochs

        self.weight_dir = weight_dir
        self.prefix = prefix

        self.agent = Agent(
            n_states=n_states,
            n_actions=n_actions
        )
        try:
            self.agent.loadWeights(f"{self.weight_dir}/{weight_file}")
        except:
            logger.error(f"Couldn't load the weights {weight_file}")

        self.agent = self.agent.to(self.device)

        self.prediction_module = TensorDictModule(
            self.agent.auxiliary_head,
            in_keys=["observation"],
            out_keys=["state_prediction"]
        )

        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(
                self.agent.actor_head,
                in_keys=["observation"],
                out_keys=["loc", "scale"]
            ),
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": self.env.action_spec.space.low,
                "high": self.env.action_spec.space.high,
            },
            return_log_prob=True,  # Used for the numerator of the importance weights
        )

        self.value_module = ValueOperator(
            module=self.agent.critic_head,
            in_keys=["observation"],
        )

        self.collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=self.device,
            storing_device=self.device,
            policy_device=self.device,
            env_device="cpu"
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=frames_per_batch,
                device=self.device,
            ),
            sampler=SamplerWithoutReplacement(),
        )

        self.advantage_module = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.value_module,
            average_gae=True,
            device=self.device,
        )

        self.aux_loss_fn = torch.nn.MSELoss()

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
            device=self.device
        )

        self.optim = torch.optim.Adam(
            self.agent.parameters(),
            lr
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, total_frames // frames_per_batch, 0.0001
        )

    def _logData(self, logs: defaultdict):
        avg_reward_str = f"average reward={logs['reward'][-1]:4.4f} (init={logs['reward'][0]:4.4f})"
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        cum_reward_str = f"cumulative reward (max): {logs['reward'][-1] * logs['step_count'][-1]:4.4f}"
        lr_str = f"lr policy: {logs['lr'][-1]:4.4f}"
        loss_str = f"loss_objective: {logs['loss_objective'][-1]:4.4f}, loss_critic: {logs['loss_critic'][-1]:4.4f}, loss_entropy: {logs['loss_entropy'][-1]:4.4f}, loss_auxiliary: {logs['loss_auxiliary'][-1]:4.4f}"

        logger.info(", ".join([avg_reward_str, stepcount_str, cum_reward_str, lr_str, loss_str]))

    def _train(self) -> defaultdict:
        logs = defaultdict(list)

        for _, tensordict_data in enumerate(tqdm(self.collector, total=self.total_frames // self.frames_per_batch)):
            with torch.no_grad():
                self.advantage_module(tensordict_data)  # The advantage signal depends on the value network trained below

            data_view = tensordict_data.reshape(-1)
            self.replay_buffer.extend(data_view.cpu())

            loss_obj, loss_critic, loss_entropy, loss_aux = 0, 0, 0, 0
            for _ in range(self.num_epochs):
                for _ in range(self.frames_per_batch // self.sub_batch_size):
                    subdata = self.replay_buffer.sample(self.sub_batch_size)

                    self.prediction_module(subdata)

                    aux_target = subdata.get(("next", "observation"))
                    aux_prediction = subdata.get("state_prediction")

                    loss_vals = self.loss_module(subdata.to(self.device))
                    aux_loss = self.aux_loss_fn(aux_prediction.to(self.device), aux_target.to(self.device))

                    loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"] + aux_loss

                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()

                    loss_obj += loss_vals["loss_objective"].item()
                    loss_critic += loss_vals["loss_critic"].item()
                    loss_entropy += loss_vals["loss_entropy"].item()
                    loss_aux += aux_loss.item()

            n = self.num_epochs * self.frames_per_batch // self.sub_batch_size
            logs["loss_objective"].append(loss_obj / n)
            logs["loss_critic"].append(loss_critic / n)
            logs["loss_entropy"].append(loss_entropy / n)
            logs["loss_auxiliary"].append(loss_aux / n)

            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            logs["lr"].append(self.optim.param_groups[0]["lr"])

            self._logData(logs)
            self.scheduler.step()
            self.collector.update_policy_weights_()

        return logs

    def train(self) -> defaultdict | None:
        data = None
        try:
            data = self._train()
        except (KeyboardInterrupt, Exception) as e:
            logger.error(f"Training stopped due to the following exception: {e}")
        finally:
            self.agent.saveWeights(self.weight_dir, prefix=self.prefix)
            try:
                self.collector.shutdown()  # ensure collector closes its envs
            except Exception as shutdown_err:
                logger.debug(f"Collector shutdown skipped: {shutdown_err}")
            try:
                self.base_env.close()      # close if still open
            except RuntimeError:
                logger.debug("Base env already closed; skipping.")
            except Exception as close_err:
                logger.warning(f"Error closing base env: {close_err}")
        return data