import logging, warnings
from collections import defaultdict
from tqdm import tqdm

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import Compose, DoubleToFloat, StepCounter, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

from ..environments import EnvironmentTorch
from .agent import Agent

class PPOTrainer:
    def __init__(self,
        env: EnvironmentTorch,
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
        self.env = TransformedEnv(
            env,
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
        self.agent.loadWeights(f"{self.weight_dir}/{weight_file}")
        self.agent = self.agent.to(self.device)

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
            env_device=self.device
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
            self.loss_module.parameters(),
            lr
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, total_frames // frames_per_batch, 0.0001
        )

    def _logData(self, logs: defaultdict):
        cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

        eval_str = (
            f"eval average reward: {logs['eval reward'][-1]: 4.4f} "
            f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
            f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
            f"eval step-count: {logs['eval step_count'][-1]}"
        )

        logger.info(", ".join([cum_reward_str, stepcount_str, lr_str, eval_str]))

    def _train(self) -> defaultdict:
        logs = defaultdict(list)

        for _, tensordict_data in enumerate(tqdm(self.collector, total=self.total_frames // self.frames_per_batch)):
            for _ in range(self.num_epochs):
                self.advantage_module(tensordict_data)  # The advantage signal depends on the value network trained below

                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view.cpu())
                for _ in range(self.frames_per_batch // self.sub_batch_size):
                    subdata = self.replay_buffer.sample(self.sub_batch_size)
                    loss_vals = self.loss_module(subdata.to(self.device))
                    loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()

            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            logs["lr"].append(self.optim.param_groups[0]["lr"])

            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = self.env.rollout(1000, self.policy_module)

                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())

                del eval_rollout

            self._logData(logs)
            self.scheduler.step()

        return logs

    def train(self) -> defaultdict | None:
        data = None
        try:
            data = self._train()
        except (KeyboardInterrupt, Exception) as e:
            logger.error(f"Training stopped due to the following exception: {e}")
        finally:
            self.agent.saveWeights(self.weight_dir, prefix=self.prefix)

        return data
