import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data import Composite, Bounded, Unbounded
from torchrl.envs import EnvBase

from .environment import Environment


class EnvironmentTorch(Environment, EnvBase):
    def __init__(self, *args, **kwargs):
        # Call the parent constructor of Environment
        super().__init__(*args, **kwargs)

        # So that TransformedEnv to work
        self._modules = {}
        self._allow_done_after_reset = False
        self.run_type_checks = False

        self.device = (
            torch.device(0)
            if torch.cuda.is_available() and torch.multiprocessing.get_start_method() != "fork"
            else torch.device("cpu")
        )

        # We need to call _make_spec to initialize the specs
        self._make_spec()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict["action"].cpu().numpy()

        obs, reward, terminated, truncated, info, agent_input = super().step(action)

        # Convert to tensors
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(1)
        terminated_t = torch.tensor(terminated, dtype=torch.bool, device=self.device).reshape(1)
        truncated_t = torch.tensor(truncated, dtype=torch.bool, device=self.device).reshape(1)
        done_t = terminated_t | truncated_t 
        next_observation_t = agent_input.squeeze(0).to(self.device)

        # Create the output tensordict
        return TensorDict(
            {
                "observation": tensordict["observation"].clone() ,
                "action": tensordict["action"].clone(),
                "terminated": terminated_t,
                "truncated": truncated_t,
                "done": done_t,
                "reward": reward_t,
                # 'next' contains the resulting state
                "next": TensorDict(
                    {
                        "observation": next_observation_t,
                        "reward": reward_t,
                        "terminated": terminated_t,
                        "truncated": truncated_t,
                        "done": done_t,
                    },
                    batch_size=torch.Size(),
                    device=self.device
                ),
            },
            batch_size=torch.Size(),
            device=self.device
        )

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._step(tensordict)

    def _reset(self, tensordict: TensorDictBase | None) -> TensorDictBase:
        obs, info, agent_input = super().reset()

        return TensorDict(
            {
                "observation": agent_input.squeeze(0).to(self.device), 
                "done": torch.tensor(False, dtype=torch.bool, device=self.device),
                "terminated": torch.tensor(False, dtype=torch.bool, device=self.device),
                "truncated": torch.tensor(False, dtype=torch.bool, device=self.device),
            },
            batch_size=torch.Size(),
            device=self.device
        )

    def reset(self, tensordict: TensorDictBase | None) -> TensorDictBase:
        return self._reset(tensordict)

    def _make_spec(self):
        # Get a sample agent input to determine the shape
        obs, info, agent_input = super().reset()
        self.close()  # Close the env opened for spec creation
        
        # Determine the shape from the agent_input
        obs_shape = tuple(agent_input.squeeze(0).shape)

        self.observation_spec = Composite(
            # Define the shape correctly
            observation=Unbounded(shape=obs_shape, device=self.device),
            shape=(),
            device=self.device
        )

        self.state_spec = self.observation_spec.clone()

        self.action_spec = Bounded(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=self.env.action_space.shape,
            device=self.device
        )

        self.reward_spec = Unbounded(shape=(1,), device=self.device)

        self.done_spec = Composite(
            terminated=Unbounded(shape=(1,), dtype=torch.bool, device=self.device),
            truncated=Unbounded(shape=(1,), dtype=torch.bool, device=self.device),
            done=Unbounded(shape=(1,), dtype=torch.bool, device=self.device),
            shape=(),
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        pass
