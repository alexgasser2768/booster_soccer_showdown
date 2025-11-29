import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor

import time
from typing import Tuple

LAYER_SIZE = 256


class Agent(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Agent, self).__init__()

        # Match the shared network architecture
        self.shared_net = nn.Sequential(
            nn.Linear(n_states, LAYER_SIZE),
            nn.LeakyReLU(),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
        )

        self.actor_head = nn.Sequential(
            self.shared_net,
            nn.Linear(LAYER_SIZE, 2*n_actions),
            NormalParamExtractor(),
        )

        self.critic_head = nn.Sequential(
            self.shared_net,
            nn.Linear(LAYER_SIZE, 1)
        )

        # Constants (adapted from booster_control/t1_utils.py)
        self.register_buffer("default_dof_pos", torch.tensor(
            [-0.2, 0.0, 0.0, 0.4, -0.25, 0.0, -0.2, 0.0, 0.0, 0.4, -0.25, 0.0], dtype=torch.float32))
        self.register_buffer("dof_stiffness", torch.tensor(
            [200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 200.0, 200.0, 200.0, 200.0, 50.0, 50.0], dtype=torch.float32))
        self.register_buffer("dof_damping", torch.tensor(
            [5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0], dtype=torch.float32))
        self.register_buffer("ctrl_min", torch.tensor(
            [-45, -45, -30, -65, -24, -15, -45, -45, -30, -65, -24, -15], dtype=torch.float32))
        self.register_buffer("ctrl_max", torch.tensor(
            [45, 45, 30, 65, 24, 15, 45, 45, 30, 65, 24, 15], dtype=torch.float32))

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loc, scale = self.actor_head(x)  # location and scale

        return loc

        # # PD control + Clamp (adapted from booster_control/t1_utils.py)
        # qpos = x[:, :12]
        # qvel = x[:, 12:24]

        # targets = self.default_dof_pos.expand(x.shape[0], -1) + loc
        # ctrl = self.dof_stiffness * (targets - qpos) - self.dof_damping * qvel

        # return torch.minimum(
        #     torch.maximum(ctrl, self.ctrl_min.expand_as(ctrl)),
        #     self.ctrl_max.expand_as(ctrl)
        # )

    def saveWeights(self, directory, prefix = "") -> str:
        name = f"{prefix}-{time.time()}.pt"
        torch.save(self.state_dict(), f"{directory}/{name}")
        return name

    def loadWeights(self, path):
        self.load_state_dict(torch.load(path))