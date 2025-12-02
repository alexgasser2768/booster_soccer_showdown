import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor

import time
from typing import Tuple

from ..booster_control import joint_velocities_to_actions

LAYER_SIZE = 512


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
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
        )

        self.actor_head = nn.Sequential(
            self.shared_net,
            nn.Linear(LAYER_SIZE, 2*n_actions),
            nn.Tanh(),
            NormalParamExtractor(),
        )

        self.critic_head = nn.Sequential(
            self.shared_net,
            nn.Linear(LAYER_SIZE, 1)
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loc, scale = self.actor_head(x)  # location and scale
        return torch.hstack([joint_velocities_to_actions(x, loc[:, :12]), loc[:, 12:]])

    def saveWeights(self, directory, prefix = "") -> str:
        name = f"{prefix}-{time.time()}.pt"
        torch.save(self.state_dict(), f"{directory}/{name}")
        return name

    def loadWeights(self, path):
        self.load_state_dict(torch.load(path))