import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor

import time
from typing import Tuple

from ..booster_control import DEVICE, joint_velocities_to_actions

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
        )

        # The following network is to learn an embedding that can be
        # used to both infer joint state (FK) and optimal action (IK). After all,
        # the next joint position and velocity depends on the action taken
        self.actor_shared = nn.Sequential(
            self.shared_net,
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
        )

        self.auxiliary_head = nn.Sequential(  # For predicting the next state (FK)
            self.actor_shared,
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
            nn.Linear(LAYER_SIZE, n_states),
        )

        self.actor_head = nn.Sequential(  # For deciding the action outputs (IK)
            self.actor_shared,
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
            nn.Linear(LAYER_SIZE, 2*n_actions),
            nn.Tanh(),
            NormalParamExtractor(),
        )

        self.critic_head = nn.Sequential(
            self.shared_net,
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
            nn.Linear(LAYER_SIZE, 1),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        loc, scale = self.actor_head(x)  # location and scale
        return joint_velocities_to_actions(x, loc), self.auxiliary_head(x)

    def saveWeights(self, directory, prefix = "") -> str:
        name = f"{prefix}-{time.time()}.pt"
        torch.save(self.state_dict(), f"{directory}/{name}")
        return name

    def loadWeights(self, path):
        self.load_state_dict(torch.load(path, map_location=DEVICE))