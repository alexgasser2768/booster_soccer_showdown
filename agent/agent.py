import torch
import torch.nn as nn

from typing import Tuple


LAYER_SIZE = 256


class Agent(nn.Module):
    def __init__(self, n_states, n_actions, std_value = 2.3):
        super(Agent, self).__init__()

        # Match the shared network architecture
        self.shared_net = nn.Sequential(
            nn.Tanh(),
            nn.Linear(n_states, LAYER_SIZE),
            nn.LeakyReLU(),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(LAYER_SIZE, n_actions),
            nn.Tanh(),
        )

        self.critic_head = nn.Linear(LAYER_SIZE, 1)

        # Log Standard Deviation (for continuous actions)
        # This is a parameter, not a layer output, and is learned directly.
        # Initialize to a small value (e.g., log(0.1) ~ -2.3)
        self.log_std = nn.Parameter(torch.zeros(1, n_actions) - std_value)

        # Constants
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
        features = self.shared_net(x)
        mu = self.actor_head(features)

        # Mean of Gaussian distribution, Standard deviation, State Value
        return mu, \
            torch.exp(self.log_std.expand_as(mu)), \
            self.critic_head(features)
    
    def getActions(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        N = obs.shape[0]  # Batch size

        actions = self(obs)[0]  # (N,12)
        targets = self.default_dof_pos.expand(N, -1) + actions

        # PD control + Clamp
        qpos = obs[:, :12]
        qvel = obs[:, 12:24]

        ctrl = self.dof_stiffness * (targets - qpos) - self.dof_damping * qvel
        return torch.minimum(
            torch.maximum(ctrl, self.ctrl_min.expand_as(ctrl)),
            self.ctrl_max.expand_as(ctrl)
        ), actions

    def saveWeights(self, path):
        torch.save(self.state_dict(), path)

    def loadWeights(self, path):
        self.load_state_dict(torch.load(path))