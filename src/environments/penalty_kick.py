import numpy as np

from .environment_torch import EnvironmentTorch


class PenaltyKickEnv(EnvironmentTorch):
    def __init__(self, headless: bool = True, max_episodes: int = 1000):
        super().__init__(env_name="LowerT1PenaltyKick-v0", headless=headless, max_episodes=max_episodes)

    def getReward(self, info: dict) -> float:
        reward = -np.linalg.norm(info['goal_team_0_rel_ball'])
        reward += 100.0 * int(info['success'])

        return reward - 1 # Penalty for each step to encourage faster goals