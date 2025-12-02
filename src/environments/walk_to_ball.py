import numpy as np

from .environment_torch import EnvironmentTorch


class WalkToBallEnv(EnvironmentTorch):
    def __init__(self, headless: bool = True, max_episodes: int = 1000):
        super().__init__(env_name="LowerT1PenaltyKick-v0", headless=headless, max_episodes=max_episodes)

    def getReward(self, obs: np.array, info: dict) -> float:
        reward = super().getReward(obs, info)
        return reward
        reward -= np.linalg.norm(info['ball_xpos_rel_robot'])
        return 1000 + reward # Penalty for each step to encourage faster goals