import numpy as np

from .environment_torch import EnvironmentTorch


class KickToTargetEnv(EnvironmentTorch):
    def __init__(self, headless: bool = True, max_episodes: int = 1000):
        super().__init__(env_name="LowerT1KickToTarget-v0", headless=headless, max_episodes=max_episodes)

    def getReward(self, info: dict) -> float:
        reward = -np.linalg.norm(
            info['ball_xpos_rel_robot'] - info['target_xpos_rel_robot']
        )
        reward += 100.0 * int(info['success'])

        return reward - 1 # Penalty for each step to encourage faster goals