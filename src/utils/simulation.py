import numpy as np

import sai_mujoco  # noqa: F401
import gymnasium as gym


class SimulationEnvironment:
    # TODO: determine how to run in headless mode
    # TODO: determine how to initialize (and reset) with random player and ball positions
    def __init__(self, env_name: str = "LowerT1KickToTarget-v0"):
        self.env = gym.make(env_name, render_mode="human")

    def reset(self) -> tuple[np.ndarray, dict]:
        '''
        Observation is (51, ) numpy array. First 24 components are joint positions (12) and velocities (12).
        Info is dictionary with additional state information
        '''
        observation, info = self.env.reset()
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def close(self):
        self.env.close()

