import numpy as np
from typing import Literal
import glfw, logging, warnings

import sai_mujoco  # noqa: F401
import gymnasium as gym
import torch

from ..booster_control import create_input_vector

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, env_name: Literal["LowerT1KickToTarget-v0", "LowerT1GoaliePenaltyKick-v0", "LowerT1ObstaclePenaltyKick-v0", "LowerT1PenaltyKick-v0"], headless: bool = False, max_episodes: int = 1000):
        self.episode_count = 0
        self.max_episodes = max_episodes

        # Render modes are 'human', 'rgb_array', 'depth_array', 'rgbd_tuple' (for headless, use anything but 'human')
        if headless:
            self.env = gym.make(env_name, render_mode="rgb_array")
        else:
            self.env = gym.make(env_name, render_mode="human")

        self.is_closed = False

        self._viewer = self.env.unwrapped.mujoco_renderer._get_viewer("human")
        self._window = self._viewer.window

        glfw.set_key_callback(self._window, self._quit)

    def getAgentInput(self, observation: np.ndarray, info: dict) -> torch.Tensor:
        return torch.tensor(create_input_vector(info, observation[:24]).reshape(1, -1), dtype=torch.float32)

    def reset(self) -> tuple[np.ndarray, dict, torch.Tensor]:
        '''
        Observation is (51, ) numpy array. First 24 components are joint positions (12) and velocities (12).
        Info is dictionary with additional state information
        '''
        observation, info = self.env.reset()

        self.episode_count += 1
        if self.episode_count > self.max_episodes:
            self.is_closed = True
            self.close()

        return self.getAgentInput(observation, info)

    def getReward(self, obs: np.array, info: dict) -> float:  # Placeholder for custom reward extraction
        joint_velocity = np.sum(np.abs(obs[12:24]))
        logger.debug(f"Joint Velocity Loss: {joint_velocity}")

        return -joint_velocity  # Return the negative absolute sum of the velocities

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict, torch.Tensor]:
        observation, reward, terminated, truncated, info = self.env.step(action)

        reward = self.getReward(observation, info)
        if terminated and not info['success']:  # Terminated = dead or success, Truncated = episode done
            reward -= 1_000_000
        elif terminated and info['success']:
            logger.info("Success!")
        elif truncated:
            logger.info("Episode done!")

        return self.getAgentInput(observation, info), reward, terminated, truncated

    def _close(self):
        try:
            glfw.set_window_should_close(self._window, True)
        except:
            pass

    def close(self, raise_if_closed=False):
        try:
            self._close()
            self.env.close()
        except:
            pass

    def _quit(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.is_closed = True
            self._close()
