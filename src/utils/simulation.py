import numpy as np
from typing import Literal
import glfw

import sai_mujoco  # noqa: F401
import gymnasium as gym


class SimulationEnvironment:
    def __init__(self, env_name: Literal["LowerT1KickToTarget-v0", "LowerT1GoaliePenaltyKick-v0", "LowerT1ObstaclePenaltyKick-v0", "LowerT1PenaltyKick-v0"], headless: bool = False, max_episodes: int = 1000):
        self.episode_count = 0
        self.max_episodes = max_episodes

        # Render modes are 'human', 'rgb_array', 'depth_array', 'rgbd_tuple' (for headless, use anything but 'human')
        if headless:
            self.env = gym.make(env_name, render_mode="rgb_array")
        else:
            self.env = gym.make(env_name, render_mode="human")

        self.exited = False

        self._viewer = self.env.unwrapped.mujoco_renderer._get_viewer("human")
        self._window = self._viewer.window

        glfw.set_key_callback(self._window, self._quit)


    def reset(self) -> tuple[np.ndarray, dict]:
        '''
        Observation is (51, ) numpy array. First 24 components are joint positions (12) and velocities (12).
        Info is dictionary with additional state information
        '''
        observation, info = self.env.reset()

        self.episode_count += 1
        if self.episode_count > self.max_episodes:
            self.exited = True
            self.close()

        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def _close(self):
        try:
            glfw.set_window_should_close(self._window, True)
        except Exception:
            pass

    def close(self):
        self._close()
        self.env.close()

    def _quit(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.exited = True
            self._close()
