import numpy as np
from typing import Literal
import glfw

import sai_mujoco  # noqa: F401
import gymnasium as gym


class SimulationEnvironment:
    # TODO: determine how to run in headless mode
    # TODO: determine how to initialize (and reset) with random player and ball positions
    def __init__(self, env_name: Literal["LowerT1KickToTarget-v0", "LowerT1GoaliePenaltyKick-v0", "LowerT1ObstaclePenaltyKick-v0", "LowerT1PenaltyKick-v0", "LowerT1GoalKeeper-v0"] = "LowerT1PenaltyKick-v0"):
        self.env = gym.make(env_name, render_mode="human")  # render_mode = ['human', 'rgb_array', 'depth_array', 'rgbd_tuple'] (for headless, use 'rgb_array' or 'depth_array')

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
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def close(self):
        self.env.close()

    def _quit(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.exited = True

            try:
                glfw.set_window_should_close(window, True)
            except Exception:
                pass
