# Copyright (c) 2024, The SAI Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

import numpy as np
from collections.abc import Callable
from scipy.spatial.transform import Rotation
import glfw
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer


class Se3Keyboard:
    """A keyboard controller for sending SE(3) commands as delta velocity.

    This class is designed to provide a keyboard controller for humanoid robot.
    It uses GLFW keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of:

    * delta vel: a 3D vector of (x, y) in meter/sec and z in rad/sec.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Rotate along z-axis            Q                 E
        Reset commands                 L
        Reset environment              P
        ============================== ================= =================
    """

    def __init__(self, renderer: MujocoRenderer, pos_sensitivity = 0.1, rot_sensitivity = 0.5):
        """Initialize the keyboard layer.

        Args:
            env: The Mujoco environment
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.4.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.8.
        """
        self._delta_vel =  np.zeros(3) # (x, y, yaw)

        self.rot_sensitivity = rot_sensitivity
        self.pos_sensitivity = pos_sensitivity
        self._should_quit = False

        # dictionary for additional callbacks
        self._additional_callbacks = dict()

        # store the viewer
        self._viewer = renderer._get_viewer("human")

        if hasattr(self._viewer, "_key_callback"):
            self._original_key_callback = self._viewer._key_callback
        else:
            self._original_key_callback = None

        # register keyboard callbacks
        self._register_callbacks()

        # track pressed keys
        self._pressed_keys = set()

        # reset environment callback
        self._reset_env_callback = None

        # create key bindings
        self._create_key_bindings()

    def __del__(self):
        """Restore the original keyboard callback."""
        if hasattr(self, "_viewer") and hasattr(self, "_original_key_callback"):
            try:
                window = self._viewer.window
                if self._original_key_callback:
                    glfw.set_key_callback(window, self._original_key_callback)
            except (AttributeError, TypeError):
                pass

    def __str__(self) -> str:
        """Returns: A string containing the information of keyboard controller."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove T1 along x-axis: W/S\n"
        msg += "\tMove T1 along y-axis: A/D\n"
        msg += "\tRotate T1 along z-axis: Q/E\n"
        msg += "\tReset commands: L\n"
        msg += "\tQuit: ESC\n"      
        msg += "\tReset environment: P"
        return msg

    """
    Operations
    """

    def should_quit(self) -> bool:
        """Return True if ESC was pressed."""
        return self._should_quit

    def reset(self):
        """Reset all command buffers to default values."""
        # default flags
        self._delta_vel = np.zeros(3)  # (x, y, yaw)
        self._should_quit = False

    def advance(self) -> np.ndarray:
        """Provides the result from keyboard event state.

        Returns:
            A 3-element array containing:
            - Elements 0-1: delta position [x, y]
            - Elements 2: delta rotation [yaw]
         """
        # return the command and gripper state
        return self._delta_vel

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def set_reset_env_callback(self, callback: Callable):
        """Set the callback function to reset the environment.

        Args:
            callback: The function to call when the P key is pressed.
        """
        self._reset_env_callback = callback

    """
    Internal helpers.
    """

    def _register_callbacks(self):
        """Register GLFW keyboard callbacks."""
        # Get the GLFW window from the viewer
        window = self._viewer.window

        # Set our key callback
        glfw.set_key_callback(window, self._on_keyboard_event)

    def _on_keyboard_event(self, window, key, scancode, action, mods):
        """GLFW keyboard callback function."""
        # Convert GLFW key to character
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self._should_quit = True
            # Optionally also request the window to close:
            try:
                glfw.set_window_should_close(window, True)
            except Exception:
                pass
            return

        try:
            # Map arrow keys directly using a dictionary
            arrow_keys = {
                glfw.KEY_LEFT: "LEFT",
                glfw.KEY_RIGHT: "RIGHT",
                glfw.KEY_UP: "UP",
                glfw.KEY_DOWN: "DOWN",
            }

            if key in arrow_keys:
                key_char = arrow_keys[key]
            else:
                key_char = chr(key).upper()
        except ValueError:
            # Not a printable character
            return

        if key_char in self._INPUT_KEY_MAPPING.keys():
            # Handle key press
            if action == glfw.PRESS:
                self._pressed_keys.add(key_char)
                self._handle_key_press(key_char)

            # Handle key release
            elif action == glfw.RELEASE:
                self._pressed_keys.discard(key_char)
                self._handle_key_release(key_char)
        else:
            if self._original_key_callback:
                self._original_key_callback(window, key, scancode, action, mods)

    def _handle_key_press(self, key_char):
        """Handle key press events."""
        # Apply the command when pressed
        if key_char == "L":
            self.reset()
        elif key_char == "P" and self._reset_env_callback:
            self._reset_env_callback()
        elif key_char in ["W", "S", "A", "D", "Q", "E"]:
            self._delta_vel += self._INPUT_KEY_MAPPING[key_char]

        # Additional callbacks
        if key_char in self._additional_callbacks:
            self._additional_callbacks[key_char]()

    def _handle_key_release(self, key_char):
        """Handle key release events."""
        # Remove the command when un-pressed
        if key_char in ["W", "S", "A", "D", "Q", "E"]:
            self._delta_vel -= self._INPUT_KEY_MAPPING[key_char]

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # x-axis (forward)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left-right)
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (rotation)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
            # reset commands
            "L": self.reset,
            "P": self._reset_env_callback,
        }