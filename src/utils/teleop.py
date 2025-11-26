from typing import Dict

from ..booster_control.se3_keyboard import Se3Keyboard
from ..booster_control.t1_utils import LowerT1JoyStick

from .simulation import SimulationEnvironment


def teleop(simulation: SimulationEnvironment, pos_sensitivity: float, rot_sensitivity: float) -> Dict[str, list]:
    # Initialize the T1 SE3 keyboard controller with the viewer
    lower_t1_robot = LowerT1JoyStick(simulation.env.unwrapped)
    keyboard_controller = Se3Keyboard(renderer=simulation.env.unwrapped.mujoco_renderer, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)

    # Set the reset environment callback
    keyboard_controller.set_reset_env_callback(simulation.reset)

    dataset = {
        "observations" : [],
        "infos": [],
        "actions" : []
    }

    while True:
        # Reset environment for new episode
        terminated, truncated = False, False
        observation, info = simulation.reset()

        while not (terminated or truncated):
            # Get keyboard input and apply it directly to the environment
            command = keyboard_controller.advance()
            ctrl, _ = lower_t1_robot.get_actions(command, observation, info)

            dataset["observations"].append(observation)
            dataset["infos"].append(info)
            dataset["actions"].append(ctrl)

            if keyboard_controller.should_quit():
                simulation.close()
                return dataset

            observation, reward, terminated, truncated, info = simulation.step(ctrl)

            if terminated or truncated:
                break
