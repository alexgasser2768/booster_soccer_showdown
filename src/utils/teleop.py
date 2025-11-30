import logging, warnings
from typing import Dict

from ..booster_control.se3_keyboard import Se3Keyboard
from ..booster_control.t1_utils import LowerT1JoyStick

from ..environments import Environment

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def teleop(simulation: Environment, pos_sensitivity: float, rot_sensitivity: float) -> Dict[str, list]:
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

    try:
        collect_data = True
        while collect_data:
            # Reset environment for new episode
            terminated, truncated = False, False
            observation, info = simulation.env.reset()
            keyboard_controller.reset()

            while not (terminated or truncated):
                # Simulate holding the UP key to move forward
                if keyboard_controller._delta_vel.sum() < 1:
                    keyboard_controller._handle_key_press('UP')

                command = keyboard_controller.advance()
                ctrl, _ = lower_t1_robot.get_actions(command, observation, info)

                dataset["observations"].append(observation)
                dataset["infos"].append(info)
                dataset["actions"].append(ctrl)

                if simulation.is_closed or keyboard_controller.should_quit():
                    simulation.close()
                    collect_data = False
                    break

                observation, _, terminated, truncated, info = simulation.env.step(ctrl)

                if terminated or truncated:
                    break

    except (Exception, KeyboardInterrupt) as e:
        simulation.close()
        logger.error(f"An error occurred during teleoperation: {e}")

    return dataset
