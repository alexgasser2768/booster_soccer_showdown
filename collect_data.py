"""
Teleoperate T1 robot in a gymnasium environment using a keyboard.
"""
import os, json
import sys

# Make repo root importable without absolute paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
sys.path.append(repo_root)

import time
import argparse
import sai_mujoco  # noqa: F401
import gymnasium as gym
import numpy as np
from booster_control.se3_keyboard import Se3Keyboard
from booster_control.t1_utils import LowerT1JoyStick
from imitation_learning.scripts.preprocessor import Preprocessor

def teleop(env_name: str = "LowerT1KickToTarget-v0", pos_sensitivity:float = 0.1, rot_sensitivity:float = 1.5, dataset_directory = "./data/data.npz"):

    env = gym.make(env_name, render_mode="human")
    lower_t1_robot = LowerT1JoyStick(env.unwrapped)
    preprocessor = Preprocessor()

    # Initialize the T1 SE3 keyboard controller with the viewer
    keyboard_controller = Se3Keyboard(renderer=env.unwrapped.mujoco_renderer, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)

    # Set the reset environment callback
    keyboard_controller.set_reset_env_callback(env.reset)

    # Print keyboard control instructions
    print("\nKeyboard Controls:")
    print(keyboard_controller)

    dataset = {
        "observations" : [],
        "actions" : []
    }

    ###### Saves info gathered
    # observation, info = env.reset()
    # with open('info.json', 'w') as f:
    #     json.dump(
    #         {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in info.items()},
    #         f
    #     )

    # Main teleoperation loop
    episode_count = 0
    while True:
        # Reset environment for new episode
        terminated = truncated = False
        observation, info = env.reset()

        episode_count += 1

        episode = {
            "observations" : [],
            "actions" : []
        }

        print(f"\nStarting episode {episode_count}")
        # Episode loop
        while not (terminated or truncated):

            preprocessed_observation = preprocessor.modify_state(observation.copy(), info.copy())
            # Get keyboard input and apply it directly to the environment
            command = keyboard_controller.advance()
            ctrl, actions = lower_t1_robot.get_actions(command, observation, info)

            episode["observations"].append(preprocessed_observation)
            episode["actions"].append(actions)

            if keyboard_controller.should_quit():
                print("\n[INFO] ESC pressed — exiting teleop.")
                dataset["observations"].extend(episode["observations"])
                dataset["actions"].extend(episode["actions"])
                directory = os.path.dirname(dataset_directory)
                print(f'length of observations: {len(dataset["observations"])}')
                if os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                np.savez(dataset_directory, observations=dataset["observations"], actions=dataset["actions"])
                env.close()
                return

            observation, reward, terminated, truncated, info = env.step(ctrl)

            if terminated or truncated:
                break

        dataset["observations"].extend(episode["observations"])
        dataset["actions"].extend(episode["actions"])
        # Print episode result
        if info.get("success", True):
            print(f"Episode {episode_count} completed successfully!")
        else:
            print(f"Episode {episode_count} completed without success")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Teleoperate T1 robot in a gymnasium environment.")
    parser.add_argument("--env", type=str, default="LowerT1GoaliePenaltyKick-v0", help="The environment to teleoperate.")
    parser.add_argument("--pos_sensitivity", type=float, default=1.5, help="SE3 Keyboard position sensitivity.")
    parser.add_argument("--rot_sensitivity", type=float, default=7.5, help="SE3 Keyboard rotation sensitivity.")
    parser.add_argument("--data_set_directory", type=str, default="./data/test.npz", help="SE3 Keyboard rotation sensitivity.")

    args = parser.parse_args()

    teleop(args.env, args.pos_sensitivity, args.rot_sensitivity, args.data_set_directory)