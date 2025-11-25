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
from booster_control.t1_utils import Preprocessor
import torch

from agent import Agent


N_STATES = 52
N_ACTIONS = 12


def create_input_vector(info: dict, joint_obs: np.ndarray) -> np.ndarray:
    """
    Creates the input vector for the model from the info dictionary and joint observations.

    Args:
        info: A dictionary containing environment and robot state information.
        joint_obs: A numpy array containing the robot's joint positions and velocities.

    Returns:
        A numpy array representing the scaled input vector for the model.
    """
    # --- 1. Joint Positions & Velocities (24 components) ---
    # As per the outline, this comes from the observation vector and is scaled with tanh.
    scaled_joint_obs = np.tanh(joint_obs)

    # --- 2. Robot State from 'info' dictionary ---

    # Robot orientation (4 components) - unit quaternion, no scaling needed
    robot_orientation = np.array(info['robot_quat'])

    # Robot angular velocity (3 components) - scaled by tanh
    robot_angular_velocity = np.tanh(np.array(info['robot_gyro']))

    # Robot linear velocity (3 components) - scaled by tanh
    robot_linear_velocity = np.tanh(np.array(info['robot_velocimeter']))

    # Robot linear acceleration (3 components) - scaled by tanh
    robot_linear_acceleration = np.tanh(np.array(info['robot_accelerometer']))

    # Ball position relative to robot (3 components) - scaled down by 25
    ball_pos_rel_robot = np.array(info['ball_xpos_rel_robot']) / 25.0

    # Ball linear velocity relative to robot (3 components) - scaled by tanh
    ball_velp_rel_robot = np.tanh(np.array(info['ball_velp_rel_robot']))

    # Ball angular velocity relative to robot (3 components) - scaled by tanh
    ball_velr_rel_robot = np.tanh(np.array(info['ball_velr_rel_robot']))

    # Determine which goalkeeper is the opponent
    # player_team is [1, 0] for AWAY team, so opponent is HOME team (team 0)
    opponent_goalkeeper_pos_key = 'goalkeeper_team_0_xpos_rel_robot'
    opponent_goalkeeper_vel_key = 'goalkeeper_team_0_velp_rel_robot'
    if info['player_team'][0] == 0: # If player is HOME team
        opponent_goalkeeper_pos_key = 'goalkeeper_team_1_xpos_rel_robot'
        opponent_goalkeeper_vel_key = 'goalkeeper_team_1_velp_rel_robot'

    # Goalkeeper position relative to robot (3 components) - scaled down by 25
    goalkeeper_pos_rel_robot = np.array(info[opponent_goalkeeper_pos_key]) / 25.0

    # Goalkeeper linear velocity relative to robot (3 components) - scaled by tanh
    goalkeeper_velp_rel_robot = np.tanh(np.array(info[opponent_goalkeeper_vel_key]))

    # --- 3. Concatenate all parts into the final input vector ---
    model_input = np.concatenate([
        scaled_joint_obs,
        robot_orientation,
        robot_angular_velocity,
        robot_linear_velocity,
        robot_linear_acceleration,
        ball_pos_rel_robot,
        ball_velp_rel_robot,
        ball_velr_rel_robot,
        goalkeeper_pos_rel_robot,
        goalkeeper_velp_rel_robot,
    ])

    return model_input


def teleop(env_name: str = "LowerT1KickToTarget-v0", pos_sensitivity:float = 0.1, rot_sensitivity:float = 1.5, dataset_directory = "./data/data.npz"):

    env = gym.make(env_name, render_mode="human")
    lower_t1_robot = LowerT1JoyStick(env.unwrapped)
    preprocessor = Preprocessor()

    # Initialize the T1 SE3 keyboard controller with the viewer
    keyboard_controller = Se3Keyboard(renderer=env.unwrapped.mujoco_renderer, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)

    # Set the reset environment callback
    keyboard_controller.set_reset_env_callback(env.reset)

    agent = Agent(n_states=N_STATES, n_actions=N_ACTIONS)
    agent.loadWeights("data/il_actor_seed_weights.pt")

    # Print keyboard control instructions
    print("\nKeyboard Controls:")
    print(keyboard_controller)

    dataset = {
        "observations" : [],
        "actions" : []
    }

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
            # command = keyboard_controller.advance()
            # ctrl, actions = lower_t1_robot.get_actions(command, observation, info)

            # model_input = torch.tensor(create_input_vector(info, observation[:24]).reshape(1, -1), dtype=torch.float32)
            model_input = torch.tensor(
                np.tanh(np.hstack([observation[:24], np.zeros((N_STATES - 24, ))])).reshape(1, -1),
                dtype=torch.float32
            )
            ctrl = agent(model_input)[0]
            ctrl = ctrl.detach().numpy().reshape((12, ))

            episode["observations"].append(preprocessed_observation)
            episode["actions"].append(ctrl)

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