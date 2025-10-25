"""
Teleoperate T1 robot in a gymnasium environment using a keyboard.
"""

import time
import argparse
import sai_mujoco  # noqa: F401
import gymnasium as gym
from se3_keyboard import Se3Keyboard
from t1_utils import LowerT1JoyStick

def teleop(env_name: str = "LowerT1GoaliePenaltyKick-v0", pos_sensitivity:float = 0.1, rot_sensitivity:float = 1.5):

    env = gym.make(env_name, render_mode="human")
    lower_t1_robot = LowerT1JoyStick(env.unwrapped)

    # Initialize the T1 SE3 keyboard controller with the viewer
    keyboard_controller = Se3Keyboard(renderer=env.unwrapped.mujoco_renderer, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)

    # Set the reset environment callback
    keyboard_controller.set_reset_env_callback(env.reset)

    # Print keyboard control instructions
    print("\nKeyboard Controls:")
    print(keyboard_controller)

    # Register Kick
    keyboard_controller.add_callback('K', lower_t1_robot.trigger_kick)

    # Main teleoperation loop
    episode_count = 0
    while True:
        # Reset environment for new episode
        terminated = truncated = False
        observation, info = env.reset()
        episode_count += 1

        print(f"\nStarting episode {episode_count}")
        # Episode loop  
        while not (terminated or truncated):
            # Get keyboard input and apply it directly to the environment
            if keyboard_controller.should_quit():
                print("\n[INFO] ESC pressed — exiting teleop.")
                env.close()
                return
            
            command = keyboard_controller.advance()
            ctrl, _ = lower_t1_robot.get_actions(command, observation, info)
            observation, reward, terminated, truncated, info = env.step(ctrl)

            if terminated or truncated:
                break
                
        # Print episode result
        if info.get("success", False):
            print(f"Episode {episode_count} completed successfully!")
        else:
            print(f"Episode {episode_count} completed without success")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Teleoperate T1 robot in a gymnasium environment.")
    parser.add_argument("--env", type=str, default="LowerT1GoaliePenaltyKick-v0", help="The environment to teleoperate.")
    parser.add_argument("--pos_sensitivity", type=float, default=0.1, help="SE3 Keyboard position sensitivity.")
    parser.add_argument("--rot_sensitivity", type=float, default=0.5, help="SE3 Keyboard rotation sensitivity.")

    args = parser.parse_args()

    teleop(args.env, args.pos_sensitivity, args.rot_sensitivity)