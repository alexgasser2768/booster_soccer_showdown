import numpy as np


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
    return np.concatenate([
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


def print_stats(data_points: np.ndarray, variable_name: str = ""):
    print(f"Statistics for raw {variable_name}:")
    print(f"  Min: {np.min(data_points):.4f}, Median: {np.median(data_points):.4f}, Mean: {np.mean(data_points):.4f}, Max: {np.max(data_points):.4f}, Std Dev: {np.std(data_points):.4f}")
    print(f"  Quantiles - 1%: {np.quantile(data_points, 0.01):.4f}, 25%: {np.quantile(data_points, 0.25):.4f}, 50%: {np.quantile(data_points, 0.5):.4f}, 75%: {np.quantile(data_points, 0.75):.4f}, 99%: {np.quantile(data_points, 0.99):.4f}")
    print("-" * 20)

    data_points = np.tanh(data_points)

    print(f"Statistics for normalized {variable_name} (tanh):")
    print(f"  Min: {np.min(data_points):.4f}, Median: {np.median(data_points):.4f}, Mean: {np.mean(data_points):.4f}, Max: {np.max(data_points):.4f}, Std Dev: {np.std(data_points):.4f}")
    print(f"  Quantiles - 1%: {np.quantile(data_points, 0.01):.4f}, 25%: {np.quantile(data_points, 0.25):.4f}, 50%: {np.quantile(data_points, 0.5):.4f}, 75%: {np.quantile(data_points, 0.75):.4f}, 99%: {np.quantile(data_points, 0.99):.4f}")
    print()