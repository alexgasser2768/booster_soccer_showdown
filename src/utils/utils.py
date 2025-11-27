import numpy as np

SCALE_FACTOR = 25.0
GOALKEEPER_WIDTH = 0.5


def create_input_vector(info: dict, joint_obs: np.ndarray) -> np.ndarray:
    """
    Creates the input vector for the model from the info dictionary and joint observations.

    Args:
        info: A dictionary containing environment and robot state information.
        joint_obs: A numpy array containing the robot's joint positions and velocities.

    Returns:
        A numpy array representing the scaled input vector for the model.
    """
    scaled_joint_obs = np.tanh(joint_obs)

    robot_orientation = np.array(info['robot_quat'])
    robot_angular_velocity = np.tanh(np.array(info['robot_gyro']))
    robot_linear_velocity = np.tanh(np.array(info['robot_velocimeter']))
    robot_linear_acceleration = np.tanh(np.array(info['robot_accelerometer']))

    ball_pos_rel_robot = np.array(info['ball_xpos_rel_robot']) / SCALE_FACTOR
    ball_velp_rel_robot = np.tanh(np.array(info['ball_velp_rel_robot']))
    ball_velr_rel_robot = np.tanh(np.array(info['ball_velr_rel_robot']))

    goal_pos_rel_robot = np.array(info[ 'goal_team_0_rel_robot']) / SCALE_FACTOR
    goalkeeper_pos_rel_robot = np.array(info['goalkeeper_team_0_xpos_rel_robot']) / SCALE_FACTOR

    target_pos_rel_robot = np.array(info['target_xpos_rel_robot']) / SCALE_FACTOR
    target_velp_rel_robot = np.tanh(np.array(info['target_velp_rel_robot']))

    if np.all(goalkeeper_pos_rel_robot == 0) and np.all(target_pos_rel_robot == 0):  # Case where no goalkeeper and no target info is given
        target_pos_rel_robot = goal_pos_rel_robot
        target_velp_rel_robot = np.zeros((3, ))
    elif np.all(target_pos_rel_robot == 0):  # If target info is not given, determine the largest open area on goal that is not guarded by the goalkeeper
        goal_width = info['goal_width']

        # area goalkeeper is blocking (considering some margin)
        block_y_min = goalkeeper_pos_rel_robot[1] - GOALKEEPER_WIDTH
        block_y_max = goalkeeper_pos_rel_robot[1] + GOALKEEPER_WIDTH

        # determine largest open area on goal
        left_open_width = max(0.0, (goal_pos_rel_robot[1] - goal_width / 2) - block_y_max)
        right_open_width = max(0.0, block_y_min - (goal_pos_rel_robot[1] + goal_width / 2))

        # position of the midpoint of the largest open area
        if left_open_width >= right_open_width:
            target_pos_rel_robot = np.array([
                goal_pos_rel_robot[0],
                (goal_pos_rel_robot[1] - goal_width / 2 + block_y_max) / 2,
                goal_pos_rel_robot[2]
            ]) / SCALE_FACTOR
        else:
            target_pos_rel_robot = np.array([
                goal_pos_rel_robot[0],
                (block_y_min + goal_pos_rel_robot[1] + goal_width / 2) / 2,
                goal_pos_rel_robot[2]
            ]) / SCALE_FACTOR

        target_velp_rel_robot = np.tanh(np.array(info['goalkeeper_team_1_velp_rel_robot']))

    return np.concatenate([
        scaled_joint_obs,
        robot_orientation,
        robot_angular_velocity,
        robot_linear_velocity,
        robot_linear_acceleration,
        ball_pos_rel_robot,
        ball_velp_rel_robot,
        ball_velr_rel_robot,
        target_pos_rel_robot,
        target_velp_rel_robot,
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