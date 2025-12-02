import torch
import numpy as np

DEVICE = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

SIGMOID = lambda x: 1 / (1 + np.exp(-x))

SCALE_FACTOR = 25.0
GOALKEEPER_WIDTH = 0.5

# Constants (adapted from booster_control/t1_utils.py)
DEFAULT_DOF_POS = torch.tensor([-0.2, 0.0, 0.0, 0.4, -0.25, 0.0, -0.2, 0.0, 0.0, 0.4, -0.25, 0.0], dtype=torch.float32, device=DEVICE)
DOF_STIFFNESS = torch.tensor([200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 200.0, 200.0, 200.0, 200.0, 50.0, 50.0], dtype=torch.float32, device=DEVICE)
DOF_DAMPING = torch.tensor([5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0], dtype=torch.float32, device=DEVICE)
CTRL_MIN = torch.tensor([-45, -45, -30, -65, -24, -15, -45, -45, -30, -65, -24, -15], dtype=torch.float32, device=DEVICE)
CTRL_MAX = torch.tensor([45, 45, 30, 65, 24, 15, 45, 45, 30, 65, 24, 15], dtype=torch.float32, device=DEVICE)


def joint_velocities_to_actions(obs: torch.tensor, actions: torch.tensor) -> torch.tensor:
    # PD control + Clamp (adapted from booster_control/t1_utils.py)
    if len(obs.shape) > 1:
        qpos = obs[:, :12]
        qvel = obs[:, 12:24]
        targets = DEFAULT_DOF_POS.expand(obs.shape[0], -1) + actions
    else:
        qpos = obs[:12]
        qvel = obs[12:24]
        targets = DEFAULT_DOF_POS + actions

    ctrl = DOF_STIFFNESS * (targets - qpos) - DOF_DAMPING * qvel
    return torch.minimum(
        torch.maximum(ctrl, CTRL_MIN.expand_as(ctrl)),
        CTRL_MAX.expand_as(ctrl)
    )


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
    target_velp_rel_robot = np.zeros((3, ))  # Target is usually static

    if np.all(goalkeeper_pos_rel_robot == 0) and np.all(target_pos_rel_robot == 0):  # Case where no goalkeeper and no target info is given
        target_pos_rel_robot = goal_pos_rel_robot
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

        target_velp_rel_robot = np.tanh(np.array(info['goalkeeper_team_0_velp_rel_robot']))

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
