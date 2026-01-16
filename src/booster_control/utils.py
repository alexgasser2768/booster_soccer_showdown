import torch
import numpy as np

DEVICE = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

SIGMOID = lambda x: 1 / (1 + np.exp(-x))

POS_SCALE = 1.0  # 25.0
VEL_SCALE = 1.0  # 100.0
GOALKEEPER_WIDTH = 0.5

# Constants (adapted from booster_control/t1_utils.py)
DEFAULT_DOF_POS = torch.tensor([-0.2, 0.0, 0.0, 0.4, -0.25, 0.0, -0.2, 0.0, 0.0, 0.4, -0.25, 0.0], dtype=torch.float32, device=DEVICE)
DOF_STIFFNESS = torch.tensor([200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 200.0, 200.0, 200.0, 200.0, 50.0, 50.0], dtype=torch.float32, device=DEVICE)
DOF_DAMPING = torch.tensor([5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0], dtype=torch.float32, device=DEVICE)
CTRL_MIN = torch.tensor([-45, -45, -30, -65, -24, -15, -45, -45, -30, -65, -24, -15], dtype=torch.float32, device=DEVICE)
CTRL_MAX = torch.tensor([45, 45, 30, 65, 24, 15, 45, 45, 30, 65, 24, 15], dtype=torch.float32, device=DEVICE)


def joint_velocities_to_actions(obs: torch.tensor, actions: torch.tensor) -> torch.tensor:
    return 100 * actions  # Let model learn torques directly
    # PD control + Clamp (adapted from booster_control/t1_utils.py)
    device = obs.device
    actions = actions.to(device)
    default_pos = DEFAULT_DOF_POS.to(device)
    dof_stiffness = DOF_STIFFNESS.to(device)
    dof_damping = DOF_DAMPING.to(device)
    ctrl_min = CTRL_MIN.to(device)
    ctrl_max = CTRL_MAX.to(device)
    if len(obs.shape) > 1:
        qpos = obs[:, :12]
        qvel = obs[:, 12:24]
        targets = default_pos.expand(obs.shape[0], -1) + actions
    else:
        qpos = obs[:12]
        qvel = obs[12:24]
        targets = default_pos + actions

    ctrl = dof_stiffness * (targets - qpos) - dof_damping * qvel
    return torch.minimum(
        torch.maximum(ctrl, ctrl_min.expand_as(ctrl)),
        ctrl_max.expand_as(ctrl)
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
    scaled_joint_obs = np.concatenate((
        joint_obs[:12] / POS_SCALE,
        joint_obs[12:24] / VEL_SCALE
    ))

    robot_orientation = np.array(info['robot_quat'])
    robot_angular_velocity = np.array(info['robot_gyro']) / VEL_SCALE
    robot_linear_velocity = np.array(info['robot_velocimeter']) / VEL_SCALE
    robot_linear_acceleration = np.array(info['robot_accelerometer']) / VEL_SCALE

    ball_pos_rel_robot = np.array(info['ball_xpos_rel_robot']) / POS_SCALE
    ball_velp_rel_robot = np.array(info['ball_velp_rel_robot']) / VEL_SCALE
    ball_velr_rel_robot = np.array(info['ball_velr_rel_robot']) / VEL_SCALE

    goal_pos_rel_robot = np.array(info[ 'goal_team_0_rel_robot']) / POS_SCALE
    goalkeeper_pos_rel_robot = np.array(info['goalkeeper_team_0_xpos_rel_robot']) / POS_SCALE

    target_pos_rel_robot = np.array(info['target_xpos_rel_robot']) / POS_SCALE
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
            ]) / POS_SCALE
        else:
            target_pos_rel_robot = np.array([
                goal_pos_rel_robot[0],
                (block_y_min + goal_pos_rel_robot[1] + goal_width / 2) / 2,
                goal_pos_rel_robot[2]
            ]) / POS_SCALE

        target_velp_rel_robot = np.array(info['goalkeeper_team_0_velp_rel_robot']) / VEL_SCALE

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
