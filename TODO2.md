# inputs

- FOR EACH POSITION WE NEED A VELOCITY (MAYBE USE ACCELERATION) reduces number of inputs

- joint positions between [-1,1] (28 joints -> 56 inputs)
- world robot position [x,y,yaw] (6 inputs)
- world ball position [x,y] (4 inputs)
- target for ball [x_min,y_min,z_min,x_max,y_max,z_max] (12 inputs)

- 78 total inputs

# outputs

- joint velocities (28 outputs)


## default observation from info

```json
{
    "length": 10.97,
    "width": 6.87,
    "goal_width": 1.6,
    "goal_height": 1.9,
    "goal_depth": 1.6,
    "goal_team_0_rel_robot": [
        -4.200000000000001,
        0.0,
        -0.7
    ],
    "goal_team_1_rel_robot": [
        17.740000000000002,
        0.0,
        -0.7
    ],
    "goal_team_0_rel_ball": [
        -2.200000000000001,
        0.0,
        0.0
    ],
    "goal_team_1_rel_ball": [
        19.740000000000002,
        0.0,
        0.0
    ],
    "ball_xpos_rel_robot": [
        -2.0,
        0.0,
        -0.7
    ],
    "ball_velp_rel_robot": [
        0.0,
        0.0,
        0.0
    ],
    "ball_velr_rel_robot": [
        0.0,
        0.0,
        0.0
    ],
    "player_team": [
        1,
        0
    ],
    "robot_accelerometer": [
        -1.3136856325115598e-16,
        4.3332978045500977e-17,
        -2.0261972540059053e-15
    ],
    "robot_gyro": [
        0.0,
        0.0,
        0.0
    ],
    "robot_velocimeter": [
        0.0,
        0.0,
        0.0
    ],
    "robot_quat": [
        0.0,
        0.0,
        0.9999997019767761,
        0.0007963267271406949
    ],
    "goalkeeper_team_0_xpos_rel_robot": [
        -4.200000000000001,
        0.0,
        -0.49999999999999994
    ],
    "goalkeeper_team_0_velp_rel_robot": [
        0.0,
        0.0,
        0.0
    ],
    "goalkeeper_team_1_xpos_rel_robot": [
        17.740000000000002,
        0.0,
        -0.49999999999999994
    ],
    "goalkeeper_team_1_velp_rel_robot": [
        0.0,
        0.0,
        0.0
    ],
    "target_xpos_rel_robot": [
        0.0,
        0.0,
        0.0
    ],
    "target_velp_rel_robot": [
        0.0,
        0.0,
        0.0
    ],
    "defender_xpos": [
        -6.77,
        0.0,
        0.7,
        -6.77,
        0.0,
        0.7,
        -6.77,
        0.0,
        0.7
    ],
    "success": false
}
```

# using a PPO RL model structure
