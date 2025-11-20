# Inputs

All inputs will be scaled to between [-1, 1]:
- The observation vector (joints positions and velocities) will be scaled by tanh
- Positions will be scaled by down by 25
- Velocities will be scaled by tanh
- Orientation will be a unit quaternion

Robot State (73 components):
- Joint positions & velocities (45 components)
- Robot orientation (4 components)
- Robot angualr velocity (3 components)
- Robot linear velocity (3 components)
- Robot linear acceleration (3 components)
- Ball position relative to robot (3 components)
- Ball linear velocity relative to robot (3 components)
- Ball angular velocity relative to robot (3 components)
- Goalkeeper position relative to robot (3 components)
- Goalkeeper linear velocity relative to robot (3 components)

# Outputs

- Joint Velocities between [-1, 1] (12 Outputs)


## default observation from info

```json
{
    "length": 10.97,
    "width": 6.87,
    "goal_width": 1.6,
    "goal_height": 1.9,
    "goal_depth": 1.6,
    "goal_team_0_rel_robot": [  // Position of HOME goal relative to robot
        -4.200000000000001,
        0.0,
        -0.7
    ],
    "goal_team_1_rel_robot": [  // Position of AWAY goal relative to robot
        17.740000000000002,
        0.0,
        -0.7
    ],
    "goal_team_0_rel_ball": [  // Position of HOME goal relative to ball
        -2.200000000000001,
        0.0,
        0.0
    ],
    "goal_team_1_rel_ball": [  // Position of AWAY goal relative to ball
        19.740000000000002,
        0.0,
        0.0
    ],
    "ball_xpos_rel_robot": [  // Position of ball relative to robot
        -2.0,
        0.0,
        -0.7
    ],
    "ball_velp_rel_robot": [  // Linear velocity of ball relative to robot
        0.0,
        0.0,
        0.0
    ],
    "ball_velr_rel_robot": [  // Angular velocity of ball relative to robot
        0.0,
        0.0,
        0.0
    ],
    "player_team": [
        1,
        0
    ],
    "robot_accelerometer": [  // Acceleration of robot
        -1.3136856325115598e-16,
        4.3332978045500977e-17,
        -2.0261972540059053e-15
    ],
    "robot_gyro": [  // Angular velocity of robot
        0.0,
        0.0,
        0.0
    ],
    "robot_velocimeter": [  // Linear velocity of robot
        0.0,
        0.0,
        0.0
    ],
    "robot_quat": [  // Orientation of robot
        0.0,
        0.0,
        0.9999997019767761,
        0.0007963267271406949
    ],
    "goalkeeper_team_0_xpos_rel_robot": [  // Position of HOME goalkeeper relative to robot
        -4.200000000000001,
        0.0,
        -0.49999999999999994
    ],
    "goalkeeper_team_0_velp_rel_robot": [  // Linear velocity of HOME goalkeeper relative to robot
        0.0,
        0.0,
        0.0
    ],
    "goalkeeper_team_1_xpos_rel_robot": [  // Position of AWAY goalkeeper relative to robot
        17.740000000000002,
        0.0,
        -0.49999999999999994
    ],
    "goalkeeper_team_1_velp_rel_robot": [  // Linear velocity of AWAY goalkeeper relative to robot
        0.0,
        0.0,
        0.0
    ],
    "target_xpos_rel_robot": [  // Position of target relative to robot
        0.0,
        0.0,
        0.0
    ],
    "target_velp_rel_robot": [  // Linear velocity of target relative to robot
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
