# Humanoid Soccer Learning Task

This project was inspired by the SAI humanoid robot competition. The goal is to teach the Booster T1 robot how
to score goals in a simulated environment. 

# Run on GPU server

```
xvfb-run -a python main.py
```

# Reward Structure

As a rule of thumb, each episode is 10 seconds unless mentioned otherwise.

## Learn running

- Behavior cloning to run
- Make the model to run in a straight line for as long as possible and as fast as possible:
    - Reward always has a +1 to incentivize continuous running
    - If he falls, terminate episode
- Make the model run in a straight line and then stop without falling
    - If he falls, terminate episode
    - Reward will be based on the difference in distance from previous episode and now:
        - If robot moved towards the target (difference is negative), the reward will be +1
        - If robot moved away from the target (difference is positive), the reward will be -1
    - Reward will be +1 if he stays within 1 meter of the target
- Make the model run in random directions and then stop without falling
    - Same reward structure as previous task
    - Target location will change to a random position every 5-10 seconds (time chosen is random)
    - Each episode will have 2-10 runs (2-10 runs * 5-10 seconds per run = 10-100 seconds)

## Learn passing

Make sure in the beginning, the target is not too far away (1-2 meters). As the model becomes better and better,
make the target further and further (Curriculum Learning).

- Make the model to pass in a straight line:
    - Reward always has a -1 for time
    - If he falls, terminate episode
    - Reward will have -x, where x is the final distance from ball to target
    - Reward will have +10/x if the ball is within 1 meter of the target
    - If ball is <= 0.1 meters from target, reward is +100 and episode terminates
- Make the model run to ball and pass:
    - Same reward structure as previous task
    - Start robot away from ball in random locations
- Make the model run in random directions and then pass:
    - Same reward structure as previous task but episode doesn't terminate when ball is close to target
    - Target location will change to a random position every 5-10 seconds (time chosen is random). Ball location doesn't change
    - Each episode will have 2-10 passes (2-10 passes * 5-10 seconds per pass = 10-100 seconds)

## Learn shooting

Shooting is essentially complicated passing in 3D with more force.

- Make the model run in random directions and then shoot with no obstacle:
    - Same reward structure as passing but if the robot makes a goal
    - The reward is +100 and episode terminates if the robot scores in the AWAY team
    - The reward is -100 and episode terminates if the robot scores in the HOME team

- Make the model run in random directions and then shoot with obstacles:
    - Same reward structure as previous task
    - If the ball hits an obstacle, no penalty (since time penalty is enough) and no termination

# RL Agent design

## Inputs

All inputs will be scaled to between [-1, 1]:
- The observation vector (joints positions and velocities) will be scaled by tanh
- Positions will be scaled by down by 25
- Velocities and acceleration will be scaled by tanh
- Orientation will be a unit quaternion

Robot State (52 components):
- Joint positions & velocities (24 components)
- Robot orientation (4 components)
- Robot angular velocity (3 components)
- Robot linear velocity (3 components)
- Robot linear acceleration (3 components)
- Ball position relative to robot (3 components)
- Ball linear velocity relative to robot (3 components)
- Ball angular velocity relative to robot (3 components)
- Goalkeeper position relative to robot (3 components)
- Goalkeeper linear velocity relative to robot (3 components)

## Outputs

- Joint Velocities between [-1, 1] (12 Outputs)

## Model Architecture

- Will use Proximal Policy Optimization (PPO)
- Deep FCN (6 Layers). Each layer is 256 neurons (Max. Neurons: 52*256 + 6*256 + 12)
- Leaky ReLU will be used (no information loss from gradient)
- Output layer is joint velocity from -1 to 1 (tanh)
    - Output will use tanh to normalize
    - The function `get_torque` in the `booster_control/t1_utils.py` file will be used to convert the normalized output into control signals

## Sample data from the different environments

### LowerT1KickToTarget-v0

```python
info = {
    'length': 10.97,
    'width': 6.87,
    'goal_width': 1.6,
    'goal_height': 1.9,
    'goal_depth': 1.6,
    'goal_team_0_rel_robot': array([-5.76995107,  4.03897027, -0.7       ]),
    'goal_team_1_rel_robot': array([16.17004893,  4.03897027, -0.7       ]),
    'goal_team_0_rel_ball': array([-6.54378036,  3.43792333,  0.        ]),
    'goal_team_1_rel_ball': array([15.39621964,  3.43792333,  0.        ]),
    'ball_xpos_rel_robot': array([ 0.7738293 ,  0.60104694, -0.7       ]),
    'ball_velp_rel_robot': array([0.07996708, 0.05742423, 0.02280047]),
    'ball_velr_rel_robot': array([ 0.48831477, -0.55803543,  0.13538543]),
    'player_team': array([1, 0]),
    'robot_accelerometer': array([-49.190372 ,   1.8504004,  23.964003 ], dtype=float32),
    'robot_gyro': array([0., 0., 0.], dtype=float32),
    'robot_velocimeter': array([0., 0., 0.], dtype=float32),
    'robot_quat': array([0.        , 0.        , 0.32422388, 0.94598037], dtype=float32),
    'goalkeeper_team_0_xpos_rel_robot': array([0., 0., 0.]),
    'goalkeeper_team_0_velp_rel_robot': array([0., 0., 0.]),
    'goalkeeper_team_1_xpos_rel_robot': array([0., 0., 0.]),
    'goalkeeper_team_1_velp_rel_robot': array([0., 0., 0.]),
    'target_xpos_rel_robot': array([12.44927069,  2.228258  , -0.699     ]),
    'target_velp_rel_robot': array([ 0.0799671 ,  0.05742587, -0.028308  ]),
    'defender_xpos': array([-5.20004893, -4.03897027,  0.7       , -5.20004893, -4.03897027,  0.7       , -5.20004893, -4.03897027,  0.7       ]),
    'success': False,
    'reward_terms': {
        'offside': np.False_,
        'success': False,
        'distance': np.float64(7.592963328178988e-06)
    }
}
```

### LowerT1GoaliePenaltyKick-v0

```python
info = {
    'length': 10.97,
    'width': 6.87,
    'goal_width': 1.6,
    'goal_height': 1.9,
    'goal_depth': 1.6,
    'goal_team_0_rel_robot': array([-4.2,  0. , -0.7]),
    'goal_team_1_rel_robot': array([17.74,  0.  , -0.7 ]),
    'goal_team_0_rel_ball': array([-2.2,  0. ,  0. ]),
    'goal_team_1_rel_ball': array([19.74,  0.  ,  0.  ]),
    'ball_xpos_rel_robot': array([-2. ,  0. , -0.7]),
    'ball_velp_rel_robot': array([-0.09953718,  0.00333679,  0.02199158]),
    'ball_velr_rel_robot': array([-0.04522436,  0.75377242,  0.13256413]),
    'player_team': array(),
    'robot_accelerometer': array([-49.769882 ,   1.5891316,  24.37023  ], dtype=float32),
    'robot_gyro': array([0., 0., 0.], dtype=float32),
    'robot_velocimeter': array([0., 0., 0.], dtype=float32),
    'robot_quat': array([0.000000e+00, 0.000000e+00, 9.999997e-01, 7.963267e-04], dtype=float32),
    'goalkeeper_team_0_xpos_rel_robot': array([-4.2,  0. , -0.5]),
    'goalkeeper_team_0_velp_rel_robot': array([-0.09953458,  1.88829238, -0.02912046]),
    'goalkeeper_team_1_xpos_rel_robot': array([17.74,  0.  , -0.5 ]),
    'goalkeeper_team_1_velp_rel_robot': array([-0.09953458,  1.88829238, -0.02912046]),
    'target_xpos_rel_robot': array([0., 0., 0.]),
    'target_velp_rel_robot': array([0., 0., 0.]),
    'defender_xpos': array([-6.77,  0.  ,  0.7 , -6.77,  0.  ,  0.7 , -6.77,  0.  ,  0.7 ]),
    'success': False,
    'reward_terms': {
        'robot_distance_ball': np.float32(0.028464139),
        'ball_vel_twd_goal': np.float64(2.604932644723651e-06),
        'offside': np.False_,
        'ball_hits': np.float64(0.0),
        'robot_fallen': False,
        'goal_scored': False,
        'ball_blocked': False
    }
}
```

### LowerT1ObstaclePenaltyKick-v0

```python
info = {
    'length': 10.97,
    'width': 6.87,
    'goal_width': 1.6,
    'goal_height': 1.9,
    'goal_depth': 1.6,
    'goal_team_0_rel_robot': array([-4.20080304e+00,  3.90488002e-05, -7.00155275e-01]),
    'goal_team_1_rel_robot': array([ 1.77391970e+01,  3.90488002e-05, -7.00155275e-01]),
    'goal_team_0_rel_ball': array([-2.19999997e+00,  1.37458607e-19, -5.28282194e-04]),
    'goal_team_1_rel_ball': array([ 1.97400000e+01,  1.37458607e-19, -5.28282194e-04]),
    'ball_xpos_rel_robot': array([-2.00080306e+00,  3.90488002e-05, -6.99626993e-01]),
    'ball_velp_rel_robot': array([-0.1888286 ,  0.01303736,  0.13430434]),
    'ball_velr_rel_robot': array([-0.06376551,  1.47738702,  0.34152387]),
    'player_team': array([1, 0]),
    'robot_accelerometer': array([-11.524847 ,   1.3702672,   4.5982413], dtype=float32),
    'robot_gyro': array([-0.05581531,  1.287113  , -0.26800087], dtype=float32),
    'robot_velocimeter': array([-0.16599128,  0.00976243,  0.0200204 ], dtype=float32),
    'robot_quat': array([-3.0923956e-03, -1.5803096e-04,  9.9999422e-01,  1.3893220e-03], dtype=float32),
    'goalkeeper_team_0_xpos_rel_robot': array([0., 0., 0.]),
    'goalkeeper_team_0_velp_rel_robot': array([0., 0., 0.]),
    'goalkeeper_team_1_xpos_rel_robot': array([0., 0., 0.]),
    'goalkeeper_team_1_velp_rel_robot': array([0., 0., 0.]),
    'target_xpos_rel_robot': array([-4.20080304,  0.39003905,  0.53795812]),
    'target_velp_rel_robot': array([-0.18882119,  0.01303736, -0.01076135]),
    'defender_xpos': array([-10.97,  -0.39,   0.6 , -10.97,  -1.17,   0.6 , -10.97,   1.17, 0.6 ]),
    'success': False,
    'reward_terms': {
        'robot_distance_ball': np.float32(0.028428555),
        'ball_vel_twd_goal': np.float64(7.414924496410714e-06),
        'offside': np.False_,
        'ball_hits': np.float64(0.0),
        'robot_fallen': False,
        'goal_scored': False,
        'ball_blocked': False
    }
}
```

### LowerT1PenaltyKick-v0

```python
info = {
    'length': 10.97,
    'width': 6.87,
    'goal_width': 1.6,
    'goal_height': 1.9,
    'goal_depth': 1.6,
    'goal_team_0_rel_robot': array([-4.2,  0. , -0.7]),
    'goal_team_1_rel_robot': array([17.74,  0.  , -0.7 ]),
    'goal_team_0_rel_ball': array([-2.2,  0. ,  0. ]),
    'goal_team_1_rel_ball': array([19.74,  0.  ,  0.  ]),
    'ball_xpos_rel_robot': array([-2. ,  0. , -0.7]),
    'ball_velp_rel_robot': array([-0.09953718,  0.00333679,  0.02199158]),
    'ball_velr_rel_robot': array([-0.04522436,  0.75377242,  0.13256413]),
    'player_team': array(),
    'robot_accelerometer': array([-49.769882 ,   1.5891316,  24.37023  ], dtype=float32),
    'robot_gyro': array([0., 0., 0.], dtype=float32),
    'robot_velocimeter': array([0., 0., 0.], dtype=float32),
    'robot_quat': array([0.000000e+00, 0.000000e+00, 9.999997e-01, 7.963267e-04], dtype=float32),
    'goalkeeper_team_0_xpos_rel_robot': array([0., 0., 0.]),
    'goalkeeper_team_0_velp_rel_robot': array([0., 0., 0.]),
    'goalkeeper_team_1_xpos_rel_robot': array([0., 0., 0.]),
    'goalkeeper_team_1_velp_rel_robot': array([0., 0., 0.]),
    'target_xpos_rel_robot': array([0., 0., 0.]),
    'target_velp_rel_robot': array([0., 0., 0.]),
    'defender_xpos': array([-6.77,  0.  ,  0.7 , -6.77,  0.  ,  0.7 , -6.77,  0.  ,  0.7 ]),
    'success': False,
    'reward_terms': {
        'robot_distance_ball': np.float32(0.028464139),
        'ball_vel_twd_goal': np.float64(2.604932644723651e-06),
        'offside': np.False_,
        'ball_hits': np.float64(0.0),
        'robot_fallen': False,
        'goal_scored': False
    }
}
```
