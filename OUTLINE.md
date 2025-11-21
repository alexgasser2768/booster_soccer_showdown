# Reward Structure

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
    - Reward will have +1 if he stays within 1 meter of the target
- Make the model run in random directions and then stop without falling
    - Same reward structure as previous task
    - Target location will change to a random position every 5-10 seconds (time chosen is random)

## Learn passing

Make sure in the beginning, the target is not too far away (1-2 meters). As the model becomes better and better,
make the target further and further (Curriculum Learning).

- Make the model to pass in a straight line:
    - Reward always has a -1 for time
    - If he falls, terminate episode
    - Reward will have -x, where x is the final distance from ball to target
    - Reward will have +10/x if the ball is within 0.5 meters of the target
    - If ball is exactly on target (<= 0.1 meters from target), reward is +100
- Make the model run to ball and pass:
    - Same reward structure as previous task
    - Start robot away from ball in random locations
- Make the model run in random directions and then pass:
    - Same reward structure as previous task
    - Target location will change to a random position every 5-10 seconds (time chosen is random). Ball location doesn't change

## Learn shooting

Shooting is essentially complicated passing in 3D with more force.

- Make the model run in random directions and then shoot with no obstacle:
    - Same reward structure as passing but if the robot makes a goal, the reward is +100 and episode terminates

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

## Outputs

- Joint Velocities between [-1, 1] (12 Outputs)

## Model Architecture

- Will use Proximal Policy Optimization (PPO)
- Deep FCN (6 Layers). Each layer is 300 neurons (Max. Neurons: 73 + 6*300 + 12)
- Leaky ReLU will be used (no information loss from gradient)
- Output layer is joint velocity from -1 to 1 (tanh)
    - Output will use tanh to normalize
    - The function `get_torque` in the `booster_control/t1_utils.py` file will be used to convert the normalized output into control signals

## Sample data from environment

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
    ],default observation from info
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

# Concerns

- Do we need to teach the model how to dripple (aka run with ball)?
- What if for pass, the model learns to run with ball towards target and then stop? Is the time penalty enough to stop this behaviour?
- Will the current training structure make the model learn to always shoot or pass when ball is nearby? In some situations, the model should get closer to the target with the ball first before shooting or passing. Is the location randomization enough for it to learn this behavior?
