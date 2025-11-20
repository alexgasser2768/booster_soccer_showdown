#

# Learn Kinematics

- [ ] Imitation Learning to run
- [ ] Make the model to run in a straight line for as long as possible and as fast as possible:
    - Reward always has a -1 for time
    - If he falls, reward is -1000 and terminate episode
    - Reward will have +x, where x is the local distance travelled in each episode
- [ ] Make the model run in a straight line and then stop without falling
    - Reward always has a -1 for time
    - If he falls, reward is -1000 and terminate episode
    - Reward will have +x, where x is the local distance travelled in each episode as long as he is before the target
    - Reward will have -x, where x is the local distance travelled in each episode if he passes the target
    - Reward will have +1 if he stays within 0.5 meters infront of the target
- [ ] Make the model run in random directions and then stop without falling
    - Reward always has a -1 for time
    - If he falls, reward is -1000 and terminate episode
    - Reward will have +x, where x is the local distance travelled in each episode as long as he is before the target
    - Reward will have -x, where x is the local distance travelled in each episode if he passes the target
    - Reward will have +1 if he stays within 0.5 meters infront of the target
    - Target location will change to a random position every 5-10 seconds (time chosen is random)

# Learn passing

Make sure in the beginning, the target is not too far away (1-2 meters). As the model becomes better and better,
make the target further and further.

- [ ] Imitation Learning on passing motion
- [ ] Imitation Learning on shooting motion
- [ ] Make the model to pass in a straight line:
    - Reward always has a -1 for time
    - If he falls, reward is -1000 and terminate episode
    - Reward will have -x, where x is the final distance from ball to target
    - Reward will have +1/x if the ball is within 0.5 meters of the target
    - If ball is exactly on target (<0.01 meters from target), reward is +100
- [ ] Make the model run to ball and pass:
    - Reward always has a -1 for time
    - If he falls, reward is -1000 and terminate episode
    - Reward will have +x, where x is the local distance travelled in each episode as long as he is before the ball
    - Reward will have -x, where x is the local distance travelled in each episode if he passes the ball
    - Reward will have +1 if he stays within 0.5 meters infront of the ball
    - Reward will have -x, where x is the final distance from ball to target
    - Reward will have +1/x if the ball is within 0.5 meters of the target
    - If ball is exactly on target (<0.01 meters from target), reward is +100
- [ ] Make the model run in random directions and then pass:
    - Reward always has a -1 for time
    - If he falls, reward is -1000 and terminate episode
    - Reward will have +x, where x is the local distance travelled in each episode as long as he is before the ball
    - Reward will have -x, where x is the local distance travelled in each episode if he passes the ball
    - Reward will have +1 if he stays within 0.5 meters infront of the ball
    - Reward will have -x, where x is the final distance from ball to target
    - Reward will have +1/x if the ball is within 0.5 meters of the target
    - If ball is exactly on target (<0.01 meters from target), reward is +100
    - Target location will change to a random position every 5-10 seconds (time chosen is random). Ball location doesn't change

## Learn shooting

Shooting is essentially complicated passing in 3D with more force.

- [ ] Make the model run in random directions and then shoot with no obstacle:
    - Reward always has a -1 for time
    - If he falls, reward is -1000 and terminate episode
    - Reward will have +x, where x is the local distance travelled in each episode as long as he is before the ball
    - Reward will have -x, where x is the local distance travelled in each episode if he passes the ball
    - Reward will have +1 if he stays within 0.5 meters infront of the ball
    - Reward will have -x, where x is the final distance from ball to goal
    - If the robot makes a goal, the reward is +100 and episode terminates

- [ ] Make the model run in random directions and then shoot with obstacles:
    - Reward always has a -1 for time
    - If he falls, reward is -1000 and terminate episode
    - Reward will have +x, where x is the local distance travelled in each episode as long as he is before the ball
    - Reward will have -x, where x is the local distance travelled in each episode if he passes the ball
    - Reward will have +1 if he stays within 0.5 meters infront of the ball
    - Reward will have -x, where x is the final distance from ball to goal
    - If the ball hits an obstacle, an extra -10 penalty will be added
    - If the robot makes a goal, the reward is +100 and episode terminates

# RL Agent design

## Robot state

The robot state will consist of the 5 past frames.

- Joint positions between [-1, 1] (28 components)
- Robot position and orientation (3 components)
- Ball position (2 components)
- Pass Target position (2 components)
- Largest open space in goal (6 components: x_min, y_min, z_min, x_max, y_max, z_max)
- Total: 41 x 5 components = 205 components

## Model Architecture

- Deep FCN (6+ Layers). Each layer is between 250 - 300 neurons (Max. Neurons: 205 + n*300 + 28)
- Leaky ReLU will be used (no information loss from gradient)
- Output layer is joint velocity from -1 to 1
    - Output will use following function: 2 * softmax(x) - 1
    - Joint velocitities will be multiplied by the max velocity for each joint

## Concerns

- Do we need to teach the model how to dripple (aka run with ball)?
- What if for pass, the model learns to run with ball towards target and then stop? How do we penalize this behaviour
- Will the current training structure make the model learn to always shoot or pass when ball is nearby? In some situations, the model should get closer to the target with the ball first before shooting or passing. How do we teach it this behaviour?
    - We can add a reward to make the robot get within 10 meters of target before passing or shooting.
- Is the penalty for falling too high? If the falling penalty is too high (higher than rewards), the model might fall into a local minima that makes it stay in place (since it is not as bad as falling) instead of taking a risk and moving around.
