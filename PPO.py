import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# input and output shapes
N_STATES = 78   # (28 joints + 3 robot pose + 2 ball pos + 6 target) *2 for velocities of each
N_ACTIONS = 28  # 28 joint velocities

# PPO Hyperparameters
PPO_EPOCHS = 50         # Number of times to update the policy using the collected data
MINIBATCH_SIZE = 64     # Size of mini-batch for gradient descent
CLIP_EPSILON = 0.2      # Clipping parameter (epsilon)
GAMMA = 0.99            # Discount factor
GAE_LAMBDA = 0.95       # GAE factor
LR = 3e-4               # Learning rate

class ActorCritic(nn.Module):
    """
    Defines the shared backbone and separate heads for the Actor and Critic networks.
    """
    def __init__(self, n_states, n_actions):
        super(ActorCritic, self).__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.actor_head = nn.Sequential(
            nn.Linear(256, n_actions),
            nn.Tanh()  # Use Tanh to constrain output between [-1, 1], then scale later
        )

        self.critic_head = nn.Linear(256, 1)

        # Log Standard Deviation (for continuous actions)
        # This is a parameter, not a layer output, and is learned directly.
        # Initialize to a small value (e.g., log(0.1) ~ -2.3)
        self.log_std = nn.Parameter(torch.zeros(1, n_actions) - 2.3)

    def forward(self, x):
        features = self.shared_net(x)

        # Actor output: Mean of the Gaussian distribution
        mu = self.actor_head(features)

        # Critic output: State Value
        value = self.critic_head(features)

        # Standard deviation (exp is used since log_std is what we learn)
        std = torch.exp(self.log_std.expand_as(mu))

        return mu, std, value

class PPOAgent:
    """
    The main PPO Agent class responsible for interaction and learning.
    """
    def __init__(self):
        self.network = ActorCritic(N_STATES, N_ACTIONS)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.clip_epsilon = CLIP_EPSILON
        self.gamma = GAMMA
        self.gae_lambda = GAE_LAMBDA

    def select_action(self, state):
        """ Selects an action based on the current policy. """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mu, std, _ = self.network(state_tensor)

        # Create a Gaussian distribution and sample an action
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Action is in [-1, 1]. You must scale this to your physical joint velocity limits.
        # e.g., actual_velocity = action.clamp(-1.0, 1.0) * MAX_VELOCITY

        return action.detach().numpy().flatten(), log_prob.item()

    def compute_gae(self, rewards, values, next_value, dones):
        """ Computes Generalized Advantage Estimation (GAE). """
        # rewards, values, dones are numpy arrays or lists from the rollout

        # Calculate advantages and returns
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        # The rewards/values/dones are usually collected from t=0 to T-1.
        # We iterate backwards from T-1 to 0.
        for t in reversed(range(len(rewards))):
            # V(s_t+1) is next_value for the last step, or V(s_t) for preceding steps
            V_tp1 = values[t+1] if t + 1 < len(values) else next_value

            # td_error: R_t + gamma * V(s_t+1) - V(s_t)
            td_error = rewards[t] + self.gamma * V_tp1 * (1 - dones[t]) - values[t]

            # GAE: Delta_t + gamma * lambda * A_{t+1}
            last_gae = td_error + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

            # Returns: GAE_t + V(s_t)
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        """ Performs the PPO policy and value updates. """

        # Convert collected data to PyTorch Tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Normalize advantages for more stable learning
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Iterate through data for PPO_EPOCHS
        for _ in range(PPO_EPOCHS):
            # Create mini-batches for efficiency
            for i in range(0, len(states), MINIBATCH_SIZE):
                s = states[i:i+MINIBATCH_SIZE]
                a = actions[i:i+MINIBATCH_SIZE]
                old_p = old_log_probs[i:i+MINIBATCH_SIZE]
                adv = advantages[i:i+MINIBATCH_SIZE]
                ret = returns[i:i+MINIBATCH_SIZE]

                # Forward pass
                mu, std, values_new = self.network(s)

                # Calculate new log probabilities
                dist = torch.distributions.Normal(mu, std)
                new_log_probs = dist.log_prob(a).sum(dim=-1)

                # Policy Loss (PPO Objective)
                # 1. Ratio of new policy to old policy
                ratio = torch.exp(new_log_probs - old_p)

                # 2. Clipped surrogate loss calculation
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv

                policy_loss = -torch.min(surr1, surr2).mean() # Maximize objective -> Minimize negative loss

                # Value Loss (Critic Update) - MSE
                value_loss = (values_new.squeeze() - ret).pow(2).mean()

                # Total Loss (You may want to add an entropy term here for exploration)
                entropy = dist.entropy().mean()
                total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy # Entropy term encourages exploration

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) # Optional: gradient clipping
                self.optimizer.step()

class RobotEnvironment:
    """
    A conceptual class for your custom robot environment interface.
    """
    def __init__(self):
        # Initialize your robot, simulator, etc.
        self.current_state = np.zeros(N_STATES)
        self.time_step = 0
        self.MAX_STEPS = 500 # Your fixed time horizon (steps)

    def reset(self):
        """ Resets the environment state for a new episode. """
        # Reset joint positions, ball, and obstacle locations
        self.time_step = 0
        self.current_state = np.random.uniform(-1, 1, N_STATES) # Example
        return self.current_state

    def step(self, action):
        """ Takes an action and returns the next state, reward, and done flag. """

        # 1. Apply action to your simulator/robot
        # E.g., self.robot.set_joint_velocities(action * MAX_VELOCITY)

        # 2. Advance simulation time and get next state
        next_state = self._simulate_step(action)
        self.time_step += 1

        # 3. Calculate Reward (The part you will customize!)
        reward = self._calculate_reward(next_state, action)

        # 4. Check for terminal condition
        done = self.time_step >= self.MAX_STEPS or self._is_task_finished(next_state) or self._is_collision(next_state)

        # Placeholder for optional info dictionary
        info = {}

        return next_state, reward, done, info

    # --- CUSTOM REWARD AND TERMINATION LOGIC ---

    def _calculate_reward(self, state, action):
        """
        Reward Function
        #TODO: need state when falls over, time step, distance travelled,
        """
            # - [ ] Make the model to run in a straight line for as long as possible and as fast as possible:
            #   - Reward always has a -1 for time
            #   - If he falls, reward is -1000 and terminate episode
            #   - Reward will have +x, where x is the local distance travelled in each episode
            # - [ ] Make the model run in a straight line and then stop without falling
            #   - Reward always has a -1 for time
            #   - If he falls, reward is -1000 and terminate episode
            #   - Reward will have +x, where x is the local distance travelled in each episode as long as he is before the target
            #   - Reward will have -x, where x is the local distance travelled in each episode if he passes the target
            #   - Reward will have +1 if he stays within 0.5 meters infront of the target
            # - [ ] Make the model run in random directions and then stop without falling
            #   - Reward always has a -1 for time
            #   - If he falls, reward is -1000 and terminate episode
            #   - Reward will have +x, where x is the local distance travelled in each episode as long as he is before the target
            #   - Reward will have -x, where x is the local distance travelled in each episode if he passes the target
            #   - Reward will have +1 if he stays within 0.5 meters infront of the target
            #   - Target location will change to a random position every 5-10 seconds (time chosen is random)
        return 0.1 # Placeholder

    def _is_task_finished(self, state):
        # Check if the ball is within the target bounds
        return False

    def _is_collision(self, state):
        # Check if any joint/link collides with an obstacle
        return False

    def _simulate_step(self, action):
        # Placeholder for interaction with your simulator (e.g., PyBullet, MuJoCo)
        # This function updates the state based on the action
        return self.current_state # Placeholder

def train():
    env = RobotEnvironment()
    agent = PPOAgent()

    max_timesteps = 1000000 # Total timesteps to train
    collect_steps = 2048    # How many steps to collect before updating the policy

    state = env.reset()

    for t in range(max_timesteps):

        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

        for step in range(collect_steps):
            # Select action
            action, log_prob = agent.select_action(state)

            # Step environment
            next_state, reward, done, _ = env.step(action)

            # Get Value Estimate for the current state (V(s_t))
            _, _, value_estimate = agent.network(torch.FloatTensor(state).unsqueeze(0))

            # Store data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value_estimate.item())
            log_probs.append(log_prob)
            dones.append(done)

            state = next_state

            if done:
                state = env.reset()

        # Get V(s_T) for the last state in the rollout (or 0 if done)
        if done:
            next_value = 0.0
        else:
            _, _, next_value_tensor = agent.network(torch.FloatTensor(state).unsqueeze(0))
            next_value = next_value_tensor.item()

        advantages, returns = agent.compute_gae(
            np.array(rewards), np.array(values), next_value, np.array(dones)
        )

        # 3. Update Policy
        agent.update_policy(states, actions, log_probs, advantages, returns)

        print(f"Timestep: {t*collect_steps}, Avg Reward: {np.mean(rewards):.2f}")

if __name__ == '__main__':
    # train() # Uncomment to start training
    print("PPO Skeleton Loaded. Focus on customizing RobotEnvironment and the _calculate_reward method.")