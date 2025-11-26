import os
import yaml, logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.utils.simulation import SimulationEnvironment
from src.learning.agent import Agent

logger = logging.getLogger(__name__)
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)


# input and output shapes
N_STATES = config['model']['states']
N_ACTIONS = config['model']['actions']

# PPO Hyperparameters
PPO_EPOCHS = config['ppo']['epochs']         # Number of times to update the policy using the collected data
MINIBATCH_SIZE = config['ppo']['batch_size']     # Size of mini-batch for gradient descent
CLIP_EPSILON = config['ppo']['clip_epsilon']      # Clipping parameter (epsilon)
GAMMA = config['ppo']['gamma']            # Discount factor
GAE_LAMBDA = config['ppo']['gae_lambda']       # GAE factor
LR = config['ppo']['learning_rate']               # Learning rate
ENTROPY_BETA = config['ppo']['entropy_beta']    # Entropy Coefficient
VALUE_LOSS_COEF = config['ppo']['value_loss_coeff']     # Value Loss Coefficient
CLIP_GRADIENTS = True                          # Whether to clip gradients or not

SEED_WEIGHTS_PATH = "./data/il_actor_seed_weights.pt"

time_step = 0
class PPOAgent:
    """
    The main PPO Agent class responsible for interaction and learning.
    """
    def __init__(self, environment: SimulationEnvironment):
        self.env = environment
        self.network = Agent(N_STATES, N_ACTIONS)
        if SEED_WEIGHTS_PATH and os.path.exists(SEED_WEIGHTS_PATH):
            state_dict = torch.load(SEED_WEIGHTS_PATH)
            temp_agent = Agent(N_STATES, N_ACTIONS)
            temp_agent.load_state_dict(state_dict)

            self.network.shared_net.load_state_dict(temp_agent.shared_net.state_dict())
            self.network.actor_head.load_state_dict(temp_agent.actor_head.state_dict())
            self.network.log_std.data.copy_(temp_agent.log_std.data)

        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.gae_lambda = GAE_LAMBDA
        self.time_step = 0

    def select_action(self, state):
        """ Selects an action based on the current policy. """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        _, mu, std, value = self.network(state_tensor)

        # Create a Gaussian distribution and sample an action
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Action is in [-1, 1]. You must scale this to your physical joint velocity limits.
        # e.g., actual_velocity = action.clamp(-1.0, 1.0) * MAX_VELOCITY

        return action.detach().numpy().flatten(), log_prob.item(), value.item()

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
            td_error = rewards[t] + GAMMA * V_tp1 * (1 - dones[t]) - values[t]

            # GAE: Delta_t + gamma * lambda * A_{t+1}
            last_gae = td_error + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_gae
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
                _, mu, std, values_new = self.network(s)

                # Calculate new log probabilities
                dist = torch.distributions.Normal(mu, std)
                new_log_probs = dist.log_prob(a).sum(dim=-1)

                # Policy Loss (PPO Objective)
                # 1. Ratio of new policy to old policy
                ratio = torch.exp(new_log_probs - old_p)

                # 2. Clipped surrogate loss calculation
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * adv

                policy_loss = -torch.min(surr1, surr2).mean() # Maximize objective -> Minimize negative loss

                # Value Loss (Critic Update) - MSE
                value_loss = (values_new.squeeze() - ret).pow(2).mean()

                # Total Loss (You may want to add an entropy term here for exploration)
                entropy = dist.entropy().mean()
                total_loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_BETA * entropy # Entropy term encourages exploration

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                if CLIP_GRADIENTS:
                    nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) # Optional: gradient clipping
                self.optimizer.step()

def _calculate_reward(terminated, info, time_step):
    """
    Reward Function
    #TODO: need state when falls over, time step, distance travelled,
    """
    # - [ ] Make the model to run in a straight line for as long as possible and as fast as possible:
        #   - Reward always has a -1 for time
        #   - If he falls, reward is -1000 and terminate episode
        #   - Reward will have +x, where x is the local distance travelled in each episode

    reward = 0.0
    reward -= time_step # Time penalty
    if terminated:
        reward -= 1000.0 # Fall penalty
    dist_to_ball = info.get('ball_xpos_rel_robot')
    if dist_to_ball is None:
        raise ValueError("Info dictionary does not contain 'ball_xpos_rel_robot' key.")
    dist_to_ball = np.linalg.norm(dist_to_ball)
    reward -= max(0, 20.0 - dist_to_ball)

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
    return reward # Placeholder

def train():
    ENV_NAME = "LowerT1KickToTarget-v0"
    env = SimulationEnvironment(ENV_NAME, headless=True)
    agent = PPOAgent(environment=env)

    max_timesteps = 1000000 # Total timesteps to train
    collect_steps = 2048    # How many steps to collect before updating the policy

    state, info = env.reset()
    time_step = 0

    for t in range(max_timesteps):

        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

        for step in range(collect_steps):
            # Select action
            action, log_prob, value_estimate = agent.select_action(state)

            # Step environment
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            ctrl, _, _, _ = agent.network(state_tensor)

            # Step environment using the PD control output 'ctrl'
            next_state, _, terminated, truncated, info = env.step(ctrl.cpu().numpy().flatten())

            # --- Custom Reward Calculation ---
            time_step += 1
            done = terminated or truncated

            reward = _calculate_reward(terminated, info, episode_time_step)

            # Store data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value_estimate)
            log_probs.append(log_prob)
            dones.append(done)

            state = next_state

            if done:
                state, _ = env.reset()
                episode_time_step = 0 # Reset episode time step

        # Get V(s_T) for the last state in the rollout (or 0 if done)
        if done:
            next_value = 0.0
        else:
            # We only need the value head output
            _, _, _, next_value_tensor = agent.network(torch.FloatTensor(state).unsqueeze(0))
            next_value = next_value_tensor.item()

        advantages, returns = agent.compute_gae(
            np.array(rewards), np.array(values), next_value, np.array(dones)
        )

        agent.update_policy(states, actions, log_probs, advantages, returns)

        print(f"Timestep: {(t+1)*collect_steps}, Avg Reward: {np.mean(rewards):.2f}")

    env.close()

if __name__ == '__main__':
    # train() # Uncomment to start training
    print("PPO Skeleton Loaded. Focus on customizing _calculate_reward method and ensuring correct Agent output is used for env.step().")