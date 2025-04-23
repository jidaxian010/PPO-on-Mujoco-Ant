import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from IPython.display import clear_output
import functools
import time
import os

# Set environment variable for headless rendering
os.environ['MUJOCO_GL'] = 'egl' # Try 'egl', 'glfw', or 'osmesa' if one doesn't work

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# PPO Actor and Critic Networks
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.action_std = action_std_init
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # Move networks to device
        self.actor.to(device)
        self.critic.to(device)
        
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.action_var = torch.full(self.action_var.shape, new_action_std * new_action_std).to(device)
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(0)
        dist = Normal(action_mean, self.action_std)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return action, action_logprob
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = Normal(action_mean, self.action_std)
        
        # For single action continuous environments
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

# PPO Agent class
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.97, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        
        return action.cpu().detach().numpy(), action_logprob.cpu().detach()
    
    def update(self, memory):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert lists to tensors
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

# Memory for storing trajectory
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# Function to save model and environment information
def save_model(agent, env_name, env_info, filename=None):
    """Save the model and environment information for later loading"""
    if filename is None:
        # Add timestamp to prevent overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{env_name.lower()}_ppo_model_{timestamp}"
    
    # Create a directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the policy state dict
    model_path = f"models/{filename}.pth"
    torch.save(agent.policy.state_dict(), model_path)
    
    # Save environment information for proper loading later
    env_info_path = f"models/{filename}_info.pt"
    torch.save(env_info, env_info_path)
    
    print(f"Model saved to {model_path}")
    print(f"Environment info saved to {env_info_path}")
    
    return model_path, env_info_path

# Train the agent
def train(env_name='Ant-v4', num_timesteps=50_000_000, episode_length=1000, 
          reward_scaling=10.0, normalize_observations=True, 
          num_evals=10, seed=1, quick_test=False, save_freq=None):
    
    # Scale down for testing
    if quick_test:
        num_timesteps = 500_000
        num_evals = 2
    
    # Environment setup
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    if normalize_observations:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformReward(env, lambda r: reward_scaling * r)
    
    # Environment information
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Store environment info for later loading
    env_info = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "env_name": env_name,
        "reward_scaling": reward_scaling,
        "normalize_observations": normalize_observations
    }
    
    print("State dimension:", state_dim)
    print("Action dimension:", action_dim)
    print("Action bounds:", env.action_space.low[0], env.action_space.high[0])
    
    # Create PPO agent with hyperparameters similar to the Brax implementation
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.97,
        K_epochs=4,  # num_updates_per_batch in original code
        eps_clip=0.2
    )
    
    # Training variables
    max_ep_len = episode_length
    time_step = 0
    i_episode = 0
    
    # Unroll and batch settings - adapted from Brax
    unroll_length = 5
    # Increase batch size for better GPU utilization
    batch_size = 2048 if not quick_test else 1024  # Reduced batch size to avoid GPU OOM
    update_timestep = batch_size * unroll_length  # Equivalent to batch collection in Brax
    
    # Adjust learning rates for larger batches
    agent.optimizer = torch.optim.Adam([
        {'params': agent.policy.actor.parameters(), 'lr': 3e-4 * 2},  # Slightly higher learning rate
        {'params': agent.policy.critic.parameters(), 'lr': 3e-4 * 2}
    ])
    
    # Tracking metrics
    episode_rewards = []
    eval_rewards = []
    episode_lengths = []
    eval_timestamps = []
    
    # Action std decay settings
    action_std = 0.6
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    action_std_decay_freq = num_timesteps // 20  # Decay a few times during training
    
    # Set save frequency if not provided
    if save_freq is None:
        save_freq = num_timesteps // 5  # Save 5 checkpoints during training
    
    # Rollout buffer
    buffer = RolloutBuffer()
    
    # For progress tracking
    start_time = datetime.now()
    jit_time = datetime.now()
    times = [start_time]
    
    # Start training
    print(f"Starting training for {num_timesteps} timesteps...")
    print(f"Quick test mode: {quick_test}")
    
    # Main training loop
    state, _ = env.reset(seed=seed)
    current_ep_reward = 0
    current_ep_length = 0
    
    while time_step < num_timesteps:
        # Select action
        action, logprob = agent.select_action(state)
        
        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store in buffer
        buffer.states.append(torch.FloatTensor(state).to(device))
        buffer.actions.append(torch.FloatTensor(action).to(device))
        buffer.logprobs.append(logprob.to(device))
        buffer.rewards.append(reward)
        buffer.is_terminals.append(done)
        
        # Update counters
        time_step += 1
        current_ep_reward += reward
        current_ep_length += 1
        
        # Update state
        state = next_state
        
        # Update PPO agent
        if time_step % update_timestep == 0:
            agent.update(buffer)
            buffer.clear()
            
            # Track progress
            eval_frequency = max(1, num_timesteps // (update_timestep * num_evals))
            if (time_step // update_timestep) % eval_frequency == 0:
                times.append(datetime.now())
                
                # Calculate average training reward instead of doing a separate evaluation
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                eval_rewards.append(avg_reward)
                eval_timestamps.append(time_step)
                
                # Plot progress
                clear_output(wait=True)
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(eval_timestamps, eval_rewards)
                plt.title('Training Progress')
                plt.xlabel('Timesteps')
                plt.ylabel(f'Average Reward (last 10 episodes)')
                
                plt.subplot(1, 2, 2)
                plt.plot(episode_rewards)
                plt.title('Episode Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.tight_layout()
                plt.show()

                # Print progress
                time_elapsed = (times[-1] - times[0]).total_seconds() / 60
                steps_per_sec = time_step / (time_elapsed * 60)
                print(f"Progress: {time_step}/{num_timesteps} ({100 * time_step / num_timesteps:.2f}%)")
                print(f"Avg Reward (last 10 episodes): {avg_reward:.2f}")
                print(f"Time elapsed: {time_elapsed:.2f} minutes")
                print(f"Steps per second: {steps_per_sec:.2f}")
            
            # Save model at regular intervals
            if save_freq > 0 and time_step % save_freq == 0:
                checkpoint_filename = f"{env_name.lower()}_ppo_checkpoint_{time_step}"
                save_model(agent, env_name, env_info, checkpoint_filename)
        
        # Action std decay
        if time_step % action_std_decay_freq == 0:
            agent.policy_old.action_std = max(agent.policy_old.action_std - action_std_decay_rate, min_action_std)
            agent.policy.action_std = agent.policy_old.action_std
            agent.policy.action_var = torch.full((action_dim,), agent.policy.action_std * agent.policy.action_std).to(device)
            agent.policy_old.action_var = agent.policy.action_var
            print(f"Time step: {time_step}, New action std: {agent.policy.action_std}")
        
        # If episode ended
        if done or current_ep_length >= max_ep_len:
            # Record episode stats
            episode_rewards.append(current_ep_reward)
            episode_lengths.append(current_ep_length)
            
            # Reset episode tracking
            state, _ = env.reset(seed=seed+i_episode)
            current_ep_reward = 0
            current_ep_length = 0
            i_episode += 1
            
            if i_episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode: {i_episode}, Avg. Reward (last 10): {avg_reward:.2f}")
    
    env.close()
    
    # Final timing
    end_time = datetime.now()
    print(f'Time to initialize: {jit_time - start_time}')
    print(f'Time to train: {end_time - jit_time}')
    print(f'Total training time: {end_time - start_time}')
    
    # Save final model
    model_path, info_path = save_model(agent, env_name, env_info)
    
    return agent, model_path, info_path

if __name__ == "__main__":
    # Just train and save the model
    print("Training ant to walk...")
    
    # Uncomment and modify these parameters as needed
    trained_agent, model_path, info_path = train(
        env_name="Ant-v4",
        # num_timesteps=1000000,
        num_timesteps=5_000_000 if torch.cuda.is_available() else 500_000,
        episode_length=1000,
        reward_scaling=10.0,
        normalize_observations=True,
        num_evals=10,
        quick_test=False,  # Set to True for a quick test, False for full training
        save_freq=100_000  # Save checkpoint every 100K steps
    )
    
    print(f"Training complete! Model saved to {model_path}")
    print(f"Run ant_evaluate.py to evaluate and visualize the trained model")