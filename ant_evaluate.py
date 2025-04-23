import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set environment variable for headless rendering
os.environ['MUJOCO_GL'] = 'egl'  # Try 'egl', 'glfw', or 'osmesa' if one doesn't work

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# PPO Actor and Critic Networks - same as in training script
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

# Function to load model and environment info
def load_model(model_path, info_path=None):
    """Load a trained model and its environment information"""
    
    # If info path not provided, try to infer it
    if info_path is None:
        info_path = model_path.replace('.pth', '_info.pt')
    
    # Try to load the environment info
    try:
        env_info = torch.load(info_path, map_location=device)
        print(f"Loaded environment info from {info_path}")
    except:
        print(f"Could not load environment info from {info_path}")
        print("Using default environment settings")
        env_info = {
            "env_name": "Ant-v4",
            "state_dim": None,  # Will detect from environment
            "action_dim": None,  # Will detect from environment
            "reward_scaling": 10.0,
            "normalize_observations": True
        }
    
    # Create the environment based on loaded info
    env_name = env_info.get("env_name", "Ant-v4")
    normalize_observations = env_info.get("normalize_observations", True)
    reward_scaling = env_info.get("reward_scaling", 10.0)
    
    # Create environment
    env = gym.make(env_name)
    
    # Apply wrappers if needed
    if normalize_observations:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformReward(env, lambda r: reward_scaling * r)
    
    # Get state and action dimensions from environment if not in info
    state_dim = env_info.get("state_dim")
    if state_dim is None:
        state_dim = env.observation_space.shape[0]
    
    action_dim = env_info.get("action_dim")
    if action_dim is None:
        action_dim = env.action_space.shape[0]
    
    # Create the policy network
    policy = ActorCritic(state_dim, action_dim, action_std_init=0.1)  # Low std for evaluation
    
    # Load the saved model
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()  # Set to evaluation mode
    
    return policy, env, env_info

# Evaluation function
def evaluate(policy, env, episodes=10, render=True, record_video=False, video_path="videos"):
    """Evaluate a trained policy on an environment"""
    
    # Set up the environment for rendering if needed
    if record_video:
        # Create directory for videos
        os.makedirs(video_path, exist_ok=True)
        env = gym.make(env.spec.id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_path)
    elif render:
        try:
            env = gym.make(env.spec.id, render_mode="human")
        except Exception as e:
            print(f"Could not create environment with human rendering: {e}")
            print("Falling back to rgb_array rendering")
            env = gym.make(env.spec.id, render_mode="rgb_array")
    
    # Run evaluation episodes
    total_rewards = []
    all_episode_states = []
    all_episode_actions = []
    
    for ep in range(episodes):
        states = []
        actions = []
        
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action, _ = policy.act(state_tensor)
                action = action.cpu().numpy()
            
            # Store state and action
            states.append(state.copy())
            actions.append(action.copy())
            
            # Execute action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            # Render if using rgb_array mode
            if render and hasattr(env, 'render_mode') and env.render_mode == "rgb_array":
                env.render()
        
        # Store episode data
        total_rewards.append(episode_reward)
        all_episode_states.append(np.array(states))
        all_episode_actions.append(np.array(actions))
        
        print(f"Evaluation episode {ep+1}/{episodes}: Reward = {episode_reward:.2f}")
    
    # Print summary
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nEvaluation results over {episodes} episodes:")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    return {
        'rewards': total_rewards,
        'states': all_episode_states,
        'actions': all_episode_actions,
        'avg_reward': avg_reward,
        'std_reward': std_reward
    }

# Function to create videos
def create_video(env_name, model_path, info_path=None, video_name=None, video_path="videos"):
    """Create a video of the agent's performance"""
    
    # Load model and environment
    policy, env, _ = load_model(model_path, info_path)
    
    # Setup video recording
    if video_name is None:
        video_name = os.path.basename(model_path).replace('.pth', '')
        
    full_video_path = os.path.join(video_path, video_name)
    os.makedirs(full_video_path, exist_ok=True)
    
    # Make environment with video recording
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, full_video_path)
    
    # Run a single episode
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Get action
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action, _ = policy.act(state_tensor)
            action = action.cpu().numpy()
        
        # Execute action
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    env.close()
    print(f"Video recording complete. Total reward: {total_reward:.2f}")
    print(f"Video saved in {full_video_path}")
    
    return full_video_path

# Function to visualize trajectories
def visualize_trajectories(results):
    """Visualize trajectories from evaluation results"""
    
    # Only plot if we have state data
    if 'states' not in results or len(results['states']) == 0:
        print("No trajectory data available for visualization")
        return
    
    # Get total number of episodes
    total_episodes = len(results['states'])
    
    # Select the last three episodes (or fewer if less than 3 episodes)
    num_episodes_to_plot = min(3, total_episodes)
    # Calculate the starting index to get the last episodes
    start_idx = max(0, total_episodes - num_episodes_to_plot)
    
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.bar(range(len(results['rewards'])), results['rewards'])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Evaluation Rewards')
    
    # Plot x-y positions
    plt.subplot(2, 2, 2)
    for i in range(start_idx, total_episodes):
        # Extract x-y position (assuming it's in the state)
        # This might need to be adjusted based on the actual state representation
        episode_idx = i - start_idx  # Relative index for legend
        episode_states = results['states'][i]
        if episode_states.shape[1] >= 2:  # Make sure we have at least x, y
            plt.plot(episode_states[:, 0], episode_states[:, 1], 
                     label=f'Episode {i+1}/{total_episodes}')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Agent Trajectory (X-Y)')
    plt.legend()
    
    # Plot joint actions
    plt.subplot(2, 2, 3)
    for i in range(start_idx, total_episodes):
        episode_idx = i - start_idx  # Relative index for legend
        episode_actions = results['actions'][i]
        time_steps = np.arange(len(episode_actions))
        
        # Plot the first 4 joint actions
        num_joints_to_plot = min(4, episode_actions.shape[1])
        for j in range(num_joints_to_plot):
            plt.plot(time_steps, episode_actions[:, j], 
                     label=f'Episode {i+1}, Joint {j+1}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Joint Action')
    plt.title('Joint Actions Over Time')
    plt.legend()
    
    # Plot some state variables over time
    plt.subplot(2, 2, 4)
    for i in range(start_idx, total_episodes):
        episode_idx = i - start_idx  # Relative index for legend
        episode_states = results['states'][i]
        time_steps = np.arange(len(episode_states))
        
        # Plot the first 4 state variables
        num_vars_to_plot = min(4, episode_states.shape[1])
        for j in range(num_vars_to_plot):
            plt.plot(time_steps, episode_states[:, j], 
                     label=f'Episode {i+1}, State {j+1}')
    
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('State Variables Over Time')
    plt.legend()
    
    # Print summary of the last three episodes
    print("\nDetails of the last episodes visualized:")
    for i in range(start_idx, total_episodes):
        print(f"Episode {i+1}/{total_episodes}: Reward = {results['rewards'][i]:.2f}, "
              f"Length = {len(results['states'][i])} steps")
    
    plt.tight_layout()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO model')
    # parser.add_argument('--model', type=str, default='models/ant-v4_ppo_model_20250422_105755.pth',
    #                     help='Path to the trained model file')
    parser.add_argument('--model', type=str, default='models/ant-v4_ppo_model_20250422_135650.pth',
                        help='Path to the trained model file')
    parser.add_argument('--info', type=str, default=None,
                        help='Path to the environment info file')
    parser.add_argument('--env', type=str, default='Ant-v4',
                        help='Environment name')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to evaluate (default: 1)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording (videos are recorded by default)')
    parser.add_argument('--video-path', type=str, default='videos',
                        help='Path to save videos')
    
    return parser.parse_args()

# Function to visualize a single episode's trajectory
def visualize_single_episode(results):
    """Visualize trajectory from a single evaluation episode"""
    
    # Only plot if we have state data
    if 'states' not in results or len(results['states']) == 0:
        print("No trajectory data available for visualization")
        return
    
    # Get the first (and only) episode data
    episode_states = results['states'][0]
    episode_actions = results['actions'][0]
    episode_reward = results['rewards'][0]
    time_steps = np.arange(len(episode_states))
    
    plt.figure(figsize=(15, 10))
    
    # Plot reward
    plt.subplot(2, 2, 1)
    plt.bar(['Episode 1'], [episode_reward])
    plt.ylabel('Total Reward')
    plt.title(f'Evaluation Reward: {episode_reward:.2f}')
    
    # Plot x-y position (trajectory)
    plt.subplot(2, 2, 2)
    if episode_states.shape[1] >= 2:  # Make sure we have at least x, y
        plt.plot(episode_states[:, 0], episode_states[:, 1])
        plt.scatter(episode_states[0, 0], episode_states[0, 1], color='green', s=100, marker='o', label='Start')
        plt.scatter(episode_states[-1, 0], episode_states[-1, 1], color='red', s=100, marker='x', label='End')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Agent Trajectory (X-Y)')
    plt.legend()
    
    # Plot joint actions
    plt.subplot(2, 2, 3)
    num_joints_to_plot = min(8, episode_actions.shape[1])
    for j in range(num_joints_to_plot):
        plt.plot(time_steps, episode_actions[:, j], label=f'Joint {j+1}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Joint Action')
    plt.title('Joint Actions Over Time')
    plt.legend()
    
    # Plot state variables
    plt.subplot(2, 2, 4)
    num_vars_to_plot = min(8, episode_states.shape[1])
    for j in range(num_vars_to_plot):
        plt.plot(time_steps, episode_states[:, j], label=f'State {j+1}')
    
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('State Variables Over Time')
    plt.legend()
    
    # Print details of the episode
    print(f"\nEpisode results: Reward = {episode_reward:.2f}, Length = {len(episode_states)} steps")
    
    plt.tight_layout()
    # Save the plot
    plot_path = "evaluation_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    print(f"Loading model from {args.model}")
    
    # Check if model file exists
    if not os.path.exists(args.model):
        # Look in models directory if path doesn't include it
        model_path = os.path.join('models', args.model)
        if os.path.exists(model_path):
            args.model = model_path
        else:
            print(f"Model file not found: {args.model}")
            model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
            if model_files:
                print("Available models:")
                for f in model_files:
                    print(f"  {f}")
            else:
                print("No model files found in 'models' directory")
            exit(1)
    
    # Load model and environment
    policy, env, env_info = load_model(args.model, args.info)
    
    # Create a video by default unless disabled
    if not args.no_video:
        print("Recording evaluation video...")
        video_path = create_video(args.env, args.model, args.info, video_path=args.video_path)
        print(f"Video saved to {video_path}")
    
    # Evaluate the model (single episode)
    print("\nRunning evaluation episode...")
    results = evaluate(policy, env, episodes=args.episodes, render=args.render)
    
    # Visualize the single episode
    visualize_single_episode(results) 