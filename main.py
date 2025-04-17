"""
Acrobot-v1 Reinforcement Learning Project

Author: Mohamad Hamadeh
License: MIT License
Date: 16/04/2025

Description:
Implementation of Q-Learning, SARSA, and DQN agents for the Acrobot-v1 environment using Gymnasium and PyTorch.

Acrobot-v1 Environment Details:
--------------------------------
Action Space: Discrete(3)
  - 0: apply -1 torque to the actuated joint
  - 1: apply 0 torque to the actuated joint
  - 2: apply 1 torque to the actuated joint

Observation Space: 
  Box([-1, -1, -1, -1, -12.566371, -28.274334],
      [1, 1, 1, 1, 12.566371, 28.274334], (6,), float32)
  (Observations include the cosine and sine of two joint angles, and the angular velocities.)

Reward Structure:
  - Each non-goal step yields a reward of -1.
  - Achieving the target height (i.e., when -cos(theta1) - cos(theta2 + theta1) > 1.0)
    terminates the episode with a reward of 0.
  - The reward threshold is -100.

Episode Termination:
  - The episode ends when the acrobot reaches the target height or when the episode exceeds 500 steps.

Note: The maximum possible cumulative reward per episode is 0, which would theoretically occur if the target is reached in a single step.

See the README.md for full documentation and usage instructions.
"""

import gymnasium as gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

# ---------------------- DQN Network ---------------------- #
# We define a neural network with two hidden layers (64 neurons each)
# that takes the 6-dimensional continuous state and outputs Q-values for 3 actions.
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action

# ---------------------- Replay Buffer ---------------------- #
# A simple experience replay buffer stores transitions and samples mini-batches for training.
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def store(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# ---------------------- Training Function (DQN) ---------------------- #
def train_dqn(env, online_net, target_net, optimizer, replay_buffer,
              train_episodes, batch_size, gamma,
              epsilon_start, epsilon_end, epsilon_decay,
              target_update_interval, train_start, train_freq, device):
    """
    Trains the DQN agent on the given environment.
    
    Parameters:
      env: The Acrobot-v1 environment.
      online_net: The network being trained.
      target_net: The target network (updated periodically).
      optimizer: The optimizer for online_net.
      replay_buffer: Buffer to store experiences.
      train_episodes: Total number of training episodes.
      batch_size: Minibatch size for training.
      gamma: Discount factor.
      epsilon_start: Initial exploration probability.
      epsilon_end: Minimum exploration probability.
      epsilon_decay: Number of steps over which to linearly decay epsilon.
      target_update_interval: Number of steps between target network updates.
      train_start: Number of transitions required in the buffer before training begins.
      train_freq: Frequency (in steps) to perform training.
      
    Returns:
      A list containing the cumulative reward for each episode.
    """
    steps = 0
    epsilon = epsilon_start
    episode_rewards = []
    
    for episode in tqdm(range(1, train_episodes + 1)):
        state, _ = env.reset()   # Gymnasium reset returns (observation, info)
        state = torch.tensor(state, dtype=torch.float32,device=device)
        done = False
        total_reward = 0
        
        while not done:
            steps += 1
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                with torch.no_grad():
                    q_values = online_net(state)
                    action = torch.argmax(q_values).item()  # Exploit
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
            
            # Store the transition in the replay buffer
            replay_buffer.store((state, action, reward, next_state_tensor, done))
            
            state = next_state_tensor
            
            # Linear decay of epsilon across steps
            epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)
            
            # Training step: only update if we have enough samples and at specified frequency
            if replay_buffer.size() >= train_start and steps % train_freq == 0:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.stack(states)
                actions = torch.tensor(actions, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.stack(next_states)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)
                
                # Compute Q-targets using the target network
                with torch.no_grad():
                    next_q_values = target_net(next_states)
                    max_next_q_values, _ = torch.max(next_q_values, dim=1)
                    targets = rewards + gamma * max_next_q_values * (1 - dones)
                
                # Current Q-value estimates for the actions taken
                current_q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                
                loss = nn.MSELoss()(current_q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Periodically update the target network with the latest online network weights
            if steps % target_update_interval == 0:
                target_net.load_state_dict(online_net.state_dict())
        
        episode_rewards.append(total_reward)
        
        if episode % 100 == 0:
            # Calculate mean reward over the last 100 episodes
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Epsilon: {epsilon:.3f}, Average Reward (last 100): {avg_reward:.2f}")
            
            # Graph mean rewards
            plot_rewards(episode_rewards, episode, "Mean Reward per train Episode (DQN)", "plots/Acrobot_DQN_Training.png")
    
    return episode_rewards

# ---------------------- Testing Function (DQN) ---------------------- #
def test_dqn(model, env_name, test_episodes, render_episodes, device):
    """
    Tests the trained DQN agent.
    
    Parameters:
      model: The trained DQN network.
      env_name: Name of the environment ("Acrobot-v1").
      test_episodes: Total number of test episodes.
      render_episodes: Number of test episodes for which rendering (visualization) is enabled.
    
    Returns:
      A list containing the cumulative reward for each test episode.
    """
    test_rewards = []
    for episode in range(1, test_episodes + 1):
        # For the first few episodes, enable rendering.
        if episode <= render_episodes:
            env = gym.make(env_name, render_mode="human")
        else:
            env = gym.make(env_name)
        
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        done = False
        total_reward = 0
        
        while not done:
            with torch.no_grad():
                action = torch.argmax(model(state)).item()  # Greedy selection (no exploration)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
        
        test_rewards.append(total_reward)
        
        if episode % 10 == 0:
            avg_last10 = np.mean(test_rewards[-10:])
            print(f"Test Episode {episode}: Total Reward = {total_reward}, Average Reward (last 10) = {avg_last10:.2f}")
            
            # Graph mean rewards
            plot_rewards(test_rewards, episode, 'Mean Reward Per test Episode (DQN)' ,'plots/Acrobot_Test_Rewards_DQN.png')
        
        env.close()
    return test_rewards

# ---------------------- Discretization Functions ---------------------- #
def create_boundaries(n_bins):
    """
    Create discretization boundaries for each dimension of the observation space.
    n_bins: number of bins per dimension (int)
    Returns a list of arrays containing bin edges for each dimension.
    """
    env = gym.make("Acrobot-v1")
    low = env.observation_space.low
    high = env.observation_space.high
    boundaries = [] # List of arrays containing bin edges for each dimension
    # For each dimension, we create n_bins-1 bin edges using linspace.
    for i in range(len(low)):
        boundaries.append(np.linspace(low[i], high[i], n_bins - 1))
    env.close()
    return boundaries

def discretize_state(state, boundaries):
    """
    Convert a continuous state into a discrete state tuple.
    state: the continuous state (array-like of 6 floats)
    boundaries: list of arrays with bin edges for each dimension.
    Returns a tuple of discrete indices.
    """
    discretized = []
    for i, value in enumerate(state):
        # np.digitize returns an index from 0 to len(boundaries[i])
        discretized.append(np.digitize(value, boundaries[i]))
    return tuple(discretized)

# Useful function if we want to read from a table that has been saved (maybe with more/less bins)
def get_qtable_shape(boundaries, num_actions):
    """
    Determine the shape of the Q-table based on discretization boundaries and number of actions.
    Each dimension has (number of boundaries + 1) discrete values.
    """
    dims = [len(b) + 1 for b in boundaries]
    dims.append(num_actions)
    return tuple(dims)

# ---------------------- Load and Save Table Functions ---------------------- #
def load_table(filename):
    """
    Load a Q-table from a file.
    filename: name of the file to load.
    Returns the loaded Q-table or None if file is not found.
    """
    if os.path.exists(filename):
        Q_table = np.load(filename) # Load as NumPy array
        print(f"Loaded Q-table from '{filename}'.")
        return Q_table
    else:
        print(f"File '{filename}' not found. Proceeding to train from scratch.")
        return None

def save_table(Q_table, filename):
    """
    Save a Q-table to a file.
    Q_table: the Q-table to save.
    filename: name of the file to save.
    """
    np.save(filename, Q_table)

# ---------------------- Plotting Function ---------------------- #
def plot_rewards(rewards_per_episode, episodes, title, save_path):
    """
    Plot the rewards per episode.
    rewards_per_episode: list of rewards for each episode.
    episodes: current episode count (for plotting x-axis)
    save_path: path to save the plot.
    """
    mean_rewards = []
    for t in range(episodes):
        # Calculate mean reward over the last 100 episodes.
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.figure()
    plt.plot(mean_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

# ---------------------- Action Selection ---------------------- #
def choose_action(state, Q_table, epsilon, num_actions):
    """
    Select an action using the epsilon-greedy policy.
    With probability epsilon, choose a random action.
    Otherwise, choose the action with the highest Q-value for the given state.
    """
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    else:
        return np.argmax(Q_table[state])

# ---------------------- Training Function (Q-Learning) ---------------------- #
def train_agent_QL(Q_table, boundaries, env, num_episodes, alpha, gamma, epsilon, min_epsilon, epsilon_decay):
    """
    Train the Q-learning agent.
    - Q_table: the initialized Q-table.
    - boundaries: discretization boundaries.
    - env: the Acrobot-v1 environment.
    - num_episodes: number of training episodes.
    - alpha: learning rate.
    - gamma: discount factor.
    - epsilon: initial exploration rate.
    - min_epsilon: minimum exploration rate.
    - epsilon_decay: multiplicative decay factor for epsilon after each episode.
    Prints the cumulative reward every 100 episodes.
    Returns the updated Q_table.
    """
    # Track rewards for mean calculation
    total_rewards = []
    best_reward = -500  # Initialize best reward to the worst possible value
    for episode in range(1, num_episodes + 1):
        state_cont, _ = env.reset()  # Gymnasium reset returns (observation, info)
        state = discretize_state(state_cont, boundaries)
        done = False
        rewards = 0

        while not done:
            action = choose_action(state, Q_table, epsilon, env.action_space.n)
            next_state_cont, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward

            next_state = discretize_state(next_state_cont, boundaries)
            # Q-Learning update rule:
            # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
            best_next = np.max(Q_table[next_state])
            Q_table[state][action] += alpha * (reward + gamma * best_next - Q_table[state][action])

            state = next_state

        if rewards > best_reward:
            best_reward = rewards
            # Save Q-table to file on new best reward (optional, this is just incase the training is interrupted)
            save_table(Q_table, "Acrobot_Tabular_QLearning.npy")

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Store total reward for this episode
        total_rewards.append(rewards)
        
        if episode % 100 == 0:
            # Calculate mean reward over the last 100 episodes
            mean_reward = np.mean(total_rewards[len(total_rewards) - 100:])
            print(f"Episode: {episode}, Epsilon: {epsilon:0.2f}, Best Reward so far = {best_reward}, Mean Reward (last 100 episodes) = {mean_reward:.2f}")
            
            # Graph mean rewards
            plot_rewards(total_rewards, episode, "Mean Reward per train Episode (Q-Learning)", "plots/Acrobot_Tabular_QLearning_Training.png")

    return Q_table

# ---------------------- Training Function (SARSA) ---------------------- #
def train_agent_SARSA(Q_table, boundaries, env, num_episodes, alpha, gamma, epsilon, min_epsilon, epsilon_decay):
    """
    Train the agent using the SARSA algorithm.
    - Q_table: the initialized Q-table.
    - boundaries: discretization boundaries.
    - env: the Acrobot-v1 environment.
    - num_episodes: number of training episodes.
    - alpha: learning rate.
    - gamma: discount factor.
    - epsilon: initial exploration rate.
    - min_epsilon: minimum exploration rate.
    - epsilon_decay: multiplicative decay factor for epsilon after each episode.
    
    The SARSA update rule is:
    Q(s, a) = Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))
    
    Cumulative reward is printed every 100 episodes.
    The Q-table is saved to disk when a new best cumulative reward is achieved.
    """
    total_rewards = []
    best_reward = -500  # Worst-case reward if target is never reached in 500 steps
    num_actions = env.action_space.n
    
    for episode in range(1, num_episodes + 1):
        state_cont, _ = env.reset()  # Gymnasium reset returns (observation, info)
        state = discretize_state(state_cont, boundaries)
        done = False
        rewards = 0
        
        # Choose initial action using epsilon-greedy policy
        action = choose_action(state, Q_table, epsilon, num_actions)
        
        while not done:
            next_state_cont, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
            next_state = discretize_state(next_state_cont, boundaries)
            
            # SARSA update: if not done, choose next action; else use 0 for Q-value
            if not done:
                next_action = choose_action(next_state, Q_table, epsilon, num_actions)
                target = reward + gamma * Q_table[next_state][next_action]
            else:
                target = reward  # Terminal state: no future Q-value
            Q_table[state][action] += alpha * (target - Q_table[state][action])
            
            # Transition to next state and action
            state, action = next_state, next_action if not done else None
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_rewards.append(rewards)
        
        # Save table if a new best reward is achieved
        if rewards > best_reward:
            best_reward = rewards
            save_table(Q_table, "Acrobot_Tabular_SARSA.npy")
        
        if episode % 100 == 0:
            mean_reward = np.mean(total_rewards[-100:])
            print(f"Episode: {episode}, Epsilon: {epsilon:0.2f}, Best Reward so far = {best_reward}, Mean Reward (last 100 episodes) = {mean_reward:.2f}")
            
            # Graph mean rewards
            plot_rewards(total_rewards, episode, "Mean Reward per train Episode (SARSA)", "plots/Acrobot_Tabular_SARSA_Training.png")
    
    return Q_table

# ---------------------- Testing Function (Q-Learning & SARSA)---------------------- #
def test_agent(Q_table, boundaries, env, num_episodes, title, save_path, render_episodes=5):
    """
    Test the trained agent.
    - For the first 'render_episodes' episodes, rendering is enabled.
    - The agent uses a greedy policy (choosing the action with the highest Q-value).
    - Prints cumulative reward every 10 episodes.
    Returns a list of cumulative rewards for all test episodes.
    """
    test_rewards = []
    
    for episode in range(1, num_episodes + 1):
        # Enable visualization for the first few episodes.
        if episode <= render_episodes:
            env = gym.make("Acrobot-v1", render_mode="human")
        else:
            env = gym.make("Acrobot-v1")
            
        state_cont, _ = env.reset()
        state = discretize_state(state_cont, boundaries)
        done = False
        rewards = 0
        
        while not done:
            # Choose the best action from the Q-table (greedy policy)
            action = np.argmax(Q_table[state])
            next_state_cont, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
            state = discretize_state(next_state_cont, boundaries)

        test_rewards.append(rewards)
        
        if episode % 10 == 0:
            best_reward = np.max(test_rewards[-10:])
            print(f"Test Episode {episode}: Total Reward = {rewards}, Best Reward (last 10 episodes) = {best_reward}")
            
            # Graph mean rewards
            plot_rewards(test_rewards, episode, title, save_path)

    return test_rewards

def main():
    # ------------------ User Input ------------------ #
    print("Select a method to run:")
    print("1: Q-Learning")
    print("2: SARSA")
    print("3: DQN")
    method_choice = input("Enter 1, 2, or 3: ").strip()

    print("Do you want to train from scratch or use an existing model/table?")
    print("1: Train from scratch")
    print("2: Use existing")
    train_choice = input("Enter 1 or 2: ").strip()

    if train_choice == "1":
        train_episodes = int(input("Enter the number of training episodes: "))
    else:
        train_episodes = None  # We will load the model/table

    test_episodes = int(input("Enter the number of test episodes: "))
    render_episodes = int(input("Enter the number of test episodes to render (eg. 5 means first 5 will render): "))

    # ------------------ Common Environment Setup ------------------ #
    env_name = "Acrobot-v1"
    env = gym.make(env_name)

    # ------------------ Execute Selected Method ------------------ #
    if method_choice == "1":
        # ---------- Q-Learning Settings ---------- #
        # Hyperparameters for tabular Q-Learning
        ALPHA = 0.1                 # Learning rate
        GAMMA = 0.99                # Discount factor
        INITIAL_EPSILON = 1.0       # Initial exploration rate
        MIN_EPSILON = 0.01          # Minimum exploration rate
        EPSILON_DECAY = 0.9999      # multiplicative decay factor per episode
        BINS_PER_DIMENSION = 16     # Number of bins per observation dimension

        q_table_filename = "Acrobot_Tabular_QLearning.npy"
        boundaries = create_boundaries(BINS_PER_DIMENSION)          # Create discretization boundaries based on the observation space
        Q_shape = get_qtable_shape(boundaries, env.action_space.n)  # Determine the shape of the Q-table.
        Q_table = None

        print("\nRunning Q-Learning for Acrobot-v1...")
        if train_choice == "2":
            Q_table = load_table(q_table_filename)
        if Q_table is None:
            # Initialize Q-table with zeros.
            Q_table = np.zeros(Q_shape)
            print("Starting training phase for Q-Learning...")
            Q_table = train_agent_QL(Q_table, boundaries, env, train_episodes,
                                     ALPHA, GAMMA, INITIAL_EPSILON, MIN_EPSILON, EPSILON_DECAY)
            save_table(Q_table, q_table_filename)
            print(f"Training complete. Q-table saved as '{q_table_filename}'.")
        else:
            print("Using existing Q-table.")

        print("Starting testing phase for Q-Learning...")
        test_rewards = test_agent(Q_table, boundaries, env, test_episodes,
                                  "Mean Reward per test Episode (Q-Learning)", "plots/Acrobot_Tabular_QLearning_Test.png",
                                  render_episodes)
        avg_reward = np.mean(test_rewards)       
        # Compute performance percentage.
        # Worst-case reward: -500 (if the acrobot never reaches the target in 500 steps)
        # Best-case reward: 0 (maximum possible cumulative reward is 0)
        # We define:
        #   performance_percentage = ((avg_reward - (-500)) / (0 - (-500)))*100
        # This means that 100% corresponds to perfect performance (0 in every episode)
        # while 0% corresponds to the worst performance (-500 in every episode).
        performance_percentage = ((avg_reward + 500) / 500) * 100
    
        print(f"\nAverage Test Reward over {test_episodes} episodes: {avg_reward:.2f}")
        print(f"Performance Percentage: {performance_percentage:.2f}%")
        print("Note: 100% means the agent achieved the maximum reward (0) in every episode,")
        print("while 0% means the agent got the worst reward (-500) in every episode.")

    elif method_choice == "2":
        # ---------- SARSA Settings ---------- #
        # Hyperparameters for SARSA
        ALPHA = 0.1
        GAMMA = 0.99
        INITIAL_EPSILON = 1.0
        MIN_EPSILON = 0.01
        EPSILON_DECAY = 0.9999
        BINS_PER_DIMENSION = 16

        q_table_filename = "Acrobot_Tabular_SARSA.npy"
        boundaries = create_boundaries(BINS_PER_DIMENSION)
        Q_table = None

        print("\nRunning SARSA for Acrobot-v1...")
        if train_choice == "2":
            Q_table = load_table(q_table_filename)
        if Q_table is None:
            # Initialize Q-table with zeros.
            Q_table = np.zeros(get_qtable_shape(boundaries, env.action_space.n))
            print("Starting training phase for SARSA...")
            Q_table = train_agent_SARSA(Q_table, boundaries, env, train_episodes,
                                        ALPHA, GAMMA, INITIAL_EPSILON, MIN_EPSILON, EPSILON_DECAY)
            save_table(Q_table, q_table_filename)
            print(f"Training complete. Q-table saved as '{q_table_filename}'.")
        else:
            print("Using existing Q-table.")

        print("Starting testing phase for SARSA...")
        test_rewards = test_agent(Q_table, boundaries, env, test_episodes,
                                  "Mean Reward per test Episode (SARSA)", "plots/Acrobot_Tabular_SARSA_Test.png",
                                  render_episodes)
        
        avg_reward = np.mean(test_rewards)
        performance_percentage = ((avg_reward + 500) / 500) * 100
        
        print(f"\nAverage Test Reward over {test_episodes} episodes: {avg_reward:.2f}")
        print(f"Performance Percentage: {performance_percentage:.2f}%")
        print("Note: 100% means the agent achieved the maximum reward (0) in every episode,")
        print("while 0% means the agent got the worst reward (-500) in every episode.")

    elif method_choice == "3":
        # ---------- DQN Settings ---------- #
        # Hyperparameters for DQN
        GAMMA = 0.99
        LEARNING_RATE = 1e-3
        BUFFER_CAPACITY = 50000         # Replay buffer capacity
        BATCH_SIZE = 64
        EPSILON_START = 1.0
        EPSILON_END = 0.02
        EPSILON_DECAY = 30000           # Total steps over which epsilon decays linearly
        TARGET_UPDATE_INTERVAL = 500    # Steps between target network updates
        TRAIN_START = 1000              # Minimum number of transitions in buffer before training starts
        TRAIN_FREQ = 4                  # Training update frequency (every n steps)

        MODEL_PATH = "Acrobot_DQN.pth"

        # Setup device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() and input('GPU is available. Use it? (y/n) ').strip().lower() == 'y' else "cpu")
        print(f"Using device: {device.type.upper()}")

        state_dim = env.observation_space.shape[0]  # For Acrobot-v1, this is 6.
        action_dim = env.action_space.n             # For Acrobot-v1, there are 3 discrete actions.

        # Initialize networks
        online_net = DQNNetwork(state_dim, action_dim).to(device)
        target_net = DQNNetwork(state_dim, action_dim).to(device)
        target_net.load_state_dict(online_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
        replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

        print("\nRunning DQN for Acrobot-v1...")
        if train_choice == "2" and os.path.exists(MODEL_PATH):
            online_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            target_net.load_state_dict(online_net.state_dict())
            print(f"Loaded model from '{MODEL_PATH}'.")
        else:
            print("Starting training phase for DQN...")
            train_dqn(env, online_net, target_net, optimizer, replay_buffer,
                      train_episodes, BATCH_SIZE, GAMMA,
                      EPSILON_START, EPSILON_END, EPSILON_DECAY,
                      TARGET_UPDATE_INTERVAL, TRAIN_START, TRAIN_FREQ, device)
            torch.save(online_net.state_dict(), MODEL_PATH)
            print(f"Training complete. Model saved to '{MODEL_PATH}'.")

        print("Starting testing phase for DQN...")
        test_rewards = test_dqn(online_net, env_name, test_episodes, render_episodes, device)
        avg_reward = np.mean(test_rewards)
        performance_percentage = ((avg_reward + 500) / 500) * 100
        print(f"\nAverage Test Reward over {test_episodes} episodes: {avg_reward:.2f}")
        print(f"Performance Percentage: {performance_percentage:.2f}%")
        print("Note: 100% means the agent achieved the maximum reward (0) in every episode,")
        print("while 0% means the agent got the worst reward (-500) in every episode.")

    else:
        print("Invalid selection. Exiting.")

    env.close()

if __name__ == "__main__":
    main()
