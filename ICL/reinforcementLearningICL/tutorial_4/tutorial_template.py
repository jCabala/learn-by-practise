"""
Tutorial 4: Value Function Approximation with Deep Q-Networks
Template File for Students

This file contains skeleton code for implementing linear Q-learning and DQN
on the CartPole environment. Complete the sections marked with TODO.
"""

from collections import deque
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# Part 1: Linear Function Approximation
# ============================================================================


def create_features(state: np.ndarray) -> np.ndarray:
    """
    Create polynomial features from the CartPole state.

    CartPole state: [position, velocity, angle, angular_velocity]

    Args:
        state: Raw state from environment (4 dimensions)

    Returns:
        features: Feature vector for linear approximation

    TODO: Implement feature creation
    - Include bias term (1.0)
    - Include original state variables
    - Include polynomial features (squares and cross terms)
    """

    # Normalize state for better numerical stability
    # CartPole bounds: position ±2.4, angle ±0.2095 radians
    normalized_state = state.copy()
    normalized_state[0] = state[0] / 2.4  # position
    normalized_state[2] = state[2] / 0.2095  # angle

    # Create polynomial features up to degree 2
    x = normalized_state

    # ============================================
    # TODO: Implement feature creation
    # ============================================

    # Hint: For a simple start, you can use:
    # - 1 (bias)
    # - state[0], state[1], state[2], state[3] (original features)
    # - state[0]**2, state[2]**2 (squared position and angle)
    #
    # More advanced: include all degree-2 polynomial features
    # Dont stress too much about this part, there are many ways to do it!

    original_features = np.array([
        1.0,  # bias
        x[0],
        x[1],
        x[2],
        x[3],  # linear terms
    ])

    squared_features = np.array([x[i] * x[j] for i in range(len(x)) for j in range(i, len(x))])

    return np.concatenate([original_features, squared_features])


class LinearQNetwork:
    """
    Linear Q-function approximation: Q(s,a) = w_a^T * phi(s)

    Each action has its own weight vector.
    """

    def __init__(self, feature_dim: int, num_actions: int, learning_rate: float = 0.01):
        """
        Initialize linear Q-network.

        Args:
            feature_dim: Dimension of feature vector
            num_actions: Number of actions
            learning_rate: Learning rate for weight updates
        """
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        # ============================================
        # TODO: Initialize weights for each action
        # ============================================

        # Initialize weights to small random values
        self.weights = np.random.randn(num_actions, feature_dim) * 0.01
        # Hint: Create a matrix of shape (num_actions, feature_dim)
        # Use small random initialization, e.g., np.random.randn() * 0.01

    def get_q_values(self, features: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for all actions given features.

        Args:
            features: Feature vector phi(s)

        Returns:
            q_values: Array of Q-values for each action
        """
        # ============================================
        # TODO: Compute Q(s,a) = w_a^T * phi(s) for all actions
        # ============================================

        q_values = np.dot(self.weights, features)
        # Hint: This is just matrix-vector multiplication

        return q_values

    def update(
        self,
        features: np.ndarray,
        action: int,
        target: float,
        current_q: float,
    ):
        """
        Perform semi-gradient TD update.

        Update rule: w_a <- w_a + alpha * [target - Q(s,a)] * phi(s)

        Args:
            features: Feature vector phi(s)
            action: Action taken
            target: TD target (r + gamma * max_a' Q(s', a'))
            current_q: Current Q-value Q(s,a)
        """
        # ============================================
        # TODO: Implement semi-gradient TD update
        # ============================================

        # Compute TD error
        td_error = target - current_q

        # Update weights for the chosen action
        self.weights[action] += self.learning_rate * td_error * features


def train_linear_q_learning(
    num_episodes: int = 1000,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_steps: float = 1000,
    render_final: bool = False,
):
    """
    Train a linear Q-learning agent on CartPole.

    Args:
        num_episodes: Number of episodes to train
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay_steps: Number of steps to decay epsilon
        render_final: Whether to render final policy
    """
    # Create environment
    env = gym.make("CartPole-v1")

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Create sample features to get feature dimension
    sample_state = env.reset()[0]
    sample_features = create_features(sample_state)
    feature_dim = len(sample_features)

    print(f"State dimension: {state_dim}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of actions: {num_actions}")

    # Initialize Q-network
    q_network = LinearQNetwork(feature_dim, num_actions, learning_rate=0.01)

    # Training loop
    episode_rewards = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # ============================================
            # TODO: Implement epsilon-greedy action selection
            # ============================================

            features = create_features(state)
            q_values = q_network.get_q_values(features)

            # Select action using epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(q_values)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # ============================================
            # TODO: Implement Q-learning update
            # ============================================

            # Compute TD target
            next_features = create_features(next_state)
            next_q_values = q_network.get_q_values(next_features)

            if done:
                target = reward  # Terminal state
            else:
                target = reward + gamma * np.max(next_q_values)  # REPLACE: add discounted max future Q-value

            # Get current Q-value
            current_q = q_values[action]

            # Update Q-network
            q_network.update(features, action, target, current_q)

            state = next_state
            episode_reward += reward

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_start * (1 - episode / epsilon_decay_steps))

        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Epsilon: {epsilon:.3f}"
            )

    env.close()

    # Plot results
    plot_training_results(episode_rewards, "Linear Q-Learning")

    # Test final policy
    if render_final:
        test_policy(q_network, is_linear=True)

    return q_network, episode_rewards


# ============================================================================
# Part 2: Deep Q-Network (DQN)
# ============================================================================


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions and allows sampling random mini-batches.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # ============================================
        # TODO: Add transition to buffer
        # ============================================

        # Hint: Store as tuple (state, action, reward, next_state, done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random mini-batch from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of batched (states, actions, rewards, next_states, dones)
        """
        # ============================================
        # TODO: Sample random batch and convert to tensors
        # ============================================

        # Hint: Use random.sample() to sample from buffer
        # Then convert each component to appropriate torch tensor

        batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]

        # Unpack batch into separate arrays
        states = torch.tensor([item[0] for item in batch], dtype=torch.float32)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        next_states = torch.tensor([item[3] for item in batch], dtype=torch.float32)
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Neural network for Q-function approximation.

    Architecture:
    - Input: state vector
    - Hidden: 2 layers with 128 units each, ReLU activation
    - Output: Q-values for each action
    """

    def __init__(self, state_dim: int, num_actions: int):
        """
        Initialize Q-network.

        Args:
            state_dim: Dimension of state space
            num_actions: Number of actions
        """
        super(QNetwork, self).__init__()

        # ============================================
        # TODO: Define network architecture
        # ============================================

        # Hint: Use nn.Sequential or define layers separately
        # Example architecture:
        # - Linear(state_dim, 128)
        # - ReLU
        # - Linear(128, 128)
        # - ReLU
        # - Linear(128, num_actions)

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Q-values for each action (batch_size, num_actions)
        """
        # ============================================
        # TODO: Implement forward pass
        # ============================================

        return self.network(state)


def train_dqn(
    num_episodes: int = 1000,
    batch_size: int = 64,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_steps: float = 1000,
    learning_rate: float = 0.0005,
    target_update_freq: int = 100,
    buffer_capacity: int = 100000,
    render_final: bool = False,
):
    """
    Train a DQN agent on CartPole.

    Args:
        num_episodes: Number of episodes to train
        batch_size: Mini-batch size for updates
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay_steps: Number of steps to decay epsilon
        learning_rate: Learning rate for Adam optimizer
        target_update_freq: Frequency of target network updates (steps)
        buffer_capacity: Replay buffer capacity
        render_final: Whether to render final policy
    """
    # Create environment
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    print(f"State dimension: {state_dim}")
    print(f"Number of actions: {num_actions}")

    # ============================================
    # TODO: Initialize networks, optimizer, and replay buffer
    # ============================================

    # Create Q-network and target network
    q_network = QNetwork(state_dim, num_actions)
    target_network = QNetwork(state_dim, num_actions)

    # Copy weights to target network
    target_network.load_state_dict(q_network.state_dict())

    # Create optimizer
    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # Training loop
    episode_rewards = []
    epsilon = epsilon_start
    step_count = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # ============================================
            # TODO: Select action using epsilon-greedy
            # ============================================

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                # Get Q-values from network
                with torch.no_grad():
                    q_values = q_network.forward(state_tensor)

                # Select greedy action
                action = q_values.argmax().item()

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Add to replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            # ============================================
            # TODO: Train if buffer has enough samples
            # ============================================

            if len(replay_buffer) >= batch_size:
                # Sample mini-batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(
                    batch_size
                )

                # Compute current Q-values
                current_q_values = q_network.forward(states)

                # Get Q-values for actions that were taken
                current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute target Q-values using target network
                with torch.no_grad():
                    next_q_values = target_network.forward(next_states)
                    max_next_q = next_q_values.max(1)[0]

                    # TD target: r + gamma * max_a' Q(s', a') (or just r if done)
                    target_q = rewards + (gamma * max_next_q * (1 - dones))

                    # Hint: rewards + gamma * max_next_q * (1 - dones)

                # Compute loss
                loss = nn.functional.mse_loss(current_q, target_q)

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network periodically
            step_count += 1
            if step_count % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())

            state = next_state
            episode_reward += reward

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_start * (1 - episode / epsilon_decay_steps))

        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Epsilon: {epsilon:.3f}, "
                f"Buffer Size: {len(replay_buffer)}"
            )

    env.close()

    # Plot results
    plot_training_results(episode_rewards, "Deep Q-Network (DQN)")

    # Test final policy
    if render_final:
        test_policy(q_network, is_linear=False)

    return q_network, episode_rewards


# ============================================================================
# Utility Functions
# ============================================================================


def plot_training_results(rewards: List[float], title: str = "Training Results"):
    """
    Plot training rewards over episodes.

    Args:
        rewards: List of episode rewards
        title: Plot title
    """
    plt.figure(figsize=(10, 5))

    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3)
    plt.title(f"{title} - Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    # Plot moving average
    plt.subplot(1, 2, 2)
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(moving_avg)
    plt.title(f"{title} - Moving Average (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=195, color="r", linestyle="--", label="Solved Threshold")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_results.png", dpi=150)
    print(f"Saved plot: {title.lower().replace(' ', '_')}_results.png")
    plt.close()


def test_policy(agent, is_linear: bool = False, num_episodes: int = 1):
    """
    Test the learned policy.

    Args:
        agent: Trained agent (LinearQNetwork or QNetwork)
        is_linear: Whether agent is linear (True) or neural network (False)
        num_episodes: Number of test episodes
    """
    env = gym.make("CartPole-v1", render_mode="human")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if is_linear:
                features = create_features(state)
                q_values = agent.get_q_values(features)
                action = np.argmax(q_values)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = agent(state_tensor)
                action = q_values.argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        print(f"Test Episode {episode + 1}: Reward = {episode_reward}")

    env.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Tutorial 4: Value Function Approximation")
    print("=" * 70)

    # Part 1: Linear Q-Learning
    print("\n" + "=" * 70)
    print("Part 1: Training Linear Q-Learning Agent")
    print("=" * 70)
    linear_agent, linear_rewards = train_linear_q_learning(
        num_episodes=1000,
        render_final=False,  # Set to True to see final policy
    )

    # Part 2: Deep Q-Network
    print("\n" + "=" * 70)
    print("Part 2: Training Deep Q-Network (DQN) Agent")
    print("=" * 70)
    dqn_agent, dqn_rewards = train_dqn(
        num_episodes=1000,
        render_final=True,  # Set to True to see final policy
    )
