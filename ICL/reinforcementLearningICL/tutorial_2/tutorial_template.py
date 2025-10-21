"""
Tutorial 2: Monte Carlo Methods
Template File for Students

This file contains skeleton code for implementing Monte Carlo policy evaluation
and control algorithms. Complete the sections marked with TODO.
"""

from turtle import done
from typing import Dict, List, Tuple

import numpy as np
from grid_world import Action, GridWorld


class MonteCarloAgent:
    """
    Monte Carlo methods for learning from experience.

    This class implements first-visit MC evaluation and on-policy MC control.
    """

    def __init__(self, env: GridWorld, gamma: float = 0.9, epsilon: float = 0.1):
        """
        Initialize the MC agent.

        Args:
            env: GridWorld environment
            gamma: Discount factor (0 <= gamma < 1)
            epsilon: Exploration parameter for epsilon-greedy policies
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.states = env.get_state_space()

        # Initialize Q-values
        self.Q = {}
        for state in self.states:
            for action in env.get_valid_actions(state):
                self.Q[(state, action)] = 0.0

        # For tracking returns (used in first-visit MC)
        self.returns = {key: [] for key in self.Q.keys()}

    def generate_episode(
        self, policy: Dict[Tuple[int, int], Action], max_steps: int = 1000
    ) -> List[Tuple[Tuple[int, int], Action, float]]:
        """
        Generate an episode following the given policy.

        Args:
            policy: Dictionary mapping states to actions
            max_steps: Maximum steps to prevent infinite loops

        Returns:
            episode: List of (state, action, reward) tuples
        """
        # ============================================
        # TODO: Implement episode generation
        # ============================================

        episode = []
        state = self.env.reset() # Resets environment and returns initial state

        for _ in range(max_steps):
            # TODO: Check if current state is terminal
            # Hint: Check if state is goal or trap
            if state == self.env.goal_pos or state in self.env.traps:
                break

            # TODO: Get action from policy for current state
            # Hint: Use policy[state] if state in policy, else choose randomly
            action = policy.get(state, None)
            if action is None:
                valid_actions = self.env.get_valid_actions(state)
                if not valid_actions:
                    break
                action = np.random.choice(valid_actions)

            # TODO: Take action in environment
            # Hint: Use self.env.step(action)
            next_state, reward, done = self.env.step(action)

            # TODO: Store (state, action, reward) in episode
            episode.append((state, action, reward))

            # TODO: Update state and check if done
            state = next_state

        return episode

    def first_visit_mc_evaluation(
        self,
        policy: Dict[Tuple[int, int], Action],
        num_episodes: int = 5000,
    ) -> Dict[Tuple[Tuple[int, int], Action], float]:
        """
        Estimate Q^pi(s,a) using first-visit Monte Carlo.

        Args:
            policy: Policy to evaluate
            num_episodes: Number of episodes to generate

        Returns:
            Q: Dictionary mapping (state, action) pairs to Q-values
        """
        # ============================================
        # TODO: Implement first-visit MC evaluation
        # ============================================

        for episode_num in range(num_episodes):
            # TODO: Generate an episode following the policy
            episode = self.generate_episode(policy)


            # TODO: Calculate returns for each step
            # Hint: Work backward from the end of the episode

            # Track which (state, action) pairs we've already seen in this episode
            visited = set()
            G = 0.0
            # Go through episode backwards
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
               
                # TODO: Update the return
                # Hint: G = reward + gamma * G
                G = reward + self.gamma * G

                # TODO: Check if this is the first visit to (state, action)
                # Hint: Check if (state, action) is not in visited set
                if (state, action) not in visited:
                    # TODO: Add (state, action) to visited set
                    visited.add((state, action))
                    # TODO: Append G to returns[(state, action)]
                    self.returns[(state, action)].append(G)
                    # TODO: Update Q[(state, action)] to be the average of all returns
                    # Hint:
                    # Q[(state, action)] = average of self.returns[(state, action)]
                    self.Q[(state, action)] = np.mean(self.returns[(state, action)])

            # Print progress
            if (episode_num + 1) % 1000 == 0:
                print(f"Completed {episode_num + 1}/{num_episodes} episodes")

        return self.Q

    def epsilon_greedy_policy(self, state: Tuple[int, int]) -> Dict[Action, float]:
        """
        Create an epsilon-greedy policy for a given state.

        Args:
            state: Current state

        Returns:
            policy_probs: Dictionary mapping actions to probabilities
        """
        # ============================================
        # TODO: Implement epsilon-greedy policy
        # ============================================

        valid_actions = self.env.get_valid_actions(state)
        num_actions = len(valid_actions)

        if num_actions == 0:
            return {}

        # Find the greedy action (action with highest Q-value)
        best_action = None
        best_value = float("-inf")

        for action in valid_actions:
            # TODO: Check if Q[(state, action)] > best_value
            # Update best_action and best_value if so
            if (state, action) in self.Q:
                if self.Q[(state, action)] > best_value:
                    best_value = self.Q[(state, action)]
                    best_action = action

        # TODO: Create epsilon-greedy policy
        # Hint: Greedy action gets probability (1 - epsilon + epsilon/num_actions)
        #       Other actions get probability (epsilon/num_actions)
        policy_probs = {}
        
        for action in valid_actions:
            if action == best_action:
                policy_probs[action] = 1 - self.epsilon + (self.epsilon / num_actions)
            else:
                policy_probs[action] = self.epsilon / num_actions

        return policy_probs

    def select_action(
        self, state: Tuple[int, int], policy_probs: Dict[Action, float]
    ) -> Action:
        """
        Select an action according to a stochastic policy.

        Args:
            state: Current state
            policy_probs: Dictionary mapping actions to probabilities

        Returns:
            action: Selected action
        """
        actions = list(policy_probs.keys())
        probs = list(policy_probs.values())

        # Normalize probabilities (in case of floating point errors)
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(actions)] * len(actions)

        return np.random.choice(actions, p=probs)

    def generate_episode_stochastic(
        self, max_steps: int = 1000
    ) -> List[Tuple[Tuple[int, int], Action, float]]:
        """
        Generate an episode following the current epsilon-greedy policy.

        Args:
            max_steps: Maximum steps to prevent infinite loops

        Returns:
            episode: List of (state, action, reward) tuples
        """
        episode = []
        state = self.env.reset()

        for _ in range(max_steps):
            if state == self.env.goal_pos or state in self.env.traps:
                break

            # Get epsilon-greedy policy for current state
            policy_probs = self.epsilon_greedy_policy(state)

            if not policy_probs:
                break

            # Select action according to policy
            action = self.select_action(state, policy_probs)

            # Take action
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))

            state = next_state

            if done:
                break

        return episode

    def on_policy_mc_control(
        self,
        num_episodes: int = 10000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
    ) -> Dict[Tuple[Tuple[int, int], Action], float]:
        """
        On-policy Monte Carlo control with epsilon-greedy exploration.

        Args:
            num_episodes: Number of episodes to train
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon

        Returns:
            Q: Learned Q-values
        """
        # ============================================
        # TODO: Implement on-policy MC control
        # ============================================

        self.epsilon = epsilon_start

        for episode_num in range(num_episodes):
            # TODO: Generate episode using current epsilon-greedy policy
            episode = self.generate_episode_stochastic()

            # TODO: Update Q-values using first-visit MC
            # Hint: Similar to first_visit_mc_evaluation, but update after each episode
            G = 0.0
            visited = set()

            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]

                # TODO: Update return
                G = reward + self.gamma * G

                # TODO: First-visit check and update
                if (state, action) not in visited:
                    visited.add((state, action))

                    # TODO: Incremental update for Q-value
                    # Hint: Append G to returns and take average
                    self.returns[(state, action)].append(G)
                    self.Q[(state, action)] = np.mean(self.returns[(state, action)])

            # TODO: Decay epsilon
            # Hint: epsilon = max(epsilon_end, epsilon * epsilon_decay)
            self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)

            # Print progress
            if (episode_num + 1) % 2000 == 0:
                print(
                    f"Episode {episode_num + 1}/{num_episodes}, "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        return self.Q

    def extract_policy(self) -> Dict[Tuple[int, int], Action]:
        """
        Extract greedy policy from Q-values.

        Returns:
            policy: Dictionary mapping states to best actions
        """
        policy = {}

        for state in self.states:
            if state == self.env.goal_pos or state in self.env.traps:
                continue

            valid_actions = self.env.get_valid_actions(state)
            if not valid_actions:
                continue

            best_action = None
            best_value = float("-inf")

            for action in valid_actions:
                if (state, action) in self.Q:
                    if self.Q[(state, action)] > best_value:
                        best_value = self.Q[(state, action)]
                        best_action = action

            if best_action is not None:
                policy[state] = best_action

        return policy


def test_mc_evaluation():
    """
    Test Monte Carlo policy evaluation on a simple maze.
    """
    from grid_world import create_simple_maze

    print("=" * 60)
    print("Testing Monte Carlo Policy Evaluation")
    print("=" * 60)

    # Create environment
    env = create_simple_maze()

    # Create a simple policy (go right and down toward goal)
    policy = {}
    for state in env.get_state_space():
        if state == env.goal_pos or state in env.traps:
            continue
        # Simple heuristic: go right if possible, else down, else any valid action
        valid_actions = env.get_valid_actions(state)
        if Action.RIGHT in valid_actions:
            policy[state] = Action.RIGHT
        elif Action.DOWN in valid_actions:
            policy[state] = Action.DOWN
        elif valid_actions:
            policy[state] = valid_actions[0]

    # Create MC agent
    agent = MonteCarloAgent(env, gamma=0.9)

    # Evaluate policy
    print("\nEvaluating policy with MC...")
    Q = agent.first_visit_mc_evaluation(policy, num_episodes=5000)

    # Visualize Q-values
    print("\nGenerating Q-value visualization...")
    env.render_q_values(
        Q, title="Q-Values (MC Evaluation)", save_path="mc_q_values_eval.png"
    )
    print("Saved: mc_q_values_eval.png")

    print("\n" + "=" * 60)


def test_on_policy_control():
    """
    Test on-policy MC control on a simple maze.
    """
    from grid_world import create_simple_maze

    print("=" * 60)
    print("Testing On-Policy Monte Carlo Control")
    print("=" * 60)

    # Create environment
    env = create_simple_maze()

    # Create MC agent
    agent = MonteCarloAgent(env, gamma=0.9)

    # Learn optimal policy
    print("\nLearning optimal policy with on-policy MC control...")
    Q = agent.on_policy_mc_control(
        num_episodes=10000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
    )

    # Extract learned policy
    policy = agent.extract_policy()

    # Print policy
    print("\nLearned Policy:")
    action_symbols = {
        Action.UP: "↑",
        Action.DOWN: "↓",
        Action.LEFT: "←",
        Action.RIGHT: "→",
    }
    for i in range(env.height):
        for j in range(env.width):
            state = (i, j)
            if state in policy:
                print(f"  {action_symbols[policy[state]]}  ", end="")
            elif state == env.goal_pos:
                print("  G  ", end="")
            elif state in env.traps:
                print("  T  ", end="")
            elif not env._is_passable(state):
                print("  #  ", end="")
            else:
                print("  ?  ", end="")
        print()

    # Visualize Q-values
    print("\nGenerating visualizations...")
    env.render_q_values(
        Q, title="Learned Q-Values (MC Control)", save_path="mc_q_values.png"
    )
    print("Saved: mc_q_values.png")

    # Visualize learned policy
    env.render_policy(
        policy, title="Learned Policy (MC Control)", save_path="mc_policy.png"
    )
    print("Saved: mc_policy.png")

    print("\n" + "=" * 60)
    print("Test complete! Check the generated images.")
    print("=" * 60)


if __name__ == "__main__":
    # Test MC evaluation
    test_mc_evaluation()

    # Test on-policy MC control
    test_on_policy_control()
