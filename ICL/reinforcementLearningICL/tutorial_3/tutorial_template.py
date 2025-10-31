"""
Tutorial 3: Temporal Difference Learning
Template File for Students

This file contains skeleton code for implementing TD(0), SARSA, Q-learning,
and Double Q-learning algorithms. Complete the sections marked with TODO.
"""

from typing import Dict, Tuple

import numpy as np
from grid_world import Action, GridWorld


class TDAgent:
    """
    Temporal Difference methods for learning from experience.

    This class implements TD(0) for policy evaluation and TD control methods
    (SARSA, Q-learning, Double Q-learning).
    """

    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.9,
        alpha: float = 0.1,
        epsilon: float = 0.1,
    ):
        """
        Initialize the TD agent.

        Args:
            env: GridWorld environment
            gamma: Discount factor (0 <= gamma < 1)
            alpha: Learning rate / step size (0 < alpha <= 1)
            epsilon: Exploration parameter for epsilon-greedy policies
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.states = env.get_state_space()

        # Initialize value function V(s) for TD(0)
        self.V = {state: 0.0 for state in self.states}

        # Initialize Q-values for control methods
        self.Q = {}
        for state in self.states:
            for action in env.get_valid_actions(state):
                self.Q[(state, action)] = 0.0

    def td0_policy_evaluation(
        self, policy: Dict[Tuple[int, int], Action], num_episodes: int = 500
    ) -> Dict[Tuple[int, int], float]:
        """
        TD(0) algorithm for policy evaluation.

        Args:
            policy: Policy to evaluate (maps states to actions)
            num_episodes: Number of episodes to run

        Returns:
            V: Estimated state values
        """
        # ============================================
        # TODO: Implement TD(0) policy evaluation
        # ============================================

        for episode in range(num_episodes):
            state = self.env.reset()

            # Run episode until termination
            for step in range(1000):  # Max steps to prevent infinite loops

                # Get action from policy
                if state in policy:
                    action = policy[state]
                else:
                    # If state not in policy, choose random valid action
                    valid_actions = self.env.get_valid_actions(state)
                    if not valid_actions:
                        break
                    action = np.random.choice(valid_actions)

                # TODO: Take action and observe reward and next state
                next_state, reward, done = self.env.step(action) 

                # TODO: Compute TD error
                # Hint: delta = reward + gamma * V(next_state) - V(state)
                # Remember: V(terminal_state) = 0
                td_error = reward + self.gamma * self.V[next_state] - self.V[state]

                # TODO: Update V(state) using TD error
                # Hint: V(state) = V(state) + alpha * td_error
                self.V[state] += self.alpha * td_error

                # Move to next state
                state = next_state

                if done:
                    break

            # Print progress
            if (episode + 1) % 100 == 0:
                print(f"TD(0) Episode {episode + 1}/{num_episodes}")

        return self.V

    def epsilon_greedy_action(self, state: Tuple[int, int]) -> Action:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            action: Selected action
        """
        # ============================================
        # TODO: Implement epsilon-greedy action selection
        # ============================================

        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return None

        # TODO: With probability epsilon, choose random action
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        # TODO: Otherwise, choose greedy action (highest Q-value)
        else:
            best_action = None
            best_value = float("-inf")

            for action in valid_actions:
                # TODO: Find action with highest Q-value
                if (state, action) in self.Q:
                    q_value = self.Q[(state, action)]
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action

            return best_action

    def sarsa(
        self,
        num_episodes: int = 5000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: float = 5000,
    ) -> Dict[Tuple[Tuple[int, int], Action], float]:
        """
        SARSA: On-policy TD control.

        Args:
            num_episodes: Number of episodes to train
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay_steps: Number of steps to decay epsilon

        Returns:
            Q: Learned Q-values
        """
        # ============================================
        # TODO: Implement SARSA
        # ============================================

        self.epsilon = epsilon_start

        for episode in range(num_episodes):
            state = self.env.reset()

            # TODO: Select initial action using epsilon-greedy
            action = None  # REPLACE THIS LINE

            for step in range(1000):

                # TODO: Take action, observe reward and next state
                next_state, reward, done = self.env.step(action)

                if done:
                    # TODO: Update Q-value for terminal transition
                    # Hint: TD target = reward (no next state value)
                    # Q(state, action) = Q(state, action) +
                    # alpha * [reward - Q(state, action)]
                    self.Q[(state, action)] = self.Q.get((state, action), 0) + self.alpha * (
                        reward - self.Q.get((state, action), 0)
                    )
                    break

                # TODO: Select next action using epsilon-greedy
                next_action = self.epsilon_greedy_action(next_state)

                # TODO: Compute TD error for SARSA
                # Hint: td_error = reward + gamma *
                # Q(next_state, next_action) - Q(state, action)
                td_error = reward + self.gamma * self.Q.get((next_state, next_action), 0) - self.Q.get((state, action), 0)

                # TODO: Update Q(state, action)
                # Hint: Q(state, action) = Q(state, action) + alpha * td_error
                self.Q[(state, action)] = self.Q.get((state, action), 0) + self.alpha * td_error

                # Move to next state and action
                state = next_state
                action = next_action

            # TODO: Decay epsilon
            self.epsilon = max(epsilon_end, epsilon_start - episode / epsilon_decay_steps * (epsilon_start - epsilon_end))

            # Print progress
            if (episode + 1) % 1000 == 0:
                print(
                    f"SARSA Episode {episode + 1}/{num_episodes}, "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        return self.Q

    def q_learning(
        self,
        num_episodes: int = 5000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: float = 5000,
    ) -> Dict[Tuple[Tuple[int, int], Action], float]:
        """
        Q-learning: Off-policy TD control.

        Args:
            num_episodes: Number of episodes to train
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay_steps: Number of steps to decay epsilon

        Returns:
            Q: Learned Q-values
        """
        # ============================================
        # TODO: Implement Q-learning
        # ============================================

        self.epsilon = epsilon_start

        for episode in range(num_episodes):
            state = self.env.reset()

            for step in range(1000):

                # TODO: Select action using epsilon-greedy
                action = self.epsilon_greedy_action(state)

                # TODO: Take action, observe reward and next state
                next_state, reward, done = self.env.step(action)

                if done:
                    # TODO: Update Q-value for terminal transition
                    self.Q[(state, action)] = self.Q.get((state, action), 0) + self.alpha * (
                        reward - self.Q.get((state, action), 0)
                    )
                    break

                # TODO: Find max Q-value for next state
                # Hint: max_q_next = max over all actions a' of Q(next_state, a')
                max_q_next = max(
                    self.Q.get((next_state, a), 0) for a in self.env.get_valid_actions(next_state)
                )

                # TODO: Compute TD error for Q-learning
                # Hint: td_error = reward + gamma * max_q_next - Q(state, action)
                td_error = reward + self.gamma * max_q_next - self.Q.get((state, action), 0)

                # TODO: Update Q(state, action)
                self.Q[(state, action)] = self.Q.get((state, action), 0) + self.alpha * td_error

                # Move to next state
                state = next_state

            # TODO: Decay epsilon
            self.epsilon = max(epsilon_end, epsilon_start - episode / epsilon_decay_steps * (epsilon_start - epsilon_end))

            # Print progress
            if (episode + 1) % 1000 == 0:
                print(
                    f"Q-learning Episode {episode + 1}/{num_episodes}, "
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


def test_td0():
    """
    Test TD(0) policy evaluation.
    """
    from grid_world import create_simple_maze

    print("=" * 60)
    print("Testing TD(0) Policy Evaluation")
    print("=" * 60)

    env = create_simple_maze()

    # Create a simple policy (go right and down toward goal)
    policy = {}
    for state in env.get_state_space():
        if state == env.goal_pos or state in env.traps:
            continue
        valid_actions = env.get_valid_actions(state)
        if Action.RIGHT in valid_actions:
            policy[state] = Action.RIGHT
        elif Action.DOWN in valid_actions:
            policy[state] = Action.DOWN
        elif valid_actions:
            policy[state] = valid_actions[0]

    agent = TDAgent(env, gamma=0.9, alpha=0.1)

    print("\nEvaluating policy with TD(0)...")
    V = agent.td0_policy_evaluation(policy, num_episodes=500)

    print("\nState Values:")
    for state in sorted(V.keys()):
        print(f"V{state} = {V[state]:.3f}")

    print("\nGenerating visualization...")
    env.render_value_function(
        V, title="State Values (TD(0))", save_path="td0_values.png"
    )
    print("Saved: td0_values.png")

    print("\n" + "=" * 60)


def test_sarsa():
    """
    Test SARSA on cliff walking environment.
    """
    from grid_world import create_cliff_walk

    print("=" * 60)
    print("Testing SARSA")
    print("=" * 60)

    env = create_cliff_walk()
    agent = TDAgent(env, gamma=0.9, alpha=0.1)

    print("\nLearning with SARSA...")
    Q = agent.sarsa(
        num_episodes=5000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=5000,
    )

    policy = agent.extract_policy()

    print("\nGenerating visualizations...")
    env.render_q_values(Q, title="SARSA Q-Values", save_path="sarsa_q_values.png")
    print("Saved: sarsa_q_values.png")

    env.render_policy(policy, title="SARSA Policy", save_path="sarsa_policy.png")
    print("Saved: sarsa_policy.png")

    print("\n" + "=" * 60)


def test_q_learning():
    """
    Test Q-learning on cliff walking environment.
    """
    from grid_world import create_cliff_walk

    print("=" * 60)
    print("Testing Q-Learning")
    print("=" * 60)

    env = create_cliff_walk()
    agent = TDAgent(env, gamma=0.9, alpha=0.1)

    print("\nLearning with Q-learning...")
    Q = agent.q_learning(
        num_episodes=5000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=5000,
    )

    policy = agent.extract_policy()

    print("\nGenerating visualizations...")
    env.render_q_values(
        Q, title="Q-Learning Q-Values", save_path="qlearning_q_values.png"
    )
    print("Saved: qlearning_q_values.png")

    env.render_policy(
        policy, title="Q-Learning Policy", save_path="qlearning_policy.png"
    )
    print("Saved: qlearning_policy.png")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test TD(0) policy evaluation
    test_td0()

    # Test SARSA
    test_sarsa()

    # Test Q-learning
    test_q_learning()

    print("\nAll tests complete! Check the generated images.")
