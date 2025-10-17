"""
Tutorial 1: Markov Decision Processes and Dynamic Programming
Template File for Students

This file contains skeleton code for implementing value iteration and policy iteration
algorithms. Complete the sections marked with TODO.
"""

from typing import Dict, Tuple

from grid_world import Action, GridWorld


class DynamicProgramming:
    """
    Dynamic Programming algorithms for solving MDPs.

    This class implements value iteration and policy iteration algorithms
    for finding optimal policies in Markov Decision Processes.
    """

    def __init__(self, env: GridWorld, gamma: float = 0.9, theta: float = 1e-6):
        """
        Initialize the DP solver.

        Args:
            env: GridWorld environment
            gamma: Discount factor (0 <= gamma < 1)
            theta: Convergence threshold for iterative algorithms
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.states = env.get_state_space()

    def value_iteration(self, max_iterations: int = 1000) -> Tuple[Dict, Dict]:
        """
        Implement the Value Iteration algorithm.

        The algorithm should:
        1. Initialize V(s) = 0 for all states
        2. Repeat until convergence:
           - For each state s, compute: V_new(s) = max_a Σ_{s'} P(s'|s,a)[R + γV(s')]
           - Check convergence: if max_s |V_new(s) - V(s)| < theta, stop
        3. Extract the optimal policy from the optimal value function

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            values: Dictionary mapping states to their optimal values
            policy: Dictionary mapping states to optimal actions
        """
        # Initialize value function
        values = {state: 0.0 for state in self.states}

        for iteration in range(max_iterations):
            # Create a copy of values for the new iteration
            new_values = values.copy()
            delta = 0.0  # Track the maximum change in value

            # Loop over all states
            for state in self.states:
                # Skip terminal states (goal and traps)
                if state == self.env.goal_pos or state in self.env.traps:
                    continue

                # Compute the value of each action from this state
                action_values = []
                valid_actions = self.env.get_valid_actions(state)

                for action in valid_actions:
                    # TODO: Compute Q(s,a) for this action
                    # Use the helper function self._compute_action_value()
                    q_value = self._compute_action_value(state, action, values)
                    action_values.append(q_value)

                # TODO: Take the maximum over all actions
                if action_values:
                    new_values[state] = max(action_values)

                # Update delta (maximum change in value)
                delta = max(delta, abs(new_values[state] - values[state]))

            # Update values
            values = new_values

            # Check for convergence
            if delta < self.theta:
                print(f"Value iteration converged in {iteration + 1} iterations")
                break

        # Extract the optimal policy
        policy = self.extract_policy(values)

        return values, policy

    def _compute_action_value(
        self,
        state: Tuple[int, int],
        action: Action,
        values: Dict[Tuple[int, int], float],
    ) -> float:
        """
        Compute Q(s,a) = Σ_{s'} P(s'|s,a)[R + γV(s')].

        For deterministic environments, this simplifies to:
        Q(s,a) = R + γV(s')

        Args:
            state: Current state
            action: Action to evaluate
            values: Current value function

        Returns:
            The action value Q(s,a)
        """
        # ============================================
        # TODO: Implement action value computation
        # ============================================

        # Get the next state (for deterministic transitions)
        next_state = self.env.get_next_state(state, action)

        # Get the reward for this transition
        reward = self.env.rewards[next_state]

        # Compute Q(s,a) = R + γV(s')
        q_value = reward + self.gamma * values[next_state]
        return q_value

    def extract_policy(
        self, values: Dict[Tuple[int, int], float]
    ) -> Dict[Tuple[int, int], Action]:
        """
        Extract the greedy policy from a value function.

        For each state, choose the action that maximizes:
        π(s) = argmax_a Σ_{s'} P(s'|s,a)[R + γV(s')]

        Args:
            values: Value function V(s)

        Returns:
            policy: Dictionary mapping states to actions
        """
        policy = {}

        # ============================================
        # TODO: Extract greedy policy
        # ============================================

        for state in self.states:
            # Skip terminal states
            if state == self.env.goal_pos or state in self.env.traps:
                continue

            valid_actions = self.env.get_valid_actions(state)
            best_action = None
            best_value = float("-inf")

            for action in valid_actions:
                # TODO: Compute Q(s,a) using self._compute_action_value()
                q_value = self._compute_action_value(state, action, values)

                # Track the best action
                if q_value > best_value:
                    best_value = q_value
                    best_action = action

            policy[state] = best_action

        return policy


def test_value_iteration():
    """
    Test the value iteration implementation on a simple GridWorld.
    """
    from grid_world import create_simple_maze

    print("=" * 60)
    print("Testing Value Iteration on Simple Maze")
    print("=" * 60)

    # Create environment
    env = create_simple_maze()

    # Create DP solver
    dp = DynamicProgramming(env, gamma=0.9)

    # Run value iteration
    print("\nRunning value iteration...")
    values, policy = dp.value_iteration()

    # Visualize results
    print("\nGenerating visualizations...")
    env.render_value_function(
        values, title="Optimal Value Function V*", save_path="value_function.png"
    )
    print("Saved: value_function.png")

    env.render_policy(policy, title="Optimal Policy π*", save_path="optimal_policy.png")
    print("Saved: optimal_policy.png")

    print("\n" + "=" * 60)
    print("Run complete! Check the generated images.")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests
    test_value_iteration()
