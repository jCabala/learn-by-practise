import io
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image


class CellType(Enum):
    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3
    TRAP = 4


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass
class VisualTheme:
    """Visual theme configuration for GridWorld rendering"""

    # Color palette
    empty_color: str = "#f8f9fa"
    wall_color: str = "#343a40"
    start_color: str = "#007bff"
    goal_color: str = "#28a745"
    trap_color: str = "#dc3545"

    # Agent and path colors
    agent_color: str = "#007bff"
    path_color: str = "#17a2b8"

    # Value function colors
    value_cmap: str = "RdYlBu_r"

    # Text and styling
    text_color: str = "#212529"
    grid_color: str = "#dee2e6"
    font_family: str = "DejaVu Sans"
    font_size: int = 12

    # Figure settings
    dpi: int = 300
    figsize: tuple[float, float] = (8, 8)


class GridWorld:
    """Customizable GridWorld environment for RL visualization"""

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        start_pos: tuple[int, int] = (0, 0),
        goal_pos: tuple[int, int] = (4, 4),
        walls: list[tuple[int, int]] | None = None,
        traps: list[tuple[int, int]] | None = None,
        rewards: dict[tuple[int, int], float] | None = None,
        step_cost: float = -0.01,
        goal_reward: float = 1.0,
        trap_penalty: float = -1.0,
        noise: float = 0.0,
    ) -> None:
        """
        Initialize GridWorld environment

        Args:
            width, height: Grid dimensions
            start_pos: Starting position (row, col)
            goal_pos: Goal position (row, col)
            walls: List of wall positions
            traps: List of trap positions
            rewards: Custom reward mapping for specific cells
            step_cost: Cost for each step
            goal_reward: Reward for reaching goal
            trap_penalty: Penalty for traps
            noise: Probability of random action (0.0 = deterministic)
        """
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.walls = walls or []
        self.traps = traps or []
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.trap_penalty = trap_penalty
        self.noise = noise

        # Initialize grid
        self.grid = np.zeros((height, width), dtype=int)
        self._setup_grid()

        # Custom rewards
        self.rewards = rewards or {}
        self._setup_rewards()

        # Current state
        self.agent_pos = start_pos
        self.done = False
        self.theme = VisualTheme()
        self._setup_matplotlib()

    def _setup_grid(self) -> None:
        """Setup the grid with walls, start, goal, and traps"""
        # Set walls
        for wall in self.walls:
            if self._is_valid_pos(wall):
                self.grid[wall] = CellType.WALL.value

        # Set special cells
        if self._is_valid_pos(self.start_pos):
            self.grid[self.start_pos] = CellType.START.value

        if self._is_valid_pos(self.goal_pos):
            self.grid[self.goal_pos] = CellType.GOAL.value

        for trap in self.traps:
            if self._is_valid_pos(trap):
                self.grid[trap] = CellType.TRAP.value

    def _setup_rewards(self) -> None:
        """Setup reward function"""
        # Default rewards for all cells
        for i in range(self.height):
            for j in range(self.width):
                pos = (i, j)
                if pos not in self.rewards:
                    if pos == self.goal_pos:
                        self.rewards[pos] = self.goal_reward
                    elif pos in self.traps:
                        self.rewards[pos] = self.trap_penalty
                    else:
                        self.rewards[pos] = self.step_cost

    def _is_valid_pos(self, pos: tuple[int, int]) -> bool:
        """Check if position is within grid bounds"""
        row, col = pos
        return 0 <= row < self.height and 0 <= col < self.width

    def _is_passable(self, pos: tuple[int, int]) -> bool:
        """Check if position can be occupied"""
        return self._is_valid_pos(pos) and self.grid[pos] != CellType.WALL.value

    def get_valid_actions(self, pos: tuple[int, int]) -> list[Action]:
        """Get valid actions from a position"""
        valid_actions = []
        for action in Action:
            new_pos = self._get_next_pos(pos, action)
            if self._is_passable(new_pos):
                valid_actions.append(action)
        return valid_actions

    def _get_next_pos(self, pos: tuple[int, int], action: Action) -> tuple[int, int]:
        """Get next position given current position and action"""
        row, col = pos
        if action == Action.UP:
            return (row - 1, col)
        elif action == Action.RIGHT:
            return (row, col + 1)
        elif action == Action.DOWN:
            return (row + 1, col)
        elif action == Action.LEFT:
            return (row, col - 1)
        return pos

    def step(self, action: Action) -> tuple[tuple[int, int], float, bool]:
        """Take a step in the environment"""
        if self.done:
            return self.agent_pos, 0, True

        # Add noise
        if self.noise > 0 and np.random.random() < self.noise:
            valid_actions = self.get_valid_actions(self.agent_pos)
            if valid_actions:
                action = np.random.choice(valid_actions)

        # Get next position
        next_pos = self._get_next_pos(self.agent_pos, action)

        # Check if move is valid
        if self._is_passable(next_pos):
            self.agent_pos = next_pos

        # Get reward
        reward = self.rewards.get(self.agent_pos, 0)

        # Check if done
        self.done = self.agent_pos == self.goal_pos or self.agent_pos in self.traps

        return self.agent_pos, reward, self.done

    def reset(self) -> tuple[int, int]:
        """Reset environment to initial state"""
        self.agent_pos = self.start_pos
        self.done = False
        return self.agent_pos

    def get_state_space(self) -> list[tuple[int, int]]:
        """Get all valid states in the environment"""
        states = []
        for i in range(self.height):
            for j in range(self.width):
                pos = (i, j)
                if self._is_passable(pos):
                    states.append(pos)
        return states

    def _setup_matplotlib(self) -> None:
        """Configure matplotlib for beautiful plots"""
        plt.rcParams["font.family"] = self.theme.font_family
        plt.rcParams["font.size"] = self.theme.font_size
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["figure.dpi"] = self.theme.dpi

    def render_environment(
        self, title: str = "GridWorld Environment", save_path: str | None = None
    ) -> Image:
        """
        Render the GridWorld environment

        Args:
            show_agent: Whether to show agent position
            agent_pos: Custom agent position (defaults to env.agent_pos)
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.theme.figsize)

        # Create color map for cell types
        colors = [
            self.theme.empty_color,  # EMPTY
            self.theme.wall_color,  # WALL
            self.theme.start_color,  # START
            self.theme.goal_color,  # GOAL
            self.theme.trap_color,  # TRAP
        ]
        cmap = ListedColormap(colors)

        # Plot grid
        ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=4)

        # Add grid lines
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color=self.theme.grid_color, linewidth=1)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color=self.theme.grid_color, linewidth=1)

        # Customize plot
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            title, fontsize=16, fontweight="bold", color=self.theme.text_color, pad=20
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.theme.dpi, bbox_inches="tight")

        return fig

    def render_value_function(
        self,
        values: dict[tuple[int, int], float],
        title: str = "Value Function",
        show_values: bool = True,
        save_path: str | None = None,
    ) -> Image.Image:
        """
        Render value function as heatmap

        Args:
            env: GridWorld environment
            values: Dictionary mapping positions to values
            title: Plot title
            show_values: Whether to show numerical values
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.theme.figsize)

        # Create value grid
        value_grid = np.full((self.height, self.width), np.nan)
        for pos, value in values.items():
            if self._is_passable(pos):
                value_grid[pos] = value

        # Create mask for walls
        mask = np.zeros((self.height, self.width), dtype=bool)
        for wall in self.walls:
            mask[wall] = True

        # Plot heatmap
        im = ax.imshow(value_grid, cmap=self.theme.value_cmap, alpha=0.8)

        # Add walls
        wall_grid = np.zeros((self.height, self.width))
        for wall in self.walls:
            wall_grid[wall] = 1
        ax.imshow(
            wall_grid,
            cmap=ListedColormap(["white", self.theme.wall_color]),
            alpha=1.0,
            vmin=0,
            vmax=1,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Value", rotation=270, labelpad=20)

        # Add value text
        if show_values:
            for i in range(self.height):
                for j in range(self.width):
                    if not np.isnan(value_grid[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{value_grid[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="black",
                            fontweight="bold",
                        )

        # Add grid lines
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color=self.theme.grid_color, linewidth=1, alpha=0.5)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color=self.theme.grid_color, linewidth=1, alpha=0.5)

        # Customize plot
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            title, fontsize=16, fontweight="bold", color=self.theme.text_color, pad=20
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.theme.dpi, bbox_inches="tight")

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def render_q_values(
        self,
        q_values: dict[tuple[tuple[int, int], Action], float],
        title: str = "Q-Values",
        save_path: str | None = None,
    ) -> Image.Image:
        """
        Render Q-values as arrows with colors indicating value magnitude.

        Args:
            q_values: Dictionary mapping (state, action) pairs to Q-values
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.theme.figsize)

        # Plot environment base
        colors = [
            self.theme.empty_color,
            self.theme.wall_color,
            self.theme.start_color,
            self.theme.goal_color,
            self.theme.trap_color,
        ]
        cmap = ListedColormap(colors)
        ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=4, alpha=0.7)

        # Get min and max Q-values for color normalization
        q_vals = [v for v in q_values.values() if v != 0]
        if q_vals:
            vmin, vmax = min(q_vals), max(q_vals)
            # Add some padding to the range
            vrange = vmax - vmin
            if vrange > 0:
                vmin -= 0.1 * vrange
                vmax += 0.1 * vrange
        else:
            vmin, vmax = 0, 1

        # Color map for Q-values
        q_cmap = plt.cm.RdYlGn  # Red (low) to Green (high)

        # Arrow directions and offsets for display
        arrow_dirs = {
            Action.UP: (0, -0.25),
            Action.RIGHT: (0.25, 0),
            Action.DOWN: (0, 0.25),
            Action.LEFT: (-0.25, 0),
        }

        # Offset from center for each action (so arrows don't overlap)
        arrow_offsets = {
            Action.UP: (0, -0.08),
            Action.RIGHT: (0.08, 0),
            Action.DOWN: (0, 0.08),
            Action.LEFT: (-0.08, 0),
        }

        # Plot Q-value arrows
        for (state, action), q_val in q_values.items():
            if (
                not self._is_passable(state)
                or state == self.goal_pos
                or state in self.traps
            ):
                continue

            # Normalize Q-value to [0, 1] for color mapping
            if vmax > vmin:
                norm_val = (q_val - vmin) / (vmax - vmin)
            else:
                norm_val = 0.5

            color = q_cmap(norm_val)

            # Get arrow direction and offset
            dx, dy = arrow_dirs[action]
            offset_x, offset_y = arrow_offsets[action]

            # Draw arrow from slightly offset center
            ax.arrow(
                state[1] + offset_x,
                state[0] + offset_y,
                dx,
                dy,
                head_width=0.12,
                head_length=0.08,
                fc=color,
                ec=color,
                linewidth=1.5,
                alpha=0.8,
            )

            # Add Q-value text near arrow tip
            text_x = state[1] + offset_x + dx * 0.7
            text_y = state[0] + offset_y + dy * 0.7
            ax.text(
                text_x,
                text_y,
                f"{q_val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=q_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Q-Value", rotation=270, labelpad=20)

        # Add grid lines
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color=self.theme.grid_color, linewidth=1)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color=self.theme.grid_color, linewidth=1)

        # Customize plot
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            title, fontsize=16, fontweight="bold", color=self.theme.text_color, pad=20
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.theme.dpi, bbox_inches="tight")

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        return Image.open(buf)

    def render_policy(
        self,
        policy: dict[tuple[int, int], Action],
        title: str = "Policy",
        save_path: str | None = None,
    ) -> Image.Image:
        """
        Render policy as arrows

        Args:
            env: GridWorld environment
            policy: Dictionary mapping positions to actions
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.theme.figsize)

        # Plot environment base
        colors = [
            self.theme.empty_color,
            self.theme.wall_color,
            self.theme.start_color,
            self.theme.goal_color,
            self.theme.trap_color,
        ]
        cmap = ListedColormap(colors)
        ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=4, alpha=0.7)

        # Arrow directions
        arrow_dirs = {
            Action.UP: (0, -0.3),
            Action.RIGHT: (0.3, 0),
            Action.DOWN: (0, 0.3),
            Action.LEFT: (-0.3, 0),
        }

        # Add policy arrows
        for pos, action in policy.items():
            if self._is_passable(pos) and pos != self.goal_pos:
                dx, dy = arrow_dirs[action]
                ax.arrow(
                    pos[1],
                    pos[0],
                    dx,
                    dy,
                    head_width=0.15,
                    head_length=0.1,
                    fc=self.theme.agent_color,
                    ec=self.theme.agent_color,
                    linewidth=2,
                    alpha=0.8,
                )

        # Add grid lines
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color=self.theme.grid_color, linewidth=1)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color=self.theme.grid_color, linewidth=1)

        # Customize plot
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            title, fontsize=16, fontweight="bold", color=self.theme.text_color, pad=20
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.theme.dpi, bbox_inches="tight")

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        return Image.open(buf)


# Example usage and predefined environments
def create_simple_maze() -> GridWorld:
    """Create a simple maze environment"""
    return GridWorld(
        width=5,
        height=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        walls=[(1, 1), (1, 2), (1, 3), (2, 3), (3, 1), (3, 3)],
        traps=[(2, 2)],
        step_cost=0,
        goal_reward=1.0,
        trap_penalty=-1.0,
    )


def create_cliff_walk() -> GridWorld:
    """Create cliff walking environment"""
    return GridWorld(
        width=12,
        height=4,
        start_pos=(3, 0),
        goal_pos=(3, 11),
        traps=[(3, i) for i in range(1, 11)],  # Cliff
        step_cost=-1,
        goal_reward=0,
        trap_penalty=-100,
    )
