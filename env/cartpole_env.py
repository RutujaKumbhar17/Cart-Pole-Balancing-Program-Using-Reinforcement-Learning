"""
cartpole_env.py
================
A clean wrapper around the Gymnasium CartPole-v1 environment.

State Space (4 continuous values):
    [0] Cart Position        : [-4.8, 4.8]
    [1] Cart Velocity        : (-inf, inf)
    [2] Pole Angle (radians) : [-0.418, 0.418]  (~±24°)
    [3] Pole Angular Velocity: (-inf, inf)

Action Space (discrete):
    0 → Push cart LEFT
    1 → Push cart RIGHT

Reward:
    +1 for every timestep the pole stays upright.
    Episode ends when pole tips >12° or cart moves ±2.4 units,
    or after 500 timesteps (solved condition).
"""

import gymnasium as gym
import numpy as np


class CartPoleEnv:
    """
    Wrapper class for Gymnasium CartPole-v1.

    Provides a consistent interface, seeding, and metadata
    for use with the DQN agent.
    """

    ENV_NAME = "CartPole-v1"
    MAX_TIMESTEPS = 500          # Episode is considered solved at ≥475 avg reward
    SOLVED_THRESHOLD = 475.0     # Average reward over 100 consecutive episodes

    def __init__(self, seed: int = 42, render_mode: str = None):
        """
        Initialise the CartPole environment.

        Args:
            seed (int): Random seed for reproducibility.
            render_mode (str): One of None, 'human', 'rgb_array'.
        """
        self.seed = seed
        self.render_mode = render_mode

        # Create the Gymnasium environment
        self.env = gym.make(self.ENV_NAME, render_mode=render_mode)

        # Cache dimension info used by the agent/network
        self.state_size = self.env.observation_space.shape[0]   # 4
        self.action_size = self.env.action_space.n               # 2

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Reset the environment to an initial state.

        Returns:
            np.ndarray: Initial state vector of shape (4,).
        """
        state, _ = self.env.reset(seed=self.seed)
        return state.astype(np.float32)

    def step(self, action: int):
        """
        Apply an action and return the transition tuple.

        Args:
            action (int): Action to take (0=left, 1=right).

        Returns:
            tuple: (next_state, reward, done, info)
                   next_state – np.ndarray of shape (4,)
                   reward     – float (+1 each timestep)
                   done       – bool (True if episode finished)
                   info       – dict (auxiliary info from env)
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state.astype(np.float32), float(reward), done, info

    def render(self):
        """Render the current environment state (only in 'human' mode)."""
        return self.env.render()

    def get_frame(self) -> np.ndarray:
        """
        Return an RGB frame of the current state (render_mode='rgb_array' required).

        Returns:
            np.ndarray: RGB image array of shape (H, W, 3).
        """
        return self.env.render()

    def close(self):
        """Close the environment and free resources."""
        self.env.close()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def state_description(self) -> list:
        """Human-readable names for each state dimension."""
        return [
            "Cart Position",
            "Cart Velocity",
            "Pole Angle (rad)",
            "Pole Angular Velocity",
        ]

    @property
    def action_description(self) -> dict:
        """Human-readable mapping from action index to meaning."""
        return {0: "Push LEFT", 1: "Push RIGHT"}

    def __repr__(self) -> str:
        return (
            f"CartPoleEnv("
            f"state_size={self.state_size}, "
            f"action_size={self.action_size}, "
            f"seed={self.seed})"
        )
