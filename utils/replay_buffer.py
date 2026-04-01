"""
replay_buffer.py
================
Experience Replay Buffer for Deep Q-Learning.

Stores (state, action, reward, next_state, done) transitions and
allows random mini-batch sampling to break temporal correlations
in the training data — a key stabilisation technique in DQN.

References:
    Mnih et al. (2015) "Human-level control through deep reinforcement learning"
    https://www.nature.com/articles/nature14236
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Fixed-size circular replay buffer.

    Once full, the oldest experience is discarded to make room
    for the newest one (FIFO strategy via collections.deque).

    Args:
        capacity (int): Maximum number of transitions to store.
        seed     (int): Random seed for reproducible sampling.
    """

    def __init__(self, capacity: int = 100_000, seed: int = 42):
        self.buffer   = deque(maxlen=capacity)
        self.capacity = capacity
        self.seed     = seed
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """
        Store a single experience transition.

        Args:
            state      : Current state  – shape (state_size,)
            action     : Action taken   – integer
            reward     : Reward received – float
            next_state : Next state     – shape (state_size,)
            done       : Whether the episode ended – bool
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = 64):
        """
        Randomly sample a mini-batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple of numpy arrays:
                states      – shape (batch_size, state_size)
                actions     – shape (batch_size,)
                rewards     – shape (batch_size,)
                next_states – shape (batch_size, state_size)
                dones       – shape (batch_size,)

        Raises:
            ValueError: If buffer contains fewer experiences than batch_size.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer has only {len(self.buffer)} samples; "
                f"need at least {batch_size}."
            )

        batch = random.sample(self.buffer, batch_size)

        # Unpack and convert to contiguous numpy arrays for PyTorch
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),   # bool → float (1.0/0.0)
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return current number of stored experiences."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Return True if the buffer has enough samples to train."""
        return len(self.buffer) >= batch_size

    def fill_ratio(self) -> float:
        """Return fraction of capacity used (0.0 – 1.0)."""
        return len(self.buffer) / self.capacity

    def clear(self) -> None:
        """Clear all stored experiences."""
        self.buffer.clear()

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer("
            f"size={len(self.buffer)}/{self.capacity}, "
            f"fill={self.fill_ratio():.1%})"
        )
