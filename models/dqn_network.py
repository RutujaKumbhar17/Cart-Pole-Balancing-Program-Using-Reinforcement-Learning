"""
dqn_network.py
==============
PyTorch neural network used to approximate the Q-value function.

Architecture:
    Input  (4)  →  Linear  → ReLU
    Hidden (128) →  Linear  → ReLU
    Hidden (128) →  Linear
    Output (2)   ← raw Q-values (one per action)

The network takes a batch of state vectors as input and outputs a
Q-value estimate for each possible action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Deep Q-Network (DQN) for CartPole-v1.

    Approximates Q(s, a) for all actions simultaneously given state s.

    Args:
        state_size  (int): Dimensionality of the state space (4 for CartPole).
        action_size (int): Number of discrete actions (2 for CartPole).
        hidden_size (int): Number of neurons in each hidden layer (default 128).
    """

    def __init__(self, state_size: int = 4, action_size: int = 2, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()

        self.state_size  = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # --- Fully-connected layers ---
        self.fc1 = nn.Linear(state_size,  hidden_size)   # Input → Hidden 1
        self.fc2 = nn.Linear(hidden_size, hidden_size)   # Hidden 1 → Hidden 2
        self.fc3 = nn.Linear(hidden_size, action_size)   # Hidden 2 → Output

        # Xavier uniform initialisation for stable early training
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        """Apply Xavier uniform initialisation to all linear layers."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): State batch of shape (batch_size, state_size).

        Returns:
            torch.Tensor: Q-value predictions of shape (batch_size, action_size).
        """
        x = F.relu(self.fc1(x))   # Input → ReLU
        x = F.relu(self.fc2(x))   # Hidden → ReLU
        x = self.fc3(x)           # Hidden → Q-values (no activation)
        return x

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_action_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: compute Q-values for a single (unbatched) state.

        Args:
            state (torch.Tensor): Shape (state_size,) or (1, state_size).

        Returns:
            torch.Tensor: Q-values of shape (action_size,).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)          # Add batch dimension
        with torch.no_grad():
            q_values = self.forward(state)
        return q_values.squeeze(0)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"DQNNetwork(\n"
            f"  (fc1): Linear({self.state_size} → {self.hidden_size})\n"
            f"  (fc2): Linear({self.hidden_size} → {self.hidden_size})\n"
            f"  (fc3): Linear({self.hidden_size} → {self.action_size})\n"
            f"  Total params: {self.count_parameters():,}\n"
            f")"
        )
