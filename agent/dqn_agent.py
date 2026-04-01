"""
dqn_agent.py
============
Deep Q-Network (DQN) agent implementation.

Implements the classic DQN algorithm from:
    Mnih et al. (2015) "Human-level control through deep reinforcement learning"
    Nature 518, 529–533. https://doi.org/10.1038/nature14236

Key components:
    ┌─────────────────────────────────────────────────────────┐
    │  Online Network   →  selects actions, trained every step │
    │  Target Network   →  provides stable TD targets          │
    │  Replay Buffer    →  breaks temporal correlations        │
    │  ε-Greedy Policy  →  balances exploration & exploitation │
    └─────────────────────────────────────────────────────────┘

Training objective (Bellman equation):
    Q_target = r  +  γ · max_a' Q_target(s', a')   if not done
    Q_target = r                                     if done

Loss = MSE( Q_online(s, a),  Q_target )
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Internal imports (relative paths resolved via sys.path in train.py)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dqn_network import DQNNetwork
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent with experience replay and target network.

    Args:
        state_size       (int)   : Dimensionality of the state space.
        action_size      (int)   : Number of discrete actions.
        learning_rate    (float) : Adam optimiser learning rate.
        gamma            (float) : Discount factor for future rewards.
        epsilon_start    (float) : Initial exploration rate.
        epsilon_min      (float) : Minimum exploration rate.
        epsilon_decay    (float) : Multiplicative decay applied each episode.
        buffer_size      (int)   : Maximum replay buffer capacity.
        batch_size       (int)   : Mini-batch size for each training step.
        target_update_ep (int)   : Episodes between target network hard updates.
        seed             (int)   : Random seed for reproducibility.
        device           (str)   : 'cpu' or 'cuda' (auto-detected if None).
    """

    def __init__(
        self,
        state_size:        int   = 4,
        action_size:       int   = 2,
        learning_rate:     float = 1e-3,
        gamma:             float = 0.99,
        epsilon_start:     float = 1.0,
        epsilon_min:       float = 0.01,
        epsilon_decay:     float = 0.995,
        buffer_size:       int   = 100_000,
        batch_size:        int   = 64,
        target_update_ep:  int   = 10,
        seed:              int   = 42,
        device:            str   = None,
    ):
        # --- Hyperparameters ---
        self.state_size        = state_size
        self.action_size       = action_size
        self.gamma             = gamma
        self.epsilon           = epsilon_start
        self.epsilon_min       = epsilon_min
        self.epsilon_decay     = epsilon_decay
        self.batch_size        = batch_size
        self.target_update_ep  = target_update_ep
        self.seed              = seed

        # --- Device ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Print detailed GPU/CPU info
        if self.device.type == "cuda":
            gpu_id   = self.device.index or 0
            gpu_name = torch.cuda.get_device_name(gpu_id)
            vram_gb  = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
            cuda_ver = torch.version.cuda
            print(f"[DQNAgent] ✓ GPU detected — using CUDA")
            print(f"           GPU  : {gpu_name}")
            print(f"           VRAM : {vram_gb:.1f} GB")
            print(f"           CUDA : {cuda_ver}")
            print(f"           cuDNN: {torch.backends.cudnn.version()}")
            # Enable cuDNN auto-tuner for faster convolution (good for fixed input size)
            torch.backends.cudnn.benchmark = True
        else:
            print(f"[DQNAgent] Using device: CPU  (no CUDA GPU found)")

        # --- Reproducibility ---
        self._set_seeds(seed)

        # --- Online and Target Networks ---
        self.online_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()   # Target net is never trained directly

        # --- Optimiser (Adam) ---
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # --- Loss function: Mean Squared Error ---
        self.criterion = nn.MSELoss()

        # --- Replay Buffer ---
        self.memory = ReplayBuffer(capacity=buffer_size, seed=seed)

        # --- Step / episode counters ---
        self.total_steps    = 0
        self.episode_count  = 0

        print(f"[DQNAgent] Network:\n{self.online_net}")

    # ------------------------------------------------------------------
    # Seed helpers
    # ------------------------------------------------------------------

    def _set_seeds(self, seed: int) -> None:
        """Set all relevant random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, epsilon: float = None) -> int:
        """
        Choose an action using ε-greedy policy.

        Args:
            state   : Current state vector.
            epsilon : Override epsilon (e.g. 0.0 for greedy). Uses
                      self.epsilon if None.

        Returns:
            int : Selected action index.
        """
        eps = epsilon if epsilon is not None else self.epsilon

        if random.random() < eps:
            # Explore: random action
            return random.randrange(self.action_size)
        else:
            # Exploit: greedy action from online network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
            return q_values.argmax(dim=1).item()

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def remember(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def learn(self) -> float:
        """
        Sample a mini-batch from the replay buffer and perform one
        gradient-descent step on the online network.

        Returns:
            float : The MSE loss value for this update step.
                    Returns 0.0 if the buffer does not yet have enough samples.
        """
        if not self.memory.is_ready(self.batch_size):
            return 0.0

        # ── 1. Sample mini-batch ─────────────────────────────────────────
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to PyTorch tensors and move to device
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # ── 2. Compute current Q-values from online network ──────────────
        # online_net(states_t) → (batch, action_size)
        # .gather selects the Q-value for the taken action
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # ── 3. Compute target Q-values using target network ──────────────
        with torch.no_grad():
            next_q_max = self.target_net(next_states_t).max(dim=1)[0]

        # Bellman target: Q = r  +  γ · max Q_target(s')  ·  (1 − done)
        target_q = rewards_t + self.gamma * next_q_max * (1.0 - dones_t)

        # ── 4. Compute loss & back-propagate ────────────────────────────
        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def update_target_network(self) -> None:
        """
        Hard copy weights from online network → target network.

        Called every `target_update_ep` episodes.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Apply multiplicative epsilon decay after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "results/dqn_cartpole.pth") -> None:
        """
        Save the online network weights to disk.

        Args:
            path (str): File path for the saved model.
        """
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "online_net":  self.online_net.state_dict(),
                "target_net":  self.target_net.state_dict(),
                "epsilon":     self.epsilon,
                "total_steps": self.total_steps,
                "device":      str(self.device),
            },
            path,
        )
        print(f"[DQNAgent] Model saved to: {os.path.abspath(path)}")

    def load(self, path: str) -> None:
        """
        Load model weights from a checkpoint file.

        Args:
            path (str): Path to the .pth checkpoint.
        """
        # weights_only=True avoids the PyTorch 2.x pickle warning
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon     = checkpoint.get("epsilon",     self.epsilon)
        self.total_steps = checkpoint.get("total_steps", 0)
        print(f"[DQNAgent] Model loaded from: {os.path.abspath(path)}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DQNAgent("
            f"state={self.state_size}, "
            f"actions={self.action_size}, "
            f"ε={self.epsilon:.4f}, "
            f"buffer={len(self.memory)}, "
            f"steps={self.total_steps})"
        )
