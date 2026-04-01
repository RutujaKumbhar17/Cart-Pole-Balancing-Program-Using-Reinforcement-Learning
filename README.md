# 🤖 CartPole-v1 Solved with Deep Q-Network (DQN)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-0081FB?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*A production-quality Deep Reinforcement Learning project that trains an intelligent agent to balance a pole on a moving cart — solving CartPole-v1 from scratch.*

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [What is Reinforcement Learning?](#-what-is-reinforcement-learning)
- [What is Deep Q-Learning (DQN)?](#-what-is-deep-q-learning-dqn)
- [Environment Details](#-environment-details)
- [Algorithm Architecture](#-algorithm-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Hyperparameters](#-hyperparameters)
- [Key Concepts Explained](#-key-concepts-explained)

---

## 🎯 Project Overview

This project implements a **Deep Q-Network (DQN)** agent that learns to solve the **CartPole-v1** environment from the Gymnasium library. The agent learns purely from interaction — starting with completely random behavior and progressively improving until it can balance the pole for **500 timesteps** consistently.

### ✨ Features

| Feature | Details |
|---|---|
| 🧠 Algorithm | Deep Q-Network (DQN) with target network |
| 🎮 Environment | Gymnasium CartPole-v1 |
| 🎬 Video Recording | Gameplay videos saved every 100 episodes |
| 📊 Visualisation | 5 professional training plots |
| 📝 Logging | CSV training log with all metrics |
| 💾 Model Saving | Checkpoint saved after training |
| ⚡ Training | ~1000 episodes, typically solves in 300–600 |

---

## 🧠 What is Reinforcement Learning?

**Reinforcement Learning (RL)** is a machine learning paradigm where an **agent** learns to make decisions by interacting with an **environment** through trial and error.

```
       ┌──────────────────────────────────────────┐
       │                                          │
  ┌────▼────┐    Action (aₜ)    ┌──────────────┐  │
  │         │  ───────────────► │              │  │
  │  Agent  │                   │  Environment │  │
  │         │  ◄─────────────── │              │  │
  └─────────┘  State (sₜ₊₁)    └──────────────┘  │
               Reward (rₜ)                         │
       │                                          │
       └──────────────────────────────────────────┘
                    Closed-loop interaction
```

### Core Concepts

| Term | Definition |
|---|---|
| **Agent** | The learning entity (our DQN network) |
| **Environment** | The world the agent interacts with (CartPole) |
| **State (s)** | The current observation the agent receives |
| **Action (a)** | A decision the agent makes (left or right) |
| **Reward (r)** | Feedback signal (+1 each timestep pole stays up) |
| **Policy (π)** | The strategy that maps states to actions |
| **Episode** | One complete run from reset to termination |

The agent's goal: **maximise cumulative reward** over an episode.

---

## 🔬 What is Deep Q-Learning (DQN)?

**Q-Learning** is a model-free RL algorithm that learns the optimal *action-value function* **Q(s, a)**, which estimates the expected cumulative reward of taking action `a` in state `s` and following the optimal policy thereafter.

### The Bellman Equation

The optimal Q-function satisfies:

```
Q*(s, a) = r  +  γ · max_{a'} Q*(s', a')
```

Where:
- `r`  = immediate reward
- `γ`  = discount factor (how much future rewards matter)
- `s'` = next state after taking action `a`

### Why "Deep"?

In small problems, Q-values can be stored in a table. For CartPole's continuous state space, we use a **neural network** to approximate Q(s, a) for all actions simultaneously — this is **Deep Q-Learning**.

### DQN Innovations (Mnih et al., 2015)

| Innovation | Purpose |
|---|---|
| **Neural Network** | Approximate Q(s,a) in continuous state spaces |
| **Experience Replay** | Store past transitions and sample randomly to break temporal correlations |
| **Target Network** | Separate, slowly-updated network for stable TD targets |
| **ε-Greedy Exploration** | Balance exploration (random) and exploitation (greedy) |

### Training Process

```
For each step:
  1. Observe state s
  2. Select action a (ε-greedy)
  3. Execute a → receive (r, s')
  4. Store (s, a, r, s', done) in replay buffer
  5. Sample random mini-batch from buffer
  6. Compute target:  y = r + γ · max Q_target(s')
  7. Minimize loss:   L = MSE(Q_online(s,a), y)
  8. Update Q_online via backpropagation
  
Every 10 episodes:
  Q_target ← Q_online  (hard copy)
```

---

## 🎮 Environment Details

```
Environment: CartPole-v1 (Gymnasium)

                    ┃  ◄── Pole (angle θ)
                    ┃
         ┌──────────┸──────────┐
         │       Cart          │ ◄── Position x
         └─────────────────────┘
    ──────────────────────────────── Track
         ◄──  0  ──►
         Push LEFT   Push RIGHT
```

### State Space (4 continuous variables)

| Index | Variable | Range |
|---|---|---|
| 0 | Cart Position | [-4.8, 4.8] |
| 1 | Cart Velocity | (-∞, +∞) |
| 2 | Pole Angle (radians) | [-0.418, 0.418] |
| 3 | Pole Angular Velocity | (-∞, +∞) |

### Action Space

| Action | Meaning |
|---|---|
| 0 | Push cart **LEFT** |
| 1 | Push cart **RIGHT** |

### Reward & Termination

- **+1** reward for every timestep the pole stays upright
- Episode ends if:
  - Pole angle > ±12°
  - Cart position > ±2.4 units
  - 500 timesteps reached (success!)
- **Solved**: Average reward ≥ 475 over 100 consecutive episodes

---

## 🏗️ Algorithm Architecture

### Neural Network (Q-function Approximator)

```
Input Layer     Hidden Layer 1    Hidden Layer 2    Output Layer
(state_size=4)  (128 neurons)     (128 neurons)     (action_size=2)

   [cart_pos ]         ┌───┐         ┌───┐        [Q(s, LEFT) ]
   [cart_vel ]  ──►    │ReLU│  ──►   │ReLU│  ──►  [Q(s, RIGHT)]
   [pole_ang ]  ──►    │    │  ──►   │    │  ──►
   [pole_vel ]         └───┘         └───┘

   Xavier init    128 neurons    128 neurons    Linear (no activation)
```

### DQN Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                       DQN AGENT                              │
│                                                              │
│  ┌─────────────┐        ┌──────────────────────────────┐    │
│  │ Environment │──state─►    Online Network Q_θ         │    │
│  │ CartPole-v1 │        │  (FC 4→128→128→2, ReLU)      │    │
│  └─────────────┘        └──────────────┬─────────────┘    │
│         ▲                              │ Q-values          │
│         │ action                       │                    │
│         │                    ε-greedy  ▼                    │
│         └────────────────── select_action()                 │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Replay Buffer                       │    │
│  │           (s, a, r, s', done) × 100K                │    │
│  └─────────────────────────┬──────────────────────────┘    │
│                             │ random mini-batch              │
│                             ▼                                │
│  ┌────────────────────────────────────────────────────┐     │
│  │   Target Network Q_θ'  (frozen, updated every 10ep) │    │
│  │   Computes stable Bellman targets:                   │    │
│  │   y = r + γ · max_a' Q_θ'(s', a')                   │    │
│  └────────────────────────────────────────────────────┘     │
│                             │                                │
│                             ▼                                │
│             Loss = MSE(Q_θ(s,a), y)                         │
│             Optimizer: Adam  →  ∂L/∂θ  →  update θ          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
CartPole-DQN-Project/
│
├── env/
│   ├── __init__.py
│   └── cartpole_env.py          # Gymnasium CartPole-v1 wrapper
│
├── agent/
│   ├── __init__.py
│   └── dqn_agent.py             # DQN agent (action selection, learning, saving)
│
├── models/
│   ├── __init__.py
│   └── dqn_network.py           # PyTorch neural network (4→128→128→2)
│
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py         # Experience replay buffer
│   ├── plotting.py              # Matplotlib training plots (dark theme)
│   ├── video_recorder.py        # OpenCV episode video recorder
│   └── logger.py                # CSV training metrics logger
│
├── training/
│   ├── __init__.py
│   └── train.py                 # ← Main training script
│
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py              # Evaluation script (greedy policy)
│
├── results/
│   ├── videos/                  # episode_0100.mp4, episode_0200.mp4, ...
│   ├── plots/                   # 01_reward_curve.png, 02_average_reward.png, ...
│   └── logs/                    # training_log.csv
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Step 1: Navigate to the project

```bash
cd CartPole-DQN-Project
```

### Step 2: (Optional) Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `gymnasium` | ≥0.29 | CartPole-v1 environment |
| `torch` | ≥2.0 | Neural network + backprop |
| `numpy` | ≥1.24 | Numerical arrays |
| `matplotlib` | ≥3.7 | Training plots |
| `pandas` | ≥2.0 | CSV reading & rolling averages |
| `tqdm` | ≥4.65 | Progress bar |
| `opencv-python` | ≥4.8 | MP4 video encoding |

---

## 🚀 How to Run

### Train the Agent

```bash
# From inside CartPole-DQN-Project/
python training/train.py
```

The training script will:
1. ✅ Print progress to the terminal (episode reward, avg, epsilon)
2. 🎬 Record MP4 gameplay videos every 100 episodes
3. 📊 Generate 5 training plots after completion
4. 📝 Save all metrics to `results/logs/training_log.csv`
5. 💾 Save the model checkpoint to `results/dqn_cartpole_final.pth`

### Evaluate the Trained Agent

```bash
# Run 20 greedy evaluation episodes
python evaluation/evaluate.py

# Run 50 episodes and record a video
python evaluation/evaluate.py --episodes 50 --record

# Load a specific checkpoint
python evaluation/evaluate.py --model results/dqn_cartpole_final.pth
```

### Expected Terminal Output (Training)

```
============================================================
   DQN CartPole-v1 – Training
============================================================
[DQNAgent] Using device: cpu
[Logger] Logging to: results/logs/training_log.csv

Training:  15%|████▌         | 150/1000 [02:31, reward=73, avg100=65.3, epsilon=0.473]

[Episode 200] Recording video ...
[VideoRecorder] Saved: results/videos/episode_0200.mp4  (143 frames)

✓ Environment SOLVED at episode 487! Avg reward = 477.2 (elapsed: 412.3s)

[Plots] Generating plots in 'results/plots' ...
[Plot] Saved: results/plots/01_reward_curve.png
[Plot] Saved: results/plots/02_average_reward.png
[Plot] Saved: results/plots/03_epsilon_decay.png
[Plot] Saved: results/plots/04_loss_curve.png
[Plot] Saved: results/plots/00_training_dashboard.png

============================================================
          TRAINING COMPLETE
============================================================
  Total Episodes       : 487
  Best Episode Reward  : 500.0
  Final Avg Reward     : 477.2
  Total Training Time  : 0:06:52
  Log saved to         : results/logs/training_log.csv
============================================================
```

---

## 📊 Results

### Training Plots Generated

After training, five plots are saved in `results/plots/`:

| Filename | Content |
|---|---|
| `00_training_dashboard.png` | Combined 2×2 overview of all metrics |
| `01_reward_curve.png` | Raw episode rewards + smoothed trend |
| `02_average_reward.png` | Rolling 100-episode average (solved line at 475) |
| `03_epsilon_decay.png` | Epsilon decay from 1.0 → 0.01 |
| `04_loss_curve.png` | MSE TD-loss curve during training |

### Videos Generated

Gameplay MP4 videos are saved in `results/videos/`:

```
episode_0100.mp4   ← Random/early policy (~30-60 reward)
episode_0200.mp4   ← Learning in progress (~100-200 reward)
episode_0300.mp4   ← Improving significantly (~200-400 reward)
episode_0400.mp4   ← Near-optimal policy (~400-500 reward)
episode_0500.mp4   ← Solved policy (~500 reward)
```

### Training Log CSV

`results/logs/training_log.csv` contains:

```csv
Episode,Reward,Average_Reward,Epsilon,Loss
1,12.0,12.0,0.99500,0.0
2,10.0,11.0,0.99003,0.0
...
487,500.0,477.2,0.09432,0.001243
```

### Typical Learning Curve

```
Reward
 500 |                                          ****•••****
 475 |─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ [SOLVED]─ ─ ─
 400 |                              ****
 300 |                        ****
 200 |                  ***
 100 |          **
  50 |  *
     └────────────────────────────────────────────────────
     0        200       400       600       800      1000
                          Episode
```

---

## ⚙️ Hyperparameters

All hyperparameters are configured at the top of `training/train.py`:

| Parameter | Value | Description |
|---|---|---|
| `total_episodes` | 1000 | Maximum training episodes |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor for future rewards |
| `epsilon_start` | 1.0 | Initial exploration probability |
| `epsilon_min` | 0.01 | Minimum exploration probability |
| `epsilon_decay` | 0.995 | Per-episode multiplicative decay |
| `buffer_size` | 100,000 | Maximum replay buffer capacity |
| `batch_size` | 64 | Mini-batch size per training step |
| `target_update_ep` | 10 | Hard target network update interval |
| `hidden_size` | 128 | Neurons per hidden layer |
| `video_interval` | 100 | Record video every N episodes |

---

## 📚 Key Concepts Explained

### Experience Replay

Without replay, consecutive experiences are highly correlated (each state follows the previous). Training on correlated data leads to unstable Q-value updates. The replay buffer stores 100,000 transitions and samples **random mini-batches**, breaking temporal correlations.

### Target Network

Using the same network to compute both predictions and targets causes a moving target problem — the Q-values chase themselves. The **target network** is a frozen copy of the online network, updated every 10 episodes. This provides stable Bellman targets.

### ε-Greedy Exploration

At the start, the agent knows nothing — it explores randomly (ε=1.0). As it learns, ε decays exponentially, shifting toward exploitation of the learned Q-function, with a floor at ε=0.01 for minor continued exploration.

### Gradient Clipping

Applied with `max_norm=1.0` to prevent exploding gradients — a common issue in early training when Q-values are not yet calibrated.

---

## 📖 References

1. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533. https://doi.org/10.1038/nature14236

2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

3. Gymnasium Documentation: https://gymnasium.farama.org/

4. PyTorch Documentation: https://pytorch.org/docs/

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ as a Reinforcement Learning Portfolio Project
</div>
