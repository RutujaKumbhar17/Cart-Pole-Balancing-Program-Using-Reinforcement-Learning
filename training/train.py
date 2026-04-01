"""
train.py
========
Main training script for the DQN CartPole-v1 agent.

Run command:
    python training/train.py

This script:
    1. Initialises the CartPole environment and DQN agent.
    2. Runs the training loop for a configurable number of episodes.
    3. Records gameplay videos every VIDEO_INTERVAL episodes.
    4. Decays epsilon after each episode.
    5. Updates the target network every TARGET_UPDATE episodes.
    6. Logs all metrics to CSV via TrainingLogger.
    7. Generates all plots after training completes.
    8. Saves the trained model checkpoint.
    9. Prints a final summary of training statistics.
"""

import os
import sys
import time
from collections import deque
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# Allow imports from the project root whether running from project root
# or from the training/ subdirectory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Standard library + third-party ──────────────────────────────────────────
import numpy as np
import torch
from tqdm import tqdm

# ── Project imports ──────────────────────────────────────────────────────────
from env.cartpole_env      import CartPoleEnv
from agent.dqn_agent       import DQNAgent
from utils.replay_buffer   import ReplayBuffer
from utils.logger          import TrainingLogger
from utils.plotting        import generate_all_plots
from utils.video_recorder  import VideoRecorder


# ============================================================================
# HYPERPARAMETERS  (edit here to experiment)
# ============================================================================

CONFIG = {
    # Training
    "total_episodes":    1000,
    "max_steps_per_ep":  500,
    "seed":              42,

    # DQN agent
    "learning_rate":     1e-3,
    "gamma":             0.99,
    "epsilon_start":     1.0,
    "epsilon_min":       0.01,
    "epsilon_decay":     0.995,
    "buffer_size":       100_000,
    "batch_size":        64,
    "target_update_ep":  10,

    # Neural network
    "hidden_size":       128,

    # Recording
    "video_interval":    60,      # Record every 60 episodes so you see learning stages

    # Output paths
    "results_dir":       "results",
    "log_dir":           "results/logs",
    "plot_dir":          "results/plots",
    "video_dir":         "results/videos",
    "model_path":        "results/dqn_cartpole_final.pth",
}


# ============================================================================
# Training loop
# ============================================================================

def train() -> None:
    """
    Execute the full DQN training pipeline.

    Steps per episode:
        1. Reset environment
        2. Loop:  select action → step → store → learn
        3. Decay epsilon
        4. Update target network (every TARGET_UPDATE episodes)
        5. Every VIDEO_INTERVAL episodes: record gameplay video
        6. Log metrics
    """
    print("\n" + "=" * 60)
    print("   DQN CartPole-v1 – Training")
    print("=" * 60)

    # ── GPU / CPU banner ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  Hardware : CUDA GPU  — {gpu_name}  ({vram_gb:.1f} GB VRAM)")
        print(f"  CUDA     : {torch.version.cuda}")
    else:
        print("  Hardware : CPU  (no CUDA GPU found)")
    
    # ── OpenCV check ──
    try:
        import cv2
        print(f"  OpenCV   : {cv2.__version__}  (Annotated videos ENABLED)")
    except ImportError:
        print("  OpenCV   : Missing  (!!! Annotated videos DISABLED !!!)")
    print()

    # ── Ensure output directories exist ─────────────────────────────────────
    for d in [CONFIG["log_dir"], CONFIG["plot_dir"], CONFIG["video_dir"]]:
        os.makedirs(d, exist_ok=True)

    # ── Initialise components ────────────────────────────────────────────────
    env    = CartPoleEnv(seed=CONFIG["seed"])
    agent  = DQNAgent(
        state_size       = env.state_size,
        action_size      = env.action_size,
        learning_rate    = CONFIG["learning_rate"],
        gamma            = CONFIG["gamma"],
        epsilon_start    = CONFIG["epsilon_start"],
        epsilon_min      = CONFIG["epsilon_min"],
        epsilon_decay    = CONFIG["epsilon_decay"],
        buffer_size      = CONFIG["buffer_size"],
        batch_size       = CONFIG["batch_size"],
        target_update_ep = CONFIG["target_update_ep"],
        seed             = CONFIG["seed"],
    )

    logger    = TrainingLogger(log_dir=CONFIG["log_dir"])
    recorder  = VideoRecorder(save_dir=CONFIG["video_dir"])

    # ── Tracking variables ───────────────────────────────────────────────────
    reward_history   = []         # Per-episode total rewards
    epsilon_history  = []         # Per-episode epsilon values
    loss_history     = []         # Per-episode mean losses

    recent_rewards   = deque(maxlen=100)   # Rolling window for avg reward
    best_reward      = -float("inf")
    start_time       = time.time()

    total_eps = CONFIG["total_episodes"]

    # ── Training progress bar ────────────────────────────────────
    pbar = tqdm(range(1, total_eps + 1), desc="Training", unit="ep", ncols=90)

    # ── Record Episode 1 BEFORE any training (pure random baseline) ──
    tqdm.write("[Ep 0] Recording BASELINE video (pure random policy, ε=1.0) ...")
    recorder.record_episode(
        agent,
        episode_number = 0,        # -> episode_0000_baseline.mp4
        seed           = CONFIG["seed"],
        record_epsilon = 1.0,      # Force random so we see an untrained collapse
    )

    for episode in pbar:
        state    = env.reset()
        ep_reward = 0.0
        ep_losses = []

        # ── Inner step loop ──────────────────────────────────────────────────
        for step in range(CONFIG["max_steps_per_ep"]):
            # 1. Select action (ε-greedy)
            action = agent.select_action(state)

            # 2. Execute action in environment
            next_state, reward, done, _ = env.step(action)

            # 3. Store transition in replay buffer
            agent.remember(state, action, reward, next_state, done)

            # 4. Train the online network
            loss = agent.learn()
            if loss > 0:
                ep_losses.append(loss)

            # 5. Advance
            state      = next_state
            ep_reward += reward

            if done:
                break

        # ── Post-episode updates ─────────────────────────────────────────────

        # Update target network every TARGET_UPDATE episodes
        if episode % CONFIG["target_update_ep"] == 0:
            agent.update_target_network()

        # Decay exploration rate
        agent.decay_epsilon()

        # Compute statistics
        mean_loss   = float(np.mean(ep_losses)) if ep_losses else 0.0
        recent_rewards.append(ep_reward)
        avg_reward  = float(np.mean(recent_rewards))

        if ep_reward > best_reward:
            best_reward = ep_reward

        # Store history for plotting
        reward_history.append(ep_reward)
        epsilon_history.append(agent.epsilon)
        loss_history.append(mean_loss)

        # Log to CSV
        logger.log(
            episode    = episode,
            reward     = ep_reward,
            avg_reward = avg_reward,
            epsilon    = agent.epsilon,
            loss       = mean_loss,
        )

        # Update progress bar description
        pbar.set_postfix(
            reward  = f"{ep_reward:.0f}",
            avg100  = f"{avg_reward:.1f}",
            epsilon = f"{agent.epsilon:.3f}",
        )

        # ── Video recording ──────────────────────────────────────────────────
        if episode % CONFIG["video_interval"] == 0:
            tqdm.write(f"")
            # record_epsilon=None  →  uses agent.epsilon (current training eps)
            # This captures the ACTUAL behaviour at this stage of learning,
            # including early unstable episodes where the pole falls quickly.
            recorder.record_episode(
                agent,
                episode_number = episode,
                seed           = CONFIG["seed"],
                record_epsilon = None,   # None = use current epsilon
            )

        # ── Early stopping: environment solved ───────────────────────────────
        if avg_reward >= 475.0 and len(recent_rewards) == 100:
            elapsed = time.time() - start_time
            tqdm.write(
                f"\n✓ Environment SOLVED at episode {episode}! "
                f"Avg reward = {avg_reward:.1f} "
                f"(elapsed: {elapsed:.1f}s)"
            )
            # Fill remaining history with current values for clean plots
            while len(reward_history) < total_eps:
                reward_history.append(ep_reward)
                epsilon_history.append(agent.epsilon)
                loss_history.append(mean_loss)
            break

    # ── Close environment ────────────────────────────────────────────────────
    env.close()
    pbar.close()

    # ── Save trained model ───────────────────────────────────────────────────
    agent.save(CONFIG["model_path"])

    # ── Generate plots ───────────────────────────────────────────────────────
    generate_all_plots(
        rewards  = reward_history,
        epsilons = epsilon_history,
        losses   = loss_history,
        save_dir = CONFIG["plot_dir"],
    )

    # ── Final summary ────────────────────────────────────────────────────────
    final_avg     = float(np.mean(list(recent_rewards)))
    elapsed_total = time.time() - start_time

    logger.print_summary(
        total_episodes  = len(reward_history),
        best_reward     = best_reward,
        final_avg       = final_avg,
    )

    print(f"  Total wall-clock time : {elapsed_total:.1f}s")

    # ── Record a final greedy video (best policy) ────────────────────────────
    print("\n[Final Video] Recording greedy (best policy) video ...")
    recorder.record_episode(
        agent,
        episode_number = 9999,
        seed           = CONFIG["seed"],
        record_epsilon = 0.0,     # Fully greedy – shows the solved policy
    )

    # ── List all saved videos ────────────────────────────────────────────────
    videos = recorder.list_videos()
    print(f"\n  Videos saved ({len(videos)} total):")
    for v in videos:
        size_kb = Path(v).stat().st_size // 1024
        print(f"    {v.name}  ({size_kb} KB)")
    print(f"  Model checkpoint      : {CONFIG['model_path']}")
    print()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    train()
