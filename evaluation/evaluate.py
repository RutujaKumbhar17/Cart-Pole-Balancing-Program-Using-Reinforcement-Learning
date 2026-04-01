"""
evaluate.py
===========
Evaluation script for a trained DQN CartPole-v1 agent.

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --model results/dqn_cartpole_final.pth --episodes 20

This script:
    1. Loads a saved model checkpoint.
    2. Runs N evaluation episodes with a fully greedy policy (ε=0).
    3. Prints per-episode rewards and a summary table.
    4. Optionally records a final evaluation video.
"""

import os
import sys
import argparse
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Project imports ───────────────────────────────────────────────────────────
from env.cartpole_env     import CartPoleEnv
from agent.dqn_agent      import DQNAgent
from utils.video_recorder import VideoRecorder


# ============================================================================
# Argument parser
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent on CartPole-v1."
    )
    parser.add_argument(
        "--model",
        type    = str,
        default = "results/dqn_cartpole_final.pth",
        help    = "Path to the saved model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--episodes",
        type    = int,
        default = 20,
        help    = "Number of evaluation episodes to run (default: 20).",
    )
    parser.add_argument(
        "--record",
        action  = "store_true",
        help    = "Record a video of the best evaluation episode.",
    )
    parser.add_argument(
        "--seed",
        type    = int,
        default = 42,
        help    = "Random seed for the evaluation environment.",
    )
    return parser.parse_args()


# ============================================================================
# Evaluation loop
# ============================================================================

def evaluate(model_path: str, num_episodes: int, record: bool, seed: int) -> None:
    """
    Load a trained DQN agent and run evaluation episodes.

    Args:
        model_path   : Path to the .pth checkpoint file.
        num_episodes : Number of episodes to evaluate.
        record       : Whether to save a video of the evaluation.
        seed         : Environment random seed.
    """
    print("\n" + "=" * 60)
    print("   DQN CartPole-v1 – Evaluation")
    print("=" * 60)

    # ── Verify checkpoint ────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        print(f"\n[ERROR] Model checkpoint not found: {model_path}")
        print("  → Run 'python training/train.py' first to train the agent.")
        sys.exit(1)

    # ── Initialise env + agent ───────────────────────────────────────────────
    env   = CartPoleEnv(seed=seed)
    agent = DQNAgent(
        state_size  = env.state_size,
        action_size = env.action_size,
        seed        = seed,
    )
    agent.load(model_path)

    # ── Run evaluation episodes ──────────────────────────────────────────────
    rewards = []

    print(f"\nRunning {num_episodes} evaluation episodes (ε = 0.0 – fully greedy):\n")
    print(f"  {'Episode':>8}  {'Reward':>8}  {'Steps':>6}")
    print("  " + "-" * 28)

    for ep in range(1, num_episodes + 1):
        state      = env.reset()
        ep_reward  = 0.0
        steps      = 0
        done       = False

        while not done:
            action = agent.select_action(state, epsilon=0.0)    # Greedy policy
            next_state, reward, done, _ = env.step(action)
            state      = next_state
            ep_reward += reward
            steps     += 1

        rewards.append(ep_reward)
        print(f"  {ep:>8}  {ep_reward:>8.1f}  {steps:>6}")

    env.close()

    # ── Summary statistics ───────────────────────────────────────────────────
    arr = np.array(rewards)
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Episodes evaluated : {num_episodes}")
    print(f"  Mean reward        : {arr.mean():.2f}")
    print(f"  Std  reward        : {arr.std():.2f}")
    print(f"  Min  reward        : {arr.min():.1f}")
    print(f"  Max  reward        : {arr.max():.1f}")
    print(f"  Solved (≥475)      : {(arr >= 475).sum()}/{num_episodes} episodes")
    print("=" * 60 + "\n")

    # ── Optional video recording ─────────────────────────────────────────────
    if record:
        print("[Evaluation] Recording a video of one greedy episode ...")
        recorder = VideoRecorder(save_dir="results/videos")
        recorder.record_episode(agent, episode_number=9999, seed=seed)
        print("[Evaluation] Video saved.\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path   = args.model,
        num_episodes = args.episodes,
        record       = args.record,
        seed         = args.seed,
    )
