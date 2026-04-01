"""
plotting.py
===========
Professional matplotlib plotting utilities for DQN training results.

Generates four publication-quality plots saved as PNG files:
    1. Reward Learning Curve          – raw episode rewards
    2. Average Reward Curve           – rolling 100-episode mean
    3. Epsilon Decay Plot             – exploration rate over episodes
    4. Training Loss Curve            – mean TD loss per episode

All plots use a consistent dark-themed style with smooth curves and
clear axis labels suitable for a portfolio / academic report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

STYLE = {
    "figure.facecolor":  "#0f0f1a",
    "axes.facecolor":    "#1a1a2e",
    "axes.edgecolor":    "#3a3a5c",
    "axes.labelcolor":   "#e0e0ff",
    "xtick.color":       "#a0a0cc",
    "ytick.color":       "#a0a0cc",
    "text.color":        "#e0e0ff",
    "grid.color":        "#2a2a4a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "axes.titlesize":    14,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "font.family":       "DejaVu Sans",
}

ACCENT_BLUE   = "#4fc3f7"
ACCENT_CYAN   = "#00e5ff"
ACCENT_ORANGE = "#ff8a65"
ACCENT_GREEN  = "#69f0ae"
ACCENT_PURPLE = "#ce93d8"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_style():
    """Apply the global dark style dictionary to matplotlib rcParams."""
    plt.rcParams.update(STYLE)


def _rolling_mean(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute a rolling mean with edge padding (no NaN at the start)."""
    series = pd.Series(data)
    return series.rolling(window=window, min_periods=1).mean().values


def _save_and_close(fig, path: Path, filename: str) -> None:
    """Save the figure, print confirmation, and close it to free memory."""
    full_path = path / filename
    fig.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {full_path.resolve()}")


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_reward_curve(rewards: list, save_dir: str = "results/plots") -> None:
    """
    Plot the raw per-episode reward learning curve.

    Args:
        rewards  : List of episode reward values.
        save_dir : Directory to save the PNG file.
    """
    _apply_style()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("DQN CartPole – Reward Learning Curve", fontsize=16, color="#e0e0ff", y=1.01)

    # Raw rewards (semi-transparent)
    ax.plot(episodes, rewards, color=ACCENT_BLUE, alpha=0.35, linewidth=0.8, label="Episode Reward")

    # Smoothed trend
    smoothed = _rolling_mean(np.array(rewards), window=20)
    ax.plot(episodes, smoothed, color=ACCENT_CYAN, linewidth=2.0, label="Smoothed (20-ep)")

    # Target line
    ax.axhline(y=475, color=ACCENT_GREEN, linestyle="--", linewidth=1.5, alpha=0.8, label="Solved (475)")

    ax.set_xlabel("Episode", labelpad=8)
    ax.set_ylabel("Total Reward", labelpad=8)
    ax.legend(facecolor="#1a1a2e", edgecolor="#3a3a5c", labelcolor="#e0e0ff", fontsize=9)
    ax.grid(True)
    ax.set_xlim(1, len(rewards))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))

    plt.tight_layout()
    _save_and_close(fig, save_path, "01_reward_curve.png")


def plot_average_reward(rewards: list, save_dir: str = "results/plots") -> None:
    """
    Plot the 100-episode rolling average reward.

    Args:
        rewards  : List of episode reward values.
        save_dir : Directory to save the PNG file.
    """
    _apply_style()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    episodes   = np.arange(1, len(rewards) + 1)
    avg_reward = _rolling_mean(np.array(rewards), window=100)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("DQN CartPole – Average Reward (100-Episode Window)", fontsize=16, color="#e0e0ff", y=1.01)

    ax.fill_between(episodes, avg_reward, alpha=0.2, color=ACCENT_ORANGE)
    ax.plot(episodes, avg_reward, color=ACCENT_ORANGE, linewidth=2.2, label="Avg Reward (100-ep)")
    ax.axhline(y=475, color=ACCENT_GREEN, linestyle="--", linewidth=1.5, alpha=0.8, label="Solved (475)")

    ax.set_xlabel("Episode", labelpad=8)
    ax.set_ylabel("Average Reward", labelpad=8)
    ax.legend(facecolor="#1a1a2e", edgecolor="#3a3a5c", labelcolor="#e0e0ff", fontsize=9)
    ax.grid(True)
    ax.set_xlim(1, len(rewards))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))

    plt.tight_layout()
    _save_and_close(fig, save_path, "02_average_reward.png")


def plot_epsilon_decay(epsilons: list, save_dir: str = "results/plots") -> None:
    """
    Plot the epsilon (exploration rate) decay over episodes.

    Args:
        epsilons : List of epsilon values (one per episode).
        save_dir : Directory to save the PNG file.
    """
    _apply_style()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(epsilons) + 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("DQN CartPole – Epsilon Decay (ε-Greedy Exploration)", fontsize=16, color="#e0e0ff", y=1.01)

    ax.fill_between(episodes, epsilons, alpha=0.15, color=ACCENT_PURPLE)
    ax.plot(episodes, epsilons, color=ACCENT_PURPLE, linewidth=2.0, label="Epsilon")
    ax.axhline(y=0.01, color=ACCENT_CYAN, linestyle=":", linewidth=1.5, alpha=0.8, label="Min ε = 0.01")

    ax.set_xlabel("Episode", labelpad=8)
    ax.set_ylabel("Epsilon (ε)", labelpad=8)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(facecolor="#1a1a2e", edgecolor="#3a3a5c", labelcolor="#e0e0ff", fontsize=9)
    ax.grid(True)
    ax.set_xlim(1, len(epsilons))

    plt.tight_layout()
    _save_and_close(fig, save_path, "03_epsilon_decay.png")


def plot_loss_curve(losses: list, save_dir: str = "results/plots") -> None:
    """
    Plot the mean TD loss per episode.

    Args:
        losses   : List of mean loss values (one per episode, 0.0 if no training).
        save_dir : Directory to save the PNG file.
    """
    _apply_style()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Filter out zero-loss episodes (before buffer is ready)
    valid_mask = np.array(losses) > 0
    episodes   = np.arange(1, len(losses) + 1)[valid_mask]
    loss_vals  = np.array(losses)[valid_mask]

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("DQN CartPole – Training Loss Curve (MSE)", fontsize=16, color="#e0e0ff", y=1.01)

    ax.plot(episodes, loss_vals, color=ACCENT_BLUE, alpha=0.4, linewidth=0.8, label="Loss (raw)")
    smoothed = _rolling_mean(loss_vals, window=20)
    ax.plot(episodes, smoothed, color=ACCENT_CYAN, linewidth=2.0, label="Smoothed (20-ep)")

    ax.set_xlabel("Episode", labelpad=8)
    ax.set_ylabel("MSE Loss", labelpad=8)
    ax.legend(facecolor="#1a1a2e", edgecolor="#3a3a5c", labelcolor="#e0e0ff", fontsize=9)
    ax.grid(True)
    ax.set_xlim(episodes[0] if len(episodes) else 1, episodes[-1] if len(episodes) else 1)

    plt.tight_layout()
    _save_and_close(fig, save_path, "04_loss_curve.png")


# ---------------------------------------------------------------------------
# Combined plot (all four in one figure)
# ---------------------------------------------------------------------------

def plot_all(
    rewards:  list,
    epsilons: list,
    losses:   list,
    save_dir: str = "results/plots",
) -> None:
    """
    Render all four training plots in a single 2×2 figure.

    Args:
        rewards  : Episode reward list.
        epsilons : Epsilon list.
        losses   : Loss list.
        save_dir : Directory to save the PNG file.
    """
    _apply_style()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    episodes    = np.arange(1, len(rewards) + 1)
    avg_reward  = _rolling_mean(np.array(rewards), window=100)
    smooth_rew  = _rolling_mean(np.array(rewards), window=20)

    valid_mask  = np.array(losses) > 0
    ep_loss     = episodes[valid_mask]
    loss_vals   = np.array(losses)[valid_mask]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("DQN CartPole-v1 – Training Dashboard", fontsize=18, color="#e0e0ff", y=1.01)
    fig.patch.set_facecolor("#0f0f1a")

    # ── Panel 1: Reward Curve ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor("#1a1a2e")
    ax.plot(episodes, rewards,    color=ACCENT_BLUE,   alpha=0.35, linewidth=0.8)
    ax.plot(episodes, smooth_rew, color=ACCENT_CYAN,   linewidth=2.0, label="Smoothed")
    ax.axhline(y=475,             color=ACCENT_GREEN,  linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_title("Episode Reward",    color="#e0e0ff")
    ax.set_xlabel("Episode",          color="#a0a0cc")
    ax.set_ylabel("Reward",           color="#a0a0cc")
    ax.grid(True)

    # ── Panel 2: Average Reward ────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_facecolor("#1a1a2e")
    ax.fill_between(episodes, avg_reward, alpha=0.2, color=ACCENT_ORANGE)
    ax.plot(episodes, avg_reward, color=ACCENT_ORANGE, linewidth=2.2, label="Avg(100)")
    ax.axhline(y=475,             color=ACCENT_GREEN,  linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_title("Average Reward (100-ep)", color="#e0e0ff")
    ax.set_xlabel("Episode",               color="#a0a0cc")
    ax.set_ylabel("Avg Reward",            color="#a0a0cc")
    ax.grid(True)

    # ── Panel 3: Epsilon Decay ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor("#1a1a2e")
    ax.fill_between(episodes, epsilons, alpha=0.15, color=ACCENT_PURPLE)
    ax.plot(episodes, epsilons,  color=ACCENT_PURPLE, linewidth=2.0, label="Epsilon")
    ax.axhline(y=0.01,           color=ACCENT_CYAN,   linestyle=":", linewidth=1.2, alpha=0.8)
    ax.set_title("Epsilon Decay", color="#e0e0ff")
    ax.set_xlabel("Episode",      color="#a0a0cc")
    ax.set_ylabel("ε",            color="#a0a0cc")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True)

    # ── Panel 4: Loss Curve ────────────────────────────────────────────
    ax = axes[1, 1]
    ax.set_facecolor("#1a1a2e")
    if len(ep_loss) > 0:
        ax.plot(ep_loss, loss_vals,                      color=ACCENT_BLUE, alpha=0.4, linewidth=0.8)
        ax.plot(ep_loss, _rolling_mean(loss_vals, 20),   color=ACCENT_CYAN, linewidth=2.0, label="Smoothed")
    ax.set_title("Training Loss (MSE)", color="#e0e0ff")
    ax.set_xlabel("Episode",            color="#a0a0cc")
    ax.set_ylabel("MSE Loss",           color="#a0a0cc")
    ax.grid(True)

    for row in axes:
        for a in row:
            for spine in a.spines.values():
                spine.set_edgecolor("#3a3a5c")
            a.tick_params(colors="#a0a0cc")

    plt.tight_layout()
    _save_and_close(fig, save_path, "00_training_dashboard.png")


# ---------------------------------------------------------------------------
# Convenience wrapper: generate all individual + combined plots
# ---------------------------------------------------------------------------

def generate_all_plots(
    rewards:  list,
    epsilons: list,
    losses:   list,
    save_dir: str = "results/plots",
) -> None:
    """
    Generate all five plot files from training history.

    Args:
        rewards  : List of episode rewards.
        epsilons : List of epsilon values per episode.
        losses   : List of mean loss values per episode.
        save_dir : Output directory (created if absent).
    """
    print(f"\n[Plots] Generating plots in '{save_dir}' ...")
    plot_reward_curve(rewards,           save_dir=save_dir)
    plot_average_reward(rewards,         save_dir=save_dir)
    plot_epsilon_decay(epsilons,         save_dir=save_dir)
    plot_loss_curve(losses,              save_dir=save_dir)
    plot_all(rewards, epsilons, losses,  save_dir=save_dir)
    print("[Plots] All plots saved.\n")
