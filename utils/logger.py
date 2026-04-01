"""
logger.py
=========
CSV-based training logger for the DQN CartPole experiment.

Records one row per episode containing:
    Episode | Reward | Average Reward | Epsilon | Loss

All data is written incrementally (no need to hold everything in memory)
so the file is valid even if training is interrupted.
"""

import csv
import os
from datetime import datetime
from pathlib import Path


# Column names for the CSV log
LOG_COLUMNS = ["Episode", "Reward", "Average_Reward", "Epsilon", "Loss"]


class TrainingLogger:
    """
    Lightweight CSV logger for training metrics.

    Creates the output directory and file automatically.
    Appends each episode's metrics to the CSV as training progresses.

    Args:
        log_dir  (str | Path): Directory where the CSV file will be saved.
        filename (str)        : Name of the CSV file (default: training_log.csv).
    """

    def __init__(self, log_dir: str = "results/logs", filename: str = "training_log.csv"):
        self.log_dir  = Path(log_dir)
        self.log_path = self.log_dir / filename

        # Ensure the directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Write header row
        self._init_csv()

        # Session start time (for reporting)
        self.start_time = datetime.now()

        print(f"[Logger] Logging to: {self.log_path.resolve()}")

    # ------------------------------------------------------------------
    # CSV management
    # ------------------------------------------------------------------

    def _init_csv(self) -> None:
        """Write the header row to a fresh CSV file."""
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()

    def log(
        self,
        episode:        int,
        reward:         float,
        avg_reward:     float,
        epsilon:        float,
        loss:           float,
    ) -> None:
        """
        Append one row of metrics to the log file.

        Args:
            episode    : Current episode number (1-indexed).
            reward     : Total reward obtained in this episode.
            avg_reward : Rolling average reward (last 100 episodes).
            epsilon    : Current exploration rate.
            loss       : Mean training loss for this episode (0.0 if no training).
        """
        row = {
            "Episode":        episode,
            "Reward":         round(reward,     2),
            "Average_Reward": round(avg_reward, 2),
            "Epsilon":        round(epsilon,    5),
            "Loss":           round(loss,       6) if loss else 0.0,
        }

        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Console helpers
    # ------------------------------------------------------------------

    def print_episode(
        self,
        episode:    int,
        total:      int,
        reward:     float,
        avg_reward: float,
        epsilon:    float,
        loss:       float,
    ) -> None:
        """Print a formatted training summary line to the terminal."""
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split(".")[0]   # Drop microseconds

        print(
            f"Ep {episode:>4}/{total} | "
            f"Reward: {reward:>6.1f} | "
            f"Avg(100): {avg_reward:>6.1f} | "
            f"ε: {epsilon:.4f} | "
            f"Loss: {loss:.4f} | "
            f"Time: {elapsed_str}"
        )

    def print_summary(
        self,
        total_episodes:  int,
        best_reward:     float,
        final_avg:       float,
    ) -> None:
        """Print a final training summary banner."""
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split(".")[0]

        print("\n" + "=" * 60)
        print("          TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Total Episodes       : {total_episodes}")
        print(f"  Best Episode Reward  : {best_reward:.1f}")
        print(f"  Final Avg Reward     : {final_avg:.1f}")
        print(f"  Total Training Time  : {elapsed_str}")
        print(f"  Log saved to         : {self.log_path.resolve()}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"TrainingLogger(path={self.log_path})"
