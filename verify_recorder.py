import sys
import os
import numpy as np
import torch
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.video_recorder import VideoRecorder

class MockAgent:
    def select_action(self, state, epsilon=0.0):
        # Always push right (action 1)
        return 1
    @property
    def epsilon(self):
        return 0.05

def verify():
    print("Verification Script — Video Recorder (v5)")
    print("-" * 40)
    
    # Check dependencies
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV: Missing (Overlays will NOT show)")
        return

    try:
        import imageio
        print(f"✓ imageio: {imageio.__version__}")
    except ImportError:
        print("✗ imageio: Missing")
        return

    recorder = VideoRecorder(save_dir="results/verification_videos", fps=30)
    agent = MockAgent()
    
    print("\nRecording a TEST episode (Baseline ε=1.0) ...")
    path = recorder.record_episode(agent, episode_number=0, seed=42, record_epsilon=1.0)
    
    if path and os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"\n✓ SUCCESS: Video saved at {path}")
        print(f"✓ File size: {size:.1f} KB")
        print("\nACTION: Open this video and check for:")
        print(" 1. Top bar: 'Angle: X.X / LIMIT: ±12.0'")
        print(" 2. Gauge: '±12 FAIL' and '±0 STABLE' regions")
        print(" 3. Boundary dashed lines and glowing danger zones")
    else:
        print("\n✗ FAILURE: Video not saved.")

if __name__ == "__main__":
    verify()
