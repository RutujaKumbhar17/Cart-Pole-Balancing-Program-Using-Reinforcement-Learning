"""
video_recorder.py  (v4 — Live Pole Recoloring + Boundary Zones)
================================================================
Renders CartPole frames with two major visual enhancements:

  1. POLE RECOLORING
     The pole is redrawn over the gymnasium render with a solid color
     that reflects the current danger level:
        GREEN  — safe        (|angle| ≤ 0.10 rad,  |cart| ≤ 1.5 m)
        YELLOW — caution     (|angle| ≤ 0.16 rad,  |cart| ≤ 2.0 m)
        RED    — about to fail (|angle| > 0.16 rad OR |cart| > 2.0 m)

  2. BOUNDARY DANGER ZONES
     The physical constraint boundaries map to the screen edges:
        ±2.4 m  →  column 0 and column 599  (left / right screen edges)
     Visible danger zones are drawn as semi-transparent colored strips.
     As the cart nears the wall, the strip glows brighter red/orange.

  3. OVERLAYS  (same as v3)
     • Top status bar   — Ep / Step / Reward / ε
     • Pole angle gauge — bottom bar, colour-coded
     • FAILED banner    — terminal frame with exact constraint description
     • SOLVED banner    — when agent survives 500 steps

Recording schedule: every 60 episodes (set by CONFIG["video_interval"])
Backend fallback:  imageio-ffmpeg → OpenCV XVID → OpenCV mp4v
"""

import os
import math
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym

# ── Backend availability ─────────────────────────────────────────────────────
try:
    import imageio
    imageio.plugins.ffmpeg.download   # ensures ffmpeg plugin exists
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CartPole rendering constants  (derived from gymnasium CartPole source)
# ═══════════════════════════════════════════════════════════════════════════════

# Screen dimensions (gymnasium CartPole uses these exact values)
SCREEN_W     = 600
SCREEN_H     = 400

# Physical limits
CART_LIMIT   = 2.4          # metres
POLE_LIMIT   = 12.0         # degrees
POLE_RAD     = math.radians(POLE_LIMIT)  # 0.2094 rad

# Pixel / metre conversion
# world visible = CART_LIMIT * 2 = 4.8 m  across SCREEN_W pixels
SCALE        = SCREEN_W / (CART_LIMIT * 2)   # 125 px / m

# Axle position (vertical row in rgb_array)
#   carty     = 100  (from screen BOTTOM in gymnasium)
#   axleoffset= 7.5
#   => row from top = SCREEN_H - (100 + 7.5) ≈ 292
AXLE_ROW     = int(SCREEN_H - (100 + 7.5))   # ≈ 292

# Pole length in pixels
#   pole physical length = 0.5 m × 2 (full stick) = 1.0 m
POLE_PX      = int(SCALE * 1.0)              # 125 px
POLE_THICK   = 12                            # drawing width in pixels

# Danger zone thresholds
POLE_SAFE    = 0.10   # rad  (~5.7°)
POLE_CAUTION = 0.16   # rad  (~9.2°)   > 76 % of limit → flash red soon
CART_SAFE    = 1.50   # m   (62.5 % of boundary)
CART_CAUTION = 2.00   # m   (83.3 % of boundary)

# Column where the physical boundaries map to
# x = ±2.4 m  →  col = ±2.4 * 125 + 300 = 0 / 600  (i.e., screen edges)
BOUND_LEFT_COL  = 0
BOUND_RIGHT_COL = SCREEN_W - 1


# ═══════════════════════════════════════════════════════════════════════════════
# Colour palette  (BGR for OpenCV)
# ═══════════════════════════════════════════════════════════════════════════════

CLR_GREEN    = (  0, 220,  60)   # safe pole
CLR_YELLOW   = (  0, 220, 220)   # caution pole
CLR_RED      = (  0,  40, 220)   # dangerous pole / fail banner
CLR_ORANGE   = (  0, 140, 255)   # boundary warning colour
CLR_WHITE    = (255, 255, 255)
CLR_BLACK    = (  0,   0,   0)
CLR_DARK     = ( 25,  25,  25)   # status bar background
CLR_GRAY     = (140, 140, 140)
CLR_TRACK    = ( 60,  60,  60)   # gauge track


# ═══════════════════════════════════════════════════════════════════════════════
# State → pixel helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cart_col(cart_x: float) -> int:
    """Convert cart centre position (metres) to pixel column."""
    return int(cart_x * SCALE + SCREEN_W / 2.0)


def _pole_pixels(cart_x: float, pole_angle: float) -> Tuple[Tuple, Tuple]:
    """
    Return (axle_pt, tip_pt) in (col, row) pixel coords for the pole.

    In the rgb_array output:
        row 0 = top of image
        axle is at AXLE_ROW, directly above the cart centre
        positive pole angle = tilt to the RIGHT (column increases)
    """
    axle_col = _cart_col(cart_x)
    axle_pt  = (axle_col, AXLE_ROW)
    tip_col  = axle_col + int(POLE_PX * math.sin(pole_angle))
    tip_row  = AXLE_ROW - int(POLE_PX * math.cos(pole_angle))
    tip_pt   = (tip_col, tip_row)
    return axle_pt, tip_pt


def _danger_level(cart_x: float, pole_angle: float) -> int:
    """
    Compute danger level from 0 (safe) to 2 (danger).
    Considers BOTH cart position and pole angle.
    """
    cart_abs  = abs(cart_x)
    angle_abs = abs(pole_angle)

    if angle_abs > POLE_CAUTION or cart_abs > CART_CAUTION:
        return 2   # RED  — about to fail
    elif angle_abs > POLE_SAFE or cart_abs > CART_SAFE:
        return 1   # YELLOW — caution
    else:
        return 0   # GREEN  — safe


def _pole_color(danger: int):
    """Return BGR colour for the given danger level."""
    return (CLR_GREEN, CLR_YELLOW, CLR_RED)[danger]


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_pole(frame_bgr: np.ndarray, state: np.ndarray) -> np.ndarray:
    """
    Overdraw the gymnasium-rendered pole with a solid coloured line.

    The colour reflects the danger level:
        GREEN  = stable / safe
        YELLOW = approaching constraint
        RED    = about to fail

    Also draws:
        • Black outline (2px wider) for contrast against any background
        • Small white pivot circle at the axle
        • Small filled circle at the pole tip
    """
    cart_x      = float(state[0])
    pole_angle  = float(state[2])
    danger      = _danger_level(cart_x, pole_angle)
    color       = _pole_color(danger)

    axle_pt, tip_pt = _pole_pixels(cart_x, pole_angle)

    # Black outline for contrast
    cv2.line(frame_bgr, axle_pt, tip_pt, CLR_BLACK, POLE_THICK + 4, cv2.LINE_AA)
    # Coloured pole
    cv2.line(frame_bgr, axle_pt, tip_pt, color,     POLE_THICK,     cv2.LINE_AA)

    # Pivot circle (white ring with coloured fill)
    cv2.circle(frame_bgr, axle_pt, 7, CLR_BLACK, -1, cv2.LINE_AA)
    cv2.circle(frame_bgr, axle_pt, 5, CLR_WHITE, -1, cv2.LINE_AA)

    # Pole tip circle
    cv2.circle(frame_bgr, tip_pt, 5, CLR_BLACK, -1, cv2.LINE_AA)
    cv2.circle(frame_bgr, tip_pt, 3, color,      -1, cv2.LINE_AA)

    return frame_bgr


def _draw_boundary_zones(frame_bgr: np.ndarray, cart_x: float) -> np.ndarray:
    """
    Draw left / right danger-zone overlays that glow brighter as the
    cart approaches the physical boundary.

    Zone width = 80 px (= 0.64 m of the screen).
    Opacity scales from 0.15 (far away) to 0.60 (very close to boundary).

    Also draws a dashed vertical line exactly at each boundary column (0 / 599),
    and a small horizontal cart-position indicator bar at the very bottom.
    """
    h, w = frame_bgr.shape[:2]
    ZONE_W = 80    # danger zone strip width in pixels

    # ── Compute proximity-based opacity for each wall ──────────────────────
    left_prox  = max(0.0, (cart_x - (-CART_LIMIT)) / (ZONE_W / SCALE))  # 0=far, 1=wall
    right_prox = max(0.0, ((CART_LIMIT) - cart_x)  / (ZONE_W / SCALE))

    left_prox  = min(1.0, 1.0 - (left_prox  / 1.0))   # invert: closer = higher
    right_prox = min(1.0, 1.0 - (right_prox / 1.0))

    def _zone_color(prox: float):
        """Interpolate between orange (far) and red (close)."""
        r = int(255)
        g = int(140 * (1 - prox))   # fade green channel out as prox→1
        return (0, g, r)             # BGR

    def _draw_zone(frame, x1, x2, prox):
        if x2 <= x1 or prox <= 0.01:
            return frame
        alpha   = 0.15 + prox * 0.50
        overlay = frame.copy()
        color   = _zone_color(prox)
        cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Left zone
    frame_bgr = _draw_zone(frame_bgr, 0, ZONE_W, left_prox)
    # Right zone
    frame_bgr = _draw_zone(frame_bgr, w - ZONE_W, w, right_prox)

    # ── Hard boundary lines (always visible, dashed orange) ────────────────
    def _dashed_vline(img, x, color, dash=12, gap=7, thick=2):
        y, toggle = 0, True
        while y < h:
            end_y = min(y + dash, h)
            if toggle:
                cv2.line(img, (x, y), (x, end_y), color, thick)
            y     += dash if toggle else gap
            toggle = not toggle

    _dashed_vline(frame_bgr, 2,   CLR_ORANGE, thick=2)
    _dashed_vline(frame_bgr, w-3, CLR_ORANGE, thick=2)

    # Boundary labels
    cv2.putText(frame_bgr, f"-{CART_LIMIT}m  BOUNDARY",
                (5, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_ORANGE, 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"BOUNDARY  +{CART_LIMIT}m",
                (w - 148, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_ORANGE, 1, cv2.LINE_AA)

    # ── Cart-position progress bar (very bottom strip) ─────────────────────
    bar_y   = h - 6
    bar_h   = 4
    # Full bar (dark background)
    cv2.rectangle(frame_bgr, (0, bar_y), (w, bar_y + bar_h), CLR_DARK, -1)
    # Filled portion: maps cart_x from [-2.4, 2.4] → [0, w]
    norm    = (cart_x + CART_LIMIT) / (2 * CART_LIMIT)   # 0.0 – 1.0
    norm    = max(0.0, min(1.0, norm))
    bar_px  = int(norm * w)
    danger  = _danger_level(cart_x, 0.0)
    bar_clr = _pole_color(danger)
    cv2.rectangle(frame_bgr, (0, bar_y), (bar_px, bar_y + bar_h), bar_clr, -1)
    # Centre tick
    cv2.line(frame_bgr, (w // 2, bar_y - 1), (w // 2, bar_y + bar_h + 1), CLR_WHITE, 1)

    return frame_bgr


def _draw_status_bar(frame_bgr: np.ndarray, episode: int, step: int,
                     reward: float, epsilon: float,
                     danger: int, state: np.ndarray) -> np.ndarray:
    """Dark top bar with episode stats. Background color hints danger level."""
    h, w = frame_bgr.shape[:2]
    bar_h = 32

    # Get current angle and cart position for display
    cart_x     = float(state[0])
    pole_angle = math.degrees(float(state[2]))

    bg_color = {0: (20, 20, 20), 1: (0, 50, 80), 2: (0, 20, 80)}[danger]
    overlay  = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), bg_color, -1)
    frame_bgr = cv2.addWeighted(overlay, 0.85, frame_bgr, 0.15, 0)

    # Status text with Current Angle and Failure Limit
    text = (
        f"Ep {episode:04d} | Step {step:03d} | Reward {reward:5.1f} | eps {epsilon:.3f}"
    )
    cv2.putText(frame_bgr, text, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, CLR_WHITE, 1, cv2.LINE_AA)

    # Angle & Limit Display (Scientific Mark)
    angle_text = f"Angle: {pole_angle:+.1f} / LIMIT: ±12.0"
    tw         = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0]
    cv2.putText(frame_bgr, angle_text, (w - tw - 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 255, 255), 1, cv2.LINE_AA)

    return frame_bgr


def _draw_pole_gauge(frame_bgr: np.ndarray, pole_angle_rad: float) -> np.ndarray:
    """Horizontal pole-angle gauge at the bottom of the frame with Scientific Legend."""
    h, w = frame_bgr.shape[:2]

    bar_y  = h - 22
    bar_h  = 10
    margin = 85
    bar_l  = margin
    bar_r  = w - margin
    bar_w  = bar_r - bar_l
    center = (bar_l + bar_r) // 2

    # Track background
    cv2.rectangle(frame_bgr, (bar_l, bar_y), (bar_r, bar_y + bar_h), CLR_TRACK, -1)
    cv2.rectangle(frame_bgr, (bar_l, bar_y), (bar_r, bar_y + bar_h), CLR_GRAY,  1)

    # Stability Visual Regions (Scientific Marks)
    # 0-6 deg: Stable (Green) | 6-9 deg: Caution (Yellow) | 9-12 deg: Danger (Red)
    s_limit  = math.radians(6)
    c_limit  = math.radians(9)
    f_limit  = math.radians(12) # POLE_RAD

    def _angle_to_px(rad):
        return center + int((rad / f_limit) * (bar_w / 2))

    # Draw colored stability zones beneath the gauge
    # Stable ±6
    # Caution ±6-9
    # Danger ±9-12
    cv2.rectangle(frame_bgr, (_angle_to_px(-s_limit), bar_y), (_angle_to_px(s_limit), bar_y+bar_h), (0, 80, 0), -1)
    cv2.rectangle(frame_bgr, (_angle_to_px(-c_limit), bar_y), (_angle_to_px(-s_limit), bar_y+bar_h), (0, 80, 80), -1)
    cv2.rectangle(frame_bgr, (_angle_to_px(s_limit), bar_y), (_angle_to_px(c_limit), bar_y+bar_h), (0, 80, 80), -1)
    cv2.rectangle(frame_bgr, (_angle_to_px(-f_limit), bar_y), (_angle_to_px(-c_limit), bar_y+bar_h), (0, 0, 80), -1)
    cv2.rectangle(frame_bgr, (_angle_to_px(c_limit), bar_y), (_angle_to_px(f_limit), bar_y+bar_h), (0, 0, 80), -1)

    # Centre zero tick
    cv2.line(frame_bgr, (center, bar_y-2), (center, bar_y+bar_h+2), CLR_WHITE, 1)

    # Angle fill indicator
    frac     = max(-1.0, min(1.0, pole_angle_rad / f_limit))
    fill_px  = int(frac * (bar_w / 2))
    danger   = _danger_level(0.0, pole_angle_rad)
    gauge_c  = _pole_color(danger)
    x1       = center if fill_px >= 0 else center + fill_px
    x2       = center + fill_px if fill_px >= 0 else center
    if x2 > x1:
        cv2.rectangle(frame_bgr, (x1, bar_y+2), (x2, bar_y+bar_h-2), gauge_c, -1)

    # Labels for stability zones
    angle_deg = math.degrees(pole_angle_rad)
    cv2.putText(frame_bgr, "±12 FAIL", (bar_r + 5, bar_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_RED, 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, "±0 STABLE", (center - 22, bar_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_WHITE, 1, cv2.LINE_AA)
    
    cv2.putText(frame_bgr, f"{angle_deg:+.1f} deg",
                (center - 22, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, gauge_c, 1, cv2.LINE_AA)

    return frame_bgr


def _draw_fail_banner(frame_bgr: np.ndarray, reason: str) -> np.ndarray:
    """Large red failure banner on the terminal frame."""
    h, w = frame_bgr.shape[:2]

    # Red screen tint
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 100), -1)
    frame_bgr = cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0)

    bx1, by1 = int(w*0.05), int(h*0.3)
    bx2, by2 = int(w*0.95), int(h*0.70)
    cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 0, 160), -1)
    cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), CLR_RED,      3)

    cv2.putText(frame_bgr, "EPISODE FAILED",
                (bx1+18, by1+44), cv2.FONT_HERSHEY_DUPLEX, 1.05, CLR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, reason,
                (bx1+14, by1+76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (110, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, "Pole was RED just before failure",
                (bx1+14, by1+100), cv2.FONT_HERSHEY_SIMPLEX, 0.44, CLR_GRAY, 1, cv2.LINE_AA)

    return frame_bgr


def _draw_success_banner(frame_bgr: np.ndarray, steps: int) -> np.ndarray:
    """Large green solved banner on the terminal frame."""
    h, w = frame_bgr.shape[:2]

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 70, 0), -1)
    frame_bgr = cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0)

    bx1, by1 = int(w*0.05), int(h*0.3)
    bx2, by2 = int(w*0.95), int(h*0.70)
    cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 70, 0),   -1)
    cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), CLR_GREEN,     3)

    cv2.putText(frame_bgr, "EPISODE SOLVED!",
                (bx1+18, by1+44), cv2.FONT_HERSHEY_DUPLEX, 1.05, CLR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Survived all {steps} steps — pole stayed GREEN",
                (bx1+14, by1+76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 255, 150), 1, cv2.LINE_AA)

    return frame_bgr


def _annotate(frame_rgb: np.ndarray, state: np.ndarray,
              episode: int, step: int, reward: float, epsilon: float) -> np.ndarray:
    """
    Apply all visual overlays to one RGB frame.
    Converts RGB → BGR for OpenCV, draws, converts back to RGB.
    """
    if not CV2_AVAILABLE:
        return frame_rgb

    cart_x     = float(state[0])
    pole_angle = float(state[2])
    danger     = _danger_level(cart_x, pole_angle)

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    bgr = _draw_boundary_zones(bgr, cart_x)   # semi-transparent danger strips + dashed lines
    bgr = _draw_pole(bgr, state)               # recolour pole GREEN / YELLOW / RED
    bgr = _draw_pole_gauge(bgr, pole_angle)    # bottom angle bar
    bgr = _draw_status_bar(bgr, episode, step, reward, epsilon, danger, state)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════════════════════
# VideoRecorder
# ═══════════════════════════════════════════════════════════════════════════════

class VideoRecorder:
    """
    Records annotated CartPole episodes.

    The POLE is redrawn every frame in GREEN / YELLOW / RED to show
    whether the agent is stable or about to violate a constraint.

    Danger-zone strips glow on whichever boundary the cart is nearing.
    The terminal frame shows a clear FAILED or SOLVED banner.

    Args:
        save_dir (str | Path): Output directory for video files.
        fps      (int)       : Frames per second (default 30).
    """

    ENV_NAME = "CartPole-v1"

    def __init__(self, save_dir: str = "results/videos", fps: int = 30):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        if IMAGEIO_AVAILABLE:
            self._backend = "imageio"
        elif CV2_AVAILABLE:
            self._backend = "opencv"
        else:
            self._backend = "none"
            print(
                "[VideoRecorder] WARNING: No video backend available.\n"
                "  pip install imageio imageio-ffmpeg opencv-python"
            )

        print(f"[VideoRecorder] Backend={self._backend} | SaveDir={self.save_dir.resolve()}")
        if not CV2_AVAILABLE:
            print("[VideoRecorder] !!! CRITICAL WARNING: opencv-python missing. No overlays will be shown !!!")
        else:
            print("[VideoRecorder] ✓ OpenCV found — Annotation engine INITIALISED.")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def record_episode(
        self,
        agent,
        episode_number:  int,
        seed:            int   = 42,
        record_epsilon:  float = None,
    ) -> Optional[str]:
        """
        Record one annotated episode.

        Args:
            agent          : DQNAgent with select_action(state, epsilon).
            episode_number : File naming.
            seed           : Env reset seed.
            record_epsilon : Override epsilon; None → use agent.epsilon.

        Returns:
            Absolute path to saved file, or None on failure.
        """
        if self._backend == "none":
            print(f"[VideoRecorder] Skipping ep {episode_number} (no backend).")
            return None

        eps = record_epsilon if record_epsilon is not None else agent.epsilon
        print(f"\n[VideoRecorder] Recording Ep {episode_number}  (ε={eps:.3f}) ...")

        frames, meta = self._collect(agent, seed, eps, episode_number)

        if not frames:
            print("[VideoRecorder] No frames — skipping.")
            return None

        outcome = "FAILED" if meta["failed"] else "SOLVED"
        print(
            f"[VideoRecorder] {len(frames)} frames | "
            f"reward={meta['reward']:.1f} | {outcome}"
        )

        path = self._save(frames, episode_number)
        if path:
            kb = Path(path).stat().st_size // 1024
            print(f"[VideoRecorder] ✓ {Path(path).name}  ({kb} KB)\n")
        else:
            print(f"[VideoRecorder] ✗ All backends failed for ep {episode_number}.\n")

        return path

    # ─────────────────────────────────────────────────────────────────────────
    # Frame collection
    # ─────────────────────────────────────────────────────────────────────────

    def _collect(
        self, agent, seed: int, epsilon: float, episode: int
    ) -> Tuple[list, dict]:
        """Run one episode, annotate every frame, return (frames, meta)."""
        env    = None
        frames = []
        meta   = {"reward": 0.0, "failed": False, "reason": "", "steps": 0}

        try:
            env = gym.make(self.ENV_NAME, render_mode="rgb_array")
            state, _ = env.reset(seed=seed)
            state    = state.astype(np.float32)

            done  = False
            step  = 0
            total = 0.0

            while not done:
                # Render BEFORE action so we capture the current state visually
                raw = env.render()
                if raw is not None:
                    annotated = _annotate(raw.copy(), state, episode, step, total, epsilon)
                    frames.append(annotated)

                action = agent.select_action(state, epsilon=epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done   = terminated or truncated
                state  = next_state.astype(np.float32)
                total += reward
                step  += 1

            # ── Terminal frame ────────────────────────────────────────────────
            raw = env.render()
            if raw is not None and CV2_AVAILABLE:
                final_rgb = _annotate(raw.copy(), state, episode, step, total, epsilon)
                bgr       = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)

                if terminated:
                    cart_x     = float(state[0])
                    angle_deg  = math.degrees(float(state[2]))

                    if abs(cart_x) >= CART_LIMIT:
                        reason = (
                            f"Cart out of bounds:  {cart_x:+.3f} m  "
                            f"(limit ±{CART_LIMIT} m)"
                        )
                    elif abs(angle_deg) >= POLE_LIMIT:
                        reason = (
                            f"Pole angle exceeded: {angle_deg:+.1f}°  "
                            f"(limit ±{POLE_LIMIT}°)"
                        )
                    else:
                        reason = "Terminal state reached"

                    meta["failed"] = True
                    meta["reason"] = reason
                    bgr = _draw_fail_banner(bgr, reason)
                else:
                    bgr = _draw_success_banner(bgr, step)

                final_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # Hold terminal banner for ~1.5 s so it's clearly visible
                hold = int(self.fps * 1.5)
                for _ in range(hold):
                    frames.append(final_rgb.copy())
            elif raw is not None:
                frames.append(raw.copy())

            meta["reward"] = total
            meta["steps"]  = step

        except Exception as exc:
            print(f"[VideoRecorder] Frame collection error: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        # Pad to at least 1 second
        if 0 < len(frames) < self.fps:
            last = frames[-1]
            while len(frames) < self.fps:
                frames.append(last.copy())

        return frames, meta

    # ─────────────────────────────────────────────────────────────────────────
    # Backend save methods
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self, frames: list, ep: int) -> Optional[str]:
        if self._backend == "imageio":
            path = self._save_imageio(frames, ep)
            if path:
                return path
        return self._save_opencv(frames, ep)

    def _save_imageio(self, frames: list, ep: int) -> Optional[str]:
        out = self.save_dir / f"episode_{ep:04d}.mp4"
        try:
            writer = imageio.get_writer(
                str(out), fps=self.fps, codec="libx264", quality=7,
                output_params=["-pix_fmt", "yuv420p"],
            )
            for f in frames:
                writer.append_data(f)
            writer.close()
            if out.exists() and out.stat().st_size > 2000:
                return str(out.resolve())
        except Exception as exc:
            print(f"[VideoRecorder] imageio error: {exc}")
        return None

    def _save_opencv(self, frames: list, ep: int) -> Optional[str]:
        if not CV2_AVAILABLE:
            return None
        h, w = frames[0].shape[:2]

        # Try XVID → .avi  (most reliable on Windows)
        avi = self.save_dir / f"episode_{ep:04d}.avi"
        writer = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(avi), fourcc, float(self.fps), (w, h))
            if writer.isOpened():
                for f in frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                writer = None
                if avi.exists() and avi.stat().st_size > 2000:
                    return str(avi.resolve())
        except Exception as exc:
            print(f"[VideoRecorder] XVID error: {exc}")
        finally:
            if writer:
                try:
                    writer.release()
                except Exception:
                    pass

        # Try mp4v → .mp4
        mp4 = self.save_dir / f"episode_{ep:04d}.mp4"
        writer = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(mp4), fourcc, float(self.fps), (w, h))
            if writer.isOpened():
                for f in frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                writer = None
                if mp4.exists() and mp4.stat().st_size > 2000:
                    return str(mp4.resolve())
        except Exception as exc:
            print(f"[VideoRecorder] mp4v error: {exc}")
        finally:
            if writer:
                try:
                    writer.release()
                except Exception:
                    pass

        return None

    # ─────────────────────────────────────────────────────────────────────────

    def list_videos(self) -> list:
        return sorted(
            list(self.save_dir.glob("*.mp4")) +
            list(self.save_dir.glob("*.avi"))
        )

    def __repr__(self) -> str:
        return (
            f"VideoRecorder("
            f"backend={self._backend}, "
            f"fps={self.fps}, "
            f"dir={self.save_dir})"
        )
