"""Rule-based hand gestures from the 21 MediaPipe hand landmarks.

Landmark ids: 0 wrist | thumb 1-4 | index 5-8 | middle 9-12 | ring 13-16 | pinky 17-20
(for each finger: MCP, PIP, DIP, TIP; thumb: CMC, MCP, IP, TIP).
Only distances and angles between landmarks are used, so it works with the
hand rotated in the image plane.
"""

from __future__ import annotations

import numpy as np

GESTURES = ("open_palm", "fist", "thumbs_up", "point", "peace")
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def finger_states(lm: np.ndarray) -> dict[str, bool]:
    """True = extended, for thumb/index/middle/ring/pinky."""
    p = np.asarray(lm, float)[:, :2]
    wrist = p[0]
    palm = np.linalg.norm(p[9] - wrist) + 1e-9          # wrist -> middle MCP, the hand's scale
    out = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        straight = _cos(p[pip] - p[mcp], p[tip] - p[pip]) > 0.5
        far = np.linalg.norm(p[tip] - wrist) > np.linalg.norm(p[pip] - wrist) * 1.1
        out[name] = bool(straight and far)
    # thumb: tip away from the palm (index MCP) and the thumb roughly straight
    t_straight = _cos(p[3] - p[2], p[4] - p[3]) > 0.3
    t_out = np.linalg.norm(p[4] - p[5]) > 0.55 * palm
    out["thumb"] = bool(t_straight and t_out)
    return out


def classify(lm: np.ndarray) -> str:
    p = np.asarray(lm, float)[:, :2]
    f = finger_states(lm)
    four = [f["index"], f["middle"], f["ring"], f["pinky"]]
    palm = np.linalg.norm(p[9] - p[0]) + 1e-9
    if all(four) and f["thumb"]:
        return "open_palm"
    if not any(four):
        thumb_up = f["thumb"] and (p[2, 1] - p[4, 1]) > 0.45 * palm      # tip clearly above the thumb MCP
        return "thumbs_up" if thumb_up else "fist"
    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
        return "peace"
    if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        return "point"
    return "none"
