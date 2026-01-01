"""Drawing helpers for the HUD overlay (BGR colors)."""

from __future__ import annotations

import cv2
import numpy as np

CYAN = (255, 220, 0)
GREEN = (80, 255, 80)
RED = (60, 60, 255)
YELLOW = (0, 220, 255)
MAGENTA = (255, 80, 255)
ORANGE = (0, 160, 255)
WHITE = (240, 240, 240)
BLACK = (0, 0, 0)

ROLE_COLORS = {"owner": GREEN, "allowed": CYAN, "unknown": RED, "pending": YELLOW}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def pt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def corner_box(img: np.ndarray, x1: float, y1: float, x2: float, y2: float, color, thickness: int = 2,
               frac: float = 0.22) -> None:
    """Box drawn as four corner brackets plus a faint full outline."""
    x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
    lx = max(6, int((x2 - x1) * frac))
    ly = max(6, int((y2 - y1) * frac))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * lx, cy), color, thickness + 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * ly), color, thickness + 1, cv2.LINE_AA)


def label(img: np.ndarray, text: str, org, color, scale: float = 0.5, above: bool = True) -> None:
    """Text on a dark tag, clamped inside the frame."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = pt(org)
    h, w = img.shape[:2]
    if above:
        y = y - 4
    else:
        y = y + th + 6
    x = min(max(0, x), max(0, w - tw - 6))
    y = min(max(th + 4, y), h - base - 2)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + base), BLACK, -1)
    cv2.rectangle(img, (x, y - th - 4), (x, y + base), color, 2)
    cv2.putText(img, text, (x + 4, y - 1), FONT, scale, color, 1, cv2.LINE_AA)


def glow_line(img: np.ndarray, p1, p2, color, thickness: int = 3) -> None:
    cv2.line(img, pt(p1), pt(p2), tuple(int(c * 0.35) for c in color), thickness + 6, cv2.LINE_AA)
    cv2.line(img, pt(p1), pt(p2), color, thickness, cv2.LINE_AA)
