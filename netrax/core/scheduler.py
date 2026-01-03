"""Decides which modules run on which frame, and keeps per-module timing stats.

Heavy modules run every N frames; their last results are reused in between.
Each module gets a different phase so e.g. pose (every 2) and objects
(every 5) don't all land on the same frame and cause a stutter.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np


class ModuleStats:
    def __init__(self, window: int = 200):
        self.times_ms: deque[float] = deque(maxlen=window)
        self.run_ts: deque[float] = deque(maxlen=600)
        self.runs = 0
        self.errors = 0

    def record(self, ms: float) -> None:
        self.times_ms.append(ms)
        self.run_ts.append(time.monotonic())
        self.runs += 1

    def fps(self, horizon: float = 2.0) -> float:
        now = time.monotonic()
        n = sum(1 for t in self.run_ts if now - t <= horizon)
        return n / horizon

    def snapshot(self) -> dict:
        t = np.array(self.times_ms) if self.times_ms else np.zeros(1)
        return {
            "ms": round(float(t.mean()), 2),
            "p95_ms": round(float(np.percentile(t, 95)), 2),
            "fps": round(self.fps(), 1),
            "runs": self.runs,
            "errors": self.errors,
        }


class Scheduler:
    def __init__(self, order: list[str]):
        self.phase = {name: i for i, name in enumerate(order)}

    def should_run(self, name: str, every_n: int, frame_idx: int) -> bool:
        if every_n <= 1:
            return True
        return (frame_idx + self.phase.get(name, 0)) % every_n == 0


class RateCounter:
    """Events per second over a sliding window."""

    def __init__(self, horizon: float = 2.0):
        self.horizon = horizon
        self.ts: deque[float] = deque(maxlen=2000)

    def tick(self) -> None:
        self.ts.append(time.monotonic())

    @property
    def rate(self) -> float:
        now = time.monotonic()
        while self.ts and now - self.ts[0] > self.horizon:
            self.ts.popleft()
        return len(self.ts) / self.horizon
