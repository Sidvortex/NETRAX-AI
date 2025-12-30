"""Tiny IoU tracker so a face keeps the same id across frames.

Recognition runs once per new track (plus a periodic recheck) instead of on
every frame, which is what makes face recognition cheap enough for 12+ FPS.
"""

from __future__ import annotations

import numpy as np


def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


class IouTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 8):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: dict[int, dict] = {}   # id -> {"bbox", "missed", "age"}
        self._next_id = 1
        self.ended: list[int] = []          # ids dropped during the last update

    def reset(self) -> None:
        self.tracks.clear()
        self.ended = []

    def update(self, boxes: list[np.ndarray]) -> list[int]:
        """Assign a track id to each box (greedy by IoU). Returns ids in input order."""
        self.ended = []
        pairs = []
        for ti, t in self.tracks.items():
            for bi, b in enumerate(boxes):
                iou = iou_xywh(t["bbox"], b)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, bi))
        pairs.sort(reverse=True)
        ids = [-1] * len(boxes)
        used_t: set[int] = set()
        for _, ti, bi in pairs:
            if ti in used_t or ids[bi] != -1:
                continue
            ids[bi] = ti
            used_t.add(ti)
        for bi, b in enumerate(boxes):
            if ids[bi] == -1:
                ids[bi] = self._next_id
                self.tracks[self._next_id] = {"bbox": b, "missed": 0, "age": 0}
                self._next_id += 1
            else:
                t = self.tracks[ids[bi]]
                t["bbox"], t["missed"] = b, 0
            self.tracks[ids[bi]]["age"] += 1
        for ti in list(self.tracks):
            if ti not in ids:
                self.tracks[ti]["missed"] += 1
                if self.tracks[ti]["missed"] > self.max_missed:
                    del self.tracks[ti]
                    self.ended.append(ti)
        return ids
