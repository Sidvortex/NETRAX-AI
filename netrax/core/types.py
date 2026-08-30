"""Result types shared by detectors, the pipeline, the hub and outputs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

import numpy as np


def _jsonable(v: Any, nd: int = 3) -> Any:
    if isinstance(v, np.ndarray):
        return np.round(v.astype(float), nd).tolist()
    if isinstance(v, (np.floating, float)):
        return round(float(v), nd)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, nd) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x, nd) for k, x in v.items()}
    return v


@dataclass
class Result:
    kind: ClassVar[str] = "result"
    _private: ClassVar[tuple[str, ...]] = ()

    def to_dict(self) -> dict:
        out = {"kind": self.kind}
        for f in fields(self):
            if f.name.startswith("_") or f.name in self._private:
                continue
            out[f.name] = _jsonable(getattr(self, f.name))
        return out


@dataclass
class Face(Result):
    kind: ClassVar[str] = "face"
    _private: ClassVar[tuple[str, ...]] = ("raw",)
    bbox: np.ndarray            # x, y, w, h (pixels)
    score: float
    landmarks: np.ndarray       # 5x2: right eye, left eye, nose tip, right mouth, left mouth
    raw: np.ndarray             # YuNet row (15 floats), needed by SFace alignCrop
    track_id: int = -1

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return float(x + w / 2), float(y + h / 2)

    @property
    def area(self) -> float:
        return float(self.bbox[2] * self.bbox[3])


@dataclass
class Identity(Result):
    kind: ClassVar[str] = "identity"
    _private: ClassVar[tuple[str, ...]] = ("bbox",)
    track_id: int
    name: str                   # person name, "unknown", or "..." while still deciding
    role: str                   # owner | allowed | unknown
    confidence: float           # best cosine similarity
    verified: bool              # confidence >= threshold and the vote agrees
    person_id: int | None = None
    attempts: int = 0           # how many recognitions this track has had
    bbox: np.ndarray | None = None  # face box, for drawing only


@dataclass
class HeadPose(Result):
    kind: ClassVar[str] = "head_pose"
    _private: ClassVar[tuple[str, ...]] = ("axes",)
    track_id: int
    yaw: float                  # + = turned towards image right
    pitch: float                # + = looking up
    roll: float                 # + = head tilted clockwise in the image
    direction: str              # center | left | right | up | down (from the camera's view)
    axes: np.ndarray | None = None  # 4x2 projected points: origin, x, y, z


@dataclass
class Emotion(Result):
    kind: ClassVar[str] = "emotion"
    _private: ClassVar[tuple[str, ...]] = ("bbox",)
    track_id: int
    label: str
    score: float
    bbox: np.ndarray | None = None  # face box, for drawing only


@dataclass
class Hand(Result):
    kind: ClassVar[str] = "hand"
    bbox: np.ndarray            # x1, y1, x2, y2
    landmarks: np.ndarray       # 21x3 (x, y pixels; z relative to wrist)
    handedness: str             # left | right (as the person sees it)
    score: float
    gesture: str = "none"       # stable gesture (after smoothing)
    raw_gesture: str = "none"   # this frame's classification


@dataclass
class Pose(Result):
    kind: ClassVar[str] = "pose"
    bbox: np.ndarray            # x1, y1, x2, y2
    keypoints: np.ndarray       # 17x3 COCO order: x, y, visibility
    score: float
    posture: str = "unknown"    # upright | slouching | leaning | calibrating | unknown
    metrics: dict = field(default_factory=dict)


@dataclass
class Detection(Result):
    kind: ClassVar[str] = "object"
    bbox: np.ndarray            # x1, y1, x2, y2
    label: str
    class_id: int
    score: float


@dataclass
class FrameState:
    """Everything known about one processed frame."""

    frame_idx: int
    width: int
    height: int
    ts: float = field(default_factory=time.time)
    results: dict[str, list[Result]] = field(default_factory=dict)
    ran: set[str] = field(default_factory=set)       # modules that actually ran on this frame
    timings_ms: dict[str, float] = field(default_factory=dict)

    def get(self, module: str) -> list:
        return self.results.get(module, [])

    def faces(self) -> list[dict]:
        """Faces merged with identity, head pose and emotion by track id."""
        extra: dict[int, dict] = {}
        for module in ("face_recog", "head_pose", "emotion"):
            for r in self.get(module):
                d = r.to_dict()
                d.pop("kind", None)
                tid = d.pop("track_id")
                if module == "emotion":
                    d = {"emotion": d["label"], "emotion_score": d["score"]}
                extra.setdefault(tid, {}).update(d)
        out = []
        for f in self.get("face_detect"):
            d = f.to_dict()
            d.pop("kind")
            d.update(extra.get(f.track_id, {}))
            out.append(d)
        return out

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_idx,
            "ts": round(self.ts, 3),
            "size": [self.width, self.height],
            "faces": self.faces(),
            "hands": [r.to_dict() for r in self.get("hands")],
            "poses": [r.to_dict() for r in self.get("pose")],
            "objects": [r.to_dict() for r in self.get("objects")],
        }
