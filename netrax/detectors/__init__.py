"""Detector plugins.

`REGISTRY` maps a module name (the key under `modules:` in config.yaml) to the
class implementing it. Order matters: modules run in this order, so a module
can read the results of the ones before it (face_recog needs face_detect).
"""

from __future__ import annotations

import importlib

REGISTRY: dict[str, str] = {
    "face_detect": "netrax.detectors.face_detect:FaceDetector",
    "face_recog": "netrax.detectors.face_recog:FaceRecognizer",
    "head_pose": "netrax.detectors.head_pose:HeadPoseEstimator",
}

# draw order: big/background things first, face labels last so they stay readable
DRAW_ORDER = ("objects", "pose", "hands", "face_detect", "face_recog", "head_pose", "emotion")


def load_class(name: str):
    mod, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)
