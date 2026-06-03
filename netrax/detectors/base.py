"""Common detector interface.

Every module is a plugin with the same three calls:

    load()                          -> load models (may download them first)
    process(frame, ctx) -> list[Result]
    draw(frame, results)            -> draw results onto a BGR frame in place

`ctx` is the FrameState being built for this frame, so a module can use the
results of modules that ran before it (face_recog reads face_detect's faces).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

from netrax import models
from netrax.core.types import FrameState, Result

if TYPE_CHECKING:
    from netrax.config import Settings


class Detector:
    name: ClassVar[str] = ""
    model_keys: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[tuple[str, ...]] = ()   # modules whose results this one reads

    def __init__(self, settings: "Settings", services: dict | None = None):
        self.settings = settings
        self.services = services or {}   # shared objects, e.g. {"store": Store}

    @property
    def cfg(self):
        return getattr(self.settings.cfg.modules, self.name)

    def model_file(self, key: str) -> str:
        return str(models.ensure(self.settings.models_dir, key))

    def load(self) -> None:
        raise NotImplementedError

    def process(self, frame: np.ndarray, ctx: FrameState) -> list[Result]:
        raise NotImplementedError

    def draw(self, frame: np.ndarray, results: list[Result]) -> None:
        pass

    def reset(self) -> None:
        """Forget per-track state (called when the video source changes)."""


def load_net(path: str) -> cv2.dnn.Net:
    return cv2.dnn.readNetFromONNX(path)   # CPU, default engine


def forward_named(net: cv2.dnn.Net, blob: np.ndarray) -> dict[str, np.ndarray]:
    """Run the net and return outputs by name (OpenCV 5 may order them differently than 4.x)."""
    names = net.getUnconnectedOutLayersNames()
    net.setInput(blob)
    outs = net.forward(names)
    return dict(zip(names, outs))
