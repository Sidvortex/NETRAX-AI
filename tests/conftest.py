import os
from pathlib import Path

import numpy as np
import pytest

from netrax.config import Settings
from netrax.core.types import Detection, Face, FrameState
from netrax.detectors.base import Detector

MODELS_DIR = Path(os.environ.get("NETRAX_MODELS", Path(__file__).resolve().parents[1] / "models"))


class StubFaces(Detector):
    """Pretends to find one face that moves a little each frame."""

    name = "face_detect"

    def load(self):
        self.calls = 0

    def process(self, frame, ctx: FrameState):
        self.calls += 1
        x = 100 + (ctx.frame_idx % 5)
        lm = np.array([[x + 15, 120], [x + 45, 120], [x + 30, 135], [x + 18, 150], [x + 42, 150]], float)
        raw = np.r_[[x, 100, 60, 70], lm.ravel(), 0.95].astype(np.float32)
        return [Face(bbox=np.array([x, 100, 60, 70], float), score=0.95, landmarks=lm, raw=raw, track_id=1)]

    def draw(self, frame, results):
        for f in results:
            x, y, w, h = f.bbox.astype(int)
            frame[y:y + h, x:x + w] = 255


class StubObjects(Detector):
    name = "objects"

    def load(self):
        self.calls = 0

    def process(self, frame, ctx):
        self.calls += 1
        return [Detection(bbox=np.array([10, 10, 50, 50], float), label="cup", class_id=41, score=0.8)]


class Broken(Detector):
    name = "pose"

    def load(self):
        pass

    def process(self, frame, ctx):
        raise RuntimeError("boom")


class FailsToLoad(Detector):
    name = "hands"

    def load(self):
        raise RuntimeError("model missing")


STUBS = {"face_detect": StubFaces, "objects": StubObjects, "pose": Broken, "hands": FailsToLoad}


def stub_factory(name, settings):
    return STUBS[name](settings)


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "config.yaml", overrides={"paths": {"models_dir": str(MODELS_DIR)}})


@pytest.fixture
def frame():
    return np.zeros((240, 320, 3), np.uint8)


def need_models(*keys):
    from netrax import models

    missing = [k for k in keys if not models.is_valid(models.model_path(MODELS_DIR, k), models.MODELS[k])]
    return pytest.mark.skipif(bool(missing), reason=f"models not downloaded: {missing} (run `netrax models --download`)")
