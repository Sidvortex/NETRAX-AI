"""Facial expression (optional): OpenCV Zoo MobileFaceNet FER model, 7 classes.

Faces are aligned to the standard 112x112 five-point template (same as SFace)
from YuNet's landmarks. Scores are smoothed per face track.
"""

from __future__ import annotations

import cv2
import numpy as np

from netrax.core import draw as D
from netrax.core.types import Emotion, FrameState
from netrax.detectors.base import Detector, forward_named, load_net

LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
TEMPLATE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def align_face(frame: np.ndarray, landmarks5: np.ndarray, size: int = 112) -> np.ndarray | None:
    M, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks5, np.float32), TEMPLATE_112, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(frame, M, (size, size))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class EmotionRecognizer(Detector):
    name = "emotion"
    model_keys = ("fer",)
    requires = ("face_detect",)

    def load(self) -> None:
        self.net = load_net(self.model_file("fer"))
        self.probs: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self.probs.clear()

    def process(self, frame: np.ndarray, ctx: FrameState) -> list[Emotion]:
        out, live = [], set()
        for f in ctx.get("face_detect"):
            face = align_face(frame, f.landmarks)
            if face is None:
                continue
            x = (cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 0.5) / 0.5
            logits = next(iter(forward_named(self.net, cv2.dnn.blobFromImage(x)).values())).ravel()
            p = softmax(logits) if (logits.min() < 0 or logits.sum() > 1.01) else logits
            prev = self.probs.get(f.track_id)
            p = p if prev is None else 0.5 * prev + 0.5 * p
            self.probs[f.track_id] = p
            live.add(f.track_id)
            i = int(np.argmax(p))
            out.append(Emotion(track_id=f.track_id, label=LABELS[i], score=float(p[i]), bbox=f.bbox))
        for tid in [t for t in self.probs if t not in live]:
            if len(self.probs) > 32:
                del self.probs[tid]
        return out

    def draw(self, frame: np.ndarray, results: list[Emotion]) -> None:
        for r in results:
            if r.bbox is not None:
                x, y, w, h = r.bbox
                D.label(frame, f"{r.label} {r.score * 100:.0f}%", (x + w, y + h), D.MAGENTA, 0.45, above=False)
