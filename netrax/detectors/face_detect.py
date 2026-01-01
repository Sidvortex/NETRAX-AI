"""Face detection with OpenCV YuNet (cv2.FaceDetectorYN).

Uses `face_detection_yunet_2026may.onnx`, the dynamic-input export that works
with OpenCV 5's DNN engine (the 2023mar file has a fixed input size).
"""

from __future__ import annotations

import cv2
import numpy as np

from netrax.core import draw as D
from netrax.core.tracker import IouTracker
from netrax.core.types import Face, FrameState
from netrax.detectors.base import Detector


class FaceDetector(Detector):
    name = "face_detect"
    model_keys = ("yunet",)

    def load(self) -> None:
        c = self.cfg
        self.det = cv2.FaceDetectorYN.create(self.model_file("yunet"), "", (320, 320),
                                             c.score_threshold, c.nms_threshold, 5000)
        self.tracker = IouTracker()
        self._size = (0, 0)
        self._thr = (c.score_threshold, c.nms_threshold)

    def reset(self) -> None:
        self.tracker.reset()

    def detect_raw(self, frame: np.ndarray) -> np.ndarray:
        """YuNet rows (N x 15) in original frame coordinates."""
        c = self.cfg
        if (c.score_threshold, c.nms_threshold) != self._thr:
            self.det.setScoreThreshold(c.score_threshold)
            self.det.setNMSThreshold(c.nms_threshold)
            self._thr = (c.score_threshold, c.nms_threshold)
        h, w = frame.shape[:2]
        s = min(1.0, c.max_side / max(h, w))
        img = frame if s == 1.0 else cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        size = (img.shape[1], img.shape[0])
        if size != self._size:
            self.det.setInputSize(size)
            self._size = size
        _, faces = self.det.detect(img)
        if faces is None:
            return np.zeros((0, 15), np.float32)
        faces = faces.copy()
        faces[:, :14] /= s
        keep = (faces[:, 2] >= c.min_face) & (faces[:, 3] >= c.min_face)
        return faces[keep]

    def process(self, frame: np.ndarray, ctx: FrameState) -> list[Face]:
        rows = self.detect_raw(frame)
        ids = self.tracker.update([r[:4] for r in rows])
        return [
            Face(bbox=r[:4].copy(), score=float(r[14]), landmarks=r[4:14].reshape(5, 2).copy(), raw=r.copy(), track_id=t)
            for r, t in zip(rows, ids)
        ]

    def draw(self, frame: np.ndarray, results: list[Face]) -> None:
        for f in results:
            x, y, w, h = f.bbox
            D.corner_box(frame, x, y, x + w, y + h, D.YELLOW, 1)
            for p in f.landmarks:
                cv2.circle(frame, D.pt(p), 2, D.CYAN, -1, cv2.LINE_AA)
            D.label(frame, f"#{f.track_id} {f.score:.2f}", (x, y + h), D.YELLOW, 0.4, above=False)
