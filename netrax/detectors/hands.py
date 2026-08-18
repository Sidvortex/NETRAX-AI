"""Hands: MediaPipe palm detector + 21-point hand landmarks (ONNX, OpenCV Zoo) -> gestures."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from netrax.core import draw as D
from netrax.core.tracker import IouTracker
from netrax.core.types import FrameState, Hand
from netrax.detectors import mediapipe_onnx as mp
from netrax.detectors.base import Detector, forward_named, load_net
from netrax.detectors.gestures import classify

PALM_SIZE = 192
HAND_SIZE = 224
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
GESTURE_ICON = {"open_palm": "OPEN PALM", "fist": "FIST", "thumbs_up": "THUMBS UP", "point": "POINT", "peace": "PEACE"}


class HandTracker(Detector):
    name = "hands"
    model_keys = ("palm", "handpose")

    def load(self) -> None:
        self.palm_net = load_net(self.model_file("palm"))
        self.hand_net = load_net(self.model_file("handpose"))
        self.tracker = IouTracker(iou_threshold=0.2, max_missed=4)
        self.history: dict[int, deque] = {}
        self.stable: dict[int, str] = {}

    def reset(self) -> None:
        self.tracker.reset()
        self.history.clear()
        self.stable.clear()

    # palm detection ----------------------------------------------------------
    def detect_palms(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        img, pad_bias = mp.letterbox(frame, PALM_SIZE)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0     # NHWC
        boxes, scores = mp.pick_outputs(forward_named(self.palm_net, blob), len(mp.PALM_ANCHORS), 18)
        palms = mp.decode_ssd(boxes, scores, mp.PALM_ANCHORS, PALM_SIZE, (w, h), pad_bias, 7,
                              self.cfg.palm_score, 0.3)
        return palms[np.argsort(-palms[:, -1])][: self.cfg.max_hands]

    # landmarks -----------------------------------------------------------------
    def landmarks(self, frame: np.ndarray, palm: np.ndarray):
        """(21x3 landmarks in image px, handedness score, confidence) or None."""
        palm_box = palm[0:4].reshape(2, 2)
        img, ebox, bias = mp.crop_and_pad(frame, palm_box, (0, 0), 4.0, square_diag=True)
        if img.size == 0:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kps = palm[4:18].reshape(7, 2) - bias
        angle = mp.rotation_to_vertical(kps[0], kps[2])
        centre = (ebox - bias).sum(0) / 2
        R = cv2.getRotationMatrix2D(tuple(map(float, centre)), angle, 1.0)
        rotated = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]))
        rk = np.c_[kps, np.ones(7)] @ R.T
        rbox = np.array([rk.min(0), rk.max(0)])
        crop, rbox, _ = mp.crop_and_pad(rotated, rbox, (0, -0.4), 3.0, square_diag=False)
        if crop.size == 0:
            return None
        blob = cv2.resize(crop, (HAND_SIZE, HAND_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)[None] / 255.0
        outs = forward_named(self.hand_net, blob)
        vals = list(outs.values())
        lm63 = [v for v in vals if v.size == 63]
        scalars = [float(v.ravel()[0]) for v in vals if v.size == 1]
        if not lm63 or len(scalars) < 2:
            raise RuntimeError(f"unexpected hand model outputs: {[v.shape for v in vals]}")
        conf, handed = scalars[0], scalars[1]
        if conf < self.cfg.landmark_conf:
            return None
        lm = lm63[0].reshape(21, 3).astype(np.float64)
        wh = rbox[1] - rbox[0]
        s = max(wh / HAND_SIZE)
        xy = mp.unrotate_points(lm, angle, rbox, R, bias, HAND_SIZE, uniform_scale=True)
        return np.c_[xy, lm[:, 2] * s], handed, conf

    # per frame -------------------------------------------------------------------
    def process(self, frame: np.ndarray, ctx: FrameState) -> list[Hand]:
        hands = []
        for palm in self.detect_palms(frame):
            r = self.landmarks(frame, palm)
            if r is None:
                continue
            lm, handed, conf = r
            x1, y1 = lm[:, :2].min(0)
            x2, y2 = lm[:, :2].max(0)
            # handedness as the MediaPipe model reports it (> 0.5 = right)
            hands.append(Hand(bbox=np.array([x1, y1, x2, y2]), landmarks=lm, handedness="right" if handed > 0.5 else "left",
                              score=conf, raw_gesture=classify(lm)))
        ids = self.tracker.update([np.array([h.bbox[0], h.bbox[1], h.bbox[2] - h.bbox[0], h.bbox[3] - h.bbox[1]])
                                   for h in hands])
        need = self.cfg.stable_frames
        for h, tid in zip(hands, ids):
            hist = self.history.setdefault(tid, deque(maxlen=need))
            if hist.maxlen != need:
                hist = self.history[tid] = deque(hist, maxlen=need)
            hist.append(h.raw_gesture)
            if len(hist) == need and len(set(hist)) == 1:
                self.stable[tid] = hist[0]
            h.gesture = self.stable.get(tid, "none")
        for tid in self.tracker.ended:
            self.history.pop(tid, None)
            self.stable.pop(tid, None)
        return hands

    def draw(self, frame: np.ndarray, results: list[Hand]) -> None:
        for h in results:
            p = h.landmarks[:, :2]
            for a, b in CONNECTIONS:
                cv2.line(frame, D.pt(p[a]), D.pt(p[b]), D.MAGENTA, 2, cv2.LINE_AA)
            for i, q in enumerate(p):
                cv2.circle(frame, D.pt(q), 4 if i in (4, 8, 12, 16, 20) else 2, D.WHITE, -1, cv2.LINE_AA)
            x1, y1, x2, y2 = h.bbox
            if h.gesture != "none":
                D.corner_box(frame, x1 - 8, y1 - 8, x2 + 8, y2 + 8, D.MAGENTA, 2)
                D.label(frame, f"{GESTURE_ICON.get(h.gesture, h.gesture)} ({h.handedness})", (x1 - 8, y1 - 8), D.MAGENTA, 0.55)
            else:
                D.label(frame, h.handedness, (x1, y1), D.MAGENTA, 0.4)
