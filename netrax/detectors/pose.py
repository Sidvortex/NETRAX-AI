"""Pose: MediaPipe person detector + BlazePose landmarks (ONNX, OpenCV Zoo), output as COCO-17.

Single person (the most confident one), which is what a desk camera needs:
posture checks and the "dance skeleton" demo. BlazePose's 33 points are
mapped to the 17 COCO keypoints so the output looks like any other pose model.
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from netrax.core import draw as D
from netrax.core.types import FrameState, Pose
from netrax.detectors import mediapipe_onnx as mp
from netrax.detectors.base import Detector, forward_named, load_net

PERSON_SIZE = 224
POSE_SIZE = 256
COCO_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle")
BLAZE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
# dance style colours per limb group (BGR)
LIMB_COLORS = {"left": D.MAGENTA, "right": D.CYAN, "torso": D.YELLOW, "head": D.WHITE}


def limb_group(a: int, b: int) -> str:
    if a <= 4 and b <= 4:
        return "head"
    if {a, b} <= {5, 6, 11, 12}:
        return "torso"
    return "left" if a % 2 == 1 else "right"


def posture_metrics(kp: np.ndarray, min_vis: float) -> dict | None:
    """head_ratio = how high the nose sits above the shoulder line, in shoulder widths; shoulder tilt in degrees."""
    nose, ls, rs = kp[0], kp[5], kp[6]
    if min(nose[2], ls[2], rs[2]) < min_vis:
        return None
    width = float(np.hypot(*(ls[:2] - rs[:2])))
    if width < 10:
        return None
    mid = (ls[:2] + rs[:2]) / 2
    a, b = (ls, rs) if ls[0] < rs[0] else (rs, ls)
    return {"head_ratio": round(float((mid[1] - nose[1]) / width), 3),
            "shoulder_tilt": round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])), 1)}


class PostureJudge:
    """Self-calibrating posture check.

    Everyone's camera angle and body differ, so instead of a fixed number we
    learn *your* upright head height (a high percentile of recent samples) and
    call it slouching when the head drops `slouch_drop` (e.g. 20%) below it.
    """

    def __init__(self, window: int = 900, min_samples: int = 30):
        self.samples: deque = deque(maxlen=window)
        self.min_samples = min_samples

    def reset(self) -> None:
        self.samples.clear()

    def __call__(self, m: dict | None, slouch_drop: float, tilt_deg: float) -> tuple[str, dict]:
        if m is None:
            return "unknown", {}
        self.samples.append(m["head_ratio"])
        if len(self.samples) < self.min_samples:
            return "calibrating", m
        baseline = float(np.percentile(self.samples, 90))
        m = dict(m, baseline=round(baseline, 3))
        if baseline > 0 and m["head_ratio"] < baseline * (1 - slouch_drop):
            return "slouching", m
        if abs(m["shoulder_tilt"]) > tilt_deg:
            return "leaning", m
        return "upright", m


class PoseEstimator(Detector):
    name = "pose"
    model_keys = ("persondet", "pose")

    def load(self) -> None:
        self.det_net = load_net(self.model_file("persondet"))
        self.pose_net = load_net(self.model_file("pose"))
        self.trail: deque = deque(maxlen=8)
        self.judge = PostureJudge()

    def reset(self) -> None:
        self.trail.clear()
        self.judge.reset()

    def detect_person(self, frame: np.ndarray) -> np.ndarray | None:
        h, w = frame.shape[:2]
        img, pad_bias = mp.letterbox(frame, PERSON_SIZE)
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 0.5) * 2.0
        blob = np.transpose(x, (2, 0, 1))[None]                                       # NCHW
        boxes, scores = mp.pick_outputs(forward_named(self.det_net, blob), len(mp.PERSON_ANCHORS), 12)
        people = mp.decode_ssd(boxes, scores, mp.PERSON_ANCHORS, PERSON_SIZE, (w, h), pad_bias, 4,
                               self.cfg.person_score, 0.3)
        if len(people) == 0:
            return None
        return people[int(np.argmax(people[:, -1]))]

    def landmarks(self, frame: np.ndarray, person: np.ndarray):
        """(33x5 landmarks [x, y, z, visibility, presence] in image px, confidence) or None."""
        kps = person[4:12].reshape(4, 2).astype(np.float64)
        mid_hip, full_body = kps[0].copy(), kps[1].copy()
        r = np.linalg.norm(mid_hip - full_body)
        if r < 8:
            return None
        full = np.array([mid_hip - r, mid_hip + r], np.int32)
        H, W = frame.shape[:2]
        clip = full.copy()
        clip[:, 0] = np.clip(clip[:, 0], 0, W)
        clip[:, 1] = np.clip(clip[:, 1], 0, H)
        crop = frame[clip[0][1]:clip[1][1], clip[0][0]:clip[1][0]]
        if crop.size == 0:
            return None
        left, top = clip[0] - full[0]
        right, bottom = full[1] - clip[1]
        crop = cv2.copyMakeBorder(crop, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=(0, 0, 0))
        pad_bias = clip[0] - np.array([left, top])
        mid_hip -= pad_bias
        full_body -= pad_bias
        angle = mp.rotation_to_vertical(mid_hip, full_body)
        R = cv2.getRotationMatrix2D(tuple(map(float, mid_hip)), angle, 1.0)
        rotated = cv2.warpAffine(crop, R, (crop.shape[1], crop.shape[0]))
        blob = cv2.resize(rotated, (POSE_SIZE, POSE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)[None] / 255.0                    # NHWC
        outs = forward_named(self.pose_net, blob)
        lm = next((v for v in outs.values() if v.size == 195), None)
        conf = next((v for v in outs.values() if v.size == 1), None)
        if lm is None or conf is None:
            raise RuntimeError(f"unexpected pose outputs: {[v.shape for v in outs.values()]}")
        conf = float(conf.ravel()[0])
        if conf < self.cfg.person_score:
            return None
        lm = lm.reshape(39, 5)[:33].astype(np.float64)
        lm[:, 3:] = 1.0 / (1.0 + np.exp(-lm[:, 3:]))
        rbox = np.array([[0, 0], [crop.shape[1], crop.shape[0]]], np.float64)
        xy = mp.unrotate_points(lm, angle, rbox, R, pad_bias, POSE_SIZE, uniform_scale=False)
        lm[:, :2] = xy
        return lm, conf

    def process(self, frame: np.ndarray, ctx: FrameState) -> list[Pose]:
        person = self.detect_person(frame)
        if person is None:
            self.trail.clear()
            return []
        r = self.landmarks(frame, person)
        if r is None:
            return []
        lm, conf = r
        kp = np.c_[lm[BLAZE_TO_COCO, :2], lm[BLAZE_TO_COCO, 3]]
        vis = kp[:, 2] >= self.cfg.min_visibility
        pts = kp[vis, :2] if vis.any() else kp[:, :2]
        x1, y1 = pts.min(0)
        x2, y2 = pts.max(0)
        c = self.cfg
        label, metrics = self.judge(posture_metrics(kp, c.min_visibility), c.slouch_drop, c.tilt_deg)
        self.trail.append(kp.copy())
        return [Pose(bbox=np.array([x1, y1, x2, y2]), keypoints=kp, score=conf, posture=label, metrics=metrics)]

    def draw(self, frame: np.ndarray, results: list[Pose]) -> None:
        dance = self.cfg.style == "dance"
        mv = self.cfg.min_visibility
        if dance and len(self.trail) > 1:
            overlay = frame.copy()
            for i, kp in enumerate(list(self.trail)[:-1]):
                for a, b in SKELETON:
                    if kp[a, 2] >= mv and kp[b, 2] >= mv:
                        cv2.line(overlay, D.pt(kp[a]), D.pt(kp[b]), LIMB_COLORS[limb_group(a, b)], 2 + i // 3, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        for p in results:
            kp = p.keypoints
            for a, b in SKELETON:
                if kp[a, 2] < mv or kp[b, 2] < mv:
                    continue
                if dance:
                    D.glow_line(frame, kp[a], kp[b], LIMB_COLORS[limb_group(a, b)], 4)
                else:
                    cv2.line(frame, D.pt(kp[a]), D.pt(kp[b]), D.GREEN, 2, cv2.LINE_AA)
            for i, q in enumerate(kp):
                if q[2] >= mv:
                    cv2.circle(frame, D.pt(q), 6 if dance else 3, D.WHITE, -1, cv2.LINE_AA)
            if p.posture != "unknown" and kp[0, 2] >= mv:
                color = {"upright": D.GREEN, "slouching": D.ORANGE, "leaning": D.YELLOW}.get(p.posture, D.WHITE)
                D.label(frame, f"posture: {p.posture}", (kp[0, 0] + 30, kp[0, 1]), color, 0.5)
