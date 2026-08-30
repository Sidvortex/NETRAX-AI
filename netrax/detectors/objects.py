"""Object detection (COCO 80 classes): NanoDet-Plus (default, fast) or YOLOX-S (slower, better).

Both are Apache-2.0 ONNX models from OpenCV Zoo. Ultralytics YOLO is AGPL-3.0,
so it is deliberately not used. Post-processing adapted from OpenCV Zoo
(models/object_detection_nanodet/nanodet.py, models/object_detection_yolox/yolox.py).
"""

from __future__ import annotations

import cv2
import numpy as np

from netrax.core import draw as D
from netrax.core.types import Detection, FrameState
from netrax.detectors.base import Detector, forward_named, load_net

COCO = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush")


def _palette(i: int) -> tuple[int, int, int]:
    h = (i * 47) % 180
    c = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


class NanoDet:
    size = 416
    strides = (8, 16, 32, 64)
    reg_max = 7
    mean = np.array([103.53, 116.28, 123.675], np.float32)
    std = np.array([57.375, 57.12, 58.395], np.float32)

    def __init__(self, path: str):
        self.net = load_net(path)
        self.anchors = []
        for s in self.strides:
            n = self.size // s
            ys, xs = np.mgrid[0:n, 0:n]
            self.anchors.append(np.c_[xs.ravel() * s + 0.5 * (s - 1), ys.ravel() * s + 0.5 * (s - 1)])
        self.project = np.arange(self.reg_max + 1, dtype=np.float32)

    def letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(h * r), int(w * r)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        out = cv2.copyMakeBorder(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA),
                                 top, self.size - nh - top, left, self.size - nw - left, cv2.BORDER_CONSTANT, value=0)
        return out, r, left, top

    def __call__(self, frame: np.ndarray, score_thr: float, nms_thr: float) -> np.ndarray:
        img, r, left, top = self.letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        blob = cv2.dnn.blobFromImage((img.astype(np.float32) - self.mean) / self.std)
        outs = list(forward_named(self.net, blob).values())
        cls_out = sorted([o for o in outs if o.shape[-1] == 80], key=lambda o: -o.shape[-2])
        box_out = sorted([o for o in outs if o.shape[-1] == 4 * (self.reg_max + 1)], key=lambda o: -o.shape[-2])
        boxes, scores = [], []
        for stride, cls, reg, anchors in zip(self.strides, cls_out, box_out, self.anchors):
            cls, reg = cls.reshape(-1, 80), reg.reshape(-1, 4, self.reg_max + 1)
            e = np.exp(reg - reg.max(-1, keepdims=True))
            dist = (e / e.sum(-1, keepdims=True)) @ self.project * stride
            x1y1 = anchors - dist[:, :2]
            x2y2 = anchors + dist[:, 2:]
            boxes.append(np.clip(np.c_[x1y1, x2y2], 0, self.size))
            scores.append(cls)
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        cls_id, conf = scores.argmax(1), scores.max(1)
        keep = conf >= score_thr
        boxes, cls_id, conf = boxes[keep], cls_id[keep], conf[keep]
        if len(boxes) == 0:
            return np.zeros((0, 6))
        xywh = np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]
        idx = np.asarray(cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls_id.tolist(), score_thr, nms_thr)).ravel()
        b = boxes[idx]
        b[:, [0, 2]] = (b[:, [0, 2]] - left) / r
        b[:, [1, 3]] = (b[:, [1, 3]] - top) / r
        return np.c_[b, conf[idx], cls_id[idx]]


class YoloX:
    size = 640
    strides = (8, 16, 32)

    def __init__(self, path: str):
        self.net = load_net(path)
        grids, strides = [], []
        for s in self.strides:
            n = self.size // s
            ys, xs = np.mgrid[0:n, 0:n]
            grids.append(np.c_[xs.ravel(), ys.ravel()])
            strides.append(np.full((n * n, 1), s))
        self.grids, self.stride_col = np.concatenate(grids), np.concatenate(strides)

    def __call__(self, frame: np.ndarray, score_thr: float, nms_thr: float) -> np.ndarray:
        h, w = frame.shape[:2]
        r = min(self.size / h, self.size / w)
        img = np.full((self.size, self.size, 3), 114, np.float32)
        img[: int(h * r), : int(w * r)] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (int(w * r), int(h * r)))
        blob = np.transpose(img, (2, 0, 1))[None]
        dets = next(iter(forward_named(self.net, blob).values()))[0].copy()
        dets[:, :2] = (dets[:, :2] + self.grids) * self.stride_col
        dets[:, 2:4] = np.exp(dets[:, 2:4]) * self.stride_col
        scores = dets[:, 4:5] * dets[:, 5:]
        cls_id, conf = scores.argmax(1), scores.max(1)
        keep = conf >= score_thr
        d, cls_id, conf = dets[keep], cls_id[keep], conf[keep]
        if len(d) == 0:
            return np.zeros((0, 6))
        xywh = np.c_[d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2, d[:, 2], d[:, 3]]
        idx = np.asarray(cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls_id.tolist(), score_thr, nms_thr)).ravel()
        b = xywh[idx] / r
        return np.c_[b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3], conf[idx], cls_id[idx]]


class ObjectDetector(Detector):
    name = "objects"
    model_keys = ("nanodet", "yolox")

    def load(self) -> None:
        kind = self.cfg.model
        self.kind = kind
        self.model = NanoDet(self.model_file("nanodet")) if kind == "nanodet" else YoloX(self.model_file("yolox"))

    def process(self, frame: np.ndarray, ctx: FrameState) -> list[Detection]:
        c = self.cfg
        rows = self.model(frame, c.score_threshold, c.nms_threshold)
        h, w = frame.shape[:2]
        wanted = set(c.classes)
        out = []
        for x1, y1, x2, y2, score, cid in rows:
            label = COCO[int(cid)]
            if wanted and label not in wanted:
                continue
            box = np.array([max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
            out.append(Detection(bbox=box, label=label, class_id=int(cid), score=float(score)))
        return out

    def draw(self, frame: np.ndarray, results: list[Detection]) -> None:
        for d in results:
            x1, y1, x2, y2 = d.bbox
            color = _palette(d.class_id)
            cv2.rectangle(frame, D.pt((x1, y1)), D.pt((x2, y2)), color, 1, cv2.LINE_AA)
            D.label(frame, f"{d.label} {d.score * 100:.0f}%", (x1, y1), color, 0.45)
