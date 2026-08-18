"""Pre/post-processing for the MediaPipe ONNX models in OpenCV Zoo.

Adapted from OpenCV Zoo (Apache-2.0):
  models/palm_detection_mediapipe/mp_palmdet.py
  models/handpose_estimation_mediapipe/mp_handpose.py
  models/person_detection_mediapipe/mp_persondet.py
  models/pose_estimation_mediapipe/mp_pose.py
Changes: anchors are generated instead of hard-coded, outputs are picked by
shape/name (OpenCV 5 may order them differently than 4.x), small cleanups.
"""

from __future__ import annotations

import cv2
import numpy as np


def ssd_anchors(layers: list[tuple[int, int]]) -> np.ndarray:
    """MediaPipe SSD anchor centres: [(feature_map_size, anchors_per_cell), ...] -> (N, 2) in 0..1."""
    out = []
    for fm, n in layers:
        ys, xs = np.mgrid[0:fm, 0:fm]
        centres = np.stack([(xs + 0.5) / fm, (ys + 0.5) / fm], -1).reshape(-1, 1, 2)
        out.append(np.repeat(centres, n, axis=1).reshape(-1, 2))
    return np.concatenate(out).astype(np.float32)


PALM_ANCHORS = ssd_anchors([(24, 2), (12, 6)])               # 192x192 input, 2016 anchors
PERSON_ANCHORS = ssd_anchors([(28, 2), (14, 2), (7, 6)])     # 224x224 input, 2254 anchors


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize keeping aspect ratio and pad to size x size (black). Returns (img, pad_bias in original px)."""
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = int(h * ratio), int(w * ratio)
    img = cv2.resize(image, (nw, nh))
    top, left = (size - nh) // 2, (size - nw) // 2
    img = cv2.copyMakeBorder(img, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    pad_bias = (np.array([left, top], np.float64) / ratio).astype(np.int32)
    return img, pad_bias


def pick_outputs(outs: dict[str, np.ndarray], n_anchors: int, box_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """(box/keypoint deltas [N, box_dim], raw scores [N]) regardless of output order."""
    boxes = scores = None
    for v in outs.values():
        a = v.reshape(n_anchors, -1)
        if a.shape[1] == box_dim:
            boxes = a
        elif a.shape[1] == 1:
            scores = a[:, 0]
    if boxes is None or scores is None:
        raise RuntimeError(f"unexpected detector outputs: {[v.shape for v in outs.values()]}")
    return boxes, scores


def decode_ssd(boxes: np.ndarray, scores: np.ndarray, anchors: np.ndarray, input_size: int, orig_wh,
               pad_bias: np.ndarray, n_kp: int, score_thr: float, nms_thr: float, top_k: int = 5000) -> np.ndarray:
    """Rows of [x1, y1, x2, y2, kp1x, kp1y, ..., score] in original image pixels."""
    scale = float(max(orig_wh))
    s = 1.0 / (1.0 + np.exp(-np.clip(scores.astype(np.float64), -100, 100)))
    cxy = boxes[:, :2] / input_size
    wh = boxes[:, 2:4] / input_size
    xy1 = (cxy - wh / 2 + anchors) * scale
    xy2 = (cxy + wh / 2 + anchors) * scale
    b = np.concatenate([xy1, xy2], 1) - np.tile(pad_bias, 2)
    keep = cv2.dnn.NMSBoxes(b.tolist(), s.tolist(), score_thr, nms_thr, top_k=top_k)
    if len(keep) == 0:
        return np.zeros((0, 4 + 2 * n_kp + 1))
    keep = np.asarray(keep).ravel()
    kp = boxes[keep, 4:4 + 2 * n_kp].reshape(-1, n_kp, 2) / input_size + anchors[keep][:, None, :]
    kp = kp * scale - pad_bias
    return np.c_[b[keep], kp.reshape(len(keep), -1), s[keep]]


def invert_affine(rotation_matrix: np.ndarray) -> np.ndarray:
    rc = rotation_matrix[:, :2].T
    t = rotation_matrix[:, 2]
    return np.c_[rc, -rc @ t]


def crop_and_pad(image: np.ndarray, bbox: np.ndarray, shift, enlarge: float, square_diag: bool):
    """Shift + enlarge a [[x1,y1],[x2,y2]] box, crop it and pad to a square.

    Returns (crop, clipped box, bias) where bias maps crop coords back to the image.
    """
    wh = bbox[1] - bbox[0]
    bbox = bbox + np.asarray(shift) * wh
    centre = bbox.sum(0) / 2
    half = (bbox[1] - bbox[0]) * enlarge / 2
    bbox = np.array([centre - half, centre + half]).astype(np.int32)
    bbox[:, 0] = np.clip(bbox[:, 0], 0, image.shape[1])
    bbox[:, 1] = np.clip(bbox[:, 1], 0, image.shape[0])
    crop = image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    side = int(np.linalg.norm(crop.shape[:2])) if square_diag else max(crop.shape[:2])
    pad_h, pad_w = side - crop.shape[0], side - crop.shape[1]
    left, top = pad_w // 2, pad_h // 2
    crop = cv2.copyMakeBorder(crop, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return crop, bbox, bbox[0] - np.array([left, top])


def rotation_to_vertical(p_base: np.ndarray, p_tip: np.ndarray) -> float:
    """Angle (degrees) that rotates the base->tip direction to point straight up."""
    r = np.pi / 2 - np.arctan2(-(p_tip[1] - p_base[1]), p_tip[0] - p_base[0])
    r = r - 2 * np.pi * np.floor((r + np.pi) / (2 * np.pi))
    return float(np.rad2deg(r))


def unrotate_points(pts: np.ndarray, angle: float, rotated_box: np.ndarray, rotation_matrix: np.ndarray,
                    pad_bias: np.ndarray, input_size: int, uniform_scale: bool) -> np.ndarray:
    """Map landmark xy from model input space back to original image pixels."""
    wh = rotated_box[1] - rotated_box[0]
    sf = wh / input_size
    s = max(sf) if uniform_scale else sf
    xy = (pts[:, :2] - input_size / 2) * s
    rot = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    xy = xy @ rot[:, :2]
    centre = np.append(rotated_box.sum(0) / 2, 1)
    orig_centre = invert_affine(rotation_matrix) @ centre
    return xy + orig_centre + pad_bias
