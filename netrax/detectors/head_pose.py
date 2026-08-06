"""Head pose (yaw / pitch / roll) from YuNet's 5 face landmarks with cv2.solvePnP.

No extra model: the 5 landmarks are fitted to a generic 3D face, which is
plenty for "is the person looking at me, left, right, up or down".
Angles are from the camera's point of view:
  yaw   > 0  face turned towards the image's right side
  pitch > 0  face tilted up
  roll  > 0  head tilted clockwise in the image
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from netrax.core import draw as D
from netrax.core.types import FrameState, HeadPose
from netrax.detectors.base import Detector

# generic face, mm, camera-style axes (x right, y down, z away from the camera), nose tip at origin.
# order = YuNet landmarks: right eye, left eye (image left/right), nose tip, right / left mouth corner
MODEL_3D = np.array([
    [-32.0, -35.0, 30.0],
    [32.0, -35.0, 30.0],
    [0.0, 0.0, 0.0],
    [-25.0, 30.0, 22.0],
    [25.0, 30.0, 22.0],
], np.float64)
AXIS_LEN = 45.0


def camera_matrix(w: int, h: int) -> np.ndarray:
    f = float(w)                      # ~53 deg horizontal FOV, typical for laptop webcams
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def angles_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    fwd = R @ np.array([0.0, 0.0, -1.0])        # where the face points
    right = R @ np.array([1.0, 0.0, 0.0])
    yaw = math.degrees(math.atan2(fwd[0], -fwd[2]))
    pitch = math.degrees(math.atan2(-fwd[1], math.hypot(fwd[0], fwd[2])))
    roll = math.degrees(math.atan2(right[1], right[0]))
    return yaw, pitch, roll


def solve(landmarks: np.ndarray, w: int, h: int):
    """(yaw, pitch, roll, rvec, tvec) or None."""
    K = camera_matrix(w, h)
    pts = np.ascontiguousarray(landmarks, np.float64).reshape(5, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, pts, K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok or tvec[2, 0] <= 0:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return (*angles_from_rotation(R), rvec, tvec)


def direction(yaw: float, pitch: float, thr: float) -> str:
    if abs(yaw) < thr and abs(pitch) < thr:
        return "center"
    if abs(yaw) >= abs(pitch):
        return "right" if yaw > 0 else "left"
    return "up" if pitch > 0 else "down"


class HeadPoseEstimator(Detector):
    name = "head_pose"
    requires = ("face_detect",)

    def load(self) -> None:
        self.smooth: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self.smooth.clear()

    def process(self, frame: np.ndarray, ctx: FrameState) -> list[HeadPose]:
        h, w = frame.shape[:2]
        c = self.cfg
        out, live = [], set()
        for f in ctx.get("face_detect"):
            r = solve(f.landmarks, w, h)
            if r is None:
                continue
            yaw, pitch, roll, rvec, tvec = r
            a = np.array([yaw, pitch, roll])
            prev = self.smooth.get(f.track_id)
            if prev is not None:
                a = c.smoothing * prev + (1 - c.smoothing) * a
            self.smooth[f.track_id] = a
            live.add(f.track_id)
            pts = np.float64([[0, 0, 0], [AXIS_LEN, 0, 0], [0, -AXIS_LEN, 0], [0, 0, -AXIS_LEN]])
            axes, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix(w, h), None)
            out.append(HeadPose(track_id=f.track_id, yaw=float(a[0]), pitch=float(a[1]), roll=float(a[2]),
                                direction=direction(a[0], a[1], c.look_threshold), axes=axes.reshape(4, 2)))
        for tid in list(self.smooth):
            if tid not in live and len(self.smooth) > 32:
                del self.smooth[tid]
        return out

    def draw(self, frame: np.ndarray, results: list[HeadPose]) -> None:
        for r in results:
            if r.axes is None:
                continue
            o, x, y, z = r.axes
            cv2.line(frame, D.pt(o), D.pt(x), D.RED, 2, cv2.LINE_AA)
            cv2.line(frame, D.pt(o), D.pt(y), D.GREEN, 2, cv2.LINE_AA)
            D.glow_line(frame, o, z, D.CYAN, 2)
            D.label(frame, f"{r.direction}  y{r.yaw:+.0f} p{r.pitch:+.0f} r{r.roll:+.0f}", z, D.CYAN, 0.4)
