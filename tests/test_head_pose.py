import math

import cv2
import numpy as np
import pytest

from netrax.detectors.head_pose import MODEL_3D, camera_matrix, direction, solve


def rot(yaw=0.0, pitch=0.0, roll=0.0):
    """Face rotation: roll about its own forward axis, then pitch (up = +), then yaw (image right = +)."""
    y, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, -math.sin(y)], [0, 1, 0], [math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def project(R, w=640, h=480, dist=600.0):
    rvec, _ = cv2.Rodrigues(R)
    pts, _ = cv2.projectPoints(MODEL_3D, rvec, np.array([[0.0], [0.0], [dist]]), camera_matrix(w, h), None)
    return pts.reshape(5, 2)


@pytest.mark.parametrize("yaw,pitch,roll", [(0, 0, 0), (25, 0, 0), (-25, 0, 0), (0, 20, 0), (0, -20, 0), (0, 0, 15), (15, -10, 5)])
def test_recovers_angles(yaw, pitch, roll):
    lm = project(rot(yaw, pitch, roll))
    y, p, r, *_ = solve(lm, 640, 480)
    assert abs(y - yaw) < 3 and abs(p - pitch) < 3 and abs(r - roll) < 3


def test_sign_conventions():
    # face turned to the image right -> nose moves right of the eye midpoint
    lm = project(rot(yaw=30))
    assert lm[2, 0] > (lm[0, 0] + lm[1, 0]) / 2
    # face up -> nose moves up towards the eyes
    up = project(rot(pitch=25))
    level = project(rot())
    assert (up[2, 1] - up[0, 1]) < (level[2, 1] - level[0, 1])


def test_direction_labels():
    assert direction(2, 3, 15) == "center"
    assert direction(30, 5, 15) == "right"
    assert direction(-30, 5, 15) == "left"
    assert direction(5, 25, 15) == "up"
    assert direction(5, -25, 15) == "down"
