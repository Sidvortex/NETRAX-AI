import math

import numpy as np
import pytest

from netrax.detectors.gestures import classify, finger_states

X = {"index": -0.3, "middle": 0.0, "ring": 0.28, "pinky": 0.52}


def hand(extended=("thumb", "index", "middle", "ring", "pinky"), thumb_up=False):
    """Synthetic right hand, palm facing the camera, fingers pointing up (image -y), 100 px scale."""
    lm = np.zeros((21, 3))
    lm[0] = (0, 0, 0)
    for name, (mcp, pip, dip, tip) in {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
                                       "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}.items():
        x = X[name]
        lm[mcp, :2] = (x, -1.0)
        if name in extended:
            lm[pip, :2], lm[dip, :2], lm[tip, :2] = (x, -1.4), (x, -1.65), (x, -1.9)
        else:  # curled down into the palm
            lm[pip, :2], lm[dip, :2], lm[tip, :2] = (x, -1.25), (x, -1.0), (x, -0.8)
    if thumb_up:   # fist with the thumb pointing up
        lm[1, :2], lm[2, :2], lm[3, :2], lm[4, :2] = (-0.4, -0.5), (-0.55, -0.9), (-0.6, -1.3), (-0.62, -1.7)
    elif "thumb" in extended:
        lm[1, :2], lm[2, :2], lm[3, :2], lm[4, :2] = (-0.35, -0.3), (-0.65, -0.55), (-0.9, -0.75), (-1.15, -0.95)
    else:          # tucked across the palm
        lm[1, :2], lm[2, :2], lm[3, :2], lm[4, :2] = (-0.35, -0.3), (-0.45, -0.6), (-0.25, -0.75), (-0.05, -0.8)
    lm[:, :2] *= 100
    lm[:, :2] += (300, 300)
    return lm


def rotate(lm, deg):
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    out = lm.copy()
    out[:, :2] = (lm[:, :2] - lm[0, :2]) @ R.T + lm[0, :2]
    return out


@pytest.mark.parametrize("lm,expected", [
    (hand(), "open_palm"),
    (hand(extended=()), "fist"),
    (hand(extended=(), thumb_up=True), "thumbs_up"),
    (hand(extended=("index",)), "point"),
    (hand(extended=("thumb", "index")), "point"),
    (hand(extended=("index", "middle")), "peace"),
    (hand(extended=("index", "middle", "ring")), "none"),
])
def test_classify(lm, expected):
    assert classify(lm) == expected


def test_finger_states():
    s = finger_states(hand(extended=("index", "middle")))
    assert s == {"index": True, "middle": True, "ring": False, "pinky": False, "thumb": False}


@pytest.mark.parametrize("deg", [-60, 35, 90, 180])
def test_rotation_invariant(deg):
    assert classify(rotate(hand(), deg)) == "open_palm"
    assert classify(rotate(hand(extended=("index", "middle")), deg)) == "peace"
    assert classify(rotate(hand(extended=()), deg)) == "fist"
