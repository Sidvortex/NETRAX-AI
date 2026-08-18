import numpy as np

from netrax.detectors import mediapipe_onnx as mp


def test_anchor_counts_and_layout():
    assert mp.PALM_ANCHORS.shape == (2016, 2)
    assert mp.PERSON_ANCHORS.shape == (2254, 2)
    # first cell of the 24x24 map, 2 anchors, then the next cell along x
    assert np.allclose(mp.PALM_ANCHORS[:3], [[0.5 / 24, 0.5 / 24]] * 2 + [[1.5 / 24, 0.5 / 24]])
    assert np.allclose(mp.PERSON_ANCHORS[-1], [6.5 / 7, 6.5 / 7])


def test_letterbox_pad_bias():
    img = np.zeros((480, 640, 3), np.uint8)
    out, bias = mp.letterbox(img, 192)
    assert out.shape == (192, 192, 3)
    assert bias.tolist() == [0, 80]   # 24 px of padding at 192 scale = 80 px in the original


def test_decode_single_box():
    anchors = np.array([[0.5, 0.5], [0.1, 0.1]], np.float32)
    boxes = np.zeros((2, 4 + 2), np.float32)
    boxes[0, 2:4] = 96          # w, h = half the input
    scores = np.array([5.0, -5.0])
    out = mp.decode_ssd(boxes, scores, anchors, 192, (640, 640), np.array([0, 0]), 1, 0.5, 0.3)
    assert out.shape == (1, 7)
    assert np.allclose(out[0, :4], [160, 160, 480, 480])
    assert np.allclose(out[0, 4:6], [320, 320])
    assert out[0, -1] > 0.99
