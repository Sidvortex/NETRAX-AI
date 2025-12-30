import numpy as np

from netrax.core.tracker import IouTracker, iou_xywh


def b(x, y, w=50, h=50):
    return np.array([x, y, w, h], float)


def test_iou():
    assert iou_xywh(b(0, 0), b(0, 0)) == 1.0
    assert iou_xywh(b(0, 0), b(100, 100)) == 0.0
    assert 0.3 < iou_xywh(b(0, 0), b(10, 10)) < 0.6


def test_ids_follow_moving_boxes():
    t = IouTracker()
    a, c = t.update([b(0, 0), b(200, 0)])
    assert a != c
    a2, c2 = t.update([b(205, 3), b(4, 2)])  # order swapped, small motion
    assert (a2, c2) == (c, a)


def test_track_ends_after_max_missed():
    t = IouTracker(max_missed=2)
    (tid,) = t.update([b(0, 0)])
    for _ in range(2):
        t.update([])
        assert t.ended == []
    t.update([])
    assert t.ended == [tid]
    (new,) = t.update([b(0, 0)])
    assert new != tid
