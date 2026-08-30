import numpy as np

from netrax.detectors.pose import BLAZE_TO_COCO, PostureJudge, limb_group, posture_metrics


def kp(nose_y=100.0, tilt=0.0, vis=1.0):
    k = np.zeros((17, 3))
    k[:, 2] = vis
    k[0] = (200, nose_y, vis)
    k[5] = (130, 200 - tilt, vis)    # left shoulder (image left here)
    k[6] = (270, 200 + tilt, vis)
    return k


def test_mapping_is_coco17():
    assert len(BLAZE_TO_COCO) == 17 and BLAZE_TO_COCO[5:7] == [11, 12]
    assert limb_group(5, 7) == "left" and limb_group(6, 8) == "right" and limb_group(5, 6) == "torso"


def test_metrics():
    m = posture_metrics(kp(nose_y=100), 0.5)
    assert m == {"head_ratio": round(100 / 140, 3), "shoulder_tilt": 0.0}
    assert posture_metrics(kp(vis=0.2), 0.5) is None


def test_judge_calibrates_then_flags_slouch_and_lean():
    j = PostureJudge(min_samples=5)
    for _ in range(4):
        assert j(posture_metrics(kp(nose_y=100), 0.5), 0.2, 12)[0] == "calibrating"
    assert j(posture_metrics(kp(nose_y=100), 0.5), 0.2, 12)[0] == "upright"
    assert j(posture_metrics(kp(nose_y=140), 0.5), 0.2, 12)[0] == "slouching"   # head dropped ~40%
    assert j(posture_metrics(kp(nose_y=105, tilt=25), 0.5), 0.2, 12)[0] == "leaning"
    assert j(None, 0.2, 12)[0] == "unknown"
