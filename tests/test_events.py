import numpy as np
import pytest

from netrax.core.events import EventBus
from netrax.core.types import Detection, Face, FrameState, Hand, Identity, Pose
from netrax.db.store import Store


@pytest.fixture
def bus(settings):
    store = Store(settings.data_dir)
    b = EventBus(settings, store)
    b.got = []
    b.on_event(b.got.append)
    yield b
    store.close()


def frame_state(ts, faces=(), idents=(), hands=None, objects=None, poses=None, ran=()):
    st = FrameState(0, 320, 240, ts=ts)
    st.results["face_detect"] = list(faces)
    st.results["face_recog"] = list(idents)
    if hands is not None:
        st.results["hands"] = hands
    if objects is not None:
        st.results["objects"] = objects
    if poses is not None:
        st.results["pose"] = poses
    st.ran = set(ran)
    return st


def face(tid):
    return Face(bbox=np.array([50, 50, 80, 80.0]), score=0.9, landmarks=np.zeros((5, 2)), raw=np.zeros(15), track_id=tid)


IMG = np.zeros((240, 320, 3), np.uint8)


def test_seen_respects_cooldown(bus):
    pid = bus.store.add_person("Mind", "owner")
    ident = Identity(track_id=1, name="Mind", role="owner", confidence=0.7, verified=True, person_id=pid, attempts=3)
    for t in (0, 1, 2, 200):
        bus(IMG, frame_state(1000 + t, [face(1)], [ident]))
    assert [e["type"] for e in bus.got] == ["seen", "seen"]
    assert bus.store.get_person(pid)["last_seen"] == 1200


def test_unknown_once_per_track_with_snapshot(bus):
    pending = Identity(track_id=7, name="unknown", role="unknown", confidence=0.1, verified=False, attempts=1)
    bus(IMG, frame_state(10, [face(7)], [pending]))
    assert bus.got == []          # not enough attempts yet
    sure = Identity(track_id=7, name="unknown", role="unknown", confidence=0.1, verified=False, attempts=3)
    for t in range(5):
        bus(IMG, frame_state(11 + t, [face(7)], [sure]))
    assert len(bus.got) == 1
    e = bus.got[0]
    assert e["type"] == "unknown" and e["snapshot_url"].startswith("/media/snapshots/")
    rows, _ = bus.store.query_events(types=["unknown"])
    assert (bus.store.data_dir / rows[0]["snapshot_path"]).exists()


def test_gestures_fire_on_change(bus):
    def hand(g):
        return Hand(bbox=np.zeros(4), landmarks=np.zeros((21, 3)), handedness="right", score=0.9, gesture=g)

    seq = ["none", "thumbs_up", "thumbs_up", "none", "peace", "thumbs_up"]
    for i, g in enumerate(seq):
        bus(IMG, frame_state(100 + i * 10, hands=[hand(g)]))
    assert [e["label"] for e in bus.got] == ["thumbs_up", "peace", "thumbs_up"]


def test_objects_ignore_and_cooldown(bus):
    objs = [Detection(np.zeros(4), "cup", 41, 0.8), Detection(np.zeros(4), "person", 0, 0.9)]
    for t in (0, 5, 10):
        bus(IMG, frame_state(t, objects=objs, ran={"objects"}))
    assert [(e["type"], e["label"]) for e in bus.got] == [("object", "cup")]
    bus(IMG, frame_state(100, objects=objs, ran={"objects"}))   # still in view: no new event
    assert len(bus.got) == 1
    bus(IMG, frame_state(200, objects=[], ran={"objects"}))
    bus(IMG, frame_state(300, objects=objs, ran={"objects"}))   # came back after cooldown
    assert len(bus.got) == 2


def test_posture_after_delay(bus):
    slouch = [Pose(bbox=np.zeros(4), keypoints=np.zeros((17, 3)), score=0.9, posture="slouching")]
    bus(IMG, frame_state(0, poses=slouch))
    bus(IMG, frame_state(10, poses=slouch))
    assert bus.got == []
    bus(IMG, frame_state(25, poses=slouch))
    bus(IMG, frame_state(30, poses=slouch))
    assert [e["label"] for e in bus.got] == ["slouching"]
