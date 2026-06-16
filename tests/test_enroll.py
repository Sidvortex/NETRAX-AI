import time

import numpy as np
import pytest

from netrax.core import enroll as E
from netrax.core.types import Face, FrameState
from netrax.db.store import Store


def textured(h=480, w=640, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def face_at(x=200, y=120, size=160, nose_dx=0.0):
    lm = np.array([[x + 50, y + 60], [x + 110, y + 60], [x + 80 + nose_dx, y + 95], [x + 55, y + 125], [x + 105, y + 125]], float)
    return Face(bbox=np.array([x, y, size, size], float), score=0.95, landmarks=lm, raw=np.zeros(15), track_id=1)


def state(*faces):
    st = FrameState(0, 640, 480)
    st.results["face_detect"] = list(faces)
    return st


@pytest.fixture
def enroller(settings, monkeypatch):
    store = Store(settings.data_dir)
    e = E.Enroller(settings, store)
    rng = np.random.default_rng(1)
    base = rng.normal(size=128)

    def fake_embed(frame, face):
        v = base + rng.normal(scale=0.3, size=128)
        return (v / np.linalg.norm(v)).astype(np.float32)

    monkeypatch.setattr(e, "embed", fake_embed)
    monkeypatch.setattr(e, "_recognizer", lambda: None)
    monkeypatch.setattr(E, "MIN_INTERVAL_S", 0.0)
    yield e
    store.close()


def test_quality_gate():
    img = textured()
    assert E.face_quality(img, face_at())[0]
    assert E.face_quality(img, face_at(size=60))[1] == "come closer to the camera"
    assert E.face_quality(img, face_at(nose_dx=40))[1] == "look at the camera"
    assert E.face_quality(np.full((480, 640, 3), 128, np.uint8), face_at())[1] == "hold still"


def test_session_captures_then_saves(enroller):
    s = enroller.start("Mind", "owner", target=4)
    img = textured()
    enroller(img, state())
    assert s.hint == "no face in view"
    enroller(img, state(face_at(), face_at(x=400)))
    assert s.hint == "only one person in view, please"
    for i in range(6):
        enroller(textured(seed=i), state(face_at()))
    assert s.state == "done" and len(s.captures) == 4
    p = enroller.store.find_person("Mind")
    assert p["role"] == "owner"
    photos = enroller.store.list_photos(p["id"])
    assert len(photos) == 4
    assert all((enroller.store.data_dir / ph["photo_path"]).exists() for ph in photos)


def test_duplicate_name_and_parallel_sessions(enroller):
    enroller.store.add_person("Taken")
    with pytest.raises(ValueError, match="already exists"):
        enroller.start("taken")
    enroller.start("A")
    with pytest.raises(ValueError, match="already enrolling"):
        enroller.start("B")
    enroller.cancel()
    assert enroller.status()["state"] == "cancelled"
    assert enroller.store.find_person("A") is None


def test_timeout_without_enough_frames_saves_nothing(enroller):
    s = enroller.start("Late", timeout_s=0.01)
    time.sleep(0.02)
    enroller(textured(), state(face_at()))
    assert s.state == "failed"
    assert enroller.store.find_person("Late") is None
