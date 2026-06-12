from collections import deque

import numpy as np

from netrax.core.types import Face
from netrax.db.store import Store
from netrax.detectors.face_recog import FaceRecognizer, _TrackMem


def make_rec(settings, store):
    r = FaceRecognizer(settings, {"store": store})
    r.store = store
    r.tracks = {}
    r._db_version = -1
    r._refresh_gallery()
    return r


def face(tid=1):
    return Face(bbox=np.array([0, 0, 50, 50.0]), score=0.9, landmarks=np.zeros((5, 2)), raw=np.zeros(15), track_id=tid)


def unit(seed):
    v = np.random.default_rng(seed).normal(size=128).astype(np.float32)
    return v / np.linalg.norm(v)


def test_match_picks_best_person(settings, tmp_path):
    store = Store(tmp_path)
    a, b = store.add_person("A", "owner"), store.add_person("B")
    va, vb = unit(1), unit(2)
    store.add_embedding(a, va)
    store.add_embedding(b, vb)
    rec = make_rec(settings, store)
    pid, score = rec.match(va * 0.9 + vb * 0.1)
    assert pid == a and score > 0.8
    assert rec.match(unit(3))[1] < 0.363


def test_majority_vote(settings, tmp_path):
    store = Store(tmp_path)
    a = store.add_person("A", "owner")
    store.add_embedding(a, unit(1))
    rec = make_rec(settings, store)
    mem = _TrackMem(votes=deque([(a, 0.7), (None, 0.2), (a, 0.6)], maxlen=5), attempts=3)
    ident = rec._identity(face(), mem)
    assert (ident.name, ident.role, ident.verified) == ("A", "owner", True)
    assert abs(ident.confidence - 0.65) < 1e-6
    mem = _TrackMem(votes=deque([(a, 0.7), (None, 0.2), (None, 0.1)], maxlen=5), attempts=3)
    ident = rec._identity(face(), mem)
    assert (ident.name, ident.verified) == ("unknown", False)
    assert ident.confidence == 0.7
    assert rec._identity(face(), _TrackMem()).name == "..."


def test_gallery_change_resets_tracks(settings, tmp_path):
    store = Store(tmp_path)
    rec = make_rec(settings, store)
    rec.tracks[5] = _TrackMem()
    pid = store.add_person("New")
    store.add_embedding(pid, unit(9))
    rec._refresh_gallery()
    assert rec.tracks == {} and len(rec._ids) == 1
