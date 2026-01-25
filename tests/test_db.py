import time

import numpy as np
import pytest

from netrax.db.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "data")
    yield s
    s.close()


def test_migrations_apply_once(tmp_path):
    s = Store(tmp_path)
    v = s.schema_version()
    assert v >= 1
    s.close()
    s2 = Store(tmp_path)          # reopening must not re-run migrations
    assert s2.schema_version() == v
    s2.close()


def test_people_crud(store):
    pid = store.add_person("Mind", "owner")
    assert store.find_person("mind")["id"] == pid          # case-insensitive
    with pytest.raises(ValueError):
        store.add_person("MIND")
    with pytest.raises(ValueError):
        store.add_person("x", role="admin")
    with pytest.raises(ValueError):
        store.add_person("unknown")
    p = store.update_person(pid, name="Sid", role="allowed")
    assert (p["name"], p["role"]) == ("Sid", "allowed")
    other = store.add_person("Guest")
    with pytest.raises(ValueError):
        store.update_person(other, name="sid")
    assert [x["name"] for x in store.list_people()] == ["Guest", "Sid"]


def test_embeddings_are_normalised_and_cascade(store):
    pid = store.add_person("A")
    photo = store.person_dir(pid) / "1.jpg"
    photo.write_bytes(b"jpg")
    v0 = store.version
    eid = store.add_embedding(pid, np.full(128, 3.0), photo_path=photo, quality=0.9)
    assert store.version > v0
    ids, mat = store.all_embeddings()
    assert ids.tolist() == [pid] and mat.shape == (1, 128)
    assert np.isclose(np.linalg.norm(mat[0]), 1.0)
    assert store.list_people()[0]["n_faces"] == 1
    assert store.list_people()[0]["cover"] == f"people/{pid}/1.jpg"
    assert store.list_photos(pid)[0]["id"] == eid
    store.delete_person(pid)
    assert store.all_embeddings()[1].shape[0] == 0
    assert not photo.exists()


def test_delete_single_photo(store):
    pid = store.add_person("A")
    p = store.person_dir(pid) / "a.jpg"
    p.write_bytes(b"x")
    eid = store.add_embedding(pid, np.ones(128), photo_path=p)
    assert store.delete_embedding(eid)
    assert not p.exists()
    assert not store.delete_embedding(eid)


def test_events_query_and_prune(store):
    pid = store.add_person("A")
    snap = store.data_dir / "snapshots" / "u.jpg"
    snap.parent.mkdir()
    snap.write_bytes(b"x")
    now = time.time()
    store.add_event("seen", "A", ts=now - 10, person_id=pid, confidence=0.7)
    store.add_event("unknown", "unknown", ts=now - 5, snapshot_path=snap, data={"track": 3})
    store.add_event("gesture", "thumbs_up", ts=now)
    rows, total = store.query_events()
    assert total == 3 and [r["type"] for r in rows] == ["gesture", "unknown", "seen"]
    rows, total = store.query_events(types=["unknown"])
    assert total == 1 and rows[0]["data"] == {"track": 3} and rows[0]["snapshot_path"] == "snapshots/u.jpg"
    rows, total = store.query_events(since=now - 6, until=now - 1)
    assert [r["type"] for r in rows] == ["unknown"]
    assert store.query_events(person_id=pid)[1] == 1
    assert store.query_events(limit=1, offset=1)[0][0]["type"] == "unknown"
    assert store.event_counts() == {"seen": 1, "unknown": 1, "gesture": 1}
    assert store.prune_events(now - 1) == 2
    assert not snap.exists()
    store.delete_person(pid)
    assert store.query_events()[1] == 1
