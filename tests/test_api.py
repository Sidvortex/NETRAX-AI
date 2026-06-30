import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from netrax.core.source import ArraySource
from netrax.hub.app import create_app
from netrax.runtime import Runtime

from .conftest import stub_factory


@pytest.fixture
def client(settings, frame):
    settings.update({"modules": {"face_recog": {"enabled": False}, "head_pose": {"enabled": False}},
                     "outputs": {"jsonl": {"enabled": False}}}, save=False)
    rt = Runtime(settings, source=ArraySource([frame], loop=True, fps=60), factory=stub_factory)
    app = create_app(rt)
    with TestClient(app) as c:
        deadline = time.time() + 5
        while rt.engine.state is None and time.time() < deadline:
            time.sleep(0.02)
        c.rt = rt
        yield c


def test_pages_and_static(client):
    for path in ("/", "/people", "/static/app.js", "/static/style.css"):
        assert client.get(path).status_code == 200


def test_status_and_state(client):
    s = client.get("/api/status").json()
    assert s["camera"]["state"] == "live"
    assert s["modules"]["face_detect"]["state"] == "ready"
    st = client.get("/api/state").json()
    assert st["faces"][0]["track_id"] == 1


def test_snapshot_is_jpeg(client):
    r = client.get("/snapshot.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_module_toggle(client):
    resp = client.post("/api/modules/objects", json={"enabled": True})
    r = resp.json()
    assert resp.status_code == 200, resp.text
    assert r["enabled"] is True
    assert client.app.state.rt.settings.cfg.modules.objects.enabled is True
    r = client.post("/api/modules/face_detect", json={"overlay": False}).json()
    assert r["overlay"] is False
    assert client.post("/api/modules/nope", json={"enabled": True}).status_code == 404


def test_people_endpoints(client):
    store = client.rt.store
    pid = store.add_person("Mind", "owner")
    photo = store.person_dir(pid) / "a.jpg"
    photo.write_bytes(b"\xff\xd8fake")
    eid = store.add_embedding(pid, np.ones(128), photo_path=photo)
    people = client.get("/api/people").json()
    assert people[0]["name"] == "Mind" and people[0]["cover_url"] == f"/media/people/{pid}/a.jpg"
    assert client.get(people[0]["cover_url"]).status_code == 200
    assert client.get("/media/../netrax.db").status_code == 404
    assert client.get("/media/netrax.db").status_code == 404
    r = client.patch(f"/api/people/{pid}", json={"name": "Sid", "role": "allowed"}).json()
    assert (r["name"], r["role"]) == ("Sid", "allowed")
    assert client.patch(f"/api/people/{pid}", json={"role": "boss"}).status_code == 422
    assert client.get(f"/api/people/{pid}/photos").json()[0]["id"] == eid
    assert client.delete(f"/api/photos/{eid}").status_code == 200
    assert client.delete(f"/api/people/{pid}").status_code == 200
    assert client.get("/api/people").json() == []
    assert client.delete(f"/api/people/{pid}").status_code == 404


def test_enroll_flow(client, monkeypatch):
    rt = client.rt
    monkeypatch.setattr(rt.enroller, "_recognizer", lambda: None)
    r = client.post("/api/enroll", json={"name": "Guest", "frames": 3}).json()
    assert r["state"] == "running" and r["target"] == 3
    assert client.post("/api/enroll", json={"name": "Other"}).status_code == 422   # one at a time
    assert client.delete("/api/enroll").json()["state"] == "cancelled"
