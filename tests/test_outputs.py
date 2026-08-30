import json

from netrax.outputs.jsonl import JsonlWriter


def test_jsonl_appends_and_respects_toggle(settings):
    w = JsonlWriter(settings)
    w({"type": "gesture", "label": "peace"})
    w({"type": "object", "label": "cup"})
    lines = w.path.read_text().splitlines()
    assert [json.loads(x)["label"] for x in lines] == ["peace", "cup"]
    settings.update({"outputs": {"jsonl": {"enabled": False}}}, save=False)
    w({"type": "x", "label": "y"})
    assert len(w.path.read_text().splitlines()) == 2


import numpy as np  # noqa: E402

from netrax.core.types import Face, FrameState, Identity  # noqa: E402
from netrax.outputs.mqtt import MqttPublisher, RateLimiter, head_track_payload, primary_face  # noqa: E402


class FakeClient:
    def __init__(self, cid):
        self.sent = []

    def will_set(self, *a, **k):
        self.will = a

    def reconnect_delay_set(self, *a):
        pass

    def connect_async(self, host, port, keepalive):
        self.target = (host, port)

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def publish(self, topic, payload, qos=0, retain=False):
        self.sent.append((topic, payload, retain))

        class R:
            def wait_for_publish(self, t):
                pass

        return R()


def test_head_track_format_and_clamp():
    face = {"bbox": [270, 190, 100, 100]}
    p, xy = head_track_payload(face, 640, 480)
    assert p == "0.500,0.500,1" and xy == (0.5, 0.5)
    p, _ = head_track_payload(None, 640, 480, last_xy=(0.25, 0.75))
    assert p == "0.250,0.750,0"
    p, _ = head_track_payload({"bbox": [-50, -50, 20, 20]}, 640, 480)
    assert p == "0.000,0.000,1"


def test_primary_face_prefers_verified_owner():
    faces = [
        {"bbox": [0, 0, 300, 300], "verified": False, "role": "unknown"},
        {"bbox": [0, 0, 50, 50], "verified": True, "role": "allowed"},
        {"bbox": [0, 0, 40, 40], "verified": True, "role": "owner"},
    ]
    assert primary_face(faces)["role"] == "owner"
    assert primary_face(faces[:1])["bbox"][2] == 300
    assert primary_face([]) is None


def test_rate_limiter():
    rl = RateLimiter()
    assert rl.ready("a", 10, now=0.0)
    assert not rl.ready("a", 10, now=0.05)
    assert rl.ready("a", 10, now=0.11)


def _state(face=True, name="Mind"):
    st = FrameState(1, 640, 480, ts=100.0)
    st.results["face_detect"] = []
    st.results["face_recog"] = []
    if face:
        st.results["face_detect"] = [Face(np.array([270, 190, 100, 100.0]), 0.9, np.zeros((5, 2)), np.zeros(15), 1)]
        st.results["face_recog"] = [Identity(1, name, "owner", 0.8, True, person_id=1)]
    return st


def test_publisher_topics(settings):
    settings.update({"outputs": {"mqtt": {"enabled": True, "head_track_hz": 60, "rate_hz": 60}}}, save=False)
    pub = MqttPublisher(settings, status_fn=lambda: {"fps": 20, "cpu": 10, "camera": {"state": "live"}, "modules": {}},
                        client_factory=FakeClient)
    pub.apply()
    assert pub.client.target == ("192.168.1.39", 1883)
    assert pub.client.will[0] == "netrax/status"
    pub.connected = True
    pub(None, _state())
    pub(None, _state())
    topics = [t for t, _, _ in pub.client.sent]
    assert topics.count("netrax/identity") == 1               # only on change
    assert ("jinx/head_track", "0.500,0.500,1", False) in pub.client.sent
    assert "netrax/faces" in topics and "netrax/status" in topics
    pub(None, _state(face=False))
    assert ("jinx/head_track", "0.500,0.500,0", False) in pub.client.sent
    ident = [json.loads(p) for t, p, _ in pub.client.sent if t == "netrax/identity"]
    assert ident[0]["name"] == "Mind" and ident[-1]["name"] is None
    pub.on_event({"type": "gesture", "label": "peace", "data": {"hand": "right"}, "ts": 1.0, "confidence": 0.9})
    assert json.loads(pub.client.sent[-1][1])["gesture"] == "peace"
    settings.update({"outputs": {"mqtt": {"enabled": False}}}, save=False)
    assert pub.client is None and pub.status() is None
