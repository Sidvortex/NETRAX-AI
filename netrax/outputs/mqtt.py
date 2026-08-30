"""Optional MQTT output (the JINX link). OFF by default; NETRAX never needs it.

Topics:
  jinx/head_track   "x,y,d"   face centre normalised 0..1, d=1 if a face is seen  (<= head_track_hz)
  netrax/identity   JSON      primary face identity, published when it changes
  netrax/faces      JSON      all faces                                         (<= rate_hz)
  netrax/pose       JSON      pose keypoints + posture                          (<= rate_hz)
  netrax/gesture    JSON      each gesture event
  netrax/objects    JSON      object detections                                 (<= rate_hz)
  netrax/status     JSON      fps, cpu, modules; retained; {"online": false} as last will
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

import numpy as np

from netrax.core.types import FrameState

log = logging.getLogger("netrax.mqtt")

ROLE_RANK = {"owner": 0, "allowed": 1, "unknown": 2, "pending": 3}


class RateLimiter:
    def __init__(self):
        self.last: dict[str, float] = {}

    def ready(self, key: str, hz: float, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if now - self.last.get(key, float("-inf")) >= 1.0 / hz:
            self.last[key] = now
            return True
        return False


def primary_face(faces: list[dict]) -> dict | None:
    """The face JINX should look at: verified owner > verified anyone > largest face."""
    if not faces:
        return None

    def key(f):
        verified = f.get("verified", False)
        rank = ROLE_RANK.get(f.get("role", "pending"), 3) if verified else 4
        area = f["bbox"][2] * f["bbox"][3]
        return (rank, -area)

    return min(faces, key=key)


def head_track_payload(face: dict | None, width: int, height: int, last_xy=(0.5, 0.5)) -> tuple[str, tuple]:
    if face is None:
        x, y = last_xy
        return f"{x:.3f},{y:.3f},0", last_xy
    bx, by, bw, bh = face["bbox"]
    x = min(1.0, max(0.0, (bx + bw / 2) / width))
    y = min(1.0, max(0.0, (by + bh / 2) / height))
    return f"{x:.3f},{y:.3f},1", (x, y)


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


def _paho_client(client_id: str):
    import paho.mqtt.client as mqtt

    return mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=True)


class MqttPublisher:
    def __init__(self, settings, status_fn: Callable[[], dict] | None = None, client_factory=_paho_client):
        self.settings = settings
        self.status_fn = status_fn
        self.client_factory = client_factory
        self.client = None
        self.connected = False
        self.error: str | None = None
        self.rl = RateLimiter()
        self._target = None
        self._last_xy = (0.5, 0.5)
        self._had_face = False
        self._identity_key = None
        self._sent = 0
        settings.on_change(lambda old, new: self.apply() if old.outputs.mqtt != new.outputs.mqtt else None)

    # connection ----------------------------------------------------------------
    def apply(self) -> None:
        c = self.settings.cfg.outputs.mqtt
        target = (c.host, c.port, c.client_id) if c.enabled else None
        if target == self._target:
            return
        self.stop()
        self._target = target
        if target:
            self.start()

    def start(self) -> None:
        c = self.settings.cfg.outputs.mqtt
        try:
            cl = self.client_factory(c.client_id)
            cl.on_connect = self._on_connect
            cl.on_disconnect = self._on_disconnect
            cl.will_set("netrax/status", _dumps({"online": False}), qos=1, retain=True)
            cl.reconnect_delay_set(1, 30)
            cl.connect_async(c.host, c.port, keepalive=30)
            cl.loop_start()
            self.client = cl
            log.info("MQTT: connecting to %s:%d", c.host, c.port)
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            log.error("MQTT start failed: %s", e)

    def stop(self) -> None:
        cl, self.client = self.client, None
        self._target = None
        if cl is not None:
            try:
                if self.connected:
                    cl.publish("netrax/status", _dumps({"online": False}), qos=1, retain=True).wait_for_publish(1)
                cl.disconnect()
                cl.loop_stop()
            except Exception:  # noqa: BLE001
                pass
        self.connected = False

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        ok = not getattr(reason_code, "is_failure", False) and reason_code == 0
        self.connected = bool(ok)
        self.error = None if ok else str(reason_code)
        log.info("MQTT connected" if ok else f"MQTT connect refused: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        if self.connected:
            log.warning("MQTT disconnected (%s), retrying", reason_code)
        self.connected = False

    def publish(self, topic: str, payload: str, retain: bool = False) -> None:
        if self.client is not None and self.connected:
            self.client.publish(topic, payload, qos=0, retain=retain)
            self._sent += 1

    def status(self) -> dict | None:
        c = self.settings.cfg.outputs.mqtt
        if not c.enabled:
            return None
        return {"connected": self.connected, "broker": f"{c.host}:{c.port}", "error": self.error, "sent": self._sent}

    # hooks ---------------------------------------------------------------------
    def __call__(self, frame: np.ndarray, state: FrameState) -> None:
        if self.client is None or not self.connected:
            return
        c = self.settings.cfg.outputs.mqtt
        faces = state.faces()
        face = primary_face(faces)
        seen = face is not None
        send_head = ((seen or self._had_face) and self.rl.ready("head", c.head_track_hz)) or \
            (not seen and self.rl.ready("head_idle", 1.0))      # no face: "x,y,0" once a second
        if send_head:
            payload, self._last_xy = head_track_payload(face, state.width, state.height, self._last_xy)
            self.publish("jinx/head_track", payload)
        self._had_face = seen

        ident = None
        if face is not None and "name" in face:
            ident = {k: face.get(k) for k in ("name", "role", "verified", "confidence", "track_id", "person_id")}
        key = (ident or {}).get("name"), (ident or {}).get("verified")
        if key != self._identity_key and (ident is None or ident.get("name") != "..."):
            self._identity_key = key
            self.publish("netrax/identity", _dumps(dict(ident or {"name": None}, ts=round(state.ts, 3))), retain=True)

        if self.rl.ready("faces", c.rate_hz) and "face_detect" in state.results:
            self.publish("netrax/faces", _dumps({"ts": round(state.ts, 3), "size": [state.width, state.height],
                                                 "faces": faces}))
        if "pose" in state.results and self.rl.ready("pose", c.rate_hz):
            self.publish("netrax/pose", _dumps({"ts": round(state.ts, 3), "poses": [p.to_dict() for p in state.get("pose")]}))
        if "objects" in state.ran and self.rl.ready("objects", c.rate_hz):
            self.publish("netrax/objects", _dumps({"ts": round(state.ts, 3),
                                                   "objects": [o.to_dict() for o in state.get("objects")]}))
        if self.status_fn and self.rl.ready("status", 1.0 / c.status_every_s):
            s = self.status_fn()
            self.publish("netrax/status", _dumps({
                "online": True, "ts": round(time.time(), 3), "fps": s.get("fps"), "cpu": s.get("cpu"),
                "camera": s.get("camera", {}).get("state"),
                "modules": {n: m["state"] for n, m in s.get("modules", {}).items() if m["state"] != "off"},
            }), retain=True)

    def on_event(self, evt: dict) -> None:
        if evt.get("type") == "gesture":
            self.publish("netrax/gesture", _dumps({"gesture": evt["label"], "hand": (evt.get("data") or {}).get("hand"),
                                                   "confidence": evt.get("confidence"), "ts": evt["ts"]}))
