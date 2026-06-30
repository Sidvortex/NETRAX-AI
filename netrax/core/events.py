"""Turns per-frame results into events: who was seen, unknown faces, gestures, objects, posture.

Events go to the SQLite DB (Events page), and to any listeners (JSONL file,
MQTT). Cooldowns keep the timeline readable: a person sitting at the desk is
"seen" once every couple of minutes, not 15 times a second.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from netrax.core import draw as D
from netrax.core.types import FrameState
from netrax.db.store import Store

log = logging.getLogger("netrax.events")

Listener = Callable[[dict], None]


class EventBus:
    def __init__(self, settings, store: Store):
        self.settings = settings
        self.store = store
        self.listeners: list[Listener] = []
        self.recent: deque[dict] = deque(maxlen=50)
        self._lock = threading.Lock()
        self._seen_logged: dict[int, float] = {}        # person_id -> ts of last "seen" event
        self._seen_touch: dict[int, float] = {}
        self._unknown_tracks: set[int] = set()
        self._last_unknown = float("-inf")
        self._gesture_last: dict[str, float] = {}
        self._stable_gestures: set[str] = set()
        self._object_last: dict[str, float] = {}
        self._slouch_since: float | None = None
        self._slouch_logged = False
        self._last_prune = 0.0

    def on_event(self, fn: Listener) -> None:
        self.listeners.append(fn)

    # emit ------------------------------------------------------------------------
    def emit(self, type: str, label: str, frame: np.ndarray | None = None, box=None, snapshot: bool = False,
             **kw) -> dict:
        ts = kw.pop("ts", None) or time.time()
        snap = self._snapshot(frame, box, type, label, ts) if snapshot and frame is not None else None
        eid = self.store.add_event(type, label, ts=ts, snapshot_path=snap, **kw)
        evt = {"id": eid, "ts": round(ts, 3), "type": type, "label": label,
               "snapshot_url": f"/media/{self.store.rel(snap)}" if snap else None}
        evt.update({k: v for k, v in kw.items() if v is not None})
        with self._lock:
            self.recent.appendleft(evt)
        for fn in list(self.listeners):
            try:
                fn(evt)
            except Exception:  # noqa: BLE001
                log.exception("event listener failed")
        return evt

    def _snapshot(self, frame: np.ndarray, box, type: str, label: str, ts: float) -> Path:
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        d = self.store.data_dir / "snapshots" / day
        d.mkdir(parents=True, exist_ok=True)
        img = frame.copy()
        if box is not None:
            x, y, w, h = box
            D.corner_box(img, x, y, x + w, y + h, D.RED if type == "unknown" else D.GREEN, 2)
        h, w = img.shape[:2]
        s = min(1.0, 960 / max(h, w))
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        safe = "".join(c if c.isalnum() else "_" for c in label)[:24]
        p = d / f"{time.strftime('%H%M%S', time.localtime(ts))}_{int(ts * 1000) % 1000:03d}_{type}_{safe}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return p

    # hook -------------------------------------------------------------------------
    def __call__(self, frame: np.ndarray, state: FrameState) -> None:
        ev = self.settings.cfg.events
        now = state.ts
        self._faces(frame, state, ev, now)
        self._gestures(state, ev, now)
        self._objects(state, ev, now)
        self._posture(state, ev, now)
        if ev.keep_days and now - self._last_prune > 3600:
            self._last_prune = now
            n = self.store.prune_events(now - ev.keep_days * 86400)
            if n:
                log.info("pruned %d events older than %d days", n, ev.keep_days)

    def _faces(self, frame, state: FrameState, ev, now: float) -> None:
        if "face_recog" not in state.results:
            return
        boxes = {f.track_id: f.bbox for f in state.get("face_detect")}
        live = set(boxes)
        self._unknown_tracks &= live           # forget tracks that left
        for ident in state.get("face_recog"):
            if ident.verified and ident.person_id is not None:
                pid = ident.person_id
                if now - self._seen_touch.get(pid, float("-inf")) > 5:
                    self._seen_touch[pid] = now
                    self.store.touch_seen(pid, now)
                if now - self._seen_logged.get(pid, float("-inf")) >= ev.seen_cooldown_s:
                    self._seen_logged[pid] = now
                    self.emit("seen", ident.name, frame, boxes.get(ident.track_id), snapshot=ev.snapshot_seen,
                              ts=now, person_id=pid, track_id=ident.track_id,
                              confidence=round(ident.confidence, 3), data={"role": ident.role})
            elif (ident.name == "unknown" and ident.attempts >= ev.unknown_after
                  and ident.track_id not in self._unknown_tracks
                  and now - self._last_unknown >= ev.unknown_cooldown_s):
                self._unknown_tracks.add(ident.track_id)
                self._last_unknown = now
                box = boxes.get(ident.track_id)
                self.emit("unknown", "unknown", frame, box, snapshot=True, ts=now, track_id=ident.track_id,
                          confidence=round(ident.confidence, 3),
                          data={"bbox": [round(float(v)) for v in box] if box is not None else None})

    def _gestures(self, state: FrameState, ev, now: float) -> None:
        if "hands" not in state.results:
            return
        current = {h.gesture for h in state.get("hands") if h.gesture != "none"}
        for g in current - self._stable_gestures:
            if now - self._gesture_last.get(g, float("-inf")) >= ev.gesture_cooldown_s:
                self._gesture_last[g] = now
                hand = next(h for h in state.get("hands") if h.gesture == g)
                self.emit("gesture", g, ts=now, confidence=round(hand.score, 3),
                          data={"hand": hand.handedness})
        self._stable_gestures = current

    def _objects(self, state: FrameState, ev, now: float) -> None:
        if "objects" not in state.ran:
            return
        best: dict[str, float] = {}
        counts: dict[str, int] = {}
        for d in state.get("objects"):
            if d.label in ev.object_ignore:
                continue
            best[d.label] = max(best.get(d.label, 0), d.score)
            counts[d.label] = counts.get(d.label, 0) + 1
        for label, score in best.items():
            if now - self._object_last.get(label, float("-inf")) >= ev.object_cooldown_s:
                self.emit("object", label, ts=now, confidence=round(score, 3), data={"count": counts[label]})
            self._object_last[label] = now     # keep refreshing while it stays in view

    def _posture(self, state: FrameState, ev, now: float) -> None:
        if "pose" not in state.results:
            return
        slouch = any(p.posture == "slouching" for p in state.get("pose"))
        if not slouch:
            self._slouch_since, self._slouch_logged = None, False
            return
        if self._slouch_since is None:
            self._slouch_since = now
        if not self._slouch_logged and now - self._slouch_since >= ev.posture_after_s:
            self._slouch_logged = True
            self.emit("posture", "slouching", ts=now, data={"for_s": round(now - self._slouch_since)})
