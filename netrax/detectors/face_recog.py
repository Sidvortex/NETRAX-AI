"""Face recognition with OpenCV SFace (cv2.FaceRecognizerSF) against the people DB.

Recognition is per *track*, not per frame: a new face is recognised a few
times quickly, then re-checked every `recheck_every` frames. Its identity is a
majority vote over the last `votes` results, so one bad frame can't flip a
name. Output per face: name, role, confidence (cosine), verified.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from netrax.core import draw as D
from netrax.core.types import Face, FrameState, Identity
from netrax.detectors.base import Detector

PENDING = "..."
UNKNOWN = "unknown"
FAST_CHECKS = 3          # recognitions done back-to-back on a new track before slowing down


@dataclass
class _TrackMem:
    votes: deque = field(default_factory=lambda: deque(maxlen=5))   # (person_id | None, score)
    last_frame: int = -10**9
    last_seen: int = 0
    attempts: int = 0


class FaceRecognizer(Detector):
    name = "face_recog"
    model_keys = ("sface",)
    requires = ("face_detect",)

    def load(self) -> None:
        self.rec = cv2.FaceRecognizerSF.create(self.model_file("sface"), "")
        self.store = self.services.get("store")
        self.tracks: dict[int, _TrackMem] = {}
        self._db_version = -1
        self._ids = np.zeros(0, np.int64)
        self._mat = np.zeros((0, 128), np.float32)
        self._people: dict[int, dict] = {}

    def reset(self) -> None:
        self.tracks.clear()

    # embeddings ------------------------------------------------------------
    def embed(self, frame: np.ndarray, face: Face) -> tuple[np.ndarray, np.ndarray]:
        """(L2-normalised 128-d feature, 112x112 aligned face crop)."""
        aligned = self.rec.alignCrop(frame, face.raw)
        feat = self.rec.feature(aligned).astype(np.float32).ravel()
        return feat / (np.linalg.norm(feat) + 1e-9), aligned

    def _refresh_gallery(self) -> None:
        if self.store is None or self.store.version == self._db_version:
            return
        self._db_version = self.store.version
        self._ids, self._mat = self.store.all_embeddings()
        self._people = {p["id"]: p for p in self.store.list_people()}
        self.tracks.clear()     # people changed (e.g. someone just enrolled): re-recognise everyone

    def match(self, feat: np.ndarray) -> tuple[int | None, float]:
        """Best person and its cosine similarity (max over that person's embeddings)."""
        if len(self._ids) == 0:
            return None, 0.0
        sims = self._mat @ feat
        best = int(np.argmax(sims))
        return int(self._ids[best]), float(sims[best])

    # per frame ---------------------------------------------------------------
    def process(self, frame: np.ndarray, ctx: FrameState) -> list[Identity]:
        self._refresh_gallery()
        c = self.cfg
        idx = ctx.frame_idx
        out = []
        for face in ctx.get("face_detect"):
            mem = self.tracks.get(face.track_id)
            if mem is None or mem.votes.maxlen != c.votes:
                mem = self.tracks[face.track_id] = _TrackMem(votes=deque(maxlen=c.votes))
            mem.last_seen = idx
            interval = 2 if mem.attempts < FAST_CHECKS else c.recheck_every
            if idx - mem.last_frame >= interval:
                pid, score = self.match(self.embed(frame, face)[0])
                mem.votes.append((pid if score >= c.threshold else None, score))
                mem.attempts += 1
                mem.last_frame = idx
            out.append(self._identity(face, mem))
        # forget tracks that left a while ago
        for tid in [t for t, m in self.tracks.items() if idx - m.last_seen > 300]:
            del self.tracks[tid]
        return out

    def _identity(self, face: Face, mem: _TrackMem) -> Identity:
        votes = list(mem.votes)
        base = dict(track_id=face.track_id, attempts=mem.attempts, bbox=face.bbox)
        if not votes:
            return Identity(name=PENDING, role="pending", confidence=0.0, verified=False, **base)
        counts = Counter(pid for pid, _ in votes if pid is not None)
        if counts:
            pid, n = counts.most_common(1)[0]
            person = self._people.get(pid)
            if person is not None and n * 2 > len(votes):
                conf = float(np.mean([s for p, s in votes if p == pid]))
                return Identity(name=person["name"], role=person["role"], confidence=conf, verified=True,
                                person_id=pid, **base)
        best = max(s for _, s in votes)
        return Identity(name=UNKNOWN, role="unknown", confidence=best, verified=False, **base)

    def draw(self, frame: np.ndarray, results: list[Identity]) -> None:
        for r in results:
            if r.bbox is None:
                continue
            x, y, w, h = r.bbox
            color = D.ROLE_COLORS.get(r.role, D.CYAN) if r.verified or r.name != PENDING else D.YELLOW
            D.corner_box(frame, x, y, x + w, y + h, color, 2)
            if r.name == PENDING:
                text = "identifying..."
            elif r.verified:
                text = f"{r.name} {r.confidence * 100:.0f}%"
            else:
                text = f"UNKNOWN {max(0.0, r.confidence) * 100:.0f}%"
            D.label(frame, text, (x, y), color, 0.55)
