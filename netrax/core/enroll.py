"""Enrollment: capture ~10 good face frames of one person from the live camera.

A frame is only kept if there is exactly one face, it is big, sharp and
roughly frontal, and enough time has passed since the last capture (so the
person can move their head a little between shots). Nothing is written to
the DB until enough frames are captured.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from netrax import models
from netrax.core.types import Face, FrameState
from netrax.db.store import Store

log = logging.getLogger("netrax.enroll")

MIN_SCORE = 0.85
MIN_FACE_PX = 90
MIN_SHARPNESS = 40.0
MAX_YAW_OFFSET = 0.28     # nose offset from eye midpoint, as a fraction of eye distance
MIN_INTERVAL_S = 0.35
MAX_SIMILAR = 0.985       # skip near-identical frames
MIN_TO_SAVE = 3


def face_quality(frame: np.ndarray, face: Face) -> tuple[bool, str, float]:
    """(ok, hint, quality 0..1) for one detected face."""
    x, y, w, h = face.bbox
    if face.score < MIN_SCORE:
        return False, "face not clear, improve lighting", 0.0
    if min(w, h) < MIN_FACE_PX:
        return False, "come closer to the camera", 0.0
    re, le, nose = face.landmarks[0], face.landmarks[1], face.landmarks[2]
    eye_d = float(np.linalg.norm(le - re)) + 1e-6
    offset = abs(float(nose[0] - (re[0] + le[0]) / 2)) / eye_d
    if offset > MAX_YAW_OFFSET:
        return False, "look at the camera", 0.0
    H, W = frame.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(W, int(x + w)), min(H, int(y + h))
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return False, "face is at the edge of the frame", 0.0
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharp < MIN_SHARPNESS:
        return False, "hold still", 0.0
    q = min(1.0, face.score) * min(1.0, sharp / 300.0) * (1.0 - offset)
    return True, "", float(q)


def face_photo(frame: np.ndarray, face: Face, size: int = 200) -> np.ndarray:
    """Square crop around the face with some margin, for the People page."""
    x, y, w, h = face.bbox
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * 1.5
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)
    pad = max(0, -x1, -y1, x2 - W, y2 - H)
    img = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else frame
    crop = img[y1 + pad:y2 + pad, x1 + pad:x2 + pad]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


@dataclass
class _Capture:
    feat: np.ndarray
    photo: np.ndarray
    quality: float


@dataclass
class EnrollSession:
    name: str
    role: str
    target: int
    timeout_s: float
    person_id: int | None = None          # set to add photos to an existing person
    started: float = field(default_factory=time.time)
    captures: list[_Capture] = field(default_factory=list)
    hint: str = "look at the camera"
    state: str = "running"                # running | done | cancelled | failed
    error: str | None = None
    result_person_id: int | None = None
    last_capture: float = 0.0

    def status(self) -> dict:
        return {
            "state": self.state, "name": self.name, "role": self.role,
            "captured": len(self.captures), "target": self.target, "hint": self.hint,
            "error": self.error, "person_id": self.result_person_id,
            "elapsed_s": round(time.time() - self.started, 1), "timeout_s": self.timeout_s,
        }


class Enroller:
    """Plugs into the engine as a hook; one session at a time."""

    def __init__(self, settings, store: Store):
        self.settings = settings
        self.store = store
        self.session: EnrollSession | None = None
        self._lock = threading.Lock()
        self._rec = None

    def _recognizer(self):
        if self._rec is None:
            self._rec = cv2.FaceRecognizerSF.create(str(models.ensure(self.settings.models_dir, "sface")), "")
        return self._rec

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        rec = self._recognizer()
        f = rec.feature(rec.alignCrop(frame, face.raw)).astype(np.float32).ravel()
        return f / (np.linalg.norm(f) + 1e-9)

    # control -----------------------------------------------------------------
    def start(self, name: str, role: str = "allowed", target: int = 10, timeout_s: float = 60.0,
              add_to_existing: bool = False) -> EnrollSession:
        from netrax.db.store import _check_role, _clean_name

        name = _clean_name(name)
        _check_role(role)
        existing = self.store.find_person(name)
        if existing and not add_to_existing:
            raise ValueError(f"{existing['name']!r} already exists; add photos to them from the People page")
        self._recognizer()   # load SFace now so errors show up immediately
        with self._lock:
            if self.session and self.session.state == "running":
                raise ValueError(f"already enrolling {self.session.name!r}")
            self.session = EnrollSession(name=name, role=role, target=max(MIN_TO_SAVE, min(target, 30)),
                                         timeout_s=timeout_s, person_id=existing["id"] if existing else None)
            return self.session

    def cancel(self) -> None:
        with self._lock:
            if self.session and self.session.state == "running":
                self.session.state = "cancelled"

    def status(self) -> dict:
        s = self.session
        return s.status() if s else {"state": "idle"}

    # hook ----------------------------------------------------------------------
    def __call__(self, frame: np.ndarray, state: FrameState) -> None:
        s = self.session
        if s is None or s.state != "running":
            return
        now = time.time()
        if now - s.started > s.timeout_s:
            if len(s.captures) >= MIN_TO_SAVE:
                self._finish(s)
            else:
                s.state, s.error = "failed", "timed out before enough good frames were captured"
            return
        faces = state.get("face_detect")
        if "face_detect" not in state.results:
            s.hint = "face detection is off: enable it to enroll"
            return
        if not faces:
            s.hint = "no face in view"
            return
        if len(faces) > 1:
            s.hint = "only one person in view, please"
            return
        if now - s.last_capture < MIN_INTERVAL_S:
            return
        ok, hint, q = face_quality(frame, faces[0])
        if not ok:
            s.hint = hint
            return
        feat = self.embed(frame, faces[0])
        if s.captures:
            sims = np.array([c.feat @ feat for c in s.captures])
            if sims.max() > MAX_SIMILAR:
                s.hint = "turn your head a little"
                return
            if len(s.captures) >= 2 and np.median(sims) < 0.3:
                s.hint = "a different face? keep the same person in view"
                return
        s.captures.append(_Capture(feat, face_photo(frame, faces[0]), q))
        s.last_capture = now
        s.hint = "good, keep moving slightly" if len(s.captures) < s.target else "saving"
        if len(s.captures) >= s.target:
            self._finish(s)

    def _finish(self, s: EnrollSession) -> None:
        try:
            pid = s.person_id or self.store.add_person(s.name, s.role)
            d = self.store.person_dir(pid)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            for i, c in enumerate(s.captures):
                p = d / f"{stamp}_{i:02d}.jpg"
                cv2.imwrite(str(p), c.photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.store.add_embedding(pid, c.feat, photo_path=p, quality=c.quality)
            s.result_person_id = pid
            s.state = "done"
            log.info("enrolled %s (%d faces)", s.name, len(s.captures))
        except Exception as e:  # noqa: BLE001
            s.state, s.error = "failed", str(e)
            log.exception("enrollment failed")

    # offline -------------------------------------------------------------------
    def enroll_images(self, name: str, role: str, paths: list[Path], detector) -> tuple[int, list[str]]:
        """Enroll from photo files (largest face in each). Returns (person_id, skipped files)."""
        existing = self.store.find_person(name)
        caps, skipped = [], []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                skipped.append(f"{p.name}: not an image")
                continue
            rows = detector.detect_raw(img)
            if len(rows) == 0:
                skipped.append(f"{p.name}: no face")
                continue
            r = max(rows, key=lambda r: r[2] * r[3])
            f = Face(bbox=r[:4], score=float(r[14]), landmarks=r[4:14].reshape(5, 2), raw=r)
            caps.append(_Capture(self.embed(img, f), face_photo(img, f), float(r[14])))
        if not caps:
            raise ValueError("no usable faces in those images")
        s = EnrollSession(name=name, role=role, target=len(caps), timeout_s=0,
                          person_id=existing["id"] if existing else None, captures=caps)
        self._finish(s)
        if s.state != "done":
            raise RuntimeError(s.error)
        return s.result_person_id, skipped
