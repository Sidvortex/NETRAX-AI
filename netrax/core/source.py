"""Frame sources: webcam index, MJPEG/RTSP URL, or video file.

A capture thread reads continuously and keeps only the newest frame, so a
slow pipeline never builds up latency (it just skips frames).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("netrax.source")


@dataclass
class Frame:
    image: np.ndarray
    seq: int
    ts: float


def describe_source(source: int | str) -> str:
    if isinstance(source, int):
        return "webcam"
    s = str(source)
    if "://" in s:
        return "stream"
    return "file"


class FrameSource:
    """Base class: start(), stop(), read(), plus a small status dict."""

    def __init__(self):
        self.state = "stopped"     # stopped | connecting | live | reconnecting | ended | error
        self.error: str | None = None
        self.fps = 0.0
        self.size: tuple[int, int] = (0, 0)
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._seq = 0

    def _publish(self, image: np.ndarray) -> None:
        with self._cond:
            self._seq += 1
            self._latest = Frame(image, self._seq, time.time())
            self.size = (image.shape[1], image.shape[0])
            self._cond.notify_all()

    def read(self, after_seq: int = 0, timeout: float = 1.0) -> Frame | None:
        """Block until a frame newer than `after_seq` exists; None on timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._latest is None or self._latest.seq <= after_seq:
                left = deadline - time.monotonic()
                if left <= 0 or self.state in ("ended", "stopped"):
                    return None
                self._cond.wait(left)
            return self._latest

    def latest(self) -> Frame | None:
        return self._latest

    def start(self) -> "FrameSource":
        return self

    def stop(self) -> None:
        self.state = "stopped"
        with self._cond:
            self._cond.notify_all()

    def status(self) -> dict:
        return {"state": self.state, "error": self.error, "fps": round(self.fps, 1), "size": list(self.size)}


class CaptureSource(FrameSource):
    def __init__(self, source: int | str, width: int = 640, height: int = 480, fps: int = 30,
                 loop_file: bool = True, mirror: bool = False, realtime: bool = True):
        super().__init__()
        self.source = source
        self.kind = describe_source(source)
        self.req_size = (width, height)
        self.req_fps = fps
        self.loop_file = loop_file
        self.mirror = mirror
        self.realtime = realtime     # pace video files to their own FPS (False = as fast as possible)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_cfg(cls, cam, realtime: bool = True) -> "CaptureSource":
        return cls(cam.source, cam.width, cam.height, cam.fps, cam.loop_file, cam.mirror, realtime)

    def _open(self) -> cv2.VideoCapture:
        if self.kind == "webcam":
            cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_size[1])
                cap.set(cv2.CAP_PROP_FPS, self.req_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        if self.kind == "file" and not Path(str(self.source)).expanduser().exists():
            raise FileNotFoundError(f"video file not found: {self.source}")
        src = str(Path(str(self.source)).expanduser()) if self.kind == "file" else str(self.source)
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
        if self.kind == "stream":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self) -> "CaptureSource":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self.state = "connecting"
        self._thread = threading.Thread(target=self._run, name="netrax-capture", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        super().stop()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                cap = self._open()
            except FileNotFoundError as e:
                self.state, self.error = "error", str(e)
                log.error("%s", e)
                return
            if not cap.isOpened():
                self.state = "reconnecting" if self.kind != "file" else "error"
                self.error = f"cannot open {self.kind} {self.source!r}"
                log.warning("%s (retry in %.1fs)", self.error, backoff)
                if self.kind == "file":
                    return
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 10)
                continue
            backoff = 0.5
            self.error = None
            self._read_loop(cap)
            cap.release()
            if self.state == "ended":
                return
            if not self._stop.is_set():
                self.state = "reconnecting"
                self._stop.wait(0.5)

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        file_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not 1 <= file_fps <= 240:
            file_fps = 25.0
        period = 1.0 / file_fps
        next_t = time.monotonic()
        t0, n = time.monotonic(), 0
        fails = 0
        while not self._stop.is_set():
            ok, img = cap.read()
            if not ok or img is None:
                if self.kind == "file":
                    if self.loop_file and n > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.state = "ended"
                    with self._cond:
                        self._cond.notify_all()
                    return
                fails += 1
                if fails > 30:
                    log.warning("camera stopped delivering frames, reconnecting")
                    return
                time.sleep(0.01)
                continue
            fails = 0
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if self.mirror:
                img = cv2.flip(img, 1)
            self.state = "live"
            self._publish(img)
            n += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                self.fps = n / (now - t0)
                t0, n = now, 0
            if self.kind == "file" and self.realtime:
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_t = time.monotonic()


class ArraySource(FrameSource):
    """Frames from memory (tests, benchmarks, single images)."""

    def __init__(self, frames: list[np.ndarray], loop: bool = False, fps: float = 0.0):
        super().__init__()
        self.frames = frames
        self.loop = loop
        self.period = 1.0 / fps if fps else 0.0
        self._i = 0
        self.kind = "array"

    def start(self) -> "ArraySource":
        self.state = "live"
        return self

    def read(self, after_seq: int = 0, timeout: float = 1.0) -> Frame | None:
        if self.state != "live":
            return None
        if self._i >= len(self.frames):
            if not self.loop or not self.frames:
                self.state = "ended"
                return None
            self._i = 0
        if self.period:
            time.sleep(self.period)
        img = self.frames[self._i]
        self._i += 1
        self._publish(img)
        return self._latest


def open_source(cam, realtime: bool = True) -> FrameSource:
    """Build a source from CameraCfg. An image file becomes a looping ArraySource."""
    src = cam.source
    if isinstance(src, str) and Path(src).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        img = cv2.imread(str(Path(src).expanduser()))
        if img is None:
            raise FileNotFoundError(f"cannot read image {src}")
        if cam.mirror:
            img = cv2.flip(img, 1)
        s = ArraySource([img], loop=True, fps=cam.fps if realtime else 0)
        s.kind = "image"
        return s
    return CaptureSource.from_cfg(cam, realtime=realtime)
