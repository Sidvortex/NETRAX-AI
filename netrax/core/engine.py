"""The engine: camera -> pipeline -> (annotated frame, results, hooks).

One processing thread pulls the newest frame, runs the pipeline, draws the
overlay and hands the result to hooks (events, outputs, enrollment). The hub
and the CLI both drive NETRAX through this class.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import cv2
import numpy as np
import psutil

from netrax.config import Config, Settings
from netrax.core.pipeline import DetectorFactory, Pipeline, default_factory
from netrax.core.scheduler import RateCounter
from netrax.core.source import FrameSource, open_source
from netrax.core.types import FrameState

log = logging.getLogger("netrax.engine")

Hook = Callable[[np.ndarray, FrameState], None]


class Engine:
    def __init__(self, settings: Settings, source: FrameSource | None = None,
                 factory: DetectorFactory = default_factory, realtime: bool = True, draw: bool = True):
        self.settings = settings
        self.realtime = realtime
        self.draw_enabled = draw
        self.pipeline = Pipeline(settings, factory=factory)
        self._fixed_source = source is not None
        self.source: FrameSource | None = source
        self.hooks: list[Hook] = []
        self.started_at = time.time()
        self.fps = RateCounter()
        self.proc_ms = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: np.ndarray | None = None
        self._annotated: np.ndarray | None = None
        self.state: FrameState | None = None
        self._seq = 0
        self._jpeg_cache: dict[tuple, tuple[int, bytes]] = {}
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)
        psutil.cpu_percent(None)
        settings.on_change(self._on_config_change)

    # lifecycle -------------------------------------------------------------
    def start(self) -> "Engine":
        self.pipeline.sync()
        if self.source is None:
            self.source = self._make_source()
        self.source.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="netrax-engine", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self.source:
            self.source.stop()
        if self._thread:
            self._thread.join(timeout=5)

    def _make_source(self) -> FrameSource:
        try:
            return open_source(self.settings.cfg.camera, realtime=self.realtime)
        except FileNotFoundError as e:
            s = FrameSource()
            s.state, s.error = "error", str(e)
            return s

    def set_source(self, source: FrameSource | None = None) -> None:
        """Swap the video source (None = rebuild from config)."""
        old = self.source
        if old:
            old.stop()
        self.source = source or self._make_source()
        self._fixed_source = source is not None
        self.pipeline.reset()
        self.source.start()

    def _on_config_change(self, old: Config, new: Config) -> None:
        if old.camera != new.camera and not self._fixed_source:
            log.info("camera settings changed, reopening source")
            self.set_source()
        if old.modules.objects.model != new.modules.objects.model:
            self.pipeline.reload("objects")
        self.pipeline.sync()

    # processing ------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> FrameState:
        """Run one frame through pipeline + overlay + hooks (also used by bench/tests)."""
        t0 = time.perf_counter()
        state = self.pipeline.process(frame)
        annotated = self.pipeline.draw(frame.copy(), state) if self.draw_enabled else frame
        for hook in list(self.hooks):
            try:
                hook(frame, state)
            except Exception:  # noqa: BLE001
                log.exception("hook %r failed", hook)
        self.proc_ms = (time.perf_counter() - t0) * 1000
        with self._lock:
            self._frame, self._annotated, self.state = frame, annotated, state
            self._seq += 1
        self.fps.tick()
        return state

    def _loop(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            src = self.source
            f = src.read(last_seq, timeout=0.5) if src else None
            if f is None:
                if src and src.state == "ended":
                    log.info("video source ended")
                    self._stop.wait(0.5)
                continue
            last_seq = f.seq
            self.process_frame(f.image)

    # outputs for the hub ---------------------------------------------------
    @property
    def seq(self) -> int:
        return self._seq

    def jpeg(self, annotated: bool = True, quality: int | None = None) -> tuple[int, bytes | None]:
        """(seq, JPEG bytes) of the newest frame, encoded at most once per frame."""
        q = quality or self.settings.cfg.hub.jpeg_quality
        with self._lock:
            seq = self._seq
            img = self._annotated if annotated else self._frame
        if img is None:
            return seq, None
        key = (annotated, q)
        cached = self._jpeg_cache.get(key)
        if cached and cached[0] == seq:
            return cached
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = (seq, buf.tobytes() if ok else None)
        self._jpeg_cache[key] = out
        return out

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame

    def status(self) -> dict:
        cam = self.settings.cfg.camera
        src = self.source.status() if self.source else {"state": "stopped"}
        src["source"] = cam.source if not self._fixed_source else getattr(self.source, "kind", "custom")
        return {
            "fps": round(self.fps.rate, 1),
            "proc_ms": round(self.proc_ms, 1),
            "cpu": round(psutil.cpu_percent(None), 1),
            "proc_cpu": round(self._proc.cpu_percent(None) / psutil.cpu_count(), 1),
            "mem_mb": round(self._proc.memory_info().rss / 1e6, 1),
            "uptime_s": round(time.time() - self.started_at),
            "camera": src,
            "modules": self.pipeline.status(),
        }
