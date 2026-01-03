"""The processing pipeline: runs enabled detector plugins on a frame, in order.

Modules load in the background (the first run may download models), so the
hub and camera come up immediately and each module turns "ready" on its own.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from netrax.config import Settings
from netrax.core.scheduler import ModuleStats, Scheduler
from netrax.core.types import FrameState, Result
from netrax.detectors import DRAW_ORDER, REGISTRY, load_class
from netrax.detectors.base import Detector

log = logging.getLogger("netrax.pipeline")

DetectorFactory = Callable[[str, Settings], Detector]


def default_factory(name: str, settings: Settings) -> Detector:
    return load_class(name)(settings)


class ModuleSlot:
    def __init__(self, name: str):
        self.name = name
        self.detector: Detector | None = None
        self.state = "off"           # off | loading | ready | error
        self.error: str | None = None
        self.stats = ModuleStats()
        self.last: list[Result] = []
        self.last_frame = -1
        self._gen = 0                # bumps on every (re)load, so stale loader threads are ignored

    def status(self) -> dict:
        d = {"state": self.state, "error": self.error}
        d.update(self.stats.snapshot())
        return d


class Pipeline:
    def __init__(self, settings: Settings, factory: DetectorFactory = default_factory,
                 order: list[str] | None = None):
        self.settings = settings
        self.factory = factory
        self.order = list(order or REGISTRY)
        self.slots = {name: ModuleSlot(name) for name in self.order}
        self.scheduler = Scheduler(self.order)
        self.overlay: dict[str, bool] = {name: True for name in self.order}
        self._lock = threading.RLock()
        self.frame_idx = 0

    # module lifecycle ------------------------------------------------------
    def enabled(self, name: str) -> bool:
        cfg = getattr(self.settings.cfg.modules, name, None)
        return bool(cfg and cfg.enabled)

    def sync(self, wait: bool = False) -> None:
        """Load newly enabled modules and drop disabled ones."""
        threads = []
        for name, slot in self.slots.items():
            if self.enabled(name):
                if slot.state in ("off", "error"):
                    threads.append(self._start_load(slot))
            elif slot.state != "off":
                with self._lock:
                    slot._gen += 1
                    slot.detector, slot.state, slot.error, slot.last = None, "off", None, []
        if wait:
            for t in threads:
                t.join()

    def reload(self, name: str) -> None:
        """Force a module to reload (e.g. a different model was picked)."""
        slot = self.slots[name]
        with self._lock:
            slot._gen += 1
            slot.detector, slot.state, slot.last = None, "off", []
        self.sync()

    def _start_load(self, slot: ModuleSlot) -> threading.Thread:
        with self._lock:
            slot._gen += 1
            gen = slot._gen
            slot.state, slot.error = "loading", None

        def run():
            t0 = time.perf_counter()
            try:
                det = self.factory(slot.name, self.settings)
                det.load()
            except Exception as e:  # noqa: BLE001 - surface any load error in the dashboard
                log.error("module %s failed to load: %s", slot.name, e)
                with self._lock:
                    if slot._gen == gen:
                        slot.state, slot.error = "error", str(e)
                return
            with self._lock:
                if slot._gen == gen:
                    slot.detector, slot.state = det, "ready"
            log.info("module %s ready (%.1fs)", slot.name, time.perf_counter() - t0)

        t = threading.Thread(target=run, name=f"load-{slot.name}", daemon=True)
        t.start()
        return t

    def reset(self) -> None:
        with self._lock:
            for slot in self.slots.values():
                slot.last, slot.last_frame = [], -1
                if slot.detector:
                    slot.detector.reset()

    def detector(self, name: str) -> Detector | None:
        slot = self.slots.get(name)
        return slot.detector if slot and slot.state == "ready" else None

    # per frame -------------------------------------------------------------
    def process(self, frame: np.ndarray) -> FrameState:
        idx = self.frame_idx
        self.frame_idx += 1
        state = FrameState(frame_idx=idx, width=frame.shape[1], height=frame.shape[0])
        mods = self.settings.cfg.modules
        for name in self.order:
            slot = self.slots[name]
            det = slot.detector
            if slot.state != "ready" or det is None:
                continue
            if any(self.slots[r].state != "ready" for r in det.requires if r in self.slots):
                continue
            every_n = getattr(getattr(mods, name), "every_n", 1)
            if self.scheduler.should_run(name, every_n, idx) or slot.last_frame < 0:
                t0 = time.perf_counter()
                try:
                    res = det.process(frame, state)
                except Exception as e:  # noqa: BLE001
                    slot.stats.errors += 1
                    slot.error = f"{type(e).__name__}: {e}"
                    if slot.stats.errors in (1, 10) or slot.stats.errors % 500 == 0:
                        log.exception("module %s failed on frame %d", name, idx)
                    res = []
                else:
                    ms = (time.perf_counter() - t0) * 1000
                    slot.stats.record(ms)
                    state.timings_ms[name] = round(ms, 2)
                    state.ran.add(name)
                slot.last, slot.last_frame = res, idx
            state.results[name] = slot.last
        return state

    def draw(self, frame: np.ndarray, state: FrameState) -> np.ndarray:
        for name in DRAW_ORDER:
            det = self.detector(name)
            if det is None or not self.overlay.get(name, True):
                continue
            res = state.get(name)
            if res:
                try:
                    det.draw(frame, res)
                except Exception:  # noqa: BLE001
                    log.exception("draw failed for %s", name)
        return frame

    def status(self) -> dict:
        return {name: slot.status() for name, slot in self.slots.items()}
