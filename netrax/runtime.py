"""Wires everything together: settings, engine and (later) DB, events and outputs."""

from __future__ import annotations

import logging

from netrax.config import Settings
from netrax.core.engine import Engine
from netrax.core.pipeline import DetectorFactory, default_factory
from netrax.core.source import FrameSource

log = logging.getLogger("netrax")


class Runtime:
    def __init__(self, settings: Settings, source: FrameSource | None = None,
                 factory: DetectorFactory = default_factory, realtime: bool = True):
        self.settings = settings
        self.engine = Engine(settings, source=source, factory=factory, realtime=realtime)

    def start(self) -> "Runtime":
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.engine.start()
        return self

    def stop(self) -> None:
        self.engine.stop()

    def status(self) -> dict:
        return self.engine.status()


def setup_logging(level: str = "INFO") -> None:
    try:  # OpenCV 5 prints a WARN per net about targets on the new engine; keep the console clean
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
