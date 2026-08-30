"""Wires everything together: settings, engine and (later) DB, events and outputs."""

from __future__ import annotations

import logging

from netrax.config import Settings
from netrax.core.engine import Engine
from netrax.core.enroll import Enroller
from netrax.core.events import EventBus
from netrax.core.pipeline import DetectorFactory
from netrax.core.source import FrameSource
from netrax.db.store import Store
from netrax.detectors import load_class
from netrax.outputs.jsonl import JsonlWriter
from netrax.outputs.mqtt import MqttPublisher

log = logging.getLogger("netrax")


class Runtime:
    def __init__(self, settings: Settings, source: FrameSource | None = None,
                 factory: DetectorFactory | None = None, realtime: bool = True):
        self.settings = settings
        self.store = Store(settings.data_dir)
        self.services = {"store": self.store}
        if factory is None:
            def factory(name: str, s: Settings):
                return load_class(name)(s, self.services)
        self.engine = Engine(settings, source=source, factory=factory, realtime=realtime)
        self.enroller = Enroller(settings, self.store)
        self.events = EventBus(settings, self.store)
        self.engine.hooks += [self.enroller, self.events]
        self.events.on_event(JsonlWriter(settings))
        self.mqtt = MqttPublisher(settings, status_fn=self.engine.status)
        self.engine.hooks.append(self.mqtt)
        self.events.on_event(self.mqtt.on_event)

    def start(self) -> "Runtime":
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.engine.start()
        self.mqtt.apply()
        return self

    def stop(self) -> None:
        self.mqtt.stop()
        self.engine.stop()
        self.store.close()

    def status(self) -> dict:
        s = self.engine.status()
        s["mqtt"] = self.mqtt.status()
        return s


def setup_logging(level: str = "INFO") -> None:
    try:  # OpenCV 5 prints a WARN per net about targets on the new engine; keep the console clean
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
