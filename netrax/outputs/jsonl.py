"""Append every event as one JSON line (data/events.jsonl), for scripts and other tools."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

MAX_BYTES = 20 * 1024 * 1024     # rotate to events.jsonl.1 past this size


class JsonlWriter:
    def __init__(self, settings):
        self.settings = settings
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self.settings.resolve(self.settings.cfg.outputs.jsonl.path)

    def __call__(self, event: dict) -> None:
        if not self.settings.cfg.outputs.jsonl.enabled:
            return
        line = json.dumps(event, separators=(",", ":"), default=str) + "\n"
        p = self.path
        with self._lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists() and p.stat().st_size > MAX_BYTES:
                os.replace(p, p.with_name(p.name + ".1"))
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(line)
