"""Configuration: one YAML file, validated with pydantic.

`config.default.yaml` (shipped with the package) holds every default.
The user's `config.yaml` only needs the keys they change; it is deep-merged
on top of the defaults. The Settings page writes the full merged config back.
"""

from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.default.yaml")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CameraCfg(_Strict):
    source: int | str = 0
    width: int = Field(640, ge=64, le=4096)
    height: int = Field(480, ge=64, le=4096)
    fps: int = Field(30, ge=1, le=120)
    loop_file: bool = True
    mirror: bool = False

    @field_validator("source", mode="before")
    @classmethod
    def _digits_to_index(cls, v: Any) -> Any:
        # "0" from a form field means webcam 0, not a file called "0"
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        return v


class HubCfg(_Strict):
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    stream_fps: int = Field(15, ge=1, le=60)
    jpeg_quality: int = Field(75, ge=10, le=100)


class PathsCfg(_Strict):
    data_dir: str = "data"
    models_dir: str = "models"


class ModuleCfg(_Strict):
    enabled: bool = False
    every_n: int = Field(1, ge=1, le=120)


class FaceDetectCfg(ModuleCfg):
    enabled: bool = True
    score_threshold: float = Field(0.75, ge=0.05, le=1.0)
    nms_threshold: float = Field(0.3, ge=0.05, le=1.0)
    max_side: int = Field(640, ge=160, le=4096)
    min_face: int = Field(40, ge=8, le=1024)


class FaceRecogCfg(ModuleCfg):
    enabled: bool = True
    threshold: float = Field(0.363, ge=0.0, le=1.0)
    recheck_every: int = Field(15, ge=1, le=600)
    votes: int = Field(5, ge=1, le=50)


class HeadPoseCfg(ModuleCfg):
    enabled: bool = True
    smoothing: float = Field(0.5, ge=0.0, le=0.95)
    look_threshold: float = Field(15, ge=1, le=60)


class HandsCfg(ModuleCfg):
    every_n: int = Field(2, ge=1, le=120)
    max_hands: int = Field(2, ge=1, le=4)
    palm_score: float = Field(0.6, ge=0.05, le=1.0)
    landmark_conf: float = Field(0.8, ge=0.05, le=1.0)
    stable_frames: int = Field(3, ge=1, le=30)


class PoseCfg(ModuleCfg):
    every_n: int = Field(2, ge=1, le=120)
    person_score: float = Field(0.5, ge=0.05, le=1.0)
    min_visibility: float = Field(0.5, ge=0.0, le=1.0)
    style: Literal["skeleton", "dance"] = "skeleton"
    slouch_ratio: float = Field(0.55, ge=0.1, le=2.0)
    tilt_deg: float = Field(12, ge=1, le=45)


class ObjectsCfg(ModuleCfg):
    model: Literal["nanodet", "yolox"] = "nanodet"
    every_n: int = Field(5, ge=1, le=120)
    score_threshold: float = Field(0.4, ge=0.05, le=1.0)
    nms_threshold: float = Field(0.5, ge=0.05, le=1.0)
    classes: list[str] = []


class EmotionCfg(ModuleCfg):
    every_n: int = Field(10, ge=1, le=120)


class ModulesCfg(_Strict):
    face_detect: FaceDetectCfg = FaceDetectCfg()
    face_recog: FaceRecogCfg = FaceRecogCfg()
    head_pose: HeadPoseCfg = HeadPoseCfg()
    hands: HandsCfg = HandsCfg()
    pose: PoseCfg = PoseCfg()
    objects: ObjectsCfg = ObjectsCfg()
    emotion: EmotionCfg = EmotionCfg()


class EventsCfg(_Strict):
    seen_cooldown_s: float = Field(120, ge=0)
    unknown_after: int = Field(3, ge=1, le=100)
    unknown_cooldown_s: float = Field(30, ge=0)
    gesture_cooldown_s: float = Field(3, ge=0)
    object_cooldown_s: float = Field(60, ge=0)
    object_ignore: list[str] = ["person"]
    posture_after_s: float = Field(20, ge=0)
    snapshot_seen: bool = False
    keep_days: int = Field(30, ge=0)


class JsonlCfg(_Strict):
    enabled: bool = True
    path: str = "data/events.jsonl"


class MqttCfg(_Strict):
    enabled: bool = False
    host: str = "192.168.1.39"
    port: int = Field(1883, ge=1, le=65535)
    client_id: str = "netrax-hub"
    head_track_hz: float = Field(12, gt=0, le=60)
    rate_hz: float = Field(5, gt=0, le=60)
    status_every_s: float = Field(5, gt=0)


class OutputsCfg(_Strict):
    jsonl: JsonlCfg = JsonlCfg()
    mqtt: MqttCfg = MqttCfg()


class Config(_Strict):
    camera: CameraCfg = CameraCfg()
    hub: HubCfg = HubCfg()
    paths: PathsCfg = PathsCfg()
    modules: ModulesCfg = ModulesCfg()
    events: EventsCfg = EventsCfg()
    outputs: OutputsCfg = OutputsCfg()


MODULE_NAMES = tuple(ModulesCfg.model_fields)


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def resolve_config_path(path: str | os.PathLike | None = None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    env = os.environ.get("NETRAX_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "config.yaml").resolve()


class Settings:
    """The live, thread-safe config plus where it is stored.

    `cfg` is replaced atomically on update, so readers can grab
    `settings.cfg` once per frame without locking.
    """

    def __init__(self, path: str | os.PathLike | None = None, overrides: dict | None = None):
        self.path = resolve_config_path(path)
        self.base_dir = self.path.parent
        self._lock = threading.Lock()
        self._listeners: list = []
        defaults = _read_yaml(DEFAULT_CONFIG_PATH)
        user = _read_yaml(self.path) if self.path.exists() else {}
        merged = deep_merge(defaults, user)
        if overrides:
            merged = deep_merge(merged, overrides)
        self.cfg: Config = Config.model_validate(merged)

    # paths -----------------------------------------------------------------
    def resolve(self, p: str | os.PathLike) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.cfg.paths.data_dir)

    @property
    def models_dir(self) -> Path:
        return self.resolve(self.cfg.paths.models_dir)

    # updates ---------------------------------------------------------------
    def on_change(self, fn) -> None:
        """Register fn(old_cfg, new_cfg), called after every successful update."""
        self._listeners.append(fn)

    def update(self, patch: dict, save: bool = True) -> Config:
        """Validate `patch` merged into the current config; raises pydantic.ValidationError."""
        with self._lock:
            old = self.cfg
            new = Config.model_validate(deep_merge(old.model_dump(), patch))
            self.cfg = new
            if save:
                self.save()
        for fn in list(self._listeners):
            fn(old, new)
        return new

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".yaml.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("# NETRAX config (written by NETRAX; comments are in netrax/config.default.yaml)\n")
            yaml.safe_dump(self.cfg.model_dump(), fh, sort_keys=False, allow_unicode=True)
        os.replace(tmp, self.path)
