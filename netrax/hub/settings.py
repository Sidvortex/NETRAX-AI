"""Settings API: read/update config.yaml, list cameras and models."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import ValidationError

from netrax import models
from netrax.config import field_choices, field_limits

router = APIRouter()
RESTART_KEYS = ("hub.host", "hub.port", "paths.data_dir", "paths.models_dir")


def list_cameras() -> list[dict]:
    """V4L2 devices (/dev/videoN) with their names, Linux only."""
    out = []
    for dev in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:] or 0)):
        try:
            name = (dev / "name").read_text().strip()
            idx = int(dev.name[5:])
            # each USB webcam usually exposes 2 nodes; index 0 of a device is the capture one
            node_index = int((dev / "index").read_text().strip()) if (dev / "index").exists() else 0
        except (OSError, ValueError):
            continue
        if node_index == 0:
            out.append({"index": idx, "name": name, "device": f"/dev/{dev.name}"})
    return out


@router.get("/api/config")
def get_config(request: Request):
    s = request.app.state.rt.settings
    return {
        "config": s.cfg.model_dump(),
        "path": str(s.path),
        "saved": s.path.exists(),
        "choices": field_choices(),
        "limits": field_limits(),
        "restart_keys": RESTART_KEYS,
    }


@router.put("/api/config")
def put_config(request: Request, patch: dict = Body(...)):
    s = request.app.state.rt.settings
    try:
        cfg = s.update(patch)
    except ValidationError as e:
        raise HTTPException(422, [{"loc": ["body", *map(str, err["loc"])], "msg": err["msg"]}
                                  for err in e.errors()]) from None
    return {"config": cfg.model_dump(), "path": str(s.path), "saved": True}


@router.get("/api/cameras")
def cameras():
    return list_cameras()


@router.get("/api/models")
def list_models(request: Request):
    mdir = request.app.state.rt.settings.models_dir
    return [
        {"key": k, "file": m.filename, "module": m.module, "license": m.license, "size_mb": round(m.size / 1e6, 1),
         "description": m.description, "source": m.source_page,
         "present": models.is_valid(models.model_path(mdir, k), m)}
        for k, m in models.MODELS.items()
    ]
