"""People API: list, rename, set role, delete, photos, enrollment."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter()


def _rt(request: Request):
    return request.app.state.rt


def _media_url(rel: str | None) -> str | None:
    return f"/media/{rel}" if rel else None


@router.get("/api/people")
def list_people(request: Request):
    out = []
    for p in _rt(request).store.list_people():
        p["cover_url"] = _media_url(p.pop("cover"))
        out.append(p)
    return out


class PersonPatch(BaseModel):
    name: str | None = None
    role: str | None = None


@router.patch("/api/people/{pid}")
def update_person(pid: int, body: PersonPatch, request: Request):
    try:
        return _rt(request).store.update_person(pid, name=body.name, role=body.role)
    except KeyError:
        raise HTTPException(404, "no such person") from None


@router.delete("/api/people/{pid}")
def delete_person(pid: int, request: Request):
    if not _rt(request).store.delete_person(pid):
        raise HTTPException(404, "no such person")
    return {"deleted": pid}


@router.get("/api/people/{pid}/photos")
def photos(pid: int, request: Request):
    store = _rt(request).store
    if store.get_person(pid) is None:
        raise HTTPException(404, "no such person")
    return [dict(p, url=_media_url(p["photo_path"])) for p in store.list_photos(pid)]


@router.delete("/api/photos/{emb_id}")
def delete_photo(emb_id: int, request: Request):
    if not _rt(request).store.delete_embedding(emb_id):
        raise HTTPException(404, "no such photo")
    return {"deleted": emb_id}


class EnrollReq(BaseModel):
    name: str
    role: str = "allowed"
    frames: int = Field(10, ge=3, le=30)
    timeout_s: float = Field(60, ge=5, le=300)
    add: bool = False


@router.post("/api/enroll")
def enroll_start(body: EnrollReq, request: Request):
    rt = _rt(request)
    if not rt.engine.pipeline.enabled("face_detect"):
        raise HTTPException(409, "enable Face detect first")
    sess = rt.enroller.start(body.name, body.role, target=body.frames, timeout_s=body.timeout_s,
                             add_to_existing=body.add)
    return sess.status()


@router.get("/api/enroll")
def enroll_status(request: Request):
    return _rt(request).enroller.status()


@router.delete("/api/enroll")
def enroll_cancel(request: Request):
    rt = _rt(request)
    rt.enroller.cancel()
    return rt.enroller.status()


@router.get("/media/{path:path}", include_in_schema=False)
def media(path: str, request: Request):
    """Serve photos/snapshots from the data dir (and nothing else)."""
    root = Path(_rt(request).store.data_dir).resolve()
    f = (root / path).resolve()
    if root not in f.parents or f.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "private, max-age=3600"})
