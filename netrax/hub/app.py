"""FastAPI hub: one process serves the camera pipeline, the API and the dashboard."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from netrax import __version__
from netrax.config import MODULE_NAMES
from netrax.hub import people
from netrax.runtime import Runtime

log = logging.getLogger("netrax.hub")
STATIC = Path(__file__).with_name("static")
PAGES = {"/": "index.html", "/people": "people.html"}


class Toggle(BaseModel):
    enabled: bool | None = None
    overlay: bool | None = None


def create_app(rt: Runtime, start_engine: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_engine:
            rt.start()
        try:
            yield
        finally:
            if start_engine:
                rt.stop()

    app = FastAPI(title="NETRAX hub", version=__version__, lifespan=lifespan)
    app.state.rt = rt

    for route, page in PAGES.items():
        app.add_api_route(route, _page(page), include_in_schema=False)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.include_router(people.router)

    @app.get("/stream.mjpg")
    async def stream(request: Request, overlay: int = 1):
        boundary = "netraxframe"

        async def gen():
            last = -1
            while not await request.is_disconnected():
                fps = rt.settings.cfg.hub.stream_fps
                seq, jpg = rt.engine.jpeg(annotated=bool(overlay))
                if jpg is not None and seq != last:
                    last = seq
                    yield (f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                           f"Content-Length: {len(jpg)}\r\n\r\n").encode() + jpg + b"\r\n"
                await asyncio.sleep(1.0 / fps)

        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/snapshot.jpg")
    def snapshot(overlay: int = 1):
        _, jpg = rt.engine.jpeg(annotated=bool(overlay))
        if jpg is None:
            raise HTTPException(503, "no frame yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status():
        return rt.status()

    @app.get("/api/state")
    def state():
        st = rt.engine.state
        return st.to_dict() if st else {}

    @app.post("/api/modules/{name}")
    def toggle_module(name: str, body: Toggle):
        if name not in MODULE_NAMES:
            raise HTTPException(404, f"unknown module {name}")
        if body.enabled is not None:
            rt.settings.update({"modules": {name: {"enabled": body.enabled}}})
        if body.overlay is not None:
            rt.engine.pipeline.overlay[name] = body.overlay
        return {"name": name, "enabled": rt.engine.pipeline.enabled(name),
                "overlay": rt.engine.pipeline.overlay.get(name, True)}

    @app.exception_handler(ValueError)
    async def value_error(_, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    return app


def _page(name: str):
    async def handler():
        return FileResponse(STATIC / name, headers={"Cache-Control": "no-cache"})

    return handler
