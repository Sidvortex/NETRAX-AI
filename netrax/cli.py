"""`netrax` command line.

    netrax hub                      camera + pipeline + web dashboard
    netrax run [--no-window]        pipeline only (OpenCV window, or headless)
    netrax enroll --name X --role owner
    netrax people                   list / rename / change role / delete
    netrax bench                    FPS per module on your CPU
    netrax doctor                   check deps, models, camera, DB
    netrax models [--download]      list or fetch the ONNX models
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from netrax import __version__

log = logging.getLogger("netrax")


def _settings(args, extra: dict | None = None):
    from netrax.config import Settings, deep_merge

    overrides: dict = {}
    if getattr(args, "source", None) is not None:
        overrides["camera"] = {"source": args.source}
    if getattr(args, "host", None):
        overrides.setdefault("hub", {})["host"] = args.host
    if getattr(args, "port", None):
        overrides.setdefault("hub", {})["port"] = args.port
    if extra:
        overrides = deep_merge(overrides, extra)
    return Settings(args.config, overrides=overrides)


# hub -----------------------------------------------------------------------
def cmd_hub(args) -> int:
    import socket

    import uvicorn

    from netrax.hub.app import create_app
    from netrax.runtime import Runtime

    settings = _settings(args)
    rt = Runtime(settings)
    app = create_app(rt)
    hub = settings.cfg.hub
    urls = [f"http://localhost:{hub.port}"]
    if hub.host in ("0.0.0.0", "::"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))     # no packet is sent; just picks the LAN interface
                urls.append(f"http://{s.getsockname()[0]}:{hub.port}  (phone/tablet on the same WiFi)")
        except OSError:
            pass
    print(f"NETRAX {__version__} hub")
    print(f"  config : {settings.path}{'' if settings.path.exists() else ' (not created yet, using defaults)'}")
    print(f"  camera : {settings.cfg.camera.source}")
    for u in urls:
        print(f"  open   : {u}")
    uvicorn.run(app, host=hub.host, port=hub.port, log_level="warning")
    return 0


# run -----------------------------------------------------------------------
def cmd_run(args) -> int:
    import cv2

    from netrax.runtime import Runtime

    settings = _settings(args)
    rt = Runtime(settings).start()
    eng = rt.engine
    window = not args.no_window
    t_end = time.time() + args.duration if args.duration else None
    last_print, last_seq = 0.0, 0
    try:
        while t_end is None or time.time() < t_end:
            if window:
                seq = eng.seq
                if seq != last_seq:
                    last_seq = seq
                    with eng._lock:
                        img = eng._annotated
                    if img is not None:
                        cv2.imshow("NETRAX (q to quit)", img)
                if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                    break
            else:
                time.sleep(0.05)
            if eng.source and eng.source.state in ("ended", "error"):
                if eng.source.state == "error":
                    print(f"source error: {eng.source.error}", file=sys.stderr)
                    return 1
                break
            now = time.time()
            if args.json and eng.state is not None and eng.state.frame_idx != getattr(cmd_run, "_last", -1):
                cmd_run._last = eng.state.frame_idx
                print(json.dumps(eng.state.to_dict()), flush=True)
            elif not args.json and now - last_print >= 1.0:
                last_print = now
                print(_summary_line(rt), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        rt.stop()
        if window:
            cv2.destroyAllWindows()
    return 0


def _summary_line(rt) -> str:
    st = rt.engine.state
    s = rt.engine.status()
    parts = [f"{s['fps']:5.1f} fps", f"cpu {s['cpu']:4.1f}%", f"cam {s['camera']['state']}"]
    if st:
        faces = st.faces()
        if faces:
            parts.append("faces: " + ", ".join(f.get("name") or f"#{f['track_id']}" for f in faces))
        g = [h.gesture for h in st.get("hands") if h.gesture != "none"]
        if g:
            parts.append("gesture: " + ",".join(g))
        objs = sorted({o.label for o in st.get("objects")})
        if objs:
            parts.append("objects: " + ",".join(objs))
    loading = [n for n, m in s["modules"].items() if m["state"] in ("loading", "error")]
    if loading:
        parts.append("(" + ", ".join(f"{n} {s['modules'][n]['state']}" for n in loading) + ")")
    return " | ".join(parts)


# models --------------------------------------------------------------------
def cmd_models(args) -> int:
    from netrax import models

    settings = _settings(args)
    mdir = settings.models_dir
    wanted = set(models.MODELS) if args.all else set()
    if args.download and not args.all:
        mods = settings.cfg.modules
        for name in type(mods).model_fields:
            if getattr(mods, name).enabled:
                wanted.update(models.keys_for_module(name, mods.objects.model))
    print(f"models dir: {mdir}")
    print(f"{'model':10} {'file':58} {'MB':>6}  {'license':10} status")
    rc = 0
    for key, spec in models.MODELS.items():
        path = models.model_path(mdir, key)
        if key in wanted and not models.is_valid(path, spec):
            try:
                models.ensure(mdir, key)
            except models.ModelError as e:
                print(f"  ! {e}", file=sys.stderr)
                rc = 1
        ok = models.is_valid(path, spec)
        print(f"{key:10} {spec.filename:58} {spec.size / 1e6:6.1f}  {spec.license:10} {'ok' if ok else 'missing'}")
    return rc


# parser --------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="netrax", description="NETRAX local vision hub")
    p.add_argument("--version", action="version", version=f"netrax {__version__}")
    p.add_argument("-c", "--config", help="config file (default: ./config.yaml or $NETRAX_CONFIG)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def source_arg(sp):
        sp.add_argument("-s", "--source", help="override camera: index, URL or file (not saved)")

    sp = sub.add_parser("hub", help="camera + dashboard at http://localhost:8000")
    source_arg(sp)
    sp.add_argument("--host")
    sp.add_argument("--port", type=int)
    sp.set_defaults(fn=cmd_hub)

    sp = sub.add_parser("run", help="run the pipeline without the dashboard")
    source_arg(sp)
    sp.add_argument("--no-window", action="store_true", help="headless (no OpenCV window)")
    sp.add_argument("--json", action="store_true", help="print one JSON line per processed frame")
    sp.add_argument("--duration", type=float, help="stop after N seconds")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("models", help="list models, or download them")
    sp.add_argument("--download", action="store_true", help="download models of enabled modules")
    sp.add_argument("--all", action="store_true", help="download every model (~108 MB)")
    sp.set_defaults(fn=cmd_models)
    return p


def main(argv: list[str] | None = None) -> int:
    from netrax.runtime import setup_logging

    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    try:
        return args.fn(args) or 0
    except KeyboardInterrupt:
        return 130
