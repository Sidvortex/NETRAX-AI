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
import os
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


# bench ---------------------------------------------------------------------
TARGET_FPS = 12.0


def _grab_frames(settings, n: int) -> list:
    """Grab up to n frames from the configured source into memory."""
    from netrax.core.source import open_source

    src = open_source(settings.cfg.camera, realtime=False).start()
    frames, seq, t_end = [], 0, time.time() + 15
    try:
        while len(frames) < n and time.time() < t_end:
            f = src.read(seq, timeout=2.0)
            if f is None:
                if src.state in ("ended", "error"):
                    break
                continue
            seq = f.seq
            frames.append(f.image)
            if getattr(src, "kind", "") == "image":
                break
    finally:
        src.stop()
    if not frames:
        raise SystemExit(f"bench: no frames from source {settings.cfg.camera.source!r} "
                         f"({src.error or src.state}). Try --source 0 or --source some_video.mp4")
    return frames


def cmd_bench(args) -> int:
    import numpy as np

    from netrax.config import MODULE_NAMES
    from netrax.core.pipeline import Pipeline
    from netrax.detectors import REGISTRY

    if args.all:
        wanted = [m for m in MODULE_NAMES if m in REGISTRY]
    else:
        wanted = [m.strip() for m in args.modules.split(",") if m.strip()]
    bad = [m for m in wanted if m not in MODULE_NAMES]
    if bad:
        print(f"unknown module(s): {', '.join(bad)}; choose from {', '.join(MODULE_NAMES)}", file=sys.stderr)
        return 2
    mods = {m: {"enabled": m in wanted} for m in MODULE_NAMES}
    if args.every_frame:
        for m in wanted:
            mods[m]["every_n"] = 1
        mods.setdefault("face_recog", {})["recheck_every"] = 1
    settings = _settings(args, {"modules": mods, "outputs": {"mqtt": {"enabled": False}}})
    wanted = [m for m in wanted if m in REGISTRY]

    print(f"NETRAX bench | source {settings.cfg.camera.source!r} | modules: {', '.join(wanted)}")
    frames = _grab_frames(settings, args.frames)
    h, w = frames[0].shape[:2]
    print(f"captured {len(frames)} frame(s) at {w}x{h}; loading models…")
    pipe = Pipeline(settings)
    pipe.sync(wait=True)
    for name in wanted:
        slot = pipe.slots[name]
        if slot.state != "ready":
            print(f"  {name}: {slot.state} {slot.error or ''}", file=sys.stderr)
            return 1
    for f in frames[:5]:                                   # warm-up (first DNN runs are slow)
        pipe.process(f)
    for slot in pipe.slots.values():
        slot.stats.__init__()
    times, faces, i = [], 0, 0
    t_end = time.perf_counter() + args.seconds
    while time.perf_counter() < t_end:
        f = frames[i % len(frames)]
        t0 = time.perf_counter()
        st = pipe.process(f)
        pipe.draw(f.copy(), st)
        times.append(time.perf_counter() - t0)
        faces += len(st.get("face_detect"))
        i += 1
    t = np.array(times) * 1000
    fps = 1000.0 / t.mean()
    print()
    print(f"{'module':12} {'runs':>6} {'avg ms':>8} {'p95 ms':>8} {'max FPS':>8}")
    for name in wanted:
        s = pipe.slots[name].stats.snapshot()
        mfps = 1000.0 / s["ms"] if s["ms"] else float("inf")
        print(f"{name:12} {s['runs']:6d} {s['ms']:8.2f} {s['p95_ms']:8.2f} {mfps:8.1f}")
    print(f"{'pipeline':12} {i:6d} {t.mean():8.2f} {np.percentile(t, 95):8.2f} {fps:8.1f}   (all modules + overlay)")
    print()
    if "face_detect" in wanted and faces == 0:
        print("note: no faces were found in these frames"
              + (", so face_recog did no work" if "face_recog" in wanted else "") + ". Sit in front of the camera.")
    verdict = "PASS" if fps >= TARGET_FPS else "FAIL"
    print(f"{verdict}: {fps:.1f} FPS {'>=' if fps >= TARGET_FPS else '<'} {TARGET_FPS:.0f} FPS target "
          f"({os.cpu_count()} CPU threads, {_cpu_name()})")
    return 0 if fps >= TARGET_FPS else 1


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or "unknown CPU"


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

    sp = sub.add_parser("bench", help="measure FPS per module on this machine")
    source_arg(sp)
    sp.add_argument("--modules", default="face_detect,face_recog", help="comma list (default: face_detect,face_recog)")
    sp.add_argument("--all", action="store_true", help="bench every module")
    sp.add_argument("--every-frame", action="store_true", help="worst case: run every module (and recognition) on every frame")
    sp.add_argument("--frames", type=int, default=60, help="frames to capture first (default 60)")
    sp.add_argument("--seconds", type=float, default=10.0, help="how long to process (default 10)")
    sp.set_defaults(fn=cmd_bench)

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
