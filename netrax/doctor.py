"""`netrax doctor`: checks dependencies, models, camera, data dir and ports."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import socket
import sys
import time

OK, WARN, FAIL = "ok", "warn", "fail"
MARK = {OK: "\033[32m✓\033[0m", WARN: "\033[33m!\033[0m", FAIL: "\033[31m✗\033[0m"}
PLAIN = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL"}

PACKAGES = [
    # import name, dist name, pacman package (None = use pip)
    ("cv2", "opencv-python", None),
    ("numpy", "numpy", "python-numpy"),
    ("yaml", "PyYAML", "python-yaml"),
    ("pydantic", "pydantic", "python-pydantic"),
    ("fastapi", "fastapi", "python-fastapi"),
    ("uvicorn", "uvicorn", "uvicorn"),
    ("psutil", "psutil", "python-psutil"),
    ("paho.mqtt", "paho-mqtt", "python-paho-mqtt"),
]


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []
        self.color = sys.stdout.isatty()

    def add(self, level: str, what: str, detail: str = "") -> None:
        self.rows.append((level, what, detail))
        mark = MARK[level] if self.color else PLAIN[level]
        print(f" {mark} {what}{(': ' + detail) if detail else ''}", flush=True)

    @property
    def failed(self) -> bool:
        return any(r[0] == FAIL for r in self.rows)


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def check_python(r: Report) -> None:
    v = sys.version_info
    lvl = OK if v >= (3, 11) else FAIL
    r.add(lvl, f"Python {v.major}.{v.minor}.{v.micro}", sys.executable)


def check_packages(r: Report) -> None:
    for mod, dist, pac in PACKAGES:
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            hint = f"sudo pacman -S {pac}" if pac else f"pip install --break-system-packages {dist}"
            r.add(FAIL, f"{dist} missing", f"{e.__class__.__name__}; install with: {hint}")
            continue
        ver = getattr(m, "__version__", None) or _version(dist) or "?"
        r.add(OK, f"{dist} {ver}")
    try:
        import cv2
    except ImportError:
        return
    major = int(cv2.__version__.split(".")[0])
    if major < 5:
        r.add(FAIL, f"OpenCV {cv2.__version__} is too old",
              "NETRAX needs OpenCV 5 (YuNet 2026may + new DNN engine). "
              "If pacman's python-opencv is installed, it may shadow pip's: "
              "pip install --break-system-packages -U 'opencv-python>=5'")
    for api in ("FaceDetectorYN", "FaceRecognizerSF"):
        if not hasattr(cv2, api):
            r.add(FAIL, f"cv2.{api} missing", "install opencv-python (not a minimal build)")
    if hasattr(cv2, "CascadeClassifier"):
        r.add(WARN, "this OpenCV still has Haar cascades", "fine, NETRAX does not use them")
    dists = [d for d in ("opencv-python", "opencv-python-headless", "opencv-contrib-python") if _version(d)]
    if len(dists) > 1:
        r.add(WARN, "several OpenCV wheels installed", ", ".join(dists) + " (keep only opencv-python)")
    gui = next((ln.split(":", 1)[1].strip() for ln in cv2.getBuildInformation().splitlines()
                if ln.strip().startswith("GUI:")), "?")
    if gui.upper() == "NONE":
        r.add(WARN, "OpenCV has no GUI backend (headless build)",
              "`netrax run` can't open a window; use --no-window or the hub, "
              "or install opencv-python instead of opencv-python-headless")


def check_config(r: Report, args):
    from netrax.cli import _settings

    try:
        s = _settings(args)
    except Exception as e:  # noqa: BLE001
        r.add(FAIL, "config invalid", str(e).splitlines()[0])
        return None
    r.add(OK, "config", f"{s.path}{'' if s.path.exists() else ' (not created yet, defaults in use)'}")
    on = [n for n in type(s.cfg.modules).model_fields if getattr(s.cfg.modules, n).enabled]
    r.add(OK, "enabled modules", ", ".join(on) or "none")
    return s


def check_models(r: Report, s, deep: bool) -> None:
    from netrax import models

    mods = s.cfg.modules
    keys: list[str] = []
    for name in type(mods).model_fields:
        if getattr(mods, name).enabled:
            keys += [k for k in models.keys_for_module(name, mods.objects.model) if k not in keys]
    if not keys:
        return
    missing = []
    for k in keys:
        spec = models.MODELS[k]
        p = models.model_path(s.models_dir, k)
        if models.is_valid(p, spec, check_hash=deep):
            r.add(OK, f"model {spec.filename}", f"{spec.size / 1e6:.1f} MB, {spec.license}")
        else:
            missing.append(spec.filename)
            why = "missing" if not p.exists() else ("wrong size" if p.stat().st_size != spec.size else "bad hash")
            r.add(WARN, f"model {spec.filename} {why}", "downloads on first use, or run: netrax models --download")


def check_dirs(r: Report, s) -> None:
    for label, d in (("data dir", s.data_dir), ("models dir", s.models_dir)):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.write_text("x")
            probe.unlink()
            r.add(OK, label, str(d))
        except OSError as e:
            r.add(FAIL, f"{label} not writable", f"{d}: {e}")


def check_camera(r: Report, s, timeout: float = 6.0) -> None:
    from netrax.core.source import open_source

    cam = s.cfg.camera
    try:
        src = open_source(cam, realtime=False).start()
    except Exception as e:  # noqa: BLE001
        r.add(FAIL, f"camera {cam.source!r}", str(e))
        return
    t0 = time.time()
    f = None
    while time.time() - t0 < timeout:
        f = src.read(0, timeout=0.5)
        if f is not None or src.state == "error":
            break
    src.stop()
    if f is None:
        hint = ""
        if isinstance(cam.source, int):
            hint = (f" (is it /dev/video{cam.source}? check `ls /dev/video*` and that you are in the "
                    "'video' group: sudo usermod -aG video $USER, then log out/in)")
        r.add(FAIL, f"camera {cam.source!r} gave no frame", (src.error or "timeout") + hint)
    else:
        h, w = f.image.shape[:2]
        r.add(OK, f"camera {cam.source!r}", f"{w}x{h}")


def check_port(r: Report, s) -> None:
    hub = s.cfg.hub
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0" if hub.host in ("0.0.0.0", "::") else hub.host, hub.port))
            r.add(OK, f"port {hub.port} free")
        except OSError:
            r.add(WARN, f"port {hub.port} in use", "a hub may already be running, or use --port")


def run(args) -> int:
    r = Report()
    print("NETRAX doctor\n")
    check_python(r)
    check_packages(r)
    s = check_config(r, args)
    if s is not None:
        check_dirs(r, s)
        check_models(r, s, deep=args.deep)
        if not args.no_camera:
            check_camera(r, s)
        check_port(r, s)
        for extra in EXTRA_CHECKS:
            extra(r, s)
    print()
    fails = sum(1 for x in r.rows if x[0] == FAIL)
    warns = sum(1 for x in r.rows if x[0] == WARN)
    print("all good" if not fails and not warns else f"{fails} problem(s), {warns} warning(s)")
    return 1 if r.failed else 0


# more checks (DB, MQTT) register here
EXTRA_CHECKS: list = []

if os.environ.get("NO_COLOR"):
    MARK = PLAIN
