# NETRAX v2

A standalone, fully local vision hub: a laptop + a camera, a web dashboard, no cloud.

> v2 is a rewrite. The v1 code lives in [`legacy/`](legacy/).

## Quick start (EndeavourOS / Arch)

```bash
sudo pacman -S --needed python-numpy python-yaml python-pydantic python-psutil python-fastapi uvicorn
pip install --break-system-packages "opencv-python>=5"
pip install --break-system-packages -e .

netrax doctor      # checks Python, OpenCV 5, models, camera
netrax hub         # http://localhost:8000
```

Work in progress: face detection (YuNet) + dashboard are in; recognition and the people DB are next.
