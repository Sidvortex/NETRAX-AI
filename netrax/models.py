"""Model registry and downloader.

All models come from OpenCV Zoo (https://github.com/opencv/opencv_zoo), pinned
to one commit. They are stored with Git LFS, so we must use
media.githubusercontent.com: `github.com/.../raw/...` returns a ~130 byte LFS
pointer file instead of the model. Every download is checked for exact size
and SHA-256 before it is moved into place.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ZOO_COMMIT = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
ZOO_MEDIA = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{ZOO_COMMIT}/models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    zoo_dir: str
    size: int
    sha256: str
    license: str
    module: str
    description: str

    @property
    def url(self) -> str:
        return f"{ZOO_MEDIA}/{self.zoo_dir}/{self.filename}"

    @property
    def source_page(self) -> str:
        return f"https://github.com/opencv/opencv_zoo/tree/main/models/{self.zoo_dir}"


MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec("yunet", "face_detection_yunet_2026may.onnx", "face_detection_yunet", 229_738,
                  "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
                  "MIT", "face_detect", "YuNet face detector (dynamic input, OpenCV 5)"),
        ModelSpec("sface", "face_recognition_sface_2021dec.onnx", "face_recognition_sface", 38_696_353,
                  "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
                  "Apache-2.0", "face_recog", "SFace 128-d face embeddings"),
        ModelSpec("palm", "palm_detection_mediapipe_2023feb.onnx", "palm_detection_mediapipe", 3_905_734,
                  "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
                  "Apache-2.0", "hands", "MediaPipe palm detector"),
        ModelSpec("handpose", "handpose_estimation_mediapipe_2023feb.onnx", "handpose_estimation_mediapipe", 4_099_621,
                  "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
                  "Apache-2.0", "hands", "MediaPipe hand landmarks (21 points)"),
        ModelSpec("persondet", "person_detection_mediapipe_2023mar.onnx", "person_detection_mediapipe", 11_990_159,
                  "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f",
                  "Apache-2.0", "pose", "MediaPipe person detector"),
        ModelSpec("pose", "pose_estimation_mediapipe_2023mar.onnx", "pose_estimation_mediapipe", 5_557_238,
                  "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f",
                  "Apache-2.0", "pose", "BlazePose full-body landmarks (33 points)"),
        ModelSpec("nanodet", "object_detection_nanodet_2022nov.onnx", "object_detection_nanodet", 3_800_954,
                  "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186",
                  "Apache-2.0", "objects", "NanoDet-Plus 416, COCO 80 classes"),
        ModelSpec("yolox", "object_detection_yolox_2022nov.onnx", "object_detection_yolox", 35_858_002,
                  "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
                  "Apache-2.0", "objects", "YOLOX-S 640, COCO 80 classes (slower, more accurate)"),
        ModelSpec("fer", "facial_expression_recognition_mobilefacenet_2022july.onnx", "facial_expression_recognition", 4_791_892,
                  "4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9",
                  "Apache-2.0", "emotion", "MobileFaceNet facial expression (7 classes)"),
    ]
}


class ModelError(RuntimeError):
    pass


LFS_POINTER_PREFIX = b"version https://git-lfs"


def model_path(models_dir: Path, key: str) -> Path:
    return Path(models_dir) / MODELS[key].filename


def is_valid(path: Path, spec: ModelSpec, check_hash: bool = False) -> bool:
    if not path.is_file() or path.stat().st_size != spec.size:
        return False
    return not check_hash or sha256_of(path) == spec.sha256


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress(done: int, total: int, name: str) -> None:
    if not sys.stderr.isatty():
        return
    pct = 100.0 * done / total if total else 0.0
    bar = "#" * int(pct // 4)
    sys.stderr.write(f"\r  {name[:44]:44} [{bar:25}] {pct:5.1f}% ")
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def download(spec: ModelSpec, dest: Path, url: str | None = None, timeout: float = 60.0) -> Path:
    """Download `spec` to `dest` (atomic), verifying size and hash."""
    url = url or spec.url
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    done = 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "netrax/2"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            head = b""
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                if len(head) < 64:
                    head += chunk[:64]
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                _progress(done, spec.size, spec.filename)
        if head.startswith(LFS_POINTER_PREFIX):
            raise ModelError(
                f"{spec.filename}: got a Git LFS pointer file, not the model. "
                "Use a media.githubusercontent.com/media/... URL, not github.com/.../raw/..."
            )
        if done != spec.size:
            raise ModelError(f"{spec.filename}: size mismatch ({done} bytes, expected {spec.size})")
        if h.hexdigest() != spec.sha256:
            raise ModelError(f"{spec.filename}: SHA-256 mismatch, download is corrupted")
        os.replace(tmp, dest)
        return dest
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def ensure(models_dir: Path, key: str, allow_download: bool = True) -> Path:
    """Return the local path of model `key`, downloading it first if needed."""
    spec = MODELS[key]
    path = model_path(models_dir, key)
    if is_valid(path, spec):
        return path
    if path.exists():
        print(f"[models] {spec.filename} has the wrong size, re-downloading", file=sys.stderr)
    if not allow_download:
        raise ModelError(f"model {spec.filename} missing in {models_dir}; run `netrax models --download`")
    print(f"[models] downloading {spec.filename} ({spec.size / 1e6:.1f} MB, {spec.license})", file=sys.stderr)
    try:
        return download(spec, path)
    except (OSError, urllib.error.URLError) as e:
        raise ModelError(f"could not download {spec.filename}: {e}") from e


def keys_for_module(module: str, objects_model: str = "nanodet") -> list[str]:
    keys = [k for k, m in MODELS.items() if m.module == module]
    if module == "objects":
        keys = [objects_model]
    return keys
