import hashlib

import pytest

from netrax import models
from netrax.models import MODELS, ModelError, ModelSpec, download, is_valid


def _spec(data: bytes) -> ModelSpec:
    return ModelSpec("t", "t.onnx", "x", len(data), hashlib.sha256(data).hexdigest(), "MIT", "face_detect", "test")


def test_registry_urls_use_lfs_media_host():
    for spec in MODELS.values():
        assert spec.url.startswith("https://media.githubusercontent.com/media/opencv/opencv_zoo/")
        assert spec.size > 100_000
        assert len(spec.sha256) == 64
        assert spec.license in {"MIT", "Apache-2.0"}


def test_download_verifies_and_is_atomic(tmp_path):
    data = b"\x08\x07onnx-bytes" * 100
    src = tmp_path / "src.onnx"
    src.write_bytes(data)
    dest = tmp_path / "models" / "t.onnx"
    download(_spec(data), dest, url=src.as_uri())
    assert dest.read_bytes() == data
    assert is_valid(dest, _spec(data), check_hash=True)


def test_lfs_pointer_is_detected(tmp_path):
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 229738\n"
    src = tmp_path / "pointer"
    src.write_bytes(pointer)
    dest = tmp_path / "t.onnx"
    with pytest.raises(ModelError, match="LFS pointer"):
        download(_spec(b"x" * 229738), dest, url=src.as_uri())
    assert not dest.exists()
    assert not dest.with_suffix(".onnx.part").exists()


def test_size_mismatch_is_rejected(tmp_path):
    src = tmp_path / "short"
    src.write_bytes(b"abc")
    with pytest.raises(ModelError, match="size mismatch"):
        download(_spec(b"abcd"), tmp_path / "t.onnx", url=src.as_uri())


def test_ensure_without_download_raises(tmp_path):
    with pytest.raises(ModelError, match="missing"):
        models.ensure(tmp_path, "yunet", allow_download=False)


def test_objects_module_picks_one_model():
    assert models.keys_for_module("objects", "yolox") == ["yolox"]
    assert set(models.keys_for_module("hands")) == {"palm", "handpose"}
