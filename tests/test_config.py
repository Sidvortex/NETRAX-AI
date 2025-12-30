import pytest
import yaml
from pydantic import ValidationError

from netrax.config import DEFAULT_CONFIG_PATH, Config, Settings, deep_merge


def test_default_yaml_matches_schema_defaults():
    # every key in the shipped YAML must exist in the schema, and vice versa
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    from_yaml = Config.model_validate(data)
    assert from_yaml == Config()


def test_missing_user_config_uses_defaults(tmp_path):
    s = Settings(tmp_path / "config.yaml")
    assert s.cfg.camera.source == 0
    assert s.cfg.modules.face_detect.enabled is True
    assert s.cfg.outputs.mqtt.enabled is False
    assert s.data_dir == tmp_path / "data"


def test_user_config_is_merged(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("camera:\n  source: video.mp4\nmodules:\n  hands:\n    enabled: true\n")
    s = Settings(p)
    assert s.cfg.camera.source == "video.mp4"
    assert s.cfg.camera.width == 640  # untouched default
    assert s.cfg.modules.hands.enabled is True
    assert s.cfg.modules.hands.every_n == 2


def test_typo_is_rejected(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("camera:\n  sorce: 1\n")
    with pytest.raises(ValidationError):
        Settings(p)


def test_digit_string_source_becomes_index(tmp_path):
    s = Settings(tmp_path / "config.yaml", overrides={"camera": {"source": "2"}})
    assert s.cfg.camera.source == 2


def test_update_validates_saves_and_notifies(tmp_path):
    p = tmp_path / "config.yaml"
    s = Settings(p)
    seen = []
    s.on_change(lambda old, new: seen.append((old.modules.face_recog.threshold, new.modules.face_recog.threshold)))
    s.update({"modules": {"face_recog": {"threshold": 0.5}}})
    assert s.cfg.modules.face_recog.threshold == 0.5
    assert seen == [(0.363, 0.5)]
    assert Settings(p).cfg.modules.face_recog.threshold == 0.5  # round trip


def test_bad_update_keeps_old_config(tmp_path):
    s = Settings(tmp_path / "config.yaml")
    with pytest.raises(ValidationError):
        s.update({"modules": {"face_recog": {"threshold": 7}}})
    assert s.cfg.modules.face_recog.threshold == 0.363
    assert not (tmp_path / "config.yaml").exists()


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": [1]}}
    b = {"x": {"y": 2}}
    out = deep_merge(a, b)
    assert out == {"x": {"y": 2, "z": [1]}}
    assert a == {"x": {"y": 1, "z": [1]}}
