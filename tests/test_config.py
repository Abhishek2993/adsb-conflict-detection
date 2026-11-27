"""Configuration round-trip tests."""

import json

import pytest

from conflict_detection.config import Config


def test_defaults_match_requirements():
    cfg = Config()
    assert cfg.separation.horizontal_nm == 5.0
    assert cfg.separation.vertical_ft == 1000.0
    assert cfg.horizon_s == 300.0
    assert cfg.prediction_step_s == 10.0
    assert cfg.cleaning.resample_s == 5.0
    assert cfg.separation.horizontal_m == pytest.approx(9260.0)
    assert cfg.separation.vertical_m == pytest.approx(304.8)


def test_save_and_load(tmp_path):
    cfg = Config(horizon_s=120.0, predictor="cv")
    path = tmp_path / "config.json"
    cfg.save(path)
    loaded = Config.from_dict(json.loads(path.read_text()))
    assert loaded == cfg


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        Config.from_dict({"horizon": 10})
