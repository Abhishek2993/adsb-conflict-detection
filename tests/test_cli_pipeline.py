"""CLI and end-to-end pipeline smoke tests (figures and BlueSky disabled)."""

import json

import pandas as pd
import pytest

from conflict_detection.cli import main
from conflict_detection.config import Config
from conflict_detection.ingest import load_scenario
from conflict_detection.pipeline import conflict_table, run_pipeline, summary


@pytest.fixture(scope="module")
def synthetic_result():
    return run_pipeline(load_scenario("data/synthetic/noisy"), Config())


def test_pipeline_on_synthetic(synthetic_result):
    res = synthetic_result
    assert res.metrics.actual_los == 6
    assert res.metrics.true_positives == 6
    assert res.truth_source.startswith("synthetic")
    s = summary(res)
    assert s["resolution_model"]["events"] == len(res.events)
    assert set(res.pred_errors.lookahead_s) == {0, 60, 120, 180, 240, 300}
    table = conflict_table(res)
    assert {"cpa_lat", "cpa_lon", "manoeuvre", "t_first_utc"} <= set(table.columns)
    assert table.cpa_lat.between(45, 49).all()


def test_cli_run_writes_outputs(tmp_path, capsys):
    out = tmp_path / "res"
    code = main(["run", "--scenario", "data/synthetic/clean", "--out", str(out), "--no-figures",
                 "--predictor", "cv", "--horizon", "240"])  # fmt: skip
    assert code == 0
    expected = ["conflicts.csv", "detections.csv", "actual_los.csv", "resolutions.csv",
                "metrics.json", "config.json", "grid.parquet", "bluesky/manifest.json"]  # fmt: skip
    for name in expected:
        assert (out / name).exists(), name
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["predictor"] == "cv" and metrics["horizon_s"] == 240
    assert metrics["detection"]["recall"] == 1.0
    assert json.loads((out / "config.json").read_text())["horizon_s"] == 240
    assert "pair search" in capsys.readouterr().out


def test_cli_config_file(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"rule": "interval", "cleaning": {"resample_s": 10.0}}))
    out = tmp_path / "res"
    assert main(["run", "--scenario", "data/synthetic/clean", "--out", str(out),
                 "--no-figures", "--config", str(cfg)]) == 0  # fmt: skip
    saved = json.loads((out / "config.json").read_text())
    assert saved["rule"] == "interval" and saved["cleaning"]["resample_s"] == 10.0


def test_cli_compare_and_benchmark(tmp_path):
    out = tmp_path / "cmp"
    assert main(["compare", "--scenario", "data/synthetic/clean", "--out", str(out)]) == 0
    table = pd.read_csv(out / "comparison.csv")
    assert len(table) == 4
    assert (table.recall == 1.0).all()
    bench = tmp_path / "bench.csv"
    assert main(["benchmark", "--sizes", "20", "60", "--out", str(bench)]) == 0
    b = pd.read_csv(bench)
    assert list(b.aircraft) == [20, 60] and (b.kdtree_pairs <= b.naive_pairs).all()


def test_cli_synthetic(tmp_path):
    assert main(["synthetic", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "noisy" / "truth.csv").exists()


def test_cli_bluesky_verify_with_manual_logs(tmp_path):
    out = tmp_path / "res"
    main(["run", "--scenario", "data/synthetic/clean", "--out", str(out), "--no-figures"])
    logs = tmp_path / "logs"
    logs.mkdir()
    # No logs present: every row reports nothing simulated but the command works.
    assert main(["bluesky-verify", "--manifest", str(out / "bluesky/manifest.json"),
                 "--logs", str(logs)]) == 0  # fmt: skip
    assert (out / "bluesky/verification.csv").exists()


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        main([])
