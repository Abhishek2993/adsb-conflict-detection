"""Smoke tests: every figure renders from a real pipeline run."""

import pandas as pd
import pytest

from conflict_detection import viz
from conflict_detection.config import Config
from conflict_detection.ingest import load_scenario
from conflict_detection.pipeline import run_pipeline, write_outputs


@pytest.fixture(scope="module")
def res():
    return run_pipeline(load_scenario("data/synthetic/clean"), Config())


def test_showcase_and_pair_separation(res):
    ev = viz.showcase_event(res)
    assert ev is not None and bool(ev.confirmed)
    sep = viz.pair_separation(res.grid, ev.icao_a, ev.icao_b)
    assert sep.h_nm.min() < 5


def test_run_figures_written(res, tmp_path):
    write_outputs(res, tmp_path, figures=True)
    assert (tmp_path / "map.html").stat().st_size > 10_000
    html = (tmp_path / "map.html").read_text()
    assert "Predicted CPA at first alert" in html
    pngs = sorted(p.name for p in (tmp_path / "figures").glob("*.png"))
    assert "map.png" in pngs
    assert any(p.startswith("resolution_") for p in pngs)
    assert any(p.startswith("separation_") for p in pngs)


def test_plotly_separation_figure(res):
    ev = viz.showcase_event(res)
    fig = viz.separation_figure(res, ev.icao_a, ev.icao_b)
    names = {t.name for t in fig.data}
    assert "horizontal (recorded)" in names and "predicted dCPA at each alert" in names


def test_gif_and_doc_figures(res, tmp_path):
    ev = viz.showcase_event(res)
    gif = viz.encounter_gif(res, ev.event_id, tmp_path / "a.gif", lead_s=30)
    assert gif.stat().st_size > 1000
    err = res.pred_errors.assign(scenario="x", predictor="kalman")
    err = pd.concat([err, err.assign(predictor="cv")])
    assert viz.prediction_error_png(err, tmp_path / "e.png").exists()
    bench = pd.DataFrame({"aircraft": [10, 100], "kdtree": [1e-3, 2e-3], "naive": [1e-3, 1e-2]})
    assert viz.scaling_png(bench, tmp_path / "s.png").exists()


def test_no_events_means_no_showcase(res):
    empty = run_pipeline(load_scenario("data/synthetic/clean"), Config(horizon_s=1.0))
    assert viz.showcase_event(empty) is None or len(empty.events) > 0
