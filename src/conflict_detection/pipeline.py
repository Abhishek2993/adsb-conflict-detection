"""End-to-end pipeline: ingest -> clean -> predict -> CPA/detect -> evaluate -> resolve.

:func:`run_pipeline` keeps everything in memory and returns a
:class:`PipelineResult`; :func:`write_outputs` writes the tables, metrics,
BlueSky scenarios and figures that the CLI and the Streamlit app read.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clean import CleaningReport, clean_reports, resample
from .config import Config
from .detect import (
    Metrics,
    SearchStats,
    actual_from_truth,
    actual_losses,
    detect_conflicts,
    evaluate,
    group_events,
)
from .geo import LocalFrame
from .ingest import Scenario
from .predict import estimate_states, prediction_errors
from .resolve import resolve_all
from .units import NM

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything one run produces (times in seconds from ``t0``)."""

    scenario: Scenario
    cfg: Config
    frame: LocalFrame
    t0: pd.Timestamp
    reports: pd.DataFrame
    cleaning: CleaningReport
    grid: pd.DataFrame
    states: pd.DataFrame
    detections: pd.DataFrame
    search: SearchStats
    events: pd.DataFrame
    actual: pd.DataFrame
    metrics: Metrics
    resolutions: pd.DataFrame
    pred_errors: pd.DataFrame
    truth_source: str
    timings_s: dict[str, float] = field(default_factory=dict)
    bluesky: pd.DataFrame | None = None

    def to_utc(self, seconds: pd.Series | np.ndarray | float) -> pd.Series | pd.Timestamp:
        """Convert pipeline seconds to UTC timestamps."""
        if np.isscalar(seconds):
            return self.t0 + pd.to_timedelta(float(seconds), unit="s")
        return self.t0 + pd.to_timedelta(pd.Series(seconds), unit="s")


def flight_hours(grid: pd.DataFrame) -> float:
    """Total tracked time of all aircraft, in hours."""
    return float(grid.groupby("icao24").t.agg(lambda s: s.max() - s.min()).sum() / 3600.0)


def summarise_prediction_errors(errors: pd.DataFrame) -> pd.DataFrame:
    """Median / 95th percentile / RMS error per look-ahead time (NM and ft)."""
    g = errors.groupby("lookahead_s")
    return pd.DataFrame(
        {
            "n": g.size(),
            "h_median_nm": g.err_h_m.median() / NM,
            "h_p95_nm": g.err_h_m.quantile(0.95) / NM,
            "h_rms_nm": np.sqrt(g.err_h_m.apply(lambda v: np.mean(v**2))) / NM,
            "v_median_ft": g.err_v_m.median() / 0.3048,
            "v_p95_ft": g.err_v_m.quantile(0.95) / 0.3048,
        }
    ).reset_index()


def run_pipeline(scenario: Scenario, cfg: Config | None = None) -> PipelineResult:
    """Run every stage on one scenario."""
    cfg = cfg or Config()
    timings: dict[str, float] = {}

    tic = time.perf_counter()
    reports, frame, cleaning = clean_reports(scenario.trajectories, cfg.cleaning)
    grid = resample(reports, cfg.cleaning, frame)
    t0 = reports.timestamp.min() - pd.to_timedelta(reports.t.min(), unit="s")
    timings["clean"] = time.perf_counter() - tic

    tic = time.perf_counter()
    eval_times = np.arange(0.0, reports.t.max() + 1e-9, cfg.eval_step_s)
    states = estimate_states(reports, eval_times, cfg)
    timings["predict"] = time.perf_counter() - tic

    tic = time.perf_counter()
    detections, search = detect_conflicts(states, cfg)
    events = group_events(detections, cfg)
    timings["detect"] = time.perf_counter() - tic

    if scenario.truth is not None:
        actual = actual_from_truth(scenario.truth, t0)
        truth_source = "synthetic truth.csv (noise-free trajectories)"
    else:
        actual = actual_losses(grid, cfg)
        truth_source = "recorded trajectories resampled to the common grid"
    metrics, actual_eval, events_eval = evaluate(
        events, detections, actual, cfg, flight_hours(grid)
    )

    tic = time.perf_counter()
    resolutions = resolve_all(events, states, cfg)
    timings["resolve"] = time.perf_counter() - tic

    errors = prediction_errors(states, grid, tuple(range(0, int(cfg.horizon_s) + 1, 60)))
    return PipelineResult(
        scenario=scenario,
        cfg=cfg,
        frame=frame,
        t0=t0,
        reports=reports,
        cleaning=cleaning,
        grid=grid,
        states=states,
        detections=detections,
        search=search,
        events=events_eval,
        actual=actual_eval,
        metrics=metrics,
        resolutions=resolutions,
        pred_errors=summarise_prediction_errors(errors),
        truth_source=truth_source,
        timings_s=timings,
    )


def _with_utc(df: pd.DataFrame, res: PipelineResult, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out:
            out[c + "_utc"] = res.to_utc(out[c]).to_numpy() if len(out) else pd.Series(dtype=str)
    return out


def conflict_table(res: PipelineResult) -> pd.DataFrame:
    """Conflict list for reporting: events + CPA location + outcome + resolution."""
    ev = _with_utc(res.events, res, ["t_first", "t_predicted_cpa"])
    if ev.empty:
        return ev
    first = res.detections.merge(
        ev[["icao_a", "icao_b", "t_first"]],
        left_on=["icao_a", "icao_b", "t_eval"],
        right_on=["icao_a", "icao_b", "t_first"],
    )
    mid_x = 0.5 * (first.cpa_x_a + first.cpa_x_b)
    mid_y = 0.5 * (first.cpa_y_a + first.cpa_y_b)
    lat, lon = res.frame.to_latlon(mid_x.to_numpy(), mid_y.to_numpy())
    loc = pd.DataFrame({"icao_a": first.icao_a, "icao_b": first.icao_b,
                        "t_first": first.t_first, "cpa_lat": lat, "cpa_lon": lon})  # fmt: skip
    ev = ev.merge(loc, on=["icao_a", "icao_b", "t_first"], how="left")
    if len(res.resolutions):
        cols = ["event_id", "manoeuvring", "manoeuvre", "resolved", "margin_after"]
        ev = ev.merge(res.resolutions[cols], on="event_id", how="left")
    if res.bluesky is not None and len(res.bluesky):
        cols = [c for c in ("event_id", "conflict_reproduced", "resolution_success")
                if c in res.bluesky]  # fmt: skip
        ev = ev.merge(res.bluesky[cols], on="event_id", how="left")
    return ev


def summary(res: PipelineResult) -> dict[str, Any]:
    """Machine-readable summary of a run (written to ``metrics.json``)."""
    resolved = res.resolutions.resolved.astype(bool) if len(res.resolutions) else pd.Series([])
    out: dict[str, Any] = {
        "scenario": res.scenario.name,
        "source": res.scenario.metadata.get("source", ""),
        "window_utc": [str(res.reports.timestamp.min()), str(res.reports.timestamp.max())],
        "aircraft": int(res.reports.icao24.nunique()),
        "reports": len(res.reports),
        "flight_hours": round(flight_hours(res.grid), 2),
        "predictor": res.cfg.predictor,
        "rule": res.cfg.rule,
        "horizon_s": res.cfg.horizon_s,
        "truth_source": res.truth_source,
        "detection": res.metrics.as_dict(),
        "pair_search": {
            **asdict(res.search),
            "reduction": round(res.search.reduction, 4),
        },
        "resolution_model": {
            "events": len(res.resolutions),
            "resolved": int(resolved.sum()),
            "success_rate": round(float(resolved.mean()), 4) if len(resolved) else None,
            "by_manoeuvre": res.resolutions.manoeuvre.value_counts().to_dict()
            if len(res.resolutions)
            else {},
        },
        "cleaning": res.cleaning.as_dict(),
        "timings_s": {k: round(v, 3) for k, v in res.timings_s.items()},
    }
    if res.bluesky is not None and len(res.bluesky) and "resolution_success" in res.bluesky:
        b = res.bluesky
        sim = b[b.simulable.astype(bool)]
        both = sim.dropna(subset=["resolution_success"])
        out["bluesky"] = {
            "encounters_exported": len(b),
            "simulated": len(sim),
            "conflict_reproduced": int(sim.conflict_reproduced.fillna(False).astype(bool).sum()),
            "resolved_runs": len(both),
            "resolution_success": int(both.resolution_success.astype(bool).sum()),
            "resolution_success_rate": round(float(both.resolution_success.astype(bool).mean()), 4)
            if len(both)
            else None,
        }
    return out


def write_outputs(
    res: PipelineResult,
    out_dir: str | Path,
    run_bluesky: bool = False,
    figures: bool = True,
) -> Path:
    """Write tables, metrics, BlueSky scenarios and figures to ``out_dir``."""
    from . import bluesky_bridge as bb

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.cfg.save(out / "config.json")
    meta = {"lat0": res.frame.lat0, "lon0": res.frame.lon0, "t0_utc": str(res.t0),
            "scenario": res.scenario.name}  # fmt: skip
    (out / "frame.json").write_text(json.dumps(meta, indent=2))

    _with_utc(res.detections, res, ["t_eval"]).to_csv(out / "detections.csv", index=False)
    _with_utc(res.actual, res, ["t_start", "t_end"]).to_csv(out / "actual_los.csv", index=False)
    _with_utc(res.resolutions, res, ["t_eval"]).to_csv(out / "resolutions.csv", index=False)
    res.pred_errors.round(4).to_csv(out / "prediction_errors.csv", index=False)
    res.grid.to_parquet(out / "grid.parquet", index=False)
    res.states.to_parquet(out / "states.parquet", index=False)

    scn_dir = out / "bluesky"
    manifest = bb.export_all(res.events, res.states, res.resolutions, res.frame, scn_dir, res.cfg)
    (scn_dir / "README.md").write_text(bb.manual_run_instructions(scn_dir))
    if run_bluesky and len(manifest):
        sim = manifest[manifest.simulable.astype(bool)]
        files = sim.unresolved.tolist() + sim.resolved.dropna().tolist()
        try:
            logs = bb.run_scenarios(files, scn_dir / "work")
            res.bluesky = bb.verify_logs(manifest, logs, res.cfg)
            res.bluesky.to_csv(scn_dir / "verification.csv", index=False)
        except RuntimeError as exc:
            log.warning("BlueSky verification skipped: %s", exc)

    conflict_table(res).to_csv(out / "conflicts.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(summary(res), indent=2, default=str))
    (out / "cleaning_report.json").write_text(json.dumps(res.cleaning.as_dict(), indent=2))
    if figures:
        from .viz import write_run_figures

        write_run_figures(res, out)
    return out
