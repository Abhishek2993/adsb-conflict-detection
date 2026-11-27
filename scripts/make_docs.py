"""Regenerate every figure and number used in README.md and docs/.

Prerequisites (slow, need BlueSky, run once):

    python -m conflict_detection run --scenario data/sample --out results/sample --bluesky
    python -m conflict_detection run --scenario data/sample --out results/sample_robust \
        --bluesky --config configs/robust_resolution.json
    python -m conflict_detection run --scenario data/synthetic/noisy \
        --out results/synthetic_noisy --bluesky
    python -m conflict_detection benchmark --out results/benchmark.csv

Then:

    python scripts/make_docs.py

The pipelines themselves are re-run here (they take seconds and are
deterministic); only the BlueSky verification tables are read back from the
results folders above.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from conflict_detection.config import Config, SeparationMinima
from conflict_detection.ingest import load_scenario
from conflict_detection.pipeline import PipelineResult, run_pipeline, summary
from conflict_detection.units import FPM, FT
from conflict_detection.viz import (
    encounter_gif,
    map_png,
    prediction_error_png,
    resolution_png,
    scaling_png,
    separation_png,
    showcase_event,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "results"
SCENARIOS = {
    "sample": "data/sample",
    "synthetic_noisy": "data/synthetic/noisy",
    "synthetic_clean": "data/synthetic/clean",
}


def attach_bluesky(res: PipelineResult, folder: str) -> PipelineResult:
    """Attach a saved BlueSky verification table (if that run exists)."""
    path = ROOT / "results" / folder / "bluesky" / "verification.csv"
    if path.exists():
        res.bluesky = pd.read_csv(path)
    return res


def bluesky_breakdown(res: PipelineResult) -> dict:
    """BlueSky outcome split by whether the predicted event was confirmed by a real LoS."""
    if res.bluesky is None or "resolution_success" not in res.bluesky:
        return {}
    b = res.bluesky.merge(res.events[["event_id", "confirmed"]], on="event_id")
    b = b[b.simulable.astype(bool)]
    out = {}
    for label, g in (("confirmed", b[b.confirmed]), ("false_alarm", b[~b.confirmed])):
        ran = g.dropna(subset=["resolution_success"])
        out[label] = {
            "events": len(g),
            "conflict_reproduced": int(
                g.conflict_reproduced.astype("boolean").fillna(False).astype(bool).sum()
            ),
            "resolution_success": int(ran.resolution_success.astype(bool).sum()),
            "resolved_runs": len(ran),
        }
    failed = res.bluesky[res.bluesky.resolved_los.astype("boolean").fillna(False).astype(bool)]
    out["failed_resolutions"] = (
        failed[["event_id", "command", "resolved_min_hdist_nm", "resolved_vsep_at_min_ft"]]
        .round(2)
        .to_dict("records")
    )
    res_tab = res.resolutions.set_index("event_id")
    out["failed_margin_in_model"] = [
        round(float(res_tab.loc[e, "margin_after"]), 3) for e in failed.event_id
    ]
    return out


def bluesky_reproduction(res: PipelineResult) -> list[dict]:
    """Was the unresolved conflict reproduced in BlueSky, by pair type?

    Level pairs (both |vertical rate| < 100 ft/min) are created level in
    BlueSky, so the outcome depends on whether they start below 1000 ft apart.
    """
    if res.bluesky is None or "unresolved_los" not in res.bluesky:
        return []
    st = res.states.set_index(["t_eval", "icao24"])
    ev = res.events.set_index("event_id")
    rows = []
    for _, b in res.bluesky.dropna(subset=["unresolved_los"]).iterrows():
        e = ev.loc[b.event_id]
        a, c = st.loc[(e.t_first, e.icao_a)], st.loc[(e.t_first, e.icao_b)]
        level = max(abs(a.vz), abs(c.vz)) / FPM < 100
        rows.append(
            {
                "pair": "level" if level else "climbing/descending",
                "start_below_1000ft": bool(abs(a.z - c.z) / FT < 1000),
                "reproduced": bool(b.unresolved_los),
            }
        )
    t = pd.DataFrame(rows)
    g = t.groupby(["pair", "start_below_1000ft"]).reproduced.agg(["size", "sum"]).reset_index()
    return g.rename(columns={"size": "events", "sum": "reproduced"}).to_dict("records")


def alert_pair_types(res: PipelineResult) -> dict:
    """Count alert events by whether either aircraft climbs/descends >= 300 ft/min."""
    st = res.states.set_index(["t_eval", "icao24"])
    out = {"level_pairs": 0, "climbing_or_descending": 0}
    for _, e in res.events.iterrows():
        a, b = st.loc[(e.t_first, e.icao_a)], st.loc[(e.t_first, e.icao_b)]
        moving = max(abs(a.vz), abs(b.vz)) / FPM >= 300
        out["climbing_or_descending" if moving else "level_pairs"] += 1
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    scen = {k: load_scenario(ROOT / v) for k, v in SCENARIOS.items()}
    results: dict = {}

    # 1. Default configuration (the requirements) on every scenario.
    base = {k: run_pipeline(s, Config()) for k, s in scen.items()}
    attach_bluesky(base["sample"], "sample")
    attach_bluesky(base["synthetic_noisy"], "synthetic_noisy")
    for k, r in base.items():
        results[f"default_{k}"] = summary(r)
    results["bluesky_breakdown_sample"] = bluesky_breakdown(base["sample"])
    results["bluesky_breakdown_synthetic_noisy"] = bluesky_breakdown(base["synthetic_noisy"])
    results["bluesky_reproduction_sample"] = bluesky_reproduction(base["sample"])

    robust = attach_bluesky(
        run_pipeline(
            scen["sample"],
            replace(Config(), resolution=replace(Config().resolution, required_margin=1.1)),
        ),
        "sample_robust",
    )
    results["robust_sample"] = summary(robust)
    results["bluesky_breakdown_sample_robust"] = bluesky_breakdown(robust)

    # 2. Predictor x rule comparison (+ level-snap ablation).
    rows, errors = [], []
    for k in ("sample", "synthetic_noisy"):
        for predictor in ("cv", "kalman"):
            for rule in ("cpa", "interval"):
                for snap in (0.0, 200.0):
                    cfg = Config(predictor=predictor, rule=rule, level_snap_fpm=snap)
                    r = run_pipeline(scen[k], cfg)
                    rows.append(
                        {
                            "scenario": k,
                            "predictor": predictor,
                            "rule": rule,
                            "level_snap_fpm": snap,
                            **r.metrics.as_dict(),
                        }
                    )
                    if rule == "cpa" and snap == 0.0:
                        errors.append(r.pred_errors.assign(scenario=k, predictor=predictor))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "comparison.csv", index=False)
    pred_err = pd.concat(errors, ignore_index=True).round(4)
    pred_err.to_csv(OUT / "prediction_errors.csv", index=False)

    # 3. Altimetry tolerance sensitivity (vertical threshold 800 ft).
    tol = Config(separation=SeparationMinima(horizontal_nm=5.0, vertical_ft=800.0))
    tol_run = run_pipeline(scen["sample"], tol)
    results["tolerance_800ft_sample"] = summary(tol_run)
    results["tolerance_800ft_alert_pairs"] = alert_pair_types(tol_run)
    tol_snap = run_pipeline(scen["sample"], replace(tol, level_snap_fpm=200.0))
    results["tolerance_800ft_level_snap_200fpm_sample"] = summary(tol_snap)
    results["tolerance_800ft_level_snap_200fpm_alert_pairs"] = alert_pair_types(tol_snap)
    actual = base["sample"].actual
    results["sample_actual_los_vsep"] = {
        "events": len(actual),
        "min_vsep_ft_min": float(actual.min_vsep_ft.min()),
        "min_vsep_ft_max": float(actual.min_vsep_ft.max()),
        "at_or_above_900ft": int((actual.min_vsep_ft >= 900).sum()),
        "min_hdist_nm_min": float(actual.min_hdist_nm.min()),
    }

    # 4. Horizon sensitivity on the noisy synthetic set.
    hz = []
    for h in (60, 120, 180, 300, 600):
        for k in ("sample", "synthetic_noisy"):
            m = run_pipeline(scen[k], Config(horizon_s=float(h))).metrics
            hz.append({"scenario": k, "horizon_s": h, **m.as_dict()})
    pd.DataFrame(hz).to_csv(OUT / "horizon_sensitivity.csv", index=False)

    # 5. Per-encounter synthetic results.
    syn = base["synthetic_noisy"]
    truth = scen["synthetic_noisy"].truth
    rows = []
    for _, t in truth.iterrows():
        a, b = sorted([t.icao_a, t.icao_b])
        ev = syn.events[(syn.events.icao_a == a) & (syn.events.icao_b == b)]
        act = syn.actual[(syn.actual.icao_a == a) & (syn.actual.icao_b == b)]
        rows.append(
            {
                "encounter": t.encounter,
                "true_dcpa_nm": round(t.dcpa_nm, 2),
                "true_vsep_ft": round(t.vsep_cpa_ft),
                "true_los": bool(t.los),
                "alert_events": len(ev),
                "first_alert_dcpa_nm": round(ev.dcpa_first_nm.iloc[0], 2) if len(ev) else None,
                "warning_s": float(act.warning_s.iloc[0])
                if len(act) and act.detected.iloc[0]
                else None,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "synthetic_encounters.csv", index=False)

    # 6. Figures.
    s = base["sample"]
    ev = showcase_event(s)
    results["showcase_sample"] = {
        k: (v.item() if hasattr(v, "item") else v) for k, v in ev.to_dict().items()
    }
    map_png(s, FIG / "map_sample.png", ev.event_id)
    separation_png(s, ev.icao_a, ev.icao_b, FIG / "separation_sample.png")
    resolution_png(s, ev.event_id, FIG / "resolution_sample.png")
    encounter_gif(s, ev.event_id, FIG / "conflict_sample.gif")
    y = base["synthetic_noisy"]
    lvl = y.resolutions[y.resolutions.kind == "level"]
    if len(lvl):
        resolution_png(y, lvl.event_id.iloc[0], FIG / "resolution_synthetic_level.png")
    turn = y.events.sort_values("t_first").iloc[-1]  # the turning encounter is last
    separation_png(y, turn.icao_a, turn.icao_b, FIG / "separation_synthetic_turn.png")
    map_png(y, FIG / "map_synthetic.png")
    prediction_error_png(pred_err, FIG / "prediction_error.png")
    bench = ROOT / "results" / "benchmark.csv"
    if bench.exists():
        b = pd.read_csv(bench)
        scaling_png(b, FIG / "prefilter_scaling.png")
        results["benchmark"] = b.round(5).to_dict("records")

    (OUT / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT} and {FIG}")


if __name__ == "__main__":
    main()
