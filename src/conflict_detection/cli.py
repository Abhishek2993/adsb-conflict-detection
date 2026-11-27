"""Command-line interface: ``python -m conflict_detection <command>``.

Commands:
    run             full pipeline on one scenario, writes results/
    compare         every predictor x detection rule on one or more scenarios
    benchmark       pair-search scaling, KD-tree pre-filter vs naive O(n^2)
    build-sample    cut the sample scenario out of traffic's bundled data
    synthetic       generate the synthetic encounter scenarios
    fetch-opensky   download a scenario from the OpenSky historical database
    bluesky-verify  score BlueSky logs (headless re-run or manual GUI logs)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def _config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.from_dict(json.loads(Path(args.config).read_text())) if args.config else Config()
    overrides = {}
    for name in ("predictor", "rule", "horizon_s", "prediction_step_s", "eval_step_s"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    if getattr(args, "resample_s", None) is not None:
        overrides["cleaning"] = replace(cfg.cleaning, resample_s=args.resample_s)
    return replace(cfg, **overrides)


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="JSON file with configuration overrides")
    p.add_argument("--predictor", choices=["cv", "kalman"], help="state predictor (kalman)")
    p.add_argument("--rule", choices=["cpa", "interval"], help="detection rule (cpa)")
    p.add_argument("--horizon", dest="horizon_s", type=float, help="look-ahead [s] (300)")
    p.add_argument("--step", dest="prediction_step_s", type=float, help="look-ahead step [s] (10)")
    p.add_argument("--eval-step", dest="eval_step_s", type=float, help="evaluation interval [s]")
    p.add_argument("--resample", dest="resample_s", type=float, help="common grid step [s] (5)")


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline and write outputs."""
    from .ingest import load_scenario
    from .pipeline import run_pipeline, summary, write_outputs

    cfg = _config_from_args(args)
    scenario = load_scenario(args.scenario)
    tic = time.perf_counter()
    res = run_pipeline(scenario, cfg)
    out = write_outputs(res, args.out, run_bluesky=args.bluesky, figures=not args.no_figures)
    s = summary(res)
    d = s["detection"]
    print(f"scenario      {s['scenario']}: {s['aircraft']} aircraft, {s['flight_hours']} flight h")
    print(f"predictor     {cfg.predictor}, rule {cfg.rule}, horizon {cfg.horizon_s:.0f} s")
    print(f"pair search   {res.search.candidate_pairs} CPA tests instead of "
          f"{res.search.naive_pairs} ({100 * res.search.reduction:.1f} % pruned)")  # fmt: skip
    print(f"conflicts     {d['alert_events']} predicted events, {d['actual_los']} actual LoS")
    print(f"detection     TP {d['true_positives']}  missed {d['missed']}  "
          f"false alarms {d['false_alarms']}  recall {d['recall']}  "
          f"precision {d['precision']}  mean warning {d['mean_warning_s']:.0f} s")  # fmt: skip
    r = s["resolution_model"]
    print(f"resolution    {r['resolved']}/{r['events']} resolved in the prediction model")
    if "bluesky" in s:
        b = s["bluesky"]
        print(
            f"bluesky       {b['conflict_reproduced']}/{b['simulated']} conflicts reproduced, "
            f"{b['resolution_success']}/{b['resolved_runs']} resolutions kept separation"
        )
    print(f"outputs       {out}  ({time.perf_counter() - tic:.1f} s)")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Evaluate every predictor x rule combination."""
    from .ingest import load_scenario
    from .pipeline import run_pipeline

    base = _config_from_args(args)
    rows, errors = [], []
    for path in args.scenario:
        scenario = load_scenario(path)
        for predictor in ("cv", "kalman"):
            for rule in ("cpa", "interval"):
                res = run_pipeline(scenario, replace(base, predictor=predictor, rule=rule))
                rows.append({"scenario": scenario.name, "predictor": predictor, "rule": rule,
                             **res.metrics.as_dict()})  # fmt: skip
                if rule == "cpa":
                    e = res.pred_errors.assign(scenario=scenario.name, predictor=predictor)
                    errors.append(e)
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv", index=False)
    pd.concat(errors).round(4).to_csv(out / "prediction_errors_comparison.csv", index=False)
    cols = ["scenario", "predictor", "rule", "actual_los", "true_positives", "missed",
            "alert_events", "false_alarms", "recall", "precision", "mean_warning_s"]  # fmt: skip
    print(table[cols].to_string(index=False))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Time the KD-tree pre-filter against the naive all-pairs CPA search."""
    from .cpa import all_pairs, candidate_pairs, compute_cpa
    from .units import FT, KT, NM

    cfg = Config()
    d_min, h_min, horizon = cfg.separation.horizontal_m, cfg.separation.vertical_m, cfg.horizon_s
    rng = np.random.default_rng(0)
    rows = []
    for n in args.sizes:
        # Constant traffic density: ~1 aircraft per 400 NM^2, like busy en-route airspace.
        half = np.sqrt(n * 400.0) / 2 * NM
        pos = np.column_stack([rng.uniform(-half, half, n), rng.uniform(-half, half, n),
                               rng.choice(np.arange(290, 420, 10), n) * 100 * FT])  # fmt: skip
        hdg = rng.uniform(0, 2 * np.pi, n)
        spd = rng.uniform(380, 500, n) * KT
        vel = np.column_stack([spd * np.sin(hdg), spd * np.cos(hdg), np.zeros(n)])
        timings = {}
        for name, finder in (
            ("kdtree", lambda p=pos, v=vel: candidate_pairs(p, v, horizon, d_min, h_min)),
            ("naive", lambda n=n: all_pairs(n)),
        ):
            if name == "naive" and n > args.max_naive:
                continue
            reps = 3
            tic = time.perf_counter()
            for _ in range(reps):
                i, j = finder()
                compute_cpa(*(pos[j] - pos[i]).T, *(vel[j] - vel[i]).T)
            timings[name] = (time.perf_counter() - tic) / reps
            timings[name + "_pairs"] = len(i)
        rows.append({"aircraft": n, **timings})
        print(rows[-1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


def cmd_build_sample(args: argparse.Namespace) -> int:
    """Rebuild data/sample from traffic's bundled data."""
    from .ingest import build_traffic_sample, save_scenario

    sc = build_traffic_sample(args.collection, args.start, args.stop)
    out = save_scenario(sc, args.out)
    print(f"wrote {out}: {sc.n_aircraft} aircraft, {len(sc.trajectories)} reports")
    return 0


def cmd_synthetic(args: argparse.Namespace) -> int:
    """Generate clean and noisy synthetic scenarios."""
    from .ingest import Scenario, save_scenario
    from .synthetic import NoiseModel, generate_scenario

    for label, noise in (("clean", None), ("noisy", NoiseModel())):
        traj, truth = generate_scenario(noise=noise, seed=args.seed)
        meta = {"name": f"synthetic_{label}", "seed": args.seed,
                "source": "conflict_detection.synthetic.generate_scenario",
                "noise": None if noise is None else noise.__dict__,
                "altitude_reference": "barometric"}  # fmt: skip
        out = save_scenario(Scenario(meta["name"], traj, meta, truth), Path(args.out) / label)
        print(f"wrote {out}")
    return 0


def cmd_fetch_opensky(args: argparse.Namespace) -> int:
    """Download a scenario from OpenSky (credentials from the environment)."""
    from .ingest import fetch_opensky, save_scenario

    sc = fetch_opensky(args.start, args.stop, tuple(args.bounds))
    out = save_scenario(sc, args.out)
    print(f"wrote {out}: {sc.n_aircraft} aircraft, {len(sc.trajectories)} reports")
    return 0


def cmd_bluesky_verify(args: argparse.Namespace) -> int:
    """Score BlueSky runs of exported scenarios."""
    from . import bluesky_bridge as bb

    manifest_path = Path(args.manifest)
    manifest = pd.DataFrame(json.loads(manifest_path.read_text()))
    if args.logs:
        logs = bb.find_manual_logs(manifest, args.logs)
    else:
        sim = manifest[manifest.simulable.astype(bool)]
        files = sim.unresolved.tolist() + sim.resolved.dropna().tolist()
        logs = bb.run_scenarios(files, manifest_path.parent / "work")
    result = bb.verify_logs(manifest, logs)
    out = Path(args.out) if args.out else manifest_path.parent / "verification.csv"
    result.to_csv(out, index=False)
    print(result.to_string(index=False))
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for all sub-commands."""
    parser = argparse.ArgumentParser(prog="conflict_detection", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the full pipeline on a scenario")
    p.add_argument("--scenario", default="data/sample", help="scenario directory or file")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--bluesky", action="store_true", help="verify resolutions in BlueSky")
    p.add_argument("--no-figures", action="store_true", help="skip map and plots")
    _add_config_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="compare predictors and detection rules")
    p.add_argument("--scenario", nargs="+", default=["data/sample"])
    p.add_argument("--out", default="results/compare")
    _add_config_args(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("benchmark", help="pair-search scaling benchmark")
    p.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000, 5000])
    p.add_argument("--max-naive", type=int, default=5000)
    p.add_argument("--out", default="results/benchmark.csv")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("build-sample", help="rebuild data/sample from traffic's bundled data")
    p.add_argument("--collection", default="switzerland")
    p.add_argument("--start", default="2018-08-01 14:00")
    p.add_argument("--stop", default="2018-08-01 15:00")
    p.add_argument("--out", default="data/sample")
    p.set_defaults(func=cmd_build_sample)

    p = sub.add_parser("synthetic", help="generate synthetic encounter scenarios")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_synthetic)

    p = sub.add_parser("fetch-opensky", help="download a scenario from OpenSky history")
    p.add_argument("--start", required=True)
    p.add_argument("--stop", required=True)
    p.add_argument("--bounds", type=float, nargs=4, required=True,
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"))  # fmt: skip
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fetch_opensky)

    p = sub.add_parser("bluesky-verify", help="score BlueSky runs of exported scenarios")
    p.add_argument("--manifest", required=True, help="results/.../bluesky/manifest.json")
    p.add_argument("--logs", help="folder with CDRLOG_*.log from a manual run (else re-run)")
    p.add_argument("--out", help="CSV to write (default: next to the manifest)")
    p.set_defaults(func=cmd_bluesky_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
