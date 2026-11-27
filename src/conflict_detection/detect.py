"""Predicted conflicts, actual losses of separation, and their comparison.

* :func:`detect_conflicts` runs the CPA test on predicted states at every
  evaluation time and returns one row per (time, pair) alert.
* :func:`group_events` merges consecutive alerts of a pair into conflict events.
* :func:`actual_losses` finds the real losses of separation on the resampled
  (recorded) trajectories.
* :func:`evaluate` matches the two and returns true positives, missed
  conflicts, false alarms and warning times.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import Config
from .cpa import all_pairs, candidate_pairs, compute_cpa, in_los, los_interval
from .units import FT, NM

PAIR = ["icao_a", "icao_b"]
EVENT_COLS = [
    "event_id", *PAIR, "callsign_a", "callsign_b", "t_first", "t_last", "n_alerts",
    "tcpa_first_s", "dcpa_first_nm", "vsep_first_ft", "dcpa_min_nm", "t_predicted_cpa",
    "severity_max",
]  # fmt: skip


def severity_score(
    dcpa_m: np.ndarray, vsep_m: np.ndarray, tcpa_s: np.ndarray, cfg: Config
) -> np.ndarray:
    """Severity in (0, 1]: how deep and how soon the predicted intrusion is.

    ``h = 1 - dCPA / D`` and ``v = 1 - |dz_CPA| / H`` measure penetration of
    the horizontal and vertical minima (1 = collision geometry, 0 = just at
    the minimum). Their geometric mean is high only if *both* are violated,
    mirroring the LoS definition. An urgency factor from 1 (now) down to 0.5
    (at the horizon) ranks imminent conflicts first.
    """
    d = cfg.separation.horizontal_m
    h = cfg.separation.vertical_m
    pen_h = np.clip(1.0 - np.asarray(dcpa_m) / d, 0.0, 1.0)
    pen_v = np.clip(1.0 - np.abs(np.asarray(vsep_m)) / h, 0.0, 1.0)
    urgency = 1.0 - 0.5 * np.clip(np.asarray(tcpa_s) / cfg.horizon_s, 0.0, 1.0)
    return np.sqrt(pen_h * pen_v) * urgency


@dataclass
class SearchStats:
    """Pair-search workload, to show the pre-filter avoids O(n^2) CPA tests."""

    eval_times: int = 0
    aircraft_states: int = 0
    naive_pairs: int = 0
    candidate_pairs: int = 0
    max_aircraft: int = 0

    @property
    def reduction(self) -> float:
        """Fraction of naive pairs removed by the pre-filter."""
        return 1.0 - self.candidate_pairs / self.naive_pairs if self.naive_pairs else 0.0


def detect_conflicts(
    states: pd.DataFrame,
    cfg: Config | None = None,
    use_prefilter: bool = True,
) -> tuple[pd.DataFrame, SearchStats]:
    """Flag predicted conflicts at every evaluation time.

    Rule ``"cpa"`` (the requirement): flag when ``0 < tCPA <= horizon``,
    ``dCPA < 5 NM`` and the vertical separation *at the CPA instant* is below
    1000 ft. Rule ``"interval"``: flag when the analytic window in which both
    minima are violated intersects ``[0, horizon]``; this also catches
    climbing/descending pairs whose vertical and horizontal closest points
    happen at different times.

    Returns the alert table and pair-search statistics.
    """
    cfg = cfg or Config()
    d_min, h_min, horizon = cfg.separation.horizontal_m, cfg.separation.vertical_m, cfg.horizon_s
    stats = SearchStats()
    rows = []
    for t_eval, g in states.groupby("t_eval", sort=True):
        n = len(g)
        stats.eval_times += 1
        stats.aircraft_states += n
        stats.max_aircraft = max(stats.max_aircraft, n)
        stats.naive_pairs += n * (n - 1) // 2
        if n < 2:
            continue
        pos = g[["x", "y", "z"]].to_numpy()
        vel = g[["vx", "vy", "vz"]].to_numpy()
        if use_prefilter:
            i, j = candidate_pairs(pos, vel, horizon, d_min, h_min)
        else:
            i, j = all_pairs(n)
        stats.candidate_pairs += len(i)
        if len(i) == 0:
            continue
        dp, dv = pos[j] - pos[i], vel[j] - vel[i]
        r = compute_cpa(*dp.T, *dv.T)
        t_in, t_out = los_interval(*dp.T, *dv.T, d_min, h_min)
        if cfg.rule == "cpa":
            flag = (r.tcpa > 0) & (r.tcpa <= horizon) & in_los(r.dcpa, r.dz_cpa, d_min, h_min)
        elif cfg.rule == "interval":
            flag = (t_in <= horizon) & (t_out > 0)
        else:
            raise ValueError(f"unknown detection rule {cfg.rule!r}")
        if not flag.any():
            continue
        i, j, k = i[flag], j[flag], np.flatnonzero(flag)
        tc = np.clip(r.tcpa[k], 0.0, None)
        icao = g.icao24.to_numpy()
        call = g.callsign.to_numpy()
        swap = icao[i] > icao[j]  # store pairs in a canonical (sorted) order
        a = np.where(swap, j, i)
        b = np.where(swap, i, j)
        cpa_a = pos[a] + vel[a] * tc[:, None]
        cpa_b = pos[b] + vel[b] * tc[:, None]
        rows.append(
            pd.DataFrame(
                {
                    "t_eval": t_eval,
                    "icao_a": icao[a],
                    "icao_b": icao[b],
                    "callsign_a": call[a],
                    "callsign_b": call[b],
                    "tcpa_s": r.tcpa[k],
                    "dcpa_nm": r.dcpa[k] / NM,
                    "vsep_cpa_ft": r.vsep_cpa[k] / FT,
                    "d_now_nm": r.d_now[k] / NM,
                    "vsep_now_ft": np.abs(r.dz_now[k]) / FT,
                    "t_los_in_s": np.clip(t_in[k], 0.0, None),
                    "severity": severity_score(r.dcpa[k], r.vsep_cpa[k], r.tcpa[k], cfg),
                    "cpa_x_a": cpa_a[:, 0],
                    "cpa_y_a": cpa_a[:, 1],
                    "cpa_x_b": cpa_b[:, 0],
                    "cpa_y_b": cpa_b[:, 1],
                }
            )
        )
    cols = ["t_eval", *PAIR, "callsign_a", "callsign_b", "tcpa_s", "dcpa_nm", "vsep_cpa_ft",
            "d_now_nm", "vsep_now_ft", "t_los_in_s", "severity",
            "cpa_x_a", "cpa_y_a", "cpa_x_b", "cpa_y_b"]  # fmt: skip
    if not rows:
        return pd.DataFrame(columns=cols), stats
    return pd.concat(rows, ignore_index=True)[cols], stats


def _runs(times: np.ndarray, max_gap: float) -> list[tuple[int, int]]:
    """Index ranges of runs in sorted ``times`` separated by more than ``max_gap``."""
    if len(times) == 0:
        return []
    breaks = np.flatnonzero(np.diff(times) > max_gap)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(times) - 1]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def group_events(detections: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Merge alerts of the same pair that are less than ``event_merge_gap_s`` apart.

    Each event keeps the geometry at its *first* alert (what a controller
    would see when the alert fires) and the worst severity over its life.
    """
    cfg = cfg or Config()
    if detections.empty:
        return pd.DataFrame(columns=EVENT_COLS)
    events = []
    for (a, b), g in detections.sort_values("t_eval").groupby(PAIR, sort=True):
        times = g.t_eval.to_numpy()
        for s, e in _runs(times, cfg.event_merge_gap_s):
            run = g.iloc[s : e + 1]
            first = run.iloc[0]
            events.append(
                {
                    "icao_a": a,
                    "icao_b": b,
                    "callsign_a": first.callsign_a,
                    "callsign_b": first.callsign_b,
                    "t_first": float(times[s]),
                    "t_last": float(times[e]),
                    "n_alerts": len(run),
                    "tcpa_first_s": float(first.tcpa_s),
                    "dcpa_first_nm": float(first.dcpa_nm),
                    "vsep_first_ft": float(first.vsep_cpa_ft),
                    "dcpa_min_nm": float(run.dcpa_nm.min()),
                    "t_predicted_cpa": float(first.t_eval + max(first.tcpa_s, 0.0)),
                    "severity_max": float(run.severity.max()),
                }
            )
    ev = pd.DataFrame(events).sort_values(["t_first", "icao_a", "icao_b"], ignore_index=True)
    ev.insert(0, "event_id", [f"C{k:03d}" for k in range(1, len(ev) + 1)])
    return ev


def actual_losses(grid: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Real losses of separation on the resampled (recorded) trajectories.

    At every grid time a KD-tree returns pairs closer than 5 NM horizontally;
    the vertical test then keeps those within 1000 ft. Consecutive LoS samples
    of a pair are merged into one event. An event is *evaluable* only if both
    aircraft were tracked for ``min_track_before_los_s`` before it started:
    a predictor cannot be blamed for a conflict that already existed when the
    aircraft first appeared in the data.
    """
    cfg = cfg or Config()
    d_min, h_min = cfg.separation.horizontal_m, cfg.separation.vertical_m
    hits = []
    for t, g in grid.groupby("t", sort=True):
        if len(g) < 2:
            continue
        xy = g[["x", "y"]].to_numpy()
        pairs = cKDTree(xy).query_pairs(d_min, output_type="ndarray")
        if len(pairs) == 0:
            continue
        i, j = pairs[:, 0], pairs[:, 1]
        z = g.z.to_numpy()
        hd = np.hypot(*(xy[i] - xy[j]).T)
        vd = np.abs(z[i] - z[j])
        ok = in_los(hd, vd, d_min, h_min)
        if not ok.any():
            continue
        icao = g.icao24.to_numpy()
        a = np.minimum(icao[i[ok]], icao[j[ok]])
        b = np.maximum(icao[i[ok]], icao[j[ok]])
        hits.append(
            pd.DataFrame(
                {"t": t, "icao_a": a, "icao_b": b, "hdist_nm": hd[ok] / NM, "vsep_ft": vd[ok] / FT}
            )
        )
    cols = ["los_id", *PAIR, "t_start", "t_end", "duration_s", "min_hdist_nm",
            "min_vsep_ft", "evaluable"]  # fmt: skip
    if not hits:
        return pd.DataFrame(columns=cols)
    hits = pd.concat(hits, ignore_index=True)
    first_seen = grid.groupby("icao24").t.min()
    gap = cfg.event_merge_gap_s
    events = []
    for (a, b), g in hits.groupby(PAIR, sort=True):
        times = g.t.to_numpy()
        for s, e in _runs(times, gap):
            run = g.iloc[s : e + 1]
            t0 = float(times[s])
            lead = t0 - max(first_seen[a], first_seen[b])
            events.append(
                {
                    "icao_a": a,
                    "icao_b": b,
                    "t_start": t0,
                    "t_end": float(times[e]),
                    "duration_s": float(times[e] - times[s]),
                    "min_hdist_nm": float(run.hdist_nm.min()),
                    "min_vsep_ft": float(run.vsep_ft.min()),
                    "evaluable": bool(lead >= cfg.min_track_before_los_s),
                }
            )
    ev = pd.DataFrame(events).sort_values(["t_start", "icao_a", "icao_b"], ignore_index=True)
    ev.insert(0, "los_id", [f"L{k:03d}" for k in range(1, len(ev) + 1)])
    return ev[cols]


def actual_from_truth(truth: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Actual LoS events from a synthetic ``truth.csv`` (noise-free ground truth)."""
    los = truth[truth.los.astype(bool)].copy()
    a = np.minimum(los.icao_a.astype(str), los.icao_b.astype(str))
    b = np.maximum(los.icao_a.astype(str), los.icao_b.astype(str))
    start = (pd.to_datetime(los.los_start, utc=True) - t0).dt.total_seconds()
    end = (pd.to_datetime(los.los_end, utc=True) - t0).dt.total_seconds()
    ev = pd.DataFrame(
        {
            "icao_a": a.to_numpy(),
            "icao_b": b.to_numpy(),
            "t_start": start.to_numpy(),
            "t_end": end.to_numpy(),
            "duration_s": (end - start).to_numpy(),
            "min_hdist_nm": los.min_hdist_nm.to_numpy(),
            "min_vsep_ft": los.vsep_cpa_ft.to_numpy(),
            "evaluable": True,
        }
    ).sort_values("t_start", ignore_index=True)
    ev.insert(0, "los_id", [f"L{k:03d}" for k in range(1, len(ev) + 1)])
    return ev


@dataclass
class Metrics:
    """Detection performance against actual losses of separation."""

    actual_los: int
    actual_los_not_evaluable: int
    true_positives: int
    missed: int
    alert_events: int
    false_alarms: int
    recall: float
    precision: float
    mean_warning_s: float
    median_warning_s: float
    min_warning_s: float
    flight_hours: float
    false_alarms_per_flight_hour: float

    def as_dict(self) -> dict[str, float | int]:
        """Metrics as a JSON-friendly dictionary."""
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def evaluate(
    events: pd.DataFrame,
    detections: pd.DataFrame,
    actual: pd.DataFrame,
    cfg: Config | None = None,
    flight_hours: float = float("nan"),
) -> tuple[Metrics, pd.DataFrame, pd.DataFrame]:
    """Match predicted conflict events with actual losses of separation.

    * True positive: an evaluable actual LoS with at least one alert for the
      same pair raised *strictly before* the LoS started and no earlier than
      ``horizon + eval_step`` before it. Warning time = LoS start - earliest
      such alert.
    * Missed conflict: an evaluable actual LoS without such an alert.
    * False alarm: a predicted event that is not followed (within the horizon
      after its last alert) by, and does not overlap, an actual LoS of the
      same pair.

    Returns ``(metrics, actual_with_outcome, events_with_outcome)``.
    """
    cfg = cfg or Config()
    window = cfg.horizon_s + cfg.eval_step_s
    act = actual.copy()
    act["detected"] = False
    act["warning_s"] = np.nan
    for idx, row in act.iterrows():
        d = detections[(detections.icao_a == row.icao_a) & (detections.icao_b == row.icao_b)]
        d = d[(d.t_eval < row.t_start) & (d.t_eval >= row.t_start - window)]
        if len(d):
            act.loc[idx, "detected"] = True
            act.loc[idx, "warning_s"] = row.t_start - d.t_eval.min()
    ev = events.copy()
    ev["confirmed"] = False
    ev["los_id"] = None
    for idx, row in ev.iterrows():
        m = act[(act.icao_a == row.icao_a) & (act.icao_b == row.icao_b)]
        m = m[(m.t_end >= row.t_first) & (m.t_start <= row.t_last + cfg.horizon_s)]
        if len(m):
            ev.loc[idx, "confirmed"] = True
            ev.loc[idx, "los_id"] = m.los_id.iloc[0]
    evaluable = act[act.evaluable.astype(bool)]
    tp = int(evaluable.detected.sum())
    missed = int((~evaluable.detected.astype(bool)).sum())
    fa = int((~ev.confirmed.astype(bool)).sum()) if len(ev) else 0
    warn = evaluable.loc[evaluable.detected.astype(bool), "warning_s"]
    metrics = Metrics(
        actual_los=len(evaluable),
        actual_los_not_evaluable=len(act) - len(evaluable),
        true_positives=tp,
        missed=missed,
        alert_events=len(ev),
        false_alarms=fa,
        recall=tp / len(evaluable) if len(evaluable) else float("nan"),
        precision=(len(ev) - fa) / len(ev) if len(ev) else float("nan"),
        mean_warning_s=float(warn.mean()) if len(warn) else float("nan"),
        median_warning_s=float(warn.median()) if len(warn) else float("nan"),
        min_warning_s=float(warn.min()) if len(warn) else float("nan"),
        flight_hours=float(flight_hours),
        false_alarms_per_flight_hour=fa / flight_hours if flight_hours > 0 else float("nan"),
    )
    return metrics, act, ev
