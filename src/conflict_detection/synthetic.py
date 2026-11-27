"""Synthetic two-aircraft encounters with known closest point of approach.

Each encounter is built *backwards from its CPA*: choose the CPA instant, the
two velocity vectors and a miss vector ``m`` perpendicular to the relative
velocity. Placing aircraft A at ``c - m/2`` and B at ``c + m/2`` at ``t_cpa``
and extrapolating each straight line makes ``|m|`` the exact miss distance and
``t_cpa`` the exact CPA time, because ``r(t) = m + v_rel (t - t_cpa)`` with
``m . v_rel = 0``. The resulting tracks are converted to latitude/longitude and
written in the same schema as real ADS-B data, optionally with realistic noise
(position jitter, 25 ft altitude quantisation, noisy velocities).

The turning encounter cannot be built that way (the turn is exactly what a
straight-line predictor cannot see), so its ground truth is measured
numerically from the noise-free trajectory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .cpa import in_los
from .geo import LocalFrame
from .units import FT, NM, fpm_to_ms, kt_to_ms, m_to_ft, ms_to_fpm, ms_to_kt, unit_to_heading

#: Default scenario centre (over the Swiss plateau, like the real sample).
DEFAULT_CENTRE = (46.9, 7.8)
DEFAULT_START = pd.Timestamp("2026-01-15 10:00:00", tz="UTC")


@dataclass(frozen=True)
class NoiseModel:
    """Measurement noise added to synthetic ADS-B reports (1 sigma values)."""

    position_m: float = 40.0  # NACp 8 corresponds to a 95 % bound of 92.6 m
    groundspeed_kt: float = 2.0
    track_deg: float = 0.5
    vertical_rate_fpm: float = 150.0
    altitude_quantum_ft: float = 25.0


@dataclass(frozen=True)
class EncounterSpec:
    """Parameters of one designed encounter."""

    name: str
    kind: str
    track_a_deg: float
    speed_a_kt: float
    track_b_deg: float
    speed_b_kt: float
    dcpa_nm: float
    vsep_cpa_ft: float = 0.0
    altitude_a_ft: float = 37000.0
    vrate_a_fpm: float = 0.0
    vrate_b_fpm: float = 0.0
    #: Optional turn of aircraft B: (start before the planned CPA [s],
    #: heading change [deg, positive = right], turn rate [deg/s]).
    turn: tuple[float, float, float] | None = None
    lateral_sign: float = 1.0


@dataclass(frozen=True)
class EncounterTruth:
    """Ground truth of a generated encounter (from the noise-free trajectories)."""

    encounter: str
    kind: str
    icao_a: str
    icao_b: str
    t_cpa: pd.Timestamp
    dcpa_nm: float
    vsep_cpa_ft: float
    min_hdist_nm: float
    los: bool
    los_start: pd.Timestamp | None
    los_end: pd.Timestamp | None
    designed_dcpa_nm: float
    designed_vsep_ft: float


def default_encounters() -> list[EncounterSpec]:
    """The standard synthetic suite: conflicts, near misses and hard cases."""
    return [
        EncounterSpec("head_on_conflict", "head-on", 90, 450, 270, 450, dcpa_nm=2.0),
        EncounterSpec("head_on_clear", "head-on", 90, 450, 270, 450, dcpa_nm=6.0),
        EncounterSpec(
            "crossing_conflict", "crossing", 0, 420, 90, 460, dcpa_nm=1.0, vsep_cpa_ft=500
        ),
        EncounterSpec(
            "crossing_adjacent_level", "crossing", 0, 420, 90, 460, dcpa_nm=0.0, vsep_cpa_ft=1000
        ),
        EncounterSpec("crossing_boundary_4_5nm", "crossing", 30, 440, 120, 440, dcpa_nm=4.5),
        EncounterSpec("overtaking_conflict", "overtaking", 45, 380, 45, 480, dcpa_nm=3.0),
        EncounterSpec("overtaking_clear", "overtaking", 45, 380, 45, 480, dcpa_nm=7.0),
        EncounterSpec(
            "climbing_crossing_conflict",
            "climbing",
            200,
            450,
            260,
            430,
            dcpa_nm=1.5,
            altitude_a_ft=33000,
            vrate_b_fpm=1500,
        ),
        EncounterSpec(
            "turn_into_conflict",
            "turning",
            90,
            450,
            270,
            450,
            dcpa_nm=8.0,
            turn=(150.0, 25.0, 1.5),
        ),
    ]


def _straight_states(
    spec: EncounterSpec, t: np.ndarray, t_cpa: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Noise-free states of A and B for a straight-line encounter."""
    ua = np.array([np.sin(np.radians(spec.track_a_deg)), np.cos(np.radians(spec.track_a_deg))])
    ub = np.array([np.sin(np.radians(spec.track_b_deg)), np.cos(np.radians(spec.track_b_deg))])
    va = ua * kt_to_ms(spec.speed_a_kt)
    vb = ub * kt_to_ms(spec.speed_b_kt)
    vrel = vb - va
    if np.hypot(*vrel) < 1e-6:
        raise ValueError(f"{spec.name}: zero relative velocity has no defined CPA")
    perp = np.array([-vrel[1], vrel[0]]) / np.hypot(*vrel)
    m = spec.lateral_sign * perp * spec.dcpa_nm * NM
    pa_c, pb_c = -m / 2.0, m / 2.0

    za_c = spec.altitude_a_ft * FT
    zb_c = za_c + spec.vsep_cpa_ft * FT
    wa, wb = fpm_to_ms(spec.vrate_a_fpm), fpm_to_ms(spec.vrate_b_fpm)
    dt = t - t_cpa

    def build(p_c: np.ndarray, v: np.ndarray, z_c: float, w: float) -> dict[str, np.ndarray]:
        return {
            "x": p_c[0] + v[0] * dt,
            "y": p_c[1] + v[1] * dt,
            "z": z_c + w * dt,
            "vx": np.full_like(dt, v[0]),
            "vy": np.full_like(dt, v[1]),
            "vz": np.full_like(dt, w),
        }

    return build(pa_c, va, za_c, wa), build(pb_c, vb, zb_c, wb)


def _turning_states(
    spec: EncounterSpec, t: np.ndarray, t_cpa: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """A flies straight; B starts on a clear straight line and then turns.

    The *planned* (pre-turn) geometry is a straight-line encounter with miss
    distance ``spec.dcpa_nm`` at ``t_cpa``; B starts turning ``turn[0]`` seconds
    before that CPA by ``turn[1]`` degrees at ``turn[2]`` deg/s.
    """
    assert spec.turn is not None
    a, b_plan = _straight_states(spec, t, t_cpa)
    lead, delta, rate = spec.turn
    t_turn = t_cpa - lead
    # Integrate B's heading profile on a fine grid, anchored at t_turn.
    dt_fine = 0.1
    t_fine = np.arange(t[0], t[-1] + dt_fine, dt_fine)
    turn_time = abs(delta) / rate
    frac = np.clip((t_fine - t_turn) / turn_time, 0.0, 1.0)
    heading = spec.track_b_deg + delta * frac
    speed = kt_to_ms(spec.speed_b_kt)
    vx = speed * np.sin(np.radians(heading))
    vy = speed * np.cos(np.radians(heading))
    # Position at t_turn from the planned straight line.
    x0 = np.interp(t_turn, t, b_plan["x"])
    y0 = np.interp(t_turn, t, b_plan["y"])
    x = x0 + np.concatenate(([0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * dt_fine)))
    y = y0 + np.concatenate(([0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * dt_fine)))
    i_turn = np.searchsorted(t_fine, t_turn)
    x -= x[i_turn] - x0
    y -= y[i_turn] - y0
    b = {
        "x": np.interp(t, t_fine, x),
        "y": np.interp(t, t_fine, y),
        "z": b_plan["z"],
        "vx": np.interp(t, t_fine, vx),
        "vy": np.interp(t, t_fine, vy),
        "vz": b_plan["vz"],
    }
    return a, b


def _to_reports(
    states: dict[str, np.ndarray],
    times: pd.DatetimeIndex,
    frame: LocalFrame,
    offset: tuple[float, float],
    icao24: str,
    callsign: str,
    rng: np.random.Generator | None,
    noise: NoiseModel,
) -> pd.DataFrame:
    """Turn noise-free ENU states into ADS-B-like reports."""
    x = states["x"] + offset[0]
    y = states["y"] + offset[1]
    z = states["z"]
    vx, vy, vz = states["vx"], states["vy"], states["vz"]
    gs = ms_to_kt(np.hypot(vx, vy))
    trk = unit_to_heading(vx, vy)
    vr = ms_to_fpm(vz)
    alt_ft = m_to_ft(z)
    if rng is not None:
        n = len(x)
        x = x + rng.normal(0, noise.position_m, n)
        y = y + rng.normal(0, noise.position_m, n)
        gs = gs + rng.normal(0, noise.groundspeed_kt, n)
        trk = np.mod(trk + rng.normal(0, noise.track_deg, n), 360)
        vr = vr + rng.normal(0, noise.vertical_rate_fpm, n)
        q = noise.altitude_quantum_ft
        alt_ft = np.round(alt_ft / q) * q
    lat, lon = frame.to_latlon(x, y)
    return pd.DataFrame(
        {
            "timestamp": times,
            "icao24": icao24,
            "callsign": callsign,
            "latitude": lat,
            "longitude": lon,
            "altitude": alt_ft,
            "groundspeed": gs,
            "track": trk,
            "vertical_rate": vr,
        }
    )


def _truth_from_states(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    times: pd.DatetimeIndex,
    d_min: float,
    h_min: float,
) -> tuple[int, float, float, np.ndarray]:
    hd = np.hypot(b["x"] - a["x"], b["y"] - a["y"])
    vd = np.abs(b["z"] - a["z"])
    k = int(np.argmin(hd))
    los = in_los(hd, vd, d_min, h_min)
    return k, float(hd[k]), float(vd[k]), los


StateBuilder = Callable[
    [EncounterSpec, np.ndarray, float], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
]


def generate_scenario(
    specs: list[EncounterSpec] | None = None,
    *,
    start: pd.Timestamp = DEFAULT_START,
    centre: tuple[float, float] = DEFAULT_CENTRE,
    slot_s: float = 900.0,
    t_cpa_s: float = 480.0,
    duration_s: float = 720.0,
    sample_s: float = 1.0,
    noise: NoiseModel | None = NoiseModel(),
    seed: int = 7,
    d_min_nm: float = 5.0,
    h_min_ft: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a scenario with one encounter per time slot.

    Args:
        specs: encounters to generate (default: :func:`default_encounters`).
        start: timestamp of the first slot.
        centre: (lat, lon) of the local frame; all encounters share the airspace.
        slot_s: spacing between encounters, so they never interact.
        t_cpa_s: CPA time measured from the start of each slot.
        duration_s: how long each aircraft is tracked.
        sample_s: ADS-B report interval.
        noise: noise model, or ``None`` for perfect measurements.
        seed: random seed, so the scenario is reproducible.

    Returns:
        ``(trajectories, truth)`` data frames.
    """
    specs = specs if specs is not None else default_encounters()
    frame = LocalFrame(*centre)
    rng = np.random.default_rng(seed) if noise is not None else None
    noise = noise or NoiseModel()
    t = np.arange(0.0, duration_s + 1e-9, sample_s)
    frames, truths = [], []
    for k, spec in enumerate(specs):
        builder: StateBuilder = _turning_states if spec.turn else _straight_states
        a, b = builder(spec, t, t_cpa_s)
        t0 = start + pd.Timedelta(seconds=k * slot_s)
        times = pd.DatetimeIndex(t0 + pd.to_timedelta(t, unit="s"))
        # Shift each encounter a little so the map does not overplot them.
        offset = (20_000.0 * np.cos(k), 20_000.0 * np.sin(k))
        icao_a, icao_b = f"f0{k:02d}0a", f"f0{k:02d}0b"
        frames.append(_to_reports(a, times, frame, offset, icao_a, f"SYN{k:02d}A", rng, noise))
        frames.append(_to_reports(b, times, frame, offset, icao_b, f"SYN{k:02d}B", rng, noise))
        idx, dmin, vsep, los = _truth_from_states(a, b, t, d_min_nm * NM, h_min_ft * FT)
        los_idx = np.flatnonzero(los)
        truths.append(
            EncounterTruth(
                encounter=spec.name,
                kind=spec.kind,
                icao_a=icao_a,
                icao_b=icao_b,
                t_cpa=times[idx],
                dcpa_nm=dmin / NM,
                vsep_cpa_ft=vsep / FT,
                min_hdist_nm=dmin / NM,
                los=bool(los.any()),
                los_start=times[los_idx[0]] if len(los_idx) else None,
                los_end=times[los_idx[-1]] if len(los_idx) else None,
                designed_dcpa_nm=spec.dcpa_nm,
                designed_vsep_ft=spec.vsep_cpa_ft,
            )
        )
    traj = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "icao24"])
    truth = pd.DataFrame([asdict(tr) for tr in truths]).round(6)
    return traj.reset_index(drop=True), truth
