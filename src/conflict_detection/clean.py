"""Clean ADS-B reports and put every aircraft on a common time grid.

Two products come out of this module:

* :func:`clean_reports` keeps the *raw* reports that pass the outlier filter,
  adds local Cartesian coordinates and velocity components. The predictors
  consume these, strictly in time order, so detection at time ``t`` never uses
  a report from after ``t``.
* :func:`resample` interpolates the cleaned reports onto a common grid
  (default 5 s) without bridging data gaps. This is the best estimate of where
  aircraft really were, so it is used for the *actual* losses of separation,
  for evaluation and for plots, but not for prediction.

All outlier tests are causal: a report is judged against the last accepted
report of the same aircraft, never against later ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CleaningConfig
from .geo import LocalFrame
from .units import FPM, FT, KT, fpm_to_ms, ft_to_m, heading_to_unit, kt_to_ms

#: Position noise allowance added to the speed gate (ADS-B NACp 8 is < 93 m, 95 %).
_POS_TOL_M = 250.0
#: Altitude allowance added to the vertical-rate gate (two 25 ft quantisation steps + margin).
_ALT_TOL_FT = 100.0
#: After this many consecutive rejections, assume the reference was the outlier.
_MAX_REJECT_RUN = 5


@dataclass
class CleaningReport:
    """How many reports each cleaning rule removed (a data quality record)."""

    input_reports: int = 0
    missing_fields: int = 0
    on_ground: int = 0
    duplicates: int = 0
    altitude_out_of_range: int = 0
    position_jumps: int = 0
    altitude_jumps: int = 0
    short_tracks: int = 0
    invalid_velocities: int = 0
    output_reports: int = 0
    input_aircraft: int = 0
    output_aircraft: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int | list[str]]:
        """Return the counters as a plain dictionary."""
        return dict(self.__dict__)


def scenario_frame(df: pd.DataFrame) -> LocalFrame:
    """Local frame centred on the median position of all reports."""
    return LocalFrame(float(df.latitude.median()), float(df.longitude.median()))


def _jump_filter(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    alt_ft: np.ndarray,
    cfg: CleaningConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal gate on implied speed and vertical rate for one aircraft.

    Returns boolean masks (position_ok, altitude_ok). A report is rejected when
    reaching it from the last accepted report would need more than
    ``max_groundspeed_kt`` (plus a noise allowance) or more than
    ``max_vertical_rate_fpm``. Position and altitude are gated separately
    because ADS-B carries them in different fields and they fail differently.
    """
    n = len(t)
    pos_ok = np.ones(n, dtype=bool)
    alt_ok = np.ones(n, dtype=bool)
    vmax = cfg.max_groundspeed_kt * KT
    wmax = cfg.max_vertical_rate_fpm * FPM / FT  # ft/s
    tl, xl, yl, rej = t[0], x[0], y[0], 0
    for k in range(1, n):
        dt = t[k] - tl
        if np.hypot(x[k] - xl, y[k] - yl) > vmax * dt + _POS_TOL_M and rej < _MAX_REJECT_RUN:
            pos_ok[k] = False
            rej += 1
            continue
        tl, xl, yl, rej = t[k], x[k], y[k], 0
    ta, al, rej = t[0], alt_ft[0], 0
    for k in range(1, n):
        dt = t[k] - ta
        if abs(alt_ft[k] - al) > wmax * dt + _ALT_TOL_FT and rej < _MAX_REJECT_RUN:
            alt_ok[k] = False
            rej += 1
            continue
        ta, al, rej = t[k], alt_ft[k], 0
    return pos_ok, alt_ok


def clean_reports(
    df: pd.DataFrame,
    cfg: CleaningConfig | None = None,
    frame: LocalFrame | None = None,
) -> tuple[pd.DataFrame, LocalFrame, CleaningReport]:
    """Filter outliers and add local Cartesian state columns.

    Output columns (in addition to the normalised schema): ``t`` [s since the
    first report], ``x``, ``y`` [m, local East/North], ``z`` [m, barometric
    altitude], ``vx``, ``vy``, ``vz`` [m/s, from the reported velocity, NaN if
    not reported].
    """
    cfg = cfg or CleaningConfig()
    rep = CleaningReport(input_reports=len(df), input_aircraft=int(df.icao24.nunique()))
    d = df.copy()

    keep = d[["timestamp", "icao24", "latitude", "longitude", "altitude"]].notna().all(axis=1)
    rep.missing_fields = int((~keep).sum())
    d = d[keep]
    if "onground" in d:
        ground = d["onground"].fillna(False).astype(bool)
        rep.on_ground = int(ground.sum())
        d = d[~ground]
    before = len(d)
    d = d.drop_duplicates(["icao24", "timestamp"], keep="first")
    rep.duplicates = before - len(d)
    in_range = d.altitude.between(cfg.min_altitude_ft, cfg.max_altitude_ft)
    rep.altitude_out_of_range = int((~in_range).sum())
    d = d[in_range].sort_values(["icao24", "timestamp"]).reset_index(drop=True)

    frame = frame or scenario_frame(d)
    t0 = d.timestamp.min()
    d["t"] = (d.timestamp - t0).dt.total_seconds()
    d["x"], d["y"] = frame.to_xy(d.latitude.to_numpy(), d.longitude.to_numpy())

    pos_ok = np.ones(len(d), dtype=bool)
    alt_ok = np.ones(len(d), dtype=bool)
    for idx in d.groupby("icao24", sort=False).indices.values():
        p, a = _jump_filter(
            d.t.to_numpy()[idx],
            d.x.to_numpy()[idx],
            d.y.to_numpy()[idx],
            d.altitude.to_numpy()[idx],
            cfg,
        )
        pos_ok[idx], alt_ok[idx] = p, a
    rep.position_jumps = int((~pos_ok).sum())
    rep.altitude_jumps = int((pos_ok & ~alt_ok).sum())
    d = d[pos_ok & alt_ok].copy()

    # Implausible velocity fields are blanked, not dropped: the position is fine.
    bad_gs = d.groundspeed > cfg.max_groundspeed_kt
    bad_vr = d.vertical_rate.abs() > cfg.max_vertical_rate_fpm
    rep.invalid_velocities = int(bad_gs.sum() + bad_vr.sum())
    d.loc[bad_gs, ["groundspeed", "track"]] = np.nan
    d.loc[bad_vr, "vertical_rate"] = np.nan

    counts = d.groupby("icao24").icao24.transform("size")
    short = counts < cfg.min_points
    rep.short_tracks = int(short.sum())
    d = d[~short]

    d["z"] = ft_to_m(d.altitude.to_numpy())
    east, north = heading_to_unit(d.track.to_numpy())
    speed = kt_to_ms(d.groundspeed.to_numpy())
    d["vx"], d["vy"] = speed * east, speed * north
    d["vz"] = fpm_to_ms(d.vertical_rate.to_numpy())
    d = d.sort_values(["timestamp", "icao24"]).reset_index(drop=True)
    rep.output_reports = len(d)
    rep.output_aircraft = int(d.icao24.nunique())
    return d, frame, rep


def resample(
    reports: pd.DataFrame,
    cfg: CleaningConfig | None = None,
    frame: LocalFrame | None = None,
) -> pd.DataFrame:
    """Interpolate cleaned reports onto a common time grid, never across gaps.

    The grid is shared by all aircraft (multiples of ``resample_s`` from the
    first report), so separations can be computed by simple joins on time.
    A grid point is produced only if the reports on both sides of it are less
    than ``max_gap_s`` apart. Velocities are interpolated as East/North
    components to avoid the 359 deg -> 1 deg wrap-around problem.

    Returns columns ``timestamp, t, icao24, callsign, x, y, z, vx, vy, vz,
    latitude, longitude, altitude, age_s`` where ``age_s`` is the time since the
    last real report (a staleness indicator).
    """
    cfg = cfg or CleaningConfig()
    step = cfg.resample_s
    t0 = reports.timestamp.min()
    out = []
    for icao, g in reports.groupby("icao24", sort=False):
        g = g.sort_values("t")
        t = g.t.to_numpy()
        grid = np.arange(np.ceil(t[0] / step) * step, t[-1] + 1e-9, step)
        if len(grid) == 0:
            continue
        k = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
        k_next = np.minimum(k + 1, len(t) - 1)
        exact = t[k] == grid
        gap = (t[k_next] - t[k]) > cfg.max_gap_s
        ok = exact | ~gap
        grid, k = grid[ok], k[ok]
        res = {"t": grid, "icao24": icao, "callsign": g.callsign.iloc[-1]}
        for col in ("x", "y", "z"):
            res[col] = np.interp(grid, t, g[col].to_numpy())
        for col in ("vx", "vy", "vz"):
            v = g[col].to_numpy()
            fin = np.isfinite(v)
            res[col] = (
                np.interp(grid, t[fin], v[fin]) if fin.sum() >= 2 else np.full(len(grid), np.nan)
            )
        res["age_s"] = grid - t[k]
        out.append(pd.DataFrame(res))
    grid_df = pd.concat(out, ignore_index=True)
    grid_df["timestamp"] = t0 + pd.to_timedelta(grid_df.t, unit="s")
    frame = frame or scenario_frame(reports)
    grid_df["latitude"], grid_df["longitude"] = frame.to_latlon(
        grid_df.x.to_numpy(), grid_df.y.to_numpy()
    )
    grid_df["altitude"] = grid_df.z / FT
    cols = ["timestamp", "t", "icao24", "callsign", "x", "y", "z", "vx", "vy", "vz",
            "latitude", "longitude", "altitude", "age_s"]  # fmt: skip
    return grid_df[cols].sort_values(["t", "icao24"]).reset_index(drop=True)
