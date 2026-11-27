"""Short look-ahead state estimation and straight-line propagation.

Two predictors share one interface. At every evaluation time ``t`` each one
returns, for every aircraft with a fresh enough report, an estimated state
``(x, y, z, vx, vy, vz)`` *at time t*; the look-ahead is then a straight line
``p(t + tau) = p(t) + v tau`` for ``tau`` in ``[0, horizon]``.

* ``cv`` (baseline): the last raw ADS-B report at or before ``t``, moved
  forward by its age with its own reported velocity.
* ``kalman``: a constant-velocity Kalman filter run over all reports up to
  ``t``. It smooths position noise and, more importantly, the noisy vertical
  rate, before the same straight-line extrapolation.

Both are *causal*: only reports with timestamps ``<= t`` are used. Aircraft
whose last report is older than ``stale_after_s`` are dropped from that
evaluation time (a missed-detection risk recorded in the hazard analysis).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, KalmanConfig
from .units import FPM

STATE_COLS = ["x", "y", "z", "vx", "vy", "vz"]


@dataclass
class KalmanTrack:
    """Filtered history of one aircraft (one row per accepted report)."""

    t: np.ndarray  # (n,) report times [s]
    state: np.ndarray  # (n, 3, 2) per axis: [position, velocity]
    cov: np.ndarray  # (n, 3, 2, 2) per-axis covariance


class ConstantVelocityKF:
    """Constant-velocity Kalman filter, decoupled into x, y and z axes.

    State per axis: ``[p, v]``. Motion model ``p' = p + v dt``, ``v' = v``,
    disturbed by a random acceleration of standard deviation ``sigma_a``
    (discrete white-noise acceleration model):

        F = [[1, dt], [0, 1]]
        Q = sigma_a^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]

    Measurements are the position (always) and the reported velocity (when
    present and enabled), with independent noise per axis. Because F, Q, H
    and R are block-diagonal by axis, three 2-state filters are exactly
    equivalent to one 6-state filter, and simpler to read.
    """

    def __init__(self, cfg: KalmanConfig | None = None) -> None:
        self.cfg = cfg or KalmanConfig()
        c = self.cfg
        self.sigma_a = np.array([c.accel_noise_h, c.accel_noise_h, c.accel_noise_v])
        self.r_pos = np.array([c.pos_noise_h_m, c.pos_noise_h_m, c.pos_noise_v_m]) ** 2
        self.r_vel = np.array([c.vel_noise_h_ms, c.vel_noise_h_ms, c.vel_noise_v_ms]) ** 2
        self.x = np.zeros((3, 2))
        self.P = np.zeros((3, 2, 2))

    def initialise(self, pos: np.ndarray, vel: np.ndarray | None = None) -> None:
        """Start the filter from one report. Unknown velocity gets a wide prior."""
        vel_known = vel is not None and np.all(np.isfinite(vel))
        self.x[:, 0] = pos
        self.x[:, 1] = vel if vel_known else 0.0
        var_v = self.r_vel if vel_known else np.full(3, 300.0**2)
        self.P[:] = 0.0
        self.P[:, 0, 0] = self.r_pos
        self.P[:, 1, 1] = var_v

    def predict(self, dt: float) -> None:
        """Propagate state and covariance by ``dt`` seconds."""
        if dt <= 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        self.x = self.x @ F.T
        self.P = F @ self.P @ F.T + self.sigma_a[:, None, None] ** 2 * q

    def update(self, pos: np.ndarray, vel: np.ndarray | None = None) -> None:
        """Fuse one report (position, and velocity where finite)."""
        for k in range(3):
            use_vel = self.cfg.use_velocity_measurements and vel is not None and np.isfinite(vel[k])
            if use_vel:
                H = np.eye(2)
                z = np.array([pos[k], vel[k]])
                R = np.diag([self.r_pos[k], self.r_vel[k]])
            else:
                H = np.array([[1.0, 0.0]])
                z = np.array([pos[k]])
                R = np.array([[self.r_pos[k]]])
            P = self.P[k]
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            self.x[k] = self.x[k] + K @ (z - H @ self.x[k])
            # Joseph form keeps P symmetric positive definite.
            I_KH = np.eye(2) - K @ H
            self.P[k] = I_KH @ P @ I_KH.T + K @ R @ K.T


def filter_track(
    t: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    cfg: KalmanConfig | None = None,
    max_gap_s: float = 60.0,
) -> KalmanTrack:
    """Run the Kalman filter forward over one aircraft's reports.

    The filter restarts after a gap longer than ``max_gap_s``: extrapolating
    a minute-old state through a gap and then fusing would let the stale
    estimate bias the new one.
    """
    kf = ConstantVelocityKF(cfg)
    n = len(t)
    states = np.empty((n, 3, 2))
    covs = np.empty((n, 3, 2, 2))
    for i in range(n):
        if i == 0 or t[i] - t[i - 1] > max_gap_s:
            kf.initialise(pos[i], vel[i])
        else:
            kf.predict(t[i] - t[i - 1])
            kf.update(pos[i], vel[i])
        states[i], covs[i] = kf.x, kf.P
    return KalmanTrack(t=t, state=states, cov=covs)


def _fill_missing_velocity(t: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Backward finite difference where the report has no velocity (causal)."""
    vel = vel.copy()
    bad = ~np.isfinite(vel).all(axis=1)
    for i in np.flatnonzero(bad):
        if i > 0 and t[i] > t[i - 1]:
            vel[i] = (pos[i] - pos[i - 1]) / (t[i] - t[i - 1])
        else:
            vel[i] = 0.0
    return vel


def _snap_level(vel: np.ndarray, threshold_fpm: float) -> np.ndarray:
    """Zero vertical rates below ``threshold_fpm`` (level-flight snapping)."""
    if threshold_fpm <= 0:
        return vel
    vel = vel.copy()
    vel[np.abs(vel[:, 2]) < threshold_fpm * FPM, 2] = 0.0
    return vel


def estimate_states(
    reports: pd.DataFrame,
    eval_times: np.ndarray,
    cfg: Config | None = None,
    method: str | None = None,
) -> pd.DataFrame:
    """Causal state estimate of every aircraft at every evaluation time.

    Args:
        reports: cleaned reports from :func:`clean.clean_reports`.
        eval_times: evaluation times [s, same origin as ``reports.t``].
        cfg: run configuration (stale threshold, Kalman tuning).
        method: ``"cv"`` or ``"kalman"`` (default: ``cfg.predictor``).

    Returns:
        Long table with columns ``t_eval, icao24, callsign, x, y, z, vx, vy,
        vz, age_s, sigma_pos_m``. ``sigma_pos_m`` is the 1-sigma horizontal
        position uncertainty of the Kalman estimate (NaN for ``cv``).
    """
    cfg = cfg or Config()
    method = method or cfg.predictor
    if method not in ("cv", "kalman"):
        raise ValueError(f"unknown predictor {method!r}")
    eval_times = np.asarray(eval_times, dtype=float)
    out = []
    for icao, g in reports.groupby("icao24", sort=False):
        g = g.sort_values("t")
        t = g.t.to_numpy()
        pos = g[["x", "y", "z"]].to_numpy()
        vel = g[["vx", "vy", "vz"]].to_numpy()
        k = np.searchsorted(t, eval_times, side="right") - 1
        valid = k >= 0
        age = np.where(valid, eval_times - t[np.maximum(k, 0)], np.inf)
        valid &= age <= cfg.stale_after_s
        if not valid.any():
            continue
        k, age, te = k[valid], age[valid], eval_times[valid]
        if method == "cv":
            v = _snap_level(_fill_missing_velocity(t, pos, vel)[k], cfg.level_snap_fpm)
            p = pos[k] + v * age[:, None]
            sigma = np.full(len(k), np.nan)
        else:
            trk = filter_track(t, pos, vel, cfg.kalman, cfg.cleaning.max_gap_s)
            s = trk.state[k]  # (m, 3, 2)
            v = _snap_level(s[:, :, 1], cfg.level_snap_fpm)
            p = s[:, :, 0] + v * age[:, None]
            # Horizontal position variance propagated over the age.
            P = trk.cov[k]
            var_x = P[:, 0, 0, 0] + 2 * age * P[:, 0, 0, 1] + age**2 * P[:, 0, 1, 1]
            var_y = P[:, 1, 0, 0] + 2 * age * P[:, 1, 0, 1] + age**2 * P[:, 1, 1, 1]
            sigma = np.sqrt(0.5 * (var_x + var_y))
        out.append(
            pd.DataFrame(
                {
                    "t_eval": te,
                    "icao24": icao,
                    "callsign": g.callsign.iloc[-1],
                    "x": p[:, 0],
                    "y": p[:, 1],
                    "z": p[:, 2],
                    "vx": v[:, 0],
                    "vy": v[:, 1],
                    "vz": v[:, 2],
                    "age_s": age,
                    "sigma_pos_m": sigma,
                }
            )
        )
    if not out:
        return pd.DataFrame(columns=["t_eval", "icao24", "callsign", *STATE_COLS, "age_s"])
    return pd.concat(out, ignore_index=True).sort_values(["t_eval", "icao24"], ignore_index=True)


def propagate(states: pd.DataFrame, horizon_s: float, step_s: float) -> np.ndarray:
    """Straight-line look-ahead positions.

    Returns an array of shape ``(n_aircraft, n_steps, 3)`` with positions at
    ``tau = 0, step, ..., horizon``.
    """
    tau = np.arange(0.0, horizon_s + 1e-9, step_s)
    p = states[["x", "y", "z"]].to_numpy()[:, None, :]
    v = states[["vx", "vy", "vz"]].to_numpy()[:, None, :]
    return p + v * tau[None, :, None]


def prediction_errors(
    states: pd.DataFrame,
    grid: pd.DataFrame,
    lookaheads_s: tuple[float, ...] = (0, 60, 120, 180, 240, 300),
) -> pd.DataFrame:
    """Horizontal and vertical error of the straight-line look-ahead.

    For each estimated state at ``t_eval`` and each look-ahead ``tau``, the
    predicted position is compared with where the aircraft actually was at
    ``t_eval + tau`` (from the resampled grid). Returns one row per
    (aircraft, t_eval, tau) that has a truth sample.
    """
    truth = grid.set_index(["icao24", "t"])[["x", "y", "z"]]
    rows = []
    for tau in lookaheads_s:
        pred = states[["icao24", "t_eval", "x", "y", "z", "vx", "vy", "vz"]].copy()
        pred["t"] = pred.t_eval + tau
        for c in ("x", "y", "z"):
            pred[c] = pred[c] + pred["v" + c] * tau
        m = pred.join(truth, on=["icao24", "t"], rsuffix="_true", how="inner")
        rows.append(
            pd.DataFrame(
                {
                    "icao24": m.icao24,
                    "t_eval": m.t_eval,
                    "lookahead_s": tau,
                    "err_h_m": np.hypot(m.x - m.x_true, m.y - m.y_true),
                    "err_v_m": (m.z - m.z_true).abs(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
