"""Kalman filter sanity and causal state estimation."""

import numpy as np
import pandas as pd
import pytest

from conflict_detection.config import Config, KalmanConfig
from conflict_detection.predict import (
    ConstantVelocityKF,
    estimate_states,
    filter_track,
    prediction_errors,
    propagate,
)

V_TRUE = np.array([230.0, -40.0, 5.0])


def cv_track(n=120, dt=5.0, noise_pos=0.0, noise_vel=0.0, seed=0, with_vel=True):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    pos = np.array([1000.0, -2000.0, 11000.0]) + t[:, None] * V_TRUE
    pos_meas = pos + rng.normal(0, 1, pos.shape) * [noise_pos, noise_pos, noise_pos / 3]
    vel = np.tile(V_TRUE, (n, 1)) + rng.normal(0, noise_vel, (n, 3))
    if not with_vel:
        vel[:] = np.nan
    return t, pos, pos_meas, vel


def test_noiseless_track_is_reproduced_exactly():
    t, pos, meas, vel = cv_track()
    trk = filter_track(t, meas, vel)
    np.testing.assert_allclose(trk.state[:, :, 0], pos, atol=1e-6)
    np.testing.assert_allclose(trk.state[-1, :, 1], V_TRUE, atol=1e-6)


def test_velocity_is_learned_from_positions_alone():
    t, _, meas, vel = cv_track(with_vel=False)
    trk = filter_track(t, meas, vel)
    np.testing.assert_allclose(trk.state[-1, :, 1], V_TRUE, atol=1e-3)


def test_filter_reduces_position_noise():
    t, pos, meas, vel = cv_track(n=400, noise_pos=100.0, noise_vel=3.0, seed=1)
    trk = filter_track(t, meas, vel, KalmanConfig(pos_noise_h_m=100.0, pos_noise_v_m=33.0))
    settled = slice(50, None)
    raw_err = np.hypot(*(meas[settled, :2] - pos[settled, :2]).T)
    kf_err = np.hypot(*(trk.state[settled, :2, 0] - pos[settled, :2]).T)
    assert np.sqrt(np.mean(kf_err**2)) < 0.5 * np.sqrt(np.mean(raw_err**2))


def test_covariance_stays_symmetric_positive_definite():
    t, _, meas, vel = cv_track(n=200, noise_pos=80.0, noise_vel=2.0, seed=2)
    trk = filter_track(t, meas, vel)
    np.testing.assert_allclose(trk.cov, np.swapaxes(trk.cov, -1, -2), atol=1e-9)
    assert (np.linalg.eigvalsh(trk.cov) > 0).all()
    # Uncertainty shrinks from its initial value as reports accumulate.
    assert trk.cov[-1, 0, 0, 0] < trk.cov[0, 0, 0, 0]


def test_predict_grows_uncertainty():
    kf = ConstantVelocityKF()
    kf.initialise(np.zeros(3), V_TRUE)
    p0 = kf.P.copy()
    kf.predict(30.0)
    assert (kf.P[:, 0, 0] > p0[:, 0, 0]).all()
    np.testing.assert_allclose(kf.x[:, 0], V_TRUE * 30.0)
    kf.predict(0.0)  # no-op
    np.testing.assert_allclose(kf.x[:, 0], V_TRUE * 30.0)


def test_filter_restarts_after_gap():
    t, _, meas, vel = cv_track(n=40)
    meas[20:] += [5000.0, 0.0, 0.0]  # track jumps after the gap
    t[20:] += 300.0
    trk = filter_track(t, meas, vel, max_gap_s=60.0)
    np.testing.assert_allclose(trk.state[20, :, 0], meas[20])


def reports_frame(t, meas, vel, icao="aaaaaa"):
    return pd.DataFrame(
        {
            "icao24": icao,
            "callsign": "X",
            "t": t,
            "x": meas[:, 0],
            "y": meas[:, 1],
            "z": meas[:, 2],
            "vx": vel[:, 0],
            "vy": vel[:, 1],
            "vz": vel[:, 2],
        }
    )


@pytest.mark.parametrize("method", ["cv", "kalman"])
def test_estimates_are_causal(method):
    t, _, meas, vel = cv_track(n=60, dt=5.0, noise_pos=50.0, noise_vel=1.0, seed=3)
    df = reports_frame(t, meas, vel)
    eval_t = np.array([100.0, 102.0])
    before = estimate_states(df, eval_t, method=method)
    tampered = df.copy()
    tampered.loc[tampered.t > 102.0, ["x", "y", "z"]] += 1e5
    after = estimate_states(tampered, eval_t, method=method)
    pd.testing.assert_frame_equal(before, after)


def test_cv_extrapolates_last_report_by_its_age():
    t, pos, meas, vel = cv_track(n=10, dt=10.0)
    st = estimate_states(reports_frame(t, meas, vel), np.array([47.0]), method="cv")
    assert st.age_s.iloc[0] == pytest.approx(7.0)
    expected = pos[4] + V_TRUE * 7.0
    np.testing.assert_allclose(st[["x", "y", "z"]].to_numpy()[0], expected)


def test_cv_uses_finite_difference_when_velocity_missing():
    t, _, meas, vel = cv_track(n=10, dt=10.0, with_vel=False)
    st = estimate_states(reports_frame(t, meas, vel), np.array([50.0]), method="cv")
    np.testing.assert_allclose(st[["vx", "vy", "vz"]].to_numpy()[0], V_TRUE)


def test_stale_and_not_yet_seen_aircraft_are_dropped():
    t, _, meas, vel = cv_track(n=10, dt=10.0)
    df = reports_frame(t, meas, vel)
    cfg = Config(stale_after_s=30.0)
    st = estimate_states(df, np.array([-5.0, 50.0, 115.0, 125.0]), cfg, "cv")
    assert st.t_eval.tolist() == [50.0, 115.0]


def test_unknown_method_rejected():
    t, _, meas, vel = cv_track(n=5)
    with pytest.raises(ValueError):
        estimate_states(reports_frame(t, meas, vel), np.array([1.0]), method="magic")


def test_empty_result_has_columns():
    t, _, meas, vel = cv_track(n=5)
    st = estimate_states(reports_frame(t, meas, vel), np.array([1e6]), method="cv")
    assert st.empty and "x" in st.columns


def test_propagate_straight_line():
    st = pd.DataFrame([{"x": 0.0, "y": 0.0, "z": 0.0, "vx": 1.0, "vy": 2.0, "vz": -1.0}])
    traj = propagate(st, horizon_s=300.0, step_s=10.0)
    assert traj.shape == (1, 31, 3)
    np.testing.assert_allclose(traj[0, -1], [300.0, 600.0, -300.0])


def test_prediction_errors_zero_for_perfect_cv():
    t, _, meas, vel = cv_track(n=80, dt=5.0)
    df = reports_frame(t, meas, vel)
    grid = df[["icao24", "t", "x", "y", "z"]]
    st = estimate_states(df, np.arange(0.0, 100.0, 10.0), method="kalman")
    err = prediction_errors(st, grid, lookaheads_s=(0, 60, 120))
    assert set(err.lookahead_s) == {0, 60, 120}
    assert err.err_h_m.max() < 1e-6 and err.err_v_m.max() < 1e-6
