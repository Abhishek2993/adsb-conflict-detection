"""Ingest normalisation, outlier filtering and gap-aware resampling."""

import numpy as np
import pandas as pd
import pytest

from conflict_detection.clean import clean_reports, resample
from conflict_detection.config import CleaningConfig
from conflict_detection.ingest import (
    Scenario,
    fetch_opensky,
    load_scenario,
    normalise,
    save_scenario,
)
from conflict_detection.synthetic import generate_scenario
from conflict_detection.units import FT, KT


def straight_track(n=60, dt=5.0, icao="abc123", alt=37000.0, gs=450.0, trk=90.0):
    """A level eastbound track near 47N 8E."""
    t = pd.Timestamp("2026-01-01 12:00", tz="UTC") + pd.to_timedelta(np.arange(n) * dt, unit="s")
    dist = np.arange(n) * dt * gs * KT
    lon = 8.0 + np.degrees(dist / (6371000 * np.cos(np.radians(47.0))))
    return pd.DataFrame(
        {
            "timestamp": t,
            "icao24": icao,
            "callsign": "TEST1 ",
            "latitude": 47.0,
            "longitude": lon,
            "altitude": alt,
            "groundspeed": gs,
            "track": trk,
            "vertical_rate": 0.0,
        }
    )


def test_normalise_opensky_si_columns():
    raw = pd.DataFrame(
        {
            "time": [1_600_000_000, 1_600_000_010],
            "icao24": ["ABC123", "ABC123"],
            "lat": [47.0, 47.01],
            "lon": [8.0, 8.0],
            "baroaltitude": [11277.6, 11277.6],
            "velocity": [231.5, 231.5],
            "heading": [0.0, 0.0],
            "vertrate": [5.08, 5.08],
        }
    )
    out = normalise(raw)
    assert out.icao24.tolist() == ["abc123", "abc123"]
    assert out.altitude.iloc[0] == pytest.approx(37000.0)
    assert out.groundspeed.iloc[0] == pytest.approx(450.0, abs=0.1)
    assert out.vertical_rate.iloc[0] == pytest.approx(1000.0, abs=0.1)
    assert str(out.timestamp.dt.tz) == "UTC"
    assert out.callsign.tolist() == ["", ""]


def test_normalise_strips_callsign_and_requires_columns():
    out = normalise(straight_track(3))
    assert out.callsign.iloc[0] == "TEST1"
    with pytest.raises(ValueError, match="missing required"):
        normalise(pd.DataFrame({"icao24": ["a"]}))


def test_scenario_round_trip(tmp_path):
    traj, truth = generate_scenario(noise=None)
    save_scenario(Scenario("demo", traj, {"source": "test"}, truth), tmp_path / "demo")
    sc = load_scenario(tmp_path / "demo")
    assert sc.name == "demo"
    assert sc.n_aircraft == traj.icao24.nunique()
    assert sc.truth is not None and sc.truth.los.sum() == truth.los.sum()
    assert str(sc.truth.t_cpa.dt.tz) == "UTC"


def test_load_single_files(tmp_path):
    df = straight_track(5)
    df.to_csv(tmp_path / "t.csv", index=False)
    df.to_parquet(tmp_path / "t.parquet")
    assert len(load_scenario(tmp_path / "t.csv").trajectories) == 5
    assert len(load_scenario(tmp_path / "t.parquet").trajectories) == 5
    with pytest.raises(FileNotFoundError):
        load_scenario(tmp_path / "nothing.txt")


def test_committed_sample_loads():
    sc = load_scenario("data/sample")
    assert sc.n_aircraft >= 50
    assert sc.metadata["altitude_reference"] == "barometric"


def test_opensky_requires_env_credentials(monkeypatch):
    monkeypatch.delenv("OPENSKY_USERNAME", raising=False)
    monkeypatch.delenv("OPENSKY_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="OPENSKY_USERNAME"):
        fetch_opensky("2024-01-01 12:00", "2024-01-01 13:00", (5.0, 45.0, 10.0, 48.0))


def test_clean_track_passes_untouched():
    df = normalise(straight_track())
    out, _, rep = clean_reports(df)
    assert rep.output_reports == len(df)
    assert rep.position_jumps == rep.altitude_jumps == 0
    # Velocity components from 450 kt due east.
    assert out.vx.mean() == pytest.approx(450 * KT)
    assert out.vy.abs().max() < 1e-9
    assert out.z.iloc[0] == pytest.approx(37000 * FT)


def test_position_and_altitude_spikes_removed():
    df = straight_track()
    df.loc[20, "latitude"] += 0.5  # ~30 NM jump in 5 s
    df.loc[30, "altitude"] = 123100.0  # out of range
    df.loc[40, "altitude"] = 33000.0  # 4000 ft in 5 s
    out, _, rep = clean_reports(normalise(df))
    assert rep.position_jumps == 1
    assert rep.altitude_out_of_range == 1
    assert rep.altitude_jumps == 1
    assert len(out) == len(df) - 3
    assert out.altitude.min() == 37000.0


def test_filter_recovers_when_reference_was_outlier():
    # The very first report is bad: after a few rejections the filter must
    # re-anchor on the good data instead of rejecting the whole track.
    df = straight_track()
    df.loc[0, "latitude"] += 0.5
    _, _, rep = clean_reports(normalise(df))
    assert rep.output_reports >= len(df) - 6


def test_ground_duplicates_short_tracks_and_bad_velocities():
    a = straight_track(icao="aaaaaa")
    a["onground"] = False
    a.loc[0, "onground"] = True
    a = pd.concat([a, a.iloc[[5]]])  # duplicate report
    a.loc[a.index[10], "groundspeed"] = 2000.0
    b = straight_track(n=5, icao="bbbbbb")
    b["onground"] = False
    out, _, rep = clean_reports(normalise(pd.concat([a, b])), CleaningConfig(min_points=10))
    assert rep.on_ground == 1
    assert rep.duplicates == 1
    assert rep.short_tracks == 5
    assert rep.invalid_velocities == 1
    assert out.icao24.unique().tolist() == ["aaaaaa"]
    assert out.groundspeed.isna().sum() == 1


def test_resample_on_common_grid_without_bridging_gaps():
    df = straight_track(n=60, dt=4.0)
    df = df.drop(index=range(20, 40))  # 84 s gap
    out, frame, _ = clean_reports(normalise(df))
    g = resample(out, CleaningConfig(resample_s=5.0, max_gap_s=60.0), frame)
    assert np.allclose(np.mod(g.t, 5.0), 0.0)
    t_gap_lo, t_gap_hi = 19 * 4.0, 40 * 4.0
    inside = (g.t > t_gap_lo) & (g.t < t_gap_hi)
    assert not inside.any()
    # A constant-latitude track is a curved parallel: in the tangent plane it
    # bows by ~60 m over 30 NM. Interpolation must stay on it.
    assert g.y.abs().max() < 100.0
    assert (g.age_s >= 0).all() and g.age_s.max() < 5.0


def test_resample_interpolates_velocity_components_across_north():
    # Track flips between 359 and 1 degrees: averaging angles would give 180.
    df = straight_track(n=20, trk=0.0)
    df["track"] = np.where(np.arange(20) % 2 == 0, 359.0, 1.0)
    out, frame, _ = clean_reports(normalise(df))
    g = resample(out, CleaningConfig(resample_s=2.5), frame)
    assert (g.vy > 0.99 * 450 * KT).all()
