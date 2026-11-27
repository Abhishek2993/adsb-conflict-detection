"""The synthetic generator reproduces its designed CPA values."""

import numpy as np
import pandas as pd
import pytest

from conflict_detection.synthetic import EncounterSpec, default_encounters, generate_scenario


@pytest.fixture(scope="module")
def clean_scenario():
    return generate_scenario(noise=None)


def test_schema(clean_scenario):
    traj, truth = clean_scenario
    assert {
        "timestamp",
        "icao24",
        "latitude",
        "longitude",
        "altitude",
        "groundspeed",
        "track",
        "vertical_rate",
    } <= set(traj.columns)
    assert traj.icao24.nunique() == 2 * len(default_encounters())
    assert len(truth) == len(default_encounters())


def test_designed_cpa_values_recovered(clean_scenario):
    _, truth = clean_scenario
    straight = truth[truth.kind != "turning"]
    # 1 s sampling: the sampled minimum is within a few metres of the analytic one.
    np.testing.assert_allclose(straight.dcpa_nm, straight.designed_dcpa_nm, atol=0.01)
    np.testing.assert_allclose(straight.vsep_cpa_ft, straight.designed_vsep_ft, atol=15)


def test_cpa_time_is_slot_offset(clean_scenario):
    _, truth = clean_scenario
    row = truth.set_index("encounter").loc["head_on_conflict"]
    assert row.t_cpa == pd.Timestamp("2026-01-15 10:08:00", tz="UTC")


def test_conflict_labels(clean_scenario):
    _, truth = clean_scenario
    labels = truth.set_index("encounter").los.to_dict()
    assert labels["head_on_conflict"] and not labels["head_on_clear"]
    assert labels["crossing_conflict"] and not labels["crossing_adjacent_level"]
    assert labels["overtaking_conflict"] and not labels["overtaking_clear"]
    assert labels["climbing_crossing_conflict"]
    assert labels["crossing_boundary_4_5nm"]
    # The turn converts an 8 NM planned miss into a loss of separation.
    assert labels["turn_into_conflict"]


def test_speeds_reported_in_knots(clean_scenario):
    traj, _ = clean_scenario
    a = traj[traj.callsign == "SYN00A"]
    assert a.groundspeed.mean() == pytest.approx(450, abs=0.01)
    assert a.track.mean() == pytest.approx(90, abs=0.01)


def test_noise_is_reproducible():
    a, _ = generate_scenario(seed=3)
    b, _ = generate_scenario(seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert (a.altitude % 25 == 0).all()


def test_zero_relative_velocity_rejected():
    spec = EncounterSpec("bad", "x", 90, 450, 90, 450, dcpa_nm=3)
    with pytest.raises(ValueError):
        generate_scenario([spec], noise=None)
