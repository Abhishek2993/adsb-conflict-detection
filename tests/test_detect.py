"""Detection thresholds, event grouping, actual LoS and evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from conflict_detection.clean import clean_reports, resample
from conflict_detection.config import Config
from conflict_detection.detect import (
    actual_from_truth,
    actual_losses,
    detect_conflicts,
    evaluate,
    group_events,
    severity_score,
)
from conflict_detection.ingest import load_scenario
from conflict_detection.predict import estimate_states
from conflict_detection.units import FPM, FT, KT, NM

CFG = Config()


def head_on_states(miss_nm, dz_ft=0.0, range_nm=40.0, t_eval=0.0, dvz_fpm=0.0):
    """Two aircraft closing head-on at 960 kt; CPA after range/960 h."""
    return pd.DataFrame(
        {
            "t_eval": t_eval,
            "icao24": ["aaaaaa", "bbbbbb"],
            "callsign": ["A", "B"],
            "x": [0.0, range_nm * NM],
            "y": [0.0, miss_nm * NM],
            "z": [37000 * FT, 37000 * FT + dz_ft * FT],
            "vx": [480 * KT, -480 * KT],
            "vy": [0.0, 0.0],
            "vz": [0.0, dvz_fpm * FPM],
        }
    )


@pytest.mark.parametrize(("miss", "flagged"), [(4.99, True), (5.01, False), (5.0, False)])
def test_horizontal_threshold_boundary(miss, flagged):
    det, _ = detect_conflicts(head_on_states(miss), CFG)
    assert (len(det) == 1) is flagged
    if flagged:
        assert det.dcpa_nm.iloc[0] == pytest.approx(4.99)
        assert det.tcpa_s.iloc[0] == pytest.approx(150.0)


@pytest.mark.parametrize(("dz", "flagged"), [(999.0, True), (1001.0, False), (1000.0, False)])
def test_vertical_threshold_boundary(dz, flagged):
    det, _ = detect_conflicts(head_on_states(1.0, dz_ft=dz), CFG)
    assert (len(det) == 1) is flagged


def test_beyond_horizon_not_flagged():
    # 100 NM apart: CPA after 375 s, beyond the 300 s horizon.
    det, _ = detect_conflicts(head_on_states(0.0, range_nm=100.0), CFG)
    assert det.empty
    det, _ = detect_conflicts(head_on_states(0.0, range_nm=100.0), Config(horizon_s=400.0))
    assert len(det) == 1


def test_diverging_pair_not_flagged():
    states = head_on_states(0.0, range_nm=-3.0)  # already passed each other
    det, _ = detect_conflicts(states, CFG)
    assert det.empty


def test_cpa_rule_vs_interval_rule_for_level_crossing():
    # 2 NM miss: horizontally inside 5 NM from 132.8 s to 167.2 s (CPA 150 s).
    # B starts 3800 ft below, climbing 2000 ft/min: within 1000 ft from 84 s
    # to 144 s, and 1200 ft ABOVE A at the horizontal CPA. The CPA-point rule
    # sees 1200 ft and stays silent; both minima are in fact violated together
    # between 132.8 s and 144 s, which the interval rule catches.
    states = head_on_states(2.0, dz_ft=-3800.0, dvz_fpm=2000.0)
    det_cpa, _ = detect_conflicts(states, Config(rule="cpa"))
    det_int, _ = detect_conflicts(states, Config(rule="interval"))
    assert det_cpa.empty
    assert len(det_int) == 1
    assert det_int.t_los_in_s.iloc[0] == pytest.approx(132.8, abs=0.1)


def test_unknown_rule_rejected():
    with pytest.raises(ValueError):
        detect_conflicts(head_on_states(1.0), Config(rule="magic"))


def test_pairs_stored_in_canonical_order_with_cpa_points():
    states = head_on_states(2.0).iloc[::-1]
    det, _ = detect_conflicts(states, CFG)
    row = det.iloc[0]
    assert (row.icao_a, row.icao_b) == ("aaaaaa", "bbbbbb")
    # CPA points: A has flown 20 NM east; B is 2 NM north of it.
    assert row.cpa_x_a / NM == pytest.approx(20.0)
    assert row.cpa_x_b / NM == pytest.approx(20.0)
    assert (row.cpa_y_b - row.cpa_y_a) / NM == pytest.approx(2.0)


def test_severity_ordering_and_range():
    d = np.array([0.0, 2.5, 4.9]) * NM
    s = severity_score(d, np.zeros(3), np.zeros(3), CFG)
    assert s[0] == pytest.approx(1.0)
    assert s[0] > s[1] > s[2] > 0
    later = severity_score(np.zeros(1), np.zeros(1), np.array([300.0]), CFG)
    assert later[0] == pytest.approx(0.5)
    assert severity_score(np.array([0.0]), np.array([1000 * FT]), np.zeros(1), CFG)[0] == 0


def test_prefilter_gives_identical_detections_on_sample():
    sc = load_scenario("data/sample")
    rep, _, _ = clean_reports(sc.trajectories)
    states = estimate_states(rep, np.arange(0, 1800, 10.0), CFG, "cv")
    fast, stats = detect_conflicts(states, CFG, use_prefilter=True)
    slow, naive = detect_conflicts(states, CFG, use_prefilter=False)
    pd.testing.assert_frame_equal(fast.reset_index(drop=True), slow.reset_index(drop=True))
    assert stats.candidate_pairs < 0.2 * naive.candidate_pairs
    assert stats.reduction > 0.8


def make_detections(times, a="aaaaaa", b="bbbbbb"):
    return pd.DataFrame(
        {
            "t_eval": times,
            "icao_a": a,
            "icao_b": b,
            "callsign_a": "A",
            "callsign_b": "B",
            "tcpa_s": 100.0,
            "dcpa_nm": 1.0,
            "vsep_cpa_ft": 0.0,
            "severity": np.linspace(0.2, 0.8, len(times)),
        }
    )


def test_group_events_splits_on_gaps():
    ev = group_events(make_detections([0, 10, 20, 100, 110]), Config(event_merge_gap_s=30))
    assert len(ev) == 2
    assert ev.n_alerts.tolist() == [3, 2]
    assert ev.event_id.tolist() == ["C001", "C002"]
    assert ev.t_predicted_cpa.iloc[0] == pytest.approx(100.0)
    assert group_events(make_detections([])).empty


def grid_two_aircraft(sep_nm, dz_ft, n=40, step=5.0, start_b=0.0):
    t = np.arange(n) * step
    rows = []
    for icao, y, z, t0 in (("aaaaaa", 0.0, 0.0, 0.0), ("bbbbbb", None, dz_ft, start_b)):
        tt = t[t >= t0]
        yy = sep_nm(tt) * NM if y is None else np.zeros(len(tt))
        rows.append(pd.DataFrame({"t": tt, "icao24": icao, "x": 0.0, "y": yy, "z": z * FT}))
    return pd.concat(rows, ignore_index=True)


def test_actual_losses_event_and_evaluable_flag():
    # Separation shrinks 10 -> 0.25 NM: LoS once below 5 NM (t > 100 s).
    grid = grid_two_aircraft(lambda t: 10 - t / 20, dz_ft=500)
    act = actual_losses(grid, CFG)
    assert len(act) == 1
    row = act.iloc[0]
    assert row.t_start == pytest.approx(105.0)
    assert row.min_vsep_ft == pytest.approx(500.0)
    assert bool(row.evaluable)
    # B appears only 30 s before the LoS: not evaluable.
    late = actual_losses(grid_two_aircraft(lambda t: 10 - t / 20, 500, start_b=75.0), CFG)
    assert not bool(late.evaluable.iloc[0])


def test_actual_losses_respects_vertical_minimum():
    assert actual_losses(grid_two_aircraft(lambda t: 1 + 0 * t, dz_ft=1000), CFG).empty


def test_evaluate_counts_tp_missed_and_false_alarms():
    actual = pd.DataFrame(
        {
            "los_id": ["L001", "L002"],
            "icao_a": ["aaaaaa", "cccccc"],
            "icao_b": ["bbbbbb", "dddddd"],
            "t_start": [500.0, 900.0],
            "t_end": [560.0, 950.0],
            "min_hdist_nm": [1.0, 2.0],
            "min_vsep_ft": [0.0, 0.0],
            "evaluable": [True, True],
        }
    )
    det = pd.concat(
        [
            make_detections([300, 310, 320, 330]),  # warns 200 s before L001
            make_detections([1000, 1010], a="eeeeee", b="ffffff"),  # never happens
        ],
        ignore_index=True,
    )
    ev = group_events(det, CFG)
    m, act, ev_out = evaluate(ev, det, actual, CFG, flight_hours=2.0)
    assert (m.true_positives, m.missed, m.false_alarms, m.alert_events) == (1, 1, 1, 2)
    assert m.recall == pytest.approx(0.5)
    assert m.precision == pytest.approx(0.5)
    assert m.mean_warning_s == pytest.approx(200.0)
    assert m.false_alarms_per_flight_hour == pytest.approx(0.5)
    assert act.detected.tolist() == [True, False]
    assert ev_out.confirmed.tolist() == [True, False]
    assert set(m.as_dict()) >= {"recall", "precision", "mean_warning_s"}


def test_alert_after_los_start_is_not_a_true_positive():
    actual = pd.DataFrame(
        {"los_id": ["L001"], "icao_a": ["aaaaaa"], "icao_b": ["bbbbbb"], "t_start": [500.0],
         "t_end": [560.0], "min_hdist_nm": [1.0], "min_vsep_ft": [0.0], "evaluable": [True]}
    )  # fmt: skip
    det = make_detections([500, 510])
    m, _, ev = evaluate(group_events(det, CFG), det, actual, CFG)
    assert m.true_positives == 0 and m.missed == 1
    assert ev.confirmed.all()  # the alert itself was not false


def test_synthetic_clean_end_to_end():
    sc = load_scenario("data/synthetic/clean")
    rep, frame, _ = clean_reports(sc.trajectories)
    states = estimate_states(rep, np.arange(0, rep.t.max(), 10.0), CFG, "cv")
    det, _ = detect_conflicts(states, CFG)
    ev = group_events(det, CFG)
    actual = actual_from_truth(sc.truth, sc.trajectories.timestamp.min())
    m, act, _ = evaluate(ev, det, actual, CFG)
    assert m.actual_los == 6
    assert m.true_positives == 6
    assert m.false_alarms == 0
    # Straight-line encounters are seen as soon as the CPA enters the horizon.
    assert act.warning_s.min() >= 120
    # The measured grid finds the same losses as the truth table.
    grid_actual = actual_losses(resample(rep, frame=frame), CFG)
    assert len(grid_actual) == 6
