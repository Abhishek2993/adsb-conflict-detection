"""Resolution manoeuvre kinematics and selection logic."""

import numpy as np
import pandas as pd
import pytest

from conflict_detection.config import Config
from conflict_detection.resolve import Manoeuvre, fly, manoeuvre_ladder, resolve_all, resolve_event
from conflict_detection.units import FPM, FT, KT, NM, unit_to_heading

CFG = Config()
TAU = np.arange(0.0, 300.0 + 1e-9, 2.0)
EAST_450 = np.array([0.0, 0.0, 37000 * FT, 450 * KT, 0.0, 0.0])


def test_ladder_is_ordered_by_cost_and_complete():
    ladder = manoeuvre_ladder(CFG)
    labels = [m.label for m in ladder]
    assert labels == [
        "speed +10 %", "speed -10 %",
        "heading +15 deg", "heading -15 deg",
        "level +1000 ft", "level -1000 ft",
        "heading +30 deg", "heading -30 deg",
    ]  # fmt: skip
    costs = [m.cost for m in ladder]
    assert costs == sorted(costs)
    assert Manoeuvre("none", 0, 0).label == "none"


def test_no_manoeuvre_is_a_straight_line():
    p = fly(EAST_450, Manoeuvre("none", 0, 0), TAU)
    np.testing.assert_allclose(p[-1], [450 * KT * 300, 0.0, 37000 * FT], atol=1e-6)


def test_heading_change_uses_bank_angle_turn_rate():
    p = fly(EAST_450, Manoeuvre("heading", 30.0, 4.0), TAU)
    v = np.diff(p[:, :2], axis=0)
    final_track = unit_to_heading(v[-1, 0], v[-1, 1])
    assert final_track == pytest.approx(120.0, abs=0.01)
    # 25 deg bank at 450 kt: ~1.13 deg/s, so a 30 deg turn takes ~27 s.
    rate = np.degrees(9.80665 * np.tan(np.radians(25)) / (450 * KT))
    assert rate == pytest.approx(1.13, abs=0.01)
    k_done = int(np.ceil(30.0 / rate / 2.0)) + 1
    tracks = unit_to_heading(v[:, 0], v[:, 1])
    assert tracks[k_done] == pytest.approx(120.0, abs=0.01)
    assert tracks[2] < 110.0
    # Speed and altitude are unchanged (2 s chords in the turn are ~0.02 %
    # shorter than the arc actually flown).
    np.testing.assert_allclose(np.hypot(*v.T) / 2.0, 450 * KT, rtol=5e-4)
    assert np.all(p[:, 2] == 37000 * FT)


def test_speed_change_accelerates_at_one_knot_per_second():
    p = fly(EAST_450, Manoeuvre("speed", -0.10, 1.0), TAU)
    gs = np.diff(p[:, 0]) / 2.0 / KT
    assert gs[-1] == pytest.approx(405.0, abs=0.5)
    assert gs[10] == pytest.approx(450 - 21, abs=1.5)  # ~21 s in


def test_level_change_stops_at_target():
    p = fly(EAST_450, Manoeuvre("level", -1000.0, 3.0), TAU)
    assert p[-1, 2] / FT == pytest.approx(36000.0)
    # 1500 ft/min: 1000 ft takes 40 s.
    assert p[10, 2] / FT == pytest.approx(37000 - 1500 * 20 / 60)
    assert p[20, 2] / FT == pytest.approx(36000.0)


def test_climbing_aircraft_keeps_its_rate_without_manoeuvre():
    st = EAST_450.copy()
    st[5] = 1500 * FPM
    p = fly(st, Manoeuvre("none", 0, 0), TAU)
    assert (p[-1, 2] - p[0, 2]) / FT == pytest.approx(7500.0)


def states_table(rows):
    cols = ["icao24", "callsign", "x", "y", "z", "vx", "vy", "vz"]
    return pd.DataFrame(rows, columns=cols).assign(t_eval=0.0)


def head_on(miss_nm=1.0, range_nm=40.0):
    return [
        ["aaaaaa", "A", 0.0, 0.0, 37000 * FT, 450 * KT, 0.0, 0.0],
        ["bbbbbb", "B", range_nm * NM, miss_nm * NM, 37000 * FT, -450 * KT, 0.0, 0.0],
    ]


EVENT = pd.Series({"event_id": "C001", "t_first": 0.0, "icao_a": "aaaaaa", "icao_b": "bbbbbb"})


def test_head_on_is_resolved_with_the_smallest_manoeuvre():
    res = resolve_event(EVENT, states_table(head_on()), CFG)
    assert res.resolved
    assert res.kind == "heading" and abs(res.value) == 15.0  # speed cannot fix head-on
    assert res.margin_before < 1.0 <= res.margin_after
    assert res.hdist_after_nm >= 5.0
    assert res.new_heading_deg in (75.0, 105.0, 255.0, 285.0)


def test_overtake_like_geometry_prefers_speed_change():
    # Slow closure from behind: a 10 % speed change is enough and is cheapest.
    rows = [
        ["aaaaaa", "A", 0.0, 0.0, 37000 * FT, 420 * KT, 0.0, 0.0],
        ["bbbbbb", "B", -6.0 * NM, 2.0 * NM, 37000 * FT, 450 * KT, 0.0, 0.0],
    ]
    res = resolve_event(EVENT, states_table(rows), CFG)
    assert res.resolved and res.kind == "speed"


def test_secondary_conflict_rejects_otherwise_valid_manoeuvre():
    base = resolve_event(EVENT, states_table(head_on()), CFG)
    assert base.kind == "heading"
    # Put level-1000 ft traffic everywhere a 15 deg turn could go: a third
    # aircraft at FL370 flying with A, 6 NM to each side of it.
    rows = [
        *head_on(),
        ["cccccc", "C", 0.0, 6.0 * NM, 37000 * FT, 450 * KT, 0.0, 0.0],
        ["dddddd", "D", 0.0, -6.0 * NM, 37000 * FT, 450 * KT, 0.0, 0.0],
        ["eeeeee", "E", 40.0 * NM, 7.0 * NM, 37000 * FT, -450 * KT, 0.0, 0.0],
        ["ffffff", "F", 40.0 * NM, -5.0 * NM, 37000 * FT, -450 * KT, 0.0, 0.0],
    ]
    res = resolve_event(EVENT, states_table(rows), CFG)
    assert res.resolved
    assert res.kind == "level"  # turns now create new conflicts with C-F


def test_unresolvable_when_already_too_close():
    rows = [
        ["aaaaaa", "A", 0.0, 0.0, 37000 * FT, 450 * KT, 0.0, 0.0],
        ["bbbbbb", "B", 0.5 * NM, 0.0, 37000 * FT, -450 * KT, 0.0, 0.0],
    ]
    res = resolve_event(EVENT, states_table(rows), CFG)
    assert not res.resolved
    assert res.manoeuvre == "none found"
    assert res.options_tried == 2 * len(manoeuvre_ladder(CFG))


def test_resolve_all_returns_one_row_per_event():
    events = pd.DataFrame([EVENT, EVENT.rename(None).replace("C001", "C002")])
    out = resolve_all(events, states_table(head_on()), CFG)
    assert out.event_id.tolist() == ["C001", "C002"]
    assert out.resolved.all()
