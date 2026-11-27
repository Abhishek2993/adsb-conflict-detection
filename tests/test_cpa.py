"""CPA maths against hand-computed head-on, crossing and overtaking cases.

All cases use nautical miles and knots so the answers can be checked on paper:
closing speed in NM per second is ``kt / 3600``.
"""

import numpy as np
import pytest

from conflict_detection.cpa import (
    all_pairs,
    candidate_pairs,
    compute_cpa,
    in_los,
    los_interval,
)
from conflict_detection.units import FPM, FT, KT, NM

D = 5 * NM
H = 1000 * FT


def rel(a_pos_nm, a_vel_kt, b_pos_nm, b_vel_kt, dz_ft=0.0, dvz_fpm=0.0):
    """Relative state B - A in SI units from NM/kt inputs."""
    dx, dy = (np.array(b_pos_nm) - np.array(a_pos_nm)) * NM
    dvx, dvy = (np.array(b_vel_kt) - np.array(a_vel_kt)) * KT
    return dx, dy, dz_ft * FT, dvx, dvy, dvz_fpm * FPM


def test_head_on():
    # A at origin eastbound 480 kt; B 40 NM east, 2 NM north, westbound 480 kt.
    # Closing at 960 kt = 16 NM/min: CPA after 40/960 h = 150 s, miss 2 NM.
    r = compute_cpa(*rel((0, 0), (480, 0), (40, 2), (-480, 0)))
    assert r.tcpa == pytest.approx(150.0)
    assert r.dcpa / NM == pytest.approx(2.0)
    assert r.d_now / NM == pytest.approx(np.hypot(40, 2))


def test_crossing_right_angle_collision_course():
    # A 20 NM west of the crossing point eastbound, B 20 NM south northbound,
    # both 400 kt: both reach the crossing point after 20/400 h = 180 s.
    r = compute_cpa(*rel((-20, 0), (400, 0), (0, -20), (0, 400)))
    assert r.tcpa == pytest.approx(180.0)
    assert r.dcpa == pytest.approx(0.0, abs=1e-6)


def test_crossing_with_miss():
    # r = (20, -17) NM, v = (-400, 400) kt. r.v = -14800, |v|^2 = 320000,
    # tCPA = 14800/320000 h = 166.5 s; r + v t = (1.5, 1.5) NM -> 2.1213 NM.
    r = compute_cpa(*rel((-20, 0), (400, 0), (0, -17), (0, 400)))
    assert r.tcpa == pytest.approx(166.5)
    assert r.dcpa / NM == pytest.approx(np.hypot(1.5, 1.5))


def test_overtaking():
    # B 10 NM behind and 3 NM left of A, 200 kt faster: CPA after 10/200 h = 180 s.
    r = compute_cpa(*rel((0, 0), (300, 0), (-10, 3), (500, 0)))
    assert r.tcpa == pytest.approx(180.0)
    assert r.dcpa / NM == pytest.approx(3.0)


def test_vertical_separation_at_cpa():
    # Overtaking geometry; B is 2000 ft below and climbing 1000 ft/min.
    # After 3 minutes it is 1000 ft above A.
    r = compute_cpa(*rel((0, 0), (300, 0), (-10, 3), (500, 0), dz_ft=-2000, dvz_fpm=1000))
    assert r.dz_cpa / FT == pytest.approx(1000.0)
    assert r.vsep_cpa / FT == pytest.approx(1000.0)


def test_diverging_pair_has_negative_tcpa():
    r = compute_cpa(*rel((0, 0), (-480, 0), (10, 0), (480, 0)))
    assert r.tcpa < 0


def test_identical_velocity_defines_tcpa_zero():
    r = compute_cpa(*rel((0, 0), (450, 0), (3, 4), (450, 0)))
    assert r.tcpa == 0.0
    assert r.dcpa / NM == pytest.approx(5.0)


def test_vectorised_over_pairs():
    cases = [
        rel((0, 0), (480, 0), (40, 2), (-480, 0)),
        rel((-20, 0), (400, 0), (0, -20), (0, 400)),
        rel((0, 0), (300, 0), (-10, 3), (500, 0)),
    ]
    stacked = [np.array(c) for c in zip(*cases, strict=True)]
    r = compute_cpa(*stacked)
    np.testing.assert_allclose(r.tcpa, [150.0, 180.0, 180.0])
    np.testing.assert_allclose(r.dcpa / NM, [2.0, 0.0, 3.0], atol=1e-9)


def test_los_interval_head_on():
    # Intrusion while |x| < sqrt(25 - 4) NM; x closes at 960 kt.
    t_in, t_out = los_interval(*rel((0, 0), (480, 0), (40, 2), (-480, 0)), D, H)
    half = np.sqrt(21) / (960 / 3600)
    assert t_in == pytest.approx(150 - half)
    assert t_out == pytest.approx(150 + half)


def test_los_interval_needs_vertical_overlap():
    # Same head-on geometry but 1000 ft apart and level: no loss of separation.
    t_in, t_out = los_interval(*rel((0, 0), (480, 0), (40, 2), (-480, 0), dz_ft=1000), D, H)
    assert t_in > t_out


def test_los_interval_vertical_window_limits_horizontal():
    # Co-located, same velocity (always horizontally in LoS); B is 3000 ft below
    # climbing 2000 ft/min: vertical intrusion between 1 and 2 minutes.
    t_in, t_out = los_interval(
        *rel((0, 0), (400, 0), (1, 0), (400, 0), dz_ft=-3000, dvz_fpm=2000), D, H
    )
    assert t_in == pytest.approx(60.0)
    assert t_out == pytest.approx(120.0)


def test_los_interval_never_close():
    t_in, t_out = los_interval(*rel((0, 0), (480, 0), (40, 6), (-480, 0)), D, H)
    assert np.isinf(t_in) and np.isinf(t_out) and t_in > t_out


def test_los_interval_static_inside():
    t_in, t_out = los_interval(*rel((0, 0), (400, 0), (1, 0), (400, 0)), D, H)
    assert t_in == -np.inf and t_out == np.inf


def _brute_force(pos, vel, horizon):
    i, j = all_pairs(len(pos))
    r = compute_cpa(*(pos[j] - pos[i]).T, *(vel[j] - vel[i]).T)
    flagged = (r.tcpa > 0) & (r.tcpa <= horizon) & (r.dcpa < D) & (r.vsep_cpa < H)
    t_in, t_out = los_interval(*(pos[j] - pos[i]).T, *(vel[j] - vel[i]).T, D, H)
    flagged |= (t_in <= horizon) & (t_out >= 0)
    return {(a, b) for a, b, f in zip(i, j, flagged, strict=True) if f}


@pytest.mark.parametrize("seed", range(5))
def test_prefilter_is_lossless(seed):
    rng = np.random.default_rng(seed)
    n = 400
    pos = np.column_stack(
        [
            rng.uniform(-150, 150, n) * NM,
            rng.uniform(-150, 150, n) * NM,
            rng.choice(np.arange(300, 400, 10), n) * 100 * FT,
        ]
    )
    hdg = rng.uniform(0, 2 * np.pi, n)
    spd = rng.uniform(380, 500, n) * KT
    vel = np.column_stack([spd * np.sin(hdg), spd * np.cos(hdg), rng.normal(0, 3, n)])
    horizon = 300.0
    expected = _brute_force(pos, vel, horizon)
    i, j = candidate_pairs(pos, vel, horizon, D, H)
    got = set(zip(i.tolist(), j.tolist(), strict=True))
    assert expected <= got
    # And it prunes most of the n(n-1)/2 = 79,800 pairs.
    assert len(got) < 0.1 * n * (n - 1) / 2


def test_prefilter_small_inputs():
    i, j = candidate_pairs(np.zeros((1, 3)), np.zeros((1, 3)), 300, D, H)
    assert len(i) == len(j) == 0
    far = np.array([[0, 0, 0], [500 * NM, 0, 0]], dtype=float)
    i, j = candidate_pairs(far, np.zeros((2, 3)), 300, D, H)
    assert len(i) == 0


def test_in_los_boundaries_and_float_artefact():
    # Adjacent flight levels are exactly 1000 ft apart: separated, even though
    # the subtraction in metres is not exactly 304.8.
    for lo, hi in [(37000, 38000), (36000, 37000), (35000, 36000), (40000, 41000)]:
        assert not in_los(0.0, hi * FT - lo * FT, D, H)
    assert in_los(4.99 * NM, 0.0, D, H)
    assert not in_los(5.01 * NM, 0.0, D, H)
    assert not in_los(5.0 * NM, 0.0, D, H)
    assert in_los(0.0, 999 * FT, D, H)
    assert not in_los(0.0, 1001 * FT, D, H)
    # Both minima must be violated at the same time.
    assert not in_los(1 * NM, 2000 * FT, D, H)
    assert not in_los(6 * NM, 0, D, H)
