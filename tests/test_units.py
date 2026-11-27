"""Unit conversion tests with hand-checked values."""

import numpy as np
import pytest

from conflict_detection import units as u


def test_exact_definitions():
    assert u.nm_to_m(1) == 1852.0
    assert u.ft_to_m(1) == pytest.approx(0.3048)
    assert u.kt_to_ms(1) == pytest.approx(1852 / 3600)
    assert u.fpm_to_ms(1) == pytest.approx(0.3048 / 60)


def test_separation_minima_in_si():
    assert u.nm_to_m(5) == pytest.approx(9260.0)
    assert u.ft_to_m(1000) == pytest.approx(304.8)


@pytest.mark.parametrize(
    ("forward", "inverse", "value"),
    [
        (u.nm_to_m, u.m_to_nm, 4.99),
        (u.ft_to_m, u.m_to_ft, 37000.0),
        (u.kt_to_ms, u.ms_to_kt, 450.0),
        (u.fpm_to_ms, u.ms_to_fpm, -1500.0),
    ],
)
def test_round_trips(forward, inverse, value):
    assert inverse(forward(value)) == pytest.approx(value)


def test_typical_cruise_speed():
    # 480 kt is about 246.9 m/s and covers 8 NM per minute.
    assert u.kt_to_ms(480) == pytest.approx(246.93, abs=0.01)
    assert u.m_to_nm(u.kt_to_ms(480) * 60) == pytest.approx(8.0)


def test_vectorised():
    out = u.m_to_nm(np.array([1852.0, 3704.0]))
    np.testing.assert_allclose(out, [1.0, 2.0])


def test_isa_reference_points():
    t, p, rho = u.isa(0.0)
    assert t == pytest.approx(288.15)
    assert p == pytest.approx(101325.0)
    assert rho == pytest.approx(1.225, abs=1e-3)
    t, p, _ = u.isa(11000.0)
    assert t == pytest.approx(216.65)
    assert p == pytest.approx(22632, rel=1e-3)
    # Isothermal layer above the tropopause.
    t, p, _ = u.isa(15000.0)
    assert t == pytest.approx(216.65)
    assert p == pytest.approx(12045, rel=2e-3)


def test_cas_equals_tas_at_sea_level():
    assert u.tas_to_cas(150.0, 0.0) == pytest.approx(150.0, rel=1e-9)


def test_cas_at_cruise_level():
    # 480 kt TAS at FL370 (ISA) is Mach 0.836, i.e. about 273 kt CAS.
    # Cross-checked against bluesky.tools.aero.vtas2cas (273.04 kt).
    cas = u.ms_to_kt(u.tas_to_cas(u.kt_to_ms(480), u.ft_to_m(37000)))
    assert cas == pytest.approx(273.04, abs=0.1)


def test_cas_tas_round_trip():
    alt = u.ft_to_m(np.array([5000.0, 25000.0, 39000.0]))
    tas = np.array([150.0, 220.0, 245.0])
    np.testing.assert_allclose(u.cas_to_tas(u.tas_to_cas(tas, alt), alt), tas, rtol=1e-9)


@pytest.mark.parametrize(
    ("track", "east", "north"),
    [(0, 0, 1), (90, 1, 0), (180, 0, -1), (270, -1, 0)],
)
def test_heading_unit_vectors(track, east, north):
    e, n = u.heading_to_unit(track)
    assert e == pytest.approx(east, abs=1e-12)
    assert n == pytest.approx(north, abs=1e-12)
    assert u.unit_to_heading(east, north) == pytest.approx(track % 360)
