"""Local frame conversions."""

import numpy as np
import pytest

from conflict_detection.geo import LocalFrame, ecef_to_geodetic, geodetic_to_ecef, haversine_m
from conflict_detection.units import NM


def test_ecef_round_trip():
    lat, lon, h = np.array([46.9, -33.9, 0.0]), np.array([7.8, 151.2, 0.0]), np.array([11000, 0, 5])
    out = ecef_to_geodetic(*geodetic_to_ecef(lat, lon, h))
    np.testing.assert_allclose(out[0], lat, atol=1e-9)
    np.testing.assert_allclose(out[1], lon, atol=1e-9)
    np.testing.assert_allclose(out[2], h, atol=1e-3)


def test_origin_maps_to_zero():
    f = LocalFrame(46.9, 7.8)
    x, y = f.to_xy(46.9, 7.8)
    assert x == pytest.approx(0, abs=1e-6) and y == pytest.approx(0, abs=1e-6)


def test_axes_point_east_and_north():
    f = LocalFrame(46.9, 7.8)
    x, y = f.to_xy(47.0, 7.8)
    assert abs(x) < 1 and y > 11000
    x, y = f.to_xy(46.9, 7.9)
    assert x > 7000 and abs(y) < 100


def test_inverse_round_trip_far_from_origin():
    f = LocalFrame(46.9, 7.8)
    e = np.array([0.0, 150_000.0, -200_000.0, 300_000.0])
    n = np.array([0.0, -120_000.0, 80_000.0, 300_000.0])
    lat, lon = f.to_latlon(e, n)
    e2, n2 = f.to_xy(lat, lon)
    np.testing.assert_allclose(e2, e, atol=0.05)
    np.testing.assert_allclose(n2, n, atol=0.05)


def test_planar_distance_matches_ellipsoid_chord():
    # Two points ~84 NM apart, either side of the origin. The tangent-plane
    # distance must agree with the straight-line (ECEF chord) distance between
    # the points on the ellipsoid to well under 0.05 NM.
    f = LocalFrame(46.9, 7.8)
    lat = np.array([46.5, 47.6])
    lon = np.array([6.5, 9.9])
    x, y = f.to_xy(lat, lon)
    planar = np.hypot(x[1] - x[0], y[1] - y[0])
    ex, ey, ez = geodetic_to_ecef(lat, lon, 0.0)
    chord = np.sqrt((ex[1] - ex[0]) ** 2 + (ey[1] - ey[0]) ** 2 + (ez[1] - ez[0]) ** 2)
    assert planar / NM == pytest.approx(chord / NM, abs=0.05)
    # The spherical haversine distance is within 0.5 % of the ellipsoid one.
    assert haversine_m(lat[0], lon[0], lat[1], lon[1]) == pytest.approx(chord, rel=0.005)
