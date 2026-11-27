"""Geodetic <-> local Cartesian (ENU) conversions on the WGS84 ellipsoid.

Geometry is done in a local East-North-Up frame centred on the scenario. Only
the East and North components are used for horizontal geometry; the vertical
coordinate stays the reported (barometric) altitude. Using the ENU *Up* axis
would be wrong for separation: 150 km from the frame origin the Earth's
curvature drops the surface by about 1.8 km along the Up axis, which would
create huge fake vertical separations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

_A = 6378137.0  # WGS84 semi-major axis [m]
_F = 1.0 / 298.257223563  # WGS84 flattening
_E2 = _F * (2.0 - _F)  # first eccentricity squared
_B = _A * (1.0 - _F)  # semi-minor axis [m]
_EP2 = (_A**2 - _B**2) / _B**2  # second eccentricity squared


def geodetic_to_ecef(
    lat_deg: ArrayLike, lon_deg: ArrayLike, h_m: ArrayLike = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic latitude/longitude/height to Earth-centred coordinates."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    h = np.asarray(h_m, dtype=float)
    n = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - _E2) + h) * np.sin(lat)
    return x, y, z


def ecef_to_geodetic(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Earth-centred coordinates to geodetic (Bowring's method, sub-mm)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    p = np.hypot(x, y)
    theta = np.arctan2(z * _A, p * _B)
    lat = np.arctan2(z + _EP2 * _B * np.sin(theta) ** 3, p - _E2 * _A * np.cos(theta) ** 3)
    lon = np.arctan2(y, x)
    n = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


@dataclass(frozen=True)
class LocalFrame:
    """Local tangent-plane frame centred on ``(lat0, lon0)`` at height zero."""

    lat0: float
    lon0: float

    def _rotation(self) -> np.ndarray:
        lat = np.radians(self.lat0)
        lon = np.radians(self.lon0)
        sl, cl = np.sin(lat), np.cos(lat)
        so, co = np.sin(lon), np.cos(lon)
        # Rows: East, North, Up unit vectors expressed in ECEF.
        return np.array(
            [
                [-so, co, 0.0],
                [-sl * co, -sl * so, cl],
                [cl * co, cl * so, sl],
            ]
        )

    def to_enu(
        self, lat_deg: ArrayLike, lon_deg: ArrayLike, h_m: ArrayLike = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project geodetic points to (east, north, up) metres in this frame."""
        x, y, z = geodetic_to_ecef(lat_deg, lon_deg, h_m)
        x0, y0, z0 = geodetic_to_ecef(self.lat0, self.lon0, 0.0)
        d = np.stack([np.ravel(x - x0), np.ravel(y - y0), np.ravel(z - z0)])
        e, n, u = self._rotation() @ d
        shape = np.shape(x)
        return e.reshape(shape), n.reshape(shape), u.reshape(shape)

    def to_xy(self, lat_deg: ArrayLike, lon_deg: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Project geodetic points (on the ellipsoid) to horizontal (east, north) metres."""
        e, n, _ = self.to_enu(lat_deg, lon_deg, 0.0)
        return e, n

    def to_latlon(self, east_m: ArrayLike, north_m: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of :meth:`to_xy`: horizontal ENU metres back to latitude/longitude.

        The Up component is chosen so that the point lies on the ellipsoid; two
        fixed-point iterations make the round trip exact to well under a metre
        at 300 km from the origin.
        """
        e = np.asarray(east_m, dtype=float)
        n = np.asarray(north_m, dtype=float)
        u = -(e**2 + n**2) / (2.0 * _A)  # first guess: drop of a sphere
        rot_t = self._rotation().T
        x0, y0, z0 = geodetic_to_ecef(self.lat0, self.lon0, 0.0)
        for _ in range(3):
            d = rot_t @ np.stack([np.ravel(e), np.ravel(n), np.ravel(u)])
            lat, lon, h = ecef_to_geodetic(d[0] + x0, d[1] + y0, d[2] + z0)
            # Move the point vertically by -h along the reference Up axis.
            u = np.ravel(u) - h
        lat = np.reshape(lat, np.shape(e))
        lon = np.reshape(lon, np.shape(e))
        return lat, lon


def haversine_m(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> np.ndarray:
    """Great-circle distance on a mean-radius sphere (used for sanity checks)."""
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
