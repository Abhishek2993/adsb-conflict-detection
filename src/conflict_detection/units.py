"""Unit conversions and ISA airspeed helpers.

The pipeline works internally in SI units (metres, seconds, metres per second).
Aviation units only appear at the edges: ingest (ADS-B reports feet, knots and
feet per minute), the separation minima (nautical miles and feet) and the
reports. Keeping every conversion in this one module makes it easy to test and
hard to get wrong.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

#: One international nautical mile in metres (exact by definition).
NM = 1852.0
#: One international foot in metres (exact by definition).
FT = 0.3048
#: One knot (NM per hour) in metres per second.
KT = NM / 3600.0
#: One foot per minute in metres per second.
FPM = FT / 60.0

# International Standard Atmosphere constants.
_G0 = 9.80665  # gravitational acceleration [m/s^2]
_R = 287.05287  # specific gas constant of dry air [J/(kg K)]
_GAMMA = 1.4  # ratio of specific heats for air
_T0 = 288.15  # sea-level temperature [K]
_P0 = 101325.0  # sea-level pressure [Pa]
_RHO0 = _P0 / (_R * _T0)  # sea-level density [kg/m^3]
_LAPSE = 0.0065  # tropospheric temperature lapse rate [K/m]
_H_TROP = 11000.0  # tropopause altitude [m]


def nm_to_m(value: ArrayLike) -> np.ndarray | float:
    """Convert nautical miles to metres."""
    return np.asarray(value, dtype=float) * NM


def m_to_nm(value: ArrayLike) -> np.ndarray | float:
    """Convert metres to nautical miles."""
    return np.asarray(value, dtype=float) / NM


def ft_to_m(value: ArrayLike) -> np.ndarray | float:
    """Convert feet to metres."""
    return np.asarray(value, dtype=float) * FT


def m_to_ft(value: ArrayLike) -> np.ndarray | float:
    """Convert metres to feet."""
    return np.asarray(value, dtype=float) / FT


def kt_to_ms(value: ArrayLike) -> np.ndarray | float:
    """Convert knots to metres per second."""
    return np.asarray(value, dtype=float) * KT


def ms_to_kt(value: ArrayLike) -> np.ndarray | float:
    """Convert metres per second to knots."""
    return np.asarray(value, dtype=float) / KT


def fpm_to_ms(value: ArrayLike) -> np.ndarray | float:
    """Convert feet per minute to metres per second."""
    return np.asarray(value, dtype=float) * FPM


def ms_to_fpm(value: ArrayLike) -> np.ndarray | float:
    """Convert metres per second to feet per minute."""
    return np.asarray(value, dtype=float) / FPM


def isa(alt_m: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ISA temperature [K], pressure [Pa] and density [kg/m^3].

    Valid from sea level to 20 km (troposphere and lower stratosphere), which
    covers every civil flight level.
    """
    h = np.asarray(alt_m, dtype=float)
    t_trop = _T0 - _LAPSE * np.minimum(h, _H_TROP)
    p_trop = _P0 * (t_trop / _T0) ** (_G0 / (_LAPSE * _R))
    temperature = t_trop
    pressure = np.where(
        h <= _H_TROP,
        p_trop,
        p_trop * np.exp(-_G0 * (h - _H_TROP) / (_R * t_trop)),
    )
    density = pressure / (_R * temperature)
    return temperature, pressure, density


def tas_to_cas(tas_ms: ArrayLike, alt_m: ArrayLike) -> np.ndarray:
    """Convert true airspeed to calibrated airspeed (compressible, ISA, no wind).

    BlueSky's ``CRE`` and ``SPD`` commands take CAS, while ADS-B reports ground
    speed. With no wind information we assume TAS = ground speed and convert.
    """
    tas = np.asarray(tas_ms, dtype=float)
    _, pressure, density = isa(alt_m)
    k = _GAMMA / (_GAMMA - 1.0)
    # Impact pressure from TAS at altitude (isentropic flow).
    qc = pressure * ((1.0 + density * tas**2 / (2.0 * k * pressure)) ** k - 1.0)
    return np.sqrt(2.0 * k * _P0 / _RHO0 * ((qc / _P0 + 1.0) ** (1.0 / k) - 1.0))


def cas_to_tas(cas_ms: ArrayLike, alt_m: ArrayLike) -> np.ndarray:
    """Convert calibrated airspeed to true airspeed (inverse of :func:`tas_to_cas`)."""
    cas = np.asarray(cas_ms, dtype=float)
    _, pressure, density = isa(alt_m)
    k = _GAMMA / (_GAMMA - 1.0)
    qc = _P0 * ((1.0 + _RHO0 * cas**2 / (2.0 * k * _P0)) ** k - 1.0)
    return np.sqrt(2.0 * k * pressure / density * ((qc / pressure + 1.0) ** (1.0 / k) - 1.0))


def heading_to_unit(track_deg: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return the (east, north) unit vector of a true track/heading in degrees."""
    rad = np.radians(np.asarray(track_deg, dtype=float))
    return np.sin(rad), np.cos(rad)


def unit_to_heading(east: ArrayLike, north: ArrayLike) -> np.ndarray:
    """Return the true track in degrees [0, 360) of an (east, north) vector."""
    return np.mod(np.degrees(np.arctan2(east, north)), 360.0)
