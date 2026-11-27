"""Closest point of approach (CPA) geometry and the spatial pair pre-filter.

For two aircraft flying straight lines at constant velocity, the relative
position is ``r(t) = r0 + v t`` with ``r0 = pB - pA`` and ``v = vB - vA``.
The squared horizontal distance ``|r(t)|^2`` is a parabola in ``t`` whose
minimum is at

    tCPA = -(r0 . v) / |v|^2

and the miss distance is ``dCPA = |r0 + v tCPA|``. The vertical separation at
that instant is ``|dz0 + dvz tCPA|``. Everything in this module is vectorised
over arrays of pairs; units are SI (m, s, m/s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

_EPS_V2 = 1e-6  # (m/s)^2: below this, the relative velocity is treated as zero
#: Comparison tolerance [m]. Flight levels 1000 ft apart differ by 304.8 m, but
#: in floating point 38000 ft - 37000 ft evaluates to 304.8000000000011 m and
#: other pairs can land just below 304.8. One micrometre is far below any
#: physical meaning and makes "exactly at the minimum" count as separated.
EPS_M = 1e-6


def in_los(h_dist: ArrayLike, v_dist: ArrayLike, d_min: float, h_min: float) -> np.ndarray:
    """True where BOTH minima are violated: ``h_dist < d_min`` and ``|v_dist| < h_min``.

    This is the single definition of a loss of separation used everywhere
    (actual LoS on recorded tracks, predicted conflicts, resolution checks).
    """
    h = np.asarray(h_dist, dtype=float)
    v = np.abs(np.asarray(v_dist, dtype=float))
    return (h < d_min - EPS_M) & (v < h_min - EPS_M)


@dataclass(frozen=True)
class CPAResult:
    """Closest-point-of-approach quantities for one or many aircraft pairs."""

    tcpa: np.ndarray  # time to CPA [s]; may be negative (diverging)
    dcpa: np.ndarray  # horizontal distance at CPA [m]
    dz_cpa: np.ndarray  # signed vertical offset B - A at CPA [m]
    d_now: np.ndarray  # current horizontal distance [m]
    dz_now: np.ndarray  # current signed vertical offset [m]

    @property
    def vsep_cpa(self) -> np.ndarray:
        """Absolute vertical separation at CPA [m]."""
        return np.abs(self.dz_cpa)


def compute_cpa(
    dx: ArrayLike,
    dy: ArrayLike,
    dz: ArrayLike,
    dvx: ArrayLike,
    dvy: ArrayLike,
    dvz: ArrayLike,
) -> CPAResult:
    """Analytic horizontal CPA of the relative motion ``B - A``.

    Args:
        dx, dy, dz: relative position of B with respect to A [m].
        dvx, dvy, dvz: relative velocity of B with respect to A [m/s].

    Returns:
        :class:`CPAResult`. When the relative horizontal velocity is (almost)
        zero the distance never changes, and ``tcpa`` is defined as 0.
    """
    dx, dy, dz = (np.asarray(a, dtype=float) for a in (dx, dy, dz))
    dvx, dvy, dvz = (np.asarray(a, dtype=float) for a in (dvx, dvy, dvz))
    v2 = dvx**2 + dvy**2
    moving = v2 > _EPS_V2
    tcpa = np.where(moving, -(dx * dvx + dy * dvy) / np.where(moving, v2, 1.0), 0.0)
    cx = dx + dvx * tcpa
    cy = dy + dvy * tcpa
    return CPAResult(
        tcpa=tcpa,
        dcpa=np.hypot(cx, cy),
        dz_cpa=dz + dvz * tcpa,
        d_now=np.hypot(dx, dy),
        dz_now=dz,
    )


def los_interval(
    dx: ArrayLike,
    dy: ArrayLike,
    dz: ArrayLike,
    dvx: ArrayLike,
    dvy: ArrayLike,
    dvz: ArrayLike,
    d_min: float,
    h_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Time interval during which both minima are violated (unbounded in time).

    The horizontal intrusion is the set of ``t`` with ``|r0 + v t| < d_min``
    (a quadratic inequality); the vertical intrusion is the set of ``t`` with
    ``|dz + dvz t| < h_min`` (a linear one). A loss of separation needs both,
    so the result is their intersection ``[t_in, t_out]``. Empty intervals are
    returned as ``t_in = +inf, t_out = -inf``.
    """
    dx, dy, dz = (np.asarray(a, dtype=float) for a in (dx, dy, dz))
    dvx, dvy, dvz = (np.asarray(a, dtype=float) for a in (dvx, dvy, dvz))
    shape = np.broadcast(dx, dy, dz, dvx, dvy, dvz).shape
    inf = np.full(shape, np.inf)
    d_min = d_min - EPS_M
    h_min = h_min - EPS_M

    # Horizontal: a t^2 + b t + c < 0
    a = dvx**2 + dvy**2
    b = 2.0 * (dx * dvx + dy * dvy)
    c = dx**2 + dy**2 - d_min**2
    moving = a > _EPS_V2
    disc = b**2 - 4.0 * a * c
    sq = np.sqrt(np.where(moving & (disc > 0), disc, 0.0))
    a_safe = np.where(moving, a, 1.0)
    root_in = np.where(disc > 0, (-b - sq) / (2 * a_safe), inf)
    root_out = np.where(disc > 0, (-b + sq) / (2 * a_safe), -inf)
    h_in = np.where(moving, root_in, np.where(c < 0, -inf, inf))
    h_out = np.where(moving, root_out, np.where(c < 0, inf, -inf))

    # Vertical: |dz + dvz t| < h_min
    climbing = np.abs(dvz) > 1e-9
    dvz_safe = np.where(climbing, dvz, 1.0)
    t1 = (-h_min - dz) / dvz_safe
    t2 = (h_min - dz) / dvz_safe
    v_in = np.where(climbing, np.minimum(t1, t2), np.where(np.abs(dz) < h_min, -inf, inf))
    v_out = np.where(climbing, np.maximum(t1, t2), np.where(np.abs(dz) < h_min, inf, -inf))

    t_in = np.maximum(h_in, v_in)
    t_out = np.minimum(h_out, v_out)
    empty = t_in >= t_out
    return np.where(empty, np.inf, t_in), np.where(empty, -np.inf, t_out)


def candidate_pairs(
    pos: ArrayLike,
    vel: ArrayLike,
    horizon_s: float,
    d_min: float,
    h_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs ``(i, j), i < j`` that *could* lose separation within the horizon.

    Each aircraft sweeps a straight segment of length ``|v| T`` over the
    horizon ``T``. Two segments can only come within ``d_min`` of each other if
    their midpoints are closer than ``h_i + h_j + d_min`` (``h`` = half segment
    length), by the triangle inequality. A KD-tree on the midpoints returns all
    pairs within ``2 max(h) + d_min``; the exact per-pair bound and an altitude
    band overlap test then prune the rest. The filter never drops a pair that
    the CPA test would flag inside ``[0, T]``, so it is lossless.

    Args:
        pos: ``(n, 3)`` positions [m] (east, north, altitude).
        vel: ``(n, 3)`` velocities [m/s].
        horizon_s: look-ahead horizon [s].
        d_min, h_min: horizontal and vertical minima [m].
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    vel = np.asarray(vel, dtype=float).reshape(-1, 3)
    n = len(pos)
    if n < 2:
        empty = np.empty(0, dtype=int)
        return empty, empty
    mid = pos[:, :2] + 0.5 * horizon_s * vel[:, :2]
    half = 0.5 * horizon_s * np.hypot(vel[:, 0], vel[:, 1])
    tree = cKDTree(mid)
    pairs = tree.query_pairs(2.0 * half.max() + d_min, output_type="ndarray")
    if len(pairs) == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty
    i, j = pairs[:, 0], pairs[:, 1]
    sep = np.hypot(*(mid[i] - mid[j]).T)
    keep = sep <= half[i] + half[j] + d_min
    # Altitude bands swept over the horizon must come within h_min.
    z_end = pos[:, 2] + horizon_s * vel[:, 2]
    z_lo = np.minimum(pos[:, 2], z_end)
    z_hi = np.maximum(pos[:, 2], z_end)
    gap = np.maximum(z_lo[i], z_lo[j]) - np.minimum(z_hi[i], z_hi[j])
    keep &= gap < h_min
    i, j = i[keep], j[keep]
    order = np.lexsort((j, i))
    return i[order], j[order]


def all_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Every pair ``(i, j), i < j`` of ``n`` aircraft (the naive O(n^2) search)."""
    i, j = np.triu_indices(n, k=1)
    return i.astype(int), j.astype(int)
