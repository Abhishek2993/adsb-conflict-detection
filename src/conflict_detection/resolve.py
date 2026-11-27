"""Simple resolution manoeuvres, checked in the prediction model.

For every conflict event the resolver takes the predicted states at the first
alert and tries, for each of the two aircraft, a ladder of manoeuvres:

==========  ==============  ====
manoeuvre   values          cost
==========  ==============  ====
speed       +/- 10 %        1
heading     +/- 15 deg      2
level       +/- 1000 ft     3
heading     +/- 30 deg      4
==========  ==============  ====

A manoeuvre is accepted if, over the look-ahead horizon, (1) the pair never
loses separation and (2) the manoeuvring aircraft does not create a new loss
of separation with any third aircraft. The cheapest accepted manoeuvre wins;
ties go to the larger separation margin. The cost order encodes "smallest
deviation from the current plan": a speed change keeps the route and level, a
small turn keeps the level, a level change affects other traffic layers, and
a 30 deg turn is the largest lateral deviation.

The manoeuvre itself is flown with simple kinematics instead of jumping to the
new state: turns at the rate given by a 25 deg bank (``g tan(phi) / V``, about
1.1 deg/s at cruise speed, as in BlueSky), speed changes at about 1 kt/s and
level changes at 1500 ft/min. The other aircraft keeps its straight line (we
have no intent information).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .cpa import in_los
from .units import FPM, FT, KT, NM, ms_to_kt, unit_to_heading

G0 = 9.80665


@dataclass(frozen=True)
class Manoeuvre:
    """One candidate resolution for one aircraft."""

    kind: str  # "none", "speed", "heading", "level"
    value: float  # fraction for speed, degrees for heading, feet for level
    cost: float

    @property
    def label(self) -> str:
        """Human-readable label, e.g. ``heading +15 deg``."""
        if self.kind == "none":
            return "none"
        unit = {"speed": "%", "heading": "deg", "level": "ft"}[self.kind]
        val = self.value * 100 if self.kind == "speed" else self.value
        return f"{self.kind} {val:+.0f} {unit}"


def manoeuvre_ladder(cfg: Config | None = None) -> list[Manoeuvre]:
    """All candidate manoeuvres ordered by cost."""
    rc = (cfg or Config()).resolution
    ladder: list[Manoeuvre] = []
    for frac in rc.speed_changes_frac:
        ladder += [Manoeuvre("speed", s * frac, 1.0) for s in (+1, -1)]
    small, *large = sorted(rc.heading_changes_deg)
    ladder += [Manoeuvre("heading", s * small, 2.0) for s in (+1, -1)]
    for dz in rc.altitude_changes_ft:
        ladder += [Manoeuvre("level", s * dz, 3.0) for s in (+1, -1)]
    for k, h in enumerate(large):
        ladder += [Manoeuvre("heading", s * h, 4.0 + k) for s in (+1, -1)]
    return sorted(ladder, key=lambda m: m.cost)


def fly(
    state: np.ndarray, man: Manoeuvre, tau: np.ndarray, cfg: Config | None = None
) -> np.ndarray:
    """Positions of one aircraft flying ``man`` from ``state`` at times ``tau``.

    Args:
        state: ``[x, y, z, vx, vy, vz]`` at ``tau = 0`` (SI units).
        man: the manoeuvre (``Manoeuvre("none", 0, 0)`` for straight flight).
        tau: increasing times from 0 [s].

    Returns:
        ``(len(tau), 3)`` positions.
    """
    rc = (cfg or Config()).resolution
    x0, y0, z0, vx, vy, vz = (float(v) for v in state)
    speed0 = float(np.hypot(vx, vy))
    hdg0 = float(unit_to_heading(vx, vy))

    hdg = np.full_like(tau, hdg0)
    speed = np.full_like(tau, speed0)
    z = z0 + vz * tau
    if man.kind == "heading" and speed0 > 1.0:
        rate = np.degrees(G0 * np.tan(np.radians(rc.bank_angle_deg)) / speed0)  # deg/s
        hdg = hdg0 + np.sign(man.value) * np.minimum(rate * tau, abs(man.value))
    elif man.kind == "speed":
        target = speed0 * (1.0 + man.value)
        accel = rc.accel_kt_s * KT
        speed = speed0 + np.sign(man.value) * np.minimum(accel * tau, abs(target - speed0))
    elif man.kind == "level":
        w = rc.climb_rate_fpm * FPM
        dz = man.value * FT
        z = z0 + np.sign(dz) * np.minimum(w * tau, abs(dz))

    rad = np.radians(hdg)
    vxs, vys = speed * np.sin(rad), speed * np.cos(rad)
    dt = np.diff(tau, prepend=tau[0])
    x = x0 + np.cumsum(0.5 * (vxs + np.r_[vxs[0], vxs[:-1]]) * dt)
    y = y0 + np.cumsum(0.5 * (vys + np.r_[vys[0], vys[:-1]]) * dt)
    return np.column_stack([x, y, z])


def _separation(
    p: np.ndarray, q: np.ndarray, cfg: Config, h_buffer: float = 1.0
) -> tuple[bool, float, float, float]:
    """LoS flag, normalised margin and (hdist, vsep) at the worst instant.

    ``h_buffer`` scales the horizontal minimum only: with 1.1 the pair must
    stay 5.5 NM apart unless it is vertically separated. The vertical minimum
    is not buffered because a one-level change ends exactly 1000 ft apart,
    which is the standard (and precisely flown) vertical resolution.
    """
    hd = np.hypot(p[..., 0] - q[..., 0], p[..., 1] - q[..., 1])
    vd = np.abs(p[..., 2] - q[..., 2])
    d_min, h_min = cfg.separation.horizontal_m, cfg.separation.vertical_m
    los = bool(in_los(hd, vd, d_min * h_buffer, h_min).any())
    # Separated if EITHER ratio >= 1; the margin is the worst instant of the larger ratio.
    ratio = np.maximum(hd / d_min, vd / h_min)
    k = np.unravel_index(np.argmin(ratio), ratio.shape)
    return los, float(ratio[k]), float(hd[k]), float(vd[k])


@dataclass
class Resolution:
    """Chosen resolution for one conflict event."""

    event_id: str
    t_eval: float
    icao_a: str
    icao_b: str
    manoeuvring: str | None
    callsign: str | None
    manoeuvre: str
    kind: str
    value: float
    cost: float
    resolved: bool
    options_tried: int
    margin_before: float
    hdist_before_nm: float
    vsep_before_ft: float
    margin_after: float
    hdist_after_nm: float
    vsep_after_ft: float
    new_heading_deg: float
    new_speed_kt: float
    new_altitude_ft: float


def resolve_event(
    event: pd.Series,
    states_at_t: pd.DataFrame,
    cfg: Config | None = None,
) -> Resolution:
    """Pick the smallest manoeuvre that removes the predicted LoS of ``event``.

    Args:
        event: one row of :func:`detect.group_events`.
        states_at_t: predicted states of *all* aircraft at ``event.t_first``.
    """
    cfg = cfg or Config()
    tau = np.arange(0.0, cfg.horizon_s + 1e-9, cfg.resolution.check_step_s)
    st = states_at_t.set_index("icao24")
    cols = ["x", "y", "z", "vx", "vy", "vz"]
    sa, sb = st.loc[event.icao_a, cols].to_numpy(float), st.loc[event.icao_b, cols].to_numpy(float)
    none = Manoeuvre("none", 0.0, 0.0)
    pa, pb = fly(sa, none, tau, cfg), fly(sb, none, tau, cfg)
    _, m0, h0, v0 = _separation(pa, pb, cfg)

    others = st.drop(index=[event.icao_a, event.icao_b])
    if len(others):
        o = others[cols].to_numpy(float)
        other_paths = o[:, None, :3] + o[:, None, 3:] * tau[None, :, None]
    else:
        other_paths = np.empty((0, len(tau), 3))

    best: tuple[float, float, str, Manoeuvre, float, float] | None = None
    tried = 0
    ladder = manoeuvre_ladder(cfg)
    for man in ladder:
        if best is not None and man.cost > best[0]:
            break  # a cheaper manoeuvre already works
        for who, own, other_path in ((event.icao_a, sa, pb), (event.icao_b, sb, pa)):
            tried += 1
            path = fly(own, man, tau, cfg)
            los, margin, hd, vd = _separation(path, other_path, cfg, cfg.resolution.required_margin)
            if los:
                continue
            if len(other_paths):
                sec_los, *_ = _separation(path[None], other_paths, cfg)
                if sec_los:
                    continue
            if best is None or man.cost < best[0] or margin > best[1]:
                best = (man.cost, margin, who, man, hd, vd)

    row = dict(
        event_id=event.event_id,
        t_eval=float(event.t_first),
        icao_a=event.icao_a,
        icao_b=event.icao_b,
        options_tried=tried,
        margin_before=m0,
        hdist_before_nm=h0 / NM,
        vsep_before_ft=v0 / FT,
    )
    if best is None:
        return Resolution(
            **row,
            manoeuvring=None,
            callsign=None,
            manoeuvre="none found",
            kind="none",
            value=0.0,
            cost=np.inf,
            resolved=False,
            margin_after=m0,
            hdist_after_nm=h0 / NM,
            vsep_after_ft=v0 / FT,
            new_heading_deg=np.nan,
            new_speed_kt=np.nan,
            new_altitude_ft=np.nan,
        )
    cost, margin, who, man, hd, vd = best
    own = sa if who == event.icao_a else sb
    hdg0 = float(unit_to_heading(own[3], own[4]))
    spd0 = float(ms_to_kt(np.hypot(own[3], own[4])))
    new_hdg = (hdg0 + man.value) % 360 if man.kind == "heading" else hdg0
    new_spd = spd0 * (1 + man.value) if man.kind == "speed" else spd0
    new_alt = own[2] / FT + (man.value if man.kind == "level" else 0.0)
    callsign = str(states_at_t.set_index("icao24").loc[who, "callsign"])
    return Resolution(
        **row,
        manoeuvring=who,
        callsign=callsign,
        manoeuvre=man.label,
        kind=man.kind,
        value=man.value,
        cost=cost,
        resolved=True,
        margin_after=margin,
        hdist_after_nm=hd / NM,
        vsep_after_ft=vd / FT,
        new_heading_deg=new_hdg,
        new_speed_kt=new_spd,
        new_altitude_ft=new_alt,
    )


def resolve_all(
    events: pd.DataFrame, states: pd.DataFrame, cfg: Config | None = None
) -> pd.DataFrame:
    """Resolve every conflict event; returns one row per event."""
    cfg = cfg or Config()
    out = []
    by_time = dict(tuple(states.groupby("t_eval")))
    for _, ev in events.iterrows():
        out.append(resolve_event(ev, by_time[ev.t_first], cfg).__dict__)
    return pd.DataFrame(out)
