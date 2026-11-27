"""Maps and plots: interactive (folium, plotly) and static (matplotlib, for the docs).

Colour roles (validated for colour-vision deficiency in light and dark mode):
aircraft A = blue, aircraft B = orange, BlueSky = aqua, background traffic and
thresholds = neutral grey, loss of separation = status red (always with a
text label, never colour alone).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .resolve import Manoeuvre, fly
from .units import FT, NM

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import PipelineResult

BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
CRITICAL = "#dc3c3d"
INK, INK2 = "#191919", "#666666"
GRID, AXIS, MUTED = "#e7e7e7", "#c2c2c2", "#b5b5b5"


# --------------------------------------------------------------------- helpers
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "font.size": 9,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    return plt


def showcase_event(res: PipelineResult) -> pd.Series | None:
    """The most severe predicted event that a real LoS confirmed (else the most severe)."""
    ev = res.events
    if ev.empty:
        return None
    confirmed = ev[ev.confirmed.astype(bool)] if "confirmed" in ev else ev
    pool = confirmed if len(confirmed) else ev
    return pool.sort_values(["severity_max", "t_first"], ascending=[False, True]).iloc[0]


def pair_separation(grid: pd.DataFrame, icao_a: str, icao_b: str) -> pd.DataFrame:
    """Horizontal [NM] and vertical [ft] separation of a pair on the common grid."""
    a = grid[grid.icao24 == icao_a].set_index("t")[["x", "y", "z"]]
    b = grid[grid.icao24 == icao_b].set_index("t")[["x", "y", "z"]]
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    return pd.DataFrame(
        {
            "t": j.index.to_numpy(),
            "h_nm": np.hypot(j.x_a - j.x_b, j.y_a - j.y_b).to_numpy() / NM,
            "v_ft": np.abs(j.z_a - j.z_b).to_numpy() / FT,
        }
    )


def _callsign(res: PipelineResult, icao: str) -> str:
    cs = res.grid.loc[res.grid.icao24 == icao, "callsign"]
    name = str(cs.iloc[-1]) if len(cs) else ""
    return name or icao


# ------------------------------------------------------------- folium map (HTML)
def trajectory_map(res: PipelineResult, path: str | Path, max_pairs: int = 15) -> Path:
    """Interactive map: all trajectories, conflict pairs, predicted CPA points, actual LoS."""
    import folium

    grid = res.grid
    m = folium.Map(location=[res.frame.lat0, res.frame.lon0], zoom_start=8, tiles="OpenStreetMap")
    traffic = folium.FeatureGroup("All traffic")
    for icao, g in grid.groupby("icao24"):
        folium.PolyLine(
            g[["latitude", "longitude"]].to_numpy().tolist(),
            color=MUTED,
            weight=1.5,
            opacity=0.8,
            tooltip=f"{_callsign(res, icao)} ({icao})",
        ).add_to(traffic)
    traffic.add_to(m)

    ev = res.events.sort_values("severity_max", ascending=False)
    table = None
    from .pipeline import conflict_table

    table = conflict_table(res).set_index("event_id") if len(ev) else None
    pairs = folium.FeatureGroup(f"Conflict pairs (top {max_pairs} by severity)")
    cpas = folium.FeatureGroup("Predicted CPA at first alert")
    for k, (_, e) in enumerate(ev.iterrows()):
        row = table.loc[e.event_id]
        confirmed = bool(e.get("confirmed", False))
        outcome = "confirmed by an actual LoS" if confirmed else "false alarm (no LoS followed)"
        html = (
            f"<b>{e.event_id}</b> {e.callsign_a or e.icao_a} / {e.callsign_b or e.icao_b}<br>"
            f"first alert {row.t_first_utc:%H:%M:%S} UTC<br>"
            f"tCPA {e.tcpa_first_s:.0f} s, dCPA {e.dcpa_first_nm:.2f} NM, "
            f"&Delta;alt {e.vsep_first_ft:.0f} ft<br>severity {e.severity_max:.2f} &middot; "
            f"{outcome}<br>resolution: {row.get('manoeuvre', 'n/a')}"
        )
        folium.CircleMarker(
            [row.cpa_lat, row.cpa_lon],
            radius=6,
            color="white",
            weight=2,
            fill=True,
            fill_color=CRITICAL if confirmed else INK2,
            fill_opacity=1.0,
            popup=folium.Popup(html, max_width=320),
            tooltip=f"{e.event_id}: {'confirmed' if confirmed else 'false alarm'}",
        ).add_to(cpas)
        if k < max_pairs:
            t0, t1 = e.t_first - 120, e.t_first + res.cfg.horizon_s
            for icao, colour in ((e.icao_a, BLUE), (e.icao_b, ORANGE)):
                g = grid[(grid.icao24 == icao) & grid.t.between(t0, t1)]
                folium.PolyLine(
                    g[["latitude", "longitude"]].to_numpy().tolist(),
                    color=colour,
                    weight=3,
                    tooltip=f"{e.event_id}: {_callsign(res, icao)}",
                ).add_to(pairs)
    pairs.add_to(m)
    cpas.add_to(m)

    los = folium.FeatureGroup("Actual loss of separation (start)")
    for _, a in res.actual.iterrows():
        g = grid[grid.icao24.isin([a.icao_a, a.icao_b]) & (grid.t == a.t_start)]
        if len(g) == 2:
            lat, lon = g.latitude.mean(), g.longitude.mean()
            folium.CircleMarker(
                [lat, lon],
                radius=10,
                color=CRITICAL,
                weight=2,
                fill=False,
                tooltip=(
                    f"{a.los_id} actual LoS {_callsign(res, a.icao_a)} / "
                    f"{_callsign(res, a.icao_b)}: min {a.min_hdist_nm:.2f} NM, "
                    f"{a.min_vsep_ft:.0f} ft"
                ),
            ).add_to(los)
    los.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
         padding: 8px 10px; border: 1px solid {AXIS}; font: 12px sans-serif; color: {INK};">
      <b>{res.scenario.name}</b><br>
      <span style="color:{MUTED}">&#9473;</span> traffic &nbsp;
      <span style="color:{BLUE}">&#9473;</span><span style="color:{ORANGE}">&#9473;</span>
      conflict pair<br>
      <span style="color:{CRITICAL}">&#9679;</span> predicted CPA, confirmed &nbsp;
      <span style="color:{INK2}">&#9679;</span> false alarm<br>
      <span style="color:{CRITICAL}">&#9711;</span> actual LoS (5 NM / 1000 ft)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    path = Path(path)
    m.save(str(path))
    return path


# ------------------------------------------------------- separation time series
def separation_figure(res: PipelineResult, icao_a: str, icao_b: str):
    """Plotly figure: horizontal and vertical separation of one pair over time."""
    from plotly.subplots import make_subplots

    sep = pair_separation(res.grid, icao_a, icao_b)
    when = res.to_utc(sep.t)
    det = res.detections[(res.detections.icao_a == icao_a) & (res.detections.icao_b == icao_b)]
    title = f"{_callsign(res, icao_a)} / {_callsign(res, icao_b)}: separation"
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Horizontal separation [NM]", "Vertical separation [ft]"),
    )
    fig.add_scatter(
        x=when,
        y=sep.h_nm,
        name="horizontal (recorded)",
        line={"color": BLUE, "width": 2},
        hovertemplate="%{y:.2f} NM<extra></extra>",
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=when,
        y=sep.v_ft,
        name="vertical (recorded)",
        line={"color": ORANGE, "width": 2},
        hovertemplate="%{y:.0f} ft<extra></extra>",
        row=2,
        col=1,
    )
    if len(det):
        fig.add_scatter(
            x=res.to_utc(det.t_eval),
            y=det.dcpa_nm,
            mode="markers",
            name="predicted dCPA at each alert",
            marker={"color": INK2, "size": 8, "symbol": "diamond"},
            hovertemplate="alert: dCPA %{y:.2f} NM<extra></extra>",
            row=1,
            col=1,
        )
    fig.add_hline(
        y=5.0,
        line={"color": INK2, "dash": "dash", "width": 1},
        annotation_text="5 NM minimum",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=1000.0,
        line={"color": INK2, "dash": "dash", "width": 1},
        annotation_text="1000 ft minimum",
        row=2,
        col=1,
    )
    act = res.actual[(res.actual.icao_a == icao_a) & (res.actual.icao_b == icao_b)]
    for _, a in act.iterrows():
        for row in (1, 2):
            fig.add_vrect(
                x0=res.to_utc(a.t_start),
                x1=res.to_utc(a.t_end),
                fillcolor=CRITICAL,
                opacity=0.12,
                line_width=0,
                row=row,
                col=1,
            )
        fig.add_annotation(
            x=res.to_utc(a.t_start),
            y=1.02,
            yref="paper",
            text="actual LoS",
            showarrow=False,
            font={"color": CRITICAL},
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=520,
        legend={"orientation": "h", "y": -0.12},
    )
    return fig


def separation_png(res: PipelineResult, icao_a: str, icao_b: str, path: str | Path) -> Path:
    """Static version of :func:`separation_figure` for the README.

    The window runs from 3 minutes before the first alert to 3 minutes after
    the end of the loss of separation, so the approach is readable.
    """
    plt = _mpl()
    sep = pair_separation(res.grid, icao_a, icao_b)
    det = res.detections[(res.detections.icao_a == icao_a) & (res.detections.icao_b == icao_b)]
    act = res.actual[(res.actual.icao_a == icao_a) & (res.actual.icao_b == icao_b)]
    t_ref = act.t_start.min() if len(act) else det.t_eval.min()
    lo = min(det.t_eval.min() if len(det) else t_ref, t_ref) - 180
    hi = (act.t_end.max() if len(act) else det.t_eval.max()) + 180
    sep = sep[sep.t.between(lo, hi)]
    x = (sep.t - t_ref) / 60.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.4), sharex=True)
    ax1.plot(x, sep.h_nm, color=BLUE, label="recorded horizontal separation")
    if len(det):
        ax1.scatter(
            (det.t_eval - t_ref) / 60.0,
            det.dcpa_nm,
            s=16,
            color=INK2,
            zorder=3,
            label="predicted miss distance (dCPA) at each alert",
        )
    ax1.axhline(5.0, color=INK2, lw=1, ls="--")
    ax1.set_ylabel("horizontal [NM]")
    ax1.set_ylim(0, max(8.0, min(float(sep.h_nm.max()), 30.0)))
    ax1.text(x.iloc[-1], 5.25, "5 NM minimum", color=INK2, fontsize=8, ha="right")
    ax2.plot(x, sep.v_ft, color=ORANGE, label="recorded vertical separation")
    ax2.axhline(1000.0, color=INK2, lw=1, ls="--")
    ax2.set_ylim(0, max(1600.0, min(float(sep.v_ft.max()) * 1.1, 4000.0)))
    ax2.text(x.iloc[-1], 1040, "1000 ft minimum", color=INK2, fontsize=8, ha="right")
    ax2.set_ylabel("vertical [ft]")
    ax2.set_xlabel("minutes relative to the start of the loss of separation")
    for _, a in act.iterrows():
        for ax in (ax1, ax2):
            ax.axvspan(
                (a.t_start - t_ref) / 60, (a.t_end - t_ref) / 60, color=CRITICAL, alpha=0.12, lw=0
            )
        ax2.text(
            (a.t_start - t_ref) / 60,
            0.04,
            " actual LoS",
            color=CRITICAL,
            fontsize=8,
            transform=ax2.get_xaxis_transform(),
        )
    if len(det):
        first = (det.t_eval.min() - t_ref) / 60
        ax1.annotate(
            f"first alert {-first * 60:.0f} s before LoS",
            xy=(first, float(det.sort_values("t_eval").dcpa_nm.iloc[0])),
            xytext=(first - 2.5, ax1.get_ylim()[1] * 0.55),
            color=INK,
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
        )
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax1.set_title(f"{_callsign(res, icao_a)} / {_callsign(res, icao_b)}: separation over time")
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------ resolution before/after
def resolution_png(res: PipelineResult, event_id: str, path: str | Path) -> Path | None:
    """Before/after plot: prediction model with and without the manoeuvre, plus BlueSky."""
    from .bluesky_bridge import read_log

    plt = _mpl()
    ev = res.events.set_index("event_id").loc[event_id]
    r = res.resolutions.set_index("event_id").loc[event_id]
    if not bool(r.resolved):
        return None
    st = res.states[res.states.t_eval == ev.t_first].set_index("icao24")
    cols = ["x", "y", "z", "vx", "vy", "vz"]
    tau = np.arange(0.0, res.cfg.horizon_s + 1e-9, 2.0)
    none = Manoeuvre("none", 0.0, 0.0)
    man = Manoeuvre(str(r.kind), float(r.value), float(r.cost))
    paths = {}
    for icao in (ev.icao_a, ev.icao_b):
        s = st.loc[icao, cols].to_numpy(float)
        paths[(icao, "before")] = fly(s, none, tau, res.cfg)
        paths[(icao, "after")] = fly(s, man if icao == r.manoeuvring else none, tau, res.cfg)

    def sep(p, q):
        return np.hypot(*(p[:, :2] - q[:, :2]).T) / NM, np.abs(p[:, 2] - q[:, 2]) / FT

    h0, v0 = sep(paths[(ev.icao_a, "before")], paths[(ev.icao_b, "before")])
    h1, v1 = sep(paths[(ev.icao_a, "after")], paths[(ev.icao_b, "after")])

    bs_runs = {}
    if res.bluesky is not None and event_id in set(res.bluesky.event_id):
        b = res.bluesky.set_index("event_id").loc[event_id]
        for label in ("unresolved", "resolved"):
            logp = b.get(f"{label}_log")
            if isinstance(logp, str) and Path(logp).exists():
                lg = read_log(logp)
                tracks = {}
                for acid in (b.acid_a, b.acid_b):
                    g = lg[lg.acid == acid].set_index("simt")
                    x, y = res.frame.to_xy(g.lat.to_numpy(), g.lon.to_numpy())
                    tracks[acid] = pd.DataFrame(
                        {"x": x, "y": y, "z": g.alt_m.to_numpy()}, index=g.index
                    )
                ja = tracks[b.acid_a].join(tracks[b.acid_b], lsuffix="_a", rsuffix="_b")
                ja = ja[ja.index <= res.cfg.horizon_s]
                bs_runs[label] = (
                    ja.index.to_numpy(),
                    np.hypot(ja.x_a - ja.x_b, ja.y_a - ja.y_b).to_numpy() / NM,
                    np.abs(ja.z_a - ja.z_b).to_numpy() / FT,
                    tracks,
                )

    fig = plt.figure(figsize=(10, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], hspace=0.35, wspace=0.25)
    ax = fig.add_subplot(gs[:, 0])
    names = {ev.icao_a: _callsign(res, ev.icao_a), ev.icao_b: _callsign(res, ev.icao_b)}
    for icao, colour in ((ev.icao_a, BLUE), (ev.icao_b, ORANGE)):
        b0, b1 = paths[(icao, "before")] / NM, paths[(icao, "after")] / NM
        ax.plot(b0[:, 0], b0[:, 1], color=colour, ls="--", lw=1.5)
        ax.plot(b1[:, 0], b1[:, 1], color=colour, lw=2, label=names[icao])
        ax.plot(b0[0, 0], b0[0, 1], "o", color=colour, ms=7, mec="white", mew=1.5)
        ax.annotate(
            names[icao],
            (b0[0, 0], b0[0, 1]),
            xytext=(6, 6),
            textcoords="offset points",
            color=INK,
            fontsize=8,
        )
    if "resolved" in bs_runs:
        for acid in bs_runs["resolved"][3]:
            tr = bs_runs["resolved"][3][acid]
            tr = tr[tr.index <= res.cfg.horizon_s]
            ax.plot(tr.x / NM, tr.y / NM, color=AQUA, lw=1.2, ls=":")
        ax.plot([], [], color=AQUA, ls=":", lw=1.2, label="BlueSky, with resolution")
    ax.plot([], [], color=INK2, ls="--", lw=1.5, label="predicted, no resolution")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("east [NM]")
    ax.set_ylabel("north [NM]")
    ax.set_title(f"{event_id}: {names[r.manoeuvring]} {r.manoeuvre}")
    ax.legend(loc="best")

    tmin = tau / 60.0
    axh = fig.add_subplot(gs[0, 1])
    axv = fig.add_subplot(gs[1, 1], sharex=axh)
    axh.plot(tmin, h0, color=INK2, ls="--", lw=1.5, label="predicted, no resolution")
    axh.plot(tmin, h1, color=BLUE, label="predicted, with resolution")
    axv.plot(tmin, v0, color=INK2, ls="--", lw=1.5)
    axv.plot(tmin, v1, color=BLUE)
    if "unresolved" in bs_runs:
        t, h, v, _ = bs_runs["unresolved"]
        axh.plot(t / 60, h, color=ORANGE, lw=1.2, ls=":", label="BlueSky, no resolution")
        axv.plot(t / 60, v, color=ORANGE, lw=1.2, ls=":")
    if "resolved" in bs_runs:
        t, h, v, _ = bs_runs["resolved"]
        axh.plot(t / 60, h, color=AQUA, lw=1.5, label="BlueSky, with resolution")
        axv.plot(t / 60, v, color=AQUA, lw=1.5)
    axh.axhline(5, color=INK2, lw=1)
    axh.text(0, 5.2, "5 NM", color=INK2, fontsize=8)
    axv.axhline(1000, color=INK2, lw=1)
    axv.text(0, 1040, "1000 ft", color=INK2, fontsize=8)
    axh.set_ylabel("horizontal [NM]")
    axv.set_ylabel("vertical [ft]")
    axv.set_xlabel("minutes after the first alert")
    axh.set_ylim(bottom=0)
    axv.set_ylim(bottom=0)
    axh.set_title("Separation of the pair")
    axh.legend(loc="upper right", fontsize=7)
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- static map
def map_png(res: PipelineResult, path: str | Path, event_id: str | None = None) -> Path:
    """Static plan-view map (lon/lat) of the scenario with one highlighted conflict."""
    plt = _mpl()
    grid = res.grid
    ev = res.events.set_index("event_id").loc[event_id] if event_id else showcase_event(res)
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, g in grid.groupby("icao24"):
        ax.plot(g.longitude, g.latitude, color=MUTED, lw=0.8)
    table = None
    if len(res.events):
        from .pipeline import conflict_table

        table = conflict_table(res)
        fa = table[~table.confirmed.astype(bool)]
        ok = table[table.confirmed.astype(bool)]
        ax.scatter(
            fa.cpa_lon,
            fa.cpa_lat,
            s=22,
            color=INK2,
            edgecolor="white",
            lw=0.8,
            zorder=3,
            label=f"predicted CPA, false alarm ({len(fa)})",
        )
        ax.scatter(
            ok.cpa_lon,
            ok.cpa_lat,
            s=30,
            color=CRITICAL,
            edgecolor="white",
            lw=0.8,
            zorder=4,
            label=f"predicted CPA, confirmed by actual LoS ({len(ok)})",
        )
    if ev is not None:
        t0, t1 = ev.t_first - 300, ev.t_first + res.cfg.horizon_s
        for icao, colour in ((ev.icao_a, BLUE), (ev.icao_b, ORANGE)):
            g = grid[(grid.icao24 == icao) & grid.t.between(t0, t1)]
            ax.plot(g.longitude, g.latitude, color=colour, lw=2.2, zorder=5)
            s = g[g.t >= ev.t_first].iloc[0]
            ax.plot(s.longitude, s.latitude, "o", color=colour, ms=7, mec="white", zorder=6)
            ax.annotate(
                f"{_callsign(res, icao)} FL{s.altitude / 100:.0f}",
                (s.longitude, s.latitude),
                xytext=(6, -12),
                textcoords="offset points",
                fontsize=8,
                color=INK,
                zorder=7,
            )
        when = res.to_utc(ev.t_first)
        ax.set_title(
            f"{res.scenario.name}: conflict {ev.name if event_id else ev.event_id} "
            f"detected {when:%H:%M:%S} UTC, tCPA {ev.tcpa_first_s:.0f} s, "
            f"dCPA {ev.dcpa_first_nm:.1f} NM, Δalt {ev.vsep_first_ft:.0f} ft"
        )
    ax.plot([], [], color=MUTED, lw=1, label=f"traffic ({grid.icao24.nunique()} aircraft)")
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_aspect(1 / np.cos(np.radians(res.frame.lat0)))
    ax.legend(loc="lower left", fontsize=7)
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------- animated GIF
def encounter_gif(
    res: PipelineResult, event_id: str, path: str | Path, lead_s: float = 150.0
) -> Path:
    """Animated plan view of one encounter with the 5-minute look-ahead vectors."""
    plt = _mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter

    ev = res.events.set_index("event_id").loc[event_id]
    pair = (ev.icao_a, ev.icao_b)
    t_start = ev.t_first - lead_s
    los = res.actual[(res.actual.icao_a == pair[0]) & (res.actual.icao_b == pair[1])]
    t_end = (los.t_end.max() if len(los) else ev.t_predicted_cpa) + 60.0
    times = np.arange(t_start, t_end + 1e-9, res.cfg.eval_step_s)
    grid = res.grid
    win = grid[grid.t.between(t_start - 300, t_end)]
    pg = win[win.icao24.isin(pair)]
    cx, cy = pg.x.mean() / NM, pg.y.mean() / NM
    span = max(np.ptp(pg.x), np.ptp(pg.y)) / NM / 2 + 12
    det = res.detections[(res.detections.icao_a == pair[0]) & (res.detections.icao_b == pair[1])]
    det = det.set_index("t_eval")
    states = res.states.set_index(["t_eval", "icao24"])
    names = {i: _callsign(res, i) for i in pair}

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=80)

    def draw(t):
        ax.clear()
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.set_xlabel("east [NM]")
        ax.set_ylabel("north [NM]")
        now = win[(win.t <= t) & (win.t > t - 600)]
        for icao, g in now.groupby("icao24"):
            if icao in pair:
                continue
            ax.plot(g.x / NM, g.y / NM, color=MUTED, lw=0.8)
            ax.plot(g.x.iloc[-1] / NM, g.y.iloc[-1] / NM, "o", color=MUTED, ms=3)
        for icao, colour in zip(pair, (BLUE, ORANGE), strict=True):
            g = now[now.icao24 == icao]
            if g.empty:
                continue
            ax.plot(g.x / NM, g.y / NM, color=colour, lw=2)
            if (t, icao) in states.index:
                s = states.loc[(t, icao)]
                end = (s.x + s.vx * res.cfg.horizon_s, s.y + s.vy * res.cfg.horizon_s)
                ax.plot(
                    [s.x / NM, end[0] / NM], [s.y / NM, end[1] / NM], color=colour, lw=1.2, ls="--"
                )
                ax.plot(s.x / NM, s.y / NM, "o", color=colour, ms=8, mec="white", mew=1.5)
                ax.annotate(
                    f"{names[icao]}\nFL{s.z / FT / 100:.0f}",
                    (s.x / NM, s.y / NM),
                    xytext=(7, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=INK,
                )
        minutes = (t - ev.t_first) / 60
        if t in det.index:
            d = det.loc[t]
            ax.plot(
                [d.cpa_x_a / NM, d.cpa_x_b / NM],
                [d.cpa_y_a / NM, d.cpa_y_b / NM],
                "x",
                color=CRITICAL,
                ms=9,
                mew=2,
            )
            msg = (
                f"PREDICTED CONFLICT  tCPA {d.tcpa_s:.0f} s  dCPA {d.dcpa_nm:.1f} NM  "
                f"Δalt {d.vsep_cpa_ft:.0f} ft"
            )
            ax.set_title(msg, color=CRITICAL, fontsize=9)
        else:
            ax.set_title("no predicted conflict within 5 min", color=INK2, fontsize=9)
        in_los = len(los) and ((los.t_start <= t) & (los.t_end >= t)).any()
        status = "   ACTUAL LOSS OF SEPARATION" if in_los else ""
        ax.text(
            0.01,
            0.01,
            f"{res.to_utc(t):%H:%M:%S} UTC ({minutes:+.1f} min){status}",
            transform=ax.transAxes,
            fontsize=8,
            color=CRITICAL if in_los else INK2,
        )
        ax.plot([], [], color=INK2, ls="--", lw=1.2, label="5-min straight-line look-ahead")
        ax.plot([], [], "x", color=CRITICAL, ms=7, label="predicted positions at CPA")
        ax.legend(loc="upper right", fontsize=7)

    anim = FuncAnimation(fig, draw, frames=times, interval=150)
    path = Path(path)
    anim.save(str(path), writer=PillowWriter(fps=6))
    plt.close(fig)
    return path


# --------------------------------------------------------------- docs figures
def prediction_error_png(errors: pd.DataFrame, path: str | Path) -> Path:
    """95th-percentile look-ahead error per scenario and predictor (small multiples)."""
    plt = _mpl()
    scen = list(dict.fromkeys(errors.scenario))
    fig, axes = plt.subplots(2, len(scen), figsize=(4.2 * len(scen), 5.2), squeeze=False)
    for c, name in enumerate(scen):
        for predictor, colour, label in (
            ("cv", ORANGE, "constant velocity (raw state)"),
            ("kalman", BLUE, "Kalman-filtered state"),
        ):
            e = errors[(errors.scenario == name) & (errors.predictor == predictor)]
            axes[0, c].plot(e.lookahead_s / 60, e.h_p95_nm, "o-", color=colour, ms=4, label=label)
            axes[1, c].plot(e.lookahead_s / 60, e.v_p95_ft, "o-", color=colour, ms=4, label=label)
        axes[0, c].set_title(name)
        axes[1, c].set_xlabel("look-ahead [min]")
    axes[0, 0].set_ylabel("horizontal error, 95th pct [NM]")
    axes[1, 0].set_ylabel("vertical error, 95th pct [ft]")
    axes[0, 0].legend(loc="upper left")
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def scaling_png(bench: pd.DataFrame, path: str | Path) -> Path:
    """Pair-search time vs number of aircraft (log-log)."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    nv = bench.dropna(subset=["naive"])
    ax.loglog(nv.aircraft, nv.naive * 1e3, "o-", color=ORANGE, ms=4, label="all pairs, O(n²)")
    ax.loglog(
        bench.aircraft,
        bench.kdtree * 1e3,
        "o-",
        color=BLUE,
        ms=4,
        label="swept-segment KD-tree pre-filter",
    )
    for df, col in ((nv, "naive"), (bench, "kdtree")):
        last = df.iloc[-1]
        ax.annotate(
            f"{last[col] * 1e3:.0f} ms",
            (last.aircraft, last[col] * 1e3),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    ax.set_xlabel("aircraft in the airspace")
    ax.set_ylabel("pair search + CPA per evaluation [ms]")
    ax.set_title("Conflict search time per evaluation step")
    ax.legend(loc="upper left")
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def write_run_figures(res: PipelineResult, out: str | Path) -> None:
    """Figures written by ``python -m conflict_detection run``."""
    out = Path(out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    trajectory_map(res, out / "map.html")
    ev = showcase_event(res)
    if ev is None:
        return
    map_png(res, figs / "map.png", ev.event_id)
    separation_figure(res, ev.icao_a, ev.icao_b).write_html(
        figs / f"separation_{ev.event_id}.html", include_plotlyjs="cdn"
    )
    separation_png(res, ev.icao_a, ev.icao_b, figs / f"separation_{ev.event_id}.png")
    resolution_png(res, ev.event_id, figs / f"resolution_{ev.event_id}.png")
