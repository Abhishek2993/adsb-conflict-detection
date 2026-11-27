"""Streamlit browser for pipeline results.

Run:
    pip install -e ".[app]"
    python -m conflict_detection run --scenario data/sample --out results/sample
    streamlit run app/viewer.py

The app reads the folders written by ``python -m conflict_detection run``
(``results/<name>/``) and can also run the pipeline on any scenario folder
under ``data/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, CRITICAL, INK2 = "#257de0", "#f47230", "#00b894", "#dc3c3d", "#666666"

st.set_page_config(page_title="ADS-B conflict detection", layout="wide")


@st.cache_data
def load_results(folder: str) -> dict:
    """Read every table of one results folder."""
    p = Path(folder)
    out = {
        "metrics": json.loads((p / "metrics.json").read_text()),
        "conflicts": pd.read_csv(p / "conflicts.csv"),
        "actual": pd.read_csv(p / "actual_los.csv"),
        "detections": pd.read_csv(p / "detections.csv"),
        "resolutions": pd.read_csv(p / "resolutions.csv"),
        "grid": pd.read_parquet(p / "grid.parquet"),
        "frame": json.loads((p / "frame.json").read_text()),
    }
    ver = p / "bluesky" / "verification.csv"
    out["bluesky"] = pd.read_csv(ver) if ver.exists() else None
    out["map"] = (p / "map.html").read_text() if (p / "map.html").exists() else None
    return out


def run_pipeline_ui() -> None:
    """Sidebar form that runs the pipeline on a scenario folder."""
    scenarios = sorted(
        str(d.relative_to(ROOT)) for d in (ROOT / "data").rglob("trajectories.parquet")
    )
    with st.sidebar.expander("Run the pipeline"):
        scen = st.selectbox("Scenario", [str(Path(s).parent) for s in scenarios])
        predictor = st.radio("Predictor", ["kalman", "cv"], horizontal=True)
        rule = st.radio("Detection rule", ["cpa", "interval"], horizontal=True)
        horizon = st.slider("Look-ahead horizon [s]", 60, 600, 300, step=30)
        if st.button("Run"):
            from conflict_detection.config import Config
            from conflict_detection.ingest import load_scenario
            from conflict_detection.pipeline import run_pipeline, write_outputs

            name = f"{Path(scen).name}_{predictor}_{rule}_{horizon}"
            with st.spinner("Running..."):
                res = run_pipeline(
                    load_scenario(ROOT / scen),
                    Config(predictor=predictor, rule=rule, horizon_s=float(horizon)),
                )
                write_outputs(res, ROOT / "results" / name)
            st.cache_data.clear()
            st.success(f"Wrote results/{name}")


def separation_plot(data: dict, row: pd.Series) -> go.Figure:
    """Horizontal and vertical separation of the selected pair (two panels, shared time)."""
    grid = data["grid"]
    a = grid[grid.icao24 == row.icao_a].set_index("t")
    b = grid[grid.icao24 == row.icao_b].set_index("t")
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    t0 = row.t_first
    j = j[(j.index > t0 - 300) & (j.index < t0 + 600)]
    minutes = (j.index - t0) / 60
    h = ((j.x_a - j.x_b) ** 2 + (j.y_a - j.y_b) ** 2) ** 0.5 / 1852
    v = (j.z_a - j.z_b).abs() / 0.3048
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Horizontal separation [NM]", "Vertical separation [ft]"),
    )
    fig.add_scatter(
        x=minutes,
        y=h,
        name="horizontal",
        line={"color": BLUE, "width": 2},
        hovertemplate="%{y:.2f} NM<extra></extra>",
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=minutes,
        y=v,
        name="vertical",
        line={"color": ORANGE, "width": 2},
        hovertemplate="%{y:.0f} ft<extra></extra>",
        row=2,
        col=1,
    )
    det = data["detections"]
    det = det[(det.icao_a == row.icao_a) & (det.icao_b == row.icao_b)]
    det = det[(det.t_eval > t0 - 300) & (det.t_eval < t0 + 600)]
    fig.add_scatter(
        x=(det.t_eval - t0) / 60,
        y=det.dcpa_nm,
        mode="markers",
        name="predicted dCPA at alert",
        marker={"color": INK2, "size": 7},
        hovertemplate="dCPA %{y:.2f} NM<extra></extra>",
        row=1,
        col=1,
    )
    fig.add_hline(y=5, line={"dash": "dash", "color": INK2, "width": 1}, row=1, col=1)
    fig.add_hline(y=1000, line={"dash": "dash", "color": INK2, "width": 1}, row=2, col=1)
    act = data["actual"]
    act = act[(act.icao_a == row.icao_a) & (act.icao_b == row.icao_b)]
    for _, x in act.iterrows():
        for r in (1, 2):
            fig.add_vrect(
                x0=(x.t_start - t0) / 60,
                x1=(x.t_end - t0) / 60,
                fillcolor=CRITICAL,
                opacity=0.12,
                line_width=0,
                row=r,
                col=1,
            )
    fig.update_xaxes(title_text="minutes after the first alert", row=2, col=1)
    fig.update_layout(
        height=480,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def main() -> None:
    """Page layout."""
    st.title("ADS-B conflict detection & resolution")
    st.caption(
        "Short-term conflict detection on real and synthetic ADS-B trajectories; "
        "separation minima 5 NM and 1000 ft. Research demonstrator, not an operational tool."
    )
    run_pipeline_ui()
    folders = sorted(str(p.parent) for p in (ROOT / "results").glob("*/metrics.json"))
    if not folders:
        st.info(
            "No results yet: run `python -m conflict_detection run --scenario data/sample "
            "--out results/sample` or use the sidebar."
        )
        return
    folder = st.sidebar.selectbox("Results", folders, format_func=lambda f: Path(f).name)
    data = load_results(folder)
    m = data["metrics"]
    d = m["detection"]

    st.subheader(f"{m['scenario']} · {m['aircraft']} aircraft · {m['flight_hours']} flight hours")
    cols = st.columns(6)
    cols[0].metric("Predicted conflicts", d["alert_events"])
    cols[1].metric("Actual LoS (evaluable)", d["actual_los"])
    cols[2].metric("Recall", f"{d['recall']:.2f}" if d["recall"] == d["recall"] else "n/a")
    cols[3].metric(
        "Precision", f"{d['precision']:.2f}" if d["precision"] == d["precision"] else "n/a"
    )
    cols[4].metric(
        "Mean warning",
        f"{d['mean_warning_s']:.0f} s" if d["mean_warning_s"] == d["mean_warning_s"] else "n/a",
    )
    r = m["resolution_model"]
    cols[5].metric("Resolved (model)", f"{r['resolved']}/{r['events']}")
    if "bluesky" in m:
        b = m["bluesky"]
        st.caption(
            f"BlueSky: {b['conflict_reproduced']}/{b['simulated']} conflicts reproduced "
            f"without resolution; {b['resolution_success']}/{b['resolved_runs']} "
            "resolutions kept separation."
        )
    st.caption(
        f"predictor `{m['predictor']}`, rule `{m['rule']}`, horizon {m['horizon_s']:.0f} s"
        f" · ground truth: {m['truth_source']}"
    )

    tab_list, tab_map, tab_los = st.tabs(["Conflicts", "Map", "Actual LoS"])
    conflicts = data["conflicts"]
    with tab_list:
        show = [
            "event_id",
            "callsign_a",
            "callsign_b",
            "t_first_utc",
            "tcpa_first_s",
            "dcpa_first_nm",
            "vsep_first_ft",
            "severity_max",
            "confirmed",
            "manoeuvre",
        ]
        show = [c for c in show if c in conflicts]
        only = st.toggle("Only conflicts confirmed by an actual LoS", value=False)
        table = conflicts[conflicts.confirmed] if only and "confirmed" in conflicts else conflicts
        st.dataframe(table[show].round(2), hide_index=True, use_container_width=True)
        if len(table):
            eid = st.selectbox("Conflict", table.event_id)
            row = conflicts.set_index("event_id").loc[eid]
            row["event_id"] = eid
            left, right = st.columns([3, 2])
            left.plotly_chart(separation_plot(data, row), use_container_width=True)
            res = data["resolutions"].set_index("event_id").loc[eid]
            right.markdown(
                f"**Resolution:** `{res.manoeuvre}` on "
                f"{res.callsign if isinstance(res.callsign, str) else '-'}"
            )
            right.markdown(
                f"Normalised separation margin in the prediction model: "
                f"{res.margin_before:.2f} → {res.margin_after:.2f} (≥ 1 = separated)"
            )
            bs = data["bluesky"]
            if bs is not None and eid in set(bs.event_id):
                v = bs.set_index("event_id").loc[eid]
                right.markdown(f"**BlueSky command:** `{v.command}`")
                if "unresolved_los" in v and v.unresolved_los == v.unresolved_los:
                    right.markdown(
                        f"- without resolution: min {v.unresolved_min_hdist_nm:.2f} NM, "
                        f"{'**LoS**' if v.unresolved_los else 'no LoS'}\n"
                        f"- with resolution: min {v.resolved_min_hdist_nm:.2f} NM at "
                        f"{v.resolved_vsep_at_min_ft:.0f} ft, "
                        f"{'**LoS**' if v.resolved_los else 'separation kept'}"
                    )
            fig_png = Path(folder) / "figures" / f"resolution_{eid}.png"
            if fig_png.exists():
                right.image(str(fig_png))
    with tab_map:
        if data["map"]:
            components.html(data["map"], height=650)
    with tab_los:
        st.dataframe(data["actual"].round(2), hide_index=True, use_container_width=True)


main()
