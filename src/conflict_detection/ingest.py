"""Load ADS-B trajectories into one normalised schema.

Every source (the committed sample, the synthetic generator, traffic's bundled
data, OpenSky history) ends up as a data frame with these columns:

==============  =====================  ======================================
column          unit                   note
==============  =====================  ======================================
timestamp       UTC datetime           one row per ADS-B report
icao24          str                    24-bit transponder address, lower case
callsign        str                    may be empty
latitude        deg                    WGS84
longitude       deg                    WGS84
altitude        ft                     barometric (pressure) altitude
groundspeed     kt                     optional (NaN if missing)
track           deg true               optional
vertical_rate   ft/min                 optional
==============  =====================  ======================================

A *scenario* is a directory with ``trajectories.parquet`` and an optional
``metadata.json`` (and, for synthetic scenarios, ``truth.csv``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .units import FPM, KT, m_to_ft

COLUMNS = [
    "timestamp",
    "icao24",
    "callsign",
    "latitude",
    "longitude",
    "altitude",
    "groundspeed",
    "track",
    "vertical_rate",
]
REQUIRED = ["timestamp", "icao24", "latitude", "longitude", "altitude"]

# Column names used by other tools, mapped to ours.
_ALIASES = {
    "time": "timestamp",
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "heading": "track",
    "alt": "altitude",
    "gs": "groundspeed",
    "vrate": "vertical_rate",
}


@dataclass
class Scenario:
    """A set of trajectories in a common airspace and time window."""

    name: str
    trajectories: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    truth: pd.DataFrame | None = None

    @property
    def n_aircraft(self) -> int:
        """Number of distinct aircraft."""
        return int(self.trajectories.icao24.nunique())


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` in the standard schema (see module docstring).

    Handles traffic/OpenSky naming (``baroaltitude`` and ``velocity`` in SI
    units, epoch-second ``time``), pyarrow-backed dtypes and stray whitespace
    in callsigns.
    """
    out = df.copy()
    # OpenSky Impala/Trino state vectors are in SI units.
    if "baroaltitude" in out and "altitude" not in out:
        out["altitude"] = m_to_ft(pd.to_numeric(out.pop("baroaltitude"), errors="coerce"))
    if "velocity" in out and "groundspeed" not in out:
        out["groundspeed"] = pd.to_numeric(out.pop("velocity"), errors="coerce") / KT
    if "vertrate" in out and "vertical_rate" not in out:
        out["vertical_rate"] = pd.to_numeric(out.pop("vertrate"), errors="coerce") / FPM
    out = out.rename(columns={k: v for k, v in _ALIASES.items() if k in out and v not in out})

    missing = [c for c in REQUIRED if c not in out]
    if missing:
        raise ValueError(f"trajectory data is missing required columns: {missing}")

    ts = out["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        ts = pd.to_datetime(ts, unit="s", utc=True)
    else:
        ts = pd.to_datetime(ts.astype("datetime64[ns, UTC]") if hasattr(ts, "dt") else ts, utc=True)
    out["timestamp"] = ts.astype("datetime64[ns, UTC]")

    for col in ["latitude", "longitude", "altitude", "groundspeed", "track", "vertical_rate"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        else:
            out[col] = np.nan
    out["icao24"] = out["icao24"].astype(str).str.strip().str.lower()
    if "callsign" in out:
        out["callsign"] = out["callsign"].astype("object").fillna("").astype(str).str.strip()
    else:
        out["callsign"] = ""
    extra = [c for c in out.columns if c not in COLUMNS and c in ("onground", "encounter")]
    out = out[COLUMNS + extra]
    return out.sort_values(["timestamp", "icao24"]).reset_index(drop=True)


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario directory, or a single ``.parquet``/``.csv`` trajectory file."""
    path = Path(path)
    if path.is_dir():
        files = [path / "trajectories.parquet"]
        if not files[0].exists():
            files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet trajectory files in {path}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        meta_file = path / "metadata.json"
        metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        truth_file = path / "truth.csv"
        truth = None
        if truth_file.exists():
            truth = pd.read_csv(truth_file)
            for col in ("t_cpa", "los_start", "los_end"):
                if col in truth:
                    truth[col] = pd.to_datetime(truth[col], utc=True)
        name = metadata.get("name", path.name)
    elif path.suffix == ".parquet":
        df, metadata, truth, name = pd.read_parquet(path), {}, None, path.stem
    elif path.suffix == ".csv":
        df, metadata, truth, name = pd.read_csv(path), {}, None, path.stem
    else:
        raise FileNotFoundError(f"cannot load scenario from {path}")
    return Scenario(name=name, trajectories=normalise(df), metadata=metadata, truth=truth)


def save_scenario(scenario: Scenario, out_dir: str | Path) -> Path:
    """Write a scenario directory (parquet + metadata, plus truth if present)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj = scenario.trajectories.copy()
    traj.to_parquet(out / "trajectories.parquet", index=False, compression="zstd")
    meta = dict(scenario.metadata)
    meta.setdefault("name", scenario.name)
    meta["n_aircraft"] = scenario.n_aircraft
    meta["n_reports"] = len(traj)
    meta["start"] = str(traj.timestamp.min())
    meta["stop"] = str(traj.timestamp.max())
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    if scenario.truth is not None:
        scenario.truth.to_csv(out / "truth.csv", index=False)
    return out


def build_traffic_sample(
    collection: str = "switzerland",
    start: str = "2018-08-01 14:00",
    stop: str = "2018-08-01 15:00",
) -> Scenario:
    """Cut a time window out of a collection bundled with ``traffic``.

    Requires the optional ``traffic`` dependency (``pip install -e .[traffic]``).
    The bundled collections are recorded by the OpenSky Network.
    """
    from traffic.data import samples

    tr = getattr(samples, collection)
    df = normalise(tr.data)
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(stop, tz="UTC")
    df = df[(df.timestamp >= t0) & (df.timestamp < t1)].reset_index(drop=True)
    meta = {
        "name": f"{collection}_{t0:%Y%m%d_%H%M}",
        "source": f"traffic.data.samples.{collection} (OpenSky Network data bundled with traffic)",
        "window_utc": [str(t0), str(t1)],
        "altitude_reference": "barometric",
    }
    return Scenario(name=meta["name"], trajectories=df, metadata=meta)


def fetch_opensky(
    start: str,
    stop: str,
    bounds: tuple[float, float, float, float],
) -> Scenario:
    """Download state vectors from the OpenSky historical database.

    Credentials are read by ``pyopensky`` from the environment only:
    ``OPENSKY_USERNAME`` and ``OPENSKY_PASSWORD`` (Trino access). Nothing is
    ever written to disk except the resulting trajectories.

    Args:
        start, stop: UTC time window, e.g. ``"2024-05-02 12:00"``.
        bounds: (west, south, east, north) in degrees.
    """
    if not (os.environ.get("OPENSKY_USERNAME") and os.environ.get("OPENSKY_PASSWORD")):
        raise RuntimeError(
            "Set OPENSKY_USERNAME and OPENSKY_PASSWORD in the environment to query "
            "the OpenSky historical database."
        )
    from traffic.data import opensky

    tr = opensky.history(start, stop, bounds=bounds)
    if tr is None or len(tr) == 0:
        raise RuntimeError("OpenSky returned no data for this window")
    df = normalise(tr.data)
    meta = {
        "name": f"opensky_{pd.Timestamp(start):%Y%m%d_%H%M}",
        "source": "OpenSky Network historical database (via traffic)",
        "window_utc": [start, stop],
        "bounds": list(bounds),
        "altitude_reference": "barometric",
    }
    return Scenario(name=meta["name"], trajectories=df, metadata=meta)
