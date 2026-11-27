"""BlueSky scenario export and log verification (BlueSky itself is optional)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conflict_detection import bluesky_bridge as bb
from conflict_detection.config import Config
from conflict_detection.geo import LocalFrame
from conflict_detection.units import FPM, FT, KT, NM

FRAME = LocalFrame(46.9, 7.8)


def states(vz_b=0.0, alt_b=37000.0):
    return pd.DataFrame(
        {
            "t_eval": 0.0,
            "icao24": ["4b1234", "3c5678"],
            "callsign": ["SWR12", "dlh 4A"],
            "x": [0.0, 40 * NM],
            "y": [0.0, 1 * NM],
            "z": [37000 * FT, alt_b * FT],
            "vx": [450 * KT, -450 * KT],
            "vy": [0.0, 0.0],
            "vz": [0.0, vz_b * FPM],
        }
    )


EVENT = pd.Series({"event_id": "C001", "t_first": 0.0, "icao_a": "3c5678", "icao_b": "4b1234"})


def resolution(kind="heading", value=15.0, **kw):
    base = {
        "event_id": "C001",
        "resolved": True,
        "manoeuvring": "4b1234",
        "manoeuvre": f"{kind} {value:+.0f}",
        "kind": kind,
        "value": value,
        "new_heading_deg": 105.0,
        "new_speed_kt": 405.0,
        "new_altitude_ft": 38000.0,
    }
    base.update(kw)
    return pd.Series(base)


def test_acid_uses_clean_callsign_or_icao():
    taken: set[str] = set()
    assert bb.acid_for("dlh 4A", "3c5678", taken) == "DLH4A"
    assert bb.acid_for("DLH4A", "aaaaaa", taken) == "AAAAAAA"  # duplicate -> ICAO based
    assert bb.acid_for("", "4b1234", taken) == "A4B1234"


def test_aircraft_init_converts_to_bluesky_units():
    init = bb.aircraft_init(states().iloc[0], "SWR12", FRAME, 360.0)
    assert init.lat == pytest.approx(46.9) and init.lon == pytest.approx(7.8)
    assert init.heading_deg == pytest.approx(90.0)
    assert init.altitude_ft == pytest.approx(37000.0)
    assert init.tas_kt == pytest.approx(450.0)
    # 450 kt TAS at FL370 = 254.09 kt CAS (bluesky.tools.aero.vtas2cas agrees).
    assert init.cas_kt == pytest.approx(254.09, abs=0.05)
    cmds = init.commands()
    assert cmds == ["CRE SWR12 A333 46.900000 7.800000 90.0 37000 254.1"]


def test_climbing_aircraft_gets_alt_command():
    init = bb.aircraft_init(states(vz_b=1500.0).iloc[1], "DLH4A", FRAME, 360.0)
    assert init.target_altitude_ft == pytest.approx(37000 + 1500 * 6)
    assert init.commands()[1] == "ALT DLH4A 46000 1500"


def test_resolution_commands():
    init = bb.aircraft_init(states().iloc[0], "SWR12", FRAME, 360.0)
    assert bb.resolution_command(resolution(), init) == "HDG SWR12 105.0"
    # 405 kt TAS at FL370 = 226.31 kt CAS.
    assert bb.resolution_command(resolution("speed", -0.1), init) == "SPD SWR12 226.3"
    assert bb.resolution_command(resolution("level", 1000.0), init) == "ALT SWR12 38000 1500"
    assert bb.resolution_command(resolution("level", -1000.0, new_altitude_ft=36000.0), init) == (
        "ALT SWR12 36000 -1500"
    )
    with pytest.raises(ValueError):
        bb.resolution_command(resolution("teleport", 0.0), init)


def test_scenario_file_structure(tmp_path):
    files = bb.export_encounter(EVENT, states(), resolution(), FRAME, tmp_path, Config())
    unres = (tmp_path / "C001_unresolved.scn").read_text().splitlines()
    res = (tmp_path / "C001_resolved.scn").read_text().splitlines()
    cmds = [line for line in unres if not line.startswith("#")]
    assert all(">" in line for line in cmds)
    assert cmds[0] == "00:00:00.00>ASAS OFF"
    assert cmds[1] == "00:00:00.00>RESO OFF"
    assert sum(">CRE " in line for line in cmds) == 2
    assert any(">CRELOG CDRLOG 1.0" in line for line in cmds)
    assert cmds[-1] == "00:06:00.00>HOLD"  # 300 s horizon + 60 s margin
    assert "00:00:00.00>HDG SWR12 105.0" in res
    assert not any("HDG" in line for line in unres)
    assert files.acid_a == "DLH4A" and files.acid_b == "SWR12"
    assert files.simulable and files.command == "HDG SWR12 105.0"


def test_unresolved_event_has_no_resolved_file(tmp_path):
    files = bb.export_encounter(EVENT, states(), None, FRAME, tmp_path, Config())
    assert files.resolved is None
    assert not (tmp_path / "C001_resolved.scn").exists()


def test_above_type_ceiling_is_flagged(tmp_path):
    files = bb.export_encounter(EVENT, states(alt_b=43000.0), None, FRAME, tmp_path, Config())
    assert not files.simulable and "ceiling" in files.note


def test_export_all_writes_manifest(tmp_path):
    events = pd.DataFrame([EVENT])
    res = pd.DataFrame([resolution()])
    man = bb.export_all(events, states(), res, FRAME, tmp_path, Config())
    assert len(man) == 1
    assert json.loads((tmp_path / "manifest.json").read_text())[0]["event_id"] == "C001"


def write_log(path, sep_nm, dz_ft=0.0, n=61):
    """A fake CDRLOG file: two aircraft at constant separation."""
    rows = ["# test log", "# simt, id, lat, lon, alt, tas, gs, trk, vs"]
    for t in range(n):
        lat, lon = FRAME.to_latlon(0.0, 0.0)
        lat2, lon2 = FRAME.to_latlon(sep_nm * NM, 0.0)
        rows.append(f"{t:.8f},AC1,{lat:.8f},{lon:.8f},11277.6,240,240,90,0")
        rows.append(f"{t:.8f},AC2,{lat2:.8f},{lon2:.8f},{11277.6 + dz_ft * FT},240,240,90,0")
    path.write_text("\n".join(rows) + "\n")
    return path


def test_read_log_and_separation(tmp_path):
    log = bb.read_log(write_log(tmp_path / "CDRLOG_x.log", 3.0))
    assert set(log.acid) == {"AC1", "AC2"}
    sep = bb.separation_from_log(log, "AC1", "AC2")
    assert sep["los"]
    assert sep["min_hdist_nm"] == pytest.approx(3.0, abs=1e-3)
    assert sep["los_duration_s"] == pytest.approx(61.0)
    clear = bb.separation_from_log(
        bb.read_log(write_log(tmp_path / "y.log", 3.0, 1000.0)), "AC1", "AC2"
    )
    assert not clear["los"]
    with pytest.raises(ValueError):
        bb.separation_from_log(log, "AC1", "NOPE")


def test_verify_logs_and_manual_matching(tmp_path):
    man = bb.export_all(
        pd.DataFrame([EVENT]), states(), pd.DataFrame([resolution()]), FRAME, tmp_path, Config()
    )
    m = man.iloc[0]
    logdir = tmp_path / "out"
    logdir.mkdir()
    write_log(logdir / "CDRLOG_C001_unresolved_20260101_10-00-00.log", 1.0)
    write_log(logdir / "CDRLOG_C001_resolved_20260101_10-01-00.log", 8.0)
    # The fake logs use AC1/AC2 as ids; point the manifest at them.
    man.loc[:, ["acid_a", "acid_b"]] = ["AC1", "AC2"]
    logs = bb.find_manual_logs(man, logdir)
    assert set(logs) == {str(Path(m.unresolved).resolve()), str(Path(m.resolved).resolve())}
    v = bb.verify_logs(man, logs)
    assert bool(v.conflict_reproduced.iloc[0]) and bool(v.resolution_success.iloc[0])
    assert "bluesky-verify" in bb.manual_run_instructions(tmp_path)


@pytest.mark.skipif(not bb.bluesky_available(), reason="bluesky-simulator not installed")
def test_bluesky_headless_run(tmp_path, monkeypatch):
    # Relative scenario paths (as written by the CLI) must still match their logs.
    monkeypatch.chdir(tmp_path)
    files = bb.export_encounter(EVENT, states(), resolution(), FRAME, Path("scn"), Config())
    logs = bb.run_scenarios([files.unresolved, files.resolved], Path("work"))
    man = pd.DataFrame([files.__dict__])
    v = bb.verify_logs(man, logs).iloc[0]
    assert bool(v.unresolved_los)  # 1 NM head-on miss: LoS in BlueSky too
    assert not bool(v.resolved_los)  # the 15 deg turn keeps 5 NM
    assert np.isfinite(v.resolved_min_hdist_nm)
