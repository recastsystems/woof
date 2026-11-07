"""Coverage and metadata contracts, with no CUDA initialization."""
import json
from pathlib import Path
import subprocess
import sys

from woof.hex import observation_scorecard as scorecard


def test_metadata_addition_gets_all_required_missing_rows():
    configuration = {"references": [{"label": "another_forecast", "role": "forecast"},
                                    {"label": "context", "role": "analysis"}],
                     "thresholds_mm": [1, 5, 10], "widths_km": [30, 90, 180]}
    frame = {"valid_time": "2026-10-10T12:00:00Z"}
    rows = scorecard.missing_rows(frame, "mrms", "archive outage", configuration)
    assert len(rows) == 18
    assert {r["label"] for r in rows} == {"hex", "another_forecast"}
    assert all(r["value"] is None and r["n"] == 0 for r in rows)
    assert len(scorecard.missing_rows(frame, "radiosonde", "absent", configuration)) == 16 * 3 * 3


def test_score_door_help_is_reachable_without_cuda():
    result = subprocess.run([sys.executable, "-m", "woof.hex.cli", "score", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--config" in result.stdout and "--leads" in result.stdout
    assert "CUDA" not in result.stderr


def test_discover_uses_run_provenance_and_previous_unselected_boundary(tmp_path):
    frames = [tmp_path / f"cuda-history.2026-10-10_{hour:02d}.00.00.nc" for hour in (0, 6, 12)]
    for frame in frames:
        frame.write_bytes(b"not weather data")
    (tmp_path / "forecast-receipt.json").write_text(json.dumps({"init": {"path": "/input/init.nc"},
        "mesh": {"grid": "/input/grid.nc", "name": "arbitrary-mesh"},
        "schedule": {"start_time": "2026-10-10_00:00:00"}}))
    config, found = scorecard.discover(tmp_path)
    assert config["init"] == "/input/init.nc"
    assert found[1]["valid_time"] == "2026-10-10T06:00:00Z"
    assert config["mesh"] == "arbitrary-mesh"


def test_markdown_uses_latest_lead_and_metadata_forecast_columns():
    config = {"start": "2026-10-10_00:00:00", "references": [{"label": "third", "role": "forecast"}],
              "thresholds_mm": [1], "widths_km": [30]}
    rows = [{"source": "asos", "quantity": "temperature", "label": "hex", "status": "MEASURED",
             "valid_time": f"2026-10-10T{h}:00:00Z", "rmse": error, "n": 12} for h, error in [("06", 9), ("12", 2)]]
    rows.append({**rows[0], "label": "third", "rmse": 1})
    text = scorecard.markdown({"run": "/run/fc", "status": "PARTIALLY_MEASURED", "configuration": config, "metrics": rows})
    assert "2.00 (12)" in text and "9.00 (12)" not in text
    assert "1.00 (12)" not in text
    assert "| mm / km | hex | third |" in text


def test_forecast_hook_queues_without_scoring_or_gpu_lease_delay(tmp_path, monkeypatch):
    (tmp_path / "cuda-history.2026-10-10_12.00.00.nc").write_bytes(b"history inventory fixture")
    init = tmp_path / "init.nc"
    init.write_bytes(b"shared prepared inode fixture")
    queued = []
    monkeypatch.setattr(scorecard, "queue_retry", lambda run, config: queued.append(config))
    monkeypatch.setattr(scorecard, "score", lambda *a, **k: (_ for _ in ()).throw(AssertionError("foreground scorer holds cards")))
    result = scorecard.forecast_hook(tmp_path, init=init, grid=tmp_path / "grid.nc", start="2026-10-10_00:00:00")
    assert result["status"] == "QUEUED" and len(queued) == 1
    assert (tmp_path / ".keep").is_file()
    pin = tmp_path / "observation-scorecard/pending-init.nc"
    assert pin.stat().st_ino == init.stat().st_ino
    card = json.loads((tmp_path / "observation-scorecard/scorecard.json").read_text())
    assert all(r["n"] == 0 and r["status"] == "NOT_MEASURED" for r in card["metrics"])


def test_a_door_run_with_scoring_off_reports_skipped_instead_of_raising(tmp_path, monkeypatch):
    """both-routes matrix, woof hex 0.4.14 candidate (2026-11-06): with WOOF_HEX_OBSERVATION_SCORING=off and the
    score stage still listed, door_scorecard raised "could not schedule CPU observation scoring" after a finished
    forecast and the country door exited 1."""
    run = tmp_path / "forecast"
    run.mkdir()
    (run / "cuda-history.2026-10-10_12.00.00.nc").write_bytes(b"history inventory fixture")
    config = tmp_path / "observation-scorecard.config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(scorecard.SCORING_ENV, "off")
    monkeypatch.setattr(scorecard, "queue_retry", lambda *a, **k: (_ for _ in ()).throw(AssertionError("queued")))
    joined = scorecard.door_scorecard(run, config)
    assert joined["status"] == "SKIPPED" and joined["joined"] == "skipped"
    assert "SKIPPED" in joined["note"] and scorecard.SCORING_ENV in joined["note"]
    card = json.loads((run / "observation-scorecard" / "scorecard.json").read_text(encoding="utf-8"))
    assert card["status"] == "SKIPPED" and card["measured_metrics"] == 0


def test_the_country_door_score_stage_finishes_with_scoring_off(tmp_path, monkeypatch):
    from woof.hex import country_door

    root = tmp_path / "work"
    (root / "forecast").mkdir(parents=True)
    (root / "mesh-rows.json").write_text("{}", encoding="utf-8")
    (root / "observation-scorecard.config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(scorecard.SCORING_ENV, "off")
    notes = []
    status = type("Status", (), {"note": lambda self, text: notes.append(text)})()
    args = type("Args", (), {"work_dir": root})()
    result = country_door._run_stage("score", args, {"commands": {}}, status)
    assert result["returncode"] == 0 and result["status"] == "SKIPPED"
    assert any("SKIPPED" in note for note in notes)


def test_small_command_receipt_avoids_loading_large_driver_arrays(tmp_path):
    run = tmp_path / "fc"
    run.mkdir()
    (run / "cuda-history.2026-10-10_12.00.00.nc").write_bytes(b"inventory fixture")
    (run / "forecast-receipt.json").write_text("deliberately unreadable array-heavy receipt")
    (tmp_path / "command.json").write_text(json.dumps({"command": ["forecast", "--init", str(tmp_path / "init.nc"),
        "--grid", str(tmp_path / "grid.nc"), "--mesh", "arbitrary", "--start-time", "2026-10-10_00:00:00"]}))
    config, frames = scorecard.discover(run)
    assert config["mesh"] == "arbitrary" and len(frames) == 1


def test_s2s_coordinator_owns_verification_without_future_data_retention(tmp_path, monkeypatch):
    (tmp_path / "cuda-history.2026-10-10_12.00.00.nc").write_bytes(b"history inventory fixture")
    init = tmp_path / "init.nc"
    init.write_bytes(b"shared prepared inode fixture")
    config = tmp_path / "coordinator.json"
    config.write_text(json.dumps({"verification_owner": "s2s verify stage"}))
    monkeypatch.setenv("WOOF_HEX_OBSERVATION_CONFIG", str(config))
    monkeypatch.setattr(scorecard, "queue_retry", lambda *a: (_ for _ in ()).throw(
        AssertionError("coordinator verification must not retain future history")))
    result = scorecard.forecast_hook(tmp_path, init=init, grid=tmp_path / "grid.nc",
                                     start="2026-10-10_00:00:00")
    assert result["status"] == "DEFERRED" and result["owner"] == "s2s verify stage"
    assert not (tmp_path / ".keep").exists()
    assert not (tmp_path / "observation-scorecard/pending-init.nc").exists()
    card = json.loads((tmp_path / "observation-scorecard/scorecard.json").read_text())
    assert card["metrics"] and all(row["status"] == "NOT_MEASURED" for row in card["metrics"])


class _ArchiveCommands:
    """Stands in for the scorer's command runner: NOMADS refuses, IGRA2 serves (no network, no binaries)."""

    def __init__(self, scratch, *, prepbufr_ok=False, igra_ok=True, soundings=3, snapshot_ok=False, por_bytes=10 << 20):
        self.scratch, self.calls = scratch, []
        self.prepbufr_ok, self.igra_ok, self.soundings = prepbufr_ok, igra_ok, soundings
        self.snapshot_ok, self.por_bytes = snapshot_ok, por_bytes

    def run(self, argv, *, json_output=True, acquisition=True, timeout=None):
        self.calls.append((list(map(str, argv)), timeout))
        tool, args = Path(str(argv[0])).name, list(map(str, argv[1:]))
        if args[0] == "--fetch-igra2-snapshot":
            if not self.snapshot_ok:
                raise RuntimeError("hex_obs_scorecard exited 2: no IGRA2 year-to-date snapshot covers the window")
            start, end = scorecard.instant(args[2]), scorecard.instant(args[3])
            Path(args[4]).write_text("station_id,latitude_deg,longitude_deg,elevation_m,valid_time,pressure_hpa,variable,value\n")
            hours = {}
            when = start.replace(minute=0, second=0) + scorecard.timedelta(hours=1)
            while when <= end:
                if when.hour in (0, 12):
                    hours[scorecard.stamp(when)] = self.soundings
                when += scorecard.timedelta(hours=1)
            return {"schema": "gpuwm-hex.igra2-snapshot/v1", "snapshot": {"name": "IGRA_v2.2_data-y2d_s20250101_e20260710_c20260710.tar"},
                    "soundings_by_nominal_hour": {k: v for k, v in hours.items() if v}, "bytes_downloaded": 1}
        if args[0] == "--igra2-station-ids":
            return {"stations": ["USM00072357", "USM00072451"], "bytes": self.por_bytes}
        if args[0] == "--fetch-input":
            if not self.prepbufr_ok:
                raise RuntimeError("hex_obs_scorecard exited 1: HTTP 403")
            Path(args[2]).write_bytes(b"BUFR")
            return {"bytes": 4}
        if args[0] == "--decode-sondes":
            Path(args[2]).write_text("listing")
            return {"schema": "gpuwm-hex.decoded-sondes/v1"}
        if tool == "rw_igra2" and args[0] == "fetch":
            if not self.igra_ok:
                raise RuntimeError("rw_igra2 exited 1: archive unreachable")
            out = Path(args[args.index("--out") + 1])
            out.mkdir(parents=True)
            files = [{"path": str(out / f"S{i}-data-beg2026.txt.zip")} for i in range(2)]
            record = {"archive": "https://example.invalid/igra2", "files": files}
            (out / "fetch.json").write_text(json.dumps(record))  # rw_igra2 keeps its record beside the files
            return record
        if args[0] == "--decode-igra":
            Path(args[1]).write_text("station_id,latitude_deg,longitude_deg,elevation_m,valid_time,pressure_hpa,variable,value\n")
            return {"schema": "gpuwm-hex.decoded-igra/v1", "soundings": self.soundings, "rows": 0}
        raise AssertionError(f"unexpected command {argv}")


def _sonde_config(tmp_path):
    return {"prepbufr_url_template": "https://example.invalid/gdas.{YYYYMMDD}/gdas.t{HH}z.prepbufr.nr",
            "sonde_sources": list(scorecard.SONDE_SOURCES), "igra2_fetch_timeout_seconds": 1500}


def test_a_past_case_scores_sondes_from_igra2_once_nomads_has_dropped_the_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(scorecard, "binary", lambda name: f"/bin/{name}")
    commands = _ArchiveCommands(tmp_path / "scratch")
    valid = scorecard.datetime.now(scorecard.timezone.utc).replace(month=1, day=2, hour=12, minute=0, second=0, microsecond=0)
    first = scorecard.acquire_sondes(valid, tmp_path / "0" / "radiosonde", commands, _sonde_config(tmp_path))
    second = scorecard.acquire_sondes(valid + scorecard.timedelta(hours=12), tmp_path / "1" / "radiosonde", commands, _sonde_config(tmp_path))
    assert first.endswith("igra2-sondes.csv") and second.endswith("igra2-sondes.csv")
    fetches = [(argv, timeout) for argv, timeout in commands.calls if Path(argv[0]).name == "rw_igra2"]
    assert len(fetches) == 1, "one archive walk serves every frame of a scoring call"
    assert fetches[0][0][fetches[0][0].index("--collection") + 1] == "y2d" and fetches[0][1] == 1500
    decode = next(argv for argv, _ in commands.calls if "--decode-igra" in argv)
    assert decode[3:5] == [scorecard.stamp(valid - scorecard.timedelta(hours=1)), scorecard.stamp(valid + scorecard.timedelta(hours=1))]
    assert len(decode) == 5 + 2
    assert json.loads((tmp_path / "0" / "radiosonde" / "igra2" / "decode.json").read_text())["collection"] == "y2d"


def test_an_earlier_year_reads_the_igra2_period_of_record(tmp_path, monkeypatch):
    monkeypatch.setattr(scorecard, "binary", lambda name: f"/bin/{name}")
    commands = _ArchiveCommands(tmp_path / "scratch")
    valid = scorecard.datetime(2023, 3, 31, 12, tzinfo=scorecard.timezone.utc)
    scorecard.acquire_sondes(valid, tmp_path / "0" / "radiosonde", commands, _sonde_config(tmp_path))
    fetch = next(argv for argv, _ in commands.calls if Path(argv[0]).name == "rw_igra2")
    assert fetch[fetch.index("--collection") + 1] == "por"
    # Only the stations the run needs, never the whole period-of-record archive (2026-10-24).
    assert fetch[fetch.index("--stations") + 1] == "USM00072357,USM00072451"


def test_a_period_of_record_read_over_its_byte_budget_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(scorecard, "binary", lambda name: f"/bin/{name}")
    commands = _ArchiveCommands(tmp_path / "scratch", por_bytes=25 << 30)
    valid = scorecard.datetime(2019, 6, 1, 0, tzinfo=scorecard.timezone.utc)
    try:
        scorecard.acquire_sondes(valid, tmp_path / "0" / "radiosonde", commands, _sonde_config(tmp_path))
    except LookupError as error:
        assert "igra2_por_max_bytes" in str(error) and "25.0 GiB" in str(error)
    else:
        raise AssertionError("a 25 GiB period-of-record read must not start")
    assert not any(Path(argv[0]).name == "rw_igra2" for argv, _ in commands.calls)


def test_a_past_case_streams_one_igra2_snapshot_for_every_sonde_frame(tmp_path, monkeypatch):
    # The 2025-05-19 12Z 15 h case: the 12Z and 00Z frames share one snapshot stream; no per-station archive.
    monkeypatch.setattr(scorecard, "binary", lambda name: f"/bin/{name}")
    commands = _ArchiveCommands(tmp_path / "scratch", snapshot_ok=True)
    first = scorecard.datetime(2025, 5, 19, 12, tzinfo=scorecard.timezone.utc)
    commands.sonde_window = (first, first + scorecard.timedelta(hours=12))
    tables = [scorecard.acquire_sondes(first + scorecard.timedelta(hours=h), tmp_path / str(h) / "radiosonde", commands,
                                       _sonde_config(tmp_path)) for h in (0, 12)]
    assert tables[0] == tables[1] and tables[0].endswith("igra2-snapshot-2025051912-2025052000.csv")
    streams = [argv for argv, _ in commands.calls if "--fetch-igra2-snapshot" in argv]
    assert len(streams) == 1 and streams[0][3:5] == ["2025-05-19T11:00:00Z", "2025-05-20T01:00:00Z"]
    assert not any(Path(argv[0]).name == "rw_igra2" for argv, _ in commands.calls)


def test_a_recent_cycle_still_scores_from_gdas_prepbufr_first(tmp_path, monkeypatch):
    monkeypatch.setattr(scorecard, "binary", lambda name: f"/bin/{name}")
    commands = _ArchiveCommands(tmp_path / "scratch", prepbufr_ok=True)
    valid = scorecard.datetime(2026, 10, 12, 12, tzinfo=scorecard.timezone.utc)
    listing = scorecard.acquire_sondes(valid, tmp_path / "0" / "radiosonde", commands, _sonde_config(tmp_path))
    assert listing.endswith("listing.txt")
    assert not any(Path(argv[0]).name == "rw_igra2" for argv, _ in commands.calls)


def test_no_archive_serving_the_hour_names_every_archive_tried(tmp_path, monkeypatch):
    monkeypatch.setattr(scorecard, "binary", lambda name: f"/bin/{name}")
    valid = scorecard.datetime(2026, 10, 12, 12, tzinfo=scorecard.timezone.utc)
    for commands, words in [(_ArchiveCommands(tmp_path / "a", igra_ok=False), "archive unreachable"),
                            (_ArchiveCommands(tmp_path / "b", soundings=0), "holds no sounding")]:
        try:
            scorecard.acquire_sondes(valid, tmp_path / commands.scratch.name / "radiosonde", commands, _sonde_config(tmp_path))
        except LookupError as error:
            text = str(error)
        else:
            raise AssertionError("an hour no archive serves must not score")
        assert "gdas_prepbufr: " in text and "HTTP 403" in text and "igra2: " in text and words in text
        assert "igra2_snapshot: " in text


def test_rw_igra2_resolves_to_the_copy_this_wheel_bundles(tmp_path, monkeypatch):
    # The published gpuwm 2.9.0 wheel does not install rw_igra2, so every past case scored no sondes
    # (daily rehearsal HX7, 2026-10-15); the wheel bundles it, and the score door reads that copy
    # before woof's ladder, unless WOOF_RW_IGRA2 names one explicitly.
    import woof.hex.engines as engines

    bundled = tmp_path / "rw_igra2"
    bundled.write_text("#!/bin/sh\n")
    bundled.chmod(0o755)
    monkeypatch.delenv("WOOF_RW_IGRA2", raising=False)
    monkeypatch.setattr(engines, "bundled_program", lambda name: str(bundled) if name == "rw_igra2" else None)
    assert scorecard.binary("rw_igra2") == str(bundled)



def test_on_demand_bridge_is_staged_when_the_ladder_finds_nothing(tmp_path, monkeypatch):
    # gpuwm 2.9.3 moved rw_asos out of the wheel to on-demand; the scorer never fetched it, so every
    # ASOS row scored NOT_MEASURED (hex 0.4.1 on gpuwm 2.9.3, 2026-10-20).
    import woof.hex.engines as engines
    import woof.rustwx as rustwx
    from woof import bridges

    staged = tmp_path / "rw_asos"
    monkeypatch.delenv("WOOF_RW_ASOS", raising=False)
    monkeypatch.setattr(engines, "bundled_program", lambda name: None)
    monkeypatch.setattr(rustwx, "crate_dir", lambda: tmp_path / "no-crate")
    monkeypatch.setattr(bridges, "find_artifact", lambda env, filename: staged if staged.is_file() else None)
    calls = []

    def fetch(name):
        calls.append(name)
        staged.write_text("#!/bin/sh\n")
        return True

    monkeypatch.setattr(bridges, "fetch_on_demand", fetch, raising=False)
    assert scorecard.binary("rw_asos") == str(staged.resolve())
    assert calls == ["rw_asos"]

    staged.unlink()
    monkeypatch.setattr(bridges, "fetch_on_demand", lambda name: False, raising=False)
    monkeypatch.setattr(bridges, "on_demand_failure", lambda name: "bundle v2.9.3 answered HTTP 404", raising=False)
    try:
        scorecard.binary("rw_asos")
    except FileNotFoundError as error:
        assert "HTTP 404" in str(error)
    else:
        raise AssertionError("a failed on-demand fetch must refuse with its reason")

def test_native_streams_score_once_with_native_provenance(tmp_path):
    for prefix in ("custom", "custom-diag"):
        for hour in (0, 3):
            (tmp_path / f"{prefix}.2026-10-16_{hour:02d}.00.00.nc").write_bytes(b"inventory")
    (tmp_path / "native-physics-request.json").write_text(json.dumps({
        "init": "/input/init.nc", "grid": "/input/grid.nc", "start": "2026-10-16_00:00:00",
        "streams": [{"name": "output", "filename": "custom.$Y-$M-$D_$h.$m.$s.nc",
                     "directions": ["output"]},
                    {"name": "diagnostics", "filename": "custom-diag.$Y-$M-$D_$h.$m.$s.nc",
                     "directions": ["output"]}]}))
    config, frames = scorecard.discover(tmp_path)
    assert config["init"] == "/input/init.nc"
    assert [f["valid_time"] for f in frames] == ["2026-10-16T00:00:00Z", "2026-10-16T03:00:00Z"]
    assert all(Path(f["history"]).name.startswith("custom.") for f in frames)

    assert all(Path(f["diagnostics"]).name.startswith("custom-diag.") for f in frames)


def test_native_scorer_reads_mpas_aliases_and_companion_diagnostics(tmp_path):
    import os
    import pytest
    nc = pytest.importorskip("netCDF4")
    binary = os.environ.get("WOOF_HEX_OBS_SCORECARD", "")
    if not binary or not Path(binary).is_file():
        pytest.skip("set WOOF_HEX_OBS_SCORECARD to the compiled scorer")
    def write(name, fields):
        path = tmp_path / name
        with nc.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as f:
            for dim, size in (("nCells",3),("nVertLevels",3),("nVertLevelsP1",4)):
                f.createDimension(dim,size)
            for key, (dims, values) in fields.items():
                f.createVariable(key,"f8",dims)[:] = values
        return str(path)
    mesh = {"latCell":(("nCells",),[0,.01,-.01]), "lonCell":(("nCells",),[0,.01,.01])}
    init = write("init.nc", {**mesh, "zgrid":(("nCells","nVertLevelsP1"),[[0,100,1000,5000]]*3)})
    history = write("history.nc", {**mesh,
        "pressure":(("nCells","nVertLevels"),[[95000,70000,40000]]*3),
        "theta":(("nCells","nVertLevels"),[[300]*3]*3),
        "uReconstructZonal":(("nCells","nVertLevels"),[[3]*3]*3),
        "uReconstructMeridional":(("nCells","nVertLevels"),[[4]*3]*3)})
    diag = write("diag.nc", {"t2m":(("nCells",),[300]*3), "u10":(("nCells",),[3]*3),
        "v10":(("nCells",),[4]*3)})
    valid="2026-10-16T00:00:00Z"
    surface=tmp_path / "surface.csv"
    surface.write_text("station_id,latitude_deg,longitude_deg,elevation_m,valid_time,variable,value,measurement\n"+
        "\n".join(f"A,0,0,0,{valid},{q},{v},{m}" for q,v,m in (
            ("temperature_k",300,"screen_temperature_2m"),("wind_u_m_s",3,"anemometer_wind_10m"),
            ("wind_v_m_s",4,"anemometer_wind_10m")))+"\n")
    sondes=tmp_path / "sondes.csv"
    sondes.write_text("station_id,latitude_deg,longitude_deg,elevation_m,valid_time,pressure_hpa,variable,value\n"+
        "\n".join(f"A,0,0,0,{valid},850,{q},{v}" for q,v in (("wind_u_m_s",3),("wind_v_m_s",4)))+"\n")
    request=tmp_path / "request.json"; result=tmp_path / "result.json"
    request.write_text(json.dumps(dict(history=history,diagnostics=diag,init=init,valid_time=valid,start=valid,
        surface=str(surface),sondes=str(sondes),references=[])))
    subprocess.run([binary,str(request),str(result)],check=True,capture_output=True,text=True)
    card=json.loads(result.read_text())
    for source, quantity, level in (("asos","temperature",0),("asos","wind",0),("radiosonde","wind",850)):
        row=next(r for r in card["metrics"] if r["source"]==source and r["quantity"]==quantity and r["level_hpa"]==level)
        assert row["status"]=="MEASURED" and row["n"]==1 and row["rmse"]==0
    assert card["provenance"]["diagnostics"]["path"]==diag


def test_completed_card_checks_binary_configuration_and_all_native_stream_bytes(tmp_path, monkeypatch):
    import hashlib
    files = {key: tmp_path / key for key in ('init', 'history', 'diagnostics', 'binary')}
    for key, path in files.items():
        path.write_bytes(key.encode())
    config = {'init': str(files['init']), 'start': '2026-10-16T00:00:00Z'}
    frames = [{'history': str(files['history']), 'diagnostics': str(files['diagnostics']),
               'valid_time': '2026-10-16T03:00:00Z'}]
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps(config))
    monkeypatch.setattr(scorecard, 'discover', lambda *a: (config, frames))
    monkeypatch.setattr(scorecard, 'binary', lambda *a: files['binary'])
    provenance = {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                  for key, path in files.items() if key != 'binary'}
    card = {'status': 'PARTIALLY_MEASURED', 'measured_metrics': 1, 'configuration': config,
            'native_binary_sha256': hashlib.sha256(files['binary'].read_bytes()).hexdigest(),
            'sources': [provenance], 'metrics': [{'valid_time': frames[0]['valid_time']}]}
    root = tmp_path / 'observation-scorecard'
    root.mkdir()
    def write(value):
        (root / 'scorecard.json').write_text(json.dumps(value))
    write(card)
    assert scorecard.completed_scorecard(tmp_path, cfg)['returncode'] == 0
    for changes in ({'status': 'NOT_MEASURED'}, {'measured_metrics': 0}, {'error': 'native failed'},
                    {'configuration': {}}, {'native_binary_sha256': 'stale'}, {'metrics': []}, {'sources': []}):
        write({**card, **changes})
        assert scorecard.completed_scorecard(tmp_path, cfg) is None
    for key in files:
        write(card)
        saved = files[key].read_bytes()
        files[key].write_bytes(b'changed')
        assert scorecard.completed_scorecard(tmp_path, cfg) is None
        files[key].write_bytes(saved)
    write(card)
    assert scorecard.completed_scorecard(tmp_path, cfg)['measured_metrics'] == 1


def test_regional_current_year_fetch_selects_only_its_stations(tmp_path, monkeypatch):
    monkeypatch.setattr(scorecard, "binary", lambda name: f"/bin/{name}")
    commands = _ArchiveCommands(tmp_path)
    config = _sonde_config(tmp_path)
    config["region"] = {"west": -99, "south": 33, "east": -95, "north": 37}
    scorecard.acquire_sondes_igra2(scorecard.datetime.now(scorecard.timezone.utc), tmp_path / "sondes", commands, config)
    fetches = [argv for argv, _ in commands.calls if Path(str(argv[0])).name == "rw_igra2"]
    assert len(fetches) == 1
    assert "--stations" in fetches[0]
    assert fetches[0][fetches[0].index("--stations") + 1] == "USM00072357,USM00072451"
