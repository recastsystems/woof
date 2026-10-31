"""Configuration, history lifecycle and CLI contracts for simulated radar."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from woof import simulated_radar as radar


def test_default_surface_and_custom_coordinates_round_trip():
    assert radar.SimulatedRadarOptions.from_mapping(None) is radar.OFF
    options = radar.SimulatedRadarOptions.from_mapping({})
    assert options.enabled and options.fields == "auto"
    assert options.formats == ("level2", "cfradial1")
    custom = radar.SimulatedRadarOptions.from_mapping({
        "sites": ["ktlx", {"id": "SIM1", "lat": 35.0, "lon": -97.0, "height_m": 340.0}],
        "elevations_deg": [0.5, 1.5], "timing": "scan",
        "formats": ["level2", "cfradial1", "cfradial2", "odim"],
    })
    assert custom.scan_strategy == "custom"
    assert custom.sites[0] == "KTLX"
    assert radar.SimulatedRadarOptions.from_mapping(custom.to_mapping()) == custom
    json.dumps(custom.to_mapping(), allow_nan=False)


@pytest.mark.parametrize("value,match", [
    ({"enable": True}, "unknown keys"),
    ({"enabled": "true"}, "enabled"),
    ({"sites": []}, "sites"),
    ({"sites": ["KTLX", "ktlx"]}, "duplicate"),
    ({"sites": ["../x"]}, "four-character"),
    ({"sites": [{"id": "TST1", "lat": 91, "lon": 0, "height_m": 0}]}, "latitude"),
    ({"formats": ["not-a-format"]}, "formats"),
    ({"formats": ["level2", "level2"]}, "formats"),
    ({"fields": ["imaginary"]}, "fields"),
    ({"timing": "instant-ish"}, "timing"),
    ({"color_tables": "rainbow"}, "color_tables"),
    ({"elevations_deg": [1, 0.5]}, "increase"),
    ({"scan_strategy": "vcp212", "elevations_deg": [0.5]}, "custom"),
    ({"range_km": float("nan")}, "finite"),
    ({"gate_spacing_m": 0}, "gate_spacing"),
    ({"gate_spacing_m": 250.5}, "whole metres"),
    ({"volume_duration_s": True}, "volume_duration"),
    ({"azimuth_step_deg": 0.0001}, "geometry"),
    ({"gate_spacing_m": 1}, "16384"),
    ({"sites": [{"id": "TST1", "lat": 35, "lon": -97, "height_m": 300}]}, "Py-ART"),
])
def test_invalid_configuration_cannot_reach_native(value, match):
    with pytest.raises(ValueError, match=match):
        radar.SimulatedRadarOptions.from_mapping(value)


def test_declared_metadata_uses_the_option_defaults():
    from woof.config import declared_key_rows
    rows = declared_key_rows()["simulated_radar"]
    options = radar.SimulatedRadarOptions.from_mapping({}).to_mapping()
    # color_tables is declared with its default but left out of a default
    # request, so a table that does not name it writes the request it
    # always did (re-recorded 2026-10-03 with the key).
    assert set(rows) == set(options) | {"color_tables", "band", "wavelength_mm"}
    assert rows["color_tables"]["default"] == "standard"
    assert {name: row["default"] for name, row in rows.items()
            if name not in {"color_tables", "band", "wavelength_mm"}} == options


@pytest.mark.parametrize("band,wavelength", [("S",107.068735),("C",53.5),("X",33.3)])
def test_band_metadata_and_wavelength_have_one_canonical_request(band,wavelength):
    explicit = radar.SimulatedRadarOptions.from_mapping({"band":band})
    wave = radar.SimulatedRadarOptions.from_mapping({"wavelength_mm":wavelength})
    assert explicit == wave
    site = radar.RadarSite("SIM1",35.,-97.,350.,wavelength_mm=wavelength)
    assert site.to_mapping().get("band","S") == band
    assert "band" not in radar.SimulatedRadarOptions.from_mapping({}).to_mapping()


@pytest.mark.parametrize("options", [{"band":"Ku"},{"wavelength_mm":float("nan")},
    {"wavelength_mm":True},{"wavelength_mm":54.},{"band":"C","wavelength_mm":33.3}])
def test_band_requests_never_substitute_a_different_table(options):
    with pytest.raises(ValueError,match="radar_"):
        radar.SimulatedRadarOptions.from_mapping(options)


def test_direct_wavelength_options_are_not_lost_during_serialization():
    assert radar.SimulatedRadarOptions(enabled=True,wavelength_mm=53.5).to_mapping()["band"] == "C"


def test_color_tables_selects_the_earlier_ppi_colours_and_stays_out_of_default_requests():
    default = radar.SimulatedRadarOptions.from_mapping({})
    assert default.color_tables == "standard"
    assert "color_tables" not in default.to_mapping()
    explicit = radar.SimulatedRadarOptions.from_mapping({"color_tables": "standard"})
    assert explicit == default and explicit.to_mapping() == default.to_mapping()
    classic = radar.SimulatedRadarOptions.from_mapping({"color_tables": "classic"})
    assert classic.to_mapping()["color_tables"] == "classic"
    assert radar.SimulatedRadarOptions.from_mapping(classic.to_mapping()) == classic


@pytest.mark.parametrize("format", radar.FORMATS)
def test_every_writer_enforces_shared_gate_limit(format):
    with pytest.raises(ValueError, match="16384 gates"):
        radar.SimulatedRadarOptions.from_mapping({"formats": [format], "gate_spacing_m": 1})


def test_experiment_admits_radar_and_refuses_pruned_wind(tmp_path):
    from woof.experiment import load_experiment
    text = """[experiment]
name = "radar-fixture"
start_time = 2026-10-01T00:00:00
run_seconds = 60.0
restart_interval_s = 0.0
[shared]
nz = 8
ztop = 12000.0
[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 20
ny = 20
dx = 3000.0
time_step = 3
history_interval_s = 60.0
[simulated_radar]
sites = ["KTLX"]
"""
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    assert load_experiment(path).simulated_radar.sites == ("KTLX",)
    path.write_text(text + '\n[output]\nhistory_drop = ["U"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="atmospheric columns"):
        load_experiment(path)


def test_simulated_radar_does_not_change_forecast_identity():
    from woof.core.model import restart_identity_payload
    from woof.experiment import experiment_config_document
    from woof.verify.cases.nest_ideal_r3 import load_scaffold
    exp = load_scaffold()
    enabled = replace(exp, simulated_radar=radar.SimulatedRadarOptions.from_mapping({}))
    assert restart_identity_payload(exp) == restart_identity_payload(enabled)
    assert "simulated_radar" not in experiment_config_document(exp)
    assert experiment_config_document(enabled)["simulated_radar"]["enabled"]


def test_live_queue_runs_before_forecast_finishes(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []
    def native(paths, *, outdir, config, volume_paths, started):
        calls.append((paths, volume_paths))
        entered.set()
        assert release.wait(5)
        return {}
    options = radar.SimulatedRadarOptions.from_mapping({})
    with radar.LiveSimulatedRadar(options, tmp_path, runner=native) as live:
        live.output_committed(domain=1, valid_time="a", path=tmp_path / "a")
        assert entered.wait(5), "radar did not begin while the forecast was active"
        live.output_committed(domain=1, valid_time="b", path=tmp_path / "b")
        release.set()
        live.drain()
    assert [([Path(p).name for p in paths], volume) for paths, volume in calls] == [
        (["a"], None), (["b"], None)]
    assert not live._thread.is_alive()


def test_scan_timing_publishes_each_history_once_with_its_successor(tmp_path):
    """Scan timing used to write every volume twice: first as a held anchor
    when it landed, then again under a second generation once its successor
    arrived. Each history now publishes once, beside its same-grid successor;
    the last history of each grid publishes at close with no successor."""
    calls = []
    def native(paths, *, outdir, config, volume_paths, started):
        calls.append(([Path(p).name for p in paths], [Path(p).name for p in volume_paths]))
        return {}
    options = radar.SimulatedRadarOptions.from_mapping({"timing": "scan"})
    with radar.LiveSimulatedRadar(options, tmp_path, runner=native) as live:
        for domain, name in ((1, "a"), (2, "x"), (1, "b"), (1, "c")):
            live.output_committed(domain=domain, valid_time=name, path=tmp_path / name)
        live.drain()
        assert calls == [(["a", "b"], ["a"]), (["b", "c"], ["b"])]
    assert calls[2:] == [(["c"], ["c"]), (["x"], ["x"])]
    published = [name for _, volume in calls for name in volume]
    assert sorted(published) == ["a", "b", "c", "x"], "every history publishes exactly once"


def test_native_failure_is_raised_at_drain_and_close(tmp_path):
    def native(*args, **kwargs):
        raise ValueError("history lacks beam height coordinates")
    live = radar.LiveSimulatedRadar(radar.SimulatedRadarOptions(enabled=True), tmp_path, runner=native)
    live.output_committed(domain=1, valid_time="a", path=tmp_path / "a")
    with pytest.raises(RuntimeError, match="beam height"):
        live.drain()
    with pytest.raises(RuntimeError, match="beam height"):
        live.close()
    assert not live._thread.is_alive()


def test_common_writer_landing_keeps_radar_when_observer_is_attached():
    from woof.io.wrfout import PerDomainWrfoutWriters
    events = []
    writers = object.__new__(PerDomainWrfoutWriters)
    writer = SimpleNamespace(paths=[], pending=0, landing_observer=None)
    writers._writers = {1: writer}
    writers._simulated_radar = SimpleNamespace(output_committed=lambda **event: events.append(("radar", event)))
    observer = SimpleNamespace(output_committed=lambda **event: events.append(("other", event)))
    writers.attach_progress_callback(observer)
    writer.landing_observer(domain=1, valid_time="a", path="file")
    assert [name for name, _ in events] == ["radar", "other"]


def test_common_writer_close_propagates_requested_product_failure():
    from woof.io.wrfout import PerDomainWrfoutWriters
    writers = object.__new__(PerDomainWrfoutWriters)
    writers._writers = {}
    def fail():
        raise RuntimeError("simulated radar failed")
    writers._simulated_radar = SimpleNamespace(close=fail)
    with pytest.raises(RuntimeError, match="simulated radar failed"):
        writers.close()


def test_synchronous_runner_uses_same_live_native_door(monkeypatch, tmp_path):
    from woof import rustwx
    calls = []
    monkeypatch.setattr(rustwx, "require_simulated_radar_binary", lambda: tmp_path / "rw_simradar")
    monkeypatch.setattr(rustwx, "simulate_radar", lambda *a, **k: calls.append((a, k)), raising=False)
    seen = []
    callback = SimpleNamespace(output_committed=lambda **e: seen.append(e))
    def forecast(*, progress_callback):
        progress_callback.output_committed(domain=1, valid_time="t", path=tmp_path / "history")
        return "complete"
    result = radar.run_with_radar(forecast, radar_options=radar.SimulatedRadarOptions(enabled=True),
                                 radar_outdir=tmp_path, progress_callback=callback)
    assert result == "complete" and len(calls) == len(seen) == 1
    assert calls[0][1]["binary"] == tmp_path / "rw_simradar", "the admitted binary is reused"


def test_replay_cli_only_passes_history_paths(monkeypatch, tmp_path):
    from woof import rustwx
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    history = input_dir / "wrfout_d01_2026-10-01_00_00_00"
    history.write_bytes(b"fixture passed to native only")
    (input_dir / (history.name + ".json")).write_text("{}", encoding="utf-8")
    (input_dir / (history.name + ".tmp")).write_text("unfinished", encoding="utf-8")
    seen = []
    monkeypatch.setattr(rustwx, "simulate_radar", lambda *a, **k: seen.append((a, k)) or {}, raising=False)
    parser = argparse.ArgumentParser()
    radar.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["simulated-radar", str(input_dir), "--sites", "KTLX", "--outdir", str(tmp_path)])
    assert args.func(args) == 0
    assert seen[0][0][0] == [history.resolve()]
    assert seen[0][1]["config"]["sites"] == ["KTLX"]


def test_cli_describe_exposes_schema_without_history(capsys):
    from woof.cli import main
    assert main(["simulated-radar", "--describe"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == radar.REQUEST_SCHEMA
    assert result["defaults"]["fields"] == "auto"


@pytest.mark.parametrize("layout", ["single", "tree"])
def test_prepared_command_preserves_radar_table_without_editing_authority(tmp_path, layout):
    from woof import stage_cli
    from woof import prepared_single_domain_forecast as single
    from woof import prepared_domain_tree_forecast as tree
    from woof.simulated_radar_config import apply_execution_options
    from woof.core.model import restart_identity_payload
    from woof.verify.cases.nest_ideal_r3 import load_scaffold

    root = tmp_path / "prepared"
    root.mkdir()
    schema = single._PROOF_SCHEMA["gfs"] if layout == "single" else single._HIERARCHY_PROOF_SCHEMA["gfs"]
    (root / "proof.json").write_text(json.dumps({
        "schema": schema, "status": "READY", "domain_count": 3,
        "input_manifest_sha256": "11" * 32,
        "prepared_cache": {"content_sha256": "22" * 32}}), encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text("unchanged prepared authority", encoding="utf-8")
    before = config.read_bytes()
    options = radar.SimulatedRadarOptions.from_mapping({"sites": ["KTLX"]})
    command = stage_cli.sim_command(
        stage_cli.resolve_bundle(root), experiment_config=config,
        wps_namelist=tmp_path / "namelist.wps", outdir=tmp_path / "output",
        simulated_radar=options)
    parsed = (single if layout == "single" else tree).build_parser().parse_args(command[3:])
    assert parsed.simulated_radar_table == options
    exp = load_scaffold()
    modified = apply_execution_options(exp, parsed.simulated_radar_table)
    assert modified.simulated_radar == options
    assert exp.simulated_radar is radar.OFF
    assert restart_identity_payload(modified) == restart_identity_payload(exp)
    assert config.read_bytes() == before


def test_prepared_radar_override_cannot_restore_missing_history_fields():
    from woof.simulated_radar_config import apply_execution_options
    from woof.io.history_selection import HistorySelection
    from woof.verify.cases.nest_ideal_r3 import load_scaffold
    exp = replace(load_scaffold(), output=HistorySelection(history_drop=("U",)))
    with pytest.raises(ValueError, match="atmospheric columns"):
        apply_execution_options(exp, radar.SimulatedRadarOptions(enabled=True))


# ---------------------------------------------------------------------------
# 2.8.4 review fixes: door admission, named refusals, cancellation,
# ensemble refusal, default-off event identity, disk pricing.
# ---------------------------------------------------------------------------

def _stub_binary(tmp_path, abi_line):
    """An executable answering ``--abi`` with ``abi_line`` (a stale build)."""
    import os
    import stat
    import sys
    script = tmp_path / "stub_simradar.py"
    script.write_text("import sys\nprint(%r)\n" % abi_line, encoding="utf-8")
    if os.name == "nt":
        launcher = tmp_path / "rw_simradar.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        launcher = tmp_path / "rw_simradar"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    return launcher


def _radar_experiment(**table):
    from woof.verify.cases.nest_ideal_r3 import load_scaffold
    options = radar.SimulatedRadarOptions.from_mapping(table)
    return replace(load_scaffold(), simulated_radar=options)


def test_a_missing_override_is_refused_at_the_forecast_door_before_the_fetch(monkeypatch, tmp_path):
    """The binary used to be resolved only when the first history landed:
    after the download, the preparation and a history interval of forecast.
    Every forecast door now asks the preparation inventory first."""
    from woof.config import experiment_preparation_refusals, validate_experiment_preparation
    monkeypatch.setenv("WOOF_RW_SIMRADAR", str(tmp_path / "no-such-rw_simradar"))
    exp = _radar_experiment(sites=["KTLX"])
    refusals = experiment_preparation_refusals(exp)
    assert refusals and refusals[-1][0] == radar.REFUSAL_LABEL
    assert "names a missing file" in refusals[-1][1] and "Next:" in refusals[-1][1]
    with pytest.raises(ValueError, match="names a missing file"):
        validate_experiment_preparation(exp)
    with pytest.raises(ValueError, match="names a missing file"):
        radar.require_admitted(exp)


def test_a_stale_binary_is_refused_at_the_door_naming_the_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOF_RW_SIMRADAR", str(_stub_binary(tmp_path, "rw_simradar --request old/v0")))
    (label, sentence), = radar.door_refusals(_radar_experiment())
    assert label == radar.REFUSAL_LABEL
    assert "different simulated radar request contract" in sentence and "Next:" in sentence


def test_radar_off_asks_the_door_nothing(monkeypatch):
    """Default off: no binary is resolved and no process runs."""
    import subprocess
    from woof.config import experiment_preparation_refusals
    from woof.verify.cases.nest_ideal_r3 import load_scaffold
    def refuse(*args, **kwargs):
        raise AssertionError("radar off must not probe rw_simradar")
    monkeypatch.setattr(subprocess, "run", refuse)
    exp = load_scaffold()
    assert radar.door_refusals(exp) == ()
    assert experiment_preparation_refusals(exp) == ()


def test_host_memory_refusal_is_named_at_the_door(monkeypatch, tmp_path):
    from woof import rustwx
    seen = {}
    monkeypatch.setattr(rustwx, "require_simulated_radar_binary", lambda: tmp_path / "rw_simradar")
    def estimate(paths, *, outdir, config, scene_shapes=(), binary=None):
        seen["shapes"] = list(scene_shapes)
        return {"memory_admitted_now": False,
                "memory_refusal": "radar scan and atmosphere require an estimated 9 bytes, "
                                  "but only 1 host bytes are available"}
    monkeypatch.setattr(rustwx, "estimate_simulated_radar", estimate)
    exp = _radar_experiment()
    (label, sentence), = radar.door_refusals(exp)
    assert "only 1 host bytes" in sentence and "Next:" in sentence
    assert seen["shapes"] == [(d.run.nx, d.run.ny, d.run.nz) for d in exp.domains]


def test_run_experiment_admits_radar_before_any_preparation(monkeypatch, tmp_path):
    from woof import runtime
    import woof.io.wrfout as wrfout
    monkeypatch.setenv("WOOF_RW_SIMRADAR", str(tmp_path / "missing"))
    def reached(*args, **kwargs):
        raise AssertionError("the run passed its radar admission")
    monkeypatch.setattr(wrfout, "quarantine_orphan_wrfouts", reached)
    with pytest.raises(ValueError, match="names a missing file"):
        runtime.run_experiment(_radar_experiment(), None, tmp_path)


def test_replay_door_prints_native_refusals_without_a_traceback(monkeypatch, tmp_path, capsys):
    from woof import rustwx
    from woof.cli import main
    history = tmp_path / "wrfout_d01_2026-10-01_00_00_00"
    history.write_bytes(b"x")
    def refuse(*args, **kwargs):
        raise rustwx.SimulatedRadarRefusal(
            "simulated radar: unknown NEXRAD site ZZZZ; use a custom latitude, longitude, "
            "and antenna height\n" + rustwx.SIMULATED_RADAR_NEXT)
    monkeypatch.setattr(rustwx, "simulate_radar", refuse)
    assert main(["simulated-radar", str(history), "--sites", "ZZZZ", "--outdir", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "unknown NEXRAD site ZZZZ" in err and "Next:" in err
    assert "Traceback" not in err


def test_replay_door_names_a_missing_override_without_a_traceback(monkeypatch, tmp_path, capsys):
    from woof.cli import main
    monkeypatch.setenv("WOOF_RW_SIMRADAR", str(tmp_path / "missing"))
    assert main(["simulated-radar", "--estimate"]) == 2
    err = capsys.readouterr().err
    assert "WOOF_RW_SIMRADAR names a missing file" in err and "Next:" in err
    assert "Traceback" not in err


def test_replay_door_refuses_a_stale_binary_without_a_traceback(monkeypatch, tmp_path, capsys):
    from woof.cli import main
    monkeypatch.setenv("WOOF_RW_SIMRADAR", str(_stub_binary(tmp_path, "rw_simradar --request old/v0")))
    history = tmp_path / "wrfout_d01_2026-10-01_00_00_00"
    history.write_bytes(b"x")
    assert main(["simulated-radar", str(history), "--outdir", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "different simulated radar request contract" in err and "Traceback" not in err


class _Child:
    """A native process stand-in that runs until terminated."""

    def __init__(self):
        self.stopped = threading.Event()
        self.terminated = False

    def poll(self):
        return 0 if self.stopped.is_set() else None

    def terminate(self):
        self.terminated = True
        self.stopped.set()

    def kill(self):
        self.stopped.set()

    def wait(self, timeout=None):
        assert self.stopped.wait(timeout or 5)
        return 0


def test_a_failed_forecast_cancels_queued_radar_and_terminates_the_child(tmp_path, capsys):
    """A forecast failure used to wait for every queued scan (six 1 s jobs
    held the error back 6 s) with the native child left running."""
    import time
    children, calls = [], []
    def native(paths, *, outdir, config, volume_paths, started):
        child = _Child()
        children.append(child)
        calls.append(paths)
        started(child)
        assert child.stopped.wait(30), "the running scan was never terminated"
        raise RuntimeError("terminated")
    began = time.perf_counter()
    with pytest.raises(KeyError):
        with radar.LiveSimulatedRadar(radar.SimulatedRadarOptions(enabled=True), tmp_path,
                                      runner=native) as live:
            for name in "abcdef":
                live.output_committed(domain=1, valid_time=name, path=tmp_path / name)
            while not children:
                time.sleep(0.01)
            raise KeyError("the forecast failed")
    assert time.perf_counter() - began < 10
    assert len(calls) == 1 and children[0].terminated
    assert not live._thread.is_alive()
    assert sorted(Path(p).name for _, p in live.unfinished()) == list("abcdef")
    err = capsys.readouterr().err
    assert "6 committed history file(s) have no radar volume" in err
    assert "woof simulated-radar" in err


def test_cancel_terminates_a_real_native_process(tmp_path):
    import subprocess
    import sys
    import time
    started_at = {}
    def native(paths, *, outdir, config, volume_paths, started):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        started_at["process"] = process
        started(process)
        process.wait()
        raise RuntimeError("terminated")
    live = radar.LiveSimulatedRadar(radar.SimulatedRadarOptions(enabled=True), tmp_path, runner=native)
    live.output_committed(domain=1, valid_time="a", path=tmp_path / "a")
    while "process" not in started_at:
        time.sleep(0.01)
    began = time.perf_counter()
    live.cancel()
    live.close()
    assert time.perf_counter() - began < 15
    assert started_at["process"].poll() is not None, "rw_simradar outlived the cancelled run"


def test_scan_timing_names_the_held_history_when_the_forecast_stops(tmp_path, capsys):
    def native(*args, **kwargs):
        return {}
    options = radar.SimulatedRadarOptions.from_mapping({"timing": "scan"})
    live = radar.LiveSimulatedRadar(options, tmp_path, runner=native)
    live.output_committed(domain=1, valid_time="a", path=tmp_path / "a")
    live.drain()
    live.cancel()
    live.output_committed(domain=1, valid_time="b", path=tmp_path / "b")
    live.close()
    assert [Path(p).name for _, p in live.unfinished()] == ["b", "a"]
    assert "2 committed history file(s)" in capsys.readouterr().err


def test_the_writer_set_cancels_radar_when_the_forecast_raises():
    from woof.io.wrfout import PerDomainWrfoutWriters
    events = []
    writers = object.__new__(PerDomainWrfoutWriters)
    writers._writers = {}
    writers._abort_event = threading.Event()
    writers._simulated_radar = SimpleNamespace(cancel=lambda: events.append("cancel"),
                                               close=lambda: events.append("close"))
    assert writers.__exit__(RuntimeError, RuntimeError("step failed"), None) is False
    assert events == ["cancel", "close"]


def _late_domain_writers(monkeypatch, simulated_radar):
    """A writer set with one late (spawned) domain and a progress observer."""
    import woof.io.wrfout as wrfout
    import woof.runtime as runtime
    built = {}
    class Writer:
        def __init__(self, **kwargs):
            built.update(kwargs)
    monkeypatch.setattr(wrfout, "AsyncDomainWrfoutWriter", Writer)
    monkeypatch.setattr(wrfout, "soil_layer_count", lambda run: 4)
    monkeypatch.setattr(runtime, "_global_wrf_attrs", lambda *a, **k: {})
    monkeypatch.setattr(runtime, "_metadata_frame", lambda *a, **k: {})
    run = SimpleNamespace(nx=4, ny=4, nz=3, dx=1000.0, dy=1000.0)
    node = SimpleNamespace(cfg=SimpleNamespace(start_time=None, run=run, output=None),
                           state=None)
    case = SimpleNamespace(geog_selection=None, initial_result=SimpleNamespace(coord=None))
    writers = object.__new__(wrfout.PerDomainWrfoutWriters)
    writers.model = SimpleNamespace(node=lambda gid: node, _prepared_by_grid_id={2: case},
                                    _feedback_provenance=None)
    writers._writers, writers._metadata_by_grid_id, writers._episode_by_grid_id = {}, {}, {}
    writers.start_time, writers.title = "2026-10-02", "t"
    writers._initial_condition = writers._source = None
    writers._abort_event = threading.Event()
    writers._simulated_radar = simulated_radar
    writers._output_observer = lambda **event: None
    writers.add_domain(2, grid=None, static_fields={})
    return writers, built


def test_a_spawned_nest_reports_no_frames_without_radar_as_in_2_8_3(monkeypatch):
    """2.8.3 built a late domain's writer with no landing observer, so a
    spawned or lifecycle nest wrote no output_committed events. With radar
    off the event stream must stay exactly that."""
    _writers, built = _late_domain_writers(monkeypatch, None)
    assert built["landing_observer"] is None


def test_a_spawned_nest_feeds_radar_when_it_is_asked_for(monkeypatch):
    writers, built = _late_domain_writers(monkeypatch, SimpleNamespace())
    assert built["landing_observer"] == writers._notify_output
    assert built["grid_id"] == 2


def test_ensemble_members_refuse_enabled_radar_before_preparing(monkeypatch, tmp_path):
    """The member leg integrates without the radar landing queue, so an
    enabled table used to be accepted and silently produce nothing."""
    import woof.case_data as case_data
    from woof import runtime
    from woof.ensemble.member import run_member
    from woof.experiment import refuse_unrouted_simulated_radar
    exp = _radar_experiment()
    monkeypatch.setattr(case_data, "load_experiment_case", lambda path: (exp, None))
    def prepared(*args, **kwargs):
        raise AssertionError("the member prepared before refusing")
    monkeypatch.setattr(runtime, "prepare_experiment_case", prepared)
    with pytest.raises(ValueError, match="does not attach \\[simulated_radar\\]"):
        run_member(base_config=tmp_path / "base.toml", member_dir=tmp_path / "m0",
                   index=0, seed=1, perturbation="none")
    refuse_unrouted_simulated_radar(replace(exp, simulated_radar=radar.OFF), "test")


def test_listed_sites_price_radar_into_the_disk_projection(monkeypatch):
    from woof import disk_budget
    from woof.verify.cases.nest_ideal_r3 import load_scaffold
    radar._site_volume_bytes.cache_clear()
    monkeypatch.setattr(radar, "_site_volume_bytes", lambda config: 1000)
    plain = disk_budget.projected_run_bytes(load_scaffold(), keep_checkpoints=None, fetch=None,
                                            chain=None, render=False)
    assert "radar_bytes" not in plain, "radar off projects exactly what it did before"
    exp = _radar_experiment(sites=["KTLX", "KVNX"])
    priced = disk_budget.projected_run_bytes(exp, keep_checkpoints=None, fetch=None,
                                             chain=None, render=False)
    frames = sum(row["history_frames"] for row in priced["domains"])
    assert priced["radar_bytes"] == 1000 * 2 * frames
    assert priced["total_bytes"] == plain["total_bytes"] + priced["radar_bytes"]
    refusal = disk_budget.disk_refusal(priced, 1)
    assert "of simulated radar" in refusal
    auto = disk_budget.projected_run_bytes(_radar_experiment(), keep_checkpoints=None,
                                           fetch=None, chain=None, render=False)
    assert auto["radar_bytes"] == 0
    assert any("sites = \"auto\"" in part for part in auto["unpriced"])


@pytest.mark.parametrize("key,value,reason", [
    ("range_km", 461, "WSR-88D"),
    ("azimuth_step_deg", 721, "zero above 720"),
    ("volume_duration_s", 3601, "0.1 deg/s"),
])
def test_every_geometry_ceiling_names_the_breakage_it_prevents(key, value, reason):
    with pytest.raises(ValueError, match=reason):
        radar.SimulatedRadarOptions.from_mapping({key: value})


def test_gate_spacing_ceiling_is_level_two_s_own_field():
    with pytest.raises(ValueError, match="Message 31"):
        radar.SimulatedRadarOptions.from_mapping({"gate_spacing_m": 40000, "range_km": 230})
    coarse = radar.SimulatedRadarOptions.from_mapping(
        {"gate_spacing_m": 40000, "range_km": 230, "formats": ["cfradial1"]})
    assert coarse.gate_spacing_m == 40000


def test_doctor_gives_every_radar_gap_a_next_command(monkeypatch, tmp_path):
    """The one-line doctor layer showed a radar gap that led nowhere."""
    from woof import doctor, rustwx
    monkeypatch.setattr(rustwx, "simulated_radar_binary", lambda: None)
    missing = doctor._simulated_radar_check()
    assert missing.status == "missing" and missing.action
    monkeypatch.undo()
    monkeypatch.setenv("WOOF_RW_SIMRADAR", str(tmp_path / "missing"))
    override = doctor._simulated_radar_check()
    assert override.status == "missing" and "unset WOOF_RW_SIMRADAR" in override.action
    stub = tmp_path / "rw_simradar"
    stub.write_bytes(b"a build that predates this release's contract")
    monkeypatch.setenv("WOOF_RW_SIMRADAR", str(stub))
    stale = doctor._simulated_radar_check()
    assert stale.status == "missing" and stale.action


# --- A run directory of full frames and 2D quick-look frames (WOOF run_19793057ff2979e3, 2026-10-29) ---

_FULL_FRAME = ("XLAT", "XLONG", "HGT", "T", "PH", "PHB", "W", "U", "V", "REFL_10CM", "T2", "U10", "V10")
#: A woof 2.9.9 quick-look frame (quicklook = { every_s = 300 }, the default history_vars): the surface
#: fields and reflectivity, plus the T and PHB 2.9.9 forced into it, but no PH, W, U or V.
_QUICK_LOOK_FRAME = ("XLAT", "XLONG", "HGT", "T", "PHB", "REFL_10CM", "T2", "U10", "V10", "RAINNC", "RAINC")


def _history(path, names):
    """A small WRF-shaped NetCDF history carrying exactly ``names``."""
    netCDF4 = pytest.importorskip("netCDF4")
    path.parent.mkdir(parents=True, exist_ok=True)
    shapes = {"XLAT": ("Time", "south_north", "west_east"), "XLONG": ("Time", "south_north", "west_east"),
              "HGT": ("Time", "south_north", "west_east"),
              "T": ("Time", "bottom_top", "south_north", "west_east"),
              "REFL_10CM": ("Time", "bottom_top", "south_north", "west_east"),
              "PH": ("Time", "bottom_top_stag", "south_north", "west_east"),
              "PHB": ("Time", "bottom_top_stag", "south_north", "west_east"),
              "W": ("Time", "bottom_top_stag", "south_north", "west_east"),
              "U": ("Time", "bottom_top", "south_north", "west_east_stag"),
              "V": ("Time", "bottom_top", "south_north_stag", "west_east")}
    with netCDF4.Dataset(str(path), "w", format="NETCDF3_64BIT_OFFSET") as ds:
        for dim, size in (("Time", None), ("bottom_top", 2), ("bottom_top_stag", 3), ("south_north", 2),
                          ("south_north_stag", 3), ("west_east", 2), ("west_east_stag", 3)):
            ds.createDimension(dim, size)
        for name in names:
            var = ds.createVariable(name, "f4", shapes.get(name, ("Time", "south_north", "west_east")))
            var[0] = 0.0
    return path


def _mixed_run(root):
    """d01 full frames at 18:00 and 19:00 with 5-min quick-look frames between them, as the writer lays them out."""
    full = [_history(root / f"wrfout_d01_2026-10-28_{t}_00", _FULL_FRAME) for t in ("18_00", "19_00")]
    quick = [_history(root / "quicklook" / f"wrfout_d01_2026-10-28_18_{m}_00", _QUICK_LOOK_FRAME)
             for m in ("05", "10")]
    return full, quick


def _admitting_native(calls):
    """A stand-in for rw_simradar that refuses a 2D frame as its header admission does (input_contract.rs)."""
    netCDF4 = pytest.importorskip("netCDF4")
    from woof.io.netcdf_serialization import netcdf4_session

    def native(paths, *, outdir, config, volume_paths, started):
        for path in paths:
            # rw_simradar reads in its own process; in-process, this stand-in obeys the process's netCDF4 lock.
            with netcdf4_session(), netCDF4.Dataset(str(path)) as ds:
                lacking = [n for n in ("PH", "W", "U", "V") if n not in ds.variables]
            if lacking:
                raise RuntimeError(f"simulated radar: radar_input_missing_columns: {Path(path).name} lacks "
                                   f"variables [{', '.join(lacking)}]")
        calls.append([Path(p).name for p in paths])
        return {}
    return native


def test_history_missing_columns_reads_the_frame_header(tmp_path):
    full, quick = _mixed_run(tmp_path)
    assert radar.history_missing_columns(full[0]) == ()
    assert radar.history_missing_columns(quick[0]) == ("PH", "U", "V", "W")
    unreadable = tmp_path / "wrfout_d01_2026-10-28_20_00_00"
    unreadable.write_bytes(b"not a NetCDF file")
    assert radar.history_missing_columns(unreadable) is None, "the native admission decides what it cannot read"


def test_live_radar_scans_full_frames_and_skips_quick_look_frames(tmp_path, capsys):
    """run_19793057ff2979e3: the first 5-min quick-look frame reached the radar, which refused it, and the
    stored refusal stopped the whole forecast. Committed in the forecast's own order, full and quick-look frames
    interleaved, only the full frames are scanned and the run's radar completes."""
    full, quick = _mixed_run(tmp_path)
    calls = []
    with radar.LiveSimulatedRadar(radar.SimulatedRadarOptions(enabled=True), tmp_path,
                                  runner=_admitting_native(calls)) as live:
        for path in (full[0], quick[0], quick[1], full[1]):
            live.output_committed(domain=1, valid_time=path.name[11:], path=path)
            live.check()
        live.drain()
    assert calls == [[full[0].name], [full[1].name]]
    assert [Path(p).name for _, p, _ in live.skipped()] == [q.name for q in quick]
    err = capsys.readouterr().err
    assert "2 history file(s) carry no vertical atmosphere" in err and "d01 x2" in err


def test_scan_timing_pairs_full_frames_across_quick_look_frames(tmp_path):
    full, quick = _mixed_run(tmp_path)
    calls = []
    options = radar.SimulatedRadarOptions.from_mapping({"timing": "scan"})
    with radar.LiveSimulatedRadar(options, tmp_path, runner=_admitting_native(calls)) as live:
        for path in (full[0], quick[0], quick[1], full[1]):
            live.output_committed(domain=1, valid_time=path.name[11:], path=path)
    assert calls == [[full[0].name, full[1].name], [full[1].name]]


def test_a_run_with_no_full_frame_refuses_naming_what_its_frames_lack(tmp_path):
    _, quick = _mixed_run(tmp_path)
    calls = []
    live = radar.LiveSimulatedRadar(radar.SimulatedRadarOptions(enabled=True), tmp_path,
                                    runner=_admitting_native(calls))
    for path in quick:
        live.output_committed(domain=1, valid_time=path.name[11:], path=path)
    with pytest.raises(RuntimeError, match=r"none of the 2 history file\(s\) carries .* lacks PH, U, V, W"):
        live.close()
    assert calls == []


def test_replay_cli_on_a_run_directory_scans_only_full_frames(monkeypatch, tmp_path, capsys):
    """`woof simulated-radar RUN_DIR`, the recovery path for a run's radar, walks quicklook/ too."""
    from woof import rustwx
    full, quick = _mixed_run(tmp_path / "run")
    seen = []
    monkeypatch.setattr(rustwx, "simulate_radar", lambda paths, **k: seen.append(list(paths)) or {}, raising=False)
    parser = argparse.ArgumentParser()
    radar.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["simulated-radar", str(tmp_path / "run"), "--sites", "KTLX", "--outdir", str(tmp_path)])
    assert args.func(args) == 0
    assert seen == [[p.resolve() for p in full]]
    assert "2 history file(s) carry no vertical atmosphere" in capsys.readouterr().err
