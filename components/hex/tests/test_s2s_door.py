"""Public S2S plan, true lag clocks and interrupted-member recovery."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from woof.hex import s2s_door
from woof.hex.cli import build_parser
from woof.hex.forecast_door import _schedule


def arguments(tmp_path, *extra):
    return build_parser().parse_args(["s2s", "--row", "global-120km", "--source", "gfs",
        "--cycle", "2026-10-16T00", "--members", "4", "--days", "30", "--work-dir", str(tmp_path), *extra])


def test_cpu_budget_respects_affinity_and_native_thread_limit(monkeypatch):
    monkeypatch.setattr(s2s_door.os, 'sched_getaffinity', lambda _: set(range(8)))
    monkeypatch.setenv('RAYON_NUM_THREADS', '3')
    assert s2s_door.cores() == 3
    monkeypatch.setenv('RAYON_NUM_THREADS', '14')
    assert s2s_door.cores() == 8
    for invalid in ('0', '-1', 'many'):
        monkeypatch.setenv('RAYON_NUM_THREADS', invalid)
        with pytest.raises(s2s_door.S2SError, match='positive integer'):
            s2s_door.cores()


def test_dry_run_public_door_has_no_side_effects(tmp_path):
    destination = tmp_path / "not-created"
    argv = [sys.executable, "-m", "woof.hex.cli", "s2s", "--row", "global-120km", "--source", "gfs",
            "--cycle", "2026-10-16T00", "--members", "4", "--days", "30", "--work-dir", str(destination), "--dry-run"]
    result = subprocess.run(argv, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert not destination.exists()
    assert len(plan["members"]) == 4
    assert [m["lag_hours"] for m in plan["members"]] == [0, 6, 12, 18]
    last = plan["members"][-1]
    assert last["start_time"] == last["valid_time"] == "2026-10-16_00:00:00"
    assert last["source_cycle"] == "2026-10-15_06:00:00" and last["lead_hours"] == 18
    assert last["init_argv"][-2:] == ["--lead-hours", "18"]
    assert last["hours"] == 720
    assert last["history_from_minutes"] == 0
    assert last["checkpoint_hours"] == "120,240,360,480,600,720"
    assert plan["surface_updating"] and plan["stochastic"]["sppt"] and plan["stochastic"]["skeb"]
    assert plan["surface"]["source"] == "member initialization SST/ice"
    assert plan["surface"]["forecast_cycle"] is None
    assert plan["surface"]["end"] == "2026-11-15_00:00:00"
    assert plan["surface"]["path"] is None
    assert set(plan["parallelism"]) == set(s2s_door.STAGES)


def test_lagged_daily_history_has_common_valid_times():
    assert _schedule(738, 1440, 600, "global", 18 * 60) == (4428, 31)
    with pytest.raises(Exception, match="does not divide"):
        _schedule(737, 1440, 600, "global", 18 * 60)
    with pytest.raises(Exception, match="whole step"):
        _schedule(738, 1440, 600, "global", 0.5)


def test_bad_days_and_lags_refuse(tmp_path):
    # A 10-day subseasonal run plans (it used to be refused by a 30..90 bound
    # that named no breakage); verification covers the whole short run.
    ten = s2s_door.plan(arguments(tmp_path, "--days", "10"))
    assert ten["days"] == 10 and ten["parallelism"]["verify"]["coverage_days"] == 10
    assert s2s_door.plan(arguments(tmp_path, "--days", "29"))["days"] == 29
    with pytest.raises(s2s_door.S2SError, match="week-1"):
        s2s_door.plan(arguments(tmp_path, "--days", "6"))
    for bad in ("-6", "1", "0,0", "junk"):
        with pytest.raises(s2s_door.S2SError):
            s2s_door.lag_hours(bad, 4, "gfs")


def test_checkpoint_search_retains_member_identity(tmp_path):
    a = tmp_path / "a1" / "mem001"
    b = tmp_path / "a2" / "mem001"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    first = a / "checkpoint.2026-10-21_00.00.00.pkl"
    latest = b / "checkpoint.2026-10-26_00.00.00.pkl"
    first.write_bytes(b"checkpoint")
    latest.write_bytes(b"checkpoint")
    (b / "checkpoint.2026-10-31_00.00.00.pkl.part").write_bytes(b"incomplete")
    attempts = [{"out": str(a.parent)}, {"out": str(b.parent)}]
    assert s2s_door.latest_checkpoint({"run_id": "mem001"}, attempts) == latest
    assert s2s_door.latest_checkpoint({"run_id": "mem002"}, attempts) is None


def test_signature_allows_different_card_placement(tmp_path):
    a = s2s_door.plan(arguments(tmp_path, "--devices", "0"))
    b = s2s_door.plan(arguments(tmp_path, "--devices", "0,1"))
    assert a["signature"] == b["signature"]


def test_forecast_child_uses_pool_uuid_for_logical_device_zero(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from woof.hex import global_ensemble
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-pool-allocation")
    args = arguments(tmp_path, "--members", "1", "--lagged", "0", "--devices", "0",
                     "--ocean-mode", "prescribed")
    value = s2s_door.plan(args)
    source = Path(value["surface"]["path"])
    source.write_bytes(b"native surface series")
    state = {"completed": [], "attempts": [], "stages": {}}
    calls = []
    def child(argv, log, env):
        calls.append(env["CUDA_VISIBLE_DEVICES"])
        return 0
    monkeypatch.setattr(global_ensemble, "_cli", child)
    monkeypatch.setattr(global_ensemble, "group_limit", lambda *a: 1)
    status = SimpleNamespace(stage=lambda *a, **kw: None, step=lambda *a, **kw: None)
    result = s2s_door.forecasts(value, args, {}, state, tmp_path / "state.json", status)
    assert calls == result["cards"] == ["GPU-pool-allocation"]
    assert state["attempts"][0]["card"] == "GPU-pool-allocation"
    assert state["completed"] == ["mem001"]


def test_products_and_verification_resolve_the_original_device_allocation(tmp_path, monkeypatch):
    import os
    from woof.hex import s2s_products
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setattr(s2s_door, "STAGES", ("products", "verify"))
    calls = []
    def stage(*args):
        calls.append(os.environ["CUDA_VISIBLE_DEVICES"])
        return {}
    monkeypatch.setattr(s2s_products, "build_products", stage)
    monkeypatch.setattr(s2s_products, "verify_first_ten_days", stage)
    from woof.hex import s2s_energy
    monkeypatch.setattr(s2s_energy, "build_energy_products", stage)
    assert s2s_door.run(arguments(tmp_path, "--devices", "1")) == 0
    assert calls == ["GPU-b", "GPU-b", "GPU-b"]


def test_interruption_after_final_checkpoint_restarts_before_end(tmp_path):
    root = tmp_path / "attempt" / "mem001"
    root.mkdir(parents=True)
    earlier = root / "checkpoint.2026-10-26_00.00.00.pkl"
    earlier.write_bytes(b"day-25")
    (root / "checkpoint.2026-10-31_00.00.00.pkl").write_bytes(b"day-30")
    member = {"run_id": "mem001", "start_time": "2026-10-01_00:00:00", "hours": 720}
    assert s2s_door.latest_checkpoint(member, [{"out": str(root.parent)}]) == earlier


def test_native_members_balanced_across_lags(tmp_path):
    value = s2s_door.plan(arguments(tmp_path, "--source", "gefs", "--lagged", "0,24"))
    assert [(m["member"], m["lag_hours"]) for m in value["members"]] == [(1, 0), (1, 24), (2, 0), (2, 24)]


def test_lagged_source_forecasts_reach_one_clock_and_manifest(tmp_path):
    from datetime import datetime, timedelta
    value = s2s_door.plan(arguments(tmp_path, "--lagged", "0,6,12,18"))
    common = datetime(2026, 10, 16)
    s2s_door.update_manifests(value, {"attempts": [], "completed": []})
    for member in value["members"]:
        source_time = datetime.strptime(member["source_cycle"], "%Y-%m-%d_%H:%M:%S")
        assert source_time + timedelta(hours=member["lead_hours"]) == common
        assert member["start_time"] == common.strftime("%Y-%m-%d_%H:%M:%S")
        manifest = json.loads(Path(member["manifest"]).read_text())
        assert manifest["provenance"] == member["provenance"]
        assert manifest["source_member"] is None
        assert manifest["valid_time"] == member["valid_time"]


def test_analysis_only_source_defaults_to_earlier_analyses_with_stochastic_fill(tmp_path):
    value = s2s_door.plan(arguments(tmp_path, "--source", "era5"))
    assert [m["lag_hours"] for m in value["members"]] == [0, 6, 12, 18]
    assert all(m["lead_hours"] == 0 for m in value["members"])
    assert len({m["init"] for m in value["members"]}) == 4
    assert all(m["stochastic"]["sppt"] and m["stochastic"]["skeb"] for m in value["members"])


@pytest.mark.parametrize("source", ["era5", "era5-arco"])
def test_analysis_lags_keep_input_clocks_and_common_output_window(tmp_path, source):
    from datetime import datetime, timedelta
    from woof.hex.forecast_door import _schedule
    value = s2s_door.plan(arguments(tmp_path, "--source", source, "--members", "8",
        "--lagged", "0,6,12,18,24,30,36,42"))
    s2s_door.update_manifests(value, {"attempts": [], "completed": []})
    for m in value["members"]:
        start = datetime.strptime(m["start_time"], "%Y-%m-%d_%H:%M:%S")
        lag = m["lag_hours"]
        assert start + timedelta(hours=lag) == datetime(2026, 10, 16)
        assert start + timedelta(hours=m["hours"]) == datetime(2026, 11, 15)
        assert m["valid_time"] == m["source_cycle"] == m["start_time"]
        assert m["provenance"]["window_start"] == value["start_time"]
        assert "--lead-hours" not in m["init_argv"]
        assert _schedule(m["hours"], 1440, 600, "global", m["history_from_minutes"])[1] == 31
        assert m["checkpoint_hours"].split(",")[0] == str(lag + 120)
        manifest = json.loads(Path(m["manifest"]).read_text())
        assert manifest["provenance"] == m["provenance"]
    assert value["surface"]["start"] == "2026-10-14_06:00:00"


def test_analysis_hourly_cadence_and_explicit_common_stochastic_fill(tmp_path):
    value = s2s_door.plan(arguments(tmp_path, "--source", "era5", "--lagged", "0,1,2,3"))
    assert [m["history_from_minutes"] for m in value["members"]] == [0, 60, 120, 180]
    repeated = s2s_door.plan(arguments(tmp_path, "--source", "era5", "--lagged", "0"))
    assert len({m["init"] for m in repeated["members"]}) == 1
    assert len({m["stochastic"]["member"] for m in repeated["members"]}) == 4
    with pytest.raises(s2s_door.S2SError, match="repeated source input"):
        s2s_door.plan(arguments(tmp_path, "--source", "era5", "--lagged", "0",
            "--stochastic", '{"sppt":false,"skeb":false}'))


def test_prepared_path_cannot_hide_another_members_source_binding(tmp_path):
    from types import SimpleNamespace
    path = tmp_path / "prepared.nc"
    path.write_bytes(b"analysis")
    args = arguments(tmp_path / "run", "--source", "era5", "--members", "2", "--lagged", "0,6",
        "--prepared-init", f"2026-10-16T00={path}", "--prepared-init", f"2026-10-15T18={path}")
    value = s2s_door.plan(args)
    # The later binding is valid. Deduplicating paths before checking each
    # member used to discard the earlier, invalid binding entirely.
    last = value["members"][-1]
    s2s_door.write_json(tmp_path / "init-source-receipt.json", {
        "source": "era5", "member": None, "cycle": last["source_cycle"], "lead_hours": 0,
        "valid_time": last["valid_time"]})
    status = SimpleNamespace(stage=lambda *a, **kw: None, step=lambda *a, **kw: None)
    with pytest.raises(s2s_door.S2SError, match="source/cycle/member mismatch"):
        s2s_door._inits(value, args, {}, status)


def test_stochastic_policy_is_resolved_before_card_groups(tmp_path):
    four = s2s_door.plan(arguments(tmp_path))
    assert all(m["stochastic"]["sppt"] and m["stochastic"]["skeb"] for m in four["members"])
    assert len({m["stochastic"]["member"] for m in four["members"]}) == 4
    assert len({m["stochastic"]["seed"] for m in four["members"]}) == 4
    thirty = s2s_door.plan(arguments(tmp_path, "--source", "gefs", "--members", "30"))
    assert thirty["stochastic"]["sppt"] and thirty["stochastic"]["skeb"]
    assert s2s_door.science_arguments(four)[0] == "--stochastic"
    s2s_door.update_manifests(four, {"attempts": [], "completed": []})
    manifest = json.loads(Path(four["members"][0]["manifest"]).read_text())
    assert manifest["stochastic"] == json.loads(json.dumps(four["members"][0]["stochastic"]))


def test_explicit_stochastic_controls_reach_each_member(tmp_path):
    value = s2s_door.plan(arguments(tmp_path, "--stochastic", '{"amplitude":0.25,"skeb":false}'))
    assert all(m["stochastic"]["amplitude"] == 0.25 and not m["stochastic"]["skeb"]
               for m in value["members"])
    with pytest.raises(s2s_door.S2SError, match="roster"):
        s2s_door.plan(arguments(tmp_path, "--stochastic", '{"member":7}'))


def test_resident_members_discover_selected_surface_series(tmp_path, monkeypatch):
    from woof.hex.surface_series_reader import discover
    source = tmp_path / "prepared.nc"
    source.write_bytes(b"native surface series")
    output = tmp_path / "attempts" / "a0001"
    for key, value in s2s_door.surface_environment({"surface": {"path": str(source)}}).items():
        monkeypatch.setenv(key, value)
    assert discover(output / "mem001").resolve() == source
    assert not output.exists()
    monkeypatch.setenv("WOOF_HEX_SURFACE_SERIES", str(tmp_path / "missing.nc"))
    with pytest.raises(FileNotFoundError, match="selected surface series"):
        discover(output / "mem001")


def test_surface_restart_uses_fresh_destination_and_reuses_complete_receipt(tmp_path, monkeypatch):
    from woof.hex import global_ensemble, s2s_energy
    monkeypatch.setattr(s2s_energy, "check_forcing", lambda value: {"objects": 121})
    args = arguments(tmp_path, "--ocean-mode", "prescribed")
    value = s2s_door.plan(args)
    assert Path(value["surface"]["path"]).name == "surface_series.nc"
    legacy = tmp_path / "surface-series.nc"
    legacy.write_bytes(b"existing series")
    assert s2s_door.plan(args)["surface"]["path"] == str(legacy)
    explicit = tmp_path / "selected.nc"
    assert s2s_door.plan(arguments(tmp_path, "--ocean-mode", "prescribed", "--surface-series", str(explicit)))["surface"]["path"] == str(explicit)
    commands = []
    def producer(argv, log, env):
        commands.append(argv)
        if len(commands) == 1:
            return 1
        Path(argv[argv.index("--out") + 1]).write_bytes(b"native-series")
        s2s_door.write_json(Path(argv[argv.index("--receipt") + 1]), {"status": "complete"})
        return 0
    monkeypatch.setattr(global_ensemble, "_cli", producer)
    with pytest.raises(s2s_door.S2SError, match="surface-series exited 1"):
        s2s_door.surface_stage(value, args, {})
    result = s2s_door.surface_stage(value, args, {})
    assert not result["reused"]
    assert commands[0][commands[0].index("--out")+1] != commands[1][commands[1].index("--out")+1]
    assert "--forecast-cycle" in commands[1]
    assert s2s_door.surface_stage(value, args, {})["reused"]
    assert len(commands) == 2
    receipt = Path(value["surface"]["path"]).with_suffix(".receipt.json")
    saved = json.loads(receipt.read_text())
    assert saved.pop("object_coverage") == {"objects": 121}
    s2s_door.write_json(receipt, saved)
    assert s2s_door.surface_stage(value, args, {})["object_coverage"] == {"objects": 121}
    assert json.loads(receipt.read_text())["object_coverage"] == {"objects": 121}


def test_resume_reuses_init_with_source_lane_clock_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from woof.hex import global_ensemble
    args = arguments(tmp_path, "--members", "1", "--lagged", "0")
    value = s2s_door.plan(args)
    member = value["members"][0]
    init = Path(member["init"])
    init.parent.mkdir(parents=True)
    init.write_bytes(b"prepared")
    s2s_door.write_json(init.parent / "init-source-receipt.json", {
        "init": {"path": str(init)}, "source": "gfs", "member": None,
        "cycle": member["start_time"], "lead_hours": 0, "valid_time": member["start_time"]})
    def unexpected(*args, **kwargs):
        raise AssertionError("a matching init must not be rebuilt during resume")
    monkeypatch.setattr(global_ensemble, "_cli", unexpected)
    status = SimpleNamespace(stage=lambda *args, **kwargs: None, step=lambda *args, **kwargs: None)
    result = s2s_door._inits(value, args, {}, status)
    assert result["built"] == 0 and result["reused"] == 1


def test_prepared_lagged_init_checks_source_lead_and_common_valid_time(tmp_path):
    from types import SimpleNamespace
    init = tmp_path / "shared/init.nc"
    init.parent.mkdir()
    init.write_bytes(b"prepared-source-forecast")
    args = arguments(tmp_path / "run", "--members", "1", "--lagged", "6",
                     "--prepared-init", f"2026-10-15T18={init}")
    value = s2s_door.plan(args)
    receipt = {"source": "gfs", "member": None, "cycle": "2026-10-15_18:00:00",
               "source_cycle": "2026-10-15_18:00:00", "lead_hours": 6,
               "valid_time": "2026-10-16_00:00:00"}
    path = init.parent / "init-source-receipt.json"
    status = SimpleNamespace(stage=lambda *args, **kwargs: None, step=lambda *args, **kwargs: None)
    s2s_door.write_json(path, receipt)
    assert s2s_door._inits(value, args, {}, status)["reused"] == 1
    for changed in ({"lead_hours": 0}, {"valid_time": receipt["source_cycle"]},
                    {"source_cycle": "2026-10-15_12:00:00"}):
        s2s_door.write_json(path, {**receipt, **changed})
        with pytest.raises(s2s_door.S2SError, match="mismatch"):
            s2s_door._inits(value, args, {}, status)



def test_prepared_init_survives_producer_cleanup_same_inode(tmp_path):
    from types import SimpleNamespace
    source = tmp_path / "producer/init.nc"
    source.parent.mkdir()
    source.write_bytes(b"real-input-placeholder-for-file-lifetime-test")
    options = arguments(tmp_path / "run", "--members", "1", "--lagged", "0",
                        "--prepared-init", f"2026-10-16T00={source}")
    value = s2s_door.plan(options)
    s2s_door.write_json(source.parent / "init-source-receipt.json", {
        "source": "gfs", "member": None, "cycle": "2026-10-16_00:00:00"})
    status = SimpleNamespace(stage=lambda *a, **k: None, step=lambda *a, **k: None)
    s2s_door._inits(value, options, {}, status)
    retained = Path(value["members"][0]["init"])
    assert source.stat().st_ino == retained.stat().st_ino
    source.unlink()
    (source.parent / "init-source-receipt.json").unlink()
    assert retained.read_bytes() == b"real-input-placeholder-for-file-lifetime-test"
    assert s2s_door._inits(value, options, {}, status)["reused"] == 1


def test_successful_recovery_clears_the_persisted_previous_error(tmp_path, monkeypatch):
    from woof.hex import global_door
    args = arguments(tmp_path, '--stage', 'mesh')
    value = s2s_door.plan(args)
    state_path = tmp_path / 's2s-state.json'
    s2s_door.write_json(state_path, {'schema': s2s_door.SCHEMA,
        'signature': value['signature'], 'attempts': [], 'completed': [], 'stages': {},
        'error': 'S2SError: previous interrupted attempt'})
    monkeypatch.setattr(global_door, 'stage_mesh', lambda *args: {'reused': True})
    assert s2s_door.run(args) == 0
    assert 'error' not in json.loads(state_path.read_text())
    assert 'error' not in json.loads((tmp_path/'s2s-receipt.json').read_text())


def test_latest_cycle_stays_pinned_when_resuming_after_a_new_cycle_posts(tmp_path, monkeypatch, capsys):
    from woof.hex import global_cycle, global_door
    resolutions = []

    def resolve(*args, **kwargs):
        cycle = "2026-10-16T00" if not resolutions else "2026-10-16T06"
        resolutions.append(cycle)
        return {"cycle": cycle}

    calls = []

    def mesh(*args):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("interrupted mesh stage")
        return {"reused": True}

    monkeypatch.setattr(global_cycle, "resolve_latest", resolve)
    monkeypatch.setattr(global_door, "stage_mesh", mesh)
    with pytest.raises(RuntimeError, match="interrupted mesh stage"):
        s2s_door.run(arguments(tmp_path, "--cycle", "latest", "--stage", "mesh"))
    original = json.loads((tmp_path / "s2s-plan.json").read_text())
    assert s2s_door.run(arguments(tmp_path, "--cycle", "latest", "--stage", "mesh")) == 0
    resumed = json.loads((tmp_path / "s2s-plan.json").read_text())
    assert original["cycle"] == resumed["cycle"] == "2026-10-16T00"
    assert original["signature"] == resumed["signature"]
    assert resolutions == ["2026-10-16T00"]
    assert "error" not in json.loads((tmp_path / "s2s-state.json").read_text())
    # Preview the restart without contacting the provider or changing state.
    files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    capsys.readouterr()
    assert s2s_door.run(arguments(tmp_path, "--cycle", "latest", "--dry-run")) == 0
    assert json.loads(capsys.readouterr().out)["signature"] == original["signature"]
    assert files == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert resolutions == ["2026-10-16T00"]
    with pytest.raises(s2s_door.S2SError, match="different S2S configuration"):
        s2s_door.run(arguments(tmp_path, "--cycle", "latest", "--stage", "mesh", "--days", "45"))
    assert len(calls) == 2


def test_latest_cycle_is_not_resolved_before_work_directory_admission(tmp_path, monkeypatch):
    import fcntl
    from woof.hex import global_cycle

    def unexpected(*args, **kwargs):
        raise AssertionError("a second command must not contact the source while the work directory is locked")

    monkeypatch.setattr(global_cycle, "resolve_latest", unexpected)
    with (tmp_path / "s2s.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(s2s_door.S2SError, match="another s2s command holds"):
            s2s_door.run(arguments(tmp_path, "--cycle", "latest", "--stage", "mesh"))


def test_prepared_init_survives_producer_cleanup(tmp_path):
    from types import SimpleNamespace
    source = tmp_path / "producer/init.nc"
    source.parent.mkdir()
    source.write_bytes(b"prepared-source-forecast")
    args = arguments(tmp_path / "run", "--members", "1", "--lagged", "0",
                     "--prepared-init", f"2026-10-16T00={source}")
    value = s2s_door.plan(args)
    member = value["members"][0]
    owned = Path(member["init"])
    assert owned != source
    receipt = {"source": "gfs", "member": None, "cycle": "2026-10-16_00:00:00",
               "source_cycle": "2026-10-16_00:00:00", "lead_hours": 0,
               "valid_time": "2026-10-16_00:00:00"}
    receipt_path = source.parent / "init-source-receipt.json"
    s2s_door.write_json(receipt_path, receipt)
    status = SimpleNamespace(stage=lambda *args, **kwargs: None, step=lambda *args, **kwargs: None)
    s2s_door._inits(value, args, {}, status)
    assert owned.stat().st_ino == source.stat().st_ino
    source.unlink()
    receipt_path.unlink()
    assert owned.read_bytes() == b"prepared-source-forecast"
    assert s2s_door._inits(value, args, {}, status)["reused"] == 1
    assert s2s_door.plan(args)["signature"] == value["signature"]


def test_mpas_route_members_get_the_door_ocean(tmp_path):
    # U-H1 (best-180 smoke j1, 2026-10-27): on the MPAS physics route (the hex default) the door's ocean reached the
    # forecast only as environment the ArWen route reads, so config_sst_update stayed false and SST and sea ice were
    # bit-identical from day 0 to day 7 with --ocean-mode prescribed --surface-series and with the prognostic default.
    from woof.hex.mpasphys.forecast_v842 import native_route
    series = tmp_path / "series.nc"
    series.write_bytes(b"dated native surface records")
    def resolved(*extra):
        value = s2s_door.plan(arguments(tmp_path / "work", "--members", "1", "--days", "7", *extra))
        forecast = build_parser().parse_args(value["commands"]["forecast"])
        config, _ = native_route(forecast, default_mode="correct")
        return value, forecast, config
    value, forecast, config = resolved("--ocean-mode", "prescribed", "--surface-series", str(series))
    assert config.options["config_sst_update"] and not config.options["config_oml1d"]
    assert str(config.service_backends["surface_update"]) == "hex-cuda"
    assert forecast.stream_input == [f"surface={series.absolute()}"]
    assert value["surface"]["forecast_route"] == "mpas"
    value, forecast, config = resolved()
    assert config.options["config_oml1d"] and not config.options["config_sst_update"]
    assert config.options["config_oml_hml0"] == value["surface"]["ocean_options"]["depth_m"]
    assert config.options["config_oml_gamma"] == value["surface"]["ocean_options"]["stratification_k_m"]
    assert "ice" in value["surface"]["sea_ice"]
    with pytest.raises(s2s_door.S2SError, match="never read"):
        resolved("--ocean-options", '{"ice_albedo": 0.6}')
    # A surface stream the caller already bound to the same series is kept once; another file is refused.
    value, forecast, _ = resolved("--ocean-mode", "prescribed", "--surface-series", str(series),
                                  f"--forecast-arg=--stream-input=surface={series}")
    assert forecast.stream_input == [f"surface={series}"]
    with pytest.raises(s2s_door.S2SError, match="give the series once"):
        resolved("--ocean-mode", "prescribed", "--surface-series", str(series),
                 f"--forecast-arg=--stream-input=surface={tmp_path / 'other.nc'}")
    # The ArWen route keeps reading the series through its own environment, unchanged.
    value = s2s_door.plan(arguments(tmp_path / "arwen", "--members", "1", "--days", "7", "--physics-row", "arwen",
                                    "--ocean-mode", "prescribed", "--surface-series", str(series)))
    assert "--stream-input" not in value["commands"]["forecast"] and "forecast_route" not in value["surface"]


def test_default_streams_read_a_bound_surface_series_at_its_interval(tmp_path):
    # U-H1: MPAS's Registry surface stream has input_interval="none"; the default streams take the series' spacing.
    from woof.hex.mpasphys import resolve_namelist
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    text = '&physics\nconfig_physics_suite="convection_permitting"\nconfig_sst_update=.true.\n/'
    layout = RegistryLayout(registry_source(), resolve_namelist(text), text,
        dict(nCells=2, nEdges=3, nVertices=2, maxEdges=6, maxEdges2=12, vertexDegree=3, nVertLevels=55))
    plain = {s.name: s for s in read_streams(write_default_streams(layout, tmp_path / "a" / "streams.atmosphere"), layout)}
    timed = {s.name: s for s in read_streams(write_default_streams(layout, tmp_path / "b" / "streams.atmosphere",
                                                                   surface_interval="1_00:00:00"), layout)}
    assert plain["surface"].input_interval == "none"
    assert timed["surface"].input_interval == "1_00:00:00"
    assert timed["surface"].fields == plain["surface"].fields and "sst" in timed["surface"].fields
