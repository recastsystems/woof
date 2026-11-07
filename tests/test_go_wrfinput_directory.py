"""Stock WRF GO keeps products, verification and native runner controls."""
import datetime
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import cli, go_cli


@pytest.fixture
def native_runner(tmp_path, monkeypatch):
    from woof import capabilities, provenance_gate, wrfinput_door, wrfinput_forecast
    (tmp_path / "namelist.input").write_text("&time_control\n/\n")
    (tmp_path / "wrfinput_d01").touch()
    monkeypatch.setattr(capabilities, "require_for_command", lambda *args: None)
    monkeypatch.setattr(provenance_gate, "announce", lambda *args: None)
    run = SimpleNamespace(experiment=SimpleNamespace(
        start_time=datetime.datetime(1974, 4, 3, 12), run_seconds=10800),
        grid_ids=[1], namelist_input=tmp_path / "namelist.input")
    monkeypatch.setattr(wrfinput_door, "resolve_wrfinput_run", lambda *args, **kwargs: run)
    calls = []

    def forecast(*args, **kwargs):
        from woof.resume import KEEP_CHECKPOINTS_ENV
        calls.append((args, kwargs, os.environ.get("WOOF_VERIFY_VISUALS"),
                      os.environ.get(KEEP_CHECKPOINTS_ENV)))
        return 17

    monkeypatch.setattr(wrfinput_forecast, "run_wrf_forecast", forecast)
    return calls


@pytest.mark.parametrize("form", ["directory", "flag"])
@pytest.mark.parametrize("visuals", [None, "0", "1"])
def test_public_native_go_preserves_controls(tmp_path, monkeypatch, native_runner,
                                            form, visuals):
    from woof.resume import KEEP_CHECKPOINTS_ENV
    if visuals is None:
        monkeypatch.delenv("WOOF_VERIFY_VISUALS", raising=False)
    else:
        monkeypatch.setenv("WOOF_VERIFY_VISUALS", visuals)
    monkeypatch.setenv(KEEP_CHECKPOINTS_ENV, "3")
    source = ["--wrfinput", str(tmp_path)] if form == "flag" else [str(tmp_path)]
    output, checkpoint = tmp_path / "out", tmp_path / "restart.npz"
    command = ["go", *source, "--outdir", str(output), "--run-stamp", "off",
               "--products", "none", "--allow-shared-gpu", "--no-supervise",
               "--run-seconds", "10800", "--checkpoint-at", "end", "--benchmark",
               "--restart", str(checkpoint), "--keep-checkpoints", "2"]
    assert cli.main(command) == 17
    assert native_runner == [((tmp_path, output), {
        "restart": checkpoint, "render_products": "none", "exclusive_gpu": False,
        "benchmark": True, "checkpoint_at": "end", "run_seconds": 10800.0,
        "allow_shared_gpu": True}, visuals, "2")]
    assert os.environ.get("WOOF_VERIFY_VISUALS") == visuals
    assert os.environ[KEEP_CHECKPOINTS_ENV] == "3"
    assert not output.exists()  # The runner owns output admission.


@pytest.mark.parametrize("form", ["directory", "flag"])
def test_explicit_visual_setting_is_scoped(tmp_path, monkeypatch, native_runner, form):
    monkeypatch.setenv("WOOF_VERIFY_VISUALS", "1")
    source = ["--wrfinput", str(tmp_path)] if form == "flag" else [str(tmp_path)]
    assert cli.main(["go", *source, "--products", "none", "--no-verify-visuals",
                     "--run-stamp", "off"]) == 17
    assert native_runner[0][2] == "0"
    assert os.environ["WOOF_VERIFY_VISUALS"] == "1"


@pytest.mark.parametrize("form", ["directory", "flag"])
def test_native_identity_overrides_reach_review_and_runner(tmp_path, monkeypatch,
                                                          native_runner, form):
    from woof import wrfinput_door
    original = wrfinput_door.resolve_wrfinput_run
    reviewed = []

    def resolve(*args, **kwargs):
        reviewed.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(wrfinput_door, "resolve_wrfinput_run", resolve)
    monkeypatch.setenv("GPUWM_HISTORY_COMPRESSION", "deflate")
    source = ["--wrfinput", str(tmp_path)] if form == "flag" else [str(tmp_path)]
    restart = tmp_path / "wrfrst_d01"
    assert cli.main(["go", *source, "--run-stamp", "off", "--explain",
                     "--history-compression", "none", "--members", "1",
                     "--keep-member-files", "--restart-from", str(restart),
                     "--rrtmg-variant", "rrtmg_legacy", "--thompson-version",
                     "wrf_461"]) == 17
    identity = {"restart_from": restart, "rrtmg_variant": "rrtmg_legacy",
                "thompson_version": "wrf_461"}
    assert reviewed == [((tmp_path,), identity)]
    passed = native_runner[0][1]
    assert {name: passed[name] for name in identity} == identity
    assert passed["ensemble_request"]["members"] == 1
    assert passed["ensemble_request"]["keep_member_files"] is True
    assert os.environ["GPUWM_HISTORY_COMPRESSION"] == "none"


@pytest.mark.parametrize("form", ["directory", "flag"])
def test_one_trajectory_cannot_silently_become_distinct_members(tmp_path, monkeypatch,
                                                               form):
    from woof import wrfinput_door
    (tmp_path / "namelist.input").write_text("&time_control /\n")
    (tmp_path / "wrfinput_d01").touch()
    monkeypatch.setattr(wrfinput_door, "resolve_wrfinput_run",
                        lambda *args, **kwargs: pytest.fail("must refuse before input IO"))
    source = ["--wrfinput", str(tmp_path)] if form == "flag" else [str(tmp_path)]
    args = cli.build_parser(go_only=True).parse_args(["go", *source, "--members", "4"])
    with pytest.raises(ValueError, match="trajectory|member"):
        go_cli.go_main(args)


@pytest.mark.parametrize("source", ["directory", "flag"])
@pytest.mark.parametrize("control", [("--prepared-root", "/unused"),
                                     ("--source", "gfs"), ("--devices", "2")])
def test_native_route_refuses_unused_controls_before_input_io(tmp_path, monkeypatch,
                                                             source, control):
    from woof import wrfinput_door
    (tmp_path / "namelist.input").write_text("&time_control /\n")
    (tmp_path / "wrfinput_d01").touch()
    monkeypatch.setattr(wrfinput_door, "resolve_wrfinput_run",
                        lambda *args: pytest.fail("must refuse before input IO"))
    tokens = ["--wrfinput", str(tmp_path)] if source == "flag" else [str(tmp_path)]
    args = cli.build_parser(go_only=True).parse_args(["go", *tokens, *control])
    with pytest.raises(go_cli.GoRefusal, match="would ignore the requested setting"):
        go_cli.go_main(args)


def test_config_source_launch_retains_existing_owner(monkeypatch):
    seen = []
    monkeypatch.setattr(go_cli, "_go_config_main",
                        lambda args, **kwargs: seen.append((args, kwargs)) or 19)
    args = cli.build_parser(go_only=True).parse_args(["go", "--source", "gfs"])
    assert go_cli.go_main(args) == 19
    assert seen == [(args, {"observer": None})]


@pytest.mark.parametrize("control", [("--benchmark",), ("--run-seconds", "60"),
                                     ("--no-supervise",), ("--allow-shared-gpu",)])
def test_config_route_refuses_native_only_controls(control):
    args = cli.build_parser(go_only=True).parse_args(["go", "forecast.toml", *control])
    with pytest.raises(ValueError, match="WRF|configuration launches"):
        go_cli.go_main(args)


def test_native_dry_run_checks_duration_without_launching(tmp_path, monkeypatch,
                                                          native_runner, capsys):
    from woof import wrfinput_forecast
    windows = []
    monkeypatch.setattr(wrfinput_forecast, "wrfinput_window_seconds",
                        lambda run, seconds: windows.append(seconds) or seconds)
    assert cli.main(["go", str(tmp_path), "--outdir", str(tmp_path / "out"),
                     "--run-seconds", "60", "--dry-run", "--run-stamp", "off"]) == 0
    assert windows == [60.0]
    assert not native_runner and not (tmp_path / "out").exists()
    assert '"route": "native-wrf"' in capsys.readouterr().out


def test_parser_registers_native_controls_once():
    args = cli.build_parser().parse_args(["go", "--wrfinput", "/native",
                                          "--checkpoint-at", "end"])
    assert args.wrfinput == Path("/native") and args.checkpoint_at == "end"
