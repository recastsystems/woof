"""The go door must launch native namelist inputs through their complete reader."""
from pathlib import Path

import pytest


def test_go_native_directory_uses_wrf_execution_and_preserves_controls(tmp_path, monkeypatch):
    from woof import capabilities, cli, provenance_gate, wrfinput_forecast

    directory = tmp_path / "native"
    directory.mkdir()
    (directory / "namelist.input").write_text("&physics bl_pbl_physics=99 /\n")
    (directory / "wrfinput_d01").touch()
    from types import SimpleNamespace
    import datetime
    monkeypatch.setattr("woof.wrfinput_door.resolve_wrfinput_run", lambda _: SimpleNamespace(
        experiment=SimpleNamespace(start_time=datetime.datetime(2023, 3, 31))))
    calls = []
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a: None)
    monkeypatch.setattr(provenance_gate, "announce", lambda *a: None)
    monkeypatch.setattr(wrfinput_forecast, "run_wrf_forecast",
                        lambda *a, **kw: calls.append((a, kw)) or 17)
    out = tmp_path / "forecast"
    assert cli.main(["go", str(directory), "--outdir", str(out),
                     "--run-seconds", "10800", "--run-stamp", "off", "--no-supervise",
                     "--allow-shared-gpu", "--checkpoint-at", "end"]) == 17
    args, controls = calls[0]
    assert args == (directory, out)
    assert controls["run_seconds"] == 10800
    assert controls["exclusive_gpu"] is False
    assert controls["allow_shared_gpu"] is True
    assert controls["checkpoint_at"] == "end"


@pytest.mark.parametrize('use_flag', [False, True])
def test_go_native_directory_refuses_unconsumed_source_flags(tmp_path, monkeypatch, use_flag):
    from woof import capabilities, cli

    directory = tmp_path / "native"
    directory.mkdir()
    (directory / "namelist.input").write_text("&physics bl_pbl_physics=99 /\n")
    (directory / "wrfinput_d01").touch()
    from types import SimpleNamespace
    import datetime
    monkeypatch.setattr("woof.wrfinput_door.resolve_wrfinput_run", lambda _: SimpleNamespace(
        experiment=SimpleNamespace(start_time=datetime.datetime(2023, 3, 31))))
    monkeypatch.setattr(capabilities, "require_for_command",
                        lambda *a: pytest.fail("invalid flag reached GPU admission"))
    with pytest.raises(SystemExit) as error:
        native = ["--wrfinput", str(directory)] if use_flag else [str(directory)]
        cli.main(["go", *native, "--source", "hrrr"])
    assert error.value.code == 2


def test_go_toml_keeps_its_source_preparation_contract():
    from woof import cli

    args = cli.build_parser(go_only=True).parse_args(["go", "case.toml"])
    assert args.command == "go"
    assert args.config == Path("case.toml")


def test_go_metem_preserves_visual_controls_through_real_run_parser(tmp_path, monkeypatch):
    import os
    from woof import capabilities, cli, metem_forecast, provenance_gate

    directory = tmp_path / "metgrid"
    directory.mkdir()
    (directory / "namelist.input").write_text("&physics bl_pbl_physics=16 /\n")
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a: None)
    monkeypatch.setattr(provenance_gate, "announce", lambda *a: None)
    monkeypatch.setenv("WOOF_VERIFY_VISUALS", "1")
    calls = []

    def forecast(*args, **controls):
        calls.append((args, controls, os.environ.get("WOOF_VERIFY_VISUALS")))
        return 23

    monkeypatch.setattr(metem_forecast, "run_metem_forecast", forecast)
    out = tmp_path / "forecast"
    assert cli.main(["go", str(directory), "--outdir", str(out),
                     "--run-seconds", "120", "--no-supervise",
                     "--allow-shared-gpu", "--products", "none",
                     "--no-verify-visuals"]) == 23
    args, controls, visuals = calls[0]
    assert args == (directory, out)
    assert controls["run_seconds"] == 120
    assert controls["exclusive_gpu"] is False
    assert controls["allow_shared_gpu"] is True
    assert visuals == "0"
    assert os.environ["WOOF_VERIFY_VISUALS"] == "1"
