"""The required stock-namelist go route reaches the production WRF owner."""
import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.go_cli import GoRefusal, _add_go_arguments, go_main


def args(*words):
    p = argparse.ArgumentParser()
    _add_go_arguments(p)
    return p.parse_args(words)


def resolved_run():
    return SimpleNamespace(experiment=SimpleNamespace(
        start_time=datetime(2000, 1, 24, 12), run_seconds=10800))


def test_go_wrfinput_calls_the_existing_forecast_owner(monkeypatch, tmp_path):
    from woof import wrfinput_forecast, wrfinput_door
    calls = []
    monkeypatch.setattr(wrfinput_door, "resolve_wrfinput_run", lambda p: resolved_run())
    monkeypatch.setattr(wrfinput_forecast, "wrfinput_window_seconds", lambda r, s: s)
    monkeypatch.setattr(wrfinput_forecast, "run_wrf_forecast",
                        lambda *a, **kw: calls.append((a, kw)) or 17)
    source, output = tmp_path / "inputs", tmp_path / "output"
    result = go_main(args("--wrfinput", str(source), "--outdir", str(output),
                          "--run-stamp", "off", "--run-seconds", "10800", "--benchmark"))
    assert result == 17
    assert calls[0][0] == (source, output)
    assert calls[0][1]["run_seconds"] == 10800
    assert calls[0][1]["benchmark"] is True


@pytest.mark.parametrize("flag,value", [("--prepared-root", "bundle"),
                                        ("--source", "gfs"),
                                        ("--tiles", "auto"),
                                        ("--nudging", "grid")])
def test_wrfinput_never_silently_ignores_prepared_options(flag, value):
    with pytest.raises(GoRefusal, match=flag):
        go_main(args("--wrfinput", "inputs", flag, value))


def test_conflicting_input_routes_refuse_before_opening_either():
    with pytest.raises(GoRefusal, match="select exactly one input owner"):
        go_main(args("missing.toml", "--wrfinput", "inputs"))


def test_wrf_only_controls_are_refused_on_existing_config_routes():
    with pytest.raises(GoRefusal, match="WRF-input execution"):
        go_main(args("missing.toml", "--benchmark"))


def test_missing_wrf_inputs_do_not_allocate_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises((ValueError, FileNotFoundError)):
        go_main(args("--wrfinput", str(tmp_path / "absent"), "--outdir", str(output)))
    assert not output.exists()


def test_stamped_wrf_launches_allocate_distinct_run_folders(monkeypatch, tmp_path):
    from woof import wrfinput_forecast, wrfinput_door
    calls = []
    monkeypatch.setattr(wrfinput_door, "resolve_wrfinput_run", lambda p: resolved_run())

    def forecast(src, dest, **kwargs):
        # GO reserves a name without making output before admission. The
        # forecast owner creates the accepted output and records its stamp.
        assert not dest.exists()
        dest.mkdir(parents=True)
        calls.append(dest)
        return 0

    monkeypatch.setattr(wrfinput_forecast, "run_wrf_forecast", forecast)
    output = tmp_path / "output"
    for _ in range(2):
        assert go_main(args("--wrfinput", str(tmp_path / "inputs"),
                            "--outdir", str(output))) == 0
    assert len(set(calls)) == 2
    assert all(p.is_dir() for p in calls)
