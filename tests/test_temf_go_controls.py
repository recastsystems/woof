"""Native GO retains products, restart and resource controls together."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_native_go_calls_shared_runner_with_products_and_native_controls(monkeypatch, tmp_path):
    from woof import cli, wrfinput_door, wrfinput_forecast
    calls = []
    monkeypatch.setattr('woof.capabilities.require_for_command', lambda *args: None)
    monkeypatch.setattr('woof.provenance_gate.announce', lambda *args: None)
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run', lambda _: SimpleNamespace(
        experiment=SimpleNamespace(start_time=datetime(1974, 4, 3, 12))))
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast',
                        lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    out = tmp_path/'forecast'
    assert cli.main(['go', '--wrfinput', str(tmp_path/'inputs'), '--outdir', str(out),
                     '--products', 'none', '--no-verify-visuals', '--run-stamp', 'off',
                     '--run-seconds', '10800', '--checkpoint-at', 'end',
                     '--benchmark', '--no-supervise', '--allow-shared-gpu',
                     '--restart', str(tmp_path/'checkpoint')]) == 0
    (positional, controls), = calls
    assert positional == (tmp_path/'inputs', out)
    assert controls == dict(restart=tmp_path/'checkpoint', allow_shared_gpu=True,
                            exclusive_gpu=False, benchmark=True, run_seconds=10800.,
                            checkpoint_at='end', render_products='none')
    assert not out.exists()


@pytest.mark.parametrize('flags', [
    ['--prepared-root', '/wrong'], ['--source', 'gfs'], ['--devices', '2'],
    ['--prepare-only'], ['another.toml'],
])
def test_unused_native_go_controls_refuse_before_loading_inputs(monkeypatch, flags):
    from woof import cli, wrfinput_door
    from woof.go_cli import GoRefusal, go_main
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run',
                        lambda _: pytest.fail('must refuse before input IO'))
    args = cli.build_parser(go_only=True).parse_args(['go', '--wrfinput', '/absent', *flags])
    with pytest.raises(GoRefusal):
        go_main(args)


def test_native_dry_run_checks_window_without_creating_output(monkeypatch, tmp_path):
    from woof import cli, wrfinput_door, wrfinput_forecast
    from woof.wrfinput_go import launch
    out = tmp_path/'forecast'
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run', lambda _: SimpleNamespace(
        experiment=SimpleNamespace(start_time=datetime(1974, 4, 3, 12)),
        grid_ids=[1], namelist_input=Path('/inputs/namelist.input')))
    windows = []
    monkeypatch.setattr(wrfinput_forecast, 'wrfinput_window_seconds',
                        lambda run, seconds: windows.append(seconds) or seconds)
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast',
                        lambda *args, **kwargs: pytest.fail('dry run launched forecast'))
    args = cli.build_parser(go_only=True).parse_args([
        'go', '--wrfinput', '/inputs', '--outdir', str(out), '--run-stamp', 'off',
        '--dry-run', '--run-seconds', '10800'])
    assert launch(args) == 0
    assert windows == [10800.]
    assert not out.exists()


def test_toml_go_refuses_native_only_controls():
    from woof import cli
    from woof.go_cli import go_main
    args = cli.build_parser(go_only=True).parse_args(['go', 'forecast.toml', '--benchmark'])
    with pytest.raises(ValueError, match='would not consume'):
        go_main(args)
