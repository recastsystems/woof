"""The go door must execute imported WRF physics and reject unread flags."""
import argparse
from pathlib import Path
import pytest
from woof import go_cli

@pytest.fixture(autouse=True)
def resolved_bundle(monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace
    import woof.wrfinput_door as door
    monkeypatch.setattr(door, "resolve_wrfinput_run", lambda *a, **k: SimpleNamespace(experiment=SimpleNamespace(start_time=datetime(2000,1,24,12))))


def _args(*tokens):
    parser = argparse.ArgumentParser()
    go_cli._add_go_arguments(parser)
    return parser.parse_args(['--wrfinput', '/case', '--outdir', '/output', '--run-stamp', 'off', *tokens])


def test_wrfinput_delegates_to_the_real_forecast_door(monkeypatch):
    from woof import wrfinput_forecast
    calls = []
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return 17
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast', run)
    assert go_cli._go_wrfinput(_args('--products', '2m_temperature')) == 17
    assert calls[0][0] == (Path('/case'), Path('/output'))
    assert calls[0][1]['render_products'] == '2m_temperature'


@pytest.mark.parametrize('tokens', [('--prepared-root', '/prepared'),
                                    ('--prepare-only',), ('--cycle', 'latest'),
                                    ('/config.toml',)])
def test_wrfinput_refuses_options_it_cannot_read(tokens):
    with pytest.raises(go_cli.GoRefusal, match='remove CONFIG|do not consume'):
        go_cli._go_wrfinput(_args(*tokens))


@pytest.mark.parametrize('spelling', ['--wrfinput', '--wrfinput='])
def test_go_stock_input_route_keeps_forecast_controls(monkeypatch, tmp_path, spelling):
    from woof.cli import main
    import woof.wrfinput_forecast as door
    calls = []
    monkeypatch.setattr(door, 'run_wrf_forecast', lambda *a, **kw: calls.append((a, kw)) or 0)
    source, output = tmp_path / 'native', tmp_path / 'gpu'
    args = ['go'] + ([spelling, str(source)] if spelling == '--wrfinput'
                     else [spelling + str(source)])
    args += ['--outdir', str(output), '--run-seconds', '21600', '--no-supervise',
             '--allow-shared-gpu', '--benchmark', '--checkpoint-at', 'end', '--run-stamp', 'off']
    assert main(args) == 0
    (directory, outdir), kwargs = calls.pop()
    assert (directory, outdir) == (source, output)
    assert kwargs['run_seconds'] == 21600
    assert kwargs['exclusive_gpu'] is False
    assert kwargs['allow_shared_gpu'] is True
    assert kwargs['benchmark'] is True
    assert kwargs['checkpoint_at'] == 'end'
