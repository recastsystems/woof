"""The native go route must execute the existing native runner and own its flags."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.go_cli import GoRefusal, _add_go_arguments, go_main


def args(*tokens):
    parser = argparse.ArgumentParser()
    _add_go_arguments(parser)
    return parser.parse_args(tokens)


def resolved():
    from datetime import datetime
    return SimpleNamespace(experiment=SimpleNamespace(start_time=datetime(2024, 5, 25, 18),
                            run_seconds=21600), grid_ids=(1,), namelist_input=Path('case/namelist.input'))


def test_native_go_dry_run_validates_without_creating_output(tmp_path, monkeypatch, capsys):
    import woof.wrfinput_door as door
    monkeypatch.setattr(door, 'resolve_wrfinput_run', lambda p: resolved())
    target = tmp_path / 'output'
    assert go_main(args('--wrfinput', 'case', '--outdir', str(target), '--dry-run')) == 0
    assert not target.exists()
    assert '"route": "native-wrf"' in capsys.readouterr().out


def test_native_go_dispatches_real_runner_and_keeps_execution_controls(tmp_path, monkeypatch):
    import woof.wrfinput_door as door
    import woof.wrfinput_forecast as runner
    monkeypatch.setattr(door, 'resolve_wrfinput_run', lambda p: resolved())
    calls = []
    def launch(*pos, **kw):
        calls.append((pos, kw))
        return 17
    monkeypatch.setattr(runner, 'run_wrf_forecast', launch)
    assert go_main(args('--wrfinput', 'case', '--outdir', str(tmp_path),
                        '--no-supervise', '--benchmark', '--products', 'none')) == 17
    pos, kw = calls.pop()
    assert pos[0] == Path('case') and pos[1].parent == tmp_path
    assert pos[1].name.startswith('run-')
    assert kw['exclusive_gpu'] is False and kw['benchmark'] is True
    assert kw['render_products'] == 'none'


@pytest.mark.parametrize('tokens', [('--cycle', '2024-05-25T18'), ('--prepare-only',),
                                     ('--smoke',)])
def test_native_go_refuses_unconsumed_controls_before_reading_inputs(tokens, monkeypatch):
    import woof.wrfinput_door as door
    def unread(path):
        pytest.fail('a refused route read its inputs')
    monkeypatch.setattr(door, 'resolve_wrfinput_run', unread)
    with pytest.raises(GoRefusal, match='do not consume'):
        go_main(args('--wrfinput', 'case', *tokens))


def test_native_controls_require_native_inputs():
    with pytest.raises(GoRefusal, match='WRF-input execution'):
        go_main(args('--benchmark'))
    with pytest.raises(GoRefusal, match='remove CONFIG'):
        go_main(args('case.toml', '--wrfinput', 'case'))


def test_native_go_keeps_the_memory_gate_override(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from woof import wrfinput_door, wrfinput_forecast
    from woof.core import resident_admission
    seen = []
    @contextmanager
    def override(value):
        seen.append(value)
        yield
    monkeypatch.setattr(resident_admission, 'memory_gate_override', override)
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run', lambda _: resolved())
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast', lambda *a, **kw: 17)
    assert go_main(args('--wrfinput', 'case', '--outdir', str(tmp_path),
                        '--no-memory-gate')) == 17
    assert seen == [True]
