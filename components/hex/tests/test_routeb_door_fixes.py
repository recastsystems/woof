"""Host contracts for MPAS stream consumers and failure timeline receipts."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from woof.hex import global_cycle, history_writer, observation_scorecard, stage_timing, time_ledger
from woof.hex.mpasphys import runtime_v842


def native_request(run, template='output.$Y-$M-$D_$h.$m.$s.nc'):
    (run/'native-physics-request.json').write_text(json.dumps({
        'schema':'mpas-native-door-v1', 'start':'2026-10-16_00:00:00',
        'grid':'/prepared/grid.nc', 'init':'/prepared/init.nc',
        'streams':[{'name':'output', 'filename':template, 'directions':['output']}]}))


@pytest.mark.parametrize('prefix', ['output', 'history', 'custom'])
def test_scorer_reads_native_provenance_and_stream(prefix, tmp_path):
    native_request(tmp_path, f'{prefix}.$Y-$M-$D_$h.$m.$s.nc')
    for hour in (3, 0):
        (tmp_path/f'{prefix}.2026-10-16_{hour:02}.00.00.nc').write_bytes(b'header inventory fixture')
    config, frames = observation_scorecard.discover(tmp_path)
    assert config['init'] == '/prepared/init.nc'
    assert config['grid'] == '/prepared/grid.nc'
    assert config['start'] == '2026-10-16_00:00:00'
    assert [f['valid_time'] for f in frames] == ['2026-10-16T00:00:00Z', '2026-10-16T03:00:00Z']
    assert all(Path(f['history']).name.startswith(prefix) for f in frames)


@pytest.mark.parametrize('ensemble', [False, True])
def test_cycle_requires_durable_native_frames(tmp_path, ensemble):
    run = tmp_path/('ensemble/g0/mem001' if ensemble else 'forecast')
    run.mkdir(parents=True)
    native_request(run, 'weather.$Y-$M-$D_$h.$m.$s.nc')
    closed = run/'weather.2026-10-16_03.00.00.nc'
    closed.write_bytes(b'closed native stream stand-in')
    (run/'weather.2026-10-16_06.00.00.nc').write_bytes(b'not certified')
    history_writer.mark_closed_frame_durable(closed)
    rows = global_cycle.frames(tmp_path, datetime(2026,10,16,tzinfo=timezone.utc))
    assert len(rows) == 1 and rows[0]['lead_hours'] == 3.0
    assert rows[0].get('member') == ('mem001' if ensemble else None)
    marker = json.loads(history_writer.durable_marker(closed).read_text())
    assert marker['bytes'] == closed.stat().st_size
    assert marker['mtime_ns'] == closed.stat().st_mtime_ns
    closed.write_bytes(b'failed rerun overwrote the certified file')
    assert global_cycle.frames(tmp_path, datetime(2026,10,16,tzinfo=timezone.utc)) == []


def test_failed_file_sync_does_not_certify_native_frame(tmp_path, monkeypatch):
    path = tmp_path/'output.2026-10-16_00.00.00.nc'
    path.write_bytes(b'new contents')
    marker = history_writer.durable_marker(path)
    marker.write_text('{}')
    def fail(fd):
        raise OSError('controlled sync failure')
    monkeypatch.setattr(history_writer.os, 'fsync', fail)
    with pytest.raises(OSError, match='controlled sync failure'):
        history_writer.mark_closed_frame_durable(path)
    assert not marker.exists()
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('fail', [False, True])
def test_stage_pairs_ledger_events_and_records_failure(tmp_path, monkeypatch, fail):
    events = []
    monkeypatch.setattr(time_ledger, 'emit', lambda name, phase, **kw: events.append((name,phase,kw)))
    rows = []
    def operation():
        with stage_timing.stage(rows, 'host-fixture', parallelism={'device':'CPU'}, timeline='prep.native.fixture'):
            if fail:
                raise ValueError('controlled stage failure')
    if fail:
        with pytest.raises(ValueError, match='controlled stage failure'):
            operation()
    else:
        operation()
    assert [e[1] for e in events] == ['begin','end']
    assert rows[0]['status'] == ('failed' if fail else 'done')
    assert rows[0]['wall_seconds'] >= 0
    assert events[-1][2]['attrs']['status'] == rows[0]['status']


@pytest.mark.parametrize('fail', [False, True])
def test_runtime_persists_early_timing_and_closes_door_spans(tmp_path, monkeypatch, fail):
    time_ledger.configure(tmp_path)
    # Exercise the real wrapper, never construct a native or CUDA session.
    def run(*args, **kwargs):
        if fail:
            raise RuntimeError('controlled pre-session failure')
        return 0
    monkeypatch.setattr(runtime_v842, '_run_native_dynamics', run)
    for name in ('prep.door','forecast.construct','forecast.integrate'):
        time_ledger.emit(name, 'begin')
    args = [None]*10 + [tmp_path]
    if fail:
        with pytest.raises(RuntimeError, match='controlled pre-session failure'):
            runtime_v842.run_native_dynamics(*args)
    else:
        assert runtime_v842.run_native_dynamics(*args) == 0
    stages = json.loads((tmp_path/'native-runtime-timings.json').read_text())['stages']
    assert stages[0]['status'] == ('failed' if fail else 'done')
    assert stages[0]['ended_utc'] and stages[0]['wall_seconds'] >= 0
    assert not time_ledger._open


@pytest.mark.parametrize('fail', [False, True])
def test_forecast_certifies_streams_only_after_runtime_return(tmp_path, monkeypatch, fail):
    from types import SimpleNamespace
    from woof.hex import licence_gate
    from woof.hex.mpasphys import forecast_v842 as forecast
    providers = {}
    # The streams deck is parsed as XML before the run (rrfs/match deck_stream_inputs), so its placeholder is an empty deck.
    for name in ('library','reader','source','init','streams'):
        path = tmp_path/name
        path.write_bytes(b'<streams/>' if name == 'streams' else b'provider path fixture')
        providers[name] = path
    out = tmp_path/'out'
    arguments = SimpleNamespace(native_physics_library=providers['library'],
        native_pool_reader=providers['reader'], mpas_source=providers['source'],
        init=providers['init'], streams=providers['streams'], out=out, hours=1)
    config = SimpleNamespace(table_key='fixture', options={'config_sst_update':False}, receipt=lambda:{})
    layout = SimpleNamespace(configs={'config_start_time':'2026-10-16_00:00:00','config_dt':600})
    monkeypatch.setattr(licence_gate, 'native_library_refusal', lambda *a:None)
    monkeypatch.setattr(forecast, 'load_table', lambda:{'fixture':{}})
    import woof.hex.physics_suites as physics_suites
    monkeypatch.setattr(physics_suites, 'load_table', lambda:{'fixture':{}})
    monkeypatch.setattr(forecast, 'bind_init_start', lambda text, path:text)
    monkeypatch.setattr(forecast, 'decode_registry_pools', lambda *a, **k: layout)  # rrfs/suites passes initial_state
    stream = forecast.Stream('output', ('output',), 'output.$Y-$M-$D_$h.$m.$s.nc', 'none','01:00:00',())
    monkeypatch.setattr(forecast, 'read_streams', lambda *a, **k:(stream,))
    frame = out/'output.2026-10-16_01.00.00.nc'
    def run(*args, **kwargs):
        frame.write_bytes(b'closed-stream stand-in, not a native weather proof')
        assert not history_writer.durable_marker(frame).exists()
        if fail:
            raise RuntimeError('controlled stream failure')
        return 0
    monkeypatch.setattr(runtime_v842, 'run_native_dynamics', run)
    if fail:
        with pytest.raises(RuntimeError, match='controlled stream failure'):
            forecast.run_forecast(arguments, config, '')
        assert not history_writer.durable_marker(frame).exists()
    else:
        assert forecast.run_forecast(arguments, config, '') == 0
        assert history_writer.durable_marker(frame).is_file()


@pytest.mark.parametrize('marker', ['[]', '{}', '{broken'])
def test_cycle_ignores_invalid_native_durability_receipt(tmp_path, marker):
    run = tmp_path/'forecast'
    run.mkdir()
    native_request(run)
    frame = run/'output.2026-10-16_00.00.00.nc'
    frame.write_bytes(b'uncertified fixture')
    history_writer.durable_marker(frame).write_text(marker)
    assert global_cycle.frames(tmp_path, datetime(2026,10,16,tzinfo=timezone.utc)) == []


def test_native_timeline_names_pass_real_c1_taxonomy():
    import ast
    from woof.timeline.taxonomy import valid_stage
    from woof.hex.mpasphys import forecast_v842
    names = []
    for module in (forecast_v842, runtime_v842):
        tree = ast.parse(Path(module.__file__).read_text())
        names.extend(keyword.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                     for keyword in node.keywords if keyword.arg == 'timeline'
                     and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str))
    assert len(names) >= 10
    assert all(valid_stage(name) for name in names), names
