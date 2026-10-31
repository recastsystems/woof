"""Feature composition through the actual forecast entry and resident runtime."""
import json
from types import SimpleNamespace as NS

import pytest

from woof.hex.mpasphys import features_v842 as features, forecast_v842 as door
from test_mpasphys_checkpoint_v842 import TEXT, write_checkpoint
from test_mpasphys_runtime_execution_v842 import harness

# The suite row's route blocks (framework, resident, services), read from the shipped table so these
# fakes keep the pieces route_table admits by row (rrfs/nf-glue made admission table-driven).
import json as _json, pathlib as _pathlib
_V842_ROW = _json.loads((_pathlib.Path(__file__).resolve().parents[1] / '..' / '..' / 'woof' / 'hex' / 'data' / 'profiles'
                        / 'mpas-physics-suites.v1.json').read_text())['mpasphys_v842']
_V842_ROUTE = {key: _V842_ROW[key] for key in ('framework', 'resident', 'services') if key in _V842_ROW}



@pytest.mark.parametrize('scenario,default_streams', [
    ('ordinary', False), ('checkpoint', False), ('restart', False),
    ('analysis', False), ('distributed', False),
    ('ordinary', True), ('checkpoint', True),
])
def test_feature_entry_preserves_clock_numerics_and_stream_owner(
        monkeypatch, tmp_path, scenario, default_streams):
    provider = tmp_path / 'provider.input'
    # Also the run's streams deck when one is named: parsed as XML before the run (rrfs/match deck_stream_inputs).
    provider.write_bytes(b'<streams/>')
    resumed = scenario in ('restart', 'analysis')
    flags = {}
    if resumed:
        flags['restart_from'] = write_checkpoint(tmp_path)[0]
    if scenario == 'analysis':
        flags['da_analysis'] = tmp_path / 'analysis.nc'
    if scenario == 'checkpoint':
        flags['checkpoint_hours'] = '0.1,0.2'
    if scenario == 'distributed':
        flags['samehost_devices'] = '0,1'
    arguments = NS(native_physics_library=provider, native_pool_reader=provider,
        mpas_source=tmp_path, init=provider,
        streams=None if default_streams else provider, out=tmp_path/'out',
        hours=0.2, history_every_minutes=6, **flags)
    config = NS(pin_set='v8.4.2', table_key='mpasphys_v842',
                backends={}, service_backends={}, active=lambda slot: False,
                options={'config_sst_update': False}, receipt=lambda: {})
    expected_start = '2026-10-17_00:06:00' if resumed else '2026-10-17_00:00:00'
    decoded, stream_calls, dispatched = [], [], []

    def decode(reader, source, config, text, sources, payload, initial_state=None, beside_transport=None):
        assert door.namelist_option(text, 'nhyd_model', 'config_start_time') == expected_start
        if resumed:
            assert door.namelist_option(text, 'restart', 'config_do_restart') is False
        decoded.append(text)
        return NS(configs=dict(config_start_time=expected_start, config_dt=60,
            config_run_duration='00:12:00', config_do_restart=False,
            config_do_DAcycling=False))

    def default(layout, path, **kwargs):
        stream_calls.append(kwargs)
        return provider

    def dispatch(*args, **kwargs):
        dispatched.append((args, kwargs))
        return 17

    monkeypatch.setattr('woof.hex.licence_gate.native_library_refusal', lambda *args: None)
    monkeypatch.setattr('woof.hex.fullstep_lts_admission.moving_region_requested', lambda: False)
    monkeypatch.setattr(door, 'load_table', lambda: {'mpasphys_v842': dict(_V842_ROUTE)})
    monkeypatch.setattr(door, 'decode_registry_pools', decode)
    monkeypatch.setattr(door, 'read_streams', lambda *args, **kwargs: ())
    monkeypatch.setattr(door, 'write_default_streams', default)
    monkeypatch.setattr(features, 'run_native_dynamics', dispatch)
    assert door.run_forecast(arguments, config, TEXT) == 17
    assert len(decoded) == len(dispatched) == 1
    args, kwargs = dispatched[0]
    assert args[6:] == (expected_start, '2026-10-17_00:12:00', 60.0,
                       6 if resumed else 12, arguments.out)
    assert args[2].configs['config_do_restart'] is resumed
    assert args[2].configs['config_do_DAcycling'] is (scenario == 'analysis')
    if scenario == 'checkpoint':
        assert kwargs['moving'].checkpoint_seconds == (360, 720)
    if default_streams:
        # The native checkpoint carrier owns these CLI checkpoints. The
        # default-stream helper must not create a second checkpoint stream.
        assert stream_calls == [dict(history_minutes=6, checkpoint_hours=None, omit=None, bound_inputs={})]
    else:
        assert stream_calls == []
    request = json.loads((arguments.out/'native-physics-request.json').read_text())
    from woof.hex import numerics
    assert request['numerics'] == numerics.receipt_block()
    stages = json.loads((arguments.out/'stage-timings.json').read_text())['stages']
    assert [stage['stage'] for stage in stages] == ['native-pool-decode', 'native-forecast-runtime']
    # U-H3: the decode row names the reader's real worker count, never a fixed one thread.
    from woof.hex.mpasphys.pool_v842 import pool_transport_threads
    assert stages[0]['parallelism']['threads'] == pool_transport_threads()
    assert stages[1]['parallelism']['cards'] == (2 if scenario == 'distributed' else 1)


@pytest.mark.parametrize('resident', [False, True])
def test_real_cycling_executor_publishes_before_background(monkeypatch, tmp_path, resident):
    h = harness(monkeypatch, tmp_path, resident=resident, injected=True)
    h.arguments.out = h.out
    h.arguments.jedi_background = True
    h.arguments.history_every_minutes = 1/6  # The fixture has ten-second steps.
    calls = []

    def recouple(arguments, layout, inputs, **kwargs):
        assert arguments is h.arguments and layout is h.layout
        updated = (*inputs[:3], h.replacements[0], inputs[4], h.replacements[1], inputs[6])
        return updated, {'fixture_analysis': True}

    def background(arguments, session, layout, names, *, step, dt, out):
        assert session is h.session and session.native is h.native
        assert h.initialized and h.clock == step and dt == 10
        # Real CyclingExecution forces publication at the otherwise deferred
        # middle boundary, using this lane's resident session.publish call.
        assert ('publish', step) in h.events
        assert h.events.index(('publish', step)) < h.events.index(('native-output', step))
        calls.append(step)
        return NS(path=f'background.{step}', receipt_path=f'background.{step}.json', sha256='fixture')

    monkeypatch.setattr(features, 'recouple_analysis', recouple)
    monkeypatch.setattr(features, 'write_background', background)
    h.execution = features.CyclingExecution(h.arguments, h.layout, h.inputs)
    assert h.run() == 0
    assert calls == [0, 1, 2]
    receipt = json.loads((h.out/'native-physics-receipt.json').read_text())
    assert receipt['completed'] and receipt['resident_physics']['enabled'] is resident
    assert receipt['native_publication'] == dict(lazy=True, published=3, deferred=0)
    assert receipt['rank_execution']['analysis'] == {'fixture_analysis': True}
    assert [row['step'] for row in receipt['rank_execution']['backgrounds']] == calls
