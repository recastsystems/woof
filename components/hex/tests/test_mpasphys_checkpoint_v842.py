"""Native sealed checkpoint export, restore and original-clock admission."""
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from woof.hex.mpasphys.checkpoint_v842 import resolve_route
from woof.hex.mpasphys.forecast_v842 import namelist_option
from woof.hex.mpasphys.moving_mpi_v842 import PoolWords, HostArrays, WindowWords, ServiceWords
from woof.hex.mpasphys.moving_v842 import STATE_FIELDS, SAVED_FIELDS


TEXT = "&nhyd_model\nconfig_start_time='2026-10-17_00:00:00'\n/\n"
CONFIG = SimpleNamespace(receipt=lambda: dict(suite='convection_permitting', options={}))


def member():
    mesh = SimpleNamespace(arrays=dict(
        areaCell=np.arange(3, dtype=np.float32)+1,
        cellsOnEdge=np.array([[0, 1], [1, 2], [2, 0], [0, 2]], np.int32),
        edgesOnCell=np.array([[0, 2, 3], [0, 1, 2], [1, 2, 3]], np.int32)))
    grid = SimpleNamespace(vertical_grid=SimpleNamespace(
        zgrid=np.arange(9, dtype=np.float32).reshape(3, 3),
        dzw=np.ones((2, 3), np.float32)))
    reference = SimpleNamespace(**{name: np.full((2, 3), index+1, np.float32)
        for index, name in enumerate(('rho_base', 'rho_theta_base', 'pressure_base', 'exner_base'))})
    state = SimpleNamespace(**{name: np.arange(8 if name == 'rho_u' else 9 if name == 'rho_w' else 18 if name == 'scalars' else 6, dtype=np.float32).reshape(
        (2, 4) if name == 'rho_u' else (3, 3) if name == 'rho_w' else (3, 2, 3) if name == 'scalars' else (2, 3)) for name in STATE_FIELDS}, time_seconds=360)
    saved = SimpleNamespace(**{name: np.ones((2, 4) if name == 'normal_velocity' else (3, 3) if name == 'vertical_velocity' else (2, 3), np.float32)
                               for name in SAVED_FIELDS})
    state.scalars[-1].view(np.uint32)[0, 0] = 0x80000000
    fields = []
    for pool, name, dimensions, shape, levels, kind in (
        ('diag_physics', 'qke', ['nVertLevels', 'nCells'], [2, 4], 2, 'real'),
        ('sfc_input', 'tslb', ['nSoilLevels', 'nCells'], [4, 4], 1, 'real'),
        ('sfc_input', 'sst', ['nCells'], [4], 1, 'real'),
        ('tend', 'rt_diabatic_tend', ['nVertLevels', 'nCells'], [2, 4], 1, 'real'),
        *((('sfc_input', name, ['nCells'], [4], 1, 'real')) for name in ('xland', 'xice', 'isltyp', 'ivgtyp')),
        ('sfc_input', 'isice_lu', [], [], 1, 'integer')):
        fields.append(dict(pool=pool, name=name, pool_name=name, dimensions=dimensions,
                           shape=shape, nlevels=levels, kind=kind, components=[]))
    layout = SimpleNamespace(fields=fields, dimensions={'nCells': 3},
        configs=dict(config_do_restart=False, config_do_DAcycling=False),
        root=ET.fromstring('<registry><streams><stream name="restart"><var name="qke"/><var name="tslb"/></stream></streams></registry>'), enabled=lambda item: True)
    native = PoolWords(layout)
    for row in fields:
        for level in range(1, row['nlevels']+1):
            words = np.full(row['shape'], level+2, np.float32 if row['kind'] == 'real' else np.int32)
            if row['name'] == 'qke':
                words.view(np.uint32)[0, 0] = 0x80000000 if level == 1 else 0x7fc12345
            native.put_field(row['pool'], row['pool_name'], words, level=level)
    session = SimpleNamespace(native=native, config=SimpleNamespace(suite='convection_permitting'),
        precipitation_window=WindowWords({'rain': np.arange(3, dtype=np.float32), 'steps': np.arange(3, dtype=np.int32)}),
        surface_service=ServiceWords({'february': 29}))
    dycore = SimpleNamespace(cp=HostArrays, atmosphere=SimpleNamespace(state=state, saved=saved))
    return (mesh, grid, None, state, reference, saved, ('qv', 'qc', 'tke')), layout, session, dycore


def write_checkpoint(tmp_path):
    inputs, layout, session, dycore = member()
    arguments = SimpleNamespace(checkpoint_hours='0.1,0.2', history_every_minutes=6)
    route, text = resolve_route(arguments, TEXT)
    assert route.resolve_schedule(Fraction(720), Fraction(60), []) == 720
    route.begin('2026-10-17_00:00:00')
    route.bind_inputs(inputs, CONFIG)
    assert not route.after_step(6, steps=12, dt=60, session=session, dycore=dycore,
        mesh=inputs[0], vertical=inputs[1].vertical_grid, names=inputs[6],
        layout=layout, out=tmp_path, columns=None)
    path = tmp_path/'checkpoint.t00000360.pkl'
    assert path.is_file()
    assert len(route.receipt()['native_checkpoints']) == 1
    return path, inputs, layout, session


def test_native_checkpoint_roundtrip_keeps_physics_levels_and_precipitation_window(tmp_path):
    path, inputs, layout, session = write_checkpoint(tmp_path)
    route, text = resolve_route(SimpleNamespace(restart_from=path, checkpoint_hours='0.2',
        history_every_minutes=6, da_analysis=tmp_path/'analysis.nc'), TEXT)
    assert route.resolve_schedule(Fraction(720), Fraction(60), []) == 360
    assert namelist_option(text, 'nhyd_model', 'config_start_time') == '2026-10-17_00:06:00'
    assert namelist_option(text, 'restart', 'config_do_restart') is False
    route.mark_restart(layout)
    assert layout.configs['config_do_restart'] and layout.configs['config_do_DAcycling']
    route.bind_inputs(inputs, CONFIG)
    state, saved, weights = route.resume_dycore(mesh=inputs[0], grid=inputs[1], state=inputs[3],
        saved=inputs[5], reference=inputs[4], names=inputs[6], target_grid='unused')
    assert weights is None and state.time_seconds == 360
    for name in STATE_FIELDS:
        np.testing.assert_array_equal(getattr(state, name).view(np.uint32), getattr(inputs[3], name).view(np.uint32))
    for name in SAVED_FIELDS:
        np.testing.assert_array_equal(getattr(saved, name), getattr(inputs[5], name))
    before = {key: value.copy() for key, value in session.native.data.items()}
    for key in session.native.data:
        session.native.data[key].fill(0)
    session.precipitation_window.restore({'rain': np.zeros(3, np.float32), 'steps': np.zeros(3, np.int32)})
    session.surface_service.restore_restart_state({'february': 28})
    restored = route.restore_physics(session, layout, None, inputs[0])
    assert restored['fields'] == 9
    for key in (('diag_physics', 'qke', 1), ('diag_physics', 'qke', 2), ('sfc_input', 'tslb', 1), ('tend', 'rt_diabatic_tend', 1)):
        np.testing.assert_array_equal(session.native.data[key].view(np.uint32), before[key].view(np.uint32))
    for name in ('sst', 'xland', 'xice', 'isltyp', 'ivgtyp'):
        np.testing.assert_array_equal(session.native.data['sfc_input', name, 1], before['sfc_input', name, 1])
    np.testing.assert_array_equal(session.precipitation_window.snapshot()['steps'], np.arange(3))
    assert session.surface_service.restart_state() == {'february': 29}


@pytest.mark.parametrize('change', ['mesh', 'vertical', 'reference', 'species'])
def test_restart_rejects_changed_identity_before_cuda(tmp_path, change):
    path, inputs, _, _ = write_checkpoint(tmp_path)
    route, _ = resolve_route(SimpleNamespace(restart_from=path, history_every_minutes=6), TEXT)
    if change == 'mesh': inputs[0].arrays['areaCell'][0] += 1
    if change == 'vertical': inputs[1].vertical_grid.zgrid[0, 0] += 1
    if change == 'reference': inputs[4].pressure_base[0, 0] += 1
    if change == 'species': inputs = (*inputs[:-1], ('qc', 'qv', 'tke'))
    with pytest.raises(ValueError, match='mesh, vertical grid, base state or species'):
        route.bind_inputs(inputs, CONFIG)


def test_restart_rejects_changed_physics_before_cuda(tmp_path):
    path, inputs, _, _ = write_checkpoint(tmp_path)
    route, _ = resolve_route(SimpleNamespace(restart_from=path, history_every_minutes=6), TEXT)
    changed = SimpleNamespace(receipt=lambda: dict(suite='convection_permitting', options={'pbl': 'changed'}))
    with pytest.raises(ValueError, match='physics configuration or numerics'):
        route.bind_inputs(inputs, changed)


def test_restart_accepts_a_changed_correction_set_but_not_changed_schemes(tmp_path):
    # THE BREAKAGE: a checkpoint written before a release added a correct-mode finding (RP7) was refused
    # ("physics configuration or numerics differ"), though a correction never changes the carried state.
    path, inputs, _, _ = write_checkpoint(tmp_path)
    route, _ = resolve_route(SimpleNamespace(restart_from=path, history_every_minutes=6), TEXT)
    added = SimpleNamespace(receipt=lambda: dict(suite='convection_permitting', options={}, corrections=['RP7']))
    route.bind_inputs(inputs, added)
    assert route.restart_corrections_change == {'checkpoint': [], 'resumed': ['RP7']}
    route, _ = resolve_route(SimpleNamespace(restart_from=path, history_every_minutes=6), TEXT)
    changed = SimpleNamespace(receipt=lambda: dict(suite='convection_permitting', options={'pbl': 'x'},
                                                   corrections=['RP7']))
    with pytest.raises(ValueError, match='physics configuration or numerics'):
        route.bind_inputs(inputs, changed)


def test_restart_accepts_a_changed_timestep_on_the_checkpoint_instant(tmp_path):
    # THE BREAKAGE: a 120 km run that went unstable at dt 600 s could not be resumed at a
    # smaller step from its last checkpoint ("native checkpoint timestep differs").
    path, _, _, _ = write_checkpoint(tmp_path)
    route, _ = resolve_route(SimpleNamespace(restart_from=path, history_every_minutes=6), TEXT)
    assert route.resolve_schedule(720, 30, []) == 360
    assert route.restart_dt_change == {'checkpoint_dt': '60', 'resumed_dt': '30'}
    assert route.receipt()['restart_dt_change'] == route.restart_dt_change
    route, _ = resolve_route(SimpleNamespace(restart_from=path, history_every_minutes=6), TEXT)
    with pytest.raises(ValueError, match='outer-step boundary'):
        route.resolve_schedule(720, 50, [])


def test_restart_rejects_clock_and_original_end_mismatches(tmp_path):
    path, _, _, _ = write_checkpoint(tmp_path)
    route, _ = resolve_route(SimpleNamespace(restart_from=path, history_every_minutes=6), TEXT)
    with pytest.raises(ValueError, match='original forecast end'):
        route.resolve_schedule(360, 60, [])


@pytest.mark.parametrize('hours', ['0', '-1', 'nan', '0.00001'])
def test_checkpoint_rejects_invalid_boundaries(hours):
    with pytest.raises((ValueError, ZeroDivisionError)):
        resolve_route(SimpleNamespace(checkpoint_hours=hours), TEXT)


def test_checkpoint_uses_native_history_interval_when_cli_omits_it():
    route, _ = resolve_route(SimpleNamespace(checkpoint_hours='0.1'), TEXT)
    streams = [SimpleNamespace(name='output', directions=('output',), output_interval='00:06:00')]
    assert route.resolve_schedule(720, 60, streams) == 720
    assert route.frame_seconds == 360
    route, _ = resolve_route(SimpleNamespace(checkpoint_hours='0.1', history_every_minutes=5), TEXT)
    with pytest.raises(ValueError, match='history frames'):
        route.resolve_schedule(720, 60, [])


def test_native_da_analysis_without_any_carrier_fails_before_pool_decode():
    with pytest.raises(ValueError, match='requires --restart-from'):
        resolve_route(SimpleNamespace(da_analysis=Path('analysis.nc')), TEXT)


def test_five_minute_cycle_accepts_the_public_float_hours_representation():
    route, _ = resolve_route(SimpleNamespace(checkpoint_hours=str(5/60),
        history_every_minutes=5), TEXT)
    assert route.resolve_schedule(Fraction(str(5/60))*3600, 60, []) == 300
    assert route.checkpoint_due(5, 60)


def test_same_mesh_moving_resume_continues_the_segment_that_sealed_it(tmp_path, monkeypatch):
    """THE BREAKAGE THIS GUARDS (fix0414/resume): a moving segment resumed on its own member (crash recovery, a stay
    on the carrier member) resumed as a native MPAS restart: its carrier held neither the sfc_input words, the
    schemes' persistent words nor the continuation, and the physics clock began again, so same-mesh stop/resume on
    the MPAS physics moving route was not bit for bit (x-build-moving-mpasphys)."""
    from woof.hex.mpasphys.checkpoint_v842 import CheckpointRoute
    from woof.hex.mpasphys.moving_v842 import MovingRoute
    inputs, layout, session, dycore = member()
    carrier = tmp_path / 'boundary.pkl'
    for name in [key for key in __import__('os').environ if key.startswith('WOOF_HEX_MOVING_')]:
        monkeypatch.delenv(name)
    monkeypatch.setenv('WOOF_HEX_MOVING_CHECKPOINT_OUT', str(carrier))
    sealing, _ = resolve_route(SimpleNamespace(history_every_minutes=6), TEXT)
    assert type(sealing) is MovingRoute
    sealing.begin('2026-10-17_00:00:00')
    sealing.finish(step=6, session=session, dycore=dycore, mesh=inputs[0], vertical=inputs[1].vertical_grid,
                   names=inputs[6], layout=layout, dt=60)
    from woof.hex.mpasphys.moving_v842 import load_carrier
    sealed = load_carrier(carrier)['physics']
    assert {'sfc_input.sst', 'sfc_input.xland'} <= set(sealed['fields'])
    assert 'continuation' in sealed and sealed['manager_elapsed'] == '360'

    monkeypatch.delenv('WOOF_HEX_MOVING_CHECKPOINT_OUT')
    monkeypatch.setenv('WOOF_HEX_MOVING_CHECKPOINT_IN', str(carrier))
    route, text = resolve_route(SimpleNamespace(history_every_minutes=6), TEXT)
    assert type(route) is CheckpointRoute and route.moving_resume and not route.fixed_mesh
    assert namelist_option(text, 'nhyd_model', 'config_start_time') == '2026-10-17_00:06:00'
    # A moving segment's --hours is its own length, never the original end.
    assert route.resolve_schedule(Fraction(360), Fraction(60), []) == 360
    route.bind_inputs(inputs, SimpleNamespace(receipt=CONFIG.receipt, mode='correct'))
    assert route.resume_continues and route.physics_clock_offset(60) == (6, 360.0)
    before = {key: value.copy() for key, value in session.native.data.items()}
    for key in session.native.data:
        session.native.data[key].fill(0)
    route.restore_physics(session, layout, None, inputs[0])
    for name in ('sst', 'xland', 'xice', 'isltyp', 'ivgtyp'):
        np.testing.assert_array_equal(session.native.data['sfc_input', name, 1], before['sfc_input', name, 1])


def test_correct_mode_native_restart_resumes_the_continuation_beside_its_file(tmp_path, monkeypatch):
    """THE BREAKAGE THIS GUARDS (fix0414/resume): a native MPAS restart restarts itimestep at 1, so the hrrrv5 suite
    re-ran its first-step blocks (MYNN-SFC, RUC) and hex restarted from its own +24 h restart differed from the
    uninterrupted run in 22.4 M of 36.4 M diag words one hour later.  Correct mode seals a continuation beside every
    restart file it writes and resumes it on config_do_restart; exact mode keeps native's restart."""
    from woof.hex.mpasphys.checkpoint_v842 import (CONTINUATION_SUFFIX, native_restart_continuation,
                                                  restart_output)
    for name in [key for key in __import__('os').environ if key.startswith('WOOF_HEX_MOVING_')]:
        monkeypatch.delenv(name)
    streams = tmp_path / 'streams.atmosphere'
    streams.write_text('<streams><immutable_stream name="restart" type="input;output" '
                       'filename_template="restart.$Y-$M-$D_$h.$m.$s.nc" input_interval="initial_only" '
                       'output_interval="01:00:00"/></streams>')
    assert restart_output(streams) == (Fraction(3600), 'restart.$Y-$M-$D_$h.$m.$s.nc')
    streams.write_text(streams.read_text().replace('output_interval="01:00:00"', 'output_interval="none"'))
    assert restart_output(streams) is None
    # A correct-mode run whose deck writes restarts seals each one's continuation beside it.
    route, _ = resolve_route(SimpleNamespace(streams=str(tmp_path / 'deck.xml')), TEXT,
                             config=SimpleNamespace(mode='correct'))
    assert route is None
    deck = tmp_path / 'deck.xml'
    deck.write_text('<streams><immutable_stream name="restart" type="input;output" '
                    'filename_template="restart.$Y-$M-$D_$h.$m.$s.nc" output_interval="01:00:00"/></streams>')
    route, _ = resolve_route(SimpleNamespace(streams=str(deck), history_every_minutes=6), TEXT,
                             config=SimpleNamespace(mode='correct', receipt=CONFIG.receipt))
    route.origin_time = '2026-10-17_00:00:00'
    assert route.resolve_schedule(Fraction(7200), Fraction(60), []) == 7200
    assert route.checkpoint_due(60, 60) and not route.checkpoint_due(30, 60)
    assert route.checkpoint_target(60, 60, tmp_path).name == 'restart.2026-10-17_01.00.00.nc' + CONTINUATION_SUFFIX
    exact, _ = resolve_route(SimpleNamespace(streams=str(deck)), TEXT, config=SimpleNamespace(mode='exact'))
    assert exact is None
    # A restart resumes the continuation beside its file in correct mode only.
    restart = tmp_path / 'restart.2026-10-17_01.00.00.nc'
    restart.write_bytes(b'')
    correct = SimpleNamespace(mode='correct')
    assert native_restart_continuation(SimpleNamespace(), correct, 'restart', restart) is None
    sidecar = Path(str(restart) + CONTINUATION_SUFFIX)
    sidecar.write_bytes(b'')
    assert native_restart_continuation(SimpleNamespace(), correct, 'restart', restart) == sidecar
    assert native_restart_continuation(SimpleNamespace(), SimpleNamespace(mode='exact'), 'restart', restart) is None
    assert native_restart_continuation(SimpleNamespace(), correct, 'input', restart) is None
