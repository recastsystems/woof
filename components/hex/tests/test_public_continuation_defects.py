"""Public restart and producer regressions, independent of analysis services."""
from datetime import timedelta
import hashlib
import json
import pickle
from types import SimpleNamespace as NS

import numpy as np
import pytest


@pytest.fixture
def stream_bridge():
    """These tests write and read real streams through the Rust stream bridge."""
    from woof.hex import engines, restart_v842
    try:
        restart_v842._bridge()
    except engines.EngineRefusal:
        pytest.skip('requires the rw_mpas_stream bridge (staged, installed or WOOF_HEX_RW_MPAS_STREAM)')


def test_physics_refusal_survives_worker_serialization():
    from woof.hex.physics_suites import PhysicsSelectionRefusal, OptionRefusal
    refusal = OptionRefusal('config_pbl_scheme', 'absent', 'no implementation', 'select a registered scheme')
    original = PhysicsSelectionRefusal((refusal,), context='requested physics')
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is type(original)
    assert restored.refusals == original.refusals
    assert str(restored) == str(original)


@pytest.mark.parametrize('pin', ['rrfs', 'rrfs-44c0849'])
@pytest.mark.parametrize('suite', ['rrfs_mesoscale_reference', 'rrfs_convection_permitting', 'hrrrv5'])
def test_native_init_soil_depth_matches_the_selected_land_scheme(tmp_path, pin, suite):
    from woof.hex.mpasphys.rrfs.cold_start_rrfs import producer_for_suite, declarations, case7_jobs, case9_jobs
    _, row = producer_for_suite(suite, pin_set=pin)
    expected = 9 if suite == 'hrrrv5' else 4
    assert row['soil']['n_soil_layers'] == expected
    start = '2026-01-01_00:00:00'
    for part in ('prepare', 'artifact'):
        declared = declarations(row, start, part)
        if 'dimensions' in declared:
            assert declared['dimensions']['nSoilLevels'] == expected
    prepare, initialize = case7_jobs(row, static=tmp_path/'static.nc', met=tmp_path/'met',
        start_time=start, nfglevels=34, nfgsoillevels=4, artifact=tmp_path/'carrier.nc',
        out=tmp_path/'init.nc', levels_file=tmp_path/'levels', aerosols=tmp_path/'aerosols')
    assert prepare['config']['soil_levels'] == expected
    assert initialize['config']['n_soil_layers'] == expected
    carrier, _ = case9_jobs(row, init=tmp_path/'init.nc', start_time=start,
        intervals=[(start, tmp_path/'met')], nfglevels=34, out_dir=tmp_path,
        carrier=tmp_path/'lbc-carrier.nc', aerosols=tmp_path/'aerosols')
    assert carrier['config']['dimensions']['nSoilLevels'] == expected


def _iau_payload(path):
    rows = []
    for name, shape in [('u_amb', (2, 5)), ('rho_amb', (2, 4)),
                        ('theta_amb', (2, 4)), ('scalars_amb', (1, 2, 4))]:
        words = np.full(shape, 0.01, np.float32).tobytes(order='F')
        (path/name).write_bytes(words)
        rows.append(dict(pool='tend_iau', name=name, shape=shape, payload=name,
                         sha256=hashlib.sha256(words).hexdigest(), origins=['iau']))
    (path/'manifest.json').write_text(json.dumps(dict(fields=rows)))


def _iau(path, *, checkpoint=None, restart=False, mode='correct', array_module=np):
    from woof.hex.mpasphys.runtime_v842 import SuiteIau
    layout = NS(configs=dict(config_IAU_option='on', config_IAU_window_length_s=10,
                            config_do_restart=restart, config_do_DAcycling=False),
                dimensions=dict(nCells=3, nEdges=4),
                indices={('state', 'moist_start'): 1, ('state', 'moist_end'): 1, ('state', 'index_qv'): 1})
    mesh = NS(arrays=dict(cellsOnEdge=np.array([[0, 1], [1, 2], [2, 0], [0, 2]])))
    dycore = NS(cp=array_module, atmosphere=NS(vertical=NS(zz=array_module.ones((2, 3), np.float32))))
    return SuiteIau(layout=layout, payload=path, config=NS(mode=mode), mesh=mesh,
                    dycore=dycore, dt=3, checkpoint=checkpoint)


def test_checkpoint_retains_the_iau_clock_and_original_increment_identity(tmp_path):
    from woof.hex.mpasphys.checkpoint_v842 import rank_continuation
    from woof.hex.iau_v842 import atm_iau_coef_window
    _iau_payload(tmp_path)
    initial = _iau(tmp_path)
    initial.position[...] = np.float32(6)
    carried = pickle.loads(pickle.dumps(rank_continuation(NS(iau_window=initial))))
    resumed = _iau(tmp_path, checkpoint=carried['iau_window'], restart=True)
    assert resumed.position.view(np.uint32) == initial.position.view(np.uint32)
    weights = [atm_iau_coef_window(initial.config, t, 3) for t in (0, 3, 6, 9, 12)]
    continued = [atm_iau_coef_window(resumed.config, float(resumed.position)+t, 3) for t in (0, 3, 6)]
    np.testing.assert_array_equal(continued, weights[2:])
    assert sum(float(w)*3 for w in weights) == pytest.approx(1)
    resumed.position[...] = 15
    assert resumed.expected_fraction() == pytest.approx(.4)
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    manifest['fields'][0]['unpartitioned_sha256'] = 'changed increment'
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='increment payload changed'):
        _iau(tmp_path, checkpoint=carried['iau_window'], restart=True)


@pytest.mark.parametrize('update', [dict(version=2), dict(elapsed_seconds=-1),
    dict(elapsed_seconds=float('nan')), dict(window_seconds=20)])
def test_invalid_iau_checkpoint_is_refused(tmp_path, update):
    _iau_payload(tmp_path)
    window = _iau(tmp_path).checkpoint_state()
    with pytest.raises(ValueError, match='IAU'):
        _iau(tmp_path, checkpoint={**window, **update}, restart=True)


def test_exact_iau_restart_keeps_native_window_reset(tmp_path):
    _iau_payload(tmp_path)
    restored = _iau(tmp_path, checkpoint=dict(elapsed_seconds=6), restart=True, mode='exact')
    assert restored.position == 0
    assert restored.checkpoint_state() is None


@pytest.mark.parametrize('restart,first,expected', [
    (False, 30, [30, 90, 150]), (True, 30, [90, 150]), (True, 60, [60, 120, 180])])
def test_history_lead_uses_original_epoch_and_native_restart_control(tmp_path, restart, first, expected):
    from test_mpasphys_streams_output_v842 import session, stream
    from woof.hex.mpasphys.streams_output_v842 import PhysicsStreams
    from dataclasses import replace
    state, _, _, initial = session(resident=False)
    start = initial + timedelta(minutes=30) if restart else initial
    state.layout.configs.update(config_start_time=start.strftime('%Y-%m-%d_%H:%M:%S'),
                               config_do_restart=restart)
    selected = replace(stream(), output_interval='01:00:00')
    with PhysicsStreams(state, (selected,), tmp_path, final_time=initial+timedelta(hours=3),
                        history_from_minutes=first) as writer:
        assert writer.initial == initial
        due = []
        for minute in range(0, 181, 10):
            when = initial+timedelta(minutes=minute)
            if writer.alarms.due('output', when):
                due.append(minute)
                writer.alarms.reset('output', when)
    assert due == expected


@pytest.mark.parametrize('module', ['engine_registry', 'cuda_suite_physics_v841'])
def test_existing_path_alias_keeps_source_precedence(monkeypatch, tmp_path, module):
    import importlib
    import sys
    from pathlib import Path
    root = tmp_path/'engine'
    core = root/'woof'/'core'
    core.mkdir(parents=True)
    (root/'woof'/'__init__.py').write_text('')
    (core/'__init__.py').write_text('')
    (root/'woof'/'physics_registry_v2.json').write_text('{}')
    (core/'mpas_column_suite.py').write_text('def column_physics_plan(physics):\n    return dict(physics=physics)\n')
    alias = tmp_path/'selected-engine'
    alias.symlink_to(root, target_is_directory=True)
    for name in tuple(sys.modules):
        if name == 'woof' or name.startswith('woof.'):
            monkeypatch.delitem(sys.modules, name)
    previous = [str(tmp_path/'higher-priority-source'), str(alias), *sys.path]
    monkeypatch.setattr(sys, 'path', previous.copy())
    try:
        seam = importlib.import_module('woof.hex.'+module)
        if module == 'engine_registry':
            assert seam.describe_selection({'mp_physics': 1}, root)['physics'] == {'mp_physics': 1}
        else:
            assert seam._engine_root(root) == root
        assert sys.path == previous
        assert Path(sys.modules['woof'].__file__).resolve().parent.parent == root
    finally:
        for name in tuple(sys.modules):
            if name == 'woof' or name.startswith('woof.'):
                del sys.modules[name]


def _restart_file(path):
    from woof.hex.output import HistoryField
    from woof.hex.restart_v842 import write_stream
    from woof.hex.streams_v842 import Stream
    fields = {}
    for name, value in dict(rho_zz=1.1, rho=7, theta_m=303, theta=500,
            rho_base=1, theta_base=300, rtheta_base=300, exner_base=.9,
            pressure_base=80000, rho_p=.2, rtheta_p=21, pressure_p=150,
            exner=.91, qv=.01).items():
        fields[name] = HistoryField(np.full((2, 3), value, np.float32), ('nVertLevels', 'nCells'))
    for name, shape, dimensions, value in [
        ('ru', (2, 4), ('nVertLevels', 'nEdges'), 8),
        ('u', (2, 4), ('nVertLevels', 'nEdges'), 1),
        ('rw', (3, 3), ('nVertLevelsP1', 'nCells'), 4),
        ('w', (3, 3), ('nVertLevelsP1', 'nCells'), .5)]:
        fields[name] = HistoryField(np.full(shape, value, np.float32), dimensions)
    mesh = NS(arrays={}, attrs={}, dimensions=dict(nCells=3, nEdges=4, nVertLevels=2, nVertLevelsP1=3))
    stream = Stream('restart', dict(type='output', io_type='pnetcdf,cdf5'),
                    tuple(('var', name, '') for name in fields))
    write_stream(path, stream, mesh, fields, '2026-01-01_00:10:00', initial_time='2026-01-01_00:00:00')
    return fields


@pytest.mark.parametrize('regional,early', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('restart', [False, True])
def test_native_input_loaders_keep_restart_coupled_words(monkeypatch, tmp_path, regional, early, restart, stream_bridge):
    """Read a real stream through each runtime loader, before the device phase."""
    from woof.hex import driver, vector, mesh as mesh_module, dynamics_v841, rust_geometry
    from woof.hex.mpasphys import runtime_v842 as runtime
    source = tmp_path/'state.nc'
    fields = _restart_file(source)
    mesh = NS(arrays=dict(areaCell=np.ones(3, np.float32), dcEdge=np.ones(4, np.float32),
                         cellsOnEdge=np.array([[0, 1], [1, 2], [2, 0], [0, 2]], np.int32)))
    vertical = NS(zz=np.ones((2, 3), np.float32), fzm=np.full(3, .5, np.float32),
                  fzp=np.full(3, .5, np.float32), zxu=np.zeros((3, 4), np.float32), n_vert_levels=2)
    grid = NS(vertical_grid=vertical, terrain_metrics=None)
    monkeypatch.setattr(mesh_module.Mesh, 'from_netcdf', lambda *a, **k: mesh)
    monkeypatch.setattr(driver, 'load_mpas_vertical_grid', lambda *a, **k: grid)
    monkeypatch.setattr(dynamics_v841, 'load_v841_reference_wind_profiles', lambda *a, **k: None)
    monkeypatch.setattr(vector, 'initialize_reconstruction_coefficients', lambda *a, **k: np.ones((3, 3, 3)))
    vectors = NS(edge_normal_vectors=np.zeros((3, 4)), local_vertical_unit_vectors=np.zeros((3, 3)),
                 cell_tangent_plane=np.zeros((2, 3, 3)))
    monkeypatch.setattr(vector, 'initialize_vector_geometry', lambda *a, **k: vectors)
    monkeypatch.setattr(rust_geometry, '_run', lambda *a, **k: (None, None, None, None))
    class Hydration:
        def __init__(self, *args): pass
        def horizontal_length(self): return np.float32(10000)
        def field_buffer(self, *args, **kwargs): return np.zeros((1, 2, 4), np.float32)
    monkeypatch.setattr(runtime, 'NativePoolHydration', Hydration)
    class Loaded(Exception): pass
    actual = driver.load_mpas_initial_state
    decoded = []
    def read(*args, **kwargs):
        decoded.append(actual(*args, **kwargs))
        raise Loaded
    monkeypatch.setattr(driver, 'load_mpas_initial_state', read)
    layout = NS(configs=dict(config_apply_lbcs=regional, config_coef_3rd_order=.25,
                            config_do_restart=restart), fields=[dict(pool='state', name='scalars', pool_name='scalars', components=())])
    arguments = NS(init=source)
    with pytest.raises(Loaded):
        if early:
            runtime._regional_source_inputs(arguments, layout)
        else:
            runtime.load_native_dynamics_inputs(arguments, layout, tmp_path)
    state, reference, saved = decoded[0]
    if restart:
        for actual_words, name in [(state.rho, 'rho_zz'), (state.rho_u, 'ru'), (state.rho_w, 'rw'),
                (saved.theta_m, 'theta_m'), (saved.density_perturbation, 'rho_p'),
                (saved.rho_theta_perturbation, 'rtheta_p'), (saved.pressure_perturbation, 'pressure_p'),
                (saved.exner, 'exner'), (reference.pressure_base, 'pressure_base')]:
            np.testing.assert_array_equal(actual_words.view(np.uint32), fields[name].values.view(np.uint32))
    else:
        assert np.all(state.rho == 7)
        assert not np.array_equal(state.rho_u, fields['ru'].values)
        assert np.all(state.rho_w[[0, -1]] == 0)


@pytest.mark.parametrize('rust_provenance', [False, True])
def test_native_static_cull_only_checks_the_provenance_it_carries(monkeypatch, tmp_path, rust_provenance, stream_bridge):
    from woof.hex import cull_door
    from woof.hex.output import HistoryField
    from woof.hex.restart_v842 import write_stream
    from woof.hex.streams_v842 import Stream
    parent = tmp_path/'static.nc'
    mesh = NS(arrays={}, attrs={}, dimensions=dict(nCells=3))
    selected = Stream('static', dict(type='output', io_type='pnetcdf,cdf5'), (('var', 'ter', ''),))
    attrs = {'rw_static_provenance_json': '{}'} if rust_provenance else {}
    write_stream(parent, selected, mesh, {'ter': HistoryField(np.zeros(3, np.float32), ('nCells',))},
                 '2026-01-01_00:00:00', initial_time='2026-01-01_00:00:00', global_attrs=attrs)
    actual_run = cull_door.subprocess.run
    calls = []
    def run(argv, **kwargs):
        # Header inspection still uses the real Rust stream artifact.
        if argv[0] != str(tmp_path/'mesh-engine'):
            return actual_run(argv, **kwargs)
        calls.append(argv)
        return NS(returncode=0, stdout='--cull-provenance-grid --cull-mark-parent', stderr='')
    monkeypatch.setattr(cull_door.subprocess, 'run', run)
    result = cull_door.cull_one(tmp_path/'mesh-engine', parent, tmp_path/'region.json',
        tmp_path/'cut.nc', provenance_grid=tmp_path/'grid-cut.nc', mark_parent=tmp_path/'parent-grid.nc')
    assert ('--cull-provenance-grid' in result['argv']) is rust_provenance
    assert '--cull-mark-parent' in result['argv']


def test_optional_boundary_layer_height_keeps_its_units(monkeypatch, tmp_path):
    from netCDF4 import Dataset
    from woof.hex.drivers import run_cuda_v841_full_physics_x4 as driver
    for name, size in [('N_CELLS', 3), ('N_EDGES', 4), ('N_LEVELS', 2),
                       ('N_INTERFACES', 3), ('N_SOIL_LEVELS', 0)]:
        monkeypatch.setattr(driver, name, size)
    static = {name: np.zeros(3, np.float32) for name in ('indexToCellID', 'latCell', 'lonCell', 'ter')}
    static.update({name: np.zeros(4, np.float32) for name in ('indexToEdgeID', 'latEdge', 'lonEdge')})
    height = np.array([100, 400, 900], np.float32)
    path = tmp_path/'history.nc'
    driver.write_snapshot_netcdf(path, dict(arrays=dict(pblh=height)), static)
    with Dataset(path) as dataset:
        assert dataset['pblh'].units == 'm'
        np.testing.assert_array_equal(dataset['pblh'][0], height)


@pytest.mark.gpu
def test_cuda_iau_checkpoint_at_partial_window_preserves_every_source_word(tmp_path):
    import cupy as cp
    from woof.hex.cuda_backend import KernelCache
    from woof.hex.mpasphys.checkpoint_v842 import rank_continuation
    _iau_payload(tmp_path)
    state = NS(rho=cp.ones((2, 3), np.float32), scalars=cp.full((1, 2, 3), .01, np.float32))
    saved = NS(theta_m=cp.full((2, 3), 300, np.float32), normal_velocity=cp.full((2, 4), 2, np.float32))
    diagnostics = NS(h_edge=cp.ones((2, 4), np.float32))
    cache = KernelCache()
    def arm(split):
        window = _iau(tmp_path, array_module=cp)
        sources = NS(rho_u=cp.zeros((2, 4), np.float32), rho_theta=cp.zeros((2, 3), np.float32),
                     rho=cp.zeros((2, 3), np.float32), scalars=cp.zeros((1, 2, 3), np.float32))
        for step in range(1, 6):
            if split and step == 3:
                carried = pickle.loads(pickle.dumps(rank_continuation(NS(iau_window=window))))
                window = _iau(tmp_path, checkpoint=carried['iau_window'], restart=True, array_module=cp)
            window.apply(itimestep=step, sources=sources, state=state, saved=saved,
                         diagnostics=diagnostics, cache=cache)
        cp.cuda.get_current_stream().synchronize()
        return {name: cp.asnumpy(value) for name, value in vars(sources).items()}, window.position
    uninterrupted, full_clock = arm(False)
    continued, restart_clock = arm(True)
    for name in uninterrupted:
        assert np.any(uninterrupted[name])
        np.testing.assert_array_equal(continued[name].view(np.uint32), uninterrupted[name].view(np.uint32),
                                      err_msg=name)
    assert full_clock == restart_clock == 15
