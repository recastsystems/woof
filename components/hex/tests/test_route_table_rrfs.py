"""Table-selected admission of the RRFS-MPAS route (item B, rrfs/nf-glue): framework, resident glue, services.

The native-free RRFS route is chosen from the mpasphys_rrfs table row (route_table), never by row key: these
checks hold whether or not the fork's ported framework (item A) is in the distribution yet.
"""
import importlib.util

import numpy as np
import pytest

from woof.hex.mpasphys import route_table
from woof.hex.mpasphys.resident_v842 import TARGET, ported_route_refusal, resident_requested
from woof.hex.mpasphys.rrfs.driver_rrfs import Backend, resolve_namelist, specification
from woof.hex.mpasphys.runtime_v842 import framework_only
from woof.hex.physics_suites import load_table


def deck(suite, extra='', **options):
    lines = [f"config_physics_suite='{suite}'"] + [f'{k}={v}' for k, v in options.items()]
    return '&physics\n' + '\n'.join(lines) + '\n/\n' + extra


def _present(module):
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def test_v842_row_declares_the_framework_and_glue_it_always_had():
    row = load_table()['mpasphys_v842']
    assert row['framework']['backend'] == 'woof.hex.mpasphys.framework_backend_v842:FrameworkPhysics'
    assert [dict(t) for t in row['resident']['targets']] == [TARGET]
    v842 = type('C', (), {'table_key': 'mpasphys_v842'})()
    assert route_table.framework_absence(v842) is None
    assert route_table.resident_scope(v842) == 'exact global convection_permitting'


def test_rrfs_framework_admission_follows_the_table_row_not_the_row_key():
    config = resolve_namelist(deck('hrrrv5'), default_backend='hex-cuda')
    absence = route_table.framework_absence(config)
    framework = specification()['framework']
    modules = {ref.split(':', 1)[0] for ref in framework.values()}
    assert (absence is None) == all(_present(m) for m in modules)
    if absence is not None:
        assert 'woof.hex.mpasphys.rrfs.framework_rrfs' in absence or 'framework_init_rrfs' in absence
        assert not framework_only(config)
    else:
        native_cpu = [s for s, b in config.backends.items() if config.active(s) and b == Backend.NATIVE_CPU]
        assert framework_only(config) == (not native_cpu)


def test_rrfs_drag_and_ocean_rows_reuse_the_source_identical_v842_slots():
    slots = specification()['slots']
    assert slots['gwdo']['cuda_entries']['bl_ugwp_gwdo']['module'] == 'ugwp_oro_v842'
    assert slots['gwdo']['cuda_entries']['bl_ysu_gwdo']['module'] == 'ysu_gwdo_v842'
    assert slots['oml']['cuda_entries']['on']['module'] == 'oml_v842'
    config = resolve_namelist(deck('hrrrv5'), default_backend='hex-cuda')
    # UGWP ships (owner ruling 2026-10-14, US Government code): hrrrv5's drag runs on the GPU, not the
    # native library (the stale UGWP hold sent it to native-cpu with 'not in this distribution').
    assert _present('woof.hex.mpasphys.ugwp_oro_v842')
    assert config.backends['gwdo'] == Backend.HEX_CUDA
    assert 'gwdo' not in {row.slot for row in config.cpu_slots}
    oml = resolve_namelist(deck('hrrrv5', config_oml1d='.true.'), default_backend='hex-cuda')
    assert oml.backends['oml'] == Backend.HEX_CUDA


def test_a_named_native_library_runs_what_the_ported_framework_cannot(monkeypatch):
    """hrrrv5 with every slot on the GPU (UGWP shipped) was refused although the run named its RRFS library."""
    from types import SimpleNamespace
    from woof.hex.mpasphys import forecast_v842, resident_v842
    config = resolve_namelist(deck('hrrrv5'), default_backend='hex-cuda')
    named = SimpleNamespace(native_physics_library='/site/libmpasphys_native_rrfs.so')
    bare = SimpleNamespace(native_physics_library=None)
    monkeypatch.delenv('MPAS_NATIVE_LIBRARY', raising=False)
    monkeypatch.setattr(resident_v842, 'ported_route_refusal', lambda config: 'resident glue absent')
    assert forecast_v842.named_library_route(named, config)
    assert not forecast_v842.named_library_route(bare, config)
    monkeypatch.setenv('MPAS_NATIVE_LIBRARY', '/site/libmpasphys_native_rrfs.so')
    assert forecast_v842.named_library_route(bare, config)
    # once the ported framework runs the selection, it is taken even when a library is named
    monkeypatch.setattr(resident_v842, 'ported_route_refusal', lambda config: None)
    assert not forecast_v842.named_library_route(named, config)


def test_rrfs_surface_service_is_a_fork_row_with_a_backend_option():
    config = resolve_namelist(deck('hrrrv5', "&hex_mpasphys\nconfig_surface_update_backend='hex-cuda'\n/"),
                              default_backend='hex-cuda')
    assert config.service_backends['surface_update'] == Backend.HEX_CUDA
    # A GPU route takes the fork service on the GPU by default; the referee and module replays stay native.
    assert resolve_namelist(deck('hrrrv5'), default_backend='hex-cuda').service_backends['surface_update'] \
        == Backend.HEX_CUDA
    assert resolve_namelist(deck('hrrrv5'), default_backend='native-cpu').service_backends['surface_update'] \
        == Backend.NATIVE_CPU
    assert resolve_namelist(deck('hrrrv5')).service_backends['surface_update'] == Backend.NATIVE_CPU
    service = route_table.service_type(config, 'surface_update', 'unused:unused')
    from woof.hex.mpasphys.rrfs.surface_rrfs import ForkSurfaceService
    assert service is ForkSurfaceService
    with pytest.raises(ValueError, match='unsupported fork surface_update'):
        resolve_namelist(deck('hrrrv5', "&hex_mpasphys\nconfig_surface_update_backend='off'\n/"))


def test_rrfs_resident_admission_needs_its_glue_and_every_active_slot_on_cuda(monkeypatch):
    config = resolve_namelist(deck('hrrrv5'), default_backend='hex-cuda',
                              backends={s: 'hex-cuda' for s in ('microphysics', 'pbl', 'surface_layer',
                                                                'land_surface', 'gwdo', 'radiation_lw',
                                                                'radiation_sw', 'cloudiness', 'seaice')})
    dims = dict(nCells=3, nCellsSolve=3, nEdges=6, nEdgesSolve=6)
    monkeypatch.setenv('WOOF_HEX_MPAS_RESIDENT', 'auto')
    assert resident_requested(config, dims) == route_table.resident_glue_present(config)
    monkeypatch.setenv('WOOF_HEX_MPAS_RESIDENT', '1')
    if not route_table.resident_glue_present(config):
        with pytest.raises(ValueError, match='table row admits'):
            resident_requested(config, dims)
    monkeypatch.setenv('WOOF_HEX_MPAS_RESIDENT', 'auto')
    for suite in ('rrfs_mesoscale_reference', 'rrfs_convection_permitting'):
        alias = resolve_namelist(deck(suite), default_backend='hex-cuda')
        assert framework_only(alias)
        assert resident_requested(alias, dims)
        assert ported_route_refusal(alias) is None
        mixed = resolve_namelist(deck(suite), default_backend='hex-cuda',
                                 backends={'microphysics': 'native-cpu'})
        assert not resident_requested(mixed, dims)


def _monthly_oracle(field, m1, m2, w1, w2, span):
    """mpas_atmphys_date_time.F monthly_interp_to_date in RKIND=4 with no contraction (numpy float32 is IEEE RN)."""
    f = np.asarray(field, dtype=np.float32)
    return (f[m2] * np.float32(w2) + f[m1] * np.float32(w1)) / np.float32(span)


@pytest.mark.parametrize('gvf,dust,lai', [(True, False, True), (False, False, True), (True, True, False)])
def test_fork_update_surface_matches_the_fork_routine_word_for_word(gvf, dust, lai):
    cp = pytest.importorskip('cupy')
    try:
        cp.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip('no CUDA device')
    from woof.hex.mpasphys.rrfs.surface_rrfs import physics_update_surface
    from woof.hex.mpasphys.surface_update_v842 import MonthlyCalendar
    rng = np.random.default_rng(7)
    n, cap = 517, 518
    months = lambda scale: np.asfortranarray(rng.uniform(0, scale, (12, cap)).astype(np.float32))
    host = dict(albedo12m=months(40), greenfrac=months(1), lai12m=months(8))
    land = rng.integers(0, 2, cap).astype(np.int32)
    sfc = {k: cp.asfortranarray(cp.asarray(v)) for k, v in host.items()}
    sfc['landmask'] = cp.asarray(land)
    for k in ('sfc_albbck', 'vegfra', 'shdmin', 'shdmax'):
        sfc[k] = cp.full(cap, -1, dtype=cp.float32)
    diag = {'lai': cp.full(cap, -1, dtype=cp.float32), 'feff': cp.full(cap, -1, dtype=cp.float32),
            'albedo_drag': cp.full(cap, -1, dtype=cp.float32),
            'feff_m_in': cp.asfortranarray(cp.asarray(months(1))),
            'albedo_drag_m_in': cp.asfortranarray(cp.asarray(months(1)))}
    calendar = MonthlyCalendar()
    date = '2023-03-31_12:00:00'
    m1, m2, w1, w2, span = MonthlyCalendar().weights(date)
    changed = physics_update_surface(date, config_sfc_albedo=True, config_gvf_update=gvf,
                                     config_dust_scheme='on' if dust else 'off',
                                     mesh={'nCellsSolve': n, 'nCells': cap}, sfc_input=sfc, diag_physics=diag,
                                     calendar=calendar, lai_allocated=lai)
    words = lambda a: np.asarray(a, dtype=np.float32).view(np.uint32)
    albedo = _monthly_oracle(host['albedo12m'][:, :n], m1, m2, w1, w2, span) / np.float32(100)
    albedo = np.where(land[:n] == 0, np.float32(0.08), albedo)
    assert np.array_equal(words(sfc['sfc_albbck'].get()[:n]), words(albedo))
    if gvf:
        assert np.array_equal(words(sfc['vegfra'].get()[:n]),
                              words(_monthly_oracle(host['greenfrac'][:, :n], m1, m2, w1, w2, span)))
        assert np.array_equal(sfc['shdmin'].get()[:n], host['greenfrac'][:, :n].min(axis=0))
        assert np.array_equal(sfc['shdmax'].get()[:n], host['greenfrac'][:, :n].max(axis=0))
    else:
        assert (sfc['vegfra'].get() == -1).all() and ('sfc_input', 'vegfra') not in changed
    if lai:
        assert np.array_equal(words(diag['lai'].get()[:n]),
                              words(_monthly_oracle(host['lai12m'][:, :n], m1, m2, w1, w2, span)))
    else:
        assert (diag['lai'].get() == -1).all()
    assert ((diag['feff'].get()[:n] != -1).all()) == dust
    # Owned cells only: the dummy/halo column keeps its words.
    assert sfc['sfc_albbck'].get()[n] == -1


def _door_arguments(**values):
    from types import SimpleNamespace
    base = dict(physics_suite=None, physics_namelist=None, physics=None, physics_mode=None,
                physics_correct_skip=None, physics_default_backend=None)
    base.update(values)
    return SimpleNamespace(**base)


def test_door_hrrrv5_takes_the_workflow_nine_soil_layers_unless_the_run_names_them(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import native_route
    config, text = native_route(_door_arguments(physics_suite='hrrrv5'))
    assert config.selectors['land_surface'] == 'sf_ruc' and config.options['num_soil_layers'] == 9
    named, _ = native_route(_door_arguments(physics_suite='hrrrv5', physics=['num_soil_layers=6']))
    assert named.options['num_soil_layers'] == 6
    noah, _ = native_route(_door_arguments(physics_suite='hrrrv5', physics=['config_lsm_scheme=sf_noah']))
    assert noah.options['num_soil_layers'] == 4
    # A run that brings its own native namelist keeps the fork Registry's default.
    deck_file = tmp_path / 'namelist.atmosphere'
    deck_file.write_text("&physics\nconfig_physics_suite='hrrrv5'\n/\n")
    own, _ = native_route(_door_arguments(physics_namelist=deck_file))
    assert own.options['num_soil_layers'] == 4
    v842, _ = native_route(_door_arguments(physics_suite='convection_permitting'))
    assert v842.options['num_soil_layers'] == 4


def test_rrfs_row_writes_its_streams_with_the_fork_contract():
    """THE BREAKAGE: the writer bound v8.4.2's Registry, so hrrrv5 refused its own history fields and ran
    with history off. The row now names the fork's Registry, writer, per-file attributes and git_version."""
    from woof.hex.mpasphys.streams_output_v842 import output_definitions
    hrrrv5 = resolve_namelist(deck('hrrrv5'), default_backend='hex-cuda')
    rrfs = route_table.stream_contract(hrrrv5)
    assert (rrfs.registry.__name__, rrfs.writer.__name__, rrfs.git_version) == (
        'woof.hex.streams_rrfs', 'woof.hex.restart_rrfs', 'unknown')
    for name, dims in (('ng', ['nVertLevels', 'nCells', 'Time']), ('zs', ['nSoilLevels', 'nCells', 'Time']),
                       ('defc_a', ['maxEdges', 'nCells'])):
        assert rrfs.registry.field_definition(name)['dimensions'] == dims
    stream = type('S', (), dict(name='output', fields=('ng', 'zs'), options={'output_interval': '01:00:00'},
                                directions=('output',), filename='history.$Y.nc', input_interval='none',
                                output_interval='01:00:00', filename_interval='01:00:00'))()
    layout = type('L', (), dict(fields=[]))()
    written = output_definitions([stream], layout, {}, stream_type=rrfs.registry.Stream)['output']
    packages = {p for gate in rrfs.registry.field_definition('ng')['gates'] for p in gate.split(';')}
    assert isinstance(written, rrfs.registry.Stream) and written.fields(packages) == ('ng', 'zs')
    assert rrfs.stream_file_attributes(written) == {'output_interval': 60}
    v842 = route_table.stream_contract(type('C', (), {'table_key': 'mpasphys_v842'})())
    assert (v842.registry.__name__, v842.writer.__name__) == ('woof.hex.streams_v842', 'woof.hex.restart_v842')
    assert v842.stream_file_attributes(written) == {}
    assert route_table.stream_contract(None).registry is v842.registry
    assert load_table()['mpasphys_rrfs']['streams_parser'] == 'woof.hex.mpasphys.rrfs.stream_xml_rrfs:stream_tree'
    assert route_table.output_diagnostics_type(hrrrv5).__name__ == 'ForkOutputDiagnostics'
