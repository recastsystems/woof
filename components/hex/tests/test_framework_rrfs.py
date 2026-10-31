"""RRFS ported framework (no native library): declarations, storage, manager initialization.

CPU tier. The bitwise module replay against libmpasphys_native_rrfs.so after its native physics_init is
tests/rrfs_framework_init_replay.py (needs the capture library and a real hydrated case).
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.rrfs.framework_rrfs import DATA, FORK_PIN, FrameworkPhysicsRRFS, registry_source
from woof.hex.mpasphys.rrfs.framework_init_rrfs import FrameworkInitializationRRFS

TREE = Path(__file__).resolve().parents[1]


def test_declarations_come_from_the_pinned_fork():
    meta = json.loads((DATA / 'metadata.json').read_text())
    assert meta['schema'] == 'mpas-framework-rrfs/v1'
    assert meta['source_pin']['rrfs-model'] == FORK_PIN
    manifest = json.loads((TREE / '../../woof/hex/data/mpasphys-native/mpas-source-rrfs.json').read_text())
    for relative, digest in meta['source_sha256'].items():
        assert manifest['files'][relative]['sha256'] == digest
    # The fork's module storage and constant words (mpas_atmphys_vars.F +245 lines over v8.4.2).
    assert len(meta['variables']) == 656 and len(meta['constants']) == 61
    assert len(meta['allocations']) == 12 and 'allocate_smoke' in meta['allocations']
    for name in ('num_landcat', 'num_soilcat', 'num_chem', 'do_chemistry', 'f_smoke_fine'):
        assert name in meta['variables']
    registry = (registry_source() / 'src/core_atmosphere/Registry.xml').read_text()
    assert '#include' not in registry and 'nwfa2d' in registry and 'nChem' in registry


def test_constant_words_and_fork_identity():
    with FrameworkPhysicsRRFS() as framework:
        assert framework.pin_set == 'rrfs' and framework.framework_only
        assert framework.build['source_pin'] == FORK_PIN
        for name, row in framework.abi['constants'].items():
            assert framework.get_constant(name).view(np.uint32) == row['word']
        assert framework.get_table('module_mp_tempo_params.tcg_racg') is None


def _module_dims(framework, cells=3, levels=4):
    values = dict(ims=1, ime=cells, jms=1, jme=1, kms=1, kme=levels+1, num_months=12, num_soils=9,
                  num_oznlevels=59, num_aerosols=0, num_aerlevels=0, cam_abs_dim1=0, cam_abs_dim2=levels+1,
                  num_landcat=20, num_soilcat=16, num_chem=2, kanthro=1, kreswoodcomb=1,
                  num_e_ant_in=1, num_e_dust_out=1)
    for name, value in values.items():
        framework.put_variable(name, np.array(value, np.int32))


def test_every_fork_allocation_set_allocates_and_frees():
    with FrameworkPhysicsRRFS() as framework:
        _module_dims(framework)
        framework.config('bb_input_prevh', 6)
        for name in framework.abi['allocations']:
            framework.call(name)
        assert framework.get_variable('landusef_p')[0].shape == (3, 20, 1)
        assert framework.get_variable('soilf_p')[0].shape == (3, 16, 1)
        assert framework.get_variable('fre_avg_p')[0].shape == (3, 1, 4)   # nblocks = 24 / 6
        assert framework.get_variable('chem_p')[0].shape == (3, 5, 1, 2)
        framework.call('deallocate_lsm')
        assert framework.get_variable('landusef_p') is None


def _initialization(configs, dimensions, start='2023-03-31_12:00:00', dt=20):
    framework = FrameworkPhysicsRRFS()
    framework.clock(start=start, stop='2023-04-01_12:00:00', dt=dt)
    session = SimpleNamespace(native=framework, layout=SimpleNamespace(dimensions=dimensions, configs=configs),
                              config=SimpleNamespace(selectors={}, active=lambda slot: False,
                                                     table_key='mpasphys_rrfs'))
    return framework, FrameworkInitializationRRFS(session)


HRRRV5 = dict(config_frac_seaice=True, config_radtlw_interval='00:15:00', config_radtsw_interval='00:15:00',
              config_conv_interval='none', config_pbl_interval='none', config_lsm_scheme='sf_ruc',
              config_microp_scheme='mp_tempo', config_microp_re=True, config_radt_lw_scheme='rrtmg_lw',
              config_radt_sw_scheme='rrtmg_sw', config_smoke_scheme='off', config_dust_scheme='off',
              config_anthro_scheme='off', config_do_restart=False, config_o3climatology=True)
DIMS = dict(nCells=5, nCellsSolve=4, nVertLevels=59, nMonths=12, nSoilLevels=9, nOznLevels=59,
            nlcat=20, nscat=16, nChem=0)


@pytest.mark.parametrize('scheme,allocated', [('cu_grell_freitas', False),
                                             ('off', False), ('cu_grell_freitas_li', True)])
def test_smoke_does_not_read_unallocated_convective_rain(scheme, allocated):
    # The real hrrrv5 resident hour failed on absent diag_physics.raincv.
    # Fork driver_smoke.F:688-692 reads it only for GF-LI, otherwise writes zero.
    from woof.hex.mpasphys.rrfs.smoke_9652bbf import _native_smoke_field_active
    options = {**HRRRV5, 'config_smoke_scheme': 'on', 'config_convection_scheme': scheme}
    assert _native_smoke_field_active('raincv', 'from_mpas', options, {}) is allocated


@pytest.mark.parametrize('dt,n_microp,dt_microp', [(20, 1, 20), (150, 2, 75), (90, 1, 90)])
def test_run_init_matches_fork_manager(dt, n_microp, dt_microp):
    framework, init = _initialization(HRRRV5, DIMS, dt=dt)
    with framework:
        init.physics_run_init(None)
        v = lambda name: framework.get_variable(name)[0][()]
        assert (v('ime'), v('kme'), v('kte'), v('num_soils')) == (4, 60, 59, 9)
        assert (v('num_landcat'), v('num_soilcat'), v('num_chem'), v('do_chemistry')) == (20, 16, 0, 0)
        assert v('n_microp') == n_microp and v('dt_microp') == np.float32(dt_microp)
        assert v('dt_radtlw') == np.float32(900) and v('dt_pbl') == np.float32(dt)
        assert (v('has_reqc'), v('has_reqi'), v('has_reqs')) == (1, 1, 1)
        assert v('xice_threshold') == np.float32(.02)
        assert framework.get_table('mpas_atmphys_manager.gmt')[0] == np.float32(12)
        assert framework.get_table('mpas_atmphys_manager.julday')[0] == 90


def test_registry_init_writes_nine_ruc_layers_on_every_local_cell():
    framework, init = _initialization(HRRRV5, DIMS)
    with framework:
        framework.put_field('sfc_input', 'dzs', np.full((9, 6), 7, np.float32))
        init.physics_registry_init(None)
        dzs = framework.get_field('sfc_input', 'dzs')
        want = np.array([.005, .025, .045, .13, .25, .35, .50, 1.00, .70], np.float32)
        assert np.array_equal(dzs[:, :5], np.repeat(want[:, None], 5, 1))
        assert np.all(dzs[:, 5] == 7)    # the dummy column keeps its words


def test_tables_init_sets_the_fork_flag():
    framework, init = _initialization(HRRRV5, DIMS)
    with framework:
        init.physics_tables_init(None)
        assert framework.get_variable('l_mp_tables')[0] == 1


def test_tempo_constant_words_come_from_the_pinned_source():
    from woof.hex.mpasphys.rrfs.framework_init_rrfs import (TEMPO_CONSTANT_ARRAYS, TEMPO_CONSTANT_SCALARS,
                                                           TEMPO_DERIVED, tempo_constant_names)
    meta = json.loads((DATA / 'metadata.json').read_text())
    constants = meta['tempo_constants']
    manifest = json.loads((TREE / '../../woof/hex/data/mpasphys-native/mpas-source-rrfs.json').read_text())
    for relative, digest in meta['tempo_constants_source'].items():
        assert manifest['files'][relative]['sha256'] == digest
    assert set(constants) == set(tempo_constant_names())
    # Everything tempo_table_init reads is a captured constant; nothing it writes is (av_g/bv_g: declaration values).
    assert set(TEMPO_CONSTANT_SCALARS) | set(TEMPO_CONSTANT_ARRAYS) <= set(constants)
    assert not {name for name, _, _ in TEMPO_DERIVED} - {'av_g', 'bv_g'} & set(constants)
    word = lambda name: np.array(constants[name]['words'], np.uint32).view(np.float32)
    assert word('av_r')[0] == np.float32(4854.0) and word('pi')[0] == np.float32(3.1415926536)
    assert constants['r_c']['shape'] == [37] and constants['nbins']['dtype'] == 'int32'
    assert constants['max_ni']['dtype'] == 'float64'


def test_ruc_landuse_outside_the_native_mapping_refuses_by_name():
    framework, init = _initialization(HRRRV5, DIMS)
    with framework:
        framework.put_text_field('sfc_input', 'mminlu', 'MODIFIED_IGBP_MODIS_NOAH_15s')
        with pytest.raises(RuntimeError, match='has no RUC table'):
            init.initialize_ruc()


def test_smoke_allocations_use_registry_emission_counts():
    # Native manager.F:478-483 reads pool names num_e_*, then copies the
    # local n_e_* pointers into module variables at lines719-724.
    # Using those local pointer names as Registry keys allocated zero-sized
    # emission buffers and the real smoke-on hour faulted on its first call.
    import xml.etree.ElementTree as ET
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist
    source = registry_source()
    root = ET.parse(source / 'src/core_atmosphere/Registry.xml').getroot()
    dimensions = {d.get('name'): 1 for d in root.findall('./dims/dim')
                  if not d.get('definition')}
    dimensions.update(nCells=4, nEdges=6, nVertices=4, nVertLevels=59,
                      maxEdges=3, maxEdges2=6, vertexDegree=3, nlcat=20, nscat=16,
                      nSoilLevels=9, nOznLevels=59, nMonths=12)
    text = ("&physics\nconfig_physics_suite='hrrrv5'\n/\n"
            "&chemistry\nconfig_smoke_scheme='on', config_dust_scheme='on', num_chem=3\n/\n")
    config = resolve_namelist(text, default_backend='hex-cuda')
    layout = RegistryLayout(source, config, text, dimensions)
    framework, init = _initialization(layout.configs, layout.dimensions)
    with framework:
        init.physics_run_init(None)
        for name, count in dict(num_e_bb_in=1, num_e_bb_out=1,
                                num_e_dust_out=2, num_e_ant_in=0, num_e_ant_out=0).items():
            assert layout.dimensions.get(name, 0) == count
            assert framework.get_variable(name)[0].item() == count
        framework.config('bb_input_prevh', layout.configs['bb_input_prevh'])
        framework.call('allocate_smoke')
        assert framework.get_variable('e_bb_in_p')[0].shape == (4, 1, 1, 1)
        assert framework.get_variable('e_bb_out_p')[0].shape == (4, 60, 1, 1)
        assert framework.get_variable('e_dust_out_p')[0].shape == (4, 60, 1, 2)
