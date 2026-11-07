"""Surface FDDA admission, allocation and tree checkpoint contracts."""
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from woof.namelist_import import _Section
from woof.spectral_nudging.config import NudgingConfig, SURFACE_KEYS, from_mapping
from woof.spectral_nudging.namelist import translate
from woof.spectral_nudging.surface_forecast import HELD, memory_shapes


def test_stock_surface_namelist_imports_each_selected_domain():
    row = translate(_Section('fdda', {
        'grid_fdda': [1, 1], 'grid_sfdda': [1, 1],
        'sgfdda_inname': ['wrfsfdda_d<domain>'],
        'sgfdda_interval_m': [180, 60], 'sgfdda_end_h': [6, 3],
        'gfdda_interval_m': [180, 180], 'gfdda_end_h': [6, 6],
        'guv': [0., 0.], 'gt': [0., 0.], 'gq': [0., 0.],
        'guv_sfc': [.0003, .0003], 'gt_sfc': [.0003, .0003], 'gq_sfc': [.0003, .0003],
        'rinblw': [250., 250.],
    }, 'surface test'), (1, 2), lambda *args: None)
    options = from_mapping(row, (1, 2))
    assert options.for_domain(1).sgfdda_interval_m == 180
    assert options.for_domain(2).sgfdda_interval_m == 60
    assert options.for_domain(1).sgfdda_inname == 'wrfsfdda_d<domain>'
    assert options.for_domain(1).guv_sfc == .0003


@pytest.mark.parametrize('mode,message', [(0, 'WRF_SURFACE_FDDA_PARENT'),
                                         (2, 'WRF_SURFACE_FDDA_PARENT')])
def test_surface_requires_native_grid_driver(mode, message):
    with pytest.raises(ValueError, match=message):
        translate(_Section('fdda', {'grid_fdda': [mode], 'grid_sfdda': [1]},
                           'surface test'), (1,), lambda *args: None)


def test_fasdas_uses_the_same_native_grid_driver():
    result=translate(_Section('fdda', {'grid_fdda':[1], 'grid_sfdda':[2],
        'sgfdda_interval_m':[180], 'sgfdda_end_h':[6], 'rinblw':[250],
        'gfdda_interval_m':[180], 'gfdda_end_h':[6]},
        'surface test'),(1,),lambda *args:None)
    assert from_mapping(result,(1,)).for_domain(1).grid_sfdda==2


def test_surface_off_preserves_existing_config_payload():
    from dataclasses import asdict
    config = NudgingConfig(enabled=True, mode='grid', convention='wrf')
    before = asdict(config)
    for key in SURFACE_KEYS:
        before.pop(key)
    assert config.to_dict() == before
    assert not (set(config.to_dict()) & set(SURFACE_KEYS))


def test_surface_memory_prices_volume_scratch_and_all_native_records():
    cfg = SimpleNamespace(nx=11, ny=7, nz=9)
    controls = NudgingConfig(enabled=True, mode='grid', convention='wrf',
        grid_sfdda=1, sgfdda_interval_m=60, sgfdda_end_h=3, rinblw=250,
        guv=0.,gt=0.,gq=0.)
    shapes = memory_shapes(cfg, controls, 7200)
    assert shapes['surface_fdda.analyses'] == (2, 2, 4, 8, 12)
    assert shapes['surface_fdda.operator_density'] == (4, 8, 12)
    assert shapes['surface_fdda.surface_fdda_u'] == (9, 7, 12)
    assert shapes['surface_fdda.combined.rtheta'] == (9, 7, 11)


def test_surface_registry_matches_serialization_and_history():
    from woof.spectral_nudging.surface_contract import registry_rows
    from woof.io.restart import STATE_SERIALIZED_ATTRS, STATE_INFRA_ATTRS
    row = registry_rows()['grid_sfdda']['1']
    assert set(row['state']) == set((*HELD, 'hfx_fdda'))
    assert set(row['state']) <= set(STATE_SERIALIZED_ATTRS)
    assert '_surface_fdda_owner' in STATE_INFRA_ATTRS


@pytest.mark.parametrize("fasdas", [False, True])
def test_native_restart_reads_held_surface_words_and_checks_cadence(monkeypatch, fasdas):
    from tests.test_wrf_warm_start import _restart_dataset, _Variable, _cfg, _expected
    from woof import netcdf_bridge
    from woof.ingest.wrfrst import read_wrfrst
    dataset = _restart_dataset()
    for name, dims in (('RUNDGDTEN', ('bottom_top', 'south_north', 'west_east_stag')),
                       ('RVNDGDTEN', ('bottom_top', 'south_north_stag', 'west_east')),
                       ('RTHNDGDTEN', ('bottom_top', 'south_north', 'west_east')),
                       ('RQVNDGDTEN', ('bottom_top', 'south_north', 'west_east')),
                       ('HFX_FDDA', ('bottom_top', 'south_north', 'west_east'))):
        shape = tuple(len(dataset.dimensions[d]) for d in dims)
        value = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) + np.uint32(0x3f000000)
        value.flat[0] = np.uint32(0x80000000)
        dataset.variables[name] = _Variable(name, dims, value.view(np.float32))
    if fasdas:
        for name in ('SDA_HFX','SDA_QFX','QNORM','HFX_BOTH','QFX_BOTH'):
            dims=('south_north','west_east')
            value=np.arange(12,dtype=np.uint32).reshape(3,4)+np.uint32(0x3f000000)
            dataset.variables[name]=_Variable(name,dims,value.view(np.float32))
    dataset.variables['STEPFG'] = _Variable('STEPFG', (), np.array(3, np.int32))
    dataset.variables['TH2_NDG_OLD'] = _Variable('TH2_NDG_OLD', ('south_north', 'west_east'), np.full((3,4), 290, np.float32))
    monkeypatch.setattr(netcdf_bridge, 'open_dataset', lambda path: dataset)
    controls = NudgingConfig(enabled=True, mode='grid', convention='wrf', grid_sfdda=2 if fasdas else 1,
        sgfdda_interval_m=180, sgfdda_end_h=6, rinblw=250, fgdt=3.6,
        guv=0., gt=0., gq=0.)
    restored = read_wrfrst('wrfrst_d01', cfg=_cfg(dt=72), expected_dimensions=_expected(), nudging_controls=controls)
    for name in ('RUNDGDTEN', 'RVNDGDTEN', 'RTHNDGDTEN', 'RQVNDGDTEN', 'HFX_FDDA',
                 *(('SDA_HFX','SDA_QFX','QNORM','HFX_BOTH','QFX_BOTH') if fasdas else ())):
        np.testing.assert_array_equal(restored.raw[name].view(np.uint32), dataset.variables[name]._data[0].view(np.uint32))
    assert restored.warm_start.dispositions['TH2_NDG_OLD'].startswith('rebuilt')
    dataset.variables['STEPFG'] = _Variable('STEPFG', (), np.array(2, np.int32))
    with pytest.raises(ValueError, match='STEPFG'):
        read_wrfrst('wrfrst_d01', cfg=_cfg(dt=72), expected_dimensions=_expected(), nudging_controls=controls)


@pytest.mark.parametrize("fasdas", [False, True])
def test_surface_held_words_survive_parent_child_checkpoint(monkeypatch, tmp_path, fasdas):
    from tests.test_restart import _sealed_tree_fixture
    from woof.io import restart
    source, start = _sealed_tree_fixture(monkeypatch, forcing_count=2,
        run_seconds=3600., payload_seed=31)
    resumed, _ = _sealed_tree_fixture(monkeypatch, forcing_count=2,
        run_seconds=3600., payload_seed=91)
    from woof.spectral_nudging.surface_forecast import FASDAS_HELD
    carried = (*HELD, "hfx_fdda", *(FASDAS_HELD if fasdas else ()))
    expected = {}
    for model, fill in ((source, True), (resumed, False)):
        for node in model.walk_parent_first():
            state = node.state
            for index, (name, template) in enumerate(zip(carried,
                    (state.u, state.v, state.thp, state.qv, state.thp,
                     *(state.thp[0] for _ in range(5))))):
                array = np.zeros_like(template)
                if fill:
                    words = np.arange(array.size, dtype=np.uint32) + np.uint32(0x3f000000 + 1000*node.cfg.grid_id + 100*index)
                    words[0] = np.uint32(0x80000000)
                    words[-1] = np.uint32(1)
                    array.view(np.uint32).flat[:] = words
                    expected[node.cfg.grid_id, name] = words.copy()
                setattr(state, name, array)
    path = restart.write_tree_restart(tmp_path, source, start + timedelta(seconds=3600))
    restart.restore_tree_restart(path, resumed)
    for node in resumed.walk_parent_first():
        for name in carried:
            np.testing.assert_array_equal(getattr(node.state, name).view(np.uint32).ravel(),
                                          expected[node.cfg.grid_id, name])
    assert not np.shares_memory(resumed.root.state.surface_fdda_theta,
                               resumed.node(2).state.surface_fdda_theta)


def test_combined_px_surface_stock_import_consumes_shared_keys_once(tmp_path):
    from tests.test_namelist_import import INPUT_TEXT, _pair
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    text = INPUT_TEXT.replace("sf_sfclay_physics = 91, 91,", "sf_sfclay_physics = 7, 7,").replace(
        "sf_surface_physics = 2, 2,", "sf_surface_physics = 7, 7,\n num_soil_layers = 2,").replace(
        "bl_pbl_physics = 11, 11,", "bl_pbl_physics = 7, 7,")
    text += """
&fdda
 grid_fdda=1,1, grid_sfdda=1,1, pxlsm_soil_nudge=1,1,
 sgfdda_interval_m=180,180, sgfdda_end_h=6,6,
 sgfdda_inname='wrfsfdda_d<domain>', io_form_sgfdda=2,
 gfdda_interval_m=180,180, gfdda_end_h=6,6, rinblw=250.,250.,
 guv=0.,0., gt=0.,0., gq=0.,0.,
/
"""
    translated, _ = import_namelists(*_pair(tmp_path, inp=text), name="pxsurface")
    out = tmp_path / "case.toml"
    out.write_text(translated)
    exp = load_experiment(out)
    for domain in exp.domains:
        assert domain.run.pxlsm_soil_nudge == 1
        assert domain.run.sgfdda_inname == "wrfsfdda_d<domain>"
        assert exp.nudging.for_domain(domain.grid_id).sgfdda_interval_m == 180
