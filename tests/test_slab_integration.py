"""The slab selector reaches real configuration, soil, history and restart seams."""
from dataclasses import replace
import numpy as np
import pytest
from woof.config import RunConfig, soil_layer_count, validate_run_config
from woof.core.slab import soil_depths, SlabParameters
from woof.core.physics_inventory import PHYSICS_SLOT_DISPATCH, LAND_SURFACE_SFCDIAGS_SCHEMES
from woof.physics_registry import physics_registry


def test_selector_registry_dispatch_agree():
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=12000., dy=12000., ztop=8000., dt=10., run_seconds=600.,
                    sf_surface_physics=1, sf_sfclay_physics=1, bl_pbl_physics=1,
                    moist=True, num_soil_layers=5)
    assert soil_layer_count(cfg) == 5
    assert PHYSICS_SLOT_DISPATCH['sf_surface_physics'][1] == '_run_slab'
    assert 1 in LAND_SURFACE_SFCDIAGS_SCHEMES
    validate_run_config(cfg)
    with pytest.raises(ValueError, match='5'):
        validate_run_config(replace(cfg, num_soil_layers=4))
    reg = physics_registry()
    opt = reg['components']['land_surface']['options']['slab']
    assert opt['implemented'] is True
    assert opt['selectors']['sf_surface_physics'] == 1
    assert opt['parameters']['num_soil_layers'] == 5
    for name in ('ifsnow', 'seaice_threshold'):
        assert reg['parameters'][name].get('implemented') is not False
        assert not reg['parameters'][name].get('unimplemented_reason')


def test_ensemble_keeps_slab_member_driver_and_selection():
    from woof.ensemble.suite_capabilities import plan_suite
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=12000., dy=12000., ztop=8000.,
                    dt=10., run_seconds=600., sf_surface_physics=1,
                    sf_sfclay_physics=1, bl_pbl_physics=1,
                    moist=True, num_soil_layers=5)
    plan = plan_suite(cfg, members=2)
    surface = next(row for row in plan.components if row.component == 'land_surface')
    assert plan.mode == 'member_local'
    assert surface.implemented and surface.option_id == 'slab'
    assert dict(surface.selectors) == {'sf_surface_physics': 1}
    assert cfg.sf_surface_physics == 1 and cfg.num_soil_layers == 5


@pytest.mark.parametrize('ifsnow', [0, 1, 2])
def test_bare_import_admits_five_layer_selector(tmp_path, ifsnow):
    from test_namelist_import import INPUT_TEXT, WPS_TEXT, _pair
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    inp = INPUT_TEXT.replace('sf_surface_physics = 2, 2,',
                            f'sf_surface_physics = 1, 1,\n ifsnow = {ifsnow},\n seaice_threshold = 270.0,')
    pair = _pair(tmp_path, WPS_TEXT, inp)
    text, report = import_namelists(*pair)
    out = tmp_path / 'slab.toml'; out.write_text(text)
    exp = load_experiment(out)
    assert exp.root.run.sf_surface_physics == 1
    assert exp.root.run.num_soil_layers == 5
    assert exp.root.run.ifsnow == ifsnow
    assert exp.root.run.seaice_threshold == 270.0
    assert not any(s.key == 'sf_surface_physics' for s in report.substitutions)


def test_seaice_threshold_and_fixed_bottom(monkeypatch):
    from woof.ingest.ruc_soil import preprocess_land_surface_soil
    from test_ruc_soil_wiring import _era5_fields
    import woof.wrf_exact as exact
    fields = _era5_fields(3, 4)
    fields['XICE'][...] = 0
    fields['SKINTEMP'][:, -1] = [269., 270., 271.]
    result = preprocess_land_surface_soil(fields, soil_type=np.full((3,4),8),
              sf_surface_physics=1, seaice_threshold=270.,
              water_temperature_policy='wrf_compat')
    assert result.soil_temperature.shape == (5,3,4)
    assert result.landmask[0,-1] == 1
    assert result.landmask[1,-1] == result.landmask[2,-1] == 0
    if not exact.ENABLED:
        assert np.float32(result.soil_temperature[4,0,-1]) == np.float32(271.4)
    assert np.all(result.soil_moisture == 0)
    monkeypatch.setattr(exact, 'ENABLED', True)
    wrf = preprocess_land_surface_soil(fields, soil_type=np.full((3,4),8),
              sf_surface_physics=1, seaice_threshold=270.,
              water_temperature_policy='wrf_compat')
    assert np.all(wrf.soil_temperature[:,0,-1] == np.float32(286.))


def test_five_layers_and_geometry_are_written_and_read(tmp_path):
    from woof.io.wrfout import WrfoutWriter
    from netCDF4 import Dataset
    zs, dzs = soil_depths()
    tslb = np.arange(5*3*4, dtype=np.float32).reshape(5,3,4) + 270
    path = tmp_path / 'wrfout_d01_slab'
    with WrfoutWriter(path, nx=4, ny=3, nz=5, dx=12000., dy=12000., soil_layers=5,
                      global_attrs={'SF_SURFACE_PHYSICS': 1}) as w:
        w.write_frame('2026-10-07_12:00:00', {'TSLB': tslb, 'ZS': zs, 'DZS': dzs})
    with Dataset(path) as ds:
        assert ds.variables['ZS'].dimensions == ('Time','soil_layers_stag')
        for name, value in [('ZS',zs), ('DZS',dzs), ('TSLB',tslb)]:
            assert np.asarray(ds.variables[name][0]).tobytes() == value.tobytes()


@pytest.mark.parametrize("changed", ["geometry", "arithmetic"])
def test_restart_roundtrip_and_geometry_identity(monkeypatch, tmp_path, changed):
    from test_restart import _cfg, _shim_driver_state, _fill_setup
    from woof.io import restart
    cfg = _cfg(moist=True, sf_surface_physics=1, num_soil_layers=5,
               sf_sfclay_physics=1, bl_pbl_physics=1)
    def setup():
        state, driver = _shim_driver_state(cfg, monkeypatch)
        _fill_setup(state)
        driver.fields['tslb'] = np.arange(5*cfg.ny*cfg.nx, dtype=np.float32).reshape(5,cfg.ny,cfg.nx) + 270
        for name in ('thc','capg','gsw','sfcevp'):
            driver.fields[name] = np.full((cfg.ny,cfg.nx), 0.04, np.float32)
        return state, driver
    state, driver = setup()
    path = restart.write_restart(tmp_path/'slab.npz', state, cfg)
    fresh, restored = setup()
    restored.fields['tslb'][...] = 0
    restart.restore_restart(path, fresh, cfg)
    for name in ('tslb','thc','capg','gsw','sfcevp'):
        assert restored.fields[name].tobytes() == driver.fields[name].tobytes()
    assert restored.slab_params == driver.slab_params
    before = restart.physics_setup_fingerprint(fresh, cfg)
    restored.slab_params = (replace(restored.slab_params, zs=(0.006,)+restored.slab_params.zs[1:])
                           if changed == "geometry" else
                           replace(restored.slab_params, wrf_exact=not restored.slab_params.wrf_exact))
    assert restart.physics_setup_fingerprint(fresh, cfg) != before
    with pytest.raises(restart.RestartMismatchError):
        restart.restore_restart(path, fresh, cfg)


def test_non_slab_checkpoint_echo_keeps_its_previous_keys():
    from woof.io.restart import configuration_echo
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=12000., dy=12000., ztop=8000.,
                    dt=10., run_seconds=600.)
    echo = configuration_echo(cfg)
    assert 'ifsnow' not in echo and 'seaice_threshold' not in echo
    slab_cfg = replace(cfg, sf_surface_physics=1, num_soil_layers=5)
    assert configuration_echo(slab_cfg)['ifsnow'] == 1
    assert 'seaice_threshold' not in configuration_echo(slab_cfg)
    assert configuration_echo(replace(slab_cfg, seaice_threshold=270.))['seaice_threshold'] == 270.


def test_non_slab_experiment_still_binds_common_seaice_threshold():
    from woof.core.model import restart_identity_payload
    from test_nest_spawn_init import _experiment
    exp = _experiment()
    def identity(threshold):
        domains = tuple(replace(domain, run=replace(domain.run,
            sf_surface_physics=2, seaice_threshold=threshold)) for domain in exp.domains)
        return restart_identity_payload(replace(exp, domains=domains))
    assert identity(270.) != identity(280.)


def test_direct_export_uses_the_slab_depths():
    from woof.wrf_direct import _direct_export_soil_geometry
    expected = soil_depths()
    actual = _direct_export_soil_geometry(1, 5)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a.astype(np.float32).view(np.uint32), e.view(np.uint32))
    with pytest.raises(ValueError, match='five soil layers'):
        _direct_export_soil_geometry(1, 4)


def test_history_check_ignores_ready_receipts(tmp_path):
    import importlib.util
    from pathlib import Path
    from woof.io.wrfout import WrfoutWriter
    script = Path(__file__).resolve().parents[1]/'tools/slab_wrf461_oracle/check_run_history.py'
    spec = importlib.util.spec_from_file_location('slab_check_history', script)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    (tmp_path/'ready').mkdir()
    (tmp_path/'ready'/'wrfout_d01_2026-10-07_18_00_00.json').write_text('{}')
    frame={'TSK': np.full((3,4),280.,np.float32), 'TSLB':np.full((5,3,4),280.,np.float32)}
    with WrfoutWriter(tmp_path/'wrfout_d01_column.nc', nx=4, ny=3, nz=5, dx=12000., dy=12000., soil_layers=5,
                      global_attrs={'SF_SURFACE_PHYSICS':1}) as w:
        w.write_frame('2026-10-07_18:00:00',frame)
        w.write_frame('2026-10-07_18:10:00',frame)
    assert mod.main(str(tmp_path)) == 0


def test_numerics_modes_have_distinct_slab_cache_inputs(monkeypatch):
    from woof import numerics
    from woof.core import kernels
    monkeypatch.setattr(numerics, "_mode", "strict")
    strict = kernels.module_source("slab")
    unrelated = kernels.module_source("noah")
    monkeypatch.setattr(numerics, "_mode", "fast")
    fast = kernels.module_source("slab")
    assert strict != fast
    assert "#define GPUWM_SLAB_STRICT 1" in strict
    assert "#define GPUWM_SLAB_STRICT 0" in fast
    assert kernels.module_source("noah") == unrelated


def test_evaporation_accumulates_with_surface_interval(monkeypatch):
    from types import SimpleNamespace
    import woof.core.physics as physics
    from woof.core.slab import SlabParameters
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=12000., dy=12000., ztop=8000.,
                    dt=10., run_seconds=600., sf_surface_physics=1,
                    num_soil_layers=5)
    params = SlabParameters.from_config(cfg)
    monkeypatch.setattr(SlabParameters, 'device_geometry', lambda self: (None, None))
    fields = {name: np.ones((4, 6), np.float32)
              for name in ('cqs2', 'mavail', 'qfx', 'sfcevp')}
    fields['sfcevp'][...] = 0
    monkeypatch.setattr(physics, 'launch_slab', lambda *a, **kw: fields['qfx'].fill(.002))
    driver = SimpleNamespace(fields=fields, slab_params=params, bldt_seconds=60., ra_lw_physics=0)
    atmosphere = {name: np.zeros((5, 4, 6), np.float32)
                  for name in ('temperature', 'qv', 'pressure')}
    for step in (1, 7):
        physics.PhysicsDriver._run_slab(driver, atmosphere, cfg, step)
    np.testing.assert_array_equal(fields['sfcevp'],
        np.full((4, 6), np.float32(np.float32(.002) * np.float32(60.)) * 2, np.float32))
