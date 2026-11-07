
from types import SimpleNamespace
import pytest
from woof.config import RunConfig, soil_layer_count
from woof.physics_registry import column_surface_contract, validate_column_surface_settings
from woof.core.physics_inventory import PHYSICS_SLOT_DISPATCH

def cfg(**changes):
    values = dict(nx=8, ny=8, nz=20, dx=3000, dy=3000, ztop=15000, dt=5, run_seconds=60, sf_surface_physics=8, sf_sfclay_physics=1, bl_pbl_physics=1,
                  ra_lw_physics=1, ra_sw_physics=1, fractional_seaice=1, num_soil_layers=3)
    values.update(changes)
    return RunConfig(**values)

def test_default_record_route_and_three_layers():
    validate_column_surface_settings(cfg())
    assert soil_layer_count(cfg()) == 3
    assert PHYSICS_SLOT_DISPATCH['sf_surface_physics'][8] == '_run_column_land_surface'
    assert len(column_surface_contract(8)['state_bindings']) == 111


def test_stock_ssib_namelist_import_keeps_the_option_and_soil_threshold(tmp_path):
    from test_namelist_import import WPS_TEXT, INPUT_TEXT
    from woof.experiment import load_experiment
    from woof.namelist_import import import_namelists
    text = INPUT_TEXT
    for before, after in (
        ('sf_surface_physics = 2, 2,', 'sf_surface_physics = 8, 8,'),
        ('sf_sfclay_physics = 91, 91,', 'sf_sfclay_physics = 1, 1,'),
        ('bl_pbl_physics = 11, 11,', 'bl_pbl_physics = 1, 1,'),
        ('ra_lw_physics = 4, 4,', 'ra_lw_physics = 1, 1,'),
        ('ra_sw_physics = 4, 4,', 'ra_sw_physics = 1, 1,'),
        ('&physics', '&physics\n num_soil_layers = 3,\n fractional_seaice = 1,\n seaice_threshold = 271.4,'),
    ):
        assert before in text
        text = text.replace(before, after)
    wps, inp = tmp_path/'namelist.wps', tmp_path/'namelist.input'
    wps.write_text(WPS_TEXT)
    inp.write_text(text)
    translated, report = import_namelists(wps, inp)
    output = tmp_path/'ssib.toml'
    output.write_text(translated)
    experiment = load_experiment(output)
    assert experiment.root.run.sf_surface_physics == 8
    assert experiment.root.run.num_soil_layers == 3
    assert experiment.root.run.fractional_seaice == 1
    assert experiment.root.run.seaice_threshold == 271.4

@pytest.mark.parametrize('key,value', [('sf_sfclay_physics',5), ('bl_pbl_physics',5),
    ('ra_lw_physics',2), ('ra_sw_physics',2), ('fractional_seaice',0), ('num_soil_layers',4), ('sf_lake_physics',1)])
def test_bad_pairing_names_wrf_breakage(key, value):
    with pytest.raises(ValueError, match=key):
        validate_column_surface_settings(cfg(**{key:value}))

def test_broadband_adapter_cannot_silently_supply_zero_spectral_flux():
    with pytest.raises(ValueError, match='four band albedos'):
        validate_column_surface_settings(cfg(ra_sw_physics=4))


@pytest.mark.gpu
def test_real_surface_driver_uses_the_column_and_carries_snow_state():
    import cupy as cp
    import numpy as np
    from datetime import datetime, timezone
    from woof.core import ssib
    from woof.core.physics import initialize_physics
    from woof.core.state import DomainState
    from woof import wrf_exact
    config = cfg()
    state = DomainState(config)
    driver = initialize_physics(state, config, radiation=lambda **kw: None,
        radiation_start_time=datetime(2024,1,1,tzinfo=timezone.utc),
        radiation_latitude=35.0, radiation_longitude=-100.0,
        landuse_dataset='SSIB', ivgtyp=7, soil_temperature=278.0,
        soil_moisture=0.25, tsk=278.0, snow=50.0, snow_depth=0.20)
    owner = driver.column_surface_params
    f = driver.fields
    # Model publishers can hand the generic record route strided views.
    # The packed output must reach the native view, not a reshape copy.
    source = f['ssib_lai']
    f['ssib_lai'] = cp.empty(source.shape, source.dtype, order='F')
    f['ssib_lai'][...] = source
    for key,value in [('fm',7),('fh',8)]:
        f[key].fill(value)
    f['surface_rainncv'].fill(0.1)
    driver._pending_rainbl.fill(0.02)
    f['_ssib_status'].fill(ssib.ERR_ICASE_UNSET)
    f['xice_save'].fill(.25)
    f['swdown'].fill(400.0); f['glw'].fill(300.0); f['coszen'].fill(0.6)
    shape = f['tsk'].shape
    a = {n: cp.full((config.nz, *shape),v, dtype=cp.float32) for n,v in
         [('dz',50),('u',5),('v',2),('qv',0.004),('temperature',278),('pressure',90000)]}
    for name,value in [('hfx_sea',0),('qfx_sea',0),('lh_sea',0),('tsk_sea',278)]:
        f[name].fill(value)
    for step in range(1,4):
        if step == 3:
            driver.bldt_seconds = 6.0
        driver._run_column_land_surface(a,config,step)
        assert float(owner.records.inputs[-1]) == driver.bldt_seconds
        assert bool(cp.isfinite(f['tsk']).all())
        assert bool(cp.isfinite(f['hfx']).all())
    assert bool(cp.any(f['ssib_veg'] == 7))
    assert float(owner.records.inputs[-2]) == 0.0
    assert f['isnow'].dtype == cp.int32
    assert owner.records.resident
    assert not hasattr(owner, 'state')
    assert owner.records.fields[ssib.STATE.index('ssib_lai')] is f['ssib_lai']
    for key,value in [('fm',7),('fh',8)]:
        assert bool(cp.all(f[key] == value))
    assert set(ssib.SSIB_REGISTRY_STATE) <= set(f)
    assert bool(cp.all(f['surface_rainncv'] == 0))
    from woof.core.surface_forcing import SURFACE_PRECIPITATION_FIELDS
    for name in SURFACE_PRECIPITATION_FIELDS + ('rainbl',):
        assert bool(cp.all(f[name] == 0))
    assert bool(cp.all(driver._pending_rainbl == 0))
    np.testing.assert_array_equal(f['xice_save'].get(), f['xice'].get())
    assert int(f['_ssib_status'].get()[0]) == 0
    assert float(cp.abs(f['hfx']).max()) > 0
    # Serialization must classify the record owner's rebuilt scratch.
    from woof.io.restart import _driver_manifest
    _driver_manifest(driver)


def test_forecast_preflight_prices_the_selected_surface_kernel():
    from woof.core.preflight import domain_kernel_modules
    from types import SimpleNamespace
    modules = domain_kernel_modules(SimpleNamespace(run=cfg(), grid_id=1), prices_refl=False)
    assert "ssib" in modules
    assert "noah" not in modules


def test_snow_solver_flag_has_a_checked_integer_health_policy():
    import numpy as np
    from woof.core.health import (gpu_integer_policy, rule_for_field,
                                  validate_fields_cpu, HealthField, FieldRule)
    assert gpu_integer_policy('surface.isnow', np.int32) is not None
    rule = rule_for_field('surface.isnow')
    assert rule == FieldRule('surface', 0.0, 1.0)
    for value in (0, 1):
        assert validate_fields_cpu([HealthField('surface.isnow', np.array([value],np.int32),rule)]).ok
    for value in (-1, 2):
        assert not validate_fields_cpu([HealthField('surface.isnow', np.array([value],np.int32),rule)]).ok


def test_spectral_prep_requires_all_four_albedos():
    from woof.core.rrtmg_legacy_prep import _require_spectral_albedo
    with pytest.raises(ValueError,match='four spectral albedos'):
        _require_spectral_albedo(8,None)
    with pytest.raises(ValueError,match='four spectral albedos'):
        _require_spectral_albedo(8,{'asdir':0.1})
    _require_spectral_albedo(8,dict(asdir=.1,asdif=.2,aldir=.3,aldif=.4))
    _require_spectral_albedo(2,None)


@pytest.mark.gpu
def test_live_precipitation_acceptance_and_clock_feed_the_record_surface():
    import cupy as cp
    import numpy as np
    from datetime import datetime, timezone
    from woof.core.physics import initialize_physics, MicrophysicsDiagnostics
    from woof.core.state import DomainState
    config = cfg(cu_physics=1, cudt_minutes=5, moist=True)
    state = DomainState(config)
    driver = initialize_physics(state, config, radiation=lambda **kw: None,
        radiation_start_time=datetime(2024,1,1,tzinfo=timezone.utc),
        radiation_latitude=35.0, radiation_longitude=-100.0,
        landuse_dataset='SSIB', ivgtyp=7)
    shape = state.mup.shape
    rain = cp.full(shape, .1, cp.float32)
    diagnostic = MicrophysicsDiagnostics(rainnc=rain.copy(),rainncv=rain,
                                         sr=cp.zeros(shape,cp.float32))
    driver.accept_microphysics(diagnostic)
    assert bool(cp.all(driver.fields['surface_rainncv'] == np.float32(.1)))
    driver.cu_pratec.fill(.01)
    driver.cu_nca.fill(100)
    driver._advance_cumulus_clock(state, config)
    expected = np.float32(.01) * np.float32(config.dt)
    assert bool(cp.all(driver.fields['surface_raincv'] == expected))


def test_history_alias_uses_the_selected_registry_description():
    from woof.io.wrf_output_schema import history_field_schema, NOAHMP_OUTPUT_FIELDS
    row = history_field_schema('ISNOW', sf_surface_physics=8)
    assert row.dtype == 'i4' and row.units == ''
    assert row.description == 'ssib-snow ISNOW'
    assert row.registry == 'Registry/registry.ssib:30'
    assert history_field_schema('ISNOW', sf_surface_physics=4) is NOAHMP_OUTPUT_FIELDS['isnowxy']


def test_rust_history_writer_publishes_the_column_alias_metadata(tmp_path):
    import numpy as np
    from woof.io.wrfout import WrfoutWriter
    with WrfoutWriter(tmp_path/'history.nc', nx=2,ny=2,nz=3,dx=1000,dy=1000,
          global_attrs={'SF_SURFACE_PHYSICS':8}, engine='rust',
          field_schema={'ISNOW':np.zeros((2,2),np.int32)}) as writer:
        variable = writer.ds.variables['ISNOW']
        assert variable.description == 'ssib-snow ISNOW'
        assert variable.units == ''
        writer.write_frame('2024-01-01_00:00:00', {'ISNOW':np.zeros((2,2),np.int32)})
