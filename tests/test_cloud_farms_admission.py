"""Stock namelist and disabled identity contracts for cloud/FARMS wiring."""
import dataclasses
import tomllib
import pytest
from woof.config import RunConfig, validate_radiation_driver_options
from woof.namelist_import import import_namelists
from test_namelist_import import INPUT_TEXT, WPS_TEXT, _pair

def cfg(**kwargs):
    return RunConfig(nx=8,ny=8,nz=39,dx=30000.,dy=30000.,ztop=20000.,dt=180.,run_seconds=360.,**kwargs)


def test_stock_cloud_farms_import_two_domains(tmp_path):
    text = INPUT_TEXT.replace('&physics\n',
        '&physics\n icloud=3, swint_opt=2, couple_farms=.true.,\n')
    output, report = import_namelists(*_pair(tmp_path,wps=WPS_TEXT,inp=text),
                                     rrtmg_variant='rrtmg_legacy')
    shared=tomllib.loads(output)['shared']
    assert shared['icloud']==3
    assert shared['swint_opt']==2
    assert shared['couple_farms'] is True
    assert not any(row.key in {'icloud','swint_opt','couple_farms'} for row in report.substitutions)


def test_missing_ice_obeys_native_fatal():
    config=cfg(icloud=3,moist=True,mp_physics=1)
    with pytest.raises(ValueError,match='missing QC or QI'):
        validate_radiation_driver_options(config)


def test_farms_stock_aerosol_seam_is_not_replaced_with_hrrr():
    config=cfg(moist=True,mp_physics=28,swint_opt=2,aer_opt=3,
                  ra_physics=4,ra_rrtmg_variant='rrtmg_legacy')
    with pytest.raises(ValueError,match='stock WRF gt_aod'):
        validate_radiation_driver_options(config)


def test_disabled_farms_does_not_change_checkpoint_echo():
    from woof.io.restart import configuration_echo
    config=cfg()
    assert 'couple_farms' not in configuration_echo(config)
    enabled=dataclasses.replace(config,swint_opt=2,couple_farms=True)
    assert configuration_echo(enabled)['couple_farms'] is True


def test_memory_inventory_covers_cloud_and_farms_with_separate_streams():
    from woof.core.preflight import cloud_farms_transient_shapes
    assert cloud_farms_transient_shapes(cfg()) == {}
    # Split Goddard/Dudhia selection used to fall outside the RRTMG-only
    # cloud-fraction resident allocation test. Both still call the adapter.
    cloud = cfg(icloud=3, ra_lw_physics=0, ra_sw_physics=1)
    shapes = cloud_farms_transient_shapes(cloud)
    assert shapes['cloud_fraction3/profiles'] == (18, 64, 39)
    assert not any(name.startswith('farms/') for name in shapes)
    farms = dataclasses.replace(cloud, swint_opt=2)
    assert cloud_farms_transient_shapes(farms)['farms/profiles'] == (12, 40, 64)


def test_stock_aerosol_and_analytic_thompson_source_are_reachable():
    from woof.config import validate_aerosol_source_options
    config=cfg(moist=True,mp_physics=28,swint_opt=2,aer_opt=3,
               ra_physics=4,ra_rrtmg_variant='rrtmg_legacy',
               rrtmg_aer3_source='wrf_461',aer_init_opt=0,wif_input_opt=1,
               mp28_aerosol_source='synthetic')
    validate_aerosol_source_options(config)
    validate_radiation_driver_options(config)
    # The analytic allocation does not waive the separate black-carbon hold.
    with pytest.raises(ValueError,match='wif_input_opt'):
        validate_aerosol_source_options(dataclasses.replace(config,wif_input_opt=2))


def test_cumulus_radiation_router_matches_native_derived_modes():
    from woof.core.cloud_fraction import icloud_cu
    assert icloud_cu(cfg(cu_physics=1,cu_rad_feedback=True)) == 2
    for cu in (3,5,93):
        assert icloud_cu(cfg(cu_physics=cu,cu_rad_feedback=True)) == 1
    assert icloud_cu(cfg(cu_physics=11)) == 2
    assert icloud_cu(cfg(cu_physics=1)) == 0


def test_native_warm_start_preserves_eclipse_scalars_and_held_ozone():
    import numpy as np
    from woof.ingest.wrfrst import _host_field,fields_restore_plan,_disposition
    raw={'ELAT_TRACK':np.float32(40.05),'O3RAD':np.zeros((3,4,5),np.float32)}
    plane=_host_field(raw,'ELAT_TRACK',(4,5),label='fields/elat_track')
    np.testing.assert_array_equal(plane,np.full((4,5),np.float32(40.05)))
    assert fields_restore_plan('ecobsc',{'ECOBSC'}) == ('fill','ECOBSC')
    assert fields_restore_plan('eclipse_initialized',raw)[:2] == ('cold',1.0)
    assert fields_restore_plan('eclipse_clear_heating',raw)[:2] == ('cold',0.0)
    assert _disposition('O3RAD').kind == 'read'
