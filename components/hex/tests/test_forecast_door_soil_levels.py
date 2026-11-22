"""The forecast door refuses an init whose soil column is another land model's than the suite's, by name.

The defect (published hex 0.5.0 and the 0.6.0 candidate, 2026-11-11): a plain forecast, default suite
convection_permitting (Noah, four soil layers), on the 120 km hrrrv5 start file (RUC, nine soil levels) stopped at
physics start with "could not broadcast input array from shape (4,1) into shape (9,40962)" and rc 2.
"""
import site_support
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.forecast_v842 import native_route


def _init(tmp_path, levels, name=None):
    import netCDF4
    path = tmp_path / (name or f'init{levels}.nc')
    with netCDF4.Dataset(path, 'w', format='NETCDF3_64BIT_OFFSET') as dataset:
        dataset.createDimension('Time', None)
        dataset.createDimension('nCells', 12)
        dataset.createDimension('nSoilLevels', levels)
        dataset.createVariable('dzs', 'f4', ('Time', 'nSoilLevels', 'nCells'))[0] = np.zeros((levels, 12), 'f4')
    return path


def _arguments(suite, init):
    return SimpleNamespace(physics_namelist=None, physics_suite=suite, physics=None, physics_backend='mpasphys',
                           init=init)


def test_default_suite_refuses_a_nine_level_ruc_init_by_name(tmp_path):
    init = _init(tmp_path, 9)
    with pytest.raises(ValueError) as refusal:
        native_route(_arguments('convection_permitting', init), default_mode='correct')
    text = str(refusal.value)
    assert 'broadcast' not in text
    assert f'{init} carries 9 soil levels (the soil column of sf_ruc)' in text
    assert "suite 'convection_permitting', whose land surface sf_noah integrates 4" in text
    assert 'Run it with --physics-suite hrrrv5' in text
    assert 'woof hex init --land-surface noah (4 levels)' in text


def test_hrrrv5_refuses_a_four_level_noah_init_by_name(tmp_path):
    init = _init(tmp_path, 4)
    with pytest.raises(ValueError) as refusal:
        native_route(_arguments('hrrrv5', init), default_mode='correct')
    text = str(refusal.value)
    assert 'carries 4 soil levels (the soil column of sf_noah or sf_noahmp)' in text
    assert "suite 'hrrrv5', whose land surface sf_ruc integrates 9" in text
    assert '--physics-suite convection_permitting or mesoscale_reference' in text
    # (Recast WOOF 1.1.7) the remedy is the init that runs hrrrv5 (its producer's, with --geog), not
    # --land-surface ruc, whose init stopped the forecast with KeyError 'nscat'
    assert 'woof hex init --physics-suite hrrrv5 --geog' in text
    assert 'which writes 9 levels' in text and '--static INIT.rrfs-static.nc' in text
    assert '--land-surface ruc' not in text


@pytest.mark.parametrize('suite,levels', [('convection_permitting', 4), ('mesoscale_reference', 4), ('hrrrv5', 9)])
def test_a_suite_takes_its_own_soil_column(tmp_path, suite, levels):
    config, _ = native_route(_arguments(suite, _init(tmp_path, levels)), default_mode='correct')
    assert config is not None and config.selectors['land_surface'] in ('sf_noah', 'sf_ruc')


def test_an_init_the_check_cannot_read_is_not_judged_here(tmp_path):
    bare = tmp_path / 'bare.nc'
    bare.write_bytes(b'')
    for init in (None, bare, tmp_path / 'missing.nc'):
        config, _ = native_route(_arguments('convection_permitting', init), default_mode='correct')
        assert config is not None


def test_a_land_surface_named_over_the_suite_decides_the_column(tmp_path):
    # hrrrv5 with Noah named takes Noah's four layers; the nine-level init is then refused, the four-level one runs on.
    arguments = _arguments('hrrrv5', _init(tmp_path, 9))
    arguments.physics = ['config_lsm_scheme=sf_noah']
    with pytest.raises(ValueError, match='land surface sf_noah integrates 4'):
        native_route(arguments, default_mode='correct')
    arguments.init = _init(tmp_path, 4)
    config, _ = native_route(arguments, default_mode='correct')
    assert config.selectors['land_surface'] == 'sf_noah'
