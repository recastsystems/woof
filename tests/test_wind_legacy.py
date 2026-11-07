"""The stock WRF V3.5 filename selector and its distinct turbine physics."""
from dataclasses import asdict, replace
import re
import tomllib
import pytest
from woof.config import RunConfig
from woof.wind_legacy import read_spec, validate_config


def config(**kw):
    return RunConfig(nx=9,ny=9,nz=40,dx=3000,dy=3000,ztop=20000,
                     dt=15,run_seconds=60,**kw)


def test_native_windspec_uses_explicit_analytic_parameters(tmp_path):
    path=tmp_path/'windspec.in'
    path.write_text('! header\n1 34.1 -79.2 100 60 .158 2 4 27 trailing native-ignored fields'
                    + ' '*256 + '! past the native record\n2,35,-80,120,80,.2,5,3D0,30\n')
    data,_=read_spec(path)
    assert [r['grid_id'] for r in data['turbines']]==[1,2]
    assert data['turbines'][0]['rated_power_mw']==2
    assert data['turbines'][1]['cut_in_m_s']==3
    assert 'types' not in data # no sampled substitute for dragcof


def test_native_windspec_bad_row_is_named(tmp_path):
    path=tmp_path/'windspec.in';path.write_text('1 34 -79 100 60 .158 2 4\n')
    with pytest.raises(ValueError,match='nine fields'):read_spec(path)


@pytest.mark.parametrize('spec',['none','ideal','windspec.in'])
def test_stock_selector_import_and_original_path(tmp_path,spec):
    from woof.namelist_import import import_namelists
    from test_namelist_import import INPUT_TEXT,_pair
    if spec=='windspec.in':
        (tmp_path/spec).write_text('1 34 -79 100 60 .158 2 4 27\n2 35 -80 120 80 .2 5 3 30\n')
    text=INPUT_TEXT.replace('&physics',f'&physics\n windturbines_spec="{spec}", td_turbgridid=2, td_ewfx=2, td_ewfy=1,')
    text=re.sub(r'bl_pbl_physics\s*=.*','bl_pbl_physics=5,5,',text)
    output,_=import_namelists(*_pair(tmp_path,inp=text),name='legacy')
    raw=tomllib.loads(output)
    if spec=='none':assert all('windturbines_spec' not in d for d in raw['domain'])
    elif spec=='ideal':
        assert 'windturbines_spec' not in raw['domain'][0]
        assert raw['domain'][1]['windturbines_spec']=='ideal'
        assert raw['domain'][1]['td_ewfx']==2
    else:assert all(d['windturbines_spec']==str(tmp_path/'windspec.in') for d in raw['domain'])


def test_legacy_and_modern_generations_cannot_be_combined():
    with pytest.raises(ValueError,match='cannot be combined'):
        config(windturbines_spec='ideal',wind_farm_scheme='mav')


def test_inactive_identity_remains_original():
    from woof.wind_table import drop_inert_config
    from woof.wind_legacy import DEFAULTS
    cfg=asdict(config());drop_inert_config(cfg)
    assert 'windturbines_spec' not in cfg
    assert not set(DEFAULTS)&cfg.keys()


def test_legacy_selector_requires_native_mynn_pair():
    with pytest.raises(ValueError,match='MYNN'):
        validate_config(config(windturbines_spec='ideal'))
