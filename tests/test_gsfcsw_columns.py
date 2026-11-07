"""WRF 4.7.1 old GSFC shortwave contracts (module identical to 4.6.1)."""
from pathlib import Path
import shutil
import numpy as np
import pytest

def test_column_door_exists():
    from woof.core.gsfcsw import gsfcsw_columns
    assert callable(gsfcsw_columns)

def test_packaged_coefficients_match_pin():
    from woof.core.gsfcsw import load_tables,TABLE_ASSET
    assert load_tables().size*4 == TABLE_ASSET.bytes

def test_one_coefficient_bit_flip_is_refused(tmp_path):
    from woof.core.gsfcsw import load_tables,TABLE_ASSET,TABLE_ROOT
    file=tmp_path/TABLE_ASSET.filename
    shutil.copyfile(TABLE_ROOT/TABLE_ASSET.filename,file)
    raw=bytearray(file.read_bytes());raw[4*758]^=1;file.write_bytes(raw)
    with pytest.raises(ValueError,match='SHA|sha|hash|digest'):
        load_tables(tmp_path)

def test_shared_loader_options_preserve_float32():
    from woof.core.kernels import module_options,module_source
    assert module_options('gsfcsw') == ('-std=c++17','--fmad=false','--ftz=false')
    source=module_source('gsfcsw')
    assert 'gfk_exp' in source and 'gfk_pow' in source
    assert '--use_fast_math' not in module_options('gsfcsw')

def test_fixture_covers_sun_clouds_and_warm_switch():
    fixture=np.load(Path(__file__).resolve().parents[1]/'tools/gsfcsw_wrf461_oracle/columns.npz')
    for nlay in (8,48):
        a=fixture[f'in_{nlay}']; f=fixture[f'flags_{nlay}'];n=nlay+1
        assert a.shape[0]>=64
        assert np.any(a[:,-2]<0) and np.any(a[:,-2]>.9)
        assert np.any(a[:,-1]>.8)
        assert np.any(f[:,0]==0) and np.any(f[:,0]==1)
        assert np.any(a[:,5*n:6*n]>0) and np.any(a[:,6*n:7*n]>0)

def test_option2_declares_state_outputs_and_cadence():
    from woof.physics_registry import physics_registry
    from woof.config import RA_SW_PHYSICS_ACCEPTED
    row=physics_registry()['components']['radiation']['options']['gsfc-shortwave']
    assert 2 in RA_SW_PHYSICS_ACCEPTED
    assert row['selectors']=={'ra_lw_physics':0,'ra_sw_physics':2}
    assert row['state_fields']==['taucldc','taucldi']
    assert row['cadence_parameter']=='radt'
    assert row['consumers']['stock_callable_class']=='woof.core.gsfcsw.GSFCSWShortwaveRadiation'
    assert 'GSW' in row['outputs'] and 'RSWTOA' in row['outputs']

def test_namelist_option2_has_native_mapping():
    from woof.namelist_import import _RA_SW_MAP
    assert _RA_SW_MAP[2][0]==2


@pytest.mark.parametrize('ghg', (None, 0, 1))
def test_gsfc_namelist_preserves_the_longwave_gas_selection(ghg):
    from test_ghg_input_camtr import _import
    physics = {'ra_lw_physics': [1], 'ra_sw_physics': [2]}
    if ghg is not None:
        physics['ghg_input'] = [ghg]
    shared, report = _import(physics)
    assert shared.get('ghg_input', 0) == (1 if ghg is None else ghg)
    assert shared['ra_sw_physics'] == 2

def test_domain_center_overrides_grid_mean_for_ozone_profile(monkeypatch):
    import sys
    from datetime import datetime
    from woof.core.gsfcsw import GSFCSWShortwaveRadiation
    monkeypatch.setitem(sys.modules,'cupy',np)
    latitude=np.full((2,2),30.01,'f4');longitude=np.zeros((2,2),'f4')
    radiation=GSFCSWShortwaveRadiation(datetime(2024,6,1),latitude,longitude,center_lat=30.)
    assert radiation.center_lat==30.
    assert radiation.restart_identity['center_lat']==30.

def test_column_requires_its_calendar_and_solar_setup():
    from woof.core.gsfcsw import packed_columns
    with pytest.raises(TypeError,match='julday'):
        packed_columns(np.zeros((1,83),'f4'),8)


def test_coefficient_words_are_restart_setup_not_carried_state(monkeypatch):
    import sys
    from datetime import datetime
    from woof.core.gsfcsw import GSFCSWShortwaveRadiation
    from woof.io.restart import _callable_state_check,RADIATION_CALLABLE_ARRAYS,RADIATION_CALLABLE_CONTAINERS
    monkeypatch.setitem(sys.modules,'cupy',np)
    scheme=GSFCSWShortwaveRadiation(datetime(2024,6,1),np.zeros((2,2),'f4'),np.zeros((2,2),'f4'),center_lat=30.)
    _callable_state_check(scheme,RADIATION_CALLABLE_ARRAYS,RADIATION_CALLABLE_CONTAINERS,'radiation')
    before=scheme.restart_identity
    scheme.gsfc_coefficients=scheme.gsfc_coefficients.copy()
    scheme.gsfc_coefficients.view('u4')[758]^=1
    assert scheme.restart_identity!=before


def test_registry_builder_retains_option_from_preport_input():
    import copy
    from woof.physics_registry import physics_registry
    from tools.build_registry import build
    registry = copy.deepcopy(physics_registry())
    registry['components']['radiation']['options'].pop('gsfc-shortwave', None)
    rebuilt = build(registry)
    row = rebuilt['components']['radiation']['options']['gsfc-shortwave']
    assert row['selectors'] == {'ra_lw_physics': 0, 'ra_sw_physics': 2}
    assert row['asset_requirements'][0]['resolution']['data_relative'] == 'wrf_radiation/gsfcsw'
    assert row['consumers']['stock_callable_class'] == 'woof.core.gsfcsw.GSFCSWShortwaveRadiation'


def test_catalog_describes_the_column_scope_without_forecast_skill():
    from woof.physics_catalog import table
    words = table()['descriptions']['radiation']['gsfc-shortwave']
    assert 'Column parity' in words
    assert 'forecast skill is unverified' in words


def test_arena_keeps_logical_operators_and_multiline_shifted_addresses():
    from tools.gsfcsw_wrf461_oracle.arena import coalesce_arrays
    source = 'real *a; real *b; if (a[1] > 0 && b[2] > 0) call(&\n b[3]);'
    result = coalesce_arrays(source)
    assert 'a[1] > 0 && b[2] > 0' in result
    assert 'call((b + (3)))' in result


def test_arena_charge_includes_partial_warp_and_chunk_limit():
    from woof.core.gsfcsw import WORKSPACE_WORDS, workspace_bytes
    assert workspace_bytes(1) == 32*WORKSPACE_WORDS*4
    assert workspace_bytes(33) == 64*WORKSPACE_WORDS*4
    assert workspace_bytes(4097) == 4096*WORKSPACE_WORDS*4


@pytest.mark.parametrize('longwave',(0,4))
def test_offline_child_passes_its_domain_center_to_gsfc(monkeypatch,longwave):
    from datetime import datetime
    from types import SimpleNamespace
    from test_offline_child_ozone import _legacy_child_config, _capture, _initial
    from woof import offline_child_run
    cfg=_legacy_child_config(ra_lw_physics=longwave,ra_sw_physics=2)
    seen=_capture(monkeypatch)
    initial=_initial(cfg)
    offline_child_run._initialize_child_physics(None,cfg,initial,None,
        datetime(2024,6,1),radiation_center_lat=30.)
    assert seen['physics_kwargs']['radiation_center_lat']==30.
    if longwave==4:assert seen['radiation_kwargs']['center_lat']==30.
