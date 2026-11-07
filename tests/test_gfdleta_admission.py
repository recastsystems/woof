from dataclasses import replace
import pytest
from woof.config import RunConfig,validate_run_config,radiation_scheme_ids
from woof.physics_registry import physics_registry
from woof.namelist_import import import_namelists

def config(**kwargs):return RunConfig(nx=8,ny=8,nz=12,dx=12000.,dy=12000.,ztop=18000.,dt=30.,run_seconds=60.,ra_lw_physics=99,ra_sw_physics=99,**kwargs)

def test_bare_config_admits_gfdl_pair():
 cfg=config();validate_run_config(cfg)
 assert radiation_scheme_ids(cfg)==(99,99)
 assert cfg.co2tf==1
 row=physics_registry()['components']['radiation']['options']['wrf-gfdleta']
 assert row['implemented'] and row['selectors']=={'ra_lw_physics':99,'ra_sw_physics':99}

@pytest.mark.parametrize('co2tf',[0,1])
def test_co2_both_arms_are_carried(co2tf):validate_run_config(config(co2tf=co2tf))

@pytest.mark.parametrize('co2tf',[-1,2,True])
def test_invalid_co2_arm_refuses(co2tf):
 with pytest.raises(ValueError,match='co2tf'):validate_run_config(config(co2tf=co2tf))

def test_public_namelist_vocabulary_imports_99_and_co2tf(tmp_path):
 from test_namelist_import import WPS_TEXT,INPUT_TEXT
 wps=tmp_path/'namelist.wps';inp=tmp_path/'namelist.input'
 wps.write_text(WPS_TEXT)
 import re
 text=re.sub(r'(ra_lw_physics\s*=)[^\n]+',r'\1 99, 99,',INPUT_TEXT)
 text=re.sub(r'(ra_sw_physics\s*=)[^\n]+',r'\1 99, 99,',text)
 text=text.replace('&physics','&physics\n co2tf = 0,')
 inp.write_text(text)
 toml,report=import_namelists(wps,inp)
 assert 'ra_lw_physics = 99' in toml and 'ra_sw_physics = 99' in toml
 assert 'co2tf = 0' in toml
 assert 'does not offer' not in str(report)

def test_explicit_work_buffers_are_priced():
 from woof.core.preflight import gfdleta_column_shapes,physics_array_shapes
 shapes=gfdleta_column_shapes(config(),column_chunk=3)
 assert shapes['gfdleta/column_work']==((3,234184),1)
 assert shapes['gfdleta/packed']==((64,1204),4)
 resident=physics_array_shapes(config())
 assert resident['radiation/gfdleta_tables']==(123192,)
 assert resident['cldfra']==(12,8,8) and resident['olr']==(8,8)


def test_table_family_override_is_the_same_root_in_plan_and_runtime(monkeypatch,tmp_path):
 from pathlib import Path
 import shutil
 from woof.physics_registry import resolve_asset_requirement
 from woof.table_assets import physics_table_root
 source=Path(__file__).parents[1]/'woof/data/gfdleta'
 root=tmp_path/'gfdleta';shutil.copytree(source,root)
 monkeypatch.setenv('WOOF_TABLE_ROOT',str(tmp_path))
 row=physics_registry()['components']['radiation']['options']['wrf-gfdleta']['asset_requirements'][0]
 result=resolve_asset_requirement(row)
 assert result['resolved'] and Path(result['root'])==root
 assert physics_table_root('gfdleta')==root
 (root/'co2-raw.npy').unlink()
 result=resolve_asset_requirement(row)
 assert not result['resolved'] and 'co2-raw.npy' in result['missing']
