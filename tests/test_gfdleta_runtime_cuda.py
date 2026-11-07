"""Complete public radiation callable, including run-level initialization."""
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
import os
import numpy as np
import pytest
pytestmark=pytest.mark.skipif(os.environ.get('GPUWM_NO_LOCAL_GPU')=='1',reason='CPU-only check')
@pytest.mark.parametrize('n',[12,24,42,60])
def test_runtime_columns(n):
 import cupy as cp
 from woof.core.gfdleta import GFDLEtaRadiation,_LAYOUT
 from woof.config import RunConfig
 desc={x['name']:x for x in _LAYOUT['fields']}
 root=Path(__file__).parent/'fixtures/gfdleta/column'
 with np.load(root/f'columns-{n}.npz') as z,np.load(root/f'init-{n}.npz') as init:
  f=z['floats']
  def get(name,nk=None):
   d=desc[name];a=f[:,d['offset']:d['offset']+(nk or 1)]
   return cp.asarray(a.T.reshape((nk,1,8) if nk else (1,8)))
  atm={key:get(name,n+1 if name=='p8w' else n) for name,key in [('pi3d','exner'),('p8w','p_interface'),('dz8w','dz'),('rho_phy','rho'),('p_phy','pressure'),('t','temperature'),('qv','qv'),('qw','qc'),('qi','qi'),('qs','qs')]}
  fields=GFDLEtaRadiation.retained_fields(n,1,8)
  fields.update({key:get(name) for name,key in [('xland','xland'),('tsk2d','tsk'),('albedo','albedo'),('snow','snow'),('vegfra','shdfac'),('glw','glw'),('rswin','swdown')]})
  for name in ['htop','hbot','htopr','hbotr','cuppt']:fields[name][...]=get(name)
  state=SimpleNamespace(znw=cp.asarray(init['sigma_full'][:n+1]),znu=cp.asarray(init['sigma_half'][:n]),p_top=1000.,elapsed_seconds=0.)
  cfg=RunConfig(nx=8,ny=1,nz=n,dx=12000.,dy=12000.,ztop=18000.,dt=30.,run_seconds=60.,ra_lw_physics=99,ra_sw_physics=99,radt=10.)
  factor=np.float32(.01745329238474369)
  lon=get('glon')/factor
  engine=GFDLEtaRadiation(datetime(2001,3,21,12),get('glat'),lon,p_top=1000.,column_chunk=3)
  result=engine(atmosphere=atm,fields=fields,state=state,cfg=cfg)
  outputs={'RTHRATENLW':result.rthratenlw,'RTHRATENSW':result.rthratensw,'RTHRATEN':result.rthratenlw+result.rthratensw,'CLDFRA':result.cldfra,'GLW':result.glw,'GSW':result.gsw,'SWDOWN':result.swdown,'SWDOWNC':fields['swdownc'],'RLWTOA':result.olr,'RSWTOA':fields['rswtoa'],'CFRACL':fields['cfracl'],'CFRACM':fields['cfracm'],'CFRACH':fields['cfrach'],'CZMEAN':result.coszen,'ACFRST':fields['acfrst'],'ACFRCV':fields['acfrcv'],'HTOPR':fields['htopr'],'HBOTR':fields['hbotr'],'CUPPT':fields['cuppt']}
  keymap={'RTHRATEN':'thraten','RTHRATENLW':'thratenlw','RTHRATENSW':'thratensw','SWDOWN':'rswin','SWDOWNC':'rswinc'}
  bad=[]
  for field,a in outputs.items():
   mask=z['fields']==field;inds=z['indices'][mask]-desc[keymap.get(field,field.lower())]['offset'];cols=z['columns'][mask]
   host=cp.asnumpy(a).reshape((-1,8)).T.view(np.uint32);got=host[cols,inds];want=z['reference'][mask];miss=np.flatnonzero(got!=want)
   if len(miss):bad.append((field,len(miss),[(int(cols[k])+1,int(inds[k])+1,f'{got[k]:08X}',f'{want[k]:08X}') for k in miss[:3]]))
  assert not bad,bad


def test_bare_99_pair_reaches_native_physics_driver():
 from tilestream import harness
 from woof.config import RunConfig
 from woof.core.physics import initialize_physics
 from woof.core.gfdleta import GFDLEtaRadiation
 cfg=RunConfig(terrain_opt=1,nx=48,ny=40,nz=12,dx=12000.,dy=12000.,ztop=18000.,dt=30.,run_seconds=60.,ra_lw_physics=99,ra_sw_physics=99)
 geo=harness.make_geography(cfg)
 state,_=harness.make_physics_state(cfg,123,geography=geo)
 driver=initialize_physics(state,cfg,radiation_start_time=datetime(2001,3,21,12),radiation_latitude=geo.lat,radiation_longitude=geo.lon)
 assert isinstance(driver.radiation_callable,GFDLEtaRadiation)
 assert driver.fields['cuppt'].shape==(40,48)
 assert driver.olr.shape==(40,48)

 # Build the same health descriptors used before the first real forecast step.
 from woof.core.health import StateHealthValidator,FieldRule
 validator=StateHealthValidator(state);validator._refresh()
 checked={f.name:f for f in validator.fields}
 for name in ['surface.ncfrst','surface.ncfrcv']:
  assert checked[name].rule==FieldRule('surface',0.)

 # Resolve the final digest after real run-level initialization, not just a
 # lazy constructor. Counters stay in the serialized manifest.
 from woof.core.gfdleta import GFDLTables
 from woof.io.restart import _driver_manifest,_callable_state_check,RestartManifestError
 engine=driver.radiation_callable
 assert type(engine.p_top) is float
 engine.tables=GFDLTables(state.znw,state.znu,float(state.p_top),start_time=engine.start_time)
 manifest=_driver_manifest(driver)
 for key in ['ncfrst','ncfrcv']:
  assert any(value is driver.fields[key] for value in manifest.values())
 with pytest.raises(RestartManifestError,match='unclassified arrays'):
  _callable_state_check(SimpleNamespace(tables=engine.tables),frozenset(),frozenset(),'radiation')
