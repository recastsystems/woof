"""Native tendency amounts: rate timing, inactive windows and continuation."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof.core import physics_tendency_output as out

def test_native_restart_policy_and_checkpoint(monkeypatch,tmp_path):
 from woof.io import restart
 from woof.ingest import wrfrst
 from woof.io.wrf_output_schema import OUTPUT_FIELDS_BY_NETCDF_NAME
 from test_restart import _cfg,_shim_state,_fill_setup
 cfg=_cfg(acc_phy_tend=1,output_interval_s=60.)
 src=_shim_state(cfg,monkeypatch);_fill_setup(src);expected={}
 for i,name in enumerate(out.PERSISTENT):
  a=src.existing_scratch(name);a[...] = np.arange(a.size,dtype='f4').reshape(a.shape)/7+i
  expected[name]=a.tobytes()
  if name in out.NAMES:
   assert wrfrst.scratch_restore_plan(name,3)[0:2]==('cold',0.)
   assert OUTPUT_FIELDS_BY_NETCDF_NAME[name.upper()].dtype=='f4'
 path=restart.write_restart(tmp_path/'tendencies.npz',src,cfg)
 dst=_shim_state(cfg,monkeypatch);_fill_setup(dst);restart.restore_restart(path,dst,cfg)
 for name,b in expected.items():assert dst.existing_scratch(name).tobytes()==b
 with np.load(path,allow_pickle=False) as d:assert 'scratch/tendency_qv_before' not in d.files
 d={'acc_phy_tend':0,'mp_physics':8};out.drop_inert(d);assert d=={'mp_physics':8}
 assert not set(out.PERSISTENT)&set(out.scratch_shapes(_cfg()))

@pytest.mark.gpu
@pytest.mark.parametrize('now',[0.,12.,60.,16777217.])
@pytest.mark.parametrize('no_heat',[0,1])
def test_real_qv_difference_window_and_inactive_shallow(now,no_heat):
 import cupy as cp
 shape=(2,4,5);cfg=SimpleNamespace(acc_phy_tend=1,nx=5,ny=4,nz=2,
  dt=12.,output_interval_s=60.,specified=True,nested=False,spec_zone=1,
  mp_physics=8,no_mp_heating=no_heat,cu_physics=3)
 arrays={k:cp.full(s,3.,dtype='f4') for k,s in out.scratch_shapes(cfg).items()}
 arrays['tendency_epoch'][...] = 0.
 qv=cp.full(shape,.01,dtype='f4');rate=cp.full(shape,.0003125,dtype='f4')
 fields={k:rate for k in ('rthcuten','rqvcuten','rucuten','rvcuten','rthblten','rqvblten','rublten','rvblten')}
 state=SimpleNamespace(physics=SimpleNamespace(fields=fields,rthratenlw=rate,rthratensw=rate),
  qv=qv,h_diabatic=rate,elapsed_seconds=now,domain_start_offset=0.,existing_scratch=arrays.get)
 out.prepare_microphysics(state,cfg)
 old=cp.asnumpy(qv);qv[...] += np.float32(.000018)
 final=cp.asnumpy(qv);out.update(state,cfg)
 reset=np.float32(now)%np.float32(60.)==0.
 for name in out.NAMES:
  actual=cp.asnumpy(arrays[name]);base=np.float32(0. if reset else 3.)
  if name in out.NAMES[6:10]:expected=np.full(shape,3.,dtype='f4')
  elif name=='aqvmpten':
   delta=np.zeros(shape,dtype='f4')
   if not no_heat:delta[:,1:-1,1:-1]=(final[:,1:-1,1:-1]-old[:,1:-1,1:-1])/np.float32(12.)
   expected=base+delta*np.float32(12.)
  else:expected=np.full(shape,base+np.float32(.0003125)*np.float32(12.),dtype='f4')
  assert np.array_equal(actual.view('u4'),expected.view('u4')),name

def test_stock_per_domain_admission():
 import importlib.util,tomllib
 from pathlib import Path
 from woof.fortran_namelist import parse_namelist_text
 from woof.namelist_import import import_parsed_namelists
 spec=importlib.util.spec_from_file_location('tend_census',Path(__file__).parents[1]/'tools/diag_wrf471_oracle/census.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);fixture,_=mod.fixture()
 inp=parse_namelist_text(fixture['INPUT_TEXT']);inp['physics']['acc_phy_tend']=[1,0]
 text,_=import_parsed_namelists(parse_namelist_text(fixture['WPS_TEXT']),inp,wps_path='namelist.wps',input_path='namelist.input',name='tendencies',source_text=None)
 obj=tomllib.loads(text);assert obj['shared']['acc_phy_tend']==1
 assert any(d.get('acc_phy_tend')==0 for d in obj['domain'])
