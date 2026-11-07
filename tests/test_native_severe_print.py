"""Native controls, carrier lifetime and horizontal reductions."""
from pathlib import Path
import importlib.util,tomllib
import numpy as np
import pytest
from test_restart import _cfg
from woof.core import afwa_diagnostics as afwa,misc_diag_print as md

def test_stock_controls_and_inert_continuation_identity():
 from woof.checkpoint_identity import drop_capability_campaign_keys
 from woof.fortran_namelist import parse_namelist_text
 from woof.namelist_import import import_parsed_namelists
 spec=importlib.util.spec_from_file_location('severe_census',Path(__file__).parents[1]/'tools/diag_wrf471_oracle/census.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);fixture,_=m.fixture()
 inp=parse_namelist_text(fixture['INPUT_TEXT']);inp['afwa']={'afwa_severe_opt':[0,1]};inp['time_control']['diag_print']=[2]
 text,_=import_parsed_namelists(parse_namelist_text(fixture['WPS_TEXT']),inp,wps_path='namelist.wps',input_path='namelist.input',name='severe',source_text=None)
 v=tomllib.loads(text)['shared'];assert v['afwa_severe_opt']==v['afwa_diag_opt']==1;assert v['diag_print']==2
 old={'mp_physics':9};new={**old,'afwa_severe_opt':0,'diag_print':0}
 drop_capability_campaign_keys(old);drop_capability_campaign_keys(new);assert new==old

def test_shared_and_private_carriers_have_explicit_lifetimes():
 from woof.core.preflight import scratch_slot_lifetime
 cfg=_cfg(moist=True,mp_physics=9,afwa_diag_opt=1,afwa_severe_opt=1,diag_print=2)
 shapes=afwa.scratch_shapes(cfg)|md.scratch_shapes(cfg)
 assert len(afwa.SEVERE_HISTORY)==15
 assert set(afwa.SEVERE_HISTORY)|set(afwa.SEVERE_STATE)<=set(afwa.PERSISTENT)
 assert not set(afwa.SEVERE_WORK)&set(afwa.PERSISTENT)
 for name in shapes:assert scratch_slot_lifetime(name) is not None,name
 assert not md.scratch_shapes(_cfg())
 assert set(md.PERSISTENT)<=set(shapes)

def test_severe_registry_requires_the_native_qng_flag():
 from woof.core.afwa_severe import active_species
 assert active_species(_cfg(mp_physics=9)) and active_species(_cfg(mp_physics=10))
 assert not active_species(_cfg(mp_physics=8))

@pytest.mark.gpu
def test_parallel_print_excludes_halos_and_reports_global_max_location():
 import cupy as cp
 from woof.core.kernels import get_kernel
 nx,ny=8,6;planes=[cp.full((ny,nx),10000,dtype='f4') for _ in range(9)]
 for k,a in enumerate(planes):a[1:5,2:6]=np.float32(k+1)
 partial=cp.empty((14,1),dtype='f4');index=cp.empty((3,1),dtype='i4');out=cp.empty(14,dtype='f4');ix=cp.empty(3,dtype='i4')
 get_kernel('misc_diag_print','misc_print_partial')((1,14),(256,),(*planes,partial,index,np.float32(12),*(np.int32(x) for x in (nx,2,6,1,5,2,10,20,100))))
 get_kernel('misc_diag_print','misc_print_finish')((14,),(128,),(partial,index,out,ix,np.int32(1)))
 v=cp.asnumpy(out);assert v[:11].tolist()==[16,32,48,64,112,80,96,176,112,128,144]
 assert v[11:13].tolist()==[5,6];assert cp.asnumpy(ix).tolist()==[2112,2112,2112]
 assert v[13]==np.float32(24)*np.float32(1e-5)

@pytest.mark.gpu
def test_tornado_duration_reads_neighbor_history_across_periodic_boundary():
 import cupy as cp
 from woof.core.kernels import get_kernel
 ny,nx=3,4;n=ny*nx
 uh=cp.full(n,75,dtype='f4');ll=cp.full(n,12,dtype='f4');lfc=cp.full(n,1000,dtype='f4');rh=cp.full(n,60,dtype='f4')
 pm=cp.zeros(n,dtype='f4');pd=cp.zeros(n,dtype='f4');pm[3]=1;pd[3]=888
 mask=cp.zeros(n,dtype='f4');dur=cp.zeros(n,dtype='f4');maximum=cp.zeros(n,dtype='f4')
 get_kernel('afwa_severe','afwa_severe_tornado')((1,),(128,),(uh,ll,lfc,rh,rh,pm,pd,mask,dur,maximum,np.float32(12),*(np.int32(x) for x in (ny,nx,0,0,0,0,1,0))))
 assert float(dur[0])==900 and float(maximum[0])==50
 assert float(dur[1])==12


def test_native_mp9_restart_does_not_require_undefined_unused_radii():
 from woof.ingest.wrfrst import unused_milbrandt_restart_radius as unused
 assert unused(_cfg(mp_physics=9,ra_physics=4),'RE_CLOUD',{})
 assert not unused(_cfg(mp_physics=8,ra_physics=4),'RE_CLOUD',{})
 assert not unused(_cfg(mp_physics=9,ra_physics=4),'RE_CLOUD',{'RE_CLOUD':0})
 assert not unused(_cfg(mp_physics=9,ra_physics=100),'RE_CLOUD',{})


def test_streaming_reduction_windows_cover_each_owned_mass_point_once():
 from types import SimpleNamespace
 from tilestream.spec import plan_tiles
 from woof.core.misc_diag_print import bind_window
 counts=np.zeros((150,150),dtype='i4')
 for tile in plan_tiles(150,150,75,75,18,periodic=False):
  state=SimpleNamespace();bind_window(state,tile,150)
  x0,x1,y0,y1=state._misc_print_window;ox,oy,nx=state._misc_print_origin
  counts[oy+y0:oy+y1,ox+x0:ox+x1]+=1
 assert np.array_equal(counts,np.ones_like(counts))
