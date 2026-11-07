"""CPU algebra/graph checks, explicitly NOT CUDA or forecast evidence."""
from dataclasses import replace
from pathlib import Path
import ctypes
import json
import shutil
import subprocess

import numpy as np
import pytest
from woof.hex import tc_climate as tc
from woof.hex.tc_climate.session import validate_schedule


def test_gpi_units_symmetry_missing():
    eta=np.array([1e-5,-1e-5,0,1e-5,1e-5])
    rh=np.array([50.,50,50,101,np.nan])
    got=tc.genesis_potential_index(np,eta,rh,np.full(5,70.),np.zeros(5))
    np.testing.assert_allclose(got[:3],[1,1,0])
    assert np.isnan(got[3:]).all()
    assert tc.genesis_potential_index(np,np.array([1e-5]),np.array([50]),
                                     np.array([70]),np.array([10]))[0] == .25
    with pytest.raises(ValueError):
        tc.genesis_potential_index(np,eta,rh,np.zeros(2),eta)


def test_recipe_and_schedule_refuse_missing_boundaries():
    r=tc.Recipe.load()
    assert validate_schedule(r,dict(steps=288,capture_steps=range(0,289,72)),300.)==5
    with pytest.raises(ValueError,match='every cadence'):
        validate_schedule(r,dict(steps=288,capture_steps=[0,288]),300.)
    with pytest.raises(ValueError,match='integral'):
        validate_schedule(r,dict(steps=288,capture_steps=[]),301.)
    for updates in ({'flood_capacity':3},{'cadence_hours':7},{'pi_wind_reduction':1.1},
                    {'candidate_capacity':True},{'max_abs_lat_deg':91}):
        with pytest.raises(ValueError):
            replace(r,**updates)


def test_qualification_duration_separate_thresholds():
    r=tc.Recipe.load()
    points=[dict(track_id=4,time_hours=i*6,wind10_ms=12,latitude_deg=20,terrain_m=0)
            for i in range(10)]
    assert len(tc.qualify_tracks(points,r))==1
    assert tc.qualify_tracks(points[:-1],r)==[]
    points[-1]['wind10_ms']=9
    assert tc.qualify_tracks(points,r)==[]


@pytest.fixture(scope='module')
def algebra(tmp_path_factory):
    """Compile production scalar CUDA expressions as host C++ for unit tests.

    The CUDA launch scheduler, atomics and memory behavior are not tested here.
    Every CUDA invocation in this harness is serialized and bounded to tiny data.
    """
    compiler=shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ needed for CPU algebra check')
    root=tmp_path_factory.mktemp('tc-algebra')
    source=Path(tc.__file__).parent
    cpp=root/'check.cpp'
    cpp.write_text('''#include <cmath>
#define __device__
#define __global__
struct Index { int x=0; };
Index blockDim{1}, blockIdx, threadIdx;
int atomicOr(int* p,int x) {int old=*p;*p|=x;return old;}
#include "'''+str(source/'kernels.cu')+'''"
#include "'''+str(source/'potential_intensity.cu')+'''"
extern "C" void pi_one(int nz,const float* p,const float* t,const float* r,
 float sst,float slp,int ocean,float* out,int* status) {
 blockIdx.x=threadIdx.x=0;
 potential_intensity(1,nz,p,t,r,&sst,&slp,&ocean,.9f,.8f,50.f,out,status);
}
extern "C" void minima_all(int n,int d,const int* g,const int* counts,
 const float* p,int* mask) {
 for(int i=0;i<n;++i) {blockIdx.x=i;minima(n,d,g,counts,p,mask);}
 blockIdx.x=0;
}
extern "C" int contour(int n,int d,const int* g,const int* counts,
 const float* xyz,const float* field,int seed,float delta,float cosine,
 int cap,int* queue,int* hash,int* errors) {
 return flood(n,d,g,counts,xyz,field,seed,seed,1,delta,cosine,cap,queue,hash,errors);
}
''')
    lib=root/'check.so'
    subprocess.run([compiler,'-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(lib)],
                   check=True,capture_output=True,text=True)
    dll=ctypes.CDLL(str(lib))
    f=ctypes.POINTER(ctypes.c_float);i=ctypes.POINTER(ctypes.c_int)
    dll.pi_one.argtypes=[ctypes.c_int,f,f,f,ctypes.c_float,ctypes.c_float,ctypes.c_int,f,i]
    dll.minima_all.argtypes=[ctypes.c_int,ctypes.c_int,i,i,f,i]
    dll.contour.argtypes=[ctypes.c_int,ctypes.c_int,i,i,f,f,ctypes.c_int,ctypes.c_float,
                          ctypes.c_float,ctypes.c_int,i,i,i]
    return dll


def pointer(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float if a.dtype==np.float32 else ctypes.c_int))


def test_pi_against_independent_reference_and_invalid_profiles(algebra):
    ref=json.loads(Path(tc.__file__).with_name('pi-reference.json').read_text())
    p,t,r=[np.array(ref[k],dtype=np.float32) for k in ('pressure_pa','temperature_k','vapor_kgkg')]
    def run(sst, ocean=1):
        pi=np.zeros(1,np.float32);status=np.zeros(1,np.int32)
        algebra.pi_one(len(p),pointer(p),pointer(t),pointer(r),sst,101000,ocean,pointer(pi),pointer(status))
        return pi[0],status[0]
    for row in ref['expected']:
        value,status=run(row['sst_k'])
        assert status==1
        assert value==pytest.approx(row['pi_ms'],abs=2e-4)
    for sst,ocean in ((302,0),(278,1),(float('nan'),1)):
        value,status=run(sst,ocean)
        assert status==0 and np.isnan(value)
    p[3]=p[2]
    value,status=run(302)
    assert status==0 and np.isnan(value)


def test_graph_closed_open_missing_overflow_and_plateau(algebra):
    # Ring graph in a small patch, entirely within the search radius.
    n=8
    graph=np.array([[(k-1)%n,(k+1)%n] for k in range(n)],np.int32)
    count=np.full(n,2,np.int32)
    angle=np.arange(n)*.002
    xyz=np.array(np.column_stack([np.cos(angle),np.sin(angle),np.zeros(n)]),np.float32)
    pressure=np.array([100000,100300,100300,100300,100300,100300,100300,100300],np.float32)
    mask=np.zeros(n,np.int32)
    algebra.minima_all(n,2,pointer(graph),pointer(count),pointer(pressure),pointer(mask))
    assert mask[0]==1 and mask[1]==0 and mask[7]==0
    def contour(cap=8,cosine=.99):
        queue=np.empty(cap,np.int32);hashed=np.empty(2*cap,np.int32);error=np.zeros(1,np.int32)
        ok=algebra.contour(n,2,pointer(graph),pointer(count),pointer(xyz),pointer(pressure),
                          0,200,cosine,cap,pointer(queue),pointer(hashed),pointer(error))
        return bool(ok),error[0]
    assert contour()==(True,0)
    pressure[1]=np.nan
    assert contour()==(False,0)
    pressure[:]=100000
    assert contour(cap=2)==(False,2)
    assert contour(cosine=.99999)==(False,0)
    pressure[:]=0
    algebra.minima_all(n,2,pointer(graph),pointer(count),pointer(pressure),pointer(mask))
    assert not mask.any()


def test_public_forecast_and_s2s_front_doors(tmp_path):
    from woof.hex.cli import build_parser
    from woof.hex import s2s_door
    path=Path(tc.__file__).with_name('recipes.json')
    parser=build_parser()
    args=parser.parse_args(['forecast','--tc-climate',str(path)])
    assert args.tc_climate==path
    assert parser.parse_args(['forecast']).tc_climate is None
    common=['s2s','--row','global-120km','--source','gfs','--cycle','2020-05-01T00',
            '--members','5','--days','214','--work-dir',str(tmp_path)]
    off=s2s_door.plan(parser.parse_args(common))
    assert 'tc_climate' not in off
    on=s2s_door.plan(parser.parse_args(common+['--tc-climate',str(path)]))
    assert len(on['members'])==5
    assert on['signature']!=off['signature']
    extra=s2s_door.science_arguments(on)
    assert extra[-2:]==['--tc-climate',str(path.resolve())]
    assert all(x['history_from_minutes']==0 for x in on['members'])
    with pytest.raises(s2s_door.S2SError,match='cadence|history-hours'):
        s2s_door.plan(parser.parse_args(common+['--tc-climate',str(path),'--history-hours','24']))


def test_diagnostic_admission_added_only_when_enabled():
    from woof.hex import forecast_door as door
    kw=dict(mesh='test',cells=100,free_bytes=100*2**30,total_bytes=100*2**30,
            headroom_bytes=0,registry={})
    base=door.admission_verdict(**kw)
    allowance=tc.workspace_allowance(100,55,5,tc.Recipe.load())
    on=door.admission_verdict(**kw,diagnostic_bytes=allowance)
    assert on.required_bytes-base.required_bytes==allowance
    assert 'diagnostic_planning_allowance_bytes' not in base.as_dict()
    assert on.as_dict()['diagnostic_planning_allowance_bytes']==allowance


@pytest.mark.gpu
def test_resident_tracker_contour_stitching_and_pi():
    """Run only inside a permitted measured-budget pool lease."""
    import cupy as cp
    r=replace(tc.Recipe.load(),candidate_capacity=8,flood_capacity=1024)
    side=41
    x,y=np.meshgrid(np.linspace(-10,10,side),np.linspace(-10,10,side))
    graph=np.full((side*side,4),-1,np.int32)
    for a in range(side):
        for b in range(side):
            graph[a*side+b]=[(a-1)*side+b if a else -1,
                            (a+1)*side+b if a<side-1 else -1,
                            a*side+b-1 if b else -1,
                            a*side+b+1 if b<side-1 else -1]
    count=np.full(side*side,4,np.int32)
    tracker=tc.DeviceTracker(graph,count,np.deg2rad(y.ravel()+20),np.deg2rad(x.ravel()),
                             np.zeros(side*side),r)
    pressure=cp.asarray((100000+25*(x*x+y*y)).ravel(),dtype=cp.float32)
    thickness=cp.asarray((10000-2*(x*x+y*y)).ravel(),dtype=cp.float32)
    wind=cp.full(side*side,20,cp.float32)
    before=pressure.copy()
    frames=[tracker.observe(h,pressure,thickness,wind) for h in range(0,55,6)]
    for frame in frames:
        ids=cp.asnumpy(frame['ids']);assert (ids>=0).sum()==1 and ids[ids>=0][0]==0
        assert cp.asnumpy(frame['errors'])[0]==0
    assert cp.array_equal(before,pressure)
    tracker.observe(60,pressure,thickness,wind*cp.nan)
    assert cp.asnumpy(tracker.observe(66,pressure,thickness,wind)['ids'])[0]==0
    module=cp.RawModule(code=Path(tc.__file__).with_name('potential_intensity.cu').read_text(),
                        options=('--std=c++17',))
    ref=json.loads(Path(tc.__file__).with_name('pi-reference.json').read_text())
    fields=[cp.asarray(ref[k],dtype=cp.float32) for k in ('pressure_pa','temperature_k','vapor_kgkg')]
    pi=cp.empty(1,cp.float32);status=cp.empty(1,cp.int32)
    module.get_function('potential_intensity')((1,),(1,),
        (1,len(fields[0]),*fields,cp.asarray([302],dtype=cp.float32),
         cp.asarray([101000],dtype=cp.float32),cp.ones(1,cp.int32),
         cp.float32(.9),cp.float32(.8),cp.float32(50),pi,status))
    assert int(cp.asnumpy(status)[0])==1
    assert float(cp.asnumpy(pi)[0])==pytest.approx(ref['expected'][1]['pi_ms'],abs=.003)
