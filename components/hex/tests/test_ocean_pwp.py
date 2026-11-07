"""PWP admission and actual device conservation checks."""
import os
from types import SimpleNamespace
import pytest
import numpy as np
from woof.hex.ocean_pwp import configuration, forecast_environment


@pytest.mark.parametrize('bad', [{'levels':1},{'levels':2.5},{'layer_m':0},
    {'relaxed_ri':.2},{'thermal_expansion':float('nan')},{'unknown':1}])
def test_reject_parameters(bad):
    with pytest.raises(ValueError):
        configuration(bad)


def test_environment_is_scoped(monkeypatch):
    monkeypatch.setenv('WOOF_HEX_OCEAN_MODE','prescribed')
    a=SimpleNamespace(ocean_mode='pwp',pwp_options='{"levels":20}',pwp_profile=None)
    with pytest.raises(RuntimeError):
        with forecast_environment(a):
            assert os.environ['WOOF_HEX_OCEAN_MODE']=='pwp'
            assert configuration()['levels']==20
            raise RuntimeError('exit')
    assert os.environ['WOOF_HEX_OCEAN_MODE']=='prescribed'
    assert 'WOOF_HEX_PWP_OPTIONS' not in os.environ


@pytest.mark.gpu
def test_device_conservation_mixing_and_restart():
    import cupy as cp
    from importlib.resources import files
    from woof.hex.ocean_pwp import KEYS
    p=configuration({})
    n,nz=40962,int(p['levels'])
    z=(np.arange(nz)+.5)*p['layer_m']
    initial=np.zeros((n,4,nz),np.float64)
    initial[:,0]=300-np.maximum(0,z-30)*.05
    initial[:,1]=35
    initial[:,2,:6]=.8
    initial[:,0,0]=297 # overturning in the surface layer
    params=cp.asarray([p[k] for k in KEYS],dtype=cp.float64)
    source=files('woof.hex').joinpath('kernels/ocean_pwp.cuh').read_text()+r'''
extern "C" __global__ void test_step(int n,double* x,const double* p,double* d){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 pwp_advance(x+((long long)i)*4*int(p[0]),p,300,0,0,0,200,-100,0,d+4*i);
}'''
    kernel=cp.RawKernel(source,'test_step',options=('--std=c++17',))
    x=cp.asarray(initial); d=cp.zeros((n,4),cp.float64)
    kernel(((n+127)//128,),(128,),(np.int32(n),x,params,d))
    first=x.copy(); saved=d.copy()
    for _ in range(2):
        kernel(((n+127)//128,),(128,),(np.int32(n),x,params,d))
    expected=x.copy()
    x[...] = first; d[...] = saved
    for _ in range(2):
        kernel(((n+127)//128,),(128,),(np.int32(n),x,params,d))
    np.testing.assert_array_equal(cp.asnumpy(x).view('u8'),cp.asnumpy(expected).view('u8'))
    got=cp.asnumpy(x)
    np.testing.assert_allclose(got[:,1:].sum(axis=2),initial[:,1:].sum(axis=2),atol=1e-10)
    expected_heat=3*300*(-100+200*(1-np.exp(-nz*p['layer_m']/p['shortwave_efold_m'])))
    change=(got[:,0].sum(axis=1)-initial[:,0].sum(axis=1))*p['density']*p['heat_capacity']*p['layer_m']
    np.testing.assert_allclose(change,expected_heat,atol=.01)
    assert np.max(cp.asnumpy(d)[:,2])==0
    assert np.max(np.abs(cp.asnumpy(d)[:,1]))<.01
    assert np.all(got[:,0,0]<300)


@pytest.mark.parametrize('mode,owner,enabled',[
    ('prescribed','published',False), ('prescribed','hex',True),
    ('pwp','published',True)])
def test_single_ice_owner_prevents_second_post_lsm_writer(monkeypatch,mode,owner,enabled):
    from woof.ocean import seaice
    from woof.hex.ocean_pwp import select_ice_owner
    monkeypatch.setenv('WOOF_HEX_OCEAN_MODE',mode)
    monkeypatch.setenv('WOOF_HEX_ICE_OWNER',owner)
    driver=SimpleNamespace(fields={k:object() for k in seaice.COLD_SEEDED})
    driver.fields['xice']=object()
    original=driver.fields.copy()
    select_ice_owner(driver)
    if not enabled:
        assert driver.fields==original
        return
    assert driver.fields=={'xice':original['xice']}
    def forbidden(*args,**kwargs):
        raise AssertionError('duplicate post-LSM ice kernel was launched')
    monkeypatch.setattr(seaice,'_launch',forbidden)
    seaice.capture_precipitation(driver,{})
    seaice.step(driver,{},SimpleNamespace())


@pytest.mark.parametrize("coupled", [False, True])
def test_held_surface_cadence_retains_last_execution_census(monkeypatch, coupled):
    import sys
    from datetime import datetime
    from woof.hex import arwen_surface_update as update, arwen_seaice as ice
    monkeypatch.setitem(sys.modules,'cupy',np)
    monkeypatch.setenv('WOOF_HEX_OCEAN_MODE','prescribed')
    monkeypatch.setenv('WOOF_HEX_ICE_OWNER','published')
    monkeypatch.setattr(update,'physics_update_deepsoiltemp',lambda *args:None)
    monkeypatch.setattr(update,'physics_update_sstskin',lambda *args:None)
    fields={k:np.ones((1,1),np.float32) for k in (*ice._SEA,*ice._INOUT,'tsk','rainbl','swdown','albedo')}
    seen=[]
    class Driver:
        noahmp_params=SimpleNamespace()
        bldt_seconds=600
        _pending_rainbl=np.zeros((1,1),np.float32)
        def _run_sfclay(self,*args):pass
        def _run_noahmp(self,*args):
            # Mimic the real Noah-MP layout builder, including a threshold
            # crossing since the last input classification. Only this executed
            # census can be paired with the corresponding physics ledger.
            count = 2-len(seen)
            census = dict(land=3, water=4-count, sea_ice=count, glacier=1)
            self.noahmp_params._surface_layout_cache = (None, None, {"census":census})
            seen.append(census)

    driver=Driver();driver.fields=fields
    seam=SimpleNamespace(_driver=driver,_cfg=SimpleNamespace(sf_sfclay_physics=1))
    backend=SimpleNamespace(_seam=seam,_rollback_snapshot=False)
    service=update.ArwenSurfaceUpdate.__new__(update.ArwenSurfaceUpdate)
    service.backend=backend;service.start=datetime(2022,1,1);service.mesh={}
    service.classification={'sea_ice_columns':3};service.physics_classification=None
    service.counts={'skin':0,'deep':0};service.records=lambda now:None;service.pools=lambda:{}
    def evolve(*args):
        service.classification={'sea_ice_columns':0}
    service.waves=None
    service.ocean=SimpleNamespace(step=evolve) if coupled else None
    monkeypatch.setattr(ice,'_energy_balance',evolve)
    backend._surface_updates=service
    ice.install_noahmp_seaice(backend,seam)
    service.advance(0,300);driver._run_noahmp({},seam._cfg,1)
    service.advance(300,300) # dynamics advances, surface tendencies held
    assert service.physics_classification==dict(sflx_land_columns=3,open_water_columns=2,sea_ice_columns=2,glacier_columns=1)
    assert service.classification=={'sea_ice_columns':0}
    service.advance(600,300);driver._run_noahmp({},seam._cfg,3)
    assert service.physics_classification==dict(sflx_land_columns=3,open_water_columns=3,sea_ice_columns=1,glacier_columns=1)
    assert [row['sea_ice'] for row in seen]==[2,1]
