"""Air, wave and water momentum closes with currents and rotated grids."""
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu

@requires_gpu
@pytest.mark.gpu
def test_water_exchange_closes_vector_budget():
    import cupy as cp
    import woof.ocean.waves.solver as solver
    source=Path(solver.__file__).with_name('feedback.cu').read_text()
    module=cp.RawModule(code=source,options=('--std=c++17',))
    from woof.certify.kernel_manifest import record_module
    record_module('test:water-stress',source=source,options=('--std=c++17',),module=module)
    n=4
    # Includes cross swell, aligned waves, nonzero current and zero relative wind.
    east=np.array([30.,20.,-10.,4.],np.float32)
    north=np.array([5.,-3.,10.,2.],np.float32)
    cu=np.array([2.,-1.,3.,4.],np.float32);cv=np.array([1.,1.,-2.,2.],np.float32)
    ang=np.array([.4,-.6,1.1,0.],np.float32);ca=np.cos(ang);sa=np.sin(ang)
    ug=ca*east+sa*north;vg=-sa*east+ca*north
    pressure=np.full(n,100000,np.float32);temp=np.full(n,300,np.float32);q=np.full(n,.02,np.float32)
    rho=pressure/(287.05*temp*(1+.61*q));ust=np.array([1.,.7,.8,0.],np.float32)
    wave=np.zeros((10,n),np.float32);wave[0]=[.3,.2,-.1,.1];wave[1]=[.2,-.1,.3,-.05]
    water=np.zeros((3,n),np.float32);water[0]=[.12,.04,-.08,.03];water[1]=[.04,-.03,.1,-.01]
    def a(x):return cp.asarray(x,dtype=cp.float32)
    tx,ty,angle=[cp.zeros(n,cp.float32) for _ in range(3)];wet=cp.ones(n,cp.float32)
    module.get_function('vector_stress')((1,),(128,),tuple(map(a,(east,north,cu,cv,ust,rho,wave,water)))+(wet,tx,ty,angle,np.int32(n)))
    # Undo the original PBL drag and infer the full stress removed from air.
    du,dv=cp.zeros(n,cp.float32),cp.zeros(n,cp.float32);dz=np.full(n,50,np.float32)
    module.get_function('air_momentum')((1,),(128,),tuple(map(a,(ug,vg,cu,cv,ust,pressure,temp,q,dz,ca,sa,wave)))+(wet,du,dv,np.int32(n)))
    speed=np.hypot(ug,vg);oldx=rho*ust**2*ug/speed;oldy=rho*ust**2*vg/speed
    airx=oldx-cp.asnumpy(du)*rho*dz;airy=oldy-cp.asnumpy(dv)*rho*dz
    aire=ca*airx-sa*airy;airn=sa*airx+ca*airy
    np.testing.assert_allclose(aire-cp.asnumpy(tx),wave[0]-water[0],atol=3e-7)
    np.testing.assert_allclose(airn-cp.asnumpy(ty),wave[1]-water[1],atol=3e-7)
    # Arbitrary bed dissipation has no route into this water-side stress.
    wave[7]=1e4
    tx2,ty2=cp.zeros_like(tx),cp.zeros_like(ty)
    module.get_function('vector_stress')((1,),(128,),tuple(map(a,(east,north,cu,cv,ust,rho,wave,water)))+(wet,tx2,ty2,angle,np.int32(n)))
    assert bool(cp.array_equal(tx,tx2)) and bool(cp.array_equal(ty,ty2))
