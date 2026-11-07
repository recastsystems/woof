"""Cover the actual CUDA transfer and the MYNN roughness overwrite defect."""
import pytest
import cupy as cp
from woof.ocean.interface import OceanGrid,Bathymetry,WaveOutputs
from woof.ocean.coupling import DeviceCoupler
pytestmark=pytest.mark.gpu

def setup():
    shape=(63,99)
    def full(x):return cp.full(shape,x,dtype=cp.float32)
    one,zero=full(1),full(0)
    g=OceanGrid(99,63,25000.,25000.,full(25),full(-82),zero,one)
    b=Bathymetry(full(-100),full(100),one,zero,'EGM2008','unit fixture')
    return g,b,full

def test_rotation_density_and_stress_remain_device_resident():
    g,b,f=setup();c=DeviceCoupler(g,b)
    first=c.handoff(u10=f(12),v10=f(5),psfc=f(100000),ustar=f(.6),temperature=f(300),qv=f(.01),time_seconds=0,grid_relative=True)
    ptr=first.psfc.data.ptr
    cp.testing.assert_allclose(first.u10,-5)
    cp.testing.assert_allclose(first.v10,12)
    rho=100000/(287.05*300*1.0061)
    cp.testing.assert_allclose(first.tau_x,-5*rho*.36/13,rtol=2e-6)
    cp.testing.assert_allclose(first.tau_y,12*rho*.36/13,rtol=2e-6)
    c.handoff(u10=f(20),v10=f(0),psfc=f(98000),ustar=f(.7),temperature=f(299),qv=f(.012),time_seconds=60)
    assert c.forcing.psfc.data.ptr==ptr
    cp.testing.assert_allclose(c.forcing.u10,20)
    cp.testing.assert_allclose(c.forcing.psfc,98000)

@pytest.mark.parametrize('stochastic',[0,1])
def test_mynn_consumes_wave_roughness_before_fluxes(stochastic):
    from woof.core.mynn_sfclay import MYNN_SURFACE_OUTPUTS,MynnSurfaceResult,launch_mynn_surface_layer
    g,b,f=setup()
    values=dict(u1=20,v1=2,t1=299,qv1=.015,p1=100000,rho1=1.16,dz1=20,u2=21,v2=2,dz2=40,
        psfc=100200,tsk=300,pblh=500,mavail=1,hfx=5,qfx=.00001,znt=.0001,qsfc=.02,ust=.7,xland=2,snowh=0)
    arrays={k:f(v) for k,v in values.items()}
    def launch(z0):
        result=MynnSurfaceResult(**{k:f(0) for k in MYNN_SURFACE_OUTPUTS})
        launch_mynn_surface_layer(arrays,f(0),f(.7),result,dx=25000,itimestep=3,
            ocean_z0=z0,spp_pbl=stochastic,pattern_spp_pbl=f(.1) if stochastic else None)
        return result
    base=launch(None);wave=launch(f(.004))
    assert bool(cp.all(cp.isfinite(wave.ust)))
    assert bool(cp.all(wave.ust>base.ust))
    cp.testing.assert_allclose(wave.znt,.004,rtol=1e-5)

def test_wave_age_roughness_uses_depth_and_preserves_land_and_ice():
    import numpy as np
    from woof.ocean.coupling import _CUDA
    g,b,f=setup();n=g.nx*g.ny
    hs,tp,ust,wet,ice,alpha,z0=f(4),f(9),f(.8),f(1),f(0),f(.0185),f(.001)
    wet[:,0]=0;ice[:,1]=1
    kernel=cp.RawKernel(_CUDA,'roughness')
    kernel(((n+255)//256,),(256,), (hs,tp,ust,b.depth,wet,ice,alpha,z0,np.int32(0),np.int32(n)))
    assert bool(cp.all(cp.isfinite(z0)))
    cp.testing.assert_allclose(z0[:,0],.001);cp.testing.assert_allclose(z0[:,1],.001)
    omega=2*np.pi/9.;k=omega*omega/9.80665
    for _ in range(12):
        th=np.tanh(k*100);k-=(9.80665*k*th-omega*omega)/(9.80665*(th+k*100*(1-th*th)))
    c=omega/k;wave_z=3.35*4*(.8/c)**3.4
    ch=min(.1,max(.005,wave_z*9.80665/.64))
    expected=min(.02,ch*.64/9.80665+.11*1.5e-5/.8)
    cp.testing.assert_allclose(z0[:,2:],expected,rtol=1e-5)


def test_myj_consumes_wave_roughness_inside_flux_iterations():
    from woof.core.myjsfc import launch_myj_sfclay
    from woof.core.physics_inventory import MYJ_SFCLAY_INOUT, MYJ_SFCLAY_OUTPUTS
    g,b,f=setup()
    columns={"dz":cp.full((3,*g.shape),40,dtype=cp.float32),
             "tke":cp.full((3,*g.shape),.1,dtype=cp.float32)}
    values=dict(u1=20,v1=2,t1=299,th1=299,qv1=.015,qc1=0,p1=100000,
                psfc=100200,tsk=300,xland=2,mavail=1,z0base=.0001,ht=0)
    surface={k:f(v) for k,v in values.items()}
    surface["xland"][:,0]=1
    def launch(z0):
        state={k:f(0) for k in MYJ_SFCLAY_INOUT}
        for k,v in dict(ust=.7,znt=.0001,thz0=300,qz0=.015,qsfc=.02,akhs=.01,akms=.01).items():state[k][...]=v
        output={k:f(0) for k in MYJ_SFCLAY_OUTPUTS}
        launch_myj_sfclay(columns,surface,state,output,itimestep=1,ocean_z0=z0)
        return state,output
    base,base_out=launch(None);wave,wave_out=launch(f(.004))
    cp.testing.assert_allclose(wave["znt"][:,1:],.004,rtol=1e-5)
    assert bool(cp.all(cp.isfinite(wave["ust"])))
    assert bool(cp.any(wave["ust"][:,1:]!=base["ust"][:,1:]))
    assert bool(cp.any(wave_out["hfx"][:,1:]!=base_out["hfx"][:,1:]))
    for k in (*MYJ_SFCLAY_INOUT,*MYJ_SFCLAY_OUTPUTS):
        a=wave[k] if k in wave else wave_out[k]
        z=base[k] if k in base else base_out[k]
        cp.testing.assert_array_equal(a[:,0],z[:,0])


@pytest.mark.parametrize('option',[1,91])
def test_mm5_consumes_incoming_wave_momentum_roughness(option):
    from woof.core.sfclay import sfclay
    g,b,f=setup()
    def launch(z0):
        return sfclay(f(20),f(2),f(299),f(.015),f(100000),f(40),f(100200),f(300),
                      f(z0),f(500),f(1),f(2),ust=f(.7),qsfc=f(.02),option=option,
                      isftcflx=2,dx=25000.)
    base=launch(.0001);wave=launch(.004)
    assert bool(cp.all(cp.isfinite(wave.ust)))
    assert bool(cp.all(wave.ust>base.ust))
    assert bool(cp.any(wave.hfx!=base.hfx))
