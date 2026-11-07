"""Prove Nc survive real transport, stepping and restart."""
import dataclasses
import numpy as np
import pytest
from conftest import requires_gpu


@requires_gpu
def test_nc_advect_through_the_same_scalar_operator():
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import advance_scalars_stage, init_moist_balanced
    nx,ny,nz = 16,8,12
    cfg=RunConfig(nx=nx,ny=ny,nz=nz,dx=500.,dy=500.,ztop=6000.,dt=10.,
                  run_seconds=0.,moist=True,mp_physics=51)
    coord=make_vertical_coord(nz)
    base=make_base_state(coord,lambda z:300.+.003*np.asarray(z,float),
                         p_surf=cfg.p_surf,ztop=cfg.ztop)
    state=init_moist_balanced(cfg,coord,base,lambda z:np.full(nz,1.e-3))
    blob=np.zeros((nz,ny,nx),np.float32)
    blob[4:8,3:6,6:10]=1.e-4
    for name in ('qi','nc'):
        getattr(state,name)[...]=cp.asarray(blob)
        getattr(state,name+'0')[...]=getattr(state,name)
    chm=state.c1h[:,None,None]*state.total_mu()[None]+state.c2h[:,None,None]
    ru=cp.zeros((nz,ny,nx+1),cp.float32)
    ru[...]=.5*(cfg.dx/cfg.dt)*chm[:,:,:1]
    rv=cp.zeros((nz,ny+1,nx),cp.float32)
    ww=cp.zeros((nz+1,ny,nx),cp.float32)
    advance_scalars_stage(state,cfg,ru,rv,ww,cfg.dt,final=True)
    for name in ('qi','nc'):
        assert float(cp.abs(getattr(state,name)-cp.asarray(blob)).max())>0
        assert getattr(state,name).tobytes()==state.qi.tobytes()


@requires_gpu
def test_real_nc_steps_write_and_restore_nc(tmp_path):
    import cupy as cp
    from test_p3_port import _p3_bubble_setup,_p3_bubble_state
    from woof.core.dycore import run_steps,stability_report
    from woof.io import restart
    from woof.core.moist import moist_species
    from woof.config import validate_run_config
    cfg,coord,base,rime=_p3_bubble_setup()
    cfg=validate_run_config(dataclasses.replace(cfg,mp_physics=51))
    state=_p3_bubble_state(cfg,coord,base,rime,u_background=20.)
    # A cloudy lower layer keeps the restart check non-vacuous for Nc.
    state.qc[1:4,:,40:60]=cp.float32(1.e-3)
    state.qv[1:4,:,40:60]=cp.float32(.02)
    state.nc[...]=cp.where(state.qc>0,cp.float32(1.e8),cp.float32(0.))
    # The native-input history writer calls this before the first step.
    # MP51 must use P3's PSD and leave all prognostic fields untouched.
    from woof.core import state_reflectivity, p3_device
    held = {name: getattr(state, name).copy() for name in moist_species(state)}
    actual = state_reflectivity.diagnose(state, cfg).copy()
    assert actual.tobytes() == p3_device.reflectivity(state).tobytes()
    for name, before in held.items():
        assert getattr(state, name).tobytes() == before.tobytes(), name
    run_steps(state,cfg,2)
    assert not stability_report(state,cfg)['nan']
    for name in ('nc',):
        assert bool(cp.isfinite(getattr(state,name)).all())
        assert bool((getattr(state,name)>0).any())
    checkpoint=restart.write_restart(tmp_path/'state.npz',state,cfg)
    header=restart.read_restart_header(checkpoint)
    assert {'state/nc'}<=set(header['array_manifest'])
    assert 'predicted-nc' in header['physics_setup']['algorithms']['microphysics']
    fresh=_p3_bubble_state(cfg,coord,base,rime)
    restart.restore_restart(checkpoint,fresh,cfg)
    for name in moist_species(state)+('th_old','qv_old'):
        assert getattr(fresh,name).tobytes()==getattr(state,name).tobytes(),name
