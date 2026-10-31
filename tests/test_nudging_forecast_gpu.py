"""Independent checks of WRF masks and persistent cadence on the GPU."""
import os
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('WOOF_DOWNSCALE_GPU_TEST') != '1', reason='explicit GPU pool lease required')


@pytest.mark.parametrize('mode,expected', [('grid',[0.,0.,.1,1.]), ('spectral',[0.,0.,1.,1.])])
def test_wrf_pbl_weights_and_potential_temperature(mode,expected):
    import cupy as cp
    from woof.spectral_nudging.config import NudgingConfig
    from woof.spectral_nudging.forecast import attach
    nz,ny,nx = 4,6,8
    state = SimpleNamespace(u=cp.zeros((nz,ny,nx+1),dtype='f'),v=cp.zeros((nz,ny+1,nx),dtype='f'),thp=cp.zeros((nz,ny,nx),dtype='f'),thb=cp.full((nz,),300.,dtype='f'),p=cp.full((nz,ny,nx),80000.,dtype='f'),php=cp.zeros((nz+1,ny,nx),dtype='f'),phb=cp.arange(nz+1,dtype='f')*9810.,ht=cp.zeros((ny,nx),dtype='f'),qv=cp.zeros((nz,ny,nx),dtype='f'),elapsed_seconds=0.)
    cfg = SimpleNamespace(use_adaptive_time_step=False,specified=True,spec_zone=1,relax_zone=0,dt=10.)
    nc = NudgingConfig(enabled=True,mode=mode,convention='wrf',guv=0.,gph=0.,gt=.01,cutoff_x=0,cutoff_y=0)
    target = cp.full((nz,ny,nx),310.,dtype='f')
    plan = attach(state,cfg,nc,{'temperature':(target,target)},interval_start=0.,interval_end=60.,pbl_height=cp.full((ny,nx),1500.,dtype='f'))
    plan.enqueue()
    cp.cuda.get_current_stream().synchronize()
    plan.check_at_seam()
    np.testing.assert_allclose(state.thp.get()[:,2,2],expected,atol=3e-5)
    np.testing.assert_array_equal(state.thp.get()[:,:,0],0.)


@pytest.mark.parametrize('zero,expected', [(0,3.6),(1,2.8)])
def test_grid_cadence_holds_or_zeros(zero,expected):
    import cupy as cp
    from woof.spectral_nudging.config import NudgingConfig
    from woof.spectral_nudging.gpu import FieldBinding,SpectralNudger
    shape = (2,6,8)
    value = cp.zeros(shape,dtype='f')
    target = cp.full(shape,10.,dtype='f')
    held = cp.zeros(shape,dtype='f')
    binding = FieldBinding(value,target,target,cp.zeros(shape,dtype='f'),cp.zeros(shape[-2:],dtype='f'),cp.ones(shape,dtype='f'),held)
    cfg = NudgingConfig(enabled=True,mode='grid',convention='wrf',guv=.01,fgdt=.5,fgdtzero=zero,boundary_cells=0)
    plan = SpectralNudger(cfg,{'u':binding},elapsed_seconds=0.,dt_seconds=10.,interval_start=0.,interval_end=60.)
    plan.enqueue()
    plan.enqueue()
    cp.cuda.get_current_stream().synchronize()
    # WRF's initial extra calculation at step 1 is held through step 2.
    np.testing.assert_allclose(value.get(),2.,atol=1e-6)
    # A restart between calculations must reuse this exact held tendency.
    saved = held.copy()
    fresh = FieldBinding(value.copy(),target,target,binding.height_agl_m,binding.pbl_height_m,binding.vertical_weight,saved)
    resumed = SpectralNudger(cfg,{'u':fresh},elapsed_seconds=20.,dt_seconds=10.,interval_start=0.,interval_end=60.)
    plan.enqueue()
    resumed.enqueue()
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_array_equal(fresh.values.get(),value.get())
    plan.enqueue();resumed.enqueue()
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_array_equal(fresh.values.get(),value.get())
    np.testing.assert_allclose(value.get(),expected,atol=1e-6)


@pytest.mark.parametrize('mode,ramp_minutes', [('grid',1.),('spectral',1.),('grid',-1.),('spectral',-1.),('grid',0.)])
def test_end_ramp_runs_past_last_analysis_and_restart(mode,ramp_minutes):
    """A positive ramp holds grid analyses, extrapolates spectral analyses."""
    import cupy as cp
    from woof.spectral_nudging.config import NudgingConfig
    from woof.spectral_nudging.gpu import FieldBinding, SpectralNudger
    shape=(2,6,8)
    target0=cp.full(shape,10.,dtype='f');target1=cp.full(shape,16.,dtype='f')
    value=cp.zeros(shape,dtype='f')
    binding=FieldBinding(value,target0,target1,cp.zeros(shape,dtype='f'),
        cp.zeros(shape[-2:],dtype='f'),cp.ones(shape,dtype='f'))
    cfg=NudgingConfig(enabled=True,mode=mode,convention='wrf',guv=.01,
        cutoff_x=0,cutoff_y=0,boundary_cells=0,gfdda_end_h=1/60,
        if_ramping=int(ramp_minutes!=0),dtramp_min=ramp_minutes)
    plan=SpectralNudger(cfg,{'u':binding},elapsed_seconds=0.,dt_seconds=10.,interval_start=0.,interval_end=60.)
    expected=0.
    for t in range(0,140,10):
        stop=60+max(0,ramp_minutes*60)
        weight=min(1,max(0,(stop-t)/abs(ramp_minutes*60))) if ramp_minutes else float(t<=60)
        target=16. if mode=='grid' and t>=60 and ramp_minutes>0 else 10.+t/10.
        expected+=.1*weight*(target-expected)
        plan.enqueue()
        cp.cuda.get_current_stream().synchronize()
        plan.check_at_seam()
        np.testing.assert_allclose(value.get(),expected,atol=5e-6,rtol=2e-6)
        if t==80:
            # Rebuild beyond the final analysis using the same endpoint pair.
            plan=SpectralNudger(cfg,{'u':binding},elapsed_seconds=90.,dt_seconds=10.,interval_start=0.,interval_end=60.)


def test_checkpoint_carries_held_nudging_tendencies(tmp_path,monkeypatch):
    from tests.test_restart import _cfg,_shim_state,_fill_setup,_fill_serialized
    from woof.io import restart
    cfg=_cfg(moist=True)
    state=_shim_state(cfg,monkeypatch)
    _fill_setup(state);_fill_serialized(state,seed=7)
    # The CPU checkpoint fixture is deliberate: the serialized field inventory
    # must carry cadence values independently of the CUDA graph object.
    for name,array in [('u',state.u),('v',state.v),('temperature',state.thp),('geopotential',state.php),('qv',state.qv)]:
        setattr(state,'nudging_tendency_'+name,np.full_like(array,.125))
    path=restart.write_restart(tmp_path/'nudged.npz',state,cfg)
    restored=_shim_state(cfg,monkeypatch)
    _fill_setup(restored)
    for name in ('u','v','temperature','geopotential','qv'):
        setattr(restored,'nudging_tendency_'+name,np.zeros_like(getattr(state,'nudging_tendency_'+name)))
    restart.restore_restart(path,restored,cfg)
    for name in ('u','v','temperature','geopotential','qv'):
        np.testing.assert_array_equal(getattr(restored,'nudging_tendency_'+name),getattr(state,'nudging_tendency_'+name))
