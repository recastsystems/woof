"""PX's two-layer state survives the production restart and history paths."""
from dataclasses import replace
import numpy as np
import pytest
from conftest import requires_gpu


def build_px(*, fractional_seaice=0):
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    cfg=RunConfig(nx=8,ny=6,nz=40,dx=3000.,dy=3000.,dt=12.,ztop=16000.,
                  run_seconds=0.,time_step_sound=4,moist=True,mp_physics=6,cu_physics=0,
                  bl_pbl_physics=7,sf_sfclay_physics=7,sf_surface_physics=7,
                  num_soil_layers=2,ra_physics=90,bldt=0.,fractional_seaice=fractional_seaice)
    coord=make_vertical_coord(cfg.nz,stretch=3.2)
    base=make_base_state(coord,lambda z:300.+.004*np.asarray(z),p_surf=cfg.p_surf,ztop=cfg.ztop)
    state=init_moist_balanced(cfg,coord,base,lambda z:.012*np.exp(-np.asarray(z)/2400.))
    state.u.fill(7.);state.v.fill(1.5)
    land=np.ones((cfg.ny,cfg.nx),np.float32);land[:,-2:]=0.
    tsk=np.where(land,301.,297.).astype(np.float32)
    landusef=np.zeros((24,*land.shape),np.float32)
    landusef[6]=land;landusef[15]=1-land
    soil=np.zeros((16,*land.shape),np.float32)
    soil[5]=land;soil[13]=1-land
    moisture=np.broadcast_to(np.where(land,.3,1.),(2,*land.shape)).copy()
    driver=initialize_physics(state,cfg,landmask=land,tsk=tsk,
        soil_temperature=np.stack((tsk,np.full_like(tsk,287.))),
        soil_moisture=moisture,liquid_moisture=moisture,
        ivgtyp=np.where(land,7,16),isltyp=np.where(land,6,14),
        vegfra=55.,tmn=287.,swdown=600.,glw=330.,pblh=500.,
        landusef=landusef,soilctop=soil,soilcbot=soil,
        radiation_start_time=__import__('datetime').datetime(2000,1,24,12),
        radiation_latitude=np.full_like(land,39.),radiation_longitude=np.full_like(land,-87.))
    return state,cfg,driver


@requires_gpu
@pytest.mark.gpu
def test_px_restart_continuation_and_history(tmp_path):
    import cupy as cp
    from woof.core.dycore import step
    from woof.io.restart import write_restart,restore_restart
    from woof.io.history_layout import live_state_history_fields
    from woof.core.pxlsm_runtime import PX_STATE_2D,PX_FRACTIONS
    a,cfg,pa=build_px()
    for _ in range(3):step(a,cfg)
    path=write_restart(tmp_path/'px.npz',a,cfg)
    b,bcfg,pb=build_px()
    restore_restart(path,b,bcfg)
    for state in (a,b):step(state,cfg)
    for name in (*PX_STATE_2D,*PX_FRACTIONS,'smois','tslb','sh2o','isltyp'):
        av,bv=cp.asnumpy(pa.fields[name]),cp.asnumpy(pb.fields[name])
        assert np.array_equal(av.view(np.uint32),bv.view(np.uint32)),name
        assert np.isfinite(av).all(),name
    for name in ('u','v','w','thp','qv','qc','qr'):
        assert cp.array_equal(getattr(a,name),getattr(b,name)).item(),name
    frame=live_state_history_fields(a)
    for name in ('LAI_PX','WWLT_PX','WFC_PX','WSAT_PX','CLAY_PX','CSAND_PX','FMSAND_PX','T2OBS','Q2OBS'):
        assert name in frame,name
    assert pa.fields['smois'].shape[0]==2
    assert pa.call_counts['noah'] >= 4
    assert cp.all(pa.fields['chklowq']==1).item()


def test_px_stock_soil_geometry_and_selector():
    from woof.config import LAND_SURFACE_SOIL_LAYERS,LAND_SURFACE_SCHEMES
    from woof.core.physics_inventory import PHYSICS_SLOT_DISPATCH
    assert 7 in LAND_SURFACE_SCHEMES
    assert LAND_SURFACE_SOIL_LAYERS[7]==(2,)
    assert PHYSICS_SLOT_DISPATCH['sf_surface_physics'][7]=='_run_pxlsm'


def test_px_soil_nudging_admits_native_analysis_stream():
    from woof.config import RunConfig, validate_run_config
    validate_run_config(RunConfig(nx=8,ny=6,nz=40,dx=3000.,dy=3000.,ztop=16000.,dt=12.,run_seconds=12.,sf_surface_physics=7, sf_sfclay_physics=7,
        bl_pbl_physics=7,num_soil_layers=2,pxlsm_soil_nudge=1))


@requires_gpu
@pytest.mark.gpu
def test_px_member_executor_matches_independent_drivers():
    from woof.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor
    from test_ensemble_physics_execution_gpu import _snapshot
    originals=[];bindings=[]
    for member in range(2):
        pair=[build_px() for _ in range(2)]
        for state,cfg,driver in pair:
            state.u.fill(6.+member)
            state.elapsed_seconds=float(member*cfg.dt)
        originals.append(pair[0])
        state,cfg,driver=pair[1]
        bindings.append(MemberPhysicsBinding(state,cfg,driver,member))
    executor=MemberPhysicsExecutor(bindings)
    for _ in range(2):
        executor.compute()
        for expected,actual in zip(originals,bindings):
            state,cfg,driver=expected
            driver.compute(state,cfg)
            assert _snapshot(state,cfg,driver)==_snapshot(actual.state,actual.cfg,actual.driver)
            state.elapsed_seconds+=cfg.dt;actual.state.elapsed_seconds+=cfg.dt
    assert bindings[0].driver.fields['smois'].data.ptr!=bindings[1].driver.fields['smois'].data.ptr


@requires_gpu
@pytest.mark.gpu
def test_px_fractional_ice_restart_is_bitwise(tmp_path):
    import cupy as cp
    from woof.core.dycore import step
    from woof.io.restart import write_restart,restore_restart
    from woof.core.pxlsm_runtime import PX_STATE_2D,PX_ICE_FIELDS
    a,cfg,pa=build_px(fractional_seaice=1)
    pa.fields['xice'][:,-2:]=cp.asarray([.08,.7],dtype=cp.float32)
    pa.fields['tsk'][:,-2:]=260.
    pa.fields['tsk_save'][:,-2:]=260.
    pa.fields['tslb'][:,:,-2:]=260.
    for _ in range(3):step(a,cfg)
    path=write_restart(tmp_path/'ice.npz',a,cfg)
    b,bcfg,pb=build_px(fractional_seaice=1)
    restore_restart(path,b,bcfg)
    for state in (a,b):step(state,cfg)
    for name in (*PX_STATE_2D,*PX_ICE_FIELDS,'smois','tslb','sh2o'):
        actual,expected=cp.asnumpy(pa.fields[name]),cp.asnumpy(pb.fields[name])
        assert np.isfinite(actual).all(),name
        assert actual.tobytes()==expected.tobytes(),name
    x=pa.fields['xice'];mask=(x>=.02)&(x<=1.)
    want=pa.fields['tsk_save']*x+(1.-x)*pa.fields['tsk_sea']
    assert cp.max(cp.abs(pa.fields['tsk'][mask]-want[mask])).item()<4.e-5


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize('feedback',[0,1])
@pytest.mark.parametrize('soil_nudge',[0,1])
def test_px_land_nest_feedback_and_tree_checkpoint(feedback,soil_nudge,monkeypatch,tmp_path):
    import cupy as cp
    from datetime import timedelta
    from woof.core.model import execute_experiment
    import woof.core.physics as physics
    from woof.io.restart import write_tree_restart,restore_tree_restart
    from woof.core.pxlsm_runtime import PX_STATE_2D,PX_FRACTIONS
    from tilestream import test_nest_executor as executor
    from tilestream.test_moving_nest import build_parent,child_cfg,domain_boundaries,parent_cfg,START_TIME
    original=physics.initialize_physics
    def fixture_fractions(state,cfg,**kwargs):
        shape=(cfg.ny,cfg.nx)
        land=kwargs.get('landmask',np.ones(shape,np.float32))
        land=cp.asarray(land,dtype=cp.float32)
        if land.ndim==0:land=cp.full(shape,land)
        lu=cp.zeros((24,*shape),cp.float32);lu[6]=land;lu[15]=1-land
        soil=cp.zeros((16,*shape),cp.float32);soil[5]=land;soil[13]=1-land
        kwargs.update(landusef=lu,soilctop=soil,soilcbot=soil)
        return original(state,cfg,**kwargs)
    monkeypatch.setattr(physics,'initialize_physics',fixture_fractions)
    monkeypatch.setattr(executor,'I_PARENT_START',41)
    monkeypatch.setattr(executor,'J_PARENT_START',41)
    monkeypatch.setattr(executor,'PARENT_DT',6.)
    cfg=parent_cfg(nx=96,ny=96,nz=24,dt=6.,mp_physics=6,bl_pbl_physics=7,
        sf_sfclay_physics=7,sf_surface_physics=7,num_soil_layers=2,
        cu_physics=0,ra_lw_physics=90,ra_sw_physics=90,pxlsm_soil_nudge=soil_nudge,sgfdda_interval_m=1)
    def assemble():
        boundaries=domain_boundaries(cfg)
        parent,geo=build_parent(cfg,boundaries=boundaries,warmup=0 if soil_nudge else 1)
        exp=replace(executor.experiment(cfg,child_cfg(cfg,nx=48,ny=48),steps=2),feedback=feedback)
        model=executor.assemble(exp,parent,geo.grid,feedback=feedback)
        if feedback:executor.initial_feedback(model)
        if soil_nudge:
            for node in model.walk_parent_first():
                attach_px_analysis(node.state,node.cfg.run,monkeypatch,tmp_path)
        return model
    model=assemble()
    result=execute_experiment(model,validate_state=True,pool_trim_per_period=False)
    assert result.steps>0
    assert (model.node(2).coupler.feedback_count>0)==bool(feedback)
    for node in model.walk_parent_first():
        assert node.state.physics.scheme_dispatch['sf_surface_physics']=='_run_pxlsm'
        assert node.state.physics.call_counts['noah']>=2
        assert node.state.physics.fields['smois'].shape[0]==2
        for name in (*PX_STATE_2D,*PX_FRACTIONS):
            assert cp.isfinite(node.state.physics.fields[name]).all().item(),name
    path=write_tree_restart(tmp_path/'tree',model,START_TIME+timedelta(seconds=12))
    resumed=assemble();restore_tree_restart(path,resumed)
    if soil_nudge:
        for node in resumed.walk_parent_first():
            attach_px_analysis(node.state,node.cfg.run,monkeypatch,tmp_path)
        for node in model.walk_parent_first():
            assert node.state._px_inputs_owner.clock.get()[0]==node.state.elapsed_seconds
        for node in resumed.walk_parent_first():
            assert node.state._px_inputs_owner.clock.get()[0]==node.state.elapsed_seconds
    for left,right in zip(model.walk_parent_first(),resumed.walk_parent_first()):
        for name in (*PX_STATE_2D,*PX_FRACTIONS,'smois','tslb','sh2o'):
            assert cp.array_equal(left.state.physics.fields[name],right.state.physics.fields[name]).item(),name



def attach_px_analysis(state,cfg,monkeypatch,tmp_path,*,run_seconds=180):
    """Two actual CUDA-selected records; constructed data tests the clock."""
    from datetime import datetime
    from woof.spectral_nudging.px_inputs import PxInputs
    import woof.netcdf_bridge as bridge
    import woof.spectral_nudging.surface_inputs as inputs
    class Dataset:
        def __init__(self,path):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def stored_words(self,names):
            return {'Times':np.array([list('2000-01-24_12:00:00'),list('2000-01-24_12:01:00'),list('2000-01-24_12:02:00')],dtype='S1')}
    def read(path,*,record,**kwargs):
        from types import SimpleNamespace
        fields={f'{v}2_NDG_{e}':np.full((cfg.ny,cfg.nx),val+record,np.float32)
                for v,val in [('T',301.),('Q',8.)] for e in ('OLD','NEW')}
        fields.update({f'SN_NDG_{e}':np.zeros((cfg.ny,cfg.nx),np.float32) for e in ('OLD','NEW')})
        return SimpleNamespace(fields=fields)
    monkeypatch.setattr(bridge,'Dataset',Dataset)
    monkeypatch.setattr(inputs,'read_native_surface',read)
    path=tmp_path/'px-forcing';path.write_bytes(b'constructed PX record clock fixture')
    return PxInputs(state,cfg,path,datetime(2000,1,24,12),run_seconds)


@requires_gpu
@pytest.mark.gpu
def test_px_analysis_restart_record_crossing_and_member_isolation(monkeypatch,tmp_path):
    import cupy as cp
    from woof.core.dycore import step
    from woof.io.restart import write_restart,restore_restart
    from woof.core.pxlsm_runtime import PX_STATE_2D
    a,cfg,pa=build_px();cfg=replace(cfg,pxlsm_soil_nudge=1,sgfdda_interval_m=1)
    live=attach_px_analysis(a,cfg,monkeypatch,tmp_path)
    for _ in range(4):step(a,cfg)
    path=write_restart(tmp_path/'px-forced.npz',a,cfg)
    b,_,pb=build_px();restore_restart(path,b,cfg)
    resumed=attach_px_analysis(b,cfg,monkeypatch,tmp_path)
    member,_,_=build_px();independent=attach_px_analysis(member,cfg,monkeypatch,tmp_path)
    for _ in range(6):
        step(a,cfg);step(b,cfg)
    for name in (*PX_STATE_2D,'smois','tslb','sh2o'):
        assert cp.array_equal(pa.fields[name].view(cp.uint32),pb.fields[name].view(cp.uint32)).item(),name
    for name in ('u','v','thp','qv'):
        assert cp.array_equal(getattr(a,name),getattr(b,name)).item(),name
    assert live.clock.get().tolist()==resumed.clock.get().tolist()==[120.,108.]
    assert independent.clock.get().tolist()==[0.,0.]
    assert live.records.data.ptr!=independent.records.data.ptr
    assert cp.all(pa.fields['t2_ndg_old']==302).item()
    assert cp.isfinite(pa.fields['t2obs']).all().item()
    assert cp.any(pa.fields['t2obs']!=0).item()
