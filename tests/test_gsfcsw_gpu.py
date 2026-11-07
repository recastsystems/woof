"""Shared-card checks: exact column words use strict numerics; wiring uses either mode."""
from pathlib import Path
import numpy as np
import pytest
cp=pytest.importorskip('cupy')
try:
    if cp.cuda.runtime.getDeviceCount()<1: raise RuntimeError('no device')
except Exception:
    pytest.skip('no CUDA device',allow_module_level=True)
from woof.core.gsfcsw import packed_columns
DATA=np.load(Path(__file__).resolve().parents[1]/'tools/gsfcsw_wrf461_oracle/columns.npz')

@pytest.mark.parametrize('nlay',(8,48))
def test_all_wrf_column_words(nlay):
    for col in range(64):
        flags=DATA[f'flags_{nlay}'][col];day,solar,latitude=DATA[f'meta_{nlay}'][col]
        got=packed_columns(cp.asarray(DATA[f'in_{nlay}'][col:col+1]),nlay,warm_rain=bool(flags[0]),f_qi=bool(flags[1]),f_qndrop=bool(flags[2]),julday=int(day),solcon=float(solar),center_lat=float(latitude))
        np.testing.assert_array_equal(cp.asnumpy(got).view('u4'),DATA[f'out_{nlay}'][col:col+1].view('u4'))

@pytest.mark.parametrize('nlay',(8,48))
def test_column_stride_and_repeat(nlay):
    col=3;flags=DATA[f'flags_{nlay}'][col];day,solar,latitude=DATA[f'meta_{nlay}'][col]
    inputs=cp.asarray(np.repeat(DATA[f'in_{nlay}'][col:col+1],37,axis=0))
    args=dict(warm_rain=bool(flags[0]),f_qi=bool(flags[1]),f_qndrop=bool(flags[2]),julday=int(day),solcon=float(solar),center_lat=float(latitude))
    expected=np.repeat(DATA[f'out_{nlay}'][col:col+1],37,axis=0).view('u4')
    for _ in range(2):
        np.testing.assert_array_equal(cp.asnumpy(packed_columns(inputs,nlay,**args)).view('u4'),expected)

def test_stock_callable_consumes_native_droplet_number():
    from datetime import datetime
    from types import SimpleNamespace
    from woof.config import RunConfig
    from woof.core.gsfcsw import GSFCSWShortwaveRadiation
    nl=8;n=nl+1;row=DATA['in_8'][5]
    def volume(offset): return cp.broadcast_to(cp.asarray(row[offset:offset+nl])[:,None,None],(nl,2,2)).copy()
    t=volume(0);p=volume(n);ex=volume(3*n);v=volume(4*n);c=volume(5*n)
    p8=cp.broadcast_to(cp.asarray(row[2*n:3*n])[:,None,None],(n,2,2)).copy()
    zero=cp.zeros_like(t)
    atmosphere={'temperature':t,'pressure':p,'exner':ex,'qv':v,'p_interface':p8}
    state=SimpleNamespace(qc=c,qi=zero,qs=zero,nc=cp.full_like(t,1e8),elapsed_seconds=0.)
    fields={'albedo':cp.full((2,2),.15,'f4'),'glw':cp.full((2,2),300.,'f4')}
    cfg=RunConfig(nx=2,ny=2,nz=nl,dx=3000.,dy=3000.,ztop=4000.,dt=15.,run_seconds=90.,mp_physics=10,ra_lw_physics=0,ra_sw_physics=2,radt=1.)
    scheme=GSFCSWShortwaveRadiation(datetime(2024,6,21,18),cp.full((2,2),45.,'f4'),cp.full((2,2),-90.,'f4'),center_lat=45.)
    scheme(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)
    initial=fields['taucldc'].copy()
    state.nc[:]=1e9
    result=scheme(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)
    assert bool(cp.any(fields['taucldc']!=initial))
    assert bool(cp.all(cp.isfinite(result.rthratensw)))
    assert scheme.update_count==2

def test_initial_history_contains_zeroed_gsfc_carriers():
    from datetime import datetime
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.core.physics import initialize_physics
    cfg=RunConfig(nx=4,ny=4,nz=8,dx=3000.,dy=3000.,ztop=4000.,dt=15.,run_seconds=90.,ra_lw_physics=0,ra_sw_physics=2)
    state=DomainState(cfg)
    driver=initialize_physics(state,cfg,glw=300.,radiation_start_time=datetime(2024,6,21,18),radiation_latitude=cp.full((4,4),45.,'f4'),radiation_longitude=cp.full((4,4),-90.,'f4'))
    output=driver.output_fields()
    for name in ('GSW','RSWTOA','TAUCLDI','TAUCLDC'):
        assert name in output
        assert bool(cp.all(output[name]==0))


def test_gsfc_held_radiation_and_carriers_resume_bitwise(tmp_path):
    from test_restart import _physics_state, _assert_restart_equal
    from woof.core.dycore import run_steps
    from woof.io import restart
    def build():
        return _physics_state(cp,ra_physics=0,ra_lw_physics=0,ra_sw_physics=2)
    state,cfg,_=build()
    run_steps(state,cfg,40)
    reference=restart.write_restart(tmp_path/'reference.npz',state,cfg)
    split,cfg_split,_=build()
    run_steps(split,cfg_split,20)
    checkpoint=restart.write_restart(tmp_path/'split.npz',split,cfg_split)
    restored,cfg_restored,_=build()
    restart.restore_restart(checkpoint,restored,cfg_restored)
    run_steps(restored,cfg_restored,20)
    resumed=restart.write_restart(tmp_path/'resumed.npz',restored,cfg_restored)
    _assert_restart_equal(reference,resumed)
    assert restored.physics.call_counts['radiation']==7
    assert bool(cp.any(restored.physics.fields['gsw'] != 0))


@pytest.mark.parametrize('feedback',(0,1))
def test_gsfc_runs_in_both_nested_feedback_modes(monkeypatch,feedback):
    from tilestream import test_moving_nest as moving, test_nest_executor as executor
    from woof.core.model import execute_experiment
    from woof.core.nest import NestCoupler
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree
    monkeypatch.setattr(executor,'I_PARENT_START',21)
    monkeypatch.setattr(executor,'J_PARENT_START',21)
    monkeypatch.setattr(executor,'PARENT_DT',6.)
    parent_cfg=moving.parent_cfg(nx=48,ny=48,nz=16,dt=6.,
        ra_physics=0,ra_lw_physics=1,ra_sw_physics=2,radt_minutes=.1,
        sf_sfclay_physics=1,nwp_diagnostics=0)
    child_cfg=moving.child_cfg(parent_cfg,nx=24,ny=24)
    boundaries=moving.domain_boundaries(parent_cfg,seconds=60.)
    parent,geo=moving.build_parent(parent_cfg,boundaries=boundaries,warmup=0)
    experiment=executor.experiment(parent_cfg,child_cfg,steps=4)
    model=assemble_idealized_tree(experiment,parent,
        coupler_factory=lambda node:NestCoupler(node,feedback=feedback),
        domain_preparer=moving._preparer)
    if feedback:executor.initial_feedback(model)
    execution=execute_experiment(model,validate_state=True,pool_trim_per_period=False)
    assert execution.steps==16
    child=model.node(2)
    assert child.coupler.force_count>0
    assert (child.coupler.feedback_count>0)==bool(feedback)
    adapters=[]
    for node in (model.root,child):
        driver=node.state.physics
        assert driver.call_counts['radiation']>=4
        assert bool(cp.any(driver.fields['gsw']>0))
        for name in ('gsw','rswtoa','taucldc','taucldi'):
            assert bool(cp.all(cp.isfinite(driver.fields[name])))
        adapters.append(driver.radiation_callable.shortwave_adapter)
    assert adapters[0].gsfc_workspace.data.ptr!=adapters[1].gsfc_workspace.data.ptr


def test_gsfc_independent_members_match_serial_on_private_cuda_streams(tmp_path):
    from test_restart import _physics_state, _assert_restart_equal
    from woof.io import restart
    references=[]
    for member in range(2):
        state,cfg,driver=_physics_state(cp,ra_physics=0,ra_lw_physics=0,ra_sw_physics=2)
        state.qc*=cp.float32(1+member)
        driver.compute(state,cfg)
        references.append(restart.write_restart(tmp_path/f'serial-{member}.npz',state,cfg))
    members=[]
    for member in range(2):
        state,cfg,driver=_physics_state(cp,ra_physics=0,ra_lw_physics=0,ra_sw_physics=2)
        state.qc*=cp.float32(1+member)
        members.append((state,cfg,driver,cp.cuda.Stream(non_blocking=True)))
    ready=cp.cuda.Event();ready.record()
    pointers=[item[2].radiation_callable.shortwave_adapter.gsfc_workspace.data.ptr for item in members]
    assert len(set(pointers))==2
    for state,cfg,driver,stream in members:
        stream.wait_event(ready)
        with stream:driver.compute(state,cfg)
    for member,(state,cfg,driver,stream) in enumerate(members):
        stream.synchronize()
        actual=restart.write_restart(tmp_path/f'concurrent-{member}.npz',state,cfg)
        _assert_restart_equal(references[member],actual)
