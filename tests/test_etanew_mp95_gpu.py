"""Real CUDA oracle checks and actual engine restart continuation."""
from pathlib import Path
import hashlib
import os
import subprocess
import sys
import numpy as np
import pytest
pytestmark = pytest.mark.gpu
ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/'tests/data/oracles/etanew'
TOOLS=ROOT/'tools/etanew_wrf461_oracle'

def _run(out, strict, plant=None):
    env=dict(os.environ)
    env['GPUWM_WRF_EXACT']='1' if strict else '0'
    cmd=[sys.executable,str(TOOLS/'run_port.py'),str(FIX/'columns_inputs.csv'),str(out)]
    if plant: cmd.extend(['--plant',plant])
    subprocess.run(cmd,env=env,check=True,capture_output=True,text=True)

@pytest.mark.parametrize('strict',[False,True])
def test_all_ten_steps_match_full_wrf_hashes(tmp_path, strict):
    pytest.importorskip('cupy')
    _run(tmp_path,strict)
    entries={name:digest for digest,name in (row.split() for row in (FIX/'BUILD-SHA256SUMS').read_text().splitlines() if len(row.split())==2)}
    for name in ('levels.csv','surface.csv','init.csv'):
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==entries['O0/'+name],name

def test_planted_one_bit_kernel_constant_is_caught(tmp_path):
    pytest.importorskip('cupy')
    original=np.asarray(np.float32(273.15)).view(np.uint32)
    changed=np.asarray(np.float32(float.fromhex('0x1.112664p+8'))).view(np.uint32)
    assert int(original)^int(changed)==1
    _run(tmp_path,True,'#define ETA_T0C 273.15f=#define ETA_T0C 0x1.112664p+8f')
    result=subprocess.run([sys.executable,str(TOOLS/'compare_etanew.py'),str(FIX),str(tmp_path),'--subset'],capture_output=True,text=True)
    assert result.returncode==1 and 'RESULT NOT EQUAL' in result.stdout

def test_real_state_restart_and_continuation_are_byte_identical(tmp_path):
    cp=pytest.importorskip('cupy')
    from test_restart import _physics_state
    from woof.core import microphysics
    from woof.io import restart
    from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS
    overrides=dict(mp_physics=95,ra_physics=0,cu_physics=0,
                   sf_sfclay_physics=0,sf_surface_physics=0,bl_pbl_physics=0)
    source,cfg,_=_physics_state(cp,**overrides)
    source.qt[2:6]=cp.float32(6e-4)
    source.f_ice_phy[2:6]=cp.float32(.65)
    source.f_rain_phy[2:6]=cp.float32(.3)
    source.f_rimef_phy[2:6]=cp.float32(2.5)
    for _ in range(5): microphysics.apply(source,cfg,10.0)
    path=restart.write_restart(tmp_path/'mid.npz',source,cfg)
    fresh,_,_=_physics_state(cp,**overrides)
    restart.restore_restart(path,fresh,cfg)
    roundtrip=restart.write_restart(tmp_path/'roundtrip.npz',fresh,cfg)
    with np.load(path,allow_pickle=False) as before, np.load(roundtrip,allow_pickle=False) as after:
        assert set(before.files)==set(after.files)
        for name in before.files:
            if name==restart._HEADER_KEY:
                continue  # Existing writer stamps the wall-clock creation time.
            assert before[name].dtype==after[name].dtype and before[name].shape==after[name].shape
            assert before[name].tobytes()==after[name].tobytes(),name
    def equal():
        for name in STATE_SERIALIZED_ATTRS:
            a=getattr(source,name,None)
            if a is not None:
                assert cp.asnumpy(a).tobytes()==cp.asnumpy(getattr(fresh,name)).tobytes(),name
    equal()
    for _ in range(5):
        microphysics.apply(source,cfg,10.0)
        microphysics.apply(fresh,cfg,10.0)
    equal()
    restart.state_manifest(source)
    restart.state_manifest(fresh)


def test_full_dynamics_restart_continuation_is_byte_identical(tmp_path):
    cp=pytest.importorskip('cupy')
    from test_restart import _physics_state
    from woof.core import dycore
    from woof.io import restart
    from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS
    overrides=dict(mp_physics=95,ra_physics=0,cu_physics=0,
                   sf_sfclay_physics=0,sf_surface_physics=0,bl_pbl_physics=0)
    source,cfg,_=_physics_state(cp,**overrides)
    source.qt[2:6]=cp.float32(6e-4)
    source.f_ice_phy[2:6]=cp.float32(.65)
    source.f_rain_phy[2:6]=cp.float32(.3)
    source.f_rimef_phy[2:6]=cp.float32(2.5)
    for _ in range(3): dycore.step(source,cfg)
    path=restart.write_restart(tmp_path/'dynamics-mid.npz',source,cfg)
    fresh,_,_=_physics_state(cp,**overrides)
    restart.restore_restart(path,fresh,cfg)
    for _ in range(3):
        dycore.step(source,cfg)
        dycore.step(fresh,cfg)
    for name in STATE_SERIALIZED_ATTRS:
        a=getattr(source,name,None)
        if a is not None:
            assert cp.asnumpy(a).tobytes()==cp.asnumpy(getattr(fresh,name)).tobytes(),name

def test_real_tree_history_alarms_do_not_request_unavailable_radar():
    from datetime import datetime
    cp=pytest.importorskip('cupy')
    from test_restart import _physics_state
    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.model import DomainNode, ExperimentState, ModelRuntimeStatus, execute_experiment
    from woof.experiment import DomainConfig, ExperimentConfig, VerticalConfig
    state,cfg,driver=_physics_state(cp,nx=16,ny=12,mp_physics=95,ra_physics=0,cu_physics=0,
        sf_sfclay_physics=0,sf_surface_physics=0,bl_pbl_physics=0,run_seconds=30.)
    state.qt[2:6]=cp.float32(6e-4)
    domain=DomainConfig(grid_id=1,parent_id=0,i_parent_start=1,j_parent_start=1,
        parent_grid_ratio=1,parent_time_step_ratio=1,history_interval_s=10.,run=cfg,time_step=10)
    exp=ExperimentConfig(name='eta-history',start_time=datetime(1974,4,3,12),run_seconds=30.,
        vertical=VerticalConfig(eta_levels=(),p_top=0.,hybrid_opt=0,etac=.2),
        projection=None,restart_interval_s=0.,domains=(domain,))
    calendar=resolve_clock(exp,lbc_interval_s=30.)
    node=DomainNode(domain,None,state,calendar.clocks()[1],None,[],None)
    model=ExperimentState(node,{1:node},build_schedule(exp,calendar),None,'a'*64)
    model._runtime_status=ModelRuntimeStatus()
    model._resumed=False; model._resume_committed_history_grid_ids=frozenset()
    model._scratch_arena=model._dycore_state_workspace=model._io_manager=None
    model._last_checkpoint=None
    times=[]
    report=execute_experiment(model,experiment=exp,pool_trim_per_period=False,
        history_handler=lambda _tree,node,_ticks: times.append(node.clock.elapsed_seconds))
    assert report.steps==3 and times==[0.,10.,20.,30.]
    assert driver.refl_10cm is None
    assert bool(cp.all(cp.isfinite(state.qt)))
