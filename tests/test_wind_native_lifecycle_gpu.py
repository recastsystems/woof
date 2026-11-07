"""Live MYNN turbine state survives disk continuation and member isolation."""
from pathlib import Path
import shutil
import numpy as np
import pytest
from conftest import requires_gpu
from test_restart import _physics_state, _assert_restart_equal
from woof.io import restart


def initial(cp, tmp_path, scheme, **overrides):
    table=Path(__file__).resolve().parents[1]/'tools/wind_mav_wrf471_oracle/wind-turbine-1.tbl'
    shutil.copyfile(table,tmp_path/'wind-turbine-1.tbl')
    name='windturbines-xy.txt' if scheme=='mav' else 'windturbines-ij.txt'
    path=tmp_path/name
    path.write_text('0 0 1 1\n400 0 1 1\n800 0 1 1\n' if scheme=='mav' else '3 2 1\n3 2 1\n4 3 1\n')
    values=dict(nx=8,ny=6,nz=40,ztop=1200.,dt=.5,mp_physics=1,
        cu_physics=0,ra_physics=0,sf_sfclay_physics=5,bl_pbl_physics=5,
        bl_mynn_tkeadvect=True,wind_farm_scheme=scheme,wind_farm_table=str(path),
        windfarm_ij=1,windfarm_induction=int(scheme=='fitch'),windfarm_wake_model=7)
    values.update(overrides)
    if scheme=='fitch_legacy':
        values.update(windturbines_spec='ideal',td_turbgridid=values.get('grid_id',1),
            td_pwfx=3,td_pwfy=2,td_ewfx=2,td_ewfy=2,wind_farm_table='',windfarm_ij=0)
    return _physics_state(cp,**values)


@requires_gpu
@pytest.mark.parametrize('scheme',['mav','fitch','fitch_legacy'])
def test_live_turbine_disk_continuation_and_member_isolation(tmp_path,scheme):
    import cupy as cp
    from woof.core.dycore import run_steps
    full,cfg,source=initial(cp,tmp_path,scheme)
    run_steps(full,cfg,6)
    if scheme!='fitch_legacy':assert bool(cp.any(source.wind_farm.power>0))
    assert bool(cp.any(source.wind_farm.tke_increment>0))
    reference=restart.write_restart(tmp_path/'full.npz',full,cfg)
    split,_,middle=initial(cp,tmp_path,scheme);run_steps(split,cfg,3)
    mid=restart.write_restart(tmp_path/'mid.npz',split,cfg)
    resumed,_,target=initial(cp,tmp_path,scheme);restart.restore_restart(mid,resumed,cfg)
    for name,value in middle.wind_farm.history_fields().items():
        cp.testing.assert_array_equal(target.wind_farm.history_fields()[name],value,err_msg=name)
    run_steps(resumed,cfg,3)
    result=restart.write_restart(tmp_path/'resumed.npz',resumed,cfg)
    _assert_restart_equal(reference,result)
    # A second ensemble member has private mutable turbine workspaces.
    other,_,member=initial(cp,tmp_path,scheme)
    before={name:cp.asnumpy(a).tobytes() for name,a in member.wind_farm.history_fields().items()}
    run_steps(full,cfg,1)
    assert before=={name:cp.asnumpy(a).tobytes() for name,a in member.wind_farm.history_fields().items()}
    assert member.wind_farm.tke_increment.data.ptr!=source.wind_farm.tke_increment.data.ptr


@requires_gpu
@pytest.mark.parametrize('scheme',['mav','fitch','fitch_legacy'])
def test_original_ensemble_executor_keeps_native_turbine_members(tmp_path,scheme):
    import cupy as cp
    from woof.ensemble.physics_execution import MemberPhysicsBinding,MemberPhysicsExecutor
    from test_ensemble_physics_execution_gpu import _snapshot
    originals=[];bindings=[]
    for member in range(2):
        pair=[initial(cp,tmp_path,scheme) for _ in range(2)]
        for state,_,_ in pair:state.u.fill(6.+member)
        originals.append(pair[0]);state,cfg,driver=pair[1]
        bindings.append(MemberPhysicsBinding(state,cfg,driver,member))
    executor=MemberPhysicsExecutor(bindings)
    for _ in range(2):
        executor.compute()
        for member,(state,cfg,driver) in enumerate(originals):
            driver.compute(state,cfg)
            assert _snapshot(state,cfg,driver)==_snapshot(bindings[member].state,cfg,bindings[member].driver)
            for key,value in driver.wind_farm.history_fields().items():
                cp.testing.assert_array_equal(value,bindings[member].driver.wind_farm.history_fields()[key])
            state.elapsed_seconds+=cfg.dt;bindings[member].state.elapsed_seconds+=cfg.dt
    assert executor.receipt()['call_counts']['compute']==[2,2]


@requires_gpu
@pytest.mark.parametrize('parent_scheme,child_scheme',[('mav','mav'),('fitch','fitch'),('mav','fitch'),('fitch','mav'),('fitch_legacy','fitch_legacy'),('fitch_legacy','mav'),('fitch','fitch_legacy')])
@pytest.mark.parametrize('feedback',[0,1])
def test_native_turbines_run_through_both_nest_feedback_modes(tmp_path,parent_scheme,child_scheme,feedback):
    from datetime import datetime,timedelta
    from types import SimpleNamespace
    import cupy as cp
    from woof.experiment import DomainConfig,ExperimentConfig,VerticalConfig,ProjectionConfig
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree
    from woof.core.nest import NestCoupler
    from woof.core.model import execute_experiment
    from woof.core.nest_fields import nest_field_kinds
    parent,pcfg,_=initial(cp,tmp_path,parent_scheme,nx=32,ny=30,dt=.75)
    child,ccfg,_=initial(cp,tmp_path,child_scheme,grid_id=2,nx=24,ny=24,dx=2000./3,dy=2000./3,dt=.25)
    start=datetime(2000,1,24,12)
    pdc=DomainConfig(1,0,1,1,1,1,3.,pcfg,time_step=0,time_step_fract_num=3,time_step_fract_den=4)
    cdc=DomainConfig(2,1,10,10,3,3,3.,ccfg)
    exp=ExperimentConfig(name='wind-native-nests',start_time=start,run_seconds=3.,
        vertical=VerticalConfig((),0.,1,.2),projection=ProjectionConfig('lambert',39.,-87.,30.,60.,-90.),
        domains=(pdc,cdc),feedback=feedback,restart_interval_s=0.)
    model=assemble_idealized_tree(exp,parent,child_initializer=lambda dc,p:SimpleNamespace(state=child),
        domain_preparer=lambda s,*args:s.physics,coupler_factory=lambda node:NestCoupler(node,feedback=feedback))
    execute_experiment(model,experiment=exp)
    assert 'qke_adv' in nest_field_kinds(ccfg)
    assert model.node(1).clock.step_count==4 and model.node(2).clock.step_count==12
    for node in model.nodes_by_grid_id.values():
        if node.state.physics.wind_farm.scheme!='fitch_legacy':assert bool(cp.any(node.state.physics.wind_farm.power>0))
        assert bool(cp.any(node.state.physics.wind_farm.tke_increment>0))
        assert bool(cp.all(cp.isfinite(node.state.qke_adv)))
    checkpoint=restart.write_tree_restart(tmp_path/'tree',model,start+timedelta(seconds=3))
    members=restart.tree_restart_members(checkpoint)
    for gid,path in members.items():
        with np.load(path,allow_pickle=False) as data:
            legacy=model.node(gid).state.physics.wind_farm.scheme=='fitch_legacy'
            prefix='wind_legacy_' if legacy else 'wind_mav_'
            assert any(prefix+'tke_increment' in name for name in data.files)
            for name in (('tke_rate','tke_increment') if legacy else ('power','tke_increment','hub','turbine_power')):
                stored=data['fields/'+prefix+name]
                cp.testing.assert_array_equal(stored,getattr(model.node(gid).state.physics.wind_farm,name))

    # Exercise the actual whole-tree restore, including registered turbine
    # aliases and the prognostic MYNN carrier on both sides of the nest.
    saved = {}
    for gid, node in model.nodes_by_grid_id.items():
        farm = node.state.physics.wind_farm
        names=(('tke_increment','tke_rate') if farm.scheme=='fitch_legacy' else
               ('power', 'tke_increment', 'tke_rate', 'hub', 'turbine_power'))
        saved[gid] = {name: cp.asnumpy(getattr(farm, name)) for name in names}
        saved[gid]['qke_adv'] = cp.asnumpy(node.state.qke_adv)
        for name in saved[gid]:
            (node.state.qke_adv if name == 'qke_adv' else getattr(farm, name)).fill(-777.)
    info = restart.restore_tree_restart(checkpoint, model)
    assert info.already_complete
    for gid, node in model.nodes_by_grid_id.items():
        farm = node.state.physics.wind_farm
        for name, expected in saved[gid].items():
            actual = node.state.qke_adv if name == 'qke_adv' else getattr(farm, name)
            cp.testing.assert_array_equal(actual, expected, err_msg=f'{gid}/{name}')


@requires_gpu
def test_global_native_induction_switch_is_inert_on_mav(tmp_path):
    import cupy as cp
    from woof.core.dycore import run_steps
    off, cfg0, driver0 = initial(cp, tmp_path, 'mav', windfarm_induction=0)
    ignored, cfg1, driver1 = initial(cp, tmp_path, 'mav', windfarm_induction=1)
    run_steps(off, cfg0, 2)
    run_steps(ignored, cfg1, 2)
    cp.testing.assert_array_equal(off.qke_adv, ignored.qke_adv)
    for key, value in driver0.wind_farm.history_fields().items():
        cp.testing.assert_array_equal(value, driver1.wind_farm.history_fields()[key])
