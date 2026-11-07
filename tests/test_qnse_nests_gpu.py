"""QNSE pair advances both real domains through the native nest scheduler."""
from dataclasses import replace
from types import MappingProxyType,SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark=[requires_gpu,pytest.mark.gpu]

@pytest.mark.parametrize('feedback',[0,1])
def test_qnse_domains_subcycle_with_both_feedback_modes(feedback):
    import cupy as cp
    from test_feedback import _feedback_experiment
    from test_physics_driver import _full_state
    from woof.core.clock import build_schedule,resolve_clock
    from woof.core.model import DomainNode,ExperimentState,ModelRuntimeStatus,execute_experiment,FeedbackScratch
    from woof.core.nest import NestCoupler
    from woof.experiment import VerticalConfig
    exp=_feedback_experiment(6.,feedback=feedback)
    domains=[];states=[]
    for domain in exp.domains:
        root=domain.grid_id==1
        state,cfg,driver=_full_state(nx=domain.run.nx,ny=domain.run.ny,nz=24,
            dt=6. if root else 2.,dx=3000. if root else 1000.,dy=3000. if root else 1000.,
            run_seconds=6.,mp_physics=6,bl_pbl_physics=4,sf_sfclay_physics=4,
            grid_id=domain.grid_id,specified=root,nested=not root,
            spec_bdy_width=5,spec_zone=1,relax_zone=4,radt=0.,bldt=0.)
        if not root:
            state._nest_restart_classification='REBUILT'
        domains.append(replace(domain,run=cfg,time_step=6 if root else None,history_interval_s=6.))
        states.append(state)
    exp=replace(exp,domains=tuple(domains),restart_interval_s=0.,
        vertical=VerticalConfig(tuple(np.linspace(1.,0.,25)),10000.,1,.2))
    clock=resolve_clock(exp,lbc_interval_s=6.);clocks=clock.clocks()
    root=DomainNode(cfg=domains[0],grid=SimpleNamespace(),state=states[0],clock=clocks[1],parent=None,children=[],coupler=None)
    child=DomainNode(cfg=domains[1],grid=SimpleNamespace(),state=states[1],clock=clocks[2],parent=root,children=[],coupler=None)
    root.children.append(child);child.coupler=NestCoupler(child,feedback=feedback)
    # Strict WRF force arithmetic requires a specified parent. Keep actual
    # forcing tables and their clock on that parent, as the real door does.
    from woof.ingest.lateral_bc import (attach_lateral_boundaries,
        bind_lateral_boundary_clock,build_lateral_boundaries,domain_boundary_snapshot)
    first=domain_boundary_snapshot(root.state)
    later={name:value.copy() for name,value in first.items()}
    later['qv']*=np.float32(1.001)
    boundaries=build_lateral_boundaries([first,later],[0.,6.],
        spec_bdy_width=root.cfg.run.spec_bdy_width,
        spec_zone=root.cfg.run.spec_zone,relax_zone=root.cfg.run.relax_zone)
    attach_lateral_boundaries(root.state,boundaries)
    bind_lateral_boundary_clock(root.state,root.clock)
    if feedback:
        child.coupler.feedback_prepare(child,FeedbackScratch())
        child.coupler.feedback_commit(child);child.coupler.feedback_finalize(child)
    model=ExperimentState(root=root,nodes_by_grid_id=MappingProxyType({1:root,2:child}),schedule=build_schedule(exp,clock),memory_ledger=None,experiment_fingerprint='qnse-pair')
    model._scratch_arena=None;model._dycore_state_workspace=None
    model._runtime_status=ModelRuntimeStatus();model._resumed=False
    model._resume_committed_history_grid_ids=frozenset();model._io_manager=None;model._last_checkpoint=None
    execute_experiment(model,validate_state=False,pool_trim_per_period=False)
    for node,calls in ((root,1),(child,3)):
        driver=node.state.physics
        assert driver.call_counts['ysu']==calls
        assert driver.scheme_dispatch['bl_pbl_physics']=='_run_qnse_pbl'
        for name in ('tke_myj','el_myj','rc_mf','massflux_edkf','detr_edkf','rv_up','t2'):
            assert bool(cp.isfinite(driver.fields[name]).all()),(feedback,node.cfg.grid_id,name)
