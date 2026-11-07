"""Execute selected Noah-MP on both live domains across a tree checkpoint."""
from dataclasses import replace
from datetime import timedelta
from types import MappingProxyType, SimpleNamespace
import pytest
from conftest import requires_gpu


def _model(seconds, feedback, options):
    from test_feedback import _feedback_experiment
    from test_noahmp_runtime import _build
    from woof.core.clock import resolve_clock, build_schedule
    from woof.core.model import DomainNode, ExperimentState, ModelRuntimeStatus, FeedbackScratch
    from woof.core.nest import NestCoupler
    exp = _feedback_experiment(seconds, feedback)
    states, domains = [], []
    for original in exp.domains:
        child = original.parent_id != 0
        state, cfg, driver = _build(nx=18 if child else 14, ny=18 if child else 14,
                                    nz=10, **options, water_columns=1,
                                    dt=4. if child else 12., dx=1000. if child else 3000.,
                                    dy=1000. if child else 3000.,
                                    grid_id=original.grid_id, nested=child)
        cfg = replace(cfg, dt=4. if child else 12., dx=1000. if child else 3000.,
                      dy=1000. if child else 3000., grid_id=original.grid_id,
                      nested=child, specified=False, run_seconds=seconds,
                      output_interval_s=24., restart_interval_s=0.)
        domains.append(replace(original, run=cfg, time_step=None if child else 12,
                               history_interval_s=24.))
        state._nest_restart_classification = 'REBUILT'
        states.append(state)
    exp = replace(exp, domains=tuple(domains), restart_interval_s=0.)
    clock=resolve_clock(exp); clocks=clock.clocks()
    root=DomainNode(domains[0], SimpleNamespace(), states[0], clocks[1], None, [], None)
    child=DomainNode(domains[1], SimpleNamespace(), states[1], clocks[2], root, [], None)
    root.children.append(child); child.coupler=NestCoupler(child, feedback=feedback)
    if feedback:
        child.coupler.feedback_prepare(child, FeedbackScratch())
        child.coupler.feedback_commit(child)
        child.coupler.feedback_finalize(child)
    model=ExperimentState(root, MappingProxyType({1:root,2:child}),build_schedule(exp,clock),None,
                          'selected-noahmp-nested-lifecycle-v1')
    model._scratch_arena=None; model._dycore_state_workspace=None
    model._runtime_status=ModelRuntimeStatus();model._resumed=False
    model._resume_committed_history_grid_ids=frozenset();model._io_manager=None;model._last_checkpoint=None
    return exp, model


@requires_gpu
@pytest.mark.parametrize('feedback',[0,1])
@pytest.mark.parametrize('options',[dict(opt_btr=2,opt_rsf=2),
    dict(opt_alb=1,opt_btr=3,opt_rsf=3), dict(opt_rsf=4),
    *(dict(opt_snf=v) for v in (2,3,4,5)), *(dict(dveg=v) for v in (1,3)),
    dict(opt_inf=2), dict(opt_run=4), dict(opt_inf=2,opt_run=4), dict(opt_run=6), dict(opt_inf=2,opt_run=6), dict(opt_run=7), dict(opt_inf=2,opt_run=7), dict(opt_rad=2), dict(opt_rad=2,opt_alb=1), dict(opt_rad=1), dict(opt_rad=1,opt_alb=1), dict(opt_frz=2), dict(opt_frz=2,opt_inf=2), dict(opt_stc=2), dict(opt_tbot=1), dict(opt_stc=3), dict(opt_stc=2,opt_tbot=1), dict(opt_crs=2), dict(opt_crs=2,opt_stc=2), dict(opt_sfc=2), dict(opt_sfc=2,opt_crs=2,opt_stc=2), *(dict(dveg=v) for v in (7,8,9)), *(dict(opt_run=v) for v in (1,2)), *(dict(opt_run=v,opt_inf=2) for v in (1,2)), dict(opt_soil=2), dict(opt_soil=3), *(dict(dveg=v) for v in (2,5,6)), *(dict(opt_run=8,opt_infdv=v) for v in (1,2,3)), dict(soiltstep=24.0), dict(soiltstep=48.0), dict(soiltstep=12.0), dict(opt_tdrn=1), dict(opt_tdrn=2), dict(noahmp_output=3,noahmp_acc_dt=0.4), pytest.param(dict(opt_irr=1,opt_irrm=0),id="irrigation-1-0"),pytest.param(dict(opt_irr=2,opt_irrm=1),id="irrigation-2-1"),pytest.param(dict(opt_irr=3,opt_irrm=1),id="irrigation-3-1"),pytest.param(dict(opt_irr=1,opt_irrm=1),id="irrigation-1-1"),pytest.param(dict(opt_irr=1,opt_irrm=2),id="irrigation-1-2"),pytest.param(dict(opt_irr=1,opt_irrm=3),id="irrigation-1-3"),pytest.param(dict(opt_crop=1),id="crop-1"),pytest.param(dict(opt_gla=2),id="glacier-2"),pytest.param(dict(opt_run=5,wtddt=0.2),id="groundwater-5")])
def test_live_selected_nested_checkpoint_continuation(feedback,options,tmp_path):
    from woof.core.model import execute_experiment
    from woof.io.restart import write_tree_restart,restore_tree_restart
    from test_ensemble_nested_concurrent_gpu import _carried_words
    exp, full=_model(24.,feedback,options)
    execute_experiment(full,validate_state=False,pool_trim_per_period=False)
    _,first=_model(12.,feedback,options)
    execute_experiment(first,validate_state=False,pool_trim_per_period=False)
    path=write_tree_restart(tmp_path,first,exp.start_time+timedelta(seconds=12))
    _, resumed=_model(24.,feedback,options);restore_tree_restart(path,resumed)
    execute_experiment(resumed,validate_state=False,pool_trim_per_period=False)
    a,b=_carried_words(full),_carried_words(resumed)
    # Boundary tables are rebuilt after restore. Compare the complete serialized
    # atmosphere, driver and surface carrier inventory and physical clocks.
    for grid in a:
        assert a[grid]['arrays']==b[grid]['arrays'],grid
        assert a[grid]['clock']==b[grid]['clock'],grid
        assert a[grid]['driver_header']==b[grid]['driver_header'],grid
        assert full.node(grid).state.physics.call_counts['noah'] > 0
    assert full.node(2).coupler.feedback_count == (3 if feedback else 0)
