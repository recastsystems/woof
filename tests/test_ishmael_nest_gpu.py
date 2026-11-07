"""Real production nest executor, both feedback modes, native habit carriers."""
from dataclasses import replace
import numpy as np
import pytest
from conftest import requires_gpu
pytestmark = [pytest.mark.gpu, requires_gpu]

@pytest.mark.parametrize("feedback", [0,1])
def test_habit_nest_runs_full_executor(feedback, monkeypatch):
    import cupy as cp
    from tilestream import test_nest_executor as nest
    from woof.core.model import execute_experiment
    from woof.physics_registry import consumer_row_for_selector
    monkeypatch.setattr(nest, "I_PARENT_START", 33)
    monkeypatch.setattr(nest, "J_PARENT_START", 33)
    cfg=nest.parent_cfg(nx=80,ny=80,nz=24,mp_physics=55,cu_physics=0,dt=6,
                        run_seconds=12,output_interval_s=12)
    child=nest.child_cfg(cfg,nx=36,ny=36)
    state,geo=nest.build_parent(cfg,boundaries=nest.domain_boundaries(cfg,seconds=3600),warmup=0)
    for habit, names in enumerate((("qi", "ni", "qvoli", "qaoli"),
                                  ("qi2", "ni2", "qvoli2", "qaoli2"),
                                  ("qi3", "ni3", "qvoli3", "qaoli3")), 1):
        for name, value in zip(names, (2.3e-6, 1e5, 9e-10, 3e-10), strict=True):
            getattr(state, name)[cfg.nz // 2:] = np.float32(value * habit)
    exp=replace(nest.experiment(cfg,child,steps=2),run_seconds=12)
    model=nest.assemble(exp,state,geo.grid,feedback=feedback)
    if feedback: nest.initial_feedback(model)
    result=execute_experiment(model,validate_state=True,pool_trim_per_period=False)
    cp.cuda.runtime.deviceSynchronize()
    coupler=model.node(2).coupler
    assert result.steps == 8
    assert coupler.force_count > 0
    assert (coupler.feedback_count > 0) == bool(feedback)
    fields=consumer_row_for_selector("microphysics","native_column",55)["transport"]
    for node in (model.node(1), model.node(2)):
        for name in fields:
            value=getattr(node.state,name)
            assert value.shape==(node.cfg.run.nz,node.cfg.run.ny,node.cfg.run.nx)
            assert bool(cp.all(cp.isfinite(value)))
        for name in ("qi", "qi2", "qi3", "qvoli", "qvoli2", "qvoli3"):
            assert bool(cp.any(getattr(node.state,name)>0)), (feedback,node.cfg.grid_id,name)
    print({"feedback":feedback,"steps":result.steps,"forces":coupler.force_count,
           "feedback_count":coupler.feedback_count,"native_transport_fields":len(fields)})
