"""BMJ executes on both sides of the real nested-domain scheduler."""
from dataclasses import replace

import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize('feedback', [0, 1])
def test_bmj_parent_child_carry_is_local_with_both_feedback_modes(monkeypatch, feedback):
    import cupy as cp
    from woof.core.model import execute_experiment
    from tilestream import test_moving_nest as moving, test_nest_executor as executor

    monkeypatch.setattr(executor, 'I_PARENT_START', 21)
    monkeypatch.setattr(executor, 'J_PARENT_START', 21)
    pcfg = moving.parent_cfg(nx=64, ny=64, nz=24, dt=6., cu_physics=2,
                             cudt_minutes=0., mp_physics=6)
    ccfg = moving.child_cfg(pcfg, nx=24, ny=24)
    boundaries = moving.domain_boundaries(pcfg, seconds=60.)
    parent, geo = moving.build_parent(pcfg, boundaries=boundaries, warmup=0)
    exp = replace(executor.experiment(pcfg, ccfg, steps=2), run_seconds=12.)
    model = executor.assemble(exp, parent, geo.grid, feedback=feedback)
    if feedback:
        executor.initial_feedback(model)
    execution = execute_experiment(model, validate_state=True, pool_trim_per_period=False)
    assert execution.steps > 0
    assert model.node(2).coupler.force_count > 0
    assert (model.node(2).coupler.feedback_count > 0) == bool(feedback)
    parent_fields, child_fields = model.root.state.physics.fields, model.node(2).state.physics.fields
    for fields, cfg in ((parent_fields, pcfg), (child_fields, ccfg)):
        assert fields['cldefi'].shape == (cfg.ny, cfg.nx)
        assert fields['qcconv'].shape == (cfg.nz, cfg.ny, cfg.nx)
        assert bool(cp.isfinite(fields['cldefi']).all())
    assert parent_fields['cldefi'].data.ptr != child_fields['cldefi'].data.ptr
    for node in (model.root, model.node(2)):
        assert node.state.physics.call_counts['cumulus'] > 0
