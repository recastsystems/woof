"""OSAS carriers and physics calls through both resident nest feedback modes."""
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize('feedback', (0, 1))
def test_osas_runs_on_both_nest_domains(feedback, monkeypatch):
    import cupy as cp
    from woof.core.model import execute_experiment
    from woof.io.history_layout import physics_history_fields
    from woof.io.restart import _driver_manifest
    from tilestream import test_moving_nest as moving
    from tilestream import test_nest_executor as executor

    monkeypatch.setattr(executor, 'I_PARENT_START', 11)
    monkeypatch.setattr(executor, 'J_PARENT_START', 11)
    monkeypatch.setattr(executor, 'PARENT_DT', 6.)
    cfg = moving.parent_cfg(nx=40, ny=40, nz=29, dt=6.,
                            run_seconds=36., cu_physics=95, cudt_minutes=.1)
    child_cfg = moving.child_cfg(cfg, nx=24, ny=24)
    boundaries = moving.domain_boundaries(cfg, seconds=21600.)
    state, geography = moving.build_parent(cfg, boundaries=boundaries)
    experiment = executor.experiment(cfg, child_cfg, steps=6)
    model = executor.assemble(experiment, state, geography.grid, feedback=feedback)
    if feedback:
        executor.initial_feedback(model)
    result = execute_experiment(model, validate_state=True, pool_trim_per_period=False)
    assert result.clocks[1].step_count == 6
    assert result.clocks[2].step_count == 18
    for node in model.walk_parent_first():
        driver = node.state.physics
        assert driver.cumulus_callable.scheme.cu_physics == 95
        assert driver.call_counts['cumulus'] >= 6
        manifest = _driver_manifest(driver)
        for key in ('cumulus/last_htop', 'cumulus/last_hbot'):
            assert manifest[key].shape == (node.cfg.run.ny, node.cfg.run.nx)
            assert bool(cp.isfinite(manifest[key]).all())
        history = physics_history_fields(driver)
        assert {'PRATEC','RAINCV','CUTOP','CUBOT'} <= history.keys()
    assert (model.node(2).coupler.feedback_count > 0) == bool(feedback)
