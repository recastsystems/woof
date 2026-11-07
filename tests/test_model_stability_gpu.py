"""Actual dycore failures stop the shared executor before output/checkpoints."""
from datetime import datetime

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _model(adaptive):
    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.model import DomainNode, ExperimentState, ModelRuntimeStatus
    from woof.experiment import DomainConfig, ExperimentConfig, VerticalConfig
    from tilestream import harness

    cfg = harness.make_config(16, 12, 4, dt=1., grid_id=1, run_seconds=4.,
        use_adaptive_time_step=adaptive, starting_time_step=1, min_time_step=1,
        max_time_step=1, step_to_output_time=True)
    state = harness.make_state(cfg)
    domain = DomainConfig(grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1, history_interval_s=1.,
        run=cfg, time_step=1)
    exp = ExperimentConfig(name="stability-control", start_time=datetime(2026, 1, 1),
        run_seconds=4., vertical=VerticalConfig(eta_levels=(), p_top=0., hybrid_opt=0, etac=.2),
        projection=None, restart_interval_s=1., domains=(domain,))
    calendar = resolve_clock(exp, lbc_interval_s=1.)
    node = DomainNode(domain, None, state, calendar.clocks()[1], None, [], None)
    model = ExperimentState(node, {1: node}, build_schedule(exp, calendar), None, "a" * 64)
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = model._dycore_state_workspace = model._io_manager = None
    model._last_checkpoint = None
    return exp, model


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("fault", ["nan", "folded-layer"])
@pytest.mark.parametrize("health_debug", [False, True])
def test_actual_failed_step_never_reaches_history_or_checkpoint(adaptive, fault, health_debug):
    import cupy as cp
    from woof.core import dycore
    from woof.core.model import execute_experiment

    exp, model = _model(adaptive)
    count = 0
    history, checkpoints = [], []
    def faulted_step(state, cfg, **kwargs):
        nonlocal count
        dycore.step(state, cfg, **kwargs)
        count += 1
        if count == 4:
            if fault == "nan":
                state.u[1, 5, 6] = cp.nan
            else:
                # A finite state can still have an inverted physical layer.
                lower = state.phb[1] if state.phb.ndim == 1 else state.phb[1, 5, 6]
                upper = state.phb[2] if state.phb.ndim == 1 else state.phb[2, 5, 6]
                state.php[2, 5, 6] = lower + state.php[1, 5, 6] - upper - 1.
    message = "non-finite state" if fault == "nan" else "non-finite vertical Courant"
    with pytest.raises(RuntimeError, match=message):
        execute_experiment(model, steppers={1: faulted_step}, experiment=exp,
            history_handler=lambda tree, node, ticks: history.append(node.clock.elapsed_seconds),
            restart_handler=lambda tree, ticks: checkpoints.append(tree.root.clock.elapsed_seconds),
            pool_trim_per_period=False, health_debug=health_debug)
    assert count == 4
    # Immediate debug checks precede clock commit. The asynchronous path
    # may commit internal progress, but the mandatory publication drain
    # must refuse before the fault's history or checkpoint is visible.
    assert model.root.clock.elapsed_seconds == (3. if health_debug else 4.)
    assert history == [0., 1., 2., 3.]
    assert checkpoints == [1., 2., 3.]


@pytest.mark.parametrize("adaptive", [False, True])
def test_stability_observation_preserves_every_healthy_prognostic_bit(adaptive):
    import cupy as cp
    from woof.core.model import execute_experiment
    from tilestream import gather

    results = []
    for validate in (False, True):
        exp, model = _model(adaptive)
        report = execute_experiment(model, experiment=exp, validate_state=validate,
                                    pool_trim_per_period=False)
        assert report.steps == 4 and model.root.clock.elapsed_seconds == 4.
        results.append({key: cp.asnumpy(value) for key, value in gather.inventory(model.root.state).items()})
    assert results[0].keys() == results[1].keys()
    for name in results[0]:
        np.testing.assert_array_equal(results[0][name], results[1][name], err_msg=name)
