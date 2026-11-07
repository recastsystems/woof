"""Live MP38 mass, number and volume through the real nested executor."""
from datetime import datetime, timedelta

import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def hail_tree(cp, feedback):
    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.model import DomainNode, ExperimentState, ModelRuntimeStatus
    from woof.core.nest import NestCoupler
    from woof.experiment import DomainConfig, ExperimentConfig, VerticalConfig
    from test_restart import _physics_state

    states, domains = [], []
    for grid, nx, ny, dx, dt in ((1, 39, 37, 6000., 6.), (2, 30, 27, 2000., 2.)):
        state, cfg, _ = _physics_state(cp, grid_id=grid, nx=nx, ny=ny, nz=24,
            dx=dx, dy=dx, dt=dt, run_seconds=12., ztop=14000.,
            mp_physics=38, sf_sfclay_physics=91, nested=grid == 2,
            specified=False, open_x=False, open_y=False)
        state.qi[12:18] = cp.float32(4.e-4)
        state.ni[12:18] = cp.float32(5.e4)
        state.qs[12:18] = cp.float32(2.e-4)
        state.qg[12:18] = cp.float32(grid * 2.e-4)
        state.ng[12:18] = cp.float32(grid * 1.e4)
        state.qvolg[12:18] = state.qg[12:18] / cp.float32(500.)
        states.append(state)
        domains.append(DomainConfig(grid_id=grid, parent_id=0 if grid == 1 else 1,
            i_parent_start=1 if grid == 1 else 10,
            j_parent_start=1 if grid == 1 else 10,
            parent_grid_ratio=1 if grid == 1 else 3,
            parent_time_step_ratio=1 if grid == 1 else 3,
            history_interval_s=12., run=cfg, time_step=6 if grid == 1 else None))
    exp = ExperimentConfig(name="hail-nest", start_time=datetime(2000, 1, 1),
        run_seconds=12., vertical=VerticalConfig((), 0., 0, .2),
        projection=None, restart_interval_s=6., feedback=feedback, domains=tuple(domains))
    calendar = resolve_clock(exp, lbc_interval_s=6.)
    root = DomainNode(domains[0], None, states[0], calendar.clocks()[1], None, [], None)
    child = DomainNode(domains[1], None, states[1], calendar.clocks()[2], root, [], None)
    root.children.append(child)
    child.coupler = NestCoupler(child, feedback=feedback)
    # Match the production constructor: nest boundary tables are rebuilt
    # from the parent after restore, rather than external forcing inputs.
    child.state._nest_restart_classification = "REBUILT"
    model = ExperimentState(root, {1: root, 2: child}, build_schedule(exp, calendar), None, "b" * 64)
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = model._dycore_state_workspace = model._io_manager = None
    model._last_checkpoint = None
    return exp, model


@pytest.mark.parametrize("feedback", [0, 1])
def test_hail_real_nested_steps_and_checkpoint_carry_live_moments(tmp_path, feedback):
    import cupy as cp
    from woof.core.model import execute_experiment
    from woof.io import restart
    from test_ensemble_nested_concurrent_gpu import _carried_words

    exp, model = hail_tree(cp, feedback)
    checkpoints = []
    def save(tree, ticks):
        checkpoints.append(restart.write_tree_restart(tmp_path, tree,
            exp.start_time + timedelta(seconds=tree.root.clock.elapsed_seconds)))
    execute_experiment(model, experiment=exp, restart_handler=save,
                       pool_trim_per_period=False)
    assert model.root.clock.elapsed_seconds == 12.
    assert model.node(2).clock.elapsed_seconds == 12.
    assert model.root.state.physics.microphysics_updates == 2
    assert model.node(2).state.physics.microphysics_updates == 6
    coupler = model.node(2).coupler
    assert coupler.force_count == 2
    assert coupler.feedback_count == (2 if feedback else 0)
    assert checkpoints
    for node in model.walk_parent_first():
        for field in ("qg", "ng", "qvolg"):
            array = getattr(node.state, field)
            assert bool(cp.all(cp.isfinite(array))), field
            assert bool(cp.any(array > 0)), field
    assert {"ng", "qvolg"} <= set(coupler._last_tables)
    for checkpoint in checkpoints:
        for path in restart._tree_restart_paths(checkpoint, {1, 2}).values():
            manifest = restart.read_restart_header(path)["array_manifest"]
            assert {"state/qg", "state/ng", "state/qvolg"} <= set(manifest)
    # Resume both feedback modes from the first complete domain set.
    reference = _carried_words(model)
    resumed_exp, resumed = hail_tree(cp, feedback)
    restart.restore_tree_restart(checkpoints[0], resumed)
    assert resumed.root.clock.elapsed_seconds == 6.
    execute_experiment(resumed, experiment=resumed_exp,
                       pool_trim_per_period=False)
    assert _carried_words(resumed) == reference
