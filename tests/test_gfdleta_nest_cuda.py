"""Both production feedback modes execute GFDL on parent and child grids."""
import gc
import pytest

from conftest import requires_gpu
from woof.gfdleta_contract import GFDLETA_FIELDS, GFDLETA_INTEGER_FIELDS


@requires_gpu
@pytest.mark.gpu
def test_gfdl_ratio_three_tree_executes_both_feedback_modes(monkeypatch):
    import cupy as cp
    from woof.core.model import execute_experiment
    from woof.ingest.lateral_bc import bind_lateral_boundary_clock
    from woof.state_digest import canonical_state_digest
    from tilestream import test_moving_nest as moving
    from tilestream import test_nest_executor as executor

    monkeypatch.setattr(executor, 'I_PARENT_START', 41)
    monkeypatch.setattr(executor, 'J_PARENT_START', 41)
    monkeypatch.setattr(executor, 'PARENT_DT', 6.)
    parents, feedback_calls = [], []
    for feedback in (0, 1):
        cfg = moving.parent_cfg(nx=96, ny=96, nz=24, dt=6., run_seconds=36.,
            mp_physics=6, cu_physics=0, ra_physics=0, ra_lw_physics=99,
            ra_sw_physics=99, radt=.5, radt_minutes=.5)
        child_cfg = moving.child_cfg(cfg, nx=48, ny=48)
        boundaries = moving.domain_boundaries(cfg, seconds=21600.)
        state, geography = moving.build_parent(cfg, boundaries=boundaries, warmup=0)
        experiment = executor.experiment(cfg, child_cfg, steps=6)
        model = executor.assemble(experiment, state, geography.grid, feedback=feedback)
        bind_lateral_boundary_clock(model.root.state, model.root.clock)
        if feedback:
            executor.initial_feedback(model)
        result = execute_experiment(model, validate_state=True, pool_trim_per_period=False)
        assert result.steps == 24 and result.forces == 6
        for node in model.walk_parent_first():
            driver = node.state.physics
            assert driver.call_counts['radiation'] == 2
            assert set(GFDLETA_FIELDS) <= driver.fields.keys()
            for name in GFDLETA_INTEGER_FIELDS:
                assert driver.fields[name].dtype == cp.int32
            assert bool(cp.all(cp.isfinite(driver.rthratenlw)))
            assert bool(cp.any(driver.rthratenlw != 0))
        parents.append(canonical_state_digest(model.root.state, model.root.clock)['sha256'])
        feedback_calls.append(model.node(2).coupler.feedback_count)
        del model, state, boundaries
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    assert parents[0] != parents[1], 'two-way feedback must change the parent'
    assert feedback_calls[1] > feedback_calls[0]


@requires_gpu
@pytest.mark.gpu
def test_gfdl_live_relocation_carries_cloud_memory_and_keeps_health_armed(monkeypatch):
    """Production relocation and continuation, with the seeded domain builder.

    The fixture supplies constant land statics and the established idealized
    physics builder. RealRelocationChildPreparer owns all continuation,
    carrier, donor-fill and scalar-clock handling without replacement.
    """
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    import cupy as cp
    from woof.core.model import execute_experiment
    from woof.core.relocation_runner import RelocationRunner
    from woof.core.nest_relocation import Placement, plan_relocation
    from woof.experiment import _build_relocation
    from woof.ingest.lateral_bc import bind_lateral_boundary_clock
    from woof.ingest.nest_init import parent_only_init
    from woof.runtime import RealRelocationChildPreparer
    from tilestream import test_moving_nest as moving
    from tilestream import test_nest_executor as executor

    monkeypatch.setattr(executor, 'I_PARENT_START', 41)
    monkeypatch.setattr(executor, 'J_PARENT_START', 41)
    monkeypatch.setattr(executor, 'PARENT_DT', 6.)
    cfg = moving.parent_cfg(nx=96, ny=96, nz=24, dt=6., run_seconds=36.,
        mp_physics=6, cu_physics=0, ra_physics=0, ra_lw_physics=99,
        ra_sw_physics=99, radt=.5, radt_minutes=.5)
    child_cfg = moving.child_cfg(cfg, nx=48, ny=48)
    boundaries = moving.domain_boundaries(cfg, seconds=21600.)
    state, geography = moving.build_parent(cfg, boundaries=boundaries, warmup=0)
    exp = executor.experiment(cfg, child_cfg, steps=6)
    relocation = _build_relocation({'relocation': {'enabled': True,
        'grid_id': 2, 'cadence_seconds': 6., 'max_move_parent_cells': 1,
        'move': [{'at_seconds': 18., 'di_parent_cells': 1}]}},
        'gfdl-lifecycle-test', exp.domains, exp.run_seconds)
    exp = replace(exp, relocation=relocation)
    model = executor.assemble(exp, state, geography.grid)
    bind_lateral_boundary_clock(model.root.state, model.root.clock)

    def statics():
        return {'LANDMASK': np.ones((48, 48), np.float32),
                'HGT_M': np.zeros((48, 48), np.float32)}

    model._prepared_by_grid_id = {2: SimpleNamespace(static_fields=statics())}
    checked = []

    class FixturePreparer(RealRelocationChildPreparer):
        def capture_outgoing(self, node):
            self.old_dc = node.cfg
            super().capture_outgoing(node)

        def _rebuild_driver(self, initialized, new_dc, parent, moved, **kwargs):
            moving._preparer(initialized.state, new_dc, initialized.grid,
                             moving.START_TIME)
            for name, value in moved.items():
                initialized.state.physics.fields[name][...] = value
            return 0.

        def __call__(self, initialized, new_dc, parent):
            expected = {name: self._captured['continuation']['fields/'+name].copy()
                        for name in GFDLETA_FIELDS}
            super().__call__(initialized, new_dc, parent)
            plan = plan_relocation(
                placement_from=Placement(grid_id=2,
                    i_parent_start=self.old_dc.i_parent_start,
                    j_parent_start=self.old_dc.j_parent_start),
                placement_to=Placement(grid_id=2,
                    i_parent_start=new_dc.i_parent_start,
                    j_parent_start=new_dc.j_parent_start, generation=1),
                parent_grid_ratio=3, child_nx=48, child_ny=48)
            (dj, sj), (di, si) = plan.window((48, 48))
            for name, old in expected.items():
                new = initialized.state.physics.fields[name]
                assert bool(cp.all(new[dj, di].view(cp.uint32) ==
                                   old[sj, si].view(cp.uint32)))
                assert bool(cp.all(new[:, -3:] == (25 if name == 'hbot' else 0)))
                assert new.dtype == old.dtype
            assert bool(cp.any(expected['ncfrst'] > 0))
            checked.append(self.last_receipt)

    preparer = FixturePreparer(exp=exp, data=None, model=model)

    def initializer(dc, parent, **kwargs):
        return replace(parent_only_init(dc, parent, **kwargs), static_fields=statics())

    runner = RelocationRunner.from_experiment(exp, schedule=model.schedule,
        on_child_built=preparer, staging='host')
    runner.initializer = initializer
    runner.static_provenance = "parent-interpolated fixture with constant land statics"
    result = execute_experiment(model, relocation_runner=runner,
        validate_state=True, pool_trim_per_period=False)
    assert result.steps == 24 and result.forces == 6
    assert len(checked) == 1 and model.node(2).cfg.i_parent_start == 42
    assert checked[0]['physics_continuation']['restored']
    assert checked[0]['radiation_carriers']['restored']
    assert model.node(2).state.physics.call_counts['radiation'] == 2
