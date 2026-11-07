"""Live GCE checkpoint, nest feedback, ensemble and analysis diagnostics."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")


@pytest.mark.parametrize("feedback", [0, 1])
def test_nested_forcing_interpolates_all_six_species(feedback):
    import cupy as cp
    from woof.core.nest import NestCoupler
    from woof.verify.npref import np_nest_force
    from test_nest_coupler import _nodes, _DeviceState

    class State(_DeviceState):
        def existing_scratch(self, slot):
            return self._scratch.get(slot)

    host_parent, host_child = _nodes()
    for node in (host_parent, host_child):
        node.cfg.run = replace(node.cfg.run, mp_physics=97, moist=True, moist_cq=True)
    parent = SimpleNamespace(cfg=host_parent.cfg, clock=host_parent.clock,
                             state=State(host_parent.state, cp))
    child = SimpleNamespace(cfg=host_child.cfg, clock=host_child.clock, parent=parent,
                            state=State(host_child.state, cp))
    coupler = NestCoupler(child, feedback=feedback)
    expected = np_nest_force(
        host_parent.state, host_child.state, coupler.registrations,
        field_names=SPECIES, parent_dt_fp32=parent.clock.spec.dt_fp32,
        parent_interval_ticks=parent.clock.spec.step_ticks,
        spec_zone=1, relax_zone=4, spec_bdy_width=5, dtype=np.float32)
    coupler.force(child)
    for name in SPECIES:
        for side in ("west", "east", "south", "north"):
            for actual, reference in zip(coupler._last_tables[name][side], expected[name][side]):
                np.testing.assert_array_equal(cp.asnumpy(actual), np.float32(reference),
                                              err_msg=f"{name}/{side}")


def test_restart_continuation_preserves_the_complete_live_state(tmp_path):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _physics_state, _assert_restart_equal

    def initial():
        state, cfg, driver = _physics_state(cp, mp_physics=97, nz=24, ztop=14000.)
        # The stock dry fixture evaporates its small cloud completely.
        # A moist warm layer keeps this checkpoint test nontrivial for qc.
        state.qv[2:6] = cp.float32(.014)
        state.qr[4:8] = cp.float32(3.e-4)
        state.qi[12:18] = cp.float32(4.e-4)
        state.qs[12:18] = cp.float32(2.e-4)
        state.qg[12:18] = cp.float32(1.e-4)
        return state, cfg, driver

    whole, cfg, _ = initial()
    run_steps(whole, cfg, 12)
    reference = restart.write_restart(tmp_path / "whole.npz", whole, cfg)
    first, cfg, _ = initial()
    run_steps(first, cfg, 5)
    for name in SPECIES:
        assert bool(cp.any(getattr(first, name) > 0.)), name
    mid = restart.write_restart(tmp_path / "mid.npz", first, cfg)
    resumed, cfg, _ = initial()
    info = restart.restore_restart(mid, resumed, cfg)
    assert info.elapsed_seconds == 50.
    run_steps(resumed, cfg, 7)
    final = restart.write_restart(tmp_path / "resumed.npz", resumed, cfg)
    _assert_restart_equal(reference, final)
    assert resumed.physics.microphysics_updates == 12
    assert bool(cp.any(resumed.h_diabatic != 0.))
    keys = set(restart.read_restart_header(final)["array_manifest"])
    assert {f"state/{s}" for s in SPECIES} <= keys
    assert {f"scratch/mp_{s}" for s in ("rainnc", "snownc", "graupelnc", "sr")} <= keys
    assert "scratch/mp_column_vt" not in keys


@pytest.mark.parametrize("feedback", [0, 1])
def test_nested_feedback_carries_all_six_species(feedback):
    import cupy as cp
    from woof.core.nest import NestCoupler
    from woof.core.model import FeedbackScratch
    from woof.core.state import DomainState
    from test_feedback import _domain, _seed_state

    configs = [_domain(1, 0, nx=14, ny=14), _domain(2, 1, nx=9, ny=9, ratio=3)]
    configs = [replace(dc, run=replace(dc.run, mp_physics=97, moist_cq=True)) for dc in configs]
    parent_state, child_state = [DomainState(dc.run) for dc in configs]
    _seed_state(parent_state, child=False)
    _seed_state(child_state, child=True)
    for i, name in enumerate(SPECIES, 1):
        getattr(parent_state, name).fill(cp.float32(i * .01))
        getattr(child_state, name).fill(cp.float32(i * .001))
    clock = SimpleNamespace(ticks=0)
    parent = SimpleNamespace(cfg=configs[0], state=parent_state, clock=clock)
    child = SimpleNamespace(cfg=configs[1], state=child_state, clock=clock, parent=parent)
    coupler = NestCoupler(child, feedback=feedback)
    before = {s: cp.asnumpy(getattr(parent_state, s)) for s in SPECIES}
    coupler.feedback_prepare(child, FeedbackScratch())
    coupler.feedback_commit(child)
    coupler.feedback_finalize(child)
    for i, name in enumerate(SPECIES, 1):
        actual = cp.asnumpy(getattr(parent_state, name))
        if feedback:
            assert actual[0, 4, 4] == pytest.approx(i * .001, abs=1.e-8), name
            outside = np.ones(actual.shape, bool)
            outside[:, 4, 4] = False
            np.testing.assert_array_equal(actual[outside], before[name][outside])
        else:
            np.testing.assert_array_equal(actual, before[name])


def test_member_executor_and_analysis_use_the_gce_device_operator():
    import cupy as cp
    from woof.core.state_reflectivity import diagnose
    from woof.core.microphysics import apply
    from woof.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor
    from test_restart import _physics_state

    states = [_physics_state(cp, mp_physics=97) for _ in range(4)]
    for index, (state, _, _) in enumerate(states):
        state.qr[2:6] = cp.float32(1.e-4 * (1 + index % 2))
    references = []
    for state, cfg, _ in states[:2]:
        apply(state, cfg, cfg.dt, refl_10cm_due=True)
        references.append({s: cp.asnumpy(getattr(state, s)).view(np.uint32) for s in SPECIES})
    executor = MemberPhysicsExecutor([
        MemberPhysicsBinding(state, cfg, driver, member)
        for member, (state, cfg, driver) in enumerate(states[2:])])
    executor.apply_microphysics(refl_10cm_due=True)
    assert executor.call_counts["microphysics"] == [1, 1]
    for expected, (state, cfg, _) in zip(references, states[2:]):
        for name in SPECIES:
            np.testing.assert_array_equal(cp.asnumpy(getattr(state, name)).view(np.uint32), expected[name])
        before = {s: cp.asnumpy(getattr(state, s)).view(np.uint32) for s in SPECIES}
        refl = diagnose(state, cfg)
        assert refl.shape == state.qv.shape
        assert bool(cp.all(cp.isfinite(refl)))
        assert float(refl.max()) > -35.
        for name in SPECIES:
            np.testing.assert_array_equal(cp.asnumpy(getattr(state, name)).view(np.uint32), before[name])


@pytest.mark.parametrize("feedback", [0, 1])
def test_complete_nested_physics_schedule(feedback):
    """Run the original atmospheric STEP through a 3:1 two-domain tree."""
    from datetime import datetime
    import cupy as cp
    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus, execute_experiment)
    from woof.core.nest import NestCoupler
    from woof.experiment import DomainConfig, ExperimentConfig, VerticalConfig
    from test_restart import _physics_state

    domains, states = [], []
    for gid, nx, spacing, dt in ((1, 24, 6000., 6.), (2, 25, 2000., 2.)):
        state, cfg, _ = _physics_state(cp, mp_physics=97, nx=nx, ny=nx,
            nz=24, ztop=14000., dx=spacing, dy=spacing, dt=dt,
            grid_id=gid, run_seconds=18., output_interval_s=18.,
            nested=gid == 2, specified=gid == 1)
        for name, value in (("qr", 3.e-4), ("qi", 4.e-4),
                            ("qs", 2.e-4), ("qg", 1.e-4)):
            getattr(state, name)[6:18] = cp.float32(value)
        domains.append(DomainConfig(grid_id=gid, parent_id=0 if gid == 1 else 1,
            i_parent_start=1 if gid == 1 else 8,
            j_parent_start=1 if gid == 1 else 8,
            parent_grid_ratio=1 if gid == 1 else 3,
            parent_time_step_ratio=1 if gid == 1 else 3,
            history_interval_s=18., time_step=6 if gid == 1 else None, run=cfg))
        states.append(state)
    exp = ExperimentConfig(name="gce-nested-physics", start_time=datetime(2023, 3, 31),
        run_seconds=18., vertical=VerticalConfig((), 0., 0, .2),
        projection=None, restart_interval_s=0., domains=tuple(domains), feedback=feedback)
    clock = resolve_clock(exp, lbc_interval_s=18.)
    clocks = clock.clocks()
    parent = DomainNode(domains[0], None, states[0], clocks[1], None, [], None)
    child = DomainNode(domains[1], None, states[1], clocks[2], parent, [], None)
    child.coupler = NestCoupler(child, feedback=feedback)
    parent.children.append(child)
    from woof.ingest.lateral_bc import (attach_lateral_boundaries,
        bind_lateral_boundary_clock, build_lateral_boundaries,
        domain_boundary_snapshot)
    first = domain_boundary_snapshot(parent.state)
    later = {name: value.copy() for name, value in first.items()}
    later["qv"] *= np.float32(1.001)
    root_run = parent.cfg.run
    boundaries = build_lateral_boundaries([first, later], [0., 18.],
        spec_bdy_width=root_run.spec_bdy_width,
        spec_zone=root_run.spec_zone, relax_zone=root_run.relax_zone)
    attach_lateral_boundaries(parent.state, boundaries)
    bind_lateral_boundary_clock(parent.state, parent.clock)
    model = ExperimentState(parent, {1: parent, 2: child}, build_schedule(exp, clock),
                            None, "gce-nested-physics")
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = model._dycore_state_workspace = model._io_manager = None
    model._last_checkpoint = None
    result = execute_experiment(model, experiment=exp, pool_trim_per_period=False)
    assert result.steps == 12  # root plus nine child substeps
    assert parent.clock.step_count == 3 and child.clock.step_count == 9
    assert child.coupler.force_count == 3
    assert child.coupler.feedback_count == (3 if feedback else 0)
    for node, count in ((parent, 3), (child, 9)):
        assert node.state.physics.microphysics_updates == count
        assert node.state.elapsed_seconds == 18.
        assert bool(cp.any(node.state.h_diabatic != 0.))
        for name in SPECIES:
            assert bool(cp.all(cp.isfinite(getattr(node.state, name)))), name
