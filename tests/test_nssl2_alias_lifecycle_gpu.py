"""Stock WRF 4.7.1 NSSL aliases retain their predicted CCN through CUDA lifecycle paths."""
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import tomllib
import numpy as np
import pytest
from conftest import requires_gpu


def _selectors(selector):
    from test_wrf_options_page import _tool
    from woof.namelist_import import import_namelists
    from woof.core.nssl2_contract import NSSL2_SELECTOR_FIELDS
    tool = _tool()
    with tempfile.TemporaryDirectory() as folder:
        wps, nml = tool._pair(Path(folder), {**tool.BASE, "mp_physics": selector})
        text, _ = import_namelists(wps, nml, rrtmg_variant="rrtmg_legacy")
    shared = tomllib.loads(text)["shared"]
    return {key: shared[key] for key in ("mp_physics", *NSSL2_SELECTOR_FIELDS) if key in shared}


def _state(selector, **overrides):
    import cupy as cp
    from test_physics_driver import _full_state
    state, cfg, driver = _full_state(**{**_selectors(selector), "cu_physics": 0,
        **overrides})
    # Force active cloud water and a nonuniform predicted CCN reservoir.
    state.qc[...] = cp.float32(1.e-4)
    state.qnn[...] = cp.arange(state.qnn.size, dtype=cp.float32).reshape(state.qnn.shape) + cp.float32(5.e8)
    return state, cfg, driver


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("selector", [17, 22])
def test_stock_alias_split_forecast_preserves_every_restart_word(selector, tmp_path):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal
    straight, cfg, _ = _state(selector)
    run_steps(straight, cfg, 8)
    reference = restart.write_restart(tmp_path / "continuous.npz", straight, cfg)
    split, cfg, driver = _state(selector)
    run_steps(split, cfg, 3)
    checkpoint = restart.write_restart(tmp_path / "middle.npz", split, cfg)
    fresh, cfg, rebuilt = _state(selector)
    restart.restore_restart(checkpoint, fresh, cfg)
    assert rebuilt.nssl2_binding is not driver.nssl2_binding
    assert bool(cp.any(fresh.qnn > 0.))
    run_steps(fresh, cfg, 5)
    resumed = restart.write_restart(tmp_path / "resumed.npz", fresh, cfg)
    _assert_restart_equal(reference, resumed)
    assert "state/qnn" in restart.read_restart_header(resumed)["array_manifest"]


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("selector", [17, 22])
def test_stock_alias_ensemble_owns_independent_predicted_ccn(selector):
    import cupy as cp
    from woof.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor
    one, cfg, first = _state(selector)
    two, _, second = _state(selector)
    two.qnn[...] *= cp.float32(.8)
    executor = MemberPhysicsExecutor((MemberPhysicsBinding(one, cfg, first, 1),
                                     MemberPhysicsBinding(two, cfg, second, 2)))
    executor.apply_microphysics()
    before = {name: getattr(one, name).copy() for name in first.nssl2_binding.mode.transported_fields}
    executor.apply_microphysics_member(1)
    for name, value in before.items():
        assert cp.array_equal(value.view(cp.uint32), getattr(one, name).view(cp.uint32)), name
    assert first.nssl2_binding is not second.nssl2_binding
    assert bool(cp.any(one.qnn != two.qnn))
    assert executor.call_counts["microphysics"] == [1, 2]

@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("selector", [17, 22])
@pytest.mark.parametrize("feedback", [0, 1])
def test_stock_alias_steps_both_domains_through_the_nest_executor(selector, feedback, tmp_path):
    from dataclasses import replace
    from woof.core.model import execute_experiment
    from woof.core.nest import NestCoupler
    from woof.experiment import (DomainConfig, ExperimentConfig,
                                  ProjectionConfig, VerticalConfig)
    from woof.verify.cases.nest_ideal_common import (
        assemble_idealized_tree)
    from test_physics_driver import _full_state

    parent, cfg, _ = _state(selector, specified=True, nx=24, ny=24, nz=16,
        dx=12000., dy=12000., dt=6., run_seconds=18., output_interval_s=18.)
    from woof.ingest.lateral_bc import (
        attach_lateral_boundaries, build_state_lateral_boundaries)
    # A fixed physical boundary makes the parent a valid specified ARW
    # domain in both fast and strict nest-coupling modes.
    attach_lateral_boundaries(parent, build_state_lateral_boundaries(
        (parent, parent), (0., 18.), spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone))
    child_cfg = replace(cfg, nx=18, ny=18, dx=4000., dy=4000., dt=2.,
                        grid_id=2, nested=True, specified=False)
    domains = (DomainConfig(grid_id=1, parent_id=0, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=18., run=cfg, time_step=6),
        DomainConfig(grid_id=2, parent_id=1, i_parent_start=7,
        j_parent_start=7, parent_grid_ratio=3, parent_time_step_ratio=3,
        history_interval_s=18., run=child_cfg))
    exp = ExperimentConfig(name="nssl-alias-nest", start_time=datetime(2000, 1, 1),
        run_seconds=18., restart_interval_s=0., domains=domains,
        vertical=VerticalConfig((), 0., cfg.hybrid_opt, cfg.etac),
        projection=ProjectionConfig("lambert", 39., -87., 30., 60., -97.),
        feedback=feedback)

    def prepare(state, dc, grid, start):
        if getattr(state, "physics", None) is None:
            from woof.core.physics import initialize_physics
            return initialize_physics(state, dc.run, swdown=450., glw=310.)
        return state.physics

    model = assemble_idealized_tree(exp, parent, domain_preparer=prepare,
        coupler_factory=lambda node: NestCoupler(node, feedback=feedback))
    execution = execute_experiment(model, experiment=exp)
    assert execution.forces == 3
    # The schedule counts feedback transaction hooks for both settings;
    # the concrete coupler enforces whether they change the parent.
    assert execution.feedback_calls == 3
    assert model.node(2).coupler.feedback_count == (3 if feedback else 0)
    for gid, calls in ((1, 3), (2, 9)):
        node = model.node(gid)
        assert node.state.physics.microphysics_updates == calls
        assert "qnn" in node.state.physics.nssl2_binding.mode.transported_fields
        import cupy as cp
        for name in node.state.physics.nssl2_binding.mode.transported_fields:
            assert bool(cp.all(cp.isfinite(getattr(node.state, name)))), name
    from woof.io import restart
    from woof.prepared_single_domain_forecast import _CheckpointPublisher
    saved = {node.cfg.grid_id: cp.asnumpy(node.state.qnn).copy()
             for node in model.walk_parent_first()}
    pending = restart.write_tree_restart(tmp_path, model,
        exp.start_time + timedelta(seconds=18), defer_publish=True)
    publisher = _CheckpointPublisher()
    publisher.submit(pending.publish)
    publisher.wait()
    root_path = model._last_checkpoint
    for path in tmp_path.glob("gpuwmrst_d*.npz"):
        header = restart.read_restart_header(path)
        with np.load(path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["state/qnn"],
                saved[int(header["config"]["grid_id"])])
    assert root_path is not None
