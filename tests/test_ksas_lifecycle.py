"""KSAS continuation, history, nested clocks and independent member state."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu


def _state():
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    from woof.verify.cases.wk82 import wk82_sounding

    cfg = RunConfig(nx=8, ny=8, nz=45, dx=12000., dy=12000., ztop=14000.,
                    dt=10., run_seconds=0., time_step_sound=4, moist=True,
                    mp_physics=6, sf_sfclay_physics=1, sf_surface_physics=2,
                    bl_pbl_physics=1, ra_physics=90, radt_minutes=1.,
                    cu_physics=14, cudt_minutes=1.)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord,
        lambda z: wk82_sounding(z)[0],
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
        lambda z: wk82_sounding(z)[1])
    state.u[...] = cp.float32(4.)
    state.v[...] = cp.float32(1.)
    land = np.ones((cfg.ny, cfg.nx))
    land[:, -1] = 0.
    tsk = np.full(land.shape, 308.)
    tsk[:, -1] = 303.
    soil_t = np.stack([tsk - .5, tsk - 1., tsk - 1.5, tsk - 2.])
    soil_m = np.full((4,) + land.shape, .35)
    soil_m[:, :, -1] = 1.
    driver = initialize_physics(state, cfg, landmask=land, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m, liquid_moisture=soil_m,
        ivgtyp=np.where(land, 10, 17), isltyp=np.where(land, 6, 14),
        vegfra=55., tmn=292., swdown=850., glw=380., pblh=1200.,
        radiation_start_time=datetime(2000, 1, 1, 18),
        radiation_latitude=np.full(land.shape, 39.),
        radiation_longitude=np.full(land.shape, -87.))
    return state, cfg, driver


def _active_handoff(state, cfg, driver, tmp_path):
    """Produce live held rates with the graded fixture's native columns.

    This continuation test needs nonzero held state at the split. A
    horizontally uniform idealized state need not trigger deep convection.
    Launch the stock callable on the existing oracle operands at that
    handoff; the uninterrupted and restored arms receive the same call.
    This is a constructed continuation check, not a new native-run replay.
    """
    import gzip
    import importlib.util
    import cupy as cp

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_ksas_lifecycle_cases",
        root / "tools/ksas_wrf461_oracle/gen_cases.py")
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    path = tmp_path / "cases.bin"
    path.write_bytes(gzip.decompress((root /
        "tests/fixtures/ksas_wrf461/cases.bin.gz").read_bytes()))
    group = cases.read(path)[0]
    assert (group["nz"], group["ncol"]) == (cfg.nz, cfg.nx * cfg.ny)
    operands = group["arrays"]
    mapping = dict(temperature="t3d", qv="qv3d", qc="qc3d", qi="qi3d",
        pressure="p3d", exner="pi3d", rho="rho3d", dz="dz8w", u="u3d", v="v3d")
    atmosphere = {name: cp.asarray(operands[key]).reshape(cfg.nz, cfg.ny, cfg.nx)
                  for name, key in mapping.items()}
    saved = state.w.copy(), state.znu.copy(), driver.fields["pblh"].copy()
    try:
        state.w[...] = cp.asarray(operands["w"]).reshape(cfg.nz + 1, cfg.ny, cfg.nx)
        state.znu[...] = cp.asarray(group["znu"])
        driver.fields["pblh"][...] = cp.asarray(operands["hpbl"]).reshape(cfg.ny, cfg.nx)
        driver.cumulus_callable.hold_pbl_height(
            cp.asarray(operands["hpbl_hold"]).reshape(cfg.ny, cfg.nx))
        driver._run_cumulus(atmosphere, state, cfg)
    finally:
        state.w[...] = saved[0]
        state.znu[...] = saved[1]
        driver.fields["pblh"][...] = saved[2]
    assert bool(cp.any(driver.cu_rates["rthcuten"] != 0.))
    assert bool(cp.any(driver.fields["ksas_rucuten"] != 0.))
    assert bool(cp.any(driver.fields["htop"] != 0.))


@requires_gpu
def test_split_run_preserves_every_restart_word(tmp_path):
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal

    state, cfg, driver = _state()
    run_steps(state, cfg, 20)
    _active_handoff(state, cfg, driver, tmp_path)
    run_steps(state, cfg, 20)
    reference = restart.write_restart(tmp_path / "continuous.npz", state, cfg)
    split, cfg, driver = _state()
    run_steps(split, cfg, 20)
    _active_handoff(split, cfg, driver, tmp_path)
    middle = restart.write_restart(tmp_path / "middle.npz", split, cfg)
    fresh, cfg, driver = _state()
    restart.restore_restart(middle, fresh, cfg)
    assert driver.cumulus_callable._workspace is None
    run_steps(fresh, cfg, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz", fresh, cfg)
    _assert_restart_equal(reference, resumed)
    assert driver.call_counts["cumulus"] == 7
    keys = restart.read_restart_header(resumed)["array_manifest"]
    for name in ("hbot", "htop", "ksas_rucuten", "ksas_rvcuten"):
        assert "fields/" + name in keys


@requires_gpu
def test_retained_bounds_reach_the_production_history_writer(tmp_path):
    import cupy as cp
    import netCDF4
    from woof.core.dycore import run_steps
    from test_wrfout_conformance import _write_real_wrfout

    state, cfg, driver = _state()
    run_steps(state, cfg, 20)
    _active_handoff(state, cfg, driver, tmp_path)
    path = tmp_path / "wrfout_d01_ksas"
    frame = _write_real_wrfout(path, state, cfg)
    with netCDF4.Dataset(path) as ds:
        assert ds.CU_PHYSICS == 14
        for key in ("hbot", "htop"):
            variable = ds.variables[key.upper()]
            assert variable.dimensions == ("Time", "south_north", "west_east")
            assert variable.dtype == np.dtype("float32")
            assert variable.FieldType == 104 and variable.stagger == ""
            expected = cp.asnumpy(driver.fields[key]).view(np.uint32)
            assert np.array_equal(frame[key.upper()].view(np.uint32), expected)
            assert np.array_equal(np.asarray(variable[0]).view(np.uint32), expected)


@requires_gpu
def test_member_executor_retains_independent_held_rates_and_workspaces(tmp_path):
    import cupy as cp
    from woof.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor

    first, cfg, one = _state()
    second, _, two = _state()
    second.qv[...] *= cp.float32(.8)
    executor = MemberPhysicsExecutor((MemberPhysicsBinding(first, cfg, one, 1),
                                     MemberPhysicsBinding(second, cfg, two, 2)))
    executor.compute()
    _active_handoff(first, cfg, one, tmp_path)
    before = {name: value.copy() for name, value in one.cu_rates.items()}
    bounds = one.fields["htop"].copy()
    momentum = one.fields["ksas_rucuten"].copy()
    second.elapsed_seconds = 60.
    executor.compute_member(1)
    for name, value in before.items():
        assert cp.array_equal(value.view(cp.uint32), one.cu_rates[name].view(cp.uint32)), name
    assert cp.array_equal(bounds.view(cp.uint32), one.fields["htop"].view(cp.uint32))
    assert cp.array_equal(momentum.view(cp.uint32), one.fields["ksas_rucuten"].view(cp.uint32))
    assert one.cumulus_callable._workspace is not two.cumulus_callable._workspace
    assert executor.call_counts["compute"] == [1, 2]


@requires_gpu
@pytest.mark.parametrize("feedback", [0, 1])
def test_both_domains_step_through_the_nest_executor(feedback):
    from dataclasses import replace
    from woof.core.model import execute_experiment
    from woof.core.nest import NestCoupler
    from woof.experiment import DomainConfig, ExperimentConfig, ProjectionConfig, VerticalConfig
    from woof.ingest.lateral_bc import attach_lateral_boundaries, build_state_lateral_boundaries
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree
    from test_physics_driver import _full_state

    parent, cfg, _ = _full_state(cu_physics=14, cudt_minutes=0., specified=True,
        nx=24, ny=24, nz=16, dx=12000., dy=12000., dt=6., run_seconds=18.,
        output_interval_s=18.)
    attach_lateral_boundaries(parent, build_state_lateral_boundaries(
        (parent, parent), (0., 18.), spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone))
    child = replace(cfg, nx=18, ny=18, dx=4000., dy=4000., dt=2.,
                    grid_id=2, nested=True, specified=False)
    domains = (DomainConfig(grid_id=1, parent_id=0, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=18., run=cfg, time_step=6),
        DomainConfig(grid_id=2, parent_id=1, i_parent_start=7,
        j_parent_start=7, parent_grid_ratio=3, parent_time_step_ratio=3,
        history_interval_s=18., run=child))
    exp = ExperimentConfig(name="ksas-nest", start_time=datetime(2000, 1, 1),
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
    assert execution.forces == 3 and execution.feedback_calls == 3
    assert model.node(2).coupler.feedback_count == (3 if feedback else 0)
    for gid, calls in ((1, 3), (2, 9)):
        node = model.node(gid)
        driver = node.state.physics
        assert driver.call_counts["cumulus"] == calls
        for name in ("hbot", "htop"):
            assert driver.fields[name].shape == (node.cfg.run.ny, node.cfg.run.nx)
        for name in ("ksas_rucuten", "ksas_rvcuten"):
            assert driver.fields[name].shape == (node.cfg.run.nz, node.cfg.run.ny, node.cfg.run.nx)
    assert model.node(1).state.physics.cumulus_callable._workspace is not \
        model.node(2).state.physics.cumulus_callable._workspace
