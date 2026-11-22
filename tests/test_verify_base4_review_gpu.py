"""Actual device and writer artifacts for the base4 landing repairs."""
import dataclasses
import hashlib
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [requires_gpu, pytest.mark.gpu]


def _physical_cfg():
    from idealised_parent_fixture import _parent_run
    return dataclasses.replace(_parent_run(), **{name: 0 for name in (
        "mp_physics", "ra_lw_physics", "ra_sw_physics", "bl_pbl_physics",
        "sf_sfclay_physics", "sf_surface_physics", "cu_physics")})


def _analysis_words(state, cfg):
    """The analysis frame's REFL_10CM words, diagnosed from ``state`` now.

    Since 18dbe1d5449 tick zero diagnoses the initial hydrometeors (the
    scheme's floor, e.g. -35 dBZ, where there are none); under
    GPUWM_WRF_EXACT=1 it is WRF's all-zero initial array.  Prognostics are
    never advanced, so diagnosing again after the write is the same frame.
    """
    import cupy as cp
    from woof.core.refl import analysis_refl_10cm
    field = analysis_refl_10cm(state, cfg, shape=(cfg.nz, cfg.ny, cfg.nx))
    words = np.asarray(cp.asnumpy(field) if isinstance(field, cp.ndarray) else field,
                       dtype=np.float32).view(np.uint32)
    if os.environ.get("GPUWM_WRF_EXACT") == "1":
        assert np.count_nonzero(words) == 0
    else:
        assert np.all(np.isfinite(words.view(np.float32)))
    return words


def test_strict_iau_real_step_keeps_dynamics_increments():
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        pytest.skip("strict process")
    import cupy as cp
    from idealised_parent_fixture import _build_parent_state
    from woof.core.dycore import step
    from woof.da.iau import IauForcing, IauNode, box_density, attach
    cfg = _physical_cfg()
    control, _ = _build_parent_state(cfg)
    forced, _ = _build_parent_state(cfg)
    increments = {name: cp.full_like(getattr(forced, name), amount)
                  for name, amount in (("u", .01), ("v", .02), ("w", .001),
                                       ("thp", .1), ("php", .01))}
    forcing = IauForcing(forced, [IauNode(increments, box_density(0, cfg.dt))])
    attach(forced, forcing)
    step(control, cfg)
    step(forced, cfg)
    cp.cuda.get_current_stream().synchronize()
    for name in increments:
        assert bool(cp.any(getattr(control, name) != getattr(forced, name))), name
        assert bool(cp.all(cp.isfinite(getattr(forced, name)))), name
    assert forcing.steps_forced == 1


def test_ordinary_cold_start_writes_real_reflectivity_artifact(tmp_path):
    from datetime import datetime
    import cupy as cp
    import netCDF4
    from idealised_parent_fixture import _build_parent_state
    from woof.core.physics import initialize_physics
    from woof.static.lambert import LambertGrid
    from woof.runtime import write_case_output
    cfg = dataclasses.replace(_physical_cfg(), mp_physics=6)
    state, coord = _build_parent_state(cfg)
    initialize_physics(state, cfg)
    grid = LambertGrid(ref_lat=35, ref_lon=-97, truelat1=30, truelat2=60,
                       stand_lon=-97, dx=cfg.dx, dy=cfg.dy,
                       e_we=cfg.nx+1, e_sn=cfg.ny+1)
    plane = np.zeros((cfg.ny, cfg.nx), np.float32)
    prepared = SimpleNamespace(cfg=cfg, grid=grid,
        initial_result=SimpleNamespace(state=state, coord=coord),
        static_fields={"HGT_M": plane, "LANDMASK": plane+1, "LU_INDEX": plane+10})
    stamp = datetime(2024, 1, 1)
    path = write_case_output(prepared, tmp_path, stamp, start_time=stamp,
                             title="analysis output regression", expect_refl_10cm=False)
    with netCDF4.Dataset(path) as dataset:
        field = dataset["REFL_10CM"][0]
        assert field.shape == (cfg.nz, cfg.ny, cfg.nx)
        np.testing.assert_array_equal(
            np.asarray(field, dtype=np.float32).view(np.uint32),
            _analysis_words(state, cfg))
    assert state.physics.refl_10cm is None


def test_native_ensemble_tick_zero_writes_real_reflectivity_artifacts(tmp_path):
    if os.environ.get("GPUWM_WRF_EXACT") == "1":
        pytest.skip("the complete multi-member forecast graph is default-mode only")
    import cupy as cp
    import netCDF4
    from idealised_parent_fixture import _parent_run, _experiment, _wire
    from woof.ensemble.native_forecast import run_initialized_native_ensemble
    from woof.ensemble.prepared_batch import native_prepared_eligibility
    from woof.io.wrfout import WrfoutWriter, state_frame
    from woof.runtime import _metadata_frame, _global_wrf_attrs
    cfg = dataclasses.replace(_parent_run(), mp_physics=8, ra_physics=4,
                              ra_lw_physics=4, ra_sw_physics=4,
                              ra_rrtmg_variant="rrtmg_legacy")
    exp = _experiment(cfg)
    model, node, _, _, _ = _wire(exp, attach_nest=False)
    metadata = _metadata_frame(node.grid, {
        "HGT_M": cp.asnumpy(node.state.ht),
        "LANDMASK": np.ones((cfg.ny, cfg.nx), np.float32),
        "LU_INDEX": np.full((cfg.ny, cfg.nx), 10, np.float32)})
    inputs = SimpleNamespace(experiment=exp, domains=(exp.root,), source="prepared",
                             execution_plan={"kind": "root"})
    eligibility = native_prepared_eligibility(inputs, node, members=4)
    assert eligibility.eligible, eligibility.receipt()
    class ArtifactCollector:
        members = 4
        keep_member_files = False
        def __init__(self):
            self.analysis = []
        def submit(self, *, state, metadata, refl_field, valid_time, member_id, **kwargs):
            if valid_time != exp.start_time:
                return
            frame = state_frame(state, include_diagnostic_pressure=True)
            frame.update(metadata)
            if refl_field is not None:
                frame["REFL_10CM"] = cp.asnumpy(refl_field)
            path = tmp_path / f"member-{member_id}.nc"
            attrs = _global_wrf_attrs(node.grid, exp.start_time, domain=cfg)
            with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                              dx=cfg.dx, dy=cfg.dy, field_schema=frame,
                              global_attrs=attrs) as writer:
                writer.write_frame(valid_time.strftime("%Y-%m-%d_%H:%M:%S"), frame)
            self.analysis.append(path)
        def require_complete(self):
            assert len(self.analysis) == 4
            return {"complete": True}
    collector = ArtifactCollector()
    # The ordinary domain's own analysis diagnostic, taken before the batch
    # adopts the node: every member's tick-zero frame carries these words.
    expected = _analysis_words(node.state, cfg)
    report = run_initialized_native_ensemble(inputs, node, members=4,
        collector=collector, available_bytes=8*1024**3, output_metadata=metadata)
    assert report is not None and report["status"] == "PASS"
    for path in collector.analysis:
        with netCDF4.Dataset(path) as dataset:
            field = np.asarray(dataset["REFL_10CM"][0], dtype=np.float32)
            assert field.shape == (cfg.nz, cfg.ny, cfg.nx)
            np.testing.assert_array_equal(field.view(np.uint32), expected)


def test_offline_child_tick_zero_writes_real_reflectivity_artifact(tmp_path):
    from datetime import datetime
    import cupy as cp
    import netCDF4
    from idealised_parent_fixture import _build_parent_state
    from woof.core.physics import initialize_physics
    from woof.static.lambert import LambertGrid
    from woof.runtime import _global_wrf_attrs
    import woof.offline_child_run as offline
    cfg = dataclasses.replace(_physical_cfg(), mp_physics=6)
    state, coord = _build_parent_state(cfg)
    initialize_physics(state, cfg)
    grid = LambertGrid(ref_lat=35, ref_lon=-97, truelat1=30, truelat2=60,
                       stand_lon=-97, dx=cfg.dx, dy=cfg.dy,
                       e_we=cfg.nx+1, e_sn=cfg.ny+1)
    lat, lon = grid.latlon_mass()
    stamp = datetime(2024, 1, 1)
    initial = SimpleNamespace(valid_time=stamp, fields={"XLAT": lat, "XLONG": lon})
    placement = SimpleNamespace(i_parent_start=1, j_parent_start=1, parent_grid_ratio=3)
    if hasattr(offline, "_history_refl_10cm"):
        reflected = offline._history_refl_10cm(state, cfg, 0)
        assert offline._history_refl_10cm(state, cfg, 120) is None
    else:
        from woof.core.refl import consume_refl_10cm, refl_10cm_is_stashed
        reflected = consume_refl_10cm(state) if refl_10cm_is_stashed(state) else None
    path = tmp_path / "offline.nc"
    offline._write_frame(path, state, cfg, initial, stamp,
        _global_wrf_attrs(grid, stamp, domain=cfg, coord=coord), placement,
        refl_field=reflected)
    with netCDF4.Dataset(path) as dataset:
        field = np.asarray(dataset["REFL_10CM"][0], dtype=np.float32)
        assert field.shape == (cfg.nz, cfg.ny, cfg.nx)
        np.testing.assert_array_equal(field.view(np.uint32), _analysis_words(state, cfg))


@pytest.mark.parametrize("boundary", ["specified", "nested"])
@pytest.mark.parametrize("first_in_loop", [True, False])
def test_direct_acoustic_matches_prepared_full_fields(boundary, first_in_loop, monkeypatch):
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        pytest.skip("strict process")
    import cupy as cp
    from woof.core import acoustic
    from woof.core.kernels import module_source
    from woof.ensemble.batch_kernel import _active_source
    from woof.verify.npref import random_acoustic_state
    direct, cfg = random_acoustic_state(seed=543, nz=7, ny=8, nx=9)
    reference, _ = random_acoustic_state(seed=543, nz=7, ny=8, nx=9)
    cfg = dataclasses.replace(cfg, specified=boundary == "specified",
                              nested=boundary == "nested", emdiv=0.0)
    source = _active_source(module_source("acoustic"), ("-DGPUWM_WRF_EXACT=1",))
    original = acoustic.get_kernel
    def checked(module, entry):
        kernel = original(module, entry)
        signature = re.search(r"void " + entry + r"\(([^)]*)\)", source)
        count = signature[1].count(",") + 1
        def launch(grid, block, args):
            # Audit the real ABI before execution, so a regression cannot
            # dereference missing pointers on the card.
            assert len(args) == count, (entry, len(args), count)
            return kernel(grid, block, args)
        return launch
    monkeypatch.setattr(acoustic, "get_kernel", checked)
    for state in (direct, reference):
        state.scratch((cfg.nz + 1, cfg.ny, cfg.nx), "rk_ww")[...] = 0
    coefficients = acoustic.prepare_acoustic_coefficients(reference, cfg, .25)
    prepared = acoustic.prepare_acoustic_substep_launch(reference, cfg, .25, coefficients)
    # The dycore seeds this carrier at stage start. The direct public API
    # must perform the same work for a caller without RK bookkeeping.
    if hasattr(acoustic, "init_wrf_acoustic_muts"):
        acoustic.init_wrf_acoustic_muts(reference)
    for first in (first_in_loop, False):
        acoustic.acoustic_substep(direct, cfg, .25, first)
        prepared(first=first)
        cp.cuda.get_current_stream().synchronize()
        for name in ("u_pp", "v_pp", "mu_pp", "th_pp", "ww_pp", "w_pp", "ph_pp", "p_pp", "al_pp"):
            got, want = cp.asnumpy(getattr(direct, name)), cp.asnumpy(getattr(reference, name))
            assert got.view(np.uint32).tobytes() == want.view(np.uint32).tobytes(), name


@pytest.mark.parametrize("producer", ["sase", "skebs", "sfire"])
def test_strict_physics_producers_leave_map_division_to_fold(producer):
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        pytest.skip("strict process")
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.core.physics import couple_sase_w_tendency
    from woof.ensemble.stochastic_execution import coupled_mass_factors
    nz, ny, nx = 3, 6, 7
    z = cp.zeros((nz, ny, nx), cp.float32)
    mu = cp.full((ny, nx), 8192, cp.float32)
    msft = cp.full((ny, nx), 1.03125, cp.float32)
    state = SimpleNamespace(p=z, mup=mu, mub2d=cp.zeros_like(mu),
        c1h=cp.ones(nz, cp.float32), c2h=cp.zeros(nz, cp.float32),
        c1f=cp.ones(nz+1, cp.float32), c2f=cp.zeros(nz+1, cp.float32),
        msft=msft, msfu=cp.full((ny, nx+1), 1.03125, cp.float32),
        msfv=cp.full((ny+1, nx), 1.03125, cp.float32), has_msf=True,
        total_mu=lambda: mu)
    cfg = SimpleNamespace(specified=False, nested=False, open_x=False, open_y=False)
    if producer == "sase":
        rw = couple_sase_w_tendency(state, cfg, cp.ones_like(z))
        assert bool(cp.all(rw[1:-1] == 8192))
        return
    if producer == "skebs":
        assert bool(cp.all(coupled_mass_factors(state, cfg)["theta"] == 8192))
        return
    th = cp.full_like(z, 32)
    qv = cp.full_like(z, .125)
    rt, rq = cp.zeros_like(z), cp.zeros_like(z)
    get_kernel("sfire_coupling", "sfire_stage_tendencies")(
        ((z.size+255)//256,), (256,),
        (th, qv, msft, rt, rq, np.int32(z.size), np.int32(nx*ny), np.int32(1)))
    assert bool(cp.all(rt == th)) and bool(cp.all(rq == qv))


def test_real_engine_full_field_hash_receipt():
    """A real terrain integration, with every serialized field hashed."""
    import cupy as cp
    from idealised_parent_fixture import _parent_run, _build_parent_state
    from woof.core.dycore import step
    from woof.core.device_inventory import state_array_shapes
    cfg = _physical_cfg()
    state, _ = _build_parent_state(cfg)
    snapshots = []
    for index in range(3):
        step(state, cfg)
        cp.cuda.get_current_stream().synchronize()
        # The real end-of-run capsule walks this manifest too. A control
        # marker must not turn a completed integration into a digest crash.
        from woof.io.restart import state_manifest
        state_manifest(state)
        snapshots.append({name: hashlib.sha256(cp.asnumpy(getattr(state, name)).tobytes()).hexdigest()
                          for name in state_array_shapes(cfg)})
        assert all(bool(cp.all(cp.isfinite(getattr(state, name))))
                   for name in ("u", "v", "w", "thp", "mup", "qv"))
    receipt = {"strict": os.environ.get("GPUWM_WRF_EXACT") == "1",
               "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz, "dt": cfg.dt,
               "steps": snapshots}
    destination = os.environ.get("BASE4_HASH_RECEIPT")
    if destination:
        from pathlib import Path
        Path(destination).write_text(json.dumps(receipt, indent=2))


def test_analysis_frame_leaves_no_values_in_the_shared_refl_scratch(monkeypatch):
    """The analysis diagnostic writes the shared ``refl_10cm`` scratch; the
    boundary ring is never rewritten by forecast microphysics, so the
    analysis values must not stay there (they froze the edge at t0)."""
    import cupy as cp
    from woof.core import refl, state_reflectivity

    slots = {}

    def scratch(shape, name):
        if name not in slots:
            slots[name] = cp.zeros(shape, dtype=cp.float32)
        return slots[name]

    def diagnose(state, cfg):
        out = state.scratch(tuple(state.qv.shape), "refl_10cm")
        out.fill(-35.0)
        return out

    monkeypatch.setattr(state_reflectivity, "diagnose", diagnose)
    state = SimpleNamespace(qv=cp.ones((3, 6, 5), cp.float32), scratch=scratch)
    frame = refl.analysis_refl_10cm(state, SimpleNamespace(mp_physics=8))
    assert bool(cp.all(frame == -35.0))
    assert not bool(cp.any(slots["refl_10cm"]))
