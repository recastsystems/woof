"""MRF (``bl_pbl_physics = 99``) on the card against WRF v4.6.1's own words.

``tools/mrf_wrf461_oracle`` runs WRF's compiled ``module_bl_mrf.o`` (the
scalar referee build, gfortran 13.3.0 -O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize) on 73 columns for 4 steps each -- convective land,
stable night, neutral windy, cloud-topped, very shallow, PBL filling a
short column, water, and probes placed on single comparisons -- and the
fixture carries every input and output word.  ``kernels/mrf.cu`` is
launched on each step's recorded inputs through the production launcher.

Two arithmetic arms run from the same kernel source:

* ``GPUWM_WRF_EXACT=1`` (--fmad=false --ftz=false): every tendency, UST,
  HOL, PBLH and KPBL is WRF's word, bit for bit, signed zeros included.
  KPBL is compared through the one declared divergence -- the port
  publishes WRF's top-down KPBL0 as the bottom-up level nz+1-KPBL0
  (kernels/mrf.cu header).
* default fast arithmetic: the shared compiler boundary permits contraction
  and retains no-FTZ. The measured budget below also holds on the current
  RTX 5090 with CuPy 14.2 and NVRTC 13.4.59.

A planted one-bit change in one kernel constant must move the answer.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.fp32_ulp import bitwise_identical, fp32_ulp_distance
from woof.verify.mrf_oracle import (LEVEL_OUTPUTS, load_mrf_oracle,
                                     run_mrf_on_oracle, wrf_kpbl_bottom_up)

#: Worst ULP distance of the DEFAULT-arithmetic kernel from WRF, per field,
#: over every step of every column.  Measured on G2 (RTX PRO 6000
#: Blackwell, sm_120, CuPy 14.2.0, NVRTC 12.9.86).
DEFAULT_MAX_ULP: dict[str, int] = {
    name: 0 for name in LEVEL_OUTPUTS + ("ust", "hol", "pbl")
}

FLOAT_FIELDS = LEVEL_OUTPUTS + ("ust", "hol", "pbl")


@pytest.fixture(scope="module")
def oracle():
    return load_mrf_oracle()


@pytest.fixture(scope="module")
def card(oracle):
    return run_mrf_on_oracle(oracle)


def _want(oracle, name):
    if name in LEVEL_OUTPUTS:
        return oracle.levels[name]
    return oracle.surface[name]


def _mask(oracle, name):
    """RQIBLTEN is a WRF word only where FLAG_QI is true (:1259)."""
    if name == "rqiblten":
        return np.broadcast_to(oracle.flag_qi[None, None, :],
                               oracle.levels[name].shape)
    return np.ones(_want(oracle, name).shape, dtype=bool)


def measured_ulp(oracle, card) -> dict[str, int]:
    out = {}
    for name in FLOAT_FIELDS:
        m = _mask(oracle, name)
        d = fp32_ulp_distance(card[name][m], _want(oracle, name)[m])
        out[name] = int(d.max())
    return out


@pytest.mark.gpu
@requires_gpu
def test_every_step_of_every_column_matches_wrf(oracle, card):
    from woof import wrf_exact

    kpbl_want = wrf_kpbl_bottom_up(oracle.kpbl, oracle.nz)
    if wrf_exact.ENABLED:
        print("strict-arithmetic max ULP", measured_ulp(oracle, card))
        for name in FLOAT_FIELDS:
            m = _mask(oracle, name)
            assert bitwise_identical(card[name][m],
                                     _want(oracle, name)[m]), (
                name, measured_ulp(oracle, card))
        np.testing.assert_array_equal(card["kpbl"], kpbl_want)
    else:
        measured = measured_ulp(oracle, card)
        print("default-arithmetic max ULP", measured)
        for name, budget in DEFAULT_MAX_ULP.items():
            assert measured[name] <= budget, (name, measured)
        np.testing.assert_array_equal(card["kpbl"], kpbl_want)


@pytest.mark.gpu
@requires_gpu
def test_flag_qi_false_leaves_rqiblten_zero(oracle, card):
    off = ~oracle.flag_qi
    assert off.any() and oracle.flag_qi.any()
    assert np.all(card["rqiblten"][:, :, off] == 0.0)


@pytest.mark.gpu
@requires_gpu
def test_a_planted_bit_in_one_constant_is_caught(oracle, card):
    """CFAC (module_bl_mrf.F:307, 7.8) moved by ONE ulp must move outputs.

    The edited source compiles through the production no-FTZ route
    (woof.core.mrf.compile_mrf_module; in a strict process through the same
    exact-mode hook) and runs through the production launcher.
    """
    from woof.core.mrf import compile_mrf_module, module_source

    src = module_source()
    anchor = "#define MRF_CFAC    7.8f"
    assert src.count(anchor) == 1
    bumped = np.nextafter(np.float32(7.8), np.float32(8.0))
    word = int(np.array([bumped], np.float32).view(np.uint32)[0])
    planted = src.replace(anchor,
                          f"#define MRF_CFAC    __int_as_float({word:#x})")
    fn = compile_mrf_module(planted).get_function("mrf_column")
    bad = run_mrf_on_oracle(oracle, kernel=fn)
    moved = [name for name in FLOAT_FIELDS
             if not bitwise_identical(bad[name][_mask(oracle, name)],
                                      card[name][_mask(oracle, name)])]
    assert moved, "a one-ulp change in CFAC moved no output word"
    vs_wrf = [name for name in FLOAT_FIELDS
              if not bitwise_identical(bad[name][_mask(oracle, name)],
                                       _want(oracle, name)[_mask(oracle,
                                                                 name)])]
    assert vs_wrf, "the planted kernel still matches WRF everywhere"
    print("planted CFAC moved", moved)


@pytest.mark.gpu
@requires_gpu
def test_a_tiled_launch_equals_a_single_one(oracle):
    """Columns go in workspace tiles; tiling must change no word.

    Step 0 of every dt = 60 s, FLAG_QI column, repeated four times along x
    (well over one 32-column block), launched once with the card-sized tile
    and once with a one-block tile; every replica must also equal the first.
    """
    import cupy as cp

    import woof.core.mrf as mrf
    from woof.verify.mrf_oracle import LEVEL_INPUTS

    dt = oracle.surface["dt"][0]
    sel = np.flatnonzero((dt == np.float32(60.0)) & oracle.flag_qi)
    assert sel.size > 8
    rep = 4

    def lev(name):
        a = oracle.levels[name][0][:, sel]
        return cp.asarray(np.ascontiguousarray(np.tile(a, (1, rep))[:, None]))

    def sfc(name):
        a = oracle.surface[name][0][sel]
        return cp.asarray(np.ascontiguousarray(np.tile(a, rep)[None]))

    def launch():
        ust = sfc("ust_in")
        res = mrf.launch_mrf(
            *(lev(n) for n in LEVEL_INPUTS), psfc=sfc("psfc"), ust=ust,
            psim=sfc("psim"), psih=sfc("psih"), xland=sfc("xland"),
            hfx=sfc("hfx"), qfx=sfc("qfx"), gz1oz0=sfc("gz1oz0"),
            wspd=sfc("wspd_in"), br=sfc("br"), dt=60.0, flag_qi=True)
        res["ust"] = ust
        return {k: cp.asnumpy(v) for k, v in res.items()}

    whole = launch()
    original = mrf.mrf_tile_columns
    try:
        mrf.mrf_tile_columns = lambda fn, ncol: mrf.MRF_BLOCK
        tiled = launch()
    finally:
        mrf.mrf_tile_columns = original
    n = sel.size
    for name, a in whole.items():
        b = tiled[name]
        assert np.array_equal(a.view(np.uint32) if a.dtype == np.float32
                              else a, b.view(np.uint32)
                              if b.dtype == np.float32 else b), name
        first = a[..., :n]
        for r in range(1, rep):
            part = a[..., r * n:(r + 1) * n]
            assert np.array_equal(
                part.view(np.uint32) if part.dtype == np.float32 else part,
                first.view(np.uint32) if first.dtype == np.float32
                else first), (name, r)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("bldt", [0.0, 1.0])
def test_short_mrf_restart_is_bit_identical(tmp_path, bldt):
    """20 steps + restart + 20 steps == 40 steps, every serialized word.

    The full-physics restart harness of tests/test_restart.py with the PBL
    slot on MRF (bl_pbl_physics=99, classic MM5 surface layer).  At
    bldt = 1 min the PBL is due every sixth 10 s step, so the held MRF
    tendencies span the step-20 boundary and must survive serialization;
    at bldt = 0 the UST MRF rewrites does, since the surface layer reads it
    back on the next step.
    """
    import cupy as cp

    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal, _physics_state

    overrides = dict(bl_pbl_physics=99, sf_sfclay_physics=91, bldt=bldt)
    state_a, cfg_a, _ = _physics_state(cp, **overrides)
    run_steps(state_a, cfg_a, 40)
    reference = restart.write_restart(tmp_path / "reference.npz",
                                      state_a, cfg_a)
    state_b, cfg_b, _ = _physics_state(cp, **overrides)
    run_steps(state_b, cfg_b, 20)
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)
    state_c, cfg_c, _ = _physics_state(cp, **overrides)
    restart.restore_restart(mid, state_c, cfg_c)
    run_steps(state_c, cfg_c, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz", state_c, cfg_c)
    _assert_restart_equal(resumed, reference)
    import hashlib
    from woof.io.restart import _HEADER_KEY
    with np.load(reference, allow_pickle=False) as archive:
        digest = hashlib.sha256()
        names = sorted(name for name in archive.files if name != _HEADER_KEY)
        for name in names:
            value = archive[name]
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes())
        print("restart payload", {"bldt": bldt, "arrays": len(names),
                                   "sha256": digest.hexdigest(),
                                   "all_array_bytes_identical": True})
    header = restart.read_restart_header(resumed)
    assert "fields/hol" in header["array_manifest"]
    f = state_c.physics.fields
    assert state_c.physics.scheme_dispatch["bl_pbl_physics"] == "_run_mrf"
    assert state_c.physics.call_counts["ysu"] == (40 if bldt == 0.0 else 7)
    assert bool(cp.isfinite(f["pblh"]).all()) and bool((f["hol"] != 0).any())
    assert bool((state_c.physics.pbl_tendencies.rtheta != 0).any())



@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("surface_layer", [1, 91])
def test_mrf_cold_start_and_surface_passthrough(surface_layer):
    """Both admitted surface layers reach MRF; its call preserves their diagnostics."""
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.core.physics import _prepare_atmosphere
    from test_restart import _physics_state

    state, cfg, driver = _physics_state(cp, bl_pbl_physics=99,
                                       sf_sfclay_physics=surface_layer)
    for name in ("ru", "rv", "rtheta", "rqv", "rqc", "rqi"):
        value = getattr(driver.pbl_tendencies, name)
        assert value is None or bool((value == 0).all()), name
    assert bool((driver.fields["hol"] == 0).all())
    run_steps(state, cfg, 1)
    preserved = {name: cp.asnumpy(driver.fields[name]).view(np.uint32).copy()
                 for name in ("zol", "regime", "wspd")}
    assert bool(cp.isfinite(driver.fields["pblh"]).all())
    driver._run_mrf(_prepare_atmosphere(state), cfg)
    for name, words in preserved.items():
        np.testing.assert_array_equal(cp.asnumpy(driver.fields[name]).view(np.uint32), words)
    from woof.io.history_layout import live_state_history_fields
    history = live_state_history_fields(state)
    assert history["HOL"] is driver.fields["hol"]
    assert history["KPBL"] is driver.fields["kpbl"]
    print("surface passthrough", {"sf_sfclay_physics": surface_layer,
                                  "zol_regime_wspd_words_identical": True})


@pytest.mark.gpu
@requires_gpu
def test_output_validation_is_capturable_and_reports_at_the_ledger_drain():
    import cupy as cp
    from woof.core.health_ledger import HealthLedger, deferring
    from woof.core.mrf import MRF_3D_OUTPUTS, validate_mrf_outputs

    out = {name: cp.zeros((3, 2, 4), cp.float32) for name in MRF_3D_OUTPUTS}
    out.update(pblh=cp.zeros((2, 4), cp.float32), hol=cp.zeros((2, 4), cp.float32))
    ust = cp.ones((2, 4), cp.float32)
    ledger = HealthLedger()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream, deferring(ledger):
        validate_mrf_outputs(out, ust)  # warm the independent validator
    stream.synchronize()
    with stream, deferring(ledger):
        stream.begin_capture()
        out["hol"].fill(cp.nan)
        validate_mrf_outputs(out, ust)
        graph = stream.end_capture()
        graph.launch(stream)
    stream.synchronize()
    assert ledger.records == 2 and ledger.drains == 0
    with pytest.raises(FloatingPointError, match="non-finite hol"):
        ledger.drain()


@pytest.mark.gpu
@requires_gpu
def test_real_native_forecast_capture():
    from woof import numerics
    from woof.verify.mrf_real_capture import replay
    if not numerics.is_strict():
        pytest.skip("real native capture is the strict module proof")
    result = replay()
    print("native real-input replay", result)
    assert result["compared_words"] == 1192
    assert result["differing_words"] == 0


@pytest.mark.gpu
@requires_gpu
def test_mrf_ensemble_members_match_their_original_drivers():
    import cupy as cp
    from woof.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor
    from woof.ensemble.suite_capabilities import selected_capabilities
    from woof.io.restart import state_manifest, _driver_manifest
    from test_restart import _physics_state

    opts = dict(bl_pbl_physics=99, sf_sfclay_physics=91)
    grouped = [_physics_state(cp, **opts) for _ in range(2)]
    original = [_physics_state(cp, **opts) for _ in range(2)]
    for roster in (grouped, original):
        for index, (state, cfg, driver) in enumerate(roster):
            state.u += cp.float32(index * 2)
            pbl = next(c for c in selected_capabilities(cfg) if c.component == "pbl")
            assert pbl.option_id == "mrf" and pbl.implemented
    executor = MemberPhysicsExecutor([
        MemberPhysicsBinding(state, cfg, driver, member_id=index)
        for index, (state, cfg, driver) in enumerate(grouped)])
    for call in range(3):
        for state, cfg, driver in original:
            state.elapsed_seconds = call * cfg.dt
            driver.compute(state, cfg)
        for state, cfg, driver in grouped:
            state.elapsed_seconds = call * cfg.dt
        executor.compute()
        for index, ((actual, _, adriver), (expected, _, edriver)) in enumerate(zip(grouped, original)):
            actual_fields, expected_fields = state_manifest(actual), state_manifest(expected)
            actual_fields.update(_driver_manifest(adriver))
            expected_fields.update(_driver_manifest(edriver))
            assert actual_fields.keys() == expected_fields.keys()
            for name in actual_fields:
                assert cp.asnumpy(actual_fields[name]).tobytes() == cp.asnumpy(expected_fields[name]).tobytes(), (index, call, name)
        for _, _, driver in original:
            driver.finish_step()
        executor.finish_step()
    print("MRF ensemble original-operation identity", executor.call_counts)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("feedback", [0, 1])
def test_mrf_nested_full_steps(feedback):
    from woof.wrf_exact import ENABLED
    if ENABLED:
        pytest.skip("This periodic idealized root has no WRF specified-boundary state; nesting is checked in default arithmetic")
    import cupy as cp
    from datetime import datetime
    from types import SimpleNamespace
    from woof.experiment import DomainConfig, ExperimentConfig, ProjectionConfig, VerticalConfig
    from woof.core.model import execute_experiment
    from woof.core.nest import NestCoupler
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree, consume_history_reflectivity
    from test_restart import _physics_state

    common = dict(nx=24, ny=24, bl_pbl_physics=99, sf_sfclay_physics=91,
                  run_seconds=24., output_interval_s=12.)
    parent, pcfg, _ = _physics_state(cp, **common, dt=12.)
    child, ccfg, _ = _physics_state(cp, **common, dt=4., dx=2000./3., dy=2000./3.,
                                  grid_id=2, nested=True, specified=False)
    root = DomainConfig(grid_id=1, parent_id=0, parent_grid_ratio=1,
                        i_parent_start=1, j_parent_start=1,
                        parent_time_step_ratio=1, history_interval_s=12.,
                        time_step=12, run=pcfg)
    nested = DomainConfig(grid_id=2, parent_id=1, parent_grid_ratio=3,
                          parent_time_step_ratio=3, i_parent_start=5, j_parent_start=5,
                          history_interval_s=12., run=ccfg)
    exp = ExperimentConfig(name="mrf-nested", start_time=datetime(2000,1,1),
        run_seconds=24., vertical=VerticalConfig((),0.,1,.2),
        projection=ProjectionConfig("lambert",35.,-97.,30.,60.,-97.),
        restart_interval_s=0., domains=(root,nested), feedback=feedback)
    model = assemble_idealized_tree(exp, parent,
        child_initializer=lambda dc,p: SimpleNamespace(state=child),
        domain_preparer=lambda *args: None,
        coupler_factory=lambda node: NestCoupler(node, feedback=feedback))
    report = execute_experiment(model, history_handler=lambda model, node, ticks:
                                consume_history_reflectivity(node, ticks))
    assert report.steps == 8 and report.forces == 2
    for state in (parent,child):
        assert state.physics.scheme_dispatch["bl_pbl_physics"] == "_run_mrf"
        assert bool(cp.isfinite(state.physics.fields["pblh"]).all())
        assert bool(cp.isfinite(state.physics.fields["hol"]).all())
        assert state.physics.call_counts["ysu"] > 0
    assert parent.physics.fields["hol"] is not child.physics.fields["hol"]
    print("MRF nested full steps", {"feedback":feedback,"steps":report.steps,
                                     "forces":report.forces,"feedback_calls":report.feedback_calls})
