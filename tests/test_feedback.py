"""Experimental two-way nest feedback activation gates."""

import hashlib
from dataclasses import replace
from datetime import datetime, timedelta
from types import MappingProxyType
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from woof.config import RunConfig
from woof.core.model import FeedbackScratch
from woof.core.nest_interp import feedback_parent_bounds, register_nest
from woof.experiment import DomainConfig


def _domain(
        grid_id, parent_id, *, nx, ny, ratio=1, run_seconds=30.0,
        root_specified=True):
    root = parent_id == 0
    run = RunConfig(
        nx=nx, ny=ny, nz=2, dx=3000.0 if root else 3000.0 / ratio,
        dy=3000.0 if root else 3000.0 / ratio, ztop=12000.0,
        dt=30.0 if root else 30.0 / ratio, run_seconds=run_seconds,
        output_interval_s=30.0, grid_id=grid_id,
        specified=root and root_specified, nested=not root,
        moist=True, mp_physics=0,
        spec_bdy_width=5, spec_zone=1, relax_zone=4)
    return DomainConfig(
        grid_id=grid_id, parent_id=parent_id,
        i_parent_start=1 if root else 4,
        j_parent_start=1 if root else 4,
        parent_grid_ratio=1 if root else ratio,
        parent_time_step_ratio=1 if root else ratio,
        history_interval_s=30.0, run=run,
        time_step=30 if root else None)


def test_feedback_parent_bounds_exclude_child_specified_zone():
    mass = register_nest(
        nri=3, nrj=3, i_parent_start=4, j_parent_start=4,
        child_nx=9, child_ny=9, parent_nx=14, parent_ny=14,
        stagger="", wrapper="bdy")
    xface = register_nest(
        nri=3, nrj=3, i_parent_start=4, j_parent_start=4,
        child_nx=9, child_ny=9, parent_nx=14, parent_ny=14,
        stagger="x", wrapper="bdy")
    assert feedback_parent_bounds(mass, spec_zone=1) == (4, 4, 4, 4)
    assert feedback_parent_bounds(xface, spec_zone=1) == (4, 5, 4, 4)


def test_feedback_refuses_only_the_mismatched_vertical_ladder():
    """One of the three construction refusals survives, and it is the nz one.

    The other two -- "no ratified reverse mass/moment mapping" and
    "requires identical active parent/child prognostic field inventories"
    -- are retired: the reverse edge resolves through the same matrix as
    the forward one and the feedback path diagnoses the parent's species
    from the child's.  Their replacement contracts are CPU-only and live in
    tests/test_nest_feedback_reverse_edge.py, because this module imports
    cupy in a module-level helper and is therefore marked ``gpu`` whole.
    The vertical one is a different item and a genuinely missing
    capability: the reverse operator has no vertical mapping.
    """
    from woof.core.microphysics_transition import (
        MP8_TO_MP18_POLICY, REVERSE_EDGE_POLICY)
    from woof.core.nest import NestCoupler

    parent_cfg = _domain(1, 0, nx=14, ny=14)
    parent_cfg = replace(
        parent_cfg,
        run=replace(parent_cfg.run, mp_physics=8, moist_cq=True))
    child_cfg = _domain(2, 1, nx=9, ny=9, ratio=3)
    child_cfg = replace(
        child_cfg,
        run=replace(
            child_cfg.run, mp_physics=18, moist_cq=True,
            nest_microphysics_transition=MP8_TO_MP18_POLICY))
    parent = SimpleNamespace(cfg=parent_cfg)
    child = SimpleNamespace(cfg=child_cfg, parent=parent)

    NestCoupler(child, feedback=0)
    coupler = NestCoupler(child, feedback=1)
    assert coupler.microphysics_reverse_transition.mixed is True
    assert coupler.microphysics_reverse_transition.policy_id == \
        REVERSE_EDGE_POLICY

    same_scheme = replace(
        child_cfg,
        run=replace(
            parent_cfg.run, grid_id=2, nx=9, ny=9, nz=3,
            dx=1000.0, dy=1000.0, dt=10.0, specified=False,
            nested=True))
    with pytest.raises(ValueError, match="horizontal-only.*vertical level"):
        NestCoupler(
            SimpleNamespace(cfg=same_scheme, parent=parent), feedback=1)


def test_feedback_provenance_is_explicit_and_feedback0_is_absent(tmp_path):
    from datetime import datetime

    from woof.experiment import (
        ExperimentConfig, ProjectionConfig, VerticalConfig)
    from woof.runtime import (
        FEEDBACK_EXPERIMENTAL_WARNING, _global_wrf_attrs,
        _write_feedback_provenance_receipt, feedback_provenance)

    root = _domain(1, 0, nx=14, ny=14)
    base = ExperimentConfig(
        name="feedback-provenance", start_time=datetime(2000, 1, 1),
        run_seconds=30.0, vertical=VerticalConfig((), 0.0, 1, 0.2),
        projection=ProjectionConfig(
            "lambert", 35.0, -97.0, 30.0, 60.0, -97.0),
        restart_interval_s=0.0, domains=(root,))
    assert feedback_provenance(base) is None
    legacy = feedback_provenance(replace(base, feedback=1))
    assert legacy["vertical_mapping"] == \
        "shared-legacy-level-count-horizontal-only"
    experimental = replace(
        base, feedback=1,
        vertical=VerticalConfig((1.0, 0.5, 0.0), 10000.0, 1, 0.2))
    receipt = feedback_provenance(experimental)
    assert receipt["feedback"] == "experimental"
    assert receipt["stock_wrf_certified"] is False
    assert receipt["vertical_mapping"] == \
        "shared-explicit-eta-ladder-horizontal-only"
    assert "not certified against stock WRF yet" in \
        FEEDBACK_EXPERIMENTAL_WARNING

    path, digest, written = _write_feedback_provenance_receipt(
        tmp_path, experimental, resumed=True)
    assert path.name == "feedback-provenance.json"
    assert len(digest) == 64
    assert written["feedback"] == "experimental"
    assert written["resumed"] is True

    grid = SimpleNamespace(
        truelat1=30.0, truelat2=60.0, stand_lon=-97.0,
        ref_lat=35.0, ref_lon=-97.0)
    attrs = _global_wrf_attrs(
        grid, base.start_time, domain=root,
        feedback=receipt)
    assert attrs["GPUWM_FEEDBACK"] == "experimental"
    assert attrs["GPUWM_FEEDBACK_VALUE"] == 1
    assert attrs["GPUWM_FEEDBACK_STOCK_WRF_CERTIFIED"] == 0
    assert "GPUWM_FEEDBACK" not in _global_wrf_attrs(
        grid, base.start_time, domain=root)


def _seed_state(state, *, child):
    import cupy as cp

    state.mub2d[...] = cp.float32(128.0)
    state.mup[...] = cp.float32(0.0)
    state.thb[...] = cp.float32(300.0)
    state.c1h[...] = cp.float32(1.0)
    state.c2h[...] = cp.float32(0.0)
    state.c1f[...] = cp.float32(1.0)
    state.c2f[...] = cp.float32(0.0)
    state.rdnw[...] = cp.float32(-0.001)
    state.p_top = np.float32(10000.0)
    for k in range(state.phb.shape[0]):
        state.phb[k, ...] = cp.float32(k * 1000.0)
    if child:
        state.u[...] = cp.float32(2.0)
        state.v[...] = cp.float32(3.0)
        state.w[...] = cp.float32(4.0)
        state.php[...] = cp.float32(8.0)
        state.thp[...] = cp.float32(0.0)
        pattern = cp.arange(1, 10, dtype=cp.float32).reshape(3, 3)
        state.thp[:, 3:6, 3:6] = pattern
        state.qv[...] = cp.float32(0.002)
        state.qc[...] = cp.float32(0.003)
        state.qr[...] = cp.float32(0.004)
    else:
        state.u[...] = cp.float32(-2.0)
        state.v[...] = cp.float32(-3.0)
        state.w[...] = cp.float32(-4.0)
        state.php[...] = cp.float32(-8.0)
        state.thp[...] = cp.float32(-9.0)
        state.qv[...] = cp.float32(0.1)
        state.qc[...] = cp.float32(0.2)
        state.qr[...] = cp.float32(0.3)


def _seed_operator_patterns(state, ratio):
    """Construct discriminating mass/U/V feedback stencils."""
    import cupy as cp

    state.thp[...] = cp.float32(0.0)
    mass = cp.arange(
        1, ratio * ratio + 1, dtype=cp.float32).reshape(ratio, ratio)
    state.thp[:, ratio:2 * ratio, ratio:2 * ratio] = mass

    state.u[...] = cp.float32(0.0)
    state.u[:, ratio:2 * ratio, ratio] = cp.arange(
        1, ratio + 1, dtype=cp.float32)
    state.u[:, ratio:2 * ratio, 2 * ratio] = cp.arange(
        ratio + 1, 2 * ratio + 1, dtype=cp.float32)

    state.v[...] = cp.float32(0.0)
    state.v[:, ratio, ratio:2 * ratio] = cp.arange(
        2 * ratio + 1, 3 * ratio + 1, dtype=cp.float32)
    state.v[:, 2 * ratio, ratio:2 * ratio] = cp.arange(
        3 * ratio + 1, 4 * ratio + 1, dtype=cp.float32)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("ratio", [4, 3], ids=["even-ratio-4", "odd-ratio-3"])
def test_feedback_restricts_exact_mass_faces_and_leaves_rim_untouched(ratio):
    import cupy as cp

    from woof.core.nest import NestCoupler
    from woof.core.preflight import nest_field_kinds
    from woof.core.state import DomainState

    parent_cfg = _domain(1, 0, nx=14, ny=14)
    child_cfg = _domain(
        2, 1, nx=ratio * 3, ny=ratio * 3, ratio=ratio)
    parent_state = DomainState(parent_cfg.run)
    child_state = DomainState(child_cfg.run)
    _seed_state(parent_state, child=False)
    _seed_state(child_state, child=True)
    _seed_operator_patterns(child_state, ratio)
    clock = SimpleNamespace(ticks=0)
    parent = SimpleNamespace(
        cfg=parent_cfg, state=parent_state, clock=clock)
    child = SimpleNamespace(
        cfg=child_cfg, state=child_state, clock=clock, parent=parent)
    coupler = NestCoupler(child, feedback=1)

    before = {}
    for kind in nest_field_kinds(parent_cfg.run):
        name = {"t": "thp", "ph": "php"}.get(kind, kind)
        value = parent_state.mup if kind == "mu" else getattr(
            parent_state, name)
        before[kind] = cp.asnumpy(value).copy()

    scratch = FeedbackScratch()
    coupler.feedback_prepare(child, scratch)
    coupler.feedback_commit(child)
    coupler.feedback_finalize(child)

    assert coupler.feedback_count == 1
    mass_average = (ratio * ratio + 1) / 2.0
    u_face_0 = (ratio + 1) / 2.0
    u_face_1 = (3 * ratio + 1) / 2.0
    v_face_0 = (5 * ratio + 1) / 2.0
    v_face_1 = (7 * ratio + 1) / 2.0
    if ratio == 4:
        # Even 4:1 copy_fcn: exact dyadic 1/16 mass and 1/4 face
        # averages over the constructed 1..16 / four-point stencils.
        assert float(parent_state.thp[0, 4, 4]) == mass_average == 8.5
        assert float(parent_state.u[0, 4, 4]) == u_face_0 == 2.5
        assert float(parent_state.u[0, 4, 5]) == u_face_1 == 6.5
        assert float(parent_state.v[0, 4, 4]) == v_face_0 == 10.5
        assert float(parent_state.v[0, 5, 4]) == v_face_1 == 14.5
    else:
        # Odd 3:1 takes copy_fcn's centered average path.  Its 1/3 and
        # 1/9 weights are FP32-rounded, so compare at the kernel's floor.
        assert float(parent_state.thp[0, 4, 4]) == pytest.approx(
            mass_average, abs=1.0e-6)
        assert float(parent_state.u[0, 4, 4]) == pytest.approx(
            u_face_0, abs=1.0e-6)
        assert float(parent_state.u[0, 4, 5]) == pytest.approx(
            u_face_1, abs=1.0e-6)
        assert float(parent_state.v[0, 4, 4]) == pytest.approx(
            v_face_0, abs=1.0e-6)
        assert float(parent_state.v[0, 5, 4]) == pytest.approx(
            v_face_1, abs=1.0e-6)
    assert float(parent_state.qv[0, 4, 4]) == pytest.approx(
        0.002, abs=1.0e-8)

    footprint = np.zeros(parent_state.mup.shape, dtype=bool)
    footprint[3:6, 3:6] = True
    mass_reg = coupler.registrations["m"]
    i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(
        mass_reg, spec_zone=child_cfg.run.spec_zone)
    interior = np.zeros_like(footprint)
    interior[j_lo:j_hi + 1, i_lo:i_hi + 1] = True
    rim = footprint & ~interior
    np.testing.assert_array_equal(
        cp.asnumpy(parent_state.thp)[:, rim], before["t"][:, rim])

    for kind in nest_field_kinds(parent_cfg.run):
        name = {"t": "thp", "ph": "php"}.get(kind, kind)
        value = parent_state.mup if kind == "mu" else getattr(
            parent_state, name)
        after = cp.asnumpy(value)
        reg = coupler.registrations[
            "x" if kind == "u" else "y" if kind == "v" else "m"]
        i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(
            reg, spec_zone=child_cfg.run.spec_zone)
        mask = np.zeros(after.shape[-2:], dtype=bool)
        mask[j_lo:j_hi + 1, i_lo:i_hi + 1] = True
        outside = np.broadcast_to(~mask, after.shape)
        np.testing.assert_array_equal(after[outside], before[kind][outside])


def _seed_child_column_mass_gradient(state, ratio):
    """A child column-mass ramp across the feedback footprint.

    ``_seed_state`` leaves ``mup = 0``, which makes the coupling mass
    ``c1h*(mub+mup)+c2h`` uniform on BOTH grids -- and a mass-weighted
    average of a uniform mass IS the unweighted average.  Every stencil
    assertion in ``test_feedback_restricts_exact_mass_faces_and_leaves_rim_
    untouched`` therefore passes whether or not the transaction couples.
    The 8 Pa/cell ramp below is what separates the two.
    """
    import cupy as cp

    ny, nx = state.mup.shape
    state.mup[...] = cp.arange(nx, dtype=cp.float32)[None, :] * cp.float32(8.0)
    return 8.0 * (3 * ratio - 1) / 2.0        # its mean over the footprint


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("ratio", [4, 3], ids=["even-ratio-4", "odd-ratio-3"])
def test_feedback_restricts_uncoupled_fields_under_a_child_mass_gradient(
        ratio):
    """WRF's feedback restricts the RAW prognostics, never coupled ones.

    ``share/mediation_feedback_domain.F`` (146 lines, read end to end) and
    ``external/RSL_LITE/feedback_domain_em_part1.F`` / ``_part2.F`` contain
    no ``couple_or_uncouple_em``; ``inc/nest_feedbackup_interp.inc:23-27``
    hands ``copy_fcn`` the bare ``ngrid%u_2`` and
    ``share/interp_fcn.F:1476-1477`` accumulates ``1./REAL(nri*nrj)`` of
    it.  Only the FORCE path couples (``share/mediation_force_domain.F:117``
    parent, ``:129`` nest) and uncouples again (``:184``, ``:196``).

    So the parent must receive the UNWEIGHTED cell/face averages -- the
    same numbers the uniform-mass sibling test asserts -- even when the
    child's column mass varies across the footprint.  A mass-coupled
    transaction returns ``mean(chm*f)/chm_parent`` instead; with this ramp
    that is 8.558 vs 8.5 for theta and 8.147 vs 6.5 for the high u face at
    ratio 4 (5.033 / 5.972 at ratio 3), so the assertions below separate
    the two by far more than FP32 noise.
    """
    import cupy as cp

    from woof.core.nest import NestCoupler
    from woof.core.preflight import nest_field_kinds
    from woof.core.state import DomainState

    parent_cfg = _domain(1, 0, nx=14, ny=14)
    child_cfg = _domain(
        2, 1, nx=ratio * 3, ny=ratio * 3, ratio=ratio)
    parent_state = DomainState(parent_cfg.run)
    child_state = DomainState(child_cfg.run)
    _seed_state(parent_state, child=False)
    _seed_state(child_state, child=True)
    _seed_operator_patterns(child_state, ratio)
    footprint_mu = _seed_child_column_mass_gradient(child_state, ratio)
    clock = SimpleNamespace(ticks=0)
    parent = SimpleNamespace(
        cfg=parent_cfg, state=parent_state, clock=clock)
    child = SimpleNamespace(
        cfg=child_cfg, state=child_state, clock=clock, parent=parent)
    coupler = NestCoupler(child, feedback=1)

    before = {}
    for kind in nest_field_kinds(parent_cfg.run):
        name = {"t": "thp", "ph": "php"}.get(kind, kind)
        value = parent_state.mup if kind == "mu" else getattr(
            parent_state, name)
        before[kind] = cp.asnumpy(value).copy()

    scratch = FeedbackScratch()
    coupler.feedback_prepare(child, scratch)
    coupler.feedback_commit(child)
    # No finalize: it only re-diagnoses p/al/alt, and every claim below is
    # about what the restriction wrote.

    # The discriminator is live only if the parent really took a non-zero
    # column mass here: with mup == 0 the coupling mass is uniform on both
    # grids and the coupled and uncoupled paths agree identically, so
    # everything below would pass either way.
    assert float(parent_state.mup[4, 4]) == pytest.approx(
        footprint_mu, abs=1.0e-4)

    for value, expect in (
            (parent_state.thp[0, 4, 4], (ratio * ratio + 1) / 2.0),
            (parent_state.u[0, 4, 4], (ratio + 1) / 2.0),
            (parent_state.u[0, 4, 5], (3 * ratio + 1) / 2.0),
            (parent_state.v[0, 4, 4], (5 * ratio + 1) / 2.0),
            (parent_state.v[0, 5, 4], (7 * ratio + 1) / 2.0)):
        assert float(value) == pytest.approx(expect, abs=1.0e-6)
    # The uniform child rows: a convex average of a constant is that
    # constant, so these pin that the mass ramp leaked into nothing else.
    assert float(parent_state.w[0, 4, 4]) == pytest.approx(4.0, abs=1.0e-6)
    assert float(parent_state.php[0, 4, 4]) == pytest.approx(8.0, abs=1.0e-6)
    assert float(parent_state.qv[0, 4, 4]) == pytest.approx(
        0.002, abs=1.0e-8)

    # copy_fcn now writes into the parent's live prognostic rather than a
    # scratch buffer, so "only the feedback rectangle" is essential.
    for kind in nest_field_kinds(parent_cfg.run):
        name = {"t": "thp", "ph": "php"}.get(kind, kind)
        value = parent_state.mup if kind == "mu" else getattr(
            parent_state, name)
        after = cp.asnumpy(value)
        reg = coupler.registrations[
            "x" if kind == "u" else "y" if kind == "v" else "m"]
        i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(
            reg, spec_zone=child_cfg.run.spec_zone)
        mask = np.zeros(after.shape[-2:], dtype=bool)
        mask[j_lo:j_hi + 1, i_lo:i_hi + 1] = True
        outside = np.broadcast_to(~mask, after.shape)
        np.testing.assert_array_equal(after[outside], before[kind][outside])


@requires_gpu
@pytest.mark.gpu
def test_feedback_theta_lands_in_the_parents_own_base_frame():
    """WRF restricts ``t_2 = theta - 300``; woof stores ``theta - thb``.

    300 is a global constant (``t0``), so parent and child mean the same
    thing by ``t_2`` and WRF writes the restriction straight across.  woof's
    ``thp`` is measured against a per-grid base profile instead, so the
    restricted value has to change reference frame on arrival.  With the
    parent's ``thb`` 9 K below the child's, the parent must end up at
    ``mean(theta_child) - thb_parent`` = 5 + 300 - 291 = 14 K, not at the
    restricted ``t_2`` of 5 K.
    """
    import cupy as cp

    from woof.core.nest import NestCoupler
    from woof.core.state import DomainState

    ratio = 3
    parent_cfg = _domain(1, 0, nx=14, ny=14)
    child_cfg = _domain(2, 1, nx=9, ny=9, ratio=ratio)
    parent_state = DomainState(parent_cfg.run)
    child_state = DomainState(child_cfg.run)
    _seed_state(parent_state, child=False)
    _seed_state(child_state, child=True)
    _seed_operator_patterns(child_state, ratio)
    parent_state.thb[...] = cp.float32(291.0)     # child keeps 300.0
    before_thp = cp.asnumpy(parent_state.thp).copy()
    clock = SimpleNamespace(ticks=0)
    parent = SimpleNamespace(
        cfg=parent_cfg, state=parent_state, clock=clock)
    child = SimpleNamespace(
        cfg=child_cfg, state=child_state, clock=clock, parent=parent)
    coupler = NestCoupler(child, feedback=1)

    scratch = FeedbackScratch()
    coupler.feedback_prepare(child, scratch)
    coupler.feedback_commit(child)

    restricted_t2 = (ratio * ratio + 1) / 2.0            # = 5.0
    assert float(parent_state.thp[0, 4, 4]) == pytest.approx(
        restricted_t2 + 300.0 - 291.0, abs=1.0e-4)
    # The rebasing is confined to the cells copy_fcn wrote.
    mass_reg = coupler.registrations["m"]
    i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(
        mass_reg, spec_zone=child_cfg.run.spec_zone)
    mask = np.zeros(parent_state.thp.shape[-2:], dtype=bool)
    mask[j_lo:j_hi + 1, i_lo:i_hi + 1] = True
    outside = np.broadcast_to(~mask, parent_state.thp.shape)
    after = cp.asnumpy(parent_state.thp)
    np.testing.assert_array_equal(after[outside], before_thp[outside])


def _feedback_experiment(run_seconds, feedback=1):
    from woof.experiment import (
        ExperimentConfig, ProjectionConfig, VerticalConfig)

    return ExperimentConfig(
        name="feedback-restart-determinism",
        start_time=datetime(2000, 1, 1),
        run_seconds=run_seconds,
        vertical=VerticalConfig((1.0, 0.5, 0.0), 10000.0, 1, 0.2),
        projection=ProjectionConfig(
            "lambert", 35.0, -97.0, 30.0, 60.0, -97.0),
        restart_interval_s=30.0,
        domains=(
            _domain(
                1, 0, nx=14, ny=14, run_seconds=run_seconds,
                root_specified=False),
            _domain(
                2, 1, nx=9, ny=9, ratio=3,
                run_seconds=run_seconds),
        ),
        feedback=feedback)


def _feedback_model(exp):
    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.model import (
        DomainNode, ExperimentState, ModelRuntimeStatus)
    from woof.core.nest import NestCoupler
    from woof.core.state import DomainState

    clock = resolve_clock(exp)
    clocks = clock.clocks()
    root_cfg, child_cfg = exp.domains
    root_state = DomainState(root_cfg.run)
    child_state = DomainState(child_cfg.run)
    _seed_state(root_state, child=False)
    _seed_state(child_state, child=True)
    root_state._nest_restart_classification = "REBUILT"
    child_state._nest_restart_classification = "REBUILT"
    grid = SimpleNamespace()
    root = DomainNode(
        cfg=root_cfg, grid=grid, state=root_state, clock=clocks[1],
        parent=None, children=[], coupler=None)
    child = DomainNode(
        cfg=child_cfg, grid=grid, state=child_state, clock=clocks[2],
        parent=root, children=[], coupler=None)
    root.children.append(child)
    child.coupler = NestCoupler(child, feedback=exp.feedback)
    if exp.feedback == 1:
        # Mirrors woof.core.model's build path: the initialization
        # transaction runs only when feedback is on.
        scratch = FeedbackScratch()
        child.coupler.feedback_prepare(child, scratch)
        child.coupler.feedback_commit(child)
        child.coupler.feedback_finalize(child)
    model = ExperimentState(
        root=root, nodes_by_grid_id=MappingProxyType({1: root, 2: child}),
        schedule=build_schedule(exp, clock), memory_ledger=None,
        experiment_fingerprint="feedback-restart-determinism-v1")
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._io_manager = None
    model._last_checkpoint = None
    return model


def _feedback_state_sha256(model):
    import cupy as cp

    from woof.io.restart import STATE_SERIALIZED_ATTRS

    cp.cuda.Stream.null.synchronize()
    digest = hashlib.sha256()
    for node in model.walk_parent_first():
        digest.update(f"d{node.cfg.grid_id:02d}\n".encode())
        for name in STATE_SERIALIZED_ATTRS:
            value = getattr(node.state, name, None)
            if value is None:
                continue
            host = cp.asnumpy(value)
            digest.update(name.encode() + b"\0")
            digest.update(str(host.shape).encode() + b"\0")
            digest.update(host.dtype.str.encode() + b"\0")
            digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


@requires_gpu
@pytest.mark.gpu
def test_feedback_restart_continuation_is_sha_identical(monkeypatch, tmp_path):
    from woof.core.model import execute_experiment
    from woof.core.preflight import nest_field_kinds
    from woof.io.restart import restore_tree_restart, write_tree_restart

    def deterministic_step(state, cfg, *, refl_10cm_due=False):
        del refl_10cm_due
        for index, kind in enumerate(nest_field_kinds(cfg)):
            name = {"t": "thp", "ph": "php"}.get(kind, kind)
            target = state.mup if kind == "mu" else getattr(state, name)
            target += np.float32(cfg.grid_id * 0.01 + index * 0.001)

    monkeypatch.setattr("woof.core.dycore.step", deterministic_step)
    full_exp = _feedback_experiment(60.0)
    unbroken = _feedback_model(full_exp)
    execute_experiment(
        unbroken, validate_state=False, pool_trim_per_period=False)
    unbroken_sha = _feedback_state_sha256(unbroken)

    first_leg = _feedback_model(_feedback_experiment(30.0))
    execute_experiment(
        first_leg, validate_state=False, pool_trim_per_period=False)
    restart_path = write_tree_restart(
        tmp_path, first_leg, full_exp.start_time + timedelta(seconds=30))

    resumed = _feedback_model(full_exp)
    restore_tree_restart(restart_path, resumed)
    assert resumed.node(2).coupler.valid is False
    # Restore starts a new execution-leg observation (5c54e47921d).
    # These counters are not checkpointed meteorological state.
    assert resumed.node(2).coupler.feedback_count == 0
    execute_experiment(
        resumed, validate_state=False, pool_trim_per_period=False)
    resumed_sha = _feedback_state_sha256(resumed)

    assert resumed_sha == unbroken_sha
    assert resumed.node(2).coupler.feedback_count == 1


#: The ``feedback = 0`` whole-tree state digest, over the tiny
#: two-domain fixture below with the dycore replaced by the same
#: deterministic step the restart test uses.
#:
#: This is not a value invented here.  It is the digest recorded for
#: **both sides** of the feedback lane's ``feedback = 0`` base/tip A/B --
#: pre-feedback base ``e7bf4d88`` and lane tip ``80561c6e`` -- in the
#: internal ledger ``PRODUCT-V11-FEEDBACK-20260730.md``, which is where
#: the changelog's "a ``feedback = 0`` run is byte-identical to the
#: pre-change tree" sentence comes from.  Freezing it here turns a
#: one-off comparison against a tree this release no longer contains
#: into something the release itself can re-run: the pre-change value is
#: the bar, and any later change that perturbs a default-configuration
#: run fails against it.
#:
#: It is a same-arithmetic tripwire, not a portability claim.  If it ever
#: disagrees on different hardware, that is worth knowing on its own
#: terms -- a card-dependent ``feedback = 0`` run would also undermine
#: the dual-run byte comparison this project uses as its corruption
#: detector -- so read a failure as "something moved", and let the A/B
#: assertion in the test say whether feedback is what moved.
FEEDBACK_ZERO_STATE_SHA256 = (
    "727ac476e0ebbf97a89be350a151c593e2b9447cc5d9b14fe0436d5f89e47557")


@requires_gpu
@pytest.mark.gpu
def test_feedback_zero_output_is_pinned_and_costs_nothing(monkeypatch):
    """`feedback = 0` output is pinned, and the bypass is provably free.

    The changelog promises that an in-place upgrade to the feedback
    release leaves a ``feedback = 0`` run byte-identical.  That was
    established by running an archived pre-feedback tree beside the lane
    tip -- a comparison this release cannot repeat, because it does not
    contain the base tree.  What it can do is carry the *number* that
    comparison produced, which is what ``FEEDBACK_ZERO_STATE_SHA256``
    above is, and re-derive it from the shipped code every time:

    1. **The pre-change digest is the bar.** A complete ``feedback = 0``
       run over the committed tiny two-domain fixture must reproduce the
       digest the pre-feedback tree produced.  Any later change that
       perturbs a default-configuration run fails here instead of
       shipping.
    2. **The bypass costs nothing.** The same run with
       ``skip_feedback_path=True`` -- the executor with the feedback
       call path structurally absent, which is what the pre-change tree
       was -- must produce the identical digest, and the coupler must
       record zero transactions.  Feedback machinery that is dormant on
       paper but perturbs state in practice fails here, and this half
       holds regardless of hardware.
    """

    from woof.core.model import execute_experiment
    from woof.core.preflight import nest_field_kinds

    def deterministic_step(state, cfg, *, refl_10cm_due=False):
        del refl_10cm_due
        for index, kind in enumerate(nest_field_kinds(cfg)):
            name = {"t": "thp", "ph": "php"}.get(kind, kind)
            target = state.mup if kind == "mu" else getattr(state, name)
            target += np.float32(cfg.grid_id * 0.01 + index * 0.001)

    monkeypatch.setattr("woof.core.dycore.step", deterministic_step)

    exp = _feedback_experiment(60.0, feedback=0)
    assert exp.feedback == 0

    dormant = _feedback_model(exp)
    assert dormant.node(2).coupler.feedback == 0
    execute_experiment(
        dormant, validate_state=False, pool_trim_per_period=False)
    dormant_sha = _feedback_state_sha256(dormant)
    # Dispatched at every table position, and a no-op at every one.
    assert dormant.node(2).coupler.feedback_count == 0

    bypassed = _feedback_model(exp)
    execute_experiment(
        bypassed, validate_state=False, pool_trim_per_period=False,
        skip_feedback_path=True)
    bypassed_sha = _feedback_state_sha256(bypassed)
    assert bypassed.node(2).coupler.feedback_count == 0

    assert dormant_sha == bypassed_sha, (
        "feedback = 0 is not free: walking the dormant three-phase "
        "transaction changed the run's state relative to an executor "
        "with the feedback call path removed")
    assert dormant_sha == FEEDBACK_ZERO_STATE_SHA256, (
        "the feedback = 0 output of this tree moved; if the change was "
        "intended, re-freeze FEEDBACK_ZERO_STATE_SHA256 and say so in "
        "the changelog, because it is an in-place-upgrade promise")


# ---------------------------------------------------------------------------
# V-6: feedback = 1 is a legal config value with real preconditions --
# say what they are before the 26-second build, not after
# ---------------------------------------------------------------------------


def test_check_advises_on_two_way_before_the_tree_refuses_it():
    """`woof check` passed feedback=1 in silence, identically to fb0.

    A test host 7 validation run authored a matched A/B pair differing only
    in the feedback value, ran `woof check` on the two-way half, got a
    clean PASS and exit 0, and learned only at prepare -- after a 26 s
    hierarchy build -- that the prepared-hierarchy route refused two-way
    nesting outright.  That ROUTE refusal is now lifted; the advisory
    survives because what still refuses is the TREE, and the coupler's
    three named preconditions are exactly what an author needs before
    paying for the ingest.  This is an advisory and changes no exit code.
    """

    from types import SimpleNamespace

    from woof.core.preflight import (
        FEEDBACK_TWO_WAY_ADVISORY, feedback_advisory,
    )

    assert feedback_advisory(SimpleNamespace(feedback=0)) is None
    assert feedback_advisory(SimpleNamespace(feedback=1)) \
        == FEEDBACK_TWO_WAY_ADVISORY
    # Both routes run it, so neither may be named as the one that cannot.
    assert "woof run" in FEEDBACK_TWO_WAY_ADVISORY
    assert "BOTH routes" in FEEDBACK_TWO_WAY_ADVISORY
    assert "refuses it at preparation" not in FEEDBACK_TWO_WAY_ADVISORY
    assert "experimental" in FEEDBACK_TWO_WAY_ADVISORY
    # ...and it names all three shapes the coupler refuses at build,
    # which are the reasons a legal feedback = 1 config still stops.
    for precondition in ("vertical level counts", "microphysics",
                         "field inventories"):
        assert precondition in FEEDBACK_TWO_WAY_ADVISORY


def test_the_hierarchy_stamps_the_experiments_own_feedback():
    """The authoring refusal is LIFTED: artifacts are coupling-agnostic.

    An initial state, sealed statics and a boundary series are
    byte-identical however the tree couples at run time, so the topology
    now stamps the experiment's feedback setting instead of refusing
    everything but 0.  What remains refused is an out-of-range value.
    """

    import pytest

    from types import SimpleNamespace

    from woof.source_hierarchy import _validated_static_one_way_topology

    parent = _domain(1, 0, nx=9, ny=9)
    child = _domain(2, 1, nx=9, ny=9, ratio=3)
    for feedback in (0, 1):
        experiment = SimpleNamespace(
            feedback=feedback, domains=(parent, child),
            start_time=datetime(2026, 1, 1))
        topology = _validated_static_one_way_topology(
            experiment, grids=(object(), object()))
        assert topology["feedback"] == feedback
        assert topology["status"] == "PASS"
    experiment = SimpleNamespace(
        feedback=3, domains=(parent, child),
        start_time=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="feedback must be 0 or 1"):
        _validated_static_one_way_topology(
            experiment, grids=(object(), object()))
