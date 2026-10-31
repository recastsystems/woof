"""CPU geometry, schedule, inventory, and assembly pins for N2a/b/c."""

from __future__ import annotations

import dataclasses
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core.clock import Schedule, build_schedule, resolve_clock
from woof.core.model import execute_experiment
from woof.static.lambert import grids_from_projection_config
from woof.verify.cases.nest_ideal_common import (
    IDENTITY_COMPONENTS, IDENTITY_SINT_HALO, RatioOneIdentityCoupler,
    assemble_idealized_tree, consume_history_reflectivity,
    davies_tendency_oracle, force_table_oracle, identity_halo_run,
    identity_oracle_verdict, prepare_idealized_domain,
    raw_specified_row_guardrail, synchronized_identity_config)
from woof.verify.cases.nest_ideal_r1_moist import (
    MOIST_PROGNOSTIC_FIELDS, load_scaffold as load_identity)
from woof.verify.cases.nest_ideal_r3 import (
    CROSSING_WINDOW_S, INTERFACE_METRIC, INTERIOR_FIELDS, INTERIOR_METRIC,
    adjudicate, load_scaffold as load_r3, placement_evidence,
    uniform_scaffold)
from woof.verify.cases.nest_null_r1 import (
    DRY_PROGNOSTIC_FIELDS, build_dry_wk82)
from woof.verify.metrics import boundary_zone_blowup
from woof.verify.nest_gates import gate
from woof.verify.npref import np_specified_relaxation
from woof.ingest.lateral_bc import FieldBoundary, SideBoundary


def test_n2a_n2b_config_schedule_and_field_inventories_are_pinned():
    n2a = load_identity(variant="n2a")
    n2b = load_identity(variant="n2b")
    for exp in (n2a, n2b):
        root, child = exp.domains
        assert (root.run.nx, root.run.ny, root.run.nz) == (48, 48, 30)
        assert (child.run.nx, child.run.ny, child.run.nz) == (48, 48, 30)
        assert (child.parent_grid_ratio,
                child.parent_time_step_ratio) == (1, 1)
        halo = identity_halo_run(exp)
        assert (halo.nx, halo.ny) == (
            root.run.nx + 2 * IDENTITY_SINT_HALO,
            root.run.ny + 2 * IDENTITY_SINT_HALO)
        schedule = build_schedule(exp, resolve_clock(exp))
        assert Schedule.op_counts(schedule.interior_period) == {
            "STEP": 2, "FORCE": 1, "FEEDBACK": 1}
        assert all((dc.run.ra_physics, dc.run.sf_sfclay_physics,
                    dc.run.sf_surface_physics, dc.run.bl_pbl_physics,
                    dc.run.cu_physics) == (0, 0, 0, 0, 0)
                   for dc in exp.domains)

    assert tuple(DRY_PROGNOSTIC_FIELDS) == (
        "u", "v", "w", "thp", "php", "mup")
    assert tuple(MOIST_PROGNOSTIC_FIELDS) == (
        "u", "v", "w", "thp", "php", "mup",
        "qv", "qc", "qr", "qi", "qs", "qg",
        "qnr", "qni", "qns", "qng", "h_diabatic")
    assert gate("N2", "null_nest_r1_dry_restriction").kind == \
        "identity_oracle"
    assert gate("N2", "identity_nest_r1_moist_wk82").kind == \
        "identity_oracle"


def test_n2_identity_scaffold_admits_complete_no_pbl_operator():
    """Both ratio-1 domains inherit the complete WRF PBL-off mixing path."""
    from woof.config import validate_run_config

    exp = load_identity(variant="n2b")
    for domain in exp.domains:
        assert validate_run_config(domain.run).bl_pbl_physics == 0


def test_identity_executable_history_cadence_covers_every_sync_tick():
    exp = synchronized_identity_config(load_identity(variant="n2b"))
    assert all(dc.history_interval_s == exp.root.run.dt
               for dc in exp.domains)
    schedule = build_schedule(exp, resolve_clock(exp))
    assert schedule.clock.run_ticks // schedule.clock.root.step_ticks == 10


def test_idealized_prepare_fills_eos_and_attaches_distinct_mp_drivers(
        monkeypatch):
    import woof.core.diagnostics as diagnostics
    import woof.core.physics as physics

    events = []
    monkeypatch.setattr(
        diagnostics, "update_diagnostics",
        lambda state, opt: events.append(("eos", state.name, opt)))

    def initialize(state, cfg, **kwargs):
        driver = SimpleNamespace(domain=cfg.grid_id, kwargs=kwargs)
        state.physics = driver
        events.append(("physics", state.name, cfg.grid_id))
        return driver

    monkeypatch.setattr(physics, "initialize_physics", initialize)
    grid = SimpleNamespace(latlon_mass=lambda: (
        np.array([[35.0]]), np.array([[-97.0]])))

    dry = load_identity(variant="n2a").root
    dry_state = SimpleNamespace(name="dry", physics=None)
    assert prepare_idealized_domain(
        dry_state, dry, grid, load_identity().start_time) is None
    assert dry_state.physics is None

    moist = load_identity(variant="n2b")
    states = []
    for dc in moist.domains:
        state = SimpleNamespace(name=f"d{dc.grid_id:02d}", physics=None)
        driver = prepare_idealized_domain(
            state, dc, grid, moist.start_time)
        assert driver is state.physics
        assert driver.domain == dc.grid_id
        assert driver.kwargs["radiation_start_time"] == moist.start_time
        np.testing.assert_array_equal(
            driver.kwargs["radiation_latitude"], [[35.0]])
        np.testing.assert_array_equal(
            driver.kwargs["radiation_longitude"], [[-97.0]])
        states.append(state)
    assert states[0].physics is not states[1].physics
    assert events == [
        ("eos", "dry", dry.run.hypsometric_opt),
        ("eos", "d01", moist.root.run.hypsometric_opt),
        ("physics", "d01", 1),
        ("eos", "d02", moist.domains[1].run.hypsometric_opt),
        ("physics", "d02", 2),
    ]


def test_microphysics_only_driver_initializes_with_cpu_array_backend(
        monkeypatch):
    from woof.core.preflight import physics_array_shapes
    import woof.core.physics as physics
    import woof.core.state as state_mod

    cfg = dataclasses.replace(
        load_identity(variant="n2b").root.run,
        nx=3, ny=2, nz=4)
    assert not physics.physics_enabled(cfg)
    assert physics.physics_driver_required(cfg)
    assert physics_array_shapes(cfg)["fields/psfc"] == (cfg.ny, cfg.nx)
    monkeypatch.setattr(state_mod, "cp", np)
    monkeypatch.setattr(state_mod, "DTYPE", np.float32)
    monkeypatch.setattr(physics, "cp", np)
    monkeypatch.setattr(physics, "DTYPE", np.float32)
    state = state_mod.DomainState(cfg)
    driver = physics.initialize_physics(state, cfg)
    assert state.physics is driver
    assert driver.mp_physics == 10
    assert driver.refl_10cm is None
    assert driver.microphysics.rainnc is state._scratch["mp_rainnc"]


def test_dry_wk82_builder_populates_positive_cpu_mirror_diagnostics(
        monkeypatch):
    import woof.core.diagnostics as diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    import woof.core.state as state_mod
    from woof.verify.cases import wk82
    from woof.verify.npref import np_calc_p_alpha

    cfg = dataclasses.replace(
        load_identity(variant="n2a").root.run,
        nx=4, ny=3, nz=8)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: wk82.wk82_sounding(z)[0],
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    monkeypatch.setattr(state_mod, "cp", np)
    monkeypatch.setattr(state_mod, "DTYPE", np.float32)
    monkeypatch.setitem(
        sys.modules, "cupy", SimpleNamespace(asarray=np.asarray))

    def diagnose(state, hypsometric_opt=1):
        p, al, alt = np_calc_p_alpha(
            state.thp, state.php, state.mup, base, coord,
            hypsometric_opt=hypsometric_opt)
        state.p[...] = p
        state.al[...] = al
        state.alt[...] = alt

    monkeypatch.setattr(diagnostics, "update_diagnostics", diagnose)
    state = build_dry_wk82(cfg, coord, base)
    assert np.all(np.isfinite(state.p)) and np.all(state.p > 0.0)
    assert np.all(np.isfinite(state.alt)) and np.all(state.alt > 0.0)


def test_verification_history_consumes_post_initial_mp_reflectivity(
        monkeypatch):
    consumed = []
    monkeypatch.setattr(
        "woof.core.refl.consume_refl_10cm",
        lambda state: consumed.append(state))
    moist_dc = load_identity(variant="n2b").root
    moist_state = SimpleNamespace(physics=SimpleNamespace(mp_physics=10))
    moist = SimpleNamespace(cfg=moist_dc, state=moist_state)
    consume_history_reflectivity(moist, 0)
    consume_history_reflectivity(moist, 6)
    assert consumed == [moist_state]

    dry_dc = load_identity(variant="n2a").root
    consume_history_reflectivity(
        SimpleNamespace(cfg=dry_dc, state=SimpleNamespace(physics=None)), 6)
    assert consumed == [moist_state]
    with pytest.raises(RuntimeError, match="microphysics but no driver"):
        consume_history_reflectivity(
            SimpleNamespace(
                cfg=moist_dc, state=SimpleNamespace(physics=None)), 6)


def test_identity_side_tables_match_packed_bdy_interp1_shapes():
    value = np.arange(3 * 7 * 11, dtype=np.float32).reshape(3, 7, 11)
    tendency = -value
    width = 5

    # _sides returns the mapping handed unchanged to attach_nest_boundaries.
    sides = RatioOneIdentityCoupler._sides(value, tendency, width)
    expected_shapes = {
        "west": (3, 7, width),
        "east": (3, 7, width),
        "south": (3, width, 11),
        "north": (3, width, 11),
    }
    for side, (side_value, side_tendency) in sides.items():
        assert side_value.shape == expected_shapes[side]
        assert side_tendency.shape == expected_shapes[side]
        assert side_value.flags.c_contiguous
        assert side_tendency.flags.c_contiguous


def _synthetic_force_inputs():
    child = (np.arange(2 * 9 * 10, dtype=np.float32)
             .reshape(2, 9, 10) * np.float32(0.25))
    parent = np.add(child, np.float32(0.5), dtype=np.float32)
    dt = np.float32(2.0)
    tendency = np.asarray(
        np.subtract(parent, child, dtype=np.float32).astype(np.float64)
        / np.float64(dt), dtype=np.float32)
    tables = {"u": RatioOneIdentityCoupler._sides(child, tendency, 5)}
    return {"u": parent}, {"u": child}, tables, dt


def _replace_force_side(tables, side, value, tendency):
    changed = {name: dict(sides) for name, sides in tables.items()}
    changed["u"][side] = (
        np.ascontiguousarray(value), np.ascontiguousarray(tendency))
    return changed


def _synthetic_davies_inputs():
    current = np.zeros((1, 9, 10), dtype=np.float32)
    target = (np.arange(current.size, dtype=np.float32)
              .reshape(current.shape) * np.float32(0.125))
    zero = np.zeros_like(target)
    sides = RatioOneIdentityCoupler._sides(target, zero, 5)
    boundary = FieldBoundary(**{
        name: SideBoundary(value, tendency)
        for name, (value, tendency) in sides.items()
    })
    expected = np_specified_relaxation(
        current, np.zeros_like(current), boundary,
        dtbc=2.0, dt=2.0, spec_zone=1, relax_zone=4,
        spec_exp=0.0, apply_relax=True).astype(np.float32)
    return current, boundary, expected


def test_f19_identity_oracle_synthetic_pass_cases_and_component_and():
    parent, child, tables, dt = _synthetic_force_inputs()
    force = force_table_oracle(
        parent, child, tables, parent_dt_fp32=dt)
    assert force["pass"]
    assert force["fields"]["u"]["y_corner_ownership_pass"]
    assert force["fields"]["u"]["east_distance_reversal_pass"]
    assert force["fields"]["u"]["north_distance_reversal_pass"]

    current, boundary, expected = _synthetic_davies_inputs()
    davies = davies_tendency_oracle(
        field_name="u", current=current, applied=expected,
        boundary=boundary, dtbc=2.0, dt=2.0,
        spec_zone=1, relax_zone=4, apply_relax=True)
    assert davies["pass"]

    components = {name: {"pass": True} for name in IDENTITY_COMPONENTS}
    assert identity_oracle_verdict(components)
    components["force_table_oracle"]["pass"] = False
    assert not identity_oracle_verdict(components)


def test_f19_force_table_orientation_flip_fails():
    parent, child, tables, dt = _synthetic_force_inputs()
    value, tendency = tables["u"]["east"]
    flipped = _replace_force_side(
        tables, "east", value[..., ::-1], tendency[..., ::-1])
    report = force_table_oracle(
        parent, child, flipped, parent_dt_fp32=dt)
    assert not report["pass"]
    assert not report["fields"]["u"]["east_distance_reversal_pass"]


def test_f19_force_table_wrong_parent_denominator_fails():
    parent, child, tables, dt = _synthetic_force_inputs()
    wrong = np.asarray(
        np.subtract(parent["u"], child["u"], dtype=np.float32)
        .astype(np.float64) / np.float64(3.0), dtype=np.float32)
    changed = {"u": RatioOneIdentityCoupler._sides(
        child["u"], wrong, 5)}
    report = force_table_oracle(
        parent, child, changed, parent_dt_fp32=dt)
    assert not report["pass"]
    assert any(side["tendency_bit_mismatches"] > 0
               for side in report["fields"]["u"]["sides"].values())


def test_f19_force_table_stale_successor_fails():
    parent, child, tables, dt = _synthetic_force_inputs()
    value, tendency = tables["u"]["west"]
    stale = _replace_force_side(
        tables, "west", value,
        np.add(tendency, np.float32(1.0), dtype=np.float32))
    report = force_table_oracle(
        parent, child, stale, parent_dt_fp32=dt)
    assert not report["pass"]
    assert not report["fields"]["u"]["sides"]["west"]["successor_pass"]


def test_f19_sign_flipped_davies_relaxation_fails():
    current, boundary, expected = _synthetic_davies_inputs()
    report = davies_tendency_oracle(
        field_name="u", current=current,
        applied=np.negative(expected, dtype=np.float32),
        boundary=boundary, dtbc=2.0, dt=2.0,
        spec_zone=1, relax_zone=4, apply_relax=True)
    assert not report["pass"]
    assert not report["formula_pass"]


def test_f19_raw_specified_cap_breach_fails():
    reference = SimpleNamespace(u=np.zeros((1, 7, 8), dtype=np.float32))
    candidate = SimpleNamespace(u=reference.u.copy())
    candidate.u[:, 0, 3] = np.float32(9.0e-6)
    report = raw_specified_row_guardrail(
        candidate, reference, active_caps={"u": 8.0e-6},
        inactive_fields=(), spec_zone=1)
    assert not report["pass"]
    assert not report["active_caps"]["u"]["pass"]


def test_f19_inactive_species_perturbation_fails_bit_identity():
    reference = SimpleNamespace(
        u=np.zeros((1, 7, 8), dtype=np.float32),
        qc=np.zeros((1, 7, 8), dtype=np.float32))
    candidate = SimpleNamespace(u=reference.u.copy(), qc=reference.qc.copy())
    candidate.qc[:, -1, 2] = np.float32(1.0e-12)
    report = raw_specified_row_guardrail(
        candidate, reference, active_caps={"u": 8.0e-6},
        inactive_fields=("qc",), spec_zone=1)
    assert not report["pass"]
    assert not report["inactive_bit_identity"]["qc"]["pass"]


def test_n2c_ratio3_geometry_schedule_and_storm_crossing_placement():
    exp = load_r3()
    root, child = exp.domains
    assert (root.run.nx, root.run.ny, root.run.nz,
            root.run.dx, root.run.dt) == (168, 168, 60, 1000.0, 6.0)
    assert (child.run.nx, child.run.ny, child.run.nz) == (120, 120, 60)
    assert child.run.dx == pytest.approx(1000.0 / 3.0)
    assert child.run.dt == 2.0
    assert (root.run.time_step_sound, child.run.time_step_sound) == (6, 6)
    assert (root.run.w_damping, child.run.w_damping) == (1, 1)
    assert (root.run.damp_opt, child.run.damp_opt) == (3, 3)
    assert (root.run.zdamp, child.run.zdamp) == (5000.0, 5000.0)
    assert (root.run.dampcoef, child.run.dampcoef) == (0.2, 0.2)
    assert (root.run.diff_6th_opt, child.run.diff_6th_opt) == (2, 2)
    assert (root.run.diff_6th_factor,
            child.run.diff_6th_factor) == (0.12, 0.06)
    assert (child.parent_grid_ratio,
            child.parent_time_step_ratio) == (3, 3)
    assert (child.i_parent_start, child.j_parent_start) == (85, 45)
    assert child.run.nx % 3 == child.run.ny % 3 == 0
    assert all((dc.run.ra_physics, dc.run.sf_sfclay_physics,
                dc.run.sf_surface_physics, dc.run.bl_pbl_physics,
                dc.run.cu_physics) == (0, 0, 0, 0, 0)
               for dc in exp.domains)
    # The southeast-quadrant child interfaces bracket the release point.
    west_parent_i = ((child.i_parent_start - 0.5) + 0.5 / 3.0)
    north_parent_j = ((child.j_parent_start - 0.5)
                      + (child.run.ny - 0.5) / 3.0)
    assert (west_parent_i - (root.run.nx + 1) / 2.0) * root.run.dx \
        == pytest.approx(166.66666666666666)
    assert (north_parent_j - (root.run.ny + 1) / 2.0) * root.run.dy \
        == pytest.approx(-166.66666666666666)

    schedule = build_schedule(exp, resolve_clock(exp))
    assert Schedule.op_counts(schedule.interior_period) == {
        "STEP": 4, "FORCE": 1, "FEEDBACK": 1}
    assert tuple(INTERIOR_FIELDS) == (
        "w", "thp", "qc", "qr", "qi", "qs", "qg")
    assert CROSSING_WINDOW_S == (0.0, 7200.0)

    placement = placement_evidence(exp)
    np.testing.assert_allclose(
        placement["shear_0_6km_uv_ms"], [29.5583, 8.4419], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        placement["right_normal_uv_ms"], [8.4419, -29.5583],
        rtol=0, atol=1e-12)
    assert placement["west_interface_from_release_km"] == pytest.approx(1/6)
    assert placement["north_interface_from_release_km"] == pytest.approx(-1/6)


def test_n2c_uniform_companion_is_full_footprint_and_colocated():
    nested = load_r3()
    uniform = uniform_scaffold()
    assert len(uniform.domains) == 1
    run = uniform.root.run
    assert (run.nx, run.ny, run.nz) == (504, 504, 60)
    assert run.dx == pytest.approx(nested.root.run.dx / 3.0)
    assert run.dt == nested.domains[1].run.dt == 2.0
    assert run.time_step_sound == 6
    assert run.w_damping == 1 and run.damp_opt == 3
    assert run.diff_6th_opt == 2 and run.diff_6th_factor == 0.06
    assert run.nx * run.dx == pytest.approx(
        nested.root.run.nx * nested.root.run.dx)


def test_n2c_imports_exact_frozen_boundary_blowup_predicate():
    assert not boundary_zone_blowup(5.0, 0.5)
    assert boundary_zone_blowup(5.0001, 0.5)
    assert not boundary_zone_blowup(10.0, 2.0)
    assert boundary_zone_blowup(float("nan"), 20.0)
    record = gate("N2", "wk_r3_boundary_zone_blowup")
    assert record.kind == "max" and record.threshold is not None


@pytest.mark.parametrize("interior", [
    float("nan"), float("inf"),
], ids=["nan-interior", "inf-interior"])
@pytest.mark.parametrize("boundary", [1.0e9, 500.0, 40.0])
def test_boundary_blowup_fires_when_its_own_yardstick_is_unmeasurable(
        boundary, interior):
    """Negative control: an unmeasurable interior must not pass the gate.

    This bound normalises the child's boundary against the same run's free
    interior, so the interior maximum IS the yardstick.  ``max(nan, 1.0)``
    is ``nan`` and ``x > nan`` is False, so before the fix a NaN or +Inf
    interior maximum switched the only boundary-reflection detector in the
    nest ladder off completely: a boundary vertical velocity of 1e9 m/s
    scored ``False`` -- did not fire -- and the gate reported PASS.
    """
    assert boundary_zone_blowup(boundary, interior)


def test_boundary_blowup_still_accepts_an_ordinary_quiet_run():
    """The added leg is monotone: nothing finite and healthy changed."""
    for boundary, interior in ((0.0, 0.0), (3.0, 2.0), (5.0, 1.0),
                               (10.0, 2.0), (25.0, 5.0)):
        assert not boundary_zone_blowup(boundary, interior)


def test_n2c_structural_verdict_is_explicit_and_missing_is_failure(tmp_path):
    statistics = {
        name: {"finite": True, "mean": 0.0, "stddev": 0.0, "rms": 0.0,
               "minimum": 0.0, "maximum": 0.0, "p01": 0.0,
               "p50": 0.0, "p99": 0.0, "nonzero_fraction": 0.0}
        for name in INTERIOR_FIELDS
    }
    nested = {
        "gates": {
            "wk_r3_boundary_zone_blowup": dataclasses.asdict(
                gate("N2", "wk_r3_boundary_zone_blowup")),
            INTERFACE_METRIC: dataclasses.asdict(gate("N2", INTERFACE_METRIC)),
            INTERIOR_METRIC: dataclasses.asdict(gate("N2", INTERIOR_METRIC)),
        },
        "boundary_zone_series": [{"pass": True}],
        "interface_boundary_vs_interior_series": [
            {"boundary_max_abs_w_ms": 0.0,
             "interior_max_abs_w_ms": 0.0}],
        "boundary_pass": True,
        "interface_cross_sections": {
            "path": "interface_cross_sections.npz", "expected_arrays": 2,
            "arrays": {
                "t_00000000_child_north_interface_w": [61, 120],
                "t_00000000_child_west_interface_w": [61, 120]}},
        "nested_interior_statistics": statistics,
    }
    uniform = {"uniform_restricted_interior_statistics": statistics}
    (tmp_path / "nested_report.json").write_text(json.dumps(nested), "utf-8")
    (tmp_path / "uniform_report.json").write_text(json.dumps(uniform), "utf-8")

    pending = adjudicate(
        tmp_path, interface_verdict=None, interior_verdict=None)
    assert not pending["pass"]
    assert pending["interface_crossing"]["verdict"] is None
    accepted = adjudicate(
        tmp_path, interface_verdict="pass", interior_verdict="pass")
    assert accepted["pass"]


class _MockCoupler:
    def __init__(self, child):
        self.child = child
        self.valid = False

    def force(self, node):
        assert node is self.child
        self.valid = True

    def feedback_prepare(self, node, out):
        out.payload = None

    def feedback_commit(self, node, *, start=False):
        pass

    def feedback_finalize(self, node, *, start=False):
        pass


def test_shared_idealized_assembly_executes_with_tiny_mock_states(monkeypatch):
    exp = synchronized_identity_config(load_identity(variant="n2a"))
    grids = grids_from_projection_config(exp)
    root_state = SimpleNamespace(elapsed_seconds=0.0)
    child_state = SimpleNamespace(elapsed_seconds=0.0)

    def initialize(_dc, _parent):
        return SimpleNamespace(state=child_state, grid=grids[1])

    prepared = []

    def prepare(state, dc, grid, start_time):
        prepared.append((state, dc.grid_id, grid, start_time))

    model = assemble_idealized_tree(
        exp, root_state, grids=grids, child_initializer=initialize,
        coupler_factory=_MockCoupler, domain_preparer=prepare)
    assert prepared == [
        (root_state, 1, grids[0], exp.start_time),
        (child_state, 2, grids[1], exp.start_time),
    ]
    calls = []
    monkeypatch.setattr(
        "woof.core.dycore.step",
        lambda state, cfg, **_kwargs: calls.append(cfg.grid_id))
    report = execute_experiment(model, validate_state=False)
    assert (report.steps, report.forces, report.feedback_calls) == (20, 10, 10)
    assert calls == [1, 2] * 10
    assert model.node(2).parent is model.root


def test_a_masked_device_does_not_abort_the_host_step_boundary(monkeypatch):
    """An unreachable CuPy pool skips the trim instead of killing the run.

    ``_trim_default_pool`` runs at every step boundary of
    ``execute_experiment``.  With CuPy INSTALLED and every device masked
    -- ``CUDA_VISIBLE_DEVICES=-1``, which tests/conftest.py plants for the
    whole session -- its first pool call answers ``cudaErrorNoDevice``,
    and before the guard that aborted the host-only assembly above with a
    CUDARuntimeError from a run that never needed a device.

    The pool is stubbed rather than probed so the gate is the same on a
    box with no CuPy at all, and so it pins the exact attribute the guard
    catches on: a rename to some other exception class would pass a
    ``try/except Exception`` and fails here.
    """

    import sys
    import types

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    class _CUDARuntimeError(Exception):
        pass

    tried = []

    def _free_all_blocks():
        tried.append("pool reached")
        raise _CUDARuntimeError(
            "cudaErrorNoDevice: no CUDA-capable device is detected")

    fake = types.ModuleType("cupy")
    fake.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(CUDARuntimeError=_CUDARuntimeError))
    fake.get_default_memory_pool = lambda: types.SimpleNamespace(
        free_all_blocks=_free_all_blocks)
    monkeypatch.setitem(sys.modules, "cupy", fake)

    from woof.core.model import _trim_default_pool

    _trim_default_pool()
    assert tried == ["pool reached"], (
        "the guard returned without attempting the trim, so it would also "
        "skip it on a healthy device")
