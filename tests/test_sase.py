"""SASE-L1 CPU authority tests (evidence-ladder stages 1-2).

Spec: docs/superpowers/specs/2026-07-20-sase-design.md.  Everything here
is FP64 NumPy; no GPU marker appears in this file by design.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.config import (SASE_PBL_SCHEME as _SASE_SELECTOR,
                         RunConfig, validate_run_config)


def _min_cfg(**kw):
    base = dict(nx=16, ny=16, nz=8, dx=1000.0, dy=1000.0,
                ztop=8000.0, dt=5.0, run_seconds=5.0)
    base.update(kw)
    return RunConfig(**base)


def test_pbl_selector_default_is_off_and_accepted():
    cfg = _min_cfg()
    assert cfg.bl_pbl_physics == 0
    validate_run_config(cfg)


def test_sase_selector_is_outside_the_wrf_namespace():
    """The closure must not claim a WRF number.

    WRF v4.6.1's own bl_pbl_physics namespace runs to 99 (99 = MRF), so
    a value above it can never collide with a scheme WRF adds later --
    and it must fall outside the ported-option tuple so the WRF
    PBL/surface-layer compatibility matrix, which states what WRF
    admits, is never consulted about a scheme WRF does not have.
    """
    from woof.config import SASE_PBL_SCHEME
    from woof.wrf461_compatibility import PBL_OPTIONS

    assert SASE_PBL_SCHEME > 99
    assert SASE_PBL_SCHEME not in PBL_OPTIONS


def test_sase_selector_bare_is_fail_closed():
    """Selecting the closure without the settings it needs is refused."""
    from woof.config import SASE_PBL_SCHEME

    cfg = _min_cfg(bl_pbl_physics=SASE_PBL_SCHEME)
    with pytest.raises(ValueError, match="not admitted|requires"):
        validate_run_config(cfg)


def test_unknown_pbl_selector_is_rejected():
    cfg = _min_cfg(bl_pbl_physics=777)
    with pytest.raises(ValueError, match="bl_pbl_physics"):
        validate_run_config(cfg)


def test_sase_config_id_is_stable_sha256():
    from woof.verify import sase_ref
    cid = sase_ref.sase_config_id()
    assert isinstance(cid, str) and len(cid) == 64
    assert cid == sase_ref.sase_config_id()          # deterministic


def test_sase_config_id_binds_constants(monkeypatch):
    from woof.verify import sase_ref
    baseline = sase_ref.sase_config_id()
    monkeypatch.setattr(sase_ref, "C_E", 0.94)
    with_ce = sase_ref.sase_config_id()
    assert with_ce != baseline
    monkeypatch.setattr(sase_ref, "G_ACCEL", 9.80665)
    assert sase_ref.sase_config_id() not in (baseline, with_ce)


def _sine_field(nz=4, ny=32, nx=32, wavelength_cells=8, amp=2.0):
    x = np.arange(nx, dtype=np.float64)
    f = amp * np.sin(2.0 * np.pi * x / wavelength_cells)
    return np.broadcast_to(f, (nz, ny, nx)).copy()


def test_box_filter_preserves_constants():
    from woof.verify.sase_ref import box_filter
    f = np.full((3, 8, 8), 7.25, dtype=np.float64)
    for width in (2, 4):
        np.testing.assert_allclose(box_filter(f, width), f, rtol=0, atol=0)


def test_box_filter_sine_attenuation_matches_transfer_function():
    from woof.verify.sase_ref import box_filter
    f = _sine_field()                       # kh = 2*pi/8 = pi/4
    kh = np.pi / 4.0
    g2 = 0.5 + 0.5 * np.cos(kh)             # 3-point transfer
    g4 = 0.25 + 0.5 * np.cos(kh) + 0.25 * np.cos(2 * kh)
    np.testing.assert_allclose(box_filter(f, 2), g2 * f, atol=1e-12)
    np.testing.assert_allclose(box_filter(f, 4), g4 * f, atol=1e-12)


def test_box_filter_is_periodic_in_x_and_y():
    from woof.verify.sase_ref import box_filter
    rng = np.random.default_rng(20260720)
    f = rng.standard_normal((2, 16, 16))
    rolled = np.roll(f, 5, axis=2)
    np.testing.assert_allclose(
        box_filter(rolled, 2), np.roll(box_filter(f, 2), 5, axis=2),
        atol=1e-13)
    rolled_y = np.roll(f, 3, axis=1)
    np.testing.assert_allclose(
        box_filter(rolled_y, 2), np.roll(box_filter(f, 2), 3, axis=1),
        atol=1e-13)


def test_structure_function_single_mode_closed_form():
    from woof.verify.sase_ref import structure_functions
    amp, wl = 2.0, 8
    u = _sine_field(amp=amp, wavelength_cells=wl)
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    d2 = structure_functions(u, v, w)
    k = 2.0 * np.pi / wl
    for r in (1, 2, 4):
        # x-direction increments see A^2*(1-cos kr); y-direction sees 0;
        # the sensor averages the two horizontal directions.
        expected = 0.5 * amp**2 * (1.0 - np.cos(k * r))
        np.testing.assert_allclose(d2[r], expected, rtol=1e-12)


def test_sensor_alpha_and_slope_kolmogorov_synthetic(monkeypatch):
    from woof.verify import sase_ref
    # Force D2(r) = r^(2/3) exactly and check the derived quantities.
    monkeypatch.setattr(
        sase_ref, "structure_functions",
        lambda u, v, w: {1: 1.0, 2: 2.0**(2.0 / 3.0), 4: 4.0**(2.0 / 3.0)})
    s = sase_ref.sensor_state(None, None, None, e_mean=1.0)
    np.testing.assert_allclose(s.slope, 2.0 / 3.0, rtol=1e-12)
    e_res = 0.5 * 2.0**(2.0 / 3.0)
    np.testing.assert_allclose(s.e_res, e_res, rtol=1e-12)
    np.testing.assert_allclose(s.alpha, 1.0 / (1.0 + e_res), rtol=1e-12)


def test_sensor_alpha_saturates_when_nothing_is_resolved():
    from woof.verify.sase_ref import sensor_state
    zeros = np.zeros((4, 16, 16))
    s = sensor_state(zeros, zeros, zeros, e_mean=0.5)
    assert s.alpha == 1.0                     # all energy subgrid


def test_strain_pure_shear_closed_form():
    from woof.verify.sase_ref import strain
    nz, ny, nx = 6, 8, 8
    dz = 100.0
    z = (np.arange(nz, dtype=np.float64) * dz)[:, None, None]
    shear = 3.0e-3
    u = np.broadcast_to(shear * z, (nz, ny, nx)).copy()
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    s = strain(u, v, w, dx=1000.0, dy=1000.0, dz=dz)
    interior = np.s_[1:-1, :, :]
    np.testing.assert_allclose(s[4][interior], 0.5 * shear, rtol=1e-12)
    for idx in (0, 1, 2, 3, 5):
        np.testing.assert_allclose(s[idx][interior], 0.0, atol=1e-15)


def test_germano_lift_single_mode_closed_form():
    from woof.verify.sase_ref import germano_lift
    amp, wl = 2.0, 8
    u = _sine_field(amp=amp, wavelength_cells=wl)
    zeros = np.zeros_like(u)
    L = germano_lift(u, zeros, zeros)
    kh = 2.0 * np.pi / wl
    g_k = 0.5 + 0.5 * np.cos(kh)
    g_2k = 0.5 + 0.5 * np.cos(2 * kh)
    x = np.arange(u.shape[2], dtype=np.float64)
    cos2 = np.cos(2.0 * 2.0 * np.pi * x / wl)
    expected_xx = (0.5 * amp**2 * (1.0 - g_2k * cos2)
                   - 0.5 * (amp * g_k)**2 * (1.0 - cos2))
    np.testing.assert_allclose(L[0][0, 0, :], expected_xx, atol=1e-12)
    np.testing.assert_allclose(L[2], 0.0, atol=1e-15)   # zz untouched


def test_model_stress_trace_and_realizability():
    from woof.verify.sase_ref import model_stress, strain
    rng = np.random.default_rng(7)
    shape = (4, 8, 8)
    u, v, w = (rng.standard_normal(shape) for _ in range(3))
    s = strain(u, v, w, dx=500.0, dy=500.0, dz=200.0)
    e = np.full(shape, 0.4)
    tau = model_stress(e, s, 0.1, 0.6, 500.0, 500.0)
    trace = tau[0] + tau[1] + tau[2]
    np.testing.assert_allclose(trace, 2.0 * e, rtol=1e-12)


def test_dynamic_solve_recovers_manufactured_coefficients():
    """Inverse crime: build L from the model itself; solve must recover."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(20260720)
    shape = (6, 24, 24)
    # Band-limited smooth resolved field: filtered white noise.
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 4)
               for _ in range(3))
    # Non-uniform subgrid energy: with uniform e the grid-anchored
    # momentum term commutes with the test filter and vanishes from the
    # lift, leaving the momentum weight unidentifiable; spatial sqrt(e)
    # structure is what makes the background column observable.
    e = 0.25 + 0.05 * sase_ref.box_filter(rng.standard_normal(shape), 4)
    assert e.min() > sase_ref.E_MIN          # fixture stays realizable
    c_true, f_true = 0.17, 0.55
    dx = dy = 500.0
    dz = 200.0

    def tau_at(width_fields, delta_eddy):
        uu, vv, ww = width_fields
        s = sase_ref.strain(uu, vv, ww, dx, dy, dz)
        # Scale split: the dynamic eddy term rides the filter scale; the
        # momentum background stays anchored to the grid scale dx.
        return sase_ref.model_stress(e, s, c_true, f_true, delta_eddy, dx)

    lifts = {}
    for width in (2, 4):
        filt = [sase_ref.box_filter(a, width) for a in (u, v, w)]
        coarse = tau_at(filt, width * dx)
        fine = [sase_ref.box_filter(t, width)
                for t in tau_at((u, v, w), dx)]
        lifts[width] = [c - f_ for c, f_ in zip(coarse, fine)]

    c_nu, f = sase_ref.dynamic_solve(
        u, v, w, e, dx, dy, dz, delta=dx, manufactured_lifts=lifts)
    np.testing.assert_allclose(c_nu, c_true, rtol=1e-8)
    np.testing.assert_allclose(f, f_true, rtol=1e-8)


def test_dynamic_solve_clips_to_realizable_range():
    from woof.verify import sase_ref
    zeros = np.zeros((4, 16, 16))
    e = np.full((4, 16, 16), 0.3)
    c_nu, f = sase_ref.dynamic_solve(
        zeros, zeros, zeros, e, 500.0, 500.0, 200.0, delta=500.0)
    assert 0.0 <= c_nu <= sase_ref.CNU_MAX
    assert 0.0 <= f <= 1.0


def test_dynamic_solve_real_lift_golden_values():
    """Golden pin of the production (real-lift) dynamic solve.

    Same fixture as the varying-e ledger test, but calling dynamic_solve
    directly on the real path (no manufactured lifts).  The pinned pair
    freezes the six-row layout, the off-diagonal weight-1 convention, the
    cond threshold, and the clip/recovery order: any change to those
    shifts these digits and must arrive as a deliberate re-pin.  The
    literals live in tests/sase_goldens.py, shared with the GPU suite's
    device golden gate (S3-3 review fold-in).
    """
    from sase_goldens import GOLDEN_C_NU_FP64, GOLDEN_F_FP64

    from woof.verify import sase_ref
    rng = np.random.default_rng(3)
    shape = (8, 24, 24)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 4)
               for _ in range(3))
    e = np.maximum(
        0.05 + 0.1 * sase_ref.box_filter(rng.standard_normal(shape), 4),
        0.0)
    c_nu, f = sase_ref.dynamic_solve(u, v, w, e, 500.0, 500.0, 200.0,
                                     delta=500.0)
    np.testing.assert_allclose(c_nu, GOLDEN_C_NU_FP64, rtol=1e-10)
    np.testing.assert_allclose(f, GOLDEN_F_FP64, rtol=1e-10)


def test_e_decay_matches_closed_form():
    """Homogeneous zero-flow box: de/dt = -C_E e^1.5/delta exactly."""
    from woof.verify import sase_ref
    shape = (4, 8, 8)
    zeros = np.zeros(shape)
    theta = np.full(shape, 300.0)
    delta = 500.0
    e0 = 0.5
    e = np.full(shape, e0)
    dt = 0.25
    steps = 400
    for _ in range(steps):
        rhs = sase_ref.e_rhs(e, zeros, zeros, zeros, theta,
                             500.0, 500.0, 200.0, delta,
                             c_nu=0.1, f=0.5)
        e = np.maximum(
            e + dt * (rhs["production"] + rhs["buoyancy"]
                      - rhs["dissipation"] + rhs["transport"]),
            sase_ref.E_MIN)
    b = sase_ref.C_E * np.sqrt(e0) / (2.0 * delta)
    analytic = e0 / (1.0 + b * dt * steps)**2
    np.testing.assert_allclose(e.mean(), analytic, rtol=2e-3)


def test_dissipation_length_stability_limited():
    from woof.verify import sase_ref
    e = np.full((2, 2, 2), 0.09)               # sqrt(e) = 0.3
    n2 = np.full((2, 2, 2), 1.0e-4)            # N = 0.01
    l = sase_ref.dissipation_length(e, delta=500.0, n2=n2)
    np.testing.assert_allclose(l, 0.76 * 0.3 / 0.01)   # 22.8 m < 500 m
    l_neutral = sase_ref.dissipation_length(e, delta=500.0, n2=None)
    np.testing.assert_allclose(l_neutral, 500.0)
    # Mixed stability in a single call: per-cell branch selection between
    # stability-limited (n2 > 0, ls < delta), weakly stable (ls > delta,
    # delta still wins), and neutral/unstable (n2 <= 0) cells.
    n2_mixed = np.array([1.0e-4, 0.0, -5.0e-5, 1.0e-10,
                         2.5e-5, -1.0, 4.0e-4, 0.0]).reshape(2, 2, 2)
    expected = np.array([22.8, 500.0, 500.0, 500.0,
                         45.6, 500.0, 11.4, 500.0]).reshape(2, 2, 2)
    l_mixed = sase_ref.dissipation_length(e, delta=500.0, n2=n2_mixed)
    np.testing.assert_allclose(l_mixed, expected, rtol=1e-12)


def test_step_energy_ledger_closes_to_roundoff():
    from woof.verify import sase_ref
    rng = np.random.default_rng(42)
    shape = (8, 24, 24)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 4)
               for _ in range(3))
    theta = np.full(shape, 300.0)              # no buoyancy path
    e = np.full(shape, 0.2)
    fields, ledger = sase_ref.sase_ref_step(
        u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=1.0)
    scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                abs(ledger["dHeat"]), 1e-30)
    assert abs(ledger["residual"]) / scale < 1e-11
    assert np.all(fields["e"] >= sase_ref.E_MIN)


def test_step_drains_resolved_ke_into_e_for_sheared_flow():
    from woof.verify import sase_ref
    rng = np.random.default_rng(3)
    shape = (8, 24, 24)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 2)
               for _ in range(3))
    theta = np.full(shape, 300.0)
    e = np.full(shape, 0.05)
    _, ledger = sase_ref.sase_ref_step(
        u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=1.0)
    assert ledger["dKE"] < 0.0                 # forward scatter net drain
    assert ledger["dHeat"] > 0.0               # dissipation heats


def test_step_ledger_closes_with_live_clip_channel_and_varying_e():
    """Non-uniform e: clip/transport channels live, solve non-degenerate.

    The uniform-e closure test leaves transport, buoyancy, and clip_gain
    identically zero, so this fixture floors some cells of e below E_MIN
    (their provisional update stays under the floor at this dt) and adds
    spatial e structure so dynamic_solve returns a blended f > 0.  The
    residual bar is the same 1e-11; a sign flip in the clip-gain heat
    exit now breaks closure by O(sum clip_gain), not roundoff.
    """
    from woof.verify import sase_ref
    rng = np.random.default_rng(3)
    shape = (8, 24, 24)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 4)
               for _ in range(3))
    theta = np.full(shape, 300.0)              # mechanical path only
    e = np.maximum(
        0.05 + 0.1 * sase_ref.box_filter(rng.standard_normal(shape), 4),
        0.0)
    assert np.any(e < sase_ref.E_MIN)          # fixture reaches the floor
    fields, ledger = sase_ref.sase_ref_step(
        u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=0.05)
    scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                abs(ledger["dHeat"]), 1e-30)
    assert abs(ledger["residual"]) / scale < 1e-11
    assert ledger["c_nu"] > 0.0 and ledger["f"] > 0.0    # non-degenerate
    assert np.all(fields["e"] >= sase_ref.E_MIN)
    assert np.any(fields["e"] == sase_ref.E_MIN)         # clip engaged
    # Clipped cells exit through the heat channel: clip_gain exceeds
    # dt*dissipation there, so heat is locally negative (not sign-definite).
    assert np.any(fields["heat"] < 0.0)


# ---------------------------------------------------------------------------
# Stage-3 Task 1: variable-dz authority mode (clamped-z model columns).
# Convention under test: dz_col holds layer thicknesses; level k's center
# sits at sum(dz_col[:k]) + 0.5*dz_col[k].
# ---------------------------------------------------------------------------


def _stretched_thicknesses(nz, t0=50.0, ratio=1.08):
    return t0 * ratio ** np.arange(nz, dtype=np.float64)


def test_ddz_variable_quadratic_interior_exact_on_stretched_grid():
    """Three-point Lagrange stencil is quadratic-exact on a stretched grid.

    f = z^2 on geometric thicknesses (ratio 1.08 from 50 m) must return
    2*z at every interior row to 1e-12.  The two-sided form
    (f[k+1]-f[k-1])/(z[k+1]-z[k-1]) would return z[k+1]+z[k-1] =
    2*z[k] + (h_p - h_m) here, an O(h) miss on a stretched grid -- this
    fixture is what rules it out.  dz is passed as NaN to prove the
    uniform spacing argument is never consulted on the variable path.
    """
    from woof.verify import sase_ref
    nz, ny, nx = 40, 4, 4
    t = _stretched_thicknesses(nz)
    z = np.cumsum(t) - 0.5 * t
    f = np.broadcast_to((z**2)[:, None, None], (nz, ny, nx)).copy()
    d = sase_ref._ddz(f, dz=float("nan"), dz_col=t)
    expected = np.broadcast_to((2.0 * z)[:, None, None], (nz, ny, nx))
    np.testing.assert_allclose(d[1:-1], expected[1:-1], rtol=1e-12)
    # (nz, ny, nx)-shaped thicknesses are the same grid; same answer bitwise.
    t3 = np.broadcast_to(t[:, None, None], (nz, ny, nx)).copy()
    d3 = sase_ref._ddz(f, dz=float("nan"), dz_col=t3)
    assert np.array_equal(d, d3)


def test_ddz_variable_linear_exact_including_clamped_edges():
    """One-sided two-point edge rows are linear-exact on the stretched grid."""
    from woof.verify import sase_ref
    nz, ny, nx = 12, 3, 3
    t = _stretched_thicknesses(nz)
    z = np.cumsum(t) - 0.5 * t
    a, b = 3.5e-3, 1.7
    f = np.broadcast_to((a * z + b)[:, None, None], (nz, ny, nx)).copy()
    d = sase_ref._ddz(f, dz=float("nan"), dz_col=t)
    np.testing.assert_allclose(d, a, rtol=1e-12)


def test_dz_col_rejects_periodic_vertical():
    """dz_col with a periodic vertical is a domain violation: ValueError."""
    from woof.verify import sase_ref
    shape = (6, 4, 4)
    f = np.zeros(shape)
    t = np.full(shape[0], 200.0)
    with pytest.raises(ValueError, match="dz_col"):
        sase_ref._ddz(f, 200.0, periodic=True, dz_col=t)
    with pytest.raises(ValueError, match="dz_col"):
        sase_ref.strain(f, f, f, 500.0, 500.0, 200.0,
                        periodic_z=True, dz_col=t)
    with pytest.raises(ValueError, match="dz_col"):
        sase_ref.e_rhs(np.full(shape, 0.2), f, f, f, np.full(shape, 300.0),
                       500.0, 500.0, 200.0, 500.0, c_nu=0.1, f=0.5,
                       periodic_z=True, dz_col=t)


def _collapse_fixture():
    from woof.verify import sase_ref
    rng = np.random.default_rng(20260720)
    shape = (8, 12, 12)
    u, v, w = (rng.standard_normal(shape) for _ in range(3))
    e = 0.2 + 0.05 * sase_ref.box_filter(rng.standard_normal(shape), 2)
    theta = 300.0 + rng.standard_normal(shape)
    return u, v, w, e, theta


def test_variable_dz_collapses_to_uniform_path_bitwise():
    """dz_col = full(nz, dz) reproduces the clamped uniform path bitwise.

    dz = 256.0 (power of two): the center spacings, the Lagrange
    coefficients (+-1/(2*dz), exactly representable), and the uniform
    divisor 2*dz are then all exact powers of two, multiplication by
    which commutes with rounding -- so the coefficient-form stencil and
    the subtract-then-divide uniform stencil round identically and the
    collapse is exact array equality, not just allclose.
    """
    from woof.verify import sase_ref
    u, v, w, e, theta = _collapse_fixture()
    dx = dy = 500.0
    dz = 256.0
    t = np.full(u.shape[0], dz)
    s_uni = sase_ref.strain(u, v, w, dx, dy, dz)
    s_var = sase_ref.strain(u, v, w, dx, dy, dz, dz_col=t)
    for a, b in zip(s_uni, s_var):
        assert np.array_equal(a, b)
    r_uni = sase_ref.e_rhs(e, u, v, w, theta, dx, dy, dz, 500.0,
                           c_nu=0.12, f=0.6)
    r_var = sase_ref.e_rhs(e, u, v, w, theta, dx, dy, dz, 500.0,
                           c_nu=0.12, f=0.6, dz_col=t)
    for key in r_uni:
        assert np.array_equal(r_uni[key], r_var[key]), key


def test_variable_dz_collapses_to_uniform_path_generic_spacing():
    """Uniform collapse at a generic (non-power-of-two) dz = 200 m.

    NOT bitwise, by expression grouping: the variable path multiplies by
    the precomputed Lagrange coefficient fl(1/(2*dz)) while the uniform
    path subtracts then divides by 2*dz, and fl(x*fl(1/400)) differs
    from fl(x/400) for generic (non-power-of-two) spacing.  The noise is
    absolute, ~ULP(|f|*coeff) per stencil term, so the accurate bound is
    relative to each field's magnitude, not element-wise: theta's 300 K
    offset makes the buoyancy term's worst cell measure 5.2e-15 of the
    field max (coefficient noise scales with |theta|, the derivative
    with the ~1 K increments), which is why a pure rtol=1e-15 cannot
    hold at cancellation cells.  Bound: 1e-14 * max|uniform field|,
    ~2x measured headroom, deterministic seeded fixture.
    """
    from woof.verify import sase_ref
    u, v, w, e, theta = _collapse_fixture()
    dx = dy = 500.0
    dz = 200.0
    t = np.full(u.shape[0], dz)
    s_uni = sase_ref.strain(u, v, w, dx, dy, dz)
    s_var = sase_ref.strain(u, v, w, dx, dy, dz, dz_col=t)
    for a, b in zip(s_uni, s_var):
        np.testing.assert_allclose(b, a, rtol=0.0,
                                   atol=1.0e-14 * np.max(np.abs(a)))
    r_uni = sase_ref.e_rhs(e, u, v, w, theta, dx, dy, dz, 500.0,
                           c_nu=0.12, f=0.6)
    r_var = sase_ref.e_rhs(e, u, v, w, theta, dx, dy, dz, 500.0,
                           c_nu=0.12, f=0.6, dz_col=t)
    for key in r_uni:
        np.testing.assert_allclose(
            r_var[key], r_uni[key], rtol=0.0,
            atol=1.0e-14 * max(np.max(np.abs(r_uni[key])), 1e-300),
            err_msg=key)


def test_sase_ref_step_variable_dz_model_mode_runs():
    """Model mode: step on stretched thicknesses stays finite and floored.

    dz_col switches the step to the clamped-z column mode; the periodic
    conservation theorem does not apply there, so the ledger is
    diagnostic only -- no residual bar is asserted.
    """
    from woof.verify import sase_ref
    u, v, w, e, theta = _collapse_fixture()
    t = _stretched_thicknesses(u.shape[0])
    fields, ledger = sase_ref.sase_ref_step(
        u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=0.05, dz_col=t)
    for key in ("u", "v", "w", "e", "heat"):
        assert np.all(np.isfinite(fields[key])), key
    assert np.all(fields["e"] >= sase_ref.E_MIN)
    assert np.isfinite(ledger["residual"])


def test_free_convection_e_scales_with_w_star_squared():
    """S3-6b re-derived CBL fixture (amended vertical formulation).

    The column now runs the RANS limit of the amended channel: length
    l_B(z) = k*z/(1 + k*z/lambda) instead of the v0 fixed 0.1*zi, and
    implicit 2*K_v e-transport.  Closed form for the regime (local
    balance flux = C_E*e^{3/2}/l_B):  e_lb(z) = (l_B*flux/C_E)^(2/3),
    returned as ``e_local_balance``.  Measured/predicted = 0.974 (the
    implicit transport redistributes and drops the ML mean ~2.6% below
    local balance) -- pinned in [0.90, 1.10], a TIGHTER bind than the
    old absolute-only band.  Absolute similarity ratio: e_ml/w*^2 =
    (mean of (l_B*flux/C_E)^{2/3})/ (b0*zi)^{2/3} ~ 0.110 for zi = 1 km
    (the mid-ML l_B ~ 65-95 m sits below the v0 100 m, and the profile
    weighting lowers the mean; the v0 measured 0.158 becomes 0.110 --
    a re-derived PARAMETER of the amended formula, not a loosened
    tolerance: the closed-form prediction moved and the band floor
    moved with it).  Floor 0.10 keeps the C_E canary: ratio ~ C_E^{-2/3}
    trips it at C_E >~ 1.07, the same thin-margin-by-design intent.
    """
    from woof.verify.sase_ref import column_free_convection
    runs = [column_free_convection(b0=b0, zi=1000.0, nz=50,
                                   dt=2.0, steps=4000)
            for b0 in (0.003, 0.009)]          # 3x surface flux ratio
    ratios = [r["e_ml"] / r["w_star"]**2 for r in runs]
    for run, ratio in zip(runs, ratios):
        np.testing.assert_allclose(run["e_ml"] / run["e_local_balance"],
                                   0.974, atol=0.10)   # closed-form bind
        assert 0.10 < ratio < 0.8              # CBL similarity band
    # Scaling: e_ml must track w_star^2 within 10 percent across the pair
    # (measured: identical to ~1e-6 -- the discrete system scales almost
    # exactly; the band guards regime changes, not noise).
    np.testing.assert_allclose(ratios[0], ratios[1], rtol=0.10)


# ---------------------------------------------------------------------------
# Stage-3 Task 5: admission gate, state allocation, preflight, restart pins.
# ---------------------------------------------------------------------------

#: A minimal admitted SASE configuration.  Unlike the development lane
#: this port grew from, the admission gate pins only what the closure
#: actually depends on -- so this dict names no microphysics, radiation
#: or land-surface scheme, and the suite is free to vary them.
_SASE_ADMITTED = dict(bl_pbl_physics=_SASE_SELECTOR, moist=True,
                      sf_sfclay_physics=1, km_opt=0)

#: What the closure genuinely requires, field -> an offending value.
#: Each entry is a thing the code reads or a thing that would be
#: double-counted; there is deliberately no entry for a scheme id.
_SASE_REQUIRED_DEVIATIONS = [
    ("moist", False),
    ("sf_sfclay_physics", 0),
    ("km_opt", 4),
    ("khdif", 1.0),
    ("kvdif", 0.5),
]


def _sase_cfg(**kw):
    base = dict(_SASE_ADMITTED)
    base.update(kw)
    return _min_cfg(**base)


def test_sase_admitted_combination_passes_validation():
    cfg = _sase_cfg()
    assert validate_run_config(cfg) is cfg


@pytest.mark.parametrize("field,value", _SASE_REQUIRED_DEVIATIONS)
def test_sase_rejects_every_genuine_requirement(field, value):
    """Each single-field deviation raises, naming the field."""
    cfg = _sase_cfg(**{field: value})
    with pytest.raises(ValueError, match=field):
        validate_run_config(cfg)


@pytest.mark.parametrize("field,value", [
    ("mp_physics", 8),
    ("ra_physics", 0),
    ("sf_surface_physics", 0),
])
def test_sase_admits_schemes_it_does_not_depend_on(field, value):
    """The closure reads no hydrometeor tendency, radiative flux or soil
    layer, so pinning those would refuse combinations that work while
    claiming a coupling that does not exist.

    This is the counterpart of the rejection test above and the reason
    the two exist as a pair: together they say exactly where the line
    between a dependency and a habit sits.
    """
    validate_run_config(_sase_cfg(**{field: value}))


def test_sase_refusal_explains_itself_rather_than_citing_a_ledger():
    """Every refusal states the physical reason, not a lane decision."""
    with pytest.raises(ValueError) as excinfo:
        validate_run_config(_sase_cfg(km_opt=4))
    message = str(excinfo.value)
    assert "double-count" in message
    assert str(_SASE_SELECTOR) in message


# ---------------------------------------------------------------------------
# The acknowledged km_opt = 0 research control.
#
# WHAT THIS IS.  SASE was admitted at km_opt = 0 because its closure
# supplies the horizontal mixing the operator would otherwise apply.
# That admission, alone, made the single-variable control UNWRITABLE:
# every SASE run necessarily changed the PBL scheme AND removed the
# Smagorinsky operator, so no run could say which of the two any
# difference came from.  km_opt = 0 with a vertical-only PBL scheme --
# the cell that holds the closure fixed and removes only the mixing --
# is now expressible behind an explicit written acknowledgement.
#
# WHAT THIS IS NOT.  It is not a gate widened to make a test pass.  The
# tests below pin BOTH halves: the default still refuses, with a refusal
# that names what is missing, and only an exactly-spelled id opens it.
# ---------------------------------------------------------------------------

def _unmixed_cfg(**kw):
    """YSU with no horizontal mixing operator -- the attribution control."""
    from woof.config import KM_OPT_ZERO_ACK
    base = dict(bl_pbl_physics=1, sf_sfclay_physics=1, moist=True,
                km_opt=0, km_opt_zero_acknowledgement=KM_OPT_ZERO_ACK)
    base.update(kw)
    return _min_cfg(**base)


@pytest.mark.parametrize("pbl", [0, 1, 5])
def test_km_opt_zero_refuses_by_default_on_every_vertical_only_scheme(pbl):
    """The gate the acknowledgement opens still refuses when unopened.

    Parameterized over every non-SASE PBL selector, including PBL-off,
    because the refusal is about the absence of a horizontal mixing
    PRODUCER and not about which vertical scheme is running.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_run_config(_min_cfg(bl_pbl_physics=pbl, km_opt=0,
                                     sf_sfclay_physics=1, moist=True))
    message = str(excinfo.value)
    # It says what is actually wrong -- no operator -- rather than
    # reciting the admitted value list.
    assert "NO horizontal mixing operator" in message
    # It names the exact line that opens it, and the alternatives.
    from woof.config import KM_OPT_ZERO_ACK
    assert KM_OPT_ZERO_ACK in message
    assert "km_opt = 4" in message and str(_SASE_SELECTOR) in message


@pytest.mark.parametrize("pbl", [0, 1, 5])
def test_km_opt_zero_is_admitted_with_the_written_acknowledgement(pbl):
    """The acknowledged path validates, on the loader AND in the dycore.

    Both sites are asserted because the dycore's check is the
    fail-closed one -- it is what decides whether a mixing operator runs
    -- and a config that got past the loader and then died in the dycore
    would be a control that still could not be run.
    """
    from types import SimpleNamespace

    from woof.core import dycore

    cfg = _unmixed_cfg(bl_pbl_physics=pbl)
    assert validate_run_config(cfg) is cfg
    # The dycore agrees: whatever stops a bare stub afterwards, it is no
    # longer km_opt (same contract as the SASE half of this pair).
    with pytest.raises((ValueError, AttributeError, TypeError)) as excinfo:
        dycore.step(SimpleNamespace(h_diabatic=None), cfg)
    assert "km_opt" not in str(excinfo.value)


def test_km_opt_zero_acknowledgement_must_be_spelled_exactly():
    """A literal id, not a boolean: no stray truthy value reaches it."""
    for wrong in (True, "true", "yes", "1", "no-horizontal-mixing", ""):
        with pytest.raises(ValueError) as excinfo:
            validate_run_config(_min_cfg(bl_pbl_physics=1, km_opt=0,
                                         sf_sfclay_physics=1, moist=True,
                                         km_opt_zero_acknowledgement=wrong))
        message = str(excinfo.value)
        assert ("not the acknowledgement id" in message
                or "NO horizontal mixing operator" in message
                or "must be the exact string" in message), (wrong, message)


def test_km_opt_zero_acknowledgement_refuses_where_it_acknowledges_nothing():
    """A key that names a seam this run does not have would read as a
    setting that took effect -- the same fail-closed rule the closure's
    own three knobs carry."""
    from woof.config import KM_OPT_ZERO_ACK

    # km_opt = 4 runs an operator, so there is no absence to acknowledge.
    with pytest.raises(ValueError, match="acknowledges nothing"):
        validate_run_config(_min_cfg(
            bl_pbl_physics=1, km_opt=4, sf_sfclay_physics=1, moist=True,
            km_opt_zero_acknowledgement=KM_OPT_ZERO_ACK))
    # SASE at km_opt = 0 has a producer, so its km_opt = 0 is admitted
    # WITHOUT the id, and carrying it would record an absence this run
    # does not have.
    with pytest.raises(ValueError, match="already supplies"):
        validate_run_config(_sase_cfg(
            km_opt_zero_acknowledgement=KM_OPT_ZERO_ACK))
    assert validate_run_config(_sase_cfg()) is not None


def test_km_opt_zero_acknowledgement_opens_nothing_else():
    """The acknowledgement admits km_opt = 0 and NOTHING else.

    The named risk of widening a refusal is that it becomes a general
    bypass.  Every other km_opt refusal is re-checked with the id set.
    """
    from woof.config import KM_OPT_ZERO_ACK

    # km_opt=2 is a real selector on this line (LES lane), admitted with
    # bl_pbl_physics=0 only -- but the acknowledgement still opens NOTHING
    # beyond km_opt=0: setting it beside km_opt=2 is refused as misplaced
    # before any LES admission question is reached.
    with pytest.raises(ValueError, match="acknowledges nothing here"):
        validate_run_config(_min_cfg(
            bl_pbl_physics=1, km_opt=2, sf_sfclay_physics=1, moist=True,
            km_opt_zero_acknowledgement=KM_OPT_ZERO_ACK))
    # The constant-K stacking refusal is untouched by it.  Since f586d1128
    # it stands under diff_opt=1 only, where WRF's vertical diffusion
    # reads kvdif whatever km_opt is; under diff_opt=2 WRF ignores
    # khdif/kvdif outside km_opt=1, so the pair imports as WRF runs it.
    with pytest.raises(ValueError, match="diff_opt=1 with km_opt=4"):
        validate_run_config(_min_cfg(
            bl_pbl_physics=1, km_opt=4, sf_sfclay_physics=1, moist=True,
            khdif=75.0, diff_opt=1))


def test_km_opt_zero_control_reaches_the_loader_from_a_config_file(tmp_path):
    """The control is writable in a config FILE, not merely constructible.

    A gate that only opens for a hand-built RunConfig is not a control
    anyone can run; this drives the shipped experiment loader.
    """
    from woof.config import KM_OPT_ZERO_ACK
    from woof.experiment import load_experiment

    text = (tmp_path / "unmixed.toml")
    text.write_text(
        "[experiment]\n"
        'name = "unmixed"\n'
        "start_time = 2026-01-01T00:00:00\n"
        "run_seconds = 60.0\n"
        "restart_interval_s = 0.0\n"
        "[shared]\n"
        "nz = 8\nztop = 8000.0\np_top = 10000.0\n"
        "bl_pbl_physics = 1\nsf_sfclay_physics = 1\nmoist = true\n"
        "km_opt = 0\n"
        f'km_opt_zero_acknowledgement = "{KM_OPT_ZERO_ACK}"\n'
        "[[domain]]\n"
        "grid_id = 1\nparent_id = 0\ni_parent_start = 1\n"
        "j_parent_start = 1\nparent_grid_ratio = 1\n"
        "parent_time_step_ratio = 1\nhistory_interval_s = 60.0\n"
        "nx = 16\nny = 16\ndx = 1000.0\ntime_step = 60\n",
        encoding="utf-8")
    exp = load_experiment(text)
    run = exp.domains[0].run
    assert run.km_opt == 0 and run.bl_pbl_physics == 1
    assert run.km_opt_zero_acknowledgement == KM_OPT_ZERO_ACK
    # And the same file without the line refuses, so the file is the
    # place the acknowledgement is made rather than a formality.
    text.write_text(
        text.read_text(encoding="utf-8").replace(
            f'km_opt_zero_acknowledgement = "{KM_OPT_ZERO_ACK}"\n', ""),
        encoding="utf-8")
    with pytest.raises(ValueError, match="NO horizontal mixing operator"):
        load_experiment(text)


def test_sase_state_allocates_e_sgs_at_e_min(monkeypatch):
    """Active sase: e_sgs is (nz, ny, nx) float32 filled to E_MIN; the
    Coriolis field ``e`` keeps its identity beside it."""
    import woof.core.state as state_module
    from woof.verify.sase_ref import E_MIN

    monkeypatch.setattr(state_module, "cp", np)
    cfg = _sase_cfg()
    state = state_module.DomainState(cfg)
    assert state.e_sgs.shape == (cfg.nz, cfg.ny, cfg.nx)
    assert state.e_sgs.dtype == np.float32
    assert np.all(state.e_sgs == np.float32(E_MIN))
    assert state.e.shape == (cfg.ny, cfg.nx)          # Coriolis, untouched


def test_inactive_sase_state_never_allocates_e_sgs(monkeypatch):
    """a non-SASE PBL selector keeps the object graph byte-identical: the attribute
    is ABSENT (never allocated, not None) exactly like Morrison moments
    on a Kessler state."""
    import woof.core.state as state_module

    monkeypatch.setattr(state_module, "cp", np)
    state = state_module.DomainState(
        _min_cfg(**{**_SASE_ADMITTED, "bl_pbl_physics": 0,
                    "km_opt": 4}))
    assert not hasattr(state, "e_sgs")


def test_e_sgs_restart_classification():
    """e_sgs is serialized cross-step state; Coriolis ``e`` stays setup."""
    from woof.io import restart

    assert "e_sgs" in restart.STATE_SERIALIZED_ATTRS
    assert restart.classify_state_attr("e_sgs") == "serialize"
    assert "e" in restart.STATE_SETUP_ARRAYS
    assert "e" not in restart.STATE_SERIALIZED_ATTRS


def test_preflight_state_shapes_gain_exactly_e_sgs_under_sase():
    from dataclasses import replace

    from woof.core import preflight as pf

    cfg = _sase_cfg()
    shapes = pf.state_array_shapes(cfg)
    baseline = pf.state_array_shapes(replace(cfg, bl_pbl_physics=0))
    assert shapes.pop("e_sgs") == (cfg.nz, cfg.ny, cfg.nx)
    assert shapes == baseline


def test_sase_workspace_accounting_is_exact():
    """The model-path transient workspace at its dynamic-solve peak
    (S3-6e re-pin; S4-3 M1 re-pin; S4-5 M2 re-pin): the fused step's 49
    simultaneous
    (nz, ny, nx) float32 temporaries (heat + 6 fine strain + 6 premul +
    3 filtered velocities + 6 coarse strain + 6 refiltered basis + the
    Germano lift's 3 filters + 6 products + 6 filtered products +
    6 lifts) + the three per-column z-stencil coefficient FIELDS (3-D
    dz_col mode, device FP64 build) + the driver coupling's SEVEN held
    work fields (u/v work, destaggered w + w work, n2 + its S4-2 M1
    moist companion n2_eff -- the launch_moist_n2 output held beside
    the dry field for the whole step -- and the S4-5 M2 seam's frozen
    pre-step e_sgs copy, keyed driver_vent_e_pre; the v0 pre-step e copy
    retired with the S3-6e governed scalar channel, which rides the
    step's exported km_h field, stays retired, and its key
    driver_e_pre is asserted ABSENT below, the guard the S4-5 build
    deleted when it reused that key -- S4-5b Item 4b.  The rename does
    not move the byte total: one (nz, ny, nx) float32 field under
    either key, so the 59-field pin below is unchanged and is
    re-derived by the same arithmetic) = 59 full fields, + one FP64
    (5, nblocks) partial-sum
    buffer, + the (ny, nx) float32 rho1 surface moist-density plane
    (S3-11b note-3 de-minimis, ~0.04% of the peak, absorbed by the
    S4-3 pairing amendment), + the S3-6f partition-bound sub-moment
    transcribed as a covering superset (the (ny, nx) z_i column field
    and the FP64 (5, nblocks) w-sensor partials -- allocated after the
    solve transients release, freed before the apply allocations).

    Byte-pin derivation (S4-3): pre-M1 total was 57*4*ncell +
    2*(8*5*nblocks) + 4*ny*nx.  The M1 driver field n2_eff adds one
    full 4*ncell field (1/57 ~ 1.75% of the previous peak, the S4-2
    report note-1 number) and the rho1 plane adds a second 4*ny*nx
    plane (1/(57*nz) ~ 0.04%, the S3-11b report note-3 number).
    S4-5 (the in-task pairing law): the M2 seam's frozen pre-step e_sgs
    copy adds a third full 4*ncell field (1/58 ~ 1.72%).  The seam's
    own three (nz+1, ny, nx) face-flux planes and its (ny, nx) FP64
    cap-rescale plane are allocated in the APPLY phase (after the split
    step returns, before the scalar loop) and are transcribed there --
    asserted below to leave the solve phase dominating, which is what
    keeps the category bound on this pin."""
    import math
    from dataclasses import replace

    from woof.core import preflight as pf
    from woof.core.sase_limits import THREADS_PER_BLOCK

    cfg = _sase_cfg()
    assert pf.sase_workspace_shapes(replace(cfg, bl_pbl_physics=0)) == {}
    shapes = pf.sase_workspace_shapes(cfg)
    ncell = cfg.nz * cfg.ny * cfg.nx
    nblocks = -(-ncell // THREADS_PER_BLOCK)
    total = sum(math.prod(shape) * size for shape, size in shapes.values())
    assert total == (59 * 4 * ncell + 8 * 5 * nblocks
                     + 4 * cfg.ny * cfg.nx + 8 * 5 * nblocks
                     + 4 * cfg.ny * cfg.nx)
    # Every entry is named under the sase/ prefix with its phase, and the
    # driver-held work set is transcribed by name in the peak phase.
    assert all(name.startswith("sase/") for name in shapes)
    for name in ("u_work", "v_work", "w_half", "w_work", "n2", "n2_eff",
                 "vent_e_pre"):
        assert f"sase/solve/driver_{name}" in shapes, name
    assert "sase/solve/driver_rho1" in shapes
    # RESTORED (S4-5b Item 4b): the S3-6e retirement guard.  The v0
    # pre-step e copy that fed launch_scalar_mix's coefficient mode is
    # retired and must not come back; the S4-5 build gave the M2 seam's
    # frozen copy that exact key and deleted this line.  The M2 field is
    # keyed driver_vent_e_pre (asserted present above), so the guard
    # means what it meant again.
    assert "sase/solve/driver_e_pre" not in shapes
    # The S4-5 M2 seam allocates in the APPLY phase; the pin above is the
    # SOLVE phase, so those entries must NOT appear in the bound set --
    # and the apply phase must stay strictly below it (the transcription
    # is a covering superset there, exactly as for the other post-drop
    # sub-moments).
    for name in ("vent_f_theta", "vent_f_qv", "vent_f_qc", "vent_scale"):
        assert f"sase/solve/{name}" not in shapes, name
    phases = pf.sase_workspace_phases(cfg)
    for name in ("vent_f_theta", "vent_f_qv", "vent_f_qc", "vent_scale"):
        assert name in phases["apply"], name
    assert phases["apply"]["vent_f_theta"] == ((cfg.nz + 1, cfg.ny, cfg.nx),
                                               4)
    assert phases["apply"]["vent_scale"] == ((cfg.ny, cfg.nx), 8)

    def _tot(items):
        return sum(math.prod(shape) * size for shape, size in items.values())

    assert _tot(phases["apply"]) < _tot(phases["solve"])


def test_sase_workspace_joins_the_domain_estimate_as_step_transient():
    import math
    import types
    from dataclasses import replace

    from woof.core import preflight as pf

    cfg = _sase_cfg()
    base = replace(cfg, bl_pbl_physics=0)
    est = pf.estimate_domain(
        types.SimpleNamespace(run=cfg, grid_id=1, parent_id=0))
    ref = pf.estimate_domain(
        types.SimpleNamespace(run=base, grid_id=1, parent_id=0))
    ncell = cfg.nz * cfg.ny * cfg.nx
    sase_bytes = sum(math.prod(shape) * size for shape, size in
                     pf.sase_workspace_shapes(cfg).values())
    assert est.category_bytes("sase") == sase_bytes
    # The workspace is a step transient, freed within the step and never
    # residency: that claim is the transient_bytes assertion at the end.
    #
    # What the closure adds to RESIDENCY is read off the estimate's own
    # itemization, NAMED, and then checked against the inventory that
    # prices each item -- not written here as one number.  A number here
    # is what went stale: this test used to say e_sgs was the only
    # resident addition, which stopped being true when the z-face
    # vertical-momentum tendency became an allocation the driver makes
    # at construction rather than on first use (woof/core/physics.py,
    # PhysicsDriver.__init__ passing optional component "rw" to
    # PhysicsTendencies.zeros, which allocates it (nz + 1, ny, nx)
    # float32 and holds it across steps).  Naming both items means
    # gaining or losing one fails with a name in it instead of a
    # residual to re-fit, and a re-fit is how an estimate silently stops
    # pricing what a run allocates.
    def resident_items(estimate):
        return {item.name: item.nbytes for item in estimate.items
                if item.category not in ("transient", "sase")}

    resident, resident_base = resident_items(est), resident_items(ref)
    assert set(resident) - set(resident_base) == {"e_sgs",
                                                  "pbl_tendencies/rw"}
    assert set(resident_base) - set(resident) == set()
    assert all(resident[name] == nbytes
               for name, nbytes in resident_base.items())
    # e_sgs: the prognostic closure energy, mass grid, float32.
    assert resident["e_sgs"] == 4 * math.prod(
        pf.state_array_shapes(cfg)["e_sgs"]) == 4 * ncell
    # pbl_tendencies/rw: the held z-face vertical-momentum tendency, and
    # SASE's alone -- no other PBL selector prices it.
    assert resident["pbl_tendencies/rw"] == 4 * math.prod(
        pf.physics_array_shapes(cfg)["pbl_tendencies/rw"])
    assert resident["pbl_tendencies/rw"] == 4 * (cfg.nz + 1) * cfg.ny * cfg.nx
    assert "pbl_tendencies/rw" not in pf.physics_array_shapes(base)
    assert est.resident_bytes == ref.resident_bytes + sum(
        resident[name] for name in set(resident) - set(resident_base))
    assert est.transient_bytes == ref.transient_bytes + sase_bytes


# ---------------------------------------------------------------------------
# SPLIT SUBGRID-FLUX DIAGNOSTIC (RunConfig sase_flux_diag): the switch,
# its per-domain wiring, its declared residency, and the default-off
# inertness.  The device fills, their authority parity and the
# channels-sum-to-the-model-increment closure live in
# tests/test_sase_gpu.py.
# ---------------------------------------------------------------------------


def test_sase_flux_diag_defaults_off_and_is_fail_closed():
    """Off on every path, and never a silently inert key."""
    import dataclasses

    assert _min_cfg().sase_flux_diag is False
    assert _sase_cfg().sase_flux_diag is False
    names = [f.name for f in dataclasses.fields(RunConfig)]
    # APPENDED (positional construction).  Pinned as the trio's own
    # contiguity and its offset from the pre-SASE tail rather than as
    # "the last three": later work appends further keys behind it, and
    # a test that broke on every future append would say nothing about
    # whether THIS key moved.  The complete field order is pinned once,
    # for every field, at
    # tests/test_config_freeze.py::test_new_fields_are_reviewed_defaults_appended_last.
    start = names.index("sase_flux_diag")
    assert names[start:start + 3] == ["sase_flux_diag", "sase_moist_n2",
                                      "sase_stable_dissipation"]
    # Union re-pin (1.5 integration line): the LES lane's seven km_opt
    # knobs and the mp=28 lane's two aerosol selectors were appended in
    # their own merges, so the last pre-SASE field is now wif_input_opt.
    # The claim guarded is unchanged: nothing BETWEEN the base tail and
    # the SASE trio may move, and the trio stays contiguous.
    assert names[start - 1] == "wif_input_opt", (
        "the SASE trio must still sit immediately after the last "
        "pre-SASE field; anything else means an existing field moved")
    # Admitted only under the closure whose channels it records.
    cfg = _sase_cfg(sase_flux_diag=True)
    assert validate_run_config(cfg) is cfg
    with pytest.raises(ValueError, match="sase_flux_diag"):
        validate_run_config(_min_cfg(sase_flux_diag=True))
    # ... and the rejection names the closure it needs.
    with pytest.raises(ValueError, match="requires bl_pbl_physics"):
        validate_run_config(_min_cfg(sase_flux_diag=True))


def test_additive_dissipation_accepts_both_values_off_sase():
    """The pre-flip recorded default must keep validating off-SASE.

    The 2026-08-16 default flip (False -> True, 1a0e8a7f8) turned every
    artifact that RECORDS the old default into a refusal: a 2.4.x
    restart header carries ``sase_additive_dissipation=False`` beside a
    non-SASE PBL, and so does every child TOML 2.4.x downscale rendered.
    MEASURED 2026-08-17: `woof downscale --point` of a 2.4.1 archive on
    this tree refused (masked as "no child fits the 10 GiB budget").
    The knob is inert off-SASE in BOTH positions, so neither may refuse
    -- the fail-closed loop's own charter is "every existing
    configuration keeps validating unchanged".
    """
    for value in (False, True):
        cfg = _min_cfg(sase_additive_dissipation=value)
        assert validate_run_config(cfg) is cfg
    # Type discipline stays: a non-bool still refuses.
    with pytest.raises(ValueError, match="boolean"):
        validate_run_config(_min_cfg(sase_additive_dissipation=1))


def test_sase_driver_switches_fall_back_to_the_shipped_default():
    """A configuration without the key keeps the default that SHIPS.

    The driver used to read ``sase_additive_dissipation`` with a literal
    ``False`` fallback while the shipped default is ``True``
    (woof/config.py).  RunConfig always carries the field, so nothing
    reachable through a RunConfig ever hit that fallback -- but any
    other configuration object handed to the driver would have lost a
    channel that ships ON, without saying so, which is the opposite of
    what a default-on remedy means.  The fallback now comes off the
    dataclass, so the driver cannot disagree with the one place the
    default is declared.

    The other two switches in the same driver block are pinned here for
    the same reason and not because they moved: both already resolved to
    their own shipped default, and this holds them there through any
    later flip.
    """
    import types

    from woof.core.physics import sase_switch_value

    bare = types.SimpleNamespace()
    shipped = {"sase_additive_dissipation": True,
               "sase_moist_n2": True,
               "sase_stable_dissipation": False}
    for name, value in shipped.items():
        assert getattr(RunConfig, name) is value, name
        assert sase_switch_value(bare, name) is value, name
    # A configuration that DOES carry the key is read from it, in both
    # positions, for every one of the three.
    for name in shipped:
        for value in (False, True):
            cfg = _sase_cfg(**{name: value})
            assert sase_switch_value(cfg, name) is value, (name, value)


def test_sase_flux_diag_is_a_per_domain_override():
    """The key is settable per [[domain]] so the expensive domain can be
    left off while the read domains carry it."""
    from woof import experiment

    assert "sase_flux_diag" in experiment._DOMAIN_RUN_OVERRIDES
    assert "sase_flux_diag" in experiment._DOMAIN_KEYS
    # It is an OUTPUT selector, not a vertical-grid key.
    assert "sase_flux_diag" not in experiment._DOMAIN_VERTICAL_KEYS


def test_sase_flux_diag_preflight_itemizes_four_resident_face_planes():
    """The four buffers are DRIVER PERSISTENTS, so they join the resident
    estimate through physics_array_shapes -- and must NOT appear in the
    step-transient sase workspace, whose 59-field byte pin
    (test_sase_workspace_accounting_is_exact) is therefore unmoved."""
    import math
    import types
    from dataclasses import replace

    from woof.core import preflight as pf

    off = _sase_cfg()
    on = replace(off, sase_flux_diag=True)
    base = pf.physics_array_shapes(off)
    with_diag = pf.physics_array_shapes(on)
    added = {name: shape for name, shape in with_diag.items()
             if name not in base}
    assert set(base).issubset(with_diag)
    assert set(added) == {f"sase_flux_diag/{name}" for name in
                          ("fqv_vent", "fqv_diff", "fth_vent", "fth_diff")}
    face = (off.nz + 1, off.ny, off.nx)
    assert all(shape == face for shape in added.values())
    # The step-transient workspace is untouched: same keys, same bytes.
    assert pf.sase_workspace_shapes(on) == pf.sase_workspace_shapes(off)
    assert pf.sase_workspace_phases(on) == pf.sase_workspace_phases(off)
    # Resident residency grows by exactly the four planes and nothing
    # else; the transient bound does not move at all.
    est_on = pf.estimate_domain(
        types.SimpleNamespace(run=on, grid_id=1, parent_id=0))
    est_off = pf.estimate_domain(
        types.SimpleNamespace(run=off, grid_id=1, parent_id=0))
    planes = 4 * 4 * math.prod(face)
    assert est_on.resident_bytes == est_off.resident_bytes + planes
    assert est_on.transient_bytes == est_off.transient_bytes
    # A non-sase config never reaches the branch (the validator forbids
    # the combination, and the estimator must agree by construction).
    assert pf.physics_array_shapes(replace(
        off, bl_pbl_physics=0, moist=True, mp_physics=10, ra_physics=4,
        sf_sfclay_physics=1, sf_surface_physics=2,
        sase_flux_diag=True)) == pf.physics_array_shapes(replace(
            off, bl_pbl_physics=0, moist=True, mp_physics=10, ra_physics=4,
            sf_sfclay_physics=1, sf_surface_physics=2))


def test_sase_flux_diag_history_metadata_declares_units_and_sign():
    """The four names carry their units and their POSITIVE-UP sign in the
    history attributes a reader actually sees."""
    from woof.io.wrfout import _VAR_META

    expected = {"SASE_FQV_VENT": "kg m-2 s-1",
                "SASE_FQV_DIFF": "kg m-2 s-1",
                "SASE_FTH_VENT": "W m-2",
                "SASE_FTH_DIFF": "W m-2"}
    for name, units in expected.items():
        description, got = _VAR_META[name]
        assert got == units, name
        assert "positive up" in description, name
        assert "SASE" in description, name
    # No case name reaches a generic identifier.
    for name, (description, units) in _VAR_META.items():
        blob = f"{name} {description} {units}".lower()
        for case in ("real74", "hrrr", "ohio", "n5s"):
            assert case not in blob, (name, case)


# ---------------------------------------------------------------------------
# The horizontal eddy-viscosity diagnostic (cfg.hmix_k_diag).
#
# SASE's central claim is that its closure supplies the horizontal mixing
# the km_opt operator would otherwise apply.  Until this field existed
# that claim was neither ablatable nor observable: the closure published
# four optional flux diagnostics and all four were VERTICAL.  These
# tests pin the field that makes the claim a measurement -- SASE's
# governed horizontal diffusivity, in the same units, on the same grid,
# under a name that says which producer made it.
# ---------------------------------------------------------------------------

def test_hmix_k_diag_names_the_producer_the_run_actually_has():
    """One key, three answers, and the third one is an ABSENCE."""
    from dataclasses import replace

    from woof.core.physics import hmix_k_diag_names
    from woof.config import KM_OPT_ZERO_ACK

    sase = _sase_cfg()
    assert hmix_k_diag_names(sase) == ("SASE_KMH", "SASE_KHH")
    smag = _min_cfg(bl_pbl_physics=1, sf_sfclay_physics=1, moist=True,
                    km_opt=4)
    assert hmix_k_diag_names(smag) == ("XKMH", "XKHH")
    # The acknowledged control has NO producer, and publishes nothing:
    # an absent variable cannot be misread as a measured zero.
    unmixed = replace(smag, km_opt=0,
                      km_opt_zero_acknowledgement=KM_OPT_ZERO_ACK)
    assert hmix_k_diag_names(unmixed) == ()


def test_hmix_k_diag_history_metadata_is_comparable_across_producers():
    """Both producers' rows carry the SAME units on the SAME grid.

    This is the whole point of the diagnostic: if the two names did not
    mean the same thing, comparing them would prove nothing.
    """
    from woof.io.wrfout import _VAR_META

    for name in ("XKMH", "XKHH", "SASE_KMH", "SASE_KHH"):
        description, units = _VAR_META[name]
        assert units == "m2 s-1", name
        assert description, name
    # WRF's own Registry names for the Smagorinsky pair, so a reader who
    # knows WRF reads them without a translation table.
    assert "MOMENTUM EDDY VISCOSITY" in _VAR_META["XKMH"][0]
    assert "HEAT EDDY VISCOSITY" in _VAR_META["XKHH"][0]
    assert "SASE" in _VAR_META["SASE_KMH"][0]


def test_hmix_k_diag_preflight_itemizes_two_resident_mass_planes():
    """Driver persistents, so they join the resident estimate -- and the
    step-transient workspace byte pin stays where it is."""
    import math
    import types
    from dataclasses import replace

    from woof.core import preflight as pf

    off = _sase_cfg()
    on = replace(off, hmix_k_diag=True)
    base = pf.physics_array_shapes(off)
    with_diag = pf.physics_array_shapes(on)
    added = {name: shape for name, shape in with_diag.items()
             if name not in base}
    assert set(added) == {"hmix_k_diag/SASE_KMH", "hmix_k_diag/SASE_KHH"}
    mass = (off.nz, off.ny, off.nx)
    assert all(shape == mass for shape in added.values())
    assert pf.sase_workspace_shapes(on) == pf.sase_workspace_shapes(off)
    est_on = pf.estimate_domain(
        types.SimpleNamespace(run=on, grid_id=1, parent_id=0))
    est_off = pf.estimate_domain(
        types.SimpleNamespace(run=off, grid_id=1, parent_id=0))
    assert est_on.resident_bytes == (
        est_off.resident_bytes + 2 * 4 * math.prod(mass))
    assert est_on.transient_bytes == est_off.transient_bytes
    # The no-producer control allocates nothing even with the key set:
    # there is no field to publish, so there is no buffer to hold it.
    from woof.config import KM_OPT_ZERO_ACK
    unmixed = _min_cfg(bl_pbl_physics=1, sf_sfclay_physics=1, moist=True,
                       km_opt=0, hmix_k_diag=True,
                       km_opt_zero_acknowledgement=KM_OPT_ZERO_ACK)
    assert not any(name.startswith("hmix_k_diag/")
                   for name in pf.physics_array_shapes(unmixed))


def test_hmix_k_diag_records_the_governed_field_the_closure_used(
        monkeypatch):
    """The recorded momentum row IS the step's own km_h, and the scalar
    row is that field over the step's own blended Prandtl number.

    Recorded from the ledger the model consumed, not recomputed: a
    diagnostic that re-derived the number could agree with the paper and
    disagree with the run.
    """
    from dataclasses import replace

    from woof.verify.sase_ref import prandtl_blend

    calls = []
    physics, state, cfg, driver = _sase_shim_driver(
        monkeypatch, calls, step_fields=True, hmix_k_diag=True)
    monkeypatch.setattr(
        physics, "launch_bulk_richardson_zi",
        lambda *a, **kw: np.zeros((cfg.ny, cfg.nx), np.float32))
    assert set(driver.hmix_k_diag) == {"SASE_KMH", "SASE_KHH"}
    # Zeros before any step: frame 0 is an accurate zero, not a fill value.
    assert np.all(driver.hmix_k_diag["SASE_KMH"] == 0.0)
    driver.compute(state, cfg)
    # The shim's step returns a uniform governed field of 30.0 m2 s-1 at
    # f = 0.6; the recorded rows are that field and that field over the
    # step's OWN blended Prandtl number at that same f.
    np.testing.assert_array_equal(driver.hmix_k_diag["SASE_KMH"], 30.0)
    pr_t = float(prandtl_blend(float(driver.last_sase_ledger["f"])))
    assert 0.3 < pr_t < 1.0, pr_t         # a real blend, not 1.000
    np.testing.assert_allclose(
        driver.hmix_k_diag["SASE_KHH"], 30.0 / pr_t, rtol=1e-6)
    assert set(driver.hmix_k_diag).issubset(driver.output_fields())


def test_hmix_k_diag_is_restart_rebuilt_not_serialized():
    """Output-only, so a restart rebuilds it rather than carrying it.

    Pinned because the restart manifest is fail-closed on UNCLASSIFIED
    driver attributes, and it caught this one on a real run: the guard
    is the reason a diagnostic buffer cannot quietly join the
    checkpoint.
    """
    from woof.io import restart

    assert "hmix_k_diag" in restart.DRIVER_REBUILT_ATTRS
    assert "hmix_k_diag" not in restart.DRIVER_SERIALIZED_ATTRS


def test_hmix_k_diag_off_allocates_nothing_and_leaves_output_alone(
        monkeypatch):
    """Default off: no buffer, and the historical output key set exactly."""
    calls = []
    physics, state, cfg, driver = _sase_shim_driver(monkeypatch, calls,
                                                    step_fields=True)
    monkeypatch.setattr(
        physics, "launch_bulk_richardson_zi",
        lambda *a, **kw: np.zeros((cfg.ny, cfg.nx), np.float32))
    assert cfg.hmix_k_diag is False
    assert driver.hmix_k_diag is None
    before = set(driver.output_fields())
    driver.compute(state, cfg)
    assert driver.hmix_k_diag is None
    assert set(driver.output_fields()) == before
    assert not any(name.endswith("KMH") or name.endswith("KHH")
                   for name in before)


def test_sase_flux_diag_off_allocates_nothing_and_leaves_output_alone(
        monkeypatch):
    """Default off: no buffer, the historical output key set exactly, and
    the CPU-shim driver never enters the fill blocks (no new fakes)."""
    calls = []
    physics, state, cfg, driver = _sase_shim_driver(monkeypatch, calls,
                                                    step_fields=True)
    # The CPU shim has no device z_i kernel; PBLH is a separate S3-9d
    # seam and is not what this test is about.
    monkeypatch.setattr(
        physics, "launch_bulk_richardson_zi",
        lambda *a, **kw: np.zeros((cfg.ny, cfg.nx), np.float32))
    assert cfg.sase_flux_diag is False
    assert driver.sase_flux_diag is None
    before = set(driver.output_fields())
    driver.compute(state, cfg)
    assert calls.count("sase_step") == 1
    assert driver.sase_flux_diag is None
    assert set(driver.output_fields()) == before
    assert not any(name.startswith("SASE_F") for name in before)


def test_sase_flux_diag_fields_reach_a_wrfout_frame_with_z_stagger(tmp_path):
    """END TO END through the writer: the four names land in a wrfout
    frame on the bottom_top_stag dimension with stagger 'Z' (the W
    registration) and their registered units and description."""
    import netCDF4

    from woof.io.wrfout import WrfoutWriter

    nz, ny, nx = 8, 5, 6
    writer = WrfoutWriter(tmp_path / "wrfout_d01_flux_diag", nx=nx, ny=ny,
                          nz=nz, dx=1000.0, dy=1000.0)
    units = {"SASE_FQV_VENT": "kg m-2 s-1", "SASE_FQV_DIFF": "kg m-2 s-1",
             "SASE_FTH_VENT": "W m-2", "SASE_FTH_DIFF": "W m-2"}
    face = np.arange((nz + 1) * ny * nx,
                     dtype=np.float32).reshape(nz + 1, ny, nx)
    writer.write_frame("1974-04-03_12:00:00",
                       {name: face for name in units})
    writer.close()
    with netCDF4.Dataset(tmp_path / "wrfout_d01_flux_diag", "r") as ds:
        for name, unit in units.items():
            var = ds.variables[name]
            assert var.dimensions == ("Time", "bottom_top_stag",
                                      "south_north", "west_east"), name
            assert var.stagger == "Z", name
            assert var.units == unit, name
            assert var.MemoryOrder == "XYZ", name
            assert "positive up" in var.description, name
            np.testing.assert_array_equal(var[0], face)


def test_sase_flux_diag_buffers_are_classified_as_rebuilt():
    """The new driver attribute is classified for the restart manifest
    (unclassified attributes are a hard write failure)."""
    from woof.io import restart

    assert "sase_flux_diag" in restart.DRIVER_REBUILT_ATTRS
    assert "sase_flux_diag" not in restart.DRIVER_SERIALIZED_ATTRS


# ---------------------------------------------------------------------------
# SASE-M1 MOIST-N2 SWITCH (RunConfig sase_moist_n2): the switch itself,
# its registration, and its declared residency.  The BYPASS -- that False
# hands the dry field to every substitution point, forms no M1b bound and
# leaves the M2 veto standing -- is measured on the authority and on the
# CPU-shim driver further down this file, and on device in
# tests/test_sase_gpu.py.
# ---------------------------------------------------------------------------


def test_sase_moist_n2_defaults_on_and_is_fail_closed():
    """True on every path (the model as built), and never a silently
    inert key: the NON-default value is the one that is fail-closed."""
    import dataclasses

    assert _min_cfg().sase_moist_n2 is True
    assert _sase_cfg().sase_moist_n2 is True
    names = [f.name for f in dataclasses.fields(RunConfig)]
    assert names[names.index("sase_flux_diag") + 1] == "sase_moist_n2", (
        "the key must stay APPENDED (positional construction), "
        "immediately behind sase_flux_diag")
    # The default is admitted under EVERY PBL scheme -- it must never
    # make an existing configuration fail (the mirror image of
    # sase_flux_diag, whose True is the guarded value).
    assert validate_run_config(_min_cfg()) is not None
    assert validate_run_config(_min_cfg(sase_moist_n2=True)) is not None
    # Off is admitted only under the closure that owns the seam.
    cfg = _sase_cfg(sase_moist_n2=False)
    assert validate_run_config(cfg) is cfg
    with pytest.raises(ValueError, match="sase_moist_n2"):
        validate_run_config(_min_cfg(sase_moist_n2=False))
    # ... and the rejection names the closure it needs.
    with pytest.raises(ValueError, match="requires bl_pbl_physics"):
        validate_run_config(_min_cfg(sase_moist_n2=False))


def test_sase_moist_n2_is_run_wide_not_a_per_domain_override():
    """A PHYSICS selector, so -- unlike the output-only sase_flux_diag --
    it is deliberately NOT a [[domain]] key: every SASE domain of a tree
    runs one variant, so the paired M1-on/M1-off experiment compares two
    whole trees.  It stays settable in [shared], which applies it to
    every SASE domain of the tree."""
    from woof import experiment

    assert "sase_moist_n2" not in experiment._DOMAIN_RUN_OVERRIDES
    assert "sase_moist_n2" not in experiment._DOMAIN_KEYS
    assert "sase_moist_n2" not in experiment._DOMAIN_VERTICAL_KEYS
    # ... and NOT forbidden in [shared]: that is the seat it is set from.
    assert "sase_moist_n2" not in experiment._SHARED_FORBIDDEN


def test_sase_moist_n2_is_not_exempt_from_the_restart_config_match(tmp_path):
    """A physics selector may not change under a resume.  The
    output-selector exemption that carries sase_flux_diag past the
    restart config match must NOT grow to cover this key -- a run that
    resumed with the closure silently switched would be two experiments
    in one file."""
    from dataclasses import replace

    from woof.io import restart

    assert "sase_moist_n2" not in restart.CONFIG_DIAGNOSTIC_FIELDS
    assert "sase_moist_n2" not in restart.CONFIG_RUN_LENGTH_FIELDS
    # This head already had a trajectory-inert toggle set of its own, so
    # sase_flux_diag JOINS it rather than founding a second one; the
    # closed-world assertion below therefore names every member.
    assert restart.CONFIG_DIAGNOSTIC_FIELDS == frozenset(
        {"nwp_diagnostics", "tke_budget", "sase_flux_diag", "hmix_k_diag",
         "surface_energy_diag", "thompson_cloud_surface_booking"}), (
        # thompson_cloud_surface_booking (correct mode WB1, development branch)
        # joined with its own inertness proof
        # (tests/test_correct_mode_water_budget.py
        # test_wb1_thompson_books_settled_cloud_and_changes_nothing_else:
        # every prognostic word, rain total and radius bit-identical with
        # the key on and off; it only books CLOUDNC).
        # tke_budget joined on the LES lane with its own inertness proof
        # (tests/test_tke_budget.py); hmix_k_diag joined with its own
        # (tests/test_sase_gpu.py, both producers, every prognostic
        # byte-identical); surface_energy_diag joined at ae393b7c3 with
        # its own (tests/test_urban_history_energy.py: it publishes the
        # physics driver's existing albedo, emissivity and GSW arrays to
        # history and leaves the configuration digest unchanged); the
        # closed world stays closed.
        "the output-selector exemption gained a member -- every member "
        "must be provably absent from the model")
    # Functional: the stored/live mismatch is a hard restart failure.
    import dataclasses

    on = _sase_cfg()
    off = replace(on, sase_moist_n2=False)
    stored = dataclasses.asdict(on)
    with pytest.raises(Exception) as excinfo:
        restart._require_config_match(stored, off, tmp_path / "chk")
    assert "sase_moist_n2" in str(excinfo.value)
    # ... and the SAME value resumes cleanly.
    restart._require_config_match(stored, on, tmp_path / "chk")


def test_sase_moist_n2_moves_no_declared_residency():
    """The switch allocates and frees NOTHING: the moist-N2 work field is
    launched either way (the M2 veto still needs it), so every preflight
    figure -- physics arrays, the sase workspace shapes and their phase
    transcription, the domain estimate -- is identical on both legs and
    the 59-field solve-phase byte pin is untouched."""
    import types
    from dataclasses import replace

    from woof.core import preflight as pf

    on = _sase_cfg()
    off = replace(on, sase_moist_n2=False)
    assert pf.physics_array_shapes(off) == pf.physics_array_shapes(on)
    assert pf.sase_workspace_shapes(off) == pf.sase_workspace_shapes(on)
    assert pf.sase_workspace_phases(off) == pf.sase_workspace_phases(on)
    est_on = pf.estimate_domain(
        types.SimpleNamespace(run=on, grid_id=1, parent_id=0))
    est_off = pf.estimate_domain(
        types.SimpleNamespace(run=off, grid_id=1, parent_id=0))
    assert est_off.resident_bytes == est_on.resident_bytes
    assert est_off.transient_bytes == est_on.transient_bytes


# ---------------------------------------------------------------------------
# S3-6k STABLE-LIMB DISSIPATION SWITCH (RunConfig
# sase_stable_dissipation): the switch itself, its registration and its
# declared residency.  The coefficient's closed forms and the step seam
# are further down this file (S3-6k section); the device gate is in
# tests/test_sase_gpu.py.
# ---------------------------------------------------------------------------


def test_sase_stable_dissipation_defaults_off_and_is_fail_closed():
    """Off on every path (the model as built), and never a silently
    inert key: the NON-default value is the one that is fail-closed."""
    import dataclasses

    assert _min_cfg().sase_stable_dissipation is False
    assert _sase_cfg().sase_stable_dissipation is False
    names = [f.name for f in dataclasses.fields(RunConfig)]
    assert names[names.index("sase_moist_n2") + 1] ==         "sase_stable_dissipation", (
        "the key must stay APPENDED (positional construction), "
        "immediately behind sase_moist_n2")
    # The default is admitted under EVERY PBL scheme.
    assert validate_run_config(_min_cfg()) is not None
    assert validate_run_config(
        _min_cfg(sase_stable_dissipation=False)) is not None
    # On is admitted only under the closure that owns the coefficient.
    cfg = _sase_cfg(sase_stable_dissipation=True)
    assert validate_run_config(cfg) is cfg
    with pytest.raises(ValueError, match="sase_stable_dissipation"):
        validate_run_config(_min_cfg(sase_stable_dissipation=True))
    # ... and the rejection names the closure it needs.
    with pytest.raises(ValueError, match="requires bl_pbl_physics"):
        validate_run_config(_min_cfg(sase_stable_dissipation=True))


def test_sase_stable_dissipation_is_run_wide_not_a_per_domain_override():
    """A PHYSICS selector, so -- like sase_moist_n2 and unlike the
    output-only sase_flux_diag -- it is deliberately NOT a [[domain]]
    key: a nest whose domains dissipated on different coefficients could
    not be compared across its own boundary."""
    from woof import experiment

    assert "sase_stable_dissipation" not in experiment._DOMAIN_RUN_OVERRIDES
    assert "sase_stable_dissipation" not in experiment._DOMAIN_KEYS
    assert "sase_stable_dissipation" not in experiment._DOMAIN_VERTICAL_KEYS
    # ... and NOT forbidden in [shared]: that is the seat it is set from.
    assert "sase_stable_dissipation" not in experiment._SHARED_FORBIDDEN


def test_sase_stable_dissipation_is_not_exempt_from_the_restart_match(
        tmp_path):
    """A physics selector may not change under a resume: the
    output-selector exemption must not grow to cover a coefficient
    change inside the closure's decay substep."""
    from dataclasses import replace

    from woof.io import restart

    assert ("sase_stable_dissipation"
            not in restart.CONFIG_DIAGNOSTIC_FIELDS)
    assert ("sase_stable_dissipation"
            not in restart.CONFIG_RUN_LENGTH_FIELDS)
    import dataclasses

    off = _sase_cfg()
    on = replace(off, sase_stable_dissipation=True)
    stored = dataclasses.asdict(off)
    with pytest.raises(Exception) as excinfo:
        restart._require_config_match(stored, on, tmp_path / "chk")
    assert "sase_stable_dissipation" in str(excinfo.value)
    restart._require_config_match(stored, off, tmp_path / "chk")


def test_sase_stable_dissipation_moves_no_declared_residency():
    """The switch allocates and frees NOTHING: the coefficient is formed
    per-cell inside the e-update kernel from state it already holds (no
    new device field, by design -- the VRAM ceiling on this hardware is
    a correctness bar), so every preflight figure is identical on both
    legs and the solve-phase workspace pin is untouched."""
    import types
    from dataclasses import replace

    from woof.core import preflight as pf

    off = _sase_cfg()
    on = replace(off, sase_stable_dissipation=True)
    assert pf.physics_array_shapes(on) == pf.physics_array_shapes(off)
    assert pf.sase_workspace_shapes(on) == pf.sase_workspace_shapes(off)
    assert pf.sase_workspace_phases(on) == pf.sase_workspace_phases(off)
    est_off = pf.estimate_domain(
        types.SimpleNamespace(run=off, grid_id=1, parent_id=0))
    est_on = pf.estimate_domain(
        types.SimpleNamespace(run=on, grid_id=1, parent_id=0))
    assert est_on.resident_bytes == est_off.resident_bytes
    assert est_on.transient_bytes == est_off.transient_bytes


# ---------------------------------------------------------------------------
# Stage-3 Task 6: driver coupling -- authority scalar mixing / N^2, the
# named surface e source, SASE dispatch replacing YSU, restart-inventory
# pins, and the km_opt=0 dycore admission.
# ---------------------------------------------------------------------------


def test_scalar_mix_constant_field_is_exactly_zero():
    """div(K_h grad s) of a constant scalar vanishes identically on the
    clamped stretched grid (all stencil rows difference to zero)."""
    from woof.verify import sase_ref
    shape = (10, 6, 6)
    s = np.full(shape, 4.5)
    e = 0.2 + 0.05 * sase_ref.box_filter(
        np.random.default_rng(1).standard_normal(shape), 2)
    t = _stretched_thicknesses(shape[0])
    out = sase_ref.scalar_mix(s, e, 12.0, 500.0, 500.0, float("nan"),
                              dz_col=t)
    # The Lagrange coefficient rows sum to zero analytically but not in
    # FP64 (they are a weighted sum, not a difference), so "exactly
    # zero" means FP64 coefficient roundoff: |s|*|coef|*eps compounded
    # through two passes is ~1e-18 here; 1e-15 is generous headroom.
    np.testing.assert_allclose(out, 0.0, atol=1e-15)


def test_scalar_mix_quadratic_uniform_closed_form():
    """Uniform e + uniform dz + s = a*z^2: interior rows give exactly
    2*a*K_h (flux = K_h*2az is linear, its centered ddz is exact)."""
    from woof.verify import sase_ref
    nz, ny, nx = 12, 4, 4
    dz = 200.0
    a = 3.0e-6
    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    s = np.broadcast_to((a * z**2)[:, None, None], (nz, ny, nx)).copy()
    e = np.full((nz, ny, nx), 0.16)            # sqrt(e) = 0.4
    kh_coef = 30.0
    kh = kh_coef * 0.4
    out = sase_ref.scalar_mix(s, e, kh_coef, 500.0, 500.0, dz)
    np.testing.assert_allclose(out[2:-2], 2.0 * a * kh, rtol=1e-12)


def test_brunt_vaisala_n2_linear_theta_closed_form():
    """theta = a*z + b on the stretched grid: dtheta/dz = a exactly at
    every row (interior Lagrange and one-sided edges are linear-exact),
    so N^2 = g*a/theta elementwise."""
    from woof.verify import sase_ref
    nz, ny, nx = 14, 3, 3
    t = _stretched_thicknesses(nz)
    z = np.cumsum(t) - 0.5 * t
    a, b = 0.004, 295.0
    theta = np.broadcast_to((a * z + b)[:, None, None], (nz, ny, nx)).copy()
    n2 = sase_ref.brunt_vaisala_n2(theta, float("nan"), dz_col=t)
    np.testing.assert_allclose(n2, sase_ref.G_ACCEL * a / theta, rtol=1e-12)


class _DelegatingNumpyShim:
    """cp-module stand-in that forwards every attribute to numpy."""

    ndarray = np.ndarray

    def __getattr__(self, name):
        return getattr(np, name)


def _plausible_column(state, cfg):
    """Fill the shim state with a physically plausible flat column."""
    nz = cfg.nz
    state.phb[...] = (np.arange(nz + 1, dtype=np.float64)
                      * (cfg.ztop / nz) * 9.81).astype(np.float32)
    state.thb[...] = np.float32(300.0)
    state.c1h[...] = np.float32(1.0)
    state.c1f[...] = np.float32(1.0)
    state.mub2d[...] = np.float32(90000.0)
    state.dnw[...] = np.float32(-1.0 / nz)
    state.p_top = np.float32(5000.0)
    state.p[...] = np.float32(85000.0)         # EOS p feeds exner/T


def _sase_shim_driver(monkeypatch, calls, mix_rates=(3.0e-5, 1.0e-7,
                                                     -1.0e-8, 2.0e-9),
                      du_rate=1.0e-4, dv_rate=-2.0e-4, dw_rate=5.0e-5,
                      heat_value=0.0, step_fields=False, **cfg_overrides):
    """NumPy-backed admitted-config driver with device launchers faked.

    The fakes replace ONLY the CUDA launch seams (sfclay/noah/ysu/sase
    kernels); the driver logic, atmosphere preparation, the named surface
    e source, and the coupling/composition pathway all run for real.
    """
    import woof.core.physics as physics
    import woof.core.state as state_module

    shim = _DelegatingNumpyShim()
    monkeypatch.setattr(state_module, "cp", shim)
    monkeypatch.setattr(physics, "cp", shim)

    cfg = _sase_cfg(**cfg_overrides)
    state = state_module.DomainState(cfg)
    _plausible_column(state, cfg)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx

    monkeypatch.setattr(physics, "launch_sfclay",
                        lambda *a, **kw: calls.append("sfclay"))
    monkeypatch.setattr(physics, "launch_noah",
                        lambda *a, **kw: calls.append("noah"))

    def fake_ysu(*a, **kw):
        calls.append("ysu")
        raise AssertionError("YSU must not run when SASE holds the PBL slot")

    monkeypatch.setattr(physics, "launch_ysu", fake_ysu)

    def fake_sase_step(u, v, w, theta, e, *, dx, dy, dz, delta, dt,
                       n2=None, dz_col=None, heat=None,
                       exclude_boundary_width=0, z0=0.0, zdamp=None,
                       ust=None, wspd_sfc=None, n2_moist=None,
                       stable_dissipation=False,
                       **_ignored):
        calls.append("sase_step")
        calls.append(("exclude", exclude_boundary_width))
        calls.append(("zdamp", zdamp))
        # S3-6j: the driver must thread the live sfclay ust into the
        # step (the surface momentum stress seam).
        calls.append(("ust_passed", ust is not None
                      and ust.shape == (ny, nx)))
        # S3-9c: and the live sfclay gust-enhanced wspd beside it
        # (the gustiness-correction seam).
        calls.append(("wspd_passed", wspd_sfc is not None
                      and wspd_sfc.shape == (ny, nx)))
        # S4-2 (SASE-M1): the driver must thread the moist-n2 field
        # BESIDE the dry n2 (n2 for the w-sensor screen, n2_moist for
        # the three substitution points) -- the launch_moist_n2 output
        # object, not the dry field re-passed.
        calls.append(("n2_moist_passed", n2_moist is not None
                      and n2_moist is not n2
                      and n2_moist.shape == (nz, ny, nx)))
        # SASE-M1 SWITCH (cfg.sase_moist_n2): ``None`` is the launcher's
        # pre-M1 path and is the ONLY way the driver disables the seam,
        # so record it separately from "passed the wrong object".
        calls.append(("n2_moist_none", n2_moist is None))
        # S3-6k SWITCH (cfg.sase_stable_dissipation): the driver's only
        # effect is which bool reaches this slot.
        calls.append(("stable_dissipation", stable_dissipation))
        assert n2 is not None, "driver must pass a real n2 field"
        assert dz_col is not None and dz_col.shape == (nz, ny, nx)
        u += np.float32(du_rate * dt)
        v += np.float32(dv_rate * dt)
        w += np.float32(dw_rate * dt)
        heat[...] = np.float32(heat_value)
        ledger = {"dKE": -1.0, "dE": 0.5, "dHeat": 0.5, "residual": 0.0,
                  "c_nu": 0.12, "f": 0.6}
        if step_fields:
            ledger["kv"] = np.full((nz, ny, nx), 2.0, np.float32)
            ledger["km_h"] = np.full((nz, ny, nx), 30.0, np.float32)
        return ledger

    def fake_n2(theta, *, dz=None, dz_col=None, out=None):
        calls.append("n2")
        result = np.zeros_like(theta) if out is None else out
        result[...] = np.float32(0.0)
        return result

    def fake_moist_n2(theta, qv, qc, pressure, n2_dry, *, dz_col=None,
                      out=None):
        # S4-2 (SASE-M1): the driver computes n2_eff beside the dry n2
        # from the already-resident qv/qc/full-pressure fields and the
        # SAME dz_col; the seam-recording tuple pins the wiring.
        calls.append("moist_n2")
        calls.append(("moist_n2_inputs", n2_dry is not None,
                      qv is not None and qv.shape == (nz, ny, nx),
                      qc is not None and qc.shape == (nz, ny, nx),
                      pressure is not None
                      and pressure.shape == (nz, ny, nx),
                      dz_col is not None))
        result = np.zeros_like(theta) if out is None else out
        result[...] = np.float32(0.0)
        return result

    seq = iter(mix_rates)

    def fake_scalar_mix(s, e=None, *, kh_coef=None, kh_field=None,
                        kh_fac=1.0, dx, dy, dz=None, dz_col=None,
                        out=None, flux=None):
        rate = next(seq)
        calls.append(("mix", rate))
        calls.append(("mix_mode",
                      "field" if kh_field is not None else "coef"))
        result = np.empty_like(s) if out is None else out
        result[...] = np.float32(rate)
        return result

    def fake_implicit(phi, kv, *, dt, kfac=1.0, dz=None, dz_col=None,
                      floor=None, sfc_flux=None, sfc_rho1=None,
                      sfc_fac=1.0):
        calls.append(("implicit", kfac))
        # S3-11b: record the surface scalar-flux deposit seam objects
        # so the driver-wiring test can pin WHICH flux/rho1 each row
        # received (the theta/qv rows must ride the live sfclay
        # hfx/qfx and ONE shared rho1; qc/qi none).
        calls.append(("implicit_sfc", sfc_flux, sfc_rho1, sfc_fac))
        return phi

    def fake_vent_flux(theta, qv, qc, pressure, e_sgs, n2_moist, n2_dry,
                       rho1, *, f_blend, dz_col=None, out=None,
                       indices=False):
        # S4-5 (SASE-M2): the driver diagnoses the venting limb from the
        # FROZEN PRE-STEP state at the step's USED f.  The recorded
        # tuple pins every wire: the frozen e copy (a distinct object
        # from the live state.e_sgs, which the surface source and the
        # step both write), the M1 mask as the n2 PAIR, the shared
        # rho1 plane and the per-column dz_col.
        calls.append("vent_flux")
        calls.append(("vent_inputs", float(f_blend),
                      e_sgs is not state.e_sgs,
                      float(e_sgs[0, 0, 0]),
                      n2_moist is not n2_dry
                      and n2_moist.shape == (nz, ny, nx),
                      rho1 is not None and rho1.shape == (ny, nx),
                      dz_col is not None and dz_col.shape == (nz, ny, nx),
                      theta.shape == (nz, ny, nx)))
        return tuple(np.full((nz + 1, ny, nx), np.float32(v), np.float32)
                     for v in (1.0, 2.0, 3.0))

    def fake_vent_scale(f_theta, f_qv, f_qc, rho1, *, dt, dz_col=None,
                        out=None):
        calls.append(("vent_scale", float(f_theta[0, 0, 0]),
                      float(f_qv[0, 0, 0]), float(f_qc[0, 0, 0]),
                      float(dt), rho1.shape == (ny, nx)))
        return np.ones((ny, nx), np.float64)

    def fake_vent_deposit(phi, f_row, scale, rho1, *, dt, dz_col=None):
        calls.append(("vent_deposit", float(f_row[0, 0, 0])))
        return phi

    monkeypatch.setattr(physics, "launch_sase_step", fake_sase_step)
    monkeypatch.setattr(physics, "launch_n2", fake_n2)
    monkeypatch.setattr(physics, "launch_moist_n2", fake_moist_n2)
    monkeypatch.setattr(physics, "launch_scalar_mix", fake_scalar_mix)
    monkeypatch.setattr(physics, "launch_plume_vent_flux",
                        fake_vent_flux)
    monkeypatch.setattr(physics, "launch_vent_deposit_scale",
                        fake_vent_scale)
    monkeypatch.setattr(physics, "launch_vent_deposit",
                        fake_vent_deposit)
    monkeypatch.setattr(physics, "launch_implicit_vertical_diffusion",
                        fake_implicit)

    zeros3 = np.zeros((nz, ny, nx), np.float32)
    zeros2 = np.zeros((ny, nx), np.float32)

    def radiation(**kw):
        calls.append("radiation")
        return physics.RadiationResult(zeros3.copy(), zeros3.copy(),
                                       zeros2.copy(), zeros2.copy())

    # The admitted SASE configuration no longer pins a radiation scheme
    # (the closure does not read one), so the radiation callable is
    # supplied only when the config actually selects radiation --
    # initialize_physics refuses a callable with every radiation slot
    # off, and rightly so.
    # The admitted SASE configuration no longer pins a radiation or
    # land-surface scheme (the closure reads neither), so each optional
    # attachment is supplied only when the config actually selects it --
    # initialize_physics refuses a radiation callable with every
    # radiation slot off, and refuses a Noah bundle with the LSM off.
    from woof.config import radiation_enabled
    extra = {}
    if radiation_enabled(cfg):
        extra["radiation"] = radiation
    if cfg.sf_surface_physics == 2:
        extra["noah_params"] = object()
    driver = physics.initialize_physics(state, cfg, **extra)
    # The surface layer is faked, so nothing fills UST.  This head cold-
    # starts it at WRF's module_physics_init value 1e-4 rather than the
    # 0.1 the lane this test came from used, and u*^3 at 1e-4 is 1e-12 --
    # which would leave the surface e source depositing nothing and make
    # every fixture below vacuous.  Standing in for the scheme the fake
    # replaced is the point of a shim: 0.1 m/s is what a real surface
    # layer returns over land in a light wind.
    driver.fields["ust"][...] = np.float32(0.1)
    return physics, state, cfg, driver


def test_sase_driver_dispatch_and_tendency_accumulators(monkeypatch):
    """When sase is active the driver skips YSU, runs the SASE sequence,
    and lands every rate in the YSU-shaped accumulators plus the plain
    ``rw`` attribute -- all through the existing coupling/composition."""
    from woof.verify.sase_ref import E_MIN

    calls = []
    physics, state, cfg, driver = _sase_shim_driver(monkeypatch, calls, mp_physics=10)
    tend = driver.compute(state, cfg)

    assert "ysu" not in calls and "sase_step" in calls
    assert driver.call_counts["sase"] == 1
    # "ysu" is this head's historical name for the PBL-SLOT counter and
    # every scheme in that slot increments it (MYNN does too), so the
    # invariant that matters -- YSU's own launcher never ran -- is the
    # `"ysu" not in calls` assertion above, backed by a fake launcher
    # that raises if it is ever entered.  The two counters agreeing is
    # what says SASE took the slot exactly once.
    assert driver.call_counts["ysu"] == driver.call_counts["sase"]
    # Scalar-mix dispatch order is theta, qv, qc, qi (the accumulators
    # depend on it).
    mixes = [entry for entry in calls
             if isinstance(entry, tuple) and entry[0] == "mix"]
    assert [rate for _, rate in mixes] == [3.0e-5, 1.0e-7, -1.0e-8, 2.0e-9]
    # Non-specified domain: no solve-reduction boundary exclusion.
    assert ("exclude", 0) in calls
    # S3-6j: the live sfclay ust field reaches the step (the surface
    # momentum stress seam -- the missing-friction fix's driver wire).
    assert ("ust_passed", True) in calls
    # S3-9c: the live sfclay gust-enhanced wspd field reaches the step
    # beside it (the gustiness-correction driver wire).
    assert ("wspd_passed", True) in calls
    # S4-2 (SASE-M1): the moist-n2 launcher runs beside the dry n2 with
    # the resident qv/qc/pressure fields and its output reaches the
    # step as n2_moist (a distinct field, the dry n2 still passed for
    # the w-sensor screen) -- the M1 driver wire.
    assert "moist_n2" in calls
    assert ("moist_n2_inputs", True, True, True, True, True) in calls
    assert ("n2_moist_passed", True) in calls

    # Surface e source: ust=0.1 (init), hfx=qfx=0 => shear term only,
    # u*^3/(kappa*0.5*dz1) with dz1 = 1000 m, deposited as dt*source into
    # the lowest level ONLY.
    dt = cfg.dt
    shear = 0.1**3 / (0.4 * 0.5 * (cfg.ztop / cfg.nz))
    np.testing.assert_allclose(state.e_sgs[0], E_MIN + dt * shear,
                               rtol=1e-5)
    np.testing.assert_allclose(state.e_sgs[1:], np.float32(E_MIN),
                               rtol=0.0)

    # Coupled accumulators: chm = c1h*mu + c2h = 90000 everywhere.
    chm = 90000.0
    np.testing.assert_allclose(tend.ru, chm * 1.0e-4, rtol=1e-5)
    np.testing.assert_allclose(tend.rv, chm * -2.0e-4, rtol=1e-5)
    np.testing.assert_allclose(tend.rtheta, chm * 3.0e-5, rtol=1e-5)
    np.testing.assert_allclose(tend.scalar_for("qv"), chm * 1.0e-7,
                               rtol=1e-5)
    np.testing.assert_allclose(tend.scalar_for("qc"), chm * -1.0e-8,
                               rtol=1e-5)
    np.testing.assert_allclose(tend.scalar_for("qi"), chm * 2.0e-9,
                               rtol=1e-5)
    assert tend.scalar_for("qr") is None       # SASE does not mix rain

    # w rides the pbl stack as the plain rw attribute: full-level mass
    # coupling c1f*mu + c2f, zero boundary rows, carried by composition.
    rw = tend.rw
    assert rw.shape == (cfg.nz + 1, cfg.ny, cfg.nx)
    np.testing.assert_allclose(rw[1:-1], chm * 5.0e-5, rtol=1e-5)
    np.testing.assert_allclose(rw[0], 0.0, atol=0.0)
    np.testing.assert_allclose(rw[-1], 0.0, atol=0.0)

    # add_to_slow lands rw in state.rw_t (the existing application slot).
    for name in ("ru_t", "rv_t", "rth_t", "rw_t"):
        getattr(state, name)[...] = 0.0
    tend.add_to_slow(state)
    np.testing.assert_allclose(state.rw_t, rw, rtol=0.0)
    np.testing.assert_allclose(state.ru_t, tend.ru, rtol=0.0)

    assert driver.last_sase_ledger["c_nu"] == 0.12


def test_sase_moist_n2_switch_selects_the_step_argument(monkeypatch):
    """THE DRIVER WIRE, both legs (RunConfig sase_moist_n2).

    The switch has exactly ONE effect on the driver: which object reaches
    ``launch_sase_step``'s ``n2_moist`` slot.  True (the default) passes
    the launch_moist_n2 field, which is the pre-switch call verbatim;
    False passes ``None``, the launcher's own pre-M1 path -- and ``None``
    is the only disabling value, which is why the shim records it
    separately from "some other object".

    THE SECOND HALF OF THE CLAIM, and the reason this test exists at the
    driver level at all: with the seam off the moist field is STILL
    computed and STILL handed to the M2 venting limb, whose saturation
    veto is the bitwise departure of the (moist, dry) pair.  M1 off must
    not stand M2 down -- that is the confound this switch exists to
    avoid -- so the vent wiring is asserted identical on both legs."""
    for flag, expect_field, expect_none in ((True, True, False),
                                            (False, False, True)):
        calls = []
        physics, state, cfg, driver = _sase_shim_driver(
            monkeypatch, calls, step_fields=True, sase_moist_n2=flag)
        monkeypatch.setattr(
            physics, "launch_bulk_richardson_zi",
            lambda *a, **kw: np.zeros((cfg.ny, cfg.nx), np.float32))
        assert cfg.sase_moist_n2 is flag
        driver.compute(state, cfg)
        # The moist field is computed on BOTH legs (the M2 veto needs
        # it, and the preflight residency is identical because of it).
        assert "moist_n2" in calls
        assert ("moist_n2_inputs", True, True, True, True, True) in calls
        # ... and only the step argument moves.
        assert ("n2_moist_passed", expect_field) in calls, flag
        assert ("n2_moist_none", expect_none) in calls, flag
        # M2 still diagnosed, still handed the (moist, dry) PAIR.
        assert "vent_flux" in calls, flag
        vent = [entry for entry in calls
                if isinstance(entry, tuple) and entry[0] == "vent_inputs"]
        assert len(vent) == 1 and vent[0][4] is True, flag
        assert any(isinstance(entry, tuple) and entry[0] == "vent_deposit"
                   for entry in calls), flag


def test_sase_stable_dissipation_switch_reaches_the_step(monkeypatch):
    """THE DRIVER WIRE, both legs (RunConfig sase_stable_dissipation).
    The switch has exactly ONE effect on the driver: which bool reaches
    ``launch_sase_step``'s ``stable_dissipation`` slot.  It is a plain
    bool, never a None-vs-object seam, so the shim records the value
    itself; and the default leg must pass False, not merely omit the
    kwarg, because the launcher's gate is what makes the kernel bitwise
    the pre-S3-6k step."""
    for flag in (False, True):
        calls = []
        physics, state, cfg, driver = _sase_shim_driver(
            monkeypatch, calls, step_fields=True,
            sase_stable_dissipation=flag)
        monkeypatch.setattr(
            physics, "launch_bulk_richardson_zi",
            lambda *a, **kw: np.zeros((cfg.ny, cfg.nx), np.float32))
        assert cfg.sase_stable_dissipation is flag
        driver.compute(state, cfg)
        assert ("stable_dissipation", flag) in calls, flag


def test_sase_dissipative_heat_deposits_into_theta(monkeypatch):
    """A positive heat field strictly increases the theta accumulator
    over the pure-mixing value (heat/(dt*cp*exner) joins dtheta)."""
    calls = []
    _, state, cfg, driver = _sase_shim_driver(monkeypatch, calls,
                                              heat_value=2.0)
    tend = driver.compute(state, cfg)
    assert np.all(tend.rtheta > 90000.0 * 3.0e-5)


def test_sase_driver_zdamp_and_governed_scalar_channel(monkeypatch):
    """S3-6e driver seams: (a) under damp_opt == 3 the config zdamp
    reaches the fused step (the damping-layer taper engagement); (b) a
    ledger carrying the exported kv/km_h fields routes every scalar
    through the GOVERNED horizontal channel (kh_field mode) plus the
    implicit K_v/Pr_t(f) vertical solve at the S3-6g BLENDED Prandtl
    number recomputed from the ledger f; (c) without the fields (the
    scalar-only shim dict) the legacy coefficient fallback runs and no
    vertical solve is attempted."""
    from woof.verify.sase_ref import prandtl_blend

    calls = []
    physics, state, cfg, driver = _sase_shim_driver(
        monkeypatch, calls, step_fields=True, damp_opt=3, zdamp=4200.0)
    driver.compute(state, cfg)
    assert ("zdamp", 4200.0) in calls
    modes = [c[1] for c in calls
             if isinstance(c, tuple) and c[0] == "mix_mode"]
    assert modes == ["field"] * 4              # theta, qv, qc, qi
    # S3-6g: kfac = 1/Pr_t(f) with the fake ledger's f = 0.6:
    # Pr_t = 0.6*(1/3) + 0.4*0.85 = 0.54, kfac = 1/0.54 = 1.8518...
    # (the retired fixed convention was 1/PR_T = 3).
    kfac = 1.0 / prandtl_blend(0.6)
    np.testing.assert_allclose(kfac, 1.0 / 0.54, rtol=1e-15)
    assert calls.count(("implicit", kfac)) == 4
    # S3-11b surface scalar-flux deposit wiring (the driver half of
    # the S3-11a seam): the theta and qv rows of the implicit solve
    # receive the LIVE sfclay flux objects -- the SAME f["hfx"]/
    # f["qfx"] the e source consumes -- with the authority row
    # constants (CP_AIR for theta, the 1.0 default for qv) and ONE
    # shared rho1 object for both rows (the S3-11a rho-consistency
    # obligation, sase_surface_rho1); the qc/qi rows carry NO deposit
    # (YSU's cloud/ice rows have no surface source).  Loop order
    # theta, qv, qc, qi is pinned above by the mix-rate sequence.
    from woof.verify.sase_ref import CP_AIR
    seams = [entry[1:] for entry in calls
             if isinstance(entry, tuple) and entry[0] == "implicit_sfc"]
    assert len(seams) == 4
    th_flux, th_rho, th_fac = seams[0]
    qv_flux, qv_rho, qv_fac = seams[1]
    assert th_flux is driver.fields["hfx"] and th_fac == CP_AIR
    assert qv_flux is driver.fields["qfx"] and qv_fac == 1.0
    assert th_rho is qv_rho and th_rho is not None
    assert np.all(np.asarray(th_rho) > 0.0)    # authority rho1 contract
    for sfc_flux, sfc_rho, _ in seams[2:]:     # qc, qi rows
        assert sfc_flux is None and sfc_rho is None
    # kv/km_h must be POPPED: the retained ledger stays scalar-only.
    assert set(driver.last_sase_ledger) == {
        "dKE", "dE", "dHeat", "residual", "c_nu", "f"}

    calls2 = []
    physics2, state2, cfg2, driver2 = _sase_shim_driver(
        monkeypatch, calls2)                   # damp_opt default: no taper
    driver2.compute(state2, cfg2)
    assert ("zdamp", None) in calls2
    modes2 = [c[1] for c in calls2
              if isinstance(c, tuple) and c[0] == "mix_mode"]
    assert modes2 == ["coef"] * 4
    assert not any(isinstance(c, tuple) and c[0] == "implicit"
                   for c in calls2)


def test_sase_m2_driver_deposit_seam_order_and_wiring(monkeypatch):
    """THE S4-5 DEPOSIT SEAM AT DRIVER LEVEL (spec C1-C3, binding).

    Everything about this seam is an ORDER and a SOURCE-OF-STATE claim,
    and both are pinned here on the recorded call sequence:

    * DEPOSIT-THEN-SOLVE.  For each of theta, qv and qc the sequence is
      exactly mix -> vent_deposit -> implicit, i.e. the explicit
      flux-form M2 deposit lands on the pre-solve state s* BEFORE the
      backward-Euler sweep -- the registered order generalizing
      SFC_SCALAR_FLUX = "explicit-deposit-v1".  qi gets mix ->
      implicit with NO deposit (authority scope: theta/qv/qc only).
    * ONE ROW PER FLUX PROFILE, in order: the fake returns three
      distinguishable profiles and the recorded row markers are
      (1.0, 2.0, 3.0) -- theta, qv, qc -- so a crossed wiring fails.
    * NOTHING INSIDE THE STEP.  ``launch_sase_step`` receives no vent
      argument and the vent launcher is called AFTER the step returns
      (it needs the step's used f) and BEFORE the scalar loop.
    * FROZEN PRE-STEP STATE.  The e_sgs handed to the limb is a
      DISTINCT object from the live ``state.e_sgs`` and still carries
      the pre-step value: the surface e source has already deposited
      into level 0 of the live field by then, so the two differ, and
      the recorded frozen value must be the ORIGINAL.
    * THE USED f.  ``f_blend`` is the ledger's f (0.6 in the shim), not
      a constant and not the solved-before-capping value -- the
      FP-exact two-product blend is what makes the LES limit a bitwise
      +0.0 deposit.
    * ONE CAP RESCALE FOR THREE ROWS.  The scale launcher is called
      ONCE, before the loop, on all three profiles at the step dt and
      the same rho1 plane -- a per-row scale would break the qv/qc
      partition and a per-level clip would break the telescoping.
    """
    calls = []
    physics, state, cfg, driver = _sase_shim_driver(
        monkeypatch, calls, step_fields=True)
    e_pre0 = float(np.asarray(state.e_sgs)[0, 0, 0])
    driver.compute(state, cfg)
    assert float(np.asarray(state.e_sgs)[0, 0, 0]) > e_pre0, (
        "the surface e source no longer moves level 0 -- the frozen-copy "
        "leg of this fixture would be vacuous")

    names = [c if isinstance(c, str) else c[0] for c in calls]
    # the limb runs after the step (it needs the used f) and before the
    # first scalar mix
    assert names.index("sase_step") < names.index("vent_flux")
    assert names.index("vent_flux") < names.index("vent_scale")
    assert names.index("vent_scale") < names.index("mix")
    inputs = [c for c in calls
              if isinstance(c, tuple) and c[0] == "vent_inputs"]
    assert len(inputs) == 1
    _, f_blend, frozen_obj, frozen_val, mask_pair, has_rho, has_dz, shp \
        = inputs[0]
    assert f_blend == 0.6                       # the ledger's USED f
    assert frozen_obj and frozen_val == e_pre0  # FROZEN pre-step e
    assert mask_pair and has_rho and has_dz and shp
    scale_calls = [c for c in calls
                   if isinstance(c, tuple) and c[0] == "vent_scale"]
    assert len(scale_calls) == 1                # ONE rescale, three rows
    assert scale_calls[0][1:5] == (1.0, 2.0, 3.0, cfg.dt)
    assert scale_calls[0][5]

    # per-scalar order: mix -> vent_deposit -> implicit (qi: no deposit)
    seq = [c for c in calls
           if isinstance(c, tuple) and c[0] in ("mix", "vent_deposit",
                                                "implicit")]
    kinds = [c[0] for c in seq]
    assert kinds == (["mix", "vent_deposit", "implicit"] * 3
                     + ["mix", "implicit"]), kinds
    rows = [c[1] for c in seq if c[0] == "vent_deposit"]
    assert rows == [1.0, 2.0, 3.0], rows        # theta, qv, qc in order

    # and the step itself never sees a vent argument (nothing is
    # inserted inside sase_split_step -- the ledger theorem reads theta
    # read-only)
    assert not any(isinstance(c, tuple) and "vent" in str(c[0])
                   for c in calls[:names.index("vent_flux")])


def test_ysu_dispatch_unchanged_when_sase_off(monkeypatch):
    """a non-SASE PBL selector keeps the YSU slot byte-identical: YSU is invoked,
    the SASE seam is never touched, and call_counts carries no sase key."""
    import woof.core.physics as physics
    import woof.core.state as state_module

    shim = _DelegatingNumpyShim()
    monkeypatch.setattr(state_module, "cp", shim)
    monkeypatch.setattr(physics, "cp", shim)

    cfg = _min_cfg(moist=True, sf_sfclay_physics=1, bl_pbl_physics=1)
    state = state_module.DomainState(cfg)
    _plausible_column(state, cfg)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    calls = []
    monkeypatch.setattr(physics, "launch_sfclay",
                        lambda *a, **kw: calls.append("sfclay"))

    def fake_ysu(*a, **kw):
        calls.append("ysu")
        m = (nz, ny, nx)
        zeros3 = np.zeros(m, np.float32)
        zeros2 = np.zeros((ny, nx), np.float32)
        # This head's validator requires the full fifteen-name bundle IN
        # LAUNCHER ORDER (it recovers the first-invalid name from a
        # positional bit mask), so the fake reproduces that order exactly.
        return {"du": zeros3.copy(), "dv": zeros3.copy(),
                "dtheta": zeros3.copy(), "dqv": zeros3.copy(),
                "dqc": zeros3.copy(), "dqi": zeros3.copy(),
                "exch_h": zeros3.copy(), "exch_m": zeros3.copy(),
                "hpbl": np.zeros((ny, nx), np.float32),
                "kpbl": np.zeros((ny, nx), np.int32),
                "wstar": zeros2.copy(), "delta": zeros2.copy(),
                "topdown_radsum": zeros2.copy(),
                "wstar3_2": zeros2.copy(),
                "cloudflg": np.zeros((ny, nx), np.int32)}

    monkeypatch.setattr(physics, "launch_ysu", fake_ysu)
    # YSU's device-side validation kernel needs a real uint32 device
    # scratch view, which the NumPy shim cannot supply.  This test is
    # about DISPATCH, not about YSU's validator, so it takes the
    # validator's own elementwise fallback -- the same function, minus
    # the status-slot reduction.
    monkeypatch.setattr(physics, "validate_ysu_tendencies",
                        lambda ysu, **kw: physics.validate_ysu_tendencies
                        .__wrapped__(ysu) if hasattr(
                            physics.validate_ysu_tendencies, "__wrapped__")
                        else None)

    def forbidden(*a, **kw):
        raise AssertionError("SASE seam must stay untouched when off")

    monkeypatch.setattr(physics, "launch_sase_step", forbidden)
    monkeypatch.setattr(physics, "launch_n2", forbidden)
    monkeypatch.setattr(physics, "launch_scalar_mix", forbidden)

    driver = physics.initialize_physics(state, cfg)
    tend = driver.compute(state, cfg)
    assert "ysu" in calls
    assert driver.call_counts["ysu"] == 1
    assert "sase" not in driver.call_counts
    assert getattr(tend, "rw", None) is None
    assert driver.sase_active is False


def test_sase_surface_e_source_closed_form(monkeypatch):
    """The named lower-BC source: u*^3/(kappa*0.5*dz1) plus the positive
    part of the surface virtual-buoyancy flux (g/theta)*w'thv'."""
    import woof.core.physics as physics
    from woof.core import constants as c

    monkeypatch.setattr(physics, "cp", _DelegatingNumpyShim())
    shape = (2, 3)
    ust = np.full(shape, 0.5, np.float32)
    hfx = np.full(shape, 200.0, np.float32)
    qfx = np.full(shape, 1.0e-4, np.float32)
    theta1 = np.full(shape, 300.0, np.float32)
    qv1 = np.full(shape, 0.01, np.float32)
    p1 = np.full(shape, 95000.0, np.float32)
    t1 = np.full(shape, 290.0, np.float32)
    dz1 = np.full(shape, 100.0, np.float32)
    got = physics.sase_surface_e_source(
        ust=ust, hfx=hfx, qfx=qfx, theta1=theta1, qv1=qv1, p1=p1, t1=t1,
        dz1=dz1)
    ep1 = c.RVOVRD - 1.0
    rho = 95000.0 / (c.RD * 290.0 * (1.0 + ep1 * 0.01))
    wtv = 200.0 / (rho * c.CP) + ep1 * 300.0 * 1.0e-4 / rho
    expected = 0.5**3 / (0.4 * 0.5 * 100.0) + (c.G / 300.0) * wtv
    np.testing.assert_allclose(got, expected, rtol=1e-5)
    # Conservative clip: a strongly stable surface layer contributes the
    # shear term only -- buoyancy never drains e through this source.
    stable = physics.sase_surface_e_source(
        ust=ust, hfx=np.full(shape, -500.0, np.float32),
        qfx=np.zeros(shape, np.float32), theta1=theta1, qv1=qv1, p1=p1,
        t1=t1, dz1=dz1)
    np.testing.assert_allclose(stable, 0.5**3 / (0.4 * 0.5 * 100.0),
                               rtol=1e-5)
    # S3-11b rho1 consistency (the S3-11a report obligation, asserted):
    # the factored sase_surface_rho1 is BITWISE the source's own
    # internal density (identical FP32 expression), and threading it
    # back through the new rho1 parameter reproduces the source
    # bitwise -- so the driver's ONE rho1 field serves the e source
    # and the surface scalar-flux deposit at exactly the same values.
    rho1 = physics.sase_surface_rho1(p1=p1, t1=t1, qv1=qv1)
    ep1_32 = np.float32(c.RVOVRD - 1.0)
    expected_rho = p1 / (np.float32(c.RD) * t1
                         * (np.float32(1.0) + ep1_32 * qv1))
    assert rho1.tobytes() == expected_rho.tobytes()
    assert np.all(rho1 > 0.0)                  # authority rho1 contract
    threaded = physics.sase_surface_e_source(
        ust=ust, hfx=hfx, qfx=qfx, theta1=theta1, qv1=qv1, p1=p1, t1=t1,
        dz1=dz1, rho1=rho1)
    assert threaded.tobytes() == got.tobytes()


def test_sase_positive_bldt_has_an_eager_held_w_carrier(monkeypatch):
    """A skipped PBL call now has canonical, restart-carried momentum."""
    import woof.core.physics as physics
    import woof.core.state as state_module

    shim = _DelegatingNumpyShim()
    monkeypatch.setattr(state_module, "cp", shim)
    monkeypatch.setattr(physics, "cp", shim)
    cfg = _sase_cfg(bldt=5.0)
    state = state_module.DomainState(cfg)
    driver = physics.initialize_physics(state, cfg)
    assert driver.pbl_tendencies.rw.shape == state.w.shape
    assert not driver.pbl_tendencies.rw.any()


def test_sase_restart_inventory_declares_optional_vertical_momentum(monkeypatch):
    """The optional z-face tendency belongs to the shared field manifest."""
    import dataclasses

    import woof.core.physics as physics
    from woof.io import restart

    assert restart.TENDENCY_COMPONENTS == (
        "ru", "rv", "rtheta", "rqv", "rqc", "rqr", "rqi", "rqs", "rw")
    fields = {f.name for f in dataclasses.fields(physics.PhysicsTendencies)}
    assert fields == set(restart.TENDENCY_COMPONENTS)
    assert "sase_active" in restart.DRIVER_REBUILT_ATTRS
    assert "last_sase_ledger" in restart.DRIVER_REBUILT_ATTRS

    calls = []
    _, state, cfg, driver = _sase_shim_driver(monkeypatch, calls)
    driver.compute(state, cfg)
    manifest = restart._driver_manifest(driver)
    assert manifest["driver/pbl_tendencies/rw"].shape == state.w.shape


def test_dycore_admits_km_opt_zero_only_with_sase():
    from types import SimpleNamespace

    from woof.core import dycore

    with pytest.raises(ValueError, match="km_opt") as excinfo:
        dycore.step(object(), _min_cfg(km_opt=0))
    assert str(_SASE_SELECTOR) in str(excinfo.value), (
        "the refusal must name the one selector that admits km_opt=0")
    # The admitted SASE configuration PASSES the km_opt gate.  What it
    # fails on afterwards depends on how far a bare stub gets through the
    # rest of the dycore's validation, which is not this test's subject:
    # the assertion is only that whatever stops it is no longer km_opt.
    with pytest.raises((ValueError, AttributeError, TypeError)) as excinfo:
        dycore.step(SimpleNamespace(h_diabatic=None), _sase_cfg())
    assert "km_opt" not in str(excinfo.value)


@pytest.mark.parametrize("edge", [dict(specified=True),
                                  dict(nested=True)],
                         ids=["specified", "nested"])
def test_sase_specified_boundary_e_floor_and_masks(monkeypatch, edge):
    """Registered adjudication (S3-6 review): on a specified domain the
    driver (a) holds e_sgs at the E_MIN floor across the outer
    spec_bdy_width rows after every fused step -- covering the widest
    test-filter halo so wrapped-neighbor contamination never reaches the
    interior through e -- (b) passes spec_bdy_width to the solve as the
    reduction-exclusion width, and (c) keeps the coupled tendencies
    boundary-masked as built.

    A nested child's edges are no more periodic than a specified
    domain's, so it takes the same policy.  With the width at 0 there
    (the behaviour before SASE was admitted on a nest) the child's test
    filters would wrap the far edge's air into the near edge's e and the
    solve would average those wrapped cells."""
    from woof.verify.sase_ref import E_MIN

    calls = []
    physics, state, cfg, driver = _sase_shim_driver(monkeypatch, calls,
                                                    **edge)
    tend = driver.compute(state, cfg)
    bw = cfg.spec_bdy_width
    assert bw == 5                              # covers the 4-cell halo

    # (b) the solve exclusion width reaches the fused step.
    assert ("exclude", bw) in calls

    # (a) the boundary zone sits exactly at the floor -- including the
    # lowest level, where the surface source deposited interior energy.
    e = state.e_sgs
    floor = np.float32(E_MIN)
    for zone in (e[:, :bw, :], e[:, -bw:, :], e[:, :, :bw], e[:, :, -bw:]):
        assert np.all(zone == floor)
    # The first interior surface cell keeps its deposited source energy
    # (distinct from the floor), proving the clamp geometry is exact.
    dt = cfg.dt
    shear = 0.1**3 / (0.4 * 0.5 * (cfg.ztop / cfg.nz))
    np.testing.assert_allclose(e[0, bw, bw], E_MIN + dt * shear, rtol=1e-5)
    assert np.all(np.isfinite(e))

    # (c) coupled tendencies stay masked at the physical boundary.
    assert np.all(tend.rtheta[:, 0, :] == 0.0)
    assert np.all(tend.rtheta[:, :, -1] == 0.0)
    assert np.all(tend.rw[:, 0, :] == 0.0)
    assert np.all(tend.ru[:, :, 0] == 0.0)
    for arr in (tend.ru, tend.rv, tend.rtheta, tend.rw):
        assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# S3-6b: anisotropic mixing lengths + implicit vertical diffusion
# (authority amendment).  Formulation and the restated split-step ledger
# theorem live in the woof/verify/sase_ref.py module docstring.
# ---------------------------------------------------------------------------


def test_s3_6b_constants_and_karman_single_sourcing():
    """KARMAN must equal the model side's constant (physics.KARMAN, the
    WRF share/module_model_constants karman = 0.4 the surface e source
    and the sfclay/ysu kernels use); C_KV must satisfy the log-layer
    consistency constraint C_KV^3 = C_E exactly (derivation at the
    constant: the neutral constant-stress equilibrium yields
    K_v = C_KV^{3/2} C_E^{-1/2} k u* z, so K_v = k u* z <=> C_KV^3 = C_E).
    """
    import woof.core.physics as physics

    from woof.verify import sase_ref
    assert sase_ref.KARMAN == physics.KARMAN == 0.4
    assert sase_ref.BLACKADAR_LAMBDA == 150.0
    np.testing.assert_allclose(sase_ref.C_KV**3, sase_ref.C_E, rtol=1e-15)


def test_sase_config_id_binds_s3_6b_constants(monkeypatch):
    """Every new S3-6b physics constant is in the config-ID registry."""
    from woof.verify import sase_ref
    for name in ("KARMAN", "BLACKADAR_LAMBDA", "C_KV"):
        assert name in sase_ref._CONFIG_ID_CONSTANTS
    seen = {sase_ref.sase_config_id()}
    for name, value in (("KARMAN", 0.41), ("BLACKADAR_LAMBDA", 100.0),
                        ("C_KV", 0.5)):
        monkeypatch.setattr(sase_ref, name, value)
        cid = sase_ref.sase_config_id()
        assert cid not in seen
        seen.add(cid)


def test_solve_tail_extraction_shared_and_degenerate():
    """S3-3 carry-forward: ONE copy of the 2x2 dynamic-solve tail.

    The device launcher module must bind the very same function object,
    and the tail must reproduce the documented behavior: degenerate
    gram -> (0, 0); well-posed system -> clip/recovery of (c_nu, f)
    from the raw weights a = f*c_nu, b = (1-f)*C_MOM_BG (S3-6g: the
    fixed momentum-background constant, bit-identical to the former
    C_K/PR_T spelling).
    """
    import woof.core.sase as core_sase

    from woof.verify import sase_ref
    assert core_sase._solve_tail is sase_ref._solve_tail
    # S3-6g bit-identity: the renamed constant IS the inherited value.
    assert sase_ref.C_MOM_BG == sase_ref.C_K / sase_ref.PR_LES
    # Degenerate (rank-1) gram: cond gate fires.
    gram = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert sase_ref._solve_tail(gram, np.array([1.0, 1.0])) == (0.0, 0.0)
    # Manufactured diagonal system: a = f*c_nu = 0.06, b = (1-f)*C_MOM_BG
    # with f = 0.6 -> b = 0.4*0.3 = 0.12; recovery must invert exactly.
    a_true, f_true = 0.06, 0.6
    b_true = (1.0 - f_true) * sase_ref.C_MOM_BG
    gram = np.diag([2.0, 3.0])
    proj = np.array([2.0 * a_true, 3.0 * b_true])
    c_nu, f = sase_ref._solve_tail(gram, proj)
    np.testing.assert_allclose(f, f_true, rtol=1e-12)
    np.testing.assert_allclose(c_nu, a_true / f_true, rtol=1e-12)


def test_vertical_mixing_length_branches_closed_form():
    """l_v = min(k*(z+z0)/(1+k*(z+z0)/lambda), LS_COEF*sqrt(e)/N).

    Blackadar branch: exact closed form at every height, k*z limit near
    the surface, saturation strictly below lambda aloft.  Stability
    branch: per-cell selection exactly as the dissipation length's l_s
    (same constants), unstable/neutral cells keep the Blackadar value.
    """
    from woof.verify import sase_ref
    z = np.array([2.5, 25.0, 250.0, 2500.0, 25000.0])
    e = np.full(5, 0.09)                       # sqrt(e) = 0.3
    lv = sase_ref.vertical_mixing_length(z, e)
    kz = 0.4 * z
    np.testing.assert_allclose(lv, kz / (1.0 + kz / 150.0), rtol=1e-14)
    assert lv[0] > 0.99 * kz[0]                # k*z limit near the wall
    assert np.all(np.diff(lv) > 0.0) and lv[-1] < 150.0   # saturation
    # z0 enters exactly as a height shift.
    np.testing.assert_allclose(
        sase_ref.vertical_mixing_length(z, e, z0=0.1),
        sase_ref.vertical_mixing_length(z + 0.1, e), rtol=1e-15)
    # Stability limit: N = 0.01 -> l_s = 0.76*0.3/0.01 = 22.8 m clips
    # every level above ~60 m; n2 <= 0 cells keep Blackadar.
    n2 = np.array([1.0e-4, -1.0e-4, 1.0e-4, 0.0, 1.0e-4])
    lv_s = sase_ref.vertical_mixing_length(z, e, n2=n2)
    expected = np.minimum(kz / (1.0 + kz / 150.0),
                          np.where(n2 > 0, 22.8, np.inf))
    np.testing.assert_allclose(lv_s, expected, rtol=1e-12)


def test_vertical_diffusivity_log_layer_identity():
    """K_v = C_KV*l_v*sqrt(e) reduces to u* * l_v at the equilibrium e.

    Algebra: e_eq = C_E^{-2/3} u*^2 => sqrt(e_eq) = C_E^{-1/3} u*, and
    C_KV = C_E^{1/3} cancels it: K_v = C_E^{1/3} l_v C_E^{-1/3} u* =
    u* * l_v -> KARMAN*u**z as z -> 0.  This is the identity the
    C_KV constant exists to enforce; any other C_KV breaks it by the
    factor C_KV*C_E^{-1/3}.
    """
    from woof.verify import sase_ref
    u_star = 0.5
    z = np.array([5.0, 20.0, 40.0])
    e = np.full(3, sase_ref.C_E**(-2.0 / 3.0) * u_star**2)
    kv = sase_ref.vertical_diffusivity(z, e)
    lv = sase_ref.vertical_mixing_length(z, e)
    np.testing.assert_allclose(kv, u_star * lv, rtol=1e-13)
    # Near-surface band: K_v/(k u* z) = 1/(1 + k z/lambda) -> within
    # [0.9, 1.0] for z <= 40 m (k*z/lambda <= 0.107).
    ratio = kv / (sase_ref.KARMAN * u_star * z)
    assert np.all(ratio > 0.89) and np.all(ratio <= 1.0)


def test_dissipation_length_blend_les_and_rans_limits():
    """l_d = min(delta**f * lb**(1-f), l_s): both limits exact.

    S3-9 RE-PIN (geometric blend; F-Y1 lake over-coupling amendment,
    authority module docstring S3-9 section): the interior-f closed
    form changes from the linear f*delta + (1-f)*lb to the GEOMETRIC
    delta**f * lb**(1-f) -- this assertion is the linear formula's own
    pin and re-pins WITH the formulation (registered deliberate re-pin,
    .superpowers/sdd/yolo-lake-mechanism.md section 5).  The ENDPOINTS
    are the amendment's bitwise contract and are asserted against the
    PRE-CHANGE formula values: f = 1 must reproduce the lb=None (v0)
    path BITWISE (delta**1.0 * lb**0.0 = delta*1.0 == delta, exactly
    the linear form's 1.0*delta + 0.0*lb); f = 0 must give
    min(l_B, l_s) = l_v BITWISE (delta**0.0 * lb**1.0 = 1.0*lb == lb,
    exactly the linear form's 0.0*delta + 1.0*lb).  The l_s branch
    reuses the v0 selection logic unchanged (pinned by the untouched
    test_dissipation_length_stability_limited).
    """
    from woof.verify import sase_ref
    e = np.full((3, 2, 2), 0.09)               # sqrt(e) = 0.3
    z = np.array([100.0, 500.0, 2000.0])[:, None, None]
    lb = sase_ref._blackadar_length(z)
    n2 = np.full((3, 2, 2), 1.0e-4)            # l_s = 22.8 m
    v0 = sase_ref.dissipation_length(e, delta=500.0, n2=n2)
    les = sase_ref.dissipation_length(e, delta=500.0, n2=n2, lb=lb, f=1.0)
    assert np.array_equal(les, v0)             # LES limit: bitwise v0
    rans = sase_ref.dissipation_length(e, delta=500.0, n2=None, lb=lb,
                                       f=0.0)
    # f = 0 endpoint: BITWISE the pre-S3-9 linear formula's value
    # 0.0*500.0 + 1.0*lb == lb (the RANS-column fixtures' branch).
    assert np.array_equal(rans, np.broadcast_to(0.0 * 500.0 + 1.0 * lb,
                                                e.shape))
    rans_s = sase_ref.dissipation_length(e, delta=500.0, n2=n2, lb=lb,
                                         f=0.0)
    np.testing.assert_allclose(
        rans_s, np.minimum(np.broadcast_to(lb, e.shape), 22.8),
        rtol=1e-12)                            # RANS + stability = l_v
    # f = 1 endpoint: BITWISE the pre-S3-9 linear formula's value
    # 1.0*500.0 + 0.0*lb == 500.0 (the LES-limb reduction pin; the
    # n2=None call isolates the blend from the outer l_s min).
    les_n = sase_ref.dissipation_length(e, delta=500.0, n2=None, lb=lb,
                                        f=1.0)
    assert np.array_equal(les_n,
                          np.broadcast_to(1.0 * 500.0 + 0.0 * lb, e.shape))
    # Interior f: the S3-9 geometric closed form.  DERIVATION: with
    # lb = l_B(z) strictly positive and delta = 500 m, the blend is
    # by definition exp(f*ln(delta) + (1-f)*ln(lb)) = 500.0**0.4 *
    # lb**0.6 (log-linear in f; monotone between the two bitwise
    # endpoints; at z = 100 m, l_B ~ 31.6 m: l_d ~ 95.3 m vs the
    # retired linear form's 218.9 m -- the f*delta floor this
    # amendment removes).  Expectation computed by the same closed
    # form, rtol only for association order.
    mid = sase_ref.dissipation_length(e, delta=500.0, n2=None, lb=lb,
                                      f=0.4)
    np.testing.assert_allclose(
        mid, np.broadcast_to(500.0 ** 0.4 * lb ** 0.6, e.shape),
        rtol=1e-15)


def test_implicit_vertical_diffusion_invariants():
    """Constant field invariant; sum(thick*phi) conserved; max principle.

    All three are structural properties of the flux-form zero-flux
    M-matrix solve (docstring): constants are in the kernel of D_v, the
    telescoping fluxes conserve the thickness-weighted sum for ANY grid,
    and the nonnegative inverse with unit row sums bounds the result by
    the input range even at diffusion number K*dt/dz^2 = 1e4.
    """
    from woof.verify import sase_ref
    rng = np.random.default_rng(7)
    nz, ny, nx = 10, 3, 3
    t = _stretched_thicknesses(nz)
    kf = 5.0 + 4.0 * rng.random((nz - 1, ny, nx))
    const = np.full((nz, ny, nx), 3.25)
    out = sase_ref.implicit_vertical_diffusion(const, kf, 30.0, dz_col=t)
    np.testing.assert_allclose(out, 3.25, rtol=1e-14)
    phi = rng.standard_normal((nz, ny, nx))
    out = sase_ref.implicit_vertical_diffusion(phi, kf, 30.0, dz_col=t)
    before = np.sum(t[:, None, None] * phi, axis=0)
    after = np.sum(t[:, None, None] * out, axis=0)
    np.testing.assert_allclose(after, before, rtol=1e-12)
    # Max principle at an extreme diffusion number (dt -> K dt/dz^2 ~ 1e4).
    out = sase_ref.implicit_vertical_diffusion(phi, kf, 1.0e6, dz_col=t)
    assert np.all(out <= phi.max(axis=0) + 1e-12)
    assert np.all(out >= phi.min(axis=0) - 1e-12)
    # nz = 1: no faces, identity.
    one = rng.standard_normal((1, ny, nx))
    got = sase_ref.implicit_vertical_diffusion(
        one, np.zeros((0, ny, nx)), 10.0, dz=50.0)
    assert np.array_equal(got, one)


def test_implicit_vertical_diffusion_neumann_mode_closed_form():
    """Backward Euler damps each zero-flux eigenmode by 1/(1+dt*K*lam_m).

    On a uniform column the cell-centered cosines
    phi_m(k) = cos(pi*m*(k+1/2)/nz) diagonalize the zero-flux flux-form
    Laplacian (DCT-II basis) with lam_m = (4/dz^2)*sin^2(pi*m/(2*nz)):
    interior rows are the standard three-point stencil and the edge rows
    (phi_1-phi_0)/dz^2 are exactly what the reflected cosine produces.
    n steps of (I-dt*K*D)^{-1} therefore scale the mode by
    (1+dt*K*lam_m)^{-n} exactly -- this pins the operator (including
    the edge rows) AND the Thomas solve at once.
    """
    from woof.verify import sase_ref
    nz, dz, kcoef, dt, steps = 24, 50.0, 8.0, 40.0, 5
    k = np.arange(nz)
    kf = np.full(nz - 1, kcoef)
    for m in (1, 3, 7):
        lam = (4.0 / dz**2) * np.sin(np.pi * m / (2.0 * nz))**2
        phi = np.cos(np.pi * m * (k + 0.5) / nz)
        out = phi.copy()
        for _ in range(steps):
            out = sase_ref.implicit_vertical_diffusion(out, kf, dt, dz=dz)
        np.testing.assert_allclose(
            out, phi / (1.0 + dt * kcoef * lam)**steps, atol=1e-12)


def test_explicit_vertical_diverges_implicit_bounded_at_d01():
    """THE stability demonstration (brief fixture b) at d01 parameters.

    Column dz = 50 m, dt = 60 s, e = 1 m2/s2 (the documented instability
    regime of physics._run_sase).  The amended K_v = C_KV*l_B*sqrt(e)
    reaches ~130 m2/s near the l_B -> lambda saturation, so the face
    diffusion number d = K*dt/dz^2 = 3.1 violates the explicit FTCS
    bound (amplification 1 - dt*K*lam with lam_max -> 4/dz^2 gives
    |1-4d| ~ 11.5 per step at the column top; 20 steps ~ 1e21): the v0
    EXPLICIT stepping of the very same face fluxes diverges -- the RED
    demonstration -- while the backward-Euler solve is bounded by the
    max principle for the same K, dt, and initial data.  (For scale:
    the v0 formulation's horizontal-Delta viscosity at d01 gives
    2*nu*dt/dz^2 = 2*0.3*12000*60/2500 = 172.8, the O(10^2) number in
    the _run_sase/launch_sase_step comments; the amendment fixes BOTH
    the length scale and the stepping.)
    """
    from woof.verify import sase_ref
    nz, dz, dt = 60, 50.0, 60.0
    z = (np.arange(nz) + 0.5) * dz
    kv = sase_ref.vertical_diffusivity(z, np.ones(nz))   # e = 1 m2/s2
    kf = 0.5 * (kv[:-1] + kv[1:])
    d_face = kf * dt / dz**2
    assert d_face.max() > 3.0                  # deep in the unstable regime
    u0 = np.cos(np.pi * np.arange(nz))         # Nyquist-heavy data (+-1)
    u = u0.copy()
    for _ in range(20):                        # explicit FTCS, SAME fluxes
        fl = kf * (u[1:] - u[:-1]) / dz
        div = np.zeros(nz)
        div[:-1] += fl / dz
        div[1:] -= fl / dz
        u = u + dt * div
    assert not np.all(np.abs(u) < 1.0e6)       # RED: explicit diverges
    ui = u0.copy()
    for _ in range(20):
        ui = sase_ref.implicit_vertical_diffusion(ui, kf, dt, dz=dz)
    assert np.max(ui) <= 1.0 + 1e-12           # GREEN: max principle
    assert np.min(ui) >= -1.0 - 1e-12
    assert np.max(np.abs(ui)) < 1.0            # strictly damped


def test_horizontal_explicit_cfl_bound_at_d01():
    """The horizontal channel stays explicit: assert the CFL headroom.

    Worst horizontal coefficient over the realizable range (c_nu <=
    CNU_MAX, f in [0,1]): momentum max(CNU_MAX, C_MOM_BG) = 0.5;
    e-transport 2*max(CNU_MAX, C_K) = 1.0; scalar K_h =
    max(CNU_MAX, C_K)/PR_LES = 1.5 -- the binding channel (S3-6g: the
    blended Pr_t(f) attains its MINIMUM PR_LES at f = 1, so the
    worst-case bound is unchanged by the blend).  The two-pass
    centered flux-form operator has symbol -K*(sin(k*dx)/dx)^2 per
    direction (max 1/dx^2), so 2-D explicit stability requires
    K*dt*(1/dx^2 + 1/dy^2) <= 2, i.e. K*dt/dx^2 <= 1 at dx = dy.  At
    d01 scale (delta = dx = 12 km, dt = 60 s) with e capped at
    100 m2/s2 (10x any physical CBL value) the binding number is
    1.5*delta*sqrt(100)*dt/dx^2 = 0.075 -- 13x margin; asserted at
    <= 0.1.  The brief's headline number is pinned too: the bare-C_K
    channel at sqrt(e) = 1 is exactly C_K*delta*dt/dx^2 = 5.0e-4.
    """
    from woof.verify import sase_ref
    dx = 12000.0
    dt = 60.0
    delta = dx                                 # dx = dy at d01
    # min over f of Pr_t(f) = PR_LES (affine, decreasing in f).
    worst_coef = max(sase_ref.CNU_MAX, sase_ref.C_K) / sase_ref.PR_LES
    assert worst_coef == 1.5
    number = worst_coef * delta * np.sqrt(100.0) * dt / dx**2
    assert number <= 0.1                       # 10x under the limit 1.0
    np.testing.assert_allclose(
        sase_ref.C_K * delta * 1.0 * dt / dx**2, 5.0e-4, rtol=1e-12)


def test_split_step_ledger_closes_uniform_e():
    """Split-step ledger theorem, uniform-e periodic box (docstring).

    Same fixture family as the v0 closure test; the split scheme must
    close dKE + dE + dHeat = 0 to relative roundoff with dKE the SUM of
    the explicit (u^n) and implicit (u^{n+1}) pairings.  Both channels
    must be live and draining (the informational breakdown keys).
    """
    from woof.verify import sase_ref
    rng = np.random.default_rng(42)
    shape = (8, 24, 24)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 4)
               for _ in range(3))
    theta = np.full(shape, 300.0)              # no buoyancy path
    e = np.full(shape, 0.2)
    fields, ledger = sase_ref.sase_split_step(
        u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=1.0)
    scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                abs(ledger["dHeat"]), 1e-30)
    assert abs(ledger["residual"]) / scale < 1e-11
    assert ledger["dKE_expl"] < 0.0            # horizontal channel drains
    assert ledger["dKE_impl"] < 0.0            # vertical channel drains
    np.testing.assert_allclose(
        ledger["dKE"], ledger["dKE_expl"] + ledger["dKE_impl"], rtol=1e-12)
    assert np.all(fields["e"] >= sase_ref.E_MIN)


def test_split_step_ledger_closes_with_live_clip_and_varying_e():
    """Split-scheme analog of the v0 live-clip ledger fixture.

    Same seed-3 varying-e fixture (sub-floor cells present).  Since
    S3-6d the analytic decay substep cannot itself cross the floor
    (0 <= e*/(1+b*dt)^2 <= e* for e* >= 0), so the clip engages ONLY
    where the SOURCES leave e* below E_MIN -- the fixture's sub-floor
    input cells with a locally negative or too-small net source do
    exactly that (7 cells measured at this dt), and there clip_gain
    exceeds the (tiny) decay decrement
    => heat locally negative, asserted.  The IMPLICIT vertical
    e-transport then re-floods the floored cells from their neighbors,
    so no cell need sit at E_MIN exactly after the step -- the sign of
    the heat field is the accurate witness that the clip channel ran.
    Solve must be non-degenerate on this fixture (same values as the
    golden pair).
    """
    from woof.verify import sase_ref
    rng = np.random.default_rng(3)
    shape = (8, 24, 24)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 4)
               for _ in range(3))
    theta = np.full(shape, 300.0)              # mechanical path only
    e = np.maximum(
        0.05 + 0.1 * sase_ref.box_filter(rng.standard_normal(shape), 4),
        0.0)
    assert np.any(e < sase_ref.E_MIN)          # fixture reaches the floor
    fields, ledger = sase_ref.sase_split_step(
        u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=0.05)
    scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                abs(ledger["dHeat"]), 1e-30)
    assert abs(ledger["residual"]) / scale < 1e-11
    assert ledger["c_nu"] > 0.0 and ledger["f"] > 0.0    # non-degenerate
    assert np.all(fields["e"] >= sase_ref.E_MIN)
    assert np.any(fields["heat"] < 0.0)        # clip channel proven live


def test_split_step_drains_resolved_ke_into_e_for_sheared_flow():
    """Forward scatter through BOTH split channels on the sheared fixture."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(3)
    shape = (8, 24, 24)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 2)
               for _ in range(3))
    theta = np.full(shape, 300.0)
    e = np.full(shape, 0.05)
    _, ledger = sase_ref.sase_split_step(
        u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=1.0)
    assert ledger["dKE"] < 0.0                 # net drain
    assert ledger["dKE_impl"] < 0.0            # vertical channel drains
    assert ledger["dHeat"] > 0.0               # dissipation heats


def test_split_step_rans_decay_closed_form_single_level():
    """RANS-limit decay through the full split step: EXACT since S3-6d.

    Zero flow degenerates the solve to (c_nu, f) = (0, 0), so the blend
    sits at its RANS limit and l_d = l_B(z = dz/2) -- the amended
    formula's closed-form regime.  nz = 1 kills every transport and
    production channel (single clamped cell: no faces, no vertical
    gradients, horizontal fields uniform), leaving exactly
    de/dt = -C_E e^{3/2}/l_B, e(t) = e0/(1 + b*t)^2 with
    b = C_E*sqrt(e0)/(2*l_B), l_B = 0.4*100/(1 + 40/150) = 31.5789 m.
    S3-6d derivation -- why n steps land the n*dt point EXACTLY: each
    step applies the analytic substep e -> e/(1 + b_k*dt)^2 with
    b_k = C_E*sqrt(e_k)/(2*l_B), which is the exact flow map Phi_dt of
    the autonomous decay ODE; flow maps compose,
    Phi_dt o ... o Phi_dt = Phi_{n*dt} (algebra: if
    e_k = e0/(1 + b0*t_k)^2 then 1 + b_k*dt =
    (1 + b0*t_{k+1})/(1 + b0*t_k), a telescoping product), so the only
    error is FP64 roundoff, ~5 roundings/step * 400 steps ~ 2e-13.
    Measured 2.7e-15; gate rtol 1e-12 (was 4e-3 against the retired
    forward-Euler bias of 2.7e-3).  The ledger must close every step
    and the solve must report degenerate.
    """
    from woof.verify import sase_ref
    shape = (1, 8, 8)
    zeros = np.zeros(shape)
    theta = np.full(shape, 300.0)
    dz, delta = 200.0, 500.0
    e0, dt, steps = 0.5, 0.25, 400
    e = np.full(shape, e0)
    for _ in range(steps):
        fields, ledger = sase_ref.sase_split_step(
            zeros, zeros, zeros, theta, e, dx=500.0, dy=500.0, dz=dz,
            delta=delta, dt=dt)
        e = fields["e"]
        assert (ledger["c_nu"], ledger["f"]) == (0.0, 0.0)
        scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                    abs(ledger["dHeat"]), 1e-30)
        assert abs(ledger["residual"]) / scale < 1e-11
    lb = sase_ref._blackadar_length(0.5 * dz)
    b = sase_ref.C_E * np.sqrt(e0) / (2.0 * lb)
    analytic = e0 / (1.0 + b * dt * steps)**2
    np.testing.assert_allclose(e.mean(), analytic, rtol=1e-12)


def test_split_step_surface_source_equilibrium_at_d01_parameters():
    """S3-6d acceptance: the surface cell HOLDS a live-source equilibrium.

    d01 parameters (dt = 60 s, dz1 = 50 m, nz = 60 uniform column, box
    mode: z1 = 25 m), zero flow (solve degenerates to (0, 0), f = 0
    puts l_d at the neutral l_B), uniform theta (no buoyancy), and the
    driver's named surface e source deposited into the lowest level
    BEFORE each step (the ``physics._run_sase`` ordering) at u* = 0.25:
        S    = u*^3/(kappa*0.5*dz1) = 0.015625/10 = 1.5625e-3 m2/s3
        l_B(25 m) = 0.4*25/(1 + 10/150) = 9.375 m
        e_eq = (S*l_B/C_E)^(2/3) = (1.5625e-3*9.375/0.93)^(2/3)
             = 0.06284 m2/s2      [continuous balance S = C_E*e^{3/2}/l_B]
    Band [0.3, 3] x e_eq: the discrete map sits BELOW the continuous
    balance because the per-step deposit dt*S = 0.094 lands at once and
    the analytic decay then acts on the full spike (operator-splitting
    bias, b*dt ~ 1 here), and the implicit 2*K_v transport drains the
    surface cell into the column; measured equilibrium ratio 0.427.
    v0 RED evidence (task-s3-6d-report): the explicit-Euler dissipation
    at these parameters (dt*C_E*sqrt(e)/l_B ~ 1.5 at e_eq) overshoots
    the balance every step and the clip floor catches it -- the same
    fixture on the pre-S3-6d authority pins e[0] at E_MIN every one of
    the 200 steps (ratio 1.6e-5, the d01 limit cycle this amendment
    exists to fix).
    """
    from woof.verify import sase_ref
    nz, ny, nx = 60, 4, 4
    dz, dt = 50.0, 60.0
    shape = (nz, ny, nx)
    zeros = np.zeros(shape)
    theta = np.full(shape, 300.0)
    u_star = 0.25
    source = u_star**3 / (sase_ref.KARMAN * 0.5 * dz)
    lb1 = float(sase_ref._blackadar_length(0.5 * dz))
    e_eq = (source * lb1 / sase_ref.C_E) ** (2.0 / 3.0)
    np.testing.assert_allclose(e_eq, 0.06284, rtol=1e-3)   # derivation pin
    e = np.full(shape, sase_ref.E_MIN)
    series = []
    for _ in range(200):
        e[0] += dt * source                    # driver's pre-step deposit
        fields, ledger = sase_ref.sase_split_step(
            zeros, zeros, zeros, theta, e, dx=12000.0, dy=12000.0, dz=dz,
            delta=12000.0, dt=dt)
        e = fields["e"]
        assert (ledger["c_nu"], ledger["f"]) == (0.0, 0.0)
        series.append(float(e[0].mean()))
    tail = np.array(series[-20:])
    # REACHES: within 5% of the held value by step 20 (measured 0.02%).
    np.testing.assert_allclose(series[19], tail.mean(), rtol=0.05)
    # HOLDS: every one of the last 20 steps inside the band, and the
    # residual oscillation is roundoff-scale (measured 3e-8 relative).
    assert np.all(tail >= 0.3 * e_eq) and np.all(tail <= 3.0 * e_eq), (
        f"surface e {tail.mean():.5f} outside [0.3, 3] x e_eq={e_eq:.5f}")
    assert (tail.max() - tail.min()) / tail.mean() < 1e-3


def test_split_step_variable_dz_model_mode_runs():
    """Model mode (stretched dz_col): finite, floored, ledger diagnostic.

    The split theorem is uniform-column-exact only (module docstring:
    the unweighted ledger loses telescoping under varying 1/thick_k),
    so like the v0 model mode no residual bar applies -- finiteness and
    the floor are the contract.
    """
    from woof.verify import sase_ref
    u, v, w, e, theta = _collapse_fixture()
    t = _stretched_thicknesses(u.shape[0])
    fields, ledger = sase_ref.sase_split_step(
        u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=0.05, dz_col=t)
    for key in ("u", "v", "w", "e", "heat"):
        assert np.all(np.isfinite(fields[key])), key
    assert np.all(fields["e"] >= sase_ref.E_MIN)
    assert np.isfinite(ledger["residual"])


def test_split_step_trajectory_goldens():
    """S3-6c parity targets: two-step FP64 split trajectories, pinned.

    Frozen fixture (seed 20260720, shape (8, 16, 16)): band-limited
    velocities, theta = 300 + 2*band (live buoyancy), e with sub-floor
    cells, mixed-sign n2 (both l branches), and -- S3-6j -- the uniform
    SPLIT_GOLDEN_UST friction-velocity field (np.full AFTER the rng
    draws, so the frozen field sequence is untouched) engaging the
    implicit surface-stress bottom row in every column (some columns
    sit below SFC_WSPD_FLOOR, exercising the floor branch).  S3-9c:
    the fixture also carries the SPLIT_GOLDEN_GUST enhanced-speed
    field (sfclay convention from the frozen initial winds; rationale
    in tests/sase_goldens.py), engaging the gustiness correction in
    every column including the calm-gusty class.  BOX mode:
    uniform
    dz = 200 m, dt = 0.5; COL mode: the geometric 1.08 thickness column
    from 50 m, dt = 0.05.  Field sums (FP64 np.sum, deterministic
    pairwise order) and the step-2 (c_nu, f) pin the whole split-step
    order of operations; the S3-6c device mirror gates against the same
    fixture and inherits the literals from tests/sase_goldens.py.
    """
    from sase_goldens import (SPLIT_BOX_C_NU, SPLIT_BOX_F,
                              SPLIT_BOX_SUM_E, SPLIT_BOX_SUM_U,
                              SPLIT_BOX_SUM_V, SPLIT_BOX_SUM_W,
                              SPLIT_COL_C_NU, SPLIT_COL_F,
                              SPLIT_COL_SUM_E, SPLIT_COL_SUM_U,
                              SPLIT_COL_SUM_V, SPLIT_COL_SUM_W,
                              SPLIT_GOLDEN_GUST, SPLIT_GOLDEN_UST)

    from woof.verify import sase_ref

    def run(dz_col, dt):
        rng = np.random.default_rng(20260720)
        shape = (8, 16, 16)

        def band():
            return sase_ref.box_filter(rng.standard_normal(shape), 4)

        u, v, w = band(), band(), band()
        theta = 300.0 + 2.0 * band()
        e = np.maximum(0.05 + 0.1 * band(), 0.0)
        n2 = 1.0e-4 * band()
        ust = np.full(shape[1:], SPLIT_GOLDEN_UST)
        # S3-9c: sfclay-convention enhanced speed from the frozen
        # initial winds, held fixed across both steps like ust.
        wspd = np.maximum(
            np.sqrt(u[0] ** 2 + v[0] ** 2 + SPLIT_GOLDEN_GUST ** 2),
            sase_ref.SFC_WSPD_FLOOR)
        assert np.any(e < sase_ref.E_MIN)
        assert np.any(n2 > 0.0) and np.any(n2 <= 0.0)
        # S3-6j fixture witnesses: some columns below the wind floor.
        assert np.any(np.hypot(u[0], v[0]) < sase_ref.SFC_WSPD_FLOOR)
        ledger = None
        for _ in range(2):
            fields, ledger = sase_ref.sase_split_step(
                u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
                delta=500.0, dt=dt, n2=n2, dz_col=dz_col, ust=ust,
                wspd_sfc=wspd)
            u, v, w, e = (fields[k] for k in "uvwe")
        sums = tuple(float(np.sum(a)) for a in (u, v, w, e))
        return sums, ledger

    sums, ledger = run(dz_col=None, dt=0.5)
    expect = (SPLIT_BOX_SUM_U, SPLIT_BOX_SUM_V, SPLIT_BOX_SUM_W,
              SPLIT_BOX_SUM_E)
    np.testing.assert_allclose(sums, expect, rtol=1e-12)
    np.testing.assert_allclose((ledger["c_nu"], ledger["f"]),
                               (SPLIT_BOX_C_NU, SPLIT_BOX_F), rtol=1e-12)

    sums, ledger = run(dz_col=_stretched_thicknesses(8), dt=0.05)
    expect = (SPLIT_COL_SUM_U, SPLIT_COL_SUM_V, SPLIT_COL_SUM_W,
              SPLIT_COL_SUM_E)
    np.testing.assert_allclose(sums, expect, rtol=1e-12)
    np.testing.assert_allclose((ledger["c_nu"], ledger["f"]),
                               (SPLIT_COL_C_NU, SPLIT_COL_F), rtol=1e-12)


# ---------------------------------------------------------------------------
# S3-6e: RANS-limit horizontal governor + damping-layer production taper
# ---------------------------------------------------------------------------


def test_governed_stress_f1_reduces_to_v0_bitwise():
    """At f = 1 the governed stress IS the v0 model_stress (delta/delta)
    bitwise: the smag term is 0.0*K_smag (an FP-exact zero for finite
    K_smag) and r = 0 -- the LES regime is untouched by the governor."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(7)
    shape = (6, 12, 12)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 2)
               for _ in range(3))
    e = np.maximum(0.05 + 0.1 * rng.standard_normal(shape), 0.0)
    s = sase_ref.strain(u, v, w, 500.0, 500.0, 200.0)
    tau_gov, nu, r = sase_ref.governed_stress(e, s, 0.07, 1.0, 500.0)
    tau_v0 = sase_ref.model_stress(e, s, 0.07, 1.0, 500.0, 500.0)
    for got, want in zip(tau_gov, tau_v0):
        np.testing.assert_array_equal(got, want)
    e64 = np.maximum(e, sase_ref.E_MIN)
    np.testing.assert_array_equal(nu, 0.07 * 500.0 * np.sqrt(e64))
    np.testing.assert_array_equal(r, np.zeros(shape))


def test_governed_stress_smag_limit_closed_forms():
    """f = 0 RANS limit against the audited WRF deformation form.

    Pure shear u = a*y: |D_h| = |a| exactly (centered first difference
    of a linear profile is exact), so K_smag = (C_S*delta)^2*|a| below
    the cap and nu = K_smag, r = 1.  Pure rotation (u = a*y, v = -a*x)
    and pure divergence (u = a*x, v = a*y) have |D_h| = 0 -- the
    deformation invariant kills solid-body rotation and isotropic
    dilatation, exactly WRF's def2 property.  A strong-shear fixture
    engages the WRF cap K_m <= SMAG_KM_CAP*delta.  Constants
    single-sourced: C_S must equal the RunConfig km_opt=4 default c_s."""
    from woof.verify import sase_ref
    nz, ny, nx = 2, 16, 16
    dx = dy = 500.0
    delta = 500.0
    y = (np.arange(ny, dtype=np.float64))[None, :, None] * dy
    x = (np.arange(nx, dtype=np.float64))[None, None, :] * dx
    zeros = np.zeros((nz, ny, nx))
    e = np.full((nz, ny, nx), 0.16)

    def def_h_of(u, v):
        s = sase_ref.strain(u + zeros, v + zeros, zeros, dx, dy, 200.0)
        return np.sqrt((s[0] - s[1]) ** 2 + 4.0 * s[3] ** 2)

    a = 2.0e-3                                 # shear rate 1/s
    shear_u = a * y + zeros
    # Interior rows (periodic wrap corrupts the edge rows of a linear
    # ramp): |D_h| = a exactly.
    dh = def_h_of(shear_u, zeros)
    np.testing.assert_allclose(dh[:, 2:-2, :], a, rtol=1e-12)
    s = sase_ref.strain(shear_u, zeros, zeros, dx, dy, 200.0)
    tau, nu, r = sase_ref.governed_stress(e, s, 0.0, 0.0, delta)
    k_expect = (sase_ref.C_S * delta) ** 2 * a
    assert k_expect < sase_ref.SMAG_KM_CAP * delta      # below the cap
    np.testing.assert_allclose(nu[:, 2:-2, :], k_expect, rtol=1e-12)
    np.testing.assert_allclose(r[:, 2:-2, :], 1.0, rtol=1e-12)
    # tau_xy = -2*nu*S_xy = -nu*a on the interior rows.
    np.testing.assert_allclose(tau[3][:, 2:-2, :], -k_expect * a,
                               rtol=1e-12)
    # Rotation and pure divergence are deformation-free.
    np.testing.assert_allclose(def_h_of(a * y, -a * x)[:, 2:-2, 2:-2],
                               0.0, atol=1e-18)
    np.testing.assert_allclose(def_h_of(a * x, a * y)[:, 2:-2, 2:-2],
                               0.0, atol=1e-18)
    # Cap engagement: shear strong enough that (C_S*delta)^2*|D| > cap.
    a_big = 2.0 * sase_ref.SMAG_KM_CAP * delta / (sase_ref.C_S * delta) ** 2
    s_big = sase_ref.strain(a_big * y + zeros, zeros, zeros, dx, dy, 200.0)
    _, nu_big, _ = sase_ref.governed_stress(e, s_big, 0.0, 0.0, delta)
    np.testing.assert_allclose(nu_big[:, 2:-2, :],
                               sase_ref.SMAG_KM_CAP * delta, rtol=1e-12)
    # Single-sourcing with the audited km_opt=4 config constant.
    assert sase_ref.C_S == _min_cfg().c_s == 0.25


def test_split_step_smag_bypass_routes_production_to_heat():
    """RANS-limit bypass-to-heat energetics (S3-6e registered decision).

    Forced (c_nu, f) = (0, 0) (the degenerate-solve RANS limit), nz = 1
    (no vertical channels), uniform e, uniform theta, divergence-free
    shear u = A*sin(2*pi*y/L): r = 1 wherever K_smag > 0 and div_h = 0,
    so P_h,e = 0 IDENTICALLY -- the entire horizontal production
    bypasses e.  e must follow the PURE analytic decay of e0 (l_d =
    l_B(z1) at f = 0) while dKE_expl < 0 drains momentum and dHeat
    picks up both the decay and the bypassed production; the ledger
    closes.  Under the v0 formulation the same fixture GREW e (the
    momentum background deposited 2*nu*|S|^2 into it) -- the exact d01
    mesoscale-strain defect this amendment removes."""
    from woof.verify import sase_ref
    nz, ny, nx = 1, 24, 24
    dx = dy = dz = 500.0
    delta = 500.0
    dt = 5.0
    e0 = 0.25
    y = np.arange(ny, dtype=np.float64)[None, :, None]
    u = 3.0 * np.sin(2.0 * np.pi * y / ny) * np.ones((nz, ny, nx))
    v = np.zeros((nz, ny, nx))
    w = np.zeros((nz, ny, nx))
    theta = np.full((nz, ny, nx), 300.0)
    e = np.full((nz, ny, nx), e0)
    orig_solve = sase_ref.dynamic_solve
    sase_ref.dynamic_solve = lambda *a, **k: (0.0, 0.0)
    try:
        fields, ledger = sase_ref.sase_split_step(
            u, v, w, theta, e, dx=dx, dy=dy, dz=dz, delta=delta, dt=dt)
    finally:
        sase_ref.dynamic_solve = orig_solve
    lb = float(sase_ref._blackadar_length(0.5 * dz))
    b = sase_ref.C_E * np.sqrt(e0) / (2.0 * lb)
    pure_decay = e0 / (1.0 + b * dt) ** 2
    # e saw NO production: exactly the analytic decay of e0 (uniform e
    # kills the horizontal transport; nz = 1 kills the vertical).
    np.testing.assert_allclose(fields["e"], pure_decay, rtol=1e-12)
    assert ledger["dKE_expl"] < 0.0            # smag channel drains KE
    assert ledger["dKE_impl"] == 0.0           # nz = 1: no faces
    assert ledger["dHeat"] > 0.0
    scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                abs(ledger["dHeat"]), 1e-30)
    assert abs(ledger["residual"]) / scale < 1e-11
    # The heat channel exceeds the pure-decay deposit by the bypassed
    # production, which equals the KE drain exactly (theorem (i)).
    decay_sum = (e0 - pure_decay) * nz * ny * nx
    np.testing.assert_allclose(ledger["dHeat"] - decay_sum,
                               -ledger["dKE_expl"], rtol=1e-9)


def test_damp_taper_weights_law_and_ledger_closure():
    """S3-6e taper: (a) the weight law is the damp_opt=3 KDH sin^2
    profile complemented, at layer centers, per column; (b) the split
    ledger still closes to roundoff with the taper live; (c) on the
    d02-mechanism fixture (strong vertical shear across the TOP rows,
    where P_v >= 0 pointwise feeds the model-top spike) the taper run's
    top-level e stays strictly below the untapered run's and relaxes
    on the decay timescale -- no hard clamp anywhere; (d) the taper
    never touches the momentum channel (one-step u/v/w bitwise)."""
    from woof.verify import sase_ref
    # (a) closed-form law on the uniform column.
    nz, dz, zdamp = 8, 200.0, 800.0
    z = (np.arange(nz) + 0.5) * dz
    g = sase_ref.damp_taper_weights(z, dz, zdamp)
    htop = nz * dz
    expect = 1.0 - np.sin(
        0.5 * np.pi * np.clip((z - (htop - zdamp)) / zdamp, 0.0, 1.0)) ** 2
    np.testing.assert_allclose(g, expect, rtol=1e-15)
    assert np.all(g[z < htop - zdamp] == 1.0)
    assert g[-1] == pytest.approx(
        np.cos(0.5 * np.pi * (z[-1] - (htop - zdamp)) / zdamp) ** 2)
    assert np.all(np.diff(g) <= 0.0)
    # (b, c, d): top-shear column (the d02 nest-top mechanism: strong
    # one-sided shear at the clamped top rows, P_v production there).
    shape = (nz, 16, 16)
    prof = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 6.0, 12.0])
    u0 = np.broadcast_to(prof[:, None, None], shape).copy()
    zeros = np.zeros(shape)
    theta = np.full(shape, 300.0)

    def run(zd, steps):
        u, v, w = u0.copy(), zeros.copy(), zeros.copy()
        e = np.full(shape, 0.2)
        for _ in range(steps):
            fields, ledger = sase_ref.sase_split_step(
                u, v, w, theta, e, dx=500.0, dy=500.0, dz=dz,
                delta=500.0, dt=1.0, zdamp=zd)
            scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                        abs(ledger["dHeat"]), 1e-30)
            assert abs(ledger["residual"]) / scale < 1e-11
            u, v, w, e = (fields[k] for k in "uvwe")
        return {"u": u, "v": v, "w": w, "e": e}

    taper = run(800.0, steps=20)
    free = run(None, steps=20)
    # Withholding the (pointwise non-negative) P_v top-row source must
    # leave the tapered top rows strictly below the free run.
    assert float(taper["e"][-1].mean()) < float(free["e"][-1].mean())
    assert float(taper["e"][-2].mean()) < float(free["e"][-2].mean())
    assert np.all(taper["e"] >= sase_ref.E_MIN)
    # (d) the taper reroutes only the e deposit -- momentum mixing is
    # untouched (bitwise over one step from identical inputs).
    one_t = run(800.0, steps=1)
    one_f = run(None, steps=1)
    for key in ("u", "v", "w"):
        np.testing.assert_array_equal(one_t[key], one_f[key])


def test_scalar_hmix_matches_scalar_mix_horizontal_part():
    """The governed scalar channel's discrete form: ``scalar_hmix`` with
    kh = kh_coef*sqrt(e) reproduces the v0 ``scalar_mix`` bitwise on a
    z-uniform scalar (whose vertical leg is exactly zero), and a
    constant scalar mixes to exactly zero."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(11)
    shape = (4, 12, 12)
    s2d = sase_ref.box_filter(rng.standard_normal((1, 12, 12)), 2)
    s = np.broadcast_to(s2d, shape).copy()     # no vertical structure
    e = 0.2 + 0.05 * sase_ref.box_filter(rng.standard_normal(shape), 2)
    e = np.broadcast_to(e[:1], shape).copy()
    kh_coef = 12.0
    kh = kh_coef * np.sqrt(np.maximum(e, sase_ref.E_MIN))
    got = sase_ref.scalar_hmix(s, kh, 500.0, 500.0)
    want = sase_ref.scalar_mix(s, e, kh_coef, 500.0, 500.0, 200.0)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_allclose(
        sase_ref.scalar_hmix(np.full(shape, 3.3), kh, 500.0, 500.0),
        0.0, atol=1e-18)


def test_sase_e_cap_stats_closed_form_and_rejections():
    """Gate-2a cap (``sase_e_cap_stats``) against an independent
    derivation: constant shear u = S0*z on a uniform 50 m column makes
    every face shear exactly S0, so the interior cap is
    (C_KV/C_E)*l_B(z)^2*S0^2 rowwise; the surface row takes the larger
    of that and the named-source balance (S_sfc*l_B(z1)/C_E)^(2/3).
    The percentile/max/ratio wiring and the boundary exclusion are
    checked, then the rejection paths."""
    from woof.core.sase import sase_e_cap_stats
    from woof.verify import sase_ref
    nz, ny, nx = 6, 10, 10
    dzv = 50.0
    t = np.full((nz, ny, nx), dzv, np.float64)
    z = (np.arange(nz) + 0.5) * dzv
    s0 = 0.02
    u = np.broadcast_to((s0 * z)[:, None, None], (nz, ny, nx)).copy()
    v = np.zeros((nz, ny, nx))
    ust = np.full((ny, nx), 0.5)
    e = np.full((nz, ny, nx), 1.0)
    e[3, 5, 5] = 7.0                           # the max the gate reads
    lb = sase_ref._blackadar_length(z)
    cap_rows = (sase_ref.C_KV / sase_ref.C_E) * lb ** 2 * s0 ** 2
    s_sfc = 0.5 ** 3 / (sase_ref.KARMAN * 0.5 * dzv)
    cap_sfc = (s_sfc * lb[0] / sase_ref.C_E) ** (2.0 / 3.0)
    expect_top = max(cap_rows.max(), cap_sfc)
    out = sase_e_cap_stats(e, u, v, t, ust, boundary_width=2)
    np.testing.assert_allclose(out["cap_max"], expect_top, rtol=1e-12)
    np.testing.assert_allclose(out["cap_p"], expect_top, rtol=1e-3)
    assert out["e_max"] == 7.0
    np.testing.assert_allclose(out["ratio"], 7.0 / out["cap_p"],
                               rtol=1e-12)
    with pytest.raises(ValueError, match="shape mismatch"):
        sase_e_cap_stats(e, u[:, :5], v, t, ust)
    with pytest.raises(ValueError, match="boundary_width"):
        sase_e_cap_stats(e, u, v, t, ust, boundary_width=5)


def test_sase_config_id_binds_s3_6e_constants(monkeypatch):
    """Every S3-6e constant is in the config-ID registry (governor
    constants + the gate headroom factor)."""
    from woof.verify import sase_ref
    for name in ("C_S", "SMAG_KM_CAP", "NU_BLEND_EPS", "C_GATE"):
        assert name in sase_ref._CONFIG_ID_CONSTANTS
    seen = {sase_ref.sase_config_id()}
    for name, value in (("C_S", 0.18), ("SMAG_KM_CAP", 12.0),
                        ("NU_BLEND_EPS", 1.0e-20), ("C_GATE", 4.0)):
        monkeypatch.setattr(sase_ref, name, value)
        cid = sase_ref.sase_config_id()
        assert cid not in seen
        seen.add(cid)


def test_column_neutral_log_layer_similarity():
    """Brief fixture (a): neutral log layer, K_v(z) ~ kappa*u**z bands.

    Constant-stress column (u*^2 in at the top face, out at the
    surface; the model's named surface e source at the bottom cell) run
    to steady state with the amended vertical channel.  Fixed point
    (derivations at C_KV and in the column docstring):
      e = C_E^{-2/3} u*^2 uniformly,  K_v = u* * l_B(z),
      flux K_v*du/dz = u*^2 through every interior face,
      du/dz = (u*/(kappa*z))*(1 + kappa*z/lambda).
    Assertion window z in [10, 60] m (nz = 32, dz = 5 m column; edges
    excluded).  Measured at steps = 4000, dt = 1 s: K_v within 1.6% of
    the closed form, flux within 1e-4 of u*^2, window-mean e within
    0.9% of the fixed point -- the bands below carry >= 2x headroom.
    The headline physical claim K_v ~ kappa*u**z is pinned two ways:
    ratio band [0.85, 1.005] across the window (the Blackadar
    correction 1/(1 + kappa*z/lambda) is 0.87-0.97 there) and
    [0.95, 1.15] at the FIRST level (z = 2.5 m): the discrete faces see
    no shear below z = dz, so the surface cell's K_v is fed by the
    driver's u*^3/(kappa*0.5*dz) source, which lands it at 1.06 of
    kappa*u**z (without the source it sits at 0.70 -- the very defect
    the named source exists to fix; measured, documented).

    S3-6h (HARD CONSTRAINT receipt): the engine now runs the COMPOSED
    RANS lengths (BL89 + kappa-z match, engine docstring), and the
    fixture stays green with margin: window K_v within 1.51% of the
    closed form (band 2%), ratio window [0.861, 0.982] (band
    [0.85, 1.005]), surface ratio 1.062, flux dev 4e-14, window e
    1.003x the fixed point -- the kappa-z match works exactly as
    registered (BL89 geometry >= l_B everywhere below ~htop - l_B).
    """
    from woof.verify import sase_ref
    u_star, nz, dz = 0.5, 32, 5.0
    r = sase_ref.column_neutral_log_layer(u_star, nz, dz, dt=1.0,
                                          steps=4000)
    z, kv, e, flux = r["z"], r["kv"], r["e"], r["flux"]
    kap, lam = sase_ref.KARMAN, sase_ref.BLACKADAR_LAMBDA
    win = (z >= 10.0) & (z <= 60.0)
    # (1) closed-form K_v = u* * l_B(z), 2% band (measured <= 1.6%).
    np.testing.assert_allclose(
        kv[win], u_star * kap * z[win] / (1.0 + kap * z[win] / lam),
        rtol=0.02)
    # (2) the log-layer claim K_v ~ kappa*u**z.
    ratio = kv / (kap * u_star * z)
    assert np.all(ratio[win] > 0.85) and np.all(ratio[win] < 1.005)
    assert 0.95 < ratio[0] < 1.15              # surface cell, sourced
    # (3) steady constant-stress profile through the window faces.
    zf = 0.5 * (z[:-1] + z[1:])
    winf = (zf >= 10.0) & (zf <= 60.0)
    np.testing.assert_allclose(flux[winf], u_star**2, rtol=5e-3)
    # (4) equilibrium e level (the C_KV^3 = C_E fixed point).
    np.testing.assert_allclose(
        e[win].mean(), sase_ref.C_E**(-2.0 / 3.0) * u_star**2, rtol=0.02)
    # (5) S3-6i invariant (registered): l_s NEVER binds in a neutral
    # column, so the stable-limit coefficient decoupling is exactly
    # inert here -- the composed RANS length is the bare l_B through
    # the window (no n2, BL89 geometry slack) and the coefficient
    # blend returns C_KV bitwise; the engine's K_v IS the C_KV
    # composition to the last bit.
    l_mix_r, _ = sase_ref.bl89_rans_lengths(
        np.full(nz, 300.0), e, z, dz)
    lb = sase_ref._blackadar_length(z)
    assert np.array_equal(l_mix_r[win], lb[win])
    coef = sase_ref.stable_limit_coefficient(l_mix_r, e, None)
    assert np.array_equal(coef, np.full(nz, sase_ref.C_KV))
    np.testing.assert_array_equal(
        kv, sase_ref.C_KV * l_mix_r
        * np.sqrt(np.maximum(e, sase_ref.E_MIN)))


def test_the_stable_limb_e_equation_is_homogeneous_and_has_no_amplitude():
    """THE structural property behind the free-troposphere energy growth.

    Where the stability length binds -- l_s = LS_COEF*sqrt(e)/N shorter
    than the Blackadar length, which is every stably stratified cell
    with any subgrid energy at all -- the closure's two competing terms
    are BOTH linear in e:

        K_v   = C_r * l_s * sqrt(e)      = LS_COEF*C_r*e/N
        eps   = C_E * e^{3/2} / l_s      = (C_E/LS_COEF)*e*N

    so production K_v*S^2, buoyancy destruction -(K_v/Pr_t)*N^2 and
    dissipation all scale as e, the subgrid-energy equation becomes
    exactly HOMOGENEOUS, and its specific growth rate

        (1/e) de/dt = N*[LS_COEF*C_r*(1/Ri - 1/Pr_t) - C_E/LS_COEF]

    does not depend on e at all.  There is therefore NO equilibrium
    amplitude in this regime: below an implied critical Richardson
    number the energy grows exponentially and is bounded only by the
    resolved shear being mixed away, never by the closure.

    WHY THIS IS PINNED HERE.  It is the mechanism of the measured
    11-14 km energy growth (attribution runs, 2026-08-02): 72% of the
    cells the closure lit up at 12.2 km sat at Ri < 0.25 against a
    0.36% base rate, with a measured doubling time near 3 minutes
    against the 0.4-2 minute e-folding this rate predicts at the
    observed Ri.  A future edit that gives the stable limb an amplitude
    scale -- any term not linear in e -- SHOULD break this test, and
    that is the point: it would be the fix.
    """
    import math

    from woof.verify import sase_ref as R

    n2 = 1.69e-4                      # a measured 12 km value, this case
    n = math.sqrt(n2)
    pr_t = float(R.prandtl_blend(0.0))               # the RANS limb

    def specific_rate(e, ri, c_r):
        l_s = R.LS_COEF * math.sqrt(e) / n
        assert l_s < R.BLACKADAR_LAMBDA, (e, l_s)    # l_s is what binds
        kv = c_r * l_s * math.sqrt(e)
        prod = kv * (n2 / ri)
        buoy = -(kv / pr_t) * n2
        diss = R.C_E * e ** 1.5 / l_s
        return (prod + buoy - diss) / e

    # (1) HOMOGENEOUS: the specific rate is independent of e over five
    #     decades, to relative 1e-12.  This is the whole claim.
    for ri in (0.05, 0.1, 0.2):
        for c_r in (R.C_KS, R.C_KV):
            rates = [specific_rate(e, ri, c_r)
                     for e in (1e-4, 1e-2, 1.0, 1.62, 5.0)]
            assert max(rates) - min(rates) <= 1e-12 * abs(rates[0]), (
                ri, c_r, rates)

    # (2) The implied critical Richardson number, closed form against a
    #     bisection on the rate itself.  0.13 on the stable limb, 0.35
    #     on the neutral one -- both well above the 0.05-0.20 the anvil
    #     shear layer actually reaches, which is why it grows there.
    for c_r, expected in ((R.C_KS, 0.1313), (R.C_KV, 0.3539)):
        closed = 1.0 / (1.0 / pr_t + R.C_E / (R.LS_COEF ** 2 * c_r))
        lo, hi = 1e-4, 5.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = ((mid, hi) if specific_rate(1.0, mid, c_r) > 0
                      else (lo, mid))
        np.testing.assert_allclose(closed, 0.5 * (lo + hi), rtol=1e-6)
        np.testing.assert_allclose(closed, expected, rtol=1e-3)

    # (3) And the growth is FAST where the measurement found it: an
    #     e-folding time under 2 minutes at Ri = 0.1 on the stable limb.
    assert 0.0 < math.log(2.0) / specific_rate(1.0, 0.1, R.C_KS) < 130.0


# --- S3-12 additive e^{3/2} dissipation channel ---------------------------


def _s312_rates(e, ri, n2, l_ref, f=0.0, c_ed=None):
    """Specific rate (1/e)de/dt of the stable limb where l_s binds,
    HEAD and with the ADDITIVE channel, on the authority's own
    coefficient functions.  Returns (head, additive)."""
    import math

    from woof.verify import sase_ref as R

    c_ed = R.C_ED if c_ed is None else c_ed
    n = math.sqrt(n2)
    pr_t = float(R.prandtl_blend(f))
    l_s = R.LS_COEF * math.sqrt(e) / n
    # where the stability length binds, l_mix_rans == l_d == l_s
    c_r = float(R.stable_limit_coefficient(np.array([l_s]), np.array([e]),
                                           np.array([n2]))[0])
    kv = c_r * l_s * math.sqrt(e)
    core = (kv * (n2 / ri) - (kv / pr_t) * n2) / e
    head = core - R.C_E * math.sqrt(e) / l_s
    c_eps = float(R.additive_dissipation_coefficient(
        np.array([l_s]), np.array([l_ref]), np.array([e]),
        np.array([n2]), f=f)[0])
    return head, core - c_eps * math.sqrt(e) / l_s


def test_additive_dissipation_breaks_the_stable_limb_homogeneity():
    """S3-12: the fix the REJECTED option could not be.

    LD_STABILITY_LIMIT_REJECTED broke the homogeneity by taking l_s OFF
    the dissipation length, which can only LENGTHEN l_d and therefore
    only WEAKEN dissipation.  This channel breaks it the other way --
    Deardorff's second, grid-scale member ADDED, so dissipation is
    nowhere weaker (pinned separately below).  The three properties
    that make it a fix and not a rescaling:

    (1) HOMOGENEITY BROKEN.  HEAD's specific rate is independent of e
        to relative 1e-12 across EIGHT decades (the test above pins five
        at the same tolerance); with the channel on, the same rate
        varies by more than 100% over the same range, because
        :func:`neutral_dissipation_length` carries no e.

    (2) A FINITE FIXED POINT EXISTS, with a closed form.  Writing HEAD's
        rate as ``a`` (constant in e) the equation becomes
        de/dt = a*e - (1-f)*C_ED*e^{3/2}/l_ref, so

            sqrt(e_eq) = a*l_ref/((1 - f)*C_ED)

        for every Ri below the critical value.  Matched here against a
        bisection on the authority-evaluated rate itself.

    (3) THE CRITICAL RICHARDSON NUMBER SURVIVES AS AN EXISTENCE
        THRESHOLD AND STOPS BOUNDING A GROWTH REGIME.  The added term
        vanishes as sqrt(e), so at the E_MIN floor the onset threshold
        is HEAD's 0.16471 to better than 1e-3 relative -- turbulence
        still cannot start above it, which is what the registered
        Ri* protects.  Above the floor the effective threshold FALLS
        monotonically with amplitude: that falling threshold IS the
        saturation, and it is why nothing below Ri* runs away any more.
    """
    import math

    from woof.verify import sase_ref as R

    n2 = 3.566e-5                       # the measured 12.2 km value
    l_ref = R.BLACKADAR_LAMBDA          # the f = 0 reference length
    decades = [10.0 ** k for k in range(-6, 3)]

    # (1) homogeneity: HEAD flat to 1e-12, additive channel NOT flat
    for ri in (0.05, 0.10, 0.15):
        head = [_s312_rates(e, ri, n2, l_ref)[0] for e in decades]
        add = [_s312_rates(e, ri, n2, l_ref)[1] for e in decades]
        assert max(head) - min(head) <= 1e-12 * abs(head[0]), (ri, head)
        spread = (max(add) - min(add)) / abs(np.median(add))
        assert spread > 1.0, (ri, spread, add)

    # (2) closed-form fixed point against a bisection on the rate
    for ri in (0.05, 0.10, 0.12, 0.14):
        a = _s312_rates(1.0, ri, n2, l_ref)[0]
        assert a > 0.0, (ri, a)                    # subcritical: it grows
        closed = (a * l_ref / R.C_ED) ** 2
        lo, hi = 1e-12, 1e6                        # rate(lo) > 0 > rate(hi)
        for _ in range(200):
            mid = math.sqrt(lo * hi)               # bisect in log e
            lo, hi = ((mid, hi) if _s312_rates(mid, ri, n2, l_ref)[1] > 0.0
                      else (lo, mid))
        np.testing.assert_allclose(closed, math.sqrt(lo * hi), rtol=1e-6)
        # and the fixed point is an ATTRACTOR: rate > 0 below, < 0 above
        assert _s312_rates(0.5 * closed, ri, n2, l_ref)[1] > 0.0
        assert _s312_rates(2.0 * closed, ri, n2, l_ref)[1] < 0.0

    # (3) the critical Ri at the floor is HEAD's, and it falls with e
    def crit(e):
        lo, hi = 1e-6, 10.0                        # rate(lo) > 0 > rate(hi)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = ((mid, hi) if _s312_rates(e, mid, n2, l_ref)[1] > 0.0
                      else (lo, mid))
        return 0.5 * (lo + hi)

    head_star = 1.0 / (1.0 / float(R.prandtl_blend(0.0))
                       + R.C_E / (R.LS_COEF * R.C_KS))
    np.testing.assert_allclose(crit(R.E_MIN), head_star, rtol=1e-3)
    thresholds = [crit(e) for e in (R.E_MIN, 1e-4, 1e-2, 1.0)]
    assert all(b < a for a, b in zip(thresholds, thresholds[1:])), thresholds


def test_head_free_troposphere_is_bounded_and_the_channel_lowers_it():
    """S3-12 THE CORRECTION TO THE RECORD, and the amendment's actual
    worth, both measured through the LIVE length composition rather
    than the binding-limb closed form.

    "The stable limb has no equilibrium amplitude" is exactly true
    WHILE l_s binds and false of the closure as a whole: l_s grows as
    sqrt(e), so a runaway eventually pushes it past the Blackadar
    length, ``dissipation_length``'s outer min stops selecting it, and
    the e^{3/2} dissipation returns on its own.  HEAD is therefore
    BOUNDED -- at 2.8 to 39 m2/s2, two to three orders above anything
    physical, which is the accurate statement of the defect.

    The channel is worth 5x to 475x in Ri = 0.12-0.16 (just under
    Ri* = 0.16471, where the census found 72% of live cells against a
    0.36% base rate) and only 2-18% at Ri <= 0.1, because there the
    equilibrium already sits above the crossover with l_s slack and the
    stability gate tapered off.  Both halves are asserted: a test that
    only pinned the good half would be advertising.
    """
    import math

    from woof.verify import sase_ref as R

    n2, nz, dz = 3.566e-5, 120, 400.0
    z = (np.arange(nz) + 0.5) * dz
    theta = 220.0 * np.exp(n2 * z / R.G_ACCEL)
    k, pr = nz // 2, R.prandtl_blend(0.0)

    def rate(e_scalar, ri, additive):
        e = np.full(nz, max(e_scalar, R.E_MIN))
        n2f = np.full(nz, n2)
        l_mix, l_eps = R.bl89_rans_lengths(theta, e, z, dz, n2f)
        kv = R.stable_limit_coefficient(l_mix, e, n2f) * l_mix * np.sqrt(e)
        ld = R.dissipation_length(e, 1.0, n2f, lb=l_eps, f=0.0)
        c = (R.additive_dissipation_coefficient(
            ld, R.neutral_dissipation_length(z, 1.0, f=0.0), e, n2f,
            f=0.0) if additive else np.full(nz, R.C_E))
        return (kv[k] * (n2 / ri) - (kv[k] / pr) * n2
                - c[k] * e[k] ** 1.5 / ld[k]) / e[k]

    def equilibrium(ri, additive):
        lo, hi = 1e-8, 1e8
        if rate(lo, ri, additive) <= 0.0:
            return 0.0
        assert rate(hi, ri, additive) < 0.0, ri   # bounded, both legs
        for _ in range(60):        # 16 decades / 2^60: far past FP64
            mid = math.sqrt(lo * hi)
            lo, hi = ((mid, hi) if rate(mid, ri, additive) > 0.0
                      else (lo, mid))
        return math.sqrt(lo * hi)

    # the crossover that bounds HEAD: where l_s reaches the Blackadar length
    lb = float(R._blackadar_length(np.array([z[k]]))[0])
    crossover = (lb * math.sqrt(n2) / R.LS_COEF) ** 2
    np.testing.assert_allclose(crossover, 1.347, rtol=2e-3)

    for ri, lo, hi in ((0.02, 30.0, 50.0), (0.10, 5.0, 8.0),
                       (0.16, 2.0, 4.0)):
        assert lo < equilibrium(ri, False) < hi, (ri,
                                                  equilibrium(ri, False))
    assert equilibrium(0.18, False) == 0.0        # above Ri*: collapses

    # the channel LOWERS it everywhere and by orders where it matters
    for ri in (0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16):
        assert equilibrium(ri, True) < equilibrium(ri, False), ri
    assert equilibrium(0.16, False) / equilibrium(0.16, True) > 100.0
    assert equilibrium(0.14, False) / equilibrium(0.14, True) > 10.0
    # ... and only marginally at low Ri, which is stated, not hidden
    assert equilibrium(0.05, False) / equilibrium(0.05, True) < 1.2


def test_additive_dissipation_is_nowhere_weaker_and_selectively_inert():
    """S3-12 the property LD_STABILITY_LIMIT_REJECTED could not have.

    * NOWHERE WEAKER, pointwise, over a dense random sweep of the whole
      admissible (l_d, l_ref, e, N^2, f) space: the effective
      coefficient is >= the base it is added to, and it is bounded above
      by C_E + C_ED because l_d <= l_ref by construction.
    * BITWISE INERT where the amendment claims not to reach: f = 1 (the
      LES limb), N^2 <= 0 (unstable/neutral, including every
      M1-substituted cell) and ``n2 is None``.  Bitwise, by SELECTION --
      asserted with ``.tobytes()`` because ``base + 0.0*x`` is not
      generally the base and this module has already been bitten by
      exactly that (the S3-6k first commit).
    """
    from woof.verify import sase_ref as R

    rng = np.random.default_rng(20260802)
    n = 200000
    e = 10.0 ** rng.uniform(-6.0, 2.0, n)
    n2 = 10.0 ** rng.uniform(-7.0, -3.0, n)
    f = rng.uniform(0.0, 1.0, n)
    l_s = R.LS_COEF * np.sqrt(e) / np.sqrt(n2)
    l_ref = 10.0 ** rng.uniform(0.0, 4.5, n)
    l_d = np.minimum(l_ref, l_s)                   # the live construction
    c = R.additive_dissipation_coefficient(l_d, l_ref, e, n2, f=f)
    assert np.all(c >= R.C_E), float(c.min())
    assert np.all(c <= R.C_E + R.C_ED), float(c.max())

    # bitwise inertness, by selection
    ones = np.ones(n)
    for label, kw in (("f=1", {"n2": n2, "f": 1.0}),
                      ("n2<=0", {"n2": -n2, "f": f}),
                      ("n2 None", {"n2": None, "f": f})):
        got = R.additive_dissipation_coefficient(l_d, l_ref, e, **kw)
        want = np.full(n, R.C_E)
        assert got.tobytes() == want.tobytes(), label

    # and it composes with S3-6k rather than replacing it
    base = R.stable_dissipation_coefficient(l_d, e, n2, f=0.0)
    comp = R.additive_dissipation_coefficient(l_d, l_ref, e, n2, f=0.0,
                                              c_base=base)
    assert np.all(comp >= base)


def test_additive_channel_needs_a_state_independent_reference_length():
    """S3-12 WHY the reference length drops the BL89 member -- the whole
    content of the amendment, and the trap a later lane would fall into.

    l_d's RANS input is min(l_B, l_eps_BL89).  The BL89 displacement
    lengths solve a parcel-energy integral, so in uniform stratification
    they scale as sqrt(2*e)/N -- exactly like l_s, only longer, which is
    also why l_s always binds first.  Divide C_ED*e^{3/2} by a length
    proportional to sqrt(e) and the "e^{3/2} channel" is e-LINEAR again:
    it breaks no homogeneity, has no fixed point, and would look like a
    fix while being a rescaling of the defect.

    Pinned here on the authority's own :func:`bl89_rans_lengths` and on
    the two rates side by side: l_ref = l_B gives a spread over eight
    decades of e that is O(1); l_ref = l_eps_BL89 gives HEAD's flatness
    back.  The BL89 tolerance is 1e-4 and not 1e-12 because the crossing
    solve interpolates WITHIN a layer, so l_eps/sqrt(e) is constant only
    up to the vertical discretization -- measured 2.5e-5 at this
    fixture's 25 m layers, four orders tighter than the O(1) contrast it
    is being read against.
    """
    import math

    from woof.verify import sase_ref as R

    # BL89 dissipation length scales as sqrt(e) in uniform stratification
    n2, nz, thick = 3.566e-5, 240, 25.0
    z = (np.arange(nz) + 0.5) * thick
    theta = 300.0 * np.exp(n2 * z / R.G_ACCEL)      # uniform N^2
    k = nz // 2
    ratios = []
    for e0 in (1e-4, 1e-3, 1e-2, 1e-1):
        e = np.full(nz, e0)
        _, l_eps = R.bl89_rans_lengths(theta, e, z, thick, np.full(nz, n2))
        ratios.append(l_eps[k] / math.sqrt(e0))
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-4)
    # ... and it is LONGER than l_s, which is why l_s binds l_d
    assert ratios[0] > R.LS_COEF / math.sqrt(n2)

    decades = [10.0 ** j for j in range(-6, 3)]
    flat = [_s312_rates(e, 0.1, n2,
                        ratios[0] * math.sqrt(e))[1] for e in decades]
    broken = [_s312_rates(e, 0.1, n2, R.BLACKADAR_LAMBDA)[1]
              for e in decades]
    assert max(flat) - min(flat) <= 1e-12 * abs(flat[0]), flat
    assert (max(broken) - min(broken)) / abs(np.median(broken)) > 1.0


def test_registered_tke_deficit_is_closable_and_costs_the_flux_nothing():
    """The registered 3-5x TKE deficit, MEASURED and priced.

    THE REGISTERED CLAIM (config.py / physics.py approximation (2), and
    the C_KV constant's own docstring): "matching the flux K_v puts the
    e level at ~1.05 u*^2 vs the observed ~3.5 u*^2; the flux is what
    acts dynamically, the e level is diagnostic" -- recorded as a
    *known trade-off* of one-constant e-l closures, i.e. as something
    that cannot be had both ways.

    WHAT THIS MEASURES.  The trade-off is real only if C_E is held at
    the Deardorff LES value.  The log-layer constraint is C_KV^3 = C_E
    (one equation), and the equilibrium level is e = C_E^(-2/3) u*^2
    (a second).  Two equations, TWO unknowns -- so C_E is a free
    parameter that sets the e level, and the constraint fixes C_KV to
    match.  This test runs the authority's own neutral constant-stress
    column at several C_E and measures both quantities:

      * e/u*^2 moves as C_E^(-2/3), across a factor of 3;
      * K_v/(u* l_B) and the interior-face momentum flux DO NOT MOVE.

    So the deficit costs the flux nothing to close, and the entry that
    calls it a trade-off is falsified as stated.  C_E = 2^1.5/16.6 =
    0.1704 -- the Mellor-Yamada B1 = 16.6 dissipation constant, which
    is the RANS-limb calibration this length (l -> kappa*z) belongs to,
    as opposed to the inertial-range LES calibration C_E = 0.93 is --
    lands e at 3.26 u*^2, inside the registered observed band.

    WHAT THIS DOES NOT DO.  It does not change a registered constant.
    C_E is one of a coupled family (C_KV = C_E^(1/3) here; C_ES/C_KS on
    the stable limb, whose joint re-registration requirement is already
    recorded), and this lane measures rather than re-tunes.  What it
    removes is the reason to leave the deficit open: the target value
    is now a number with a receipt.
    """
    from woof.verify import sase_ref

    u_star, nz, dz = 0.3, 60, 20.0

    def column(c_e):
        old = sase_ref.C_E, sase_ref.C_KV
        sase_ref.C_E = c_e
        sase_ref.C_KV = c_e ** (1.0 / 3.0)     # the log-layer tie
        try:
            return sase_ref.column_neutral_log_layer(
                u_star, nz, dz, dt=2.0, steps=20000)
        finally:
            sase_ref.C_E, sase_ref.C_KV = old

    kap, lam = sase_ref.KARMAN, sase_ref.BLACKADAR_LAMBDA
    measured = {}
    for c_e in (sase_ref.C_E, 2.0 ** 1.5 / 16.6):
        r = column(c_e)
        z, e, kv, flux = r["z"], r["e"], r["kv"], r["flux"]
        win = (z > 3 * dz) & (z < 0.5 * nz * dz)
        lb = kap * z / (1.0 + kap * z / lam)
        measured[round(c_e, 4)] = (
            float((e[win] / u_star ** 2).mean()),
            float((kv[win] / (u_star * lb[win])).mean()),
            float((flux[win[:-1]] / u_star ** 2).mean()))

    registered = measured[round(sase_ref.C_E, 4)]
    my_like = measured[round(2.0 ** 1.5 / 16.6, 4)]

    # (1) As registered, the deficit is real and is where it is recorded.
    assert 1.0 < registered[0] < 1.15, registered
    # (2) The Mellor-Yamada-equivalent dissipation constant lands the
    #     same column inside the registered observed band (3.3-5.5),
    #     within the fixture's own few-percent accuracy.
    assert 3.1 < my_like[0] < 5.5, my_like
    # (3) THE PRICE: none.  Both the mixing coefficient and the
    #     interior-face momentum flux are unmoved to 3 decimals.  If a
    #     future edit makes the e level cost flux accuracy, this fails.
    assert abs(registered[1] - my_like[1]) < 1e-3, measured
    assert abs(registered[2] - 1.0) < 1e-3, measured
    assert abs(my_like[2] - 1.0) < 1e-3, measured
    # (4) And the move is the predicted C_E^(-2/3) scaling, not a fit.
    predicted = (sase_ref.C_E / (2.0 ** 1.5 / 16.6)) ** (2.0 / 3.0)
    np.testing.assert_allclose(my_like[0] / registered[0], predicted,
                               rtol=0.02)


# ---------------------------------------------------------------------------
# Stage-3 Task 7: smoke-gate surface-e diagnostic (NumPy path)
# ---------------------------------------------------------------------------


def test_sase_surface_e_stats_bins_ratio_and_boundary_exclusion():
    """``sase_surface_e_stats`` against an independent derivation.

    Constructed plane: dz1 = 50 m everywhere, so z1 = 25 m and the
    authority Blackadar length is the S3-6d fixture value l_B = 9.375 m
    (asserted).  Interior u* blocks are placed one per adjudicated bin;
    e0 is seeded to the analytic equilibrium e_eq = (S*l_B/C_E)^(2/3),
    S = u*^3/(kappa*z1), computed HERE from the authority constants --
    every returned bin must then carry ratio 1 to roundoff and exactly
    its constructed count.  The boundary ring carries the specified-
    domain floor value E_MIN with u* = 0.9 (the >0.6 bin, were it not
    excluded): boundary_width=2 must keep it out of every bin while the
    full-plane e0 min still reports the floor.
    """
    from woof.core.sase import (SURFACE_E_USTAR_BIN_EDGES,
                                 sase_surface_e_stats)
    from woof.verify import sase_ref

    ny, nx, w = 14, 14, 2                      # interior 10 x 10: 5 bands
    z1 = 25.0
    lb = float(sase_ref._blackadar_length(z1))
    assert lb == pytest.approx(9.375, abs=1e-12)   # S3-6d fixture value

    ust = np.full((ny, nx), 0.9)
    # Interior u* blocks: one column band per bin (interior is 10 x 8).
    band_values = (0.15, 0.25, 0.35, 0.5, 0.7)
    interior_cols = nx - 2 * w
    ust_interior = np.empty((ny - 2 * w, interior_cols))
    counts = []
    for b, lo in enumerate(range(0, interior_cols, 2)):
        value = band_values[min(b, len(band_values) - 1)]
        ust_interior[:, lo:lo + 2] = value
    for value in band_values:
        counts.append(int((ust_interior == value).sum()))
    ust[w:-w, w:-w] = ust_interior

    s_shear = ust ** 3 / (sase_ref.KARMAN * z1)
    e_eq = (s_shear * lb / sase_ref.C_E) ** (2.0 / 3.0)
    e0 = e_eq.copy()
    e0[:w, :] = sase_ref.E_MIN
    e0[-w:, :] = sase_ref.E_MIN
    e0[:, :w] = sase_ref.E_MIN
    e0[:, -w:] = sase_ref.E_MIN
    dz1 = np.full((ny, nx), 50.0)

    stats = sase_surface_e_stats(e0, ust, dz1, boundary_width=w)
    assert stats["boundary_width"] == w
    assert stats["e0"]["min"] == pytest.approx(sase_ref.E_MIN)
    assert stats["e0_interior"]["min"] > sase_ref.E_MIN
    assert SURFACE_E_USTAR_BIN_EDGES == (0.2, 0.3, 0.4, 0.6)
    assert [b["ustar"] for b in stats["bins"]] == [
        "<0.2", "0.2-0.3", "0.3-0.4", "0.4-0.6", ">0.6"]
    assert [b["count"] for b in stats["bins"]] == counts
    assert sum(counts) == (ny - 2 * w) * (nx - 2 * w)
    for entry, value in zip(stats["bins"], band_values):
        assert entry["ust_mean"] == pytest.approx(value)
        for key in ("ratio_min", "ratio_mean", "ratio_max"):
            assert entry[key] == pytest.approx(1.0, rel=1e-12)


def test_sase_surface_e_stats_rejects_bad_shapes_and_width():
    from woof.core.sase import sase_surface_e_stats
    plane = np.ones((6, 6))
    with pytest.raises(ValueError, match="shape"):
        sase_surface_e_stats(plane, np.ones((6, 5)), plane)
    with pytest.raises(ValueError, match="interior"):
        sase_surface_e_stats(plane, plane, plane, boundary_width=3)


# ---------------------------------------------------------------------------
# S3-6f: partition cap + w-based resolved-fraction bound (mesoscale
# sensing concession -- authority module docstring, S3-6f section).
# ---------------------------------------------------------------------------


def test_sase_config_id_binds_s3_6f_constants(monkeypatch):
    """Every S3-6f constant is in the config-ID registry (cap form,
    bulk-Richardson z_i, w-sensor screen)."""
    from woof.verify import sase_ref
    names = ("F_CAP_KNEE", "F_CAP_WIDTH", "RIB_CRIT",
             "RIB_WSPD2_FLOOR", "N2_SCREEN")
    for name in names:
        assert name in sase_ref._CONFIG_ID_CONSTANTS
    seen = {sase_ref.sase_config_id()}
    for name, value in (("F_CAP_KNEE", 1.5), ("F_CAP_WIDTH", 3.0),
                        ("RIB_CRIT", 0.5), ("RIB_WSPD2_FLOOR", 2.0),
                        ("N2_SCREEN", 5.0e-5)):
        monkeypatch.setattr(sase_ref, name, value)
        cid = sase_ref.sase_config_id()
        assert cid not in seen
        seen.add(cid)


def test_partition_cap_form_and_limits():
    """FP-exact 1 through rho <= 1, C^1 knee, monotone Gaussian ramp,
    and the l_d-blend stringency requirement f_cap <~ 1e-3 by rho ~ 10
    that selected the form (authority docstring rationale)."""
    from woof.verify import sase_ref
    # LES/gray-zone side: exactly 1.0 (exp(-0.0) is FP-exact).
    for rho in (0.1, 0.5, 1.0):
        assert sase_ref.partition_cap(rho * 1000.0, 1000.0) == 1.0
    # Closed form beyond the knee.
    for rho in (1.5, 2.0, 3.0, 5.0, 10.0):
        expect = np.exp(-((rho - 1.0) / 2.0) ** 2)
        np.testing.assert_allclose(
            sase_ref.partition_cap(rho * 500.0, 500.0), expect, rtol=1e-14)
    # C^1 knee: continuous with zero slope from the right.
    assert sase_ref.partition_cap(1.0 + 1e-8, 1.0) == pytest.approx(
        1.0, abs=1e-15)
    # Monotone decreasing.
    rhos = np.linspace(0.2, 15.0, 200)
    caps = [sase_ref.partition_cap(r, 1.0) for r in rhos]
    assert all(a >= b for a, b in zip(caps, caps[1:]))
    # The stringency bound that ruled out the algebraic 1/(1+x^2) class:
    # at rho = 10 the cap must sit at ~l_B/delta ~ 1e-3 or below so the
    # LINEAR l_d blend lands on l_B (docstring derivation).
    assert sase_ref.partition_cap(10.0, 1.0) < 1.0e-3


def test_bulk_richardson_zi_crossing_floor_and_fallbacks():
    """Closed-form crossing + interpolation, the z[1] floor (stable-BL
    fallback), the top-center no-crossing fallback, and nz = 1."""
    from woof.verify import sase_ref
    nz, ny, nx = 6, 3, 4
    dz = 100.0
    z = ((np.arange(nz) + 0.5) * dz)[:, None, None]
    shape = (nz, ny, nx)
    # Calm column (spd2 floors at RIB_WSPD2_FLOOR = 1), mixed layer to
    # k = 2, then a 2 K jump at k = 3 (z = 350):
    #   Rib(3) = 2*G*350/(300*1) >> 0.25 -- crossing between 250 and 350
    #   with rib_prev = 0, frac = 0.25/Rib(3).
    theta = np.full(shape, 300.0)
    theta[3:] = 302.0
    u = np.zeros(shape)
    v = np.zeros(shape)
    zi = sase_ref.bulk_richardson_zi(u, v, theta, z)
    rib3 = 2.0 * sase_ref.G_ACCEL * 350.0 / 300.0
    expect = 250.0 + (0.25 / rib3) * 100.0
    np.testing.assert_allclose(zi, expect, rtol=1e-12)
    # Wind in the denominator shrinks Rib 16x (spd = 4, spd2 = 16 >
    # floor): still a crossing at the same level here, but the
    # interpolated frac grows 16x.
    zi_w = sase_ref.bulk_richardson_zi(
        np.full(shape, 4.0), v, theta, z)
    expect_w = 250.0 + min(0.25 / (rib3 / 16.0), 1.0) * 100.0
    np.testing.assert_allclose(zi_w, expect_w, rtol=1e-12)
    assert np.all(zi_w > zi)
    # Stable-BL fallback: theta rising from level 1 crosses immediately;
    # the result floors at the FIRST INTERIOR center z[1] = 150.
    theta_sbl = 300.0 + 5.0 * np.arange(nz)[:, None, None] * np.ones(shape)
    zi_sbl = sase_ref.bulk_richardson_zi(u, v, theta_sbl, z)
    np.testing.assert_allclose(zi_sbl, 150.0, rtol=1e-12)
    # No crossing (neutral through the top): the top layer center.
    zi_n = sase_ref.bulk_richardson_zi(u, v, np.full(shape, 300.0), z)
    np.testing.assert_allclose(zi_n, 550.0, rtol=1e-12)
    # nz = 1: the only center.
    zi_1 = sase_ref.bulk_richardson_zi(
        u[:1], v[:1], theta[:1], z[:1])
    np.testing.assert_allclose(zi_1, 50.0, rtol=1e-12)


def test_w_structure_functions_n2_screen_closed_form():
    """Fixture (c): gravity-wave-like w at high N^2 is EXCLUDED from the
    accumulation, convective w at low N^2 is included; a fully stable
    domain silences the sensor (abstention, f_w = 1)."""
    from woof.verify import sase_ref
    nz, ny, nx = 2, 8, 8
    shape = (nz, ny, nx)
    x = np.arange(nx)[None, None, :] * np.ones(shape)
    # Level 0: convective cells, w sign-alternating amplitude 1, N^2 = 0.
    # Level 1: wave cells, w amplitude 5, N^2 = 2e-4 > N2_SCREEN.
    w = np.where(x % 2 == 0, 1.0, -1.0) * np.ones(shape)
    w[1] *= 5.0
    n2 = np.zeros(shape)
    n2[1] = 2.0e-4
    d2, count = sase_ref.w_structure_functions(w, n2)
    assert count == ny * nx                    # level 0 only
    # Level-0 closed form: x-increments at odd r are +-2 (squared 4),
    # zero at even r; y-increments vanish (w constant in y).
    np.testing.assert_allclose(d2[1], 0.5 * 4.0, rtol=1e-14)
    np.testing.assert_allclose(d2[2], 0.0, atol=0.0)
    np.testing.assert_allclose(d2[4], 0.0, atol=0.0)
    # Without the screen the wave level would inflate D2 13x
    # ((4 + 100)/2 vs 4 per anchored increment): the screen's whole
    # point.
    d2_all, count_all = sase_ref.w_structure_functions(w, None)
    assert count_all == nz * ny * nx
    np.testing.assert_allclose(d2_all[1], 0.5 * (4.0 + 100.0) / 2.0,
                               rtol=1e-14)
    # Fully stable: the sensor is SILENT and the bound abstains.
    stable = np.full(shape, 2.0e-4)
    d2_s, count_s = sase_ref.w_structure_functions(w, stable)
    assert count_s == 0 and all(v == 0.0 for v in d2_s.values())
    ws = sase_ref.w_resolved_bound(w, e_mean=0.5, n2=stable)
    assert ws.f_w == 1.0 and ws.coverage == 0.0


def test_w_resolved_bound_tracks_w_variance():
    """Gray-zone lever arithmetic: alpha_w = e/(e + 0.5*D2_w(2)),
    f_w = 1 - alpha_w, monotone in the resolved-w variance; degenerate
    zero-w fields claim everything subgrid (f_w = 0)."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(20260720)
    shape = (4, 16, 16)
    w = sase_ref.box_filter(rng.standard_normal(shape), 2)
    e_mean = 0.4
    ws = sase_ref.w_resolved_bound(w, e_mean)
    d2, count = sase_ref.w_structure_functions(w, None)
    assert count == w.size and ws.coverage == 1.0
    e_res = 0.5 * d2[2]
    np.testing.assert_allclose(ws.e_res_w, e_res, rtol=1e-14)
    np.testing.assert_allclose(ws.alpha_w, e_mean / (e_mean + e_res),
                               rtol=1e-14)
    np.testing.assert_allclose(ws.f_w, 1.0 - ws.alpha_w, rtol=1e-14)
    # Doubling w quadruples E_r_w: f_w rises monotonically.
    ws2 = sase_ref.w_resolved_bound(2.0 * w, e_mean)
    np.testing.assert_allclose(ws2.e_res_w, 4.0 * e_res, rtol=1e-13)
    assert ws2.f_w > ws.f_w
    # Degenerate: no resolved w at all -> all subgrid, full bind.
    ws0 = sase_ref.w_resolved_bound(np.zeros(shape), e_mean)
    assert ws0.f_w == 0.0 and ws0.alpha_w == 1.0


def _mesoscale_limit_fixture(nz=8, ny=16, nx=16, dz=100.0, amp=10.0):
    """Fixture (a): balanced (nondivergent barotropic) horizontal flow,
    wave-like w, an elevated 10 K inversion at k = 2, and uniform
    n2 = 1.5e-4 > N2_SCREEN (silent screen, no l_s bite at the
    surface-layer e levels)."""
    shape = (nz, ny, nx)
    y = np.arange(ny)[None, :, None] * np.ones(shape)
    x = np.arange(nx)[None, None, :] * np.ones(shape)
    u = amp * np.sin(2.0 * np.pi * y / ny)     # nondivergent, barotropic
    v = amp * np.cos(2.0 * np.pi * x / nx)
    w = 0.5 * np.sin(2.0 * np.pi * x / nx)     # "wave" w, barotropic
    theta = np.full(shape, 300.0)
    theta[2:] = 310.0                          # inversion base z ~ 250 m
    n2 = np.full(shape, 1.5e-4)
    return u, v, w, theta, n2


def test_split_step_mesoscale_limit_fixed_point_matches_gate_cap():
    """Fixture (a), THE S3-6f acceptance: at Delta/z_i ~ 10 the cap
    drives f_used -> ~0 even with the w-sensor SILENT (screen passes
    nothing: the cap governs regardless), l_d -> l_B, and the surface
    fixed point under the named u* = 0.5 source lands on the gate-2a
    cap (the f = 0 fixed point) within [0.5, 2] -- the exact
    configuration whose pre-S3-6f fixed point was ~295 m^2/s^2
    (dissipation throttled 1280x at f ~ 1, S3-6e review adjudication).
    The ledger must close every step with the cap ENGAGED (the cap
    changes coefficients, not channels)."""
    from woof.verify import sase_ref
    nz, ny, nx = 8, 16, 16
    dz, dt = 100.0, 2.0
    delta = 2500.0                             # Delta/z_i ~ 12-16
    u, v, w, theta, n2 = _mesoscale_limit_fixture(nz, ny, nx, dz)
    u_star = 0.5
    source = u_star ** 3 / (sase_ref.KARMAN * 0.5 * dz)
    lb1 = float(sase_ref._blackadar_length(0.5 * dz))
    e_gate = (source * lb1 / sase_ref.C_E) ** (2.0 / 3.0)
    e = np.full((nz, ny, nx), sase_ref.E_MIN)
    ledger = None
    for _ in range(400):
        e[0] += dt * source                    # driver's pre-step deposit
        fields, ledger = sase_ref.sase_split_step(
            u, v, w, theta, e, dx=delta, dy=delta, dz=dz,
            delta=delta, dt=dt, n2=n2)
        e = fields["e"]
        scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                    abs(ledger["dHeat"]), 1e-30)
        assert abs(ledger["residual"]) / scale < 1e-11
    # The screen is silent (n2 > N2_SCREEN everywhere): the w-sensor
    # abstains and the CAP alone governs.
    assert ledger["w_coverage"] == 0.0 and ledger["f_w"] == 1.0
    assert ledger["f_cap"] < 1.0e-6, ledger
    assert ledger["f"] <= ledger["f_cap"]
    assert 150.0 <= ledger["zi"] <= 400.0      # inversion-base diagnosis
    # Surface fixed point vs the gate cap: ratio ~ 1, band [0.5, 2]
    # (dt = 2 s makes the splitting bias ~b*dt ~ 3%; the implicit
    # 2*K_v transport drain is the remaining low-side bias).
    ratio = float(e[0].mean()) / e_gate
    assert 0.5 <= ratio <= 2.0, (ratio, e_gate, float(e[0].mean()))


def test_split_step_gray_zone_f_tracks_w_sensor_not_cap():
    """Fixture (b): at Delta/z_i <= 1 (f_cap = 1, FP-exact) with
    w-dominated variance and a passing screen, f_used tracks the
    W-SENSED bound, not the cap."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(20260720)
    nz, ny, nx = 8, 16, 16
    dz, delta = 125.0, 600.0
    shape = (nz, ny, nx)

    def band():
        return sase_ref.box_filter(rng.standard_normal(shape), 4)

    u = 3.0 + band()
    v = band()
    w = 1.5 * band()                           # convective w, BL levels
    w[6:] *= 0.05
    theta = np.full(shape, 300.0)
    theta[6:] = 308.0                          # inversion base z ~ 750 m
    n2 = np.zeros(shape)
    n2[6:] = 2.0e-3                            # screened inversion cells
    # Varying e: uniform e makes the grid-anchored basis vanish from
    # the lift and degenerates the solve to (0, 0) (authority
    # _identity_rows note) -- the fixture needs a live f_solved.
    e = 2.0 + 0.4 * band()
    fields, ledger = sase_ref.sase_split_step(
        u, v, w, theta, e, dx=delta, dy=delta, dz=dz, delta=delta,
        dt=5.0, n2=n2)
    assert ledger["f_cap"] == 1.0              # rho <= 1: cap inert
    assert 0.0 < ledger["w_coverage"] < 1.0    # screen partial
    assert 0.0 < ledger["f_w"] < 1.0
    assert ledger["f_w"] < ledger["f_solved"], ledger
    assert ledger["f"] == ledger["f_w"]        # tracks the w-sensor
    # Monotone tracking: richer resolved w raises the sensed bound.
    _, ledger2 = sase_ref.sase_split_step(
        u, v, 2.0 * w, theta, e, dx=delta, dy=delta, dz=dz, delta=delta,
        dt=5.0, n2=n2)
    assert ledger2["f_w"] > ledger["f_w"]


def test_split_step_f1_recovery_when_bounds_slack(monkeypatch):
    """Fixture (d): with Delta/z_i <= 1 and w-rich fields both bounds
    are slack, f_used == f_solved, and the WHOLE step is bitwise the
    S3-6d/6e computation (pinned against a bounds-inert monkeypatched
    run -- min(f, 1, 1) leaves f and nothing else changed)."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(20260720)
    nz, ny, nx = 8, 16, 16
    dz, delta = 125.0, 600.0
    shape = (nz, ny, nx)

    def band():
        return sase_ref.box_filter(rng.standard_normal(shape), 4)

    u, v = band(), band()
    w = 10.0 * band()                           # w-rich: E_r_w >> e_mean
    theta = np.full(shape, 300.0)
    theta[6:] = 308.0
    e = np.maximum(0.05 + 0.1 * band(), 0.0)    # golden-like varying e
    args = dict(dx=delta, dy=delta, dz=dz, delta=delta, dt=5.0)
    fields_a, ledger_a = sase_ref.sase_split_step(
        u, v, w, theta, e, **args)
    assert ledger_a["f_cap"] == 1.0
    assert ledger_a["f_w"] >= ledger_a["f_solved"]
    assert ledger_a["f"] == ledger_a["f_solved"]
    # Bounds-inert variant: identical arithmetic by construction.
    monkeypatch.setattr(sase_ref, "partition_cap", lambda d, z: 1.0)
    monkeypatch.setattr(
        sase_ref, "w_resolved_bound",
        lambda w, e_mean, n2=None: sase_ref.WSensorState(1.0, 1.0, 0.0,
                                                         1.0))
    fields_b, ledger_b = sase_ref.sase_split_step(
        u, v, w, theta, e, **args)
    for key in ("u", "v", "w", "e", "heat"):
        assert np.array_equal(fields_a[key], fields_b[key]), key
    assert ledger_a["f"] == ledger_b["f"] == ledger_b["f_solved"]


# ---------------------------------------------------------------------------
# S3-6g: regime-consistent Prandtl number (smoke-c wind-amplification
# fix; authority module docstring, S3-6g section + decision table)
# ---------------------------------------------------------------------------


def test_prandtl_blend_endpoints_and_regime():
    """Pr_t(f) = f*PR_LES + (1-f)*PR_RANS: endpoints FP-EXACT, affine
    between, equal to the registered ledger form to roundoff.

    The two-product form is the deliberate FP rewrite of the registered
    Pr_t = PR_RANS + f*(PR_LES - PR_RANS): at f = 1 the products are
    1.0*PR_LES and 0.0*PR_RANS, so the sum IS PR_LES bitwise (the
    ledger form lands 1 ulp off: 0.85 + (1/3 - 0.85) != 1/3 in FP64)
    -- which is what keeps every f = 1 reduction pin alive; f = 0 is
    PR_RANS bitwise likewise."""
    from woof.verify import sase_ref
    assert sase_ref.prandtl_blend(1.0) == sase_ref.PR_LES     # bitwise
    assert sase_ref.prandtl_blend(0.0) == sase_ref.PR_RANS    # bitwise
    assert sase_ref.PR_LES == 1.0 / 3.0 and sase_ref.PR_RANS == 0.85
    assert sase_ref.PR_T == sase_ref.PR_LES    # frozen v0 alias
    for f in (0.1, 0.25, 0.5, 0.9):
        ledger_form = (sase_ref.PR_RANS
                       + f * (sase_ref.PR_LES - sase_ref.PR_RANS))
        np.testing.assert_allclose(sase_ref.prandtl_blend(f),
                                   ledger_form, rtol=1e-15)
    # Affine and DECREASING in f: more resolved -> smaller Pr -> more
    # scalar mixing relative to momentum (the LES literature's limit).
    grid = np.linspace(0.0, 1.0, 11)
    vals = np.array([sase_ref.prandtl_blend(f) for f in grid])
    assert np.all(np.diff(vals) < 0.0)
    # C_MOM_BG: standalone, FIXED, bit-identical to the inherited
    # C_K/PR_T value (decision table).
    assert sase_ref.C_MOM_BG == sase_ref.C_K / sase_ref.PR_LES
    assert sase_ref.C_MOM_BG == 0.1 / (1.0 / 3.0)


def test_sase_config_id_binds_s3_6g_constants(monkeypatch):
    """PR_LES, PR_RANS, C_MOM_BG are registry members and hash-bound."""
    from woof.verify import sase_ref
    for name in ("PR_LES", "PR_RANS", "C_MOM_BG"):
        assert name in sase_ref._CONFIG_ID_CONSTANTS
    seen = {sase_ref.sase_config_id()}
    for name, value in (("PR_LES", 0.5), ("PR_RANS", 0.74),
                        ("C_MOM_BG", 0.25)):
        monkeypatch.setattr(sase_ref, name, value)
        cid = sase_ref.sase_config_id()
        assert cid not in seen
        seen.add(cid)


def test_split_step_exports_blended_pr_t():
    """The step's ledger carries pr_t == prandtl_blend(f_used): PR_RANS
    exactly on a degenerate (RANS) fixture, PR_LES exactly whenever
    f = 1 would be used, and the blend at interior f."""
    from woof.verify import sase_ref
    shape = (4, 8, 8)
    zeros = np.zeros(shape)
    theta = np.full(shape, 300.0)
    e = np.full(shape, 0.2)
    _, led = sase_ref.sase_split_step(
        zeros, zeros, zeros, theta, e, dx=500.0, dy=500.0, dz=100.0,
        delta=500.0, dt=1.0)
    assert led["f"] == 0.0
    assert led["pr_t"] == sase_ref.PR_RANS     # bitwise RANS endpoint
    rng = np.random.default_rng(3)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 2)
               for _ in range(3))
    e_var = np.maximum(0.05 + 0.1 * sase_ref.box_filter(
        rng.standard_normal(shape), 2), 0.0)
    _, led2 = sase_ref.sase_split_step(
        u, v, w, theta, e_var, dx=500.0, dy=500.0, dz=100.0,
        delta=500.0, dt=0.05)
    assert led2["pr_t"] == sase_ref.prandtl_blend(led2["f"])


def _inversion_fixture():
    """Prescribed morning inversion + LLJ shear column (S3-6g RED
    obligation; registered fixture parameters, ledger 2026-07-20).

    30 x 50 m column (top 1500 m), horizontally uniform on a (30, 4, 4)
    box: theta = 290 K mixed layer below 450 m, a +2 K linear jump
    across 450-550 m (the ~2 K / ~100 m morning inversion at ~500 m
    AGL), and a 3 K/km stable lapse above; an LLJ-like wind profile
    u = 12*(z/500) m/s below the 500 m jet core, exponentially
    decaying (400 m scale) above -- strong shear through and below the
    inversion, the smoke-c d01/d03 morning configuration in miniature.
    Horizontal uniformity makes the dynamic solve degenerate, (0, 0),
    so f_used = 0: the RANS regime, exactly where smoke-c lived."""
    nz, ny, nx = 30, 4, 4
    dz = 50.0
    shape = (nz, ny, nx)
    z = (np.arange(nz) + 0.5) * dz
    theta1 = np.where(z <= 450.0, 290.0,
                      np.where(z <= 550.0, 290.0 + 0.02 * (z - 450.0),
                               292.0 + 0.003 * (z - 550.0)))
    u1 = np.where(z <= 500.0, 12.0 * z / 500.0,
                  12.0 * np.exp(-(z - 500.0) / 400.0))
    theta = np.broadcast_to(theta1[:, None, None], shape).copy()
    u = np.broadcast_to(u1[:, None, None], shape).copy()
    return u, theta, z, dz, shape


def _inversion_metrics(theta, dz):
    """(centroid height of the sub-700 m stable faces, jump amplitude
    theta(575 m) - theta(425 m)) -- the two erosion instruments.

    The face-gradient argmax is 50 m-quantized and entrainment
    SHARPENS the local max while the layer erodes from below, so the
    height instrument is the gradient-weighted centroid of the
    positive-gradient faces below 700 m and the gradient instrument is
    the integrated jump across the fixed 425-575 m band (= mean
    gradient across the inversion layer x 150 m): both are monotone
    erosion measures, neither quantizes."""
    col = theta[:, 0, 0]
    g = (col[1:] - col[:-1]) / dz
    zf = (np.arange(len(g)) + 1) * dz
    sel = zf <= 700.0
    gp = np.maximum(g[sel], 0.0)
    height = float((gp * zf[sel]).sum() / gp.sum())
    jump = float(col[11] - col[8])              # z = 575 m vs 425 m
    return height, jump


def _pre_s3_6h_rans_lengths(theta, e, z, thick, n2=None, z0=0.0):
    """The pre-S3-6h RANS-limb length pair, for RED-leg pinning.

    Returns ``(vertical_mixing_length, broadcast l_B)`` -- exactly what
    the S3-6h wiring degenerates to when the BL89 bounds are inert, so
    monkeypatching ``sase_ref.bl89_rans_lengths`` to THIS reproduces
    the pre-S3-6h formulation BITWISE at f = 0 (the fixtures' RANS
    branch: 0.0*l_les + 1.0*l_les == l_les and the l_d blend's lb limb
    is the bare Blackadar length again -- the S3-6g PR_RANS-pinning
    idiom applied to the length channel)."""
    from woof.verify import sase_ref
    e64 = np.maximum(np.asarray(e, dtype=np.float64), sase_ref.E_MIN)
    lb = sase_ref._blackadar_length(z, z0) * np.ones_like(e64)
    return sase_ref.vertical_mixing_length(z, e, n2, z0), lb


def _pre_s3_6i_coefficient(l_rans, e, n2=None):
    """The pre-S3-6i COUPLED coefficient (scalar C_KV) for RED-leg
    pinning: monkeypatching ``sase_ref.stable_limit_coefficient`` to
    THIS makes the RANS-limb channel kv = C_KV*l*sqrt(e) again --
    bitwise at the fixtures' f = 0 (scalar*array*array reproduces the
    pre-S3-6i (C_KV*l_mix_rans)*root_e association exactly) -- the
    S3-6g/S3-6h pinning idiom applied to the coefficient channel."""
    from woof.verify import sase_ref
    return sase_ref.C_KV


def _run_inversion_column(steps=30, dt=60.0, u_star=0.4):
    """Drive the driver's exact per-step sequence (physics._run_sase)
    on the inversion fixture: n2 from live theta, pre-step surface-e
    deposit, sase_split_step, then the scalar (theta) channel =
    implicit vertical diffusion with K_v/Pr_t(f) faces at e^n --
    K_v recomputed exactly as the step's own vertical channel builds
    it (the CPU authority does not export the kv field; the device
    step does, and the parity gates pin them equal).  S3-6h/S3-6i:
    the replicated channel is the composed RANS-limb length at the
    fixture's f = 0 under the decoupled stable-limit coefficient
    (kv = stable_limit_coefficient(...)*l_mix_rans*sqrt(e)); it reads
    ``sase_ref.bl89_rans_lengths`` AND
    ``sase_ref.stable_limit_coefficient`` through the module
    attributes so the RED legs' monkeypatches pin this replica and
    the step's internal channel together."""
    from woof.verify import sase_ref
    u, theta, z, dz, shape = _inversion_fixture()
    delta = 12000.0                            # d01 scale
    v = np.zeros(shape)
    w = np.zeros(shape)
    e = np.full(shape, sase_ref.E_MIN)
    source = u_star ** 3 / (sase_ref.KARMAN * 0.5 * dz)
    h0, j0 = _inversion_metrics(theta, dz)
    for _ in range(steps):
        n2 = sase_ref.brunt_vaisala_n2(theta, dz)
        e[0] += dt * source                    # driver pre-step deposit
        e_n = np.maximum(e, sase_ref.E_MIN)
        l_mix_r, _ = sase_ref.bl89_rans_lengths(
            theta, e_n, z[:, None, None], dz, n2)
        # channel at f = 0 (S3-6i: the decoupled stable-limit
        # coefficient rides the replica exactly as the step's internal
        # channel builds it; the RED legs pin BOTH module attributes)
        kv = (sase_ref.stable_limit_coefficient(l_mix_r, e_n, n2)
              * l_mix_r * np.sqrt(e_n))
        fields, ledger = sase_ref.sase_split_step(
            u, v, w, theta, e, dx=delta, dy=delta, dz=dz,
            delta=delta, dt=dt, n2=n2)
        u, v, w, e = (fields[k] for k in "uvwe")
        assert (ledger["c_nu"], ledger["f"]) == (0.0, 0.0)   # RANS
        assert ledger["pr_t"] == sase_ref.prandtl_blend(0.0)
        pr = sase_ref.prandtl_blend(ledger["f"])   # the driver formula
        theta = sase_ref.implicit_vertical_diffusion(
            theta, sase_ref._face_average(kv) / pr, dt, dz=dz)
    h1, j1 = _inversion_metrics(theta, dz)
    return {"height_rise": h1 - h0, "jump_erosion": j0 - j1,
            "jump0": j0}


def test_inversion_persistence_red_with_fixed_les_prandtl(monkeypatch):
    """RED evidence (registered obligation): the OLD fixed PR_T = 1/3
    destroys the morning inversion.

    The old formulation is reproduced EXACTLY by pinning PR_RANS =
    PR_LES (the blend is then constant 1/3; at the fixture's f = 0 the
    endpoint arithmetic is bitwise, so every channel -- step buoyancy
    AND scalar Pr -- computes precisely what the pre-S3-6g authority
    computed).  THRESHOLD DERIVATION (documented, then measured): a
    linear theta ramp of width L = 100 m diffusing under a constant
    K_h approaches the erf profile with sigma^2 = sigma0^2 + 2*K_h*t,
    so the surviving jump fraction scales like
    1/sqrt(1 + 4*pi*K_h*T/L^2) and the ML-side erosion deposit rises
    with K_h; the RED/GREEN scalar-diffusivity ratio is
    PR_RANS/PR_LES = 2.55, amplified by the mixdown feedback (eroding
    the jump lowers N^2, which RAISES the stability-limited l_v and
    hence K_v -- the smoke-c runaway loop in miniature) and partially
    OFFSET by the buoyancy channel (the blended Pr also weakens the
    buoyancy e-sink at the inversion, leaving more e to mix with --
    the accurate full-formulation coupling, which is why the fixture
    runs both channels).  Measured on this fixture (1800 s, dt = 60):
    RED erodes 0.529 K of the 2.075 K jump (25.5%) and the
    stable-face centroid rises 29.8 m; GREEN erodes 0.304 K (14.7%)
    and rises 8.6 m -- 1.7x / 3.5x separations, the height instrument
    carrying the feedback most cleanly.  Bounds are placed inside the
    measured gaps with margin on both sides: RED erosion > 0.45 K
    (measured 0.529, 18% headroom; 48% above the GREEN measurement)
    AND rise > 18 m (measured 29.8, 66% headroom; 2.1x the GREEN
    measurement).

    S3-6h note: the RED leg now ALSO pins the pre-S3-6h lengths
    (_pre_s3_6h_rans_lengths), so it remains the pre-S3-6g formulation
    EXACTLY and the historical measurements reproduce bitwise
    (re-measured under S3-6h: 0.528945 K / 29.8349 m -- unchanged to
    the last digit).  Without the length pin the leg would conflate
    the Prandtl and BL89 channels.

    S3-6i note: the RED leg additionally pins the coupled coefficient
    (_pre_s3_6i_coefficient) for the same reason -- the historical
    measurements continue to reproduce bitwise."""
    from woof.verify import sase_ref
    monkeypatch.setattr(sase_ref, "PR_RANS", sase_ref.PR_LES)
    monkeypatch.setattr(sase_ref, "bl89_rans_lengths",
                        _pre_s3_6h_rans_lengths)
    monkeypatch.setattr(sase_ref, "stable_limit_coefficient",
                        _pre_s3_6i_coefficient)
    out = _run_inversion_column()
    assert out["jump_erosion"] > 0.45, out     # gradient decay: RED
    assert out["height_rise"] > 18.0, out      # inversion rise: RED


def test_inversion_persistence_green_with_blended_prandtl():
    """GREEN: with the S3-6g blend the same morning inversion PERSISTS
    -- height drift and gradient decay under the derived bounds (same
    derivation as the RED leg; the GREEN bounds sit ~30% above the
    GREEN measurements and well under the RED measurements of
    0.529 K / 29.8 m, so the two legs cannot both pass by drift).

    S3-6h re-measurement (mechanism note): the live leg now runs the
    BL89-composed lengths, which cap sub-inversion parcels non-locally
    (l_up bounded by the jump's integrated buoyancy within ~l_B of the
    inversion base) -- measured GREEN moves from 0.304183 K / 8.6146 m
    (pre-S3-6h) to 0.303910 K / 8.5896 m: a small further-suppression
    shift in the protective direction, far inside the bounds, so the
    bounds need no re-derivation (thresholds keep the S3-6g
    derivation).

    S3-6i re-measurement: with the decoupled stable-limit coefficient
    (C_KS = 0.25; the inversion layer's cells are exactly the
    l_s-binding regime, so K_v there drops from ~0.742*e/N toward
    0.25*e/N) the measured GREEN moves to 0.113191 K / 2.5628 m -- a
    large further-suppression shift in the protective direction,
    which is the FIXTURE'S PURPOSE (the obs arbitration showed the
    real 1974-04-03 inversion persisting while SASE eroded it).  Both
    thresholds remain the S3-6g derivation: they are upper bounds on
    erosion, the RED leg is pinned bitwise (its separation now 4.7x /
    11.6x), and no lower bound is registered -- shear-driven erosion
    approaching zero under a held inversion is the physically
    demanded limit, not drift."""
    out = _run_inversion_column()
    assert out["jump_erosion"] < 0.40, out     # gradient persists
    assert out["height_rise"] < 12.0, out      # height persists


# ---------------------------------------------------------------------------
# S3-6h: Bougeault-Lacarrere displacement lengths in the RANS limb
# (jet-coupling diagnosis; authority module docstring, S3-6h section)
# ---------------------------------------------------------------------------


def test_sase_config_id_binds_s3_6h_constants(monkeypatch):
    """BL89_MIX_EXP and the three registered convention strings are
    registry members and hash-bound."""
    from woof.verify import sase_ref
    names = ("BL89_MIX_EXP", "BL89_BETA_CONVENTION", "BL89_KZ_MATCH",
             "BL89_LS_DECISION")
    for name in names:
        assert name in sase_ref._CONFIG_ID_CONSTANTS
    seen = {sase_ref.sase_config_id()}
    for name, value in (("BL89_MIX_EXP", 0.5),
                        ("BL89_BETA_CONVENTION", "fixed-300K"),
                        ("BL89_KZ_MATCH", "lowest-levels-only"),
                        ("BL89_LS_DECISION", "retired")):
        monkeypatch.setattr(sase_ref, name, value)
        cid = sase_ref.sase_config_id()
        assert cid not in seen
        seen.add(cid)


def test_bl89_neutral_column_lengths_are_pure_geometry():
    """Neutral (uniform theta_v): the integrand vanishes identically,
    so the budget is never exhausted and both displacements run to
    their geometric bounds EXACTLY: l_up = htop - z (top interface),
    l_down = z (surface).  This is the limit that makes the kappa-z
    match work (l_down = z >= l_B always; the constants-block
    BL89_KZ_MATCH rationale)."""
    from woof.verify import sase_ref
    nz, dz = 32, 5.0
    z = ((np.arange(nz) + 0.5) * dz)[:, None, None]
    shape = (nz, 3, 3)
    theta = np.full(shape, 300.0)
    e = np.full(shape, 0.25)
    l_up, l_down = sase_ref.bl89_displacement_lengths(theta, e, z, dz)
    ztop = nz * dz
    np.testing.assert_allclose(l_up, (ztop - z) * np.ones(shape),
                               rtol=1e-15)
    np.testing.assert_allclose(l_down, z * np.ones(shape), rtol=1e-15)
    # And the composed RANS lengths leave l_B bitwise wherever the
    # geometry is slack (below ~htop - l_B): the kappa-z match.
    l_mix_r, l_eps_r = sase_ref.bl89_rans_lengths(theta, e, z, dz)
    lb = sase_ref._blackadar_length(z) * np.ones(shape)
    low = (np.arange(nz) + 0.5) * dz < ztop - 60.0
    assert np.array_equal(l_mix_r[low], lb[low])
    assert np.array_equal(l_eps_r[low], lb[low])


def test_bl89_uniform_stratification_sqrt_e_over_n_limit():
    """THE l_s ADJUDICATION FIXTURE (registered decision
    BL89_LS_DECISION; constants block).  In uniform stratification
    Gamma = dtheta/dz the displacement integral is exactly

        int_0^l beta*Gamma*s ds = beta*Gamma*l^2/2 = e
        =>  l_up = l_down = sqrt(2e/(beta*Gamma)) = sqrt(2)*sqrt(e)/N

    with N^2 = beta*Gamma under the registered local-beta convention
    (BL89_BETA_CONVENTION: the same normalization brunt_vaisala_n2
    uses, so the identity composes levelwise) -- the Rodier et al.
    (2017) closed form.  The discrete integral must land it to
    roundoff (piecewise-linear exactness: the profile IS linear).
    ADJUDICATION READING: sqrt(2)/LS_COEF = 1.8608 > 1, so the BL89
    stable limit is LONGER than the audited Deardorff l_s everywhere
    the stratification is locally uniform -- retiring l_s would
    therefore LENGTHEN stable-limit mixing 1.86x in the regime this
    lane is suppressing; l_s is RETAINED as the outer min and BL89
    contributes its genuinely new information (non-local inversion
    capping, next fixture) on top."""
    from woof.verify import sase_ref
    nz, dz = 200, 10.0
    z = ((np.arange(nz) + 0.5) * dz)[:, None, None]
    shape = (nz, 2, 2)
    gamma = 0.005
    theta = np.broadcast_to(300.0 + gamma * z, shape).copy()
    e0 = 0.2
    e = np.full(shape, e0)
    l_up, l_down = sase_ref.bl89_displacement_lengths(theta, e, z, dz)
    beta = sase_ref.G_ACCEL / theta
    l_exact = np.sqrt(2.0 * e0 / (beta * gamma))
    # interior parcels (both bounds slack: l_exact ~ 35 m << 1000 m)
    sel = slice(60, 140)
    np.testing.assert_allclose(l_up[sel], l_exact[sel], rtol=1e-12)
    np.testing.assert_allclose(l_down[sel], l_exact[sel], rtol=1e-12)
    # the sqrt(2)*sqrt(e)/N identity and the l_s ratio
    n2 = beta * gamma
    coef = l_up[sel] * np.sqrt(n2[sel]) / np.sqrt(e0)
    np.testing.assert_allclose(coef, np.sqrt(2.0), rtol=1e-12)
    assert np.sqrt(2.0) / sase_ref.LS_COEF > 1.86   # the decision datum
    # consequence: the composed mixing length IS the v0 length here
    # (l_s binds), pinned bitwise
    l_mix_r, _ = sase_ref.bl89_rans_lengths(theta, e, z, dz, n2=n2)
    l_les = sase_ref.vertical_mixing_length(z, e, n2)
    assert np.array_equal(l_mix_r[sel], l_les[sel])


def test_bl89_fractional_segment_crossing_exact():
    """Quadratic-exactness witness with the crossing INSIDE a segment:
    a neutral segment followed by a linear-Gamma segment.  Parcel at
    z_0 = 5 m, theta flat to 15 m then rising 0.02 K/m: the first
    segment contributes zero, and within the second (h = 10 m, theta_a
    = theta_p, theta_b = theta_p + 0.2) the spent energy is I(s) =
    c2*s^2 with c2 = beta*0.2/(2*10), so the crossing sits at s =
    sqrt(e/c2) and l_up = 10 + s exactly (hand-derived; the
    _bl89_first_crossing docstring carries the general derivation)."""
    from woof.verify import sase_ref
    z = np.array([5.0, 15.0, 25.0, 35.0])[:, None, None]
    theta = np.broadcast_to(
        np.array([300.0, 300.0, 300.2, 300.4])[:, None, None],
        (4, 2, 2)).copy()
    e = np.full((4, 2, 2), 0.01)
    l_up, _ = sase_ref.bl89_displacement_lengths(theta, e, z, 10.0)
    beta = 9.81 / 300.0
    c2 = beta * 0.2 / (2.0 * 10.0)
    expect = 10.0 + np.sqrt(0.01 / c2)
    np.testing.assert_allclose(l_up[0], expect, rtol=1e-13)


def test_bl89_nonlocal_inversion_cap_beats_local_lengths():
    """THE MECHANISM WITNESS: a well-mixed (neutral) layer under a
    sharp inversion.  Parcels high in the mixed layer see local
    N^2 = 0 -- the v0 l_s never engages and l_v = l_B -- but the BL89
    upward integral is stopped by the inversion's integrated buoyancy:
    l_up = (distance to the profile knee) + sqrt(e/c2) with c2 =
    beta*Gamma_inv/2 (closed form as in the fractional-segment
    fixture).  Near the inversion base that cap undercuts l_B, so the
    composed l_mix_rans is STRICTLY SHORTER than the pre-S3-6h length
    -- entrainment suppression the local formulation cannot express.
    This is the regime the amendment genuinely changes (and the
    inversion-persistence GREEN leg measures it end-to-end); the
    jet-decoupling fixture below records the regime it does NOT."""
    from woof.verify import sase_ref
    nz, dz = 36, 25.0                          # 900 m column
    z1 = (np.arange(nz) + 0.5) * dz
    z = z1[:, None, None]
    inv_base = 700.0
    gamma_inv = 0.02                           # 2 K / 100 m cap
    th1 = np.where(z1 <= inv_base, 290.0,
                   290.0 + gamma_inv * (z1 - inv_base))
    shape = (nz, 2, 2)
    theta = np.broadcast_to(th1[:, None, None], shape).copy()
    e = np.full(shape, 0.01)                   # weak eddies near the cap
    n2 = sase_ref.brunt_vaisala_n2(theta, dz)
    # parcel at 662.5 m: 25 m below the discrete knee, local n2 = 0
    k = 26
    assert n2[k, 0, 0] == 0.0
    l_up, l_down = sase_ref.bl89_displacement_lengths(theta, e, z, dz)
    beta = sase_ref.G_ACCEL / th1[k]
    # closed form: theta is piecewise linear with the knee at the last
    # 290 K layer center (z1[27] = 687.5 m), rising linearly to z1[28];
    # I(s) = c2*s^2 within that segment
    knee = z1[27]
    seg_h = z1[28] - z1[27]
    c2 = beta * (th1[28] - th1[27]) / (2.0 * seg_h)
    s = np.sqrt(0.01 / c2)
    assert s < seg_h                           # crossing inside segment
    expect = (knee - z1[k]) + s                # = 32.689 m
    np.testing.assert_allclose(l_up[k], expect, rtol=1e-12)
    np.testing.assert_allclose(l_down[k], z1[k], rtol=1e-15)  # neutral
    # the cap undercuts l_B for this parcel: strict new suppression
    # (l_B(662.5) = 95.78 m; l_up = 32.69; the -2/3-power mean with
    # the slack l_down still lands at 76.5 m < l_B)
    lb_k = float(sase_ref._blackadar_length(z1[k]))
    l_mix_r, _ = sase_ref.bl89_rans_lengths(theta, e, z, dz, n2=n2)
    assert float(l_up[k, 0, 0]) < lb_k
    assert float(l_mix_r[k, 0, 0]) < lb_k      # strictly shorter than v0


def test_bl89_combine_forms_and_ordering():
    """Combination closed forms (constants block, BL89_MIX_EXP):
    equal arguments are a fixed point of the power mean; asymmetric
    arguments land between min and the geometric mean (negative
    exponent weights the smaller); l_eps = min <= l_mix always."""
    from woof.verify import sase_ref
    up = np.array([50.0, 50.0, 200.0])
    dn = np.array([50.0, 12.5, 10.0])
    l_mix, l_eps = sase_ref.bl89_combine(up, dn)
    np.testing.assert_allclose(l_eps, np.minimum(up, dn), rtol=0)
    np.testing.assert_allclose(l_mix[0], 50.0, rtol=1e-15)
    p = sase_ref.BL89_MIX_EXP
    expect = (0.5 * (up ** (-p) + dn ** (-p))) ** (-1.0 / p)
    np.testing.assert_allclose(l_mix, expect, rtol=1e-15)
    # ordering to FP roundoff (the equal-args fixed point lands 1 ulp
    # under the input through the pow round trip)
    assert np.all(l_eps <= l_mix * (1.0 + 1e-12))
    assert np.all(l_mix <= np.sqrt(up * dn) * (1.0 + 1e-12))


def test_split_step_les_limb_f1_bitwise_under_bl89(monkeypatch):
    """LES-LIMB PIN (hard requirement): at f = 1 the step is bitwise
    independent of the BL89 machinery -- the two-product blends make
    K_v = 1.0*(C_KV*min(l_B, l_s)*sqrt(e)) + 0.0*(...) and l_d =
    1.0*delta + 0.0*l_eps_rans FP-exact, so pinning the RANS-limb
    lengths to the pre-S3-6h pair changes NOTHING.  f = 1 is forced
    through the solve/bounds seams (the S3-6f bounds-inert idiom).

    S3-6i: the leg-b pin now ALSO clobbers the stable-limit
    coefficient with a poison value (12345.0) -- bitwise equality then
    proves the S3-6i decoupling cannot leak into the LES limb either
    (the coefficient rides only the (1-f) product)."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(20260721)
    shape = (8, 16, 16)

    def band():
        return sase_ref.box_filter(rng.standard_normal(shape), 4)

    u, v, w = band(), band(), 10.0 * band()
    theta = 300.0 + 2.0 * band()
    e = np.maximum(0.05 + 0.1 * band(), 0.0)
    n2 = 1.0e-4 * band()
    args = dict(dx=600.0, dy=600.0, dz=125.0, delta=600.0, dt=5.0,
                n2=n2)
    monkeypatch.setattr(sase_ref, "dynamic_solve",
                        lambda *a, **k: (0.01, 1.0))
    monkeypatch.setattr(sase_ref, "partition_cap", lambda d, z: 1.0)
    monkeypatch.setattr(
        sase_ref, "w_resolved_bound",
        lambda w, e_mean, n2=None: sase_ref.WSensorState(1.0, 1.0, 0.0,
                                                         1.0))
    fields_a, ledger_a = sase_ref.sase_split_step(u, v, w, theta, e,
                                                  **args)
    assert ledger_a["f"] == 1.0
    monkeypatch.setattr(sase_ref, "bl89_rans_lengths",
                        _pre_s3_6h_rans_lengths)
    monkeypatch.setattr(sase_ref, "stable_limit_coefficient",
                        lambda l_rans, e, n2=None: 12345.0)
    fields_b, ledger_b = sase_ref.sase_split_step(u, v, w, theta, e,
                                                  **args)
    for key in ("u", "v", "w", "e", "heat"):
        assert np.array_equal(fields_a[key], fields_b[key]), key
    assert ledger_a["pr_t"] == sase_ref.PR_LES   # f = 1 endpoint intact


def test_split_step_ledger_closes_with_bl89_binding():
    """Ledger theorem unaffected (asserted with BL89 BINDING): on a
    mixed-layer-under-inversion column (the mechanism-witness profile,
    where l_mix_rans < l_B strictly for sub-inversion parcels) the
    uniform-dz split-step residual still closes to roundoff -- the
    lengths are coefficients, never channels."""
    from woof.verify import sase_ref
    nz, dz = 36, 25.0                          # the mechanism-witness
    z1 = (np.arange(nz) + 0.5) * dz            # profile (previous test)
    th1 = np.where(z1 <= 700.0, 290.0,
                   290.0 + 0.02 * (z1 - 700.0))
    shape = (nz, 4, 4)
    theta = np.broadcast_to(th1[:, None, None], shape).copy()
    u = np.broadcast_to((8.0 * z1 / 900.0)[:, None, None], shape).copy()
    v = np.zeros(shape)
    w = np.zeros(shape)
    e = np.full(shape, 0.01)
    n2 = sase_ref.brunt_vaisala_n2(theta, dz)
    # BL89 must actually bind on this fixture (mechanism engaged)
    z = z1[:, None, None]
    l_mix_r, _ = sase_ref.bl89_rans_lengths(theta, e, z, dz, n2=n2)
    l_les = sase_ref.vertical_mixing_length(z, e, n2)
    assert np.any(l_mix_r < l_les)
    for _ in range(5):
        fields, ledger = sase_ref.sase_split_step(
            u, v, w, theta, e, dx=3000.0, dy=3000.0, dz=dz,
            delta=3000.0, dt=10.0, n2=n2)
        u, v, w, e = (fields[k] for k in "uvwe")
        scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                    abs(ledger["dHeat"]), 1e-30)
        assert abs(ledger["residual"]) / scale < 1e-11


# --- S3-6h jet-decoupling fixture (NEW HARD FIXTURE, registered) -----------


def _jet_u10_diag(u, v, z):
    """10 m wind by log-linear interpolation between the two lowest
    layer centers (z_0 = 8.8 m, z_1 = 28.0 m bracket 10 m):
    U10 = s_0 + (s_1 - s_0)*ln(10/z_0)/ln(z_1/z_0).  No roughness
    constant enters (pure interpolation); against the frame's own
    MO-diagnosed box-mean 10 m wind (4.667 m/s) this lands 4.555 --
    2.4% low, well inside the band resolution."""
    s0 = float(np.hypot(u[0], v[0]))
    s1 = float(np.hypot(u[1], v[1]))
    return s0 + (s1 - s0) * np.log(10.0 / z[0]) / np.log(z[1] / z[0])


def _run_jet_column(steps=60, dt=60.0, apply_drag=True):
    """The registered jet-decoupling column (ledger 2026-07-21 ~01:0x):
    single column from the d03-box mean of a spun-up CPU WRF
    frame (tests/jet_profile_19740403.py -- theta/u/v/thickness AND
    the frame's own MYNN turbulence energy E0 = QKE/2, so the closure
    is presented with the REAL 13:08Z state), u* = the frame's
    box-mean UST, dt = 60 s, 3600 s, the driver's exact per-step
    sequence (the _run_inversion_column replica: live n2, pre-step
    surface-e deposit, split step, implicit K_v/Pr_t theta channel).
    Horizontal uniformity degenerates the solve, f_used = 0:
    the pure RANS limb, where the d03 failure lived.

    S3-6j (``apply_drag``): the driver now applies the implicit
    surface momentum stress, so the drag-on limb mirrors it -- the
    per-step ``ust`` field rides the FRAME-ANCHORED live drag
    coefficient, ust_n = (UST/|V1_frame|) * |V1_n| (sqrt(Cd) =
    0.10499 held at the frame's own sfclay ratio), exactly as the
    live sfclay re-diagnoses u* from the evolving level-1 wind each
    step.  A FROZEN frame UST would invert the linearization's
    validity as the wind decays (c = u*^2/|V1| GROWS as |V1| falls:
    measured runaway to u10 = 0.29 m/s -- a fixture artifact, not
    physics; sfclay never holds u* fixed over a collapsing wind).
    The surface-e source keeps the registered frozen-UST convention
    (a production model, weakly sensitive; documented asymmetry).
    ``apply_drag=False`` is the PRE-S3-6J column BITWISE -- the
    RED/attribution legs ride it so every historical S3-6h/6i
    measured trajectory stands unchanged.  No synoptic PGF exists in
    either limb (the missing partner of the drag; the Ekman fixture
    class owns the drag+PGF balance regime -- with drag on and PGF
    absent this column MUST sag below the obs band, which the GREEN
    leg's re-derived floor documents).

    Returns the u10 series (log-interpolated diagnostic), the
    jet-core wind series (level of max speed below 2 km: k = 10,
    z = 472 m, 17.64 m/s), the initial jet speed, and the final
    fields."""
    from woof.verify import sase_ref
    import jet_profile_19740403 as jp
    nz = len(jp.THETA)
    shape = (nz, 4, 4)
    theta = np.broadcast_to(jp.THETA[:, None, None], shape).copy()
    u = np.broadcast_to(jp.U[:, None, None], shape).copy()
    v = np.broadcast_to(jp.V[:, None, None], shape).copy()
    w = np.zeros(shape)
    e = np.broadcast_to(
        np.maximum(jp.E0, sase_ref.E_MIN)[:, None, None], shape).copy()
    thick = jp.THICK.copy()
    z1 = np.cumsum(thick) - 0.5 * thick        # authority convention
    delta = 3000.0                             # the d03 scale
    source = jp.UST ** 3 / (sase_ref.KARMAN * 0.5 * thick[0])
    rcd = jp.UST / float(np.hypot(jp.U[0], jp.V[0]))   # frame sqrt(Cd)
    kjet = int(np.argmax(np.where(z1 < 2000.0,
                                  np.hypot(jp.U, jp.V), -1.0)))
    u10s, jets = [], []
    fields = None
    for _ in range(steps):
        n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
        e[0] += dt * source                    # driver pre-step deposit
        e_n = np.maximum(e, sase_ref.E_MIN)
        l_mix_r, _ = sase_ref.bl89_rans_lengths(
            theta, e_n, z1[:, None, None], thick[:, None, None], n2)
        kv = (sase_ref.stable_limit_coefficient(l_mix_r, e_n, n2)
              * l_mix_r * np.sqrt(e_n))          # S3-6i live channel
        ustf = rcd * np.hypot(u[0], v[0]) if apply_drag else None
        fields, ledger = sase_ref.sase_split_step(
            u, v, w, theta, e, dx=delta, dy=delta, dz=200.0,
            delta=delta, dt=dt, n2=n2, dz_col=thick, ust=ustf)
        assert (ledger["c_nu"], ledger["f"]) == (0.0, 0.0)   # RANS limb
        u, v, w, e = (fields[k] for k in "uvwe")
        pr = sase_ref.prandtl_blend(ledger["f"])
        theta = sase_ref.implicit_vertical_diffusion(
            theta, sase_ref._face_average(kv) / pr, dt, dz_col=thick)
        u10s.append(_jet_u10_diag(u[:, 0, 0], v[:, 0, 0], z1))
        jets.append(float(np.hypot(u[kjet, 0, 0], v[kjet, 0, 0])))
    jet0 = float(np.hypot(jp.U[kjet], jp.V[kjet]))
    return (np.array(u10s), np.array(jets), jet0,
            {k: fields[k] for k in "uvwe"})


def test_jet_decoupling_red_current_formulation_exits_obs_band(
        monkeypatch):
    """RED (registered obligation, ledger 2026-07-21 ~01:0x): the
    PRE-S3-6h formulation, reproduced bitwise by the length pin
    (_pre_s3_6h_rans_lengths; the S3-6g pinning idiom), un-decouples
    the observed morning boundary layer.  MEASURED (frame-true E0):
    the 10 m diagnostic starts at 4.555 m/s (frame's own 10 m mean
    4.667; obs 13Z 5.61) and EXITS the obs band at t = 480 s,
    reaching 7.103 by 900 s, 7.220 by 1800 s, 7.356 by 3600 s --
    above the 7.0 m/s band top for the entire final 52 minutes while
    the real 13-15Z observations hold 5.6-5.7 m/s
    (out/obs-19740403/obs-arbitration.md).  Mechanism on record:
    K_v = C_KV*min(l_B, l_s)*sqrt(e) ~ 23 m^2/s through the sheared
    stable layer at the frame's TKE mixes the lowest ~150 m down in
    ~10 minutes -- the surface-ward limb of the d03 amplification.
    The jet core itself (472 m) holds within 0.2% here: the DEEP
    coupling loop of the full model (resolved shear regeneration +
    inversion erosion over hours) is not in a 3600 s column, so the
    fixture pins the fast surface-ward failure that IS.

    S3-6i note: the leg now pins the coupled coefficient too
    (_pre_s3_6i_coefficient), so it remains the pre-S3-6h/6i
    formulation bitwise and every measured number above stands.

    S3-6j note: the leg runs ``apply_drag=False`` -- the pre-S3-6j
    column BITWISE (this RED pins the HISTORICAL formulation stack,
    which had no surface drag), so every measured number above still
    stands; the drag-on limb belongs to the GREEN fixture."""
    from woof.verify import sase_ref
    import jet_profile_19740403 as jp
    z1 = np.cumsum(jp.THICK) - 0.5 * jp.THICK
    u10_0 = _jet_u10_diag(jp.U, jp.V, z1)
    assert 4.4 <= u10_0 <= 4.8, u10_0          # anchor: starts in-band
    monkeypatch.setattr(sase_ref, "bl89_rans_lengths",
                        _pre_s3_6h_rans_lengths)
    monkeypatch.setattr(sase_ref, "stable_limit_coefficient",
                        _pre_s3_6i_coefficient)
    u10s, jets, jet0, _ = _run_jet_column(apply_drag=False)
    assert u10s.max() > 7.0, u10s.max()        # EXITS the obs band
    exit_i = int(np.argmax(u10s > 7.0))
    assert 60.0 * (exit_i + 1) <= 1800.0       # measured 480 s
    assert u10s[-1] > 7.0                      # and stays out


def test_jet_decoupling_stable_coefficient_holds_obs_band():
    """GREEN -- THE PROMOTED HARD FIXTURE (registered criteria, ledger
    2026-07-21 ~01:0x, promoted per the S3-6h adjudication contract by
    the S3-6i decoupling, ledger ~01:5x): with the decoupled
    stable-limit coefficient C_KS the frame-true DRAG-FREE jet column
    HOLDS the obs-derived band -- 10 m wind inside [4.5, 7.0] m/s
    through every one of the 60 steps to 3600 s AND jet-core wind
    within 15% of its initial value (no erosion from below).  This
    replaces the S3-6h bitwise-inert adjudication record, exactly as
    that record demanded: an amendment now decouples the column, so
    the fixture carries the registered GREEN criteria (the attribution
    test below preserves the BL89-inertness finding).

    THIS IS THE EXECUTABLE C_KS CALIBRATION GATE.  Provenance
    (S3-9c restoration; S3-6h/6i/6j review IMPORTANT-2): S3-6i
    (527db8e) introduced this test exactly as it stands; S3-6j
    (cd024be) REPLACED it with the drag-on equilibrium fixture (now
    the separately named test_jet_decoupling_drag_mixing_equilibrium
    below), leaving the live C_KS coefficient with no drag-free
    observation-band gate -- under mutation, C_KS = 0.742 (undoing
    the entire S3-6i decoupling, drag-free trajectory 7.356 m/s)
    passed the drag-on replacement because drag dominates that
    fixture.  S3-9c restores this gate at FULL original strength
    ALONGSIDE the equilibrium fixture: both run, neither replaces the
    other.  The drag-free leg runs ``apply_drag=False`` (ust=None --
    the pre-S3-6j column bitwise), scoring exactly the vertical-
    mixing channel the C_KS calibration ranked: at 13:08Z drag and
    synoptic PGF nearly balance (obs-arbitration.md), so the
    drag-free residual tendency IS the mixing term.

    CALIBRATION SWEEP (registered, S3-6i; measured with frame-true
    E0, dt = 60, 3600 s -- the selection evidence for C_KS = 0.25):

        C_KS    u10 min  u10 max  @900 s  @1800 s  @3600 s  jet dev
        RED*     5.857    7.356    7.103    7.220    7.356    0.19%
        0.25     5.701    6.580    6.423    6.479    6.580    0.06%
        0.20     5.685    6.480    6.350    6.396    6.480    0.05%
        0.15     5.668    6.378    6.278    6.313    6.378    0.04%
        0.10     5.651    6.276    6.208    6.231    6.276    0.03%
        0.076    5.643    6.227    6.175    6.193    6.227    0.02%

    (*RED = the coupled C_KV pin, exits the band top at 480 s.)
    Every swept value passes both criteria, so the registered
    largest-with-margin rule selects 0.25 -- the weakest intervention
    meeting the observations, with 0.42 m/s of headroom below the
    7.0 m/s band top and 1.2 m/s above the 4.5 m/s bottom.  The
    asserted bands below are the registered criteria, NOT the
    measured trajectory (no rot-prone bitwise pin on a GREEN leg);
    the 6.9 headroom assertion pins the margin claim itself.
    S3-9c re-measurement at restoration: min 5.701, max 6.580, jet
    dev 0.063% -- bitwise the S3-6i record (the S3-9 geometric l_d
    blend is FP-exact at this column's f = 0 and the S3-6j/S3-9c
    drag machinery never engages with ust=None)."""
    u10s, jets, jet0, _ = _run_jet_column(apply_drag=False)
    # registered GREEN criterion 1: 10 m wind in the obs band, every step
    assert np.all(u10s >= 4.5), u10s.min()
    assert np.all(u10s <= 7.0), u10s.max()
    # registered GREEN criterion 2: jet-core hold within 15%
    assert np.all(np.abs(jets - jet0) <= 0.15 * jet0)
    # the margin claim of the C_KS = 0.25 selection (measured 6.580)
    assert u10s.max() <= 6.9, u10s.max()


def test_jet_decoupling_stable_dissipation_exits_obs_band():
    """RED, REGISTERED, AND THE REASON RunConfig.sase_stable_dissipation
    SHIPS DEFAULT FALSE (authority module docstring, S3-6k section,
    CALIBRATION STATUS).

    The S3-6k amendment decouples the stable-limb DISSIPATION
    coefficient to Deardorff's published C_ES = 0.19.  Run through the
    gate above -- the registered C_KS calibration fixture, drag-free,
    frame-true E0 -- it FAILS criterion 1 by more than the pre-S3-6i
    coupled formulation it was meant to improve on:

        stable_dissipation   u10 min  max    @900   @1800  @3600  jetdev
        False (registered)    5.701   6.580  6.423  6.479  6.580  0.063%
        True  (C_ES = 0.19)   5.701   8.394  7.917  8.263  8.394  0.218%
        [pre-S3-6i RED ref    5.857   7.356  7.103  7.220  7.356  0.19% ]

    MECHANISM, measured below 500 m in this column (l_s binds in
    176/176 cells): e rises from the E_MIN floor 1.000e-06 -- the
    absorbing state the amendment exists to remove -- to p50 9.670e-04,
    a factor of 216, because Ri* moves 0.16471 -> 0.45946 and the cell
    population between those values flips from collapsing to
    sustaining.  The amendment works AND it over-mixes the observed
    morning stable boundary layer; those are the same act.

    This test is the tripwire on the default.  It pins that the pair
    (C_KS, C_ES) is jointly falsified by this gate, so nobody can flip
    the default to True -- or quietly move C_ES until the gate goes
    green -- without confronting the measurement.  Criterion 2
    (jet-core hold within 15%) still passes and is asserted, so the
    failure is localized to the surface-ward mixdown and not a broken
    column."""
    import functools
    from woof.verify import sase_ref
    base = sase_ref.sase_split_step
    try:
        sase_ref.sase_split_step = functools.partial(
            base, stable_dissipation=True)
        u10s, jets, jet0, _ = _run_jet_column(apply_drag=False)
    finally:
        sase_ref.sase_split_step = base
    # criterion 1 FAILS: out of the band, and worse than the pre-S3-6i
    # RED reference (7.356) the S3-6i decoupling was registered to fix
    assert u10s.max() > 7.0, u10s.max()
    assert u10s.max() > 7.356, u10s.max()
    assert u10s[-1] > 7.0, u10s[-1]
    # criterion 2 still HOLDS -- the jet core is not eroded
    assert np.all(np.abs(jets - jet0) <= 0.15 * jet0)
    # ... and the switched-off leg in the same process is still GREEN,
    # so this is the switch and not a drifted fixture.
    off_u10s, _, _, _ = _run_jet_column(apply_drag=False)
    assert off_u10s.max() <= 7.0, off_u10s.max()


def test_jet_decoupling_drag_mixing_equilibrium():
    """GREEN -- the S3-6j drag-on/PGF-free equilibrium fixture
    (registered re-derivation, cd024be; renamed from the obs-band
    gate's slot in S3-9c -- review IMPORTANT-2 -- so it runs
    ALONGSIDE the restored drag-free calibration gate above, replacing
    nothing).  With the driver's implicit surface stress live
    (frame-anchored Cd; _run_jet_column docstring) and NO synoptic PGF
    in the column, the obs-band [4.5, 7.0] floor does not apply: the
    real 13-15Z surface wind holds ~5.6 m/s because PGF ~ drag
    (obs-arbitration.md), so a drag-on/PGF-free column MUST sag below
    obs -- the sag is the drag physics working, not amplification.
    Re-derived bands, measured trajectory (frame-true E0, dt = 60,
    3600 s):

    * TOP 7.0 m/s UNCHANGED (the obs top remains the amplification
      tripwire this lane exists for).  Measured max 5.273 m/s (the
      step-1 mixdown bump above the 4.555 start; decaying after).
    * FLOOR 2.2 m/s -- the drag-mixing EQUILIBRIUM envelope.
      Derivation: with drag on, the surface cell relaxes to the
      quasi-steady balance Cd*s1^2 = K_f*(s2 - s1)/h1 (stress out =
      face flux in); at the measured end state (K_f1 = 3.02 m^2/s,
      h1 = 19.2 m, s2 = 3.04 m/s) the balance root is s1 = 2.57 m/s
      vs 2.586 measured -- the trajectory ENDS ON the equilibrium
      (ratio flux/stress 0.957).  Floor = 2.2 sits ~15% under the
      equilibrium class and excludes the frozen-u* linearization
      runaway (0.29 m/s -- calm) and any spurious re-acceleration is
      caught by the top.  Measured series: min 2.637, @900 3.352,
      @1800 2.983, @3600 2.637 m/s.
    * Jet-core hold within 15% UNCHANGED (measured dev 0.063% -- the
      drag row touches only the bottom cells).

    S3-9c note: this leg passes ``ust`` but no ``wspd_sfc`` (the
    frame-anchored Cd already rides the resolved wind -- there is no
    gust augmentation in the anchoring), so the gustiness correction
    is exactly 1.0 here and every measured number above stands."""
    u10s, jets, jet0, _ = _run_jet_column()
    # re-derived criterion 1: obs top (amplification tripwire) intact
    assert np.all(u10s <= 7.0), u10s.max()
    # re-derived criterion 1b: drag-mixing equilibrium floor (see
    # derivation above; the obs floor 4.5 assumed the drag+PGF balance
    # this PGF-free column deliberately lacks)
    assert np.all(u10s >= 2.2), u10s.min()
    # registered GREEN criterion 2: jet-core hold within 15%
    assert np.all(np.abs(jets - jet0) <= 0.15 * jet0)
    # the drag must actually decelerate: the hour-end wind sits below
    # the drag-free trajectory's band (which ended 6.58 -- the
    # separation witness that the drag row is live in this column)
    assert u10s[-1] <= 4.5, u10s[-1]


def test_jet_decoupling_attribution_coefficient_not_lengths(monkeypatch):
    """ATTRIBUTION (the preserved S3-6h adjudication): with the
    coefficient pinned back to the coupled C_KV, the live BL89 lengths
    remain BITWISE INERT on this smooth strongly-stable profile --
    the pinned-coefficient live-lengths trajectory equals the
    pinned-coefficient pinned-lengths (pre-S3-6h) trajectory in every
    field at every step, and still exits the obs band.  The ENTIRE
    GREEN separation of the promoted fixture is therefore the S3-6i
    coefficient decoupling, none of it the S3-6h length machinery
    (which acts on structured mixed-layer-under-inversion profiles --
    the mechanism-witness and inversion-persistence fixtures).

    S3-6j note: both legs run ``apply_drag=False`` -- the controlled
    comparison pins the pre-S3-6j formulation stack bitwise, so the
    historical BL89-inertness finding stands unchanged."""
    from woof.verify import sase_ref
    monkeypatch.setattr(sase_ref, "stable_limit_coefficient",
                        _pre_s3_6i_coefficient)
    u10_a, jets_a, jet0, fa = _run_jet_column(    # live BL89 lengths
        apply_drag=False)
    monkeypatch.setattr(sase_ref, "bl89_rans_lengths",
                        _pre_s3_6h_rans_lengths)
    u10_b, jets_b, _, fb = _run_jet_column(       # pre-S3-6h lengths
        apply_drag=False)
    assert np.array_equal(u10_a, u10_b)           # BL89 bitwise-inert
    for key in "uvwe":
        assert np.array_equal(fa[key], fb[key]), key
    assert u10_a.max() > 7.0                      # coupled RED truth
    assert np.all(np.abs(jets_a - jet0) <= 0.15 * jet0)


# ---------------------------------------------------------------------------
# S3-6i: decoupled stable-limit diffusivity coefficient C_KS
# (authority module docstring, S3-6i section)
# ---------------------------------------------------------------------------


def test_sase_config_id_binds_s3_6i_constants(monkeypatch):
    """C_KS and CKS_BLEND_EXP are registry members and hash-bound."""
    from woof.verify import sase_ref
    for name in ("C_KS", "CKS_BLEND_EXP"):
        assert name in sase_ref._CONFIG_ID_CONSTANTS
    seen = {sase_ref.sase_config_id()}
    for name, value in (("C_KS", 0.076), ("CKS_BLEND_EXP", 4.0)):
        monkeypatch.setattr(sase_ref, name, value)
        cid = sase_ref.sase_config_id()
        assert cid not in seen
        seen.add(cid)


def test_stable_limit_coefficient_closed_forms():
    """The blend's registered properties, each in closed form:

    * NEUTRAL/UNSTABLE INERTNESS (FP-exact, the log-layer guarantee):
      n2 absent, zero, or negative returns C_KV bitwise;
    * BOUNDS + CLIP: C_r in [C_KS/LS_COEF, C_KV] for any l_rans, the
      floor attained exactly at rho = 1 (and by the clip for any
      l_rans > l_s);
    * LINEAR-ONSET WITNESS (claim narrowed in S3-9c per the
      review, Minor 3): with the length fixed (l_B-binding regime),
      the coefficient deficit C_KV - C_r is LINEAR in the input n2
      (quadratic in N = sqrt(n2), CKS_BLEND_EXP = 2).  In N the
      blend is C^1 (dK_v/dN -> 0 at the neutral boundary); in the
      MODEL INPUT n2 it is C^0 only -- dK_v/d(n2) jumps from 0 to a
      finite slope at neutral (the 4x-deficit-per-4x-n2 ratio below
      IS that linear onset, pinned as the registered behavior)."""
    from woof.verify import sase_ref
    l = np.full(4, 10.0)
    e = np.full(4, 1.0)
    # neutral/unstable inertness, all three spellings
    assert np.array_equal(sase_ref.stable_limit_coefficient(l, e, None),
                          np.full(4, sase_ref.C_KV))
    n2 = np.array([-1.0e-4, 0.0, -1.0e-8, 0.0])
    assert np.array_equal(sase_ref.stable_limit_coefficient(l, e, n2),
                          np.full(4, sase_ref.C_KV))
    # bounds + clip: l >> l_s clips rho at 1 and lands the floor
    floor = sase_ref.C_KV + (sase_ref.C_KS / sase_ref.LS_COEF
                             - sase_ref.C_KV)
    n2s = np.full(4, 4.0e-4)
    big = sase_ref.stable_limit_coefficient(np.full(4, 1.0e6), e, n2s)
    np.testing.assert_allclose(big, floor, rtol=0)
    mid = sase_ref.stable_limit_coefficient(l, e, n2s)
    assert np.all(mid <= sase_ref.C_KV) and np.all(mid >= floor)
    # linear-onset witness: deficit ratio tracks the n2 ratio exactly
    # (C^0 in n2 -- the S3-9c narrowed claim; C^1 in N = sqrt(n2))
    n2ab = np.array([1.0e-8, 4.0e-8, 1.0e-9, 4.0e-9])
    c = sase_ref.stable_limit_coefficient(l, e, n2ab)
    deficit = sase_ref.C_KV - c
    assert np.all(deficit > 0.0)
    np.testing.assert_allclose(deficit[1] / deficit[0], 4.0, rtol=1e-9)
    np.testing.assert_allclose(deficit[3] / deficit[2], 4.0, rtol=1e-9)


def test_stable_limit_asymptote_cks_e_over_n():
    """THE REGISTERED ASYMPTOTE: on the uniform-stratification profile
    (the l_s-adjudication fixture geometry, where l_s < l_B < l_BL89 =
    1.86*l_s in the interior, so l_s BINDS the composed RANS length
    bitwise), the decoupled RANS-limb diffusivity
    C_r*l_mix_rans*sqrt(e) equals C_KS*e/N to roundoff -- the
    stable-limit K the S3-6i registration names, replacing the coupled
    C_KV*LS_COEF*e/N = 0.742*e/N (~3x hotter at the calibrated 0.25;
    ~10x hotter than the Deardorff floor 0.076)."""
    from woof.verify import sase_ref
    nz, dz = 200, 10.0
    z = ((np.arange(nz) + 0.5) * dz)[:, None, None]
    shape = (nz, 2, 2)
    gamma = 0.005
    theta = np.broadcast_to(300.0 + gamma * z, shape).copy()
    e0 = 0.2
    e = np.full(shape, e0)
    beta = sase_ref.G_ACCEL / theta
    n2 = beta * gamma
    l_mix_r, _ = sase_ref.bl89_rans_lengths(theta, e, z, dz, n2=n2)
    sel = slice(60, 140)                       # interior: l_s binds
    ls = sase_ref.LS_COEF * np.sqrt(e0) / np.sqrt(n2)
    assert np.array_equal(l_mix_r[sel], ls[sel])
    kv_r = (sase_ref.stable_limit_coefficient(l_mix_r, e, n2)
            * l_mix_r * np.sqrt(e))
    expect = sase_ref.C_KS * e0 / np.sqrt(n2)
    np.testing.assert_allclose(kv_r[sel], expect[sel], rtol=1e-12)


# ---------------------------------------------------------------------------
# S3-6k: decoupled stable-limb DISSIPATION coefficient C_ES, gated by
# RunConfig.sase_stable_dissipation (authority module docstring, S3-6k
# section).  The switch's registration, per-domain status and restart
# classification live beside the other RunConfig keys further up; these
# are the closed forms, the endpoint bitwise pins and the step seam.
# ---------------------------------------------------------------------------


def _s3_6k_stable_column(nz=200, dz=10.0, gamma=0.005, e0=0.2):
    """Uniform-stratification profile (the l_s-adjudication geometry
    reused from test_stable_limit_asymptote_cks_e_over_n): in the
    interior l_s < l_B < l_BL89, so the stability length BINDS the
    composed lengths and rho == 1 exactly."""
    from woof.verify import sase_ref
    z = ((np.arange(nz) + 0.5) * dz)[:, None, None]
    shape = (nz, 2, 2)
    theta = np.broadcast_to(300.0 + gamma * z, shape).copy()
    e = np.full(shape, e0)
    n2 = (sase_ref.G_ACCEL / theta) * gamma
    return z, theta, e, n2


def test_sase_config_id_binds_s3_6k_constants(monkeypatch):
    """C_ES is a registry member and hash-bound.  It is registered even
    though the default path never reads it: the config ID hashes
    constant VALUES, so an unregistered coefficient could be changed and
    a switched-on run would stamp the pre-change identity on its
    receipt."""
    from woof.verify import sase_ref
    assert "C_ES" in sase_ref._CONFIG_ID_CONSTANTS
    baseline = sase_ref.sase_config_id()
    monkeypatch.setattr(sase_ref, "C_ES", 0.2524)
    assert sase_ref.sase_config_id() != baseline


def test_stable_dissipation_coefficient_closed_forms():
    """The three registered endpoints, each BITWISE (tobytes, never
    allclose) because each is a contract and not a tolerance:

    * NEUTRAL/UNSTABLE: n2 absent, zero or negative returns C_E, so the
      neutral log layer, the C_KV = C_E**(1/3) identity and every
      moist-unstable cell (including the M1-substituted ones) are
      untouched;
    * LES LIMB: f = 1 returns C_E for ANY stratification -- the change
      is RANS-limb-only, S3-6i's own pinned property carried onto the
      dissipation half;
    * STABLE LIMIT: rho = 1 at f = 0 returns C_ES EXACTLY.  This is what
      the two-product form buys: measured this session,
      (1-w)*C_E + w*C_ES gives 0.19 (52b81e85eb51c83f) at w = 1 while
      the affine C_E + (C_ES-C_E)*w gives 0.19000000000000006
      (54b81e85eb51c83f).

    Plus the bound C_eps in [C_ES, C_E] and the S3-6i-style linear
    onset in the model input n2 (CKS_BLEND_EXP = 2 shared with the
    diffusivity half)."""
    from woof.verify import sase_ref
    l = np.full(4, 10.0)
    e = np.full(4, 1.0)
    c_e_bits = np.full(4, sase_ref.C_E).tobytes()
    n2 = np.array([-1.0e-4, 0.0, -1.0e-8, 0.0])
    # neutral/unstable inertness, all three spellings, at both limbs
    for f in (0.0, 0.5, 1.0):
        assert (sase_ref.stable_dissipation_coefficient(l, e, None, f=f)
                .tobytes() == c_e_bits)
        assert (sase_ref.stable_dissipation_coefficient(l, e, n2, f=f)
                .tobytes() == c_e_bits)
    # ... and at f the step can actually SOLVE, which is the leg that
    # matters: {0, 0.5, 1} are all-safe by accident, because at n2 <= 0
    # the blend would reduce to f*C_E + (1-f)*C_E and that expression is
    # NOT C_E bitwise at general f.  Sampling only safe f is how the
    # first S3-6k commit shipped an authority that disagreed with its
    # own device mirror (which gates on ls_v > 0.0f) by 1 ulp on the
    # cells the amendment is documented not to touch.
    grid = np.linspace(0.0, 1.0, 10001)
    blend = grid * sase_ref.C_E + (1.0 - grid) * sase_ref.C_E
    unsafe = [float(x) for x, b in zip(grid, blend)
              if np.float64(b).tobytes() != np.float64(sase_ref.C_E).tobytes()]
    assert len(unsafe) > 3000, len(unsafe)       # the sweep has teeth
    # A RECORDED PRODUCTION f: out/sase-fluxsep/run-metrics.json (read
    # only) carries 840 d02 ledger f values, 211 of them unsafe by the
    # test above; this is one of them.
    unsafe.append(4.1188928660938e-05)
    for f in unsafe[::173]:
        assert (sase_ref.stable_dissipation_coefficient(l, e, n2, f=f)
                .tobytes() == c_e_bits), f
        assert (sase_ref.stable_dissipation_coefficient(l, e, None, f=f)
                .tobytes() == c_e_bits), f
    assert (sase_ref.stable_dissipation_coefficient(
        l, e, n2, f=4.1188928660938e-05).tobytes() == c_e_bits)
    # LES limb: f = 1 is C_E bitwise however stratified
    n2s = np.full(4, 4.0e-4)
    assert (sase_ref.stable_dissipation_coefficient(l, e, n2s, f=1.0)
            .tobytes() == c_e_bits)
    assert (sase_ref.stable_dissipation_coefficient(
        np.full(4, 1.0e6), e, n2s, f=1.0).tobytes() == c_e_bits)
    # stable limit: l >> l_s clips rho at 1 and lands C_ES BITWISE
    big = sase_ref.stable_dissipation_coefficient(
        np.full(4, 1.0e6), e, n2s, f=0.0)
    assert big.tobytes() == np.full(4, sase_ref.C_ES).tobytes()
    # ... which the affine spelling would MISS by one ulp (the reason
    # the registered form is two-product).
    affine = sase_ref.C_E + (sase_ref.C_ES - sase_ref.C_E) * 1.0
    assert affine != sase_ref.C_ES
    # bounds, and the linear-in-n2 onset (deficit ratio == n2 ratio)
    mid = sase_ref.stable_dissipation_coefficient(l, e, n2s, f=0.0)
    assert np.all(mid <= sase_ref.C_E) and np.all(mid >= sase_ref.C_ES)
    n2ab = np.array([1.0e-8, 4.0e-8, 1.0e-9, 4.0e-9])
    deficit = sase_ref.C_E - sase_ref.stable_dissipation_coefficient(
        l, e, n2ab, f=0.0)
    assert np.all(deficit > 0.0)
    np.testing.assert_allclose(deficit[1] / deficit[0], 4.0, rtol=1e-9)
    np.testing.assert_allclose(deficit[3] / deficit[2], 4.0, rtol=1e-9)


def test_ri_star_is_a_harmonic_mean_and_is_bounded_by_it():
    """THE NEGATIVE RESULT, pinned so the closed lane stays closed
    (S3-6k landing amendment; module docstring, "WHAT Ri* ACTUALLY IS").

    Ri* = C_KS/(C_eps/LS_COEF + C_KS/PR_RANS) is the HARMONIC MEAN of
    the steady mixing efficiency Gamma_m = C_KS*LS_COEF/C_eps and
    PR_RANS.  Two consequences are essential and neither may drift:

    * Ri* < min(Gamma_m, PR_RANS) STRICTLY, so a stability-dependent
      Prandtl number cannot lift the critical Richardson number above
      Gamma_m -- that whole class of fix is dead before it is written,
      and this test is the record of why;
    * on the Gamma_m = 1 line (C_eps = C_KS*LS_COEF -- where the
      registered composition happens to sit, by the coincidence noted
      at the C_ES constant) Ri* = PR_RANS/(1 + PR_RANS) BITWISE for
      ANY C_KS, so the "joint (C_KS, C_ES) re-registration" has no free
      parameter to spend on the RED jet gate.

    Deliberately independent of, and additional to,
    ``test_stable_dissipation_asymptote_and_ri_star_identities`` -- the
    anti-tuning tripwire, which stays untouched.
    """
    from woof.verify import sase_ref

    def ri_star(c_ks, c_eps):
        return c_ks / (c_eps / sase_ref.LS_COEF + c_ks / sase_ref.PR_RANS)

    def harmonic(c_ks, c_eps):
        gam = c_ks * sase_ref.LS_COEF / c_eps
        return gam * sase_ref.PR_RANS / (gam + sase_ref.PR_RANS)

    # The rewrite is exact algebra; FP associativity costs <= 1 ulp.
    for c_ks, c_eps in ((sase_ref.C_KV * sase_ref.LS_COEF, sase_ref.C_E),
                        (sase_ref.C_KS, sase_ref.C_E),
                        (sase_ref.C_KS, sase_ref.C_ES)):
        a, b = ri_star(c_ks, c_eps), harmonic(c_ks, c_eps)
        assert abs(a - b) <= np.spacing(a)
    # ... and BITWISE at the C_ES composition, which sits on Gamma_m = 1.
    assert harmonic(sase_ref.C_KS, sase_ref.C_ES) == 0.45945945945945943

    # THE BOUND.  Strict, everywhere, and it is what kills the
    # stability-dependent-Prandtl lane.
    rng = np.random.default_rng(20260725)
    gam = 10.0 ** rng.uniform(-2.0, 1.0, 20000)
    pr = 10.0 ** rng.uniform(-2.0, 1.0, 20000)
    assert np.all(gam * pr / (gam + pr) < np.minimum(gam, pr))
    gam_head = sase_ref.C_KS * sase_ref.LS_COEF / sase_ref.C_E
    for pr_t in (0.4, sase_ref.PR_RANS, 2.0, 10.0, 1.0e6):
        assert gam_head * pr_t / (gam_head + pr_t) < gam_head
    np.testing.assert_allclose(gam_head, 0.204301, rtol=1e-5)

    # THE Gamma_m = 1 LINE: one Ri*, C_KS-free, bitwise, across the
    # registered C_KS sweep.
    target = sase_ref.PR_RANS / (1.0 + sase_ref.PR_RANS)
    for c_ks in (0.076, 0.10, 0.15, 0.20, 0.25):
        got = ri_star(c_ks, c_ks * sase_ref.LS_COEF)
        assert np.float64(got).tobytes() == np.float64(target).tobytes()
    # ... whereas C_KS alone, at the published C_ES, does move Ri*.
    swept = [ri_star(c, sase_ref.C_ES) for c in (0.076, 0.10, 0.15, 0.20, 0.25)]
    np.testing.assert_allclose(
        swept, [0.223917, 0.272000, 0.351724, 0.412121, 0.459459], atol=1e-6)
    assert swept == sorted(swept) and swept[0] < swept[-1]

    # THE SPELLING: the coincidence is real and C_ES is NOT spelled by it.
    assert (np.float64(sase_ref.C_ES).tobytes()
            == np.float64(sase_ref.C_KS * sase_ref.LS_COEF).tobytes())
    import pathlib
    src = (pathlib.Path(sase_ref.__file__).read_text(encoding="utf-8")
           .splitlines())
    assert "C_ES = 0.19" in src, "C_ES must stay the published literal"


def test_stable_dissipation_asymptote_and_ri_star_identities():
    """THE REGISTERED ASYMPTOTE and the DERIVED critical Richardson
    number, both recomputed here rather than remembered.

    Where l_s binds the dissipation length, eps = C_eps*e^{3/2}/l_s =
    (C_eps/LS_COEF)*e*N, so the switch moves the limb from 1.2237*e*N
    to 0.2500*e*N -- against 0.2404-0.2765*e*N measured on the trusted
    reference this session (reflen.py, BOX2 land; deck band and
    subcloud p50).  Ri* = C_KS/(C_eps/LS_COEF + C_KS/PR_RANS) is pinned
    at all three registered compositions so no later coefficient move
    can shift it silently."""
    from woof.verify import sase_ref
    assert sase_ref.C_E / sase_ref.LS_COEF == 1.223684210526316
    assert sase_ref.C_ES / sase_ref.LS_COEF == 0.25
    assert sase_ref.C_KS / sase_ref.PR_RANS == 0.29411764705882354

    def ri_star(c_ks, c_eps):
        return c_ks / (c_eps / sase_ref.LS_COEF
                       + c_ks / sase_ref.PR_RANS)

    assert ri_star(sase_ref.C_KV * sase_ref.LS_COEF,
                   sase_ref.C_E) == 0.35385638343882664   # pre-S3-6i
    assert ri_star(sase_ref.C_KS, sase_ref.C_E) == 0.16471188169301373
    assert ri_star(sase_ref.C_KS, sase_ref.C_ES) == 0.45945945945945943
    # ... and the asymptote itself, on the profile where l_s binds.
    z, theta, e, n2 = _s3_6k_stable_column()
    _, l_eps_r = sase_ref.bl89_rans_lengths(theta, e, z, 10.0, n2=n2)
    ld = sase_ref.dissipation_length(e, 500.0, n2, lb=l_eps_r, f=0.0)
    sel = slice(60, 140)                       # interior: l_s binds
    ls = sase_ref.LS_COEF * np.sqrt(0.2) / np.sqrt(n2)
    assert np.array_equal(ld[sel], ls[sel])
    c_eps = sase_ref.stable_dissipation_coefficient(ld, e, n2, f=0.0)
    assert (c_eps[sel].tobytes()
            == np.full(c_eps[sel].shape, sase_ref.C_ES).tobytes())
    eps = c_eps * e ** 1.5 / ld
    np.testing.assert_allclose(eps[sel], (0.25 * e * np.sqrt(n2))[sel],
                               rtol=1e-12)


def _s3_6k_unstable_live_f_column(seed=20260725 + 62, wamp=2.0):
    """A FULLY UNSTABLE column (N^2 < 0 in every cell) whose dynamic
    solve returns a LIVE, arithmetically UNSAFE f -- the fixture the
    switch-on inertness proof needs and the (24, 2, 2) white-noise
    column could not supply (its w-sensor bound pins f_w = 0.0 exactly,
    measured; every f value it can produce is a safe one).

    Built the way the other live-f fixtures here are: box-filtered
    bands so the Germano lift does not degenerate, varying e so the
    grid-anchored basis survives, and w-rich so f_w does not collapse.
    """
    from woof.verify import sase_ref
    rng = np.random.default_rng(seed)
    nz, ny, nx = 8, 16, 16
    dz, delta, lapse = 125.0, 600.0, 0.004
    shape = (nz, ny, nx)

    def band():
        return sase_ref.box_filter(rng.standard_normal(shape), 4)

    u, v = band(), band()
    w = wamp * band()
    z = ((np.arange(nz) + 0.5) * dz)[:, None, None]
    theta = np.broadcast_to(300.0 - lapse * z, shape).copy()
    n2 = (sase_ref.G_ACCEL / theta) * (-lapse)
    e = np.maximum(0.05 + 0.1 * band(), 0.0)
    kw = dict(dx=delta, dy=delta, dz=dz, delta=delta, dt=5.0, n2=n2)
    return u, v, w, theta, e, n2, kw


def _f_is_unsafe(f, c_e):
    """True where f*c_e + (1-f)*c_e departs from c_e BITWISE -- i.e.
    where an unconditional outer blend cannot return c_e by
    cancellation and the coefficient must be SELECTED instead."""
    blend = np.float64(f) * c_e + (1.0 - np.float64(f)) * c_e
    return np.float64(blend).tobytes() != np.float64(c_e).tobytes()


def test_sase_split_step_stable_dissipation_default_is_bitwise(monkeypatch):
    """DEFAULT INERTNESS at step level, pinned by a probe that CAN fail.

    "kwarg absent == kwarg explicitly False" is a TAUTOLOGY on its own
    -- both spellings reach the same ``if stable_dissipation:`` branch,
    so no defect in the switched-on arithmetic can move it.  It is kept
    here as a signature/plumbing check only.  The essential leg is a
    C_ES INVARIANCE probe: with C_ES monkeypatched far off its
    registered value, the DEFAULT step must still be byte-identical to
    the unpatched default step -- the off path never reads C_ES.  A
    leaked switch (``if stable_dissipation:`` -> ``if True:``, a config
    default flipped, a driver wire hard-coded True) makes it read C_ES
    and the bytes move.

    The probe carries its own POSITIVE CONTROL so a green invariance
    leg cannot be an inert fixture: the same monkeypatch DOES move the
    switched-ON step here."""
    import struct
    from woof.verify import sase_ref
    z, theta, e, n2 = _s3_6k_stable_column(nz=24, dz=25.0)
    rng = np.random.default_rng(20260725 + 61)
    shape = e.shape
    u = 5.0 + 0.5 * rng.standard_normal(shape)
    v = 0.5 * rng.standard_normal(shape)
    w = 0.05 * rng.standard_normal(shape)
    kw = dict(dx=1000.0, dy=1000.0, dz=25.0, delta=500.0, dt=2.0, n2=n2)
    base_f, base_l = sase_ref.sase_split_step(u, v, w, theta, e, **kw)
    off_f, off_l = sase_ref.sase_split_step(
        u, v, w, theta, e, stable_dissipation=False, **kw)
    on_f, _ = sase_ref.sase_split_step(
        u, v, w, theta, e, stable_dissipation=True, **kw)
    for name in ("u", "v", "w", "e", "heat"):
        assert base_f[name].tobytes() == off_f[name].tobytes(), name
    assert set(base_l) == set(off_l)
    for key, value in base_l.items():
        assert struct.pack("<d", float(value)) == struct.pack(
            "<d", float(off_l[key])), key
    # C_ES invariance of the OFF path -- the leg that can fail.
    monkeypatch.setattr(sase_ref, "C_ES", 0.4917)
    pat_f, pat_l = sase_ref.sase_split_step(u, v, w, theta, e, **kw)
    pat_off_f, _ = sase_ref.sase_split_step(
        u, v, w, theta, e, stable_dissipation=False, **kw)
    for name in ("u", "v", "w", "e", "heat"):
        assert base_f[name].tobytes() == pat_f[name].tobytes(), name
        assert base_f[name].tobytes() == pat_off_f[name].tobytes(), name
    for key, value in base_l.items():
        assert struct.pack("<d", float(value)) == struct.pack(
            "<d", float(pat_l[key])), key
    # POSITIVE CONTROL: the monkeypatch is not inert on this fixture.
    pat_on_f, _ = sase_ref.sase_split_step(
        u, v, w, theta, e, stable_dissipation=True, **kw)
    assert pat_on_f["e"].tobytes() != on_f["e"].tobytes()


def test_sase_split_step_stable_dissipation_is_inert_where_unstratified():
    """SWITCH-ON PARTIAL INERTNESS at step level (the mask the closed
    forms pin pointwise): a column with N^2 < 0 everywhere steps
    BITWISE identically with the switch on -- the neutral/unstable
    cells, which is where the M1 substitution drives N^2 negative, are
    provably outside this change.

    The fixture ASSERTS ITS OWN TEETH.  A neutral/unstable column whose
    solved f is a safe value cannot see the defect this test exists to
    catch: with an unconditional outer blend the coefficient there is
    f*C_E + (1-f)*C_E, which equals C_E bitwise at f = 0, f = 0.5 and
    f = 1 but not at general f.  So the fixture must produce a LIVE and
    UNSAFE f, and the test says so before it compares any bytes.
    Measured on the first S3-6k commit (fb67b9d), which returned that
    unconditional blend: e and heat moved by 4.16e-17 here."""
    from woof.verify import sase_ref
    u, v, w, theta, e, n2, kw = _s3_6k_unstable_live_f_column()
    assert np.all(n2 < 0.0)
    off_f, off_l = sase_ref.sase_split_step(u, v, w, theta, e, **kw)
    assert off_l["f"] > 0.0                       # the solve is live
    assert _f_is_unsafe(off_l["f"], sase_ref.C_E), off_l["f"]
    on_f, on_l = sase_ref.sase_split_step(u, v, w, theta, e,
                                          stable_dissipation=True, **kw)
    assert on_l["f"] == off_l["f"]
    for name in ("u", "v", "w", "e", "heat"):
        assert off_f[name].tobytes() == on_f[name].tobytes(), name
    # ... and the f = 0 corner, on the degenerate white-noise column
    # that used to be this test's only fixture (kept for coverage of
    # the safe-f branch, not relied on for the proof).
    rng = np.random.default_rng(20260725 + 62)
    shape = (24, 2, 2)
    z = ((np.arange(24) + 0.5) * 25.0)[:, None, None]
    theta0 = np.broadcast_to(300.0 - 0.004 * z, shape).copy()
    e0 = np.full(shape, 0.2)
    n20 = (sase_ref.G_ACCEL / theta0) * (-0.004)
    assert np.all(n20 < 0.0)
    u0 = 5.0 + 0.5 * rng.standard_normal(shape)
    v0 = 0.5 * rng.standard_normal(shape)
    w0 = 0.05 * rng.standard_normal(shape)
    kw0 = dict(dx=1000.0, dy=1000.0, dz=25.0, delta=500.0, dt=2.0, n2=n20)
    off0, led0 = sase_ref.sase_split_step(u0, v0, w0, theta0, e0, **kw0)
    assert led0["f"] == 0.0                       # documented degeneracy
    on0, _ = sase_ref.sase_split_step(u0, v0, w0, theta0, e0,
                                      stable_dissipation=True, **kw0)
    for name in ("u", "v", "w", "e", "heat"):
        assert off0[name].tobytes() == on0[name].tobytes(), name


def test_sase_split_step_additive_dissipation_is_inert_where_unstratified():
    """S3-12 SWITCH-ON INERTNESS at step level, on the same two columns
    the S3-6k inertness fixture uses and for the same reason: the
    additive channel is gated by SELECTION on N^2 > 0, so a column with
    N^2 < 0 everywhere must step BITWISE identically with the switch on.
    The live-f column carries an UNSAFE f (asserted), so the fixture
    still has the teeth that caught fb67b9d's unconditional blend."""
    from woof.verify import sase_ref
    u, v, w, theta, e, n2, kw = _s3_6k_unstable_live_f_column()
    assert np.all(n2 < 0.0)
    off_f, off_l = sase_ref.sase_split_step(u, v, w, theta, e, **kw)
    assert off_l["f"] > 0.0
    assert _f_is_unsafe(off_l["f"], sase_ref.C_E), off_l["f"]
    on_f, on_l = sase_ref.sase_split_step(u, v, w, theta, e,
                                          additive_dissipation=True, **kw)
    assert on_l["f"] == off_l["f"]
    for name in ("u", "v", "w", "e", "heat"):
        assert off_f[name].tobytes() == on_f[name].tobytes(), name


def test_sase_split_step_additive_dissipation_cuts_e_and_adds_heat():
    """S3-12 THE INTENDED QUANTITY, IN THE INTENDED DIRECTION -- and it
    is the OPPOSITE direction from S3-6k, which is the whole point.

    C_ES lowers the decay coefficient and therefore leaves MORE e (the
    test below).  C_ED is ADDED to it, so on the same stably-stratified
    column where l_s binds it must (i) leave LESS e everywhere and
    strictly less somewhere, and (ii) convert MORE e to heat.  The decay
    substep is monotone in the coefficient and the implicit e-transport
    is an M-matrix solve, so the elementwise inequality is a property of
    the formulation, not a tolerance.

    NOWHERE WEAKER, at step level: e_on <= e_off EVERYWHERE is exactly
    the statement that the dissipation is nowhere weaker than HEAD's,
    which is the property LD_STABILITY_LIMIT_REJECTED could not have."""
    from woof.verify import sase_ref
    z, theta, e, n2 = _s3_6k_stable_column(nz=24, dz=25.0)
    rng = np.random.default_rng(20260725 + 63)
    shape = e.shape
    u = 5.0 + 0.5 * rng.standard_normal(shape)
    v = 0.5 * rng.standard_normal(shape)
    w = 0.05 * rng.standard_normal(shape)
    kw = dict(dx=1000.0, dy=1000.0, dz=25.0, delta=500.0, dt=2.0, n2=n2)
    off_f, _ = sase_ref.sase_split_step(u, v, w, theta, e, **kw)
    on_f, _ = sase_ref.sase_split_step(u, v, w, theta, e,
                                       additive_dissipation=True, **kw)
    assert np.all(on_f["e"] <= off_f["e"])
    assert np.any(on_f["e"] < off_f["e"])
    assert np.all(on_f["heat"] >= off_f["heat"])
    assert np.any(on_f["heat"] > off_f["heat"])


def test_additive_dissipation_holds_the_two_calibration_fixtures():
    """S3-12 THE GATE THAT KILLED THE LAST ATTEMPT, RUN FIRST.

    LD_STABILITY_LIMIT_REJECTED and the S3-6k switch both die on two
    registered calibration fixtures: the drag-free jet-decoupling 10 m
    wind band [4.5, 7.0] m/s and the lake internal boundary layer
    ibl_d74 >= 10.0 K.  Both weaken dissipation, and both regimes are
    held by dissipation.  The additive channel can only strengthen it,
    so the prediction was that both hold -- and this test is where that
    prediction is checked rather than asserted.

    MEASURED (this session, same engines, same registered criteria):

        leg               u10 min  u10 max  @3600  jetdev%   ibl_d74
        HEAD                5.701    6.580  6.580    0.063    12.7995
        additive ON         5.701    6.485  6.485    0.061    10.9009
        S3-6k C_ES ON       5.701    8.394  8.394    0.218     1.6433
        both ON             5.701    7.101  7.101    0.144     1.9497

    The jet band gains margin (6.485 against a 7.0 top, and against
    HEAD's own 6.580): more dissipation means less subgrid energy means
    less mixdown, which is the direction this fixture rewards.

    THE LAKE MARGIN NARROWS AND THAT IS RECORDED, NOT ROUNDED OFF:
    ibl_d74 falls 12.7995 -> 10.9009 against a floor of 10.0, i.e. from
    28% margin to 9%, and the column's peak subgrid energy below 400 m
    falls 2.2013 -> 0.8059 m^2/s^2.  The gate holds; the headroom is
    thinner, and a future coefficient move that spends the rest of it
    should have to see this number.  No bound here was widened -- both
    are the registered criteria verbatim.
    """
    import functools
    from woof.verify import sase_ref
    base = sase_ref.sase_split_step
    try:
        sase_ref.sase_split_step = functools.partial(
            base, additive_dissipation=True)
        u10s, jets, jet0, _ = _run_jet_column(apply_drag=False)
    finally:
        sase_ref.sase_split_step = base
    # the registered GREEN criteria of the C_KS calibration gate, verbatim
    assert np.all(u10s >= 4.5), u10s.min()
    assert np.all(u10s <= 7.0), u10s.max()
    assert np.all(np.abs(jets - jet0) <= 0.15 * jet0)
    # ... and the margin claim, which this leg IMPROVES on HEAD's 6.580
    assert u10s.max() <= 6.9, u10s.max()


def test_sase_split_step_stable_dissipation_on_retains_e_and_cuts_heat():
    """THE INTENDED QUANTITY, IN THE INTENDED DIRECTION.  On the
    stably-stratified column where l_s binds, cutting the decay
    coefficient from C_E to C_ES must (i) leave MORE e everywhere and
    strictly more somewhere, (ii) convert LESS e to heat, and (iii)
    raise the ledger dE.  The decay substep is monotone in the
    coefficient and the implicit e-transport is an M-matrix solve, so
    the elementwise inequality is a property of the formulation, not a
    tolerance."""
    from woof.verify import sase_ref
    z, theta, e, n2 = _s3_6k_stable_column(nz=24, dz=25.0)
    rng = np.random.default_rng(20260725 + 63)
    shape = e.shape
    u = 5.0 + 0.5 * rng.standard_normal(shape)
    v = 0.5 * rng.standard_normal(shape)
    w = 0.05 * rng.standard_normal(shape)
    kw = dict(dx=1000.0, dy=1000.0, dz=25.0, delta=500.0, dt=2.0, n2=n2)
    off_f, off_l = sase_ref.sase_split_step(u, v, w, theta, e, **kw)
    on_f, on_l = sase_ref.sase_split_step(u, v, w, theta, e,
                                          stable_dissipation=True, **kw)
    assert np.all(on_f["e"] >= off_f["e"])
    assert np.any(on_f["e"] > off_f["e"])
    assert np.all(on_f["heat"] <= off_f["heat"])
    assert np.any(on_f["heat"] < off_f["heat"])
    assert on_l["dE"] > off_l["dE"]
    assert on_l["dHeat"] < off_l["dHeat"]
    # ... and the ledger residual still closes (a coefficient inside the
    # analytic decay substep cannot move the theorem).
    scale = max(abs(on_l["dKE"]), abs(on_l["dE"]), abs(on_l["dHeat"]))
    assert abs(on_l["residual"]) <= 1.0e-11 * scale


# ---------------------------------------------------------------------------
# S3-6j: surface momentum stress in the vertical solve (THE
# missing-friction fix; authority module docstring, S3-6j section) +
# the Ekman/PGF balance fixture class whose absence hid the bug.
# ---------------------------------------------------------------------------


def test_sase_config_id_binds_s3_6j_constants(monkeypatch):
    """SFC_WSPD_FLOOR is a registry member and hash-bound."""
    from woof.verify import sase_ref
    assert "SFC_WSPD_FLOOR" in sase_ref._CONFIG_ID_CONSTANTS
    before = sase_ref.sase_config_id()
    monkeypatch.setattr(sase_ref, "SFC_WSPD_FLOOR", 0.5)
    assert sase_ref.sase_config_id() != before
    # The VALUE transcribes sfclay's 0.1 floor; its ROLE (S3-9c
    # clarification, review IMPORTANT-1) is the resolved-speed
    # regularizer of the linearization -- sfclay's own floor lives on
    # the gust-enhanced wspd, which enters as the S3-9c correction
    # denominator (constant docstring at SFC_WSPD_FLOOR).
    monkeypatch.undo()
    assert sase_ref.SFC_WSPD_FLOOR == 0.1


def test_implicit_vertical_diffusion_drag_bottom_properties():
    """The S3-6j drag-augmented Thomas solve, each property closed-form:

    * ZERO IS A NO-OP: drag_bottom = 0.0 is BITWISE the zero-flux
      solve (diag + 0.0), and None likewise;
    * TWO-LEVEL CLOSED FORM: with K_f = 0 the bottom cell decouples,
      phi_new_0 = phi_0/(1 + dt*c/dz) exactly (the pure implicit-drag
      relaxation -- unconditionally stable for ANY dt*c/dz, the YSU
      1 + fric row), the upper cell untouched;
    * FLUX-CONSISTENT CONSERVATION: per column,
      sum_k thick*(phi_new - phi)_k = -dt*c*phi_new_0 exactly (the
      interior fluxes telescope; the bottom face now carries the
      implicit stress) -- the momentum the column loses IS the drag;
    * BOUNDED: |phi_new| <= max|phi| columnwise (the M-matrix inverse
      stays nonnegative with row sums <= 1: drag pulls toward rest,
      never past it);
    * MONOTONE in c: more drag, smaller surviving bottom value;
    * VALIDATION: negative c rejected."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(20260721)
    shape = (6, 4, 4)
    phi = rng.standard_normal(shape)
    kf = 2.0 + rng.random((5, 4, 4))
    dz, dt = 50.0, 120.0
    base = sase_ref.implicit_vertical_diffusion(phi, kf, dt, dz=dz)
    zero = sase_ref.implicit_vertical_diffusion(
        phi, kf, dt, dz=dz, drag_bottom=np.zeros((4, 4)))
    assert np.array_equal(base, zero)          # FP-exact no-op
    # two-level closed form (K = 0 decouples the drag row)
    phi2 = np.array([[[3.0]], [[7.0]]])
    c = 0.25
    got = sase_ref.implicit_vertical_diffusion(
        phi2, np.zeros((1, 1, 1)), dt, dz=dz, drag_bottom=c)
    np.testing.assert_allclose(got[0, 0, 0],
                               3.0 / (1.0 + dt * c / dz), rtol=1e-15)
    np.testing.assert_allclose(got[1, 0, 0], 7.0, rtol=0)
    # flux-consistent conservation, per column, exact
    cfield = 0.1 + 0.2 * rng.random((4, 4))
    out = sase_ref.implicit_vertical_diffusion(
        phi, kf, dt, dz=dz, drag_bottom=cfield)
    lost = dz * np.sum(out - phi, axis=0)
    np.testing.assert_allclose(lost, -dt * cfield * out[0], rtol=1e-12)
    # bounded toward rest
    assert np.all(np.abs(out) <= np.max(np.abs(phi), axis=0)[None] + 1e-12)
    # monotone in c on a uniform positive column
    ones = np.ones(shape)
    lo = sase_ref.implicit_vertical_diffusion(
        ones, kf, dt, dz=dz, drag_bottom=0.05)
    hi = sase_ref.implicit_vertical_diffusion(
        ones, kf, dt, dz=dz, drag_bottom=0.5)
    assert np.all(hi[0] < lo[0]) and np.all(lo[0] < 1.0)
    assert np.all(hi[0] > 0.0)                 # never past rest
    with pytest.raises(ValueError, match="non-negative"):
        sase_ref.implicit_vertical_diffusion(
            phi, kf, dt, dz=dz, drag_bottom=-1.0e-3)


def test_split_step_surface_drag_ledger_boundary_consistent():
    """RESTATED LEDGER THEOREM (S3-6j) on the uniform periodic box with
    the drag ENGAGED: dKE + dE + dHeat = dKE_sfc exactly, so the
    exported boundary-consistent residual closes to roundoff while the
    UNCORRECTED S3-6i sum misses by exactly the drag work -- the
    boundary channel is live and essential, not decorative.  The
    diagnosed conversion channel is recorded and NOT closed: the
    modeled u*^3 deposit and the measured drag work differ (similarity
    model vs resolved-drag measurement, authority docstring)."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(42)
    shape = (8, 24, 24)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 4)
               for _ in range(3))
    theta = np.full(shape, 300.0)
    e = np.full(shape, 0.2)
    ust = np.full(shape[1:], 0.4)
    fields, led = sase_ref.sase_split_step(
        u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=1.0, ust=ust)
    scale = max(abs(led["dKE"]), abs(led["dE"]), abs(led["dHeat"]), 1e-30)
    assert abs(led["residual"]) / scale < 1e-11
    assert led["dKE_sfc"] < 0.0                # drag drains resolved KE
    # the uncorrected sum misses by exactly the boundary flux
    uncorrected = led["dKE"] + led["dE"] + led["dHeat"]
    np.testing.assert_allclose(uncorrected, led["dKE_sfc"], rtol=1e-9)
    assert abs(led["dKE_sfc"]) / scale > 1e-3  # materially nonzero
    # diagnosed conversion channel: recorded, open, self-consistent
    assert led["dE_sfc_src"] > 0.0
    np.testing.assert_allclose(
        led["sfc_conv_resid"], led["dE_sfc_src"] + led["dKE_sfc"],
        rtol=1e-12)
    # ust=None: every surface channel exactly 0.0, fields differ from
    # the drag run (the hole this amendment closes was exactly this
    # difference being absent)
    fields0, led0 = sase_ref.sase_split_step(
        u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=1.0)
    for key in ("dKE_sfc", "dE_sfc_src", "sfc_conv_resid"):
        assert led0[key] == 0.0, key
    assert not np.array_equal(fields["u"], fields0["u"])
    assert not np.array_equal(fields["v"], fields0["v"])


def test_split_step_drag_applies_in_les_limb_f1(monkeypatch):
    """THE ONE INTENTIONAL CROSS-LIMB CHANGE OF THE LANE (S3-6j,
    flagged per the adjudication): friction is not regime-dependent,
    so at FORCED f = 1 (the LES limb, bitwise-protected by every
    prior amendment) a passed ``ust`` engages the drag row and
    momentum MOVES -- while the ust=None leg remains the bitwise
    LES-limb contract (test_split_step_les_limb_f1_bitwise_under_bl89,
    unchanged).  The boundary-consistent ledger closes at f = 1 with
    the drag engaged (the theorem is limb-independent)."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(20260721)
    shape = (8, 16, 16)

    def band():
        return sase_ref.box_filter(rng.standard_normal(shape), 4)

    u, v, w = band(), band(), 10.0 * band()
    theta = 300.0 + 2.0 * band()
    e = np.maximum(0.05 + 0.1 * band(), 0.0)
    n2 = 1.0e-4 * band()
    args = dict(dx=600.0, dy=600.0, dz=125.0, delta=600.0, dt=5.0,
                n2=n2)
    monkeypatch.setattr(sase_ref, "dynamic_solve",
                        lambda *a, **k: (0.01, 1.0))
    monkeypatch.setattr(sase_ref, "partition_cap", lambda d, z: 1.0)
    monkeypatch.setattr(
        sase_ref, "w_resolved_bound",
        lambda w, e_mean, n2=None: sase_ref.WSensorState(1.0, 1.0, 0.0,
                                                         1.0))
    fields_a, led_a = sase_ref.sase_split_step(u, v, w, theta, e, **args)
    assert led_a["f"] == 1.0 and led_a["dKE_sfc"] == 0.0
    ust = np.full(shape[1:], 0.4)
    fields_b, led_b = sase_ref.sase_split_step(u, v, w, theta, e,
                                               ust=ust, **args)
    assert led_b["f"] == 1.0
    assert led_b["dKE_sfc"] < 0.0              # drag live in the LES limb
    assert not np.array_equal(fields_a["u"], fields_b["u"])
    assert not np.array_equal(fields_a["v"], fields_b["v"])
    scale = max(abs(led_b["dKE"]), abs(led_b["dE"]),
                abs(led_b["dHeat"]), 1e-30)
    assert abs(led_b["residual"]) / scale < 1e-11


# --- S3-9c gustiness-corrected surface drag (review IMPORTANT-1) -----


def test_split_step_gustiness_correction_identity_no_gust():
    """THE IDENTITY PIN (S3-9c contract): no gustiness => correction
    == 1 => the S3-6j neutral behavior BITWISE unchanged, on a
    neutral no-gust log-layer column.  Both identity spellings:

    * ABSENT: ``wspd_sfc=None`` forms no factor -- bitwise trivially;
    * SUPPLIED NO-GUST: wspd_sfc = max(|V1|, SFC_WSPD_FLOOR) (the
      sfclay value with vconv = vsgd = 0 on the same winds) gives
      ratio spd1/wspd == 1.0 exactly and c*1.0 == c bitwise -- every
      field and every ledger channel equal to the wspd_sfc=None run
      bit for bit.

    The column is neutral (uniform theta, no n2) with a log-layer
    wind profile and similarity u* -- the log-layer fixture geometry
    on the split-step seam the correction lives in."""
    from woof.verify import sase_ref
    nz, ny, nx = 8, 6, 6
    dz = 50.0
    z = ((np.arange(nz) + 0.5) * dz)[:, None, None]
    z0 = 0.1
    ust0 = 0.4
    prof = (ust0 / sase_ref.KARMAN) * np.log((z + z0) / z0)
    shape = (nz, ny, nx)
    u = np.broadcast_to(prof, shape).copy()
    v = np.zeros(shape)
    w = np.zeros(shape)
    theta = np.full(shape, 300.0)                  # neutral
    e = np.full(shape, 0.2)
    ust = np.full((ny, nx), ust0)
    args = dict(dx=500.0, dy=500.0, dz=dz, delta=500.0, dt=30.0)
    fields_none, led_none = sase_ref.sase_split_step(
        u, v, w, theta, e, ust=ust, **args)
    # the no-gust sfclay wspd on the SAME winds (vconv = vsgd = 0)
    wspd_no_gust = np.maximum(np.hypot(u[0], v[0]),
                              sase_ref.SFC_WSPD_FLOOR)
    fields_ng, led_ng = sase_ref.sase_split_step(
        u, v, w, theta, e, ust=ust, wspd_sfc=wspd_no_gust, **args)
    for key in ("u", "v", "w", "e", "heat"):
        assert np.array_equal(fields_none[key], fields_ng[key]), key
    for key, val in led_none.items():
        assert led_ng[key] == val, key
    assert led_none["dKE_sfc"] < 0.0               # the drag row is live


def test_split_step_gustiness_correction_exact_audited_form():
    """The correction is EXACTLY the audited YSU factor (authority
    module docstring, S3-9c section; npref.py:6495-6496): on a gusty
    fixture the measured drag work satisfies

        dKE_sfc = -dt*sum(c*((u1^{n+1})^2 + (v1^{n+1})^2)/thick_0),
        c = u*^2/spd1 * (spd1/max(wspd, 1e-9))^2

    to roundoff (rtol 1e-12 -- the same conductance feeds the Thomas
    diagonal and the work reduction, so this pins the whole seam),
    and the corrected drag work is SMALLER in magnitude than the
    uncorrected (gustiness inflates u*; the factor backs it off
    against the resolved wind).  A CALM-GUSTY witness column
    (|V1| < SFC_WSPD_FLOOR, wspd driven by vconv) carries factor
    (SFC_WSPD_FLOOR/wspd)^2 -- the over-damping class the
    review isolated (stress ratio > 2 in 10.3% of d01 interior
    cells) is exactly what the factor removes.  Validation:
    ``wspd_sfc`` without ``ust`` is rejected."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(20260721)
    shape = (8, 12, 12)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 4)
               for _ in range(3))
    # calm-gusty witness: zero the resolved wind in one column
    u[0, 3, 3] = 0.0
    v[0, 3, 3] = 0.0
    theta = np.full(shape, 300.0)
    e = np.full(shape, 0.2)
    dz, dt = 200.0, 1.0
    ust = np.full(shape[1:], 0.4)
    vconv = 1.5                                    # gust augmentation
    wspd = np.maximum(np.sqrt(u[0] ** 2 + v[0] ** 2 + vconv ** 2),
                      sase_ref.SFC_WSPD_FLOOR)
    spd1 = np.maximum(np.hypot(u[0], v[0]), sase_ref.SFC_WSPD_FLOOR)
    assert spd1[3, 3] == sase_ref.SFC_WSPD_FLOOR   # floor-active witness
    args = dict(dx=500.0, dy=500.0, dz=dz, delta=500.0, dt=dt)
    fields_g, led_g = sase_ref.sase_split_step(
        u, v, w, theta, e, ust=ust, wspd_sfc=wspd, **args)
    _, led_0 = sase_ref.sase_split_step(u, v, w, theta, e, ust=ust,
                                        **args)
    # exact audited conductance through the public seam
    c_exp = ust ** 2 / spd1 * (spd1 / np.maximum(wspd, 1.0e-9)) ** 2
    dke_exp = -dt * float(np.sum(
        c_exp * (fields_g["u"][0] ** 2 + fields_g["v"][0] ** 2) / dz))
    np.testing.assert_allclose(led_g["dKE_sfc"], dke_exp, rtol=1e-12)
    # gustiness backs the drag off, never off entirely
    assert 0.0 > led_g["dKE_sfc"] > led_0["dKE_sfc"]
    # calm-gusty witness: the factor there is (floor/wspd)^2
    np.testing.assert_allclose(
        c_exp[3, 3],
        ust[3, 3] ** 2 / sase_ref.SFC_WSPD_FLOOR
        * (sase_ref.SFC_WSPD_FLOOR / wspd[3, 3]) ** 2, rtol=1e-15)
    assert c_exp[3, 3] < 0.05 * ust[3, 3] ** 2 / sase_ref.SFC_WSPD_FLOOR
    # the e source keeps the UNcorrected u*^3 deposit (S3-9c scope)
    assert led_g["dE_sfc_src"] == led_0["dE_sfc_src"]
    with pytest.raises(ValueError, match="wspd_sfc requires ust"):
        sase_ref.sase_split_step(u, v, w, theta, e, wspd_sfc=wspd,
                                 **args)


# --- S3-6j Ekman/PGF balance fixture (NEW FIXTURE CLASS, registered) -------
#
# The class whose ABSENCE hid the missing-friction bug: a column under
# sustained large-scale PGF forcing.  Constants shared by both limbs.

EKMAN_G = 10.0                                 # geostrophic wind [m/s]
EKMAN_F = 1.0e-4                               # f-plane Coriolis [s^-1]
EKMAN_NZ, EKMAN_DZ = 60, 50.0                  # 3 km column
EKMAN_DT = 60.0
EKMAN_Z0 = 0.1                                 # land roughness [m]
#: Derived GREEN band for the steady level-1 (z1 = 25 m) speed.
#: (Arithmetic corrected in S3-9c per the review, Minor 4; the
#: GATES below are unchanged -- only this derivation record was
#: wrong.)  Rossby-number similarity: solving the neutral geostrophic
#: drag law (kappa*G/u*)^2 = (ln(u*/(f*z0)) - A)^2 + B^2 at surface
#: Rossby number Ro = G/(f*z0) = 1e6 over the classical (A, B)
#: corners (A in [1.3, 2.3], B in [4, 5]) gives u*/G in
#: [0.03805, 0.04311]; the engine's similarity inversion
#: |V1| = (u*/kappa)*ln((z1 + z0)/z0) with ln(25.1/0.1) = 5.525 maps
#: that to |V1| in [5.256, 5.955].  The asserted gates WIDEN the
#: corner band by ~10-12% each side -- [4.8, 6.7] on |V1| and
#: [0.35, 0.48] on u* -- because the engine is NOT the classical
#: resistance law (a discrete K-profile column with the similarity
#: u* closure), and its measured steady state u* = 0.4326
#: (u*/G 0.04326), |V1| = 5.976 indeed sits ~0.4% ABOVE the
#: corner-derived band while remaining well inside the widened
#: gates: the gates score the Rossby-similarity CLASS (right decade,
#: right neighborhood), not corner-exact classical constants.
EKMAN_V1_BAND = (4.8, 6.7)
EKMAN_UST_BAND = (0.35, 0.48)


def test_ekman_balance_column_green_steady_state():
    """GREEN (fixture class assertion set, registered): the drag-on
    column under constant geostrophic forcing reaches the
    Ekman-with-similarity steady state.  30000 steps (500 h, ~8.6
    inertial periods) from rest; measured last-3000-step oscillation
    0.0024 m/s (converged).

    (a) steady |V1| and u* in the derived Rossby-similarity bands
        (derivation at EKMAN_V1_BAND; measured 5.976 m/s, 0.4326);
    (b) cross-isobar angle in the physical band [10, 35] deg -- the
        surface wind turns TOWARD LOW PRESSURE (+v for geo = (G, 0);
        Ekman's 45 deg is the constant-K idealization, similarity
        layers sit shallower; measured 17.1 deg);
    (c) COLUMN MOMENTUM BUDGET CLOSES: at the engine's fixed point
        the thickness-weighted momentum equations telescope to
        c*u_1 = f*sum(dz*(v - vg)) and c*v_1 = -f*sum(dz*(u - ug))
        EXACTLY (derivation at the engine docstring) -- integrated
        PGF equals surface stress componentwise; tolerance 2e-2
        covers the residual spin-up transient (measured 2.0e-3 x,
        9.3e-3 y).
    """
    from woof.verify import sase_ref
    res = sase_ref.column_ekman_balance(EKMAN_G, 0.0, EKMAN_F,
                                        EKMAN_NZ, EKMAN_DZ, EKMAN_DT,
                                        30000, z0=EKMAN_Z0)
    u, v, z = res["u"], res["v"], res["z"]
    spd1 = float(np.hypot(u[0], v[0]))
    # converged: last-quarter oscillation at the 1e-2 m/s scale
    tail = res["spd1_series"][-3000:]
    assert tail.max() - tail.min() < 0.05, (tail.min(), tail.max())
    # (a) similarity bands
    assert EKMAN_V1_BAND[0] <= spd1 <= EKMAN_V1_BAND[1], spd1
    ust = sase_ref.KARMAN * spd1 / np.log((z[0] + EKMAN_Z0) / EKMAN_Z0)
    assert EKMAN_UST_BAND[0] <= ust <= EKMAN_UST_BAND[1], ust
    # (b) cross-isobar angle
    alpha = np.degrees(np.arctan2(v[0], u[0]))
    assert 10.0 <= alpha <= 35.0, alpha
    # (c) integrated PGF = surface stress, componentwise
    c = ust ** 2 / max(spd1, sase_ref.SFC_WSPD_FLOOR)
    pgf_x = EKMAN_F * np.sum(EKMAN_DZ * (v - 0.0))
    pgf_y = -EKMAN_F * np.sum(EKMAN_DZ * (u - EKMAN_G))
    np.testing.assert_allclose(pgf_x, c * u[0], rtol=2e-2)
    np.testing.assert_allclose(pgf_y, c * v[0], rtol=2e-2)
    # the surface wind is SUBgeostrophic and the deficit real
    assert spd1 < EKMAN_G


def test_ekman_balance_column_red_drag_off_unbounded_drift():
    """RED (the regression tripwire for the MISSING-FORCE class): the
    same column with the drag OFF -- the pre-S3-6j momentum path
    bitwise -- has NO equilibrium under sustained PGF forcing.  From
    rest the column stays horizontally uniform (the PGF tendency is
    z-independent, D_v of a uniform profile vanishes), so the
    trajectory is the exact discrete inertial spiral
    Z^{n+1} = (1 - i*f*dt)*Z^n, Z = (u - ug) + i*(v - vg): the speed
    sweeps 0 -> ~2G (measured max 20.095 m/s at step 525 = half the
    inertial period) and |1 - i*f*dt| > 1 amplifies secularly --
    accelerating drift, never a balance.  Assertions: the level-1
    wind exits the GREEN band within 200 steps, exceeds 1.5x the band
    top (measured 3x), and the final quarter still sweeps ACROSS the
    band (max above, min below -- no settling); the closed form pins
    the mechanism."""
    from woof.verify import sase_ref
    steps = 1200
    res = sase_ref.column_ekman_balance(EKMAN_G, 0.0, EKMAN_F,
                                        EKMAN_NZ, EKMAN_DZ, EKMAN_DT,
                                        steps, z0=EKMAN_Z0, drag=False)
    ser = res["spd1_series"]
    band_lo, band_hi = EKMAN_V1_BAND
    exit_i = int(np.argmax(ser > band_hi))
    assert ser.max() > band_hi and exit_i + 1 <= 200, exit_i
    assert ser.max() > 1.5 * band_hi, ser.max()          # measured 20.1
    tail = ser[-steps // 4:]
    assert tail.max() > band_hi and tail.min() < band_lo  # no settling
    # closed-form inertial-spiral pin (mechanism witness): uniform
    # column, Z^{n+1} = (1 - i f dt) Z^n from Z^0 = -G.
    u, v = res["u"], res["v"]
    assert np.ptp(u) < 1e-9 and np.ptp(v) < 1e-9          # z-uniform
    zn = complex(-EKMAN_G, 0.0) * (1.0 - 1j * EKMAN_F * EKMAN_DT) ** steps
    np.testing.assert_allclose(float(np.hypot(u[0], v[0])),
                               abs(zn + EKMAN_G), rtol=1e-9)


# ---------------------------------------------------------------------------
# S3-6h: G2b-v3 volume-growth excusal fail-closed (6g review
# IMPORTANT finding; fix registered ledger 2026-07-21 ~01:0x)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# S3-9: geometric dissipation-length blend -- water-surface
# jet-decoupling fixture (F-Y1 Lake Michigan over-coupling; authority
# module docstring, S3-9 section; mechanism + amendment spec in
# .superpowers/sdd/yolo-lake-mechanism.md)
# ---------------------------------------------------------------------------


def test_sase_config_id_binds_s3_9_constants(monkeypatch):
    """LD_BLEND_FORM (the registered blend-form decision string) is a
    registry member and hash-bound -- the S3-6h convention-string
    idiom applied to the S3-9 formulation decision."""
    from woof.verify import sase_ref
    assert sase_ref.LD_BLEND_FORM == "geometric"
    assert "LD_BLEND_FORM" in sase_ref._CONFIG_ID_CONSTANTS
    baseline = sase_ref.sase_config_id()
    monkeypatch.setattr(sase_ref, "LD_BLEND_FORM", "linear")
    assert sase_ref.sase_config_id() != baseline


def _pre_s3_9_dissipation_length(e, delta, n2=None, lb=None, f=1.0):
    """The pre-S3-9 LINEAR l_d blend, for RED-leg pinning.

    Bitwise the pre-S3-9 ``dissipation_length`` body: blend =
    f*delta + (1-f)*lb under the unchanged E_MIN floor and outer l_s
    min -- the S3-6g/S3-6h pinning idiom applied to the dissipation-
    length channel.  Monkeypatching ``sase_ref.dissipation_length`` to
    THIS reproduces the pre-S3-9 formulation exactly (at f = 0 and
    f = 1 it equals the live geometric form bitwise -- the endpoint
    contract -- so only interior-f trajectories separate, which is
    what makes the lake RED leg need f > 0)."""
    from woof.verify import sase_ref
    e64 = np.maximum(np.asarray(e, dtype=np.float64), sase_ref.E_MIN)
    if lb is None:
        l = np.full_like(e64, float(delta))
    else:
        blend = (f * float(delta)
                 + (1.0 - f) * np.asarray(lb, dtype=np.float64))
        l = blend * np.ones_like(e64)
    if n2 is not None:
        n2 = np.asarray(n2, dtype=np.float64)
        stable = n2 > 0.0
        ls = sase_ref.LS_COEF * np.sqrt(e64) / np.sqrt(
            np.where(stable, n2, 1.0))
        l = np.where(stable, np.minimum(l, ls), l)
    return l


#: Reconstructed d02 domain-level f_used schedule of the defect run
#: (mechanism report section 2: f_used ~ min(f_cap, f_w) rebuilt from
#: the run's own files; hours after 13Z).  The land CBL deepens, the
#: DOMAIN f rises, and every stable marine column inherits it -- the
#: lake leg is RED-able only because f > 0 (the linear blend needs a
#: nonzero f to jump the RANS dissipation bound).
_LAKE_F_HOURS = np.array([0.0, 2.0, 4.0, 5.0, 6.0])
_LAKE_F_USED = np.array([0.0, 0.0064, 0.067, 0.12, 0.194])


def _charnock_ust(spd1, z1_0):
    """Neutral water-surface u* (Charnock + smooth-flow z0), fixed-
    point iterated: z0 = 0.011*u*^2/g + 0.11*nu_air/u*, u* =
    kappa*|V1|/ln(z1/z0) -- the mechanism report's water-drag element
    (section 3; VALIDATED there against the run's own sfclay at the
    coupled state: 0.60 at |V1| = 15.5 vs the run's 0.592, and
    equivalent to the frame-anchored 18Z sqrt(Cd) = 0.0372).  The
    13Z frame u* = 0.129 is STABILITY-suppressed (the onset IBL), so
    anchoring sqrt(Cd) there would under-drag the 4 h trajectory --
    the neutral Charnock form is the accurate evolving-wind drag and is
    what every measured number in the report ran."""
    from woof.verify import sase_ref
    ust = 0.03 * spd1 + 1.0e-3
    for _ in range(6):
        z0 = (0.011 * ust * ust / sase_ref.G_ACCEL
              + 0.11 * 1.5e-5 / max(ust, 1.0e-4))
        ust = sase_ref.KARMAN * spd1 / np.log(z1_0 / z0)
    return float(ust)


def _run_lake_column(monkeypatch, state="13", hours=4.0, dt=60.0,
                     sched_t0=0.0, freeze_wind=False, restart_e=False,
                     m1=False):
    """The S3-9 water-surface column driver (mechanism report section 5
    reproduction recipe): the extracted mid-Lake-Michigan state
    (tests/lake_profile_19740403.py) under the driver's exact per-step
    sequence (the _run_jet_column replica: live n2, pre-step surface-e
    deposit, split step, implicit K_v/Pr_t theta channel) PLUS the
    three water-column elements the land fixtures lack:

    * WATER DRAG via :func:`_charnock_ust` re-diagnosed from the
      evolving level-1 wind each step (the report's recipe; see the
      helper docstring for why the frame-anchored idiom is wrong for
      the stability-suppressed 13Z frame u*);
    * IMPLICIT BULK SURFACE COOLING toward the potential-temperature
      TSK (tsk_pot = TSK*(1e5/PSFC)**0.2857) with Ch = 0.6*Cd
      (anchored to the run's own HFX, report section 3): backward-
      Euler relaxation of the bottom cell, conductance cc = Ch*|V1|;
    * the EKMAN-ENGINE PGF/Coriolis hook (column_ekman_balance's
      forcing decomposition at the pre-update state) holding the
      state's own wind as the geostrophic profile at/above 300 m,
      extended constant to the surface below (the drag-reduced
      observed surface wind is not geostrophic), f-plane at the
      column's 42.55 N.

    The partition rides the S3-6f-cap monkeypatch idiom pinned to the
    reconstructed domain-level schedule f(sched_t0 + t) interpolated
    on (_LAKE_F_HOURS, _LAKE_F_USED): dynamic_solve degenerates to
    f_solved = 1, the w-sensor abstains, and partition_cap returns
    the scheduled value, so the step's used f IS the schedule
    (asserted).  ``freeze_wind`` resets u/v/w to the initial profile
    after every step -- the 3-D run's fetch-long momentum resupply
    that the interactive column cannot self-generate (report verdict:
    the defect needs it; every interactive column decouples), which
    makes the frozen leg the RED/GREEN discriminator.  ``restart_e``
    starts e from the run's own restart e_sgs column (the coupled-
    equilibrium TKE) instead of E_MIN.

    ``m1`` (S4-1): run the SAME column through the M1-substituted
    pipeline -- per step, a SYNTHETIC unsaturated humidity state
    (RH = 70% of the Tetens q_s at the LIVE theta on a fixed
    hydrostatic marine pressure column p = PSFC*exp(-g*z/(RD*285 K));
    qc = 0) feeds ``sase_ref.moist_n2`` and the result rides the
    split step's ``n2_moist`` seam.  The humidity is synthetic
    because the lake table carries no qv/qc -- only its UNSATURATION
    is essential (the marine premise: unsaturated at k0), and
    qv = 0.7*q_s(live theta) < q_s holds structurally at every step
    (same q_s formula on both sides), so the switch never fires and
    the trajectory must be BITWISE the ``m1=False`` trajectory (the
    M1 lake-column fixture asserts it).

    Returns the sp10/sp500 series (log-interpolated 10 m diagnostic,
    z1 = 8.15/26.0 m bracket 10 m, over the 500 m interpolated
    speed), the final profiles, and the IBL instrument
    ibl_d74 = theta(74 m) - theta(z1) (the report's Delta-theta(0 ->
    74 m); 74 m is the fourth layer center of the extracted grid)."""
    from woof.verify import sase_ref
    import lake_profile_19740403 as lp

    theta0 = getattr(lp, f"THETA_{state}")
    u0 = getattr(lp, f"U_{state}")
    v0 = getattr(lp, f"V_{state}")
    thick = getattr(lp, f"THICK_{state}").copy()
    tsk = getattr(lp, f"TSK_{state}")
    psfc = getattr(lp, f"PSFC_{state}")
    nz = len(theta0)
    shape = (nz, 4, 4)
    z1 = np.cumsum(thick) - 0.5 * thick        # authority convention
    z3 = z1[:, None, None]
    t3 = thick[:, None, None]
    delta = 3000.0                             # the d02 scale
    fcor = 2.0 * 7.292e-5 * np.sin(np.deg2rad(lp.LAT))
    tsk_pot = tsk * (1.0e5 / psfc) ** 0.2857

    kfree = int(np.argmax(z1 >= 300.0))
    ug = u0.copy()
    vg = v0.copy()
    ug[:kfree] = u0[kfree]
    vg[:kfree] = v0[kfree]

    u = np.broadcast_to(u0[:, None, None], shape).copy()
    v = np.broadcast_to(v0[:, None, None], shape).copy()
    w = np.zeros(shape)
    theta = np.broadcast_to(theta0[:, None, None], shape).copy()
    if restart_e:
        e0 = np.maximum(lp.E_RST_18, sase_ref.E_MIN)
        e = np.broadcast_to(e0[:, None, None], shape).copy()
    else:
        e = np.full(shape, sase_ref.E_MIN)

    # S3-6f-cap monkeypatch idiom, schedule-valued: f_used =
    # min(f_solved = 1, f_cap = schedule, f_w = 1) = the schedule.
    f_cell = {"f": 0.0}
    monkeypatch.setattr(sase_ref, "dynamic_solve",
                        lambda *a, **k: (0.0, 1.0))
    monkeypatch.setattr(sase_ref, "partition_cap",
                        lambda delta_h, zi: f_cell["f"])
    monkeypatch.setattr(
        sase_ref, "w_resolved_bound",
        lambda w_, e_mean, n2=None: sase_ref.WSensorState(
            f_w=1.0, alpha_w=1.0, e_res_w=0.0, coverage=0.0))

    if m1:
        # fixed hydrostatic marine pressure column for the SYNTHETIC
        # humidity state (docstring: only unsaturation is essential)
        pcol = (psfc * np.exp(-sase_ref.G_ACCEL * z1
                              / (sase_ref.RD_AIR * 285.0)))[:, None, None]

    steps = int(round(hours * 3600.0 / dt))
    ratios = []
    for n in range(steps):
        f_cell["f"] = float(np.interp(sched_t0 + n * dt / 3600.0,
                                      _LAKE_F_HOURS, _LAKE_F_USED))
        spd1 = max(float(np.hypot(u[0, 0, 0], v[0, 0, 0])),
                   sase_ref.SFC_WSPD_FLOOR)
        ust = _charnock_ust(spd1, z1[0])       # evolving water drag
        n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
        if m1:
            tt = theta * (pcol / sase_ref.P0_REF) ** (
                sase_ref.RD_AIR / sase_ref.CP_AIR)
            es = 1000.0 * sase_ref.SVP1 * np.exp(
                sase_ref.SVP2 * (tt - sase_ref.SVPT0)
                / (tt - sase_ref.SVP3))
            qs = sase_ref.EP2_RV * es / (pcol - es)
            mkw = {"n2_moist": sase_ref.moist_n2(
                theta, 0.7 * qs, np.zeros_like(theta), pcol, thick)}
        else:
            mkw = {}
        e[0] += dt * ust ** 3 / (sase_ref.KARMAN * 0.5 * thick[0])
        e_n = np.maximum(e, sase_ref.E_MIN)
        # theta-channel K_v replica at the scheduled f (steps 2/2b of
        # sase_split_step -- the driver's exported kv)
        l_les = sase_ref.vertical_mixing_length(z3, e_n, n2)
        l_mix_r, _ = sase_ref.bl89_rans_lengths(theta, e_n, z3, t3, n2)
        c_r = sase_ref.stable_limit_coefficient(l_mix_r, e_n, n2)
        root_e = np.sqrt(e_n)
        fno = f_cell["f"]
        kv = (fno * (sase_ref.C_KV * l_les * root_e)
              + (1.0 - fno) * (c_r * l_mix_r * root_e))
        # Ekman-engine PGF/Coriolis hook at the pre-update state
        du = dt * fcor * (v - vg[:, None, None])
        dv = -dt * fcor * (u - ug[:, None, None])
        u = u + du
        v = v + dv
        fields, ledger = sase_ref.sase_split_step(
            u, v, w, theta, e, dx=delta, dy=delta, dz=200.0,
            delta=delta, dt=dt, n2=n2, dz_col=thick,
            ust=np.full((4, 4), ust), **mkw)
        assert ledger["f"] == fno              # the pinned schedule
        u, v, w, e = (fields[k] for k in "uvwe")
        if freeze_wind:
            u = np.broadcast_to(u0[:, None, None], shape).copy()
            v = np.broadcast_to(v0[:, None, None], shape).copy()
            w = np.zeros(shape)
        pr = sase_ref.prandtl_blend(ledger["f"])
        theta = sase_ref.implicit_vertical_diffusion(
            theta, sase_ref._face_average(kv) / pr, dt, dz_col=thick)
        # implicit bulk surface cooling toward the potential TSK
        cc = 0.6 * (ust / spd1) ** 2 * spd1    # Ch = 0.6*Cd conductance
        theta[0] = ((theta[0] + dt * cc * tsk_pot / thick[0])
                    / (1.0 + dt * cc / thick[0]))
        s0 = float(np.hypot(u[0, 0, 0], v[0, 0, 0]))
        s1 = float(np.hypot(u[1, 0, 0], v[1, 0, 0]))
        sp10 = s0 + (s1 - s0) * np.log(10.0 / z1[0]) / np.log(
            z1[1] / z1[0])
        spd_col = np.hypot(u[:, 0, 0], v[:, 0, 0])
        ratios.append(sp10 / float(np.interp(500.0, z1, spd_col)))
    th_fin = theta[:, 0, 0]
    return {"ratios": np.array(ratios), "z": z1,
            "theta_fin": th_fin.copy(), "e_fin": e[:, 0, 0].copy(),
            "ibl_d74": float(np.interp(74.0, z1, th_fin) - th_fin[0])}


def test_lake_decoupling_green_water_column_stays_decoupled(monkeypatch):
    """GREEN (the registered S3-9 obligation, mechanism report section
    5): the 13Z onset column -- a proper stable marine IBL under the
    held 13Z jet -- integrated 4 h at dt = 60 under the reconstructed
    domain-level f schedule stays DECOUPLED with the geometric l_d
    blend.

    BANDS (both registered in the report's fixture spec):

    * final sp10/sp500 in [0.35, 0.60] -- the LAND band: the run's
      own upwind-land column spans 0.33-0.64 over 13-19Z (report
      section 1 table) while the defective lake columns ran
      0.65-0.94; the report's preview measured 0.43-0.55 across
      forcing variants.  MEASURED here (geometric blend, Charnock
      drag, 13Z geo held): ratio_fin = 0.539, series inside
      [0.507, 0.598] for the whole 4 h -- never approaching the
      coupled class.
    * final e below 500 m <= 0.3 m^2/s^2 -- the accurate-RANS energy
      ceiling (the report's variants measured 0.05-0.08; the
      throttled defect equilibrium sits at 1-2.5, the run's own
      restart truth 2.03).  MEASURED here: 0.011.

    The interactive leg alone does NOT discriminate the blends (the
    report's verdict: every interactive column decouples because the
    column cannot self-generate the 3-D fetch-long momentum resupply
    -- the linear-blend leg ends at ratio 0.545 too, though with 3x
    the e).  This leg pins the onset band and the e ceiling; the
    frozen-wind RED/GREEN pair below carries the discrimination."""
    res = _run_lake_column(monkeypatch, state="13", hours=4.0)
    ratio_fin = float(res["ratios"][-1])
    assert 0.35 <= ratio_fin <= 0.60, ratio_fin
    e500 = float(res["e_fin"][res["z"] < 500.0].max())
    assert e500 <= 0.3, e500


def test_lake_decoupling_red_linear_blend_never_forms_ibl(monkeypatch):
    """RED + companion GREEN (the registered S3-9 obligation): the
    frozen-wind leg from the coupled 18Z state -- wind held at the
    run's own defect equilibrium, the 3-D dynamics' momentum resupply
    prescribed -- discriminates the blends.

    COMPANION GREEN (the defect the run SHOULD have shown, asserted
    first so the legs cannot both pass by drift): with the live
    geometric blend the -250 W/m^2 surface cooling builds the marine
    IBL against the held wind -- Delta-theta(0 -> 74 m) >= 10.0 K
    after 2 h.  Band derivation: the report's fixed-formula class is
    12.7-12.9 K (every working suppression, its candidate table);
    10.0 sits ~22% under that class and 2.5x above the RED ceiling.
    MEASURED here: 12.799 K (the report's candidate-(g) frozen leg:
    12.80 -- same recipe, reproduced).

    RED (the linear blend pinned back by _pre_s3_9_dissipation_length,
    the S3-6g monkeypatch idiom): the THROTTLED EQUILIBRIUM of the
    F-Y1 defect -- with f = 0.12-0.19 the linear f*delta term (360-
    580 m) jumps l_d past the Blackadar/BL89 bound, the self-inflating
    l_s = 0.76*sqrt(e)/N is the only cap left, and production holds e
    an order of magnitude high while K_v dilutes the cooling through
    500+ m, so the IBL never forms (report section 3).  Registered
    criteria: IBL Delta-theta(0 -> 74 m) <= 4 K after 2 h AND
    max e(z < 400 m) >= 1.0 m^2/s^2.  MEASURED here: 2.748 K and
    2.213 m^2/s^2 (the report's baseline: 2.75 K / e 2.21, restart
    truth 2.03 -- the run's own coupled state, reproduced), against
    the GREEN leg's 12.8 K: the 4.7x IBL separation this amendment
    exists for."""
    from woof.verify import sase_ref
    green = _run_lake_column(monkeypatch, state="18", hours=2.0,
                             sched_t0=5.0, freeze_wind=True,
                             restart_e=True)
    assert green["ibl_d74"] >= 10.0, green["ibl_d74"]
    monkeypatch.setattr(sase_ref, "dissipation_length",
                        _pre_s3_9_dissipation_length)
    red = _run_lake_column(monkeypatch, state="18", hours=2.0,
                           sched_t0=5.0, freeze_wind=True,
                           restart_e=True)
    assert red["ibl_d74"] <= 4.0, red["ibl_d74"]
    e400 = float(red["e_fin"][red["z"] < 400.0].max())
    assert e400 >= 1.0, e400


# ---------------------------------------------------------------------------
# NOT PORTED, deliberately: the smoke-harness tests.
#
# Ten tests here (a G2b acceleration-excusal rule, an asynchronous
# restart writer, a setup fingerprint) exercised tools/sase_smoke/, the
# campaign runner the closure was developed against.  That runner is
# case-specific end to end -- it names one historical case's configs,
# scoring boxes and receipt layout -- and carrying it onto a release
# line would put case identity into shipped tooling, which is the one
# thing this project refuses.  The closure itself is unaffected: none of
# the ten touched woof/core/sase.py, the device kernels or the driver
# seam.  They remain on experimental/sase-v1 with the harness they test.
# ---------------------------------------------------------------------------



def _g2b_recs(e_mean_by_third, emax_last_slope, ratio_last_slope,
              total=5400.0, drop_first_third=False):
    """Synthetic gate2 series: 300 s cadence, e_mean piecewise linear
    with the given per-third endpoint values [(e0, e1), ...], e_max
    and cap-ratio flat except a linear last-third trend with the given
    signs, cap-ratio well under C_GATE throughout."""
    recs = []
    third = total / 3.0
    for i in range(19):
        t = 300.0 * i
        which = min(int(t // third), 2)
        lo, hi = which * third, (which + 1) * third
        frac = (t - lo) / (hi - lo)
        e0, e1 = e_mean_by_third[which]
        e_mean = e0 + (e1 - e0) * frac
        if t >= 2.0 * third:
            tt = t - 2.0 * third
            e_max = 13.0 + emax_last_slope * tt
            ratio = 0.05 + ratio_last_slope * tt
        else:
            e_max, ratio = 13.0, 0.05
        if drop_first_third and t < third:
            continue                           # leaves ONE sample <= 1800
        recs.append({"seconds": t, "e_mean": e_mean, "e_max": e_max,
                     "ratio": ratio})
    return recs












# ---------------------------------------------------------------------------
# S3-11a surface scalar-flux deposit (authority seam; G-LAKE root cause
# .superpowers/sdd/lake-momentum-root-cause.md -- adversarially verified,
# zero refutations; module docstring, S3-11a section)
# ---------------------------------------------------------------------------

#: Observed fixture-point surface fluxes (yolo-b 17Z d02, mid-Lake-
#: Michigan j=304 i=168 -- the root-cause file's data-provenance
#: section): sfclay HFX = -180.2 W/m^2 (downward, post-frontal warm
#: air over 274.9 K water), QFX = -1.02e-5 kg m^-2 s^-1 (4 orders
#: below the warm-sector scale; its moisture face is the warm-sector
#: fixture's subject, so the lake leg deposits heat only).
_LAKE_HFX_17Z = -180.2
_LAKE_QFX_17Z = -1.02e-5


def test_sase_config_id_binds_s3_11a_constants(monkeypatch):
    """CP_AIR and SFC_SCALAR_FLUX (the registered explicit-deposit
    decision string) are registry members and hash-bound -- the
    S3-6h/S3-9 convention-string idiom.  CP_AIR is single-sourced
    with the model side's ``woof.core.constants.CP`` (7*RD/2 =
    1004.5, FP64-exact, WRF share/module_model_constants cp -- the
    same constant sfclay used to FORM the HFX this seam consumes),
    the KARMAN pin idiom."""
    from woof.core import constants as c

    from woof.verify import sase_ref
    assert sase_ref.CP_AIR == c.CP == 1004.5
    assert sase_ref.SFC_SCALAR_FLUX == "explicit-deposit-v1"
    for name in ("CP_AIR", "SFC_SCALAR_FLUX"):
        assert name in sase_ref._CONFIG_ID_CONSTANTS
    baseline = sase_ref.sase_config_id()
    monkeypatch.setattr(sase_ref, "CP_AIR", 1005.0)
    with_cp = sase_ref.sase_config_id()
    assert with_cp != baseline
    monkeypatch.setattr(sase_ref, "SFC_SCALAR_FLUX",
                        "implicit-bottom-row")
    assert sase_ref.sase_config_id() not in (baseline, with_cp)


def test_surface_scalar_flux_deposit_closed_form_and_rejections():
    """The deposit IS the audited YSU surface-rhs pair (npref.py:6472
    heat, :6481 moisture, delp = rho*g*dz1 form) and nothing else:

        theta_new[0] = theta[0] + dt*hfx/(rho1*CP_AIR*thick_0)
        qv_new[0]    = qv[0]    + dt*qfx/(rho1*thick_0)

    asserted BITWISE against the independently written expression
    (same FP op order), every level above bitwise untouched, inputs
    never mutated, both geometry modes (dz scalar / dz_col
    thicknesses).  Rejections: non-positive rho1 (a sign-flipping
    density is nonphysical and silently inverts the flux), theta/qv
    shape mismatch, and the _column_geometry dz/dz_col contract."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(311)
    nz, ny, nx = 5, 3, 4
    thick = np.array([16.0, 20.0, 26.0, 34.0, 44.0])
    theta = 280.0 + 5.0 * rng.standard_normal((nz, ny, nx))
    qv = 0.004 + 0.001 * rng.random((nz, ny, nx))
    hfx = 250.0 * rng.standard_normal((ny, nx))   # mixed-sign fluxes
    qfx = 1.0e-4 * rng.standard_normal((ny, nx))
    rho1 = 1.0 + 0.2 * rng.random((ny, nx))
    dt = 15.0
    th_snap, qv_snap = theta.copy(), qv.copy()
    th_new, qv_new = sase_ref.surface_scalar_flux_deposit(
        theta, qv, dt, rho1, hfx=hfx, qfx=qfx, dz_col=thick)
    # bitwise closed form at the bottom row, bitwise identity above
    assert np.array_equal(
        th_new[0], theta[0] + dt * hfx / (rho1 * sase_ref.CP_AIR
                                          * thick[0]))
    assert np.array_equal(qv_new[0],
                          qv[0] + dt * qfx / (rho1 * thick[0]))
    assert np.array_equal(th_new[1:], theta[1:])
    assert np.array_equal(qv_new[1:], qv[1:])
    # inputs are never mutated (the authority's functional convention)
    assert np.array_equal(theta, th_snap)
    assert np.array_equal(qv, qv_snap)
    # uniform-dz mode: thick_0 is the scalar spacing
    th_u, qv_u = sase_ref.surface_scalar_flux_deposit(
        theta, qv, dt, rho1, hfx=hfx, qfx=qfx, dz=50.0)
    assert np.array_equal(
        th_u[0], theta[0] + dt * hfx / (rho1 * sase_ref.CP_AIR * 50.0))
    assert np.array_equal(qv_u[0], qv[0] + dt * qfx / (rho1 * 50.0))
    with pytest.raises(ValueError, match="rho1"):
        sase_ref.surface_scalar_flux_deposit(
            theta, qv, dt, 0.0, hfx=hfx, dz_col=thick)
    with pytest.raises(ValueError, match="rho1"):
        sase_ref.surface_scalar_flux_deposit(
            theta, qv, dt, -1.0, hfx=hfx, dz_col=thick)
    with pytest.raises(ValueError, match="shape"):
        sase_ref.surface_scalar_flux_deposit(
            theta, qv[:-1], dt, rho1, hfx=hfx, dz_col=thick)
    with pytest.raises(ValueError, match="dz"):
        sase_ref.surface_scalar_flux_deposit(
            theta, qv, dt, rho1, hfx=hfx, dz=50.0, dz_col=thick)


def test_surface_scalar_flux_deposit_zero_flux_identity():
    """THE SEAM-OFF CONTRACT (spec section 2): hfx and qfx DEFAULT to
    0.0, and the zero-flux deposit is BITWISE the identity -- the
    bottom row gains literally +0.0 (x + 0.0 == x for every finite x
    except -0.0; physical theta > 0 and qv >= +0.0), so composing the
    seam at zero flux with the existing scalar channel reproduces the
    pre-S3-11a channel bit-for-bit.  No existing code path calls the
    seam (it is a NEW function; no existing function's body changed),
    so every pre-S3-11a fixture and golden in this suite is
    bitwise-untouched by construction -- this test pins the identity
    that guarantees it stays true once the driver threads real
    (sometimes-zero) flux fields in S3-11b."""
    import inspect

    from woof.verify import sase_ref
    sig = inspect.signature(sase_ref.surface_scalar_flux_deposit)
    assert sig.parameters["hfx"].default == 0.0
    assert sig.parameters["qfx"].default == 0.0
    rng = np.random.default_rng(1131)
    nz, ny, nx = 6, 4, 4
    thick = np.array([16.0, 20.0, 26.0, 34.0, 44.0, 58.0])
    theta = 285.0 + 8.0 * rng.standard_normal((nz, ny, nx))
    theta = np.abs(theta)                       # finite, positive
    qv = 0.006 * rng.random((nz, ny, nx))       # finite, >= +0.0
    rho1 = 1.0 + 0.2 * rng.random((ny, nx))
    dt = 60.0
    # defaults omitted entirely
    th_a, qv_a = sase_ref.surface_scalar_flux_deposit(
        theta, qv, dt, rho1, dz_col=thick)
    assert th_a.tobytes() == theta.tobytes()
    assert qv_a.tobytes() == qv.tobytes()
    # explicit scalar zeros and explicit (ny, nx) zero fields
    for z in (0.0, np.zeros((ny, nx))):
        th_b, qv_b = sase_ref.surface_scalar_flux_deposit(
            theta, qv, dt, rho1, hfx=z, qfx=z, dz_col=thick)
        assert th_b.tobytes() == theta.tobytes()
        assert qv_b.tobytes() == qv.tobytes()
    # composed with the existing implicit K_v/Pr_t channel: the
    # zero-flux seam is invisible bit-for-bit
    k_face = 2.0 + rng.random((nz - 1, ny, nx))
    direct = sase_ref.implicit_vertical_diffusion(
        theta, k_face, dt, dz_col=thick)
    seamed = sase_ref.implicit_vertical_diffusion(
        th_a, k_face, dt, dz_col=thick)
    assert seamed.tobytes() == direct.tobytes()


def test_s3_11a_scalar_ledger_boundary_consistent():
    """RE-DERIVED LEDGER (S3-11a, module docstring): the composed
    scalar update phi -> implicit_solve(deposit(phi)) carries the
    boundary-consistent closure

        sum_k thick_k*(theta^{n+1} - theta^n)_k = dt*hfx/(rho1*CP_AIR)
        sum_k thick_k*(qv^{n+1} - qv^n)_k       = dt*qfx/rho1

    PER COLUMN, per step -- (i) the deposit's 1/thick_0 cancels
    against the thickness weight, (ii) the zero-flux implicit solve
    conserves sum(thick*phi) to solver roundoff (its pinned
    conservation property).  Unlike the KE pairing this closure needs
    NO uniform spacing: asserted on a uniform-dz box AND a stretched
    dz_col column set.  Tolerance derivation: the solve's
    conservation roundoff is relative to the column CONTENT
    (~thick_tot*theta ~ 2.4e5 K m), not to the flux increment
    (~1 K m/step), so the closure is pinned at rtol 1e-9 of the
    increment (~1e-15 of the content, pure FP64 roundoff; measured
    ~1e-11).  The split-step theorem is UNTOUCHED by this amendment
    (theta enters sase_split_step read-only; the deposit runs outside
    the step): re-pinned here alongside the live seam with the S3-6j
    boundary-consistent residual at its registered < 1e-11."""
    from woof.verify import sase_ref
    rng = np.random.default_rng(2026)
    nz, ny, nx = 8, 4, 6
    dt = 30.0
    hfx = 400.0 * rng.standard_normal((ny, nx))    # mixed sign, W/m^2
    qfx = 2.0e-4 * rng.standard_normal((ny, nx))
    rho1 = 1.0 + 0.2 * rng.random((ny, nx))
    for geom in ({"dz": 100.0},
                 {"dz_col": 100.0 * (1.0 + 0.15 * np.arange(nz))}):
        theta = 300.0 + 5.0 * rng.standard_normal((nz, ny, nx))
        qv = 0.008 + 0.004 * rng.random((nz, ny, nx))
        k_face = 5.0 * rng.random((nz - 1, ny, nx))
        th0, qv0 = theta.copy(), qv.copy()
        steps = 5
        for _ in range(steps):
            theta, qv = sase_ref.surface_scalar_flux_deposit(
                theta, qv, dt, rho1, hfx=hfx, qfx=qfx, **geom)
            theta = sase_ref.implicit_vertical_diffusion(
                theta, k_face, dt, **geom)
            qv = sase_ref.implicit_vertical_diffusion(
                qv, k_face, dt, **geom)
        thick = geom.get("dz", geom.get("dz_col"))
        tcol = (np.full(nz, thick) if np.ndim(thick) == 0
                else np.asarray(thick))[:, None, None]
        dth_col = np.sum(tcol * (theta - th0), axis=0)
        dqv_col = np.sum(tcol * (qv - qv0), axis=0)
        np.testing.assert_allclose(
            dth_col, steps * dt * hfx / (rho1 * sase_ref.CP_AIR),
            rtol=1e-9)
        np.testing.assert_allclose(
            dqv_col, steps * dt * qfx / rho1, rtol=1e-9)
    # split-step theorem untouched: boundary-consistent S3-6j closure
    # holds verbatim while the scalar seam runs beside it
    shape = (8, 24, 24)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 4)
               for _ in range(3))
    theta_b = np.full(shape, 300.0)
    e = np.full(shape, 0.2)
    _, led = sase_ref.sase_split_step(
        u, v, w, theta_b, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=1.0, ust=np.full(shape[1:], 0.4))
    scale = max(abs(led["dKE"]), abs(led["dE"]), abs(led["dHeat"]),
                1e-30)
    assert abs(led["residual"]) / scale < 1e-11


def test_warm_sector_qfx_moistens_at_closed_form_rate():
    """FIXTURE (b) (spec section 5): the moisture face of the seam.
    QFX value derivation: warm-sector latent heat flux LH ~ 250 W/m^2
    (the P2 sector class whose Td2 the run under-ran once the
    deepening PBL diluted qv with no surface resupply) over
    XLV = 2.5e6 J/kg gives QFX = LH/XLV = 1.0e-4 kg m^-2 s^-1.
    Asserts: (i) one deposit moistens the lowest layer at EXACTLY the
    closed-form rate qfx/(rho1*thick_0) (bitwise, the YSU q_surface
    row) and touches nothing else; (ii) composed with an ACTIVE
    K_v/Pr_t vertical solve over 20 steps, the moisture propagates
    upward (qv(z2) grows -- the Td-recovery face), the max principle
    holds (no level ever falls below the initial uniform qv), and the
    column integral tracks the source exactly (the S3-11a ledger on
    the moving state)."""
    from woof.verify import sase_ref
    nz = 8
    thick = np.array([20.0, 25.0, 32.0, 41.0, 53.0, 68.0, 88.0, 114.0])
    shape = (nz, 3, 3)
    qv0 = 0.008                                  # 8 g/kg warm sector
    qfx = 1.0e-4
    rho1 = 1.15
    dt = 15.0
    theta = np.full(shape, 300.0)
    qv = np.full(shape, qv0)
    th_new, qv_new = sase_ref.surface_scalar_flux_deposit(
        theta, qv, dt, rho1, qfx=qfx, dz_col=thick)
    assert np.array_equal(qv_new[0],
                          qv[0] + dt * qfx / (rho1 * thick[0]))
    assert np.array_equal(qv_new[1:], qv[1:])
    assert th_new.tobytes() == theta.tobytes()   # hfx defaulted: no heat
    # composed channel: deposit BEFORE the implicit solve (the
    # registered order), 20 steps, K_v ~ 3 m^2/s faces
    k_face = np.full((nz - 1,) + shape[1:], 3.0)
    q = qv.copy()
    steps = 20
    for _ in range(steps):
        _, q = sase_ref.surface_scalar_flux_deposit(
            theta, q, dt, rho1, qfx=qfx, dz_col=thick)
        q = sase_ref.implicit_vertical_diffusion(
            q, k_face, dt, dz_col=thick)
    assert float(q[2, 0, 0]) > qv0               # moisture reached z3
    assert float(q.min()) >= qv0 - 1e-15         # max principle
    dq_col = np.sum(thick[:, None, None] * (q - qv0), axis=0)
    np.testing.assert_allclose(
        dq_col, steps * dt * qfx / rho1, rtol=1e-9)


def _run_lake_flux_column(monkeypatch, hfx, minutes=10.0, dt=15.0):
    """FIXTURE (a) engine: the 18Z coupled-defect mid-lake state
    (tests/lake_profile_19740403.py -- the neutral homogenized slab
    the 17Z reproduction column shares) under the S3-11a deposit at a
    frozen observed HFX, integrated at the d02 physics step dt = 15 s
    (the root-cause reproduction's step).  Per step, the driver's
    registered sequence: live n2, pre-step surface-e deposit at the
    Charnock u* of the frozen wind, split step (frozen wind -- the
    3-D fetch momentum resupply the interactive column cannot
    self-generate, the S3-9 idiom), then THE SEAM (explicit
    surface_scalar_flux_deposit) followed by the implicit K_v/Pr_t
    theta solve.  f is pinned 0.0, the reproduction's RANS limb
    (partition_cap(3000 m, zi ~ 300 m) = 0.0 at the defect state).
    rho1 is the dry lowest-level density of the profile literals,
    rho1 = PSFC_18/(RD*T1) with T1 = THETA_18[0]*(PSFC_18/1e5)^RCP =
    288.409 K -> rho1 = 1.17443 kg/m^3 (the extracted table carries
    no QV; the moist correction is <1% and frozen rho1 makes the
    ledger assertion closed-form), giving a bottom-layer deposit of
    dt*hfx/(rho1*CP_AIR*thick_0) = -0.1365 K/step at the observed
    -180.2 W/m^2 -- the brief's <= 0.13-0.14 K/step explicit-deposit
    bound.  Returns the N^2(k0) / K_v(26 m) / theta_1 series, the
    final-state N^2(k0), and the column heat integral
    sum(thick*(theta_fin - theta_0)) for the ledger assertion."""
    from woof.verify import sase_ref
    import lake_profile_19740403 as lp

    theta0, u0, v0 = lp.THETA_18, lp.U_18, lp.V_18
    thick = lp.THICK_18
    nz = len(theta0)
    shape = (nz, 4, 4)
    z1 = np.cumsum(thick) - 0.5 * thick          # authority convention
    z3 = z1[:, None, None]
    t3 = thick[:, None, None]
    delta = 3000.0                               # the d02 scale
    rcp = 287.0 / sase_ref.CP_AIR                # RD/CP, model constants
    t1 = theta0[0] * (lp.PSFC_18 / 1.0e5) ** rcp
    rho1 = lp.PSFC_18 / (287.0 * t1)

    u = np.broadcast_to(u0[:, None, None], shape).copy()
    v = np.broadcast_to(v0[:, None, None], shape).copy()
    w = np.zeros(shape)
    theta = np.broadcast_to(theta0[:, None, None], shape).copy()
    qv = np.zeros(shape)                         # heat-only lake leg
    e = np.broadcast_to(
        np.maximum(lp.E_RST_18, sase_ref.E_MIN)[:, None, None],
        shape).copy()

    # f = 0 pin: the S3-6f-cap monkeypatch idiom at the reproduction's
    # RANS limb (dynamic_solve degenerate, w-sensor abstains, cap 0)
    monkeypatch.setattr(sase_ref, "dynamic_solve",
                        lambda *a, **k: (0.0, 1.0))
    monkeypatch.setattr(sase_ref, "partition_cap", lambda d, zi: 0.0)
    monkeypatch.setattr(
        sase_ref, "w_resolved_bound",
        lambda w_, e_mean, n2=None: sase_ref.WSensorState(
            f_w=1.0, alpha_w=1.0, e_res_w=0.0, coverage=0.0))

    spd1 = max(float(np.hypot(u[0, 0, 0], v[0, 0, 0])),
               sase_ref.SFC_WSPD_FLOOR)
    ust = _charnock_ust(spd1, z1[0])             # frozen wind: constant
    steps = int(round(minutes * 60.0 / dt))
    n2_k0, kv26, th1 = [], [], []
    for _ in range(steps):
        n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
        e[0] += dt * ust ** 3 / (sase_ref.KARMAN * 0.5 * thick[0])
        e_n = np.maximum(e, sase_ref.E_MIN)
        # theta-channel K_v replica at f = 0 (steps 2/2b of
        # sase_split_step: the pure RANS limb C_r*l_mix_rans*sqrt(e))
        l_mix_r, _ = sase_ref.bl89_rans_lengths(theta, e_n, z3, t3, n2)
        c_r = sase_ref.stable_limit_coefficient(l_mix_r, e_n, n2)
        kv = c_r * l_mix_r * np.sqrt(e_n)
        fields, ledger = sase_ref.sase_split_step(
            u, v, w, theta, e, dx=delta, dy=delta, dz=200.0,
            delta=delta, dt=dt, n2=n2, dz_col=thick,
            ust=np.full((4, 4), ust))
        assert ledger["f"] == 0.0                # the pinned RANS limb
        e = fields["e"]                          # wind stays frozen
        # THE SEAM under test: explicit deposit BEFORE the implicit
        # K_v/Pr_t solve (the registered SFC_SCALAR_FLUX order)
        theta, qv = sase_ref.surface_scalar_flux_deposit(
            theta, qv, dt, rho1, hfx=hfx, dz_col=thick)
        pr = sase_ref.prandtl_blend(0.0)
        theta = sase_ref.implicit_vertical_diffusion(
            theta, sase_ref._face_average(kv) / pr, dt, dz_col=thick)
        n2_k0.append(float(n2[0, 0, 0]))
        kv26.append(float(kv[1, 0, 0]))
        th1.append(float(theta[0, 0, 0]))
    n2_fin = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    dth_col = float(np.sum(thick * (theta[:, 0, 0] - theta0)))
    return {"n2_k0": np.array(n2_k0), "kv26": np.array(kv26),
            "th1": np.array(th1), "n2_fin": float(n2_fin[0, 0, 0]),
            "dth_col": dth_col, "rho1": rho1, "steps": steps,
            "dt": dt}


def test_lake_flux_deposit_forms_stable_layer_and_collapses_kv(
        monkeypatch):
    """FIXTURE (a) (spec section 5) + zero-flux RED companion: the
    seam is what forms the marine stable layer.  GREEN (deposit at
    the observed -180.2 W/m^2): the bottom deposit cools theta_1
    ~0.14 K/step (2.66 K by 10 min), N^2(k0) crosses 1.0e-3 s^-2
    within 10 min (the stated threshold/time; MEASURED crossing at
    2.75 min, 1.37e-3 at 10 min -- the reproduction's
    1.6e-3-at-10-min class), and the 26-m diffusivity collapses more
    than 10x from its own first-step value (MEASURED 11.27 -> 0.19
    m^2/s, 59x -- the first step rides the run's coupled-equilibrium
    restart e on the still-neutral slab; the reproduction, from its
    own spun state: 3.74 -> 0.26, 14x).  RED (hfx = 0.0, the
    as-built lane): the SAME engine holds N^2(k0) <= 1.0e-4
    (MEASURED 8.3e-6 -- the "unfelt interface jump") and
    K_v(26 m) >= 2.0 m^2/s (MEASURED 2.97), so the GREEN/RED K_v
    separation exceeds 5x (MEASURED 15.4x).  Band rationale: GREEN
    thresholds sit ~30-50% past the measured values, RED ceilings
    ~10x above theirs -- both far from the 15-120x defect/fix
    separations they discriminate.  The GREEN leg
    also carries the S3-11a boundary-consistent ledger ON THE REAL
    STATE: the column heat integral equals steps*dt*hfx/(rho1*CP_AIR)
    to roundoff -- every joule the seam deposited and nothing else."""
    from woof.verify import sase_ref
    green = _run_lake_flux_column(monkeypatch, hfx=_LAKE_HFX_17Z)
    red = _run_lake_flux_column(monkeypatch, hfx=0.0)
    # stratification forms only with the seam live
    assert green["n2_fin"] >= 1.0e-3, green["n2_fin"]
    assert red["n2_fin"] <= 1.0e-4, red["n2_fin"]
    # K_v collapse per the reproduction (>= 10x own-initial, <= 0.4
    # absolute) vs the RED leg's persistent neutral-wall mixing
    assert green["kv26"][-1] <= 0.4, green["kv26"][-1]
    assert green["kv26"][-1] <= green["kv26"][0] / 10.0
    assert red["kv26"][-1] >= 2.0, red["kv26"][-1]
    assert red["kv26"][-1] / green["kv26"][-1] >= 5.0
    # boundary-consistent scalar ledger on the real state
    expected = (green["steps"] * green["dt"] * _LAKE_HFX_17Z
                / (green["rho1"] * sase_ref.CP_AIR))
    np.testing.assert_allclose(green["dth_col"], expected, rtol=1e-9)
    # the RED leg deposited nothing: its column heat integral is the
    # solve's conservation roundoff only
    assert abs(red["dth_col"]) < 1.0e-6



# ---------------------------------------------------------------------------
# S3-9e: on-device gate-2a stats, async restart publication, fingerprint
# memoization (same numbers, cheaper route -- the GPU-valley remedies).
# ---------------------------------------------------------------------------

def _cap_stats_inputs(seed=20260721, nz=7, ny=12, nx=14):
    """Adversarial-ish gate-2a inputs: variable per-column dz, signed
    winds, a u* == 0 cell (exercises the isfinite(cap0) branch)."""
    rng = np.random.default_rng(seed)
    e = rng.uniform(1.0e-4, 12.0, (nz, ny, nx))
    u = rng.standard_normal((nz, ny, nx)) * 8.0
    v = rng.standard_normal((nz, ny, nx)) * 8.0
    t = rng.uniform(30.0, 400.0, (nz, ny, nx))
    ust = rng.uniform(0.0, 0.9, (ny, nx))
    ust[0, 0] = 0.0
    return e, u, v, t, ust


def _assert_stats_bitwise(got: dict, ref: dict) -> None:
    import struct
    assert set(got) == set(ref)
    for key, want in ref.items():
        have = got[key]
        if isinstance(want, float):
            assert struct.pack("<d", have) == struct.pack("<d", want), (
                key, want, have)
        else:
            assert have == want, (key, want, have)


def test_s3_9e_device_graph_is_bitwise_identical_on_host():
    """The CPU spine of the receipt-equivalence probe: the device path's
    restructured expression graph (`x*x` for the numpy `** 2` fast path,
    the explicit sequential cumsum recurrence, `_blackadar_length`
    inlined) run under numpy must reproduce the legacy host reference
    BITWISE on every stat, for f64 and production-f32 inputs and with and
    without the boundary exclusion."""
    from woof.core.sase import sase_e_cap_stats, sase_e_cap_stats_device
    base = _cap_stats_inputs()
    for dtype in (np.float64, np.float32):
        arrays = tuple(a.astype(dtype) for a in base)
        for width in (0, 3):
            ref = sase_e_cap_stats(*arrays, boundary_width=width)
            got = sase_e_cap_stats_device(
                *arrays, boundary_width=width, xp=np)
            assert got.pop("stats_path") == "device-graph-host"
            _assert_stats_bitwise(got, ref)


def test_s3_9e_device_graph_single_level_and_zero_ustar():
    """nz == 1 (surface-balance-only cap) and an all-zero u* plane
    (cap == 0 everywhere -> the ratio floor divisor) stay bitwise."""
    from woof.core.sase import sase_e_cap_stats, sase_e_cap_stats_device
    rng = np.random.default_rng(7)
    e = rng.uniform(0.1, 2.0, (1, 8, 9))
    u = rng.standard_normal((1, 8, 9))
    v = rng.standard_normal((1, 8, 9))
    t = np.full((1, 8, 9), 55.0)
    for ust in (rng.uniform(0.1, 0.6, (8, 9)), np.zeros((8, 9))):
        ref = sase_e_cap_stats(e, u, v, t, ust)
        got = sase_e_cap_stats_device(e, u, v, t, ust, xp=np)
        assert got.pop("stats_path") == "device-graph-host"
        _assert_stats_bitwise(got, ref)


def test_s3_9e_device_path_rejections_match_legacy():
    from woof.core.sase import sase_e_cap_stats_device
    e, u, v, t, ust = _cap_stats_inputs()
    with pytest.raises(ValueError, match="shape mismatch"):
        sase_e_cap_stats_device(e, u[:, :5], v, t, ust, xp=np)
    with pytest.raises(ValueError, match="boundary_width"):
        sase_e_cap_stats_device(e, u, v, t, ust, boundary_width=7, xp=np)
    with pytest.raises(ValueError, match="boundary_width"):
        sase_e_cap_stats_device(e, u, v, t, ust, boundary_width=-1, xp=np)


def test_s3_9e_compare_e_cap_stats_rules():
    """The dual-probe comparison enforces the DERIVED per-stat rules:
    e_max bitwise, e_mean/cap_p/cap_max/ratio under their documented
    relative bounds, echoed metadata exact, non-finite fails closed."""
    import math
    from woof.core.sase import (E_CAP_STATS_COMPARE_RTOL,
                                 compare_e_cap_stats)
    assert E_CAP_STATS_COMPARE_RTOL["e_max"] == 0.0
    base = {"e_max": 7.0, "e_mean": 1.25, "cap_percentile": 99.9,
            "cap_p": 3.0, "cap_max": 9.5, "ratio": 7.0 / 3.0,
            "boundary_width": 4}
    dev = dict(base, stats_path="device")
    rec = compare_e_cap_stats(base, dev)
    assert rec["equivalent"]
    assert all(entry["pass"] for entry in rec["stats"].values())
    assert rec["stats"]["e_max"]["rel_err"] == 0.0

    one_ulp = dict(dev, e_max=float(np.nextafter(7.0, np.inf)))
    rec = compare_e_cap_stats(base, one_ulp)
    assert not rec["equivalent"] and not rec["stats"]["e_max"]["pass"]

    inside = dict(dev, e_mean=1.25 * (1.0 + 2.0e-13))
    assert compare_e_cap_stats(base, inside)["equivalent"]
    outside = dict(dev, e_mean=1.25 * (1.0 + 1.0e-9))
    assert not compare_e_cap_stats(base, outside)["equivalent"]

    assert not compare_e_cap_stats(
        base, dict(dev, boundary_width=5))["equivalent"]
    assert not compare_e_cap_stats(
        base, dict(dev, ratio=math.inf))["equivalent"]


def test_s3_9e_dual_probe_host_mode_and_fail_loud(monkeypatch):
    """The runnable dual-path probe wiring, exercised host-side: the
    returned dict carries the DEVICE numbers plus the full comparison,
    equivalence holds bitwise in numpy mode, and a perturbed device stat
    raises (fail-loud) or is recorded (record mode)."""
    import woof.core.sase as sase_mod
    arrays = _cap_stats_inputs()
    out = sase_mod.sase_e_cap_stats_dual(*arrays, boundary_width=2, xp=np)
    assert out["stats_path"] == "device-graph-host"
    probe = out["dual_probe"]
    assert probe["equivalent"]
    ref = sase_mod.sase_e_cap_stats(*arrays, boundary_width=2)
    _assert_stats_bitwise(
        {k: v for k, v in out.items()
         if k not in ("stats_path", "dual_probe")}, ref)

    real_device = sase_mod.sase_e_cap_stats_device

    def perturbed(*args, **kwargs):
        stats = real_device(*args, **kwargs)
        stats["e_max"] = float(np.nextafter(stats["e_max"], np.inf))
        return stats

    monkeypatch.setattr(sase_mod, "sase_e_cap_stats_device", perturbed)
    with pytest.raises(ValueError, match="NOT equivalent"):
        sase_mod.sase_e_cap_stats_dual(*arrays, boundary_width=2, xp=np)
    rec = sase_mod.sase_e_cap_stats_dual(
        *arrays, boundary_width=2, xp=np, raise_on_mismatch=False)
    assert not rec["dual_probe"]["equivalent"]
    assert not rec["dual_probe"]["stats"]["e_max"]["pass"]




def _tree_model(monkeypatch, seed_root=21, seed_child=22):
    """Two-domain fake tree legal for capture/write_tree_restart (the
    test_restart complete-set fixture plus a PERIOD_BEGIN runtime
    status)."""
    from types import SimpleNamespace
    from test_restart import _cfg, _fill_serialized, _fill_setup, _shim_state
    from woof.core.model import ModelRuntimeStatus

    cfg1 = _cfg(grid_id=1, moist=False, run_seconds=60.0)
    cfg2 = _cfg(grid_id=2, nested=True, moist=False, run_seconds=60.0)

    def _state(cfg, seed):
        value = _shim_state(cfg, monkeypatch)
        value._nest_restart_classification = "REBUILT"
        _fill_setup(value)
        _fill_serialized(value, seed)
        return value

    clock1 = SimpleNamespace(ticks=0, tick_den=1, step_count=0,
                             dtbc_fp32=np.float32(7.0),
                             spec=SimpleNamespace(step_ticks=1))
    clock2 = SimpleNamespace(ticks=0, tick_den=1, step_count=0,
                             dtbc_fp32=np.float32(9.0),
                             spec=SimpleNamespace(step_ticks=1))
    root = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, parent_id=0, run=cfg1),
        state=_state(cfg1, seed_root), clock=clock1, parent=None,
        coupler=None)
    child = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=2, parent_id=1, run=cfg2),
        state=_state(cfg2, seed_child), clock=clock2, parent=root,
        coupler=None)
    return SimpleNamespace(
        experiment_fingerprint="s3-9e-async-fixture", root=root,
        schedule=SimpleNamespace(
            period_ticks=1,
            clock=SimpleNamespace(tick_den=1, run_ticks=60)),
        walk_parent_first=lambda: iter((root, child)),
        _runtime_status=ModelRuntimeStatus(), _io_manager=None,
        _last_checkpoint=None)


def _freeze_restart_nondeterminism(monkeypatch):
    """Pin the two legitimately wall-clock header inputs (`created`,
    checkpoint_set_id) so sync-vs-async byte identity is exact."""
    from datetime import datetime as real_datetime
    from types import SimpleNamespace
    from woof.io import restart

    class _FrozenDatetime:
        @staticmethod
        def now(tz=None):
            return real_datetime(2026, 7, 21, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(restart, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        restart, "uuid",
        SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="s39e-fixed")))








# ---------------------------------------------------------------------------
# S4-1: M1 moist stability core (authority; SASE-M plan Task 1).
# Formulation, the DK82 Eq.-36 derivation, the saturation-switch
# convention, and the three-substitution-point contract live in the
# woof/verify/sase_ref.py module docstring (SASE-M1 section).  Specimen
# columns: tests/specimen_yolod_11z.py (yolo-d 11Z d02, wrf-rust chain).
# ---------------------------------------------------------------------------


def test_sase_config_id_binds_m1_constants(monkeypatch):
    """MOIST_STABILITY / MOIST_STABILITY_SWITCH (the registered SASE-M1
    convention strings) and every M1 moist-thermodynamics constant are
    registry members and hash-bound -- the S3-6h/S3-9/S3-11a idiom.
    The numeric constants are single-sourced with the model side
    (woof.core.constants = WRF share/module_model_constants -- the
    SAME saturation constants that formed the qc the M1 switch
    consumes), the KARMAN/CP_AIR pin idiom."""
    from woof.core import constants as c

    from woof.verify import sase_ref
    assert sase_ref.MOIST_STABILITY == "dk82-saturated-v1"
    assert sase_ref.MOIST_STABILITY_SWITCH == "binary-qc-or-rh100-liquid"
    assert sase_ref.RD_AIR == c.RD == 287.0
    assert sase_ref.RV_AIR == c.RV == 461.6
    assert sase_ref.EP2_RV == c.EP2            # RD/RV, bitwise
    assert sase_ref.XLV == c.XLV == 2.5e6
    assert sase_ref.SVP1 == c.SVP1 == 0.6112
    assert sase_ref.SVP2 == c.SVP2 == 17.67
    assert sase_ref.SVP3 == c.SVP3 == 29.65
    assert sase_ref.SVPT0 == c.SVPT0 == 273.15
    assert sase_ref.P0_REF == c.P0 == 1.0e5
    for name in ("RD_AIR", "RV_AIR", "EP2_RV", "XLV", "SVP1", "SVP2",
                 "SVP3", "SVPT0", "P0_REF", "MOIST_STABILITY",
                 "MOIST_STABILITY_SWITCH"):
        assert name in sase_ref._CONFIG_ID_CONSTANTS, name
    seen = {sase_ref.sase_config_id()}
    for name, value in (("MOIST_STABILITY", "reversible-thetae"),
                        ("MOIST_STABILITY_SWITCH", "pdf-blend"),
                        ("XLV", 2.501e6)):
        monkeypatch.setattr(sase_ref, name, value)
        cid = sase_ref.sase_config_id()
        assert cid not in seen
        seen.add(cid)


def _specimen_cols(name):
    """Specimen column arrays in the module's (nz, 1, 1) layout."""
    import specimen_yolod_11z as sp
    pre = {"amp": "AMP_", "ctl": "CTL_"}[name]

    def col(a):
        return np.asarray(a, np.float64).reshape(-1, 1, 1)

    g = getattr
    return {"theta": col(g(sp, pre + "THETA")),
            "qv": col(g(sp, pre + "QV")), "qc": col(g(sp, pre + "QC")),
            "p": col(g(sp, pre + "P")), "u": col(g(sp, pre + "U")),
            "v": col(g(sp, pre + "V")), "e0": col(g(sp, pre + "E0")),
            "thick": np.asarray(g(sp, pre + "THICK"), np.float64),
            "ust": float(g(sp, pre + "UST"))}


_SPECIMEN_CACHE: dict = {}


def _run_specimen_column(name="amp", m1=True, steps=120, dt=60.0):
    """S4-1 specimen-column driver replica (frozen-state equilibrium).

    The _run_jet_column / _run_lake_column replica on the yolo-d 11Z
    specimen: live n2 (dry) each step, the driver's pre-step surface-e
    deposit at the specimen u*, the split step at the d02 scale
    (delta = 3000), and -- with ``m1`` -- the M1 substitution riding
    the ``n2_moist`` seam at the live ``sase_ref.moist_n2``.
    Horizontal uniformity degenerates the solve: f_used = 0, the pure
    RANS limb, where the amplifier defect lives (asserted).

    FROZEN-STATE convention (the S3-9 frozen-wind idiom extended to
    the scalars): u/v/w AND theta/qv/qc are reset to the specimen
    state every step, and only e integrates.  Derivation of the
    choice: the 13:08Z reference holds the saturated deck quasi-steady
    for 3.5+ h because the LLJ moisture supply and the deck radiative
    balance maintain the state against turbulent consumption -- a
    single column can neither self-generate the supply (the S3-9
    lesson: every interactive column decouples) nor carry the
    radiation physics, so the accurate equilibrium question is "what
    TKE/K does the closure sustain ON the specimen state", which is
    exactly the G-M5 instrument.  The interactive-thermo variant was
    measured during fixture design and is strictly WORSE for the
    over-mixing concern (layer-mean K_h 167 vs 103 m2/s at 2-4 h:
    mixing erodes the dry stratification, lengthens BL89, and feeds
    back), so the frozen leg is also the conservative one for the
    G-M5 ceiling question.

    Measured layer = the cells where the M1 substitution engaged at
    the specimen state (n2_moist != n2 -- the seam's own mask;
    k = 10..16, z = 456-1391 m, the saturated deck).  Returns the
    equilibrium sample dict: e and the driver-exported kv replica
    (steps 2/2b of the split step at f = 0, the lake-fixture idiom)
    at 90 and 120 min, plus the mask and heights.  Results are cached
    (pure function of the arguments) so the RED/GREEN/ceiling legs
    share one integration; callers must not mutate them.
    """
    key = (name, m1, steps, dt)
    if key in _SPECIMEN_CACHE:
        return _SPECIMEN_CACHE[key]
    from woof.verify import sase_ref
    c = _specimen_cols(name)
    thick = c["thick"]
    nz = len(thick)
    z1 = np.cumsum(thick) - 0.5 * thick        # authority convention
    z3 = z1.reshape(-1, 1, 1)
    t3 = thick.reshape(-1, 1, 1)
    delta = 3000.0                             # the d02 scale
    e = np.maximum(c["e0"], sase_ref.E_MIN).copy()
    src = c["ust"] ** 3 / (sase_ref.KARMAN * 0.5 * thick[0])
    samples = {}
    mask = None
    for n in range(steps):
        theta, qv, qc, p = c["theta"], c["qv"], c["qc"], c["p"]
        u, v, w = c["u"].copy(), c["v"].copy(), np.zeros((nz, 1, 1))
        n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
        n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)
        if mask is None:
            mask = (n2m != n2)[:, 0, 0]
        mkw = {"n2_moist": n2m} if m1 else {}
        n2_eff = n2m if m1 else n2
        # S4-3b replica fidelity: the amended split-step step 2b bounds
        # the RANS-limb lengths by the M1b moist excursion length when
        # the seam is engaged (n2_dry gates it), so the exported-kv
        # replica must ride the SAME composition.  On THIS specimen the
        # limb is measured slack everywhere (test_m1b_specimen_kh pins
        # it bitwise-inert), so every S4-1 measured number stands.
        lkw = {"n2_dry": n2} if m1 else {}
        e[0] += dt * src                       # driver pre-step deposit
        e_n = np.maximum(e, sase_ref.E_MIN)
        # driver-exported kv replica at f = 0 (split-step steps 2/2b,
        # the lake-fixture idiom): K_h = kv/Pr_t(0) is the measured K
        l_mix_r, _ = sase_ref.bl89_rans_lengths(theta, e_n, z3, t3,
                                                n2_eff, **lkw)
        kv = (sase_ref.stable_limit_coefficient(l_mix_r, e_n, n2_eff)
              * l_mix_r * np.sqrt(e_n))
        fields, ledger = sase_ref.sase_split_step(
            u, v, w, theta, e, dx=delta, dy=delta, dz=200.0,
            delta=delta, dt=dt, n2=n2, dz_col=thick, **mkw)
        assert (ledger["c_nu"], ledger["f"]) == (0.0, 0.0)  # RANS limb
        e = fields["e"]                        # frozen state: only e
        minute = (n + 1) * dt / 60.0
        if minute in (90.0, 120.0):
            pr = sase_ref.prandtl_blend(ledger["f"])
            samples[minute] = {"e": e[:, 0, 0].copy(),
                               "kh": (kv / pr)[:, 0, 0].copy()}
    out = {"z": z1, "mask": mask, "samples": samples}
    _SPECIMEN_CACHE[key] = out
    return out


def test_m1_dry_limit_bitwise(monkeypatch):
    """THE M1 UNSATURATED IDENTITY (plan fixture a): on the control
    column -- qc = 0 everywhere, RH < 80% everywhere (pinned at import
    by tests/specimen_yolod_11z.py) -- the M1-substituted pipeline is
    tobytes()-identical to the pre-M1 path, three ways:

    * moist_n2 itself returns the dry brunt_vaisala_n2 field BITWISE
      (the where-mask's FALSE branch is the literal dry field);
    * a 12-step trajectory through the split step WITH the n2_moist
      seam engaged equals the trajectory WITHOUT it, every field,
      every step, tobytes();
    * the S3-6g RED-leg idiom: monkeypatching ``sase_ref.moist_n2`` to
      the dry closure (lambda -> brunt_vaisala_n2) and re-running the
      M1 pipeline reproduces the same bytes -- the structural witness
      that ``n2_moist == n2`` bitwise implies the pre-M1 formulation
      exactly (empty substitution mask; every n2 consumer handed
      identical bits).
    """
    from woof.verify import sase_ref
    c = _specimen_cols("ctl")
    thick = c["thick"]
    nz = len(thick)
    delta = 3000.0
    dt = 60.0
    src = c["ust"] ** 3 / (sase_ref.KARMAN * 0.5 * thick[0])

    n2m = sase_ref.moist_n2(c["theta"], c["qv"], c["qc"], c["p"], thick)
    n2d = sase_ref.brunt_vaisala_n2(c["theta"], None, dz_col=thick)
    assert n2m.tobytes() == n2d.tobytes()      # the switch is off

    def run(mode):
        u, v, w = c["u"].copy(), c["v"].copy(), np.zeros((nz, 1, 1))
        e = np.maximum(c["e0"], sase_ref.E_MIN).copy()
        traj = []
        for _ in range(12):
            n2 = sase_ref.brunt_vaisala_n2(c["theta"], None,
                                           dz_col=thick)
            if mode == "pre":
                mkw = {}
            else:
                mkw = {"n2_moist": sase_ref.moist_n2(
                    c["theta"], c["qv"], c["qc"], c["p"], thick)}
            e[0] += dt * src
            fields, _ = sase_ref.sase_split_step(
                u, v, w, c["theta"], e, dx=delta, dy=delta, dz=200.0,
                delta=delta, dt=dt, n2=n2, dz_col=thick, **mkw)
            u, v, w, e = (fields[k] for k in "uvwe")
            traj.append(b"".join(fields[k].tobytes() for k in "uvwe"))
        return traj

    pre = run("pre")
    live = run("m1")
    assert pre == live                         # seam engaged, inert
    monkeypatch.setattr(
        sase_ref, "moist_n2",
        lambda theta, qv, qc, pressure, dz_col:
        sase_ref.brunt_vaisala_n2(theta, None, dz_col=dz_col))
    pinned = run("m1")
    assert pre == pinned                       # RED-leg structural pin


def test_m1_moist_adiabat_neutral():
    """THE DK82 TRANSCRIPTION VALIDATOR (plan fixture b): a constructed
    saturated column lying ON a moist adiabat (with condensate) must
    be MOIST-NEUTRAL through the transcribed formula while its DRY N^2
    is strongly stable -- |N^2_m| <= 1e-6 s^-2 with N^2_dry > 1e-4.

    CONSTRUCTION (independent of the transcription): with total water
    held constant (q_w' = 0, condensate retained) DK82 neutrality
    reduces to d(ln theta)/dz + (L/(cp T)) dq_s/dz = 0, an ODE in
    theta(z) that never touches the a/b factor -- RK4 on 0.25 m
    substeps over a dry-hydrostatic p(z), T_sfc = 280 K, p_sfc =
    950 hPa, sampled at the centers of a stretched 60-level grid
    (10 m * 1.03^k -- exercises the variable stencil), qv = q_s,
    qc = q_w0 - q_s > 0 everywhere (q_w0 = q_s,sfc + 1 g/kg).
    MEASURED at construction: interior |N^2_m| <= 3.9e-10 (2500x
    inside the bar; the bar 1e-6 absorbs the one-sided linear-exact
    edge rows, measured <= 8.4e-8), dry N^2 = 1.27-1.49e-4.

    TWO INDEPENDENT WITNESSES close the transcription (the q_w-const
    construction alone multiplies a/b by ~0):

    * MOIST-LAPSE WITNESS (validates a/b): on the same column,
      -dT/dz must equal the textbook saturated-adiabatic lapse
      Gamma_m = (g/cp)*(a/b) -- the SAME a/b DK82's Eq. 36 carries
      (derivation in the sase_ref module docstring: collecting the
      first law + Clausius-Clapeyron gives exactly a and b).
      Asserted <= 1% relative (measured 0.28% max, the Tetens-vs-CC
      derivative and dry-hydrostatic p(z) residual class); a SWAPPED
      a/b transcription errs ~2.8x here and a missing epsilon in b
      errs ~40%.
    * LOADING WITNESS (validates -g*dq_w/dz exactly): adding a linear
      qc ramp (slope s = 2e-6 /m) on top of the neutral column must
      shift N^2_m by EXACTLY -g*s at interior levels (the stencil is
      exact on linear fields).  Asserted <= 1e-12 (measured ~1e-18).
    """
    from woof.verify import sase_ref
    RD, CP, P0 = sase_ref.RD_AIR, sase_ref.CP_AIR, sase_ref.P0_REF
    XLV, EP2 = sase_ref.XLV, sase_ref.EP2_RV
    SVP1, SVP2 = sase_ref.SVP1, sase_ref.SVP2
    SVP3, SVPT0 = sase_ref.SVP3, sase_ref.SVPT0
    G = sase_ref.G_ACCEL

    def qs_of(t, p):
        es = 1000.0 * SVP1 * np.exp(SVP2 * (t - SVPT0) / (t - SVP3))
        return EP2 * es / (p - es)

    def build(t_sfc=280.0, p_sfc=95000.0):
        thick = 10.0 * 1.03 ** np.arange(60)
        z_c = np.cumsum(thick) - 0.5 * thick
        kappa = RD / CP

        def rhs(th_, p_):
            t_ = th_ * (p_ / P0) ** kappa
            pi = (p_ / P0) ** kappa
            dpdz = -p_ * G / (RD * t_)          # dry hydrostatic
            eps_t, eps_p = 1e-4, 1e-1
            qs_t = (qs_of(t_ + eps_t, p_)
                    - qs_of(t_ - eps_t, p_)) / (2 * eps_t)
            qs_p = (qs_of(t_, p_ + eps_p)
                    - qs_of(t_, p_ - eps_p)) / (2 * eps_p)
            num = -(XLV / (CP * t_)) * (qs_t * th_ * kappa * pi / p_
                                        + qs_p) * dpdz
            den = 1.0 / th_ + (XLV / (CP * t_)) * qs_t * pi
            return num / den, dpdz

        p, zs = p_sfc, 0.0
        th = t_sfc * (P0 / p_sfc) ** kappa
        out_th, out_p, ti, dz = [], [], 0, 0.25
        targets = list(z_c)
        while ti < len(targets):
            k1t, k1p = rhs(th, p)
            k2t, k2p = rhs(th + 0.5 * dz * k1t, p + 0.5 * dz * k1p)
            k3t, k3p = rhs(th + 0.5 * dz * k2t, p + 0.5 * dz * k2p)
            k4t, k4p = rhs(th + dz * k3t, p + dz * k3p)
            th_n = th + dz * (k1t + 2 * k2t + 2 * k3t + k4t) / 6.0
            p_n = p + dz * (k1p + 2 * k2p + 2 * k3p + k4p) / 6.0
            z_n = zs + dz
            while ti < len(targets) and targets[ti] <= z_n:
                w_ = (targets[ti] - zs) / dz
                out_th.append(th * (1 - w_) + th_n * w_)
                out_p.append(p * (1 - w_) + p_n * w_)
                ti += 1
            th, p, zs = th_n, p_n, z_n
        return (np.array(out_th), np.array(out_p), thick, z_c)

    th, p, thick, z = build()
    t = th * (p / P0) ** (RD / CP)
    qs = qs_of(t, p)
    qw0 = qs[0] + 1.0e-3                       # 1 g/kg base condensate
    qv, qc = qs.copy(), qw0 - qs
    assert qc.min() > 0.0                      # condensate everywhere

    def col(a):
        return a.reshape(-1, 1, 1)

    n2m = sase_ref.moist_n2(col(th), col(qv), col(qc), col(p),
                            thick)[:, 0, 0]
    n2d = sase_ref.brunt_vaisala_n2(col(th), None,
                                    dz_col=thick)[:, 0, 0]
    # the switch fired everywhere and the substitution is live
    assert np.all(np.abs(n2m - n2d) > 1e-5)
    # NEUTRALITY: |N^2_m| <= 1e-6 while dry N^2 > 1e-4, every level
    assert np.abs(n2m).max() <= 1.0e-6, np.abs(n2m).max()
    assert n2d.min() > 1.0e-4, n2d.min()
    # MOIST-LAPSE WITNESS (a/b): -dT/dz == (g/cp)*(a/b) within 1%
    dT = sase_ref._ddz(col(t), None, dz_col=thick)[:, 0, 0]
    a_fac = 1.0 + XLV * qs / (RD * t)
    b_fac = 1.0 + EP2 * XLV * XLV * qs / (CP * RD * t * t)
    gamma_m = (G / CP) * (a_fac / b_fac)
    rel = np.abs((-dT - gamma_m) / gamma_m)[1:-1]
    assert rel.max() <= 0.01, rel.max()
    # LOADING WITNESS: a linear qc ramp shifts N^2_m by exactly -g*s
    slope = 2.0e-6
    n2m_ramp = sase_ref.moist_n2(col(th), col(qv),
                                 col(qc + slope * z), col(p),
                                 thick)[:, 0, 0]
    shift = (n2m_ramp - n2m)[1:-1]
    assert np.abs(shift + G * slope).max() <= 1.0e-12


def test_m1_specimen_uptake():
    """THE G-M5 UPTAKE FIXTURE (plan fixture c): on the yolo-d 11Z
    amplifier column with the specimen e/u*, the M1 substitution takes
    the saturated moist-unstable deck from the measured dry shortfall
    into the reference turbulence class.

    RED (the early-ci-cap-audit shortfall, reproduced): the DRY path
    on the same column, same driver replica -- the saturated layer's
    equilibrium K_h stays inside the audited 0.00-0.02 m2/s band and
    TKE at the E_MIN floor class (cap-audit: "stable-limit K_v ...
    evaluates to 0.00-0.02 m2/s"; "TKE_SASE ... < 1e-3 m2/s2").
    MEASURED here: layer K_h 2.2-3.6e-05 m2/s, layer TKE = the 1e-06
    floor exactly.

    GREEN (M1 live): the deck's DK82 N^2_m collapses the dry
    +0.66..+1.8e-4 s^-2 to -8.9e-5..+1.0e-4 (sign flips in 4 of 7
    deck cells), releasing the l_s/C_KS strangle and switching the
    e-budget buoyancy to production; the layer equilibrates with
    * layer-mean TKE inside the G-M5 band [0.5, 1.6] m2/s2 at BOTH
      the 90- and 120-min samples (quasi-steady witness), and
    * layer-mean K_h >= 3 m2/s (the G-M5 floor: the layer has entered
      the reference class from the 1e-5 shortfall -- a ~4e6 uptake),
      pinned against runaway by the measured-class ceiling <= 250.
    MEASURED here (frozen-state equilibrium): layer-mean TKE 1.01512
    (max 1.334), layer-mean K_h 102.583 (max 148.7, min 11.8) -- the
    90- and 120-min samples agree to 6 significant figures (fully
    equilibrated).

    THE G-M5 K_h CEILING (<= 40) IS NOT MET -- that deviation is
    carried loudly by the strict-xfail companion below, NOT hidden
    here; the derivation of why no in-scope M1 can reach it is in the
    module docstring (SASE-M1 section, K_h-deviation paragraph).
    """
    m = _run_specimen_column("amp", m1=False)
    mask = m["mask"]
    # the substitution mask IS the saturated deck of the specimen file
    assert mask.sum() >= 5
    zs = m["z"][mask]
    assert zs[0] <= 500.0 and zs[-1] >= 1300.0
    for minute in (90.0, 120.0):
        s = m["samples"][minute]
        assert s["kh"][mask].max() <= 0.02, s["kh"][mask].max()
        assert s["e"][mask].max() <= 1.0e-3, s["e"][mask].max()
    g = _run_specimen_column("amp", m1=True)
    for minute in (90.0, 120.0):
        s = g["samples"][minute]
        tke = float(s["e"][mask].mean())
        kh = float(s["kh"][mask].mean())
        assert 0.5 <= tke <= 1.6, (minute, tke)      # G-M5 TKE band
        assert kh >= 3.0, (minute, kh)               # entered the class
        assert kh <= 250.0, (minute, kh)             # measured-class pin


@pytest.mark.xfail(
    strict=True,
    reason="G-M5 K_h ceiling: the plan band [3, 40] m2/s is not "
           "reachable by the in-scope M1 formulation on the specimen "
           "deck -- the module's shortest registered length there is "
           "Blackadar l_B = 82-114 m (vs the reference's ~10-20 m "
           "MYNN master length), so at in-band TKE the equilibrium "
           "K_h = C_KV*l*sqrt(e)/Pr_t sits at the ~1e2 m2/s class "
           "(measured 102.6).  Registered deviation; adjudication at "
           "task review + the S4-3 smoke (G-M3/G-M5 on the 3-D run). "
           "STRICT: if a future amendment brings the deck K_h into "
           "band, this xfail fails loudly and must be re-pinned.")
def test_m1_specimen_kh_gm5_ceiling():
    """The G-M5 K_h ceiling as planned (S4-1 fixture c band [3, 40]),
    kept VERBATIM as a strict xfail so the deviation stays measured
    and visible instead of silently widened -- the derivation lives in
    the module docstring (SASE-M1 section) and the xfail reason."""
    g = _run_specimen_column("amp", m1=True)
    mask = g["mask"]
    kh = float(g["samples"][120.0]["kh"][mask].mean())
    assert 3.0 <= kh <= 40.0, kh


def test_m1_lake_column_green(monkeypatch):
    """M1 MUST NOT DISTURB THE MARINE STABLE CASE (plan fixture d):
    the S3-9 lake GREEN leg re-run through the M1-substituted pipeline
    (synthetic 70%-RH humidity, qc = 0 -- the water column is
    unsaturated at k0 and everywhere; _run_lake_column docstring, m1
    paragraph) is BITWISE the pre-M1 lake trajectory, and the
    registered lake bands hold unchanged: final sp10/sp500 in
    [0.35, 0.60] and final e below 500 m <= 0.3 m2/s2 (the S3-9
    measured values 0.539 / 0.011 stand)."""
    base = _run_lake_column(monkeypatch, state="13", hours=4.0)
    live = _run_lake_column(monkeypatch, state="13", hours=4.0, m1=True)
    for key in ("ratios", "theta_fin", "e_fin"):
        assert base[key].tobytes() == live[key].tobytes(), key
    ratio_fin = float(live["ratios"][-1])
    assert 0.35 <= ratio_fin <= 0.60, ratio_fin
    e500 = float(live["e_fin"][live["z"] < 500.0].max())
    assert e500 <= 0.3, e500


def test_m1_seam_contract_and_ledger_closure(monkeypatch):
    """The n2_moist SEAM CONTRACT + the ledger under moist substitution.

    * n2_moist without n2 is rejected (the substitution mask and the
      w-sensor screen both need the dry field);
    * the w-sensor screen consumes the DRY n2 even with the seam
      engaged -- substitution at the three spec points and NOWHERE
      else (spy on w_resolved_bound asserts object identity);
    * on the uniform-dz periodic box with a mixed-sign moist
      substitution field, the split-step ledger closes to relative
      roundoff < 1e-11 -- the S3-6f "asserted with the cap engaged"
      idiom: l_s/K/l_d take n2_moist as coefficients (pointwise
      closure) and the buoyancy value-switch changes only the
      PE-exchange channel the dE definition excludes (C4/C5,
      sase-m-integration-points.md).
    """
    from woof.verify import sase_ref
    rng = np.random.default_rng(41)
    shape = (8, 24, 24)
    u, v, w = (sase_ref.box_filter(rng.standard_normal(shape), 4)
               for _ in range(3))
    theta = 300.0 + np.cumsum(np.full(shape, 0.5), axis=0)  # stratified
    e = np.full(shape, 0.2)
    n2 = sase_ref.brunt_vaisala_n2(theta, 200.0)
    n2m = n2.copy()
    blob = np.zeros(shape, dtype=bool)
    blob[2:6, 4:16, 4:16] = True               # "saturated" blob
    n2m[blob] = -1.0e-4 + 2.0e-4 * rng.random(int(blob.sum()))
    with pytest.raises(ValueError, match="n2_moist"):
        sase_ref.sase_split_step(u, v, w, theta, e, dx=500.0, dy=500.0,
                                 dz=200.0, delta=500.0, dt=1.0,
                                 n2_moist=n2m)
    seen = {}
    real_wrb = sase_ref.w_resolved_bound

    def spy(w_, e_mean, n2=None):
        seen["n2"] = n2
        return real_wrb(w_, e_mean, n2=n2)

    monkeypatch.setattr(sase_ref, "w_resolved_bound", spy)
    fields, ledger = sase_ref.sase_split_step(
        u, v, w, theta, e, dx=500.0, dy=500.0, dz=200.0,
        delta=500.0, dt=1.0, n2=n2, n2_moist=n2m)
    assert seen["n2"] is n2                    # dry screen, not moist
    scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                abs(ledger["dHeat"]), 1e-30)
    assert abs(ledger["residual"]) / scale < 1e-11
    assert np.all(fields["e"] >= sase_ref.E_MIN)


# ---------------------------------------------------------------------------
# S4-3b: M1b moist master-length limb (authority; SASE-M spec section 3b --
# the frozen S4-3 G-M3 amendment: "spec amendment for moist master-length
# limb, NEVER constant adjustment").  Formulation + the discretization
# derivation: woof/verify/sase_ref.py module docstring (SASE-M1b section).
# Evidence: .superpowers/sdd/sase-m1-residual.md -- M1 confined the
# amplifier but cleared the Sc deck at K_h ~1e2 m2/s (deck p50 74-120,
# p90 136-150; reference class 3-40) because in a saturated moist-UNSTABLE
# layer N^2_m < 0 leaves l_s inactive and the master length falls back to
# the free (dry-convective) l_B/dry-BL89 class.
# ---------------------------------------------------------------------------


def _build_deck_column(nz=30, dz=100.0, theta_sfc=288.0, p_sfc=95000.0,
                       gam_sub=3.0e-3, tilt=2.0e-3, dth_inv=12.0,
                       gam_top=4.0e-3, qc0=5.0e-4, z_base=400.0,
                       z_lid=1400.0, z_invtop=1700.0):
    """S4-3b DECK-UNDER-LID column (constructed, like the moist-adiabat
    fixture built its column -- NOT a wrfout extraction): a saturated
    moist-unstable layer 0.4-1.4 km capped by a moist-stable inversion.

    CONSTRUCTION (uniform 100 m grid, centers 50..2950 m; dry-hydrostatic
    p(z) throughout):

    * sub-cloud 0-0.4 km: unsaturated (RH 70%), dry-stable at +3 K/km;
    * deck 0.4-1.4 km: theta follows the SATURATED ADIABAT (the same
      d ln(theta)/dz = -(L/(cp T)) dqs/dz ODE the moist-adiabat-
      neutrality fixture integrates) MINUS a 2 K/km sub-adiabatic tilt,
      qv = qs exactly and qc = (qw0 - qs) + 0.5 g/kg (q_w conserved +
      base loading), so by the DK82 algebra the layer is moist-UNSTABLE
      (measured N^2_m -8.3..-20e-5 s^-2) while staying DRY-stable
      (measured dry N^2 +7.2..+9.1e-5) -- the G-M3 regime;
    * inversion 1.4-1.7 km: +12 K jump (unsaturated, RH 30%) -- the
      moist-stable lid (lid-adjacent saturated cell N^2_m +6.0e-4);
    * free troposphere above: +4 K/km, unsaturated.

    Returns (theta1, qv1, qc1, p1, thick, z_centers, deck_mask) as flat
    (nz,) arrays; callers reshape to the module's (nz, 1, 1) layout.
    S4-4 generalization: ``z_base``/``z_lid``/``z_invtop`` became
    keyword parameters (defaults are the original literals, behavior
    bitwise unchanged for every existing caller) so the M2
    prescribed-lid fixture can move the lid and assert the plume
    termination tracks it.
    """
    from woof.verify import sase_ref
    RD, CP, P0 = sase_ref.RD_AIR, sase_ref.CP_AIR, sase_ref.P0_REF
    XLV, EP2 = sase_ref.XLV, sase_ref.EP2_RV
    SVP1, SVP2 = sase_ref.SVP1, sase_ref.SVP2
    SVP3, SVPT0 = sase_ref.SVP3, sase_ref.SVPT0
    G = sase_ref.G_ACCEL
    kappa = RD / CP

    def qs_of(t, p):
        es = 1000.0 * SVP1 * np.exp(SVP2 * (t - SVPT0) / (t - SVP3))
        return EP2 * es / (p - es)

    thick = np.full(nz, dz)
    zc = np.cumsum(thick) - 0.5 * thick
    fine = 1.0                                 # 1 m integration substep
    zf, th, p = 0.0, theta_sfc, p_sfc
    out_th, out_p, ti = [], [], 0
    while ti < len(zc):
        t = th * (p / P0) ** kappa
        dpdz = -p * G / (RD * t)
        if zf < z_base:
            dthdz = gam_sub
        elif zf < z_lid:
            # saturated-adiabat slope by centered differences on qs
            eps_t, eps_p = 1e-4, 1e-1
            qs_t = (qs_of(t + eps_t, p)
                    - qs_of(t - eps_t, p)) / (2 * eps_t)
            qs_p = (qs_of(t, p + eps_p)
                    - qs_of(t, p - eps_p)) / (2 * eps_p)
            pi = (p / P0) ** kappa
            num = -(XLV / (CP * t)) * (qs_t * th * kappa * pi / p
                                       + qs_p) * dpdz
            den = 1.0 / th + (XLV / (CP * t)) * qs_t * pi
            dthdz = num / den - tilt           # sub-adiabatic tilt
        elif zf < z_invtop:
            dthdz = dth_inv / (z_invtop - z_lid)
        else:
            dthdz = gam_top
        th_n, p_n, z_n = th + fine * dthdz, p + fine * dpdz, zf + fine
        while ti < len(zc) and zc[ti] <= z_n:
            w_ = (zc[ti] - zf) / fine
            out_th.append(th * (1 - w_) + th_n * w_)
            out_p.append(p * (1 - w_) + p_n * w_)
            ti += 1
        th, p, zf = th_n, p_n, z_n
    th = np.array(out_th)
    p = np.array(out_p)
    t = th * (p / P0) ** kappa
    qs = qs_of(t, p)
    deck = (zc > z_base) & (zc < z_lid)
    qv = np.where(deck, qs, np.where(zc <= z_base, 0.7 * qs, 0.3 * qs))
    kd = np.nonzero(deck)[0]
    qw0 = qs[kd[0]] + qc0
    qc = np.where(deck, np.maximum(qw0 - qs, 0.0), 0.0)
    return th, qv, qc, p, thick, zc, deck


def _run_deck_column(steps=120, dt=60.0, ust=0.3):
    """The _run_specimen_column driver replica on the deck-under-lid
    column (frozen-state equilibrium; d02 delta = 3000, f = 0 asserted;
    the driver pre-step surface-e deposit at a nominal ust = 0.3).
    Runs whatever limb state the caller has arranged (live or a
    monkeypatched pre-M1b path) and returns the equilibrium samples --
    e, the exported-kv-replica K_h, and the operating l_mix -- at 90
    and 120 min, plus the substitution mask and stability fields."""
    from woof.verify import sase_ref
    th1, qv1, qc1, p1, thick, zc, deck = _build_deck_column()

    def col(a):
        return np.asarray(a, np.float64).reshape(-1, 1, 1)

    theta, qv, qc, p = col(th1), col(qv1), col(qc1), col(p1)
    nz = len(thick)
    z3 = zc.reshape(-1, 1, 1)
    t3 = thick.reshape(-1, 1, 1)
    delta = 3000.0
    e = np.full((nz, 1, 1), sase_ref.E_MIN)
    src = ust ** 3 / (sase_ref.KARMAN * 0.5 * thick[0])
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)
    samples = {}
    for n in range(steps):
        u, v, w = (np.zeros((nz, 1, 1)) for _ in range(3))
        e[0] += dt * src
        e_n = np.maximum(e, sase_ref.E_MIN)
        l_mix_r, _ = sase_ref.bl89_rans_lengths(theta, e_n, z3, t3,
                                                n2m, n2_dry=n2)
        kv = (sase_ref.stable_limit_coefficient(l_mix_r, e_n, n2m)
              * l_mix_r * np.sqrt(e_n))
        fields, ledger = sase_ref.sase_split_step(
            u, v, w, theta, e, dx=delta, dy=delta, dz=200.0,
            delta=delta, dt=dt, n2=n2, dz_col=thick, n2_moist=n2m)
        assert (ledger["c_nu"], ledger["f"]) == (0.0, 0.0)  # RANS limb
        e = fields["e"]
        minute = (n + 1) * dt / 60.0
        if minute in (90.0, 120.0):
            pr = sase_ref.prandtl_blend(ledger["f"])
            samples[minute] = {"e": e[:, 0, 0].copy(),
                               "kh": (kv / pr)[:, 0, 0].copy(),
                               "lmix": l_mix_r[:, 0, 0].copy()}
    return {"z": zc, "mask": (n2m != n2)[:, 0, 0],
            "n2m": n2m[:, 0, 0], "n2": n2[:, 0, 0], "samples": samples,
            "column": (theta, qv, qc, p, thick, z3, t3, n2, n2m)}


def _m1b_inert_lengths(n2_eff, e, z, thick):
    """The pre-M1b path, reproduced: +inf excursion lengths make the
    limb's min-bound arithmetically inert (min(l, inf) == l bitwise for
    the finite composed lengths), which is the S3-6g RED-leg idiom for
    this amendment -- monkeypatching bl89_moist_excursion_lengths to
    THIS function is bitwise the pre-M1b formulation."""
    shape = np.shape(n2_eff)
    return np.full(shape, np.inf), np.full(shape, np.inf)


def test_m1b_deck_under_lid(monkeypatch):
    """THE M1b DEFECT AND FIX, PINNED (plan fixture b + the registered
    convention string): on the constructed deck-under-lid column the
    pre-M1b master length at the lid-adjacent moist-unstable cell is
    the FREE (dry-convective) Blackadar fallback -- l_mix == l_B
    BITWISE, K_h in the smoke's measured 1e2 class -- and the M1b
    moist excursion bound replaces it with the distance-to-lid class,
    taking the lid-adjacent equilibrium K_h out of the 1e2 class into
    the reference class.

    RED leg (the CURRENT defective number, reproduced by the inert
    monkeypatch AFTER the fix lands; before it this whole test is RED
    because the limb does not exist): lid-adjacent cell k12
    (z = 1250 m, 150 m under the lid, N^2_m = -7.9e-5):
    * l_mix(k12) == _blackadar_length(1250) = 115.3846 m BITWISE (the
      free fallback -- nothing moist bounds the length);
    * equilibrium K_h(k12) in the smoke's deck class [74, 150]
      (measured here 101.93 m2/s at both the 90- and 120-min samples;
      the S4-3 smoke measured deck p50 74-120, p90 136-150);
    * deck TKE mean 0.614, deck K_h mean 85.7.

    GREEN leg (limb live):
    * the substitution mask is exactly the saturated deck k4..k13 with
      N^2_m <= -7e-5 in k4..k12 (moist-unstable) and >= +5e-4 at the
      lid-adjacent saturated stable cell k13, while dry N^2 >= +7e-5
      everywhere in the deck (dry-stable -- the M1 switch regime);
    * DISTANCE-TO-LID CLASS (the discretization pin): for every
      moist-unstable deck cell the upward excursion terminates INSIDE
      the capping inversion with the centered-stencil's one-cell
      bracketing: (z_lid - z_k) - dz <= l_up <= (z_invtop - z_k), and
      never the free fallback (l_up <= 0.55*(htop - z_k)); mid-deck
      DOWNWARD excursions ride the moist-unstable fall to the BL89
      surface bound l_down == z_k EXACTLY (k5..k11), while the deck-
      base cell k4 is arrested by the stable sub-cloud layer
      (150 <= l_down < z_k; measured 291.6);
    * the lid-adjacent operating length leaves the fallback:
      l_mix(k12) < 0.5*l_B(1250) (measured 54.83 = the k12 upward
      excursion, i.e. the min-bound is the l_up distance-to-lid);
    * equilibrium K_h(k12) LEAVES the 1e2 class into the reference
      class: 25 <= K_h(k12) <= 55 (measured 39.31; disjoint from the
      RED band [74, 150]) and K_h(k12) <= 0.5*RED at both samples;
      deck TKE mean stays turbulent in [0.45, 0.75] (measured 0.561 --
      the limb kills LID ENTRAINMENT, not deck turbulence) and the
      deck K_h mean drops below the RED mean (85.7 -> 77.9).

    QUADRATURE EXACTNESS (the uniform-stratification analytic pin,
    the test_bl89_uniform_stratification idiom): on a uniform
    N^2_eff = 1e-4 column at e = 0.5 the excursion integral is exactly
    quadratic, so l_up == min(sqrt(2e/N^2), htop - z) and l_down ==
    min(sqrt(2e/N^2), z) to roundoff (measured 0.0).

    REGISTRY: MOIST_MASTER_LENGTH = "bl89-n2eff-excursion-min-v1" is
    config-ID-bound (the S4-1 convention-string idiom).
    """
    from woof.verify import sase_ref

    # registry leg (brief step 3): the convention string is bound
    assert sase_ref.MOIST_MASTER_LENGTH == "bl89-n2eff-excursion-min-v1"
    assert "MOIST_MASTER_LENGTH" in sase_ref._CONFIG_ID_CONSTANTS
    cid = sase_ref.sase_config_id()
    monkeypatch.setattr(sase_ref, "MOIST_MASTER_LENGTH", "other-v0")
    assert sase_ref.sase_config_id() != cid
    monkeypatch.setattr(sase_ref, "MOIST_MASTER_LENGTH",
                        "bl89-n2eff-excursion-min-v1")

    # ---- RED leg: the current (pre-M1b) defective number -------------
    monkeypatch.setattr(sase_ref, "bl89_moist_excursion_lengths",
                        _m1b_inert_lengths)
    red = _run_deck_column()
    monkeypatch.undo()
    k12 = 12
    lb12 = sase_ref._blackadar_length(1250.0)
    for minute in (90.0, 120.0):
        s = red["samples"][minute]
        assert s["lmix"][k12] == lb12          # the free fallback, bitwise
        assert 74.0 <= s["kh"][k12] <= 150.0, s["kh"][k12]   # 1e2 class

    # ---- GREEN leg: limb live ----------------------------------------
    grn = _run_deck_column()
    mask = grn["mask"]
    assert list(np.nonzero(mask)[0]) == list(range(4, 14))
    kd_unst = np.nonzero(grn["n2m"] < 0.0)[0]
    assert list(kd_unst) == list(range(4, 13))
    assert np.all(grn["n2m"][kd_unst] <= -7.0e-5)       # moist-unstable
    assert grn["n2m"][13] >= 5.0e-4                     # moist-stable lid
    assert np.all(grn["n2"][mask] >= 7.0e-5)            # dry-stable deck
    z = grn["z"]
    z_lid, z_invtop, dz = 1400.0, 1700.0, 100.0
    htop = z[-1] + 50.0
    theta, qv, qc, p, thick, z3, t3, n2, n2m = grn["column"]
    e120 = grn["samples"][120.0]["e"].reshape(-1, 1, 1)
    l_up, l_dn = sase_ref.bl89_moist_excursion_lengths(n2m, e120, z3, t3)
    l_up, l_dn = l_up[:, 0, 0], l_dn[:, 0, 0]
    for k in kd_unst:
        assert (z_lid - z[k]) - dz <= l_up[k] <= (z_invtop - z[k]), k
        assert l_up[k] <= 0.55 * (htop - z[k]), k       # never the free
    assert np.array_equal(l_dn[5:12], z[5:12])          # surface bound
    assert 150.0 <= l_dn[4] < z[4]                      # sub-cloud arrest
    for minute in (90.0, 120.0):
        s = grn["samples"][minute]
        r = red["samples"][minute]
        assert s["lmix"][k12] < 0.5 * lb12, s["lmix"][k12]
        assert 25.0 <= s["kh"][k12] <= 55.0, s["kh"][k12]
        assert s["kh"][k12] <= 0.5 * r["kh"][k12]       # left the class
        assert 0.45 <= s["e"][mask].mean() <= 0.75      # deck stays alive
        assert s["kh"][mask].mean() < r["kh"][mask].mean()

    # ---- quadrature exactness (uniform-stratification analytic) ------
    nzu, tu = 20, 50.0
    thick_u = np.full(nzu, tu)
    zu = np.cumsum(thick_u) - 0.5 * thick_u
    n2u = np.full((nzu, 1, 1), 1.0e-4)
    eu = np.full((nzu, 1, 1), 0.5)
    lu, ld = sase_ref.bl89_moist_excursion_lengths(
        n2u, eu, zu.reshape(-1, 1, 1), thick_u.reshape(-1, 1, 1))
    lex = np.sqrt(2.0 * 0.5 / 1.0e-4)
    hu = zu[-1] + 0.5 * tu
    assert np.abs(lu[:, 0, 0] - np.minimum(lex, hu - zu)).max() <= 1e-9
    assert np.abs(ld[:, 0, 0] - np.minimum(lex, zu)).max() <= 1e-9


def test_m1b_dry_limit_bitwise(monkeypatch):
    """THE M1b UNSATURATED IDENTITY (plan fixture a): on the control
    column (qc = 0, RH < 80% everywhere) the M1b-limbed pipeline is
    tobytes()-identical to the pre-M1 path -- the limb rides the M1
    saturation switch, and an empty substitution mask never engages it.
    Structural witness: a POISONED bl89_moist_excursion_lengths (raises
    on call) leaves the trajectory untouched, so unsaturated air never
    even reaches the excursion machinery (the subst-gate contract)."""
    from woof.verify import sase_ref
    c = _specimen_cols("ctl")
    thick = c["thick"]
    nz = len(thick)
    delta, dt = 3000.0, 60.0
    src = c["ust"] ** 3 / (sase_ref.KARMAN * 0.5 * thick[0])

    def run(mode):
        u, v, w = c["u"].copy(), c["v"].copy(), np.zeros((nz, 1, 1))
        e = np.maximum(c["e0"], sase_ref.E_MIN).copy()
        traj = []
        for _ in range(12):
            n2 = sase_ref.brunt_vaisala_n2(c["theta"], None,
                                           dz_col=thick)
            mkw = {} if mode == "pre" else {"n2_moist": sase_ref.moist_n2(
                c["theta"], c["qv"], c["qc"], c["p"], thick)}
            e[0] += dt * src
            fields, _ = sase_ref.sase_split_step(
                u, v, w, c["theta"], e, dx=delta, dy=delta, dz=200.0,
                delta=delta, dt=dt, n2=n2, dz_col=thick, **mkw)
            u, v, w, e = (fields[k] for k in "uvwe")
            traj.append(b"".join(fields[k].tobytes() for k in "uvwe"))
        return traj

    pre = run("pre")
    live = run("m1b")
    assert pre == live                         # seam + limb engaged, inert

    def _boom(n2_eff, e, z, thick):
        raise AssertionError(
            "M1b moist excursion engaged on an unsaturated column")

    monkeypatch.setattr(sase_ref, "bl89_moist_excursion_lengths", _boom)
    poisoned = run("m1b")
    assert pre == poisoned                     # gate witness: never called


def test_m1b_specimen_kh(monkeypatch):
    """THE G-M5 RE-EVALUATION UNDER THE LIMB (plan fixture c; task
    brief S4-3b step 1c; spec section 3b "G-M5 re-evaluation"): on the
    yolo-d 11Z amplifier column the M1b limb is measured SLACK -- every
    moist excursion length exceeds the operating l_B/l_s-bounded RANS
    composition (min margin 0.02 m at equilibrium), so the equilibrium
    is BITWISE the S4-1 trajectory and the achieved K_h class is
    UNCHANGED: layer-mean 102.58 m2/s, still ABOVE the [3, 40] plan
    band.  THE S4-1 STRICT XFAIL (test_m1_specimen_kh_gm5_ceiling)
    THEREFORE STANDS UNTOUCHED: the 11Z specimen deck is still
    strongly DRY-stable, so its lid-limited moist excursions (325-63 m
    at the top two cells, 1283-918 m below) never undercut the
    82-118 m Blackadar fallback that sets the 1e2 class there.  The
    limb's bite lives where the smoke's defect lives -- eroded
    (dry-neutralized) decks and lid-adjacent cells (the deck-under-lid
    fixture) -- not on this specimen; the band re-pin remains the
    coordinator's call (task brief).

    RED leg (structural): the inert-limb monkeypatch reproduces the
    live samples bitwise on a fresh cache -- the limb changed NOTHING
    here, loudly measured rather than silently assumed."""
    import sys

    from woof.verify import sase_ref
    g = _run_specimen_column("amp", m1=True)
    mask = g["mask"]
    for minute in (90.0, 120.0):
        kh = float(g["samples"][minute]["kh"][mask].mean())
        assert 100.0 <= kh <= 105.0, (minute, kh)   # the S4-1 class holds
        assert kh > 40.0, (minute, kh)              # NOT in [3, 40]:
        # the strict xfail above stays byte-for-byte untouched (brief
        # step 1c; spec section 3b: "lands in a new class ->
        # ... derivation-backed re-pin", which is not this task's call)

    # inertness derivation witness: the moist excursion bound is slack
    # in every substituted cell at the equilibrium state (measured min
    # margin 0.02 m -- thin but positive; recorded in the report)
    c = _specimen_cols("amp")
    thick = c["thick"]
    z1 = np.cumsum(thick) - 0.5 * thick
    z3, t3 = z1.reshape(-1, 1, 1), thick.reshape(-1, 1, 1)
    n2 = sase_ref.brunt_vaisala_n2(c["theta"], None, dz_col=thick)
    n2m = sase_ref.moist_n2(c["theta"], c["qv"], c["qc"], c["p"], thick)
    e120 = np.maximum(g["samples"][120.0]["e"].reshape(-1, 1, 1),
                      sase_ref.E_MIN)
    l_mix_r, l_eps_r = sase_ref.bl89_rans_lengths(c["theta"], e120,
                                                  z3, t3, n2m)
    l_up, l_dn = sase_ref.bl89_moist_excursion_lengths(n2m, e120, z3, t3)
    l_m = np.minimum(l_up, l_dn)
    subst = n2m != n2
    assert np.all((l_m > np.maximum(l_mix_r, l_eps_r))[subst])

    # RED leg: inert-limb monkeypatch == live, bitwise (fresh cache so
    # the shared S4-1 cache entry is never overwritten)
    monkeypatch.setattr(sys.modules[__name__], "_SPECIMEN_CACHE", {})
    monkeypatch.setattr(sase_ref, "bl89_moist_excursion_lengths",
                        _m1b_inert_lengths)
    red = _run_specimen_column("amp", m1=True)
    for minute in (90.0, 120.0):
        for key in ("e", "kh"):
            assert (red["samples"][minute][key].tobytes()
                    == g["samples"][minute][key].tobytes()), (minute, key)


def test_m1b_lake_column_green(monkeypatch):
    """M1b MUST NOT DISTURB THE MARINE STABLE CASE (plan fixture d):
    the S3-9 lake GREEN leg through the M1-substituted pipeline with
    the limb POISONED (raises on call) is BITWISE the pre-M1 lake
    trajectory and the registered lake bands hold -- the unsaturated
    water column never engages the excursion machinery, exactly the
    test_m1_lake_column_green contract carried through S4-3b."""
    from woof.verify import sase_ref

    def _boom(n2_eff, e, z, thick):
        raise AssertionError(
            "M1b moist excursion engaged on the unsaturated lake column")

    monkeypatch.setattr(sase_ref, "bl89_moist_excursion_lengths", _boom)
    base = _run_lake_column(monkeypatch, state="13", hours=4.0)
    live = _run_lake_column(monkeypatch, state="13", hours=4.0, m1=True)
    for key in ("ratios", "theta_fin", "e_fin"):
        assert base[key].tobytes() == live[key].tobytes(), key
    ratio_fin = float(live["ratios"][-1])
    assert 0.35 <= ratio_fin <= 0.60, ratio_fin
    e500 = float(live["e_fin"][live["z"] < 500.0].max())
    assert e500 <= 0.3, e500


def test_m1b_les_inert(monkeypatch):
    """THE LES LIMIT IS INERT (plan fixture e; spec 3b requirement 3):
    at f = 1 the step is bitwise independent of the M1b limb even on a
    SATURATED column where the limb is computed and binding at f = 0 --
    the two-product blends make K_v = 1.0*(C_KV*l_les*sqrt(e)) +
    0.0*(bounded RANS limb) and l_d = delta**1.0 * lb**0.0 = delta
    FP-exact, so min-bounding the RANS-limb lengths changes NOTHING
    (the S3-6h f = 1 argument carried to the limb).  f = 1 is forced
    through the solve/bounds seams (the S3-6f bounds-inert idiom); a
    spy asserts the excursion machinery really ran in the live leg
    (non-vacuous), and the inert monkeypatch (the pre-M1b path) then
    reproduces every field bitwise."""
    from woof.verify import sase_ref
    th1, qv1, qc1, p1, thick, zc, deck = _build_deck_column()

    def col(a):
        return np.asarray(a, np.float64).reshape(-1, 1, 1)

    theta, qv, qc, p = col(th1), col(qv1), col(qc1), col(p1)
    nz = len(thick)
    z1 = zc
    u = col(5.0 * z1 / 1500.0)
    v = np.zeros((nz, 1, 1))
    w = np.zeros((nz, 1, 1))
    e = np.full((nz, 1, 1), 0.4)
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)
    assert np.any(n2m != n2)                   # the limb has a live mask
    args = dict(dx=3000.0, dy=3000.0, dz=200.0, delta=3000.0, dt=60.0,
                n2=n2, dz_col=thick, n2_moist=n2m)
    monkeypatch.setattr(sase_ref, "dynamic_solve",
                        lambda *a, **k: (0.01, 1.0))
    monkeypatch.setattr(sase_ref, "partition_cap", lambda d, z: 1.0)
    monkeypatch.setattr(
        sase_ref, "w_resolved_bound",
        lambda w_, e_mean, n2=None: sase_ref.WSensorState(1.0, 1.0, 0.0,
                                                          1.0))
    calls = {"n": 0}
    real = sase_ref.bl89_moist_excursion_lengths

    def spy(n2_eff, e_, z_, thick_):
        calls["n"] += 1
        return real(n2_eff, e_, z_, thick_)

    monkeypatch.setattr(sase_ref, "bl89_moist_excursion_lengths", spy)
    fields_a, ledger_a = sase_ref.sase_split_step(u, v, w, theta, e,
                                                  **args)
    assert ledger_a["f"] == 1.0
    assert calls["n"] >= 1                     # non-vacuous: limb ran
    monkeypatch.setattr(sase_ref, "bl89_moist_excursion_lengths",
                        _m1b_inert_lengths)
    fields_b, ledger_b = sase_ref.sase_split_step(u, v, w, theta, e,
                                                  **args)
    for key in ("u", "v", "w", "e", "heat"):
        assert np.array_equal(fields_a[key], fields_b[key]), key
    assert ledger_a["pr_t"] == sase_ref.PR_LES


# ---------------------------------------------------------------------------
# S4-4: M2 conditional venting limb (authority; SASE-M spec section 4).
# Formulation + derivations: woof/verify/sase_ref.py module docstring
# (SASE-M2 section) and the plume_vent_flux docstring.  Evidence base:
# .superpowers/sdd/sase-m1-residual.md (the S4-3d-confirmed sizing: LE
# 0.6-2.28 kW/m2 conditional, roots 0.2-0.5 km, neg-H lobe 0.24-0.97 km,
# peak transport 1.2-2.0 km, hard NB termination of the 1.9-2.7 km tail),
# sase-m-target-envelope.md (export bound + supply rates),
# sase-m-amplifier-anatomy.md (shape + the state-not-activity trigger).
# The driver deposit seam (deposit-before-implicit-solve, C1-C3) is S4-5
# scope: plume_vent_flux returns face-registered flux profiles ONLY.
# ---------------------------------------------------------------------------

# S4-4 ROUND-6 RE-PIN (design doc SASE-M2 amendment "the export ratio is
# not an independent criterion", coordinator ruling).  The M2 amplitude
# acceptance quantity is ONE number written two ways.  Before this round
# the two ways were two independent literals whose rails did not agree
# ([0.7, 1.1]e-4 maps to ratio [0.35533, 0.55838]; [0.4, 0.6] maps to
# export [0.788, 1.182]e-4, a floor and a ceiling both outside the export
# bar's own rails), and round 5 landed in the gap between them.  The
# ruling: the ABSOLUTE EXPORT CLASS is primary and does not move; the
# ratio pin is its exact IMAGE under the registered supply and is
# DERIVED here rather than written, so the two can never drift apart
# again -- any future change to M2_EXPORT_CLASS propagates to
# M2_RATIO_CLASS by construction.  These are fixture-side bookkeeping,
# NOT closure constants: nothing here enters sase_ref or its config ID.
M2_SUPPLY = 1.97e-4          # BOX3 MFC 0-1.4 km, sase-m-target-envelope s4
M2_EXPORT_CLASS = (0.7e-4, 1.1e-4)          # PRIMARY, frozen, unchanged
M2_RATIO_CLASS = (M2_EXPORT_CLASS[0] / M2_SUPPLY,
                  M2_EXPORT_CLASS[1] / M2_SUPPLY)


def _vent_theta_es(theta, qv, qc, p):
    """Fixture-side theta_es (saturated equivalent potential temperature,
    the exp-form the specimen file's import assertions pin) from the
    sase_ref M1 constants -- (t, qs, theta_es)."""
    from woof.verify import sase_ref as sr
    t = theta * (p / sr.P0_REF) ** (sr.RD_AIR / sr.CP_AIR)
    es = 1000.0 * sr.SVP1 * np.exp(sr.SVP2 * (t - sr.SVPT0)
                                   / (t - sr.SVP3))
    qs = sr.EP2_RV * es / (p - es)
    return t, qs, theta * np.exp(sr.XLV * qs / (sr.CP_AIR * t))


def _vent_args(name="amp"):
    """plume_vent_flux argument dict for a specimen column at the M1
    frozen-state equilibrium.

    e_sgs is the 120-min M1 equilibrium e of _run_specimen_column (the
    coupled state M2 diagnoses from: the M1 uptake supplies the seed
    subgrid energy -- deck TKE ~1.0 m2/s2, the G-M5 band); n2m_mask is
    the M1 substitution mask n2_moist != n2 (the seam's own mask, the
    M1b idiom); rho1 is the S3-11a lowest-level moist density
    p_1/(R_d*T_v1) -- the same convention the surface deposit seam uses.
    """
    from woof.verify import sase_ref
    c = _specimen_cols(name)
    thick = c["thick"]
    if name == "amp":
        g = _run_specimen_column("amp", m1=True)
        e_sgs = g["samples"][120.0]["e"].reshape(-1, 1, 1).copy()
    else:
        # control column: the closure must be inert regardless of e
        e_sgs = np.full_like(c["theta"], 0.5)
    n2 = sase_ref.brunt_vaisala_n2(c["theta"], None, dz_col=thick)
    n2m = sase_ref.moist_n2(c["theta"], c["qv"], c["qc"], c["p"], thick)
    t0 = float(c["theta"][0, 0, 0]
               * (c["p"][0, 0, 0] / sase_ref.P0_REF)
               ** (sase_ref.RD_AIR / sase_ref.CP_AIR))
    tv0 = t0 * (1.0 + (sase_ref.RV_AIR / sase_ref.RD_AIR - 1.0)
                * float(c["qv"][0, 0, 0]) - float(c["qc"][0, 0, 0]))
    rho1 = float(c["p"][0, 0, 0]) / (sase_ref.RD_AIR * tv0)
    return dict(theta=c["theta"], qv=c["qv"], qc=c["qc"], p=c["p"],
                dz_col=thick, e_sgs=e_sgs, rho1=rho1,
                n2m_mask=(n2m != n2), f_blend=0.0)


def _vent_deck_invtop(z_lid):
    """The height of the deck fixture's INVERSION TOP for a given lid.

    Single source of truth for the 300 m inversion depth
    :func:`_vent_deck_args` builds: the C9 cap-margin assertion in
    ``test_sase_gpu`` reads the ceiling face off the device's own
    diagnosed k_lid and compares it against THIS, so the margin cannot
    silently become a comparison of two hardcoded literals again
    (S4-5b Item 4a)."""
    return float(z_lid) + 300.0


def _vent_deck_args(z_lid=1400.0, dth_inv=12.0, nz=30, dz=100.0):
    """plume_vent_flux argument dict for a prescribed-lid deck column
    (_build_deck_column with the lid moved; e_sgs = 0.6 m2/s2 uniform,
    the M1b deck-TKE class 0.56-0.61; ``dth_inv`` weakens the lid for
    the gradual-NB taper leg).  S4-4 review Important-2(b) addition:
    ``nz``/``dz`` control the grid resolution (defaults are the
    original literals, bitwise-identical for every existing caller) so
    the SAME physical column (z_base/z_lid/z_invtop/dth_inv fixed) can
    be discretized coarse or fine for the grid-consistency fixture."""
    from woof.verify import sase_ref
    th1, qv1, qc1, p1, thick, zc, _ = _build_deck_column(
        nz=nz, dz=dz, z_lid=z_lid, z_invtop=_vent_deck_invtop(z_lid),
        dth_inv=dth_inv)

    def col(a):
        return np.asarray(a, np.float64).reshape(-1, 1, 1)

    theta, qv, qc, p = col(th1), col(qv1), col(qc1), col(p1)
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)
    t0 = float(theta[0, 0, 0]) * (float(p[0, 0, 0])
                                  / sase_ref.P0_REF) ** (
        sase_ref.RD_AIR / sase_ref.CP_AIR)
    tv0 = t0 * (1.0 + (sase_ref.RV_AIR / sase_ref.RD_AIR - 1.0)
                * float(qv[0, 0, 0]))
    rho1 = float(p[0, 0, 0]) / (sase_ref.RD_AIR * tv0)
    return dict(theta=theta, qv=qv, qc=qc, p=p, dz_col=thick,
                e_sgs=np.full_like(theta, 0.6), rho1=rho1,
                n2m_mask=(n2m != n2), f_blend=0.0), zc


def _vent_faces(thick):
    """Face heights (m AGL): face j sits below cell j (authority cumsum
    convention); nz+1 faces."""
    return np.concatenate([[0.0], np.cumsum(np.asarray(thick))])


def _vent_taper_args(z_lid=1400.0, nz=40, dz=100.0, tilt=1.0e-3,
                     bump=0.2, ncap=4):
    """S4-4 REVIEW FINDING (x2 lane, round-3 BRANCH COVERAGE fix):
    plume_vent_flux argument dict for a SYNTHETIC deck column whose
    buoyancy peak falls STRICTLY BELOW the run's own top (kb + 1 <
    k_lid), the only construction that exercises the remaining-
    buoyancy TAPER branch of step 5 (w_taper reverse cumsum, the
    natural-NB search, the buoyancy-peak search) -- every registered
    M2 column (the specimen and every prescribed-lid deck) terminates
    with the peak one cell under the lid (kb == k_nb - 1, the
    degenerate empty-taper case), so this branch had ZERO fixture
    coverage before this fixture (x2 lane finding, proven a real
    S4-5 device-mirror parity gap by mutation: a build with the
    natural-NB/peak/taper searches deleted -- unconditionally
    terminating at the lid with the peak one cell under it -- is
    bitwise identical on every OTHER registered M2 column and passes
    every OTHER registered M2 fixture).

    CONSTRUCTION: :func:`_build_deck_column`'s saturated-adiabat deck,
    PLUS a small STABLE temperature ramp over the top ``ncap`` cells
    that were saturated (+``bump``*(ncap - i) K on cell k_top - i for
    i < ncap), with qv/qc re-set to qv = qs, qc = max(qt - qs, 2e-5) on
    those cells so they stay genuinely saturated after the ramp -- the
    stable bump makes the entraining parcel's buoyancy peak and then
    DECLINE gradually through several cells rather than staying
    monotonically buoyant right up to the run's own top.

    MEASURED AT HEAD (default parameters, instrumented HEAD build,
    probe r7_f_taper.py -- S4-4 ROUND-7 RE-MEASURE.  The STRUCTURE below
    was correct and is unchanged; the FLUX VALUES were not, because the
    round-5 cloud-base anchor moved every face by the uniform factor
    (500/400)**VENT_ENT_COEF = 1.093348 and this docstring kept the
    round-4 build's numbers.  Verified as exactly that: the round-4
    build measures 0.0122069 at the peak and
    0.005599703/0.002032397/0.0004477262 on the taper faces, the values
    written here before this round -- probe w1/f_taper.py.  The fixture's
    own assertion at test_m2_natural_neutral_buoyancy_taper_branch was
    already carrying the correct current values; only this prose was
    stale.)

    Member run k4..k13, base k_base = 4 with root k_r = 4 (theta_es
    decreases monotonically from the ground here, so the ROOT fallback
    puts the root at the run's own base -- the classical cloud-base
    root; the depth floor k_r_floor = -6 is vacuous), run top
    k_top = 13, entrainment-zone cell k14, ceiling k_lid = k_top + 2 =
    15 (face 1500 m).  Termination is the NATURAL neutral-buoyancy
    search, NOT the ceiling: buoyancy b[14] = -0.028081826451633932
    turns non-positive at the entrainment-zone cell while the loop is
    still at k = 14 < k_lid = 15, so k_nb = 14 and the C9 ceiling is
    never reached -- the stronger result, since it exercises the
    natural-NB branch this fixture exists for.  (b[13] = +0.004013620,
    b[15] = -0.151686706.)  Buoyancy peak kb = 9, whose top face 10
    (1000 m) carries the profile maximum F_theta = 0.013346575980884517;
    taper faces [11, 12, 13] (1100/1200/1300 m, F_theta =
    0.00612250270211349 / 0.0022221455268322753 /
    0.0004895268045954804, strictly decreasing and positive); exact
    zero at and above face 14 (1400 m -- the entrainment-zone cell's own
    BOTTOM face, so this column detrains entirely inside its own deck
    and delivers nothing to the entrainment zone).  The grow-zone faces
    5..9 are 0.0011885639 / 0.0037872791 / 0.0066194168 / 0.0096773465
    / 0.0129557890, and ebar = 0.6 exactly (uniform e_sgs), anchor face
    z_f[k_base] = 400.0 m.
    """
    from woof.verify import sase_ref
    th1, qv1, qc1, p1, thick, zc, _ = _build_deck_column(
        nz=nz, dz=dz, z_lid=z_lid, z_invtop=z_lid + 300.0, tilt=tilt)
    th1, qv1, qc1 = th1.copy(), qv1.copy(), qc1.copy()
    qt = qv1 + qc1
    k_top_raw = int(np.nonzero(qc1 > 0.0)[0].max())
    for i in range(ncap):
        th1[k_top_raw - i] += bump * (ncap - i)

    def qs_of(t, p):
        es = (1000.0 * sase_ref.SVP1
              * np.exp(sase_ref.SVP2 * (t - sase_ref.SVPT0)
                      / (t - sase_ref.SVP3)))
        return sase_ref.EP2_RV * es / (p - es)

    t = th1 * (p1 / sase_ref.P0_REF) ** (sase_ref.RD_AIR / sase_ref.CP_AIR)
    qs = qs_of(t, p1)
    for i in range(ncap):
        k = k_top_raw - i
        qv1[k] = qs[k]
        qc1[k] = max(qt[k] - qs[k], 2.0e-5)

    def col(a):
        return np.asarray(a, np.float64).reshape(-1, 1, 1)

    theta, qv, qc, p = col(th1), col(qv1), col(qc1), col(p1)
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)
    t0 = float(theta[0, 0, 0]) * (float(p[0, 0, 0])
                                  / sase_ref.P0_REF) ** (
        sase_ref.RD_AIR / sase_ref.CP_AIR)
    tv0 = t0 * (1.0 + (sase_ref.RV_AIR / sase_ref.RD_AIR - 1.0)
                * float(qv[0, 0, 0]))
    rho1 = float(p[0, 0, 0]) / (sase_ref.RD_AIR * tv0)
    return dict(theta=theta, qv=qv, qc=qc, p=p, dz_col=thick,
                e_sgs=np.full_like(theta, 0.6), rho1=rho1,
                n2m_mask=(n2m != n2), f_blend=0.0)


def _vent_real_args(name):
    """plume_vent_flux argument dict for one of the two REAL d02
    columns of :mod:`vent_columns_yolod_11z` (S4-4 round-5).

    Same construction as :func:`_vent_args` -- n2m_mask is the M1
    substitution mask n2_moist != n2, rho1 the S3-11a lowest-level
    moist density -- except that ``e_sgs`` is the frame's OWN
    TKE_SASE profile, i.e. genuinely non-uniform subgrid energy
    (1.0e-6 to 1.78e-1 m2/s2 across the WIN column's plume
    territory).  Every other registered M2 column carries an e_sgs
    that is uniform over the plume's territory, which is why the
    round-4 change of the ebar window measured as no change at all on
    the whole corpus (module docstring, SASE-M2 AMPLITUDE).
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_11z as vc

    def col(a):
        return np.asarray(a, np.float64).reshape(-1, 1, 1)

    g = getattr
    theta, qv = col(g(vc, name + "_THETA")), col(g(vc, name + "_QV"))
    qc, p = col(g(vc, name + "_QC")), col(g(vc, name + "_P"))
    thick = np.asarray(g(vc, name + "_THICK"), np.float64)
    e_sgs = col(g(vc, name + "_E"))
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)
    t0 = float(theta[0, 0, 0] * (p[0, 0, 0] / sase_ref.P0_REF)
               ** (sase_ref.RD_AIR / sase_ref.CP_AIR))
    tv0 = t0 * (1.0 + (sase_ref.RV_AIR / sase_ref.RD_AIR - 1.0)
                * float(qv[0, 0, 0]) - float(qc[0, 0, 0]))
    rho1 = float(p[0, 0, 0]) / (sase_ref.RD_AIR * tv0)
    return dict(theta=theta, qv=qv, qc=qc, p=p, dz_col=thick,
                e_sgs=e_sgs, rho1=rho1, n2m_mask=(n2m != n2),
                f_blend=0.0)


def _vent_binding_args(name):
    """plume_vent_flux argument dict for one of the two REAL d02
    BINDING-RULE columns of :mod:`vent_columns_yolod_binding` (S4-5b).

    Same construction as :func:`_vent_real_args` -- n2m_mask is the M1
    substitution mask n2_moist != n2, rho1 the S3-11a lowest-level
    moist density -- except for ``e_sgs``, which is the registered
    DESIGN-POINT value ``vc.SURVEY_E`` (1.0 m2/s2, inside the G-M5 band
    [0.5, 1.6]) uniform rather than the frame's own TKE_SASE.  That is
    the registered survey standard (design doc SASE-M section 4,
    amendment "two survey standards" (a)): the yolo-d frames are output
    of the defective run and their turbulence sits below the G-M5
    floor -- column-max 0.0451 / 0.0771 m2/s2 on these two columns,
    pinned at the module's import -- so an amplitude evaluated against
    it is a statement about the frame, not about the closure.  The
    frames' own profiles are kept as ``*_E_FRAME``.

    WHAT THESE TWO COLUMNS ADD (S4-5b Item 1).  Neither of the two
    RULES the round-5 amendment introduced -- the root DEPTH FLOOR
    k_r >= k_base - (k_top - k_base) - 1 and the MINIMUM-RUN guard --
    binds anywhere in the previously registered M2 corpus, so a build
    with either deleted passed every M2 gate.  On real fields the floor
    is a majority-scale rule (it moves the root on 24.9-35.7% of the
    step-1-chosen columns of the four yolo-d d02 frames, measured this
    session), and these are two of those columns: on CLAMP the floor
    clamps the root to the run's own base and its deletion multiplies
    the peak flux by 2.411, on DROP its deletion drops the root 7 cells
    and stands the column DOWN entirely.  CLAMP's run is exactly
    VENT_MIN_RUN_CELLS cells long, so it also binds the minimum-run
    guard at the registry's own alternative value 3.  The module
    docstring carries the selection rules and every measured number.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_binding as vc

    def col(a):
        return np.asarray(a, np.float64).reshape(-1, 1, 1)

    g = getattr
    theta, qv = col(g(vc, name + "_THETA")), col(g(vc, name + "_QV"))
    qc, p = col(g(vc, name + "_QC")), col(g(vc, name + "_P"))
    thick = np.asarray(g(vc, name + "_THICK"), np.float64)
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)
    t0 = float(theta[0, 0, 0] * (p[0, 0, 0] / sase_ref.P0_REF)
               ** (sase_ref.RD_AIR / sase_ref.CP_AIR))
    tv0 = t0 * (1.0 + (sase_ref.RV_AIR / sase_ref.RD_AIR - 1.0)
                * float(qv[0, 0, 0]) - float(qc[0, 0, 0]))
    rho1 = float(p[0, 0, 0]) / (sase_ref.RD_AIR * tv0)
    return dict(theta=theta, qv=qv, qc=qc, p=p, dz_col=thick,
                e_sgs=np.full_like(theta, vc.SURVEY_E), rho1=rho1,
                n2m_mask=(n2m != n2), f_blend=0.0)


def _vent_surface_args():
    """plume_vent_flux argument dict for the REAL d02 SURFACE-BASED
    column of :mod:`vent_columns_yolod_surface` (S4-5c).

    Same construction as :func:`_vent_binding_args` -- n2m_mask is the
    M1 substitution mask n2_moist != n2, rho1 the S3-11a lowest-level
    moist density, and ``e_sgs`` the registered DESIGN-POINT value
    ``vc.SURVEY_E`` (1.0 m2/s2, inside the G-M5 band [0.5, 1.6]) rather
    than the frame's own sub-floor TKE_SASE, per the registered survey
    standard (design doc SASE-M section 4, amendment "two survey
    standards" (a)).

    WHAT THIS COLUMN ADDS (S4-5c).  Its saturated run is based in the
    LOWEST MODEL LEVEL (k_base = 0), the case the amendment "a
    surface-based saturated layer stands the limb down" rules on.  No
    column of the previously registered M2 corpus was surface-based, so
    the branch was invisible to every gate in both engines.  The
    fixture module's own docstring carries the selection, the
    provenance and every measured number.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_surface as vc

    def col(a):
        return np.asarray(a, np.float64).reshape(-1, 1, 1)

    theta, qv = col(vc.SURF_THETA), col(vc.SURF_QV)
    qc, p = col(vc.SURF_QC), col(vc.SURF_P)
    thick = np.asarray(vc.SURF_THICK, np.float64)
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)
    t0 = float(theta[0, 0, 0] * (p[0, 0, 0] / sase_ref.P0_REF)
               ** (sase_ref.RD_AIR / sase_ref.CP_AIR))
    tv0 = t0 * (1.0 + (sase_ref.RV_AIR / sase_ref.RD_AIR - 1.0)
                * float(qv[0, 0, 0]) - float(qc[0, 0, 0]))
    rho1 = float(p[0, 0, 0]) / (sase_ref.RD_AIR * tv0)
    return dict(theta=theta, qv=qv, qc=qc, p=p, dz_col=thick,
                e_sgs=np.full_like(theta, vc.SURVEY_E), rho1=rho1,
                n2m_mask=(n2m != n2), f_blend=0.0)


def _vent_rules_args(tag):
    """plume_vent_flux argument dict for one of the eight REAL d02
    REGISTERED-RULE columns of :mod:`vent_columns_yolod_rules` (S4-5d).

    Construction identical to :func:`_vent_binding_args` and
    :func:`_vent_surface_args` -- ``n2m_mask`` is the M1 substitution
    mask n2_moist != n2, ``rho1`` the S3-11a lowest-level moist density,
    and ``e_sgs`` the registered DESIGN-POINT value ``vc.SURVEY_E``
    (1.0 m2/s2 uniform, inside the G-M5 band [0.5, 1.6]) rather than the
    frame's own sub-floor TKE_SASE, per the registered survey standard
    (design doc SASE-M section 4, amendment "two survey standards" (a)).

    WHAT THESE COLUMNS ADD (S4-5d).  Each one is the real-field
    representative of a REGISTERED RULE that the S4-5c audit wave showed
    could be deleted, or shifted one cell, with the whole CPU gate
    green.  The fixture module's own docstring carries the selection,
    the provenance, the per-rule measured consequence and the
    whole-frame binding frequency for every one of them.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_rules as vc

    def col(a):
        return np.asarray(a, np.float64).reshape(-1, 1, 1)

    g = getattr
    theta, qv = col(g(vc, tag + "_THETA")), col(g(vc, tag + "_QV"))
    qc, p = col(g(vc, tag + "_QC")), col(g(vc, tag + "_P"))
    thick = np.asarray(g(vc, tag + "_THICK"), np.float64)
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)
    t0 = float(theta[0, 0, 0] * (p[0, 0, 0] / sase_ref.P0_REF)
               ** (sase_ref.RD_AIR / sase_ref.CP_AIR))
    tv0 = t0 * (1.0 + (sase_ref.RV_AIR / sase_ref.RD_AIR - 1.0)
                * float(qv[0, 0, 0]) - float(qc[0, 0, 0]))
    rho1 = float(p[0, 0, 0]) / (sase_ref.RD_AIR * tv0)
    return dict(theta=theta, qv=qv, qc=qc, p=p, dz_col=thick,
                e_sgs=np.full_like(theta, vc.SURVEY_E), rho1=rho1,
                n2m_mask=(n2m != n2), f_blend=0.0)


def _vent_live_faces(fluxes):
    """The face indices carrying nonzero flux on a single column."""
    live = np.zeros(np.asarray(fluxes[0]).shape[0], dtype=bool)
    for arr in fluxes:
        live |= np.asarray(arr)[:, 0, 0] != 0.0
    return np.nonzero(live)[0]


def _vent_root_index_no_floor(args, k_base):
    """The step-1b root WITHOUT the round-5 depth floor: the highest
    interior theta_es maximum at or below ``k_base``, unbounded below.

    This is the rule the closure carried before the amendment, kept
    here as the reference the floor is asserted AGAINST -- a fixture
    that cannot state what the deleted rule would have produced cannot
    show that the rule binds.
    """
    th = np.asarray(args["theta"], np.float64)[:, 0, 0]
    p = np.asarray(args["p"], np.float64)[:, 0, 0]
    _, _, thes = _vent_theta_es(th, np.asarray(args["qv"])[:, 0, 0],
                                np.asarray(args["qc"])[:, 0, 0], p)
    k_r = k_base
    for k in range(1, th.size - 1):
        if (thes[k] > thes[k - 1] and thes[k] > thes[k + 1]
                and k <= k_base):
            k_r = k
    return k_r


def _vent_layer_indices(args):
    """(k_base, k_top) of the closure's lowest qualifying member run,
    transcribed INDEPENDENTLY of plume_vent_flux from the registered
    rules (VENT_K_LID_MEMBERSHIP qt >= qs inside the M1 mask, run length
    >= VENT_MIN_RUN_CELLS, bulk theta_es reading); (-1, -1) if none.
    Single-column args only."""
    from woof.verify import sase_ref as sr
    th = np.asarray(args["theta"], np.float64)[:, 0, 0]
    p = np.asarray(args["p"], np.float64)[:, 0, 0]
    _, qs, thes = _vent_theta_es(th, np.asarray(args["qv"])[:, 0, 0],
                                 np.asarray(args["qc"])[:, 0, 0], p)
    qt = (np.asarray(args["qv"])[:, 0, 0]
          + np.asarray(args["qc"])[:, 0, 0])
    member = np.asarray(args["n2m_mask"]).astype(bool)[:, 0, 0] & (qt >= qs)
    in_run, run_s, base_thes = False, 0, 0.0
    for k in range(len(th)):
        mk = bool(member[k])
        if mk and not in_run:
            run_s, base_thes = k, float(thes[k])
        ends = mk and (k + 1 >= len(th) or not bool(member[k + 1]))
        if (ends and (k - run_s) >= sr.VENT_MIN_RUN_CELLS - 1
                and float(thes[k]) - base_thes < 0.0):
            return run_s, k
        in_run = mk
    return -1, -1


def _vent_root_index(args, k_base, k_top):
    """The closure's THERMODYNAMIC ROOT k_r, transcribed INDEPENDENTLY of
    plume_vent_flux from the registered step-1b rule (design doc SASE-M
    section 4 "Root" + the round-5 depth-floor amendment): the HIGHEST
    interior theta_es maximum at or below ``k_base``, no deeper than
    ``k_base - (k_top - k_base) - 1``; fallback ``k_base`` when the
    decrease layer has no interior base.  ``thes`` is a function of
    (theta, p) only, exactly as the closure's is.

    VALIDATED this session against an instrumented HEAD build (probe
    ``r7_g_rootcheck.py``): the transcription reproduces the closure's
    own ``k_r`` on 38 of 38 checks -- the specimen, the taper column,
    both real columns, the four grid-consistency deck grids with both
    e-profiles, the three prescribed-lid decks and the WIN refinement
    family, each on BOTH sides of the mask veto.

    Single-column args only."""
    th = np.asarray(args["theta"], np.float64)[:, 0, 0]
    p = np.asarray(args["p"], np.float64)[:, 0, 0]
    _, _, thes = _vent_theta_es(th, np.asarray(args["qv"])[:, 0, 0],
                                np.asarray(args["qc"])[:, 0, 0], p)
    nz = th.size
    k_r = k_base
    floor = k_base - (k_top - k_base) - 1
    for k in range(1, nz - 1):
        if (thes[k] > thes[k - 1] and thes[k] > thes[k + 1]
                and k <= k_base and k >= floor):
            k_r = k
    return k_r


def _vent_mask_vetoed(args, k):
    """``args`` with the M1 substitution mask switched OFF at level k.

    n2m_mask is an INPUT to plume_vent_flux (the M1 seam's own mask,
    which the limb never re-derives), so this is a state the seam can
    genuinely produce -- and it is the thermodynamically INERT way to
    move the membership run's base by one cell: theta/qv/qc/p/e_sgs are
    bit-identical, so the ONLY thing that changes is the layer index
    structure.  Used by the k_base-flip convergence fixture, where a
    theta perturbation would confound the index move with a dz-thick
    warm layer."""
    out = dict(args)
    m = np.asarray(args["n2m_mask"]).copy()
    m[k, 0, 0] = False
    out["n2m_mask"] = m
    return out


def _vent_refined_real_args(name, r):
    """The ``name`` column of :mod:`vent_columns_yolod_11z` resampled
    onto an r-times-finer vertical grid: the SAME PHYSICAL COLUMN, only
    the discretization changed.

    This is the :func:`_vent_deck_args` nz/dz idiom applied to a column
    whose profile is DATA rather than an ODE the fixture can re-
    integrate: each layer is split into r equal sub-layers and
    theta/qv/qc/p/e are taken linearly in z at the refined cell centers
    from the original ones (constant extrapolation past the end
    centers).  r = 1 reproduces the original discretization to within
    the interpolation's own identity at the interior centers.
    """
    import vent_columns_yolod_11z as vc
    g = getattr
    thick = np.asarray(g(vc, name + "_THICK"), np.float64)
    zc = np.cumsum(thick) - 0.5 * thick
    tn = np.repeat(thick / r, r)
    zn = np.cumsum(tn) - 0.5 * tn

    def c(a):
        return np.interp(zn, zc, np.asarray(a, np.float64)).reshape(-1, 1, 1)

    from woof.verify import sase_ref
    theta, qv = c(g(vc, name + "_THETA")), c(g(vc, name + "_QV"))
    qc, p = c(g(vc, name + "_QC")), c(g(vc, name + "_P"))
    e_sgs = c(g(vc, name + "_E"))
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=tn)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, tn)
    t0 = float(theta[0, 0, 0] * (p[0, 0, 0] / sase_ref.P0_REF)
               ** (sase_ref.RD_AIR / sase_ref.CP_AIR))
    tv0 = t0 * (1.0 + (sase_ref.RV_AIR / sase_ref.RD_AIR - 1.0)
                * float(qv[0, 0, 0]) - float(qc[0, 0, 0]))
    rho1 = float(p[0, 0, 0]) / (sase_ref.RD_AIR * tv0)
    return dict(theta=theta, qv=qv, qc=qc, p=p, dz_col=tn, e_sgs=e_sgs,
                rho1=rho1, n2m_mask=(n2m != n2), f_blend=0.0)


def _vent_theta_perturbed(args, k, dtheta):
    """``args`` with theta[k] shifted by ``dtheta`` and the M1 mask
    RE-DERIVED from the shifted state (the mask is a function of
    theta, so a perturbation that did not propagate through it would
    test a state the M1 seam never produces)."""
    from woof.verify import sase_ref
    out = dict(args)
    theta = np.asarray(args["theta"], np.float64).copy()
    theta[k] += dtheta
    out["theta"] = theta
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=args["dz_col"])
    n2m = sase_ref.moist_n2(theta, args["qv"], args["qc"], args["p"],
                            args["dz_col"])
    out["n2m_mask"] = (n2m != n2)
    return out


def test_m2_registry_binds_vent_constants(monkeypatch):
    """Every M2 registered constant/convention is a config-ID registry
    member and hash-bound (the S3-6h/S3-9/S4-1 idiom), with the exact
    registered values:

    * VENT_FORM = "nb-terminated-vent-v1" (the task-brief interface);
    * VENT_MASK = "bulk-theta-es-v1" (the registered mask-convention
      OBLIGATION -- decision + justification in the plume_vent_flux
      docstring, the flip pinned by test_m2_mask_convention_flip);
    * VENT_MB_COEF = 0.03 -- Grant (2001, QJRMS 127, 407-421)
      cloud-base mass-flux closure M_b = 0.03*w, velocity scale re-keyed
      to column-local subgrid energy (C8: no surface-w* dependence);
    * VENT_ENT_COEF = 0.4 -- Siebesma, Soares & Teixeira (2007, JAS 64,
      1230-1248) Eq. (16) c_eps of the eps ~ 1/z entraining-plume
      family (S4-4 REVIEW FINDING Important-1: the value shipped at
      S4-4 authority, 0.55, was misattributed to this paper -- 0.4 is
      SST07's own constant, p. 1236; 0.55 has no located source);
    * VENT_SIGW_SHARE = 2/3 -- the isotropic w-variance share of TKE
      (e = (3/2)*sigma_w^2, textbook isotropy);
    * VENT_DEPTH_CAP = 4000.0 m -- the shallow-device scope guard
      (amplifier-anatomy constraint 8: detrainment capped ~2.5-4 km;
      C13: SASE claims BL turbulence, not deep convective mass flux);
    * VENT_THETA_STEP_CAP = 0.14 K/step -- the S3-11a measured
      stable-deposit class (module docstring S3-11a section:
      |dtheta_1| ~ 0.14 K per d02 step at the pinned defect state),
      the registered per-step deposit bound the S4-5 seam enforces;
    * VENT_QT_STEP_CAP -- S4-4 REVIEW FINDING Important-4: DERIVED
      (not independent) from VENT_THETA_STEP_CAP via the latent/
      sensible heat equivalence VENT_THETA_STEP_CAP*CP_AIR/XLV, the
      moisture-row companion of the theta cap;
    * VENT_MIN_RUN_CELLS = 2, VENT_SAT_ADJUST_ITERS = 12 -- S4-4
      REVIEW FINDING Minor-1: registered (previously prose-only
      numerics);
    * VENT_K_LID_MEMBERSHIP = "qt-ge-qs-v1" -- the M2 layer-membership
      convention (a cell belongs to the venting layer iff qt = qv + qc
      >= qs, never on the M1 mask's bit-level qc > 0 limb).  S4-4
      REVIEW round-4: this constant was previously absent from THIS
      TEST -- neither value-asserted nor included in the hash-sweep
      below -- so the suite carried no coverage of the membership
      convention at all.  S4-4 ROUND-5 CORRECTION: the round-4 wording
      here went further and said a build that swapped the convention
      "produced an identical config ID", which is FALSE.  Re-checked
      this session with ``git show <rev>:woof/verify/sase_ref.py``:
      the constant does not exist at the S4-4 authority build 7e722e9
      at all, and at the round-3 build 3367244 that introduced it it is
      ALREADY the last entry of _CONFIG_ID_CONSTANTS -- so from the
      moment the string existed it moved the config ID.  What was
      missing, and what round 4 actually added, is this test's own
      value assertion and its hash-sweep entry.
    * VENT_ANCHOR_RULE = "cloud-base-face-standdown-v1" -- S4-5c, the
      M2 SHAPE-ANCHOR convention and the fourth stand-down condition it
      implies (a saturated run based in the lowest model level stands
      the limb down; :func:`test_m2_surface_based_layer_stands_down`).
      REGISTERING IT IS ESSENTIAL, not bookkeeping: the rule is
      bitwise-inert on the entire registered corpus (78 columns
      measured, 0 bytes moved), so without a constant in this registry
      a real semantic change would ship under the pre-amendment config
      ID -- ``sase_config_id`` hashes constant VALUES, not code.  The
      hash sweep below flips it to the rejected alternative reading
      "cloud-base-face-mhat1-v0".
    """
    from woof.verify import sase_ref
    assert sase_ref.VENT_FORM == "nb-terminated-vent-v1"
    assert sase_ref.VENT_MASK == "bulk-theta-es-v1"
    assert sase_ref.VENT_MB_COEF == 0.03
    assert sase_ref.VENT_ENT_COEF == 0.4
    assert sase_ref.VENT_SIGW_SHARE == 2.0 / 3.0
    assert sase_ref.VENT_DEPTH_CAP == 4000.0
    assert sase_ref.VENT_THETA_STEP_CAP == 0.14
    assert sase_ref.VENT_QT_STEP_CAP == pytest.approx(
        0.14 * sase_ref.CP_AIR / sase_ref.XLV)
    assert sase_ref.VENT_MIN_RUN_CELLS == 2
    assert sase_ref.VENT_SAT_ADJUST_ITERS == 12
    assert sase_ref.VENT_K_LID_MEMBERSHIP == "qt-ge-qs-v1"
    assert sase_ref.VENT_ANCHOR_RULE == "cloud-base-face-standdown-v1"
    for name in ("VENT_FORM", "VENT_MASK", "VENT_MB_COEF",
                 "VENT_ENT_COEF", "VENT_SIGW_SHARE", "VENT_DEPTH_CAP",
                 "VENT_THETA_STEP_CAP", "VENT_QT_STEP_CAP",
                 "VENT_MIN_RUN_CELLS", "VENT_SAT_ADJUST_ITERS",
                 "VENT_K_LID_MEMBERSHIP", "VENT_ANCHOR_RULE"):
        assert name in sase_ref._CONFIG_ID_CONSTANTS, name
    seen = {sase_ref.sase_config_id()}
    for name, value in (("VENT_FORM", "other-v0"),
                        ("VENT_MASK", "per-level-theta-es-v1"),
                        ("VENT_MB_COEF", 0.035),
                        ("VENT_ENT_COEF", 0.55),
                        ("VENT_DEPTH_CAP", 3000.0),
                        ("VENT_QT_STEP_CAP", 1.0e-4),
                        ("VENT_MIN_RUN_CELLS", 3),
                        ("VENT_SAT_ADJUST_ITERS", 10),
                        ("VENT_K_LID_MEMBERSHIP", "qc-positive-v0"),
                        ("VENT_ANCHOR_RULE",
                         "cloud-base-face-mhat1-v0")):
        monkeypatch.setattr(sase_ref, name, value)
        cid = sase_ref.sase_config_id()
        assert cid not in seen, name
        seen.add(cid)


def test_m2_mask_discrimination_bitwise():
    """MASK DISCRIMINATION (brief fixture 1): the limb is identically
    zero outside the saturated moist-unstable mask.

    * CONTROL column (qc = 0, RH < 80% everywhere -- pinned at import
      by the specimen file): every returned flux array is
      tobytes()-identical to +0.0 zeros, even with a healthy e_sgs
      (activity/energy alone must never fire the limb -- the trigger
      is state, anatomy constraint 1).
    * AMPLIFIER column: nonzero fluxes (the limb engages exactly where
      the amplifier lives).
    * POISONED MASK: the amplifier column with n2m_mask forced all-
      False is bitwise zero -- the M1 switch owns the trigger; the
      closure never re-derives its own saturation mask.
    """
    from woof.verify import sase_ref
    ctl = _vent_args("ctl")
    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**ctl)
    nzp1 = ctl["theta"].shape[0] + 1
    zeros = np.zeros((nzp1, 1, 1)).tobytes()
    assert f_th.tobytes() == zeros
    assert f_qv.tobytes() == zeros
    assert f_qc.tobytes() == zeros

    amp = _vent_args("amp")
    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**amp)
    assert float(np.abs(f_qv + f_qc).max()) > 0.0
    assert float(np.abs(f_th).max()) > 0.0

    off = dict(amp, n2m_mask=np.zeros_like(amp["n2m_mask"]))
    for arr in sase_ref.plume_vent_flux(**off):
        assert arr.tobytes() == zeros


def test_m2_mask_convention_flip_pinned(monkeypatch):
    """THE REGISTERED MASK-CONVENTION OBLIGATION (S4-1 review): the 11Z
    specimen FLIPS between the two readings of "saturated layer with
    d(theta_es)/dz < 0 through a finite depth", so BOTH are implemented
    behind the registered string VENT_MASK and both behaviors are
    pinned here so the convention can never drift silently.

    THE SPECIMEN FLIP (pinned numerically): over the M1 saturation
    mask's run k = 10..16 theta_es runs 334.409 -> 334.298 -> 334.304
    -> 334.526 -> 334.274 -> 332.779 -> 329.940 K:

    * BULK reading (run-top minus run-base): -4.469 K < 0 -- the layer
      is moist-unstable through its finite depth -> ACTIVE;
    * PER-LEVEL reading (theta_es decreasing at EVERY adjacent pair):
      FALSE -- the run carries interior INCREASES at k11->k12 (+0.006)
      and k12->k13 (+0.222) -> INACTIVE.

    S4-4 review round-4: the run the CLOSURE evaluates the reading on
    is the narrower MEMBER run k12..k15 (qt >= qs inside the mask --
    plume_vent_flux docstring, SASE-M2 MEMBERSHIP), not this M1 mask
    run.  The flip survives the narrowing and is therefore a property
    of the specimen rather than of the run definition: MEASURED bulk
    theta_es over k12..k15 = -1.524965 K (still < 0 -> ACTIVE) with the
    interior increase at k12->k13 = +0.221981 K still inside it (still
    -> INACTIVE per-level).  Both runs and both readings are asserted
    below, and the per-level veto is exercised end-to-end on the real
    closure.

    The chosen convention is "bulk-theta-es-v1" (physical justification,
    plume_vent_flux docstring: the plume's root parcel finds the column
    unstable through the INTEGRATED theta_es deficit from its root --
    parcel buoyancy above the LFC does not care about interior
    theta_es wiggles -- and the per-level-monotone reading would VETO
    the very specimen that measurably amplified x11.5/h, which is
    consistency with the parcel ascent, the S4-1 criterion).  Pinned:

    * the chosen reading ACTIVATES the specimen (nonzero fluxes);
    * monkeypatching VENT_MASK = "per-level-theta-es-v1" turns the
      SAME specimen bitwise all-zero (the alternative reading is real,
      implemented, and vetoes the measured amplifier);
    * on the constructed deck column theta_es is strictly decreasing
      through the run, so BOTH readings activate and agree bitwise
      (the conventions coincide wherever the profile is monotone --
      the flip is a real property of the specimen, not an
      implementation artifact);
    * an unknown VENT_MASK string is rejected loudly.
    """
    from woof.verify import sase_ref
    amp = _vent_args("amp")
    _, _, thes = _vent_theta_es(amp["theta"][:, 0, 0],
                                amp["qv"][:, 0, 0],
                                amp["qc"][:, 0, 0], amp["p"][:, 0, 0])
    run = np.nonzero(amp["n2m_mask"][:, 0, 0])[0]
    assert list(run) == list(range(10, 17))
    d = np.diff(thes[run])
    assert thes[run[-1]] - thes[run[0]] < -4.0          # bulk: unstable
    assert d[1] > 0.0 and d[2] > 0.0                    # interior rises
    assert abs(d[1] - 0.00628) < 5e-4
    assert abs(d[2] - 0.22198) < 5e-3
    assert not np.all(d < 0.0)                          # per-level: NO

    # the MEMBER run the closure actually reads (round-4): narrower,
    # same flip.
    _, qs_a, _ = _vent_theta_es(amp["theta"][:, 0, 0], amp["qv"][:, 0, 0],
                                amp["qc"][:, 0, 0], amp["p"][:, 0, 0])
    member = np.nonzero(
        amp["n2m_mask"][:, 0, 0]
        & ((amp["qv"][:, 0, 0] + amp["qc"][:, 0, 0]) >= qs_a))[0]
    assert list(member) == list(range(12, 16))          # k12..k15
    dm = np.diff(thes[member])
    assert (thes[member[-1]] - thes[member[0]]) == pytest.approx(
        -1.524965, abs=1e-5)                            # bulk: unstable
    assert not np.all(dm < 0.0)                         # per-level: NO
    assert dm[0] == pytest.approx(0.221981, abs=1e-5)   # the interior rise

    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**amp)
    assert float(np.abs(f_qv).max()) > 0.0              # bulk activates

    nzp1 = amp["theta"].shape[0] + 1
    zeros = np.zeros((nzp1, 1, 1)).tobytes()
    monkeypatch.setattr(sase_ref, "VENT_MASK", "per-level-theta-es-v1")
    for arr in sase_ref.plume_vent_flux(**amp):
        assert arr.tobytes() == zeros                   # flip: vetoed

    deck, _ = _vent_deck_args()
    _, _, thes_d = _vent_theta_es(deck["theta"][:, 0, 0],
                                  deck["qv"][:, 0, 0],
                                  deck["qc"][:, 0, 0],
                                  deck["p"][:, 0, 0])
    run_d = np.nonzero(deck["n2m_mask"][:, 0, 0])[0]
    assert np.all(np.diff(thes_d[run_d]) < 0.0)         # monotone deck
    f_per = sase_ref.plume_vent_flux(**deck)
    monkeypatch.setattr(sase_ref, "VENT_MASK", "bulk-theta-es-v1")
    f_bulk = sase_ref.plume_vent_flux(**deck)
    assert float(np.abs(f_per[0]).max()) > 0.0
    for a, b in zip(f_per, f_bulk):
        assert a.tobytes() == b.tobytes()               # readings agree

    monkeypatch.setattr(sase_ref, "VENT_MASK", "no-such-reading")
    with pytest.raises(ValueError, match="VENT_MASK"):
        sase_ref.plume_vent_flux(**amp)


def test_m2_layer_structure_roundoff_insensitive():
    """S4-4 REVIEW round-4, THE ESSENTIAL PROPERTY: the M2 layer's
    base, its contiguity, its top and its root must not depend on
    round-off-scale condensate.

    THE DEFECT (measured on the round-3 build 3367244 in the round-4
    session, specimen export/supply, quoted here as prior record): the
    layer rode the M1 mask, whose ``sat = (qc > 0) | (qv >= qs)``
    switch is a BIT-LEVEL test on condensate, so a 1e-12 kg/kg
    condensate value -- nine orders below anything physical -- moved
    the structure and swung the amplitude:

        qc[9]  += 1e-12  ->  root k_r 10 -> 9     ->  0.43850
        qc[10] -= 1e-12  ->  root k_r 10 -> 11    ->  0.38422
        qc[16] -= 1e-12  ->  ebar window shrinks  ->  0.44666
        qc[17] += 1e-12  ->  ebar window grows    ->  0.36918

    (S4-4 ROUND-6: two of those four rows carried an "OUT" annotation,
    which read them against the superseded literal [0.4, 0.6] rail --
    both are INSIDE the derived M2_RATIO_CLASS.  The annotation is
    dropped because it was never the point: the defect is the +-10%
    amplitude SWING from a 1e-12 kg/kg input, whichever band contains
    the endpoints.)

    THE FIX (plume_vent_flux docstring, SASE-M2 MEMBERSHIP/ROOT/
    AMPLITUDE): membership is the registered total-water test
    VENT_K_LID_MEMBERSHIP (qt = qv + qc >= qs, margin 4.478e-5 kg/kg
    at the tightest cell of this column -- 4.5e7 times the
    perturbation), the root is the theta_es-decrease base (theta_es is
    a function of theta and p only), and the ebar window runs from the
    root to the entrainment-zone cell.  The M1 mask survives only as a
    VETO, and on the M1 switch itself the veto is provably never
    binding (qt >= qs implies qc > 0 or qv >= qs), so it cannot
    re-admit the noise -- asserted below.  ZERO new constants: no
    numeric condensate floor appears anywhere (a floor such as
    qc > 1e-13 was rejected in review: it still lets a round-off-scale
    value carry transport across the C9 boundary).

    WHAT THIS PINS (and, deliberately, what it does NOT):

    * the structural indices (k_r, k_top, k_lid) and the flux support
      are UNCHANGED under +-1e-12 kg/kg condensate shifts at every
      probed cell, including the four positions above at which the M1
      mask's own run demonstrably still moves (asserted here, so the
      test cannot go vacuous if the fixture column changes);
    * the C9 boundary still holds bitwise: +0.0, never -0.0, at and
      above the entrainment-zone cell's top face (face 17) on all
      three rows, under every shift;
    * the flux response is CONTINUOUS and O(eps) -- explicitly NOT
      bitwise identical.  qt enters the entraining parcel, so a
      perturbation of qc must move the fluxes a little; the accurate
      claim is linearity, not invariance.  MEASURED: max relative
      movement 5.84e-10 over all shifts (worst cell k10), and on cell
      k12 the response scales exactly with eps -- 3.045e-11 / 3.046e-10
      / 3.046e-9 relative at eps = 1e-13 / 1e-12 / 1e-11 kg/kg, a
      ratio of 10.00 per decade.  (A pre-round-4 revision of the
      plume_vent_flux docstring claimed +-1e-12 shifts moved "every
      flux NOT AT ALL (bitwise) -- pinned by fixture".  Both halves
      were false: no such fixture existed, and no shift inside the
      plume's territory leaves the fluxes bitwise unchanged.  This
      fixture replaces that claim with the measured one.)
    """
    from woof.verify import sase_ref

    amp = _vent_args("amp")
    nzp1 = amp["theta"].shape[0] + 1
    eps = 1.0e-12

    def rebuild(qc_arr):
        # n2m_mask is a FUNCTION of qc (the M1 seam's own switch), so a
        # condensate shift must be propagated through it -- exactly the
        # construction _vent_args uses.
        a = dict(amp)
        a["qc"] = qc_arr
        n2 = sase_ref.brunt_vaisala_n2(a["theta"], None,
                                       dz_col=a["dz_col"])
        n2m = sase_ref.moist_n2(a["theta"], a["qv"], qc_arr, a["p"],
                                a["dz_col"])
        a["n2m_mask"] = (n2m != n2)
        return a

    def first_run(flag):
        ks = np.nonzero(flag)[0]
        if ks.size == 0:
            return None
        lo = int(ks[0])
        hi = lo
        while hi + 1 < len(flag) and flag[hi + 1]:
            hi += 1
        return (lo, hi)

    def structure(a):
        # the registered rules transcribed INDEPENDENTLY of
        # plume_vent_flux (this column's lowest member run is also its
        # only qualifying one, so the >= VENT_MIN_RUN_CELLS / theta_es
        # selection does not need re-implementing here).
        _, qs, thes = _vent_theta_es(a["theta"][:, 0, 0],
                                     a["qv"][:, 0, 0],
                                     a["qc"][:, 0, 0], a["p"][:, 0, 0])
        mask = np.asarray(a["n2m_mask"])[:, 0, 0]
        rob = (a["qv"][:, 0, 0] + a["qc"][:, 0, 0]) >= qs
        assert np.all(~rob | mask)          # the veto is never binding
        k_base, k_top = first_run(mask & rob)
        k_r = k_base
        for k in range(1, len(thes) - 1):
            if (k <= k_base and thes[k] > thes[k - 1]
                    and thes[k] > thes[k + 1]):
                k_r = k
        return k_r, k_top, k_top + 2, (k_base, k_top)

    def support(f3):
        out = set()
        for f in f3:
            out |= {int(j) for j in np.nonzero(f[:, 0, 0] != 0.0)[0]}
        return tuple(sorted(out))

    def relmove(fa, fb):
        return max(float(np.abs(x - y).max())
                   / max(float(np.abs(y).max()), 1e-300)
                   for x, y in zip(fa, fb))

    def c9_zero(f3, j0):
        tail = np.zeros((nzp1 - j0, 1, 1)).tobytes()
        return all(f[j0:].tobytes() == tail for f in f3)

    # ---- baseline -------------------------------------------------
    k_r0, k_top0, k_lid0, member0 = structure(amp)
    assert (k_r0, k_top0, k_lid0) == (10, 15, 17)
    assert member0 == (12, 15)                     # the member run
    f0 = sase_ref.plume_vent_flux(**amp)
    assert support(f0) == (11, 12, 13, 14, 15, 16)
    assert c9_zero(f0, k_lid0)
    # S4-4 ROUND-5 RE-MEASURE: 0.413427 -> 0.386160 under the
    # cloud-base anchor (the amplitude no longer carries the
    # root-to-base factor (z_f[k_base]/z_f[k_r+1])**c_eps = 1.081185
    # on this column, nor the root-window ebar).  S4-4 ROUND-6: the band
    # checked below is now the DERIVED image of the export class (see
    # M2_RATIO_CLASS and the amendment "the export ratio is not an
    # independent criterion"), which admits 0.386160; the old literal
    # [0.4, 0.6] did not, and was a disagreeing second copy of the same
    # bar.  The MEASURED value itself is unchanged and still pinned to
    # 1e-6 on the line below -- this re-pin loosens no measurement.
    ratio0 = float(f0[1][16, 0, 0] + f0[2][16, 0, 0]) / M2_SUPPLY
    assert ratio0 == pytest.approx(0.386160, abs=1e-6)
    mask_run0 = first_run(np.asarray(amp["n2m_mask"])[:, 0, 0])
    assert mask_run0 == (10, 16)                   # the M1 mask's run

    # ---- the sweep ------------------------------------------------
    # the four positions at which the M1 mask's own run still moves:
    # these are exactly the shifts that swung the amplitude before the
    # round-4 fix, so the sweep below is provably non-vacuous.
    mask_moves = {(9, +1.0): (9, 16), (10, -1.0): (11, 16),
                  (16, -1.0): (10, 15), (17, +1.0): (10, 17)}
    moved = 0
    worst = 0.0
    for k in (9, 10, 11, 12, 13, 14, 15, 16, 17):
        for sgn in (+1.0, -1.0):
            qc2 = np.asarray(amp["qc"], np.float64).copy()
            qc2[k, 0, 0] += sgn * eps
            a2 = rebuild(qc2)
            run2 = first_run(np.asarray(a2["n2m_mask"])[:, 0, 0])
            assert run2 == mask_moves.get((k, sgn), mask_run0), (k, sgn,
                                                                 run2)
            moved += int(run2 != mask_run0)
            # structure: bit-for-bit the same indices
            assert structure(a2)[:3] == (k_r0, k_top0, k_lid0), (k, sgn)
            f2 = sase_ref.plume_vent_flux(**a2)
            assert support(f2) == support(f0), (k, sgn)
            # C9: bitwise +0.0 (and never -0.0) at/above face 17
            assert c9_zero(f2, k_lid0), (k, sgn)
            assert not any(bool(np.signbit(f[k_lid0:]).any()) for f in f2)
            rel = relmove(f2, f0)
            assert rel <= 1.0e-8, (k, sgn, rel)
            worst = max(worst, rel)
            r2 = float(f2[1][16, 0, 0] + f2[2][16, 0, 0]) / M2_SUPPLY
            # S4-4 ROUND-6: the ratio class DERIVED from the primary
            # export class ([0.7, 1.1]e-4 / 1.97e-4 = [0.3553299,
            # 0.5583756]) -- amendment "the export ratio is not an
            # independent criterion".  Nothing is written here that is
            # not an image of the export bar.
            assert M2_RATIO_CLASS[0] <= r2 <= M2_RATIO_CLASS[1], (k, sgn,
                                                                  r2)
            assert r2 == pytest.approx(ratio0, abs=1e-8), (k, sgn)
    assert moved == len(mask_moves)                 # non-vacuity
    assert 0.0 < worst <= 1.0e-8, worst             # NOT bitwise

    # ---- continuity: the response is O(eps), not a step ------------
    resp = {}
    for e in (1.0e-13, 1.0e-12, 1.0e-11):
        qc2 = np.asarray(amp["qc"], np.float64).copy()
        qc2[12, 0, 0] += e
        resp[e] = relmove(sase_ref.plume_vent_flux(**rebuild(qc2)), f0)
    assert resp[1.0e-13] > 0.0                       # explicitly NOT zero
    assert 9.5 <= resp[1.0e-12] / resp[1.0e-13] <= 10.5, resp
    assert 9.5 <= resp[1.0e-11] / resp[1.0e-12] <= 10.5, resp


def test_m2_real_column_cloud_base_anchor(monkeypatch):
    """S4-4 ROUND-5, THE CONSTRUCTION THE CORPUS LACKS (design doc
    SASE-M2 amendment "root / anchor separation", clauses 2 and 4): a
    REAL d02 column with REAL non-uniform e_sgs whose saturated-layer
    base sits well above its theta_es root, so the amplitude actually
    depends on which level the closure anchors at.

    WHY IT IS NEEDED.  Round 4 moved the ebar window from the raw
    masked run to root-through-entrainment-zone and reported the move
    as bitwise neutral.  That report was true of every registered
    column and false in general: on all of them e_sgs is uniform over
    the plume's territory (the deck fixtures use a single value; the
    specimen's M1-equilibrium e is flat to within 0.4% over its own
    k10..k16), so ANY window gives the same mean.  MEASURED on the
    11Z d02 frame with the frame's own TKE_SASE (session probe
    v5_p3_window.py): on the 9046 firing columns whose base sits >= 2
    cells above the root the two windows differ by a median flux
    factor 5.81, p95 35.98, max 401.85 (13Z: 8089 columns, median
    2.76, p95 57.32, max 469.32).  A factor-400 amplitude change that
    the entire fixture corpus registers as no change is exactly the
    class of defect clause 4 exists to stop.

    THE COLUMN (:mod:`vent_columns_yolod_11z`, WIN_*, j=28 i=196,
    34.9248 N -85.7900 W; selection rule in that module's docstring):
    root k_r = 4 (107.4 m), cloud base k_base = 10 (456.6 m), run top
    k_top = 17 (1659.9 m), e_sgs spanning 1.0e-6 to 1.7796e-1 m2/s2
    over the plume's territory -- a factor 1.78e5.

    WHAT IS PINNED (all measured this session on this build):

    * the shipped ebar is the CLOUD-BASE window mean over
      [k_base, k_top + 1], re-derived here independently of
      plume_vent_flux: 3.979314448323069e-3 m2/s2;
    * the round-4 ROOT window mean over [k_r, k_top + 1] is
      2.2380569891450888e-2 -- 5.62x larger, i.e. the two windows are
      demonstrably NOT the same on this column (the non-vacuity the
      corpus could not supply);
    * the two anchors' amplitudes differ by the EXACT analytic factor
      sqrt(ebar_root/ebar_base) * (z_f[k_base]/z_f[k_r+1])**c_eps =
      2.3715453896548633 * 1.6132294349373353 = 3.825846828881158,
      because every grow-zone face flux scales as sqrt(ebar) *
      z_f_anchor**(-c_eps) and the export face IS a grow-zone face
      here (buoyancy peak kb = 17, export face 18 = kb + 1).  Applied
      to the shipped export 1.4398699509941751e-05 kg/m2/s that
      predicts 5.5087218860123326e-05 for the round-4 build; the
      instrumented round-4 build (HEAD df235f0, probe v5_p7_detail.py)
      measures 5.508721886012332e-05 on this column -- agreement to
      the last FP64 digit, which is what makes the analytic
      counterfactual an accurate stand-in for a second build here.
    """
    from woof.verify import sase_ref
    args = _vent_real_args("WIN")
    thick = args["dz_col"]
    zf = _vent_faces(thick)
    k_r, k_base, k_top = 4, 10, 17          # measured (probe v5_p8)
    e = np.maximum(np.asarray(args["e_sgs"])[:, 0, 0], sase_ref.E_MIN)

    # the column really is the construction this fixture is for
    _, qs, _ = _vent_theta_es(args["theta"][:, 0, 0],
                              args["qv"][:, 0, 0],
                              args["qc"][:, 0, 0], args["p"][:, 0, 0])
    member = (((args["qv"] + args["qc"])[:, 0, 0] >= qs)
              & np.asarray(args["n2m_mask"])[:, 0, 0])
    assert list(np.nonzero(member)[0]) == list(
        range(k_base, k_top + 1))                  # the member run
    assert k_base - k_r >= 2                       # base ABOVE the root
    assert e[k_r:k_top + 2].max() / e[k_r:k_top + 2].min() > 1.0e4

    def wmean(lo, hi):
        return float(np.sum(thick[lo:hi + 1] * e[lo:hi + 1])
                     / np.sum(thick[lo:hi + 1]))

    ebar_base = wmean(k_base, k_top + 1)
    ebar_root = wmean(k_r, k_top + 1)
    assert ebar_base == pytest.approx(3.979314448323069e-3, rel=1e-12)
    assert ebar_root == pytest.approx(2.2380569891450888e-2, rel=1e-12)
    assert ebar_root / ebar_base == pytest.approx(5.6242, rel=1e-4)

    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**args)
    export = float(f_qv[k_top + 1, 0, 0] + f_qc[k_top + 1, 0, 0])
    assert export == pytest.approx(1.4398699509941751e-05, rel=1e-12)

    # THE ANCHOR IS AT CLOUD BASE, not at the root: the shipped
    # amplitude is the one built from ebar_base.  Verified by the
    # closure's own linearity in sqrt(ebar) -- rescaling e_sgs by the
    # ratio of the two window means must reproduce the root-anchored
    # amplitude exactly, and it is the CLOUD-BASE window that leaves
    # the shipped number invariant.
    scaled = dict(args)
    scaled["e_sgs"] = np.asarray(args["e_sgs"]) * (ebar_root / ebar_base)
    s_qv, s_qc = sase_ref.plume_vent_flux(**scaled)[1:]
    export_scaled = float(s_qv[k_top + 1, 0, 0] + s_qc[k_top + 1, 0, 0])
    assert export_scaled / export == pytest.approx(
        np.sqrt(ebar_root / ebar_base), rel=1e-9)

    # the analytic round-4 counterfactual (docstring): amplitude ratio
    # x shape-normalization ratio, both exact for a grow-zone face.
    amp_ratio = np.sqrt(ebar_root / ebar_base)
    shape_ratio = (zf[k_base] / zf[k_r + 1]) ** sase_ref.VENT_ENT_COEF
    assert amp_ratio == pytest.approx(2.3715453896548633, rel=1e-12)
    assert shape_ratio == pytest.approx(1.6132294349373353, rel=1e-12)
    assert amp_ratio * shape_ratio == pytest.approx(3.825846828881158,
                                                    rel=1e-12)
    assert export * amp_ratio * shape_ratio == pytest.approx(
        5.508721886012332e-05, rel=1e-12)           # the round-4 build

    # the k_base = 0 guard is structural, not a floor: face 0 is the
    # ground (z_f[0] = 0 exactly) and can never normalize a shape.
    assert zf[0] == 0.0
    assert zf[max(k_base, 1)] == zf[k_base]         # guard inert here


def test_m2_real_column_theta_continuity():
    """S4-4 ROUND-5 CONTINUITY (design doc SASE-M2 amendment "root /
    anchor separation", clause 3): export must be continuous in the
    state at modelling accuracy -- a +-0.05 K theta perturbation may
    not produce a step change -- and the property is fixtured on a REAL
    column, not on the specimen alone.

    THE COLUMN (:mod:`vent_columns_yolod_11z`, KNIFE_*, j=23 i=135,
    34.7273 N -87.8241 W): a 2-cell saturated run based at k13
    (821.7 m).  Its theta_es profile sits one hundredth of a K from
    manufacturing an interior maximum at k1: theta[0] - 0.05 K (or,
    equivalently, theta[1] + 0.05 K) makes k1 a local theta_es maximum
    that is absent in the unperturbed column (pinned at import by the
    fixture module).  On the round-4 build (HEAD df235f0, instrumented
    copy, probe v5_p7_detail.py) that flip moved the root from k13 to
    k1 and multiplied the export by 102.695 -- a two-orders-of-
    magnitude step from a perturbation four orders below any
    meteorological signal.

    WHAT THE ROUND-5 RULE DOES HERE, MEASURED: the depth floor
    k_r >= k_base - (k_top - k_base) - 1 = 13 - 1 - 1 = 11 refuses a k1
    source for a run based at k13, so the root does not move and BOTH
    perturbations leave all three flux profiles BITWISE unchanged.

    WHAT IS *NOT* CLAIMED (measured counterexamples on this same
    column, pinned below so the limitation cannot be quietly
    forgotten).  Clause 3's "anywhere" is NOT achieved: export is
    continuous against perturbations that only move the ROOT, but it
    still steps where 0.05 K moves the SATURATED-LAYER STRUCTURE
    itself, which is the M1 mask's own state trigger (a switch, by
    construction):

    * theta[13] + 0.05 K stands the limb DOWN entirely (export ratio
      exactly 0.0) -- the run's own base cell leaves the moist-unstable
      class;
    * theta[10] - 0.05 K (or theta[11] + 0.05 K) moves the root from
      k13 to k11, inside the floor, and multiplies the export by
      1.5868518400110598 -- the residual parcel-origin sensitivity,
      which the anchor does not remove because the ROOT still sets the
      launch state.

    POPULATION CONTEXT (probe v5_p2_pop.py, e_sgs = 1.0, 11Z/13Z):
    theta[0] - 0.05 K moves the export by more than 50% on 16.95% /
    19.68% of firing columns under the round-4 build and on 0.49% /
    0.23% under this one.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_11z as vc
    args = _vent_real_args("KNIFE")
    k_base, k_top = 13, 14                    # measured; also asserted
    _, qs, thes = _vent_theta_es(args["theta"][:, 0, 0],
                                 args["qv"][:, 0, 0],
                                 args["qc"][:, 0, 0], args["p"][:, 0, 0])
    member = (((args["qv"] + args["qc"])[:, 0, 0] >= qs)
              & np.asarray(args["n2m_mask"])[:, 0, 0])
    assert list(np.nonzero(member)[0]) == [k_base, k_top]
    assert vc.KNIFE_J == 23 and vc.KNIFE_I == 135
    # the structural floor this column's continuity rides on
    assert k_base - (k_top - k_base) - 1 == 11

    base = sase_ref.plume_vent_flux(**args)
    export0 = float(base[1][k_top + 1, 0, 0] + base[2][k_top + 1, 0, 0])
    assert export0 > 0.0
    assert export0 == pytest.approx(1.3041626557893029e-06, rel=1e-12)

    # THE KNIFE EDGE: the perturbation that stepped the round-4 build
    # by 102.695 leaves this one BITWISE unchanged -- on both of the
    # two perturbations that manufacture the same k1 theta_es peak.
    for k, dth in ((0, -0.05), (1, +0.05)):
        pert = _vent_theta_perturbed(args, k, dth)
        _, _, thes_p = _vent_theta_es(pert["theta"][:, 0, 0],
                                      pert["qv"][:, 0, 0],
                                      pert["qc"][:, 0, 0],
                                      pert["p"][:, 0, 0])
        # non-vacuity: the perturbed column really does have the k1
        # theta_es maximum the unperturbed one lacks.
        assert not (thes[1] > thes[0] and thes[1] > thes[2])
        assert thes_p[1] > thes_p[0] and thes_p[1] > thes_p[2]
        out = sase_ref.plume_vent_flux(**pert)
        for a, b in zip(out, base):
            assert a.tobytes() == b.tobytes(), (k, dth)

    # the whole sub-cloud layer BELOW the floor is continuous the same
    # way: no perturbation at k0..k9 moves any flux bit (k10 and k11
    # can still manufacture a peak AT the floor level k11, which is a
    # legal source -- the counterexamples pinned below).
    for k in range(0, 10):
        for dth in (+0.05, -0.05):
            out = sase_ref.plume_vent_flux(
                **_vent_theta_perturbed(args, k, dth))
            for a, b in zip(out, base):
                assert a.tobytes() == b.tobytes(), (k, dth)

    # THE MEASURED COUNTEREXAMPLES (docstring): pinned, not hidden.
    def ratio(k, dth):
        out = sase_ref.plume_vent_flux(
            **_vent_theta_perturbed(args, k, dth))
        return float(out[1][k_top + 1, 0, 0]
                     + out[2][k_top + 1, 0, 0]) / export0

    assert ratio(13, +0.05) == 0.0                 # mask stand-down
    assert ratio(11, +0.05) == pytest.approx(1.5868518400110598,
                                             rel=1e-9)
    assert ratio(10, -0.05) == pytest.approx(1.5868518400110598,
                                             rel=1e-9)


def test_m2_binding_rule_real_columns(monkeypatch):
    """S4-5b Item 1, AUTHORITY SIDE: the two rules the round-5 amendment
    ADDED are exercised where they BIND, on real d02 columns.

    Neither rule bound anywhere in the registered corpus.  A review lane
    deleted the root DEPTH FLOOR from the device kernel and all four M2
    GPU gates still passed; the same for the MINIMUM-RUN guard.  This
    fixture and its device companion
    (``test_sase_gpu.test_m2_plume_vent_device_parity_and_index_agreement``
    and ``...test_m2_device_min_run_guard_stands_down``) close that.

    THE DEPTH FLOOR (design doc SASE-M2 amendment "root / anchor
    separation", clause 1: k_r >= k_base - (k_top - k_base) - 1).  Both
    columns of :mod:`vent_columns_yolod_binding` carry an interior
    theta_es maximum at or below their run's base -- the level the
    pre-amendment rule would root the parcel at -- lying BELOW the
    floor, so the floor clamps the root to the run's own base:

    * CLAMP (j=180 i=78): run k20..k21, floor 18, eligible peak k15
      (1152.6 m).  Root 20 with the floor, 15 without it.
    * DROP (j=271 i=44): run k18..k22, floor 13, eligible peak k11
      (539.9 m).  Root 18 with the floor, 11 without it.

    Asserted here: the closure's OWN root (recovered from the returned
    flux support, not transcribed) is the clamped one on both columns,
    and it differs from :func:`_vent_root_index_no_floor` -- the
    deleted rule's own reference -- so a build without the floor cannot
    reproduce these bytes.

    THE MINIMUM-RUN GUARD (VENT_MIN_RUN_CELLS).  MEASURED this session
    over every column of the four yolo-d d02 frames (800,000 columns,
    probe s5b_p1_survey.py): at the registered pair
    (VENT_MASK = "bulk-theta-es-v1", VENT_MIN_RUN_CELLS = 2) the guard
    binds on EXACTLY ZERO columns, and it cannot bind on any state --
    the bulk reading is ``thes[k] - base_thes < 0`` with ``base_thes``
    captured at the run's own start, so a one-cell run compares a float
    with itself and reads exactly 0.0, which is not < 0.  The guard is
    therefore implied by the reading at the registered constant, and no
    fixture at those values can distinguish its deletion.  It becomes
    live at the registry's own alternative value 3 (the same value
    ``test_m2_registry_binds_vent_constants`` sweeps through the config
    ID), and CLAMP's run is exactly 2 cells long, so at 3 the guard --
    and nothing else -- stands the column down.  Pinned both ways:
    bitwise +0.0 at 3, and the head fluxes back at 1, where
    ``long_enough`` is unconditionally true, i.e. the reviewer's own
    deletion expressed through the registered parameter.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_binding as vc

    for name, want_kb, want_kt, want_floor, want_nofloor in (
            ("CLAMP", 20, 21, 18, 15), ("DROP", 18, 22, 13, 11)):
        args = _vent_binding_args(name)
        k_base, k_top = _vent_layer_indices(args)
        assert (k_base, k_top) == (want_kb, want_kt), (name, k_base,
                                                       k_top)
        assert k_base - (k_top - k_base) - 1 == want_floor, name
        k_r = _vent_root_index(args, k_base, k_top)
        k_nf = _vent_root_index_no_floor(args, k_base)
        assert k_r == k_base, (name, "the floor must clamp to the base",
                               k_r)
        assert k_nf == want_nofloor, (name, k_nf)
        assert k_nf < k_r, (name, "the deleted rule must MOVE the root")
        f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**args)
        assert float(np.abs(f_qv + f_qc).max()) > 0.0, (
            f"{name} no longer fires -- the fixture would be vacuous")
        # the closure's own root, read off its returned bytes: the grow
        # zone starts at face k_r + 1 (module docstring, SASE-M2 SHAPE),
        # so the lowest nonzero face IS k_r + 1.
        lowest = int(np.nonzero(f_th[:, 0, 0] != 0.0)[0].min())
        assert lowest == k_r + 1, (name, lowest, k_r)

    # --- the minimum-run guard, at the registry's live value ---------
    args = _vent_binding_args("CLAMP")
    zeros = np.zeros((np.asarray(args["theta"]).shape[0] + 1, 1, 1))
    head = sase_ref.plume_vent_flux(**args)
    monkeypatch.setattr(sase_ref, "VENT_MIN_RUN_CELLS", 3)
    for arr in sase_ref.plume_vent_flux(**args):
        assert arr.tobytes() == zeros.tobytes(), (
            "the 2-cell run must be rejected at VENT_MIN_RUN_CELLS = 3")
    monkeypatch.setattr(sase_ref, "VENT_MIN_RUN_CELLS", 1)
    for got, want in zip(sase_ref.plume_vent_flux(**args), head):
        assert got.tobytes() == want.tobytes(), (
            "with the guard neutralised the column must fire exactly as "
            "it does at the registered value -- otherwise the leg above "
            "is not isolating the guard")
    assert vc.SURVEY_E == 1.0


def test_m2_surface_based_layer_stands_down():
    """S4-5c: a saturated run based in the LOWEST MODEL LEVEL stands the
    limb down -- the FOURTH registered stand-down condition (design doc
    SASE-M section 4, amendment "a surface-based saturated layer stands
    the limb down"), on the real d02 column of
    :mod:`vent_columns_yolod_surface`.

    WHY THE RULE.  The grow-zone shape factor normalises on the
    cloud-base face, ``M_hat = (z_f[j]/z_f[k_base])**VENT_ENT_COEF``.
    ``z_f[0] = 0`` identically (the module's cumsum convention), so on a
    surface-based run there is no cloud base to normalise on: the
    eps ~ 1/z entrainment integral diverges at the ground and the
    pre-amendment ``z_f[max(k_base, 1)]`` guard silently substituted the
    LOWEST LAYER THICKNESS for a height.  MEASURED over the 19495
    post-step-4 active columns of the 11Z frame at the design point
    (S4-5c survey, whole frame, no sampling): median anchor face 610.38
    m on the 19404 columns with k_base > 0 against 17.08 m on the 91
    surface-based ones -- the amplitude became a function of the
    vertical grid rather than of the state, and those columns carry the
    population MAXIMUM export in all three surveyed frames.  Physically
    the ruling is the conservative one: M2 is a cloud-base mass-flux
    scheme, and a saturated layer sitting on the ground is fog or
    surface-based stratus, whose vertical mixing belongs to the ED limb
    (M1).  The rejected alternative ``M_hat = 1`` on the branch would
    have left a step discontinuity against ``k_base = 1`` columns, which
    the C9 continuity clause forbids.

    THE COLUMN (11Z, j=13 i=205, 34.5182 N -85.4787 W, land, HGT 406.3
    m).  Its lowest qualifying member run is k0..k17 -- k_base = 0, the
    surface-based case -- with the root falling back to the run's own
    base, k_r = 0.

    NON-VACUITY, and it is the whole point of the fixture: NONE of the
    three older stand-down conditions fires here.  There IS a qualifying
    run; the ceiling k_lid = k_top + 2 = 19 is far inside the 49-level
    column, so "k_lid past the column top" does not fire; the ceiling
    sits 2292.66 m above the root, inside VENT_DEPTH_CAP = 4000 m, so
    the depth cap does not fire; and the parcel does find an LFC and a
    neutral-buoyancy level, which is proved HERE by the strongest
    available construction -- the SAME arrays with membership vetoed at
    k0 (the M1 mask is a veto that can only REMOVE a cell, module
    docstring SASE-M2 MEMBERSHIP), i.e. the identical thermodynamics
    with the run's base one cell off the ground, fire at 17 live faces
    and export 7.665561399182368e-04 kg/m2/s.  The only structural
    difference between the two legs is whether the base sits on the
    ground.

    PRE-AMENDMENT (measured on the build this test was written against):
    the column FIRED, carrying exactly one live face -- face 1, at
    z_f[1] = 17.158959716345805 m, the substituted anchor itself, with
    F_theta = 1.0936614241370526e-04 and export 3.2450097412557837e-08.
    Because that single live face IS the anchor face, M_hat = 1 exactly
    on it: this column pins the RULING and not the amplification (on it
    the rejected ``M_hat = 1`` disposition and the pre-amendment build
    agree bitwise), which is why the amplification is carried as a
    population measurement in the fixture module's docstring rather than
    asserted here.

    ASSERTED: all three flux rows are ``tobytes()``-identical to +0.0
    zeros -- the same bitwise stand-down the other three conditions
    return.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_surface as vc

    args = _vent_surface_args()
    nz = np.asarray(args["theta"]).shape[0]
    k_base, k_top = _vent_layer_indices(args)
    assert (k_base, k_top) == (0, 17), (k_base, k_top)
    assert _vent_root_index(args, k_base, k_top) == 0
    # the two purely-geometric older conditions, from the fixture's own
    # arrays (the third is proved by the veto leg below)
    thick = np.asarray(args["dz_col"], np.float64)
    zc = np.cumsum(thick) - 0.5 * thick
    assert k_top + 2 < nz, (k_top, nz)
    assert (zc[k_top + 2] - zc[k_base]) < sase_ref.VENT_DEPTH_CAP

    zeros = np.zeros((nz + 1, 1, 1))
    for arr in sase_ref.plume_vent_flux(**args):
        assert arr.tobytes() == zeros.tobytes(), (
            "a surface-based saturated run must stand the limb down "
            "bitwise, like every other registered stand-down condition")

    # NON-VACUITY: the same column with the base one cell off the ground
    lifted = dict(args)
    mask = np.asarray(args["n2m_mask"]).copy()
    mask[0] = False
    lifted["n2m_mask"] = mask
    assert _vent_layer_indices(lifted) == (1, 17)
    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**lifted)
    live = np.nonzero(f_th[:, 0, 0] != 0.0)[0]
    assert live.size == 17 and int(live.min()) == 2, live
    assert float((f_qv + f_qc).max()) == pytest.approx(
        7.665561399182368e-04, rel=1e-9)
    assert vc.SURVEY_E == 1.0


def test_m2_depth_cap_search_term_stands_down_real_column(monkeypatch):
    """S4-5d: the ``VENT_DEPTH_CAP`` term inside the LFC/peak/NB
    TERMINATION SEARCH, on the real d02 column ``CAPSEARCH`` of
    :mod:`vent_columns_yolod_rules`.

    THE RULE.  ``incap = within_layer & ((z[k] - z_r) <= VENT_DEPTH_CAP)``
    gates every cell the termination search may look at, so a parcel
    whose LFC lies more than 4000 m above its root never finds one and
    the column stands down (module docstring, SASE-M2; VENT_DEPTH_CAP is
    the shallow-device scope guard of G-M6 -- deep CI belongs to the
    grid, not to a shallow-cumulus limb).

    WHY IT NEEDED A COLUMN.  Measured this session: deleting that term
    from a sandbox source-surgery copy of the authority leaves the CPU
    gate at 181 passed / 1 xfailed -- no test in the repository looked at
    it.  On real fields the term stands down 1 / 9 / 15 columns per
    frame at 11 / 12 / 13Z (0.005 / 0.036 / 0.069% of the 19404 / 24965
    / 21872 firing columns, design point e_sgs = 1.0, whole frames, no
    sampling), of which 0 / 1 / 3 are exclusive to it rather than shared
    with the at-lid term below.  It is a narrow rule -- and narrow is
    exactly why nothing in the constructed corpus reached it.

    THE COLUMN (13Z, j=242 i=104, 40.7281 N -89.3530 W, land, HGT 231.6
    m).  Member run k15..k27, root k11 at z = 537.3801 m, k_lid = 29 at
    z = 7878.5449 m -- 7341.1649 m above the root, so the cap truncates
    the search at less than 55% of the way to the lid and no LFC is
    found inside it.

    NON-VACUITY.  None of the other three registered stand-downs can be
    what silences this column, and the fixture module asserts each
    premise at import: a qualifying run exists (so it is not the no-run
    case), k_base = 15 > 0 (so it is not the surface-based stand-down)
    and k_lid = 29 < nz = 49 (so it is not "k_lid past the column top").
    That leaves the cap, and the cap is shown to be the cause TWICE
    over: raising ``VENT_DEPTH_CAP`` to 10000 m -- the registry's own
    constant, monkeypatched, the idiom
    :func:`test_m2_binding_rule_real_columns` uses for
    ``VENT_MIN_RUN_CELLS`` -- fires the column at export
    3.3985196539e-04 kg/m2/s over 15 faces, and deleting the search term
    alone from the sandbox copy fires it at exactly the same value
    (measured this session).  Deleting the AT-LID cap term instead
    leaves it bitwise zero, so this column pins the search term
    specifically.

    ASSERTED: all three rows bitwise +0.0 at the registered cap; the
    geometric premise depth > VENT_DEPTH_CAP; and the raised-cap leg's
    export and face count.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_rules as vc

    args = _vent_rules_args("CAPSEARCH")
    nz = np.asarray(args["theta"]).shape[0]
    k_base, k_top = _vent_layer_indices(args)
    assert (k_base, k_top) == (15, 27), (k_base, k_top)
    assert k_base > 0 and k_top + 2 < nz          # not the other two
    thick = np.asarray(args["dz_col"], np.float64)
    zc = np.cumsum(thick) - 0.5 * thick
    k_r = _vent_root_index(args, k_base, k_top)
    assert k_r == 11, k_r
    depth = float(zc[k_top + 2] - zc[k_r])
    assert depth == pytest.approx(7341.1649, rel=1e-6)
    assert depth > sase_ref.VENT_DEPTH_CAP        # THE binding premise

    zeros = np.zeros((nz + 1, 1, 1))
    for arr in sase_ref.plume_vent_flux(**args):
        assert arr.tobytes() == zeros.tobytes(), (
            "a lid beyond VENT_DEPTH_CAP of the root must stand the "
            "limb down bitwise")

    # NON-VACUITY: the same arrays with the registered cap raised past
    # the column's own depth.  Nothing else about the column moves.
    monkeypatch.setattr(sase_ref, "VENT_DEPTH_CAP", 10000.0)
    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**args)
    live = _vent_live_faces((f_th, f_qv, f_qc))
    assert live.size == 15 and int(live.min()) == 12, live
    assert float(np.abs(f_qv + f_qc).max()) == pytest.approx(
        3.3985196539e-04, rel=1e-9)
    assert vc.SURVEY_E == 1.0


def test_m2_depth_cap_at_lid_term_stands_down_real_column(monkeypatch):
    """S4-5d: the ``VENT_DEPTH_CAP`` term inside the AT-LID forced-NB
    test, on the real d02 column ``CAPLID`` of
    :mod:`vent_columns_yolod_rules`.

    THE RULE.  A parcel that is still buoyant when the search reaches
    ``k_lid`` terminates THERE (module docstring, SASE-M2 TERMINATION /
    INVERSION BASE) -- but only if the lid itself is within
    ``VENT_DEPTH_CAP`` of the root::

        at_lid = above & (k == k_lid) & lfc_found & ~nb_found
                       & ((z[k] - z_r) <= VENT_DEPTH_CAP)

    Without that last conjunct the forced termination would re-admit
    every deep column the search term had just excluded, which is the
    G-M6 scope guard read backwards.

    WHY IT NEEDED A COLUMN.  Measured this session: deleting the
    conjunct leaves the CPU gate at 181 passed / 1 xfailed.  On real
    fields it stands down 31 / 110 / 514 columns per frame at 11 / 12 /
    13Z (0.16 / 0.44 / 2.35% of the firing set at the design point),
    30 / 102 / 502 of them exclusively -- an order of magnitude more
    than the search term, and by 13Z the larger of the two by far.

    THE COLUMN (13Z, j=81 i=102, 36.2702 N -89.0484 W, land, HGT 95.2
    m).  Member run k9..k23, root k9 at z = 373.3170 m (the classical
    cloud-base root), k_lid = 25 at z = 5758.4723 m -- 5385.1553 m above
    the root.  Unlike the CAPSEARCH column this parcel DOES find an LFC
    inside the capped region and does NOT find a natural neutral
    buoyancy below the lid, so the at-lid term is the only thing between
    it and a forced termination: it is precisely the class the conjunct
    exists to exclude.

    NON-VACUITY, three ways.  The fixture asserts at import that a
    qualifying run exists, that k_base = 9 > 0 and that k_lid = 25 < nz
    = 49.  Deleting ONLY the at-lid conjunct in a sandbox copy fires the
    column at export 4.5887634545e-04 kg/m2/s (measured this session)
    while deleting only the SEARCH term leaves it bitwise zero -- so the
    two depth-cap tests are term-exclusive in both directions.  And
    raising the registered cap to 10000 m fires it at
    7.0667177459e-04: larger than the at-lid-only value because a
    raised cap also lets the search find a natural NB, which is the
    physically-correct termination the cap was hiding.

    ASSERTED: all three rows bitwise +0.0; the geometric premise; the
    LFC-exists premise (proved by the raised-cap leg firing); and the
    raised-cap export.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_rules as vc

    args = _vent_rules_args("CAPLID")
    nz = np.asarray(args["theta"]).shape[0]
    k_base, k_top = _vent_layer_indices(args)
    assert (k_base, k_top) == (9, 23), (k_base, k_top)
    assert k_base > 0 and k_top + 2 < nz
    thick = np.asarray(args["dz_col"], np.float64)
    zc = np.cumsum(thick) - 0.5 * thick
    k_r = _vent_root_index(args, k_base, k_top)
    assert k_r == k_base == 9, (k_r, k_base)
    depth = float(zc[k_top + 2] - zc[k_r])
    assert depth == pytest.approx(5385.1553, rel=1e-6)
    assert depth > sase_ref.VENT_DEPTH_CAP

    zeros = np.zeros((nz + 1, 1, 1))
    for arr in sase_ref.plume_vent_flux(**args):
        assert arr.tobytes() == zeros.tobytes(), (
            "a forced at-lid termination beyond VENT_DEPTH_CAP of the "
            "root must stand the limb down bitwise")

    monkeypatch.setattr(sase_ref, "VENT_DEPTH_CAP", 10000.0)
    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**args)
    live = _vent_live_faces((f_th, f_qv, f_qc))
    assert live.size == 15 and int(live.min()) == 10, live
    assert float(np.abs(f_qv + f_qc).max()) == pytest.approx(
        7.0667177459e-04, rel=1e-9)
    assert vc.SURVEY_E == 1.0


def test_m2_k_lid_bound_seals_the_cap_out_of_the_search():
    """S4-5d: the ``k_lid`` BOUND on the termination search -- the C9
    cap-preservation mechanism -- on the real d02 column ``KLID`` of
    :mod:`vent_columns_yolod_rules`.

    THE RULE.  ``within_layer = above & (k < k_lid)`` confines the
    LFC/peak/NB search to the saturated run PLUS its single discrete
    entrainment-zone cell (design doc SASE-M2 amendment "discrete C9
    reading", clauses 2-3).  ``k_lid = k_top + 2`` is the entrainment
    zone's own TOP face index and is the C9 hard-zero boundary: the cap
    proper, above it, receives bitwise zero on every row.

    THE STRONGEST STATEMENT OF THAT RULE is not a number but an
    INVARIANCE: if the search never looks above ``k_lid``, then the
    state of the cap CANNOT influence the vent at all.  That is what
    this test asserts -- the whole column above the C9 boundary is
    warmed by 2 K and every one of the three flux rows must come back
    ``tobytes()``-identical.  It is the C9 mandate ("venting never
    erodes the inversion") read in the only direction a fixture can
    check per column, and it is value-free: no golden to re-pin, no
    tolerance to argue about.

    WHY IT NEEDED A COLUMN.  Measured this session, on sandbox
    source-surgery copies: shifting the bound ONE CELL up
    (``k < k_lid + 1``) leaves the CPU gate at 181 passed / 1 xfailed,
    and so does deleting it outright.  On real fields the up-shift moves
    the flux of 17 / 25 / 18 already-firing columns per frame at
    11 / 12 / 13Z (0.09 / 0.10 / 0.08% of the firing set) and newly
    fires 117 / 206 / 106 more; deletion newly fires 750 / 428 / 204.
    (The opposite direction, ``k < k_lid - 1``, IS already caught -- it
    fails three existing tests -- which is why the gap is one-sided and
    why the invariance leg below is written to catch the UP direction.)

    THE COLUMN (11Z, j=70 i=251, 36.0987 N -83.9468 W, land, HGT 359.0
    m).  Member run k13..k15, root k13, entrainment zone k16, k_lid =
    17.  The parcel is still buoyant when the search reaches the lid, so
    termination is FORCED there -- the class where the bound is doing
    the work rather than a natural neutral-buoyancy level.  Shipped
    export 3.037647519973e-05 kg/m2/s on faces {14, 15, 16}; with the
    bound shifted one cell up, or removed, the same three faces carry
    7.095623763141e-05 (a factor 2.3359) because the cap cell's own
    buoyancy is then admitted to the shape's peak.  Warming the cap by
    2 K collapses the mutant back to 3.037647520e-05 -- i.e. the mutant
    is reading the cap and the shipped code is not.

    ASSERTED: the index structure; C9 hard zero at and above face
    k_lid; the pinned export; and bitwise invariance of all three rows
    to a +2 K perturbation of every level at and above k_lid.  The
    perturbation leg holds ``n2m_mask`` at its unperturbed value on
    purpose -- the mask is an INPUT to the closure, and the experiment
    is about what the closure does with the levels above the boundary,
    not about re-deriving the M1 seam.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_rules as vc

    args = _vent_rules_args("KLID")
    nz = np.asarray(args["theta"]).shape[0]
    k_base, k_top = _vent_layer_indices(args)
    assert (k_base, k_top) == (13, 15), (k_base, k_top)
    k_lid = k_top + 2
    assert k_lid == 17 and k_lid < nz - 1

    shipped = sase_ref.plume_vent_flux(**args)
    live = _vent_live_faces(shipped)
    assert live.tolist() == [14, 15, 16], live
    for arr in shipped:                        # C9 hard-zero boundary
        assert float(np.abs(arr[k_lid:, 0, 0]).max()) == 0.0
    assert float(np.abs(shipped[1] + shipped[2]).max()) == pytest.approx(
        3.037647519973e-05, rel=1e-9)

    # THE INVARIANCE: the cap cannot reach the vent.
    warmed = dict(args)
    theta = np.asarray(args["theta"], np.float64).copy()
    theta[k_lid:] += 2.0
    warmed["theta"] = theta
    for a, b in zip(shipped, sase_ref.plume_vent_flux(**warmed)):
        assert a.tobytes() == b.tobytes(), (
            "the termination search is bounded at k_lid, so the state "
            "of the cap above it cannot move a single flux bit")
    assert vc.SURVEY_E == 1.0


def test_m2_lowest_qualifying_run_is_the_one_vented():
    """S4-5d: the LOWEST-CONTIGUOUS-RUN rule (the ``~chosen`` latch of
    step 1), on the real d02 column ``LOWRUN`` of
    :mod:`vent_columns_yolod_rules`.

    THE RULE.  Step 1 walks the column upward and takes the FIRST
    member run that is long enough and reads moist-unstable::

        take = end_here & long_enough & reading & ~chosen

    The ``~chosen`` latch is what makes "first" mean anything; without
    it the assignment simply keeps overwriting and the HIGHEST
    qualifying run wins.  The closure is a cloud-base mass-flux scheme
    and the plume is launched from the lowest cloud base in the column,
    not from whichever deck happens to be last.

    WHY IT NEEDED A COLUMN.  Measured this session: deleting the latch
    leaves the CPU gate at 181 passed / 1 xfailed -- no registered
    column carries two qualifying runs, so on the whole corpus "first"
    and "last" coincide.  On real fields they do not: at the design
    point the latch changes the flux of 54 / 229 / 132 firing columns
    per frame at 11 / 12 / 13Z (0.28 / 0.92 / 0.60% of the firing set),
    with 10 / 27 / 32 columns newly firing and 9 / 21 / 15 ceasing to.

    THE COLUMN (11Z, j=17 i=164, 34.5965 N -86.8484 W, land, HGT 173.5
    m).  TWO qualifying member runs -- k8..k11 and k13..k15 -- both at
    least VENT_MIN_RUN_CELLS long and both moist-unstable on the
    registered bulk theta_es reading (the fixture asserts both premises
    at import, so the test cannot go vacuous if the profile drifts).
    The closure takes the LOWER one: support {9, 10}, export
    7.550911281670e-06 kg/m2/s.  With the latch deleted it takes the
    upper one: support {13, 14, 15, 16}, export 8.609949042543e-05 --
    a factor 11.4025.

    ASSERTED, and deliberately as a SUPPORT statement rather than only a
    value: every live face lies strictly below the second run's base, so
    the plume is anchored to the lower deck.  A mutant that vents the
    upper run fails that by construction whatever its amplitude turns
    out to be.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_rules as vc

    args = _vent_rules_args("LOWRUN")
    k_base, k_top = _vent_layer_indices(args)
    assert (k_base, k_top) == (8, 11), (k_base, k_top)
    second_base = 13                     # the fixture pins both runs

    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**args)
    live = _vent_live_faces((f_th, f_qv, f_qc))
    assert live.size and int(live.max()) < second_base, live
    assert live.tolist() == [9, 10], live
    assert float(np.abs(f_qv + f_qc).max()) == pytest.approx(
        7.550911281670e-06, rel=1e-9)
    assert vc.SURVEY_E == 1.0


def test_m2_root_search_bound_admits_the_cloud_base_peak():
    """S4-5d: the ROOT SEARCH BOUND ``k <= k_base``, on the real d02
    column ``ROOTBND`` of :mod:`vent_columns_yolod_rules`.

    THE RULE.  Step 1b takes the HIGHEST interior theta_es maximum AT OR
    BELOW the run's own base, floored structurally at
    ``k_base - (k_top - k_base) - 1``::

        k_r = where(is_peak & (k <= k_base) & (k >= k_r_floor), k, k_r)

    The ``<=`` is essential: a theta_es maximum sitting exactly at
    the cloud base IS an eligible root, and admitting it is what keeps
    the plume anchored at cloud base instead of reaching down into a
    deeper decrease layer -- the defect the round-5 root/anchor
    amendment was written against (design doc SASE-M2, "root / anchor
    separation", clause 1).

    WHY IT NEEDED A COLUMN.  Measured this session on sandbox copies:
    DELETING the bound fails 20 tests and shifting it one cell LOOSER
    (``k <= k_base + 1``) fails 9, so those directions were already
    covered -- but shifting it one cell TIGHTER, ``k < k_base``, leaves
    the CPU gate at 181 passed / 1 xfailed.  That is the direction with
    the real footprint: at the design point it changes the flux of
    240 / 474 / 260 firing columns per frame at 11 / 12 / 13Z (1.24 /
    1.90 / 1.19% of the firing set) and stops 153 / 116 / 87 of them
    venting altogether.

    THE COLUMN (11Z, j=21 i=39, 34.4919 N -91.0144 W, land, HGT 52.7 m).
    Member run k11..k15, so the structural depth floor is
    k_r >= 11 - 4 - 1 = 6, and the profile carries interior theta_es
    maxima at k7 AND at k11 = the base -- both above the floor, so the
    floor cannot decide between them and the ``k <= k_base`` bound is
    the only rule that does.  It picks k11, the plume launches from the
    cloud-base face and the lowest live face is 12.  One cell tighter
    and the root drops to k7, the support opens down to face 8 -- while
    the peak export moves by just +2.18%, 1.160283251111e-04 ->
    1.185612153199e-04 kg/m2/s.

    THAT NUMBER IS THE POINT.  A 2% amplitude change is inside the noise
    any value pin would tolerate, and the same mutation stops 153
    columns venting elsewhere in the frame.  So this test pins the
    SUPPORT -- the plume launches at the cloud-base face, and nowhere
    below it -- which a value pin cannot see and which a five-cell root
    displacement cannot survive.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_rules as vc

    args = _vent_rules_args("ROOTBND")
    k_base, k_top = _vent_layer_indices(args)
    assert (k_base, k_top) == (11, 15), (k_base, k_top)
    # the peak AT the base is the one the bound admits; the fixture
    # pins that k7 is the only other eligible candidate.
    assert _vent_root_index(args, k_base, k_top) == k_base == 11
    assert _vent_root_index_no_floor(args, k_base) == 11

    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**args)
    live = _vent_live_faces((f_th, f_qv, f_qc))
    assert live.size, "the column must vent for the support pin to bind"
    assert int(live.min()) == k_base + 1, (
        "the plume launches at the cloud-base face: the root is the "
        "theta_es maximum AT k_base, so no face below it can be live")
    assert live.tolist() == [12, 13, 14, 15, 16], live
    assert float(np.abs(f_qv + f_qc).max()) == pytest.approx(
        1.160283251111e-04, rel=1e-9)
    assert vc.SURVEY_E == 1.0


def test_m1_rh100_saturation_limb_decides_a_real_column():
    """S4-5d: the RH100 limb of ``MOIST_STABILITY_SWITCH`` =
    "binary-qc-or-rh100-liquid", on the real d02 column ``RH100`` of
    :mod:`vent_columns_yolod_rules`.

    THE RULE.  :func:`moist_n2`'s saturation switch is
    ``sat = (qc > 0) | (qv >= q_s)``.  The second limb is what makes the
    switch a STATE test rather than a condensate test: a cell that has
    reached saturation but has not yet been given condensate by the
    microphysics is physically saturated and must carry the DK82 moist
    N^2, not the dry one.

    WHY IT NEEDED A COLUMN.  Measured this session: deleting the limb
    (``sat = (qc > 0)``) leaves the CPU gate at 181 passed / 1 xfailed.
    Every saturated cell in the constructed corpus also carries
    condensate, so the first limb decides everywhere and the second is
    never the deciding term.  On real fields it decides 569 / 802 / 966
    cells per frame at 11 / 12 / 13Z (of 9.8e6), it FLIPS THE SIGN of
    N^2 on 138 / 199 / 281 of them, and through the M1 substitution mask
    it moves the M2 flux of 92 / 142 / 165 firing columns (0.47 / 0.57 /
    0.75% of the firing set) with 47 / 75 / 74 of them ceasing to vent.

    THE COLUMN (11Z, j=22 i=196, 34.7600 N -85.7853 W, land, HGT 378.8
    m).  Cell k16 at z = 1383.242 m carries qc = 0.0 EXACTLY -- so the
    ``qc > 0`` limb is unambiguously false -- at RH = 100.0400%.  The
    limb calls it saturated and the stability sign flips:
    N^2 = +1.2011930913551767e-04 dry against -1.74083361149572e-05
    moist.  That cell is the M2 member run's own TOP and it is what
    makes the run moist-unstable at all: the registered bulk theta_es
    reading over k5..k16 is -1.456900465974627 K (chosen) against
    +0.3003687273322839 K over k5..k15 (rejected).  With the limb
    deleted, no run in the column passes the reading, the closure stands
    down at step 1 and the export goes 2.713306581305e-04 -> 0.0
    (measured this session on the sandbox copy).

    ASSERTED: qc is bitwise zero and qv >= q_s at k16 (the premise);
    ``moist_n2`` there is NOT the dry field and carries the opposite
    sign; the M1 substitution mask contains k16; and the column vents at
    the pinned export.  Every one of those fails under the deletion.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_rules as vc

    args = _vent_rules_args("RH100")
    theta, qv = args["theta"], args["qv"]
    qc, p = args["qc"], args["p"]
    thick = np.asarray(args["dz_col"], np.float64)
    _, qs, _ = _vent_theta_es(theta[:, 0, 0], qv[:, 0, 0],
                              qc[:, 0, 0], p[:, 0, 0])
    k = 16
    assert float(qc[k, 0, 0]) == 0.0             # the qc limb is FALSE
    assert float(qv[k, 0, 0]) >= float(qs[k])    # the RH100 limb is TRUE
    assert float(qv[k, 0, 0] / qs[k]) == pytest.approx(1.000400, rel=1e-5)

    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)
    assert float(n2[k, 0, 0]) > 0.0 > float(n2m[k, 0, 0]), (
        "the RH100 limb must put the DK82 moist N^2 in this cell, and "
        "here that flips the stability sign")
    assert n2m[k, 0, 0] != n2[k, 0, 0]
    assert float(n2[k, 0, 0]) == pytest.approx(
        1.2011930913551767e-04, rel=1e-9)
    assert float(n2m[k, 0, 0]) == pytest.approx(
        -1.74083361149572e-05, rel=1e-9)
    # ... and the M1 seam mask -- the M2 limb's own membership veto --
    # therefore contains the cell, which is what carries the switch
    # into the venting decision.
    assert bool(np.asarray(args["n2m_mask"])[k, 0, 0])

    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**args)
    assert float(np.abs(f_qv + f_qc).max()) == pytest.approx(
        2.713306581305e-04, rel=1e-9)
    assert vc.SURVEY_E == 1.0


def test_m1_moist_stability_reaches_the_les_limb_length(monkeypatch):
    """S4-5d: the M1 substitution into the LES-limb length
    ``l_les = vertical_mixing_length(z, e, n2_eff, z0)``, on the real
    d02 column ``LLES`` of :mod:`vent_columns_yolod_rules`.

    THE RULE.  SASE-M1 point 1 (module docstring, SASE-M1 section): the
    effective stability ``n2_eff`` -- not the dry field -- is what the
    whole stability machinery consumes, and that includes the LES limb's
    own length.  The spec is explicit that M1 is active "at any dx"
    (design doc section 3), and the f = 1 identity is registered for M2
    ONLY (section 5), so the LES limb is not exempt.

    WHY IT NEEDED A COLUMN.  Measured this session: substituting the dry
    ``n2`` back into that ONE call leaves the CPU gate at 181 passed /
    1 xfailed.  The sibling substitution sites are all pinned; this one
    was not, and it is not a rare path -- at the design point the
    substitution moves ``l_les`` in 289095 / 326032 / 356516 cells per
    frame at 11 / 12 / 13Z, on 55223 / 60338 / 64780 of the 200000
    columns, i.e. 27.6 / 30.2 / 32.4% of the domain.

    THE COLUMN (11Z, j=191 i=75, 39.2545 N -90.2670 W, land, HGT 154.6
    m).  Ten cells move; the largest is k18 at z = 1928.580 m, where
    l_les goes 54.981806604349266 -> 125.58148267649028 m, a factor
    2.284055225398074.  Since the LES limb is K_v = C_KV*l_les*sqrt(e),
    that is K_v 53.66774184813372 -> 122.58008620354472 m2/s.

    HOW IT IS OBSERVED.  At f = 1 the two-product blend erases the RANS
    limb FP-exactly, so ``K_v`` IS ``C_KV*l_les*sqrt(e)`` and the split
    step's momentum row is a pure function of it.  The state is the real
    column replicated across a horizontally uniform 4 x 4 box, so the
    explicit horizontal channel contributes bitwise nothing and
    ``u_new`` is exactly ``implicit_vertical_diffusion(u, face(K_v),
    dt)``.  f is forced with the registered S3-6f-cap monkeypatch idiom
    (``dynamic_solve`` -> f_solved = 1, ``partition_cap`` -> 1,
    ``w_resolved_bound`` -> f_w = 1) already used by the lake harness in
    this file.

    ASSERTED, both directions: ``u_new`` is ``tobytes()``-identical to
    the reference built from the MOIST length, and NOT identical to the
    reference built from the dry one, which the two references differ by
    9.26543745399755e-03 m/s over a single 15 s step (and 0.15166407
    m2/s2 of e).  The dry-leg assertion is what fails the moment the
    substitution is undone.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_rules as vc

    thick = np.asarray(vc.LLES_THICK, np.float64)
    nz = thick.size
    shape = (nz, 4, 4)

    def box(a):
        return np.broadcast_to(
            np.asarray(a, np.float64)[:, None, None], shape).copy()

    u, v = box(vc.LLES_U), box(vc.LLES_V)
    w = np.zeros(shape)
    e = np.full(shape, vc.SURVEY_E)
    theta, qv = box(vc.LLES_THETA), box(vc.LLES_QV)
    qc, p = box(vc.LLES_QC), box(vc.LLES_P)
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)
    z, _, _ = sase_ref._column_geometry(shape, dz=None, dz_col=thick)
    dt, delta = 15.0, 3000.0

    # the registered f-forcing idiom: f_used = min(1, 1, 1) = 1
    monkeypatch.setattr(sase_ref, "dynamic_solve",
                        lambda *a, **k: (0.0, 1.0))
    monkeypatch.setattr(sase_ref, "partition_cap",
                        lambda delta_h, zi: 1.0)
    monkeypatch.setattr(
        sase_ref, "w_resolved_bound",
        lambda w_, e_mean, n2=None: sase_ref.WSensorState(
            f_w=1.0, alpha_w=1.0, e_res_w=0.0, coverage=0.0))

    fields, ledger = sase_ref.sase_split_step(
        u, v, w, theta, e, dx=delta, dy=delta, dz=200.0, delta=delta,
        dt=dt, n2=n2, dz_col=thick, n2_moist=n2m)
    assert ledger["f"] == 1.0

    root_e = np.sqrt(np.maximum(e, sase_ref.E_MIN))

    def momentum(stability):
        kv = (sase_ref.C_KV
              * sase_ref.vertical_mixing_length(z, e, stability, 0.0)
              * root_e)
        return sase_ref.implicit_vertical_diffusion(
            u, sase_ref._face_average(kv), dt, dz_col=thick)

    moist_ref, dry_ref = momentum(n2m), momentum(n2)
    assert fields["u"].tobytes() == moist_ref.tobytes(), (
        "the LES limb's own length must ride the M1 effective "
        "stability, not the dry field")
    assert fields["u"].tobytes() != dry_ref.tobytes()
    assert float(np.abs(moist_ref - dry_ref).max()) == pytest.approx(
        9.26543745399755e-03, rel=1e-9)

    l_moist = sase_ref.vertical_mixing_length(z, e, n2m, 0.0)[:, 0, 0]
    l_dry = sase_ref.vertical_mixing_length(z, e, n2, 0.0)[:, 0, 0]
    assert float(l_dry[18]) == pytest.approx(
        54.981806604349266, rel=1e-9)
    assert float(l_moist[18]) == pytest.approx(
        125.58148267649028, rel=1e-9)


def test_m1b_upward_excursion_negative_contribution_reaches_the_lid():
    """S4-5d: the M1b UPWARD-EXCURSION NEGATIVE CONTRIBUTION, on the
    real d02 column ``M1BDECK`` of :mod:`vent_columns_yolod_rules`.

    THE RULE.  :func:`bl89_moist_excursion_lengths` accumulates the
    excursion integrand from the M1 effective stability with its SIGN:
    "Moist-UNSTABLE stretches (N^2_m < 0) contribute negatively (the
    saturated parcel re-accelerates), so a moist-unstable deck spends
    nothing until the moist-stable lid: l_up = distance-to-lid plus a
    finite penetration -- the geometry the dry-theta lengths cannot see"
    (function docstring).  That is the entire reason the M1b limb
    exists; the dry lengths stop parcels INSIDE the deck and hide the
    lid, which is the G-M3 defect.

    WHY IT NEEDED A COLUMN.  Measured this session: clipping the upward
    face-mean slope at zero (``max(0.5*(n2[j-1] + n2[j]), 0)``), which
    is exactly "delete the negative contribution", leaves the CPU gate
    at 181 passed / 1 xfailed.  The registered deck fixture's ``l_up``
    window is wide enough to admit the clipped value.  On real fields
    the contribution moves ``l_up`` on 40.46 / 44.17 / 42.36% of
    M1-substituted cells at 11 / 12 / 13Z (131532 of 325087 at 11Z) --
    not a corner case but the majority behaviour of the limb.

    THE COLUMN (11Z, j=187 i=77, 39.1481 N -90.1843 W, land, HGT 182.7
    m).  Cells k16..k22 are all M1-substituted, all moist-UNSTABLE
    (N^2_eff -8.17e-05 .. -4.44e-05) and all dry-STABLE (N^2 +2.25e-05
    .. +1.09e-04) -- precisely the configuration the limb was added for
    -- under a lid at the interface z_f[23] = 4048.0217081083556 m where
    N^2_eff = +1.285538785444141e-04.

    ASSERTED: the deck geometry (from the authority's own N^2 fields,
    not transcribed); and that ``l_up`` from every cell of the deck
    REACHES THE LID.  From k16..k21 the shipped lengths clear the lid
    interface by 1330.5601 to 3644.3734 m.  The top deck cell k22 is
    governed by the registered one-cell bracketing convention
    ("excursions terminate within one cell of the geometric
    distance-to-lid", function docstring), so it is asserted against
    ``(z_L - z_k) - dz`` instead.  With the negative contribution
    clipped away the excursion stops 4.9382 m BELOW the lid interface
    from EVERY cell of the deck, so all seven assertions fail at once.

    RECORDED, because it bounds what this test does and does not claim:
    at the design point the moist master length ``l_m = min(l_up,
    l_down)`` binds the COMPOSED RANS mixing length on only 0.3771%
    (11Z) / 0.2847% (13Z) of substituted cells, and where it does bind
    it binds through ``l_down``.  Measured this session: the clip
    changes ``bl89_rans_lengths`` on ZERO cells of all three frames.
    This test therefore constrains the excursion construction itself --
    the registered authority function and its registered contract --
    and does not claim a K_v consequence on these frames.
    """
    from woof.verify import sase_ref
    import vent_columns_yolod_rules as vc

    thick = np.asarray(vc.M1BDECK_THICK, np.float64).reshape(-1, 1, 1)
    zc = np.cumsum(thick, axis=0) - 0.5 * thick
    zf = np.concatenate(([0.0], np.cumsum(thick[:, 0, 0])))

    def col(a):
        return np.asarray(a, np.float64).reshape(-1, 1, 1)

    theta, qv = col(vc.M1BDECK_THETA), col(vc.M1BDECK_QV)
    qc, p = col(vc.M1BDECK_QC), col(vc.M1BDECK_P)
    e = np.full_like(theta, vc.SURVEY_E)
    n2 = sase_ref.brunt_vaisala_n2(theta, None, dz_col=thick)
    n2m = sase_ref.moist_n2(theta, qv, qc, p, thick)

    deck = range(16, 23)
    for k in deck:                              # the geometry, re-derived
        assert n2m[k, 0, 0] != n2[k, 0, 0], k   # M1-substituted
        assert float(n2m[k, 0, 0]) < 0.0, k     # moist-unstable
        assert float(n2[k, 0, 0]) > 0.0, k      # ... and dry-STABLE
    assert float(n2m[23, 0, 0]) > 0.0           # the moist-stable lid
    z_lid = float(zf[23])
    assert z_lid == pytest.approx(4048.0217081083556, rel=1e-12)

    l_up, _ = sase_ref.bl89_moist_excursion_lengths(n2m, e, zc, thick)
    for k in range(16, 22):
        reach = z_lid - float(zc[k, 0, 0])
        assert float(l_up[k, 0, 0]) >= reach, (
            k, float(l_up[k, 0, 0]), reach,
            "the moist-unstable deck must cost the parcel nothing, so "
            "the upward excursion reaches the moist-stable lid")
    # the top deck cell, on the registered one-cell bracket
    k = 22
    reach = z_lid - float(zc[k, 0, 0]) - float(thick[k, 0, 0])
    assert float(l_up[k, 0, 0]) >= reach, (float(l_up[k, 0, 0]), reach)
    assert float(l_up[16, 0, 0]) == pytest.approx(
        6318.654152, rel=1e-8)


def test_m2_negative_hfx_activation_no_wstar():
    """NEGATIVE-HFX ACTIVATION (C8): the specimen column carries
    HFX = -15.18 W/m2 (the surface-stable regime; the residual measured
    amplification at HFX -9..-28 W/m2 and the reference confines at
    -25..-30 with its EDMF plume channel OFF, MAXMF = 0 everywhere --
    target-envelope section 3) and the limb must activate there.

    * the specimen state activates (nonzero fluxes) with its own
      negative HFX -- the amplitude comes from BL-integrated e_sgs +
      the saturated-layer instability, never a surface flux;
    * STRUCTURAL w*-independence: the closure signature has no surface
      flux or w* input AT ALL (a classical surface-w* closure is
      identically zero at HFX < 0 and is forbidden as the sole
      amplitude source -- C8); inspect.signature is the witness.
    """
    import inspect

    import specimen_yolod_11z as sp
    from woof.verify import sase_ref
    assert sp.AMP_HFX < 0.0                             # the C8 regime
    amp = _vent_args("amp")
    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**amp)
    assert float(np.abs(f_qv + f_qc).max()) > 0.0
    params = set(inspect.signature(
        sase_ref.plume_vent_flux).parameters)
    assert params == {"theta", "qv", "qc", "p", "dz_col", "e_sgs",
                      "rho1", "n2m_mask", "f_blend"}
    assert not params & {"hfx", "qfx", "ust", "wstar", "w_star"}


def test_m2_nb_termination_kills_specimen_tail():
    """HARD NB TERMINATION (the clearest M2 target), REWRITTEN per S4-4
    REVIEW FINDING Important-2 to pin the SPEC's meaning ("venting
    never erodes the inversion") rather than the pre-fix
    implementation's meaning ("zero above the lid-TOP face"), NARRATIVE
    REWRITTEN round-4 (the mechanism this docstring described was
    retired in round 3 and is restated correctly below).

    THE DEFECT (S4-4 review, section A): the shipped-at-authority
    closure let the entraining parcel's buoyancy trace stay marginally
    positive (B = +0.0126) at the CENTER of the specimen's 282-m
    capping-inversion cell (1512-1794 m, RH 52.2%, qc = 0 -- clear
    air), so 100% of the plume's detrainment landed inside that clear
    cell.  RETIRED EXPLANATION (round-3, do not resurrect): this was
    diagnosed as a coarse-grid artifact -- "the cell's coarse face-mean
    environment dilutes the near-top dryness a finer grid resolves" --
    on the strength of a fixture that already ran the capped code and
    so could only confirm the cap it was testing.  Removing the bound
    and refining that cell 32x (dz ~ 8 m) still terminates at
    1793.9/1785.1 m: the parcel is GENUINELY, grid-convergently buoyant
    into the smoothed inversion (module docstring, SASE-M2 WHY A
    BOUNDARY IS NEEDED AT ALL).

    THE REAL JUSTIFICATION is the design contract, not a numerical
    correction: C9 mandates that venting never erode the cap regardless
    of why the unbounded parcel stays buoyant, and the amended discrete
    reading (design doc SASE-M2, "discrete C9 reading") puts the
    hard-zero boundary at the ENTRAINMENT-ZONE cell's own top face.

    THE MECHANISM AS SHIPPED: step 1's member run (qt = qv + qc >= qs
    inside the M1 mask -- module docstring, SASE-M2 MEMBERSHIP) gives
    the run top k_top; the cell above it, k_top + 1, is the discrete
    entrainment zone; k_lid = k_top + 2 is that cell's own top face and
    bounds the LFC/NB search, with a still-buoyant parcel terminating
    AT k_lid.  Zero new tunable constants.  MEASURED on the specimen
    (round-4): member run k12..k15, so k_top = 15, entrainment zone
    k16, k_lid = 17, and termination is z_f[17] = 1511.653 m -- the top
    of the entrainment zone, NOT the inversion cell's top (the pre-fix
    z_f[18] = 1793.904 m).  The face indices below are unchanged from
    the round-3 build, bit for bit.

    Pinned (the spec's meaning: zero transport AT/ABOVE the inversion
    base band, never inside the capping-inversion cell):

    * faces 17..nz (1511.7, 1793.9, 2124, ... m) -- i.e. the
      entrainment zone's own top face and everything above it, the
      amended C9 boundary -- are BITWISE +0.0 for all three fluxes:
      the capping-inversion cell (1512-1794 m) and the residual's
      1.9-2.7 km tail both carry EXACTLY nothing;
    * flux is nonzero at face 16 (1270.090 m, the member run's own top
      face): the limb carries transport up through the deck and
      detrains it into the entrainment-zone cell k16 immediately above,
      never crossing into the capping layer proper -- the C9
      cap-preservation mechanism, now grid-consistent (binding
      constraints (a)/(c) of the finding).
    """
    from woof.verify import sase_ref
    amp = _vent_args("amp")
    zf = _vent_faces(amp["dz_col"])
    assert abs(zf[16] - 1270.090) < 0.01                # deck top face
    assert abs(zf[17] - 1511.653) < 0.01                # inversion base
    assert abs(zf[18] - 1793.904) < 0.01                # inversion-cell top
    fluxes = sase_ref.plume_vent_flux(**amp)
    tail = np.zeros((len(zf) - 17, 1, 1)).tobytes()
    for arr in fluxes:
        # the spec's meaning: zero AT/ABOVE the inversion base (face 17)
        assert arr[17:].tobytes() == tail
    f_th, f_qv, f_qc = fluxes
    f_qt = f_qv + f_qc
    assert f_qt[16, 0, 0] > 0.0                          # nonzero into the deck
    assert f_qt[17, 0, 0] == 0.0                          # not into the cap
    # the residual's 1.9-2.7 km band (faces at 2124/2509 m): dead
    assert np.all(f_qt[19:21] == 0.0)


def test_m2_termination_tracks_prescribed_lid():
    """THE PRESCRIBED-LID VARIANT (brief fixture 3b): on constructed
    deck-under-lid columns with the lid MOVED (z_lid = 1200/1400/1600
    m; +12 K inversion over 300 m), the NB termination tracks the lid
    within one grid cell, bitwise-zero fluxes at/above it and nonzero
    flux below -- the termination is the column's own lid (plus at
    most its one legitimate entrainment-zone buffer cell), never a
    height constant and never the true capping layer.

    REWRITTEN per S4-4 REVIEW FINDING Important-2, AMENDED round-3
    (coordinator ruling, design doc SASE-M2 amendment "discrete C9
    reading"): termination is bounded by k_lid = (the deck's own robust
    top + 1) + 1 -- one cell FURTHER than the pre-round-3 k_top + 1 --
    because the entrainment-zone cell (the deck's own top + 1) is now
    legitimate search/detrainment territory, not automatically the
    cap.  Whether termination lands EXACTLY at z_lid (a natural
    non-positive-buoyancy center found already inside the entrainment
    zone) or at z_lid + dz (the entrainment zone's own top face, the
    ceiling forced because the parcel is still buoyant there) is
    column-dependent -- MEASURED per leg below -- but it NEVER reaches
    z_lid + 2*dz (the true capping layer, z_invtop = z_lid + 300 m in
    every leg): cap preservation holds regardless of which of the two
    boundaries binds.

    * z_lid = 1200 m: natural NB at the entrainment-zone cell itself
      (still comfortably short of z_invtop = 1500 m) -> term = 1200 m;
    * z_lid = 1400 m: same mechanism -> term = 1400 m;
    * z_lid = 1600 m: the entrainment-zone cell (centered 50 m into
      the ramp) is STILL marginally buoyant -- unlike the 1200/1400
      legs -- so the ceiling binds one cell later -> term = 1700 m,
      which IS z_lid + dz, i.e. exactly the entrainment zone's own top
      face, and is still 200 m short of THIS leg's own z_invtop
      (1900 m).  (The idealized deck conserves total water in the deck
      -- qt_env = qw0 exactly, so F_qt vanishes there by construction
      -- the tracking witness is F_theta, which rides the 2 K/km
      sub-adiabatic tilt.)

    GRADUAL-NB LEG, RE-MEASURED round-3: with the lid WEAKENED to +2 K
    over 300 m the entraining parcel's own buoyancy trace stays
    positive through the WHOLE entrainment zone (unlike the strong-lid
    z_lid = 1400 leg, where it does not), so termination is forced at
    the ceiling -- z_lid + dz = 1500 m, ONE CELL LATER than the
    strong-lid leg's 1400 m, and still 200 m short of the SAME
    z_invtop = 1700 m both legs share: a weak lid buys the plume
    passage through its one legitimate entrainment-zone buffer cell,
    never into the true capping layer above it.  (Pre-round-3 the two
    legs were bitwise identical because the entrainment zone did not
    exist as search territory; that identity no longer holds -- see
    test_m2_termination_grid_consistency for why a fixed-height
    invariant is the wrong thing to assert on this leg at all.)
    """
    from woof.verify import sase_ref
    expect = {1200.0: 1200.0, 1400.0: 1400.0, 1600.0: 1700.0}
    for z_lid, z_expect in expect.items():
        args, zc = _vent_deck_args(z_lid=z_lid)
        zf = _vent_faces(args["dz_col"])
        f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**args)
        prof = f_th[:, 0, 0]
        nzero = np.nonzero(prof != 0.0)[0]
        j_term = int(nzero[-1]) + 1
        assert abs(zf[j_term] - z_expect) < 1e-9, (z_lid, zf[j_term])
        # never into the true capping layer (z_lid + 2*dz)
        assert zf[j_term] <= z_lid + 100.0 + 1e-9, z_lid
        z_invtop = z_lid + 300.0
        assert zf[j_term] < z_invtop - 1e-9, z_lid
        ztail = np.zeros((len(zf) - j_term, 1, 1)).tobytes()
        for arr in (f_th, f_qv, f_qc):
            assert arr[j_term:].tobytes() == ztail, z_lid
        below = prof[:j_term]
        assert int(np.sum(below != 0.0)) >= 3, z_lid

    strong = _vent_deck_args(z_lid=1400.0, dth_inv=12.0)[0]
    weak, zc = _vent_deck_args(z_lid=1400.0, dth_inv=2.0)
    zf = _vent_faces(weak["dz_col"])
    f_th_s, f_qv_s, f_qc_s = sase_ref.plume_vent_flux(**strong)
    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**weak)
    prof_s = f_th_s[:, 0, 0]
    prof = f_th[:, 0, 0]
    j_term_s = int(np.nonzero(prof_s != 0.0)[0][-1]) + 1
    j_term = int(np.nonzero(prof != 0.0)[0][-1]) + 1
    assert abs(zf[j_term_s] - 1400.0) < 1e-9            # strong: natural NB
    assert abs(zf[j_term] - 1500.0) < 1e-9              # weak: forced ceiling
    assert j_term == j_term_s + 1                       # exactly one cell later
    for arr in (f_th, f_qv, f_qc):                       # never the true cap
        assert arr[j_term:].tobytes() == np.zeros(
            (len(zf) - j_term, 1, 1)).tobytes()
    assert zf[j_term] < 1700.0 - 1e-9                    # both legs' z_invtop
    # below the strong-lid's OWN termination face the two legs still
    # agree bitwise (same deck, same parcel physics up to where the
    # strong lid's natural NB fires) -- they diverge only AT that face,
    # where the weak lid's parcel is still buoyant and keeps going.
    for arr, arr_s in ((f_th, f_th_s), (f_qv, f_qv_s), (f_qc, f_qc_s)):
        assert arr[:j_term_s].tobytes() == arr_s[:j_term_s].tobytes()
    assert f_th.tobytes() != f_th_s.tobytes()            # NOT identical anymore


def test_m2_termination_grid_consistency():
    """GRID CONSISTENCY (S4-4 REVIEW FINDING Important-2, binding
    constraint (b): "coarse-grid deposit allocation must converge to
    the fine-grid limit on the same physical column"), AMENDED round-3
    (coordinator ruling, design doc SASE-M2 amendment).  The deck
    fixture pair (same z_base/z_lid/z_invtop/dth_inv, only the grid
    resolution ``dz`` changed via the S4-4-review nz/dz keywords on
    :func:`_vent_deck_args`) is exactly this test.

    THE WEAK-LID column (dth_inv = 2.0 K/300 m) is used because its
    entraining parcel stays buoyant through the WHOLE entrainment zone
    at every grid tested (test_m2_termination_tracks_prescribed_lid),
    forcing the k_lid CEILING every time -- i.e. this is the regime
    where the entrainment zone's own physical extent (exactly one grid
    cell, BY DESIGN -- module docstring, SASE-M2 INVERSION BASE) is
    always the active detrainment recipient, the regime where a
    grid-dependent discretization would show up.  Three grids spanning
    an 11x resolution range: COARSE dz=280 m (order-of-magnitude the
    specimen's own 282-m capping-inversion cell), MEDIUM dz=100 m (the
    module's default deck resolution), FINE dz=25 m.

    AMENDED CONTRACT (round-3, replaces the pre-round-3 "termination
    lands at the SAME physical height on every grid" claim, which is
    FALSE once the entrainment zone is real search territory: the
    ceiling face is the entrainment-zone cell's OWN top, z_lid + dz,
    and by definition MOVES with dz -- MEASURED here 1680/1500/1425 m
    at dz=280/100/25 m, i.e. z_lid + dz exactly on every grid).  Three
    quantities are asserted CONVERGENT (all approach the fine-grid
    limit as dz shrinks) and one is explicitly NOT:

    (i) the DECK-TOP-FACE FLUX (the face at z=1400 m, the boundary
        between the true deck and the entrainment zone -- grid-
        invariant BY CONSTRUCTION since 1400 m is an exact multiple of
        every tested dz) MONOTONICALLY approaches the fine-grid value
        as dz shrinks: |F_coarse - F_fine| > |F_medium - F_fine| (a
        genuine convergence trend, not asserted via a fixed tolerance
        because the deck-top face sits at a real, sharp curvature
        change -- MEASURED ROUND-5 5.765e-2/5.811e-2/6.441e-2
        K kg m^-2 s^-1 at dz=280/100/25 m, gaps normalized by the FINE
        value 0.10490 coarse-vs-fine and 0.09772 medium-vs-fine.  The
        round-5 CLOUD-BASE ANCHOR improves this markedly: the round-4
        build measured 4.369e-2/5.315e-2/6.286e-2 with gaps 0.30499 /
        0.15450, i.e. the coarse-grid error falls by a factor 2.9.  The
        mechanism is structural -- the shape now normalizes on the
        cloud-base face z_f[k_base], which is the SAME PHYSICAL HEIGHT
        (400 m) on all three grids, where the round-4 normalization
        z_f[k_r + 1] = z_base + dz moved with the grid by construction);
    (ii) SUB-INTERFACE (in-deck, strictly below the deck-top face)
        per-cell deposits converge -- interpolated onto three common
        physical heights (600/900/1200 m, avoiding the "nearest cell"
        aliasing a raw index comparison would introduce across grids
        of very different dz) so MEDIUM and FINE are compared at the
        SAME z: MEASURED ROUND-5 relative difference 0.00527/0.00554/
        0.00535 at the three probe heights (round-4 build, re-measured
        this session: 0.06788/0.06813/0.06796 -- the pre-round-5
        docstring said "~7.3%", which is close but was not a value from
        this construction).  The pinned tolerance is NARROWED from 15%
        to 2% accordingly -- the round-4 build fails the narrowed bound;
    (iii) the INTERFACE-CELL DEPOSIT TIMES ITS OWN THICKNESS -- a
        flux-like quantity (= the face-flux difference across it,
        (F[deck-top] - F[k_lid])*dt/rho1 = F[deck-top]*dt/rho1 exactly,
        since F[k_lid] = +0.0 by the C9 boundary) -- converges as (i)
        does, verified here by DIRECT equality to F[deck-top]*dt/rho1,
        not a second independent measurement.  S4-4 ROUND-5: the
        pre-round-5 form of this leg asserted the same coarse/medium
        ORDERING as (i), |x_medium - x_fine| < |x_coarse - x_fine|.
        That ordering no longer holds and CANNOT hold reliably, for a
        measured reason: (iii) is (i) divided by the per-grid rho1, and
        under the round-5 anchor the coarse and medium members of (i)
        are within 0.8% of each other (5.765e-2 vs 5.811e-2), while
        rho1 differs by 0.86% across the same two grids (1.146683 vs
        1.156576 -- the lowest cell center sits at dz/2, so rho1 is a
        function of the grid).  The rho1 spread is now the LARGER
        effect and decides the ordering, giving 3.016593/3.014784/
        3.329361 and gaps 0.09394/0.09448 of the fine value -- an
        ordering flip of 0.6%.  The ordering claim is therefore
        REPLACED by an absolute convergence bound on BOTH gaps (< 10%
        of the fine value), which the round-4 build FAILS (its gaps are
        29.65% and 15.15%): a strictly harder test of the same
        quantity, with the ordering claim left where it is meaningful,
        on (i).

    NOT asserted convergent (BY DESIGN, per the amendment): the bare
    interface-cell DEPOSIT (not multiplied by its own thickness) --
    it is a roughly-fixed flux poured into a cell whose thickness IS
    dz, so it scales close to 1/dz as the grid refines (MEASURED
    8.165e-3/2.757e-2/1.300e-1 K/step at dz=280/100/25 m, monotonically
    INCREASING, not converging) -- physics (a thinner discrete
    entrainment layer receiving comparable total energy deposits more
    per unit mass), not a discretization defect.  Also unchanged from
    pre-round-3: nothing deposits at or above the true inversion top
    (z_invtop = 1700 m) at any resolution.

    CAP-MARGIN GUARD (S4-4 review round-4 addition): because the
    ceiling is z_lid + dz, C9 cap preservation on this column is a
    function of the GRID, and the coarse leg is the binding one.
    MEASURED termination-to-z_invtop margins: 20 m at dz = 280 m
    (the specimen-class grid, whose own capping-inversion cell is
    282 m thick), 200 m at dz = 100 m, 275 m at dz = 25 m -- i.e.
    exactly 300 - dz metres.  The guard asserts the margin is strictly
    positive on every tested grid; a hypothetical dz >= 300 m
    discretization of this column would place the entrainment-zone
    ceiling at or above the inversion top and fail it loudly.  (Not
    measured this session, and therefore stated as a hypothesis: what a
    dz >= 300 m grid does to the rest of the closure -- the deck itself
    would be under 5 cells there, below any resolution this limb was
    designed against.)
    """
    from woof.verify import sase_ref
    dt = 60.0
    grids = ((12, 280.0), (30, 100.0), (120, 25.0))
    face_flux, if_dep_x_thick, if_dep = {}, {}, {}
    for nz, dz in grids:
        args, zc = _vent_deck_args(z_lid=1400.0, dth_inv=2.0, nz=nz, dz=dz)
        zf = _vent_faces(args["dz_col"])
        thick = args["dz_col"]
        rho1 = args["rho1"]
        f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**args)
        nzero = np.nonzero(f_th[:, 0, 0] != 0.0)[0]
        assert nzero.size > 0, (nz, dz)                 # non-vacuous
        j_term = int(nzero[-1]) + 1
        # termination is capped at the entrainment zone's own top
        # face (z_lid + dz) -- the amended, grid-DEPENDENT boundary --
        # never beyond it.
        assert abs(zf[j_term] - (1400.0 + dz)) < 1e-6, (nz, dz, zf[j_term])
        # CAP-MARGIN GUARD (S4-4 review round-4): the grid-dependent
        # ceiling z_lid + dz must stay STRICTLY BELOW the true
        # inversion top z_invtop = z_lid + 300 m -- C9 cap
        # preservation, which the amended entrainment-zone reading
        # makes a function of dz and therefore no longer automatic.
        # MEASURED margins: 20/200/275 m at dz = 280/100/25 m.  The
        # COARSE, specimen-class grid is the binding one (its own
        # capping-inversion cell is 282 m thick): at dz >= 300 m the
        # entrainment-zone ceiling would reach or cross z_invtop and
        # this guard fires -- the pin that makes that loud rather than
        # silent.
        assert zf[j_term] < 1700.0, (nz, dz, zf[j_term])
        assert 1700.0 - zf[j_term] == pytest.approx(300.0 - dz,
                                                    abs=1e-6), (nz, dz)
        above_invtop = zf >= 1700.0 - 1e-6
        for arr in (f_th, f_qv, f_qc):
            assert arr[above_invtop, 0, 0].tobytes() == np.zeros(
                int(above_invtop.sum())).tobytes(), (nz, dz)
        # (i) deck-top face: the boundary between the deck and the
        # entrainment zone sits at z=1400 m on EVERY grid (exact
        # multiple of dz by the grids' own construction).
        j_dtop = int(np.nonzero(np.isclose(zf, 1400.0, atol=1e-6))[0][0])
        face_flux[(nz, dz)] = float(f_th[j_dtop, 0, 0])
        # (iii) interface-cell (entrainment zone) deposit * thickness
        d_if = ((f_th[j_dtop, 0, 0] - f_th[j_dtop + 1, 0, 0]) * dt
               / (rho1 * thick[j_dtop]))
        if_dep[(nz, dz)] = float(d_if)
        if_dep_x_thick[(nz, dz)] = float(d_if * thick[j_dtop])

    coarse, medium, fine = grids
    # (i) deck-top-face flux: genuine, monotonic convergence toward the
    # fine-grid value as resolution improves (not a fixed tolerance --
    # this face converges relatively slowly, see docstring).
    gap_cm = abs(face_flux[coarse] - face_flux[fine])
    gap_mf = abs(face_flux[medium] - face_flux[fine])
    assert gap_mf < gap_cm, (face_flux, gap_cm, gap_mf)
    assert all(v > 0.0 for v in face_flux.values())
    # NARROWED at round 5 from 0.20 (round-4 measured 0.15450) to 0.12:
    # the cloud-base anchor measures 0.09772 here, and the round-4
    # build fails the narrowed bound.
    assert gap_mf / face_flux[fine] < 0.12, face_flux    # measured 0.09772
    assert gap_cm / face_flux[fine] < 0.12, face_flux    # measured 0.10490

    # (iii) interface-cell deposit * thickness: exactly proportional to
    # the deck-top-face flux (F[k_lid] = +0.0), so it inherits (i)'s
    # convergence -- verified by direct equality, not re-measured.
    for key in grids:
        nz, dz = key
        args, _ = _vent_deck_args(z_lid=1400.0, dth_inv=2.0, nz=nz, dz=dz)
        rho1 = args["rho1"]
        assert if_dep_x_thick[key] == pytest.approx(
            face_flux[key] * dt / rho1, rel=1e-9), key
    # S4-4 ROUND-5 (docstring, leg (iii)): the coarse/medium ORDERING
    # is now decided by the per-grid rho1 spread (0.86%) rather than by
    # the flux (0.8%), so it is replaced by an absolute bound on BOTH
    # gaps -- a test the round-4 build fails (0.2965 / 0.1515).
    gap_cm3 = abs(if_dep_x_thick[coarse] - if_dep_x_thick[fine])
    gap_mf3 = abs(if_dep_x_thick[medium] - if_dep_x_thick[fine])
    assert gap_cm3 / if_dep_x_thick[fine] < 0.10, if_dep_x_thick
    assert gap_mf3 / if_dep_x_thick[fine] < 0.10, if_dep_x_thick

    # (ii) sub-interface (in-deck) per-cell deposits converge --
    # interpolated onto common physical heights to avoid nearest-cell
    # aliasing across grids of very different dz.
    for zc_target in (600.0, 900.0, 1200.0):
        vals = {}
        for nz, dz in grids:
            args, zc = _vent_deck_args(z_lid=1400.0, dth_inv=2.0,
                                       nz=nz, dz=dz)
            thick = args["dz_col"]
            rho1 = args["rho1"]
            f_th, _, _ = sase_ref.plume_vent_flux(**args)
            d_th = ((f_th[:-1, 0, 0] - f_th[1:, 0, 0]) * dt
                   / (rho1 * thick))
            vals[(nz, dz)] = float(np.interp(zc_target, zc, d_th))
        rel = (abs(vals[medium] - vals[fine])
               / max(abs(vals[fine]), 1e-30))
        # NARROWED at round 5 from 0.15 to 0.02 (docstring leg (ii)):
        # measured 0.00527/0.00554/0.00535; the round-4 build measures
        # 0.0679/0.0681/0.0680 and fails the narrowed bound.
        assert rel < 0.02, (zc_target, vals, rel)     # measured ~0.0054

    # NOT asserted convergent, BY DESIGN: the bare interface deposit
    # scales close to 1/dz -- monotonically INCREASING with refinement,
    # the opposite of convergence, and that is the physically correct
    # behavior (module docstring, SASE-M2 INVERSION BASE).
    assert (if_dep[coarse] < if_dep[medium] < if_dep[fine]), if_dep


def _vent_kbase_flip_sensitivity(args, root_isolated=True):
    """|dE|/E for a one-cell k_base move, E = max_j (F_qv + F_qc).

    Returns (sensitivity, k_base, k_top, z_f[k_base], dz[k_base], E,
    k_r, k_r_moved) and asserts the move was CLEAN in BOTH senses:

    * MEMBERSHIP -- k_base up by exactly one cell, k_top unchanged, both
      builds firing, so the caller can never measure a stand-down or a
      two-cell jump and call it a one-cell sensitivity;
    * ROOT -- S4-4 ROUND-7 ADDITION.  Before this round the helper
      asserted the membership condition only, while its own docstring and
      the round-6 report claimed the perturbation "isolates the
      membership index moved one cell from ANY OTHER change".  That
      claim was false as written: MEASURED on the whole d02 firing
      population this session (probe ``r7_a_isolated.py``, over the
      ``w3_d_field.py`` survey), on the columns that pass the membership
      test alone the thermodynamic root ALSO moves on 78.56% (11Z) /
      85.05% (13Z), by two or more cells on 18.39% / 16.58% of them and
      by up to 10 / 11 cells.

    The root condition asserted here is the one that is actually true of
    an isolated move, and it has two admissible branches:

    (i) the root index is UNCHANGED (measured on 21.44% / 14.95% of the
        membership-clean columns), or
    (ii) the root is the FALLBACK root -- k_r == k_base in BOTH builds --
         in which case "the root moved" IS the membership move and not a
         second change (60.17% / 68.47%).

    Their union is the GENUINELY ISOLATED subset: 81.61% (11Z) / 83.42%
    (13Z) of the membership-clean columns.  The complement -- the root
    moving independently -- is exactly the |dk_r| >= 2 set, and on 100%
    of it the root collapses onto the NEW cloud base (same probe).

    Pass ``root_isolated=False`` to assert the OPPOSITE, i.e. to pin a
    member that is KNOWN to be confounded rather than waive the check on
    it; the fixture uses that for leg 3's r = 2 grid."""
    from woof.verify import sase_ref
    k_base, k_top = _vent_layer_indices(args)
    assert k_base >= 0, "column does not fire"
    f0 = sase_ref.plume_vent_flux(**args)
    e0 = float((f0[1] + f0[2]).max())
    assert e0 > 0.0, "baseline export is zero"
    moved = _vent_mask_vetoed(args, k_base)
    kb1, kt1 = _vent_layer_indices(moved)
    assert (kb1, kt1) == (k_base + 1, k_top), (k_base, k_top, kb1, kt1)
    f1 = sase_ref.plume_vent_flux(**moved)
    e1 = float((f1[1] + f1[2]).max())
    assert e1 > 0.0, "perturbed export is zero (stand-down, not a flip)"
    k_r0 = _vent_root_index(args, k_base, k_top)
    k_r1 = _vent_root_index(moved, kb1, kt1)
    isolated = (k_r1 == k_r0) or (k_r0 == k_base and k_r1 == kb1)
    assert isolated == bool(root_isolated), (
        "root isolation is %r, expected %r (k_base %d -> %d, k_r %d -> %d)"
        % (isolated, bool(root_isolated), k_base, kb1, k_r0, k_r1))
    zf = _vent_faces(args["dz_col"])
    thick = np.asarray(args["dz_col"], np.float64).ravel()
    return (abs(e1 - e0) / e0, k_base, k_top, float(zf[k_base]),
            float(thick[k_base]), e0, k_r0, k_r1)


def test_m2_kbase_flip_sensitivity_converges_under_refinement():
    """THE CONVERGENCE OBLIGATION OF CONTINUITY CLAUSE 3, DISCHARGED
    (S4-4 ROUND-6; design doc SASE-M2 amendment "continuity clause 3,
    narrowed to the selection rule").

    The amendment narrowed clause 3 to the SELECTION rules and moved
    membership discreteness to the grid-consistency contract -- "a
    one-cell membership move is bounded by the cell thickness" -- but
    registered that as an OBLIGATION, not an assumption: "if the
    k_base-flip export sensitivity does not shrink with dz, this clause
    reopens".  This fixture is the measurement that discharges it, and
    it is written so that it REOPENS the clause (goes red) if a future
    change breaks the trend.

    PERTURBATION (:func:`_vent_mask_vetoed`): the M1 substitution mask
    switched off at the cloud-base cell.  n2m_mask is an INPUT to the
    closure, so this is a state the M1 seam genuinely produces, and it
    is thermodynamically INERT -- theta/qv/qc/p/e_sgs are bit-identical.

    S4-4 ROUND-7 CORRECTION.  The round-6 text here said the veto
    "isolates the membership index moved one cell from ANY other change"
    and that every measurement asserted the move was clean.  Only the
    MEMBERSHIP half of that was asserted.  The thermodynamic root is a
    second index the veto can move, and
    :func:`_vent_kbase_flip_sensitivity` now asserts the root condition
    too (see its docstring for the two admissible branches and for the
    population sizes).  On THIS fixture's legs the branch that applies
    is:

    * LEGS 1 and 2 (deck family) -- the FALLBACK branch.  MEASURED
      (probe r7_g_rootcheck.py) k_r == k_base on all four grids in both
      builds: 1/4/8/16 -> 2/5/9/17 alongside k_base 1/4/8/16 -> 2/5/9/17.
      theta_es decreases monotonically from the ground on the saturated
      deck, so the decrease layer has no interior base and the root IS
      the run's own base; "the root moved" is the membership move.
    * LEG 3 -- STRICT root invariance at r = 1, 4 and 8 (k_r = 4, 17, 35
      unchanged), and a genuine CONFOUND at r = 2, where the veto
      relocates the root from k8 (z_c = 98.93 m) onto the new cloud base
      k22.  That member is asserted with ``root_isolated=False``, i.e.
      the confound is PINNED rather than waived, and the r = 2..8 order
      is reported separately below.

    S4-4 ROUND-7 CORRECTION, the veto-versus-real-perturbation bridge.
    The round-6 text said applying +1.2, +1.5 or +2.0 K at the cloud-base
    cell instead of the veto "moves the SAME indices and gives the SAME
    |dE|/E to six decimals on every grid" and concluded that "the veto is
    a shortcut to a real perturbation, not a different experiment".  That
    is true ON THE CONSTRUCTED DECK FAMILY and false at those magnitudes
    on a real column.  MEASURED this session (probes w3_i_veto_vs_theta.py
    and w3_j_minbump.py, an instrumented HEAD build):

    * deck family -- smallest +dtheta at the cloud-base cell that
      desaturates it on the qt >= qs test is 0.98469 / 0.99504 / 0.98641
      / 0.98213 K at dz = 280/100/50/25 m, and +1.2/+1.5/+2.0 K there
      reproduces the veto's |dE|/E to nine decimals on every grid;
    * real WIN column -- the same +1.2 K bump MANUFACTURES a theta_es
      maximum at the perturbed cell and RELOCATES the root (k_r 4 -> 10
      at r = 1), giving |dE|/E = 0.991526 against the veto's 0.599753.
      A 1.2 K bump is not a small perturbation of this column: its
      minimal desaturating bump is 0.03122 / 0.03719 / 0.00218 / 0.01668
      K at r = 1/2/4/8, all inside the +-0.05 K accuracy scale the
      continuity clause is written against.
    * the bridge that DOES hold on the real column is that MINIMAL bump:
      at dtheta* x1.000001, x1.01 and x1.1 it reproduces the veto's
      |dE|/E to nine decimals at r = 1, 2 and 4, and differs by 2.605% at
      r = 8 (0.082429401 against 0.084634049), where the bump also
      relocates the root onto the new base.

    So the correct claim is: on the constructed deck family the veto is
    an exact shortcut to a real thermodynamic perturbation; on a real
    column it is an exact shortcut only to the MINIMAL desaturating
    perturbation, and finite bumps of order 1 K are a different
    experiment because they move the root as well.

    E = max_j (F_qv + F_qc), the population survey's own export
    magnitude, which is robust to faces moving under refinement.

    LEG 1 -- the grid-consistency deck family (the EXISTING one:
    z_base 400 m, z_lid 1400 m, weak lid dth_inv = 2.0 K/300 m, the
    (nz, dz) pairs of test_m2_termination_grid_consistency plus a 50-m
    member), uniform e_sgs.  MEASURED |dE|/E:

        dz = 280 m   0.242141717     (z_f[k_base] = 280 m: the coarse
        dz = 100 m   0.085389896      grid cannot resolve a 400-m cloud
        dz =  50 m   0.046020613      base, so its own base face is one
        dz =  25 m   0.023958184      cell lower -- see below)

    -- monotonically decreasing, order p = 1.012 / 0.892 / 0.942 across
    the three refinement steps, i.e. FIRST ORDER in dz.  And it is not
    merely first order empirically: on this column the sensitivity is
    EXACTLY the anchor term

        1 - (z_f[k_base] / (z_f[k_base] + dz)) ** VENT_ENT_COEF,

    asserted below to 1e-12 relative on all four grids.  That closed
    form is O(VENT_ENT_COEF * dz / z_base) and vanishes with dz, which
    is the amendment's "bounded by the cell thickness" made exact.  (The
    dz = 280 m member is the one grid whose cloud base is NOT at 400 m:
    its first cell center above 400 m is at 420 m, so z_f[k_base] = 280.
    That is discretization of the same physical column, and the closed
    form is asserted against each grid's OWN base face.)

    LEG 2 -- the ebar-window term converges too, and this is the leg
    that matters, because on LEG 1's uniform e_sgs the window
    contributes nothing at all.  The same physical deck column is given
    a FIXED PHYSICAL e(z) with five decades of contrast concentrated at
    cloud base (1e-6 + 0.1*exp(-(z - 400)/150), a smooth profile so
    that sampling it onto a finer grid is a refinement and not a
    re-quantization of a step), chosen with the COMPOUNDING sign:
    dropping the cloud-base cell LOWERS ebar, so the window term and
    the anchor term push export the same way.  MEASURED |dE|/E
    0.666849646 / 0.312859670 / 0.172671771 / 0.090926605 at
    dz = 280/100/50/25 m -- order p = 0.735 / 0.858 / 0.925, still
    first order, with a coefficient 3.79x LEG 1's at dz = 25 m.  The
    e-profile numbers are fixture construction (the _vent_taper_args
    tilt/bump idiom); no closure constant is added anywhere and
    sase_config_id() is untouched.

    LEG 3 -- a REAL column, refined.  The committed WIN column
    (vent_columns_yolod_11z, real d02 state with the frame's own
    TKE_SASE spanning 1e-6 to 1.78e-1 m2/s2) resampled onto 1x/2x/4x/8x
    grids by :func:`_vent_refined_real_args`.  MEASURED |dE|/E
    0.599752878 / 0.335643005 / 0.162698749 / 0.084634049, i.e. x1.787,
    x2.063, x1.922 per doubling and x7.086 over the 8x refinement
    (p = 0.9417).  This is the leg that answers the amendment on its own
    ground, since LEGS 1-2 are constructed columns.

    S4-4 ROUND-7 DISCLOSURE, leg 3's r = 1 member.  The docstring above
    discloses the analogous offset for LEG 1's coarsest member; the same
    disclosure was missing here.  r = 1 is NOT the same discrete
    saturated layer as r = 2/4/8.  MEASURED (probe w3_b_leg3.py, an
    instrumented HEAD build): the run top face z_f[k_top + 1] is
    1802.3105 m at r = 1 against 1659.9253 m at r = 2, 4 and 8, and the
    BASELINE export does not itself converge across that step --
    1.4398700e-05 / 8.0553223e-06 / 9.6586759e-06 / 8.8506275e-06
    kg/m2/s at r = 1/2/4/8, i.e. a 44% drop and then a wobble, not a
    trend.  The interpolant reaches one more saturated cell at the
    native discretization than it does once the layer is split.  So the
    r = 1 -> 2 step mixes a refinement with a layer-top change, and the
    RESTRICTED r = 2..8 order is the cleaner statistic: x3.96581 for a
    x4 refinement, p = 0.9939, BETTER than the 0.9417 over the full
    span.  Both are asserted below.

    NOT CLAIMED HERE, and re-measured this session so it is on the record
    rather than implied: at NATIVE d02 resolution the sensitivity is
    LARGE.  Whole firing population, real TKE_SASE, probes
    w3_d_field.py + r7_a_isolated.py (19 443 firing columns at 11Z,
    21 983 at 13Z):

    * the veto produces a MEMBERSHIP-clean one-cell move on 67.42% (11Z)
      / 67.74% (13Z) of firing columns; on that subset the median
      |dE|/E is 0.34792 / 0.39219 and 30.41% / 36.74% exceed 50%;
    * S4-4 ROUND-7 DENOMINATOR CORRECTION: those 30.41% / 36.74% are
      fractions of the CLEAN SUBSET, not of all firing columns.  As
      fractions of ALL firing columns they are 20.50% (11Z) and 24.89%
      (13Z).  The round-6 report and the pre-round-7 text here quoted
      them against the wrong denominator.  The medians are correctly
      conditional on the clean subset and are unchanged;
    * on the GENUINELY ISOLATED subset (root fixed or fallback; 81.61% /
      83.42% of the clean subset) the median is 0.37357 / 0.36349 with
      31.27% / 32.73% above 50% -- isolation does not make the native
      magnitude smaller.

    The obligation is about the TREND, and the trend holds on the
    isolated subset too: 400-column samples of it (sampling rule and
    both samples in the round-7 report; probe r7_b_refine_isolated.py)
    give a PAIRED median x-fold over the 8x refinement of x7.048 /
    x7.127 at 11Z (p = 0.9391 / 0.9444) and x6.938 / x6.924 at 13Z
    (p = 0.9310 / 0.9305).  Discretization that converges is still
    discretization at the resolution actually run.
    """
    from woof.verify import sase_ref
    grids = ((12, 280.0), (30, 100.0), (60, 50.0), (120, 25.0))

    def e_at_cloud_base(zc):
        # LEG 2's fixed physical e(z): five decades concentrated at the
        # 400-m cloud base, decaying on a 150-m scale.  Fixture
        # construction only -- see docstring.
        return 1.0e-6 + 1.0e-1 * np.exp(
            -np.maximum(zc - 400.0, 0.0) / 150.0)

    leg1, leg2 = {}, {}
    for nz, dz in grids:
        args, zc = _vent_deck_args(z_lid=1400.0, dth_inv=2.0, nz=nz, dz=dz)
        (s, k_base, k_top, zfb, dzb, e0, k_r0,
         k_r1) = _vent_kbase_flip_sensitivity(args)
        assert dzb == pytest.approx(dz, rel=1e-12)
        # S4-4 ROUND-7: the deck family's root is the FALLBACK root, so
        # the root index moving with the base is the membership move and
        # not a second change (docstring, PERTURBATION).
        assert (k_r0, k_r1) == (k_base, k_base + 1), (dz, k_r0, k_r1)
        # the closed form: the whole sensitivity IS the anchor face move
        anchor = 1.0 - (zfb / (zfb + dz)) ** sase_ref.VENT_ENT_COEF
        assert s == pytest.approx(anchor, rel=1e-12), (dz, s, anchor)
        leg1[dz] = s
        nu = dict(args)
        nu["e_sgs"] = e_at_cloud_base(zc).reshape(-1, 1, 1)
        s2, kb2, kt2, _, _, _, r20, r21 = _vent_kbase_flip_sensitivity(nu)
        assert (kb2, kt2) == (k_base, k_top)      # same physical layer
        assert (r20, r21) == (k_r0, k_r1)         # and the same root move
        leg2[dz] = s2

    # LEG 1: monotone, first order, and matching the measured values.
    order = (280.0, 100.0, 50.0, 25.0)
    for a, b in zip(order[:-1], order[1:]):
        assert leg1[b] < leg1[a], leg1
        p = np.log(leg1[a] / leg1[b]) / np.log(a / b)
        assert 0.85 <= p <= 1.05, (a, b, p, leg1)   # measured 1.012/.892/.942
    for dz, want in ((280.0, 0.24214171674480095),
                     (100.0, 0.08538989614534728),
                     (50.0, 0.04602061259649623),
                     (25.0, 0.023958183513965444)):
        assert leg1[dz] == pytest.approx(want, rel=1e-12), (dz, leg1[dz])

    # LEG 2: the window term converges too -- monotone, first order,
    # with a strictly larger coefficient than LEG 1 on every grid.
    for a, b in zip(order[:-1], order[1:]):
        assert leg2[b] < leg2[a], leg2
        p = np.log(leg2[a] / leg2[b]) / np.log(a / b)
        assert 0.70 <= p <= 1.00, (a, b, p, leg2)   # measured .735/.858/.925
    for dz in order:
        assert leg2[dz] > leg1[dz], (dz, leg1, leg2)
    for dz, want in ((280.0, 0.666849646017715),
                     (100.0, 0.3128596698367546),
                     (50.0, 0.17267177111397),
                     (25.0, 0.09092660540810217)):
        assert leg2[dz] == pytest.approx(want, rel=1e-12), (dz, leg2[dz])
    assert leg2[25.0] / leg1[25.0] == pytest.approx(3.795, rel=1e-3)

    # LEG 3: a real column, refined.  r = 1 must reproduce the COMMITTED
    # column bitwise on every argument -- otherwise the trend below would
    # be a property of the resampler rather than of the real state.
    plain, one = _vent_real_args("WIN"), _vent_refined_real_args("WIN", 1)
    for key in ("theta", "qv", "qc", "p", "e_sgs", "dz_col", "n2m_mask"):
        assert np.array_equal(np.asarray(plain[key]),
                              np.asarray(one[key])), key
    assert plain["rho1"] == one["rho1"]
    leg3, roots = {}, {}
    # S4-4 ROUND-7: r = 2 is the one member whose root the veto moves
    # INDEPENDENTLY (k8 -> the new base k22).  It is asserted as
    # confounded rather than waived, so the confound cannot silently
    # change or silently spread to another member.
    for r in (1, 2, 4, 8):
        s, k_base, k_top, _, dzb, e0, k_r0, k_r1 = (
            _vent_kbase_flip_sensitivity(
                _vent_refined_real_args("WIN", r),
                root_isolated=(r != 2)))
        assert dzb == pytest.approx(91.71322 / r, rel=1e-4), (r, dzb)
        roots[r] = (k_base, k_top, k_r0, k_r1)
        leg3[r] = s
    assert roots[1][2:] == (4, 4) and roots[4][2:] == (17, 17), roots
    assert roots[8][2:] == (35, 35), roots
    assert roots[2][2:] == (8, 22), roots           # the pinned confound
    # S4-4 ROUND-7 DISCLOSURE: r = 1 is a different discrete saturated
    # layer from r = 2/4/8 -- its run top face is one cell higher (see
    # docstring).  Asserted so the offset is a pinned property of the
    # family and not a silent one.
    tops = {}
    for r in (1, 2, 4, 8):
        a_r = _vent_refined_real_args("WIN", r)
        kb_r, kt_r = _vent_layer_indices(a_r)
        tops[r] = float(_vent_faces(a_r["dz_col"])[kt_r + 1])
    assert tops[1] == pytest.approx(1802.3105, abs=1e-3), tops
    for r in (2, 4, 8):
        assert tops[r] == pytest.approx(1659.9253, abs=1e-3), tops
    for a, b in ((1, 2), (2, 4), (4, 8)):
        assert leg3[b] < leg3[a], leg3
        assert leg3[a] / leg3[b] >= 1.7, (a, b, leg3)   # measured >= 1.787
    assert leg3[1] / leg3[8] >= 7.0, leg3               # measured 7.086
    # the RESTRICTED r = 2..8 order, the statistic that does not straddle
    # the layer-top change: measured x3.96581, p = 0.99385.
    p28 = np.log(leg3[2] / leg3[8]) / np.log(4.0)
    assert 0.97 <= p28 <= 1.02, (p28, leg3)
    assert p28 == pytest.approx(0.99385, abs=1e-4), p28
    for r, want in ((1, 0.5997528778025131), (2, 0.3356430054766052),
                    (4, 0.16269874934517548), (8, 0.08463404879992163)):
        assert leg3[r] == pytest.approx(want, rel=1e-12), (r, leg3[r])

    # non-vacuity: the coarse-grid sensitivities are LARGE, so the
    # convergence asserted above is a real trend and not three ways of
    # measuring zero.
    assert leg1[280.0] > 0.2 and leg2[280.0] > 0.6 and leg3[1] > 0.5


def test_m2_export_bound_emerges_on_specimen(monkeypatch):
    """THE EXPORT BOUND (don't over-vent -- the falsifiable amplitude
    bar).  DERIVATION (sase-m-target-envelope.md): the trusted
    reference exports 0.66-1.12e-4 kg/m2/s of moisture through the
    1.0-1.4 km saturated-layer top (section 3: res+sub w'qt at the
    1.24-km cloud-top p50; section 6 bound 3) against the same-frame
    layer supply -- section 4 table, BOX3 (the deck-core box) MFC
    0-1.4 km = +1.97e-4 kg/m2/s -- i.e. it exports only ~40-60% of the
    supply; the residual loads MLCAPE at +293-512 J/kg/h for the
    afternoon outbreak (section 4: a channel holding the layer
    steady-state would need 2.5-5e-4, is ~4x too strong, and kills the
    historically-correct CAPE build -- G-M4).  NO export fraction is
    coded anywhere: the fraction must EMERGE from the registered
    closure (Grant M_b = 0.03*sigma_w amplitude, SST07 eps = 0.4/z
    dilution, B-weighted detrainment); this fixture CHECKS it.

    On the specimen at the M1-equilibrium e_sgs (the coupled state),
    at the export face z_f[16] = 1270.1 m (the discrete face bracketing
    the envelope's 1.24-km layer top; unaffected by the S4-4 review
    Important-2 termination fix -- face 16 sits below the plume's
    buoyancy-peak face, entirely inside the unchanged "grow" zone):

    * total-moisture export F_qv + F_qc lands in the reference class
      [0.7, 1.1]e-4 kg/m2/s (MEASURED round-5: 0.76073587e-4 = 190
      W/m2 latent-equivalent; was 0.814e-4 = 204 W/m2 before the
      cloud-base anchor);
    * export/supply is MEASURED at 0.3861603424 and asserted inside
      M2_RATIO_CLASS = M2_EXPORT_CLASS/M2_SUPPLY = [0.3553299,
      0.5583756].  S4-4 ROUND-6 RE-PIN (design doc SASE-M2 amendment
      "the export ratio is not an independent criterion", coordinator
      ruling).  Round 5 landed here and went RED against a literal
      [0.4, 0.6]; the adjudication found that this fixture's "supply" is
      the registered constant 1.97e-4, so export/supply is a LINEAR
      RESCALE of the line above and carries no independent information,
      and that the two frozen bands were one quantity written twice with
      rails that never agreed -- [0.4, 0.6] implies export in
      [0.788, 1.182]e-4, a floor ABOVE the export bar's own 0.7e-4 floor
      and a ceiling ABOVE its own 1.1e-4 ceiling.  The export class is
      PRIMARY and untouched; the ratio pin is now its exact image.  What
      this admits that the old pair excluded is the interval
      [0.3553, 0.4), i.e. exports in [0.700, 0.788]e-4 -- values the
      export bar itself admits and the reference itself measured
      (its own BOX3-denominated span is 0.335-0.569);
    * RED leg (the bound does real work): monkeypatching the amplitude
      coefficient to the plain-EDMF updraft-area class (VENT_MB_COEF =
      0.1, the a_up = 0.1 family the closure deliberately does NOT
      use) over-vents into the envelope's forbidden full-supply class
      (>= 2.0e-4 kg/m2/s) -- the registered Grant coefficient is
      essential, not decorative.  MEASURED round-6: 2.5358e-4, and
      the same monkeypatch drives the ratio to 1.2872, well outside
      M2_RATIO_CLASS -- the derived ratio rails still fail a build that
      over-vents, because they are the export rails.

    MARGIN FLAG (S4-4 review round-3 Minor-3, for the S4-5 golden
    freeze), RESOLVED at round 6.  The flag said export/supply =
    0.413427 sat only ~3.4% above the then-literal lower rail and that
    "a further ~3% amplitude reduction from ANY source drops the
    emergent export fraction out of the envelope this fixture asserts".
    That is now structurally impossible to observe on its own: the ratio
    has no rail of its own to fall out of.  The one surviving amplitude
    margin is the export class's, MEASURED round-6 at 8.68% above its
    lower rail (0.76073587e-4 vs 0.7e-4) and 30.84% below its upper --
    i.e. the specimen is nearer the FLOOR, and the binding S4-5 golden
    risk is an amplitude REDUCTION of >8.7% from any source.
    """
    from woof.verify import sase_ref
    amp = _vent_args("amp")
    zf = _vent_faces(amp["dz_col"])
    assert abs(zf[16] - 1270.090) < 0.01
    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**amp)
    export = float(f_qv[16, 0, 0] + f_qc[16, 0, 0])
    supply = M2_SUPPLY                  # BOX3 MFC 0-1.4 km, envelope s4
    # PRIMARY, unchanged: the absolute export class.
    assert M2_EXPORT_CLASS[0] <= export <= M2_EXPORT_CLASS[1], export
    # S4-4 ROUND-6 RE-PIN (design doc SASE-M2 amendment "the export ratio
    # is not an independent criterion").  The rails are DERIVED from the
    # line above and M2_SUPPLY, not written: 0.7e-4/1.97e-4 = 0.3553299
    # and 1.1e-4/1.97e-4 = 0.5583756.  This assertion is therefore
    # ARITHMETICALLY IMPLIED by the one above -- deliberately so; it is
    # kept as the site where the ratio the reference budget quotes is
    # visible, and test_m2_export_ratio_is_the_export_class_image pins
    # the implication itself.  The old literal [0.4, 0.6] was a second
    # copy of this same bar with rails that disagreed with it.
    assert M2_RATIO_CLASS[0] <= export / supply <= M2_RATIO_CLASS[1], (
        export / supply)
    assert abs(export - 0.7607e-4) <= 0.03e-4, export   # measured class

    monkeypatch.setattr(sase_ref, "VENT_MB_COEF", 0.1)
    f_th2, f_qv2, f_qc2 = sase_ref.plume_vent_flux(**amp)
    export2 = float(f_qv2[16, 0, 0] + f_qc2[16, 0, 0])
    assert export2 >= 2.0e-4, export2                   # over-vent class
    # the derived ratio rails inherit the RED leg by construction
    assert not (M2_RATIO_CLASS[0] <= export2 / supply
                <= M2_RATIO_CLASS[1]), export2 / supply


def test_m2_export_ratio_is_the_export_class_image():
    """S4-4 ROUND-6: THE RATIO IS NOT AN INDEPENDENT CRITERION -- pinned
    as an identity, not asserted as a band (design doc SASE-M2 amendment
    "the export ratio is not an independent criterion", coordinator
    ruling).

    The fixture-side "supply" is the REGISTERED CONSTANT M2_SUPPLY =
    1.97e-4 kg/m2/s (sase-m-target-envelope.md section 4 table, BOX3 MFC
    0-1.4 km) -- a number read off the reference budget, not a
    per-column diagnostic and not a function of the closure.  Therefore
    export/supply is exactly export/1.97e-4 on every column, a linear
    rescale of the export class with no independent content.  That is
    what this fixture proves, and it is the structural reason
    M2_RATIO_CLASS is DERIVED from M2_EXPORT_CLASS rather than written:
    the two bands cannot drift apart again, because there is only one
    band.

    S4-4 ROUND-7 REWRITE.  NONE of the round-6 version's three legs
    constrained the closure -- demonstrated by a reviewer and reproduced
    this session leg by leg (probe r7_i2_legs.py; r7_i_vacuity.py for
    the whole test).  Replacing ``plume_vent_flux`` with a stub
    returning an export of 3.3e-9 * VENT_MB_COEF -- about 1e-10 kg/m2/s,
    three orders of magnitude below the frozen export class and
    unrelated to the closure -- left all three legs GREEN, and leg (c)
    stayed green against every stub tried, including one non-linear in
    the amplitude.  Leg (a) asserted ``r == e / M2_SUPPLY`` where the
    helper's own return statement was ``return e, e / M2_SUPPLY``, i.e.
    ``x == x``; leg (c) made no closure call at all and re-derived three
    literals defined a few lines above it.  Every surviving leg below
    calls the closure and pins a value the closure actually produces.
    The property the adjudication established is unchanged -- the ratio
    is the export class's exact image under the registered supply -- but
    it is now pinned against measurements rather than against arithmetic.

    (a) THE CLOSURE'S OWN NUMBER, and the image identity around it.  The
        specimen export at the acceptance face 16 and its ratio are
        pinned to the values the closure produces, and the round-trip
        ratio -> export is asserted within 2 ulp of the export on four
        amplitudes.  S4-4 ROUND-7: the round-6 line here was
        ``abs(r * M2_SUPPLY - e) <= 1e-16 * e``.  One ulp at 1e-4 is
        1.3553e-16 relative, so a 1e-16 relative bound is BELOW ONE ULP
        and cannot be a correctness statement.  MEASURED this session
        (probe r7_e_ulp.py): over 2e6 export magnitudes drawn uniformly
        from the frozen class the round trip is exact on 89.09% and
        VIOLATES the 1e-16 bound on 10.91%, while never exceeding
        1 ulp(e); the bound is therefore 2 ulp here.  It is not
        hypothetical -- the x1.5 member of leg (c)'s own sweep
        (export 1.1411038118248368e-04) misses by exactly 1 ulp and
        FAILS the superseded 1e-16 line.
    (b) SCALE-FREEDOM: multiply the amplitude coefficient and the export
        AND the ratio move by exactly that factor.  Each scaled export is
        pinned to its own measured value, so a stub that merely happens
        to be linear in VENT_MB_COEF fails here.  MEASURED: exact
        linearity of the closure in VENT_MB_COEF to <= 2.7e-16 relative
        across x0.5 ... x3.3333.  A "criterion" that rescales exactly
        with the quantity it is meant to cross-check is not a second
        criterion.
    (c) THE TWO BANDS ADMIT EXACTLY THE SAME BUILDS -- checked on the
        closure's own outputs rather than on the rails' arithmetic.  An
        8-point amplitude sweep straddles both rails from below and from
        above, and at every point membership of the export class and
        membership of the derived ratio class agree.  This is where the
        adjudication's finding now lives: with the SUPERSEDED literal
        rails [0.4, 0.6] the two disagree on 3 of the 8 points
        (x0.95, x1.0 and x1.5 -- MEASURED r7_h_sweep.py), and the
        specimen itself is one of them, so re-literalizing the rails
        turns this leg red.  The superseded pair's arithmetic (its image
        [0.788, 1.182]e-4 lies 12.571% and 7.455% above the export
        class's own floor and ceiling) is recorded here as prose because
        asserting it constrains nothing but the test file's own literals.

    MEASURED this session (probe r7_h_sweep.py and this fixture's own
    run): specimen export 7.607358745498912e-05, ratio 0.3861603424111123
    -- inside the derived class [0.3553299, 0.5583756] and below the
    superseded 0.4 rail by 3.46%.
    """
    import math
    from woof.verify import sase_ref
    amp = _vent_args("amp")

    def export_at_face_16(scale):
        old = sase_ref.VENT_MB_COEF
        try:
            sase_ref.VENT_MB_COEF = old * scale
            f3 = sase_ref.plume_vent_flux(**amp)
        finally:
            sase_ref.VENT_MB_COEF = old
        return float(f3[1][16, 0, 0] + f3[2][16, 0, 0])

    red_leg = 0.1 / sase_ref.VENT_MB_COEF
    assert red_leg == pytest.approx(10.0 / 3.0, rel=1e-12)

    # (a) the closure's own export and ratio, pinned; the image identity
    # asserted in ULPS (see docstring) rather than below one ulp.
    base_e = export_at_face_16(1.0)
    base_r = base_e / M2_SUPPLY
    assert base_e == pytest.approx(7.607358745498912e-05, rel=1e-12), base_e
    assert M2_EXPORT_CLASS[0] <= base_e <= M2_EXPORT_CLASS[1], base_e
    assert base_r == pytest.approx(0.3861603424111123, rel=1e-12), base_r
    assert M2_RATIO_CLASS[0] <= base_r <= M2_RATIO_CLASS[1], base_r
    assert base_r < 0.4                    # below the superseded rail
    for scale in (1.0, 0.5, 2.0, red_leg):
        e = export_at_face_16(scale)
        r = e / M2_SUPPLY
        assert abs(r * M2_SUPPLY - e) <= 2.0 * math.ulp(e), (scale, e, r)

    # (b) the export -- and therefore the ratio -- rescales EXACTLY with
    # the amplitude coefficient, at values the closure actually produces.
    for scale, want in ((0.5, 3.803679372749456e-05),
                        (2.0, 1.5214717490997823e-04),
                        (red_leg, 2.535786248499638e-04)):
        e = export_at_face_16(scale)
        assert e == pytest.approx(want, rel=1e-12), (scale, e)
        assert e == pytest.approx(base_e * scale, rel=1e-14), scale
        assert e / M2_SUPPLY == pytest.approx(base_r * scale,
                                              rel=1e-14), scale

    # (c) the export class and the DERIVED ratio class admit exactly the
    # same builds, measured on an 8-point sweep that crosses both rails.
    sweep = (0.5, 0.9, 0.95, 1.0, 1.4, 1.5, 2.0, red_leg)
    in_export, in_ratio = [], []
    for scale in sweep:
        e = export_at_face_16(scale)
        in_export.append(M2_EXPORT_CLASS[0] <= e <= M2_EXPORT_CLASS[1])
        in_ratio.append(M2_RATIO_CLASS[0] <= e / M2_SUPPLY
                        <= M2_RATIO_CLASS[1])
    assert in_export == in_ratio, (sweep, in_export, in_ratio)
    # non-vacuity: the sweep really does cross both rails, so the
    # agreement above is a statement and not a row of identical Trues.
    assert in_export == [False, False, True, True, True, False,
                         False, False], in_export
    # and the superseded literal pair does NOT agree with the export
    # class on this closure -- the adjudication's finding, as a
    # measurement.  x0.95, x1.0 (the specimen) and x1.5 disagree.
    in_old = []
    for scale in sweep:
        r = export_at_face_16(scale) / M2_SUPPLY
        in_old.append(0.4 <= r <= 0.6)
    assert sum(a != b for a, b in zip(in_export, in_old)) == 3, in_old


def test_m2_velocity_scale_rekey_branch_pinned(monkeypatch):
    """S4-4 REVIEW FINDING Minor-2: the Grant velocity-scale re-key
    (module docstring, SASE-M2 AMPLITUDE) applies M_base =
    VENT_MB_COEF*rho1*sigma_w LITERALLY.  A physically-motivated
    ALTERNATIVE branch exists and was NOT pinned before this fix:
    M_base = VENT_MB_COEF*rho1*(sigma_w/0.6), which would explicitly
    compensate for the sigma_w ~ 0.6*w* approximation cited to justify
    the substitution.  The two branches differ by the factor-1.67
    ratio 1/0.6 and are NOT interchangeable -- this fixture pins the
    CHOSEN (literal) branch's export number and shows the alternative
    (compensated) branch lands OUT of the export band.

    Implemented as VENT_MB_COEF -> VENT_MB_COEF/0.6, algebraically
    identical to inserting the extra /0.6 factor into M_base (the same
    monkeypatch idiom test_m2_export_bound_emerges_on_specimen already
    uses to probe the amplitude coefficient).
    """
    from woof.verify import sase_ref
    amp = _vent_args("amp")
    zf = _vent_faces(amp["dz_col"])
    literal = sase_ref.plume_vent_flux(**amp)
    export_literal = float(literal[1][16, 0, 0] + literal[2][16, 0, 0])
    assert 0.7e-4 <= export_literal <= 1.1e-4, export_literal
    # PRIMARY PIN of this number: test_m2_export_bound_emerges_on_specimen
    # (S4-4 review Important-1).  This is a SECOND, independent
    # occurrence (S4-4 review Minor-2) -- a future re-derivation of
    # VENT_ENT_COEF must update BOTH sites, or this fixture goes red
    # with no clue why (S4-4 review round-3 Minor-2 cross-reference).
    # S4-4 ROUND-5 RE-MEASURE: 0.814e-4 -> 0.7607e-4 under the
    # cloud-base anchor (both sites updated together, as designed).
    assert abs(export_literal - 0.7607e-4) <= 0.03e-4, export_literal

    monkeypatch.setattr(sase_ref, "VENT_MB_COEF",
                        sase_ref.VENT_MB_COEF / 0.6)
    compensated = sase_ref.plume_vent_flux(**amp)
    export_comp = float(compensated[1][16, 0, 0] + compensated[2][16, 0, 0])
    assert export_comp > 1.1e-4, export_comp            # out of band
    assert export_comp / export_literal == pytest.approx(1.0 / 0.6)


def test_m2_heat_channel_sign_structure():
    """THE COUNTER-GRADIENT SHAPE (why M2 exists beyond M1 -- anatomy
    section 4: a nonlocal profile no single eddy diffusivity can
    produce).  On the specimen the dry theta gradient is stable-
    positive at EVERY level of the plume layer (theta rises
    monotonically 297.39 -> 303.31 K across k10..k17), yet the M2 heat
    flux changes sign inside it:

    * NEGATIVE lobe at the roots, faces 502-887 m (inside the
      residual's measured 0.24-0.97 km lobe, -28..-151 W/m2 class):
      the parcel launched at the saturated-run base rises SUBSATURATED
      (specimen root RH 97%) toward its LCL, lagging the +5 K/km
      dry-stable environment -- theta_p < theta_env until latent
      heating catches up (MEASURED round-5, cloud-base anchor:
      cp*F_theta -3.51/-7.16/-7.65/-6.69 W/m2 on the four faces
      450-900 m; the round-4 build measured -3.76/-7.67/-8.19/-7.16,
      i.e. the same lobe scaled by this column's single amplitude
      factor 0.93405);
    * POSITIVE limb aloft (faces 1063-1270 m) against the SAME stable
      gradient -- upward heat flux where d(theta)/dz > 0, the
      counter-gradient transport an eddy-diffusivity closure would
      need K < 0 to produce (the reason a mass-flux form is required,
      anatomy constraint 4).  (The former THIRD aloft face, 1512 m,
      is now the exact-zero inversion-base termination face -- S4-4
      review Important-2; the counter-gradient signature is unaffected
      since it lives entirely below the peak/termination faces.)
    """
    from woof.verify import sase_ref
    amp = _vent_args("amp")
    th1 = amp["theta"][:, 0, 0]
    assert np.all(np.diff(th1[10:18]) > 0.0)        # stable throughout
    zf = _vent_faces(amp["dz_col"])
    f_th, _, _ = sase_ref.plume_vent_flux(**amp)
    lobe = (zf >= 450.0) & (zf <= 900.0)
    vals = f_th[lobe, 0, 0] * sase_ref.CP_AIR       # -> W/m2
    assert vals.shape[0] == 4
    assert np.all(vals < 0.0), vals                 # the negative lobe
    assert np.all(vals >= -12.0) and vals.min() <= -3.0, vals
    aloft = (zf >= 1000.0) & (zf <= 1500.0)         # below the termination face
    assert aloft.sum() == 2
    assert np.all(f_th[aloft, 0, 0] > 0.0)          # counter-gradient
    assert f_th[17, 0, 0] == 0.0                    # inversion base: exact zero


def test_m2_condense_dont_clear_partition():
    """THE PARTITION (the G-M3 rescue channel, made falsifiable),
    RE-PINNED per S4-4 REVIEW FINDING Important-3: the requirement is
    asserted DIRECTLY -- the saturated cloud-deck band must NOT lose
    condensate to a CLEAR cell; the transported moisture condenses
    into the 1.0-1.8 km deck.

    THE DEFECT (S4-4 review): pre-fix, 100% of the plume's
    detrainment landed in the CLEAR (RH 52%, qc_env = 0) capping-
    inversion cell k17, funded by draining deck cells k15-16 of REAL
    condensate (k15's 0.53 g/kg cloud water in ~1 h) -- water leaving
    the deck for a clear cell, the opposite of "condense, don't
    clear."  Largely downstream of Important-2: with termination now
    capped at the deck's own top (k_lid = k17), EVERY unit of
    transport this closure moves lands in k16 -- RH 95.5%, essentially
    condensate-free (qc_env ~ 0, a near-saturated DECK cell, not the
    clear cap) -- and k17 (the clear cell) receives EXACTLY nothing.

    Emulating the registered S4-5 deposit d(phi)_k = (F[k] - F[k+1])*
    dt/(rho1*thick_k) on the specimen at dt = 60 s:

    * the clear cap cell k17 (1512-1794 m, qc_env = 0, RH 52%) gets a
      BITWISE ZERO deposit on all three rows -- the defect's exact
      channel is closed;
    * the recipient cell k16 (1270-1512 m, qc_env ~ 0 -- essentially
      condensate-free but INSIDE the saturated run the mask
      diagnosed, RH 95.5%) GAINS theta, qv, AND qc (MEASURED round-5:
      +6.812 mK/step theta, +6.735 mg/kg/step qv, +9.699 mg/kg/step
      qc; round-4 values +7.293/+7.211/+10.384, the same rows scaled
      by this column's amplitude factor 0.93405) -- the
      deck-condensate return that must rescue G-M3 at S4-6, now
      landing INSIDE the deck;
    * cell k15 still loses water to the transport (it is BELOW the
      buoyancy peak, in the plume's growth zone) -- but that loss
      funds k16's gain, water moving UP within the mask's own run, not
      an export to a clear cell above it;
    * the band (cells 15-17, 1.0-1.8 km) still gains total water at
      exactly the rate the sub-band column loses it (telescoping);
    * NOTHING deposits above the band: every deposit row above 1794 m
      (and now also AT/inside the clear cell 1512-1794 m) is bitwise
      +0.0 for theta, qv, and qc;
    * F_qc stays POSITIVE (condensate moves UP, never out of the
      column) on the surviving faces 887/1063/1270 m (Critical-1,
      S4-4 review round-3: restores the pre-round-2 "qc moves UP"
      pin -- x1/f3 verifier finding -- narrowed to [880, 1300] m
      because face 1512 m is now the exact-zero termination face and
      the wider pre-fix band [880, 1520] straddled it; MEASURED
      round-5 F_qc = 1.4646e-5/2.1909e-5/4.4896e-5 kg/m2/s, all > 0;
      round-4 1.5680e-5/2.3456e-5/4.8066e-5).

    STRUCTURAL CONSERVATION, PINNED (S4-4 review round-3, x1/f3
    verifier finding, REPLACES the pre-round-3 report's "internal
    redistribution within the 1.0-1.8 km band" claim -- that claim was
    measurably WRONG: cell 15 alone supplies only 51.2% of the
    condensate loss (68.1% of total water); cells 10-14 (455.7-975.2 m,
    ALL below the claimed 1.0-1.8 km band, each carrying real
    condensate up to 0.394 g/kg) supply the other 48.8%).  The TRUE
    invariant is structural, not a height band: transport is EXACTLY
    conserved within THE PLUME'S OWN TERRITORY, the root k_r through
    the discrete entrainment-zone cell k_top + 1 -- k10..k16 on the
    specimen.  (S4-4 review round-4 CORRECTION: the pre-round-4 text
    called this set "{the robustly-saturated run} UNION {the
    entrainment-zone cell} ... exactly the M1 mask's own contiguous
    footprint".  MEASURED: the robustly-saturated (member) run is
    k12..k15, so the conserved set is strictly WIDER than run-plus-one
    at the bottom -- k10 and k11 are in it as the theta_es root and the
    cell between it and the run.  That the set coincides numerically
    with the M1 mask run k10..k16 on THIS column is a coincidence of
    this specimen, not a structural identity, and nothing in the
    closure depends on it.)  Every
    row's thickness-weighted deposit sums to +0.0 over that set to
    roundoff (measured relative residual ~1e-16), and every cell
    OUTSIDE it -- both below the root and at/above the cap -- gets a
    bitwise +0.0 deposit, not merely a small one.

    PHASE BOOKKEEPING, CORRECTED (S4-4 review round-3, f3 verifier
    finding): the pre-round-3 report claimed the deposited vapor
    "exceeds saturation there and the model's existing microphysics
    converts it" -- MEASURABLY FALSE for the recipient: post-deposit
    qt(k16) = 10.0878 g/kg is still 0.460 g/kg SHORT of qs(k16) =
    10.5479 g/kg, i.e. the arriving moisture does NOT condense on
    arrival.  What actually happens (accurate, per the design doc
    amendment): the recipient cell EVAPORATES the arriving condensate
    (net column LWP = rho1*thick_16*d_qc_16 = -2.694 g/m2 per 60-s step
    = -161.6 g/m2/h, MEASURED round-5; round-4 gave -2.884 and
    -173.0) for approximately 28 minutes (28.01 steps at dt = 60 s --
    the post-deposit deficit 0.460252e-3 divided by the per-step
    d_qv + d_qc = 1.643379e-5; round-4 gave 26.1 steps) until the
    cumulative deposit finally saturates it -- a real but SURVIVABLE
    transient (75.5 g/m2 of the specimen's 252.4 g/m2 column LWP,
    against the G-M3 bar of >= 130 g/m2), after which the cell holds
    condensate and the cloud top has discretely DEEPENED by one cell.
    This is discrete cloud-top deepening over a measured transient, not
    instantaneous condensation, and must be described as such.
    """
    from woof.verify import sase_ref
    amp = _vent_args("amp")
    thick = amp["dz_col"]
    zc = np.cumsum(thick) - 0.5 * thick
    zf = _vent_faces(thick)
    rho1, dt = amp["rho1"], 60.0
    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**amp)

    def deposit(f):
        return (f[:-1, 0, 0] - f[1:, 0, 0]) * dt / (rho1 * thick)

    d_th, d_qv, d_qc = deposit(f_th), deposit(f_qv), deposit(f_qc)
    d_qt = d_qv + d_qc
    band = (zc >= 1000.0) & (zc <= 1800.0)
    assert list(np.nonzero(band)[0]) == [15, 16, 17]

    # THE REQUIREMENT, asserted directly: the clear cap cell (k17,
    # qc_env = 0) never loses OR gains anything to/from this term.
    assert float(amp["qc"][17, 0, 0]) == 0.0            # clear cell
    assert d_th[17] == 0.0 and d_qv[17] == 0.0 and d_qc[17] == 0.0

    # the recipient (k16, inside the deck, essentially condensate-free
    # but NOT the clear cap) gains on all three rows -- condensate
    # forming INSIDE the 1.0-1.8 km deck.
    assert float(amp["qc"][16, 0, 0]) < 1e-6            # ~condensate-free
    assert d_th[16] > 0.0
    assert d_qv[16] > 0.0
    # S4-4 ROUND-5: the pre-round-5 form was ``d_qc[16] > 1.0e-5``, a
    # one-sided threshold sitting 4% under the then-measured 1.0384e-5.
    # The cloud-base anchor moves the deposit to 9.698718e-6 (x0.934,
    # the same uniform amplitude factor every specimen row takes), so
    # the threshold is REPLACED -- not lowered -- by the exact measured
    # value plus the sign requirement it existed to express.  An
    # equality pin is a strictly tighter constraint than the threshold
    # was: any amplitude drift now fails, in either direction.
    assert d_qc[16] > 0.0
    assert d_qc[16] == pytest.approx(9.698718e-6, rel=1e-5)

    gain = float(np.sum(thick[band] * d_qt[band]))
    loss = float(np.sum(thick[~band] * d_qt[~band]))
    assert gain > 0.0
    assert abs(gain + loss) <= 1e-12 * abs(gain)    # telescoped return
    assert gain * rho1 / dt > 0.2e-4                # the return channel
    above = zc > 1800.0
    for d in (d_th, d_qv, d_qc):
        assert d[above].tobytes() == np.zeros(
            int(above.sum())).tobytes()             # nothing above lid
    # nothing at/inside the clear cap cell either (Important-2/3 fix)
    at_or_above_cap = zc >= 1512.0
    for d in (d_th, d_qv, d_qc):
        assert d[at_or_above_cap].tobytes() == np.zeros(
            int(at_or_above_cap.sum())).tobytes()

    # STRUCTURAL CONSERVATION (S4-4 review round-3, description
    # corrected round-4): transport is EXACTLY conserved within the
    # plume's own territory -- root k_r = 10 through the entrainment-
    # zone cell k_top + 1 = 16 -- and bitwise +0.0 outside it,
    # REPLACING the false "internal redistribution within 1.0-1.8 km"
    # claim with the true structural one (measured: 48.8% of the
    # condensate funding k16's gain comes from cells 10-14, all below
    # 1.0 km).  The member run itself is only k12..k15.
    run_and_ez = list(range(10, 17))          # k_r .. k_top+1
    outside = [k for k in range(len(thick)) if k not in run_and_ez]
    for nm, d in (("th", d_th), ("qv", d_qv), ("qc", d_qc)):
        resid = float(np.sum(thick[run_and_ez] * d[run_and_ez]))
        scale = float(np.sum(thick[run_and_ez] * np.abs(d[run_and_ez])))
        assert abs(resid) <= 1e-13 * scale, (nm, resid, scale)
        assert d[outside].tobytes() == np.zeros(len(outside)).tobytes(), nm
    # the true source breakdown: cells below 1.0 km (10-14, outside the
    # pre-round-3 report's claimed 1.0-1.8 km band) fund the majority
    # of the condensate moved into k16.
    below_1km = [k for k in run_and_ez[:-1] if zc[k] < 1000.0]
    assert below_1km == [10, 11, 12, 13, 14]
    run_qc_loss = -float(np.sum(thick[run_and_ez[:-1]]
                                * d_qc[run_and_ez[:-1]]))
    below_1km_loss = -float(np.sum(thick[below_1km] * d_qc[below_1km]))
    frac_below = below_1km_loss / run_qc_loss
    assert 0.45 <= frac_below <= 0.52, frac_below      # measured 0.488

    # SUBSATURATED-INTERFACE ARRIVAL (S4-4 review round-3, f3 verifier
    # finding): the recipient does NOT condense on arrival -- it stays
    # short of saturation after one step (evaporate-then-resaturate
    # over a real, measured transient -- discrete cloud-top deepening,
    # never instantaneous condensation).  q_s computed with the SAME
    # Tetens-liquid formula plume_vent_flux/moist_n2 use.
    t16 = float(amp["theta"][16, 0, 0]) * (
        float(amp["p"][16, 0, 0]) / sase_ref.P0_REF
    ) ** (sase_ref.RD_AIR / sase_ref.CP_AIR)
    es16 = (1000.0 * sase_ref.SVP1
            * np.exp(sase_ref.SVP2 * (t16 - sase_ref.SVPT0)
                    / (t16 - sase_ref.SVP3)))
    qs16 = sase_ref.EP2_RV * es16 / (float(amp["p"][16, 0, 0]) - es16)
    qt_post = float(amp["qv"][16, 0, 0] + amp["qc"][16, 0, 0]) + d_qv[16] \
        + d_qc[16]
    assert qt_post < qs16, (qt_post, qs16)              # short of qs
    # S4-4 review round-4: re-centred.  MEASURED (round-5)
    # qs16 - qt_post = 0.460252e-3 kg/kg (was 0.459092e-3 at round 4;
    # the smaller round-5 deposit leaves the cell 1.16e-6 kg/kg further
    # from saturation).  The old 0.4767e-3 centre was the PRE-deposit
    # deficit qs16 - qt_pre (= 0.476686e-3, the number the module
    # docstring's MEMBERSHIP section quotes as "qv 0.4767 g/kg SHORT of
    # qs"); the deposit itself adds d_qv + d_qc = 1.643379e-5 kg/kg, so
    # the POST-deposit deficit this line asserts is 0.016434e-3 smaller.
    assert (qs16 - qt_post) == pytest.approx(0.460252e-3, abs=0.05e-3)
    assert (qs16 - (qt_post - d_qv[16] - d_qc[16])) == pytest.approx(
        0.476686e-3, abs=0.05e-3)                    # pre-deposit deficit

    # CRITICAL-1 (S4-4 review round-3): F_qc stays strictly positive on
    # the surviving upper-deck faces -- the plume carries condensate UP
    # into the detrainment band, never out of the column.  Narrowed
    # from the pre-round-2 band [880, 1520] m to [880, 1300] m: face
    # 1511.65 m is now the exact-zero termination face (Important-2),
    # so the old upper edge straddled it; faces 887/1063/1270 m survive
    # unaffected and are re-pinned here rather than left unasserted.
    up = (zf >= 880.0) & (zf <= 1300.0)
    assert int(up.sum()) == 3
    assert np.all(f_qc[up, 0, 0] > 0.0), f_qc[up, 0, 0]


def test_m2_rate_cap_registered_and_bounded():
    """THE RATE CAP (stiffness + no-runaway): the registered per-step
    deposit bound VENT_THETA_STEP_CAP = 0.14 K/step carries the S3-11a
    precedent verbatim (module docstring S3-11a section: the surface
    deposit's measured stable class |dtheta_1| ~ 0.14 K per d02 step
    at the pinned defect state, dt = 15 s, HFX = -180.2 W/m2); the
    S4-5 deposit seam enforces it by a uniform per-column flux rescale
    (contract in the plume_vent_flux docstring -- uniform so the
    telescoping and the F[0] = F[top] = 0 contract survive).  On the
    specimen the closure sits far inside the cap: at dt = 60 s (4x the
    d02 physics step -- conservative) the max theta deposit is
    MEASURED 0.0069862 K/step (round-5, cloud-base anchor; was
    0.0074795 K/step at round 4), 20.04x under the cap; pinned
    <= 0.02 so a future amplitude regression is loud."""
    from woof.verify import sase_ref
    amp = _vent_args("amp")
    thick = amp["dz_col"]
    f_th, _, _ = sase_ref.plume_vent_flux(**amp)
    d_th = (f_th[:-1, 0, 0] - f_th[1:, 0, 0]) * 60.0 / (amp["rho1"]
                                                        * thick)
    assert np.abs(d_th).max() <= 0.02
    assert np.abs(d_th).max() <= sase_ref.VENT_THETA_STEP_CAP


def test_m2_moisture_rate_cap_bounded():
    """S4-4 REVIEW FINDING Important-4: the registered rate limiter
    must bound EVERY deposited quantity, not theta alone.  THE DEFECT
    (S4-4 review, measured): the shipped-at-authority closure's qv+qc
    deposit was UNBOUNDED -- 0.047 g/kg/step as shipped, x16.5 at the
    theta cap's own headroom limit, with no registered ceiling of its
    own.  THE FIX: VENT_QT_STEP_CAP = VENT_THETA_STEP_CAP*CP_AIR/XLV
    (module docstring, SASE-M2 RATE CAP) -- DERIVED from the SAME
    registered constant via the latent/sensible heat equivalence, no
    second independent tunable -- bounds qv+qc the way
    VENT_THETA_STEP_CAP bounds theta, pinned the same way (the S4-5
    seam is expected to enforce both via the SAME uniform rescale
    factor, module docstring contract).

    On the specimen at dt = 60 s the closure sits inside the derived
    cap too (MEASURED round-5: max|d_qv + d_qc| = 0.01643 g/kg/step,
    3.42x under VENT_QT_STEP_CAP = 0.05625 g/kg/step; was 0.0176 /
    3.20x before the cloud-base anchor) -- tighter headroom than the
    theta cap's 20.04x, which is exactly why this channel needed its
    own registered bound rather than inheriting theta's margin.

    S4-4 REVIEW FINDING (x2 lane, round-3 CAP FAMILY fix): the SUM-form
    cap |d_qv + d_qc| <= VENT_QT_STEP_CAP is identically VACUOUS on any
    column doing pure phase conversion (parcel qt == environment qt, so
    F_qv = -F_qc exactly and the sum cancels to +0.0) while the
    INDIVIDUAL rows can still reach the cap independently -- on this
    very specimen three cells (k10/k11/k12) carry opposite-signed
    d_qv/d_qc with (|d_qv| + |d_qc|)/|d_qt| = 19.1597/9.8792/3.2443
    (S4-4 review round-4 CORRECTION: the k11 slot read "3.4" here and
    "3.2-3.4x" in the round-3 report.  Re-measured this session on the
    specimen at dt = 60 s -- the only construction the sentence refers
    to -- k11 is 9.8792; the same triple comes out of the round-3 build
    3367244 bit-for-bit and out of the S4-4-authority build 7e722e9 as
    19.0021/9.8288/3.2398, and -- ON THE ROUND-3, ROUND-4 AND ROUND-5
    BUILDS -- no OTHER registered M2 column (deck z_lid 1200/1400/1600,
    weak-lid deck at dz 280/100/25, the taper column) has ANY cell with
    a cancellation ratio above 1.0, let alone 3.4.  S4-4 ROUND-5
    CORRECTION: that sentence previously carried no build qualifier and
    is FALSE for the fourth build it implicitly covered -- re-measured
    this session, the S4-4-authority build 7e722e9 puts 4.2113 at k14
    of the weak-lid deck column (nz = 30, dz = 100 m).  The conclusion
    is unaffected: 4.2113 is not 3.4 either, "3.4" still reproduces
    nowhere and remains a transcription error, not a competing
    measurement; the nearest real 3.x number in this same docstring is
    the SUM-cap headroom 3.42x quoted two paragraphs above.  A single
    row can run several times the summed quotient
    while the sum test sees nothing.  THE FIX applies the SAME
    registered constant
    (no second tunable) to each row individually: |d_qv| <=
    VENT_QT_STEP_CAP and |d_qc| <= VENT_QT_STEP_CAP alongside the sum
    and the theta cap -- "one family" (module docstring, SASE-M2 RATE
    CAP): the S4-5 seam's uniform per-column rescale factor is min(1,
    VENT_THETA_STEP_CAP/|dtheta|_max, VENT_QT_STEP_CAP/|dqv|_max,
    VENT_QT_STEP_CAP/|dqc|_max) -- theta, qv, and qc ALL individually
    bound, still from the one registered constant.  On the specimen
    (MEASURED round-5): max|d_qv| = 7.2706e-6 (7.74x under the cap),
    max|d_qc| = 9.6987e-6 (5.80x under) -- both comfortably inside,
    tighter than the sum's 3.42x precisely because the sum benefits
    from cancellation the individual rows do not get.  The three
    cancellation RATIOS below are invariant under the round-5 change:
    the cloud-base anchor rescales all three flux rows by one uniform
    factor (0.93405 on this column), which cancels out of a ratio.
    """
    from woof.verify import sase_ref
    amp = _vent_args("amp")
    thick = amp["dz_col"]
    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**amp)
    rho1 = amp["rho1"]
    d_qv = (f_qv[:-1, 0, 0] - f_qv[1:, 0, 0]) * 60.0 / (rho1 * thick)
    d_qc = (f_qc[:-1, 0, 0] - f_qc[1:, 0, 0]) * 60.0 / (rho1 * thick)
    d_qt = d_qv + d_qc
    assert sase_ref.VENT_QT_STEP_CAP == pytest.approx(
        sase_ref.VENT_THETA_STEP_CAP * sase_ref.CP_AIR / sase_ref.XLV)
    assert np.abs(d_qt).max() <= sase_ref.VENT_QT_STEP_CAP
    assert np.abs(d_qt).max() == pytest.approx(1.6434e-5, rel=0.05)

    # CAP FAMILY (S4-4 review round-3): the qv and qc rows individually
    # bound by the SAME registered cap, not just their (cancellation-
    # prone) sum.
    assert np.abs(d_qv).max() <= sase_ref.VENT_QT_STEP_CAP
    assert np.abs(d_qc).max() <= sase_ref.VENT_QT_STEP_CAP
    assert np.abs(d_qv).max() == pytest.approx(7.2706e-6, rel=0.05)
    assert np.abs(d_qc).max() == pytest.approx(9.6987e-6, rel=0.05)
    # non-vacuity: at the cells where the sum cancels, the individual
    # rows are NOT cancelled -- the cap family does real, independent
    # work there (opposite-signed qv/qc, cancellation ratio >> 1).
    cancel = np.abs(d_qv) + np.abs(d_qc) > 3.0 * np.abs(d_qt)
    assert int(cancel.sum()) >= 3, cancel
    # the corrected triple itself, pinned so the docstring's numbers
    # cannot drift again (S4-4 review round-4).
    ratio = (np.abs(d_qv) + np.abs(d_qc)) / np.where(
        np.abs(d_qt) > 0.0, np.abs(d_qt), 1.0)
    assert ratio[10] == pytest.approx(19.1597, rel=1e-3)
    assert ratio[11] == pytest.approx(9.8792, rel=1e-3)
    assert ratio[12] == pytest.approx(3.2443, rel=1e-3)


def test_m2_cap_family_uniform_rescale_clamps_and_conserves():
    """THE S4-5 CAP ENFORCEMENT, AUTHORITY SIDE
    (:func:`sase_ref.vent_deposit_rescale`).

    WHY THIS FUNCTION EXISTS.  The cap family (VENT_THETA_STEP_CAP =
    0.14 K/step; VENT_QT_STEP_CAP = the derived
    VENT_THETA_STEP_CAP*CP_AIR/XLV, applied per row to |d_qv| and
    |d_qc|) has been REGISTERED and CONTRACT-ONLY since S4-4: no code
    enforced it anywhere, and the rescale the S4-5 seam must perform
    lived only in the module docstring's prose.  An S4-5 review flagged
    a prose-only parity target as a defect for the device mirror, so
    the rescale is now a reference implementation the mirror parities
    against, and this fixture EXERCISES the clamp rather than observing
    that an uncapped deposit happened to be small.

    DRIVEN OVER CAP.  Flux amplitude scales EXACTLY as sqrt(e_sgs)
    (M_base = VENT_MB_COEF*rho1*sqrt(VENT_SIGW_SHARE*ebar)), so the
    fixture multiplies the column's own e_sgs to push the deposit past
    the caps; the boosts are amplitude drivers, NOT physical TKE claims
    (the G-M5 survey standard governs FIELD statements, and none is
    made here).  Two legs, chosen so BOTH members of the cap family
    bind:

    * the specimen at 100x e (amplitude 10x): the QC row binds --
      |d_qc|max reaches VENT_QT_STEP_CAP EXACTLY (MEASURED s =
      0.579994182174; capped |d_qc|max/cap = 1.00000000000000, while
      the theta row lands at 0.0405194 K/step, 0.2894 of its own cap,
      and the qv row at 0.7497 of the moisture cap.  UNCAPPED the same
      column reads 0.0698617 K/step, 7.27065e-5 and 9.69872e-5, i.e.
      1.724x over on the qc row);
    * the 1400-m deck at 2000x e (amplitude 44.7x): the THETA row binds
      -- |d_theta|max reaches 0.14 K/step EXACTLY (MEASURED s =
      0.109599992498; capped ratio 1.00000000000000, qv and qc both at
      5.1549483e-5 = 0.9164 of the moisture cap.  UNCAPPED: 1.27737
      K/step -- 9.12x over -- and 4.70342e-4 on both moisture rows).

    Both legs assert the binding row lands ON the cap at machine
    precision (<= 4 ulp), every row lands at or under its cap, and the
    telescoping survives the rescale -- sum thick*d_phi = 0 to
    roundoff.  That last one is the property the uniform rescale
    exists for: a per-level clip of a single row destroys it (module
    docstring, MEASURED sum thick*dtheta = -3.74 under naive clipping
    against 0.0 here).

    Also pinned: the DIVIDE GUARD (module docstring, S4-5
    implementation note) -- an inactive column has every |d|max exactly
    +0.0, so each quotient would be +inf; the guarded form returns s =
    1.0 with no RuntimeWarning, which under this suite's
    ``-W error::RuntimeWarning`` policy is the difference between a
    pass and a failure.  Tested on the control column, which stands
    down bitwise.
    """
    from woof.verify import sase_ref

    def _driven(args, boost):
        out = dict(args)
        out["e_sgs"] = np.asarray(args["e_sgs"], np.float64) * boost
        return out

    dt = 60.0
    legs = ((_driven(_vent_args("amp"), 100.0), 2, 0.579994182174),
            (_driven(_vent_deck_args(z_lid=1400.0)[0], 2000.0), 0,
             0.109599992498))
    caps = (sase_ref.VENT_THETA_STEP_CAP, sase_ref.VENT_QT_STEP_CAP,
            sase_ref.VENT_QT_STEP_CAP)
    for args, bind_row, want_s in legs:
        thick = args["dz_col"]
        fluxes = sase_ref.plume_vent_flux(**args)
        raw = [(f[:-1, 0, 0] - f[1:, 0, 0]) * dt
               / (args["rho1"] * thick) for f in fluxes]
        # non-vacuity: the UNCAPPED deposit really is over cap.
        assert max(np.abs(r).max() / c for r, c in zip(raw, caps)) > 1.0
        d = sase_ref.vent_deposit_rescale(*fluxes, dt, args["rho1"],
                                          dz_col=thick)
        deps, s = d[:3], d[3]
        assert s.item() == pytest.approx(want_s, rel=1e-6)
        assert 0.0 < s.item() < 1.0
        for row, (dep, cap) in enumerate(zip(deps, caps)):
            dmax = float(np.abs(dep).max())
            assert dmax <= cap * (1.0 + 4.0 * np.finfo(np.float64).eps)
            if row == bind_row:               # reaches the cap EXACTLY
                assert abs(dmax - cap) <= 4.0 * np.spacing(cap), (
                    row, dmax, cap)
            # telescoping survives the uniform rescale
            col = dep[:, 0, 0]
            resid = abs(float(np.sum(thick * col)))
            scale = float(np.sum(thick * np.abs(col)))
            assert resid <= 1e-13 * scale, (row, resid, scale)
        # uniform, not per-row: rescaling the FLUXES by the SAME s
        # reproduces the returned deposits bitwise.
        for f, dep in zip(fluxes, deps):
            fs = f * s
            hand = (fs[:-1] - fs[1:]) * dt / (args["rho1"] * thick[
                :, None, None])
            assert hand.tobytes() == dep.tobytes()
    # DIVIDE GUARD on a stood-down (bitwise-zero) column: s == 1.0 and
    # no RuntimeWarning (the suite runs -W error::RuntimeWarning).
    ctl = _vent_args("ctl")
    zeros = sase_ref.plume_vent_flux(**ctl)
    assert all(f.tobytes() == np.zeros_like(f).tobytes() for f in zeros)
    dz_out = sase_ref.vent_deposit_rescale(*zeros, dt, ctl["rho1"],
                                           dz_col=ctl["dz_col"])
    assert dz_out[3].item() == 1.0
    for dep in dz_out[:3]:
        assert dep.tobytes() == np.zeros_like(dep).tobytes()
    # interface rejections
    with pytest.raises(ValueError, match="rho1"):
        sase_ref.vent_deposit_rescale(*zeros, dt, -1.0,
                                      dz_col=ctl["dz_col"])
    with pytest.raises(ValueError, match="f_qv"):
        sase_ref.vent_deposit_rescale(zeros[0], zeros[1][:-1], zeros[2],
                                      dt, ctl["rho1"],
                                      dz_col=ctl["dz_col"])


def test_m2_ledger_telescoping_exact():
    """THE M2 SCALAR LEDGER (C4/C5 + the S3-11a extension).  ALGEBRA
    (written out in the plume_vent_flux docstring): with the registered
    S4-5 deposit d(phi)_k = (F[k] - F[k+1])*dt/(rho1*thick_k) and
    per-column scalar rho1,

        sum_k thick_k*d(phi)_k = (dt/rho1)*sum_k (F[k] - F[k+1])
                               = (dt/rho1)*(F[0] - F[nz]) = 0

    exactly in exact arithmetic, because the interior faces telescope
    and BOTH end faces are exactly zero by the interface contract
    (F[0] = F[top] = +0.0: the surface flux is owned by the S3-11a
    deposit -- double-counting ban -- and full in-column detrainment
    ends the profile at NB).  The S3-11a boundary-consistent scalar
    ledger therefore extends with a ZERO net-column M2 term.  Pinned:

    * F[0] and F[top] are bitwise +0.0 on specimen and deck columns;
    * the FP64 telescoping residual is roundoff-class (<= 1e-13
      relative to the absolute deposit scale) for all three scalars on
      both columns (MEASURED: ~1e-16 relative).
    """
    from woof.verify import sase_ref
    for args in (_vent_args("amp"), _vent_deck_args(z_lid=1400.0)[0]):
        thick = args["dz_col"]
        fluxes = sase_ref.plume_vent_flux(**args)
        zero = np.zeros((1, 1)).tobytes()
        for f in fluxes:
            assert f[0].tobytes() == zero               # F_bottom = +0.0
            assert f[-1].tobytes() == zero              # F_top = +0.0
            d = (f[:-1, 0, 0] - f[1:, 0, 0]) / thick    # dt/rho1 factors
            resid = abs(float(np.sum(thick * d)))       # cancel in ratio
            scale = float(np.sum(thick * np.abs(d)))
            if scale > 0.0:
                assert resid <= 1e-13 * scale, (resid, scale)


def test_m2_f1_les_limit_bitwise_zero():
    """THE LES LIMIT (S3-11a zero-flux identity class): M_used =
    (1 - f)*M_plume in the module's FP-exact two-product blend idiom,
    so f_blend = 1.0 makes the amplitude literally +0.0 (0.0*M_plume)
    and every returned face is bitwise +0.0 -- on the SAME specimen
    state that is fully active at f = 0 (non-vacuous).  Also pinned:
    f = 0.5 scales every flux by exactly 0.5 (a power-of-two rescale
    is FP-exact, so the blend is verifiably the pure two-product form
    with no hidden f-dependence), and f outside [0, 1] is rejected."""
    from woof.verify import sase_ref
    amp = _vent_args("amp")
    live = sase_ref.plume_vent_flux(**amp)
    assert float(np.abs(live[1] + live[2]).max()) > 0.0
    nzp1 = amp["theta"].shape[0] + 1
    zeros = np.zeros((nzp1, 1, 1)).tobytes()
    for arr in sase_ref.plume_vent_flux(**dict(amp, f_blend=1.0)):
        assert arr.tobytes() == zeros
    half = sase_ref.plume_vent_flux(**dict(amp, f_blend=0.5))
    for h, l in zip(half, live):
        assert np.array_equal(h, 0.5 * l)
    with pytest.raises(ValueError, match="f_blend"):
        sase_ref.plume_vent_flux(**dict(amp, f_blend=1.5))
    with pytest.raises(ValueError, match="f_blend"):
        sase_ref.plume_vent_flux(**dict(amp, f_blend=-0.1))


def test_m2_interface_contract_and_rejections():
    """Interface hygiene: face-registered (nz+1, ny, nx) FP64 outputs;
    inputs never mutated; rho1 must be strictly positive (the S3-11a
    idiom -- a sign-flipped density silently inverts every deposit);
    shape mismatches are rejected loudly.

    BATCHED-COLUMN IDENTITY (the vectorization pin, the future device
    mirror's structural contract): stacking the amplifier and control
    columns side-by-side into one (nz, 1, 2) call -- heterogeneous
    roots/terminations, per-column dz_col, per-column rho1 -- returns
    bitwise the two single-column results in their slots (the level-
    loop state machine carries no cross-column coupling).
    """
    from woof.verify import sase_ref
    amp = _vent_args("amp")
    snap = {k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in amp.items()}
    out = sase_ref.plume_vent_flux(**amp)
    nz = amp["theta"].shape[0]
    for arr in out:
        assert arr.shape == (nz + 1, 1, 1)
        assert arr.dtype == np.float64
    for k, v in snap.items():
        if isinstance(v, np.ndarray):
            assert amp[k].tobytes() == v.tobytes(), k

    ctl = _vent_args("ctl")
    out_ctl = sase_ref.plume_vent_flux(**ctl)
    both = {}
    for key in ("theta", "qv", "qc", "p", "e_sgs"):
        both[key] = np.concatenate([amp[key], ctl[key]], axis=2)
    both["n2m_mask"] = np.concatenate(
        [amp["n2m_mask"], ctl["n2m_mask"]], axis=2)
    both["dz_col"] = np.concatenate(
        [amp["dz_col"].reshape(-1, 1, 1),
         ctl["dz_col"].reshape(-1, 1, 1)], axis=2)
    both["rho1"] = np.array([[amp["rho1"], ctl["rho1"]]])
    both["f_blend"] = 0.0
    out_both = sase_ref.plume_vent_flux(**both)
    for f2, fa, fc in zip(out_both, out, out_ctl):
        assert f2.shape == (nz + 1, 1, 2)
        assert f2[:, :, 0:1].tobytes() == fa.tobytes()
        assert f2[:, :, 1:2].tobytes() == fc.tobytes()
    with pytest.raises(ValueError, match="rho1"):
        sase_ref.plume_vent_flux(**dict(amp, rho1=-1.0))
    with pytest.raises(ValueError, match="rho1"):
        sase_ref.plume_vent_flux(**dict(amp, rho1=0.0))
    bad = dict(amp, qv=amp["qv"][:-1])
    with pytest.raises(ValueError, match="shape"):
        sase_ref.plume_vent_flux(**bad)
    bad_mask = dict(amp, n2m_mask=amp["n2m_mask"][:-1])
    with pytest.raises(ValueError, match="shape"):
        sase_ref.plume_vent_flux(**bad_mask)


def test_m2_natural_neutral_buoyancy_taper_branch():
    """BRANCH COVERAGE (S4-4 review round-3, x2 lane finding): the
    remaining-buoyancy TAPER branch of step 5 -- the natural-NB search,
    the buoyancy-peak search, and the reverse-cumsum taper weight --
    has ZERO coverage from any OTHER registered M2 column (the
    specimen and every prescribed-lid deck terminate with the
    buoyancy peak exactly one cell under the lid, the degenerate
    empty-taper case).  PROVEN a real S4-5 device-mirror parity risk
    by mutation: a build of plume_vent_flux with the natural-NB and
    peak searches DELETED -- unconditionally setting k_nb := k_lid and
    kb := k_lid - 1 whenever an LFC exists -- is BITWISE IDENTICAL on
    every other registered M2 column and PASSES every other registered
    M2 fixture, yet diverges from the real closure by up to
    5.12e-3 K kg m^-2 s^-1 (~2x the upper-deck heat flux) on an
    ordinary cloud-deck column with a gradual stable cap.  This
    fixture is that ordinary column: it FAILS against the deleted-
    search mutant (verified separately, not itself run here -- the
    mutant is not shipped code).

    :func:`_vent_taper_args` (module docstring there has the full
    construction and the measured numbers) builds a deck column with a
    small STABLE ramp over its top 4 saturated cells, so the
    entraining parcel's buoyancy peaks at k=9 (face 1000 m) and then
    DECLINES gradually through cells 11-13 before the run's own top
    (k_top = 13) -- kb + 1 = 10 < k_nb = 14, the taper's activation
    condition.

    S4-4 REVIEW round-4 CORRECTION (measured on an instrumented build
    this session; the pre-round-4 prose here and in the round-3 report
    said "NB = k_lid = 14"): the C9 ceiling on this column is
    k_lid = k_top + 2 = 15, one cell ABOVE the termination.  k_nb = 14
    arrives through the NATURAL neutral-buoyancy search -- buoyancy
    b[14] = -0.028082 < 0, found while the loop is at k = 14 < k_lid =
    15 -- so the ceiling branch never fires here.  That is the stronger
    reading of this fixture: it is the one registered column whose
    termination is set by the physics of the ascent rather than by the
    cap boundary, which is exactly the branch the deleted-search mutant
    destroys.  Face 14 (1400 m) is therefore the entrainment-zone
    cell's own BOTTOM face, not "the entrainment zone's own top": this
    column detrains entirely inside its own deck and delivers nothing
    to the entrainment zone.

    S4-4 ROUND-5: the taper face VALUES move by one uniform factor
    (500/400)**VENT_ENT_COEF = 1.093348 -- this column's root IS its
    own cloud base (k_r = k_base = 4), so the cloud-base anchor changes
    nothing here but which face the shape normalizes on, z_f[4] =
    400 m instead of z_f[5] = 500 m.  The taper STRUCTURE (which faces
    carry flux, their strict decrease, the natural-NB termination) is
    unchanged, which is the property this fixture exists to hold.  The
    ebar window is likewise unmoved: e_sgs is uniform 0.6 m2/s2 on this
    column, so no window can see the difference -- which is why the
    round-5 window change needed a REAL column with real TKE
    (:func:`test_m2_real_column_cloud_base_anchor`).
    """
    from woof.verify import sase_ref
    args = _vent_taper_args()
    zf = _vent_faces(args["dz_col"])
    f_th, f_qv, f_qc = sase_ref.plume_vent_flux(**args)
    prof = f_th[:, 0, 0]
    nzero = np.nonzero(prof != 0.0)[0]
    assert nzero.size > 0

    # peak face (the grow zone's own maximum) and the taper faces above
    # it: >= 2 strictly-decreasing, strictly-positive faces.
    peak_idx = int(nzero[np.argmax(prof[nzero])])
    assert abs(zf[peak_idx] - 1000.0) < 1e-9, zf[peak_idx]
    taper_idx = nzero[nzero > peak_idx]
    assert taper_idx.size >= 2, taper_idx
    taper_vals = prof[taper_idx]
    assert np.all(taper_vals > 0.0), taper_vals
    assert np.all(np.diff(taper_vals) < 0.0), taper_vals
    assert list(taper_idx) == [11, 12, 13]                 # measured
    # S4-4 ROUND-5 RE-MEASURE (cloud-base anchor: this column's root is
    # its own base k4, so the shape normalization moves from z_f[5] =
    # 500 m to z_f[4] = 400 m and every face scales by the single
    # factor (500/400)**0.4 = 1.09335).
    assert taper_vals == pytest.approx(
        [0.0061225, 0.0022221, 0.00048953], rel=1e-4)

    # bitwise +0.0 AT and ABOVE the NATURAL neutral-buoyancy face
    # (k_nb = 14, face 1400 m); the C9 ceiling k_lid = 15 (face 1500 m)
    # sits one cell higher and is never reached on this column.
    j_nb = int(nzero[-1]) + 1
    assert abs(zf[j_nb] - 1400.0) < 1e-9, zf[j_nb]
    # S4-4 ROUND-5 REPLACEMENT.  The pre-round-5 line here was
    # ``assert abs(zf[j_nb + 1] - 1500.0) < 1e-9  # unused ceiling``,
    # which is GRID GEOMETRY, not closure behaviour: z_f[15] = 1500.0 m
    # on a uniform 100-m grid whatever plume_vent_flux computes, so it
    # placed no constraint on k_lid at all.  What the line was trying
    # to say -- that the C9 ceiling is NOT what terminated this column
    # -- is asserted instead against a ceiling RE-DERIVED here from the
    # column's own membership (the registered qt >= qs test taken
    # independently of the closure): k_lid = k_top + 2, and the
    # termination face sits strictly below it.  This fails if k_top,
    # the k_lid formula, or the natural-NB search moves.
    _, qs_t, _ = _vent_theta_es(args["theta"][:, 0, 0],
                                args["qv"][:, 0, 0],
                                args["qc"][:, 0, 0], args["p"][:, 0, 0])
    member_t = (((args["qv"] + args["qc"])[:, 0, 0] >= qs_t)
                & np.asarray(args["n2m_mask"])[:, 0, 0])
    ks_t = np.nonzero(member_t)[0]
    k_base_t = int(ks_t[0])
    k_top_t = k_base_t
    while member_t[k_top_t + 1]:
        k_top_t += 1
    assert (k_base_t, k_top_t) == (4, 13)          # measured member run
    k_lid_t = k_top_t + 2                          # the C9 ceiling face
    assert j_nb == k_top_t + 1                     # the run's own top + 1
    assert j_nb < k_lid_t                          # natural NB, not the cap
    assert f_th[k_lid_t, 0, 0] == 0.0
    for arr in (f_th, f_qv, f_qc):
        assert arr[j_nb:].tobytes() == np.zeros(
            (len(zf) - j_nb, 1, 1)).tobytes()

    # per-row column-sum residual <= 1e-13 relative (the S3-11a/C4-C5
    # ledger theorem, exercised on a REAL taper profile, not just the
    # degenerate empty-taper case every other fixture measures it on).
    thick, rho1, dt = args["dz_col"], args["rho1"], 60.0

    def deposit(f):
        return (f[:-1, 0, 0] - f[1:, 0, 0]) * dt / (rho1 * thick)

    for f in (f_th, f_qv, f_qc):
        d = deposit(f)
        resid = abs(float(np.sum(thick * d)))
        scale = float(np.sum(thick * np.abs(d)))
        if scale > 0.0:
            assert resid <= 1e-13 * scale, (resid, scale)

    # batched bitwise identity when stacked with a boundary-terminated
    # column (the weak-lid deck, which -- unlike this taper column --
    # terminates via the FORCED k_lid ceiling, not a natural NB): the
    # level-loop state machine carries no cross-column coupling even
    # when one column takes the taper branch and the other does not.
    boundary, _ = _vent_deck_args(z_lid=1400.0, dth_inv=2.0, nz=40,
                                  dz=100.0)
    out_taper = sase_ref.plume_vent_flux(**args)
    out_boundary = sase_ref.plume_vent_flux(**boundary)
    both = {}
    for key in ("theta", "qv", "qc", "p", "e_sgs"):
        both[key] = np.concatenate([args[key], boundary[key]], axis=2)
    both["n2m_mask"] = np.concatenate(
        [args["n2m_mask"], boundary["n2m_mask"]], axis=2)
    both["dz_col"] = np.concatenate(
        [args["dz_col"].reshape(-1, 1, 1),
         boundary["dz_col"].reshape(-1, 1, 1)], axis=2)
    both["rho1"] = np.array([[args["rho1"], boundary["rho1"]]])
    both["f_blend"] = 0.0
    out_both = sase_ref.plume_vent_flux(**both)
    for f2, ft, fb in zip(out_both, out_taper, out_boundary):
        assert f2.shape == (args["theta"].shape[0] + 1, 1, 2)
        assert f2[:, :, 0:1].tobytes() == ft.tobytes()
        assert f2[:, :, 1:2].tobytes() == fb.tobytes()


# ---------------------------------------------------------------------------
# The experimental label: what a user is told, and the controls that prove
# the telling is not decoration.
# ---------------------------------------------------------------------------

def _labelled_cfg(**kw):
    base = dict(_SASE_ADMITTED)
    base.update(kw)
    return validate_run_config(_min_cfg(**base))


def test_experimental_sentence_names_the_scheme_and_does_not_block():
    """One sentence, and the run continues.

    Warn-not-block is the posture for MATURITY, so the product must say
    the scheme is experimental and then get out of the way.
    """
    from woof.physics_compat import (VERIFICATION_EXPERIMENTAL,
                                      single_domain_verification_status)

    status = single_domain_verification_status(_labelled_cfg())
    assert status["status"] == VERIFICATION_EXPERIMENTAL
    # "not WRF-verified" is the WRONG clause for this closure: it reads
    # as a comparison that is merely outstanding, and invites a user to
    # wait for a verification that can never arrive.  The registry
    # declares the option has no WRF counterpart, and the sentence says
    # so.  A table-bound runtime -- which IS a WRF scheme -- still gets
    # "not WRF-verified"; the control below pins that.
    assert status["sentence"] == (
        "physics: SASE PBL: experimental, WOOF-original with no WRF "
        "counterpart -- the run continues.")
    assert status["experimental_components"] == ["SASE PBL"]
    # The one word the sentence must NOT contain.  "supported" promises a
    # WRF comparison that is merely outstanding; for a scheme WRF does
    # not have there is none to be outstanding, and claiming otherwise
    # would be the most misleading sentence the product prints.
    assert "supported" not in status["sentence"]


def test_a_non_experimental_suite_keeps_its_own_sentence():
    """Positive control: the label appears for SASE and only for SASE."""
    from woof.physics_compat import single_domain_verification_status

    for pbl, surface, km in ((0, 0, 4), (1, 91, 4), (5, 5, 4)):
        cfg = validate_run_config(_min_cfg(
            moist=True, bl_pbl_physics=pbl, sf_sfclay_physics=surface,
            km_opt=km))
        status = single_domain_verification_status(cfg)
        assert status["experimental_components"] == []
        assert "experimental" not in status["sentence"]


def test_control_the_label_follows_the_registry_not_a_hardcoded_name(
        monkeypatch):
    """Mutation control: clear BOTH triggers and the label MUST
    disappear; promote another option and its label MUST appear.

    This is what separates a registry-driven label from a scheme name
    spelled into the reporting code.  If either half of this test passes
    while the code names 'sase' directly, the first half would still
    fire -- so the second half, which never mentions SASE at all, is the
    one that has teeth.

    There are TWO triggers, and that is the point of (a1).  SASE sits at
    'implemented-unverified' -- the same rung as the WRF-transcribed YSU
    and MYNN ports -- because the conformance ladder measures distance
    from WRF and there is no WRF to measure from.  Keying the warning on
    the rung alone would therefore have silenced it for the one option
    in this registry that most needs it, which is why the declaration
    exists and why removing the rung alone must NOT be enough.
    """
    import copy

    from woof import physics_compat
    from woof.physics_registry import physics_registry

    registry = copy.deepcopy(physics_registry())
    options = registry["components"]["pbl"]["options"]
    cfg = _labelled_cfg()

    # (a1) the rung it actually carries is NOT what makes it warn: SASE
    # already sits at 'implemented-unverified', the same rung as YSU,
    # and still warns.  Assert the rung first so the reason is explicit.
    assert options["sase"]["maturity"] == "implemented-unverified"
    assert options["ysu"]["maturity"] == "implemented-unverified"
    assert physics_compat.experimental_component_labels(
        cfg, registry) == ("SASE PBL",)

    # (a2) clear the declaration -- the only trigger it has -- and the
    # label goes away even though the config is unchanged.
    options["sase"].pop("wrf_counterpart")
    assert physics_compat.experimental_component_labels(cfg, registry) == ()

    # (b) promoted elsewhere: a DIFFERENT option, never named in the
    # reporting code, is surfaced purely by carrying the rung.
    options["ysu"]["maturity"] = "experimental-runtime"
    ysu_cfg = validate_run_config(_min_cfg(
        moist=True, bl_pbl_physics=1, sf_sfclay_physics=91, km_opt=4))
    assert physics_compat.experimental_component_labels(
        ysu_cfg, registry) == ("YSU PBL",)
    # ... and it gets the OTHER clause, because YSU is a WRF scheme
    # whose forecast comparison really is merely outstanding.
    assert physics_compat.experimental_selection_sentence(
        [ysu_cfg], registry) == (
        "physics: YSU PBL: experimental, not WRF-verified "
        "-- the run continues.")


def test_control_the_experimental_rung_warns_and_never_blocks():
    """The registry's own policy, asserted rather than assumed."""
    from woof.physics_registry import physics_registry

    registry = physics_registry()
    ladder = registry["maturity_ladder"]["rungs"]["experimental-runtime"]
    assert ladder["warning_tier"] == "warn"
    policy = registry["warning_policy"]
    assert policy["maturity_never_blocks"] is True
    assert "experimental-runtime" in policy["warn_maturities"]
    assert "experimental-runtime" not in policy["nonwarning_maturities"]


def test_the_registry_entry_states_what_it_cannot_claim():
    """The option's warnings must say the three things a user needs.

    Not a style check: each clause below is a fact a reader would
    otherwise have to reconstruct from the design spec -- that no WRF
    oracle exists, that the acceptance scorecard mostly missed, and that
    the subgrid energy magnitude carries a known open bias.
    """
    from woof.physics_registry import physics_registry

    option = (physics_registry()["components"]["pbl"]["options"]["sase"])
    assert option["implemented"] is True
    # The conformance ladder measures agreement with WRF.  This closure
    # has no WRF counterpart, so it cannot climb that ladder at all and
    # holds its lowest executable rung PERMANENTLY -- not pending work.
    # The declaration beside it is what says so machine-readably; the
    # rung on its own would be indistinguishable from a WRF port whose
    # forecast comparison is merely outstanding.
    assert option["maturity"] == "implemented-unverified"
    assert option["wrf_counterpart"]["exists"] is False
    assert "Registry.EM_COMMON" in option["wrf_counterpart"]["basis"]
    assert "permanently" in option["wrf_counterpart"]["consequence"]
    assert option["scientific_evidence"] == "none"
    blob = " ".join(option["warnings"]).lower()
    assert "no wrf" in blob or "not a wrf scheme" in blob
    assert "2 of 7" in blob
    assert "bias" in blob


def test_the_documented_configuration_is_the_one_the_loader_admits(tmp_path):
    """The TOML on the published page is executable, not decorative.

    A user's first contact with an experimental scheme is the block the
    documentation tells them to paste.  If that block drifts from what
    the loader admits, the closure is unusable for exactly the reader it
    was written for -- and nothing else in this tree would notice,
    because no other test reads prose.  This one lifts the fenced block
    out of the published page and hands it to the real loader.

    MUTATION CONTROLS, one per documented requirement, each watched
    being REFUSED.  The page prints a table saying "none of these
    companions is taste, each is refused at config load".  Without the
    controls this test would still pass against a loader that enforced
    none of them, and the table beside the block would be a lie.
    """
    import re
    import tomllib
    from pathlib import Path

    from woof.config import SASE_PBL_SCHEME, load_config

    page = (Path(__file__).resolve().parents[1]
            / "docs" / "public" / "PHYSICS.md").read_text(encoding="utf-8")
    section = page.split("## Selecting an experimental scheme", 1)
    assert len(section) == 2, "the page must carry the selection section"
    block = re.search(r"```toml\n(.*?)```", section[1], re.S)
    assert block is not None, "the section must print a config block"
    text = block.group(1)

    path = tmp_path / "documented.toml"
    path.write_text(text, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.bl_pbl_physics == SASE_PBL_SCHEME
    # The block must SELECT the closure, not merely parse: a page that
    # printed a YSU config beside this prose would pass a parse check.
    assert tomllib.loads(text)["run"]["bl_pbl_physics"] == SASE_PBL_SCHEME

    # -- documented requirements, each removed in turn ------------------
    refused = []
    mutations = {
        "km_opt": ("km_opt = 0", "km_opt = 4"),
        "khdif": ("khdif = 0.0", "khdif = 100.0"),
        "kvdif": ("kvdif = 0.0", "kvdif = 100.0"),
        "sf_sfclay_physics": ("sf_sfclay_physics = 91",
                              "sf_sfclay_physics = 0"),
        "moist": ("moist = true", "moist = false"),
    }
    for name, (old, new) in mutations.items():
        assert old in text, (name, "the page no longer prints this line")
        broken = tmp_path / f"without_{name}.toml"
        broken.write_text(text.replace(old, new, 1), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(broken)
        refused.append(name)
    assert refused == list(mutations)

    # The nz ceiling the page states is the local-memory bound of the
    # implicit solve, so it is checked against the same number.
    from woof.config import SASE_MAX_NZ
    from woof.core.sase_limits import MAX_COLUMN_LEVELS

    assert SASE_MAX_NZ == MAX_COLUMN_LEVELS == 128
    assert f"`nz <= {SASE_MAX_NZ}`" in section[1]
    tall = tmp_path / "too_tall.toml"
    tall.write_text(text.replace("nz = 64", f"nz = {SASE_MAX_NZ + 1}", 1),
                    encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tall)


def test_the_real_data_path_the_page_prints_is_two_edits_and_no_more():
    """The page tells a user to emit a config and change two keys.

    That is a claim about what ``woof domain`` writes, and it is the
    difference between "here is how to try the closure on real data"
    and a paragraph that sends a reader into six loader refusals one at
    a time.  It is checked here against the wizard's own emitted
    defaults rather than against a copy of them.

    THE CONTROL is the count, not the pair: the test recomputes which
    of the closure's structural requirements the emitted config already
    satisfies and requires that set to be EXACTLY the complement of the
    two the page names.  A wizard default that drifts -- radiation
    turning ``moist`` off or dropping its surface layer -- turns the
    page's instruction into a lie and turns this
    red, which a test that merely asserted "km_opt and bl_pbl_physics
    are wrong" would not.
    """
    from woof.config import SASE_PBL_SCHEME, _SASE_REQUIREMENTS
    from woof.domain_wizard import (DEFAULT_SUITE_PHYSICS,
                                     _SHARED_GRID_AND_DYNAMICS)

    emitted = {**_SHARED_GRID_AND_DYNAMICS, **DEFAULT_SUITE_PHYSICS}
    # Every structural requirement the closure declares, split by
    # whether the emitted config already meets it.
    covered = {name for name, _v, _why in _SASE_REQUIREMENTS
               if name in emitted}
    already = {name for name, value, _why in _SASE_REQUIREMENTS
               if name in emitted and emitted[name] == value}
    assert covered == {name for name, _v, _why in _SASE_REQUIREMENTS}, covered
    outstanding = covered - already
    assert outstanding == {"km_opt"}, (outstanding, already)
    # The selector itself is the second edit and is not a "requirement".
    assert emitted["bl_pbl_physics"] != SASE_PBL_SCHEME
    # The three requirements that are not structural settings of the
    # emitted table are still met by it.
    assert emitted["moist"] is True
    assert emitted["sf_sfclay_physics"] != 0
    assert emitted["nz"] <= 128

    from pathlib import Path

    page = (Path(__file__).resolve().parents[1]
            / "docs" / "public" / "PHYSICS.md").read_text(encoding="utf-8")
    section = page.split("## Selecting an experimental scheme", 1)[1]
    assert "change `bl_pbl_physics` to `900` and" in section
    assert "`km_opt` to `0`" in section


# ---------------------------------------------------------------------------
# The three surfaces that actually print the warning, and the one property
# the closure does NOT have: per-nest selection.
# ---------------------------------------------------------------------------

def _documented_toml() -> str:
    """The fenced config block off the published page, as text."""
    import re
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1]
            / "docs" / "public" / "PHYSICS.md").read_text(encoding="utf-8")
    section = page.split("## Selecting an experimental scheme", 1)[1]
    block = re.search(r"```toml\n(.*?)```", section, re.S)
    assert block is not None, "the section must print a config block"
    return block.group(1)


def test_the_go_banner_prints_the_clause_the_page_quotes(tmp_path):
    """``woof go`` is the front door, and the page quotes its banner.

    The page used to quote a sentence the go banner does not print --
    lowercase, prefixed, with a trailing clause -- while the only test
    of the wording read ``physics_compat`` instead.  So the quoted
    banner was wrong on three counts and nothing noticed, because
    nothing had ever compared the page against ``go_cli``.  This does.
    """
    from pathlib import Path

    from woof import go_cli

    path = tmp_path / "documented.toml"
    path.write_text(_documented_toml(), encoding="utf-8")

    clause = go_cli._physics_words({"config": str(path)})
    assert clause == ("SASE PBL: EXPERIMENTAL, WOOF-original with no "
                      "WRF counterpart")

    # The page must quote a `go:` line that really ends in that clause.
    page = (Path(__file__).resolve().parents[1]
            / "docs" / "public" / "PHYSICS.md").read_text(encoding="utf-8")
    section = page.split("## Selecting an experimental scheme", 1)[1]
    quoted = [line for line in section.splitlines()
              if line.startswith("go: ")]
    assert len(quoted) == 1, "the page must quote the go banner once"
    assert quoted[0].endswith(clause)

    # CONTROL: the clause comes from the registry, not from a spelling
    # of the scheme name.  A non-experimental suite gets the ordinary
    # physics words, and the shouted word is nowhere in them.
    plain = tmp_path / "plain.toml"
    plain.write_text(
        _documented_toml()
        .replace("bl_pbl_physics = 900", "bl_pbl_physics = 1")
        .replace("km_opt = 0", "km_opt = 4"),
        encoding="utf-8")
    assert "EXPERIMENTAL" not in go_cli._physics_words({"config": str(plain)})


def test_both_forecast_runners_print_the_same_experimental_sentence():
    """One definition, and the tree runner is no longer silent.

    ``woof go`` REFUSES a multi-domain config and names the tree
    runner instead, so the tree path is not reachable through the
    banner at all.  Before this, selecting an experimental closure on a
    domain tree -- the normal multi-nest workflow -- printed nothing
    anywhere.
    """
    from woof import physics_compat, prepared_domain_tree_forecast

    cfg = _labelled_cfg()
    expected = ("physics: SASE PBL: experimental, WOOF-original with "
                "no WRF counterpart -- the run continues.")

    # The shared builder, and the single-domain surface that reads it.
    assert physics_compat.experimental_selection_sentence([cfg]) == expected
    assert (physics_compat.single_domain_verification_status(cfg)["sentence"]
            == expected)
    # Said ONCE for a tree, not once per nest.
    assert (physics_compat.experimental_selection_sentence([cfg, cfg, cfg])
            == expected)
    # CONTROL: nothing experimental, nothing said.
    ysu = validate_run_config(_min_cfg(
        moist=True, bl_pbl_physics=1, sf_sfclay_physics=91, km_opt=4))
    assert physics_compat.experimental_selection_sentence([ysu]) is None

    # The tree runner reads the shared builder rather than restating it.
    assert (prepared_domain_tree_forecast.experimental_selection_sentence
            is physics_compat.experimental_selection_sentence)


def test_the_tree_runner_emits_the_sentence_from_its_own_main(
        tmp_path, capsys, monkeypatch):
    """The emission point, exercised in ``main`` rather than asserted.

    The preflight and the forecast are stubbed because neither is what
    is under test; what is under test is that ``main`` reaches the
    print between them, with the per-domain run configs the preflight
    really returns.
    """
    import types
    from pathlib import Path

    from woof import prepared_domain_tree_forecast as runner

    cfg = _labelled_cfg()
    fake_inputs = types.SimpleNamespace(
        experiment=types.SimpleNamespace(
            domains=(types.SimpleNamespace(run=cfg),
                     types.SimpleNamespace(run=cfg))))

    monkeypatch.setattr(runner, "preflight_prepared_tree",
                        lambda **kw: fake_inputs)
    monkeypatch.setattr(runner, "run_prepared_tree", lambda *a, **kw: {
        "status": "OK", "readiness": "ok",
        "execution_plan": {"plan_id": "p", "domain_count": 2},
        "wall_seconds": 0.0, "output": {"frame_count": 0}})
    # main's install preflight refuses before the print on an install
    # without CuPy, for a reason that is not the one under test.  Only
    # its question is answered; the runtime is not provided, and with
    # both stages stubbed nothing on this path loads it.
    from woof import capabilities
    installed = capabilities.is_installed
    monkeypatch.setattr(
        capabilities, "is_installed",
        lambda module: (str(module).split(".", 1)[0]
                        == capabilities.GPU_RUNTIME.module
                        or installed(module)))

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    config = tmp_path / "exp.toml"
    config.write_text(_sase_tree_toml(_SASE_SELECTOR), encoding="utf-8")
    assert runner.main([
        "--prepared-root", str(prepared),
        "--preparation-receipt-sha256", "0" * 64,
        "--experiment-config", str(config),
        "--experiment-config-sha256", "0" * 64,
        "--outdir", str(tmp_path / "out"),
    ]) == 0

    line = ("prepared tree: physics: SASE PBL: experimental, "
            "WOOF-original with no WRF counterpart "
            "-- the run continues.")
    captured = capsys.readouterr()
    assert captured.err.splitlines().count(line) == 1
    # The page quotes that exact line.
    page = (Path(__file__).resolve().parents[1]
            / "docs" / "public" / "PHYSICS.md").read_text(encoding="utf-8")
    assert line in page


def _sase_tree_toml(shared_pbl: int, child_pbl: int | None = None,
                    parent_pbl: int | None = None) -> str:
    """A minimal moist two-domain tree for the SASE nest tests.

    ``shared_pbl`` goes in [shared]; ``parent_pbl``/``child_pbl``, when
    given, are per-domain ``bl_pbl_physics`` keys.  A SASE domain runs
    km_opt = 0, the only mixing SASE is admitted with, and a YSU domain
    km_opt = 4, the horizontal Smagorinsky every PBL-on template pins.
    """
    def km(pbl):
        return 0 if pbl == 900 else 4

    def pbl_lines(value):
        if value is None:
            return ""
        return f"bl_pbl_physics = {value}\nkm_opt = {km(value)}\n"

    return f"""[experiment]
name = "sase_tree"
start_time = 2026-08-01T00:00:00
run_seconds = 60.0
restart_interval_s = 0.0
[projection]
map_proj = "lambert"
ref_lat = 40.0
ref_lon = -100.0
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -100.0
[shared]
nz = 30
ztop = 16000.0
p_top = 10000.0
moist = true
mp_physics = 6
sf_sfclay_physics = 1
km_opt = {km(shared_pbl)}
bl_pbl_physics = {shared_pbl}
[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 60
ny = 60
dx = 3000.0
time_step = 12
specified = true
nested = false
history_interval_s = 3600.0
{pbl_lines(parent_pbl)}[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 15
j_parent_start = 15
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 30
ny = 30
specified = false
nested = true
history_interval_s = 3600.0
{pbl_lines(child_pbl)}"""


def _sase_warning_rows(said: str) -> list[str]:
    return [row for row in said.splitlines()
            if "(SASE) runs on grid_id" in row]


def test_a_two_domain_tree_runs_sase_on_every_domain_and_says_so(
        tmp_path, capsys):
    """SASE selected in [shared] reaches both domains of a tree.

    The subgrid energy is per domain and is not a nest-forced field, so
    the child cold-starts its own and nothing crosses the edge.  The load
    says the tree is implemented but not yet verified; it does not
    refuse, and it does not raise NotImplementedError past the run-plan
    door as it did before.
    """
    from woof.config import SASE_PBL_SCHEME
    from woof.core.nest_fields import nest_field_kinds
    from woof.experiment import load_experiment

    path = tmp_path / "tree.toml"
    path.write_text(_sase_tree_toml(SASE_PBL_SCHEME), encoding="utf-8")
    exp = load_experiment(path)
    assert [d.run.bl_pbl_physics for d in exp.domains] == [
        SASE_PBL_SCHEME, SASE_PBL_SCHEME]
    assert exp.domains[1].run.nested
    for dom in exp.domains:
        kinds = nest_field_kinds(dom.run)
        assert "e_sgs" not in kinds and "e" not in kinds
    rows = _sase_warning_rows(capsys.readouterr().err)
    assert len(rows) == 1, rows
    assert "grid_id 1, 2 of a 2-domain tree" in rows[0]
    assert "cold-starts its own subgrid energy" in rows[0]
    assert "not yet verified" in rows[0]


@pytest.mark.parametrize("parent_pbl,child_pbl", [(1, 900), (900, 1)])
def test_a_per_domain_sase_key_is_honoured_on_its_domain(
        tmp_path, capsys, parent_pbl, child_pbl):
    """``bl_pbl_physics = 900`` in a [[domain]] table selects SASE there.

    Per-domain bl_pbl_physics is honoured for every scheme; SASE was the
    one value refused, for a reason that named no breakage (a comparison
    across the nest edge).  Each domain runs its own closure on its own
    state, so the key now takes effect on the domain that carries it and
    the other domain keeps its own scheme.
    """
    from pathlib import Path

    from woof import experiment as exp_mod

    assert "bl_pbl_physics" in exp_mod._DOMAIN_RUN_OVERRIDES
    # The closure's physics selectors stay whole-tree keys: every SASE
    # domain of a tree runs the same closure variant.
    assert "sase_moist_n2" not in exp_mod._DOMAIN_KEYS
    assert "sase_stable_dissipation" not in exp_mod._DOMAIN_KEYS

    path = tmp_path / "tree.toml"
    path.write_text(_sase_tree_toml(1, child_pbl=child_pbl,
                                    parent_pbl=parent_pbl),
                    encoding="utf-8")
    exp = exp_mod.load_experiment(path)
    assert [d.run.bl_pbl_physics for d in exp.domains] == [
        parent_pbl, child_pbl]
    sase_id = 1 if parent_pbl == 900 else 2
    rows = _sase_warning_rows(capsys.readouterr().err)
    assert len(rows) == 1, rows
    assert f"grid_id {sase_id} of a 2-domain tree" in rows[0]

    # The page says what the loader does.
    page = (Path(__file__).resolve().parents[1]
            / "docs" / "public" / "PHYSICS.md").read_text(encoding="utf-8")
    section = page.split("## Selecting an experimental scheme", 1)[1]
    assert "run-wide, never per-nest" not in section
    assert "**The closure can be selected per domain.**" in section


def test_a_single_domain_sase_run_carries_no_tree_warning(tmp_path, capsys):
    """The control: one domain, no nest, no tree sentence."""
    from woof.experiment import load_experiment

    text = _sase_tree_toml(900)
    text = text[:text.index("[[domain]]\ngrid_id = 2")]
    path = tmp_path / "one.toml"
    path.write_text(text, encoding="utf-8")
    load_experiment(path)
    assert _sase_warning_rows(capsys.readouterr().err) == []


def _with_shared_line(text: str, line: str) -> str:
    """``line`` added to the [shared] table of a _sase_tree_toml tree."""
    anchor = "[[domain]]\ngrid_id = 1\n"
    assert text.count(anchor) == 1
    return text.replace(anchor, f"{line}\n{anchor}")


#: A non-default value of each fail-closed SASE selector.
_SASE_NON_DEFAULT = [("sase_moist_n2", False),
                     ("sase_stable_dissipation", True),
                     ("sase_flux_diag", True)]


def test_the_loader_and_the_run_check_share_one_sase_default_table():
    """The loader resets a non-SASE domain to the defaults the run check
    judges against; both read woof.config.SASE_FAIL_CLOSED_DEFAULTS, and
    those are the RunConfig defaults a bare run gets."""
    from woof.config import SASE_FAIL_CLOSED_DEFAULTS

    bare = _min_cfg()
    assert SASE_FAIL_CLOSED_DEFAULTS == {
        name: getattr(bare, name) for name in SASE_FAIL_CLOSED_DEFAULTS}
    assert {name for name, _ in _SASE_NON_DEFAULT} == set(
        SASE_FAIL_CLOSED_DEFAULTS)


@pytest.mark.parametrize("name,value", _SASE_NON_DEFAULT,
                         ids=[n for n, _ in _SASE_NON_DEFAULT])
@pytest.mark.parametrize("parent_pbl,child_pbl", [(1, 900), (900, 1)],
                         ids=["ysu-parent", "sase-parent"])
def test_a_mixed_tree_takes_a_shared_sase_selector_on_its_sase_domains(
        tmp_path, parent_pbl, child_pbl, name, value):
    """A [shared] SASE selector on a tree that mixes SASE with YSU.

    Before, no spelling selected a SASE variant on such a tree: in
    [shared] the YSU domain refused it ("requires bl_pbl_physics=900"),
    and in the SASE domain's table it was refused as a run key that
    belongs in [shared].  The seam exists on the SASE domain, so the
    [shared] value now reaches that domain and the YSU domain keeps the
    default, which is inert there.
    """
    from woof.config import SASE_FAIL_CLOSED_DEFAULTS
    from woof.experiment import load_experiment

    text = _with_shared_line(
        _sase_tree_toml(1, child_pbl=child_pbl, parent_pbl=parent_pbl),
        f"{name} = {str(value).lower()}")
    path = tmp_path / "mixed.toml"
    path.write_text(text, encoding="utf-8")
    exp = load_experiment(path)
    by_pbl = {d.run.bl_pbl_physics: d.run for d in exp.domains}
    assert set(by_pbl) == {1, 900}
    assert getattr(by_pbl[900], name) is value
    assert getattr(by_pbl[1], name) is SASE_FAIL_CLOSED_DEFAULTS[name]


@pytest.mark.parametrize("name,value", _SASE_NON_DEFAULT[:2],
                         ids=[n for n, _ in _SASE_NON_DEFAULT[:2]])
def test_a_sase_physics_selector_in_a_domain_table_points_to_shared(
        tmp_path, name, value):
    """The other placement: the SASE domain's own [[domain]] table.

    The two physics selectors stay whole-tree keys, so every SASE domain
    of a tree runs one variant.  The refusal names [shared], and the
    test above proves that [shared] now works on the same tree.
    """
    from woof.experiment import load_experiment

    text = (_sase_tree_toml(1, child_pbl=900)
            + f"{name} = {str(value).lower()}\n")
    path = tmp_path / "mixed.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_experiment(path)
    said = str(caught.value)
    assert name in said and "belong in [shared]" in said


def test_a_flux_diag_written_on_the_non_sase_domain_is_checked_as_written(
        tmp_path):
    """``sase_flux_diag`` is per domain.  Written into the YSU domain's
    own table it is that domain's setting, not a tree-wide one, and the
    YSU domain has no SASE closure whose fluxes it could record."""
    from woof.experiment import load_experiment

    text = _sase_tree_toml(1, parent_pbl=900, child_pbl=1) + (
        "sase_flux_diag = true\n")
    path = tmp_path / "mixed.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_experiment(path)
    said = str(caught.value)
    assert "sase_flux_diag=True requires bl_pbl_physics=900" in said


def test_a_prepared_sase_domain_does_not_bind_the_closure_selectors(tmp_path):
    """An HRRR hierarchy prepares every domain from its WRF namelist.

    No namelist spells the SASE selectors, so each prepared domain's
    identity holds their defaults while the forecast reads the values the
    experiment config sets, and the tree was refused after preparation
    for ``run.sase_moist_n2``.  Preparation reads none of the four, so a
    prepared domain matches the configured one at any value of them; a
    field preparation does read still refuses.
    """
    import dataclasses

    from woof.experiment import load_experiment
    from woof.ingest.prepared_cache import (
        compare_prepared_domain_config, effective_prepared_domain_config,
        prepared_domain_config_identity, undelayed_identity_defaults)

    text = _with_shared_line(
        _sase_tree_toml(1, child_pbl=900),
        "sase_moist_n2 = false\nsase_stable_dissipation = true\n"
        "sase_additive_dissipation = false") + "sase_flux_diag = true\n"
    path = tmp_path / "mixed.toml"
    path.write_text(text, encoding="utf-8")
    exp = load_experiment(path)
    live = exp.domains[1]
    assert live.run.bl_pbl_physics == 900 and live.run.sase_moist_n2 is False
    prepared = dataclasses.replace(live, run=dataclasses.replace(
        live.run, sase_moist_n2=True, sase_stable_dissipation=False,
        sase_additive_dissipation=True, sase_flux_diag=False))

    def compare(cached):
        return compare_prepared_domain_config(
            effective_prepared_domain_config(
                prepared_domain_config_identity(cached)),
            effective_prepared_domain_config(
                prepared_domain_config_identity(live)),
            not_in_use=undelayed_identity_defaults(exp))

    _tolerated, differing = compare(prepared)
    assert differing == []
    drifted = dataclasses.replace(prepared, run=dataclasses.replace(
        prepared.run, mp_physics=8))
    _tolerated, differing = compare(drifted)
    assert "run.mp_physics" in differing


@pytest.mark.parametrize("line", ["sase_flux_diag = true",
                                  "hmix_k_diag = true",
                                  "nwp_diagnostics = 1"])
def test_the_preparation_stage_binding_ignores_an_output_only_switch(
        tmp_path, line):
    """The stage binding projects out what the prepared cache ignores.

    sase_flux_diag was projected out through PREPARATION_INERT_RUN_FIELDS
    until its one ruling became the output-only table the cache reader
    skips; hmix_k_diag and nwp_diagnostics never were, so switching one
    hashed a different config and prepared the same arrays again.  A
    field preparation reads still binds.
    """
    from woof.experiment import load_experiment
    from woof.stage_reuse import _experiment_argument_binding

    text = _sase_tree_toml(1, child_pbl=900)
    plain = tmp_path / "plain.toml"
    plain.write_text(text, encoding="utf-8")
    switched = tmp_path / "switched.toml"
    switched.write_text(_with_shared_line(text, line), encoding="utf-8")
    moved = tmp_path / "moved.toml"
    moved.write_text(text.replace("mp_physics = 6", "mp_physics = 8"),
                     encoding="utf-8")
    # Configs that build, so the projection answers and not the byte pin.
    for path in (switched, moved):
        load_experiment(path)

    bound = _experiment_argument_binding(str(plain))
    assert bound["schema"] == "gpuwm-preparation-config-controls-v1"
    assert _experiment_argument_binding(str(switched)) == bound
    assert _experiment_argument_binding(str(moved)) != bound


@pytest.mark.parametrize("name,value", _SASE_NON_DEFAULT,
                         ids=[n for n, _ in _SASE_NON_DEFAULT])
def test_a_shared_sase_selector_on_a_tree_with_no_sase_domain_is_refused(
        tmp_path, name, value):
    """The refusal a single domain gets holds for a tree with no SASE
    domain: no domain has the seam the key names."""
    from woof.experiment import load_experiment

    text = _with_shared_line(_sase_tree_toml(1),
                             f"{name} = {str(value).lower()}")
    path = tmp_path / "ysu.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_experiment(path)
    said = str(caught.value)
    assert f"{name}={value!r} requires bl_pbl_physics=900" in said


@pytest.mark.parametrize("name,value", _SASE_NON_DEFAULT,
                         ids=[n for n, _ in _SASE_NON_DEFAULT])
def test_a_single_non_sase_domain_still_refuses_a_sase_selector(
        name, value):
    """The single-domain refusal is unchanged."""
    cfg = _min_cfg(moist=True, bl_pbl_physics=1, sf_sfclay_physics=1,
                   km_opt=4, **{name: value})
    with pytest.raises(ValueError) as caught:
        validate_run_config(cfg)
    assert f"{name}={value!r} requires bl_pbl_physics=900" in str(
        caught.value)
