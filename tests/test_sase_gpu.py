"""SASE-L1 device kernels vs the frozen FP64 authority (stage-3 Task 2).

Parity semantics (S3-1 carry-forward): device inputs are the FP32 cast of
band-limited FP64 noise (authority ``box_filter`` of standard normals),
and the authority reference is evaluated on the SAME FP32-cast inputs
promoted to FP64 -- every gate therefore measures kernel arithmetic, not
input quantization.  The error metric is scale-relative,
``max|got - ref| / max|ref|``: the stencils cancel to near zero at
individual cells (measured in S3-1), so element-wise rtol is meaningless
there while a bound relative to the field scale is exactly what the FP32
production arithmetic can promise.

Gates (plan Task 2): filters/strain/stress <= 2e-6; structure-function
scalars <= 5e-6; Germano lift <= 1e-5 (two filter applications compound).
Trace identity ``tau_kk == 2e`` on device to <= 2 ULP FP32, elementwise.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.config import SASE_PBL_SCHEME as _SASE_SELECTOR

from conftest import requires_gpu
from sase_goldens import (GOLDEN_C_NU_FP64, GOLDEN_DEVICE_BY_CARD,
                          GOLDEN_F_FP64)

pytestmark = pytest.mark.gpu

SHAPE = (20, 48, 64)
DX = DY = 500.0
DZ = 200.0
DELTA = 500.0
SEED = 20260720


def _zero_radiation_boundary_fluxes(cfg, surface):
    """Complete the zero-radiation stub's declared history/energy outputs."""
    import cupy as cp
    from woof.core.physics import radiation_boundary_flux_names
    return {name: cp.zeros_like(surface) for name in radiation_boundary_flux_names(cfg)}


def _zero_radiation(**kw):
    """Shared zero-flux fixture for the admitted driver integrations."""
    import cupy as cp
    from woof.core.physics import RadiationResult
    cfg = kw["cfg"]
    z3 = cp.zeros((cfg.nz, cfg.ny, cfg.nx), cp.float32)
    z2 = cp.zeros((cfg.ny, cfg.nx), cp.float32)
    return RadiationResult(z3, cp.zeros_like(z3), z2, cp.zeros_like(z2),
                           coszen=cp.zeros_like(z2),
                           boundary_fluxes=_zero_radiation_boundary_fluxes(cfg, z2))


def _moist_driver_state(cfg, **vertical_kwargs):
    """The common deterministic sounding; each test sets its own winds/deck."""
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    coord = make_vertical_coord(cfg.nz, **vertical_kwargs)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.004 * np.asarray(z, np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    return init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.010 * np.exp(-np.asarray(z, np.float64) / 2400.0))


def _band_limited32(rng, shape=SHAPE):
    """FP32 cast of an authority-box-filtered standard-normal field."""
    from woof.verify.sase_ref import box_filter
    return box_filter(rng.standard_normal(shape), 4).astype(np.float32)


def _velocities32(seed=SEED):
    rng = np.random.default_rng(seed)
    return tuple(_band_limited32(rng) for _ in range(3))


def _stretched_dz(nz=SHAPE[0], t0=50.0, ratio=1.08):
    return t0 * ratio ** np.arange(nz, dtype=np.float64)


def _max_rel(got, ref):
    """(scale-relative error, max-abs error) of got against FP64 ref."""
    ref = np.asarray(ref, dtype=np.float64)
    err = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))
    return err / float(np.max(np.abs(ref))), err


@requires_gpu
@pytest.mark.parametrize("width", [2, 4])
def test_box_filter_parity(width):
    import cupy as cp
    from woof.core.sase import launch_box_filter
    from woof.verify.sase_ref import box_filter
    f32, _, _ = _velocities32()
    got = cp.asnumpy(launch_box_filter(cp.asarray(f32), width))
    ref = box_filter(f32.astype(np.float64), width)
    rel, err = _max_rel(got, ref)
    assert rel <= 2e-6, (f"box_filter width={width}: max_rel={rel:.3e} "
                         f"max_abs={err:.3e} (gate 2e-6)")


@requires_gpu
def test_box_filter_periodic_wrap_on_device():
    """Filter of a rolled field == rolled filter, bitwise (same FP32 ops)."""
    import cupy as cp
    from woof.core.sase import launch_box_filter
    f32, _, _ = _velocities32()
    for width in (2, 4):
        base = cp.asnumpy(launch_box_filter(cp.asarray(f32), width))
        rolled = cp.asnumpy(launch_box_filter(
            cp.asarray(np.roll(np.roll(f32, 5, axis=2), 3, axis=1)), width))
        np.testing.assert_array_equal(
            rolled, np.roll(np.roll(base, 5, axis=2), 3, axis=1))


@requires_gpu
def test_structure_functions_parity():
    import cupy as cp
    from woof.core.sase import launch_structure_functions
    from woof.verify.sase_ref import structure_functions
    u32, v32, w32 = _velocities32()
    got = launch_structure_functions(*(cp.asarray(a) for a in (u32, v32, w32)))
    ref = structure_functions(u32, v32, w32)   # promotes to FP64 internally
    assert set(got) == {1, 2, 4}
    for r in (1, 2, 4):
        rel = abs(got[r] - ref[r]) / abs(ref[r])
        assert rel <= 5e-6, (f"D2(r={r}): device={got[r]!r} ref={ref[r]!r} "
                             f"rel={rel:.3e} (gate 5e-6)")


@requires_gpu
def test_strain_parity_uniform_dz():
    import cupy as cp
    from woof.core.sase import launch_strain
    from woof.verify.sase_ref import strain
    u32, v32, w32 = _velocities32()
    got = launch_strain(*(cp.asarray(a) for a in (u32, v32, w32)),
                        dx=DX, dy=DY, dz=DZ)
    ref = strain(u32, v32, w32, DX, DY, DZ)
    names = ("xx", "yy", "zz", "xy", "xz", "yz")
    for name, g, r in zip(names, got, ref):
        rel, err = _max_rel(cp.asnumpy(g), r)
        assert rel <= 2e-6, (f"strain[{name}] uniform dz: max_rel={rel:.3e} "
                             f"max_abs={err:.3e} (gate 2e-6)")


@requires_gpu
def test_strain_parity_stretched_dz_col():
    """Variable-dz clamped-z strain on geometric 1.08 thicknesses from 50 m.

    The authority gets dz=NaN, proving the uniform spacing argument is
    never consulted on the variable path (S3-1 convention).
    """
    import cupy as cp
    from woof.core.sase import launch_strain
    from woof.verify.sase_ref import strain
    u32, v32, w32 = _velocities32()
    t = _stretched_dz()
    got = launch_strain(*(cp.asarray(a) for a in (u32, v32, w32)),
                        dx=DX, dy=DY, dz_col=t)
    ref = strain(u32, v32, w32, DX, DY, float("nan"), dz_col=t)
    names = ("xx", "yy", "zz", "xy", "xz", "yz")
    for name, g, r in zip(names, got, ref):
        rel, err = _max_rel(cp.asnumpy(g), r)
        assert rel <= 2e-6, (f"strain[{name}] stretched dz_col: "
                             f"max_rel={rel:.3e} max_abs={err:.3e} "
                             f"(gate 2e-6)")


@requires_gpu
def test_germano_lift_parity_width2():
    import cupy as cp
    from woof.core.sase import launch_germano_lift
    from woof.verify.sase_ref import germano_lift
    u32, v32, w32 = _velocities32()
    got = launch_germano_lift(*(cp.asarray(a) for a in (u32, v32, w32)),
                              width=2)
    ref = germano_lift(u32, v32, w32)          # promotes to FP64 internally
    for k, (g, r) in enumerate(zip(got, ref)):
        rel, err = _max_rel(cp.asnumpy(g), r)
        assert rel <= 1e-5, (f"lift[{k}] width=2: max_rel={rel:.3e} "
                             f"max_abs={err:.3e} (gate 1e-5)")


@requires_gpu
def test_germano_lift_parity_width4():
    """Width-parameterized lift vs the authority's general construction
    (the `_identity_rows` form; germano_lift is its width-2 special case).
    """
    import cupy as cp
    from woof.core.sase import launch_germano_lift
    from woof.verify.sase_ref import _PAIRS, box_filter
    u32, v32, w32 = _velocities32()
    got = launch_germano_lift(*(cp.asarray(a) for a in (u32, v32, w32)),
                              width=4)
    vel = [a.astype(np.float64) for a in (u32, v32, w32)]
    filt = [box_filter(a, 4) for a in vel]
    ref = [box_filter(vel[i] * vel[j], 4) - filt[i] * filt[j]
           for i, j in _PAIRS]
    for k, (g, r) in enumerate(zip(got, ref)):
        rel, err = _max_rel(cp.asnumpy(g), r)
        assert rel <= 1e-5, (f"lift[{k}] width=4: max_rel={rel:.3e} "
                             f"max_abs={err:.3e} (gate 1e-5)")


def _e_field32(seed=SEED + 1):
    """Band-limited positive e with a few cells forced under E_MIN."""
    from woof.verify.sase_ref import box_filter
    rng = np.random.default_rng(seed)
    e = 0.2 + 0.05 * box_filter(rng.standard_normal(SHAPE), 4)
    e32 = e.astype(np.float32)
    e32[0, :2, :2] = 0.0                       # exercises the E_MIN floor
    return e32


@requires_gpu
def test_model_stress_parity():
    import cupy as cp
    from woof.core.sase import launch_model_stress
    from woof.verify.sase_ref import model_stress, strain
    u32, v32, w32 = _velocities32()
    s32 = [s.astype(np.float32)
           for s in strain(u32, v32, w32, DX, DY, DZ)]
    e32 = _e_field32()
    c_nu, f = 0.12, 0.6
    got = launch_model_stress(cp.asarray(e32),
                              [cp.asarray(s) for s in s32],
                              c_nu, f, 2.0 * DELTA, DELTA)
    ref = model_stress(e32.astype(np.float64),
                       [s.astype(np.float64) for s in s32],
                       c_nu, f, 2.0 * DELTA, DELTA)
    names = ("xx", "yy", "zz", "xy", "xz", "yz")
    for name, g, r in zip(names, got, ref):
        rel, err = _max_rel(cp.asnumpy(g), r)
        assert rel <= 2e-6, (f"tau[{name}]: max_rel={rel:.3e} "
                             f"max_abs={err:.3e} (gate 2e-6)")


@requires_gpu
def test_model_stress_trace_identity_device_ulp():
    """Realizability contract: tau_kk == 2*max(e, E_MIN) on device.

    Asserted elementwise to <= 2 ULP FP32, with ULP measured at the
    magnitude of the participating terms (the largest of |tau_xx|,
    |tau_yy|, |tau_zz|, 2e per cell).  That is the only meaningful FP32
    statement: storing any exact tau in FP32 already costs up to 0.5 ULP
    at |tau_ii| per component, so a bound in ULPs of 2e alone would be
    unsatisfiable by ANY FP32 stress field wherever |tau| >> 2e.  The
    kernel closes tau_zz from the trace identity (algebraically identical
    to the authority's expression), which bounds the residual by ~1.5 ULP
    by construction.  Strain comes from the device kernel (stretched
    dz_col), so this exercises the full device strain->stress chain.
    """
    import cupy as cp
    from woof.core.sase import launch_model_stress, launch_strain
    from woof.verify.sase_ref import E_MIN
    u32, v32, w32 = _velocities32()
    s_dev = launch_strain(*(cp.asarray(a) for a in (u32, v32, w32)),
                          dx=DX, dy=DY, dz_col=_stretched_dz())
    e32 = _e_field32()
    tau = launch_model_stress(cp.asarray(e32), s_dev, 0.12, 0.6,
                              2.0 * DELTA, DELTA)
    txx, tyy, tzz = (cp.asnumpy(t) for t in tau[:3])
    e_floor = np.maximum(e32, np.float32(E_MIN))
    target = 2.0 * e_floor.astype(np.float64)  # exact in FP32 and FP64
    trace = (txx.astype(np.float64) + tyy.astype(np.float64)
             + tzz.astype(np.float64))
    mag32 = np.max(np.stack([np.abs(txx), np.abs(tyy), np.abs(tzz),
                             (2.0 * e_floor).astype(np.float32)]), axis=0)
    ulp = np.spacing(mag32).astype(np.float64)
    resid = np.abs(trace - target)
    worst = float(np.max(resid / ulp))
    assert np.all(resid <= 2.0 * ulp), (
        f"trace identity: worst residual {worst:.3f} ULP "
        f"(max_abs={float(resid.max()):.3e}, gate 2 ULP FP32 at the "
        f"participating magnitude)")
    # The floored cells must satisfy the identity against E_MIN, not raw e.
    assert np.all(resid[0, :2, :2] <= 2.0 * ulp[0, :2, :2])


@requires_gpu
def test_governed_stress_parity_and_f1_reduction():
    """S3-6e ``launch_governed_stress`` vs the authority
    ``governed_stress``: six stress components and the governed
    diffusivity field km at the 2e-6 stress gate; the smag share r
    (dimensionless, in [0, 1]) at 2e-6 max-abs.  At f = 1 the governed
    kernel must reduce to ``sase_model_stress`` (delta/delta) bitwise --
    the FP-exact 0*K_smag no-op -- with r identically zero."""
    import cupy as cp
    from woof.core.sase import launch_governed_stress, launch_model_stress
    from woof.verify.sase_ref import governed_stress, strain
    u32, v32, w32 = _velocities32()
    s32 = [s.astype(np.float32)
           for s in strain(u32, v32, w32, DX, DY, DZ)]
    e32 = _e_field32()
    c_nu, f = 0.12, 0.6
    s_dev = [cp.asarray(s) for s in s32]
    taus, km, r = launch_governed_stress(cp.asarray(e32), s_dev,
                                         c_nu, f, DELTA)
    ref_tau, ref_km, ref_r = governed_stress(
        e32.astype(np.float64), [s.astype(np.float64) for s in s32],
        c_nu, f, DELTA)
    for name, g, rr in zip(("xx", "yy", "zz", "xy", "xz", "yz"),
                           taus, ref_tau):
        rel, err = _max_rel(cp.asnumpy(g), rr)
        assert rel <= 2e-6, (f"gov tau[{name}]: max_rel={rel:.3e} "
                             f"max_abs={err:.3e} (gate 2e-6)")
    rel, err = _max_rel(cp.asnumpy(km), ref_km)
    assert rel <= 2e-6, (f"gov km: max_rel={rel:.3e} max_abs={err:.3e}")
    r_host = cp.asnumpy(r).astype(np.float64)
    assert float(np.max(np.abs(r_host - ref_r))) <= 2e-6
    assert r_host.min() >= 0.0 and r_host.max() <= 1.0
    # f = 1 bitwise reduction to the v0 stress at delta/delta.
    taus1, km1, r1 = launch_governed_stress(cp.asarray(e32), s_dev,
                                            c_nu, 1.0, DELTA)
    tau_v0 = launch_model_stress(cp.asarray(e32), s_dev, c_nu, 1.0,
                                 DELTA, DELTA)
    for g, want in zip(taus1, tau_v0):
        np.testing.assert_array_equal(cp.asnumpy(g), cp.asnumpy(want))
    assert float(cp.abs(r1).max()) == 0.0


@requires_gpu
def test_launcher_validation():
    import cupy as cp
    from woof.core.sase import launch_box_filter, launch_strain
    u = cp.zeros(SHAPE, dtype=cp.float32)
    with pytest.raises(ValueError, match="width"):
        launch_box_filter(u, 3)
    with pytest.raises(ValueError, match="dz"):
        launch_strain(u, u, u, dx=DX, dy=DY)               # neither
    with pytest.raises(ValueError, match="dz"):
        launch_strain(u, u, u, dx=DX, dy=DY, dz=DZ,
                      dz_col=_stretched_dz())              # both
    with pytest.raises(ValueError, match="dz_col"):
        # 3-D thicknesses must be a DEVICE float32 field (the on-device
        # FP64 coefficient build); a host array is rejected loudly.
        launch_strain(u, u, u, dx=DX, dy=DY,
                      dz_col=np.full(SHAPE, 200.0))
    with pytest.raises(ValueError, match="dz_col"):
        # Wrong-shaped device 3-D thicknesses are rejected too.
        launch_strain(u, u, u, dx=DX, dy=DY,
                      dz_col=cp.full((SHAPE[0], 2, 2), 200.0,
                                     dtype=cp.float32))
    bad = cp.zeros(SHAPE, dtype=cp.float64)
    with pytest.raises(ValueError, match="float32"):
        launch_box_filter(bad, 2)


# ---------------------------------------------------------------------------
# Stage-3 Task 3: device dynamic partition solve + sensor state.
#
# The five Gram/projection scalars (a.a, a.b, b.b, a.r, b.r) are
# accumulated on device in FP64 (structure-partial reduction idiom); the
# host finishes with the authority's own 2x2 tail (np.linalg.cond gate at
# 1e12, np.linalg.solve, the clip/recovery order of dynamic_solve),
# line-identical arithmetic on the device-summed moments.
# ---------------------------------------------------------------------------


@requires_gpu
def test_sase_tpb_single_source():
    """The device SASE_TPB/SASE_KMAX and the host constants cannot drift.

    The values are injected as a compile-time tier through this head's
    per-translation-unit define mechanism, so the closure carries its own
    tier and no scheme name reaches the generic loader.  sase.cu must not
    hard-define either name, and the host launch constants must be read
    from the same tuple that is handed to the compiler.
    """
    from pathlib import Path
    import woof.core.kernels as kernels
    import woof.core.sase as sase
    src = (Path(kernels.__file__).parent / "sase.cu").read_text()
    # The source carries #ifndef-guarded defaults -- the kf.cu/refl.cu
    # idiom -- so the plain loader can compile it and the local-frame
    # census can measure it.  What must never happen is the guard being
    # dropped (the tier would stop reaching the device) or the default
    # drifting from the tier (the census would measure a frame no run
    # ever has).  Both are pinned here.
    for name in ("SASE_TPB", "SASE_KMAX"):
        assert f"#ifndef {name}" in src, (
            f"sase.cu must guard {name} so the launcher's tier wins")
        assert name in sase._DEFINE_VALUES
        default = next(
            int(line.split()[-1]) for line in src.splitlines()
            if line.startswith(f"#define {name} "))
        assert default == sase._DEFINE_VALUES[name], (
            f"{name} in-source default {default} has drifted from the "
            f"launcher tier {sase._DEFINE_VALUES[name]}")
    assert sase._TPB == sase._DEFINE_VALUES["SASE_TPB"]
    assert sase._KMAX == sase._DEFINE_VALUES["SASE_KMAX"]
    # The tier is what the loader is actually handed, not a second copy.
    assert dict(sase._INT_DEFINES) == sase._DEFINE_VALUES
    # Every int define sizes a power-of-two shared-memory tree reduction,
    # and the int/float define namespaces must never collide (that would
    # emit two #defines for one name, last one silently winning).
    from woof.core.constants import CUDA_DEFINES
    assert sase._DEFINE_VALUES.keys().isdisjoint(CUDA_DEFINES)
    assert all(v > 0 and (v & (v - 1)) == 0
               for v in sase._DEFINE_VALUES.values())


def _solve_fixture32(seed, shape, e_offset, e_amp, e_floor0):
    """FP32 casts of the authority's band-limited solve fixtures."""
    from woof.verify.sase_ref import box_filter
    rng = np.random.default_rng(seed)
    u, v, w = (box_filter(rng.standard_normal(shape), 4).astype(np.float32)
               for _ in range(3))
    e = e_offset + e_amp * box_filter(rng.standard_normal(shape), 4)
    if e_floor0:
        e = np.maximum(e, 0.0)
    return u, v, w, e.astype(np.float32)


@requires_gpu
def test_dynamic_solve_device_recovers_manufactured_coefficients():
    """Inverse crime on device: manufactured lifts built from the DEVICE
    model-stress kernels at both test levels; the device solve must
    recover (c_nu, f) = (0.17, 0.55) to rel <= 1e-5 (FP32 forward data,
    FP64 in-kernel accumulation).  Mirrors the authority fixture
    (test_dynamic_solve_recovers_manufactured_coefficients) including the
    scale split: coarse tau rides width*dx, the momentum background stays
    grid-anchored at dx, and e is non-uniform so the grid-anchored column
    is observable.
    """
    import cupy as cp
    from woof.core.sase import (launch_box_filter, launch_dynamic_solve,
                                 launch_model_stress, launch_strain)
    shape = (6, 24, 24)
    u32, v32, w32, e32 = _solve_fixture32(20260720, shape, 0.25, 0.05, False)
    assert e32.min() > 1.0e-6                  # fixture stays realizable
    c_true, f_true = 0.17, 0.55
    dx = dy = 500.0
    dz = 200.0
    dev = [cp.asarray(a) for a in (u32, v32, w32)]
    e_dev = cp.asarray(e32)
    s_fine = launch_strain(*dev, dx=dx, dy=dy, dz=dz)
    lifts = {}
    for width in (2, 4):
        filt = [launch_box_filter(a, width) for a in dev]
        s_coarse = launch_strain(*filt, dx=dx, dy=dy, dz=dz)
        coarse = launch_model_stress(e_dev, s_coarse, c_true, f_true,
                                     width * dx, dx)
        fine = [launch_box_filter(t, width)
                for t in launch_model_stress(e_dev, s_fine, c_true, f_true,
                                             dx, dx)]
        lifts[width] = [cp.ascontiguousarray(c - f_)
                        for c, f_ in zip(coarse, fine)]
    c_nu, f = launch_dynamic_solve(*dev, e_dev, dx=dx, dy=dy, dz=dz,
                                   delta=dx, manufactured_lifts=lifts)
    rel_c = abs(c_nu - c_true) / c_true
    rel_f = abs(f - f_true) / f_true
    assert rel_c <= 1e-5 and rel_f <= 1e-5, (
        f"manufactured recovery: c_nu={c_nu!r} (rel {rel_c:.3e}), "
        f"f={f!r} (rel {rel_f:.3e}), gate 1e-5")


#: The FP64 authority goldens and the pinned FP32 device goldens for the
#: frozen seed-3 varying-e fixture live in tests/sase_goldens.py, shared
#: with the CPU suite's authority pin (S3-3 review fold-in).


@requires_gpu
def test_dynamic_solve_device_real_lift_golden():
    """Real-lift golden gate: the device solve on the FP32 cast of the
    frozen seed-3 fixture lands within rel 5e-4 of the FP64 goldens
    (input quantization + FP32 forward arithmetic both included), and
    reproduces the pinned device goldens of THIS card exactly
    (deterministic reduction; drift canary for kernel/toolchain changes).
    The pair is per card (sase_goldens.GOLDEN_DEVICE_BY_CARD: before A146
    the 5090 pair failed on an RTX 4090 by rel 1.48e-08 in f; since, the
    5090, 5070 Ti and 4090 read one pair); a card with no pair
    skips, naming itself, rather than failing for the card and not the
    code, and the release card stage names that skip in its receipt.
    """
    import cupy as cp
    from woof.core.sase import launch_dynamic_solve
    props = cp.cuda.runtime.getDeviceProperties(0)
    card = props["name"]
    card = card.decode() if isinstance(card, bytes) else str(card)
    if card not in GOLDEN_DEVICE_BY_CARD:
        pytest.skip(f"the SASE device golden pair is per card and none is recorded for "
                    f"{card!r} (compute capability {props['major']}.{props['minor']}); "
                    f"pairs exist for {sorted(GOLDEN_DEVICE_BY_CARD)}.  Record this card's "
                    "(c_nu, f) from two processes in tests/sase_goldens.py with the reading "
                    "against the FP64 authority and the nearest recorded card.")
    golden_c_nu, golden_f = GOLDEN_DEVICE_BY_CARD[card]
    u32, v32, w32, e32 = _solve_fixture32(3, (8, 24, 24), 0.05, 0.1, True)
    dev = [cp.asarray(a) for a in (u32, v32, w32, e32)]
    c_nu, f = launch_dynamic_solve(*dev, dx=500.0, dy=500.0, dz=200.0,
                                   delta=500.0)
    rel_c = abs(c_nu - GOLDEN_C_NU_FP64) / GOLDEN_C_NU_FP64
    rel_f = abs(f - GOLDEN_F_FP64) / GOLDEN_F_FP64
    assert rel_c <= 5e-4 and rel_f <= 5e-4, (
        f"real-lift golden: c_nu={c_nu!r} (rel {rel_c:.3e}), "
        f"f={f!r} (rel {rel_f:.3e}), gate 5e-4")
    np.testing.assert_allclose(c_nu, golden_c_nu, rtol=1e-9,
                               err_msg=f"c_nu moved on {card}; a deliberate change re-pins "
                                       "GOLDEN_DEVICE_BY_CARD with its reading")
    np.testing.assert_allclose(f, golden_f, rtol=1e-9,
                               err_msg=f"f moved on {card}; a deliberate change re-pins "
                                       "GOLDEN_DEVICE_BY_CARD with its reading")
    # S3-3 review fold-in: the golden pin rests on launch-to-launch
    # bitwise determinism (fixed block count and reduction order, host
    # np sum) -- assert it directly with a second invocation.
    assert launch_dynamic_solve(*dev, dx=500.0, dy=500.0, dz=200.0,
                                delta=500.0) == (c_nu, f)


@requires_gpu
def test_dynamic_solve_device_degenerate_uniform_e():
    """Uniform e: the grid-anchored momentum basis commutes with the test
    filter and vanishes from the lift, so the Gram matrix is numerically
    rank-1 and the cond > 1e12 gate must return exactly (0.0, 0.0).
    Checked with live velocities (FP32 non-commutation noise is the only
    content of the b column) and with zero velocities (Gram exactly
    zero).
    """
    import cupy as cp
    from woof.core.sase import launch_dynamic_solve
    u32, v32, w32 = _velocities32()
    e_uni = cp.full(SHAPE, 0.3, dtype=cp.float32)
    got = launch_dynamic_solve(
        *(cp.asarray(a) for a in (u32, v32, w32)), e_uni,
        dx=DX, dy=DY, dz=DZ, delta=DELTA)
    assert got == (0.0, 0.0), f"uniform-e live velocities: got {got!r}"
    zeros = cp.zeros(SHAPE, dtype=cp.float32)
    got0 = launch_dynamic_solve(zeros, zeros, zeros, e_uni,
                                dx=DX, dy=DY, dz=DZ, delta=DELTA)
    assert got0 == (0.0, 0.0), f"uniform-e zero velocities: got {got0!r}"


@requires_gpu
def test_sensor_state_device_parity():
    """sensor_state from device D2 sums: scalars match the authority (on
    the same FP32-cast inputs) to rel <= 5e-6; the host tail applies the
    authority's formulas to the device-reduced structure functions.
    """
    import cupy as cp
    from woof.core.sase import launch_sensor_state
    from woof.verify.sase_ref import sensor_state
    u32, v32, w32 = _velocities32()
    e_mean = 0.23
    got = launch_sensor_state(
        *(cp.asarray(a) for a in (u32, v32, w32)), e_mean=e_mean)
    ref = sensor_state(u32, v32, w32, e_mean)  # promotes to FP64 internally
    for name in ("alpha", "slope", "e_res"):
        g, r = getattr(got, name), getattr(ref, name)
        rel = abs(g - r) / abs(r)
        assert rel <= 5e-6, (f"sensor_state.{name}: device={g!r} ref={r!r} "
                             f"rel={rel:.3e} (gate 5e-6)")


@requires_gpu
def test_sensor_state_device_degenerate():
    """Zero resolved field: everything subgrid -- alpha=1, slope=0 exactly
    (the authority's degenerate branch, taken on the device D2 sums)."""
    import cupy as cp
    from woof.core.sase import launch_sensor_state
    zeros = cp.zeros(SHAPE, dtype=cp.float32)
    got = launch_sensor_state(zeros, zeros, zeros, e_mean=0.5)
    assert got.alpha == 1.0 and got.slope == 0.0 and got.e_res == 0.0


@requires_gpu
def test_dynamic_solve_launcher_validation():
    import cupy as cp
    from woof.core.sase import launch_dynamic_solve
    u = cp.zeros(SHAPE, dtype=cp.float32)
    with pytest.raises(ValueError, match="manufactured_lifts"):
        launch_dynamic_solve(u, u, u, u, dx=DX, dy=DY, dz=DZ, delta=DELTA,
                             manufactured_lifts={2: [u] * 6})   # missing 4
    with pytest.raises(ValueError, match="manufactured_lifts"):
        launch_dynamic_solve(u, u, u, u, dx=DX, dy=DY, dz=DZ, delta=DELTA,
                             manufactured_lifts={2: [u] * 6,
                                                 4: [u] * 5})   # 5 comps
    bad = cp.zeros(SHAPE, dtype=cp.float64)
    with pytest.raises(ValueError, match="float32"):
        launch_dynamic_solve(u, u, u, bad, dx=DX, dy=DY, dz=DZ, delta=DELTA)


# ---------------------------------------------------------------------------
# S3-6c: fused device SPLIT step + implicit vertical channel.
#
# launch_sase_step now mirrors the S3-6b authority ``sase_split_step``
# (solve -> vertical channel -> stress -> horizontal-explicit tendencies
# -> FP64 per-column Thomas momentum solve -> implicit-flux production
# -> e update/clip-to-heat -> implicit e-transport, in place, FP64
# split-ledger reductions).  The v0 explicit step and its gates are
# RETIRED (S3-6b report section 7); the retargeted gates are: (a)
# 10-step trajectory parity against the FP64 authority split step; (b)
# 2-step trajectory parity against the pinned SPLIT_* goldens (BOX +
# COL); (c) the FP32 split-ledger closure characterization on the
# horizontally periodic uniform-dz CLAMPED box -- supersedes the v0
# characterized bound as the spec 4.2 artifact; (d) the nz=1 RANS decay
# law; (e) Thomas solver unit parity on stretched columns; (f) the
# d01-column stability fixture (the configuration the explicit path
# could not run).
# ---------------------------------------------------------------------------


STEP_SHAPE = (12, 24, 32)
STEP_DT = 0.5                                  # exactly representable


def _step_fixture32(seed=SEED + 2, shape=STEP_SHAPE):
    """Uniform-box step fixture: band-limited FP32 velocities, varying
    theta (live buoyancy), mixed-sign n2 (both dissipation-length
    branches), and sub-floor e cells (clip-to-heat engaged), mirroring
    the authority's varying-e ledger fixture."""
    from woof.verify.sase_ref import E_MIN, box_filter
    rng = np.random.default_rng(seed)

    def band():
        return box_filter(rng.standard_normal(shape), 4)

    u, v, w = (band().astype(np.float32) for _ in range(3))
    theta = (300.0 + 2.0 * band()).astype(np.float32)
    e = np.maximum(0.05 + 0.1 * band(), 0.0).astype(np.float32)
    n2 = (1.0e-4 * band()).astype(np.float32)
    assert np.any(e < E_MIN)                   # clip channel live
    assert np.any(n2 > 0.0) and np.any(n2 <= 0.0)   # both l branches
    return u, v, w, theta, e, n2


@requires_gpu
def test_sase_split_step_trajectory_parity_10_steps():
    """Gate (a): 10 fused device split steps track the FP64
    ``sase_split_step`` on the uniform clamped box to max-rel <= 5e-5
    per field (scale-relative; FP32 drift compounds), per-step growth
    recorded.  Both trajectories start from the same FP32-cast fields;
    the reference feeds its own FP64 fields forward, so the measurement
    is accurate compounding drift, not per-step arithmetic alone."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    from woof.verify.sase_ref import sase_split_step
    u32, v32, w32, th32, e32, n232 = _step_fixture32()
    dev = {n: cp.asarray(a)
           for n, a in zip("uvwte", (u32, v32, w32, th32, e32))}
    n2_dev = cp.asarray(n232)
    th64 = th32.astype(np.float64)
    n264 = n232.astype(np.float64)
    ref = {n: a.astype(np.float64)
           for n, a in zip("uvwe", (u32, v32, w32, e32))}
    growth = []
    for step in range(10):
        launch_sase_step(dev["u"], dev["v"], dev["w"], dev["t"], dev["e"],
                         dx=DX, dy=DY, dz=DZ, delta=DELTA, dt=STEP_DT,
                         n2=n2_dev)
        fields, _ = sase_split_step(ref["u"], ref["v"], ref["w"], th64,
                                    ref["e"], dx=DX, dy=DY, dz=DZ,
                                    delta=DELTA, dt=STEP_DT, n2=n264)
        ref = {n: fields[n] for n in "uvwe"}
        rels = {n: _max_rel(cp.asnumpy(dev[n]), ref[n])[0] for n in "uvwe"}
        growth.append(rels)
        print(f"step {step + 1:2d}: "
              + "  ".join(f"{n}={rels[n]:.3e}" for n in "uvwe"))
    worst = max(r for rels in growth for r in rels.values())
    assert worst <= 5e-5, (
        f"10-step split trajectory parity: worst max-rel {worst:.3e} "
        f"(gate 5e-5); per-step growth {growth}")


@requires_gpu
def test_sase_split_step_trajectory_goldens_parity():
    """Gate (b): the SPLIT_* trajectory goldens on device (S3-6b parity
    targets).  Two device split steps on the FP32 cast of the frozen
    seed-20260720 (8, 16, 16) fixture, BOX (dz=200, dt=0.5) and COL
    (1.08-geometric from 50 m, dt=0.05) modes: (i) fieldwise max-rel vs
    the FP64 authority stepped from the SAME FP32-cast inputs <= 5e-5
    (the trajectory-gate tier); (ii) device FP64 field sums vs the
    pinned FP64-input golden literals to rel <= 5e-5 (input quantization
    included); (iii) the step-2 (c_nu, f) vs the golden pair.

    S3-6j: the fixture carries the SPLIT_GOLDEN_UST drag field (the
    goldens re-pin, rationale in tests/sase_goldens.py), so this gate
    now certifies the device drag row against the amended authority
    on both thickness modes.  S3-9c: it also carries the
    SPLIT_GOLDEN_GUST enhanced-speed field (sfclay convention from
    the frozen initial winds -- shared construction with the CPU
    suite, rationale in tests/sase_goldens.py), certifying the device
    gustiness-correction factor the same way."""
    import cupy as cp
    from sase_goldens import (SPLIT_BOX_C_NU, SPLIT_BOX_F,
                              SPLIT_BOX_SUM_E, SPLIT_BOX_SUM_U,
                              SPLIT_BOX_SUM_V, SPLIT_BOX_SUM_W,
                              SPLIT_COL_C_NU, SPLIT_COL_F,
                              SPLIT_COL_SUM_E, SPLIT_COL_SUM_U,
                              SPLIT_COL_SUM_V, SPLIT_COL_SUM_W,
                              SPLIT_GOLDEN_GUST, SPLIT_GOLDEN_UST)

    from woof.core.sase import launch_sase_step
    from woof.verify.sase_ref import (E_MIN, SFC_WSPD_FLOOR, box_filter,
                                       sase_split_step)

    def fixture32():
        rng = np.random.default_rng(20260720)
        shape = (8, 16, 16)

        def band():
            return box_filter(rng.standard_normal(shape), 4)

        u, v, w = band(), band(), band()
        theta = 300.0 + 2.0 * band()
        e = np.maximum(0.05 + 0.1 * band(), 0.0)
        n2 = 1.0e-4 * band()
        # S3-6j: the goldens fixture engages the drag row (np.full
        # AFTER the rng draws -- shared literal with the CPU suite).
        ust = np.full(shape[1:], SPLIT_GOLDEN_UST)
        # S3-9c: sfclay-convention enhanced speed from the frozen
        # initial winds (shared literal + construction with the CPU
        # suite), engaging the gustiness correction.
        wspd = np.maximum(
            np.sqrt(u[0] ** 2 + v[0] ** 2 + SPLIT_GOLDEN_GUST ** 2),
            SFC_WSPD_FLOOR)
        assert np.any(e < E_MIN)
        assert np.any(n2 > 0.0) and np.any(n2 <= 0.0)
        return [a.astype(np.float32)
                for a in (u, v, w, theta, e, n2, ust, wspd)]

    def run(dz_col, dt):
        u32, v32, w32, th32, e32, n232, ust32, wspd32 = fixture32()
        dev = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
        n2_dev = cp.asarray(n232)
        ust_dev = cp.asarray(ust32)
        wspd_dev = cp.asarray(wspd32)
        th64, n264 = th32.astype(np.float64), n232.astype(np.float64)
        ust64 = ust32.astype(np.float64)
        wspd64 = wspd32.astype(np.float64)
        ref = {n: a.astype(np.float64)
               for n, a in zip("uvwe", (u32, v32, w32, e32))}
        ledger = None
        for _ in range(2):
            ledger = launch_sase_step(
                dev[0], dev[1], dev[2], dev[3], dev[4], dx=500.0, dy=500.0,
                dz=200.0, delta=500.0, dt=dt, n2=n2_dev, dz_col=dz_col,
                ust=ust_dev, wspd_sfc=wspd_dev)
            fields, _ = sase_split_step(
                ref["u"], ref["v"], ref["w"], th64, ref["e"], dx=500.0,
                dy=500.0, dz=200.0, delta=500.0, dt=dt, n2=n264,
                dz_col=dz_col, ust=ust64, wspd_sfc=wspd64)
            ref = {n: fields[n] for n in "uvwe"}
        rels = {n: _max_rel(cp.asnumpy(d), ref[n])[0]
                for n, d in zip("uvwe", (dev[0], dev[1], dev[2], dev[4]))}
        sums = tuple(float(cp.asnumpy(d).astype(np.float64).sum())
                     for d in (dev[0], dev[1], dev[2], dev[4]))
        return rels, sums, ledger

    for label, dz_col, dt, golden_sums, golden_cf in (
            ("BOX", None, 0.5,
             (SPLIT_BOX_SUM_U, SPLIT_BOX_SUM_V, SPLIT_BOX_SUM_W,
              SPLIT_BOX_SUM_E), (SPLIT_BOX_C_NU, SPLIT_BOX_F)),
            ("COL", _stretched_dz(8), 0.05,
             (SPLIT_COL_SUM_U, SPLIT_COL_SUM_V, SPLIT_COL_SUM_W,
              SPLIT_COL_SUM_E), (SPLIT_COL_C_NU, SPLIT_COL_F))):
        rels, sums, ledger = run(dz_col, dt)
        worst = max(rels.values())
        print(f"{label}: fieldwise " + "  ".join(
            f"{n}={rels[n]:.3e}" for n in "uvwe"))
        assert worst <= 5e-5, (
            f"{label} fieldwise split parity: worst {worst:.3e} (gate 5e-5)")
        for name, got, want in zip("uvwe", sums, golden_sums):
            rel = abs(got - want) / abs(want)
            print(f"{label} sum[{name}]: device={got!r} golden={want!r} "
                  f"rel={rel:.3e}")
            assert rel <= 5e-5, (
                f"{label} golden sum {name}: rel {rel:.3e} (gate 5e-5)")
        np.testing.assert_allclose((ledger["c_nu"], ledger["f"]),
                                   golden_cf, rtol=5e-4)


@requires_gpu
def test_sase_split_step_surface_drag_parity_and_closure():
    """S3-6j device gate: the drag-engaged fused step vs the amended
    FP64 authority.  Five compounding steps on the STEP_SHAPE fixture
    with a varying (ny, nx) ust field (0.25 + 0.2*|band|, floor-active
    columns included) and -- S3-9c -- a varying gust-enhanced wspd
    field (sfclay convention, vconv-class augmentation 0.3 + |band|:
    the device gustiness factor compounds through all five steps
    against the FP64 authority, gusty and floor-active columns
    included): (i) fieldwise trajectory parity <= 5e-5 per
    step; (ii) the BOUNDARY-CONSISTENT FP32 ledger closure
    |dKE + dE + dHeat - dKE_sfc|/scale <= 1e-5 every step (the spec
    tier of gate (c), now with the boundary channel live); (iii) the
    device dKE_sfc tracks the authority's (drag work is a leading
    channel, not noise) and dE_sfc_src matches the authority FP64
    reduction; (iv) ust=None on device remains the no-drag step with
    all surface channels exactly 0.0; (v) launcher validation
    (S3-9c: wspd_sfc shape/dtype and the wspd-without-ust
    rejection)."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    from woof.verify.sase_ref import (SFC_WSPD_FLOOR, box_filter,
                                       sase_split_step)
    u32, v32, w32, th32, e32, n232 = _step_fixture32(seed=SEED + 6)
    rng = np.random.default_rng(SEED + 7)
    ust32 = (0.25 + 0.2 * np.abs(
        box_filter(rng.standard_normal(STEP_SHAPE), 4)[0]))\
        .astype(np.float32)
    # S3-9c: sfclay-convention enhanced speed at a varying vconv-class
    # augmentation (frozen across the compounding steps, like ust).
    vconv = 0.3 + np.abs(box_filter(rng.standard_normal(STEP_SHAPE),
                                    4)[0])
    wspd32 = np.maximum(
        np.sqrt(u32.astype(np.float64)[0] ** 2
                + v32.astype(np.float64)[0] ** 2 + vconv ** 2),
        SFC_WSPD_FLOOR).astype(np.float32)
    dev = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
    n2_dev = cp.asarray(n232)
    ust_dev = cp.ascontiguousarray(cp.asarray(ust32))
    wspd_dev = cp.ascontiguousarray(cp.asarray(wspd32))
    th64, n264 = th32.astype(np.float64), n232.astype(np.float64)
    ust64 = ust32.astype(np.float64)
    wspd64 = wspd32.astype(np.float64)
    ref = {n: a.astype(np.float64)
           for n, a in zip("uvwe", (u32, v32, w32, e32))}
    for step in range(5):
        ledger = launch_sase_step(
            dev[0], dev[1], dev[2], dev[3], dev[4], dx=DX, dy=DY, dz=DZ,
            delta=DELTA, dt=STEP_DT, n2=n2_dev, ust=ust_dev,
            wspd_sfc=wspd_dev)
        fields, led_ref = sase_split_step(
            ref["u"], ref["v"], ref["w"], th64, ref["e"], dx=DX, dy=DY,
            dz=DZ, delta=DELTA, dt=STEP_DT, n2=n264, ust=ust64,
            wspd_sfc=wspd64)
        ref = {n: fields[n] for n in "uvwe"}
        rels = {n: _max_rel(cp.asnumpy(d), ref[n])[0]
                for n, d in zip("uvwe", (dev[0], dev[1], dev[2], dev[4]))}
        worst = max(rels.values())
        scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                    abs(ledger["dHeat"]), 1e-30)
        closure = abs(ledger["residual"]) / scale
        print(f"step {step + 1}: worst={worst:.3e} closure={closure:.3e} "
              f"dKE_sfc={ledger['dKE_sfc']:+.6e} "
              f"(ref {led_ref['dKE_sfc']:+.6e}) "
              f"conv_resid={ledger['sfc_conv_resid']:+.6e}")
        assert worst <= 5e-5, (step, rels)
        assert closure <= 1e-5, (step, closure)
        assert ledger["dKE_sfc"] < 0.0
        np.testing.assert_allclose(ledger["dKE_sfc"],
                                   led_ref["dKE_sfc"], rtol=1e-3)
        np.testing.assert_allclose(ledger["dE_sfc_src"],
                                   led_ref["dE_sfc_src"], rtol=1e-9)
    # (iv) ust=None: surface channels exactly zero (the pre-S3-6j step)
    led0 = launch_sase_step(
        dev[0], dev[1], dev[2], dev[3], dev[4], dx=DX, dy=DY, dz=DZ,
        delta=DELTA, dt=STEP_DT, n2=n2_dev)
    for key in ("dKE_sfc", "dE_sfc_src", "sfc_conv_resid"):
        assert led0[key] == 0.0, key
    # (v) validation: wrong shape / dtype rejected
    with pytest.raises(ValueError, match="ust"):
        launch_sase_step(dev[0], dev[1], dev[2], dev[3], dev[4], dx=DX,
                         dy=DY, dz=DZ, delta=DELTA, dt=STEP_DT,
                         ust=cp.zeros((3, 3), dtype=cp.float32))
    with pytest.raises(ValueError, match="ust"):
        launch_sase_step(dev[0], dev[1], dev[2], dev[3], dev[4], dx=DX,
                         dy=DY, dz=DZ, delta=DELTA, dt=STEP_DT,
                         ust=cp.zeros(STEP_SHAPE[1:], dtype=cp.float64))
    # S3-9c: wspd_sfc validation -- requires ust, and shape/dtype gate
    with pytest.raises(ValueError, match="wspd_sfc requires ust"):
        launch_sase_step(dev[0], dev[1], dev[2], dev[3], dev[4], dx=DX,
                         dy=DY, dz=DZ, delta=DELTA, dt=STEP_DT,
                         wspd_sfc=wspd_dev)
    with pytest.raises(ValueError, match="wspd_sfc"):
        launch_sase_step(dev[0], dev[1], dev[2], dev[3], dev[4], dx=DX,
                         dy=DY, dz=DZ, delta=DELTA, dt=STEP_DT,
                         ust=ust_dev,
                         wspd_sfc=cp.zeros((3, 3), dtype=cp.float32))


@requires_gpu
def test_implicit_vertical_diffusion_device_parity():
    """Gate (e): Thomas solver unit parity <= 2e-6 (scale-relative) vs
    the FP64 authority ``implicit_vertical_diffusion`` at a diffusion
    number far beyond any explicit bound, on all three thickness modes
    (uniform dz, stretched shared (nz,) column, per-column terrain
    field) and both channel factors (kfac 1 and the e-transport 2).
    Also pins the discrete conservation sum(thick*phi) (FP32 storage
    noise only) and the floor fold."""
    import cupy as cp
    from woof.core.sase import launch_implicit_vertical_diffusion
    from woof.verify.sase_ref import (E_MIN, _face_average, box_filter,
                                       implicit_vertical_diffusion)
    rng = np.random.default_rng(SEED + 9)
    phi32 = box_filter(rng.standard_normal(SHAPE), 4).astype(np.float32)
    kv32 = (5.0 + 2.0 * box_filter(rng.standard_normal(SHAPE), 4)
            ).astype(np.float32)
    assert kv32.min() > 0.0
    t_shared = _stretched_dz().astype(np.float32)
    t_percol32 = _per_column_dz32()
    dt = 400.0                                 # diffusion number >> 1
    cases = (
        ("uniform kfac=1", {"dz": DZ}, {"dz": float(DZ)}, 1.0),
        ("shared kfac=2", {"dz_col": t_shared.astype(np.float64)},
         {"dz_col": t_shared.astype(np.float64)}, 2.0),
        ("per-column kfac=1", {"dz_col": cp.asarray(t_percol32)},
         {"dz_col": t_percol32.astype(np.float64)}, 1.0),
    )
    for label, dev_kw, ref_kw, kfac in cases:
        phi_dev = cp.asarray(phi32)
        got = launch_implicit_vertical_diffusion(
            phi_dev, cp.asarray(kv32), dt=dt, kfac=kfac, **dev_kw)
        assert got is phi_dev                  # in place
        kf = kfac * _face_average(kv32.astype(np.float64))
        ref = implicit_vertical_diffusion(phi32.astype(np.float64), kf,
                                          dt, **ref_kw)
        rel, err = _max_rel(cp.asnumpy(phi_dev), ref)
        print(f"thomas parity [{label}]: max_rel={rel:.3e} "
              f"max_abs={err:.3e}")
        assert rel <= 2e-6, (f"thomas [{label}]: max_rel={rel:.3e} "
                             f"max_abs={err:.3e} (gate 2e-6)")
    # Conservation on the stretched shared column: sum(thick*phi) exact
    # up to FP32 storage noise (the flux-form telescoping pin).
    t64 = t_shared.astype(np.float64)
    phi_dev = cp.asarray(phi32)
    before = float(np.sum(t64[:, None, None] * phi32.astype(np.float64)))
    launch_implicit_vertical_diffusion(phi_dev, cp.asarray(kv32), dt=dt,
                                       dz_col=t64)
    after = float(np.sum(t64[:, None, None]
                         * cp.asnumpy(phi_dev).astype(np.float64)))
    scale = float(np.sum(t64[:, None, None]
                         * np.abs(phi32.astype(np.float64))))
    cons = abs(after - before) / scale
    print(f"thomas conservation: |d(sum thick*phi)|/scale = {cons:.3e}")
    assert cons <= 1e-6
    # Floor fold (the e channel's E_MIN re-floor).
    low = cp.full(SHAPE, 0.5 * E_MIN, dtype=cp.float32)
    launch_implicit_vertical_diffusion(low, cp.asarray(kv32), dt=dt,
                                       dz=DZ, floor=E_MIN)
    assert float(low.min()) >= np.float32(E_MIN)


@requires_gpu
def test_sase_split_step_fp32_ledger_closure_characterized():
    """Gate (c) -- THE spec 4.2 artifact, superseding the v0
    characterized bound: FP32 split-ledger closure over 20 seeded steps
    of the fused device split step on the horizontally periodic
    UNIFORM-dz clamped box (the restated theorem's domain -- no
    periodic vertical exists anymore).  Channels are the theorem's:
    dKE_expl + dKE_impl + dE + dHeat = 0; |residual| /
    max(|dKE|, |dE|, |dHeat|) must stay under the 1e-5 spec gate every
    step; the characterized bound (the max of the printed series,
    measured on the RTX 5090) is recorded in
    .superpowers/sdd/task-s3-6e-report.md (1.305e-6 with the S3-6e
    governed/tapered formulation, superseding S3-6d's 1.185e-6 and
    S3-6c's 1.022e-6).  The reductions are FP64 and deterministic, so
    the series is stable run-to-run."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    u32, v32, w32, th32, e32, n232 = _step_fixture32(seed=SEED + 4)
    dev = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
    n2_dev = cp.asarray(n232)
    rels = []
    for step in range(20):
        ledger = launch_sase_step(*dev, dx=DX, dy=DY, dz=DZ, delta=DELTA,
                                  dt=STEP_DT, n2=n2_dev)
        assert ledger["dKE"] == ledger["dKE_expl"] + ledger["dKE_impl"]
        scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                    abs(ledger["dHeat"]), 1e-30)
        rels.append(abs(ledger["residual"]) / scale)
        print(f"step {step + 1:2d}: |residual|/scale = {rels[-1]:.3e}  "
              f"(dKE_expl={ledger['dKE_expl']:+.6e} "
              f"dKE_impl={ledger['dKE_impl']:+.6e} "
              f"dE={ledger['dE']:+.6e} dHeat={ledger['dHeat']:+.6e})")
    worst = max(rels)
    print(f"characterized FP32 split-ledger bound (max over 20 steps): "
          f"{worst:.3e}")
    assert worst <= 1e-5, (
        f"FP32 split-ledger closure: max |residual|/scale {worst:.3e} "
        f"over 20 steps (spec gate 1e-5); series {rels}")


@requires_gpu
def test_sase_split_step_decay_law_device():
    """Gate (d): the RANS decay law through the device split step
    (authority nz=1 fixture) -- EXACT since S3-6d.  Zero flow
    degenerates the solve to (0, 0), so f = 0 puts the blended
    dissipation length at l_B(dz/2); nz = 1 kills every transport/
    production channel (single clamped cell), leaving
    de/dt = -C_E e^1.5/l_B.  The S3-6d analytic substep is the exact
    flow map of that ODE (flow maps compose -- the CPU fixture has the
    telescoping algebra and lands 2.7e-15), so 400 device steps at
    dt=0.25 must land on e0/(1+bt)^2 up to FP32 storage noise alone:
    measured rel 7.4e-7 on the RTX 5090 (the per-step FP32 rounding of
    the stored e is contracted by the decay map, so it never
    accumulates linearly); gate 1e-5, ~13x headroom (was 5e-3 against
    the retired forward-Euler bias of 2.7e-3)."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    from woof.verify.sase_ref import C_E, _blackadar_length
    shape = (1, 8, 8)
    e0, dt, steps, dz, delta = 0.5, 0.25, 400, 200.0, 500.0
    zeros = [cp.zeros(shape, dtype=cp.float32) for _ in range(3)]
    theta = cp.full(shape, 300.0, dtype=cp.float32)
    e = cp.full(shape, e0, dtype=cp.float32)
    for _ in range(steps):
        ledger = launch_sase_step(*zeros, theta, e, dx=500.0, dy=500.0,
                                  dz=dz, delta=delta, dt=dt)
        assert ledger["dKE"] == 0.0            # no momentum channel at all
        assert (ledger["c_nu"], ledger["f"]) == (0.0, 0.0)   # degenerate
    lb = float(_blackadar_length(0.5 * dz))
    b = C_E * np.sqrt(e0) / (2.0 * lb)
    analytic = e0 / (1.0 + b * dt * steps)**2
    measured = float(cp.asnumpy(e).mean())
    rel = abs(measured - analytic) / analytic
    print(f"decay: measured e={measured:.6f} analytic={analytic:.6f} "
          f"rel={rel:.3e}")
    assert rel <= 1e-5, (f"device split decay law: e_mean={measured!r} "
                         f"analytic={analytic!r} rel={rel:.3e} (gate 1e-5)")


@requires_gpu
def test_sase_split_step_model_mode_stretched_dz_smoke():
    """Clamped variable-dz model mode (dz_col; the driver's integration
    path): three split steps on geometric 1.08 thicknesses from 50 m
    stay finite, keep e floored at E_MIN, and fill the heat output.
    The split theorem is uniform-column-exact only, so the ledger is
    diagnostic here -- finiteness asserted, no residual bar (authority
    test_split_step_variable_dz_model_mode_runs semantics)."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    from woof.verify.sase_ref import E_MIN
    u32, v32, w32, th32, e32, n232 = _step_fixture32(seed=SEED + 5)
    dev = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
    heat = cp.empty(STEP_SHAPE, dtype=cp.float32)
    t = _stretched_dz(STEP_SHAPE[0])
    for _ in range(3):
        ledger = launch_sase_step(*dev, dx=DX, dy=DY, dz=DZ, delta=DELTA,
                                  dt=0.05, n2=cp.asarray(n232), dz_col=t,
                                  heat=heat)
        for key in ("dKE", "dE", "dHeat", "residual", "c_nu", "f",
                    "dKE_expl", "dKE_impl"):
            assert np.isfinite(ledger[key]), key
    host = [cp.asnumpy(a) for a in dev] + [cp.asnumpy(heat)]
    for name, arr in zip(("u", "v", "w", "theta", "e", "heat"), host):
        assert np.all(np.isfinite(arr)), name
    assert np.all(host[4] >= E_MIN)            # e floored elementwise


@requires_gpu
def test_sase_split_step_taper_parity_and_top_row_suppression():
    """S3-6e damping-layer taper on device.  (i) Two tapered device
    split steps on the uniform box fixture track the FP64 authority
    (same zdamp) at the 5e-5 trajectory tier and keep the ledger under
    the 1e-5 spec closure gate.  (ii) On the top-shear column (the d02
    nest-top spike mechanism: P_v >= 0 concentrated at the clamped top
    rows) the tapered top-row e stays strictly below the untapered
    run's after 10 steps.  (iii) The per-column dz_col taper path
    (device FP64 cumsum geometry) matches the shared-column path on a
    broadcast column bitwise in kv and to 1e-6 in e."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    from woof.verify.sase_ref import sase_split_step
    zdamp = 1000.0                             # box htop = 12*200 = 2400
    # (i) parity + closure with the taper live.
    u32, v32, w32, th32, e32, n232 = _step_fixture32(seed=SEED + 11)
    dev = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
    n2_dev = cp.asarray(n232)
    th64, n264 = th32.astype(np.float64), n232.astype(np.float64)
    ref = {n: a.astype(np.float64)
           for n, a in zip("uvwe", (u32, v32, w32, e32))}
    for _ in range(2):
        ledger = launch_sase_step(*dev, dx=DX, dy=DY, dz=DZ, delta=DELTA,
                                  dt=STEP_DT, n2=n2_dev, zdamp=zdamp)
        scale = max(abs(ledger["dKE"]), abs(ledger["dE"]),
                    abs(ledger["dHeat"]), 1e-30)
        assert abs(ledger["residual"]) / scale <= 1e-5
        fields, _ = sase_split_step(ref["u"], ref["v"], ref["w"], th64,
                                    ref["e"], dx=DX, dy=DY, dz=DZ,
                                    delta=DELTA, dt=STEP_DT, n2=n264,
                                    zdamp=zdamp)
        ref = {n: fields[n] for n in "uvwe"}
    rels = {n: _max_rel(cp.asnumpy(d), ref[n])[0]
            for n, d in zip("uvwe", (dev[0], dev[1], dev[2], dev[4]))}
    worst = max(rels.values())
    print("taper parity: " + "  ".join(f"{n}={rels[n]:.3e}"
                                       for n in "uvwe"))
    assert worst <= 5e-5, (f"tapered split parity: worst {worst:.3e} "
                           f"(gate 5e-5)")
    # (ii) top-shear suppression.
    nz = STEP_SHAPE[0]
    prof = np.zeros(nz, np.float32)
    prof[-3:] = np.array([2.0, 6.0, 12.0], np.float32)
    u_prof = np.broadcast_to(prof[:, None, None], STEP_SHAPE).copy()
    theta_u = np.full(STEP_SHAPE, 300.0, np.float32)

    def run_column(zd):
        u = cp.asarray(u_prof)
        v = cp.zeros(STEP_SHAPE, dtype=cp.float32)
        w = cp.zeros(STEP_SHAPE, dtype=cp.float32)
        th = cp.asarray(theta_u)
        e = cp.full(STEP_SHAPE, 0.2, dtype=cp.float32)
        for _ in range(10):
            launch_sase_step(u, v, w, th, e, dx=DX, dy=DY, dz=DZ,
                             delta=DELTA, dt=STEP_DT, zdamp=zd)
        return cp.asnumpy(e)

    e_t = run_column(zdamp)
    e_f = run_column(None)
    assert float(e_t[-1].mean()) < float(e_f[-1].mean())
    assert float(e_t[-2].mean()) < float(e_f[-2].mean())
    # (iii) shared-column vs broadcast per-column taper geometry.
    t32 = _stretched_dz(nz).astype(np.float32)
    t3 = np.broadcast_to(t32[:, None, None], STEP_SHAPE).copy()
    dev_a = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
    dev_b = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
    led_a = launch_sase_step(*dev_a, dx=DX, dy=DY, dz=DZ, delta=DELTA,
                             dt=0.05, n2=n2_dev,
                             dz_col=t32.astype(np.float64), zdamp=300.0)
    led_b = launch_sase_step(*dev_b, dx=DX, dy=DY, dz=DZ, delta=DELTA,
                             dt=0.05, n2=n2_dev, dz_col=cp.asarray(t3),
                             zdamp=300.0)
    np.testing.assert_array_equal(cp.asnumpy(led_a["kv"]),
                                  cp.asnumpy(led_b["kv"]))
    rel, err = _max_rel(cp.asnumpy(dev_b[4]),
                        cp.asnumpy(dev_a[4]).astype(np.float64))
    assert rel <= 1e-6, (f"taper shared vs per-column: e max_rel="
                         f"{rel:.3e} max_abs={err:.3e} (gate 1e-6)")


@requires_gpu
def test_scalar_mix_field_mode_parity_and_validation():
    """S3-6e governed scalar channel: the kh_field mode of
    ``launch_scalar_mix`` vs the authority ``scalar_hmix`` with
    kh = kh_fac*km (gate 5e-6, two chained stencil passes on an O(1)
    scalar); the two K_h modes are mutually exclusive."""
    import cupy as cp
    from woof.core.sase import launch_scalar_mix
    from woof.verify.sase_ref import scalar_hmix
    rng = np.random.default_rng(SEED + 13)
    s32 = _band_limited32(rng)
    km32 = (20.0 + 5.0 * _band_limited32(rng)).astype(np.float32)
    kh_fac = 3.0
    got = launch_scalar_mix(cp.asarray(s32), kh_field=cp.asarray(km32),
                            kh_fac=kh_fac, dx=DX, dy=DY)
    ref = scalar_hmix(s32.astype(np.float64),
                      kh_fac * km32.astype(np.float64), DX, DY)
    rel, err = _max_rel(cp.asnumpy(got), ref)
    assert rel <= 5e-6, (f"field-mode scalar mix: max_rel={rel:.3e} "
                         f"max_abs={err:.3e} (gate 5e-6)")
    s_dev = cp.asarray(s32)
    e_dev = cp.asarray(_e_field32())
    with pytest.raises(ValueError, match="exactly one"):
        launch_scalar_mix(s_dev, e_dev, kh_coef=1.0,
                          kh_field=cp.asarray(km32), dx=DX, dy=DY)
    with pytest.raises(ValueError, match="exactly one"):
        launch_scalar_mix(s_dev, dx=DX, dy=DY)
    with pytest.raises(ValueError, match="coefficient mode"):
        launch_scalar_mix(s_dev, e_dev, dx=DX, dy=DY)


@requires_gpu
def test_sase_split_step_d01_column_parameters_stable():
    """Gate (f): the fixture that used to be impossible.  The S3-6b
    stability-demonstration column at driver-like parameters -- nz = 60
    layers of dz1 = 50 m (the documented d01 regime: dz = 50 m,
    dt = 60 s, e = 1 m2/s2), delta = dx = dy = 12 km, weakly stable
    stratification (l_s stays above l_B so K_v reaches its ~130 m2/s
    l_B-saturation value), sheared flow, e spun up to 1 -- runs 50
    device split steps bounded.  The explicit-regime witness is
    asserted first: the amended K_v at e = 1 puts the face diffusion
    number K*dt/dz^2 above 3 (the CPU RED fixture's regime, >1e21
    amplification over 20 explicit steps), so only the implicit
    vertical channel can hold this configuration."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    from woof.verify.sase_ref import E_MIN, vertical_diffusivity
    nz, ny, nx = 60, 8, 8
    t = np.full(nz, 50.0)                      # dz1 = 50 m, d01 regime
    z = np.cumsum(t) - 0.5 * t
    dx = dy = delta = 12000.0
    dt = 60.0
    theta_prof = 300.0 + 3.0e-4 * z            # weakly stable (~1e-5 N^2)
    n2_prof = 9.81 / theta_prof * 3.0e-4
    # RED-regime witness: the very fluxes the split step will solve
    # implicitly are deep in the explicit-unstable regime.
    kv_prof = vertical_diffusivity(z, np.ones(nz), n2=n2_prof)
    kf = 0.5 * (kv_prof[:-1] + kv_prof[1:])
    d_face = float((kf * dt / 50.0**2).max())
    assert d_face > 3.0, f"fixture left the unstable regime: {d_face}"
    shape = (nz, ny, nx)
    u = cp.asarray(np.broadcast_to(
        (5.0 + 8.0 * z / z[-1])[:, None, None], shape).astype(np.float32)
        .copy())
    v = cp.full(shape, 1.0, dtype=cp.float32)
    w = cp.zeros(shape, dtype=cp.float32)
    theta = cp.asarray(np.broadcast_to(
        theta_prof[:, None, None], shape).astype(np.float32).copy())
    e = cp.full(shape, 1.0, dtype=cp.float32)  # spun up
    n2 = cp.asarray(np.broadcast_to(
        n2_prof[:, None, None], shape).astype(np.float32).copy())
    heat = cp.empty(shape, dtype=cp.float32)
    e_max_series = []
    for step in range(50):
        ledger = launch_sase_step(u, v, w, theta, e, dx=dx, dy=dy,
                                  dz=float(t.mean()), delta=delta, dt=dt,
                                  n2=n2, dz_col=t, heat=heat)
        for key in ("dKE", "dE", "dHeat", "residual", "dKE_expl",
                    "dKE_impl"):
            assert np.isfinite(ledger[key]), (key, step)
        e_max_series.append(float(e.max()))
    for name, arr in (("u", u), ("v", v), ("w", w), ("e", e),
                      ("heat", heat)):
        host = cp.asnumpy(arr)
        assert np.all(np.isfinite(host)), name
    e_host = cp.asnumpy(e)
    assert np.all(e_host >= np.float32(E_MIN))
    assert max(e_max_series) <= 50.0, (
        f"e unbounded: max series {max(e_max_series)}")
    assert float(cp.abs(u).max()) <= 20.0      # initial max 13; bounded
    assert float(cp.abs(w).max()) <= 5.0
    print(f"d01 column: 50 steps at dt=60, dz1=50 m, d_face={d_face:.2f}; "
          f"final e_max={e_max_series[-1]:.4f}, "
          f"u_max={float(cp.abs(u).max()):.3f} -- STABLE")


@requires_gpu
def test_sase_driver_d01_column_parameters_stable_50_steps():
    """Gate (f), driver level: 50 full model steps at dt = 60 s on an
    ADMITTED d01-like configuration (dx = dy = 12 km, tanh-stretched
    coordinate with dz1 ~ 50 m, moist Morrison, sfclay + Noah, zero-flux
    radiation stub through the sanctioned injection seam, bldt = 0 so
    SASE runs every step) with e_sgs spun up to 1 m2/s2 -- the driver
    configuration the v0 explicit path could not survive one step of
    (vertical diffusion number O(10^2), physics._run_sase history
    note).  Every prognostic and e_sgs must stay finite and bounded
    over all 50 steps."""
    import cupy as cp

    from woof.config import RunConfig, validate_run_config
    from woof.core import dycore
    from woof.core.physics import initialize_physics
    from woof.verify.sase_ref import E_MIN

    cfg = RunConfig(nx=12, ny=10, nz=40, dx=12000.0, dy=12000.0,
                    ztop=16000.0, dt=60.0, run_seconds=3000.0,
                    time_step_sound=4, moist=True, mp_physics=10,
                    ra_physics=4, sf_sfclay_physics=1,
                    sf_surface_physics=2, km_opt=0,
                    bl_pbl_physics=_SASE_SELECTOR)
    assert validate_run_config(cfg) is cfg     # the admitted combination
    state = _moist_driver_state(cfg, stretch=1.6)
    z_half = state.height_half()
    dz1 = 2.0 * float(z_half[0])
    assert 40.0 <= dz1 <= 65.0, f"fixture dz1 drifted: {dz1:.1f} m"
    # Sheared flow + spun-up e: the d01 instability regime of the
    # retired formulation.
    shear = (5.0 + 8.0 * z_half / cfg.ztop).astype(np.float32)
    state.u[...] = cp.asarray(
        np.broadcast_to(shear[:, None, None],
                        (cfg.nz, cfg.ny, cfg.nx + 1)))
    state.v[...] = cp.float32(1.0)
    state.e_sgs[...] = cp.float32(1.0)         # e spun up

    initialize_physics(state, cfg, landmask=1.0, tsk=302.0,
                       swdown=400.0, glw=320.0, radiation=_zero_radiation)
    for step in range(50):
        dycore.step(state, cfg)                # physics runs inside
        e = state.e_sgs
        assert bool(cp.isfinite(e).all()), f"e_sgs NaN at step {step + 1}"
        e_max = float(e.max())
        assert float(e.min()) >= np.float32(E_MIN) and e_max <= 50.0, (
            f"e_sgs unbounded at step {step + 1}: max {e_max}")
    for name in ("u", "v", "w", "thp", "qv", "e_sgs"):
        arr = cp.asnumpy(getattr(state, name))
        assert np.all(np.isfinite(arr)), name
    assert float(cp.abs(state.u).max()) <= 40.0
    assert float(cp.abs(state.w).max()) <= 20.0
    print(f"driver d01: 50 steps at dt=60, dz1={dz1:.1f} m; "
          f"final e_max={float(state.e_sgs.max()):.4f}, "
          f"u_max={float(cp.abs(state.u).max()):.2f} -- STABLE")


@requires_gpu
def test_sase_step_launcher_validation():
    import cupy as cp
    from woof.core.sase import (launch_implicit_vertical_diffusion,
                                 launch_sase_step, launch_scalar_mix)
    u = cp.zeros(SHAPE, dtype=cp.float32)
    bad = cp.zeros(SHAPE, dtype=cp.float64)
    with pytest.raises(ValueError, match="float32"):
        launch_sase_step(u, u, u, bad, u, dx=DX, dy=DY, dz=DZ,
                         delta=DELTA, dt=0.5)              # theta dtype
    with pytest.raises(ValueError, match="float32"):
        launch_sase_step(u, u, u, u, u, dx=DX, dy=DY, dz=DZ,
                         delta=DELTA, dt=0.5, n2=bad)      # n2 dtype
    small = cp.zeros((2, 2, 2), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_sase_step(u, u, u, u, u, dx=DX, dy=DY, dz=DZ,
                         delta=DELTA, dt=0.5, heat=small)  # heat shape
    # The in-thread Thomas sweeps bound the column depth at SASE_KMAX.
    deep = cp.zeros((256, 4, 4), dtype=cp.float32)
    with pytest.raises(ValueError, match="SASE_KMAX"):
        launch_implicit_vertical_diffusion(deep, deep, dt=1.0, dz=50.0)
    with pytest.raises(ValueError, match="SASE_KMAX"):
        launch_sase_step(deep, deep, deep, deep, deep, dx=DX, dy=DY,
                         dz=DZ, delta=DELTA, dt=0.5)
    with pytest.raises(ValueError, match="dz"):
        launch_implicit_vertical_diffusion(u, u, dt=1.0)   # neither
    # The horizontal-only scalar channel takes exactly two flux buffers.
    with pytest.raises(ValueError, match="flux"):
        launch_scalar_mix(u, u, kh_coef=1.0, dx=DX, dy=DY,
                          flux=[u, u, u])


# ---------------------------------------------------------------------------
# Stage-3 Task 6: per-column (nz, ny, nx) dz_col, the K_h scalar-mix /
# N^2 launchers, and the driver-level coupling smoke.
# ---------------------------------------------------------------------------


def _per_column_dz32(shape=SHAPE):
    """Terrain-like per-column thicknesses: the geometric 1.08 column
    modulated +-30 percent by a smooth horizontal surface factor."""
    nz, ny, nx = shape
    t = _stretched_dz(nz)
    yy, xx = np.meshgrid(np.arange(ny, dtype=np.float64),
                         np.arange(nx, dtype=np.float64), indexing="ij")
    surface = (1.0 + 0.3 * np.sin(2.0 * np.pi * yy / ny)
               * np.cos(2.0 * np.pi * xx / nx))
    return (t[:, None, None] * surface[None]).astype(np.float32)


@requires_gpu
def test_strain_parity_per_column_dz_col():
    """3-D dz_col parity gate (Task 6 carry-forward 1): per-column
    terrain-like thicknesses through the device FP64 coefficient build
    (sase_ddz_coefficients, z_mode 3, folded edge rows) vs the FP64
    authority on the SAME FP32 thicknesses promoted to FP64 (dz=NaN
    proves the uniform argument is never consulted).  Gate 2e-6
    scale-relative, mirroring the shared-column stretched gate."""
    import cupy as cp
    from woof.core.sase import launch_strain
    from woof.verify.sase_ref import strain
    u32, v32, w32 = _velocities32()
    t32 = _per_column_dz32()
    got = launch_strain(*(cp.asarray(a) for a in (u32, v32, w32)),
                        dx=DX, dy=DY, dz_col=cp.asarray(t32))
    ref = strain(u32, v32, w32, DX, DY, float("nan"),
                 dz_col=t32.astype(np.float64))
    names = ("xx", "yy", "zz", "xy", "xz", "yz")
    for name, g, r in zip(names, got, ref):
        rel, err = _max_rel(cp.asnumpy(g), r)
        assert rel <= 2e-6, (f"strain[{name}] per-column dz_col: "
                             f"max_rel={rel:.3e} max_abs={err:.3e} "
                             f"(gate 2e-6)")


@requires_gpu
def test_scalar_mix_and_n2_parity_per_column():
    """launch_scalar_mix (S3-6c: HORIZONTAL-only, the explicit half of
    the split scalar channel) vs the horizontal part of the authority
    machinery, and launch_n2 vs ``brunt_vaisala_n2`` on the per-column
    grid, at TWO characterized scales:

    * an O(1) band-limited scalar (no offset): pure kernel-arithmetic
      parity, gate 5e-6 (two chained stencil passes);
    * a theta-like 300-K-offset field: the centered horizontal
      differences cancel the offset only up to FP32 rounding at
      magnitude ~300, gated at 5e-5 (same offset-cancellation bound the
      vertical operators carry; characterized at Task 6).

    The horizontal reference is composed from the frozen authority
    operators (``_ddx``/``_ddy`` and the ``scalar_mix`` K_h
    construction) because the full-3D ``scalar_mix`` is no longer a
    device gate: its explicit vertical leg is superseded by the
    implicit K_v/Pr_t(f) channel (gated by the Thomas parity test).
    """
    import cupy as cp
    from woof.core.sase import launch_n2, launch_scalar_mix
    from woof.verify.sase_ref import (E_MIN, _ddx, _ddy, box_filter,
                                       brunt_vaisala_n2)
    rng = np.random.default_rng(SEED + 7)
    band = box_filter(rng.standard_normal(SHAPE), 4)
    s32_o1 = band.astype(np.float32)           # O(1), offset-free
    s32_th = (300.0 + 2.0 * band).astype(np.float32)
    e32 = _e_field32()
    t32 = _per_column_dz32()
    t_dev = cp.asarray(t32)
    t64 = t32.astype(np.float64)
    kh_coef = 37.5
    flux = [cp.empty(SHAPE, dtype=cp.float32) for _ in range(2)]
    out = cp.empty(SHAPE, dtype=cp.float32)
    kh = kh_coef * np.sqrt(np.maximum(e32.astype(np.float64), E_MIN))
    for label, s32, gate in (("O(1)", s32_o1, 5e-6),
                             ("theta-offset", s32_th, 5e-5)):
        got = launch_scalar_mix(cp.asarray(s32), cp.asarray(e32),
                                kh_coef=kh_coef, dx=DX, dy=DY,
                                out=out, flux=flux)
        assert got is out                      # buffer routing honored
        s64 = s32.astype(np.float64)
        ref = (_ddx(kh * _ddx(s64, DX), DX) + _ddy(kh * _ddy(s64, DY), DY))
        rel, err = _max_rel(cp.asnumpy(got), ref)
        assert rel <= gate, (f"horizontal scalar_mix [{label}]: "
                             f"max_rel={rel:.3e} max_abs={err:.3e} "
                             f"(gate {gate:.0e})")
    n2_got = launch_n2(cp.asarray(s32_th), dz_col=t_dev)
    n2_ref = brunt_vaisala_n2(s32_th, float("nan"), dz_col=t64)
    rel2, err2 = _max_rel(cp.asnumpy(n2_got), n2_ref)
    assert rel2 <= 5e-5, (f"n2 per-column (theta-offset field): "
                          f"max_rel={rel2:.3e} max_abs={err2:.3e} "
                          f"(gate 5e-5, offset-cancellation bound)")


@requires_gpu
def test_sase_step_per_column_broadcast_matches_shared_column():
    """A (nz, ny, nx) dz_col broadcasting one shared column reproduces
    the (nz,) split step to coefficient-cast roundoff: the host FP64
    cumsum grouping and the device FP64 direct-h grouping agree to
    <= 1 ULP FP64 before the shared FP32 cast (interior rows cast
    identically; only the folded edge rows change divide-vs-multiply
    grouping).  The internal solve sees identical inputs, so (c_nu, f)
    must match EXACTLY; the vertical channel and Thomas sweeps read the
    same FP32 thicknesses through both t_modes, so the kv field must
    match BITWISE."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    u32, v32, w32, th32, e32, n232 = _step_fixture32(seed=SEED + 6)
    t32 = _stretched_dz(STEP_SHAPE[0]).astype(np.float32)
    t3 = np.broadcast_to(t32[:, None, None], STEP_SHAPE).copy()
    dev_a = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
    dev_b = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
    ledger_a = launch_sase_step(*dev_a, dx=DX, dy=DY, dz=DZ, delta=DELTA,
                                dt=0.05, n2=cp.asarray(n232),
                                dz_col=t32.astype(np.float64))
    ledger_b = launch_sase_step(*dev_b, dx=DX, dy=DY, dz=DZ, delta=DELTA,
                                dt=0.05, n2=cp.asarray(n232),
                                dz_col=cp.asarray(t3))
    assert (ledger_a["c_nu"], ledger_a["f"]) == (ledger_b["c_nu"],
                                                 ledger_b["f"])
    np.testing.assert_array_equal(cp.asnumpy(ledger_a["kv"]),
                                  cp.asnumpy(ledger_b["kv"]))
    for name, a, b in zip("uvwte", dev_a, dev_b):
        rel, err = _max_rel(cp.asnumpy(b), cp.asnumpy(a).astype(np.float64))
        assert rel <= 1e-6, (f"{name}: shared vs per-column broadcast "
                             f"max_rel={rel:.3e} max_abs={err:.3e} "
                             f"(gate 1e-6)")


@requires_gpu
def test_sase_driver_step_smoke():
    """Driver-level coupling smoke on a small admitted-config moist
    Morrison state: one PhysicsDriver.compute (SASE runs, YSU does not;
    tendencies finite; e_sgs floored/bounded) followed by one full
    dycore.step on the km_opt=0 SASE path (prognostics + e_sgs stay
    finite).  Radiation is a zero-flux stub through the sanctioned
    callable-injection seam so the gate isolates the SASE coupling."""
    import cupy as cp

    from woof.config import RunConfig, validate_run_config
    from woof.core import dycore
    from woof.core.physics import initialize_physics
    from woof.verify.sase_ref import E_MIN

    cfg = RunConfig(nx=12, ny=10, nz=16, dx=2000.0, dy=2000.0,
                    ztop=8000.0, dt=5.0, run_seconds=5.0,
                    time_step_sound=4, moist=True, mp_physics=10,
                    ra_physics=4, sf_sfclay_physics=1,
                    sf_surface_physics=2, km_opt=0,
                    bl_pbl_physics=_SASE_SELECTOR)
    assert validate_run_config(cfg) is cfg     # the admitted combination
    state = _moist_driver_state(cfg)
    # Sheared flow so the strain/production channels are live.
    z_half = state.height_half()
    shear = (5.0 + 8.0 * z_half / cfg.ztop).astype(np.float32)
    state.u[...] = cp.asarray(
        np.broadcast_to(shear[:, None, None],
                        (cfg.nz, cfg.ny, cfg.nx + 1)))
    state.v[...] = cp.float32(1.0)

    driver = initialize_physics(state, cfg, landmask=1.0, tsk=302.0,
                                swdown=400.0, glw=320.0,
                                radiation=_zero_radiation)
    tend = driver.compute(state, cfg)

    assert driver.call_counts["sase"] == 1
    # "ysu" is this head's shared PBL-SLOT counter; the
    # invariant that matters is that the SASE seam ran.
    assert driver.call_counts["ysu"] == driver.call_counts["sase"]
    assert driver.call_counts["sfclay"] == 1 and driver.call_counts["noah"] == 1
    ledger = driver.last_sase_ledger
    for key in ("dKE", "dE", "dHeat", "residual", "dKE_expl", "dKE_impl"):
        assert np.isfinite(ledger[key]), key
    assert 0.0 <= ledger["c_nu"] <= 0.5 and 0.0 <= ledger["f"] <= 1.0
    # S3-6g: the retained ledger carries the blended Prandtl number
    # the step used, inside [PR_LES, PR_RANS] and equal to the blend
    # at the retained f.
    from woof.verify.sase_ref import PR_LES, PR_RANS, prandtl_blend
    assert ledger["pr_t"] == prandtl_blend(ledger["f"])
    assert PR_LES <= ledger["pr_t"] <= PR_RANS
    # S3-6c: the driver pops the split step's kv field for the scalar
    # channel; the retained ledger must stay scalar-only.
    assert "kv" not in ledger

    e = cp.asnumpy(state.e_sgs)
    assert np.all(np.isfinite(e))
    assert np.all(e >= np.float32(E_MIN))      # floor holds elementwise
    assert float(e.max()) <= 50.0              # smoke-gate bound

    for name in ("ru", "rv", "rtheta", "rqv", "rqc", "rqi"):
        arr = getattr(tend, name)
        assert arr is not None and bool(cp.isfinite(arr).all()), name
    rw = tend.rw
    assert rw.shape == (cfg.nz + 1, cfg.ny, cfg.nx)
    assert bool(cp.isfinite(rw).all())
    assert not cp.asnumpy(rw[0]).any() and not cp.asnumpy(rw[-1]).any()

    # One full model step through the km_opt=0 SASE dycore path.
    dycore.step(state, cfg)
    for name in ("u", "v", "w", "thp", "qv", "e_sgs"):
        arr = cp.asnumpy(getattr(state, name))
        assert np.all(np.isfinite(arr)), name
    e2 = cp.asnumpy(state.e_sgs)
    assert np.all(e2 >= np.float32(E_MIN)) and float(e2.max()) <= 50.0


@requires_gpu
def test_dynamic_solve_boundary_exclusion_is_exact():
    """Registered specified-boundary adjudication, reduction half: with
    exclude_boundary_width=5, NaN-poisoning the entire outer ring of
    every solve input leaves (c_nu, f) BITWISE unchanged -- the widest
    basis/lift stencil reach from row 0 is 4 cells, so every
    poison-touched contribution lies inside the excluded zone and the
    deterministic FP64 reduction sums exactly the same interior terms.
    Without exclusion the same poison corrupts the moments (sanity that
    the poison is potent)."""
    import cupy as cp
    from woof.core.sase import launch_dynamic_solve
    u32, v32, w32, e32 = _solve_fixture32(3, (8, 24, 24), 0.05, 0.1, True)
    dev = [cp.asarray(a) for a in (u32, v32, w32, e32)]
    clean = launch_dynamic_solve(*dev, dx=500.0, dy=500.0, dz=200.0,
                                 delta=500.0, exclude_boundary_width=5)
    poisoned = [a.copy() for a in dev]
    for a in poisoned:
        a[:, 0, :] = cp.float32(np.nan)
        a[:, -1, :] = cp.float32(np.nan)
        a[:, :, 0] = cp.float32(np.nan)
        a[:, :, -1] = cp.float32(np.nan)
    masked = launch_dynamic_solve(*poisoned, dx=500.0, dy=500.0, dz=200.0,
                                  delta=500.0, exclude_boundary_width=5)
    assert masked == clean, (
        f"boundary poison leaked into the excluded solve: {masked!r} != "
        f"{clean!r}")
    try:
        unmasked = launch_dynamic_solve(*poisoned, dx=500.0, dy=500.0,
                                        dz=200.0, delta=500.0)
    except np.linalg.LinAlgError:
        unmasked = None            # NaN moments blow up the cond gate
    assert unmasked != clean                   # the poison is potent
    with pytest.raises(ValueError, match="exclude_boundary_width"):
        launch_dynamic_solve(*dev, dx=500.0, dy=500.0, dz=200.0,
                             delta=500.0, exclude_boundary_width=12)


# ---------------------------------------------------------------------------
# S3-6f: partition-bound device kernels (bulk-Richardson z_i column
# sweep + N^2-screened w-sensor reduction) and the step-level bounds.
# ---------------------------------------------------------------------------


@requires_gpu
def test_bulk_richardson_zi_device_parity():
    """Per-column z_i vs the authority on shared FP32 inputs: uniform
    dz, shared stretched column, and the per-column 3-D thickness
    field.  Both sides run FP64 in-column arithmetic, so the gate is
    tight (the residual is the z-center construction ordering)."""
    import cupy as cp
    from woof.core.sase import launch_bulk_richardson_zi
    from woof.verify.sase_ref import bulk_richardson_zi

    rng = np.random.default_rng(SEED)
    shape = (16, 24, 24)
    u32 = 5.0 * _band_limited32(rng, shape)
    v32 = 5.0 * _band_limited32(rng, shape)
    th32 = (300.0 + np.cumsum(
        rng.uniform(0.0, 0.8, shape), axis=0)).astype(np.float32)
    dev = tuple(cp.asarray(a) for a in (u32, v32, th32))

    nz = shape[0]
    z_uni = ((np.arange(nz) + 0.5) * DZ)[:, None, None]
    ref = bulk_richardson_zi(u32, v32, th32, z_uni)
    got = cp.asnumpy(launch_bulk_richardson_zi(*dev, dz=DZ))
    rel, _ = _max_rel(got, ref)
    assert rel <= 1e-5, f"uniform zi parity {rel:.3e}"

    t = _stretched_dz(nz)
    z_str = (np.cumsum(t) - 0.5 * t)[:, None, None]
    ref_s = bulk_richardson_zi(u32, v32, th32, z_str)
    got_s = cp.asnumpy(launch_bulk_richardson_zi(*dev, dz_col=t))
    rel_s, _ = _max_rel(got_s, ref_s)
    assert rel_s <= 1e-5, f"stretched zi parity {rel_s:.3e}"

    # Per-column 3-D thickness field (t_mode 3): same shared column
    # broadcast, must agree with the shared-column path bitwise.
    t3 = cp.ascontiguousarray(cp.broadcast_to(
        cp.asarray(t.astype(np.float32))[:, None, None], shape))
    got_3 = cp.asnumpy(launch_bulk_richardson_zi(*dev, dz_col=t3))
    np.testing.assert_array_equal(got_3, got_s)


@requires_gpu
def test_w_sensor_moments_device_parity_and_screen():
    """N^2-screened w moments vs the authority: exact passing count,
    D2 sums at the structure-function gate, interior floored-e mean;
    the silent screen and the boundary-exclusion mask (poison
    invariance, the solve-reduction idiom)."""
    import cupy as cp
    from woof.core.sase import launch_w_sensor_moments
    from woof.verify.sase_ref import (E_MIN, _w_bound_tail,
                                       w_structure_functions)

    rng = np.random.default_rng(SEED)
    shape = (12, 24, 24)
    w32 = _band_limited32(rng, shape)
    e32 = np.maximum(0.05 + 0.1 * _band_limited32(rng, shape),
                     0.0).astype(np.float32)
    n232 = (1.5e-4 * _band_limited32(rng, shape)).astype(np.float32)
    dev_w, dev_e, dev_n2 = (cp.asarray(a) for a in (w32, e32, n232))

    d2_ref, cnt_ref = w_structure_functions(w32, n232)
    assert 0 < cnt_ref < w32.size              # the screen is live
    d2, cnt, e_mean = launch_w_sensor_moments(dev_w, dev_e, dev_n2)
    assert cnt == cnt_ref
    for r in (1, 2, 4):
        rel = abs(d2[r] - d2_ref[r]) / abs(d2_ref[r])
        assert rel <= 5e-6, f"d2w[{r}] parity {rel:.3e}"
    e_mean_ref = float(np.mean(np.maximum(e32.astype(np.float64), E_MIN)))
    assert abs(e_mean - e_mean_ref) / e_mean_ref <= 1e-12
    # The shared tail therefore agrees to the same order.
    ws_ref = _w_bound_tail(d2_ref[2], cnt_ref, w32.size, e_mean_ref)
    ws_dev = _w_bound_tail(d2[2], cnt, w32.size, e_mean)
    assert abs(ws_dev.f_w - ws_ref.f_w) <= 5e-6

    # n2 omitted: every cell passes (authority convention).
    d2_all, cnt_all, _ = launch_w_sensor_moments(dev_w, dev_e)
    d2_all_ref, cnt_all_ref = w_structure_functions(w32, None)
    assert cnt_all == cnt_all_ref == w32.size
    rel = abs(d2_all[2] - d2_all_ref[2]) / abs(d2_all_ref[2])
    assert rel <= 5e-6

    # Silent screen: fully stable domain, zero passing cells.
    stable = cp.full(shape, 2.0e-4, dtype=cp.float32)
    d2_s, cnt_s, _ = launch_w_sensor_moments(dev_w, dev_e, stable)
    assert cnt_s == 0 and all(v == 0.0 for v in d2_s.values())

    # Boundary exclusion: poison the outer ring; the masked moments
    # must equal the clean masked moments (anchor-cell mask, exactly
    # the solve-reduction semantics).
    clean = launch_w_sensor_moments(dev_w, dev_e, dev_n2,
                                    exclude_boundary_width=5)
    pw, pe, pn = (a.copy() for a in (dev_w, dev_e, dev_n2))
    for a in (pw, pe, pn):
        a[:, 0, :] = cp.float32(np.nan)
        a[:, -1, :] = cp.float32(np.nan)
        a[:, :, 0] = cp.float32(np.nan)
        a[:, :, -1] = cp.float32(np.nan)
    masked = launch_w_sensor_moments(pw, pe, pn,
                                     exclude_boundary_width=5)
    # NOTE the poisoned WRAP reads: increments anchored inside the
    # interior can reach poisoned rows only when the anchor sits within
    # r of the excluded ring -- bw = 5 > r_max = 4 keeps every anchored
    # read clean, which is what this asserts.
    assert masked[1] == clean[1]
    assert masked[0] == clean[0] and masked[2] == clean[2]
    with pytest.raises(ValueError, match="exclude_boundary_width"):
        launch_w_sensor_moments(dev_w, dev_e, dev_n2,
                                exclude_boundary_width=12)


@requires_gpu
def test_sase_split_step_partition_bounds_device():
    """Step-level S3-6f parity: the device step's partition-bound
    ledger scalars (f, f_solved, f_cap, f_w, zi, w_coverage) against
    the authority step on the same FP32 fixture, for (i) a
    cap-engaging mesoscale fixture with a SILENT screen and (ii) a
    gray-zone fixture where the w-bound binds."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    from woof.verify import sase_ref

    nz, ny, nx = 8, 16, 16
    shape = (nz, ny, nx)
    dz = 100.0

    def dev_fields(*fields):
        return [cp.ascontiguousarray(cp.asarray(a.astype(np.float32)))
                for a in fields]

    # (i) mesoscale: balanced barotropic flow, 10 K inversion at k = 2,
    # uniform n2 above the screen -- cap engages, sensor abstains.
    y = np.arange(ny)[None, :, None] * np.ones(shape)
    x = np.arange(nx)[None, None, :] * np.ones(shape)
    u = 10.0 * np.sin(2.0 * np.pi * y / ny)
    v = 10.0 * np.cos(2.0 * np.pi * x / nx)
    w = 0.5 * np.sin(2.0 * np.pi * x / nx)
    theta = np.full(shape, 300.0)
    theta[2:] = 310.0
    n2 = np.full(shape, 1.5e-4)
    e = np.full(shape, 0.05)
    delta = 2500.0
    u32 = u.astype(np.float32)
    _, ref = sase_ref.sase_split_step(
        u32, v.astype(np.float32), w.astype(np.float32),
        theta.astype(np.float32), e.astype(np.float32),
        dx=delta, dy=delta, dz=dz, delta=delta, dt=2.0,
        n2=n2.astype(np.float32))
    dev = dev_fields(u, v, w, theta, e)
    led = launch_sase_step(*dev, dx=delta, dy=delta, dz=dz, delta=delta,
                           dt=2.0, n2=dev_fields(n2)[0])
    assert led["w_coverage"] == 0.0 == ref["w_coverage"]
    assert led["f_w"] == 1.0 == ref["f_w"]
    assert abs(led["zi"] - ref["zi"]) / ref["zi"] <= 1e-5
    assert led["f_cap"] < 1e-6 and ref["f_cap"] < 1e-6
    assert led["f"] <= led["f_cap"]
    # S3-6g: the exported blended Prandtl number rides the same f
    # (cap-engaged: pr_t -> PR_RANS as f -> 0).
    assert led["pr_t"] == sase_ref.prandtl_blend(led["f"])
    assert abs(led["pr_t"] - sase_ref.PR_RANS) < 1e-6

    # (ii) gray zone: w-dominated variance, partial screen, f = f_w.
    rng = np.random.default_rng(SEED)

    def band():
        return sase_ref.box_filter(rng.standard_normal(shape), 4)

    u2, v2 = 3.0 + band(), band()
    w2 = 1.5 * band()
    w2[6:] *= 0.05
    th2 = np.full(shape, 300.0)
    th2[6:] = 308.0
    n22 = np.zeros(shape)
    n22[6:] = 2.0e-3
    e2 = 2.0 + 0.4 * band()
    delta2 = 600.0
    _, ref2 = sase_ref.sase_split_step(
        u2.astype(np.float32), v2.astype(np.float32),
        w2.astype(np.float32), th2.astype(np.float32),
        e2.astype(np.float32), dx=delta2, dy=delta2, dz=125.0,
        delta=delta2, dt=5.0, n2=n22.astype(np.float32))
    dev2 = dev_fields(u2, v2, w2, th2, e2)
    led2 = launch_sase_step(*dev2, dx=delta2, dy=delta2, dz=125.0,
                            delta=delta2, dt=5.0,
                            n2=dev_fields(n22)[0])
    assert led2["f_cap"] == 1.0 == ref2["f_cap"]
    assert led2["w_coverage"] == ref2["w_coverage"]
    assert abs(led2["f_w"] - ref2["f_w"]) <= 5e-6
    assert led2["f"] == led2["f_w"]            # the w-bound binds
    assert abs(led2["f"] - ref2["f"]) <= 5e-6
    # S3-6g pr_t parity: the blend is Lipschitz with constant
    # |PR_LES - PR_RANS| ~ 0.517, so the f agreement bounds it.
    assert led2["pr_t"] == sase_ref.prandtl_blend(led2["f"])
    assert abs(led2["pr_t"] - ref2["pr_t"]) <= 3e-6


# ---------------------------------------------------------------------------
# S3-6h: BL89 vertical-channel kernel parity
# ---------------------------------------------------------------------------


@requires_gpu
def test_vertical_channel_bl89_parity_and_f_blend():
    """S3-6h/S3-6i unit parity: the amended ``sase_vertical_channel``
    kernel (in-thread FP64 BL89 column sweep, authority
    ``bl89_rans_lengths`` + S3-6i ``stable_limit_coefficient``
    composition -- the decoupled stable-limit coefficient C_r on the
    RANS limb and the two-product K blend) against the FP64 authority
    on a structured mixed-layer-under-inversion column set with
    band-limited perturbations -- the profile class where the BL89
    bounds BIND (the mechanism regime), plus live n2 so the l_s
    branch (and with it the S3-6i coefficient blend) engages.
    Gates: kv and leps scale-relative <= 2e-6 at f = 0 (pure RANS
    limb), f = 0.3 (interior blend), and f = 1 (LES limb, where kv
    must ALSO be bitwise-independent of theta -- the FP-exact
    two-product endpoint); uniform-dz and stretched shared-column
    thickness modes.  The fixture must engage BOTH mechanisms (BL89
    binding asserted below; stable cells assert a strict C_r < C_KV
    deficit)."""
    import cupy as cp
    from woof.core.sase import launch_vertical_channel
    from woof.verify import sase_ref

    rng = np.random.default_rng(20260721)
    nz, ny, nx = 24, 16, 16
    shape = (nz, ny, nx)

    def band():
        return sase_ref.box_filter(rng.standard_normal(shape), 4)

    for label, dz, dz_col in (("BOX", 25.0, None),
                              ("COL", None, _stretched_dz(nz, 20.0))):
        if dz_col is None:
            thick = np.full(nz, dz)
        else:
            thick = np.asarray(dz_col, dtype=np.float64)
        z1 = np.cumsum(thick) - 0.5 * thick
        th1 = np.where(z1 <= 0.6 * z1[-1], 290.0,
                       290.0 + 0.02 * (z1 - 0.6 * z1[-1]))
        theta64 = (np.broadcast_to(th1[:, None, None], shape)
                   + 0.05 * band()).astype(np.float32).astype(np.float64)
        e64 = np.maximum(0.05 + 0.1 * band(), 0.0) \
            .astype(np.float32).astype(np.float64)
        n264 = sase_ref.brunt_vaisala_n2(
            theta64, dz, dz_col=None if dz_col is None else thick) \
            .astype(np.float32).astype(np.float64)
        dev = {
            "e": cp.asarray(e64.astype(np.float32)),
            "theta": cp.asarray(theta64.astype(np.float32)),
            "n2": cp.asarray(n264.astype(np.float32)),
        }
        z = z1[:, None, None]
        tcol = thick[:, None, None]
        for f in (0.0, 0.3, 1.0):
            kv_d, leps_d = launch_vertical_channel(
                dev["e"], dev["theta"], f=f, n2=dev["n2"],
                dz=dz, dz_col=dz_col)
            l_les = sase_ref.vertical_mixing_length(z, e64, n264)
            l_mix_r, l_eps_r = sase_ref.bl89_rans_lengths(
                theta64, e64, z, tcol, n264)
            e_fl = np.maximum(e64, sase_ref.E_MIN)
            root_e = np.sqrt(e_fl)
            c_r = sase_ref.stable_limit_coefficient(l_mix_r, e_fl, n264)
            kv_ref = (f * (sase_ref.C_KV * l_les * root_e)
                      + (1.0 - f) * (c_r * l_mix_r * root_e))
            rel_kv, _ = _max_rel(cp.asnumpy(kv_d), kv_ref)
            rel_le, _ = _max_rel(cp.asnumpy(leps_d), l_eps_r)
            print(f"{label} f={f}: kv rel={rel_kv:.3e} "
                  f"leps rel={rel_le:.3e}")
            assert rel_kv <= 2e-6, (label, f, rel_kv)
            assert rel_le <= 2e-6, (label, f, rel_le)
        # the mechanisms must actually engage on this fixture
        l_mix_r, l_eps_r = sase_ref.bl89_rans_lengths(
            theta64, e64, z, tcol, n264)
        l_les = sase_ref.vertical_mixing_length(z, e64, n264)
        assert np.any(l_mix_r < l_les), label
        c_chk = sase_ref.stable_limit_coefficient(
            l_mix_r, np.maximum(e64, sase_ref.E_MIN), n264)
        assert np.any(c_chk < sase_ref.C_KV), label   # S3-6i engaged
        # f = 1 LES limb: kv bitwise-independent of the theta profile
        kv_a, _ = launch_vertical_channel(
            dev["e"], dev["theta"], f=1.0, n2=dev["n2"],
            dz=dz, dz_col=dz_col)
        flipped = cp.asarray(theta64[::-1].copy().astype(np.float32))
        kv_b, _ = launch_vertical_channel(
            dev["e"], flipped, f=1.0, n2=dev["n2"],
            dz=dz, dz_col=dz_col)
        assert bool((kv_a == kv_b).all()), label


# ---------------------------------------------------------------------------
# S3-9b: geometric dissipation-length blend on device (mirror of the
# S3-9 authority amendment) -- endpoint pins + interior-f closed form.
# ---------------------------------------------------------------------------


@requires_gpu
def test_split_e_update_ld_blend_endpoints_bitwise_device():
    """S3-9b l_d-blend pins on the device e-update kernel (authority
    contract: test_dissipation_length_blend_les_and_rans_limits).

    At f = 0 and f = 1 the geometric blend must land BITWISE on the
    pre-S3-9 linear formula's endpoint values -- l = 0*delta + 1*leps
    == leps (f = 0) and l = 1*delta + 0*leps == delta (f = 1) -- so
    the device e result must be bit-identical to its pre-change value.
    The expectation is an INDEPENDENTLY-WRITTEN host mirror of the
    pre-change kernel arithmetic (the linear form's own endpoint
    expressions, the FP32 l_s min, the S3-6d FP64 analytic decay with
    one FP32 rounding at the store, the FP32 E_MIN clip) -- never the
    new device blend itself, so an endpoint regression (device
    pow(x, 1.0) misses bitwise x by 1 ulp on the RTX 5090, which is
    why the kernel branches the endpoints explicitly -- kernel
    comment) fails loudly.  The kernel is driven directly with zeroed
    source/transport fields and uniform theta (dthdz == 0 exactly, so
    e* == e^n bitwise and the decay is the ONLY live channel); dt is
    a power of two so b*dt is exact and FMA contraction of
    1 + b*dt cannot skew the mirror.  The interior-f leg pins the
    geometric closed form at the kernel-arithmetic tier (the CPU
    pin's closed-form idiom) and asserts the retired linear form is
    measurably NOT what the kernel computes."""
    import cupy as cp
    import woof.core.sase as sase
    from woof.core.sase import _kern
    from woof.core.state import DTYPE
    from woof.verify.sase_ref import (C_E, C_ED, C_ES, CKS_BLEND_EXP,
                                       E_MIN, G_ACCEL, LS_COEF, box_filter)

    nz, ny, nx = 6, 8, 8
    shape = (nz, ny, nx)
    rng = np.random.default_rng(SEED + 17)
    e32 = np.maximum(0.05 + 0.1 * box_filter(rng.standard_normal(shape), 4),
                     0.0).astype(np.float32)
    e32[0, :2, :2] = 0.0                       # E_MIN floor cells live
    leps32 = (5.0 + 30.0 * np.abs(
        box_filter(rng.standard_normal(shape), 4))).astype(np.float32)
    n232 = (1.0e-4 * box_filter(rng.standard_normal(shape), 4)) \
        .astype(np.float32)
    assert np.any(n232 > 0.0) and np.any(n232 <= 0.0)   # both l branches
    delta, dt, pr_t = 500.0, 0.5, 0.7
    ncell = nz * ny * nx
    nblocks = (ncell + sase._TPB - 1) // sase._TPB

    def run_device(f):
        e_dev = cp.asarray(e32)
        heat = cp.empty(shape, dtype=DTYPE)
        theta = cp.full(shape, 300.0, dtype=DTYPE)
        zeros = cp.zeros(shape, dtype=DTYPE)   # kv/ph_e/ph_heat/pv/fx/fy
        zcm, zc0, zcp, dz_s, two_dz, h_lo, h_hi, z_mode = \
            sase._z_stencil(shape, dz=DZ)
        partials = cp.zeros((4, nblocks), dtype=cp.float64)
        lepsd = cp.asarray(leps32)
        _kern("sase_split_e_update")(
            (nblocks,), (sase._TPB,),
            (e_dev, heat, theta, cp.asarray(n232), np.int32(1),
             e_dev, np.int32(0),                 # gated M1 dry-n2 dummy
             zeros, lepsd,
             lepsd, np.int32(0),                 # gated S3-12 lb dummy
             zeros, zeros, zeros,
             zeros, zeros, e_dev, np.int32(0),   # gated taper dummy
             partials, zcm, zc0, zcp,
             DTYPE(2.0 * DX), DTYPE(2.0 * DY), dz_s, two_dz,
             h_lo, h_hi, z_mode, DTYPE(dt), DTYPE(f), DTYPE(delta),
             DTYPE(pr_t), DTYPE(C_E), DTYPE(LS_COEF),
             # S3-6k gate OFF: the kernel's decay multiplicand is then
             # the literal c_e, so this fixture keeps pinning exactly
             # the arithmetic it always pinned.
             DTYPE(C_ES), DTYPE(CKS_BLEND_EXP), np.int32(0),
             # S3-12 gate OFF on the same argument.
             DTYPE(C_ED),
             DTYPE(G_ACCEL),
             DTYPE(E_MIN), np.int32(nblocks),
             np.int32(nz), np.int32(ny), np.int32(nx)))
        return cp.asnumpy(e_dev)

    def mirror(l32):
        """Kernel-line host mirror of the post-blend e path for a GIVEN
        FP32 length field (pre-change arithmetic: nothing here touches
        the new blend)."""
        e_minf = np.float32(E_MIN)
        root_e = np.sqrt(np.maximum(e32, e_minf))            # sqrtf(ec)
        l = l32.copy()
        pos = n232 > np.float32(0.0)
        ls = (np.float32(LS_COEF) * root_e[pos]) / np.sqrt(n232[pos])
        l[pos] = np.minimum(l[pos], ls)                      # fminf
        es64 = e32.astype(np.float64)          # e* == e^n (sources zero)
        b = (np.float64(np.float32(C_E))
             * np.sqrt(np.maximum(es64, np.float64(e_minf)))
             / (2.0 * l.astype(np.float64)))
        fac = 1.0 + b * np.float64(np.float32(dt))
        e_dec = (es64 / (fac * fac)).astype(np.float32)
        return np.maximum(e_dec, e_minf)
    # f = 0 endpoint: BITWISE the pre-S3-9 linear formula's value
    # 0*delta + 1*leps == leps (the authority pin's arithmetic trick).
    got0 = run_device(0.0)
    want0 = mirror(np.float32(0.0) * np.float32(delta)
                   + np.float32(1.0) * leps32)
    np.testing.assert_array_equal(got0, want0)
    # f = 1 endpoint: BITWISE the pre-S3-9 linear formula's value
    # 1*delta + 0*leps == delta.
    got1 = run_device(1.0)
    want1 = mirror(np.float32(1.0) * np.float32(delta)
                   + np.float32(0.0) * leps32)
    np.testing.assert_array_equal(got1, want1)
    assert np.any(got0 != got1)                # the pin is potent
    # Interior f = 0.4: the S3-9 geometric closed form (FP64 pow, one
    # FP32 rounding -- kernel-arithmetic tier; pow ulp differences
    # between libm and device pass through the FP32 cast), and the
    # retired linear form must be measurably rejected.
    f32 = np.float32(0.4)
    got_mid = run_device(0.4)
    l_geo = (np.float64(np.float32(delta)) ** np.float64(f32)
             * leps32.astype(np.float64)
             ** (1.0 - np.float64(f32))).astype(np.float32)
    rel_geo, _ = _max_rel(got_mid, mirror(l_geo))
    assert rel_geo <= 2e-6, f"interior-f geometric mirror: {rel_geo:.3e}"
    l_lin = f32 * np.float32(delta) + (np.float32(1.0) - f32) * leps32
    rel_lin, _ = _max_rel(got_mid, mirror(l_lin))
    assert rel_lin > 1e-4, (
        f"interior f indistinguishable from the retired linear blend: "
        f"{rel_lin:.3e}")


# ---------------------------------------------------------------------------
# S3-11b: surface scalar-flux deposit -- device mirror of the S3-11a
# authority seam (surface_scalar_flux_deposit; SFC_SCALAR_FLUX =
# "explicit-deposit-v1"; root cause
# .superpowers/sdd/lake-momentum-root-cause.md), fused into the bottom
# rhs of the sase_thomas_scalar sweep and wired through
# launch_implicit_vertical_diffusion (sfc_flux/sfc_rho1/sfc_fac).
# ---------------------------------------------------------------------------


@requires_gpu
def test_thomas_scalar_surface_deposit_parity_and_validation():
    """S3-11b unit gate: the fused deposit+solve vs the FP64 authority
    composition ``implicit_vertical_diffusion(surface_scalar_flux_
    deposit(phi))`` on FP32-cast inputs, all three thickness modes,
    both authority rows (sfc_fac = CP_AIR theta / 1.0 qv), mixed-sign
    fluxes, at a diffusion number far beyond any explicit bound.  Gate
    2e-6 scale-relative -- the established Thomas unit tier with NO
    widening: the kernel forms the deposit in FP64 with the authority's
    exact op order (dt*flux / ((rho1*fac)*thick_0)) and feeds the sweep
    directly, so no FP divergence beyond the pre-existing sweep
    arithmetic exists (the deposited bottom value never rounds through
    an intermediate FP32 store -- kernel comment).  Also pins the
    launcher rejections (paired args, shape/dtype, positive sfc_fac,
    the faceless nz == 1 refusal)."""
    import cupy as cp
    from woof.core.sase import launch_implicit_vertical_diffusion
    from woof.verify.sase_ref import (CP_AIR, _face_average, box_filter,
                                       implicit_vertical_diffusion,
                                       surface_scalar_flux_deposit)
    rng = np.random.default_rng(SEED + 21)
    nz, ny, nx = SHAPE
    phi32 = box_filter(rng.standard_normal(SHAPE), 4).astype(np.float32)
    kv32 = (5.0 + 2.0 * box_filter(rng.standard_normal(SHAPE), 4)
            ).astype(np.float32)
    assert kv32.min() > 0.0
    rho32 = (1.0 + 0.2 * rng.random((ny, nx))).astype(np.float32)
    hfx32 = (250.0 * rng.standard_normal((ny, nx))).astype(np.float32)
    # qv-row flux scaled so its deposit is O(phi) and the row's
    # arithmetic is actually exercised (a physical 1e-4 QFX against
    # O(1) noise would vanish under the scale-relative gate; the
    # physical-magnitude row is pinned bitwise by the warm-sector
    # fixture below).
    qfx32 = (0.5 * rng.standard_normal((ny, nx))).astype(np.float32)
    assert np.any(hfx32 > 0.0) and np.any(hfx32 < 0.0)
    t_shared = _stretched_dz().astype(np.float32)
    t_percol32 = _per_column_dz32()
    dt = 400.0                                 # diffusion number >> 1
    cases = (
        ("uniform", {"dz": DZ}, {"dz": float(DZ)}),
        ("shared", {"dz_col": t_shared.astype(np.float64)},
         {"dz_col": t_shared.astype(np.float64)}),
        ("per-column", {"dz_col": cp.asarray(t_percol32)},
         {"dz_col": t_percol32.astype(np.float64)}),
    )
    rows = (("theta", hfx32, CP_AIR), ("qv", qfx32, 1.0))
    for label, dev_kw, ref_kw in cases:
        for row, flux32, fac in rows:
            phi_dev = cp.asarray(phi32)
            got = launch_implicit_vertical_diffusion(
                phi_dev, cp.asarray(kv32), dt=dt, **dev_kw,
                sfc_flux=cp.asarray(flux32), sfc_rho1=cp.asarray(rho32),
                sfc_fac=fac)
            assert got is phi_dev              # in place
            # FP64 authority composition on the SAME FP32-cast inputs
            # (parity semantics, module docstring).
            dep_kw = ({"hfx": flux32.astype(np.float64)} if row == "theta"
                      else {"qfx": flux32.astype(np.float64)})
            th64, qv64 = surface_scalar_flux_deposit(
                phi32.astype(np.float64), phi32.astype(np.float64), dt,
                rho32.astype(np.float64), **dep_kw, **ref_kw)
            ref = implicit_vertical_diffusion(
                th64 if row == "theta" else qv64,
                _face_average(kv32.astype(np.float64)), dt, **ref_kw)
            rel, err = _max_rel(cp.asnumpy(phi_dev), ref)
            print(f"deposit+thomas parity [{label}/{row}]: "
                  f"max_rel={rel:.3e} max_abs={err:.3e}")
            assert rel <= 2e-6, (f"[{label}/{row}]: max_rel={rel:.3e} "
                                 f"(gate 2e-6)")
    # Launcher rejections.
    phi_dev = cp.asarray(phi32)
    kv_dev = cp.asarray(kv32)
    flux_dev = cp.asarray(hfx32)
    rho_dev = cp.asarray(rho32)
    with pytest.raises(ValueError, match="together"):
        launch_implicit_vertical_diffusion(phi_dev, kv_dev, dt=dt, dz=DZ,
                                           sfc_flux=flux_dev)
    with pytest.raises(ValueError, match="together"):
        launch_implicit_vertical_diffusion(phi_dev, kv_dev, dt=dt, dz=DZ,
                                           sfc_rho1=rho_dev)
    # A lone row constant is rejected, not silently ignored (review
    # nit): sfc_fac has nothing to scale without the flux pair.
    with pytest.raises(ValueError, match="requires sfc_flux"):
        launch_implicit_vertical_diffusion(phi_dev, kv_dev, dt=dt, dz=DZ,
                                           sfc_fac=1004.5)
    with pytest.raises(ValueError, match="sfc_flux"):
        launch_implicit_vertical_diffusion(
            phi_dev, kv_dev, dt=dt, dz=DZ,
            sfc_flux=cp.asarray(hfx32.astype(np.float64)),
            sfc_rho1=rho_dev)
    with pytest.raises(ValueError, match="sfc_rho1"):
        launch_implicit_vertical_diffusion(
            phi_dev, kv_dev, dt=dt, dz=DZ, sfc_flux=flux_dev,
            sfc_rho1=cp.asarray(rho32[:-1]))
    with pytest.raises(ValueError, match="sfc_fac"):
        launch_implicit_vertical_diffusion(
            phi_dev, kv_dev, dt=dt, dz=DZ, sfc_flux=flux_dev,
            sfc_rho1=rho_dev, sfc_fac=0.0)
    one = cp.zeros((1, ny, nx), dtype=cp.float32)
    with pytest.raises(ValueError, match="faceless"):
        launch_implicit_vertical_diffusion(
            one, one, dt=dt, dz=DZ, sfc_flux=flux_dev, sfc_rho1=rho_dev)


@requires_gpu
def test_thomas_scalar_zero_flux_bitwise_identity():
    """THE S3-11b seam-off pin (spec section 4, mirroring 11a's
    zero-flux identity): with the deposit arguments given but the flux
    field zero, the device sweep is BITWISE-identical to the pre-change
    path (the sfc_flux=None call) -- every thickness mode, both channel
    factors, and STRONGER than the authority contract: the in-kernel
    ``flux != 0.0`` guard adds nothing at all, so identity holds even
    for planted -0.0 bottom values (where the authority's unguarded
    ``x + 0.0`` would flip the sign bit -- the S3-11a docstring caveat,
    unreachable for physical theta/qv; the one documented FP
    divergence) and for a -0.0 flux field.  A mixed-support flux pins
    the per-cell OFF-able contract: zero-flux columns stay bitwise
    while every flux-carrying column moves."""
    import cupy as cp
    from woof.core.sase import launch_implicit_vertical_diffusion
    from woof.verify.sase_ref import box_filter
    rng = np.random.default_rng(SEED + 22)
    nz, ny, nx = SHAPE
    phi32 = box_filter(rng.standard_normal(SHAPE), 4).astype(np.float32)
    assert np.any(phi32 < 0.0)                 # signed field
    phi32[0, 0, 0] = np.float32(0.0)           # planted bottom zeros
    phi32[0, 0, 1] = np.float32(-0.0)
    assert np.signbit(phi32[0, 0, 1])
    kv32 = (5.0 + 2.0 * box_filter(rng.standard_normal(SHAPE), 4)
            ).astype(np.float32)
    rho32 = (1.0 + 0.2 * rng.random((ny, nx))).astype(np.float32)
    t_shared = _stretched_dz().astype(np.float32)
    dt = 400.0
    zero_p = cp.zeros((ny, nx), dtype=cp.float32)
    zero_n = cp.asarray(np.full((ny, nx), -0.0, np.float32))
    assert bool(cp.signbit(zero_n).all())
    rho_dev = cp.asarray(rho32)
    for label, geo in (("uniform", {"dz": DZ}),
                       ("shared", {"dz_col": t_shared.astype(np.float64)}),
                       ("per-column", {"dz_col": cp.asarray(
                           _per_column_dz32())})):
        for kfac in (1.0, 2.0):
            base = cp.asarray(phi32)
            launch_implicit_vertical_diffusion(base, cp.asarray(kv32),
                                               dt=dt, kfac=kfac, **geo)
            base_bytes = cp.asnumpy(base).tobytes()
            for zlabel, zflux in (("+0.0", zero_p), ("-0.0", zero_n)):
                seamed = cp.asarray(phi32)
                launch_implicit_vertical_diffusion(
                    seamed, cp.asarray(kv32), dt=dt, kfac=kfac, **geo,
                    sfc_flux=zflux, sfc_rho1=rho_dev, sfc_fac=1004.5)
                assert cp.asnumpy(seamed).tobytes() == base_bytes, (
                    f"zero-flux identity broken [{label}/kfac={kfac}/"
                    f"{zlabel}]")
    # Mixed support: the seam is OFF-able per cell through the flux
    # alone (driver contract -- no config flag exists).
    mask = rng.random((ny, nx)) < 0.5
    assert mask.any() and (~mask).any()
    flux32 = np.where(mask, np.float32(-180.2), np.float32(0.0)
                      ).astype(np.float32)
    base = cp.asarray(phi32)
    launch_implicit_vertical_diffusion(base, cp.asarray(kv32), dt=dt,
                                       dz=DZ)
    seamed = cp.asarray(phi32)
    launch_implicit_vertical_diffusion(
        seamed, cp.asarray(kv32), dt=dt, dz=DZ,
        sfc_flux=cp.asarray(flux32), sfc_rho1=rho_dev, sfc_fac=1004.5)
    got, ref = cp.asnumpy(seamed), cp.asnumpy(base)
    same_col = np.all(got == ref, axis=0)      # solve is column-local
    assert same_col[~mask].all()               # zero-flux columns bitwise
    assert not same_col[mask].any()            # every flux column moved


@requires_gpu
def test_warm_sector_qfx_device_deposit_closed_form_and_ledger():
    """FIXTURE (b) on device (spec section 5; CPU twin
    test_warm_sector_qfx_moistens_at_closed_form_rate): the moisture
    face of the seam at the physical warm-sector QFX = 1.0e-4
    kg m^-2 s^-1 (LH ~ 250 W/m^2 over XLV = 2.5e6 -- the P2 Td2-miss
    class).  (i) With kv = 0 the sweep degenerates to the exact
    identity (r = 0 columns: unit diagonal, division by 1.0 exact), so
    one fused call IS one bare deposit: the bottom row must equal the
    FP64 closed form qv0 + dt*qfx/((rho1*1.0)*thick_0) rounded once to
    FP32 -- BITWISE, every level above bitwise-untouched, and a
    depositless theta companion bitwise-unchanged.  (ii) Composed over
    20 steps with active K_v = 3: upward moisture propagation, the max
    principle to FP32 storage slack (the FP64 sweep preserves it
    exactly; only the per-step FP32 store can dip, <= ~ULP(qv)/2 ~
    5e-10), the S3-11a boundary-consistent column ledger at rtol 1e-4
    (RATIFIED 2026-07-21 from the provisional 1e-3: MEASURED 1.961e-5
    on the RTX 5090, dual-run identical -- the FP32-storage class the
    derivation predicted; band = 5.1x measured), and trajectory
    parity vs the FP64 authority composition at the established 5e-5
    trajectory tier (MEASURED 2.660e-7)."""
    import cupy as cp
    from woof.core.sase import launch_implicit_vertical_diffusion
    from woof.verify.sase_ref import (implicit_vertical_diffusion,
                                       surface_scalar_flux_deposit)
    nz = 8
    thick = np.array([20.0, 25.0, 32.0, 41.0, 53.0, 68.0, 88.0, 114.0])
    shape = (nz, 3, 3)
    qv0, qfx, rho1, dt = 0.008, 1.0e-4, 1.15, 15.0
    qv32 = np.full(shape, qv0, np.float32)
    qfx_dev = cp.full((3, 3), np.float32(qfx), dtype=cp.float32)
    rho_dev = cp.full((3, 3), np.float32(rho1), dtype=cp.float32)
    # (i) bare deposit through the kv = 0 identity solve -- bitwise.
    kv0 = cp.zeros(shape, dtype=cp.float32)
    q_dev = cp.asarray(qv32)
    launch_implicit_vertical_diffusion(q_dev, kv0, dt=dt, dz_col=thick,
                                       sfc_flux=qfx_dev, sfc_rho1=rho_dev)
    t0 = np.float64(np.float32(thick[0]))      # kernel FP32 thickness
    dep = (np.float64(np.float32(dt)) * np.float64(np.float32(qfx))
           / ((np.float64(np.float32(rho1)) * 1.0) * t0))
    expect0 = np.float32(np.float64(qv32[0, 0, 0]) + dep)
    got = cp.asnumpy(q_dev)
    assert got[0].tobytes() == np.full((3, 3), expect0,
                                       np.float32).tobytes()
    assert got[1:].tobytes() == qv32[1:].tobytes()
    assert expect0 > np.float32(qv0)           # it moistened
    # theta companion: no deposit row given -- bitwise identity.
    th32 = np.full(shape, 300.0, np.float32)
    th_dev = cp.asarray(th32)
    launch_implicit_vertical_diffusion(th_dev, kv0, dt=dt, dz_col=thick)
    assert cp.asnumpy(th_dev).tobytes() == th32.tobytes()
    # (ii) composed channel: 20 steps against active K_v.
    kv3 = cp.full(shape, 3.0, dtype=cp.float32)
    kf64 = np.full((nz - 1, 3, 3), 3.0)        # exact face mean of 3.0
    q_dev = cp.asarray(qv32)
    ref = qv32.astype(np.float64)
    dummy = np.zeros(shape)
    rho64 = np.float64(np.float32(rho1))
    qfx64 = np.float64(np.float32(qfx))
    steps = 20
    for _ in range(steps):
        launch_implicit_vertical_diffusion(
            q_dev, kv3, dt=dt, dz_col=thick, sfc_flux=qfx_dev,
            sfc_rho1=rho_dev)
        _, ref = surface_scalar_flux_deposit(
            dummy, ref, dt, rho64, qfx=qfx64, dz_col=thick)
        ref = implicit_vertical_diffusion(ref, kf64, dt, dz_col=thick)
    got = cp.asnumpy(q_dev).astype(np.float64)
    rel, err = _max_rel(got, ref)
    print(f"warm-sector 20-step parity: max_rel={rel:.3e} "
          f"max_abs={err:.3e}")
    assert rel <= 5e-5                         # trajectory tier
    assert float(got[2, 0, 0]) > qv0           # moisture reached z3
    assert float(got.min()) >= qv0 - 1e-9      # FP32-slack max principle
    dq_col = np.sum(thick[:, None, None] * (got - np.float64(
        np.float32(qv0))), axis=0)
    want = steps * dt * qfx64 / rho64
    print(f"warm-sector ledger: max rel err "
          f"{float(np.max(np.abs(dq_col - want)) / want):.3e}")
    np.testing.assert_allclose(dq_col, want, rtol=1e-4)


def _run_lake_flux_column_device(monkeypatch, hfx, rho1, minutes=10.0,
                                 dt=15.0):
    """FIXTURE (a) engine, device half: the CPU engine
    test_sase._run_lake_flux_column transliterated onto the device
    launchers, step for step -- live launch_n2, the pre-step surface-e
    scalar deposit at the frozen Charnock u*, the f = 0-pinned
    launch_sase_step on frozen wind copies (e advances in place, winds
    are discarded -- the 3-D fetch momentum resupply idiom), the
    launch_vertical_channel f = 0 RANS-limb kv (the same replica the
    CPU engine builds from bl89_rans_lengths/stable_limit_coefficient),
    then THE SEAM: the fused deposit + implicit K_v/Pr_t(0) theta solve
    (launch_implicit_vertical_diffusion sfc_flux/sfc_rho1/sfc_fac =
    CP_AIR).  The f = 0 pin rides the launcher's own host-side tail
    helpers (launch_dynamic_solve/partition_cap/_w_bound_tail on
    woof.core.sase -- the device twin of the CPU engine's sase_ref
    monkeypatches).  ``rho1`` is the CPU engine's dry fixture density
    (its return dict), cast FP32 for the device field.  Returns the
    same series/ledger dict shape as the CPU engine."""
    import cupy as cp
    import woof.core.sase as sase_mod
    import lake_profile_19740403 as lp
    from woof.core.sase import (launch_implicit_vertical_diffusion,
                                 launch_n2, launch_sase_step,
                                 launch_vertical_channel)
    from woof.verify import sase_ref
    from test_sase import _charnock_ust

    monkeypatch.setattr(sase_mod, "launch_dynamic_solve",
                        lambda *a, **k: (0.0, 1.0))
    monkeypatch.setattr(sase_mod, "partition_cap", lambda d, zi: 0.0)
    monkeypatch.setattr(
        sase_mod, "_w_bound_tail",
        lambda *a, **k: sase_ref.WSensorState(
            f_w=1.0, alpha_w=1.0, e_res_w=0.0, coverage=0.0))

    theta0, u0, v0 = lp.THETA_18, lp.U_18, lp.V_18
    thick = lp.THICK_18
    nz = len(theta0)
    shape = (nz, 4, 4)
    z1 = np.cumsum(thick) - 0.5 * thick
    delta = 3000.0                             # the d02 scale
    spd1 = max(float(np.hypot(u0[0], v0[0])), sase_ref.SFC_WSPD_FLOOR)
    ust = _charnock_ust(spd1, z1[0])

    u32 = np.broadcast_to(u0[:, None, None], shape).astype(np.float32)
    v32 = np.broadcast_to(v0[:, None, None], shape).astype(np.float32)
    theta_dev = cp.asarray(
        np.broadcast_to(theta0[:, None, None], shape).astype(np.float32))
    th0_32 = cp.asnumpy(theta_dev).astype(np.float64)
    e_dev = cp.asarray(np.broadcast_to(
        np.maximum(lp.E_RST_18, sase_ref.E_MIN)[:, None, None],
        shape).astype(np.float32))
    w32 = np.zeros(shape, np.float32)
    ust_dev = cp.asarray(np.full(shape[1:], ust, np.float32))
    hfx_dev = cp.asarray(np.full(shape[1:], hfx, np.float32))
    rho_dev = cp.asarray(np.full(shape[1:], rho1, np.float32))
    pr = float(sase_ref.prandtl_blend(0.0))
    e_src = np.float32(dt * ust ** 3
                       / (sase_ref.KARMAN * 0.5 * float(thick[0])))
    steps = int(round(minutes * 60.0 / dt))
    n2_k0, kv26, th1 = [], [], []
    for _ in range(steps):
        n2_dev = launch_n2(theta_dev, dz_col=thick)
        e_dev[0] += e_src
        kv_dev, _ = launch_vertical_channel(e_dev, theta_dev, f=0.0,
                                            n2=n2_dev, dz_col=thick)
        u_step, v_step = cp.asarray(u32), cp.asarray(v32)  # frozen wind
        w_step = cp.asarray(w32)
        ledger = launch_sase_step(
            u_step, v_step, w_step, theta_dev, e_dev, dx=delta, dy=delta,
            dz=200.0, delta=delta, dt=dt, n2=n2_dev, dz_col=thick,
            ust=ust_dev)
        assert ledger["f"] == 0.0              # the pinned RANS limb
        # THE SEAM under test: fused explicit deposit + implicit
        # K_v/Pr_t(0) theta solve (registered SFC_SCALAR_FLUX order).
        launch_implicit_vertical_diffusion(
            theta_dev, kv_dev, dt=dt, kfac=1.0 / pr, dz_col=thick,
            sfc_flux=hfx_dev, sfc_rho1=rho_dev,
            sfc_fac=float(sase_ref.CP_AIR))
        n2_k0.append(float(n2_dev[0, 0, 0]))
        kv26.append(float(kv_dev[1, 0, 0]))
        th1.append(float(theta_dev[0, 0, 0]))
    n2_fin = launch_n2(theta_dev, dz_col=thick)
    th_fin = cp.asnumpy(theta_dev).astype(np.float64)
    dth_col = float(np.sum(thick * (th_fin[:, 0, 0] - th0_32[:, 0, 0])))
    return {"n2_k0": np.array(n2_k0), "kv26": np.array(kv26),
            "th1": np.array(th1), "n2_fin": float(n2_fin[0, 0, 0]),
            "dth_col": dth_col, "rho1": rho1, "steps": steps, "dt": dt}


@requires_gpu
def test_lake_flux_device_stable_layer_and_kv_collapse(monkeypatch):
    """FIXTURE (a) on device (spec section 5): the seam forms the
    marine stable layer ON THE DEVICE PATH, against the CPU authority
    engine (test_sase._run_lake_flux_column -- imported, not copied,
    so the two engines cannot drift) at the observed yolo-b 17Z
    HFX = -180.2 W/m^2, plus the zero-flux RED companion THROUGH the
    live seam (flux field of zeros -- exercising the in-kernel guard
    on a real trajectory).  Mechanism bands are the CPU fixture's
    (measured there: N^2 1.37e-3 vs 8.3e-6, K_v 11.27 -> 0.19 vs
    2.97, separation 15.4x -- 5x-100x wide vs the asserted
    thresholds; device reproduced n2_fin 1.368e-3/8.293e-6 and kv26
    11.273 -> 0.193/2.968).  Parity bands vs the CPU trajectories
    (reference on the FP64 profile, device on its FP32 cast -- input
    quantization deliberately included), RATIFIED 2026-07-21 from the
    provisional derivation-only values against the RTX 5090
    measurement (dual-run identical), each ~4-5x its measured worst
    leg: theta_1 series <= 3e-4 K abs (MEASURED 5.237e-5 GREEN /
    6.853e-5 RED; was 0.02); kv26 series <= 1e-4 of the series scale
    (MEASURED 1.857e-5/2.628e-5; was 2e-2); N^2 series <= 5e-3
    (MEASURED 7.300e-5 GREEN / 1.257e-3 RED -- the RED leg divides by
    a ~1.3e-4 near-neutral scale, quantization-dominated; was 2e-2);
    the boundary-consistent column heat ledger at rtol 5e-3 of the
    -91.65 K m increment (MEASURED 1.525e-3 = 0.140 K m of FP32
    storage noise on the ~2.4e5 K m column content, the derivation's
    predicted class, band 3.3x measured -- the provisional value was
    already in the 3-5x window and stands); RED |dth_col| <= 0.5 K m
    (MEASURED 1.393e-1 -- the same storage-noise magnitude as the
    GREEN residual, as it must be: the deposit itself is
    ledger-exact; was 1.0)."""
    from test_sase import _LAKE_HFX_17Z, _run_lake_flux_column

    green_cpu = _run_lake_flux_column(monkeypatch, hfx=_LAKE_HFX_17Z)
    red_cpu = _run_lake_flux_column(monkeypatch, hfx=0.0)
    green = _run_lake_flux_column_device(monkeypatch, hfx=_LAKE_HFX_17Z,
                                         rho1=green_cpu["rho1"])
    red = _run_lake_flux_column_device(monkeypatch, hfx=0.0,
                                       rho1=green_cpu["rho1"])
    from woof.verify.sase_ref import CP_AIR
    # Mechanism on device: stratification forms only with the seam live.
    print(f"device GREEN: n2_fin={green['n2_fin']:.3e} "
          f"kv26 {green['kv26'][0]:.3f}->{green['kv26'][-1]:.3f} "
          f"th1 {green['th1'][0]:.2f}->{green['th1'][-1]:.2f}")
    print(f"device RED:   n2_fin={red['n2_fin']:.3e} "
          f"kv26 {red['kv26'][0]:.3f}->{red['kv26'][-1]:.3f}")
    assert green["n2_fin"] >= 1.0e-3, green["n2_fin"]
    assert red["n2_fin"] <= 1.0e-4, red["n2_fin"]
    assert green["kv26"][-1] <= 0.4, green["kv26"][-1]
    assert green["kv26"][-1] <= green["kv26"][0] / 10.0
    assert red["kv26"][-1] >= 2.0, red["kv26"][-1]
    assert red["kv26"][-1] / green["kv26"][-1] >= 5.0
    # Device-vs-authority trajectory parity (bands: docstring).
    for leg, dev, ref in (("GREEN", green, green_cpu),
                          ("RED", red, red_cpu)):
        dth = float(np.max(np.abs(dev["th1"] - ref["th1"])))
        dkv = (float(np.max(np.abs(dev["kv26"] - ref["kv26"])))
               / float(np.max(ref["kv26"])))
        dn2 = (float(np.max(np.abs(dev["n2_k0"] - ref["n2_k0"])))
               / max(float(np.max(np.abs(ref["n2_k0"]))), 1e-30))
        print(f"{leg} parity: th1 {dth:.3e} K, kv26 {dkv:.3e}, "
              f"n2 {dn2:.3e}")
        assert dth <= 3e-4, f"{leg} th1 series drift {dth:.3e} K"
        assert dkv <= 1e-4, f"{leg} kv26 series drift {dkv:.3e}"
        assert dn2 <= 5e-3, f"{leg} n2 series drift {dn2:.3e}"
    # Boundary-consistent scalar ledger on the device state: every
    # joule the seam deposited and nothing else (FP32 band: docstring).
    expected = (green["steps"] * green["dt"] * _LAKE_HFX_17Z
                / (green["rho1"] * CP_AIR))
    print(f"lake ledger: dth_col={green['dth_col']:.6f} K m "
          f"(expected {expected:.6f}, rel err "
          f"{abs(green['dth_col'] - expected) / abs(expected):.3e}); "
          f"RED dth_col={red['dth_col']:.3e} K m")
    np.testing.assert_allclose(green["dth_col"], expected, rtol=5e-3)
    # RED deposited nothing: column heat integral is FP32 storage/
    # solver noise only (measured 0.139 K m -- the same magnitude as
    # the GREEN residual, docstring).
    assert abs(red["dth_col"]) <= 0.5, red["dth_col"]


# ---------------------------------------------------------------------------
# S3-9e: on-device gate-2a receipt-equivalence probe (written 2026-07-21
# while yolo-c owned the card; VALIDATED same day after seal -- receipt at
# out/s3-9e-gpu-validation/validation-receipt.json).  The CPU spine
# (bitwise device-graph-vs-legacy identity under numpy) runs in
# tests/test_sase.py; these pin the remaining cupy-vs-numpy kernel
# semantics under the derived tolerances of E_CAP_STATS_COMPARE_RTOL.
# Validation summary (900-s real74 3-dom smokes, each run twice,
# bitwise-identical pairs): 90/90 dual probes equivalent at production
# shapes; sampler share 0.23% of integrate wall (from 6.32%); d03 FP64
# stats transient 1.369 GiB; VRAM peaks 24.9-25.8 GiB under the 29 GiB
# bar; restart-boundary deltas within ~2-4 s (from +7-9 s sync).
# ---------------------------------------------------------------------------

def _gate2a_device_inputs(seed, shape, cast=np.float32):
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    e = rng.uniform(1.0e-4, 40.0, shape).astype(cast)
    u = (rng.standard_normal(shape) * 15.0).astype(cast)
    v = (rng.standard_normal(shape) * 15.0).astype(cast)
    t = rng.uniform(40.0, 600.0, shape).astype(cast)
    ust = rng.uniform(0.0, 1.2, (ny, nx)).astype(cast)
    ust[0, 0] = cast(0.0)
    return e, u, v, t, ust


@requires_gpu
def test_gate2a_device_stats_equivalent_to_host_reference():
    """Dual-path probe on device arrays: fail-loud comparison under the
    derived per-stat tolerances, with the e_max bitwise anchor asserted
    explicitly (a max reduction returns one of its exactly-converted
    inputs on both paths)."""
    import cupy as cp
    from woof.core.sase import sase_e_cap_stats, sase_e_cap_stats_dual

    host = _gate2a_device_inputs(20260721, (49, 96, 96))
    dev = tuple(cp.asarray(a) for a in host)
    out = sase_e_cap_stats_dual(*dev, boundary_width=5,
                                percentile=99.9)   # raises on mismatch
    assert out["stats_path"] == "device"
    probe = out["dual_probe"]
    assert probe["equivalent"]
    assert probe["stats"]["e_max"]["rel_err"] == 0.0
    reference = sase_e_cap_stats(*host, boundary_width=5, percentile=99.9)
    assert out["e_max"] == reference["e_max"]


@requires_gpu
def test_gate2a_device_stats_equivalent_at_d03_shape():
    """The production d03 shape (49 x 501 x 501, spec_bdy_width=5): the
    scale the valley forensics measured.  Also bounds the device path's
    FP64 transient footprint implicitly -- an allocation failure here is
    a VRAM regression signal for the 29-30 GiB hardware bar review."""
    import cupy as cp
    from woof.core.sase import sase_e_cap_stats_dual

    host = _gate2a_device_inputs(19740403, (49, 501, 501))
    dev = tuple(cp.asarray(a) for a in host)
    out = sase_e_cap_stats_dual(*dev, boundary_width=5,
                                percentile=99.9)   # raises on mismatch
    assert out["dual_probe"]["equivalent"]
    assert out["stats_path"] == "device"
    for key in ("e_max", "e_mean", "cap_p", "cap_max", "ratio"):
        assert np.isfinite(out[key]), (key, out[key])


@requires_gpu
def test_gate2a_device_stats_schema_matches_legacy():
    """Receipt schema guard: device dict == legacy keys + stats_path,
    and the echoed metadata is identical."""
    import cupy as cp
    from woof.core.sase import sase_e_cap_stats, sase_e_cap_stats_device

    host = _gate2a_device_inputs(42, (12, 24, 24))
    dev = tuple(cp.asarray(a) for a in host)
    got = sase_e_cap_stats_device(*dev, boundary_width=3)
    ref = sase_e_cap_stats(*host, boundary_width=3)
    assert set(got) == set(ref) | {"stats_path"}
    assert got["cap_percentile"] == ref["cap_percentile"]
    assert got["boundary_width"] == ref["boundary_width"]
    assert all(isinstance(got[k], float) for k in
               ("e_max", "e_mean", "cap_p", "cap_max", "ratio"))


# ---------------------------------------------------------------------------
# S4-2: SASE-M1 moist stability -- device mirror (sase_moist_n2 /
# launch_moist_n2) + the launch_sase_step n2_moist seam, against the S4-1
# authority (sase_ref.moist_n2 / sase_split_step n2_moist; report
# .superpowers/sdd/task-s4-1-report.md).  Specimen columns:
# tests/specimen_yolod_11z.py (yolo-d 11Z d02 amplifier + control).
# ---------------------------------------------------------------------------


def _specimen_device(name, ny=4, nx=4):
    """Specimen column broadcast to a (nz, ny, nx) FP32 device tile
    (the lake-device-engine idiom: horizontal uniformity keeps every
    column identical while exercising the real launch geometry)."""
    import cupy as cp
    from test_sase import _specimen_cols
    c = _specimen_cols(name)
    nz = len(c["thick"])

    def tile(a):
        return cp.asarray(np.ascontiguousarray(np.broadcast_to(
            a.reshape(nz, 1, 1), (nz, ny, nx)).astype(np.float32)))

    return c, {k: tile(c[k]) for k in ("theta", "qv", "qc", "p", "u",
                                       "v", "e0")}


@requires_gpu
def test_moist_n2_device_parity_and_unsaturated_identity():
    """THE M1 UNIT MIRROR GATE (S4-2): launch_moist_n2 vs the FP64
    authority ``moist_n2`` on BOTH yolo-d 11Z specimen columns
    (FP32-cast inputs, authority on the same casts promoted -- the
    established parity semantics), plus the structural identity and
    switch pins:

    * AMPLIFIER: the device substitution mask (out != n2_dry bits)
      equals the authority mask (the saturated deck, >= 5 cells,
      456-1391 m); saturated cells parity <= 2e-6 scale-relative (the
      established tier -- the kernel is FP64 end to end in the
      authority op order, so the measured value is the FP32-store
      class, orders better); unsaturated cells BITWISE the dry input
      (tobytes) -- the switch-false branch adds/changes nothing.
    * CONTROL: the ENTIRE output bitwise == the launch_n2 field (the
      M1 unsaturated-identity contract on device -- a fully
      unsaturated column returns the pre-M1 stability bytes exactly).
    * Thickness-mode invariance: the (nz,) shared column and the
      per-column (nz, ny, nx) broadcast of the same thicknesses
      produce BITWISE identical fields (both promote the same FP32
      thicknesses to the same FP64 in-thread geometry).
    * Launcher validation: missing dz_col, wrong-shaped qv, and the
      SASE_KMAX column bound are rejected.
    """
    import cupy as cp
    from woof.core.sase import launch_moist_n2, launch_n2
    from woof.verify.sase_ref import brunt_vaisala_n2, moist_n2

    for name, want_moist in (("amp", True), ("ctl", False)):
        c, dev = _specimen_device(name)
        thick32 = c["thick"].astype(np.float32)
        thick64 = thick32.astype(np.float64)   # authority on the casts
        nz = len(thick64)
        n2_dev = launch_n2(dev["theta"], dz_col=thick64)
        got = launch_moist_n2(dev["theta"], dev["qv"], dev["qc"],
                              dev["p"], n2_dev, dz_col=thick64)
        th32 = cp.asnumpy(dev["theta"]).astype(np.float64)
        qv32 = cp.asnumpy(dev["qv"]).astype(np.float64)
        qc32 = cp.asnumpy(dev["qc"]).astype(np.float64)
        p32 = cp.asnumpy(dev["p"]).astype(np.float64)
        ref = moist_n2(th32, qv32, qc32, p32, thick64)
        ref_dry = brunt_vaisala_n2(th32, None, dz_col=thick64)
        mask_ref = ref != ref_dry              # authority substitution
        got_h = cp.asnumpy(got)
        dry_h = cp.asnumpy(n2_dev)
        mask_dev = got_h != dry_h
        np.testing.assert_array_equal(mask_dev, mask_ref)
        if want_moist:
            assert mask_ref[:, 0, 0].sum() >= 5
            zc = np.cumsum(thick64) - 0.5 * thick64
            zs = zc[mask_ref[:, 0, 0]]
            assert zs[0] <= 500.0 and zs[-1] >= 1300.0
            rel, err = _max_rel(got_h[mask_ref], ref[mask_ref])
            print(f"moist_n2 [{name}] saturated-cell parity: "
                  f"max_rel={rel:.3e} max_abs={err:.3e} "
                  f"({int(mask_ref.sum())} cells)")
            assert rel <= 2e-6, (f"moist_n2 saturated parity "
                                 f"{rel:.3e} (gate 2e-6)")
            # switch-false cells: the literal dry bits, verbatim.
            assert (got_h[~mask_ref].tobytes()
                    == dry_h[~mask_ref].tobytes())
        else:
            assert not mask_ref.any()
            assert got_h.tobytes() == dry_h.tobytes()
        # Thickness-mode invariance: per-column broadcast == shared.
        t3 = cp.asarray(np.ascontiguousarray(np.broadcast_to(
            thick32[:, None, None], got.shape)))
        got3 = launch_moist_n2(dev["theta"], dev["qv"], dev["qc"],
                               dev["p"], n2_dev, dz_col=t3)
        assert cp.asnumpy(got3).tobytes() == got_h.tobytes()
    # Launcher validation.
    c, dev = _specimen_device("amp")
    with pytest.raises(ValueError, match="dz_col"):
        launch_moist_n2(dev["theta"], dev["qv"], dev["qc"], dev["p"],
                        dev["theta"])
    bad = cp.zeros((2, 2, 2), dtype=cp.float32)
    with pytest.raises(ValueError, match="qv"):
        launch_moist_n2(dev["theta"], bad, dev["qc"], dev["p"],
                        dev["theta"], dz_col=c["thick"])
    deep = cp.zeros((256, 2, 2), dtype=cp.float32)
    with pytest.raises(ValueError, match="SASE_KMAX"):
        launch_moist_n2(deep, deep, deep, deep, deep,
                        dz_col=np.full(256, 50.0))


@requires_gpu
def test_sase_step_moist_seam_inert_identity_and_w_sensor_dry():
    """THE SEAM CONTRACT ON DEVICE (S4-2): launch_sase_step's
    ``n2_moist`` argument, three pins:

    * VALIDATION: n2_moist without n2 raises (the authority
      ValueError, mirrored); wrong dtype rejected.
    * INERT-SEAM BITWISE IDENTITY (the S3-11b zero-guard idiom at step
      level): n2_moist bitwise-equal to n2 (a distinct copy -- every
      cell switch-false) must reproduce the n2-only step EXACTLY --
      u/v/w/e/heat tobytes()-identical and every ledger scalar
      equal -- because the moist-n2 kernel's false branch copies dry
      bits and the e-update's substitution mask is then empty (nothing
      is added, not even +0.0).
    * W-SENSOR KEEPS DRY N2 (the authority's non-substitution point,
      device twin of the CPU spy fixture): a WILDLY different n2_moist
      (dry n2 shifted -0.5 -- every cell substituted) must leave the
      partition diagnostics f_solved/f_cap/f_w/zi/w_coverage and the
      used f EXACTLY equal to the dry run's (they are computed from
      u/v/w/e/theta/n2 only -- if n2_moist ever reached the w-sensor
      screen these would move), while kv and the stepped e MUST differ
      (the moist field really reached points 1-3).
    """
    import cupy as cp
    from woof.core.sase import launch_sase_step
    u32, v32, w32, th32, e32, n232 = _step_fixture32(seed=SEED + 21)

    def dev_fields():
        return [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]

    n2_dev = cp.asarray(n232)
    with pytest.raises(ValueError, match="n2_moist requires n2"):
        launch_sase_step(*dev_fields(), dx=DX, dy=DY, dz=DZ, delta=DELTA,
                         dt=0.05, n2_moist=n2_dev)
    with pytest.raises(ValueError, match="float32"):
        launch_sase_step(*dev_fields(), dx=DX, dy=DY, dz=DZ, delta=DELTA,
                         dt=0.05, n2=n2_dev,
                         n2_moist=cp.zeros(STEP_SHAPE, dtype=cp.float64))
    kw = dict(dx=DX, dy=DY, dz=DZ, delta=DELTA, dt=STEP_DT)
    # Leg A: dry reference (pre-M1 path).
    a = dev_fields()
    heat_a = cp.empty(STEP_SHAPE, dtype=cp.float32)
    led_a = launch_sase_step(*a, n2=n2_dev, heat=heat_a, **kw)
    # Leg B: inert seam -- bit-equal copy through n2_moist.
    b = dev_fields()
    heat_b = cp.empty(STEP_SHAPE, dtype=cp.float32)
    led_b = launch_sase_step(*b, n2=n2_dev, heat=heat_b,
                             n2_moist=n2_dev.copy(), **kw)
    for name, xa, xb in zip("uvwte", a, b):
        assert (cp.asnumpy(xa).tobytes() == cp.asnumpy(xb).tobytes()), (
            f"inert seam changed {name}")
    assert cp.asnumpy(heat_a).tobytes() == cp.asnumpy(heat_b).tobytes()
    ka = {k: v for k, v in led_a.items() if not isinstance(v, cp.ndarray)}
    kb = {k: v for k, v in led_b.items() if not isinstance(v, cp.ndarray)}
    assert ka == kb
    assert (cp.asnumpy(led_a["kv"]).tobytes()
            == cp.asnumpy(led_b["kv"]).tobytes())
    # Leg C: every-cell substitution -- the w-sensor must not see it.
    cdev = dev_fields()
    led_c = launch_sase_step(*cdev, n2=n2_dev,
                             n2_moist=cp.ascontiguousarray(
                                 n2_dev - cp.float32(0.5)), **kw)
    for key in ("f", "f_solved", "f_cap", "f_w", "zi", "w_coverage",
                "c_nu", "pr_t"):
        assert led_c[key] == led_a[key], (key, led_c[key], led_a[key])
    assert (cp.asnumpy(led_c["kv"]).tobytes()
            != cp.asnumpy(led_a["kv"]).tobytes())   # points 1/3 moved
    assert (cp.asnumpy(cdev[4]).tobytes()
            != cp.asnumpy(a[4]).tobytes())          # point 2 moved e


def _run_specimen_column_device(name="amp", m1=True, steps=120, dt=60.0):
    """S4-2 device twin of the CPU specimen-column driver replica
    ``test_sase._run_specimen_column`` (imported as the authority
    reference by the caller, so the two engines cannot drift):
    frozen-state equilibrium on the yolo-d 11Z specimen -- live
    launch_n2 + launch_moist_n2 each step, the driver's pre-step
    surface-e deposit at the specimen u*, launch_sase_step at the d02
    scale (delta = 3000) with the n2_moist seam engaged (m1) or absent
    (the pre-M1 RED leg), winds AND thermo frozen (fresh device copies
    each step; only e integrates -- the registered FROZEN-STATE
    convention, derivation at the CPU engine).  Horizontal uniformity
    degenerates the device solve to exactly (0.0, 0.0)
    (test_dynamic_solve_device_degenerate_uniform_e), so f_used = 0 --
    the pure RANS limb -- with NO monkeypatch (the CPU engine asserts
    the same naturally).  The kv sample replica is
    launch_vertical_channel at f = 0 on n2_eff -- exactly the CPU
    engine's bl89/stable-limit composition.  Returns the CPU engine's
    dict shape plus the per-step e trajectory bytes (identity pins).
    """
    import cupy as cp
    from woof.core.sase import (launch_moist_n2, launch_n2,
                                 launch_sase_step,
                                 launch_vertical_channel)
    from woof.verify import sase_ref
    c, dev = _specimen_device(name)
    thick = c["thick"]
    nz = len(thick)
    delta = 3000.0
    e_dev = cp.maximum(dev["e0"], cp.float32(sase_ref.E_MIN))
    src = c["ust"] ** 3 / (sase_ref.KARMAN * 0.5 * thick[0])
    e_src = np.float32(dt * src)
    w32 = np.zeros(e_dev.shape, np.float32)
    pr = float(sase_ref.prandtl_blend(0.0))
    samples = {}
    mask = None
    traj = []
    for n in range(steps):
        n2_dev = launch_n2(dev["theta"], dz_col=thick)
        n2m_dev = launch_moist_n2(dev["theta"], dev["qv"], dev["qc"],
                                  dev["p"], n2_dev, dz_col=thick)
        if mask is None:
            mask = (cp.asnumpy(n2m_dev) != cp.asnumpy(n2_dev))[:, 0, 0]
        n2_eff = n2m_dev if m1 else n2_dev
        e_dev[0] += e_src                      # driver pre-step deposit
        # S4-3c replica fidelity (mirror of the CPU replica's S4-3b
        # amendment): the amended split-step step 2b bounds the
        # RANS-limb lengths by the M1b moist excursion length when the
        # seam is engaged (n2_dry gates it), so the exported-kv replica
        # must ride the SAME composition.  On THIS specimen the limb is
        # measured slack (CPU pin test_m1b_specimen_kh: bitwise-inert,
        # min margin 0.02 m), so every S4-2 measured number stands.
        lkw = {"n2_dry": n2_dev} if m1 else {}
        kv_dev, _ = launch_vertical_channel(e_dev, dev["theta"], f=0.0,
                                            n2=n2_eff, dz_col=thick,
                                            **lkw)
        u_step, v_step = dev["u"].copy(), dev["v"].copy()
        w_step = cp.asarray(w32)
        ledger = launch_sase_step(
            u_step, v_step, w_step, dev["theta"], e_dev, dx=delta,
            dy=delta, dz=200.0, delta=delta, dt=dt, n2=n2_dev,
            dz_col=thick, n2_moist=(n2m_dev if m1 else None))
        assert (ledger["c_nu"], ledger["f"]) == (0.0, 0.0)  # RANS limb
        traj.append(cp.asnumpy(e_dev).tobytes())
        minute = (n + 1) * dt / 60.0
        if minute in (90.0, 120.0):
            e_h = cp.asnumpy(e_dev).astype(np.float64)
            kh_h = (cp.asnumpy(kv_dev).astype(np.float64) / pr)
            samples[minute] = {"e": e_h[:, 0, 0].copy(),
                               "kh": kh_h[:, 0, 0].copy()}
    z1 = np.cumsum(thick) - 0.5 * thick
    return {"z": z1, "mask": mask, "samples": samples, "traj": traj}


@requires_gpu
def test_m1_specimen_device_uptake_and_dry_shortfall():
    """THE G-M5 UPTAKE FIXTURE ON DEVICE (S4-2 twin of the CPU
    test_m1_specimen_uptake; authority engine IMPORTED, not copied):

    * RED (pre-M1 device path, no n2_moist): the amplifier deck's
      equilibrium reproduces the early-ci-cap-audit dry shortfall --
      layer K_h <= 0.02 m2/s, layer TKE <= 1e-3 m2/s2 (the CPU RED
      bands verbatim).
    * GREEN (n2_moist live): layer-mean TKE inside the G-M5 band
      [0.5, 1.6] at BOTH 90 and 120 min, layer-mean K_h >= 3 (entered
      the reference class from the ~1e-5 shortfall) and <= 250 (the
      measured-class runaway pin).  These mirror the PASSING CPU
      fixture's achieved-number pins; the [3, 40] G-M5 ceiling xfail
      stays CPU-side UNCHANGED (S4-1 registered deviation, coordinator
      adjudication) -- deliberately NOT duplicated here.
    * DEVICE-VS-AUTHORITY PARITY on both specimen columns: the device
      substitution mask equals the CPU engine's; GREEN layer-mean TKE
      and K_h vs the FP64 CPU trajectory at <= 5e-7 relative
      (RATIFIED 2026-07-22 from the provisional 2e-2 derivation band:
      MEASURED worst 1.261e-7 on the RTX 5090, dual-run identical --
      the frozen-state equilibrium is a fixed-point attractor of the
      bit-identical frozen inputs, so the FP32 trajectory converges to
      the FP64 equilibrium at FP32 resolution rather than accumulating
      drift; band = 4.0x measured).
    """
    from test_sase import _run_specimen_column

    cpu_red = _run_specimen_column("amp", m1=False)
    cpu_grn = _run_specimen_column("amp", m1=True)
    dev_red = _run_specimen_column_device("amp", m1=False)
    dev_grn = _run_specimen_column_device("amp", m1=True)
    mask = cpu_red["mask"]
    np.testing.assert_array_equal(dev_red["mask"], mask)
    np.testing.assert_array_equal(dev_grn["mask"], mask)
    assert mask.sum() >= 5
    for minute in (90.0, 120.0):
        s = dev_red["samples"][minute]
        assert s["kh"][mask].max() <= 0.02, s["kh"][mask].max()
        assert s["e"][mask].max() <= 1.0e-3, s["e"][mask].max()
    worst = 0.0
    for minute in (90.0, 120.0):
        s = dev_grn["samples"][minute]
        r = cpu_grn["samples"][minute]
        tke = float(s["e"][mask].mean())
        kh = float(s["kh"][mask].mean())
        assert 0.5 <= tke <= 1.6, (minute, tke)      # G-M5 TKE band
        assert kh >= 3.0, (minute, kh)               # entered the class
        assert kh <= 250.0, (minute, kh)             # measured-class pin
        rel_e = abs(tke - float(r["e"][mask].mean())) \
            / float(r["e"][mask].mean())
        rel_k = abs(kh - float(r["kh"][mask].mean())) \
            / float(r["kh"][mask].mean())
        worst = max(worst, rel_e, rel_k)
        print(f"specimen GREEN {minute:.0f} min: device TKE={tke:.5f} "
              f"K_h={kh:.3f} (CPU {float(r['e'][mask].mean()):.5f}/"
              f"{float(r['kh'][mask].mean()):.3f}); "
              f"rel e={rel_e:.3e} kh={rel_k:.3e}")
    assert worst <= 5e-7, f"GREEN layer-mean parity {worst:.3e}"


@requires_gpu
def test_m1_control_column_device_live_seam_bitwise():
    """THE DRIVER-LEVEL UNSATURATED IDENTITY ON DEVICE (S4-2 twin of
    the CPU test_m1_dry_limit_bitwise): on the control column -- qc = 0
    everywhere, RH < 78% everywhere (pinned at import by the specimen
    file) -- the trajectory THROUGH the live n2_moist seam is
    tobytes()-identical to the pre-M1 device trajectory at every one of
    12 steps, and the moist-n2 field itself is bitwise the dry field
    each step.  This is the binding S4-2 requirement pin: unsaturated
    columns bitwise identical to the pre-M1 device path -- the switch
    mask false adds/changes nothing anywhere in the step."""
    pre = _run_specimen_column_device("ctl", m1=False, steps=12)
    live = _run_specimen_column_device("ctl", m1=True, steps=12)
    assert not pre["mask"].any()               # the switch never fired
    assert pre["traj"] == live["traj"]         # seam engaged, inert


def _least_pass_ms(fn, reps, passes=3):
    """ms/call of ``fn``: one warm call, then ``passes`` event-timed passes of
    ``reps`` calls each, and the least of them.

    The least pass is what the card alone costs: a pass that shared the card
    with another process reads slow, while a regression slows every pass.
    On an idle card the passes agree to well under 1%, so the pins below
    guard the same thing they guarded as single passes.

    That absorbs a burst of another process's work, not a card shared for
    the whole test.  MEASURED 2026-09-29 on the RTX 5070 Ti with a
    concurrent float32 matmul stream holding it at 69-73%: moist_n2 read
    16.085 ms (its idle 8.614, pin 14.2, so it fails) and the M1b increment
    12.296 ms (idle 6.592, pin 13.2).  The pins are not loosened to cover
    that, for the reason the M2 test states: a bound that survived a 2x
    contention factor would guard nothing, so these are idle-card pins."""
    import cupy as cp

    fn()                                       # warm (pool/compile)
    readings = []
    for _ in range(passes):
        s0, e0 = cp.cuda.Event(), cp.cuda.Event()
        s0.record()
        for _ in range(reps):
            fn()
        e0.record()
        e0.synchronize()
        readings.append(cp.cuda.get_elapsed_time(s0, e0) / reps)
    return min(readings)


def _device_name():
    import cupy as cp

    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(name, (bytes, bytearray)):
        name = bytes(name).decode("utf-8", errors="replace")
    return name


def _assert_card_pin(table, measured_ms, message):
    """Assert ``table``'s pin for this card, or report an uncalibrated one.

    The derivation ``_M2_VENT_D02_PIN_MS`` states: a pin is a factor over
    ONE card's own idle baseline, so asserting it on another card measures
    the hardware.  The M1 and M1b pins were single numbers from the RTX 5090
    and failed on two slower sm_120 cards with nothing regressed (the RTX
    5070 Ti and an RTX PRO 4500 read moist_n2 at 8.6 and 8.9 ms against the
    5090's 6.5 ms pin, the M1b increment at 6.6 and 6.8 against 6.0)."""
    card = _device_name()
    pin = table.get(card)
    if pin is None:
        print(f"  UNCALIBRATED CARD {card!r}: {measured_ms:.3f} ms RECORDED, "
              f"not asserted -- measure this card's idle baseline and add "
              f"its row before relying on this pin here")
        return
    assert measured_ms <= pin, message.format(ms=measured_ms, card=card,
                                              pin=pin)


#: moist_n2 at the d02 shape (49, 501, 501), ms/call, per card: 1.65x the
#: card's own idle baseline, the factor the RTX 5090 pin has carried since
#: 2026-07-22 (docstring of the test below).
_M1_MOIST_N2_D02_PIN_MS = {
    # 3.95 ms baseline, 2026-07-22.
    "NVIDIA GeForce RTX 5090": 6.5,
    # 8.656 ms idle baseline, the slower of two runs of five passes
    # (8.631-8.657 ms), 2026-09-29, test host 4, NVRTC 13.4.92, driver 595.91.07.
    "NVIDIA GeForce RTX 5070 Ti": 14.2,
}

#: The M1b vertical-channel increment at the d02 shape, ms/call, per card:
#: about 2x the card's own idle baseline, the RTX 5090 derivation (6.0 ms
#: over its 2.965/2.957 ms dual-run pair, 2026-07-22).
_M1B_INCREMENT_D02_PIN_MS = {
    "NVIDIA GeForce RTX 5090": 6.0,
    # 6.622 ms idle baseline, the slower of two runs of five passes
    # (6.622-6.630 ms), 2026-09-29, test host 4, NVRTC 13.4.92, driver 595.91.07.
    "NVIDIA GeForce RTX 5070 Ti": 13.2,
}


@requires_gpu
def test_sase_driver_d02_first_light_moist_50_steps(monkeypatch):
    """M1 FIRST LIGHT, DRIVER LEVEL (S4-2): 50 full model steps on an
    ADMITTED d02-class configuration at production-class column count
    (dx = dy = 3 km, dt = 15 s, 49 x 250 x 250, moist Morrison,
    sfclay + Noah, zero-flux radiation stub through the sanctioned
    injection seam) carrying a SUSTAINED saturated stratocumulus-like
    deck (qc = 0.2 g/kg AND qv = qs,liq in the 450-1400 m band over a
    130 x 130 warm-sector patch; qv capped at 0.7*qs elsewhere so the
    synthetic profile's cold top rows do not saturate spuriously) --
    the first driver executions with the n2_moist seam live, with
    saturated and unsaturated columns both present.  Logged per early
    step: how many columns switch moist (>= 1 substituted cell), via
    the same launch_n2/launch_moist_n2 pair the driver runs; the deck
    must ENGAGE from step 0 and STAY engaged (RH = 100% + condensate
    is Morrison-stable, unlike a bare-qc plant that evaporates in one
    step).  Stability bands mirror the d01 driver gate: prognostics +
    e_sgs finite, e_sgs in [E_MIN, 50], |u| <= 40, |w| <= 20.

    KERNEL-TIME BUDGET CHECK: the plan's <= 5% d01-d03 kernel-time
    budget is a PER-RUN receipt bar (run_yolo.py's
    sase_share_of_integrate_wall against the full production step:
    RRTMG, Morrison hydrometeors, nesting -- none of which this
    stripped fixture carries, so a seam-vs-wall share here would be
    denominator-inflated ~6x and meaningless either way).  What CAN be
    bound at unit scale is the M1 INCREMENT, calibrated against the
    production receipt: yolo-d (pre-M1) measured the d02 seam at
    62.12 ms/call (out/sase-yolo-d/run-metrics.json seam_ms_mean,
    2026-07-22 baseline; combined sase_share_of_integrate_wall
    6.91%).  launch_moist_n2 at the EXACT d02 shape (49, 501, 501)
    measured 3.95 ms/call (regime-independent: the cost is the pass-1
    FP64 pow/exp saturation chain, needed at every cell for the
    switch itself) = 6.4% of the pre-M1 d02 seam, projecting the
    combined share to ~7.3% (+ ~0.4% of integrate wall).  BINDING PIN,
    per card (``_M1_MOIST_N2_D02_PIN_MS``): <= 6.5 ms/call at that shape
    on the RTX 5090 (1.65x measured -- regression guard, not a budget
    verdict), and 1.65x its own idle baseline on every other calibrated
    card, read as the least of three passes; the budget verdict itself is
    re-scored on the S4-3 smoke receipt (adjudicated, and flagged in
    the S4-2 report)."""
    import time

    import cupy as cp
    from woof.config import RunConfig, validate_run_config
    from woof.core import dycore
    from woof.core import physics as physics_mod
    from woof.core.physics import initialize_physics
    from woof.core.sase import (launch_moist_n2, launch_n2,
                                 launch_sase_step)
    from woof.verify.sase_ref import (CP_AIR, E_MIN, EP2_RV, P0_REF,
                                       RD_AIR, SVP1, SVP2, SVP3, SVPT0)

    monkeypatch.setenv("WOOF_SASE_DEBUG_TIMING", "1")
    cfg = RunConfig(nx=250, ny=250, nz=49, dx=3000.0, dy=3000.0,
                    ztop=16000.0, dt=15.0, run_seconds=750.0,
                    time_step_sound=4, moist=True, mp_physics=10,
                    ra_physics=4, sf_sfclay_physics=1,
                    sf_surface_physics=2, km_opt=0,
                    bl_pbl_physics=_SASE_SELECTOR)
    assert validate_run_config(cfg) is cfg     # the admitted combination
    state = _moist_driver_state(cfg, stretch=1.6)
    z_half = np.asarray(state.height_half(), np.float64)
    # Sustained saturated deck: liquid qs from the initial state (the
    # authority's own Tetens/Exner chain), qv = qs + qc = 0.2 g/kg in
    # the deck patch (Morrison-stable: RH 100% with condensate neither
    # evaporates nor needs to condense), qv capped at 0.7*qs everywhere
    # else (kills the synthetic profile's spurious cold-top
    # saturation, so the switch-count log measures THE DECK and the
    # control columns are genuinely unsaturated).
    th_tot = cp.asnumpy(state.total_theta()).astype(np.float64)
    p_eos = cp.asnumpy(state.p).astype(np.float64)
    t_h = th_tot * (p_eos / P0_REF) ** (RD_AIR / CP_AIR)
    es_h = 1000.0 * SVP1 * np.exp(SVP2 * (t_h - SVPT0) / (t_h - SVP3))
    qs_h = EP2_RV * es_h / (p_eos - es_h)
    deck = (z_half >= 450.0) & (z_half <= 1400.0)
    patch = np.zeros((cfg.ny, cfg.nx), bool)
    patch[60:190, 60:190] = True
    deck3 = deck[:, None, None] & patch[None]
    qv_h = np.minimum(cp.asnumpy(state.qv).astype(np.float64),
                      0.7 * qs_h)
    qv_h[deck3] = qs_h[deck3]
    qc_h = np.zeros_like(qv_h)
    qc_h[deck3] = 2.0e-4
    state.qv[...] = cp.asarray(qv_h.astype(np.float32))
    state.qc[...] = cp.asarray(qc_h.astype(np.float32))
    shear = (5.0 + 8.0 * z_half / cfg.ztop).astype(np.float32)
    state.u[...] = cp.asarray(np.broadcast_to(
        shear[:, None, None], (cfg.nz, cfg.ny, cfg.nx + 1)))
    state.v[...] = cp.float32(1.0)
    state.e_sgs[...] = cp.float32(0.01)
    n_patch = int(patch.sum())                 # 16900 saturated columns

    initialize_physics(state, cfg, landmask=1.0, tsk=302.0,
                       swdown=400.0, glw=320.0, radiation=_zero_radiation)
    # (The opt-in per-call CUDA-event instrumentation is not part
    # of this port -- it existed to feed a campaign harness that did not
    # come with it -- so the seam-share print below is dropped and the
    # stability assertions, which are the test's subject, remain.)
    switch_counts = []
    wall_steps = 0.0
    for step in range(50):
        if step < 5 or step == 49:
            atm = physics_mod._prepare_atmosphere(state)
            n2 = launch_n2(atm["theta"], dz_col=atm["dz"])
            n2m = launch_moist_n2(atm["theta"], atm["qv"], atm["qc"],
                                  atm["pressure"], n2, dz_col=atm["dz"])
            count = int((n2m != n2).any(axis=0).sum())
            switch_counts.append((step, count))
            del atm, n2, n2m
        t0 = time.perf_counter()
        dycore.step(state, cfg)                # physics runs inside
        cp.cuda.Stream.null.synchronize()      # accurate GPU step wall
        wall_steps += time.perf_counter() - t0
        e = state.e_sgs
        assert bool(cp.isfinite(e).all()), f"e_sgs NaN at step {step + 1}"
        e_max = float(e.max())
        assert float(e.min()) >= np.float32(E_MIN) and e_max <= 50.0, (
            f"e_sgs unbounded at step {step + 1}: max {e_max}")
    print("d02 first-light moist switch counts (step, columns): "
          + ", ".join(f"({s}, {c})" for s, c in switch_counts))
    assert switch_counts[0][1] >= n_patch, (
        f"moist path did not engage: {switch_counts[0]} vs the "
        f"{n_patch}-column saturated patch")
    # The deck must STAY engaged through the early steps and to the
    # end (>= half the patch: Morrison/dynamics may nibble the edges).
    for s, c in switch_counts[1:]:
        assert c >= n_patch // 2, (
            f"moist path collapsed at step {s}: {c} columns")
    for name in ("u", "v", "w", "thp", "qv", "qc", "e_sgs"):
        arr = cp.asnumpy(getattr(state, name))
        assert np.all(np.isfinite(arr)), name
    assert float(cp.abs(state.u).max()) <= 40.0
    assert float(cp.abs(state.w).max()) <= 20.0
    cp.cuda.Stream.null.synchronize()
    print(f"driver first light: 50 steps at dt=15 on "
          f"(49, 250, 250); final e_max={float(state.e_sgs.max()):.4f}, "
          f"u_max={float(cp.abs(state.u).max()):.2f} -- STABLE; "
          f"stripped-step wall {wall_steps * 1000.0:.1f} ms")
    shape_p = (49, 501, 501)
    rng = np.random.default_rng(SEED + 30)
    zp = np.cumsum(np.full(49, 300.0)) - 150.0
    th_p = cp.ascontiguousarray(cp.asarray(np.broadcast_to(
        (300.0 + 0.004 * zp).astype(np.float32)[:, None, None],
        shape_p)))
    qv_p = cp.full(shape_p, 8.0e-3, dtype=cp.float32)
    qc_p = cp.zeros(shape_p, dtype=cp.float32)
    qc_p[8:16] = cp.float32(2.0e-4)            # a live saturated deck
    p_p = cp.ascontiguousarray(cp.asarray(np.broadcast_to(
        (95000.0 * np.exp(-zp / 8000.0)).astype(np.float32)
        [:, None, None], shape_p)))
    dz_p = cp.full(shape_p, 300.0, dtype=cp.float32)
    u_p = cp.asarray((rng.standard_normal(shape_p) * 5.0)
                     .astype(np.float32))
    v_p = cp.asarray((rng.standard_normal(shape_p) * 5.0)
                     .astype(np.float32))
    w_p = cp.zeros(shape_p, dtype=cp.float32)
    e_p = cp.full(shape_p, 0.5, dtype=cp.float32)

    n2_p = launch_n2(th_p, dz_col=dz_p)
    t_moist = _least_pass_ms(lambda: launch_moist_n2(
        th_p, qv_p, qc_p, p_p, n2_p, dz_col=dz_p), 10)
    n2m_p = launch_moist_n2(th_p, qv_p, qc_p, p_p, n2_p, dz_col=dz_p)
    t_step = _least_pass_ms(lambda: launch_sase_step(
        u_p, v_p, w_p, th_p, e_p, dx=3000.0, dy=3000.0, dz=300.0,
        delta=3000.0, dt=15.0, n2=n2_p, dz_col=dz_p, n2_moist=n2m_p), 3,
        passes=1)
    print(f"d02-shape M1 increment: moist_n2 {t_moist:.3f} ms/call "
          f"(vs yolo-d pre-M1 d02 seam mean 62.12 ms: "
          f"{100 * t_moist / 62.12:.2f}%; one launch_sase_step here "
          f"{t_step:.3f} ms)")
    _assert_card_pin(_M1_MOIST_N2_D02_PIN_MS, t_moist, (
        "moist_n2 at the production d02 shape regressed to {ms:.3f} "
        "ms/call on {card} (pin {pin} ms = 1.65x that card's idle "
        "baseline; table above the test)"))


# ---------------------------------------------------------------------------
# S4-3c: SASE-M1b moist master-length limb -- device mirror (the
# sase_vertical_channel n2_dry seam + the launch_sase_step step-2b wiring)
# against the S4-3b authority (sase_ref.bl89_moist_excursion_lengths /
# bl89_rans_lengths n2_dry seam, commit 697bc55).  Fixtures: the
# constructed deck-under-lid column (imported from tests/test_sase.py --
# the G-M3 regime where the limb BINDS) and the yolo-d 11Z specimen
# columns (where the limb is CPU-pinned slack).
# ---------------------------------------------------------------------------


def _deck_cast32():
    """The S4-3b deck-under-lid fixture column, cast ONCE to FP32.

    Both engines below consume these same casts (device as tiles, the
    authority reference promoted back to FP64) -- the module's parity
    semantics: gates measure kernel arithmetic, not input quantization
    of the FP64-constructed column."""
    from test_sase import _build_deck_column
    th1, qv1, qc1, p1, thick, zc, deck = _build_deck_column()
    cast = {k: np.asarray(a, np.float64).astype(np.float32)
            for k, a in (("theta", th1), ("qv", qv1), ("qc", qc1),
                         ("p", p1))}
    return cast, np.asarray(thick, np.float64), zc, deck


def _deck_device(ny=4, nx=4):
    """FP32 device tiles of the deck-under-lid casts (the
    lake-device-engine idiom: horizontal uniformity keeps every column
    identical while exercising the real launch geometry)."""
    import cupy as cp
    cast, thick, zc, deck = _deck_cast32()
    nz = len(thick)

    def tile(a):
        return cp.asarray(np.ascontiguousarray(np.broadcast_to(
            a.reshape(nz, 1, 1), (nz, ny, nx))))

    return {k: tile(v) for k, v in cast.items()}, cast, thick, zc, deck


@requires_gpu
def test_m1b_vertical_channel_limb_parity_and_identity():
    """THE M1b UNIT MIRROR GATE (S4-3c): the amended
    ``sase_vertical_channel`` n2_dry seam vs the FP64 authority
    ``bl89_rans_lengths(..., n2_dry=...)`` + ``stable_limit_coefficient``
    composition (the S4-3b step-2b amendment), on the fixture classes
    that pin both limb states:

    * DECK-UNDER-LID (the limb BINDS -- the G-M3 regime): kv and leps
      scale-relative <= 2e-6 (the established tier) at f = 0 (pure RANS
      limb, where the G-M3 defect lives), f = 0.3 (interior blend), and
      f = 1; non-vacuity asserted through the authority (the bounded
      lid-adjacent k12 lengths leave the unbounded composition) AND on
      device (kv/leps with n2_dry differ from the no-limb call at f=0).
    * f = 1 LES ENDPOINT: device kv with the limb engaged is BITWISE
      the no-limb kv (the FP-exact two-product argument -- the limb
      cannot reach the LES limb; authority test_m1b_les_inert twin at
      kernel level).  leps may move (its l_d consumer takes the delta
      branch at f = 1 -- inert downstream, the authority contract).
    * BOTH 11Z SPECIMEN COLUMNS at the established <= 2e-6 tier with
      the seam engaged (amp: live substitution mask, limb slack by the
      CPU pin; ctl: empty mask).
    * UNSATURATED BITWISE IDENTITY (ctl): n2_moist is bitwise the dry
      field, so the with-limb call must reproduce the no-limb call's
      exact bytes (the empty-mask gate contract on device -- nothing
      is added, not even +0.0).
    * Launcher validation: n2_dry without n2 raises (the authority
      contract, mirrored); wrong-shaped n2_dry rejected.
    """
    import cupy as cp
    from woof.core.sase import (launch_moist_n2, launch_n2,
                                 launch_vertical_channel)
    from woof.verify import sase_ref

    # ---- deck-under-lid: the binding-limb parity leg -----------------
    dev, cast, thick, zc, deck = _deck_device()
    nz = len(thick)
    shape = (nz, 4, 4)
    n2_dev = launch_n2(dev["theta"], dz_col=thick)
    n2m_dev = launch_moist_n2(dev["theta"], dev["qv"], dev["qc"],
                              dev["p"], n2_dev, dz_col=thick)
    # authority on the same casts promoted -- the device n2 fields are
    # themselves the shared inputs here, so the substitution mask
    # (n2 != n2_dry) agrees bit for bit between the engines.
    n2m64 = cp.asnumpy(n2m_dev).astype(np.float64)
    n264 = cp.asnumpy(n2_dev).astype(np.float64)
    theta64 = cp.asnumpy(dev["theta"]).astype(np.float64)
    assert np.any(n2m64 != n264)               # live mask (deck cells)
    rng = np.random.default_rng(SEED + 40)
    e64 = np.maximum(
        0.35 + 0.3 * sase_ref.box_filter(rng.standard_normal(shape), 4),
        0.01).astype(np.float32).astype(np.float64)
    e_dev = cp.asarray(e64.astype(np.float32))
    z = zc.reshape(-1, 1, 1)
    tcol = thick.reshape(-1, 1, 1)
    # non-vacuity through the authority: the limb bites, and at the
    # lid-adjacent cell k12 specifically (the G-M3 mechanism cell).
    l_mix_nb, l_eps_nb = sase_ref.bl89_rans_lengths(theta64, e64, z,
                                                    tcol, n2m64)
    l_mix_b, l_eps_b = sase_ref.bl89_rans_lengths(theta64, e64, z, tcol,
                                                  n2m64, n2_dry=n264)
    assert np.all(l_mix_b[12] < l_mix_nb[12])
    assert np.all(l_eps_b[12] < l_eps_nb[12])
    e_fl = np.maximum(e64, sase_ref.E_MIN)
    root_e = np.sqrt(e_fl)
    l_les = sase_ref.vertical_mixing_length(z, e64, n2m64)
    c_r = sase_ref.stable_limit_coefficient(l_mix_b, e_fl, n2m64)
    for f in (0.0, 0.3, 1.0):
        kv_d, leps_d = launch_vertical_channel(
            e_dev, dev["theta"], f=f, n2=n2m_dev, n2_dry=n2_dev,
            dz_col=thick)
        kv_ref = (f * (sase_ref.C_KV * l_les * root_e)
                  + (1.0 - f) * (c_r * l_mix_b * root_e))
        rel_kv, _ = _max_rel(cp.asnumpy(kv_d), kv_ref)
        rel_le, _ = _max_rel(cp.asnumpy(leps_d), l_eps_b)
        print(f"deck f={f}: kv rel={rel_kv:.3e} leps rel={rel_le:.3e}")
        assert rel_kv <= 2e-6, (f, rel_kv)
        assert rel_le <= 2e-6, (f, rel_le)
    # device non-vacuity + the f = 1 FP-exact endpoint
    kv_w0, le_w0 = launch_vertical_channel(e_dev, dev["theta"], f=0.0,
                                           n2=n2m_dev, n2_dry=n2_dev,
                                           dz_col=thick)
    kv_n0, le_n0 = launch_vertical_channel(e_dev, dev["theta"], f=0.0,
                                           n2=n2m_dev, dz_col=thick)
    assert bool((cp.asnumpy(kv_w0)[12] < cp.asnumpy(kv_n0)[12]).all())
    assert bool((cp.asnumpy(le_w0)[12] < cp.asnumpy(le_n0)[12]).all())
    kv_w1, _ = launch_vertical_channel(e_dev, dev["theta"], f=1.0,
                                       n2=n2m_dev, n2_dry=n2_dev,
                                       dz_col=thick)
    kv_n1, _ = launch_vertical_channel(e_dev, dev["theta"], f=1.0,
                                       n2=n2m_dev, dz_col=thick)
    assert cp.asnumpy(kv_w1).tobytes() == cp.asnumpy(kv_n1).tobytes()

    # ---- both specimen columns at the established tier ---------------
    for name in ("amp", "ctl"):
        c, sdev = _specimen_device(name)
        thick64 = c["thick"].astype(np.float32).astype(np.float64)
        n2s_dev = launch_n2(sdev["theta"], dz_col=thick64)
        n2sm_dev = launch_moist_n2(sdev["theta"], sdev["qv"], sdev["qc"],
                                   sdev["p"], n2s_dev, dz_col=thick64)
        e_s = cp.maximum(sdev["e0"], cp.float32(sase_ref.E_MIN))
        kv_d, leps_d = launch_vertical_channel(
            e_s, sdev["theta"], f=0.0, n2=n2sm_dev, n2_dry=n2s_dev,
            dz_col=thick64)
        th64 = cp.asnumpy(sdev["theta"]).astype(np.float64)
        nm64 = cp.asnumpy(n2sm_dev).astype(np.float64)
        nd64 = cp.asnumpy(n2s_dev).astype(np.float64)
        es64 = cp.asnumpy(e_s).astype(np.float64)
        zs = (np.cumsum(thick64) - 0.5 * thick64).reshape(-1, 1, 1)
        ts = thick64.reshape(-1, 1, 1)
        lm_r, le_r = sase_ref.bl89_rans_lengths(th64, es64, zs, ts,
                                                nm64, n2_dry=nd64)
        efl = np.maximum(es64, sase_ref.E_MIN)
        cr_s = sase_ref.stable_limit_coefficient(lm_r, efl, nm64)
        kv_ref = cr_s * lm_r * np.sqrt(efl)
        rel_kv, _ = _max_rel(cp.asnumpy(kv_d), kv_ref)
        rel_le, _ = _max_rel(cp.asnumpy(leps_d), le_r)
        print(f"specimen [{name}]: kv rel={rel_kv:.3e} "
              f"leps rel={rel_le:.3e}")
        assert rel_kv <= 2e-6, (name, rel_kv)
        assert rel_le <= 2e-6, (name, rel_le)
        if name == "ctl":
            # empty mask: with-limb bitwise == no-limb (the ctl field
            # is bit-copied dry everywhere, so the branch never runs).
            assert (cp.asnumpy(n2sm_dev).tobytes()
                    == cp.asnumpy(n2s_dev).tobytes())
            kv_a, le_a = launch_vertical_channel(
                e_s, sdev["theta"], f=0.0, n2=n2sm_dev, dz_col=thick64)
            assert (cp.asnumpy(kv_d).tobytes()
                    == cp.asnumpy(kv_a).tobytes())
            assert (cp.asnumpy(leps_d).tobytes()
                    == cp.asnumpy(le_a).tobytes())
    # ---- launcher validation -----------------------------------------
    with pytest.raises(ValueError, match="n2_dry requires n2"):
        launch_vertical_channel(e_s, sdev["theta"], f=0.0,
                                n2_dry=n2s_dev, dz_col=thick64)
    bad = cp.zeros((2, 2, 2), dtype=cp.float32)
    with pytest.raises(ValueError, match="n2_dry"):
        launch_vertical_channel(e_s, sdev["theta"], f=0.0, n2=n2sm_dev,
                                n2_dry=bad, dz_col=thick64)


@requires_gpu
def test_m1b_uniform_stratification_analytic_device():
    """QUADRATURE EXACTNESS ON DEVICE (the CPU test_m1b_deck_under_lid
    analytic leg's kernel twin): on a uniform N^2_eff = 1e-4 column at
    e = 0.005 the excursion integral is exactly quadratic, so
    l_m = sqrt(2e/N^2) = 10 m (independent anchor -- the parity gates
    compare against the authority, this one against algebra).  With
    constant theta the dry BL89 pair never crosses (free fallbacks
    htop - z and z) and l_B >= 25 m from the second center up, so the
    stored leps in every substituted cell k >= 1 must be the analytic
    l_m through the limb's min -- while at k = 0 the Blackadar floor
    (9.375 m < 10 m) keeps the composition and the limb is inert
    bitwise.  n2_dry differs everywhere (all cells substituted); the
    l_s branch rides n2_eff and binds only kv (leps is l_s-free), so
    the anchor is clean.  Non-vacuity: the no-limb leps is >= 25 m in
    the asserted cells."""
    import cupy as cp
    from woof.core.sase import launch_vertical_channel
    from woof.verify import sase_ref

    nz, dz = 20, 50.0
    shape = (nz, 4, 4)
    e32, n232 = np.float32(0.005), np.float32(1.0e-4)
    theta_dev = cp.full(shape, np.float32(300.0), dtype=cp.float32)
    e_dev = cp.full(shape, e32, dtype=cp.float32)
    n2e_dev = cp.full(shape, n232, dtype=cp.float32)
    n2d_dev = cp.full(shape, np.float32(5.0e-5), dtype=cp.float32)
    kv_w, le_w = launch_vertical_channel(e_dev, theta_dev, f=0.0,
                                         n2=n2e_dev, n2_dry=n2d_dev,
                                         dz=dz)
    kv_n, le_n = launch_vertical_channel(e_dev, theta_dev, f=0.0,
                                         n2=n2e_dev, dz=dz)
    lex = float(np.sqrt(2.0 * np.float64(e32) / np.float64(n232)))
    le_w_h = cp.asnumpy(le_w).astype(np.float64)
    le_n_h = cp.asnumpy(le_n).astype(np.float64)
    assert np.all(le_n_h[1:] >= 20.0)          # non-vacuous: limb bites
    err = np.abs(le_w_h[1:] - lex).max()
    print(f"uniform-stratification device anchor: l_m={lex:.6f} "
          f"max|leps - l_m|={err:.3e}")
    assert err <= 1e-5 * lex, err
    # k = 0: the Blackadar floor undercuts l_m -- limb inert bitwise.
    assert (cp.asnumpy(le_w)[0].tobytes()
            == cp.asnumpy(le_n)[0].tobytes())


def _run_deck_column_device(steps=120, dt=60.0, ust=0.3):
    """S4-3c device twin of the CPU deck-under-lid engine
    ``test_sase._run_deck_column`` on the FP32-cast column
    (``_deck_cast32``, shared bit-for-bit with the CPU-cast reference
    ``_run_deck_column_cpu_cast`` below): frozen-state equilibrium --
    n2/n2_eff once from the frozen thermo, the driver pre-step
    surface-e deposit at ust = 0.3, the exported-kv replica
    ``launch_vertical_channel(f=0, n2_dry=...)`` (the S4-3b-amended
    composition), and ``launch_sase_step`` with the n2_moist seam (the
    limb rides inside, S4-3c) each step; winds and thermo frozen, only
    e integrates.  Horizontal uniformity degenerates the device solve
    to exactly (0.0, 0.0), so f_used = 0 (asserted) -- the pure RANS
    limb where the G-M3 defect lives.  Returns the CPU engine's dict
    shape plus the frozen device state for the caller's limb-binding
    witness."""
    import cupy as cp
    from woof.core.sase import (launch_moist_n2, launch_n2,
                                 launch_sase_step,
                                 launch_vertical_channel)
    from woof.verify import sase_ref
    dev, cast, thick, zc, deck = _deck_device()
    nz = len(thick)
    delta = 3000.0
    e_dev = cp.full((nz, 4, 4), np.float32(sase_ref.E_MIN),
                    dtype=cp.float32)
    src = ust ** 3 / (sase_ref.KARMAN * 0.5 * thick[0])
    e_src = np.float32(dt * src)
    pr = float(sase_ref.prandtl_blend(0.0))
    n2_dev = launch_n2(dev["theta"], dz_col=thick)
    n2m_dev = launch_moist_n2(dev["theta"], dev["qv"], dev["qc"],
                              dev["p"], n2_dev, dz_col=thick)
    mask = (cp.asnumpy(n2m_dev) != cp.asnumpy(n2_dev))[:, 0, 0]
    samples = {}
    for n in range(steps):
        e_dev[0] += e_src                      # driver pre-step deposit
        kv_dev, _ = launch_vertical_channel(e_dev, dev["theta"], f=0.0,
                                            n2=n2m_dev, n2_dry=n2_dev,
                                            dz_col=thick)
        u, v, w = (cp.zeros((nz, 4, 4), dtype=cp.float32)
                   for _ in range(3))
        ledger = launch_sase_step(
            u, v, w, dev["theta"], e_dev, dx=delta, dy=delta, dz=200.0,
            delta=delta, dt=dt, n2=n2_dev, dz_col=thick,
            n2_moist=n2m_dev)
        assert (ledger["c_nu"], ledger["f"]) == (0.0, 0.0)  # RANS limb
        minute = (n + 1) * dt / 60.0
        if minute in (90.0, 120.0):
            e_h = cp.asnumpy(e_dev).astype(np.float64)
            kh_h = cp.asnumpy(kv_dev).astype(np.float64) / pr
            samples[minute] = {"e": e_h[:, 0, 0].copy(),
                               "kh": kh_h[:, 0, 0].copy()}
    return {"z": zc, "mask": mask, "samples": samples,
            "state": (dev, e_dev, n2_dev, n2m_dev, thick)}


def _run_deck_column_cpu_cast(steps=120, dt=60.0, ust=0.3):
    """The FP64 authority replica of ``test_sase._run_deck_column`` on
    the SAME FP32 casts the device engine consumes (promoted -- the
    module's parity semantics), sample keys matching the device twin.
    The stepping loop is the S4-3b CPU engine's verbatim: authority
    n2/moist_n2 once, pre-step deposit, the n2_dry-amended
    ``bl89_rans_lengths`` exported-kv replica, ``sase_split_step`` with
    the n2_moist seam."""
    from woof.verify import sase_ref
    cast, thick, zc, deck = _deck_cast32()
    nz = len(thick)

    def col(a):
        return a.astype(np.float64).reshape(-1, 1, 1)

    theta, qv, qc, p = (col(cast[k]) for k in ("theta", "qv", "qc", "p"))
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
                               "kh": (kv / pr)[:, 0, 0].copy()}
    return {"z": zc, "mask": (n2m != n2)[:, 0, 0], "samples": samples}


@requires_gpu
def test_m1b_deck_under_lid_device():
    """THE G-M3 FIX ON DEVICE (S4-3c; the essential behavior of the
    task brief): on the deck-under-lid column the CPU authority pinned
    the lid-adjacent equilibrium K_h transition 101.9 (pre-M1b free
    fallback, the smoke's measured 1e2 deck class [74, 150]) -> 39.3
    (limb live, reference class) -- test_m1b_deck_under_lid.  The
    device engine must reproduce the GREEN side of that transition:

    * substitution mask exactly the saturated deck k4..k13, equal
      between the engines;
    * lid-adjacent equilibrium K_h(k12) in the CPU GREEN band
      [25, 55] m2/s at BOTH samples -- the reference class, disjoint
      from the CPU-pinned RED band [74, 150] (the transition
      reproduced: the device limb kills lid entrainment);
    * deck TKE mean in the CPU GREEN band [0.45, 0.75] (the limb does
      not kill deck turbulence);
    * LIMB-BINDING WITNESS at the frozen equilibrium state (non-
      vacuous on device): the vertical channel WITH n2_dry undercuts
      the no-limb call at k12 in kv AND leps, every column;
    * DEVICE-VS-AUTHORITY PARITY on the shared FP32 casts: deck
      layer-mean TKE and K_h and the pointwise K_h(k12) vs the FP64
      CPU-cast replica at <= 5e-7 (layer means) / <= 2e-6 (pointwise)
      -- the S4-2 ratified frozen-equilibrium tiers: the equilibrium
      is a fixed-point attractor of the bit-identical frozen inputs,
      so the FP32 trajectory converges to the FP64 equilibrium at
      FP32 resolution (measured S4-2 worst 1.261e-7 layer-mean on
      this card; pointwise singles carry no averaging, hence the
      established 2e-6 unit tier there).
    """
    import cupy as cp
    from woof.core.sase import launch_vertical_channel

    dev = _run_deck_column_device()
    ref = _run_deck_column_cpu_cast()
    np.testing.assert_array_equal(dev["mask"], ref["mask"])
    assert list(np.nonzero(dev["mask"])[0]) == list(range(4, 14))
    mask = dev["mask"]
    k12 = 12
    worst_mean, worst_pt = 0.0, 0.0
    for minute in (90.0, 120.0):
        s, r = dev["samples"][minute], ref["samples"][minute]
        kh12 = float(s["kh"][k12])
        tke = float(s["e"][mask].mean())
        assert 25.0 <= kh12 <= 55.0, (minute, kh12)  # reference class
        assert 0.45 <= tke <= 0.75, (minute, tke)    # deck stays alive
        rel_pt = abs(kh12 - float(r["kh"][k12])) / float(r["kh"][k12])
        rel_e = (abs(tke - float(r["e"][mask].mean()))
                 / float(r["e"][mask].mean()))
        rel_k = (abs(float(s["kh"][mask].mean())
                     - float(r["kh"][mask].mean()))
                 / float(r["kh"][mask].mean()))
        worst_mean = max(worst_mean, rel_e, rel_k)
        worst_pt = max(worst_pt, rel_pt)
        print(f"deck GREEN {minute:.0f} min: device K_h(k12)={kh12:.3f} "
              f"(CPU {float(r['kh'][k12]):.3f}), deck TKE={tke:.4f}; "
              f"rel k12={rel_pt:.3e} means e={rel_e:.3e} kh={rel_k:.3e}")
    assert worst_mean <= 5e-7, f"deck layer-mean parity {worst_mean:.3e}"
    assert worst_pt <= 2e-6, f"deck K_h(k12) parity {worst_pt:.3e}"
    # limb-binding witness at the equilibrium state (every column).
    tiles, e_dev, n2_dev, n2m_dev, thick = dev["state"]
    kv_w, le_w = launch_vertical_channel(e_dev, tiles["theta"], f=0.0,
                                         n2=n2m_dev, n2_dry=n2_dev,
                                         dz_col=thick)
    kv_n, le_n = launch_vertical_channel(e_dev, tiles["theta"], f=0.0,
                                         n2=n2m_dev, dz_col=thick)
    assert bool((cp.asnumpy(kv_w)[k12] < cp.asnumpy(kv_n)[k12]).all())
    assert bool((cp.asnumpy(le_w)[k12] < cp.asnumpy(le_n)[k12]).all())


@requires_gpu
def test_m1b_vertical_channel_increment_d02_shape():
    """KERNEL-TIME BUDGET, M1b INCREMENT (C12; the S4-2 report idiom):
    the limb's cost at the exact production d02 shape (49, 501, 501),
    measured as launch_vertical_channel WITH n2_dry minus WITHOUT on a
    field where EVERY column carries an 8-level substituted
    moist-unstable deck under a stable lid (worst case: production
    decks cover a fraction of columns, and unsubstituted cells skip
    the O(nz^2) excursion sweep entirely -- the mask gate).  Reported
    against the yolo-d pre-M1 d02 seam baseline 62.12 ms/call
    (out/sase-yolo-d/run-metrics.json seam_ms_mean, 2026-07-22); the
    budget verdict itself is re-scored on the S4-3d smoke receipt
    (adjudicated -- the S4-2 report's carried flag).
    BINDING PIN, per card (``_M1B_INCREMENT_D02_PIN_MS``): increment
    <= 6.0 ms/call at this shape on the RTX 5090 (regression guard at
    ~2x the measured 2.965/2.957 ms dual-run pair on the RTX 5090,
    2026-07-22; a worst-case-mask increment of 4.8% of the pre-M1 seam
    -- the in-production increment is smaller by the saturated-column
    fraction), and about 2x its own idle baseline on every other
    calibrated card; each side of the difference is the least of three
    passes."""
    import cupy as cp
    from woof.core.sase import launch_vertical_channel

    shape = (49, 501, 501)
    zp = np.cumsum(np.full(49, 300.0)) - 150.0
    th_p = cp.ascontiguousarray(cp.asarray(np.broadcast_to(
        (300.0 + 0.004 * zp).astype(np.float32)[:, None, None], shape)))
    e_p = cp.full(shape, np.float32(0.5), dtype=cp.float32)
    n2d_p = cp.full(shape, np.float32(1.0e-4), dtype=cp.float32)
    n2e_p = n2d_p.copy()
    n2e_p[8:16] = cp.float32(-1.0e-4)          # substituted deck, all
    n2e_p = cp.ascontiguousarray(n2e_p)        # columns (worst case)
    dz_p = cp.full(shape, 300.0, dtype=cp.float32)

    t_base = _least_pass_ms(lambda: launch_vertical_channel(
        e_p, th_p, f=0.0, n2=n2e_p, dz_col=dz_p), 10)
    t_limb = _least_pass_ms(lambda: launch_vertical_channel(
        e_p, th_p, f=0.0, n2=n2e_p, n2_dry=n2d_p, dz_col=dz_p), 10)
    inc = t_limb - t_base
    print(f"d02-shape M1b increment: channel {t_base:.3f} -> "
          f"{t_limb:.3f} ms/call, increment {inc:.3f} ms = "
          f"{100 * inc / 62.12:.2f}% of the pre-M1 d02 seam "
          f"(worst-case every-column mask)")
    _assert_card_pin(_M1B_INCREMENT_D02_PIN_MS, inc, (
        "M1b vertical-channel increment regressed to {ms:.3f} ms at the "
        "d02 shape on {card} (pin {pin} ms = about 2x that card's idle "
        "baseline; table above the test)"))


# ---------------------------------------------------------------------------
# S4-5: SASE-M2 conditional venting limb -- device mirror
# (sase_plume_vent_flux) and the driver deposit seam
# (sase_vent_deposit_scale / sase_vent_deposit) against the S4-4
# authority (sase_ref.plume_vent_flux at 1e3fd3b, sase_ref
# .vent_deposit_rescale).  Every fixture column of the CPU corpus is
# mirrored here, and every GPU number is DUAL-RUN (this card has no ECC:
# a measurement that is not bit-identical across two launches does not
# count).
#
# THE INDEX GATE.  Flux tolerance is NOT sufficient for this limb: a
# one-cell move in any selected level changes the flux by a median 35%
# on the clean real-field population and 47-53% on the strictly
# root-invariant subset (design doc SASE-M2 amendment, "root / anchor
# separation"; S4-4 report).  Index agreement is therefore a
# SEPARATE pass/fail gate, asserted BIT-EXACTLY on all seven diagnosed
# levels (k_base, k_top, k_r, k_lid, LFC, NB, buoyancy peak) against a
# reference transcribed INDEPENDENTLY of both engines.
# ---------------------------------------------------------------------------


def _vent_cast32(args):
    """The CPU fixture args with every field FP32-cast then promoted --
    the established parity semantics (the device reads FP32, so the
    authority reference must read the same bits)."""
    out = dict(args)
    for key in ("theta", "qv", "qc", "p", "e_sgs"):
        out[key] = np.asarray(args[key], np.float64).astype(
            np.float32).astype(np.float64)
    out["dz_col"] = np.asarray(args["dz_col"], np.float64).astype(
        np.float32).astype(np.float64)
    out["rho1"] = np.float64(np.float32(args["rho1"]))
    return out


def _vent_qs(args):
    """Saturation mixing ratio on the fixture's own (theta, p), from the
    authority's Tetens/Exner constants (the fixtures' ``_vent_theta_es``
    chain)."""
    from woof.verify import sase_ref as sr
    th = np.asarray(args["theta"], np.float64)
    p = np.asarray(args["p"], np.float64)
    t = th * (p / sr.P0_REF) ** (sr.RD_AIR / sr.CP_AIR)
    es = 1000.0 * sr.SVP1 * np.exp(sr.SVP2 * (t - sr.SVPT0)
                                   / (t - sr.SVP3))
    return sr.EP2_RV * es / (p - es)


def _vent_device(args, ny=3, nx=5):
    """(device kwargs, FP32-cast authority args, FP32-cast thicknesses)
    for a single-column M2 fixture broadcast to a (nz, ny, nx) tile.

    The M1 substitution mask reaches the kernel the way every other
    M-limb consumes it -- as the BITWISE departure n2_moist != n2_dry of
    two fields the driver already holds -- so the fixture builds a
    synthetic (0.0, 1.0) pair whose departure IS the fixture's own
    n2m_mask.  That is exact: the driver's own pair is produced by the
    same kind of bit comparison (physics.py _run_sase, launch_moist_n2).
    """
    import cupy as cp
    nz = np.asarray(args["theta"]).shape[0]

    def tile(a):
        a = np.asarray(a, np.float64).reshape(nz, 1, 1)
        return cp.asarray(np.ascontiguousarray(np.broadcast_to(
            a.astype(np.float32), (nz, ny, nx))))

    mask = np.ascontiguousarray(np.broadcast_to(
        np.asarray(args["n2m_mask"]).reshape(nz, 1, 1), (nz, ny, nx)))
    ref = _vent_cast32(args)                    # mask kept at (nz, 1, 1)
    dev = dict(
        theta=tile(args["theta"]), qv=tile(args["qv"]),
        qc=tile(args["qc"]), pressure=tile(args["p"]),
        e_sgs=tile(args["e_sgs"]),
        n2_moist=cp.asarray(np.where(mask, np.float32(1.0),
                                     np.float32(0.0)).astype(np.float32)),
        n2_dry=cp.zeros((nz, ny, nx), dtype=cp.float32),
        rho1=cp.asarray(np.full((ny, nx), np.float32(args["rho1"]))))
    return dev, ref, np.asarray(ref["dz_col"], np.float64)


def _vent_launch(dev, thick, f_blend=0.0, indices=True):
    """DUAL-RUN launch (the no-ECC policy): two independent launches,
    asserted bit-identical, returning the first."""
    import cupy as cp
    from woof.core.sase import launch_plume_vent_flux
    runs = []
    for _ in range(2):
        got = launch_plume_vent_flux(
            dev["theta"], dev["qv"], dev["qc"], dev["pressure"],
            dev["e_sgs"], dev["n2_moist"], dev["n2_dry"], dev["rho1"],
            f_blend=f_blend, dz_col=thick, indices=indices)
        runs.append([cp.asnumpy(a) for a in got])
    for a, b in zip(*runs):
        assert a.tobytes() == b.tobytes(), (
            "dual-run mismatch: the two launches are not bit-identical")
    return runs[0]


def _vent_termination_ref(args, k_base, k_top, k_r):
    """(k_lid, k_lfc, k_nb, kb) of the closure's step-3 ascent and
    step-4 LFC/NB/peak searches, transcribed INDEPENDENTLY of
    plume_vent_flux from the registered rules (module docstring,
    SASE-M2 ASCENT / TERMINATION / INVERSION BASE): a SCALAR sequential
    column sweep against the authority's vectorized where()-masked one.
    ``(-1, -1, -1, -1)`` when the column stands down for want of a
    neutral-buoyancy level.  Single-column args only."""
    from woof.verify import sase_ref as sr
    th = np.asarray(args["theta"], np.float64)[:, 0, 0]
    qv = np.asarray(args["qv"], np.float64)[:, 0, 0]
    qc = np.asarray(args["qc"], np.float64)[:, 0, 0]
    p = np.asarray(args["p"], np.float64)[:, 0, 0]
    thick = np.broadcast_to(np.asarray(args["dz_col"], np.float64),
                            th.shape).copy()
    nz = th.size
    z = np.cumsum(thick) - 0.5 * thick
    t_env = th * (p / sr.P0_REF) ** (sr.RD_AIR / sr.CP_AIR)
    qt_env = qv + qc
    thl_env = th - (sr.XLV / sr.CP_AIR) * qc * th / t_env
    rvm1 = sr.RV_AIR / sr.RD_AIR - 1.0
    b = np.zeros(nz)
    thl_p = qt_p = 0.0
    started = False
    for k in range(nz):
        if k > 0 and started:
            thl_f = 0.5 * (thl_env[k - 1] + thl_env[k])
            qt_f = 0.5 * (qt_env[k - 1] + qt_env[k])
            fac = (z[k - 1] / z[k]) ** sr.VENT_ENT_COEF
            thl_n = thl_f + (thl_p - thl_f) * fac
            qt_n = qt_f + (qt_p - qt_f) * fac
            th_p, qv_p, qc_p = sr._vent_saturation_adjust(thl_n, qt_n,
                                                          p[k])
            tv_p = float(th_p) * (1.0 + rvm1 * float(qv_p) - float(qc_p))
            tv_e = th[k] * (1.0 + rvm1 * qv[k] - qc[k])
            b[k] = sr.G_ACCEL * (tv_p - tv_e) / tv_e
            thl_p, qt_p = float(thl_n), float(qt_n)
        if k == k_r:
            thl_p, qt_p = thl_env[k], qt_env[k]
            started = True
    k_lid = k_top + 2
    zr = z[k_r]
    lfc = nb = False
    k_nb = kb = k_lfc = -1
    bmax = 0.0
    for k in range(nz):
        above = k > k_r
        incap = (above and k < k_lid
                 and (z[k] - zr) <= sr.VENT_DEPTH_CAP)
        pos = b[k] > 0.0
        if incap and not lfc and pos:
            k_lfc, lfc = k, True
        if incap and lfc and not nb and pos and b[k] > bmax:
            bmax, kb = b[k], k
        if incap and lfc and not nb and not pos:
            k_nb, nb = k, True
        if (above and k == k_lid and lfc and not nb
                and (z[k] - zr) <= sr.VENT_DEPTH_CAP):
            k_nb, nb = k, True
    # k_nb == -1 IS the stand-down signal; k_lfc is reported either way
    # so a caller can name WHICH termination branch fired.
    return k_lid, k_lfc, k_nb, kb


def _vent_indices_ref(args):
    """The seven diagnosed indices (k_base, k_top, k_r, k_lid, k_lfc,
    k_nb, kb), all ``-1`` on a stood-down column.

    k_base/k_top come from ``test_sase._vent_layer_indices`` and k_r
    from ``test_sase._vent_root_index`` -- the CPU corpus's own
    independent transcriptions, VALIDATED 38/38 against an instrumented
    authority build (S4-4) -- and the termination triple from
    :func:`_vent_termination_ref` here.  None of the three shares code
    with ``plume_vent_flux``.
    """
    from test_sase import _vent_layer_indices, _vent_root_index
    k_base, k_top = _vent_layer_indices(args)
    if k_base < 0:
        return (-1,) * 7
    # S4-5c, VENT_ANCHOR_RULE: a run based in the LOWEST MODEL LEVEL
    # stands the limb down at step 1a, before any index is diagnosed --
    # so the kernel leaves every idx row at its -1 initializer, exactly
    # as it does for the three older stand-downs.
    if k_base == 0:
        return (-1,) * 7
    k_r = _vent_root_index(args, k_base, k_top)
    k_lid, k_lfc, k_nb, kb = _vent_termination_ref(args, k_base, k_top,
                                                   k_r)
    if k_nb < 0:
        return (-1,) * 7
    return (k_base, k_top, k_r, k_lid, k_lfc, k_nb, kb)


def _vent_parity_columns():
    """The S4-5 GPU parity set: the 11Z specimen pair, the prescribed-lid
    deck family, the three grid-consistency grids, the taper-branch
    column, both REAL d02 columns (the only registered columns with
    genuinely non-uniform e_sgs), and the two S4-5b REAL d02
    BINDING-RULE columns on which the round-5 root depth floor actually
    clamps the root, and the S4-5c REAL d02 SURFACE-BASED column, whose
    saturated run is based in the lowest model level (the fourth
    registered stand-down, VENT_ANCHOR_RULE -- a branch no other
    registered column reaches in either engine), and the six S4-5d REAL
    d02 REGISTERED-RULE columns, each of which is the population's own
    representative of one M2 rule that could be deleted or shifted one
    cell with both engines' gates green."""
    from test_sase import (_vent_args, _vent_binding_args,
                           _vent_deck_args, _vent_real_args,
                           _vent_rules_args, _vent_surface_args,
                           _vent_taper_args)
    cols = [("specimen-amp", _vent_args("amp")),
            ("specimen-ctl", _vent_args("ctl")),
            ("deck-1200", _vent_deck_args(z_lid=1200.0)[0]),
            ("deck-1400", _vent_deck_args(z_lid=1400.0)[0]),
            ("deck-1600", _vent_deck_args(z_lid=1600.0)[0]),
            ("deck-weak-lid",
             _vent_deck_args(z_lid=1400.0, dth_inv=2.0)[0])]
    for nz, dz in ((12, 280.0), (30, 100.0), (120, 25.0)):
        cols.append((f"grid-dz{int(dz)}",
                     _vent_deck_args(z_lid=1400.0, dth_inv=2.0, nz=nz,
                                     dz=dz)[0]))
    cols.append(("taper-branch", _vent_taper_args()))
    for name in ("WIN", "KNIFE"):
        cols.append((f"real-{name}", _vent_real_args(name)))
    for name in ("CLAMP", "DROP"):
        cols.append((f"binding-{name}", _vent_binding_args(name)))
    cols.append(("surface-based", _vent_surface_args()))
    for name in ("CAPSEARCH", "CAPLID", "KLID", "RH100", "LOWRUN",
                 "ROOTBND"):
        cols.append((f"rule-{name}", _vent_rules_args(name)))
    return cols


@requires_gpu
def test_m2_plume_vent_device_parity_and_index_agreement():
    """THE M2 UNIT MIRROR GATE (S4-5): ``launch_plume_vent_flux`` vs the
    FP64 authority ``plume_vent_flux`` on the WHOLE registered fixture
    family, with the INDEX gate stated and asserted SEPARATELY from the
    flux gate.

    Columns (21): the yolo-d 11Z specimen amplifier and its control; the
    prescribed-lid decks at z_lid = 1200/1400/1600 m and the weak-lid
    (dth_inv = 2 K) column; the three grid-consistency discretizations
    of the weak-lid column (dz = 280/100/25 m, an 11x resolution span);
    the taper-branch column (the ONLY registered column that exercises
    the natural-NB / buoyancy-peak / remaining-buoyancy-taper searches
    -- a build with those searches deleted agreed with 16 of 17 CPU
    fixtures bitwise, so its absence would leave a live physics branch
    unmirrored); BOTH real d02 columns of vent_columns_yolod_11z,
    which are the only registered columns carrying genuinely
    non-uniform e_sgs and whose WIN member has its cloud base six cells
    above its root -- the construction that separates the
    root-anchored ebar window from the cloud-base-anchored one; and the
    two S4-5b BINDING-RULE columns of vent_columns_yolod_binding.

    THE BINDING-RULE PAIR (S4-5b Item 1) is what makes GATE 1 detect a
    DELETED RULE rather than only a mis-transcribed one.  On all 12 of
    the columns above the round-5 root DEPTH FLOOR
    ``k_r >= k_base - (k_top - k_base) - 1`` never binds -- 8 of them
    have no interior theta_es maximum at or below k_base at all and the
    3 that do sit above the floor -- so a device build with the floor
    term dropped agreed with every one of them and passed all four M2
    gates.  CLAMP (real d02, j=180 i=78: run k20..k21, floor 18,
    eligible theta_es peak at k15) and DROP (j=271 i=44: run k18..k22,
    floor 13, eligible peak at k11) are two of the 33.6% of 11Z
    step-1-chosen columns on which the floor moves the root, and on
    both of them it clamps the root to the run's own base.  Their
    reference k_r therefore DIFFERS from the pre-amendment rule's, so
    GATE 1 fails on a device build without the floor (measured on the
    mutated kernel, S4-5b report).  Both carry design-point turbulence
    (SURVEY_E = 1.0 m2/s2, the G-M5 band) per the registered survey
    standard, not the defective frames' own TKE.

    THE SURFACE-BASED COLUMN (S4-5c; real d02 11Z, j=13 i=205) is the
    corpus's only member whose saturated run is based in the LOWEST
    MODEL LEVEL, the case the amendment "a surface-based saturated layer
    stands the limb down" rules on (VENT_ANCHOR_RULE).  It is a
    STAND-DOWN column here -- all seven indices -1 and all three rows
    bitwise +0.0 -- and it is the only registered column that reaches
    the kernel's step-1a return at all, so without it a device build
    that dropped that return would agree with all 14 others.  Its own
    non-vacuity (pre-amendment it fired, and none of the three older
    stand-downs applies to it) is established on the authority side by
    ``test_sase.test_m2_surface_based_layer_stands_down``; carrying it
    here binds the DEVICE to the same ruling.  Design-point turbulence
    (SURVEY_E = 1.0 m2/s2) per the registered survey standard.

    THE SIX REGISTERED-RULE COLUMNS (S4-5d;
    :mod:`vent_columns_yolod_rules`) extend the same argument to five
    more M2 rules the audit wave showed both engines' gates could not
    see.  Each was found by scanning every column of the 11Z / 13Z
    frames for the place the rule actually decides, and each is carried
    here so the DEVICE is bound to the same decision the authority
    makes.  Two are STAND-DOWN columns whose seven indices are all -1
    -- rule-CAPSEARCH (13Z j=242 i=104) reaches the kernel's
    ``!nb_found`` return through the ``depth_cap`` term inside the
    search, and rule-CAPLID (13Z j=81 i=102) reaches it through the
    ``depth_cap`` term of the AT-LID forced-NB test, so the two pin the
    two terms separately (deleting either one alone fires exactly one of
    them; measured on the authority this session).  Four fire and pin an
    index: rule-KLID (11Z j=70 i=251, k_nb = k_lid = 17 -- termination
    FORCED at the C9 boundary, the class where the ``k < k_lid`` search
    bound is essential), rule-RH100 (11Z j=22 i=196, whose member run
    top k16 is saturated only through the ``qv >= q_s`` limb of
    MOIST_STABILITY_SWITCH, so a device build without that limb diagnoses
    NO run at all here), rule-LOWRUN (11Z j=17 i=164, the corpus's only
    column with TWO qualifying member runs -- k8..k11 and k13..k15 -- so
    it is the only one that can tell the kernel's ``!chosen`` latch from
    its absence) and rule-ROOTBND (11Z j=21 i=39, whose eligible
    theta_es maxima at k7 AND k11 = k_base make the kernel's
    ``k <= k_base`` bound the only rule that picks between them: k_r = 11
    on this build, k_r = 7 with the bound one cell tighter).  All six
    carry design-point turbulence (SURVEY_E = 1.0 m2/s2).  Their
    authority-side non-vacuity is established by the six S4-5d tests in
    ``test_sase.py``; the GPU suite was not run in the session that
    added them (CPU-only), so the first device execution is the user's.

    GATE 1, INDEX AGREEMENT (pass/fail, bit-exact, no tolerance).  All
    seven diagnosed levels -- k_base, k_top, k_r, k_lid, the LFC, the
    neutral-buoyancy level and the buoyancy peak -- must equal the
    reference EXACTLY on every column.  A one-cell move in any of them
    is a median 35% flux change on the clean real-field population and
    47-53% on the strictly root-invariant subset (design doc SASE-M2
    amendment; S4-4 report), so a 2e-6 flux agreement on a
    column whose indices differ is not agreement at all.  The reference
    is transcribed independently of both engines (see
    :func:`_vent_indices_ref`).

    GATE 2, FLUX PARITY: <= 2e-6 scale-relative on all three rows, the
    established tier.  The kernel is FP64 end to end in the authority's
    exact op order with one FP32 rounding at the face store, so the
    measured value is the FP32-store class, orders better.

    Structural pins carried on every column: the end faces F[0] and
    F[nz] are bitwise +0.0 (never -0.0, never computed-then-zeroed --
    the interface contract), and the device flux SUPPORT (the set of
    faces carrying a nonzero value) equals the authority's exactly,
    which ties the reported indices back to the authority's OWN returned
    bytes rather than only to the transcription.

    DUAL-RUN on every launch (no ECC on this card): each column is
    launched twice and the two results asserted bit-identical.
    """
    import cupy as cp
    from woof.verify import sase_ref
    worst_rel = 0.0
    rows = []
    for name, args in _vent_parity_columns():
        dev, ref, thick = _vent_device(args)
        f_th, f_qv, f_qc, idx = _vent_launch(dev, thick)
        got_idx = tuple(int(v) for v in idx[:, 0, 0])
        want_idx = _vent_indices_ref(ref)
        assert got_idx == want_idx, (
            f"[{name}] INDEX GATE: device "
            f"(k_base,k_top,k_r,k_lid,LFC,NB,peak)={got_idx} vs "
            f"reference {want_idx}")
        # the index field is uniform across the broadcast tile
        assert np.all(idx == idx[:, :1, :1])
        a_th, a_qv, a_qc = sase_ref.plume_vent_flux(**ref)
        col_rel = 0.0
        for lbl, got, aut in (("theta", f_th, a_th), ("qv", f_qv, a_qv),
                              ("qc", f_qc, a_qc)):
            zero = np.zeros((1,) + got.shape[1:]).astype(
                np.float32).tobytes()
            assert got[0].tobytes() == zero, (name, lbl, "F[0]")
            assert got[-1].tobytes() == zero, (name, lbl, "F[top]")
            if np.max(np.abs(aut)) == 0.0:
                assert got.tobytes() == np.zeros_like(got).tobytes()
                continue
            rel, _ = _max_rel(got, aut)
            col_rel = max(col_rel, rel)
            assert rel <= 2e-6, (f"[{name}] F_{lbl} parity {rel:.3e} "
                                 f"(gate 2e-6)")
            sup_dev = tuple(np.nonzero(got[:, 0, 0] != 0.0)[0])
            sup_ref = tuple(np.nonzero(aut[:, 0, 0] != 0.0)[0])
            assert sup_dev == sup_ref, (name, lbl, sup_dev, sup_ref)
        worst_rel = max(worst_rel, col_rel)
        rows.append((name, got_idx, col_rel))
    print("M2 device parity (dual-run, bit-identical):")
    for name, ix, rel in rows:
        print(f"  {name:16s} idx(kb,ktop,kr,klid,lfc,nb,pk)={ix} "
              f"max_rel={rel:.3e}")
    print(f"  worst scale-relative flux error over {len(rows)} "
          f"columns: {worst_rel:.3e} (gate 2e-6); index agreement "
          f"{len(rows)}/{len(rows)} exact")
    # THE C9 CAP MARGIN, READ FROM THE ENGINES (S4-5b Item 4a).  The
    # previous form of this block was `margin = 1700.0 - (1400.0 + dz);
    # assert margin > 0.0` -- three hardcoded literals with `nz`
    # unused, evaluating 20>0, 200>0 and 275>0, so it could not fail
    # for any change to the code under test while being presented as
    # asserting the margin on every tested grid.  What the grid family
    # actually has to show is that the C9 hard-zero boundary -- the
    # entrainment-zone cell's own TOP face, i.e. the DEVICE's diagnosed
    # k_lid converted to a height on that grid's own thicknesses --
    # stays below the fixture's inversion top on every discretization,
    # and that every row is bitwise +0.0 from that face up (cap
    # preservation by construction).  The inversion top comes from
    # _vent_deck_invtop, the same function the fixture builds the
    # column with, so the two sides of the margin can never drift into
    # being two independent literals again.
    from test_sase import _vent_deck_args, _vent_deck_invtop
    z_invtop = _vent_deck_invtop(1400.0)
    margins = []
    for nz_g, dz_g in ((12, 280.0), (30, 100.0), (120, 25.0)):
        args_g = _vent_deck_args(z_lid=1400.0, dth_inv=2.0, nz=nz_g,
                                 dz=dz_g)[0]
        dev_g, _, thick_g = _vent_device(args_g)
        got_g = _vent_launch(dev_g, thick_g)
        k_lid = int(got_g[3][3, 0, 0])
        assert k_lid > 0, (nz_g, dz_g, "the grid column stood down -- "
                                       "this leg would be vacuous")
        tcol = np.broadcast_to(np.asarray(thick_g, np.float64),
                               (nz_g,))
        ceiling = float(np.concatenate([[0.0], np.cumsum(tcol)])[k_lid])
        margin = z_invtop - ceiling
        assert margin > 0.0, (nz_g, dz_g, k_lid, ceiling, margin)
        for lbl, arr in (("theta", got_g[0]), ("qv", got_g[1]),
                         ("qc", got_g[2])):
            tail = np.zeros_like(arr[k_lid:])
            assert arr[k_lid:].tobytes() == tail.tobytes(), (
                dz_g, lbl, "C9: the cap proper is not bitwise +0.0")
        margins.append((dz_g, k_lid, ceiling, margin))
    print("  C9 cap margin on the deck grid family (device k_lid on "
          f"each grid's own thicknesses, inversion top {z_invtop:g} m): "
          + ", ".join(f"dz={d:g} m -> k_lid={k}, ceiling {c:.1f} m, "
                      f"margin {m:.1f} m" for d, k, c, m in margins))


@requires_gpu
def test_m2_plume_vent_device_stand_down_and_identity_bitwise():
    """THE M2 IDENTITY / STAND-DOWN CONTRACTS ON DEVICE (S4-5).  Every
    leg asserts BITWISE +0.0 (``tobytes`` against a zero buffer), never
    a small number and never -0.0:

    * f_blend = 1 (the LES limit): the FP-exact two-product blend
      M_used = (1 - f)*M_base is +0.0, the face gate closes, every row
      is bitwise zero -- on the specimen amplifier, i.e. a column that
      fires hard at f = 0;
    * MASK-OFF: an all-false M1 mask (n2_moist bitwise equal to n2_dry)
      stands the limb down completely.  This is the M1 seam's veto and
      the pre-M2 identity: a driver whose M1 seam never fires deposits
      literally nothing;
    * NO QUALIFYING RUN: the control column (unsaturated) stands down at
      step 1, with all seven indices reported -1;
    * NO LFC BELOW k_lid: the deck column with its member run truncated
      to the bottom two cells (mask vetoed above), so k_lid sits below
      the parcel's own level of free convection;
    * k_lid BEYOND THE COLUMN TOP: a member run reaching the last two
      cells puts k_lid = k_top + 2 past nz - 1, so the loop never
      reaches it and no NB is found;
    * VENT_DEPTH_CAP: the same deck column stretched so the run sits
      more than 4000 m above the root.

    CONDENSATE INSENSITIVITY ON DEVICE (item A1 of the S4-5 mirror
    pitfalls; CPU authority
    ``test_m2_layer_structure_roundoff_insensitive``).  The whole layer
    structure rides qt >= qs, not the M1 mask's bit-level qc > 0 limb,
    so a round-off-scale condensate shift must move NO index.  The
    epsilon is the AUTHORITY's own +-1e-12 kg/kg (the registered
    tolerance of the C9 amendment: "+-1e-12 qc perturbations must move
    nothing"), so this leg and the CPU fixture probe the same
    perturbation.

    S4-5b Item 2 -- the epsilon was widened to 1e-9 in the S4-5 build,
    justified by "1e-12 is below FP32 resolution at qc ~ 5e-4 (ULP
    3e-11) and would make a device leg vacuous".  RE-MEASURED this
    session: that justification describes the DECK cells k12-k15, where
    the column's qc is 6.529e-5 to 5.310e-4 with FP32 ulp 7.3e-12 to
    5.8e-11 -- cells this leg never touches.  The leg probes k = 9, 10,
    16 and 17, where the column's own qc is 0.0, 1.0516e-13,
    5.6843e-14 and 0.0, so the FP32 ulp there is 1.401e-45 / 6.776e-21
    / 3.388e-21 / 1.401e-45 and 1e-12 is 1.476e8 to 7.136e32 ulps --
    eight to thirty-two orders ABOVE FP32 resolution, not below it.  At
    1e-12 the fixture's own non-vacuity gate -- the M1 mask, which is a
    function of qc and is RE-DERIVED from each shifted state -- fires on
    4 of the 8 shifts, so the leg has real content and the epsilon is
    restored to the authority's.

    WHAT THE RESPONSE ACTUALLY IS, measured rather than assumed.  At
    1e-12 the device flux field comes back BITWISE unchanged on all 8
    shifts (worst relative movement exactly 0.000e+00), where the 1e-9
    leg measured a small nonzero movement.  That is not the leg going
    vacuous, it is the FP32 face store: the shift moves qt_env by ~1e-12
    against a qt of ~1e-2, a 1e-10 RELATIVE perturbation, which the
    FP64 sweep carries but the single FP32 rounding at the face store
    discards.  The assertions are unchanged and still bound the response
    from above (< 1e-4 relative), so they hold for either outcome.
    Asserted at the four cells the CPU fixture
    probes: indices bitwise unchanged, device-vs-authority parity holds
    on the SHIFTED state too, the C9 boundary stays bitwise zero, and
    the flux response is bounded rather than assumed bitwise -- the
    accurate claim, since qt enters the entraining parcel.
    """
    import cupy as cp
    from woof.verify import sase_ref
    from test_sase import _vent_args, _vent_deck_args

    def all_zero(arrs):
        return all(a.tobytes() == np.zeros_like(a).tobytes()
                   for a in arrs)

    # --- f = 1 LES limit, on a column that fires hard at f = 0 -------
    amp = _vent_args("amp")
    dev, ref, thick = _vent_device(amp)
    live = _vent_launch(dev, thick, f_blend=0.0)
    assert np.max(np.abs(live[0])) > 0.0             # non-vacuous
    les = _vent_launch(dev, thick, f_blend=1.0)
    assert all_zero(les[:3]), "f = 1 is not a bitwise +0.0 deposit"
    # the STRUCTURE is amplitude-independent (f enters only the
    # two-product blend), so the diagnosed indices are unchanged -- the
    # zeros come from the closed face gate, not from a stand-down.
    assert les[3].tobytes() == live[3].tobytes()

    # --- mask off (the M1 veto / pre-M2 identity) -------------------
    off = dict(dev)
    off["n2_moist"] = dev["n2_dry"].copy()
    assert all_zero(_vent_launch(off, thick)[:3])

    # --- no qualifying run: the control column ----------------------
    dctl, rctl, tctl = _vent_device(_vent_args("ctl"))
    ctl = _vent_launch(dctl, tctl)
    assert all_zero(ctl[:3])
    assert np.all(ctl[3] == -1)

    # The three TERMINATION stand-downs share a signature -- a
    # qualifying run EXISTS and the fluxes are still bitwise zero -- so
    # each leg names its own branch by the quantity that distinguishes
    # it, and every leg is checked against the authority first (a
    # fixture that stopped exercising its branch fails loudly here
    # rather than passing vacuously).
    from test_sase import _vent_layer_indices, _vent_root_index
    deck = _vent_deck_args(z_lid=1400.0)[0]

    def _zcent(a):
        nzc = np.asarray(a["theta"]).shape[0]
        t = np.broadcast_to(np.asarray(a["dz_col"], np.float64),
                            (nzc,)).astype(np.float64)
        return np.cumsum(t) - 0.5 * t

    def _leg(tag, a, checks):
        dv, rf, tk = _vent_device(a)
        aut = sase_ref.plume_vent_flux(**rf)
        assert all_zero(aut), (
            f"fixture no longer exercises the {tag} branch")
        kb, kt = _vent_layer_indices(rf)
        assert kb >= 0, (
            f"{tag}: the run itself vanished -- this leg would be "
            f"testing the step-1 stand-down instead")
        checks(rf, kb, kt)
        got = _vent_launch(dv, tk)
        assert all_zero(got[:3]), tag
        assert np.all(got[3] == -1), tag
        return kb, kt

    # --- NO LFC BELOW k_lid ------------------------------------------
    # The registered mechanism (design doc SASE-M2 amendment: the deep
    # root stands the limb down on 15-19% of real firing columns "no
    # LFC below k_lid").  Built by cooling one sub-cloud cell, which
    # puts a theta_es maximum at k2 -- a root TWO cells below the run's
    # base, inside RH-70% air -- and vetoing the mask above k5 so the
    # run is k4..k5 and k_lid = 7.  The parcel arrives at the deck
    # DILUTED and unsaturated: its virtual-temperature deficit from the
    # missing vapour outweighs the theta lag of the saturated adiabat,
    # so B stays <= 0 through the whole bounded search.  The depth is
    # 500 m, three orders inside VENT_DEPTH_CAP, which is what
    # distinguishes this leg from the next one.
    nolfc = dict(deck)
    th_c = np.asarray(deck["theta"], np.float64).copy()
    th_c[1] -= 1.0
    nolfc["theta"] = th_c
    n2_c = sase_ref.brunt_vaisala_n2(th_c, None, dz_col=deck["dz_col"])
    n2m_c = sase_ref.moist_n2(th_c, deck["qv"], deck["qc"], deck["p"],
                              deck["dz_col"])
    m_c = (n2m_c != n2_c)
    m_c[6:] = False
    nolfc["n2m_mask"] = m_c

    def _check_nolfc(rf, kb, kt):
        assert (kb, kt) == (4, 5), (kb, kt)
        k_r = _vent_root_index(rf, kb, kt)
        assert k_r == 2, k_r                        # root below the base
        z = _zcent(rf)
        depth = z[kt + 2] - z[k_r]
        assert depth < 0.2 * sase_ref.VENT_DEPTH_CAP, depth
        assert _vent_termination_ref(rf, kb, kt, k_r)[1] == -1

    _leg("no-LFC", nolfc, _check_nolfc)

    # --- k_lid BEYOND VENT_DEPTH_CAP ---------------------------------
    # The weak-lid column (its parcel stays buoyant to the ceiling on
    # every grid, so termination is the k_lid branch) discretized with
    # 500 m in-run layers: the ceiling then sits 5500 m above the root,
    # past the 4000 m shallow-device scope guard.
    weak = _vent_deck_args(z_lid=1400.0, dth_inv=2.0)[0]
    deep = dict(weak)
    t_deep = np.broadcast_to(np.asarray(weak["dz_col"], np.float64),
                             (np.asarray(weak["theta"]).shape[0],)).copy()
    t_deep[4:16] = 500.0
    deep["dz_col"] = t_deep

    def _check_deep(rf, kb, kt):
        k_r = _vent_root_index(rf, kb, kt)
        z = _zcent(rf)
        depth = z[kt + 2] - z[k_r]
        assert depth > sase_ref.VENT_DEPTH_CAP, depth

    _leg("VENT_DEPTH_CAP", deep, _check_deep)

    # --- k_lid PAST THE COLUMN TOP -----------------------------------
    # A member run in the top two cells puts k_lid = k_top + 2 past
    # nz - 1, so the search loop never reaches the ceiling and no NB is
    # ever found.
    top = dict(deck)
    nz_d = np.asarray(deck["theta"]).shape[0]
    qs_top = _vent_qs(deck)
    qv_top = np.asarray(deck["qv"], np.float64).copy()
    qc_top = np.asarray(deck["qc"], np.float64).copy()
    qv_top[nz_d - 2:] = qs_top[nz_d - 2:]
    qc_top[nz_d - 2:] = 1.0e-4      # qt clears qs with FP32 headroom
    top["qv"] = qv_top
    top["qc"] = qc_top
    mt = np.zeros_like(np.asarray(deck["n2m_mask"]))
    mt[nz_d - 2:] = True
    top["n2m_mask"] = mt

    def _check_top(rf, kb, kt):
        assert kt + 2 > np.asarray(rf["theta"]).shape[0] - 1, kt

    _leg("k_lid-past-top", top, _check_top)

    # --- condensate insensitivity (item A1) -------------------------
    eps = 1.0e-12                    # the authority's own (S4-5b Item 2)
    base_idx = tuple(int(v) for v in live[3][:, 0, 0])
    assert base_idx == (12, 15, 10, 17, 15, 17, 16)   # the pinned column
    moved_mask = 0
    worst_move = 0.0
    for k in (9, 10, 16, 17):
        for sign in (+1.0, -1.0):
            sh = dict(amp)
            qc_s = np.asarray(amp["qc"], np.float64).copy()
            qc_s[k] += sign * eps
            sh["qc"] = qc_s
            n2 = sase_ref.brunt_vaisala_n2(sh["theta"], None,
                                           dz_col=sh["dz_col"])
            n2m = sase_ref.moist_n2(sh["theta"], sh["qv"], qc_s,
                                    sh["p"], sh["dz_col"])
            sh["n2m_mask"] = (n2m != n2)
            if (np.asarray(sh["n2m_mask"]).tobytes()
                    != np.asarray(amp["n2m_mask"]).tobytes()):
                moved_mask += 1
            dsh, rsh, tsh = _vent_device(sh)
            got = _vent_launch(dsh, tsh)
            assert tuple(int(v) for v in got[3][:, 0, 0]) == base_idx, (
                f"condensate shift {sign * eps:+g} kg/kg at k{k} moved "
                f"an index")
            aut = sase_ref.plume_vent_flux(**rsh)
            for g, a in zip(got[:3], aut):
                rel, _ = _max_rel(g, a)
                assert rel <= 2e-6, (k, sign, rel)
                # C9: bitwise zero at and above the entrainment zone's
                # own top face (face 17 on this column)
                tail = np.zeros_like(g[17:])
                assert g[17:].tobytes() == tail.tobytes()
            move = max(float(np.abs(g - b).max())
                       / max(float(np.abs(b).max()), 1e-300)
                       for g, b in zip(got[:3], live[:3]))
            worst_move = max(worst_move, move)
    assert moved_mask > 0, (
        "the shifts no longer move the M1 mask -- the veto leg of this "
        "fixture has gone vacuous")
    assert worst_move < 1e-4, worst_move          # continuous, O(eps)
    print(f"M2 device condensate insensitivity: indices bitwise "
          f"unchanged under +-{eps:g} kg/kg at k9/k10/k16/k17 "
          f"({moved_mask}/8 shifts moved the M1 mask itself); worst "
          f"relative flux movement {worst_move:.3e}")


@requires_gpu
def test_m2_device_min_run_guard_and_mask_convention_binding(monkeypatch):
    """S4-5b Item 1, THE MINIMUM-RUN GUARD (and the mask-convention
    parameter that shares its loop) ON DEVICE.

    A review lane replaced the kernel's
    ``bool long_enough = (k - run_s) >= (min_run_cells - 1)`` with
    ``true`` and all four M2 GPU gates still passed.  MEASURED this
    session over every column of the four yolo-d d02 frames (800,000
    columns, probe s5b_p1_survey.py): at the registered pair
    (VENT_MASK = "bulk-theta-es-v1", VENT_MIN_RUN_CELLS = 2) the guard
    binds on EXACTLY ZERO columns -- and it CANNOT bind on any state,
    because the bulk reading is ``thes[k] - base_thes < 0`` with
    ``base_thes`` captured at the run's own start, so a one-cell run
    compares a float with itself, reads exactly 0.0, and is rejected by
    the READING before the guard is consulted.  At the registered
    values the guard is therefore implied by the reading and its
    deletion is a provable no-op: no fixture at those values, real or
    constructed, can detect it, and adding one would be theatre.

    It is live at the registry's own other values, both of which are
    config-ID members swept by
    ``test_sase.test_m2_registry_binds_vent_constants``, and this gate
    exercises it there -- which is also what makes the kernel's
    ``min_run_cells`` and ``per_level`` PARAMETERS constrained at all
    (the suite launched every previous M2 column at min_run_cells = 2
    and per_level = 0, so the kernel's ``mono`` limb was never
    executed):

    * VENT_MIN_RUN_CELLS = 3 on the CLAMP column of
      vent_columns_yolod_binding, whose member run is exactly 2 cells
      (k20..k21) and which has no other qualifying run: the guard, and
      nothing else, stands it down.  Authority and device both bitwise
      +0.0, indices all -1.  A kernel whose guard is deleted fires here
      (measured on the mutated kernel, S4-5b report).
    * VENT_MIN_RUN_CELLS = 1 -- the reviewer's ``true`` expressed
      through the registered parameter -- returns the SAME bytes the
      registered value 2 returns on that column, which is what proves
      the leg above isolates the guard rather than some other
      difference.
    * VENT_MASK = "per-level-theta-es-v1": the specimen amplifier's
      run carries interior theta_es increases, so the per-level reading
      VETOES it (``test_m2_mask_convention_flip_pinned`` pins that at
      the authority).  The device must flip the same way, bitwise, and
      must agree bitwise with the authority on a monotone deck column
      where the two readings coincide.

    DUAL-RUN on every launch.
    """
    from woof.core import sase as sase_dev
    from woof.verify import sase_ref
    from test_sase import _vent_args, _vent_binding_args, _vent_deck_args

    def all_zero(arrs):
        return all(a.tobytes() == np.zeros_like(a).tobytes()
                   for a in arrs)

    args = _vent_binding_args("CLAMP")
    dev, ref, thick = _vent_device(args)
    head = _vent_launch(dev, thick)
    assert np.max(np.abs(head[0])) > 0.0, (
        "the CLAMP column no longer fires at the registered constants "
        "-- this gate would be vacuous")
    assert sase_ref.VENT_MIN_RUN_CELLS == 2

    def set_min_run(value):
        monkeypatch.setattr(sase_ref, "VENT_MIN_RUN_CELLS", value)
        monkeypatch.setattr(sase_dev, "VENT_MIN_RUN_CELLS", value)

    set_min_run(3)
    assert all_zero(sase_ref.plume_vent_flux(**ref)), (
        "authority: the 2-cell run must be rejected at 3")
    got = _vent_launch(dev, thick)
    assert all_zero(got[:3]), (
        "device: the minimum-run guard did not stand the column down "
        "at VENT_MIN_RUN_CELLS = 3")
    assert np.all(got[3] == -1)

    set_min_run(1)
    neutral = _vent_launch(dev, thick)
    for a, b in zip(neutral, head):
        assert a.tobytes() == b.tobytes(), (
            "with the guard neutralised the device must reproduce the "
            "registered-value bytes exactly")
    set_min_run(2)

    # --- the mask-convention parameter (per_level) ------------------
    amp = _vent_args("amp")
    dev_a, ref_a, tk_a = _vent_device(amp)
    live_a = _vent_launch(dev_a, tk_a)
    assert np.max(np.abs(live_a[1])) > 0.0
    deck = _vent_deck_args(z_lid=1400.0)[0]
    dev_d, ref_d, tk_d = _vent_device(deck)
    bulk_d = _vent_launch(dev_d, tk_d)
    monkeypatch.setattr(sase_ref, "VENT_MASK", "per-level-theta-es-v1")
    monkeypatch.setattr(sase_dev, "VENT_MASK", "per-level-theta-es-v1")
    assert all_zero(sase_ref.plume_vent_flux(**ref_a))
    per_a = _vent_launch(dev_a, tk_a)
    assert all_zero(per_a[:3]), (
        "device: the per-level reading must veto the specimen")
    assert np.all(per_a[3] == -1)
    per_d = _vent_launch(dev_d, tk_d)
    for a, b in zip(per_d, bulk_d):
        assert a.tobytes() == b.tobytes(), (
            "device: the two readings must coincide on a monotone deck")
    for got_row, aut in zip(per_d[:3], sase_ref.plume_vent_flux(**ref_d)):
        rel, _ = _max_rel(got_row, aut)
        assert rel <= 2e-6, rel
    print("M2 device registry-parameter gates: minimum-run guard stands "
          "the 2-cell CLAMP column down at VENT_MIN_RUN_CELLS = 3 "
          "(bitwise +0.0, indices -1) and returns the registered bytes "
          "at 1; per_level = 1 vetoes the specimen bitwise and is "
          "bitwise identical to the bulk reading on the monotone deck")


@requires_gpu
def test_m2_vent_deposit_cap_and_ledger_device():
    """THE S4-5 DEPOSIT SEAM ON DEVICE: ``launch_vent_deposit_scale``
    and ``launch_vent_deposit`` against the authority
    ``vent_deposit_rescale``.

    THE CLAMP IS EXERCISED, not observed.  Flux amplitude scales exactly
    as sqrt(e_sgs), so the legs drive the column's own e_sgs up until
    the registered cap family binds (the boosts are amplitude drivers,
    not physical TKE claims -- no field statement is made here, so the
    G-M5 survey standard does not apply):

    * specimen at 100x e: the QC row binds (s ~ 0.58);
    * 1400-m deck at 2000x e: the THETA row binds (s ~ 0.1096);
    * specimen at 1x e: nothing binds, s == 1.0 EXACTLY (the
      pass-through leg -- the seam must be inert under cap);
    * the control column: inactive, every |d|max == +0.0, and the
      DIVIDE GUARD must return s == 1.0 without forming an infinity.

    Pinned on every leg: the scale plane matches the authority to <=
    2e-6, the deposited rows match the authority's capped deposits to
    <= 2e-6, the binding row lands ON its cap, and the ledger
    telescopes -- sum_k thick_k*d_phi_k = 0 to roundoff, which is the
    property the UNIFORM rescale exists for (a per-level clip of one
    row destroys it: measured -3.74 at the authority).

    THE LEDGER PIN'S MEASUREMENT CONDITION (S4-5b Item 5d; stated with
    the pin, as the timing pin's idle-card condition now is).  The 1e-6
    relative residual is measured with the deposit applied to a ZERO
    pre-solve state (``phi = cp.zeros``), which is what makes the rows
    BE the deposits and lets them be compared with the authority
    directly -- but it also means the FP32 rounding of ``phi + d`` acts
    on a value equal to the deposit itself.  The driver's background is
    the physical field: theta ~ 300 K against a deposit of order 0.1 K,
    a background ~1e3 larger, so the recovered residual there is
    correspondingly larger and the 1e-6 figure does NOT generalise to
    the driver.  The final leg below measures the same column on a
    physical background and pins that number separately, with its own
    derivation, rather than leaving the extrapolation to the reader.

    DUAL-RUN on every launch.
    """
    import cupy as cp
    from woof.core.sase import (launch_vent_deposit,
                                 launch_vent_deposit_scale)
    from woof.verify import sase_ref
    from test_sase import _vent_args, _vent_deck_args

    def driven(args, boost):
        out = dict(args)
        out["e_sgs"] = np.asarray(args["e_sgs"], np.float64) * boost
        return out

    dt = 60.0
    caps = (sase_ref.VENT_THETA_STEP_CAP, sase_ref.VENT_QT_STEP_CAP,
            sase_ref.VENT_QT_STEP_CAP)
    legs = (("specimen-1x", _vent_args("amp"), None),
            ("specimen-100x", driven(_vent_args("amp"), 100.0), 2),
            ("deck-2000x", driven(_vent_deck_args(z_lid=1400.0)[0],
                                  2000.0), 0),
            ("control", _vent_args("ctl"), None))
    for name, args, bind in legs:
        dev, ref, thick = _vent_device(args)
        f_th, f_qv, f_qc = _vent_launch(dev, thick, indices=False)[:3]
        fd = [cp.asarray(a) for a in (f_th, f_qv, f_qc)]
        scales = []
        for _ in range(2):
            scales.append(cp.asnumpy(launch_vent_deposit_scale(
                *fd, dev["rho1"], dt=dt, dz_col=thick)))
        assert scales[0].tobytes() == scales[1].tobytes(), (
            f"[{name}] dual-run mismatch on the cap scale")
        s_dev = scales[0]
        a_th, a_qv, a_qc = sase_ref.plume_vent_flux(**ref)
        d0, d1, d2, s_ref = sase_ref.vent_deposit_rescale(
            a_th, a_qv, a_qc, dt, ref["rho1"], dz_col=thick)
        assert np.allclose(s_dev, np.broadcast_to(s_ref, s_dev.shape),
                           rtol=2e-6, atol=0.0), (name, s_dev[0, 0],
                                                  s_ref.ravel()[0])
        if bind is None:
            assert np.all(s_dev == 1.0), (name, s_dev.min())
        else:
            assert 0.0 < float(s_dev[0, 0]) < 1.0
            dmax = float(np.abs((d0, d1, d2)[bind]).max())
            assert abs(dmax - caps[bind]) <= 4.0 * np.spacing(caps[bind])
        # the deposit itself, applied to a zero pre-solve state so the
        # rows ARE the deposits
        nz = np.asarray(ref["theta"]).shape[0]
        scale_dev = launch_vent_deposit_scale(*fd, dev["rho1"], dt=dt,
                                              dz_col=thick)
        for row, (frow, dref) in enumerate(zip(fd, (d0, d1, d2))):
            outs = []
            for _ in range(2):
                phi = cp.zeros(dev["theta"].shape, dtype=cp.float32)
                launch_vent_deposit(phi, frow, scale_dev, dev["rho1"],
                                    dt=dt, dz_col=thick)
                outs.append(cp.asnumpy(phi))
            assert outs[0].tobytes() == outs[1].tobytes(), (
                f"[{name}] dual-run mismatch on the deposit row {row}")
            got = outs[0]
            want = np.broadcast_to(dref, got.shape)
            if np.max(np.abs(want)) == 0.0:
                assert got.tobytes() == np.zeros_like(got).tobytes()
                continue
            rel, _ = _max_rel(got, want)
            assert rel <= 2e-6, (name, row, rel)
            # ledger: the net column term telescopes to zero.  CONDITION
            # (S4-5b Item 5d): measured on the ZERO background above --
            # the driver's own background is ~1e3 larger and is measured
            # separately after this loop.
            tcol = np.asarray(thick, np.float64)
            col = got[:, 0, 0].astype(np.float64)
            resid = abs(float(np.sum(tcol * col)))
            scale_abs = float(np.sum(tcol * np.abs(col)))
            assert resid <= 1e-6 * scale_abs, (name, row, resid,
                                               scale_abs)
        print(f"M2 deposit seam [{name}]: s = {float(s_dev[0, 0]):.9g}"
              + ("" if bind is None
                 else f" (row {bind} on its cap)"))

    # --- the same ledger on a PHYSICAL background (S4-5b Item 5d) ----
    # The driver never deposits onto zeros: the theta row's pre-solve
    # state is ~300 K.  Deposit the specimen's theta flux onto exactly
    # that and recover d = phi - 300 in FP64.  The recovered deposit now
    # carries one FP32 rounding at the BACKGROUND's exponent, half an
    # ulp of 300 K = 1.526e-5 K per level, against per-level deposits of
    # order 1e-2 K -- so the residual/scale ratio is bounded by roughly
    # nz*0.5*ulp(300)*thick / sum(thick*|d|), three orders above the
    # zero-background 1e-6 and entirely explained by that one rounding.
    dev, ref, thick = _vent_device(driven(_vent_args("amp"), 100.0))
    f_th, f_qv, f_qc = _vent_launch(dev, thick, indices=False)[:3]
    fd = [cp.asarray(a) for a in (f_th, f_qv, f_qc)]
    scale_dev = launch_vent_deposit_scale(*fd, dev["rho1"], dt=dt,
                                          dz_col=thick)
    bg = np.float32(300.0)
    outs = []
    for _ in range(2):
        phi = cp.full(dev["theta"].shape, bg, dtype=cp.float32)
        launch_vent_deposit(phi, fd[0], scale_dev, dev["rho1"], dt=dt,
                            dz_col=thick)
        outs.append(cp.asnumpy(phi))
    assert outs[0].tobytes() == outs[1].tobytes(), (
        "dual-run mismatch on the physical-background deposit")
    tcol = np.asarray(thick, np.float64)
    col = outs[0][:, 0, 0].astype(np.float64) - float(bg)
    resid_bg = abs(float(np.sum(tcol * col)))
    scale_bg = float(np.sum(tcol * np.abs(col)))
    half_ulp = 0.5 * float(np.spacing(bg))
    ulp_bound = float(np.sum(tcol * half_ulp))
    print(f"M2 deposit ledger, PHYSICAL background {float(bg):g} K: "
          f"residual {resid_bg:.6e} against scale {scale_bg:.6e} "
          f"(ratio {resid_bg / scale_bg:.3e}); one-rounding bound "
          f"{ulp_bound:.6e} ({ulp_bound / scale_bg:.3e}) -- the "
          f"zero-background pin above reads 1e-6 on the same column")
    assert resid_bg <= ulp_bound, (resid_bg, ulp_bound)


@requires_gpu
def test_m2_vent_batched_column_stack_device():
    """BATCHED HETEROGENEOUS COLUMNS (the launch-geometry gate): a
    (nz, ny, nx) stack in which every (j, i) carries a DIFFERENT
    registered column -- three prescribed-lid decks, the weak-lid
    column, a mask-off (stood-down) column and a no-LFC column, tiled
    over the whole horizontal plane -- must reproduce, cell for cell and
    BITWISE, the single-column launches of the same columns.

    This is what proves the one-thread-per-column sweep carries no
    cross-column state: the fixture family above runs on horizontally
    uniform tiles, where a thread that read a neighbour's column would
    still agree.  Dual-run.
    """
    import cupy as cp
    from woof.core.sase import launch_plume_vent_flux
    from test_sase import _vent_deck_args

    members = [_vent_deck_args(z_lid=1200.0)[0],
               _vent_deck_args(z_lid=1400.0)[0],
               _vent_deck_args(z_lid=1600.0)[0],
               _vent_deck_args(z_lid=1400.0, dth_inv=2.0)[0]]
    off = dict(members[1])
    off["n2m_mask"] = np.zeros_like(np.asarray(off["n2m_mask"]))
    members.append(off)
    nolfc = dict(members[1])
    m = np.asarray(members[1]["n2m_mask"]).copy()
    m[6:] = False
    nolfc["n2m_mask"] = m
    members.append(nolfc)
    nz = np.asarray(members[0]["theta"]).shape[0]
    ny, nx = 6, 11
    thick = np.asarray(members[0]["dz_col"], np.float64).astype(
        np.float32).astype(np.float64)
    for a in members:                       # one shared grid, by design
        assert np.asarray(a["dz_col"]).shape == (nz,)

    def pick(j, i):
        return members[(j * nx + i) % len(members)]

    def field(key, dtype=np.float32):
        out = np.empty((nz, ny, nx), dtype)
        for j in range(ny):
            for i in range(nx):
                out[:, j, i] = np.asarray(pick(j, i)[key],
                                          np.float64).reshape(nz)
        return np.ascontiguousarray(out)

    mask = np.empty((nz, ny, nx), bool)
    rho = np.empty((ny, nx), np.float32)
    for j in range(ny):
        for i in range(nx):
            mask[:, j, i] = np.asarray(pick(j, i)["n2m_mask"]).reshape(nz)
            rho[j, i] = np.float32(pick(j, i)["rho1"])
    dev = dict(
        theta=cp.asarray(field("theta")), qv=cp.asarray(field("qv")),
        qc=cp.asarray(field("qc")), pressure=cp.asarray(field("p")),
        e_sgs=cp.asarray(field("e_sgs")),
        n2_moist=cp.asarray(np.where(mask, np.float32(1.0),
                                     np.float32(0.0)).astype(np.float32)),
        n2_dry=cp.zeros((nz, ny, nx), dtype=cp.float32),
        rho1=cp.asarray(rho))
    stack = _vent_launch(dev, thick)
    singles = [_vent_launch(*_vent_device(a)[::2]) for a in members]
    for j in range(ny):
        for i in range(nx):
            ref = singles[(j * nx + i) % len(members)]
            for row in range(3):
                assert (stack[row][:, j, i].tobytes()
                        == ref[row][:, 0, 0].tobytes()), (j, i, row)
            assert (stack[3][:, j, i].tobytes()
                    == ref[3][:, 0, 0].tobytes()), (j, i, "idx")
    fired = int((stack[3][0] >= 0).sum())
    print(f"M2 batched stack: {ny}x{nx} columns from "
          f"{len(members)} distinct fixtures, {fired} firing, all "
          f"bitwise equal to their single-column launches")


@requires_gpu
def test_m2_vent_launcher_validation():
    """Launcher contracts (the S4-2 idiom): the required ``dz_col``, the
    SASE_KMAX column bound, field/plane shape+dtype checks, the
    ``f_blend`` domain (mirroring the authority ValueError), and the
    FP64 scale-plane contract of the two deposit launchers."""
    import cupy as cp
    from woof.core.sase import (launch_plume_vent_flux,
                                 launch_vent_deposit,
                                 launch_vent_deposit_scale)
    from test_sase import _vent_args
    dev, ref, thick = _vent_device(_vent_args("amp"))
    ok = dict(dev)
    call = lambda **kw: launch_plume_vent_flux(  # noqa: E731
        kw.get("theta", ok["theta"]), kw.get("qv", ok["qv"]),
        kw.get("qc", ok["qc"]), kw.get("pressure", ok["pressure"]),
        kw.get("e_sgs", ok["e_sgs"]), kw.get("n2_moist", ok["n2_moist"]),
        kw.get("n2_dry", ok["n2_dry"]), kw.get("rho1", ok["rho1"]),
        f_blend=kw.get("f_blend", 0.0),
        dz_col=kw.get("dz_col", thick))
    with pytest.raises(ValueError, match="dz_col"):
        call(dz_col=None)
    with pytest.raises(ValueError, match="f_blend"):
        call(f_blend=1.5)
    with pytest.raises(ValueError, match="qv"):
        call(qv=cp.zeros((2, 2, 2), dtype=cp.float32))
    with pytest.raises(ValueError, match="rho1"):
        call(rho1=cp.zeros((2, 2), dtype=cp.float32))
    with pytest.raises(ValueError, match="rho1"):
        call(rho1=cp.zeros(ok["rho1"].shape, dtype=cp.float64))
    deep = cp.zeros((256, 2, 2), dtype=cp.float32)
    with pytest.raises(ValueError, match="SASE_KMAX"):
        launch_plume_vent_flux(deep, deep, deep, deep, deep, deep, deep,
                               cp.zeros((2, 2), dtype=cp.float32),
                               f_blend=0.0,
                               dz_col=np.full(256, 50.0))
    f_th, f_qv, f_qc = _vent_launch(dev, thick, indices=False)[:3]
    fd = [cp.asarray(a) for a in (f_th, f_qv, f_qc)]
    with pytest.raises(ValueError, match="dz_col"):
        launch_vent_deposit_scale(*fd, dev["rho1"], dt=60.0)
    with pytest.raises(ValueError, match="f_qv"):
        launch_vent_deposit_scale(fd[0], fd[1][:-1], fd[2], dev["rho1"],
                                  dt=60.0, dz_col=thick)
    scale = launch_vent_deposit_scale(*fd, dev["rho1"], dt=60.0,
                                      dz_col=thick)
    assert scale.dtype == np.float64
    phi = cp.zeros(dev["theta"].shape, dtype=cp.float32)
    with pytest.raises(ValueError, match="scale"):
        launch_vent_deposit(phi, fd[0], scale.astype(cp.float32),
                            dev["rho1"], dt=60.0, dz_col=thick)
    with pytest.raises(ValueError, match="f_row"):
        launch_vent_deposit(phi, fd[0][:-1], scale, dev["rho1"],
                            dt=60.0, dz_col=thick)


def _vent_seam_time(dev, dz, phi, reps=10):
    """Mean ms/call of the whole S4-5 driver seam (flux kernel + cap
    scale + the three deposit rows) at the given field set."""
    import cupy as cp
    from woof.core.sase import (launch_plume_vent_flux,
                                 launch_vent_deposit,
                                 launch_vent_deposit_scale)

    def seam():
        f = launch_plume_vent_flux(
            dev["theta"], dev["qv"], dev["qc"], dev["pressure"],
            dev["e_sgs"], dev["n2_moist"], dev["n2_dry"], dev["rho1"],
            f_blend=0.0, dz_col=dz)
        s = launch_vent_deposit_scale(*f, dev["rho1"], dt=15.0,
                                      dz_col=dz)
        for row in f:
            launch_vent_deposit(phi, row, s, dev["rho1"], dt=15.0,
                                dz_col=dz)

    seam()                                      # warm (pool/compile)
    s0, e0 = cp.cuda.Event(), cp.cuda.Event()
    s0.record()
    for _ in range(reps):
        seam()
    e0.record()
    e0.synchronize()
    return cp.cuda.get_elapsed_time(s0, e0) / reps


#: Wall-clock pin per card, ms/call at the d02 shape.  A kernel-time pin
#: is calibration-relative: each entry is 1.22x that card's OWN measured
#: worst-case dual-run baseline on an idle card, which is the derivation
#: the 30.0 ms RTX 5090 pin has carried since 2026-07-24.  Asserting one
#: card's calibration on another card measures the hardware, not a
#: regression -- the first sm_89 run (RTX 4090, driver 590.48.01, CUDA
#: 13.1, 2026-08-03, arwen-stress-4090) measured 32.587/32.062 ms on an
#: idle card, i.e. the architecture is 1.32x the 5090 baseline on this
#: seam with nothing regressed.  Keyed on the device NAME, not the
#: compute capability, because sm_89 alone spans a 4x SM-count range and
#: a per-arch key would assert the 4090's budget on an RTX 4060.  A card
#: absent from this table has no baseline, so its measurement is
#: REPORTED, not asserted; calibrate a 1.22x worst-case dual-run entry
#: before relying on the budget there.
_M2_VENT_D02_PIN_MS = {
    # 24.6 ms worst-case dual-run baseline, 2026-07-24.
    "NVIDIA GeForce RTX 5090": 30.0,
    # 32.587 ms worst-case dual-run baseline, 2026-08-03 (driver
    # 590.48.01, CUDA/NVRTC 13.1, first user-zero sm_89 stress run).
    "NVIDIA GeForce RTX 4090": 39.8,
}


@requires_gpu
def test_m2_vent_kernel_time_increment_d02_shape():
    """KERNEL-TIME BUDGET, M2 INCREMENT (spec C12; the S4-2/S4-3c report
    idiom): the seam's cost at the EXACT production d02 shape
    (49, 501, 501) -- launch_plume_vent_flux + launch_vent_deposit_scale
    + three launch_vent_deposit rows -- on a WORST-CASE field where
    EVERY column carries a firing 8-cell saturated deck (production
    decks cover a fraction of columns, and a non-firing column exits the
    kernel after step 1).

    Reported against the yolo-d pre-M1 d02 seam baseline 62.12 ms/call
    (out/sase-yolo-d/run-metrics.json seam_ms_mean, 2026-07-22), the
    same denominator the M1 (6.4%) and M1b (4.8%) increments were quoted
    against; the budget VERDICT (<= 5% d01-d03) is re-scored on the S4-6
    smoke receipt, exactly as those two were.  DUAL-RUN: the pair of
    timing passes is reported, and the PIN is on the slower one.  The
    pin itself is PER CARD (``_M2_VENT_D02_PIN_MS``): each calibrated
    device asserts 1.22x its own worst-case dual-run baseline, and an
    uncalibrated device records the measurement without asserting.

    MEASUREMENT CONDITION (stated, not a tolerance): this is a wall-clock
    kernel timing and it requires an IDLE card -- the standing rule
    ("GPU authorized when the card is free") is a precondition of
    the number, not just of the courtesy.  MEASURED under a concurrent
    GPU-heavy test suite the same seam reads 50.8-52.5 ms, i.e. 2.1x,
    and this pin fails.  The pin is NOT loosened to cover that: a 30 ms
    bound on an idle card is a real regression guard, and a bound that
    survived a 2x contention factor would guard nothing.
    """
    import cupy as cp
    from woof.core.sase import (launch_plume_vent_flux,
                                 launch_vent_deposit,
                                 launch_vent_deposit_scale)
    shape = (49, 501, 501)
    zp = np.cumsum(np.full(49, 300.0)) - 150.0
    th = np.broadcast_to((300.0 + 0.004 * zp)[:, None, None],
                         shape).astype(np.float32).copy()
    p = np.broadcast_to((95000.0 * np.exp(-zp / 8000.0))[:, None, None],
                        shape).astype(np.float32).copy()
    # a genuinely SATURATED deck (qt >= qs is the membership test): qv
    # set to qs from the authority's own Tetens/Exner chain on the same
    # (theta, p), plus condensate.
    from woof.verify.sase_ref import (CP_AIR, EP2_RV, P0_REF, RD_AIR,
                                       SVP1, SVP2, SVP3, SVPT0)
    t_h = th.astype(np.float64) * (p.astype(np.float64)
                                   / P0_REF) ** (RD_AIR / CP_AIR)
    es_h = 1000.0 * SVP1 * np.exp(SVP2 * (t_h - SVPT0) / (t_h - SVP3))
    qs_h = EP2_RV * es_h / (p.astype(np.float64) - es_h)
    qv = (0.7 * qs_h).astype(np.float32)
    qv[4:12] = qs_h[4:12].astype(np.float32)
    qc = np.zeros(shape, np.float32)
    qc[4:12] = np.float32(5.0e-4)
    n2d = np.zeros(shape, np.float32)
    n2m = np.zeros(shape, np.float32)
    n2m[4:12] = np.float32(1.0)                # every column masked
    dev = dict(theta=cp.asarray(th), qv=cp.asarray(qv),
               qc=cp.asarray(qc), pressure=cp.asarray(p),
               e_sgs=cp.full(shape, np.float32(1.0), dtype=cp.float32),
               n2_moist=cp.asarray(n2m), n2_dry=cp.asarray(n2d),
               rho1=cp.full((501, 501), np.float32(1.1),
                            dtype=cp.float32))
    dz = cp.full(shape, np.float32(300.0), dtype=cp.float32)
    phi = cp.zeros(shape, dtype=cp.float32)

    runs = (_vent_seam_time(dev, dz, phi), _vent_seam_time(dev, dz, phi))
    inc = max(runs)
    # the limb really fired on every column (non-vacuity)
    _, _, _, idx = launch_plume_vent_flux(
        dev["theta"], dev["qv"], dev["qc"], dev["pressure"],
        dev["e_sgs"], dev["n2_moist"], dev["n2_dry"], dev["rho1"],
        f_blend=0.0, dz_col=dz, indices=True)
    fired = float((cp.asnumpy(idx)[0] >= 0).mean())
    assert fired == 1.0, fired
    print(f"d02-shape M2 seam increment: {runs[0]:.3f} / {runs[1]:.3f} "
          f"ms/call (dual-run), worst {inc:.3f} ms = "
          f"{100 * inc / 62.12:.2f}% of the pre-M1 d02 seam mean "
          f"62.12 ms -- worst-case every-column firing deck")
    # The MASK-OFF floor: what the seam costs on a column that never
    # fires (pass 1 -- the FP64 pow/exp theta_es chain -- runs at every
    # cell because the membership test itself needs it, then the kernel
    # exits).  Production decks cover a fraction of the domain, so this
    # is the share of the increment that is NOT avoidable.
    dev_off = dict(dev)
    dev_off["n2_moist"] = dev["n2_dry"]
    floor_ms = _vent_seam_time(dev_off, dz, phi)
    print(f"  mask-off floor (no column fires): {floor_ms:.3f} ms/call "
          f"= {100 * floor_ms / 62.12:.2f}% of the same denominator; "
          f"the firing surcharge is {inc - floor_ms:.3f} ms and scales "
          f"with the firing-column fraction (spec mask occupancy "
          f"0.05-13%)")
    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(device_name, (bytes, bytearray)):
        device_name = bytes(device_name).decode("utf-8", errors="replace")
    pin_ms = _M2_VENT_D02_PIN_MS.get(device_name)
    if pin_ms is None:
        print(f"  UNCALIBRATED CARD {device_name!r}: worst {inc:.3f} "
              f"ms/call RECORDED, not asserted -- add a 1.22x worst-case "
              f"dual-run baseline entry to _M2_VENT_D02_PIN_MS before "
              f"relying on this seam's budget here")
    else:
        assert inc <= pin_ms, (
            f"M2 seam at the production d02 shape regressed to {inc:.3f} "
            f"ms/call on {device_name} (pin {pin_ms} ms = 1.22x that "
            f"card's worst-case dual-run idle baseline; table above)")


@requires_gpu
def test_sase_driver_d02_first_light_vent_50_steps(monkeypatch):
    """M2 FIRST LIGHT, DRIVER LEVEL (S4-5; the S4-2 idiom): 50 full
    model steps on the SAME admitted d02-class configuration the M1
    first light used (dx = dy = 3 km, dt = 15 s, 49 x 250 x 250, moist
    Morrison, sfclay + Noah, zero-flux radiation stub) with the M2
    DEPOSIT SEAM LIVE in the driver -- the first executions in which the
    venting limb's flux actually enters the model.

    The column carries a sustained saturated stratocumulus-like deck
    (qv = qs,liq AND qc = 0.2 g/kg in the 450-1400 m band over a
    130 x 130 warm-sector patch, qv capped at 0.7*qs elsewhere), which
    on this profile is genuinely MOIST-UNSTABLE: theta_es falls 7.17 K
    across the deck levels k8..k17 while theta rises 3.5 K, so the
    registered mask fires and the limb has something to do.

    Logged per early step, through the SAME launcher chain the driver
    runs: how many columns FIRE (a diagnosed neutral-buoyancy level --
    the post-step-4 ``active`` set, the registered survey set), and how
    many have their cap-family rescale ENGAGE (s < 1).  The limb must
    engage from step 0 and stay engaged.

    Stability bands, identical to the M1 first light (never widened):
    prognostics + e_sgs finite, e_sgs in [E_MIN, 50], |u| <= 40,
    |w| <= 20.  The M2 deposit is a NEW explicit source in the scalar
    rows, so this is the gate that says it does not destabilize them:
    additionally qv and qc must stay non-negative and finite and theta
    must stay in a physical band.

    NOT A FIELD SURVEY.  This is a synthetic first-light fixture, so no
    amplitude conclusion is drawn from it and the registered survey
    standard (evaluate at design-point turbulence inside the G-M5 band
    [0.5, 1.6] m2/s2, quote the set/sampling rule/seed) does not apply:
    the numbers printed here are engagement counts and stability bands,
    not flux magnitudes offered as evidence about the closure.
    """
    import time

    import cupy as cp
    from woof.config import RunConfig, validate_run_config
    from woof.core import dycore
    from woof.core import physics as physics_mod
    from woof.core.physics import initialize_physics, sase_surface_rho1
    from woof.core.sase import (launch_moist_n2, launch_n2,
                                 launch_plume_vent_flux,
                                 launch_vent_deposit_scale)
    from woof.verify.sase_ref import (CP_AIR, E_MIN, EP2_RV, P0_REF,
                                       RD_AIR, SVP1, SVP2, SVP3, SVPT0)

    cfg = RunConfig(nx=250, ny=250, nz=49, dx=3000.0, dy=3000.0,
                    ztop=16000.0, dt=15.0, run_seconds=750.0,
                    time_step_sound=4, moist=True, mp_physics=10,
                    ra_physics=4, sf_sfclay_physics=1,
                    sf_surface_physics=2, km_opt=0,
                    bl_pbl_physics=_SASE_SELECTOR)
    assert validate_run_config(cfg) is cfg
    state = _moist_driver_state(cfg, stretch=1.6)
    z_half = np.asarray(state.height_half(), np.float64)
    th_tot = cp.asnumpy(state.total_theta()).astype(np.float64)
    p_eos = cp.asnumpy(state.p).astype(np.float64)
    t_h = th_tot * (p_eos / P0_REF) ** (RD_AIR / CP_AIR)
    es_h = 1000.0 * SVP1 * np.exp(SVP2 * (t_h - SVPT0) / (t_h - SVP3))
    qs_h = EP2_RV * es_h / (p_eos - es_h)
    deck = (z_half >= 450.0) & (z_half <= 1400.0)
    patch = np.zeros((cfg.ny, cfg.nx), bool)
    patch[60:190, 60:190] = True
    deck3 = deck[:, None, None] & patch[None]
    qv_h = np.minimum(cp.asnumpy(state.qv).astype(np.float64), 0.7 * qs_h)
    qv_h[deck3] = qs_h[deck3]
    qc_h = np.zeros_like(qv_h)
    # 0.6 g/kg, not the M1 first light's 0.2: MEASURED here, the limb's
    # own first-step moisture export at design-point e removes ~1e-4
    # kg/kg from the source cells in one 15 s step, so a deck planted at
    # EXACTLY RH 100% with 0.2 g/kg de-saturates (qt < qs) for a step
    # before microphysics restores it and the limb stands down on the
    # whole patch at step 1.  That is the fixture's headroom, not a
    # closure defect -- the M1 mask (qc > 0) never noticed, M2's
    # noise-immune qt >= qs membership did -- but a first light that
    # blinks off is not a first light, so the deck is planted with
    # enough condensate to survive its own export.
    qc_h[deck3] = 6.0e-4
    state.qv[...] = cp.asarray(qv_h.astype(np.float32))
    state.qc[...] = cp.asarray(qc_h.astype(np.float32))
    shear = (5.0 + 8.0 * z_half / cfg.ztop).astype(np.float32)
    state.u[...] = cp.asarray(np.broadcast_to(
        shear[:, None, None], (cfg.nz, cfg.ny, cfg.nx + 1)))
    state.v[...] = cp.float32(1.0)
    # Design-point subgrid energy: the limb's amplitude rides sqrt(e),
    # and the M1 uptake is what supplies it in production.  Seeding
    # inside the G-M5 reference band [0.5, 1.6] m2/s2 makes the FIRST
    # step exercise the seam rather than waiting on spin-up.
    state.e_sgs[...] = cp.float32(1.0)
    n_patch = int(patch.sum())

    initialize_physics(state, cfg, landmask=1.0, tsk=302.0,
                       swdown=400.0, glw=320.0, radiation=_zero_radiation)

    def vent_census():
        """(firing columns, capped columns, max |s| deficit) through the
        SAME launcher chain _run_sase drives."""
        atm = physics_mod._prepare_atmosphere(state)
        n2 = launch_n2(atm["theta"], dz_col=atm["dz"])
        n2m = launch_moist_n2(atm["theta"], atm["qv"], atm["qc"],
                              atm["pressure"], n2, dz_col=atm["dz"])
        rho1 = sase_surface_rho1(p1=atm["p_interface"][0],
                                 t1=atm["temperature"][0],
                                 qv1=atm["qv"][0])
        f_th, f_qv, f_qc, idx = launch_plume_vent_flux(
            atm["theta"], atm["qv"], atm["qc"], atm["pressure"],
            state.e_sgs, n2m, n2, rho1, f_blend=0.0, dz_col=atm["dz"],
            indices=True)
        s = launch_vent_deposit_scale(f_th, f_qv, f_qc, rho1,
                                      dt=cfg.dt, dz_col=atm["dz"])
        fired = int((cp.asnumpy(idx)[5] >= 0).sum())
        s_h = cp.asnumpy(s)
        capped = int((s_h < 1.0).sum())
        peak = float(np.abs(cp.asnumpy(f_qv) + cp.asnumpy(f_qc)).max())
        del atm, n2, n2m, f_th, f_qv, f_qc, idx, s
        return fired, capped, peak

    census = []
    wall_steps = 0.0
    for step in range(50):
        if step < 5 or step == 49:
            census.append((step,) + vent_census())
        t0 = time.perf_counter()
        dycore.step(state, cfg)                # physics runs inside
        cp.cuda.Stream.null.synchronize()
        wall_steps += time.perf_counter() - t0
        e = state.e_sgs
        assert bool(cp.isfinite(e).all()), f"e_sgs NaN at step {step + 1}"
        e_max = float(e.max())
        assert float(e.min()) >= np.float32(E_MIN) and e_max <= 50.0, (
            f"e_sgs unbounded at step {step + 1}: max {e_max}")
    print("d02 M2 first light (step, firing columns, capped columns, "
          "peak |F_qv+F_qc| kg/m2/s): "
          + ", ".join(f"({s}, {f}, {c}, {p:.3e})"
                      for s, f, c, p in census))
    # NON-VACUITY: the limb must actually deposit, from step 0 on.
    assert census[0][1] >= n_patch // 2, (
        f"M2 never engaged: {census[0][1]} firing columns vs the "
        f"{n_patch}-column saturated patch")
    for s, fired, _, _ in census[1:]:
        assert fired >= n_patch // 2, (
            f"M2 collapsed at step {s}: {fired} firing columns")
    for name in ("u", "v", "w", "thp", "qv", "qc", "e_sgs"):
        arr = cp.asnumpy(getattr(state, name))
        assert np.all(np.isfinite(arr)), name
    assert float(cp.abs(state.u).max()) <= 40.0
    assert float(cp.abs(state.w).max()) <= 20.0
    # The deposit is a NEW explicit source in the scalar rows: the
    # moisture rows must stay physical and theta must stay in band.
    assert float(state.qv.min()) >= 0.0
    assert float(state.qc.min()) >= 0.0
    th_end = cp.asnumpy(state.total_theta())
    assert float(th_end.min()) > 250.0 and float(th_end.max()) < 500.0
    print(f"driver d02 M2 first light: 50 steps at dt=15 on "
          f"(49, 250, 250); final e_max={float(state.e_sgs.max()):.4f}, "
          f"u_max={float(cp.abs(state.u).max()):.2f}, "
          f"w_max={float(cp.abs(state.w).max()):.3f}, "
          f"theta in [{float(th_end.min()):.1f}, "
          f"{float(th_end.max()):.1f}] K -- STABLE "
          f"({wall_steps * 1000.0:.0f} ms of stripped-step wall)")


@requires_gpu
def test_m2_driver_deposit_end_to_end_separation(monkeypatch):
    """THE END-TO-END GATE (S4-5b Item 3): the driver's M2 deposit
    changes the PROGNOSTIC fields, by a measured amount, in the
    direction a venting limb must move them.

    Before this test no test in either suite would have failed if
    ``launch_vent_deposit`` were silently inert on device.  The CPU shim
    test pins call ORDER against NumPy fakes; the device tests exercise
    the kernels standalone; and the d02 first-light test's engagement
    census calls the launchers ITSELF as a sidecar, with its own
    hardcoded ``f_blend = 0.0``, so every one of its assertions passes
    unchanged with the driver's deposit disabled.

    WHAT THIS RUNS.  The same admitted d02-class configuration the
    first lights use (dx = dy = 3 km, dt = 15 s, moist Morrison,
    sfclay + Noah, zero-flux radiation stub, a saturated
    moist-unstable deck planted over a warm-sector patch, design-point
    e_sgs = 1.0 m2/s2 inside the G-M5 band), at 49 x 120 x 120 for
    ``STEPS`` full model steps, TWICE from the identical initial state:
    once as the driver ships, and once with ``launch_vent_deposit``
    monkeypatched to a no-op -- exactly the "silently inert" failure
    mode -- with every other launcher, including the flux and
    cap-scale kernels, still running.

    WHAT IS ASSERTED (all three fail if the deposit is inert, and all
    three fail if the driver hands the limb f = 1, since the FP-exact
    two-product blend makes that a bitwise +0.0 deposit):

    1. SEPARATION.  max |qv_live - qv_off| over the field exceeds the
       floor pinned below, and likewise for qc and theta.
    2. DIRECTION AND SIGN.  Over the patch, the deposit REMOVES total
       water from the plume's source levels and DELIVERS it above the
       deck: patch-mean d(qv + qc) < 0 below the deck top and > 0 in
       the band above it.  That is the moisture channel the spec says
       M2 exists for (design doc SASE-M2 amendment striking the
       counter-gradient clause: "M2's justification is the MOISTURE
       channel").
    3. LEDGER.  The separation's own column integral telescopes:
       |sum_k thick_k * d(qv + qc)_k| stays a small fraction of
       sum_k thick_k * |d(qv + qc)_k| on the patch mean, which is the
       F[0] = F[nz] = +0.0 boundary contract surviving all the way
       through the implicit solve into the prognostic field.

    The f = 1 branch is ALSO named directly: the driver's own used
    partition fraction is read back off the retained ledger and
    asserted < 1, so a run in which the LES limit had silently swallowed
    the limb fails with that message rather than with an opaque
    separation floor.

    MEASURED this session at 49 x 120 x 120, 5 steps, f_used = 0.0:
    max|dtheta| = 6.988525e-02 K, max|dqv| = 1.606317e-04 kg/kg,
    max|dqc| = 3.489794e-05 kg/kg; patch-mean d(qv + qc)
    = -1.730126e-05 below 1400 m and +2.180941e-05 above it; column
    ledger residual 2.617851e-04 against scale 4.188142e-02, ratio
    6.251e-03.  The pinned floors below are those values divided by 4 --
    an order-of-magnitude band, not a tolerance: the quantity is a
    deterministic difference of two deterministic runs, and the band
    exists so a fixture drift that halves the engagement still fails.

    DUAL-RUN (no ECC on this card): the live configuration is run TWICE
    from its own freshly built initial state and the two final
    prognostic fields are asserted bitwise identical before either is
    differenced against the suppressed run.
    """
    import cupy as cp
    from woof.config import RunConfig, validate_run_config
    from woof.core import dycore
    from woof.core import physics as physics_mod
    from woof.core.physics import initialize_physics
    from woof.verify.sase_ref import (CP_AIR, EP2_RV, P0_REF, RD_AIR,
                                       SVP1, SVP2, SVP3, SVPT0)

    ny = nx = 120
    steps = 5
    z_deck = (450.0, 1400.0)

    def build():
        """The identical initial condition, built twice from the same
        deterministic construction (no RNG anywhere in it)."""
        cfg = RunConfig(nx=nx, ny=ny, nz=49, dx=3000.0, dy=3000.0,
                        ztop=16000.0, dt=15.0,
                        run_seconds=float(steps) * 15.0,
                        time_step_sound=4, moist=True, mp_physics=10,
                        ra_physics=4, sf_sfclay_physics=1,
                        sf_surface_physics=2, km_opt=0, bl_pbl_physics=_SASE_SELECTOR)
        assert validate_run_config(cfg) is cfg
        state = _moist_driver_state(cfg, stretch=1.6)
        zh = np.asarray(state.height_half(), np.float64)
        th_tot = cp.asnumpy(state.total_theta()).astype(np.float64)
        p_eos = cp.asnumpy(state.p).astype(np.float64)
        t_h = th_tot * (p_eos / P0_REF) ** (RD_AIR / CP_AIR)
        es_h = 1000.0 * SVP1 * np.exp(SVP2 * (t_h - SVPT0)
                                      / (t_h - SVP3))
        qs_h = EP2_RV * es_h / (p_eos - es_h)
        deck = (zh >= z_deck[0]) & (zh <= z_deck[1])
        pat = np.zeros((cfg.ny, cfg.nx), bool)
        pat[30:90, 30:90] = True
        deck3 = deck[:, None, None] & pat[None]
        qv_h = np.minimum(cp.asnumpy(state.qv).astype(np.float64),
                          0.7 * qs_h)
        qv_h[deck3] = qs_h[deck3]
        qc_h = np.zeros_like(qv_h)
        qc_h[deck3] = 6.0e-4
        state.qv[...] = cp.asarray(qv_h.astype(np.float32))
        state.qc[...] = cp.asarray(qc_h.astype(np.float32))
        shear = (5.0 + 8.0 * zh / cfg.ztop).astype(np.float32)
        state.u[...] = cp.asarray(np.broadcast_to(
            shear[:, None, None], (cfg.nz, cfg.ny, cfg.nx + 1)))
        state.v[...] = cp.float32(1.0)
        state.e_sgs[...] = cp.float32(1.0)
        drv = initialize_physics(state, cfg, landmask=1.0, tsk=302.0,
                                 swdown=400.0, glw=320.0,
                                 radiation=_zero_radiation)
        return cfg, state, drv, zh, pat

    def advance(cfg, state):
        for _ in range(steps):
            dycore.step(state, cfg)
        cp.cuda.Stream.null.synchronize()
        return (cp.asnumpy(state.total_theta()).astype(np.float64),
                cp.asnumpy(state.qv).astype(np.float64),
                cp.asnumpy(state.qc).astype(np.float64))

    cfg, state, drv, z_half, patch = build()
    live = advance(cfg, state)
    f_used = float(drv.last_sase_ledger["f"])
    thick = cp.asnumpy(physics_mod._prepare_atmosphere(state)["dz"]
                       )[:, 0, 0].astype(np.float64)
    del state, drv
    cp.get_default_memory_pool().free_all_blocks()
    # DUAL-RUN: the same configuration again, from its own fresh state
    cfg_b, state_b, drv_b, _, _ = build()
    live_b = advance(cfg_b, state_b)
    del state_b, drv_b
    cp.get_default_memory_pool().free_all_blocks()
    for a, b, lbl in zip(live, live_b, ("theta", "qv", "qc")):
        assert a.tobytes() == b.tobytes(), (
            f"dual-run mismatch on the live driver run ({lbl})")

    calls = []

    def inert_deposit(phi, f_row, scale, rho1, *, dt, dz_col=None):
        """The failure mode itself: the driver's deposit does nothing."""
        calls.append(1)
        return phi

    monkeypatch.setattr(physics_mod, "launch_vent_deposit",
                        inert_deposit)
    cfg2, state2, drv2, _, _ = build()
    off = advance(cfg2, state2)
    del state2, drv2
    cp.get_default_memory_pool().free_all_blocks()
    assert calls, ("the suppressed run never reached the driver's "
                   "deposit call -- this test would prove nothing")

    assert f_used < 1.0, (
        f"the driver handed the limb f = {f_used}: at f = 1 the "
        f"two-product blend makes the deposit bitwise +0.0 and this "
        f"gate would be measuring nothing")

    d_th, d_qv, d_qc = (a - b for a, b in zip(live, off))
    d_qt = d_qv + d_qc
    m_th = float(np.abs(d_th).max())
    m_qv = float(np.abs(d_qv).max())
    m_qc = float(np.abs(d_qc).max())
    deck_top = z_deck[1]
    below = (z_half > z_deck[0]) & (z_half <= deck_top)
    above = (z_half > deck_top) & (z_half <= deck_top + 1200.0)
    mean_below = float(np.mean(d_qt[below][:, patch]))
    mean_above = float(np.mean(d_qt[above][:, patch]))
    col = np.mean(d_qt[:, patch], axis=1)
    resid = abs(float(np.sum(thick * col)))
    scale = float(np.sum(thick * np.abs(col)))
    print(f"M2 driver end-to-end separation over {steps} steps at "
          f"(49, {ny}, {nx}), f_used={f_used:.6f}: "
          f"max|dtheta|={m_th:.6e} K, max|dqv|={m_qv:.6e}, "
          f"max|dqc|={m_qc:.6e} kg/kg; patch-mean d(qv+qc) "
          f"{mean_below:.6e} below {deck_top:g} m and "
          f"{mean_above:.6e} above it; column ledger residual "
          f"{resid:.6e} against scale {scale:.6e} "
          f"({resid / max(scale, 1e-300):.3e})")
    assert m_qv > 4.0e-5, m_qv                 # measured 1.606317e-04
    assert m_qc > 8.7e-6, m_qc                 # measured 3.489794e-05
    assert m_th > 1.7e-2, m_th                 # measured 6.988525e-02
    assert mean_below < 0.0, mean_below
    assert mean_above > 0.0, mean_above
    assert resid <= 0.05 * scale, (resid, scale)


# ---------------------------------------------------------------------------
# SPLIT SUBGRID-FLUX DIAGNOSTIC (RunConfig sase_flux_diag): the device
# fills.  Four z-FACE history fields, (nz+1, ny, nx) FP32, POSITIVE
# UPWARD, recording the closure's own vertical subgrid moisture and heat
# flux with the M2 vent channel SEPARATED from the K_v implicit
# diffusion channel.  Gates below: the vent channel is the CAP-SCALED
# flux the deposit applied (bitwise); the K_v channel is the solver's
# own flux (authority parity) and its convergence IS the solver's
# increment; the two channels SUM to the model's total increment; the
# vent channel is bitwise +0.0 wherever the M2 limb stands down; and
# enabling the whole thing leaves every prognostic bitwise unchanged.
# ---------------------------------------------------------------------------


@requires_gpu
def test_flux_diag_vent_channel_is_the_capped_deposited_flux():
    """GATE: SASE_F*_VENT is ``fac * scale * F``, formed in the deposit
    kernel's own op order -- i.e. the flux the model APPLIED, not the
    unscaled profile ``launch_plume_vent_flux`` returns.  Asserted
    BITWISE against the FP64 product, and the recorded field is then
    shown to reproduce the deposit's own increment."""
    import cupy as cp
    from woof.core.sase import launch_vent_deposit, launch_vent_flux_diag
    from woof.verify.sase_ref import CP_AIR

    nz, ny, nx = SHAPE
    rng = np.random.default_rng(SEED + 71)
    f_row32 = np.ascontiguousarray(
        (1.0e-4 * rng.standard_normal((nz + 1, ny, nx))).astype(np.float32))
    f_row32[0] = np.float32(0.0)                 # the interface contract
    f_row32[nz] = np.float32(0.0)
    scale64 = np.ascontiguousarray(
        rng.uniform(0.2, 1.0, (ny, nx)).astype(np.float64))
    f_dev = cp.asarray(f_row32)
    s_dev = cp.asarray(scale64)
    zeros2 = np.zeros((ny, nx), np.float32)

    for fac in (1.0, float(CP_AIR)):
        out = cp.empty((nz + 1, ny, nx), dtype=cp.float32)
        got = launch_vent_flux_diag(f_dev, s_dev, out, fac=fac)
        assert got is out                        # writes the buffer given
        ref = (fac * (scale64[None] * f_row32.astype(np.float64))
               ).astype(np.float32)
        host = cp.asnumpy(out)
        assert host.tobytes() == ref.tobytes(), fac
        # The end faces stay the literal +0.0 through the multiply.
        assert host[0].tobytes() == zeros2.tobytes(), fac
        assert host[nz].tobytes() == zeros2.tobytes(), fac

    # The recorded field reproduces the deposit the model actually made.
    thick32 = _per_column_dz32()
    rho1_32 = np.ascontiguousarray(
        rng.uniform(0.9, 1.2, (ny, nx)).astype(np.float32))
    dt = 30.0
    phi0 = np.ascontiguousarray(
        (1.0e-2 + 1.0e-3 * rng.standard_normal((nz, ny, nx))
         ).astype(np.float32))
    phi_dev = cp.asarray(phi0)
    launch_vent_deposit(phi_dev, f_dev, s_dev, cp.asarray(rho1_32),
                        dt=dt, dz_col=cp.asarray(thick32))
    applied = cp.asnumpy(phi_dev).astype(np.float64) - phi0.astype(np.float64)
    assert np.abs(applied).max() > 0.0           # non-vacuous
    out = cp.empty((nz + 1, ny, nx), dtype=cp.float32)
    launch_vent_flux_diag(f_dev, s_dev, out, fac=1.0)
    fd = cp.asnumpy(out).astype(np.float64)
    from_diag = ((fd[:-1] - fd[1:]) * dt
                 / (rho1_32.astype(np.float64)[None]
                    * thick32.astype(np.float64)))
    rel, err = _max_rel(from_diag, applied)
    print(f"vent diag reproduces the deposit: max_rel={rel:.3e} "
          f"max_abs={err:.3e}")
    assert rel <= 2e-6, (rel, err)


@requires_gpu
def test_flux_diag_diffusion_channel_is_the_solver_flux():
    """GATE: SASE_F*_DIFF recovered from the POST-SOLVE field equals the
    FP64 authority face flux (scale-relative <= 2e-6) AND its own
    convergence reproduces the solver's increment -- the statement that
    this is the flux the Thomas sweep used and not a nearby number.
    rho1 cancels out of the convergence, which is why both channels may
    share the deposit's single rho1 plane."""
    import cupy as cp
    from woof.core.sase import (launch_diff_flux_diag,
                                 launch_implicit_vertical_diffusion)
    from woof.verify.sase_ref import _face_average, box_filter

    nz, ny, nx = SHAPE
    rng = np.random.default_rng(SEED + 72)
    phi32 = box_filter(rng.standard_normal(SHAPE), 4).astype(np.float32)
    kv32 = (5.0 + 2.0 * box_filter(rng.standard_normal(SHAPE), 4)
            ).astype(np.float32)
    assert kv32.min() > 0.0
    thick32 = _per_column_dz32()
    rho1_32 = np.ascontiguousarray(
        rng.uniform(0.9, 1.2, (ny, nx)).astype(np.float32))
    dt, kfac = 400.0, 1.0 / 0.7429              # the driver's 1/Pr_t(f)

    phi_dev = cp.asarray(phi32)
    launch_implicit_vertical_diffusion(phi_dev, cp.asarray(kv32), dt=dt,
                                       kfac=kfac,
                                       dz_col=cp.asarray(thick32))
    phi_new = cp.asnumpy(phi_dev)
    out = cp.empty((nz + 1, ny, nx), dtype=cp.float32)
    got = launch_diff_flux_diag(phi_dev, cp.asarray(kv32),
                                cp.asarray(rho1_32), out, kfac=kfac,
                                fac=1.0, dz_col=cp.asarray(thick32))
    assert got is out
    flux = cp.asnumpy(out)

    # (a) end faces are the literal +0.0 (the shared interface contract).
    zeros2 = np.zeros((ny, nx), np.float32)
    assert flux[0].tobytes() == zeros2.tobytes()
    assert flux[nz].tobytes() == zeros2.tobytes()

    # (b) authority parity on the interior faces.
    t64 = thick32.astype(np.float64)
    h64 = 0.5 * (t64[:-1] + t64[1:])
    kf64 = kfac * _face_average(kv32.astype(np.float64))
    p64 = phi_new.astype(np.float64)
    ref = -(rho1_32.astype(np.float64)[None] * kf64
            * (p64[1:] - p64[:-1]) / h64)
    rel, err = _max_rel(flux[1:nz], ref)
    print(f"diffusion flux authority parity: max_rel={rel:.3e} "
          f"max_abs={err:.3e}")
    assert rel <= 2e-6, (rel, err)

    # (c) the convergence IS the solver's increment (zero-flux operator,
    # no surface deposit), so the recovery is exact rather than nearby.
    f64 = flux.astype(np.float64)
    conv = ((f64[:-1] - f64[1:]) * dt
            / (rho1_32.astype(np.float64)[None] * t64))
    inc = p64 - phi32.astype(np.float64)
    rel, err = _max_rel(conv, inc)
    print(f"diffusion flux convergence vs solver increment: "
          f"max_rel={rel:.3e} max_abs={err:.3e}")
    assert rel <= 2e-6, (rel, err)


@requires_gpu
def test_flux_diag_two_channels_sum_to_the_model_increment():
    """GATE: the two channels are exactly separable and exactly
    additive.  Reproducing the driver's registered deposit-then-solve
    order on one scalar row,

        (F_vent[k]-F_vent[k+1] + F_diff[k]-F_diff[k+1])*dt/(rho1*t_k)
            == phi_new[k] - phi_before[k]

    which is what turns "which channel carried the moisture" into an
    arithmetic statement rather than an opinion.  Both channels are read
    in the SAME kg m-2 s-1 currency over the SAME rho1 plane; a 3-D
    density would break this identity."""
    import cupy as cp
    from woof.core.sase import (launch_diff_flux_diag,
                                 launch_implicit_vertical_diffusion,
                                 launch_vent_deposit,
                                 launch_vent_deposit_scale,
                                 launch_vent_flux_diag)
    from woof.verify.sase_ref import box_filter

    nz, ny, nx = SHAPE
    rng = np.random.default_rng(SEED + 73)
    qv32 = (1.0e-2 + 1.0e-3 * box_filter(rng.standard_normal(SHAPE), 4)
            ).astype(np.float32)
    kv32 = (5.0 + 2.0 * box_filter(rng.standard_normal(SHAPE), 4)
            ).astype(np.float32)
    thick32 = _per_column_dz32()
    rho1_32 = np.ascontiguousarray(
        rng.uniform(0.9, 1.2, (ny, nx)).astype(np.float32))
    # A face-flux profile big enough that the registered cap BINDS on
    # some columns -- the scale is exactly what a raw-profile diagnostic
    # would get wrong, so it must be exercised here.
    f_qv = np.ascontiguousarray(
        (2.0e-3 * box_filter(rng.standard_normal((nz + 1, ny, nx)), 4)
         ).astype(np.float32))
    f_qv[0] = np.float32(0.0)
    f_qv[nz] = np.float32(0.0)
    f_zero = np.zeros((nz + 1, ny, nx), np.float32)
    dt, kfac = 60.0, 1.0 / 0.7429

    d_thick, d_rho1 = cp.asarray(thick32), cp.asarray(rho1_32)
    d_fqv, d_kv = cp.asarray(f_qv), cp.asarray(kv32)
    scale = launch_vent_deposit_scale(cp.asarray(f_zero), d_fqv,
                                      cp.asarray(f_zero), d_rho1, dt=dt,
                                      dz_col=d_thick)
    assert float(cp.asnumpy(scale).min()) < 1.0, (
        "the cap never binds on this fixture -- the scale leg is vacuous")

    phi = cp.asarray(qv32)
    before = cp.asnumpy(phi).astype(np.float64)
    launch_vent_deposit(phi, d_fqv, scale, d_rho1, dt=dt, dz_col=d_thick)
    launch_implicit_vertical_diffusion(phi, d_kv, dt=dt, kfac=kfac,
                                       dz_col=d_thick)
    after = cp.asnumpy(phi).astype(np.float64)

    vent_out = cp.empty((nz + 1, ny, nx), dtype=cp.float32)
    diff_out = cp.empty((nz + 1, ny, nx), dtype=cp.float32)
    launch_vent_flux_diag(d_fqv, scale, vent_out, fac=1.0)
    launch_diff_flux_diag(phi, d_kv, d_rho1, diff_out, kfac=kfac,
                          fac=1.0, dz_col=d_thick)
    total = (cp.asnumpy(vent_out).astype(np.float64)
             + cp.asnumpy(diff_out).astype(np.float64))
    conv = ((total[:-1] - total[1:]) * dt
            / (rho1_32.astype(np.float64)[None]
               * thick32.astype(np.float64)))
    rel, err = _max_rel(conv, after - before)
    # THE RIGHT BOUND HERE IS THE STATE'S OWN FP32 RESOLUTION, not a
    # relative gate on the increment.  ``after - before`` is a
    # difference of two FP32 fields whose values are O(1e-2), so the
    # increment it reports is quantized at ~1 ulp of 1e-2 no matter how
    # exact the flux split is; the increment itself is only O(2e-4) of
    # the field, so a relative metric divides a quantization floor by a
    # small number.  MEASURED this session: residual 1.284e-09 =
    # 1.4 ulp of the state (scale-relative 5.409e-06).  The FP64
    # authority-side statement of the same identity closes to 1.7e-14
    # relative, which is what says the split is exact and this is
    # storage noise.
    ulp = float(np.spacing(np.float32(np.abs(before).max())))
    print(f"vent + diffusion channels vs the model increment: "
          f"max_abs={err:.3e} = {err / ulp:.2f} FP32 ulp of the state; "
          f"max_rel={rel:.3e}")
    assert err <= 8.0 * ulp, (err, ulp, err / ulp)
    assert rel <= 5e-5, (rel, err)
    # Non-vacuity: BOTH channels are live, and neither is so much
    # smaller than the other that the identity would hold without it.
    vent_only = float(np.abs(cp.asnumpy(vent_out)).max())
    diff_only = float(np.abs(cp.asnumpy(diff_out)).max())
    print(f"channel magnitudes: |F_vent|max={vent_only:.3e} "
          f"|F_diff|max={diff_only:.3e} kg m-2 s-1")
    assert vent_only > 0.0 and diff_only > 0.0
    assert 1e-4 < vent_only / diff_only < 1e4


@requires_gpu
def test_flux_diag_vent_channel_is_bitwise_zero_where_m2_stands_down():
    """GATE: where the M2 limb stands down the vent channel is BITWISE
    +0.0 -- never a small number, never -0.0.  Read on the registered
    stand-down legs of the M2 identity contract (the LES limit f = 1 on
    a column that fires hard at f = 0, the M1 mask-off veto, and the
    unsaturated control column), with the firing column asserted
    non-vacuous first."""
    import cupy as cp
    from woof.core.sase import launch_vent_flux_diag
    from woof.verify.sase_ref import CP_AIR
    from test_sase import _vent_args

    dev, ref, thick = _vent_device(_vent_args("amp"))
    nz = dev["theta"].shape[0]
    ny, nx = dev["rho1"].shape
    scale = cp.ones((ny, nx), dtype=cp.float64)
    zeros = np.zeros((nz + 1, ny, nx), np.float32)

    def recorded(dev_args, f_blend):
        rows = _vent_launch(dev_args, thick, f_blend=f_blend,
                            indices=False)
        out = []
        for row, fac in ((1, 1.0), (0, float(CP_AIR))):
            buf = cp.empty((nz + 1, ny, nx), dtype=cp.float32)
            launch_vent_flux_diag(cp.asarray(rows[row]), scale, buf,
                                  fac=fac)
            out.append(cp.asnumpy(buf))
        return out

    live = recorded(dev, 0.0)
    assert max(float(np.abs(a).max()) for a in live) > 0.0, (
        "the firing fixture is vacuous")
    mask_off = dict(dev, n2_moist=dev["n2_dry"].copy())
    ctl, _cref, ctl_thick = _vent_device(_vent_args("ctl"))
    legs = (("LES limit f=1", recorded(dev, 1.0)),
            ("M1 mask off", recorded(mask_off, 0.0)))
    for label, got in legs:
        for arr in got:
            assert arr.tobytes() == zeros.tobytes(), label
    # The unsaturated control column stands down at step 1; its own
    # thicknesses, so it goes through the same helper independently.
    rows = _vent_launch(ctl, ctl_thick, f_blend=0.0, indices=False)
    nzc = ctl["theta"].shape[0]
    nyc, nxc = ctl["rho1"].shape
    zc = np.zeros((nzc + 1, nyc, nxc), np.float32)
    for row, fac in ((1, 1.0), (0, float(CP_AIR))):
        buf = cp.empty((nzc + 1, nyc, nxc), dtype=cp.float32)
        launch_vent_flux_diag(cp.asarray(rows[row]),
                              cp.ones((nyc, nxc), dtype=cp.float64), buf,
                              fac=fac)
        assert cp.asnumpy(buf).tobytes() == zc.tobytes(), "control column"


@requires_gpu
def test_flux_diag_enabled_leaves_every_prognostic_bitwise_identical():
    """THE INERTNESS GATE.  The same tiny SASE integration run with
    ``sase_flux_diag`` False and True, in one process, compared BYTE FOR
    BYTE across every serialized prognostic (``tobytes``, never
    ``allclose``).  A diagnostic that perturbs the state is worse than no
    diagnostic, so this is measured rather than argued.  The enabled run
    additionally has to produce the four fields with the registered
    shape/dtype/finiteness and a nonzero flux somewhere, so the gate
    cannot be passed by a diagnostic that writes nothing."""
    import cupy as cp

    from woof.config import RunConfig, validate_run_config
    from woof.core import dycore
    from woof.core.physics import initialize_physics
    from woof.io.restart import STATE_SERIALIZED_ATTRS

    def run(flux_diag, steps=6):
        cfg = RunConfig(nx=12, ny=10, nz=16, dx=2000.0, dy=2000.0,
                        ztop=8000.0, dt=5.0, run_seconds=5.0,
                        time_step_sound=4, moist=True, mp_physics=10,
                        ra_physics=4, sf_sfclay_physics=1,
                        sf_surface_physics=2,
                        km_opt=0, bl_pbl_physics=_SASE_SELECTOR,
                        sase_flux_diag=flux_diag)
        assert validate_run_config(cfg) is cfg
        state = _moist_driver_state(cfg)
        z_half = state.height_half()
        shear = (5.0 + 8.0 * z_half / cfg.ztop).astype(np.float32)
        state.u[...] = cp.asarray(np.broadcast_to(
            shear[:, None, None], (cfg.nz, cfg.ny, cfg.nx + 1)))
        state.v[...] = cp.float32(1.0)

        driver = initialize_physics(state, cfg, landmask=1.0, tsk=302.0,
                                    swdown=400.0, glw=320.0,
                                    radiation=_zero_radiation)
        for _ in range(steps):
            driver.compute(state, cfg)
            dycore.step(state, cfg)
        dump = {name: cp.asnumpy(getattr(state, name))
                for name in STATE_SERIALIZED_ATTRS
                if getattr(state, name, None) is not None}
        return cfg, driver, dump

    cfg_off, driver_off, off = run(False)
    cfg_on, driver_on, on = run(True)

    assert driver_off.sase_flux_diag is None
    assert driver_on.sase_flux_diag is not None
    assert set(off) == set(on) and len(off) >= 10
    for name in sorted(off):
        assert off[name].tobytes() == on[name].tobytes(), (
            f"{name}: the diagnostic perturbed the state")

    # The output seam gained EXACTLY the four names.
    added = set(driver_on.output_fields()) - set(driver_off.output_fields())
    assert added == {"SASE_FQV_VENT", "SASE_FQV_DIFF",
                     "SASE_FTH_VENT", "SASE_FTH_DIFF"}, added

    face = (cfg_on.nz + 1, cfg_on.ny, cfg_on.nx)
    fields = driver_on.output_fields()
    nonzero = 0
    for name in sorted(added):
        arr = fields[name]
        assert arr.shape == face, (name, arr.shape)
        assert arr.dtype == cp.float32, (name, arr.dtype)
        host = cp.asnumpy(arr)
        assert np.all(np.isfinite(host)), name
        # Face 0 and face nz carry the shared interface contract.
        assert not host[0].any() and not host[-1].any(), name
        nonzero += int(np.abs(host).max() > 0.0)
    assert nonzero >= 1, ("every recorded channel is identically zero -- "
                          "the gate would pass on a diagnostic that "
                          "writes nothing")
    # This idealized column never saturates, so the M2 limb stands down
    # and the VENT channels are legitimately +0.0 here (the firing case
    # is the unit gates above).  Pin that reading rather than leave it
    # ambiguous, and pin the name -> buffer mapping with sentinels so a
    # swapped pair cannot hide behind two zero fields.
    for name in ("SASE_FQV_VENT", "SASE_FTH_VENT"):
        host = cp.asnumpy(fields[name])
        assert host.tobytes() == np.zeros(face, np.float32).tobytes(), name
    for key, sentinel in (("fqv_vent", 1.0), ("fqv_diff", 2.0),
                          ("fth_vent", 3.0), ("fth_diff", 4.0)):
        driver_on.sase_flux_diag[key][...] = cp.float32(sentinel)
    marked = driver_on.output_fields()
    for name, sentinel in (("SASE_FQV_VENT", 1.0), ("SASE_FQV_DIFF", 2.0),
                           ("SASE_FTH_VENT", 3.0), ("SASE_FTH_DIFF", 4.0)):
        assert float(cp.asnumpy(marked[name]).min()) == sentinel, name
        assert float(cp.asnumpy(marked[name]).max()) == sentinel, name
    print(f"flux diag inertness: {len(off)} prognostics bitwise identical "
          f"over 6 steps; {nonzero} of 4 recorded channels nonzero "
          f"(the vent pair stands down on this column)")


@requires_gpu
@pytest.mark.parametrize("producer", ["sase", "smagorinsky"])
def test_hmix_k_diag_enabled_leaves_every_prognostic_bitwise_identical(
        producer):
    """THE INERTNESS GATE for the horizontal eddy-viscosity diagnostic,
    which is what admits ``hmix_k_diag`` to the output-only switches a
    restart and a prepared cache may see change.  The same tiny
    integration runs with the switch False and True, under each of the two
    producers it records, and every serialized prognostic is compared BYTE
    FOR BYTE.  The history seam is read every step in both arms so the
    copy out of the producer runs inside the integration it must not
    touch, and the enabled arm must record a nonzero viscosity so the gate
    cannot pass on a diagnostic that writes nothing."""
    import cupy as cp

    from woof.config import RunConfig, validate_run_config
    from woof.core import dycore
    from woof.core.physics import initialize_physics
    from woof.io.restart import STATE_SERIALIZED_ATTRS

    mixing = (dict(km_opt=0, bl_pbl_physics=_SASE_SELECTOR)
              if producer == "sase" else dict(km_opt=4, bl_pbl_physics=1))
    names = (("SASE_KMH", "SASE_KHH") if producer == "sase"
             else ("XKMH", "XKHH"))

    def run(enabled, steps=6):
        cfg = RunConfig(nx=12, ny=10, nz=16, dx=2000.0, dy=2000.0,
                        ztop=8000.0, dt=5.0, run_seconds=5.0,
                        time_step_sound=4, moist=True, mp_physics=10,
                        ra_physics=4, sf_sfclay_physics=1,
                        sf_surface_physics=2, hmix_k_diag=enabled,
                        **mixing)
        assert validate_run_config(cfg) is cfg
        state = _moist_driver_state(cfg)
        z_half = state.height_half()
        shear = (5.0 + 8.0 * z_half / cfg.ztop).astype(np.float32)
        # Horizontally varying wind: both producers scale with the
        # horizontal deformation, which a flow uniform in x and y has none of.
        rows = np.arange(cfg.ny, dtype=np.float64)
        columns = np.arange(cfg.nx, dtype=np.float64)
        u_wave = 2.0 * np.sin(2.0 * np.pi * rows / cfg.ny)
        v_wave = 1.5 * np.cos(2.0 * np.pi * columns / cfg.nx)
        state.u[...] = cp.asarray((
            shear[:, None, None] + u_wave[None, :, None]
            + np.zeros((1, 1, cfg.nx + 1))).astype(np.float32))
        state.v[...] = cp.asarray((
            1.0 + v_wave[None, None, :]
            + np.zeros((cfg.nz, cfg.ny + 1, 1))).astype(np.float32))

        driver = initialize_physics(state, cfg, landmask=1.0, tsk=302.0,
                                    swdown=400.0, glw=320.0,
                                    radiation=_zero_radiation)
        recorded = 0.0
        for _ in range(steps):
            driver.compute(state, cfg)
            dycore.step(state, cfg)
            fields = driver.output_fields()
            if enabled:
                recorded = max(recorded, max(
                    float(cp.abs(fields[name]).max()) for name in names))
        dump = {name: cp.asnumpy(getattr(state, name))
                for name in STATE_SERIALIZED_ATTRS
                if getattr(state, name, None) is not None}
        return driver, dump, recorded

    driver_off, off, _ = run(False)
    driver_on, on, recorded = run(True)

    assert driver_off.hmix_k_diag is None
    assert set(driver_on.hmix_k_diag) == set(names)
    assert set(off) == set(on) and len(off) >= 10
    for name in sorted(off):
        assert off[name].tobytes() == on[name].tobytes(), (
            f"{name}: the {producer} viscosity diagnostic perturbed the "
            f"state")
    assert recorded > 0.0, (
        f"the {producer} viscosity pair stayed zero for every step")
    print(f"hmix_k_diag inertness ({producer}): {len(off)} prognostics "
          f"bitwise identical over 6 steps; max recorded viscosity "
          f"{recorded:.3g} m2 s-1")


# ---------------------------------------------------------------------------
# S3-6k: decoupled stable-limb DISSIPATION coefficient on device
# (RunConfig sase_stable_dissipation -> launch_sase_step
# stable_dissipation -> the sase_split_e_update has_ces gate).
# ---------------------------------------------------------------------------


@requires_gpu
def test_sase_step_stable_dissipation_device_gate_and_masks():
    """THE DEVICE GATE, three pins, all BITWISE (tobytes, never
    allclose):

    * DEFAULT INERTNESS: the kwarg absent and the kwarg explicitly False
      step identically -- ``has_ces == 0`` leaves the kernel's decay
      multiplicand the literal ``c_e``, so nothing is added, not even
      ``*1.0``;
    * UNSTRATIFIED INERTNESS WITH THE SWITCH ON: a frame whose n2 is
      non-positive everywhere steps identically on both legs.  Those are
      exactly the cells the M1 substitution drives negative, so the
      device seam is provably absent from the cloud-top
      over-entrainment defect;
    * THE SEAM ACTUALLY FIRES on the mixed-sign frame, and in the
      direction the formulation predicts: cutting the decay coefficient
      leaves at least as much e in every cell and strictly more
      somewhere."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    u32, v32, w32, th32, e32, n232 = _step_fixture32(seed=SEED + 31)

    def dev():
        return [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]

    kw = dict(dx=DX, dy=DY, dz=DZ, delta=DELTA, dt=STEP_DT)
    n2_dev = cp.asarray(n232)

    def leg(n2_host, **extra):
        fields = dev()
        heat = cp.empty(STEP_SHAPE, dtype=cp.float32)
        led = launch_sase_step(*fields, n2=cp.asarray(n2_host), heat=heat,
                               **kw, **extra)
        return fields, cp.asnumpy(heat), led

    # 1. absent vs explicitly False
    a_f, a_heat, a_led = leg(n232)
    b_f, b_heat, b_led = leg(n232, stable_dissipation=False)
    for name, xa, xb in zip("uvwte", a_f, b_f):
        assert cp.asnumpy(xa).tobytes() == cp.asnumpy(xb).tobytes(), name
    assert a_heat.tobytes() == b_heat.tobytes()
    assert (cp.asnumpy(a_led["kv"]).tobytes()
            == cp.asnumpy(b_led["kv"]).tobytes())
    scal = lambda d: {k: v for k, v in d.items()
                      if not isinstance(v, cp.ndarray)}
    assert scal(a_led) == scal(b_led)

    # 2. unstratified frame: the switch is bitwise absent
    n2_neg = -np.abs(n232).astype(np.float32)
    assert np.all(n2_neg <= 0.0)
    c_f, c_heat, _ = leg(n2_neg)
    d_f, d_heat, _ = leg(n2_neg, stable_dissipation=True)
    for name, xc, xd in zip("uvwte", c_f, d_f):
        assert cp.asnumpy(xc).tobytes() == cp.asnumpy(xd).tobytes(), (
            f"the switch moved {name} in a column with no stable cell")
    assert c_heat.tobytes() == d_heat.tobytes()

    # 3. mixed-sign frame: the seam fires, in the predicted direction
    e_f, _, _ = leg(n232, stable_dissipation=True)
    off_e = cp.asnumpy(a_f[4])
    on_e = cp.asnumpy(e_f[4])
    assert on_e.tobytes() != off_e.tobytes(), "the device seam never fired"
    assert np.all(on_e >= off_e)
    assert np.any(on_e > off_e)
    moved = int(np.count_nonzero(on_e != off_e))
    print(f"S3-6k device seam: {moved} of {on_e.size} cells moved, "
          f"max de = {float(np.max(on_e - off_e)):.3e}")


@requires_gpu
def test_sase_step_stable_dissipation_gate_matches_authority_unstratified():
    """CPU/DEVICE STRUCTURAL AGREEMENT on the cells S3-6k must not touch.

    The kernel gate is ``if (has_ces && ls_v > 0.0f)``: where no
    stability length exists the decay multiplicand stays the LITERAL
    ``(double)c_e``, i.e. exactly the has_ces == 0 multiplicand.  The
    FP64 authority has to make the same SELECTION.  It cannot get there
    by cancellation -- at n2 <= 0 an unconditional outer blend reduces
    to f*C_E + (1-f)*C_E, and that expression misses C_E by one ulp at
    36.6% of f in [0, 1] (measured over a uniform 10001-point grid).
    The first S3-6k commit (fb67b9d) shipped the unconditional form, so
    the authority and its mirror disagreed by 1 ulp precisely where the
    amendment is documented to be absent.

    The fixture ASSERTS ITS OWN TEETH: both engines' solved f must be
    UNSAFE values, else the comparison passes for free.  The frames the
    gate test above uses happen to solve safe f, which is why that test
    -- device-only and on a safe f -- could not see this."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    from woof.verify import sase_ref
    from woof.verify.sase_ref import sase_split_step
    u32, v32, w32, th32, e32, n232 = _step_fixture32(seed=SEED + 29)
    n2_neg = -np.abs(n232).astype(np.float32)
    assert np.all(n2_neg < 0.0)
    kw = dict(dx=DX, dy=DY, dz=DZ, delta=DELTA, dt=STEP_DT)

    def unsafe(f):
        blend = float(f) * sase_ref.C_E + (1.0 - float(f)) * sase_ref.C_E
        return (np.float64(blend).tobytes()
                != np.float64(sase_ref.C_E).tobytes())

    def device_leg(flag):
        dev = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
        heat = cp.empty(STEP_SHAPE, dtype=cp.float32)
        led = launch_sase_step(*dev, n2=cp.asarray(n2_neg), heat=heat,
                               stable_dissipation=flag, **kw)
        return ([cp.asnumpy(a) for a in dev], cp.asnumpy(heat), led)

    def host_leg(flag):
        ref = [a.astype(np.float64) for a in (u32, v32, w32, th32, e32)]
        return sase_split_step(*ref[:4], ref[4],
                               n2=n2_neg.astype(np.float64),
                               stable_dissipation=flag, **kw)

    d_off, h_off, led_off = device_leg(False)
    d_on, h_on, led_on = device_leg(True)
    c_off, cled_off = host_leg(False)
    c_on, cled_on = host_leg(True)
    assert unsafe(led_on["f"]), (
        f"device f = {led_on['f']!r} is a safe value -- this pin needs "
        "an unsafe one to have teeth")
    assert unsafe(cled_on["f"]), (
        f"authority f = {cled_on['f']!r} is a safe value -- this pin "
        "needs an unsafe one to have teeth")
    for name, xa, xb in zip("uvwte", d_off, d_on):
        assert xa.tobytes() == xb.tobytes(), f"device moved {name}"
    assert h_off.tobytes() == h_on.tobytes(), "device moved heat"
    for name in ("u", "v", "w", "e", "heat"):
        assert c_off[name].tobytes() == c_on[name].tobytes(), (
            f"the FP64 authority moved {name} where the device did not "
            "-- the neutral/unstable gate is not mirrored")
    print("S3-6k gate mirror: device f = %.17g, authority f = %.17g, "
          "both unsafe; both engines bitwise inert on n2 < 0"
          % (led_on["f"], cled_on["f"]))


@requires_gpu
def test_sase_step_stable_dissipation_authority_parity():
    """CPU/DEVICE AGREEMENT with the switch ON: one device step against
    the FP64 authority stepped from the SAME FP32-cast inputs, at the
    established split-step trajectory tier (5e-5 scale-relative).  This
    one is allclose-tier on purpose -- it is a cross-precision check,
    not an inertness proof (the bitwise contracts are the gate above).
    The OFF leg rides along as the control, and the two legs are
    asserted DIFFERENT on both engines so the comparison cannot be
    passed by a switch that does nothing."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    from woof.verify.sase_ref import sase_split_step
    u32, v32, w32, th32, e32, n232 = _step_fixture32(seed=SEED + 32)
    ref64 = [a.astype(np.float64) for a in (u32, v32, w32, th32, e32)]
    n264 = n232.astype(np.float64)
    kw = dict(dx=DX, dy=DY, dz=DZ, delta=DELTA, dt=STEP_DT)
    worst = {}
    got = {}
    for flag in (False, True):
        dev = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
        launch_sase_step(*dev, n2=cp.asarray(n232),
                         stable_dissipation=flag, **kw)
        fields, _ = sase_split_step(*ref64[:4], ref64[4], n2=n264,
                                    stable_dissipation=flag, **kw)
        rels = {n: _max_rel(cp.asnumpy(d), fields[n])[0]
                for n, d in zip("uvwe", (dev[0], dev[1], dev[2], dev[4]))}
        worst[flag] = max(rels.values())
        got[flag] = (cp.asnumpy(dev[4]), fields["e"])
        print(f"S3-6k parity (stable_dissipation={flag}): " + "  ".join(
            f"{n}={r:.3e}" for n, r in rels.items()))
    assert worst[True] <= 5e-5, worst
    assert worst[False] <= 5e-5, worst
    # The switch is not a no-op on EITHER engine.
    assert got[True][0].tobytes() != got[False][0].tobytes()
    assert got[True][1].tobytes() != got[False][1].tobytes()


@requires_gpu
def test_vanishing_friction_velocity_does_not_saturate_or_go_non_finite():
    """The trap this closure must not repeat.

    A sibling scheme in this tree carries a latent defect of a very
    specific shape on CuPy's compile route: it forms u*^3 in FP32, and
    that route flushes FP32 subnormals to zero in every arithmetic
    mechanism the FTZ receipt measures (the loader passes no ftz flag, and
    CuPy's appended -ftz=true is what makes that route flush).  Below
    u* ~ 1e-13 the cube underflows to exactly zero, a later ratio
    becomes inf/inf = NaN, and a min() against a ceiling returns the
    NON-NaN operand -- so the exchange coefficient silently SATURATES at
    its ceiling instead of failing.  Silent saturation is worse than a
    crash: nothing in the run says anything is wrong.

    Two properties keep SASE out of that shape, and this test pins both
    rather than trusting a reading of the source:

    (a) the surface energy source's u*^3 is evaluated where a vanishing
        u* must give a vanishing shear production and nothing else --
        no ratio is formed from it, so there is no inf/inf to make;
    (b) the implicit surface drag divides by WIND SPEED, floored at
        SFC_WSPD_FLOOR, never by u*.

    The sweep walks u* from an ordinary value down through the FP32
    subnormal cliff to exactly zero and requires the result to stay
    finite, to stay monotone, and -- the saturation check -- never to
    sit at a constant while the input is still changing.
    """
    import cupy as cp

    from woof.core.physics import sase_surface_e_source
    from woof.verify.sase_ref import SFC_WSPD_FLOOR

    shape = (4, 5)
    ones = cp.ones(shape, dtype=cp.float32)
    # Straddles the FP32 subnormal boundary: u*^3 underflows below about
    # 1e-13, and 0.0 is the exact endpoint.
    ladder = [0.3, 1e-2, 1e-4, 1e-6, 1e-9, 1e-12, 1e-13, 1e-14, 1e-20, 0.0]
    produced = []
    for ust in ladder:
        source = sase_surface_e_source(
            ust=cp.full(shape, ust, dtype=cp.float32),
            hfx=cp.zeros(shape, dtype=cp.float32),      # shear term alone
            qfx=cp.zeros(shape, dtype=cp.float32),
            theta1=300.0 * ones, qv1=cp.zeros(shape, dtype=cp.float32),
            p1=95000.0 * ones, t1=290.0 * ones, dz1=50.0 * ones)
        value = cp.asnumpy(source)
        assert np.all(np.isfinite(value)), f"non-finite at u*={ust}"
        assert np.all(value >= 0.0), f"negative production at u*={ust}"
        produced.append(float(value.flat[0]))

    # Monotone down to zero, and ZERO at zero -- not a floor, not a
    # ceiling, not a NaN that a later min() would hide.
    assert produced[-1] == 0.0
    for earlier, later in zip(produced, produced[1:]):
        assert later <= earlier, produced
    # The saturation check.  A silently saturating path returns the SAME
    # number across the cliff; a correct one keeps shrinking until it
    # underflows to zero.  Requiring strict decrease over the ordinary
    # decades is what would have caught the sibling defect.
    assert produced[0] > produced[1] > produced[2] > produced[3] > 0.0

    # (b) the drag conductance divides by a FLOORED wind speed, so the
    # floor -- not u* -- bounds it, and it is FP64 on device.
    assert SFC_WSPD_FLOOR > 0.0
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "woof" / "core" / "kernels" / "sase.cu").read_text()
    assert "double spd1 = fmax(hypot(u1, v1), sfc_floor);" in src, (
        "the surface drag row must keep its FP64 floored-speed form")
    assert "__fdividef" not in src and "rsqrtf" not in src, (
        "no fast-math intrinsic may enter this kernel: they change the "
        "underflow behaviour the checks above depend on")


@pytest.mark.gpu
@requires_gpu
def test_sase_nan_guard_names_the_domain_and_the_degenerate_producer():
    """R0: the closure's numerics guard, and the three claims it makes.

    The PBL-slot guard the other schemes carry answers one question --
    "which rate went non-finite" -- and that answer sends a reader to
    the closure's source when the cause is almost always upstream of
    it: a surface layer that handed over a zero friction velocity, a
    land-surface scheme that handed over a non-finite heat flux, a
    collapsed first layer.  So this refusal makes THREE claims, and all
    three are essential:

      1. the rate that failed,
      2. the DOMAIN it failed on -- a nest that diverges while its
         parent stays healthy is a different bug from a run-wide one,
         and a message without a grid id cannot tell them apart,
      3. which PRODUCER INPUTS were already degenerate before the
         closure touched them -- or, when none were, that none were.

    THREE MUTATION CONTROLS run inline below, each a guard wrong in
    exactly one way, each WATCHED failing the claim that catches it.  A
    control that does not fail here is a claim this test is not really
    making.

    The healthy path is also pinned: finite rates raise nothing and
    leave every array bitwise untouched, because a guard that quietly
    repaired its inputs would be a physics change wearing a check's
    clothes.
    """
    import cupy as cp

    from woof.core.physics import validate_sase_tendencies

    shape_3d, shape_2d = (4, 3, 5), (3, 5)
    rates = {
        "du": cp.full(shape_3d, 1.0e-3, cp.float32),
        "dv": cp.full(shape_3d, -2.0e-3, cp.float32),
        "dtheta": cp.full(shape_3d, 4.0e-4, cp.float32),
        "dqv": cp.full(shape_3d, 1.0e-7, cp.float32),
        "dw": cp.full(shape_3d, 3.0e-5, cp.float32),
    }
    healthy = {
        "ust": cp.full(shape_2d, 0.28, cp.float32),
        "wspd": cp.full(shape_2d, 4.5, cp.float32),
        "hfx": cp.full(shape_2d, 120.0, cp.float32),
        "qfx": cp.full(shape_2d, 1.0e-4, cp.float32),
        "dz1": cp.full(shape_2d, 40.0, cp.float32),
        "p1": cp.full(shape_2d, 95000.0, cp.float32),
        "t1": cp.full(shape_2d, 291.0, cp.float32),
    }

    # -- the healthy path changes nothing -------------------------------
    before = {name: value.copy() for name, value in rates.items()}
    validate_sase_tendencies(rates, grid_id=2, producer_inputs=healthy)
    for name, value in rates.items():
        cp.testing.assert_array_equal(value, before[name])

    # -- a degenerate producer is named, with its count ------------------
    degenerate = {name: value.copy() for name, value in healthy.items()}
    degenerate["ust"][0, 0] = cp.float32(0.0)     # the documented failure
    degenerate["hfx"][1, 2] = cp.nan
    broken = {name: value.copy() for name, value in rates.items()}
    broken["du"][0, 0, 0] = cp.nan
    with pytest.raises(FloatingPointError) as raised:
        validate_sase_tendencies(
            broken, grid_id=3, producer_inputs=degenerate)
    message = str(raised.value)

    def claims(text):
        """The three claims, evaluated independently."""
        return (
            "SASE returned non-finite du tendency" in text,
            "on domain 3" in text,
            "ust (1 <= 0)" in text and "hfx (1 non-finite)" in text,
        )

    assert claims(message) == (True, True, True), message
    # A NaN cell is reported ONCE, as non-finite -- ``<= 0`` is False at
    # NaN by IEEE rule, so hfx must not also be counted non-positive.
    assert "hfx (1 non-finite, " not in message, message
    # Inputs the closure only needs to be finite are not accused of a
    # sign problem they cannot have.
    assert "wspd" not in message and "qfx" not in message, message

    # -- clean producers say so, rather than saying nothing --------------
    with pytest.raises(FloatingPointError) as raised_clean:
        validate_sase_tendencies(broken, grid_id=3, producer_inputs=healthy)
    clean_message = str(raised_clean.value)
    assert "closure's own arithmetic produced it" in clean_message
    assert "already degenerate" not in clean_message

    # -- MUTATION CONTROLS, each watched failing ------------------------
    def mutant_rate_only(text):
        return "SASE returned non-finite du tendency"

    def mutant_no_forensics(text):
        return "SASE returned non-finite du tendency on domain 3"

    def mutant_always_blames(text):
        return ("SASE returned non-finite du tendency on domain 3"
                "; producer inputs already degenerate: ust (1 <= 0), "
                "hfx (1 non-finite)")

    fired = []
    # M1 drops the domain: claim 2 and claim 3 must both go False.
    m1 = claims(mutant_rate_only(message))
    assert m1 == (True, False, False), m1
    fired.append("M1")
    # M2 keeps the domain and drops the forensics: claim 3 alone falls.
    m2 = claims(mutant_no_forensics(message))
    assert m2 == (True, True, False), m2
    fired.append("M2")
    # M3 blames the producer unconditionally.  It survives the
    # degenerate case -- which is exactly why the CLEAN case above is a
    # separate assertion: that is the check M3 fails.
    assert claims(mutant_always_blames(message)) == (True, True, True)
    m3_clean = mutant_always_blames(clean_message)
    assert "closure's own arithmetic produced it" not in m3_clean
    assert "already degenerate" in m3_clean
    fired.append("M3")
    assert fired == ["M1", "M2", "M3"]

    # -- the seam stays wired -------------------------------------------
    # The forensic form is worth nothing if the driver stops handing it
    # the two arguments, and that is a silent regression: the guard
    # still runs, still refuses, and just stops being able to say where.
    from pathlib import Path

    driver = (Path(__file__).resolve().parents[1]
              / "woof" / "core" / "physics.py").read_text()
    # rsplit: the LAST occurrence is the driver's call, the first is the
    # definition.
    call = driver.rsplit("validate_sase_tendencies(\n", 1)[1][:700]
    assert "grid_id=cfg.grid_id" in call, call
    assert "producer_inputs={" in call, call
    for name in ("ust", "wspd", "hfx", "qfx", "dz1", "p1", "t1"):
        assert f'"{name}"' in call, (name, call)


# ---------------------------------------------------------------------------
# S3-12: ADDITIVE e^{3/2} DISSIPATION CHANNEL on device (RunConfig
# sase_additive_dissipation -> launch_sase_step additive_dissipation ->
# the sase_split_e_update has_ced gate + the sase_blackadar_length
# reference field).  Parity is scored in float32 ULP at the
# kernel-arithmetic tier (woof.core.fp32_ulp), the Noah-MP harness
# convention: the measured value on this hardware is MAX ULP 0 on every
# leg, and that measurement is the pinned gate.  Both negative controls
# below are proven to fire (10^4-ULP class), so a wrong constant or the
# wrong reference-length member cannot pass silently.
# ---------------------------------------------------------------------------


def _ulp32(got, want):
    from woof.core.fp32_ulp import fp32_ulp_distance
    return fp32_ulp_distance(np.asarray(got, np.float32),
                             np.asarray(want, np.float32))


@requires_gpu
def test_blackadar_length_device_bitwise_and_z_convention_control():
    """S3-12 geometry kernel: ``sase_blackadar_length`` must store the
    EXACT FP32 image of the authority's FP64 l_B(z + z0) -- max ULP 0,
    all three thickness modes -- because the kernel transcribes the
    authority's own op order (uniform z as the product (k+0.5)*dz,
    thickness modes as cumsum-then-half-layer-subtraction), NOT the
    vertical-channel kernel's half-step accumulation, whose last-ulp
    drift the 2e-6 gates absorb but a bitwise pin would see.

    NEGATIVE CONTROL, proven to fire: a mirror built on the INTERFACE
    heights (cumsum without the half-layer subtraction) -- the z
    convention a careless transcription lands on -- must differ on
    every cell, so this pin demonstrably distinguishes the two."""
    import cupy as cp
    from woof.core.sase import launch_blackadar_length
    from woof.verify.sase_ref import _blackadar_length

    nz, ny, nx = 24, 8, 12
    shape = (nz, ny, nx)
    dz = 37.5                                     # FP32-exact
    z0 = 0.1
    rng = np.random.default_rng(SEED + 41)
    t32 = _stretched_dz(nz, 20.0).astype(np.float32)
    t64 = t32.astype(np.float64)
    tf32 = (t32[:, None, None]
            * (1.0 + 0.05 * rng.standard_normal((1, ny, nx)))
            ).astype(np.float32)
    tf64 = tf32.astype(np.float64)
    legs = (
        ("BOX", dict(dz=dz),
         (np.arange(nz, dtype=np.float64) + 0.5) * dz,
         np.broadcast_to(
             np.full((nz, 1, 1), dz).cumsum(axis=0), shape)),
        ("COL", dict(dz_col=t64),
         np.cumsum(t64) - 0.5 * t64,
         np.broadcast_to(np.cumsum(t64)[:, None, None], shape)),
        ("FIELD", dict(dz_col=cp.asarray(tf32)),
         np.cumsum(tf64, axis=0) - 0.5 * tf64,
         np.cumsum(np.broadcast_to(tf64, shape), axis=0)),
    )
    for label, kw, z, z_if in legs:
        got = cp.asnumpy(launch_blackadar_length(shape, z0=z0, **kw))
        want = np.broadcast_to(
            np.asarray(_blackadar_length(z, z0), np.float64).reshape(
                (nz, 1, 1) if z.ndim == 1 else z.shape),
            shape).astype(np.float32)
        worst = int(_ulp32(got, want).max())
        print(f"{label}: l_B max ulp = {worst}")
        assert worst == 0, (label, worst)
        # the z-convention control: interface heights must NOT pass
        wrong = np.asarray(_blackadar_length(z_if, z0),
                           np.float64).astype(np.float32)
        fired = int((_ulp32(got, wrong) > 0).sum())
        assert fired == got.size, (
            f"{label}: the interface-height mirror agrees on "
            f"{got.size - fired} cells -- the pin cannot see the z "
            "convention")


@requires_gpu
def test_split_e_update_additive_channel_kernel_ulp_and_controls():
    """S3-12 kernel-arithmetic-tier parity, scored in float32 ULP.

    The kernel is driven directly with zeroed source/transport fields
    and uniform theta (e* == e^n bitwise; the decay substep is the only
    live channel -- the S3-9b fixture idiom), against an INDEPENDENTLY
    WRITTEN host mirror of the amended arithmetic: FP32 l_d with the
    endpoint-branched geometric blend and the l_s min, then the FP64
    coefficient C_eps = C_E + (1-f)*w*C_ED*(l_d/l_ref) on the SELECTED
    stable cells with l_ref endpoint-branched over the FP32 l_B field,
    the FP64 analytic decay, one FP32 rounding at the store, the FP32
    E_MIN clip.  dt is a power of two so b*dt is exact and FMA
    contraction cannot skew the mirror.

    THE REGIME FIXTURE covers what the receipts name as live:
    three whole planes at the measured 12.2 km stability N^2 = 3.566e-5
    with e at the registered Ri = 0.12/0.14/0.16 fixed-point amplitudes
    (0.96 / 0.21 / 0.006 m2/s2 -- the band where the census found 72%
    of live cells), a weak-stability plane (rho < 1, the taper), a
    neutral and an unstable plane (the selection), band-limited
    mixed-sign planes, and sub-floor e cells (the clip channel).

    GATE: max ULP 0 for e AND heat on every leg -- f = 0 (pure RANS),
    the recorded production f = 4.1188928660938e-05, interior f = 0.4
    (both device pows live), and f = 1 (LES limb, which must ALSO be
    bitwise the has_ced == 0 result).  0 is the MEASURED value on this
    hardware (RTX 5090), not an aspiration: the FP64 pow ulp
    differences between device libm and numpy sit ~2^29 below the FP32
    store's rounding step and are absorbed (the S3-9b argument, here
    re-measured through the new channel).

    TWO NEGATIVE CONTROLS, each proven to fire at the 10^3-ULP class or
    better on the engineered stable cells:
    (A) a mirror whose reference length is the state-DEPENDENT leps
        member -- the exact trap the amendment exists to avoid (an
        e-linear channel dressed as e^{3/2}) -- must miss;
    (B) a channel-off mirror must miss wherever the channel fires (and
        must MATCH at f = 1, which is the LES-limb inertness measured
        from the other side)."""
    import cupy as cp
    import woof.core.sase as sase
    from woof.core.sase import _kern
    from woof.core.state import DTYPE
    from woof.verify.sase_ref import (C_E, C_ED, C_ES, CKS_BLEND_EXP,
                                       E_MIN, G_ACCEL, LS_COEF,
                                       _blackadar_length, box_filter)

    nz, ny, nx = 8, 8, 16
    shape = (nz, ny, nx)
    rng = np.random.default_rng(SEED + 42)
    e32 = np.maximum(0.05 + 0.4 * box_filter(rng.standard_normal(shape), 4),
                     0.0).astype(np.float32)
    e32[0, :2, :2] = 0.0                          # E_MIN clip cells live
    e32[3] = np.float32(0.96)                     # Ri ~ 0.12 amplitude
    e32[4] = np.float32(0.21)                     # Ri ~ 0.14 amplitude
    e32[5] = np.float32(0.006)                    # Ri ~ 0.16 amplitude
    n232 = np.empty(shape, np.float32)
    n232[:2] = (1e-4 * box_filter(rng.standard_normal(shape), 4)
                ).astype(np.float32)[:2]
    n232[2] = np.float32(0.0)                     # neutral: selection
    n232[3:6] = np.float32(3.566e-5)              # the live stable band
    n232[6] = np.float32(-5.0e-5)                 # unstable: selection
    n232[7] = np.float32(1.0e-7)                  # weak: rho < 1 taper
    z1 = (np.arange(nz) + 0.5) * 25.0
    lb32 = (_blackadar_length(z1, 0.1)[:, None, None]
            * np.ones(shape)).astype(np.float32)
    leps32 = np.minimum(lb32, (5.0 + 30.0 * np.abs(
        box_filter(rng.standard_normal(shape), 4))).astype(np.float32))
    assert np.any(leps32 != lb32)                 # control A has teeth
    delta, dt, pr_t = 500.0, 0.5, 0.7
    ncell = nz * ny * nx
    nblocks = (ncell + sase._TPB - 1) // sase._TPB

    def run_device(f, has_ced):
        e_dev = cp.asarray(e32)
        heat = cp.empty(shape, dtype=DTYPE)
        theta = cp.full(shape, 300.0, dtype=DTYPE)
        zeros = cp.zeros(shape, dtype=DTYPE)
        zcm, zc0, zcp, dz_s, two_dz, h_lo, h_hi, z_mode = \
            sase._z_stencil(shape, dz=DZ)
        partials = cp.zeros((4, nblocks), dtype=cp.float64)
        lepsd = cp.asarray(leps32)
        _kern("sase_split_e_update")(
            (nblocks,), (sase._TPB,),
            (e_dev, heat, theta, cp.asarray(n232), np.int32(1),
             e_dev, np.int32(0),                 # gated M1 dry-n2 dummy
             zeros, lepsd,
             cp.asarray(lb32), np.int32(has_ced),
             zeros, zeros, zeros,
             zeros, zeros, e_dev, np.int32(0),   # gated taper dummy
             partials, zcm, zc0, zcp,
             DTYPE(2.0 * DX), DTYPE(2.0 * DY), dz_s, two_dz,
             h_lo, h_hi, z_mode, DTYPE(dt), DTYPE(f), DTYPE(delta),
             DTYPE(pr_t), DTYPE(C_E), DTYPE(LS_COEF),
             DTYPE(C_ES), DTYPE(CKS_BLEND_EXP), np.int32(0),
             DTYPE(C_ED),
             DTYPE(G_ACCEL),
             DTYPE(E_MIN), np.int32(nblocks),
             np.int32(nz), np.int32(ny), np.int32(nx)))
        return cp.asnumpy(e_dev), cp.asnumpy(heat)

    def mirror(f, c_ed, ref32):
        """Independently written host mirror (docstring); ``c_ed=0``
        is the channel-off mirror, ``ref32`` the reference-length
        field the l_ref blend rides."""
        e_minf = np.float32(E_MIN)
        ec = np.maximum(e32, e_minf)
        root_e = np.sqrt(ec)                     # sqrtf
        f32 = np.float32(f)
        if f32 == np.float32(0.0):
            l = leps32.copy()
        elif f32 == np.float32(1.0):
            l = np.full_like(leps32, np.float32(delta))
        else:
            l = (np.float64(np.float32(delta)) ** np.float64(f32)
                 * leps32.astype(np.float64)
                 ** (1.0 - np.float64(f32))).astype(np.float32)
        ls_v = np.zeros_like(l)
        pos = n232 > np.float32(0.0)
        ls_v[pos] = (np.float32(LS_COEF) * root_e[pos]) / np.sqrt(n232[pos])
        l[pos] = np.minimum(l[pos], ls_v[pos])
        es64 = e32.astype(np.float64)            # e* == e^n
        c_eps = np.full(shape, np.float64(np.float32(C_E)))
        live = ls_v > np.float32(0.0)
        if c_ed != 0.0:
            rho = np.minimum(l[live].astype(np.float64)
                             / ls_v[live].astype(np.float64), 1.0)
            w = rho ** np.float64(np.float32(CKS_BLEND_EXP))
            if f32 == np.float32(0.0):
                lref = ref32[live].astype(np.float64)
            elif f32 == np.float32(1.0):
                lref = np.float64(np.float32(delta))
            else:
                lref = (np.float64(np.float32(delta)) ** np.float64(f32)
                        * ref32[live].astype(np.float64)
                        ** (1.0 - np.float64(f32)))
            c_eps[live] = c_eps[live] + (
                (1.0 - np.float64(f32)) * w
                * np.float64(np.float32(c_ed))
                * (l[live].astype(np.float64) / lref))
        b = (c_eps * np.sqrt(np.maximum(es64, np.float64(e_minf)))
             / (2.0 * l.astype(np.float64)))
        fac = 1.0 + b * np.float64(np.float32(dt))
        e_dec = (es64 / (fac * fac)).astype(np.float32)
        decay = e32 - e_dec                      # e* == e^n
        e_clip = np.maximum(e_dec, e_minf)
        heat = decay - (e_clip - e_dec)
        return e_clip, heat, live

    for f in (0.0, 4.1188928660938e-05, 0.4, 1.0):
        got_e, got_h = run_device(f, 1)
        want_e, want_h, live = mirror(f, C_ED, lb32)
        worst_e = int(_ulp32(got_e, want_e).max())
        worst_h = int(_ulp32(got_h, want_h).max())
        # Negative control B: the channel-off mirror.
        ctrl_b = _ulp32(got_e, mirror(f, 0.0, lb32)[0])
        # Negative control A: the state-dependent reference member.
        ctrl_a = _ulp32(got_e, mirror(f, C_ED, leps32)[0])
        print(f"f={f}: e ulp={worst_e} heat ulp={worst_h} "
              f"live={int(live.sum())}  ctrl-B max={int(ctrl_b.max())} "
              f"fired={int((ctrl_b > 0).sum())}  "
              f"ctrl-A max={int(ctrl_a.max())} "
              f"fired={int((ctrl_a > 0).sum())}")
        assert worst_e == 0, (f, worst_e)         # the MEASURED gate
        assert worst_h == 0, (f, worst_h)
        assert int(live.sum()) >= 500             # the fixture has teeth
        if np.float32(f) != np.float32(1.0):
            # Both controls fire, at the 10^3-ULP class or better --
            # a wrong constant or the wrong length member CANNOT pass.
            assert int(ctrl_b.max()) > 1000, int(ctrl_b.max())
            assert int((ctrl_b > 0).sum()) > 400
            assert int(ctrl_a.max()) > 1000, int(ctrl_a.max())
            assert int((ctrl_a > 0).sum()) > 400
        else:
            # LES limb: the channel is arithmetic +0.0 -- the ON run,
            # the OFF mirror and the wrong-member mirror all coincide,
            # and the ON run is bitwise the has_ced == 0 kernel.
            assert int(ctrl_b.max()) == 0
            assert int(ctrl_a.max()) == 0
            off_e, off_h = run_device(f, 0)
            np.testing.assert_array_equal(got_e, off_e)
            np.testing.assert_array_equal(got_h, off_h)
    # The OFF path is bitwise the pre-S3-12 kernel (channel-off mirror).
    got_e, got_h = run_device(0.4, 0)
    off_e, off_h, _ = mirror(0.4, 0.0, lb32)
    np.testing.assert_array_equal(got_e, off_e)
    np.testing.assert_array_equal(got_h, off_h)


@requires_gpu
def test_sase_step_additive_dissipation_device_gate_and_masks():
    """THE S3-12 DEVICE GATE, the S3-6k pin set extended, all BITWISE
    (tobytes, never allclose):

    * DEFAULT INERTNESS: the kwarg absent and the kwarg explicitly
      False step identically -- ``has_ced == 0`` adds nothing, not even
      +0.0, and no l_B field is launched;
    * UNSTRATIFIED INERTNESS WITH THE SWITCH ON: a frame whose n2 is
      non-positive everywhere steps identically on both legs (the
      shared ``ls_v > 0.0f`` SELECTION -- the same cells the M1
      substitution drives negative, so the channel is provably absent
      from the cloud-top over-entrainment defect);
    * n2 = None INERTNESS WITH THE SWITCH ON: the test-box path has no
      stability machinery at all, and the channel must be bitwise
      absent there too (the authority's ``n2 is None`` selection);
    * THE SEAM FIRES on the mixed-sign frame, in the OPPOSITE direction
      from S3-6k -- strictly MORE dissipation: e nowhere higher and
      strictly lower somewhere (the nowhere-weaker property, device
      side);
    * COMPOSITION: both switches ON differs bitwise from each switch
      alone (S3-6k selects the base, S3-12 adds to it -- two seams, not
      one)."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    u32, v32, w32, th32, e32, n232 = _step_fixture32(seed=SEED + 43)

    def dev():
        return [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]

    kw = dict(dx=DX, dy=DY, dz=DZ, delta=DELTA, dt=STEP_DT)

    def leg(n2_host, **extra):
        fields = dev()
        heat = cp.empty(STEP_SHAPE, dtype=cp.float32)
        led = launch_sase_step(
            *fields, n2=None if n2_host is None else cp.asarray(n2_host),
            heat=heat, **kw, **extra)
        return ([cp.asnumpy(a) for a in fields], cp.asnumpy(heat), led)

    # 1. absent vs explicitly False
    a_f, a_heat, a_led = leg(n232)
    b_f, b_heat, _ = leg(n232, additive_dissipation=False)
    for name, xa, xb in zip("uvwte", a_f, b_f):
        assert xa.tobytes() == xb.tobytes(), name
    assert a_heat.tobytes() == b_heat.tobytes()

    # 2. unstratified frame: the switch is bitwise absent
    n2_neg = -np.abs(n232).astype(np.float32)
    assert np.all(n2_neg <= 0.0)
    c_f, c_heat, _ = leg(n2_neg)
    d_f, d_heat, _ = leg(n2_neg, additive_dissipation=True)
    for name, xc, xd in zip("uvwte", c_f, d_f):
        assert xc.tobytes() == xd.tobytes(), (
            f"the switch moved {name} in a column with no stable cell")
    assert c_heat.tobytes() == d_heat.tobytes()

    # 3. n2 = None test box: bitwise absent again
    e_f, e_heat, _ = leg(None)
    f_f, f_heat, _ = leg(None, additive_dissipation=True)
    for name, xe, xf in zip("uvwte", e_f, f_f):
        assert xe.tobytes() == xf.tobytes(), (
            f"the switch moved {name} on the n2=None box")
    assert e_heat.tobytes() == f_heat.tobytes()

    # 4. mixed-sign frame: fires, and dissipation is nowhere weaker
    g_f, g_heat, _ = leg(n232, additive_dissipation=True)
    off_e = a_f[4]
    on_e = g_f[4]
    assert on_e.tobytes() != off_e.tobytes(), "the device seam never fired"
    assert np.all(on_e <= off_e)              # nowhere MORE energy
    assert np.any(on_e < off_e)               # strictly less somewhere
    moved = int(np.count_nonzero(on_e != off_e))
    print(f"S3-12 device seam: {moved} of {on_e.size} cells moved, "
          f"max -de = {float(np.max(off_e - on_e)):.3e}")

    # 5. composition: both ON is its own leg
    h_f, _, _ = leg(n232, stable_dissipation=True)
    i_f, _, _ = leg(n232, stable_dissipation=True,
                    additive_dissipation=True)
    assert i_f[4].tobytes() != h_f[4].tobytes()   # adds to the S3-6k base
    assert i_f[4].tobytes() != g_f[4].tobytes()   # base swap is visible


@requires_gpu
def test_sase_step_additive_dissipation_authority_parity():
    """CPU/DEVICE AGREEMENT with the switch ON: one device step against
    the FP64 authority stepped from the SAME FP32-cast inputs, at the
    established split-step trajectory tier (5e-5 scale-relative), on
    the additive leg AND the composed (S3-6k + S3-12) leg, with the OFF
    leg as control.  Allclose-tier on purpose -- a cross-precision
    check, not an inertness proof (the bitwise contracts are the gate
    above; the ULP-0 contract is the kernel-tier harness).  The legs
    are asserted pairwise DIFFERENT on both engines so the comparison
    cannot be passed by a switch that does nothing."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    from woof.verify.sase_ref import sase_split_step
    u32, v32, w32, th32, e32, n232 = _step_fixture32(seed=SEED + 44)
    ref64 = [a.astype(np.float64) for a in (u32, v32, w32, th32, e32)]
    n264 = n232.astype(np.float64)
    kw = dict(dx=DX, dy=DY, dz=DZ, delta=DELTA, dt=STEP_DT)
    got = {}
    for label, flags in (
            ("off", {}),
            ("additive", dict(additive_dissipation=True)),
            ("composed", dict(additive_dissipation=True,
                              stable_dissipation=True))):
        dev = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
        launch_sase_step(*dev, n2=cp.asarray(n232), **flags, **kw)
        fields, _ = sase_split_step(*ref64[:4], ref64[4], n2=n264,
                                    **flags, **kw)
        rels = {n: _max_rel(cp.asnumpy(d), fields[n])[0]
                for n, d in zip("uvwe", (dev[0], dev[1], dev[2], dev[4]))}
        print(f"S3-12 parity ({label}): " + "  ".join(
            f"{n}={r:.3e}" for n, r in rels.items()))
        assert max(rels.values()) <= 5e-5, (label, rels)
        got[label] = (cp.asnumpy(dev[4]), fields["e"])
    for a, b in (("off", "additive"), ("additive", "composed"),
                 ("off", "composed")):
        assert got[a][0].tobytes() != got[b][0].tobytes(), (a, b)
        assert got[a][1].tobytes() != got[b][1].tobytes(), (a, b)


@requires_gpu
def test_sase_step_additive_trajectory_parity_10_steps_stable_limb():
    """S3-12 COMPOUNDING trajectory parity on the regime the channel
    exists for: a mixed-layer-under-inversion column set (the BL89
    fixture class -- the stable limb the GABLS1 benchmark scores on
    the CPU engine, with n2 from the model's own brunt_vaisala_n2 so
    the l_s branch and the additive gate are live in the inversion),
    10 fused device split steps with additive_dissipation=True against
    the FP64 authority fed its own FP64 fields forward -- accurate
    compounding drift at the established 5e-5 gate, per-step growth
    recorded.  The OFF trajectory rides along and must DIVERGE from
    the ON trajectory on both engines (the switch does real work on
    this fixture), while both stay inside the same gate."""
    import cupy as cp
    from woof.core.sase import launch_sase_step
    from woof.verify import sase_ref
    from woof.verify.sase_ref import sase_split_step

    rng = np.random.default_rng(SEED + 45)
    nz, ny, nx = 24, 16, 16
    shape = (nz, ny, nx)
    dz = 25.0

    def band():
        return sase_ref.box_filter(rng.standard_normal(shape), 4)

    z1 = (np.arange(nz) + 0.5) * dz
    th1 = np.where(z1 <= 0.6 * z1[-1], 290.0,
                   290.0 + 0.02 * (z1 - 0.6 * z1[-1]))
    th32 = (np.broadcast_to(th1[:, None, None], shape)
            + 0.05 * band()).astype(np.float32)
    u32 = (5.0 + band()).astype(np.float32)
    v32 = band().astype(np.float32)
    w32 = (0.1 * band()).astype(np.float32)
    e32 = np.maximum(0.05 + 0.1 * band(), 0.0).astype(np.float32)
    n232 = sase_ref.brunt_vaisala_n2(
        th32.astype(np.float64), dz).astype(np.float32)
    assert np.any(n232 > 0.0)                     # the inversion is live
    kw = dict(dx=DX, dy=DY, dz=dz, delta=DELTA, dt=STEP_DT)
    th64, n264 = th32.astype(np.float64), n232.astype(np.float64)

    def trajectory(flag):
        dev = [cp.asarray(a) for a in (u32, v32, w32, th32, e32)]
        ref = {n: a.astype(np.float64)
               for n, a in zip("uvwe", (u32, v32, w32, e32))}
        growth = []
        for step in range(10):
            launch_sase_step(*dev, n2=cp.asarray(n232),
                             additive_dissipation=flag, **kw)
            fields, _ = sase_split_step(ref["u"], ref["v"], ref["w"],
                                        th64, ref["e"], n2=n264,
                                        additive_dissipation=flag, **kw)
            ref = {n: fields[n] for n in "uvwe"}
            rels = {n: _max_rel(cp.asnumpy(d), ref[n])[0]
                    for n, d in zip("uvwe",
                                    (dev[0], dev[1], dev[2], dev[4]))}
            growth.append(rels)
        worst = max(r for rels in growth for r in rels.values())
        tag = "ON " if flag else "OFF"
        for step, rels in enumerate(growth):
            print(f"{tag} step {step + 1:2d}: "
                  + "  ".join(f"{n}={rels[n]:.3e}" for n in "uvwe"))
        return worst, cp.asnumpy(dev[4]), ref["e"]

    worst_on, e_dev_on, e_ref_on = trajectory(True)
    worst_off, e_dev_off, e_ref_off = trajectory(False)
    assert worst_on <= 5e-5, worst_on
    assert worst_off <= 5e-5, worst_off
    # the switch does real work on this fixture, on BOTH engines
    assert e_dev_on.tobytes() != e_dev_off.tobytes()
    assert e_ref_on.tobytes() != e_ref_off.tobytes()


@requires_gpu
def test_sase_driver_additive_switch_is_selectable_and_fires():
    """SELECTABILITY, driver tier: ``RunConfig.sase_additive_
    dissipation`` on an admitted SASE configuration must (i) validate
    in BOTH positions, (ii) reach the device step through
    ``PhysicsDriver._run_sase`` -- before this seam the switch was
    authority-side only and a GPU run that set it got the channel
    silently DROPPED -- and (iii) move the prognostic subgrid energy on
    a stably stratified column set, while (iv) two OFF runs stay
    bitwise identical (the comparison is deterministic, so the ON/OFF
    difference is the switch and nothing else).

    THE DEFAULT IS TRUE (woof/config.py since 1a0e8a7f8; authority
    module docstring, S3-12 section, "DEFAULT: TRUE, AND WHY"), so the
    ON leg is the BARE default and the OFF leg pins
    ``sase_additive_dissipation=False`` explicitly -- the authority's
    own idiom for the un-channeled formulation, the one every RED leg
    uses.  A bare ``RunConfig`` taken as the OFF leg IS the ON leg:
    from the default flip until this construction the test compared
    the default against itself and read a wired switch as "not wired"
    (RTX 4090, NVRTC 13.3.33, receipt
    proof/sase-additive-switch-276/repro-7ec6c8e2b.xml).  (v) pins
    fixed-means-default: an explicit True is bitwise the bare
    default."""
    import cupy as cp
    from dataclasses import replace

    from woof.config import RunConfig, validate_run_config
    from woof.core.physics import initialize_physics
    from woof.verify.sase_ref import E_MIN

    cfg_on = RunConfig(nx=12, ny=10, nz=16, dx=2000.0, dy=2000.0,
                       ztop=8000.0, dt=5.0, run_seconds=5.0,
                       time_step_sound=4, moist=True, mp_physics=10,
                       ra_physics=4, sf_sfclay_physics=1,
                       sf_surface_physics=2, km_opt=0,
                       bl_pbl_physics=_SASE_SELECTOR)
    assert cfg_on.sase_additive_dissipation is True, (
        "the ON leg is the bare default; a default that moved needs "
        "this test's legs moved with it, never a bare config as OFF")
    cfg_off = replace(cfg_on, sase_additive_dissipation=False)
    cfg_on_explicit = replace(cfg_on, sase_additive_dissipation=True)
    for cfg in (cfg_on, cfg_off, cfg_on_explicit):
        assert validate_run_config(cfg) is cfg           # (i) admitted

    def run(cfg):
        state = _moist_driver_state(cfg)
        z_half = state.height_half()
        shear = (5.0 + 8.0 * z_half / cfg.ztop).astype(np.float32)
        state.u[...] = cp.asarray(
            np.broadcast_to(shear[:, None, None],
                            (cfg.nz, cfg.ny, cfg.nx + 1)))
        state.v[...] = cp.float32(1.0)

        driver = initialize_physics(state, cfg, landmask=1.0, tsk=302.0,
                                    swdown=400.0, glw=320.0,
                                    radiation=_zero_radiation)
        driver.compute(state, cfg)
        assert driver.call_counts["sase"] == 1
        return cp.asnumpy(state.e_sgs)

    e_off_1 = run(cfg_off)
    e_off_2 = run(cfg_off)
    assert e_off_1.tobytes() == e_off_2.tobytes(), (
        "the OFF leg is not deterministic -- the ON/OFF comparison "
        "below would be meaningless")                     # (iv)
    e_on = run(cfg_on)
    assert e_on.tobytes() != e_off_1.tobytes(), (
        "sase_additive_dissipation=True left the device step bitwise "
        "equal to an explicit False -- the switch is not wired")  # (ii, iii)
    # the channel only ever STRENGTHENS dissipation
    assert np.all(e_on <= e_off_1)
    assert np.all(np.isfinite(e_on)) and np.all(e_on >= np.float32(E_MIN))
    moved = int(np.count_nonzero(e_on != e_off_1))
    assert moved > 0
    # (v) fixed-means-default: the bare default IS the explicit True
    assert run(cfg_on_explicit).tobytes() == e_on.tobytes(), (
        "an explicit sase_additive_dissipation=True differs from the "
        "bare default -- the default is not the channel")
    print(f"driver seam: {moved} of {e_on.size} e_sgs cells moved "
          f"under sase_additive_dissipation=True against an explicit "
          f"False; max OFF - ON = {float(np.max(e_off_1 - e_on)):.3e}")
