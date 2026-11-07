# tests/test_diff6.py
"""Task 7: WRF 6th-order monotonic horizontal numerical diffusion.

Kernel-vs-mirror (rtol 1e-4) plus the plan's acceptance gates: a 2dx
checkerboard loses exactly ``diff_6th_factor`` of its amplitude per applied
step (within 2%), a smooth 8dx wave is damped at < 1/50 of that rate, the
monotonic option (diff_6th_opt=2) creates no new extrema on a sharp front
(while the unlimited operator provably does), and the ``dycore.step`` gate:
``diff_6th_opt=0`` leaves the step untouched, ``=2`` damps theta' at the
configured rate and the moisture scalars at three times it (WRF normalizes
the scalar rows by the RK sub-step -- see :data:`SCALAR_FACTOR`).
"""
import dataclasses
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu

FACTOR = 0.12

# WRF hands ``sixth_order_diffusion`` the caller's dt, and the two callers
# do not agree: ``rk_tendency`` passes ``grid%dt`` for u/v/w/theta
# (dyn_em/module_em.F:882/894/907/919) while ``rk_scalar_tend`` passes its
# ``dt_step`` for the moist/scalar/tke rows (:1425), which on the
# ``rk_step == 1`` pass that owns the call is ``dt_rk = grid%dt/3.``
# (dyn_em/solve_em.F:599, :2211).  ``diff_6th_coef =
# diff_6th_factor*0.015625/(2.0*dt)`` (module_big_step_utilities_em.F:6321),
# so the scalar filter removes three times the 2dx amplitude per step.
SCALAR_FACTOR = 3.0 * FACTOR


def _checkerboard(ny, nx):
    return (-1.0) ** np.add.outer(np.arange(ny), np.arange(nx))


def _random_case(stagger, nz=8, ny=6, nx=12, seed=0):
    """Random fp32 field + face mass + hybrid coeffs for one stagger."""
    rng = np.random.default_rng(seed)
    nlev = nz + 1 if stagger == "z" else nz
    shape = {"": (nlev, ny, nx), "z": (nlev, ny, nx),
             "x": (nlev, ny, nx + 1), "y": (nlev, ny + 1, nx)}[stagger]
    f = rng.standard_normal(shape).astype(np.float32)
    if stagger == "x":
        f[..., -1] = f[..., 0]
    if stagger == "y":
        f[:, -1, :] = f[:, 0, :]
    mut = (1.0 + 0.5 * rng.random((ny, nx))).astype(np.float32)
    c1 = (0.5 + 0.5 * rng.random(nlev)).astype(np.float32)
    c2 = (0.5 * rng.random(nlev)).astype(np.float32)
    return f, mut, c1, c2


def _dev_tend(f, mut, c1, c2, factor, dt, opt, stagger):
    """Run the diff6 kernel once; return the coupled tendency on host."""
    import cupy as cp

    from woof.core.dycore import launch_diff6
    fd = cp.asarray(f, dtype=cp.float32)
    tend = cp.zeros_like(fd)
    launch_diff6(fd, tend, cp.asarray(mut, dtype=cp.float32),
                 cp.asarray(c1, dtype=cp.float32),
                 cp.asarray(c2, dtype=cp.float32), factor, dt, opt,
                 stagger=stagger)
    return cp.asnumpy(tend)


@requires_gpu
@pytest.mark.parametrize("opt", [1, 2])
@pytest.mark.parametrize("stagger", ["", "x", "y", "z"])
def test_diff6_matches_reference(stagger, opt):
    from woof.verify.npref import np_diff6
    f, mut, c1, c2 = _random_case(stagger, seed=7)
    dt = 2.0
    got = _dev_tend(f, mut, c1, c2, FACTOR, dt, opt, stagger)
    ref = np_diff6(f.astype(np.float64), mut, c1, c2, FACTOR, dt, opt,
                   stagger=stagger)
    # FP32-FLOOR: absolute tolerance covers float32 cancellation at a
    # zero/near-zero reference; rtol=1e-4 remains the signal-scale gate.
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)


@requires_gpu
@pytest.mark.parametrize("open_x,open_y",
                         [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("stagger", ["", "x", "y", "z"])
def test_diff6_open_strips_match_mirror(stagger, open_x, open_y):
    """Open-BC bounds of the diff6 application path (boundary-aware
    kernel + _zero_open_strips) vs the float64 mirror with the same flags.

    WRF sixth_order_diffusion excludes 3 entries per open side on every
    axis and stagger; the outermost computed boundary-normal STAGGERED
    face (u's nx-3 under open_x, v's ny-3 under open_y) is computed with
    WRF's accurate read of the stored true boundary datum field(ide)
    (kernel ``bndx``/``bndy``; the pre-fix tree zeroed that face instead,
    see tests/test_diff6_boundary_face.py).  The exact zero/live strip
    per axis is pinned mirror-independently below.
    """
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.dycore import _zero_open_strips, launch_diff6
    from woof.verify.npref import np_diff6

    nz, ny, nx = 8, 10, 12          # both live regions nonempty at width 3
    f, mut, c1, c2 = _random_case(stagger, nz=nz, ny=ny, nx=nx, seed=13)
    if stagger == "x":              # independent boundary-face datum: the
        f[..., -1] = 0.7 * f[..., 0] + 0.3   # accurate read must be
    if stagger == "y":                       # distinguishable from the wrap
        f[:, -1, :] = 0.7 * f[:, 0, :] + 0.3
    dt = 2.0
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, ztop=1.0, dt=dt,
                    run_seconds=0.0, open_x=open_x, open_y=open_y)
    fd = cp.asarray(f, dtype=cp.float32)
    tend = cp.zeros_like(fd)
    launch_diff6(fd, tend, cp.asarray(mut, dtype=cp.float32),
                 cp.asarray(c1, dtype=cp.float32),
                 cp.asarray(c2, dtype=cp.float32), FACTOR, dt, 2,
                 stagger=stagger, bnd_x=open_x, bnd_y=open_y)
    _zero_open_strips(tend, cfg, 3)              # as apply_diff6 does
    got = cp.asnumpy(tend)
    ref = np_diff6(f.astype(np.float64), mut, c1, c2, FACTOR, dt, 2,
                   stagger=stagger, open_x=open_x, open_y=open_y)
    # FP32-FLOOR: absolute tolerance covers float32 cancellation at a
    # zero/near-zero reference; rtol=1e-4 remains the signal-scale gate.
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)
    # Pin the strip per axis (independent of the mirror): 3 zeroed entries
    # per open side on every stagger; first live entries on both sides
    # nonzero -- including the staggered-normal high face nx-3/ny-3 that
    # WRF computes with the boundary datum (RED-pinned at 4d2ce99).
    if open_x:
        assert (got[:, :, :3] == 0.0).all()
        assert (got[:, :, -3:] == 0.0).all()
        assert np.abs(got[:, :, 3]).max() > 0.0
        assert np.abs(got[:, :, -4]).max() > 0.0
    else:
        assert np.abs(got[:, :, 0]).max() > 0.0    # no over-zeroing in x
    if open_y:
        assert (got[:, :3, :] == 0.0).all()
        assert (got[:, -3:, :] == 0.0).all()
        assert np.abs(got[:, 3, :]).max() > 0.0
        assert np.abs(got[:, -4, :]).max() > 0.0
    else:
        assert np.abs(got[:, 0, :]).max() > 0.0    # no over-zeroing in y


def test_mirror_checkerboard_normalization_float64():
    # The factor/2^6 transcription in float64: a 2dx checkerboard loses
    # EXACTLY diff_6th_factor of its amplitude per step (uniform mu, the
    # coupled tendency uncoupled by the same mass).
    from woof.verify.npref import np_diff6
    ny = nx = 8
    f = _checkerboard(ny, nx)[None]
    mu0, dt = 1.7, 3.0
    mut = np.full((ny, nx), mu0)
    tend = np_diff6(f, mut, np.ones(1), np.zeros(1), FACTOR, dt, 2)
    np.testing.assert_allclose(f + dt * tend / mu0, (1.0 - FACTOR) * f,
                               rtol=1e-12)


@requires_gpu
def test_eight_internal_steps_apply_one_outer_clock_diff6_factor():
    """real74's 8 x 7.5 s structure retains WRF's 60 s 2dx damping."""
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.dycore import (_PROGNOSTICS, _clock_scaled_diff6_factor,
                                   apply_diff6)
    from woof.verify.npref import np_clock_scaled_diff6_factor

    cfg = RunConfig(nx=8, ny=8, nz=6, dx=1000.0, dy=1000.0,
                    ztop=6000.0, dt=7.5, clock_dt=60.0, run_seconds=0.0,
                    diff_6th_opt=2, diff_6th_factor=FACTOR)
    effective = _clock_scaled_diff6_factor(cfg)
    np.testing.assert_allclose(effective, np_clock_scaled_diff6_factor(cfg),
                               rtol=0.0, atol=1.0e-16)
    np.testing.assert_allclose((1.0 - effective) ** 8, 1.0 - FACTOR,
                               rtol=0.0, atol=2.0e-16)

    state = _atrest(cfg)
    checker = _checkerboard(cfg.ny, cfg.nx)
    state.thp[...] = cp.asarray(
        np.broadcast_to(0.1 * checker, state.thp.shape), dtype=cp.float32)
    initial = _cb_amp(cp.asnumpy(state.thp).astype(np.float64), checker)
    for _ in range(8):
        for name in _PROGNOSTICS:
            getattr(state, name + "0")[...] = getattr(state, name)
        apply_diff6(state, cfg)
    final = _cb_amp(cp.asnumpy(state.thp).astype(np.float64), checker)
    np.testing.assert_allclose(final / initial, 1.0 - FACTOR,
                               rtol=2.0e-6, atol=2.0e-7)


@requires_gpu
@pytest.mark.parametrize("opt", [1, 2])
def test_checkerboard_damped_by_configured_fraction(opt):
    # Acceptance: 2dx checkerboard damped by the configured fraction per
    # step within 2% (the checkerboard is down-gradient at every face, so
    # the monotonic option must not alter the rate).
    dt = 5.0
    ny = nx = 16
    f = _checkerboard(ny, nx)[None].astype(np.float32)
    mut = np.ones((ny, nx), np.float32)
    one = np.ones(1, np.float32)
    tend = _dev_tend(f, mut, one, 0.0 * one, FACTOR, dt, opt, "")
    fnew = f + dt * tend                       # mu = 1: coupling drops out
    damped = 1.0 - np.max(np.abs(fnew)) / np.max(np.abs(f))
    assert abs(damped - FACTOR) < 0.02 * FACTOR
    np.testing.assert_allclose(fnew, (1.0 - FACTOR) * f, rtol=2e-2,
                               atol=1e-6)


@requires_gpu
def test_smooth_wave_selectivity():
    # Acceptance: an 8dx wave is damped at < 1/50 the 2dx checkerboard rate.
    dt = 5.0
    ny, nx = 4, 64
    x = np.arange(nx)
    f = np.ascontiguousarray(
        np.broadcast_to(np.sin(2.0 * np.pi * x / 8.0), (1, ny, nx))
    ).astype(np.float32)
    mut = np.ones((ny, nx), np.float32)
    one = np.ones(1, np.float32)
    tend = _dev_tend(f, mut, one, 0.0 * one, FACTOR, dt, 2, "")
    fnew = f + dt * tend
    damped = 1.0 - np.max(np.abs(fnew)) / np.max(np.abs(f))
    assert 0.0 < damped < FACTOR / 50.0


@requires_gpu
def test_monotonic_front_preserves_bounds():
    # Acceptance: with diff_6th_opt=2 no new extrema appear when a sharp
    # front is diffused; the unlimited operator (opt=1) demonstrably
    # overshoots, so the bound check is not vacuous.
    import cupy as cp

    from woof.core.dycore import launch_diff6
    dt = 2.0
    ny, nx = 4, 48
    f0 = np.zeros((1, ny, nx), np.float32)
    f0[:, :, : nx // 2] = 1.0
    mut = cp.ones((ny, nx), cp.float32)
    one = cp.ones(1, cp.float32)

    def run(opt, n=50):
        fd = cp.asarray(f0)
        tend = cp.zeros_like(fd)
        for _ in range(n):
            tend[...] = 0
            launch_diff6(fd, tend, mut, one, 0.0 * one, FACTOR, dt, opt)
            fd += np.float32(dt) * tend
        return cp.asnumpy(fd)

    mono = run(2)
    # FP32-FLOOR: absolute tolerance covers float32 cancellation at a
    # zero/near-zero reference; rtol=1e-4 remains the signal-scale gate.
    extrema_floor = 8.0 * np.finfo(np.float32).eps
    assert mono.max() <= 1.0 + extrema_floor
    assert mono.min() >= -extrema_floor
    plain = run(1)
    assert plain.max() > 1.0 + 1e-3 or plain.min() < -1e-3


@requires_gpu
def test_apply_diff6_matches_mirror_on_random_state():
    # Full coupled application (all staggerings, hybrid coeffs, real base
    # state): apply_diff6 reads the time-t *0 copies and must reproduce
    # f + dt*np_diff6(...)/chm field by field.
    #
    # T7-3 (Phase 2 final review): the fixture perturbs mup0 away from mup
    # so the two masses of the application are distinguishable -- the
    # TENDENCY couples with the time-t mass (mub2d + mup0, WRF mut) while
    # the increment UNCOUPLES with the post-step mass (mub2d + mup).  With
    # the previous mup0 == mup fixture a mass swap between the two roles
    # was undetectable (verified: the swapped implementation passed).
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.dycore import _PROGNOSTICS, apply_diff6
    from woof.verify.npref import np_diff6, random_acoustic_state

    s, cfg0 = random_acoustic_state(seed=11, nz=8, ny=6, nx=12)
    cfg = dataclasses.replace(cfg0, **{"diff_6th_opt": 2,
                       "diff_6th_factor": 0.3})
    for name in _PROGNOSTICS:
        getattr(s, name + "0")[...] = getattr(s, name)
    rng = np.random.default_rng(311)
    s.mup0[...] = s.mup + cp.asarray(                  # ~9% of mub: far
        6.0e3 * rng.standard_normal(s.mup.shape),      # above the mirror
        dtype=s.mup.dtype)                             # comparison rtol
    host = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
            for n in ("u", "v", "w", "thp", "mup", "mup0", "mub2d",
                      "c1h", "c2h", "c1f", "c2f")}
    apply_diff6(s, cfg)

    mu_t = host["mub2d"] + host["mup0"]                # tendency coupling
    mu = host["mub2d"] + host["mup"]                   # increment uncoupling
    mux = 0.5 * (mu + np.roll(mu, 1, axis=1))
    mux = np.concatenate([mux, mux[:, :1]], axis=1)
    muy = 0.5 * (mu + np.roll(mu, 1, axis=0))
    muy = np.concatenate([muy, muy[:1, :]], axis=0)
    c1h = host["c1h"][:, None, None]
    c2h = host["c2h"][:, None, None]
    c1f = host["c1f"][:, None, None]
    c2f = host["c2f"][:, None, None]
    cases = (("u", "x", host["c1h"], host["c2h"], c1h * mux[None] + c2h),
             ("v", "y", host["c1h"], host["c2h"], c1h * muy[None] + c2h),
             ("w", "z", host["c1f"], host["c2f"], c1f * mu[None] + c2f),
             ("thp", "", host["c1h"], host["c2h"], c1h * mu[None] + c2h))
    for name, stag, c1, c2, chm in cases:
        tend = np_diff6(host[name], mu_t, c1, c2, cfg.diff_6th_factor,
                        cfg.dt, cfg.diff_6th_opt, stagger=stag)
        want = host[name] + cfg.dt * tend / chm
        # FP32-FLOOR: absolute tolerance covers float32 cancellation at a
        # zero/near-zero reference; rtol=1e-4 remains the signal-scale gate.
        np.testing.assert_allclose(cp.asnumpy(getattr(s, name)), want,
                                   rtol=1e-4, atol=1e-6, err_msg=name)
        # Mass-swap detectability: the mirror with the two masses exchanged
        # differs from the correct answer by far more than the comparison
        # tolerance, so the assertion above cannot pass a swapped
        # implementation.
        tend_sw = np_diff6(host[name], mu, c1, c2, cfg.diff_6th_factor,
                           cfg.dt, cfg.diff_6th_opt, stagger=stag)
        want_sw = host[name] + cfg.dt * tend_sw / chm
        gap = np.max(np.abs(want_sw - want))
        tol = 1e-4 * np.max(np.abs(want)) + 1e-6
        assert gap > 10.0 * tol, (name, gap, tol)


@requires_gpu
def test_apply_diff6_shared_alias_poison_is_bitwise_exact():
    """A poisoned shared backing reproduces independent diff6 allocations."""
    import types

    import cupy as cp

    from woof.config import RunConfig
    from woof.core.dycore import apply_diff6
    from woof.core.moist import MORRISON_SPECIES, SPECIES
    from woof.core.state import build_shared_scratch_arena
    from woof.verify.npref import random_acoustic_state

    reference, cfg0 = random_acoustic_state(
        seed=711, nz=8, ny=10, nx=12, moist=True, mp_physics=10)
    aliased, _ = random_acoustic_state(
        seed=711, nz=8, ny=10, nx=12, moist=True, mp_physics=10)
    cfg = dataclasses.replace(cfg0, **{"diff_6th_opt": 2,
                       "diff_6th_factor": 0.3})
    for state in (reference, aliased):
        for name in ("u", "v", "w", "thp"):
            getattr(state, name + "0")[...] = getattr(state, name)
        for index, name in enumerate((*SPECIES, *MORRISON_SPECIES), 1):
            field = getattr(state, name)
            values = cp.arange(field.size, dtype=cp.float32).reshape(
                field.shape)
            field[...] = cp.float32(index) * cp.float32(1.0e-9) * values
            getattr(state, name + "0")[...] = field

    arena = build_shared_scratch_arena(
        (types.SimpleNamespace(run=cfg),))
    aliased._scratch_arena = arena
    arena.poison()

    apply_diff6(reference, cfg)
    apply_diff6(aliased, cfg)
    cp.cuda.get_current_stream().synchronize()

    views = [aliased.scratch(arena.slot_shapes[slot], slot)
             for slot in ("diff6_x", "diff6_y", "diff6_z", "diff6_m")]
    assert len({int(view.data.ptr) for view in views}) == 1
    for name in ("u", "v", "w", "thp", *SPECIES, *MORRISON_SPECIES):
        cp.testing.assert_array_equal(
            getattr(aliased, name), getattr(reference, name), err_msg=name)


# ---- diff_6th_slopeopt = 1: terrain-slope taper of the 6th-order fluxes -----
#
# WRF sixth_order_diffusion, slopeopt branch (module_big_step_utilities_em.F
# :6487-6501 x / :6569-6583 y): when diff_6th_slopeopt >= 1 each face flux is
# scaled by slopedamp = MAX(1 - dzmax/dzthresh, 0) with dzthresh =
# diff_6th_thresh*9.81*dx (the routine's literal 9.81) and dzmax the
# BASE-state (phb) face geopotential jump at the field's own level index k,
# msf-scaled per face (msfux for x slopes, msfvy for y; u/v variants take the
# max over their two adjacent mass faces).  Registry defaults slopeopt 0 /
# thresh 0.10 (Registry.EM_COMMON:2858-2859); the reference namelist sets
# diff_6th_slopeopt = 1 (namelist.input:94).


def _dev_tend_slope(f, mut, c1, c2, factor, dt, opt, stagger, phb,
                    msfu=None, msfv=None, slopeopt=1, thresh=0.10,
                    dx=1000.0, dy=1000.0):
    """Run the diff6 kernel once with the slope taper; tendency on host."""
    import cupy as cp

    from woof.core.dycore import launch_diff6
    fd = cp.asarray(f, dtype=cp.float32)
    tend = cp.zeros_like(fd)
    launch_diff6(fd, tend, cp.asarray(mut, dtype=cp.float32),
                 cp.asarray(c1, dtype=cp.float32),
                 cp.asarray(c2, dtype=cp.float32), factor, dt, opt,
                 stagger=stagger, phb=cp.asarray(phb, dtype=cp.float32),
                 msfu=None if msfu is None else cp.asarray(msfu, cp.float32),
                 msfv=None if msfv is None else cp.asarray(msfv, cp.float32),
                 slopeopt=slopeopt, thresh=thresh, dx=dx, dy=dy)
    return cp.asnumpy(tend)


@requires_gpu
def test_diff6_slopeopt_taper_factor_on_uniform_slope():
    # A phb ramp in x with dz = 0.5*thresh*dx per face gives slopedamp =
    # 0.5 at every x flux; a field varying only in x has zero y fluxes, so
    # the tapered tendency is exactly half the untapered one.  A ramp past
    # the threshold (dz >= thresh*dx) kills the x diffusion entirely; a
    # pure-y field is bitwise untouched by the x ramp (slopedamp_y = 1
    # exactly, phb flat in y).
    nz, ny, nx = 4, 6, 16
    dx = dy = 1000.0
    thresh = 0.10
    rng = np.random.default_rng(5)
    gx = rng.standard_normal((nz, 1, nx)).astype(np.float32)
    f = np.ascontiguousarray(np.broadcast_to(gx, (nz, ny, nx)))
    mut = (1.0 + 0.5 * rng.random((ny, nx))).astype(np.float32)
    c1 = np.ones(nz, np.float32)
    c2 = np.zeros(nz, np.float32)
    dz_half = 0.5 * thresh * dx                # -> slopedamp = 0.5
    phb = np.ascontiguousarray(np.broadcast_to(
        (9.81 * dz_half * np.arange(nx))[None, None, :],
        (nz + 1, ny, nx))).astype(np.float32)

    t_off = _dev_tend(f, mut, c1, c2, FACTOR, 2.0, 2, "")
    assert np.abs(t_off).max() > 0.0
    t_half = _dev_tend_slope(f, mut, c1, c2, FACTOR, 2.0, 2, "", phb,
                             thresh=thresh, dx=dx, dy=dy)
    # Columns 0 and nx-1 straddle the periodic wrap face, where the
    # non-periodic ramp jumps by (nx-1) faces -> slopedamp = 0 there; all
    # interior columns see the uniform 0.5 taper on both their faces.
    np.testing.assert_allclose(t_half[:, :, 1:-1], 0.5 * t_off[:, :, 1:-1],
                               rtol=1e-5, atol=1e-8)
    assert np.abs(t_half[:, :, [0, -1]]).max() \
        < np.abs(0.5 * t_off[:, :, [0, -1]]).max()

    t_zero = _dev_tend_slope(f, mut, c1, c2, FACTOR, 2.0, 2, "", 2.5 * phb,
                             thresh=thresh, dx=dx, dy=dy)
    np.testing.assert_array_equal(t_zero, np.zeros_like(t_zero))

    gy = rng.standard_normal((nz, ny, 1)).astype(np.float32)
    fy = np.ascontiguousarray(np.broadcast_to(gy, (nz, ny, nx)))
    ty_off = _dev_tend(fy, mut, c1, c2, FACTOR, 2.0, 2, "")
    ty_on = _dev_tend_slope(fy, mut, c1, c2, FACTOR, 2.0, 2, "", phb,
                            thresh=thresh, dx=dx, dy=dy)
    np.testing.assert_array_equal(ty_on, ty_off)


@requires_gpu
@pytest.mark.parametrize("stagger", ["", "x", "y", "z"])
def test_diff6_slopeopt_matches_mirror(stagger):
    # Kernel vs float64 mirror with a random terrain-like 3-D phb whose
    # face slopes straddle the taper range, plus non-identity msfu/msfv
    # (the WRF dzmax msf scaling), for every stagger variant.
    from woof.verify.npref import np_diff6
    nz, ny, nx = 6, 8, 12
    f, mut, c1, c2 = _random_case(stagger, nz=nz, ny=ny, nx=nx, seed=29)
    rng = np.random.default_rng(37)
    dx = dy = 500.0                            # dzthresh: 50 m per face
    z = (60.0 * np.cumsum(rng.random((ny, nx)), axis=1)
         + 40.0 * np.cumsum(rng.random((ny, nx)), axis=0))
    phb = (9.81 * (z[None] + 200.0 * np.arange(nz + 2)[:, None, None])
           ).astype(np.float32)[:nz + 1]
    msfu = (1.0 + 0.1 * rng.random((ny, nx + 1))).astype(np.float32)
    msfv = (1.0 + 0.1 * rng.random((ny + 1, nx))).astype(np.float32)
    msfu[:, -1] = msfu[:, 0]
    msfv[-1, :] = msfv[0, :]

    got = _dev_tend_slope(f, mut, c1, c2, FACTOR, 2.0, 2, stagger, phb,
                          msfu=msfu, msfv=msfv, dx=dx, dy=dy)
    ref = np_diff6(f.astype(np.float64), mut, c1, c2, FACTOR, 2.0, 2,
                   stagger=stagger, phb=phb, msfu=msfu, msfv=msfv,
                   slopeopt=1, thresh=0.10, dx=dx, dy=dy)
    # The compiled WRF oracle found that tendency maps are independent
    # of the slope taper. This legacy mirror specializes tendency maps
    # to identity, so restore the field's own tendency map here.
    if stagger == "x":
        ref *= msfu[None]
    elif stagger == "y":
        ref *= msfv[None]
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-7)
    off = _dev_tend(f, mut, c1, c2, FACTOR, 2.0, 2, stagger)
    assert np.abs(got - off).max() > 0.0       # the taper actually bites


@requires_gpu
def test_apply_diff6_slopeopt_uses_state_terrain():
    # apply_diff6 must thread cfg.diff_6th_slopeopt/diff_6th_thresh and
    # the state's 3-D phb (+ msf) into the kernel: over the bell-ridge
    # hybrid base (800 m over halfwidth 1500 m at dx = 500 m, far past the
    # 50 m dzthresh) the tapered theta' update differs from the untapered
    # one and matches the float64 mirror with the same slope inputs.
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.dycore import _PROGNOSTICS, apply_diff6
    from woof.verify.npref import np_diff6, random_acoustic_state

    def run(slopeopt):
        s, cfg0 = random_acoustic_state(seed=11, nz=8, ny=6, nx=12,
                                        hybrid_opt=2, hill_height=800.0)
        cfg = dataclasses.replace(cfg0, **{"diff_6th_opt": 2,
                           "diff_6th_factor": 0.3,
                           "diff_6th_slopeopt": slopeopt})
        for name in _PROGNOSTICS:
            getattr(s, name + "0")[...] = getattr(s, name)
        apply_diff6(s, cfg)
        return s, cfg

    s1, cfg = run(1)
    s0, _ = run(0)
    assert float(cp.abs(s1.thp - s0.thp).max()) > 0.0   # taper is live

    host = {n: cp.asnumpy(getattr(s1, n)).astype(np.float64)
            for n in ("thp0", "mup", "mub2d", "c1h", "c2h", "phb",
                      "msfu", "msfv")}
    mu = host["mub2d"] + host["mup"]           # mup0 == mup in this fixture
    c1h = host["c1h"][:, None, None]
    c2h = host["c2h"][:, None, None]
    tend = np_diff6(host["thp0"], mu, host["c1h"], host["c2h"],
                    cfg.diff_6th_factor, cfg.dt, cfg.diff_6th_opt,
                    stagger="", phb=host["phb"], msfu=host["msfu"],
                    msfv=host["msfv"], slopeopt=1,
                    thresh=cfg.diff_6th_thresh, dx=cfg.dx, dy=cfg.dy)
    want = host["thp0"] + cfg.dt * tend / (c1h * mu[None] + c2h)
    np.testing.assert_allclose(cp.asnumpy(s1.thp), want, rtol=1e-4,
                               atol=1e-6)


def test_diff6_slopeopt_config_surface():
    # Defaults keep the untapered operator (bitwise); the real74
    # production configs carry the reference namelist's slopeopt=1 with
    # the Registry-default 0.10 threshold.
    from woof.config import RunConfig
    from woof.verify.cases.real74_d01 import config, phase3_config
    base = dict(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, ztop=1.0, dt=1.0,
                run_seconds=0.0)
    assert RunConfig(**base).diff_6th_slopeopt == 0
    assert RunConfig(**base).diff_6th_thresh == 0.10
    assert phase3_config().diff_6th_slopeopt == 1
    assert phase3_config().diff_6th_thresh == 0.10
    assert config().diff_6th_slopeopt == 1


def _atrest(cfg):
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc,
                        lambda z: np.full_like(np.asarray(z, float), 300.0),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    return init_at_rest(cfg, vc, b)


def _cb_amp(a, cb):
    """Checkerboard-mode amplitude of a (nz, ny, nx) field (projection)."""
    a = a - a.mean(axis=(1, 2), keepdims=True)
    return float((a * cb[None]).mean() / (cb * cb).mean())


_STEP_BASE = dict(nx=8, ny=8, nz=6, dx=1000.0, dy=1000.0, ztop=6000.0,
                  dt=0.05, run_seconds=0.0, time_step_sound=4)


@requires_gpu
def test_step_adds_one_time_t_diff6_tendency_to_all_three_rk_stages(
        monkeypatch):
    """WRF computes stage-1 tendf once and consumes it on every RK pass."""
    import cupy as cp

    from woof.config import RunConfig
    from woof.core import dycore
    from woof.verify.npref import np_diff6

    cfg = RunConfig(**_STEP_BASE, diff_6th_opt=2,
                    diff_6th_factor=FACTOR)
    state = _atrest(cfg)
    cb = _checkerboard(cfg.ny, cfg.nx)
    state.thp[...] = cp.asarray(
        np.broadcast_to(0.1 * cb, state.thp.shape), dtype=cp.float32)
    initial = cp.asnumpy(state.thp).astype(np.float64)
    mu_t = cp.asnumpy(state.mub2d + state.mup).astype(np.float64)
    expected = np_diff6(
        initial, mu_t, cp.asnumpy(state.c1h), cp.asnumpy(state.c2h),
        dycore._clock_scaled_diff6_factor(cfg), cfg.dt,
        cfg.diff_6th_opt)

    seen = []

    def prepare_init(_state, _cfg):
        def record_stage_tendency():
            seen.append(cp.asnumpy(_state.rth_t).astype(np.float64))
        return record_stage_tendency

    monkeypatch.setattr(dycore, "_prepare_small_step_init_launch",
                        prepare_init)
    monkeypatch.setattr(dycore, "_prepare_small_step_finish_launch",
                        lambda *_a, **_k: lambda: None)
    monkeypatch.setattr(dycore, "prepare_acoustic_coefficients",
                        lambda *_a, **_k: None)
    monkeypatch.setattr(dycore, "prepare_acoustic_substep_launch",
                        lambda *_a, **_k: lambda **_kw: None)
    monkeypatch.setattr(dycore, "stage_fluxes",
                        lambda s, _cfg, **_k: (s.u, s.v, s.w))
    # add_diffusion_tendencies is gone from the model step: km_opt=1 is
    # WRF's once-per-step mixing package now (woof/core/diffusion.py).
    for name in ("_add_slow_tendencies",
                 "apply_w_damping", "apply_state_lateral_boundaries",
                 "apply_open_radiative_bc", "apply_open_zero_gradient",
                 "apply_state_boundary_values", "set_w_surface",
                 "update_diagnostics", "apply_diff6"):
        monkeypatch.setattr(dycore, name, lambda *_a, **_k: None)

    dycore.step(state, cfg)

    assert len(seen) == 3
    assert np.abs(expected).max() > 0.0
    for stage, got in enumerate(seen, start=1):
        np.testing.assert_allclose(got, expected, rtol=1.0e-4,
                                   atol=1.0e-8,
                                   err_msg=f"RK stage {stage}")


@requires_gpu
def test_diff6_dry_and_scalar_rows_take_different_dt(monkeypatch):
    """WRF's two ``sixth_order_diffusion`` callers pass different ``dt``.

    ``rk_tendency`` hands ``grid%dt`` to the u/v/w/theta calls
    (dyn_em/module_em.F:882/894/907/919, all inside ``forward_step:
    IF( rk_step == 1 )`` at :800, and the routine's ``dt`` dummy is bound to
    ``grid%dt`` at dyn_em/solve_em.F:892).  ``rk_scalar_tend`` hands its own
    ``dt_step`` to the moist/scalar/tke call (:1425, inside ``rk_step_1``
    at :1378), and every ``rk_scalar_tend`` call site passes ``dt_rk``
    (solve_em.F:2211, :2380, :2473, :2635, :2777), which is ``grid%dt/3.``
    on ``rk_step == 1`` for ``rk_ord = 3`` (:596-600).  The argument is the
    filter strength -- ``diff_6th_coef = diff_6th_factor*0.015625/(2.0*dt)``
    (module_big_step_utilities_em.F:6321) -- so a single ``dt`` for every
    row is a third of WRF's scalar filter.  Recorded per row rather than
    measured, so the pin names the divergence instead of a damping ratio.
    """
    from woof.config import RunConfig
    from woof.core import dycore

    cfg = RunConfig(**_STEP_BASE, moist=True, diff_6th_opt=2,
                    diff_6th_factor=FACTOR)
    state = _atrest(cfg)
    seen = {}

    def record(f, tend, mut, c1, c2, factor, dt, opt, **kw):
        seen[id(f)] = dt

    monkeypatch.setattr(dycore, "launch_diff6", record)
    dycore.prepare_fixed_tendencies(state, cfg)

    for name in ("u0", "v0", "w0", "thp0"):
        assert seen[id(getattr(state, name))] == cfg.dt, name
    for name in ("qv0", "qc0", "qr0"):
        assert seen[id(getattr(state, name))] == cfg.dt / 3.0, name


@requires_gpu
def test_step_gate_dry_theta_checkerboard():
    # dycore gate: diff_6th_opt=0 leaves a theta' checkerboard essentially
    # untouched over one tiny step; opt=2 removes the configured fraction
    # (measured against the opt=0 twin so the small acoustic response of
    # the unbalanced checkerboard cancels).
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.dycore import step
    cb = _checkerboard(_STEP_BASE["ny"], _STEP_BASE["nx"])
    amp = {}
    for opt in (0, 2):
        cfg = RunConfig(**_STEP_BASE, diff_6th_opt=opt,
                        diff_6th_factor=FACTOR)
        s = _atrest(cfg)
        s.thp[...] = cp.asarray(
            np.broadcast_to(0.1 * cb, (cfg.nz, cfg.ny, cfg.nx)),
            dtype=cp.float32)
        step(s, cfg)
        amp[opt] = _cb_amp(cp.asnumpy(s.thp).astype(np.float64), cb)
    assert abs(amp[0] - 0.1) < 1e-3               # opt=0: no diffusion
    ratio = amp[2] / amp[0]
    assert abs(ratio - (1.0 - FACTOR)) < 0.02 * (1.0 - FACTOR)


@requires_gpu
def test_step_moist_scalar_checkerboard():
    # Moisture path through dycore.step: a qv checkerboard is damped by
    # three times the configured fraction per step, because WRF normalizes
    # the rk_scalar_tend rows by dt/3 (SCALAR_FACTOR); opt=0 twin as the
    # reference.
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.dycore import step
    cb = _checkerboard(_STEP_BASE["ny"], _STEP_BASE["nx"])
    amp = {}
    for opt in (0, 2):
        cfg = RunConfig(**_STEP_BASE, moist=True, diff_6th_opt=opt,
                        diff_6th_factor=FACTOR)
        s = _atrest(cfg)
        s.qv[...] = cp.asarray(
            np.broadcast_to(0.005 + 0.001 * cb, (cfg.nz, cfg.ny, cfg.nx)),
            dtype=cp.float32)
        step(s, cfg)
        amp[opt] = _cb_amp(cp.asnumpy(s.qv).astype(np.float64), cb)
    assert abs(amp[0] - 0.001) < 1e-5             # opt=0: no diffusion
    ratio = amp[2] / amp[0]
    assert abs(ratio - (1.0 - SCALAR_FACTOR)) < 0.02 * (1.0 - SCALAR_FACTOR)


@requires_gpu
def test_step_moist_mix6_off_spares_moisture_and_keeps_damping_theta():
    """Divergence-ledger L4, measured on the trajectory rather than argued.

    WRF's ``moist_mix6_off`` gates one call: ``sixth_order_diffusion`` on
    the moist array (dyn_em/module_em.F:1421).  So the pin is a PAIR, in one
    run: the qv checkerboard must survive the step essentially untouched
    while the theta' checkerboard in the same state loses exactly the
    configured fraction.  Either half alone would pass for a gate that had
    turned off too much or too little.
    """
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.dycore import step
    cb = _checkerboard(_STEP_BASE["ny"], _STEP_BASE["nx"])
    qv_amp, th_amp = {}, {}
    for mix6_off in (False, True):
        for opt in (0, 2):
            cfg = RunConfig(**_STEP_BASE, moist=True, diff_6th_opt=opt,
                            diff_6th_factor=FACTOR,
                            moist_mix6_off=mix6_off)
            s = _atrest(cfg)
            s.qv[...] = cp.asarray(
                np.broadcast_to(0.005 + 0.001 * cb,
                                (cfg.nz, cfg.ny, cfg.nx)),
                dtype=cp.float32)
            s.thp[...] = cp.asarray(
                np.broadcast_to(0.1 * cb, (cfg.nz, cfg.ny, cfg.nx)),
                dtype=cp.float32)
            step(s, cfg)
            qv_amp[(mix6_off, opt)] = _cb_amp(
                cp.asnumpy(s.qv).astype(np.float64), cb)
            th_amp[(mix6_off, opt)] = _cb_amp(
                cp.asnumpy(s.thp).astype(np.float64), cb)

    # moist_mix6_off = false is WRF's default and must be exactly the
    # behaviour the sibling test already pins.
    on = qv_amp[(False, 2)] / qv_amp[(False, 0)]
    assert abs(on - (1.0 - SCALAR_FACTOR)) < 0.02 * (1.0 - SCALAR_FACTOR)

    # ... and with it true the moisture keeps its amplitude ...
    off = qv_amp[(True, 2)] / qv_amp[(True, 0)]
    assert abs(off - 1.0) < 1.0e-3, (
        "moist_mix6_off = true still damped the moisture checkerboard")

    # ... while theta is damped identically either way: the switch is per
    # WRF ARRAY, and theta is not in the moist array.
    for mix6_off in (False, True):
        ratio = th_amp[(mix6_off, 2)] / th_amp[(mix6_off, 0)]
        assert abs(ratio - (1.0 - FACTOR)) < 0.02 * (1.0 - FACTOR), (
            f"theta damping moved with moist_mix6_off = {mix6_off}")


@requires_gpu
def test_diff6_verification_helper_preserves_the_dry_scalar_normalization():
    """The retained verifier must measure production's distinct row strengths.

    On a periodic 2dx mode with constant horizontal mass, WRF removes F
    of theta and 3F of moisture: the dry and scalar callers pass dt and
    dt/3 respectively. Compare both to that analytic amplitude and to the
    production held tendency, so a shared but wrong normalization cannot
    pass by agreement alone.
    """
    import cupy as cp

    from woof.config import RunConfig
    from woof.core import dycore

    cfg = RunConfig(**_STEP_BASE, moist=True, mp_physics=6, km_opt=0,
                    diff_6th_opt=2, diff_6th_factor=FACTOR)
    state = _atrest(cfg)
    checker = _checkerboard(cfg.ny, cfg.nx)
    for name, offset, amplitude in (("thp", 0.0, 0.1), ("qv", 0.01, 0.001)):
        field = getattr(state, name)
        field[...] = cp.asarray(
            np.broadcast_to(offset + amplitude * checker, field.shape),
            dtype=cp.float32)
        getattr(state, name + "0")[...] = field
    before = {name: cp.asnumpy(getattr(state, name)).copy()
              for name in ("thp", "qv")}
    dycore.prepare_fixed_tendencies(state, cfg)
    chm = (state.c1h[:, None, None] * (state.mub2d + state.mup)[None]
           + state.c2h[:, None, None])
    predicted = {
        name: before[name] + cp.asnumpy(
            cp.float32(cfg.dt) * state.scratch(state.p.shape, slot) / chm)
        for name, slot in (("thp", "smag_rth"), ("qv", "smag_rqv"))
    }
    dycore.apply_diff6(state, cfg)
    for name, strength in (("thp", FACTOR), ("qv", 3.0 * FACTOR)):
        actual = cp.asnumpy(getattr(state, name))
        np.testing.assert_allclose(actual, predicted[name], rtol=2.0e-6,
                                   atol=2.0e-8, err_msg=name)
        ratio = _cb_amp(actual, checker) / _cb_amp(before[name], checker)
        np.testing.assert_allclose(ratio, 1.0 - strength,
                                   rtol=2.0e-6, atol=2.0e-7, err_msg=name)
