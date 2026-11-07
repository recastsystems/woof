# tests/test_openbc.py  (Phase 2 Task 9: open lateral boundaries + w-damping;
# Task 11 prerequisite: open-aware advection bounds + additive radiative faces)
import dataclasses
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu

_DATA = Path(__file__).parent / "data"


# ---- radiative boundary-normal velocity tendency (kernel vs mirror) --------

@requires_gpu
def test_open_u_radiative_matches_mirror():
    import cupy as cp
    from woof.core.dycore import apply_open_radiative_bc
    from woof.verify.npref import np_open_u_radiative, random_acoustic_state
    s, cfg = random_acoustic_state(seed=13)
    cfg = dataclasses.replace(cfg, open_x=True)
    before = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
              for n in ("ru_t", "u", "mup", "mub2d", "c1h", "c2h")}
    apply_open_radiative_bc(s, cfg)
    coord = type("C", (), {"c1h": before["c1h"], "c2h": before["c2h"]})()
    ref = np_open_u_radiative(before["ru_t"], before["u"],
                              before["mub2d"] + before["mup"], coord, cfg.dx)
    got = cp.asnumpy(s.ru_t)
    # only the two boundary faces are overridden; the interior is untouched
    np.testing.assert_array_equal(got[:, :, 1:-1],
                                  before["ru_t"][:, :, 1:-1].astype(np.float32))
    np.testing.assert_allclose(got[:, :, 0], ref[:, :, 0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(got[:, :, -1], ref[:, :, -1], rtol=1e-4, atol=1e-4)


@requires_gpu
def test_open_v_radiative_matches_mirror():
    import cupy as cp
    from woof.core.dycore import apply_open_radiative_bc
    from woof.verify.npref import np_open_v_radiative, random_acoustic_state
    s, cfg = random_acoustic_state(seed=17, ny=6)
    cfg = dataclasses.replace(cfg, open_y=True)
    before = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
              for n in ("rv_t", "v", "mup", "mub2d", "c1h", "c2h")}
    ru_t_before = cp.asnumpy(s.ru_t)
    apply_open_radiative_bc(s, cfg)
    # open_y alone must not touch the x-normal tendency
    np.testing.assert_array_equal(cp.asnumpy(s.ru_t), ru_t_before)
    coord = type("C", (), {"c1h": before["c1h"], "c2h": before["c2h"]})()
    ref = np_open_v_radiative(before["rv_t"], before["v"],
                              before["mub2d"] + before["mup"], coord, cfg.dy)
    got = cp.asnumpy(s.rv_t)
    np.testing.assert_array_equal(got[:, 1:-1, :],
                                  before["rv_t"][:, 1:-1, :].astype(np.float32))
    np.testing.assert_allclose(got[:, 0, :], ref[:, 0, :], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(got[:, -1, :], ref[:, -1, :], rtol=1e-4, atol=1e-4)


@requires_gpu
def test_open_radiative_outbound_only():
    # At-rest u: the west face has ub = min(-cb*m, 0) < 0 (radiation armed but
    # the gradient of a uniform field is 0); a uniform INFLOW exceeding cb
    # clamps ub to 0 at both faces -> zero radiative increment.  Task 11
    # prerequisite: the radiative term now ADDS to the accumulated tendency
    # (WRF module_advect_em.F:1252/1267 ``tendency = tendency + ...``), so a
    # clamped/zero-gradient face leaves the prior tendency value in place.
    import cupy as cp
    from woof.core.dycore import OPEN_CB, apply_open_radiative_bc
    from woof.verify.npref import random_acoustic_state
    s, cfg = random_acoustic_state(seed=19)
    cfg = dataclasses.replace(cfg, open_x=True)
    # inflow everywhere, faster than the radiation speed: west needs u > +cb,
    # east needs u < -cb to clamp -- use a west-boundary check only.
    s.u[...] = OPEN_CB + 5.0                     # strong eastward flow
    s.ru_t[...] = 777.0
    apply_open_radiative_bc(s, cfg)
    got = cp.asnumpy(s.ru_t)
    np.testing.assert_array_equal(got[:, :, 0], 777.0)  # west: clamped, adds 0
    np.testing.assert_array_equal(got[:, :, -1], 777.0)  # east: du/dx = 0


@requires_gpu
def test_step_domain_mass_closes_against_observed_boundary_flux():
    """The production acoustic mass update closes against an independently
    telescoped lateral flux, including all final-stage acoustic substeps."""
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.dycore import step
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    cfg = RunConfig(nx=10, ny=8, nz=8, dx=1000.0, dy=1000.0,
                    ztop=8000.0, dt=0.5, run_seconds=0.5,
                    open_x=True, open_y=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, _stratified_sounding,
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    state.u[...] = cp.asarray(
        np.linspace(-2.0, 5.0, cfg.nx + 1, dtype=np.float32)
        [None, None, :])
    state.v[...] = cp.asarray(
        np.linspace(-1.0, 3.0, cfg.ny + 1, dtype=np.float32)
        [None, :, None])

    mass0 = float(cp.sum(
        (state.mub2d + state.mup).astype(cp.float64)))
    increments = []
    step(state, cfg, mass_flux_observer=increments.append)
    mass1 = float(cp.sum(
        (state.mub2d + state.mup).astype(cp.float64)))

    assert len(increments) == cfg.time_step_sound
    assert abs(sum(increments)) > 0.0
    residual = abs((mass1 - mass0) - sum(increments)) / mass0
    assert residual < 1.0e-5


# ---- w-damping (WRF w_damp, module_big_step_utilities_em.F) -----------------

@requires_gpu
def test_w_damp_matches_mirror_and_threshold():
    import cupy as cp
    from woof.core.dycore import apply_w_damping
    from woof.verify.npref import np_w_damp, random_acoustic_state
    s, cfg = random_acoustic_state(seed=23)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    cfg = dataclasses.replace(cfg, w_damping=1)
    rng = np.random.default_rng(29)
    # synthetic Omega spanning both sides of the CFL threshold: vert_cfl =
    # |ww/(c1f*mu+c2f)*rdnw*dt|, mu ~ 6.4e4, rdnw ~ -8, dt = 3 -> the unit
    # Courant number needs |ww| ~ 2.7e3; use 0..2x that.
    ww = (5.4e3 * rng.uniform(-1.0, 1.0, (nz + 1, ny, nx))).astype(np.float32)
    w = rng.standard_normal((nz + 1, ny, nx)).astype(np.float32)
    s.w[...] = cp.asarray(w)
    rw0 = rng.standard_normal((nz + 1, ny, nx)).astype(np.float32)
    s.rw_t[...] = cp.asarray(rw0)
    mut = cp.asnumpy(s.mub2d + s.mup).astype(np.float64)
    coord = type("C", (), {
        "c1f": cp.asnumpy(s.c1f).astype(np.float64),
        "c2f": cp.asnumpy(s.c2f).astype(np.float64),
        "rdnw": cp.asnumpy(s.rdnw).astype(np.float64)})()
    apply_w_damping(s, cfg, cp.asarray(ww))
    ref = np_w_damp(rw0.astype(np.float64), ww.astype(np.float64),
                    w.astype(np.float64), mut, coord, cfg.dt)
    got = cp.asnumpy(s.rw_t)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
    # engages ONLY above the threshold: sub-threshold points are bitwise
    # untouched, and at least one point on each side exists in this draw.
    c1f = coord.c1f[:, None, None]
    c2f = coord.c2f[:, None, None]
    cfl = np.zeros((nz + 1, ny, nx))
    cfl[1:nz] = np.abs(ww[1:nz] / (c1f[1:nz] * mut[None] + c2f[1:nz])
                       * coord.rdnw[1:nz, None, None] * cfg.dt)
    quiet = cfl <= 1.0
    assert quiet[1:nz].any() and (~quiet[1:nz]).any()
    np.testing.assert_array_equal(got[quiet], rw0[quiet])
    # boundary w levels never damped (WRF loops k = 2, kde-1)
    np.testing.assert_array_equal(got[0], rw0[0])
    np.testing.assert_array_equal(got[nz], rw0[nz])


@requires_gpu
def test_w_damp_noop_when_disabled():
    import cupy as cp
    from woof.core.dycore import apply_w_damping
    from woof.verify.npref import random_acoustic_state
    s, cfg = random_acoustic_state(seed=31)
    ww = cp.full((cfg.nz + 1, cfg.ny, cfg.nx), 1.0e5, dtype=cp.float32)
    before = cp.asnumpy(s.rw_t)
    apply_w_damping(s, cfg, ww)                  # cfg.w_damping = 0 default
    np.testing.assert_array_equal(cp.asnumpy(s.rw_t), before)


# ---- acoustic-substep boundary gate (mirror coverage) -----------------------

@requires_gpu
def test_acoustic_open_boundary_faces_match_mirror():
    # With open BCs the boundary-normal u''/v'' at the boundary faces feel
    # only the large-step tendency (WRF advance_uv skips their pressure
    # gradient); interior faces are the periodic update.
    import cupy as cp
    from woof.core.acoustic import acoustic_substep_explicit
    from woof.verify.npref import (np_advance_uv, random_acoustic_state,
                                    s_meta)
    s, cfg = random_acoustic_state(seed=37, ny=6)
    cfg = dataclasses.replace(cfg, open_x=True, open_y=True)
    before = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
              for n in ("u_pp", "v_pp", "mu_pp", "th_pp", "p_pp",
                        "ph_pp", "p_pp_old")}
    acoustic_substep_explicit(s, cfg, dtau=1.0, first=False)
    u_ref, v_ref = np_advance_uv(before, s_meta(s), cfg, dtau=1.0)
    np.testing.assert_allclose(cp.asnumpy(s.u_pp), u_ref, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(s.v_pp), v_ref, rtol=1e-4, atol=1e-6)
    # and the gate really changed the boundary faces vs the periodic answer
    cfg_per = dataclasses.replace(cfg, open_x=False, open_y=False)
    u_per, v_per = np_advance_uv(before, s_meta(s), cfg_per, dtau=1.0)
    assert np.max(np.abs(u_ref[:, :, 0] - u_per[:, :, 0])) > 0.0
    assert np.max(np.abs(v_ref[:, 0, :] - v_per[:, 0, :])) > 0.0


# ---- zero-gradient outbound scalars/theta ------------------------------------

@requires_gpu
def test_open_zero_gradient_outbound_only():
    import cupy as cp
    from woof.core.dycore import apply_open_zero_gradient
    from woof.verify.npref import random_acoustic_state
    s, cfg = random_acoustic_state(seed=41, ny=6)
    cfg = dataclasses.replace(cfg, open_x=True)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    # west boundary: rows 0..2 outbound (u < 0), rows 3.. inbound
    u = np.zeros((nz, ny, nx + 1), dtype=np.float32)
    u[:, :3, 0] = -1.0
    u[:, 3:, 0] = 1.0
    u[:, :, -1] = -1.0                            # east face: inflow everywhere
    s.u[...] = cp.asarray(u)
    thp = cp.asnumpy(s.thp).copy()
    w = cp.asnumpy(s.w).copy()
    apply_open_zero_gradient(s, cfg)
    got_th = cp.asnumpy(s.thp)
    got_w = cp.asnumpy(s.w)
    # theta': outbound rows copied from the interior neighbour, inbound kept
    np.testing.assert_array_equal(got_th[:, :3, 0], thp[:, :3, 1])
    np.testing.assert_array_equal(got_th[:, 3:, 0], thp[:, 3:, 0])
    np.testing.assert_array_equal(got_th[:, :, -1], thp[:, :, -1])  # inflow
    # w: interior levels only; boundary levels 0/nz are BC-pinned
    np.testing.assert_array_equal(got_w[1:nz, :3, 0], w[1:nz, :3, 1])
    np.testing.assert_array_equal(got_w[0], w[0])
    np.testing.assert_array_equal(got_w[nz], w[nz])
    # periodic default: exact no-op
    s2, cfg2 = random_acoustic_state(seed=41, ny=6)
    th2 = cp.asnumpy(s2.thp).copy()
    apply_open_zero_gradient(s2, cfg2)
    np.testing.assert_array_equal(cp.asnumpy(s2.thp), th2)


# ---- integration: at rest + radiation ---------------------------------------

def _stratified_sounding(z):
    # constant N^2 = 1e-4 s^-2: theta(z) = 300*exp(N^2 z / g)
    return 300.0 * np.exp(1.0e-4 * np.asarray(z, dtype=np.float64) / 9.81)


@requires_gpu
def test_open_at_rest_stays_at_rest(monkeypatch):
    # Open BCs + w-damping on an at-rest balanced state must add nothing:
    # the only w signal is the Phase-1 FP32 discrete-balance residual.
    # Since 890e2523a a radiative open domain forms rhs_ph with WRF's
    # native operation order (slow_geopotential_open) at every point, the
    # interior included, while a periodic domain keeps the fused default
    # slow_geopotential, so the two residuals round differently in w and
    # php (1.5e-11 in w, 5.4e-9 in php after 50 steps, measured on an
    # RTX 5070 Ti).  The periodic reference therefore forms rhs_ph with
    # that same native kernel, both boundaries closed so its stencils wrap,
    # and the open run must equal it bit for bit: the open rows, radiation
    # and w-damping contribute exactly nothing.
    import cupy as cp
    from woof.config import RunConfig
    from woof.core import dycore
    from woof.core.dycore import run_steps
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    def native_rhs_ph(state, cfg, ww, *, add_vertical):
        dycore._launch_open_geopotential(state, cfg, ww,
                                         add_vertical=add_vertical)

    def run(open_bc, native=False):
        cfg = RunConfig(nx=16, ny=8, nz=10, dx=500.0, dy=500.0, ztop=8000.0,
                        dt=2.0, run_seconds=0.0,
                        open_x=open_bc, open_y=open_bc,
                        w_damping=1 if open_bc else 0)
        coord = make_vertical_coord(cfg.nz)
        base = make_base_state(coord, _stratified_sounding, p_surf=cfg.p_surf,
                               ztop=cfg.ztop)
        s = init_at_rest(cfg, coord, base)
        with monkeypatch.context() as patch:
            if native:
                patch.setattr(dycore, "_launch_slow_geopotential",
                              native_rhs_ph)
            run_steps(s, cfg, 50)
        return s

    s_open, s_native, s_per = run(True), run(False, native=True), run(False)
    for name in ("u", "v", "w", "thp", "php", "mup"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(s_open, name)),
                                      cp.asnumpy(getattr(s_native, name)))
    # The fields rhs_ph does not round stay identical to the default
    # periodic run as well.
    for name in ("u", "v", "thp", "mup"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(s_open, name)),
                                      cp.asnumpy(getattr(s_per, name)))
    assert float(cp.abs(s_open.u).max()) < 1e-5
    assert float(cp.abs(s_open.w).max()) < 1e-3   # FP32 balance residual


def _packet_energy(s, cfg, i0, i1):
    """Float64 perturbation energy proxy over cells [i0, i1) of a 2-D run:
    kinetic (u at faces i0..i1-1, interior w) + available potential
    (g*theta'/(N*theta_b))^2, unweighted sums (uniform-enough column)."""
    import cupy as cp
    g, n2 = 9.81, 1.0e-4
    u = cp.asnumpy(s.u[:, :, i0:i1]).astype(np.float64)
    w = cp.asnumpy(s.w[1:-1, :, i0:i1]).astype(np.float64)
    thp = cp.asnumpy(s.thp[:, :, i0:i1]).astype(np.float64)
    thb = cp.asnumpy(s.thb).astype(np.float64)[:, None, None]
    ape = (g * thp / (np.sqrt(n2) * thb)) ** 2
    return float(np.sum(u * u) + np.sum(w * w) + np.sum(ape))


@requires_gpu
def test_gravity_wave_packet_radiates():
    # Acceptance gate: a gravity-wave packet exits the open x boundaries with
    # < 10% reflected amplitude, measured against a doubled-width periodic
    # control (which holds the true no-reflection solution in the subdomain).
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.dycore import run_steps
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_theta_perturbation

    nx, nz, dx, ztop = 192, 40, 500.0, 8000.0
    T = 3000.0
    dt = 2.0

    def build(nx_run, open_x, x0):
        cfg = RunConfig(nx=nx_run, ny=1, nz=nz, dx=dx, dy=dx, ztop=ztop,
                        dt=dt, run_seconds=0.0, open_x=open_x)
        coord = make_vertical_coord(nz)
        base = make_base_state(coord, _stratified_sounding,
                               p_surf=cfg.p_surf, ztop=ztop)

        def thp_func(x, z):
            # mode-1 vertical structure (phase speed ~ N*H/pi ~ 25 m/s,
            # matched to the OPEN_CB = 25 radiation speed of the KW-open
            # BC, WRF's cb), compact in x
            envelope = 0.1 * np.exp(-((x - x0) / 10000.0) ** 2)
            return (np.sin(np.pi * np.asarray(z) / ztop)[:, None, None]
                    * envelope[None, None, :])

        s = init_theta_perturbation(cfg, coord, base, thp_func)
        return s, cfg

    L = nx * dx
    s_open, cfg_open = build(nx, True, 0.0)            # packet at domain centre
    s_ctl, cfg_ctl = build(2 * nx, False, -L / 2.0)    # same absolute position
    e0 = _packet_energy(s_open, cfg_open, 0, nx)
    assert e0 > 0.0

    nsteps = int(round(T / dt))
    run_steps(s_open, cfg_open, nsteps)
    run_steps(s_ctl, cfg_ctl, nsteps)

    e_open = _packet_energy(s_open, cfg_open, 0, nx)
    e_true = _packet_energy(s_ctl, cfg_ctl, 0, nx)     # control subdomain
    assert np.isfinite(e_open) and np.isfinite(e_true)
    # the packet really left the subdomain in the control run
    assert e_true < 0.05 * e0, (e_true, e0)
    # reflected amplitude = sqrt(excess energy / initial packet energy)
    refl = np.sqrt(max(e_open - e_true, 0.0) / e0)
    assert refl < 0.10, (refl, e_open, e_true, e0)


# ---- external-mode divergence damping (WRF emdiv, Task 11) ------------------

def test_emdiv_mudf_zeroed_only_at_stage_one():
    """WRF small_step_prep zeros MUDF under IF(rk_step==1) ONLY.

    Zeroing every RK stage (the pre-fix behavior) discards the carried
    column-mass tendency and applies emdiv with history on 4 of 7 acoustic
    iterations instead of WRF's 6 of 7: the d03 MSLP ringing source.
    """
    import ast
    import woof.core.dycore as dycore_mod
    source = Path(dycore_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    zero_sites = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "mudf"):
            zero_sites.append(node)
    assert len(zero_sites) == 1
    guards = [node for node in ast.walk(tree)
              if isinstance(node, ast.If)
              and any(z in ast.walk(node) for z in zero_sites)
              and isinstance(node.test, ast.Compare)
              and isinstance(node.test.left, ast.Name)
              and node.test.left.id == "istage"]
    assert guards, "mudf zeroing must be guarded by the istage == 0 test"


@requires_gpu
def test_emdiv_filter_matches_reference():
    import cupy as cp
    from woof.core.dycore import apply_emdiv_filter
    from woof.verify.npref import np_emdiv_uv, random_acoustic_state
    for open_flags in ((False, False), (True, True)):
        s, cfg = random_acoustic_state(seed=47, ny=6)
        cfg = dataclasses.replace(cfg, emdiv=0.01,
                                  open_x=open_flags[0], open_y=open_flags[1])
        rng = np.random.default_rng(53)
        mudf = rng.standard_normal((cfg.ny, cfg.nx)).astype(np.float32)
        u0 = cp.asnumpy(s.u_pp).astype(np.float64)
        v0 = cp.asnumpy(s.v_pp).astype(np.float64)
        coord = type("C", (), {"c1h": cp.asnumpy(s.c1h).astype(np.float64)})()
        apply_emdiv_filter(s, cfg, cp.asarray(mudf))
        u_ref, v_ref = np_emdiv_uv(u0, v0, mudf.astype(np.float64),
                                   coord, cfg)
        np.testing.assert_allclose(cp.asnumpy(s.u_pp), u_ref,
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(cp.asnumpy(s.v_pp), v_ref,
                                   rtol=1e-5, atol=1e-5)
        if cfg.open_x:                      # boundary faces excluded
            np.testing.assert_array_equal(cp.asnumpy(s.u_pp)[:, :, 0],
                                          u0[:, :, 0].astype(np.float32))


@requires_gpu
def test_advance_mu_th_stores_mudf_before_lossy_mass_update():
    """WRF stores ``dmdt + mu_tend`` directly, before updating FP32 mu.

    A sub-ULP mass increment must therefore remain visible to emdiv even
    though adding it to a large ``mu_pp`` rounds the state back to itself.
    Reconstructing MUDF by subtracting the two rounded states loses it.
    """
    import cupy as cp
    from woof.core.acoustic import acoustic_substep_explicit
    from woof.core.state import DTYPE
    from woof.verify.npref import random_acoustic_state

    s, cfg = random_acoustic_state(seed=161, ny=6)
    for name in ("u", "v", "u_pp", "v_pp", "ru_t", "rv_t", "p_pp",
                 "p_pp_old", "ph_pp", "php", "al_pp", "rth_t"):
        getattr(s, name)[...] = 0.0
    large = DTYPE(2**24)
    s.mu_pp[...] = large
    s.rmu_t[...] = DTYPE(1.0)
    mudf = cp.full(s.mup.shape, cp.nan, dtype=DTYPE)

    acoustic_substep_explicit(
        s, cfg, dtau=0.25, first=True, mudf=mudf)

    np.testing.assert_array_equal(cp.asnumpy(s.mu_pp), np.float32(large))
    np.testing.assert_array_equal(cp.asnumpy(mudf), np.float32(1.0))


@requires_gpu
@pytest.mark.parametrize("open_bc", [False, True])
def test_emdiv_mudf_direct_storage_matches_mirror_dmdt(open_bc):
    """The ``mudf`` driving
    ``apply_emdiv_filter`` is WRF ``advance_mu_t``'s column-mass tendency
    ``dmdt + mu_tend``, stored directly before the rounded mass update.
    Drive three substeps and compare that stored value with the independent
    float64 mirror-derived tendency, plus the first-substep no-op.
    """
    import cupy as cp
    from woof.core.acoustic import acoustic_substep
    from woof.core.dycore import apply_emdiv_filter
    from woof.core.state import DTYPE
    from woof.verify.npref import (np_advance_mu_th, np_advance_uv,
                                    random_acoustic_state, snapshot)
    s, cfg = random_acoustic_state(seed=61, ny=6)
    cfg = dataclasses.replace(cfg, emdiv=0.01,
                              open_x=open_bc, open_y=open_bc)
    dtau = 1.0
    mudf = cp.zeros((cfg.ny, cfg.nx), dtype=DTYPE)   # WRF small_step_prep
    for i in range(3):
        u_pre = cp.asnumpy(s.u_pp)
        v_pre = cp.asnumpy(s.v_pp)
        apply_emdiv_filter(s, cfg, mudf)             # previous substep's mudf
        if i == 0:                                   # zeroed mudf: exact no-op
            np.testing.assert_array_equal(cp.asnumpy(s.u_pp), u_pre)
            np.testing.assert_array_equal(cp.asnumpy(s.v_pp), v_pre)
        else:                                        # engaged afterwards
            assert np.abs(cp.asnumpy(s.u_pp) - u_pre).max() > 0.0
        pp = snapshot(s)                             # filtered pre-substep
        acoustic_substep(s, cfg, dtau, first=(i == 0), mudf=mudf)
        u_new, v_new = np_advance_uv(pp, pp, cfg, dtau, first=(i == 0))
        mu_new, _, _ = np_advance_mu_th(
            {**pp, "u_pp": u_new, "v_pp": v_new}, pp, cfg, dtau)
        ref = (mu_new - pp["mu_pp"]) / dtau          # mirror dmdt + mu_tend
        np.testing.assert_allclose(cp.asnumpy(mudf), ref,
                                   rtol=1e-4, atol=2e-3)
        assert np.abs(ref).max() > 1.0               # non-vacuous magnitude


@requires_gpu
def test_step_emdiv_uses_direct_mudf_wiring():
    """Production passes MUDF into the acoustic mass kernel and engages it."""
    import inspect

    import cupy as cp
    from woof.core import dycore
    from woof.verify.npref import random_acoustic_state
    source = inspect.getsource(dycore.step)
    assert "mudf=mudf" in source
    assert "launch_emdiv_mudf" not in source
    s, cfg = random_acoustic_state(seed=67, ny=6)
    cfg = dataclasses.replace(cfg, emdiv=0.01)
    dycore.step(s, cfg)
    mudf = s.scratch((cfg.ny, cfg.nx), "acoustic_mudf")
    assert float(cp.abs(mudf).max()) > 0.0


# ---- acoustic advance_mu_th open-boundary ghost reads ------------------------

@requires_gpu
def test_acoustic_mu_th_open_matches_mirror():
    # With open BCs the advance_mu_th kernel's cross-boundary neighbour
    # reads clamp (WRF zero-gradient ghosts) instead of wrapping: kernel vs
    # the float64 mirrors with the same open cfg, plus proof that the open
    # answer differs from the periodic one at the boundary cells.
    import cupy as cp
    from woof.core.acoustic import acoustic_substep_explicit
    from woof.verify.npref import (np_advance_mu_th, np_advance_uv,
                                    random_acoustic_state, snapshot)
    s, cfg = random_acoustic_state(seed=59, ny=6)
    cfg = dataclasses.replace(cfg, open_x=True, open_y=True)
    pp = snapshot(s)
    acoustic_substep_explicit(s, cfg, dtau=1.0, first=False)
    u_new, v_new = np_advance_uv(pp, pp, cfg, dtau=1.0)
    mu_ref, ww_ref, th_ref = np_advance_mu_th(
        {**pp, "u_pp": u_new, "v_pp": v_new}, pp, cfg, dtau=1.0)
    np.testing.assert_allclose(cp.asnumpy(s.mu_pp), mu_ref,
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cp.asnumpy(s.th_pp), th_ref,
                               rtol=1e-4, atol=2e-3)
    np.testing.assert_allclose(cp.asnumpy(s.ww_pp), ww_ref,
                               rtol=1e-4, atol=1e-4)
    cfg_per = dataclasses.replace(cfg, open_x=False, open_y=False)
    mu_per, _, th_per = np_advance_mu_th(
        {**pp, "u_pp": u_new, "v_pp": v_new}, pp, cfg_per, dtau=1.0)
    assert np.max(np.abs(th_ref[:, :, 0] - th_per[:, :, 0])) > 0.0
    assert np.max(np.abs(mu_ref[0, :] - mu_per[0, :])) > 0.0


# ---- Task 11 prerequisite: open-aware advection bounds ----------------------

def _advection_inputs(seed=20260713, nz=12, ny=12, nx=16):
    """Random staggered fields + coupled fluxes for the flux-div kernels."""
    rng = np.random.default_rng(seed)
    fields = {
        "q": rng.normal(300, 5, (nz, ny, nx)),
        "u": rng.normal(0, 5, (nz, ny, nx + 1)),
        "v": rng.normal(0, 5, (nz, ny + 1, nx)),
        "w": rng.normal(0, 1, (nz + 1, ny, nx)),
        "ru": rng.normal(0, 10, (nz, ny, nx + 1)),
        "rv": rng.normal(0, 10, (nz, ny + 1, nx)),
        "rw": rng.normal(0, 1, (nz + 1, ny, nx)),
    }
    fields["w"][0] = fields["w"][-1] = 0.0
    fields["rw"][0] = fields["rw"][-1] = 0.0
    return {k: a.astype(np.float32) for k, a in fields.items()}


@requires_gpu
def test_advection_periodic_path_bitwise_regression():
    # Existing periodic tendencies remain BITWISE unchanged below the lid:
    # compare against outputs captured from the pre-change kernels on a
    # fixed seed (tests/data/advection_periodic_regression.npz).
    #
    # PER CARD since 2026-09-17 (tests/_card_pins.py).  The original
    # capture (2026-07-19, card unrecorded) against the same launchers on
    # test host 1's RTX 4090 (compute capability 8.9, driver 610.57.04):
    # tend_scalar differs in 446 of 1536 words, max 1.953e-03 on a field of
    # scale 1.5e+04 (max relative 4.4e-06, about one ULP), which is the
    # reading the 2026-09-03 recapture note (962c068da) took on an RTX 3080
    # against the same file: the 3080 and the 4090 agree with each other
    # and not with the box that made the original.  advection.cu was not
    # edited between the capture and either reading.  The 4090 file
    # (tests/data/advection_periodic_regression.rtx4090.npz) was written by
    # tools/recapture_card_pins.py --pin advection_periodic_regression with
    # its per-entry reading beside it in tests/data/receipts/pin-gates/;
    # the inputs are the original file's.  A card with no file skips.
    import cupy as cp
    import _card_pins
    from woof.core.advection import (launch_flux_div_scalar,
                                      launch_flux_div_u, launch_flux_div_v,
                                      launch_flux_div_w)
    from woof.core.grid import make_vertical_coord
    reason = _card_pins.skip_reason("advection_periodic_regression")
    if reason is not None:
        pytest.skip(reason)
    ref = np.load(_card_pins.path("advection_periodic_regression"))
    nz, ny, nx = ref["q"].shape
    coord = make_vertical_coord(nz)
    d = {k: cp.asarray(ref[k]) for k in ("q", "u", "v", "w", "ru", "rv", "rw")}
    for field, launcher, key in (
            (d["q"], launch_flux_div_scalar, "tend_scalar"),
            (d["u"], launch_flux_div_u, "tend_u"),
            (d["v"], launch_flux_div_v, "tend_v"),
            (d["w"], launch_flux_div_w, "tend_w")):
        tend = cp.zeros(ref[key].shape, cp.float32)
        launcher(field, d["ru"], d["rv"], d["rw"], tend, coord, 100.0, 100.0)
        got = cp.asnumpy(tend)
        if key == "tend_w":
            # The old capture omitted WRF's nonzero lid contribution.
            # Native Fortran lid controls and full-array word receipts in
            # test_advect_wrf471_parity.py now guard that corrected row.
            np.testing.assert_array_equal(got[:-1], ref[key][:-1])
        else:
            np.testing.assert_array_equal(got, ref[key])


@requires_gpu
@pytest.mark.parametrize("open_x,open_y",
                         [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("which", ["scalar", "u", "v", "w"])
def test_flux_div_open_matches_mirror(which, open_x, open_y):
    # WRF open-BC advection (module_advect_em.F): degraded near-boundary
    # stencils, boundary-normal loop-bound exclusions, and the non-cb open
    # advective terms at the boundary cells -- device kernel vs the float64
    # npref mirror with the same open flags.
    import cupy as cp
    from woof.core import advection as adv
    from woof.core.grid import make_vertical_coord
    from woof.verify import npref
    f = _advection_inputs()
    nz, ny, nx = f["q"].shape
    coord = make_vertical_coord(nz)
    launchers = {"scalar": adv.launch_flux_div_scalar,
                 "u": adv.launch_flux_div_u,
                 "v": adv.launch_flux_div_v,
                 "w": adv.launch_flux_div_w}
    mirrors = {"scalar": npref.np_flux_div_scalar,
               "u": npref.np_flux_div_u,
               "v": npref.np_flux_div_v,
               "w": npref.np_flux_div_w}
    key = "q" if which == "scalar" else which
    args64 = [f[key].astype(np.float64), f["ru"].astype(np.float64),
              f["rv"].astype(np.float64), f["rw"].astype(np.float64)]
    ref = mirrors[which](*args64, coord, 100.0, 100.0,
                         open_x=open_x, open_y=open_y)
    tend = cp.zeros(ref.shape, cp.float32)
    launchers[which](cp.asarray(f[key]), cp.asarray(f["ru"]),
                     cp.asarray(f["rv"]), cp.asarray(f["rw"]), tend, coord,
                     100.0, 100.0, open_x=open_x, open_y=open_y)
    np.testing.assert_allclose(cp.asnumpy(tend), ref, rtol=5e-4, atol=5e-4)


@requires_gpu
@pytest.mark.parametrize("which", ["scalar", "u", "v", "w"])
def test_flux_div_open_msf_matches_mirror(which):
    # The real74 production path: specified/open boundaries + non-uniform
    # map factors.  WRF weights the horizontal flux divergence by the map
    # factor at the tendency point under open/specified exactly as in the
    # interior (module_advect_em.F mrdx = msftx*rdx at :3644, msfux :740,
    # msfvy :2676, msftx :5096) -- kernel vs float64 npref mirror.
    import cupy as cp
    from woof.core import advection as adv
    from woof.core.grid import make_vertical_coord
    from woof.verify import npref
    f = _advection_inputs()
    nz, ny, nx = f["q"].shape
    coord = make_vertical_coord(nz)
    launchers = {"scalar": adv.launch_flux_div_scalar,
                 "u": adv.launch_flux_div_u,
                 "v": adv.launch_flux_div_v,
                 "w": adv.launch_flux_div_w}
    mirrors = {"scalar": npref.np_flux_div_scalar,
               "u": npref.np_flux_div_u,
               "v": npref.np_flux_div_v,
               "w": npref.np_flux_div_w}
    shapes = {"scalar": (ny, nx), "u": (ny, nx + 1), "v": (ny + 1, nx),
              "w": (ny, nx)}
    rng = np.random.default_rng(17)
    msf = (1.0 + 0.05 * rng.random(shapes[which])).astype(np.float32)
    key = "q" if which == "scalar" else which
    args64 = [f[key].astype(np.float64), f["ru"].astype(np.float64),
              f["rv"].astype(np.float64), f["rw"].astype(np.float64)]
    ref = mirrors[which](*args64, coord, 100.0, 100.0,
                         open_x=True, open_y=True,
                         msf=msf.astype(np.float64))
    tend = cp.zeros(ref.shape, cp.float32)
    launchers[which](cp.asarray(f[key]), cp.asarray(f["ru"]),
                     cp.asarray(f["rv"]), cp.asarray(f["rw"]), tend, coord,
                     100.0, 100.0, open_x=True, open_y=True,
                     msf=cp.asarray(msf))
    np.testing.assert_allclose(cp.asnumpy(tend), ref, rtol=5e-4, atol=5e-4)


@requires_gpu
def test_flux_div_specified_msf_hand_pin():
    """Specified + non-uniform msf regression pin, hand-computed.

    Constant scalar q == 1 makes every flux-stencil order reduce to the
    advecting face flux itself, so with ru = a*i, rv = b*j, rw = 0 the WRF
    tendency is hand-computable at every cell (module_advect_em.F
    advect_scalar):

      interior cells:        -msf(j,i) * (a/dx + b/dy)     (mrdx/mrdy)
      x-boundary cells:      -a/dx - msf(j,i)*b/dy   (the open non-cb term
                             carries plain rdx, F:4119-4126; ub-term
                             vanishes for constant q)
      y-boundary cells:      -msf(j,i)*a/dx - b/dy
      corner cells:          -a/dx - b/dy

    Pre-fix the open/specified kernel path dropped msf entirely and
    produced -(a/dx + b/dy) everywhere, so this pins the fix wherever
    msf != 1.
    """
    import cupy as cp
    from woof.core.advection import launch_flux_div_scalar
    from woof.core.grid import make_vertical_coord
    nz, ny, nx = 4, 8, 9
    dx = dy = 100.0
    a, b = 3.0, -2.0
    coord = make_vertical_coord(nz)
    q = cp.ones((nz, ny, nx), cp.float32)
    ru = cp.asarray(np.broadcast_to(a * np.arange(nx + 1, dtype=np.float32),
                                    (nz, ny, nx + 1)))
    rv = cp.asarray(np.broadcast_to(
        b * np.arange(ny + 1, dtype=np.float32)[:, None], (nz, ny + 1, nx)))
    rw = cp.zeros((nz + 1, ny, nx), cp.float32)
    rng = np.random.default_rng(5)
    msf = (0.96 + 0.05 * rng.random((ny, nx))).astype(np.float32)
    tend = cp.zeros((nz, ny, nx), cp.float32)
    launch_flux_div_scalar(q, ru, rv, rw, tend, coord, dx, dy,
                           open_x=True, open_y=True, msf=cp.asarray(msf))
    got = cp.asnumpy(tend)

    xbnd = np.zeros((ny, nx), bool)
    xbnd[:, 0] = xbnd[:, -1] = True
    ybnd = np.zeros((ny, nx), bool)
    ybnd[0, :] = ybnd[-1, :] = True
    wx = np.where(xbnd, 1.0, msf.astype(np.float64))   # plain rdx at non-cb
    wy = np.where(ybnd, 1.0, msf.astype(np.float64))
    expected = -(wx * a / dx + wy * b / dy)
    np.testing.assert_allclose(got, np.broadcast_to(expected, (nz, ny, nx)),
                               rtol=1e-5, atol=1e-6)


@requires_gpu
@pytest.mark.parametrize("axis,shear",
                         [("x", False), ("x", True), ("y", False)])
def test_open_sheared_flow_stays_uniform(axis, shear):
    """Task 11 prerequisite acceptance: uniform boundary-normal flow through
    open boundaries stays uniform -- including the 3-cell boundary strip --
    after 500 steps, while a tangential-wind packet (an exactly passive
    tracer of the dycore: no y-variation => no divergence, no pressure
    response) rides the flow OUT of the domain and does not come back.

    Against the Task 9 kernels this fails by construction: the periodic
    stencil wrap re-injects the outgoing packet at the inflow boundary and
    it circulates forever (max|tangential| stays at the ~1 m/s packet
    amplitude).  With the WRF open-aware bounds the packet exits and the
    domain returns to the uniform state.

    Bound (documented in the task-11 report): 0.05 m/s = 5% of the packet
    amplitude, covering the degraded-stencil dispersive tail that has not
    fully flushed plus FP32 noise; measured residuals are well below it.
    """
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.dycore import run_steps
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    n_along, n_across, nz = 20, 4, 10
    dx, ztop, dt, nsteps = 1000.0, 10000.0, 6.0, 500
    if axis == "x":
        cfg = RunConfig(nx=n_along, ny=n_across, nz=nz, dx=dx, dy=dx,
                        ztop=ztop, dt=dt, run_seconds=0.0, open_x=True)
    else:
        cfg = RunConfig(nx=n_across, ny=n_along, nz=nz, dx=dx, dy=dx,
                        ztop=ztop, dt=dt, run_seconds=0.0, open_y=True)
    coord = make_vertical_coord(nz)
    base = make_base_state(coord, _stratified_sounding, p_surf=cfg.p_surf,
                           ztop=ztop)
    s = init_at_rest(cfg, coord, base)
    z = s.height_half()
    prof = (8.0 + 4.0 * z / ztop) if shear else np.full(nz, 10.0)
    prof32 = cp.asarray(prof, dtype=cp.float32)[:, None, None]
    # packet: gaussian along the flow axis, 5 km upstream of center, uniform
    # across and in z; 1 m/s amplitude
    xc = (np.arange(n_along) + 0.5) * dx - 0.5 * n_along * dx
    packet = np.exp(-((xc + 5000.0) / 1200.0) ** 2).astype(np.float32)
    if axis == "x":
        s.u[...] = prof32
        s.v += cp.asarray(packet)[None, None, :]
        normal, tangential = s.u, s.v
    else:
        s.v[...] = prof32
        s.u += cp.asarray(packet)[None, :, None]
        normal, tangential = s.v, s.u

    run_steps(s, cfg, nsteps)
    n_err = float(cp.abs(normal - prof32).max())
    t_err = float(cp.abs(tangential).max())
    w_err = float(cp.abs(s.w).max())
    assert np.isfinite(n_err) and np.isfinite(t_err) and np.isfinite(w_err)
    assert t_err < 0.05, f"tangential packet residual {t_err}"
    assert n_err < 0.05, f"boundary-normal wind drifted {n_err}"
    assert w_err < 0.05, f"w developed {w_err}"


@requires_gpu
@pytest.mark.parametrize("axis", ["x", "y"])
def test_open_sheared_flow_uniform_with_dissipation(axis):
    """Phase 3 Task 2: sheared-flow open-BC regression WITH the production
    dissipation package enabled -- km_opt=4 (2-D Smagorinsky, c_s=0.25) +
    the monotonic 6th-order filter (diff_6th_opt=2, factor 0.12) + WRF's
    emdiv=0.01 external-mode damper -- the exact configuration in which
    the Task-12 final-review strip leak hid (the Phase 2 sheared-flow test
    above runs NO dissipation, so the open-BC dissipation bounds were
    never integration-tested).

    Phase 1 (integration): a compact BOUNDARY-NORMAL wind pulse rides the
    sheared through-flow out of the open boundary; after 500 steps the
    flow must return to the uniform sheared profile (measured residuals on
    the fixed tree: normal 0.0029 m/s, tangential 0.0, w 0.0014 m/s --
    gates at 0.02/1e-3/0.01 carry ~7x margin).  This pins the whole
    open-BC dissipation path against gross regressions (the pre-Task-11
    wrapped strips drifted 2e-2 in mass and blew up the boundary strip).

    Phase 2 (operator, on the integration-evolved state): fresh pulses are
    planted by both boundaries and ``apply_diff6`` is applied exactly as
    ``step`` applies it; the WRF ``sixth_order_diffusion`` open-BC
    exclusion strips must be EXACT zero increments -- 3 entries per open
    side on EVERY axis and stagger (WRF's loop bounds) -- while the first
    live entries inboard are nonzero, including the outermost computed
    staggered-normal face (u's nx-3 under open_x / v's ny-3 under
    open_y), which WRF computes reading the true boundary datum
    field(ide) and the boundary-aware kernel now computes the same way
    (diff6.cu ``bndx``/``bndy``; RED-pinned value-wise in
    tests/test_diff6_boundary_face.py -- the d3db03b-era tree zeroed that
    face as a documented conservative deviation, and the pre-d3db03b tree
    computed it from a WRAPPED flux, which is neither).  The
    once-per-step Smagorinsky tendencies honor WRF's width-1 exclusion.

    Accuracy note (measured while calibrating): in phase-1 flow space the
    HISTORICAL single-face leak is masked by the monotonic limiter to
    ~1e-3 m/s (pre-fix 0.0043 vs post-fix 0.0029 final residual) -- which
    is exactly why it hid in WK82 -- so the amplitude gates alone cannot
    resolve it; the phase-2 exact-strip assertions carry that
    discrimination (verified: this test fails on the pre-fix tree).
    """
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.dycore import (_PROGNOSTICS, add_smag2d_tendencies,
                                   apply_diff6, run_steps)
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import PhysicsTendencies
    from woof.core.state import init_at_rest

    class _ZeroPBL:
        """Keep this regression scoped to the horizontal/open operators."""

        def __init__(self, state):
            self.tendencies = PhysicsTendencies.zeros(state)

        def compute(self, _state, _cfg):
            return self.tendencies

    n_along, n_across, nz = 20, 4, 10
    dx, ztop, dt, nsteps = 1000.0, 10000.0, 6.0, 500
    diss = dict(km_opt=4, c_s=0.25, diff_6th_opt=2, diff_6th_factor=0.12,
                emdiv=0.01, bl_pbl_physics=1)
    if axis == "x":
        cfg = RunConfig(nx=n_along, ny=n_across, nz=nz, dx=dx, dy=dx,
                        ztop=ztop, dt=dt, run_seconds=0.0, open_x=True,
                        **diss)
    else:
        cfg = RunConfig(nx=n_across, ny=n_along, nz=nz, dx=dx, dy=dx,
                        ztop=ztop, dt=dt, run_seconds=0.0, open_y=True,
                        **diss)
    coord = make_vertical_coord(nz)
    base = make_base_state(coord, _stratified_sounding, p_surf=cfg.p_surf,
                           ztop=ztop)
    s = init_at_rest(cfg, coord, base)
    s.physics = _ZeroPBL(s)
    z = s.height_half()
    prof = 8.0 + 4.0 * z / ztop                     # sheared normal wind
    prof32 = cp.asarray(prof, dtype=cp.float32)[:, None, None]
    xf = np.arange(n_along + 1) * dx - 0.5 * n_along * dx   # face coords
    pulse = np.exp(-((xf + 5000.0) / 1200.0) ** 2).astype(np.float32)
    if axis == "x":
        s.u[...] = prof32 + cp.asarray(pulse)[None, None, :]
        normal, tangential = s.u, s.v
    else:
        s.v[...] = prof32 + cp.asarray(pulse)[None, :, None]
        normal, tangential = s.v, s.u

    # ---- phase 1: the pulse exits; the sheared flow returns to uniform ----
    run_steps(s, cfg, nsteps)
    n_err = float(cp.abs(normal - prof32).max())
    t_err = float(cp.abs(tangential).max())
    w_err = float(cp.abs(s.w).max())
    assert np.isfinite(n_err) and np.isfinite(t_err) and np.isfinite(w_err)
    assert n_err < 0.02, f"boundary-normal wind drifted {n_err}"
    assert t_err < 1e-3, f"tangential wind developed {t_err}"
    assert w_err < 0.01, f"w developed {w_err}"

    # ---- phase 2: exact WRF exclusion strips on the evolved state ---------
    # Fresh pulses by BOTH boundaries put real gradients across every strip
    # stencil (normal + tangential wind and theta'), then apply the
    # dissipation operators exactly as step() does.
    lo = np.exp(-((xf - (-0.5 * n_along * dx + 3000.0)) / 1500.0) ** 2)
    hi = np.exp(-((xf - (0.5 * n_along * dx - 3000.0)) / 1500.0) ** 2)
    bumps = 2.0 * (lo + hi).astype(np.float32)
    bc = 0.5 * (bumps[:-1] + bumps[1:])             # at mass centers
    if axis == "x":
        s.u += cp.asarray(bumps)[None, None, :]
        s.v += cp.asarray(bc)[None, None, :]
        s.thp += cp.asarray(bc)[None, None, :]
    else:
        s.v += cp.asarray(bumps)[None, :, None]
        s.u += cp.asarray(bc)[None, :, None]
        s.thp += cp.asarray(bc)[None, :, None]
    for name in _PROGNOSTICS:                       # as step()'s prologue
        getattr(s, name + "0")[...] = getattr(s, name)
    before = {n: cp.asnumpy(getattr(s, n))
              for n in ("u", "v", "w", "thp")}
    apply_diff6(s, cfg)
    d = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
         - before[n].astype(np.float64) for n in before}

    def cut(a, sl):
        """Slice along the open axis (x for open_x, y for open_y)."""
        return a[:, :, sl] if axis == "x" else a[:, sl, :]

    stag, m1, m2 = ("u", "v", "thp") if axis == "x" else ("v", "u", "thp")
    # staggered-normal axis: WRF computes ids+3..ide-3; the outermost
    # computed face (-4, WRF's ide-3) now takes the accurate boundary-datum
    # read (diff6.cu bndx/bndy), so exactly WRF's 3 excluded faces zero
    assert (cut(d[stag], slice(None, 3)) == 0.0).all()
    assert (cut(d[stag], slice(-3, None)) == 0.0).all(), \
        "staggered-normal high-side strip leaked (pre-d3db03b behavior)"
    assert np.abs(cut(d[stag], 3)).max() > 0.0      # first live faces
    assert np.abs(cut(d[stag], -4)).max() > 0.0     # WRF's ide-3 face
    assert np.abs(cut(d[stag], -5)).max() > 0.0
    for name in (m1, m2):                           # mass axes: 3 per side
        assert (cut(d[name], slice(None, 3)) == 0.0).all(), name
        assert (cut(d[name], slice(-3, None)) == 0.0).all(), name
        assert np.abs(cut(d[name], 3)).max() > 0.0, name
        assert np.abs(cut(d[name], -4)).max() > 0.0, name
    assert (cut(d["w"], slice(None, 3)) == 0.0).all()
    assert (cut(d["w"], slice(-3, None)) == 0.0).all()

    # Smagorinsky once-per-step tendencies honor WRF's width-1 exclusion
    # (the accurate boundary-datum read at the live staggered face is pinned
    # value-wise in tests/test_smag2d.py).
    for name in ("ru_t", "rv_t", "rw_t", "rth_t"):
        getattr(s, name)[...] = 0
    add_smag2d_tendencies(s, cfg, first=True)
    rs = cp.asnumpy(s.ru_t if axis == "x" else s.rv_t)
    rm = cp.asnumpy(s.rth_t)
    assert (cut(rs, 0) == 0.0).all() and (cut(rs, -1) == 0.0).all()
    assert np.abs(cut(rs, 1)).max() > 0.0
    assert np.abs(cut(rs, -2)).max() > 0.0          # live via accurate read
    assert (cut(rm, 0) == 0.0).all() and (cut(rm, -1) == 0.0).all()
    assert np.abs(cut(rm, 1)).max() > 0.0


@requires_gpu
@pytest.mark.parametrize("which", ["scalar", "u", "v", "w"])
def test_flux_div_specified_matches_mirror(which):
    # The real74 production combination in full: specified BCs (spec flag,
    # WRF's `specified` logical in module_advect_em.F) + non-uniform map
    # factors -- kernel vs float64 mirror.  Under specified the non-cb
    # open advective terms do NOT fire (WRF gates them on open_xs/xe/ys/ye
    # only) and u/v get the "specified uses upstream normal wind at
    # boundaries" substitution in the boundary-adjacent 2nd-order fluxes
    # (advect_u F:690-723, advect_v F:1978-2013).
    import cupy as cp
    from woof.core import advection as adv
    from woof.core.grid import make_vertical_coord
    from woof.verify import npref
    f = _advection_inputs()
    nz, ny, nx = f["q"].shape
    coord = make_vertical_coord(nz)
    launchers = {"scalar": adv.launch_flux_div_scalar,
                 "u": adv.launch_flux_div_u,
                 "v": adv.launch_flux_div_v,
                 "w": adv.launch_flux_div_w}
    mirrors = {"scalar": npref.np_flux_div_scalar,
               "u": npref.np_flux_div_u,
               "v": npref.np_flux_div_v,
               "w": npref.np_flux_div_w}
    shapes = {"scalar": (ny, nx), "u": (ny, nx + 1), "v": (ny + 1, nx),
              "w": (ny, nx)}
    rng = np.random.default_rng(41)
    msf = (1.0 + 0.05 * rng.random(shapes[which])).astype(np.float32)
    key = "q" if which == "scalar" else which
    args64 = [f[key].astype(np.float64), f["ru"].astype(np.float64),
              f["rv"].astype(np.float64), f["rw"].astype(np.float64)]
    ref = mirrors[which](*args64, coord, 100.0, 100.0,
                         open_x=True, open_y=True, spec=True,
                         msf=msf.astype(np.float64))
    tend = cp.zeros(ref.shape, cp.float32)
    launchers[which](cp.asarray(f[key]), cp.asarray(f["ru"]),
                     cp.asarray(f["rv"]), cp.asarray(f["rw"]), tend, coord,
                     100.0, 100.0, open_x=True, open_y=True, spec=True,
                     msf=cp.asarray(msf))
    np.testing.assert_allclose(cp.asnumpy(tend), ref, rtol=5e-4, atol=5e-4)


@requires_gpu
def test_specified_upstream_normal_wind_substitution_hand_pin():
    """WRF's 'specified uses upstream normal wind at boundaries'
    (module_advect_em.F advect_u :690-723): in the 2nd-order x flux one
    mass center in from the boundary, the outer stencil value ub is
    replaced by the inner one when the boundary-normal wind points OUT of
    the domain -- fqx(ids+1): ub = u(ids) unless u(ids+1) < 0, fqx(ide):
    ub = u(ide) unless u(ide-1) > 0.  Hand-computed on a flow-free
    background (rv = rw = 0, u constant in y/z), asserting both the
    substituted (outflow) and unsubstituted (inflow) cases; the open path
    without the spec flag must keep the plain 2nd-order flux."""
    import cupy as cp
    from woof.core.advection import launch_flux_div_u
    from woof.core.grid import make_vertical_coord
    nz, ny, nx = 2, 4, 9
    dx = 100.0
    vc = make_vertical_coord(nz)
    # u profile: outflow at the west boundary region (u < 0), inflow at
    # the east boundary region (u < 0 means eastern inflow -> no subst).
    uk = np.array([-2.0, -1.5, 3.0, 1.0, -1.0, 2.0, 0.5, -0.5, 1.0, -3.0])
    u = cp.asarray(np.broadcast_to(uk[None, None, :], (nz, ny, nx + 1)),
                   dtype=cp.float32)
    ru = cp.asarray(np.broadcast_to(4.0 * uk[None, None, :],
                                    (nz, ny, nx + 1)), dtype=cp.float32)
    rv = cp.zeros((nz, ny + 1, nx), cp.float32)
    rw = cp.zeros((nz + 1, ny, nx), cp.float32)

    def tendency(spec):
        tend = cp.zeros((nz, ny, nx + 1), cp.float32)
        launch_flux_div_u(u, ru, rv, rw, tend, vc, dx, dx,
                          open_x=True, open_y=False, spec=spec)
        return cp.asnumpy(tend)

    ruk = 4.0 * uk
    velx = 0.5 * (ruk[:-1] + ruk[1:])              # mass centers 0..nx-1

    def flux3h(qm2, qm1, q0, qp1, vel):
        return (vel * (7.0 * (q0 + qm1) - (qp1 + qm2))
                - abs(vel) * (3.0 * (q0 - qm1) - (qp1 - qm2))) / 12.0

    # WEST: m=0 flux; u[1] = -1.5 < 0 -> substitution active under spec
    f0_open = 0.5 * velx[0] * (uk[0] + uk[1])
    f0_spec = 0.5 * velx[0] * (uk[1] + uk[1])
    f1 = flux3h(uk[0], uk[1], uk[2], uk[3], velx[1])
    np.testing.assert_allclose(tendency(False)[:, :, 1],
                               -(f1 - f0_open) / dx, rtol=1e-5)
    np.testing.assert_allclose(tendency(True)[:, :, 1],
                               -(f1 - f0_spec) / dx, rtol=1e-5)
    # EAST: m=nx-1 flux; u[nx-1] = uk[8] = 1.0 > 0 -> substitution active
    fe_open = 0.5 * velx[nx - 1] * (uk[nx - 1] + uk[nx])
    fe_spec = 0.5 * velx[nx - 1] * (uk[nx - 1] + uk[nx - 1])
    fem1 = flux3h(uk[nx - 3], uk[nx - 2], uk[nx - 1], uk[nx],
                  velx[nx - 2])
    np.testing.assert_allclose(tendency(False)[:, :, nx - 1],
                               -(fe_open - fem1) / dx, rtol=1e-5)
    np.testing.assert_allclose(tendency(True)[:, :, nx - 1],
                               -(fe_spec - fem1) / dx, rtol=1e-5)


@requires_gpu
def test_specified_gates_non_cb_open_terms():
    """Under specified BCs WRF computes NO advective tendency at the
    outermost cells along the specified axis (advect_scalar bounds
    i_start=ids+1/i_end=ide-2, F:4037-4038; j analogues F:4056-4057) --
    the non-cb open radiative terms are gated on config_flags%open_xs/xe/
    ys/ye only (F:4115-4177).  Same constant-field construction as the
    FIX-A hand pin: with spec the x-boundary cells lose the -a/dx term and
    keep only the msf-weighted y part; corners lose both."""
    import cupy as cp
    from woof.core.advection import launch_flux_div_scalar
    from woof.core.grid import make_vertical_coord
    nz, ny, nx = 4, 8, 9
    dx = dy = 100.0
    a, b = 3.0, -2.0
    coord = make_vertical_coord(nz)
    q = cp.ones((nz, ny, nx), cp.float32)
    ru = cp.asarray(np.broadcast_to(a * np.arange(nx + 1, dtype=np.float32),
                                    (nz, ny, nx + 1)))
    rv = cp.asarray(np.broadcast_to(
        b * np.arange(ny + 1, dtype=np.float32)[:, None], (nz, ny + 1, nx)))
    rw = cp.zeros((nz + 1, ny, nx), cp.float32)
    rng = np.random.default_rng(5)
    msf = (0.96 + 0.05 * rng.random((ny, nx))).astype(np.float32)
    tend = cp.zeros((nz, ny, nx), cp.float32)
    launch_flux_div_scalar(q, ru, rv, rw, tend, coord, dx, dy,
                           open_x=True, open_y=True, spec=True,
                           msf=cp.asarray(msf))
    got = cp.asnumpy(tend)
    xbnd = np.zeros((ny, nx), bool)
    xbnd[:, 0] = xbnd[:, -1] = True
    ybnd = np.zeros((ny, nx), bool)
    ybnd[0, :] = ybnd[-1, :] = True
    wx = np.where(xbnd, 0.0, msf.astype(np.float64))   # specified: no term
    wy = np.where(ybnd, 0.0, msf.astype(np.float64))
    expected = -(wx * a / dx + wy * b / dy)
    np.testing.assert_allclose(got, np.broadcast_to(expected, (nz, ny, nx)),
                               rtol=1e-5, atol=1e-6)


@requires_gpu
def test_emdiv_step_runs_under_specified_boundaries():
    """Full step() with emdiv>0 on a SPECIFIED domain.

    The mudf strip zeroing hands a 2-D (ny, nx) buffer to
    _zero_open_strips, whose y-axis indexing was 3-D-only -- latent
    while emdiv defaulted to 0.0 (nothing in the suite configured it
    through step()), caught by the native-dt60 experiments.  One full
    step must complete finite; the emdiv increment must actually engage
    (mudf is nonzero after the first substep of stage 1).
    """
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.dycore import step
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest
    from woof.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)

    cfg = RunConfig(nx=12, ny=10, nz=6, dx=12000.0, dy=12000.0,
                    ztop=12000.0, dt=60.0, run_seconds=60.0, moist=True,
                    specified=True, emdiv=0.01)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    states = [init_at_rest(cfg, coord, base) for _ in range(3)]
    for s in states:
        s.qv[...] = cp.float32(0.01)
        s.qv0[...] = s.qv
        s.thp[...] += cp.float32(0.5)  # non-trivial mass tendency
    boundaries = build_state_lateral_boundaries(
        states, [0.0, 21600.0, 43200.0])
    attach_lateral_boundaries(states[0], boundaries)
    step(states[0], cfg)
    for name in ("u", "v", "w", "thp", "php", "mup", "qv"):
        assert bool(cp.isfinite(getattr(states[0], name)).all()), name
