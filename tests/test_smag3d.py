# tests/test_smag3d.py
"""3-D Smagorinsky closure, WRF km_opt=3 (P6 LES engine track).

R0 battery per the LES program spec: float64-mirror comparisons for every
new kernel (calculate_N2, smag_km's four exchange coefficients,
vertical_diffusion_s, the isfflx=0/2 prescribed-flux surface branches),
regime coverage for every clamp/limiter (mix_upper_bound cap and the 1e-6
floor engaged-somewhere/not-everywhere, N^2 reduction engaged in stable and
not in neutral stratification, saturated N^2 branch engaged in cloud),
analytic self-checks (uniform-shear K, Kh = 3 Km, column conservation of
the vertical scalar operator), mutation sensitivity (the mirror must fail
when D13/D23 are dropped; zero xkhv must zero the vertical scalar mixing),
and the dycore wiring proof that km_opt=3 closes the engine's
zero-vertical-scalar-mixing gap that km_opt=4 has by definition.

Formula authority: handoffs/P6-LES-WPL0-AUTHORITY-RECEIPT.md (WRF v4.6.1
module_diffusion_em.F read on the pinned node tree; bundle == node on every
LES-relevant file).
"""
import numpy as np
import pytest
from conftest import requires_gpu

from woof.config import RunConfig
from woof.core.grid import make_base_state, make_vertical_coord

CS = 0.18          # em_les reference value (README.les)
PR = 1.0 / 3.0


def _cfg(nx=12, ny=6, nz=8, **kw):
    kw.setdefault("km_opt", 3)
    kw.setdefault("bl_pbl_physics", 0)
    return RunConfig(nx=nx, ny=ny, nz=nz, dx=200.0, dy=200.0, ztop=2000.0,
                    dt=2.0, run_seconds=0.0, c_s=CS, **kw)


def _dry_state(cfg, theta_func, thp_func=None, seed=None, wind_amp=0.0):
    import cupy as cp
    from woof.core.diagnostics import update_diagnostics
    from woof.core.state import init_at_rest, init_theta_perturbation
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, theta_func, p_surf=cfg.p_surf, ztop=cfg.ztop)
    if thp_func is None:
        s = init_at_rest(cfg, vc, b)
    else:
        s = init_theta_perturbation(cfg, vc, b, thp_func)
    if seed is not None:
        rng = np.random.default_rng(seed)
        u = wind_amp * rng.standard_normal((cfg.nz, cfg.ny, cfg.nx + 1))
        v = wind_amp * rng.standard_normal((cfg.nz, cfg.ny + 1, cfg.nx))
        w = wind_amp * rng.standard_normal((cfg.nz + 1, cfg.ny, cfg.nx))
        u[:, :, cfg.nx] = u[:, :, 0]
        v[:, cfg.ny, :] = v[:, 0, :]
        w[0] = 0.0
        w[cfg.nz] = 0.0
        s.u[...] = cp.asarray(u, dtype=cp.float32)
        s.v[...] = cp.asarray(v, dtype=cp.float32)
        s.w[...] = cp.asarray(w, dtype=cp.float32)
    update_diagnostics(s, cfg.hypsometric_opt)
    return s


def _host(state):
    """Float64 mirror inputs pulled from one device state (flat phb)."""
    import cupy as cp
    phb = cp.asnumpy(state.phb).astype(np.float64)
    php = cp.asnumpy(state.php).astype(np.float64)
    phi = phb[:, None, None] + php if phb.ndim == 1 else phb + php
    return {
        "u": cp.asnumpy(state.u), "v": cp.asnumpy(state.v),
        "w": cp.asnumpy(state.w), "phi": phi,
        "thp": cp.asnumpy(state.thp), "thb": cp.asnumpy(state.thb),
        "p": cp.asnumpy(state.p), "alt": cp.asnumpy(state.alt),
        "dnw": cp.asnumpy(state.dnw).astype(np.float64),
        "dn": cp.asnumpy(state.dn).astype(np.float64),
        "fnm": cp.asnumpy(state.fnm).astype(np.float64),
        "fnp": cp.asnumpy(state.fnp).astype(np.float64),
        "cf1": float(state.cf1), "cf2": float(state.cf2),
        "cf3": float(state.cf3),
    }


def _launch_k(state, cfg):
    """Run the production km_opt=3 K launcher; return the four K fields."""
    import cupy as cp
    from woof.core.dycore import launch_wrf_smag3d_km
    nz, ny, nx = state.p.shape
    km = state.scratch((nz, ny, nx), "smag_km")
    kh = state.scratch((nz, ny, nx), "smag_kh")
    launch_wrf_smag3d_km(state, cfg, km, kh, time_t=False)
    kmv = state.scratch((nz, ny, nx), "smag_kmv")
    khv = state.scratch((nz, ny, nx), "smag_khv")
    return (cp.asnumpy(km), cp.asnumpy(kh),
            cp.asnumpy(kmv), cp.asnumpy(khv))


# ---------------------------------------------------------------------------
# calculate_N2
# ---------------------------------------------------------------------------

@requires_gpu
def test_wrf_calc_n2_matches_mirror_dry():
    import cupy as cp
    from woof.core.dycore import launch_wrf_calc_n2
    from woof.verify.npref import np_wrf_calc_n2
    cfg = _cfg()

    def thp_func(x, z):
        rng = np.random.default_rng(3)
        return 0.5 * rng.standard_normal((cfg.nz, cfg.ny, cfg.nx))

    s = _dry_state(cfg, lambda z: 300.0 + 0.004 * np.asarray(z, float),
                   thp_func=thp_func)
    bn2 = cp.zeros(s.p.shape, dtype=cp.float32)
    launch_wrf_calc_n2(s, cfg, bn2, time_t=False)
    h = _host(s)
    ref = np_wrf_calc_n2(h["thp"], h["thb"], h["p"], h["phi"],
                         cf1=h["cf1"], cf2=h["cf2"], cf3=h["cf3"])
    np.testing.assert_allclose(cp.asnumpy(bn2), ref, rtol=2e-3, atol=2e-7)
    # A stably stratified base state must give positive interior N^2.
    assert ref[1:-1].mean() > 0.0
    # WRF's top copy: BN2(ktf) == BN2(ktf-1).
    np.testing.assert_array_equal(cp.asnumpy(bn2)[-1], cp.asnumpy(bn2)[-2])


@requires_gpu
def test_wrf_calc_n2_matches_mirror_moist_saturated_branch():
    import cupy as cp
    from woof.core.diagnostics import update_diagnostics
    from woof.core.dycore import launch_wrf_calc_n2
    from woof.core.moist import init_moist_balanced
    from woof.verify.npref import np_wrf_calc_n2
    cfg = _cfg(moist=True, mp_physics=1)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, lambda z: 300.0 + 0.004 * np.asarray(z, float),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_moist_balanced(cfg, vc, b,
                            lambda z: 0.012 * np.exp(
                                -np.asarray(z, float) / 2500.0))
    # Engage the saturated branch in a patch: cloud water above qc_cr.
    qc = np.zeros(s.p.shape, dtype=np.float32)
    qc[2:5, 1:3, 2:6] = 5.0e-4
    s.qc[...] = cp.asarray(qc)
    update_diagnostics(s, cfg.hypsometric_opt)
    bn2 = cp.zeros(s.p.shape, dtype=cp.float32)
    launch_wrf_calc_n2(s, cfg, bn2, time_t=False)
    h = _host(s)
    qv = cp.asnumpy(s.qv)
    ref = np_wrf_calc_n2(h["thp"], h["thb"], h["p"], h["phi"],
                         cf1=h["cf1"], cf2=h["cf2"], cf3=h["cf3"],
                         qv=qv, qc=qc)
    np.testing.assert_allclose(cp.asnumpy(bn2), ref, rtol=5e-3, atol=5e-7)
    # Regime coverage: the saturated branch fired somewhere, not everywhere.
    assert (qc >= 1.0e-5).any() and not (qc >= 1.0e-5).all()


# ---------------------------------------------------------------------------
# smag_km: the four exchange coefficients
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("isotropic", [0, 1])
@pytest.mark.parametrize("amp", [0.5, 80.0])
def test_wrf_smag3d_km_matches_mirror(isotropic, amp):
    from woof.verify.npref import np_wrf_calc_n2, np_wrf_smag3d_km
    cfg = _cfg(mix_isotropic=isotropic)
    s = _dry_state(cfg, lambda z: 300.0 + 0.004 * np.asarray(z, float),
                   seed=1, wind_amp=amp)
    dev = _launch_k(s, cfg)
    h = _host(s)
    bn2 = np_wrf_calc_n2(h["thp"], h["thb"], h["p"], h["phi"],
                         cf1=h["cf1"], cf2=h["cf2"], cf3=h["cf3"])
    ref = np_wrf_smag3d_km(
        h["u"], h["v"], h["w"], h["phi"], bn2,
        dx=cfg.dx, dy=cfg.dy, dt=cfg.dt, c_s=cfg.c_s,
        mix_upper_bound=cfg.mix_upper_bound, mix_isotropic=isotropic,
        fnm=h["fnm"], fnp=h["fnp"], dn=h["dn"], dnw=h["dnw"],
        cf1=h["cf1"], cf2=h["cf2"], cf3=h["cf3"])
    for name, d, r in zip(("xkmh", "xkhh", "xkmv", "xkhv"), dev, ref):
        scale = max(float(np.abs(r).max()), 1.0e-30)
        np.testing.assert_allclose(
            d, r, rtol=2e-3, atol=5e-4 * scale, err_msg=name)
    if amp > 10.0:
        # mix_upper_bound cap engaged somewhere, not everywhere.  The
        # binding cap differs per branch: anisotropic xkmv caps at
        # mix_upper_bound*dz^2/dt (dz < sqrt(dx*dy) here); isotropic
        # xkmh caps at mix_upper_bound*dx*dy/dt.
        if isotropic == 0:
            xkmv_ref = ref[2]
            caps = np.zeros_like(xkmv_ref)
            G = 9.81
            for k in range(cfg.nz):
                dz = (h["phi"][k + 1] - h["phi"][k]) / G
                caps[k] = cfg.mix_upper_bound * dz * dz / cfg.dt
            engaged = np.isclose(xkmv_ref, caps, rtol=1e-12)
        else:
            cap = cfg.mix_upper_bound * cfg.dx * cfg.dy / cfg.dt
            engaged = np.isclose(ref[0], cap, rtol=1e-12)
        assert engaged.any() and not engaged.all()


@requires_gpu
def test_smag3d_mirror_without_vertical_terms_must_fail():
    """Mutation control: dropping D13/D23/D33 from the invariant is
    detectable -- the crippled reduction disagrees with the device on a
    vertically sheared flow (the km_opt=4 flat reduction cannot stand in
    for the 3-D closure)."""
    from woof.verify.npref import (np_smag2d_km, np_wrf_calc_n2,
                                    np_wrf_smag3d_km)
    cfg = _cfg(mix_isotropic=0)
    s = _dry_state(cfg, lambda z: 300.0 + 0.004 * np.asarray(z, float),
                   seed=2, wind_amp=2.0)
    xkmh_dev = _launch_k(s, cfg)[0]
    h = _host(s)
    bn2 = np_wrf_calc_n2(h["thp"], h["thb"], h["p"], h["phi"],
                         cf1=h["cf1"], cf2=h["cf2"], cf3=h["cf3"])
    full = np_wrf_smag3d_km(
        h["u"], h["v"], h["w"], h["phi"], bn2,
        dx=cfg.dx, dy=cfg.dy, dt=cfg.dt, c_s=cfg.c_s,
        mix_upper_bound=cfg.mix_upper_bound, mix_isotropic=0,
        fnm=h["fnm"], fnp=h["fnp"], dn=h["dn"], dnw=h["dnw"],
        cf1=h["cf1"], cf2=h["cf2"], cf3=h["cf3"])[0]
    # The full mirror agrees; the vertical-blind km_opt=4 K does not.
    scale = float(np.abs(full).max())
    np.testing.assert_allclose(xkmh_dev, full, rtol=2e-3, atol=5e-4 * scale)
    crippled = np_smag2d_km(h["u"], h["v"], cfg.dx, cfg.dy, cfg.c_s)[0]
    assert float(np.abs(crippled - full).max()) > 1e-3 * scale


@requires_gpu
def test_smag3d_neutral_uniform_shear_analytic():
    """u = S*z, neutral theta: def2 = S^2 at interior mass levels, so
    xkmv = c_s^2 dz^2 S (anisotropic) and Kh = 3 Km exactly (before
    caps -- WRF's constant-Prandtl km_opt=3 treatment, unlike km_opt=2's
    stability-dependent vertical Pr)."""
    import cupy as cp
    S = 0.01
    cfg = _cfg(mix_isotropic=0)
    s = _dry_state(cfg, lambda z: 300.0 + 0.0 * np.asarray(z, float))
    z = s.height_half()
    zz = np.asarray(cp.asnumpy(z) if hasattr(z, "get") else z, float)
    if zz.ndim == 1:
        zz = np.broadcast_to(zz[:, None, None],
                             (cfg.nz, cfg.ny, cfg.nx)).copy()
    prof = S * zz
    u = np.concatenate([prof, prof[:, :, :1]], axis=2)
    s.u[...] = cp.asarray(u, dtype=cp.float32)
    from woof.core.diagnostics import update_diagnostics
    update_diagnostics(s, cfg.hypsometric_opt)
    xkmh, xkhh, xkmv, xkhv = _launch_k(s, cfg)
    h = _host(s)
    G = 9.81
    for k in range(1, cfg.nz - 1):        # interior: all 4 D13 corners live
        dz = float((h["phi"][k + 1] - h["phi"][k]).mean()) / G
        expect = CS * CS * dz * dz * S
        np.testing.assert_allclose(xkmv[k], expect, rtol=5e-3,
                                   err_msg=f"level {k}")
    np.testing.assert_allclose(xkhv, 3.0 * xkmv, rtol=1e-5)
    np.testing.assert_allclose(xkhh, 3.0 * xkmh, rtol=1e-5)


@requires_gpu
def test_smag3d_buoyancy_limiter_regimes():
    """Stable stratification at rest floors K exactly (N^2/Pr > D^2 = 0);
    an unstable slab (BN2 < 0) lifts K above the floor with no shear at
    all -- the buoyancy term itself produces mixing."""
    cfg = _cfg(mix_isotropic=0)
    stable = _dry_state(cfg, lambda z: 300.0 + 0.005 * np.asarray(z, float))
    xkmh, _, xkmv, xkhv = _launch_k(stable, cfg)
    floor_h = 1.0e-6 * cfg.dx * cfg.dy
    np.testing.assert_allclose(xkmh, floor_h, rtol=1e-6)

    def unstable_slab(x, z):
        # Overturn the lowest ~quarter of the column.
        zz = z[:, None, None] if np.ndim(z) == 1 else z
        pert = np.where(zz < 500.0, 3.0 * (1.0 - zz / 500.0), 0.0)
        return pert * np.ones((cfg.nz, cfg.ny, cfg.nx))

    unstable = _dry_state(
        cfg, lambda z: 300.0 + 0.005 * np.asarray(z, float),
        thp_func=unstable_slab)
    xkmh_u = _launch_k(unstable, cfg)[0]
    assert float(xkmh_u.max()) > 2.0 * floor_h
    assert (np.isclose(xkmh_u, floor_h, rtol=1e-6)).any()  # not everywhere


# ---------------------------------------------------------------------------
# vertical_diffusion_s + the prescribed-flux surface branches
# ---------------------------------------------------------------------------

def _vertical_buffers(state):
    import cupy as cp
    return {name: cp.zeros(getattr(state, field).shape, dtype=cp.float32)
            for name, field in (("ru", "u"), ("rv", "v"), ("rw", "w"),
                                ("rth", "thp"))}


@requires_gpu
def test_wrf_vd_s_matches_mirror_and_conserves():
    import cupy as cp
    from woof.core.dycore import launch_wrf_smag2d_vertical
    from woof.verify.npref import np_wrf_vertical_diffusion_s
    cfg = _cfg(isfflx=1)                   # no surface fields -> inert
    s = _dry_state(cfg, lambda z: 300.0 + 0.004 * np.asarray(z, float),
                   thp_func=lambda x, z: 1.5 * np.cos(
                       np.linspace(0, 3 * np.pi, cfg.nz))[:, None, None]
                   * np.ones((cfg.nz, cfg.ny, cfg.nx)),
                   seed=4, wind_amp=0.0)
    nz, ny, nx = s.p.shape
    rng = np.random.default_rng(7)
    khv_h = np.abs(rng.standard_normal((nz, ny, nx))).astype(np.float32)
    khv = cp.asarray(khv_h)
    km0 = cp.zeros((nz, ny, nx), dtype=cp.float32)
    bufs = _vertical_buffers(s)
    launch_wrf_smag2d_vertical(
        s, cfg, km0, ru=bufs["ru"], rv=bufs["rv"], rw=bufs["rw"],
        rth=bufs["rth"], rqv=None, time_t=False,
        kmv=km0, khv=khv, scalar_rows=[(s.thp, bufs["rth"], True)])
    h = _host(s)
    rho = 1.0 / h["alt"].astype(np.float64)
    ref = np_wrf_vertical_diffusion_s(
        h["thp"], khv_h.astype(np.float64), rho, h["phi"],
        fnm=h["fnm"], fnp=h["fnp"], dnw=h["dnw"], thb=h["thb"])
    got = cp.asnumpy(bufs["rth"])
    scale = max(float(np.abs(ref).max()), 1e-30)
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=5e-4 * scale)
    # Column conservation: sum_k tend*dnw telescopes to H3(top)-H3(sfc)=0.
    col = (ref * h["dnw"][:, None, None]).sum(axis=0)
    assert float(np.abs(col).max()) <= 1e-10 * scale
    col_dev = (got.astype(np.float64)
               * h["dnw"][:, None, None]).sum(axis=0)
    assert float(np.abs(col_dev).max()) <= 1e-5 * scale
    # Momentum rows saw K = 0 everywhere: untouched.
    assert not cp.asnumpy(bufs["ru"]).any()
    assert not cp.asnumpy(bufs["rw"]).any()

    # Mutation control (AC-L4.3(c) operator arm): zero xkhv -> zero mixing.
    bufs2 = _vertical_buffers(s)
    launch_wrf_smag2d_vertical(
        s, cfg, km0, ru=bufs2["ru"], rv=bufs2["rv"], rw=bufs2["rw"],
        rth=bufs2["rth"], rqv=None, time_t=False,
        kmv=km0, khv=km0, scalar_rows=[(s.thp, bufs2["rth"], True)])
    assert not cp.asnumpy(bufs2["rth"]).any()


@requires_gpu
def test_prescribed_flux_surface_isfflx0_matches_mirror():
    import cupy as cp
    from woof.core.dycore import launch_wrf_smag2d_vertical
    from woof.verify.npref import (np_wrf_surface_heat_const,
                                    np_wrf_surface_mom_cd0)
    cfg = _cfg(isfflx=0, tke_heat_flux=0.24, tke_drag_coefficient=0.0013)
    s = _dry_state(cfg, lambda z: 300.0 + 0.004 * np.asarray(z, float))
    s.u[...] = cp.float32(4.0)
    s.v[...] = cp.float32(-3.0)
    from woof.core.diagnostics import update_diagnostics
    update_diagnostics(s, cfg.hypsometric_opt)
    nz, ny, nx = s.p.shape
    km0 = cp.zeros((nz, ny, nx), dtype=cp.float32)
    bufs = _vertical_buffers(s)
    launch_wrf_smag2d_vertical(
        s, cfg, km0, ru=bufs["ru"], rv=bufs["rv"], rw=bufs["rw"],
        rth=bufs["rth"], rqv=None, time_t=False)
    h = _host(s)
    rho0 = 1.0 / h["alt"][0].astype(np.float64)
    dnw0 = float(h["dnw"][0])
    ref_th = np_wrf_surface_heat_const(0.24, rho0, dnw0)
    got_th = cp.asnumpy(bufs["rth"])
    np.testing.assert_allclose(got_th[0], ref_th, rtol=2e-6)
    assert (got_th[0] > 0.0).all()          # positive flux warms level 0
    assert not got_th[1:].any()
    ru_ref, rv_ref = np_wrf_surface_mom_cd0(
        h["u"][0], h["v"][0], rho0, 0.0013, dnw0)
    np.testing.assert_allclose(cp.asnumpy(bufs["ru"])[0], ru_ref,
                               rtol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(bufs["rv"])[0], rv_ref,
                               rtol=2e-5)
    # Drag opposes the wind (dnw < 0 makes the increment sign-opposite u).
    assert (cp.asnumpy(bufs["ru"])[0] < 0.0).all()
    assert (cp.asnumpy(bufs["rv"])[0] > 0.0).all()


@requires_gpu
def test_prescribed_flux_surface_isfflx2_hybrid():
    """isfflx=2: momentum from USTM, heat CONSTANT, moisture from QFX --
    the em_les hybrid, distinct from both 0 and 1."""
    from types import SimpleNamespace

    import cupy as cp

    from woof.core import constants as c
    from woof.core.diagnostics import update_diagnostics
    from woof.core.dycore import launch_wrf_smag2d_vertical
    from woof.core.moist import init_moist_balanced
    from woof.verify.npref import np_wrf_surface_heat_const
    cfg = _cfg(isfflx=2, tke_heat_flux=0.24, moist=True, mp_physics=1,
               sf_sfclay_physics=1)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, lambda z: 300.0 + 0.004 * np.asarray(z, float),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_moist_balanced(cfg, vc, b, lambda z: 0.008 + 0.0
                            * np.asarray(z, float))
    s.u[...] = cp.float32(4.0)
    s.v[...] = cp.float32(3.0)
    update_diagnostics(s, cfg.hypsometric_opt)
    ny, nx = s.p.shape[1:]
    s.physics = SimpleNamespace(fields={
        "ustm": cp.full((ny, nx), 0.5, dtype=cp.float32),
        "hfx": cp.full((ny, nx), 500.0, dtype=cp.float32),
        "qfx": cp.full((ny, nx), 1.0e-4, dtype=cp.float32),
    })
    nz = s.p.shape[0]
    km0 = cp.zeros((nz, ny, nx), dtype=cp.float32)
    bufs = _vertical_buffers(s)
    rqv = cp.zeros_like(s.qv)
    launch_wrf_smag2d_vertical(
        s, cfg, km0, ru=bufs["ru"], rv=bufs["rv"], rw=bufs["rw"],
        rth=bufs["rth"], rqv=rqv, time_t=False)
    h = _host(s)
    qv0 = cp.asnumpy(s.qv)[0].astype(np.float64)
    rho0 = (1.0 + qv0) / h["alt"][0].astype(np.float64)
    dnw0 = float(h["dnw"][0])
    # Heat: the CONSTANT flux, NOT hfx/cpm (hfx=500 would give a much
    # larger increment) -- the distinguishing isfflx=2 assertion.
    ref_th = np_wrf_surface_heat_const(0.24, rho0, dnw0)
    np.testing.assert_allclose(cp.asnumpy(bufs["rth"])[0], ref_th,
                               rtol=2e-6)
    wrong_th = -c.G * 500.0 / (c.CP * (1.0 + 0.8 * qv0)) / dnw0
    assert float(np.abs(cp.asnumpy(bufs["rth"])[0] - wrong_th).min()) \
        > 0.1 * float(np.abs(wrong_th).max())
    # Moisture: applied from qfx (isfflx=2 keeps the surface routine's
    # moisture flux).
    ref_qv = -c.G * 1.0e-4 / dnw0
    np.testing.assert_allclose(cp.asnumpy(rqv)[0], ref_qv, rtol=2e-6)
    # Momentum: ustm-based (nonzero with ustm > 0).
    assert cp.asnumpy(bufs["ru"])[0].any()


# ---------------------------------------------------------------------------
# Dycore wiring: the gap-closing assertions
# ---------------------------------------------------------------------------

@requires_gpu
def test_km_opt3_interior_vertical_scalar_mixing_engages():
    """THE engine gap this closure exists to fix: under km_opt=4 the
    interior theta forward tendency has no vertical component (xkhv = 0
    by definition), so a horizontally uniform stratified column gets a
    ZERO smag_rth buffer.  km_opt=3 must mix it."""
    import cupy as cp
    from woof.core.dycore import prepare_fixed_tendencies

    def thp_func(x, z):
        prof = 1.5 * np.cos(np.linspace(0, 3 * np.pi, 8))
        return prof[:, None, None] * np.ones((8, 6, 12))

    results = {}
    for km_opt in (3, 4):
        cfg = _cfg(km_opt=km_opt, isfflx=1)
        s = _dry_state(cfg, lambda z: 300.0 + 0.004 * np.asarray(z, float),
                       thp_func=thp_func)
        for name in ("u", "v", "w", "thp", "php", "mup"):
            getattr(s, name + "0")[...] = getattr(s, name)
        prepare_fixed_tendencies(s, cfg)
        results[km_opt] = cp.asnumpy(
            s.scratch(s.thp.shape, "smag_rth")).copy()
    # Horizontally uniform field: horizontal fluxes vanish identically,
    # so anything in the buffer is vertical mixing.
    assert not results[4][1:-1].any()
    assert float(np.abs(results[3][1:-1]).max()) > 0.0


@requires_gpu
def test_step_km_opt3_prescribed_flux_cbl_smoke():
    """Five full steps of a small prescribed-flux CBL: finite fields, the
    surface heat flux warms the lowest level, and turning the flux off
    removes the warming (disconnected-driver mutation control)."""
    import cupy as cp
    from woof.core.dycore import run_steps, stability_report
    from woof.core.state import init_theta_perturbation

    def build(heat_flux):
        cfg = RunConfig(nx=16, ny=16, nz=16, dx=100.0, dy=100.0,
                        ztop=1600.0, dt=0.5, run_seconds=0.0,
                        km_opt=3, c_s=CS, mix_isotropic=1,
                        bl_pbl_physics=0, isfflx=0,
                        tke_heat_flux=heat_flux,
                        tke_drag_coefficient=0.0013,
                        time_step_sound=4)
        vc = make_vertical_coord(cfg.nz)
        b = make_base_state(vc, lambda z: 300.0 + 0.003
                            * np.asarray(z, float),
                            p_surf=cfg.p_surf, ztop=cfg.ztop)

        def thp_func(x, z):
            rng = np.random.default_rng(11)
            pert = np.zeros((cfg.nz, cfg.ny, cfg.nx))
            pert[:4] = 0.1 * rng.standard_normal((4, cfg.ny, cfg.nx))
            return pert

        s = init_theta_perturbation(cfg, vc, b, thp_func)
        return s, cfg

    s_on, cfg_on = build(0.24)
    s_off, cfg_off = build(0.0)
    th0_on = float(s_on.thp[0].mean())
    th0_off = float(s_off.thp[0].mean())
    run_steps(s_on, cfg_on, 5)
    run_steps(s_off, cfg_off, 5)
    rep = stability_report(s_on, cfg_on)
    assert not rep["nan"]
    assert rep["cfl"] is not None and rep["cfl"] < 1.0
    warm_on = float(s_on.thp[0].mean()) - th0_on
    warm_off = float(s_off.thp[0].mean()) - th0_off
    # 5 steps x 0.5 s of Q_s=0.24 K m/s over dz~100 m: O(6e-3) K.
    assert warm_on > 1e-3
    assert warm_on > 5.0 * abs(warm_off)
    assert np.isfinite(cp.asnumpy(s_on.thp)).all()
    assert np.isfinite(cp.asnumpy(s_on.u)).all()
    assert np.isfinite(cp.asnumpy(s_on.w)).all()


# ---------------------------------------------------------------------------
# Config admission (CPU)
# ---------------------------------------------------------------------------

def test_km_opt3_config_admission():
    from woof.config import validate_run_config
    base = dict(nx=16, ny=16, nz=16, dx=100.0, dy=100.0, ztop=1600.0,
                dt=0.5, run_seconds=0.0)
    ok = validate_run_config(RunConfig(
        **base, km_opt=3, bl_pbl_physics=0, isfflx=0,
        tke_heat_flux=0.24, tke_drag_coefficient=0.0013))
    assert ok.km_opt == 3
    validate_run_config(RunConfig(**base, km_opt=3, isfflx=2,
                                  bl_pbl_physics=0))
    # Under diff_opt = 2 WRF ignores khdif/kvdif outside km_opt = 1, and so
    # does woof/core/diffusion.py, so km_opt = 3 beside a dormant constant
    # imports as written (890e2523a, WRF fire namelists).  diff_opt = 1
    # still refuses it (f586d1128): WRF's diff_opt = 1 vertical diffusion
    # reads kvdif whatever km_opt is, and this engine would drop that term.
    dormant = validate_run_config(RunConfig(**base, km_opt=3, khdif=75.0,
                                            kvdif=25.0))
    assert (dormant.diff_opt, dormant.km_opt, dormant.khdif,
            dormant.kvdif) == (2, 3, 75.0, 25.0)
    for option in (2, 4):
        for name in ("khdif", "kvdif"):
            with pytest.raises(ValueError, match="khdif"):
                validate_run_config(RunConfig(**base, diff_opt=1,
                                              km_opt=option,
                                              **{name: 75.0}))
    # km_opt=1 under diff_opt=2 with the PBL off runs WRF's
    # vertical_diffusion_2 surface arms (isotropic_km into the same
    # package), so it consumes the prescribed forcing; diff_opt=1 does not.
    validate_run_config(RunConfig(**base, km_opt=1, isfflx=2,
                                  tke_heat_flux=0.24))
    validate_run_config(RunConfig(**base, km_opt=1, isfflx=0,
                                  sf_sfclay_physics=0,
                                  tke_drag_coefficient=0.0013))
    with pytest.raises(ValueError, match="isfflx=2"):
        validate_run_config(RunConfig(**base, km_opt=1, diff_opt=1,
                                      isfflx=2))
    with pytest.raises(ValueError, match="isfflx=0"):
        validate_run_config(RunConfig(**base, km_opt=1, diff_opt=1,
                                      isfflx=0, sf_sfclay_physics=0))
    with pytest.raises(ValueError, match="mix_isotropic"):
        validate_run_config(RunConfig(**base, km_opt=3, mix_isotropic=2))


def test_per_domain_turbulence_override_surface():
    """The per-domain schema admits the turbulence row (parent-PBL +
    child-LES topology; selection is a configuration capability,
    implemented-unverified for nested LES children)."""
    from woof.experiment import _DOMAIN_RUN_OVERRIDES
    for key in ("km_opt", "bl_pbl_physics", "sf_sfclay_physics", "c_s",
                "diff_6th_opt", "mix_isotropic", "mix_upper_bound",
                "isfflx", "tke_heat_flux", "tke_drag_coefficient"):
        assert key in _DOMAIN_RUN_OVERRIDES, key


# ---------------------------------------------------------------------------
# The w rows' exchange coefficient (nested-LES defect, 2026-08-01)
# ---------------------------------------------------------------------------

def _anisotropic_cfg(**kw):
    """A grid whose first layer is far thinner than its spacing -- the LES
    regime where xkmh and xkmv are not interchangeable.  dz ~ 25 m against
    dx = 250 m, so (dx/dz)^2 ~ 100."""
    kw.setdefault("km_opt", 3)
    kw.setdefault("bl_pbl_physics", 0)
    kw.setdefault("mix_isotropic", 0)
    return RunConfig(nx=16, ny=16, nz=48, dx=250.0, dy=250.0, ztop=1200.0,
                     dt=1.25, run_seconds=0.0, c_s=CS, **kw)


@requires_gpu
def test_the_w_rows_take_the_vertical_momentum_coefficient():
    """WRF pairs every stress with the coefficient of ITS directions.

    ``horizontal_diffusion_w_2`` is the divergence of tau13/tau23 and WRF
    hands it ``xkmv`` (module_diffusion_em.F:2998-3006; the dummy argument
    is spelled ``xkmv`` at :3524), the same coefficient
    ``vertical_diffusion_u_2``/``_v_2`` give those two stresses
    (:4128-4147).  ``vertical_diffusion_w_2`` is the divergence of tau33 =
    2 K dw/dz, a vertical-vertical stress, and WRF hands it ``xkmh``
    (:4145-4155) -- the one place the pairing breaks.  WOOF-ArWen gives BOTH w
    rows ``xkmv``: the horizontal one to match WRF, the vertical one as a
    documented divergence (see the next test for why).

    ``smag2d_km`` sets ``xkmv = xkmh`` for km_opt=4 (:2035), so this is an
    identity there and no km_opt=4 trajectory can move.
    """
    from woof.core.dycore import _horizontal_w_km

    cfg3 = _anisotropic_cfg()
    s3 = _dry_state(cfg3, lambda z: 300.0 + 0.003 * np.asarray(z, float))
    assert _horizontal_w_km(s3, cfg3) is s3.scratch(s3.p.shape, "smag_kmv")

    cfg4 = _anisotropic_cfg(km_opt=4)
    s4 = _dry_state(cfg4, lambda z: 300.0 + 0.003 * np.asarray(z, float))
    # km_opt=4 must keep taking the caller's xkmh, which IS WRF's xkmv.
    assert _horizontal_w_km(s4, cfg4) is None


@requires_gpu
def test_mix_upper_bound_makes_the_vertical_w_operator_stable_only_with_kmv():
    """Why the vertical w row diverges from WRF, in one inequality.

    ``vertical_diffusion_w_2`` is explicit, so it needs
    2*K*dt/dz^2 <= 1/2.  ``smag_km`` caps the two coefficients on
    DIFFERENT length scales (module_diffusion_em.F:1890-1908):

        xkmv <= mix_upper_bound * mlen_v^2 / dt,  mlen_v = dz
        xkmh <= mix_upper_bound * mlen_h^2 / dt,  mlen_h^2 = dx*dy

    so 2*xkmv*dt/dz^2 <= 2*mix_upper_bound = 0.2 at the Registry default
    -- WRF's own cap IS the stability guarantee for this operator, and it
    only guarantees it for xkmv.  With xkmh the same quantity is bounded
    only by 2*mix_upper_bound*(dx/dz)^2, which on any anisotropic LES grid
    is far above 1/2.  Matching WRF here would mean transcribing an
    operator that cannot run.
    """
    import cupy as cp
    cfg = _anisotropic_cfg()
    s = _dry_state(cfg, lambda z: 300.0 + 0.0005 * np.asarray(z, float),
                   seed=5, wind_amp=2.0)
    km, _kh, kmv, _khv = _launch_k(s, cfg)
    nz = cfg.nz
    phb = cp.asnumpy(s.phb)
    php = cp.asnumpy(s.php)
    phi = phb[:, None, None] + php if phb.ndim == 1 else phb + php
    dz = (phi[1:nz + 1] - phi[0:nz]) / 9.81

    kmv_number = 2.0 * kmv * cfg.dt / (dz * dz)
    km_number = 2.0 * km * cfg.dt / (dz * dz)
    # The cap's guarantee, held everywhere with no clamp of ArWen's own.
    assert kmv_number.max() <= 2.0 * cfg.mix_upper_bound + 1e-6
    assert kmv_number.max() <= 0.5
    # Mutation control: the coefficient WRF actually passes breaks it, so
    # this test fails the moment the vertical w row goes back to xkmh.
    assert km_number.max() > 0.5
    # And the two are genuinely different fields, not a relabelling.
    assert float(np.abs(km - kmv).max()) > 0.0


@requires_gpu
def test_vertical_diffusion_w_is_launched_with_kmv(monkeypatch):
    """Pin the actual launch, not just the helper: record the K pointer
    every vertical kernel receives.  This is the assertion that fails the
    moment ``wrf_smag_vd_w`` goes back to WRF's ``xkmh``."""
    from woof.core import dycore

    cfg = _anisotropic_cfg()
    s = _dry_state(cfg, lambda z: 300.0 + 0.0005 * np.asarray(z, float),
                   seed=7, wind_amp=2.0)
    for name in ("u", "v", "w", "thp", "php", "mup"):
        getattr(s, name + "0")[...] = getattr(s, name)
    nz, ny, nx = s.p.shape
    km = s.scratch((nz, ny, nx), "smag_km")
    kh = s.scratch((nz, ny, nx), "smag_kh")
    dycore.launch_wrf_smag3d_km(s, cfg, km, kh, time_t=True)
    kmv = s.scratch((nz, ny, nx), "smag_kmv")
    khv = s.scratch((nz, ny, nx), "smag_khv")

    seen: dict = {}
    watched = {"wrf_smag_vd_u": 0, "wrf_smag_vd_v": 0, "wrf_smag_vd_w": 0}
    real_get = dycore.get_kernel

    def spy(module, name):
        kernel = real_get(module, name)
        if name not in watched:
            return kernel

        def launch(grid, block, args):
            # common args, then the K, then the tendency.
            arrays = [a for a in args if getattr(a, "ndim", 0) == 3]
            seen[name] = arrays[-2].data.ptr
            return kernel(grid, block, args)
        return launch

    monkeypatch.setattr(dycore, "get_kernel", spy)
    buf = {n: s.scratch(getattr(s, f).shape, n) for n, f in (
        ("smag_ru", "u"), ("smag_rv", "v"), ("smag_rw", "w"),
        ("smag_rth", "thp"))}
    dycore.launch_wrf_smag2d_vertical(
        s, cfg, km, ru=buf["smag_ru"], rv=buf["smag_rv"],
        rw=buf["smag_rw"], rth=buf["smag_rth"], rqv=None, time_t=True,
        kmv=kmv, khv=khv, scalar_rows=None)

    assert set(seen) == set(watched), seen
    assert seen["wrf_smag_vd_u"] == kmv.data.ptr
    assert seen["wrf_smag_vd_v"] == kmv.data.ptr
    # THE fix.  WRF passes xkmh here (module_diffusion_em.F:4145-4155).
    assert seen["wrf_smag_vd_w"] == kmv.data.ptr
    assert seen["wrf_smag_vd_w"] != km.data.ptr

    # km_opt=4's identity arm: no vertical pair, everything takes xkmh.
    seen.clear()
    dycore.launch_wrf_smag2d_vertical(
        s, cfg, km, ru=buf["smag_ru"], rv=buf["smag_rv"],
        rw=buf["smag_rw"], rth=buf["smag_rth"], rqv=None, time_t=True,
        kmv=None, khv=None, scalar_rows=None)
    assert seen["wrf_smag_vd_u"] == km.data.ptr
    assert seen["wrf_smag_vd_w"] == km.data.ptr
