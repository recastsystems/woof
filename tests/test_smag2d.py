# tests/test_smag2d.py
"""2-D Smagorinsky horizontal mixing, WRF km_opt=4 (Phase 2 Task 8).

K_m = (c_s*Delta)^2 * |D_h| from the coordinate-surface horizontal
deformation (WRF module_diffusion_em.F cal_deform_and_div + smag2d_km),
applied through the WRF horizontal_diffusion flux stencils.  Covered here:
kernel-vs-float64-mirror on random fields for the K computation AND all
four application staggers (the scalar path per the WRF Prandtl carry-over:
scalars get K_m/prandtl = 3x the momentum diffusivity), the WRF v4.6.1
'v'-branch quirk that the v normal (y) fluxes carry no mass coupling
(pinned mirror-independently), the analytic pure-shear K, deformation-free
solid-body rotation, and the dycore wiring (compute once on RK stage 1,
apply every stage; moisture mixing).
"""
import dataclasses
import numpy as np
import pytest
from conftest import requires_gpu
from woof.config import RunConfig
from woof.core.grid import make_base_state, make_vertical_coord

pytestmark = pytest.mark.gpu

CS = 0.25


def _random_uv(nz=8, ny=6, nx=12, seed=0, amp=1.0):
    """Random staggered winds with the house duplicate column/row set."""
    rng = np.random.default_rng(seed)
    u = amp * rng.standard_normal((nz, ny, nx + 1))
    v = amp * rng.standard_normal((nz, ny + 1, nx))
    u[:, :, nx] = u[:, :, 0]
    v[:, ny, :] = v[:, 0, :]
    return u, v


@requires_gpu
def test_smag2d_km_matches_reference():
    import cupy as cp
    from woof.core.dycore import launch_smag2d_km
    from woof.verify.npref import np_smag2d_km
    nz, ny, nx = 8, 6, 12
    dx, dy = 100.0, 200.0
    # amp=1: cap inactive; amp=500: the WRF min(K, 10*mlen) cap engages
    for seed, amp in ((0, 1.0), (1, 500.0)):
        u, v = _random_uv(nz, ny, nx, seed=seed, amp=amp)
        km = cp.zeros((nz, ny, nx), dtype=cp.float32)
        kh = cp.zeros_like(km)
        launch_smag2d_km(cp.asarray(u, dtype=cp.float32),
                         cp.asarray(v, dtype=cp.float32),
                         km, kh, dx, dy, CS)
        km_ref, kh_ref = np_smag2d_km(u, v, dx, dy, CS)
        # FP32-FLOOR: absolute tolerance covers float32 cancellation at a
        # zero/near-zero reference; rtol=1e-4 remains the signal-scale gate.
        np.testing.assert_allclose(cp.asnumpy(km), km_ref,
                                   rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(cp.asnumpy(kh), kh_ref,
                                   rtol=1e-4, atol=1e-6)
        # WRF Prandtl semantics: scalar K is K_m/prandtl = 3x momentum K
        np.testing.assert_allclose(kh_ref, 3.0 * km_ref, rtol=1e-12)
        if amp > 1.0:
            cap = 10.0 * np.sqrt(dx * dy)
            assert (km_ref == cap).any()      # cap engaged somewhere...
            assert (km_ref < cap).any()       # ...but not everywhere
            assert km_ref.max() <= cap


@requires_gpu
def test_smag2d_hd_matches_reference():
    import cupy as cp
    from woof.core.dycore import launch_smag2d_hd
    from woof.verify.npref import np_smag2d_hd
    nz, ny, nx = 8, 6, 12
    dx, dy = 1.0, 2.0                          # O(1) keeps FP32-vs-64 tight
    rng = np.random.default_rng(7)
    xk = rng.uniform(0.5, 2.0, (nz, ny, nx))
    mut = rng.uniform(0.8, 1.2, (ny, nx))
    fields = {
        "": rng.standard_normal((nz, ny, nx)),
        "x": rng.standard_normal((nz, ny, nx + 1)),
        "y": rng.standard_normal((nz, ny + 1, nx)),
        "z": rng.standard_normal((nz + 1, ny, nx)),
    }
    fields["x"][:, :, nx] = fields["x"][:, :, 0]
    fields["y"][:, ny, :] = fields["y"][:, 0, :]
    for stag, f in fields.items():
        nlev = f.shape[0]
        c1 = rng.uniform(0.9, 1.1, nlev)
        c2 = rng.uniform(0.0, 0.2, nlev)
        tend = cp.zeros(f.shape, dtype=cp.float32)
        launch_smag2d_hd(cp.asarray(f, dtype=cp.float32),
                         cp.asarray(xk, dtype=cp.float32),
                         cp.asarray(mut, dtype=cp.float32),
                         cp.asarray(c1, dtype=cp.float32),
                         cp.asarray(c2, dtype=cp.float32),
                         dx, dy, tend, stagger=stag)
        ref = np_smag2d_hd(f, xk, mut, c1, c2, dx, dy, stagger=stag)
        # FP32-FLOOR: absolute tolerance covers float32 cancellation at a
        # zero/near-zero reference; rtol=1e-4 remains the signal-scale gate.
        np.testing.assert_allclose(cp.asnumpy(tend), ref,
                                   rtol=1e-4, atol=1e-5, err_msg=stag)
        if stag == "z":                        # BC-pinned w levels untouched
            assert np.all(ref[0] == 0.0) and np.all(ref[-1] == 0.0)
            assert float(cp.abs(tend[0]).max()) == 0.0
            assert float(cp.abs(tend[-1]).max()) == 0.0


@requires_gpu
@pytest.mark.parametrize("open_x,open_y",
                         [(True, False), (False, True), (True, True)])
def test_smag2d_hd_open_matches_mirror(open_x, open_y):
    """Open-BC smag application (kernel accurate boundary read + width-1
    strip zeroing) vs the float64 mirror with the same flags, all four
    staggers.

    The boundary-normal outermost computed face (u at nx-1 under open_x /
    v at ny-1 under open_y) must use the TRUE boundary datum -- WRF's
    horizontal_diffusion 'u' computes i_end = ide-1 with field(i+1) =
    u(ide) (module_big_step_utilities_em.F:2786-2787/2819; 'v' analogous
    at 2834-2837/2861) -- NOT the wrapped opposite-boundary value (the
    Task-12 final-review fix).  The staggered fields here carry a
    boundary column/row DISTINCT from column/row 0, so the wrap and the
    accurate read provably differ.
    """
    import cupy as cp

    from woof.core.dycore import _zero_open_strips, launch_smag2d_hd
    from woof.verify.npref import np_smag2d_hd
    nz, ny, nx = 8, 6, 12
    dx, dy = 1.0, 2.0
    rng = np.random.default_rng(11)
    xk = rng.uniform(0.5, 2.0, (nz, ny, nx))
    mut = rng.uniform(0.8, 1.2, (ny, nx))
    fields = {
        "": rng.standard_normal((nz, ny, nx)),
        "x": rng.standard_normal((nz, ny, nx + 1)),
        "y": rng.standard_normal((nz, ny + 1, nx)),
        "z": rng.standard_normal((nz + 1, ny, nx)),
    }
    # deliberately NO duplicate column/row: under open BCs the boundary
    # face is a genuine degree of freedom distinct from the opposite side.
    for stag, f in fields.items():
        nlev = f.shape[0]
        c1 = rng.uniform(0.9, 1.1, nlev)
        c2 = rng.uniform(0.0, 0.2, nlev)
        cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, ztop=1.0,
                        dt=1.0, run_seconds=0.0,
                        open_x=open_x, open_y=open_y)
        tend = cp.zeros(f.shape, dtype=cp.float32)
        launch_smag2d_hd(cp.asarray(f, dtype=cp.float32),
                         cp.asarray(xk, dtype=cp.float32),
                         cp.asarray(mut, dtype=cp.float32),
                         cp.asarray(c1, dtype=cp.float32),
                         cp.asarray(c2, dtype=cp.float32),
                         dx, dy, tend, stagger=stag,
                         open_x=open_x, open_y=open_y)
        _zero_open_strips(tend, cfg, 1)   # as add_smag2d_tendencies does
        got = cp.asnumpy(tend)
        ref = np_smag2d_hd(f, xk, mut, c1, c2, dx, dy, stagger=stag,
                           open_x=open_x, open_y=open_y)
        # FP32-FLOOR: absolute tolerance covers float32 cancellation at a
        # zero/near-zero reference; rtol=1e-4 remains the signal-scale gate.
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5,
                                   err_msg=stag)
        # WRF's excluded width-1 strip is zero; first live entry is not.
        if open_x:
            assert (got[:, :, :1] == 0.0).all(), stag
            assert (got[:, :, -1:] == 0.0).all(), stag
            assert np.abs(got[:, :, 1]).max() > 0.0, stag
            assert np.abs(got[:, :, -2]).max() > 0.0, stag
        if open_y:
            assert (got[:, :1, :] == 0.0).all(), stag
            assert (got[:, -1:, :] == 0.0).all(), stag
            assert np.abs(got[:, 1, :]).max() > 0.0, stag
            assert np.abs(got[:, -2, :]).max() > 0.0, stag
        # Accurate-read pin, mirror-independently meaningful: at the
        # boundary-normal live face the open answer must differ from the
        # periodic wrap's (the boundary datum differs from column/row 0
        # by construction) and the device kernel must match the OPEN form.
        ref_per = np_smag2d_hd(f, xk, mut, c1, c2, dx, dy, stagger=stag)
        if stag == "x" and open_x:
            assert np.abs(ref[:, :, nx - 1]
                          - ref_per[:, :, nx - 1]).max() > 0.0
        if stag == "y" and open_y:
            assert np.abs(ref[:, ny - 1, :]
                          - ref_per[:, ny - 1, :]).max() > 0.0


@requires_gpu
def test_smag2d_hd_v_yflux_not_mass_coupled():
    """WRF v4.6.1 quirk, transcribed exactly: horizontal_diffusion's 'v'
    branch computes its normal (y) fluxes as mkrdym/p = (msfty/msftx)*
    xkmhd*rdy with NO (c1*MUT+c2) factor (module_big_step_utilities_em.F
    lines 2854-2855), unlike the 'u' branch normal fluxes.  With v varying
    only in y the transverse (x) fluxes vanish, so the v tendency must be
    completely independent of mut -- asserted here against WRF's form
    directly, so a mirror+kernel shared re-coupling cannot hide."""
    import cupy as cp
    from woof.core.dycore import launch_smag2d_hd
    from woof.verify.npref import np_smag2d_hd
    nz, ny, nx = 4, 6, 8
    dx, dy = 1.0, 2.0
    rng = np.random.default_rng(3)
    xk = rng.uniform(0.5, 2.0, (nz, ny, nx))
    c1 = rng.uniform(0.9, 1.1, nz)
    c2 = rng.uniform(0.0, 0.2, nz)
    prof = rng.standard_normal(ny + 1)
    prof[ny] = prof[0]                         # duplicate v row
    v = np.broadcast_to(prof[None, :, None], (nz, ny + 1, nx)).copy()
    tends = []
    for mut in (np.ones((ny, nx)), rng.uniform(50.0, 100.0, (ny, nx))):
        ref = np_smag2d_hd(v, xk, mut, c1, c2, dx, dy, stagger="y")
        tend = cp.zeros(v.shape, dtype=cp.float32)
        launch_smag2d_hd(cp.asarray(v, dtype=cp.float32),
                         cp.asarray(xk, dtype=cp.float32),
                         cp.asarray(mut, dtype=cp.float32),
                         cp.asarray(c1, dtype=cp.float32),
                         cp.asarray(c2, dtype=cp.float32),
                         dx, dy, tend, stagger="y")
        # FP32-FLOOR: absolute tolerance covers float32 cancellation at a
        # zero/near-zero reference; rtol=1e-4 remains the signal-scale gate.
        np.testing.assert_allclose(cp.asnumpy(tend), ref,
                                   rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(ref, np_smag2d_hd(
            v, xk, np.zeros((ny, nx)), c1, c2, dx, dy, stagger="y"))
        tends.append(cp.asnumpy(tend))
    assert np.abs(tends[0]).max() > 0.0        # y-mixing is active
    np.testing.assert_array_equal(tends[0], tends[1])


@requires_gpu
def test_pure_shear_analytic_k():
    """A periodic u(y), v=0 pure shear matches its full-field discrete K.

    The former linear profile was non-periodic and dropped two boundary
    rows.  This sinusoid is periodic, so the independent analytic gate now
    covers every cell, including both seams.
    """
    import cupy as cp
    from woof.core.dycore import launch_smag2d_km
    nz, ny, nx = 4, 16, 16
    dx = dy = 1000.0
    y_u = (np.arange(ny) + 0.5) * dy
    amplitude = 8.0
    u_profile = amplitude * np.sin(2.0 * np.pi * y_u / (ny * dy))
    u = np.broadcast_to(u_profile[None, :, None],
                        (nz, ny, nx + 1)).astype(np.float32).copy()
    v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
    km = cp.zeros((nz, ny, nx), dtype=cp.float32)
    kh = cp.zeros_like(km)
    launch_smag2d_km(cp.asarray(u), cp.asarray(v), km, kh, dx, dy, CS)
    d12_corner = (u_profile - np.roll(u_profile, 1)) / dy
    shear = 0.5 * (d12_corner + np.roll(d12_corner, -1))
    k_exact = CS * CS * dx * dy * np.abs(shear)
    k_exact = np.broadcast_to(k_exact[None, :, None], (nz, ny, nx))
    np.testing.assert_allclose(cp.asnumpy(km), k_exact, rtol=0.01,
                               atol=1e-5 * k_exact.max())
    np.testing.assert_allclose(cp.asnumpy(kh), 3.0 * k_exact, rtol=0.01,
                               atol=3e-5 * k_exact.max())


@requires_gpu
def test_solid_body_rotation_zero_k():
    """u = -omega*y, v = +omega*x is deformation-free: K ~ 0.

    omega and the spacings are chosen exactly representable in FP32 so the
    discrete cancellation is exact; the acceptance gate is |K| < 1e-8 of
    the shear-case K at the same rate magnitude.
    """
    import cupy as cp
    from woof.core.dycore import launch_smag2d_km
    nz, ny, nx = 4, 16, 16
    dx = dy = 1000.0
    omega = 2.0 ** -13
    y_u = (np.arange(ny) + 0.5) * dy           # u rows (mass rows)
    x_v = (np.arange(nx) + 0.5) * dx           # v columns (mass columns)
    u = np.broadcast_to((-omega * y_u)[None, :, None],
                        (nz, ny, nx + 1)).astype(np.float32).copy()
    v = np.broadcast_to((omega * x_v)[None, None, :],
                        (nz, ny + 1, nx)).astype(np.float32).copy()
    km = cp.zeros((nz, ny, nx), dtype=cp.float32)
    kh = cp.zeros_like(km)
    launch_smag2d_km(cp.asarray(u), cp.asarray(v), km, kh, dx, dy, CS)
    k_shear = CS * CS * dx * dy * omega        # shear case at rate omega
    # A globally linear rotation is not periodic: the omitted seam rows and
    # columns contain the deliberate coordinate jump.  The asserted interior
    # is the full domain on which the analytic solid-body field is continuous;
    # random mirror tests above cover every periodic seam cell separately.
    k_int = cp.asnumpy(km)[:, 2:-2, 2:-2]
    assert np.abs(k_int).max() < 1e-8 * k_shear


def _flat_setup(nx, ny, nz, **kw):
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0, ztop=8000.0,
                    dt=6.0, run_seconds=0.0, time_step_sound=4, **kw)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    return cfg, vc, b


@requires_gpu
@pytest.mark.parametrize("boundary_mode", ["specified", "nested"])
def test_wrf_smag_physical_boundary_k_zero_and_first_interior_scalar(
        boundary_mode):
    """WRF leaves active outer K rows zero under specified/nested BCs.

    The first interior scalar source is pinned to a direct flat-coordinate
    reduction of ``horizontal_diffusion_s_2``.  In particular, its west
    flux averages the zero boundary K with the live first-interior K; copying
    K(1) outward (the reopened M16 bug) changes this value.
    """
    import cupy as cp

    from woof.core.constants import G
    from woof.core.diagnostics import update_diagnostics
    from woof.core.dycore import (launch_wrf_smag2d_hd,
                                   launch_wrf_smag2d_km,
                                   launch_wrf_smag2d_vertical)
    from woof.core.state import init_at_rest

    nx, ny, nz = 10, 8, 6
    cfg, vc, b = _flat_setup(
        nx, ny, nz, km_opt=4, bl_pbl_physics=1,
        **{boundary_mode: True})
    state = init_at_rest(cfg, vc, b)
    assert state.qv is None
    y = (np.arange(ny) + 0.5) * cfg.dy
    shear = 8.0 * np.sin(2.0 * np.pi * y / (ny * cfg.dy))
    state.u[...] = cp.asarray(np.broadcast_to(
        shear[None, :, None], (nz, ny, nx + 1)), dtype=cp.float32)
    x_profile = (0.03 * np.arange(nx, dtype=np.float32) ** 2)
    state.thp[...] = cp.asarray(np.broadcast_to(
        x_profile[None, None, :], (nz, ny, nx)), dtype=cp.float32)
    update_diagnostics(state, cfg.hypsometric_opt)

    km = state.scratch((nz, ny, nx), "smag_km")
    kh = state.scratch((nz, ny, nx), "smag_kh")
    launch_wrf_smag2d_km(state, cfg, km, kh, time_t=False)
    source = cp.zeros_like(state.thp)
    launch_wrf_smag2d_hd(
        state, cfg, state.thp, kh, source, stagger="", time_t=False,
        full_theta=True)

    km_h = cp.asnumpy(km).astype(np.float64)
    kh_h = cp.asnumpy(kh).astype(np.float64)
    for array in (km_h, kh_h):
        np.testing.assert_array_equal(array[:, :, 0], 0.0)
        np.testing.assert_array_equal(array[:, :, -1], 0.0)
        np.testing.assert_array_equal(array[:, 0, :], 0.0)
        np.testing.assert_array_equal(array[:, -1, :], 0.0)
    assert np.max(km_h[:, 1:-1, 1:-1]) > 0.0

    rho = 1.0 / cp.asnumpy(state.alt).astype(np.float64)
    thb = cp.asnumpy(state.thb).astype(np.float64)
    if thb.ndim == 1:
        thb = thb[:, None, None]
    field = cp.asnumpy(state.thp).astype(np.float64) + thb - 300.0
    phb = cp.asnumpy(state.phb).astype(np.float64)
    if phb.ndim == 1:
        phb = phb[:, None, None]
    phi = cp.asnumpy(state.php).astype(np.float64) + phb
    dnw = cp.asnumpy(state.dnw).astype(np.float64)
    k, j, i = 2, 2, 1

    def h1(iface):
        left, right = iface - 1, iface
        return (-0.25 * (rho[k, j, left] + rho[k, j, right])
                * (kh_h[k, j, left] + kh_h[k, j, right])
                * (field[k, j, right] - field[k, j, left]) / cfg.dx)

    rdzw = G / (phi[k + 1, j, i] - phi[k, j, i])
    expected = (G / (dnw[k] * rdzw)
                * (h1(i + 1) - h1(i)) / cfg.dx)
    got = float(cp.asnumpy(source)[k, j, i])
    assert expected != 0.0
    np.testing.assert_allclose(got, expected, rtol=5.0e-4,
                               atol=1.0e-7 * abs(expected))


@requires_gpu
def test_wrf_smag_production_terrain_maps_momentum_authority():
    """Compiled production K, momentum, and theta match independent WRF math.

    This is the M16 closure fixture missing from the retained flat-kernel
    tests: total geopotential has slopes in both horizontal directions, all
    C-grid map factors are non-unity/nonuniform, u/v/w vary in x/y/z, and the
    float64 authority retains tau12, tau13, and tau23 cross components.  The
    same launch set also pins vapor-loaded density and 3-D full base theta.
    """
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.dycore import (launch_wrf_smag2d_hd,
                                   launch_wrf_smag2d_km,
                                   launch_wrf_smag2d_vertical)
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.smag2d import (wrf_periodic_momentum_authority,
                                  wrf_periodic_scalar_metric_tendency)
    from woof.core.state import init_at_rest

    nx, ny, nz = 8, 7, 6
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=900.0, dy=1100.0, ztop=9000.0,
        dt=2.0, run_seconds=0.0, terrain_opt=1, hybrid_opt=2,
        km_opt=4, bl_pbl_physics=1, c_s=0.25, moist=True)
    coord = make_vertical_coord(nz, hybrid_opt=cfg.hybrid_opt,
                                etac=cfg.etac)
    xx = 2.0 * np.pi * np.arange(nx) / nx
    yy = 2.0 * np.pi * np.arange(ny) / ny
    terrain = (260.0 + 110.0 * np.sin(xx)[None, :]
               + 75.0 * np.cos(yy)[:, None]
               + 35.0 * np.sin(yy[:, None] + xx[None, :]))
    base = make_base_state(
        coord, lambda z: 300.0 + 0.004 * np.asarray(z, dtype=np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=terrain)
    state = init_at_rest(cfg, coord, base, terrain_z=terrain)

    msft = (1.08 + 0.055 * np.sin(xx)[None, :]
            + 0.035 * np.cos(yy)[:, None]
            + 0.018 * np.sin(2.0 * yy[:, None] + xx[None, :]))
    msfu_core = 0.5 * (msft + np.roll(msft, 1, axis=1))
    msfv_core = 0.5 * (msft + np.roll(msft, 1, axis=0))
    msfu = np.concatenate([msfu_core, msfu_core[:, :1]], axis=1)
    msfv = np.concatenate([msfv_core, msfv_core[:1, :]], axis=0)
    state.set_map_coriolis(msft=msft, msfu=msfu, msfv=msfv)

    xf = 2.0 * np.pi * np.arange(nx) / nx
    xm = 2.0 * np.pi * (np.arange(nx) + 0.5) / nx
    yf = 2.0 * np.pi * np.arange(ny) / ny
    ym = 2.0 * np.pi * (np.arange(ny) + 0.5) / ny
    zm = np.arange(nz, dtype=np.float64) / (nz - 1)
    zw = np.arange(nz + 1, dtype=np.float64) / nz
    u_core = (
        4.5 * np.sin(xf[None, None, :] + 0.35 * ym[None, :, None])
        + 2.2 * np.cos(ym[None, :, None])
        + 3.0 * zm[:, None, None]
          * (1.0 + 0.22 * np.cos(xf[None, None, :])))
    v_core = (
        -3.8 * np.cos(xm[None, None, :] - 0.30 * yf[None, :, None])
        + 2.7 * np.sin(yf[None, :, None] + 0.25 * xm[None, None, :])
        - 2.1 * zm[:, None, None]
          * (1.0 + 0.17 * np.sin(yf[None, :, None])))
    w = (np.sin(np.pi * zw)[:, None, None]
         * (1.4 * np.sin(xm[None, None, :] + 0.65 * ym[None, :, None])
            + 0.55 * np.cos(2.0 * xm[None, None, :]
                            - ym[None, :, None])))
    qv = (0.018 + 0.004 * np.sin(xm)[None, None, :]
          + 0.003 * np.cos(ym)[None, :, None]
          + 0.002 * zm[:, None, None])
    thp = ((1.25 + 0.35 * zm[:, None, None])
           * np.sin(xm[None, None, :] + 0.45 * ym[None, :, None])
           + 0.55 * np.cos(2.0 * ym[None, :, None]
                           - 0.30 * xm[None, None, :]))
    u = np.concatenate([u_core, u_core[:, :, :1]], axis=2)
    v = np.concatenate([v_core, v_core[:, :1, :]], axis=1)
    state.u[...] = cp.asarray(u, dtype=cp.float32)
    state.v[...] = cp.asarray(v, dtype=cp.float32)
    state.w[...] = cp.asarray(w, dtype=cp.float32)
    state.qv[...] = cp.asarray(qv, dtype=cp.float32)
    state.thp[...] = cp.asarray(thp, dtype=cp.float32)
    update_diagnostics(state, cfg.hypsometric_opt)

    km = state.scratch((nz, ny, nx), "smag_km")
    kh = state.scratch((nz, ny, nx), "smag_kh")
    deformation = launch_wrf_smag2d_km(
        state, cfg, km, kh, time_t=False)
    ru = cp.zeros_like(state.u)
    rv = cp.zeros_like(state.v)
    rw = cp.zeros_like(state.w)
    launch_wrf_smag2d_hd(
        state, cfg, state.u, km, ru, stagger="x", time_t=False,
        deformation=deformation)
    launch_wrf_smag2d_hd(
        state, cfg, state.v, km, rv, stagger="y", time_t=False,
        deformation=deformation)
    launch_wrf_smag2d_hd(
        state, cfg, state.w, km, rw, stagger="z", time_t=False)
    rth = cp.zeros_like(state.thp)
    launch_wrf_smag2d_hd(
        state, cfg, state.thp, kh, rth, stagger="", time_t=False,
        full_theta=True)
    ru_vertical = cp.zeros_like(state.u)
    rv_vertical = cp.zeros_like(state.v)
    rw_vertical = cp.zeros_like(state.w)
    rth_vertical = cp.zeros_like(state.thp)
    rqv_vertical = cp.zeros_like(state.qv)
    launch_wrf_smag2d_vertical(
        state, cfg, km, ru=ru_vertical, rv=rv_vertical, rw=rw_vertical,
        rth=rth_vertical, rqv=rqv_vertical, time_t=False)

    phb = cp.asnumpy(state.phb).astype(np.float64)
    if phb.ndim == 1:
        phb = phb[:, None, None]
    phi = cp.asnumpy(state.php).astype(np.float64) + phb
    alt_h = cp.asnumpy(state.alt).astype(np.float64)
    qv_h = cp.asnumpy(state.qv).astype(np.float64)
    rho = (1.0 + qv_h) / alt_h
    authority = wrf_periodic_momentum_authority(
        u=cp.asnumpy(state.u), v=cp.asnumpy(state.v),
        w=cp.asnumpy(state.w), phi=phi,
        rho=rho,
        msft=cp.asnumpy(state.msft), msfu=cp.asnumpy(state.msfu),
        msfv=cp.asnumpy(state.msfv), fnm=cp.asnumpy(state.fnm),
        fnp=cp.asnumpy(state.fnp), dn=cp.asnumpy(state.dn),
        dnw=cp.asnumpy(state.dnw), cf1=float(state.cf1),
        cf2=float(state.cf2), cf3=float(state.cf3), dx=cfg.dx, dy=cfg.dy,
        c_s=cfg.c_s)
    dry_rho_authority = wrf_periodic_momentum_authority(
        u=cp.asnumpy(state.u), v=cp.asnumpy(state.v),
        w=cp.asnumpy(state.w), phi=phi, rho=1.0 / alt_h,
        msft=cp.asnumpy(state.msft), msfu=cp.asnumpy(state.msfu),
        msfv=cp.asnumpy(state.msfv), fnm=cp.asnumpy(state.fnm),
        fnp=cp.asnumpy(state.fnp), dn=cp.asnumpy(state.dn),
        dnw=cp.asnumpy(state.dnw), cf1=float(state.cf1),
        cf2=float(state.cf2), cf3=float(state.cf3), dx=cfg.dx, dy=cfg.dy,
        c_s=cfg.c_s)

    assert np.ptp(phi[0]) > 100.0 * 9.81
    assert np.ptp(msft) > 0.05 and not np.allclose(msft, 1.0)
    km_ref = authority["km"]
    km_scale = max(float(np.max(np.abs(km_ref))), 1.0e-12)
    np.testing.assert_allclose(
        cp.asnumpy(km), km_ref, rtol=2.0e-3, atol=2.0e-5 * km_scale)
    np.testing.assert_allclose(
        cp.asnumpy(kh), authority["kh"], rtol=2.0e-3,
        atol=6.0e-5 * km_scale)

    got = {"ru": cp.asnumpy(ru), "rv": cp.asnumpy(rv),
           "rw": cp.asnumpy(rw)}
    for name in ("ru", "rv", "rw"):
        ref = authority[name]
        scale = max(float(np.max(np.abs(ref))), 1.0e-10)
        assert scale > 1.0e-6
        np.testing.assert_allclose(
            got[name], ref, rtol=3.0e-3,
            atol=4.0e-5 * scale, err_msg=name)
        dry_delta = np.max(np.abs(ref - dry_rho_authority[name]))
        assert dry_delta > 5.0e-3 * scale
    for name, actual in (
            ("ru_vertical", ru_vertical),
            ("rv_vertical", rv_vertical),
            ("rw_vertical", rw_vertical)):
        ref = authority[name]
        scale = max(float(np.max(np.abs(ref))), 1.0e-10)
        assert scale > 1.0e-6
        np.testing.assert_allclose(
            cp.asnumpy(actual), ref, rtol=3.0e-3,
            atol=4.0e-5 * scale, err_msg=name)
    # xkhv is exactly zero under km_opt=4; with no attached surface writer
    # the held USTM/HFX/QFX are cold zero as well.
    np.testing.assert_array_equal(cp.asnumpy(rth_vertical), 0.0)
    np.testing.assert_array_equal(cp.asnumpy(rqv_vertical), 0.0)

    assert (np.max(np.abs(authority["ru_cross"]))
            > 1.0e-3 * np.max(np.abs(authority["ru"])))
    assert (np.max(np.abs(authority["rv_cross"]))
            > 1.0e-3 * np.max(np.abs(authority["rv"])))
    for component in ("rw_x", "rw_y"):
        assert (np.max(np.abs(authority[component]))
                > 1.0e-3 * np.max(np.abs(authority["rw"])))

    thb_h = cp.asnumpy(state.thb).astype(np.float64)
    assert thb_h.ndim == 3 and thb_h.shape == thp.shape
    scalar_args = dict(
        kh=authority["kh"], rho=rho, phi=phi,
        dnw=cp.asnumpy(state.dnw), dn=cp.asnumpy(state.dn),
        fnm=cp.asnumpy(state.fnm), fnp=cp.asnumpy(state.fnp),
        cf1=float(state.cf1), cf2=float(state.cf2), cf3=float(state.cf3),
        dx=cfg.dx, dy=cfg.dy, msft=cp.asnumpy(state.msft),
        msfu=cp.asnumpy(state.msfu), msfv=cp.asnumpy(state.msfv))
    theta_ref = wrf_periodic_scalar_metric_tendency(
        field=cp.asnumpy(state.thp).astype(np.float64) + thb_h - 300.0,
        **scalar_args)
    perturbation_only = wrf_periodic_scalar_metric_tendency(
        field=cp.asnumpy(state.thp), **scalar_args)
    theta_scale = max(float(np.max(np.abs(theta_ref))), 1.0e-10)
    np.testing.assert_allclose(
        cp.asnumpy(rth), theta_ref, rtol=3.0e-3,
        atol=4.0e-5 * theta_scale)
    assert np.max(np.abs(theta_ref - perturbation_only)) \
        > 1.0e-3 * theta_scale


@requires_gpu
def test_wrf_smag_pbl_off_surface_flux_policy():
    """Pin vertical_diffusion_2's isfflx=1 wall/heat/moisture branches."""
    from types import SimpleNamespace

    import cupy as cp

    from woof.core import constants as c
    from woof.core.diagnostics import update_diagnostics
    from woof.core.dycore import launch_wrf_smag2d_vertical
    from woof.core.moist import init_moist_balanced

    nx, ny, nz = 6, 5, 4
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0, ztop=6000.0,
        dt=2.0, run_seconds=0.0, km_opt=4, bl_pbl_physics=0,
        sf_sfclay_physics=1, moist=True)
    coord = make_vertical_coord(nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.004 * np.asarray(z),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base, lambda z: 0.01 + 0.0 * np.asarray(z))
    state.u[...] = cp.float32(4.0)
    state.v[...] = cp.float32(3.0)
    update_diagnostics(state, cfg.hypsometric_opt)

    shape = (ny, nx)
    state.physics = SimpleNamespace(fields={
        "ustm": cp.full(shape, 0.5, dtype=cp.float32),
        "hfx": cp.full(shape, 100.0, dtype=cp.float32),
        "qfx": cp.full(shape, 1.0e-4, dtype=cp.float32),
    })
    km = cp.zeros((nz, ny, nx), dtype=cp.float32)
    ru = cp.zeros_like(state.u)
    rv = cp.zeros_like(state.v)
    rw = cp.zeros_like(state.w)
    rth = cp.zeros_like(state.thp)
    rqv = cp.zeros_like(state.qv)
    launch_wrf_smag2d_vertical(
        state, cfg, km, ru=ru, rv=rv, rw=rw, rth=rth, rqv=rqv,
        time_t=False)

    dnw0 = float(cp.asnumpy(state.dnw)[0])
    rho = cp.asnumpy((1.0 + state.qv[0]) / state.alt[0])
    expected_u = c.G * (0.5 ** 2) * (4.0 / 5.0) * rho / dnw0
    expected_v = c.G * (0.5 ** 2) * (3.0 / 5.0) * rho / dnw0
    expected_th = (
        -c.G * 100.0 / (c.CP * (1.0 + 0.8 * 0.01)) / dnw0)
    expected_qv = -c.G * 1.0e-4 / dnw0
    np.testing.assert_allclose(cp.asnumpy(ru[0, :, :nx]), expected_u,
                               rtol=2.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(cp.asnumpy(rv[0, :ny]), expected_v,
                               rtol=2.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(cp.asnumpy(rth[0]), expected_th,
                               rtol=2.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(cp.asnumpy(rqv[0]), expected_qv,
                               rtol=2.0e-6, atol=1.0e-10)
    np.testing.assert_array_equal(cp.asnumpy(ru[1:]), 0.0)
    np.testing.assert_array_equal(cp.asnumpy(rv[1:]), 0.0)
    np.testing.assert_array_equal(cp.asnumpy(rw), 0.0)
    np.testing.assert_array_equal(cp.asnumpy(rth[1:]), 0.0)
    np.testing.assert_array_equal(cp.asnumpy(rqv[1:]), 0.0)


@requires_gpu
def test_step_rejects_unsupported_km_opt():
    # km_opt=3 joined the admitted set with the 3-D Smagorinsky port
    # (tests/test_smag3d.py), km_opt=2 with the prognostic-TKE port
    # (tests/test_tke_km2.py) and km_opt=5 with the SMS-3DTKE port, so every
    # WRF value runs; a value WRF does not define is still refused.
    from woof.core.dycore import step
    from woof.core.state import init_at_rest
    cfg, vc, b = _flat_setup(8, 1, 8, km_opt=6)
    s = init_at_rest(cfg, vc, b)
    with pytest.raises(ValueError, match="km_opt"):
        step(s, cfg)


@requires_gpu
def test_step_km_opt4_dycore_wiring():
    """WRF timing semantics: K comes from the TIME-T fields on RK stage 1.

    Step 1 of a bubble started at rest therefore has K = 0 identically and
    must match the km_opt=1 run exactly; once winds exist the mixing
    engages and the runs diverge.
    """
    import cupy as cp
    from woof.core.dycore import run_steps, step
    from woof.core.physics import PhysicsTendencies
    from woof.core.state import init_theta_perturbation

    class _ZeroPBL:
        """Isolate horizontal km_opt=4 while satisfying its PBL-on scope."""

        def __init__(self, state):
            self.tendencies = PhysicsTendencies.zeros(state)

        def compute(self, _state, _cfg):
            return self.tendencies

    def build(km_opt):
        cfg, vc, b = _flat_setup(
            16, 16, 8, km_opt=km_opt,
            bl_pbl_physics=(1 if km_opt == 4 else 0))

        def thp_func(x, z):
            r = np.sqrt((x[None, None, :] / 2000.0) ** 2
                        + ((z[:, None, None] - 2000.0) / 1000.0) ** 2)
            return (2.0 * np.maximum(0.0, 1.0 - r)
                    * np.ones((1, cfg.ny, 1)))
        state = init_theta_perturbation(cfg, vc, b, thp_func)
        if km_opt == 4:
            state.physics = _ZeroPBL(state)
        return cfg, state

    cfg1, s1 = build(1)
    cfg4, s4 = build(4)
    step(s1, cfg1)
    step(s4, cfg4)
    for name in ("u", "v", "w", "thp", "php", "mup"):
        assert bool((getattr(s1, name) == getattr(s4, name)).all()), name
    run_steps(s1, cfg1, 3)
    run_steps(s4, cfg4, 3)
    assert float(cp.abs(s1.u - s4.u).max()) > 0.0    # mixing engaged
    assert float(cp.abs(s1.thp - s4.thp).max()) > 0.0
    for s in (s1, s4):
        assert np.isfinite(cp.asnumpy(s.thp)).all()
        assert np.isfinite(cp.asnumpy(s.u)).all()


def _coupled_qv_mass(s):
    """FP64 coupled vapor mass sum((-dnw)*(c1h*mu+c2h)*qv)."""
    import cupy as cp
    dnw = cp.asnumpy(s.dnw).astype(np.float64)[:, None, None]
    c1h = cp.asnumpy(s.c1h).astype(np.float64)[:, None, None]
    c2h = cp.asnumpy(s.c2h).astype(np.float64)[:, None, None]
    mu = cp.asnumpy(s.mub2d + s.mup).astype(np.float64)
    qv = cp.asnumpy(s.qv).astype(np.float64)
    return float(((-dnw) * (c1h * mu[None] + c2h) * qv).sum())


@requires_gpu
def test_step_km_opt4_moist_mixing():
    """Moisture receives the K_h mixing: km_opt=4 changes qv measurably vs
    the km_opt=1 control under sheared flow and follows WRF's source budget.

    The moist-balanced qv blob has horizontally varying total geopotential,
    so WRF's terrain-coordinate ``horizontal_diffusion_s`` metric terms do
    not telescope to exact zero (the historical flat shortcut did).  Pin the
    CUDA source pointwise to the independent float64 WRF algebra and compare
    final coupled mass with that accumulated authority budget.  A separate
    host test retains exact conservation for truly flat zx=zy=0 coordinates.
    """
    import cupy as cp
    from woof.core.diagnostics import update_diagnostics
    from woof.core.dycore import run_steps, step
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import PhysicsTendencies
    from woof.core.smag2d import wrf_periodic_scalar_metric_tendency
    nx = ny = 16
    nz = 8

    class _ZeroPBL:
        """Isolate horizontal/scalar diffusion from PBL tendencies."""

        def __init__(self, state):
            self.tendencies = PhysicsTendencies.zeros(state)

        def compute(self, _state, _cfg):
            return self.tendencies

    def build(km_opt):
        cfg, vc, b = _flat_setup(
            nx, ny, nz, km_opt=km_opt, moist=True,
            bl_pbl_physics=(1 if km_opt == 4 else 0))
        cfg = dataclasses.replace(cfg, **{"dt": 2.0})

        def qv_func(z):
            q = np.zeros((nz, ny, nx))
            q[:3, 6:10, 6:10] = 0.01           # compact low-level blob
            return q
        s = init_moist_balanced(cfg, vc, b, qv_func)
        y = (np.arange(ny) + 0.5) * cfg.dy     # sinusoidal shear u(y)
        prof = 10.0 * np.sin(2.0 * np.pi * y / (ny * cfg.dy))
        s.u[...] = cp.asarray(np.broadcast_to(
            prof[None, :, None], (nz, ny, nx + 1)), dtype=cp.float32)
        if km_opt == 4:
            s.physics = _ZeroPBL(s)
        return cfg, s

    cfg1, s1 = build(1)
    cfg4, s4 = build(4)
    m0 = _coupled_qv_mass(s4)
    run_steps(s1, cfg1, 2)
    expected_source_delta = 0.0
    dnw = cp.asnumpy(s4.dnw).astype(np.float64)
    dn = cp.asnumpy(s4.dn).astype(np.float64)
    fnm = cp.asnumpy(s4.fnm).astype(np.float64)
    fnp = cp.asnumpy(s4.fnp).astype(np.float64)
    for _ in range(2):
        # Capture the time-t inputs.  step() recomputes the same diagnostics
        # immediately before its held Smagorinsky source.
        update_diagnostics(s4, cfg4.hypsometric_opt)
        field = cp.asnumpy(s4.qv).astype(np.float64)
        thp = cp.asnumpy(s4.thp).astype(np.float64)
        thb = cp.asnumpy(s4.thb).astype(np.float64)
        if thb.ndim == 1:
            thb = thb[:, None, None]
        theta_minus_t0 = thp + thb - 300.0
        rho = (1.0 + field) / cp.asnumpy(s4.alt).astype(np.float64)
        phb = cp.asnumpy(s4.phb).astype(np.float64)
        if phb.ndim == 1:
            phb = phb[:, None, None]
        phi = cp.asnumpy(s4.php).astype(np.float64) + phb
        step(s4, cfg4)
        kh = cp.asnumpy(s4.scratch(field.shape, "smag_kh")).astype(
            np.float64)
        authority = wrf_periodic_scalar_metric_tendency(
            field=field, kh=kh, rho=rho, phi=phi,
            dnw=dnw, dn=dn, fnm=fnm, fnp=fnp,
            cf1=float(s4.cf1), cf2=float(s4.cf2), cf3=float(s4.cf3),
            dx=cfg4.dx, dy=cfg4.dy,
        )
        cuda_source = cp.asnumpy(
            s4.scratch(field.shape, "smag_rqv")).astype(np.float64)
        scale = max(float(np.max(np.abs(authority))), 1.0e-12)
        np.testing.assert_allclose(cuda_source, authority, rtol=5.0e-4,
                                   atol=3.0e-6 * scale)
        theta_authority = wrf_periodic_scalar_metric_tendency(
            field=theta_minus_t0, kh=kh, rho=rho, phi=phi,
            dnw=dnw, dn=dn, fnm=fnm, fnp=fnp,
            cf1=float(s4.cf1), cf2=float(s4.cf2), cf3=float(s4.cf3),
            dx=cfg4.dx, dy=cfg4.dy,
        )
        perturbation_only = wrf_periodic_scalar_metric_tendency(
            field=thp, kh=kh, rho=rho, phi=phi,
            dnw=dnw, dn=dn, fnm=fnm, fnp=fnp,
            cf1=float(s4.cf1), cf2=float(s4.cf2), cf3=float(s4.cf3),
            dx=cfg4.dx, dy=cfg4.dy,
        )
        cuda_theta = cp.asnumpy(
            s4.scratch(field.shape, "smag_rth")).astype(np.float64)
        theta_scale = max(float(np.max(np.abs(theta_authority))), 1.0e-12)
        np.testing.assert_allclose(
            cuda_theta, theta_authority, rtol=5.0e-4,
            atol=2.0e-5 * theta_scale)
        assert np.max(np.abs(theta_authority - perturbation_only)) \
            > 1.0e-5 * theta_scale
        expected_source_delta += cfg4.dt * float(np.sum(
            -dnw[:, None, None] * authority, dtype=np.float64))
    dq = float(cp.abs(s4.qv - s1.qv).max())
    assert dq > 1e-6                           # scalar mixing engaged
    actual_delta = _coupled_qv_mass(s4) - m0
    assert abs(expected_source_delta) / m0 > 1.0e-6
    assert abs(actual_delta - expected_source_delta) / m0 <= 1.0e-6
    assert np.isfinite(cp.asnumpy(s4.qv)).all()
    assert float(s4.qv.min()) > -1e-6          # mixing may not blow up q


@requires_gpu
@pytest.mark.parametrize(
    "diff6", (0, 2), ids=("smag-only-mixed-arena", "smag-diff6"))
def test_smag_diff6_alias_poison_is_bitwise_exact(diff6):
    """Every km_opt=4 path keeps simultaneous x/y scratch distinct."""
    import types

    import cupy as cp

    from woof.core.dycore import step
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import PhysicsTendencies
    from woof.core.state import build_shared_scratch_arena

    nx = ny = 16
    nz = 8
    cfg, vc, base = _flat_setup(
        nx, ny, nz, km_opt=4, moist=True, bl_pbl_physics=1)
    cfg = dataclasses.replace(cfg, **{"dt": 2.0,
                       "diff_6th_opt": diff6,
                       "diff_6th_factor": 0.12 if diff6 else 0.0})

    class _ZeroPBL:
        def __init__(self, state):
            self.tendencies = PhysicsTendencies.zeros(state)

        def compute(self, _state, _cfg):
            return self.tendencies

    def build():
        def qv_func(_z):
            q = np.zeros((nz, ny, nx), dtype=np.float64)
            q[:4, 5:11, 4:12] = 0.012
            return q

        state = init_moist_balanced(cfg, vc, base, qv_func)
        y = (np.arange(ny) + 0.5) * cfg.dy
        profile = 9.0 * np.sin(2.0 * np.pi * y / (ny * cfg.dy))
        state.u[...] = cp.asarray(np.broadcast_to(
            profile[None, :, None], (nz, ny, nx + 1)), dtype=cp.float32)
        state.physics = _ZeroPBL(state)
        return state

    reference = build()
    aliased = build()
    arena_domains = [types.SimpleNamespace(run=cfg)]
    if not diff6:
        # Exercise the cross-domain case: another domain contributes z/m to
        # the shared arena even though this Smag domain itself is diff6-off.
        diff_cfg = dataclasses.replace(cfg, **{"km_opt": 1,
                                "diff_6th_opt": 2,
                                "diff_6th_factor": 0.12})
        arena_domains.append(types.SimpleNamespace(run=diff_cfg))
    arena = build_shared_scratch_arena(tuple(arena_domains))
    aliased._scratch_arena = arena

    x = arena.view(arena.slot_shapes["diff6_x"], "diff6_x")
    y = arena.view(arena.slot_shapes["diff6_y"], "diff6_y")
    z = arena.view(arena.slot_shapes["diff6_z"], "diff6_z")
    mass = arena.view(arena.slot_shapes["diff6_m"], "diff6_m")
    assert x.data.ptr == z.data.ptr == mass.data.ptr
    assert y.data.ptr != z.data.ptr

    compared = ("u", "v", "w", "thp", "php", "mup",
                "qv", "qc", "qr", "p", "al", "alt")
    for _ in range(2):
        step(reference, cfg)
        z.fill(cp.float32(np.nan))
        y.fill(cp.float32(np.nan))
        step(aliased, cfg)
        cp.cuda.get_current_stream().synchronize()
        for name in compared:
            cp.testing.assert_array_equal(
                getattr(aliased, name), getattr(reference, name), err_msg=name)
        assert aliased.elapsed_seconds == reference.elapsed_seconds


@requires_gpu
def test_smag_acoustic_alias_poison_is_bitwise_exact():
    """Poisoned shared K/acoustic backings reproduce separate allocations."""
    import types

    import cupy as cp

    from woof.core.dycore import step
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import PhysicsTendencies
    from woof.core.state import build_shared_scratch_arena

    nx = ny = 16
    nz = 8
    cfg, vc, base = _flat_setup(
        nx, ny, nz, km_opt=4, moist=True, bl_pbl_physics=1)
    cfg = dataclasses.replace(cfg, **{"dt": 2.0})

    class _ZeroPBL:
        def __init__(self, state):
            self.tendencies = PhysicsTendencies.zeros(state)

        def compute(self, _state, _cfg):
            return self.tendencies

    def build():
        def qv_func(_z):
            q = np.zeros((nz, ny, nx), dtype=np.float64)
            q[:4, 5:11, 4:12] = 0.012
            return q

        state = init_moist_balanced(cfg, vc, base, qv_func)
        y = (np.arange(ny) + 0.5) * cfg.dy
        profile = 9.0 * np.sin(2.0 * np.pi * y / (ny * cfg.dy))
        state.u[...] = cp.asarray(np.broadcast_to(
            profile[None, :, None], (nz, ny, nx + 1)), dtype=cp.float32)
        state.physics = _ZeroPBL(state)
        return state

    reference = build()
    aliased = build()
    arena = build_shared_scratch_arena(
        (types.SimpleNamespace(run=cfg),))
    aliased._scratch_arena = arena

    pairs = (("smag_km", "acoustic_alpha"),
             ("smag_kh", "acoustic_gamma"))
    for smag, acoustic in pairs:
        assert (arena.view(arena.slot_shapes[smag], smag).data.ptr
                == arena.view(arena.slot_shapes[acoustic], acoustic).data.ptr)

    compared = ("u", "v", "w", "thp", "php", "mup",
                "qv", "qc", "qr", "p", "al", "alt")
    for _ in range(2):
        step(reference, cfg)
        for _smag, acoustic in pairs:
            arena.view(arena.slot_shapes[acoustic], acoustic).fill(
                cp.float32(np.nan))
        step(aliased, cfg)
        cp.cuda.get_current_stream().synchronize()
        for name in compared:
            cp.testing.assert_array_equal(
                getattr(aliased, name), getattr(reference, name), err_msg=name)
        assert aliased.elapsed_seconds == reference.elapsed_seconds
