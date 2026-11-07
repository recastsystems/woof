"""YSU PBL column scheme (Phase 3 Task 11).

The float64 mirror and CUDA kernel follow WRF v4.6.1
``phys/physics_mmm/bl_ysu.F90`` for the non-BEP path, including the
Registry-default top-down cloud mixing option.  These tests cover both surface-stability
branches, countergradient/entrainment mixing, the implicit column solve,
and the plan's six-hour convective-growth and stable-decay cases.
"""

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig, load_config


def _grid(nz=36, ztop=6000.0):
    """Hydrostatic synthetic column in the kernel's ``(nz, ny, nx)`` form."""
    z_ifc = np.linspace(0.0, ztop, nz + 1)
    z = 0.5 * (z_ifc[:-1] + z_ifc[1:])
    dz = np.diff(z_ifc)
    p_ifc = 100000.0 * np.exp(-z_ifc / 8200.0)
    p = 0.5 * (p_ifc[:-1] + p_ifc[1:])
    exner = (p / 100000.0) ** (287.0 / 1004.5)
    return z, dz, p, p_ifc, exner


def _column(theta=None, u=None, v=None, qv=None, *, nz=36, ztop=6000.0,
            hfx=0.0, qfx=0.0, ust=0.0, br=0.0, xland=1.0, dt=60.0):
    z, dz, p, p_ifc, exner = _grid(nz, ztop)
    theta = 300.0 + 0.003 * z if theta is None else np.asarray(theta, float)
    u = np.full(nz, 8.0) if u is None else np.asarray(u, float)
    v = np.zeros(nz) if v is None else np.asarray(v, float)
    qv = 0.010 * np.exp(-z / 2400.0) if qv is None else np.asarray(qv, float)
    zero = np.zeros(nz)
    return dict(
        u=u, v=v, theta=theta, qv=qv, qc=zero.copy(), qi=zero.copy(),
        p=p, p_interface=p_ifc, exner=exner, dz=dz,
        psfc=100000.0, znt=0.10, ust=ust, hfx=hfx, qfx=qfx,
        wspd=max(float(np.hypot(u[0], v[0])), 0.1), br=br,
        psim=np.log(max(z[0] / 0.10, 1.01)),
        psih=np.log(max(z[0] / 0.01, 1.01)), xland=xland,
        u10=float(u[0]), v10=float(v[0]), dt=dt,
    )


def _batched_columns(nz=32, ny=2, nx=5, seed=11):
    """Randomized stable and unstable columns for the CUDA comparison."""
    z, dz1, p1, pi1, ex1 = _grid(nz, 5000.0)
    rng = np.random.default_rng(seed)
    shp = (nz, ny, nx)
    theta = 298.0 + 0.004 * z[:, None, None]
    theta = np.broadcast_to(theta, shp).copy()
    theta += rng.normal(0.0, 0.08, shp)
    u = 5.0 + 0.002 * z[:, None, None] + rng.normal(0.0, 0.15, shp)
    v = -1.0 + 0.001 * z[:, None, None] + rng.normal(0.0, 0.15, shp)
    qv = 0.011 * np.exp(-z[:, None, None] / 2200.0)
    qv = np.broadcast_to(qv, shp).copy() * rng.uniform(0.95, 1.05, shp)
    qc = np.where((z[:, None, None] > 700.0)
                  & (z[:, None, None] < 1400.0), 2.0e-5, 0.0)
    qc = np.broadcast_to(qc, shp).copy()
    qi = np.zeros(shp)
    f3 = lambda a: np.ascontiguousarray(np.broadcast_to(a, shp), np.float32)
    f2 = lambda a: np.ascontiguousarray(a, np.float32)
    br = np.array([[-0.08, 0.12, -0.03, 0.20, -0.15],
                   [0.10, -0.05, 0.18, -0.12, 0.06]])
    hfx = np.where(br <= 0.0, 180.0, -25.0)
    qfx = np.where(br <= 0.0, 8.0e-5, 0.0)
    ust = np.where(br <= 0.0, 0.42, 0.24)
    # A calm column (ust = hfx = qfx = 0) in the device batch: WRF runs the
    # whole scheme there, and so do the kernel and the mirror.
    hfx[0, 0] = qfx[0, 0] = ust[0, 0] = 0.0
    wspd = np.hypot(u[0], v[0])
    return dict(
        u=f2(u), v=f2(v), theta=f2(theta), qv=f2(qv), qc=f2(qc),
        qi=f2(qi), p=f3(p1[:, None, None]),
        p_interface=np.ascontiguousarray(
            np.broadcast_to(pi1[:, None, None], (nz + 1, ny, nx)),
            np.float32),
        exner=f3(ex1[:, None, None]), dz=f3(dz1[:, None, None]),
        psfc=f2(np.full((ny, nx), 100000.0)),
        znt=f2(np.full((ny, nx), 0.10)), ust=f2(ust), hfx=f2(hfx),
        qfx=f2(qfx), wspd=f2(wspd), br=f2(br),
        psim=f2(np.full((ny, nx), 6.5)),
        psih=f2(np.full((ny, nx), 8.5)),
        xland=f2(np.where(np.indices((ny, nx))[1] == nx - 1, 2.0, 1.0)),
        u10=f2(u[0]), v10=f2(v[0]), dt=45.0,
    )


def test_bl_pbl_config_default_and_toml(tmp_path):
    base = dict(nx=8, ny=2, nz=12, dx=1000.0, dy=1000.0, ztop=6000.0,
                dt=10.0, run_seconds=60.0)
    assert RunConfig(**base).bl_pbl_physics == 0
    assert RunConfig(**base).ysu_topdown_pblmix == 1
    cfg_file = tmp_path / "ysu.toml"
    cfg_file.write_text(
        "[grid]\nnx=8\nny=2\nnz=12\ndx=1000.0\ndy=1000.0\n"
        "ztop=6000.0\n[run]\ndt=10.0\nrun_seconds=60.0\n"
        "[dynamics]\nbl_pbl_physics=1\nysu_topdown_pblmix=0\n"
        # YSU reads UST/HFX/QFX/WSPD/RMOL from the surface layer, so
        # validate_run_config refuses bl_pbl_physics without one -- the same
        # pair woof/core/physics.py initialize_physics has always refused at
        # driver construction.  91 is classic MM5.
        "sf_sfclay_physics=91\n")
    assert load_config(cfg_file).bl_pbl_physics == 1
    assert load_config(cfg_file).ysu_topdown_pblmix == 0
    # 6 (MYNN3) is a WRF selector with no port; 7 served here until the
    # ACM2 port made it a valid value.
    cfg_file.write_text(cfg_file.read_text().replace("bl_pbl_physics=1",
                                                      "bl_pbl_physics=6"))
    with pytest.raises(ValueError, match="bl_pbl_physics"):
        load_config(cfg_file)


@pytest.mark.gpu
@requires_gpu
def test_ysu_topdown_cloud_radiative_mixing_f90_fixture():
    """Hand fixture for v4.6.1 bl_ysu.F90:839-897 and 943-964."""
    import cupy as cp

    from woof.core import constants as c
    from woof.core.ysu import launch_ysu
    from woof.verify.npref import np_ysu_column

    z, _, _, _, _ = _grid()
    theta = np.where(z < 900.0, 298.0 + 0.001 * z,
                     300.0 + 0.004 * (z - 900.0))
    args = _column(theta=theta, hfx=180.0, qfx=6.0e-5,
                   ust=0.40, br=-0.06, dt=60.0)
    cloud = (z > 400.0) & (z < 1800.0)
    args["qc"][cloud] = 2.0e-4
    args["rthraten"] = np.where(cloud, -2.0e-5, 0.0)

    ref = np_ysu_column(**args, ysu_topdown_pblmix=1)
    off = np_ysu_column(**args, ysu_topdown_pblmix=0)
    assert ref["cloudflg"] and not off["cloudflg"]
    assert off["topdown_radsum"] == 0.0 == off["wstar3_2"]

    # F90's radsum loop converts negative theta heating to W m-2 over
    # levels 1:kpbl-1.  Rebuild it independently from the fixture.
    kt = ref["kpbl"] - 2
    expected_radsum = 0.0
    for k in range(kt + 1):
        radflux = (args["rthraten"][k] * args["exner"][k] * c.CP / c.G
                   * (args["p_interface"][k]
                      - args["p_interface"][k + 1]))
        if radflux < 0.0:
            expected_radsum += abs(radflux)
    rho_top = (args["p"][kt]
               / (c.RD * args["theta"][kt] * args["exner"][kt]
                  * (1.0 + (c.RVOVRD - 1.0) * args["qv"][kt])))
    expected_wstar3_2 = (c.G / (args["theta"][kt]
                                      * (1.0 + (c.RVOVRD - 1.0)
                                         * args["qv"][kt]))
                         * (expected_radsum / rho_top / c.CP)
                         * ref["hpbl"])
    assert ref["topdown_radsum"] == pytest.approx(expected_radsum, rel=1e-13)
    assert ref["wstar3_2"] == pytest.approx(expected_wstar3_2, rel=1e-13)
    assert ref["topdown_radsum"] == pytest.approx(11.398764015571423)
    assert ref["wstar3_2"] == pytest.approx(0.28175264114865023)

    # Independent sign gate: radiative top-down mixing warms the mixed side
    # of the inversion and increases cooling immediately above it.  A shared
    # kernel/mirror sign error must not pass on agreement alone.
    kt = ref["kpbl"] - 2
    assert ref["dtheta"][kt] > off["dtheta"][kt]
    assert ref["dtheta"][kt + 1] < off["dtheta"][kt + 1]

    nz = args["theta"].size
    col3 = {name: cp.asarray(value[:, None, None], dtype=cp.float32)
            for name, value in args.items()
            if isinstance(value, np.ndarray) and value.shape == (nz,)}
    col3["p_interface"] = cp.asarray(
        args["p_interface"][:, None, None], dtype=cp.float32)
    surf = {name: cp.asarray([[args[name]]], dtype=cp.float32)
            for name in ("psfc", "znt", "ust", "hfx", "qfx", "wspd",
                         "br", "psim", "psih", "xland", "u10", "v10")}
    got = launch_ysu(
        **{name: col3[name] for name in
           ("u", "v", "theta", "qv", "qc", "qi", "p", "p_interface",
            "exner", "dz", "rthraten")},
        **surf, dt=args["dt"], ysu_topdown_pblmix=1)
    assert int(got["cloudflg"][0, 0]) == 1
    assert float(got["topdown_radsum"][0, 0]) == pytest.approx(
        expected_radsum, rel=3e-5)
    assert float(got["wstar3_2"][0, 0]) == pytest.approx(
        expected_wstar3_2, rel=3e-5)
    np.testing.assert_allclose(cp.asnumpy(got["dtheta"])[:, 0, 0],
                               ref["dtheta"], rtol=5e-4, atol=5e-7)


@pytest.mark.gpu
@requires_gpu
def test_ysu_topdown_revives_stable_surface_stratocumulus_pbl():
    """F90:732-768 revives a nocturnal liquid-unstable cloud layer."""
    import cupy as cp

    from woof.core.ysu import launch_ysu
    from woof.verify.npref import np_ysu_column

    nz, ztop = 24, 1500.0
    z, _, _, _, _ = _grid(nz, ztop)
    theta = 289.0 + 0.0015 * z
    args = _column(theta=theta, nz=nz, ztop=ztop, hfx=-10.0,
                   qfx=0.0, ust=0.25, br=0.05, dt=60.0)
    cloud = (z >= 350.0) & (z <= 800.0)
    args["qc"][cloud] = 6.0e-4
    args["rthraten"] = np.where(cloud, -2.0e-5, 0.0)

    ref = np_ysu_column(**args, ysu_topdown_pblmix=1)
    assert ref["kpbl"] == 14
    assert 780.0 < ref["hpbl"] < 820.0
    assert ref["cloudflg"]
    assert ref["topdown_radsum"] > 0.0
    assert ref["wstar3_2"] > 0.0

    col3 = {name: cp.asarray(value[:, None, None], dtype=cp.float32)
            for name, value in args.items()
            if isinstance(value, np.ndarray) and value.shape == (nz,)}
    col3["p_interface"] = cp.asarray(
        args["p_interface"][:, None, None], dtype=cp.float32)
    surf = {name: cp.asarray([[args[name]]], dtype=cp.float32)
            for name in ("psfc", "znt", "ust", "hfx", "qfx", "wspd",
                         "br", "psim", "psih", "xland", "u10", "v10")}
    got = launch_ysu(
        **{name: col3[name] for name in
           ("u", "v", "theta", "qv", "qc", "qi", "p", "p_interface",
            "exner", "dz", "rthraten")},
        **surf, dt=args["dt"], ysu_topdown_pblmix=1)
    assert int(got["kpbl"][0, 0]) == 14
    assert int(got["cloudflg"][0, 0]) == 1
    assert float(got["topdown_radsum"][0, 0]) > 0.0
    np.testing.assert_allclose(cp.asnumpy(got["dtheta"])[:, 0, 0],
                               ref["dtheta"], rtol=5e-4, atol=5e-7)


def test_a_calm_column_runs_the_whole_scheme_as_wrf_does():
    """``ust = hfx = qfx = 0`` is not a no-op: ``bl_ysu_run`` has no zero-flux
    exit, and on the WRF v4.6.1 oracle's case 12 it computes a 703 m PBL with
    nine levels in it.  The float64 mirror used to return zeros here (the
    short circuit kernels/ysu.cu also carried until golden/ports-pbl); it now
    reproduces WRF's diagnosis from WRF's own inputs.  CPU only -- the card's
    bitwise grade of the same column is
    test_ysu_wrf461_parity::test_a_calm_column_runs_wrfs_whole_scheme.
    """
    from woof.verify.npref import np_ysu_column
    from woof.verify.ysu_oracle import load_ysu_oracle

    fixture = load_ysu_oracle()
    i = fixture.cases.index(12)
    inputs = fixture.inputs
    for name in ("ust", "hfx", "qfx"):
        assert float(inputs[name][0, i]) == 0.0, name
    column = {name: inputs[name][:, 0, i].astype(np.float64)
              for name in ("u", "v", "theta", "qv", "qc", "qi", "p",
                           "exner", "dz", "rthraten")}
    surface = {name: float(inputs[name][0, i])
               for name in ("psfc", "znt", "ust", "hfx", "qfx", "wspd",
                            "br", "psim", "psih", "xland", "u10", "v10")}
    out = np_ysu_column(
        column["u"], column["v"], column["theta"], column["qv"],
        column["qc"], column["qi"], column["p"],
        inputs["p_interface"][:, 0, i].astype(np.float64), column["exner"],
        column["dz"], dt=fixture.dt, rthraten=column["rthraten"],
        ysu_topdown_pblmix=fixture.topdown[i], **surface)
    assert out["kpbl"] == int(fixture.kpbl_reference[0, i]) == 9
    # float64 mirror against WRF's float32 word: measured 6.3e-7 relative.
    assert out["hpbl"] == pytest.approx(
        float(fixture.surface_reference["hpbl"][0, i]), rel=1e-5)
    np.testing.assert_allclose(
        out["exch_m"], fixture.level_reference["exch_mx"][:, 0, i],
        rtol=1e-5, atol=0.0)
    assert np.any(np.asarray(out["du"]) != 0.0)
    for name in ("du", "dv", "dtheta", "dqv", "exch_h", "exch_m"):
        assert np.isfinite(np.asarray(out[name])).all(), name


def test_convective_surface_heating_grows_pbl_over_six_hours():
    """A capped morning layer deepens into a realistic daytime PBL."""
    from woof.verify.npref import np_ysu_column

    z, *_ = _grid()
    theta = np.where(z < 350.0, 298.0 + 0.001 * z,
                     300.4 + 0.004 * (z - 350.0))
    u = 6.0 + 0.001 * z
    qv = 0.011 * np.exp(-z / 2200.0)
    args = _column(theta=theta, u=u, qv=qv, hfx=240.0, qfx=7.0e-5,
                   ust=0.40, br=-0.08, dt=60.0)
    hpbl = []
    for _ in range(6 * 60):
        out = np_ysu_column(**args)
        hpbl.append(out["hpbl"])
        for field, tend in (("theta", "dtheta"), ("u", "du"),
                            ("v", "dv"), ("qv", "dqv"),
                            ("qc", "dqc"), ("qi", "dqi")):
            args[field] = args[field] + args["dt"] * out[tend]
        args["wspd"] = max(float(np.hypot(args["u"][0], args["v"][0])), 0.1)
    assert hpbl[0] < 800.0
    assert 900.0 <= hpbl[-1] <= 3500.0
    assert hpbl[-1] >= hpbl[0] + 500.0
    assert np.isfinite(args["theta"]).all()
    assert args["qv"].min() >= 0.0


def test_stable_boundary_layer_wind_decays():
    """Stable surface drag and local diffusion reduce low-level kinetic energy."""
    from woof.verify.npref import np_ysu_column

    z, *_ = _grid()
    theta = 296.0 + 0.009 * z
    u = 11.0 + 0.0015 * z
    v = 2.0 - 0.0003 * z
    args = _column(theta=theta, u=u, v=v, hfx=-35.0, qfx=0.0,
                   ust=0.30, br=0.14, dt=30.0)
    ke0 = np.mean(args["u"][:4] ** 2 + args["v"][:4] ** 2)
    hpbl = []
    for _ in range(3 * 120):
        out = np_ysu_column(**args)
        hpbl.append(out["hpbl"])
        for field, tend in (("theta", "dtheta"), ("u", "du"),
                            ("v", "dv"), ("qv", "dqv")):
            args[field] = args[field] + args["dt"] * out[tend]
        args["wspd"] = max(float(np.hypot(args["u"][0], args["v"][0])), 0.1)
    ke1 = np.mean(args["u"][:4] ** 2 + args["v"][:4] ** 2)
    assert ke1 < 0.92 * ke0
    assert 0.0 < hpbl[-1] < 1000.0
    assert np.isfinite(args["theta"]).all()


def test_the_thermal_enhanced_sweep_cannot_raise_pblflg():
    """bl_ysu.F90:703-728 is guarded ``if(pblflg(i))`` three times.

    Convective onset: weakly negative ``br``, a small positive ``hfx`` and a
    residual nocturnal cap on the first two levels.  The FIRST guess
    (:618-647) puts the PBL top at 95.4 m, below ``zq(i,2) = 166.7 m``, so
    :646 sets ``kpbl = 1`` and :647 sets ``pblflg = .false.``.  WRF then
    skips the thermal-enhanced Richardson sweep entirely -- :684-698 can only
    LOWER ``pblflg``, never raise it -- and the column spends the step in the
    local-K regime with no countergradient, no entrainment and no ``delta``.

    Ungated, the enhanced sweep runs anyway, the ``vpert`` excess punches
    through the cap and the column comes back ``kpbl = 4``, ``hpbl = 431 m``,
    ``delta = 10.2`` with an entrainment-boosted ``exch_h`` of 22 m2/s.  That
    is a different PBL onset time, not a last-ULP difference.

    ``ysu_topdown_pblmix = 0`` isolates the guard: WRF's theta-li extension
    (:732-751) carries no ``pblflg`` test and legitimately revives the column
    on the default path, which would mask the divergence rather than refute
    it.
    """
    from woof.verify.npref import np_ysu_column

    nz = 36
    z, *_ = _grid(nz)
    theta = np.empty(nz)
    theta[0] = 300.0
    for k in range(1, nz):
        theta[k] = theta[k - 1] + (0.1 if k <= 2 else 3.0)
    args = _column(theta=theta, qv=np.full(nz, 0.008), nz=nz,
                   hfx=60.0, qfx=2.0e-5, ust=0.30, br=-0.001, dt=60.0)
    args["ysu_topdown_pblmix"] = 0
    out = np_ysu_column(**args)

    # The first guess sat below zq(i,2); WRF leaves it there.
    assert out["kpbl"] == 1
    assert out["hpbl"] == pytest.approx(95.4339, abs=1.0e-3)
    assert out["hpbl"] < args["dz"][0]
    # pblflg false => no entrainment layer and background K only.
    assert out["delta"] == 0.0
    assert np.asarray(out["exch_h"]).max() == pytest.approx(0.01, abs=1e-9)


def test_a_theta_li_revival_that_leaves_kpbl_at_one_is_extinguished():
    """bl_ysu.F90:765 and :766 are two independent statements.

    The theta-li extension raises ``pblflg`` at :748 on an iteration that
    leaves ``definebrup`` set, so when that is the LAST iteration ``kpbl`` is
    never reassigned and the column reaches :755 with ``pblflg`` true and
    ``kpbl == 1``.  :764 then reads WRF's own ``za(i,0)`` overrun (inherited
    here on purpose, and unchanged by this test), and :766 -- which does NOT
    hang off :765's ``hpbl < zq(i,2)`` test -- kills ``pblflg`` regardless of
    what that junk interpolation produced.

    Fixture: stable surface (``br > 0``, so the first guess clamps to
    ``kpbl = 1`` and ``pblflg`` is already false), a 2 K surface inversion to
    keep the interpolated ``hpbl`` above ``zq(i,2)``, and 2 g/kg of liquid on
    level ``nz-2`` only, so the theta-li scan's single unstable level is its
    final one.

    With :766 nested inside :765 the column survives into :830 with
    ``k = kpbl(i) - 1 == 0`` -- one element before the column -- and reports
    ``delta = 12.0`` with an entrainment-boosted ``exch_h``.  In CUDA that
    same index is a negative offset into caller-owned device memory
    (``ysu.cu``'s ``theta[kt * st + col]`` at ``col - st``); in this float64
    mirror NumPy wraps it to the model top instead, so the mirror answers
    where the kernel would fault, and only the flag state separates them.
    """
    from woof.verify.npref import np_ysu_column

    nz, ztop = 36, 1800.0
    z, dz, *_ = _grid(nz, ztop)
    theta = np.empty(nz)
    theta[0] = 296.0
    theta[1] = 298.0
    theta[2:] = theta[1] + 0.001 * (z[2:] - z[1])
    qc = np.zeros(nz)
    qc[nz - 2] = 2.0e-3
    args = _column(theta=theta, u=np.full(nz, 11.0),
                   qv=np.full(nz, 0.008), nz=nz, ztop=ztop,
                   hfx=-35.0, qfx=0.0, ust=0.50, br=0.14, dt=30.0)
    args["qc"] = qc
    args["ysu_topdown_pblmix"] = 1
    out = np_ysu_column(**args)

    # The revival really happened and really left kpbl at 1 ...
    assert out["kpbl"] == 1
    # ... and WRF's za(i,0) interpolation is inherited unchanged, above
    # zq(i,2) = 50 m, so :765 does NOT fire and only :766 can extinguish it.
    assert out["hpbl"] == pytest.approx(113.3243, abs=1.0e-3)
    assert out["hpbl"] > dz[0]
    # :766 fired: no entrainment block, background K only.
    assert out["delta"] == 0.0
    assert np.asarray(out["exch_h"]).max() == pytest.approx(0.01, abs=1e-9)


@pytest.mark.parametrize("module", ("ysu", "ysu_480"))
def test_the_kernel_spells_wrfs_two_pblflg_rules_like_the_mirror(module):
    """The two branch structures the tests above measure, in each live unit.

    ``test_ysu_kernel_matches_float64_mirror`` is the numerical statement
    that the CUDA column and ``np_ysu_column`` are one transcription, and it
    needs a device: the fixture in ``woof/data/ysu/oracle/ysu-surface.csv``
    has no column with ``kpbl == 1`` (minimum 2, case 13), so neither of
    these two rules is reachable from the shipped oracle even on a GPU box.
    They decide whether a column is convective at all, so they are asserted
    here as well, textually, where the assertion runs everywhere.
    """
    from woof.core.kernels import _unit_text

    source = _unit_text(module)
    code = [line.strip() for line in source.splitlines()]
    code = [line for line in code if line and not line.startswith("//")]

    # bl_ysu.F90:703-728 -- every statement of the thermal-enhanced sweep
    # sits under if(pblflg(i)); :684-698 can only lower the flag.  What the
    # guarded block does with the sweep's result is a separate question
    # (bl_ysu.F90:765's clamp placement, par-pbl-ysu-03), so only the guard
    # itself is asserted here.
    start = code.index("if (sfcflg && sflux > 0.0f) {")
    end = code.index("} else pblflg = false;", start)
    block = code[start + 1:end]
    call = next(k for k, line in enumerate(block)
                if line.startswith("ysu_diagnose("))
    assert block[call - 1] == "if (pblflg) {"
    assert "pblflg = true" not in " ".join(block)

    # bl_ysu.F90:765 and :766 -- two independent statements.  Nesting the
    # second inside the first is what let a theta-li revival reach :833
    # with kpbl == 1 and index one level below the column.
    interp = code.index("hpbl = za[kh - 1] + frac * (za[kh] - za[kh - 1]);")
    assert code[interp + 1] == "if (hpbl < zq[1]) kpbl = 1;"
    assert code[interp + 2] == "if (kpbl <= 1) pblflg = false;"


@pytest.mark.gpu
@requires_gpu
def test_ysu_kernel_matches_float64_mirror():
    """The explicit historical engine path and its float64 mirror.

    The corrected default is graded against independent release Fortran in
    tools/ysu_generations_oracle, including the WRF driver round trips.
    """
    import cupy as cp
    from woof.core.ysu import launch_ysu
    from woof.verify.npref import np_ysu_column

    host = _batched_columns()
    dt = host.pop("dt")
    dev = {name: cp.asarray(value) for name, value in host.items()}
    got = launch_ysu(**dev, dt=dt, ysu_version="wrf_461")

    nz, ny, nx = host["theta"].shape
    ref = {name: np.empty((nz, ny, nx), np.float64)
           for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                        "exch_h", "exch_m")}
    ref.update({name: np.empty((ny, nx), np.float64)
                for name in ("hpbl", "kpbl", "wstar", "delta")})
    column_names = ("u", "v", "theta", "qv", "qc", "qi", "p",
                    "p_interface", "exner", "dz")
    surface_names = ("psfc", "znt", "ust", "hfx", "qfx", "wspd", "br",
                     "psim", "psih", "xland", "u10", "v10")
    for j in range(ny):
        for i in range(nx):
            kw = {name: host[name][:, j, i].astype(np.float64)
                  for name in column_names}
            kw.update({name: float(host[name][j, i]) for name in surface_names})
            col = np_ysu_column(**kw, dt=dt, ysu_version="wrf_461")
            for name in ref:
                if ref[name].ndim == 3:
                    ref[name][:, j, i] = col[name]
                else:
                    ref[name][j, i] = col[name]

    for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi"):
        np.testing.assert_allclose(cp.asnumpy(got[name]), ref[name],
                                   rtol=3e-4,
                                   atol=5e-7 if name == "dtheta" else 2e-7,
                                   err_msg=name)
    for name in ("exch_h", "exch_m"):
        np.testing.assert_allclose(cp.asnumpy(got[name]), ref[name],
                                   rtol=2e-3, atol=2e-3, err_msg=name)
    for name in ("hpbl", "wstar", "delta"):
        np.testing.assert_allclose(cp.asnumpy(got[name]), ref[name],
                                   rtol=3e-4, atol=3e-2, err_msg=name)
    np.testing.assert_array_equal(cp.asnumpy(got["kpbl"]), ref["kpbl"])
    assert (ref["exch_h"][:, :, :] >= 0.0).all()
    assert (ref["exch_m"][:, :, :] >= 0.0).all()


def _device_column(args):
    """Push one ``_column`` dict onto the device in launcher form."""
    import cupy as cp

    nz = args["theta"].size
    col = {name: cp.asarray(value[:, None, None], dtype=cp.float32)
           for name, value in args.items()
           if isinstance(value, np.ndarray) and value.shape == (nz,)}
    col["p_interface"] = cp.asarray(
        args["p_interface"][:, None, None], dtype=cp.float32)
    surf = {name: cp.asarray([[args[name]]], dtype=cp.float32)
            for name in ("psfc", "znt", "ust", "hfx", "qfx", "wspd",
                         "br", "psim", "psih", "xland", "u10", "v10")}
    return col, surf
def _single_column_device(args):
    """Reshape a ``_column`` dict into device ``(nz, 1, 1)`` launch inputs."""
    import cupy as cp

    f3 = lambda a: cp.asarray(np.ascontiguousarray(
        np.asarray(a, np.float32).reshape(-1, 1, 1)))
    f2 = lambda a: cp.asarray(np.full((1, 1), a, np.float32))
    dev = {name: f3(args[name])
           for name in ("u", "v", "theta", "qv", "qc", "qi", "p",
                        "p_interface", "exner", "dz")}
    dev.update({name: f2(args[name])
                for name in ("psfc", "znt", "ust", "hfx", "qfx", "wspd",
                             "br", "psim", "psih", "xland", "u10", "v10")})
    return dev


@pytest.mark.gpu
@requires_gpu
# Over WATER (2.8.9's form of this test, development branch) both generations
# keep momentum finite under a NaN HFX: vconv is 0 there, and the WRF 4.8
# kernel (graded word for word against WRF 4.8 by
# tools/ysu_generations_oracle/gate.py) passes HFX to no momentum term on a
# stable water column.  Measured golden/wrf-fixes on sm_120 and sm_89.  The
# YSU lane's ("wrf_480", "du") row was written for its LAND column.
@pytest.mark.parametrize("version,first_output", [("wrf_461", "dtheta"), ("wrf_480", "dtheta")])
def test_nonfinite_hfx_surfaces_as_dtheta_and_the_refusal_names_hfx(version, first_output):
    """The 1.5.2 field report, reproduced ON A STABLE SURFACE.

    Both generations must name the bad HFX.

    A user lost an ERA5-forced run on its FIRST step to ``YSU returned
    non-finite dtheta tendency``.  The name is a false lead.  ``dtheta``
    is written from the heat solve, whose ONLY surface term is HFX
    (kernels/ysu.cu:480 ``rhs[0] = theta0 - 300 + hf/(CP/G)/delp[0]*dt2``),
    so a non-finite HFX handed IN leaves as a dtheta produced OUT.

    THAT ASYMMETRY IS A PROPERTY OF THE STABLE BRANCH, and this fixture
    selects it deliberately (``br > 0`` -> ``sfcflg`` false at
    kernels/ysu.cu:165).  Under ``sfcflg`` the kernel takes
    ``wstar3 = wstar = 0`` as a literal (:177), so the surface buoyancy
    flux reaches nothing else and ``rhs[0]`` above is HFX's only route
    into the column.  Measured on the shipping kernel with entirely
    finite inputs, perturbing HFX 120 -> 60 and 120 -> 240 moves
    ``dtheta`` by 48% and 97% and moves ``du``, ``dqv`` and ``exch_m``
    by exactly 0.0000% -- max|delta| identically 0.  The claim is not
    approximately true here, it is bitwise true.

    It is FALSE on an unstable surface, where the buoyancy flux sets the
    convective velocity scale and therefore the mixed-layer diffusivity
    for momentum and moisture too; the same perturbation moves ``du`` by
    16% and 73% there.  That branch is pinned by the sibling test
    ``test_nonfinite_hfx_on_an_unstable_surface_poisons_the_column``.
    Before the sm_120 DAZ work this test passed with ``br=-0.20``, but
    only because ``ysu_max(NaN, 0)`` returned 0 and manufactured "zero
    surface buoyancy flux" out of "no answer"; the float64 authority
    (woof/verify/npref.py:6340, Python ``max``) never agreed.

    Part 1 reproduces it against the real kernel.  Part 2 pins the
    refusal to naming HFX, which is what the shipped message did not do.

    WATER, since development branch.  Every YSU column now takes WRF's default
    ctopo surface-drag arm (bl_ysu.F90:1308; module_bl_ysu.F:404 passes
    ctopo = 1 on every WRF run), whose drag weight vconvlim reads the surface
    buoyancy flux through the Beljaars vconv on LAND (bl_ysu.F90:1295-1298,
    fluxc = max(sflux, 0)).  So on a land column HFX reaches du and dv as it
    does in WRF, and a NaN HFX poisons them too.  Over water WRF sets
    vconv = 0, and the asymmetry this test is about holds as before.
    """
    import cupy as cp

    from woof.core.physics import validate_ysu_tendencies
    from woof.core.ysu import launch_ysu, validate_ysu_outputs

    # A nocturnal stable column over water: positive bulk Richardson number
    # with the downward heat flux that goes with it.
    args = _column(hfx=-35.0, qfx=1.0e-5, ust=0.20, br=0.20, xland=2.0,
                   dt=60.0)
    col, surf = _device_column(args)

    # -- part 1: the kernel reproduces the user's exact signature --------
    healthy = launch_ysu(**col, **surf, dt=args["dt"], ysu_version=version)
    assert all(bool(cp.isfinite(healthy[name]).all())
               for name in ("du", "dv", "dtheta")), "control must be clean"

    surf["hfx"][...] = cp.float32(cp.nan)
    out = launch_ysu(**col, **surf, dt=args["dt"], ysu_version=version)
    assert bool(cp.isfinite(out["du"]).all())
    assert bool(cp.isfinite(out["dv"]).all())
    assert not bool(cp.isfinite(out["dtheta"]).all())
    # Every OTHER output stays finite: this is what makes the report
    # identifiable rather than merely one blowup among many.
    for name in ("dqv", "dqc", "dqi", "exch_h", "exch_m", "hpbl",
                 "wstar", "delta", "topdown_radsum", "wstar3_2"):
        assert bool(cp.isfinite(out[name]).all()), name
    status = cp.zeros((1,), cp.uint32)
    assert validate_ysu_outputs(out, status) == first_output

    # -- part 2: the refusal names the degenerate PRODUCER ---------------
    producers = {
        "ust": surf["ust"], "hfx": surf["hfx"], "qfx": surf["qfx"],
        "wspd": surf["wspd"], "br": surf["br"], "znt": surf["znt"],
        "psim": surf["psim"], "psih": surf["psih"],
        "dz1": col["dz"][0], "p1": col["p_interface"][0],
        "dp1": col["p_interface"][0] - col["p_interface"][1],
    }
    with pytest.raises(FloatingPointError) as raised:
        validate_ysu_tendencies(out, status=status, grid_id=1,
                                producer_inputs=producers)
    message = str(raised.value)

    def claims(text):
        return (
            f"non-finite {first_output} tendency" in text,
            "on domain 1" in text,                    # where
            "already degenerate" in text,             # something came in bad
            "hfx (1 non-finite)" in text,             # WHICH one, with a count
        )

    assert claims(message) == (True, True, True, True), message
    # A NaN cell is counted once.  ``<= 0`` is False at NaN by IEEE rule.
    assert "hfx (1 non-finite, " not in message, message
    # Healthy producers are not accused.  ust/dz1/p1/dp1 are all positive
    # here, and wspd/br/qfx carry no sign requirement at all.
    for clean in ("ust", "qfx", "wspd", "br", "znt", "psim", "psih",
                  "dz1", "p1", "dp1"):
        assert f"{clean} (" not in message, (clean, message)

    # -- clean producers say so, rather than saying nothing --------------
    healthy["dtheta"][0, 0, 0] = cp.float32(cp.nan)
    producers["hfx"] = cp.asarray([[-35.0]], dtype=cp.float32)
    with pytest.raises(FloatingPointError) as raised_clean:
        validate_ysu_tendencies(healthy, status=status, grid_id=1,
                                producer_inputs=producers)
    clean_message = str(raised_clean.value)
    assert "scheme's own arithmetic produced it" in clean_message
    assert "already degenerate" not in clean_message

    # -- the pre-fix message is exactly what the report showed -----------
    # Without producer inputs the refusal is byte-identical to 1.5.2's,
    # so every existing caller and test keeps its contract -- and this
    # line is the control proving the new detail is what closes the gap.
    with pytest.raises(FloatingPointError) as raised_bare:
        validate_ysu_tendencies(out, status=status)
    assert (str(raised_bare.value)
            == f"YSU returned non-finite {first_output} tendency")

    # -- MUTATION CONTROLS, each watched failing -------------------------
    def mutant_output_only(_):
        return "YSU returned non-finite dtheta tendency"

    def mutant_no_input_name(_):
        return ("YSU returned non-finite dtheta tendency on domain 1"
                "; producer inputs already degenerate")

    def mutant_always_blames(_):
        return ("YSU returned non-finite dtheta tendency on domain 1"
                "; producer inputs already degenerate: ust (1 <= 0), "
                "hfx (1 non-finite)")

    assert claims(mutant_output_only(message)) != (True, True, True, True)
    assert claims(mutant_no_input_name(message)) != (True, True, True, True)
    assert "ust (" in mutant_always_blames(message)


@pytest.mark.gpu
@requires_gpu
def test_nonfinite_hfx_on_an_unstable_surface_poisons_the_column():
    """The other branch: HFX reaches momentum and moisture through w*.

    On an unstable surface (``br <= 0`` -> ``sfcflg`` true,
    kernels/ysu.cu:165) the surface buoyancy flux is not confined to the
    heat solve.  It sets the convective velocity scale --
    ``wstar3 = govrth * ysu_max(sflux, 0) * hpbl`` at :173 -- which sets
    ``wsk`` (:373), which sets ``km`` (:424), which sets the MOMENTUM
    eddy diffusivity ``xkzm`` (:433) that the u and v solves read at
    :560/:570/:586.  So "momentum never reads HFX" is not true of YSU
    here, and this is a fact about the SCHEME, not about NaN semantics:
    with entirely finite inputs, perturbing HFX 120 -> 60 and 120 -> 240
    on this fixture moves ``du`` by 16.3% and 73.4%, ``dqv`` by 45.0%
    and 96.8%, and ``exch_m`` by 40.1% and 109.4%.

    So a non-finite HFX must poison the whole column, and this test pins
    that it does.  Before the sm_120 DAZ work it did not: ``ysu_max``
    was ``a > b ? a : b``, which returns ``b`` at a NaN first argument,
    so ``ysu_max(NaN, 0)`` produced a plausible "zero surface buoyancy
    flux" and the column came back looking stable-branch clean.  That is
    the same laundering class as the exch_h = 1000 m2/s pin: a
    physically meaningful value manufactured from no answer.  WRF's
    ``amax1`` at a NaN argument is processor-dependent, i.e. undefined,
    and the defined answer for max(invalid, 0) is invalid.

    The change is strictly LOUDER, never quieter, which is what keeps
    the user's original bug from returning by this route:
    ``validate_ysu_outputs`` still fires, and the producer forensics
    still names ``hfx``.  Only the output LABEL moves, from ``dtheta``
    to ``du``, because ``du`` is first in launcher order.

    If a future change re-introduces a bound on ``ysu_max(sflux, 0)``
    this test goes red, and that is intended: the bound would be
    fabricating surface buoyancy the surface layer never reported.
    """
    import cupy as cp

    from woof.core.physics import validate_ysu_tendencies
    from woof.core.ysu import launch_ysu, validate_ysu_outputs
    from woof.verify.npref import np_ysu_column

    args = _column(hfx=120.0, qfx=8.0e-5, ust=0.30, br=-0.20, dt=60.0)
    col, surf = _device_column(args)

    healthy = launch_ysu(**col, **surf, dt=args["dt"])
    assert all(bool(cp.isfinite(healthy[name]).all())
               for name in ("du", "dv", "dtheta", "dqv", "exch_m")), \
        "control must be clean"

    surf["hfx"][...] = cp.float32(cp.nan)
    out = launch_ysu(**col, **surf, dt=args["dt"])

    # Everything the buoyancy flux reaches goes non-finite.  This is the
    # list the float64 authority produces for the same fixture, so the
    # kernel is held to its own mirror rather than to a hand-written
    # expectation.
    poisoned = ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                "exch_h", "exch_m", "wstar")
    for name in poisoned:
        assert not bool(cp.isfinite(out[name]).all()), name
    ref_args = dict(args)
    ref_args["hfx"] = float("nan")
    ref_dt = ref_args.pop("dt")
    mirror = np_ysu_column(**ref_args, dt=ref_dt)
    mirror_bad = sorted(
        name for name, value in mirror.items()
        if np.asarray(value, float).dtype.kind == "f"
        and not np.isfinite(np.asarray(value, float)).all())
    assert mirror_bad == sorted(poisoned), \
        f"kernel must agree with npref, its authority: {mirror_bad}"

    # hpbl and the geometry survive: they are diagnosed before the flux
    # enters, so the refusal still has a column to describe.
    for name in ("hpbl", "delta"):
        assert bool(cp.isfinite(out[name]).all()), name

    # The refusal is LOUDER, not quieter.  The label moves to du, but
    # the diagnosis -- which producer was already bad -- is unchanged,
    # and that is the half the user needed.
    status = cp.zeros((1,), cp.uint32)
    assert validate_ysu_outputs(out, status) == "du"
    producers = {
        "ust": surf["ust"], "hfx": surf["hfx"], "qfx": surf["qfx"],
        "wspd": surf["wspd"], "br": surf["br"], "znt": surf["znt"],
        "psim": surf["psim"], "psih": surf["psih"],
        "dz1": col["dz"][0], "p1": col["p_interface"][0],
        "dp1": col["p_interface"][0] - col["p_interface"][1],
    }
    with pytest.raises(FloatingPointError) as raised:
        validate_ysu_tendencies(out, status=status, grid_id=1,
                                producer_inputs=producers)
    message = str(raised.value)
    assert "non-finite du tendency" in message, message
    assert "on domain 1" in message, message
    assert "producer inputs already degenerate" in message, message
    assert "hfx (1 non-finite)" in message, message
    for clean in ("ust", "qfx", "wspd", "br", "znt", "psim", "psih",
                  "dz1", "p1", "dp1"):
        assert f"{clean} (" not in message, (clean, message)
def test_daz_flushed_ust_cube_keeps_exch_h_physical():
    """ust=1e-13 through the real kernel: exch_h physical, not the 1000 pin.

    sm_120 flushes FP32 subnormals in all arithmetic, so ``us*us*us`` is
    exactly 0 for ust=1e-13 (a *normal* float32) and WRF's prfac2 quotient
    (bl_ysu.F90:948) became Inf/Inf = NaN.  Pre-guard, ``ysu_min`` then
    laundered the NaN into ``xkzh = xkzmax``: exch_h pinned at 1000 m2/s
    where the float64 authority says ~131, and validate_ysu_outputs saw
    nothing.  This test drives the shipping kernel, not a mock, and holds
    both halves of the fix: the value is the mirror's, and nothing anywhere
    in the outputs is non-finite or riding the xkzmax rail.
    """
    import cupy as cp
    from woof.core.ysu import launch_ysu, validate_ysu_outputs
    from woof.verify.npref import np_ysu_column

    args = _column(hfx=240.0, qfx=7.0e-5, ust=1.0e-13, br=-0.08, dt=60.0)
    ref = np_ysu_column(**args)
    out = launch_ysu(**_single_column_device(args), dt=args["dt"])
    assert validate_ysu_outputs(out, cp.zeros(1, cp.uint32)) is None

    exch_h = out["exch_h"].get()[:, 0, 0]
    exch_m = out["exch_m"].get()[:, 0, 0]
    assert np.isfinite(exch_h).all() and np.isfinite(exch_m).all()
    # The laundered failure mode is exactly the xkzmax ceiling.
    assert not np.any(exch_h == np.float32(1000.0))
    assert not np.any(exch_m == np.float32(1000.0))
    np.testing.assert_allclose(exch_h.max(), np.asarray(ref["exch_h"]).max(),
                               rtol=1.0e-3)
    np.testing.assert_allclose(
        exch_h, np.asarray(ref["exch_h"]).reshape(-1), rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@requires_gpu
def test_ysu_producer_input_view_binds_the_fields_the_kernel_reads():
    """The lazy view must name WRF's fm/fh bindings and the layer mass."""
    import cupy as cp

    from woof.core.physics import _YsuProducerInputs

    fields = {name: cp.full((2, 3), 0.5, cp.float32)
              for name in ("ust", "hfx", "qfx", "wspd", "br", "znt")}
    fields["fm"] = cp.full((2, 3), 7.0, cp.float32)
    fields["fh"] = cp.full((2, 3), 9.0, cp.float32)
    atmosphere = {
        "dz": cp.full((4, 2, 3), 30.0, cp.float32),
        "p_interface": cp.asarray(
            np.broadcast_to(
                np.array([1.0e5, 9.7e4, 9.4e4, 9.1e4, 8.8e4])[:, None, None],
                (5, 2, 3)).copy(), dtype=cp.float32),
    }
    view = _YsuProducerInputs(fields, atmosphere)
    assert set(view) == {"ust", "hfx", "qfx", "wspd", "br", "znt", "psim",
                         "psih", "dz1", "p1", "dp1"}
    # YSU's PSIM/PSIH dummies are WRF's full similarity denominators.
    assert float(view["psim"][0, 0]) == 7.0
    assert float(view["psih"][0, 0]) == 9.0
    assert float(view["dz1"][0, 0]) == 30.0
    assert float(view["p1"][0, 0]) == pytest.approx(1.0e5)
    assert float(view["dp1"][0, 0]) == pytest.approx(3.0e3)
    assert view["dp1"].shape == (2, 3)
    with pytest.raises(KeyError):
        view["tsk"]
def test_exactly_zero_ust_with_live_fluxes_is_defined():
    """ust == 0 with nonzero heat flux: WRF's prfac2 is then 0/0 (the parity suite's case-13 pathology).  The
    kernel and the float64 mirror now both take the defined algebraic limit
    instead of laundering or propagating a NaN."""
    import cupy as cp
    from woof.core.ysu import launch_ysu, validate_ysu_outputs
    from woof.verify.npref import np_ysu_column

    args = _column(hfx=240.0, qfx=7.0e-5, ust=0.0, br=-0.08, dt=60.0)
    ref = np_ysu_column(**args)
    out = launch_ysu(**_single_column_device(args), dt=args["dt"])
    assert validate_ysu_outputs(out, cp.zeros(1, cp.uint32)) is None
    exch_h = out["exch_h"].get()[:, 0, 0]
    assert np.isfinite(exch_h).all()
    assert np.isfinite(np.asarray(ref["exch_h"])).all()
    assert not np.any(exch_h == np.float32(1000.0))
    np.testing.assert_allclose(
        exch_h, np.asarray(ref["exch_h"]).reshape(-1), rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@requires_gpu
def test_ysu_min_max_propagate_nan_and_keep_finite_semantics():
    """Device-level pin on the real compiled helpers, not a re-implementation.

    ``ysu_min(NaN, xkzmax)`` returning xkzmax is how a poisoned diffusivity
    became a plausible 1000 m2/s.  The helpers now return the NaN, so it
    reaches validate_ysu_outputs; and for every non-NaN first argument they
    are bit-identical to the old ``a < b ? a : b`` (ties and signed zeros
    still select b).  Compiled from the same module source the model runs.
    """
    import cupy as cp
    from woof.core.kernels import module_source

    probe = """
extern "C" __global__ void ysu_minmax_probe(
    const real *a, const real *b, real *mn, real *mx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mn[i] = ysu_min(a[i], b[i]);
    mx[i] = ysu_max(a[i], b[i]);
}
"""
    mod = cp.RawModule(code=module_source("ysu") + probe,
                       options=("-std=c++17",))
    kernel = mod.get_function("ysu_minmax_probe")
    nan = np.float32(np.nan)
    pairs = np.array([
        (nan, 1000.0), (1000.0, nan), (-0.0, 0.0), (0.0, -0.0),
        (3.0, 7.0), (7.0, 3.0), (5.0, 5.0),
    ], dtype=np.float32)
    a = cp.asarray(np.ascontiguousarray(pairs[:, 0]))
    b = cp.asarray(np.ascontiguousarray(pairs[:, 1]))
    mn = cp.empty(len(pairs), cp.float32)
    mx = cp.empty(len(pairs), cp.float32)
    kernel((1,), (len(pairs),), (a, b, mn, mx, np.int32(len(pairs))))
    mn, mx = mn.get(), mx.get()

    # The laundering pin: a NaN in either slot comes back out.
    assert np.isnan(mn[0]) and np.isnan(mx[0])
    assert np.isnan(mn[1]) and np.isnan(mx[1])
    # Old finite semantics, including b-selection on ties and signed zeros.
    view = lambda x: np.asarray(x, np.float32).view(np.uint32)
    assert view(mn[2]) == view(np.float32(0.0))     # min(-0, +0) -> b
    assert view(mn[3]) == view(np.float32(-0.0))    # min(+0, -0) -> b
    assert view(mx[2]) == view(np.float32(0.0))
    assert view(mx[3]) == view(np.float32(-0.0))
    assert mn[4] == 3.0 and mn[5] == 3.0 and mn[6] == 5.0
    assert mx[4] == 7.0 and mx[5] == 7.0 and mx[6] == 5.0
