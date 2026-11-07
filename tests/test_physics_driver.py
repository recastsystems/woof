"""WRF-ordered non-timesplit physics driver and RK3 plumbing.

The driver runs radiation -> SFCLAY -> Noah -> YSU -> cumulus before RK3,
holds the resulting slow tendencies across all three stages, preserves the
land/water split, and exposes the physics diagnostics used by wrfout.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta

import numpy as np
import pytest
from _run_config_fake import run_config_fake

from conftest import requires_gpu
from woof.config import RunConfig, load_config


def _base_config(**overrides):
    values = dict(nx=6, ny=2, nz=16, dx=2000.0, dy=2000.0,
                  ztop=8000.0, dt=10.0, run_seconds=0.0,
                  time_step_sound=4, moist=True,
                  sf_sfclay_physics=1, sf_surface_physics=2,
                  bl_pbl_physics=1)
    values.update(overrides)
    return RunConfig(**values)


def _full_state(*, snow=0.0, snow_depth=0.0, radiation=None,
                 cumulus=None, radiation_start_time=None,
                 radiation_latitude=None, radiation_longitude=None,
                 landuse_dataset="MODIFIED_IGBP_MODIS_NOAH",
                 **cfg_overrides):
    """Small horizontally homogeneous moist state plus mixed land/water."""
    import cupy as cp

    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    cfg = _base_config(**cfg_overrides)
    coord = make_vertical_coord(cfg.nz)
    theta = lambda z: 298.0 + 0.004 * np.asarray(z, np.float64)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.010 * np.exp(-np.asarray(z, np.float64) / 2400.0))
    state.u[...] = cp.float32(6.0)
    state.v[...] = cp.float32(0.5)

    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    if cfg.nx > 1:
        landmask[:, -1] = 0.0
    tsk = np.full((cfg.ny, cfg.nx), 299.0)
    tsk[:, landmask[0] == 0.0] = 296.0
    soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.31)
    soil_m[:, landmask == 0.0] = 1.0
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m, ivgtyp=np.where(landmask, 10, 17),
        isltyp=np.where(landmask, 6, 14), vegfra=55.0, tmn=286.0,
        snow=snow, snow_depth=snow_depth,
        swdown=450.0, glw=310.0, pblh=700.0,
        radiation=radiation, cumulus=cumulus, landuse_dataset=landuse_dataset,
        radiation_start_time=radiation_start_time,
        radiation_latitude=radiation_latitude,
        radiation_longitude=radiation_longitude)
    return state, cfg, driver


def _cpu_wsm6_diagnostics_driver(monkeypatch, *, dt=60.0):
    """Small CPU mirror of the named WSM6 diagnostic handoff."""
    from types import SimpleNamespace

    import woof.core.physics as physics
    from woof.core.microphysics import MicrophysicsDiagnostics

    monkeypatch.setattr(physics, "cp", np)
    shape = (2, 3)
    zeros = lambda: np.zeros(shape, np.float32)
    driver = object.__new__(physics.PhysicsDriver)
    driver.state = SimpleNamespace(mup=zeros())
    driver.mp_physics = 6
    (driver._sr_roundoff_upper, driver._sr_roundoff_max_ulps,
     driver._wsm6_minor_loops) = physics._sr_roundoff_envelope(6, dt)
    driver.microphysics = MicrophysicsDiagnostics(
        rainnc=zeros(), rainncv=zeros(), sr=zeros(),
        snownc=zeros(), snowncv=zeros(),
        graupelnc=zeros(), graupelncv=zeros())
    driver.microphysics_updates = 0
    driver._pending_rainbl = zeros()
    driver.ruc_params = None
    driver.noahmp_params = None
    return physics, driver, MicrophysicsDiagnostics, shape


def test_wsm6_sr_accepts_only_proven_wrf_fp32_roundoff_without_mutation(
        monkeypatch):
    physics, driver, diagnostics, shape = _cpu_wsm6_diagnostics_driver(
        monkeypatch, dt=60.0)
    assert driver._wsm6_minor_loops == 1
    assert driver._sr_roundoff_max_ulps == 7
    upper_bits = np.asarray(
        driver._sr_roundoff_upper, np.float32).view(np.uint32).item()
    assert upper_bits - np.asarray(np.float32(1.0)).view(np.uint32).item() == 7

    one_ulp = np.nextafter(
        np.float32(1.0), np.float32(np.inf), dtype=np.float32)
    sr = np.full(shape, driver._sr_roundoff_upper, np.float32)
    driver.accept_microphysics(diagnostics(
        rainnc=np.ones(shape, np.float32),
        rainncv=np.ones(shape, np.float32), sr=sr))
    # Acceptance must not hide the WRF-produced bit pattern by clipping it.
    np.testing.assert_array_equal(driver.microphysics.sr, sr)
    assert driver.microphysics.sr.view(np.uint32)[0, 0] == upper_bits
    assert driver.microphysics.sr[0, 0] >= one_ulp

    beyond = np.asarray(upper_bits + 1, np.uint32).view(np.float32)
    with pytest.raises(ValueError, match="roundoff_ulps=7"):
        driver.accept_microphysics(diagnostics(
            rainnc=np.ones(shape, np.float32),
            rainncv=np.ones(shape, np.float32),
            sr=np.full(shape, beyond, np.float32)))
    with pytest.raises(ValueError, match="validated range"):
        driver.accept_microphysics(diagnostics(
            rainnc=np.ones(shape, np.float32),
            rainncv=np.ones(shape, np.float32),
            sr=np.full(shape, -np.float32(1.0e-7), np.float32)))
    with pytest.raises(FloatingPointError, match="non-finite"):
        driver.accept_microphysics(diagnostics(
            rainnc=np.ones(shape, np.float32),
            rainncv=np.ones(shape, np.float32),
            sr=np.full(shape, np.nan, np.float32)))


@pytest.mark.parametrize("mp_physics", [6, 16, 24, 26])
def test_wsm6_family_sr_envelope_follows_the_step_it_is_handed(
        monkeypatch, mp_physics):
    """A driver built on a one-loop step, then run on a two-loop step."""
    physics, driver, diagnostics, shape = _cpu_wsm6_diagnostics_driver(
        monkeypatch, dt=60.0)
    driver.mp_physics = mp_physics
    (driver._sr_roundoff_upper, driver._sr_roundoff_max_ulps,
     driver._wsm6_minor_loops) = physics._sr_roundoff_envelope(
        mp_physics, 60.0)
    one_loop = physics._sr_roundoff_envelope(mp_physics, 60.0)
    two_loops = physics._sr_roundoff_envelope(mp_physics, 240.0)
    assert two_loops[0] > one_loop[0]

    def result(upper):
        return diagnostics(
            rainnc=np.ones(shape, np.float32),
            rainncv=np.ones(shape, np.float32),
            sr=np.full(shape, upper, np.float32))

    # The constructor's envelope refuses what two minor loops can produce.
    with pytest.raises(ValueError, match="validated range"):
        driver.accept_microphysics(result(two_loops[0]))
    driver.accept_microphysics(result(two_loops[0]), dt=240.0)
    assert (driver._sr_roundoff_upper, driver._sr_roundoff_max_ulps,
            driver._wsm6_minor_loops) == two_loops
    # And it tightens again when the step comes back down.
    with pytest.raises(ValueError, match="validated range"):
        driver.accept_microphysics(result(two_loops[0]), dt=60.0)
    assert driver._wsm6_minor_loops == one_loop[2]
    driver.accept_microphysics(result(one_loop[0]), dt=90.0)
    assert driver._wsm6_minor_loops == 1


def test_wsm6_sr_roundoff_envelope_scales_with_minor_loop_count():
    from fractions import Fraction

    from woof.core.physics import _wsm6_sr_roundoff_limit

    one_loop = _wsm6_sr_roundoff_limit(60.0)
    two_loops = _wsm6_sr_roundoff_limit(240.0)
    assert one_loop[1:] == (3, 1)
    assert two_loops[1:] == (4, 2)
    assert two_loops[0] > one_loop[0] > np.float32(1.0)
    scale = 1 << 24
    for _upper, max_ulps, loops in (one_loop, two_loops):
        adds = loops - 1
        analytic = Fraction(
            (scale + 1) ** 3 * (scale - 3),
            scale ** 2 * (scale - 6) * (scale - 2 * adds))
        encoded = Fraction(scale + 2 * max_ulps, scale)
        next_encoded = Fraction(scale + 2 * (max_ulps + 1), scale)
        assert encoded <= analytic < next_encoded
    with pytest.raises(ValueError, match=r"ULP\(1\) linearity"):
        _wsm6_sr_roundoff_limit(600_000_000.0)


def test_physics_interval_config_defaults_roundtrip_and_validation(tmp_path):
    """Interval defaults roundtrip, and a negative cudt_minutes is refused.

    ``moist = true`` in the TOML is essential: 573939c moved the
    driver's refusal of a cumulus scheme on a dry state to ``load_config``
    (the same rule validate_run_config has always applied to mp_physics),
    and this test's original TOML -- ``cu_physics = 1`` with no moisture --
    was exactly the dead configuration that refusal exists for.  WRF
    v4.6.1 (d66e442) cannot even express it: ``Registry.EM_COMMON:3014``
    (``package passiveqv mp_physics==0 - moist:qv``) allocates QV in every
    run including microphysics-off ones, so no WRF namelist runs
    Kain-Fritsch without a QV array, while woof's ``moist=false`` means
    ``state.qv is None``.  (This test was missed by 573939c's repair sweep
    -- which made test_experiment.py's cudt leg gain the same key --
    because this module's helpers import cupy, which marks the whole file
    ``gpu`` and hid it from the CPU tallies that sweep was checked with.)
    """
    cfg = _base_config(sf_sfclay_physics=0, sf_surface_physics=0,
                       bl_pbl_physics=0)
    assert cfg.radt == 0.0
    assert cfg.bldt == 0.0
    assert cfg.ra_physics == 0
    assert cfg.cu_physics == 0
    assert cfg.radt_minutes == 12.0
    assert cfg.cudt_minutes == 5.0

    path = tmp_path / "physics.toml"
    path.write_text(
        "[grid]\nnx=6\nny=2\nnz=16\ndx=2000.0\ndy=2000.0\n"
        "ztop=8000.0\n[run]\ndt=10.0\nrun_seconds=60.0\n"
        "[dynamics]\nmoist=true\nra_physics=90\ncu_physics=1\n"
        "radt_minutes=12.0\ncudt_minutes=5.0\nbldt=5.0\n")
    loaded = load_config(path)
    assert loaded.ra_physics == 90
    assert loaded.cu_physics == 1
    assert loaded.radt_minutes == 12.0
    assert loaded.cudt_minutes == 5.0
    assert loaded.bldt == 5.0

    path.write_text(path.read_text().replace(
        "cudt_minutes=5.0", "cudt_minutes=-1.0"))
    with pytest.raises(ValueError, match="cudt_minutes"):
        load_config(path)


def _cpu_surface_driver(monkeypatch, *, sf_surface_physics):
    """Host driver fixture with real device SFCDIAGS for surface ordering."""
    from types import SimpleNamespace
    import woof.core.physics as physics

    monkeypatch.setattr(physics, "cp", np)
    # refresh_any accepts CuPy carriers. This synthetic driver uses host
    # arrays; move only the actual diagnostic call across that boundary.
    def host_sfcdiags(fields, qv1, **constants):
        import cupy as cp
        names = ("psfc", "tsk", "cqs2", "chs2", "qsfc", "qfx", "hfx",
                 "q2", "t2", "th2")
        device = {name: cp.asarray(fields[name]) for name in names}
        physics.noah_sfcdiags.refresh(device, cp.asarray(qv1), **constants)
        for name in ("q2", "t2", "th2"):
            fields[name][...] = cp.asnumpy(device[name])
    monkeypatch.setattr(physics.noah_sfcdiags, "refresh_any", host_sfcdiags)
    atmosphere = {
        "pressure": np.array([[[97500.0, 87500.0]]], np.float32),
        "p_interface": np.array(
            [[[100000.0, 90000.0]], [[95000.0, 85000.0]]], np.float32),
        # SFCDIAGS needs the lowest model level's vapour, which is what
        # WRF's own remedy for an out-of-range 2 m value publishes
        # (module_sf_noahdrv.F:1276-1282).
        "qv": np.array([[[9.0e-3, 7.0e-3]]], np.float32),
    }
    monkeypatch.setattr(
        physics, "_prepare_atmosphere", lambda state: atmosphere)
    state = SimpleNamespace(elapsed_seconds=0.0, p=atmosphere["pressure"],
                            c1h=np.ones(1, np.float32), c2h=np.zeros(1, np.float32),
                            total_mu=lambda: np.ones((1, 2), np.float32))
    cfg = run_config_fake(
        dt=60.0, bldt=0.0, ra_physics=0, sf_sfclay_physics=91,
        sf_surface_physics=sf_surface_physics, bl_pbl_physics=1,
        cu_physics=0, ra_rrtmg_variant="rte_rrtmgp")
    driver = object.__new__(physics.PhysicsDriver)
    driver.state = state
    driver._wrf_rthraten = None
    driver._wrf_rthraten_coupled = False
    driver.swint = None
    driver.fields = {
        name: np.full((1, 2), value, np.float32)
        for name, value in {
            "psfc": -123.0, "tsk": 290.0, "hfx": 0.0, "qfx": 0.0,
            "qsfc": 0.0, "chs2": 0.0, "cqs2": 0.0,
            "t2": -999.0, "q2": -999.0, "th2": -999.0,
            "lh": 0.0, "grdflx": 0.0, "achfx": 0.0, "aclhf": 0.0, "acgrdflx": 0.0,
        }.items()
    }
    driver.fields["p_hyd"] = np.zeros_like(state.p)
    driver.fields["xland"] = np.array([[1.0, 2.0]], np.float32)
    from woof.core.physics_inventory import PBL_HISTORY_RATES
    driver.fields.update({name: np.zeros_like(state.p) for name in PBL_HISTORY_RATES})
    driver.surface_enabled = True
    # No sub-radiation-step solar interpolation (PhysicsDriver.__init__ sets
    # swint = None unless swint_opt is on); this fixture skips __init__.
    driver.swint = None
    # No wind farm (PhysicsDriver.__init__ sets farms = None unless a
    # wind-farm scheme is selected).
    driver.farms = None
    # No TC surface bundle (PhysicsDriver.__init__ sets tc_surface from
    # woof.tc.surface.create_surface, None when tc_flux_curve and tc_spray
    # are off and tc_wave_roughness is unset -- the hurricane fold
    # ba5f12d375).
    driver.tc_surface = None
    # No fog settling (__init__ sets fog_settling = None unless the
    # grav_settling fog option is selected; c35dbafafb).
    driver.fog_settling = None
    # THE CARRIER CONTRACT, wired rather than dodged.  A hand-built driver
    # is exactly the construction path the consumption check exists to
    # cover, so this fixture goes through it like a real driver: both of
    # Noah's carriers are declared as the constants this fixture's stub
    # schemes assume.  Deleting either declaration (or the check itself)
    # turns every sf_surface_physics=2 test below red.
    driver.carriers = physics.CarrierContract()
    driver.carriers.declare(
        "glw", source=physics.CARRIER_SOURCE_DECLARED_CONSTANT)
    driver.carriers.declare(
        "swdown", source=physics.CARRIER_SOURCE_DECLARED_CONSTANT)
    if sf_surface_physics == 1:
        driver.carriers.declare(
            "gsw", source=physics.CARRIER_SOURCE_DECLARED_CONSTANT)
    driver.radt_seconds = 720.0
    driver.stepbl = 1
    driver.radt_minutes = 12.0
    driver.cudt_minutes = 5.0
    driver.call_counts = {
        "radiation": 0, "sfclay": 0, "noah": 0, "ysu": 0,
        "cumulus": 0, "cumulus_history": 0,
    }
    driver.tendencies = object()
    driver._compose_tendencies = lambda cfg: None
    return physics, state, cfg, driver, atmosphere


def test_surface_step_refreshes_psfc_before_sfclay_without_noah(monkeypatch):
    """WRF assigns PSFC before selecting any surface/LSM scheme."""
    _, state, cfg, driver, atmosphere = _cpu_surface_driver(
        monkeypatch, sf_surface_physics=0)
    seen = {}

    def sfclay(atmosphere_arg, cfg_arg):
        seen["psfc"] = driver.fields["psfc"].copy()

    driver._run_sfclay = sfclay
    driver._run_noah = lambda *_: pytest.fail("Noah must remain disabled")
    driver._run_ysu = lambda *_: None

    driver.compute(state, cfg)

    np.testing.assert_array_equal(
        seen["psfc"], atmosphere["p_interface"][0])
    assert driver.call_counts["noah"] == 0


@pytest.mark.parametrize("land_scheme", [1, 2])
# Gate successor: test_post_noah_surface_diagnostics_match_wrf_sfcdiags_before_ysu; the assertions cover the admitted behavior.
def test_post_lsm_surface_diagnostics_match_wrf_sfcdiags_before_ysu(
        monkeypatch, land_scheme):
    """Slab/Noah-mutated fluxes and skin feed WRF SFCDIAGS before PBL."""
    _, state, cfg, driver, atmosphere = _cpu_surface_driver(
        monkeypatch, sf_surface_physics=land_scheme)
    events = []
    seen = {}

    def sfclay(atmosphere_arg, cfg_arg):
        events.append("sfclay")

    def noah(atmosphere_arg, cfg_arg, itimestep_arg):
        events.append("noah")
        assert itimestep_arg == 1
        for name, values in {
            "tsk": [310.0, 270.0], "hfx": [155.0, 999.0],
            "qfx": [1.2e-3, 999.0], "qsfc": [0.018, 0.004],
            # Ordinary-land noahdrv.F:1275 overwrites CHS2=CQS2 before
            # SFCDIAGS refreshes T2.  Water remains on its skipped value.
            "chs2": [0.020, 0.0], "cqs2": [0.020, 0.0],
        }.items():
            driver.fields[name][0] = np.asarray(values, np.float32)

    def ysu(atmosphere_arg, cfg_arg):
        events.append("ysu")
        seen.update({name: driver.fields[name].copy()
                     for name in ("t2", "q2", "th2")})

    driver._run_sfclay = sfclay
    driver._run_noah = noah
    driver._run_slab = noah
    driver._run_ysu = ysu

    driver.compute(state, cfg)

    fields = driver.fields
    psfc = atmosphere["p_interface"][0]
    rho = psfc / (np.float32(287.0) * fields["tsk"])
    expected_q2 = fields["qsfc"].copy()
    expected_q2[0, 0] -= fields["qfx"][0, 0] / (
        rho[0, 0] * fields["cqs2"][0, 0])
    # This fixture's land column pairs a large evaporative flux with a tiny
    # exchange coefficient, so WRF's unbounded inversion
    # (module_sf_sfcdiags.F:56) lands outside the range a mixing ratio can
    # occupy.  The engine then publishes the lowest model level instead,
    # which is what WRF's own commented-out CQS2 = CHS remedy produces
    # (module_sf_noahdrv.F:1276-1282).  The water column keeps QSFC, and
    # T2/TH2 are untouched by the moisture guard.
    assert expected_q2[0, 0] < 0.0
    expected_q2[0, 0] = atmosphere["qv"][0][0, 0]
    expected_t2 = fields["tsk"].copy()
    expected_t2[0, 0] -= fields["hfx"][0, 0] / (
        rho[0, 0] * np.float32(1004.5) * fields["chs2"][0, 0])
    expected_th2 = expected_t2 * np.power(
        np.float32(1.0e5) / psfc, np.float32(287.0 / 1004.5))

    assert events == ["sfclay", "noah", "ysu"]
    np.testing.assert_allclose(seen["q2"], expected_q2, rtol=0.0,
                               atol=2.0e-7)
    np.testing.assert_allclose(seen["t2"], expected_t2, rtol=0.0,
                               atol=2.0e-5)
    np.testing.assert_allclose(seen["th2"], expected_th2, rtol=0.0,
                               atol=2.0e-5)


@pytest.mark.gpu
@requires_gpu
def test_ysu_tendencies_are_mass_coupled_and_staggered_like_wrf():
    import cupy as cp

    from woof.core.physics import couple_ysu_tendencies

    state, cfg, _ = _full_state(sf_surface_physics=0)
    nz, ny, nx = state.p.shape
    msft = np.full((ny, nx), 1.10)
    msfu = np.full((ny, nx + 1), 1.20)
    msfv = np.full((ny + 1, nx), 0.90)
    state.set_map_coriolis(msft=msft, msfu=msfu, msfv=msfv)
    k = cp.arange(1, nz + 1, dtype=cp.float32)[:, None, None]
    pattern = cp.arange(1, ny * nx + 1, dtype=cp.float32).reshape(1, ny, nx)
    ysu = {
        "du": k * pattern * cp.float32(1.0e-5),
        "dv": -k * pattern * cp.float32(2.0e-5),
        "dtheta": k * pattern * cp.float32(3.0e-6),
        "dqv": -k * pattern * cp.float32(4.0e-10),
        "dqc": k * pattern * cp.float32(5.0e-11),
        "dqi": -k * pattern * cp.float32(6.0e-12),
    }
    got = couple_ysu_tendencies(state, cfg, ysu)

    chm = (state.c1h[:, None, None] * state.total_mu()[None]
           + state.c2h[:, None, None])
    mass_u = chm * ysu["du"]
    mass_v = chm * ysu["dv"]
    u_faces = 0.5 * (mass_u + cp.roll(mass_u, 1, axis=2))
    u_faces = cp.concatenate([u_faces, u_faces[:, :, :1]], axis=2)
    v_faces = 0.5 * (mass_v + cp.roll(mass_v, 1, axis=1))
    v_faces = cp.concatenate([v_faces, v_faces[:, :1, :]], axis=1)

    from woof.wrf_exact import ENABLED as strict
    # Strict keeps the undivided phy_bl_ten slot; rk_addtend_dry divides
    # the complete held sum later. Default predivides this contribution.
    cp.testing.assert_array_equal(got.ru, u_faces if strict else u_faces / state.msfu[None])
    cp.testing.assert_array_equal(got.rv, v_faces if strict else v_faces / state.msfv[None])
    expected_theta = chm * ysu["dtheta"]
    cp.testing.assert_array_equal(got.rtheta, expected_theta if strict else expected_theta / state.msft[None])
    cp.testing.assert_array_equal(got.rqv, chm * ysu["dqv"])
    cp.testing.assert_array_equal(got.rqc, chm * ysu["dqc"])
    # RQIBLTEN couples like the other moist scalars (WRF phy_bl_ten
    # F_QI branch) and reaches the scalar update through scalar_for.
    cp.testing.assert_array_equal(got.rqi, chm * ysu["dqi"])
    assert got.scalar_for("qi") is got.rqi
    assert got.scalar_for("qs") is got.rqs
    # A scheme result without dqi (no ice moist set) couples to None.
    without = couple_ysu_tendencies(
        state, cfg, {name: ysu[name] for name in
                     ("du", "dv", "dtheta", "dqv", "dqc")})
    assert without.rqi is None and without.scalar_for("qi") is None
    assert without.rqs is None and without.scalar_for("qs") is None


@pytest.mark.gpu
@requires_gpu
def test_ysu_nan_guard_preserves_all_finite_rates_and_rejects_nonfinite():
    import cupy as cp

    from woof.core.physics import validate_ysu_tendencies

    finite = {
        "du": cp.full((4, 3, 4), 8.0, cp.float32),
        "dv": cp.full((4, 3, 4), -4.0, cp.float32),
        "dtheta": cp.full((4, 3, 4), 18.0, cp.float32),
        "dqv": cp.full((4, 3, 4), 2.0e-4, cp.float32),
        "dqc": cp.full((4, 3, 4), -3.0e-4, cp.float32),
        "hpbl": cp.ones((3, 4), cp.float32),
    }
    before = {name: value.copy() for name, value in finite.items()}
    validate_ysu_tendencies(finite)
    for name in finite:
        cp.testing.assert_array_equal(finite[name], before[name])

    nonfinite = {name: value.copy() for name, value in finite.items()}
    nonfinite["du"][0, 0, 0] = cp.nan
    with pytest.raises(FloatingPointError, match="non-finite du"):
        validate_ysu_tendencies(nonfinite)

    native = {
        name: cp.ones((4, 3, 4), cp.float32)
        for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                     "exch_h", "exch_m")
    }
    native.update(
        hpbl=cp.ones((3, 4), cp.float32),
        kpbl=cp.ones((3, 4), cp.int32),
        wstar=cp.ones((3, 4), cp.float32),
        delta=cp.ones((3, 4), cp.float32),
        topdown_radsum=cp.ones((3, 4), cp.float32),
        wstar3_2=cp.ones((3, 4), cp.float32),
        cloudflg=cp.ones((3, 4), cp.int32))
    status = cp.full((1,), 0xFFFFFFFF, cp.uint32)
    before = {name: value.copy() for name, value in native.items()}
    validate_ysu_tendencies(native, status=status)
    assert int(status[0].item()) == 0
    for name in native:
        cp.testing.assert_array_equal(native[name], before[name])

    native["hpbl"][0, 0] = cp.inf
    native["dv"][0, 0, 0] = cp.nan
    with pytest.raises(FloatingPointError, match="non-finite dv"):
        validate_ysu_tendencies(native, status=status)

    native["hpbl"][0, 0] = cp.float32(1.0)
    native["dv"][0, 0, 0] = cp.float32(1.0)
    for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                 "exch_h", "exch_m", "hpbl", "wstar", "delta",
                 "topdown_radsum", "wstar3_2"):
        native[name].flat[0] = cp.nan
        with pytest.raises(
                FloatingPointError, match=rf"non-finite {name} tendency"):
            validate_ysu_tendencies(native, status=status)
        native[name].flat[0] = cp.float32(1.0)
    native["kpbl"].flat[0] = cp.iinfo(cp.int32).min
    native["cloudflg"].flat[0] = cp.iinfo(cp.int32).max
    validate_ysu_tendencies(native, status=status)


@requires_gpu
@pytest.mark.gpu
def test_wrf_fixed_step_predicates_remain_driver_specific():
    from woof.core.physics import (
        _cumulus_step_due,
        _radiation_step_due,
        _surface_pbl_step_due,
    )

    assert [step for step in range(1, 17)
            if _radiation_step_due(step, 12, 12.0)] == [1, 13]
    assert [step for step in range(1, 17)
            if _cumulus_step_due(step, 5, 5.0)] == [1, 5, 10, 15]
    assert [step for step in range(1, 17)
            if _surface_pbl_step_due(step, 5, 5.0)] == [1, 5, 10, 15]
    assert all(_radiation_step_due(step, 1, 1.0)
               for step in range(1, 17))

    acceptance_steps = [
        step for step in range(1, 5761)
        if _radiation_step_due(step, 8, 1.0)
    ]
    assert len(acceptance_steps) == 720
    assert acceptance_steps[:2] == [1, 9]
    assert acceptance_steps[-1] == 5753


@pytest.mark.gpu
@requires_gpu
def test_wrf_order_and_stepra_stepcu_event_calendar(monkeypatch):
    import cupy as cp

    from woof.core.physics import CumulusResult, RadiationResult

    events = []

    def radiation(**kwargs):
        state = kwargs["state"]
        events.append((state.elapsed_seconds, "radiation"))
        nz, ny, nx = state.p.shape
        return RadiationResult(
            cp.zeros((nz, ny, nx), cp.float32),
            cp.zeros((nz, ny, nx), cp.float32),
            cp.zeros((ny, nx), cp.float32),
            cp.zeros((ny, nx), cp.float32))

    # Scheme callables cannot override the driver's STEPRA calendar.
    radiation.due_at = lambda _: True

    def cumulus(**kwargs):
        state = kwargs["state"]
        events.append((state.elapsed_seconds, "cumulus"))
        nz, ny, nx = state.p.shape
        zeros = cp.zeros((nz, ny, nx), cp.float32)
        return CumulusResult(zeros, zeros)

    state, cfg, driver = _full_state(
        dt=60.0, bldt=5.0, ra_physics=90, cu_physics=1,
        radt_minutes=12.0, cudt_minutes=5.0,
        radiation=radiation, cumulus=cumulus)

    monkeypatch.setattr(driver, "_run_sfclay",
                        lambda *_: events.append((state.elapsed_seconds,
                                                  "sfclay")))
    monkeypatch.setattr(driver, "_run_noah",
                        lambda *_: events.append((state.elapsed_seconds,
                                                  "noah")))
    monkeypatch.setattr(driver, "_run_ysu",
                        lambda *_: events.append((state.elapsed_seconds,
                                                  "ysu")))

    for now in np.arange(0.0, 901.0, 60.0):
        state.elapsed_seconds = now
        driver.compute(state, cfg)

    assert [event for event in events if event[1] == "radiation"] == [
        (0.0, "radiation"), (720.0, "radiation")]
    assert [event for event in events if event[1] == "cumulus"] == [
        (0.0, "cumulus"), (240.0, "cumulus"),
        (540.0, "cumulus"), (840.0, "cumulus")]
    for name in ("sfclay", "noah", "ysu"):
        assert [event for event in events if event[1] == name] == [
            (0.0, name), (240.0, name), (540.0, name), (840.0, name)]
    assert [name for now, name in events if now == 0.0] == [
        "radiation", "sfclay", "noah", "ysu", "cumulus"]


@pytest.mark.gpu
@requires_gpu
def test_radiation_and_cumulus_attachment_off_calendar_waits_until_due():
    import cupy as cp

    from woof.core.physics import CumulusResult, RadiationResult

    events = []

    def radiation(**kwargs):
        state = kwargs["state"]
        events.append((state.elapsed_seconds, "radiation"))
        nz, ny, nx = state.p.shape
        return RadiationResult(
            cp.zeros((nz, ny, nx), cp.float32),
            cp.zeros((nz, ny, nx), cp.float32),
            cp.zeros((ny, nx), cp.float32),
            cp.zeros((ny, nx), cp.float32))

    def cumulus(**kwargs):
        state = kwargs["state"]
        events.append((state.elapsed_seconds, "cumulus"))
        nz, ny, nx = state.p.shape
        zeros = cp.zeros((nz, ny, nx), cp.float32)
        return CumulusResult(zeros, zeros)

    state, cfg, driver = _full_state(
        sf_sfclay_physics=0, sf_surface_physics=0, bl_pbl_physics=0,
        dt=60.0, ra_physics=90, cu_physics=1,
        radt_minutes=12.0, cudt_minutes=5.0,
        radiation=radiation, cumulus=cumulus)

    # The driver's first invocation is ITIMESTEP=2, not WRF's initial step.
    state.elapsed_seconds = 60.0
    driver.compute(state, cfg)
    assert events == []

    state.elapsed_seconds = 240.0  # ITIMESTEP=5: STEPCU is due.
    driver.compute(state, cfg)
    state.elapsed_seconds = 720.0  # ITIMESTEP=13: STEPRA is due.
    driver.compute(state, cfg)
    assert events == [(240.0, "cumulus"), (720.0, "radiation")]


@pytest.mark.gpu
@requires_gpu
def test_radiation_and_cumulus_contracts_hold_and_accumulate_rainc():
    """Results WITHOUT nca_seconds keep the Task-1 attachment contract:
    wholesale replacement on due calls plus a per-due-call RAINC
    increment.  The production KF path carries nca_seconds and follows
    WRF's NCA persistence instead (see the test below).  This mp=0 state
    has no frozen prognostics, so the custom attachment supplies its own
    energy-consistent warm-rain QC/QR closure and omits QI/QS."""
    import cupy as cp

    from woof.core.physics import CumulusResult, RadiationResult
    from woof.io.wrfout import state_frame

    returned = {}

    def radiation(**kwargs):
        state = kwargs["state"]
        nz, ny, nx = state.p.shape
        result = RadiationResult(
            cp.full((nz, ny, nx), 2.0e-5, cp.float32),
            cp.full((nz, ny, nx), 7.0e-6, cp.float32),
            cp.full((ny, nx), 640.0, cp.float32),
            cp.full((ny, nx), 315.0, cp.float32))
        returned["radiation"] = result
        return result

    def cumulus(**kwargs):
        state = kwargs["state"]
        nz, ny, nx = state.p.shape
        result = CumulusResult(
            cp.full((nz, ny, nx), 3.0e-5, cp.float32),
            cp.full((nz, ny, nx), -4.0e-8, cp.float32),
            rqccuten=cp.full((nz, ny, nx), 10.5e-9, cp.float32),
            rqrcuten=cp.full((nz, ny, nx), 12.5e-9, cp.float32),
            rainc=cp.full((ny, nx), 0.25, cp.float32))
        returned["cumulus"] = result
        return result

    state, cfg, driver = _full_state(
        sf_sfclay_physics=0, sf_surface_physics=0, bl_pbl_physics=0,
        dt=60.0, ra_physics=90, cu_physics=1,
        radt_minutes=12.0, cudt_minutes=5.0,
        radiation=radiation, cumulus=cumulus)
    got = driver.compute(state, cfg)
    chm = (state.c1h[:, None, None] * state.total_mu()[None]
           + state.c2h[:, None, None])
    cp.testing.assert_array_equal(
        got.rtheta,
        chm * (returned["radiation"].rthratenlw
               + returned["radiation"].rthratensw)
        + chm * returned["cumulus"].rthcuten)
    cp.testing.assert_array_equal(
        driver.rthratenlw, returned["radiation"].rthratenlw)
    from woof.wrf_exact import ENABLED as strict
    input_lw = returned["radiation"].rthratenlw
    input_sw = returned["radiation"].rthratensw
    # radiation_driver.F:2870 publishes total minus saved longwave.
    expected_sw = (input_lw + input_sw) - input_lw if strict else input_sw
    cp.testing.assert_array_equal(driver.rthratensw, expected_sw)
    cp.testing.assert_array_equal(
        got.rqv, chm * returned["cumulus"].rqvcuten)
    cp.testing.assert_array_equal(
        got.rqc, chm * returned["cumulus"].rqccuten)
    cp.testing.assert_array_equal(
        got.rqr, chm * returned["cumulus"].rqrcuten)
    assert got.rqi is None and got.rqs is None
    # KF's held producer rates carry their mass-coupled words through
    # transport between compute() and the step epilogue's
    # finish_history_mass (native calculate_phy_tend / phy_prep_part2;
    # fold 1eb3e8076a, tests/test_kf_held_mass.py), so after compute they
    # are the raw rate times this step's chm, word for word.
    cp.testing.assert_array_equal(
        driver.cu_rates["rqccuten"],
        chm * returned["cumulus"].rqccuten)
    cp.testing.assert_array_equal(
        driver.cu_rates["rqrcuten"],
        chm * returned["cumulus"].rqrcuten)
    cp.testing.assert_array_equal(driver.cu_rates["rqicuten"], 0.0)
    cp.testing.assert_array_equal(driver.cu_rates["rqscuten"], 0.0)
    cp.testing.assert_array_equal(driver.fields["swdown"], 640.0)
    cp.testing.assert_array_equal(driver.fields["glw"], 315.0)

    # The driver owns the held copies: mutating a scheme work array after
    # return must not alter the per-step tendency stack.
    returned["radiation"].rthratenlw[...] = 99.0
    returned["radiation"].rthratensw[...] = 99.0
    returned["cumulus"].rqvcuten[...] = 99.0
    held = got.rtheta.copy(), got.rqv.copy()
    state.elapsed_seconds = 60.0
    again = driver.compute(state, cfg)
    cp.testing.assert_array_equal(again.rtheta, held[0])
    cp.testing.assert_array_equal(again.rqv, held[1])
    assert driver.call_counts["radiation"] == 1
    assert driver.call_counts["cumulus"] == 1

    frame = state_frame(state)
    cp.testing.assert_array_equal(driver.rainc, 0.25)
    np.testing.assert_array_equal(
        frame["RAINC"], np.full((cfg.ny, cfg.nx), 0.25, np.float32))
    state.elapsed_seconds = 240.0
    driver.compute(state, cfg)
    cp.testing.assert_array_equal(driver.rainc, 0.5)


@pytest.mark.gpu
@requires_gpu
def test_cumulus_nca_results_accumulate_rainc_every_step_into_wrfout():
    """Task 6b wrfout regression: NCA results follow WRF ``advance_ppt``.

    RAINC grows by PRATEC*DT on EVERY model step, not only on STEPCU
    events (module_physics_addtendc.F:2141); the stored tendencies are
    zeroed once NINT(NCA/DT) <= 1 and reach the RK forcing one step
    later; the rain rate survives expiry until the column's next scheme
    call (2216-2217 are commented out in WRF).
    """
    import cupy as cp

    from woof.core.physics import CumulusResult
    from woof.io.wrfout import state_frame

    calls = []
    pratec_value = float(np.float32(3.0e-4))

    def cumulus(**kwargs):
        state = kwargs["state"]
        calls.append(float(state.elapsed_seconds))
        nz, ny, nx = state.p.shape
        return CumulusResult(
            cp.full((nz, ny, nx), 1.0e-5, cp.float32),
            cp.zeros((nz, ny, nx), cp.float32),
            nca_seconds=cp.full((ny, nx), 120.0, cp.float32),  # NIC=2
            pratec=cp.full((ny, nx), 3.0e-4, cp.float32))

    state, cfg, driver = _full_state(
        sf_sfclay_physics=0, sf_surface_physics=0, bl_pbl_physics=0,
        dt=60.0, cu_physics=1, cudt_minutes=5.0, cumulus=cumulus)

    finish_calls = []
    finish_step = driver.finish_step

    def counted_finish_step():
        finish_calls.append(float(state.elapsed_seconds))
        finish_step()

    driver.finish_step = counted_finish_step
    rtheta_max = []
    for step, now in enumerate((0.0, 60.0, 120.0, 180.0, 240.0)):
        state.elapsed_seconds = now
        held = driver.compute(state, cfg)
        rtheta_max.append(float(cp.abs(held.rtheta).max()))
        frame = state_frame(state)
        np.testing.assert_allclose(
            frame["RAINC"],
            np.full((cfg.ny, cfg.nx), (step + 1) * 60.0 * pratec_value,
                    np.float32), rtol=1.0e-5)
    # Due calls at ITIMESTEP 1 and 5 only; the 120 s NCA expired after the
    # 60 s step (NINT(60/60) <= 1), so steps at 120 s and 180 s ran with
    # zeroed tendencies while RAINC kept accumulating, and the 240 s due
    # call readmitted the column.
    assert calls == [0.0, 240.0]
    assert rtheta_max[0] > 0.0 and rtheta_max[1] > 0.0
    assert rtheta_max[2] == rtheta_max[3] == 0.0
    assert rtheta_max[4] > 0.0
    # Each completed model step probes the expiry mask once in advance_ppt.
    # The known-clean compute entry does not repeat the device-to-host probe.
    assert finish_calls == [0.0, 60.0, 120.0, 180.0, 240.0]
    assert driver._cu_expiry_pending is False
    cp.testing.assert_array_equal(
        driver.cu_raincv, cp.float32(60.0) * cp.float32(3.0e-4))


@pytest.mark.gpu
@requires_gpu
def test_cumulus_clock_arithmetic_follows_model_clock_under_substeps():
    """Task 6b audit: KF driver DT is cfg.clock_dt, not the substep dt.

    The real74 compatibility integrator advances eight 7.5 s substeps per
    60 s model-clock step.  WRF's KF driver formulas -- the 0.5*DT hold
    boundary and NINT(NCA/DT) expiry (module_cu_kfeta.F:410,
    module_physics_addtendc.F:2211-2228), RAINC += PRATEC*DT (2141), and
    the once-per-step advance_ppt placement (solve_em.F:3558-3571) -- are
    defined on the model clock, the same idiom as the Davies and
    diff_6th clock coefficients.
    """
    import cupy as cp

    from woof.core.physics import CumulusResult

    pratec_value = float(np.float32(3.0e-4))
    calls = []

    def cumulus(**kwargs):
        state = kwargs["state"]
        calls.append(float(state.elapsed_seconds))
        nz, ny, nx = state.p.shape
        return CumulusResult(
            cp.full((nz, ny, nx), 1.0e-5, cp.float32),
            cp.zeros((nz, ny, nx), cp.float32),
            nca_seconds=cp.full((ny, nx), 120.0, cp.float32),  # NIC=2
            pratec=cp.full((ny, nx), 3.0e-4, cp.float32))

    state, cfg, driver = _full_state(
        sf_sfclay_physics=0, sf_surface_physics=0, bl_pbl_physics=0,
        dt=7.5, clock_dt=60.0, cu_physics=1, cudt_minutes=5.0,
        cumulus=cumulus)
    # The T1 STEPCU calendar stays on internal steps (due at substep
    # ITIMESTEP 1 and 40); only the DT arithmetic moves to the clock.
    assert driver.stepcu == 40

    nca_trace = []
    rainc_trace = []
    rtheta_active = []
    for substep in range(48):                    # six model-clock steps
        state.elapsed_seconds = substep * 7.5
        held = driver.compute(state, cfg)
        nca_trace.append(float(driver.cu_nca[0, 0]))
        rainc_trace.append(float(driver.rainc[0, 0]))
        rtheta_active.append(bool(cp.abs(held.rtheta).max() > 0))

    assert calls == [0.0, 292.5]
    # NCA decrements by 60 s exactly once per clock step, on its final
    # substep; the 120 s hold expires on the NINT(NCA/60) <= 1 boundary.
    assert nca_trace[0] == nca_trace[6] == 120.0
    assert nca_trace[7] == nca_trace[14] == 60.0
    assert nca_trace[15] == nca_trace[38] == 0.0
    assert nca_trace[39] == 60.0        # readmitted at the 292.5 s due call
    assert nca_trace[47] == 0.0
    # Tendencies stay live through the expiry substep's compose, go to
    # zero for the following clock steps, and return on readmission.
    assert rtheta_active[15] and not rtheta_active[16]
    assert not rtheta_active[38] and rtheta_active[39]
    # RAINC accumulates PRATEC*60 once per clock step (not 8x PRATEC*7.5
    # per substep -- same total, but stepwise on the clock).
    assert rainc_trace[6] == 0.0
    assert rainc_trace[7] == pytest.approx(60.0 * pratec_value, rel=1e-6)
    assert rainc_trace[38] == pytest.approx(4 * 60.0 * pratec_value,
                                            rel=1e-5)
    assert rainc_trace[47] == pytest.approx(6 * 60.0 * pratec_value,
                                            rel=1e-5)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("bad_name", ["rthratenlw", "rthratensw"])
def test_scheme_output_shapes_are_checked_at_the_driver_boundary(bad_name):
    import cupy as cp

    from woof.core.physics import RadiationResult

    def bad_radiation(**kwargs):
        state = kwargs["state"]
        nz, ny, nx = state.p.shape
        heating = {
            "rthratenlw": cp.zeros((nz, ny, nx), cp.float32),
            "rthratensw": cp.zeros((nz, ny, nx), cp.float32),
        }
        heating[bad_name] = cp.zeros((nz, ny, nx + 1), cp.float32)
        return RadiationResult(
            heating["rthratenlw"],
            heating["rthratensw"],
            cp.zeros((ny, nx), cp.float32),
            cp.zeros((ny, nx), cp.float32))

    state, cfg, driver = _full_state(
        sf_sfclay_physics=0, sf_surface_physics=0, bl_pbl_physics=0,
        ra_physics=90, radiation=bad_radiation)
    with pytest.raises(ValueError, match=f"radiation {bad_name}"):
        driver.compute(state, cfg)


@requires_gpu
@pytest.mark.gpu
def test_radiation_result_requires_separate_lw_and_sw_tendencies():
    from woof.core.physics import RadiationResult

    with pytest.raises(TypeError, match="rthratensw"):
        RadiationResult(
            rthratenlw=np.zeros((1, 1, 1), np.float32),
            swdown=np.zeros((1, 1), np.float32),
            glw=np.zeros((1, 1), np.float32))


@pytest.mark.gpu
@requires_gpu
def test_analytic_proxy_is_bit_identical_through_ra_physics_90():
    import cupy as cp

    from woof.core.analytic_radiation import (
        AnalyticClearSkyRadiation,
        analytic_clear_sky_forcing,
    )
    from woof.core.diagnostics import update_diagnostics

    start = datetime(1974, 4, 3, 12)
    latitude = cp.full((1, 1), 40.0, cp.float32)
    longitude = cp.full((1, 1), -100.0, cp.float32)

    old_state, old_cfg, old_driver = _full_state(
        nx=1, ny=1, nz=24, dt=600.0, time_step_sound=4,
        radt_minutes=0.0, bldt=10.0)
    new_state, new_cfg, new_driver = _full_state(
        nx=1, ny=1, nz=24, dt=600.0, time_step_sound=4,
        ra_physics=90, radt_minutes=0.0, bldt=10.0,
        radiation_start_time=start, radiation_latitude=latitude,
        radiation_longitude=longitude)
    assert isinstance(new_driver.radiation_callable,
                      AnalyticClearSkyRadiation)

    for _ in range(24 * 6):
        update_diagnostics(old_state)
        update_diagnostics(new_state)
        valid = start + timedelta(seconds=float(old_state.elapsed_seconds))
        # The seam identity pins PLUMBING: both paths must consume the
        # same driver-prepared inputs.  Since the E2 p_hyd wiring
        # (phy_prep hydrostatic pressures for the physics seams), those
        # inputs come from _prepare_atmosphere, not raw state.p -- and
        # the proxy derives its own temperature from theta with the
        # pinned Phase-3 287/1004 FP32 exponent (analytic_radiation.py),
        # so the direct call must reproduce that exact derivation.
        from woof.core.physics import _prepare_atmosphere
        atmosphere = _prepare_atmosphere(old_state)
        temperature = (atmosphere["theta"][0]
                       * (atmosphere["pressure"][0]
                          / cp.float32(100000.0))
                       ** cp.float32(287.0 / 1004.0))
        swdown, glw = analytic_clear_sky_forcing(
            valid, latitude, longitude, temperature,
            atmosphere["qv"][0], atmosphere["pressure"][0])
        old_driver.set_forcing(swdown=swdown, glw=glw)
        old_tendencies = old_driver.compute(old_state, old_cfg)
        new_tendencies = new_driver.compute(new_state, new_cfg)

        cp.testing.assert_array_equal(
            new_driver.fields["swdown"], old_driver.fields["swdown"])
        cp.testing.assert_array_equal(
            new_driver.fields["glw"], old_driver.fields["glw"])
        for name in ("ru", "rv", "rtheta", "rqv", "rqc", "rqr"):
            cp.testing.assert_array_equal(
                getattr(new_tendencies, name), getattr(old_tendencies, name))

        for state, driver in ((old_state, old_driver),
                              (new_state, new_driver)):
            ysu = driver.last_ysu
            dt = cp.float32(old_cfg.dt)
            state.u += dt * ysu["du"]
            state.v += dt * ysu["dv"]
            state.thp += dt * ysu["dtheta"]
            state.qv[...] = cp.maximum(
                state.qv + dt * ysu["dqv"], 0.0)
            state.qc[...] = cp.maximum(
                state.qc + dt * ysu["dqc"], 0.0)
            state.elapsed_seconds += old_cfg.dt


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_scheme_is_selected_through_ra_physics_4():
    import cupy as cp
    from woof.core.rrtmgp import RRTMGPRadiation

    start = datetime(1974, 4, 3, 18)
    latitude = cp.full((1, 1), 40.0, cp.float32)
    longitude = cp.full((1, 1), -100.0, cp.float32)
    _state, _cfg, driver = _full_state(
        nx=1, ny=1, nz=16, ra_physics=4, radt_minutes=12.0,
        radiation_start_time=start, radiation_latitude=latitude,
        radiation_longitude=longitude)
    assert isinstance(driver.radiation_callable, RRTMGPRadiation)


@pytest.mark.gpu
@requires_gpu
def test_physics_computed_once_pre_rk_and_added_to_all_three_stages(monkeypatch):
    import cupy as cp

    from woof.core.dycore import step
    import woof.core.dycore as dycore
    from woof.core.physics import PhysicsTendencies
    from woof.wrf_exact import ENABLED as strict

    state, cfg, _ = _full_state(sf_surface_physics=0)
    events = []
    tendency = PhysicsTendencies.zeros(state)
    original_add = tendency.add_to_slow

    def add_spy(s):
        events.append("add")
        original_add(s)

    tendency.add_to_slow = add_spy
    original_fold = dycore._fold_physics_wrf
    def fold_spy(s, specs, held):
        assert held is tendency
        events.append("fold")
        return original_fold(s, specs, held)
    monkeypatch.setattr(dycore, "_fold_physics_wrf", fold_spy)

    class DriverSpy:
        def compute(self, s, c):
            events.append("compute")
            return tendency

        def finish_history_mass(self, s, c, **kwargs):
            events.append("finish_history")

    state.physics = DriverSpy()
    step(state, cfg)
    expected = (["compute", "fold", "finish_history"] if strict else
                ["compute", "add", "add", "add", "finish_history"])
    assert events == expected


@pytest.mark.gpu
@requires_gpu
def test_land_water_mask_and_diagnostics_are_preserved_for_output():
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.io.wrfout import state_frame

    state, cfg, driver = _full_state(dt=30.0)
    water = (slice(None), -1)
    land_tsk = driver.fields["tsk"][:, :-1].copy()
    water_tsk = driver.fields["tsk"][water].copy()
    water_soil = driver.fields["smois"][:, :, -1].copy()
    update_diagnostics(state)
    tendencies = driver.compute(state, cfg)

    cp.testing.assert_array_equal(driver.fields["tsk"][water], water_tsk)
    cp.testing.assert_array_equal(driver.fields["smois"][:, :, -1], water_soil)
    assert not bool(cp.array_equal(driver.fields["tsk"][:, :-1], land_tsk))
    assert bool(cp.isfinite(tendencies.rtheta).all())
    assert bool(cp.isfinite(driver.fields["u10"]).all())
    assert bool(cp.isfinite(driver.fields["t2"]).all())
    cp.testing.assert_array_equal(driver.fields["xland"][:, -1], 2.0)

    frame = state_frame(state)
    expected = {"TSK", "T2", "Q2", "U10", "V10", "UST", "HFX",
                "QFX", "LH", "PBLH", "GRDFLX"}
    assert expected <= set(frame)
    for name in expected:
        assert frame[name].shape == (cfg.ny, cfg.nx)
        assert frame[name].dtype == np.float32
        assert np.isfinite(frame[name]).all(), name


@pytest.mark.gpu
@requires_gpu
def test_kessler_precipitation_accumulates_into_noah_rainbl(monkeypatch):
    """Named RAINNCV increments are consumed once on each due surface call."""
    import cupy as cp
    import woof.core.physics as physics

    from woof.core.diagnostics import update_diagnostics
    from woof.core.microphysics import MicrophysicsDiagnostics

    state, cfg, driver = _full_state(bl_pbl_physics=0)
    seen = []

    def capture(dev, params, dt, dzs, **kwargs):
        seen.append(cp.asnumpy(dev["rainbl"]).copy())

    monkeypatch.setattr(physics, "launch_noah", capture)
    shape = (cfg.ny, cfg.nx)
    zeros = cp.zeros(shape, cp.float32)
    driver.accept_microphysics(MicrophysicsDiagnostics(
        rainnc=cp.full(shape, 3.0, cp.float32),
        rainncv=cp.full(shape, 3.0, cp.float32), sr=zeros))
    update_diagnostics(state)
    driver.compute(state, cfg)
    cp.testing.assert_array_equal(driver.fields["rainbl"], 0.0)

    driver.accept_microphysics(MicrophysicsDiagnostics(
        rainnc=cp.full(shape, 4.25, cp.float32),
        rainncv=cp.full(shape, 1.25, cp.float32), sr=zeros))
    state.elapsed_seconds += cfg.dt
    update_diagnostics(state)
    driver.compute(state, cfg)
    np.testing.assert_array_equal(seen[0], np.full((cfg.ny, cfg.nx), 3.0,
                                                   np.float32))
    np.testing.assert_array_equal(seen[1], np.full((cfg.ny, cfg.nx), 1.25,
                                                   np.float32))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mp_physics", [6, 26])
def test_wsm6_sr_real_device_accepts_upper_and_rejects_beyond_envelope(mp_physics):
    """Exercise the production CuPy comparisons on an actual device."""
    import cupy as cp

    from woof.core.microphysics import MicrophysicsDiagnostics

    _state, cfg, driver = _full_state(
        mp_physics=mp_physics, bl_pbl_physics=0, dt=60.0)
    shape = (cfg.ny, cfg.nx)
    accepted = cp.full(shape, driver._sr_roundoff_upper, cp.float32)
    ones = cp.ones(shape, cp.float32)
    driver.accept_microphysics(MicrophysicsDiagnostics(
        rainnc=ones, rainncv=ones, sr=accepted))
    cp.testing.assert_array_equal(driver.microphysics.sr, accepted)

    upper_bits = np.asarray(
        driver._sr_roundoff_upper, np.float32).view(np.uint32).item()
    beyond = np.asarray(upper_bits + 1, np.uint32).view(np.float32)
    with pytest.raises(ValueError, match=f"roundoff_ulps={driver._sr_roundoff_max_ulps}"):
        driver.accept_microphysics(MicrophysicsDiagnostics(
            rainnc=ones, rainncv=ones,
            sr=cp.full(shape, beyond, cp.float32)))


@pytest.mark.gpu
@requires_gpu
def test_kf_convective_rain_reaches_noah_rainbl(monkeypatch):
    """WRF wets the surface with BOTH precipitation kinds every step:
    RAINBL += RAINCV + RAINNCV (module_surface_driver.F:1566, RAINCV =
    PRATEC*DT).  Final-review MAJOR: the driver previously fed Noah only
    the microphysics half, so soil moisture never saw KF rain."""
    import cupy as cp
    import woof.core.physics as physics

    from woof.core.diagnostics import update_diagnostics
    from woof.core.physics import CumulusResult

    rate = 2.0e-4  # mm s-1 convective rain rate

    def cumulus(**kwargs):
        st = kwargs["state"]
        nz, ny, nx = st.p.shape
        return CumulusResult(
            cp.zeros((nz, ny, nx), cp.float32),
            cp.zeros((nz, ny, nx), cp.float32),
            rainc=cp.zeros((ny, nx), cp.float32),
            nca_seconds=cp.full((ny, nx), 1800.0, cp.float32),
            pratec=cp.full((ny, nx), rate, cp.float32))

    state, cfg, driver = _full_state(
        bl_pbl_physics=0, dt=60.0, cu_physics=1, cudt_minutes=5.0,
        cumulus=cumulus)
    seen = []

    def capture(dev, params, dt, dzs, **kwargs):
        seen.append(cp.asnumpy(dev["rainbl"]).copy())

    monkeypatch.setattr(physics, "launch_noah", capture)
    update_diagnostics(state)
    driver.compute(state, cfg)          # step 1: KF fires, PRATEC held
    state.elapsed_seconds += cfg.dt
    update_diagnostics(state)
    driver.compute(state, cfg)          # step 2: surface sees held rain
    # Step 1's post-compose advance accumulated PRATEC*dt into the
    # pending bucket; the next due surface call consumes it exactly once.
    expected = np.float32(rate) * np.float32(cfg.dt)
    np.testing.assert_array_equal(
        seen[1], np.full((cfg.ny, cfg.nx), expected, np.float32))
    # RAINC accumulated the same held rate over both steps.
    np.testing.assert_allclose(
        cp.asnumpy(driver.rainc),
        np.full((cfg.ny, cfg.nx), 2.0 * rate * cfg.dt, np.float32),
        rtol=1.0e-6)


@pytest.mark.gpu
@requires_gpu
def test_morrison_sr_drives_noah_frpcpn_through_named_contract(monkeypatch):
    """Morrison SR replaces the temperature proxy when Noah uses FRPCPN."""
    import cupy as cp
    import woof.core.physics as physics

    from woof.core.diagnostics import update_diagnostics
    from woof.core.microphysics import MicrophysicsDiagnostics

    state, cfg, driver = _full_state(mp_physics=10, bl_pbl_physics=0)
    shape = (cfg.ny, cfg.nx)
    frozen = cp.full(shape, 0.73, cp.float32)
    driver.accept_microphysics(MicrophysicsDiagnostics(
        rainnc=cp.full(shape, 2.0, cp.float32),
        rainncv=cp.full(shape, 0.4, cp.float32), sr=frozen))
    seen = {}

    def capture(dev, params, dt, dzs, **kwargs):
        seen["sr"] = cp.asnumpy(dev["sr"]).copy()
        seen["rainbl"] = cp.asnumpy(dev["rainbl"]).copy()
        seen["frpcpn"] = kwargs["frpcpn"]

    monkeypatch.setattr(physics, "launch_noah", capture)
    update_diagnostics(state)
    driver.compute(state, cfg)

    np.testing.assert_array_equal(seen["sr"], np.full(shape, 0.73, np.float32))
    np.testing.assert_array_equal(
        seen["rainbl"], np.full(shape, 0.4, np.float32))
    assert seen["frpcpn"] is True
    with pytest.raises(ValueError, match="unknown physics forcing"):
        driver.set_forcing(sr=0.0)


@pytest.mark.gpu
@requires_gpu
def test_nssl_complete_diagnostics_drive_noah_frpcpn(monkeypatch):
    """NSSL requires every category diagnostic and supplies Noah's SR."""
    import cupy as cp
    import woof.core.physics as physics

    from woof.core.diagnostics import update_diagnostics
    from woof.core.microphysics import MicrophysicsDiagnostics

    state, cfg, driver = _full_state(mp_physics=18, bl_pbl_physics=0)
    shape = (cfg.ny, cfg.nx)
    values = {
        "rainnc": 2.0, "rainncv": 0.4, "sr": 0.73,
        "snownc": 3.0, "snowncv": 0.3,
        "graupelnc": 4.0, "graupelncv": 0.2,
        "hailnc": 5.0, "hailncv": 0.1,
    }
    for name, value in values.items():
        getattr(driver.microphysics, name)[...] = cp.float32(value)
    complete = MicrophysicsDiagnostics(**{
        name: getattr(driver.microphysics, name) for name in values
    })
    for name in (
            "snownc", "snowncv", "graupelnc", "graupelncv",
            "hailnc", "hailncv"):
        with pytest.raises(ValueError, match=name.upper()):
            driver.accept_microphysics(
                dataclasses.replace(complete, **{name: None}))
        with pytest.raises(
                ValueError, match="canonical PhysicsDriver array"):
            driver.accept_microphysics(dataclasses.replace(
                complete,
                **{name: getattr(complete, name).copy()}))

    driver.accept_microphysics(complete)
    for name, value in values.items():
        cp.testing.assert_array_equal(
            getattr(driver.microphysics, name), cp.float32(value))
    seen = {}

    def capture(dev, params, dt, dzs, **kwargs):
        seen["sr"] = cp.asnumpy(dev["sr"]).copy()
        seen["rainbl"] = cp.asnumpy(dev["rainbl"]).copy()
        seen["frpcpn"] = kwargs["frpcpn"]

    monkeypatch.setattr(physics, "launch_noah", capture)
    update_diagnostics(state)
    driver.compute(state, cfg)

    np.testing.assert_array_equal(seen["sr"], np.full(shape, 0.73, np.float32))
    np.testing.assert_array_equal(
        seen["rainbl"], np.full(shape, 0.4, np.float32))
    assert seen["frpcpn"] is True


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize(
    ("mp_physics", "has_frozen", "has_hail"),
    [(1, False, False), (6, True, False), (8, True, False),
     (10, True, False), (18, True, True)],
)
def test_every_microphysics_selector_writes_the_species_surface_seam(
        mp_physics, has_frozen, has_hail):
    """The real selector contracts populate the persistent ARW forcing."""
    import cupy as cp

    from woof.core.microphysics import MicrophysicsDiagnostics

    _state, cfg, driver = _full_state(
        mp_physics=mp_physics, sf_surface_physics=4,
        bl_pbl_physics=0,
        radiation_start_time=datetime(2026, 7, 1, 18),
        radiation_latitude=40.0, radiation_longitude=-100.0)
    shape = (cfg.ny, cfg.nx)
    values = {
        "rainnc": 2.0, "rainncv": 0.4, "sr": 0.75,
        "snownc": 0.3, "snowncv": 0.3,
        "graupelnc": 0.2, "graupelncv": 0.2,
        "hailnc": 0.1, "hailncv": 0.1,
    }
    supported = {"rainnc", "rainncv", "sr"}
    if has_frozen:
        supported.update(
            {"snownc", "snowncv", "graupelnc", "graupelncv"})
    if has_hail:
        supported.update({"hailnc", "hailncv"})

    if mp_physics == 18:
        for name in supported:
            getattr(driver.microphysics, name)[...] = cp.float32(values[name])
        diagnostics = MicrophysicsDiagnostics(**{
            name: getattr(driver.microphysics, name) for name in supported
        })
    else:
        diagnostics = MicrophysicsDiagnostics(**{
            name: cp.full(shape, values[name], cp.float32)
            for name in supported
        })
    driver.accept_microphysics(diagnostics)

    expected = {
        "surface_rainncv": 0.4,
        "surface_snowncv": 0.3 if has_frozen else 0.0,
        "surface_graupelncv": 0.2 if has_frozen else 0.0,
        "surface_hailncv": 0.1 if has_hail else 0.0,
    }
    for name, value in expected.items():
        cp.testing.assert_array_equal(
            driver.fields[name], cp.full(shape, value, cp.float32))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mp_physics", [1, 6, 8, 10])
def test_native_microphysics_validation_batches_all_fields_and_resets(
        mp_physics):
    import cupy as cp

    from woof.core.microphysics import MicrophysicsDiagnostics
    from woof.core.physics import microphysics_scratch_slots

    state, cfg, driver = _full_state(
        mp_physics=mp_physics, bl_pbl_physics=0)
    names = tuple(
        name for name, _slot in microphysics_scratch_slots(mp_physics))
    values = {}
    for name in names:
        value = getattr(driver.microphysics, name)
        value[...] = cp.float32(0.5 if name == "sr" else 1.0)
        values[name] = value
    diagnostics = MicrophysicsDiagnostics(**values)
    status = state.scratch(
        (1,), "physics_validation_status").view(cp.uint32)
    status[...] = cp.uint32(0xFFFFFFFF)

    driver.accept_microphysics(diagnostics)
    assert int(status[0].item()) == 0

    labels = {
        "rainnc": "RAINNC", "rainncv": "RAINNCV", "sr": "SR",
        "snownc": "SNOWNC", "snowncv": "SNOWNCV",
        "graupelnc": "GRAUPELNC", "graupelncv": "GRAUPELNCV",
    }
    for name in names:
        values[name].flat[0] = cp.nan
        with pytest.raises(
                FloatingPointError,
                match=rf"microphysics {labels[name]} contains a non-finite"):
            driver.accept_microphysics(diagnostics)
        values[name].flat[0] = cp.float32(
            0.5 if name == "sr" else 1.0)

    values[names[-1]].flat[0] = cp.nan
    values[names[0]].flat[0] = cp.inf
    with pytest.raises(
            FloatingPointError,
            match=rf"microphysics {labels[names[0]]} contains a non-finite"):
        driver.accept_microphysics(diagnostics)
    values[names[-1]].flat[0] = cp.float32(
        0.5 if names[-1] == "sr" else 1.0)
    values[names[0]].flat[0] = cp.float32(
        0.5 if names[0] == "sr" else 1.0)

    values["sr"].flat[0] = cp.float32(-0.25)
    with pytest.raises(ValueError, match="validated range"):
        driver.accept_microphysics(diagnostics)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("scheme", [1, 10], ids=["kf", "kfcup"])
def test_native_kf_validation_preserves_field_order_and_inputs(monkeypatch, scheme):
    import cupy as cp

    from woof.core.kf import KainFritsch, KFCumulusPotential
    from woof.core.physics import _NativeKFCumulusResult, _is_native_kf_result

    state, cfg, driver = _full_state(
        mp_physics=10, cu_physics=scheme, nz=40, bl_pbl_physics=0,
        sf_sfclay_physics=0, sf_surface_physics=0)
    owner = driver.cumulus_callable
    producer = KFCumulusPotential if scheme == 10 else KainFritsch
    assert type(owner) is producer
    shape = tuple(state.p.shape)
    surface = shape[1:]
    names = (
        "rthcuten", "rqvcuten", "rqccuten", "rqicuten",
        "rqrcuten", "rqscuten", "nca_seconds", "pratec",
    )
    labels = (
        "cumulus rthcuten", "cumulus rqvcuten", "cumulus rqccuten",
        "cumulus rqicuten", "cumulus rqrcuten", "cumulus rqscuten",
        "cumulus nca_seconds", "cumulus PRATEC",
    )
    values = {
        name: cp.zeros(shape if index < 6 else surface, cp.float32)
        for index, name in enumerate(names)
    }
    result = _NativeKFCumulusResult(
        owner=owner, **values,
        rainc=cp.full(surface, cp.nan, cp.float32))

    monkeypatch.setattr(
        producer, "__call__", lambda self, **_kwargs: result)
    assert _is_native_kf_result(result, owner, state, cfg, shape)
    assert not _is_native_kf_result(result, producer(), state, cfg, shape)
    before = {name: value.copy() for name, value in values.items()}
    status = state.scratch(
        (1,), "physics_validation_status").view(cp.uint32)
    status[...] = cp.uint32(0xFFFFFFFF)
    driver._run_cumulus({}, state, cfg)
    assert int(status[0].item()) == 0
    for name in names:
        cp.testing.assert_array_equal(values[name], before[name])
    for value in values.values():
        value[...] = cp.float32(9.0)
    for name in names[:6]:
        cp.testing.assert_array_equal(driver.cu_rates[name], before[name])
    cp.testing.assert_array_equal(
        driver.cu_nca, before["nca_seconds"])
    cp.testing.assert_array_equal(
        driver.cu_pratec, before["pratec"])
    for value in values.values():
        value[...] = cp.float32(0.0)

    for name, label in zip(names, labels):
        values[name].flat[0] = cp.nan
        with pytest.raises(
                FloatingPointError,
                match=rf"{label} contains a non-finite value"):
            driver._run_cumulus({}, state, cfg)
        values[name].flat[0] = cp.float32(0.0)

    values[names[-1]].flat[0] = cp.inf
    values[names[0]].flat[0] = cp.nan
    with pytest.raises(
            FloatingPointError,
            match="cumulus rthcuten contains a non-finite value"):
        driver._run_cumulus({}, state, cfg)


@pytest.mark.gpu
@requires_gpu
def test_kessler_liquid_sr_overrides_cold_temperature_in_noah(monkeypatch):
    """WRF passes Kessler's SR=0 to Noah even when cold rain is present."""
    import cupy as cp
    import woof.core.physics as physics

    from woof.core.diagnostics import update_diagnostics
    from woof.core.microphysics import MicrophysicsDiagnostics

    state, cfg, driver = _full_state(mp_physics=1, bl_pbl_physics=0)
    shape = (cfg.ny, cfg.nx)
    # Make the temperature fallback unambiguously frozen.  Kessler's named
    # SR contract remains zero because its precipitation is liquid rain.
    state.thp[...] = cp.float32(-45.0)
    driver.accept_microphysics(MicrophysicsDiagnostics(
        rainnc=cp.full(shape, 2.0, cp.float32),
        rainncv=cp.full(shape, 0.4, cp.float32),
        sr=cp.zeros(shape, cp.float32)))
    seen = {}

    def capture(dev, params, dt, dzs, **kwargs):
        seen["sr"] = cp.asnumpy(dev["sr"]).copy()
        seen["rainbl"] = cp.asnumpy(dev["rainbl"]).copy()
        seen["frpcpn"] = kwargs["frpcpn"]

    monkeypatch.setattr(physics, "launch_noah", capture)
    update_diagnostics(state)
    prepared = physics._prepare_atmosphere(state)
    assert bool(cp.all(prepared["temperature"][0] <= cp.float32(273.15)))
    driver.compute(state, cfg)

    np.testing.assert_array_equal(seen["sr"], np.zeros(shape, np.float32))
    np.testing.assert_array_equal(
        seen["rainbl"], np.full(shape, 0.4, np.float32))
    assert seen["frpcpn"] is True


@pytest.mark.gpu
@requires_gpu
def test_ysu_seam_receives_full_similarity_denominators(monkeypatch):
    """The driver binds YSU's psim/psih to SFCLAY's FM/FH, never raw psi.

    WRF's PBL driver passes YSU the full similarity denominators
    ln(z/z0)-psi (module_pbl_driver.F:1228, ``PSIM=fm, PSIH=fhh``); YSU
    reconstructs z/L as br*fm**2/fh, which is 0/0 at neutral for the raw
    integrated psi corrections SFCLAY also exports.
    """
    import cupy as cp
    import woof.core.physics as physics

    from woof.core.diagnostics import update_diagnostics

    state, cfg, driver = _full_state()
    real_launch = physics.launch_ysu
    seen = {}

    def capture(*args, **kwargs):
        seen["psim"] = kwargs["psim"]
        seen["psih"] = kwargs["psih"]
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(physics, "launch_ysu", capture)
    update_diagnostics(state)
    driver.compute(state, cfg)

    f = driver.fields
    assert seen["psim"] is f["fm"]
    assert seen["psih"] is f["fh"]
    # Guard against the identity check going vacuous through aliasing: the
    # raw corrections and full denominators must actually disagree here.
    assert not bool(cp.allclose(f["fm"], f["psim"]))
    assert not bool(cp.allclose(f["fh"], f["psih"]))


@pytest.mark.gpu
@requires_gpu
def test_prepare_atmosphere_feeds_frozen_state_species_to_the_seams():
    """Prognostic qi/qs reach PBL and radiation; absent species stay zero."""
    import cupy as cp
    from woof.core.physics import _prepare_atmosphere

    state, _, _ = _full_state()
    assert getattr(state, "qi", None) is None
    zero_qi = _prepare_atmosphere(state)["qi"]
    assert not bool(zero_qi.any())
    zero_qs = _prepare_atmosphere(state)["qs"]
    assert not bool(zero_qs.any())

    state.qi = cp.full(state.p.shape, cp.float32(2.0e-5))
    state.qs = cp.full(state.p.shape, cp.float32(3.0e-5))
    got = _prepare_atmosphere(state)
    assert got["qi"] is state.qi
    assert got["qs"] is state.qs
    assert bool(got["qi"].any())
    assert bool(got["qs"].any())


@pytest.mark.gpu
@requires_gpu
def test_composed_full_physics_carries_rqiblten():
    """The production composer (radiation AND cumulus active) must carry
    the PBL rqi; the batch review caught _compose_tendencies dropping it
    whenever the non-identity composition path ran, so the real74
    configuration silently lost RQIBLTEN."""
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.physics import CumulusResult, RadiationResult

    def radiation(**kwargs):
        st = kwargs["state"]
        nz, ny, nx = st.p.shape
        return RadiationResult(
            cp.zeros((nz, ny, nx), cp.float32),
            cp.zeros((nz, ny, nx), cp.float32),
            cp.full((ny, nx), 500.0, cp.float32),
            cp.full((ny, nx), 300.0, cp.float32))

    def cumulus(**kwargs):
        st = kwargs["state"]
        nz, ny, nx = st.p.shape
        return CumulusResult(
            cp.zeros((nz, ny, nx), cp.float32),
            cp.zeros((nz, ny, nx), cp.float32),
            rainc=cp.zeros((ny, nx), cp.float32))

    state, cfg, driver = _full_state(
        dt=60.0, ra_physics=90, cu_physics=1, radt_minutes=12.0,
        cudt_minutes=5.0, radiation=radiation, cumulus=cumulus)
    nz = state.p.shape[0]
    # Height-decaying ice so YSU mixing has a real gradient to act on.
    state.qi = cp.ascontiguousarray(
        cp.float32(2.0e-5)
        * cp.exp(-cp.arange(nz, dtype=cp.float32) / cp.float32(4.0))
        [:, None, None] * cp.ones_like(state.p))
    update_diagnostics(state)
    got = driver.compute(state, cfg)
    rqi = got.scalar_for("qi")
    assert rqi is not None
    # Radiation/cumulus stubs contribute no qi, so the composed field is
    # exactly the PBL coupling -- and it must be non-vacuously nonzero.
    cp.testing.assert_array_equal(rqi, driver.pbl_tendencies.rqi)
    assert bool(cp.any(rqi != 0))


@pytest.mark.gpu
@requires_gpu
def test_noah_receives_wrf_layer_mid_sfcprs(monkeypatch):
    """Noah's SFCPRS is the lowest-layer mid pressure from interfaces,
    SFCPRS = 0.5*(P8W(kts)+P8W(kts+1)) (module_sf_noahdrv.F:795), not
    the half-level p_phy."""
    import cupy as cp
    import woof.core.physics as physics

    from woof.core.diagnostics import update_diagnostics

    state, cfg, driver = _full_state()
    seen = {}
    real_launch = physics.launch_noah

    def capture(fields, *args, **kwargs):
        seen["sfcprs"] = cp.asarray(fields["sfcprs"]).copy()
        seen["psfc"] = cp.asarray(fields["psfc"]).copy()
        return real_launch(fields, *args, **kwargs)

    monkeypatch.setattr(physics, "launch_noah", capture)
    update_diagnostics(state)
    driver.compute(state, cfg)

    atmosphere = physics._prepare_atmosphere(state)
    expected = cp.float32(0.5) * (atmosphere["p_interface"][0]
                                  + atmosphere["p_interface"][1])
    cp.testing.assert_array_equal(seen["sfcprs"], expected)
    # Under WRF's phy_prep hydrostatic build the lowest half level IS the
    # interface mean, so SFCPRS == pressure[0] identically (that is WRF's
    # own arrangement); pin the hydrostatic construction instead: the top
    # interface is exactly p_top and the column is strictly monotone
    # (module_big_step_utilities_em.F:4943-4970).
    cp.testing.assert_array_equal(seen["sfcprs"], atmosphere["pressure"][0])
    assert float(cp.abs(atmosphere["p_interface"][-1]
                        - cp.float32(state.p_top)).max()) == 0.0
    assert bool((atmosphere["p_interface"][:-1]
                 > atmosphere["p_interface"][1:]).all())
    cp.testing.assert_array_equal(seen["psfc"], atmosphere["p_interface"][0])


@pytest.mark.gpu
@requires_gpu
def test_olr_is_published_only_by_a_declaring_longwave_scheme():
    """OLR's presence is the statement "a TOA longwave producer ran".

    WRF's own OLR row is core (``Registry.EM_COMMON:1839`` is a ``misc``
    field with no package gate), so stock WRF writes it in every run.
    woof's radiation diagnostics are instead ABSENT when nothing produced
    them -- SWDOWN/GLW disappear with ``radiation_active``, XKMH/XKHH
    disappear with their operator -- and OLR follows that convention one
    step further: a shortwave-only pair and the surface-flux analytic
    proxy compute no top-of-atmosphere flux, and a zero for them would be
    a measured-looking number no scheme evaluated.
    """
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.physics import RadiationResult
    from woof.io import restart

    def surface_only(*, atmosphere, fields, state, cfg):
        zeros = cp.zeros_like(state.p)
        surface = cp.full(state.mup.shape, 300.0, cp.float32)
        return RadiationResult(zeros, zeros, surface, surface)

    def with_toa(*, atmosphere, fields, state, cfg):
        result = surface_only(atmosphere=atmosphere, fields=fields,
                              state=state, cfg=cfg)
        result.olr = cp.full(state.mup.shape, 237.5, cp.float32)
        return result

    with_toa.publishes_olr = True

    def declared_but_silent(**kwargs):
        return surface_only(**kwargs)

    declared_but_silent.publishes_olr = True

    # Radiation off: OLR is absent for the same reason GLW is.
    _, _, off = _full_state()
    assert off.olr is None
    assert "GLW" not in off.output_fields()
    assert "OLR" not in off.output_fields()

    # Radiation on, no TOA producer: the surface fluxes appear, OLR does not.
    state, cfg, driver = _full_state(
        nx=1, ny=1, ra_physics=90, radt_minutes=0.0, radiation=surface_only)
    update_diagnostics(state)
    driver.compute(state, cfg)
    assert {"SWDOWN", "GLW"} <= set(driver.output_fields())
    assert "OLR" not in driver.output_fields()
    assert driver.olr is None

    # A declaring scheme publishes it, and the pre-first-call value is the
    # zero WRF's own zero-initialised array carries into its t=0 frame.
    state, cfg, driver = _full_state(
        nx=1, ny=1, ra_physics=90, radt_minutes=0.0, radiation=with_toa)
    published = driver.output_fields()["OLR"]
    assert published.shape == (cfg.ny, cfg.nx)
    assert published.dtype == cp.float32
    assert float(cp.abs(published).max()) == 0.0
    update_diagnostics(state)
    driver.compute(state, cfg)
    # Filled in place, so the writer's frame keeps pointing at one buffer.
    assert driver.output_fields()["OLR"] is published
    assert float(published[0, 0]) == 237.5

    # A scheme that declares the flux and then withholds it is a broken
    # scheme, not a configuration in which OLR is undefined.
    state, cfg, driver = _full_state(
        nx=1, ny=1, ra_physics=90, radt_minutes=0.0,
        radiation=declared_but_silent)
    update_diagnostics(state)
    with pytest.raises(ValueError, match="publishes_olr"):
        driver.compute(state, cfg)

    # The last computed flux must survive a resume between radiation calls.
    # Its checkpoint-only namespace preserves that diagnostic without adding
    # it to the prognostic driver fields used by the tile sweep.
    assert "olr" in restart.DRIVER_CHECKPOINT_ONLY_ATTRS
    assert "olr" not in restart.DRIVER_REBUILT_ATTRS
    assert "olr" not in restart.DRIVER_SERIALIZED_ATTRS


@pytest.mark.gpu
@requires_gpu
def test_noah_surface_optics_feed_the_next_radiation_call():
    """WRF ordering exposes aged snow albedo/emissivity one STEPRA later."""
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.physics import RadiationResult

    seen = []

    def capture(*, atmosphere, fields, state, cfg):
        seen.append((cp.asnumpy(fields["albedo"]).copy(),
                     cp.asnumpy(fields["emiss"]).copy()))
        zeros = cp.zeros_like(state.p)
        surface = cp.full(state.mup.shape, 400.0, cp.float32)
        return RadiationResult(zeros, zeros, surface,
                               cp.full_like(surface, 330.0))

    state, cfg, driver = _full_state(
        nx=1, ny=1, snow=70.0, snow_depth=0.35,
        ra_physics=90, radt_minutes=0.0, radiation=capture)
    update_diagnostics(state)
    driver.compute(state, cfg)
    noah_albedo = cp.asnumpy(driver.fields["albedo"]).copy()
    noah_emiss = cp.asnumpy(driver.fields["emiss"]).copy()
    assert float(noah_albedo[0, 0]) > float(driver.fields["albbck"][0, 0])

    state.elapsed_seconds += cfg.dt
    update_diagnostics(state)
    driver.compute(state, cfg)
    np.testing.assert_array_equal(seen[1][0], noah_albedo)
    np.testing.assert_array_equal(seen[1][1], noah_emiss)


@pytest.mark.gpu
@requires_gpu
def test_t6a_single_column_rrtmgp_day_cycle_measured_gate():
    """Measured 39N/87W RRTMGP -> Noah -> YSU 24 h acceptance envelope."""
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics

    state, cfg, driver = _full_state(
        nx=1, ny=1, nz=24, dt=600.0, time_step_sound=4,
        mp_physics=10, ra_physics=4, radt_minutes=10.0, bldt=10.0,
        radiation_start_time=datetime(1974, 4, 3, 12),
        radiation_latitude=np.array([[39.0]]),
        radiation_longitude=np.array([[-87.0]]))
    tsk = []
    hpbl = []
    swdown = []
    glw = []
    for n in range(24 * 6):
        update_diagnostics(state)
        driver.compute(state, cfg)
        ysu = driver.last_ysu
        state.u += cp.float32(cfg.dt) * ysu["du"]
        state.v += cp.float32(cfg.dt) * ysu["dv"]
        state.thp += cp.float32(cfg.dt) * (
            ysu["dtheta"] + driver.rthratenlw + driver.rthratensw)
        state.qv[...] = cp.maximum(
            state.qv + cp.float32(cfg.dt) * ysu["dqv"], 0.0)
        state.qc[...] = cp.maximum(
            state.qc + cp.float32(cfg.dt) * ysu["dqc"], 0.0)
        state.elapsed_seconds += cfg.dt
        tsk.append(float(driver.fields["tsk"][0, 0]))
        hpbl.append(float(driver.fields["pblh"][0, 0]))
        swdown.append(float(driver.fields["swdown"][0, 0]))
        glw.append(float(driver.fields["glw"][0, 0]))

    for values in (tsk, hpbl, swdown, glw):
        assert np.isfinite(values).all()
    assert 240.0 <= min(tsk) <= max(tsk) <= 335.0
    assert max(tsk) - min(tsk) >= 10.0
    assert 5.0 <= int(np.argmax(tsk)) / 6.0 <= 20.0
    assert 800.0 <= max(swdown) <= 1000.0
    assert 5.0 <= int(np.argmax(swdown)) / 6.0 <= 7.0
    assert 300.0 <= min(glw) <= max(glw) <= 450.0
    assert 1000.0 <= max(hpbl) <= 2500.0
    assert driver.call_counts["radiation"] == 144
    assert driver.ysu_nan_guard_fires == 0
    assert bool(cp.isfinite(state.total_theta()).all())
    assert bool(cp.isfinite(state.qv).all()) and float(state.qv.min()) >= 0.0
    assert bool(cp.isfinite(driver.fields["smois"]).all())
    assert 0.019 <= float(driver.fields["smois"].min())
    assert float(driver.fields["smois"].max()) <= 1.0


@pytest.mark.gpu
@requires_gpu
def test_derived_snow_depth_column_survives_six_hour_diurnal_cycle():
    """Regression for the real74 SWE-only snowpack collapse."""
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.ingest.soil import preprocess_noah_soil

    soil_fields = {
        "LANDSEA": np.ones((1, 1)),
        "SKINTEMP": np.full((1, 1), 285.0),
        "ST000007": np.full((1, 1), 284.0),
        "ST007028": np.full((1, 1), 283.0),
        "ST028100": np.full((1, 1), 282.0),
        "ST100289": np.full((1, 1), 281.0),
        "TMN": np.full((1, 1), 280.0),
        "SM000007": np.full((1, 1), 0.30),
        "SM007028": np.full((1, 1), 0.30),
        "SM028100": np.full((1, 1), 0.30),
        "SM100289": np.full((1, 1), 0.30),
        "SNOW": np.full((1, 1), 70.0),
    }
    soil = preprocess_noah_soil(
        soil_fields, soil_type=np.full((1, 1), 6))
    assert float(soil.snow_depth[0, 0]) == pytest.approx(0.35)

    state, cfg, driver = _full_state(
        nx=1, ny=1, nz=24, dt=600.0, time_step_sound=4,
        snow=soil.snow_water, snow_depth=soil.snow_depth, bldt=10.0)
    tsk = []
    for n in range(6 * 6):
        hour = 8.0 + n / 6.0
        shortwave = max(0.0, 850.0 * np.sin(np.pi * (hour - 6.0) / 12.0))
        driver.set_forcing(swdown=shortwave, glw=300.0)
        update_diagnostics(state)
        driver.compute(state, cfg)
        ysu = driver.last_ysu
        state.u += cp.float32(cfg.dt) * ysu["du"]
        state.v += cp.float32(cfg.dt) * ysu["dv"]
        state.thp += cp.float32(cfg.dt) * ysu["dtheta"]
        state.qv[...] = cp.maximum(
            state.qv + cp.float32(cfg.dt) * ysu["dqv"], 0.0)
        state.qc[...] = cp.maximum(
            state.qc + cp.float32(cfg.dt) * ysu["dqc"], 0.0)
        state.elapsed_seconds += cfg.dt
        tsk.append(float(driver.fields["tsk"][0, 0]))

    assert np.isfinite(tsk).all()
    assert 200.0 <= min(tsk) <= max(tsk) <= 320.0
    assert driver.ysu_nan_guard_fires == 0


@pytest.mark.gpu
@requires_gpu
def test_moist_bubble_with_full_physics_remains_stable():
    import cupy as cp

    from woof.core.dycore import run_steps, stability_report
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import initialize_physics
    from woof.verify.cases import moist_bubble
    from woof.verify.cases.wk82 import wk82_sounding

    cfg = dataclasses.replace(
        moist_bubble.default_config(), sf_sfclay_physics=1,
        sf_surface_physics=2, bl_pbl_physics=1,
        mp_physics=10, ra_physics=4, radt=12.0)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = moist_bubble.build(cfg, coord, base)
    initialize_physics(
        state, cfg, landmask=1.0, tsk=301.0,
        soil_temperature=np.array([299.0, 298.0, 296.0, 294.0]),
        soil_moisture=0.30, liquid_moisture=0.30,
        ivgtyp=10, isltyp=6, vegfra=55.0, tmn=289.0,
        swdown=350.0, glw=310.0, pblh=600.0,
        radiation_start_time=datetime(1974, 4, 3, 12),
        radiation_latitude=np.full((cfg.ny, cfg.nx), 39.0),
        radiation_longitude=np.full((cfg.ny, cfg.nx), -87.0))

    n_steps = int(round(cfg.run_seconds / cfg.dt))
    w_max = qc_max = 0.0
    for _ in range(n_steps):
        run_steps(state, cfg, 1)
        w_max = max(w_max, float(state.w.max()))
        qc_max = max(qc_max, float(state.qc.max()))

    report = stability_report(state, cfg)
    assert not report["nan"]
    assert 5.0 < w_max < 40.0
    assert qc_max >= 1.0e-3
    assert all(bool(cp.isfinite(field).all())
               for field in (state.qv, state.qc, state.qr, state.qi, state.qs,
                             state.qg, state.nc, state.nr, state.ni, state.ns,
                             state.ng))
    assert min(float(state.qv.min()), float(state.qc.min()),
               float(state.qr.min()), float(state.qi.min()),
               float(state.qs.min()), float(state.qg.min())) >= 0.0
    assert state.elapsed_seconds == 3600.0
    assert state.physics.call_counts["sfclay"] == n_steps == 600
    assert state.physics.call_counts["noah"] == n_steps
    assert state.physics.call_counts["ysu"] == n_steps
    # WRF's default offset is MOD(ITIMESTEP, STEPRA)==1: calls at
    # 1,121,241,361,481 for this 600-step integration.
    assert state.physics.call_counts["radiation"] == 5


@pytest.mark.gpu
@requires_gpu
def test_prepare_atmosphere_builds_wrf_hydrostatic_pressures():
    """E2: the physics seams receive phy_prep's hydrostatic p_hyd/p_hyd_w
    (module_big_step_utilities_em.F:4943-4970), monotone by construction,
    while the EOS pressure stays on the state for microphysics/dycore
    (solve_em.F:3724) and exner/temperature remain WRF's pi_phy/t_phy."""
    import cupy as cp
    import woof.core.physics as physics

    from woof.core import constants as c
    from woof.core.diagnostics import update_diagnostics

    state, _, _ = _full_state()
    update_diagnostics(state)
    atmosphere = physics._prepare_atmosphere(state)

    # Independent float64 recomputation of the downward integration.
    nz, ny, nx = state.p.shape
    qtot = np.zeros((nz, ny, nx))
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        species = getattr(state, name, None)
        if species is not None:
            qtot += cp.asnumpy(species).astype(np.float64)
    chm = (cp.asnumpy(state.c1h)[:, None, None].astype(np.float64)
           * cp.asnumpy(state.total_mu())[None].astype(np.float64)
           + cp.asnumpy(state.c2h)[:, None, None].astype(np.float64))
    layer = (1.0 + qtot) * chm * cp.asnumpy(state.dnw)[:, None, None]
    expected = np.empty((nz + 1, ny, nx))
    expected[nz] = float(state.p_top)
    for k in range(nz - 1, -1, -1):
        expected[k] = expected[k + 1] - layer[k]
    np.testing.assert_allclose(cp.asnumpy(atmosphere["p_interface"]),
                               expected, rtol=2.0e-6)
    np.testing.assert_allclose(cp.asnumpy(atmosphere["pressure"]),
                               0.5 * (expected[:-1] + expected[1:]),
                               rtol=2.0e-6)
    # The physics feed is its own hydrostatic construction, no longer an
    # alias of the EOS field (for this at-rest balanced fixture the two
    # AGREE numerically -- the split only shows under nonhydrostatic
    # motion, which is exactly why WRF separates them).
    assert atmosphere["pressure"] is not state.p
    # ...while exner stays the EOS-based pi_phy. The independent scalar
    # WRF oracle uses REAL(4) division followed by its linked powf. CUDA's
    # power is a different implementation and cannot be the word referee.
    import ctypes
    import ctypes.util
    oracle = ctypes.CDLL(ctypes.util.find_library("m") or "msvcrt")
    powf = oracle.powf
    powf.argtypes = (ctypes.c_float, ctypes.c_float)
    powf.restype = ctypes.c_float
    ratio = cp.asnumpy(state.p) / np.float32(c.P0)
    stock_exner = np.array(
        [powf(float(x), float(np.float32(c.RCP))) for x in ratio.flat],
        np.float32).reshape(ratio.shape)
    from woof.wrf_exact import ENABLED as strict
    if strict:
        np.testing.assert_array_equal(
            cp.asnumpy(atmosphere["exner"]).view("u4"), stock_exner.view("u4"))
    else:
        # A default process keeps 2.9.1's array power (physics._physics_exner).
        cp.testing.assert_array_equal(
            atmosphere["exner"], (state.p / cp.float32(c.P0)) ** cp.float32(c.RCP))
