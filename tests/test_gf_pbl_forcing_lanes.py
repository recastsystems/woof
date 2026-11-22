"""GF's RTHBLTEN/RQVBLTEN lanes on the ARW path, with a BOUND driver.

THE BREAKAGE THIS FILE PREVENTS, named twice because it happened twice:

1.  The Grell-Freitas adapter reads its boundary-layer forcing off the
    bound :class:`~woof.core.physics.PhysicsDriver`
    (``gf_rthblten``/``gf_rqvblten``).  Every other GF test in this suite
    drives the adapter with ``driver=None`` -- the parity harnesses launch
    the kernel directly, and ``test_gf_engine_smoke`` asserts the binding
    exists without ever reading a lane's VALUE.  So the whole family stayed
    553-passed green across the merge that switched those two GFDRV inputs
    from hard zeros to real rates, because the tested path reads zeros
    either way.  A real ARW forecast does not: it binds the driver, the PBL
    slot runs before cumulus in the same ``compute()``, and GFDRV integrates
    the boundary layer's actual heating and moistening.  This file exercises
    the bound path and asserts the lanes are non-zero AND that GF's answer
    moves when they are.

2.  Retention was written into ``_run_ysu`` alone, so the fix landed for one
    PBL scheme out of five.  A run pairing GF with MYJ, MYNN, Shin-Hong or
    SASE went on feeding the cumulus scheme zeros -- wrong by exactly the
    argument that made the YSU change right.  The parametrised case below
    covers every scheme the engine ships in the PBL slot, so a sixth scheme
    added later cannot quietly reintroduce the hole.

WRF's cumulus driver hands RTHBLTEN/RQVBLTEN to GFDRV without asking which
scheme filled them (module_cumulus_driver.F), and the arrays persist between
PBL calls, so last-computed retention is WRF's own contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

cp = pytest.importorskip("cupy")


#: A moist, unstable, warm-surfaced column set that GF actually triggers on.
#: Chosen by measurement, not by taste: with the drier/cooler soundings the
#: rest of the suite uses, every column is rejected by the trigger and the
#: differential assertion below would compare zero against zero -- an
#: instrument that reports PASS while measuring nothing.
#: ``_assert_not_vacuous`` keeps that failure mode from being silent.
_THETA_SURFACE_K = 296.0
_THETA_LAPSE_K_PER_M = 0.004
_QV_SURFACE = 0.020
_QV_SCALE_HEIGHT_M = 3500.0
_TSK_K = 305.0

_BASE_CONFIG = dict(
    nx=8, ny=8, nz=40, dx=12000.0, dy=12000.0, ztop=18000.0,
    dt=60.0, run_seconds=0.0, moist=True, mp_physics=10,
    bldt=0.0, cu_physics=3, cu_gf_version="wrf_461", cudt_minutes=0.0)


def _pbl_cases():
    """Every scheme the driver dispatches into the PBL slot.

    Each entry carries the extra selectors that scheme's own admission
    rules demand -- the Eta surface layer and a soil column for MYJ, the
    MYNN surface layer for MYNN, the closure switches for SASE.  They are
    the pairing requirements, not tuning.
    """
    from woof.config import SASE_PBL_SCHEME

    return [
        pytest.param(dict(bl_pbl_physics=1, sf_sfclay_physics=91),
                     id="ysu"),
        pytest.param(dict(bl_pbl_physics=2, sf_sfclay_physics=2,
                          sf_surface_physics=2, num_soil_layers=4),
                     id="myj"),
        pytest.param(dict(bl_pbl_physics=5, sf_sfclay_physics=5),
                     id="mynn"),
        pytest.param(dict(bl_pbl_physics=11, sf_sfclay_physics=91),
                     id="shinhong"),
        pytest.param(dict(bl_pbl_physics=SASE_PBL_SCHEME,
                          sf_sfclay_physics=91, km_opt=0,
                          khdif=0.0, kvdif=0.0),
                     id="sase"),
    ]


def _make_state(cfg):
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord,
        lambda z: _THETA_SURFACE_K + _THETA_LAPSE_K_PER_M * np.asarray(z),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    return init_moist_balanced(
        cfg, coord, base,
        lambda z: _QV_SURFACE * np.exp(-np.asarray(z) / _QV_SCALE_HEIGHT_M))


def _driver_for(cfg):
    from woof.core.physics import (
        DECLARED_CONSTANT_GLW_WM2, initialize_physics)

    state = _make_state(cfg)
    # Declared constant downward radiation: this file's subject is the PBL
    # -> cumulus forcing seam, not the sky, and MYJ's Noah pairing reads GLW
    # every surface step.  Declaring it is the carrier contract's own
    # remedy; inventing it is what the contract refuses.
    driver = initialize_physics(
        state, cfg, landmask=1.0, tsk=_TSK_K,
        glw=DECLARED_CONSTANT_GLW_WM2, swdown=0.0)
    return state, driver


class _LanesWithheld:
    """The driver as the PRE-UPGRADE adapter saw it: no forcing lanes.

    Forwards every attribute to the real driver except the four GF
    auxiliary lanes, which read ``None`` -- and ``None`` is exactly what
    ``GrellFreitas.held_lane`` turns into zeros.  This is the control arm:
    the same engine, the same state, the same kernel, with only the lanes
    withheld.
    """

    _WITHHELD = ("gf_rthblten", "gf_rqvblten",
                 "gf_rthdynten", "gf_rqvdynten")

    def __init__(self, driver):
        object.__setattr__(self, "_driver", driver)

    def __getattr__(self, name):
        if name in _LanesWithheld._WITHHELD:
            return None
        return getattr(object.__getattribute__(self, "_driver"), name)


def _lane_withholding_adapter():
    """A GF adapter that can never see the driver's forcing lanes.

    The withholding MUST live in ``bind_driver``, not in a one-shot
    re-bind after construction: ``PhysicsDriver._run_cumulus`` calls
    ``bind(self)`` at the top of EVERY due cumulus call, so a proxy
    installed from outside is overwritten by the real driver on the first
    step and the control arm silently becomes a second copy of the
    treatment arm.  That mistake produced an exact-0.0 delta while this
    file was being written, which is the A/B law's own signature for "the
    experiment never ran".
    """
    from woof.core.gf import GrellFreitas

    adapter = GrellFreitas()
    bind = adapter.bind_driver

    def withhold(driver):
        bind(_LanesWithheld(driver))

    # Preserve the production adapter's type and native-output admission.
    # A subclass changes that admission as well as hiding the lanes, so it
    # is not a control for the four forcing arguments alone.
    adapter.bind_driver = withhold
    return adapter


def _stats(array):
    flat = array.astype(cp.float64).ravel()
    return (float(cp.max(cp.abs(flat))),
            float(cp.sqrt(cp.mean(flat * flat))))


def _assert_not_vacuous(rate, rainc):
    """Refuse a PASS that measured nothing.

    An all-zero cumulus response means the trigger rejected every column,
    and then 'the two arms differ' can never be true and 'the two arms
    agree' means nothing.  A/B law: the delta is only evidence once the
    experiment is shown to have run.
    """
    assert float(cp.max(cp.abs(rate))) > 0.0, (
        "GF produced no convective heating anywhere, so this case cannot "
        "measure whether the forcing lanes reached the kernel")
    assert float(cp.max(rainc)) > 0.0, (
        "GF accumulated no convective precipitation, so this case cannot "
        "measure whether the forcing lanes reached the kernel")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("selectors", _pbl_cases())
def test_every_pbl_scheme_hands_grell_freitas_its_real_forcing(selectors):
    """The lanes are BOUND and NON-ZERO for every scheme in the PBL slot.

    RED before the retention generalisation for four of the five schemes
    (only YSU retained), and RED for all five before the GF seam merge.
    """
    from woof.config import RunConfig, validate_run_config

    cfg = validate_run_config(RunConfig(**_BASE_CONFIG, **selectors))
    state, driver = _driver_for(cfg)
    driver.compute(state, cfg)

    # The PBL slot ran, and it ran before cumulus in the same compute().
    assert driver.call_counts["ysu"] == 1
    assert driver.call_counts["cumulus"] == 1
    # The adapter is bound to THIS driver -- the coverage gap is a GF
    # adapter with driver=None, which reads zeros no matter what the PBL
    # computed.
    assert driver.cumulus_callable._driver is driver

    for name in ("gf_rthblten", "gf_rqvblten"):
        lane = getattr(driver, name)
        assert lane is not None, (
            f"{name} is None, so GFDRV received hard zeros for its "
            f"boundary-layer forcing under bl_pbl_physics="
            f"{cfg.bl_pbl_physics}")
        assert lane.shape == state.p.shape
        assert bool(cp.isfinite(lane).all()), name
        assert float(cp.max(cp.abs(lane))) > 0.0, (
            f"{name} is identically zero, which is what the pre-upgrade "
            f"adapter fed; the PBL slot computed real rates this step")

    # The advective pair is BOUND to the dycore's own export buffers from
    # driver construction (task #243 closed the last empty lane).  This
    # file used to assert both were None; that assertion was correct and
    # is now false, and the replacement keeps the same job -- naming what
    # GFDRV's RTHFTEN/RQVFTEN inputs are actually connected to, so a
    # regression that unbinds them is noticed here.
    #
    # They are still all-zero at this point and that is the CORRECT value:
    # ``compute()`` alone advances no dynamics, so no step has exported
    # anything yet.  ``test_the_arw_step_fills_the_advective_lanes`` below
    # runs the real dycore and proves the lane arrives non-zero.
    assert driver.gf_rthdynten is state.rthften
    assert driver.gf_rqvdynten is state.rqvften
    for name in ("gf_rthdynten", "gf_rqvdynten"):
        lane = getattr(driver, name)
        assert lane is not None, (
            f"{name} is None, so GFDRV received hard zeros for its "
            f"advective forcing on the ARW path")
        assert lane.shape == state.p.shape
        assert bool(cp.isfinite(lane).all()), name


@pytest.mark.gpu
@requires_gpu
def test_the_arw_step_fills_the_advective_lanes():
    """The other half of the four-lane set, proven through the real dycore.

    The cell above shows the advective lanes are BOUND; this one shows a
    real ARW step FILLS them, non-zero and finite, so the four inputs
    GFDRV sums are all live on this path.  The full contract -- that the
    export is pure advection, the uncoupling, the restart classification --
    lives in tests/test_dycore_advective_forcing_export.py; what belongs
    here is that GF's own four-lane set is complete.
    """
    import numpy as np

    from woof.config import RunConfig, validate_run_config
    from woof.core import dycore

    cfg = validate_run_config(RunConfig(
        **_BASE_CONFIG, bl_pbl_physics=1, sf_sfclay_physics=91))
    state, driver = _driver_for(cfg)
    nz, ny, nx = state.p.shape
    two_pi = 2.0 * np.pi
    x = np.arange(nx, dtype=np.float64)[None, None, :]
    y = np.arange(ny, dtype=np.float64)[None, :, None]
    # A divergent flow over a horizontally structured scalar field: a
    # uniform state advects nothing, and this cell would then measure zero
    # while the export worked (the instrument-validation rule).
    state.u[...] = cp.asarray(np.broadcast_to(
        5.0 + 2.0 * np.cos(two_pi * np.arange(nx + 1)[None, None, :]
                           / max(nx, 1)),
        (nz, ny, nx + 1)).copy(), dtype=cp.float32)
    wave = np.sin(two_pi * x / max(nx, 1)) * np.cos(two_pi * y / max(ny, 1))
    state.thp += cp.asarray(
        np.broadcast_to(0.75 * wave, (nz, ny, nx)).copy(), dtype=cp.float32)
    state.qv += cp.asarray(
        np.broadcast_to(1.0e-3 * wave, (nz, ny, nx)).copy(),
        dtype=cp.float32)
    cp.maximum(state.qv, cp.float32(0.0), out=state.qv)

    dycore.step(state, cfg)

    for name in ("gf_rthdynten", "gf_rqvdynten"):
        lane = getattr(driver, name)
        assert bool(cp.isfinite(lane).all()), name
        assert float(cp.max(cp.abs(lane))) > 0.0, (
            f"{name} is identically zero after a real ARW step, so GFDRV "
            f"is back to the hard zeros it was fed before task #243")


@pytest.mark.gpu
@requires_gpu
def test_the_forcing_lanes_change_what_grell_freitas_computes():
    """Feeding the lanes is a NUMERICS change, and here is its size.

    Two runs of the same engine from the same state through the same
    kernel, differing only in whether the driver's PBL forcing lanes are
    visible to the adapter.  RED against any version that feeds zeros:
    there the two arms are bit-identical and the delta is exactly 0.
    """
    from woof.config import RunConfig, validate_run_config

    cfg = validate_run_config(RunConfig(
        **_BASE_CONFIG, bl_pbl_physics=1, sf_sfclay_physics=91))

    state_fed, driver_fed = _driver_for(cfg)
    driver_fed.compute(state_fed, cfg)
    rate_fed = driver_fed.cu_rates["rthcuten"].copy()
    qv_fed = driver_fed.cu_rates["rqvcuten"].copy()
    rainc_fed = driver_fed.rainc.copy()

    state_zero, driver_zero = _driver_for(cfg)
    driver_zero.cumulus_callable = _lane_withholding_adapter()
    assert type(driver_zero.cumulus_callable) is type(driver_fed.cumulus_callable)
    driver_zero.compute(state_zero, cfg)
    # INSTRUMENT VALIDATION, before the delta is read: the control arm
    # really did withhold the lanes for the whole step.  Without this the
    # test can pass while measuring the treatment against itself.
    assert driver_zero.gf_rthblten is not None, (
        "the PBL slot did not retain its rates, so the control arm is not "
        "a control -- it is an unfixed driver")
    assert driver_zero.cumulus_callable._driver.gf_rthblten is None, (
        "the control arm's adapter can see the forcing lanes, so both arms "
        "ran the same numerics and any delta below is meaningless")
    rate_zero = driver_zero.cu_rates["rthcuten"].copy()
    rainc_zero = driver_zero.rainc.copy()

    # Instrument validation, both arms, before any delta is read.
    _assert_not_vacuous(rate_fed, rainc_fed)
    _assert_not_vacuous(rate_zero, rainc_zero)
    # The lanes the fed arm actually carried.
    lane_absmax, _ = _stats(driver_fed.gf_rthblten)
    assert lane_absmax > 0.0

    # THE GATE.  A version that feeds zeros produces rate_fed == rate_zero
    # and every number below is 0.0.
    delta = rate_fed - rate_zero
    delta_absmax = float(cp.max(cp.abs(delta)))
    rainc_delta_absmax = float(cp.max(cp.abs(rainc_fed - rainc_zero)))
    assert delta_absmax > 0.0, (
        "withholding the PBL forcing lanes left GF's heating rate "
        "bit-identical: the adapter is not reading them, so this "
        "configuration is running the pre-upgrade numerics")
    assert rainc_delta_absmax > 0.0, (
        "withholding the PBL forcing lanes left GF's convective "
        "precipitation bit-identical")

    # RECORDED, measured on this fixture at the tip that introduced the
    # lanes.  The band is wide on purpose -- it is a magnitude gate, not a
    # bitwise one, because FP32 convective closures are not bit-portable
    # across cards.  What it catches is the two failures that matter: a
    # lane silently reverting to zeros (delta collapses to 0) and a lane
    # arriving in the wrong units or the wrong variable (delta explodes).
    recorded_rate_absmax = 2.6395e-03      # K s-1, fed arm
    recorded_delta_absmax = 2.6741e-06     # default K s-1, fed minus withheld
    from woof.wrf_exact import ENABLED
    if ENABLED:
        import json
        from pathlib import Path
        # Independent stock GFDRV, compiled with the scalar WRF flags,
        # matches all eight outputs of both captured arms (40,960 words).
        # Its response differs from the original default arithmetic pin.
        stock = json.loads((Path(__file__).parent / "data/oracles/gf/"
                            "gf-pbl-forcing-strict.json").read_text())
        assert all(row["unequal"] == 0 for row in stock["rows"])
        recorded_delta_absmax = stock["reference_theta_delta"]
    recorded_lane_absmax = 7.9803e-04      # K s-1, retained RTHBLTEN
    for measured, recorded, label in (
            (float(cp.max(cp.abs(rate_fed))), recorded_rate_absmax,
             "rthcuten absmax"),
            (delta_absmax, recorded_delta_absmax, "rthcuten delta absmax"),
            (lane_absmax, recorded_lane_absmax, "gf_rthblten absmax")):
        assert 0.25 * recorded <= measured <= 4.0 * recorded, (
            f"{label} = {measured:.5e}, recorded {recorded:.5e}")

    # The moisture lane moves too, so this is not a heating-only artefact.
    qv_zero = driver_zero.cu_rates["rqvcuten"]
    assert float(cp.max(cp.abs(qv_fed - qv_zero))) > 0.0


@pytest.mark.gpu
@requires_gpu
def test_a_driverless_grell_freitas_still_feeds_zeros():
    """The documented escape hatch stays open, and stays documented.

    ``held_lane`` returns zeros for an unbound adapter; that is what the
    parity harnesses rely on and what every pre-upgrade caller got.  This
    pins it so the lane plumbing cannot start requiring a driver.
    """
    from woof.config import RunConfig, validate_run_config
    from woof.core.gf import GrellFreitas
    from woof.core.physics import _prepare_atmosphere

    cfg = validate_run_config(RunConfig(
        **_BASE_CONFIG, bl_pbl_physics=1, sf_sfclay_physics=91))
    state, driver = _driver_for(cfg)
    driver.compute(state, cfg)

    unbound = GrellFreitas()
    assert unbound._driver is None
    result = unbound(atmosphere=_prepare_atmosphere(state),
                     fields=driver.fields, state=state, cfg=cfg)
    assert bool(cp.isfinite(result.rthcuten).all())
