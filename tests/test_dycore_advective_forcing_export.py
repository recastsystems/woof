"""The ARW dycore's pure-advective theta/qv forcing pair (RTHFTEN/RQVFTEN).

THE BREAKAGE THIS FILE PREVENTS, in three parts.

1.  **The lane was empty.**  WRF's GFDRV builds its forced states ``tn``/
    ``qo`` from four forcing inputs; ``tests/test_gf_pbl_forcing_lanes.py``
    closed the boundary-layer pair for every PBL scheme, and the MPAS
    driver exports the advective pair, but nothing in the ARW dycore
    assigned ``gf_rthdynten``/``gf_rqvdynten`` -- so a real ARW Grell-
    Freitas forecast fed the scheme hard zeros for RTHFTEN/RQVFTEN while
    WRF fed it the step's advection.  The cells below run the REAL dycore
    step and assert the lanes arrive non-zero, finite, and bound to the
    driver the cumulus adapter reads.

2.  **The double-count trap.**  ``module_cumulus_driver.F:867`` pre-folds
    ``RTHRATEN + RTHBLTEN`` into ``RTHFTEN`` for ``G3SCHEME`` and
    ``NTIEDTKESCHEME`` and deliberately NOT for ``GFSCHEME``; GF sums the
    three lanes itself (``gf.cu:4428``).  An export taken one line later in
    ``dycore.step`` -- after ``physics_tendencies.add_to_slow`` or after
    ``add_h_diabatic_tendency`` -- would hand GF the radiative, boundary-
    layer and latent heating a second time.  The zero-wind cell below is
    the instrument: pure advection is exactly zero there, so ANY contamination
    from the physics or h_diabatic slots shows up as a non-zero export while
    those slots are loudly non-zero.

3.  **The coupling.**  ``rth_t`` is WRF-coupled and carries an extra
    ``1/msfty``; GF wants an uncoupled K s-1 rate, exactly what
    ``h_diabatic`` is.  The round-trip cell pins the export as the precise
    inverse of ``dycore.add_h_diabatic_tendency`` including the map-factor
    branch, so a run over real terrain cannot silently feed GF a
    mass-coupled number that merely looks like a rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

cp = pytest.importorskip("cupy")


#: The #237 fixture: a moist, unstable, warm-surfaced column set GF triggers
#: on.  Shared deliberately -- the two files measure the same seam from the
#: two sides, and a drift between the soundings would make the recorded
#: magnitudes here incomparable with the ones recorded there.
_THETA_SURFACE_K = 296.0
_THETA_LAPSE_K_PER_M = 0.004
_QV_SURFACE = 0.020
_QV_SCALE_HEIGHT_M = 3500.0
_TSK_K = 305.0

_BASE_CONFIG = dict(
    nx=8, ny=8, nz=40, dx=12000.0, dy=12000.0, ztop=18000.0,
    dt=60.0, run_seconds=0.0, moist=True, mp_physics=10,
    bldt=0.0, cu_physics=3, cudt_minutes=0.0,
    # The column this test was built on convects under GFDRV; WRF 4.8.0 GFL
    # (the default since 3.0.0) reads the same pair but does not trigger here.
    cu_gf_version="wrf_461",
    bl_pbl_physics=1, sf_sfclay_physics=91)


def _config(**overrides):
    from woof.config import RunConfig, validate_run_config

    merged = dict(_BASE_CONFIG)
    merged.update(overrides)
    return validate_run_config(RunConfig(**merged))


def _state(cfg):
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

    state = _state(cfg)
    driver = initialize_physics(
        state, cfg, landmask=1.0, tsk=_TSK_K,
        glw=DECLARED_CONSTANT_GLW_WM2, swdown=0.0)
    return state, driver


def _sheared_flow(state):
    """A flow AND a theta/qv field with real horizontal structure.

    Both halves are needed and the reason is worth recording: the balanced
    initial state is horizontally uniform, and a wind that varies only
    across its own component's direction has zero divergence -- so a
    "sheared" flow over a uniform field advects exactly nothing and every
    assertion below would read zero while the export worked perfectly.
    Instrument first: give the scalars a horizontal wave and the flow a
    divergent part, then the advective tendency is genuinely non-zero.
    """
    nz, ny, nx = state.p.shape
    z = np.arange(nz, dtype=np.float64)[:, None, None]
    y = np.arange(ny, dtype=np.float64)[None, :, None]
    x = np.arange(nx, dtype=np.float64)[None, None, :]
    xf = np.arange(nx + 1, dtype=np.float64)[None, None, :]
    yf = np.arange(ny + 1, dtype=np.float64)[None, :, None]
    two_pi = 2.0 * np.pi

    u = (4.0 + 0.35 * z + 1.5 * np.sin(two_pi * y / max(ny - 1, 1))
         + 2.0 * np.cos(two_pi * xf / max(nx, 1)))
    state.u[...] = cp.asarray(
        np.broadcast_to(u, (nz, ny, nx + 1)).copy(), dtype=cp.float32)
    v = (1.0 + 0.9 * np.cos(two_pi * x / max(nx - 1, 1))
         + 1.2 * np.sin(two_pi * yf / max(ny, 1)))
    state.v[...] = cp.asarray(
        np.broadcast_to(v, (nz, ny + 1, nx)).copy(), dtype=cp.float32)

    wave = (np.sin(two_pi * x / max(nx, 1))
            * np.cos(two_pi * y / max(ny, 1)))
    state.thp += cp.asarray(
        np.broadcast_to(0.75 * wave, (nz, ny, nx)).copy(), dtype=cp.float32)
    state.qv += cp.asarray(
        np.broadcast_to(1.0e-3 * wave, (nz, ny, nx)).copy(),
        dtype=cp.float32)
    cp.maximum(state.qv, cp.float32(0.0), out=state.qv)


def _absmax(array) -> float:
    return float(cp.max(cp.abs(array.astype(cp.float64))))


# ---------------------------------------------------------------------------
# Registration: the pair is priced, classified and gated wherever state is.
# ---------------------------------------------------------------------------

def test_the_pair_is_allocated_only_where_a_cumulus_scheme_reads_it():
    """A run whose cumulus scheme never reads the lane pays nothing.

    The predicate is a TABLE of scheme ids
    (:data:`woof.config.CUMULUS_ADVECTIVE_FORCING_SCHEMES`), so admitting
    G3/GD/NTiedtke later is one entry rather than a second code path --
    and a Kain-Fritsch or cumulus-off run keeps its existing VRAM
    projection to the byte.
    """
    from woof.config import CUMULUS_ADVECTIVE_FORCING_SCHEMES
    from woof.core import preflight as pf

    assert 3 in CUMULUS_ADVECTIVE_FORCING_SCHEMES
    mass = (_BASE_CONFIG["nz"], _BASE_CONFIG["ny"], _BASE_CONFIG["nx"])
    gf = pf.state_array_shapes(_config())
    assert gf["rthften"] == mass and gf["rqvften"] == mass
    for cu_physics in (0, 1):
        other = pf.state_array_shapes(_config(cu_physics=cu_physics))
        assert "rthften" not in other and "rqvften" not in other, cu_physics
    dry = pf.state_array_shapes(
        _config(cu_physics=0, moist=False, mp_physics=0, bl_pbl_physics=0,
                sf_sfclay_physics=0, bldt=0.0))
    assert "rthften" not in dry and "rqvften" not in dry


def test_the_pair_is_serialized_state_not_rebuilt_state():
    """Written at stage 1 of step N, read at the TOP of step N+1.

    That is the h_diabatic lifecycle exactly: nothing between a resume and
    the first cumulus call can refill it, because the producer is the
    dycore stage that has not run yet.  The MPAS lanes are REBUILT because
    their caller refills them inside every ``run_phase1``; once an ARW
    producer exists that argument no longer covers this pair.
    """
    from woof.io import restart
    from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS

    for name in ("rthften", "rqvften"):
        assert name in STATE_SERIALIZED_ATTRS, name
        assert restart.classify_state_attr(name) == "serialize", name
        assert name not in restart.STATE_REBUILT_ATTRS, name


def test_the_health_gate_reads_the_pair_as_a_held_tendency():
    """A rate, not a field: no physical bound, finiteness enforced."""
    from woof.core.health import rule_for_field

    for name in ("rthften", "rqvften"):
        rule = rule_for_field(name)
        assert rule.status_class == "held_tendency", name
        assert rule.lower is None and rule.upper is None, name
        assert rule_for_field(
            f"state.{name}").status_class == "held_tendency", name


def test_a_prepared_tree_from_an_earlier_build_still_runs():
    """The pair is COLD-START ZERO in a prepared cache, so it may be absent.

    THE BREAKAGE.  A prepared tree is a user artifact people keep and
    re-run; its cache carries the restart contract's state inventory, and
    the reader compares that inventory to the active config's.  Adding two
    names to the contract would make every ``cu_physics = 3`` prepared tree
    on disk fail with "prepared cache state inventory differs from the
    active config" -- a refusal naming neither the change nor a remedy,
    for a difference that carries no information.

    And it carries none because a prepared cache IS the t = 0 state: no
    dynamics step has run, so RTHFTEN/RQVFTEN are zero by construction,
    which is also WRF's own start_em cold start.  A cache missing exactly
    that pair restores it as the zeros it would have stored.

    A CHECKPOINT IS THE OPPOSITE and must stay so: it is mid-trajectory,
    the pair is genuinely non-zero there, and dropping it changes the
    forecast -- so the restart reader refuses, by name, and this
    tolerance must not be copied to it.
    """
    from woof.ingest.prepared_cache import (
        PreparedCacheMismatchError, reconcile_cached_state_inventory)
    from woof.io.restart import ADVECTIVE_FORCING_STATE

    expected = ["u", "v", "thp", "qv", "h_diabatic", *ADVECTIVE_FORCING_STATE]
    earlier = ["u", "v", "thp", "qv", "h_diabatic"]

    assert reconcile_cached_state_inventory(expected, expected) == expected
    assert reconcile_cached_state_inventory(earlier, expected) == earlier

    # A cache from a genuinely different configuration is still refused,
    # and so is one carrying a name this config does not have.
    with pytest.raises(PreparedCacheMismatchError) as missing:
        reconcile_cached_state_inventory(["u", "v", "thp"], expected)
    assert "qv" in str(missing.value)
    assert "h_diabatic" in str(missing.value)
    with pytest.raises(PreparedCacheMismatchError) as extra:
        reconcile_cached_state_inventory([*expected, "qh"], expected)
    assert "qh" in str(extra.value)


@pytest.mark.gpu
@requires_gpu
def test_the_health_census_covers_the_pair():
    """A NaN in the export must trip the same gate every carrier does."""
    from woof.core import health

    cfg = _config()
    state, _ = _driver_for(cfg)
    names = {field.name for field in
             health.collect_state_fields(state, backend="gpu")}
    assert {"rthften", "rqvften"} <= names


# ---------------------------------------------------------------------------
# The uncoupling, pinned as the exact inverse of the h_diabatic coupling.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_the_export_is_the_exact_inverse_of_the_h_diabatic_coupling():
    """``rth_t * msft / (c1h*mut + c2h)`` round-trips through the coupler.

    Map factors ON, because that is the branch a real projected domain
    takes and the branch a "the numbers looked like a rate" export would
    silently get wrong.
    """
    from woof.core.dycore import (
        add_h_diabatic_tendency, capture_advective_theta_forcing)

    cfg = _config()
    state, _ = _driver_for(cfg)
    nz, ny, nx = state.p.shape
    rng = np.random.default_rng(20260820)
    original = cp.asarray(
        rng.normal(0.0, 3.0e2, (nz, ny, nx)), dtype=cp.float32)
    state.rth_t[...] = original
    state.msft[...] = cp.asarray(
        0.92 + 0.15 * rng.random((ny, nx)), dtype=cp.float32)
    state.has_msf = True

    capture_advective_theta_forcing(state)
    assert bool(cp.isfinite(state.rthften).all())
    # An uncoupled K/s rate is O(1e-3..1e-1), not the O(1e2) coupled number.
    assert _absmax(state.rthften) < 1.0e-1 * _absmax(original)

    # The round trip: hand the export back to the coupler and the coupled
    # tendency comes out again.
    state.h_diabatic[...] = state.rthften
    state.rth_t[...] = 0
    add_h_diabatic_tendency(state)
    residual = _absmax(state.rth_t - original) / _absmax(original)
    assert residual < 1.0e-5, (
        f"the export is not the inverse of the coupling: relative "
        f"round-trip residual {residual:.3e}")


# ---------------------------------------------------------------------------
# The real dycore step: the lane arrives, and it is pure advection.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_a_real_step_feeds_grell_freitas_a_nonzero_finite_advective_pair():
    """RED at every tip before this lane: both lanes were ``None``."""
    from woof.core import dycore

    cfg = _config()
    state, driver = _driver_for(cfg)
    _sheared_flow(state)

    assert state.rthften is not None and state.rqvften is not None
    # The adapter reads the DRIVER attribute; the dycore writes the STATE
    # buffer.  If those are two different arrays the export never lands.
    assert driver.gf_rthdynten is state.rthften
    assert driver.gf_rqvdynten is state.rqvften
    assert _absmax(state.rthften) == 0.0, "not zero before the first step"

    dycore.step(state, cfg)

    for name in ("rthften", "rqvften"):
        lane = getattr(state, name)
        assert lane.shape == state.p.shape
        assert bool(cp.isfinite(lane).all()), name
        assert _absmax(lane) > 0.0, (
            f"{name} is identically zero after a real step, so GFDRV is "
            f"still receiving the hard zeros this lane exists to replace")
    # Still the same buffers: the driver binding must survive the step.
    assert driver.gf_rthdynten is state.rthften
    assert driver.gf_rqvdynten is state.rqvften


@pytest.mark.gpu
@requires_gpu
def test_the_export_is_pure_advection_and_not_the_physics_slots():
    """THE TRAP CELL.  Zero wind, loud physics, zero export.

    With ``u = v = w = 0`` the stage transport fluxes vanish, so the pure
    advective tendency of theta and qv is exactly zero.  Every other
    contribution to ``rth_t`` is deliberately loud here: the physics
    composition ran, and ``h_diabatic`` is preloaded with a large retained
    heating.  An export taken after ``physics_tendencies.add_to_slow`` or
    after ``add_h_diabatic_tendency`` would carry them and GF would
    integrate the boundary layer and the latent heating TWICE
    (module_cumulus_driver.F:867 does that fold for G3/NTiedtke, never for
    GFSCHEME).
    """
    from woof.core import dycore

    cfg = _config()
    state, driver = _driver_for(cfg)
    state.u[...] = 0
    state.v[...] = 0
    state.w[...] = 0
    # A retained heating far larger than anything one step produces, so a
    # fold would be unmistakable rather than marginal.
    preloaded = 5.0e-2                                     # K s-1
    state.h_diabatic[...] = cp.float32(preloaded)

    dycore.step(state, cfg)

    # INSTRUMENT VALIDATION, before the verdict: the contaminating slots
    # really were non-zero on this step.  Without this the cell passes by
    # measuring nothing.
    assert driver.gf_rthblten is not None
    assert _absmax(driver.gf_rthblten) > 0.0, (
        "the PBL slot produced nothing, so this cell cannot tell a pure "
        "advective export from one that folded the boundary layer in")
    assert _absmax(driver.tendencies.rtheta) > 0.0

    # THE VERDICT.  Pure advection with no flow is zero.
    assert _absmax(state.rthften) == 0.0, (
        "the theta export is non-zero with no flow: it is carrying a "
        "physics or h_diabatic contribution, and GF will double-count it")
    assert _absmax(state.rqvften) == 0.0, (
        "the qv export is non-zero with no flow: it is carrying a source "
        "term rather than the advective tendency")


@pytest.mark.gpu
@requires_gpu
def test_feeding_the_advective_pair_changes_what_grell_freitas_computes():
    """The magnitude, measured the way #237 measured the PBL pair.

    Two runs of the same engine from the same state through the same
    kernel, differing only in whether the ADVECTIVE lanes are visible to
    the GF adapter.  A version that exports zeros produces an exact-0.0
    delta, which is the A/B law's own signature for "the experiment never
    ran".
    """
    from woof.core import dycore
    from woof.core.gf import GrellFreitas

    class _AdvectiveLanesWithheld:
        _WITHHELD = ("gf_rthdynten", "gf_rqvdynten")

        def __init__(self, driver):
            object.__setattr__(self, "_driver", driver)

        def __getattr__(self, name):
            if name in _AdvectiveLanesWithheld._WITHHELD:
                return None
            return getattr(object.__getattribute__(self, "_driver"), name)

    class _WithheldAdvectiveGF(GrellFreitas):
        # The withholding lives in bind_driver because _run_cumulus
        # re-binds the real driver at the top of EVERY due call; a proxy
        # installed from outside is overwritten on the first step and the
        # control arm silently becomes a copy of the treatment arm.
        def bind_driver(self, driver):
            super().bind_driver(_AdvectiveLanesWithheld(driver))

    cfg = _config()
    state_fed, driver_fed = _driver_for(cfg)
    _sheared_flow(state_fed)
    state_zero, driver_zero = _driver_for(cfg)
    _sheared_flow(state_zero)
    driver_zero.cumulus_callable = _WithheldAdvectiveGF()

    # Step 1 writes the export; step 2's cumulus call is the first that can
    # read it (the one-step lag h_diabatic has by the same construction).
    for _ in range(2):
        dycore.step(state_fed, cfg)
        dycore.step(state_zero, cfg)

    assert driver_zero.cumulus_callable._driver.gf_rthdynten is None, (
        "the control arm's adapter can see the advective lanes, so both "
        "arms ran the same numerics and the delta below is meaningless")
    assert _absmax(state_fed.rthften) > 0.0
    assert _absmax(state_zero.rthften) > 0.0, (
        "the control arm's DYCORE must still export -- only the adapter's "
        "view is withheld, or this is two different models")

    rate_fed = driver_fed.cu_rates["rthcuten"]
    rate_zero = driver_zero.cu_rates["rthcuten"]
    assert _absmax(rate_fed) > 0.0, "GF produced no heating in the fed arm"
    delta = _absmax(rate_fed - rate_zero)
    assert delta > 0.0, (
        "withholding the advective forcing lanes left GF's heating rate "
        "bit-identical: the export is not reaching the kernel, so this "
        "configuration is running the pre-export numerics")
    # RECORDED on this fixture at the tip that introduced the export, with
    # the band #237 used and for its reasons: a magnitude gate, not a
    # bitwise one, because FP32 convective closures are not bit-portable
    # across cards.  What it catches is a lane silently reverting to zeros
    # (every number collapses) and a lane arriving in the wrong units or
    # off the wrong variable (they explode).  For scale, #237's PBL-lane
    # delta on the same engine was 2.67e-06 K s-1; the advective lane
    # moves GF by three orders of magnitude more, which is what makes it
    # the lane worth closing.
    recorded_rthften_absmax = 6.72035e-02     # K s-1
    recorded_rqvften_absmax = 3.08361e-06     # kg kg-1 s-1
    recorded_rate_absmax = 3.11390e-03        # K s-1, fed arm
    recorded_delta_absmax = 3.38154e-03       # K s-1, fed minus withheld
    for measured, recorded, label in (
            (_absmax(state_fed.rthften), recorded_rthften_absmax,
             "rthften absmax"),
            (_absmax(state_fed.rqvften), recorded_rqvften_absmax,
             "rqvften absmax"),
            (_absmax(rate_fed), recorded_rate_absmax, "rthcuten absmax"),
            (delta, recorded_delta_absmax, "rthcuten delta absmax")):
        assert 0.25 * recorded <= measured <= 4.0 * recorded, (
            f"{label} = {measured:.5e}, recorded {recorded:.5e}")

    # The moisture side moves too, so this is not a heating-only artefact.
    assert _absmax(driver_fed.cu_rates["rqvcuten"]
                   - driver_zero.cu_rates["rqvcuten"]) > 0.0
