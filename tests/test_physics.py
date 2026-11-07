"""``woof/core/physics.py`` as woof's ``phys/module_physics_init.F``.

WHY THIS FILE EXISTS
--------------------
``initialize_physics`` is woof's ``phy_init``.  WRF's ``phy_init`` ends with
a call to ``mp_init`` (``phys/module_physics_init.F:1635``), and ``mp_init``'s
``CASE (THOMPSONAERO)`` arm (``:4522-4538``) calls ``thompson_init`` -- which
is where WRF installs the synthetic CCN/IN profile whenever the water- and
ice-friendly aerosol fields arrive unset
(``phys/module_mp_thompson.F:493-515`` for CCN, ``:531-551`` for IN).

For four waves woof had the fill (``woof.core.microphysics.
microphysics_init``, oracle-gated against WRF's own post-``thompson_init``
snapshot) and NO PRODUCTION CALLER.  Every mp=28 forecast therefore started at
``nwfa = nifa = 0`` and was clamped to WRF's floors at
``module_mp_thompson.F:3979-3982`` -- a maritime-clean CCN population
everywhere, for the whole run, with nothing NaN, nothing negative and no
health check tripped.  Measured over 150 steps it was worth 5.6x fewer cloud
droplets and +74.2% domain-total RAINNC
(``tests/test_mp28_forecast_smoke.py::
test_the_aerosol_profile_changes_the_forecast_measurably``).

This file gates the wiring, and it gates the three properties that make the
wiring correct rather than merely present:

1. it runs, and it installs WRF's own arithmetic (not "the field became
   nonzero");
2. it runs EXACTLY ONCE per domain -- a per-step call would overwrite an
   advected, activated and scavenged aerosol field with the synthetic profile
   every single step, which is a worse bug than the one being fixed;
3. it is RESTART-SAFE -- ``thompson_init`` reaches its fill through two
   independent ``MAXVAL`` presence tests (``:493`` and ``:531``), so a domain
   that already carries aerosol (a resumed checkpoint, or a future WIF
   ingest) is left bit-for-bit untouched.

It also gates the rest of mp=28's physics-driver admission, each against the
WRF line that puts 28 in the set rather than against mp=8's spelling.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

from woof.config import RunConfig, validate_run_config
from woof.core.physics import DECLARED_CONSTANT_GLW_WM2  # noqa: E402

#: The idealised constant downward longwave these fixtures declare.
#:
#: ``woof.core.physics.initialize_physics`` no longer defaults ``glw``
#: (300.0 through 1.8.7): a land-surface suite with no longwave scheme
#: must state where its downward longwave comes from instead of being
#: handed a plausible-looking 300 W m-2 nobody chose.  These are
#: idealised columns; the constant is the right answer for them and this
#: is where they say so.  The VALUE is 1.8.7's default, so every fixture
#: below integrates exactly the numbers it always did.
_IDEALISED_GLW = DECLARED_CONSTANT_GLW_WM2


# ---------------------------------------------------------------------------
# Host-only admission gates.
# ---------------------------------------------------------------------------

#: WRF's mp=28 surface accumulator set, exactly as
#: ``phys/module_microphysics_driver.F``'s ``CASE (THOMPSONAERO)`` arm binds
#: it into ``mp_gt_driver``: RAINNC/RAINNCV (:1085-1086), SNOWNC/SNOWNCV
#: (:1087-1088), GRAUPELNC/GRAUPELNCV (:1089-1090) and SR (:1091).  No HAILNC
#: -- that is NSSL's, and mp_gt_driver has no hail category at all.
_MP28_ACCUMULATORS = ("rainnc", "rainncv", "sr", "snownc", "snowncv",
                      "graupelnc", "graupelncv")


def test_microphysics_scratch_slots_admits_28_with_mp8s_accumulator_set():
    """``microphysics_scratch_slots(28)`` must be ``microphysics_scratch_slots(8)``.

    BEFORE THIS TEST: ``physics.py:315`` read ``if mp_physics in (6, 8, 10)``
    and mp=28 fell through to ``return ()``, so a PhysicsDriver on an mp=28
    domain allocated three private zero-filled surface arrays instead of
    aliasing the seven canonical ``mp_*`` scratch accumulators the aerosol
    adapter actually writes (``woof/core/microphysics_aerosol.py:263-269``).

    The WRF authority is the driver arm, not mp=8's spelling: ``CASE
    (THOMPSONAERO)`` at ``phys/module_microphysics_driver.F:1029`` calls
    ``mp_gt_driver`` with RAINNC, RAINNCV, SNOWNC, SNOWNCV, GRAUPELNC,
    GRAUPELNCV and SR and with no hail argument -- the identical set ``CASE
    (THOMPSON)`` binds.
    """
    from woof.core.physics import microphysics_scratch_slots

    slots28 = microphysics_scratch_slots(28)
    assert slots28, (
        "mp_physics=28 has no canonical accumulator set, so its PhysicsDriver "
        "allocates private zero arrays and the surface/LSM seam never sees "
        "the scheme's own RAINNCV")
    assert tuple(name for name, _slot in slots28) == _MP28_ACCUMULATORS
    assert slots28 == microphysics_scratch_slots(8), (
        "mp=28 must alias exactly mp=8's slots: the aerosol adapter writes "
        "mp_rainnc/mp_rainncv/mp_snownc/mp_snowncv/mp_graupelnc/"
        "mp_graupelncv/mp_sr, the same seven names")
    # ... and nothing about mp=8, mp=6, mp=10, mp=18 or mp=1 moved.
    assert microphysics_scratch_slots(1)[0] == ("rainnc", "mp_rainnc")
    assert len(microphysics_scratch_slots(1)) == 3
    assert len(microphysics_scratch_slots(18)) == 9
    assert microphysics_scratch_slots(0) == ()


def test_the_pbl_rqi_budget_admits_28():
    """YSU returns ``dqi`` for every scheme whose moist package carries QI.

    ``Registry/Registry.EM_COMMON:3036`` declares the ``thompsonaero``
    package as ``moist:qv,qc,qr,qi,qs,qg``, so ``F_QI`` is true for mp=28 and
    ``module_first_rk_step_part1.F:1112``'s ``CALL pbl_driver`` hands
    ``moist(...,P_QI), F_QI=F_QI`` (:1199) to the PBL driver exactly as it
    does for THOMPSON.

    BEFORE THIS TEST: ``physics.py:296`` read ``(6, 8, 10, 18)`` and an
    mp=28 + YSU domain silently dropped the PBL ice tendency.
    """
    from woof.core.physics import (_composed_optional_tendency_components,
                                    _pbl_optional_tendency_components)

    cfg = _mp_only_cfg(mp_physics=28, bl_pbl_physics=1, sf_sfclay_physics=1)
    assert _pbl_optional_tendency_components(cfg) == ("rqi",)
    assert "rqi" in _composed_optional_tendency_components(cfg)
    # The PBL-off and the dry cases are unchanged.
    assert _pbl_optional_tendency_components(
        _mp_only_cfg(mp_physics=28)) == ()
    assert _pbl_optional_tendency_components(
        _mp_only_cfg(mp_physics=1, bl_pbl_physics=1,
                     sf_sfclay_physics=1)) == ()


def test_the_kf_phase_contract_admits_28_as_separate_ice_snow():
    """KF's ``F_QI``/``F_QS`` feedback branch for mp=28 is THOMPSON's.

    ``module_cumulus_driver.F:1043`` passes ``F_QI=f_qi, F_QS=f_qs`` into
    ``KF_eta_CPS``; both are true for mp=28 because
    ``Registry/Registry.EM_COMMON:3036`` puts ``qi`` and ``qs`` in the
    ``thompsonaero`` moist package.  That is the SEPARATE_ICE_SNOW branch.

    BEFORE THIS TEST: ``kf.py:74`` listed ``(6, 8, 10, 18)`` and
    ``kf_phase_mode_for_microphysics(28)`` raised ``ValueError``, so an
    mp=28 + KF configuration could not be constructed at all.
    """
    from woof.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics
    from woof.core.physics import _cumulus_optional_tendency_components

    assert kf_phase_mode_for_microphysics(28) == \
        KFPhaseMode.SEPARATE_ICE_SNOW
    assert kf_phase_mode_for_microphysics(8) == KFPhaseMode.SEPARATE_ICE_SNOW
    assert kf_phase_mode_for_microphysics(1) == KFPhaseMode.WARM_RAIN
    assert kf_phase_mode_for_microphysics(0) == \
        KFPhaseMode.NO_SEPARATE_SNOW
    # mp 95 (old Eta Ferrier) is not ported; 7 joined with its port.
    with pytest.raises(ValueError):
        kf_phase_mode_for_microphysics(95)

    cfg = _mp_only_cfg(mp_physics=28, cu_physics=1)
    assert _cumulus_optional_tendency_components(cfg) == ("rqr", "rqi", "rqs")


def test_the_land_surface_seam_takes_28s_own_sr_not_a_temperature_proxy():
    """SR reaches the LSM from the scheme for every scheme that produces it.

    ``phys/module_microphysics_driver.F``'s ``CASE (THOMPSONAERO)`` arm binds
    ``SR=SR`` into ``mp_gt_driver`` (:1091) exactly as ``CASE (THOMPSON)``
    does, and ``mp_gt_driver`` fills it (``module_mp_thompson.F``'s
    per-column ``SR`` write).  WRF's Noah driver then consumes it with
    ``FRPCPN=.true.``.

    BEFORE THIS TEST: all three LSM runners in physics.py spelled the set
    ``(1, 6, 8, 10, 18)``, so an mp=28 + Noah/RUC/Noah-MP domain substituted
    ``T(k=1) <= 273.15`` for the scheme's own frozen fraction AND ran Noah
    with ``frpcpn=False`` -- a different partitioning of every precipitation
    event over land.
    """
    from woof.core.physics import microphysics_scheme_sr_available

    assert microphysics_scheme_sr_available(28)
    for mp in (1, 6, 8, 10, 18):
        assert microphysics_scheme_sr_available(mp), mp
    assert not microphysics_scheme_sr_available(0)


@requires_gpu
def test_noah_receives_28s_scheme_sr_and_wrfs_frpcpn_flag(monkeypatch):
    """The predicate above, executed through the real Noah runner.

    ``PhysicsDriver._run_noah`` is called with a minimal atmosphere and
    ``launch_noah`` is intercepted, so what is asserted is the two values
    WRF's ``module_sf_noahdrv.F`` seam actually receives: the SR array (the
    scheme's own frozen fraction, which for mp=28 is the ``mp_sr`` scratch
    slot the aerosol adapter writes) and ``FRPCPN``.

    BEFORE THIS TEST: mp=28 fell out of the tuple, so ``f["sr"]`` was
    ``T(kts) <= 273.15`` -- 1.0 in every column colder than freezing,
    regardless of what the scheme produced -- and ``frpcpn`` was False.
    """
    import cupy as cp

    from woof.core import physics as physics_mod
    from woof.core.physics import initialize_physics

    _tables_or_skip()
    cfg = validate_run_config(_mp_only_cfg(
        sf_sfclay_physics=1, sf_surface_physics=2, num_soil_layers=4))
    state = _balanced_state(cp, cfg)
    driver = initialize_physics(state, cfg, glw=_IDEALISED_GLW)

    # A scheme SR that the temperature proxy could never produce: 0.25
    # everywhere, in a column that is BELOW freezing at the surface (so the
    # proxy would say 1.0).
    driver.microphysics.sr[...] = cp.float32(0.25)
    shape = state.mup.shape
    nz = cfg.nz
    atmosphere = {
        "p_interface": cp.asarray(
            np.linspace(1.0e5, 1.0e4, nz + 1, dtype=np.float32)
        )[:, None, None] * cp.ones((nz + 1, *shape), dtype=cp.float32),
        "temperature": cp.full((nz, *shape), np.float32(263.0)),
        "qv": cp.zeros((nz, *shape), dtype=cp.float32),
        "dz": cp.full((nz, *shape), np.float32(50.0)),
    }

    seen = {}

    def fake_launch_noah(fields, params, dt, dzs, *, frpcpn, **kwargs):
        seen["frpcpn"] = frpcpn
        seen["sr"] = cp.asnumpy(fields["sr"]).copy()

    monkeypatch.setattr(physics_mod, "launch_noah", fake_launch_noah)
    driver._run_noah(atmosphere, cfg, itimestep=1)

    assert seen["frpcpn"] is True, (
        "Noah ran with FRPCPN=.false. for mp=28, so it re-derived the frozen "
        "fraction from air temperature instead of using the scheme's SR")
    assert np.allclose(seen["sr"], 0.25), (
        "Noah received the temperature proxy instead of mp=28's own SR "
        f"(got {seen['sr'].min()}..{seen['sr'].max()})")


# ---------------------------------------------------------------------------
# The device gates: the one call, once, and restart-safe.
# ---------------------------------------------------------------------------

def _mp_only_cfg(**overrides) -> RunConfig:
    """A minimal admissible config; mp-only (no radiation/surface/PBL/cu)."""
    values = dict(
        nx=8, ny=6, nz=10, dx=2000.0, dy=2000.0, ztop=10000.0,
        dt=6.0, run_seconds=30.0, moist=True, mp_physics=28,
    )
    values.update(overrides)
    return RunConfig(**values)


def _tables_or_skip():
    """Skip only when CCN_ACTIVATE.BIN is genuinely absent.

    The asset ships as of 2026-08-01 (MP28_PORT_SPEC.md blocking unknown 1,
    reversed), so this guard does not fire on a clean checkout.  It stays as
    defence for a tree missing the file or pointed elsewhere by an override,
    and names that one asset rather than swallowing every load failure.
    """
    from woof.core.thompson_aerosol_contract import (
        MissingAerosolTableAsset, resolve_aerosol_table_root,
        resolve_ccn_activation_path)
    try:
        resolve_ccn_activation_path(None, resolve_aerosol_table_root(None))
    except MissingAerosolTableAsset as exc:                # pragma: no cover
        pytest.skip(f"CCN_ACTIVATE.BIN unavailable: {exc}")


def _balanced_state(cp, cfg):
    """A balanced moist WK82 column, the state a forecast would start from."""
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.verify.cases.wk82 import wk82_sounding, wk82_theta

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, wk82_theta, 1.0e5, cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                                lambda z: wk82_sounding(z)[1], None)
    state.qv0[...] = state.qv
    return state


def _wrf_reference_profile(cp, state, cfg):
    """WRF ``thompson_init``:499-514, recomputed in float64 on the host.

    Deliberately a transcription of the Fortran rather than a call into the
    port: the point is to check the port's arithmetic, not to compare it
    with itself.  ``hgt`` is the FULL (w) level height above sea level,
    ``(phb + php)/G``, because ``dyn_em/start_em.F:873`` fills WRF's
    ``z_at_q`` argument from the Z-staggered ``ph_2 + phb``.
    """
    from woof.core import constants as c

    naCCN0, naCCN1 = 300.0e6, 50.0e6    # module_mp_thompson.F:96-97
    naIN0, naIN1 = 1.5e6, 0.5e6         # module_mp_thompson.F:94-95

    phb = cp.asnumpy(state.phb).astype(np.float64)
    php = cp.asnumpy(state.php).astype(np.float64)
    if phb.ndim == 1:
        phb = phb[:, None, None]
    z8w = (phb + php) / float(c.G)
    hgt = z8w[:cfg.nz]                            # (nz, ny, nx)

    surface = hgt[0]
    h_01 = np.where(surface <= 1000.0, 0.8,
                    np.where(surface >= 2500.0, 0.01,
                             0.8 * np.cos(surface * 0.001 - 1.0)))
    niCCN3 = -1.0 * np.log(naCCN1 / naCCN0) / h_01
    niIN3 = -1.0 * np.log(naIN1 / naIN0) / h_01

    # :508 / :546 -- level 1 uses the LEVEL-2 height difference, not zero.
    agl = hgt - surface[None, :, :]
    agl = np.concatenate([(hgt[1] - hgt[0])[None, :, :], agl[1:]], axis=0)
    nwfa = naCCN1 + naCCN0 * np.exp(-(agl / 1000.0) * niCCN3)
    nifa = naIN1 + naIN0 * np.exp(-(agl / 1000.0) * niIN3)
    # :509-510 -- the surface emission, from the FILLED lowest-level value.
    z1 = hgt[1] - hgt[0]
    nwfa2d = nwfa[0] * 0.000196 * (50.0 / z1)
    return nwfa, nifa, nwfa2d


@requires_gpu
def test_initialize_physics_installs_wrfs_synthetic_ccn_and_in_profile():
    """THE ONE LINE.  ``initialize_physics`` is woof's ``phy_init``.

    BEFORE THIS TEST: nothing anywhere in ``gpuwm/`` called
    ``microphysics_init``, so this assertion read ``nwfa == 0`` and the
    receipt did not exist.

    The comparison is against WRF's own arithmetic recomputed on the host in
    float64 (``_wrf_reference_profile``), including the two details a
    transcription gets wrong: the ``k=1`` level uses the LEVEL-2 height
    difference (:508), and ``nwfa2d`` is derived from the FILLED lowest-level
    value with ``z1 = hgt(i,2,j) - hgt(i,1,j)`` (:509-510).  ``nifa2d`` is
    NOT derived -- WRF never writes one -- and ``nc`` is never touched.
    """
    import cupy as cp

    from woof.core.physics import initialize_physics

    _tables_or_skip()
    cfg = validate_run_config(_mp_only_cfg())
    state = _balanced_state(cp, cfg)
    assert float(cp.max(state.nwfa)) == 0.0
    assert float(cp.max(state.nifa)) == 0.0

    driver = initialize_physics(state, cfg, glw=_IDEALISED_GLW)
    cp.cuda.Stream.null.synchronize()

    assert driver.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": True, "in": True}}, (
        "initialize_physics did not run thompson_init's CCN/IN fill")

    nwfa_ref, nifa_ref, nwfa2d_ref = _wrf_reference_profile(cp, state, cfg)
    nwfa = cp.asnumpy(state.nwfa).astype(np.float64)
    nifa = cp.asnumpy(state.nifa).astype(np.float64)
    nwfa2d = cp.asnumpy(state.nwfa2d).astype(np.float64)

    assert np.allclose(nwfa, nwfa_ref, rtol=2.0e-6, atol=0.0), (
        f"nwfa max rel err "
        f"{np.max(np.abs(nwfa - nwfa_ref) / nwfa_ref):.3e}")
    assert np.allclose(nifa, nifa_ref, rtol=2.0e-6, atol=0.0)
    assert np.allclose(nwfa2d, nwfa2d_ref, rtol=2.0e-6, atol=0.0)

    # WRF's own floors and the boundary-layer-following decay (:96-99).
    assert nwfa.min() >= 50.0e6 * (1.0 - 1.0e-6)
    assert nifa.min() >= 0.5e6 * (1.0 - 1.0e-6)
    assert nwfa[0].min() > nwfa[-1].max()
    # thompson_init derives no ice-nuclei surface flux and never touches nc.
    assert float(cp.max(cp.abs(state.nifa2d))) == 0.0
    assert float(cp.max(cp.abs(state.nc))) == 0.0


@requires_gpu
def test_initialize_physics_is_the_only_caller_and_calls_it_once_per_domain():
    """ONCE per domain, not once per step.

    WRF calls ``mp_init`` from ``phy_init`` (``module_physics_init.F:1635``)
    and nothing inside ``mp_gt_driver`` ever refills the profile
    (``module_mp_thompson.F:1070-1500`` contains no ``thompson_init`` call).
    A per-step call would overwrite an advected, activated and scavenged
    aerosol field with the synthetic profile on every step while leaving
    every clamp and every bound intact -- silent, plausible, and worse than
    the defect it would be replacing.

    Counted across a real multi-step integration, not asserted about the
    source: the counter is installed BEFORE ``initialize_physics`` and read
    after five full RK3 steps of ``dycore.step``.
    """
    import cupy as cp

    from woof.core import dycore, microphysics
    from woof.core.physics import initialize_physics

    _tables_or_skip()
    cfg = validate_run_config(_mp_only_cfg())
    state = _balanced_state(cp, cfg)

    calls: list[int] = []
    real_init = microphysics.microphysics_init

    def counting(target, run_cfg):
        calls.append(int(run_cfg.mp_physics))
        return real_init(target, run_cfg)

    microphysics.microphysics_init = counting
    try:
        initialize_physics(state, cfg, glw=_IDEALISED_GLW)
        assert calls == [28], (
            "initialize_physics did not call microphysics_init exactly once "
            f"(calls: {calls})")
        for _ in range(5):
            dycore.step(state, cfg)
        cp.cuda.Stream.null.synchronize()
    finally:
        microphysics.microphysics_init = real_init

    assert calls == [28], (
        "microphysics_init ran more than once per domain -- it was called "
        f"{len(calls)} times across five steps, so the synthetic profile is "
        "overwriting the transported aerosol field")
    assert float(state.elapsed_seconds) == 5.0 * cfg.dt


@requires_gpu
def test_initialize_physics_does_not_refill_a_domain_that_carries_aerosol():
    """RESTART SAFETY, and the future WIF ingest, in one assertion.

    ``thompson_init`` reaches its CCN fill only when
    ``MAXVAL(nwfa(...)) < eps`` (``module_mp_thompson.F:490``/``:493``) and its IN
    fill only when the same test passes on ``nifa`` (``:528``/``:531``) -- two
    INDEPENDENT domain-wide reductions, which is why a domain can legitimately
    receive one fill and not the other.  ``mp_init`` calls ``thompson_init``
    on restart as well as on a cold start
    (``module_physics_init.F:4525``: ``start_of_simulation .or. restart .or.
    cycling``), so in WRF the presence test is the ONLY thing standing
    between a resumed forecast and having its aerosol history erased.  woof
    reaches the same end by a different route -- see
    :func:`test_a_resumed_mp28_domain_keeps_its_checkpointed_aerosol` -- but
    this property is the one that matters the moment a WIF ingest, a nest
    initialisation or a cycled analysis supplies aerosol.

    The check is BITWISE on the raw bytes, not a tolerance.
    """
    import cupy as cp

    from woof.core.physics import initialize_physics

    _tables_or_skip()
    cfg = validate_run_config(_mp_only_cfg())

    # (a) both present -> nothing is refilled, nwfa2d is not re-derived.
    state = _balanced_state(cp, cfg)
    rng = np.random.default_rng(28)
    carried_nwfa = (rng.uniform(2.0e7, 4.0e8, state.nwfa.shape)
                    .astype(np.float32))
    carried_nifa = (rng.uniform(1.0e4, 2.0e6, state.nifa.shape)
                    .astype(np.float32))
    carried_2d = (rng.uniform(1.0e3, 5.0e3, state.nwfa2d.shape)
                  .astype(np.float32))
    state.nwfa[...] = cp.asarray(carried_nwfa)
    state.nifa[...] = cp.asarray(carried_nifa)
    state.nwfa2d[...] = cp.asarray(carried_2d)

    driver = initialize_physics(state, cfg, glw=_IDEALISED_GLW)
    cp.cuda.Stream.null.synchronize()
    assert driver.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": False, "in": False}}
    assert np.array_equal(cp.asnumpy(state.nwfa), carried_nwfa), (
        "initialize_physics overwrote an aerosol-bearing domain's nwfa; a "
        "restarted mp=28 forecast would lose its aerosol history")
    assert np.array_equal(cp.asnumpy(state.nifa), carried_nifa)
    assert np.array_equal(cp.asnumpy(state.nwfa2d), carried_2d)

    # (b) the two decisions really are independent: CCN present, IN absent.
    half = _balanced_state(cp, cfg)
    half.nwfa[...] = cp.asarray(carried_nwfa)
    half.nifa[...] = 0.0
    half_driver = initialize_physics(half, cfg, glw=_IDEALISED_GLW)
    cp.cuda.Stream.null.synchronize()
    assert half_driver.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": False, "in": True}}
    assert np.array_equal(cp.asnumpy(half.nwfa), carried_nwfa)
    assert float(cp.min(half.nifa)) >= 0.5e6 * (1.0 - 1.0e-6)


@requires_gpu
def test_a_resumed_mp28_domain_keeps_its_checkpointed_aerosol(tmp_path):
    """The SYSTEM-level restart claim, in woof's own resume order.

    WRF protects a resumed forecast's aerosol with ``thompson_init``'s
    presence test, because WRF calls ``mp_init`` on restart
    (``module_physics_init.F:4525``).  woof's order is different and must be
    checked as it actually is: ``woof/runtime.py`` runs the deterministic
    preparation (which now includes ``initialize_physics``, and therefore the
    profile fill on an all-zero cold state) and only THEN calls
    ``restore_restart``, which overwrites every serialized array in place.

    So the resumed domain integrates the CHECKPOINTED aerosol, and the
    synthetic profile installed a moment earlier is discarded.  That is the
    property a user cares about, it is not the same property as the presence
    guard, and before the init call was wired it could not be asked at all.

    Asserted BITWISE, on ``nwfa``, ``nifa`` and the derived ``nwfa2d``.
    """
    import cupy as cp

    from woof.core.physics import initialize_physics
    from woof.io import restart

    _tables_or_skip()
    cfg = validate_run_config(_mp_only_cfg())

    # (1) A run that has been going for a while: distinctive aerosol.
    source = _balanced_state(cp, cfg)
    initialize_physics(source, cfg, glw=_IDEALISED_GLW)
    rng = np.random.default_rng(2028)
    carried = {
        "nwfa": rng.uniform(2.0e7, 4.0e8, source.nwfa.shape).astype(np.float32),
        "nifa": rng.uniform(1.0e4, 2.0e6, source.nifa.shape).astype(np.float32),
        "nwfa2d": rng.uniform(1.0e3, 9.0e3,
                              source.nwfa2d.shape).astype(np.float32),
    }
    for name, value in carried.items():
        getattr(source, name)[...] = cp.asarray(value)
    path = tmp_path / "mp28-resume.npz"
    restart.write_restart(path, source, cfg)

    # (2) The resume: fresh domain -> initialize_physics (fill RUNS, because
    #     the cold state is all-zero) -> restore_restart.
    resumed = _balanced_state(cp, cfg)
    driver = initialize_physics(resumed, cfg, glw=_IDEALISED_GLW)
    cp.cuda.Stream.null.synchronize()
    assert driver.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": True, "in": True}}, (
        "the cold prepared state was not all-zero aerosol, so this test is "
        "not exercising woof's real resume order")
    filled_nwfa = cp.asnumpy(resumed.nwfa).copy()
    assert not np.array_equal(filled_nwfa, carried["nwfa"])

    restart.restore_restart(path, resumed, cfg)
    cp.cuda.Stream.null.synchronize()

    for name, value in carried.items():
        assert np.array_equal(cp.asnumpy(getattr(resumed, name)), value), (
            f"{name} did not survive the resume: the synthetic profile "
            "installed by initialize_physics is still in the state, so a "
            "resumed mp=28 forecast has lost its aerosol history")


@requires_gpu
def test_the_init_hook_is_an_unconditional_no_op_away_from_mp28():
    """The call in ``initialize_physics`` is unconditional; it must be inert.

    ``microphysics_init`` returns ``{}`` for every scheme woof shipped
    before mp=28 (``microphysics.py:771``), which is what lets the caller be
    one line with no selector test around it.  Asserted here through the
    real driver, on a real mp=8 ``DomainState``, because an empty receipt
    from a stub proves nothing about the production path.
    """
    import cupy as cp

    from woof.core.physics import initialize_physics

    cfg = validate_run_config(_mp_only_cfg(mp_physics=8))
    state = _balanced_state(cp, cfg)
    driver = initialize_physics(state, cfg, glw=_IDEALISED_GLW)
    cp.cuda.Stream.null.synchronize()
    assert driver.microphysics_init_receipt == {}
    assert getattr(state, "nwfa", None) is None, (
        "an mp=8 state grew aerosol fields")


@requires_gpu
def test_the_mp28_driver_aliases_the_schemes_own_accumulators():
    """The consequence of ``microphysics_scratch_slots(28)``, executed.

    With no slot row the driver allocated three private zero surface arrays
    and ``accept_microphysics`` COPIED the scheme's result into them every
    step.  With the row, ``driver.microphysics.rainnc`` IS
    ``state.scratch(..., "mp_rainnc")`` -- the same device buffer the aerosol
    adapter accumulates into (``microphysics_aerosol.py:263-269``) -- so the
    surface seam reads the live accumulator rather than a copy taken one
    step ago.
    """
    import cupy as cp

    from woof.core import dycore
    from woof.core.physics import initialize_physics

    _tables_or_skip()
    cfg = validate_run_config(_mp_only_cfg())
    state = _balanced_state(cp, cfg)
    driver = initialize_physics(state, cfg, glw=_IDEALISED_GLW)

    for name, slot in (("rainnc", "mp_rainnc"), ("rainncv", "mp_rainncv"),
                       ("sr", "mp_sr"), ("snownc", "mp_snownc"),
                       ("snowncv", "mp_snowncv"),
                       ("graupelnc", "mp_graupelnc"),
                       ("graupelncv", "mp_graupelncv")):
        assert getattr(driver.microphysics, name) is state.scratch(
            state.mup.shape, slot), name
    assert driver.microphysics.hailnc is None
    assert driver.microphysics.hailncv is None

    dycore.step(state, cfg)
    cp.cuda.Stream.null.synchronize()
    assert driver.microphysics_updates == 1
    # Still the same buffers after a real accept_microphysics.
    assert driver.microphysics.rainnc is state.scratch(
        state.mup.shape, "mp_rainnc")


@requires_gpu
def test_mp28_integrates_with_the_full_physics_driver_stack():
    """END TO END: mp=28 through ``dycore.step`` with a real physics stack.

    Everything this package admitted, running together on one domain --
    MM5 surface layer, Noah LSM and YSU PBL alongside aerosol-aware Thompson
    -- because each admission was decided from a different WRF line and the
    only way to find out whether they compose is to compose them:

    * the profile is installed once by ``initialize_physics`` (WRF
      ``mp_init``, ``module_physics_init.F:1635``);
    * ``pbl_tendencies.rqi`` is materialized and stays finite, because
      ``Registry/Registry.EM_COMMON:3036`` puts ``qi`` in the mp=28 moist
      package and YSU therefore returns ``dqi``;
    * the surface seam takes the SCHEME's SR, not a temperature proxy;
    * the driver's accumulators are the scheme's own scratch slots.

    BEFORE THIS TEST: this configuration ran with no aerosol profile, with
    the PBL ice tendency silently dropped, and with Noah re-deriving the
    frozen precipitation fraction from air temperature.
    """
    import cupy as cp

    from woof.core import dycore
    from woof.core.physics import initialize_physics

    _tables_or_skip()
    cfg = validate_run_config(_mp_only_cfg(
        sf_sfclay_physics=1, sf_surface_physics=2, num_soil_layers=4,
        bl_pbl_physics=1, bldt=0.0))
    state = _balanced_state(cp, cfg)
    # BOTH of Noah's carriers declared, not just GLW.  This composition
    # runs Noah with radiation off over land, which is exactly the class
    # the carrier contract (woof/core/radiation_carriers.py) refuses
    # when SWDOWN has no producer: the pre-contract behaviour was to
    # integrate the buffer's allocation zeros silently.  A declared 0.0
    # is the same number with a source -- an idealised dark sky, on the
    # same terms as _IDEALISED_GLW -- and initialize_physics does NOT
    # default-declare it, because a silent default is the defect.
    driver = initialize_physics(state, cfg, glw=_IDEALISED_GLW, swdown=0.0)
    cp.cuda.Stream.null.synchronize()

    assert driver.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": True, "in": True}}
    initial_nwfa = cp.asnumpy(state.nwfa).copy()
    assert initial_nwfa.min() >= 50.0e6 * (1.0 - 1.0e-6)

    # Seed the scheme's SR slot with a value the temperature proxy cannot
    # produce on this warm sounding (the proxy is ``T(kts) <= 273.15``, which
    # is 0.0 in every column of a WK82 profile).  The first step's surface
    # call reads the slot BEFORE microphysics rewrites it, so this is the
    # non-vacuous form of the seam assertion inside a real integration.
    driver.microphysics.sr[...] = cp.float32(0.25)
    dycore.step(state, cfg)
    cp.cuda.Stream.null.synchronize()
    assert bool(cp.all(driver.fields["sr"] == cp.float32(0.25))), (
        "Noah received a temperature proxy rather than mp=28's own SR "
        f"(min {float(cp.min(driver.fields['sr']))}, "
        f"max {float(cp.max(driver.fields['sr']))})")

    for _ in range(2):
        dycore.step(state, cfg)
    cp.cuda.Stream.null.synchronize()

    # The PBL ice tendency exists and is finite -- the budget preflight
    # prices and the composition physics.py performs.
    assert driver.pbl_tendencies.rqi is not None, (
        "mp=28 + YSU composed no rqi stack, so the PBL ice tendency is "
        "being dropped")
    assert bool(cp.all(cp.isfinite(driver.pbl_tendencies.rqi)))

    # Every prognostic and every radiation-facing diagnostic survived.
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni",
                 "nwfa", "nifa", "effc", "effi", "effs", "thp", "u", "v",
                 "w"):
        value = getattr(state, name)
        assert bool(cp.all(cp.isfinite(value))), name
    # WRF's terminal aerosol floors hold on the whole (periodic) domain.
    assert float(cp.min(state.nwfa)) >= 11.1e6 * (1.0 - 1.0e-6)
    assert float(cp.min(state.nifa)) >= 5.0e3 * (1.0 - 1.0e-6)

    # The surface seam saw the scheme's SR, and the accumulators are the
    # scheme's own slots.
    assert driver.microphysics.sr is state.scratch(state.mup.shape, "mp_sr")
    assert bool(cp.all(driver.fields["sr"] == driver.microphysics.sr)), (
        "after three steps the surface seam and the scheme's SR slot "
        "disagree")
    assert driver.microphysics_updates == 3

    from woof.core import health
    health.StateHealthValidator(state).require_healthy(
        phase="mp28-full-physics-stack")


@requires_gpu
def test_the_mp28_health_descriptor_census_is_re_derived_not_assumed():
    """What admitting mp=28's accumulator set costs the descriptor ceiling.

    ``health.collect_state_fields`` auto-walks the driver, and the count is
    capped INSIDE it by ``MAX_HEALTH_FIELDS`` -- a cap a forecast hits
    mid-run, not at admission (``tools/health_field_census.py``).  Giving
    mp=28 a ``microphysics_scratch_slots`` row changes that count, so the
    count is MEASURED here rather than assumed unchanged.

    Measured on one identical small domain (mp + MM5 + Noah + YSU),
    at driver construction:

      ==========  ===========  ==================
      mp_physics  descriptors  microphysics rows
      ==========  ===========  ==================
      8           142          14
      10          146          14
      18          154          18
      28 (before) 134           3
      28 (after)  145          14
      ==========  ===========  ==================

    The +11 is the seven canonical accumulators plus their seven scratch
    aliases replacing three private arrays.  What matters for the ceiling is
    that mp=28 remains BELOW mp=18, which is the row that holds the recorded
    four-domain peak -- so admitting mp=28 does not move the peak, and the
    published ceiling measurement stands.

    The absolute numbers are not pinned (a sibling scheme adding a field
    would move them all); the relations that decide the ceiling are.
    """
    import cupy as cp

    from woof.core import health
    from woof.core.physics import initialize_physics

    _tables_or_skip()
    counts = {}
    rows = {}
    for mp in (8, 18, 28):
        cfg = validate_run_config(_mp_only_cfg(
            mp_physics=mp, sf_sfclay_physics=1, sf_surface_physics=2,
            num_soil_layers=4, bl_pbl_physics=1))
        state = _balanced_state(cp, cfg)
        initialize_physics(state, cfg, glw=_IDEALISED_GLW)
        cp.cuda.Stream.null.synchronize()
        names = [f.name for f in health.collect_state_fields(state)]
        counts[mp] = len(names)
        rows[mp] = sum(1 for name in names if "microphysics" in name)

    print(f"\nmp=28 health descriptor census: {counts} "
          f"(microphysics rows {rows})")
    assert rows[28] == rows[8] == 14, (
        "mp=28 no longer publishes mp=8's accumulator set to the health "
        f"gate: {rows}")
    assert counts[28] == counts[8] + 3, (
        "mp=28's inventory should be mp=8's plus exactly nc, nwfa and nifa; "
        f"measured {counts}")
    assert counts[28] < counts[18], (
        "mp=28 has become the widest health inventory, so the recorded "
        f"four-domain descriptor peak must be re-measured: {counts}")
    assert counts[28] < health.MAX_HEALTH_FIELDS
