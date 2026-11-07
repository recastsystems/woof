"""P3 (mp_physics=50) is a configuration a user can name, and it radiates.

THE BREAKAGE THESE GATES PREVENT
--------------------------------
P3 merged correct and fast -- a CUDA port measured against WRF's unmodified
``phys/module_mp_p3.F`` -- and then sat unreachable, in two separate ways
that no existing gate could see:

1. NO PROFILE SELECTED IT.  ``SINGLE_DOMAIN_PHYSICS_PROFILES`` carried
   sixteen suites and not one chose mp=50, so there was no configuration a
   menu, a ``--physics-profile`` flag or a route declaration could name.
   That half is CLOSED on this line: ``P3_LEGACY_RRTMG_PROFILE_ID`` ships,
   is on the wizard's choice list, and is the profile every gate in
   section 3 below binds.  The gates stay because the property they pin --
   a scheme reachable from the doors a picker actually asks -- is what
   regressed, and a suite can fall back out of any of those tables.

2. NEITHER RADIATION ARM COULD RUN IT.  ``cloud_optics_scheme(50)`` raised
   (no row in ``_MP_CLOUD_OPTICS_SCHEME``), and the config-time refusal
   that hid this named ``ra_rrtmg_variant='rrtmg_legacy'`` as the way
   through -- an arm that ALSO raised, from
   ``legacy_scheme_declares_radii(50, 1)``, at the first radiation call.
   Both shipped P3 configs named that dead remedy.  ``woof check`` returns
   0 without ever calling radiation, so the green check on those configs
   was not evidence of anything.

Every gate below fails on the tree that had those defects.

THE WRF AUTHORITY, IN ONE PLACE
-------------------------------
Read on a stock WRF v4.7.1 tree whose ``phys/module_mp_p3.F`` hashes to the
same ``716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6``
the Fortran oracle pins; v4.6.1 line numbers second where the text moved.

* ``Registry/Registry.EM_COMMON:3043`` (v4.6.1 :3038) --
  ``package p3_1category mp_physics==50 -
  moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib;
  state:re_cloud,re_ice,vmi3d,rhopo3d,di3d,refl_10cm,th_old,qv_old``.
  No ``qs`` in ``moist`` and no ``re_snow`` in ``state``: one ice category
  spans the snow-to-graupel continuum through rime mass and rime volume.
* ``phys/module_physics_init.F:1018`` (:1017) puts P3_1CATEGORY in the
  ``use_mp_re`` disjunction that sets ``has_reqc = has_reqi = has_reqs = 1``
  at :1022-1024 (:1021-1023); :1027-1034 (:1026-1033), commented "for P3,
  to ensure correct coupling with predicted effective radii", then sets
  ``has_reqs = 0``.  The row is 1, 1, 0.
* ``phys/module_radiation_driver.F:3879-3887`` (same lines in both) is
  cal_cldfra1's P3 arm: ``IF (F_QI .and. F_QC .and. .not. F_QS)`` gives
  QCLD = QI + QC and weight = QI/QCLD.
* ``phys/module_ra_rrtmg_lw.F:12250-12261`` and
  ``phys/module_ra_rrtmg_sw.F:10851-10863`` (same lines in both) remap the
  species under ``has_reqs == 0 .and. has_reqi /= 0 .and. has_reqc /= 0``:
  ``inflg = iceflg = 5``, ``resnow1D = MAX(10., re_ice*1.E6)``,
  ``QS1D = QI3D``, ``QI1D = 0.``, ``reice1D = 10.``
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

# Imported inside each test, not at module scope.  A module-scope import of
# the profile id turns "the profile does not exist" into a COLLECTION error
# that takes the whole file down at once; per-test imports let each gate
# fail on its own claim, which is what makes the red side readable.


# ---------------------------------------------------------------------------
# 1.  The cloud-optics row, and that something reads it.
# ---------------------------------------------------------------------------

def test_p3_has_a_cloud_optics_row_and_it_is_wrfs_p3_coupling():
    """mp=50 resolves, and to its OWN coupling rather than a borrowed one.

    Borrowing Thompson's or Morrison's row would hand RRTMGP a snow
    effective radius P3 never computed, which is inventing physics; falling
    through to Kessler's is how mp=28 spent four waves radiating overcast
    ice as clear sky.  Raising, which is what this did before, kills the
    run at the first radiation call.
    """
    from woof.core.rrtmgp import (
        _MP_CLOUD_OPTICS_SCHEME, cloud_optics_scheme,
        scheme_has_snow_species, scheme_is_ice_active)

    assert _MP_CLOUD_OPTICS_SCHEME[50] == "p3"
    assert cloud_optics_scheme(50) == "p3"
    # Not borrowed from any six-species scheme, and not Kessler.
    assert cloud_optics_scheme(50) not in (
        cloud_optics_scheme(8), cloud_optics_scheme(10),
        cloud_optics_scheme(18), cloud_optics_scheme(1))

    # F_QI and F_QS separate here and nowhere else, which is exactly what
    # Registry.EM_COMMON:3043's moist:qv,qc,qr,qi says.
    assert scheme_is_ice_active("p3") is True
    assert scheme_has_snow_species("p3") is False
    for other in ("wsm6", "thompson", "morrison", "nssl"):
        assert scheme_is_ice_active(other) is True
        assert scheme_has_snow_species(other) is True
    assert scheme_is_ice_active("kessler") is False
    assert scheme_has_snow_species("kessler") is False


def test_the_p3_selector_is_no_longer_refused_against_rte_rrtmgp():
    """The retired guard is retired at BOTH surfaces that carried it.

    ``woof.config`` refused mp=50 with the 4/4 pair, and the registry
    option mirrored that refusal in ``constraints.refused_when``.  A guard
    that outlives its defect refuses working configurations -- the 7,500 m
    mesh-floor failure verbatim -- so neither may still fire.
    """
    import json

    from woof.config import RunConfig, validate_run_config
    from woof.physics_registry import physics_registry

    cfg = RunConfig(
        nx=4, ny=3, nz=12, dx=2000.0, dy=2000.0, ztop=8000.0, dt=10.0,
        run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=50, ra_lw_physics=4, ra_sw_physics=4)
    validate_run_config(cfg)          # raised NotImplementedError before

    option = physics_registry()["components"]["microphysics"]["options"][
        "p3-mp50"]
    refusals = json.dumps(option["constraints"].get("refused_when", []))
    assert "rte-rrtmgp" not in refusals, refusals
    # ...and the option is reachable by a NAMED template now, not only as
    # a component override.
    assert option["reachability"]["state"] == "template"


def test_both_radiation_arms_carry_p3s_coupling():
    """The legacy arm was the refusal's advised remedy and it also raised.

    ``legacy_scheme_declares_radii(50, 1)`` raised NotImplementedError --
    no row in ``_MP_DECLARES_RADII`` -- so a user who followed the refusal's
    own advice died at the first radiation call instead of at the door.
    The fix is not a blanket True: has_reqs is asked separately, because
    the wrapper's P3 branch needs has_reqc=1, has_reqi=1, has_reqs=0 to
    fire at all (module_ra_rrtmg_lw.F:12252).
    """
    from woof.core.rrtmg_legacy import (
        legacy_cloud_fraction_flags, legacy_scheme_declares_radii,
        legacy_scheme_has_req)

    assert legacy_scheme_declares_radii(50, 1) is True
    # The triple, not one boolean: has_reqc and has_reqi are 1 and has_reqs
    # is the 0 that makes the wrapper take its P3 branch at all.
    assert legacy_scheme_has_req(50, 1) == (1, 1, 0)
    # F_QI without F_QS, which is cal_cldfra1's own P3 arm.
    assert legacy_cloud_fraction_flags(50) == (True, False)

    # use_mp_re=0 still switches every flag off, for P3 as for anything.
    assert legacy_scheme_declares_radii(50, 0) is False
    assert legacy_scheme_has_req(50, 0) == (0, 0, 0)

    # And nothing else moved: every six-species scheme still answers the
    # same to all three flags, which is what it did when there was one.
    for mp_physics in (6, 8, 16, 18, 28):
        has_reqc, has_reqi, has_reqs = legacy_scheme_has_req(mp_physics, 1)
        assert has_reqc == has_reqi == has_reqs
        assert legacy_cloud_fraction_flags(mp_physics) == (True, True)


def test_the_two_radiation_engines_agree_that_p3_has_no_snow_radius():
    """One WRF fact, two adapters, no room to disagree.

    The pairing that used to make ice clouds appear or vanish with the
    radiation selector is exactly this kind of split.
    """
    from woof.core.rrtmg_legacy import (
        legacy_cloud_fraction_flags, legacy_scheme_has_req)
    from woof.core.rrtmgp import (
        cloud_optics_scheme, scheme_has_snow_species, scheme_is_ice_active)

    scheme = cloud_optics_scheme(50)
    f_qi, f_qs = legacy_cloud_fraction_flags(50)
    assert scheme_is_ice_active(scheme) == f_qi
    assert scheme_has_snow_species(scheme) == f_qs
    assert scheme_has_snow_species(scheme) is False
    assert legacy_scheme_has_req(50, 1)[2] == 0


# ---------------------------------------------------------------------------
# 2.  The row is READ: the coupling, end to end, on a card.
# ---------------------------------------------------------------------------

@requires_gpu
def test_p3_cloud_fraction_takes_wrfs_own_p3_arm():
    """cal_cldfra1 must accept F_QI without F_QS, and answer WRF's way.

    Two failures live here.  The port used to REFUSE the moisture set
    outright (``f_qi != f_qs`` -> NotImplementedError), which is a P3 run
    dying at its first radiation call.  Answering it with the
    :3870-3877 arm instead would read a QS array mp=50 never allocates.

    The arms are not interchangeable on P3's own data: with the frozen mass
    that a six-species scheme would split between qi and qs, the P3 arm and
    the Morrison arm agree; with only the qi part they do not, and the
    difference decides whether the column is cloudy at all.
    """
    import cupy as cp

    from woof.core.rrtmgp import cal_cldfra1

    # float32 throughout: the two-arm equivalence below is an arithmetic
    # claim about the port, and a float64 qi+qs cast down afterwards would
    # differ from the kernel's own float32 add by an ulp for reasons that
    # have nothing to do with the arms.
    nz = 20
    f32 = np.float32
    plev = np.geomspace(100000.0, 1.1, nz + 1).astype(f32)
    play = np.sqrt(plev[:-1] * plev[1:]).astype(f32)
    tlay = np.linspace(290.0, 215.0, nz).astype(f32)
    qv = np.geomspace(8.0e-3, 1.0e-6, nz).astype(f32)
    qc = np.zeros(nz, f32)
    qc[4:8] = 4.0e-4
    qi = np.zeros(nz, f32)
    qi[10:14] = 3.0e-4
    qs = np.zeros(nz, f32)
    qs[10:14] = 5.0e-4

    def dev(x):
        return cp.asarray(np.asarray(x)[None, :], dtype=cp.float32)

    def fraction(q_ice, q_snow, *, f_qs):
        return cp.asnumpy(cal_cldfra1(
            dev(qv), dev(qc), dev(q_ice), dev(q_snow), dev(tlay), dev(play),
            f_qc=True, f_qi=True, f_qs=f_qs)).ravel()

    # WRF's P3 arm, transcribed here from module_radiation_driver.F rather
    # than from the port: QCLD = QI + QC, weight = QI/QCLD, and the rest of
    # cal_cldfra1 unchanged -- so the check is against the Fortran, not
    # against the code under test.
    p3 = fraction(qi, np.zeros(nz), f_qs=False)
    qcld = qi + qc
    weight = np.where(qcld < 1.0e-12, 0.0, qi / np.maximum(qcld, 1.0e-12))
    tlay = tlay.astype(np.float64)
    play = play.astype(np.float64)
    qv = qv.astype(np.float64)
    tc = tlay - 273.15
    esw = 1000.0 * 0.61078 * np.exp(17.2693882 * tc / (tlay - 35.86))
    esi = 1000.0 * 0.61078 * np.exp(21.8745584 * tc / (tlay - 7.66))
    ep2 = 287.0 / 461.6
    qvs = (1.0 - weight) * (ep2 * esw / (play - esw)) \
        + weight * (ep2 * esi / (play - esi))
    rhum = qv / qvs
    arg = np.maximum(-6.9, -100.0 * qcld / np.maximum(
        1.0e-10, qvs - qv) ** 0.49)
    expected = np.power(np.maximum(1.0e-10, rhum), 0.25) * (1.0 - np.exp(arg))
    expected = np.where(expected < 0.01, 0.0, expected)
    expected = np.where(qcld < 1.0e-12, 0.0,
                        np.where(rhum >= 1.0, 1.0, expected))
    assert np.allclose(p3, expected, rtol=2.0e-6, atol=1.0e-7)

    # Non-vacuous in both directions.  P3's one category carrying the whole
    # frozen mass reproduces the six-species answer...
    six_species = fraction(qi, qs, f_qs=True)
    assert np.allclose(fraction(qi + qs, np.zeros(nz, f32), f_qs=False),
                       six_species, rtol=1.0e-6, atol=1.0e-9)
    # ...and dropping the snow part changes it, which is the arm being read
    # rather than the flags being ignored: those frozen layers fall under
    # the 0.01 truncation and the column radiates clear.
    frozen = slice(10, 14)
    assert float(six_species[frozen].max()) > 0.01
    assert float(p3[frozen].max()) == 0.0
    assert float(p3.max()) > 0.0

    # The Ferrier moisture set (f_qs without f_qi) is ported since the Eta
    # Ferrier mp95 recovery (74abec86b8) and reads WRF's carried F_ICE_PHY;
    # without that carrier it refuses by name rather than guess the phase.
    with pytest.raises(ValueError, match="requires carried F_ICE_PHY"):
        cal_cldfra1(dev(qv), dev(qc), dev(qi), dev(qs), dev(tlay), dev(play),
                    f_qc=True, f_qi=False, f_qs=True)


@requires_gpu
def test_p3_radiation_paths_transcribe_wrfs_ice_into_snow_remap():
    """The row must be READ, not merely present.

    WRF does not radiate P3's ice through the ice parameterisation: under
    has_reqs=0 with has_reqc/has_reqi set it moves the mass to the SNOW
    species at P3's own ice radius and empties the ice path
    (module_ra_rrtmg_lw.F:12250-12261, module_ra_rrtmg_sw.F:10851-10863),
    then applies the iceflg=5 discount.  A branch that merged qi at
    ``reice1D = 10.`` instead would radiate every P3 ice cloud at a fixed
    10 um.
    """
    import cupy as cp

    from woof.core.rrtmgp import hydrometeor_paths

    nz = 6
    plev_col = np.geomspace(1.0e5, 1.0e4, nz + 1)
    plev = cp.asarray(plev_col[None, :], dtype=cp.float32)
    qc = np.zeros((1, nz), np.float32)
    qc[0, 1] = 5.0e-4
    qi = np.zeros((1, nz), np.float32)
    qi[0, 2:4] = 3.0e-4
    cldfra = np.full((1, nz), 0.5, np.float32)

    def paths(effi_um, effc_um=8.0):
        return hydrometeor_paths(
            plev, cp.asarray(qc), qr=None, qi=cp.asarray(qi), qs=None,
            microphysics="p3", cldfra=cp.asarray(cldfra),
            effc=cp.full((1, nz), np.float32(effc_um), cp.float32),
            effi=cp.full((1, nz), np.float32(effi_um), cp.float32),
            validate=True)

    mass_path = np.abs(np.diff(plev_col)) * np.float32(1000.0 / 9.80665)

    for effi_um, expected_re_s in ((4.0, 10.0),   # MAX(10., re_ice)
                                   (40.0, 40.0),
                                   (400.0, 130.0)):  # capped at 130
        got = paths(effi_um)
        re_s = max(10.0, effi_um)
        factor = min(0.99, (130.0 / re_s) ** 2) if re_s > 130.0 else 0.99
        ciwp = cp.asnumpy(got.ciwp).ravel()
        dgice = cp.asnumpy(got.dgice).ravel()
        # QS1D = QI3D, QI1D = 0: the frozen path is the discounted P3 ice
        # mass and nothing else, in-cloud through max(0.01, cldfra).
        want_ciwp = qi.ravel() * factor * mass_path / 0.5
        assert np.allclose(ciwp, want_ciwp, rtol=1.0e-5), effi_um
        # resnow1D drives the merged single-species DIAMETER wherever there
        # is frozen mass; reice1D = 10. rides an empty path and never
        # reaches it.
        frozen = qi.ravel() > 0.0
        assert np.allclose(dgice[frozen],
                           np.clip(2.0 * expected_re_s, 10.0, 180.0),
                           rtol=1.0e-5), effi_um
        # Liquid is P3's own droplet radius where there is cloud water.
        reliq = cp.asnumpy(got.reliq).ravel()
        assert np.allclose(reliq[qc.ravel() > 0.0], 8.0, rtol=1.0e-5)

    # A snow radius is refused rather than consumed: P3 never computes one.
    with pytest.raises(ValueError, match="no snow effective radius"):
        hydrometeor_paths(
            plev, cp.asarray(qc), qr=None, qi=cp.asarray(qi), qs=None,
            microphysics="p3", effc=cp.full((1, nz), 8.0, cp.float32),
            effi=cp.full((1, nz), 25.0, cp.float32),
            effs=cp.full((1, nz), 90.0, cp.float32))
    # And a snow MASS is refused: mp=50 allocates no qs at all.
    with pytest.raises(ValueError, match="nonzero snow mixing ratio"):
        hydrometeor_paths(
            plev, cp.asarray(qc), qr=None, qi=cp.asarray(qi),
            qs=cp.full((1, nz), 1.0e-4, cp.float32),
            microphysics="p3", effc=cp.full((1, nz), 8.0, cp.float32),
            effi=cp.full((1, nz), 25.0, cp.float32), validate=True)


@requires_gpu
def test_a_p3_radiation_call_consumes_the_scheme_radii_and_asks_for_no_snow():
    """The coupling, end to end, through the shipped adapter.

    The state is P3's real one: no ``qs``, no ``effs``, ``effc``/``effi``
    only, which is what ``woof/core/state.py`` allocates for mp=50 and
    what ``woof/core/p3.py`` writes from WRF's diag_effc_3d/diag_effi_3d.
    An adapter that demanded a snow radius would raise on this state --
    which is what the legacy arm did -- and one that ignored the supplied
    radii would return the same fluxes for both.
    """
    from datetime import datetime
    from types import SimpleNamespace

    import cupy as cp

    from woof.core.rrtmgp import RRTMGPRadiation

    nz, ny, nx = 20, 1, 1
    plev_col = np.geomspace(100000.0, 1.1, nz + 1)
    play_col = np.sqrt(plev_col[:-1] * plev_col[1:])
    t_col = np.linspace(290.0, 215.0, nz)
    exner_col = (play_col / 100000.0) ** (287.0 / 1004.0)
    shape = (nz, ny, nx)

    def expand(x):
        return cp.asarray(
            np.broadcast_to(x[:, None, None], shape).copy(), dtype=cp.float32)

    qc = cp.zeros(shape, cp.float32)
    qi = cp.zeros(shape, cp.float32)
    qc[4:8] = 4.0e-4
    qi[10:14] = 8.0e-4
    atmosphere = {
        "pressure": expand(play_col),
        "p_interface": cp.asarray(np.broadcast_to(
            plev_col[:, None, None], (nz + 1, ny, nx)).copy(),
            dtype=cp.float32),
        "temperature": expand(t_col),
        "theta": expand(t_col / exner_col),
        "exner": expand(exner_col),
        "qv": expand(np.geomspace(8.0e-3, 1.0e-6, nz)),
        "qc": qc,
        "qi": qi,
    }
    fields = {
        "tsk": cp.full((ny, nx), 288.0, cp.float32),
        "albedo": cp.full((ny, nx), 0.18, cp.float32),
        "emiss": cp.full((ny, nx), 0.96, cp.float32),
        "xland": cp.ones((ny, nx), cp.float32),
    }

    def call(effc_um, effi_um):
        state = SimpleNamespace(
            elapsed_seconds=0.0, qc=qc, qi=qi,
            qr=cp.zeros(shape, cp.float32),
            effc=cp.full(shape, np.float32(effc_um), cp.float32),
            effi=cp.full(shape, np.float32(effi_um), cp.float32),
            physics=SimpleNamespace(microphysics_updates=1))
        assert not hasattr(state, "qs") and not hasattr(state, "effs")
        radiation = RRTMGPRadiation(
            datetime(1974, 4, 3, 18), cp.asarray([[40.0]]),
            cp.asarray([[-100.0]]), trace_gas_overrides={"co2": 330.0e-6})
        result = radiation(
            atmosphere=atmosphere, fields=fields, state=state,
            cfg=SimpleNamespace(mp_physics=50, dt=60.0, radt=12.0,
                                radt_minutes=12.0))
        return np.concatenate([
            np.ravel(cp.asnumpy(getattr(result, name)))
            for name in ("rthratenlw", "rthratensw", "swdown", "glw")])

    base = call(8.0, 25.0)
    assert np.all(np.isfinite(base))
    # The ICE radius moves the answer: it is the snow radius of the remap.
    assert not np.array_equal(base, call(8.0, 60.0))
    # The CLOUD radius moves the answer: has_reqc = 1.
    assert not np.array_equal(base, call(16.0, 25.0))


# ---------------------------------------------------------------------------
# 3.  The front door: a profile, a config, a choice list.
# ---------------------------------------------------------------------------

def test_a_shipped_profile_selects_p3():
    """Without this row, mp=50 is engine-proven and unshipped.

    Sixteen profiles shipped and none chose P3, so no menu entry, no
    ``--physics-profile`` value and no route declaration could reach it.
    """
    from woof.physics_compat import (P3_LEGACY_RRTMG_PROFILE_ID,
                                      SINGLE_DOMAIN_PHYSICS_PROFILES,
                                      single_domain_runtime_switches)
    assert P3_LEGACY_RRTMG_PROFILE_ID in SINGLE_DOMAIN_PHYSICS_PROFILES
    assert single_domain_runtime_switches(
        P3_LEGACY_RRTMG_PROFILE_ID)["mp_physics"] == 50
    selecting = [
        profile for profile in SINGLE_DOMAIN_PHYSICS_PROFILES
        if int(single_domain_runtime_switches(profile)["mp_physics"]) == 50]
    assert selecting == [P3_LEGACY_RRTMG_PROFILE_ID]


def test_the_p3_profile_moves_exactly_one_switch_off_its_sibling():
    """A suite must isolate the change it is for.

    The shipped row is the Thompson legacy-RRTMG suite with mp_physics
    8 -> 50 and nothing else, so a paired run measures the microphysics
    rather than a composition.  A second value drifting in here would
    silently make every P3-versus-Thompson comparison uninterpretable,
    and the comparison is the reason the suite was composed this way.
    """
    from woof.physics_compat import (P3_LEGACY_RRTMG_PROFILE_ID,
                                      THOMPSON_LEGACY_RRTMG_PROFILE_ID,
                                      single_domain_runtime_switches)
    p3 = single_domain_runtime_switches(P3_LEGACY_RRTMG_PROFILE_ID)
    twin = single_domain_runtime_switches(THOMPSON_LEGACY_RRTMG_PROFILE_ID)
    assert set(p3) == set(twin)
    differing = {name for name in p3 if p3[name] != twin[name]}
    assert differing == {"mp_physics"}, differing


def test_no_existing_profile_changed_its_microphysics():
    """P3 is SELECTABLE, not the default.  Nothing else may have moved.

    Adding a suite must not renumber a shipped one: every existing profile
    keeps the scheme it had, and the sixteen that shipped before are all
    still present.
    """
    from woof.physics_compat import (P3_LEGACY_RRTMG_PROFILE_ID,
                                      SINGLE_DOMAIN_PHYSICS_PROFILES,
                                      single_domain_runtime_switches)
    before = {
        "wsm6-ysu-mm5-noah-no-radiation-v1": 6,
        "kessler-mp1-ysu-mm5-noah-dudhia-v1": 1,
        "thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1": 8,
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1": 8,
        "thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1": 8,
        "thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1": 8,
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1": 10,
        "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-wrf-comparison-candidate-v1": 18,
        "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-wrf-comparison-candidate-v1": 18,
        "wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1": 6,
        "wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1": 6,
        "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1": 6,
        "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1": 6,
        "wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1": 6,
        "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1": 6,
        "wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1": 6,
        "wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1": 6,
    }
    for profile, mp_physics in before.items():
        assert profile in SINGLE_DOMAIN_PHYSICS_PROFILES, profile
        assert single_domain_runtime_switches(
            profile)["mp_physics"] == mp_physics, profile
    # The menu gained the six COMPOSITION suites with audit R-067 -- each
    # an implemented option's first named front door -- so the equality
    # names them rather than refusing the addition.  Naming them is the
    # point: a seventh row appearing without a line here still fails.
    from woof.physics_compat import COMPOSITION_SUITE_PROFILE_IDS
    from woof.physics_registry import template_ids_with_components

    # And the menu gained one more row when it stopped being one route's
    # declared list and became every fixed-template route's: the WSM6 + KF
    # template on the aggregate RTE+RRTMGP option, which the prepared
    # single-domain route has always declared and whose runtime product
    # that route used to hand-type in its own module.  Named here through
    # its composition rather than its id, because its id carries the
    # forcing source it was registered on.
    aggregate_kf = template_ids_with_components(
        microphysics="wsm6-mp6", cumulus="kain-fritsch",
        radiation="rte-rrtmgp")
    assert len(aggregate_kf) == 1
    assert single_domain_runtime_switches(aggregate_kf[0])["mp_physics"] == 6

    # And the two Thompson members of the MYNN + RUC pair, each its WSM6
    # row with the microphysics moved: new rows, so they are named here
    # rather than folded into ``before``, and they move nothing in it.
    from woof.physics_compat import (
        THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
        THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)
    thompson_mynn_ruc = {THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
                         THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID}
    for profile in thompson_mynn_ruc:
        assert single_domain_runtime_switches(profile)["mp_physics"] == 8

    # And the three siblings of that Thompson MYNN + RUC row the 2.8.6
    # lanes registered and the fixed-template routes now declare
    # (woof.physics_profile_siblings, 31436e8cb): the monthly-albedo
    # legacy-RRTMG template (322308291), its alb_sol 1 twin (67d633a15)
    # and the GSD MYNN 4.1 Thompson-aerosol suite (4193eb0da).  New rows,
    # each with its own microphysics, named here so a further row still
    # fails.
    thompson_mynn_ruc_siblings = {
        "thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1": 8,
        "thompson-mp8-mynn-mynn-ruc-monthly-solar-rrtmg-legacy-v1": 8,
        "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1": 28,
    }
    for profile, mp_physics in thompson_mynn_ruc_siblings.items():
        assert single_domain_runtime_switches(
            profile)["mp_physics"] == mp_physics, profile
    assert not set(thompson_mynn_ruc_siblings) & set(before)

    assert (set(SINGLE_DOMAIN_PHYSICS_PROFILES)
            == set(before) | {P3_LEGACY_RRTMG_PROFILE_ID}
            | set(COMPOSITION_SUITE_PROFILE_IDS) | set(aggregate_kf)
            | thompson_mynn_ruc | set(thompson_mynn_ruc_siblings)
            | {"lin-mp2-ysu-mm5-noah-rrtmg-legacy-implemented-unverified-v1",
                   "morrison-mp10-ysu-mm5-slab-kf-rte-rrtmgp-v1"})
    # ...and none of those six moved an EXISTING profile's microphysics,
    # which is what this test is named for: each is its own row.
    assert not set(COMPOSITION_SUITE_PROFILE_IDS) & set(before)


def test_the_p3_profile_switches_are_admissible_to_the_shipped_validator():
    """The suite is checked by the engine's own door, not by inspection.

    A profile that cannot be validated is a menu entry that fails after the
    user picks it.
    """
    from woof.physics_compat import (P3_LEGACY_RRTMG_PROFILE_ID,
                                      single_domain_runtime_switches)
    from woof.config import RunConfig, validate_run_config

    cfg = RunConfig(
        nx=41, ny=41, nz=40, dx=3000.0, dy=3000.0, ztop=20000.0, dt=15.0,
        run_seconds=600.0, time_step_sound=4,
        **single_domain_runtime_switches(P3_LEGACY_RRTMG_PROFILE_ID))
    validate_run_config(cfg)
    assert cfg.mp_physics == 50
    # And BOTH arms are wired now, so neither the profile's own legacy
    # engine nor the RTE+RRTMGP variant is a remedy that dies at the first
    # radiation call.  The profile keeps the legacy composition it was
    # issued under; the assertion is that the other arm is no longer a
    # trap for anyone who selects it by config.
    from woof.core.rrtmg_legacy import legacy_scheme_has_req
    from woof.core.rrtmgp import cloud_optics_scheme
    assert cloud_optics_scheme(cfg.mp_physics) == "p3"
    assert legacy_scheme_has_req(cfg.mp_physics, 1) == (1, 1, 0)


def test_the_p3_profile_resolves_through_the_registry_front_door():
    """Selection has to survive the plan validator, not just the table.

    ``validate_physics_plan`` is what every runner calls; a template the
    tuple names and the registry cannot resolve is not reachable.
    """
    from woof.physics_compat import (P3_LEGACY_RRTMG_PROFILE_ID,
                                      single_domain_runtime_switches)
    from woof.physics_registry import (
        physics_registry, registry_sha256, validate_physics_plan)

    registry = physics_registry()
    assert P3_LEGACY_RRTMG_PROFILE_ID in registry["templates"]
    assert registry["templates"][P3_LEGACY_RRTMG_PROFILE_ID]["components"][
        "microphysics"] == "p3-mp50"

    # The (route, source) pair is DERIVED, not typed.  The shipped P3 suite
    # is registered on a restricted set of routes rather than on every one
    # -- it carries the Kessler-rule registration its own physics_compat
    # docstring describes -- so naming a route here would pin THAT choice
    # instead of the property this gate is for, and would go red the next
    # time the registration widens.  What must stay true is that at least
    # one route/source pair offers the template and that the plan validator
    # then calls it launchable; a template no route offers is exactly the
    # unreachable state this file exists to prevent.
    offers = [
        (runner_id, source)
        for runner_id, route in sorted(registry["runner_routes"].items())
        for source, ids in sorted(route.get("source_template_ids", {}).items())
        if P3_LEGACY_RRTMG_PROFILE_ID in ids]
    assert offers, "no route/source pair offers the P3 template"
    runner_id, source_id = offers[0]

    plan = {
        "schema": registry["plan_schema"],
        "plan_id": "p3-front-door-proof-v1",
        "registry_sha256": registry_sha256(),
        "context": {
            "source_id": source_id,
            "runner_id": runner_id,
            "topology_id": registry["runner_routes"][runner_id][
                "topology_ids"][0],
        },
        "domains": [{"domain_id": "d01",
                     "template_id": P3_LEGACY_RRTMG_PROFILE_ID}],
        "edges": [],
    }
    report = validate_physics_plan(plan)
    assert report["launchable"] is True, report["errors"]
    resolved = report["resolved_domains"][0]["settings"]
    for name, value in single_domain_runtime_switches(P3_LEGACY_RRTMG_PROFILE_ID).items():
        assert resolved[name] == value, name


def test_the_p3_profile_is_on_the_front_doors_choice_list():
    """A profile no door offers is not a front door.

    The HRRR benchmark publishes the profile tuple as its
    ``--physics-profile`` choices and the registry route must declare the
    same list in the same order; a template present in one and not the
    other is a menu entry that fails at plan validation.
    """
    from woof.physics_compat import P3_LEGACY_RRTMG_PROFILE_ID
    from woof.physics_registry import physics_registry
    from tools.hrrr_single_domain_benchmark import runner_capabilities

    capabilities = runner_capabilities()
    assert P3_LEGACY_RRTMG_PROFILE_ID in capabilities["physics_profile_ids"]
    route = physics_registry()["runner_routes"][capabilities["runner"]]
    assert (route["source_template_ids"]["hrrr"]
            + route["expert_template_ids"]["hrrr"]
            ) == capabilities["physics_profile_ids"]


@pytest.fixture
def version_identity_bound(monkeypatch):
    """Stand down ONE unrelated, pre-existing refusal for the loader test.

    ``woof.experiment.build_experiment`` calls
    :func:`woof.provenance_gate.require_version_identity`, which refuses
    whenever the running tree's ``pyproject.toml`` version and the version
    the installed distribution reports disagree.  Running a suite out of a
    worktree is exactly that case on this machine: the interpreter resolves
    the ``woof`` distribution to the main checkout's metadata while the
    code executing is the worktree's, so EVERY experiment-config load
    refuses regardless of what the config says.  It is a real refusal doing
    its real job, and it is not about P3 -- the same condition already
    fails ``tests/test_composition_pricing.py`` on the branch this work
    forked from.

    So this silences exactly that one check and nothing else: the loader,
    the per-domain physics resolution and the radiation-arm assertions
    below all still run for real.  It is self-retiring -- once the tree
    under test is bound to its own metadata the early return fires and the
    gate exercises a completely unpatched loader.
    """
    import woof.provenance_gate as gate

    if gate.version_identity_refusal() is None:
        return                                  # already bound; no patch
    monkeypatch.setattr(gate, "version_identity_refusal",
                        lambda prov=None: None)


@pytest.mark.parametrize("config_name", ("p3_mp50_shared.toml",
                                         "p3_mp50_domain.toml"))
def test_the_shipped_p3_configs_select_p3_and_name_a_wired_radiation_arm(
        config_name, version_identity_bound):
    """Both selector spellings, and neither may name a dead remedy.

    woof resolves physics per DOMAIN and a selector may be written once
    under ``[shared]`` or on an individual ``[[domain]]``; the two reach
    RunConfig differently, and mp=50 reached the 1.9 assembly unpriceable
    through both.  What this ALSO checks is the half that ``woof check``
    cannot see: the radiation variant each file names has a coupling for
    mp=50.  Both files used to name ``rrtmg_legacy``, whose adapter raised
    NotImplementedError for 50 at the first radiation call.
    """
    from pathlib import Path

    from woof.config import RRTMG_VARIANT_LEGACY
    from woof.core.rrtmg_legacy import legacy_scheme_declares_radii
    from woof.core.rrtmgp import cloud_optics_scheme
    from woof.experiment import load_experiment

    path = Path(__file__).resolve().parents[1] / "configs" / config_name
    experiment = load_experiment(path)
    runs = [domain.run for domain in experiment.domains]
    assert [run.mp_physics for run in runs] == [50] * len(runs)

    for run in runs:
        if (run.ra_lw_physics, run.ra_sw_physics) != (4, 4):
            continue
        if run.ra_rrtmg_variant == RRTMG_VARIANT_LEGACY:
            assert legacy_scheme_declares_radii(run.mp_physics,
                                                run.use_mp_re) is True
        else:
            assert cloud_optics_scheme(run.mp_physics) == "p3"


def test_preflight_prices_the_two_radius_columns_p3_actually_packs():
    """Three would budget an array mp=50 never allocates.

    The rail and the adapter must budget the same thing.  P3 packs effc and
    effi and no effs, because ``woof/core/state.py`` allocates no effs for
    mp=50 -- the same fact as Registry.EM_COMMON:3043's missing re_snow.
    """
    from woof.config import RunConfig
    from woof.core.preflight import rrtmgp_column_shapes

    cfg = RunConfig(
        nx=4, ny=3, nz=12, dx=2000.0, dy=2000.0, ztop=8000.0, dt=10.0,
        run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=50, ra_physics=4, radt=12.0)
    shapes = set(rrtmgp_column_shapes(cfg))
    assert {"columns/effc", "columns/effi"} <= shapes
    assert "columns/effs" not in shapes


def test_p3_is_on_the_menu_every_picker_asks():
    """A row in a table nothing enumerates is not a selectable suite.

    ``SINGLE_DOMAIN_PHYSICS_PROFILES`` is the runner's table.  The DOOR is
    ``woof.physics_menu.WIZARD_PHYSICS_PROFILES``: it is the argparse
    ``choices`` of ``woof domain --physics-profile``, and it is the axis
    ``woof run-plan --physics-profiles`` crosses with every registered
    source to answer "what can this model run".  That document is the one
    computed object a GUI, a script or a person can ask; a suite absent
    from it cannot be picked from any door, and the wizard's parser
    REFUSES its id outright.  P3 landed in the runner's table and not in
    this one, so mp=50 was reachable only by hand-writing a TOML.
    """
    import argparse

    from woof.domain_wizard import register_cli
    from woof.physics_compat import P3_LEGACY_RRTMG_PROFILE_ID
    from woof.physics_menu import WIZARD_PHYSICS_PROFILES, shipped_profiles

    assert P3_LEGACY_RRTMG_PROFILE_ID in WIZARD_PHYSICS_PROFILES
    assert P3_LEGACY_RRTMG_PROFILE_ID in shipped_profiles()

    subparsers = argparse.ArgumentParser().add_subparsers()
    register_cli(subparsers)
    action, = [candidate
               for candidate in subparsers.choices["domain"]._actions
               if candidate.dest == "physics_profile"]
    assert P3_LEGACY_RRTMG_PROFILE_ID in action.choices


def test_the_menu_offers_p3_wherever_it_offers_its_own_twin():
    """P3 is the Thompson legacy-RRTMG row with mp 8 -> 50 and nothing else.

    So the two suites can differ on a route's answer only where that route
    judges the MICROPHYSICS.  The ADMISSIBILITY must therefore agree at
    every source -- which is what this gate pins -- so a source that offers
    the Thompson twin and hides P3 (or the reverse) is a defect in the menu
    rather than a physics fact, named by id instead of by a count that
    would go stale.  Refusal SENTENCES may still differ where a route
    judges the scheme itself, and that difference is real.

    It also pins the ordering property the module's own docstring claims:
    P3 must not be the head, because the head is what
    ``default_profile_for`` hands every source that admits it, and adding
    a suite must not move any source's default.  The head is the Morrison
    suite, which is why that id is asserted by name below and not derived
    from the twin: the twin is P3's composition sibling, the head is a
    separate fact about ordering, and fusing the two would let a default
    move without this gate noticing.
    """
    from woof.physics_compat import (MORRISON_PROFILE_ID,
                                      P3_LEGACY_RRTMG_PROFILE_ID,
                                      THOMPSON_LEGACY_RRTMG_PROFILE_ID)
    from woof.physics_menu import (WIZARD_PHYSICS_PROFILES,
                                    admissible_profiles,
                                    default_profile_for)
    from woof.runplan import physics_profile_menu

    assert WIZARD_PHYSICS_PROFILES[0] != P3_LEGACY_RRTMG_PROFILE_ID

    document = physics_profile_menu()
    assert "error" not in document
    disagreeing = []
    for source in document["sources"]:
        by_id = {row["profile_id"]: row for row in source["profiles"]}
        p3 = by_id[P3_LEGACY_RRTMG_PROFILE_ID]
        twin = by_id[THOMPSON_LEGACY_RRTMG_PROFILE_ID]
        if p3["admissible"] != twin["admissible"]:
            disagreeing.append((source["source_id"], p3["why_not"],
                                twin["why_not"]))
        # P3 may not be a source's default merely by being listed: the
        # default is the head of the admissible set, and the head is
        # ahead of P3 in every set that contains both.
        assert not p3["is_default"], source["source_id"]
    assert disagreeing == []
    assert P3_LEGACY_RRTMG_PROFILE_ID in admissible_profiles("gfs")
    assert default_profile_for("gfs") == MORRISON_PROFILE_ID


def test_the_p3_ice_radius_band_is_the_shipped_tables_own_range():
    """The band that killed a real forecast, derived instead of typed.

    THE BREAKAGE.  A 6 h GFS forecast on the shipped p3 suite died at its
    first radiation call -- 12 steps in, radt = 12 min -- with "effi is
    outside the physical-plausibility band [1.0, 600.0] microns ... a radii
    writer probably emitted another unit".  No writer had.  Over 5,593,644
    cells the field read min 1.3832 and max 21495 microns, and those are
    EXACTLY the endpoints of column 6 of the shipped lookup table, which is
    the only thing ``diag_effi`` is ever assigned from (module_mp_p3.F:1610
    and woof/core/kernels/p3.cu:1830).  P3 is the one shipped scheme WRF
    does NOT clamp in microphysics -- module_microphysics_driver.F:1597-1598
    hands ``re_ice`` straight out -- because RADIATION caps it, at 130 um
    with the ``MIN(0.99, (130/res)^2)`` mass discount
    (module_ra_rrtmg_lw.F:12515-12532, _sw.F:11055-11067).

    So the band is READ OFF the table rather than written down, and this
    gate pins that it is: change the table and the band follows; type a
    number and this fails.
    """
    import numpy as np

    from woof.core.p3_tables import load_lookup_table_1, p3_table_root
    from woof.core.rrtmgp import (EFFECTIVE_RADIUS_PLAUSIBLE_UM,
                                   P3_ICE_RADIUS_TABLE_COLUMN,
                                   effective_radius_bands,
                                   p3_ice_radius_band_um)

    itab, _ = load_lookup_table_1(p3_table_root())
    column = itab[:, :, :, P3_ICE_RADIUS_TABLE_COLUMN - 1]
    scale = np.float32(1.0e6)
    lower, upper = p3_ice_radius_band_um()
    # float32, the same multiply woof/core/p3.py performs on the way to
    # state.effi.  A float64 product lands one ULP off and the cell holding
    # the table maximum would then test outside its own band.
    assert lower == float(np.float32(column.min()) * scale)
    assert upper == float(np.float32(column.max()) * scale)

    bands = effective_radius_bands("p3")
    assert bands["effi"][1] == upper
    # The snow slot carries the ICE radius under WRF's P3 remap
    # (resnow1D = MAX(10., re_ice*1.E6), _lw.F:12256, _sw.F:10857), so it
    # is judged by the ice radius's range.
    assert bands["effs"][1] == upper
    # The liquid band is untouched: effc comes from P3's own gamma
    # relation (module_mp_p3.F:1557), not from any table.
    assert bands["effc"] == EFFECTIVE_RADIUS_PLAUSIBLE_UM["effc"]
    # NOT A WIDENED TOLERANCE: no other scheme's band moves at all.
    for scheme in ("kessler", "wsm6", "thompson", "nssl", "morrison"):
        assert effective_radius_bands(scheme) == EFFECTIVE_RADIUS_PLAUSIBLE_UM
    # And the check keeps its teeth: P3 returns METRES and woof converts,
    # so an unconverted emission is ~1.4e-6 and still falls out the bottom.
    assert float(column.min()) < bands["effi"][0]
    assert float(column.max()) < bands["effi"][0]


@requires_gpu
def test_p3s_largest_table_radius_reaches_radiation_at_wrfs_130um_cap():
    """The value that killed the forecast now runs, capped WRF's way.

    Two halves, and both matter.  (1) A column carrying P3's table-maximum
    ice radius is ACCEPTED by the validating path -- that is the defect
    this closes.  (2) It is then capped and discounted exactly as WRF's
    iceflg = 5 block does, so admitting it does not mean radiating a
    21.5 mm crystal: the radius that reaches cloud optics is 130 um (the
    adapter's diameter form, 2 x 130, clipped to the table's 180 um top)
    and the mass is scaled by (130/res)^2.
    """
    import cupy as cp
    import numpy as np

    from woof.core.rrtmgp import (effective_radius_bands,
                                   hydrometeor_paths,
                                   p3_ice_radius_band_um)

    ncol, nlay = 4, 3
    plev = cp.asarray(
        np.linspace(100000.0, 10000.0, nlay + 1, dtype=np.float32)[None, :]
        * np.ones((ncol, 1), dtype=np.float32))
    qc = cp.zeros((ncol, nlay), dtype=cp.float32)
    qi = cp.full((ncol, nlay), 1.0e-4, dtype=cp.float32)
    effc = cp.full((ncol, nlay), 10.0, dtype=cp.float32)
    _, biggest = p3_ice_radius_band_um()
    effi = cp.full((ncol, nlay), np.float32(biggest), dtype=cp.float32)

    paths = hydrometeor_paths(
        plev, qc, qi=qi, microphysics="p3", effc=effc, effi=effi,
        validate=True)

    # (1) accepted -- and the generic band would have refused it.
    assert biggest > effective_radius_bands("thompson")["effi"][1]
    # (2) capped: 2 x 130 um = 260 um is above the table top, so the clip
    # to 180 um is what shows; the point is that nothing anywhere near
    # 2 x 21495 survives.
    dgice = cp.asnumpy(paths.dgice)
    assert float(dgice.max()) <= 180.0
    # and discounted: the ice path is qi x MIN(0.99, (130/res)^2) x dp/g,
    # which at this radius is a factor of 3.66e-5, not 0.99.
    factor = (130.0 / biggest) ** 2
    dp = np.abs(np.diff(cp.asnumpy(plev)[0]))
    expected = 1.0e-4 * factor * dp[0] / 9.80665 * 1000.0
    ciwp = cp.asnumpy(paths.ciwp)
    assert float(ciwp[0, 0]) == pytest.approx(expected, rel=2.0e-3)


#: The one column, out of 114,156, that took a real 6 h GFS forecast
#: non-finite.  Lifted from the live device state at the P3 call that did it
#: (call 284, 2026-08-29 22:44Z, grid point i=115 j=12 of the shipped
#: p3-mp50 suite at 12 km), so the gate below runs on the model's own
#: numbers rather than on a constructed cold column.
_OVERFLOW_COLUMN = "p3_cold_cloud_overflow_column.npz"


def _cold_cloud_column():
    import pathlib

    import numpy as np

    path = (pathlib.Path(__file__).resolve().parent / "data"
            / _OVERFLOW_COLUMN)
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def test_the_cold_cloud_column_still_overflows_wrfs_own_expression():
    """The gate's teeth: prove the hazard is real before proving it fixed.

    ``module_mp_p3.F:2994-2996`` is
    ``Q_nuc = cons6*gamma(7.+mu_c)*tmp1*dum**2`` and Fortran does not fix
    the association, so the partial product is what overflows.  This
    reconstructs the expression from the column's OWN state, in float32,
    left to right, and requires +Inf -- if a future change made the inputs
    tame, the fix below would be passing for no reason and this says so.
    """
    import numpy as np

    from woof.core.p3 import AIMM, CONS6, NCCNST, _gammaf, f32, get_cloud_dsd2

    column = _cold_cloud_column()
    k = 46  # The overflowing layer of the captured forecast column.
    pressure = f32(column["pres"][k])
    theta = f32(column["th"][k])
    exner = f32((pressure / f32(1.0e5)) ** f32(287.0 / 1004.5))
    temperature = f32(theta * exner)
    rho = f32(pressure / (f32(287.0) * temperature
                          * (f32(1.0) + f32(0.608) * column["qv"][k])))
    assert 195.0 < float(temperature) < 205.0
    _nc, mu_c, _nu, lamc, _cdist, cdist1 = get_cloud_dsd2(
        f32(column["qc"][k]), NCCNST / rho, rho, f32(1.0))

    with np.errstate(over="ignore"):
        dum = f32((f32(1.0) / lamc) ** f32(3.0))
        tmp1 = f32(cdist1 * np.exp(AIMM * (f32(273.15) - temperature)))
        partial = f32(CONS6 * _gammaf(f32(7.0) + mu_c) * tmp1)
        chain = f32(partial * dum ** f32(2.0))
    assert not np.isfinite(partial), "the partial product must overflow"
    assert not np.isfinite(chain)
    # ... while the mathematical answer is an ordinary float32.
    exact = np.float64(CONS6) * np.float64(_gammaf(f32(7.0) + mu_c)) \
        * np.float64(cdist1) \
        * np.exp(np.float64(AIMM) * (np.float64(273.15)
                                     - np.float64(temperature))) \
        * np.float64(dum) ** 2
    assert np.isfinite(np.float32(exact))
    assert 1.0e-3 < float(exact) < 1.0e-2


def test_p3_returns_a_finite_column_where_the_forecast_went_nan():
    """The CPU arm on the model's own numbers, at the call that failed.

    Before the overflow rescue this returned NaN in ``th`` and ``qv`` at
    k = 45 and 46 -- the same two cells, and the same two fields, the
    full-state health gate reported when the forecast died at
    ``post-d01-sync.d01: thp(45, 12, 115)``.
    """
    import numpy as np

    from woof.core.p3 import mp_p3_wrapper_wrf, p3_init

    column = _cold_cloud_column()

    def slab(name):
        return np.ascontiguousarray(
            column[name].astype(np.float32)).reshape(1, -1)

    th, qv, qc, qr = slab("th"), slab("qv"), slab("qc"), slab("qr")
    nr, qi, ni = slab("nr"), slab("qi"), slab("ni")
    qir, qib = slab("qir"), slab("qib")
    th_old, qv_old = slab("th_old"), slab("qv_old")
    pres, dz = slab("pres"), slab("dz")
    nk = th.shape[1]
    zeros2 = lambda: np.zeros((1, nk), np.float32)   # noqa: E731
    zeros1 = lambda: np.zeros(1, np.float32)         # noqa: E731
    refl, effc, effi = zeros2(), zeros2(), zeros2()
    vmi, di, rhopo = zeros2(), zeros2(), zeros2()
    rainnc, rainncv, sr = zeros1(), zeros1(), zeros1()
    snownc, snowncv = zeros1(), zeros1()

    mp_p3_wrapper_wrf(
        th, qv, qc, qr, nr, qi, qir, ni, qib, th_old, qv_old, pres, dz,
        float(column["dt"]), int(column["it"]), rainnc, rainncv, sr,
        snownc, snowncv, refl, effc, effi, vmi, di, rhopo,
        runtime=p3_init())

    for name, array in (("th", th), ("qv", qv), ("qc", qc), ("qr", qr),
                        ("nr", nr), ("qi", qi), ("ni", ni), ("qir", qir),
                        ("qib", qib), ("effc", effc), ("effi", effi)):
        assert np.isfinite(array).all(), f"{name} went non-finite"
    # and it is a physical answer, not a clamp to zero: the cold cloud
    # water freezes, and the column keeps its ice.
    assert float(qv[0, 45]) > 0.0
    assert 150.0 < float(th[0, 45] * (pres[0, 45] / 1.0e5) ** (287.0 / 1004.5)) < 260.0


@requires_gpu
def test_the_cuda_arm_is_finite_where_the_forecast_died():
    """The device half of the same fix, on the same column.

    The rescue is written twice -- ``woof/core/p3.py`` and
    ``woof/core/kernels/p3.cu`` -- and it is the CUDA one that a real run
    executes, so it needs its own gate.  Compiling the shipped source with
    the rescue branch forced OFF reproduces the original defect exactly
    (NaN in ``th`` and ``qv`` at k = 45 and 46 and nowhere else), which is
    what makes the ON result evidence rather than a coincidence.

    WHAT THIS GATE DELIBERATELY DOES NOT ASSERT.  Through 2.8.7 the CUDA
    and CPU arms agreed on theta here to 3.03e-5 relative but disagreed in
    the ICE family (device ``qir`` = 0 where the CPU arm carried
    5e-10..2.3e-6, ``qi`` up to 2.5 %).  That was the device's subnormal
    flush (cupy.RawModule's appended ``-ftz=true``), fixed 2026-10-07 on
    development branch: re-measured on test host, theta is now bit-identical
    between the arms and the ice family differs by at most 9e-11 kg/kg
    (``ni`` by 292 /kg) in a few cells.  That remainder is the CPU arm's
    own distance from WRF (its NumPy float32 libm calls); the device arm
    equals WRF 4.6.1's Fortran word for word on this column (driven from
    it = 1) except levels 44-46, where WRF's float32 overflow strikes and
    WRF returns NaN (tools/p3_wrf461_oracle, X02).  Pinning a
    tolerance on the CPU arm here would accept that open gap, so the gate
    stays on finiteness and theta.
    """
    import cupy as cp
    import numpy as np

    from woof.core import p3_device as PD
    from woof.core.p3 import mp_p3_wrapper_wrf, p3_init

    column = _cold_cloud_column()
    nk = int(column["th"].size)
    names = ("th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib",
             "th_old", "qv_old", "pres", "dz")
    host = {name: np.ascontiguousarray(
        column[name].astype(np.float32)).reshape(1, -1) for name in names}

    def device_run(source=None):
        fields = {name: cp.asarray(value.reshape(1, nk).T.copy())
                  for name, value in host.items()}
        fields["ssat"] = cp.zeros((nk, 1), cp.float32)
        fields["nc"] = cp.zeros((nk, 1), cp.float32)
        diag = {slot: cp.zeros((nk, 1), cp.float32) for slot in PD.DIAG_SLOTS}
        surf = {slot: cp.zeros(1, cp.float32) for slot in PD.SURF_SLOTS}
        PD.run_p3_device(
            fields, diag, surf, workspace=PD.make_workspace(1, nk),
            tables=PD.device_tables(p3_init()), dt=float(column["dt"]),
            it=int(column["it"]), log_predictNc=False,
            arm=PD.CONFIG_ARM["cuda"],
            module=None if source is None else PD.p3_module(source=source))
        return {name: cp.asnumpy(value).ravel()
                for name, value in fields.items()}

    live = device_run()
    for name, values in live.items():
        assert np.isfinite(values).all(), f"cuda {name} went non-finite"

    # TEETH: the same kernel with the rescue branch compiled out is the
    # defect, in the same two cells.
    disabled_source = PD.p3_source().replace(
        "if (!isfinite(Q_nuc) || !isfinite(N_nuc)) {", "if (false) {")
    assert disabled_source != PD.p3_source(), "the rescue branch moved"
    broken = device_run(disabled_source)
    nan_cells = sorted(int(k) for k in np.flatnonzero(~np.isfinite(broken["th"])))
    assert nan_cells == [45, 46], nan_cells
    assert sorted(int(k) for k in
                  np.flatnonzero(~np.isfinite(broken["qv"]))) == [45, 46]

    # And the arms agree on the field that failed.
    cpu = {name: value.copy() for name, value in host.items()}
    zeros2 = [np.zeros((1, nk), np.float32) for _ in range(6)]
    zeros1 = [np.zeros(1, np.float32) for _ in range(5)]
    mp_p3_wrapper_wrf(
        cpu["th"], cpu["qv"], cpu["qc"], cpu["qr"], cpu["nr"], cpu["qi"],
        cpu["qir"], cpu["ni"], cpu["qib"], cpu["th_old"], cpu["qv_old"],
        cpu["pres"], cpu["dz"], float(column["dt"]), int(column["it"]),
        *zeros1, *zeros2, runtime=p3_init())
    assert np.isfinite(cpu["th"]).all()
    relative = np.abs(live["th"] - cpu["th"].ravel()) / np.abs(cpu["th"].ravel())
    assert float(relative.max()) < 1.0e-4, float(relative.max())
