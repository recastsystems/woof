"""CPU controls for complete microphysics-to-radiation declarations."""
import pytest


#: Every mp_physics selector ``validate_run_config`` accepts.  Asserted
#: against the validator below so this tuple cannot drift away from the
#: contract it claims to mirror -- and it HAD drifted: 9 (Milbrandt-Yau)
#: and 50 (P3 one-category) were admitted by production while this tuple
#: still read the pre-1.9 eight, so the three cross-checks that iterate it
#: had quietly stopped covering two schemes.  RE-DERIVED by running the
#: production validator over range(0, 107), exactly as
#: ``test_the_accepted_selector_list_this_module_uses_is_the_real_one``
#: does; not hand-extended.
_ACCEPTED_MP_PHYSICS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 24, 26, 27, 28, 30, 38, 40, 50, 51, 52, 53, 55, 56, 95, 96, 97, 106, 900)

#: Coupled selectors the run door still holds for a named guard: RCON (29)
#: has its cloud-optics row, but validate_run_config refuses it until its
#: release speed proof lands (RCON-SPEED-01). Retire with that guard.
_HELD_COUPLED_MP_PHYSICS = (29,)

#: The admitted selectors without a cloud-optics coupling: FAST_SBM (30)
#: alone, refused against the 4/4 pair with the legacy-RRTMG remedy until
#: its bin effective-radius integrals are qualified.  Before it this was
#: empty.  Purdue Lin (2) joins it: Lin produces no effective radii
#: (Registry.EM_COMMON:3016), so the legacy engine diagnoses its radii.  P3 (50) left this tuple when its single-ice
#: remap row landed; Milbrandt-Yau (9) left it when its own-radii row
#: landed (``9: "milbrandt2"``: the block WRF ships commented out at
#: module_mp_milbrandt2mom.F:3351-3378, evaluated over the transported
#: number moments).  The tuple stays so the partition checks below keep
#: asserting that nothing is admitted and uncoupled.
# 3.0 ports UDM (27), old Eta Ferrier (95) and MAD-WRF (96) carry recorded
# exclusions in woof.core.rrtmgp too: legacy RRTMG serves them.
_NO_RTE_RRTMGP_CLOUD_OPTICS = (2, 13, 27, 30, 56, 95, 96)

#: The selectors the RTE+RRTMGP cross-checks below may actually drive.
#: Derived, so a scheme cannot be dropped from coverage by hand.
_RTE_RRTMGP_COUPLED_MP_PHYSICS = tuple(
    mp for mp in _ACCEPTED_MP_PHYSICS
    if mp not in _NO_RTE_RRTMGP_CLOUD_OPTICS)


def test_every_accepted_selector_is_judged():
    """The partition is exact, and the uncoupled half really refuses.

    Every selector ``validate_run_config`` admits is either coupled to
    RTE+RRTMGP by a row in ``_MP_CLOUD_OPTICS_SCHEME`` or refused against
    the 4/4 pair by name, with the remedy stated.  Nothing may be in
    neither set: that is the state mp=50 shipped in at 1.9 -- admitted,
    uncoupled, unrefused -- and it died at the first radiation call rather
    than at the door.
    """
    from woof.config import RunConfig, validate_run_config
    from woof.core.rrtmgp import _MP_CLOUD_OPTICS_SCHEME

    coupled = set(_MP_CLOUD_OPTICS_SCHEME)
    uncoupled = set(_NO_RTE_RRTMGP_CLOUD_OPTICS)
    assert coupled.isdisjoint(uncoupled)
    assert coupled | uncoupled == set(_ACCEPTED_MP_PHYSICS) | set(_HELD_COUPLED_MP_PHYSICS), (
        "these accepted selectors are neither coupled nor refused: "
        f"{sorted(set(_ACCEPTED_MP_PHYSICS) - coupled - uncoupled)}")
    assert uncoupled == {2, 13, 27, 30, 56, 95, 96}, (
        "the accepted selectors refused against RTE+RRTMGP are FAST_SBM, SBU, NTU and Lin; "
        "50, then 9, were retired by writing their rows")

    # Every coupled selector validates against the 4/4 pair on the DEFAULT
    # variant -- the pairing every bare configuration lands on.  mp=9 is
    # in this loop now; it used to be the one refusal, and with the
    # variant defaulting to rte-rrtmgp that refusal fired on every bare
    # mp=9 run.
    for mp_physics in sorted(coupled - set(_HELD_COUPLED_MP_PHYSICS)):
        if mp_physics == 0:
            continue
        cfg = validate_run_config(RunConfig(
            nx=4, ny=3, nz=12, dx=2000.0, dy=2000.0, ztop=8000.0, dt=10.0,
            run_seconds=0.0, time_step_sound=4, moist=True,
            mp_physics=mp_physics, ra_lw_physics=4, ra_sw_physics=4))
        assert cfg.ra_rrtmg_variant == "rte-rrtmgp", mp_physics


def test_the_uncoupled_selectors_are_a_decision_in_the_module_that_raises():
    """Keep exclusions explicit and retire them when the coupling exists."""
    from woof.core.rrtmgp import (
        _CLOUD_OPTICS_REMEDY,
        _MP_CLOUD_OPTICS_SCHEME, _NO_CLOUD_OPTICS_COUPLING,
        cloud_optics_scheme, scheme_is_ice_active, scheme_has_snow_species)

    # 1. The partition is stated HERE, in the module that raises.
    coupled = set(_MP_CLOUD_OPTICS_SCHEME)
    recorded = set(_NO_CLOUD_OPTICS_COUPLING)
    assert coupled.isdisjoint(recorded)
    assert coupled | recorded == set(_ACCEPTED_MP_PHYSICS) | set(_HELD_COUPLED_MP_PHYSICS), (
        "these accepted selectors have neither a cloud-optics row nor a "
        "recorded reason for not having one: "
        f"{sorted(set(_ACCEPTED_MP_PHYSICS) - coupled - recorded)}")
    assert (tuple(sorted(_NO_CLOUD_OPTICS_COUPLING))
            == tuple(sorted(_NO_RTE_RRTMGP_CLOUD_OPTICS)))
    assert set(_NO_CLOUD_OPTICS_COUPLING) == {2, 13, 27, 30, 56, 95, 96}, (
        "the recorded exclusions are Lin, SBU-YLin, NTU and FAST_SBM; implemented "
        "couplings retire their own refusal rows")
    assert "Lin supplies no effective radii" in _NO_CLOUD_OPTICS_COUPLING[2]
    with pytest.raises(NotImplementedError, match="Lin supplies no effective radii"):
        cloud_optics_scheme(2)

    # 2. The remedy sentence still exists for the refusal path an UNJUDGED
    #    selector takes (checked at 5.), and still names both live doors.
    assert "rrtmg_legacy" in _CLOUD_OPTICS_REMEDY
    assert "Dudhia" in _CLOUD_OPTICS_REMEDY

    # 3. P3's implemented coupling consumes its single ice category without
    #    claiming a snow species or asking for an unallocated snow radius.
    assert cloud_optics_scheme(50) == "p3"
    assert scheme_is_ice_active("p3")
    assert not scheme_has_snow_species("p3")
    assert 50 not in recorded

    # 4. Milbrandt-Yau's implemented coupling is its OWN row (not a borrowed
    #    Morrison or Kessler one), and Registry.EM_COMMON:3025's
    #    ``moist:qv,qc,qr,qi,qs,qg,qh`` makes it ice-active WITH a snow
    #    species, so cal_cldfra1 takes the QCLD = QI + QC + QS arm.
    assert cloud_optics_scheme(9) == "milbrandt2"
    assert scheme_is_ice_active("milbrandt2")
    assert scheme_has_snow_species("milbrandt2")
    assert 9 not in recorded

    # 5. The record does not soften the gate.  An unjudged selector still
    #    fails closed, and is told how to make the omission speak.
    with pytest.raises(NotImplementedError) as unjudged:
        cloud_optics_scheme(37)
    assert "add a row" in str(unjudged.value)
    assert "_NO_CLOUD_OPTICS_COUPLING" in str(unjudged.value)


def test_the_module_and_the_registry_builder_refuse_the_same_selectors():
    """Two authorities, one partition.

    ``tools/build_registry.py`` writes a ``refused_when`` constraint into
    the shipped registry for every implemented scheme with no
    cloud-optics row, keyed by its own reason table.  If that table and
    the module's diverge, the registry refuses a pairing the runtime
    couples or -- the direction that costs a forecast -- couples one the
    runtime raises on at the first radiation call.
    """
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.build_registry import _NO_RTE_RRTMGP_CLOUD_OPTICS_REASON

    from woof.core.rrtmgp import _NO_CLOUD_OPTICS_COUPLING

    assert (sorted(_NO_CLOUD_OPTICS_COUPLING)
            == sorted(_NO_RTE_RRTMGP_CLOUD_OPTICS_REASON))


def test_the_accepted_selector_list_this_module_uses_is_the_real_one():
    """``_ACCEPTED_MP_PHYSICS`` must be exactly what RunConfig admits.

    The three cross-checks below iterate this tuple; if it silently drifted
    away from ``woof/config.py`` they would stop covering a selector
    without failing.

    ``admits`` classifies a refusal by WHICH SELECTOR it names, not by one
    sentence's wording.  The generic tail says "mp_physics must be ...",
    but a scheme woof deliberately does not port gets a NAMED refusal
    instead -- WDM5/WDM7 (woof/config.py), and the P3 siblings on their
    own lane -- and those say why rather than reciting the admitted set.
    Matching one wording made this helper re-raise the named refusals, so
    the drift detector this test exists to be became an ERROR at the first
    scheme that got a good refusal message.  It still cannot be silenced: a
    ValueError that does not name ``mp_physics`` at all still propagates,
    and the loop below is what decides admission, never a literal.
    """
    from woof.config import RunConfig, validate_run_config

    refusals = {}

    def admits(mp_physics):
        try:
            validate_run_config(RunConfig(
                nx=4, ny=3, nz=12, dx=2000.0, dy=2000.0, ztop=8000.0,
                dt=10.0, run_seconds=0.0, time_step_sound=4, moist=True,
                mp_physics=mp_physics))
        except ValueError as error:
            message = str(error)
            if not message.startswith("mp_physics"):
                raise
            refusals[mp_physics] = message
            return False
        return True

    from woof.microphysics_schemes import NAMED_MP_IDS
    admitted = tuple(mp for mp in dict.fromkeys((*range(0, 107), *NAMED_MP_IDS)) if admits(mp))
    assert admitted == _ACCEPTED_MP_PHYSICS
    # WDM5 and WDM7 now have native ports and pass this same door.
    assert 14 not in refusals and 26 not in refusals
    assert all("mp_physics must be" in refusals[mp]
               for mp in (2, 5, 55) if mp in refusals)


def test_thompsonaero_mp28_resolves_to_the_thompson_cloud_optics_coupling():
    """mp_physics=28 must get classic Thompson's radiative coupling.

    WRF v4.6.1 authority, all in the stock tree:

    * ``Registry/Registry.EM_COMMON:3036`` --
      ``package thompsonaero mp_physics==28 - moist:qv,qc,qr,qi,qs,qg;
      scalar:...;state:re_cloud,re_ice,re_snow`` -- the same ``moist``
      inventory and the same three ``re_*`` state fields as line 3024's
      ``package thompson mp_physics==8``.
    * ``phys/module_physics_init.F:1005-1006`` names THOMPSON and
      THOMPSONAERO in ONE disjunction setting
      ``has_reqc = has_reqi = has_reqs = 1`` (:1021-1023); the P3 /
      Jensen-Ishmael ``has_reqs = 0`` override (:1027-1033) does not list
      THOMPSONAERO.
    * ``phys/module_radiation_driver.F``'s ``cal_cldfra1`` branches on
      ``mp_physics`` only for Ferrier (:3926-3937); both Thompson packages
      take the ``F_QI .and. F_QC .and. F_QS`` arm at :3870-3877.  The RRTMG
      wrappers likewise test the selector only for Ferrier/HWRF
      (``module_ra_rrtmg_lw.F:12131-12136``,
      ``module_ra_rrtmg_sw.F:10732-10737``).
    """
    from woof.core.rrtmgp import (
        _MP_CLOUD_OPTICS_SCHEME, cloud_optics_scheme, scheme_is_ice_active)

    assert cloud_optics_scheme(28) == "thompson"
    assert cloud_optics_scheme(28) == cloud_optics_scheme(8)
    assert scheme_is_ice_active(cloud_optics_scheme(28)) is True
    # Every accepted selector is judged; nothing falls through.  A
    # selector either has a cloud-optics row or is refused against
    # RTE+RRTMGP by name -- see test_every_accepted_selector_is_judged.
    assert (tuple(sorted(_MP_CLOUD_OPTICS_SCHEME))
            == tuple(sorted(mp for mp in _ACCEPTED_MP_PHYSICS + _HELD_COUPLED_MP_PHYSICS
                            if mp not in _NO_RTE_RRTMGP_CLOUD_OPTICS)))


def test_wdm6_mp16_resolves_to_the_wsm6_cloud_optics_coupling():
    """mp_physics=16 must get WSM6's radiative coupling, not Kessler's.

    WRF v4.6.1 authority, all in the stock tree:

    * ``Registry/Registry.EM_COMMON:3031`` --
      ``package wdm6scheme mp_physics==16 - moist:qv,qc,qr,qi,qs,qg;
      scalar:qnn,qnc,qnr;state:re_cloud,re_ice,re_snow`` -- WSM6's
      (:3021) ``moist`` inventory character for character, plus three
      transported numbers and the same three ``re_*`` state fields.
    * ``phys/module_physics_init.F:1013`` names WDM6SCHEME in the same
      ``use_mp_re`` disjunction as WSM6SCHEME (:1010), setting
      ``has_reqc = has_reqi = has_reqs = 1`` (:1021-1023); the P3 /
      Jensen-Ishmael ``has_reqs = 0`` override (:1027-1033) omits it.
    * ``module_radiation_driver.F``'s ``cal_cldfra1`` branches on
      ``mp_physics`` only for Ferrier (:3926-3937); WDM6 takes the
      ``F_QI .and. F_QC .and. F_QS`` arm at :3870-3877.

    The consequence of getting this wrong is not subtle and is the reason
    the table fails closed: an unmapped selector used to fall through to
    Kessler, and mp=28 spent four waves radiating overcast ice as clear
    sky.  What is CHECKED here is that mp=16 lands on the explicit-radius,
    ice-active branch -- WDM6's droplet radius is built from prognostic nc
    (effectRad_wdm6, module_mp_wdm6.F:3203-3213), but that changes the
    values the scheme supplies, not the branch that consumes them.
    """
    from woof.core.rrtmgp import (_MP_CLOUD_OPTICS_SCHEME,
                                   cloud_optics_scheme, scheme_is_ice_active)

    assert cloud_optics_scheme(16) == "wsm6"
    assert cloud_optics_scheme(16) == cloud_optics_scheme(6)
    assert scheme_is_ice_active(cloud_optics_scheme(16)) is True
    assert cloud_optics_scheme(16) != "kessler"
    # Every accepted selector is judged; nothing falls through.  A
    # selector either has a cloud-optics row or is refused against
    # RTE+RRTMGP by name -- see test_every_accepted_selector_is_judged.
    assert (tuple(sorted(_MP_CLOUD_OPTICS_SCHEME))
            == tuple(sorted(mp for mp in _ACCEPTED_MP_PHYSICS + _HELD_COUPLED_MP_PHYSICS
                            if mp not in _NO_RTE_RRTMGP_CLOUD_OPTICS)))
    # ... and the legacy engine agrees, which is the pairing that used to
    # make ice clouds appear or vanish with the radiation selector.
    from woof.core.rrtmg_legacy import _MP_DECLARES_RADII, legacy_ice_active

    assert legacy_ice_active(16) is True
    assert _MP_DECLARES_RADII[16] is True


def test_cloud_optics_scheme_fails_closed_on_an_unmapped_selector():
    """No silent Kessler default.

    A ``.get(mp, "kessler")`` default is exactly how mp=28 spent four waves
    radiating its ice clouds as clear sky, so an unmapped selector must
    raise instead of inheriting constant radii and an ice-free cloud
    fraction.
    """
    from woof.core.rrtmgp import cloud_optics_scheme

    for unmapped in (2, 37, 95):
        with pytest.raises(NotImplementedError, match="cloud-optics"):
            cloud_optics_scheme(unmapped)


def test_both_radiation_engines_agree_on_which_schemes_carry_ice():
    """The RTE+RRTMGP and legacy-RRTMG adapters must not disagree.

    ``woof/core/rrtmg_legacy.py`` already carried the WRF judgement for
    mp=28 (``_MP_DECLARES_RADII[28] = True``,
    ``_LEGACY_ICE_ACTIVE_MICROPHYSICS`` contains 28) while the default
    engine did not, so an operator's ice clouds appeared or vanished
    depending on which radiation engine they selected.  Both sides derive
    from the same Registry ``moist`` package, so they are pinned equal here
    for every selector woof accepts.
    """
    from woof.core.rrtmgp import (
        cloud_optics_scheme, scheme_is_ice_active, scheme_has_snow_species)
    from woof.core.rrtmg_legacy import (
        _MP_DECLARES_RADII, legacy_cloud_fraction_flags)

    for mp_physics in _RTE_RRTMGP_COUPLED_MP_PHYSICS:
        scheme = cloud_optics_scheme(mp_physics)
        flags = (scheme_is_ice_active(scheme), scheme_has_snow_species(scheme))
        assert flags == legacy_cloud_fraction_flags(mp_physics), (mp_physics, flags)
        # One-way implication: a scheme WRF hands its radii to
        # (module_physics_init.F:1005-1024 has_req*) must not land on
        # Kessler's constant 10 um / 50 um pair here.  Morrison is False in
        # the legacy table by WRF's own omission and is exempt from the
        # converse -- the RRTMGP adapter reconstructs its radii from the
        # number moments instead.
        if _MP_DECLARES_RADII[mp_physics]:
            assert scheme != "kessler", (
                f"mp_physics={mp_physics} declares effective radii to WRF's "
                "radiation but resolves to the constant-radius branch")




def test_ferrier_coupling_uses_total_ice_without_a_snow_species():
    from woof.core.rrtmgp import (cloud_optics_scheme, scheme_is_ice_active,
                                  scheme_has_snow_species)
    assert cloud_optics_scheme(5) == 'ferrier'
    assert scheme_is_ice_active('ferrier')
    assert not scheme_has_snow_species('ferrier')
