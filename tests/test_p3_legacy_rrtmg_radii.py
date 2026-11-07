"""P3 (mp_physics=50) reaches WRF's legacy-RRTMG radii coupling.

WHAT WAS BROKEN.  ``woof.core.rrtmg_legacy._MP_DECLARES_RADII`` is this
port's transcription of WRF v4.6.1's ``use_mp_re`` scheme table, and it
had no row for mp=50, so ``legacy_scheme_declares_radii(50, 1)`` raised
NotImplementedError.  The adapter calls it unconditionally on every
radiation step (``woof/core/rrtmg_legacy.py`` __call__), so a P3
forecast on ``ra_rrtmg_variant="rrtmg_legacy"`` died at its FIRST
radiation call -- and that variant was the way through that
``woof.config.validate_p3_radiation`` named while it still refused P3 on
the 4/4 RTE+RRTMGP pair (both retired once rrtmgp's ``50: "p3"`` row
coupled that pair), and the one ``configs/p3_mp50_shared.toml`` and
``configs/p3_mp50_domain.toml`` deliberately take.

WHY THE ROW IS NOT A BOOLEAN.  WRF builds the flags in two blocks
(``phys/module_physics_init.F``): the :988-1024 disjunction names
P3_1CATEGORY at :1017 and sets has_reqc = has_reqi = has_reqs = 1, then
the :1027-1033 P3 / Jensen-Ishmael override re-zeroes has_reqs.  P3 is
the first scheme in this table whose triple is not uniform, and both
uniform answers are wrong for it:

* all-True would demand a snow radius P3 never computes -- its package
  is ``moist:qv,qc,qr,qi`` with ``state:re_cloud,re_ice`` and no qs and
  no re_snow (``Registry/Registry.EM_COMMON:3038``), which is why
  woof's mp=50 state allocates effc/effi and no effs -- and it would
  route the one ice category into BOTH the ice and the snow optics.
* all-False would throw away the two radii P3 does predict, which is
  exactly what naming P3 in the :1004-1023 disjunction exists to prevent.

WHAT (1, 1, 0) BUYS.  It is the sole trigger of the wrapper's "special
case for P3 microphysics" (``module_ra_rrtmg_lw.F:12250-12261``,
``module_ra_rrtmg_sw.F:10853``), which this port already transcribed at
``woof/core/rrtmg_legacy_prep.py``.  Before the table row existed, no
adapter state could produce that triple and the transcribed branch was
unreachable code.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.core import rrtmg_legacy_prep as prep
from woof.core.rrtmg_legacy import (
    _MP_DECLARES_RADII,
    _MP_SNOW_RADII_SUPPRESSED,
    legacy_scheme_declares_radii,
    legacy_scheme_has_req,
)

from woof.config import P3_MP_PHYSICS as PORTED_P3_VARIANTS

P3_MP_PHYSICS = 50

#: WRF's second use_mp_re block (module_physics_init.F:1027-1033) names the
#: P3 family AND Jensen-Ishmael (mp=55, which also carries ice categories
#: with no snow species), so both reach the (1, 1, 0) triple.
JENSEN_ISHMAEL_MP_PHYSICS = 55
SNOW_RADII_SUPPRESSED = set(PORTED_P3_VARIANTS) | {JENSEN_ISHMAEL_MP_PHYSICS}


def test_p3_is_in_wrfs_use_mp_re_table_with_snow_suppressed():
    """module_physics_init.F:1017 names P3, :1027-1033 zeroes has_reqs."""
    assert _MP_DECLARES_RADII[P3_MP_PHYSICS] is True
    assert legacy_scheme_declares_radii(P3_MP_PHYSICS, 1) is True
    assert P3_MP_PHYSICS in _MP_SNOW_RADII_SUPPRESSED
    assert legacy_scheme_has_req(P3_MP_PHYSICS, 1) == (1, 1, 0)
    # The whole block sits inside WRF's IF (use_mp_re .EQ. 1) guard
    # (:988 .. :1058), so use_mp_re=0 zeroes all three, override included.
    assert legacy_scheme_has_req(P3_MP_PHYSICS, 0) == (0, 0, 0)


def test_the_uniform_schemes_keep_their_uniform_triples():
    """The override touches P3 and nothing else in the table."""
    for mp_physics in sorted(_MP_DECLARES_RADII):
        if mp_physics in SNOW_RADII_SUPPRESSED:
            continue
        flag = int(_MP_DECLARES_RADII[mp_physics])
        assert legacy_scheme_has_req(mp_physics, 1) == (flag, flag, flag)
    assert set(_MP_SNOW_RADII_SUPPRESSED) <= set(_MP_DECLARES_RADII)


def test_p3_is_the_only_selector_that_reaches_wrfs_p3_radii_branch():
    """(1, 1, 0) is the wrapper's P3 trigger, and every admitted P3 variant produces it.

    This is the assertion that fails without the table row: before it,
    every scheme's triple was uniform, so nothing could satisfy
    ``has_reqs == 0 and has_reqi != 0 and has_reqc != 0`` and the
    transcribed branch was dead code.
    """
    reaching = {
        mp_physics
        for mp_physics in _MP_DECLARES_RADII
        for use_mp_re in (0, 1)
        if (lambda t: t[2] == 0 and t[1] != 0 and t[0] != 0)(
            legacy_scheme_has_req(mp_physics, use_mp_re))
    }
    assert reaching == SNOW_RADII_SUPPRESSED


def _radii_case(has_req, *, re_snow=None):
    """Drive prep's radii block with one scheme's has_req triple."""
    kte = 4
    qi = np.array([0.0, 1.0e-4, 4.0e-4, 0.0], np.float32)
    qs_in = np.zeros(kte, np.float32)      # P3 has no snow species
    re_ice_m = np.array([25.0, 60.0, 300.0, 25.0], np.float32) * 1.0e-6
    out = prep._effective_radii(
        kte, 1, has_req[0], has_req[1], has_req[2],
        np.full(kte, 265.0, np.float32),             # t3d
        np.array([0.0, 1.0, 1.0, 0.0], np.float32),  # cldfra3d
        np.float32(1.0),                             # xland (land)
        np.full(kte, 12.0e-6, np.float32),           # re_cloud
        re_ice_m,
        np.zeros(kte, np.float32) if re_snow is None else re_snow,
        qi, qs_in, qi.copy())
    return out, qi, re_ice_m


def test_p3_triple_moves_the_single_ice_category_into_the_snow_optics():
    """module_ra_rrtmg_lw.F:12250-12261, transcribed and now reachable."""
    (inflg, iceflg, recloud1d, reice1d, resnow1d, qs1d, qi1d), qi, re_ice_m \
        = _radii_case(legacy_scheme_has_req(P3_MP_PHYSICS, 1))

    assert (inflg, iceflg) == (5, 5)
    # resnow1D = MAX(10., re_ice*1.E6) -- P3's OWN predicted ice radius.
    np.testing.assert_array_equal(
        resnow1d, np.maximum(np.float32(10.0),
                             (re_ice_m * np.float32(1.0e6)).astype(
                                 np.float32)))
    # QS1D = QI3D (the RAW array), QI1D = 0., reice1D = 10.
    np.testing.assert_array_equal(qs1d, qi)
    np.testing.assert_array_equal(qi1d, np.zeros(4, np.float32))
    np.testing.assert_array_equal(reice1d, np.full(4, 10.0, np.float32))
    # has_reqc = 1 still takes P3's cloud radius through the inflg>=3 path.
    assert float(recloud1d.min()) >= 2.5


def test_the_two_uniform_answers_would_both_be_wrong_for_p3():
    """Neither all-True nor all-False reproduces WRF for mp=50."""
    (_, _, _, reice_off, resnow_off, qs_off, qi_off), qi, _ = \
        _radii_case((0, 0, 0))
    # all-False discards P3's radii: ice falls back to EM_CORE's 10 um
    # constant and the single ice category never reaches the snow slot.
    np.testing.assert_array_equal(reice_off, np.full(4, 10.0, np.float32))
    np.testing.assert_array_equal(resnow_off, np.full(4, 10.0, np.float32))
    np.testing.assert_array_equal(qs_off, np.zeros(4, np.float32))
    np.testing.assert_array_equal(qi_off, qi)

    # all-True asks for a snow radius P3 does not allocate: prep fails
    # closed rather than radiating zeros for it.
    with pytest.raises(ValueError, match="fail-closed"):
        prep._require_radii("lwrad_prep", 1, 1, 1,
                            np.zeros(4, np.float32),
                            np.zeros(4, np.float32), None)
    # and given one anyway it takes the has_reqs!=0 branch, which never
    # reaches the P3 case: P3's ice would then be radiated as ice AND as
    # an invented snow species.
    (inflg_on, iceflg_on, _, _, resnow_on, qs_on, qi_on), qi, _ = \
        _radii_case((1, 1, 1), re_snow=np.full(4, 40.0e-6, np.float32))
    assert (inflg_on, iceflg_on) == (5, 5)
    np.testing.assert_array_equal(resnow_on, np.full(4, 40.0, np.float32))
    np.testing.assert_array_equal(qs_on, np.zeros(4, np.float32))
    np.testing.assert_array_equal(qi_on, qi)


def test_p3_ice_radius_cannot_trip_cldprmcs_snow_fatal():
    """The 130 um clamp, module_ra_rrtmg_lw.F:12519-12522.

    P3's lookup-table effective radius is unbounded above at the model's
    scale (the shipped p3_lookupTable_1.dat-v5.4_2momI reaches 2.1e4 um),
    and under (1, 1, 0) it lands in the snow slot, whose optics carry
    WRF's [5, 140] um wrf_error_fatal.  The wrapper clamps to 130 um with
    an area-conserving mass reduction first, so the fatal is unreachable
    -- this is why P3 is admitted where Morrison (EFFI to 525 um on the
    ICE slot, which has no such clamp) is not.
    """
    kte = 3
    resnow1d = np.array([50.0, 130.0, 21495.0], np.float32)
    qs1d = np.full(kte, 1.0e-4, np.float32)
    (_, _, cswpth, _, _, res, clamped) = prep._cloud_properties(
        kte, 5, 5, np.zeros(kte, np.float32), np.zeros(kte, np.float32),
        qs1d, np.ones(kte, np.float32),
        np.full(kte, 1000.0, np.float32), np.full(kte, 260.0, np.float32),
        9.81, np.float32(1.0), np.float32(0.0), np.float32(0.0),
        np.zeros(kte, np.float32), np.zeros(kte, np.float32), resnow1d)
    assert float(res.max()) <= 130.0
    assert 5.0 <= float(res.min()) <= 140.0
    np.testing.assert_array_equal(
        clamped, np.array([50.0, 130.0, 130.0], np.float32))
    # the clamped layer's snow path is scaled by (130/resnow)^2, not kept.
    assert float(cswpth[2]) < float(cswpth[1])
