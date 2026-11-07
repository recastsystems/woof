"""LSMRUC against unmodified WRF under the operational HRRR surface switches.

``woof/data/ruc/oracle/lsmruc_hrrr_switches.csv`` is
``tools/ruc_wrf461_oracle/run_lsmruc_frac.F90``: the ``lsmruc.csv`` driver
with ``xice_threshold = 0.02`` (what the surface driver passes when
``fractional_seaice = 1``) and ``rdlai2d = .true.`` in both runs, and its
sea-ice columns moved into the band ``0.02 <= xice < 0.5`` that switch opens
(0.04, 0.12, 0.25, 0.45), plus two land columns at ``0 < xice < 0.02``.

The HRRR v4.1.21 fork's LSMRUC reads ``xice_threshold`` in the same two
expressions as WRF v4.6.1 (fork module_sf_ruclsm.F:845 and :1127 against
4.6.1 :860 and :1106) and ``rdlai2d`` only to skip the table LAI in SOILVEGIN
(fork :7028-7075 against 4.6.1 :6867-6913). This fixture proves the switch
wiring against WRF 4.6.1. It does not establish complete column identity
against the older fork.

Rebuilt on test host (development branch) with the toolchain ``lsmruc.csv``
was pinned on -- GNU Fortran 13.3.0 on glibc 2.39, the C library every WOOF
float32 libm word is graded against -- by ``build_lsmruc_frac.sh`` against
pristine WRF v4.6.1.  The earlier copy was built on glibc 2.43, whose
``tanhf``/``expf``/``powf`` round differently, and carried a 37-word residue
that was that library's, not the port's.  Against this build the port is
bitwise on every word but SFCEVP, which WRF counts twice and WOOF once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.test_ruc import (
    LSMRUC_ALIAS,
    RUC_DRIVER_COLUMN_STATE,
    RUC_DRIVER_PROFILE_STATE,
    _lsmruc_call,
    _lsmruc_expected,
    _lsmruc_oracle,
    _sfctmp_ulp,
    ruc_land_surface_step,
)

FIXTURE = (Path(__file__).parents[1] / "woof" / "data" / "ruc" / "oracle"
           / "lsmruc_hrrr_switches.csv")

#: Every output word the port does not reproduce bitwise, ``(field, group,
#: level)`` -> ULP (groups are 0-based (run, step, column) in file order;
#: level 0 is a column field): none, once SFCEVP is graded on WRF's single
#: count (``tests/test_ruc.py::_lsmruc_expected``).
RESIDUE = {}


def _replay(field, groups, **override):
    residue = {}
    for case in range(len(groups)):
        values, keywords = _lsmruc_call(field, case)
        keywords.update(override)
        actual = ruc_land_surface_step(values, **keywords)
        for name in RUC_DRIVER_PROFILE_STATE:
            got = np.asarray(getattr(actual, name), dtype=np.float32)[:, 0]
            want = field[LSMRUC_ALIAS.get(name, name)][:, case]
            for level in range(9):
                if got[level].view(np.uint32) != want[level].view(np.uint32):
                    residue[(name, case, level + 1)] = int(_sfctmp_ulp(
                        got[level:level + 1], want[level:level + 1]))
        for name in RUC_DRIVER_COLUMN_STATE:
            got = np.asarray(getattr(actual, name), dtype=np.float32)
            want = _lsmruc_expected(field, name)[0, case:case + 1]
            if got.view(np.uint32)[0] != want.view(np.uint32)[0]:
                residue[(name, case, 0)] = int(_sfctmp_ulp(got, want))
    return residue


def _ice_groups(field, groups):
    threshold = field["xice_threshold"][0]
    xice = field["xice"][0]
    return [case for case in range(len(groups))
            if threshold[case] <= xice[case] < np.float32(0.5)]


def test_fixture_is_the_hrrr_switch_set():
    groups, field = _lsmruc_oracle(FIXTURE)
    assert len(groups) == 48
    assert np.all(field["xice_threshold"] == np.float32(0.02))
    assert np.all(field["rdlai2d"])
    ice = _ice_groups(field, groups)
    # two sea-ice columns per run, two steps each, all in the opened band
    assert len(ice) == 8
    sub = [case for case in range(len(groups))
           if 0.0 < field["xice"][0, case] < field["xice_threshold"][0, case]]
    assert len(sub) == 4


def test_lsmruc_at_threshold_002_with_rdlai2d_matches_unmodified_wrf461():
    groups, field = _lsmruc_oracle(FIXTURE)
    residue = _replay(field, groups)
    assert residue == RESIDUE
    # The columns the switch moves are bitwise WRF.
    ice = set(_ice_groups(field, groups))
    assert ice
    on_ice = {key: ulp for key, ulp in residue.items() if key[1] in ice}
    assert on_ice == {}


def test_the_fixture_sees_the_threshold_and_rdlai2d():
    """Negative controls: the old pin and the table LAI each break it."""

    groups, field = _lsmruc_oracle(FIXTURE)
    ice = set(_ice_groups(field, groups))
    pinned = _replay(field, groups, xice_threshold=0.5)
    moved = {key[1] for key, ulp in pinned.items()
             if RESIDUE.get(key) != ulp}
    assert ice <= moved, sorted(ice - moved)
    table = _replay(field, groups, rdlai2d=False)
    assert {key for key, ulp in table.items() if RESIDUE.get(key) != ulp}, (
        "rdlai2d is invisible to the fixture")


def test_prescribed_lai_matches_the_pinned_forks_actual_column():
    """Prove the changed field against the fork without claiming other outputs."""
    groups, field = _lsmruc_oracle(FIXTURE.with_name("lsmruc_hrrr_v4121.csv"))
    changed = 0
    for case in range(len(groups)):
        values, keywords = _lsmruc_call(field, case)
        actual = ruc_land_surface_step(values, **keywords, soilprop="wrf_45")
        np.testing.assert_array_equal(
            np.asarray(actual.lai, np.float32), field["lai"][0, case:case + 1])
        pinned = ruc_land_surface_step(
            values, **{**keywords, "rdlai2d": False}, soilprop="wrf_45")
        changed += int(np.any(np.asarray(pinned.lai, np.float32)
                             != field["lai"][0, case:case + 1]))
    assert changed > 0
