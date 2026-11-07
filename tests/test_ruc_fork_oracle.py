"""The RUC port under its WRF v4.0-4.5 names against the operational branch.

``woof/data/ruc/oracle_fork/`` holds the WRF v4.6.1 oracle harnesses
(``tools/ruc_wrf461_oracle/run_*.F90``, unchanged) compiled against the
operational RAP/HRRR branch's ``module_sf_ruclsm.F`` (NOAA-EMC/HRRR v4.1.21,
``sorc/hrrr_wrfarw.fd/WRFV3.9/phys``) instead of WRF v4.6.1's; rebuild with
``tools/ruc_hrrr_fork_oracle/build.sh``.  The harness inputs are the v4.6.1
oracle's, so every case runs the same column through the other lineage.

Each test here is one of ``tests/test_ruc.py``'s oracle tests, run with the
fixture directory pointed at the branch's output and the port's entry points
bound to the branch's lineage names (``soilprop``, ``snow``).  Routines the
two lineages share (``transf``, ``soilmoist``, ``soiltemp``, ``sice``,
``snowseaice``, ``qsn``, ``ruclsminit``) give the same numbers under both
builds, so they are not repeated.
"""

from functools import partial
import shutil
from pathlib import Path

import numpy as np
import pytest

import test_ruc as T

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "woof" / "data" / "ruc" / "oracle_fork"
SHARED = ROOT / "woof" / "data" / "ruc" / "oracle"


@pytest.fixture
def fork(monkeypatch, tmp_path):
    """Point test_ruc's fixtures at the branch oracle, the port at wrf_45."""
    oracle = tmp_path / "woof" / "data" / "ruc" / "oracle"
    oracle.mkdir(parents=True)
    for source in SHARED.glob("*.csv"):
        shutil.copy(source, oracle / source.name)
    for source in FORK.glob("*.csv"):
        shutil.copy(source, oracle / source.name)
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(T, "__file__", str(tmp_path / "tests" / "test_ruc.py"))
    lineage = {"soilprop": "wrf_45"}
    snow = {"soilprop": "wrf_45", "snow": "wrf_45"}
    for name, keywords in (
        ("ruc_soil_properties", lineage),
        ("ruc_soil_step", lineage),
        ("ruc_snow_temperature_step", {"snow": "wrf_45"}),
        ("ruc_snow_soil_step", snow),
    ):
        if hasattr(T, name):
            monkeypatch.setattr(T, name, partial(getattr(T, name), **keywords))
    return T


def test_the_branch_oracle_is_present_and_differs_where_the_lineages_do():
    for name in ("soilprop", "soil", "snowtemp", "snowtemp_contract",
                 "snowsoil", "snowsoil_contract", "sfctmp", "lsmruc"):
        branch = (FORK / f"{name}.csv").read_text()
        assert branch.splitlines()[0] == (
            SHARED / f"{name}.csv").read_text().splitlines()[0], name
        assert branch != (SHARED / f"{name}.csv").read_text(), name


def test_soil_properties_match_the_branch(fork):
    fork.test_soil_properties_match_unmodified_wrf_oracle()


def test_snow_free_soil_step_matches_the_branch(fork):
    fork.test_complete_snow_free_soil_step_matches_unmodified_wrf_oracle()


@pytest.mark.parametrize("fixture", sorted(T._SNOWTEMP_FIXTURES))
def test_snow_temperature_matches_the_branch(fork, fixture):
    fork.test_snow_temperature_step_matches_unmodified_wrf_bit_for_bit(fixture)


def test_snow_soil_step_matches_the_branch(fork):
    fork.test_snow_soil_step_matches_unmodified_wrf_bit_for_bit()


def test_snow_soil_contract_matches_the_branch(fork):
    fork.test_snow_soil_step_contract_fixture_matches_unmodified_wrf_bit_for_bit()


#: SNOWFALLAC: the branch accumulates metres, the port millimetres (its
#: output schema's unit and ``wrf_461``'s), so its increment is compared at
#: a factor of 1000 instead of bitwise.
def _same_snowfall(got, after, before):
    port = (float(got) - float(before)) * 1.0e-3
    branch = float(after) - float(before)
    tolerance = (4 * float(np.spacing(np.float32(after)))
                 + 4.0e-3 * float(np.spacing(np.float32(got))))
    return abs(port - branch) <= tolerance


def test_sfctmp_matches_the_branch_with_pinned_residue_and_snowfall_units():
    names, field = T._sfctmp_oracle(FORK / "sfctmp.csv")
    residue = {}
    snowfall = 0
    for case in range(len(names)):
        values, keywords = T._sfctmp_call(field, case)
        actual = T.ruc_surface_temperature_step(
            values, **keywords, soilprop="wrf_45", snow="wrf_45")
        for name in T.RUC_SFCTMP_PROFILE_OUTPUTS:
            got = np.asarray(getattr(actual, name), dtype=np.float32)[:, 0]
            expected = field[f"{name}_after"][:, case]
            for level in range(9):
                if got[level].view(np.uint32) != expected[level].view(np.uint32):
                    residue[(name, level + 1, case + 1)] = T._sfctmp_ulp(
                        got[level:level + 1], expected[level:level + 1])
        for name in T.RUC_SFCTMP_COLUMN_OUTPUTS + ("iland",):
            got = np.asarray(getattr(actual, name), dtype=np.float32)
            expected = field[f"{name}_after"][0, case:case + 1]
            if name == "snowfallac":
                before = T._sfctmp_before(field, name)[0, case]
                snowfall += int(expected[0] != before)
                if not _same_snowfall(got[0], expected[0], before):
                    residue[(name, 0, case + 1)] = (float(got[0]),
                                                    float(expected[0]))
                continue
            if got.view(np.uint32)[0] != expected.view(np.uint32)[0]:
                residue[(name, 0, case + 1)] = T._sfctmp_ulp(got, expected)
    unexplained = {
        key: value for key, value in residue.items()
        if key not in FORK_SFCTMP_RESIDUE
        or value > FORK_SFCTMP_RESIDUE[key]
    }
    assert not unexplained, unexplained
    assert snowfall >= 5


#: Every cell the sfctmp dispatch does not reproduce bitwise against the
#: committed branch build, measured.  The committed oracle_fork CSVs were built
#: with gfortran 15.2 on a different C library (build.sh header), and these
#: are that library's float32 libm words, not the port's arithmetic: rebuilt
#: on test host with the glibc 2.39 toolchain every other RUC oracle uses
#: (development branch), this fixture reproduces with no residue at all.
#: The v4.6.1 fixture carried the same eleven-cell map until RUC moved to
#: WOOF's float32 libm.  Replacing the committed fork CSVs with the 2.39
#: build is a named follow-up for the fork lineage's owner: the rebuild also
#: moves harness inputs (sfctmp znt_before, LSMRUC run-2 znt/z0), so it is a
#: fixture regeneration with its own review, not a pin edit.
FORK_SFCTMP_RESIDUE = {
    ("dew", 0, 15): 22, ("eeta", 0, 15): 28, ("evapl", 0, 15): 28,
    ("fltot", 0, 15): 16384, ("qcg", 0, 15): 28, ("qfx", 0, 15): 17,
    ("qsg", 0, 15): 17, ("qvg", 0, 15): 17, ("s", 0, 15): 2,
}

#: Every cell the driver does not reproduce bitwise against the committed
#: branch build.  ``chklowq`` is not snow: the branch's LSMRUC clears it only
#: under MYJ (``(myjpbl).and.(qvatm.ge.q2sat*0.95)...``), v4.6.1 under any
#: boundary layer, and only MYJ reads it.  The rest is the committed
#: fixture's C library (see :data:`FORK_SFCTMP_RESIDUE`): against the W1
#: glibc 2.39 rebuild the port differs only in ``chklowq`` and in run-2
#: ``znt``/``z0`` that the rebuilt harness seeds differently.  Group 13's
#: snow words are a branch flip (melt-out on one side of a rounding boundary)
#: that the 2.39 rebuild does not show.
LSMRUC_BRANCH_RESIDUE = {
    ("chklowq", 10, 0): 1065353216,
    ("chklowq", 22, 0): 1065353216,
    ("chklowq", 34, 0): 1065353216,
    ("chklowq", 46, 0): 1065353216,
    ("rhosnf", 8, 0): 1, ("rhosnf", 20, 0): 1, ("rhosnf", 25, 0): 1,
    ("rhosnf", 32, 0): 2, ("rhosnf", 37, 0): 1, ("rhosnf", 44, 0): 2,
    ("acrunoff", 13, 0): 46, ("grdflx", 13, 0): 1047, ("hfx", 13, 0): 420,
    ("lh", 13, 0): 450, ("qfx", 13, 0): 333, ("qsfc", 13, 0): 40,
    ("qsg", 13, 0): 40, ("qvg", 13, 0): 40, ("sfcrunoff", 13, 0): 46,
    ("snom", 13, 0): 10076, ("snow", 13, 0): 78,
    ("snowfallac", 13, 0): (7.083414077758789, 0.014216944575309753),
    ("snowh", 13, 0): 200, ("soilmois", 13, 1): 3, ("soilmois", 13, 2): 1,
    ("soilt", 13, 0): 1,
    ("grdflx", 25, 0): 1, ("snowc", 25, 0): 1, ("snowh", 25, 0): 1,
    ("grdflx", 37, 0): 2, ("hfx", 37, 0): 160, ("lh", 37, 0): 106,
    ("qfx", 37, 0): 79, ("qsfc", 37, 0): 27, ("qsg", 37, 0): 27,
    ("qvg", 37, 0): 27, ("sh2o", 37, 1): 29, ("sh2o", 37, 2): 24,
    ("snowh", 37, 0): 1, ("soilt", 37, 0): 1, ("soilt1", 37, 0): 1,
    ("tsnav", 37, 0): 256, ("tso", 37, 1): 1, ("tso", 37, 2): 1,
}


def test_lsmruc_matches_the_branch_but_for_the_pinned_residue():
    groups, field = T._lsmruc_oracle(FORK / "lsmruc.csv")
    residue = {}
    snowfall = 0
    for case in range(len(groups)):
        values, keywords = T._lsmruc_call(field, case)
        actual = T.ruc_land_surface_step(
            values, **keywords, soilprop="wrf_45", snow="wrf_45",
            irrigation="wrf_45", qvg_cold_start="air")
        for name in T.RUC_DRIVER_PROFILE_STATE:
            got = np.asarray(getattr(actual, name), dtype=np.float32)[:, 0]
            expected = field[T.LSMRUC_ALIAS.get(name, name)][:, case]
            for level in range(9):
                if got[level].view(np.uint32) != expected[level].view(np.uint32):
                    residue[(name, case, level + 1)] = T._sfctmp_ulp(
                        got[level:level + 1], expected[level:level + 1])
        for name in T.RUC_DRIVER_COLUMN_STATE:
            got = np.asarray(getattr(actual, name), dtype=np.float32)
            expected = T._lsmruc_expected(field, name)[0, case:case + 1]
            if name == "snowfallac":
                # LSMRUC zeroes it on the first step (:529).
                before = (np.float32(0.0) if int(field["ktau"][0, case]) == 1
                          else field["snowfallac_i"][0, case])
                snowfall += int(expected[0] != before)
                if not _same_snowfall(got[0], expected[0], before):
                    residue[(name, case, 0)] = (float(got[0]),
                                                float(expected[0]))
                continue
            if got.view(np.uint32)[0] != expected.view(np.uint32)[0]:
                residue[(name, case, 0)] = T._sfctmp_ulp(got, expected)
    unexplained = {
        key: value for key, value in residue.items()
        if key not in LSMRUC_BRANCH_RESIDUE
        or value > LSMRUC_BRANCH_RESIDUE[key]
    }
    assert not unexplained, unexplained
    assert snowfall >= 10
