"""A named WRF V4 release imports with the scheme generations it runs.

Origin: the OU Oman-UAE case (2026-10-28, run B).  A WRF 4.6 namelist
imported with the line alone ("4") ran WRF 4.8's YSU entrainment cap and
solar constants, the engine's newest, so WOOF strict differed from
wrf.exe 4.6.1 strict before the first radiation call.  The release now
pins them, from ``--wrf-version 4.6.1`` or from the files' own TITLE on
the WRF-file door.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from types import SimpleNamespace
import tomllib

import pytest

from woof.namelist_contract import _base_pair
from woof.namelist_import import (WRF_RELEASE_GENERATIONS,
                                   import_parsed_namelists,
                                   wrf_release_generations,
                                   wrf_version_argument)
from woof.verify_exact.replay import WRF_461_REFERENCE_GENERATIONS

WRF_461 = {"solar_generation": "wrf_461", "ysu_version": "wrf_461",
           "urban_water_balance": "wrf_471", "noahmp_snow_age": "wrf_461"}


@pytest.mark.parametrize("version, expected", [
    ("4.6.1", WRF_461),
    ("4.6", WRF_461),
    ("4.7.1", {k: v for k, v in WRF_461.items() if k != "noahmp_snow_age"}),
    ("4.8", {}),
    ("4.9.1", {}),
    ("4", {}),
    ("3", {}),
    ("3.9", {}),
])
def test_a_release_names_the_generations_its_code_runs(version, expected):
    assert wrf_release_generations(version) == expected


def test_the_replay_table_and_the_release_table_agree_on_461():
    """The verify-exact replay pinned these four by hand; the importer
    now derives the same four from the release."""
    assert wrf_release_generations("4.6.1") == WRF_461_REFERENCE_GENERATIONS
    assert {row[0] for row in WRF_RELEASE_GENERATIONS} == set(WRF_461)


def test_the_flag_takes_a_release_and_refuses_nonsense():
    assert wrf_version_argument("4.6.1") == "4.6.1"
    assert wrf_version_argument("4") == "4"
    for bad in ("5.0", "4.x", "latest"):
        with pytest.raises(argparse.ArgumentTypeError):
            wrf_version_argument(bad)


def _shared(version):
    wps, inp = deepcopy(_base_pair())
    text, report = import_parsed_namelists(
        wps, inp, wps_path="namelist.wps", input_path="namelist.input",
        wrf_version=version, wrf_version_source=f"--wrf-version {version}")
    return tomllib.loads(text)["shared"], report


def test_importing_a_461_namelist_runs_wrf_461_ysu_and_sun():
    shared, report = _shared("4.6.1")
    for key, value in WRF_461.items():
        assert shared[key] == value
    reasons = {row.key: row.reason for row in report.defaults_applied}
    assert "WRF 4.6.1" in reasons["ysu_version"]
    assert "--wrf-version 4.6.1" in reasons["solar_generation"]


def test_a_bare_line_keeps_the_engine_defaults():
    shared, _ = _shared("4")
    for key in WRF_461:
        assert key not in shared


def test_the_door_reads_the_release_from_the_files_title():
    from woof.wrfinput_door import producing_wrf_release

    def meta(title):
        return SimpleNamespace(global_attributes={"TITLE": title})

    real = " OUTPUT FROM REAL_EM V4.6.1 PREPROCESSOR"
    export = " OUTPUT FROM GPUWM NATIVE WRF V4.6.1-COMPAT DIRECT EXPORTER"
    assert producing_wrf_release({1: meta(real), 2: meta(real)}) == "4.6.1"
    assert producing_wrf_release({1: meta(export)}) == "4.6.1"
    assert producing_wrf_release({1: meta(" OUTPUT FROM REAL_EM V4.7 X")}) == "4.7"
    assert producing_wrf_release({1: meta(real),
                                  2: meta(" OUTPUT FROM REAL_EM V4.7.1 X")}) is None
    assert producing_wrf_release({1: meta(" OUTPUT FROM REAL_EM V3.9pre#2 X")}) is None
