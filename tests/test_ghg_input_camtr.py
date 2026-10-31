"""WRF ghg_input = 1: CAMtr gases, bitwise to WRF v4.6.1's read_CAMgases.

The oracle rows are REAL(8) words printed by WRF v4.6.1
phys/module_ra_clWRF_support.F itself (compiled with the pinned strict
FCOPTIM, -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize, gfortran
13) reading run/CAMtr_volume_mixing_ratio.SSP245, at six (year, REAL(4)
julian) points: mid-year interpolation, a leap year either side of the
table's mid-June knot, the CFC onset era and an early-year day.

Origin: the OU Oman-UAE case (2026-10-28): wrf.exe 4.6.1 ran its default
ghg_input = 1 and WOOF had only the fixed gases, so longwave differed in
every column at step 1.
"""
from __future__ import annotations

import hashlib
import struct
from copy import deepcopy
import tomllib

import numpy as np
import pytest

from woof.core.camtr_gases import (CAMTR_DEFAULT, CAMTR_SHA256,
                                    camtr_trace_gases)

ORACLE = """\
2026 43680000 3F3C592F19FFADC8 3E969C5AC3F60EA6 3EC088F3789C6DEC 3DEBA4CD116178AA 3E0004939A2A2D6E
2026 43680555 3F3C592FE478444C 3E969C5B02EC8B89 3EC088F3C607393F 3DEBA4CBB0BCE64E 3E0004930A20840F
2024 426E0000 3F3BDDE11B9339FA 3E967A8320302699 3EC058C81613FD47 3DEC96DF9F759607 3E0069D283311925
2024 43B6FEB8 3F3C0778813B5F69 3E9685ECEE8C9FDE 3EC0698B3884B953 3DEC44E8CD5366EF 3E0047690B004829
1950 41240000 3F347EEBCD45E94C 3E9370A6A56EE8FA 3EB38040A2880E4D 3D6C8F9679620B50 3D99E46CC5A2B584
2026 42C80000 3F3C470920B5874C 3E96975FEBA05474 3EC082284552636F 3DEBC839BD951D98 3E00134FF6645905"""


def test_the_packaged_table_is_wrfs_ssp245():
    assert hashlib.sha256(CAMTR_DEFAULT.read_bytes()).hexdigest() == CAMTR_SHA256


@pytest.mark.parametrize("row", ORACLE.splitlines())
def test_camtr_gases_are_wrfs_words(row):
    fields = row.split()
    year = int(fields[0])
    julian = np.frombuffer(bytes.fromhex(fields[1])[::-1], np.float32)[0]
    want = [struct.unpack(">d", bytes.fromhex(word))[0] for word in fields[2:]]
    got = camtr_trace_gases(year, julian)
    assert [got[name] for name in ("co2", "n2o", "ch4", "cfc11", "cfc12")] == want


def test_the_legacy_adapter_takes_the_file_gases_per_call():
    from woof.core.rrtmg_legacy_prep import lw_trace_gases
    from woof.core.rrtmg_sw import option4_trace_gases
    from woof.core.trace_gases import FileGases, LEGACY_LW_GASES, LEGACY_SW_GASES, trace_gas_subset

    gases = camtr_trace_gases(2026, np.float32(232.0))
    lw = lw_trace_gases(2026, trace_gas_subset(FileGases(gases), LEGACY_LW_GASES))
    for name in ("co2", "n2o", "ch4", "cfc11", "cfc12"):
        assert lw[name] == np.float32(gases[name])
    co2, ch4, n2o, _o2 = option4_trace_gases(
        2026, trace_gas_subset(FileGases(gases), LEGACY_SW_GASES))
    assert (co2, ch4, n2o) == tuple(np.float32(gases[k]) for k in ("co2", "ch4", "n2o"))
    # A pre-1944 year has no CFCs in the file: zero reaches the solver.
    early = camtr_trace_gases(1930, np.float32(10.0))
    assert early["cfc11"] == 0.0
    assert lw_trace_gases(1930, trace_gas_subset(FileGases(early), LEGACY_LW_GASES))["cfc11"] == 0.0


def _import(physics=None, version="4", named=True):
    from woof.namelist_contract import _base_pair
    from woof.namelist_import import import_parsed_namelists
    wps, inp = deepcopy(_base_pair())
    inp["physics"].update(physics or {})
    text, report = import_parsed_namelists(
        wps, inp, wps_path="namelist.wps", input_path="namelist.input",
        wrf_version=version, rrtmg_variant="rrtmg_legacy",
        **({"wrf_version_source": f"--wrf-version {version}"} if named else {}))
    return tomllib.loads(text)["shared"], report


def test_a_v4_namelist_that_omits_ghg_input_runs_wrfs_default():
    inp_physics = {"ra_lw_physics": [4], "ra_sw_physics": [4]}
    shared, report = _import(inp_physics)
    assert shared.get("ghg_input") == 1
    assert any(row.key == "ghg_input" for row in report.defaults_applied)
    shared, _ = _import({**inp_physics, "ghg_input": [0]})
    assert "ghg_input" not in shared
    shared, _ = _import(inp_physics, version="3")
    assert "ghg_input" not in shared
    # A bare import names no WRF release (the operational HRRR namelist is a
    # WRF 3.9-fork file): it keeps the engine's fixed gases.
    shared, _ = _import(inp_physics, named=False)
    assert "ghg_input" not in shared


def test_ghg_input_is_one_tree_wide_value_inert_where_unread():
    """WRF's ghg_input is one namelist value; a domain whose radiation
    reads no gases ignores it (a d02 with radiation off validates)."""
    from woof.config import RunConfig, validate_run_config
    validate_run_config(RunConfig(
        nx=20, ny=20, nz=10, dx=3000.0, dy=3000.0, ztop=20000.0, dt=18.0,
        run_seconds=60.0, ra_physics=0, ghg_input=1))
