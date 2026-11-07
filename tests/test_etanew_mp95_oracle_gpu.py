"""Eta Ferrier (mp_physics=95) graded against WRF v4.6.1's own Fortran.

The reference is ``tools/etanew_wrf461_oracle``: byte-unmodified
``phys/module_mp_etanew.F`` built three ways on the oracle host (-O0, the
WRF scalar referee's flags, and WRF's own compiled object), all three
bitwise equal, run over 80 columns x 50 levels for 10 carried steps.  The
committed fixtures keep steps 1, 2 and 10 of every level, every surface
row, the restart vector and the digests of both saturation tables.

* strict (``GPUWM_WRF_EXACT=1``): every output field 0 ULP;
* default arithmetic: inside the stated per-field tolerance below;
* a one-bit change to one kernel constant (the air-density reference
  RHO0 of the rain fall-speed correction) must fail the comparison.

Each arm runs the port in a child process, because the arithmetic mode is
fixed when ``woof.wrf_exact`` is imported.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import cupy  # noqa: F401  -- device test module (tests/conftest.py)
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "oracles" / "etanew"
TOOL = ROOT / "tools" / "etanew_wrf461_oracle"

#: Default-arithmetic tolerance (NVRTC contracts multiply-adds; the
#: Fortran does not).  Measured on the oracle columns; see REPORT.
DEFAULT_MAX_REL = {
    "th": 1e-5, "qv": 1e-2, "qt": 1e-2, "qc": 1e-2, "qr": 1e-2,
    "qs": 1e-2, "f_ice": 1e-2, "f_rain": 1e-2, "f_rimef": 1e-2,
    "rainnc": 1e-2, "rainncv": 1e-2, "sr": 1e-2, "value": 1e-6,
}


def _compare_module():
    spec = importlib.util.spec_from_file_location(
        "compare_etanew", TOOL / "compare_etanew.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_port(out: Path, *extra: str, strict: bool = True) -> None:
    env = dict(os.environ)
    env["GPUWM_WRF_EXACT"] = "1" if strict else "0"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                       if p])
    subprocess.run(
        [sys.executable, str(TOOL / "run_port.py"),
         str(FIXTURES / "columns_inputs.csv"), str(out), *extra],
        check=True, env=env, cwd=ROOT)


@requires_gpu
def test_strict_port_is_bitwise_equal_to_wrf(tmp_path):
    _run_port(tmp_path)
    ok, table = _compare_module().compare(
        FIXTURES, tmp_path, FIXTURES / "columns_inputs.csv", subset=True)
    bad = [row for row in table if row["differ"]]
    assert ok and not bad, bad


@requires_gpu
def test_default_arithmetic_is_inside_the_stated_tolerance(tmp_path):
    _run_port(tmp_path, strict=False)
    _, table = _compare_module().compare(
        FIXTURES, tmp_path, FIXTURES / "columns_inputs.csv", subset=True)
    over = [row for row in table
            if row["max_rel"] > DEFAULT_MAX_REL[row["field"]]]
    assert not over, over


@requires_gpu
def test_a_one_bit_constant_change_is_caught(tmp_path):
    _run_port(tmp_path, "--plant",
              "#define ETA_RHO0 1.194f=#define ETA_RHO0 0x1.31a9fep+0f")
    ok, table = _compare_module().compare(
        FIXTURES, tmp_path, FIXTURES / "columns_inputs.csv", subset=True)
    assert not ok
    assert any(row["differ"] for row in table if row["field"] == "qr")
