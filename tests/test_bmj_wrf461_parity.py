"""Betts-Miller-Janjic (cu_physics = 2) against WRF v4.6.1, word for word.

The oracle is the byte-unmodified module_cu_bmj.F built and run by
tools/bmj_wrf461_oracle/build.sh: BMJINIT's tables, then four consecutive
BMJDRV calls over 96 columns in twelve families, in two arms
(bmj_rad_feedback off at DT=60 s, STEPCU=1; on at DT=30 s, STEPCU=3), so
the carried CLDEFI is read back by the scheme that wrote it.  Each step's
inputs are in the capture, so the port replays them and carries only its
own CLDEFI.  The committed capture is tests/data/oracles/bmj/oracle/*.bin.gz.

Two oracle executables share one object file: ``stock`` (GAMMA bound to
the C library's tgammaf) and ``crgamma`` (a correctly rounded tgammaf
interposed).  WOOF's gamma is correctly rounded by owner ruling -- a
DECLARED DIVERGENCE, the Grell-Freitas precedent -- so the port is graded
at 0 ULP against ``crgamma`` on every field, and against ``stock`` it may
differ only on the three fields that gamma reaches.

Three graders run the same kernel source: the host harness (the exact
nvrtc string compiled as C++, no GPU), and the device under default and
strict (GPUWM_WRF_EXACT=1) arithmetic.
"""

from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE_TOOL = ROOT / "tools" / "bmj_wrf461_oracle"
DATA = ROOT / "tests" / "data" / "oracles" / "bmj" / "oracle"
sys.path.insert(0, str(ORACLE_TOOL))
import bmj_oracle_io as io  # noqa: E402

#: The only fields the declared gamma divergence can reach
#: (module_cu_bmj.F:308 feeds PVPR -> CCLDFRA -> QCCONV/QICONV).
GAMMA_FIELDS = {"ccldfra", "qcconv", "qiconv"}
#: Largest stock-arm distance on those fields in the committed capture.
GAMMA_MAX_ULP = 6


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    """The capture, decompressed and digest-checked.  Arm 1 runs with the
    feedback off, so GAMMA is never called and the two oracle executables
    write byte-identical files (oracle-sha256sums.txt records both digests
    equal); only one is committed."""
    import hashlib

    out = tmp_path_factory.mktemp("bmj-oracle")
    for gz in DATA.glob("*.bin.gz"):
        with gzip.open(gz, "rb") as src, open(out / gz.name[:-3], "wb") as dst:
            shutil.copyfileobj(src, dst)
    shutil.copyfile(out / "stock-arm1.bin", out / "crgamma-arm1.bin")
    for line in (DATA / "oracle-sha256sums.txt").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name
    return out


def _spec():
    from woof.core.column_cumulus import column_kernel_spec
    spec = column_kernel_spec(2)
    assert spec is not None
    return spec


# ------------------------------------------------------------ metadata
def test_every_column_kernel_scheme_is_admitted():
    """The registry's column-kernel rows, the config's admitted set and
    the route's dispatch set are one set: a row nobody admits would be
    unreachable, an admitted value with no row would raise at attach."""
    from woof.config import COLUMN_KERNEL_CU_SCHEMES, CU_SCHEMES
    from woof.core.column_cumulus import column_kernel_selectors
    from woof.namelist_import import _CU_ALLOWED

    assert column_kernel_selectors() == frozenset(COLUMN_KERNEL_CU_SCHEMES)
    assert set(COLUMN_KERNEL_CU_SCHEMES) <= set(CU_SCHEMES)
    assert set(COLUMN_KERNEL_CU_SCHEMES) <= _CU_ALLOWED


def test_the_row_matches_the_kernel_and_the_contract():
    from woof.checkpoint_identity import CUMULUS_ALGORITHM_IDENTITIES
    from woof.physics_vertical_contract import BMJ_VERTICAL_LEVEL_BOUNDS

    spec = _spec()
    src = (ROOT / "woof" / "core" / "kernels" / "bmj.cu").read_text("utf-8")
    assert f'extern "C" __global__ void {spec.entry}(' in src
    assert f'extern "C" __global__ void {spec.tables["entry"]}(' in src
    assert spec.identity == CUMULUS_ALGORITHM_IDENTITIES[2]
    assert spec.vertical_minimum == BMJ_VERTICAL_LEVEL_BOUNDS[0]
    # BMJ_NSLOTS is the last enumerator; the row must reserve exactly it.
    enum = src.split("enum {", 1)[1].split("};", 1)[0]
    slots = [s.strip() for s in enum.replace("\n", " ").split(",") if s.strip()]
    assert slots[-1] == "BMJ_NSLOTS"
    assert spec.workspace_slots == len(slots) - 1
    # BMJINIT's cold start: CLDEFI = AVGEFI = (EFIMN+1.)*0.5 in float32.
    avgefi = np.float32(np.float32(np.float32(0.2) + np.float32(1.0))
                        * np.float32(0.5))
    assert spec.carried == (("cldefi", float(avgefi)),)


# ------------------------------------------------------------ host grader
@pytest.fixture(scope="module")
def host(oracle, tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("no host C++ compiler")
    import bmj_host_parity as hp
    build = tmp_path_factory.mktemp("bmj-host")
    return hp, hp.build(build, None), build


def test_host_tables_are_bmjinit_word_for_word(host, oracle):
    hp, lib, _ = host
    rep = hp.run(lib, oracle)
    assert rep["tables_max_ulp"] == 0, rep
    for name, arm in rep["arms"].items():
        for field, ulp in arm["max_ulp"].items():
            if name.endswith("crgamma") or field not in GAMMA_FIELDS:
                assert ulp == 0, (name, field, ulp)
            else:
                assert ulp <= GAMMA_MAX_ULP, (name, field, ulp)


def test_the_capture_reaches_every_arm_it_claims(host, oracle):
    """Deep, deep-falling-back-to-shallow, shallow, the shallow refusals and
    the quick exit are all reached, so 0 ULP is not a statement about one
    branch."""
    hp, lib, _ = host
    rep = hp.run(lib, oracle)
    paths = rep["arms"]["arm1-crgamma"]["paths"]
    reached = {int(k) for k in paths}
    assert {0, 1, 2}.issubset(reached), reached          # quick, deep, shallow
    assert any(p >= 16 for p in reached), reached          # deep -> shallow
    assert any(3 <= (p & 15) <= 9 for p in reached), reached   # a shallow exit


def test_a_planted_one_bit_constant_is_caught(host, oracle, tmp_path):
    """TREL 2400.0f -> 0x45160001 (one mantissa bit): every arm moves."""
    hp, _, _ = host
    lib = hp.build(tmp_path, "BMJ_TREL    2400.0f=BMJ_TREL    2400.000244140625f")
    rep = hp.run(lib, oracle)
    for name, arm in rep["arms"].items():
        assert max(arm["max_ulp"].values()) > 0, name
