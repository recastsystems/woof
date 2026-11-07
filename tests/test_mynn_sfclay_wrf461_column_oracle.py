"""CPU half of the WRF 4.6.1 MYNN surface-layer column oracle gate.

THE BREAKAGE THIS PREVENTS
--------------------------
Until development branch the CUDA MYNN surface layer
(sf_sfclay_physics=5) called CUDA's expf/logf/powf/atanf, fused the psi table
interpolation into one FMA and associated several products differently from
gfortran.  Against WRF 4.6.1's SFCLAY1D_mynn over the 110-column oracle's
eight spp_pbl=0 option sets that was 28,153 of 181,720 compared values wrong,
up to 423 ULP (HFX), under the strict build (32,040 under default arithmetic).  The fix is the unit's own libm header, a --fmad=false compile
and gfortran's association; tests/test_mynn_sfclay_wrf461_column_oracle_gpu.py
grades it at 0 ULP.  This module keeps the pieces that make that possible from
drifting without a device:

* the committed fixture is the one ``columns.py`` builds, byte for byte;
* ``mynn_libm.cuh`` is mynn_pbl.cu's libm block, character for character
  once comments are removed (the PBL lane proves those words bitwise);
* the surface unit compiles without FMA contraction and with the header, and
  its source calls no CUDA transcendental and no fused multiply-add.
"""

from __future__ import annotations

import gzip
import hashlib
import re
from pathlib import Path

from tools.mynn_sfclay_wrf461_column_oracle import columns as oracle_columns
from tools.mynn_sfclay_wrf461_column_oracle.compare import read_oracle

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "mynn" / "sfclay-columns-wrf461"
KERNELS = ROOT / "woof" / "core" / "kernels"

#: SHA-256 of the uncompressed oracle file build.sh wrote on test host
#: (gfortran 13.3.0, glibc 2.39, Xeon 8559C); also in PROVENANCE.txt.
ORACLE_SHA256 = (
    "c938d98dcd99430804f99b2f0cf00ba15c98e2b87e3f88adb8e69801ad8b027c")


def _strip_comments(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"\s*//.*$", "", line)
        if line.strip() or (out and out[-1].strip()):
            out.append(line)
    return "\n".join(out).strip() + "\n"


def _cut(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i:text.index(end, i) + len(end)]


def test_fixture_inputs_are_what_columns_py_builds():
    names, arrays = oracle_columns.build_columns()
    assert len(names) >= 64
    assert (FIXTURE / "columns.bin").read_bytes() == \
        oracle_columns.columns_bytes(arrays)
    assert (FIXTURE / "configs.txt").read_text(encoding="ascii") == \
        oracle_columns.configs_text()
    assert (FIXTURE / "column-names.txt").read_text(
        encoding="ascii").split() == names


def test_oracle_fixture_is_the_recorded_build_and_parses():
    blob = gzip.decompress((FIXTURE / "oracle-o2.bin.gz").read_bytes())
    assert hashlib.sha256(blob).hexdigest() == ORACLE_SHA256
    ncol, records = read_oracle(FIXTURE / "oracle-o2.bin.gz")
    assert ncol == len(oracle_columns.build_columns()[0])
    assert len(records) == sum(c[6] for c in oracle_columns.CONFIGS)
    regimes = set()
    for _, _, _, _, out in records:
        regimes.update(int(r) for r in out["regime"])
    assert regimes == {1, 2, 3, 4}


def test_surface_libm_header_is_the_mynn_pbl_block():
    pbl = (KERNELS / "mynn_pbl.cu").read_text(encoding="utf-8")
    header = (KERNELS / "mynn_libm.cuh").read_text(encoding="utf-8")
    want = (_strip_comments(_cut(
        pbl, "__device__ __forceinline__ real mynn_add(",
        "return mynn_gt(a, b) ? b : a;\n}\n"))
        + "\n" + _strip_comments(_cut(
            pbl, "__constant__ unsigned long long MYNN_LOGF_TAB[32]",
            "#undef MYNN_ATAN_T\n}\n")))
    assert _strip_comments(header) == want


def test_surface_unit_compiles_without_fma_and_with_its_libm():
    from woof.core.kernels import EXTRA_HEADERS, module_options, \
        module_source
    assert EXTRA_HEADERS["mynn_surface"] == ("mynn_libm.cuh", "surface_subnormal.cuh")
    assert module_options("mynn_surface") == ("-std=c++17", "--fmad=false", "--ftz=false")
    assert "mynn_glibc_powf" in module_source("mynn_surface")


def test_surface_unit_keeps_its_heavy_helpers_out_of_line():
    # Breakage: with these helpers __forceinline__, NVRTC took 33-40 s on the
    # wrf_461 form and 12-13 minutes on gsl_wrf39 (test host, 2026-10-07),
    # paid on every first run on a new machine and every cache miss; out of
    # line, both forms compile in about 1.5 s with the same words.
    code = (KERNELS / "mynn_surface.cu").read_text(encoding="utf-8")
    for name in ("mynn_psim_stable_full", "mynn_psih_stable_full",
                 "mynn_psim_unstable_full", "mynn_psih_unstable_full",
                 "mynn_table", "mynn_li_etal_2010", "mynn_zolrib",
                 "mynn_zolri2", "mynn_zolri"):
        assert re.search(
            r"__device__ __noinline__ real " + name + r"\(", code), name


def test_surface_unit_calls_no_cuda_transcendental_or_fma():
    code = _strip_comments(
        (KERNELS / "mynn_surface.cu").read_text(encoding="utf-8"))
    banned = re.findall(
        r"(?<![\w.])(expf|logf|powf|atanf|log10f|exp2f|__expf|__logf|"
        r"__powf|__fmaf_rn|fmaf|__fmaf_rz)\s*\(", code)
    assert banned == []
