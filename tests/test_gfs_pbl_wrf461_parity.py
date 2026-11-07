"""The GFS PBL (bl_pbl_physics=3) against WRF v4.6.1's own words.

The fixtures under woof/data/gfs_pbl/oracle/ are the complete records of
tools/gfs_pbl_wrf461_oracle: the byte-unmodified module_bl_gfs.F built at
the scalar WRF referee's flags and driven over 72 columns per grid (six
regime families and six branch-point columns) on four vertical grids, four
calls each, with and without cloud ice.  Every call's input record and every
output word is in the record, so each implementation is replayed call by
call and graded word for word: the six tendencies, UST, WSPD, PBLH, KPBL.

* the NumPy reference (woof.verify.gfs_pbl_ref) -- CPU, on a glibc host,
  because the two real powers are the C library's own pow/powf there;
* the CUDA kernel, default and strict arithmetic, a planted one-bit
  constant and the chunked launch: tests/test_gfs_pbl_wrf461_gpu.py;
* the constants header must be exactly what the pinned WRF source yields.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tools" / "gfs_pbl_wrf461_oracle"
FIXTURES = ROOT / "woof" / "data" / "gfs_pbl" / "oracle"
sys.path.insert(0, str(ORACLE))

import compare  # noqa: E402
import oracle_io  # noqa: E402

FILES = ("cases-g16-ice60", "cases-g16-noice180", "cases-g40-ice60",
         "cases-g40-noice180", "cases-g61-ice60", "cases-g61-noice180",
         "cases-g16top-ice60", "cases-g16top-noice180")


def _manifest() -> dict:
    return json.loads((FIXTURES / "MANIFEST.json").read_text("utf-8"))


def _fixture_paths(tmp_path) -> list[Path]:
    out = []
    for name in FILES:
        raw = lzma.decompress((FIXTURES / f"{name}.out.xz").read_bytes())
        path = tmp_path / f"{name}.out"
        path.write_bytes(raw)
        out.append(path)
    return out


def _glibc() -> bool:
    return platform.system() == "Linux" and platform.libc_ver()[0] == "glibc"


def test_fixtures_are_the_recorded_oracle_words(tmp_path):
    """Every packaged record hashes to the oracle run of record."""
    manifest = _manifest()
    assert sorted(manifest["records"]) == sorted(f"{n}.out" for n in FILES)
    for path in _fixture_paths(tmp_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest["records"][path.name], path.name
        rec = oracle_io.read_oracle(path)
        assert rec["ncol"] == 72 and rec["nstep"] == 4


def test_the_constants_header_is_the_pinned_sources_words():
    """gfs_pbl_constants.cuh regenerates byte for byte from upstream/."""
    sys.path.insert(0, str(ORACLE))
    import gen_constants
    header = (ROOT / "woof" / "data" / "gfs_pbl"
              / "gfs_pbl_constants.cuh").read_text("utf-8")
    assert gen_constants.build(ORACLE / "upstream") == header


@pytest.mark.skipif(not _glibc(), reason="the two real powers are graded "
                    "against the C library the oracle called (glibc)")
def test_the_reference_reproduces_every_oracle_word(tmp_path):
    report = compare.grade(_fixture_paths(tmp_path), "ref")
    assert report["words"] == 468864
    assert report["differing_words"] == 0, report["fields"]


def test_substituted_coefficient_header_is_refused(tmp_path):
    from woof.table_assets import read_table_header
    root = ROOT / "woof/data/gfs_pbl"
    header = root / "gfs_pbl_constants.cuh"
    (tmp_path / "MANIFEST.json").write_bytes((root / "MANIFEST.json").read_bytes())
    data = header.read_bytes()
    current = tmp_path / header.name
    current.write_bytes(data)
    assert read_table_header(current) == data.decode("utf-8")
    changed = bytearray(data)
    changed[-2] ^= 1
    current.write_bytes(changed)
    with pytest.raises(ValueError, match="coefficient words outside the oracle"):
        read_table_header(current)


def test_near_lid_columns_really_reach_the_physical_model_top(tmp_path):
    for path in _fixture_paths(tmp_path):
        if "g16top" not in path.name:
            continue
        record = oracle_io.read_oracle(path)
        height = record["steps"][0]["outputs"]["pbl"][55:66]
        # The 3 km lid is explicit generator metadata. Nine deep columns
        # reach 94% of it; this tests actual oracle output, not a case label.
        assert int(np.count_nonzero(height > 0.9 * 3000.0)) == 9
        assert float(height.max()) < 3000.0
