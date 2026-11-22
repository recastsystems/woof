"""WRF 4.8.0 Shin-Hong on the GPU against the native WRF 4.8.0 objects.

tests/data/oracles/shinhong480_columns holds tools/shinhong480_oracle's
1,161-column corpus (the wide-column regimes at six grid spacings, plus the
YSU counter-gradient, stratocumulus/dissipation and topo_wind arms) and the
outputs of the native WRF 4.8.0 strict objects (wrf.exe f6f47f86's
bl_shinhong.o and module_bl_shinhong.o, GNU 13.3 -O2 -fno-fast-math
-ffp-contract=off -fno-tree-vectorize):

* corpus.wrf-patched-out.bin.gz: the labelled-patched wrapper (TKE copy-in,
  tools/shinhong480_oracle/patch_tke_copyin.py), the referee of record;
* corpus.wrf-out.bin.gz: the pristine wrapper, whose TKE/EL are undefined
  words (it never copies TKE_PBL in) -- every other output is defined.

Under GPUWM_WRF_EXACT=1 the kernel reproduces both word for word (TKE/EL
against the patched referee only).  Default arithmetic is not graded here.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

DATA = (Path(__file__).resolve().parent / "data" / "oracles"
        / "shinhong480_columns")

pytestmark = pytest.mark.skipif(
    os.environ.get("GPUWM_WRF_EXACT") != "1",
    reason="graded under strict arithmetic (GPUWM_WRF_EXACT=1) only")


@pytest.fixture(scope="module")
def replay():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception:  # noqa: BLE001
        pytest.skip("no CUDA device")
    from tools.shinhong480_oracle.columns import read_v2, run_port
    recs = read_v2(DATA / "corpus.bin.gz")
    return recs, run_port(recs)


def test_every_word_matches_the_patched_native_referee(replay):
    from tools.shinhong480_oracle.columns import compare, read_wrf_out
    recs, port = replay
    rep = compare(recs, read_wrf_out(DATA / "corpus.wrf-patched-out.bin.gz",
                                     recs), port)
    assert rep["words"] == 543816
    assert rep["different"] == 0, {k: v for k, v in rep["fields"].items()
                                   if v["different"]}


def test_every_defined_word_matches_the_pristine_native_objects(replay):
    from tools.shinhong480_oracle.columns import compare, read_wrf_out
    recs, port = replay
    rep = compare(recs, read_wrf_out(DATA / "corpus.wrf-out.bin.gz", recs),
                  port)
    bad = {k for k, v in rep["fields"].items() if v["different"]}
    assert bad <= {"tke", "el"}, bad
