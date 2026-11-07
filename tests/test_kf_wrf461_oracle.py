"""Kain-Fritsch (cu_physics=1) graded word for word against WRF v4.6.1.

The fixture under tests/data/oracles/kf is the output of the byte-unmodified
``phys/module_cu_kfeta.F`` (gfortran 13.3.0 -O0, glibc 2.39) on 150
hydrostatically consistent columns -- tropical ocean, midlatitude loaded gun,
stable night, trade-wind shallow, cold/snow, polar low, elevated convection
over a cold dome, high terrain, dry desert, tropical-cyclone core, very dry,
supersaturated layers, edge cases -- across grid length, time step, cumulus
period, vertical resolution and all five of KF_eta_CPS's feedback branches.
tools/kf_wrf461_oracle builds and runs it; its README has the recipe.

The breakage each gate prevents, named:

* ``test_the_gpu_kernel_matches_wrf_word_for_word`` -- any drift of
  kernels/kf.cu, of the W0AVG update, of the device table layout or of the
  compile route from WRF's arithmetic.  Before the 2026-10-07 port the
  kernel matched WRF on no convecting column at all.
* ``test_the_flushed_route_is_what_the_unflushed_compile_prevents`` -- the
  ``--ftz=false`` route.  CuPy's RawModule appends ``-ftz=true``; KF's
  condensate decays through the subnormal range, and the flushed build
  leaves the oracle.  This proves the gate above can fail and why the unit
  compiles where it does.
* the CPU gates pin the fixture's identity and coverage, so the GPU gate
  cannot pass on a shrunken or regenerated set.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "kf"
COLUMNS = FIXTURE / "columns-structured.bin"
META = FIXTURE / "columns-structured.json"
ORACLE = FIXTURE / "oracle-structured-O0.out"
#: (columns, per-column record, WRF output, column count) per graded set.
SETS = {
    "structured": (COLUMNS, META, ORACLE, 150),
    "real74": (FIXTURE / "columns-real74.bin",
               FIXTURE / "columns-real74.json",
               FIXTURE / "oracle-real74-O0.out", 80),
}

#: The fixture's bytes, as tools/kf_wrf461_oracle produced them on W1.
FIXTURE_SHA256 = {
    "columns-structured.bin":
        "076b6488b0ecb1dc515d970f0f07e68203433a06368d453d93dc92a19bed313d",
    "columns-structured.json":
        "9c0349fdbf0c6711af87b80c9f10be7280c935b27528100ee0bf580b867b0492",
    "oracle-structured-O0.out":
        "9746cc490192b3e381c09d2fbc86ca72b553cb64d4305d62acae4dfdc13e5fbf",
    "columns-real74.bin":
        "83d7cf4e54fcbb4e9161f4cc3aa0fec29f4b42665ddeea6c82d5cddf741bd00e",
    "columns-real74.json":
        "f6d4e3dbf9584e09501386549303614922fd3e54e142c8e15b76b5a16a2404c2",
    "oracle-real74-O0.out":
        "28315ac761f8b48899e05ad48ab95b9ca90f84a917accfb1025c0db7c50991a3",
}
#: The pinned WRF source the oracle refuses to build without.
KFETA_LF_SHA256 = (
    "e6376c2d85c45470f49d545b25d513b5ec111bf36b87beebc740bf42825c6e5f")


def _harness():
    from tools.kf_wrf461_oracle import compare_woof, kf_columns
    return kf_columns, compare_woof


def test_the_fixture_is_the_recorded_oracle_output():
    for name, digest in FIXTURE_SHA256.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() \
            == digest, name
    readme = (FIXTURE / "README.md").read_text(encoding="utf-8")
    for digest in FIXTURE_SHA256.values():
        assert digest in readme
    build = (ROOT / "tools" / "kf_wrf461_oracle" / "build.sh").read_text(
        encoding="utf-8")
    assert KFETA_LF_SHA256 in build
    assert KFETA_LF_SHA256 in readme


def test_the_fixture_covers_every_regime_branch_and_outcome():
    kf_columns, _ = _harness()
    cols = kf_columns.read_columns(COLUMNS)
    outs = kf_columns.read_outputs(ORACLE)
    meta = json.loads(META.read_text(encoding="utf-8"))
    assert len(cols) == len(outs) == len(meta) == 150
    assert {m["regime"] for m in meta} == set(kf_columns.REGIMES)
    assert {m["branch"] for m in meta} == set(kf_columns.BRANCHES)
    # Deep (0), shallow (1) and no convection (2), each well populated.
    shall = Counter(int(o["shall"]) for o in outs)
    assert shall == {0: 55, 1: 35, 2: 60}
    assert not any(o["fatal"] for o in outs)
    assert not any(o["unit98"] for o in outs)
    # The axes that move the scheme.
    assert {m["nz"] for m in meta} >= {8, 12, 35, 40, 45, 50, 60, 80, 128}
    assert {m["cudt"] for m in meta} == {0.0, 5.0, 10.0}
    assert any(float(2400.0 / m["dt"]).is_integer() is False
               and int(o["shall"]) == 1 for m, o in zip(meta, outs)), (
        "no shallow column at a DT that does not divide 2400 s: the TIMEC "
        "re-set at module_cu_kfeta.F:2572 would go ungraded")
    assert any(m["adapt"] for m in meta)
    # Every branch convects somewhere, or its feedback arm is ungraded.
    deep_by_branch = Counter(m["branch"] for m, o in zip(meta, outs)
                             if int(o["shall"]) < 2)
    assert set(deep_by_branch) == set(kf_columns.BRANCHES)


def test_the_ulp_metric_counts_what_it_claims():
    _, compare_woof = _harness()
    one = np.float32(1.0)
    up = np.nextafter(one, np.float32(2.0))
    assert compare_woof.ulp_distance([one], [up])[0] == 1
    assert compare_woof.ulp_distance([np.float32(0.0)],
                                     [np.float32(-0.0)])[0] == 0
    tiny = np.float32(1.0e-45)
    assert compare_woof.ulp_distance([tiny], [-tiny])[0] == 2
    assert compare_woof.ulp_distance([one], [one])[0] == 0


def test_the_real_model_columns_convect_deep_and_shallow():
    kf_columns, _ = _harness()
    columns, meta_path, oracle, count = SETS["real74"]
    outs = kf_columns.read_outputs(oracle)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(outs) == len(meta) == count
    outcome = Counter((m["regime"], int(o["shall"]))
                      for m, o in zip(meta, outs))
    assert outcome == {("real74_unstable", 0): 40,
                       ("real74_stable", 1): 30, ("real74_stable", 2): 10}
    assert {m["branch"] for m in meta} == set(kf_columns.BRANCHES)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", sorted(SETS))
def test_the_gpu_kernel_matches_wrf_word_for_word(name):
    columns, meta_path, oracle, count = SETS[name]
    _, compare_woof = _harness()
    report = compare_woof.run(str(columns), str(oracle), label=name,
                              meta_path=str(meta_path))
    assert report["graded"] == count
    assert report["mismatched_words"] == 0, json.dumps(
        {k: v for k, v in report["per_field"].items() if v["mismatched"]},
        indent=1)
    assert report["exact_columns"] == count
    assert report["max_ulp"] == 0
    # The grade is not vacuous: it read every tendency word of every column.
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert report["per_field"]["rthcuten"]["words"] == sum(
        m["nz"] for m in meta)


@pytest.mark.gpu
@requires_gpu
def test_the_flushed_route_is_what_the_unflushed_compile_prevents(
        monkeypatch):
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        pytest.skip("the strict build forces --ftz=false on every compile, "
                    "so the flushed control cannot be built under it")
    import cupy as cp
    from woof.core import kf
    from woof.core.kernels import module_source

    flushed = cp.RawModule(code=module_source("kf"),
                           options=("-std=c++17", "--fmad=false"))
    # _kf_function takes the trigger since c35dbafafb; the flushed control
    # is the default trigger-1 module only.
    def flushed_function(name, trigger=1):
        assert trigger == 1, trigger
        return flushed.get_function(name)
    monkeypatch.setattr(kf, "_kf_function", flushed_function)
    _, compare_woof = _harness()
    report = compare_woof.run(str(COLUMNS), str(ORACLE), label="flushed",
                              meta_path=str(META))
    assert kf.KF_MODULE_OPTIONS == ("-std=c++17", "--fmad=false",
                                    "--ftz=false")
    capability = int(cp.cuda.Device().compute_capability)
    if report["mismatched_words"] == 0 and capability < 100:
        # MEASURED 2026-10-12 on test host A (RTX 4090, sm_89, NVRTC 12.9): the
        # -ftz=true build matches WRF on all 71,968 words of this set and
        # all 38,112 of real74, while the same control misses on Blackwell
        # (sm_120: G1 today, and 41,127 of 1,848,256 sweep words on W1).  The
        # flush that breaks the port is the Blackwell targets'; on an older
        # target this control has nothing to show, and the unflushed route
        # above is graded there all the same.
        pytest.skip(f"sm_{capability}: the flushed build matches WRF on "
                    "this set, so the control is vacuous on this target")
    assert report["mismatched_words"] > 0, (
        "the -ftz=true build matched WRF on every word: either the fixture "
        "no longer reaches the subnormal range or the control is not "
        "flushing; either way the unflushed route lost its named reason")
