"""CAMMGMP (mp_physics = 11) on the card against WRF v4.6.1's own objects.

The fixtures and their provenance are pinned without a device in
tests/test_cammgmp_wrf461_fixtures.py; this module runs the recovered column
launcher (woof.core.cammgmp.cammgmp_step) over the oracle deck as a
free-running chain and grades every output word of every column-step.

The unmodified WRF output is the stock diagnostic. The changed-ndrop
referee and kernel are attribution instruments only. These generated
soundings cannot satisfy a real-input module replay gate.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE = Path(os.environ.get("CAMMGMP_ORACLE_DIR")
              or ROOT / "tests/data/oracles/cammgmp")


def _need(name):
    path = ORACLE / name
    if not path.exists():
        pytest.skip(f"CAMMGMP oracle fixture {path} is absent")
    return path


# ---------------------------------------------------------------- device --

def _gpu():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover
        pytest.skip("no CUDA device")
    return cp


def _grade(referee, *, reproduce_defect=True):
    from tools.cammgmp_wrf461_oracle.compare_gpu import grade, run_group
    from tools.cammgmp_wrf461_oracle.oracle_io import read_deck, read_outputs
    deck = read_deck(_need("cases.bin"))
    oracle = read_outputs(_need(referee))
    gpu = [run_group(g, reproduce_defect=reproduce_defect) for g in deck]
    return grade(oracle, gpu), oracle, gpu


@pytest.mark.gpu
def test_support_functions_match_wrf_words():
    cp = _gpu()
    from woof.core.cammgmp import _module
    from tools.cammgmp_wrf461_oracle.oracle_io import read_helpers
    h = read_helpers(_need("helpers.bin"))
    kernel = _module().get_function("cammgmp_helpers")
    err = cp.zeros(1, dtype=cp.int32)

    def run(x, which):
        xd = cp.asarray(x)
        yd = cp.empty_like(xd)
        n = x.size
        kernel(((n + 127) // 128,), (128,), (xd, yd, np.int32(n), np.int32(which), err))
        return cp.asnumpy(yd)

    t = h["t"]
    assert np.array_equal(run(t, 0).view(np.uint64), h["estblf"].view(np.uint64))
    liq = run(t, 1)
    assert np.array_equal(liq.view(np.uint64), h["polysvp_liq"].view(np.uint64))
    assert np.array_equal(run(t, 2).view(np.uint64), h["polysvp_ice"].view(np.uint64))
    assert np.array_equal(run(h["erf_x"], 3).view(np.uint64), h["erf"].view(np.uint64))
    assert np.array_equal(run(h["gamma_x"], 4).view(np.uint64), h["gamma"].view(np.uint64))
    assert int(err.get()[0]) == 0


@pytest.mark.gpu
def test_columns_match_the_corrected_wrf_build_word_for_word():
    _gpu()
    rows, _, _ = _grade("fixed-out.bin", reproduce_defect=False)
    bad = {name: r for name, r in rows.items() if r["differ"]}
    assert not bad, bad


@pytest.mark.gpu
def test_columns_match_stock_wrf_word_for_word():
    _gpu()
    rows, _, _ = _grade("pristine-out.bin", reproduce_defect=True)
    bad = {name: r for name, r in rows.items() if r["differ"]}
    assert not bad, bad
    # and the correction is not a no-op on this deck
    rows_fixed, _, _ = _grade("pristine-out.bin", reproduce_defect=False)
    assert sum(r["differ"] for r in rows_fixed.values()) > 0


@pytest.mark.gpu
def test_a_planted_one_bit_change_is_caught():
    """One flipped bit in a kernel constant (MG's rhosu) fails the grade."""
    cp = _gpu()
    import woof.core.cammgmp as cam
    from woof.core.kernels import module_source
    from tools.cammgmp_wrf461_oracle.gen_words import read_words
    words = read_words(_need("words.txt"))
    good = words["cldwat2m_micro.rhosu"].hex()
    bits = np.float64(words["cldwat2m_micro.rhosu"]).view(np.uint64) ^ np.uint64(1)
    planted = float(bits.view(np.float64)).hex()
    from woof.core.kernels import module_source_int_defines
    src = module_source_int_defines("cammgmp", (("CAMMGMP_CORRECTED_NDROP_DIAGNOSTIC", 1),))
    assert src.count(f"#define MG_RHOSU {good}") == 1
    src = src.replace(f"#define MG_RHOSU {good}", f"#define MG_RHOSU {planted}")
    from woof.core.kernels import compile_cam_module_no_ftz
    mod = compile_cam_module_no_ftz("cammgmp", src, suffix="[planted-bit]")
    cam._upload_estbl(mod)
    original = cam._module_corrected_ndrop
    try:
        cam._module_corrected_ndrop = lambda: mod
        rows, _, _ = _grade("fixed-out.bin", reproduce_defect=False)
    finally:
        cam._module_corrected_ndrop = original
    assert sum(r["differ"] for r in rows.values()) > 0
