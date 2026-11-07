"""Morrison (mp_physics=10) against WRF v4.6.1 on weather-regime columns.

``tests/data/oracles/morrison_regimes/regime-columns.npz`` holds two sets
written by ``tools/morrison_wrf461_oracle/regime_columns.py`` and returned
by WRF's own ``MP_MORR_TWO_MOMENT`` (``run_morrison_regimes.F90``, the
byte-unmodified v4.6.1 module at gfortran -O0, glibc 2.39): 106 columns
each, twelve weather regimes x 8 seeded variants plus ten edge columns.

* ``m0_dt72_nz50``: graupel (morr_rimed_ice=0), dt 72 s, 50 levels, one
  call;
* ``m1_dt20_nz50_x30``: hail (morr_rimed_ice=1), dt 20 s, 50 levels, 30
  consecutive calls (each call's output is the next call's input).

The standard: built with ``MORR_ROW37_OFF`` (WRF's arithmetic alone) every
returned word is bitwise WRF's.  The production build keeps the
owner-ruled row-37 corrections, and on these columns only the
deposition-nucleation cap fires: in cold (T < 215 K) ice-supersaturated
cirrus and lower-stratosphere columns, where the cap keeps each nucleated
crystal at its seed mass and WRF's FUDGEF rescale shrinks the mass but not
the number.  Every other column, all ten other regimes included, is
bitwise in production too.  The wide freezing rates fire only where WRF's
own output is not finite (the stress sets, tools/morrison_wrf461_oracle/
row37_attribution.py); no regime column reaches them.

Every grade runs three ways: one call over all columns, the forecast
step's column calls (``work_chunk_columns``, ceil(ncol / NWORK) columns a
call, the chunking that keeps the WORK stack at one volume), and one
column a call.  WRF's columns are independent, so all three must return
WRF's words.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "oracles" / "morrison_regimes" / (
    "regime-columns.npz")
TOOLS = REPO / "tools" / "morrison_wrf461_oracle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

#: Columns whose production words may differ (the nucleation cap's
#: regimes), and how many of them differ per set, measured on the test host's RTX PRO
#: 6000 (strict and default arithmetic alike).
CAP_REGIMES = {"cirrus_anvil", "edge_stratosphere", "edge_supersat_homog"}
PRODUCTION_DIFFERING_COLUMNS = {"m0_dt72_nz50": 4, "m1_dt20_nz50_x30": 3}


def _load():
    import regime_columns as rc
    return rc, *rc.load_npz(FIXTURE)


def test_fixture_covers_every_regime_and_both_modes():
    rc, sets, labels = _load()
    assert [s[0] for s in sets] == list(PRODUCTION_DIFFERING_COLUMNS)
    assert set(labels) == set(rc.REGIMES) | set(rc.EDGES)
    assert len(labels) == 12 * 8 + 10
    assert {s[3] for s in sets} == {0, 1}           # graupel and hail
    assert {s[4] for s in sets} == {1, 30}          # one call and 30
    for name, ncol, nz, mode, steps, dt, fields, surf, want in sets:
        assert ncol == len(labels) and nz == 50
        for key in rc.LEVEL_OUT:
            assert np.isfinite(want[key]).all(), (name, key)


def _variant(row37_off: bool):
    import cupy as cp
    from woof.core import kernels
    original = kernels.load_module

    def load(name):
        if name != "morrison" or not row37_off:
            return original(name)
        module = cp.RawModule(
            code="#define MORR_ROW37_OFF 1" + chr(10)
            + kernels.module_source("morrison"),
            options=kernels.module_options("morrison"))
        module.compile()
        return module

    kernels.load_module = load
    kernels.get_kernel.cache_clear()
    return original


def _differing_columns(rc, got, want, ncol):
    bad = np.zeros(ncol, bool)
    for key in rc.LEVEL_OUT + rc.SURF:
        g = np.ascontiguousarray(got[key], np.float32)
        w = np.ascontiguousarray(want[key], np.float32)
        b = (g.view(np.uint32) != w.view(np.uint32)) & ~(
            np.isnan(g) & np.isnan(w))
        bad |= b.any(axis=0) if b.ndim == 2 else b
    return bad


CHUNKINGS = ("whole", "forecast", "single")


def _columns_per_call(chunking, ncol):
    from woof.core.morrison_constants import work_chunk_columns
    return {"whole": None, "forecast": work_chunk_columns(ncol),
            "single": 1}[chunking]


def _run(row37_off, chunking="whole"):
    from woof.core import kernels
    rc, sets, labels = _load()
    original = _variant(row37_off)
    try:
        out = {}
        for name, ncol, nz, mode, steps, dt, fields, surf, want in sets:
            got, _ = rc.run_gpu(ncol, nz, mode, dt, fields, surf, steps,
                                _columns_per_call(chunking, ncol))
            out[name] = (_differing_columns(rc, got, want, ncol), fields)
        return out, labels
    finally:
        kernels.load_module = original
        kernels.get_kernel.cache_clear()


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.skipif(not __import__("woof.wrf_exact", fromlist=["ENABLED"]).ENABLED,
                    reason="WRF word-for-word oracle of the strict (GPUWM_WRF_EXACT=1) path; a default process runs 2.9.1's arithmetic")
def test_wrf_arithmetic_is_bitwise_on_every_regime_column(chunking):
    out, _ = _run(row37_off=True, chunking=chunking)
    for name, (bad, _) in out.items():
        assert int(bad.sum()) == 0, name


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.skipif(not __import__("woof.wrf_exact", fromlist=["ENABLED"]).ENABLED,
                    reason="WRF word-for-word oracle of the strict (GPUWM_WRF_EXACT=1) path; a default process runs 2.9.1's arithmetic")
def test_production_differs_only_where_the_nucleation_cap_fires(chunking):
    out, labels = _run(row37_off=False, chunking=chunking)
    labels = np.asarray(labels)
    for name, (bad, fields) in out.items():
        assert int(bad.sum()) == PRODUCTION_DIFFERING_COLUMNS[name], name
        assert set(labels[bad].tolist()) <= CAP_REGIMES, name
        tmin = (fields["th"] * fields["pii"]).min(axis=0)
        assert float(tmin[bad].max()) < 215.0, name
