"""WOOF's Shin-Hong kernel against WRF v4.6.1 on 903 realistic columns, word for word.

The fixture in ``tests/data/oracles/shinhong_columns`` is the wide column
oracle (``tools/shinhong_column_oracle``): 129 base columns in 18 families
(convective land dry, moist and desert, the morning transition, stable night
land weak and radiative, arctic mixed-phase snow, ice fog, spring snow,
tropical trades, stratocumulus, cold-air outbreak and warm-advection fog over
the ocean, high terrain by day, by night and under snow, windy overcast
near-neutral, and 19 edge probes), each run at six grid spacings
(100 m to 27 km, dt from 1 s to 150 s) with the TKE diagnostic on, and once
more with it off.  The byte-unmodified WRF v4.6.1 ``module_bl_shinhong.F``
(sha256 99f44dbe...) wrote every output word, called the way
``module_pbl_driver.F`` calls it, on test host (gfortran 13.3.0, glibc 2.39).

``wrf-out.bin`` is THE reference: WRF's own phys/ flags (-O2
-funroll-loops) with the vectoriser off, so every libm call is scalar.  The
-O0 build (``wrf-out-O0.bin``) and WRF's exact shipping flags with the
vectoriser on (``wrf-out-O2vec.bin``) are pinned beside it.  What separates
-O0 from every optimised WRF build is that gfortran 13.3 at -O0 leaves the
17 ``x**2.`` sites as calls to the C library's powf(x, 2.0), which on that
host is not x*x for 769,747 normal float32 inputs (``pow2-receipt.txt``);
-O2 folds them to x*x, as WOOF spells them.

The defect this file was written to catch, and does: the kernel read the
model's theta directly as WRF's ``thx``.  WRF never sees theta: phy_prep
hands the scheme t = theta*pi (one float32 multiply) and the scheme forms
thx = t/pi (module_bl_shinhong.F:559), which is not theta on 2,016 of the
41,755 fixture levels.  Before the fix the strict-build kernel differed from
WRF on about 1,100 lanes of every tendency, 3,100 of exch_h and hpbl in 77
columns (up to 117 ULP); after it, on none.

Measured 2026-10-07 on W1 cards 1 and 3 (RTX PRO 6000 Blackwell Server Edition,
NVRTC 12.9.86, cupy 14.2.0), strict (GPUWM_WRF_EXACT=1) and default
arithmetic, one process each.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from woof.verify.shinhong_column_oracle import (
    FAMILIES, LEVEL_FIELDS, OOB_FIELDS, SHINHONG_COLUMNS_DIR, SURFACE_FIELDS,
    compare, load_columns, wrf_passthrough_violations)

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "shinhong_column_oracle"
KERNEL = ROOT / "woof" / "core" / "kernels" / "shinhong.cu"

#: The two edge-probe bases built for WRF's own prfac2 0/0
#: (module_bl_shinhong.F:1010: unstable br with ust**3 == wstar**3 == 0,
#: an input WRF's surface layers never produce), x 7 runs each.
NAN_PROBE_COLUMNS = tuple(range(812, 826))
#: kpbl == kte with tke_diag = 1: WRF reads q2xk(kte+1), one past the end
#: (:1557); its tke/el there are whatever the stack held, so excluded.
OOB_COLUMNS = tuple(range(840, 846))

#: NaN lanes WRF writes in the probe columns, per field, every build.
WRF_NAN_LANES = {"dtheta": 700, "exch_h": 259}
#: tke lanes in the probe columns where the scheme's amax1(q2xk, epsq2l)
#: (:1602) meets a NaN.  Fortran leaves MAX of a NaN processor dependent:
#: WRF's optimised builds (maxss) return epsq2l, so tke = 0.005, while -O0
#: and the WOOF kernel (NumPy semantics, kept so the NaN stays visible)
#: return NaN.  The engine repairs a NaN tke to 0.005
#: (repair_shinhong_passenger_outputs) and treats the NaN dtheta beside it
#: as fatal, so no run continues on either word.
TKE_MAX_NAN_LANES = 588

#: Default arithmetic only: CuPy compiles with -ftz=true.  Every differing
#: arithmetic word of the default build is a flush effect, below the
#: smallest normal float32 in absolute difference: (differing lanes, lanes
#: where WRF wrote a subnormal and the kernel a zero, worst ULP over the
#: rest, which are tiny results moved by a flushed intermediate).
DEFAULT_FLUSH = {"dv": (13, 11, 1), "dqc": (704, 687, 11),
                 "dqi": (705, 692, 21)}


@functools.cache
def _columns(ref: str = "wrf-out.bin"):
    return load_columns(SHINHONG_COLUMNS_DIR, outputs=ref)


def _gunzip(name: str) -> bytes:
    return gzip.decompress((SHINHONG_COLUMNS_DIR / f"{name}.gz").read_bytes())


# --------------------------------------------------------------------------
# CPU: the fixture is what it says it is.
# --------------------------------------------------------------------------

def test_fixture_bytes_are_the_ones_the_oracle_run_recorded():
    recorded = {}
    for line in (SHINHONG_COLUMNS_DIR / "oracle-sha256sums.txt").read_text(
            encoding="ascii").splitlines():
        digest, path = line.split(maxsplit=1)
        recorded[path.strip().rsplit("/", 1)[-1]] = digest
    assert recorded["module_bl_shinhong.F"] == (
        "99f44dbeb5e586b96be14424b8ab27c9986ffbd81f007f41fb8528d8ea466d56")
    for name in ("columns-in.bin", "wrf-out.bin", "wrf-out-O0.bin",
                 "wrf-out-O2vec.bin"):
        assert hashlib.sha256(_gunzip(name)).hexdigest() == recorded[name], name
    for name in ("make_columns.py", "run_shinhong_columns.F90"):
        assert hashlib.sha256((TOOL / name).read_bytes()).hexdigest() == (
            recorded[name]), f"{name} changed since the fixture was recorded"


def test_the_generator_reproduces_the_inputs_bit_for_bit(tmp_path):
    """The columns are a function of make_columns.py alone."""
    spec = importlib.util.spec_from_file_location(
        "shinhong_make_columns", TOOL / "make_columns.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    recs, _labels = mod.records()
    out = tmp_path / "columns-in.bin"
    mod.write_records(recs, out)
    assert out.read_bytes() == _gunzip("columns-in.bin")


def test_coverage_spans_the_regimes_and_the_edges():
    cols = _columns()
    assert len(cols) == 903
    assert len({c.base for c in cols}) == 129
    assert {c.family_name for c in cols} == set(FAMILIES.values())
    dx = {float(c.scalars["dx"]) for c in cols}
    assert dx == {100.0, 333.0, 1000.0, 3000.0, 9000.0, 27000.0}
    assert any(c.scalars["dy"] == 4 * c.scalars["dx"] for c in cols)
    assert {c.tke_diag for c in cols} == {0, 1}
    assert any(c.scalars["xland"] == 2.0 for c in cols)          # ocean
    assert min(float(c.scalars["psfc"]) for c in cols) < 60000.0  # terrain
    assert any(float(c.scalars["corf"]) < 0 for c in cols)       # south
    assert any(float(c.scalars["corf"]) == 0 for c in cols)      # equator
    assert any(c.levels["qi"].max() > 0 for c in cols)           # ice
    assert any(float(c.levels["t"][0]) < 250.0 for c in cols)    # cold snow
    hfx = [float(c.scalars["hfx"]) for c in cols]
    assert min(hfx) < -50.0 and max(hfx) > 600.0
    kpbl = [c.wrf_kpbl for c in cols]
    assert (min(kpbl), max(kpbl)) == (1, 56)
    assert sum(c.wrf_kpbl == c.nz for c in cols) == 7     # PBL fills column
    assert tuple(c.index for c in cols if any(
        np.isnan(v).any() for v in c.wrf.values())) == NAN_PROBE_COLUMNS
    assert tuple(c.index for c in cols if c.oob_tke) == OOB_COLUMNS


def test_wrf_hands_the_inout_surface_fields_back_unchanged():
    """ust, znt, wspd, u10, v10 and regime are INOUT in WRF's wrapper and
    come back word for word on arm A, which is why the kernel has no such
    outputs."""
    assert wrf_passthrough_violations(_columns()) == []


def test_the_theta_round_trip_is_real_on_this_fixture():
    """The premise of the thx fix, measured: (theta*pi)/pi is not theta."""
    cols = _columns()
    moved = total = 0
    for c in cols:
        th, pi = c.levels["theta"], c.levels["pi"]
        assert np.array_equal((th * pi).view(np.uint32),
                              c.levels["t"].view(np.uint32))
        thx = (c.levels["t"] / pi).astype(np.float32)
        moved += int((thx.view(np.uint32) != th.view(np.uint32)).sum())
        total += th.size
    assert (moved, total) == (2016, 41755)


def test_the_kernel_forms_thx_as_wrf_does():
    """CPU guard on the fix: thx = (theta*exner)/exner, IEEE division."""
    src = KERNEL.read_text(encoding="utf-8")
    assert "#define SH_TX(k)    (SH_G(theta, k) * SH_G(exner, k))" in src
    assert "#define SH_THX(k)   __fdiv_rn(SH_TX(k), SH_G(exner, k))" in src
    assert "#define SH_THX(k)   SH_G(theta, k)" not in src


def test_the_reference_builds_differ_only_where_named():
    """WRF against WRF: -O2 (vectoriser off), -O0 and the shipping -O2
    vectorised build, no port involved."""
    ref = _columns()
    o0 = _columns("wrf-out-O0.bin")
    vec = _columns("wrf-out-O2vec.bin")
    for other, expect in ((vec, {}), (o0, {"rublten": 12, "rvblten": 7,
                                           "exch_h": 14, "el": 2})):
        moved = {}
        for a, b in zip(ref, other):
            assert a.wrf_kpbl == b.wrf_kpbl
            for name in a.wrf:
                if a.oob_tke and name in OOB_FIELDS:
                    continue
                x, y = a.wrf[name], b.wrf[name]
                nan = np.isnan(x) | np.isnan(y)
                d = int(((x.view(np.uint32) != y.view(np.uint32)) & ~nan).sum())
                if d:
                    moved[name] = moved.get(name, 0) + d
        # tke in the 0/0 probe columns is NaN at -O0 and 0.005 optimised.
        if other is o0:
            assert moved.pop("tke") == 11
        assert moved == expect


# --------------------------------------------------------------------------
# GPU: the kernel against every word.
# --------------------------------------------------------------------------

from conftest import requires_gpu  # noqa: E402


@functools.cache
def _port():
    from woof.verify.shinhong_column_oracle import run_port

    return run_port(_columns())


def _strict() -> bool:
    from woof import wrf_exact

    return wrf_exact.ENABLED


@pytest.mark.gpu
@requires_gpu
def test_the_kernel_is_wrf_word_for_word_on_every_arithmetic_lane():
    import cupy  # noqa: F401

    cols = _columns()
    result = compare(cols, _port())
    assert result["kpbl_mismatch"] == []
    assert result["oob_excluded"] == 2 * len(OOB_COLUMNS)
    fields = result["fields"]
    assert set(fields) == set(LEVEL_FIELDS) | set(SURFACE_FIELDS)
    flush = {} if _strict() else DEFAULT_FLUSH
    tiny = float(np.finfo(np.float32).tiny)
    for name, st in fields.items():
        nan_pay = st.nan_payload_mismatch
        geo = st.nan_geometry_mismatch
        expect_geo = TKE_MAX_NAN_LANES if name == "tke" else 0
        assert geo == expect_geo, (name, st.as_dict())
        assert nan_pay == WRF_NAN_LANES.get(name, 0), (name, st.as_dict())
        arithmetic = st.differing - nan_pay - geo
        assert (arithmetic, st.ftz_lanes, st.max_ulp_outside_ftz) == (
            flush.get(name, (0, 0, 0))), (name, st.as_dict())
        assert st.max_abs < tiny, (name, st.as_dict())


@pytest.mark.gpu
@requires_gpu
def test_outside_the_two_nan_probes_every_word_is_wrfs():
    """The headline in its plainest form: on the 889 columns where WRF's
    arithmetic is defined, every output word of the strict build is WRF's
    word, signed zeros included."""
    import cupy  # noqa: F401

    if not _strict():
        pytest.skip("the strict build is the 0 ULP claim; the default build "
                    "is held to its flush lanes by the test above")
    cols = _columns()
    port = _port()
    nan_cols = set(NAN_PROBE_COLUMNS)
    checked = 0
    for c in cols:
        if c.index in nan_cols:
            continue
        got = port[c.index]
        for name, wname in {**LEVEL_FIELDS, **SURFACE_FIELDS}.items():
            if c.oob_tke and name in OOB_FIELDS:
                continue
            g = np.ascontiguousarray(got[name], np.float32).reshape(-1)
            w = c.wrf[wname].reshape(-1)
            assert np.array_equal(g.view(np.uint32), w.view(np.uint32)), (
                c.index, c.family_name, name)
            checked += g.size
        assert int(got["kpbl"].reshape(-1)[0]) == c.wrf_kpbl
    assert checked == 371_742


@pytest.mark.gpu
@requires_gpu
def test_a_one_ulp_theta_error_is_visible_to_this_gate():
    """Non-vacuity of the thx fix's gate: the old kernel's error was a
    1 ULP thx on 2,016 levels.  Move theta by 1 ULP on exactly those levels
    (so the kernel's thx becomes the old kernel's model theta there, but
    its tx stays WRF's t) and the bitwise comparison must see it."""
    import dataclasses

    import cupy  # noqa: F401

    from woof.verify.shinhong_column_oracle import run_port

    picked = []
    for c in _columns():
        if (c.tke_diag != 1 or float(c.scalars["dx"]) != 1000.0
                or c.index in NAN_PROBE_COLUMNS):
            continue
        th, pi = c.levels["theta"], c.levels["pi"]
        thx = (c.levels["t"] / pi).astype(np.float32)
        moved = thx.view(np.uint32) != th.view(np.uint32)
        if not moved.any():
            continue
        # theta' with (theta'*pi)/pi == theta on the moved levels does not
        # exist in general, so instead hold the kernel to WRF's own words:
        # its answer on theta must differ from its answer on theta nudged
        # by 1 ULP where WRF's thx moved.
        nudged = th.copy()
        nudged[moved] = np.nextafter(th[moved], np.float32(1e9))
        picked.append(dataclasses.replace(
            c, levels={**c.levels, "theta": nudged}))
    assert picked
    result = compare(picked, run_port(picked))
    assert result["fields"]["dtheta"].differing > 0
