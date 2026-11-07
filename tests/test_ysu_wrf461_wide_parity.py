"""YSU against WRF v4.6.1's own build, on 131 columns, word for word.

``tools/ysu_wrf461_oracle/build_wide.sh`` runs the byte-unmodified
``bl_ysu_run`` (phys/physics_mmm/bl_ysu.F90 at d66e442f) on the columns
``make_wide_columns.py`` writes: convective land, stable night land, ocean,
high terrain with and without topo_wind, snow and ice, stratocumulus under
top-down mixing, tropical moist, and 39 edge probes (signed zeros and
subnormals in br, ust, qv, qc, qi, u, v, u10, v10; zero surface coupling;
wspd and wscale**4 below the floors the kernel used to apply; the qc
threshold straddled; a PBL filling the column; nz from 4 to 120; dt from 1
to 300 s).  Each column is one call made with the argument set
module_bl_ysu.F passes in a WRF run.

Two builds of that Fortran are stored.  ``ysu-wide-wrf.bin`` is WRF's own
configure.defaults flags (-O2 -ftree-vectorize -funroll-loops), the
arithmetic a WRF run performs, and it is the reference.  ``-O0`` is kept
because the two are not the same function: GCC folds pow(x,2.0) to x*x at
-O1 and above, and powf is not correctly rounded at exact ties.

The standard here is the strict build (GPUWM_WRF_EXACT=1: no FMA, no FTZ,
IEEE division and square root): every word the kernel writes on every
column where WRF's own output is finite is WRF's word.  Under default
arithmetic -ftz=true flushes subnormals, and the only differences allowed
are cloud and ice tendency words within the subnormal range of WRF's
(absolute error below 1.2e-38 kg/kg/s; 80 words on this fixture).  Measured on
an RTX PRO 6000 (sm_120), NVRTC 12.9, together with a 20,000-column
randomized sweep (make_wide_columns.py --sweep 20000 7001) that is not
committed: strict 0 differing words of 899,350 per field; default only
dqc (9,900 words) and dqi (7,489), each within 1.2e-38 kg/kg/s of WRF's.

Five columns are outside the claim because WRF itself writes NaN there:
sfcflg is true and ust**3 underflows to zero or a subnormal, so
bl_ysu.F90:948's prfac2 = 15.9*w/ust3/(1+4*karman*w/ust3) is 0/0 or
Inf/Inf.  The kernel returns finite numbers on all five; that is a WRF bug,
documented and not copied.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify.ysu_wide_oracle import (
    LEVEL_FIELDS, REGIMES, SURFACE_FIELDS, TOPO_FIELDS, YSU_WIDE_DIR,
    load_wide_oracle)

#: Columns where WRF's own words are NaN: ust**3 below the smallest normal
#: float under an unstable surface (ust 0, the smallest subnormal, 0 with
#: heating, 1e-13), and the zero-coupling cloud-topped column with br < 0.
WRF_NONFINITE = (99, 100, 102, 103, 126)

#: Words where the -O0 build of the same Fortran differs from WRF's own
#: build on this fixture: two columns (54: exch at k 15; 113: du, dqv and
#: exch), all from pow(x,2.0) at an exact tie.  Pinned so the reference
#: choice stays visible.
STOCK_VS_O0_DIFFERING_WORDS = 6


def _strict() -> bool:
    from woof import wrf_exact
    return bool(wrf_exact.ENABLED)


def _subnormal(a: np.ndarray) -> np.ndarray:
    tiny = np.float32(np.finfo(np.float32).smallest_normal)
    return (a != 0) & (np.abs(a) < tiny)


# --------------------------------------------------------------------------
# CPU: the fixture is what it says it is
# --------------------------------------------------------------------------

def test_fixture_hashes_and_build_receipt():
    rows = (YSU_WIDE_DIR / "oracle-sha256sums.txt").read_text(
        encoding="ascii").splitlines()
    recorded = {line.split()[1].rsplit("/", 1)[-1]: line.split()[0]
                for line in rows}
    for name in ("ysu-wide-inputs.bin", "ysu-wide-wrf.bin",
                 "ysu-wide-wrf-O0.bin", "ysu-wide-inputs.json"):
        blob = (YSU_WIDE_DIR / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == recorded[name], name
    for source in ("bl_ysu.F90", "ccpp_kind_types.F", "run_bl_ysu_wide.F90",
                   "make_wide_columns.py"):
        assert source in recorded, source
    receipt = (YSU_WIDE_DIR / "build-receipt.txt").read_text(encoding="ascii")
    assert "d66e442fccc04111067e29274c9f9eaccc3cef28" in receipt
    assert "_ZGV" not in receipt, "a libmvec vector symbol entered a build"
    assert "FMA instructions in either object: 0" in receipt
    assert "GNU Fortran" in receipt and "GLIBC 2.39" in receipt


def test_fixture_spans_the_regimes_and_the_edges():
    cols = load_wide_oracle()
    assert len(cols) >= 64
    per = {name: 0 for name in REGIMES.values()}
    for c in cols:
        per[c.regime_name] += 1
    assert all(n >= 8 for n in per.values()), per
    assert len({c.nz for c in cols}) >= 6
    assert len({float(c.scalars["dt"]) for c in cols}) >= 6
    assert {c.topdown for c in cols} == {0, 1}
    assert sum(c.topo for c in cols) >= 8
    assert any(c.scalars["xland"] == 2 for c in cols)
    assert any(c.scalars["psfc"] < 65000 for c in cols)
    assert any(c.levels["theta"][0] < 255 for c in cols)
    assert any(np.any(c.levels["qi"] > 0) for c in cols)
    assert any(np.any(c.levels["rthraten"] < 0) for c in cols)
    sub = np.float32(np.finfo(np.float32).smallest_subnormal)
    br = np.asarray([c.scalars["br"] for c in cols])
    assert (br == sub).any() and (br == -sub).any()
    assert ((br == 0) & np.signbit(br)).any()
    ust = np.asarray([c.scalars["ust"] for c in cols])
    zero_coupled = [c.index for c in cols if c.scalars["ust"] == 0
                    and c.scalars["hfx"] == 0 and c.scalars["qfx"] == 0]
    assert len(zero_coupled) >= 3 and (ust == sub).any()
    assert min(float(c.scalars["wspd"]) for c in cols) < 1e-9
    # the zero-coupling stable column is inside the claim: WRF computes a
    # PBL there, which the kernel used to short-circuit to dz(1)
    stable_zero = [i for i in zero_coupled if cols[i].scalars["br"] > 0]
    assert stable_zero and all(cols[i].wrf_finite for i in stable_zero)
    assert all(cols[i].wrf_kpbl > 1 for i in stable_zero)


def test_wrf_nonfinite_columns_are_exactly_its_underflowed_ust_cubed_ones():
    cols = load_wide_oracle()
    assert tuple(c.index for c in cols if not c.wrf_finite) == WRF_NONFINITE
    tiny = np.float32(np.finfo(np.float32).smallest_normal)
    for i in WRF_NONFINITE:
        s = cols[i].scalars
        ust3 = np.float32(s["ust"]) ** np.float32(3)
        assert ust3 < tiny and s["br"] < 0, i


def test_wrfs_own_two_builds_differ_and_the_reference_is_wrfs_flags():
    """WRF against WRF, no port: the -O0 and WRF-flags builds of the same
    source write different words, so which one is the reference matters."""
    stock = load_wide_oracle(reference="stock")
    o0 = load_wide_oracle(reference="O0")
    differing = 0
    for a, b in zip(stock, o0):
        if not a.wrf_finite:
            continue
        for name in list(LEVEL_FIELDS.values()) + list(SURFACE_FIELDS.values()):
            differing += int(np.count_nonzero(
                a.wrf[name].view(np.uint32) != b.wrf[name].view(np.uint32)))
    assert differing == STOCK_VS_O0_DIFFERING_WORDS


# --------------------------------------------------------------------------
# GPU: the port, word for word
# --------------------------------------------------------------------------

def _port(cols):
    # The WRF v4.6.1 generation: from 2.9.0 the default ysu_version is
    # wrf_480, which this oracle does not grade.
    from woof.verify.ysu_wide_oracle import run_port_wide
    return run_port_wide(cols, ysu_version="wrf_461")


def _fields(c):
    out = dict(LEVEL_FIELDS)
    out.update(SURFACE_FIELDS)
    if c.topo:
        out.update(TOPO_FIELDS)
    return out


@pytest.mark.gpu
@requires_gpu
def test_ysu_is_wrf_word_for_word_on_every_finite_column():
    """Strict build: every word.  The default wrf_461 arm is 2.9.1's
    reproduction kernel (kernels/ysu_default.cu), graded by
    tests/test_ysu_wrf461_parity.py and tests/test_ysu_generation_batch_gpu.py,
    not by this oracle."""
    import cupy  # noqa: F401

    if not _strict():
        pytest.skip("graded under GPUWM_WRF_EXACT=1; the default wrf_461 arm "
                    "is graded by test_ysu_wrf461_parity")
    cols = load_wide_oracle()
    port = _port(cols)
    strict = _strict()
    flushed = 0
    for c in cols:
        if not c.wrf_finite:
            continue
        got = port[c.index]
        assert int(got["kpbl"].reshape(-1)[0]) == c.wrf_kpbl, c.index
        for name, wname in _fields(c).items():
            g = np.ascontiguousarray(got[name], np.float32).reshape(-1)
            w = np.ascontiguousarray(c.wrf[wname], np.float32).reshape(-1)
            diff = g.view(np.uint32) != w.view(np.uint32)
            if strict:
                assert not diff.any(), (c.index, c.regime_name, name,
                                        np.flatnonzero(diff) + 1)
            else:
                # -ftz: only the cloud and ice tendencies, and only inside
                # the subnormal range -- a subnormal WRF word written as
                # zero, or a tiny word (down to ~1e-34) computed through a
                # flushed subnormal intermediate of the implicit solve.
                tiny = np.float64(np.finfo(np.float32).smallest_normal)
                if diff.any():
                    assert name in ("dqc", "dqi"), (c.index, name)
                    err = np.abs(g[diff].astype(np.float64)
                                 - w[diff].astype(np.float64))
                    assert np.all(err < tiny), (c.index, name, err.max())
                flushed += int(diff.sum())
    if not strict:
        assert flushed > 0, "the fixture no longer reaches a flushed lane"


@pytest.mark.gpu
@requires_gpu
def test_ysu_differs_from_the_O0_build_only_where_wrfs_builds_do():
    import cupy  # noqa: F401

    if not _strict():
        pytest.skip("graded under GPUWM_WRF_EXACT=1, where the port is WRF")
    stock = load_wide_oracle(reference="stock")
    o0 = load_wide_oracle(reference="O0")
    port = _port(stock)
    for a, b in zip(stock, o0):
        if not a.wrf_finite:
            continue
        for name, wname in _fields(a).items():
            g = np.ascontiguousarray(port[a.index][name], np.float32)
            g = g.reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(
                g != b.wrf[wname].view(np.uint32),
                a.wrf[wname].view(np.uint32) != b.wrf[wname].view(np.uint32),
                err_msg=f"column {a.index} {name}")


@pytest.mark.gpu
@requires_gpu
def test_ysu_is_finite_where_wrf_divides_zero_by_zero():
    import cupy  # noqa: F401

    cols = load_wide_oracle()
    port = _port(cols)
    for i in WRF_NONFINITE:
        for name in LEVEL_FIELDS:
            assert np.all(np.isfinite(port[i][name])), (i, name)
        assert int(port[i]["kpbl"][0]) == cols[i].wrf_kpbl, i


@pytest.mark.gpu
@requires_gpu
def test_ysu_tanhf_is_the_reference_platforms_tanhf():
    """get_pblh's tanh, against stored words of the reference platform's
    tanhf (glibc 2.39, test host).  The first block is every argument of a W1
    sweep where that tanhf is not the correctly rounded tanh, which the
    kernel used before development branch; tools/ysu_wrf461_oracle/
    tanhf_sweep.py checks all 2,159,017,986 floats in [-0.5, 22]."""
    import cupy as cp

    from woof.core.kernels import _extra_header_text, _preamble, module_options

    if not _strict():
        pytest.skip("ysu_tanhf is the strict unit's (ysu_480_topo.cuh); a default "
                    "process composes ysu from ysu_topo_default.cuh")
    blob = (YSU_WIDE_DIR / "tanhf-ref.bin").read_bytes()
    nbad, nspread = np.frombuffer(blob, "<i4", 2)
    n = int(nbad + nspread)
    x = np.frombuffer(blob, "<f4", n, 8).astype(np.float32)
    want = np.frombuffer(blob, "<f4", n, 8 + 4 * n).astype(np.float32)
    assert nbad >= 4000
    correctly_rounded = np.tanh(x[:nbad].astype(np.float64)).astype(np.float32)
    assert np.count_nonzero(correctly_rounded.view(np.uint32)
                            != want[:nbad].view(np.uint32)) == nbad
    src = (_preamble() + _extra_header_text("ysu_480")
           + 'extern "C" __global__ void p(const float*x,float*o,int n)'
             '{int i=blockIdx.x*blockDim.x+threadIdx.x;'
             'if(i<n)o[i]=ysu_tanhf(x[i]);}')
    fn = cp.RawModule(code=src, options=module_options("ysu_480")).get_function("p")
    d = cp.asarray(x)
    o = cp.empty_like(d)
    fn(((n + 255) // 256,), (256,), (d, o, np.int32(n)))
    np.testing.assert_array_equal(cp.asnumpy(o).view(np.uint32),
                                  want.view(np.uint32))
