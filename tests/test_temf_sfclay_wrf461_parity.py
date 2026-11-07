"""The TEMF surface layer's CUDA kernel against WRF v4.6.1, word for word.

``woof.core.temf_sfclay.launch_temf_sfclay`` (the launcher
``PhysicsDriver._run_temf_sfclay`` calls) is driven over every step of every
column of the oracle's fixtures of record (tests/temf_sfclay_oracle.py,
78 columns x 8 steps), carrying HFX and WM_TEMF from its OWN previous
outputs and taking only the external feed (atmosphere, skin, and the TEMF
PBL's TE_TEMF/HD_TEMF) from the record.  Every float32 output word is
compared for equality.

Referees, and the two declared divergences:

* arm ``wrf-crlog10`` (WRF's object with a correctly rounded log10f): 0 ULP
  on every field, every column, every step;
* arm ``wrf`` (WRF's object on the C library): 0 ULP everywhere except the
  columns where the C library's log10f word is not the correctly rounded
  one -- and those columns are exactly the ones where the two WRF arms
  differ from each other (the log10-words regime);
* MAVAIL = 0 (dry-mavail0 regime): WRF writes 0/0 NaN into EXCH_TEMF and
  CHS; WOOF writes the limit, the exchange coefficient before the MAVAIL
  factor.  Asserted as such, not skipped.

Both arithmetic modes run this one kernel: every float32 operation is a
round-to-nearest intrinsic, so the default build and GPUWM_WRF_EXACT=1 are
graded against the same zero table.
"""

from __future__ import annotations

import numpy as np
import pytest

from temf_sfclay_oracle import dims, load, regime_columns

#: fixture input name -> kernel input name
INPUT_NAMES = {"u": "u1", "v": "v1", "th": "th1", "qv": "qv1", "p": "p1",
               "pi": "pi1", "rho": "rho1", "z": "z1", "ht": "ht",
               "psfc": "psfc", "znt": "znt", "mavail": "mavail",
               "tsk": "tsk", "fcor": "fcor", "hd": "hd_temf"}
#: fixture output name -> kernel output (or carried) name
OUTPUT_NAMES = {"ust": "ust", "hfx": "hfx", "qfx": "qfx", "lh": "lh",
                "flhc": "flhc", "flqc": "flqc", "chs": "chs", "chs2": "chs2",
                "cqs2": "cqs2", "cpm": "cpm", "qgh": "qgh", "qsfc": "qsfc",
                "u10": "u10", "v10": "v10", "th2": "th2", "t2": "t2",
                "q2": "q2", "exch": "exch_temf", "wm": "wm_temf", "te": "te1"}

#: Worst ULP distance per field against the correctly-rounded-log10 arm,
#: over every column and step outside the MAVAIL = 0 cells of EXCH/CHS.
#: Measured, asserted for equality; both arithmetic modes.
CRLOG10_MAX_ULP = {name: 0 for name in OUTPUT_NAMES}


def _launch_product(inputs, carried, outputs):
    from woof.core.temf_sfclay import launch_temf_sfclay
    launch_temf_sfclay(inputs, carried, outputs)


def replay(launch=_launch_product) -> dict[str, np.ndarray]:
    """Run every step, carrying HFX and WM_TEMF; return (nstep, ncol) words."""
    import cupy as cp

    from woof.core.physics_inventory import TEMF_SFCLAY_COLUMN_OUTPUTS

    fx = load("wrf")
    nstep, ncol = dims()
    got = {name: np.empty((nstep, ncol), np.float32) for name in OUTPUT_NAMES}
    got["znt"] = np.empty((nstep, ncol), np.float32)
    hfx = cp.asarray(fx["in:hfx"][0][None, :])
    wm = cp.asarray(fx["in:wm"][0][None, :])
    for s in range(nstep):
        inputs = {kernel: cp.asarray(np.ascontiguousarray(
            fx[f"in:{name}"][s][None, :])) for name, kernel in INPUT_NAMES.items()}
        carried = {"te1": cp.asarray(fx["in:te"][s][None, :]),
                   "hfx": hfx, "wm_temf": wm}
        outputs = {name: cp.full((1, ncol), np.nan, dtype=np.float32)
                   for name in TEMF_SFCLAY_COLUMN_OUTPUTS}
        launch(inputs, carried, outputs)
        pool = {**outputs, **carried}
        for name, kernel in OUTPUT_NAMES.items():
            got[name][s] = cp.asnumpy(pool[kernel])[0]
        got["znt"][s] = cp.asnumpy(inputs["znt"])[0]
    return got


def _ulp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from woof.core.fp32_ulp import fp32_ulp_distance
    return np.asarray(fp32_ulp_distance(a, b))


def _dry_cells():
    cols = regime_columns("dry-mavail0")
    nstep, ncol = dims()
    mask = np.zeros((nstep, ncol), dtype=bool)
    mask[:, cols] = True
    return mask


def _differing_columns(a: dict, b: dict, names) -> list[set[int]]:
    """Per step, the columns where any named field's word differs."""
    nstep, _ = dims()
    dry = _dry_cells()
    out = []
    for s in range(nstep):
        cols = set()
        for name in names:
            x = a[name][s].view(np.uint32)
            y = b[name][s].view(np.uint32)
            diff = x != y
            if name in ("exch", "chs"):
                diff &= ~dry[s]
            cols.update(np.flatnonzero(diff).tolist())
        out.append(cols)
    return out


@pytest.fixture(scope="module")
def got():
    return replay()


@pytest.mark.gpu
def test_every_word_equals_wrf_with_a_correctly_rounded_log10(got):
    import cupy  # noqa: F401

    want = load("wrf-crlog10")
    dry = _dry_cells()
    worst = {}
    for name in OUTPUT_NAMES:
        a, b = got[name], want[f"out:{name}"]
        keep = ~dry if name in ("exch", "chs") else np.ones_like(dry)
        bits_equal = a.view(np.uint32) == b.view(np.uint32)
        worst[name] = int(_ulp(a[keep], b[keep]).max())
        assert bits_equal[keep].all(), (
            f"{name}: {(~bits_equal[keep]).sum()} words differ, "
            f"worst {worst[name]} ULP")
    assert worst == CRLOG10_MAX_ULP
    # ZNT is an input of the scheme: it passes through untouched.
    assert np.array_equal(got["znt"].view(np.uint32),
                          want["out:znt"].view(np.uint32))


@pytest.mark.gpu
def test_against_stock_wrf_only_the_log10_columns_differ(got):
    """WOOF vs stock WRF differ exactly where stock WRF's two arms do."""
    import cupy  # noqa: F401

    stock, crlog = load("wrf"), load("wrf-crlog10")
    stock_out = {k[4:]: v for k, v in stock.items() if k.startswith("out:")}
    crlog_out = {k[4:]: v for k, v in crlog.items() if k.startswith("out:")}
    ours_vs_stock = _differing_columns(got, stock_out, OUTPUT_NAMES)
    wrf_vs_wrf = _differing_columns(crlog_out, stock_out, OUTPUT_NAMES)
    assert ours_vs_stock == wrf_vs_wrf
    flagged = set().union(*wrf_vs_wrf)
    assert flagged, "the fixture no longer exercises a log10 word"
    assert flagged <= set(regime_columns("log10-words").tolist())


@pytest.mark.gpu
def test_mavail_zero_is_the_limit_not_wrfs_nan(got):
    import cupy  # noqa: F401

    want = load("wrf-crlog10")
    cols = regime_columns("dry-mavail0")
    for name in ("exch", "chs"):
        assert np.isnan(want[f"out:{name}"][:, cols]).all(), (
            f"WRF no longer writes 0/0 into {name} at MAVAIL = 0")
        assert np.isfinite(got[name][:, cols]).all()
        assert (got[name][:, cols] > 0).all()
    # The limit: EXCH_TEMF is the moisture coefficient before the MAVAIL
    # factor, CHS that over RHO, as WRF's own formula reads at MAVAIL != 0.
    rho = load("wrf")["in:rho"][:, cols]
    assert np.array_equal(
        got["chs"][:, cols].view(np.uint32),
        (got["exch"][:, cols] / rho).astype(np.float32).view(np.uint32))
    assert (got["flqc"][:, cols] == 0).all()
    assert (got["qfx"][:, cols] == 0).all()


def _compile_unit(source: str):
    import cupy as cp

    from woof.core.kernels import module_options
    from woof.core.kernels import compile_noftz_module, MODULE_KEY_ROOT
    module = compile_noftz_module("temf_sfclay", source, f"{MODULE_KEY_ROOT}:temf_sfclay")
    return module.get_function("temf_sfclay_column")


def _launch_with(kernel):
    import cupy as cp

    from woof.core.physics_inventory import (TEMF_SFCLAY_CARRIED,
                                              TEMF_SFCLAY_INPUTS,
                                              TEMF_SFCLAY_COLUMN_OUTPUTS)

    def launch(inputs, carried, outputs):
        n = int(inputs["psfc"].size)
        args = ([inputs[k] for k in TEMF_SFCLAY_INPUTS]
                + [carried[k] for k in TEMF_SFCLAY_CARRIED]
                + [outputs[k] for k in TEMF_SFCLAY_COLUMN_OUTPUTS])
        kernel(((n + 127) // 128,), (128,), tuple(args) + (np.int32(n),))
        cp.cuda.runtime.deviceSynchronize()
    return launch


#: One-bit plants, each in one kernel constant: (anchor, replacement).
PLANTS = {
    "ftau0": ("const real ftau0 = TC_FTAU0;",
              "const real ftau0 = __uint_as_float(0x3E2E147Cu);"),
    "TEmin": ("const real TEmin = TC_TEMIN;",
              "const real TEmin = __uint_as_float(0x3A83126Fu ^ 1u);"),
    "Cf": ("const real Cf = TC_CF;",
           "const real Cf = __uint_as_float(0x3E3D70A4u ^ 1u);"),
}


@pytest.mark.gpu
@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_a_one_bit_plant_in_a_kernel_constant_is_caught(plant):
    import cupy  # noqa: F401

    from woof.core.kernels import module_source

    source = module_source("temf_sfclay")
    anchor, replacement = PLANTS[plant]
    assert source.count(anchor) == 1
    # negative control: the harness passes the unplanted source
    clean = replay(_launch_with(_compile_unit(source)))
    want = load("wrf-crlog10")
    dry = _dry_cells()

    def differing(words):
        count = 0
        for name in OUTPUT_NAMES:
            keep = ~dry if name in ("exch", "chs") else np.ones_like(dry)
            count += int((words[name].view(np.uint32)
                          != want[f"out:{name}"].view(np.uint32))[keep].sum())
        return count

    assert differing(clean) == 0
    planted = replay(_launch_with(_compile_unit(
        source.replace(anchor, replacement))))
    assert differing(planted) > 0, f"a one-bit change to {plant} went unseen"


# --------------------------------------------------------------------------
# WOOF's own binary32 log10 on the device
# --------------------------------------------------------------------------
_PROBE = r"""
extern "C" __global__ void woof_log10_probe(const unsigned int* in,
                                            float* out, const long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = woof_log10f(__uint_as_float(in[i]));
}
"""


def _device_log10(words: np.ndarray) -> np.ndarray:
    import cupy as cp

    from woof.core.kernels import module_source
    source = module_source("temf_sfclay")
    from woof.core.kernels import compile_noftz_module, MODULE_KEY_ROOT
    kernel = compile_noftz_module("temf_sfclay", source + _PROBE, f"{MODULE_KEY_ROOT}:temf_sfclay").get_function(
        "woof_log10_probe")
    d_in = cp.asarray(words)
    d_out = cp.empty(words.shape, dtype=np.float32)
    n = words.size
    kernel(((n + 255) // 256,), (256,), (d_in, d_out, np.int64(n)))
    return cp.asnumpy(d_out)


@pytest.mark.gpu
def test_the_device_log10_is_correctly_rounded():
    """Dense sweep: [2^-30, 2^30) every 5th word plus every 97th subnormal.

    The reference is binary64 log10 rounded once, with any word whose
    binary64 value lies within 8 binary64 ULPs of a binary32 rounding
    boundary set aside (it cannot settle such a word); the exhaustive
    proof over every positive word against MPFR is the oracle receipt
    log10-exhaustive.txt, which grades the C compilation of the same
    header.
    """
    import cupy  # noqa: F401

    lo = np.float32(2.0 ** -30).view(np.uint32)
    hi = np.float32(2.0 ** 30).view(np.uint32)
    words = np.concatenate([np.arange(lo, hi, 5, dtype=np.uint32),
                            np.arange(1, 0x00800000, 97, dtype=np.uint32)])
    undecided = 0
    for start in range(0, words.size, 1 << 24):
        chunk = words[start:start + (1 << 24)]
        got = _device_log10(chunk)
        y = np.log10(chunk.view(np.float32).astype(np.float64))
        f = y.astype(np.float32)
        toward = np.nextafter(f, np.where(y > f, np.inf, -np.inf).astype(
            np.float32))
        mid = (f.astype(np.float64) + toward.astype(np.float64)) / 2
        near = np.abs(y - mid) <= 8 * np.spacing(np.abs(y))
        undecided += int(near.sum())
        ok = got.view(np.uint32) == f.view(np.uint32)
        assert ok[~near].all(), chunk[~near & ~ok][:5]
    assert undecided < 64
    special = np.array([0x00000000, 0x80000000, 0x7F800000, 0x7FC00000,
                        0xBF800000, 0x3F800000], dtype=np.uint32)
    out = _device_log10(special)
    assert np.isneginf(out[0]) and np.isneginf(out[1])
    assert np.isposinf(out[2]) and np.isnan(out[3]) and np.isnan(out[4])
    assert out[5].view(np.uint32) == 0

@pytest.mark.gpu
def test_ieee_loader_keeps_a_subnormal_result():
    import cupy as cp
    from woof.core.kernels import compile_noftz_module, MODULE_KEY_ROOT
    source = r"""
    extern "C" __global__ void ieee_probe(const float* x, float* y) {
        y[0] = __fmul_rn(x[0], x[1]);
    }
    """
    kernel = compile_noftz_module("temf_sfclay", source, f"{MODULE_KEY_ROOT}:temf_sfclay").get_function("ieee_probe")
    x = cp.asarray(np.array([2.0**-70, 2.0**-70], np.float32))
    y = cp.zeros(1, np.float32)
    kernel((1,), (1,), (x,y))
    assert int(cp.asnumpy(y).view(np.uint32)[0]) == 512

@pytest.mark.gpu
def test_driver_uses_shared_pbl_storage_and_carries_surface_state():
    """Run the actual driver method, without replacing its kernel."""
    import cupy as cp
    from types import SimpleNamespace
    from woof.config import RunConfig
    from woof.core.physics import PhysicsDriver
    from woof.core.physics_inventory import TEMF_SFCLAY_COLUMN_OUTPUTS
    fx = load("wrf")
    nstep, ncol = dims()
    f = {name: cp.zeros((1,ncol), np.float32) for name in TEMF_SFCLAY_COLUMN_OUTPUTS}
    f["te_temf"] = cp.zeros((4,1,ncol), np.float32)
    f["te_temf"][1:] = np.float32(123.0)
    f["wm_temf"] = cp.asarray(fx["in:wm"][0][None,:])
    f["hfx"] = cp.asarray(fx["in:hfx"][0][None,:])
    f["mol"] = cp.full((1,ncol), np.float32(77.0))
    driver = object.__new__(PhysicsDriver)
    driver.fields = f
    driver.state = SimpleNamespace()
    f["temf_tskold"] = cp.zeros((1,ncol), np.float32)
    cfg = RunConfig(nx=8, ny=8, nz=4, dx=1000., dy=1000., ztop=4000., dt=1., run_seconds=8., sf_sfclay_physics=10, bl_pbl_physics=10)
    expected_wm = f["wm_temf"].copy()
    expected_hfx = f["hfx"].copy()
    wm_storage, te_storage = f["wm_temf"].data.ptr, f["te_temf"].data.ptr
    for s in range(nstep):
        level = {name: cp.asarray(fx[f"in:{name}"][s][None,None,:])
                 for name in ("u","v","th","qv","p","pi","rho","z")}
        f["te_temf"][0] = cp.asarray(fx["in:te"][s][None,:])
        for k, name in (("znt","znt"),("mavail","mavail"),("tsk","tsk"),("hd_temf","hd")):
            f[k] = cp.asarray(fx[f"in:{name}"][s][None,:])
        driver.state.ht = cp.asarray(fx["in:ht"][s][None,:])
        driver.state.f = cp.asarray(fx["in:fcor"][s][None,:])
        # Equal adjacent interfaces reproduce the recorded mass height exactly.
        atmosphere = {"u":level["u"], "v":level["v"], "theta":level["th"],
            "qv":level["qv"], "pressure":level["p"], "exner":level["pi"],
            "rho":level["rho"],
            "z_interface":cp.concatenate([level["z"],level["z"]]),
            "p_interface":cp.asarray(fx["in:psfc"][s][None,None,:])}
        driver.state.alt = cp.ascontiguousarray(
            (np.float32(1.0) + level["qv"]) / level["rho"])
        z, rho = driver._temf_columns(atmosphere)
        inputs = {kernel: cp.asarray(fx[f"in:{name}"][s][None,:])
                  for name, kernel in INPUT_NAMES.items()}
        inputs["z1"], inputs["rho1"] = z[0], rho[0]
        expected = {name: cp.zeros((1,ncol), np.float32)
                    for name in TEMF_SFCLAY_COLUMN_OUTPUTS}
        expected_carried = {"te1":f["te_temf"][0].copy(),
                            "hfx":expected_hfx, "wm_temf":expected_wm}
        _launch_product(inputs, expected_carried, expected)
        driver._run_temf_sfclay(atmosphere,cfg)
        expected_pool = {**expected, **expected_carried}

        for name,k in OUTPUT_NAMES.items():
            got = f["te_temf"][0] if k=="te1" else f[k]
            assert np.array_equal(cp.asnumpy(got)[0].view(np.uint32),
                                  cp.asnumpy(expected_pool[k])[0].view(np.uint32)), (s,name)
    assert f["wm_temf"].data.ptr == wm_storage
    assert f["te_temf"].data.ptr == te_storage
    assert bool(cp.all(f["te_temf"][1:]==np.float32(123.0)))
    assert bool(cp.all(f["mol"]==np.float32(77.0)))
