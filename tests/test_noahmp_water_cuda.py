"""Device acceptance gate for the Noah-MP WATER kernel.

Same bar as the CPU side: bitwise identity with the unmodified WRF v4.6.1
module over the committed oracle fixture.  Nothing is relaxed for the GPU --
the gate is ``max_ulp == 0`` and no tolerance is applied anywhere.

Four things this has to prove, not one:

1. ``k_water`` reproduces every ``output`` row of ``noahmp-water.csv`` bit for
   bit, on all 36 cases, including the whole snow/soil column.
2. It reproduces the fixture's ``probe`` stage too -- QSNSUB, QSNFRO, QSEVA,
   QSDEW and SNOFLOW never reach a WRF output but decide a branch, so a
   compensating pair of errors either side of SNOWWATER cannot pass on the
   composed answer alone.
3. ``glibc_expf`` and ``glibc_powf`` are bitwise right on the device over the
   sweeps in ``noahmp-water-libm.csv`` -- the *live glibc 2.39 symbols'* own
   output, since gfortran lowers ``REAL(4) EXP`` to ``expf@plt`` and ``**``
   with a real exponent to ``powf@plt``.
4. The gate can fail.  A deliberately perturbed constant must be rejected, or
   the three checks above are unfalsifiable.

And one thing specific to this lane: ``noahmp_water.cu`` carries *copies* of
``noahmp_soilwater.cu`` and ``noahmp_snow.cu``, because CuPy's RawModule
compiles from a string with no include path.
``test_imported_sections_match_their_sources`` re-derives both copies from
their source files by the documented transform and requires byte equality, so
a change to either lane that this file has not picked up fails a test rather
than silently forking a transcription that is already gated at max_ulp 0.
That test needs no GPU, so it now lives in
``tests/test_noahmp_kernel_source_scans.py`` -- while it was written HERE,
above this module's ``pytest.importorskip("cupy")``, it ran on no box at
all, which is the opposite of the "runs everywhere" this paragraph used to
claim for it.

Everything compared here comes from the oracle, not from the CPU
transcription, so a shared mistake in the two ports cannot pass this file.
"""

from __future__ import annotations

import csv
import os
import re
import struct
import sys
from collections import defaultdict

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_KDIR = os.path.join(_ROOT, "woof", "core", "kernels")
_DATA = os.path.join(_ROOT, "tests", "data", "oracles", "noahmp")
FIXTURE = os.path.join(_DATA, "noahmp-water.csv")
PROBE = os.path.join(_DATA, "noahmp-water-libm.csv")

NSOIL = 4
NSNOW = 3
NFULL = NSOIL + NSNOW

# Must match the layout documented in noahmp_water.cu.
NSTATE = 37
ST = {"snowh": 0, "sneqv": 1, "snice": 2, "snliq": 5, "stc": 8, "zsnso": 15,
      "dzsnso": 22, "sh2o": 29, "sice": 33}

W_P_STRIDE = 26
P_SLOT = {"smcmax": 0, "smcwlt": 4, "bexp": 8, "dksat": 12, "dwsat": 16}
P_SCALAR = {"kdt": 20, "frzx": 21, "slope": 22, "ch2op": 23, "ssi": 24,
            "snow_ret_fac": 25}

W_INT_STRIDE = 9
W_IN_STRIDE = NSTATE + 39
# SOLDRAIN is an additional production diagnostic after the 36 oracle
# slots. Allocate its word too, without adding or changing oracle values.
W_OUT_STRIDE = NSTATE + 37

IN = {"dt": 0, "fcev": 1, "fctr": 2, "elai": 3, "esai": 4, "fveg": 5,
      "bdfall": 6, "sfctmp": 7, "qvap": 8, "qdew": 9, "qsnow": 10,
      "qrain": 11, "snowhin": 12, "ponding": 13, "canliq": 14, "canice": 15,
      "tv": 16, "wslake": 17, "acc_qinsur": 18, "acc_qseva": 19}
IN_VEC = {"zsoil": 20, "btrani": 24, "acc_etrani": 28, "smc": 32,
          "ficeold": 36}

OUT_VEC = {"smc": 0, "acc_etrani": 10}
OUT = {"canliq": 4, "canice": 5, "tv": 6, "wslake": 7, "acc_qinsur": 8,
       "acc_qseva": 9, "cmc": 14, "ecan": 15, "etran": 16, "fwet": 17,
       "runsrf": 18, "runsub": 19, "qtldrn": 20, "ponding1": 21,
       "ponding2": 22, "qsnbot": 23, "qsnsub": 24, "qsnfro": 25,
       "qsubc": 26, "qfroc": 27, "qfrzc": 28, "qmeltc": 29, "qevac": 30,
       "qdewc": 31}
OUT_PROBE = {"qseva": 32, "qsdew": 33, "qinsur": 34, "snoflow": 35}


def _f32(hexbits: str) -> np.float32:
    return np.uint32(int(hexbits, 16)).view(np.float32)


def _bits(x) -> str:
    return struct.pack(">f", np.float32(x)).hex().upper()


def _load():
    with open(FIXTURE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "oracle fixture is empty"
    table = defaultdict(dict)
    cases = []
    for r in rows:
        key = (r["field"], int(r["index"]))
        v = int(r["value"]) if r["dtype"] == "int" else _f32(r["bits"])
        table[(r["case"], r["stage"])][key] = v
        if r["stage"] == "input" and r["case"] not in cases:
            cases.append(r["case"])
    return table, sorted(cases)


TABLE, CASES = _load()


# The checks that need no GPU -- the imported-section drift check and its
# falsifiability control -- are in
# tests/test_noahmp_kernel_source_scans.py.  This file's own docstring says
# of them "That test needs no GPU and runs everywhere"; it ran nowhere,
# because it sat ABOVE the `cp = pytest.importorskip("cupy")` below and a
# module-level skip fires at import, taking everything above it too.  The
# section transform is shared from tests/noahmp_kernel_sources.py, because
# the device gate below re-reads the same kernel text.
from noahmp_kernel_sources import read_kernel as _read  # noqa: E402

# ---------------------------------------------------------------------------
# Device gate
# ---------------------------------------------------------------------------

cp = pytest.importorskip("cupy")

from woof.core.kernels import get_kernel, load_module  # noqa: E402


def _pack():
    n = len(CASES)
    par = np.zeros((n, W_P_STRIDE), dtype=np.float32)
    fin = np.zeros((n, W_IN_STRIDE), dtype=np.float32)
    iin = np.zeros((n, W_INT_STRIDE), dtype=np.int32)
    for i, c in enumerate(CASES):
        p = TABLE[(c, "param")]
        for name, base in P_SLOT.items():
            for k in range(NSOIL):
                par[i, base + k] = p[(name, k + 1)]
        for name, slot in P_SCALAR.items():
            par[i, slot] = p[(name, 0)]

        x = TABLE[(c, "input")]
        fin[i, ST["snowh"]] = x[("snowh", 0)]
        fin[i, ST["sneqv"]] = x[("sneqv", 0)]
        for k in range(NSNOW):
            fin[i, ST["snice"] + k] = x[("snice", k - NSNOW + 1)]
            fin[i, ST["snliq"] + k] = x[("snliq", k - NSNOW + 1)]
        for k in range(NFULL):
            j = k - NSNOW + 1
            fin[i, ST["stc"] + k] = x[("stc", j)]
            fin[i, ST["zsnso"] + k] = x[("zsnso", j)]
            fin[i, ST["dzsnso"] + k] = x[("dzsnso", j)]
        for k in range(NSOIL):
            fin[i, ST["sh2o"] + k] = x[("sh2o", k + 1)]
            fin[i, ST["sice"] + k] = x[("sice", k + 1)]
        for name, slot in IN.items():
            fin[i, NSTATE + slot] = x[(name, 0)]
        for name, base in IN_VEC.items():
            lo = -NSNOW + 1 if name == "ficeold" else 1
            cnt = NSNOW if name == "ficeold" else NSOIL
            for k in range(cnt):
                fin[i, NSTATE + base + k] = x[(name, lo + k)]

        iin[i, 0] = x[("isnow", 0)]
        iin[i, 1] = p[("urban_flag", 0)]
        iin[i, 2] = p[("nroot", 0)]
        iin[i, 3] = x[("ist", 0)]
        iin[i, 4] = x[("frozen_canopy", 0)]
        iin[i, 5] = x[("frozen_ground", 0)]
        for k in range(NSNOW):
            iin[i, 6 + k] = x[("imelt", k - NSNOW + 1)]
    return par, fin, iin


def _run(func=None):
    par, fin, iin = _pack()
    n = len(CASES)
    d_out = cp.zeros((n, W_OUT_STRIDE), dtype=cp.float32)
    d_iout = cp.zeros(n, dtype=cp.int32)
    fn = func if func is not None else get_kernel("noahmp_water", "k_water")
    threads = 128
    blocks = (n + threads - 1) // threads
    fn((blocks,), (threads,),
       (cp.asarray(par), cp.asarray(fin), cp.asarray(iin), d_out, d_iout,
        np.int32(n)))
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(d_out), cp.asnumpy(d_iout)


def test_water_kernel_matches_oracle():
    out, iout = _run()
    bad = []
    for i, c in enumerate(CASES):
        want = TABLE[(c, "output")]
        if int(iout[i]) != int(want[("isnow", 0)]):
            bad.append(f"{c}/isnow gpu={int(iout[i])} "
                       f"oracle={int(want[('isnow', 0)])}")
        checks = []
        checks.append((ST["snowh"], "snowh", 0))
        checks.append((ST["sneqv"], "sneqv", 0))
        for k in range(NSNOW):
            checks.append((ST["snice"] + k, "snice", k - NSNOW + 1))
            checks.append((ST["snliq"] + k, "snliq", k - NSNOW + 1))
        for k in range(NFULL):
            j = k - NSNOW + 1
            checks.append((ST["stc"] + k, "stc", j))
            checks.append((ST["zsnso"] + k, "zsnso", j))
            checks.append((ST["dzsnso"] + k, "dzsnso", j))
        for k in range(NSOIL):
            checks.append((ST["sh2o"] + k, "sh2o", k + 1))
            checks.append((ST["sice"] + k, "sice", k + 1))
        for name, base in OUT_VEC.items():
            for k in range(NSOIL):
                checks.append((NSTATE + base + k, name, k + 1))
        for name, slot in OUT.items():
            checks.append((NSTATE + slot, name, 0))
        for slot, field, index in checks:
            got = np.float32(out[i, slot])
            if _bits(got) != _bits(want[(field, index)]):
                bad.append(f"{c}/{field}[{index}] gpu={_bits(got)} "
                           f"oracle={_bits(want[(field, index)])}")
    assert not bad, "; ".join(bad[:16]) + f"  ({len(bad)} total)"


def test_water_kernel_matches_the_probe_stage():
    """The locals that decide a branch but never reach a WRF output.

    QSEVA is compared after 6167's 0.001 scaling, which is the value the
    kernel returns; the fixture's probe is the pre-scaling one, so the
    comparison rescales it -- and on the FROZEN_GROUND branch 6148 has zeroed
    it, which is asserted rather than skipped.
    """
    out, _ = _run()
    milli = np.float32(0.001)
    bad = []
    for i, c in enumerate(CASES):
        pr = TABLE[(c, "probe")]
        x = TABLE[(c, "input")]
        for name in ("snoflow",):
            got = np.float32(out[i, NSTATE + OUT_PROBE[name]])
            if _bits(got) != _bits(pr[(name, 0)]):
                bad.append(f"{c}/{name} gpu={_bits(got)} "
                           f"oracle={_bits(pr[(name, 0)])}")
        frozen = bool(x[("frozen_ground", 0)])
        want_qseva = (np.float32(0.0) if frozen
                      else np.float32(np.float32(pr[("qseva", 0)]) * milli))
        want_qsdew = np.float32(0.0) if frozen else np.float32(pr[("qsdew", 0)])
        for name, want in (("qseva", want_qseva), ("qsdew", want_qsdew)):
            got = np.float32(out[i, NSTATE + OUT_PROBE[name]])
            if _bits(got) != _bits(want):
                bad.append(f"{c}/{name} gpu={_bits(got)} want={_bits(want)}")
    assert not bad, "; ".join(bad[:16])


def _probe_rows(prefix):
    with open(PROBE, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["name"].strip().startswith(prefix)]
    assert rows, f"no {prefix} rows in the probe fixture"
    return rows


def test_glibc_expf_on_device():
    """Both expf sweeps: SOILWATER's FCR range and COMPACT's far-negative one."""
    bad = []
    for prefix, lo, hi, n in (("expf_fcr_", -4.0, 0.0, 257),
                              ("expf_compact_", -40.0, 0.0, 257)):
        rows = {r["name"].strip(): r["bits"].upper()
                for r in _probe_rows(prefix)}
        assert len(rows) == n, (prefix, len(rows))
        x = np.empty(n, dtype=np.float32)
        want = []
        for i in range(n):
            frac = np.float32(np.float32(i) / np.float32(256.0))
            x[i] = (np.float32(lo) * np.float32(np.float32(1.0) - frac)
                    if prefix == "expf_fcr_"
                    else np.float32(np.float32(lo) * frac))
            want.append(rows[f"{prefix}{i}"])
        d_y = cp.zeros(n, dtype=cp.float32)
        get_kernel("noahmp_water", "k_water_expf")(
            ((n + 127) // 128,), (128,), (cp.asarray(x), d_y, np.int32(n)))
        cp.cuda.runtime.deviceSynchronize()
        y = cp.asnumpy(d_y)
        for i in range(n):
            if _bits(y[i]) != want[i]:
                bad.append(f"{prefix}{i} x={_bits(x[i])} gpu={_bits(y[i])} "
                           f"glibc={want[i]}")
    assert not bad, "; ".join(bad[:12]) + f"  ({len(bad)} total)"


def test_glibc_powf_on_device():
    """CANWATER's **0.667 and both WDFCND exponents."""
    bad = []
    specs = (("powf_fwet_", 0.667, 257, lambda i: np.float32(
                  np.float32(i) / np.float32(256.0))),
             ("powf_wdf_", 6.74, 129, lambda i: np.float32(
                  np.float32(0.01) + np.float32(
                      np.float32(np.float32(1.0) - np.float32(0.01))
                      * np.float32(np.float32(i) / np.float32(128.0))))),
             ("powf_wcnd_", 12.48, 129, lambda i: np.float32(
                  np.float32(0.01) + np.float32(
                      np.float32(np.float32(1.0) - np.float32(0.01))
                      * np.float32(np.float32(i) / np.float32(128.0))))))
    for prefix, expon, n, xf in specs:
        rows = {r["name"].strip(): r["bits"].upper()
                for r in _probe_rows(prefix)}
        assert len(rows) == n, (prefix, len(rows))
        x = np.asarray([xf(i) for i in range(n)], dtype=np.float32)
        e = np.full(n, np.float32(expon), dtype=np.float32)
        d_y = cp.zeros(n, dtype=cp.float32)
        get_kernel("noahmp_water", "k_water_powf")(
            ((n + 127) // 128,), (128,),
            (cp.asarray(x), cp.asarray(e), d_y, np.int32(n)))
        cp.cuda.runtime.deviceSynchronize()
        y = cp.asnumpy(d_y)
        for i in range(n):
            if _bits(y[i]) != rows[f"{prefix}{i}"]:
                bad.append(f"{prefix}{i} x={_bits(x[i])} gpu={_bits(y[i])} "
                           f"glibc={rows[f'{prefix}{i}']}")
    assert not bad, "; ".join(bad[:12]) + f"  ({len(bad)} total)"


def test_the_device_gate_can_fail():
    """Perturb one __constant__ entry and require the comparison to reject it.

    Without this every test above is unfalsifiable: a kernel that silently
    returned the host's own inputs would pass them.  The perturbation is
    WATER's own 0.001 -- the mm/s -> m/s conversion at 6159-6170, which is the
    one constant this file adds on top of the two imported sections -- nudged
    by a relative 1e-3, the same size the CPU mutation study uses.
    """
    src = _read("noahmp_water.cu")
    assert "0x3A83126Fu, /* 0.001" in src
    mutated = src.replace("0x3A83126Fu, /* 0.001", "0x3A85879Cu, /* 0.001", 1)
    assert mutated != src

    from woof.core.kernels import _preamble
    mod = cp.RawModule(code=_preamble() + mutated, options=("-std=c++17",))
    mod.compile()
    out, _ = _run(mod.get_function("k_water"))

    moved = 0
    for i, c in enumerate(CASES):
        want = TABLE[(c, "output")]
        if _bits(out[i, NSTATE + OUT["acc_qinsur"]]) != \
                _bits(want[("acc_qinsur", 0)]):
            moved += 1
    assert moved > 0, ("perturbing the mm/s -> m/s conversion did not move "
                       "ACC_QINSUR on any case; the device gate cannot fail")


def test_the_column_kernel_does_not_spill():
    """A whole-column kernel that spilled to local memory would be a defect.

    This replaces an assertion on ``cp.get_default_memory_pool().used_bytes()``
    that was flaky by construction and did not measure spilling at all.

    Flaky: ``used_bytes()`` is process-global and counts every live block in
    the pool, so any array another test in the same process still holds is
    included.  Measured on the reference RTX 5090: with a single unrelated
    16 MiB array alive, the value is 16,777,216 both before and after this
    kernel runs -- above the old 8 MiB bound -- so the test failed or passed
    on the basis of who ran before it.  The delta attributable to the kernel
    is 0, because the pool recycles the packing buffers.

    Wrong quantity: register spills and per-thread local arrays do not go to
    the memory pool.  They go to local memory, which CUDA reports per function
    as ``local_size_bytes``.  That is what this asserts.

    Measured, same box, ``-std=c++17``: ``local_size_bytes`` 224,
    ``num_regs`` 123, ``shared_size_bytes`` 0.  The bound is 2 KiB, which is
    nine times the measured frame and still catches a real spill: recompiling
    the identical source under ``-maxrregcount=32`` raises the frame to 608
    bytes and under ``-maxrregcount=24`` to 656, and a genuine whole-column
    spill in a routine with this many locals is thousands.  The
    ``test_the_spill_gate_can_fail`` companion pins that sensitivity instead
    of asserting it.
    """
    fn = get_kernel("noahmp_water", "k_water")
    attributes = fn.attributes
    assert attributes["local_size_bytes"] <= 2048, attributes
    assert attributes["shared_size_bytes"] == 0, attributes
    assert load_module("noahmp_water") is not None


def test_the_spill_gate_can_fail():
    """Force a spill and require the bound above to reject it.

    Compiling the same source with a 24-register cap makes the compiler spill;
    if the frame did not grow, the gate above would be measuring nothing.
    """
    from woof.core.kernels import _preamble

    source = _preamble() + _read("noahmp_water.cu")
    baseline = cp.RawModule(code=source, options=("-std=c++17",))
    baseline.compile()
    capped = cp.RawModule(
        code=source, options=("-std=c++17", "-maxrregcount=24"))
    capped.compile()
    small = baseline.get_function("k_water").attributes["local_size_bytes"]
    large = capped.get_function("k_water").attributes["local_size_bytes"]
    assert large > small, (small, large)


def test_the_runtime_batch_reproduces_the_fixture():
    """The gate on what the forecast actually calls.

    ``test_water_kernel_matches_oracle`` hand-builds the flat rows the kernel
    indexes.  The runtime does not: it packs a physical positional ``water()``
    call through :func:`woof.core.noahmp_water_gpu.evaluate_water_calls`, and
    the device result is written back into the caller's ``SnowColumn`` and a
    ``WaterFluxes``.  The nine in-place column arrays, the five parameter
    vectors, the ``ist``/``imelt``/``frozen_*`` integer row and the output
    field order are exercised only here, and all of it must be bitwise.
    """
    import test_noahmp_water as host

    from woof.core.noahmp_water_gpu import evaluate_water_calls

    cases = host.CASES
    calls = [(host._call(case), {}) for case in cases]
    fluxes = evaluate_water_calls(calls)
    assert len(fluxes) == len(cases)

    bad = []
    for case, (args, _kw), out in zip(cases, calls, fluxes):
        col = args[1]
        want = host.TABLE[(case, "output")]
        got = host._produced(col, out)
        for key, value in got.items():
            if key not in want:
                continue
            if key == ("isnow", 0):
                if int(value) != int(want[key]):
                    bad.append((case, key, int(value), int(want[key])))
                continue
            if host._bits(value) != host._bits(want[key]):
                bad.append((case, key, host._bits(value),
                            host._bits(want[key])))
    assert not bad, bad[:10]


#: Measured on the reference card, one ULP on each live entry argument of
#: fixture case ``wat_acc_nonzero``, counting the emitted columns that moved.
#: The first version of the test below nudged the entry ``SMC`` and asserted
#: something moved; nothing does.  That is not a defect in the batch -- WATER
#: rebuilds ``SMC(k) = SH2O(k) + SICE(k)`` at :6212 and SOILWATER works on
#: SH2O/SICE, so the entry SMC is dead on every path under this identity.
_ONE_ULP_REACH = {
    "zsoil": 5, "fctr": 3, "dt": 2, "fcev": 2, "canliq": 2, "acc_qinsur": 2,
    "tv": 1, "btrani": 1,
    "smc": 0, "acc_etrani": 0, "acc_qseva": 0, "elai": 0, "esai": 0,
    "fveg": 0, "bdfall": 0, "sfctmp": 0, "qvap": 0, "qrain": 0,
}


def test_the_runtime_water_batch_gate_can_fail():
    """Perturb one column's entry ZSOIL by one ULP; the gate must reject it.

    Which argument is a measurement, not a guess: see :data:`_ONE_ULP_REACH`.
    ZSOIL moves five emitted columns and is a per-layer vector, so it exercises
    the vector packing as well as the pairing.  The other columns must be
    untouched, which is the claim a batch has to make.
    """
    import struct

    import test_noahmp_water as host

    from woof.core.noahmp_water_gpu import CALL_NAMES, evaluate_water_calls

    cases = host.CASES
    calls = [(host._call(case), {}) for case in cases]
    args = list(calls[0][0])
    zsoil = np.array(args[CALL_NAMES.index("zsoil")], dtype=np.float32).copy()
    word = struct.unpack("<I", struct.pack("<f", float(zsoil[0])))[0]
    zsoil[0] = struct.unpack("<f", struct.pack("<I", word + 1))[0]
    args[CALL_NAMES.index("zsoil")] = zsoil
    calls[0] = (tuple(args), {})
    fluxes = evaluate_water_calls(calls)

    moved = 0
    for index, (case, (args, _kw), out) in enumerate(
            zip(cases, calls, fluxes)):
        want = host.TABLE[(case, "output")]
        got = host._produced(args[1], out)
        differing = [k for k, v in got.items()
                     if k in want and k != ("isnow", 0)
                     and host._bits(v) != host._bits(want[k])]
        if index == 0:
            moved = len(differing)
        else:
            assert not differing, (
                f"perturbing case 0 moved {case}: {differing[:5]}")
    assert moved >= 4, (
        f"a one-ULP entry-ZSOIL perturbation moved only {moved} columns; the "
        "batch round trip is not sensitive to its vector inputs")


def test_the_entry_smc_really_is_dead_in_water():
    """Record the measurement that rejected the first falsification attempt.

    WATER rebuilds ``SMC(k) = SH2O(k) + SICE(k)`` at :6212 and SOILWATER reads
    SH2O/SICE, so the entry SMC is not observable on any path under the pinned
    identity.  This is not a property to assume: if a future change makes it
    live, this says so rather than leaving a quietly weaker gate in place.
    """
    import struct

    import test_noahmp_water as host

    from woof.core.noahmp_water_gpu import CALL_NAMES, evaluate_water_calls

    cases = host.CASES
    base_calls = [(host._call(case), {}) for case in cases]
    base = evaluate_water_calls(base_calls)
    baseline = [host._produced(args[1], out)
                for (args, _kw), out in zip(base_calls, base)]

    calls = [(host._call(case), {}) for case in cases]
    args = list(calls[0][0])
    smc = np.array(args[CALL_NAMES.index("smc")], dtype=np.float32).copy()
    word = struct.unpack("<I", struct.pack("<f", float(smc[0])))[0]
    smc[0] = struct.unpack("<f", struct.pack("<I", word + 1))[0]
    args[CALL_NAMES.index("smc")] = smc
    calls[0] = (tuple(args), {})
    got = host._produced(calls[0][0][1], evaluate_water_calls(calls)[0])

    moved = [key for key, value in got.items()
             if host._bits(value) != host._bits(baseline[0][key])]
    assert not moved, (
        f"the entry SMC is now observable ({moved[:5]}); the falsification in "
        "this file may be re-pointed at it")
