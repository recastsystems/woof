"""RUC's float32 libm words against the C library WRF's Fortran links.

``tests/data/oracles/ruc_lsm/libm-words.json`` holds words the C library
(glibc 2.39, the toolchain of every RUC WRF oracle) returned on test host, for
arguments where its float32 word is NOT the float64 value rounded once --
exactly the arguments the RUC port used to get wrong when it rounded float64
evaluations (and, on the device, called CUDA's cosf/logf).  Written by
``tools/ruc_lsm_gpu_oracle/libm_sweep.py``, which also sweeps every float32
in each routine's RUC range on host and device.

The host mirrors in :mod:`woof.core.ruc` must reproduce every word; on a
card the device routines in ``ruc.cu`` / ``glibc_flt32.cuh`` /
``glibc_trig_flt32.cuh`` must too.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
WORDS = json.loads((ROOT / "tests" / "data" / "oracles" / "ruc_lsm"
                    / "libm-words.json").read_text())["words"]
UNARY = ("cosf", "tanhf", "expm1f", "expf", "log10f", "logf")


def _f(bits):
    return np.uint32(bits).view(np.float32)


def _host():
    from woof.core import ruc
    return {"cosf": ruc._f32_cos, "tanhf": ruc._f32_tanh,
            "expm1f": ruc._f32_expm1, "expf": ruc._f32_exp,
            "log10f": ruc._f32_log10, "logf": ruc._f32_log}


def test_the_table_discriminates():
    """Every unary routine carries words a rounded float64 would miss."""
    for name in UNARY:
        assert len(WORDS[name]) >= 50, name
    # The new-snow density argument (tabs = 270 K) the old TANH missed.
    probe = {x: y for x, y in WORDS["tanhf"]}
    x = int(np.float32(0.9974991083145142).view(np.uint32))
    assert probe[x] == int(np.float32(0.7605419158935547).view(np.uint32))


@pytest.mark.parametrize("name", UNARY)
def test_host_mirror_reproduces_every_word(name):
    fn = _host()[name]
    bad = [(hex(x), hex(y), hex(int(np.float32(fn(_f(x))).view(np.uint32))))
           for x, y in WORDS[name]
           if int(np.float32(fn(_f(x))).view(np.uint32)) != y]
    assert not bad, f"{name}: {len(bad)} words differ, e.g. {bad[:4]}"


def test_host_powf_reproduces_every_word():
    from woof.core import ruc
    bad = [(hex(a), hex(b), hex(c)) for a, b, c in WORDS["powf"]
           if int(ruc._f32_pow(_f(a), _f(b)).view(np.uint32)) != c]
    assert not bad, bad[:4]


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", UNARY + ("powf",))
def test_device_routine_reproduces_every_word(name):
    tool = ROOT / "tools" / "ruc_lsm_gpu_oracle"
    if str(tool) not in sys.path:
        sys.path.insert(0, str(tool))
    sweep = importlib.import_module("libm_sweep")
    kernel = sweep.device_kernel()
    rows = np.asarray(WORDS[name], dtype=np.uint64).astype(np.uint32)
    x = np.ascontiguousarray(rows[:, 0]).view(np.float32)
    e = (np.ascontiguousarray(rows[:, 1]).view(np.float32)
         if name == "powf" else None)
    want = rows[:, -1]
    got = sweep.call_device(kernel, sweep.WHICH[name], x, e).view(np.uint32)
    assert np.array_equal(got, want), (
        f"{name}: {int(np.count_nonzero(got != want))} of {want.size} differ")
