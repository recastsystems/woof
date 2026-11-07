"""Kessler (mp_physics=1) against unmodified WRF v4.6.1, word for word.

The fixture (tests/data/oracles/kessler/, PROVENANCE.md) is SUBROUTINE
kessler of the byte-unmodified ``phys/module_mp_kessler.F`` built by
gfortran 13.3.0 -O0 (scalar glibc 2.39 libm, no FMA) and run over 104
columns x 50 levels in 13 regimes and five dt cases (25 calls).  The
production launcher ``woof.core.microphysics.launch_kessler`` must land on
every output word of every call:

* ``free``: each case starts from the inputs and the device integrates on
  its own, so one rounding difference would compound into later calls;
* ``replay``: each call starts from WRF's previous outputs.

What makes it exact (development branch), all default-on:

1. the six ``REAL**REAL`` powers and the one ``EXP`` call WOOF's own float32
   ``gfk_pow``/``gfk_exp`` (kernels/__init__.py ``_EXTRA_HEADERS``), not
   CUDA's ``powf``/``expf``;
2. the unit compiles with ``--fmad=false`` (``_NO_FMAD_MODULES``);
3. ``qr = (qr + prod - qr)`` is evaluated as written.  In float32 it is not
   ``prod``; the old transcription's shortcut cost up to 509,224 ULP on
   small qr words in replay.

Before the fix the strict build missed on every field (max 42 ULP qv, 1 ULP
theta, 7 ULP RAINNCV in replay).  After it, the strict build
(GPUWM_WRF_EXACT=1) is bitwise on all 525,200 words in both modes.  The
default build differs from strict only by CuPy's injected ``-ftz=true``; its
one measured residue is qr words of magnitude below 4e-36 that a flushed
subnormal intermediate moves by less than 2**-120 (112 of 130,000 qr words
free running, max |diff| 6.3e-38).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.verify import kessler_oracle as ko

#: Largest difference a default-arithmetic (FTZ) residue may have.  The
#: breakage this names: a flushed subnormal intermediate (every subnormal is
#: below 2**-126), nothing larger.
FTZ_RESIDUE_BOUND = 2.0 ** -120


def test_fixture_is_the_documented_oracle():
    inputs, wrf = ko.load_fixture()
    nz, ncol = inputs["t"].shape
    assert (nz, ncol) == (50, 104)
    assert len(set(map(str, inputs["regime"]))) == 13
    assert inputs["dt"].tolist() == [5.0, 18.0, 60.0, 120.0, 300.0]
    total = int(inputs["nsteps"].sum())
    assert total == 25
    for name in ko.LEVEL_FIELDS:
        assert wrf[name].shape == (total, nz, ncol)
        assert wrf[name].dtype == np.float32
        assert np.all(np.isfinite(wrf[name]))
    for name in ko.SURFACE_FIELDS:
        assert wrf[name].shape == (total, ncol)
    # the edge columns really carry the words they claim to
    qr = inputs["qr"]
    assert np.any((qr != 0) & (np.abs(qr) < np.finfo(np.float32).tiny))
    assert np.any(np.signbit(inputs["qc"]) & (inputs["qc"] == 0))
    assert np.count_nonzero(inputs["qc"] == np.float32(0.001)) > 0


def test_kernel_spells_the_wrf_arithmetic():
    """The three causes of the measured miss stay fixed in the source."""
    from woof.core import kernels
    src = (Path(kernels.__file__).parent / "kessler.cu").read_text(
        encoding="utf-8")
    code = "\n".join(line.split("//")[0] for line in src.splitlines())
    assert "powf(" not in code and "expf(" not in code
    assert code.count("gfk_pow(") == 6 and code.count("gfk_exp(") == 1
    assert "qrk = (qrk + prodk[k]) - qrk;" in code
    assert kernels.EXTRA_HEADERS["kessler"] == ("glibc_flt32.cuh",)
    assert kernels.module_options("kessler") == ("-std=c++17",
                                                 "--fmad=false")


def test_cpu_mirror_follows_the_file_expression():
    from woof.verify import npref
    import inspect
    assert "qr = (qr + prodk) - qr" in inspect.getsource(
        npref.np_kessler_column)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mode", ("free", "replay"))
def test_device_kessler_is_bitwise_wrf_on_every_call(mode):
    from woof import wrf_exact
    table = ko.measure(mode=mode)
    if wrf_exact.ENABLED:
        assert {k: v for k, v in table.items()
                if v["not_bit_identical"]} == {}, table
        return
    # Default arithmetic: identical to strict except where FTZ flushed a
    # subnormal intermediate.
    inputs, wrf = ko.load_fixture()
    got = ko.run_gpu(inputs, mode=mode, wrf=wrf)
    for name in ko.OUTPUT_FIELDS:
        g, w = got[name], wrf[name]
        miss = g.view(np.uint32) != w.view(np.uint32)
        if name != "qr":
            assert not miss.any(), (name, table[name])
            continue
        diff = np.abs(g[miss].astype(np.float64) - w[miss].astype(np.float64))
        assert np.all(diff < FTZ_RESIDUE_BOUND), table[name]
