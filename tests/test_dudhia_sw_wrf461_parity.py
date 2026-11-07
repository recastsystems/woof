"""Dudhia shortwave (ra_sw_physics=1) against unmodified WRF v4.6.1, bitwise.

The fixture (tests/data/oracles/dudhia_sw/, PROVENANCE.md) is SUBROUTINE
SWRAD of the byte-unmodified ``phys/module_ra_sw.F`` built by gfortran
13.3.0 -O0 (scalar glibc 2.39 libm, no FMA), run over 82 columns x 50 levels
in 28 regimes (named regimes plus cosine-zenith branch words and other edge
columns) under four (SOLCON, swrad_scat, icloud) cases.  Graded outputs are
RTHRATEN (SWRAD's TTEN/pi3D into a zeroed tendency), GSW and the radiation
driver's SWDOWN = GSW/(1.-ALBEDO).

Before development branch the CuPy array-op transcription missed on every
field (strict build: 4,380 of 16,400 RTHRATEN words, max 233,266 ULP; GSW
116 of 328, max 683 ULP; SWDOWN 136 of 328, max 811 ULP).  What makes it
exact, all default-on:

1. ``REAL**0.635`` and ``ALOG10`` are the C library's functions in the WRF
   build: the kernel uses WOOF's own gfk_pow and a base-10 logarithm built
   on gfk_log, both equal to the C library on every float32 in [1, +inf]
   (tools/dudhia_sw_wrf461_oracle/sweep_libm.py);
2. the table interpolation is evaluated in SWPARA's own form
   (XI = frac + IIL first, weights XI-IIL, IU-XI, YJ-JJL, JU-YJ);
3. every operation is separately rounded, in float32 (COSZEN included: the
   adapter used to pass a float64 cosine into the recurrence);
4. CSSCA is swinit's float32 product and SWDOWN is the driver's
   GSW/(1.-ALBEDO), not SDOWN(kte+1).

After the fix both the strict build (GPUWM_WRF_EXACT=1) and default
arithmetic are bitwise on every word, through the column API and the
production adapter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.verify import dudhia_sw_oracle as do


def test_fixture_is_the_documented_oracle():
    inputs, wrf = do.load_fixture()
    nz, ncol = inputs["t"].shape
    assert (nz, ncol) == (50, 82)
    assert len(set(map(str, inputs["regime"]))) == 28
    assert inputs["solcon"].tolist() == pytest.approx(
        [1367.0, 1412.3, 1321.8, 1360.0])
    assert inputs["swrad_scat"].tolist() == pytest.approx(
        [1.0, 0.7, 1.0, 6.0])
    assert inputs["icloud"].tolist() == [1, 1, 0, 1]
    ncase = inputs["solcon"].size
    assert wrf["rthraten"].shape == (ncase, nz, ncol)
    for name in ("gsw", "swdown"):
        assert wrf[name].shape == (ncase, ncol)
    for name in do.OUTPUT_FIELDS:
        assert wrf[name].dtype == np.float32
        assert np.all(np.isfinite(wrf[name]))
    # Night columns (COSZEN <= 1e-9) return zeros; the first word above
    # 1e-9 is a day column.
    night = inputs["coszen"] <= np.float32(1.0e-9)
    assert night.sum() >= 10
    assert not np.any(wrf["rthraten"][:, :, night])
    assert not np.any(wrf["gsw"][:, night])
    assert np.all(wrf["gsw"][:, ~night] > 0)
    # The cosine-zenith branch words are the exact float32 thresholds.
    mu = set(inputs["coszen"].tolist())
    for word in (0.2, 0.5, 1.0):
        assert float(np.float32(word)) in mu
    assert np.any(inputs["qv"] < 0) and np.any(inputs["qc"] < 0)


def test_kernel_spells_the_wrf_arithmetic():
    """The causes of the measured miss stay fixed in the source."""
    from woof.core import kernels
    src = (Path(kernels.__file__).parent / "dudhia_sw.cu").read_text(
        encoding="utf-8")
    code = "\n".join(line.split("//")[0] for line in src.splitlines())
    for builtin in ("powf(", "log10f(", "logf(", "__powf", "__log10f"):
        assert builtin not in code.replace("dsw_log10f(", ""), builtin
    assert code.count("gfk_pow(") == 1
    assert code.count("dsw_log10f(") == 2      # definition and one call
    assert "FSUB(xi, (float)iil)" in code and "FSUB((float)iu, xi)" in code
    assert "swdown[col] = FDIV(g, FSUB(1.0f, alb));" in code
    assert kernels.EXTRA_HEADERS["dudhia_sw"] == ("glibc_flt32.cuh",)


def test_host_mirror_tracks_the_oracle():
    """The NumPy path follows the same expressions; NumPy's own power and
    log10 keep it from being bitwise, so it is held to a measured bound."""
    from woof.core.dudhia import dudhia_shortwave_columns
    inputs, wrf = do.load_fixture()
    names = ("t", "p", "qv", "qc", "qr", "qi", "qs", "qg", "dz8w")
    cols = [np.ascontiguousarray(inputs[n][::-1].T) for n in names]
    exner = np.ascontiguousarray(inputs["pii"][::-1].T)
    for c in range(inputs["solcon"].size):
        heat, swdown, gsw = dudhia_shortwave_columns(
            *cols, inputs["coszen"], inputs["albedo"], exner=exner,
            solcon=float(inputs["solcon"][c]),
            swrad_scat=float(inputs["swrad_scat"][c]),
            icloud=int(inputs["icloud"][c]))
        np.testing.assert_allclose(heat[:, ::-1].T, wrf["rthraten"][c],
                                   rtol=HOST_RTOL, atol=HOST_ATOL_HEAT)
        np.testing.assert_allclose(gsw, wrf["gsw"][c], rtol=HOST_RTOL,
                                   atol=HOST_ATOL_FLUX)
        np.testing.assert_allclose(swdown, wrf["swdown"][c], rtol=HOST_RTOL,
                                   atol=HOST_ATOL_FLUX)


#: Host-mirror bounds.  The breakage they name: the NumPy path drifting
#: away from SWPARA (a lost term, a wrong table word, a missing clamp),
#: which moves words by orders of magnitude more.  Measured on the fixture
#: (W1, 2026-10-07): NumPy's own float32 power/log10, amplified by the
#: TOTABS-OLDABS cancellation, leave up to 2.7e-10 K/s in RTHRATEN (relative
#: 1.5e-3 on the smallest heating words) and 4.9e-4 W m-2 in GSW/SWDOWN;
#: 60 to 76 percent of RTHRATEN words are bit-identical anyway.
HOST_RTOL = 5.0e-3
HOST_ATOL_HEAT = 1.0e-9
HOST_ATOL_FLUX = 2.0e-3


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("layout", ("columns", "adapter"))
def test_device_dudhia_is_bitwise_wrf(layout):
    table = do.measure(layout=layout)
    assert {k: v for k, v in table.items() if v["not_bit_identical"]} == {}, \
        table


@pytest.mark.gpu
@requires_gpu
def test_float64_coszen_is_rounded_to_wrf_real_first():
    """The adapter's solar geometry is float64; SWPARA's COSZEN is a REAL."""
    import cupy as cp
    from woof.core.dudhia import dudhia_shortwave_columns
    inputs, _ = do.load_fixture()
    names = ("t", "p", "qv", "qc", "qr", "qi", "qs", "qg", "dz8w")
    cols = [cp.asarray(np.ascontiguousarray(inputs[n][::-1].T))
            for n in names]
    mu64 = inputs["coszen"].astype(np.float64) * (1.0 + 2.0 ** -40)
    kw = dict(solcon=1367.0, swrad_scat=1.0, icloud=1)
    a = dudhia_shortwave_columns(*cols, cp.asarray(mu64),
                                 cp.asarray(inputs["albedo"]), **kw)
    b = dudhia_shortwave_columns(
        *cols, cp.asarray(mu64.astype(np.float32)),
        cp.asarray(inputs["albedo"]), **kw)
    for x, y in zip(a, b):
        assert x.dtype == cp.float32
        assert cp.asnumpy(x).tobytes() == cp.asnumpy(y).tobytes()
