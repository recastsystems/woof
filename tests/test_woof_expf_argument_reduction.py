"""WOOF's float32 expf (gfk_exp) fuses its argument reduction as the reference does.

The reference expf every WRF column oracle in this tree links is the C
library's on an x86-64 host with FMA, which is the FMA build of the
algorithm gfk_exp spells.  In that build the product InvLn2N*x feeds only
the add that forms kd and the subtract that forms r, so both become fused
multiply-adds and r keeps the product's rounding error.  gfk_exp rounded
the product separately until development branch, and over all 2^32 inputs
that differed from the host's expf on exactly two (MEASURED on test host, Xeon
Platinum 8559C, glibc 2.39, by tools/noah_wrf461_oracle/libm_exhaustive.py
and tools/noah_wrf461_oracle/expf_variant_probe.c):

    x = 0x4202422f (32.564632)   host 0x56fc9f1c, separately rounded 0x56fc9f1b
    x = 0xc27c65d9 (-63.099461)  host 0x11fa2993, separately rounded 0x11fa2992

The fused form equals the host on all 2^32.  This pins the two words on the
device, and computes the separately rounded form beside it so the test can
see the difference it guards (it fails if the two forms ever agree here).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

KERNELS = Path(__file__).resolve().parents[1] / "woof" / "core" / "kernels"

#: input word -> the host expf's word on W1 (the fused form's word).
HOST_WORDS = {0x4202422F: 0x56FC9F1C, 0xC27C65D9: 0x11FA2993}
#: input word -> the separately rounded form's word, one ULP below.
SEPARATE_WORDS = {0x4202422F: 0x56FC9F1B, 0xC27C65D9: 0x11FA2992}

_PROBE = r"""
// The separately rounded reduction gfk_exp used before, kept here only as
// the negative control.
__device__ float separate_exp(float x)
{
    double xd = (double)x;
    double z = DMUL(0x1.71547652b82fep+0 * 32.0, xd);
    return gfk_d2f_rn(gfk_exp2_core(
        z, GFK_EXP2F_SHIFT,
        GFK_EXP2F_P0 / 32.0 / 32.0 / 32.0,
        GFK_EXP2F_P1 / 32.0 / 32.0,
        GFK_EXP2F_P2 / 32.0, 0ULL));
}

extern "C" __global__ void expf_probe(const float* x, float* fused,
                                      float* separate, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fused[i] = gfk_exp(x[i]);
    separate[i] = separate_exp(x[i]);
}
"""


def test_gfk_exp_spells_the_reduction_as_fused_multiply_adds():
    source = (KERNELS / "glibc_flt32.cuh").read_text(encoding="utf-8")
    body = source[source.index("__device__ float gfk_exp(float x)"):]
    body = body[:body.index("\n}\n")]
    assert "__fma_rn(invln2n, xd, GFK_EXP2F_SHIFT)" in body
    assert "__fma_rn(invln2n, xd, -kd)" in body


@pytest.mark.gpu
@requires_gpu
def test_gfk_exp_gives_the_host_words_where_the_two_forms_differ():
    cp = pytest.importorskip("cupy")
    header = (KERNELS / "glibc_flt32.cuh").read_text(encoding="utf-8")
    module = cp.RawModule(code=header + _PROBE,
                          options=("-std=c++17", "--fmad=false"))
    probe = module.get_function("expf_probe")
    words = np.array(sorted(HOST_WORDS), dtype=np.uint32)
    x = cp.asarray(words.view(np.float32))
    fused = cp.empty_like(x)
    separate = cp.empty_like(x)
    probe((1,), (32,), (x, fused, separate, np.int32(x.size)))
    got = cp.asnumpy(fused).view(np.uint32)
    old = cp.asnumpy(separate).view(np.uint32)
    for i, word in enumerate(words):
        assert int(got[i]) == HOST_WORDS[int(word)], hex(int(word))
        assert int(old[i]) == SEPARATE_WORDS[int(word)], hex(int(word))
        assert got[i] != old[i]
