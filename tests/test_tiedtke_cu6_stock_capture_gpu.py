"""cu6 against a real step of WRF 4.6.1 as its stock gfortran build runs it.

The fixture is one WRF tile row (166 columns, 44 levels) of WRF's own
cu_tiedtke wrapper inputs and outputs, captured at step 100 of the WRF-JAX
paper case (June 2021, 28N 75W, 3 km, 18 s step) from WRF 4.6.1 built with
its stock configure (option 34: gfortran 15.2 -O2 -ftree-vectorize, glibc
2.43, wrf.exe sha256 bda5e5e6...), 3 x 4 tiles.  It is the row of that
step whose scalar replay differs from WRF in the most words.

The breakage this prevents: that build evaluates tiecnv's conversion loop
(F:575-593) four columns at a time and binds F:588's exp to libmvec's
_ZGVbN4v_expf, which differs from the scalar expf on 3.6 per cent of its
arguments.  A kernel without the lane rule differs from that WRF in
804,133 of the step's 66,483,267 output words; with it, in none.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

cp = pytest.importorskip("cupy")

FIXTURE = (Path(__file__).resolve().parent / "data" / "oracles" / "cu6_wrf461_stock"
           / "row_s00100_j260_i334-499.npz")
OUT = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten")


def _replay(libmvec, fixture=FIXTURE, referee_glibc=(2,43)):
    from woof.core.tiedtke import INPUTS, column_step

    data = np.load(fixture)
    itimestep, stepcu, f_qc, f_qi = (int(v) for v in data["meta"][:4])
    inputs = {n: cp.asarray(data["in_" + n]) for n in INPUTS}
    outputs = {n: cp.asarray(data["wrf_in_" + n]) for n in OUT}
    outputs["raincv"] = cp.asarray(data["wrf_in_raincv"])
    outputs["pratec"] = cp.asarray(data["wrf_in_pratec"])
    outputs["cu_act_flag"] = cp.zeros(data["in_t3d"].shape[1], dtype=cp.int32)
    out = column_step(inputs, dt=float(data["dt"][0]), itimestep=itimestep,
                      stepcu=stepcu, f_qc=f_qc == 1, f_qi=f_qi == 1,
                      outputs=outputs, libmvec=libmvec, referee_glibc=referee_glibc)
    differ = {}
    for n in OUT + ("raincv", "pratec"):
        want = data["wrf_out_" + n].view(np.uint32)
        have = cp.asnumpy(out[n]).view(np.uint32)
        differ[n] = int(np.count_nonzero(want != have))
    return differ


@pytest.mark.gpu
@requires_gpu
def test_stock_build_lanes_reproduce_the_captured_wrf_row_word_for_word():
    differ = _replay(True)
    assert sum(differ.values()) == 0, differ


@pytest.mark.gpu
@requires_gpu
def test_the_scalar_referee_differs_on_this_row_so_the_fixture_tests_the_lanes():
    # Without the lane rule the row is not WRF's: the fixture can tell the
    # two builds apart.
    assert sum(_replay(False).values()) > 0


@pytest.mark.gpu
@requires_gpu
def test_strict239_tangent_reproduces_the_captured_wrf_row_word_for_word():
    fixture = (FIXTURE.parents[1] / "cu6_wrf461_strict239"
               / "row_s00100_j274_i334-499.npz")
    differ = _replay(False, fixture, (2,39))
    assert sum(differ.values()) == 0, differ
