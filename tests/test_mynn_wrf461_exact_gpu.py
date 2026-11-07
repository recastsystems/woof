"""The CUDA MYNN boundary layer against unmodified WRF v4.6.1, word for word.

Six column families (convective day, stable night with dew, marine
stratocumulus, shallow cumulus, a sub-freezing cold pool with fog, ice and
frost, high wind) are integrated for twelve 20 s steps -- a cold start and
eleven warm steps -- by WRF v4.6.1's own ``mynn_bl_driver``
(tools/mynn_pbl_wrf461_oracle/run_driver_families.F90, gfortran 13.3.0 -O0
-ffp-contract=off, glibc 2.39), for both mixing-length options HRRR and WRF
use.  The device driver must land on every output word of every step:

* ``replay``: each step starts from the oracle's recorded inputs;
* ``free``: only step 1 does, and the device integrates the eleven warm
  steps from its own output, so a single rounding difference anywhere would
  compound into later steps.

What makes it exact (development branch, the YSU recipe): the unit compiles
with ``--fmad=false`` (woof/core/kernels/__init__.py ``_NO_FMAD_MODULES``)
and calls WOOF's own float32 exponential, logarithm, arctangent, hyperbolic
tangent and power routines, graded against the oracle host's elementary
function results by tools/mynn_pbl_wrf461_oracle/libm_sweep.py.

The CPU transcription (woof/core/mynn_pbl.py) is held to the same fixtures
here too, so the two ports and WRF are one function on these columns.
"""

from __future__ import annotations

import numpy as np
import pytest
import hashlib
import json
import os
from pathlib import Path
from test_mynn_wrf461_cycled_carry_gpu import mynn_extended_oracles

from conftest import requires_gpu

import _mynn_families as F

#: Measured max ULP per output over all steps, families and both modes;
#: a nonzero entry names its cause.  Ratchet only.
DEVICE_ULP: dict[int, dict[str, int]] = {
    1: {name: 0 for name in (*F.PROFILE_OUTPUTS, *F.COLUMN_OUTPUTS,
                             *F.INDEX_OUTPUTS)},
    2: {name: 0 for name in (*F.PROFILE_OUTPUTS, *F.COLUMN_OUTPUTS,
                             *F.INDEX_OUTPUTS)},
}


@pytest.mark.parametrize("mixlength", (1, 2))
@pytest.mark.parametrize("replay", (True, False), ids=("replay", "free"))
def test_cpu_driver_is_bitwise_wrf_on_every_family_and_step(mixlength, replay):
    from woof.core.mynn_pbl import mynn_bl_driver
    table = F.bit_mismatch_table(F.integrate(mynn_bl_driver, mixlength,
                                             replay=replay))
    assert {name: max(steps) for name, steps in table.items()
            if max(steps)} == {}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nsteps", (12, 120))
@pytest.mark.parametrize("mixlength", (1, 2))
@pytest.mark.parametrize("replay", (True, False), ids=("replay", "free"))
def test_device_driver_is_bitwise_wrf_on_every_family_and_step(mixlength,
                                                               replay, nsteps, mynn_extended_oracles):
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_bl_driver_cuda

    def driver(values, **kwargs):
        return mynn_bl_driver_cuda(
            {name: cp.asarray(np.ascontiguousarray(value))
             for name, value in values.items()}, **kwargs)

    steps = None if nsteps == 12 else F.load(
        mixlength, path=mynn_extended_oracles(f"driver-stock-120-{mixlength}.csv.gz"))
    results = F.integrate(driver, mixlength, replay=replay, to_host=cp.asnumpy,
                          steps=steps)
    table = F.bit_mismatch_table(results)
    if root := os.environ.get("MYNN_CARRY_RECEIPTS"):
        mode = "replay" if replay else "free"
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / f"cold-{nsteps}-{mixlength}-{mode}.json").write_text(
            json.dumps({"words": sum(value.size for _, _, want in results
                                     for value in want.values()),
                        "different_words": sum(sum(v) for v in table.values()),
                        "first_step_hashes": {
                            name: hashlib.sha256(value.tobytes()).hexdigest()
                            for name, value in results[0][1].items()
                            if name in results[0][2]}}, sort_keys=True))
    assert not any(max(steps) for steps in table.values()), (
        table, F.max_ulp_table(results))
