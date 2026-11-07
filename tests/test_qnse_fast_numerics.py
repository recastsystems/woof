"""Production QNSE stays finite and close to the native strict fixture.

These error bounds are production regression checks, not bitwise proof.
The strict replay tests remain responsible for every output word.
"""
import os
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [requires_gpu, pytest.mark.gpu,
              pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT") == "1",
                                 reason="production arithmetic check")]


@pytest.mark.parametrize("arm", ["mf0", "mf1"])
def test_fast_pbl_four_call_error_against_native(arm):
    import test_qnse_wrf461_parity as oracle
    fx = oracle.load(arm)
    carried = oracle._state_from_oracle(fx, 1)
    mapping = dict(oracle.QNSE_LEVEL_OUT)
    mapping.update({name: name for name in (*oracle.STATE_OUT, "pblh", "kpbl")})
    if int(fx["meta_mfshconv"]):
        mapping.update({name: name for name in oracle.MF_OUT})
    for step in (1, 2, 3, 4):
        got = oracle._run(fx, step, carried)
        for name, native_name in mapping.items():
            actual = np.asarray(got[name], np.float64)
            native = np.asarray(fx[f"s{step}/{native_name}"], np.float64)
            assert np.isfinite(actual).all(), (arm, step, name)
            scale = np.linalg.norm(native.ravel())
            error = np.linalg.norm((actual - native).ravel())
            # 0.20% was measured on the recovered fixture on RTX4090.
            # A 1% relative L2 limit permits FP32 cancellation while still
            # rejecting wrong constants, missing mixing or wrong dispatch.
            assert error <= .01 * scale if scale else error == 0, (arm, step, name, error, scale)
        carried = oracle._carry(got)
        if carried["rc_mf"] is None:
            carried["rc_mf"] = np.zeros_like(carried["tke"])


def test_fast_surface_primary_outputs_against_native():
    from woof.verify import qnsesfc_oracle
    report = qnsesfc_oracle.measure(qnsesfc_oracle.load())
    for mode in ("replay", "free_run"):
        for name in ("t2", "th2", "thz0"):
            assert report[mode][name]["max_abs"] < .001, (mode, name)
        for name in ("hfx", "lh"):
            assert report[mode][name]["max_abs"] < .01, (mode, name)
