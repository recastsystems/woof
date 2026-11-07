"""Cold-start state and prescribed arithmetic for QNSE surface layer."""
import numpy as np
import pytest
from conftest import requires_gpu


@pytest.mark.parametrize("strict", [False, True])
def test_qnse_surface_math_follows_the_requested_mode(monkeypatch, strict):
    from woof.core import kernels
    import woof.wrf_exact as exact
    monkeypatch.setattr(exact, "ENABLED", strict)
    assert kernels.module_options("qnsesfc") == ((
        "-std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true",
        "--prec-sqrt=true", "-DGPUWM_QNSE_STRICT=1") if strict else ("-std=c++17", "--ftz=false"))


@requires_gpu
@pytest.mark.gpu
def test_qnsesfcinit_cold_start_matches_wrf_em_ustar():
    import cupy as cp
    from test_qnse_runtime import _build
    state, cfg, driver = _build(0)
    got = cp.asnumpy(driver.fields["ust"])
    expected = np.full(got.shape, np.float32(0.1))
    assert got.tobytes() == expected.tobytes()


def test_current_frame_reading_prices_both_qnse_modules():
    from woof.core.kernel_frame_recordings import QNSE_SM89_NVRTC_12_9
    from woof.core.preflight import KERNEL_MAX_LOCAL_SIZE_BYTES
    assert QNSE_SM89_NVRTC_12_9.frames["qnsepbl"] == KERNEL_MAX_LOCAL_SIZE_BYTES["qnsepbl"]
    assert QNSE_SM89_NVRTC_12_9.frames["qnsesfc"] == KERNEL_MAX_LOCAL_SIZE_BYTES["qnsesfc"]
