"""Device-free comparator and compiler-option checks for CAMMGMP."""
import numpy as np
import pytest
from tools.cammgmp_wrf461_oracle.compare_gpu import grade, OUT_NAME
from tools.cammgmp_wrf461_oracle.make_cases import OUT_2D, OUT_3D

def pair():
    fields = {n: np.zeros((2, 1) if n in OUT_3D else (1,), np.float32)
              for n in OUT_3D + OUT_2D}
    return [{"steps": [fields]}], [[{OUT_NAME.get(n, n): a.copy() for n, a in fields.items()}]]

def test_empty_gpu_result_cannot_pass():
    oracle, _ = pair()
    with pytest.raises(ValueError, match="group"):
        grade(oracle, [])

def test_missing_step_cannot_pass():
    oracle, _ = pair()
    with pytest.raises(ValueError, match="step"):
        grade(oracle, [[]])

def test_broadcast_shape_cannot_pass():
    oracle, gpu = pair()
    gpu[0][0]["th"] = np.zeros((1,), np.float32)
    with pytest.raises(ValueError, match="shape"):
        grade(oracle, gpu)

def test_nan_cannot_be_graded_as_weather():
    oracle, gpu = pair()
    oracle[0]["steps"][0]["th"][0, 0] = np.nan
    gpu[0][0]["th"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        grade(oracle, gpu)

def test_one_bit_output_change_is_caught():
    oracle, gpu = pair()
    gpu[0][0]["qv"].view(np.uint32)[0, 0] ^= np.uint32(1)
    rows = grade(oracle, gpu)
    assert rows["qv"]["differ"] == 1
    assert rows["qv"]["max_ulp"] == 1

def test_cam_uses_shared_no_ftz_options():
    from woof.core.kernels import module_options
    from woof.numerics import is_strict
    expected = (("-std=c++17", "--fmad=false", "--ftz=false") if is_strict()
                else ("-std=c++17", "--ftz=false", "-DGPUWM_CAM_FAST=1"))
    assert module_options("cammgmp") == expected
    assert module_options("morrison") == ("-std=c++17",)
    assert module_options("lake") == ("-std=c++17", "--fmad=false")
