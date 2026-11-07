"""Strict diagnostic comparisons must reject a numerically hidden change."""
import numpy as np
import pytest

from woof.verify.uwshcu_oracle import grade
from tools.cam5_wrf471_oracle.diagnose import compare


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_signed_zero_and_one_bit_are_differences(dtype):
    ref = np.array([0.0, 1.0], dtype=dtype)
    actual = ref.copy()
    actual[0] = -0.0
    word = np.uint32 if dtype == np.float32 else np.uint64
    actual.view(word)[1] ^= word(1)
    assert grade(actual, ref) == (2, 1, 1)
    assert compare({"x": actual}, {"x": ref})["differ"] == 2


@pytest.mark.parametrize("actual", [np.array([np.nan], dtype="f4"),
                                    np.array([np.inf], dtype="f4"),
                                    np.ones((1, 1), dtype="f4"),
                                    np.ones(1, dtype="f8")])
def test_invalid_output_cannot_pass(actual):
    expected = np.ones(1, dtype="f4")
    with pytest.raises(ValueError):
        grade(actual, expected)
    with pytest.raises(ValueError):
        compare({"x": actual}, {"x": expected})
