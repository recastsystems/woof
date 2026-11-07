"""Reject corrupted exactness receipts."""
import numpy as np
from woof.verify.temf_oracle import compare, ulp_distance

def test_word_comparison_keeps_signed_zero_and_nan_payloads():
    a = np.array([0x00000000, 0x7fc00001], dtype=np.uint32).view(np.float32)
    b = np.array([0x80000000, 0x7fc00002], dtype=np.uint32).view(np.float32)
    assert ulp_distance(a, b).tolist() == [0, 0]
    assert compare({"f": a}, {"f": b})["f"]["mismatched"] == 2

def test_one_bit_difference_is_counted():
    a = np.array([0x3f800000], dtype=np.uint32).view(np.float32)
    b = np.array([0x3f800001], dtype=np.uint32).view(np.float32)
    result = compare({"f": a}, {"f": b})["f"]
    assert result["mismatched"] == 1
    assert result["max_ulp"] == 1
