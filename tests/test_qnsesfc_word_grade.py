"""The surface oracle must detect every altered float32 word."""
import numpy as np
import pytest

from woof.verify.qnsesfc_oracle import _stats


@pytest.mark.parametrize("words", [(0x00000000, 0x80000000),
                                   (0x7FC00001, 0x7FC00002),
                                   (0x3F800000, 0x3F800001)])
def test_surface_grade_detects_every_altered_word(words):
    got, want = (np.asarray([word], np.uint32).view(np.float32)
                 for word in words)
    assert _stats(got, want)["mismatch"] == 1


def test_surface_grade_accepts_identical_signed_zero_and_nan_payload():
    values = np.asarray([0x80000000, 0x7FC00001, 0x3F800000], np.uint32)
    assert _stats(values.view(np.float32), values.view(np.float32))["mismatch"] == 0
