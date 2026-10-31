"""The NSSL route fills calc_prate_max's rolling rates as native 44c0849 does for every microphysics scheme (CPU)."""
from types import SimpleNamespace

import numpy as np

from woof.hex.mpasphys.rrfs import nssl_rrfs


def test_rolling_windows_follow_native_sub_limits():
    rate = nssl_rrfs._rolling_prate(SimpleNamespace(), np, 20.0, 5)
    assert list(rate["limits"]) == [4, 16, 31] and list(rate["times"]) == [60.0, 300.0, 600.0]
    assert rate["rolling"].shape == (31, 5)
    # a step coarser than one minute: native allocates no rolling array (sub_limit == 1)
    assert nssl_rrfs._rolling_prate(SimpleNamespace(), np, 720.0, 5)["rolling"] is None
    # 7 s does not divide 60/300/600: native rounds the window count and shortens the window time
    odd = nssl_rrfs._rolling_prate(SimpleNamespace(), np, 7.0, 5)
    assert list(odd["limits"]) == [10, 44, 87] and list(odd["times"]) == [63.0, 301.0, 602.0]


def test_driver_source_has_the_rate_kernel():
    assert 'extern "C" __global__ void nssl_prate(' in nssl_rrfs.RADAR
