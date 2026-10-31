"""Prepared work arrays given back at the physics-to-dycore hand-off: holders are told, then rebind.

THE BREAKAGE: rrfs/c3-8card-speed 826f1d6 (mv3km hrrrv5 1 card, 2026-10-29) released the prepared arrays every
physics call while the resident driver preparation kept writing the old ones; RUC read the new zeroed arrays and took
its pinned native fatal-error branch at step 2. 1960569 then gated the release on < 15 % free memory (+~6 GB per card
on conus3km 8 cards). The release is back on by default and every holder drops its views through on_prepared_release.
"""
from types import SimpleNamespace as NS

import numpy as np

from woof.hex.mpasphys.device_pools_v842 import NativeDevicePoolViews


def views(monkeypatch, mode=None):
    if mode is None:
        monkeypatch.delenv('WOOF_HEX_RELEASE_PREPARED', raising=False)
    else:
        monkeypatch.setenv('WOOF_HEX_RELEASE_PREPARED', mode)
    monkeypatch.delenv('WOOF_HEX_RELEASE_PREPARED_AUDIT', raising=False)
    v = NativeDevicePoolViews(NS(native=None), NS(zeros=np.zeros))
    v.resident = True
    host = np.zeros((4, 2, 1), np.float32)
    v.resident_buffers[('prepared', 'zero_p')] = (host, np.zeros((2, 4), np.float32), (1, 1, 1))
    v.resident_buffers[('prepared', 'kept_p')] = (host, np.ones((2, 4), np.float32), (1, 1, 1))
    v._prepared_nonzero.add(('prepared', 'kept_p'))
    return v


def test_release_is_default_and_tells_every_holder(monkeypatch):
    v = views(monkeypatch)
    calls = []
    v.on_prepared_release(lambda: calls.append(v.prepared_epoch))
    assert v.release_prepared() == 2 * 4 * 4
    assert calls == [1] and v.prepared_epoch == 1
    assert ('prepared', 'zero_p') not in v.resident_buffers
    assert ('prepared', 'kept_p') in v.resident_buffers
    again = v.prepared('zero_p')
    assert again.shape == (2, 4) and not again.any()
    assert v.resident_buffers[('prepared', 'zero_p')][1] is again


def test_never_keeps_arrays_and_holders(monkeypatch):
    v = views(monkeypatch, 'never')
    calls = []
    v.on_prepared_release(lambda: calls.append(1))
    assert v.release_prepared() == 0
    assert calls == [] and v.prepared_epoch == 0
    assert ('prepared', 'zero_p') in v.resident_buffers
