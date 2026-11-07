"""CPU stand-in tests for table ownership and upload ordering, not physics."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from woof.core import device_cache, pxlsm


@pytest.fixture
def cuda_standin(monkeypatch):
    state = SimpleNamespace(device=0, stream=1, uploads=[], events=[], waits=[])

    class Device:
        def __init__(self):
            self.id = state.device

        def synchronize(self):
            raise AssertionError("table lookup must not synchronize the host")

    class Event:
        def __init__(self, **kwargs):
            self.done = False
            state.events.append(self)

        def record(self):
            self.source = state.stream

    def current_stream():
        return SimpleNamespace(ptr=state.stream, wait_event=state.waits.append)

    def upload(host):
        array = host.copy()
        state.uploads.append(array)
        return array

    fake = SimpleNamespace(asarray=upload, float32=np.float32, int32=np.int32,
                           cuda=SimpleNamespace(Device=Device, Event=Event,
                                                get_current_stream=current_stream))
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(device_cache, "_install_prebuilt_images", lambda: None)
    # A fresh cache for this test, with no teardown device synchronization.
    factory = device_cache.cuda_cache(ready=True)(pxlsm._device_table.__wrapped__)
    monkeypatch.setattr(pxlsm, "_device_table", factory)
    return state


def test_table_is_uploaded_once_and_ordered_on_other_streams(cuda_standin):
    state = cuda_standin
    first = pxlsm._device_table(24)
    assert pxlsm._device_table(24) is first
    assert len(state.uploads) == len(state.events) == 1
    assert not state.waits
    state.stream = 2
    assert pxlsm._device_table(24) is first
    assert state.waits == state.events
    assert len(state.uploads) == 1
    state.events[0].done = True
    state.waits.clear()
    assert pxlsm._device_table(24) is first
    assert not state.waits
    state.device = 1
    assert pxlsm._device_table(24) is not first
    assert len(state.uploads) == 2


def test_warm_launches_reuse_the_table_without_a_host_upload(cuda_standin, monkeypatch):
    from woof.core import kernels
    launches = []
    monkeypatch.setattr(kernels, "get_kernel", lambda *args: lambda *call: launches.append(call))
    ny, nx, nlcat = 4, 5, 24
    fields = {name: np.zeros((ny, nx), dtype=np.float32) for name in
              pxlsm.PXLSM_INPUTS + pxlsm.PXLSM_CARRIED + pxlsm.PXLSM_PUBLISHED}
    fields.update(isltyp=np.ones((ny, nx), dtype=np.int32),
                  smois=np.zeros((2, ny, nx), dtype=np.float32),
                  tslb=np.zeros((2, ny, nx), dtype=np.float32))
    atmosphere = {name: np.zeros((ny, nx), dtype=np.float32)
                  for name in pxlsm.PXLSM_ATMOSPHERE}
    pxlsm.prepare_pxlsm(nlcat)
    for step in (1, 2):
        pxlsm.launch_pxlsm(fields, atmosphere,
                          landusef=np.zeros((nlcat, ny, nx), dtype=np.float32),
                          soilcbot=np.zeros((16, ny, nx), dtype=np.float32),
                          dt=10., dtbl=10., itimestep=step, curr_secs=(step-1)*10.,
                          smois_init=1, soil_nudge=0, modis_veg=0,
                          anal_interval=0, domain_shape=(ny, nx))
    assert len(launches) == 2
    assert len(cuda_standin.uploads) == 1
    assert launches[0][2][20] is launches[1][2][20] is cuda_standin.uploads[0]
