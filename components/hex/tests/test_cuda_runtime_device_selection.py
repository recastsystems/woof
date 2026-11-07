"""CUDA constructors retain the card assigned to their current rank thread."""

from concurrent.futures import ThreadPoolExecutor
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from woof.hex.cuda_backend.runtime import CudaRefusal, require_cuda


@pytest.fixture()
def assigned_cards(monkeypatch, tmp_path):
    current = threading.local()
    uses = []
    properties = []
    lock = threading.Lock()

    def get_device():
        return getattr(current, "device", 0)

    def get_properties(device):
        with lock:
            properties.append((threading.get_ident(), device))
        return {
            "major": 12, "minor": 0, "name": f"test card {device}",
            "totalGlobalMem": 1024, "multiProcessorCount": 1,
        }

    class Device:
        def __init__(self, device):
            self.id = device

        def use(self):
            current.device = self.id
            with lock:
                uses.append((threading.get_ident(), self.id))

    nvrtc = ModuleType("cupy.cuda.nvrtc")
    nvrtc.getSupportedArchs = lambda: (120,)
    nvrtc.getVersion = lambda: (13, 0)
    cuda = ModuleType("cupy.cuda")
    cuda.Device = Device
    cuda.runtime = SimpleNamespace(
        getDeviceCount=lambda: 4, getDevice=get_device,
        getDeviceProperties=get_properties, runtimeGetVersion=lambda: 13000,
        driverGetVersion=lambda: 13000,
    )
    cuda.nvrtc = nvrtc
    cupy = ModuleType("cupy")
    cupy.cuda = cuda
    cupy.__version__ = "test runtime"
    # Array ownership now goes through the shared CPU/CUDA/HIP dispatcher.
    monkeypatch.setattr("woof.backend.array_module", lambda: cupy)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.nvrtc", nvrtc)
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path))
    yield SimpleNamespace(current=current, uses=uses, properties=properties, cache=tmp_path)


def test_unspecified_device_stays_on_current_rank_card(assigned_cards):
    assigned_cards.current.device = 3
    capability = require_cuda(cache_dir=assigned_cards.cache)
    assert capability.device_id == 3
    assert assigned_cards.current.device == 3
    assert assigned_cards.uses[-1][1] == 3
    assert assigned_cards.properties[-1][1] == 3


def test_explicit_ordinal_still_selects_that_card(assigned_cards):
    assigned_cards.current.device = 3
    capability = require_cuda(device_id=1, cache_dir=assigned_cards.cache)
    assert capability.device_id == 1
    assert assigned_cards.current.device == 1
    assert assigned_cards.uses[-1][1] == 1


def test_two_rank_threads_keep_independent_current_devices(assigned_cards):
    rendezvous = threading.Barrier(2, timeout=1)

    def worker(device):
        assigned_cards.current.device = device
        rendezvous.wait()
        capabilities = [require_cuda(cache_dir=assigned_cards.cache).device_id for _ in range(3)]
        return capabilities, assigned_cards.current.device

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, device) for device in (1, 2)]
        assert [job.result(timeout=2) for job in jobs] == [([1, 1, 1], 1), ([2, 2, 2], 2)]
    assert {device for _, device in assigned_cards.uses} == {1, 2}


@pytest.mark.parametrize("explicit", [-1, 4])
def test_invalid_explicit_ordinal_keeps_existing_refusal(assigned_cards, explicit):
    assigned_cards.current.device = 2
    with pytest.raises(CudaRefusal, match="outside"):
        require_cuda(device_id=explicit, cache_dir=assigned_cards.cache)
    assert assigned_cards.current.device == 2
    assert not assigned_cards.uses
