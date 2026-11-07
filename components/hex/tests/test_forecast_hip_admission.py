"""The forecast front door must use the same HIP admission as its kernels."""
import sys
from types import SimpleNamespace

import pytest

from woof.hex import forecast_door as door
from woof.hex.cuda_backend import runtime


def test_hip_front_door_uses_runtime_compile_and_launch(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=True))))
    seen = []
    def admit():
        seen.append(True)
        return SimpleNamespace(as_dict=lambda: {
            "backend": "hip", "sm": "gfx942", "warp_size": 64,
            "compute_capability": None, "hiprtc_version": [7, 2]})
    monkeypatch.setattr(runtime, "require_cuda", admit)
    monkeypatch.setattr(door, "read_nvrtc_targets", lambda: pytest.fail("HIP has no NVRTC targets"))
    verdict = door.admit_architecture()
    assert seen == [True]
    assert verdict["admitted"] and verdict["sm"] == "gfx942"
    assert verdict["compute_capability"] is None


def test_hip_compile_failure_is_not_admitted(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=True))))
    def refuse():
        raise runtime.CudaRefusal("HIP compile/launch admission failed")
    monkeypatch.setattr(runtime, "require_cuda", refuse)
    with pytest.raises(door.ForecastDoorRefusal, match="forecast's kernels cannot run"):
        door.admit_architecture()


def test_hip_card_profile_uses_native_resident_threads(monkeypatch):
    properties = {"name": b"AMD Instinct MI300X", "multiProcessorCount": 304,
                  "maxThreadsPerMultiProcessor": 2048}
    hip = SimpleNamespace(is_hip=True, getDeviceProperties=lambda ordinal: properties)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(runtime=hip)))
    monkeypatch.delenv("WOOF_HEX_NO_LOCAL_GPU", raising=False)
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    profile = door.read_card_profile()
    assert profile.multiprocessors == 304
    assert profile.max_threads_per_sm == 2048
