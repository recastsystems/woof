"""ROCm telemetry protocol checks; these do not simulate AMD execution."""
import json
from types import SimpleNamespace

import pytest

from woof import gpu_smi


def test_rocm_inventory_preserves_bytes_and_physical_identity(monkeypatch):
    payload = {"card0": {"Unique ID": "0x1234", "PCI Bus": "0000:03:00.0",
        "VRAM Total Memory (B)": "17179869184", "VRAM Total Used Memory (B)": "1073741824",
        "Card Series": "AMD Radeon", "GPU use (%)": "73", "Driver version": "6.4"}}
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(payload))
    monkeypatch.setattr(gpu_smi.subprocess, "run", run)
    row, = gpu_smi.inventory()
    assert calls[0][0] == "rocm-smi"
    assert (row["memory.total"], row["memory.used"], row["memory.free"]) == (16384, 1024, 15360)
    assert row["uuid"] == "GPU-AMD-1234"
    assert row["pci.bus_id"] == "0000:03:00.0"
    payload["card0"]["Unique ID"] = "0x0"
    with pytest.raises(RuntimeError, match="unique ID unavailable"):
        gpu_smi.inventory()


def test_amd_mask_uses_runtime_ordinals_and_refuses_ambiguous_rocr(monkeypatch):
    monkeypatch.setenv("WOOF_BACKEND", "hip")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu_smi, "inventory", lambda: [{"uuid": "GPU-AMD-1234", "index": "2", "pci.bus_id": "0000:03:00.0"}])
    assert gpu_smi.visibility_environment("GPU-AMD-1234") == {"CUDA_VISIBLE_DEVICES": "2", "HIP_VISIBLE_DEVICES": "2"}
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="ambiguous physical remapping"):
        gpu_smi.visibility_environment("GPU-AMD-1234")


def test_amd_missing_smi_does_not_select_nvidia(monkeypatch):
    monkeypatch.setenv("WOOF_BACKEND", "hip")
    monkeypatch.setattr(gpu_smi.shutil, "which", lambda name: "/bin/nvidia-smi" if name == "nvidia-smi" else None)
    assert gpu_smi.executable() is None


def test_report_labels_rocm_driver_without_cuda_capability(monkeypatch):
    from woof.report_bundle import _gpu_environment
    monkeypatch.setenv("WOOF_BACKEND", "hip")
    monkeypatch.setattr(gpu_smi, "inventory", lambda: [{"index": "0", "name": "AMD GPU", "memory.total": 16384,
        "driver_version": "6.4", "pci.bus_id": "0000:03:00.0"}])
    result = _gpu_environment()
    assert result["probe"] == "rocm-smi"
    assert result["backend"] == "hip"
    assert result["devices"][0]["total_memory_bytes"] == 17179869184
    assert "compute_capability" not in result["devices"][0]
    assert "nvidia_driver_version" not in result


def test_shared_target_uses_gfx_identity_for_the_requested_device():
    from woof.backend import compilation_target, image_target
    runtime = SimpleNamespace(is_hip=True, getDeviceProperties=lambda i: {"gcnArchName": [b"gfx942", b"gfx1100"][i]})
    xp = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
    assert compilation_target(1, xp=xp) == "gfx1100"
    assert image_target(0, xp=xp) == "gfx942"


def test_hip_limit_uses_public_size_t_abi_instead_of_cupy_stub(monkeypatch):
    import ctypes
    import sys
    from woof.hip_backend import runtime_limit
    calls = []
    def read(pointer, limit):
        calls.append(("read", limit))
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_size_t))[0] = 8192
        return 0
    def write(limit, value):
        calls.append(("write", limit, value))
        return 0
    lib = SimpleNamespace(hipDeviceGetLimit=read, hipDeviceSetLimit=write)
    runtime = SimpleNamespace(is_hip=True, deviceSetLimit=lambda *a: pytest.fail("CuPy HIP setter is a stub"))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
    monkeypatch.setattr(ctypes, "CDLL", lambda name: lib)
    assert runtime_limit(0) == 8192
    assert runtime_limit(0, 16384) is None
    assert calls == [("read", 0), ("write", 0, 16384)]
    assert lib.hipDeviceGetLimit.argtypes == [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    assert lib.hipDeviceSetLimit.argtypes == [ctypes.c_int, ctypes.c_size_t]
