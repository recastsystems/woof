"""One CUDA device/driver/runtime probe, shared by every receipt that needs it.

The probe lives here rather than inside
:func:`woof.native_wrf_distribution.verify_runtime` because a probe bound to
the native-WRF bridge inventory cannot be called from a forecast: the bridge
receipt asks the device what it is, and nothing else in the tree can ask the
same question of the same code.  Both consumers import this module and call
the same function object, so a change to what the probe measures reaches every
receipt that carries a device block instead of only the one it was written for.

The block's key set is the published ``gpuwm-native-wrf-runtime-v1`` shape;
``tests/data/native_wrf_runtime_gpu_block_pre_change.json`` holds it as
transcribed from the source that predates this module.
"""

from __future__ import annotations

from typing import Any

#: Status of a block whose caller declared it did not want a device probed.
GPU_STATUS_NOT_CHECKED = "NOT_CHECKED"

#: Status of a block whose device answered every question asked of it.
GPU_STATUS_PASS = "PASS"

#: CUDA runtime versions the sealed distribution accepts, as
#: ``[inclusive, exclusive)`` in CuPy's integer encoding.
CUDA_RUNTIME_RANGE = (12_000, 13_000)


def gpu_cuda_stack_identity(*, require_gpu: bool = True) -> dict[str, Any]:
    """Device, driver and CUDA-runtime identity as the installed stack reports it.

    With ``require_gpu`` false the caller has declared it is not probing a
    device, and the block says so rather than reporting a value nothing
    measured.  With ``require_gpu`` true the probe fails closed: no device, or
    a CUDA runtime outside :data:`CUDA_RUNTIME_RANGE`, raises rather than
    returning a block a receipt could carry as if it had passed.
    """
    if not require_gpu:
        return {"required": False, "status": GPU_STATUS_NOT_CHECKED}

    import cupy as cp

    count = int(cp.cuda.runtime.getDeviceCount())
    if count < 1:
        raise RuntimeError("CuPy reports no CUDA devices")
    # Ordinal 0 is process-local.  Supervised runs bind the selected physical
    # UUID through CUDA_VISIBLE_DEVICES before this CUDA-facing process starts,
    # so multi-GPU orchestration intentionally sees its pinned card as 0.
    device = cp.cuda.Device(0)
    with device:
        value = cp.arange(16, dtype=cp.float32)
        checksum = float(cp.asnumpy(value.sum()))
        properties = cp.cuda.runtime.getDeviceProperties(0)
        runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
    low, high = CUDA_RUNTIME_RANGE
    is_hip = bool(getattr(cp.cuda.runtime, 'is_hip', False))
    if not is_hip and not low <= runtime_version < high:
        raise RuntimeError(
            "native-WRF runtime requires a CUDA 12.x runtime, got "
            f"{runtime_version}")
    raw_name = properties.get("name", b"")
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode("utf-8", errors="replace")
    if is_hip:
        return _hip_stack_identity(cp, properties, raw_name, count, runtime_version, checksum)
    return {
        "required": True,
        "status": GPU_STATUS_PASS,
        "device_count": count,
        "device_0_name": str(raw_name),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "runtime_version": runtime_version,
        "test_checksum": checksum,
    }


def _hip_stack_identity(cp, properties, name, count, runtime_version, checksum):
    arch = properties['gcnArchName']
    return {
        'required': True, 'status': GPU_STATUS_PASS, 'backend': 'hip',
        'architecture': arch.decode() if isinstance(arch, bytes) else str(arch),
        'device_count': count, 'device_0_name': str(name),
        'driver_version': int(cp.cuda.runtime.driverGetVersion()),
        'runtime_version': runtime_version, 'test_checksum': checksum,
    }
