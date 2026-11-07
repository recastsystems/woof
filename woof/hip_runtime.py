# SPDX-License-Identifier: Apache-2.0
"""Public HIP host APIs that CuPy does not expose on ROCm."""
from __future__ import annotations

import ctypes as C
import functools
import os
from pathlib import Path


@functools.lru_cache(maxsize=1)
def library():
    override = os.environ.get('WOOF_HIP_RUNTIME_LIBRARY')
    if override:
        return C.CDLL(override)
    root = Path(os.environ.get('ROCM_PATH', os.environ.get('ROCM_HOME', '/opt/rocm')))
    return C.CDLL(str(root / 'lib/libamdhip64.so'))


def _checked(name, arguments, *values):
    lib = library()
    call = getattr(lib, name, None) or getattr(lib, 'gpuwm_' + name)
    call.argtypes, call.restype = arguments, C.c_int
    status = call(*values)
    if status:
        raise RuntimeError(f'{name} failed with HIP error {status}')


def host_register(address, size, flags=1):
    """Register an existing mmap; CuPy 14's HIP hostRegister is a stub."""
    _checked('hipHostRegister', [C.c_void_p, C.c_size_t, C.c_uint],
             int(address), int(size), int(flags))


def host_unregister(address):
    _checked('hipHostUnregister', [C.c_void_p], int(address))


def mapped_pointer(address):
    pointer = C.c_void_p()
    _checked('hipHostGetDevicePointer', [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint],
             C.byref(pointer), int(address), 0)
    return int(pointer.value)


def memcpy3d_async(parameters, stream):
    """Linear pitched copies, using hipMemcpy3DParms' CUDA-compatible ABI.

    Both runtimes define the same ordered position/pitched-pointer/extent
    fields. tilestream.gather guards the offsets; the GPU transport probe
    exercises nonzero offsets and padded row and plane strides.
    """
    _checked('hipMemcpy3DAsync', [C.c_void_p, C.c_void_p], int(parameters), int(stream))


def memcpy3d_function(runtime):
    # The explicit library is used only by the HIP-on-NVIDIA API probe.
    # Default CUDA still calls CuPy's original function directly.
    if runtime.is_hip or (os.environ.get('WOOF_BACKEND') == 'hip-nvidia'
                          and os.environ.get('WOOF_HIP_RUNTIME_LIBRARY')):
        return memcpy3d_async
    return runtime.memcpy3DAsync


def peer_links(devices):
    """HSA's actual link type (XGMI=4), independent of CPU NUMA nodes."""
    rows = {}
    for source in set(devices):
        for destination in set(devices):
            if source == destination:
                continue
            kind, hops = C.c_uint(), C.c_uint()
            try:
                _checked('hipExtGetLinkTypeAndHopCount',
                         [C.c_int, C.c_int, C.POINTER(C.c_uint), C.POINTER(C.c_uint)],
                         source, destination, C.byref(kind), C.byref(hops))
                rows[source, destination] = dict(type='xgmi' if kind.value == 4 else 'other',
                                                  hsa_type=kind.value, hops=hops.value)
            except (AttributeError, RuntimeError) as error:
                rows[source, destination] = dict(type='unknown', reason=str(error))
    return rows


def active_blocks_per_multiprocessor(function, block_size, dynamic_shared=0):
    """Query the loaded HIP image; CuPy 14.2's driver shim is a stub."""
    call = library().hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    call.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
    call.restype = C.c_int
    result = C.c_int()
    status = call(C.byref(result), C.c_void_p(int(function)), int(block_size), int(dynamic_shared))
    if status:
        raise RuntimeError(f'HIP occupancy query failed: {status}')
    return int(result.value)


_DEVICE_PROPERTIES = {
    "MultiProcessorCount": "multiProcessorCount",
    "MaxThreadsPerMultiProcessor": "maxThreadsPerMultiProcessor",
    "WarpSize": "warpSize",
    "MaxThreadsPerBlock": "maxThreadsPerBlock",
    "MaxSharedMemoryPerBlock": "sharedMemPerBlock",
    "MaxSharedMemoryPerMultiprocessor": "maxSharedMemoryPerMultiProcessor",
    "MaxRegistersPerBlock": "regsPerBlock",
    "L2CacheSize": "l2CacheSize",
    "ClockRate": "clockRate",
    "MemoryClockRate": "memoryClockRate",
    "GlobalMemoryBusWidth": "memoryBusWidth",
}


def device_attribute(cp, name, device=None):
    """Native HIP property, or the unchanged CUDA attribute query."""
    runtime = cp.cuda.runtime
    if device is None:
        device = runtime.getDevice()
    if not getattr(runtime, "is_hip", False):
        return int(runtime.deviceGetAttribute(
            getattr(runtime, "cudaDevAttr" + name), device))
    properties = runtime.getDeviceProperties(device)
    key = _DEVICE_PROPERTIES.get(name)
    if key is None or key not in properties:
        raise ValueError(f"No native HIP property for device attribute {name}")
    return int(properties[key])


def initialize_device_attributes(cp):
    """Repair CuPy's cached CUDA-enum attributes from native HIP properties.

    CuPy 14's HIP attribute shim can report one multiprocessor on a 304-CU
    device. Column launchers then size their workspace for one CU, issuing
    thousands of underfilled tiles. Device.attributes shares its dictionary
    by ordinal; update that cache once, without changing any physics caller.
    CUDA and HIP-on-NVIDIA keep their original attribute query.
    """
    if not cp.cuda.runtime.is_hip:
        return {}
    changes = {}
    for ordinal in range(cp.cuda.runtime.getDeviceCount()):
        properties = cp.cuda.runtime.getDeviceProperties(ordinal)
        attributes = cp.cuda.Device(ordinal).attributes
        changed = {}
        for name, prop in _DEVICE_PROPERTIES.items():
            if prop in properties:
                value = int(properties[prop])
                if attributes.get(name) != value:
                    changed[name] = (attributes.get(name), value)
                    attributes[name] = value
        changes[ordinal] = changed
    return changes


def wait_mode(mode, devices=None):
    """Preserve non-scheduling flags; report unsupported live-context changes.

    ROCm has no public hipCtxSetFlags. hipSetDeviceFlags rejects an active
    context and leaves error 216 in the thread's last-error slot; a later
    rocThrust launch then reports that stale error. Leave live contexts alone
    and retain their actual scheduling policy in the receipt.
    """
    hip = library()
    want = {'auto': 0, 'spin': 1, 'yield': 2, 'blocking': 4}[mode]
    current = C.c_int()
    if hip.hipGetDevice(C.byref(current)):
        return {}
    if devices is None:
        count = C.c_int()
        if hip.hipGetDeviceCount(C.byref(count)):
            return {}
        devices = range(count.value)
    result = {}
    for ordinal in devices:
        flags, active = C.c_uint(), C.c_int()
        code = hip.hipDevicePrimaryCtxGetState(int(ordinal), C.byref(flags), C.byref(active))
        how = 'unreadable' if code else 'unchanged'
        if not code:
            if flags.value & 7 == want:
                how = 'already'
            elif not active.value:
                previous = hip.hipPeekAtLastError()
                if previous:
                    how = f'unchanged (pre-existing HIP {previous})'
                else:
                    code = hip.hipDevicePrimaryCtxSetFlags(int(ordinal), (flags.value & ~7) | want)
                    how = 'creation flags' if code == 0 else f'unchanged (HIP {code})'
                    # ROCm 7.2 can report inactive above yet reject this call
                    # with 216. Consume only this newly created, handled
                    # scheduling error, preserving unrelated GPU failures.
                    if code == 216 and hip.hipPeekAtLastError() == 216:
                        hip.hipGetLastError()
            else:
                how = 'unchanged (active HIP context)'
        result[int(ordinal)] = f'{mode} {how}'
    return result
