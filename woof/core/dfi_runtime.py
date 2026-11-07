"""Setup binding and whole-pass submission for the Rust DFI runtime.

The CUDA driver, not a Python callback, is passed to Rust. Python only
constructs the finite graph and keeps its resources alive. Integration is
one Rust submission and has no Python step loop or per-step readback.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import sys
import threading

BRIDGE_ENV = 'WOOF_DFI_BRIDGE'
ABI_VERSION = 1
_lock = threading.Lock()
_library = None
_driver = None


class NativePass(C.Structure):
    _fields_ = [('start',C.c_int64),('stop',C.c_int64),
                ('backward',C.c_uint32),('filter',C.c_uint32),
                ('reset',C.c_uint32),('reserved',C.c_uint32)]


def library_name():
    if os.name == 'nt':
        return 'arwen_dfi.dll'
    return 'libarwen_dfi.dylib' if sys.platform == 'darwin' else 'libarwen_dfi.so'


def library_candidates():
    from woof.bridges import default_bridge_dir, packaged_bridge_dir
    root = Path(__file__).resolve().parents[2]
    name = library_name()
    override = os.environ.get(BRIDGE_ENV)
    candidates = [Path(override)] if override else []
    candidates.extend((root/'tools/dfi-runtime/target/release'/name,
                       root/'tools/dfi-runtime/target/debug'/name,
                       root/'libexec/bridges'/name,
                       packaged_bridge_dir()/name,
                       default_bridge_dir()/name))
    return tuple(candidates)


def resolve():
    override = os.environ.get(BRIDGE_ENV)
    if override and not Path(override).is_file():
        raise FileNotFoundError(f'{BRIDGE_ENV} names a missing file: {override}')
    for candidate in library_candidates():
        if candidate.is_file():
            return candidate.resolve()
    from woof.bridges import artifact_remedy
    raise FileNotFoundError('Rust DFI runtime was not found; '+
        artifact_remedy(env_var=BRIDGE_ENV, filename=library_name(),
            subject='DFI integration', crate_relative='tools/dfi-runtime',
            one_liner='cargo build --release --locked --offline', artifact='arwen_dfi'))


def load():
    global _library
    with _lock:
        if _library is None:
            lib = C.CDLL(str(resolve()))
            lib.aw_dfi_runtime_abi.restype = C.c_uint32
            version = lib.aw_dfi_runtime_abi()
            if version != ABI_VERSION:
                raise RuntimeError(f'Rust DFI ABI {version}; required {ABI_VERSION}')
            lib.aw_dfi_submit_compiled_pass.argtypes = (
                C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64, C.c_int64,
                C.c_int64, C.c_uint64)
            lib.aw_dfi_submit_compiled_pass.restype = C.c_int32
            lib.aw_dfi_weights.argtypes = (C.c_int32, C.c_float, C.c_float,
                C.c_int32, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t))
            lib.aw_dfi_weights.restype = C.c_int32
            lib.aw_dfi_lifecycle.argtypes = (C.c_int32,C.c_int64,C.c_int64,
                C.c_uint32,C.POINTER(NativePass),C.c_size_t,
                C.POINTER(C.c_size_t),C.POINTER(C.c_int64))
            lib.aw_dfi_lifecycle.restype = C.c_int32
            _library = lib
        return _library


def submit_compiled_pass(graph, stream, *, start_ticks, stop_ticks,
                         step_ticks, steps):
    """Submit once, returning asynchronously; callers own the completion boundary."""
    global _driver
    lib = load()
    with _lock:
        if _driver is None:
            _driver = C.WinDLL('nvcuda.dll') if os.name == 'nt' else C.CDLL('libcuda.so.1')
        driver = _driver
    status = lib.aw_dfi_submit_compiled_pass(
        C.cast(driver.cuGraphLaunch, C.c_void_p), graph.graphExec, stream.ptr,
        start_ticks, stop_ticks, step_ticks, steps)
    if status:
        raise RuntimeError(f'Rust DFI compiled-pass submission failed ({status}); '
                           'the pass cannot be retried on this state')


def weights(duration, dt, cutoff, selector):
    """Construct the native REAL coefficient vector once at setup, in Rust."""
    import numpy as np
    if selector == 8:
        import warnings
        from woof.dfi_control import RHO_LABEL
        warnings.warn('Opt-in DFI filter 8: '+RHO_LABEL, stacklevel=2)
    if (not isinstance(duration, int) or not 0 < duration < (1 << 31)
            or not np.isfinite(dt) or dt == 0):
        raise ValueError('DFI filter needs positive integer seconds and nonzero dt')
    capacity = int(1+duration/abs(dt))
    out = np.empty(capacity, np.float32)
    written = C.c_size_t()
    status = load().aw_dfi_weights(duration,dt,cutoff,selector,out.ctypes.data,
                                  capacity,C.byref(written))
    if status or written.value != capacity:
        raise ValueError(f'native DFI filter construction failed ({status}, {written.value}/{capacity})')
    return out


def lifecycle(selector, backward_stop, forward_stop, *, restart=False):
    """Resolve native DFL/DDFI/TDFI setup windows and forecast start."""
    passes = (NativePass*2)()
    count, forecast_start = C.c_size_t(),C.c_int64()
    status = load().aw_dfi_lifecycle(selector,backward_stop,forward_stop,
        int(restart),passes,2,C.byref(count),C.byref(forecast_start))
    if status:
        raise ValueError(f'native DFI lifecycle construction failed ({status})')
    return tuple(passes[:count.value]),forecast_start.value

