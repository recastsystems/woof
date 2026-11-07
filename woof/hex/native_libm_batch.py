"""Batch the existing GNU float function pointers, without substituting a math library.

The optional Rust bridge removes Python/ctypes dispatch per element. Its source
is embedded so the preparation producer digest covers every executed byte.
Hosts without rustc retain the scalar authority. No CUDA context is created.
"""
from functools import lru_cache
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading

import numpy as np

from . import time_ledger

_SOURCE = r'''
// Apache-2.0. The callback is the caller's exact GNU sinf symbol.
use std::{slice, thread};
#[no_mangle]
pub unsafe extern "C" fn batch_sin(
    input: *const f32, output: *mut f32, length: usize, workers: usize,
    sinf: unsafe extern "C" fn(f32) -> f32,
) {
    // Python owns disjoint, contiguous arrays until this synchronous call ends.
    let input = unsafe { slice::from_raw_parts(input, length) };
    let output = unsafe { slice::from_raw_parts_mut(output, length) };
    let workers = workers.max(1).min(length.max(1));
    let chunk = length.div_ceil(workers).max(1);
    if workers == 1 {
        for (a, b) in input.iter().zip(output) { *b = unsafe { sinf(*a) }; }
    } else {
        thread::scope(|scope| {
            for (a, b) in input.chunks(chunk).zip(output.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (x, y) in a.iter().zip(b) { *y = unsafe { sinf(*x) }; }
                });
            }
        });
    }
}
#[no_mangle]
pub unsafe extern "C" fn batch_binary(
    left: *const f32, right: *const f32, output: *mut f32,
    length: usize, workers: usize,
    function: unsafe extern "C" fn(f32, f32) -> f32,
) {
    let left = unsafe { slice::from_raw_parts(left, length) };
    let right = unsafe { slice::from_raw_parts(right, length) };
    let output = unsafe { slice::from_raw_parts_mut(output, length) };
    let workers = workers.max(1).min(length.max(1));
    let chunk = length.div_ceil(workers).max(1);
    if workers == 1 {
        for ((a, b), y) in left.iter().zip(right).zip(output) {
            *y = unsafe { function(*a, *b) };
        }
    } else {
        thread::scope(|scope| {
            for ((a, b), y) in left.chunks(chunk).zip(right.chunks(chunk)).zip(output.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for ((a, b), y) in a.iter().zip(b).zip(y) {
                        *y = unsafe { function(*a, *b) };
                    }
                });
            }
        });
    }
}

'''
_BUILD_LOCK = threading.Lock()


def source_key() -> str:
    """The bridge's address: its embedded source, for this machine and system."""
    return hashlib.sha256((_SOURCE + platform.machine() + platform.system()).encode()).hexdigest()


def packaged_name(key: str) -> str:
    """The wheel's prebuilt bridge for ``key`` (setup.py builds it into woof/hex/libexec)."""
    return f"libhex_libm_batch-{key}.so"


@lru_cache(maxsize=1)
def _library():
    # Source-addressed, atomically published; neither a compiler nor a build is
    # required when the same bridge already exists in the native CPU cache.
    #
    # THE BREAKAGE THIS PREVENTS (30 km globe over 4 cards on the reference system box, 2026-10-27): a host without
    # rustc (every WOOF run box: the image never carries the toolchain) had no bridge, so every batch fell
    # back to one ctypes call per element on the GIL: the vertical velocity damping of each rank's mesh took
    # ~25 s of the dycore build at 1.7 cores for 4 ranks, and every new region mesh paid it.  The wheel now
    # carries the bridge built from this same source (setup.py), found by the same source key first.
    key = source_key()
    packaged = Path(__file__).with_name("libexec") / packaged_name(key)
    root = Path(os.environ.get("WOOF_HEX_NATIVE_CPU_CACHE", str(Path.home()/".cache"/"woof"/"dycore"))) / key
    target = packaged if packaged.is_file() else root / "libm_batch.so"
    with _BUILD_LOCK:
        if not target.is_file():
            compiler = shutil.which("rustc")
            if compiler is None:
                return None
            with time_ledger.stage("prep.libm_bridge_build"):
                root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="build-", dir=root) as temporary:
                    source = Path(temporary)/"libm_batch.rs"
                    source.write_text(_SOURCE)
                    built = Path(temporary)/"libm_batch.so"
                    subprocess.run([compiler, "--edition=2021", "--crate-type=cdylib",
                                    "-C", "opt-level=2", str(source), "-o", str(built)],
                                   check=True, capture_output=True)
                    os.replace(built, target)
        library = ctypes.CDLL(str(target))
    pointer = ctypes.POINTER(ctypes.c_float)
    library.batch_sin.argtypes = [pointer, pointer, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
    library.batch_sin.restype = None
    library.batch_binary.argtypes = [pointer, pointer, pointer, ctypes.c_size_t,
                                    ctypes.c_size_t, ctypes.c_void_p]
    library.batch_binary.restype = None
    return library


def intrinsic_batch(native, values, right=None):
    """Return a batch or None for the caller's unchanged scalar fallback.

Non-finite inputs stay on that fallback to retain its Python float conversion
and NaN payload behavior. Finite float32 to Python float round trips are exact.
"""
    from .host_parallel import all_finite
    if not all_finite(values) or (right is not None and not all_finite(right)):
        return None
    library = _library()
    if library is None:
        return None
    source = np.ascontiguousarray(values)
    result = np.empty(source.shape, dtype=np.float32)
    workers = max(1, int(os.environ.get("WOOF_HEX_HOSTPREP_THREADS", "4")))
    workers = min(workers, max(1, source.size // 65536))
    pointer = ctypes.POINTER(ctypes.c_float)
    if right is None:
        library.batch_sin(source.ctypes.data_as(pointer), result.ctypes.data_as(pointer),
                          source.size, workers, ctypes.cast(native, ctypes.c_void_p))
    else:
        other = np.ascontiguousarray(right)
        if other.shape != source.shape:
            raise ValueError("batched intrinsic arguments must have identical shapes")
        library.batch_binary(source.ctypes.data_as(pointer), other.ctypes.data_as(pointer),
                             result.ctypes.data_as(pointer), source.size, workers,
                             ctypes.cast(native, ctypes.c_void_p))
    return result


# Existing callers retain their exact unary API.
sin_batch = intrinsic_batch
