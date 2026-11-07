"""Bandwidth-oriented launches for unchanged FP32 elementwise operations."""
from __future__ import annotations

import cupy as cp
import numpy as np

from woof.core.kernels import get_kernel

_TPB = 128
_WORDS_PER_THREAD = 4


def _supported(arrays):
    return all(isinstance(a, cp.ndarray) and a.dtype == np.dtype(np.float32)
               and a.flags.c_contiguous for a in arrays)


def _overlap(a, b):
    return (a.data.ptr < b.data.ptr + b.nbytes
            and b.data.ptr < a.data.ptr + a.nbytes)


def add_array(src, dst):
    """Add one contiguous row without changing partial-alias semantics."""
    if (not _supported((src, dst)) or src.shape != dst.shape
            or (_overlap(src, dst) and src.data.ptr != dst.data.ptr)):
        return False
    if dst.size:
        kernel = get_kernel('bandwidth_glue', 'glue_add')
        kernel(((dst.size + _TPB * _WORDS_PER_THREAD - 1)
                // (_TPB * _WORDS_PER_THREAD),), (_TPB,),
               (src, dst, np.uint64(dst.size)))
    return True


def prepare_add_arrays(state, pairs):
    """Bind independent ``dst += src`` rows in one four-value launch.

    Unsupported layouts and overlapping rows keep ordered CuPy assignments.
    An exact source/destination alias within one row is safe: every element
    reads its old value before writing its sum.
    """
    if not pairs or len(pairs) > 64:
        return None
    arrays = tuple(a for src, dst in pairs for a in (src, dst))
    if (not _supported(arrays)
            or any(src.shape != dst.shape for src, dst in pairs)):
        return None
    key = tuple((a.data.ptr, a.size) for a in arrays)
    cached = getattr(state, '_glue_add_launch', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    for row, (src, dst) in enumerate(pairs):
        if _overlap(src, dst) and src.data.ptr != dst.data.ptr:
            return None
        if any(_overlap(dst, other)
               for j, pair in enumerate(pairs) if j != row for other in pair):
            return None
    rows = np.zeros((64, 3), dtype=np.uint64)
    rows[:len(pairs)] = [(src.data.ptr, dst.data.ptr, dst.size)
                        for src, dst in pairs]
    table = rows.reshape(-1).view(np.dtype(('V', rows.nbytes)))
    nmax = max(dst.size for src, dst in pairs)
    grid = (max(1, (nmax + _TPB * _WORDS_PER_THREAD - 1)
                // (_TPB * _WORDS_PER_THREAD)), len(pairs))
    kernel = get_kernel('bandwidth_glue', 'glue_add_arrays')

    def launch():
        kernel(grid, (_TPB,), (table,))

    setattr(state, '_glue_add_launch', (key, launch))
    return launch


def total_theta(state):
    """Construct ``thb + thp`` with four values per thread."""
    thb, thp = state.thb, state.thp
    if (not _supported((thb, thp)) or thp.ndim != 3
            or thb.shape not in (thp.shape, (thp.shape[0],))
            or not thp.size):
        return (state.prognostic_theta() if getattr(state, "use_theta_m", 0)
                else state.total_theta())
    # Keep the original total_theta temporary's lifetime and pool reuse.
    out = cp.empty_like(thp)
    kernel = get_kernel('bandwidth_glue', 'glue_total_theta')
    kernel(((thp.size + _TPB * _WORDS_PER_THREAD - 1)
            // (_TPB * _WORDS_PER_THREAD),), (_TPB,),
           (thb, thp, out, np.uint64(thp.size),
            np.int32(thp.shape[1] * thp.shape[2]), np.int32(thb.ndim == 3)))
    return out


def capture_theta_forcing(state):
    """Fuse the eager theta-rate export, preserving every FP32 boundary."""
    dst = state.rthften
    arrays = (state.rth_t, state.mub2d, state.mup, state.c1h, state.c2h,
              state.msft, dst)
    if not _supported(arrays) or state.rth_t.ndim != 3:
        return False
    nz, ny, nx = state.rth_t.shape
    if (dst.shape != (nz, ny, nx) or not dst.size
            or any(a.shape != (ny, nx) for a in arrays[1:3] + arrays[5:6])
            or any(a.shape != (nz,) for a in arrays[3:5])
            or any(_overlap(dst, a) for a in arrays[:-1])):
        return False
    kernel = get_kernel('bandwidth_glue', 'glue_capture_theta_forcing')
    kernel(((dst.size + _TPB * _WORDS_PER_THREAD - 1)
            // (_TPB * _WORDS_PER_THREAD),), (_TPB,),
           (*arrays, np.uint64(dst.size), np.int32(ny * nx),
            np.int32(state.has_msf)))
    return True
