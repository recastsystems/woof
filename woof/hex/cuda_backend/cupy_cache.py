"""Where CuPy keeps compiled kernel images on disk.

THE BREAKAGE (hex 0.5.0 sealing pass GPU suite, 2026-11-02, test_hex_aot_gpu): CuPy 14 reads ``CUPY_CACHE_DIR``
once, when ``cupy.cuda.compiler`` is imported, into a module-level ``DiskKernelCacheBackend``. Setting the
variable afterwards (a KernelCache's own ``cache_dir``, the per-architecture cache, the numeric-route probe's
temporary cache) no longer moved the cache: a KernelCache given a fresh directory loaded images another run had
left in the box's shared cache and never compiled. :func:`bind` sets the variable and, once CuPy's compiler is
loaded, points its disk backend at the same directory. A backend a service installed itself is left alone.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import sys


def _compiler():
    # Before cupy.cuda.compiler is imported the variable alone decides; never import CuPy from here.
    compiler = sys.modules.get("cupy.cuda.compiler")
    return compiler if hasattr(compiler, "_set_kernel_cache_backend") else None


def bind(directory) -> None:
    """Make ``directory`` CuPy's kernel disk cache for this process."""
    directory = str(directory)
    os.environ["CUPY_CACHE_DIR"] = directory
    compiler = _compiler()
    if compiler is None:
        return
    from cupy.cuda._compiler_cache import DiskKernelCacheBackend
    backend = getattr(compiler, "_kernel_cache_backend", None)
    if type(backend) is not DiskKernelCacheBackend or getattr(backend, "_cache_dir", None) == directory:
        return
    compiler._set_kernel_cache_backend(DiskKernelCacheBackend(directory))


@contextmanager
def bound(directory):
    """:func:`bind` for the duration of a block, then restore the previous variable and backend."""
    previous = os.environ.get("CUPY_CACHE_DIR")
    compiler = _compiler()
    backend = getattr(compiler, "_kernel_cache_backend", None) if compiler is not None else None
    bind(directory)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUPY_CACHE_DIR", None)
        else:
            os.environ["CUPY_CACHE_DIR"] = previous
        if backend is not None:
            compiler._set_kernel_cache_backend(backend)
