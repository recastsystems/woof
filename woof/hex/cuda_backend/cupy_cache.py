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


#: ``0`` keeps a forecast's kernel images in its own ``--scratch`` (the route through hex 0.6.1).
SHARED_ENV = "WOOF_HEX_SHARED_KERNEL_IMAGES"


def forecast_directory(scratch):
    """The kernel image cache the v8.4.1 forecast driver binds: the box's, unless switched off.

    THE COST THIS REMOVES (hex 0.6.1; fork jacobbieker/woof PR 488): the driver bound CuPy's disk
    cache to the run's own ``--scratch``, and every door run makes a fresh one, so each run compiled
    again whatever the box kernel store (woof.hex.kernel_warm) does not hold and rewrote every direct
    NVRTC image there (the fork measured step 1 of a 1 h 4,565-cell run at 45.7 s with a warm kernel
    root, 3.6 s with this). CuPy's disk key, the direct-image identity and the kernel plan's probe
    are each taken after the final compiler options are applied (numerics mode included), so fast,
    strict and exact images never share an entry. The one thing a shared image keeps from the run
    that compiled it is CuPy's temporary program path, which only an ``assert`` failure message
    prints. ``WOOF_HEX_SHARED_KERNEL_IMAGES=0`` restores the per-run cache, and a box root the run
    cannot create ``cupy`` in falls back to it as well.
    """
    if os.environ.get(SHARED_ENV, "1") == "0":
        return scratch
    from ..kernel_warm import cache_root
    shared = cache_root() / "cupy"
    try:
        shared.mkdir(parents=True, exist_ok=True)
    except OSError:
        # A read-only box root must not stop a forecast that can write its own scratch.
        return scratch
    return shared
