"""Fixed-order wave reductions: the DIA quartet exchange as a gather, and the partition sums.

THE BREAKAGE THIS PREVENTS (woof hex 0.4.14, 2026-11-06): the DIA scatter and the partition diagnostics summed
with float atomicAdd, so two identical runs gave different wave words (4,863,958 of 44,807,348 after 3 h on
x1.40962, 22,315 already at t=0) and, with roughness feedback on, different atmospheres. ``ordered.cu`` gives
every output word one owner thread and one summation order. The quartet map is built once, on the device.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SOURCE = Path(__file__).with_name("ordered.cu")
#: Threads per cell block for the DIA gather (a multiple of 32).
BLOCK = 256
#: The default dynamic shared-memory limit without the opt-in attribute.
_DEFAULT_SHARED = 48 * 1024


def compile_module(*, key, nf=None, nd=None, dia_coefficient=None, fast_math=True):
    """The fixed-order kernels as their own module, with the solver's spectral sizes and DIA coefficient.

    ``key`` names the module in the kernel manifest, recorded here beside its one compile site.
    """
    import cupy as cp
    from woof.certify.kernel_manifest import record_module
    options = ["--std=c++17"]
    if fast_math:
        options.append("--use_fast_math")
    if nf is not None:
        options.append(f"-DNF={int(nf)}")
    if nd is not None:
        options.append(f"-DND={int(nd)}")
    if dia_coefficient is not None:
        options.append(f"-DDIA_COEFFICIENT={dia_coefficient}f")
    source = SOURCE.read_text()
    options = tuple(options)
    module = cp.RawModule(code=source, options=options)
    record_module(key, source=source, options=options, module=module)
    return source, options, module


class OrderedDia:
    """The DIA quartet exchange for one (frequency table, direction count), gathered in a fixed order.

    The constructor builds the quartet geometry and the destination-sorted contribution list on the device
    (once; it may synchronise). ``launch`` only enqueues one kernel: no allocation, copy or host read, so it
    is safe inside graph capture and on the shared asynchronous ABI.
    """

    def __init__(self, module, frequency, df, nd, logq, *, diagonal):
        import cupy as cp
        self.frequency, self.df = frequency, df
        self.nf, self.nd = int(frequency.size), int(nd)
        self.logq = np.float32(logq)
        ns = self.ns = self.nf * self.nd
        self.geometry = cp.empty(ns * 8, cp.float32)
        self.targets = cp.empty(ns * 16, cp.int32)
        module.get_function("wave_dia_quartets")(((ns + 127) // 128,), (128,), (
            frequency, np.int32(self.nf), np.int32(self.nd), self.logq, self.geometry, self.targets))
        flat = self.targets.astype(cp.int64)
        code = cp.arange(flat.size, dtype=cp.int64)
        keep = flat >= 0
        dest, code = flat[keep], code[keep]
        # Unique keys: destination first, then the source's (bin, mirror, leg, corner) code. Every destination
        # adds its contributions in this one order on every run.
        order = cp.argsort(dest * flat.size + code)
        self.entries = code[order].astype(cp.int32)
        self.offsets = cp.zeros(ns + 1, cp.int32)
        self.offsets[1:] = cp.cumsum(cp.bincount(dest, minlength=ns)).astype(cp.int32)
        self.kernel = module.get_function("wave_dia_ordered")
        self.words = (7 if diagonal else 3) * ns
        device = cp.cuda.Device()
        limit = int(device.attributes.get("MaxSharedMemoryPerBlockOptin", _DEFAULT_SHARED))
        need = 4 * self.words
        self.slab = None
        if need <= _DEFAULT_SHARED:
            self.shared = need
        elif need <= limit:
            self.kernel.max_dynamic_shared_size_bytes = need
            self.shared = need
        else:
            # A spectrum too large for one block's shared memory works from a per-block global slab instead.
            self.blocks = 4 * int(device.attributes["MultiProcessorCount"])
            self.slab = cp.empty(self.blocks * self.words, cp.float32)
            self.shared = 0
        self.diagonal = bool(diagonal)

    def launch(self, e, rate, diagonal, frequency, df, depth, moments, nc, *, dt=0., limited=False):
        """Add the DIA rate (and the diagonal of its Jacobian) of ``nc`` cells onto ``rate``/``diagonal``."""
        nc = int(nc)
        if nc <= 0:
            return
        if diagonal is not None and not self.diagonal:
            raise ValueError("this DIA map was built without the Jacobian diagonal's shared memory")
        blocks = nc if self.slab is None else min(nc, self.blocks)
        self.kernel((blocks,), (BLOCK,), (
            e, rate, np.uintp(0) if diagonal is None else diagonal, frequency, df, depth, moments,
            self.geometry, self.targets, self.offsets, self.entries,
            np.uintp(0) if self.slab is None else self.slab,
            np.int32(nc), np.int32(self.nf), np.int32(self.nd), np.float32(dt), np.int32(bool(limited)),
            np.int32(self.words)), shared_mem=self.shared)
