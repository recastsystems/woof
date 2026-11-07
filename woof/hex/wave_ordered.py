"""Fixed-order wave reductions for woof's spectral wave solver (the DIA quartet exchange and the partition sums).

THE BREAKAGE THIS PREVENTS (hex 0.4.14, 2026-11-06): woof's wave kernels (``diagnostics.cu`` basin and partition
sums, ``kernels.cu`` DIA scatter) summed with float atomicAdd, so threads added in whatever order they arrived.
Two identical 1-card runs (x1.40962 120 km, waves on, 3 h) differed in 4,863,958 of 44,807,348 wave words,
22,315 already in the t=0 frame, and with roughness feedback on, 2 rank processes vs 2 rank threads differed in
386,196,279 atmosphere words. ``kernels/wave_ordered.cu`` gives every output word one owner thread and one
summation order: the same bits on every run, and on processes and threads alike when their forcing is the same.

The kernels and ``OrderedDia`` are byte-for-byte the code woof carries from its next release
(``woof/ocean/waves/ordered.cu`` and ``ordered.py``). ``install`` is the bridge until hex's engine floor has
them: a solver whose woof already orders its reductions (``solver.ordered_reductions``) is left alone; an
older solver gets the ordered kernels in its kernel table, with no other change to its code or state.
"""
from __future__ import annotations

from importlib.resources import files
import json
import math

import numpy as np

SOURCE = files("woof.hex").joinpath("kernels/wave_ordered.cu")
#: Threads per cell block for the DIA gather (a multiple of 32).
BLOCK = 256
#: The default dynamic shared-memory limit without the opt-in attribute.
_DEFAULT_SHARED = 48 * 1024


def compile_module(*, nf=None, nd=None, dia_coefficient=None, fast_math=True):
    """The fixed-order kernels as their own module, with the solver's spectral sizes and DIA coefficient."""
    import cupy as cp
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
    return source, options, cp.RawModule(code=source, options=options)


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


class _OrderedDiaRates:
    """Stands in for woof's ``wave_dia_rates`` RawKernel in an older solver's kernel table: same call, same
    arguments, the ordered gather inside. It refuses a call it was not built for instead of running a wrong map."""

    def __init__(self, dia):
        self.dia = dia

    def __call__(self, grid, block, args, **_):
        e, rate, diagonal, frequency, df, depth, moments, nc, nf, nd, logq = args
        dia = self.dia
        if (int(nf), int(nd)) != (dia.nf, dia.nd) or np.float32(logq) != dia.logq                 or frequency.data.ptr != dia.frequency.data.ptr or df.data.ptr != dia.df.data.ptr:
            raise ValueError("ordered DIA map was built for another frequency table: "
                             f"nf={nf} nd={nd} logq={logq} vs nf={dia.nf} nd={dia.nd} logq={dia.logq}")
        dia.launch(e, rate, diagonal, frequency, df, depth, moments, int(nc))


class _NotLaunched:
    """An atomic-sum kernel the solver lists but never launches; a launch would bring the run-to-run noise back."""

    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"{self.name} sums with float atomics and has no ordered form; the wave solver does not "
                           "launch it (woof.hex.wave_ordered)")


def install(solver):
    """Give a woof ``ActionSolver`` fixed-order reductions; returns who supplied them ("woof" or "hexcore")."""
    if getattr(solver, "ordered_reductions", None):
        return "woof"
    import cupy as cp
    from woof.ocean.waves import action
    packs = json.loads((action.PACKAGE / "source_packs.json").read_text())
    source, options, module = compile_module(nf=solver.nf, nd=solver.nd,
                                             dia_coefficient=packs[solver.source_pack]["dia_coefficient"])
    try:
        from woof.certify.kernel_manifest import record_module
        record_module(f"woof.hex.wave_ordered:{solver.source_pack}:{solver.nf}:{solver.nd}", source=source,
                      options=options, module=module)
    except ImportError:
        pass
    with cp.cuda.Device(solver.state.device.id):
        dia = OrderedDia(module, solver.frequency_ext, solver.df_ext, solver.nd, math.log(solver.spectral.ratio),
                         diagonal=True)
    solver.kernels["wave_dia_rates"] = _OrderedDiaRates(dia)
    solver.kernels["wave_merge_basins"] = module.get_function("wave_merge_basins_ordered")
    solver.kernels["wave_diagnose"] = module.get_function("wave_diagnose_ordered")
    for name in ("wave_basin_energy",):
        if name in solver.kernels:
            solver.kernels[name] = _NotLaunched(name)
    solver._ordered_module, solver._ordered_dia = module, dia
    solver.ordered_reductions = "hexcore"
    return "hexcore"
