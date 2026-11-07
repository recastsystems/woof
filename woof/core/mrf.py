"""MRF planetary-boundary-layer column kernel (``bl_pbl_physics = 99``).

Device launcher for the WRF v4.6.1 ``phys/module_bl_mrf.F`` transcription in
``kernels/mrf.cu``: the Hong-Pan (1996) nonlocal PBL, YSU's predecessor.
The scheme consumes the mass-point wind, temperature, moisture, hydrostatic
pressure, Exner function, layer depths and mass-level heights, and the MM5
surface layer's UST/PSIM/PSIH/GZ1OZ0/WSPD/BR/HFX/QFX.  It returns WRF's six
tendencies (RTHBLTEN already divided by PI3D, as WRF's wrapper does),
PBLH, HOL and KPBL, and rewrites UST in place (``module_bl_mrf.F:774-781``:
MRF recomputes the friction velocity from WSPD and floors it at 0.1 m/s
over land, and the surface layer reads it back on the next step).

Array layout is the engine's ``(nz, ny, nx)`` with x fastest; one CUDA
thread owns one column.  The column arrays live in a global workspace sized
to the columns in flight (the ysu.cu layout), so no level count is compiled
in.  The authority is the oracle fixture of the byte-unmodified WRF module
(``tools/mrf_wrf461_oracle``, ``woof/data/mrf/oracle``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import cupy as cp

from woof.core.physics_inventory import (MRF_BLOCK, MRF_TILE_BLOCKS_PER_SM,
                                          MRFWS_SLOTS, mrf_workspace_floats)
from woof.core.state import DTYPE
from woof.physics_vertical_contract import (
    MRF_VERTICAL_LEVEL_BOUNDS, outside_vertical_bounds,
    refuse_vertical_levels)

__all__ = ["MODULE_OPTIONS", "MRF_3D_OUTPUTS", "MRF_2D_OUTPUTS",
           "compile_mrf_module", "launch_mrf", "module_source", "mrf_kernel",
           "mrf_tile_columns"]

#: Re-exported for the workspace pin (``tests/test_mrf_port.py``).
WORKSPACE_SLOTS = MRFWS_SLOTS

#: The no-FTZ route every kernel graded word for word against a gfortran
#: oracle takes (woof/core/terrain_drag.py, woof/core/ruc_spp.py): NVRTC
#: directly through :mod:`woof.nvrtc_ptx_cache`, because ``cupy.RawModule``
#: appends ``-ftz=true`` after the caller's options and NVRTC honours the
#: last one. The compiler boundary drops ``--fmad=false`` in fast mode.
#: Strict numerics preserves it for the native module replay.
MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.kernels:mrf"


def module_source() -> str:
    """The exact source NVRTC compiles: the loader's own assembly of mrf.cu
    (preamble, its allow-listed float32 math header, the unit)."""
    from woof.core.kernels import module_source as assemble
    return assemble("mrf")


def compile_mrf_module(source: str):
    """Compile ``source`` on the current device through the no-FTZ route.

    The production module is this function applied to :func:`module_source`;
    the planted-constant gate compiles its edited copy through it too, so the
    gate exercises the route the forecast runs.
    """
    from woof import nvrtc_ptx_cache as compiler
    from woof.kernel_compile_notice import observe_module_compile
    from woof.certify.kernel_manifest import record_module

    with observe_module_compile(MODULE_KEY):
        image, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "mrf.cu")
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


@lru_cache(maxsize=None)
def _module(device: int):
    source = module_source()
    with cp.cuda.Device(device):
        module = compile_mrf_module(source)
    return module


def mrf_kernel():
    """The production ``mrf_column`` for the current device."""
    return _module(cp.cuda.Device().id).get_function("mrf_column")

MRF_3D_INPUTS = ("u", "v", "t", "qv", "qc", "qi", "p", "pi", "dz8w", "z")
MRF_2D_INPUTS = ("psfc", "ust", "psim", "psih", "xland", "hfx", "qfx",
                 "gz1oz0", "wspd", "br")
MRF_3D_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                  "rqiblten")
MRF_2D_OUTPUTS = ("pblh", "hol", "kpbl")


def mrf_tile_columns(fn, ncol: int) -> int:
    """Columns to keep in flight: enough to fill the card, and no more.

    The workspace is charged per column in flight, so the tile is the
    card's resident capacity for this kernel and never more than the
    domain (the ysu.py rule, :func:`woof.core.ysu.ysu_tile_columns`).
    """
    dev = cp.cuda.Device()
    sms = dev.attributes["MultiProcessorCount"]
    per_sm = MRF_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        resident = driver.occupancyMaxActiveBlocksPerMultiprocessor(
            fn.kernel.ptr, MRF_BLOCK, 0)
        per_sm = min(per_sm, int(resident))
    except Exception:                              # noqa: BLE001
        pass                                       # keep the measured value
    per_sm = max(1, int(per_sm))
    return int(max(MRF_BLOCK, min(int(ncol), sms * per_sm * MRF_BLOCK)))


def _check(name, arr, shape):
    if (not isinstance(arr, cp.ndarray) or arr.shape != shape
            or arr.dtype != DTYPE or not arr.flags.c_contiguous):
        raise ValueError(f"{name} must be a C-contiguous float32 CuPy array "
                         f"with shape {shape}")


def launch_mrf(u, v, t, qv, qc, qi, p, pi, dz8w, z, *, psfc, ust, psim,
               psih, xland, hfx, qfx, gz1oz0, wspd, br, dt: float,
               flag_qi: bool, kernel=None):
    """Run MRF on every column and return its outputs.

    3-D inputs are ``(nz, ny, nx)`` mass-level fields: ``t`` is temperature
    (WRF's T3D), ``p`` the hydrostatic pressure the PBL driver receives
    (P3D = p_hyd), ``pi`` the Exner function (PI3D), ``dz8w`` the layer
    depths and ``z`` the mass-level heights above sea level (WRF's Z).
    2-D inputs are ``(ny, nx)``.  ``ust`` is INOUT and is overwritten.

    Returns a dict of the six tendencies, ``pblh``, ``hol`` (float32) and
    ``kpbl`` (int32, one-based, BOTTOM-UP: the declared divergence in
    kernels/mrf.cu's header). ``kpbl_native`` retains WRF's unconverted
    top-down integer for direct native module replay. With ``flag_qi`` false WRF never writes
    RQIBLTEN (module_bl_mrf.F:1259) and the cloud-ice column is mixed as
    zeros; ``rqiblten`` is then returned as zeros.

    ``kernel`` is the compiled ``mrf_column`` to launch (default: the
    package's own); the planted-constant gate passes a deliberately
    edited compile through it so the gate exercises this launcher.
    """
    shape = t.shape
    if len(shape) != 3:
        raise ValueError(f"t must have shape (nz, ny, nx), got {shape}")
    nz, ny, nx = shape
    if outside_vertical_bounds(nz, MRF_VERTICAL_LEVEL_BOUNDS):
        raise refuse_vertical_levels(
            "MRF PBL", MRF_VERTICAL_LEVEL_BOUNDS, nz,
            breakage=(
                "the implicit diffusion needs an interior level to solve "
                "for, and the bulk-Richardson PBL-height scan starts one "
                "level above the lowest (module_bl_mrf.F:819), so a "
                "one-level column has neither."))
    for name, arr in zip(MRF_3D_INPUTS, (u, v, t, qv, qc, qi, p, pi, dz8w,
                                         z)):
        _check(name, arr, shape)
    for name, arr in zip(MRF_2D_INPUTS, (psfc, ust, psim, psih, xland, hfx,
                                         qfx, gz1oz0, wspd, br)):
        _check(name, arr, (ny, nx))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"MRF requires a positive finite dt, got {dt}")

    out = {name: cp.empty_like(t) for name in MRF_3D_OUTPUTS}
    if not flag_qi:
        out["rqiblten"].fill(0.0)
    out.update(pblh=cp.empty((ny, nx), dtype=DTYPE),
               hol=cp.empty((ny, nx), dtype=DTYPE),
               kpbl=cp.empty((ny, nx), dtype=cp.int32),
               kpbl_native=cp.empty((ny, nx), dtype=cp.int32))
    ncol = ny * nx
    if ncol == 0:
        return out
    fn = kernel if kernel is not None else mrf_kernel()
    tile = mrf_tile_columns(fn, ncol)
    wskp = nz + 2
    ws = cp.empty(mrf_workspace_floats(nz, tile), dtype=DTYPE)
    try:
        for lo in range(0, ncol, tile):
            count = min(tile, ncol - lo)
            blocks = (count + MRF_BLOCK - 1) // MRF_BLOCK
            fn((blocks,), (MRF_BLOCK,),
               (u, v, t, qv, qc, qi, p, pi, dz8w, z,
                psfc, ust, psim, psih, xland, hfx, qfx, gz1oz0, wspd, br,
                DTYPE(dt), np.int32(1 if flag_qi else 0),
                out["rublten"], out["rvblten"], out["rthblten"],
                out["rqvblten"], out["rqcblten"], out["rqiblten"],
                out["pblh"], out["hol"], out["kpbl"], out["kpbl_native"],
                np.int32(nz), np.int32(ny), np.int32(nx),
                ws, np.int32(wskp), np.int32(lo)))
    finally:
        del ws
    return out


def validate_mrf_outputs(out, ust) -> None:
    """Record all floating-point output faults without draining the step.

    Production installs a HealthLedger and reports failures at its existing
    synchronization boundary. Standalone callers retain immediate errors.
    The validator is compiled separately from the column arithmetic.
    """
    from woof.core.kernels import get_kernel
    from woof.core import health_ledger

    names = MRF_3D_OUTPUTS + ("pblh", "ust", "hol")
    values = tuple(out[name] for name in MRF_3D_OUTPUTS)
    values += (out["pblh"], ust, out["hol"])
    count_3d, count_2d = out["rublten"].size, ust.size
    status = cp.zeros(1, dtype=cp.uint32)
    if max(count_3d, count_2d):
        get_kernel("mrf_validation", "mrf_validate_outputs")(
            ((max(count_3d, count_2d) + 255) // 256,), (256,),
            values + (status, np.int64(count_3d), np.int64(count_2d)))

    def refuse(flags):
        bad = next(name for bit, name in enumerate(names)
                   if flags & (1 << bit))
        raise FloatingPointError(f"the MRF PBL returned non-finite {bad}")

    flags = health_ledger.read_status(status, site="mrf", describe=refuse)
    if flags:
        refuse(flags)
