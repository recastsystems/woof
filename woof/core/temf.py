"""The TEMF pair on the card: sf_sfclay_physics = 10 with bl_pbl_physics = 10.

Device launcher for ``kernels/temf.cu``, WRF v4.6.1's Total Energy - Mass
Flux surface layer (phys/module_sf_temfsfclay.F) and PBL
(phys/module_bl_temf.F), one thread per column.  WRF runs the two only
together (share/module_check_a_mundo.F:2287-2304), and so does woof
(:func:`woof.config.validate_selector_couplings`).

The direct NVRTC loader permits FMA by default and retains subnormals for
IEEE division. The proof opt-in, ``GPUWM_WRF_EXACT=1``, disables contraction. Compiler
options are owned by the shared loader so proof and runtime use one policy.

The PBL kernel retains native WRF arithmetic in both runtime and replay.
``verify_stock_wrf=True`` selects the original surface ABI for historical
oracle replay; the production surface launcher has its own complete ABI.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from woof.core.physics_inventory import (
    TEMF_BLOCK, TEMF_PBL_STATE_3D, TEMF_PBL_STATE_2D, TEMF_SFCLAY_INOUT,
    TEMF_SFCLAY_OUTPUTS, temf_chunk_columns, temf_workspace_floats)

MODULE_KEY = "woof.core.temf:temf"

#: Lowest-level and surface inputs of the surface layer, kernel order.
SFCLAY_INPUTS = ("u1", "v1", "th1", "qv1", "p1", "pi1", "rho1", "z1", "ht",
                 "psfc", "znt", "mavail", "tsk", "fcor", "hd_temf")
#: (nz, ncol) PBL inputs, kernel order; ``p8w`` is (nz+1, ncol).
PBL_COLUMN_INPUTS = ("u", "v", "th", "t", "qv", "qc", "p", "pi", "rho", "z")
PBL_SURFACE_INPUTS = ("znt", "ht", "hfx", "qfx", "tsk", "fcor")
PBL_SURFACE_INOUT = ("ust", "pblh", "kpbl", "u10", "v10", "t2")
PBL_TENDENCIES = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten")

#: The kernel's first-call vertical bound: the scheme reads levels 1..3
#: of its own grid unconditionally (MFCTH(2) reads THUP(3), :864), so a
#: column needs three mass levels; there is no ceiling (global workspace).
from woof.physics_vertical_contract import TEMF_VERTICAL_LEVEL_BOUNDS
VERTICAL_LEVEL_BOUNDS = TEMF_VERTICAL_LEVEL_BOUNDS


def module_source(*, verify_stock_wrf: bool = False) -> str:
    """The exact source string NVRTC compiles."""
    from woof.core.kernels import module_source as shared_source
    head = "#define TEMF_VERIFY_STOCK_WRF 1\n" if verify_stock_wrf else ""
    return head + shared_source("temf")


@lru_cache(maxsize=None)
def _module(device: int, verify_stock_wrf: bool):
    """Direct NVRTC compilation, one module per device and verification arm."""
    import cupy as cp
    from woof.core.kernels import compile_noftz_module
    source = module_source(verify_stock_wrf=verify_stock_wrf)
    key = MODULE_KEY + (":verify-stock-wrf" if verify_stock_wrf else "")
    with cp.cuda.Device(device):
        return compile_noftz_module("temf", source, key)


def _kernel(name: str, verify_stock_wrf: bool = False):
    import cupy as cp

    return _module(cp.cuda.Device().id, bool(verify_stock_wrf)).get_function(
        name)


def _check(arrays, shape, dtype, label):
    import cupy as cp

    for name, array in arrays.items():
        if (not isinstance(array, cp.ndarray) or array.shape != shape
                or array.dtype != dtype or not array.flags.c_contiguous):
            raise ValueError(
                f"TEMF {label} {name!r} must be a C-contiguous {np.dtype(dtype)}"
                f" CuPy array of shape {shape}")


def launch_temf_sfclay(surface, state, outputs, *,
                       verify_stock_wrf: bool = False) -> None:
    """One TEMFSFCLAY call over every column, in place.

    ``surface`` holds :data:`SFCLAY_INPUTS` (the lowest model level and the
    surface fields), ``state`` the WRF INOUT set
    (:data:`woof.core.physics_inventory.TEMF_SFCLAY_INOUT`, ``te1`` being
    a contiguous view of the carried TE_TEMF's surface level), ``outputs``
    the pure outputs (``TEMF_SFCLAY_OUTPUTS``).  All arrays share one
    surface shape and are contiguous float32.
    """
    if not verify_stock_wrf:
        from woof.core.temf_sfclay import launch_temf_sfclay as column
        from woof.core.physics_inventory import TEMF_SFCLAY_COLUMN_OUTPUTS
        pool = {**outputs, **state}
        column(surface,
               {k: state[k] for k in ("te1", "hfx", "wm_temf")},
               {k: pool[k] for k in TEMF_SFCLAY_COLUMN_OUTPUTS})
        return
    import cupy as cp

    shape = surface["psfc"].shape
    _check({n: surface[n] for n in SFCLAY_INPUTS}, shape, cp.float32,
           "surface input")
    _check({n: state[n] for n in TEMF_SFCLAY_INOUT}, shape, cp.float32,
           "surface state")
    _check({n: outputs[n] for n in TEMF_SFCLAY_OUTPUTS}, shape, cp.float32,
           "surface output")
    n = int(np.prod(shape))
    blocks = (n + TEMF_BLOCK - 1) // TEMF_BLOCK
    _kernel("temf_sfclay_column", verify_stock_wrf)(
        (blocks,), (TEMF_BLOCK,),
        tuple(surface[k] for k in SFCLAY_INPUTS)
        + tuple(state[k] for k in TEMF_SFCLAY_INOUT)
        + tuple(outputs[k] for k in TEMF_SFCLAY_OUTPUTS)
        + (np.int32(n),))


def launch_temf_pbl(columns, p8w, surface, inout, carried, carried2d,
                    tendencies, *, dt: float,
                    workspace=None,
                    verify_stock_wrf: bool = False) -> None:
    """One TEMFPBL call over every column, in place.

    ``columns`` holds :data:`PBL_COLUMN_INPUTS` as (nz, ...) arrays (``z``
    the mass-level heights above sea level, WRF's ``z``), ``p8w`` the
    (nz+1, ...) interface pressures, ``surface`` :data:`PBL_SURFACE_INPUTS`,
    ``inout`` :data:`PBL_SURFACE_INOUT` (``kpbl`` int32), ``carried`` the
    thirteen carried (nz, ...) TEMF fields and ``carried2d`` the four 2-D
    ones (``woof.core.physics_inventory.TEMF_PBL_STATE_*``), and
    ``tendencies`` the five (nz, ...) outputs of :data:`PBL_TENDENCIES`.
    ``dt`` is WRF's DTBL.
    """
    import cupy as cp

    nz = int(columns["th"].shape[0])
    shape2 = tuple(columns["th"].shape[1:])
    shape3 = (nz, *shape2)
    if nz < VERTICAL_LEVEL_BOUNDS[0]:
        from woof.physics_vertical_contract import refuse_vertical_levels
        raise refuse_vertical_levels(
            "TEMF PBL", VERTICAL_LEVEL_BOUNDS, nz,
            breakage=("the scheme's second turbulence level reads the "
                      "updraft at its third level (module_bl_temf.F:864), "
                      "which a column of fewer than three levels lacks"))
    _check({n: columns[n] for n in PBL_COLUMN_INPUTS}, shape3, cp.float32,
           "column input")
    _check({"p8w": p8w}, (nz + 1, *shape2), cp.float32, "column input")
    _check({n: surface[n] for n in PBL_SURFACE_INPUTS}, shape2, cp.float32,
           "surface input")
    _check({n: inout[n] for n in PBL_SURFACE_INOUT if n != "kpbl"}, shape2,
           cp.float32, "surface state")
    _check({"kpbl": inout["kpbl"]}, shape2, cp.int32, "surface state")
    _check({n: carried[n] for n in TEMF_PBL_STATE_3D}, shape3, cp.float32,
           "carried state")
    _check({n: carried2d[n] for n in TEMF_PBL_STATE_2D}, shape2, cp.float32,
           "carried state")
    _check({n: tendencies[n] for n in PBL_TENDENCIES}, shape3, cp.float32,
           "tendency")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"TEMF PBL requires a positive finite dt, got {dt}")
    ncol = int(np.prod(shape2))
    kp = nz + 2
    chunk = temf_chunk_columns(nz, ncol)
    workspace_size = temf_workspace_floats(nz, chunk)
    ws = (cp.empty(workspace_size, dtype=cp.float32)
          if workspace is None else workspace)
    _check({"workspace": ws}, (workspace_size,), cp.float32, "workspace")
    kernel = _kernel("temf_pbl_column", verify_stock_wrf)
    args_head = (tuple(columns[k] for k in PBL_COLUMN_INPUTS) + (p8w,)
                 + tuple(surface[k] for k in PBL_SURFACE_INPUTS)
                 + tuple(inout[k] for k in PBL_SURFACE_INOUT)
                 + tuple(carried[k] for k in TEMF_PBL_STATE_3D)
                 + tuple(carried2d[k] for k in TEMF_PBL_STATE_2D)
                 + tuple(tendencies[k] for k in PBL_TENDENCIES))
    for lo in range(0, ncol, chunk):
        count = min(chunk, ncol - lo)
        blocks = (count + TEMF_BLOCK - 1) // TEMF_BLOCK
        kernel((blocks,), (TEMF_BLOCK,),
               args_head + (np.float32(dt), np.int32(nz), np.int32(ncol), ws,
                            np.int32(kp), np.int32(lo)))
