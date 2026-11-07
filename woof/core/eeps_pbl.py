"""EEPS (TKE + TKE-dissipation) PBL, ``bl_pbl_physics = 16``.

CUDA launcher around ``kernels/eeps.cu``, the column transcription of WRF
v4.6.1 ``phys/module_bl_eepsilon.F`` (Zhang, Wang and Xue 2020 MWR 148,
1121-1145; Langland and Liou 1996).  The scheme predicts TKE (PEK) and its
dissipation rate (PEP) with two implicit diffusion equations, closes the
eddy diffusivities as ``K = c2 * PEK**2 / PEP``, adds a countergradient
heat/moisture flux in convective layers and a top-down TKE source under
stratocumulus, and does its own implicit vertical diffusion of momentum,
heat, vapour, cloud water and cloud ice: it returns finished tendencies.

Carried state, as WRF v4.7.1's Registry declares it
(Registry.EM_COMMON:1051-1054, package line :3185):

* ``pek_adv``/``pep_adv`` -- the ADVECTED pair, members of WRF's ``scalar``
  array (``i0rusdf=(bdy_interp:dt)``: zero in the input, restart, nest
  boundary interpolation).  The column reads these and writes them back;
  the dycore transports them between PBL calls like every scalar-array
  species.
* ``pek_pbl``/``pep_pbl`` -- the PBL's own copy (``hr``: history and
  restart).  The column reads it only on the first model step, when WRF
  copies it into the advected pair (``itimestep == 1``, :227-230), so a
  cold start is EEPSINIT's minpek/minpep everywhere.

Arithmetic: the default uses CUDA float32 transcendental functions and
permits FMA contraction, with subnormals retained for IEEE division.
``GPUWM_WRF_EXACT=1`` selects the pinned scalar
math and disables contraction through the shared strict
compiler contract.  The column oracle in
``tools/eeps_wrf461_oracle`` supplies synthetic regression columns for
``tests/test_eeps_wrf461_parity.py``. The same replay tool also compares
real native forecast captures; see docs/public/wrf-comparison/eeps-port.md.
WRF never initialises RADSUM, the cloud-top
flux sum (:748); here it starts at zero. An active native out-of-bounds
SZV(KMINRAD-2) expression (:734) is reported by the device health status.

Storage: a coalesced global workspace holds 22 column arrays for the
bounded resident tile; tridiagonal rows are formed where they are solved. It grows with the run's own column depth,
without charging a deep local frame to the CUDA context's thread capacity.
"""
from __future__ import annotations


import numpy as np

from woof.physics_vertical_contract import (
    EEPS_VERTICAL_LEVEL_BOUNDS, outside_vertical_bounds,
    refuse_vertical_levels)

#: Base options. The shared strict compiler contract adds proof options.
MODULE_OPTIONS = ("-std=c++17", "--ftz=false")
MODULE_KEY = "woof.core.kernels:eeps"
_SOURCES = ("glibc_flt32.cuh", "eeps.cu")
from woof.core.physics_inventory import (EEPS_BLOCK as _BLOCK,
    EEPS_TILE_BLOCKS_PER_SM, eeps_workspace_floats)

#: The column-depth contract (woof.physics_vertical_contract): a floor of
#: three levels, where the TKE and epsilon solves over rows 1..km-1
#: (module_bl_eepsilon.F:861-867, :883-889) still have an interior row, and
#: no ceiling, since every work array is compiled one column deep.
VERTICAL_LEVEL_BOUNDS = EEPS_VERTICAL_LEVEL_BOUNDS

#: Column inputs, bottom-up (nz, ny, nx), in kernel argument order.
COLUMN_INPUTS = ("u", "v", "t", "qv", "qc", "qi", "qr", "qs", "qg", "p",
                 "pi", "rho", "rthraten", "dz8w")
#: Surface inputs (ny, nx), in kernel argument order.  XLAND and DX reach
#: WRF's EEPS but eeps2d reads neither.
SURFACE_INPUTS = ("psfc", "qsfc", "tsk", "ust", "rmol", "wspd", "hfx", "qfx")
#: Carried state, rewritten in place, in kernel argument order.
CARRIED = ("pek_pbl", "pep_pbl", "pek_adv", "pep_adv")
#: Mass-level outputs, in kernel argument order.
TENDENCY_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                    "rqcblten", "rqiblten")
#: Full-level outputs on WRF's kts..kte+1 levels: (nz + 1, ny, nx).
FULL_LEVEL_OUTPUTS = ("exch_h", "exch_m")

#: EEPSINIT's cold-start values (module_bl_eepsilon.F:37-38, :1131-1132).
EEPS_MINPEK = np.float32(1.0e-4)
EEPS_MINPEP = np.float32(1.0e-7)


def module_source(nz: int) -> str:
    """The same shared source assembly used by the runtime loader."""
    from woof.core.kernels import module_source_int_defines
    return module_source_int_defines("eeps", (("EEPS_NZ", int(nz)),))


def _module(device: int, nz: int):
    """Use the shared device-bound integer-tier loader and numerics policy."""
    import cupy as cp
    from woof.core.kernels import load_module_int_defines
    with cp.cuda.Device(device):
        return load_module_int_defines("eeps", (("EEPS_NZ", int(nz)),))


def _kernel(nz: int):
    import cupy as cp

    return _module(cp.cuda.Device().id, int(nz)).get_function("eeps_column")


def tile_columns(fn, columns: int) -> int:
    """Bound the workspace by the card's resident column capacity."""
    import cupy as cp
    from cupy_backends.cuda.api import driver

    resident = driver.occupancyMaxActiveBlocksPerMultiprocessor(
        getattr(fn, "kernel", fn).ptr, _BLOCK, 0)
    blocks = max(1, min(EEPS_TILE_BLOCKS_PER_SM, int(resident)))
    capacity = int(cp.cuda.Device().attributes["MultiProcessorCount"]) * blocks * _BLOCK
    wanted = max(_BLOCK, min(int(columns), capacity))
    return ((wanted + _BLOCK - 1) // _BLOCK) * _BLOCK


def local_frame_bytes(nz: int) -> int:
    """The compiled column's per-thread local frame for ``nz`` levels."""
    import cupy as cp
    from cupy.cuda import driver

    fn = _kernel(nz)
    return int(driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        getattr(fn, "kernel", fn).ptr))


def refuse_column_depth(nz: int) -> None:
    if outside_vertical_bounds(int(nz), VERTICAL_LEVEL_BOUNDS):
        raise refuse_vertical_levels(
            "EEPS PBL", VERTICAL_LEVEL_BOUNDS, int(nz),
            breakage=(
                "its TKE/epsilon solves run over rows 1..km-1 bounded by "
                "the surface row, and a shallower column has no interior "
                "row, so the solver would index outside the column"))


def eeps_step(columns, surface, carried, *, dt: float, first_step: bool,
              kernel=None):
    """Run one EEPS call over every column; return the output dict.

    ``columns`` maps :data:`COLUMN_INPUTS` to bottom-up float32
    ``(nz, ny, nx)`` arrays (an absent QR/QS/QG may be a zero plane: WRF
    then passes 0. and the column sees exactly 0.); ``surface`` maps
    :data:`SURFACE_INPUTS` to ``(ny, nx)``; ``carried`` maps
    :data:`CARRIED` to ``(nz, ny, nx)`` arrays that are rewritten in place.
    ``first_step`` is WRF's ``itimestep == 1``.  ``kernel`` replaces the
    compiled column (the parity suite's planted-constant control launches a
    deliberately wrong build through this same argument list).
    """
    import cupy as cp

    f32 = np.float32
    nz = int(columns["u"].shape[0])
    refuse_column_depth(nz)
    shape = tuple(int(s) for s in columns["u"].shape[1:])
    n = int(np.prod(shape))
    for name in COLUMN_INPUTS:
        a = columns[name]
        if a.shape != (nz, *shape) or a.dtype != f32 or not a.flags.c_contiguous:
            raise ValueError(f"EEPS column input {name!r} must be contiguous "
                             f"float32 {(nz, *shape)}, got {a.shape} {a.dtype}")
    for name in SURFACE_INPUTS:
        a = surface[name]
        if a.shape != shape or a.dtype != f32 or not a.flags.c_contiguous:
            raise ValueError(f"EEPS surface input {name!r} must be contiguous "
                             f"float32 {shape}")
    for name in CARRIED:
        a = carried[name]
        if a.shape != (nz, *shape) or a.dtype != f32 or not a.flags.c_contiguous:
            raise ValueError(f"EEPS carried field {name!r} must be contiguous "
                             f"float32 {(nz, *shape)}")
    out = {name: cp.empty((nz, *shape), dtype=f32)
           for name in TENDENCY_OUTPUTS}
    for name in FULL_LEVEL_OUTPUTS:
        out[name] = cp.empty((nz + 1, *shape), dtype=f32)
    out["pblh"] = cp.empty(shape, dtype=f32)
    out["kpbl"] = cp.empty(shape, dtype=cp.int32)
    status = cp.zeros(1, dtype=cp.uint32)
    fn = kernel if kernel is not None else _kernel(nz)
    tile = tile_columns(fn, n)
    workspace = cp.empty(eeps_workspace_floats(nz, tile), dtype=f32)
    args = (tuple(columns[name] for name in COLUMN_INPUTS)
            + tuple(surface[name] for name in SURFACE_INPUTS)
            + tuple(carried[name] for name in CARRIED)
            + tuple(out[name] for name in TENDENCY_OUTPUTS)
            + tuple(out[name] for name in FULL_LEVEL_OUTPUTS)
            + (out["pblh"], out["kpbl"], status, f32(dt),
               np.int32(bool(first_step)), np.int32(n), workspace))
    for offset in range(0, n, tile):
        count = min(tile, n - offset)
        fn(((count + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
           args + (np.int32(offset),))
    from woof.core.health_ledger import read_status
    def refuse(flags):
        if flags & 2:
            raise FloatingPointError(
                "EEPS reached native-undefined SZV(KMINRAD-2), "
                "module_bl_eepsilon.F:734; no boundary clamp is applied")
        raise FloatingPointError("EEPS returned a non-finite output or carried scalar")
    flags = read_status(status, site="eeps-column", describe=refuse)
    if flags:
        refuse(flags)
    return out
