"""GFS nonlocal PBL (Hong-Pan), ``bl_pbl_physics = 3``.

CUDA launcher around ``kernels/gfs_pbl.cu``, the device transcription of WRF
v4.6.1 ``phys/module_bl_gfs.F``: the ``BL_GFS`` wrapper, ``MONINP`` and the
``TRIDI2``/``TRIDIN``/``TRIDIT`` tridiagonal solvers.  Graded word for word
against the gfortran build of the byte-unmodified module in
``tools/gfs_pbl_wrf461_oracle`` (tests/test_gfs_pbl_wrf461_parity.py).

WHAT THE SCHEME OWNS.  A first-order nonlocal K-profile closure: the PBL
height from a bulk Richardson number with a thermal excess, a cubic
K profile below it scaled by the surface velocity scale, a local
Richardson-number closure above it, countergradient heat transport, and its
OWN implicit vertical diffusion of heat, vapour, momentum and the
condensate tracers (cloud water, plus cloud ice when the microphysics
carries it).  So, like MYJ and UW, it returns finished TENDENCIES.  Its
column runs in binary64 (``kind_phys`` is 64-bit in module_gfs_machine.F);
the WRF-facing fields are float32.

WHAT IT READS FROM THE REST OF THE ENGINE (module_pbl_driver.F:1406-1424):
the surface layer's UST, PSIM, PSIH, GZ1OZ0, WSPD and BR, the surface
fluxes HFX and QFX, and the PBL step DTBL.  It WRITES back UST (as
``sqrt(stress)`` from ``GZ1OZ0 - PSIM``) and WSPD (the first-level wind
speed plus 1e-9), the two INOUT words of the wrapper, and publishes PBLH
and KPBL.

WORKSPACE.  The column arrays live in a binary64 global pool sized from the
level count, processed in column chunks that fit a VRAM budget, so no level
count is compiled in.
"""

from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

from woof.core.pbl_driver_constants import pbl_driver_constants_f32
from woof.core.device_cache import cuda_cache
from woof.core.state import DTYPE
from woof.core.physics_inventory import (
    GFS_PBL_BLOCK, GFS_PBL_COLUMN_OUTPUTS, GFS_PBL_SURFACE_OUTPUTS,
    GFS_PBL_WORKSPACE_BUDGET_BYTES, gfs_pbl_chunk_columns,
    gfs_pbl_workspace_slots)
from woof.physics_vertical_contract import (
    GFS_PBL_VERTICAL_LEVEL_BOUNDS, outside_vertical_bounds,
    refuse_vertical_levels)

#: Mass-level float32 inputs ``(nz, ny, nx)``, in kernel argument order:
#: WRF's U3D/V3D (A-grid), T3D (t_phy), QV3D/QC3D/QI3D, P3D (p_hyd), PI3D
#: (pi_phy), DZ8W and Z (the mass-level height above sea level phy_prep
#: builds as 0.5*(z_at_w(k)+z_at_w(k+1)), module_big_step_utilities_em.F).
GFS_PBL_COLUMN_INPUTS = ("u", "v", "t", "qv", "qc", "qi", "p", "exner",
                         "dz", "z")
#: Read-only surface inputs ``(ny, nx)``, in kernel argument order.
GFS_PBL_SURFACE_INPUTS = ("psfc", "psim", "psih", "hfx", "qfx", "gz1oz0",
                          "br")

VERTICAL_LEVEL_BOUNDS = GFS_PBL_VERTICAL_LEVEL_BOUNDS

_KERNEL_DIR = Path(__file__).parent / "kernels"


def module_source() -> str:
    """The exact source the loader hands NVRTC (preamble, headers, unit)."""
    from woof.core.kernels import module_source as assemble
    return assemble("gfs_pbl")


#: Compile options.  ``--ftz=false``: the BL_GFS wrapper's binary32 half
#: (Q1 = q/(1+q), the pressure and geopotential conversions) meets
#: subnormal condensate on real columns, and the gfortran build keeps
#: those words; CuPy's RawModule would append -ftz=true, so this unit takes
#: the direct NVRTC route the exact legacy-radiation units use.
MODULE_OPTIONS = ("-std=c++17", "--ftz=false")


@cuda_cache(maxsize=None)
def _module():
    from woof.core.kernels import load_module
    return load_module("gfs_pbl")


def _check(name, array, shape, dtype):
    if (array.shape != shape or array.dtype != dtype
            or not array.flags.c_contiguous):
        raise ValueError(
            f"the GFS PBL requires {name} as a contiguous {np.dtype(dtype)} "
            f"array of shape {shape}, got {array.dtype} {array.shape}")


def gfs_pbl_step(columns, surface, *, dt: float, flqi: bool,
                 budget_bytes: int = GFS_PBL_WORKSPACE_BUDGET_BYTES,
                 module=None) -> dict:
    """Run one BL_GFS call over every column and return its outputs.

    ``columns`` holds :data:`GFS_PBL_COLUMN_INPUTS` ``(nz, ny, nx)`` and
    ``surface`` holds :data:`GFS_PBL_SURFACE_INPUTS` plus ``wspd``, all
    float32 with WRF's k = 1 first.  ``wspd`` is the wrapper's INOUT word
    and is rewritten in place.  ``dt`` is WRF's DTBL (the PBL interval in
    seconds).  ``flqi`` is WRF's ``P_QI >= PARAM_FIRST_SCALAR``: the
    tracer solve then carries cloud ice (NTRAC = 3) and RQIBLTEN is real;
    without ice RQIBLTEN stays the driver's zero.

    Returns the WRF names plus the ``du/dv/dtheta/dqv/dqc/dqi`` aliases the
    PBL slot's coupling reads -- the same device arrays under both keys.
    ``module`` substitutes a compiled module (the planted-constant test).
    """
    nz, ny, nx = columns["u"].shape
    if outside_vertical_bounds(nz, VERTICAL_LEVEL_BOUNDS):
        raise refuse_vertical_levels(
            "GFS PBL", VERTICAL_LEVEL_BOUNDS, nz,
            breakage=(
                "MONINP searches for the PBL top over k = 2..KM/2 and "
                "interpolates against level k-1, so with fewer than four "
                "levels the search is empty and the height reads ZL(0), "
                "outside the column."))
    shape = (ny, nx)
    for name in GFS_PBL_COLUMN_INPUTS:
        _check(name, columns[name], (nz, ny, nx), DTYPE)
    for name in GFS_PBL_SURFACE_INPUTS + ("wspd",):
        _check(name, surface[name], shape, DTYPE)
    out = {name: (cp.zeros if name == "rqiblten" and not flqi else cp.empty)(
               (nz, ny, nx), dtype=DTYPE)
           for name in GFS_PBL_COLUMN_OUTPUTS}
    out["ust"] = cp.empty(shape, dtype=DTYPE)
    out["pblh"] = cp.empty(shape, dtype=DTYPE)
    out["kpbl"] = cp.empty(shape, dtype=cp.int32)
    ncols = ny * nx
    chunk = gfs_pbl_chunk_columns(nz, ncols, budget_bytes)
    pool = cp.empty(gfs_pbl_workspace_slots(nz) * chunk, dtype=cp.float64)
    kernel = (module or _module()).get_function("gfs_pbl_columns")
    fixed = (tuple(columns[name] for name in GFS_PBL_COLUMN_INPUTS)
             + tuple(surface[name] for name in GFS_PBL_SURFACE_INPUTS)
             + (surface["wspd"], out["ust"], out["pblh"], out["kpbl"])
             + tuple(out[name] for name in GFS_PBL_COLUMN_OUTPUTS)
             + (pool,) + pbl_driver_constants_f32()
             + (np.float32(dt), np.int32(bool(flqi)), np.int32(nz),
                np.int32(ncols)))
    for col0 in range(0, ncols, chunk):
        n = min(chunk, ncols - col0)
        blocks = (n + GFS_PBL_BLOCK - 1) // GFS_PBL_BLOCK
        kernel((blocks,), (GFS_PBL_BLOCK,),
               fixed + (np.int32(col0), np.int32(n)))
    out.update(du=out["rublten"], dv=out["rvblten"], dtheta=out["rthblten"],
               dqv=out["rqvblten"], dqc=out["rqcblten"])
    if flqi:
        # WRF's phy_bl_ten adds RQIBLTEN exactly when the moist set carries
        # ice (module_physics_addtendc.F, IF(F_QI)); without it the row is
        # the driver's zero and is withheld rather than coupled.
        out["dqi"] = out["rqiblten"]
    return out


def mass_level_heights(z_interface: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    """phy_prep's ``z = 0.5*(z_at_w(k) + z_at_w(k+1))`` in binary32, in place."""
    cp.add(z_interface[:-1], z_interface[1:], out=out)
    out *= DTYPE(0.5)
    return out


def validate_gfs_pbl_outputs(outputs) -> str | None:
    """Check outputs through the step ledger without a per-call host read."""
    from woof.core.health_ledger import read_status, deferred_note
    names = GFS_PBL_COLUMN_OUTPUTS + ("ust", "pblh")
    n, ns = outputs[names[0]].size, outputs["ust"].size
    status = cp.zeros(1, dtype=cp.uint32)
    if n:
        _module().get_function("gfs_pbl_finite")(
            ((n + 255) // 256,), (256,),
            (np.uint64(n), np.uint64(ns),
             *(outputs[name] for name in names), status))

    def describe(flags):
        name = next(name for bit, name in enumerate(names) if flags & (1 << bit))
        raise FloatingPointError(
            f"the GFS PBL returned non-finite {name}" + deferred_note())

    flags = read_status(status, site="gfs-pbl:outputs", describe=describe)
    return next((name for bit, name in enumerate(names)
                 if flags & (1 << bit)), None)
