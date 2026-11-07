"""Bougeault-Lacarrere TKE PBL (BouLac), ``bl_pbl_physics = 8``.

CUDA launcher around ``kernels/boulac.cu``, the transcription of WRF v4.6.1
``phys/module_bl_boulac.F`` (byte-identical to v4.7.1's), graded word for
word against that module's gfortran object by
``tools/boulac_wrf461_oracle`` (tests/test_boulac_wrf461_parity.py).

WHAT THE SCHEME OWNS.  BouLac predicts TKE with Bougeault and Lacarrere's
(1989) parcel-displacement length scales: for every level the distance a
parcel with that level's TKE can rise and sink against the stratification,
combined into a dissipation length (geometric mean) and a mixing length
(minimum), each capped by the distance to the ground and, under BEP, by the
building length scale.  It then does its OWN implicit vertical diffusion of
U, V, theta, QV and QC with a Troen-Mahrt counter-gradient heat term, so it
returns finished TENDENCIES, the MYJ shape.  Its surface coupling is UST,
HFX and QFX only, which the revised MM5, classic MM5 and Eta surface layers
all publish -- exactly the isfc 1 / 2 pairing WRF's physics_init enforces.

Under BEP / BEP+BEM (``sf_urban_physics`` 2 or 3) the urban model's implicit
and explicit source terms for momentum, heat, moisture and TKE enter the
same solves, the ground distance is blended with the urban one by the urban
fraction and written back, and the building length scale caps both lengths
(module_bl_boulac.F:261-284, :372-382).

THE ONE DECLARED DIVERGENCE: WRF's rural momentum drag
``-ust**2/dz/|V1|`` is 0/0 when ``ust`` and the lowest-level wind are both
zero, and WRF writes a NaN U/V column there.  The port writes the drag's
limit, -0. (kernel header).  Every finite WRF word is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

import cupy as cp
import numpy as np

from woof.core.kernels import get_kernel
from woof.core.state import DTYPE
from woof.physics_vertical_contract import (
    BOULAC_VERTICAL_LEVEL_BOUNDS, outside_vertical_bounds,
    refuse_vertical_levels)

# The persistent field tables and the output roster live in the
# runtime-free inventory, so the VRAM estimate prices exactly what
# initialize_physics and this launcher allocate.
from woof.core.physics_inventory import (  # noqa: E402
    BOULAC_BEP_INPUTS, BOULAC_BLOCK, BOULAC_COLUMN_OUTPUTS,
    BOULAC_FLUX_OUTPUTS)

_TPB = BOULAC_BLOCK

#: The launcher's own level-count bound; restated for the standalone
#: preparation gate as woof.physics_vertical_contract.
VERTICAL_LEVEL_BOUNDS = BOULAC_VERTICAL_LEVEL_BOUNDS

#: Mass-level float32 inputs, ``(nz, ny, nx)``, in kernel argument order.
BOULAC_COLUMN_INPUTS = ("dz", "u", "v", "th", "rho", "qv", "qc")
#: Surface float32 inputs, ``(ny, nx)``, in kernel argument order.
BOULAC_SURFACE_INPUTS = ("hfx", "qfx", "ust")

#: WRF's cold start: BOULACINIT writes TKE_PBL = 0.0001 on the mass levels
#: and zeroes the tendencies and EXCH_H (module_bl_boulac.F:940-954).  It is
#: also the scheme's TKE floor ``temin`` (:29, :546-548).
BOULAC_TKE_COLD_START = 1.0e-4


def launch_boulac(columns: Mapping, surface: Mapping, tke, *, dt: float,
                  cp_air: float, g: float, outputs: Mapping | None = None,
                  bep: Mapping | None = None, frc_urb2d=None,
                  idiff: int = 0) -> dict:
    """Run one BouLac step and return its outputs (allocated if not given).

    ``tke`` is the carried TKE_PBL column, rewritten in place.  ``bep`` is
    the urban PBL handoff (:data:`BOULAC_BEP_INPUTS`, ``sf_bep`` with nz+1
    faces) and ``frc_urb2d`` the urban fraction; supplying them is WRF's
    ``flag_bep`` (sf_urban_physics 2/3), and ``dlg_bep`` is then rewritten
    in place as WRF does.  ``outputs`` may carry the persistent flux
    diagnostics (WU_TUR..WQ_TUR) so that ``idiff = 1`` leaves them as they
    were, as WRF's grid arrays are left.
    """
    shape = surface["hfx"].shape
    if len(shape) != 2:
        raise ValueError("launch_boulac requires (ny, nx) surface arrays")
    nz = int(columns["dz"].shape[0])
    if outside_vertical_bounds(nz, BOULAC_VERTICAL_LEVEL_BOUNDS):
        raise refuse_vertical_levels(
            "BouLac PBL", BOULAC_VERTICAL_LEVEL_BOUNDS, nz,
            breakage=(
                "the kernel holds at most "
                f"{BOULAC_VERTICAL_LEVEL_BOUNDS[1]} levels per column "
                "(BOULAC_KMAX in kernels/boulac.cu) -- raise the define and "
                "re-measure rather than truncating -- and below three levels "
                "the shear and buoyancy terms, which live on the interior "
                "levels kts+1..kte-1, do not exist."))
    if isinstance(idiff, bool) or not isinstance(idiff, (int, np.integer)):
        raise TypeError("BouLac idiff must be an integer")
    flag_bep = bep is not None
    if flag_bep != (frc_urb2d is not None):
        raise ValueError("BouLac's BEP coupling needs both the source terms "
                         "and the urban fraction, or neither")
    if flag_bep and set(bep) != set(BOULAC_BEP_INPUTS):
        raise ValueError(f"BouLac BEP keys must be exactly {BOULAC_BEP_INPUTS}")

    def check(array, expected, what):
        if (array.shape != expected or array.dtype != DTYPE
                or not array.flags.c_contiguous):
            raise ValueError(
                f"launch_boulac requires a contiguous float32 {expected} "
                f"{what}")

    column_arrays = [columns[name] for name in BOULAC_COLUMN_INPUTS]
    for name, array in zip(BOULAC_COLUMN_INPUTS, column_arrays):
        check(array, (nz, *shape), name)
    check(tke, (nz, *shape), "tke")
    surface_arrays = [surface[name] for name in BOULAC_SURFACE_INPUTS]
    for name, array in zip(BOULAC_SURFACE_INPUTS, surface_arrays):
        check(array, shape, name)

    out = dict(outputs) if outputs is not None else {}
    for name in BOULAC_COLUMN_OUTPUTS + BOULAC_FLUX_OUTPUTS:
        if name not in out:
            out[name] = cp.zeros((nz, *shape), dtype=DTYPE)
        check(out[name], (nz, *shape), name)
    if "pblh" not in out:
        out["pblh"] = cp.zeros(shape, dtype=DTYPE)
    check(out["pblh"], shape, "pblh")

    if flag_bep:
        bep_arrays = []
        for name in BOULAC_BEP_INPUTS:
            check(bep[name], (nz + int(name == "sf_bep"), *shape), name)
            bep_arrays.append(bep[name])
        check(frc_urb2d, shape, "frc_urb2d")
        frc = frc_urb2d
    else:
        # The kernel never reads these when flag_bep is 0; CuPy cannot
        # marshal a null pointer, so reuse the existing device allocation.
        dummy = tke
        bep_arrays = [dummy] * len(BOULAC_BEP_INPUTS)
        frc = dummy

    n = int(np.prod(shape))
    if n == 0:
        return _aliased(out)
    blocks = (n + _TPB - 1) // _TPB
    kernel = get_kernel("boulac", "boulac_column")
    kernel((blocks,), (_TPB,),
           tuple(column_arrays) + (tke,) + tuple(surface_arrays)
           + (frc, out["pblh"])
           + tuple(out[name] for name in BOULAC_COLUMN_OUTPUTS)
           + tuple(out[name] for name in BOULAC_FLUX_OUTPUTS)
           + tuple(bep_arrays)
           + (DTYPE(dt), DTYPE(cp_air), DTYPE(g), np.int32(flag_bep),
              np.int32(idiff), np.int32(nz), np.int32(n)))
    return _aliased(out)


def _aliased(out: dict) -> dict:
    """Add the PBL slot's ``du/dv/dtheta/dqv/dqc`` names: the SAME arrays."""
    out.update(du=out["rublten"], dv=out["rvblten"],
               dtheta=out["rthblten"], dqv=out["rqvblten"],
               dqc=out["rqcblten"])
    return out


def validate_boulac_outputs(outputs, tke) -> str | None:
    """Name the first non-finite BouLac output field, or ``None``.

    The port's one divergence from WRF exists to make this hold: WRF's
    only non-finite path in the scheme is the calm, zero-stress 0/0 the
    kernel replaces by its limit. A NaN here is therefore a defect.
    This verification helper deliberately synchronizes and is never called
    by the production driver.
    """
    for name in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                 "el_pbl", "exch_h", "pblh"):
        if not bool(cp.all(cp.isfinite(outputs[name]))):
            return name
    if not bool(cp.all(cp.isfinite(tke))):
        return "tke_pbl"
    return None
