"""The K-epsilon PBL (Zonato et al. 2022), ``bl_pbl_physics = 17``.

CUDA launcher around ``kernels/keps.cu``, the device transcription of WRF
v4.7.1 ``phys/module_bl_keps.F``: the 3-D driver ``keps`` and the column
scheme ``keps1D`` with every routine it calls. The earlier v4.6.1 synthetic
oracle is historical evidence only. The v4.7.1 real-input replay is recorded
separately, including the native module's undefined variables.

WHAT THE SCHEME OWNS.  Prognostic turbulent kinetic energy, its dissipation
rate and the turbulent potential energy (WRF's TKE_PBL, DISS_PBL, TPE_PBL),
advanced each call by an analytic solution of the coupled TKE/dissipation
source terms followed by implicit vertical diffusion; a Prandtl number
profile from the surface layer (PR_PBL); a counter-gradient heat flux from
the TPE; and the diagnostic turbulent fluxes WU_TUR, WV_TUR, WT_TUR.  The
first two calls of a run (WRF's ``xtime <= 2``, which is ``itimestep``)
reset the state to the scheme's floors and return zero tendencies.

WHAT IT READS FROM THE REST OF THE ENGINE (module_pbl_driver.F:1877-1906):
the surface layer's UST, HFX, QFX, BR, ZNT, PSIM and PSIH; the radiation
step's held RTHRATEN; the hydrostatic P8W; the column's DZ8W, RHO, PI and
U/V/TH/QV/QC.  It receives TSK, MOL, FRC_URB2D, FLAG_BEP and the BEP source
terms and reads none of them (the kernel header has the proof).

THE THREE ADVECTED SCALARS (TKE_ADV, DISS_ADV, TPE_ADV) are WRF's transport
copies of the state, used only when the module's compile-time parameter
``bl_keps_adv`` is 1; WRF v4.7.1 ships it as 0, so the scheme sets them to
its floors on the first two calls and never reads or writes them again.
They are registered scalar-package fields and use the engine's common
mass-coupled transport. The closure keeps the shipped bl_keps_adv=0 arm.
"""

from __future__ import annotations

import numpy as np

from woof.core.physics_inventory import (
    KEPS_BLOCK, KEPS_DIAGNOSTICS_3D, KEPS_SLOTS, KEPS_STATE_2D,
    KEPS_STATE_3D, KEPS_TILE_BLOCKS_PER_SM, keps_workspace_floats)
from woof.physics_vertical_contract import (
    KEPS_VERTICAL_LEVEL_BOUNDS, outside_vertical_bounds,
    refuse_vertical_levels)

#: The launcher's own level-count bound, restated for the preparation gate
#: as woof.physics_vertical_contract.KEPS_VERTICAL_LEVEL_BOUNDS.
VERTICAL_LEVEL_BOUNDS = KEPS_VERTICAL_LEVEL_BOUNDS

#: WRF module_model_constants: cp = 7*r_d/2 with r_d = 287, g = 9.81.
KEPS_CP = 1004.5
KEPS_G = 9.81

#: Mass-level float32 inputs, ``(nz, ny, nx)``, in kernel argument order.
KEPS_MASS_INPUTS = ("u", "v", "th", "qv", "qc", "rho", "dz8w", "pi",
                    "rthraten")
#: Interface input, ``(nz+1, ny, nx)``.
KEPS_FULL_INPUTS = ("p8w",)
#: Surface inputs, ``(ny, nx)``.
KEPS_SURFACE_INPUTS = ("ust", "hfx", "qfx", "br", "znt", "psim", "psih")
#: Outputs, ``(nz, ny, nx)``, in kernel argument order (WRF names).
KEPS_OUTPUTS_3D = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                   "wu_tur", "wv_tur", "wt_tur", "exch_h", "exch_m")
#: The PBL slot's names for the five tendencies.
KEPS_RATE_ALIASES = {"du": "rublten", "dv": "rvblten", "dtheta": "rthblten",
                     "dqv": "rqvblten", "dqc": "rqcblten"}

MODULE_KEY = "woof.core.kernels:keps"
#: Runtime uses the shared fast defaults. GPUWM_WRF_EXACT=1 applies strict
#: options at the shared compiler boundary, including CuPy's final FTZ option.
from woof.core.kernels import module_options
MODULE_OPTIONS = module_options("keps")


def keps_source() -> str:
    """The exact source string the loader hands NVRTC."""
    from woof.core.kernels import module_source
    return module_source("keps")


def load_keps_module(source: str | None = None, *, nz: int | None = None):
    """Use the shared loader for normal calls and the planted control."""
    import cupy as cp
    from woof.core.kernels import load_module, load_module_int_defines
    from woof.wrf_exact import ENABLED
    # A runtime depth is specialized once and cached per device. Keep the
    # original generic image and options for word-exact native replay.
    specialize = nz is not None and not ENABLED
    if source is None:
        if specialize:
            return load_module_int_defines("keps", (("KEPS_NZ", int(nz)),))
        return load_module("keps")
    from woof.certify.kernel_manifest import record_module
    options = MODULE_OPTIONS + ((f"-DKEPS_NZ={int(nz)}",) if specialize else ())
    module = cp.RawModule(code=source, options=options)
    record_module(MODULE_KEY, source=source, options=options, module=module)
    return module


def keps_tile_threads(columns: int, multiprocessors: int) -> int:
    """Threads in flight: enough for every column, at most the resident
    tile the workspace is priced for (physics_inventory)."""
    blocks = (int(columns) + KEPS_BLOCK - 1) // KEPS_BLOCK
    cap = max(1, int(multiprocessors) * KEPS_TILE_BLOCKS_PER_SM)
    return min(blocks, cap) * KEPS_BLOCK


def _check(name, array, shape):
    import cupy as cp
    if (not isinstance(array, cp.ndarray) or array.dtype != np.float32
            or array.shape != shape or not array.flags.c_contiguous):
        raise ValueError(
            f"the K-epsilon PBL requires {name} as a contiguous float32 "
            f"device array of shape {shape}, got "
            f"{getattr(array, 'dtype', type(array))} "
            f"{getattr(array, 'shape', None)}")


def keps_step(inputs, state, *, dt: float, itimestep: int,
              cp_air: float = KEPS_CP, g: float = KEPS_G,
              source: str | None = None) -> dict:
    """Run one ``keps`` call over every column.

    ``inputs`` holds :data:`KEPS_MASS_INPUTS` ``(nz, ny, nx)``, ``p8w``
    ``(nz+1, ny, nx)`` and :data:`KEPS_SURFACE_INPUTS` ``(ny, nx)``, float32,
    WRF's k = 1 first.  ``state`` holds the carried
    :data:`woof.core.physics_inventory.KEPS_STATE_3D` arrays and ``pblh``,
    updated in place.  ``dt`` is the interval the call advances (WRF passes
    the model step, which is the PBL interval at WRF's default bldt = 0) and
    ``itimestep`` WRF's one-based step, whose values 1 and 2 are the
    scheme's reset.

    Returns the WRF-named outputs plus ``du/dv/dtheta/dqv/dqc``, the PBL
    slot's names for the same device arrays.
    """
    import cupy as cp

    nz, ny, nx = inputs["u"].shape
    if outside_vertical_bounds(nz, VERTICAL_LEVEL_BOUNDS):
        raise refuse_vertical_levels(
            "K-epsilon PBL", VERTICAL_LEVEL_BOUNDS, nz,
            breakage=(
                "GET_PBLH's exit safeguard fires at level kte-1 and the "
                "theta-v search starts above the lowest 500 m, so with fewer "
                "levels the PBL-height search has no level to test; select "
                "at least four model levels."))
    shape3, shape2 = (nz, ny, nx), (ny, nx)
    for name in KEPS_MASS_INPUTS:
        _check(name, inputs[name], shape3)
    _check("p8w", inputs["p8w"], (nz + 1, ny, nx))
    for name in KEPS_SURFACE_INPUTS:
        _check(name, inputs[name], shape2)
    for name in KEPS_STATE_3D:
        _check(name, state[name], shape3)
    for name in KEPS_STATE_2D:
        _check(name, state[name], shape2)
    out = {name: cp.empty(shape3, dtype=cp.float32) for name in KEPS_OUTPUTS_3D}
    columns = ny * nx
    device = cp.cuda.Device()
    sms = int(device.attributes["MultiProcessorCount"])
    tile = keps_tile_threads(columns, sms)
    ws = cp.empty(keps_workspace_floats(nz, tile), dtype=cp.float32)
    kernel = load_keps_module(source, nz=nz).get_function("keps_columns")
    kernel((tile // KEPS_BLOCK,), (KEPS_BLOCK,), (
        *(inputs[name] for name in KEPS_MASS_INPUTS), inputs["p8w"],
        *(inputs[name] for name in KEPS_SURFACE_INPUTS),
        *(state[name] for name in KEPS_STATE_3D), state["pblh"],
        *(out[name] for name in KEPS_OUTPUTS_3D),
        ws, np.int32(nz), np.int32(columns), np.float32(dt),
        np.int32(itimestep), np.float32(cp_air), np.float32(g)))
    del ws
    for alias, name in KEPS_RATE_ALIASES.items():
        out[alias] = out[name]
    out["pblh"] = state["pblh"]
    return out


__all__ = ["KEPS_CP", "KEPS_G", "KEPS_FULL_INPUTS", "KEPS_MASS_INPUTS",
           "KEPS_OUTPUTS_3D", "KEPS_RATE_ALIASES", "KEPS_SURFACE_INPUTS",
           "MODULE_KEY", "MODULE_OPTIONS", "VERTICAL_LEVEL_BOUNDS",
           "keps_source", "keps_step", "keps_tile_threads",
           "load_keps_module", "KEPS_DIAGNOSTICS_3D", "KEPS_SLOTS"]
