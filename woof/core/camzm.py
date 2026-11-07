"""CAM5 Zhang-McFarlane deep convection, ``cu_physics = 7`` (CAMZMSCHEME).

CUDA launcher around ``kernels/camzm.cu``, the device transcription of WRF
v4.6.1 ``phys/module_cu_camzm_driver.F`` (camzm_driver, get_tpert,
zm_conv_init) and ``phys/module_cu_camzm.F`` (zm_convr with its dilute
entraining-plume CAPE, cloud model and CAPE closure, zm_conv_evap, momtran,
convtran). Generated stock-module diagnostics are described in
``tools/cam5_wrf471_oracle/README.md``. This recovered launcher is not
connected to model dispatch and has no real-input replay qualification.

WHAT THE SCHEME OWNS.  Deep convection only: a bulk updraft/downdraft pair
launched from the level of maximum moist static energy at or below the PBL
top, triggered when the dilute-parcel CAPE exceeds 70 J/kg, closed by
removing the CAPE over tau = max(600 s, 3600 s * min(1, dx/275 km)); the
rain it makes evaporates on the way down (Sundqvist), it carries momentum
and cloud liquid/ice, and the detrained condensate is split into liquid and
ice with number tendencies.  Its outputs are WRF's: RTHCUTEN, RQVCUTEN,
RQCCUTEN, RQICUTEN, RQCNCUTEN, RQINCUTEN, RUCUTEN, RVCUTEN (mass points),
RAINCV/PRATEC, CAPE, the cloud top/base and the CAM diagnostics.

WHAT IT READS FROM THE REST OF THE ENGINE (module_cumulus_driver.F:1340-
1382): the state (TH, T, P, PI, QV, QC, QI, U, V on mass points, Z, Z_AT_W,
DZ8W, P8W, HT, PSFC), the radiation step's CLDFRA, and get_tpert's
convective temperature excess.  WRF gives get_tpert two arms and no other:
the UW PBL's own TPERT2D (``bl_pbl_physics = 9``), or one computed from
the MYJ PBL's TKE_PBL/KPBL/PBLH and the MYJ surface layer's TSK/UST/HFX/
QFX/MAVAIL (``bl_pbl_physics = 2`` with ``sf_sfclay_physics = 2``); any other
PBL is a WRF fatal error, and this launcher refuses it by the same name.

CARRIED STATE.  None inside the scheme: every output is recomputed on each
due call (camzm_driver assigns every one), so a restart needs only what the
cumulus slot already holds between due calls (the tendencies and PRATEC,
module_cumulus_driver.F's persistent arrays).

WORKSPACE.  CAM's automatic arrays live in a per-column global pool
(uwpbl_common.cuh Ws), so there is no compile-time level ceiling; the
launcher sizes the pool from the level count and processes the domain in
column chunks that fit a VRAM budget.  A pool overflow is detected on the
device and refused here by name rather than returning a corrupted column.

ARITHMETIC.  Binary64 like CAM (real(r8)), float32 where camzm_driver works
in WRF's default REAL, every operation rounding-pinned and every
transcendental WOOF's own libm word, so the default and the strict
(``GPUWM_WRF_EXACT=1``) compiles give the same words.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from woof.core.device_cache import cuda_cache
from woof.core.kernels import load_module

#: Kernel input table, in kernels/camzm.cu's CZ_IN_* order.
from woof.core.cam5_memory import CAMZM_MASS_INPUTS
from woof.core.cam5_memory import CAMZM_FULL_INPUTS
from woof.core.cam5_memory import CAMZM_SURFACE_INPUTS
CAMZM_INPUTS = CAMZM_MASS_INPUTS + CAMZM_FULL_INPUTS + CAMZM_SURFACE_INPUTS

#: Kernel output table, in kernels/camzm.cu's CZ_OUT_* order.
from woof.core.cam5_memory import CAMZM_MASS_OUTPUTS
from woof.core.cam5_memory import CAMZM_FULL_OUTPUTS
from woof.core.cam5_memory import CAMZM_SURFACE_OUTPUTS
CAMZM_OUTPUTS = CAMZM_MASS_OUTPUTS + CAMZM_FULL_OUTPUTS + CAMZM_SURFACE_OUTPUTS
from woof.core.cam5_memory import CAMZM_INT_OUTPUTS

#: What a forecast step needs (the tendencies WRF's physics_addtendc applies
#: for CAMZMSCHEME, :1485-1533, and the rain and cloud-top words the cumulus
#: slot keeps).  RTHCUTEN and RQVCUTEN are always computed: the kernel
#: accumulates them in their own output words, as camzm_driver does.
from woof.core.cam5_memory import CAMZM_FORECAST_OUTPUTS

#: The two get_tpert arms (module_cu_camzm_driver.F:927-1020).
MYJ_PBL, MYJ_SFCLAY, UW_PBL = 2, 2, 9

from woof.core.cam5_memory import CAMZM_BLOCK
#: binary64 workspace slots per column per (level+1): the driver's arrays
#: (47) + zm_convr's (27) + the deepest callee (cldprp, 17) with margin.
from woof.core.cam5_memory import CAMZM_R8_SLOTS_PER_LEVEL
from woof.core.cam5_memory import CAMZM_WORKSPACE_BUDGET_BYTES

_STATUS = {
    1: ("the ZM column workspace overflowed its pool; raise "
        "CAMZM_R8_SLOTS_PER_LEVEL in woof/core/camzm.py: the overflowing "
        "columns' outputs are not trustworthy"),
    3: ("ZM_CONV IENTROPY: Tmix did not converge (module_cu_camzm.F:3798, "
        "a WRF endrun): the entraining parcel's entropy could not be "
        "inverted for a temperature in 100 Newton steps, so the column has "
        "no defined CAPE"),
    4: ("CAMZMSCHEME needs a PBL height from a PBL scheme "
        "(module_cu_camzm_driver.F:948, a WRF fatal): get_tpert's MYJ arm "
        "read PBLH <= 0"),
    5: ("CAMZMSCHEME requires MYJPBLSCHEME with MYJSFCSCHEME, or "
        "CAMUWPBLSCHEME (module_cu_camzm_driver.F:978-1018, a WRF fatal)"),
}


def camzm_workspace_slots(nz: int) -> int:
    """binary64 pool slots one column needs at ``nz`` mass levels."""
    return CAMZM_R8_SLOTS_PER_LEVEL * (int(nz) + 1)


def camzm_chunk_columns(nz: int, ncols: int,
                        budget_bytes: int = CAMZM_WORKSPACE_BUDGET_BYTES) -> int:
    """Columns per launch so the pool fits ``budget_bytes`` (at least one)."""
    from woof.core.cam5_memory import chunk_columns
    return chunk_columns(nz,ncols,CAMZM_R8_SLOTS_PER_LEVEL,budget_bytes,CAMZM_BLOCK)


from woof.core.camzm_contract import pbl_arm_refusal


@cuda_cache(maxsize=None, ready=True)
def _module():
    """The compiled module, with wv_saturation's table uploaded once PER CARD."""
    from woof.core.cam5_support import upload_saturation_table
    module = load_module("camzm")
    upload_saturation_table(module)
    return module


def _check(name, array, shape, dtype):
    if (array.shape != shape or array.dtype != dtype
            or not array.flags.c_contiguous):
        raise ValueError(
            f"CAM Zhang-McFarlane requires {name} as a contiguous "
            f"{np.dtype(dtype)} array of shape {shape}, got "
            f"{array.dtype} {array.shape}")


def camzm_step(inputs, *, dt: float, dx: float, cudt: float, itimestep: int,
               bl_pbl_physics: int, sf_sfclay_physics: int,
               diagnostics: bool = False, outputs=None,
               budget_bytes: int = CAMZM_WORKSPACE_BUDGET_BYTES, workspace=None) -> dict:
    """Run one due camzm_driver call over every column and return its outputs.

    ``inputs`` holds :data:`CAMZM_MASS_INPUTS` ``(nz, ny, nx)``,
    :data:`CAMZM_FULL_INPUTS` ``(nz+1, ny, nx)``, :data:`CAMZM_SURFACE_INPUTS`
    ``(ny, nx)`` float32 and ``kpbl`` ``(ny, nx)`` int32, WRF's k = 1 (the
    surface) first.  ``dt`` is the model step, ``cudt`` WRF's cumulus
    interval in minutes (0 = every step; the scheme's own time scale is
    cudt*60 when nonzero), ``itimestep`` WRF's one-based step (1 zeroes the
    cloud fraction, :425).  Arrays the selected PBL arm does not read
    (``tpert`` under MYJ; the MYJ fields under the UW PBL) may hold anything.

    Returns float32 arrays under WRF's names: :data:`CAMZM_FORECAST_OUTPUTS`
    always, every name in :data:`CAMZM_OUTPUTS`/:data:`CAMZM_INT_OUTPUTS`
    with ``diagnostics=True``, or the names in ``outputs``.
    """
    why = pbl_arm_refusal(bl_pbl_physics, sf_sfclay_physics)
    if why is not None:
        raise ValueError(why)
    nz, ny, nx = inputs["th"].shape
    if nz < 4:
        # The breakage this prevents: zm_convr starts its level loops at
        # msg + 2 = 2 with limcnv = 2 and cldprp places the cloud top at
        # limcnv + 1 = 3 or higher; fewer than four layers leave no column a
        # plume can be launched, entrain and detrain in, and k-1/k+1 reads
        # run off the column.
        raise ValueError(
            "CAM Zhang-McFarlane needs at least four model levels "
            f"(limcnv = 2 fixes the highest cloud top at level 3); got {nz}")
    shape = (ny, nx)
    for name in CAMZM_MASS_INPUTS:
        _check(name, inputs[name], (nz, ny, nx), np.float32)
    for name in CAMZM_FULL_INPUTS:
        _check(name, inputs[name], (nz + 1, ny, nx), np.float32)
    for name in CAMZM_SURFACE_INPUTS:
        _check(name, inputs[name], shape, np.float32)
    _check("kpbl", inputs["kpbl"], shape, np.int32)

    if diagnostics:
        want = set(CAMZM_OUTPUTS) | set(CAMZM_INT_OUTPUTS)
    else:
        want = set(CAMZM_FORECAST_OUTPUTS) | set(outputs or ())
    want |= {"rthcuten", "rqvcuten"}
    unknown = want - set(CAMZM_OUTPUTS) - set(CAMZM_INT_OUTPUTS)
    if unknown:
        raise ValueError(f"unknown CAM Zhang-McFarlane outputs {sorted(unknown)}")
    if workspace is None:
        from woof.core.cam5_workspace import CamWorkspace
        workspace = CamWorkspace()
    out = {}
    for name in CAMZM_MASS_OUTPUTS:
        if name in want:
            out[name] = workspace.array("cu7/" + name, (nz, ny, nx), np.float32)
    for name in CAMZM_FULL_OUTPUTS:
        if name in want:
            out[name] = workspace.array("cu7/" + name, (nz + 1, ny, nx), np.float32)
    for name in CAMZM_SURFACE_OUTPUTS:
        if name in want:
            out[name] = workspace.array("cu7/" + name, shape, np.float32)
    for name in CAMZM_INT_OUTPUTS:
        if name in want:
            out[name] = workspace.array("cu7/" + name, shape, np.int32)

    in_ptrs = workspace.pointers("cu7/inputs", (inputs[n] for n in CAMZM_INPUTS))
    out_ptrs = workspace.pointers("cu7/outputs", (out.get(n) for n in CAMZM_OUTPUTS))
    iout_ptrs = workspace.pointers("cu7/ints", (out.get(n) for n in CAMZM_INT_OUTPUTS))
    ncols = ny * nx
    chunk = camzm_chunk_columns(nz, ncols, budget_bytes)
    r8cap = camzm_workspace_slots(nz)
    r8pool = workspace.array("cu7/r8", r8cap * chunk, cp.float64)
    err = workspace.array("cu7/status", 2, cp.int32, clear=True)
    err[1:].fill(-1)
    kernel = _module().get_function("camzm_columns")
    for col0 in range(0, ncols, chunk):
        n = min(chunk, ncols - col0)
        blocks = (n + CAMZM_BLOCK - 1) // CAMZM_BLOCK
        kernel((blocks,), (CAMZM_BLOCK,),
               (in_ptrs, inputs["kpbl"], out_ptrs, iout_ptrs,
                np.int64(ncols), np.int64(col0), np.int32(n), np.int32(chunk),
                np.int32(nz), np.float32(dt), np.float32(dx),
                np.float32(cudt), np.int32(itimestep),
                np.int32(bl_pbl_physics), np.int32(sf_sfclay_physics),
                r8pool, np.int32(r8cap), err))
    from woof.core.health_ledger import read_status
    def describe(status):
        raise RuntimeError(_STATUS.get(status, f"CAMZM device status {status}"))
    read_status(err, site="cam5/cu7", describe=describe)
    return out
