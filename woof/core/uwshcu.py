"""UW shallow cumulus (Park and Bretherton 2009), ``shcu_physics = 2``.

CUDA launcher around ``kernels/uwshcu.cu``, the device transcription of
WRF v4.6.1 ``phys/module_shcu_camuwshcu_driver.F`` (camuwshcu_driver) and
``phys/module_shcu_camuwshcu.F`` (compute_uwshcu_inv, compute_uwshcu and
its private routines, findsp included), with the CAM support layer it
calls: ``module_cam_wv_saturation.F`` (estblf, fqsatd, shared with the UW
PBL port) and ``module_cam_error_function.F`` (erfc). Generated stock-module
diagnostics are described in ``tools/cam5_wrf471_oracle/README.md``. The launcher is connected to model dispatch, state IO and restart. Real
active-column native replay results and measured performance limits are
recorded in ``docs/public/cam5-suite.md``.

WHAT THE SCHEME OWNS.  A single bulk entraining-detraining plume per
column: source air from the PBL (lowest-layer qt, minimum thvl), a cloud-
base mass flux from the PBL-mean TKE and the convective inhibition (an
implicit-CIN second pass blends the CIN of the column the first pass's
tendencies would leave), buoyancy-sorting lateral mixing, penetrative
entrainment above the level of neutral buoyancy, precipitation and its
evaporation, detrained condensate, and compensating subsidence.  The
column is binary64 (CAM's ``real(r8)``); the WRF-facing fields are
float32 with the driver's own float32 conversions.

WHAT IT READS FROM THE REST OF THE ENGINE (module_shallowcu_driver.F:
517-553, module_first_rk_step_part1.F:1569-1650): the hydrostatic pressure
``p_hyd``/``p_hyd_w``, ``pi_phy``, phy_prep's heights, the moist species
including the liquid and ice number when the microphysics carries them,
the PBL's ``PBLH`` and ``TKE_PBL`` (interfaces, nz+1), terrain height, and
its own carried ``CUSH`` (convective scale height).  The cloud fractions
the driver passes are copied into compute_uwshcu and never read.

WORKSPACE.  CAM's automatic arrays live in a per-column global pool
(uwpbl_common.cuh ``Ws``), sized from the level count, processed in
column chunks that fit a VRAM budget; an overflow is detected on the
device and refused here by name.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from woof.core.device_cache import cuda_cache
from woof.core.kernels import load_module
from woof.core.state import DTYPE
from woof.core.cam5_support import upload_saturation_table
from woof.core.physics_inventory import (
    UWSHCU_BLOCK, UWSHCU_FULL_INPUTS, UWSHCU_FULL_OUTPUTS,
    UWSHCU_MASS_INPUTS, UWSHCU_MASS_OUTPUTS, UWSHCU_SURFACE_INPUTS,
    UWSHCU_SURFACE_OUTPUTS, UWSHCU_WORKSPACE_BUDGET_BYTES,
    uwshcu_chunk_columns, uwshcu_workspace_slots)

#: compute_uwshcu needs a PBL-top interface above the first and an updraft
#: layer above it (kinv >= 2 and the k = krel .. mkx-1 loop); WRF's own
#: smallest grids are far above this.
UWSHCU_MIN_LEVELS = 3


@cuda_cache(maxsize=None, ready=True)
def _module():
    """The compiled module, with the saturation table uploaded once PER CARD.

    The table is WRF's gestbl estbl as the UW PBL oracle holds it
    (woof/core/uwpbl_constants.py; WRF v4.6.1's module_cam_gffgch.F,
    module_cam_wv_saturation.F and module_cam_esinti.F are byte-identical
    to the v4.7.1 files that oracle pinned).  The UW shallow cumulus oracle
    records its own host's table and tests/test_uwshcu_wrf461_parity.py
    grades this upload against it word for word.
    """
    module = load_module("uwshcu")
    upload_saturation_table(module)
    return module


_ERRORS = {
    1: ("the UW shallow cumulus column workspace overflowed its pool; raise "
        "UWSHCU_R8_SLOTS_PER_LEVEL in woof/core/physics_inventory.py: the "
        "overflowing columns' outputs are not trustworthy"),
    2: ("the UW shallow cumulus hit WRF's 'Critical error in mu calculation "
        "in UW_ShCu' (module_shcu_camuwshcu.F:2172-2177), which WRF treats "
        "as fatal: the cloud-base closure has no root above the TKE-limited "
        "minimum"),
    4: ("the UW shallow cumulus hit WRF's findsp fatal (module_shcu_camuwshcu"
        ".F:5486/5512: the wet-bulb iteration did not converge or did not "
        "conserve enthalpy), which WRF treats as fatal"),
}


def _raise_status(status: int) -> None:
    if status:
        raise RuntimeError("; ".join(msg for bit, msg in _ERRORS.items()
                                     if status & bit))


def uwshcu_core(columns: dict, *, dt: float, module=None) -> dict:
    """compute_uwshcu_inv over ``ncol`` columns of binary64 CAM inputs.

    ``columns`` holds ``ps0, zs0, tke`` shaped ``(nk+1, ncol)`` and ``p0,
    z0, dp0, u0, v0, qv0, ql0, qi0, t0, trl, tri`` shaped ``(nk, ncol)``
    (row 0 = model top, compute_uwshcu_inv's own orientation; ``trl``/
    ``tri`` are the NUMLIQ/NUMICE constituents), plus ``pblh`` and ``cush``
    ``(ncol,)``.  Returns every compute_uwshcu_inv output in the same
    orientation; ``cush`` is returned updated.  Used by the parity tests;
    the forecast path is :func:`uwshcu_step`.
    """
    nk, ncol = columns["p0"].shape
    f8 = cp.float64
    cin = {n: cp.ascontiguousarray(cp.asarray(columns[n], dtype=f8))
           for n in ("ps0", "zs0", "p0", "z0", "dp0", "u0", "v0", "qv0",
                     "ql0", "qi0", "t0", "trl", "tri", "tke", "pblh")}
    cush = cp.array(columns["cush"], dtype=f8)
    full = ("umf", "slflx", "qtflx", "flxprc1", "flxsnow1")
    mass = ("qvten", "qlten", "qiten", "sten", "uten", "vten", "qrten",
            "qsten", "evapc", "cufrc", "qcu", "qlu", "qiu", "qc", "trtl",
            "trti")
    sfc = ("precip", "snow", "cbmf", "rliq", "cnt", "cnb")
    out = {n: cp.zeros((nk + 1, ncol), f8) for n in full}
    out.update({n: cp.zeros((nk, ncol), f8) for n in mass})
    out.update({n: cp.zeros(ncol, f8) for n in sfc})
    r8cap, i4cap = uwshcu_workspace_slots(nk)
    r8pool = cp.empty(r8cap * ncol, dtype=f8)
    i4pool = cp.empty(max(i4cap, 1) * ncol, dtype=cp.int32)
    err = cp.zeros(1, dtype=cp.int32)
    kernel = (module or _module()).get_function("uwshcu_core_columns")
    blocks = (ncol + UWSHCU_BLOCK - 1) // UWSHCU_BLOCK
    kernel((blocks,), (UWSHCU_BLOCK,),
           tuple(cin[n] for n in ("ps0", "zs0", "p0", "z0", "dp0", "u0",
                                  "v0", "qv0", "ql0", "qi0", "t0", "trl",
                                  "tri", "tke", "pblh"))
           + (cush, np.float64(dt))
           + tuple(out[n] for n in full + mass + sfc)
           + (np.int64(ncol), np.int32(nk), r8pool, i4pool,
              np.int32(r8cap), np.int32(i4cap), err))
    _raise_status(int(err.get()[0]))
    out["cush"] = cush
    return out


def uwshcu_step(inputs: dict, carried: dict, *, dt: float, has_qnc: bool,
                has_qni: bool,
                budget_bytes: int = UWSHCU_WORKSPACE_BUDGET_BYTES, workspace=None) -> dict:
    """Run one camuwshcu_driver call over every column.

    ``inputs`` holds :data:`UWSHCU_MASS_INPUTS` ``(nz, ny, nx)``,
    :data:`UWSHCU_FULL_INPUTS` ``(nz+1, ny, nx)`` and
    :data:`UWSHCU_SURFACE_INPUTS` ``(ny, nx)``, float32, WRF's k = 1 first.
    ``carried`` holds ``cush`` ``(ny, nx)``, updated in place.  ``qnc``/
    ``qni`` are WRF's zero dummy slot where the microphysics does not carry
    the species, and ``has_qnc``/``has_qni`` say whether WRF writes
    RQCNSHTEN/RQINSHTEN (``p_qnc``/``p_qni >= param_first_scalar``,
    driver lines 425-432); where it does not, those outputs are zero.

    Returns :data:`UWSHCU_MASS_OUTPUTS`, :data:`UWSHCU_FULL_OUTPUTS` and
    :data:`UWSHCU_SURFACE_OUTPUTS` under WRF's names.
    """
    nz, ny, nx = inputs["u"].shape
    if nz < UWSHCU_MIN_LEVELS:
        raise ValueError(
            f"the UW shallow cumulus needs at least {UWSHCU_MIN_LEVELS} "
            f"model levels (a PBL-top interface above the first and an "
            f"updraft layer above it), got {nz}")
    shape = (ny, nx)
    for name in UWSHCU_MASS_INPUTS:
        _check(name, inputs[name], (nz, ny, nx))
    for name in UWSHCU_FULL_INPUTS:
        _check(name, inputs[name], (nz + 1, ny, nx))
    for name in UWSHCU_SURFACE_INPUTS:
        _check(name, inputs[name], shape)
    _check("cush", carried["cush"], shape)
    if workspace is None:
        from woof.core.cam5_workspace import CamWorkspace
        workspace = CamWorkspace()
    out = {name: workspace.array("shcu2/" + name, (nz, ny, nx), DTYPE, clear=True)
           for name in UWSHCU_MASS_OUTPUTS}
    out.update({name: workspace.array("shcu2/" + name, (nz + 1, ny, nx), DTYPE, clear=True)
                for name in UWSHCU_FULL_OUTPUTS})
    out.update({name: workspace.array("shcu2/" + name, shape, DTYPE, clear=True)
                for name in UWSHCU_SURFACE_OUTPUTS})
    ncols = ny * nx
    chunk = uwshcu_chunk_columns(nz, ncols, budget_bytes)
    r8cap, i4cap = uwshcu_workspace_slots(nz)
    r8pool = workspace.array("shcu2/r8", r8cap * chunk, cp.float64)
    i4pool = workspace.array("shcu2/i4", max(i4cap, 1) * chunk, cp.int32)
    err = workspace.array("shcu2/status", 1, cp.int32, clear=True)
    kernel = _module().get_function("uwshcu_driver_columns")
    args_fixed = (
        tuple(inputs[n] for n in UWSHCU_MASS_INPUTS)
        + tuple(inputs[n] for n in UWSHCU_FULL_INPUTS)
        + tuple(inputs[n] for n in UWSHCU_SURFACE_INPUTS)
        + (carried["cush"],)
        + tuple(out[n] for n in UWSHCU_MASS_OUTPUTS)
        + tuple(out[n] for n in UWSHCU_FULL_OUTPUTS)
        + tuple(out[n] for n in UWSHCU_SURFACE_OUTPUTS))
    for col0 in range(0, ncols, chunk):
        n = min(chunk, ncols - col0)
        blocks = (n + UWSHCU_BLOCK - 1) // UWSHCU_BLOCK
        kernel((blocks,), (UWSHCU_BLOCK,),
               args_fixed + (np.int64(ncols), np.int64(col0), np.int32(n),
                             np.int32(chunk), np.int32(nz), np.float32(dt),
                             np.int32(bool(has_qnc)), np.int32(bool(has_qni)),
                             r8pool, i4pool, np.int32(r8cap),
                             np.int32(i4cap), err))
    from woof.core.health_ledger import read_status
    read_status(err, site="cam5/shcu2", describe=_raise_status)
    return out


def _check(name, array, shape):
    if (array.shape != shape or array.dtype != DTYPE
            or not array.flags.c_contiguous):
        raise ValueError(
            f"the UW shallow cumulus requires {name} as a contiguous "
            f"{np.dtype(DTYPE)} array of shape {shape}, got {array.dtype} "
            f"{array.shape}")
