"""UW moist-turbulence PBL (Bretherton and Park 2009), ``bl_pbl_physics = 9``.

CUDA launcher around ``kernels/uwpbl.cu``, the device transcription of WRF
v4.7.1 ``phys/module_bl_camuwpbl_driver.F`` and the CAM modules it calls
(``module_cam_bl_eddy_diff.F``: compute_eddy_diff, trbintd, sfdiag,
caleddy, exacol, zisocl, compute_cubic; ``module_cam_bl_diffusion_solver.F``:
compute_vdiff and its LU solver; ``module_cam_wv_saturation.F``: the
saturation lookups).  The CPU reference is ``woof.verify.uwpbl_ref``; both
are graded bit for bit against the gfortran oracle in
``tools/uwpbl_wrf471_oracle`` (tests/test_uwpbl_*_wrf471_parity.py).

WHAT THE SCHEME OWNS.  A diagnostic-TKE moist turbulence closure: it finds
convective layers (CLs) from the moist Richardson number, entrains at their
top (and base) with a wstar closure, adds stable turbulent layers, iterates
the eddy diffusivities five times against a provisional implicit diffusion
of (u, v, sl, qt), then solves the implicit diffusion of u, v, dry static
energy and the five CAM constituents (Q, CLDLIQ, CLDICE, NUMLIQ, NUMICE)
with a fully implicit surface stress.  Everything is binary64 (CAM's
``real(r8)``); only the WRF-facing inputs and outputs are float32.

WHAT IT READS FROM THE REST OF THE ENGINE (module_pbl_driver.F:1937-1956):
the surface layer's HFX, QFX and UST; the radiation step's longwave heating
RTHRATENLW and cloud fraction CLDFRA (both held between radiation calls,
exactly as WRF's grid arrays are); the model dt (WRF passes DT, not the
PBL interval, to this scheme); the microphysics' QC, QI and ice number.
Its carried state is the pair of interface diffusivities (WRF's EXCH_M /
EXCH_H, nz+1 levels) and the residual surface stress TAURESX/TAURESY.

WORKSPACE.  CAM's automatic arrays live in a per-column global pool
(``uwpbl_common.cuh`` Ws), so there is no compile-time level ceiling; the
launcher sizes the pool from the level count and processes the domain in
column chunks that fit a VRAM budget.  A pool overflow is detected on the
device and refused here by name rather than returning a corrupted column.
"""

from __future__ import annotations

from woof.core.device_cache import cuda_cache

import cupy as cp
import numpy as np

from woof.core.kernels import load_module
from woof.core.state import DTYPE
from woof.core.uwpbl_constants import ESTBL
from woof.physics_vertical_contract import (
    UWPBL_VERTICAL_LEVEL_BOUNDS, outside_vertical_bounds,
    refuse_vertical_levels)


#: The launcher's own level-count bound; restated for the standalone
#: preparation gate as woof.physics_vertical_contract.
#: UWPBL_VERTICAL_LEVEL_BOUNDS (the two are held equal by
#: tests/test_uwpbl_integration.py).
VERTICAL_LEVEL_BOUNDS = UWPBL_VERTICAL_LEVEL_BOUNDS

# The persistent field tables, the output roster and the workspace sizing
# live in the runtime-free inventory, so the VRAM estimate prices exactly
# what initialize_physics and this launcher allocate.
from woof.core.physics_inventory import (  # noqa: E402
    UWPBL_BLOCK, UWPBL_DIAGNOSTICS_2D, UWPBL_DIAGNOSTICS_FULL, UWPBL_HELD_3D,
    UWPBL_I4_SLOTS_PER_LEVEL, UWPBL_MASS_OUTPUTS, UWPBL_R8_SLOTS_PER_LEVEL,
    UWPBL_STATE_2D, UWPBL_STATE_FULL, UWPBL_SURFACE_OUTPUTS,
    UWPBL_WORKSPACE_BUDGET_BYTES, uwpbl_chunk_columns, uwpbl_workspace_slots)

_TPB = UWPBL_BLOCK

#: Mass-level float32 inputs, in kernel argument order.
UWPBL_MASS_INPUTS = ("u", "v", "th", "rho", "qv", "qc", "qi", "qnc", "qni",
                     "p", "z", "t", "cldfra", "exner", "rthratenlw",
                     "wsedl3d")
#: Interface float32 inputs, (nz+1, ny, nx).
UWPBL_FULL_INPUTS = ("p8w", "z_at_w")
#: Surface float32 inputs, (ny, nx).
UWPBL_SURFACE_INPUTS = ("hfx", "qfx", "ust", "ht")


@cuda_cache(maxsize=None, ready=True)
def _module():
    """The compiled module, with WRF's saturation table uploaded once PER CARD.

    Keyed by card and published with its upload event: the table is the
    module's own global memory, which a module loaded on one card does not
    give another card, and a slab on another stream must not launch before
    the upload lands.

    ``UW_ESTBL`` is gestbl's table as the oracle holds it
    (woof/core/uwpbl_constants.py): no host transcendental builds it.
    """
    module = load_module("uwpbl")
    table = np.asarray(ESTBL, dtype=np.float64)
    cp.ndarray(table.shape, dtype=cp.float64,
               memptr=module.get_global("UW_ESTBL")).set(table)
    return module


def _check(name, array, shape, dtype):
    if (array.shape != shape or array.dtype != dtype
            or not array.flags.c_contiguous):
        raise ValueError(
            f"the UW PBL requires {name} as a contiguous {np.dtype(dtype)} "
            f"array of shape {shape}, got {array.dtype} {array.shape}")


def uwpbl_step(inputs, carried, *, dt: float, itimestep: int,
               budget_bytes: int = UWPBL_WORKSPACE_BUDGET_BYTES,
               workspace=None) -> dict:
    """Run one camuwpbl call over every column and return its outputs.

    ``inputs`` holds :data:`UWPBL_MASS_INPUTS` ``(nz, ny, nx)``,
    :data:`UWPBL_FULL_INPUTS` ``(nz+1, ny, nx)`` and
    :data:`UWPBL_SURFACE_INPUTS` ``(ny, nx)``, float32, WRF's k = 1 first.
    ``carried`` holds ``kvm3d``/``kvh3d`` ``(nz+1, ny, nx)`` and
    ``tauresx2d``/``tauresy2d`` ``(ny, nx)``, updated in place.  ``dt`` is
    the MODEL time step (module_pbl_driver.F:1937 passes DT=dt, not DTBL)
    and ``itimestep`` WRF's one-based step, whose value 1 is the scheme's
    first-step reset.

    Returns the WRF names plus the ``du/dv/dtheta/dqv/dqc/dqi/dqni`` aliases
    the PBL slot's coupling reads -- the same device arrays under both keys.
    """
    nz, ny, nx = inputs["u"].shape
    if outside_vertical_bounds(nz, VERTICAL_LEVEL_BOUNDS):
        raise refuse_vertical_levels(
            "UW PBL", VERTICAL_LEVEL_BOUNDS, nz,
            breakage=(
                "the implicit diffusion needs at least one interior "
                "interface (compute_vdiff's k = 2..pver rows) and the "
                "scheme's Richardson numbers live on interfaces above the "
                "surface; select at least two model levels."))
    shape = (ny, nx)
    for name in UWPBL_MASS_INPUTS:
        _check(name, inputs[name], (nz, ny, nx), DTYPE)
    for name in UWPBL_FULL_INPUTS:
        _check(name, inputs[name], (nz + 1, ny, nx), DTYPE)
    for name in UWPBL_SURFACE_INPUTS:
        _check(name, inputs[name], shape, DTYPE)
    for name in ("kvm3d", "kvh3d"):
        _check(name, carried[name], (nz + 1, ny, nx), DTYPE)
    for name in ("tauresx2d", "tauresy2d"):
        _check(name, carried[name], shape, DTYPE)

    def allocate(name, shape, dtype=DTYPE, *, clear=False):
        if workspace is not None:
            return workspace.array('uwpbl/' + name, shape, dtype, clear=clear)
        return (cp.zeros if clear else cp.empty)(shape, dtype=dtype)

    out = {name: allocate(name, (nz, ny, nx))
           for name in UWPBL_MASS_OUTPUTS}
    for name in UWPBL_DIAGNOSTICS_FULL:
        out[name] = allocate(name, (nz + 1, ny, nx))
    for name in UWPBL_SURFACE_OUTPUTS:
        out[name] = allocate(name, shape)
    out["kpbl2d"] = allocate('kpbl2d', shape, cp.int32)

    ncols = ny * nx
    chunk = uwpbl_chunk_columns(nz, ncols, budget_bytes)
    r8cap, i4cap = uwpbl_workspace_slots(nz)
    r8pool = allocate('r8pool', r8cap * chunk, cp.float64)
    i4pool = allocate('i4pool', i4cap * chunk, cp.int32)
    err = allocate('status', 1, cp.int32, clear=True)
    kernel = _module().get_function("uwpbl_columns")
    args_fixed = (
        tuple(inputs[name] for name in UWPBL_MASS_INPUTS)
        + tuple(inputs[name] for name in UWPBL_FULL_INPUTS)
        + tuple(inputs[name] for name in UWPBL_SURFACE_INPUTS)
        + (carried["kvm3d"], carried["kvh3d"],
           carried["tauresx2d"], carried["tauresy2d"])
        + tuple(out[name] for name in UWPBL_MASS_OUTPUTS)
        + tuple(out[name] for name in UWPBL_DIAGNOSTICS_FULL)
        + tuple(out[name] for name in UWPBL_SURFACE_OUTPUTS)
        + (out["kpbl2d"],))
    for col0 in range(0, ncols, chunk):
        n = min(chunk, ncols - col0)
        blocks = (n + _TPB - 1) // _TPB
        kernel((blocks,), (_TPB,),
               args_fixed + (np.int64(ncols), np.int64(col0), np.int32(n),
                             np.int32(chunk), np.int32(nz),
                             np.float32(dt), np.int32(itimestep),
                             r8pool, i4pool, np.int32(r8cap),
                             np.int32(i4cap), err))
    def describe(status):
        # uwpbl_driver.cuh sets 2 where camuwpbl would call endrun on a
        # non-empty compute_vdiff errstring (module_cam_bl_diffusion_solver.F
        # :300-302); the driver's fixed field list cannot reach it.
        if status & 2:
            raise RuntimeError(
                "the UW PBL's diffusion solver refused its field list "
                "('diffusion_solver.compute_vdiff: must diffuse s if diffusing "
                "u or v'), which WRF treats as fatal")
        raise RuntimeError(
            "the UW PBL column workspace overflowed its pool "
            f"({r8cap} binary64 / {i4cap} int32 slots per column at "
            f"nz={nz}); raise UWPBL_R8_SLOTS_PER_LEVEL or "
            "UWPBL_I4_SLOTS_PER_LEVEL in woof/core/physics_inventory.py: "
            "the overflowing columns' outputs are not trustworthy")
    from woof.core.health_ledger import read_status
    read_status(err, site='uwpbl/column', describe=describe)
    out.update({"du": out["rublten"], "dv": out["rvblten"],
                "dtheta": out["rthblten"], "dqv": out["rqvblten"],
                "dqc": out["rqcblten"], "dqi": out["rqiblten"],
                "dqni": out["rqniblten"], "pblh": out["pblh2d"],
                "kpbl": out["kpbl2d"]})
    return out


def uwpbl_ice_number_name(cfg) -> str | None:
    """WOOF's name for WRF's ``P_QNI`` under ``cfg``, or None.

    ``woof.config.UW_PBL_ICE_NUMBER_SPECIES`` is the table and cites the
    Registry package lines.  None means WRF hands the scheme its zero
    dummy slot (``scalar(:,:,:,1)``) and discards RQNIBLTEN.
    """
    from woof.config import UW_PBL_ICE_NUMBER_SPECIES
    return UW_PBL_ICE_NUMBER_SPECIES.get(int(cfg.mp_physics))


def uwpbl_radiation_cloud_fraction(atmosphere, cfg, out, *, state=None) -> None:
    """Write the radiation step's CLDFRA into ``out`` (nz, ny, nx), in place.

    WRF's radiation driver zeroes CLDFRA, then with ``icloud == 1`` and
    ``F_QC .OR. F_QI`` calls ``cal_cldfra1`` on the step's moist fields,
    ``t_phy`` and the hydrostatic ``p_hyd`` (module_radiation_driver.F:
    1309-1332; module_first_rk_step_part1.F:280 and :288 bind P=p_hyd,
    T=t_phy), and the array holds until the next due radiation step.  The
    MYNN boundary-layer-cloud and CAMMGMP overrides at :1402-1446 need
    their own schemes and cannot be active beside this PBL.  The Xu-Randall
    arithmetic and the F_QI/F_QS arm selection are the SAME functions the
    engine's radiation adapters call (woof.core.rrtmgp.cal_cldfra1,
    woof.core.rrtmg_legacy.legacy_cloud_fraction_flags), so the PBL sees
    the fraction the radiation used.
    """
    out[...] = 0.0
    if int(getattr(cfg, "icloud", 1)) != 1 or not cfg.moist:
        return
    if int(cfg.mp_physics) == 0:
        # passiveqv (Registry.EM_COMMON:3014) allocates qv alone, so F_QC
        # and F_QI are both false and cal_cldfra1 is never called.
        return
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
    from woof.core.rrtmgp import cal_cldfra1
    f_qi, f_qs = legacy_cloud_fraction_flags(int(cfg.mp_physics))
    nz = out.shape[0]
    cols = out.size // nz

    def flat(name):
        return cp.ascontiguousarray(atmosphere[name]).reshape(nz, cols)

    cldfra = cal_cldfra1(flat("qv"), flat("qc"), flat("qi"), flat("qs"),
                         flat("temperature"), flat("pressure"),
                         f_qc=True, f_qi=f_qi, f_qs=f_qs,
                         mp_physics=int(cfg.mp_physics),
                         f_ice_phy=cp.ascontiguousarray(state.f_ice_phy).reshape(nz, cols)
                         if int(cfg.mp_physics)==95 else None)
    out[...] = cldfra.reshape(out.shape)
