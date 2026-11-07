"""CAM5.1 Morrison-Gettelman two-moment microphysics, ``mp_physics = 11``.

CUDA launcher around ``kernels/cammgmp.cu``, the device transcription of WRF
v4.6.1 ``phys/module_mp_cammgmp_driver.F`` (CAMMGMP with its simple
macrophysics) and the CAM modules it calls: ``module_cam_mp_cldwat2m_micro.F``
(MG1 mmicro_pcond and its GAMMA), ``module_cam_mp_microp_aero.F``
(microp_aero_ts, nucleati), ``module_cam_mp_ndrop.F`` (dropmixnuc,
activate_modal), ``module_cam_wv_saturation.F`` (polysvp, estblf,
vqsatd_water) and ``module_cam_error_function.F``.  The synthetic column oracle in
``tools/cammgmp_wrf461_oracle`` is diagnostic only. Real forecast capture
and runtime integration remain unqualified. Stock WRF indexing is the
default; the alternate ndrop read is a private attribution instrument.

WHAT THE SCHEME OWNS.  Everything is binary64 inside (CAM's ``real(r8)``);
the WRF-facing fields are float32.  Prognostic: theta, qv, cloud liquid and
ice mass, droplet and ice number.  Rain and snow are DIAGNOSTIC in MG1:
QRAIN/QSNOW/QNRAIN/QNSNOW are rewritten every call from the scheme's own
precipitation fluxes and are never read.  Carried between calls, and
restart state in WRF (Registry/registry.cam:168-177, all 'r'):
CLDFRA_MP_ALL (the macrophysics' previous cloud fraction), CLDFRA_OLD_MP
(the microphysics' previous stratus fraction), and the diagnostic outputs
RH_OLD_MP, LCD_OLD_MP, CLDFRA_MP, CLDFRAI, CLDFRAL, CLDFRA_CONV, LRADIUS,
IRADIUS, WSEDL3D.  WRF's itimestep == 1 zeroes the carried fractions
(module_mp_cammgmp_driver.F:607-621).

WHAT IT READS FROM THE REST OF THE ENGINE (module_microphysics_driver.F:
2686-2716, bound in solve_em.F): the step-start ``t_phy`` and hydrostatic
``p_hyd``/``p_hyd_w`` of the physics prep, the post-dynamics theta, pi_phy,
z and alt, the boundary layer's TKE_PBL and EXCH_H on interfaces, XLAND and
SNOW, the namelist's prescribed aerosol (accum_mode, aitken_mode,
coarse_mode), and -- only under cu_physics = 7 / shcu_physics = 2 -- the CAM
ZM and UW shallow convection's mass fluxes, in-cloud water and shallow
fraction.  Under any other convection WRF zeroes those inputs, as here.

WORKSPACE.  CAM's automatic arrays live in a per-column global pool (the UW
PBL's ``Ws``), sized from the level count; the domain runs in column chunks
that fit a VRAM budget.  Status words from the device are refused by name.
"""

from __future__ import annotations

from woof.core.device_cache import cuda_cache

import cupy as cp
import numpy as np

from woof.core.kernels import load_module, load_module_int_defines
from woof.core.state import DTYPE
from woof.core.uwpbl_constants import ESTBL

#: Threads per block (one column per thread).
from woof.core.cam5_memory import CAMMGMP_BLOCK
#: binary64 workspace slots per level of one column: the driver's ~62
#: column arrays plus mmicro_pcond's ~116 (the deeper of the two nested
#: callees), with headroom; the pool overflow word refuses an undersized
#: pool rather than returning a corrupted column.
from woof.core.cam5_memory import CAMMGMP_R8_SLOTS_PER_LEVEL
#: Device bytes one call may spend on the pools.
from woof.core.cam5_memory import CAMMGMP_WORKSPACE_BUDGET_BYTES
#: The scheme's own level bounds: dropmixnuc forms zs(pver-1) from two
#: layers, so one layer is meaningless; 512 bounds the pool, not physics.
VERTICAL_LEVEL_BOUNDS = (2, 512)

#: Mass-level float32 inputs (nz, ny, nx), kernel order.
from woof.core.cam5_memory import CAMMGMP_MASS_INPUTS
#: Interface float32 inputs (nz+1, ny, nx), kernel order.
from woof.core.cam5_memory import CAMMGMP_FULL_INPUTS
#: Surface float32 inputs (ny, nx).
from woof.core.cam5_memory import CAMMGMP_SURFACE_INPUTS
#: In-out mass-level fields (nz, ny, nx), kernel order.
from woof.core.cam5_memory import CAMMGMP_INOUT_3D
#: In-out surface fields (ny, nx), kernel order.
from woof.core.cam5_memory import CAMMGMP_INOUT_2D

_STATUS = {
    1: ("the CAMMGMP column workspace overflowed its pool; raise "
        "CAMMGMP_R8_SLOTS_PER_LEVEL in woof/core/cammgmp.py: the "
        "overflowing columns' outputs are not trustworthy"),
    2: ("MG's GAMMA (module_cam_mp_cldwat2m_micro.F:3557) was asked for a "
        "negative non-integer argument, whose reflection branch needs SIN; "
        "MG only evaluates GAMMA at pgam + 1..7 with pgam in [2, 15], so the "
        "column state is corrupt"),
    3: ("findsp_water did not converge (module_mp_cammgmp_driver.F:2455-"
        "2469); WRF stops the run here with wrf_error_fatal('FINDSP')"),
    4: ("findsp_water lost enthalpy beyond 1e-4 (module_mp_cammgmp_driver.F:"
        "2470-2485); WRF stops the run here with wrf_error_fatal('FINDSP')"),
}


def cammgmp_workspace_slots(nz: int) -> int:
    return CAMMGMP_R8_SLOTS_PER_LEVEL * (int(nz) + 1)


def cammgmp_chunk_columns(nz: int, ncols: int,
                          budget_bytes: int = CAMMGMP_WORKSPACE_BUDGET_BYTES) -> int:
    from woof.core.cam5_memory import chunk_columns
    return chunk_columns(nz,max(int(ncols),1),CAMMGMP_R8_SLOTS_PER_LEVEL,
                         budget_bytes,CAMMGMP_BLOCK,pool_first=True)


def _upload_estbl(module):
    from woof.core.cam5_support import upload_saturation_table
    upload_saturation_table(module)
    return module


@cuda_cache(maxsize=None, ready=True)
def _module():
    """The compiled module with wv_saturation's estbl uploaded once per card.

    ``UW_ESTBL`` is gestbl's table as the UW PBL oracle recorded it
    (woof/core/uwpbl_constants.py); the CAMMGMP oracle's own dump of
    wv_saturation's estbl (tools/cammgmp_wrf461_oracle words.txt) is the
    same 250 words, which tests/test_cammgmp_wrf461_parity.py checks.
    """
    return _upload_estbl(load_module("cammgmp"))


def _module_reproducing_ndrop_defect():
    """Compatibility name for the stock WRF module."""
    return _module()


@cuda_cache(maxsize=None, ready=True)
def _module_corrected_ndrop():
    """Altered-source attribution instrument. Never a native WRF referee."""
    return _upload_estbl(load_module_int_defines(
        "cammgmp", (("CAMMGMP_CORRECTED_NDROP_DIAGNOSTIC", 1),)))


def _check(name, array, shape, dtype=DTYPE):
    if (array.shape != shape or array.dtype != dtype
            or not array.flags.c_contiguous):
        raise ValueError(
            f"CAMMGMP requires {name} as a contiguous {np.dtype(dtype)} array "
            f"of shape {shape}, got {array.dtype} {array.shape}")


def cammgmp_step(inputs, inout, *, dt: float, itimestep: int,
                 cu_coupled: bool, shcu_coupled: bool, accum_mode: float,
                 aitken_mode: float, coarse_mode: float,
                 budget_bytes: int = CAMMGMP_WORKSPACE_BUDGET_BYTES,
                 _reproduce_ndrop_defect: bool = True, workspace=None) -> None:
    """One CAMMGMP call over every column, updating ``inout`` in place.

    ``inputs`` holds :data:`CAMMGMP_MASS_INPUTS` ``(nz, ny, nx)``,
    :data:`CAMMGMP_FULL_INPUTS` ``(nz+1, ny, nx)`` and
    :data:`CAMMGMP_SURFACE_INPUTS` ``(ny, nx)``; ``inout`` holds
    :data:`CAMMGMP_INOUT_3D` and :data:`CAMMGMP_INOUT_2D`.  All float32,
    WRF's k = 1 at the bottom.  ``itimestep`` is WRF's one-based step (1
    zeroes the carried cloud fractions and the first-step TKE/K_h).
    ``cu_coupled``/``shcu_coupled`` are WRF's ``cu_physics == CAMZMSCHEME``
    and ``shcu_physics == CAMUWSHCUSCHEME`` selections of the convective
    inputs (module_mp_cammgmp_driver.F:763-793).
    """
    nz, ny, nx = inout["th"].shape
    lo, hi = VERTICAL_LEVEL_BOUNDS
    if nz < lo or nz > hi:
        from woof.physics_vertical_contract import refuse_vertical_levels
        raise refuse_vertical_levels(
            "CAMMGMP microphysics", VERTICAL_LEVEL_BOUNDS, nz,
            breakage=("dropmixnuc's layer spacing zs(pver) copies zs(pver-1) "
                      "and needs two layers; the upper bound sizes the "
                      "column workspace pool"))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"CAMMGMP needs a finite positive dt, got {dt}")
    for name in CAMMGMP_MASS_INPUTS:
        _check(name, inputs[name], (nz, ny, nx))
    for name in CAMMGMP_FULL_INPUTS:
        _check(name, inputs[name], (nz + 1, ny, nx))
    for name in CAMMGMP_SURFACE_INPUTS:
        _check(name, inputs[name], (ny, nx))
    for name in CAMMGMP_INOUT_3D:
        _check(name, inout[name], (nz, ny, nx))
    for name in CAMMGMP_INOUT_2D:
        _check(name, inout[name], (ny, nx))

    ncols = ny * nx
    chunk = cammgmp_chunk_columns(nz, ncols, budget_bytes)
    r8cap = cammgmp_workspace_slots(nz)
    if workspace is None:
        from woof.core.cam5_workspace import CamWorkspace
        workspace = CamWorkspace()
    pool = workspace.array("mp11/r8", r8cap * chunk, cp.float64)
    err = workspace.array("mp11/status", 1, cp.int32, clear=True)
    module = (_module() if _reproduce_ndrop_defect
              else _module_corrected_ndrop())
    kernel = module.get_function("cammgmp_columns")
    args_fixed = (
        (inputs["p8w"], inputs["p_hyd"], inputs["t_phy"], inputs["pi_phy"],
         inputs["z"], inputs["tke"], inputs["kvh"], inputs["alt"],
         inputs["icwmrsh"], inputs["icwmrdp"], inputs["shfrc"],
         inputs["cmfmc"], inputs["cmfmc2"], inputs["ht"], inputs["xland"],
         inputs["snowh"])
        + tuple(inout[name] for name in CAMMGMP_INOUT_3D)
        + tuple(inout[name] for name in CAMMGMP_INOUT_2D))
    for col0 in range(0, ncols, chunk):
        n = min(chunk, ncols - col0)
        blocks = (n + CAMMGMP_BLOCK - 1) // CAMMGMP_BLOCK
        kernel((blocks,), (CAMMGMP_BLOCK,),
               args_fixed + (np.int64(ncols), np.int64(col0), np.int32(n),
                             np.int32(chunk), np.int32(nz), np.float32(dt),
                             np.int32(itimestep), np.int32(bool(cu_coupled)),
                             np.int32(bool(shcu_coupled)),
                             np.float32(accum_mode), np.float32(aitken_mode),
                             np.float32(coarse_mode), pool, np.int32(r8cap),
                             err))
    from woof.core.health_ledger import read_status
    def describe(status):
        raise RuntimeError(_STATUS.get(status, f"CAMMGMP device status {status}"))
    read_status(err, site="cam5/mp11", describe=describe)

def apply(state, cfg, dt, *, refl_10cm_due=False):
    """Bind the qualified CAM driver to the post-RK microphysics slot."""
    from woof.core import constants as c
    from woof.core.microphysics import (
        MicrophysicsDiagnostics, moist_physics_finish, save_pre_mp_theta,
    )
    from woof.core.morrison import launch_exner

    driver = getattr(state, "physics", None)
    if driver is None:
        raise ValueError("CAMMGMP requires the physics driver's saved step-start "
                         "temperature, hydrostatic pressure and CAM convection exports")
    from woof.core.cam5_runtime import workspace as cam_workspace
    f = driver.fields
    nz, ny, nx = state.p.shape
    shape = (nz, ny, nx)
    theta = state.scratch(shape, "mp_th")
    pi = state.scratch(shape, "mp_pii")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta[...] = thb + state.thp
    launch_exner(state.p, pi)
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    inputs = {
        "p_hyd": f["cam_p_hyd"], "t_phy": f["cam_t_phy"],
        "pi_phy": pi, "z": cp.ascontiguousarray(DTYPE(0.5)*(z8w[:-1]+z8w[1:])),
        "alt": cp.ascontiguousarray(state.alt), "p8w": f["cam_p8w"],
        "tke": f["tke_pbl"], "kvh": f["uw_kvh"] if cfg.bl_pbl_physics == 9 else f["cam_kvh"],
        "icwmrsh": f["icwmrsh"], "icwmrdp": f["icwmrdp"],
        "shfrc": f["shfrc3d"], "cmfmc": f["cmfmc"], "cmfmc2": f["cmfmc2"],
        "ht": cp.ascontiguousarray(state.ht, dtype=DTYPE),
        "xland": f["xland"], "snowh": f["snowh"],
    }
    inout = {name: getattr(state, name) for name in CAMMGMP_INOUT_3D if name != "th"}
    inout["th"] = theta
    inout.update({name: state.scratch((ny, nx), "mp_" + name)
                  for name in CAMMGMP_INOUT_2D})
    epoch = float(getattr(state, "domain_start_offset", 0.0) or 0.0)
    step = int(np.floor((float(state.elapsed_seconds)-epoch)/float(cfg.dt)+0.5))+1
    save_pre_mp_theta(state)
    cammgmp_step(inputs, inout, dt=dt, itimestep=step, cu_coupled=cfg.cu_physics == 7,
                 shcu_coupled=cfg.shcu_physics == 2, accum_mode=cfg.accum_mode,
                 aitken_mode=cfg.aitken_mode, coarse_mode=cfg.coarse_mode,
                 workspace=cam_workspace(driver))
    state.effc[...] = state.lradius
    state.effi[...] = state.iradius
    if refl_10cm_due:
        # CAMMGMP never assigns WRF's REFL_10CM. Retain its native initialized
        # zero field rather than invoking a different microphysics operator.
        from woof.core.refl import stash_refl_10cm
        refl = state.scratch((nz, ny, nx), "refl_10cm")
        refl.fill(0)
        stash_refl_10cm(state, refl)
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(**{name: inout[name] for name in CAMMGMP_INOUT_2D})
