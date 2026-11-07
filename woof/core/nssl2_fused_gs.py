"""Single-state fused GS launcher for WRF NSSL option 18.

The public process launchers in :mod:`woof.core.nssl2` are isolated oracle
slices.  Some of those slices intentionally overlap, so they are not a valid
production dispatcher.  This module is the only production entry point for
the main ``nssl_2mom_gs`` process: it accepts the durable concentration-space
driver workspace, diagnoses all rates from that state, applies the shared WRF
limiters, and writes one aggregate update.

Production selection is owned by the outer microphysics boundary; this module
remains a transport-free numerical stage and never dispatches itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np

from woof.core.kernels import get_kernel
from woof.core.nssl2_shape import DEFAULT_SHAPES, shape_kernel
from woof.core.nssl2_driver_support import (
    NSSL2DriverWorkspace, _step32, _validate_surface_fields,
)
from woof.core.state import DTYPE

_FIELD_COUNT = 16
_THREADS = 128
NSSL2_FUSED_TEMPERATURE_SCRATCH = "nssl2_fused_temperature"
NSSL2_PRIMARY_ICE_TARGET_SCRATCH = "nssl2_primary_ice_target"


def _validate_workspace(workspace: NSSL2DriverWorkspace) -> int:
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    expected = (_FIELD_COUNT, *workspace.shape)
    if workspace.state.shape != expected:
        raise ValueError(
            f"workspace state must have shape {expected}, got "
            f"{workspace.state.shape}"
        )
    if workspace.state.dtype != DTYPE:
        raise TypeError(
            f"workspace state must be float32, got {workspace.state.dtype}"
        )
    if not workspace.state.flags.c_contiguous:
        raise ValueError("workspace state must be C-contiguous")
    return int(np.prod(workspace.shape, dtype=np.int64))


def _validate_vertical_velocity(
    shape: tuple[int, int, int], vertical_velocity
) -> None:
    expected = (shape[0] + 1, shape[1], shape[2])
    if vertical_velocity.shape != expected:
        raise ValueError(
            "vertical_velocity must be the WRF interface field with shape "
            f"{expected}, got {vertical_velocity.shape}"
        )
    if vertical_velocity.dtype != DTYPE:
        raise TypeError(
            "vertical_velocity must be float32, got "
            f"{vertical_velocity.dtype}"
        )
    if not vertical_velocity.flags.c_contiguous:
        raise ValueError("vertical_velocity must be C-contiguous")


_TABLES_READY: dict[tuple, object] = {}
_TABLES_INIT_LOCK = Lock()


def _ensure_init_tables(shapes=DEFAULT_SHAPES) -> None:
    """Build WRF's init-time tables (nssl_2mom_init) in the GS module once.

    The fused kernel reads WRF's Seifert collection coefficients, its
    incomplete-gamma ratio tables and ventilation factors from a device
    global that ``nssl2_gs_init_tables`` fills from WRF's own GAMMA_DP,
    GAMXINFDP, delbk and delabk.  The build is idempotent: under stream
    capture it is simply recorded into the graph.
    """
    import cupy as cp

    device = cp.cuda.runtime.getDevice()
    stream = cp.cuda.get_current_stream()
    key = (device, shapes)
    ready = _TABLES_READY.get(key)
    if ready is None:
        if stream.is_capturing():
            shape_kernel("nssl2_fused_gs", "nssl2_gs_init_tables", shapes)(
                (4,), (256,), ())
            return
        # The cache is shared by ensemble streams. Publish a device event,
        # rather than a host flag that could expose unfinished table writes.
        with _TABLES_INIT_LOCK:
            ready = _TABLES_READY.get(key)
            if ready is None:
                shape_kernel("nssl2_fused_gs", "nssl2_gs_init_tables", shapes)(
                    (4,), (256,), ())
                ready = cp.cuda.Event(disable_timing=True)
                ready.record(stream)
                _TABLES_READY[key] = ready
    stream.wait_event(ready)


def launch_fused_gs(
    workspace: NSSL2DriverWorkspace,
    full_theta,
    air_density,
    pressure_pa,
    exner,
    vertical_velocity,
    temperature_k,
    primary_ice_target_m3,
    dz,
    dt_s: float,
    *,
    hail_on: bool = True,
    loaded_vapor=None,
    shapes=DEFAULT_SHAPES,
) -> None:
    """Advance the complete default option-18 GS slab exactly once.

    ``loaded_vapor`` is the vapour mixing ratio as the driver loaded it
    (the Registry ``qv``, which nothing changes before the final scatter).
    WRF diagnoses the primary-ice target ``t7`` from it before the
    first-step number initialisation moves sub-threshold condensate into
    vapour (``module_mp_nssl_2mom.F:2834-2856`` then ``:2950-2952``).
    Omitted, the workspace vapour is used, which is the same array content
    on every step but the first.

    ``hail_on`` selects WRF's hail category switch.  ``True`` is the resolved
    option-18 default (``nssl_hail_on=1``, ``lhl>1``).  ``False`` is the
    ``nssl_hail_on=0`` variant, where ``module_mp_nssl_2mom.F:1445-1447`` sets
    ``lhl=0`` and the entire graupel<->hail conversion block at :19860 is
    skipped with its rates left at the zeros written just above it
    (:19847-19857).

    ``workspace`` is the object returned by
    :func:`woof.core.nssl2_driver_support.gather_initialize_and_sediment`.
    Its mass moments are kg/kg, number moments are #/m3, and graupel/hail
    volume moments are m3/m3 of air.  Every environmental field is a
    contiguous cell-centred ``(nz, ny, nx)`` FP32 array except
    ``vertical_velocity``, which is GPUWM's interface field with shape
    ``(nz + 1, ny, nx)``. The fused kernel centres it onto mass levels ONCE,
    which is all ``nssl_2mom_gs`` does: the microphysics driver hands NSSL
    the staggered ``grid%w_2`` and ``module_mp_nssl_2mom.F:2827`` copies it
    into the GS slab without de-staggering, so ``:14174-14176``'s
    ``kp1 = Min(nz, kgs+1); wvel = 0.5*(w(kp1)+w(kgs))`` IS the
    interface-to-mass average.  Its clamp is on the upper interface, and at
    the top mass level it degenerates to ``wvel = w(nz)``.

    ``temperature_k`` and ``primary_ice_target_m3`` are required runtime-owned
    scratch arrays. A deterministic prepass overwrites them from the immutable
    pre-GS state: ``temperature_k=full_theta*exner`` and the default
    ``icenucopt=1`` outer-driver ``t7`` diagnosis. This prepass is required so
    neighbouring t7 reads in GS cannot race cells already updated by the fused
    kernel. ``full_theta``, both scratch arrays, and ``workspace.state`` are
    updated in place; the other environmental fields are read-only.

    The CUDA implementation owns source-order rate diagnosis, the exact WRF
    shared donor/category limiters and vapor re-sums, the upstream frozen-vapor
    test-adjustment ceiling, one aggregate mass/number/volume/theta update,
    and the final option-18 moment bounds. The two post-update iterative GS
    saturation-adjustment blocks are skipped by the exact default
    ``ipconc=5``/``ibfc=1`` guards. It never invokes the overlapping isolated
    process launchers.
    """
    if not isinstance(hail_on, bool):
        raise TypeError("hail_on must be bool")
    size = _validate_workspace(workspace)
    fields = {
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "temperature_k": temperature_k,
        "primary_ice_target_m3": primary_ice_target_m3,
        "dz": dz,
    }
    _validate_surface_fields(fields, workspace.shape)
    _validate_vertical_velocity(workspace.shape, vertical_velocity)
    if loaded_vapor is None:
        loaded_vapor = workspace.field("qv")
    else:
        _validate_surface_fields({"loaded_vapor": loaded_vapor}, workspace.shape)
    nz, ny, nx = workspace.shape
    step = _step32(dt_s)

    blocks = (size + _THREADS - 1) // _THREADS
    launch = ((blocks,), (_THREADS,))
    from woof.wrf_exact import ENABLED
    if ENABLED or shapes != DEFAULT_SHAPES:
        _ensure_init_tables(shapes)
    shape_kernel("nssl2_fused_gs", "nssl2_prepare_fused_gs", shapes)(
        *launch,
        (
            temperature_k,
            primary_ice_target_m3,
            loaded_vapor if ENABLED or shapes != DEFAULT_SHAPES else workspace.state,
            full_theta,
            air_density,
            pressure_pa,
            exner,
            np.int32(size),
        ),
    )
    from woof.core.nssl2_shape import requires_cd_carry
    carry=requires_cd_carry(shapes)
    if carry:
        cd=workspace.drag_cd
        if cd is None:raise ValueError("NSSL gathered CD scratch was not prepared")
        groups=(nz*nx+63)//64
        cd[5].view(np.int32).reshape(-1)[:2*ny*groups].fill(-1)
        shape_kernel("nssl2_fused_gs","nssl2_cd_gs_prefix",shapes)(
            ((nz*nx+127)//128,ny),(128,),(workspace.state,air_density,temperature_k,pressure_pa,
                         cd,np.int32(nz),np.int32(ny),np.int32(nx),np.int32(hail_on)))
        shape_kernel("nssl2_fused_gs","nssl2_cd_gs_counts",shapes)(
            (ny,),(128,),(cd,np.int32(nz),np.int32(ny),np.int32(nx)))
        shape_kernel("nssl2_fused_gs","nssl2_cd_gs_groups",shapes)(
            ((nz*nx+127)//128,ny),(128,),(cd,np.int32(nz),np.int32(ny),np.int32(nx)))
        shape_kernel("nssl2_fused_gs","nssl2_cd_gs_finish",shapes)(
            *launch,(cd,np.int32(nz),np.int32(ny),np.int32(nx)))
    shape_kernel("nssl2_fused_gs", "nssl2_fused_gs", shapes)(
        *launch,
        (
            workspace.state,
            full_theta,
            air_density,
            pressure_pa,
            exner,
            temperature_k,
            vertical_velocity,
            primary_ice_target_m3,
            dz,
            step,
            np.int32(nz),
            np.int32(ny * nx),
            np.int32(size),
            np.int32(1 if hail_on else 0),
            *((workspace.drag_cd,np.int32(nx)) if carry else ()),
        ),
    )


@dataclass(frozen=True)
class NSSL2FusedGS:
    """Runtime-stage adapter with restart-rebuilt t0/t7 scratch."""

    temperature_k: object
    primary_ice_target_m3: object
    dt_s: float
    hail_on: bool = True
    shapes: tuple[float, float] = DEFAULT_SHAPES

    def __call__(self, workspace: NSSL2DriverWorkspace, fields, /) -> None:
        launch_fused_gs(
            workspace,
            fields.theta,
            fields.rho,
            fields.pressure,
            fields.pii,
            fields.w,
            self.temperature_k,
            self.primary_ice_target_m3,
            fields.dz,
            self.dt_s,
            hail_on=self.hail_on,
            shapes=self.shapes,
            loaded_vapor=getattr(fields, "qv", None),
        )


__all__ = [
    "NSSL2FusedGS",
    "NSSL2_FUSED_TEMPERATURE_SCRATCH",
    "NSSL2_PRIMARY_ICE_TARGET_SCRATCH",
    "launch_fused_gs",
]
