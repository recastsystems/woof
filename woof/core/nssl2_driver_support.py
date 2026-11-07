"""Pre-GS WRF v4.6.1 NSSL option-18 driver support.

This module owns the low-level state transport path that surrounds NSSL's main
gather/scatter process routine.  It deliberately does not make ``mp_physics=18``
selectable and does not call GS, NUCOND, radar, effective-radius, finish, restart,
or preflight code.

The 16 prognostics are gathered once.  Number and volume moments remain in the
scheme's concentration convention throughout initialization, KF number-moment
diagnosis, and all five sedimentation categories, then are scattered once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.nssl2_shape import DEFAULT_SHAPES, shape_kernel
from woof.core.nssl2_contract import DEFAULT_RESTART_FIELDS
from woof.core.state import DTYPE

_FIELD_COUNT = 16
_SHALLOW_KMAX = 64
_KMAX = 256
_ELEMENT_TPB = 256
_COLUMN_TPB = 64

_QV = 0
_QC = 1
_QR = 2
_QI = 3
_QS = 4
_QG = 5
_QH = 6
_NC = 7
_NR = 8
_NI = 9
_NS = 10
_NG = 11
_NH = 12
_NN = 13
_VG = 14
_VH = 15

NSSL2_DRIVER_FIELDS = DEFAULT_RESTART_FIELDS
NSSL2_SEDIMENT_EXPORTS = ("rain", "ice", "snow", "graupel", "hail")
NSSL2_DRIVER_STATE_SCRATCH = "nssl2_driver_state"
NSSL2_DRIVER_SURFACE_EXPORT_SCRATCH = "nssl2_driver_surface_export"
NSSL2_DRIVER_IGNORED_ACCUMULATOR_SCRATCH = \
    "nssl2_driver_ignored_accumulator"
_FIELD_INDEX = {name: index for index, name in enumerate(NSSL2_DRIVER_FIELDS)}


@dataclass(frozen=True)
class NSSL2DriverWorkspace:
    """Durable pre-GS NSSL state in native internal concentration units.

    ``state`` has shape ``(16, nz, ny, nx)``. Mass fields remain kg/kg,
    number fields are #/m3, predicted CCN is #/m3, and the two volume fields
    are m3 hydrometeor/m3 air. It is the production seam for later GS, NUCOND,
    QVEXCESS, and diagnostic phases; none of those phases should re-gather.

    ``category_surface_export`` has shape ``(5, ny, nx)`` in the fixed order
    rain/ice/snow/graupel/hail.  The cloud-ice row is kg/m2 over the step and
    is not reduced (WRF's four-category reducer omits cloud-ice fallout); the
    four reduced rows carry WRF's surface ``xfall``, which the reducer sums
    before multiplying by ``dt*rho`` exactly as WRF does.
    ``ignored_accumulator`` preserves the likewise-unreduced cloud-droplet
    surface export after sedimentation, allowing an exact water-budget receipt
    without adding it to WRF's precipitation accumulators.
    """

    state: object
    category_surface_export: object
    shape: tuple[int, int, int]
    ignored_accumulator: object | None = None
    drag_cd: object | None = None

    def field(self, name: str):
        """Return one mutable internal-unit field view by Registry name."""
        try:
            index = _FIELD_INDEX[name]
        except KeyError as exc:
            raise KeyError(f"unknown NSSL driver field {name!r}") from exc
        return self.state[index]

    @property
    def ice_surface_export(self):
        """Cloud-ice surface export in kg/m2 over this step."""
        return self.category_surface_export[1]

    @property
    def cloud_surface_export(self):
        """Cloud-droplet surface export in kg/m2 over this step."""
        return self.ignored_accumulator


def validate_nssl2_driver_workspace(
        workspace: NSSL2DriverWorkspace,
        shape: tuple[int, int, int], /) -> None:
    """Validate every persistent driver buffer without copying or reducing."""
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    shape = tuple(shape)
    if tuple(workspace.shape) != shape:
        raise ValueError(
            f"workspace has shape {workspace.shape}, expected {shape}")
    nz, ny, nx = shape
    buffers = {
        "workspace.state": (workspace.state, (_FIELD_COUNT, *shape)),
        "workspace.category_surface_export": (
            workspace.category_surface_export, (5, ny, nx)),
        "workspace.ignored_accumulator": (
            workspace.ignored_accumulator, (ny, nx)),
    }
    if workspace.drag_cd is not None:
        buffers["workspace.drag_cd"] = (workspace.drag_cd, (7, nz + 1, ny, nx))
    for name, (value, expected) in buffers.items():
        if value is None:
            raise ValueError(f"{name} is required for a reusable workspace")
        if tuple(value.shape) != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got "
                f"{tuple(value.shape)}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def _validate_volume_fields(
        fields: dict[str, object],
        ) -> tuple[tuple[int, int, int], int]:
    first = next(iter(fields.values()))
    shape = first.shape
    if len(shape) != 3:
        raise ValueError(f"NSSL driver fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(
            f"NSSL driver support requires 2 <= nz <= {_KMAX}, got {nz}")
    _validate_surface_fields(fields, shape)
    return (nz, ny, nx), int(np.prod(shape, dtype=np.int64))


def _validate_surface_fields(
        fields: dict[str, object], shape: tuple[int, ...],
        ) -> None:
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def _step32(dt_s: float) -> np.float32:
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    converted = np.float32(step)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")
    return converted


def _velocity_gamma_table(device_id, shapes=DEFAULT_SHAPES):
    """Captured calls use base kernels without creating or caching a table."""
    import cupy as cp

    if cp.cuda.get_current_stream().is_capturing():
        return None
    return _cached_velocity_gamma_table(device_id, shapes)


@lru_cache(maxsize=None)
def _cached_velocity_gamma_table(device_id, shapes=DEFAULT_SHAPES):
    """Fill with device tgamma once and preserve the cross-stream dependency."""
    import cupy as cp

    with cp.cuda.Device(device_id):
        table = cp.empty((2, 120), dtype=cp.float64)
        kernel = shape_kernel("nssl2_driver_support", "nssl2_fill_velocity_gamma", shapes)
        kernel((2,), (64,), (table,))
        ready = cp.cuda.Event(disable_timing=True)
        ready.record()
    return table, ready, kernel


def gather_initialize_and_sediment(
        air_density, dz,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        dt_s: float, *, temperature_k,
        first_step: bool = False, cu_used: bool = False,
        qrcuten=None, qscuten=None, qicuten=None, qccuten=None,
        workspace: NSSL2DriverWorkspace | None = None,
        predicted_ccn: bool = True,
        nssl_cccn: float = 0.5e9,
        shapes=DEFAULT_SHAPES,
        ) -> NSSL2DriverWorkspace:
    """Gather once, initialize moments, diagnose KF numbers, and sediment.

    Volume fields use contiguous ``(nz, ny, nx)`` FP32 arrays.  Mass fields are
    kg/kg; number fields are #/kg; graupel/hail volume fields are m3/kg of dry
    air; density is kg/m3; ``temperature_k`` is absolute temperature; and
    ``dz`` is metres. Inputs are never mutated. The returned workspace keeps
    all moments in internal concentration units so GS and subsequent phases can
    run before the single final scatter.

    ``first_step`` selects WRF's exact ``itimestep == 1`` ``calcnfromq`` path.
    When ``cu_used`` is true, each supplied KF ``q*cuten`` rate is converted to
    a step mass increment and passed through exact ``calcnfromcuten`` number
    diagnosis.  Missing KF arrays are zero rates.  As in WRF, these rates do not
    add mass here: dynamics has already applied their mass tendencies.

    ``predicted_ccn`` is WRF's ``nssl_ccn_on``.  ``True`` is the resolved
    option-18 default and loads the prognostic ``qnn`` field.  ``False`` is
    the ``nssl_ccn_on=0`` variant (deprecated ``mp_physics=17``/``22``),
    where the Registry never allocates ``qnn`` and the module diagnoses the
    unactivated CCN from the base concentration every step
    (``module_mp_nssl_2mom.F:2734``).  ``qnn`` is then neither read nor
    written by the scheme.
    """
    if not isinstance(first_step, bool):
        raise TypeError("first_step must be bool")
    if not isinstance(cu_used, bool):
        raise TypeError("cu_used must be bool")
    if not isinstance(predicted_ccn, bool):
        raise TypeError("predicted_ccn must be bool")

    from woof.core.nssl2_contract import (
        nssl2_background_ccn_per_kg, nssl2_cccn_concentration,
    )
    background_ccn_per_kg = nssl2_background_ccn_per_kg(nssl_cccn)
    volume_fields = {
        "air_density": air_density,
        "temperature_k": temperature_k,
        "dz": dz,
        "qv": qv,
        "qc": qc,
        "qr": qr,
        "qi": qi,
        "qs": qs,
        "qg": qg,
        "qh": qh,
        "qndrop": qndrop,
        "qnr": qnr,
        "qni": qni,
        "qns": qns,
        "qng": qng,
        "qnh": qnh,
        "qnn": qnn,
        "qvolg": qvolg,
        "qvolh": qvolh,
    }
    rates = {
        "qrcuten": qrcuten,
        "qscuten": qscuten,
        "qicuten": qicuten,
        "qccuten": qccuten,
    }
    volume_fields.update({
        name: value for name, value in rates.items() if value is not None
    })
    (nz, ny, nx), size = _validate_volume_fields(volume_fields)
    step = _step32(dt_s)

    if workspace is not None:
        validate_nssl2_driver_workspace(workspace, (nz, ny, nx))
        if cu_used and any(value is None for value in rates.values()):
            raise ValueError(
                "a reusable NSSL workspace with cu_used=True requires all "
                "four KF rate arrays")

    # CuPy is imported lazily so CPU-only contract and lint tests remain usable.
    import cupy as cp

    state = (cp.empty((_FIELD_COUNT, nz, ny, nx), dtype=DTYPE)
             if workspace is None else workspace.state)
    zero_rate = None
    rate_args = []
    for value in rates.values():
        if value is None:
            if not cu_used:
                # The CUDA gather branch does not dereference KF pointers
                # when CU is disabled. Reuse a valid volume pointer instead
                # of allocating a full-volume zero field.
                rate_args.append(qv)
            else:
                if zero_rate is None:
                    zero_rate = cp.zeros_like(qv)
                rate_args.append(zero_rate)
        else:
            rate_args.append(value)

    element_blocks = (size + _ELEMENT_TPB - 1) // _ELEMENT_TPB
    coefficient_args = ()
    if nssl2_cccn_concentration(nssl_cccn) == np.float32(0.5e9):
        gather_kernel = shape_kernel(
            "nssl2_driver_support", "nssl2_driver_gather_initialize", shapes)
    else:
        gather_kernel = shape_kernel(
            "nssl2_driver_support", "nssl2_driver_gather_initialize", shapes,
            (("NSSL2_VARIABLE_CCCN", 1),))
        coefficient_args = (background_ccn_per_kg,)
    gather_kernel(
        (element_blocks,), (_ELEMENT_TPB,),
        (air_density, qv, qc, qr, qi, qs, qg, qh,
         qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
         *rate_args, state, step, np.int32(first_step), np.int32(cu_used),
         np.int32(size), np.int32(1 if predicted_ccn else 0), *coefficient_args))

    ncol = ny * nx
    column_blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    if workspace is None:
        category_export = cp.empty((5, ny, nx), dtype=DTYPE)
        ignored_accumulator = cp.zeros((ny, nx), dtype=DTYPE)
    else:
        category_export = workspace.category_surface_export
        ignored_accumulator = workspace.ignored_accumulator
        # The five precipitating/ice kernels read-modify-write this temporary
        # WRF accumulator. Cloud fallout overwrites it with its own separate
        # export receipt after those launches.
        ignored_accumulator[...] = DTYPE(0.0)
    suffix = "64" if nz <= _SHALLOW_KMAX else "256"

    gamma = None
    if isinstance(state, getattr(cp, "ndarray", ())):
        gamma = _velocity_gamma_table(cp.cuda.runtime.getDevice(), shapes)
    if gamma is not None:
        gamma_table, gamma_ready, _ = gamma
        cp.cuda.get_current_stream().wait_event(gamma_ready)

    from woof.core.nssl2_shape import requires_cd_carry
    carry=requires_cd_carry(shapes)
    drag_cd=None
    if carry:
        drag_cd=cp.empty((7,nz+1,ny,nx),dtype=DTYPE) if workspace is None else workspace.drag_cd
        if drag_cd is None:raise ValueError("NSSL gathered CD scratch was not prepared")
        shape_kernel("nssl2_driver_support","nssl2_cd_sediment",shapes)(
            (column_blocks,),(_COLUMN_TPB,),
            (state,air_density,drag_cd,np.int32(nz),np.int32(ny),np.int32(nx),np.int32(1)))

    if gamma is not None and nz <= _SHALLOW_KMAX:
        shape_kernel("nssl2_driver_support", "nssl2_sediment_all_parallel_64", shapes)(
            (ncol, 6), (_COLUMN_TPB,),
            (air_density, temperature_k, state, dz, ignored_accumulator,
             category_export, gamma_table, step,
             np.int32(nz), np.int32(ny), np.int32(nx), *((drag_cd,) if carry else ())))
    else:
        dense_suffix = "cached_256" if gamma is not None else suffix
        sediment_calls = (
            (f"nssl2_rain_sediment_{suffix}", _QR, _NR, None, 0),
            (f"nssl2_ice_sediment_{suffix}", _QI, _NI, None, 1),
            (f"nssl2_snow_sediment_{suffix}", _QS, _NS, None, 2),
            (f"nssl2_graupel_sediment_{dense_suffix}", _QG, _NG, _VG, 3),
            (f"nssl2_hail_sediment_{dense_suffix}", _QH, _NH, _VH, 4),
        )
        for kernel_name, mass_index, number_index, volume_index, export_index in sediment_calls:
            arguments = [air_density, state[mass_index], state[number_index]]
            if volume_index is not None:
                arguments.append(state[volume_index])
            arguments.extend((
                dz, ignored_accumulator, category_export[export_index], step,
                np.int32(nz), np.int32(ny), np.int32(nx),
            ))
            if volume_index is not None and gamma is not None:
                arguments.append(gamma_table[export_index - 3])
            if volume_index is not None and carry:
                arguments.append(drag_cd[export_index-3,0])
            shape_kernel("nssl2_driver_support", kernel_name, shapes)(
                (column_blocks,), (_COLUMN_TPB,), tuple(arguments))

        # Cloud overwrites the ignored accumulator after the other categories.
        shape_kernel("nssl2_driver_support", f"nssl2_cloud_sediment_{suffix}", shapes)(
            (column_blocks,), (_COLUMN_TPB,),
            (air_density, temperature_k, state[_QC], state[_NC], dz,
             ignored_accumulator, step,
             np.int32(nz), np.int32(ny), np.int32(nx)))

    return NSSL2DriverWorkspace(
        state=state, category_surface_export=category_export,
        shape=(nz, ny, nx), ignored_accumulator=ignored_accumulator,drag_cd=drag_cd)


def reduce_nssl2_precipitation(
        workspace: NSSL2DriverWorkspace,
        rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, hailnc, hailncv, sr,
        *, air_density, dt_s: float,
        ) -> None:
    """Apply WRF's four-category surface precipitation reducer and SR.

    WRF sums the categories' surface ``xfall`` before multiplying by
    ``dtp*dn1`` (``module_mp_nssl_2mom.F:3008-3061``); the workspace rows
    carry that ``xfall``, so the reducer needs the step and the cell density
    (only the lowest level is read).
    """
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    _, ny, nx = workspace.shape
    if tuple(air_density.shape) != tuple(workspace.shape):
        raise ValueError(
            f"air_density has shape {air_density.shape}, expected "
            f"{workspace.shape}")
    step = _step32(dt_s)
    _validate_surface_fields({
        "rainnc": rainnc,
        "rainncv": rainncv,
        "snownc": snownc,
        "snowncv": snowncv,
        "graupelnc": graupelnc,
        "graupelncv": graupelncv,
        "hailnc": hailnc,
        "hailncv": hailncv,
        "sr": sr,
    }, (ny, nx))
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    from woof.wrf_exact import ENABLED
    density_args = (air_density,) if ENABLED else ()
    step_args = (step,) if ENABLED else ()
    get_kernel(
        "nssl2_driver_support", "nssl2_driver_reduce_precipitation",
    )((blocks,), (_COLUMN_TPB,),
      (workspace.category_surface_export, *density_args,
       rainnc, rainncv, snownc, snowncv,
       graupelnc, graupelncv, hailnc, hailncv, sr, *step_args, np.int32(ncol)))


def scatter_nssl2_driver_workspace(
        workspace: NSSL2DriverWorkspace, air_density,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        *, predicted_ccn: bool = True,
        ) -> None:
    """Perform the one final concentration-to-Registry scatter in place.

    ``predicted_ccn=False`` reproduces WRF's ``nssl_ccn_on=0`` store, which
    skips the CCN write entirely (``module_mp_nssl_2mom.F:3283`` requires
    ``flag_ccn``); ``qnn`` is left exactly as the caller supplied it.
    """
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    if not isinstance(predicted_ccn, bool):
        raise TypeError("predicted_ccn must be bool")
    fields = {
        "air_density": air_density,
        "qv": qv,
        "qc": qc,
        "qr": qr,
        "qi": qi,
        "qs": qs,
        "qg": qg,
        "qh": qh,
        "qndrop": qndrop,
        "qnr": qnr,
        "qni": qni,
        "qns": qns,
        "qng": qng,
        "qnh": qnh,
        "qnn": qnn,
        "qvolg": qvolg,
        "qvolh": qvolh,
    }
    shape, size = _validate_volume_fields(fields)
    if shape != workspace.shape:
        raise ValueError(
            f"workspace has shape {workspace.shape}, output fields have "
            f"shape {shape}")
    blocks = (size + _ELEMENT_TPB - 1) // _ELEMENT_TPB
    get_kernel("nssl2_driver_support", "nssl2_driver_scatter")(
        (blocks,), (_ELEMENT_TPB,),
        (air_density, workspace.state, qv, qc, qr, qi, qs, qg, qh,
         qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
         np.int32(size), np.int32(1 if predicted_ccn else 0)))


def launch_nssl2_driver_support(
        air_density, dz,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, hailnc, hailncv, sr,
        dt_s: float, *, temperature_k,
        first_step: bool = False, cu_used: bool = False,
        qrcuten=None, qscuten=None, qicuten=None, qccuten=None,
        ) -> NSSL2DriverWorkspace:
    """Round-trip convenience wrapper for isolated official-oracle gates.

    Production integration must use the three phase functions directly so GS,
    NUCOND/QVEXCESS, and diagnostics operate on the durable concentration-space
    workspace before :func:`scatter_nssl2_driver_workspace` is called once.
    """
    workspace = gather_initialize_and_sediment(
        air_density, dz,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        dt_s, temperature_k=temperature_k,
        first_step=first_step, cu_used=cu_used,
        qrcuten=qrcuten, qscuten=qscuten,
        qicuten=qicuten, qccuten=qccuten)
    reduce_nssl2_precipitation(
        workspace, rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, hailnc, hailncv, sr,
        air_density=air_density, dt_s=dt_s)
    scatter_nssl2_driver_workspace(
        workspace, air_density,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh)
    return workspace


__all__ = [
    "NSSL2_DRIVER_IGNORED_ACCUMULATOR_SCRATCH",
    "NSSL2_DRIVER_STATE_SCRATCH",
    "NSSL2_DRIVER_SURFACE_EXPORT_SCRATCH",
    "NSSL2_DRIVER_FIELDS",
    "NSSL2_SEDIMENT_EXPORTS",
    "NSSL2DriverWorkspace",
    "gather_initialize_and_sediment",
    "launch_nssl2_driver_support",
    "reduce_nssl2_precipitation",
    "scatter_nssl2_driver_workspace",
    "validate_nssl2_driver_workspace",
]
