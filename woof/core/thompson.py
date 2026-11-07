"""Classic Thompson (WRF ``mp_physics=8``) CUDA implementation.

The production adapter composes these source, adjustment, sedimentation and
diagnostic launchers with canonical coefficient tables. Numerical reference
checks establish their stated fixture coverage, not general forecast skill.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from woof.core import constants as _constants
from woof.core.kernels import get_kernel
from woof.core.state import DTYPE

_COLUMN_TPB = 32
_SHALLOW_KMAX = 64
_KMAX = 256


@lru_cache(None)
def _warm_network_threads(device_id: int) -> int:
    """Cache the warm network width for each device architecture."""
    import cupy as cp

    major = int(cp.cuda.runtime.getDeviceProperties(device_id)["major"])
    # Wider blocks avoid excess block scheduling on these architectures.
    # The independent cell arithmetic is identical at either launch width.
    return 256 if major in (9, 10) else 64

#: Fallout of a column of at most ``_SHALLOW_KMAX`` levels runs through the
#: ``*_levels_*`` kernels: eight columns a block (thirty-two for graupel,
#: compacted to the active ones), one thread per level, byte-identical to the
#: column kernels (tests/test_thompson_speed_shortcuts.py).  Those kernels
#: need a block barrier, so they are compiled for the device only; the host
#: kernel backend (tools/thompson_real_column_parity), which runs a block's
#: threads one after another, turns this off and takes the column kernels.
LEVEL_PARALLEL_FALLOUT = True
_LEVEL_BLOCK = (8, _SHALLOW_KMAX)


def _launch_fallout(level_symbol, column_symbol, level_parallel: bool,
                    ncol: int, arguments, *, columns_per_block: int = 8):
    """Launch one fallout kernel over ``ncol`` columns."""
    if level_parallel:
        get_kernel("thompson", level_symbol)(
            ((ncol + columns_per_block - 1) // columns_per_block,),
            _LEVEL_BLOCK, arguments)
    else:
        get_kernel("thompson", column_symbol)(
            ((ncol + _COLUMN_TPB - 1) // _COLUMN_TPB,), (_COLUMN_TPB,),
            arguments)
VERTICAL_LEVEL_BOUNDS = (2, _KMAX)


def _arrays_overlap(left, right) -> bool:
    """Return whether two contiguous host/device arrays share any storage."""
    if left is right:
        return True

    def interval(value):
        interface = getattr(value, "__cuda_array_interface__", None)
        address_space = "cuda"
        if interface is None:
            interface = getattr(value, "__array_interface__", None)
            address_space = "host"
        if interface is None:
            return None
        pointer = int(interface["data"][0] or 0)
        nbytes = int(value.nbytes)
        return address_space, pointer, pointer + nbytes

    left_interval = interval(left)
    right_interval = interval(right)
    if left_interval is None or right_interval is None:
        return False
    left_space, left_start, left_end = left_interval
    right_space, right_start, right_end = right_interval
    return (left_space == right_space
            and left_start < right_end
            and right_start < left_end)


def _validate_fields(fields: dict[str, object]) -> tuple[tuple[int, ...], int]:
    first = next(iter(fields.values()))
    shape = first.shape
    if not shape:
        raise ValueError("Thompson fields must be arrays")
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    return shape, int(np.prod(shape, dtype=np.int64))


def _validate_fp64_fortran_table(name, value, shape) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be float64, got {value.dtype}")
    if not value.flags.f_contiguous:
        raise ValueError(f"{name} must be Fortran-contiguous")


def launch_warm_saturation_adjust(temperature, pressure, qv, qc) -> None:
    """Apply WRF's isolated warm cloud-vapor saturation adjustment.

    This state-changing slice assumes zero incoming microphysics tendencies;
    ice processes and sedimentation remain outside its admitted contract.
    """
    _, size = _validate_fields({
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "qc": qc,
    })
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_warm_saturation_adjust")(
        (blocks,), (threads,),
        (temperature, pressure, qv, qc, np.int32(size)))


def launch_cloud_saturation_adjust(
        temperature, pressure, qv, qc, *, reference_density=None,
        reference_temperature=None, condensation_marker=None,
        cloud_presence=None) -> None:
    """Apply WRF's liquid-cloud saturation adjustment at any temperature.

    When supplied, ``reference_density`` records the post-process,
    pre-adjustment density used by same-call hydrometeor sedimentation.
    ``reference_temperature`` additionally records the pre-adjustment
    temperature used by WRF's held snow-moment diagnostics and requires the
    density output to be supplied as well.
    ``condensation_marker`` records a positive phase change before storage
    rounding, retaining the decision that suppresses same-call rain evaporation.
    ``cloud_presence`` receives WRF's L_qc(k) as the adjustment leaves it
    (module_mp_thompson.F:3215-3223, cleared at :3485), 1 or 0 at every
    level, which the cloud fallout's column gate reads (:3645); it requires
    the three outputs above.
    """
    fields = {
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "qc": qc,
    }
    if cloud_presence is not None:
        if (condensation_marker is None or reference_density is None
                or reference_temperature is None):
            raise ValueError(
                "cloud_presence requires reference_density, "
                "reference_temperature and condensation_marker")
        fields["cloud_presence"] = cloud_presence
    if reference_density is not None:
        fields["reference_density"] = reference_density
    if reference_temperature is not None:
        if reference_density is None:
            raise ValueError(
                "reference_temperature requires reference_density")
        fields["reference_temperature"] = reference_temperature
    if condensation_marker is not None:
        fields["condensation_marker"] = condensation_marker
    _, size = _validate_fields(fields)
    if condensation_marker is not None:
        for name, value in fields.items():
            if name != "condensation_marker" and _arrays_overlap(condensation_marker, value):
                raise ValueError(
                    f"condensation_marker must not alias {name}: preserve "
                    "the state and decision in separate storage")
    if cloud_presence is not None:
        for name, value in fields.items():
            if name != "cloud_presence" and _arrays_overlap(
                    cloud_presence, value):
                raise ValueError(
                    f"cloud_presence must not alias {name}")
    threads = 256
    blocks = (size + threads - 1) // threads
    if cloud_presence is not None:
        get_kernel(
            "thompson", "thompson_cloud_saturation_adjust_with_presence")(
                (blocks,), (threads,),
                (temperature, pressure, qv, qc, reference_density,
                 reference_temperature, condensation_marker, cloud_presence,
                 np.int32(size)))
    elif condensation_marker is not None:
        get_kernel("thompson", "thompson_cloud_saturation_adjust_with_history")(
            (blocks,), (threads,),
            (temperature, pressure, qv, qc,
             reference_density if reference_density is not None else np.uint64(0),
             reference_temperature if reference_temperature is not None else np.uint64(0),
             condensation_marker, np.int32(size)))
    elif reference_density is None:
        get_kernel("thompson", "thompson_cloud_saturation_adjust")(
            (blocks,), (threads,),
            (temperature, pressure, qv, qc, np.int32(size)))
    elif reference_temperature is None:
        get_kernel(
            "thompson", "thompson_cloud_saturation_adjust_with_density")(
                (blocks,), (threads,),
                (temperature, pressure, qv, qc, reference_density,
                 np.int32(size)))
    else:
        get_kernel(
            "thompson", "thompson_cloud_saturation_adjust_with_state")(
                (blocks,), (threads,),
                (temperature, pressure, qv, qc, reference_density,
                 reference_temperature, np.int32(size)))


def launch_warm_autoconversion(
        qc, qr, nr, temperature, pressure, qv, dt: float) -> None:
    """Apply WRF's isolated Berry-Reinhardt cloud-to-rain conversion.

    The admitted slice assumes no incoming rain.  Existing-rain accretion and
    self-collection remain part of the still-open warm-rain process network.
    """
    _, size = _validate_fields({
        "qc": qc,
        "qr": qr,
        "nr": nr,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_warm_autoconversion")(
        (blocks,), (threads,),
        (qc, qr, nr, temperature, pressure, qv,
         DTYPE(dt), np.int32(size)))


def launch_rain_self_collection(
        qr, nr, temperature, pressure, qv, dt: float) -> None:
    """Apply WRF's isolated Seifert rain self-collection number sink."""
    _, size = _validate_fields({
        "qr": qr,
        "nr": nr,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_rain_self_collection")(
        (blocks,), (threads,),
        (qr, nr, temperature, pressure, qv,
         DTYPE(dt), np.int32(size)))


def launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, dt: float,
        *, reference_density=None, reference_temperature=None,
        graupel_melt_marker=None, source_density=None,
        condensation_marker=None,
        density_carries_rain_presence: bool = False) -> None:
    """Apply WRF's ordinary subsaturated-rain evaporation process.

    This admitted slice covers the Srivastava-Coen branch for an already
    bounded two-moment rain distribution.  When supplied, ``source_density``
    retains the rain mass and number concentrations formed before cloud
    adjustment.  ``reference_density`` carries that density into fallout
    unless rain evaporation executes and refreshes the concentrations with
    its own incoming density.  An optional
    ``reference_temperature`` preserves held snow moments and requires the
    density output.  ``graupel_melt_marker`` carries WRF's held
    ``prr_gml > 0`` decision so rain evaporation is reduced where liquid is
    still coating melting graupel; it requires the density-only output form.
    A positive ``condensation_marker`` suppresses this process, retaining
    WRF's cloud-adjustment decision independently of saturation roundoff.
    ``density_carries_rain_presence`` also writes WRF's L_qr into
    ``reference_density`` (zero where :3236 failed) and marks the :3568
    rewrite (negative), for ``launch_rain_sedimentation``'s and
    ``launch_snow_sedimentation``'s presence forms; it requires the history
    form (``source_density`` or ``condensation_marker``).
    """
    fields = {
        "qr": qr,
        "nr": nr,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    }
    if reference_density is not None:
        fields["reference_density"] = reference_density
    if reference_temperature is not None:
        if reference_density is None:
            raise ValueError(
                "reference_temperature requires reference_density")
        fields["reference_temperature"] = reference_temperature
    if graupel_melt_marker is not None:
        if reference_density is None or reference_temperature is not None:
            raise ValueError(
                "graupel_melt_marker requires reference_density and is "
                "incompatible with reference_temperature")
        fields["graupel_melt_marker"] = graupel_melt_marker
    if source_density is not None:
        if reference_density is None or reference_temperature is not None:
            raise ValueError(
                "source_density requires the density-only output form: "
                "pass reference_density without reference_temperature")
        fields["source_density"] = source_density
    if condensation_marker is not None:
        if reference_density is None or reference_temperature is not None:
            raise ValueError(
                "condensation_marker requires the density-only output form: "
                "pass reference_density without reference_temperature")
        fields["condensation_marker"] = condensation_marker
    _, size = _validate_fields(fields)
    if (graupel_melt_marker is not None
            and _arrays_overlap(graupel_melt_marker, reference_density)):
        raise ValueError(
            "graupel_melt_marker must not alias reference_density: the "
            "CUDA kernel reads the held marker while writing RHOF")
    for held in ("source_density", "condensation_marker"):
        if held in fields:
            for name in ("qr", "nr", "temperature", "qv", "reference_density"):
                if _arrays_overlap(fields[held], fields[name]):
                    raise ValueError(
                        f"{held} must not alias {name}: retain its "
                        "incoming state in separate storage")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    if density_carries_rain_presence and (
            source_density is None and condensation_marker is None):
        raise ValueError(
            "density_carries_rain_presence requires the history form: pass "
            "source_density or condensation_marker")
    if source_density is not None or condensation_marker is not None:
        get_kernel("thompson", (
            "thompson_rain_evaporation_with_density_history_and_presence"
            if density_carries_rain_presence
            else "thompson_rain_evaporation_with_density_history"))(
            (blocks,), (threads,),
            (qr, nr, temperature, pressure, qv, reference_density,
             graupel_melt_marker if graupel_melt_marker is not None else np.uint64(0),
             source_density if source_density is not None else np.uint64(0),
             condensation_marker if condensation_marker is not None else np.uint64(0),
             DTYPE(dt), np.int32(size)))
    elif reference_density is None:
        get_kernel("thompson", "thompson_rain_evaporation")(
            (blocks,), (threads,),
            (qr, nr, temperature, pressure, qv,
             DTYPE(dt), np.int32(size)))
    elif graupel_melt_marker is not None:
        get_kernel(
            "thompson",
            "thompson_rain_evaporation_with_density_and_graupel_melt_marker",
        )(
            (blocks,), (threads,),
            (qr, nr, temperature, pressure, qv, reference_density,
             graupel_melt_marker, DTYPE(dt), np.int32(size)))
    elif reference_temperature is None:
        get_kernel("thompson", "thompson_rain_evaporation_with_density")(
            (blocks,), (threads,),
            (qr, nr, temperature, pressure, qv, reference_density,
             DTYPE(dt), np.int32(size)))
    else:
        get_kernel("thompson", "thompson_rain_evaporation_with_state")(
            (blocks,), (threads,),
            (qr, nr, temperature, pressure, qv, reference_density,
             reference_temperature, DTYPE(dt), np.int32(size)))


def launch_snow_sublimation(
        qs, temperature, pressure, qv, dt: float) -> None:
    """Apply WRF's cold snow vapor-deposition/sublimation process."""
    _, size = _validate_fields({
        "qs": qs,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_snow_sublimation")(
        (blocks,), (threads,),
        (qs, temperature, pressure, qv, DTYPE(dt), np.int32(size)))


def launch_graupel_sublimation(
        qg, temperature, pressure, qv, dt: float) -> None:
    """Apply WRF's cold, subsaturated graupel-sublimation process."""
    _, size = _validate_fields({
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_graupel_sublimation")(
        (blocks,), (threads,),
        (qg, temperature, pressure, qv, DTYPE(dt), np.int32(size)))


def launch_snow_melting(
        qs, qr, nr, temperature, pressure, qv, dt: float) -> None:
    """Apply WRF's saturated warm snow-to-rain melting branch."""
    _, size = _validate_fields({
        "qs": qs,
        "qr": qr,
        "nr": nr,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_snow_melting")(
        (blocks,), (threads,),
        (qs, qr, nr, temperature, pressure, qv,
         DTYPE(dt), np.int32(size)))


def launch_graupel_melting(
        qg, qr, nr, graupel_number, temperature, pressure, qv,
        dt: float) -> None:
    """Apply WRF's saturated warm graupel-to-rain melting branch.

    ``graupel_number`` is caller-owned FP32 scratch.  Classic mp=8 diagnoses
    this non-persistent moment at the start of each call, evolves it through
    melting, and consumes it during same-call fallout.
    """
    _, size = _validate_fields({
        "qg": qg,
        "qr": qr,
        "nr": nr,
        "graupel_number": graupel_number,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_graupel_melting")(
        (blocks,), (threads,),
        (qg, qr, nr, graupel_number, temperature, pressure, qv,
         DTYPE(dt), np.int32(size)))


def launch_warm_rain_collection(
        qc, qr, nr, temperature, pressure, qv,
        rain_cloud_efficiency, dt: float) -> None:
    """Apply WRF rain self-collection and rain-cloud accretion together.

    ``rain_cloud_efficiency`` is the canonical FP64 Fortran-ordered
    ``t_Efrw(100, 100)`` table from a validated classic Thompson table set.
    The admitted slice assumes cloud mass remains below the autoconversion
    threshold so all process rates share the same incoming rain state.
    """
    _, size = _validate_fields({
        "qc": qc,
        "qr": qr,
        "nr": nr,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    _validate_fp64_fortran_table(
        "rain_cloud_efficiency", rain_cloud_efficiency, (100, 100))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_warm_rain_collection")(
        (blocks,), (threads,),
        (qc, qr, nr, temperature, pressure, qv, rain_cloud_efficiency,
         DTYPE(dt), np.int32(size)))


def launch_warm_process_network(
        qc, qr, nr, temperature, pressure, qv,
        rain_cloud_efficiency, dt: float) -> None:
    """Run the fused simultaneous classic-Thompson warm-rain network.

    Berry-Reinhardt autoconversion, rain-cloud accretion, and Seifert rain
    self-collection are all diagnosed from one immutable incoming state.  The
    shared WRF cloud-water source cap is then applied once and the categories
    are updated once.  This is the first executable slice of the unified
    process driver, not yet a selectable complete ``mp_physics=8`` scheme.
    """
    _, size = _validate_fields({
        "qc": qc,
        "qr": qr,
        "nr": nr,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    _validate_fp64_fortran_table(
        "rain_cloud_efficiency", rain_cloud_efficiency, (100, 100))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_warm_process_network")(
        (blocks,), (threads,),
        (qc, qr, nr, temperature, pressure, qv, rain_cloud_efficiency,
         DTYPE(dt), np.int32(size)))


def launch_ice_autoconversion(
        qi, ni, qs, temperature, pressure, qv,
        ice_to_snow_mass, ice_to_snow_number, dt: float) -> None:
    """Apply WRF's lookup-table cloud-ice to snow autoconversion.

    The tables are the canonical FP64 Fortran-ordered ``tps_iaus`` and
    ``tni_iaus`` arrays from a validated classic Thompson table set.  This
    admitted slice assumes ice saturation and no competing frozen processes.
    """
    _, size = _validate_fields({
        "qi": qi,
        "ni": ni,
        "qs": qs,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    _validate_fp64_fortran_table(
        "ice_to_snow_mass", ice_to_snow_mass, (64, 55))
    _validate_fp64_fortran_table(
        "ice_to_snow_number", ice_to_snow_number, (64, 55))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_ice_autoconversion")(
        (blocks,), (threads,),
        (qi, ni, qs, temperature, pressure, qv,
         ice_to_snow_mass, ice_to_snow_number,
         DTYPE(dt), np.int32(size)))


def launch_ice_deposition(
        qi, ni, qs, temperature, pressure, qv,
        ice_deposition_partition, dt: float) -> None:
    """Apply classic Thompson cloud-ice deposition/sublimation.

    ``ice_deposition_partition`` is the canonical FP64 Fortran-ordered
    ``tpi_ide(64,55)`` table.  Positive vapor deposition is partitioned
    between cloud ice and snow exactly as WRF does; subsaturated cloud ice
    instead sublimates from the ice category and reduces its number moment.
    """
    _, size = _validate_fields({
        "qi": qi,
        "ni": ni,
        "qs": qs,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    _validate_fp64_fortran_table(
        "ice_deposition_partition", ice_deposition_partition, (64, 55))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_ice_deposition")(
        (blocks,), (threads,),
        (qi, ni, qs, temperature, pressure, qv,
         ice_deposition_partition, DTYPE(dt), np.int32(size)))


def launch_frozen_vapor_network(
        qi, ni, qs, qg, qr, nr, temperature, pressure, qv,
        ice_deposition_partition, ice_to_snow_mass, ice_to_snow_number,
        dt: float, *, rain_snow_tables=None, rain_graupel_tables=None,
        rain_freezing_tables=None, qc=None, rain_cloud_efficiency=None,
        cloud_freezing_tables=None, graupel_number_shadow=None,
        snow_velocity_boost=None) -> None:
    """Apply simultaneous classic nucleation and cold-ice source exchange.

    Non-aerosol Cooper ice nucleation and every frozen-species vapor rate are
    diagnosed with ice autoconversion plus snow and rain collection of cloud
    ice from one incoming state. WRF's shared vapor, cloud-ice, and rain mass
    limiters are applied in driver order, including their deliberately held
    number and paired-category quirks, before one state update.
    """
    fields = {
        "qi": qi,
        "ni": ni,
        "qs": qs,
        "qg": qg,
        "qr": qr,
        "nr": nr,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    }
    if qc is not None:
        fields["qc"] = qc
    if graupel_number_shadow is not None:
        fields["graupel_number_shadow"] = graupel_number_shadow
    if snow_velocity_boost is not None:
        fields["snow_velocity_boost"] = snow_velocity_boost
    _, size = _validate_fields(fields)
    _validate_fp64_fortran_table(
        "ice_deposition_partition", ice_deposition_partition, (64, 55))
    _validate_fp64_fortran_table(
        "ice_to_snow_mass", ice_to_snow_mass, (64, 55))
    _validate_fp64_fortran_table(
        "ice_to_snow_number", ice_to_snow_number, (64, 55))
    cold_groups = (
        rain_snow_tables, rain_graupel_tables, rain_freezing_tables)
    supplied = tuple(group is not None for group in cold_groups)
    if any(supplied) and not all(supplied):
        raise ValueError(
            "rain_snow_tables, rain_graupel_tables, and "
            "rain_freezing_tables must be supplied together")
    include_cold_rain = int(all(supplied))
    if include_cold_rain:
        try:
            rain_snow_values = tuple(rain_snow_tables)
            rain_graupel_values = tuple(rain_graupel_tables)
            rain_freezing_values = tuple(rain_freezing_tables)
        except TypeError as exc:
            raise TypeError("all cold-rain table groups must be iterable") \
                from exc
        rain_snow_names = (
            "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
            "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
            "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
        )
        rain_graupel_names = (
            "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg",
            "tnr_gacr",
        )
        rain_freezing_names = (
            "rain_to_ice_mass", "rain_to_ice_number",
            "rain_to_graupel_mass", "rain_to_graupel_number",
        )
        for label, values, names in (
                ("rain_snow_tables", rain_snow_values, rain_snow_names),
                ("rain_graupel_tables", rain_graupel_values,
                 rain_graupel_names),
                ("rain_freezing_tables", rain_freezing_values,
                 rain_freezing_names)):
            if len(values) != len(names):
                raise ValueError(
                    f"{label} must contain {len(names)} arrays, "
                    f"got {len(values)}")
        for name, table in zip(
                rain_snow_names, rain_snow_values, strict=True):
            _validate_fp64_fortran_table(name, table, (37, 9, 37, 37))
        for name, table in zip(
                rain_graupel_names, rain_graupel_values, strict=True):
            _validate_fp64_fortran_table(
                name, table, (37, 37, 1, 37, 37))
        for name, table in zip(
                rain_freezing_names, rain_freezing_values, strict=True):
            _validate_fp64_fortran_table(name, table, (37, 37, 45, 55))
    else:
        # The global kernel keeps one ABI for focused and production gates.
        # Disabled branches never dereference these canonical device pointers.
        rain_snow_values = (ice_deposition_partition,) * 12
        rain_graupel_values = (ice_deposition_partition,) * 5
        rain_freezing_values = (ice_deposition_partition,) * 4
    cloud_group = (qc, rain_cloud_efficiency, cloud_freezing_tables)
    cloud_supplied = tuple(value is not None for value in cloud_group)
    if any(cloud_supplied) and not all(cloud_supplied):
        raise ValueError(
            "qc, rain_cloud_efficiency, and cloud_freezing_tables "
            "must be supplied together")
    include_cold_cloud = int(all(cloud_supplied))
    include_snow_rime_conversion = int(snow_velocity_boost is not None)
    if include_snow_rime_conversion and not include_cold_cloud:
        raise ValueError(
            "snow_velocity_boost requires the complete cold-cloud group")
    if include_cold_cloud:
        _validate_fp64_fortran_table(
            "rain_cloud_efficiency", rain_cloud_efficiency, (100, 100))
        try:
            cloud_freezing_values = tuple(cloud_freezing_tables)
        except TypeError as exc:
            raise TypeError("cloud_freezing_tables must be iterable") from exc
        if len(cloud_freezing_values) != 2:
            raise ValueError(
                "cloud_freezing_tables must contain 2 arrays, got "
                f"{len(cloud_freezing_values)}")
        for name, table in zip(
                ("cloud_to_ice_mass", "cloud_to_ice_number"),
                cloud_freezing_values, strict=True):
            _validate_fp64_fortran_table(name, table, (37, 100, 45, 55))
        qc_value = qc
        rain_cloud_efficiency_value = rain_cloud_efficiency
    else:
        # The disabled branch never dereferences this non-restrict pointer.
        qc_value = qi
        rain_cloud_efficiency_value = ice_deposition_partition
        cloud_freezing_values = (ice_deposition_partition,) * 2
    snow_velocity_boost_value = (
        qg if snow_velocity_boost is None else snow_velocity_boost)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    # Small blocks distribute this register-heavy cell kernel across SMs.
    # The cell mapping and floating-point expression order are unchanged.
    threads = 64
    blocks = (size + threads - 1) // threads
    common = (
        qi, ni, qs, qg, qr, nr, temperature, pressure, qv,
        ice_deposition_partition, ice_to_snow_mass, ice_to_snow_number,
        *rain_snow_values, *rain_graupel_values, *rain_freezing_values,
    )
    if include_cold_cloud or graupel_number_shadow is not None:
        graupel_number_value = (
            qg if graupel_number_shadow is None else graupel_number_shadow)
        get_kernel(
            "thompson", "thompson_frozen_vapor_cloud_network")(
                (blocks,), (threads,),
                (qi, ni, qs, qg, qr, nr, temperature, pressure, qv,
                 qc_value,
                 ice_deposition_partition, ice_to_snow_mass,
                 ice_to_snow_number,
                 *rain_snow_values, *rain_graupel_values,
                  *rain_freezing_values,
                  rain_cloud_efficiency_value, *cloud_freezing_values,
                  graupel_number_value,
                  snow_velocity_boost_value,
                  np.int32(graupel_number_shadow is not None),
                  np.int32(include_snow_rime_conversion),
                  np.int32(include_cold_rain), np.int32(include_cold_cloud),
                  DTYPE(dt), np.int32(size)))
    else:
        # Preserve the sealed focused/full-cold ABI and generated code.  The
        # larger cloud-overlap kernel is a separate admission surface so a
        # disabled feature cannot perturb existing floating-point paths via
        # register allocation or restrict-alias optimization.  An output-due
        # ng shadow intentionally selects the larger kernel above even when
        # qc is absent because only that ABI carries the diagnostic scratch;
        # include_cold_cloud=0 keeps every cloud branch disabled.
        get_kernel("thompson", "thompson_frozen_vapor_network")(
            (blocks,), (threads,),
            (*common, np.int32(include_cold_rain),
             DTYPE(dt), np.int32(size)))


def launch_frozen_vapor_network_from_owner(
        qi, ni, qs, qg, qr, nr, temperature, pressure, qv,
        table_owner, dt: float, *, qc=None,
        graupel_number_shadow=None, snow_velocity_boost=None) -> None:
    """Launch the fused cold source from one verified runtime table owner.

    This is the production-facing coefficient seam: callers cannot reorder
    the 21 cold-rain records or accidentally mix arrays from different table
    roots.  It does not by itself make ``mp_physics=8`` selectable.
    """
    from woof.core.thompson_runtime import DeviceClassicTableSet

    if (not isinstance(table_owner, DeviceClassicTableSet)
            or not table_owner.roundtrip_verified):
        raise TypeError(
            "table_owner must be a verified DeviceClassicTableSet")
    if qc is not None and snow_velocity_boost is None:
        raise ValueError(
            "production cold-cloud Thompson requires snow_velocity_boost")
    if qc is None and snow_velocity_boost is not None:
        raise ValueError("snow_velocity_boost requires qc")
    tables = table_owner.cold_source_tables
    launch_frozen_vapor_network(
        qi, ni, qs, qg, qr, nr, temperature, pressure, qv,
        tables.ice_deposition_partition,
        tables.ice_to_snow_mass,
        tables.ice_to_snow_number,
        dt,
        rain_snow_tables=tables.rain_snow_tables,
        rain_graupel_tables=tables.rain_graupel_tables,
        rain_freezing_tables=tables.rain_freezing_tables,
        qc=qc,
        rain_cloud_efficiency=(
            tables.rain_cloud_efficiency if qc is not None else None),
        cloud_freezing_tables=(
            tables.cloud_freezing_tables if qc is not None else None),
        graupel_number_shadow=graupel_number_shadow,
        snow_velocity_boost=snow_velocity_boost,
    )


def launch_warm_frozen_source_network(
        qc, qr, nr, qs, qg, graupel_number_shadow,
        graupel_melt_marker, snow_melt_marker,
        temperature, pressure, qv, rain_cloud_efficiency,
        snow_cloud_efficiency, rain_snow_tables,
        rain_graupel_tables, dt: float) -> None:
    """Apply WRF-ordered warm-level rain and frozen-hydrometeor sources.

    The companion fused cold kernel already owns all ambient sub-freezing
    levels.  ``graupel_melt_marker`` must initially contain the held
    ``T >= 273.15 K`` mask from call entry; the kernel consumes that mask,
    then overwrites it with WRF's held ``prr_gml > 0`` decision.
    ``snow_melt_marker`` receives the independent held ``prr_sml > 0``
    decision used by snow fallout.  Both outputs are explicitly zeroed for
    cold-entry cells.
    WRF's diagnosed wet-bulb temperature selects the rain/snow and
    rain/graupel collision and melting branches.  ``graupel_number_shadow``
    is classic Thompson's transient same-call ``ng1d`` moment, not a
    prognostic GPUWM state variable.
    """
    _, size = _validate_fields({
        "qc": qc,
        "qr": qr,
        "nr": nr,
        "qs": qs,
        "qg": qg,
        "graupel_number_shadow": graupel_number_shadow,
        "graupel_melt_marker": graupel_melt_marker,
        "snow_melt_marker": snow_melt_marker,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if _arrays_overlap(graupel_melt_marker, snow_melt_marker):
        raise ValueError(
            "snow_melt_marker must not alias graupel_melt_marker")
    _validate_fp64_fortran_table(
        "rain_cloud_efficiency", rain_cloud_efficiency, (100, 100))
    _validate_fp64_fortran_table(
        "snow_cloud_efficiency", snow_cloud_efficiency, (100, 100))
    rain_snow_values = tuple(rain_snow_tables)
    rain_graupel_values = tuple(rain_graupel_tables)
    rain_snow_names = (
        "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
        "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
        "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
    )
    rain_graupel_names = (
        "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
    )
    if len(rain_snow_values) != len(rain_snow_names):
        raise ValueError(
            "rain_snow_tables must contain 12 arrays, got "
            f"{len(rain_snow_values)}")
    if len(rain_graupel_values) != len(rain_graupel_names):
        raise ValueError(
            "rain_graupel_tables must contain 5 arrays, got "
            f"{len(rain_graupel_values)}")
    for name, table in zip(
            rain_snow_names, rain_snow_values, strict=True):
        _validate_fp64_fortran_table(name, table, (37, 9, 37, 37))
    for name, table in zip(
            rain_graupel_names, rain_graupel_values, strict=True):
        _validate_fp64_fortran_table(name, table, (37, 37, 1, 37, 37))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    device_id = getattr(getattr(qc, "device", None), "id", None)
    threads = 64 if device_id is None else _warm_network_threads(device_id)
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_warm_frozen_source_network")(
        (blocks,), (threads,),
        (qc, qr, nr, qs, qg, graupel_number_shadow,
         graupel_melt_marker, snow_melt_marker,
         temperature, pressure, qv,
         rain_cloud_efficiency, snow_cloud_efficiency,
         *rain_snow_values, *rain_graupel_values,
         DTYPE(dt), np.int32(size)))


def launch_warm_frozen_source_network_from_owner(
        qc, qr, nr, qs, qg, graupel_number_shadow,
        graupel_melt_marker, snow_melt_marker,
        temperature, pressure, qv, table_owner, dt: float) -> None:
    """Launch the complete ambient-warm source path from one table owner."""
    from woof.core.thompson_runtime import DeviceClassicTableSet

    if (not isinstance(table_owner, DeviceClassicTableSet)
            or not table_owner.roundtrip_verified):
        raise TypeError(
            "table_owner must be a verified DeviceClassicTableSet")
    tables = table_owner.cold_source_tables
    launch_warm_frozen_source_network(
        qc, qr, nr, qs, qg, graupel_number_shadow,
        graupel_melt_marker, snow_melt_marker,
        temperature, pressure, qv,
        tables.rain_cloud_efficiency, table_owner.t_Efsw,
        tables.rain_snow_tables, tables.rain_graupel_tables, dt)


def launch_final_phase_cleanup(
        qc, qi, ni, temperature, pressure, qv, *,
        micro_columns=None) -> None:
    """Apply Thompson's post-fallout instantaneous cloud phase cleanup.

    ``micro_columns`` (the ``(ny, nx)`` flag
    :func:`launch_microphysics_columns` writes at entry) adds WRF's terminal
    vapour floor, 1.E-10 at every level of a column with microphysics
    (:3974), and leaves a column WRF returned from at :2020 with only the
    entry rewrite's zero cloud and ice.  The production adapter always
    passes it.
    """
    shape, size = _validate_fields({
        "qc": qc,
        "qi": qi,
        "ni": ni,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    threads = 256
    blocks = (size + threads - 1) // threads
    if micro_columns is None:
        get_kernel("thompson", "thompson_final_phase_cleanup")(
            (blocks,), (threads,),
            (qc, qi, ni, temperature, pressure, qv, np.int32(size)))
        return
    if len(shape) != 3 or micro_columns.shape != shape[1:]:
        raise ValueError(f"micro_columns must have shape {shape[1:]}, "
                         f"got {micro_columns.shape}")
    _validate_fields({"micro_columns": micro_columns})
    get_kernel("thompson", "thompson_final_phase_cleanup_with_columns")(
        (blocks,), (threads,),
        (qc, qi, ni, temperature, pressure, qv, micro_columns,
         np.int32(shape[1] * shape[2]), np.int32(size)))


def launch_microphysics_columns(
        qc, qi, qr, qs, qg, temperature, pressure, qv,
        micro_columns) -> None:
    """WRF's per-column ``no_micro`` decision, :1646, :1827-1990, :2020.

    Writes 1.0 into ``micro_columns`` (``(ny, nx)`` float32) for a column
    with microphysics and 0.0 for one ``mp_thompson`` leaves at :2020: every
    entry cloud, ice, rain, snow and graupel mixing ratio at or below R1 and
    no level supersaturated over ice.  Call it on the entry state.
    """
    shape, _ = _validate_fields({
        "qc": qc, "qi": qi, "qr": qr, "qs": qs, "qg": qg,
        "temperature": temperature, "pressure": pressure, "qv": qv,
    })
    if len(shape) != 3:
        raise ValueError(f"fields must be (nz, ny, nx), got {shape}")
    nz, ny, nx = shape
    if micro_columns.shape != (ny, nx):
        raise ValueError(f"micro_columns must have shape {(ny, nx)}, "
                         f"got {micro_columns.shape}")
    _validate_fields({"micro_columns": micro_columns})
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    get_kernel("thompson", "thompson_microphysics_columns")(
        (blocks,), (_COLUMN_TPB,),
        (qc, qi, qr, qs, qg, temperature, pressure, qv, micro_columns,
         np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_ice_nucleation(
        qi, ni, temperature, pressure, qv, dt: float) -> None:
    """Apply classic non-aerosol Thompson deposition nucleation.

    This is WRF option 8's Cooper temperature-dependent branch.  Aerosol-
    aware DeMott/Phillips/Koop paths belong to option 28 and are deliberately
    outside the classic ``mp_physics=8`` contract.
    """
    _, size = _validate_fields({
        "qi": qi,
        "ni": ni,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_ice_nucleation")(
        (blocks,), (threads,),
        (qi, ni, temperature, pressure, qv,
         DTYPE(dt), np.int32(size)))


def launch_rain_freezing(
        qr, nr, qi, ni, qg, temperature, pressure, qv,
        rain_to_ice_mass, rain_to_ice_number,
        rain_to_graupel_mass, rain_to_graupel_number,
        dt: float) -> None:
    """Apply classic Thompson's table-driven freezing of rain drops.

    The four tables are the canonical FP64 Fortran-ordered ``tpi_qrfz``,
    ``tni_qrfz``, ``tpg_qrfz``, and ``tnr_qrfz`` arrays.  This admitted
    slice includes the classic 1000-m^-3 ice-nuclei lookup, homogeneous
    freezing below ``HGFR``, category mass/number transfer, and WRF's
    post-process rain/ice size bounds.
    """
    _, size = _validate_fields({
        "qr": qr,
        "nr": nr,
        "qi": qi,
        "ni": ni,
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    for name, table in (
            ("rain_to_ice_mass", rain_to_ice_mass),
            ("rain_to_ice_number", rain_to_ice_number),
            ("rain_to_graupel_mass", rain_to_graupel_mass),
            ("rain_to_graupel_number", rain_to_graupel_number)):
        _validate_fp64_fortran_table(name, table, (37, 37, 45, 55))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_rain_freezing")(
        (blocks,), (threads,),
        (qr, nr, qi, ni, qg, temperature, pressure, qv,
         rain_to_ice_mass, rain_to_ice_number,
         rain_to_graupel_mass, rain_to_graupel_number,
         DTYPE(dt), np.int32(size)))


def launch_cloud_freezing(
        qc, qi, ni, temperature, pressure, qv,
        cloud_to_ice_mass, cloud_to_ice_number,
        dt: float) -> None:
    """Apply classic Thompson's table-driven freezing of cloud droplets."""
    _, size = _validate_fields({
        "qc": qc,
        "qi": qi,
        "ni": ni,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    _validate_fp64_fortran_table(
        "cloud_to_ice_mass", cloud_to_ice_mass, (37, 100, 45, 55))
    _validate_fp64_fortran_table(
        "cloud_to_ice_number", cloud_to_ice_number, (37, 100, 45, 55))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_cloud_freezing")(
        (blocks,), (threads,),
        (qc, qi, ni, temperature, pressure, qv,
         cloud_to_ice_mass, cloud_to_ice_number,
         DTYPE(dt), np.int32(size)))


def launch_cold_cloud_source_network(
        qc, qr, nr, qi, ni, qs, qg, temperature, pressure, qv,
        rain_cloud_efficiency, cloud_to_ice_mass, cloud_to_ice_number,
        dt: float) -> None:
    """Apply the simultaneous classic cold cloud-water source group.

    Autoconversion, rain accretion/self-collection, table cloud freezing,
    snow/graupel riming, and Hallett-Mossop splinters all use one incoming
    state.  WRF's shared cloud-water mass limiter is applied exactly once;
    its intentionally held number and splinter rates are preserved.
    """
    _, size = _validate_fields({
        "qc": qc,
        "qr": qr,
        "nr": nr,
        "qi": qi,
        "ni": ni,
        "qs": qs,
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    _validate_fp64_fortran_table(
        "rain_cloud_efficiency", rain_cloud_efficiency, (100, 100))
    _validate_fp64_fortran_table(
        "cloud_to_ice_mass", cloud_to_ice_mass, (37, 100, 45, 55))
    _validate_fp64_fortran_table(
        "cloud_to_ice_number", cloud_to_ice_number, (37, 100, 45, 55))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_cold_cloud_source_network")(
        (blocks,), (threads,),
        (qc, qr, nr, qi, ni, qs, qg, temperature, pressure, qv,
         rain_cloud_efficiency, cloud_to_ice_mass, cloud_to_ice_number,
         DTYPE(dt), np.int32(size)))


def launch_graupel_cloud_riming(
        qc, qg, qi, ni, temperature, pressure, qv, dt: float) -> None:
    """Apply cold graupel-cloud riming and Hallett-Mossop splintering.

    This is an independently admitted classic-mp8 slice.  Warm wet-growth
    collection remains outside the slice and the complete Thompson driver
    remains fail-closed.
    """
    _, size = _validate_fields({
        "qc": qc,
        "qg": qg,
        "qi": qi,
        "ni": ni,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_graupel_cloud_riming")(
        (blocks,), (threads,),
        (qc, qg, qi, ni, temperature, pressure, qv,
         DTYPE(dt), np.int32(size)))


def launch_snow_cloud_riming(
        qc, qs, temperature, pressure, qv, dt: float) -> None:
    """Apply classic Thompson cold snow-cloud collection on the GPU.

    Deposition-conditioned partial snow-to-graupel conversion remains outside
    this independently admitted slice.
    """
    _, size = _validate_fields({
        "qc": qc,
        "qs": qs,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_snow_cloud_riming")(
        (blocks,), (threads,),
        (qc, qs, temperature, pressure, qv,
         DTYPE(dt), np.int32(size)))


def launch_snow_rime_conversion(
        qc, qs, qg, temperature, pressure, qv,
        velocity_boost, dt: float) -> None:
    """Apply deposition-conditioned partial rimed-snow conversion.

    This final isolated classic-process slice evaluates snow/cloud riming and
    snow vapor exchange from the same incoming state, partitions sufficiently
    dense riming into graupel, and records WRF's same-call snow fall-speed
    boost in caller-owned FP32 scratch.
    """
    _, size = _validate_fields({
        "qc": qc,
        "qs": qs,
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "velocity_boost": velocity_boost,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_snow_rime_conversion")(
        (blocks,), (threads,),
        (qc, qs, qg, temperature, pressure, qv, velocity_boost,
         DTYPE(dt), np.int32(size)))


def launch_snow_ice_collection(
        qi, ni, qs, temperature, pressure, qv, dt: float) -> None:
    """Apply classic Thompson snow collection of cloud ice on the GPU."""
    _, size = _validate_fields({
        "qi": qi,
        "ni": ni,
        "qs": qs,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_snow_ice_collection")(
        (blocks,), (threads,),
        (qi, ni, qs, temperature, pressure, qv,
         DTYPE(dt), np.int32(size)))


def launch_rain_ice_collection(
        qr, nr, qi, ni, qg, temperature, pressure, qv, dt: float) -> None:
    """Apply classic Thompson rain/ice collection and rain self-collection."""
    _, size = _validate_fields({
        "qr": qr,
        "nr": nr,
        "qi": qi,
        "ni": ni,
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_rain_ice_collection")(
        (blocks,), (threads,),
        (qr, nr, qi, ni, qg, temperature, pressure, qv,
         DTYPE(dt), np.int32(size)))


def launch_rain_snow_collection(
        qr, nr, qs, qg, temperature, pressure, qv,
        tables, dt: float) -> None:
    """Apply classic Thompson's table-driven cold rain/snow collision.

    ``tables`` is the ordered twelve-table ``qr_acr_qsV2`` asset set from a
    validated classic Thompson table bundle.  Every table is canonical FP64
    Fortran order with shape ``(37, 9, 37, 37)``.  The slice also applies the
    simultaneous rain self-collection rate diagnosed from the incoming state.
    """
    _, size = _validate_fields({
        "qr": qr,
        "nr": nr,
        "qs": qs,
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    table_names = (
        "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
        "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
        "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
    )
    try:
        table_values = tuple(tables)
    except TypeError as exc:
        raise TypeError("tables must be an iterable of twelve arrays") from exc
    if len(table_values) != len(table_names):
        raise ValueError(
            f"tables must contain {len(table_names)} arrays, "
            f"got {len(table_values)}")
    for name, table in zip(table_names, table_values, strict=True):
        _validate_fp64_fortran_table(name, table, (37, 9, 37, 37))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_rain_snow_collection")(
        (blocks,), (threads,),
        (qr, nr, qs, qg, temperature, pressure, qv,
         *table_values, DTYPE(dt), np.int32(size)))


def launch_rain_graupel_collection(
        qr, nr, qg, temperature, pressure, qv,
        tables, dt: float) -> None:
    """Apply classic Thompson's table-driven cold rain/graupel collision.

    ``tables`` is the ordered five-table ``qr_acr_qg_V4`` asset set from a
    validated classic Thompson bundle.  Every table is canonical FP64
    Fortran order with shape ``(37, 37, 1, 37, 37)``.  The CUDA slice reads
    the one graupel-density slab the tables hold (thompson_racg_index), not
    WRF v4.6.1's out-of-bounds idx_bg1=5 subscript: a declared divergence.
    """
    _, size = _validate_fields({
        "qr": qr,
        "nr": nr,
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    table_names = (
        "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
    )
    try:
        table_values = tuple(tables)
    except TypeError as exc:
        raise TypeError("tables must be an iterable of five arrays") from exc
    if len(table_values) != len(table_names):
        raise ValueError(
            f"tables must contain {len(table_names)} arrays, "
            f"got {len(table_values)}")
    for name, table in zip(table_names, table_values, strict=True):
        _validate_fp64_fortran_table(name, table, (37, 37, 1, 37, 37))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_rain_graupel_collection")(
        (blocks,), (threads,),
        (qr, nr, qg, temperature, pressure, qv,
         *table_values, DTYPE(dt), np.int32(size)))


def launch_cold_rain_snow_graupel_network(
        qr, nr, qs, qg, temperature, pressure, qv,
        rain_snow_tables, rain_graupel_tables, dt: float) -> None:
    """Apply the simultaneous classic cold-rain collision network.

    Unlike the independently admitted process launchers, this production
    slice diagnoses rain/snow collision, rain/graupel collision, and rain
    self-collection from one incoming state.  It then applies classic WRF's
    shared rain-mass limiter once.  Number sinks intentionally retain WRF's
    unscaled semantics when that mass limiter activates.

    ``rain_snow_tables`` contains the twelve canonical ``qr_acr_qsV2``
    arrays with shape ``(37, 9, 37, 37)``.  ``rain_graupel_tables`` contains
    the five canonical ``qr_acr_qg_V4`` arrays with shape
    ``(37, 37, 1, 37, 37)``.  All arrays must be FP64 Fortran-contiguous
    device arrays from the validated Thompson runtime owner.
    """
    _, size = _validate_fields({
        "qr": qr,
        "nr": nr,
        "qs": qs,
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    rain_snow_names = (
        "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
        "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
        "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
    )
    rain_graupel_names = (
        "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
    )
    try:
        rain_snow_values = tuple(rain_snow_tables)
    except TypeError as exc:
        raise TypeError(
            "rain_snow_tables must be an iterable of twelve arrays") from exc
    try:
        rain_graupel_values = tuple(rain_graupel_tables)
    except TypeError as exc:
        raise TypeError(
            "rain_graupel_tables must be an iterable of five arrays") from exc
    if len(rain_snow_values) != len(rain_snow_names):
        raise ValueError(
            f"rain_snow_tables must contain {len(rain_snow_names)} arrays, "
            f"got {len(rain_snow_values)}")
    if len(rain_graupel_values) != len(rain_graupel_names):
        raise ValueError(
            "rain_graupel_tables must contain "
            f"{len(rain_graupel_names)} arrays, "
            f"got {len(rain_graupel_values)}")
    for name, table in zip(
            rain_snow_names, rain_snow_values, strict=True):
        _validate_fp64_fortran_table(name, table, (37, 9, 37, 37))
    for name, table in zip(
            rain_graupel_names, rain_graupel_values, strict=True):
        _validate_fp64_fortran_table(name, table, (37, 37, 1, 37, 37))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_cold_rain_snow_graupel_network")(
        (blocks,), (threads,),
        (qr, nr, qs, qg, temperature, pressure, qv,
         *rain_snow_values, *rain_graupel_values,
         DTYPE(dt), np.int32(size)))


def launch_cold_rain_source_network(
        qr, nr, qi, ni, qs, qg, temperature, pressure, qv,
        rain_snow_tables, rain_graupel_tables, rain_freezing_tables,
        dt: float) -> None:
    """Apply the complete simultaneous classic cold-rain source group.

    The kernel diagnoses table-driven rain freezing, rain/ice collection,
    rain/snow collision, rain/graupel collision, and rain self-collection
    from one incoming state.  It preserves WRF's separate cloud-ice bound,
    shared rain-mass bound, held number rates, and legacy post-bound mass-pair
    behavior before updating all categories once.
    """
    _, size = _validate_fields({
        "qr": qr,
        "nr": nr,
        "qi": qi,
        "ni": ni,
        "qs": qs,
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    rain_snow_names = (
        "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
        "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
        "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
    )
    rain_graupel_names = (
        "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
    )
    rain_freezing_names = (
        "rain_to_ice_mass", "rain_to_ice_number",
        "rain_to_graupel_mass", "rain_to_graupel_number",
    )
    try:
        rain_snow_values = tuple(rain_snow_tables)
        rain_graupel_values = tuple(rain_graupel_tables)
        rain_freezing_values = tuple(rain_freezing_tables)
    except TypeError as exc:
        raise TypeError("all table groups must be iterable") from exc
    expected_lengths = (
        ("rain_snow_tables", rain_snow_values, rain_snow_names),
        ("rain_graupel_tables", rain_graupel_values,
         rain_graupel_names),
        ("rain_freezing_tables", rain_freezing_values,
         rain_freezing_names),
    )
    for label, values, names in expected_lengths:
        if len(values) != len(names):
            raise ValueError(
                f"{label} must contain {len(names)} arrays, "
                f"got {len(values)}")
    for name, table in zip(
            rain_snow_names, rain_snow_values, strict=True):
        _validate_fp64_fortran_table(name, table, (37, 9, 37, 37))
    for name, table in zip(
            rain_graupel_names, rain_graupel_values, strict=True):
        _validate_fp64_fortran_table(name, table, (37, 37, 1, 37, 37))
    for name, table in zip(
            rain_freezing_names, rain_freezing_values, strict=True):
        _validate_fp64_fortran_table(name, table, (37, 37, 45, 55))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_cold_rain_source_network")(
        (blocks,), (threads,),
        (qr, nr, qi, ni, qs, qg, temperature, pressure, qv,
         *rain_snow_values, *rain_graupel_values, *rain_freezing_values,
         DTYPE(dt), np.int32(size)))


def launch_effective_radius(
        temperature, pressure, qv, qc, qi, ni, qs, effc, effi, effs,
        ) -> None:
    """Run the classic-mp8 WRF effective-radius diagnostic on the GPU."""
    fields = {
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "qc": qc,
        "qi": qi,
        "ni": ni,
        "qs": qs,
        "effc": effc,
        "effi": effi,
        "effs": effs,
    }
    _, size = _validate_fields(fields)
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_effective_radius")(
        (blocks,), (threads,),
        (temperature, pressure, qv, qc, qi, ni, qs,
         effc, effi, effs, np.int32(size)))


def launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz, rainnc, rainncv,
        dt: float, *, reference_density=None,
        accumulate_surface: bool = False,
        density_carries_rain_presence: bool = False) -> None:
    """Apply the independently admitted WRF two-moment rain fallout slice.

    Volume fields are contiguous FP32 ``(nz, ny, nx)`` arrays and surface
    precipitation fields are contiguous FP32 ``(ny, nx)`` arrays.  This
    launcher intentionally covers sedimentation only; the complete Thompson
    process driver remains fail-closed.  ``reference_density`` is optional;
    the evaporation composition uses it to preserve WRF's pre-evaporation
    volumetric rain state while fallout uses the updated environmental density.
    ``density_carries_rain_presence`` says that density is ZERO exactly where
    WRF's L_qr failed (the mp=28 rain evaporation writes it so); the fallout
    then takes L_qr from it instead of from the post-evaporation mixing ratio.
    """
    if density_carries_rain_presence and reference_density is None:
        raise ValueError("density_carries_rain_presence needs the "
                         "reference_density that carries it")
    fields = {
        "qr": qr,
        "nr": nr,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "dz": dz,
    }
    if reference_density is not None:
        fields["reference_density"] = reference_density
    shape, _ = _validate_fields(fields)
    if len(shape) != 3:
        raise ValueError(f"Thompson rain fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(f"Thompson rain sedimentation requires 2 <= nz <= "
                         f"{_KMAX}, got {nz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if not isinstance(accumulate_surface, (bool, np.bool_)):
        raise TypeError("accumulate_surface must be boolean")
    surface_shape = (ny, nx)
    for name, value in (("rainnc", rainnc), ("rainncv", rainncv)):
        if value.shape != surface_shape:
            raise ValueError(f"{name} must have shape {surface_shape}, "
                             f"got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    kernel_name = ("thompson_rain_sediment_64" if nz <= _SHALLOW_KMAX
                   else "thompson_rain_sediment_256")
    if density_carries_rain_presence:
        kernel_name += "_with_presence"
    elif reference_density is not None:
        kernel_name += "_with_density"
    arguments = (qr, nr, temperature, pressure, qv)
    if reference_density is not None:
        arguments += (reference_density,)
    arguments += (dz, rainnc, rainncv, np.int32(accumulate_surface),
                  DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx))
    _launch_fallout(
        "thompson_rain_sediment_levels_64_with_presence", kernel_name,
        LEVEL_PARALLEL_FALLOUT and nz <= _SHALLOW_KMAX
        and density_carries_rain_presence, ny * nx, arguments)


def launch_ice_sedimentation(
        qi, ni, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt: float, *,
        reference_density=None) -> None:
    """Apply the independently admitted WRF cloud-ice fallout slice.

    ``reference_density`` preserves WRF's pre-evaporation volumetric ice
    state when this process is composed with simultaneous rain evaporation.
    """
    fields = {
        "qi": qi,
        "ni": ni,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "dz": dz,
    }
    if reference_density is not None:
        fields["reference_density"] = reference_density
    shape, _ = _validate_fields(fields)
    if len(shape) != 3:
        raise ValueError(f"Thompson ice fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(f"Thompson ice sedimentation requires 2 <= nz <= "
                         f"{_KMAX}, got {nz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    surface_shape = (ny, nx)
    surface_fields = {
        "rainnc": rainnc,
        "rainncv": rainncv,
        "snownc": snownc,
        "snowncv": snowncv,
    }
    for name, value in surface_fields.items():
        if value.shape != surface_shape:
            raise ValueError(f"{name} must have shape {surface_shape}, "
                             f"got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    kernel_name = ("thompson_ice_sediment_64" if nz <= _SHALLOW_KMAX
                   else "thompson_ice_sediment_256")
    if reference_density is not None:
        kernel_name += "_with_density"
    arguments = (qi, ni, temperature, pressure, qv)
    if reference_density is not None:
        arguments += (reference_density,)
    arguments += (dz, rainnc, rainncv, snownc, snowncv,
                  DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx))
    _launch_fallout(
        "thompson_ice_sediment_levels_64_with_density", kernel_name,
        LEVEL_PARALLEL_FALLOUT and nz <= _SHALLOW_KMAX
        and reference_density is not None, ny * nx, arguments)


def launch_cloud_sedimentation(
        qc, temperature, pressure, qv, vertical_velocity, dz,
        dt: float, *, reference_density=None,
        rain_active_columns=None, cloud_active_columns=None) -> None:
    """Apply WRF cloud-water fallout.

    ``reference_density`` preserves the volumetric cloud mass formed just
    before WRF's saturation adjustment while tendency conversion uses the
    updated environmental density.  ``rain_active_columns`` optionally holds
    WRF's post-source ``ANY(L_qr)`` mask: those columns refresh the fall-speed
    density in the preceding rain pass; all other columns retain the held
    density.  The latter is valid only with ``reference_density``.
    ``cloud_active_columns`` optionally carries WRF's held post-source,
    pre-adjustment ``ANY(L_qc)`` guard so newly condensed cloud in an
    entry-empty column waits until the next microphysics call to fall.
    """
    shape, _ = _validate_fields({
        "qc": qc,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "vertical_velocity": vertical_velocity,
        "dz": dz,
    })
    if reference_density is not None:
        _validate_fields({
            "qc": qc,
            "reference_density": reference_density,
        })
    if len(shape) != 3:
        raise ValueError(f"Thompson cloud fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(f"Thompson cloud sedimentation requires 2 <= nz <= "
                         f"{_KMAX}, got {nz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    if rain_active_columns is not None:
        if reference_density is None:
            raise ValueError(
                "rain_active_columns requires reference_density")
        surface_shape = (ny, nx)
        if rain_active_columns.shape != surface_shape:
            raise ValueError(
                f"rain_active_columns must have shape {surface_shape}, "
                f"got {rain_active_columns.shape}")
        if rain_active_columns.dtype != DTYPE:
            raise TypeError(
                "rain_active_columns must be float32, got "
                f"{rain_active_columns.dtype}")
        if not rain_active_columns.flags.c_contiguous:
            raise ValueError("rain_active_columns must be C-contiguous")
    if cloud_active_columns is not None:
        if reference_density is None or rain_active_columns is None:
            raise ValueError(
                "cloud_active_columns requires reference_density and "
                "rain_active_columns")
        surface_shape = (ny, nx)
        if cloud_active_columns.shape != surface_shape:
            raise ValueError(
                f"cloud_active_columns must have shape {surface_shape}, "
                f"got {cloud_active_columns.shape}")
        if cloud_active_columns.dtype != DTYPE:
            raise TypeError(
                "cloud_active_columns must be float32, got "
                f"{cloud_active_columns.dtype}")
        if not cloud_active_columns.flags.c_contiguous:
            raise ValueError("cloud_active_columns must be C-contiguous")

    kernel_name = ("thompson_cloud_sediment_64" if nz <= _SHALLOW_KMAX
                   else "thompson_cloud_sediment_256")
    if cloud_active_columns is not None:
        kernel_name += "_with_density_and_masks"
    elif rain_active_columns is not None:
        kernel_name += "_with_density_and_rain"
    elif reference_density is not None:
        kernel_name += "_with_density"
    arguments = (qc, temperature, pressure, qv)
    if reference_density is not None:
        arguments += (reference_density,)
    if rain_active_columns is not None:
        arguments += (rain_active_columns,)
    if cloud_active_columns is not None:
        arguments += (cloud_active_columns,)
    arguments += (vertical_velocity, dz, DTYPE(dt),
                  np.int32(nz), np.int32(ny), np.int32(nx))
    _launch_fallout(
        "thompson_cloud_sediment_levels_64_with_density_and_masks",
        kernel_name,
        LEVEL_PARALLEL_FALLOUT and nz <= _SHALLOW_KMAX
        and cloud_active_columns is not None, ny * nx, arguments)


def launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt: float, *,
        accumulate_surface: bool = False,
        snow_melt_marker=None, melt_rain_qr=None, melt_rain_nr=None,
        reference_density=None, reference_temperature=None,
        velocity_boost=None, melt_rain_density=None,
        melt_rain_density_carries_presence: bool = False) -> None:
    """Apply the independently admitted WRF one-moment snow fallout slice.

    Set ``accumulate_surface`` only when an earlier frozen-species fallout
    kernel already populated this step's ``RAINNCV``/``SNOWNCV`` values.
    ``reference_density`` preserves WRF's pre-evaporation volumetric snow
    state while fallout uses the updated environmental density.
    ``reference_temperature`` preserves WRF's held pre-adjustment snow
    moments and requires ``reference_density``.
    ``velocity_boost`` carries WRF's deposition-conditioned rime boost.
    ``snow_melt_marker`` is the held ``prr_sml > 0`` decision; it gates the
    WRF snow/rain velocity blend while ``melt_rain_qr``/``melt_rain_nr``
    provide the post-source rain distribution.  The production path combines
    all three with held state, as WRF does in one snow-fallout call.
    ``melt_rain_density`` is the rain fallout's own reference density (the
    buffer ``launch_rain_sedimentation`` is handed): with it the blend reads
    WRF's rr(k) and the rain fall speed vtrk(k) as the rain pass forms them
    (:3612-3634), a level without rain inheriting the speed above it, as
    :3722-3724 do.  ``melt_rain_density_carries_presence`` matches
    ``launch_rain_sedimentation``'s ``density_carries_rain_presence``.
    """
    fields = {
        "qs": qs,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "dz": dz,
    }
    melt_inputs = (snow_melt_marker, melt_rain_qr, melt_rain_nr)
    if any(value is None for value in melt_inputs) and not all(
            value is None for value in melt_inputs):
        raise ValueError(
            "snow_melt_marker, melt_rain_qr, and melt_rain_nr must be "
            "supplied together")
    if melt_rain_qr is not None:
        fields["snow_melt_marker"] = snow_melt_marker
        fields["melt_rain_qr"] = melt_rain_qr
        fields["melt_rain_nr"] = melt_rain_nr
    if reference_density is not None:
        fields["reference_density"] = reference_density
    if reference_temperature is not None:
        if reference_density is None:
            raise ValueError(
                "reference_temperature requires reference_density")
        fields["reference_temperature"] = reference_temperature
    if velocity_boost is not None:
        if reference_temperature is None:
            raise ValueError(
                "velocity_boost requires reference_temperature")
        fields["velocity_boost"] = velocity_boost
    if melt_rain_density is not None:
        if melt_rain_qr is None or velocity_boost is None:
            raise ValueError(
                "melt_rain_density requires the melt-rain inputs and "
                "velocity_boost")
        fields["melt_rain_density"] = melt_rain_density
    elif melt_rain_density_carries_presence:
        raise ValueError("melt_rain_density_carries_presence needs the "
                         "melt_rain_density that carries it")
    shape, _ = _validate_fields(fields)
    if (snow_melt_marker is not None
            and _arrays_overlap(snow_melt_marker, qs)):
        raise ValueError("snow_melt_marker must not alias qs")
    if len(shape) != 3:
        raise ValueError(f"Thompson snow fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(f"Thompson snow sedimentation requires 2 <= nz <= "
                         f"{_KMAX}, got {nz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if not isinstance(accumulate_surface, (bool, np.bool_)):
        raise TypeError("accumulate_surface must be boolean")
    surface_shape = (ny, nx)
    surface_fields = {
        "rainnc": rainnc,
        "rainncv": rainncv,
        "snownc": snownc,
        "snowncv": snowncv,
    }
    for name, value in surface_fields.items():
        if value.shape != surface_shape:
            raise ValueError(f"{name} must have shape {surface_shape}, "
                             f"got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    kernel_name = ("thompson_snow_sediment_64" if nz <= _SHALLOW_KMAX
                   else "thompson_snow_sediment_256")
    if melt_rain_density is not None:
        kernel_name += "_with_rain_profile"
        if melt_rain_density_carries_presence:
            kernel_name += "_and_presence"
    elif velocity_boost is not None and melt_rain_qr is not None:
        kernel_name += "_with_melt_rain_and_state_and_boost"
    elif velocity_boost is not None:
        kernel_name += "_with_state_and_boost"
    elif melt_rain_qr is not None and reference_temperature is not None:
        kernel_name += "_with_melt_rain_and_state"
    elif melt_rain_qr is not None and reference_density is not None:
        kernel_name += "_with_melt_rain_and_density"
    elif melt_rain_qr is not None:
        kernel_name += "_with_melt_rain"
    elif reference_temperature is not None:
        kernel_name += "_with_state"
    elif reference_density is not None:
        kernel_name += "_with_density"
    arguments = (qs,)
    if melt_rain_qr is not None:
        arguments += (snow_melt_marker, melt_rain_qr, melt_rain_nr)
    if melt_rain_density is not None:
        arguments += (melt_rain_density,)
    arguments += (temperature, pressure, qv)
    if reference_density is not None:
        arguments += (reference_density,)
    if reference_temperature is not None:
        arguments += (reference_temperature,)
    if velocity_boost is not None:
        arguments += (velocity_boost,)
    arguments += (dz,
                  rainnc, rainncv, snownc, snowncv,
                  np.int32(accumulate_surface), DTYPE(dt),
                  np.int32(nz), np.int32(ny), np.int32(nx))
    _launch_fallout(
        "thompson_snow_sediment_levels_64_with_rain_profile_and_presence",
        kernel_name,
        LEVEL_PARALLEL_FALLOUT and nz <= _SHALLOW_KMAX
        and melt_rain_density_carries_presence, ny * nx, arguments)


def launch_graupel_fallout_column_mask(
        entry_active, qg, active_columns) -> None:
    """Resolve classic WRF's held graupel fallout activity by column.

    In WRF's fixed-density classic branch, ``L_qg(k)`` is true for fallout
    only when graupel was active on entry *and* remains active after the
    source update (WRF-v4.6.1 ``module_mp_thompson.F`` 1917-1948 and
    3264-3303).  Sedimentation is then guarded by ``ANY(L_qg)`` over the
    complete column (3902-3938), so a qualifying level enables every updated
    graupel level in that column.  ``entry_active`` is a contiguous FP32
    zero/one marker with shape ``(nz, ny, nx)``; accepting FP32 lets the
    coupled adapter lifetime-alias it with a reference-state buffer that is
    overwritten immediately after this launch.  ``active_columns`` is
    contiguous FP32 zero/one ``(ny, nx)`` output, allowing another coupled
    adapter lifetime alias with its not-yet-refreshed ``SR`` diagnostic.
    """
    shape, _ = _validate_fields({"entry_active": entry_active, "qg": qg})
    if len(shape) != 3:
        raise ValueError(f"Thompson graupel fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    surface_shape = (ny, nx)
    if active_columns.shape != surface_shape:
        raise ValueError(
            f"active_columns must have shape {surface_shape}, "
            f"got {active_columns.shape}")
    if active_columns.dtype != DTYPE:
        raise TypeError(
            f"active_columns must be float32, got {active_columns.dtype}")
    if not active_columns.flags.c_contiguous:
        raise ValueError("active_columns must be C-contiguous")
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    get_kernel("thompson", "thompson_graupel_fallout_column_mask")(
        (blocks,), (_COLUMN_TPB,),
        (entry_active, qg, active_columns,
         np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_hydrometeor_column_mask(
        mixing_ratio, active_columns) -> None:
    """Write a zero/one column mask for post-source hydrometeor activity."""
    shape, _ = _validate_fields({"mixing_ratio": mixing_ratio})
    if len(shape) != 3:
        raise ValueError(f"Thompson hydrometeor field must be 3-D, got {shape}")
    nz, ny, nx = shape
    surface_shape = (ny, nx)
    if active_columns.shape != surface_shape:
        raise ValueError(
            f"active_columns must have shape {surface_shape}, "
            f"got {active_columns.shape}")
    if active_columns.dtype != DTYPE:
        raise TypeError(
            f"active_columns must be float32, got {active_columns.dtype}")
    if not active_columns.flags.c_contiguous:
        raise ValueError("active_columns must be C-contiguous")
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    get_kernel("thompson", "thompson_hydrometeor_column_mask")(
        (blocks,), (_COLUMN_TPB,),
        (mixing_ratio, active_columns,
         np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_classic_graupel_number_init(
        qg, temperature, pressure, qv, graupel_number_shadow) -> None:
    """Diagnose classic WRF's transient per-call graupel number moment."""
    shape, size = _validate_fields({
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "graupel_number_shadow": graupel_number_shadow,
    })
    if len(shape) != 3:
        raise ValueError(f"Thompson graupel fields must be 3-D, got {shape}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_classic_graupel_number_init")(
        (blocks,), (threads,),
        (qg, temperature, pressure, qv, graupel_number_shadow,
         np.int32(size)))


def launch_classic_graupel_number_finalize(
        qg, temperature, pressure, qv, graupel_number_shadow) -> None:
    """Apply WRF's one final classic-ng bound after all call tendencies.

    Also WRF's terminal graupel zero (module_mp_thompson.F:4058-4063): a
    graupel mixing ratio at or below R1 is written as zero, mass and number,
    in every column -- ``qg`` is written in place.
    """
    shape, size = _validate_fields({
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "graupel_number_shadow": graupel_number_shadow,
    })
    if len(shape) != 3:
        raise ValueError(f"Thompson graupel fields must be 3-D, got {shape}")
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("thompson", "thompson_classic_graupel_number_finalize")(
        (blocks,), (threads,),
        (qg, temperature, pressure, qv, graupel_number_shadow,
         np.int32(size)))


def launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, dt: float, *,
        graupel_number=None, reference_density=None,
        active_columns=None, graupel_number_shadow=None,
        accumulate_surface: bool = False) -> None:
    """Apply the independently admitted WRF classic-graupel fallout slice.

    ``graupel_number`` supplies the evolved same-call scratch moment after a
    process such as melting; otherwise classic mp=8 diagnoses it from mass.
    ``reference_density`` preserves WRF's pre-evaporation volumetric graupel
    state while fallout uses the updated environmental density.
    ``active_columns`` optionally carries classic WRF's held ``ANY(L_qg)``
    fallout guard.  The guarded path is intentionally narrow: it requires the
    classic diagnostic-number branch and held density used by the coupled
    adapter, leaving every previously admitted kernel byte-for-byte intact.
    ``graupel_number_shadow`` optionally evolves classic WRF's private
    same-call ``ng1d`` moment alongside mass fallout for an output-due
    REFL_10CM diagnosis.  It never changes the mass velocity or transported
    state and is only admitted with the guarded coupled-adapter path.
    Set ``accumulate_surface`` when an earlier species already populated the
    current-step total precipitation fields.
    """
    fields = {
        "qg": qg,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "dz": dz,
    }
    if graupel_number is not None:
        fields["graupel_number"] = graupel_number
    if reference_density is not None:
        fields["reference_density"] = reference_density
    if graupel_number_shadow is not None:
        fields["graupel_number_shadow"] = graupel_number_shadow
    shape, _ = _validate_fields(fields)
    if len(shape) != 3:
        raise ValueError(f"Thompson graupel fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(f"Thompson graupel sedimentation requires "
                         f"2 <= nz <= {_KMAX}, got {nz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if not isinstance(accumulate_surface, (bool, np.bool_)):
        raise TypeError("accumulate_surface must be boolean")
    surface_shape = (ny, nx)
    if active_columns is not None:
        if graupel_number is not None or reference_density is None:
            raise ValueError(
                "active_columns requires classic graupel without an explicit "
                "number moment and with reference_density")
        if active_columns.shape != surface_shape:
            raise ValueError(
                f"active_columns must have shape {surface_shape}, "
                f"got {active_columns.shape}")
        if active_columns.dtype != DTYPE:
            raise TypeError(
                f"active_columns must be float32, got {active_columns.dtype}")
        if not active_columns.flags.c_contiguous:
            raise ValueError("active_columns must be C-contiguous")
    if graupel_number_shadow is not None:
        if active_columns is None or reference_density is None:
            raise ValueError(
                "graupel_number_shadow requires active_columns and "
                "reference_density")
        if graupel_number is not None:
            raise ValueError(
                "graupel_number_shadow cannot be combined with the explicit "
                "graupel_number mass-velocity branch")
    surface_fields = {
        "rainnc": rainnc,
        "rainncv": rainncv,
        "graupelnc": graupelnc,
        "graupelncv": graupelncv,
    }
    for name, value in surface_fields.items():
        if value.shape != surface_shape:
            raise ValueError(f"{name} must have shape {surface_shape}, "
                             f"got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    kernel_name = ("thompson_graupel_sediment_64"
                   if nz <= _SHALLOW_KMAX
                   else "thompson_graupel_sediment_256")
    if graupel_number is not None and reference_density is not None:
        kernel_name += "_with_number_and_density"
    elif graupel_number is not None:
        kernel_name += "_with_number"
    elif reference_density is not None:
        kernel_name += "_with_density"
    if active_columns is not None:
        kernel_name += "_and_column_mask"
    if graupel_number_shadow is not None:
        kernel_name += "_and_shadow"
    arguments = (qg,)
    if graupel_number is not None:
        arguments += (graupel_number,)
    arguments += (temperature, pressure, qv)
    if reference_density is not None:
        arguments += (reference_density,)
    arguments += (dz, rainnc, rainncv, graupelnc, graupelncv)
    if active_columns is not None:
        arguments += (active_columns,)
    if graupel_number_shadow is not None:
        arguments += (graupel_number_shadow,)
    arguments += (np.int32(accumulate_surface), DTYPE(dt),
                  np.int32(nz), np.int32(ny), np.int32(nx))
    _launch_fallout(
        "thompson_graupel_sediment_levels_64_with_density_and_column_mask"
        "_and_shadow", kernel_name,
        LEVEL_PARALLEL_FALLOUT and nz <= _SHALLOW_KMAX
        and graupel_number_shadow is not None, ny * nx, arguments,
        columns_per_block=32)


# Positive and negative ice-vapor exchange share the exact WRF rate equation.
# Keep the original admission names source-compatible while exposing names
# suitable for the complete process driver.
launch_snow_vapor_exchange = launch_snow_sublimation


# The classic adapter's own arithmetic (woof/core/microphysics.py:
# _apply_thompson), fused into four launches by the thompson_adapter_*
# kernels.  Each writes exactly what the CuPy operations and small kernels it
# replaces wrote; tests/test_thompson_speed_glue.py holds them bit for bit.
_ELEMENT_TPB = 256


def _element_grid(size: int) -> tuple[int]:
    return ((size + _ELEMENT_TPB - 1) // _ELEMENT_TPB,)


def launch_adapter_prepare(
        thb, thp, phb, php, th, pii, temperature, dz, saved_theta,
        qc, qi, ni, qr, nr, qs, qg, entry_graupel, warm_entry, graupelncv,
        micro_columns, *, qv=None) -> None:
    """The classic adapter's entry: thermodynamics, masks and WRF's rewrite.

    ``th = thb + thp``, ``temperature = th * pii`` (``pii`` is read: the
    adapter keeps CuPy's ``power`` for it), the layer depths from
    ``(phb + php) / G`` with ``G`` passed as a value (NVRTC for sm_120 does
    not round a division by a literal constant the IEEE way; CuPy divides by
    a value), the entry graupel mask and the entry-temperature warm marker,
    GRAUPELNCV reset, the pre-microphysics theta saved into ``saved_theta``
    (``save_pre_mp_theta``), every species at or below R1 zeroed with its
    number, and ``micro_columns`` cleared for :func:`launch_adapter_entry`.
    ``thb`` and ``phb`` may be the vertical-only base profiles.
    """
    shape, size = _validate_fields({
        "thp": thp, "th": th, "pii": pii,
        "temperature": temperature, "dz": dz, "saved_theta": saved_theta,
        "qc": qc, "qi": qi, "ni": ni, "qr": qr, "nr": nr, "qs": qs,
        "qg": qg, "entry_graupel": entry_graupel, "warm_entry": warm_entry})
    nz, ny, nx = shape
    interface = (nz + 1, ny, nx)
    if php.shape != interface or php.dtype != DTYPE:
        raise ValueError(f"php must be float32 {interface}, got "
                         f"{php.dtype} {php.shape}")
    if thb.shape not in (shape, (nz,)) or phb.shape not in (interface,
                                                           (nz + 1,)):
        raise ValueError("thb and phb must be full or vertical-only")
    for name, value in (("graupelncv", graupelncv),
                        ("micro_columns", micro_columns)):
        if value.shape != (ny, nx) or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {(ny, nx)}")
    if qv is None:
        prepare = get_kernel("thompson", "thompson_adapter_prepare")
        extra = ()
    else:
        from woof.core.kernels import get_kernel_int_defines
        prepare = get_kernel_int_defines("thompson", "thompson_adapter_prepare",
                                         (("GPUWM_USE_THETA_M", 1),))
        extra = (qv,)
    prepare(
        _element_grid(size), (_ELEMENT_TPB,),
        (thb, thp, phb, php, th, pii, temperature, dz,
         saved_theta, qc, qi, ni, qr, nr, qs, qg, entry_graupel,
         warm_entry, graupelncv, micro_columns, DTYPE(_constants.G),
         np.int32(size), np.int32(ny * nx), np.int32(thb.ndim == 3),
         np.int32(phb.ndim == 3), *extra))


def launch_adapter_entry(
        qc, qi, qr, qs, qg, temperature, pressure, qv,
        graupel_number_shadow, micro_columns) -> None:
    """The private graupel number and WRF's no-microphysics column flag.

    The same values :func:`launch_classic_graupel_number_init` and
    :func:`launch_microphysics_columns` write, one thread per cell; the
    flag must have been cleared by :func:`launch_adapter_prepare`.
    """
    shape, size = _validate_fields({
        "qc": qc, "qi": qi, "qr": qr, "qs": qs, "qg": qg,
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "graupel_number_shadow": graupel_number_shadow})
    nz, ny, nx = shape
    if micro_columns.shape != (ny, nx) or micro_columns.dtype != DTYPE:
        raise ValueError(f"micro_columns must be float32 {(ny, nx)}")
    get_kernel("thompson", "thompson_adapter_entry")(
        _element_grid(size), (_ELEMENT_TPB,),
        (qc, qi, qr, qs, qg, temperature, pressure, qv,
         graupel_number_shadow, micro_columns, np.int32(size),
         np.int32(ny * nx)))


def launch_adapter_masks(qr, qg, entry_graupel, rain_columns,
                         graupel_columns) -> None:
    """The post-source rain and graupel fallout column masks in one launch.

    :func:`launch_hydrometeor_column_mask` on ``qr`` and
    :func:`launch_graupel_fallout_column_mask`.
    """
    shape, _ = _validate_fields({"qr": qr, "qg": qg,
                                 "entry_graupel": entry_graupel})
    nz, ny, nx = shape
    for name, value in (("rain_columns", rain_columns),
                        ("graupel_columns", graupel_columns)):
        if value.shape != (ny, nx) or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {(ny, nx)}")
    ncol = ny * nx
    get_kernel("thompson", "thompson_adapter_masks")(
        ((ncol + _COLUMN_TPB - 1) // _COLUMN_TPB,), (_COLUMN_TPB,),
        (qr, qg, entry_graupel, rain_columns, graupel_columns,
         np.int32(nz), np.int32(ncol)))


def launch_adapter_finish(
        temperature, pii, th, thp, h_diabatic, rainncv, snowncv, graupelncv,
        sr, cfg, dt: float, *, state=None) -> None:
    """Theta from temperature, moist_physics_finish, and SR.

    ``th = temperature / pii``; with ``cfg.no_mp_heating == 0`` the
    increment over the theta ``h_diabatic`` holds is clamped to
    ``+/- DTYPE(cfg.mp_tend_lim) * DTYPE(dt)``, the REAL product WRF forms
    (module_big_step_utilities_em.F:5706-5707), added to ``thp`` and kept as
    ``h_diabatic = increment / dt`` (woof.core.microphysics.
    moist_physics_finish), else ``h_diabatic = 0``; then
    ``sr = where(rainncv > 1e-12, minimum(1, (snowncv + graupelncv) /
    rainncv), 0)``.
    """
    shape, size = _validate_fields({
        "temperature": temperature, "pii": pii, "th": th, "thp": thp,
        "h_diabatic": h_diabatic})
    nz, ny, nx = shape
    for name, value in (("rainncv", rainncv), ("snowncv", snowncv),
                        ("graupelncv", graupelncv), ("sr", sr)):
        if value.shape != (ny, nx) or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {(ny, nx)}")
    moist_theta = state is not None and getattr(state, "use_theta_m", 0)
    if moist_theta:
        from woof.core.kernels import get_kernel_int_defines
        finish = get_kernel_int_defines("thompson", "thompson_adapter_finish",
                                        (("GPUWM_USE_THETA_M", 1),))
    else:
        finish = get_kernel("thompson", "thompson_adapter_finish")
    finish(
        _element_grid(size), (_ELEMENT_TPB,),
        (temperature, pii, th, thp, h_diabatic, rainncv, snowncv,
         # WRF multiplies the REAL mp_tend_lim by the REAL dt.  Rounding
         # the double product instead moves a clamped increment by one
         # float32 unit (0.001 K/s at dt = 20 s: 0x3ca3d70a, not WRF's
         # 0x3ca3d70b), and with it thp and h_diabatic.
         graupelncv, sr, DTYPE(DTYPE(cfg.mp_tend_lim) * DTYPE(dt)), DTYPE(dt),
         np.int32(cfg.no_mp_heating != 0), np.int32(size),
         np.int32(ny * nx)))
    if moist_theta:
        from woof.core.theta_m_physics import finish_microphysics
        finish_microphysics(state, cfg, th, dt)


__all__ = [
    "launch_adapter_entry",
    "launch_adapter_finish",
    "launch_adapter_masks",
    "launch_adapter_prepare",
    "launch_cloud_freezing",
    "launch_cloud_saturation_adjust",
    "launch_cloud_sedimentation",
    "launch_classic_graupel_number_init",
    "launch_classic_graupel_number_finalize",
    "launch_cold_cloud_source_network",
    "launch_cold_rain_source_network",
    "launch_cold_rain_snow_graupel_network",
    "launch_effective_radius",
    "launch_final_phase_cleanup",
    "launch_frozen_vapor_network",
    "launch_frozen_vapor_network_from_owner",
    "launch_graupel_cloud_riming",
    "launch_graupel_fallout_column_mask",
    "launch_hydrometeor_column_mask",
    "launch_microphysics_columns",
    "launch_graupel_sedimentation",
    "launch_graupel_melting",
    "launch_graupel_sublimation",
    "launch_ice_autoconversion",
    "launch_ice_deposition",
    "launch_ice_nucleation",
    "launch_ice_sedimentation",
    "launch_rain_evaporation",
    "launch_rain_freezing",
    "launch_rain_graupel_collection",
    "launch_rain_ice_collection",
    "launch_rain_snow_collection",
    "launch_rain_sedimentation",
    "launch_rain_self_collection",
    "launch_snow_sublimation",
    "launch_snow_cloud_riming",
    "launch_snow_ice_collection",
    "launch_snow_melting",
    "launch_snow_rime_conversion",
    "launch_snow_sedimentation",
    "launch_snow_vapor_exchange",
    "launch_warm_autoconversion",
    "launch_warm_process_network",
    "launch_warm_frozen_source_network",
    "launch_warm_frozen_source_network_from_owner",
    "launch_warm_rain_collection",
    "launch_warm_saturation_adjust",
]
