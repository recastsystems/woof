"""Official-WRF NSSL S-band reflectivity diagnostic.

The narrow launcher reproduces the default WRF v4.6.1 ``radardd02`` path
for option 18, including every transported ice category, predicted moments,
variable graupel/hail density, and the native zero-dBZ floor.  It remains an
independently testable diagnostic; this module does not enable option 18.
"""

from __future__ import annotations

import numpy as np
from woof.core.device_cache import cuda_cache

from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.state import DTYPE
from woof.core.nssl2_shape import DEFAULT_SHAPES, shape_kernel


_TPB = 256


@cuda_cache(maxsize=None)
def _radar_kernel(shapes=DEFAULT_SHAPES):
    """Match the selected native libm without changing fast/default images.

    glibc 2.41 replaced log10f's split binary32 evaluation. The native WRF
    source pin alone therefore does not identify its radar words. Use the
    existing strict referee ABI selector, also used by surface-layer tables.
    """
    from woof.wrf_exact import ENABLED, referee_glibc
    if ENABLED and referee_glibc() >= (2, 41):
        return shape_kernel(
            "nssl2_diagnostics", "nssl2_radardd02", shapes,
            (("NSSL2_ROUNDED_LOG10", 1),))
    return shape_kernel("nssl2_diagnostics", "nssl2_radardd02", shapes)


def _validate_fields(fields: dict[str, object]) -> tuple[tuple[int, ...], int]:
    first = next(iter(fields.values()))
    shape = first.shape
    if len(shape) != 3:
        raise ValueError(f"NSSL reflectivity fields must be 3-D, got {shape}")
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    return shape, int(np.prod(shape, dtype=np.int64))


def _validate_values(fields: dict[str, object]) -> None:
    import cupy as cp

    for name, value in fields.items():
        if bool(cp.any(~cp.isfinite(value))):
            raise ValueError(f"{name} must contain only finite values")
        if name in ("air_density", "temperature_k"):
            if bool(cp.any(value <= DTYPE(0.0))):
                raise ValueError(f"{name} must be strictly positive")
        elif bool(cp.any(value < DTYPE(0.0))):
            raise ValueError(f"{name} must be nonnegative")


def launch_radardd02(
        air_density, temperature_k,
        qr, qi, qs, qg, qh,
        qnr, qni, qns, qng, qnh, qvolg, qvolh, refl_10cm, *,
        output_due: bool,
        concentration_space: bool = False,
        validate_values: bool = True, shapes=DEFAULT_SHAPES) -> bool:
    """Diagnose NSSL ``refl_10cm`` in dBZ when an output is due.

    All arrays are contiguous FP32 ``(nz, ny, nx)`` fields.  Mass and volume
    moments use kg/kg and, by default, m3/kg of dry air; number moments default
    to #/kg dry air.  Set ``concentration_space=True`` when moments are direct
    views of the internal NSSL slab (#/m3 and m3/m3); no density conversion is
    then performed.  The result is WRF's S-band equivalent reflectivity with a
    0-dBZ floor.

    ``output_due=False`` is a strict no-op: it returns before inspecting any
    array or loading/compiling the CUDA module, and leaves ``refl_10cm``
    bitwise untouched.  This preserves WRF's history-output-only cost gate.
    """
    if not isinstance(output_due, bool):
        raise TypeError("output_due must be bool")
    if not output_due:
        return False
    if not isinstance(concentration_space, bool):
        raise TypeError("concentration_space must be bool")

    inputs = {
        "air_density": air_density,
        "temperature_k": temperature_k,
        "qr": qr,
        "qi": qi,
        "qs": qs,
        "qg": qg,
        "qh": qh,
        "qnr": qnr,
        "qni": qni,
        "qns": qns,
        "qng": qng,
        "qnh": qnh,
        "qvolg": qvolg,
        "qvolh": qvolh,
    }
    _, size = _validate_fields({**inputs, "refl_10cm": refl_10cm})
    import cupy as cp

    if any(cp.may_share_memory(refl_10cm, value)
           for value in inputs.values()):
        raise ValueError("refl_10cm must not alias an NSSL input field")
    if validate_values:
        _validate_values(inputs)

    blocks = (size + _TPB - 1) // _TPB
    _radar_kernel(shapes)(
        (blocks,), (_TPB,),
        (air_density, temperature_k,
         qr, qi, qs, qg, qh,
         qnr, qni, qns, qng, qnh, qvolg, qvolh, refl_10cm,
         np.int32(concentration_space),
         np.int32(size)))
    return True


def diagnose_radardd02_if_due(
        state, air_density, temperature_k, refl_10cm, *, output_due: bool,
        concentration_space: bool = False,
        validate_values: bool = True, shapes=DEFAULT_SHAPES):
    """State adapter for :func:`launch_radardd02`, without output stashing.

    The caller owns both the pre-admitted output buffer and history-frame
    handoff.  Returning ``None`` when output is not due keeps this adapter
    independent of scratch-lifetime/integration plumbing and guarantees that
    no state field or output value is touched on non-output steps.
    """
    if not isinstance(output_due, bool):
        raise TypeError("output_due must be bool")
    if not output_due:
        return None

    required = (
        "qr", "qi", "qs", "qg", "qh", "qnr", "qni", "qns",
        "qng", "qnh", "qvolg", "qvolh")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "NSSL reflectivity lacks state fields: " + ", ".join(missing))
    launch_radardd02(
        air_density, temperature_k,
        state.qr, state.qi, state.qs, state.qg, state.qh,
        state.qnr, state.qni, state.qns, state.qng, state.qnh,
        state.qvolg, state.qvolh, refl_10cm,
        output_due=True, concentration_space=concentration_space,
        validate_values=validate_values, shapes=shapes)
    return refl_10cm


__all__ = ["diagnose_radardd02_if_due", "launch_radardd02"]
