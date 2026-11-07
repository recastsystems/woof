"""Production WRF v4.6.1 NSSL ``NUCOND`` warm-phase stage.

This module is intentionally independent of the still fail-closed option-18
coordinator.  It exposes the complete default ``ipconc=5`` warm ``NUCOND``
stage at a narrow array boundary: joint cloud/rain condensation (``rcond=2``),
cloud evaporation, predicted-CCN/droplet coupling, default ``irenuc=2``
renucleation, the ordinary and 1.9-ratio maximum-supersaturation ``QVEXCESS``
paths, and WRF's moment tail (``module_mp_nssl_2mom.F:11726-12205``): every
category below its threshold or without number is vaporized in WRF's order,
graupel/hail volumes are held inside their density bounds and graupel number
above its minimum intercept, and predicted CCN relaxes toward background.
"""

from __future__ import annotations

import math

import numpy as np

from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.state import DTYPE
from woof.core.nssl2_shape import DEFAULT_SHAPES, shape_kernel


_TPB = 256


def _validate_cell_fields(fields: dict[str, object]) -> tuple[tuple[int, ...], int]:
    first = next(iter(fields.values()))
    shape = first.shape
    if len(shape) != 3:
        raise ValueError(f"NSSL NUCOND fields must be 3-D, got {shape}")
    if shape[0] < 3:
        raise ValueError(f"NSSL NUCOND requires nz >= 3, got {shape[0]}")
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    return shape, int(np.prod(shape, dtype=np.int64))


def _validate_values(
        fields: dict[str, object], positive: tuple[str, ...],
        nonnegative: tuple[str, ...]) -> None:
    import cupy as cp

    for name, value in fields.items():
        if bool(cp.any(~cp.isfinite(value))):
            raise ValueError(f"{name} must contain only finite values")
        if name in positive:
            if bool(cp.any(value <= DTYPE(0.0))):
                raise ValueError(f"{name} must be strictly positive")
        elif name in nonnegative and bool(cp.any(value < DTYPE(0.0))):
            raise ValueError(f"{name} must be nonnegative")


def launch_nucond(
        full_theta, air_density, pressure_pa, exner, w_interface,
        qv, qc, qr, qi, qs, qndrop, qnr, qni, qns, qnn, dt_s: float, *,
        supersaturation_scratch=None,
        concentration_space: bool = False,
        predicted_ccn: bool = True,
        nssl_cccn: float = 0.5e9,
        validate_values: bool = True,
        frozen=None, shapes=DEFAULT_SHAPES) -> None:
    """Advance the exact default MP18 warm ``NUCOND`` stage in place.

    Cell fields use contiguous FP32 ``(nz, ny, nx)`` storage.  ``w_interface``
    is FP32 ``(nz + 1, ny, nx)`` so the kernel can reproduce WRF's
    ``0.5 * (w(k) + w(k+1))`` mass-level velocity.  Water fields are kg/kg
    dry air.  ``qi``, ``qs``, ``qni`` and ``qns`` take part in WRF's moment
    tail: sub-threshold or numberless ice and snow are vaporized, and the
    clear-cloud CCN restoration is gated on the ice-plus-snow mass the tail
    leaves.  ``frozen`` is the graupel/hail sextet ``(qg, qh, qng, qnh,
    qvolg, qvolh)`` the tail also cleans and density-bounds; the production
    hook always passes it.  It requires ``concentration_space=True``: the
    graupel minimum-intercept check reads number concentration (#/m3).
    Omitting it skips only the graupel and hail arms.  By default
    number/CCN fields use Registry #/kg dry air.  Set
    ``concentration_space=True`` for production slab views already held in
    NSSL's internal #/m3 convention; that path performs no density round-trip.
    Potential temperature is K, pressure Pa, and density kg/m3.

    ``supersaturation_scratch`` may supply a reusable contiguous FP32 cell
    field.  The prepass reproduces WRF's immutable ``ssfilt`` field, avoiding
    an in-place neighbor-read race during existing-cloud renucleation.

    ``predicted_ccn`` is WRF's ``nssl_ccn_on``.  ``False`` selects the
    ``nssl_ccn_on=0`` variant, where the driver raises ``renucfrac`` to 1.0
    (``module_mp_nssl_2mom.F:2555-2557``) so the nucleation pool ``cnuc``
    becomes the actual diagnosed CCN instead of the background floor, and
    the low-temperature updraft limiter at :10120-10127 is armed.  The
    ``imaxsupopt=4`` saturation-adjustment activation is unaffected: it
    spells out ``Max(ccnc, cwnccn)`` at :11559 rather than using ``cnuc``.

    Value validation is synchronous and fail-closed by default.  Production
    callers that have already passed the global device health gate may set
    ``validate_values=False`` to avoid the redundant reductions; structural
    shape/dtype/contiguity and timestep checks are never optional.
    """
    from woof.core.nssl2_contract import (
        nssl2_background_ccn_per_kg, nssl2_cccn_concentration,
    )
    background_ccn_per_kg = nssl2_background_ccn_per_kg(nssl_cccn)
    fields = {
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "qv": qv,
        "qc": qc,
        "qr": qr,
        "qi": qi,
        "qs": qs,
        "qndrop": qndrop,
        "qnr": qnr,
        "qni": qni,
        "qns": qns,
        "qnn": qnn,
    }
    frozen_names = ("qg", "qh", "qng", "qnh", "qvolg", "qvolh")
    if frozen is not None:
        frozen = tuple(frozen)
        if len(frozen) != len(frozen_names):
            raise ValueError(
                "frozen must be the six arrays (qg, qh, qng, qnh, qvolg, "
                "qvolh)")
        if not concentration_space:
            raise ValueError(
                "frozen requires concentration_space=True: the graupel "
                "minimum-intercept check reads number concentration in #/m3")
        fields.update(zip(frozen_names, frozen))
    shape, size = _validate_cell_fields(fields)
    w_shape = (shape[0] + 1, shape[1], shape[2])
    if w_interface.shape != w_shape:
        raise ValueError(
            f"w_interface must have shape {w_shape}, got {w_interface.shape}")
    if w_interface.dtype != DTYPE:
        raise TypeError(f"w_interface must be float32, got {w_interface.dtype}")
    if not w_interface.flags.c_contiguous:
        raise ValueError("w_interface must be C-contiguous")
    if not isinstance(concentration_space, bool):
        raise TypeError("concentration_space must be bool")
    if not isinstance(predicted_ccn, bool):
        raise TypeError("predicted_ccn must be bool")

    if supersaturation_scratch is None:
        import cupy as cp

        supersaturation_scratch = cp.empty_like(qv)
    else:
        import cupy as cp

        scratch_shape, _ = _validate_cell_fields(
            {"supersaturation_scratch": supersaturation_scratch})
        if scratch_shape != shape:
            raise ValueError(
                "supersaturation_scratch must have shape "
                f"{shape}, got {scratch_shape}")
        if any(cp.may_share_memory(supersaturation_scratch, value)
               for value in fields.values()):
            raise ValueError(
                "supersaturation_scratch must not alias a NUCOND field")

    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    if validate_values:
        _validate_values(
            {**fields, "w_interface": w_interface},
            ("full_theta", "air_density", "pressure_pa", "exner"),
            ("qv", "qc", "qr", "qi", "qs", "qndrop", "qnr", "qni",
             "qns", "qnn", *frozen_names))

    if frozen is None:
        # The kernel reads these only when has_frozen is set; any valid
        # pointer will do, and qv is one.
        frozen_args = (qv,) * len(frozen_names)
    else:
        frozen_args = frozen
    from woof.wrf_exact import ENABLED
    frozen_kernel_args = (*frozen_args, np.int32(0 if frozen is None else 1)) if ENABLED or shapes != DEFAULT_SHAPES else ()
    blocks = (size + _TPB - 1) // _TPB
    shape_kernel("nssl2_nucond", "nssl2_nucond_supersaturation", shapes)(
        (blocks,), (_TPB,),
        (full_theta, pressure_pa, exner, qv, supersaturation_scratch,
         np.int32(size)))
    coefficient = nssl2_cccn_concentration(nssl_cccn)
    # The default ABI keeps literal constants so fast CUDA retains its
    # established contraction and rounding. Only an override takes scalars.
    coefficient_args = ()
    if coefficient == np.float32(0.5e9):
        kernel = shape_kernel("nssl2_nucond", "nssl2_nucond_default", shapes)
    else:
        kernel = shape_kernel(
            "nssl2_nucond", "nssl2_nucond_default", shapes,
            (("NSSL2_VARIABLE_CCCN", 1),))
        coefficient_args = (background_ccn_per_kg, coefficient)
    kernel(
        (blocks,), (_TPB,),
        (full_theta, air_density, pressure_pa, exner, w_interface,
         supersaturation_scratch,
         qv, qc, qr, qi, qs, qndrop, qnr, qni, qns, qnn,
         *frozen_kernel_args, step32,
         np.int32(concentration_space),
         np.int32(size), np.int32(shape[0]),
         np.int32(shape[1] * shape[2]),
         np.int32(1 if predicted_ccn else 0), *coefficient_args))


__all__ = ["launch_nucond"]
