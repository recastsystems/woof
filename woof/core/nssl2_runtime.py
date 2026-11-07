"""Explicit DomainState boundary for the NSSL option-18 coordinator.

This module does not make ``mp_physics=18`` selectable.  It prepares the
WRF microphysics fields owned by :class:`~woof.core.state.DomainState`, binds
the canonical Registry/precipitation/KF authorities, and calls the already
admitted fail-closed production coordinator only when a complete explicit
hook bundle is supplied.  In particular, there is no fallback or no-op fused
GS implementation.

The adapter returns :class:`~woof.core.microphysics.MicrophysicsDiagnostics`
for the existing outer ``PhysicsDriver.accept_microphysics`` boundary.  That
outer acceptance remains the sole authority that advances
``microphysics_updates`` and the pending Noah rain handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol

import cupy as cp
import numpy as np

from woof.config import RunConfig
from woof.core import constants as c
from woof.core.microphysics import (
    MicrophysicsDiagnostics,
    moist_physics_finish,
    save_pre_mp_theta,
)
from woof.core.nssl2_contract import (
    DEFAULT_RESTART_FIELDS,
    NSSL2Mode,
    pinned_zero_fields,
    require_ported_nssl2_mode,
    resolve_nssl2_mode_for_config,
)
from woof.core.nssl2_driver_support import NSSL2DriverWorkspace, _step32
from woof.core.nssl2_production_coordinator import (
    NSSL2PrecipitationFields,
    NSSL2ProductionConfigurationError,
    NSSL2ProductionHooks,
    NSSL2RegistryFields,
    run_nssl2_production_step,
    _condensation_hooks,
)
from woof.core.refl import refl_10cm_is_stashed, stash_refl_10cm
from woof.core.state import DTYPE, DomainState


_REGISTRY_NAMES = DEFAULT_RESTART_FIELDS
_PRECIPITATION_SLOTS = (
    ("rainnc", "mp_rainnc"),
    ("rainncv", "mp_rainncv"),
    ("snownc", "mp_snownc"),
    ("snowncv", "mp_snowncv"),
    ("graupelnc", "mp_graupelnc"),
    ("graupelncv", "mp_graupelncv"),
    ("hailnc", "mp_hailnc"),
    ("hailncv", "mp_hailncv"),
    ("sr", "mp_sr"),
)
_KF_RATE_ARGUMENTS = (
    ("qrcuten", "rqrcuten"),
    ("qscuten", "rqscuten"),
    ("qicuten", "rqicuten"),
    ("qccuten", "rqccuten"),
)


class NSSL2RuntimeStageCallback(Protocol):
    """Explicit production stage over one concentration workspace/context."""

    def __call__(
            self, workspace: NSSL2DriverWorkspace,
            fields: "NSSL2RuntimeFields", /) -> None:
        """Mutate the admitted workspace and prepared thermodynamics."""


@dataclass(frozen=True)
class NSSL2RuntimeHooks:
    """Required runtime stages; no numerical stage has a default.

    Supply either separate ``nucond`` and ``qv_excess`` hooks or one combined
    ``nucond_qvexcess`` hook.  The final configured condensation hook is also
    the temperature-publication boundary for output-due radar diagnostics.
    """

    fused_gs: NSSL2RuntimeStageCallback | None = None
    nucond: NSSL2RuntimeStageCallback | None = None
    qv_excess: NSSL2RuntimeStageCallback | None = None
    nucond_qvexcess: NSSL2RuntimeStageCallback | None = None


@dataclass(frozen=True)
class NSSL2RuntimeFields:
    """WRF ``moist_physics_prep_em`` fields for one NSSL invocation.

    ``theta`` is full dry potential temperature (K), ``rho`` is dry-air
    density (kg m-3), ``pressure`` is full pressure (Pa), ``pii`` is Exner,
    ``dz`` is full-geopotential layer depth (m), and ``w`` is the staggered
    physical vertical velocity with shape ``(nz + 1, ny, nx)``.  ``qv`` is
    the Registry vapour as loaded (nothing writes it before the final
    scatter); the fused GS diagnoses WRF's primary-ice target from it.
    """

    theta: object
    rho: object
    pressure: object
    pii: object
    dz: object
    w: object
    qv: object = None


def _require_callback(name: str, value):
    if value is None:
        raise NSSL2ProductionConfigurationError(
            f"required NSSL runtime hook {name!r} is absent")
    if not callable(value):
        raise TypeError(f"NSSL runtime hook {name!r} must be callable")
    return value


def _validated_hooks(hooks: NSSL2RuntimeHooks):
    if not isinstance(hooks, NSSL2RuntimeHooks):
        raise TypeError("hooks must be NSSL2RuntimeHooks")
    fused_gs = _require_callback("fused_gs", hooks.fused_gs)
    return fused_gs, _condensation_hooks(
        hooks, _require_callback, hook_label="runtime hooks")


def pin_absent_nssl2_fields(
        state: DomainState, mode: NSSL2Mode) -> tuple[str, ...]:
    """Zero every Registry field the resolved variant does not carry.

    In WRF a variant's absent categories are absent because the Registry
    package never declares them: there is no array to advect, sediment,
    write to history or restore from a restart, and the scheme's own
    hail/CCN-indexed rates read structural zeros.  woof allocates the full
    option-18 field set unconditionally (one DomainState layout for every
    mode, which is what keeps the arena and the restart manifest fixed), so
    "the field does not exist" has to be enforced rather than inherited.
    This is that enforcement, and it is deliberately idempotent and
    unconditional rather than a check: dynamics, a nest feedback, a DA
    increment, or a restored checkpoint can each deposit mass in a slot the
    active variant has no physics for, and WRF's answer in every one of
    those cases is that the value goes nowhere.

    Called at domain construction (before the first output can publish a
    field the run does not have) and once per microphysics step (so nothing
    upstream can reintroduce one).  Returns the field names it pinned --
    empty for the default lane, which therefore does no work at all here.
    """
    require_ported_nssl2_mode(mode)
    pinned = pinned_zero_fields(mode)
    for name in pinned:
        array = getattr(state, name, None)
        if array is None:
            raise NSSL2ProductionConfigurationError(
                f"NSSL variant pins {name} to zero but the DomainState "
                "does not carry it")
        array.fill(DTYPE(0.0))
    return pinned


def _require_bool(name: str, value) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")


def _validated_binding(binding, state, hooks, step):
    if binding is None:
        return None
    # Lazy import avoids the default_hooks -> runtime import cycle.
    from woof.core.nssl2_default_hooks import NSSL2ProductionBinding
    if not isinstance(binding, NSSL2ProductionBinding):
        raise TypeError("binding must be NSSL2ProductionBinding or None")
    if binding.hooks is not hooks:
        raise NSSL2ProductionConfigurationError(
            "NSSL runtime hooks do not belong to the supplied binding")
    binding.validate(state, step)
    return binding.workspace


def _require_array(value, shape: tuple[int, ...], name: str) -> None:
    if value is None:
        raise ValueError(f"NSSL runtime requires {name}")
    if tuple(value.shape) != shape:
        raise ValueError(
            f"NSSL runtime {name} must have shape {shape}, got "
            f"{tuple(value.shape)}")
    if value.dtype != DTYPE:
        raise TypeError(
            f"NSSL runtime {name} must be float32, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"NSSL runtime {name} must be C-contiguous")


def _validate_state_structure(state: DomainState) -> tuple[int, int, int]:
    pressure = getattr(state, "p", None)
    if pressure is None or len(pressure.shape) != 3:
        raise ValueError("NSSL runtime requires a three-dimensional state.p")
    shape = tuple(pressure.shape)
    nz, ny, nx = shape
    if nz < 2:
        raise ValueError(f"NSSL runtime requires nz >= 2, got {nz}")
    for name in ("p", "alt", "thp", "h_diabatic", *_REGISTRY_NAMES,
                 "effc", "effi", "effs"):
        _require_array(getattr(state, name, None), shape, f"state.{name}")
    _require_array(getattr(state, "php", None), (nz + 1, ny, nx),
                   "state.php")
    _require_array(getattr(state, "w", None), (nz + 1, ny, nx),
                   "state.w")

    thb = getattr(state, "thb", None)
    if thb is None or tuple(thb.shape) not in ((nz,), shape):
        raise ValueError(
            f"NSSL runtime state.thb must have shape {(nz,)} or {shape}, "
            f"got {None if thb is None else tuple(thb.shape)}")
    if thb.dtype != DTYPE or not thb.flags.c_contiguous:
        raise TypeError("NSSL runtime state.thb must be contiguous float32")
    phb = getattr(state, "phb", None)
    full_shape = (nz + 1, ny, nx)
    if phb is None or tuple(phb.shape) not in ((nz + 1,), full_shape):
        raise ValueError(
            f"NSSL runtime state.phb must have shape {(nz + 1,)} or "
            f"{full_shape}, got {None if phb is None else tuple(phb.shape)}")
    if phb.dtype != DTYPE or not phb.flags.c_contiguous:
        raise TypeError("NSSL runtime state.phb must be contiguous float32")
    return shape


def _validate_driver_bindings(state, cfg, shape):
    driver = getattr(state, "physics", None)
    if driver is None:
        raise NSSL2ProductionConfigurationError(
            "NSSL runtime requires an attached PhysicsDriver")
    if getattr(driver, "state", None) is not state:
        raise NSSL2ProductionConfigurationError(
            "NSSL PhysicsDriver is attached to a different DomainState")
    if int(getattr(driver, "mp_physics", -1)) != 18:
        raise NSSL2ProductionConfigurationError(
            "NSSL runtime requires an MP18 PhysicsDriver")
    updates = getattr(driver, "microphysics_updates", None)
    if (isinstance(updates, (bool, np.bool_))
            or not isinstance(updates, Integral) or updates < 0):
        raise NSSL2ProductionConfigurationError(
            "NSSL first-call authority microphysics_updates must be a "
            f"non-negative integer, got {updates!r}")

    surface_shape = shape[1:]
    pool = getattr(state, "_scratch", None)
    if not isinstance(pool, dict):
        raise NSSL2ProductionConfigurationError(
            "NSSL runtime requires the DomainState scratch registry")
    microphysics = getattr(driver, "microphysics", None)
    precipitation_values = {}
    for component, slot in _PRECIPITATION_SLOTS:
        canonical = pool.get(slot)
        value = (None if microphysics is None
                 else getattr(microphysics, component, None))
        if value is None or value is not canonical:
            raise NSSL2ProductionConfigurationError(
                f"NSSL precipitation {component} must alias canonical "
                f"scratch slot {slot}")
        _require_array(value, surface_shape, f"scratch.{slot}")
        precipitation_values[component] = value

    raw_rates = {}
    if cfg.cu_physics:
        rates = getattr(driver, "cu_rates", None)
        if not isinstance(rates, dict):
            raise NSSL2ProductionConfigurationError(
                "NSSL KF coupling requires persistent PhysicsDriver.cu_rates")
        for argument, rate_name in _KF_RATE_ARGUMENTS:
            value = rates.get(rate_name)
            canonical = pool.get(f"cu_{rate_name}")
            if value is None or value is not canonical:
                raise NSSL2ProductionConfigurationError(
                    f"NSSL KF raw-rate authority {rate_name} must alias "
                    f"canonical scratch slot cu_{rate_name}")
            _require_array(value, shape, f"driver.cu_rates[{rate_name!r}]")
            raw_rates[argument] = value
    elif getattr(driver, "cu_rates", None) is not None:
        raise NSSL2ProductionConfigurationError(
            "NSSL runtime found KF raw rates while cu_physics is disabled")
    return driver, precipitation_values, raw_rates


def _prepare_fields(state, shape) -> NSSL2RuntimeFields:
    nz, ny, nx = shape
    theta = state.scratch(shape, "mp_th")
    rho = state.scratch(shape, "mp_rho")
    pii = state.scratch(shape, "mp_pii")
    dz = state.scratch(shape, "mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
               else thb + state.thp)
    rho[...] = DTYPE(1.0) / state.alt
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz[...] = z8w[1:] - z8w[:-1]
    return NSSL2RuntimeFields(
        theta=theta, rho=rho, pressure=state.p, pii=pii, dz=dz,
        w=state.w, qv=state.qv)


def _validate_prepared_values(fields: NSSL2RuntimeFields) -> None:
    for name, value in (
            ("theta", fields.theta), ("rho", fields.rho),
            ("pressure", fields.pressure), ("pii", fields.pii),
            ("dz", fields.dz), ("w", fields.w)):
        if bool(cp.any(~cp.isfinite(value))):
            raise ValueError(f"NSSL prepared {name} must be finite")
    for name, value in (
            ("theta", fields.theta), ("rho", fields.rho),
            ("pressure", fields.pressure), ("pii", fields.pii),
            ("dz", fields.dz)):
        if bool(cp.any(value <= DTYPE(0.0))):
            raise ValueError(f"NSSL prepared {name} must be strictly positive")


def _convert_radii_to_microns(state, *, validate_values: bool) -> None:
    metre_to_micron = DTYPE(1.0e6)
    for value in (state.effc, state.effi, state.effs):
        cp.multiply(value, metre_to_micron, out=value)
    if not validate_values:
        return
    bounds = (
        ("effc", state.effc, 2.51e-6, 50.0e-6),
        ("effi", state.effi, 10.01e-6, 125.0e-6),
        ("effs", state.effs, 25.0e-6, 999.0e-6),
    )
    for name, value, lower_m, upper_m in bounds:
        lower = DTYPE(lower_m) * metre_to_micron
        upper = DTYPE(upper_m) * metre_to_micron
        invalid = (~cp.isfinite(value)) | (value < lower) | (value > upper)
        if bool(cp.any(invalid)):
            raise RuntimeError(
                f"NSSL {name} escaped native [{float(lower)}, "
                f"{float(upper)}] micron bounds")


def apply_nssl2_production(
        state: DomainState, cfg: RunConfig, dt_s: float,
        hooks: NSSL2RuntimeHooks, *, output_due: bool = False,
        radiation_due: bool = False,
        validate_values: bool = True,
        binding=None) -> MicrophysicsDiagnostics:
    """Run the explicit MP18 adapter without unlocking the global selector.

    Configuration, hook, state, driver, alias, and due-buffer authorities are
    checked before ``h_diabatic`` is parked.  Once admitted, the exact order is
    delegated to :func:`run_nssl2_production_step`.  Absolute temperature is
    formed before gather for cloud-droplet sedimentation, then refreshed from
    post-process theta after the final condensation/QVEXCESS hook and before
    the coordinator's concentration-space radar diagnostic.
    Radius conversion and moist finish occur in the coordinator's one final
    finish callback; REFL is stashed once only after the whole coordinator
    call succeeds.

    The returned diagnostic aliases the nine canonical persistent slots.
    Callers must pass it once to ``state.physics.accept_microphysics`` after
    this function returns, matching the existing dycore adapter contract.

    ``radiation_due`` remains a public gate for isolated adapter tests.  Any
    eventual MP18 selector integration must pass ``radiation_due=True`` on
    every microphysics invocation: ``state.effc/effi/effs`` are the completed
    scheme's persistent next-radiation-state diagnostics, matching the
    existing WSM6/Morrison adapter cadence rather than the radiation driver's
    less frequent call schedule.
    """
    fused_gs, condensation = _validated_hooks(hooks)
    for name, value in (
            ("output_due", output_due), ("radiation_due", radiation_due),
            ("validate_values", validate_values)):
        _require_bool(name, value)
    step = _step32(dt_s)
    if not isinstance(cfg, RunConfig):
        raise TypeError("cfg must be RunConfig")
    if int(cfg.mp_physics) != 18:
        raise NSSL2ProductionConfigurationError(
            "NSSL runtime adapter requires cfg.mp_physics == 18")
    if not cfg.moist:
        raise NSSL2ProductionConfigurationError(
            "NSSL runtime adapter requires cfg.moist=True")
    # The RunConfig selectors are the authority for the variant, and
    # validate_run_config has already refused any unported combination.
    # Re-resolve here so an adapter called directly cannot bypass the gate,
    # and cross-check the persistent binding: hail and CCN treatment change
    # which prognostics exist, so they must not drift mid-run.
    mode = require_ported_nssl2_mode(resolve_nssl2_mode_for_config(cfg))
    if binding is not None and getattr(binding, "mode", None) != mode:
        raise NSSL2ProductionConfigurationError(
            "NSSL production binding was built for variant mode "
            f"{getattr(binding, 'mode', None)!r} but the RunConfig resolves "
            f"to {mode!r}")
    if binding is not None and getattr(binding, "nssl_cccn", 0.5e9) != cfg.nssl_cccn:
        raise NSSL2ProductionConfigurationError("NSSL production binding nssl_cccn differs from RunConfig")
    from woof.core.nssl2_shape import resolved_shapes
    shapes = resolved_shapes(cfg.nssl_alphah, cfg.nssl_alphahl, cfg.nssl_icdx, cfg.nssl_icdxhl)
    if binding is not None and getattr(binding, "shapes", (0.0, 1.0)) != shapes:
        raise NSSL2ProductionConfigurationError("NSSL production binding shapes differ from RunConfig")
    bound_workspace = _validated_binding(binding, state, hooks, step)
    shape = _validate_state_structure(state)
    # Second pin of the variant's absent categories.  woof.core.
    # microphysics.apply already pinned them ahead of the spec-zone ring
    # snapshot, which is the seam that owns the ring; this one makes the
    # adapter self-sufficient for a caller that reaches it directly, and
    # is a no-op for the default lane.
    pin_absent_nssl2_fields(state, mode)
    driver, precipitation_values, raw_rates = _validate_driver_bindings(
        state, cfg, shape)
    if binding is not None and getattr(
            driver, "nssl2_binding", None) is not binding:
        raise NSSL2ProductionConfigurationError(
            "NSSL runtime binding is not the attached PhysicsDriver binding")
    if output_due and refl_10cm_is_stashed(state):
        raise NSSL2ProductionConfigurationError(
            "NSSL REFL_10CM stash was not consumed before the due call")

    fields = _prepare_fields(state, shape)
    if validate_values:
        _validate_prepared_values(fields)
    registry = NSSL2RegistryFields(**{
        name: getattr(state, name) for name in _REGISTRY_NAMES
    })
    precipitation = NSSL2PrecipitationFields(**precipitation_values)

    temperature = state.scratch(shape, "refl_t")
    cp.multiply(fields.theta, fields.pii, out=temperature)
    reflectivity = None
    if output_due:
        reflectivity = state.scratch(shape, "refl_10cm")

    def wrap_stage(callback, *, publishes_temperature: bool = False):
        def wrapped(workspace):
            callback(workspace, fields)
            if publishes_temperature:
                temperature[...] = fields.theta * fields.pii
        return wrapped

    wrapped_condensation = [
        wrap_stage(callback, publishes_temperature=(
            output_due and index == len(condensation) - 1))
        for index, callback in enumerate(condensation)
    ]

    def finish(_workspace):
        if radiation_due:
            _convert_radii_to_microns(
                state, validate_values=validate_values)
        moist_physics_finish(state, cfg, fields.theta, step)

    condensation_names = (("nucond_qvexcess",) if len(wrapped_condensation) == 1
                          else ("nucond", "qv_excess"))
    coordinator_hooks = NSSL2ProductionHooks(
        fused_gs=wrap_stage(fused_gs),
        moist_physics_finish=finish,
        **dict(zip(condensation_names, wrapped_condensation)),
    )

    save_pre_mp_theta(state)
    coordinator_kwargs = dict(
        first_step=(int(driver.microphysics_updates) == 0),
        cu_used=bool(cfg.cu_physics),
        output_due=output_due,
        temperature_k=temperature,
        refl_10cm=reflectivity,
        radiation_due=radiation_due,
        re_cloud_m=(state.effc if radiation_due else None),
        re_ice_m=(state.effi if radiation_due else None),
        re_snow_m=(state.effs if radiation_due else None),
        validate_values=validate_values,
        mode=mode,
        nssl_cccn=cfg.nssl_cccn, shapes=shapes,
        **raw_rates,
    )
    if bound_workspace is not None:
        coordinator_kwargs["workspace"] = bound_workspace
    run_nssl2_production_step(
        fields.rho,
        fields.dz,
        registry,
        precipitation,
        coordinator_hooks,
        step,
        **coordinator_kwargs,
    )
    if output_due:
        stash_refl_10cm(state, reflectivity)
    return MicrophysicsDiagnostics(**precipitation_values)


__all__ = [
    "NSSL2RuntimeFields",
    "NSSL2RuntimeHooks",
    "NSSL2RuntimeStageCallback",
    "apply_nssl2_production",
]
