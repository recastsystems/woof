"""Fail-closed production ordering for WRF NSSL option 18.

This module only coordinates already admitted numerical stages.  It does not
make ``mp_physics=18`` selectable.  The durable driver workspace is gathered
once, remains in native concentration units through every process and
diagnostic, and is scattered exactly once before the outer moist-physics
finish callback.

The fused GS implementation is intentionally injected behind a typed callback
seam while that implementation is completed independently.  NUCOND and
QVEXCESS must also be supplied explicitly, either as two callbacks or as one
combined callback whose name records both stages.  No required production
stage has a no-op default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from woof.core.nssl2_contract import (
    DEFAULT_MODE,
    NSSL2Mode,
    require_ported_nssl2_mode,
)
from woof.core.nssl2_diagnostics import launch_radardd02
from woof.core.nssl2_driver_support import (
    NSSL2DriverWorkspace,
    gather_initialize_and_sediment,
    reduce_nssl2_precipitation,
    scatter_nssl2_driver_workspace,
)
from woof.core.nssl2_radiation import (
    launch_effective_radius_concentration,
)


class NSSL2ProductionConfigurationError(RuntimeError):
    """A required production stage or due output was not configured."""


class NSSL2FusedGSCallback(Protocol):
    """Typed seam for the in-progress fused GS process implementation."""

    def __call__(self, workspace: NSSL2DriverWorkspace, /) -> None:
        """Mutate ``workspace`` in native NSSL concentration units."""


class NSSL2WorkspaceCallback(Protocol):
    """A required process stage over the durable concentration workspace."""

    def __call__(self, workspace: NSSL2DriverWorkspace, /) -> None:
        """Mutate ``workspace`` without gathering or scattering it."""


class NSSL2MoistPhysicsFinishCallback(Protocol):
    """Outer moist-physics finish invoked after the Registry scatter."""

    def __call__(self, workspace: NSSL2DriverWorkspace, /) -> None:
        """Finish theta/heating state after the NSSL driver has returned."""


@dataclass(frozen=True)
class NSSL2RegistryFields:
    """The 16 Registry prognostics in the driver's fixed field order."""

    qv: object
    qc: object
    qr: object
    qi: object
    qs: object
    qg: object
    qh: object
    qndrop: object
    qnr: object
    qni: object
    qns: object
    qng: object
    qnh: object
    qnn: object
    qvolg: object
    qvolh: object

    def as_tuple(self) -> tuple[object, ...]:
        """Return fields in :mod:`nssl2_driver_support` argument order."""
        return (
            self.qv,
            self.qc,
            self.qr,
            self.qi,
            self.qs,
            self.qg,
            self.qh,
            self.qndrop,
            self.qnr,
            self.qni,
            self.qns,
            self.qng,
            self.qnh,
            self.qnn,
            self.qvolg,
            self.qvolh,
        )


@dataclass(frozen=True)
class NSSL2PrecipitationFields:
    """Persistent and per-step four-category precipitation outputs."""

    rainnc: object
    rainncv: object
    snownc: object
    snowncv: object
    graupelnc: object
    graupelncv: object
    hailnc: object
    hailncv: object
    sr: object

    def as_tuple(self) -> tuple[object, ...]:
        return (
            self.rainnc,
            self.rainncv,
            self.snownc,
            self.snowncv,
            self.graupelnc,
            self.graupelncv,
            self.hailnc,
            self.hailncv,
            self.sr,
        )


@dataclass(frozen=True)
class NSSL2ProductionHooks:
    """Required process callbacks; every missing stage fails before gather.

    Supply either ``nucond`` plus ``qv_excess`` as separate callbacks, or one
    ``nucond_qvexcess`` callback for an implementation (such as the admitted
    production NUCOND launcher) that performs both stages atomically.  Mixing
    the two forms is rejected because it could execute QVEXCESS twice.
    """

    fused_gs: NSSL2FusedGSCallback | None = None
    nucond: NSSL2WorkspaceCallback | None = None
    qv_excess: NSSL2WorkspaceCallback | None = None
    nucond_qvexcess: NSSL2WorkspaceCallback | None = None
    moist_physics_finish: NSSL2MoistPhysicsFinishCallback | None = None


def _require_callback(name: str, callback):
    if callback is None:
        raise NSSL2ProductionConfigurationError(
            f"required NSSL production hook {name!r} is absent"
        )
    if not callable(callback):
        raise TypeError(f"NSSL production hook {name!r} must be callable")
    return callback


def _validated_hooks(hooks: NSSL2ProductionHooks):
    if not isinstance(hooks, NSSL2ProductionHooks):
        raise TypeError("hooks must be NSSL2ProductionHooks")

    fused_gs = _require_callback("fused_gs", hooks.fused_gs)
    finish = _require_callback("moist_physics_finish", hooks.moist_physics_finish)

    return fused_gs, _condensation_hooks(hooks, _require_callback), finish


def _condensation_hooks(hooks, require_callback, *, hook_label="hooks"):
    if hooks.nucond_qvexcess is not None:
        if hooks.nucond is not None or hooks.qv_excess is not None:
            raise NSSL2ProductionConfigurationError(
                "configure either nucond_qvexcess or separate nucond and "
                f"qv_excess {hook_label}, not both"
            )
        condensation_hooks = (
            require_callback("nucond_qvexcess", hooks.nucond_qvexcess),
        )
    else:
        condensation_hooks = (
            require_callback("nucond", hooks.nucond),
            require_callback("qv_excess", hooks.qv_excess),
        )
    return condensation_hooks


def _validate_due_configuration(
    *,
    output_due: bool,
    radiation_due: bool,
    validate_values: bool,
    temperature_k,
    refl_10cm,
    re_cloud_m,
    re_ice_m,
    re_snow_m,
) -> None:
    for name, value in (
        ("output_due", output_due),
        ("radiation_due", radiation_due),
        ("validate_values", validate_values),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool")

    if temperature_k is None:
        raise NSSL2ProductionConfigurationError(
            "NSSL cloud-droplet sedimentation requires temperature_k"
        )

    if output_due:
        missing = [
            name
            for name, value in (
                ("refl_10cm", refl_10cm),
            )
            if value is None
        ]
        if missing:
            raise NSSL2ProductionConfigurationError(
                "NSSL radar output is due but lacks: " + ", ".join(missing)
            )

    if radiation_due:
        missing = [
            name
            for name, value in (
                ("re_cloud_m", re_cloud_m),
                ("re_ice_m", re_ice_m),
                ("re_snow_m", re_snow_m),
            )
            if value is None
        ]
        if missing:
            raise NSSL2ProductionConfigurationError(
                "NSSL effective radii are due but lack: " + ", ".join(missing)
            )


def run_nssl2_production_step(
    air_density,
    dz,
    registry: NSSL2RegistryFields,
    precipitation: NSSL2PrecipitationFields,
    hooks: NSSL2ProductionHooks,
    dt_s: float,
    *,
    first_step: bool = False,
    cu_used: bool = False,
    qrcuten=None,
    qscuten=None,
    qicuten=None,
    qccuten=None,
    output_due: bool = False,
    temperature_k=None,
    refl_10cm=None,
    radiation_due: bool = False,
    re_cloud_m=None,
    re_ice_m=None,
    re_snow_m=None,
    validate_values: bool = True,
    workspace: NSSL2DriverWorkspace | None = None,
    mode: NSSL2Mode = DEFAULT_MODE,
    nssl_cccn: float = 0.5e9,
    shapes=(0.0, 1.0),
) -> NSSL2DriverWorkspace:
    """Run one fail-closed NSSL production step in official driver order.

    The exact successful sequence is:

    ``gather/init/sediment -> precip reducer -> fused GS -> NUCOND ->``
    ``QVEXCESS -> due radar -> due radii -> scatter -> moist finish``.

    A combined ``nucond_qvexcess`` hook occupies the two adjacent stages with
    one callback.  All callbacks receive the same durable workspace.  Radar
    and radius diagnostics read direct concentration-space field views before
    the sole scatter.  Effective-radius outputs use the admitted launcher's
    metre convention.

    Required callbacks and due output buffers are validated before gather, so
    a configuration error cannot leave the Registry partially advanced.

    ``mode`` is the resolved WRF variant (see
    :func:`woof.core.nssl2_contract.resolve_nssl2_mode`).  It governs the
    CCN load and store at this seam; the hail switch reaches the GS stage
    through the caller's ``fused_gs`` callback, which owns it.  A callback
    that DECLARES its switch (``NSSL2FusedGS.hail_on``, and anything else
    exposing that attribute) is cross-checked against the mode below, so
    the two halves of a variant cannot disagree.  A bare closure declares
    nothing and cannot be checked here; on the shipped path that gap is
    closed by ``NSSL2ProductionBinding.validate``, which the front door
    (woof/core/microphysics.py) refuses to run without.
    """
    if not isinstance(registry, NSSL2RegistryFields):
        raise TypeError("registry must be NSSL2RegistryFields")
    require_ported_nssl2_mode(mode)
    if not isinstance(precipitation, NSSL2PrecipitationFields):
        raise TypeError("precipitation must be NSSL2PrecipitationFields")
    if workspace is not None and not isinstance(
            workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace or None")

    fused_gs, condensation_hooks, finish = _validated_hooks(hooks)
    declared_hail = getattr(fused_gs, "hail_on", None)
    if declared_hail is not None and bool(declared_hail) is not mode.hail:
        raise NSSL2ProductionConfigurationError(
            f"NSSL fused-GS callback declares hail_on={declared_hail!r} "
            f"but the resolved variant mode has hail={mode.hail!r}")
    _validate_due_configuration(
        output_due=output_due,
        radiation_due=radiation_due,
        validate_values=validate_values,
        temperature_k=temperature_k,
        refl_10cm=refl_10cm,
        re_cloud_m=re_cloud_m,
        re_ice_m=re_ice_m,
        re_snow_m=re_snow_m,
    )

    registry_fields = registry.as_tuple()
    gather_kwargs = {
        "first_step": first_step,
        "cu_used": cu_used,
        "qrcuten": qrcuten,
        "qscuten": qscuten,
        "qicuten": qicuten,
        "qccuten": qccuten,
        "predicted_ccn": mode.predicted_ccn,
        "nssl_cccn": nssl_cccn,
        "shapes": shapes,
    }
    if workspace is not None:
        # Keep the direct/oracle path backward compatible while making the
        # production selector's persistent ownership explicit.
        gather_kwargs["workspace"] = workspace
    workspace = gather_initialize_and_sediment(
        air_density,
        dz,
        *registry_fields,
        dt_s,
        temperature_k=temperature_k,
        **gather_kwargs,
    )

    reduce_nssl2_precipitation(
        workspace, *precipitation.as_tuple(),
        air_density=air_density, dt_s=dt_s)

    fused_gs(workspace)
    for condensation_hook in condensation_hooks:
        condensation_hook(workspace)

    if output_due:
        launch_radardd02(
            air_density,
            temperature_k,
            workspace.field("qr"),
            workspace.field("qi"),
            workspace.field("qs"),
            workspace.field("qg"),
            workspace.field("qh"),
            workspace.field("qnr"),
            workspace.field("qni"),
            workspace.field("qns"),
            workspace.field("qng"),
            workspace.field("qnh"),
            workspace.field("qvolg"),
            workspace.field("qvolh"),
            refl_10cm,
            output_due=True,
            concentration_space=True,
            shapes=shapes,
            validate_values=validate_values,
        )

    if radiation_due:
        launch_effective_radius_concentration(
            air_density,
            workspace.field("qc"),
            workspace.field("qndrop"),
            workspace.field("qi"),
            workspace.field("qni"),
            workspace.field("qs"),
            workspace.field("qns"),
            re_cloud_m,
            re_ice_m,
            re_snow_m,
            validate_values=validate_values,
        )

    scatter_nssl2_driver_workspace(
        workspace, air_density, *registry_fields,
        predicted_ccn=mode.predicted_ccn)
    finish(workspace)
    return workspace


__all__ = [
    "NSSL2FusedGSCallback",
    "NSSL2MoistPhysicsFinishCallback",
    "NSSL2PrecipitationFields",
    "NSSL2ProductionConfigurationError",
    "NSSL2ProductionHooks",
    "NSSL2RegistryFields",
    "NSSL2WorkspaceCallback",
    "run_nssl2_production_step",
]
