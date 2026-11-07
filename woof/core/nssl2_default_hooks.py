"""Concrete default hook wiring for the NSSL option-18 runtime adapter.

This module binds the already admitted full default ``NUCOND`` launcher to
the durable concentration-space workspace.  That launcher includes WRF's
internal maximum-supersaturation ``QVEXCESS`` call and caller update, so the
runtime receives it as one combined ``nucond_qvexcess`` hook.  The standalone
QVEXCESS module remains an independently validated seam; it is not executed a
second time here.

The fused GS callback is always supplied explicitly.  There is no placeholder
or no-op default, and importing this module does not unlock ``mp_physics=18``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from woof.core.nssl2_driver_support import (
    _step32,
    NSSL2_DRIVER_IGNORED_ACCUMULATOR_SCRATCH,
    NSSL2_DRIVER_STATE_SCRATCH,
    NSSL2_DRIVER_SURFACE_EXPORT_SCRATCH,
    NSSL2DriverWorkspace,
    validate_nssl2_driver_workspace,
)
from woof.core.nssl2_contract import (
    DEFAULT_MODE,
    NSSL2Mode,
    require_ported_nssl2_mode,
)
from woof.core.nssl2_nucond import launch_nucond
from woof.core.nssl2_production_coordinator import (
    NSSL2ProductionConfigurationError,
)
from woof.core.nssl2_runtime import (
    NSSL2RuntimeFields,
    NSSL2RuntimeHooks,
    NSSL2RuntimeStageCallback,
)
from woof.core.state import DTYPE


NSSL2_NUCOND_SCRATCH = "nssl2_nucond_ss"


def _validate_scratch(
        value, shape: tuple[int, ...],
        name: str = NSSL2_NUCOND_SCRATCH) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got "
            f"{tuple(value.shape)}")
    if value.dtype != DTYPE:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _required_state_shape(state) -> tuple[int, int, int]:
    pressure = getattr(state, "p", None)
    w_interface = getattr(state, "w", None)
    if pressure is None or len(getattr(pressure, "shape", ())) != 3:
        raise NSSL2ProductionConfigurationError(
            "NSSL default hooks require three-dimensional state.p")
    shape = tuple(pressure.shape)
    expected_w_shape = (shape[0] + 1, shape[1], shape[2])
    if w_interface is None or tuple(w_interface.shape) != expected_w_shape:
        raise NSSL2ProductionConfigurationError(
            "NSSL default hooks require state.w with shape "
            f"{expected_w_shape}")
    return shape


def make_nssl2_default_runtime_hooks(
        state, dt_s: float, fused_gs: NSSL2RuntimeStageCallback, *,
        validate_values: bool = True,
        mode: NSSL2Mode = DEFAULT_MODE,
        nssl_cccn: float = 0.5e9, shapes=(0.0, 1.0)) -> NSSL2RuntimeHooks:
    """Bind the production default NUCOND+QVEXCESS stage to ``state``.

    ``fused_gs`` is the separately admitted shared-limiter GS implementation.
    The returned bundle can be passed directly to
    :func:`woof.core.nssl2_runtime.apply_nssl2_production`.

    The supersaturation-filter array is a named DomainState scratch field,
    registered as write-before-read in preflight.  Number and CCN moments are
    passed directly in NSSL's internal #/m3 convention; no Registry round trip
    or density conversion occurs between GS and condensation.
    """
    if fused_gs is None:
        raise NSSL2ProductionConfigurationError(
            "required NSSL default hook 'fused_gs' is absent")
    if not callable(fused_gs):
        raise TypeError("NSSL default hook 'fused_gs' must be callable")
    if not isinstance(validate_values, bool):
        raise TypeError("validate_values must be bool")
    require_ported_nssl2_mode(mode)
    step = _step32(dt_s)

    shape = _required_state_shape(state)
    pressure = state.p
    w_interface = state.w
    try:
        supersaturation_scratch = state.scratch(
            shape, "nssl2_nucond_ss")
    except AttributeError as exc:
        raise NSSL2ProductionConfigurationError(
            "NSSL default hooks require DomainState.scratch") from exc
    _validate_scratch(supersaturation_scratch, shape)

    def nucond_qvexcess(
            workspace: NSSL2DriverWorkspace,
            fields: NSSL2RuntimeFields, /) -> None:
        if not isinstance(workspace, NSSL2DriverWorkspace):
            raise TypeError("workspace must be NSSL2DriverWorkspace")
        if not isinstance(fields, NSSL2RuntimeFields):
            raise TypeError("fields must be NSSL2RuntimeFields")
        if workspace.shape != shape:
            raise ValueError(
                f"workspace has shape {workspace.shape}, expected {shape}")
        if fields.pressure is not pressure or fields.w is not w_interface:
            raise NSSL2ProductionConfigurationError(
                "NSSL default hooks received fields from a different "
                "DomainState")
        launch_nucond(
            fields.theta,
            fields.rho,
            fields.pressure,
            fields.pii,
            fields.w,
            workspace.field("qv"),
            workspace.field("qc"),
            workspace.field("qr"),
            workspace.field("qi"),
            workspace.field("qs"),
            workspace.field("qndrop"),
            workspace.field("qnr"),
            workspace.field("qni"),
            workspace.field("qns"),
            workspace.field("qnn"),
            step,
            supersaturation_scratch=supersaturation_scratch,
            concentration_space=True,
            predicted_ccn=mode.predicted_ccn,
            nssl_cccn=nssl_cccn, shapes=shapes,
            validate_values=validate_values,
            frozen=(
                workspace.field("qg"),
                workspace.field("qh"),
                workspace.field("qng"),
                workspace.field("qnh"),
                workspace.field("qvolg"),
                workspace.field("qvolh"),
            ),
        )

    return NSSL2RuntimeHooks(
        fused_gs=fused_gs,
        nucond_qvexcess=nucond_qvexcess,
    )


@dataclass(frozen=True)
class NSSL2ProductionBinding:
    """Restart-rebuilt, domain-owned MP18 callbacks and device buffers."""

    state: object
    shape: tuple[int, int, int]
    dt_s: np.float32
    workspace: NSSL2DriverWorkspace
    fused_gs: object
    hooks: NSSL2RuntimeHooks
    nucond_scratch: object
    mode: NSSL2Mode = DEFAULT_MODE
    nssl_cccn: float = 0.5e9
    shapes: tuple[float, float] = (0.0, 1.0)

    def validate(self, state, dt_s: float, /) -> None:
        """Fail before mutation if ownership, structure, or step drifted."""
        if state is not self.state:
            raise NSSL2ProductionConfigurationError(
                "NSSL production binding belongs to a different DomainState")
        step = _step32(dt_s)
        if self.dt_s.dtype != np.dtype(np.float32):
            raise TypeError("NSSL production binding dt_s must be float32")
        if self.dt_s != step:
            raise NSSL2ProductionConfigurationError(
                "NSSL production binding timestep differs from runtime: "
                f"{float(self.dt_s)} != {float(step)}")
        if tuple(_required_state_shape(state)) != tuple(self.shape):
            raise NSSL2ProductionConfigurationError(
                "NSSL production binding shape differs from DomainState")
        if self.hooks.fused_gs is not self.fused_gs:
            raise NSSL2ProductionConfigurationError(
                "NSSL production binding fused hook identity changed")
        if getattr(self.fused_gs, "dt_s", None) != self.dt_s:
            raise NSSL2ProductionConfigurationError(
                "NSSL fused hook timestep differs from binding")
        require_ported_nssl2_mode(self.mode)
        if getattr(self.fused_gs, "hail_on", None) is not self.mode.hail:
            raise NSSL2ProductionConfigurationError(
                "NSSL fused hook hail switch differs from the resolved "
                "variant mode")

        if getattr(self.fused_gs, "shapes", (0.0, 1.0)) != self.shapes:
            raise NSSL2ProductionConfigurationError("NSSL fused hook shape coefficients differ from binding")
        validate_nssl2_driver_workspace(self.workspace, self.shape)
        pool = getattr(state, "_scratch", None)
        if not isinstance(pool, dict):
            raise NSSL2ProductionConfigurationError(
                "NSSL production binding requires DomainState scratch registry")
        expected = {
            NSSL2_DRIVER_STATE_SCRATCH: (self.workspace.state, (16, *self.shape)),
            NSSL2_DRIVER_SURFACE_EXPORT_SCRATCH: (
                self.workspace.category_surface_export, (5, *self.shape[1:])),
            NSSL2_DRIVER_IGNORED_ACCUMULATOR_SCRATCH: (
                self.workspace.ignored_accumulator, self.shape[1:]),
            "nssl2_fused_temperature": (self.fused_gs.temperature_k, self.shape),
            "nssl2_primary_ice_target": (
                self.fused_gs.primary_ice_target_m3, self.shape),
            NSSL2_NUCOND_SCRATCH: (self.nucond_scratch, self.shape),
        }
        from woof.core.nssl2_shape import requires_cd_carry
        if requires_cd_carry(self.shapes):
            if self.workspace.drag_cd is None:
                raise NSSL2ProductionConfigurationError(
                    "NSSL gathered CD scratch was not prepared")
            nz, ny, nx = self.shape
            expected["nssl2_drag_cd_carry"] = (
                self.workspace.drag_cd, (7, nz + 1, ny, nx))
        for name, (value, shape) in expected.items():
            if pool.get(name) is not value:
                raise NSSL2ProductionConfigurationError(
                    f"NSSL production binding scratch {name} lost its "
                    "canonical DomainState identity")
            _validate_scratch(value, shape, name)

    def with_step(self, dt_s: float, /) -> "NSSL2ProductionBinding":
        """This binding at another model step, sharing every buffer.

        The step is the only thing the fused GS callback and the
        NUCOND+QVEXCESS hook take from construction; the workspace, the
        fused-GS t0/t7 scratch and the supersaturation filter are all
        named DomainState scratch that the kernels fully overwrite each
        call.  A changed step (the adaptive clock moves it every root
        step) therefore rebinds the two callbacks and keeps the buffers,
        which costs two small Python objects and no device work.

        Returns ``self`` when the float32 step is unchanged, so a fixed
        clock never builds anything.
        """
        step = _step32(dt_s)
        if step == self.dt_s:
            return self
        fused_gs = replace(self.fused_gs, dt_s=step)
        hooks = make_nssl2_default_runtime_hooks(
            self.state, step, fused_gs, validate_values=False,
            mode=self.mode, nssl_cccn=self.nssl_cccn, shapes=self.shapes)
        binding = replace(self, dt_s=step, fused_gs=fused_gs, hooks=hooks)
        binding.validate(self.state, step)
        return binding


def make_nssl2_production_binding(
        state, dt_s: float, *,
        mode: NSSL2Mode = DEFAULT_MODE,
        nssl_cccn: float = 0.5e9, shapes=(0.0, 1.0)) -> NSSL2ProductionBinding:
    """Build the one persistent MP18 binding for a domain.

    ``mode`` is the resolved NSSL variant from
    :func:`woof.core.nssl2_contract.resolve_nssl2_mode`.  It is frozen into
    the binding, so a domain cannot change hail or CCN treatment mid-run:
    ``validate`` re-checks that the fused hook still carries the same hail
    switch on every step.  The step is not frozen: ``dt_s`` is the step the
    domain starts on, and :meth:`NSSL2ProductionBinding.with_step` moves
    the binding to any later one.
    """
    if not mode.two_moment:
        from woof.core.nssl1 import make_binding
        return make_binding(state,dt_s,mode,nssl_cccn=nssl_cccn,shapes=shapes)
    require_ported_nssl2_mode(mode)
    from woof.core.nssl2_shape import resolved_shapes, requires_cd_carry
    shapes = resolved_shapes(*shapes)
    step = _step32(dt_s)
    shape = _required_state_shape(state)
    nz, ny, nx = shape
    try:
        driver_state = state.scratch(
            (16, nz, ny, nx), "nssl2_driver_state")
        surface_export = state.scratch(
            (5, ny, nx), "nssl2_driver_surface_export")
        ignored_accumulator = state.scratch(
            (ny, nx), "nssl2_driver_ignored_accumulator")
        fused_temperature = state.scratch(
            shape, "nssl2_fused_temperature")
        primary_ice_target = state.scratch(
            shape, "nssl2_primary_ice_target")
    except AttributeError as exc:
        raise NSSL2ProductionConfigurationError(
            "NSSL production binding requires DomainState.scratch") from exc

    workspace = NSSL2DriverWorkspace(
        state=driver_state,
        category_surface_export=surface_export,
        shape=shape,
        ignored_accumulator=ignored_accumulator,
        drag_cd=(state.scratch((7,nz+1,ny,nx),"nssl2_drag_cd_carry")
                 if requires_cd_carry(shapes) else None),
    )
    from woof.core.nssl2_fused_gs import NSSL2FusedGS
    fused_gs = NSSL2FusedGS(
        temperature_k=fused_temperature,
        primary_ice_target_m3=primary_ice_target,
        dt_s=step,
        hail_on=mode.hail, shapes=shapes,
    )
    hooks = make_nssl2_default_runtime_hooks(
        state, step, fused_gs, validate_values=False, mode=mode, nssl_cccn=nssl_cccn, shapes=shapes)
    binding = NSSL2ProductionBinding(
        state=state,
        shape=shape,
        dt_s=step,
        workspace=workspace,
        fused_gs=fused_gs,
        hooks=hooks,
        nucond_scratch=state._scratch[NSSL2_NUCOND_SCRATCH],
        mode=mode,
        nssl_cccn=nssl_cccn, shapes=shapes,
    )
    binding.validate(state, step)
    return binding


__all__ = [
    "NSSL2_NUCOND_SCRATCH",
    "NSSL2ProductionBinding",
    "make_nssl2_default_runtime_hooks",
    "make_nssl2_production_binding",
]
