"""Physics-entry rendezvous for concurrent original atmospheric steps.

The original callback refreshes its own clock and cadence exactly once.
Metadata is captured when all members reach physics, after diagnostics.
The genuine driver owner is restored before the callback can publish output.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading

import numpy as np

from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
from woof.ensemble.packed_production_physics import PackedProductionPhysics


def production_physics_memory_plan(drivers, configs, *, reuse_original_workspaces=False):
    """Declare gather banks, returns, leaf workspaces and call reservations.

    Shared allocations here belong to one all-member call owner and contain
    separate member stripes. They never broadcast one member's field. CUDA
    context, events, original forecasts and allocator retention are separate
    caller reservations. This reads live array metadata, never field words.
    """
    from woof.core.physics_inventory import MYNN_SURFACE_OUTPUTS, radiation_boundary_flux_names
    from woof.core.mynn_sfclay import MYNN_SURFACE_INPUTS
    from woof.core.mynn_pbl_runtime import (_ATMOSPHERE_LAYERS, _COLUMN_FIELDS, _STATE_FIELD,
        MYNN_PBL_DIAGNOSTICS_2D, MYNN_PBL_DIAGNOSTICS_INT_2D)
    from woof.core.mynn_pbl_scratch import resolve_mynn_column_chunk
    from woof.core.ruc_runtime import RUC_STATE_BINDING, RUC_PROFILE_BINDING
    from woof.core import ruc_memory as layout
    from woof.core.surface_forcing import SURFACE_PRECIPITATION_FIELDS
    from woof.ensemble.batch_mynn import mynn_pbl_output_plan
    from woof.ensemble.batch_ruc import workspace_allocations
    from woof.ensemble.batch_rrtmgp import rrtmgp_call_memory_inventory
    drivers, configs = tuple(drivers), tuple(configs)
    if not drivers or len(drivers) != len(configs):
        raise ValueError("physics memory declaration needs every original driver and configuration")
    if type(reuse_original_workspaces) is not bool:
        raise TypeError("original workspace reuse needs an explicit boolean ownership declaration")
    members, driver, cfg = len(drivers), drivers[0], configs[0]
    ny, nx, nz = int(cfg.ny), int(cfg.nx), int(cfg.nz)
    specs, reserve = [], 0
    def bank(namespace, name, shape, dtype="float32", joined=True):
        shape = tuple(shape)
        actual = ((shape[0], members * shape[1], shape[2])
                  if joined and len(shape) == 3 else (members,) + shape)
        specs.append(BatchArraySpec(f"{namespace}:{name}", actual, "shared", dtype))
    def field_banks(namespace, names, *, joined=True):
        for name in sorted(names):
            value = driver.fields[name]
            bank(namespace, name, value.shape, value.dtype.str, joined)
    surface_calls = 2 if driver.mynn_sfclay_sea_result is not None else 1
    for occurrence in range(surface_calls):
        namespace = f"surface:{('surface', occurrence)}"
        for name in MYNN_SURFACE_INPUTS:
            bank(namespace + ":inputs", name, (ny, nx))
        for name in MYNN_SURFACE_OUTPUTS:
            bank(namespace + ":outputs", name, (ny, nx))
        for name in ("mol", "ustm"):
            bank(namespace, name, (ny, nx))
    namespace = f"land:{('land', 0)}"
    supplied = set(RUC_STATE_BINDING.values()) | set(RUC_PROFILE_BINDING.values()) | {
        "z3d", "p8w", "t3d", "qv3d", "qc3d", "rho3d", "frzfrac", "tbot", "rainncv", "snowncv", "graupelncv"}
    names = (set(RUC_STATE_BINDING) | set(RUC_PROFILE_BINDING) | set(layout.DRIVER_EXTRAS)
        | (set(layout.DRIVER_INPUT_NAMES) - supplied) | set(SURFACE_PRECIPITATION_FIELDS)
        | {"ivgtyp", "isltyp", "sr", "tmn", "psfc", "t2", "th2", "q2", "albbck", "chs", "flhc", "flqc"}
        | {"ruc_" + name for name in ("infiltr", "smelt", "runoff1", "runoff2")})
    if cfg.ruc_irrigation == "wrf_45" and driver.fields.get("landusef") is not None:
        names.add("landusef")
    field_banks(namespace + ":fields", names, joined=False)
    for name in ("dz", "pressure", "temperature", "qv", "qc", "rho"):
        bank(namespace + ":atmosphere", name, (nz, ny, nx), joined=False)
    for name, (shape, dtype) in workspace_allocations(members, (ny, nx), driver.fields["smois"].shape[0]).items():
        specs.append(BatchArraySpec(namespace + ":workspace:" + name, shape, "shared", dtype))
    namespace = f"pbl:{('pbl', 0)}"
    names = set(_STATE_FIELD.values()) | {name for _, name in _COLUMN_FIELDS} | {
        "exch_h", "exch_m", "rmol", "pblh", "kpbl", *MYNN_PBL_DIAGNOSTICS_2D, *MYNN_PBL_DIAGNOSTICS_INT_2D}
    field_banks(namespace + ":fields", names)
    for name in sorted({name for _, name in _ATMOSPHERE_LAYERS}):
        bank(namespace + ":atmosphere", name, (nz, ny, nx))
    bank(namespace, "w", (nz + 1, ny, nx))
    for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi"):
        bank(namespace + ":returned", name, (nz, ny, nx), joined=False)
    chunk = getattr(driver.state, "_mynn_rank_column_chunk", None)
    if chunk is None:
        chunk = resolve_mynn_column_chunk(nz)
    pbl = mynn_pbl_output_plan(nz=nz, ny=ny, nx=nx, members=members, column_chunk=chunk,
                              bl_mynn_version=cfg.bl_mynn_version,
                              borrow_scratch=reuse_original_workspaces and ny * nx >= chunk)
    reserve += pbl.reserved_bytes
    specs.extend(BatchArraySpec(namespace + ":workspace:" + spec.name, spec.shape,
                               spec.ownership, spec.dtype) for spec in pbl.arrays)
    namespace = f"radiation:{('radiation', 0)}"
    for name in ("pressure", "temperature", "exner", "qv", "qc", "qi"):
        bank(namespace + ":atmosphere", name, (nz, ny, nx))
    bank(namespace + ":atmosphere", "p_interface", (nz + 1, ny, nx))
    field_banks(namespace + ":fields", (name for name in
        ("tsk", "albedo", "emiss", "glw", "xland", "swddir", "qc_bl", "qi_bl", "cldfra_bl") if name in driver.fields))
    for name in ("qc", "qr", "qi", "qs", "effc", "effr", "effi", "effs", "nc", "nr", "ni", "ns", "qnc", "qnr", "qni", "qns"):
        value = getattr(driver.state, name, None)
        if value is not None:
            bank(namespace + ":state", name, value.shape, value.dtype.str)
    for name in ("rthratenlw", "rthratensw", "cldfra"):
        bank(namespace + ":returned", name, (nz, ny, nx), joined=False)
    for name in ("swdown", "glw", "gsw", "coszen", "olr", "swddir", "swddif"):
        bank(namespace + ":returned", name, (ny, nx), joined=False)
    # The packed adapter splits each named top/bottom flux into its own
    # contiguous member bank, in addition to the native flux-stack reserve.
    for name in radiation_boundary_flux_names(cfg):
        bank(namespace + ":returned", "boundary_fluxes/" + name, (ny, nx), joined=False)
    adapter = driver.radiation_callable
    native_workspace = getattr(adapter, "chunk_workspace", None) is not None
    memory = rrtmgp_call_memory_inventory(cfg, members=members, ny=ny, nx=nx,
        p_top=driver.state.p_top, column_chunk=adapter.column_chunk, native_workspace=native_workspace)
    for name in ("latitude", "longitude"):
        specs.append(BatchArraySpec(namespace + ":geometry:" + name, (ny, nx), "member"))
    if native_workspace and not reuse_original_workspaces:
        specs.append(BatchArraySpec(namespace + ":workspace", (memory["solver_workspace_bytes"],), "shared", "uint8"))
    reserve += (memory["column_transient_bytes"] + memory["additional_call_buffer_bytes"]
                + memory["workspace_less_chunk_envelope_bytes"])
    device = int(adapter.latitude_deg.device.id)
    for table_name in ("lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables"):
        table = getattr(adapter, table_name)
        if device not in table._device:
            for name, value in vars(table).items():
                if isinstance(value, np.ndarray) and value.size:
                    dtype = "uint8" if value.dtype == np.dtype("bool") else "float32"
                    specs.append(BatchArraySpec(namespace + f":new_table:{table_name}:{name}", value.shape, "shared", dtype))
    if int(getattr(cfg, "cu_physics", 0)) == 99:
        from woof.ensemble.batch_kf import kf_old_column_layout, kf_old_call_reservation
        namespace = f"cumulus:{('cumulus', 0)}"
        inputs, outputs = kf_old_column_layout(nz, ny, nx, edrates=bool(cfg.kf_edrates))
        for name, (shape, dtype) in inputs.items():
            bank(namespace + ":inputs", name, shape, dtype)
        for name, (shape, dtype) in outputs.items():
            bank(namespace + ":returned", name, shape, dtype, joined=False)
        reserve += kf_old_call_reservation(nz, ny, nx, members=members,
            edrates=bool(cfg.kf_edrates))
    return BatchMemoryPlan(tuple(specs), reserved_bytes=reserve)


class _ScheduledDriver:
    """Replace compute only while delegating every remaining ordinary method."""
    def __init__(self, group, member):
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_member", member)

    def __getattr__(self, name):
        return getattr(self._group.owner.drivers[self._member], name)

    def __setattr__(self, name, value):
        setattr(self._group.owner.drivers[self._member], name, value)

    def compute(self, state, cfg):
        return self._group._compute(self._member, state, cfg)


class ScheduledProductionPhysics:
    """Arm one equal-frontier group and join at its actual physics entry."""
    def __init__(self, drivers, configs, *, available_bytes, array_module=None,
                 reuse_original_workspaces=False):
        drivers, configs = tuple(drivers), tuple(configs)
        self.plan = production_physics_memory_plan(drivers, configs,
            reuse_original_workspaces=reuse_original_workspaces)
        self.plan.admit(len(drivers), available_bytes=available_bytes)
        self.owner = PackedProductionPhysics(drivers,
            available_bytes=self.plan.required_bytes(len(drivers)), array_module=array_module,
            reuse_original_workspaces=reuse_original_workspaces)
        self._initialize_coordination()

    def _initialize_coordination(self):
        self.proxies = tuple(_ScheduledDriver(self, member) for member in range(self.owner.members))
        self._condition, self._armed, self._error = threading.Condition(), False, None
        self.receipt = {"complete_forecast_qualified": False, "entry_groups": 0,
            "physics_entry": "after original before_step and diagnostic refresh",
            "clock_policy": "unchanged original callbacks and live configurations"}

    def arm(self):
        with self._condition:
            if self._armed or self.owner._active:
                raise RuntimeError("scheduled physics group is already armed")
            self._configs, self._callbacks_finished, self._error = {}, set(), None
            self._armed = True

    def abort(self, error):
        with self._condition:
            if self._error is None:
                self._error = error
            if self.owner._active:
                self.owner.abort(error)
            self._condition.notify_all()

    def callback_finished(self, member):
        """Wake entry waiters even when a callback never reached dycore."""
        if not isinstance(member, int) or isinstance(member, bool) or not 0 <= member < self.owner.members:
            raise ValueError("scheduled callback completion needs its original member slot")
        with self._condition:
            self._callbacks_finished.add(member)
            self._condition.notify_all()

    @contextmanager
    def member_scope(self, member, state=None):
        if not self._armed or not 0 <= member < self.owner.members:
            raise RuntimeError("scheduled physics proxy needs an armed original member callback")
        original = self.owner.drivers[member]
        state = original.state if state is None else state
        if state is not original.state or state.physics is not original:
            raise ValueError("scheduled physics state/driver owner changed; using the former member binding would substitute a rebuilt owner")
        state.physics = self.proxies[member]
        try:
            yield
        except BaseException as error:
            self.abort(error)
            raise
        finally:
            state.physics = original
            self.callback_finished(member)

    def wrap_step(self, member, ordinary_step):
        def step(state, cfg, **kwargs):
            with self.member_scope(member, state):
                return ordinary_step(state, cfg, **kwargs)
        return step

    def _compute(self, member, state, cfg):
        with self._condition:
            if not self._armed or member in self._configs or state is not self.owner.drivers[member].state:
                error = ValueError("scheduled physics entry did not match one original solve per member")
                self.abort(error)
                raise error
            self._configs[member] = cfg
            if len(self._configs) == self.owner.members:
                try:
                    self.owner.begin(tuple(self._configs[index] for index in range(self.owner.members)))
                except BaseException as error:
                    self.abort(error)
                self._condition.notify_all()
            while not self.owner._active and self._error is None:
                if self._callbacks_finished - self._configs.keys():
                    self.abort(ValueError("an original callback finished without reaching its physics entry; no equal-cadence member group exists"))
                    break
                self._condition.wait()
            if self._error is not None:
                raise self._error
        return self.owner.member_compute(member, state, cfg)

    def finish(self):
        if self._callbacks_finished != set(range(self.owner.members)):
            raise RuntimeError("scheduled physics finish precedes an original member callback's completion")
        error = self._error
        try:
            if self.owner._active:
                try:
                    self.owner.finish()
                except BaseException as completion_error:
                    if error is None:
                        raise
                    if completion_error is not error:
                        error.add_note(f"physics queue completion also failed: {type(completion_error).__name__}: {completion_error}")
            elif error is None:
                raise RuntimeError("original callbacks completed without one joined physics entry per member")
            if error is not None:
                raise error
            self.receipt["entry_groups"] += 1
        finally:
            self._armed = False

    def close(self):
        if self._armed:
            raise RuntimeError("scheduled physics cannot release an armed callback group")
        self.owner.close()


__all__ = ["ScheduledProductionPhysics", "production_physics_memory_plan"]
