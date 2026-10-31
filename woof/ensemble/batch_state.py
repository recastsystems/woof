"""Prepared-host dycore state packing with explicit member ownership.

This is an internal allocation/copy seam, not a forecast driver or complete
forecast admission model. Shape authority stays in ``state_array_shapes`` and
``scratch_slot_registry``. Callers supply already prepared NumPy arrays; no
single-member CUDA states are retained alongside the new device backings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timezone
from operator import index
import struct
from types import MappingProxyType

import numpy as np

from woof.core.device_inventory import state_array_shapes
from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan, BatchStorage


# Candidates only. A field is member-owned until a caller requests sharing
# and every already prepared member buffer has the same dtype, shape and bytes.
SHARED_STATE_CANDIDATES = frozenset({
    "ht", "mub2d", "thb", "pb", "alb", "phb", "dphb_resid",
    "msft", "msfu", "msfv", "msfv_inv", "f", "e", "sina", "cosa",
    "c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f",
    "dc3f", "dc4f", "dnw", "rdnw", "dn", "rdn", "fnp", "fnm",
    "znu", "znw",
})
_REQUIRED_SCALARS = frozenset({
    "mub", "p_top", "cf1", "cf2", "cf3", "cfn", "cfn1",
    "has_msf", "rotational", "elapsed_seconds",
})
_CLOCK_FIELDS = frozenset({
    "ticks", "step_ticks", "tick_den", "run_ticks", "step_count",
    "dt_fp32", "dtbc_fp32",
})
_CONTROLS = frozenset({
    "physics", "lateral_boundaries", "_scratch", "_scratch_arena",
    "_host_setup_state", "_phb_host",
    "_ensemble_surface_state",
})
_BATCH_ATTRIBUTES = frozenset({
    "cfg", "storage", "plan", "members", "clock", "scalars",
    "phb_host_members", "scratch", "existing_scratch", "member_view",
    "scratch_member_view", "from_prepared",
})
_SCRATCH_PREFIX = "scratch:"


class BatchStateUnsupported(ValueError):
    """Prepared state cannot be packed without changing member semantics."""


def _temporal_key(value):
    """Typed calendar identity, retaining wall time, fold and timezone policy."""
    if type(value) is date:
        return ("date", value.isoformat())
    if type(value) is not datetime:
        return None
    zone = value.tzinfo
    if zone is None:
        zone_key = None
    elif type(zone) is timezone:
        offset = zone.utcoffset(value)
        zone_key = ("fixed_offset", offset.days, offset.seconds, offset.microseconds, value.tzname())
    else:
        from zoneinfo import ZoneInfo
        if type(zone) is not ZoneInfo or not isinstance(zone.key, str):
            raise BatchStateUnsupported("calendar timezone has no immutable named transition authority; joined members cannot substitute future calendar transitions")
        zone_key = ("zoneinfo", zone.key)
    return ("datetime", value.isoformat(timespec="microseconds"), value.fold, zone_key)


def _exact_key(value):
    """Typed byte comparison, including NaN payloads and signed scalar zero."""
    if isinstance(value, np.generic):
        return ("numpy", value.dtype.str, value.tobytes())
    calendar = _temporal_key(value)
    if calendar is not None:
        return calendar
    if type(value) is float:
        return ("float", struct.pack("!d", value))
    if value is None or type(value) in (bool, int, str, bytes):
        return (type(value).__name__, value)
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value).__module__, type(value).__qualname__, tuple(
            (field.name, _exact_key(getattr(value, field.name)))
            for field in fields(value)))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("state metadata needs string keys")
        return ("mapping", tuple((key, _exact_key(value[key])) for key in sorted(value)))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_exact_key(item) for item in value))
    raise BatchStateUnsupported(
        f"unclassified state metadata type {type(value).__name__}; its mutable "
        "state has no member ownership contract")


def _clock_snapshot(clock):
    if isinstance(clock, Mapping):
        snapshot = dict(clock)
    elif hasattr(clock, "__slots__"):
        snapshot = {name: getattr(clock, name) for name in clock.__slots__}
    else:
        raise BatchStateUnsupported("batch packing needs an explicit integer clock snapshot")
    missing = _CLOCK_FIELDS - snapshot.keys()
    if missing:
        raise BatchStateUnsupported(f"clock snapshot omits {sorted(missing)}; member timing cannot be verified")
    if snapshot.get("adaptive_state") is not None:
        raise BatchStateUnsupported("adaptive member clock state needs a schedule identity proof before batching")
    for name in ("tick_den", "step_ticks", "run_ticks", "ticks", "step_count"):
        value = snapshot[name]
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"clock {name} must be an integer")
        value = index(value)
        if value < 0 or (name in ("tick_den", "step_ticks") and value == 0):
            raise ValueError(f"clock {name} has invalid value {value}")
    if snapshot["ticks"] > snapshot["run_ticks"]:
        raise ValueError("clock ticks exceed the declared run interval")
    for name in ("dt_fp32", "dtbc_fp32"):
        if not isinstance(snapshot[name], np.float32):
            raise TypeError(f"clock {name} must retain its float32 scalar bytes")
        if not np.isfinite(snapshot[name]):
            raise ValueError(f"clock {name} must be finite")
    if snapshot["dt_fp32"] <= 0 or snapshot["dtbc_fp32"] < 0:
        raise ValueError("clock step must be positive and boundary age must be nonnegative")
    _exact_key(snapshot)
    return snapshot


def _host_array(value, shape, name, *, dtype=np.float32):
    if not isinstance(value, np.ndarray):
        raise BatchStateUnsupported(
            f"{name} needs an already prepared host array; packing resident "
            "single-member device states would retain an unpriced second live set")
    if tuple(value.shape) != tuple(shape) or value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} shape/dtype {value.shape}/{value.dtype} differs from {tuple(shape)}/{np.dtype(dtype)}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} needs a contiguous prepared host buffer; implicit layout conversion is not packing")
    return value


def _copy_host(destination, source):
    """Copy directly into the admitted backing, without a device input slab."""
    if isinstance(destination, np.ndarray):
        np.copyto(destination, source, casting="no")
    else:
        setter = getattr(destination, "set", None)
        if setter is None:
            raise BatchStateUnsupported("device backing needs a direct host set operation; implicit array conversion would allocate an unpriced device input slab")
        setter(source)


def _finish_host_copies(storage, array_module):
    if all(isinstance(value, np.ndarray) for value in storage.arrays.values()):
        return
    cuda = getattr(array_module, "cuda", None)
    stream = getattr(cuda, "get_current_stream", None)
    if stream is None:
        raise BatchStateUnsupported("device array module needs current-stream synchronization before prepared host buffers can be released")
    stream().synchronize()


def _shared_names(cfg, names):
    names = tuple(names)
    if len(set(names)) != len(names):
        raise ValueError("duplicate shared fields obscure the ownership inventory")
    unknown = set(names) - SHARED_STATE_CANDIDATES
    if unknown:
        raise BatchStateUnsupported(f"{sorted(unknown)} are not audited shared state candidates")
    absent = set(names) - state_array_shapes(cfg).keys()
    if absent:
        raise ValueError(f"shared fields {sorted(absent)} are absent from this configuration")
    return frozenset(names)


def state_array_specs(cfg, *, shared_fields=()):
    """Derive allocation shapes once from the existing constructor inventory.

    Default ownership is member for every field, including new inventory rows.
    These declarations alone do not establish that sharing is safe.
    """
    shared = _shared_names(cfg, shared_fields)
    return tuple(BatchArraySpec(name, shape, "shared" if name in shared else "member")
                 for name, shape in state_array_shapes(cfg).items())


@dataclass(frozen=True)
class PreparedHostMember:
    """Caller-owned CPU preparation, with a complete array and scalar inventory.

    Inputs must remain unchanged during packing. Direct GPU copies complete on
    the current stream before packing returns, without temporary device input
    slabs. Callers remain responsible for pinned-host memory reservations.
    The resulting batch retains no reference to these inputs.
    ``phb_host`` is the already prepared host cache, never reconstructed here.
    """

    cfg: object
    arrays: Mapping[str, np.ndarray]
    scalars: Mapping[str, object]
    clock: object
    scratch: Mapping[str, np.ndarray] | None = None
    phb_host: np.ndarray | None = None


def _state_specs(cfg, shared_fields, extra_specs):
    """Add explicitly priced member carriers without changing scalar inventory."""
    specs = state_array_specs(cfg, shared_fields=shared_fields)
    extra_specs = tuple(extra_specs)
    occupied = {spec.name for spec in specs} | _CONTROLS | _BATCH_ATTRIBUTES
    for spec in extra_specs:
        if not isinstance(spec, BatchArraySpec):
            raise TypeError("additional state carriers need BatchArraySpec declarations")
        if (not spec.name.isidentifier() or spec.name.startswith("_")
                or spec.name in occupied):
            raise ValueError(f"additional carrier {spec.name!r} collides with state ownership")
        if spec.ownership != "member":
            raise BatchStateUnsupported("additional mutable carriers need independent member ownership")
        occupied.add(spec.name)
    return specs + extra_specs


def _validate_prepared(members, shared_fields, scratch_slots, extra_specs=()):
    members = tuple(members)
    if not members or any(not isinstance(member, PreparedHostMember) for member in members):
        raise TypeError("batch packing needs one PreparedHostMember per member")
    cfg = members[0].cfg
    if getattr(cfg, "use_adaptive_time_step", False):
        raise BatchStateUnsupported("adaptive timesteps could change a member trajectory; this state packing path uses a fixed common clock")
    specs = _state_specs(cfg, shared_fields, extra_specs)
    shapes = {spec.name: spec.shape for spec in specs}
    dtypes = {spec.name: spec.dtype for spec in specs}
    cfg_key = _exact_key(cfg)
    scalar_key = None
    clock_key = None
    clocks = []
    scratch_sets = []
    for member_index, member in enumerate(members):
        if _exact_key(member.cfg) != cfg_key:
            raise BatchStateUnsupported(f"member {member_index} configuration/grid differs; one batch ABI cannot use both")
        missing, extra = shapes.keys() - member.arrays.keys(), member.arrays.keys() - shapes.keys()
        if missing or extra:
            raise BatchStateUnsupported(f"member {member_index} array inventory differs: missing {sorted(missing)}, unclassified {sorted(extra)}")
        for name, shape in shapes.items():
            _host_array(member.arrays[name], shape, name, dtype=dtypes[name])
        missing_scalars = _REQUIRED_SCALARS - member.scalars.keys()
        if missing_scalars:
            raise BatchStateUnsupported(f"member {member_index} scalar inventory omits {sorted(missing_scalars)}")
        collision = set(member.scalars) & (set(shapes) | _CONTROLS | _BATCH_ATTRIBUTES)
        if collision:
            raise BatchStateUnsupported(f"scalar metadata collides with state ownership: {sorted(collision)}")
        this_scalar_key = _exact_key(member.scalars)
        snapshot = _clock_snapshot(member.clock)
        this_clock_key = _exact_key(snapshot)
        if member_index and this_scalar_key != scalar_key:
            raise BatchStateUnsupported(f"member {member_index} scalar metadata differs in value/type/bytes; one launch would substitute another member's scalar")
        if member_index and this_clock_key != clock_key:
            raise BatchStateUnsupported(f"member {member_index} clock differs; a common step would change its schedule")
        with np.errstate(over="ignore", invalid="ignore"):
            configured_dt = np.float32(cfg.dt)
        if (not np.isfinite(configured_dt) or configured_dt <= 0
                or configured_dt.tobytes() != snapshot["dt_fp32"].tobytes()):
            raise BatchStateUnsupported(
                f"member {member_index} clock step differs from configuration; "
                "kernels and output calendars would advance different intervals")
        if snapshot["step_ticks"] / snapshot["tick_den"] != float(cfg.dt):
            raise BatchStateUnsupported(
                f"member {member_index} integer clock step differs from configuration; "
                "output ticks and numerical elapsed seconds would diverge")
        if float(member.scalars["elapsed_seconds"]) != snapshot["ticks"] / snapshot["tick_den"]:
            raise BatchStateUnsupported(
                f"member {member_index} elapsed seconds differ from clock ticks; "
                "boundary selection would use a different initial time")
        scalar_key, clock_key = this_scalar_key, this_clock_key
        clocks.append(snapshot)
        scratch_sets.append(dict(member.scratch or {}))
        if member.phb_host is not None:
            if not isinstance(member.phb_host, np.ndarray):
                raise TypeError("phb_host must be the prepared host cache")
            if member.phb_host.shape != shapes["phb"] or member.phb_host.dtype.kind != "f":
                raise ValueError("phb_host shape/type differs from the prepared base geopotential")
    for spec in specs:
        if spec.ownership == "shared":
            reference = members[0].arrays[spec.name].tobytes(order="C")
            for member_index, member in enumerate(members[1:], 1):
                if member.arrays[spec.name].tobytes(order="C") != reference:
                    raise BatchStateUnsupported(f"shared {spec.name} differs in member {member_index}; sharing would change its single-run bytes")
    if any(set(scratch) != set(scratch_sets[0]) for scratch in scratch_sets[1:]):
        raise BatchStateUnsupported("member scratch inventories differ; a retained scratch carrier would be missing")
    # Shapes remain the existing registry's authority. Existing sequential
    # lifetime aliases are intentionally not reused across simultaneous members.
    from woof.core.preflight import scratch_slot_registry
    registry = scratch_slot_registry(cfg)
    requested = dict(scratch_slots or {})
    for slot in scratch_sets[0]:
        requested.setdefault(slot, scratch_sets[0][slot].dtype)
    unknown_slots = set(requested) - registry.keys()
    if unknown_slots:
        raise BatchStateUnsupported(f"unclassified scratch slots {sorted(unknown_slots)} have no batch allocation shape")
    scratch_specs = []
    for slot, dtype in requested.items():
        spec = BatchArraySpec(_SCRATCH_PREFIX + slot, registry[slot], "member", dtype=dtype)
        for scratch in scratch_sets:
            if slot in scratch:
                _host_array(scratch[slot], spec.shape, slot, dtype=spec.dtype)
        scratch_specs.append(spec)
    return members, specs + tuple(scratch_specs), clocks[0], tuple(requested)


class BatchedDomainState:
    """Array container for an eventual batched dycore driver, with no stepping.

    ``plan`` covers these state/scratch allocations and the caller's explicit
    reserve only. It does not price forcing, physics, products, output queues,
    kernel-local memory, CUDA context or allocator retention by itself.
    """

    @classmethod
    def from_prepared(cls, members: Sequence[PreparedHostMember], *, array_module,
                      available_bytes, shared_fields=(), scratch_slots=None,
                      reserved_bytes=0, allocation_quantum=512, extra_specs=()):
        members, specs, clock, slots = _validate_prepared(
            members, shared_fields, scratch_slots, extra_specs)
        plan = BatchMemoryPlan(specs, reserved_bytes=reserved_bytes,
                               allocation_quantum=allocation_quantum)
        storage = BatchStorage(plan, len(members), array_module=array_module,
                               available_bytes=available_bytes)
        try:
            for spec in specs:
                if spec.name.startswith(_SCRATCH_PREFIX):
                    slot = spec.name[len(_SCRATCH_PREFIX):]
                    for member_index, member in enumerate(members):
                        value = (member.scratch or {}).get(slot)
                        if value is not None:
                            _copy_host(storage.arrays[spec.name][member_index], value)
                elif spec.ownership == "shared":
                    _copy_host(storage.arrays[spec.name], members[0].arrays[spec.name])
                else:
                    for member_index, member in enumerate(members):
                        _copy_host(storage.arrays[spec.name][member_index], member.arrays[spec.name])
        finally:
            _finish_host_copies(storage, array_module)
        # Only our own backings and detached metadata survive preparation.
        result = cls()
        result.cfg = deepcopy(members[0].cfg)
        result.storage = storage
        result.plan = plan
        result.members = len(members)
        result.clock = deepcopy(clock)
        result.scalars = MappingProxyType(deepcopy(dict(members[0].scalars)))
        result._scratch = {slot: storage.arrays[_SCRATCH_PREFIX + slot] for slot in slots}
        result.phb_host_members = tuple(None if member.phb_host is None else
                                       member.phb_host.copy(order="C") for member in members)
        for value in result.phb_host_members:
            if value is not None:
                value.flags.writeable = False
        if array_module is np:
            for spec in specs:
                if spec.ownership == "shared":
                    storage.arrays[spec.name].flags.writeable = False
        result._host_setup_state = array_module is np
        result.physics = None
        result.lateral_boundaries = None
        return result

    def __getattr__(self, name):
        storage = self.__dict__.get("storage")
        if storage is not None and name in storage.arrays and not name.startswith(_SCRATCH_PREFIX):
            return storage.arrays[name]
        scalars = self.__dict__.get("scalars", {})
        if name in scalars:
            return scalars[name]
        raise AttributeError(name)

    def member_view(self, name, member):
        return self.storage.member_view(name, member)

    def scratch(self, shape, slot, dtype=None):
        """Return a planned (member, ...) scratch backing; never allocate lazily."""
        if slot not in self._scratch:
            raise BatchStateUnsupported(f"scratch {slot!r} is not planned; lazy allocation would bypass batch state admission")
        spec = self.storage.specs[_SCRATCH_PREFIX + slot]
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        if shape != spec.shape or np.dtype(np.float32 if dtype is None else dtype) != np.dtype(spec.dtype):
            raise ValueError(f"scratch {slot!r} shape/dtype differs from its planned backing")
        return self._scratch[slot]

    def existing_scratch(self, slot):
        return self._scratch.get(slot)

    def scratch_member_view(self, slot, member):
        return self.storage.member_view(_SCRATCH_PREFIX + slot, member)


def batch_from_members(members, configs, *, array_module, available_bytes,
                       clocks=None, shared_fields=(), scratch_slots=None,
                       reserved_bytes=0, allocation_quantum=512, extra_specs=()):
    """Preserve an existing N=1 state, or pack explicitly host-prepared states.

    N>1 accepts only NumPy setup states and explicit clock snapshots. It never
    calls a member integrator or copies device state back to host. The caller
    can discard CPU preparation after its transfer is complete; no original
    states are retained by the returned container. Forcing object batching is
    separate from this state-allocation seam.
    """
    members, configs = tuple(members), tuple(configs)
    extra_specs = tuple(extra_specs)
    if not members or len(members) != len(configs):
        raise ValueError("each prepared member needs its own configuration")
    for member in members:
        if getattr(member, "physics", None) is not None:
            raise BatchStateUnsupported("attached physics driver has mutable caches without a member dimension; the physics-off state path cannot retain it")
    if len(members) == 1:
        return members[0]
    if clocks is None:
        raise BatchStateUnsupported("N>1 prepared states need explicit clocks; elapsed seconds alone cannot prove a common schedule")
    clocks = tuple(clocks)
    if len(clocks) != len(members):
        raise ValueError("each prepared member needs its own clock snapshot")
    prepared = []
    for member, cfg, clock in zip(members, configs, clocks):
        if not getattr(member, "_host_setup_state", False):
            raise BatchStateUnsupported("N>1 packing requires explicitly prepared host states; resident single-member states would double the unpriced GPU footprint")
        if getattr(member, "lateral_boundaries", None) is not None:
            raise BatchStateUnsupported("attached lateral forcing contains member tables/caches without a batch ABI; supply a separately inventoried batched forcing owner")
        if getattr(member, "_scratch_arena", None) is not None:
            raise BatchStateUnsupported("sequential scratch arena lifetimes cannot alias simultaneously active ensemble members")
        shapes = {spec.name: spec.shape for spec in
                  _state_specs(cfg, shared_fields, extra_specs)}
        attributes = vars(member)
        extra_arrays = {name for name, value in attributes.items()
                        if (isinstance(value, np.ndarray) or hasattr(value, "__cuda_array_interface__"))
                        and name not in shapes and name != "_phb_host"}
        if extra_arrays:
            raise BatchStateUnsupported(f"unclassified member array inventory: {sorted(extra_arrays)}")
        arrays = {name: getattr(member, name, None) for name in shapes}
        scalars = {name: value for name, value in attributes.items()
                   if name not in shapes and name not in _CONTROLS}
        prepared.append(PreparedHostMember(cfg, arrays, scalars, clock,
                                           scratch=getattr(member, "_scratch", {}),
                                           phb_host=getattr(member, "_phb_host", None)))
    return BatchedDomainState.from_prepared(
        prepared, array_module=array_module, available_bytes=available_bytes,
        shared_fields=shared_fields, scratch_slots=scratch_slots,
        reserved_bytes=reserved_bytes, allocation_quantum=allocation_quantum,
        extra_specs=extra_specs)
