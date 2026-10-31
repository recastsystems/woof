"""Admitted member tables and dry lateral-boundary RK components.

Source tables are already prepared. This module packs their declared bytes,
binds the existing time-law/relaxation/finalization kernels and preserves the
common clock's interval ownership. It does not prepare meteorological forcing,
implement a forecast executor or attach a rolling nested forcing producer.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math

import numpy as np

from woof.core.kernels import get_kernel
from woof.ensemble.batch_acoustic import _scalar_state
from woof.ensemble.batch_kernel import KernelSpec, PointerSpec, prepare_batch_kernel_launch
from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported, _copy_host, _finish_host_copies
from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan, BatchStorage
from woof.ingest import lateral_bc as original

_SIDES = ("west", "east", "south", "north")
_ATTRS = {"theta": "thp", "phi": "php", "mu": "mup"}
_KINDS = {"u": 0, "v": 1, "theta": 2, "phi": 3, "mu": 4, "qv": 5, "w": 6}
_THREADS = 256


def _field_shape(cfg, name):
    if name == "mu":
        return 1, cfg.ny, cfg.nx
    return (cfg.nz + int(name in ("phi", "w")),
            cfg.ny + int(name == "v"), cfg.nx + int(name == "u"))


def _side_shape(shape, side, width):
    nz, ny, nx = shape
    return (nz, ny, width) if side in ("west", "east") else (nz, width, nx)


def _table_key(interval, field, side, item):
    return f"table:{interval}:{field}:{side}:{item}"


def _pairs(names, values):
    return tuple(zip(names.split(), values, strict=True))


@dataclass(frozen=True)
class BoundarySelection:
    interval: int
    offset: np.float32


class MemberBoundaryTables:
    """Explicit table/evaluation/weight allocations shared by RK components.

    Tables have a real member dimension. Only identical clock/shape/time-law
    topology is grouped. Per-side nonlinear/linear pipelines cannot be silently
    substituted because their round points differ. State and held-tendency
    allocations remain in the separate admitted BatchedDomainState plan.
    """

    @classmethod
    def allocation_declarations(cls, boundaries, cfg, *, reserved_bytes=0, allocation_quantum=512):
        """Validate and price the exact tables without allocating device memory."""
        boundaries = tuple(boundaries)
        if not boundaries:
            raise ValueError("boundary tables need at least one prepared member")
        first = boundaries[0]
        if not first.intervals:
            raise ValueError("prepared forcing has no interval to select at solve entry")
        fields = tuple(sorted(first.intervals[0].fields))
        dry = {"u", "v", "theta", "phi", "mu"}
        from woof.core.device_inventory import state_array_shapes
        allocated = state_array_shapes(cfg)
        scalar_fields = set(original.COUPLED_SCALAR_STATE_FIELDS) & allocated.keys()
        if not dry <= set(fields) or set(fields) - (dry | {"w"} | scalar_fields):
            raise BatchStateUnsupported(
                "member tables require u/v/theta/phi/mu; optional scalar tables "
                "must name fields allocated by this configuration")
        if cfg.nested and len(first.intervals) != 1:
            raise BatchStateUnsupported(
                "nested forcing is a staged one-interval FORCE snapshot; a rolling "
                "producer needs a generation/reload ownership contract")
        width = int(first.spec_bdy_width)
        if width != cfg.spec_bdy_width or first.spec_zone != cfg.spec_zone or first.relax_zone != cfg.relax_zone:
            raise ValueError("prepared boundary zones differ from the runtime configuration")
        times = tuple((iv.start_seconds, iv.end_seconds) for iv in first.intervals)
        topology = tuple(tuple(getattr(iv.fields[name], side).time_law is not None
                               for name in fields for side in _SIDES)
                         for iv in first.intervals)
        if cfg.nested and any(any(flags) for flags in topology):
            raise BatchStateUnsupported(
                "nested FORCE tables are already interpolated value/tendency pairs; "
                "nonlinear host coefficients have no rolling nested scalar ABI")
        for member, source in enumerate(boundaries):
            if (source.spec_bdy_width, source.spec_zone, source.relax_zone, source.seam_sides) != (
                    width, cfg.spec_zone, cfg.relax_zone, first.seam_sides):
                raise ValueError(f"member {member} boundary zones/seam ownership differs")
            if tuple((iv.start_seconds, iv.end_seconds) for iv in source.intervals) != times:
                raise ValueError(f"member {member} boundary interval clock differs")
            if any(set(iv.fields) != set(fields) for iv in source.intervals):
                raise ValueError(f"member {member} boundary field inventory differs")
            flags = tuple(tuple(getattr(iv.fields[name], side).time_law is not None
                                for name in fields for side in _SIDES)
                          for iv in source.intervals)
            if flags != topology:
                raise BatchStateUnsupported(
                    "members use different boundary time-law pipelines; group by "
                    "time-law topology to preserve each scalar reconstruction's words")
        specs = []
        for interval, flags in enumerate(topology):
            for name in fields:
                shape = _field_shape(cfg, name)
                original._validate_frame_domain(shape[-2], shape[-1], width, f"{name} member boundary")
                for side in _SIDES:
                    expected = _side_shape(shape, side, width)
                    source_side = getattr(first.intervals[interval].fields[name], side)
                    for item, _value in source_side.array_items():
                        for member, source in enumerate(boundaries):
                            value = dict(getattr(source.intervals[interval].fields[name], side).array_items())[item]
                            if value.shape != expected:
                                raise ValueError(f"{name}/{side}/{item} member {member} has wrong boundary shape")
                        specs.append(BatchArraySpec(_table_key(interval, name, side, item), expected, "member"))
            if any(flags):
                for name in fields:
                    for side in _SIDES:
                        shape = _side_shape(_field_shape(cfg, name), side, width)
                        for item in ("value", "tendency"):
                            specs.append(BatchArraySpec(f"evaluated:{interval}:{name}:{side}:{item}", shape, "member"))
        specs += [BatchArraySpec("weights:fcx", (width,), "shared"),
                  BatchArraySpec("weights:gcx", (width,), "shared")]
        plan = BatchMemoryPlan(tuple(specs), reserved_bytes, allocation_quantum)
        return boundaries, fields, width, times, topology, plan

    @classmethod
    def memory_plan(cls, boundaries, cfg, *, reserved_bytes=0, allocation_quantum=512):
        return cls.allocation_declarations(boundaries, cfg, reserved_bytes=reserved_bytes,
                                          allocation_quantum=allocation_quantum)[-1]

    @classmethod
    def from_prepared(cls, boundaries, cfg, *, array_module, available_bytes,
                      reserved_bytes=0, allocation_quantum=512):
        boundaries, fields, width, times, topology, plan = cls.allocation_declarations(
            boundaries, cfg, reserved_bytes=reserved_bytes, allocation_quantum=allocation_quantum)
        first = boundaries[0]
        storage = BatchStorage(plan, len(boundaries), array_module=array_module, available_bytes=available_bytes)
        for interval in range(len(times)):
            for name in fields:
                for side in _SIDES:
                    for member, source in enumerate(boundaries):
                        for item, value in getattr(source.intervals[interval].fields[name], side).array_items():
                            _copy_host(storage.arrays[_table_key(interval, name, side, item)][member],
                                       np.ascontiguousarray(value, dtype=np.float32))
        dt = np.float32(cfg.dt) if cfg.nested else original.lateral_boundary_clock_dt(cfg)
        spec_exp = 0.0 if cfg.nested else cfg.spec_exp
        weights = original._weights(width, cfg.spec_zone, cfg.relax_zone, dt, spec_exp,
                                    wrf_real=bool(cfg.nested), timescale_s=original.relax_timescale_seconds(cfg))
        for name, value in zip(("weights:fcx", "weights:gcx"), weights):
            _copy_host(storage.arrays[name], value)
        if not cfg.nested and set(fields) & original._HYDROMETEOR_RING_FIELDS:
            # Each member's ring serves its hydrometeors held as a single
            # run's are (lateral_bc._held_host_sides), the same bytes.
            radius = original.ring_cell_radius(cfg)
            for interval in range(len(times)):
                for member, source in enumerate(boundaries):
                    held = original._held_host_sides(source.intervals[interval], radius)
                    for (name, side, item), table in held.items():
                        _copy_host(storage.arrays[_table_key(interval, name, side, item)][member], table)
                    _finish_host_copies(storage, array_module)
        _finish_host_copies(storage, array_module)
        result = cls()
        result.storage, result.plan, result.members = storage, plan, len(boundaries)
        result.boundaries, result.times, result.fields = boundaries, times, fields
        result.topology, result.cfg = topology, deepcopy(cfg)
        result.width, result.dt, result.spec_exp = width, dt, spec_exp
        result.seam_sides = tuple(first.seam_sides)
        result._evaluated_key = None
        result._views = {}
        result._view_owners = {}
        result._view_cache = {}
        return result

    def select(self, clock):
        """Select at solve entry, retaining the clock's post-increment DTBC."""
        if not isinstance(clock.dtbc_launch_fp32, np.float32) or not isinstance(clock.dt_fp32, np.float32):
            raise TypeError("boundary clock must retain float32 DT and launch DTBC")
        if clock.dt_fp32.tobytes() != np.float32(self.cfg.dt).tobytes():
            raise ValueError("boundary clock step differs from the fixed admitted model step")
        if not math.isfinite(float(clock.dtbc_launch_fp32)):
            raise ValueError("boundary launch DTBC must be finite")
        interval = 0 if self.cfg.nested else original.interval_index(
            self.boundaries[0].intervals, clock.elapsed_seconds)
        return BoundarySelection(interval, clock.dtbc_launch_fp32)

    def value(self, selection, name, side, item, *, evaluated=False):
        key = (f"evaluated:{selection.interval}:{name}:{side}:{item}" if evaluated
               else _table_key(selection.interval, name, side, item))
        return self.storage.arrays[key]

    def evaluate(self, selection):
        """Run the original per-side reconstruction only when its interval needs it."""
        if not any(self.topology[selection.interval]):
            return selection, False
        key = (selection.interval, float(selection.offset))
        if self._evaluated_key != key:
            for name in self.fields:
                for side in _SIDES:
                    host_side = getattr(self.boundaries[0].intervals[selection.interval].fields[name], side)
                    items = ["value", "tendency"]
                    entry = "evaluate_linear_boundary"
                    if host_side.time_law is not None:
                        entry = "evaluate_rational_boundary"
                        items += ["quadratic", "denominator_rate"]
                    arrays = tuple(self.value(selection, name, side, item) for item in items)
                    outputs = tuple(self.value(selection, name, side, item, evaluated=True)
                                    for item in ("value", "tendency"))
                    names = ["value", "tendency"] + (["quadratic", "rate"] if len(items) == 4 else [])
                    names += ["out_value", "out_tendency"]
                    pointers = _pairs(" ".join(names), arrays + outputs)
                    count = math.prod(outputs[0].shape[1:])
                    self.bind("lbc_time", entry, arrays + outputs + (selection.offset, np.int32(count)),
                              pointers, ((count + 255) // 256,))()
            self._evaluated_key = key
        return BoundarySelection(selection.interval, np.float32(0.0)), True

    def bind(self, module, entry, args, pointers, grid, *, state=None):
        inventory = {id(array): (spec.ownership, array.strides[0] if spec.ownership == "member" else 0)
                     for name, spec in self.storage.specs.items() for array in (self.storage.arrays[name],)}
        if state is not None:
            inventory.update({id(array): (spec.ownership, state.storage.pointer_stride_bytes(name))
                              for name, spec in state.storage.specs.items()
                              for array in (state.storage.arrays[name],)})
        inventory.update(self._views)
        specs, strides = [], {}
        for name, value in pointers:
            if id(value) not in inventory:
                raise BatchStateUnsupported(f"{name} is outside the admitted state/boundary inventory")
            role, stride = inventory[id(value)]
            specs.append(PointerSpec(name, role))
            strides[name] = int(stride)
        spec = KernelSpec(module, entry, tuple(specs))
        if self.members == 1:
            scalar_args = tuple(value[0] if id(value) in inventory and inventory[id(value)][0] == "member"
                                else value for value in args)
            kernel = get_kernel(module, entry)
            return lambda: kernel(grid, (_THREADS,), scalar_args)
        return prepare_batch_kernel_launch(spec, self.members, grid, (_THREADS,), args,
                                           pointer_strides=strides)

    def register_member_view(self, value, parent):
        """Register a no-copy logical slice of one admitted member backing."""
        if len(value.shape) < 2 or len(parent.shape) < 2:
            raise ValueError("nested held view needs complete member backing shapes")
        start = int(value.__cuda_array_interface__["data"][0])
        parent_start = int(parent.__cuda_array_interface__["data"][0])
        stride = int(parent.strides[0])
        slab_bytes = math.prod(value.shape[1:]) * 4
        if (value.shape[0] != self.members or parent.shape[0] != self.members
                or value.dtype != np.dtype("float32") or parent.dtype != np.dtype("float32")
                or stride <= 0 or stride % 4 or slab_bytes <= 0
                or (self.members > 1 and value.strides[0] != stride)
                or start % 4 or start < parent_start
                or start + (self.members - 1) * stride + slab_bytes > parent_start + parent.nbytes):
            raise ValueError("nested held view is outside its admitted member backing")
        expected = 4
        for extent, actual in zip(reversed(value.shape[1:]), reversed(value.strides[1:])):
            if actual != expected:
                raise ValueError("nested held view requires contiguous inner axes")
            expected *= int(extent)
        # A singleton prefix reshape can replace stride[0] with the logical
        # prefix size. It never jumps to another member, so the difference is
        # immaterial. Its pointer/span must still belong to the actual parent
        # allocation; it cannot make an unrelated allocation look admitted.
        memories = []
        for array in (value, parent):
            data = getattr(array, "data", None)
            memory = getattr(data, "mem", None)
            numbers = (getattr(memory, "ptr", None), getattr(memory, "size", None), getattr(data, "ptr", None))
            if (memory is None or any(isinstance(number, (bool, np.bool_))
                    or not isinstance(number, (int, np.integer)) for number in numbers)
                    or (type(memory).__name__ == "UnownedMemory" and getattr(memory, "owner", None) is None)):
                raise ValueError("nested held view requires owned allocation pointer/size bounds")
            base, capacity, address = (int(number) for number in numbers)
            if (base < 0 or capacity <= 0 or base + capacity > 2**64
                    or address != int(array.__cuda_array_interface__["data"][0])
                    or address < base or address + array.nbytes > base + capacity):
                raise ValueError("nested held view exceeds its owned allocation bounds")
            memories.append((base, capacity))
        allocation_base, allocation_bytes = memories[0]
        if (memories[0] != memories[1]
                or start + (self.members - 1) * stride + slab_bytes > allocation_base + allocation_bytes):
            raise ValueError("nested held view does not share its admitted parent allocation")
        self._views[id(value)] = ("member", stride)
        self._view_owners[id(value)] = (value, parent)

    def workspace_view(self, parent, shape):
        key = (id(parent), tuple(shape))
        if key not in self._view_cache:
            value = parent[:, :math.prod(shape)].reshape((self.members,) + tuple(shape))
            self.register_member_view(value, parent)
            self._view_cache[key] = value
        return self._view_cache[key]

    def scalar_field(self, selection, name, evaluated):
        return original._DeviceFieldBoundary(**{
            side: original._DeviceSideBoundary(
                self.value(selection, name, side, "value", evaluated=evaluated)[0],
                self.value(selection, name, side, "tendency", evaluated=evaluated)[0])
            for side in _SIDES})

    def sides(self, selection, name, evaluated):
        return tuple(self.value(selection, name, side, item, evaluated=evaluated)
                     for side in _SIDES for item in ("value", "tendency"))


def _state_check(state, cfg, tables):
    if not isinstance(state, BatchedDomainState) or state.members != tables.members:
        raise TypeError("dry boundary component needs matching admitted state and member tables")
    if state.storage.specs["p"].shape != (cfg.nz, cfg.ny, cfg.nx):
        raise ValueError("boundary grid differs from the admitted prognostic grid")
    if (cfg.spec_bdy_width, cfg.spec_zone, cfg.relax_zone, bool(cfg.nested)) != (
            tables.width, tables.cfg.spec_zone, tables.cfg.relax_zone, bool(tables.cfg.nested)):
        raise ValueError("boundary runtime zones differ from their table plan")


def _field(state, name):
    return getattr(state, _ATTRS.get(name, name))


class _BoundaryScalarState:
    def __init__(self, state, tables):
        self.view = _scalar_state(state)
        self.lateral_boundaries = tables

    def __getattr__(self, name):
        return getattr(self.view, name)

    def scratch(self, shape, slot, dtype=None):
        return self.view.scratch(shape, slot, dtype)


def _relax(state, cfg, tables, selection, evaluated, name, tendency, held, *,
           apply_relax, clear_specified=False, add_held=False, source=None, source_mu=None):
    shape = _field_shape(cfg, name)
    nz, ny, nx = shape
    active_width = max(cfg.spec_zone, cfg.relax_zone)
    relax_sides = original._relax_side_mask(tables)
    original._validate_relaxation_window(ny, nx, active_width, relax_sides, f"{name} member relaxation")
    active_width = original._frame_rings(ny, nx, active_width)
    frame_count = original._perimeter_count(ny, nx, active_width)
    value = _field(state, name) if source is None else source
    divide_msf = bool(state.has_msf and name in ("theta", "phi") and apply_relax)
    if state.members == 1:
        scalar = _BoundaryScalarState(state, tables)
        boundary = tables.scalar_field(selection, name, evaluated)
        weights = (tables.storage.arrays["weights:fcx"], tables.storage.arrays["weights:gcx"])
        scalar_tendency = tendency[0].reshape(shape)
        scalar_held = held[0].reshape(shape)

        def launch_original():
            original._launch_state_relaxation(
                scalar, name, scalar_tendency, boundary,
                dtbc=selection.offset, dt=tables.dt, spec_zone=cfg.spec_zone,
                relax_zone=cfg.relax_zone, spec_exp=tables.spec_exp,
                apply_relax=apply_relax, weights=weights, clear_specified=clear_specified,
                add_held=scalar_held if add_held else None, divide_msf=divide_msf,
                source_field=None if source is None else source[0],
                source_mup=None if source_mu is None else source_mu[0],
                timescale_s=original.relax_timescale_seconds(cfg))

        return launch_original
    u = value if name == "u" else state.u
    v = value if name == "v" else state.v
    w = value if name == "w" else state.w
    thp = value if name == "theta" else state.thp
    php = value if name == "phi" else state.php
    mass = state.mup if source_mu is None else source_mu
    base = (tendency, held, state.mub2d, mass, u, v, w, thp, state.thb, php, value,
            state.c1h, state.c2h, state.c1f, state.c2f, state.msft, state.msfu, state.msfv)
    sides = tables.sides(selection, name, evaluated)
    weights = (tables.storage.arrays["weights:fcx"], tables.storage.arrays["weights:gcx"])
    args = base + sides + weights + (
        selection.offset, np.int32(tables.width), np.int32(cfg.spec_zone), np.int32(cfg.relax_zone),
        np.int32(apply_relax), np.int32(clear_specified), np.int32(add_held), np.int32(divide_msf),
        np.int32(state.has_msf), np.int32(len(state.storage.specs["thb"].shape) == 3),
        np.int32(_KINDS.get(name, 7)), np.int32(nz), np.int32(ny), np.int32(nx),
        np.int32(frame_count), np.int32(relax_sides))
    names = "field_tend held mub2d mup u v w thp thb php scalar c1h c2h c1f c2f msft msfu msfv west west_t east east_t south south_t north north_t fcx gcx"
    raw = tables.bind("lbc_state", "state_specified_relaxation", args,
                      _pairs(names, base + sides + weights), ((nz * frame_count + 255) // 256,), state=state)

    def launch():
        raw()
        interior = (slice(None), slice(None), slice(active_width, ny - active_width), slice(active_width, nx - active_width))
        target = tendency.reshape((state.members,) + shape)
        carried = held.reshape((state.members,) + shape)
        if divide_msf:
            factor = state.msft[:, None] if state.storage.specs["msft"].ownership == "member" else state.msft[None, None]
            target[interior] /= factor[..., active_width:ny - active_width, active_width:nx - active_width]
        if add_held:
            target[interior] += carried[interior]

    return launch


def prepare_dry_lateral_tendencies(state, cfg, tables, clock, *, rk_stage):
    """Bind the real stage-1 held relaxation and all-stage specified tendency work."""
    if not (cfg.specified or cfg.nested):
        return lambda: None
    _state_check(state, cfg, tables)
    if rk_stage not in (0, 1, 2):
        raise ValueError("rk_stage must be 0, 1, or 2")
    selection, evaluated = tables.evaluate(tables.select(clock))
    rows = [("u", state.ru_t), ("v", state.rv_t), ("theta", state.rth_t), ("phi", state.rph_t)]
    if cfg.nested or original.specified_relaxes_w(cfg):
        if "w" not in tables.fields:
            raise RuntimeError("nested/relax_w forcing needs a staged w boundary table")
        rows.append(("w", state.rw_t))
    calls = []
    nested_backing = None
    if cfg.nested:
        capacity = max(math.prod(_field_shape(cfg, name)) for name, _ in rows)
        nested_backing = state.scratch((capacity,), "lbc_nested_relax")
    for name, tendency in rows:
        shape = _field_shape(cfg, name)
        # Each persistent held carrier is declared by the existing scratch
        # inventory; it is never allocated as a hidden full-field temporary.
        if cfg.nested:
            held = tables.workspace_view(nested_backing, shape)
        else:
            held = state.scratch(shape, "lbc_relax_" + name)
        source = getattr(state, {"u": "u0", "v": "v0", "w": "w0", "theta": "thp0", "phi": "php0"}[name]) if cfg.nested else None
        if rk_stage == 0 or cfg.nested:
            recompute = _relax(state, cfg, tables, selection, evaluated, name, held, held,
                               apply_relax=True, clear_specified=True, source=source,
                               source_mu=state.mup0 if cfg.nested else None)
            calls.append((held, recompute))
        apply = _relax(state, cfg, tables, selection, evaluated, name, tendency, held,
                       apply_relax=False, add_held=True)
        calls.append((None, apply))
    mu = _relax(state, cfg, tables, selection, evaluated, "mu", state.rmu_t, state.rmu_t,
                apply_relax=rk_stage == 0)

    def launch():
        for clear, operation in calls:
            if clear is not None:
                clear.fill(np.float32(0.0))
            operation()
        mu()

    return launch


def prepare_dry_boundary_values(state, cfg, tables, clock):
    """Install MU, then original whole-grid couple/install/uncouple finalizers."""
    if not (cfg.specified or cfg.nested):
        return lambda: None
    _state_check(state, cfg, tables)
    selection, evaluated = tables.evaluate(tables.select(clock))
    from woof.boundary_fields import SCALAR_ARRAY_BOUNDARY_FIELDS
    scalar_names = [name for name in tables.fields
                    if name in original.COUPLED_SCALAR_STATE_FIELDS
                    and (cfg.nested or name not in SCALAR_ARRAY_BOUNDARY_FIELDS)]
    if state.members == 1:
        scalar = _BoundaryScalarState(state, tables)
        names = ["u", "v"] + (["w"] if cfg.nested or original.specified_relaxes_w(cfg) else []) + ["theta", "phi"] + scalar_names

        def launch_original():
            old = original._launch_mu_boundary_values(
                scalar, tables.scalar_field(selection, "mu", evaluated), selection.offset, cfg.spec_zone)
            for name in names:
                original._launch_finalize_field(
                    scalar, name, tables.scalar_field(selection, name, evaluated),
                    old, selection.offset, cfg.spec_zone)

        return launch_original
    frame_count = original._perimeter_count(cfg.ny, cfg.nx, cfg.spec_zone)
    old = state.scratch((frame_count,), f"lbc_old_mup_frame_{cfg.spec_zone}")
    sides = tables.sides(selection, "mu", evaluated)
    arrays = (state.mup, old) + sides
    install = tables.bind("lbc_state", "install_mu_boundary", arrays + (
        selection.offset, np.int32(tables.width), np.int32(cfg.spec_zone),
        np.int32(cfg.ny), np.int32(cfg.nx), np.int32(frame_count)),
        _pairs("mup old_mup_frame west west_t east east_t south south_t north north_t", arrays),
        ((frame_count + 255) // 256,), state=state)
    names = ["u", "v"] + (["w"] if cfg.nested or original.specified_relaxes_w(cfg) else []) + ["theta", "phi"] + scalar_names
    finalizers = []
    for name in names:
        target = _field(state, name)
        shape = _field_shape(cfg, name)
        arrays = (target, old, state.mub2d, state.mup, state.thb, target,
                  state.c1h, state.c2h, state.c1f, state.c2f, state.msft, state.msfu, state.msfv)
        sides = tables.sides(selection, name, evaluated)
        args = arrays + sides + (selection.offset, np.int32(tables.width), np.int32(cfg.spec_zone),
            np.int32(state.has_msf), np.int32(len(state.storage.specs["thb"].shape) == 3),
            np.int32(_KINDS.get(name, 7)), *(np.int32(n) for n in shape), np.int32(cfg.ny), np.int32(cfg.nx))
        finalizers.append(tables.bind("lbc_state", "finalize_state_field", args,
            _pairs("target old_mup_frame mub2d mup thb scalar_unused c1h c2h c1f c2f msft msfu msfv west west_t east east_t south south_t north north_t", arrays + sides),
            ((math.prod(shape) + 255) // 256,), state=state))

    def launch():
        install()
        for finalize in finalizers:
            finalize()

    return launch


def prepare_specified_w_zero_gradient(state, cfg, tables):
    """Apply the original Y-corner-owned w frame copy when tables do not own w."""
    if not cfg.specified or cfg.nested or original.specified_relaxes_w(cfg):
        return lambda: None
    _state_check(state, cfg, tables)
    if state.members == 1:
        scalar = _BoundaryScalarState(state, tables)
        return lambda: original.apply_specified_w_zero_gradient(scalar, cfg)
    frame_count = original._perimeter_count(cfg.ny, cfg.nx, cfg.spec_zone)
    spec = KernelSpec("lbc_flow", "zero_gradient_w", (PointerSpec("field", "member"),))
    args = (state.w, np.int32(cfg.spec_zone), np.int32(cfg.nz + 1), np.int32(cfg.ny),
            np.int32(cfg.nx), np.int32(frame_count))
    return prepare_batch_kernel_launch(spec, state.members,
        (((cfg.nz + 1) * frame_count + 255) // 256,), (_THREADS,), args,
        pointer_strides={"field": state.storage.pointer_stride_bytes("w")})
