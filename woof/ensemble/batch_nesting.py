"""Member-indexed resident parent FORCE and feedback operators.

The stock coupling, SINT boundary, restriction and smoother kernels retain
their arithmetic bodies. Their state, scratch and rolling-table pointers
advance by the admitted member stride. This component owns an edge, not a
forecast scheduler or physics driver. Its callers own domain stepping and
must preserve parent STEP, FORCE, child STEP and feedback order.
"""
from __future__ import annotations

from hashlib import sha256
from dataclasses import fields as dataclass_fields, is_dataclass
from fractions import Fraction
from math import prod
from operator import index
from pathlib import Path

import numpy as np

from woof.core.nest_interp import (
    bdy_width, feedback_parent_bounds, smoother_parent_window, SMDSM_XNU,
)
from woof.ensemble.batch_kernel import (
    KernelSpec, PointerSpec, prepare_batch_kernel_launch, prepare_batch_source_launch,
)
from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported, _exact_key
from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan, BatchStorage

CONTRACT = "gpuwm-member-nest-edge-v1"
_SIDES = ("west", "east", "south", "north")
_GEOMETRY = ("ci", "ip", "cj", "jp", "xig", "xjg")
_ATTR = {"t": "thp", "ph": "php", "mu": "mup"}
_APPLICATION = {"t": "theta", "ph": "phi"}
_STAGGER = {"u": "x", "v": "y"}
_SIGNED = frozenset({"u", "v", "w", "t", "ph", "mu"})
_THREADS = 256
_NEST_OPTIONS = ("-std=c++17", "-fmad=false")


def nesting_source():
    """The stock nest loader's exact translation unit and compile policy."""
    from woof.core.kernels import _preamble
    from woof.core import nest_interp
    return _preamble() + (Path(nest_interp.__file__).parent / "kernels/nest.cu").read_text()


def _member_array(value, members, shape, label):
    if (value.dtype != np.dtype("float32") or tuple(value.shape) != (members, *shape)
            or not hasattr(value, "__cuda_array_interface__")):
        raise ValueError(f"{label} needs complete member float32 CUDA slabs with shape {(members, *shape)}")
    # The audited launch accepts either contiguous storage or an allocation-
    # bounded prefix with its actual leading stride. It checks those bounds.
    return value


def _span(array):
    start = int(array.data.ptr)
    size = int(array.nbytes) if array.flags.c_contiguous else (
        (int(array.shape[0]) - 1) * int(array.strides[0])
        + prod(int(n) for n in array.shape[1:]) * int(array.dtype.itemsize))
    return start, start + size


def _require_disjoint(arrays, label):
    ranges = sorted((*_span(array), name) for name, array in arrays)
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"{label} allocations {previous[2]} and {current[2]} overlap")


def _bind(module, entry, members, count, args, pointers, *, source=None):
    if count <= 0:
        def empty():
            return None
        empty.binding_receipt = {"entry": entry, "members": members, "zero_trip": True}
        return empty
    roles = tuple(PointerSpec(name, role, value.dtype) for name, value, role in pointers)
    strides = {name: 0 if role == "shared" else int(value.strides[0])
               for name, value, role in pointers}
    spec = KernelSpec(module, entry, roles, _NEST_OPTIONS if source is not None else ("-std=c++17",))
    grid = ((int(count) + _THREADS - 1) // _THREADS,)
    if source is not None:
        return prepare_batch_source_launch(source, spec, members, grid, (_THREADS,), args,
                                           pointer_strides=strides)
    return prepare_batch_kernel_launch(spec, members, grid, (_THREADS,), args,
                                       pointer_strides=strides)


def prepare_member_bdy_interp1(parent, child, registration, *, members,
                               parent_dt_fp32, parent_interval_ticks,
                               out, geometry, spec_zone=1, relax_zone=4,
                               spec_bdy_width=5):
    """Build every member's four-side value/tendency tables in one launch."""
    reg = registration
    if reg.wrapper != "bdy":
        raise ValueError("member FORCE needs the original bdy-wrapper registration")
    if (isinstance(parent_interval_ticks, bool) or int(parent_interval_ticks) <= 0
            or not isinstance(parent_dt_fp32, np.float32)
            or not np.isfinite(parent_dt_fp32) or parent_dt_fp32 <= 0):
        raise ValueError("member FORCE requires a positive original parent interval and FP32 parent step")
    if parent.ndim != 4:
        raise ValueError("member FORCE parent needs (members, levels, y, x) slabs")
    nz = int(parent.shape[1])
    _member_array(parent, members, (nz, reg.nyp, reg.nxp), "FORCE parent")
    _member_array(child, members, (nz, reg.nyc, reg.nxc), "FORCE child")
    width = bdy_width(spec_zone, relax_zone, spec_bdy_width)
    expected = {side: (nz, reg.nyc, width) if side in ("west", "east")
                else (nz, width, reg.nxc) for side in _SIDES}
    if set(out) != set(_SIDES) or set(geometry) != set(_GEOMETRY):
        raise ValueError("member FORCE needs the complete original side and geometry inventories")
    for side in _SIDES:
        for item, value in zip(("value", "tendency"), out[side], strict=True):
            _member_array(value, members, expected[side], f"{side} {item}")
    for name in _GEOMETRY:
        value, host = geometry[name], getattr(reg, name)
        if (tuple(value.shape) != tuple(host.shape) or value.dtype != host.dtype
                or not value.flags.c_contiguous):
            raise ValueError(f"member FORCE geometry {name} differs from its registered shared table")
    _require_disjoint((("parent", parent), ("child", child),
        *((f"{side}:{item}", value) for side in _SIDES
          for item, value in zip(("value", "tendency"), out[side], strict=True))), "member FORCE")
    table_names = ("west_val", "west_tend", "east_val", "east_tend",
                   "south_val", "south_tend", "north_val", "north_tend")
    tables = tuple(value for side in _SIDES for value in out[side])
    geom_names = ("ci_map", "ip_map", "cj_map", "jp_map", "xig", "xjg")
    geom = tuple(geometry[name] for name in _GEOMETRY)
    args = (parent, child, *tables, *geom, parent_dt_fp32, np.int32(width),
            *(np.int32(n) for n in (nz, reg.nyc, reg.nxc, reg.nyp, reg.nxp)))
    pointers = (("cfld", parent, "member"), ("nfld", child, "member"))
    pointers += tuple((name, value, "member") for name, value in zip(table_names, tables, strict=True))
    pointers += tuple((name, value, "shared") for name, value in zip(geom_names, geom, strict=True))
    return _bind("nest", "nest_bdy_interp1_all_sides", members,
                 2 * nz * width * (reg.nyc + reg.nxc), args, pointers, source=nesting_source())


def prepare_member_copy_fcn(parent, child, registration, *, members, spec_zone=1):
    """Restrict each raw child into only its own parent overlap."""
    reg = registration
    if parent.ndim != 4:
        raise ValueError("member feedback parent needs complete member level slabs")
    nz = int(parent.shape[1])
    _member_array(parent, members, (nz, reg.nyp, reg.nxp), "feedback parent")
    _member_array(child, members, (nz, reg.nyc, reg.nxc), "feedback child")
    _require_disjoint((("parent", parent), ("child", child)), "member feedback")
    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg, spec_zone=spec_zone)
    ni, nj = max(0, ihi - ilo + 1), max(0, jhi - jlo + 1)
    scalars = (reg.i_parent_start, reg.j_parent_start, reg.nri, reg.nrj, spec_zone,
               reg.xstag, reg.ystag, nz, reg.nyp, reg.nxp, reg.nyc, reg.nxc,
               0, 0, 0, 0, ilo, jlo, ni, nj)
    args = (parent, child, *(np.int32(n) for n in scalars))
    return _bind("nest", "nest_copy_fcn", members, nz * ni * nj, args,
                 (("cfld", parent, "member"), ("nfld", child, "member")), source=nesting_source())


def prepare_member_smoother(parent, registration, *, members, smooth_option, scratch):
    """Keep the stock J/I pass order within every member's own scratch."""
    reg = registration
    if smooth_option not in (0, 1, 2):
        raise ValueError("member smoother needs the original option 0, 1 or 2")
    nz = int(parent.shape[1])
    _member_array(parent, members, (nz, reg.nyp, reg.nxp), "smoother parent")
    _member_array(scratch, members, (nz, reg.nyp, reg.nxp), "smoother scratch")
    _require_disjoint((("parent", parent), ("scratch", scratch)), "member smoother")
    i0, j0, ni, nj = smoother_parent_window(reg)
    launches = []
    if smooth_option and ni > 0 and nj > 0:
        dims = tuple(np.int32(n) for n in (i0, j0, ni, nj, nz, reg.nyp, reg.nxp))
        mode = np.int32(0 if smooth_option == 1 else 1)
        for xnu in SMDSM_XNU if smooth_option == 2 else (np.float32(0),):
            args = (parent, scratch, mode, xnu, *dims)
            pointers = (("cfld", parent, "member"), ("scr", scratch, "member"))
            launches.append(_bind("nest", "nest_smooth_j", members, nz * (nj + 2) * (ni + 2),
                                  args, pointers, source=nesting_source()))
            launches.append(_bind("nest", "nest_smooth_i", members, nz * nj * ni,
                                  args, pointers, source=nesting_source()))
    def launch():
        for operation in launches:
            operation()
    launch.binding_receipt = {"members": members, "smooth_option": smooth_option,
        "launches": [dict(operation.binding_receipt) for operation in launches]}
    return launch


def _shape(state, kind):
    shape = state.storage.specs[_ATTR.get(kind, kind)].shape
    return (1, *shape) if kind == "mu" else tuple(shape)


def coupled_field_pointer_names(kind):
    """The original coupled-unit kernel's complete pointer dependency row."""
    return (_ATTR.get(kind, kind), "mub2d", "mup", "thb", "c1h", "c2h", "c1f", "c2f",
            "msft", "msfu", "msfv")


def member_edge_state_fields(cfg, fields, *, parent_diagnostics=False):
    """Only audited kernel reads and outputs used by a resident nest edge."""
    from woof.ensemble.batch_diagnostics import diagnostic_pointer_fields
    names = {name for kind in fields for name in coupled_field_pointer_names(kind)}
    if parent_diagnostics:
        names.update(name for _, name in diagnostic_pointer_fields(moist=bool(cfg.moist)))
    return frozenset(names)


def prepared_tree_edge_state_fields(domains):
    """Union each grid's actual child and parent edge dependencies."""
    from woof.core.preflight import nest_field_kinds
    by_id = {int(domain.grid_id): domain for domain in domains}
    result = {gid: set() for gid in by_id}
    for child in domains:
        if not child.parent_id:
            continue
        gid, parent_id = int(child.grid_id), int(child.parent_id)
        parent = by_id[parent_id]
        fields = tuple(nest_field_kinds(child.run))
        result[gid].update(member_edge_state_fields(child.run, fields))
        # The existing edge binder prepares parent EOS diagnostics even for
        # feedback off. Keep every pointer that binder validates and retains.
        result[parent_id].update(member_edge_state_fields(parent.run, fields, parent_diagnostics=True))
    return {gid: frozenset(names) for gid, names in result.items()}


def prepare_member_couple_nest_field(state, kind, *, out, window=None, frame_width=None):
    """The stock coupled-unit producer over all admitted member states."""
    if not isinstance(state, BatchedDomainState):
        raise TypeError("member nest coupling requires inventoried BatchedDomainState")
    from woof.ingest import lateral_bc
    if kind not in ({"u", "v", "w", "t", "ph", "mu"} | lateral_bc.COUPLED_SCALAR_STATE_FIELDS):
        raise ValueError(f"unsupported member nest coupling field {kind!r}")
    shape, members = _shape(state, kind), state.members
    _member_array(out, members, shape, "coupled member field")
    target_name = _ATTR.get(kind, kind)
    names = coupled_field_pointer_names(kind)
    parameters = ("target", "mub2d", "mup", "thb", "c1h", "c2h", "c1f", "c2f", "msft", "msfu", "msfv")
    arrays = tuple(state.storage.arrays[name] for name in names)
    out_start, out_end = _span(out)
    for name, source in zip(names, arrays, strict=True):
        source_start, source_end = _span(source)
        if out_start < source_end and source_start < out_end:
            raise ValueError(f"member coupled output overlaps its read-only source {name}")
    pointers = tuple((parameter, value, state.storage.specs[name].ownership)
                     for parameter, name, value in zip(parameters, names, arrays, strict=True))
    pointers += (("out", out, "member"),)
    nz, ny, nx = shape
    mny, mnx = state.storage.specs["mup"].shape
    field_kind = {"u": 0, "v": 1, "t": 2, "ph": 3, "mu": 4, "qv": 5, "w": 6}.get(kind, 7)
    args = (*arrays, out, *(np.int32(n) for n in (state.has_msf,
            len(state.storage.specs["thb"].shape) == 3, field_kind, nz, ny, nx, mny, mnx)))
    entry, count = "couple_nest_field", prod(shape)
    if window is not None and frame_width is not None:
        raise ValueError("member coupling cannot enumerate a window and frame together")
    if window is not None:
        from woof.core.streaming import window_slices
        _, y, x = window_slices(shape, window)
        entry, count = "couple_nest_field_window", nz * max(0, y.stop - y.start) * max(0, x.stop - x.start)
        args += tuple(np.int32(n) for n in (y.start, x.start, y.stop - y.start, x.stop - x.start))
    elif frame_width is not None:
        width = lateral_bc._frame_rings(ny, nx, int(frame_width))
        points = lateral_bc._perimeter_count(ny, nx, width)
        entry, count = "couple_nest_field_frame", nz * points
        args += (np.int32(width), np.int32(points))
    operation = _bind("lbc_state", entry, members, count, args, pointers)
    from woof.wrf_exact import ENABLED
    if ENABLED and frame_width is not None and kind in ("u", "v"):
        from woof.core.nest_round_trip import couple_halo_faces
        def launch():
            operation()
            couple_halo_faces(state, kind, out)
        launch.binding_receipt = operation.binding_receipt
        return launch
    return operation


def _base_theta(state):
    spec = state.storage.specs["thb"]
    rows = 1 if spec.ownership == "shared" else state.members
    if len(spec.shape) == 1:
        return state.thb.reshape(rows, spec.shape[0], 1, 1)
    return state.thb.reshape(rows, *spec.shape)


_ROUND_TRIP_SOURCE = r'''
extern "C" __global__ void member_nest_round_trip(
    float *field, const float *mub, const float *mu, const float *halo,
    const float *c1, const float *c2, const float *msf,
    int kind, int mapped, int nz, int ny, int nx, int mny, int mnx) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= nz * ny * nx) return;
    int i = n % nx, j = (n / nx) % ny, k = n / (nx * ny);
    int x = min(i, mnx - 1), y = min(j, mny - 1);
    int q = y * mnx + x;
    float a = c1[k] * mub[q] + c2[k];
    float b = c1[k] * mu[q];
    float mass = a + b;
    float factor, inverse;
    if (kind == 3 || kind == 4) {
        int axis = kind == 3 ? i : j;
        int end = kind == 3 ? mnx : mny;
        if (axis < end) {
            int p = kind == 3 ? y * mnx + max(i - 1, 0)
                             : max(j - 1, 0) * mnx + x;
            float ap = c1[k] * mub[p] + c2[k];
            float bp = c1[k] * (axis ? mu[p] : halo[p]);
            float sum = ((a + ap) + b) + bp;
            float metric = mapped ? msf[j * nx + i] : 1.0f;
            factor = (0.5f * sum) / metric;
            inverse = (2.0f * metric) / sum;
        } else {
            float metric = mapped ? msf[j * nx + i] : 1.0f;
            factor = mass / metric;
            inverse = metric / mass;
        }
    } else if (kind == 2 && mapped) {
        float metric = msf[q];
        factor = mass / metric;
        inverse = metric / mass;
    } else {
        factor = mass;
        inverse = 1.0f / mass;
    }
    // Separate FP32 operations match the scalar CuPy round-trip launches.
    float coupled = field[n] * factor;
    field[n] = coupled * inverse;
}
'''


def prepare_member_round_trip(state, fields):
    """Bind WRF's in-place FORCE round trip without workspace or member loops."""
    from woof.state_serialization_contract import carried_wrf_mu_halo
    if not (state.cfg.specified or state.cfg.nested):
        raise BatchStateUnsupported("strict nest FORCE round trip needs a specified or nested domain")
    halo = carried_wrf_mu_halo(state)
    halo_name = "scratch:wrf_exact_mu_halo" if halo is not None else "mup"
    launches = []
    for kind in fields:
        if kind == "mu":
            continue
        full = kind in ("w", "ph")
        metric = {"u": "msfu", "v": "msfv"}.get(kind, "msft")
        names = (_ATTR.get(kind, kind), "mub2d", "mup", halo_name,
                 "c1f" if full else "c1h", "c2f" if full else "c2h", metric)
        arrays = tuple(state.storage.arrays[name] for name in names)
        parameters = ("field", "mub", "mu", "halo", "c1", "c2", "msf")
        pointers = tuple((parameter, array, state.storage.specs[name].ownership)
                         for parameter, array, name in zip(parameters, arrays, names, strict=True))
        shape = _shape(state, kind)
        mode = {"ph": 1, "w": 2, "u": 3, "v": 4}.get(kind, 0)
        args = (*arrays, *(np.int32(n) for n in
                 (mode, state.has_msf, *shape, *state.storage.specs["mup"].shape)))
        launches.append(_bind("nest", "member_nest_round_trip", state.members,
                              prod(shape), args, pointers, source=_ROUND_TRIP_SOURCE))
    return tuple(launches)


def _scratch_name(endpoint, shape):
    return f"nest:{endpoint}_scratch:" + ":".join(str(extent) for extent in shape)


def member_nest_memory_plan(parent, child, registrations, fields):
    """Exact edge fields and geometry before any CUDA allocation."""
    fields = tuple(fields)
    # Reuse only identical layouts. A prefix reshape of a padded flat bank
    # does not guarantee contiguous inner axes on the CUDA array backend,
    # particularly for the singleton mass-field level. Every kernel here
    # uses the original dense slab layout, so price and allocate that exact
    # shape rather than admitting an unproven view stride.
    specs = []
    for endpoint, state in (("parent", parent), ("child", child)):
        for shape in dict.fromkeys(_shape(state, kind) for kind in fields):
            specs.append(BatchArraySpec(_scratch_name(endpoint, shape), shape, "member"))
    width = bdy_width(child.cfg.spec_zone, child.cfg.relax_zone, child.cfg.spec_bdy_width)
    for kind in fields:
        nz, ny, nx = _shape(child, kind)
        for side in _SIDES:
            shape = (nz, ny, width) if side in ("west", "east") else (nz, width, nx)
            for item in ("value", "tendency"):
                specs.append(BatchArraySpec(f"table:0:{_APPLICATION.get(kind, kind)}:{side}:{item}", shape, "member"))
    for stagger, reg in registrations.items():
        for name in _GEOMETRY:
            host = getattr(reg, name)
            specs.append(BatchArraySpec(f"geometry:{stagger}:{name}", host.shape, "shared", host.dtype))
    return BatchMemoryPlan(tuple(specs), reserved_bytes=0)


class PreparedMemberNestEdge:
    """One resident packed edge, with explicit FORCE/feedback transactions.

    Numerical launches iterate fields and smoother phases, never members.
    Member clock checks and bookkeeping stay ordinary integer metadata.
    Mixed scheme edge diagnosis, parent ozone transfer, active inflow and
    streamed stores require their separate component bindings. This class
    does not remove any forecast-level eligibility guard.
    """
    def __init__(self, parent, child, *, registrations, fields, parent_dt_fp32,
                 parent_interval_ticks, parent_tick_den, available_bytes, feedback=1,
                 smooth_option=0, array_module=None, member_ids=None,
                 parent_clocks=None, child_clocks=None):
        if (not isinstance(parent, BatchedDomainState) or not isinstance(child, BatchedDomainState)
                or parent.members != child.members):
            raise TypeError("member nest edge requires two matching inventoried member rosters")
        if feedback not in (0, 1) or smooth_option not in (0, 1, 2):
            raise ValueError("member feedback and smoothing retain the original options")
        if parent.cfg.nz != child.cfg.nz:
            raise BatchStateUnsupported("member feedback has no vertical mapping between different domain level counts")
        if not isinstance(parent_dt_fp32, np.float32) or not np.isfinite(parent_dt_fp32) or parent_dt_fp32 <= 0:
            raise ValueError("member nest edge requires the original positive FP32 parent step")
        if isinstance(parent_interval_ticks, bool) or int(parent_interval_ticks) <= 0:
            raise ValueError("member nest edge requires a positive original parent interval")
        if isinstance(parent_tick_den, bool) or int(parent_tick_den) <= 0:
            raise ValueError("member nest edge requires a positive original clock denominator")
        self.parent, self.child, self.members = parent, child, child.members
        self.fields = tuple(fields)
        if (not self.fields or len(set(self.fields)) != len(self.fields) or "mu" not in self.fields):
            raise ValueError("member nest edge needs unique active fields including mass")
        self.registrations = dict(registrations)
        if set(self.registrations) != {"m", "x", "y"}:
            raise ValueError("member edge needs all three original stagger registrations")
        self.member_ids = tuple(range(self.members)) if member_ids is None else tuple(member_ids)
        if len(self.member_ids) != self.members or len(set(self.member_ids)) != self.members:
            raise ValueError("member edge IDs must name every packed slot exactly once")
        self.feedback, self.smooth_option = int(feedback), int(smooth_option)
        self.parent_dt_fp32 = parent_dt_fp32
        self.parent_interval_ticks = int(parent_interval_ticks)
        self._clock_contract = (parent_dt_fp32.tobytes(), self.parent_interval_ticks)
        self._tick_den = int(parent_tick_den)
        self._clock_roster = None
        if parent_clocks is not None or child_clocks is not None:
            if parent_clocks is None or child_clocks is None:
                raise ValueError("an edge clock binding needs both original clock rosters")
            self._clock_roster = self._clock_pairs(parent_clocks, child_clocks)[1]
        self.interval_rebind_count = 0
        self.force_count = self.feedback_count = self.generation = 0
        self.valid = False
        self._prepared_feedback = None
        self._feedback_committed = False
        if array_module is None:
            import cupy as array_module
        self.xp = array_module
        from woof.wrf_exact import ENABLED
        self._wrf_exact = ENABLED
        self.plan = member_nest_memory_plan(parent, child, self.registrations, self.fields)
        self.storage = BatchStorage(self.plan, self.members, array_module=array_module, available_bytes=available_bytes)
        self.geometry = {}
        for stagger, reg in self.registrations.items():
            self.geometry[stagger] = {}
            for name in _GEOMETRY:
                target = self.storage.arrays[f"geometry:{stagger}:{name}"]
                target[...] = array_module.asarray(getattr(reg, name))
                self.geometry[stagger][name] = target
        array_module.cuda.get_current_stream().synchronize()
        self.tables, self._forces, self._restrictions, self._smoothers = {}, [], {}, {}
        from woof.wrf_exact import ENABLED as wrf_exact
        self._round_trips = ()
        if wrf_exact:
            self._round_trips = (prepare_member_round_trip(child, self.fields)
                                 + prepare_member_round_trip(parent, self.fields))
        self._parent_views, self._child_views = {}, {}
        width = bdy_width(child.cfg.spec_zone, child.cfg.relax_zone, child.cfg.spec_bdy_width)
        from woof.core.nest import NEST_FORCE_HALO_PARENT_CELLS
        mass_reg = self.registrations["m"]
        i0, j0 = mass_reg.i_parent_start - 1, mass_reg.j_parent_start - 1
        ni, nj = -(-child.cfg.nx // mass_reg.nri), -(-child.cfg.ny // mass_reg.nrj)
        pad = NEST_FORCE_HALO_PARENT_CELLS
        parent_window = (j0 - pad, j0 + nj + pad, i0 - pad, i0 + ni + pad)
        for kind in self.fields:
            reg = self.registrations[_STAGGER.get(kind, "m")]
            ps = self.storage.arrays[_scratch_name("parent", _shape(parent, kind))]
            cs = self.storage.arrays[_scratch_name("child", _shape(child, kind))]
            self._parent_views[kind], self._child_views[kind] = ps, cs
            app = _APPLICATION.get(kind, kind)
            out = {side: tuple(self.storage.arrays[f"table:0:{app}:{side}:{item}"]
                               for item in ("value", "tendency")) for side in _SIDES}
            self.tables[app] = out
            couple_parent = prepare_member_couple_nest_field(parent, kind, out=ps, window=parent_window)
            couple_child = prepare_member_couple_nest_field(child, kind, out=cs, frame_width=width)
            interpolate = prepare_member_bdy_interp1(ps, cs, reg, members=self.members,
                parent_dt_fp32=parent_dt_fp32, parent_interval_ticks=parent_interval_ticks,
                out=out, geometry=self.geometry[_STAGGER.get(kind, "m")],
                spec_zone=child.cfg.spec_zone, relax_zone=child.cfg.relax_zone,
                spec_bdy_width=child.cfg.spec_bdy_width)
            self._forces.append((couple_parent, couple_child, interpolate))
            destination = getattr(parent, _ATTR.get(kind, kind))
            source = (cs if kind == "t" and not self._wrf_exact
                      else getattr(child, _ATTR.get(kind, kind)))
            if kind == "mu":
                destination = destination[:, None]
                source = source[:, None]
            self._restrictions[kind] = prepare_member_copy_fcn(destination, source, reg,
                members=self.members, spec_zone=child.cfg.spec_zone)
            self._smoothers[kind] = prepare_member_smoother(destination, reg, members=self.members,
                smooth_option=smooth_option, scratch=ps)
        from woof.ensemble.batch_diagnostics import prepare_update_diagnostics
        reg = self.registrations["m"]
        ilo, ihi, jlo, jhi = feedback_parent_bounds(reg, spec_zone=child.cfg.spec_zone)
        i0, j0, ni, nj = smoother_parent_window(reg)
        if smooth_option and ni > 0 and nj > 0:
            ilo, ihi = min(ilo, i0), max(ihi, i0 + ni - 1)
            jlo, jhi = min(jlo, j0), max(jhi, j0 + nj - 1)
        self._diagnose = prepare_update_diagnostics(parent, parent.cfg.hypsometric_opt,
            window=(jlo, ilo, jhi - jlo + 1, ihi - ilo + 1))
        self._bound = self._binding_key()

    def _binding_key(self):
        from woof.core.adaptive_clock import ADAPTIVE_DERIVED_RUN_FIELDS
        def config(cfg):
            names = (field.name for field in dataclass_fields(cfg)) if is_dataclass(cfg) else vars(cfg)
            return tuple((name, _exact_key(getattr(cfg, name))) for name in names
                         if name not in ADAPTIVE_DERIVED_RUN_FIELDS)
        buffers = tuple((id(owner), tuple((name, id(array), int(array.data.ptr), tuple(array.shape), tuple(array.strides))
                                      for name, array in owner.storage.arrays.items()))
                     for owner in (self.parent, self.child, self))
        geometry = tuple((stagger, id(reg), tuple(getattr(reg, name) for name in (
            "nri", "nrj", "i_parent_start", "j_parent_start", "nxc", "nyc", "nxp", "nyp", "xstag", "ystag", "wrapper")),
            tuple(getattr(reg, name).tobytes() for name in _GEOMETRY),
            tuple(id(self.geometry[stagger][name]) for name in _GEOMETRY))
            for stagger, reg in sorted(self.registrations.items()))
        views = (tuple((name, id(value)) for name, value in self._parent_views.items()),
                 tuple((name, id(value)) for name, value in self._child_views.items()),
                 tuple((name, side, tuple(id(value) for value in values))
                       for name, sides in self.tables.items() for side, values in sides.items()))
        return buffers, geometry, views, config(self.parent.cfg), config(self.child.cfg), self.parent.has_msf, self.child.has_msf

    def _require_bindings(self):
        if self._binding_key() != self._bound:
            raise BatchStateUnsupported("member nest state or edge backing changed; rebind before coupling")

    def _clock_pairs(self, parent_clocks, child_clocks):
        parents, children = tuple(parent_clocks), tuple(child_clocks)
        if len(parents) != self.members or len(children) != self.members:
            raise ValueError("member FORCE needs every original parent/child clock pair")
        if len({id(clock) for clock in parents + children}) != 2 * self.members:
            raise BatchStateUnsupported("member FORCE clock roster aliases another endpoint or member")
        roster = tuple((id(parent), id(child)) for parent, child in zip(parents, children, strict=True))
        if self._clock_roster is not None and roster != self._clock_roster:
            raise BatchStateUnsupported("member FORCE clock roster changed from its original endpoint owners")
        for clocks, bank in ((parents, self.parent), (children, self.child)):
            nodes = getattr(bank, "nodes", None)
            if nodes is not None and any(node.clock is not clock for node, clock in zip(nodes, clocks, strict=True)):
                raise BatchStateUnsupported("member FORCE clocks differ from the current original state bindings")
        pairs = tuple(zip(parents, children, strict=True))
        for parent, child in pairs:
            if (int(parent.spec.grid_id) != int(self.parent.cfg.grid_id)
                    or int(child.spec.grid_id) != int(self.child.cfg.grid_id)
                    or int(child.spec.parent_id) != int(parent.spec.grid_id)):
                raise BatchStateUnsupported("member FORCE clock roster belongs to different domain endpoints")
        return pairs, roster

    def validate_parent_interval(self, *, parent_clocks, child_clocks, require_bound=False):
        """Validate live clock metadata before preparing any new launch args."""
        self._require_bindings()
        if self._prepared_feedback is not None:
            raise ValueError("member FORCE needs no unfinished feedback transaction")
        pairs, roster = self._clock_pairs(parent_clocks, child_clocks)
        from woof.core.adaptive_timestep import real_time_fp32
        snapshots = []
        for parent, child in pairs:
            row = []
            for clock in (parent, child):
                words = []
                for name in ("tick_den", "step_ticks", "ticks", "run_ticks", "step_count"):
                    value = getattr(clock, name)
                    if isinstance(value, (bool, np.bool_)):
                        raise BatchStateUnsupported("member FORCE clock words must retain their original integer types")
                    try:
                        value = index(value)
                    except TypeError as error:
                        raise BatchStateUnsupported("member FORCE clock words must retain their original integer types") from error
                    if value < 0 or (name in {"tick_den", "step_ticks"} and value == 0):
                        raise BatchStateUnsupported("member FORCE clock has a nonpositive interval or invalid calendar word")
                    words.append(value)
                if not isinstance(clock.dt_fp32, np.float32) or not np.isfinite(clock.dt_fp32) or clock.dt_fp32 <= 0:
                    raise BatchStateUnsupported("member FORCE clock lacks its original positive FP32 step")
                if clock.ticks > clock.run_ticks:
                    raise BatchStateUnsupported("member FORCE clock advanced beyond its original run calendar")
                row.append((*words, clock.dt_fp32.tobytes(), getattr(clock, "adaptive_state", None) is not None))
            if parent.tick_den != self._tick_den or child.tick_den != self._tick_den:
                raise BatchStateUnsupported("member FORCE clock lattice differs across endpoints or packed members; regroup/rebind before launch")
            if parent.ticks - child.ticks != parent.step_ticks:
                raise BatchStateUnsupported("member FORCE parent lead or step differs from its bound original interval; regroup/rebind before launch")
            contract = parent.dt_fp32.tobytes(), parent.step_ticks
            if require_bound and contract != self._clock_contract:
                raise BatchStateUnsupported("member FORCE parent lead or step differs from its bound original interval; regroup/rebind before launch")
            for clock in (parent, child):
                adaptive = getattr(clock, "adaptive_state", None) is not None
                # Fixed clocks retain the original chained-division image.
                # Changed intervals and published controllers use REAL time.
                changed = clock.step_ticks != clock.spec.step_ticks
                if adaptive or (not require_bound and changed):
                    expected = real_time_fp32(Fraction(clock.step_ticks, clock.tick_den))
                else:
                    expected = clock.spec.dt_fp32
                if clock.dt_fp32.tobytes() != expected.tobytes():
                    raise BatchStateUnsupported("member FORCE FP32 step differs from its original controller or fixed interval image")
            snapshots.append(tuple(row))
        if not require_bound and any(row != snapshots[0] for row in snapshots[1:]):
            raise BatchStateUnsupported("member FORCE has divergent current member clocks; one rebound interval cannot substitute them")
        return pairs[0][0].dt_fp32, int(pairs[0][0].step_ticks), roster, tuple(snapshots)

    def rebind_parent_interval(self, *, parent_clocks, child_clocks):
        """Rebuild only scalar-argument FORCE launches for one live interval.

        Every launch is prepared and every original binding is revalidated
        before publication. Coupling, tables, geometry, feedback and GPU
        allocations remain owned by the existing edge.
        """
        parents, children = tuple(parent_clocks), tuple(child_clocks)
        desired = self.validate_parent_interval(parent_clocks=parents, child_clocks=children)
        dt, ticks, roster, _ = desired
        if (dt.tobytes(), ticks) == self._clock_contract:
            self._clock_roster = roster
            return False
        prepared = []
        for kind, (couple_parent, couple_child, _) in zip(self.fields, self._forces, strict=True):
            stagger = _STAGGER.get(kind, "m")
            interpolate = prepare_member_bdy_interp1(self._parent_views[kind], self._child_views[kind],
                self.registrations[stagger], members=self.members, parent_dt_fp32=dt,
                parent_interval_ticks=ticks, out=self.tables[_APPLICATION.get(kind, kind)],
                geometry=self.geometry[stagger], spec_zone=self.child.cfg.spec_zone,
                relax_zone=self.child.cfg.relax_zone, spec_bdy_width=self.child.cfg.spec_bdy_width)
            prepared.append((couple_parent, couple_child, interpolate))
        current = self.validate_parent_interval(parent_clocks=parents, child_clocks=children)
        if desired != current:
            raise BatchStateUnsupported("member FORCE clocks changed while rebound launches were prepared; no interval was published")
        self._forces, self.parent_dt_fp32, self.parent_interval_ticks = prepared, dt, ticks
        self._clock_contract, self._clock_roster = (dt.tobytes(), ticks), roster
        self.interval_rebind_count += 1
        self.valid = False
        return True

    def force(self, *, parent_clocks, child_clocks):
        """Produce rolling tables only after each own parent leads its child."""
        parent_clocks, child_clocks = tuple(parent_clocks), tuple(child_clocks)
        self.validate_parent_interval(parent_clocks=parent_clocks, child_clocks=child_clocks, require_bound=True)
        pairs = tuple(zip(parent_clocks, child_clocks, strict=True))
        for row in self._forces:
            for operation in row:
                operation()
        for operation in self._round_trips:
            operation()
        for _, child in pairs:
            child.mark_force()
        self.force_count += 1
        self.generation += 1
        self.valid = True
        return self.tables

    def feedback_prepare(self, *, parent_clocks, child_clocks):
        if not self.feedback:
            return
        self._require_bindings()
        pairs = tuple(zip(parent_clocks, child_clocks, strict=True))
        if len(pairs) != self.members or self._prepared_feedback is not None:
            raise RuntimeError("member feedback needs every clock pair and a new transaction")
        if any(int(parent.ticks) != int(child.ticks) or parent.tick_den != child.tick_den
               for parent, child in pairs):
            raise RuntimeError("member feedback requires each parent and child at the same original integer time")
        self._prepared_feedback = (pairs, tuple((int(p.ticks), int(c.ticks)) for p, c in pairs))
        self._feedback_committed = False

    def feedback_commit(self):
        if not self.feedback:
            return
        if self._prepared_feedback is None:
            raise RuntimeError("member feedback commit has no prepared transaction")
        self._require_bindings()
        if self._feedback_committed:
            raise RuntimeError("member feedback transaction was already committed")
        pairs, ticks = self._prepared_feedback
        if tuple((int(p.ticks), int(c.ticks)) for p, c in pairs) != ticks:
            raise RuntimeError("member feedback clocks changed after prepare")
        from woof.wrf_exact import ENABLED as wrf_exact
        order = ("mu",) + tuple(kind for kind in self.fields if kind != "mu")
        for kind in order:
            reg = self.registrations[_STAGGER.get(kind, "m")]
            if kind == "t" and not self._wrf_exact:
                raw = self._child_views[kind]
                self.xp.copyto(raw, self.child.thp)
                if not wrf_exact:
                    raw += _base_theta(self.child)
                    raw -= np.float32(300)
            self._restrictions[kind]()
            if kind == "t" and not self._wrf_exact:
                ilo, ihi, jlo, jhi = feedback_parent_bounds(reg, spec_zone=self.child.cfg.spec_zone)
                if ihi >= ilo and jhi >= jlo:
                    window = self.parent.thp[..., jlo:jhi + 1, ilo:ihi + 1]
                    window += np.float32(300)
                    base = _base_theta(self.parent)
                    window -= base if base.shape[-2:] == (1, 1) else base[..., jlo:jhi + 1, ilo:ihi + 1]
        if self.smooth_option:
            for kind in self.fields:
                self._smoothers[kind]()
                if kind not in _SIGNED:
                    reg = self.registrations[_STAGGER.get(kind, "m")]
                    i0, j0, ni, nj = smoother_parent_window(reg)
                    if ni > 0 and nj > 0:
                        field = getattr(self.parent, _ATTR.get(kind, kind))
                        window = field[..., j0:j0 + nj, i0:i0 + ni]
                        self.xp.maximum(window, np.float32(0), out=window)
        self.feedback_count += 1
        self._feedback_committed = True

    def feedback_finalize(self):
        if not self.feedback:
            return
        if self._prepared_feedback is None:
            raise RuntimeError("member feedback finalize has no prepared transaction")
        if not self._feedback_committed:
            raise RuntimeError("member feedback must commit before final diagnostics")
        self._require_bindings()
        if not self._wrf_exact:
            self._diagnose()
        self._prepared_feedback = None
        self._feedback_committed = False

    def invalidate(self):
        self.valid = False

    def receipt(self):
        return {"contract": CONTRACT, "members": self.members, "member_ids": list(self.member_ids),
            "execution": "member_indexed_gpu_nest_coupling", "force_count": self.force_count,
            "feedback_count": self.feedback_count, "generation": self.generation, "valid": self.valid,
            "fields": list(self.fields), "parent_dt_fp32_hex": self.parent_dt_fp32.tobytes().hex(),
            "parent_interval_ticks": self.parent_interval_ticks, "smooth_option": self.smooth_option,
            "parent_tick_den": self._tick_den,
            "interval_rebind_count": self.interval_rebind_count,
            "edge_plan_bytes": self.plan.required_bytes(self.members),
            "edge_allocations": list(self.plan.inventory(self.members)),
            "nest_source_sha256": sha256(nesting_source().encode()).hexdigest(),
            "nest_compile_options": list(_NEST_OPTIONS),
            "round_trip_source_sha256": sha256(_ROUND_TRIP_SOURCE.encode()).hexdigest(),
            "numerical_member_loops": 0, "forecast_graph_admission_changed": False}


__all__ = ["CONTRACT", "nesting_source", "prepare_member_bdy_interp1", "prepare_member_copy_fcn",
           "prepare_member_smoother", "prepare_member_couple_nest_field", "member_nest_memory_plan",
           "PreparedMemberNestEdge"]
