"""Shared preparation and legacy-radiation geometry for member columns.

This module does not advance independent member drivers. A scalar bootstrap
may initialize common land/physics metadata once, with its transient memory
reported, before private fields are copied to an admitted member bank.
"""

from __future__ import annotations

import ast
from copy import copy, deepcopy
from dataclasses import dataclass, is_dataclass, fields as dataclass_fields, replace
from hashlib import sha256
import inspect
from operator import index
import textwrap

import numpy as np

from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan, BatchStorage


def legacy_member_source(source, *, members, member_columns):
    """Integer-only ozone sharing and member-interleaved LW scatter."""
    members = _positive(members, "radiation members")
    member_columns = _positive(member_columns, "radiation member columns")
    replacements = {
        "const float* oz = ozmixt + (long long)col * levsiz;":
            "const float* oz = ozmixt + ((long long)col % __ensemble_member_columns) * levsiz;",
        "long long dest = (long long)k * ncol + c0 + col;":
            "long long logical = c0 + col;\n    long long actual = (logical % __ensemble_members) * __ensemble_member_columns + logical / __ensemble_members;\n    long long dest = (long long)k * ncol + actual;",
        "glw[c0 + col]": "glw[actual]",
        "olr[c0 + col]": "olr[actual]",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError(f"legacy radiation indexing changed at {before!r}; member adapter needs a new audit")
        source = source.replace(before, after)
    return (f"#define __ensemble_members {int(members)}\n"
            f"#define __ensemble_member_columns {int(member_columns)}\n" + source)


def _positive(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer")
    value = index(value)
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _clone_tree(value, array_type, resolve_array):
    """Rebind arrays while retaining aliases and frozen diagnostic records."""
    if isinstance(value, array_type):
        return resolve_array(value)
    if isinstance(value, dict):
        return {name: _clone_tree(item, array_type, resolve_array)
                for name, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _clone_tree(getattr(value, field.name), array_type, resolve_array)
                                  for field in dataclass_fields(value) if field.init})
    if isinstance(value, tuple):
        return tuple(_clone_tree(item, array_type, resolve_array) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item, array_type, resolve_array) for item in value]
    # Scalar control owners can contain mutable dictionaries (the radiation
    # carrier contract does). They must not retain the bootstrap's controls.
    return deepcopy(value)


_REPEAT_SOURCE = r'''
extern "C" __global__ void ensemble_geometry_repeat_words(
    const unsigned int *source, unsigned int *destination,
    unsigned long long cells, int members) {
    unsigned long long point = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= cells * (unsigned long long)members) return;
    destination[point] = source[point % cells];
}
'''


#: How many times RRTMGLegacyRadiation.__call__ subscripts each shared surface
#: field, every one of them as ``surf[field][sel]``: xland at the shortwave call
#: and again for the aer_opt=3 land/water aerosol type (rrtmg_aer3_type_source =
#: "landmask").  The audit below refuses on any other count, because an
#: unrewritten site would read one member's columns for the whole batch.
_LEGACY_SHARED_SURFACE_SITES = {"xland": 2}


def prepare_legacy_member_radiation(original, *, members, available_bytes,
                                   array_module=None, shared_surface_fields=()):
    """Reuse one legacy table/geometry owner and batch both spectra.

    Latitude, longitude and monthly ozone interpolation remain one bank.
    Chunk selectors interleave members, so a native chunk advances all
    members' columns. McICA remains the unchanged per-pressure stream. The
    underlying native engines retain their priced chunk workspaces.
    """
    members = _positive(members, "radiation members")
    if members == 1:
        return original
    if array_module is None:
        import cupy as array_module
    from woof.core import rrtmg_legacy as legacy
    from woof.certify.kernel_manifest import record_module
    if not isinstance(original, legacy.RRTMGLegacyRadiation):
        raise TypeError("member legacy radiation needs the initialized native legacy adapter")
    if original._ozone_provider is not None:
        raise ValueError("parent-routed ozone needs member-aware parent column tables before radiation can share geometry")
    ny, nx = original.latitude_deg.shape
    cells = ny * nx
    shared_surface_fields = tuple(shared_surface_fields)
    if set(shared_surface_fields) - {"xland"}:
        raise ValueError("legacy radiation sharing here owns immutable xland only")
    device = int(array_module.cuda.runtime.getDevice())
    plan = BatchMemoryPlan((BatchArraySpec("radiation:coszen", (ny, nx), "member"),), reserved_bytes=0)
    storage = BatchStorage(plan, members, array_module=array_module, available_bytes=available_bytes)
    repeated = storage.arrays["radiation:coszen"].reshape(members * ny, nx)
    repeat_module = array_module.RawModule(code=_REPEAT_SOURCE, options=("--std=c++17",))
    repeat = repeat_module.get_function("ensemble_geometry_repeat_words")
    record_module("woof.ensemble.batch_physics_init:geometry-repeat", source=_REPEAT_SOURCE,
                  options=("--std=c++17",), module=repeat_module)

    # The adapter glue keeps its explicit unflushed PTX arithmetic and its
    # native compile route. All other entries are the original functions.
    path = legacy.Path(legacy.__file__).parent / "kernels" / "rrtmg_legacy_adapter.cu"
    scalar_source = path.read_text(encoding="utf-8")
    source = legacy_member_source(scalar_source, members=members, member_columns=cells)
    from cupy.cuda import compiler
    binary, _ = compiler.compile_using_nvrtc(source, ("-std=c++17", "--ftz=false"), None,
                                            "rrtmg_legacy_member_adapter.cu")
    module = array_module.cuda.function.Module()
    module.load(binary.encode() if isinstance(binary, str) else binary)
    record_module(f"woof.ensemble.batch_physics_init:legacy-radiation[members={members},columns={cells}]",
                  source=source, options=("-std=c++17", "--ftz=false"), module=module)
    modified_entries = {name: module.get_function(name) for name in ("rla_ozn_p_int", "rla_lw_out_grid")}

    def native_entry(name):
        return modified_entries[name] if name in modified_entries else legacy._adapter_kernel(name)

    def device_geometry(source_bank, selected):
        # Selected columns are a device vector, not a member loop. The small
        # native prep chunk consumes this integer gather result immediately.
        return source_bank[selected % cells]

    def host_geometry(source_bank, selected):
        return source_bank[np.asarray(selected) % cells]

    def repeat_indices(source_indices):
        source_indices = np.asarray(source_indices, dtype=np.int64)
        return (source_indices[:, None] + np.arange(members, dtype=np.int64)[None] * cells).reshape(-1)

    def member_selection(start, end):
        logical = array_module.arange(start, end, dtype=array_module.int64)
        return (logical % members) * cells + logical // members

    def chunk_size(value):
        return max(members, int(value) // members * members)

    def coszen_result(bank, unused_ny, unused_nx):
        repeat(((cells * members + 255) // 256,), (256,),
               (bank, repeated, np.uint64(cells), np.int32(members)))
        return repeated

    method_source = textwrap.dedent(inspect.getsource(legacy.RRTMGLegacyRadiation.__call__))
    tree = ast.parse(method_source)
    counts = {"shape": 0, "ozone_shape": 0, "lat": 0, "coszen": 0,
              "day": 0, "night": 0, "lw_selector": 0, "lw_chunk": 0, "sw_chunk": 0, "result": 0,
              "surface": 0}

    class MemberColumns(ast.NodeTransformer):
        def visit_Compare(self, node):
            text = ast.unparse(node)
            if text == "self.latitude_deg.shape != (ny, nx)":
                counts["shape"] += 1
                return ast.parse(f"self.latitude_deg.shape != ({ny}, {nx})", mode="eval").body
            if text == "ozmixt.shape != (ncol, pin.size)":
                counts["ozone_shape"] += 1
                return ast.parse(f"ozmixt.shape != ({cells}, pin.size)", mode="eval").body
            return self.generic_visit(node)

        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Subscript) and ast.unparse(node.value) in tuple(f"surf['{field}']" for field in shared_surface_fields):
                counts["surface"] += 1
                return ast.Call(ast.Name("__ensemble_device_geometry", ast.Load()),
                                [node.value, self.visit(node.slice)], [])
            if isinstance(node.value, ast.Name) and node.value.id in ("lat_d", "coszen_d"):
                counts["lat" if node.value.id == "lat_d" else "coszen"] += 1
                return ast.Call(ast.Name("__ensemble_device_geometry", ast.Load()),
                                [node.value, self.visit(node.slice)], [])
            if isinstance(node.value, ast.Name) and node.value.id == "coszen":
                return ast.Call(ast.Name("__ensemble_host_geometry", ast.Load()),
                                [node.value, self.visit(node.slice)], [])
            return self.generic_visit(node)

        def visit_Assign(self, node):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if names in (["day_idx"], ["night_idx"]):
                counts["day" if names == ["day_idx"] else "night"] += 1
                node.value = ast.Call(ast.Name("__ensemble_repeat_indices", ast.Load()), [node.value], [])
            if names in (["chunk_lw"], ["chunk_sw"]):
                counts["lw_chunk" if names == ["chunk_lw"] else "sw_chunk"] += 1
                node.value = ast.Call(ast.Name("__ensemble_chunk_size", ast.Load()), [node.value], [])
            if names == ["sel"] and ast.unparse(node.value) == "slice(c0, c1)":
                counts["lw_selector"] += 1
                node.value = ast.parse("__ensemble_member_selection(c0, c1)", mode="eval").body
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node) == "coszen_d.reshape(ny, nx)":
                counts["result"] += 1
                return ast.parse("__ensemble_coszen_result(coszen_d, ny, nx)", mode="eval").body
            return self.generic_visit(node)

    tree = MemberColumns().visit(tree)
    if counts != {"shape": 1, "ozone_shape": 1, "lat": 1, "coszen": 1,
                  "day": 1, "night": 1, "lw_selector": 1, "lw_chunk": 1, "sw_chunk": 1, "result": 1,
                  "surface": sum(_LEGACY_SHARED_SURFACE_SITES[field] for field in shared_surface_fields)}:
        raise ValueError(f"legacy radiation member call-site audit changed: {counts}")
    ast.fix_missing_locations(tree)
    namespace = dict(legacy.RRTMGLegacyRadiation.__call__.__globals__)
    namespace.update(_adapter_kernel=native_entry,
                     __ensemble_device_geometry=device_geometry,
                     __ensemble_host_geometry=host_geometry,
                     __ensemble_repeat_indices=repeat_indices,
                     __ensemble_member_selection=member_selection,
                     __ensemble_chunk_size=chunk_size,
                     __ensemble_coszen_result=coszen_result)
    exec(compile(tree, "<admitted-member-legacy-radiation>", "exec"), namespace)
    # Preserve the native public type marker consumed by existing ozone and
    # radiation-composition metadata helpers. The receipt distinguishes the
    # member indexing from the native scalar implementation.
    cls = type(type(original).__name__, (type(original),),
               {"__module__": type(original).__module__, "__call__": namespace["__call__"]})
    result = cls.__new__(cls)
    result.__dict__.update(original.__dict__)
    result._ensemble_radiation_owners = (original, module, storage, repeat)
    result._ensemble_radiation_receipt = {
        "members": members, "geometry_bank_shape": [ny, nx], "geometry_shared_once": True,
        "scalar_source_sha256": sha256(scalar_source.encode()).hexdigest(),
        "source_sha256": sha256(source.encode()).hexdigest(),
        "loaded_image_sha256": sha256(binary.encode() if isinstance(binary, str) else binary).hexdigest(),
        "python_scalar_source_sha256": sha256(method_source.encode()).hexdigest(),
        "mcica_seed_policy": "unchanged pressure-derived per-column streams",
        "chunk_order": "cell then member", "output_allocations": list(plan.inventory(members)),
        "shared_surface_fields": list(shared_surface_fields),
        "native_chunk_workspace_policy": "existing legacy_radiation_vram_bytes with member-aligned chunk sizes",
        "device": device}
    return result


class InitializedMemberPhysics:
    """Owned driver and banks, with explicit pack/compute/microphysics seams."""

    def __init__(self, batch, driver, state, bank, atmosphere, microphysics, receipt):
        self.batch, self.driver, self.state, self.bank = batch, driver, state, bank
        self.atmosphere, self.microphysics = atmosphere, microphysics
        self.receipt = receipt

    def pack_model_state(self):
        for name in self.receipt["model_member_fields"]:
            self.bank.pack(name, self.batch.storage.arrays[name])
        self.state.elapsed_seconds = float(self.batch.elapsed_seconds)

    def compute(self):
        self.pack_model_state()
        return self.driver.compute(self.state, self.batch.cfg)

    def apply_microphysics(self, *, refl_10cm_due=False):
        self.pack_model_state()
        result = self.microphysics(refl_10cm_due=refl_10cm_due)
        self.driver.accept_microphysics(result, dt=self.batch.cfg.dt)
        # The original scalar adapter mutates only these prognostic/state
        # fields. Packing masks/temporaries cannot become a state update.
        from woof.core.physics_inventory import ring_guard_row
        changed = tuple(ring_guard_row(getattr(self.batch.cfg, "mp_physics", 8))["state_fields"]) + ("h_diabatic",)
        for name in changed:
            if name in self.receipt["model_member_fields"]:
                self.bank.unpack(name, self.batch.storage.arrays[name])
        return result


_BOOTSTRAP_SHARED_OWNERS = frozenset({"noah_params", "radiation_callable", "cumulus_callable",
                                     "noahmp_params", "noahmp_geometry", "ruc_params", "state"})


def _bootstrap_shared_owners(cfg):
    return (_BOOTSTRAP_SHARED_OWNERS - {"cumulus_callable"}
            if getattr(cfg, "cu_physics", 0) == 99 else _BOOTSTRAP_SHARED_OWNERS)
#: Static surface fields the packed land launch can take per member when the
#: members' bootstraps disagree on them. xland, lakemask and landmask stay
#: shared: the radiation and surface-layer launches bind them as one bank.
DEMOTABLE_SURFACE_FIELDS = frozenset({"ivgtyp", "isltyp", "shdmin", "shdmax", "tmn", "snoalb", "embck"})


def _bootstrap_array_inventory(cfg, bootstrap, array_type, *, alias_paths=None, member_owned_surface=()):
    """One array/alias authority for native planning and construction.

    ``alias_paths``, when a dict is supplied, receives every visited path
    mapped to the entry name that owns its bytes, so a caller can find an
    aliased array (``driver/microphysics/rainnc`` is ``driver/fields/rainnc``)
    in a snapshot keyed by canonical path. ``member_owned_surface`` names
    static surface fields that differ between the members of one pack (deep
    soil temperature from another preparation chain, say); they are packed
    per member instead of shared, which the land launch supports.
    """
    from woof.ensemble.batch_physics import NOAH_SHARED_FIELDS
    member_owned_surface = frozenset(member_owned_surface)
    if member_owned_surface - DEMOTABLE_SURFACE_FIELDS:
        raise ValueError(f"{sorted(member_owned_surface - DEMOTABLE_SURFACE_FIELDS)} are bound as shared "
                         "by the radiation or surface-layer launch and cannot be member-owned")
    surface_shared = (NOAH_SHARED_FIELDS | {"lakemask", "landmask"}) - member_owned_surface
    shared_paths = {f"driver/fields/{field}" for field in surface_shared}
    entries, pointers = [], {}
    def register(value, path):
        pointer = getattr(value.data, "ptr", None)
        if pointer is None:
            pointer = value.__array_interface__["data"][0]
        key = (int(pointer), int(value.nbytes), value.dtype.str)
        if key not in pointers:
            if value.ndim not in (2, 3) or value.dtype not in (np.dtype("float32"), np.dtype("int32"), np.dtype("uint32")):
                raise ValueError(f"physics bootstrap array {path} has no admitted member-column shape")
            if not value.flags.c_contiguous or value.shape[-2:] not in ((cfg.ny, cfg.nx), (cfg.ny, cfg.nx + 1), (cfg.ny + 1, cfg.nx)):
                raise ValueError(f"physics bootstrap array {path} has no contiguous member-grid slab")
            name = "bootstrap_" + str(len(entries))
            ownership = "shared" if path in shared_paths else "member"
            entries.append((name, value, path, ownership))
            pointers[key] = name
        if alias_paths is not None:
            alias_paths[path] = pointers[key]
        return pointers[key]
    def inspect_tree(value, path):
        if isinstance(value, array_type):
            register(value, path)
        elif isinstance(value, dict):
            for name, item in value.items():
                inspect_tree(item, f"{path}/{name}")
        elif is_dataclass(value) and not isinstance(value, type):
            for field in dataclass_fields(value):
                inspect_tree(getattr(value, field.name), f"{path}/{field.name}")
        elif isinstance(value, (tuple, list)):
            for pos, item in enumerate(value):
                inspect_tree(item, f"{path}/{pos}")
    for name, value in bootstrap.__dict__.items():
        if name == "cumulus_callable" and getattr(cfg, "cu_physics", 0) == 99:
            from woof.core.kf import KainFritsch
            from woof.ensemble.batch_kf import KF_OLD_CARRIERS
            if type(value) is not KainFritsch or value.scheme != 99:
                raise ValueError("packed CU99 needs the native KainFritsch adapter")
            for carrier in KF_OLD_CARRIERS:
                inspect_tree(getattr(value, carrier), f"driver/cumulus_callable/{carrier}")
        elif name not in _BOOTSTRAP_SHARED_OWNERS:
            inspect_tree(value, f"driver/{name}")
    return entries, pointers, surface_shared


def _owner_digest(value, array_type):
    """A byte digest of one shared driver owner (tables, geometry, controls).

    Arrays contribute shape, dtype and bytes; plain scalars contribute their
    typed value; objects (the radiation adapter is one, and it is callable)
    contribute their attributes; routines, classes and loaded modules
    contribute their type name only, because they carry no member words.
    """
    import inspect
    digest = sha256()
    def walk(item, path):
        digest.update(path.encode())
        if isinstance(item, (array_type, np.ndarray)):
            host = item.get() if hasattr(item, "get") else item
            host = np.ascontiguousarray(host)
            digest.update(f"{host.shape}{host.dtype.str}".encode())
            digest.update(host.tobytes())
        elif isinstance(item, dict):
            for name in sorted(item, key=str):
                walk(item[name], f"{path}/{name}")
        elif is_dataclass(item) and not isinstance(item, type):
            for field in dataclass_fields(item):
                walk(getattr(item, field.name), f"{path}/{field.name}")
        elif isinstance(item, (tuple, list)):
            for position, child in enumerate(item):
                walk(child, f"{path}/{position}")
        elif isinstance(item, (np.generic,)):
            digest.update(item.dtype.str.encode() + item.tobytes())
        elif item is None or isinstance(item, (bool, int, float, str, bytes)):
            digest.update(repr((type(item).__name__, item)).encode())
        elif (inspect.isroutine(item) or inspect.isclass(item) or inspect.ismodule(item)
              or type(item).__module__.startswith("cupy")):
            digest.update(type(item).__name__.encode())
        elif hasattr(item, "__dict__"):
            for name in sorted(vars(item)):
                walk(getattr(item, name), f"{path}/{name}")
        else:
            digest.update(type(item).__name__.encode())
    walk(value, "")
    return digest.hexdigest()


@dataclass(frozen=True)
class PhysicsBootstrapSnapshot:
    """Host words of one ordinary initialized driver, keyed by inventory path.

    ``structure`` lists every unique allocation as (path, shape, dtype,
    ownership) in inventory order; ``arrays`` holds its host words by that
    canonical path; ``aliases`` maps every visited path to the canonical
    one; ``owner_digests`` fingerprints the owners a packed driver takes
    from the root bootstrap (radiation tables and geometry, land tables)
    so a member whose bootstrap built different ones is refused by name.
    """
    structure: tuple
    arrays: object
    aliases: object
    owner_digests: object

    @property
    def nbytes(self):
        return sum(int(value.nbytes) for value in self.arrays.values())

    @property
    def layout(self):
        """Paths, shapes and dtypes without ownership, which a pack decides."""
        return tuple((path, shape, dtype) for path, shape, dtype, _ownership in self.structure)

    def member_array(self, path):
        return self.arrays[self.aliases[path]]


def bootstrap_physics_snapshot(cfg, bootstrap_driver, *, array_module=None):
    """Snapshot an initialized driver's words to host, for one member slot."""
    if array_module is None:
        import cupy as array_module
    from types import MappingProxyType
    aliases = {}
    entries, _pointers, _surface_shared = _bootstrap_array_inventory(
        cfg, bootstrap_driver, array_module.ndarray, alias_paths=aliases)
    arrays, structure = {}, []
    for _name, value, path, ownership in entries:
        host = value.get() if hasattr(value, "get") else np.array(value, copy=True)
        arrays[path] = np.ascontiguousarray(host)
        structure.append((path, tuple(int(n) for n in value.shape), value.dtype.str, ownership))
    names = {name: path for name, _value, path, _ownership in entries}
    digests = {owner: _owner_digest(getattr(bootstrap_driver, owner, None), array_module.ndarray)
               for owner in sorted(_bootstrap_shared_owners(cfg) - {"state"})}
    return PhysicsBootstrapSnapshot(tuple(structure), MappingProxyType(arrays),
        MappingProxyType({path: names[name] for path, name in aliases.items()}),
        MappingProxyType(digests))


def bootstrap_physics_structure(cfg, bootstrap_driver, *, array_module=None):
    """The inventory structure and shared-owner digests of a live driver, no array download."""
    if array_module is None:
        import cupy as array_module
    entries, _pointers, _surface_shared = _bootstrap_array_inventory(cfg, bootstrap_driver, array_module.ndarray)
    structure = tuple((path, tuple(int(n) for n in value.shape), value.dtype.str, ownership)
                      for _name, value, path, ownership in entries)
    shared = {path: value for _name, value, path, ownership in entries if ownership == "shared"}
    digests = {owner: _owner_digest(getattr(bootstrap_driver, owner, None), array_module.ndarray)
               for owner in sorted(_bootstrap_shared_owners(cfg) - {"state"})}
    return structure, shared, digests


@dataclass(frozen=True)
class NativeBootstrapAllocationPlans:
    model_names: tuple
    scratch_fields: tuple
    column_fields: tuple
    bank_plan: BatchMemoryPlan
    shared_plan: BatchMemoryPlan
    status_plan: BatchMemoryPlan
    bootstrap_array_paths: tuple


def _bootstrap_allocation_declarations(cfg, state_specs, entries):
    from woof.ensemble.batch_physics import (
        ColumnField, column_memory_plan, thompson_column_scratch_fields,
        registered_column_scratch_fields)
    if cfg.mp_physics == 55:
        scratch_fields = registered_column_scratch_fields(cfg, nz=cfg.nz, ny=cfg.ny, nx=cfg.nx, reflectivity=True)
    else:
        scratch_fields = thompson_column_scratch_fields(
            nz=cfg.nz, ny=cfg.ny, nx=cfg.nx, reflectivity=True,
            cloud_booking=getattr(cfg, "thompson_cloud_surface_booking", False) is True,
            use_theta_m=getattr(cfg, "use_theta_m", 0))
    model_names = tuple(name for name in (
        "p", "alt", "thp", "php", "u", "v", "w", "mup", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr",
        "h_diabatic", "effc", "effi", "effs", "effr", "rthften", "rqvften",
        "qi2", "qi3", "qvoli", "qaoli", "ni2", "qvoli2", "qaoli2", "ni3", "qvoli3", "qaoli3")
        if name in state_specs and state_specs[name].ownership == "member")
    declarations = tuple(ColumnField(name, state_specs[name].shape, state_specs[name].dtype) for name in model_names)
    declarations += tuple(scratch_fields)
    declarations += tuple(ColumnField(name, value.shape, value.dtype.str) for name, value, _, ownership in entries if ownership == "member")
    shared_plan = BatchMemoryPlan(tuple(BatchArraySpec(f"physics:{name}", value.shape, "shared", value.dtype.str)
                                       for name, value, _, ownership in entries if ownership == "shared"), reserved_bytes=0)
    status_plan = BatchMemoryPlan((BatchArraySpec("physics:validation_status", (1,), "shared"),), reserved_bytes=0)
    return NativeBootstrapAllocationPlans(model_names, tuple(scratch_fields), declarations,
        column_memory_plan(declarations), shared_plan, status_plan,
        tuple((name, path) for name, _, path, _ in entries))


def native_bootstrap_allocation_plans(cfg, bootstrap_driver, *, state_specs, array_type=None,
                                      member_owned_surface=()):
    """Exact native bootstrap declarations, without allocation or source loads.

    The result contains shapes and paths only. It retains no bootstrap array
    or driver references, so a planning receipt cannot keep a scalar member
    alive after the construction seam releases it.
    """
    if array_type is None:
        array_type = type(bootstrap_driver.state.p)
    if not isinstance(state_specs, dict):
        state_specs = {spec.name: spec for spec in state_specs}
    entries, _pointers, _surface_shared = _bootstrap_array_inventory(cfg, bootstrap_driver, array_type,
                                                                     member_owned_surface=member_owned_surface)
    return _bootstrap_allocation_declarations(cfg, state_specs, entries)


def _native_bootstrap_requirements(batch):
    """The existing qualified packed suite, independent of source format."""
    from woof.ensemble.batch_state import BatchedDomainState
    if not isinstance(batch, BatchedDomainState):
        raise TypeError("physics initialization needs admitted BatchedDomainState")
    cfg, members = batch.cfg, batch.members
    if (cfg.mp_physics not in (8, 55) or (cfg.sf_surface_physics, cfg.bl_pbl_physics) != (2, 1)
            or cfg.sf_sfclay_physics not in (1, 91)):
        raise ValueError("this initializer owns Thompson or ISHMAEL with Noah/YSU/MM5")
    if members == 1:
        raise ValueError("one member uses its ordinary prepared physics driver and original initializer")
    from woof.physics_compat import rrtmg_variant, RRTMG_VARIANT_LEGACY
    if rrtmg_variant(cfg) != RRTMG_VARIANT_LEGACY:
        raise ValueError("this initialization binds the actual legacy RRTMG suite; the selected solver needs its own qualified factory")
    needed_shared = ("thb", "phb", "mub2d", "c1h", "c2h", "dnw", "fnm", "fnp", "msft", "msfu", "msfv")
    if any(batch.storage.specs[name].ownership != "shared" for name in needed_shared):
        raise ValueError("common physics requires byte-verified shared base/coordinate/metric banks before initialization")
    return cfg, members, needed_shared


def initialize_wrfinput_member_physics(batch, restored, *, start_time, landuse=None,
                                      available_bytes, column_chunk=None,
                                      constant_glw_wm2=None, fractional_seaice=False,
                                      cam_ozone=None, trace_gas_overrides=None,
                                      array_module=None, hydro_parent_geometry=None):
    """Use the ordinary WRF-input initializer once, then pack its bootstrap.

    This is for a common initial/surface source before member perturbations.
    Distinct surface/soil sources need their own admitted initializer banks.
    """
    if array_module is None:
        import cupy as array_module
    cfg, _members, _needed_shared = _native_bootstrap_requirements(batch)
    from woof.ingest.wrfinput import restore_domain_state, initialize_wrfinput_physics
    from woof.core.radiation_composition import make_radiation
    pool = array_module.get_default_memory_pool()
    used_before = pool.used_bytes()
    bootstrap_state = restore_domain_state(restored, cfg)
    radiation = make_radiation(cfg, start_time, restored.raw["XLAT"], restored.raw["XLONG"],
                               p_top=bootstrap_state.p_top, column_chunk=column_chunk,
                               trace_gas_overrides=trace_gas_overrides)
    # The RRTMGP chunk is not the column count of engines that size their
    # own workers (CAM: priced by preflight at its own DEFAULT_WORKERS).
    if column_chunk is not None and not getattr(radiation, "sizes_own_columns", False):
        radiation.column_chunk = _positive(column_chunk, "radiation column chunk")
    bootstrap = initialize_wrfinput_physics(bootstrap_state, restored, cfg,
        radiation=radiation, radiation_start_time=start_time,
        radiation_latitude=restored.raw["XLAT"], radiation_longitude=restored.raw["XLONG"],
        landuse=landuse, constant_glw_wm2=constant_glw_wm2,
        fractional_seaice=fractional_seaice, cam_ozone=cam_ozone,
        hydro_parent_geometry=hydro_parent_geometry)
    if bootstrap is None:
        # The single-run door now runs every-scheme-off; the member banks
        # are built from a driver, so the ensemble keeps the refusal
        # initialize_physics used to raise here.
        raise ValueError("ensemble member physics requires at least one enabled "
                         "physics scheme")
    array_module.cuda.get_current_stream().synchronize()
    bootstrap_used = pool.used_bytes() - used_before
    result = initialize_member_physics_from_bootstrap(batch, bootstrap_state, bootstrap,
        available_bytes=available_bytes, bootstrap_pool_live_increment_bytes=bootstrap_used,
        initialization_source="one ordinary WRF-input bootstrap before member perturbations",
        array_module=array_module)
    del bootstrap, bootstrap_state
    array_module.cuda.get_current_stream().synchronize()
    result.receipt["pool_live_after_initialization_bytes"] = pool.used_bytes()
    return result


def initialize_member_physics_from_bootstrap(batch, bootstrap_state, bootstrap_driver, *,
                                           available_bytes, bootstrap_pool_live_increment_bytes=None,
                                           initialization_source="one ordinary prepared bootstrap before member perturbations",
                                           array_module=None, member_bootstraps=None,
                                           member_owned_surface=()):
    """Pack an ordinary initialized driver using the qualified native suite.

    The bootstrap must come from the same common source as the admitted
    batch, before perturbations or forecast stepping. The source's original
    initializer is the authority for its land and physics words. Its
    existing radiation tables and geometry are reused without another load.

    ``member_bootstraps`` gives each member slot its own initialized words:
    one entry per member, either None (that slot takes the live bootstrap
    driver's words) or a :class:`PhysicsBootstrapSnapshot` taken from that
    member's own ordinary bootstrap. Every snapshot must have the same
    inventory structure as the live driver; the shared surface fields and
    the owners taken from the root (radiation, land tables) must match it
    byte for byte, which the caller establishes before allocating here.
    Without it every slot is a copy of the live bootstrap.

    On success the bootstrap state/driver cycle is detached. The caller must
    release its construction handles before measuring retained residency.
    They are not retained by the returned adapter. No external object is
    cleared. A supplied pool increment records the caller's measurement;
    None means that construction increment was not measured.
    """
    if array_module is None:
        import cupy as array_module
    cfg, members, needed_shared = _native_bootstrap_requirements(batch)
    if member_bootstraps is not None:
        member_bootstraps = tuple(member_bootstraps)
        if len(member_bootstraps) != members:
            raise ValueError("member physics bootstraps must name one entry per admitted member slot")
        if any(item is not None and not isinstance(item, PhysicsBootstrapSnapshot) for item in member_bootstraps):
            raise TypeError("member physics bootstraps must be PhysicsBootstrapSnapshot or None")
    from woof.ensemble.batch_physics import (
        ColumnField, ColumnBuffers, PackedThompsonState, PreparedPackedPhysicsAtmosphere,
        PreparedPhysicsCoupling, bind_owned_default_column_driver,
        prepare_thompson_column_batch, prepare_registered_column_batch,
    )
    from woof.core.radiation_composition import legacy_radiation_adapter
    from woof.core.physics import PhysicsDriver
    if not isinstance(bootstrap_driver, PhysicsDriver):
        raise TypeError("native member bootstrap requires an ordinary initialized PhysicsDriver")
    if bootstrap_driver.state is not bootstrap_state or bootstrap_state.physics is not bootstrap_driver:
        raise ValueError("ordinary bootstrap state and physics driver must own each other")
    if bootstrap_pool_live_increment_bytes is not None:
        if isinstance(bootstrap_pool_live_increment_bytes, (bool, np.bool_)):
            raise TypeError("bootstrap pool increment must be a byte count")
        bootstrap_pool_live_increment_bytes = index(bootstrap_pool_live_increment_bytes)
        if bootstrap_pool_live_increment_bytes < 0:
            raise ValueError("bootstrap pool increment cannot be negative")
    bootstrap = bootstrap_driver
    radiation = bootstrap.radiation_callable
    pool = array_module.get_default_memory_pool()

    # Collect each unique mutable allocation and preserve its aliases.
    alias_paths = {}
    entries, pointers, surface_shared = _bootstrap_array_inventory(cfg, bootstrap, array_module.ndarray,
                                                                   alias_paths=alias_paths,
                                                                   member_owned_surface=member_owned_surface)
    shared_names = _BOOTSTRAP_SHARED_OWNERS
    if member_bootstraps is not None:
        layout = tuple((path, tuple(int(n) for n in value.shape), value.dtype.str)
                       for _name, value, path, _ownership in entries)
        for slot, snapshot in enumerate(member_bootstraps):
            if snapshot is not None and snapshot.layout != layout:
                raise ValueError(f"member slot {slot} physics bootstrap inventory differs from the live driver; "
                                 "one packed launch cannot bind both")
    canonical_path = {name: path for name, _value, path, _ownership in entries}

    def slot_words(slot, source, path):
        """The device words member ``slot`` takes for the live driver's ``path``."""
        snapshot = None if member_bootstraps is None else member_bootstraps[slot]
        if snapshot is None:
            return source
        return array_module.asarray(snapshot.arrays[canonical_path[alias_paths[path]]])
    # Canonical precipitation aliases point at the future packed scratch.
    plans = _bootstrap_allocation_declarations(cfg, batch.storage.specs, entries)
    scratch_fields, model_names, declarations = plans.scratch_fields, plans.model_names, plans.column_fields
    bank_plan, shared_plan, status_plan = plans.bank_plan, plans.shared_plan, plans.status_plan
    shared_bytes = shared_plan.required_bytes(members)
    if bank_plan.required_bytes(members) + shared_bytes > available_bytes:
        raise MemoryError("packed model, land and persistent physics banks exceed admitted memory")
    bank = ColumnBuffers(declarations, members=members, available_bytes=available_bytes, array_module=array_module)
    shared_bank = BatchStorage(shared_plan, members, array_module=array_module,
                               available_bytes=available_bytes - bank_plan.required_bytes(members))
    for name in model_names:
        bank.pack(name, batch.storage.arrays[name])
    for name, source, path, ownership in entries:
        # Bootstrap replication is a GPU word copy. All copies are private
        # state, not additional static preparations or independent drivers.
        # With member bootstraps each slot receives its own member's words.
        if ownership == "shared":
            array_module.copyto(shared_bank.arrays[f"physics:{name}"], source)
            continue
        view = bank.packed(name)
        if source.ndim == 2:
            target = view.reshape(members, *source.shape)
            if member_bootstraps is None:
                array_module.copyto(target, source[None])
            else:
                for slot in range(members):
                    target[slot] = slot_words(slot, source, path)
        else:
            target = view.reshape(source.shape[0], members, *source.shape[1:])
            if member_bootstraps is None:
                array_module.copyto(target, source[:, None])
            else:
                for slot in range(members):
                    target[:, slot] = slot_words(slot, source, path)

    def resolve_array(value):
        key = (int(value.data.ptr), int(value.nbytes), value.dtype.str)
        name = pointers[key]
        if f"physics:{name}" in shared_bank.arrays:
            return shared_bank.arrays[f"physics:{name}"]
        return bank.packed(name)

    arrays = {name: bank.packed(name) for name in model_names}
    arrays.update({name: batch.storage.arrays[name] for name in needed_shared})
    arrays.update(p_top=bootstrap_state.p_top, elapsed_seconds=float(batch.elapsed_seconds),
                  has_msf=bool(batch.has_msf), use_theta_m=int(getattr(cfg, "use_theta_m", 0)))
    # The physics validator uses a compact one-word control buffer; it has
    # no member arithmetic and is explicitly counted outside column fields.
    status = BatchStorage(status_plan, members, array_module=array_module,
                          available_bytes=available_bytes - bank_plan.required_bytes(members) - shared_bytes)
    scratch = {field.name: bank.packed(field.name) for field in scratch_fields}
    if cfg.cu_physics == 99:
        # These are the ordinary driver's canonical restart/health slots.
        # They have already been counted through the driver's array aliases;
        # bind them to that same packed backing rather than a second copy.
        for slot, value in bootstrap_state._scratch.items():
            if slot.startswith("cu_"):
                scratch[slot] = resolve_array(value)
    scratch["physics_validation_status"] = status.arrays["physics:validation_status"]
    state = PackedThompsonState(arrays, scratch)
    state.qh = None
    if cfg.mp_physics == 55:
        state.qs = state.qg = None
    driver = PhysicsDriver.__new__(PhysicsDriver)
    for name, value in bootstrap.__dict__.items():
        setattr(driver, name, value if name in shared_names else _clone_tree(value, array_module.ndarray, resolve_array))
    driver.state, state.physics = state, driver
    if cfg.cu_physics == 99:
        from woof.ensemble.batch_kf import KF_OLD_CARRIERS
        adapter = copy(bootstrap.cumulus_callable)
        for name in KF_OLD_CARRIERS:
            value = getattr(adapter, name)
            setattr(adapter, name, None if value is None else resolve_array(value))
        adapter._history_state = state
        adapter._driver_ref = None
        adapter.bind_driver(driver)
        driver.cumulus_callable = adapter
    # The common initializer's post-RK diagnostic contract aliases exactly
    # the admitted microphysics accumulators, not a duplicated bootstrap set.
    from woof.core.microphysics import MicrophysicsDiagnostics
    driver.microphysics = MicrophysicsDiagnostics(
        **{name: scratch.get(slot) for name, slot in (
            ("rainnc", "mp_rainnc"), ("rainncv", "mp_rainncv"), ("sr", "mp_sr"),
            ("snownc", "mp_snownc"), ("snowncv", "mp_snowncv"),
            ("graupelnc", "mp_graupelnc"), ("graupelncv", "mp_graupelncv"))})
    for name, slot in (("rainnc", "mp_rainnc"), ("snownc", "mp_snownc"), ("graupelnc", "mp_graupelnc")):
        original_value = getattr(bootstrap.microphysics, name)
        if original_value is None:
            continue
        target = scratch[slot].reshape(members, cfg.ny, cfg.nx)
        if member_bootstraps is None:
            array_module.copyto(target, original_value[None])
        else:
            for member in range(members):
                target[member] = slot_words(member, original_value, f"driver/microphysics/{name}")
    # Restore each scheme's persistent diagnostic and surface owner into
    # the packed level/member layout used by output and checkpoint views.
    from woof.ensemble.batch_physics import THOMPSON_CLOUD_BOOKING_SLOTS
    slots = THOMPSON_CLOUD_BOOKING_SLOTS
    if cfg.mp_physics == 55:
        from woof.physics_registry import consumer_row_for_selector
        native = consumer_row_for_selector("microphysics", "native_column", 55)
        slots += tuple(native["surfaces"].values()) + tuple(native["diagnostics"].values())
    for slot in slots:
        restored_words = batch.storage.arrays.get("scratch:" + slot)
        if slot in scratch and restored_words is not None:
            field = bank.fields[slot]
            bank.pack(slot, restored_words.reshape((members,) + field.shape))
    driver.tendencies = driver.pbl_tendencies if bootstrap.tendencies is bootstrap.pbl_tendencies else driver.tendencies
    driver.sfclay_result = type(bootstrap.sfclay_result)(**{field.name: driver.fields[field.name]
        for field in dataclass_fields(bootstrap.sfclay_result)})
    driver._ensemble_shared_surface_fields = tuple(sorted(surface_shared))
    used_bank = bank_plan.required_bytes(members) + status_plan.required_bytes(members) + shared_bytes
    budget_left = available_bytes - used_bank
    atmosphere = PreparedPackedPhysicsAtmosphere(state, members=members, ny=cfg.ny, nx=cfg.nx,
        c1h=state.c1h, c2h=state.c2h, dnw=state.dnw, p_top=state.p_top,
        mup=state.mup, mub2d=state.mub2d, available_bytes=budget_left, array_module=array_module)
    budget_left -= atmosphere.columns.storage.plan.required_bytes(members)
    state.total_mu = lambda: atmosphere.packed_mut
    atmosphere()
    common = dict(members=members, nz=cfg.nz, ny=cfg.ny, nx=cfg.nx,
        c1h=state.c1h, c2h=state.c2h, mut=atmosphere.packed_mut,
        msft=state.msft, msfu=state.msfu, msfv=state.msfv, has_msf=state.has_msf, array_module=array_module)
    pbl = PreparedPhysicsCoupling(**common, available_bytes=budget_left)
    budget_left -= pbl.plan.required_bytes(members)
    rad = PreparedPhysicsCoupling(**common, available_bytes=budget_left)
    budget_left -= rad.plan.required_bytes(members)
    cu = None
    if cfg.cu_physics == 99:
        cu = PreparedPhysicsCoupling(**common, available_bytes=budget_left)
        budget_left -= cu.plan.required_bytes(members)
        from woof.ensemble.batch_kf import kf_old_driver_reservation
        budget_left -= kf_old_driver_reservation(cfg.nz, cfg.ny, cfg.nx,
            members=members, edrates=cfg.kf_edrates,
            multiprocessors=array_module.cuda.Device().attributes["MultiProcessorCount"])
        if budget_left < 0:
            raise MemoryError("packed CU99 due-call workspace exceeds admitted memory")
    native_radiation = prepare_legacy_member_radiation(legacy_radiation_adapter(radiation), members=members,
                                                       available_bytes=budget_left, array_module=array_module,
                                                       shared_surface_fields=("xland",))
    driver.radiation_callable = native_radiation
    budget_left -= native_radiation._ensemble_radiation_owners[2].plan.required_bytes(members)
    bind_owned_default_column_driver(driver, cfg, atmosphere, members=members, ny=cfg.ny, nx=cfg.nx,
        pbl_coupling=pbl, radiation_coupling=rad, cumulus_coupling=cu,
        available_bytes=budget_left, array_module=array_module)
    budget_left -= driver._ensemble_native_column_receipt["required_binding_bytes"]
    prepare_microphysics = prepare_registered_column_batch if cfg.mp_physics == 55 else prepare_thompson_column_batch
    microphysics = prepare_microphysics(state, cfg, members=members, ny=cfg.ny, nx=cfg.nx, dt=cfg.dt,
                                       available_bytes=budget_left, array_module=array_module)
    receipt = {"members": members, "model_member_fields": list(model_names),
        "initialization_source": initialization_source,
        "bootstrap_pool_live_increment_bytes": bootstrap_pool_live_increment_bytes,
        "bootstrap_handle_policy": "caller releases construction handles after native binding",
        "packed_bank_allocations": list(bank_plan.inventory(members)),
        "shared_surface_allocations": list(shared_plan.inventory(members)),
        "shared_surface_fields": sorted(surface_shared),
        "bootstrap_array_paths": {name: path for name, _, path, _ in entries},
        "member_bootstraps": ("live bootstrap replicated into every slot" if member_bootstraps is None else
            ["live bootstrap" if item is None else f"own bootstrap snapshot, {item.nbytes} host bytes"
             for item in member_bootstraps]),
        "member_owned_surface_fields": sorted(member_owned_surface),
        "scalar_member_drivers_retained": 0, "native_driver": driver._ensemble_native_column_receipt,
        "radiation": native_radiation._ensemble_radiation_receipt}
    # Retain only the native radiation/table owner; the scalar state's
    # forecast/land arrays are released before the first clock advance.
    bootstrap_state.physics = None
    bootstrap.state = None
    source = value = original_value = None
    # resolve_array closes over `pointers`, so it is released by rebinding:
    # a deleted closure cell is a NameError for any later call, and the
    # undefined-name gate cannot tell that none follows.
    pointers = None
    del bootstrap, bootstrap_driver, bootstrap_state, entries
    array_module.cuda.get_current_stream().synchronize()
    receipt["pool_live_after_initialization_bytes"] = pool.used_bytes()
    result = InitializedMemberPhysics(batch, driver, state, bank, atmosphere, microphysics, receipt)
    result.shared_surface_storage = shared_bank
    result.validation_storage = status
    return result
