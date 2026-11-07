"""Shared source preparation for original config-driven member forecasts.

Only source and geometry work is reused. Clocks, prognostics, physics drivers,
parent adjustments, tile stores and checkpoint restoration remain member owned.
The ordinary runner with no scope follows its original preparation functions.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from types import MappingProxyType

import numpy as np


_SOURCE = ContextVar("gpuwm_runtime_preparation_source", default=None)
_MEMBER_INPUT = ContextVar("gpuwm_runtime_preparation_member", default=None)
_ROOT_BUILD = ContextVar("gpuwm_runtime_preparation_root_build", default=None)


def current_runtime_preparation():
    return _SOURCE.get()


def root_preparation_is_building():
    return _ROOT_BUILD.get() is not None


def runtime_input_catalog(data):
    """The original catalog builder, memoized only in an ensemble scope."""
    from woof.ingest.preflight import build_input_catalog
    source = current_runtime_preparation()
    return build_input_catalog(data) if source is None else source.catalog(data)


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=lambda item:
                      json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": value.dtype.str,
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, np.generic):
        return {"dtype": value.dtype.str, "words": value.tobytes().hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Configuration carriers used by CPU orchestration tests and adapters.
    if isinstance(value, SimpleNamespace):
        return _plain(vars(value))
    raise TypeError(f"source preparation identity cannot bind {type(value).__name__}")


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _preparation_experiment_identity(exp):
    """The existing root identity without forecast-time parameter sets.

    Parameter values change physics initialization and integration, while
    cold source preparation retains its original fields. Removing this one
    field at either None or an active set preserves the former default key
    and lets parameter members restore the same prepared source words.
    Member trajectories and checkpoint identities still bind the set.
    """
    document = _plain(exp)
    document.pop("physics_params", None)
    return document


def _grid_identity(grid):
    names = ("map_proj", "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon",
             "known_x", "known_y", "dx", "dy", "e_we", "e_sn")
    identity = {name: getattr(grid, name) for name in names if hasattr(grid, name)}
    identity["grid_type"] = type(grid).__module__ + "." + type(grid).__qualname__
    for name in ("latlon_mass", "latlon_u", "latlon_v", "mapfac_m", "mapfac_u", "mapfac_v"):
        getter = getattr(grid, name, None)
        if callable(getter):
            identity[name] = _plain(getter())
    return identity


def _device_id():
    # This is asked only after an original GPU preparation has been selected.
    import cupy as cp
    return int(cp.cuda.runtime.getDevice())


def _binding_identity(member):
    if member is None:
        return None
    manifests = tuple(binding.verify() for binding in member.source_manifests)
    donors = tuple(binding.verify() for binding in member.donor_manifests)
    return {"trajectory_sha256": member.trajectory.identity,
            "member_id": member.member_id, "seed": member.seed,
            "recipe_sha256": member.recipe_sha256,
            "geometry_sha256": member.geometry_sha256,
            "boundary_valid_times": [time.isoformat() for time in member.boundary_valid_times],
            "source_manifests": manifests, "donor_manifests": donors,
            "prepared_authority": getattr(member.inputs, "authority_sha256", None),
            "prepared_source_identity": getattr(member.inputs, "source_identity", None),
            "preparation": member.preparation_receipt}


def _preparation_binding(binding):
    """Labels select stochastic owners after deterministic preparation.

    Keep the actual source, donor, recipe and prepared-field authorities in
    every cache key. Two labels on those same inputs may share cold words,
    while their independent clocks, physics initialization and RNG still run.
    """
    if binding is None:
        return None
    return {name: value for name, value in binding.items()
            if name not in {"member_id", "seed"}}


def _host_horizontal(snapshot):
    """Detach a typed source mapping with the existing backend transfer."""
    if snapshot is None:
        return None
    from woof.ingest.preprocess_backend import _host
    mapped = {name: _host(value) for name, value in snapshot.fields.items()}
    for value in mapped.values():
        value.flags.writeable = False
    if is_dataclass(snapshot):
        return replace(snapshot, fields=MappingProxyType(mapped))
    return SimpleNamespace(**{**vars(snapshot), "fields": MappingProxyType(mapped)})


def _retained_array_bytes(values):
    """Unique retained host payload spans; no CUDA imports or allocations."""
    arrays, seen = [], set()
    def walk(value):
        if isinstance(value, np.ndarray):
            arrays.append(value)
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                walk(getattr(value, field.name))
        elif isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
        elif isinstance(value, SimpleNamespace):
            for item in vars(value).values():
                walk(item)
    walk(values)
    spans = []
    for array in arrays:
        if not array.size:
            continue
        pointer = int(array.__array_interface__["data"][0])
        low = high = 0
        for extent, stride in zip(array.shape, array.strides):
            delta = (extent - 1) * stride
            low += min(0, delta)
            high += max(0, delta)
        spans.append((pointer + low, pointer + high + array.itemsize))
    total, end = 0, -1
    for begin, stop in sorted(spans):
        total += max(0, stop - max(begin, end))
        end = max(end, stop)
    return total


@dataclass(frozen=True)
class RuntimeMemberInputs:
    """One config-driven source selection handed to the ensemble session."""
    experiment: object
    case_data: object
    member_input: object = None

    @property
    def boundary_interval_seconds(self):
        return getattr(self.case_data, "forcing_interval_s", None)


@dataclass
class _RootCapture:
    key: str
    path: Path
    inputs: object = None


class RuntimePreparationSource:
    """Run-owned immutable prepared inputs, restored by original constructors.

    The root cache is the existing prepared-cache format. No arbitrary state
    or physics object is copied. Parent-independent child input mapping is
    retained as typed host words; every parent-dependent operation still runs.
    """
    CONTRACT = "gpuwm-ensemble-runtime-preparation.v1"

    def __init__(self, cache_root):
        self.root = Path(cache_root).resolve()
        if self.root.exists():
            raise FileExistsError("shared runtime preparation requires a new owned cache directory")
        self.root.mkdir(parents=True)
        self._lock = threading.RLock()
        self._catalogs, self._forcing, self._statics = {}, {}, {}
        self._roots, self._children = {}, {}
        self._root_bindings = {}
        self._created_files, self._deleted_files = {}, []
        self._closed = False
        self._counts = {"catalog_builds": 0, "forcing_decodes": 0, "static_builds": 0,
                        "root_preparations": 0, "root_restores": 0,
                        "child_input_preparations": 0, "child_input_reuses": 0}

    @contextmanager
    def scope(self, member_input=None):
        if self._closed:
            raise RuntimeError("shared preparation cache is closed")
        source = _SOURCE.set(self)
        member = _MEMBER_INPUT.set(member_input)
        try:
            yield self
        finally:
            _MEMBER_INPUT.reset(member)
            _SOURCE.reset(source)

    def catalog(self, data):
        from woof.ingest.preflight import build_input_catalog
        binding = _binding_identity(_MEMBER_INPUT.get())
        key = _digest({"case_data": data, "member_source": _preparation_binding(binding)})
        with self._lock:
            if key not in self._catalogs:
                self._catalogs[key] = build_input_catalog(data)
                self._counts["catalog_builds"] += 1
            return self._catalogs[key]

    def forcing_snapshots(self, data, catalog, *, build):
        binding = _binding_identity(_MEMBER_INPUT.get())
        key = _digest({"case_data": data, "catalog": catalog.fingerprint,
                       "member_source": _preparation_binding(binding)})
        with self._lock:
            if key not in self._forcing:
                self._forcing[key] = build()
                self._counts["forcing_decodes"] += 1
            return self._forcing[key]

    def static_fields(self, grid, geog_root, *, selection, static_highres,
                      domain_id, case_date, build):
        key = _digest({"grid": _grid_identity(grid), "geog_root": Path(geog_root),
                       "selection": selection, "static_highres": static_highres,
                       "domain_id": domain_id, "case_date": str(case_date)})
        with self._lock:
            if key not in self._statics:
                fields = dict(build())
                for value in fields.values():
                    if isinstance(value, np.ndarray):
                        value.flags.writeable = False
                self._statics[key] = (fields, _digest(fields))
                self._counts["static_builds"] += 1
            # Root terrain blending replaces a dictionary entry. Each member
            # gets its own mapping while all unchanged numeric arrays are shared.
            return dict(self._statics[key][0])

    def capture_root_inputs(self, **inputs):
        """Seal cold initialized source words before ordinary physics runs."""
        capture = _ROOT_BUILD.get()
        if capture is None or capture.inputs is not None:
            return
        from woof.ingest.case_store import CaseStoreRequest, write_case_store_input
        capture.inputs = write_case_store_input(CaseStoreRequest(capture.path,
            backend=inputs.pop("preprocess_backend")), **inputs)
        for path in capture.path.rglob("*"):
            if path.is_file():
                self._created_files[path] = path.stat().st_size

    def prepare_root(self, exp, data, *, grid, selection, catalog,
                     scratch_arena, dycore_state_workspace, store_request, build):
        """Restore a matching root, or run the unchanged first preparation."""
        from woof import runtime
        static = runtime.case_static_fields(grid, data.geog_root, selection=selection,
            static_highres=getattr(data, "static_highres", None), domain_id=exp.root.grid_id,
            case_date=exp.start_time.date())
        binding = _binding_identity(_MEMBER_INPUT.get())
        projection = _grid_identity(grid)
        static_sha256 = _digest(static)
        key = _digest({"experiment": _preparation_experiment_identity(exp), "case_data": data,
            "catalog": catalog.fingerprint, "projection": projection,
            "static": static_sha256, "member_source": _preparation_binding(binding),
            "host_initialization": store_request is not None,
            "preparation_device": _device_id()})
        with self._lock:
            saved = self._roots.get(key)
            if saved is not None:
                members = self._root_bindings[key]["member_sources"]
                if binding not in members:
                    members.append(binding)
                self._counts["root_restores"] += 1
                return self._restore_root(saved, scratch_arena=scratch_arena,
                    dycore_state_workspace=dycore_state_workspace,
                    host_initialization=store_request is not None)
            capture = _RootCapture(key, self.root / ("root-" + key[:24]))
            token = _ROOT_BUILD.set(capture)
            try:
                prepared = build()
            finally:
                _ROOT_BUILD.reset(token)
            self._counts["root_preparations"] += 1
            if prepared.store_input is not None:
                # The original host-store cache is transaction-owned and may
                # be removed after its first load. Link its verified immutable
                # payload into this run-owned cache before that context ends.
                from woof.filesystem_paths import copy_verified
                capture.path.mkdir()
                for path in prepared.store_input.path.iterdir():
                    if path.is_file():
                        target = capture.path / path.name
                        copy_verified(path, target)
                        self._created_files[target] = target.stat().st_size
                capture.inputs = replace(prepared.store_input, path=capture.path)
            if capture.inputs is None:
                raise RuntimeError("original root preparation did not publish its cold source cache")
            original = prepared.initial_result
            template_initial = (replace(original, state=None) if is_dataclass(original)
                                else SimpleNamespace(**{name: value for name, value in vars(original).items()
                                                        if name != "state"}))
            template = replace(prepared, initial_result=template_initial, store_input=capture.inputs,
                               streamed_store=None, final_analysis=_host_horizontal(prepared.final_analysis))
            self._roots[key] = template
            self._root_bindings[key] = {"catalog_sha256": catalog.fingerprint,
                "projection_sha256": _digest(projection), "static_sha256": static_sha256,
                "member_source": binding,
                "member_sources": [binding],
                "boundary_valid_times": [time.isoformat() for time in template.forcing_times]}
            return prepared

    def _restore_root(self, template, *, scratch_arena, dycore_state_workspace, host_initialization):
        if host_initialization:
            return template
        from woof.ingest.prepared_cache import restore_prepared_cache
        from woof.native_wrf_contract import native_static_export_fields
        from woof import runtime
        inputs, cfg = template.store_input, template.cfg
        restored = restore_prepared_cache(inputs.path, expected_identity=inputs.identity, cfg=cfg,
            static=native_static_export_fields(template.static_fields, template.grid),
            scratch_arena=scratch_arena, dycore_state_workspace=dycore_state_workspace)
        surface, context = restored.surface.fields, inputs.physics
        soil = SimpleNamespace(**{name: surface[key] for name, key in {
            "tsk": "TSK", "soil_temperature": "TSLB", "soil_moisture": "SMOIS",
            "liquid_moisture": "SH2O", "deep_soil_temperature": "TMN", "xice": "SEAICE",
            "snow_water": "SNOW", "snow_depth": "SNOWH"}.items()})
        original = template.initial_result
        changes = {"state": restored.initial_result.state, "coord": restored.initial_result.coord,
                   "base": restored.initial_result.base,
                   "surface_pressure": restored.initial_result.surface_pressure,
                   "surface_qv": restored.initial_result.surface_qv}
        initial = (replace(original, **changes) if is_dataclass(original)
                   else SimpleNamespace(**{**vars(original), **changes}))
        runtime._initialize_real_case_physics(initial, cfg, restored.met, soil,
            {"SST": context.sst}, template.static_fields, inputs.landuse_attrs,
            template.grid, template.forcing_times[0], vertical=context.vertical,
            reconciled_soil_type=context.reconciled_soil_type,
            trace_gas_overrides=context.trace_gas_overrides,
            radiation_column_chunk=context.radiation_column_chunk,
            constant_glw_wm2=inputs.constant_glw_wm2,
            **({"cam_ozone": context.cam_ozone} if context.cam_ozone is not None else {}))
        return replace(template, initial_result=initial, store_input=None)

    def prepare_child_input(self, domain, grid, catalog, source_orography, *,
                            preprocess_backend, preprocess_workers, cpu_bridge, build):
        # This typed input explicitly excludes the parent's evolving state.
        # finalize_prepared_child remains the original dependency barrier.
        backend = preprocess_backend if isinstance(preprocess_backend, str) else preprocess_backend.name
        binding = _binding_identity(_MEMBER_INPUT.get())
        key = _digest({"domain": domain, "projection": _grid_identity(grid),
                       "catalog": catalog.fingerprint, "source_orography": source_orography,
                       "backend": backend, "workers": preprocess_workers,
                       "cpu_bridge": cpu_bridge, "device": _device_id(),
                       "member_source": _preparation_binding(binding)})
        with self._lock:
            if key not in self._children:
                original = build()
                self._children[key] = replace(original, horizontal=_host_horizontal(original.horizontal))
                self._counts["child_input_preparations"] += 1
                return original
            else:
                self._counts["child_input_reuses"] += 1
            # A new typed domain owner retains the exact current DomainConfig.
            return replace(self._children[key], domain=domain)

    def receipt(self):
        with self._lock:
            return {"contract": self.CONTRACT, "counts": dict(self._counts),
                    "root_inputs": [{"identity_sha256": key,
                        "binding": self._root_bindings[key],
                        "preparation": dict(case.store_input.receipt)} for key, case in self._roots.items()],
                    "static_identities": [row[1] for row in self._statics.values()],
                    "host_retained_payload_bytes": _retained_array_bytes((self._catalogs, self._forcing,
                        self._statics, self._roots, self._children)),
                    "device_retained_payload_bytes": 0,
                    "device_policy": "typed cold roots and mapped source arrays are retained on the host; original preprocessing selection remains device bound",
                    "execution_policy": "original member clocks, physics, parent adjustments, stores and restart restoration",
                    "created_files": [{"path": str(path), "bytes": size}
                                      for path, size in self._created_files.items()],
                    "deleted_files": list(self._deleted_files), "closed": self._closed}

    def close(self):
        """Release retained input owners and remove only this source's files."""
        with self._lock:
            if self._closed:
                return
            self._catalogs.clear()
            self._forcing.clear()
            self._statics.clear()
            self._roots.clear()
            self._children.clear()
            for path, size in self._created_files.items():
                if path.is_file():
                    path.unlink()
                    self._deleted_files.append({"path": str(path), "bytes": size})
            # Only directories created below our new, exclusive root belong
            # to this source. Never recursively remove an existing caller path.
            for path in sorted(self.root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if path.is_dir():
                    path.rmdir()
            self.root.rmdir()
            self._closed = True


__all__ = ["RuntimeMemberInputs", "RuntimePreparationSource", "current_runtime_preparation",
           "root_preparation_is_building", "runtime_input_catalog"]
