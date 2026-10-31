"""Common CUDA/parallel-CPU contract for native preprocessing.

The forecast model itself remains CUDA-only.  This module selects the
backend used by the source-grid and WRF-real setup transforms that precede
the model allocation.  Both implementations consume and emit FP32 arrays;
the CPU implementation delegates interpolation arithmetic to the packaged
Rust bridge and uses deterministic NumPy elementwise helpers for the small
vector and humidity transforms.

The masked surface fields (soil moisture and temperature, snow, skin
temperature, sea ice) take WPS metgrid's masked chain in float64 in the
same Rust library under BOTH backends, parallel across target cells: the
CPU backend on its own library and CPU/memory worker budget, the CUDA
backend on the library the resolution
ladder picks and its host workers, every CPU the process may use when
none were given (:meth:`CudaPreprocessBackend.wps_masked_chain_engine`).  Each receipt
names that library as ``masked_surface_chain``.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field
from functools import wraps
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock
from types import MappingProxyType

import numpy as np

from woof.ingest.backend_contract import _sha256

from woof.ingest.memory_refusal import InitializationMemoryRefused
from woof.ingest.cpu_backend import (
    CPU_BACKEND_ABI,
    MASKED_STENCIL_ENTRY,
    MASKED_STENCIL_IMPLEMENTATION,
    WPS_MASKED_CHAIN_ENTRY,
    WPS_MASKED_CHAIN_IMPLEMENTATION,
    CpuPreprocessBackend,
    automatic_workers,
    available_cpu_count,
    masked_fields_cpu_backend,
    shared_cpu_backend,
)


PREPROCESS_IMPLEMENTATION_SCHEMA = "gpuwm-preprocess-implementation-v2"
PSFC_MAPPING_POLICY = "canonical-f32-coordinate-f64-bilinear-single-round-v1"
VERTICAL_STENCIL_POLICY = "wrf-v4.6.1-strict-fp32-zap-close-levels-v1"
#: v2 widened the co-location bound from 2^-21 (four FP32 epsilons,
#: rounding only) to 2^-16, which also holds the vapour weight a native
#: full-pressure top level carries over its dry target.
VERTICAL_ENDPOINT_POLICY = "native-pressure-top-colocation-relative-2pow-16-v2"
#: The bound itself, shared with woof.ingest.vert and the Rust/CUDA
#: operators.
VERTICAL_ENDPOINT_RELATIVE_TOLERANCE = 2.0 ** -16


@contextmanager
def preprocessing_math_scope(backend, *, cpu_bridge=None, workers=None):
    """Bind host setup math to an actual selected CPU preparation library."""
    from woof.core import portable_math
    selected = None
    publish = False
    if isinstance(backend, ParallelCpuPreprocessBackend):
        selected = backend._native.path
        publish = getattr(backend, "_explicit_math_selection", False)
        if workers is None:
            workers = getattr(backend, "host_step_workers", None)
    elif isinstance(backend, str) and backend.strip().lower() == "cpu":
        selected = cpu_bridge
        publish = selected is not None
    cap = None
    if selected is not None:
        # The preparation package's own CPU budget (affinity and every cgroup
        # quota).  woof.live_products answers the same question for live
        # frames, but the standalone preparation wheel does not carry it.
        cap = min(available_cpu_count(), automatic_workers() if workers is None else portable_math._positive_workers(workers))
    with portable_math.cpu_bridge_scope(selected, publish_selection=publish, worker_cap=cap):
        yield


def preprocess_math_call(function=None, *, prepared_parameter=None, options_parameter=None,
                         fixed_backend=None):
    """Keep a preparation's selected library across its original host setup."""
    if function is None:
        return lambda original: preprocess_math_call(original, prepared_parameter=prepared_parameter,
            options_parameter=options_parameter, fixed_backend=fixed_backend)
    signature = inspect.signature(function)
    @wraps(function)
    def selected(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bridge = bound.arguments.get("cpu_bridge", bound.arguments.get("cpu_preprocess_bridge"))
        workers = bound.arguments.get("column_workers")
        if workers is None:
            workers = bound.arguments.get("preprocess_workers")
        if options_parameter is not None:
            options = bound.arguments[options_parameter]
            backend = getattr(options, "preprocess_backend", None)
            bridge = getattr(options, "cpu_preprocess_bridge", None)
            workers = getattr(options, "preprocess_workers", None)
        elif prepared_parameter is not None:
            backend = bound.arguments[prepared_parameter].preprocess_backend
        else:
            parameter = signature.parameters.get("preprocess_backend")
            backend = bound.arguments.get("preprocess_backend", fixed_backend if parameter is None else parameter.default)
        with preprocessing_math_scope(backend, cpu_bridge=bridge, workers=workers):
            return function(*args, **kwargs)
    return selected

_COMMON_IMPLEMENTATION_SOURCES = (
    # Child preparation changes when its capture or compile contract changes.
    # Never restore a pre-fix prepared bundle under the new SINT contract.
    "woof/core/nest_interp.py",
    "woof/core/diagnostics.py",
    "woof/ingest/nest_init.py",
    "woof/numerics.py",
    "woof/core/host_libm.py",
    "woof/core/noahmp_libm.py",
    "woof/core/thompson_entry.py",
    "woof/core/state.py",
    "woof/ingest/cold_start_cpu.py",
    "woof/ingest/host_arrays.py",
    "woof/ingest/preparation_fingerprints.py",
    "woof/ingest/horiz.py",
    "woof/ingest/preprocess_backend.py",
    "woof/ingest/interpolation_support.py",
    "woof/ingest/atmospheric_window.py",
    "woof/ingest/real.py",
    "woof/ingest/lake_temperature.py",
    "woof/ingest/water_temperature.py",
    "woof/ingest/vert.py",
)


def _implementation_tree(backend: str) -> dict[str, object]:
    """Bind the trajectory-changing preprocessing source implementation."""

    root = Path(__file__).resolve().parents[2]
    names = list(_COMMON_IMPLEMENTATION_SOURCES)
    if backend == "cpu":
        names.append("woof/ingest/cpu_backend.py")
    elif backend == "cuda":
        names.append("woof/core/kernels/nest.cu")
        names.append("woof/core/kernels/diagnostics.cu")
        # cpu_backend.py rides with the kernel: a vertical column deeper
        # than the kernel's top tier runs on the CPU bridge
        # (woof.ingest.vert.WRF_VERT_INTERP_LEVEL_TIERS).
        names.append("woof/core/kernels/vert_interp.cu")
        names.append("woof/core/kernels/glibc_flt32.cuh")
        names.append("woof/ingest/cpu_backend.py")
        # The card twins whose bits the cuda preparation publishes: the
        # fused horizontal step, initialize_real's column work and the
        # Thompson cold-start closure, with the libm twins they include.
        names.extend(("woof/core/kernels/horizontal.cu",
                      "woof/ingest/bounded_cuda.py",
                      "woof/ingest/real_device.py",
                      "woof/core/kernels/real_init.cu",
                      "woof/core/kernels/real_init_common.cuh",
                      "woof/core/kernels/real_init_math.cu",
                      "woof/ingest/closure_device.py",
                      "woof/core/kernels/thompson_cold_start.cu",
                      "woof/core/kernels/portable_libm64.cuh"))
    else:  # pragma: no cover - internal call sites own the finite inventory
        raise ValueError(f"unsupported provenance backend {backend!r}")
    files = {}
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"preprocessing implementation source is missing: {path}")
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    encoded = json.dumps(
        files, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return {
        "schema": "gpuwm-preprocess-source-tree-v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def _shared_contracts() -> dict[str, object]:
    return {
        "surface_pressure_mapping": {
            "policy": PSFC_MAPPING_POLICY,
            "output_dtype": "float32",
        },
        "vertical_stencil": {
            "policy": VERTICAL_STENCIL_POLICY,
            "zap_close_levels_pa": 500.0,
            "predicate": "separation < zap_close_levels",
            "top_endpoint_policy": VERTICAL_ENDPOINT_POLICY,
            "top_endpoint_relative_tolerance": VERTICAL_ENDPOINT_RELATIVE_TOLERANCE,
        },
    }


def _record_vertical_route(routes: list, route: dict[str, object]) -> None:
    """Add one vertical route to a backend's receipt record, once."""

    if route not in routes:
        routes.append(route)


def _selection_block(backend) -> dict[str, object]:
    """The receipt's ``selection`` entry: requested selector, backend, reason."""

    selection = getattr(backend, "selection", None)
    return {} if selection is None else {"selection": dict(selection)}


#: Members of a preprocessing receipt that record a measurement of the
#: machine or of this run rather than a choice that shapes a prepared
#: array, each with what it measures.  The receipt (the preparation proof)
#: keeps every one; a prepared cache's identity and its hashed metadata
#: bind :func:`preprocess_identity`, the receipt without them.  Bound, they
#: made two preparations of the same inputs differ in content digest by
#: the measurement alone (A138: the card's free bytes split one of three
#: CUDA pairs in the A136 L7a proof), so a cache could not be recognised
#: as the same preparation on another run, box or worker count.
PREPROCESS_RECEIPT_MEASUREMENTS = MappingProxyType({
    "selection.chunking": "device staging budget and byte-neutral kernel batch sizes",
    "selection.host_fit": "host-retained preparation envelope against available RAM",
    "parallelism": "requested and effective native workers and host CPU limits",
    "selection.device_fit": (
        "the preparation's device price against the card's measured "
        "free and total bytes"),
    "selection.device_load": (
        "the card's free bytes and utilization when auto read it"),
    "selection.reason": (
        "why this backend runs, which quotes the measured utilization or "
        "free memory when auto leaves the card"),
    "host_cpu_count": "the machine's CPU count",
    "workers": (
        "the CPU worker count; every array is byte-identical at every "
        "count (tests/test_preprocess_cpu_backend.py)"),
    "masked_surface_chain.workers": (
        "the masked surface fields' worker count; byte-identical at every "
        "count (tests/test_masked_stencil_native.py)"),
})


def preprocess_identity(receipt):
    """The part of a preprocessing receipt a prepared cache binds.

    ``receipt`` without :data:`PREPROCESS_RECEIPT_MEASUREMENTS`.  Members
    not removed are the receipt's own objects, so a vertical route list
    a backend still appends to reads the same here.  Anything that is
    not a mapping is returned unchanged.
    """

    if not isinstance(receipt, Mapping):
        return receipt
    bound = dict(receipt)
    for path in PREPROCESS_RECEIPT_MEASUREMENTS:
        parent, _, leaf = path.rpartition(".")
        if not parent:
            bound.pop(leaf, None)
        elif isinstance(bound.get(parent), Mapping):
            bound[parent] = {key: value for key, value
                             in bound[parent].items() if key != leaf}
    return bound


def preprocess_measurements(receipt) -> dict[str, object]:
    """What :func:`preprocess_identity` takes out of ``receipt``.

    The :data:`PREPROCESS_RECEIPT_MEASUREMENTS` members ``receipt``
    carries, nested as the receipt nests them, for a route whose hashed
    implementation document binds the identity and whose receipt must
    still say what the preparation measured.  Anything that is not a
    mapping measured nothing.
    """

    measured: dict[str, object] = {}
    if not isinstance(receipt, Mapping):
        return measured
    for path in PREPROCESS_RECEIPT_MEASUREMENTS:
        parent, _, leaf = path.rpartition(".")
        holder = receipt.get(parent) if parent else receipt
        if not isinstance(holder, Mapping) or leaf not in holder:
            continue
        if parent:
            measured.setdefault(parent, {})[leaf] = holder[leaf]
        else:
            measured[leaf] = holder[leaf]
    return measured


def preprocess_selection_identity(selection):
    """A receipt's ``selection`` block less what it measured.

    For a route that records the selection apart from the receipt (the
    met_em preparation's ``metgrid-import.json``) and binds that record
    into an identity.
    """

    return preprocess_identity({"selection": selection})["selection"]


def preprocess_reports_identity(document, *, key: str = "preprocess_backend"):
    """``document`` with every ``key`` member (a backend receipt) bound.

    For reports that carry a backend's receipt inside them (the HRRR
    mapping reports, a child's soil mapping) and are written into a
    prepared cache's hashed metadata: each such receipt becomes its
    :func:`preprocess_identity`.
    """

    if isinstance(document, Mapping):
        return {name: (preprocess_identity(value) if name == key
                       else preprocess_reports_identity(value, key=key))
                for name, value in document.items()}
    if isinstance(document, (list, tuple)):
        return [preprocess_reports_identity(value, key=key)
                for value in document]
    return document


def preprocess_identity_matches(bound, receipt) -> bool:
    """Whether a cache's bound preprocessing is this receipt's.

    A cache written since A138 binds :func:`preprocess_identity` of the
    receipt; one written before it bound the whole receipt, measurements
    included, which is still the same preparation exactly, so it keeps
    restoring.  Anything else is a different preparation.
    """

    return bound == preprocess_identity(receipt) or bound == receipt


def _bridge_identity(native) -> dict[str, object]:
    return {
        "name": native.path.name,
        "sha256": _sha256(native.path),
        "abi_version": native.abi_version,
    }


def _masked_chain_receipt(native, workers=None) -> dict[str, object]:
    """What maps the masked surface fields, for a backend's receipt.

    The library is named only when it carries both entries the masked
    fields run on (the WPS chain and the native HRRR route's soil
    stencil); one that predates either is recorded as unavailable with
    the entries it lacks, because the preparation refuses it by name at
    the first such field and a receipt must not claim an entry the
    library does not have.  ``workers`` is the worker count those
    entries were given, ``auto`` when none was (the CPU backend's
    automatic count, or every CPU under CUDA).
    """

    receipt: dict[str, object] = {
        "implementation": WPS_MASKED_CHAIN_IMPLEMENTATION,
        "entry": WPS_MASKED_CHAIN_ENTRY,
        "stencil_implementation": MASKED_STENCIL_IMPLEMENTATION,
        "stencil_entry": MASKED_STENCIL_ENTRY,
        "workers": "auto" if workers is None else int(workers),
    }
    missing = [name for name, present in (
        (WPS_MASKED_CHAIN_ENTRY,
         getattr(native, "wps_masked_chain_entry", False)),
        (MASKED_STENCIL_ENTRY,
         getattr(native, "masked_stencil_entry", False)),
    ) if not present]
    if missing:
        receipt["bridge"] = None
        receipt["unavailable"] = (
            f"the CPU preprocessing library {native.path.name} predates "
            f"{' and '.join(missing)}")
    else:
        receipt["bridge"] = _bridge_identity(native)
    return receipt


def _host(value, *, dtype=None) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=dtype)


def _masked_nearest_cpu(field, latitude, longitude, target_lat, target_lon,
                        source_landmask, target_landmask, *, surface="match",
                        fill_value=0.0, search_radius=8, strict=True,
                        native=None, workers=None):
    """The bounded WPS surface-nearest operator in float32.

    Runs in the Rust preprocessing library (``gpuwm_masked_nearest_f32``)
    on ``workers`` threads (the automatic count when None), byte-identical
    to the NumPy scan kept as its test oracle
    (``woof/verify/water_blend_oracle.py``).  ``native`` is the backend's
    own library; without one, the library the resolution ladder picks.
    """

    from woof.ingest.horiz import _regular_coordinates

    if isinstance(search_radius, (bool, np.bool_)) \
            or not isinstance(search_radius, (int, np.integer)) \
            or int(search_radius) < 0:
        raise ValueError("search_radius must be a non-negative integer")
    y_raw, x_raw = _regular_coordinates(
        latitude, longitude, target_lat, target_lon)
    field = np.asarray(_host(field), dtype=np.float32)
    source_landmask = np.asarray(_host(source_landmask), dtype=np.bool_)
    target_landmask = np.asarray(_host(target_landmask), dtype=np.bool_)
    source_shape = (len(latitude), len(longitude))
    if field.ndim != 2 or field.shape != source_shape \
            or source_landmask.shape != source_shape:
        raise ValueError("field/source_landmask shape does not match source axes")
    if target_landmask.shape != y_raw.shape:
        raise ValueError(
            "target_landmask shape does not match target coordinates")
    if surface not in ("match", "land", "water"):
        raise ValueError("surface must be 'match', 'land', or 'water'")
    if native is None:
        from woof.ingest.cpu_backend import shared_cpu_backend

        native = shared_cpu_backend()
    values, unmatched = native.masked_nearest(
        field, source_landmask, y_raw, x_raw, target_landmask,
        surface=surface, fill_value=fill_value, radius=int(search_radius),
        workers=workers)
    if strict and unmatched:
        raise ValueError("no matching source surface within search_radius")
    return np.ascontiguousarray(values, dtype=np.float32)


def _rotate_earth_to_grid_cpu(u_earth, v_earth, sinalpha, cosalpha):
    u = np.asarray(_host(u_earth), dtype=np.float32)
    v = np.asarray(_host(v_earth), dtype=np.float32)
    sina = np.asarray(_host(sinalpha), dtype=np.float32)
    cosa = np.asarray(_host(cosalpha), dtype=np.float32)
    if u.shape != v.shape:
        raise ValueError("u_earth and v_earth shapes differ")
    return (np.asarray(u * cosa + v * sina, dtype=np.float32),
            np.asarray(v * cosa - u * sina, dtype=np.float32))


def _era5_rh_to_water_cpu(relative_humidity, temperature):
    """WPS v4.6 ``rrpr.F:fix_gfs_rh`` mixed-phase RH to liquid RH (CPU).

    Same transcription as the CUDA path: Murphy and Koop 2005 ice saturation
    vapor pressure, Bolton 1980 liquid, linear 253.15-273.15 K blend, only
    applied at and below freezing.  Float64 setup math, FP32 result.
    """
    rh = np.asarray(_host(relative_humidity), dtype=np.float64)
    t = np.asarray(_host(temperature), dtype=np.float64)
    if rh.shape != t.shape:
        raise ValueError("relative_humidity and temperature shapes differ")
    # Elementwise, so it runs as contiguous flat slabs on host threads
    # (woof.ingest.host_parallel): every element takes exactly the
    # arithmetic it took in one pass, so the result is byte-identical.
    # One pass of this was ~12 % of a CPU GFS preparation's main thread.
    from woof.ingest.host_parallel import host_threads, map_ordered

    threads = (host_threads(rh.size // _RH_SLAB_MIN_VALUES)
               if rh.size >= 2 * _RH_SLAB_MIN_VALUES else 1)
    if threads == 1:
        return _rh_to_water_block(rh, t, workers=None)
    rh_flat = np.ascontiguousarray(rh).reshape(-1)
    t_flat = np.ascontiguousarray(t).reshape(-1)
    edges = np.linspace(0, rh_flat.size, threads + 1).astype(np.int64)
    slabs = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    parts = map_ordered(
        lambda slab: _rh_to_water_block(
            rh_flat[slab[0]:slab[1]], t_flat[slab[0]:slab[1]], workers=1),
        slabs, workers=threads)
    return np.concatenate(parts).reshape(rh.shape)


#: Smallest flat slab the RH conversion hands one host thread.
_RH_SLAB_MIN_VALUES = 1 << 18


def _rh_to_water_block(rh, t, *, workers):
    from woof.core import portable_math as pm

    eis = 0.01 * pm.exp(9.550426 - 5723.265 / t + 3.53068 * pm.log(t, workers=workers)
                        - 0.00728332 * t, workers=workers)
    ews = 6.112 * pm.exp(17.67 * (t - 273.15) / ((t - 273.15) + 243.5),
                         workers=workers)
    frac = (273.15 - t) / 20.0
    blended = frac * eis + (1.0 - frac) * ews
    r = np.where(t > 253.15, blended, eis)
    converted = np.where(t <= 273.15, rh * (r / ews), rh)
    return np.asarray(converted, dtype=np.float32)


@dataclass(frozen=True)
class _CpuVerticalPlan:
    backend: "ParallelCpuPreprocessBackend"
    source_pressure: np.ndarray
    surface_pressure: np.ndarray
    target_pressure: np.ndarray
    _geometry: dict = dataclass_field(default_factory=dict, compare=False, repr=False)
    _geometry_lock: object = dataclass_field(default_factory=RLock, compare=False, repr=False)

    def apply(self, field, surface_value, **options):
        with self._geometry_lock:
            return self._apply(field, surface_value, **options)

    def _apply(self, field, surface_value, **options):
        # CUDA can skip a redundant finite-value scan after a caller-side
        # validation.  The native CPU ABI always validates at its boundary.
        options.pop("values_are_finite", None)
        known = {"interp_in_logp", "extrap", "force_sfc_in_vinterp",
                 "zap_close_levels", "vboundb"}
        geometry_types = (isinstance(options.get("interp_in_logp", True), (bool, np.bool_))
            and isinstance(options.get("force_sfc_in_vinterp", 1), (int, np.integer))
            and isinstance(options.get("zap_close_levels", 500.0), (int, float, np.integer, np.floating)))
        if not (set(options) - known) and geometry_types:
            logp = options.get("interp_in_logp", True)
            force = options.get("force_sfc_in_vinterp", 1)
            zap = options.get("zap_close_levels", 500.0)
            key = (bool(logp), int(force), float(zap))
            native = self._geometry.get(key)
            if key not in self._geometry:
                # One immutable geometry per plan, with all retained native
                # bytes priced. A mode change releases the previous owner.
                for previous in self._geometry.values():
                    if previous is not None:
                        previous.close()
                self._geometry.clear()
                native = self.backend._native.prepare_vertical_geometry(
                    self.source_pressure, self.surface_pressure,
                    self.target_pressure, interp_in_logp=logp,
                    force_sfc_in_vinterp=force, zap_close_levels=zap,
                    workers=self.backend.workers)
                self._geometry[key] = native
            if native is not None:
                result = native.apply(field, surface_value,
                    self.source_pressure, self.surface_pressure, self.target_pressure,
                    extrap=options.get("extrap", "constant"),
                    vboundb=options.get("vboundb", 4), workers=self.backend.workers)
                if result is not None:
                    return result
                native.close()
                self._geometry[key] = None
        return self.backend._native.wrf_vertical_interpolate(
            field, surface_value, self.source_pressure,
            self.surface_pressure, self.target_pressure,
            workers=self.backend.workers, **options)


@dataclass(frozen=True)
class _CudaVerticalPlan:
    raw_plan: object

    def apply(self, field, surface_value, **options):
        from woof.ingest.vert import _wrf_vert_interp_gpu_prepared

        return _wrf_vert_interp_gpu_prepared(
            field, surface_value, self.raw_plan, **options)


class ParallelCpuPreprocessBackend:
    """Production adapter for the deterministic packaged Rust CPU bridge."""

    name = "cpu"
    implementation = "rust-parallel-fp32-v1"
    array_module = np

    def __init__(self, *, workers: int | None = None,
                 bridge: Path | str | None = None):
        if isinstance(workers, (bool, np.bool_)) or (
                workers is not None
                and not isinstance(workers, (int, np.integer))):
            raise TypeError("workers must be an integer")
        if workers is not None and int(workers) < 1:
            raise ValueError("workers must be positive")
        from woof.ingest.preparation_workers import effective_workers
        self.requested_workers = None if workers is None else int(workers)
        self.workers = effective_workers(workers)
        self._native = CpuPreprocessBackend(bridge)
        self._explicit_math_selection = bridge is not None
        self._vertical_routes: list[dict[str, object]] = []
        #: How :func:`resolve_preprocess_backend` chose this backend;
        #: ``None`` for one constructed directly.
        self.selection: dict[str, object] | None = None

    @staticmethod
    def float32(value):
        return np.asarray(_host(value), dtype=np.float32)

    @staticmethod
    def bool_array(value):
        return np.asarray(_host(value), dtype=np.bool_)

    def regular_plan(self, latitude, longitude, target_lat, target_lon):
        native = self._native.regular_plan(
            latitude, longitude, target_lat, target_lon)

        return self._bind_native_plan(native)

    def indexed_plan(self, source_shape, y, x):
        """Bind an exact projected-source fractional-index plan."""

        native = self._native.indexed_plan(source_shape, y, x)
        return self._bind_native_plan(native)

    @property
    def indexed_donor_interp(self) -> bool:
        """Whether this library carries the exact-donor horizontal entry."""

        return bool(getattr(self._native, "indexed_donor_interp", False))

    def indexed_donor_plan(self, source_shape, donor_y, donor_x,
                           fraction_y, fraction_x):
        """Bind an exact projected-source integer-donor plan."""

        native = self._native.indexed_donor_plan(
            source_shape, donor_y, donor_x, fraction_y, fraction_x)
        return self._bind_native_plan(native)

    def _bind_native_plan(self, native):
        workers = self.workers

        class BoundPlan:
            source_shape = native.source_shape
            target_shape = native.target_shape

            def apply(bound_self, field, method="parabolic", *, source_support=False):
                return native.apply(
                    field, method=method, workers=workers,
                    source_support=source_support)

        return BoundPlan()

    def masked_nearest(self, *args, **kwargs):
        return _masked_nearest_cpu(
            *args, **kwargs, native=self._native,
            workers=self.host_step_workers)

    @property
    def host_step_workers(self) -> int:
        """The threads every Rust host step of this backend runs on: its
        own worker count, or the available CPU/memory budget."""

        return (self.workers if self.workers is not None
                else automatic_workers())

    def at_workers(self, workers: int) -> "ParallelCpuPreprocessBackend":
        """This backend on ``workers`` threads, as one slot of its budget.

        A preparation that splits its worker budget into concurrent slots
        runs each slot on this backend with fewer threads.  That is this
        backend's choice, not a second one, so the slot runs this
        library, carries this ``selection`` and adds to this record of
        vertical routes: its receipt is this backend's but for the
        worker counts.  A slot resolved again by name recorded a CPU that
        auto fell to as one "named by the caller" and began an empty
        route record, and the HRRR root preparation refused every slot
        of a CPU run with two or more boundary hours on that difference.
        """

        slot = ParallelCpuPreprocessBackend(
            workers=workers, bridge=self._native.path)
        slot.selection = (None if self.selection is None
                          else dict(self.selection))
        slot._vertical_routes = self._vertical_routes
        slot._explicit_math_selection = self._explicit_math_selection
        return slot

    def wps_masked_chain_engine(self):
        """The library and worker count the masked surface chain runs on.

        This backend's own library, so an explicit ``cpu_bridge`` reaches
        the masked fields too, and its own worker count (the automatic
        count when none was given, as for every thread the
        CPU preparation starts on its own).  A library without the chain
        is refused by name with the remedy.
        """

        self._native.require_wps_masked_chain()
        return self._native, self.host_step_workers

    @staticmethod
    def rotate_earth_to_grid(*args):
        return _rotate_earth_to_grid_cpu(*args)

    def era5_rh_to_water(self, *args):
        with preprocessing_math_scope(self):
            return _era5_rh_to_water_cpu(*args)

    def prepare_wrf_vertical(self, source_pressure, surface_pressure,
                             target_pressure):
        source = np.ascontiguousarray(_host(source_pressure), dtype=np.float32)
        surface = np.ascontiguousarray(
            _host(surface_pressure), dtype=np.float32)
        target = np.ascontiguousarray(_host(target_pressure), dtype=np.float32)
        source_levels = int(source.shape[0]) if source.ndim else 0
        _record_vertical_route(self._vertical_routes, {
            "source_levels": source_levels,
            "column_levels": source_levels + 1,
            "backend": "cpu",
            "kernel_level_tier": None,
            "reason": "the CPU preprocessing backend was selected",
        })
        return _CpuVerticalPlan(self, source, surface, target)

    def receipt(self) -> dict[str, object]:
        """Return proof metadata that binds the native CPU implementation.

        ``vertical_interpolation`` is this backend's own record of every
        vertical geometry it has prepared, shared rather than copied, so a
        receipt taken before the preparation and written after it (every
        preparation proof) carries what actually ran.
        """

        from woof.core import portable_math
        binding = portable_math.current_cpu_bridge_binding()
        math_selection = {}
        if binding is not None and binding.publish_selection:
            if binding.path != self._native.path.resolve():
                raise ValueError("CPU preprocessing receipt differs from its active host setup math library")
            math_selection = {"host_setup_math": {
                "implementation": portable_math.implementation(),
                "bridge": {"name": binding.path.name, "sha256": _sha256(binding.path)},
                "dispatcher_sha256": _sha256(Path(portable_math.__file__)),
                "portable_math_generation": (portable_math.PORTABLE_MATH_VERSION if binding.library is not None else None)}}
        from woof.ingest.preparation_workers import worker_receipt
        native = getattr(getattr(self._native, "_library", None), "gpuwm_preprocess_cpu_parallelism", None)
        native_effective = None
        if native is not None:
            import ctypes
            native.argtypes = [ctypes.c_size_t]
            native.restype = ctypes.c_size_t
            native_effective = int(native(self.host_step_workers))
        return {
            "schema": PREPROCESS_IMPLEMENTATION_SCHEMA,
            "backend": self.name,
            "implementation": self.implementation,
            **math_selection,
            "workers": self.workers if self.workers is not None else "auto",
            "host_cpu_count": os.cpu_count(),
            "parallelism": worker_receipt(self.requested_workers, native_effective=native_effective),
            "bridge": {
                "name": self._native.path.name,
                "sha256": _sha256(self._native.path),
                "abi_version": self._native.abi_version,
                "required_abi_version": CPU_BACKEND_ABI,
            },
            "contracts": _shared_contracts(),
            "implementation_tree": _implementation_tree("cpu"),
            "vertical_interpolation": self._vertical_routes,
            "masked_surface_chain": _masked_chain_receipt(
                self._native, self.workers),
            **_selection_block(self),
        }


class CudaPreprocessBackend:
    """Adapter around the existing CuPy/JIT CUDA preprocessing kernels."""

    name = "cuda"
    implementation = "cupy-fp32"
    workers = 1

    def __init__(self, *, host_workers: int | None = None):
        self._vertical_routes: list[dict[str, object]] = []
        #: The CPU workers the host steps take (the masked surface fields,
        #: which run in the Rust preprocessing library under this backend
        #: too); ``None`` for every CPU the process may use.
        self.host_workers = _checked_workers(host_workers)
        #: How :func:`resolve_preprocess_backend` chose this backend;
        #: ``None`` for one constructed directly.
        self.selection: dict[str, object] | None = None

    @property
    def array_module(self):
        from woof.ingest.horiz import _cupy

        return _cupy()

    def float32(self, value):
        from woof.ingest.horiz import _float32_gpu

        return _float32_gpu(value)

    def bool_array(self, value):
        cp = self.array_module
        return cp.asarray(value, dtype=cp.bool_)

    @staticmethod
    def regular_plan(latitude, longitude, target_lat, target_lon):
        from woof.ingest.horiz import _RegularGpuPlan

        return _RegularGpuPlan(
            latitude, longitude, target_lat, target_lon)

    @staticmethod
    def masked_nearest(*args, **kwargs):
        from woof.ingest.horiz import masked_nearest_gpu

        return masked_nearest_gpu(*args, **kwargs)

    def wps_masked_chain_engine(self):
        """The library and worker count the masked surface chain runs on.

        The masked fields are copied to the host for this chain, which
        runs in the Rust preprocessing library the resolution ladder
        picks, on this backend's host workers (every CPU the process may
        use when none was given), exactly as it does under the CPU
        backend.  A missing or outdated library is refused by name with
        the remedy.
        """

        native = masked_fields_cpu_backend()
        native.require_wps_masked_chain()
        return native, self.effective_host_workers

    @property
    def host_step_workers(self) -> int:
        """The threads the Rust host steps take (:attr:`effective_host_workers`)."""

        return self.effective_host_workers

    @property
    def effective_host_workers(self) -> int:
        """The threads the host steps run on: ``host_workers``, or every
        CPU the process may use when none was given."""

        return (self.host_workers if self.host_workers is not None
                else available_cpu_count())

    @staticmethod
    def rotate_earth_to_grid(*args):
        from woof.ingest.horiz import rotate_earth_to_grid_gpu

        return rotate_earth_to_grid_gpu(*args)

    @staticmethod
    def era5_rh_to_water(*args):
        from woof.ingest.horiz import _era5_rh_to_water_gpu

        return _era5_rh_to_water_gpu(*args)

    def prepare_wrf_vertical(self, source_pressure, surface_pressure,
                             target_pressure):
        """Prepare one pressure geometry on the kernel tier that holds it.

        A column deeper than the kernel's top tier is handed to the CPU
        bridge by the plan itself; either way the route, the source level
        count and the reason land in this backend's receipt.
        """
        from woof.ingest.vert import (
            _prepare_wrf_vert_interp_geometry, wrf_vertical_route)

        raw_plan = _prepare_wrf_vert_interp_geometry(
            source_pressure, surface_pressure, target_pressure)
        route = wrf_vertical_route(raw_plan.source_shape[0])
        if raw_plan.cpu_bridge is not None:
            route["cpu_bridge"] = _bridge_identity(raw_plan.cpu_bridge)
        _record_vertical_route(self._vertical_routes, route)
        return _CudaVerticalPlan(raw_plan)

    def receipt(self) -> dict[str, object]:
        """Return proof metadata for the selected CuPy CUDA implementation.

        ``vertical_interpolation`` is this backend's own record of every
        vertical geometry it has prepared: which engine ran it (the CUDA
        kernel at a named column tier, or the CPU bridge for a column deeper
        than the top tier), the source level count and why.  The list is
        shared rather than copied, so a receipt taken before the
        preparation and written after it (every preparation proof) carries
        what actually ran.
        """

        from woof.ingest.horiz import _cupy
        cp = _cupy()
        return {
            "schema": PREPROCESS_IMPLEMENTATION_SCHEMA,
            "backend": self.name,
            "implementation": self.implementation,
            "workers": self.workers,
            "cupy_version": cp.__version__,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "contracts": _shared_contracts(),
            "implementation_tree": _implementation_tree("cuda"),
            "vertical_interpolation": self._vertical_routes,
            "masked_surface_chain": _cuda_masked_chain_receipt(
                self.host_workers),
            **_selection_block(self),
        }


def _cuda_masked_chain_receipt(host_workers=None) -> dict[str, object]:
    """The CUDA backend's masked-chain library, or why there is none.

    A receipt is taken before the preparation it describes, so a missing
    library is recorded here rather than raised: the preparation refuses
    it by name at the first masked field.
    """

    try:
        return _masked_chain_receipt(shared_cpu_backend(), host_workers)
    except (OSError, RuntimeError) as error:
        return {
            "implementation": WPS_MASKED_CHAIN_IMPLEMENTATION,
            "entry": WPS_MASKED_CHAIN_ENTRY,
            "stencil_implementation": MASKED_STENCIL_IMPLEMENTATION,
            "stencil_entry": MASKED_STENCIL_ENTRY,
            "workers": ("auto" if host_workers is None
                        else int(host_workers)),
            "bridge": None,
            "unavailable": str(error).splitlines()[0],
        }


def _with_host_workers(backend, workers):
    """``backend`` with ``workers`` as the worker count of its host steps."""

    if workers is not None:
        backend.host_workers = _checked_workers(workers)
    return backend


def _checked_workers(workers):
    """A worker count as an int, or None; refused unless a positive integer."""

    if isinstance(workers, (bool, np.bool_)) or (
            workers is not None
            and not isinstance(workers, (int, np.integer))):
        raise TypeError("workers must be an integer")
    if workers is not None and int(workers) < 1:
        raise ValueError("workers must be positive")
    from woof.ingest.preparation_workers import effective_workers
    return None if workers is None else effective_workers(workers)


def _gpu_runtime_installed() -> bool:
    """Whether cupy RESOLVES here, without importing it.

    The cheap presence half only, same split as
    :func:`woof.capabilities.is_installed` (which this defers to):
    "installed" and "working" are two claims, and ``woof doctor`` makes
    the second one.  Asked at resolve time so an explicit ``cuda``
    request on a CPU-only install is refused with the remedy BEFORE any
    bytes are decoded, instead of dying in the first kernel's
    ``RuntimeError: CuPy is required for GPU horizontal interpolation``
    after minutes of work.
    """

    from woof.capabilities import is_installed

    return is_installed("cupy")


#: The auto->cpu announcement's dedup set: nest initialization and the
#: per-snapshot helpers re-resolve the same selector string, and four
#: copies of one sentence is noise.  One line per distinct reason per
#: process.
_ANNOUNCED_AUTO_REASONS: set[str] = set()


def _announce_auto_cpu(reason: str) -> None:
    """Say, once, that ``auto`` chose the CPU backend and why.

    ONE line, because a bare-default preparation on a CPU-only box is a
    supported route (fixed-means-default), not an error -- but a reader
    watching a prep that used to run on a card deserves to be told which
    engine is running and what changed the answer.
    """

    if reason in _ANNOUNCED_AUTO_REASONS:
        return
    _ANNOUNCED_AUTO_REASONS.add(reason)
    from woof.explain import warn

    warn(f"preprocess backend auto: {reason}, so source-grid/WRF-real "
         "preprocessing runs on the deterministic parallel CPU backend",
         why="The CPU backend is the packaged Rust bridge, held to "
             "numeric parity with the CUDA path by "
             "tests/test_preprocess_cpu_backend.py; only this setup "
             "half runs there.  The forecast model itself is CUDA-only "
             "and still needs a card.  Force the choice with "
             "--preprocess-backend cpu or cuda.")


#: GPU preprocessing certification, keyed by CUDA runtime MAJOR.
#:
#: ``auto`` prepares on the card only where the runtime's major has a row
#: here and CuPy is at least that row's ``cupy_major_minimum``; everywhere
#: else it prepares on the CPU backend and says why in one line.  The gate
#: prevents one breakage: a CUDA preparation on a runtime whose NVRTC-built
#: interpolation kernels were never shown to reproduce the CPU backend
#: within the declared parity rules, so an initial state could differ from
#: the one the same case prepares on the CPU with nothing recording it.
#:
#: A major gets a row only with a passing certification record under
#: ``tests/data/preprocess_cuda_certification/cuda-<major>.json``: the
#: ``gpu``-marked CPU/CUDA parity tests in tests/test_preprocess_cpu_backend.py
#: plus one real-data preparation made on both backends and compared by
#: tools/verify_wrf_backend_parity.py.
#: tests/test_preprocess_cuda_certification.py refuses a row without one.
#: ``extra`` is the install extra carrying that major's CuPy wheel, which
#: the remedy below names.
#:
#: This is a PREPROCESSING certification.  The sealed native-WRF
#: distribution pins its own runtime family separately
#: (:data:`woof.gpu_stack_identity.CUDA_RUNTIME_RANGE`), because that
#: bundle ships one CuPy wheel.
CERTIFIED_PREPROCESS_CUDA_MAJORS = MappingProxyType({
    12: MappingProxyType({"cupy_major_minimum": 13, "extra": "gpu-cu12"}),
    13: MappingProxyType({"cupy_major_minimum": 14, "extra": "gpu-cu13"}),
})


def _certified_majors_text() -> str:
    return " and ".join(
        f"CUDA {major}" for major in sorted(CERTIFIED_PREPROCESS_CUDA_MAJORS))


def _gpu_preprocess_remedy() -> str:
    """The refusal for an explicit ``cuda`` request with no cupy.

    Built from :data:`CERTIFIED_PREPROCESS_CUDA_MAJORS`, so the remedy
    names exactly the pairs the resolver certifies: an install line for
    a runtime ``auto`` would still decline is not a remedy.
    """

    lines = [
        "CUDA preprocessing was requested but this install cannot import "
        "cupy, so every GPU interpolation kernel is unreachable.  Refusing "
        "here, before any source bytes are decoded, rather than in the "
        "first kernel after them.",
        f"  # GPU preprocessing is certified on {_certified_majors_text()}; "
        "install the CuPy wheel for this box's CUDA major:",
    ]
    for major, row in sorted(CERTIFIED_PREPROCESS_CUDA_MAJORS.items()):
        lines.append(f"  # on a CUDA {major}.x box:")
        lines.append(f"  remedy: pip install 'recast-woof[{row['extra']}]'")
    lines.append(
        "  # or run the same preparation off-GPU: --preprocess-backend cpu")
    lines.append("  # (or auto, which picks the CPU backend on this install)")
    return "\n".join(lines)


_GPU_PREPROCESS_REMEDY = _gpu_preprocess_remedy()

#: The ``reason`` a selection records when the caller named the backend.
NAMED_BY_CALLER = "named by the caller"


def _selection(requested: str, backend, reason: str, *, device_load=None):
    """Attach how this backend was chosen; its receipt carries it.

    ``requested`` is the selector as given (``auto``, ``cuda`` or
    ``cpu``), ``backend`` is what runs, ``reason`` is why.  A CPU
    preparation the reader did not ask for therefore names its cause in
    the preparation receipt, not only in a log line that may be gone.
    """

    backend.selection = {
        "requested": requested,
        "backend": backend.name,
        "reason": reason,
    }
    if device_load is not None:
        backend.selection["device_load"] = device_load
    return backend


# Utilization is telemetry, never a CPU/CUDA arithmetic choice. The same
# historical moving-nest SINT/EOS mismatch is fixed at preparation; actual
# prepared-array equality remains checked before repeatability comparisons.
# Keep this threshold for the log and multi-card placement only.
AUTO_BUSY_UTILIZATION_PERCENT = 50


def _auto_device_load(cp):
    """Use the fit's live instrument; missing telemetry keeps the prior choice.

    Returns ``(reason or None, load)``; ``load`` carries the free and total
    bytes the same reading saw, which :func:`admit_preparation` weighs the
    preparation's price against, so auto reads the card once.

    A card the probe could not open through CUDA is not missing telemetry:
    the preparation waits for that certified card to become available.
    The probe reports this only for a CUDA runtime error; a failure of the
    probe's own reads is missing telemetry, not a backend decision.
    """
    from woof.core.device_probe import device_memory_probe_subprocess

    probe = device_memory_probe_subprocess(
        device=int(cp.cuda.runtime.getDevice()), report_failure=True)
    if probe is None:
        return None, None
    cuda_error = probe.get("cuda_error")
    if cuda_error is None:
        free = probe.get("free_bytes")
        total = probe.get("total_bytes")
        utilization = probe.get("utilization_gpu_percent")
    else:
        nvml = probe.get("nvml") or {}
        used = nvml.get("used_bytes")
        total = nvml.get("total_bytes")
        utilization = nvml.get("utilization_gpu_percent")
        free = (total - used if type(used) is int and type(total) is int
                and 0 <= used <= total else None)
    load = {
        "probe": "device_memory_probe_subprocess",
        "free_bytes": free,
        "total_bytes": total,
        "utilization_gpu_percent": utilization,
        "busy_utilization_threshold_percent": AUTO_BUSY_UTILIZATION_PERCENT,
    }
    if cuda_error is not None:
        load["cuda_error"] = cuda_error
    reasons = []
    if (isinstance(utilization, (int, float))
            and not isinstance(utilization, bool)
            and AUTO_BUSY_UTILIZATION_PERCENT <= utilization <= 100):
        reasons.append(
            f"GPU utilization {utilization:g}% meets the busy threshold "
            f"of {AUTO_BUSY_UTILIZATION_PERCENT}%")
    if cuda_error is not None:
        reasons.append(f"CUDA could not open the card ({cuda_error})")
    return ("; ".join(reasons) if reasons else None), load


def _certified_cuda(cp) -> tuple[bool, str]:
    """Whether this CuPy/runtime pair is certified, and the sentence saying so."""

    runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
    major = runtime_version // 1000
    cupy_version = str(cp.__version__)
    cupy_major = int(cupy_version.split(".", 1)[0])
    row = CERTIFIED_PREPROCESS_CUDA_MAJORS.get(major)
    if row is None:
        return False, (
            f"CUDA runtime {runtime_version} (CUDA {major}) is not "
            f"certified for GPU preprocessing (certified: "
            f"{_certified_majors_text()})")
    minimum = int(row["cupy_major_minimum"])
    if cupy_major < minimum:
        return False, (
            f"cupy {cupy_version} is older than cupy {minimum}, the oldest "
            f"certified for GPU preprocessing on CUDA {major}")
    return True, (
        f"cupy {cupy_version} on CUDA runtime {runtime_version} is "
        f"certified for GPU preprocessing (CUDA {major})")


def resolve_preprocess_backend(backend="cuda", *, workers: int | None = None,
                               cpu_bridge: Path | str | None = None,
                               reason: str | None = None, price=None):
    """Resolve a public backend selector without silently changing policy.

    ``reason`` is for a caller whose POLICY, not a person, named the
    backend (a host-tiled configuration prepares on the CPU); it replaces
    :data:`NAMED_BY_CALLER` in the receipt's ``selection`` block.  That
    caller prints its own line; ``auto`` prints one whenever it lands on
    the CPU.

    ``price`` (a :class:`woof.ingest.preparation_price.PreparationDevicePrice`,
    or a callable returning one) is what the preparation needs on the
    card.  Given one, a CUDA answer is weighed against the card's free
    memory before anything is allocated (:func:`admit_preparation`): CUDA
    waits for occupied memory and refuses a price beyond the card's capacity.
    Load never changes the preparation backend. A callable is priced only
    when the answer is CUDA.
    """
    chosen = _resolve_preprocess_backend(
        backend, workers=workers, cpu_bridge=cpu_bridge, reason=reason)
    if price is None:
        return chosen
    # auto's reading was taken in this same call, so it is the decision's
    # own reading; reuse it rather than probe the card twice.
    load = (getattr(chosen, "selection", None) or {}).get("device_load")
    return admit_preparation(chosen, price, workers=workers, probe=load)


def _resolve_preprocess_backend(backend="cuda", *, workers=None,
                                cpu_bridge=None, reason=None):

    if backend is None:
        backend = "cuda"
    if not isinstance(backend, str):
        required = (
            "regular_plan", "masked_nearest", "rotate_earth_to_grid",
            "era5_rh_to_water", "prepare_wrf_vertical", "receipt",
        )
        if not all(callable(getattr(backend, name, None)) for name in required):
            raise TypeError("custom preprocessing backend is incomplete")
        if workers is not None or cpu_bridge is not None:
            raise ValueError(
                "workers/cpu_bridge cannot accompany a backend object")
        return backend
    normalized = backend.strip().lower()
    from woof.local_gpu import no_local_gpu
    if no_local_gpu() and normalized in ("cuda", "auto"):
        refusal = ("GPUWM_NO_LOCAL_GPU forbids local CUDA preprocessing; "
                   "select backend='cpu' on this machine")
        if normalized == "cuda":
            raise ValueError(refusal)
        if cpu_bridge is not None:
            raise ValueError("cpu_bridge cannot accompany backend='auto'")
        if reason is not None:
            raise ValueError(
                "a selection reason accompanies a named backend, not auto")
        _announce_auto_cpu(refusal)
        return _selection("auto", ParallelCpuPreprocessBackend(
            workers=workers), refusal)
    if reason is not None and (not isinstance(reason, str)
                               or not reason.strip()):
        raise ValueError("a backend selection reason must be a sentence")
    named = NAMED_BY_CALLER if reason is None else " ".join(reason.split())
    sealed_backends = _sealed_distribution_preprocess_backends()
    if normalized == "cuda":
        if sealed_backends is not None and "cuda" not in sealed_backends:
            raise ValueError(
                "CUDA preprocessing is absent from the sealed native "
                "distribution")
        # workers reaches the host steps the CUDA backend runs on the CPU
        # (the masked surface fields).  cpu_bridge stays the CPU
        # backend's: under CUDA the host library is the one the
        # resolution ladder picks, and a second way to name it would be
        # a second answer to which library ran.
        if cpu_bridge is not None:
            raise ValueError(
                "cpu_bridge applies only to the CPU backend; under the CUDA "
                "backend the host steps use the library the resolution "
                "ladder picks (set WOOF_CPU_PREPROCESS_BRIDGE to choose it)")
        _mask_to_the_least_busy_card()
        if not _gpu_runtime_installed():
            # An EXPLICIT cuda request on a CPU-only install: a named
            # refusal with the remedy, at the front of the work.  The
            # bare default reaches the CPU backend through "auto" and
            # never lands here.
            raise ValueError(_GPU_PREPROCESS_REMEDY)
        _wait_for_the_card_blocking()
        return _selection(
            "cuda", _with_host_workers(CudaPreprocessBackend(), workers),
            named)
    if normalized == "cpu":
        if sealed_backends is not None and "cpu" not in sealed_backends:
            raise ValueError(
                "CPU preprocessing is absent from the sealed native "
                "distribution")
        return _selection("cpu", ParallelCpuPreprocessBackend(
            workers=workers, bridge=cpu_bridge), named)
    if normalized == "auto":
        if cpu_bridge is not None:
            raise ValueError("cpu_bridge cannot accompany backend='auto'")
        if reason is not None:
            # auto records its own reason; a caller's would be dropped.
            raise ValueError(
                "a selection reason accompanies a named backend, not auto")
        # The CPU fallback is the bare default's road on every box where
        # CUDA is unusable or uncertified (fixed-means-default: the prep
        # doors default to "auto"), so EVERY fall is announced, once per
        # reason, and recorded in the receipt: a silent swap puts a whole
        # preparation on the host's cores beside an idle card.
        device_load = None
        certified_candidate = None
        if sealed_backends is not None and "cuda" not in sealed_backends:
            fallback = ("the sealed native distribution carries no CUDA "
                        "preprocessing")
        else:
            try:
                _mask_to_the_least_busy_card()
                candidate = _with_host_workers(
                    CudaPreprocessBackend(), workers)
                cp = candidate.array_module
                if int(cp.cuda.runtime.getDeviceCount()) <= 0:
                    fallback = "no CUDA device is visible here"
                else:
                    certified, sentence = _certified_cuda(cp)
                    if certified:
                        # Admission and telemetry are outside the capability
                        # catch below: contention must never select CPU math.
                        certified_candidate = (candidate, cp, sentence)
                    else:
                        fallback = sentence
            except (AttributeError, ImportError, RuntimeError,
                    ValueError) as error:
                # A device error is a RuntimeError too, and naming it "not
                # installed" sends a reader with a working cupy to
                # reinstall it.  An installed cupy that fails to import
                # arrives wrapped in the "CuPy is required" RuntimeError,
                # so the cause is what is quoted.
                cause = error.__cause__ or error
                detail = (str(cause).strip().splitlines() or [""])[0][:160]
                quoted = f" ({detail})" if detail else ""
                if "cudaErrorNoDevice" in detail:
                    # CUDA_VISIBLE_DEVICES="" or no card: the runtime
                    # raises here instead of counting zero devices.
                    fallback = "no CUDA device is visible here" + quoted
                elif _gpu_runtime_installed():
                    fallback = ("cupy is installed but could not be loaded"
                                if isinstance(cause, ImportError)
                                else "cupy is installed but CUDA could not "
                                "start") + quoted
                else:
                    fallback = "cupy is not installed here"
        if certified_candidate is not None:
            candidate, cp, sentence = certified_candidate
            busy_reason, device_load = _auto_device_load(cp)
            if busy_reason:
                print(f"note: preprocess backend auto: {busy_reason}; keeping CUDA; "
                      "waiting for device/memory admission when required",
                      file=sys.stderr, flush=True)
            _wait_for_the_card_blocking()
            chosen = _selection("auto", candidate, sentence, device_load=device_load)
            if device_load and device_load.get("cuda_error"):
                reading, waited = _wait_for_preparation_memory(chosen, 0, device_load)
                chosen.selection.update(initial_device_load=device_load, device_load=reading,
                                        device_wait_seconds=waited)
            return chosen
        _announce_auto_cpu(fallback)
        return _selection(
            "auto", ParallelCpuPreprocessBackend(workers=workers), fallback,
            device_load=device_load)
    raise ValueError("backend must be 'cuda', 'cpu', or 'auto'")


def _cuda_not_yet_initialized() -> bool:
    """True when this process has not called ``cuInit`` (asked without calling it)."""

    import ctypes

    name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    try:
        driver = ctypes.CDLL(name)
    except OSError:
        return False
    count = ctypes.c_int()
    # CUDA_ERROR_NOT_INITIALIZED (3): the driver answers without starting.
    return driver.cuDeviceGetCount(ctypes.byref(count)) == 3


def _mask_to_the_least_busy_card(environ=None) -> str | None:
    """On a multi-card host with no mask, prepare on the least-busy card, not card 0.

    THE BREAKAGE: a preparation with no ``CUDA_VISIBLE_DEVICES`` on a shared
    multi-card box took CUDA's device 0 whatever ran there (ml-probs T1 on
    V8X, 2026-10-27: every ``gfs_direct`` preparation held a context on card
    0, another lane's card, while cards 4-7 were the lane's).  The forecast
    already refuses an ambiguous card (``supervisor.select_gpu``); a
    preparation is short and needs no lock, so it takes the visible card
    with the most free memory among those under the busy threshold and
    names it.  Applied only before this process starts CUDA, only when no
    mask is set and more than one card is listed; with none under the
    threshold nothing is masked and ``auto`` reads the busy card as before.
    Returns the chosen UUID or None.
    """

    environ = os.environ if environ is None else environ
    if environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    cards = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not parts[1].startswith("GPU-"):
            continue
        try:
            cards.append((int(parts[0]), parts[1], int(parts[2]), int(parts[3])))
        except ValueError:
            continue
    if completed.returncode != 0 or len(cards) < 2:
        return None
    idle = [card for card in cards if card[3] < AUTO_BUSY_UTILIZATION_PERCENT]
    if not idle or not _cuda_not_yet_initialized():
        return None
    index, uuid, free_mib, utilization = max(idle, key=lambda card: (card[2], -card[0]))
    environ["CUDA_VISIBLE_DEVICES"] = uuid
    print(f"preprocess: {len(cards)} cards visible and no CUDA_VISIBLE_DEVICES; "
          f"preparing on card {index} ({uuid}, {free_mib} MiB free, "
          f"{utilization}% busy): the most free memory of the cards under "
          f"{AUTO_BUSY_UTILIZATION_PERCENT}% busy (set CUDA_VISIBLE_DEVICES "
          "to choose)",
          file=sys.stderr, flush=True)
    return uuid


def _wait_for_the_card_blocking() -> None:
    """A CUDA preparation's host thread sleeps on the card, never spins.

    Set once the CUDA backend is chosen and before its first allocation, so
    the primary context is created with blocking sync
    (:mod:`woof.cuda_wait`); host-side only, no kernel or value changes.
    """
    from woof.cuda_wait import apply

    apply()


class PreparationDeviceRefused(InitializationMemoryRefused):
    """An explicit CUDA preparation whose price does not fit the card."""


#: The one command-line spelling every preparation door takes.
CPU_PREPARATION_LINE = "--preprocess-backend cpu"


def _device_reading(backend, load):
    """``(free, total, reading)`` of the card: auto's reading or a new probe."""

    if load is not None and type(load.get("free_bytes")) is int:
        total = load.get("total_bytes")
        return load["free_bytes"], (total if type(total) is int else None), load
    from woof.core.device_probe import device_memory_probe_subprocess

    try:
        device = int(backend.array_module.cuda.runtime.getDevice())
    except Exception:                     # the probe's own default device
        device = None
    kwargs = {"report_failure": True}
    if device is not None:
        kwargs["device"] = device
    probe = device_memory_probe_subprocess(**kwargs)
    if probe is None or probe.get("cuda_error") is not None:
        return None, None, probe
    free, total = probe.get("free_bytes"), probe.get("total_bytes")
    if type(free) is not int:
        return None, None, probe
    return free, (total if type(total) is int else None), probe


def preparation_device_fit(price, free, total) -> dict:
    """The receipt's ``device_fit`` block for one priced selection."""

    record = dict(price.record())
    record.update({
        "need_bytes": int(price.need_bytes), "free_bytes": free,
        "total_bytes": total,
        "fits": None if free is None else int(price.need_bytes) <= int(free),
    })
    return record


def _gib(value) -> str:
    return f"{int(value) / 2**30:.1f} GiB"


def preparation_device_sentence(price, free, total) -> str:
    """``the CUDA preparation needs X and the card has Y free of Z``."""

    of = "" if total is None else f" of {_gib(total)}"
    return (f"the CUDA preparation needs {_gib(price.need_bytes)} "
            f"({price.summary()}) and the card has {_gib(free)} free{of}")


def preparation_refusal_message(price, free, total, *,
                                cpu_line: str = CPU_PREPARATION_LINE) -> str:
    """The explicit-cuda refusal: the terms, the breakage, the CPU line."""

    of = "" if total is None else f" of {_gib(total)}"
    return (
        "--preprocess-backend cuda refused before anything was allocated: "
        f"this preparation needs {_gib(price.need_bytes)} on the card "
        f"({price.summary()}) and the card has {_gib(free)} free{of}.  "
        "Started, it would stop with a CUDA out-of-memory "
        f"{price.stage}, after its inputs were decoded.\n"
        f"  remedy: run the same command with {cpu_line} "
        "(the CPU preparation holds this in host memory instead), or with "
        "a larger card. Automatic selection never changes to CPU because "
        "another job occupies the card.")


def _wait_for_preparation_memory(backend, need, reading, *, price=None):
    """Wait for other work to release memory, preserving the CUDA choice.

    The surrounding launcher owns admission/leases. This is the last memory
    check before allocation, not another lease or an arithmetic fallback.
    Unknown telemetry retains CUDA; an impossible capacity refuses by name.
    """
    import time

    began = time.monotonic()
    announced = False
    while True:
        free, total, observed = _device_reading(backend, reading)
        cuda_error = (observed or {}).get("cuda_error")
        if total is not None and need > total:
            detail = (preparation_refusal_message(price, free or 0, total)
                      if price is not None else "CUDA device is unavailable")
            raise PreparationDeviceRefused(detail)
        if not cuda_error and (free is None or free >= need):
            return observed, time.monotonic() - began
        if not announced:
            available = "unread" if free is None else _gib(free)
            print(f"note: CUDA preparation: waiting for {_gib(need)} of device memory "
                  f"({available} free{'; ' + str(cuda_error) if cuda_error else ''}); "
                  "keeping CUDA to preserve the prepared fields", file=sys.stderr, flush=True)
            announced = True
        time.sleep(1.0)
        reading = None


def admit_preparation(backend, price, *, workers=None, probe=None):
    """Weigh a resolved backend against its preparation's device price.

    The decision every CUDA preparation door takes BEFORE its first device
    allocation. Mapped preparation keeps completed arrays on the host and
    stages bounded CUDA batches when the whole build exceeds a 1 GiB pool.
    This leaves the card available for a concurrent forecast. Its receipt
    reserves the batch peak so a forecast can start beside later forcing hours.
    The bounded batches retain the whole array envelope in host RAM. A host
    that cannot hold that envelope keeps the full-device preparation when
    the whole build fits the card's free memory (2.8.4 ran every such
    preparation there, and it retains nothing on the host), recorded in
    ``host_fit`` with ``fits: False`` and the route taken. The host refusal
    stands where neither the host nor the card holds the build, and where a
    bounded backend was already admitted on a decoded price, because a
    chained forecast may have reserved only that backend's batch.
    A CPU backend passes untouched. A CUDA backend waits for occupied
    memory, logging the wait and recording its duration. A price beyond
    the card's capacity is refused with :class:`PreparationDeviceRefused`.
    Neither automatic nor explicit CUDA switches arithmetic because of load.
    The priced selection records ``device_fit`` after admission.
    A card whose free memory cannot be read keeps the choice, recorded
    with ``fits: None``: an unread card never refuses.

    ``probe`` is a reading taken at this decision (the one
    :func:`resolve_preprocess_backend` just took, or a test's stand-in
    card); otherwise one subprocess probe is taken now, and this process
    never stands up a context to take it.  The reading ``auto`` recorded
    when the backend was resolved is NOT reused here: GFS and ERA5 resolve
    before their host decode and decide after it, minutes later, and a
    card other work shares (a chained preparation beside a forecast) can
    fill in between, so that reading kept CUDA on a card that no longer
    held the preparation.

    ``price`` may be a callable returning the price: it is called only
    for a CUDA backend, so a preparation already on the CPU is never
    priced.
    """

    if price is None or getattr(backend, "name", None) != "cuda":
        return backend
    if callable(price):
        price = price()
    selection = dict(getattr(backend, "selection", None) or {})
    free, total, _reading = _device_reading(backend, probe)
    fit = preparation_device_fit(price, free, total)
    context = int(price.terms.get("cuda_context", 0))
    if (price.route == "mapped" and free is not None
            and (int(price.need_bytes) > min(int(free), 1024**3 + context)
                 or getattr(backend, "bounded_cuda", False))):
        from woof.ingest.bounded_cuda import BoundedCudaPreprocessBackend
        source_staging = int(getattr(price, "chunk_source_bytes", 0))
        minimum = max(256 * 1024**2, int(getattr(price, "chunk_minimum_bytes", 0)))
        budget = min(1024**3, max(0, int(free) - context - 256 * 1024**2))
        if budget >= minimum:
            from woof.ingest.preparation_workers import host_available_bytes
            from woof.ingest.memory_refusal import InitializationMemoryRefused
            from woof.ingest.boundary_stream import process_memory_bytes
            from woof.ingest.preparation_price import PREPARATION_FLOOR_BASIS
            host_available = host_available_bytes()
            host_need = max(0, int(price.need_bytes) - context)
            is_floor = price.basis == PREPARATION_FLOOR_BASIS
            previous_host = selection.get("host_fit") or {}
            baseline = (previous_host.get("producer_rss_bytes")
                        if not previous_host.get("price_is_floor", True) else None)
            if baseline is None:
                process_memory = process_memory_bytes()
                baseline = None if process_memory is None else int(process_memory[0])
            host_fit = {
                "need_bytes": host_need, "available_bytes": host_available,
                "fits": None if host_available is None else host_need <= host_available,
                "price_is_floor": is_floor, "producer_rss_bytes": baseline,
                "producer_peak_bytes": None if baseline is None else baseline + host_need,
                "basis": "whole-build array envelope retained on host; decoded source is already resident"}
            if host_available is not None and host_need > host_available:
                # A bounded backend admitted on a decoded price is a
                # published contract: a chained forecast may hold the card
                # against that backend's one batch, so the whole build
                # cannot move onto the card under it.
                bounded_on_decoded_price = (
                    getattr(backend, "bounded_cuda", False)
                    and not previous_host.get("price_is_floor", True))
                if (int(price.need_bytes) > int(free)
                        or bounded_on_decoded_price):
                    raise InitializationMemoryRefused(
                        "Bounded CUDA preparation retains its state and working fields in host RAM: "
                        f"the array envelope needs {_gib(host_need)} and {_gib(host_available)} is available. "
                        "Starting would exhaust host memory before the prepared state can be sealed. "
                        "Free host memory or use a host with more available RAM.")
                # The whole build fits the card and the host cannot retain
                # it: the full-device preparation, which holds its arrays
                # on the card and needs no host envelope.  Refusing here
                # refused single-card runs 2.8.4 prepared (a 700x600x50
                # mapped build priced at 14.3 GiB on a 22 GiB card beside
                # 6 GiB of host RAM).
                route = ("full-device preparation: the host cannot retain "
                         "the bounded batches' array envelope and the whole "
                         "build fits the card's free memory")
                if getattr(backend, "bounded_cuda", False):
                    backend = CudaPreprocessBackend(
                        host_workers=getattr(backend, "host_workers", workers))
                if (selection.get("host_fit") or {}).get("route") != route:
                    import sys
                    print("CUDA preparation: the host has "
                          f"{_gib(host_available)} available for the "
                          f"{_gib(host_need)} bounded batches would retain "
                          "there, and the whole build "
                          f"({_gib(price.need_bytes)}) fits the card's "
                          f"{_gib(free)} free; preparing on the card",
                          file=sys.stderr)
                selection.pop("chunking", None)
                backend.selection = dict(
                    selection, device_fit=fit,
                    host_fit=dict(host_fit, route=route))
                return backend
            if not getattr(backend, "bounded_cuda", False):
                backend = BoundedCudaPreprocessBackend(
                    device_budget_bytes=budget,
                    source_staging_bytes=source_staging,
                    host_workers=getattr(backend, "host_workers", workers))
            else:
                backend.device_budget_bytes = budget
                backend.source_staging_bytes = source_staging
                backend.chunk_cells = max(1, (budget - source_staging) // 1024)
            fit = dict(fit, unchunked_need_bytes=int(price.need_bytes),
                       unchunked=price.record(), need_bytes=budget + context,
                       terms={"staging_pool": budget, "cuda_context": context},
                       phase="bounded CUDA batches",
                       phases={"bounded CUDA batches": budget + context},
                       basis="bounded kernel batches with host-retained completed arrays",
                       fits=True)
            backend.selection = dict(selection, device_fit=fit, host_fit=host_fit, chunking={
                "schema": "gpuwm-cuda-preparation-chunks-v1",
                "device_pool_budget_bytes": budget,
                "source_staging_bytes": source_staging,
                "minimum_batch_bytes": minimum,
                "chunk_cells": backend.chunk_cells,
                "retained_arrays": "host",
                "operators": "unchanged CUDA kernels; complete vertical columns"})
            if not selection.get("chunking"):
                import sys
                print(f"CUDA preparation: staging at most {_gib(budget)} on the card; "
                      "completed arrays stay on the host "
                      f"(whole-build estimate {_gib(price.need_bytes)})", file=sys.stderr)
            return backend
    if free is None or int(price.need_bytes) <= int(free):
        backend.selection = dict(selection, device_fit=fit)
        return backend
    # Wait for capacity without changing the selected arithmetic. A mapped
    # build may need only one bounded batch; its normal host-fit gate above
    # remains responsible for deciding whether that route is admissible.
    need = int(price.need_bytes)
    if price.route == "mapped":
        minimum = max(256 * 1024**2, int(getattr(price, "chunk_minimum_bytes", 0)))
        if minimum <= 1024**3:
            need = min(need, minimum + context + 256 * 1024**2)
    reading, waited = _wait_for_preparation_memory(backend, need, _reading, price=price)
    backend.selection = dict(selection, device_wait_seconds=(
        selection.get("device_wait_seconds", 0.0) + waited))
    return admit_preparation(backend, price, workers=workers, probe=reading)


def decide_preparation_device(requested: str, price, *, probe=None
                              ) -> tuple[str, dict | None]:
    """The same decision for a door that passes the backend by NAME.

    Returns ``("cuda" | "cpu", selection)``.  ``requested`` is ``cuda``
    or ``auto`` (``cpu`` is returned unchanged); the door must already
    hold CUDA for its own work, so auto here only chooses where the
    preparation's transforms run.
    """

    normalized = str(requested).strip().lower()
    from woof.local_gpu import no_local_gpu
    if no_local_gpu() and normalized in ("cuda", "auto"):
        refusal = ("GPUWM_NO_LOCAL_GPU forbids local CUDA preprocessing; "
                   "select backend='cpu' on this machine")
        if normalized == "cuda":
            raise ValueError(refusal)
        _announce_auto_cpu(refusal)
        return "cpu", {"requested": "auto", "backend": "cpu", "reason": refusal}
    if normalized == "cpu" or price is None:
        return normalized, None
    if callable(price):
        price = price()
    if normalized not in ("cuda", "auto"):
        raise ValueError("backend must be 'cuda', 'cpu', or 'auto'")
    free, total, _reading = _device_reading(None, probe)
    fit = preparation_device_fit(price, free, total)
    selection = {"requested": normalized, "device_fit": fit}
    if free is None or int(price.need_bytes) <= int(free):
        selection.update(backend="cuda", reason=(
            NAMED_BY_CALLER if normalized == "cuda" else
            "the CUDA preparation fits the card's free memory"
            if free is not None else "the card's free memory was not read"))
        return "cuda", selection
    reading, waited = _wait_for_preparation_memory(None, int(price.need_bytes),
                                                    _reading, price=price)
    name, selection = decide_preparation_device(normalized, price, probe=reading)
    selection["device_wait_seconds"] = waited
    return name, selection


def _sealed_distribution_preprocess_backends() -> tuple[str, ...] | None:
    """Return the exact backend inventory of an installed native package.

    The inventory is read from a document that has already been checked
    against its WHOLE schema
    (:func:`woof.runtime_manifest.validate_manifest`), not against the
    one key this function needs.  Reading a key at a time is how a
    manifest missing ``contract`` entirely passed the identity gate,
    ran a preparation for minutes, and then died here -- correctly, and
    far too late.  Same refusal, hoisted to the first read.
    """

    from woof.runtime_manifest import manifest_from_environment

    bound = manifest_from_environment()
    if bound is None:
        return None
    _path, payload = bound
    backends = payload["contract"]["platform"]["backends"]
    return tuple(backends)


def start_time_backend(backend):
    """The backend the START time is built on: the whole card when it fits.

    ``admit_preparation`` stages a mapped build over 1 GiB in bounded
    batches so a forecast can run beside the LATER forcing times.  The
    start time is built before the head exists, so no forecast sits beside
    it yet, and the bounded batches cost it about 40 s of per-batch host
    round trips on the HRRR CONUS grid (126.8 s against 86.4 s full-device,
    every prepared array byte-identical).  When the whole build's price
    fits the free bytes the admission just read, the start time takes the
    full-device backend; its caller moves the result to the host
    (:func:`move_start_result_to_host`) and releases the pool before the
    head is admitted, so the later bounded times and any forecast beside
    them see the card exactly as the bounded start left it.  Same CUDA
    kernels, same vertical-route record (shared with ``backend``) and the
    same scientific receipt; the route taken is a measurement under
    ``selection.chunking.start_time``.  Anything else returns ``backend``.
    """

    if not getattr(backend, "bounded_cuda", False):
        return backend
    selection = getattr(backend, "selection", None) or {}
    fit = selection.get("device_fit") or {}
    chunking = selection.get("chunking")
    need, free = fit.get("unchunked_need_bytes"), fit.get("free_bytes")
    if not isinstance(chunking, dict) or type(need) is not int or type(free) is not int:
        return backend
    if need > free:
        chunking["start_time"] = {
            "route": "bounded CUDA batches",
            "reason": f"the whole build needs {_gib(need)} and the card has {_gib(free)} free"}
        return backend
    full = CudaPreprocessBackend(host_workers=backend.host_workers)
    # One receipt: the start time's vertical geometries land in the
    # bounded backend's own route list, as they do when it builds them.
    full._vertical_routes = backend._vertical_routes
    full.selection = selection
    chunking["start_time"] = {
        "route": "full-device preparation, result moved to host before the head",
        "reason": f"the whole build needs {_gib(need)} and the card has {_gib(free)} free"}
    return full


def move_start_result_to_host(*objects) -> int:
    """Copy every CuPy array these objects hold to the host, in place.

    The full-device start time (:func:`start_time_backend`) leaves its met
    fields, state and base on the card; the bounded start leaves the same
    values in NumPy.  This turns the first into the second, attribute by
    attribute, so everything after the start (soil, head, boundary frames,
    the producer's reservation) runs on what the bounded start hands it.
    A device-to-host copy is exact.  Returns the bytes moved.
    """

    import dataclasses
    try:
        import cupy as cp
    except ImportError:          # no CUDA, nothing on a card
        return 0
    seen: set[int] = set()
    copies: dict[int, tuple[object, np.ndarray]] = {}
    moved = 0

    def host(value):
        nonlocal moved
        if isinstance(value, cp.ndarray):
            # One host copy per device array, so two names for one array
            # stay one array.
            if id(value) not in copies:
                moved += int(value.nbytes)
                copies[id(value)] = (value, value.get())
            return copies[id(value)][1]
        if isinstance(value, tuple) and not hasattr(value, "_fields"):
            converted = tuple(host(item) for item in value)
            return value if all(a is b for a, b in zip(converted, value)) else converted
        visit(value)
        return value

    def visit(value):
        if id(value) in seen or isinstance(value, (str, bytes, int, float, np.ndarray)):
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for key in list(value):
                converted = host(value[key])
                if converted is not value[key]:
                    value[key] = converted
        elif isinstance(value, list):
            for position, item in enumerate(value):
                converted = host(item)
                if converted is not item:
                    value[position] = converted
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for item in dataclasses.fields(value):
                current = getattr(value, item.name)
                converted = host(current)
                if converted is not current:
                    object.__setattr__(value, item.name, converted)
        elif hasattr(value, "__dict__") and type(value).__module__.startswith("woof."):
            for key, item in list(vars(value).items()):
                converted = host(item)
                if converted is not item:
                    object.__setattr__(value, key, converted)
            if hasattr(value, "_host_setup_state"):
                object.__setattr__(value, "_host_setup_state", True)

    # Gather every device array first and copy them on several threads,
    # each on its own stream: one thread's pageable copies moved 22 GB in
    # 5.8 s at HRRR CONUS.  A device-to-host copy is exact either way.
    gather: dict[int, object] = {}

    def collect(value):
        if isinstance(value, cp.ndarray):
            gather.setdefault(id(value), value)
            return value
        if isinstance(value, tuple) and not hasattr(value, "_fields"):
            for item in value:
                collect(item)
            return value
        walk(value, collect)
        return value

    walked: set[int] = set()

    def walk(value, each):
        if id(value) in walked or isinstance(value, (str, bytes, int, float, np.ndarray)):
            return
        walked.add(id(value))
        if isinstance(value, dict):
            for key in list(value):
                each(value[key])
        elif isinstance(value, list):
            for item in value:
                each(item)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for item in dataclasses.fields(value):
                each(getattr(value, item.name))
        elif hasattr(value, "__dict__") and type(value).__module__.startswith("woof."):
            for item in list(vars(value).values()):
                each(item)

    for value in objects:
        collect(value)
    if gather:
        from concurrent.futures import ThreadPoolExecutor
        device = cp.cuda.runtime.getDevice()

        def fetch(array):
            cp.cuda.Device(device).use()
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                host = array.get(stream=stream)
            stream.synchronize()
            return host

        cp.cuda.runtime.deviceSynchronize()
        items = sorted(gather.items(), key=lambda item: -int(item[1].nbytes))
        with ThreadPoolExecutor(max_workers=min(8, len(items))) as pool:
            for (key, array), copied in zip(items, pool.map(lambda item: fetch(item[1]), items)):
                moved += int(array.nbytes)
                copies[key] = (array, copied)
    for value in objects:
        visit(value)
    return moved


def release_backend_memory(backend) -> None:
    """Hand this backend's unreferenced blocks back to the device.

    Streaming ingest drops one forcing time's arrays before it builds the
    next, but CuPy's pool keeps every freed block, so the machine-visible
    footprint would stay pinned at the high-water mark of the very first
    time even though nothing is using it.  Asking the pool to release
    what nobody holds is what turns the Python-level release into a
    device-level one.

    ``gc.collect()`` runs first because a released :class:`DomainState`
    can be reachable only through a reference cycle, and a cycle-collected
    array is not freed until the collector runs.  Live arrays are never
    touched: :meth:`free_all_blocks` releases unreferenced blocks only.
    This is a memory operation with no numerical effect whatsoever.
    """

    # Best effort, by design.  This function exists to hand back memory
    # nobody is using; it decides nothing and computes nothing, so it must
    # never be the reason a preparation fails -- nor a reason anything
    # else behaves differently.  Both guards below are about that.
    #
    # A backend with no device pool has nothing to hand back: the CPU
    # backend's arrays are NumPy, freed by refcount, and a CPU stand-in
    # swapped in for cupy is the same story.  Returning before the
    # collector matters because gc.collect() is a WHOLE-PROCESS event --
    # charging every caller of a no-op release for a full collection is
    # a side effect on code that never asked for one.
    pools = []
    if getattr(backend, "name", None) == "cuda":
        if getattr(backend, "bounded_cuda", False):
            from woof.ingest.horiz import _cupy
            module = _cupy()
        else:
            module = backend.array_module
        pools = [getattr(module, accessor, None)
                 for accessor in ("get_default_memory_pool",
                                  "get_default_pinned_memory_pool")]
        pools = [accessor for accessor in pools if accessor is not None]
    if not pools:
        return
    # Only now, and only because it pays for itself: a released
    # DomainState can be reachable through a reference cycle, and a
    # cycle-collected array is not returned to the pool until the
    # collector runs.  This is what turns the Python-level release into
    # a device-level one.  No numerical effect whatsoever.
    gc.collect()
    for accessor in pools:
        accessor().free_all_blocks()


__all__ = [
    "CudaPreprocessBackend",
    "ParallelCpuPreprocessBackend",
    "PREPROCESS_RECEIPT_MEASUREMENTS",
    "PreparationDeviceRefused",
    "admit_preparation",
    "decide_preparation_device",
    "preprocess_identity",
    "preprocess_identity_matches",
    "preprocess_measurements",
    "preprocess_reports_identity",
    "preprocess_selection_identity",
    "release_backend_memory",
    "resolve_preprocess_backend",
]
