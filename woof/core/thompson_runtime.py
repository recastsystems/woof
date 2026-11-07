"""Runtime ownership for canonical classic-Thompson coefficient tables.

The table parser in :mod:`woof.core.thompson_contract` proves the bytes on
disk.  This module closes the second half of that boundary: it checks the
complete in-memory inventory, uploads every record without a precision or
layout conversion, verifies a device round trip, and retains one process-local
owner per CUDA device. The production ``mp_physics=8`` adapter loads this
owner before its simultaneous source and fallout calls.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import threading
import json
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from woof.core.thompson_contract import (
    AUXILIARY_TABLE_RECORDS,
    CLASSIC_TABLE_ASSETS,
    GENERATED_TABLE_FILES,
    TABLE_SETS_BY_VERSION,
    ClassicTableSet,
    TableAsset,
    TableRecord,
    load_validated_classic_tables,
)


RAIN_SNOW_TABLE_NAMES = (
    "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
    "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
    "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
)
RAIN_GRAUPEL_TABLE_NAMES = (
    "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
)
RAIN_FREEZING_TABLE_NAMES = (
    "tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz",
)


@dataclass(frozen=True)
class DeviceClassicColdSourceTables:
    """Canonical device-table groups consumed by the fused cold source."""

    ice_deposition_partition: object
    ice_to_snow_mass: object
    ice_to_snow_number: object
    rain_snow_tables: tuple[object, ...]
    rain_graupel_tables: tuple[object, ...]
    rain_freezing_tables: tuple[object, ...]
    rain_cloud_efficiency: object
    cloud_freezing_tables: tuple[object, object]
    identity_sha256: str


def _classic_records(version: str = "wrf_461") -> tuple[TableRecord, ...]:
    generated_files = (GENERATED_TABLE_FILES if version == "wrf_461"
                       else TABLE_SETS_BY_VERSION[version][0])
    records = tuple(
        record
        for group in generated_files.values()
        for record in group
    ) + AUXILIARY_TABLE_RECORDS
    names = tuple(record.name for record in records)
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate records in the Thompson table contract")
    return records


def _payload_sha256(value: np.ndarray) -> str:
    # A Fortran-ordered array's transpose is C-contiguous with the same
    # bytes in the same order, so it is hashed in place instead of copied.
    if value.flags.f_contiguous:
        return hashlib.sha256(value.T).hexdigest()
    return hashlib.sha256(value.tobytes(order="F")).hexdigest()


def _synchronize(backend) -> None:
    """Wait for the uploads this thread queued, on the stream it queued them.

    That is the CURRENT stream.  The legacy NULL stream was synchronized
    here, which is the current stream on the resident road and waits for
    nothing a split slab's non-blocking stream queued.
    """
    cuda = getattr(backend, "cuda", None)
    current = getattr(cuda, "get_current_stream", None)
    if current is not None:
        current().synchronize()
        return
    stream = getattr(cuda, "Stream", None)
    null = getattr(stream, "null", None)
    synchronize = getattr(null, "synchronize", None)
    if synchronize is not None:
        synchronize()


def _synchronize_device(backend) -> None:
    """Wait for every stream on the current card (a replaced table's readers)."""
    cuda = getattr(backend, "cuda", None)
    device = getattr(cuda, "Device", None)
    if device is not None and hasattr(device(), "synchronize"):
        device().synchronize()


def _pinned_staging(backend, nbytes: int):
    """One page-locked host buffer for the round trip, or None.

    Copying into page-locked memory is several times faster than CuPy's
    pageable copy, which also allocates a fresh host array per table.
    """
    allocate = getattr(getattr(backend, "cuda", None),
                       "alloc_pinned_memory", None)
    if allocate is None:
        return None
    return allocate(nbytes)


def _to_host_fortran(backend, value, *, staging=None) -> np.ndarray:
    if staging is not None and hasattr(value, "get"):
        result = np.ndarray(value.shape, dtype=np.float64,
                            buffer=staging, order="F")
        value.get(out=result, blocking=True)
        return result
    asnumpy = getattr(backend, "asnumpy", None)
    if asnumpy is None:
        result = np.array(value, dtype=np.float64, order="F", copy=True)
    else:
        try:
            result = asnumpy(value, order="F")
        except TypeError:
            result = asnumpy(value)
        result = np.array(result, dtype=np.float64, order="F", copy=False)
    return result


@dataclass(frozen=True)
class DeviceTableSet:
    """One verified, immutable mapping of canonical tables on a device."""

    root: Path
    arrays: Mapping[str, object]
    payload_bytes: int
    array_sha256: Mapping[str, str]
    identity_json: str
    identity_sha256: str
    device_id: int | None
    upload_seconds: float
    verification_seconds: float
    roundtrip_verified: bool

    def __getattr__(self, name: str):
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def identity(self) -> dict[str, object]:
        """Return a detached JSON-compatible restart identity."""
        return json.loads(self.identity_json)


@dataclass(frozen=True)
class DeviceClassicTableSet(DeviceTableSet):
    """Canonical classic tables, including the fused cold-source ABI."""

    @property
    def cold_source_tables(self) -> DeviceClassicColdSourceTables:
        """Return the exact ordered groups for the fused cold-source ABI."""
        return DeviceClassicColdSourceTables(
            ice_deposition_partition=self.arrays["tpi_ide"],
            ice_to_snow_mass=self.arrays["tps_iaus"],
            ice_to_snow_number=self.arrays["tni_iaus"],
            rain_snow_tables=tuple(
                self.arrays[name] for name in RAIN_SNOW_TABLE_NAMES),
            rain_graupel_tables=tuple(
                self.arrays[name] for name in RAIN_GRAUPEL_TABLE_NAMES),
            rain_freezing_tables=tuple(
                self.arrays[name] for name in RAIN_FREEZING_TABLE_NAMES),
            rain_cloud_efficiency=self.arrays["t_Efrw"],
            cloud_freezing_tables=(
                self.arrays["tpi_qcfz"], self.arrays["tni_qcfz"]),
            identity_sha256=self.identity_sha256,
        )


def _validate_device_table(prefix, name, value, shape, payload_bytes):
    """Check the uploaded layout, dtype and payload against the host contract."""
    if value.shape != shape:
        raise RuntimeError(
            f"{prefix} device table {name} has shape {value.shape}; "
            f"expected {shape}")
    if np.dtype(value.dtype) != np.dtype(np.float64):
        raise RuntimeError(
            f"{prefix} device table {name} has dtype {value.dtype}; "
            "expected float64")
    if not value.flags.f_contiguous:
        raise RuntimeError(
            f"{prefix} device table {name} is not Fortran-contiguous")
    if value.nbytes != payload_bytes:
        raise RuntimeError(
            f"{prefix} device table {name} has {value.nbytes} bytes; "
            f"expected {payload_bytes}")


def _upload_table_set(
        table_set: ClassicTableSet, backend, *,
        records: tuple[TableRecord, ...],
        expected_assets: tuple[TableAsset, ...],
        device_id: int | None,
        verify_roundtrip: bool,
        ) -> DeviceClassicTableSet:
    """Upload a table set after enforcing an exact supplied contract.

    ``records`` and ``expected_assets`` are explicit so small CPU unit tests can
    exercise the same validator without allocating the 380 MB production set.
    The public loader below always supplies the canonical WRF contract.
    """
    if table_set.assets != expected_assets:
        raise ValueError("Thompson table-set asset identity is not canonical")

    expected = {record.name: record for record in records}
    actual_names = set(table_set.arrays)
    expected_names = set(expected)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"Thompson table inventory mismatch: missing={missing}, extra={extra}")

    payload_bytes = 0
    for name, record in expected.items():
        value = table_set.arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Thompson host table {name} must be a NumPy array")
        if value.shape != record.shape:
            raise ValueError(
                f"Thompson host table {name} has shape {value.shape}; "
                f"expected {record.shape}")
        if value.dtype != np.dtype(np.float64):
            raise TypeError(
                f"Thompson host table {name} has dtype {value.dtype}; "
                "expected float64")
        if not value.flags.f_contiguous:
            raise ValueError(f"Thompson host table {name} is not Fortran-contiguous")
        if value.flags.writeable:
            raise ValueError(f"Thompson host table {name} is not immutable")
        if value.nbytes != record.payload_bytes:
            raise RuntimeError(
                f"Thompson host table {name} has {value.nbytes} bytes; "
                f"expected {record.payload_bytes}")
        payload_bytes += value.nbytes
    # hashlib releases the GIL while it hashes, so the arrays hash in parallel.
    with ThreadPoolExecutor(max_workers=4) as pool:
        host_hashes = dict(zip(expected, pool.map(
            _payload_sha256,
            (table_set.arrays[name] for name in expected))))

    upload_start = perf_counter()
    device_arrays = {
        name: backend.asarray(
            table_set.arrays[name], dtype=backend.float64, order="F")
        for name in expected
    }
    _synchronize(backend)
    upload_seconds = perf_counter() - upload_start

    for name, record in expected.items():
        _validate_device_table("Thompson", name, device_arrays[name],
                               record.shape, record.payload_bytes)

    verification_seconds = 0.0
    if verify_roundtrip:
        verify_start = perf_counter()
        staging = _pinned_staging(backend, max(
            record.payload_bytes for record in expected.values()))
        for name in expected:
            returned = _to_host_fortran(
                backend, device_arrays[name], staging=staging)
            if returned.shape != table_set.arrays[name].shape:
                raise RuntimeError(
                    f"Thompson device round trip changed {name} shape")
            if not returned.flags.f_contiguous:
                raise RuntimeError(
                    f"Thompson device round trip changed {name} layout")
            # The returned bytes must equal the uploaded ones word for word
            # (signed zeros and NaN payloads included); the digest is only
            # computed to name a mismatch.
            if not np.array_equal(returned.T.reshape(-1).view(np.uint8),
                                  table_set.arrays[name].T.reshape(-1).view(
                                      np.uint8)):
                returned_hash = _payload_sha256(returned)
                raise RuntimeError(
                    f"Thompson device round trip changed {name}: "
                    f"{returned_hash} != {host_hashes[name]}")
        _synchronize(backend)
        verification_seconds = perf_counter() - verify_start

    identity_json = json.dumps(
        table_set.identity, sort_keys=True, separators=(",", ":"))
    identity_sha256 = hashlib.sha256(identity_json.encode("ascii")).hexdigest()
    return DeviceClassicTableSet(
        root=table_set.root,
        arrays=MappingProxyType(device_arrays),
        payload_bytes=payload_bytes,
        array_sha256=MappingProxyType(host_hashes),
        identity_json=identity_json,
        identity_sha256=identity_sha256,
        device_id=device_id,
        upload_seconds=upload_seconds,
        verification_seconds=verification_seconds,
        roundtrip_verified=verify_roundtrip,
    )


def _graupel_coefficient():
    from woof.physics_params import active
    parameters = active()
    if parameters is not None and "thompson.graupel_fall" in parameters.changed():
        return np.float32(parameters.value("thompson.graupel_fall"))
    return None


def _regenerate_graupel(result, backend, coefficient):
    """Integrate WRF's collision table on the card once, before publication.

    No host table generator, output transfer or timestep synchronization.
    Each entry preserves WRF's complete ordered 100 by 100 quadrature.
    """
    from woof.core.kernels import load_module, module_source
    start = perf_counter()
    module = load_module("thompson_collision")
    nr = backend.empty(37*37*100, dtype=np.float64)
    ng = backend.empty_like(nr)
    pairs = backend.empty(50000, dtype=np.float64)
    arrays = dict(result.arrays)
    outputs = [backend.empty_like(arrays[name], order="F")
               for name in RAIN_GRAUPEL_TABLE_NAMES]
    module.get_function("collision_distributions")(
        ((nr.size+127)//128,), (128,), (nr, ng))
    module.get_function("collision_pairs")(
        (79,), (128,), (coefficient, pairs))
    count = 37**4
    module.get_function("collision_integrate")(
        ((count+127)//128,), (128,), (nr, ng, pairs, *outputs))
    # Publish on every domain stream only after this initialization stream
    # finishes. Temporaries stay owned until that boundary.
    _synchronize(backend)
    arrays.update(zip(RAIN_GRAUPEL_TABLE_NAMES, outputs))
    identity = json.loads(result.identity_json)
    identity["generated_collision"] = {
        "native_pin": "d66e442fccc04111067e29274c9f9eaccc3cef28",
        "algorithm_sha256": hashlib.sha256(
            module_source("thompson_collision").encode("utf-8")).hexdigest(),
        "av_g_float32_hex": float(coefficient).hex(),
        "records": list(RAIN_GRAUPEL_TABLE_NAMES),
        "quadrature": [100, 100], "backend": "CUDA",
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    hashes = {name: value for name, value in result.array_sha256.items()
              if name not in RAIN_GRAUPEL_TABLE_NAMES}
    return replace(result, arrays=MappingProxyType(arrays),
                   array_sha256=MappingProxyType(hashes),
                   identity_json=encoded,
                   identity_sha256=hashlib.sha256(encoded.encode("ascii")).hexdigest(),
                   upload_seconds=result.upload_seconds+perf_counter()-start)


_DEVICE_CACHE: dict[tuple[Path, int, int | None], DeviceClassicTableSet] = {}
_DEVICE_CACHE_LOCK = threading.RLock()


def load_classic_device_tables(
        path: str | Path, *, backend=None,
        verify_roundtrip: bool = True,
        cache: bool = True,
        version: str = "wrf_461",
        ) -> DeviceClassicTableSet:
    """Validate and upload every canonical classic-Thompson WRF table once.

    ``version`` is RunConfig.thompson_version and picks the table set
    (woof.core.thompson_contract.TABLE_SETS_BY_VERSION): the WRF v4.6.1
    set by default, the operational WRF 3.9 fork's for ``"wrf_39_noaa"``.
    """
    if backend is None:
        import cupy as backend

    if version not in TABLE_SETS_BY_VERSION:
        raise ValueError(
            f"no Thompson table set for thompson_version={version!r}")
    coefficient = _graupel_coefficient()
    if coefficient is not None and version != "wrf_461":
        raise ValueError("graupel coefficient requires the WRF 4.6.1 collision quadrature")
    root = Path(path).resolve()
    cuda = getattr(backend, "cuda", None)
    device = getattr(cuda, "Device", None)
    device_id = int(device().id) if device is not None else None
    key = ((root, id(backend), device_id) if version == "wrf_461"
           else (root, id(backend), device_id, version))
    if coefficient is not None:
        key = (*key, float(coefficient).hex())
    # A split domain's slabs step on their own non-blocking streams, and the
    # first microphysics call of each loads these tables at once.
    # Each uploaded its own copy on its own stream,
    # synchronized the legacy NULL stream (which waits for none of them),
    # and published it; a slab arriving after a publish took another slab's
    # copy with no ordering against its upload, and a copy replaced in the
    # cache went back to its stream's memory pool while another slab still
    # read it -- the defect measured on RRTMGP's tables (rrtmgp._upload_once).
    # So the load is serialized, the upload is waited on its own stream, and
    # a published copy is never replaced while it can be read.
    with _DEVICE_CACHE_LOCK:
        existing = _DEVICE_CACHE.get(key) if cache else None
        if existing is not None and (
                existing.roundtrip_verified or not verify_roundtrip):
            return existing

        # This re-reads from canonical, content-addressed files.  A caller cannot
        # smuggle a manually constructed ClassicTableSet across the trust boundary.
        table_set = load_validated_classic_tables(root, version)
        result = _upload_table_set(
            table_set, backend,
            records=_classic_records(version),
            expected_assets=TABLE_SETS_BY_VERSION[version][1],
            device_id=device_id,
            verify_roundtrip=verify_roundtrip,
        )
        if coefficient is not None:
            result = _regenerate_graupel(result, backend, coefficient)
        if cache:
            if existing is not None:
                # A copy being replaced (a verified load asked for after
                # an unverified one) may still be read on another stream.
                _synchronize_device(backend)
            _DEVICE_CACHE[key] = result
        return result


__all__ = [
    "DeviceClassicColdSourceTables",
    "DeviceClassicTableSet",
    "RAIN_FREEZING_TABLE_NAMES",
    "RAIN_GRAUPEL_TABLE_NAMES",
    "RAIN_SNOW_TABLE_NAMES",
    "load_classic_device_tables",
]
