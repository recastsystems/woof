"""Published MPAS mesh loading and topology/geometry validation.

MPAS writes connectivity as one-based indices with zero denoting a missing
entry.  The authority representation in this package is zero-based with
``-1`` padding.  Padding is canonicalized from the corresponding count field
because some otherwise-valid MPAS mesh files repeat the final used entry in
unused ``cellsOnCell`` and ``edgesOnCell`` slots.

The topology conventions checked here are the conventions consumed by
``atm_compute_signs`` in frozen MPAS-A v8.2.3:
``src/core_atmosphere/mpas_atm_core.F:1137-1224``.  The metric relationships
are the MPAS C-grid discretization represented by the mesh-spec fields.

A REGIONAL (limited-area) mesh -- a bounded disk culled from a parent sphere,
carrying the ``bdyMaskCell/Edge/Vertex`` triple WITH A NONEMPTY BOUNDARY ZONE
-- is validated against the conventions measured on native
MPAS-Limited-Area culls of two published meshes (2026-08-25, real bytes,
both culls 100 percent).  The zone, not the schema, is what makes a mesh
regional: native MPAS-A writes the same triple all-zero into a GLOBAL mesh's
static file and the unified ``rw_mpas_static`` follows that convention on
every static this project generates, so classifying on presence alone refused
both the published x1.40962 and every generated global mesh as a corrupt cull
(measured 2026-08-26; see :meth:`Mesh.validate`).  The conventions:

* masks are integers 0 (interior) through 7 (outermost ring), edge/vertex
  masks are the MINIMUM of their present cells' masks, neighbouring cell
  masks never differ by more than 1, and every ring-``k`` cell touches at
  least one ring-``k-1`` cell.  That last one used to be "ring populations
  grow outward", which holds on a uniform mesh and fails on a variable-
  resolution one -- see :func:`regional_ring_shell_errors` for the three
  culls that measured it;
* absent-element sentinels (file zeros, canonical ``-1``) appear only on
  ring-7 elements and only in the five arrays of
  :data:`REGIONAL_SENTINEL_ARRAYS`; every other connectivity array is
  complete because the cull keeps every edge and vertex of a kept cell
  (``nEdgesOnEdge`` is NOT shrunk -- sentinels sit inside the declared row);
* the Euler characteristic ``nCells - nEdges + nVertices`` is 1 (a disk),
  not the closed sphere's 2, and the whole-sphere area identities do not
  apply -- the local kite/area identities and the arc-length checks still do.

Carrying the triple is NOT what makes a mesh regional; carrying a boundary
ZONE is.  MPAS writes an all-zero ``bdyMask`` triple on a global mesh (every
element interior, no ring at all), and the unified ``rw_mpas_static`` writer
follows that convention, so every static this project GENERATES ships the
triple.  Presence alone therefore classified a closed sphere as a disk and
refused it for having Euler characteristic 2 and empty rings 1..7 -- which is
exactly what a global mesh is.  MEASURED 2026-08-26 on the generated graded
mesh ``v20.80.151649``: bound clean, then refused at load by that rule, as
would every generated-static row (``u96.64002``, every graded row).  The test
is a NONZERO mask value, and it is the whole difference; an INCOMPLETE triple
stays refused on presence, because a half-written triple cannot say which
sentinels are boundary and which are corruption.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import re
import struct
from typing import Any, Callable, Iterator, Mapping

import numpy as np
from netCDF4 import Dataset
from numpy.typing import NDArray

from .errors import MeshValidationError
from .host_parallel import run_blocks, threads


_COUNTED_CONNECTIVITY = {
    "cellsOnCell": "nEdgesOnCell",
    "edgesOnCell": "nEdgesOnCell",
    "verticesOnCell": "nEdgesOnCell",
    "edgesOnEdge": "nEdgesOnEdge",
    "advCellsForEdge": "nAdvCellsForEdge",
}

_CORE_CONNECTIVITY = {
    "cellsOnCell",
    "edgesOnCell",
    "verticesOnCell",
    "cellsOnEdge",
    "verticesOnEdge",
    "cellsOnVertex",
    "edgesOnVertex",
    "edgesOnEdge",
}

#: The mask triple a regional (limited-area) cull adds.  All three or none:
#: a partial triple cannot say which sentinels are boundary and which are
#: corruption, so it is refused rather than guessed at.
REGIONAL_BOUNDARY_MASK_NAMES = ("bdyMaskCell", "bdyMaskEdge", "bdyMaskVertex")

#: Rings 1..7, outermost ring 7 touching the cut, 0 interior.  Measured on
#: the native culls; MPAS-A v8.4.1 stages its specified/relaxation zones on
#: exactly these seven rings.
REGIONAL_BOUNDARY_ZONE_WIDTH = 7


def _regional_boundary_zone_present(arrays: Mapping[str, Any]) -> bool:
    """True when the complete mask triple carries at least one ring element.

    THE BREAKAGE THIS PREVENTS (measured 2026-08-26, ``v20.80.151649`` on
    the proving RTX 5090): an all-zero triple is MPAS's global convention and the
    unified ``rw_mpas_static`` writes it on every generated static, so
    classifying on PRESENCE refused every generated global mesh at load --
    "not a bounded disk: nCells-nEdges+nVertices = 2, not 1" and "rings
    [1..7] are empty", which is the correct description of a sphere.
    """

    if not all(name in arrays for name in REGIONAL_BOUNDARY_MASK_NAMES):
        return False
    return any(
        bool(np.any(np.asarray(arrays[name]) != 0))
        for name in REGIONAL_BOUNDARY_MASK_NAMES
    )

#: The only connectivity arrays in which a native cull stores an
#: absent-element sentinel (file zero, canonical -1), and only on ring-7
#: rows.  ``edgesOnCell``, ``verticesOnCell`` and ``verticesOnEdge`` are
#: complete by construction -- the cull keeps every edge and vertex of a
#: kept cell -- so a sentinel there is a lost element, never a boundary.
REGIONAL_SENTINEL_ARRAYS = (
    "cellsOnCell",
    "cellsOnEdge",
    "edgesOnEdge",
    "cellsOnVertex",
    "edgesOnVertex",
)

#: Ceiling on the RELATIVE half of the arc-length tolerance.  Eight binary32
#: epsilons: a stored binary32 length rounds by at most half an epsilon, so
#: this carries sixteen such roundings and nothing wider.  It is a ceiling
#: rather than the value, so a binary64 mesh keeps its own tighter bound.
_SPHERICAL_ARC_MAX_RTOL = 8.0 * float(np.finfo(np.float32).eps)

LONGITUDE_TRIG_EQUIVALENCE_ATOL = 8.0 * np.finfo(np.float64).eps
GRID_ROTATE_SOURCE_SHA256 = (
    "2be1c67cd2700ffd65b41f241c8858c0e24ca2b67bc7655465ef3807ab654d36"
)
GRID_ROTATE_SOURCE_HEAD = "4b5c11b4be471498da36a2637ad1cf49962b3d05"
GRID_ROTATE_DEFAULT_REAL_PI = float(
    np.float32(2.0) * np.arcsin(np.float32(1.0))
)
GRID_ROTATE_PI_DELTA = GRID_ROTATE_DEFAULT_REAL_PI - float(np.pi)
GRID_ROTATE_CONVERT_XL_EPS = float(np.float32(1.0e-10))
GRID_ROTATE_PRODUCER_MAX_ULP_ERROR = 2.0
GRID_ROTATE_LATITUDE_MAX_ULP_ERROR = 256.0
GRID_ROTATE_CORRECTED_ATAN2_ATOL = 16.0 * np.finfo(np.float64).eps
_BINARY64_MESH_RTOL = 5.0e-10
_GEOMETRY_COORDINATES = tuple(
    f"{component}{entity}"
    for entity in ("Cell", "Edge", "Vertex")
    for component in ("x", "y", "z", "lat", "lon")
)
_TOPOLOGY_WITNESSES = (
    "nEdgesOnCell",
    "nEdgesOnEdge",
    "cellsOnCell",
    "edgesOnCell",
    "verticesOnCell",
    "cellsOnEdge",
    "verticesOnEdge",
    "cellsOnVertex",
    "edgesOnVertex",
    "edgesOnEdge",
    "indexToCellID",
    "indexToEdgeID",
    "indexToVertexID",
)


def _camel_to_snake(name: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _plain_attribute(value: Any) -> Any:
    """Detach a netCDF attribute from its open Dataset."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


_NetCDFData = tuple[
    dict[str, int],
    dict[str, Any],
    dict[str, NDArray[Any]],
    dict[str, tuple[str, ...]],
    dict[str, dict[str, Any]],
]


_NETCDF_COPY_MIN_BYTES = 1 << 20


def _copy_netcdf_array(value: NDArray[Any]) -> NDArray[Any]:
    return np.array(value, copy=True)


def _read_netcdf(
    path: Path,
    *, exclude_variables: frozenset[str] = frozenset(),
) -> _NetCDFData:
    dimensions: dict[str, int] = {}
    attributes: dict[str, Any] = {}
    arrays: dict[str, NDArray[Any]] = {}
    variable_dimensions: dict[str, tuple[str, ...]] = {}
    variable_attributes: dict[str, dict[str, Any]] = {}

    # netCDF-C calls stay on the caller. Only independent NumPy copies overlap
    # the next variable read. One copy worker avoids bandwidth contention with
    # decoding; at most two pending variables retain their read buffers.
    parallel = threads() > 1
    pending: deque[tuple[str, Future[NDArray[Any]]]] = deque()

    def finish_one() -> None:
        name, future = pending[0]
        arrays[name] = future.result()
        pending.popleft()

    def finish_all() -> None:
        while pending:
            finish_one()

    # Whole numeric variables of a classic file are read from their header offsets on every host-copy thread
    # (woof.hex.classic_netcdf: the same arrays netCDF4 returns); the rest go through netCDF4 as before.
    classic = None
    if parallel and os.environ.get("WOOF_HEX_CLASSIC_READ", "").strip().lower() not in ("off", "0", "no"):
        from .classic_netcdf import extents
        try:
            classic = extents(path)
        except (OSError, ValueError, KeyError, struct.error):
            classic = None

    executor = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="hex-netcdf-copy")
                if parallel else nullcontext(None))
    with executor as pool:
        with Dataset(path, mode="r") as dataset:
            try:
                dataset.set_auto_mask(False)
                dimensions.update({name: len(dim) for name, dim in dataset.dimensions.items()})
                attributes.update(
                    {name: _plain_attribute(getattr(dataset, name)) for name in dataset.ncattrs()}
                )
                read_ahead: dict[str, NDArray[Any]] = {}
                if classic is not None:
                    from .classic_netcdf import read_variables, readable
                    table, records = classic
                    selected = [name for name, variable in dataset.variables.items()
                                if name not in exclude_variables and name in table
                                and readable(table[name], records, variable.ncattrs())
                                and tuple(table[name].shape) == tuple(variable.shape)
                                and np.dtype(table[name].dtype).newbyteorder("=") == variable.dtype]
                    read_ahead = read_variables(path, selected, table, workers=threads())
                for name, variable in dataset.variables.items():
                    if name in exclude_variables:
                        continue
                    if name in read_ahead:
                        arrays[name] = read_ahead.pop(name)
                        variable_dimensions[name] = tuple(variable.dimensions)
                        variable_attributes[name] = {
                            attr: _plain_attribute(getattr(variable, attr))
                            for attr in variable.ncattrs()
                        }
                        continue
                    if len(pending) >= 2:
                        finish_one()
                    source = variable[...]
                    if (pool is not None and source.nbytes >= _NETCDF_COPY_MIN_BYTES
                            and not source.dtype.hasobject):
                        # Reserve the original insertion order until the owned
                        # copy is ready. The source never escapes this function.
                        arrays[name] = source
                        pending.append((name, pool.submit(_copy_netcdf_array, source)))
                    else:
                        arrays[name] = _copy_netcdf_array(source)
                    variable_dimensions[name] = tuple(variable.dimensions)
                    variable_attributes[name] = {
                        attr: _plain_attribute(getattr(variable, attr))
                        for attr in variable.ncattrs()
                    }
            except BaseException:
                # An earlier copy failure wins over a later read/attribute
                # failure. Drain before Dataset.__exit__, retaining its normal
                # close behavior even when a read or copy failed.
                finish_all()
                raise
            finish_all()

    return dimensions, attributes, arrays, variable_dimensions, variable_attributes


def _looks_like_connectivity(name: str, array: NDArray[Any]) -> bool:
    if name in _CORE_CONNECTIVITY:
        return True
    if array.dtype.kind not in "iu" or name.startswith("indexTo"):
        return False
    if re.match(r"^n[A-Z]", name):
        return False
    lowered = name.lower()
    return any(
        token in lowered
        for token in (
            "cellson",
            "cellsfor",
            "edgeson",
            "edgesfor",
            "verticeson",
            "verticesfor",
        )
    )


def _canonicalize_connectivity(arrays: dict[str, NDArray[Any]]) -> tuple[str, ...]:
    converted: list[str] = []
    for name, source in tuple(arrays.items()):
        if not _looks_like_connectivity(name, source):
            continue
        target = np.empty_like(source, dtype=np.int64)

        def convert(block: slice) -> None:
            # Cast before subtracting, including unsigned stored indices.
            np.subtract(source[block], 1, out=target[block], dtype=np.int64,
                        casting="unsafe")

        if source.ndim:
            run_blocks(convert, source.shape[0], elements=source.size, worker_limit=4)
        else:
            target[...] = np.asarray(source, dtype=np.int64) - 1
        arrays[name] = target
        converted.append(name)

    for name, count_name in _COUNTED_CONNECTIVITY.items():
        if name not in arrays or count_name not in arrays:
            continue
        array = arrays[name]
        counts = np.asarray(arrays[count_name], dtype=np.int64)
        if array.ndim != 2 or counts.shape != (array.shape[0],):
            continue
        slots = np.arange(array.shape[1])[None, :]

        def pad(block: slice) -> None:
            padding = slots >= counts[block, None]
            array[block][padding] = -1

        run_blocks(pad, array.shape[0], elements=array.size, worker_limit=4)

    return tuple(sorted(converted))


def _spherical_arc_lengths(
    coordinates: NDArray[np.float64], endpoints: NDArray[np.int64],
    radius: float, *, clip_missing: bool = False,
) -> NDArray[np.float64]:
    """Keep each row's original NumPy arithmetic, with bounded worker gathers.

    The cold mesh loader checks millions of edges. Gathering both full
    coordinate tables before computing cross products serializes that work
    and allocates several full-size temporaries. Blocks contain whole rows,
    so the three-component sums retain their order and validation tolerances.
    """
    result = np.empty(endpoints.shape[0], dtype=np.float64)

    def measure(block: slice) -> None:
        pairs = endpoints[block]
        if clip_missing:
            pairs = np.clip(pairs, 0, None)
        first = coordinates[pairs[:, 0]]
        second = coordinates[pairs[:, 1]]
        result[block] = radius * np.arctan2(
            np.linalg.norm(np.cross(first, second), axis=1),
            np.sum(first * second, axis=1),
        )

    run_blocks(measure, len(endpoints), elements=endpoints.shape[0] * 6)
    return result


def _spherical_coordinate_checks(
    xyz: NDArray[np.float64], latitude: NDArray[np.float64],
    longitude: NDArray[np.float64], radius: float, metric_rtol: float,
    suffix: str,
) -> tuple[NDArray[np.float64] | None, list[str]]:
    """Validate complete coordinate rows without serial whole-mesh trig.

    Norms reduce only the original three components within each row. Each
    coordinate and trig expression keeps its original operation order.
    Boolean conjunction across blocks preserves the original error order,
    including the non-finite check's precedence over geometric checks.
    """
    normalized = np.empty_like(xyz)
    outcomes: list[tuple[bool, bool, bool]] = []

    def check(block: slice) -> None:
        points, lat, lon = xyz[block], latitude[block], longitude[block]
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lat + lon)):
            outcomes.append((False, False, False))
            return
        norms = np.linalg.norm(points, axis=1)
        on_radius = bool(np.allclose(norms, radius, rtol=metric_rtol, atol=0.0))
        expected = radius * np.stack(
            (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)),
            axis=1,
        )
        consistent = bool(np.allclose(
            points, expected, rtol=metric_rtol, atol=radius * metric_rtol,
        ))
        normalized[block] = points / norms[:, None]
        outcomes.append((True, on_radius, consistent))

    run_blocks(check, len(xyz), elements=len(xyz) * 12)
    if not all(result[0] for result in outcomes):
        return None, [f"{suffix} coordinates contain non-finite values"]
    errors = []
    if not all(result[1] for result in outcomes):
        errors.append(f"{suffix} Cartesian coordinates are not on sphere_radius")
    if not all(result[2] for result in outcomes):
        errors.append(f"{suffix} Cartesian and lat/lon coordinates disagree")
    return normalized, errors


def _bad_examples(indices: NDArray[Any], limit: int = 5) -> str:
    flat = np.asarray(indices).ravel()
    return ", ".join(str(int(index)) for index in flat[:limit])


def _sha256_file(path: Path) -> str:
    # The real SHA-256, computed once per file version (woof.hex.file_digest):
    # the 15 km grid and static were hashed again on every mesh load
    # (4.7 s of a warm host preparation, 8-card RTX PRO 6000 host 2026-10-10).
    from .file_digest import sha256_file as file_version_sha256
    return file_version_sha256(path)


def _sha256_array(value: Any) -> str:
    """Hash an array with its dtype and shape, not just its raw payload."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    # Stable mesh carriers need no additional whole-payload bytes allocation.
    # Keep the historical representation for object arrays, which cannot be
    # viewed as uint8; mesh coordinate admission permits binary64 only.
    payload = (array.tobytes(order="C") if array.dtype.hasobject else
               memoryview(array.reshape(-1).view(np.uint8)))
    digest.update(payload)
    return digest.hexdigest()


_PARALLEL_ARRAY_DIGEST_BYTES = 16 * 1024 * 1024


@contextmanager
def _array_digest_items(values: Mapping[str, np.ndarray]) -> Iterator[Iterator[tuple[str, str]]]:
    """Hash independent stable carriers, retaining each complete SHA stream.

    Reuse the bounded authority-fingerprint approach from cuda_driver. Ordered
    results preserve first-refusal order, including comparisons between hashes.
    Exiting the context joins every worker before a caller can roll back arrays.
    Four workers bound memory-bandwidth contention with the input file hashes.
    """
    workers = min(4, len(values), threads())
    large = sum(value.nbytes for value in values.values()) >= _PARALLEL_ARRAY_DIGEST_BYTES
    error_policy = np.geterr()

    def record(item: tuple[str, np.ndarray]) -> tuple[str, str]:
        name, value = item
        with np.errstate(**error_policy):
            return name, _sha256_array(value)

    if workers <= 1 or not large:
        yield map(record, values.items())
    else:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="hex-mesh-array-digest") as pool:
            yield pool.map(record, values.items())


class _ValidationChecks:
    """Run :meth:`Mesh.validate`'s independent checks on host threads.

    Checks start in submission order (the executor's queue is first in, first
    out) under the caller's NumPy error policy; only the caller waits on a
    check, so no check waits on another.  With one host-copy thread
    (``WOOF_HEX_HOST_COPY_THREADS=1``) every check runs on the caller at
    submission and its result or exception is kept for the caller to take in
    order.
    """

    def __init__(self) -> None:
        workers = min(12, threads())
        self._policy = np.geterr()
        self._pool = (ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-mesh-validate")
                      if workers > 1 else None)

    def _run(self, function: Callable[..., Any], args: tuple[Any, ...],
             kwargs: dict[str, Any]) -> Any:
        with np.errstate(**self._policy):
            return function(*args, **kwargs)

    def submit(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        if self._pool is not None:
            return self._pool.submit(self._run, function, args, kwargs)
        future: Future[Any] = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future

    def __enter__(self) -> "_ValidationChecks":
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)


def _joined_rows(blocks: Mapping[int, NDArray[Any]]) -> NDArray[Any]:
    """Pieces found block by block (row indices, selected words), joined in row order."""

    if not blocks:
        return np.empty(0, dtype=np.intp)
    parts = [blocks[start] for start in sorted(blocks)]
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _blocked_flatnonzero(condition: Callable[[slice], NDArray[np.bool_]], rows: int, *,
                         elements: int) -> NDArray[np.intp]:
    """``np.flatnonzero`` of a per-row condition, evaluated in row blocks and joined in row order."""

    found: dict[int, NDArray[np.intp]] = {}

    def check(block: slice) -> None:
        found[block.start] = np.flatnonzero(condition(block)) + block.start

    run_blocks(check, rows, elements=elements, worker_limit=8)
    return _joined_rows(found)


def _blocked_allclose(first: NDArray[Any], second: NDArray[Any], *, rtol: float, atol: float) -> bool:
    """``np.allclose`` of two same-shape 1-D arrays, as the conjunction of its row blocks' ``np.allclose``.

    ``allclose`` is ``all(isclose(...))`` elementwise, so a block split decides the same boolean.
    """

    if first.ndim != 1 or first.shape != second.shape:
        return bool(np.allclose(first, second, rtol=rtol, atol=atol))
    failed: list[bool] = []

    def check(block: slice) -> None:
        if not np.allclose(first[block], second[block], rtol=rtol, atol=atol):
            failed.append(True)

    run_blocks(check, first.shape[0], elements=first.size)
    return not failed


def regional_boundary_mask_digest(masks: Mapping[str, Any]) -> str:
    """SHA-256 of the bdyMask triple a regional registry row pins.

    Fixed encoding -- each name in :data:`REGIONAL_BOUNDARY_MASK_NAMES`
    order, a NUL, then the little-endian int32 payload -- so the digest a
    ``mesh-check`` receipt prints is byte-for-byte the digest a registry row
    declares and the bind verifies.  Two definitions would eventually
    disagree and admit a mesh whose rings nobody checked.
    """

    digest = hashlib.sha256()
    for name in REGIONAL_BOUNDARY_MASK_NAMES:
        if name not in masks:
            raise MeshValidationError(
                f"regional mask digest requires {name}; a partial triple "
                "cannot identify the boundary rings"
            )
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            np.ascontiguousarray(np.asarray(masks[name], dtype="<i4")).tobytes(
                order="C"
            )
        )
    return digest.hexdigest()


def regional_ring_shell_errors(
    mask_cell: NDArray[np.int64],
    source_cells: NDArray[np.int64],
    neighbour_cells: NDArray[np.int64],
    zone_width: int,
) -> list[str]:
    """Refuse a boundary zone whose rings do not sit on the ring inside them.

    THE BREAKAGE THIS PREVENTS: a torn or renumbered relaxation zone.  Ring
    ``k`` is staged against ring ``k-1`` every step; a ring-``k`` cell with no
    ring-``k-1`` neighbour is forced from a ring that does not touch it, and
    the boundary blend has a hole nothing announces.

    THIS REPLACES A TEST THAT DID NOT IMPLY ITS OWN CLAIM, and the counter-
    example is measured (2026-08-27, the proving RTX 5090, evidence/nest-ratio-20260827/).
    The check here used to be that ring POPULATIONS grow outward, on the
    stated reasoning that "each ring wraps the previous in a growing shell".
    Populations are a proxy for wrapping on a UNIFORM mesh only, and the only
    meshes this program culls are variable resolution.  Three culls of one
    parent, one binary, one run -- one of them byte-identical to the
    registered ``r4.75.11020`` -- were handed to the old check and two were
    refused:

    ======  ==============================  ==============================
    cull    ring cell counts 1..7           ring mean cell width, km
    ======  ==============================  ==============================
    d045    207 209 212 215 218 217 225     5.03 -> 5.49
    d070    254 251 249 242 241 237 232     6.31 -> 7.71
    d100    240 245 251 257 263 269 275     8.99 -> 9.65
    ======  ==============================  ==============================

    ``d070``'s rings grow outward into the parent's coarsening ramp: each ring
    is 2.9 per cent wider than the one inside it, so the shell's PERIMETER
    grows 11.6 per cent over the seven rings while its POPULATION falls 8.7
    per cent.  Fewer, larger cells wrapping a larger circumference is not a
    tear.  The arithmetic is exact -- with a perimeter of ``P`` cells, one ring
    outward gives ``(P + 2*pi) / (1 + g)`` for a fractional width growth ``g``,
    which shrinks whenever ``g > 2*pi/P`` -- so on any cull big enough to
    matter, a few per cent of grading per ring inverts the old test.

    Every one of the three satisfies the property asserted here, and so does
    every native cull the conventions were measured on.  The claims the old
    test was reaching for are kept and are checked elsewhere: no empty ring,
    neighbouring masks never differing by more than one, and Euler
    characteristic 1.
    """

    errors: list[str] = []
    inner = mask_cell[neighbour_cells] == mask_cell[source_cells] - 1
    touches_inner = np.zeros(mask_cell.size, dtype=bool)
    touches_inner[source_cells[inner]] = True
    for ring in range(1, zone_width + 1):
        members = np.flatnonzero(mask_cell == ring)
        if members.size == 0:
            continue  # empty rings are reported by their own check
        orphans = members[~touches_inner[members]]
        if orphans.size:
            errors.append(
                f"bdyMaskCell ring {ring} has cells with no ring-{ring - 1} "
                "neighbour at "
                + _bad_examples(orphans)
                + f": ring {ring} is staged against ring {ring - 1} every "
                "step, so a cell that touches no cell of the ring inside it "
                "is blended from a ring that does not reach it -- a torn or "
                "renumbered zone"
            )
    return errors


#: Coordinate representations this build knows how to read, and whether each
#: stores its components as binary64.  A file declaring anything else is
#: refused rather than guessed at: see :func:`_coordinate_frame_errors`.
COORDINATE_REPRESENTATIONS: dict[str, bool] = {
    "binary32_earth_centred": False,
    "binary64_earth_centred": True,
}

#: Global attribute naming the representation.  Absent means
#: ``binary32_earth_centred`` -- what every published mesh carries.
COORDINATE_REPRESENTATION_ATTR = "rw_coordinate_representation"

#: Global attribute carrying the origin the stored components are measured
#: FROM, in metres.  Absent means the Earth's centre.  A reader reconstructs an
#: absolute position as ``stored + origin``.
COORDINATE_ORIGIN_ATTR = "rw_coordinate_origin_xyz"


def _coordinate_frame_errors(attrs: dict[str, Any], arrays: dict[str, Any]) -> list[str]:
    """Refuse a file whose declared coordinate frame is not the one it holds.

    WHAT THIS DOES NOT DO.  It does not decide the storage tolerance;
    :func:`spherical_arc_tolerance` reads that off the coordinate dtype and is
    already right with no attribute at all.  An UNDECLARED file is therefore
    read exactly as before, and it has to be: the published
    ``x1.40962.grid.nc`` and ``x4.163842.grid.nc`` store binary64 coordinates
    and carry no attribute, so reading silence as binary32 would refuse the two
    meshes the port is anchored to.

    WHAT IT DOES.  Every check downstream loosens with the dtype it finds, so a
    fine mesh rewritten from binary64 to binary32 -- a rescale, a conversion
    through a tool that only writes floats -- loses 5.4e8 of its coordinate
    precision and passes all of them, because they all move with it.  On a
    115 m dual edge that is 8.4e-3 of orthogonality defect in the point set the
    dycore's operators are built from where the generator was gated at 1.4e-11.
    A declaration of what the PRODUCER stored is the only thing that notices.

    Three refusals, each naming what it prevents:

    * an UNKNOWN representation -- it carries no statement this build can
      check, so a rewritten file would pass unremarked;
    * a declaration that DISAGREES with the stored dtype -- the file was
      rewritten after its producer wrote it, in the direction the message
      names;
    * an ORIGIN that is not the Earth's centre -- this build reads the stored
      components as absolute positions, so it would place the whole mesh
      somewhere else on the Earth, and terrain, lateral boundaries and every
      rendered product would land at the wrong point with no arithmetic
      anywhere disagreeing.
    """

    errors: list[str] = []
    names = [axis + family for family in ("Cell", "Edge", "Vertex") for axis in "xyz"]
    present = [arrays[n] for n in names if n in arrays]
    if not present:
        return errors
    stored_is_binary64 = all(np.asarray(a).dtype.itemsize >= 8 for a in present)
    mixed = stored_is_binary64 != any(np.asarray(a).dtype.itemsize >= 8 for a in present)
    if mixed:
        errors.append(
            "Cartesian coordinate arrays are stored at MIXED widths; the storage "
            "tolerance is derived from one quantum and cannot serve two"
        )
        return errors

    declared = attrs.get(COORDINATE_REPRESENTATION_ATTR)
    if declared is None:
        # Read by dtype, as every consumer already reads it.  Absence is not a
        # claim, and must never be treated as one.
        return errors
    representation = str(declared).strip()
    if representation not in COORDINATE_REPRESENTATIONS:
        errors.append(
            f"{COORDINATE_REPRESENTATION_ATTR}={representation!r} is not a "
            f"representation this build knows "
            f"({sorted(COORDINATE_REPRESENTATIONS)}). The attribute records what "
            "the file's producer stored, so that a file rewritten at a different "
            "width is caught rather than read at the width it now happens to "
            "have; a tag this build cannot interpret carries no such statement "
            "and would be ignored in silence"
        )
        return errors
    if COORDINATE_REPRESENTATIONS[representation] != stored_is_binary64:
        says = "binary64" if COORDINATE_REPRESENTATIONS[representation] else "binary32"
        holds = "binary64" if stored_is_binary64 else "binary32"
        consequence = (
            "A demotion loses 5.37e+08 of the coordinate precision the mesh was "
            "generated and gated at, and the point set the dycore builds its edge "
            "normals, tangent planes and reconstruction coefficients from stops "
            "being orthogonal to that accuracy."
            if says == "binary64"
            else "A promotion changes no value but proves the arrays are not the "
            "bytes the producer wrote, so nothing else in the file is the bytes "
            "it wrote either."
        )
        errors.append(
            f"{COORDINATE_REPRESENTATION_ATTR}={representation!r} but "
            f"xCell/yCell/zCell are stored as {holds}, so this file was rewritten "
            f"after its producer wrote it. {consequence} Every storage check here "
            "derives its tolerance from the dtype it FINDS -- 2*sqrt(3) coordinate "
            "ULP, 1.73 m at binary32 and 3.2e-9 m at binary64 -- so all of them "
            "move with the rewrite and none of them would say a word. Rebuild the "
            "static from its grid rather than editing either the attribute or the "
            "arrays"
        )
        return errors

    origin = attrs.get(COORDINATE_ORIGIN_ATTR)
    if origin is not None:
        vector = np.asarray(origin, dtype=np.float64).ravel()
        if vector.size != 3 or not np.all(np.isfinite(vector)):
            errors.append(
                f"{COORDINATE_ORIGIN_ATTR} must be three finite metres; a frame "
                "that cannot be read is a frame that would be guessed"
            )
        elif np.any(vector != 0.0):
            offset = float(np.linalg.norm(vector)) / 1000.0
            errors.append(
                f"{COORDINATE_ORIGIN_ATTR}={vector.tolist()} m is not the Earth's "
                f"centre. This build reads the stored components as absolute "
                f"positions, so it would place the whole mesh {offset:.1f} km from "
                "where the file puts it, and terrain lookup, lateral-boundary "
                "interpolation and every rendered product would land at the wrong "
                "point with no arithmetic anywhere disagreeing"
            )
    return errors


def spherical_arc_tolerance(
    radius: float, coordinate_dtype: Any, metric_rtol: float = _SPHERICAL_ARC_MAX_RTOL
) -> tuple[float, float]:
    """Tolerance for comparing a stored edge length against a recomputed arc.

    The comparison has two independent error sources and a purely relative
    bound can only express one of them.

    * The stored length rounds to its own storage precision.  That is a
      RELATIVE error and ``metric_rtol`` carries it.
    * The Cartesian coordinates the arc is recomputed FROM are quantized at
      the sphere radius, and that error does not shrink with the edge.  At
      binary32 and Earth radius the coordinate spacing is
      ``np.spacing(float32(6371220.0)) == 0.5`` metres whether the edge is
      8 km or 800 km long, so it is an ABSOLUTE floor.

    Deriving the floor: each of an endpoint's three components rounds by at
    most half a coordinate spacing, bounding the endpoint's displacement by
    ``sqrt(3)/2`` spacings; the two endpoints of an edge can move in opposite
    directions along it, so the arc carries at most ``sqrt(3)`` spacings.  The
    factor of two on top carries a mesh whose coordinates were rounded a
    second time -- :func:`load_precision_preserving_mesh_pair` rescales them
    between two sphere radii, which is exactly that second rounding.

    Without the floor a defect-free mesh is refused for its own storage
    precision: at 8.7 km vertex spacing the quantization alone is ~1e-5 of the
    edge, which sits on top of a 2.0e-5 relative bound and makes admission a
    coin flip.  With the floor named, the relative half no longer has to
    stand in for it, so it is capped at what binary32 storage actually needs
    (``8 * eps`` ~ 9.5e-7) rather than the 2.0e-5 that let metre-scale
    corruption hide at coarse spacing.
    """

    spacing = float(np.spacing(np.asarray(abs(radius), dtype=coordinate_dtype)))
    return min(float(metric_rtol), _SPHERICAL_ARC_MAX_RTOL), 2.0 * math.sqrt(3.0) * spacing


#: The two orderings of ``verticesOnCell`` against ``edgesOnCell`` that
#: published MPAS meshes carry.  ``edge-start``: vertex ``j`` opens edge
#: ``j``, so edge ``j`` joins vertices ``j`` and ``j+1`` (MPAS-Tools, NCAR's
#: x1 meshes, RRFS v2's conus12km, and the rw-mpas generator).
#: ``edge-end``: vertex ``j`` closes edge ``j``, so edge ``j`` joins vertices
#: ``j-1`` and ``j`` (RRFS v2's public conus3km, every one of its 2,341,818
#: cells, measured 2026-10-05).  Both are the same counter-clockwise ring;
#: they differ only in which slot the ring starts at.
VERTEX_SLOT_CONVENTIONS = ("edge-start", "edge-end")

#: Closure identities of a binary64 producer cannot be more precise than the
#: arcs it builds its areas from.  An arc taken from a dot product (acos)
#: carries a relative error of about ``u / theta**2`` for an arc of
#: ``theta`` radians, ``u`` the binary64 unit roundoff.  A kite is built from
#: half arcs (cell centre to edge midpoint, edge midpoint to vertex), which
#: quarters ``theta**2``, its area takes two of them, and a closure compares
#: two such areas: 4 * 2 * 2 = 16.
_CLOSURE_CONDITIONING_FACTOR = 16.0

#: The binary32 closure bound the validator has always held for metrics
#: stored at binary32 precision.  Unchanged here: the measured binary32
#: statics (conus12km, conus3km) close at 1.4e-7, inside it.
_BINARY32_METRIC_RTOL = 2.0e-5


def vertex_slot_convention(
    counts: Any,
    edges_on_cell: Any,
    vertices_on_cell: Any,
    vertices_on_edge: Any,
) -> dict[str, Any]:
    """Detect which slot ordering a mesh stores ``verticesOnCell`` in.

    MPAS consumes ``verticesOnCell`` in its stored order and pairs vertex
    ``j`` with edge ``j`` nowhere in the time integration: every sum over a
    cell's vertices (``pv_cell``, the kinetic-energy vertex correction) goes
    through ``kiteForCell``, which finds the kite by searching
    ``cellsOnVertex`` and so holds under either ordering.  The convention is
    therefore a property to DETECT and check reciprocity under, not one to
    rewrite: nothing here reorders the stored array.

    Returns the per-mesh verdict, the per-cell counts, the first cells that
    satisfy neither ordering, and ``view``: the stored rows re-indexed so the
    edge-start reciprocity check (vertices ``j`` and ``j+1`` bound edge
    ``j``) holds exactly when the mesh is consistent under its own detected
    convention.  ``view`` is a validation index map only; it never reaches
    the dycore.
    """

    counts = np.asarray(counts, dtype=np.int64)
    edges = np.asarray(edges_on_cell, dtype=np.int64)
    vertices = np.asarray(vertices_on_cell, dtype=np.int64)
    endpoints = np.asarray(vertices_on_edge, dtype=np.int64)
    n_cells, max_edges = edges.shape
    slots = np.arange(max_edges)[None, :]
    start_cells = np.empty(n_cells, dtype=bool)
    end_cells = np.empty(n_cells, dtype=bool)

    def check(block: slice) -> None:
        local_counts = counts[block, None]
        local_vertices = vertices[block]
        used = slots < local_counts
        rows = np.arange(len(local_vertices))[:, None]
        following = np.where(slots + 1 < local_counts, slots + 1, 0)
        preceding = np.where(slots >= 1, slots - 1, local_counts - 1)
        safe_edges = np.where(used, edges[block], 0)
        first = endpoints[safe_edges, 0]
        second = endpoints[safe_edges, 1]

        def bounds(candidate: NDArray[np.int64]) -> NDArray[np.bool_]:
            return (candidate == first) | (candidate == second)

        own = bounds(local_vertices)
        start_cells[block] = (
            (own & bounds(local_vertices[rows, following])) | ~used
        ).all(axis=1)
        end_cells[block] = (
            (own & bounds(local_vertices[rows, preceding])) | ~used
        ).all(axis=1)

    # Complete rows are independent. Global counts and first bad-cell indices
    # are still derived in stored order after every worker has finished.
    run_blocks(check, n_cells, elements=edges.size)
    n_start = int(np.count_nonzero(start_cells))
    n_end = int(np.count_nonzero(end_cells))
    neither = np.flatnonzero(~start_cells & ~end_cells)
    if n_start == n_cells:
        convention = "edge-start"
    elif n_end == n_cells:
        convention = "edge-end"
    elif n_start and n_end:
        convention = "mixed"
    else:
        convention = "none"
    # The view follows the ordering most cells hold, so a reciprocity check
    # run on it names only the cells that break that ordering instead of
    # every cell of a consistently edge-end mesh.
    if n_end > n_start:
        view = np.empty(vertices.shape, dtype=vertices.dtype)

        def rotate(block: slice) -> None:
            local_counts = counts[block, None]
            local_vertices = vertices[block]
            rows = np.arange(len(local_vertices))[:, None]
            preceding = np.where(slots >= 1, slots - 1, local_counts - 1)
            view[block] = np.where(
                slots < local_counts, local_vertices[rows, preceding], -1,
            )

        run_blocks(rotate, n_cells, elements=vertices.size)
    else:
        view = vertices
    return {
        "convention": convention,
        "edge_start_cells": n_start,
        "edge_end_cells": n_end,
        "neither_cells": int(neither.size),
        "first_neither_cells": [int(cell) for cell in neither[:5]],
        "view": view,
    }


def _edge_weight_errors(
    weights: NDArray[Any], counts: NDArray[np.int64], *, check_padding: bool,
) -> list[str]:
    """Check finite weights and zero padding without a whole-mesh mask.

    Each complete edge row retains the original NumPy predicates. Only
    boolean failure flags cross blocks, so neither weight words nor the
    finite-before-padding refusal order depends on the worker schedule.
    Invalid stencil counts still suppress the padding check at the caller.
    """
    slots = np.arange(weights.shape[1])[None, :]
    nonfinite: list[bool] = []
    nonzero_padding: list[bool] = []

    def check(block: slice) -> None:
        values = weights[block]
        if not np.all(np.isfinite(values)):
            nonfinite.append(True)
        if check_padding:
            padding = slots >= counts[block, None]
            if np.any(values[padding] != 0):
                nonzero_padding.append(True)

    run_blocks(check, weights.shape[0], elements=weights.size)
    errors = []
    if nonfinite:
        errors.append("weightsOnEdge contains non-finite values")
    if nonzero_padding:
        errors.append("weightsOnEdge padding is not zero")
    return errors


def _stored_significant_bits(array: Any) -> int:
    """Significand bits the stored VALUES carry, measured, not read off dtype.

    A binary64 array whose every value is exactly a binary32 number stores
    binary32 precision whatever its dtype says; a writer that widened
    binary32 metrics on output leaves exactly that.  Returns 24 for such an
    array (and for a binary32 array), 53 otherwise.
    """

    values = np.asarray(array)
    if values.dtype.itemsize <= 4:
        return 24
    widened = values.astype(np.float64, copy=False)
    finite = widened[np.isfinite(widened)]
    if finite.size and np.array_equal(
        finite.astype(np.float32).astype(np.float64), finite
    ):
        return 24
    return 53


def closure_tolerance(
    metric_arrays: Mapping[str, Any],
    *,
    radius: float | None,
    edge_lengths: Any | None,
) -> dict[str, Any]:
    """The area-closure tolerance this file's stored precision admits.

    Two measured quantities set it.  The STORAGE precision is the fewest
    significand bits any metric array actually carries
    (:func:`_stored_significant_bits`): binary32-precision metrics keep the
    long-standing binary32 bound.  For binary64 storage the bound is set by
    how precisely a binary64 producer CAN compute a small element's area:
    ``16 * u / theta_min**2`` with ``theta_min`` the mesh's shortest
    ``dcEdge``/``dvEdge`` in radians (see
    :data:`_CLOSURE_CONDITIONING_FACTOR`).  On RRFS v2's conus3km grid
    (``theta_min`` 1.97e-4, its shortest edge 1,254 m) that is 4.6e-8,
    and the file measures 8.1e-10 (triangles) and 5.9e-10 (cells); its
    stored ``areaTriangle`` disagrees
    with the binary64 spherical area of its own cell-centre triangle by up to
    4.2e-10, the same ``u / theta**2`` scale, while every arc length agrees
    to 8e-13.  The previous fixed 5e-10 remains the binary64 bound wherever
    this term is smaller -- coarse meshes are not loosened, because nothing
    measured says they need it.

    This returns the MESH-WIDE summary (the loosest element, set by the
    shortest edge) for receipts and refusal text; :meth:`Mesh.validate`
    holds each triangle and cell to the same formula with its OWN shortest
    edge, so a graded mesh's coarse elements keep their tight bound.
    """

    bits = min(
        (_stored_significant_bits(array) for array in metric_arrays.values()),
        default=53,
    )
    unit_roundoff = float(np.ldexp(1.0, -bits))
    theta_min = None
    conditioning = 0.0
    if bits <= 24:
        rtol = _BINARY32_METRIC_RTOL
        basis = "binary32 storage bound"
    else:
        if radius is not None and edge_lengths is not None and radius > 0:
            lengths = np.asarray(edge_lengths, dtype=np.float64)
            lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
            if lengths.size:
                theta_min = float(lengths.min() / radius)
                conditioning = (
                    _CLOSURE_CONDITIONING_FACTOR
                    * float(np.ldexp(1.0, -53))
                    / (theta_min * theta_min)
                )
        rtol = max(_BINARY64_MESH_RTOL, conditioning)
        basis = (
            "binary64 producer conditioning 16*u/theta_min**2"
            if conditioning > _BINARY64_MESH_RTOL
            else "binary64 bound (conditioning term below it)"
        )
    return {
        "stored_significant_bits": bits,
        "unit_roundoff": unit_roundoff,
        "theta_min_radians": theta_min,
        "conditioning_rtol": conditioning,
        "rtol": float(rtol),
        "basis": basis,
    }


def _closure_statement(tolerance: Mapping[str, Any]) -> str:
    theta = tolerance["theta_min_radians"]
    theta_text = "n/a" if theta is None else f"{theta:.3e} rad"
    return (
        f"bound {tolerance['rtol']:.3e} ({tolerance['basis']}; stored metrics "
        f"carry {tolerance['stored_significant_bits']} significand bits, "
        f"shortest edge {theta_text})"
    )


def _wrapped_angle_delta(left: Any, right: Any) -> NDArray[np.float64]:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    return np.asarray((first - second + np.pi) % (2.0 * np.pi) - np.pi)


def _maximum_ulp_error(actual: Any, expected: Any) -> float:
    candidate = np.asarray(actual, dtype=np.float64)
    authority = np.asarray(expected, dtype=np.float64)
    if candidate.shape != authority.shape:
        return float("inf")
    axis = int(np.argmax(candidate.shape)) if candidate.ndim else 0
    maxima: list[float] = []
    policy = np.geterr()

    def check(block: slice) -> None:
        index = [slice(None)] * candidate.ndim
        if index:
            index[axis] = block
        with np.errstate(**policy):
            gaps = np.abs(candidate[tuple(index)] - authority[tuple(index)])
            spacing = np.abs(np.spacing(authority[tuple(index)]))
            with np.errstate(over="ignore", invalid="ignore"):
                ratios = np.divide(
                    gaps,
                    spacing,
                    out=np.full(gaps.shape, np.inf, dtype=np.float64),
                    where=spacing > 0.0,
                )
            ratios[gaps == 0.0] = 0.0
            maxima.append(float(np.max(ratios, initial=0.0)))

    # Maxima preserve NaN propagation and do not reassociate floating sums.
    # Callback/log policies belong to the caller's thread-local error handler.
    if any(value in ("call", "log") for value in policy.values()):
        check(slice(0, candidate.shape[axis] if candidate.ndim else 1))
    else:
        run_blocks(check, candidate.shape[axis] if candidate.ndim else 1,
                   elements=candidate.size * 8, worker_limit=4)
    return float(np.max(maxima, initial=0.0))


def _latitude_longitude_xyz(
    latitude: NDArray[np.float64], longitude: NDArray[np.float64], radius: float,
) -> NDArray[np.float64]:
    """Build independent Cartesian rows with the original ufunc order."""
    xyz = np.empty((latitude.size, 3), dtype=np.float64)
    policy = np.geterr()

    def construct(block: slice) -> None:
        lat, lon = latitude[block], longitude[block]
        with np.errstate(**policy):
            xyz[block] = radius * np.stack(
                (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)),
                axis=1,
            )

    if any(value in ("call", "log") for value in policy.values()):
        construct(slice(0, latitude.size))
    else:
        run_blocks(construct, latitude.size, elements=latitude.size * 12, worker_limit=4)
    return xyz


def _grid_rotate_convert_xl_longitude(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reproduce the pinned Fortran ``convert_xl`` longitude arithmetic.

    The producer stores its result in binary64, but ``pii = 2.*asin(1.0)`` is
    evaluated in default-real binary32.  ``multiplicity`` records how many
    copies of the resulting pi error each literal quadrant branch introduces.
    """

    policy = np.geterr()
    if x.ndim != 1 or any(value in ("call", "log") for value in policy.values()):
        return _grid_rotate_convert_xl_longitude_block(x, y)
    longitude = np.empty(x.shape, dtype=np.float64)
    multiplicity = np.empty(x.shape, dtype=np.float64)

    def construct(block: slice) -> None:
        with np.errstate(**policy):
            longitude[block], multiplicity[block] = _grid_rotate_convert_xl_longitude_block(
                x[block], y[block]
            )

    run_blocks(construct, x.size, elements=x.size * 16, worker_limit=4)
    return longitude, multiplicity


def _grid_rotate_convert_xl_longitude_block(
    x: NDArray[np.float64], y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    longitude = np.zeros(x.shape, dtype=np.float64)
    multiplicity = np.zeros(x.shape, dtype=np.float64)
    x_active = np.abs(x) > GRID_ROTATE_CONVERT_XL_EPS
    y_active = np.abs(y) > GRID_ROTATE_CONVERT_XL_EPS
    both_active = x_active & y_active
    both_indices = np.flatnonzero(both_active)
    base = np.arctan(np.abs(y[both_active] / x[both_active]))
    longitude[both_active] = base

    second_quadrant = (x[both_indices] <= 0.0) & (y[both_indices] >= 0.0)
    indices = both_indices[second_quadrant]
    longitude[indices] = GRID_ROTATE_DEFAULT_REAL_PI - base[second_quadrant]
    multiplicity[indices] = 1.0

    third_quadrant = (x[both_indices] <= 0.0) & (y[both_indices] < 0.0)
    indices = both_indices[third_quadrant]
    longitude[indices] = base[third_quadrant] + GRID_ROTATE_DEFAULT_REAL_PI
    multiplicity[indices] = 1.0

    fourth_quadrant = (x[both_indices] >= 0.0) & (y[both_indices] <= 0.0)
    indices = both_indices[fourth_quadrant]
    longitude[indices] = 2.0 * GRID_ROTATE_DEFAULT_REAL_PI - base[fourth_quadrant]
    multiplicity[indices] = 2.0

    x_axis = x_active & ~y_active
    negative_x_axis = x_axis & (x <= 0.0)
    longitude[negative_x_axis] = GRID_ROTATE_DEFAULT_REAL_PI
    multiplicity[negative_x_axis] = 1.0

    y_axis = ~x_active & y_active
    positive_y_axis = y_axis & (y > 0.0)
    longitude[positive_y_axis] = GRID_ROTATE_DEFAULT_REAL_PI / 2.0
    multiplicity[positive_y_axis] = 0.5
    negative_y_axis = y_axis & (y <= 0.0)
    longitude[negative_y_axis] = 3.0 * GRID_ROTATE_DEFAULT_REAL_PI / 2.0
    multiplicity[negative_y_axis] = 1.5
    return longitude, multiplicity


def validate_longitude_normalization(
    source: Any, normalized: Any
) -> dict[str, float | str]:
    """Prove a ``[-pi, pi)`` longitude normalization preserves sin/cos."""

    original = np.asarray(source, dtype=np.float64)
    candidate = np.asarray(normalized, dtype=np.float64)
    if original.shape != candidate.shape:
        raise ValueError("longitude normalization requires same-shaped finite arrays")

    # Each element uses the original NumPy expression. Only boolean AND and
    # finite nonnegative maxima cross block boundaries, so the exact receipt
    # scalars and error precedence are independent of the worker schedule.
    outcomes: list[tuple[bool, bool, bool, float, float]] = []
    axis = int(np.argmax(original.shape)) if original.ndim else 0

    def check(block: slice) -> None:
        index = [slice(None)] * original.ndim
        if index:
            index[axis] = block
        before, after = original[tuple(index)], candidate[tuple(index)]
        if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
            outcomes.append((False, False, False, 0.0, 0.0))
            return
        expected = (before + np.pi) % (2.0 * np.pi) - np.pi
        mapped = bool(np.array_equal(after, expected))
        in_range = not bool(np.any(after < -np.pi) or np.any(after >= np.pi))
        sin_error = float(np.max(np.abs(np.sin(after) - np.sin(before)), initial=0.0))
        cos_error = float(np.max(np.abs(np.cos(after) - np.cos(before)), initial=0.0))
        outcomes.append((True, mapped, in_range, sin_error, cos_error))

    run_blocks(check, original.shape[axis] if original.ndim else 1,
               elements=original.size * 12)
    if not all(row[0] for row in outcomes):
        raise ValueError("longitude normalization requires same-shaped finite arrays")
    if not all(row[1] for row in outcomes):
        raise ValueError("longitude normalization is not the exact modulo mapping")
    if not all(row[2] for row in outcomes):
        raise ValueError("normalized longitude lies outside [-pi, pi)")
    sin_gap = max(row[3] for row in outcomes)
    cos_gap = max(row[4] for row in outcomes)
    if (
        sin_gap > LONGITUDE_TRIG_EQUIVALENCE_ATOL
        or cos_gap > LONGITUDE_TRIG_EQUIVALENCE_ATOL
    ):
        raise ValueError("longitude modulo mapping is not trig-equivalent")
    return {
        "method": "(longitude + pi) modulo 2pi minus pi",
        "sin_max_abs_gap": sin_gap,
        "cos_max_abs_gap": cos_gap,
        "trig_equivalence_atol": LONGITUDE_TRIG_EQUIVALENCE_ATOL,
    }


def normalize_longitudes(source: Any) -> NDArray[np.float64]:
    """Return the exact evidenced longitude normalization in binary64."""

    original = np.asarray(source, dtype=np.float64)
    normalized = np.ascontiguousarray((original + np.pi) % (2.0 * np.pi) - np.pi)
    validate_longitude_normalization(original, normalized)
    return normalized


class Mesh:
    """In-memory MPAS mesh with canonical Python connectivity.

    Arrays retain their official MPAS camel-case names in :attr:`arrays` and
    are also accessible as attributes in either camel case (``cellsOnEdge``)
    or snake case (``cells_on_edge``).  The same applies to dimensions and
    global attributes.
    """

    #: Fields that record only where the mesh bytes were read from.  The
    #: host-preparation cache (woof.hex.prep_cache) leaves them out of its key,
    #: because every array read from those files is hashed already, so the
    #: same mesh at another path (a moved case, a cache baked into an image)
    #: is served.  A cached function whose reachable code names one of them
    #: is keyed on it after all.
    #:
    #: ``partition_source_mesh`` (a same-host rank's local mesh names the whole mesh it was cut from,
    #: partition_local_mesh_v841) is left out on the same rule: every local array a cached function reads is
    #: hashed already, and one whose reachable code names the source mesh is keyed on it after all, unless it
    #: declares its result independent of it (the deformation weights, mixing_v841).  No Mesh method reads it.  THE COST THIS REMOVES (RRFS hrrrv5 limited area, same-host rank processes, mv3km 4
    #: ranks on V8X6000, 2026-10-29): every cached call on a rank's local mesh (advection coefficients, ...)
    #: hashed the whole source mesh, so every rank process read all 2.0 GB of the shared mesh copy into its
    #: resident memory (23.9 GB per rank process on NOAA's conus3km) and spent the SHA-256 work on it.
    prep_cache_provenance = ("grid_path", "static_path", "provenance", "partition_source_mesh")

    #: Process-local device caches some consumers attach to a mesh (the stochastic spherical-harmonic basis,
    #: woof.hex.stochastic.MeshHarmonics.cached).  They hold CUDA arrays and events of this process.
    _DEVICE_CACHES = ("_stochastic_harmonics",)

    def __getstate__(self):
        """A pickled mesh carries its host arrays, never a device cache of the process that pickled it.

        THE BREAKAGE THIS PREVENTS: the MPAS-suite checkpoint pickles the mesh; with --stochastic the 1.18 GB
        GPU harmonic basis rode inside it (a 2.45 GB checkpoint instead of 1.27 GB), and the restart unpickled
        those CUDA arrays and events before its own CUDA setup and crashed at exit (rc 139, HEX-PHYSICS-PARITY
        run group 6 #22).  The consumer rebuilds its cache on first use.
        """
        state = dict(self.__dict__)
        for name in self._DEVICE_CACHES:
            state.pop(name, None)
        return state

    def __init__(
        self,
        *,
        arrays: Mapping[str, NDArray[Any]],
        dimensions: Mapping[str, int],
        attrs: Mapping[str, Any],
        grid_attrs: Mapping[str, Any] | None = None,
        static_attrs: Mapping[str, Any] | None = None,
        variable_dimensions: Mapping[str, tuple[str, ...]] | None = None,
        variable_attrs: Mapping[str, Mapping[str, Any]] | None = None,
        variable_sources: Mapping[str, str] | None = None,
        grid_path: str | Path | None = None,
        static_path: str | Path | None = None,
        converted_connectivity: tuple[str, ...] = (),
    ) -> None:
        self.arrays = {name: np.asarray(value) for name, value in arrays.items()}
        self.variables = self.arrays
        self.dimensions = dict(dimensions)
        self.attrs = dict(attrs)
        self.grid_attrs = dict(grid_attrs or attrs)
        self.static_attrs = dict(static_attrs or {})
        self.variable_dimensions = dict(variable_dimensions or {})
        self.variable_attrs = {
            name: dict(value) for name, value in (variable_attrs or {}).items()
        }
        self.variable_sources = dict(variable_sources or {})
        self.grid_path = None if grid_path is None else str(Path(grid_path).resolve())
        self.static_path = None if static_path is None else str(Path(static_path).resolve())
        self.converted_connectivity = tuple(converted_connectivity)
        self.provenance = {
            "grid_path": self.grid_path,
            "static_path": self.static_path,
            "grid_attrs": self.grid_attrs,
            "static_attrs": self.static_attrs,
            "variable_sources": self.variable_sources,
            "connectivity_indexing": "zero-based with -1 padding",
        }
        self._aliases: dict[str, str] = {}
        for name in (*self.arrays, *self.dimensions, *self.attrs):
            self._aliases.setdefault(_camel_to_snake(name), name)

    @classmethod
    def from_netcdf(
        cls,
        grid_path: str | Path,
        static_path: str | Path | None = None,
        *,
        validate: bool = True,
        native_parent_chain: bool = False,
        exclude_variables: frozenset[str] = frozenset(),
        _grid_reader: Callable[[Path], _NetCDFData] | None = None,
    ) -> "Mesh":
        """Load an official MPAS grid and optionally overlay a static file.

        All variables from the grid are retained.  A static file contributes
        all of its variables and replaces same-named grid variables, which is
        important because published grid files commonly use a unit sphere
        while initialized static files contain Earth-scaled metrics and useful
        fields such as ``ter``, ``landmask``, and ``coeffs_reconstruct``.
        Per-file attributes and per-variable source labels remain available in
        :attr:`provenance`.

        ``exclude_variables`` skips named payloads in both files before reading
        them. The native physics route uses this for separately loaded state
        fields. Dimensions, global attributes and retained variable metadata
        still follow the same overlay and validation rules.
        """

        grid = Path(grid_path).expanduser().resolve(strict=True)
        (
            dimensions,
            grid_attrs,
            arrays,
            variable_dimensions,
            variable_attrs,
        ) = ((_read_netcdf(grid, exclude_variables=exclude_variables)
              if exclude_variables else _read_netcdf(grid))
             if _grid_reader is None else _grid_reader(grid))
        if _grid_reader is not None and exclude_variables:
            # A shared reader result may serve several loads; filter copies of
            # its tables instead of mutating them.
            arrays, variable_dimensions, variable_attrs = (
                {name: value for name, value in table.items()
                 if name not in exclude_variables}
                for table in (arrays, variable_dimensions, variable_attrs))
        sources = {name: "grid" for name in arrays}
        effective_attrs = dict(grid_attrs)
        static_attrs: dict[str, Any] = {}
        static: Path | None = None

        if static_path is not None:
            static = Path(static_path).expanduser().resolve(strict=True)
            (
                static_dimensions,
                static_attrs,
                static_arrays,
                static_variable_dimensions,
                static_variable_attrs,
            ) = (_read_netcdf(static, exclude_variables=exclude_variables)
                 if exclude_variables else _read_netcdf(static))

            incompatible = {
                name: (dimensions[name], size)
                for name, size in static_dimensions.items()
                if name in dimensions and dimensions[name] != size
            }
            if incompatible:
                raise MeshValidationError(
                    f"static/grid dimensions disagree: {incompatible}"
                )

            grid_id = str(grid_attrs.get("file_id", "")).strip()
            parent_id = str(static_attrs.get("parent_id", "")).strip()
            # MPAS-Model-rrfs/src/framework/mpas_bootstrapping.F:320-330
            # at 1d6e370e appends inherited IDs with NEW_LINE('A').
            # Only the explicitly selected fork accepts this native lineage.
            parents = parent_id.splitlines() if native_parent_chain else [parent_id]
            if grid_id and parent_id and grid_id not in parents:
                raise MeshValidationError(
                    "static parent_id does not match grid file_id: "
                    f"{parent_id!r} != {grid_id!r}"
                )

            dimensions.update(static_dimensions)
            effective_attrs.update(static_attrs)
            arrays.update(static_arrays)
            variable_dimensions.update(static_variable_dimensions)
            variable_attrs.update(static_variable_attrs)
            sources.update({name: "static" for name in static_arrays})

        converted = _canonicalize_connectivity(arrays)
        mesh = cls(
            arrays=arrays,
            dimensions=dimensions,
            attrs=effective_attrs,
            grid_attrs=grid_attrs,
            static_attrs=static_attrs,
            variable_dimensions=variable_dimensions,
            variable_attrs=variable_attrs,
            variable_sources=sources,
            grid_path=grid,
            static_path=static,
            converted_connectivity=converted,
        )
        if validate:
            mesh.validate()
        return mesh

    def __getattr__(self, name: str) -> Any:
        arrays = self.__dict__.get("arrays", {})
        if name in arrays:
            return arrays[name]
        dimensions = self.__dict__.get("dimensions", {})
        if name in dimensions:
            return dimensions[name]
        attrs = self.__dict__.get("attrs", {})
        if name in attrs:
            return attrs[name]
        official = self.__dict__.get("_aliases", {}).get(name)
        if official in arrays:
            return arrays[official]
        if official in dimensions:
            return dimensions[official]
        if official in attrs:
            return attrs[official]
        raise AttributeError(f"Mesh has no MPAS field, dimension, or attribute {name!r}")

    def __getitem__(self, name: str) -> Any:
        try:
            return getattr(self, name)
        except AttributeError as error:
            raise KeyError(name) from error

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            getattr(self, name)
        except AttributeError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(self.arrays)

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return getattr(self, name)
        except AttributeError:
            return default

    def copy(self, *, deep: bool = True) -> "Mesh":
        """Copy the mesh; deep copies are convenient for mutation controls."""

        arrays = {
            name: value.copy() if deep else value for name, value in self.arrays.items()
        }
        return type(self)(
            arrays=arrays,
            dimensions=self.dimensions,
            attrs=self.attrs,
            grid_attrs=self.grid_attrs,
            static_attrs=self.static_attrs,
            variable_dimensions=self.variable_dimensions,
            variable_attrs=self.variable_attrs,
            variable_sources=self.variable_sources,
            grid_path=self.grid_path,
            static_path=self.static_path,
            converted_connectivity=self.converted_connectivity,
        )

    @property
    def is_regional(self) -> bool:
        """True when the complete ``bdyMask`` triple carries a boundary ZONE.

        A global mesh may carry the triple all-zero -- that is MPAS's own
        convention for "every element interior", and the unified
        ``rw_mpas_static`` writer emits it on every static this project
        generates.  Classifying on presence alone made a closed sphere a
        disk; see the module docstring for the measured refusal.
        """

        return _regional_boundary_zone_present(self.arrays)

    def validate(self) -> "Mesh":
        """Validate the complete MPAS C-grid contract, global or regional.

        Validation deliberately includes both local reciprocity and global
        metric identities.  A corrupt mesh therefore cannot pass merely
        because every integer happens to be in range.

        A mesh carrying the full ``bdyMaskCell/Edge/Vertex`` triple is a
        regional cull and is validated against the measured regional
        conventions (module docstring): sentinel-tolerant ranges keyed to
        :data:`REGIONAL_SENTINEL_ARRAYS` with a ring-7 placement rule,
        reciprocity exempted only where an absent neighbour makes it
        undefined, Euler characteristic 1 instead of 2, the mask-derivation
        rules, and rings that each sit on the ring inside them.  Every refusal names
        the concrete breakage instead of reporting the disk as a corrupt
        sphere.
        """

        errors: list[str] = []

        required_dimensions = (
            "nCells",
            "nEdges",
            "nVertices",
            "maxEdges",
            "maxEdges2",
            "vertexDegree",
        )
        missing_dimensions = [
            name for name in required_dimensions if name not in self.dimensions
        ]
        if missing_dimensions:
            raise MeshValidationError(
                f"missing required dimensions: {', '.join(missing_dimensions)}"
            )

        n_cells = int(self.dimensions["nCells"])
        n_edges = int(self.dimensions["nEdges"])
        n_vertices = int(self.dimensions["nVertices"])
        max_edges = int(self.dimensions["maxEdges"])
        max_edges2 = int(self.dimensions["maxEdges2"])
        vertex_degree = int(self.dimensions["vertexDegree"])
        if min(n_cells, n_edges, n_vertices, max_edges, max_edges2, vertex_degree) <= 0:
            errors.append("all core mesh dimensions must be positive")

        bdy_present = [
            name for name in REGIONAL_BOUNDARY_MASK_NAMES if name in self.arrays
        ]
        triple_complete = len(bdy_present) == len(REGIONAL_BOUNDARY_MASK_NAMES)
        # THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26, all three fleet
        # cards on the published x1.40962 at the anchored 120 s timestep, and
        # v20.80.151649 on the proving RTX 5090): carrying the triple is NECESSARY
        # for a mesh to be regional and it is not SUFFICIENT.  Native MPAS-A
        # writes bdyMaskCell/Edge/Vertex into the static file of a GLOBAL mesh
        # too, all zero -- the published x1.40962 static (NCAR,
        # init_atmosphere v8.2.0) carries all three, 0/40962, 0/122880,
        # 0/81920 nonzero -- and the unified rw_mpas_static follows that
        # convention on every static this project generates.  So presence
        # alone classified the published global mesh AND every generated one
        # as a regional cull, and refused them with four checks a sphere
        # cannot pass.  Both of those refusals were themselves the proof it is
        # global: "nCells-nEdges+nVertices = 2, not 1" is the Euler
        # characteristic of a closed sphere, and "bdyMask rings [1..7] are
        # empty" is what an all-zero mask means -- every element interior, no
        # boundary zone at all.  The registry agreed the row was global
        # (boundary_zone_width and bdy_mask_sha256 both unset); this validator
        # was the second, disagreeing classifier.
        #
        # A cull HAS a boundary zone: that is what culling produces.  So the
        # rule is the triple plus a nonempty zone, which is content, not
        # schema, and which cannot be satisfied by a global mesh.  The
        # incompleteness refusal below stays keyed to PRESENCE -- a
        # half-written triple is refused whatever its values.
        regional = triple_complete and _regional_boundary_zone_present(self.arrays)
        if triple_complete and not regional and n_cells - n_edges + n_vertices != 2:
            errors.append(
                "the boundary-mask triple is present and entirely zero, but "
                f"nCells-nEdges+nVertices = {n_cells - n_edges + n_vertices}, "
                "not the closed sphere's 2.  An all-zero triple means every "
                "element is interior, which only a global mesh can be, and a "
                "global mesh is a closed sphere -- so this file is neither a "
                "sphere nor a cull with a boundary zone, and no convention "
                "this validator holds describes it"
            )
        if bdy_present and not triple_complete:
            missing_masks = [
                name for name in REGIONAL_BOUNDARY_MASK_NAMES if name not in self.arrays
            ]
            raise MeshValidationError(
                "regional boundary masks are incomplete: "
                f"{', '.join(bdy_present)} present without {', '.join(missing_masks)}. "
                "Without the full triple the validator cannot tell an "
                "absent-neighbour sentinel from index corruption, so the mesh "
                "can be neither admitted nor refused by name"
            )
        zone_width = REGIONAL_BOUNDARY_ZONE_WIDTH
        mask_cell = mask_edge = mask_vertex = None
        if regional:
            for name, count in (
                ("bdyMaskCell", n_cells),
                ("bdyMaskEdge", n_edges),
                ("bdyMaskVertex", n_vertices),
            ):
                mask = np.asarray(self.arrays[name])
                if mask.dtype.kind not in "iu":
                    raise MeshValidationError(
                        f"{name} has dtype {mask.dtype}, not an integer ring "
                        "number: a non-integer ring stages an element in no "
                        "boundary ring at all"
                    )
                if mask.shape != (count,):
                    raise MeshValidationError(
                        f"{name} has shape {mask.shape}, expected ({count},): "
                        "a mask that does not cover every element leaves "
                        "elements with no ring assignment"
                    )
                bad_mask = np.flatnonzero((mask < 0) | (mask > zone_width))
                if bad_mask.size:
                    raise MeshValidationError(
                        f"{name} carries ring numbers outside [0,{zone_width}] "
                        f"at indices {_bad_examples(bad_mask)}: the boundary "
                        f"zone is {zone_width} rings, so a value outside them "
                        "belongs to no ring and the runtime would neither "
                        "force nor freely integrate the element"
                    )
            mask_cell = np.asarray(self.arrays["bdyMaskCell"], dtype=np.int64)
            mask_edge = np.asarray(self.arrays["bdyMaskEdge"], dtype=np.int64)
            mask_vertex = np.asarray(self.arrays["bdyMaskVertex"], dtype=np.int64)

        expected_shapes = {
            "nEdgesOnCell": (n_cells,),
            "cellsOnCell": (n_cells, max_edges),
            "edgesOnCell": (n_cells, max_edges),
            "verticesOnCell": (n_cells, max_edges),
            "cellsOnEdge": (n_edges, 2),
            "verticesOnEdge": (n_edges, 2),
            "cellsOnVertex": (n_vertices, vertex_degree),
            "edgesOnVertex": (n_vertices, vertex_degree),
            "nEdgesOnEdge": (n_edges,),
            "edgesOnEdge": (n_edges, max_edges2),
            "weightsOnEdge": (n_edges, max_edges2),
            "dcEdge": (n_edges,),
            "dvEdge": (n_edges,),
            "areaCell": (n_cells,),
            "areaTriangle": (n_vertices,),
            "kiteAreasOnVertex": (n_vertices, vertex_degree),
        }
        missing_arrays = [name for name in expected_shapes if name not in self.arrays]
        if missing_arrays:
            raise MeshValidationError(
                f"missing required mesh fields: {', '.join(missing_arrays)}"
            )
        wrong_shapes = {
            name: (self.arrays[name].shape, expected)
            for name, expected in expected_shapes.items()
            if self.arrays[name].shape != expected
        }
        if wrong_shapes:
            raise MeshValidationError(f"mesh field shapes disagree: {wrong_shapes}")

        n_edges_on_cell = np.asarray(self.nEdgesOnCell, dtype=np.int64)
        n_edges_on_edge = np.asarray(self.nEdgesOnEdge, dtype=np.int64)
        cells_on_cell = np.asarray(self.cellsOnCell, dtype=np.int64)
        edges_on_cell = np.asarray(self.edgesOnCell, dtype=np.int64)
        vertices_on_cell = np.asarray(self.verticesOnCell, dtype=np.int64)
        cells_on_edge = np.asarray(self.cellsOnEdge, dtype=np.int64)
        vertices_on_edge = np.asarray(self.verticesOnEdge, dtype=np.int64)
        cells_on_vertex = np.asarray(self.cellsOnVertex, dtype=np.int64)
        edges_on_vertex = np.asarray(self.edgesOnVertex, dtype=np.int64)
        edges_on_edge = np.asarray(self.edgesOnEdge, dtype=np.int64)

        bad_counts = np.flatnonzero(
            (n_edges_on_cell < 3) | (n_edges_on_cell > max_edges)
        )
        if bad_counts.size:
            errors.append(
                "nEdgesOnCell outside [3,maxEdges] at cells "
                + _bad_examples(bad_counts)
            )
        bad_stencil_counts = np.flatnonzero(
            (n_edges_on_edge < 0) | (n_edges_on_edge > max_edges2)
        )
        if bad_stencil_counts.size:
            errors.append(
                "nEdgesOnEdge outside [0,maxEdges2] at edges "
                + _bad_examples(bad_stencil_counts)
            )

        counted_ranges = (
            ("cellsOnCell", cells_on_cell, n_edges_on_cell, n_cells),
            ("edgesOnCell", edges_on_cell, n_edges_on_cell, n_edges),
            ("verticesOnCell", vertices_on_cell, n_edges_on_cell, n_vertices),
            ("edgesOnEdge", edges_on_edge, n_edges_on_edge, n_edges),
        )
        # A sentinel in an array the cull keeps complete is a LOST element,
        # not a boundary; each message names what the loss breaks at runtime.
        regional_complete_breakage = {
            "edgesOnCell": (
                "the cull keeps every edge of a kept cell, so the cell's "
                "divergence/KE stencils would read a nonexistent edge"
            ),
            "verticesOnCell": (
                "the cull keeps every vertex of a kept cell, so the cell's "
                "vorticity gather would read a nonexistent vertex"
            ),
            "verticesOnEdge": (
                "every kept edge keeps both endpoints, so the tangential and "
                "kite terms at this edge would read a nonexistent vertex"
            ),
        }
        row_mask_for = {
            "cellsOnCell": mask_cell,
            "edgesOnCell": mask_cell,
            "verticesOnCell": mask_cell,
            "edgesOnEdge": mask_edge,
        }
        counts_safe = not bad_counts.size and not bad_stencil_counts.size
        fixed_ranges = (
            ("cellsOnEdge", cells_on_edge, n_cells, mask_edge),
            ("verticesOnEdge", vertices_on_edge, n_vertices, mask_edge),
            ("cellsOnVertex", cells_on_vertex, n_cells, mask_vertex),
            ("edgesOnVertex", edges_on_vertex, n_edges, mask_vertex),
        )
        positive_metrics = (
            "dcEdge",
            "dvEdge",
            "areaCell",
            "areaTriangle",
            "kiteAreasOnVertex",
        )

        # THE COST THIS REMOVES (NOAA RRFS v2 conus3km, 2,341,818 cells and 7.03 M edges, V8X6000, 2026-10-29):
        # these checks were 5.6-8.7 s of the conus3km start-up's 10-11 s Mesh.from_netcdf, 1.4-2.1 threads busy
        # on a 128-core host, one whole-mesh NumPy check after another.  They only read the arrays, so the
        # independent ones run at once on host threads (_ValidationChecks) and a row-wise range check runs in
        # row blocks whose flags and row indices are joined in row order.  Each check keeps its own predicates,
        # a check whose gate (counts_safe, fixed_valid, topology_ranges_valid) is false still never runs, the
        # refusals are listed in the serial order, and an exception a check raises is raised where the serial
        # order raised it.
        with _ValidationChecks() as checks:

            def counted_range(name: str, array: NDArray[np.int64], counts: NDArray[np.int64],
                              upper: int) -> tuple[list[str], bool]:
                found: list[str] = []
                sentinel_allowed = regional and name in REGIONAL_SENTINEL_ARRAYS
                lower = -1 if sentinel_allowed else 0
                slots = np.arange(array.shape[1])[None, :]
                used_outside: list[bool] = []
                non_canonical: list[bool] = []
                sentinel_blocks: dict[int, NDArray[np.intp]] = {}

                def check(block: slice) -> None:
                    rows = array[block]
                    used_mask = slots < counts[block, None]
                    used = rows[used_mask]
                    if np.flatnonzero((used < lower) | (used >= upper)).size:
                        used_outside.append(True)
                    if regional:
                        sentinel_blocks[block.start] = np.flatnonzero(
                            ((rows == -1) & used_mask).any(axis=1)
                        ) + block.start
                    padding = rows[~used_mask]
                    if np.any(padding != -1):
                        non_canonical.append(True)

                run_blocks(check, array.shape[0], elements=array.size, worker_limit=8)
                if used_outside:
                    found.append(
                        f"{name} has used entries outside [0,{upper})"
                    )
                if regional:
                    sentinel_rows = _joined_rows(sentinel_blocks)
                    if sentinel_allowed:
                        misplaced = sentinel_rows[
                            row_mask_for[name][sentinel_rows] != zone_width
                        ]
                        if misplaced.size:
                            found.append(
                                f"{name} marks a neighbour absent below the "
                                "outermost ring at rows "
                                + _bad_examples(misplaced)
                                + f": only ring-{zone_width} elements border "
                                "the cut, so an absent reference deeper in "
                                "the zone means the cull cut inside the "
                                "relaxation rings"
                            )
                    elif sentinel_rows.size:
                        lost = "an edge" if name == "edgesOnCell" else "a corner"
                        found.append(
                            f"{name} lost {lost} at cells "
                            + _bad_examples(sentinel_rows)
                            + ": "
                            + regional_complete_breakage[name]
                        )
                if non_canonical:
                    found.append(f"{name} padding is not canonical -1")
                return found, bool(used_outside)

            def fixed_range(name: str, array: NDArray[np.int64], upper: int,
                            row_mask: NDArray[np.int64] | None) -> tuple[list[str], bool]:
                found: list[str] = []
                valid = True
                sentinel_allowed = regional and name in REGIONAL_SENTINEL_ARRAYS
                lower = -1 if sentinel_allowed else 0
                outside: list[bool] = []
                sentinel_blocks: dict[int, NDArray[np.intp]] = {}

                def check(block: slice) -> None:
                    rows = array[block]
                    if np.argwhere((rows < lower) | (rows >= upper)).size:
                        outside.append(True)
                    if regional:
                        sentinel_blocks[block.start] = np.flatnonzero(
                            (rows == -1).any(axis=1)
                        ) + block.start

                run_blocks(check, array.shape[0], elements=array.size, worker_limit=8)
                if outside:
                    valid = False
                    found.append(f"{name} contains entries outside [0,{upper})")
                if regional:
                    sentinel_rows = _joined_rows(sentinel_blocks)
                    if sentinel_allowed:
                        misplaced = sentinel_rows[row_mask[sentinel_rows] != zone_width]
                        if misplaced.size:
                            valid = False
                            found.append(
                                f"{name} marks a neighbour absent below the "
                                "outermost ring at rows "
                                + _bad_examples(misplaced)
                                + f": only ring-{zone_width} elements border the "
                                "cut, so an absent reference deeper in the zone "
                                "means the cull cut inside the relaxation rings"
                            )
                    elif sentinel_rows.size:
                        valid = False
                        found.append(
                            "verticesOnEdge lost an endpoint at edges "
                            + _bad_examples(sentinel_rows)
                            + ": "
                            + regional_complete_breakage["verticesOnEdge"]
                        )
                return found, valid

            def metric_value_errors() -> list[str]:
                found: list[str] = []
                for name in positive_metrics:
                    metric = np.asarray(self.arrays[name])
                    if not np.all(np.isfinite(metric)) or np.any(metric <= 0):
                        found.append(f"{name} must be finite and strictly positive")
                for optional in ("meshDensity", "nominalMinDc"):
                    if optional in self.arrays:
                        metric = np.asarray(self.arrays[optional])
                        if not np.all(np.isfinite(metric)) or np.any(metric <= 0):
                            found.append(f"{optional} must be finite and strictly positive")
                if "angleEdge" in self.arrays:
                    angles = np.asarray(self.angleEdge)
                    angle_tol = 1.0e-5 if angles.dtype.itemsize <= 4 else 1.0e-12
                    if (
                        not np.all(np.isfinite(angles))
                        or np.any(np.abs(angles) > np.pi + angle_tol)
                    ):
                        found.append("angleEdge must be finite and in [-pi,pi]")
                return found

            metric_arrays = [
                np.asarray(self.arrays[name])
                for name in positive_metrics
                if name in self.arrays
            ]

            def measured_metric_rtol() -> float:
                # Storage precision is MEASURED from the values, not read off the
                # dtype: binary64 arrays holding only binary32 numbers carry binary32
                # precision and get the binary32 bound.
                has_float32 = any(
                    _stored_significant_bits(array) <= 24 for array in metric_arrays
                )
                return _BINARY32_METRIC_RTOL if has_float32 else _BINARY64_MESH_RTOL

            try:
                closure_radius: float | None = float(self.attrs["sphere_radius"])
            except (KeyError, TypeError, ValueError):
                closure_radius = None

            def measured_closure() -> dict[str, Any]:
                return closure_tolerance(
                    {name: self.arrays[name] for name in ("areaCell", "areaTriangle", "kiteAreasOnVertex")},
                    radius=closure_radius,
                    edge_lengths=np.concatenate(
                        (
                            np.asarray(self.dcEdge, dtype=np.float64).ravel(),
                            np.asarray(self.dvEdge, dtype=np.float64).ravel(),
                        )
                    ),
                )

            # The checks no gate holds back start now.
            counted = (
                [checks.submit(counted_range, *row) for row in counted_ranges]
                if counts_safe else []
            )
            fixed = [checks.submit(fixed_range, *row) for row in fixed_ranges]
            weight_check = checks.submit(
                _edge_weight_errors, np.asarray(self.weightsOnEdge), n_edges_on_edge,
                check_padding=not bad_stencil_counts.size,
            )
            metric_check = checks.submit(metric_value_errors)
            metric_rtol_check = checks.submit(measured_metric_rtol)
            closure_check = checks.submit(measured_closure)

            used_outside_any = False
            for future in counted:
                found, used_outside = future.result()
                errors.extend(found)
                used_outside_any = used_outside_any or used_outside
            fixed_valid = True
            for future in fixed:
                found, valid = future.result()
                errors.extend(found)
                fixed_valid = fixed_valid and valid

            if regional:
                euler = n_cells - n_edges + n_vertices
                if euler != 1:
                    errors.append(
                        f"regional mesh is not a bounded disk: "
                        f"nCells-nEdges+nVertices = {euler}, not 1; the cull kept "
                        "or dropped elements inconsistently (a hole, a tear, or a "
                        "duplicate), so the boundary rings do not enclose the "
                        "interior"
                    )
                # The closed-sphere incidence identities, corrected by exactly the
                # measured sentinel counts: an edge with one absent cell appears
                # once in edgesOnCell instead of twice, and a culled edge leaves
                # one empty slot per kept endpoint in edgesOnVertex.
                coe_sentinels = int(np.count_nonzero(cells_on_edge == -1))
                eov_sentinels = int(np.count_nonzero(edges_on_vertex == -1))
                if (
                    not bad_counts.size
                    and int(n_edges_on_cell.sum()) != 2 * n_edges - coe_sentinels
                ):
                    errors.append(
                        f"sum(nEdgesOnCell) = {int(n_edges_on_cell.sum())} does "
                        f"not equal 2*nEdges minus cut-adjacent slots "
                        f"({2 * n_edges - coe_sentinels}): cells reference edges "
                        "the file does not carry, or the file carries edges no "
                        "kept cell owns"
                    )
                if vertex_degree * n_vertices != 2 * n_edges + eov_sentinels:
                    errors.append(
                        f"vertexDegree*nVertices = {vertex_degree * n_vertices} "
                        f"does not equal 2*nEdges plus culled edge slots "
                        f"({2 * n_edges + eov_sentinels}): vertices carry more or "
                        "fewer edge slots than the kept edges provide"
                    )
            else:
                if n_cells - n_edges + n_vertices != 2:
                    errors.append(
                        "closed-sphere Euler characteristic is not nCells-nEdges+nVertices=2"
                    )
                if not bad_counts.size and int(n_edges_on_cell.sum()) != 2 * n_edges:
                    errors.append("sum(nEdgesOnCell) does not equal 2*nEdges")
                if vertex_degree * n_vertices != 2 * n_edges:
                    errors.append("vertexDegree*nVertices does not equal 2*nEdges")

            # Only a counted range writes "used entries outside", and every
            # counted-range refusal is in ``errors`` by now; the regional mask
            # refusals that the serial order listed before this test never carry
            # those words.
            topology_ranges_valid = (
                counts_safe
                and fixed_valid
                and not any("used entries outside" in error for error in errors)
            )

            regional_checks: list[Future[list[str]]] = []
            if regional and counts_safe and fixed_valid:
                huge = np.iinfo(np.int64).max

                def edge_mask_errors() -> list[str]:
                    found: list[str] = []
                    orphan_edges = _blocked_flatnonzero(
                        lambda block: (cells_on_edge[block] < 0).all(axis=1),
                        cells_on_edge.shape[0], elements=cells_on_edge.size,
                    )
                    if orphan_edges.size:
                        found.append(
                            "edges with no present cell at "
                            + _bad_examples(orphan_edges)
                            + ": every kept edge borders at least one kept cell; an "
                            "orphan edge has no column to take a tendency from"
                        )
                    else:

                        def wrong_edge(block: slice) -> NDArray[np.bool_]:
                            adjacent_masks = np.where(
                                cells_on_edge[block] >= 0,
                                mask_cell[np.clip(cells_on_edge[block], 0, None)],
                                huge,
                            )
                            return adjacent_masks.min(axis=1) != mask_edge[block]

                        wrong_edges = _blocked_flatnonzero(
                            wrong_edge, cells_on_edge.shape[0], elements=cells_on_edge.size,
                        )
                        if wrong_edges.size:
                            found.append(
                                "bdyMaskEdge is not the minimum of its present cells' "
                                "masks at edges "
                                + _bad_examples(wrong_edges)
                                + ": zone-staged edge tendencies key on this ring "
                                "number, so a wrong value forces or frees the edge in "
                                "the wrong ring"
                            )
                    return found

                def vertex_mask_errors() -> list[str]:
                    found: list[str] = []
                    orphan_vertices = _blocked_flatnonzero(
                        lambda block: (cells_on_vertex[block] < 0).all(axis=1),
                        cells_on_vertex.shape[0], elements=cells_on_vertex.size,
                    )
                    if orphan_vertices.size:
                        found.append(
                            "vertices with no present cell at "
                            + _bad_examples(orphan_vertices)
                            + ": every kept vertex is a corner of at least one kept "
                            "cell; an orphan vertex has no kite to weight"
                        )
                    else:

                        def wrong_vertex(block: slice) -> NDArray[np.bool_]:
                            vertex_masks = np.where(
                                cells_on_vertex[block] >= 0,
                                mask_cell[np.clip(cells_on_vertex[block], 0, None)],
                                huge,
                            )
                            return vertex_masks.min(axis=1) != mask_vertex[block]

                        wrong_vertices = _blocked_flatnonzero(
                            wrong_vertex, cells_on_vertex.shape[0], elements=cells_on_vertex.size,
                        )
                        if wrong_vertices.size:
                            found.append(
                                "bdyMaskVertex is not the minimum of its present "
                                "cells' masks at vertices "
                                + _bad_examples(wrong_vertices)
                                + ": zone-staged vorticity terms key on this ring "
                                "number, so a wrong value stages the vertex in the "
                                "wrong ring"
                            )
                    return found

                def ring_errors() -> list[str]:
                    found: list[str] = []
                    # Each row block holds whole cells in stored order, so the
                    # joined pairs are the row-major boolean selection and the
                    # joined per-block np.unique is the sorted whole-mesh one.
                    neighbour_blocks: dict[int, NDArray[np.int64]] = {}
                    source_blocks: dict[int, NDArray[np.int64]] = {}
                    jump_blocks: dict[int, NDArray[np.int64]] = {}

                    def pairs(block: slice) -> None:
                        rows = cells_on_cell[block]
                        used_slots = (
                            np.arange(max_edges)[None, :] < n_edges_on_cell[block, None]
                        ) & (rows >= 0)
                        neighbours = rows[used_slots]
                        sources = np.broadcast_to(
                            np.arange(block.start, block.start + rows.shape[0])[:, None], rows.shape
                        )[used_slots]
                        jumps = (
                            np.abs(mask_cell[sources] - mask_cell[neighbours]) > 1
                        )
                        neighbour_blocks[block.start] = neighbours
                        source_blocks[block.start] = sources
                        jump_blocks[block.start] = np.unique(sources[jumps])

                    run_blocks(pairs, n_cells, elements=cells_on_cell.size, worker_limit=8)
                    neighbour_cells = _joined_rows(neighbour_blocks)
                    source_cells = _joined_rows(source_blocks)
                    jumped = _joined_rows(jump_blocks)
                    if jumped.size:
                        found.append(
                            "bdyMaskCell jumps by more than 1 between neighbouring "
                            "cells at "
                            + _bad_examples(jumped)
                            + ": rings advance one cell per ring from the cut, so a "
                            "jump means renumbered or torn rings"
                        )
                    for label, mask in (
                        ("bdyMaskCell", mask_cell),
                        ("bdyMaskEdge", mask_edge),
                        ("bdyMaskVertex", mask_vertex),
                    ):
                        ring_counts = np.bincount(mask, minlength=zone_width + 1)
                        if ring_counts[0] == 0:
                            found.append(
                                f"{label} has no interior (ring-0) elements: the "
                                "subset is all boundary zone, leaving nothing for the "
                                "relaxation rings to protect"
                            )
                        empty_rings = [
                            ring
                            for ring in range(1, zone_width + 1)
                            if ring_counts[ring] == 0
                        ]
                        if empty_rings:
                            found.append(
                                f"{label} rings {empty_rings} are empty: the "
                                f"{zone_width}-ring zone is a contiguous shell, so an "
                                "empty ring means renumbered masks and zone-staged "
                                "tendencies skipping a ring"
                            )
                    found.extend(
                        regional_ring_shell_errors(
                            mask_cell, source_cells, neighbour_cells, zone_width
                        )
                    )
                    return found

                regional_checks = [checks.submit(edge_mask_errors),
                                   checks.submit(vertex_mask_errors),
                                   checks.submit(ring_errors)]

            topology_check: Future[list[str]] | None = None
            if topology_ranges_valid:

                def topology_check_errors() -> list[str]:
                    from .rust_hostprep import topology_errors

                    found: list[str] = []
                    # The slot pairing of verticesOnCell with edgesOnCell is a
                    # per-file convention (VERTEX_SLOT_CONVENTIONS), detected here
                    # and checked under the convention the file holds.  THE BREAKAGE
                    # THE OLD FIXED PAIRING CAUSED (measured 2026-10-05): RRFS v2's
                    # public conus3km grid stores every cell edge-end, so a check
                    # that assumed edge-start refused a mesh whose rings are
                    # complete and reciprocal, at cells 0..4.  The stored array is
                    # not modified; the Rust check reads an index map of it.
                    slot_convention = vertex_slot_convention(
                        n_edges_on_cell, edges_on_cell, vertices_on_cell, vertices_on_edge
                    )
                    if slot_convention["convention"] == "mixed":
                        found.append(
                            "verticesOnCell mixes slot conventions: "
                            f"{slot_convention['edge_start_cells']} cells put vertex j "
                            "at the start of edge j and "
                            f"{slot_convention['edge_end_cells']} at its end.  No "
                            "producer writes a mixed file; a row rotated by a "
                            "corrupting writer looks exactly like this, and accepting "
                            "it per cell would let that corruption pass as a convention"
                        )
                    found.extend(topology_errors(
                        n_vertices=n_vertices, max_edges2=max_edges2,
                        vertex_degree=vertex_degree, regional=regional,
                        counts=n_edges_on_cell, edges=edges_on_cell,
                        cells_on_cell=cells_on_cell,
                        vertices_on_cell=slot_convention["view"],
                        cells_on_edge=cells_on_edge, vertices_on_edge=vertices_on_edge,
                        cells_on_vertex=cells_on_vertex, edges_on_vertex=edges_on_vertex,
                        edge_counts=n_edges_on_edge, edges_on_edge=edges_on_edge,
                    ))
                    return found

                topology_check = checks.submit(topology_check_errors)

            area_cell = np.asarray(self.areaCell, dtype=np.float64)
            area_triangle = np.asarray(self.areaTriangle, dtype=np.float64)
            kite_areas = np.asarray(self.kiteAreasOnVertex, dtype=np.float64)

            def closure_fails(total: NDArray[np.float64], area: NDArray[np.float64], rtol: Any) -> bool:
                return not bool(np.all(np.abs(total - area) <= rtol * np.abs(area)))

            def closure_gap(closure: Mapping[str, Any], total: NDArray[np.float64],
                            area: NDArray[np.float64], rtol: Any) -> str:
                relative = np.abs(total - area) / np.maximum(np.abs(area), np.finfo(np.float64).tiny)
                bound = np.broadcast_to(np.asarray(rtol, dtype=np.float64), relative.shape)
                ratio = np.where(np.isfinite(relative), relative / bound, np.inf)
                worst = int(np.argmax(ratio)) if ratio.size else 0
                observed = float(relative[worst]) if relative.size else 0.0
                return (
                    f"worst relative gap {observed:.3e} at index {worst} exceeds its "
                    f"element bound {float(bound[worst]):.3e}; mesh-wide "
                    + _closure_statement(closure)
                )

            # The closure and coordinate checks need the measured storage
            # precision.  A measurement that raised ends the serial order there,
            # so nothing after it starts; its exception is raised in order below.
            closure_checks: list[Future[list[str]]] = []
            measured = (metric_rtol_check.exception() is None
                        and closure_check.exception() is None)
            metric_rtol = metric_rtol_check.result() if measured else None
            if measured:
                closure = closure_check.result()
                closure_rtol = closure["rtol"]

                # The conditioning term is per ELEMENT: each triangle and cell is
                # held to 16 u / theta**2 for its OWN shortest edge, so the coarse
                # part of a graded mesh is not loosened by its finest element.
                # Without usable edge indices the mesh-wide bound stands in.
                per_element = (
                    closure["stored_significant_bits"] > 24
                    and closure["theta_min_radians"] is not None
                    and topology_ranges_valid
                )
                if per_element:
                    edge_theta = np.minimum(
                        np.asarray(self.dcEdge, dtype=np.float64),
                        np.asarray(self.dvEdge, dtype=np.float64),
                    ) / float(closure_radius)
                    edge_theta = np.where(
                        np.isfinite(edge_theta) & (edge_theta > 0), edge_theta, np.inf
                    )

                def element_rtol(edges: NDArray[np.int64], used: NDArray[np.bool_]) -> NDArray[np.float64]:
                    theta = np.where(used, edge_theta[np.clip(edges, 0, None)], np.inf).min(axis=1)
                    conditioning = (
                        _CLOSURE_CONDITIONING_FACTOR * float(np.ldexp(1.0, -53))
                        / np.where(np.isfinite(theta), theta * theta, np.inf)
                    )
                    return np.maximum(_BINARY64_MESH_RTOL, conditioning)

                def triangle_closure_errors() -> list[str]:
                    rtol_vertex: Any = closure_rtol
                    if per_element:
                        rtol_vertex = element_rtol(edges_on_vertex, edges_on_vertex >= 0)
                    triangle_sum = kite_areas.sum(axis=1)
                    if closure_fails(triangle_sum, area_triangle, rtol_vertex):
                        return [
                            "kiteAreasOnVertex rows do not sum to areaTriangle: "
                            + closure_gap(closure, triangle_sum, area_triangle, rtol_vertex)
                            + ".  The vertex vorticity is divided by areaTriangle and "
                            "carried to cells by these kites, so kites that do not tile "
                            "the triangle move circulation between cells"
                        ]
                    return []

                def cell_closure_errors() -> list[str]:
                    rtol_cell: Any = closure_rtol
                    if per_element:
                        rtol_cell = element_rtol(
                            edges_on_cell,
                            (np.arange(max_edges)[None, :] < n_edges_on_cell[:, None])
                            & (edges_on_cell >= 0),
                        )
                    cell_kite_sum = np.zeros(n_cells, dtype=np.float64)
                    # np.add.at WRAPS a negative index: a -1 sentinel folds the
                    # absent neighbour's kite into cell nCells-1 and fabricates a
                    # failure there against a defect-free mesh.  Measured on the
                    # native regional culls: relative error 100.46 (quick cull) and
                    # 394.39 (reference cull) at the wrap target.  Present slots
                    # only; on a global mesh no sentinel reaches here, so the sum is
                    # unchanged.
                    flat_cells = cells_on_vertex.ravel()
                    flat_kites = kite_areas.ravel()
                    present_slots = flat_cells >= 0
                    np.add.at(
                        cell_kite_sum, flat_cells[present_slots], flat_kites[present_slots]
                    )
                    if closure_fails(cell_kite_sum, area_cell, rtol_cell):
                        return [
                            "kites belonging to each cell do not sum to areaCell: "
                            + closure_gap(closure, cell_kite_sum, area_cell, rtol_cell)
                            + ".  areaCell divides every flux divergence, so a cell "
                            "whose kites do not tile it gains or loses mass the "
                            "fluxes did not carry"
                        ]
                    return []

                closure_checks.append(checks.submit(triangle_closure_errors))
                if fixed_valid:
                    closure_checks.append(checks.submit(cell_closure_errors))

            # Spherical coordinates: the three families start now, the two arc
            # checks once the cell and vertex families are known.
            on_sphere = str(self.attrs.get("on_a_sphere", "NO")).strip().upper() == "YES"
            radius = -1.0
            radius_errors: list[str] = []
            families: list[tuple[Future[Any] | None, list[str]]] = []
            if on_sphere and measured:
                try:
                    radius = float(self.attrs["sphere_radius"])
                except (KeyError, TypeError, ValueError):
                    radius_errors.append("spherical mesh lacks a positive sphere_radius attribute")
                    radius = -1.0
                if not np.isfinite(radius) or radius <= 0:
                    radius_errors.append("sphere_radius must be finite and strictly positive")

                def family_checks(suffix: str, count: int, names: tuple[str, ...],
                                  spherical_names: tuple[str, str]
                                  ) -> tuple[NDArray[np.float64] | None, list[str]]:
                    xyz = np.stack(
                        [np.asarray(self.arrays[name], dtype=np.float64) for name in names],
                        axis=1,
                    )
                    lat = np.asarray(self.arrays[spherical_names[0]], dtype=np.float64)
                    lon = np.asarray(self.arrays[spherical_names[1]], dtype=np.float64)
                    if xyz.shape != (count, 3) or lat.shape != (count,) or lon.shape != (count,):
                        return None, [f"{suffix} spherical coordinate shapes are invalid"]
                    return _spherical_coordinate_checks(
                        xyz, lat, lon, radius, metric_rtol, suffix,
                    )

                if radius > 0:
                    coordinate_shapes = {
                        "Cell": n_cells,
                        "Edge": n_edges,
                        "Vertex": n_vertices,
                    }
                    for suffix, count in coordinate_shapes.items():
                        names = tuple(axis + suffix for axis in "xyz")
                        spherical_names = ("lat" + suffix, "lon" + suffix)
                        missing = [
                            name for name in (*names, *spherical_names) if name not in self.arrays
                        ]
                        if missing:
                            families.append((None, [
                                f"spherical mesh lacks coordinate fields: {', '.join(missing)}"
                            ]))
                            continue
                        families.append((checks.submit(
                            family_checks, suffix, count, names, spherical_names), []))

            coordinates: dict[str, NDArray[np.float64]] = {}
            arc_checks: list[Future[list[str]]] = []
            families_raised = any(
                future is not None and future.exception() is not None
                for future, _ in families
            )
            if not families_raised:
                for (future, _), suffix in zip(families, ("Cell", "Edge", "Vertex")):
                    if future is not None:
                        normalized, _ = future.result()
                        if normalized is not None:
                            coordinates[suffix] = normalized
            if not families_raised and topology_ranges_valid and all(
                suffix in coordinates for suffix in ("Cell", "Vertex")
            ):

                def arc_errors(family: str, stored_name: str, endpoints: NDArray[np.int64],
                               clip_missing: bool, verifiable_only: bool, message: str) -> list[str]:
                    expected = _spherical_arc_lengths(
                        coordinates[family], endpoints, radius, clip_missing=clip_missing,
                    )
                    # The floor comes from the dtype the COORDINATES are
                    # stored in, not from the metric arrays: the quantization
                    # being tolerated is theirs.  A file may carry binary64
                    # metrics over binary32 coordinates, and reading the floor
                    # off the metrics would leave that file with no floor at
                    # all.
                    coordinate_dtype = min(
                        (
                            np.asarray(self.arrays[axis + family]).dtype
                            for axis in "xyz"
                        ),
                        key=lambda dtype: dtype.itemsize,
                    )
                    arc_rtol, arc_atol = spherical_arc_tolerance(
                        radius, coordinate_dtype, metric_rtol
                    )
                    stored = np.asarray(self.arrays[stored_name], dtype=np.float64)
                    compared = expected
                    if verifiable_only:
                        # Indexing coordinates with a regional -1 sentinel would
                        # WRAP to the last cell and fabricate an arc-length
                        # disagreement on every cut-adjacent edge.  The culled
                        # neighbour's coordinates are not in the file, so its
                        # dcEdge (copied from the parent) is unverifiable here
                        # and exempt; both dv endpoints are always present.
                        verifiable = (cells_on_edge >= 0).all(axis=1)
                        stored = stored[verifiable]
                        compared = expected[verifiable]
                    if not _blocked_allclose(stored, compared, rtol=arc_rtol, atol=arc_atol):
                        return [message]
                    return []

                arc_checks = [
                    checks.submit(
                        arc_errors, "Cell", "dcEdge", cells_on_edge, True, True,
                        "dcEdge disagrees with spherical cell-center arc length",
                    ),
                    checks.submit(
                        arc_errors, "Vertex", "dvEdge", vertices_on_edge, False, False,
                        "dvEdge disagrees with spherical vertex arc length",
                    ),
                ]

            # ---- the refusals, in the serial order ----
            for future in regional_checks:
                errors.extend(future.result())
            if topology_check is not None:
                errors.extend(topology_check.result())
            errors.extend(weight_check.result())
            errors.extend(metric_check.result())
            metric_rtol_check.result()
            closure = closure_check.result()
            for future in closure_checks:
                errors.extend(future.result())

            # ---- a DECLARED coordinate frame must be the one the file holds ---
            #
            # The storage tolerance below is derived from the dtype the file
            # carries, not from this attribute, and is already right without it.
            # What a declaration adds is a record of what the PRODUCER stored: a
            # mesh rewritten from binary64 to binary32 loses 5.4e8 of coordinate
            # precision and passes every check here, because every check derives
            # its tolerance from the dtype it finds and so moves with the rewrite.
            #
            # AN UNDECLARED FILE IS READ BY ITS DTYPE, which is already correct
            # and is what every published mesh relies on -- x1.40962.grid.nc and
            # x4.163842.grid.nc store binary64 coordinates and carry no attribute.
            # The declaration, written by rw-mpas `staticfile::coordframe` on files
            # with no native MPAS-A counterpart, records what the PRODUCER stored,
            # so a file rewritten at a different width is caught instead of being
            # read at the width it now happens to have -- with every tolerance
            # having quietly moved with it.
            frame_errors = _coordinate_frame_errors(self.attrs, self.arrays)
            errors.extend(frame_errors)

            if on_sphere:
                errors.extend(radius_errors)
                if np.isfinite(radius) and radius > 0 and not regional:
                    # A regional disk covers a fraction of the sphere (7.25
                    # percent on the measured quick cull), so the whole-sphere
                    # closure identities apply only to the global contract.  The
                    # local kite/area identities above and the arc-length checks
                    # below still bind every regional element.
                    if not np.isclose(
                        area_cell.sum(), 4.0 * np.pi * radius * radius, rtol=closure["rtol"]
                    ):
                        errors.append("sum(areaCell) does not equal 4*pi*sphere_radius**2")
                    elif not np.isclose(
                        area_triangle.sum(), 4.0 * np.pi * radius * radius, rtol=closure["rtol"]
                    ):
                        errors.append("sum(areaTriangle) does not equal 4*pi*sphere_radius**2")
                for future, found in families:
                    errors.extend(found)
                    if future is not None:
                        errors.extend(future.result()[1])
                for future in arc_checks:
                    errors.extend(future.result())

        if errors:
            raise MeshValidationError(
                "MPAS mesh validation failed:\n - " + "\n - ".join(errors)
            )
        return self


def reconcile_grid_rotate_longitudes(
    mesh: Mesh, *, _file_digest: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Reconcile only the proven MPAS-Tools ``grid_rotate`` pi defect.

    A mesh that is already Cartesian/lat-lon consistent is only mapped to the
    established ``[-pi, pi)`` representation.  An inconsistent mesh is
    admitted only when all three entity families reproduce the pinned
    ``grid_rotate.f90`` ``convert_xl`` formula to at most two binary64 ULP.
    The branch-specific binary32-pi error is then removed in memory and the
    unchanged strict :meth:`Mesh.validate` contract must pass.

    No input file, Cartesian coordinate, latitude, topology, metric, or
    validation tolerance is modified.
    """

    if str(mesh.attrs.get("on_a_sphere", "NO")).strip().upper() != "YES":
        raise MeshValidationError(
            "grid_rotate longitude reconciliation requires a spherical mesh"
        )
    try:
        radius = float(mesh.attrs["sphere_radius"])
    except (KeyError, TypeError, ValueError) as error:
        raise MeshValidationError(
            "grid_rotate longitude reconciliation requires sphere_radius"
        ) from error
    if not np.isfinite(radius) or radius <= 0.0:
        raise MeshValidationError(
            "grid_rotate longitude reconciliation requires positive sphere_radius"
        )

    staged: dict[str, NDArray[np.float64]] = {}
    originals: dict[str, NDArray[Any]] = {}
    original_sources: dict[str, str | None] = {}
    entity_evidence: dict[str, Any] = {}
    modes: list[str] = []
    changed_fields: list[str] = []

    for entity in ("Cell", "Edge", "Vertex"):
        coordinate_names = tuple(f"{axis}{entity}" for axis in "xyz")
        latitude_name = f"lat{entity}"
        longitude_name = f"lon{entity}"
        required = (*coordinate_names, latitude_name, longitude_name)
        missing = [name for name in required if name not in mesh.arrays]
        if missing:
            raise MeshValidationError(
                "grid_rotate longitude reconciliation lacks fields: "
                + ", ".join(missing)
            )
        values = {name: np.asarray(mesh.arrays[name]) for name in required}
        if any(value.dtype != np.dtype(np.float64) for value in values.values()):
            raise MeshValidationError(
                f"grid_rotate longitude reconciliation requires binary64 {entity} "
                "coordinates"
            )
        shapes = {value.shape for value in values.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise MeshValidationError(
                f"grid_rotate longitude reconciliation found invalid {entity} shapes"
            )
        if any(not np.all(np.isfinite(value)) for value in values.values()):
            raise MeshValidationError(
                f"grid_rotate longitude reconciliation found non-finite {entity} values"
            )

        x, y, z = (np.asarray(values[name], dtype=np.float64) for name in coordinate_names)
        latitude = np.asarray(values[latitude_name], dtype=np.float64)
        longitude = np.asarray(values[longitude_name], dtype=np.float64)
        xyz = np.stack((x, y, z), axis=1)
        norms = np.linalg.norm(xyz, axis=1)
        if not np.allclose(
            norms,
            radius,
            rtol=_BINARY64_MESH_RTOL,
            atol=0.0,
        ):
            raise MeshValidationError(
                f"grid_rotate longitude reconciliation refuses off-sphere {entity} xyz"
            )

        producer_latitude = np.arcsin(z / norms)
        latitude_ulp_error = _maximum_ulp_error(latitude, producer_latitude)
        authoritative_longitude = np.arctan2(y, x)
        expected_pre = _latitude_longitude_xyz(latitude, longitude, radius)
        pre_max_xyz_gap = float(np.max(np.abs(xyz - expected_pre), initial=0.0))
        cartesian_consistent = bool(
            np.allclose(
                xyz,
                expected_pre,
                rtol=_BINARY64_MESH_RTOL,
                atol=radius * _BINARY64_MESH_RTOL,
            )
        )

        producer_gap: float | None = None
        producer_ulp_error: float | None = None
        correction_counts: dict[str, int] | None = None
        class_counts: list[int] | None = None
        if cartesian_consistent:
            mode = "already_cartesian_consistent"
            corrected_unwrapped = longitude
            candidate = normalize_longitudes(corrected_unwrapped)
        else:
            mode = "grid_rotate_default_real_pi_reconciled"
            producer_longitude, multiplicity = _grid_rotate_convert_xl_longitude(x, y)
            producer_gap = float(
                np.max(np.abs(longitude - producer_longitude), initial=0.0)
            )
            producer_ulp_error = _maximum_ulp_error(longitude, producer_longitude)
            if producer_ulp_error > GRID_ROTATE_PRODUCER_MAX_ULP_ERROR:
                raise MeshValidationError(
                    f"grid_rotate longitude reconciliation refuses {entity}: raw "
                    "longitude does not reproduce the pinned producer formula "
                    f"within {GRID_ROTATE_PRODUCER_MAX_ULP_ERROR:g} ULP "
                    f"(observed {producer_ulp_error:g})"
                )
            if latitude_ulp_error > GRID_ROTATE_LATITUDE_MAX_ULP_ERROR:
                raise MeshValidationError(
                    f"grid_rotate longitude reconciliation refuses {entity}: latitude "
                    "does not reproduce convert_xl "
                    f"(observed {latitude_ulp_error:g} ULP)"
                )
            corrected_unwrapped = longitude - multiplicity * GRID_ROTATE_PI_DELTA
            candidate = normalize_longitudes(corrected_unwrapped)
            authoritative_normalized = normalize_longitudes(authoritative_longitude)
            corrected_gap = float(
                np.max(
                    np.abs(_wrapped_angle_delta(candidate, authoritative_normalized)),
                    initial=0.0,
                )
            )
            if corrected_gap > GRID_ROTATE_CORRECTED_ATAN2_ATOL:
                raise MeshValidationError(
                    f"grid_rotate longitude reconciliation refuses {entity}: branch "
                    f"correction misses Cartesian atan2 by {corrected_gap:.17g} radians"
                )
            correction_counts = {
                label: int(np.count_nonzero(multiplicity == value))
                for label, value in (
                    ("0", 0.0),
                    ("0.5", 0.5),
                    ("1", 1.0),
                    ("1.5", 1.5),
                    ("2", 2.0),
                )
            }
            class_counts = [
                correction_counts["0"],
                correction_counts["1"],
                correction_counts["2"],
            ]

        canonicalization = validate_longitude_normalization(
            corrected_unwrapped, candidate
        )
        expected_post = _latitude_longitude_xyz(latitude, candidate, radius)
        post_max_xyz_gap = float(np.max(np.abs(xyz - expected_post), initial=0.0))
        post_atan2_gap = float(
            np.max(
                np.abs(_wrapped_angle_delta(candidate, authoritative_longitude)),
                initial=0.0,
            )
        )
        with _array_digest_items(dict(values, normalizedLongitude=candidate)) as results:
            pre_hashes = dict(results)
        staged[longitude_name] = np.ascontiguousarray(candidate)
        originals[longitude_name] = mesh.arrays[longitude_name]
        original_sources[longitude_name] = mesh.variable_sources.get(longitude_name)
        modes.append(mode)
        if not np.array_equal(longitude, candidate):
            changed_fields.append(longitude_name)
        entity_evidence[entity] = {
            "mode": mode,
            "count": int(longitude.size),
            "dtype": "float64",
            "producer_formula_max_abs_gap_radians": producer_gap,
            "producer_formula_max_ulp_error": producer_ulp_error,
            "producer_latitude_max_ulp_error": latitude_ulp_error,
            "pi_error_multiplicity_counts": correction_counts,
            "delta_class_counts_0_1_2": class_counts,
            "pre_max_cartesian_component_gap": pre_max_xyz_gap,
            "post_max_cartesian_component_gap": post_max_xyz_gap,
            "pre_max_atan2_angular_gap_radians": float(
                np.max(
                    np.abs(_wrapped_angle_delta(authoritative_longitude, longitude)),
                    initial=0.0,
                )
            ),
            "post_max_atan2_angular_gap_radians": post_atan2_gap,
            "longitude_sha256_pre": pre_hashes[longitude_name],
            "longitude_sha256_post": pre_hashes["normalizedLongitude"],
            "latitude_sha256_pre": pre_hashes[latitude_name],
            "latitude_sha256_post": pre_hashes[latitude_name],
            "cartesian_sha256_pre": {
                name: pre_hashes[name] for name in coordinate_names
            },
            "cartesian_sha256_post": {
                name: pre_hashes[name] for name in coordinate_names
            },
            "canonicalization": canonicalization,
        }

    if len(set(modes)) != 1:
        raise MeshValidationError(
            "grid_rotate longitude reconciliation refuses mixed producer signatures "
            f"across Cell/Edge/Vertex: {modes}"
        )

    try:
        for name, candidate in staged.items():
            mesh.arrays[name] = candidate
            mesh.variable_sources[name] = (
                "grid_binary64_grid_rotate_default_real_pi_reconciled_from_cartesian"
                if modes[0] == "grid_rotate_default_real_pi_reconciled"
                else "grid_binary64_equivalent_longitude_range"
            )
        mesh.validate()
        unchanged = {}
        for entity, details in entity_evidence.items():
            for name, expected_hash in details["cartesian_sha256_pre"].items():
                unchanged[name] = (expected_hash, "Cartesian")
            latitude_name = f"lat{entity}"
            unchanged[latitude_name] = (details["latitude_sha256_pre"], "latitude")
        with _array_digest_items({name: mesh.arrays[name] for name in unchanged}) as results:
            for name, actual in results:
                expected, category = unchanged[name]
                if actual != expected:
                    raise RuntimeError(
                        f"longitude reconciliation changed {category} field {name}"
                    )
    except Exception:
        for name, original in originals.items():
            mesh.arrays[name] = original
            source = original_sources[name]
            if source is None:
                mesh.variable_sources.pop(name, None)
            else:
                mesh.variable_sources[name] = source
        raise

    grid_path = None if mesh.grid_path is None else Path(mesh.grid_path)
    evidence = {
        "schema": "mpas-port.grid-rotate-longitude-reconciliation/v1",
        "mode": modes[0],
        "in_memory_only": True,
        "input_grid_name": None if grid_path is None else grid_path.name,
        "input_grid_sha256": (
            None if grid_path is None or not grid_path.is_file()
            else (_sha256_file if _file_digest is None else _file_digest)(grid_path)
        ),
        "producer_authority": {
            "repository": "MPAS-Dev/MPAS-Tools",
            "head": GRID_ROTATE_SOURCE_HEAD,
            "relative_path": "mesh_tools/grid_rotate/grid_rotate.f90",
            "sha256": GRID_ROTATE_SOURCE_SHA256,
            "source_lines": "15,25,447-501",
            "pi_expression": "pii = 2.*asin(1.0)",
            "storage_kind": "RKIND=8",
            "expression_kind": "unsuffixed default real (binary32)",
        },
        "producer_default_real_pi": GRID_ROTATE_DEFAULT_REAL_PI,
        "true_binary64_pi": float(np.pi),
        "default_real_pi_delta_radians": GRID_ROTATE_PI_DELTA,
        "producer_match_max_ulp": GRID_ROTATE_PRODUCER_MAX_ULP_ERROR,
        "latitude_match_max_ulp": GRID_ROTATE_LATITUDE_MAX_ULP_ERROR,
        "corrected_atan2_atol_radians": GRID_ROTATE_CORRECTED_ATAN2_ATOL,
        "validation_tolerance_changed": False,
        "strict_mesh_validation_after_reconciliation": "passed",
        "changed_longitude_fields": changed_fields,
        "cartesian_latitude_topology_metrics_changed": False,
        "entities": entity_evidence,
    }
    mesh.provenance["grid_rotate_longitude_reconciliation"] = evidence
    return evidence


def load_precision_preserving_mesh_pair(
    grid_path: str | Path,
    static_path: str | Path,
) -> tuple[Mesh, Mesh, dict[str, Any]]:
    """Build the precision pair while independent input hashes run beside it.

    Hash workers read plain files only. All netCDF calls and mesh admission
    stay on the caller thread. Each file retains its original SHA-256 stream;
    a changed file version is hashed again at the original consumption point.
    """
    from .file_digest import MIN_CACHED_BYTES, _version

    grid = Path(grid_path).expanduser().resolve(strict=True)
    static = Path(static_path).expanduser().resolve(strict=True)
    workers = min(2, threads())
    if workers == 1 or max(grid.stat().st_size, static.stat().st_size) < MIN_CACHED_BYTES:
        return _load_precision_preserving_mesh_pair(grid, static)

    def fingerprint(path: Path) -> tuple[str, str, str]:
        before = _version(path.stat())
        value = _sha256_file(path)
        return before, _version(path.stat()), value

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-mesh-digest") as pool:
        pending = {path: pool.submit(fingerprint, path) for path in dict.fromkeys((grid, static))}

        def digest(path: Path) -> str:
            try:
                before, after, value = pending[path].result()
                if before == after == _version(path.stat()):
                    return value
            except OSError:
                # Retry at the old call site, retaining admission/error order.
                pass
            return _sha256_file(path)

        return _load_precision_preserving_mesh_pair(grid, static, _file_digest=digest)


def _load_precision_preserving_mesh_pair(
    grid_path: str | Path, static_path: str | Path, *,
    _file_digest: Callable[[Path], str] | None = None,
) -> tuple[Mesh, Mesh, dict[str, Any]]:
    """Merge static fields while retaining binary64 official grid geometry.

    The returned dynamics mesh uses the static file's Earth radius, metrics,
    terrain and other initialized fields.  Its Cartesian coordinates are the
    official grid coordinates scaled to that radius.  Both it and the
    grid-only output mesh use the exact trig-equivalent ``[-pi, pi)``
    longitude representation.  No mesh-validation tolerance is changed.
    """

    if _file_digest is None:
        _file_digest = _sha256_file
    grid = Path(grid_path).expanduser().resolve(strict=True)
    static = Path(static_path).expanduser().resolve(strict=True)
    grid_data: _NetCDFData | None = None

    def read_grid_once(path: Path) -> _NetCDFData:
        nonlocal grid_data
        if grid_data is None:
            data = _read_netcdf(path)
            # Canonicalization and longitude reconciliation replace dictionary
            # entries, leaving the original stored arrays intact for the overlay.
            grid_data = (dict(data[0]), dict(data[1]), dict(data[2]),
                         dict(data[3]), dict(data[4]))
            return data
        dimensions, attrs, arrays, variable_dimensions, variable_attrs = grid_data
        grid_data = None
        # The two carriers have always owned independent arrays. Copy their
        # stored words instead of decoding the entire grid a second time.
        # Bound bandwidth contention, and retain C/F order and attribute types.
        workers = min(4, threads())
        if workers == 1:
            copies = {name: value.copy(order="K") for name, value in arrays.items()}
        else:
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="hex-grid-copy") as pool:
                copies = dict(zip(arrays, pool.map(lambda a: a.copy(order="K"), arrays.values())))
        return (
            dict(dimensions),
            {name: _plain_attribute(value) for name, value in attrs.items()},
            copies,
            dict(variable_dimensions),
            {name: {key: _plain_attribute(value) for key, value in values.items()}
             for name, values in variable_attrs.items()},
        )

    output_mesh = Mesh.from_netcdf(grid, validate=False, _grid_reader=read_grid_once)
    longitude_reconciliation = reconcile_grid_rotate_longitudes(output_mesh, _file_digest=_file_digest)
    dynamics_mesh = Mesh.from_netcdf(grid, static, validate=False, _grid_reader=read_grid_once)

    raw_overlay: dict[str, Any]
    try:
        dynamics_mesh.validate()
    except MeshValidationError as error:
        raw_overlay = {"status": "failed", "error": str(error)}
    else:
        raw_overlay = {"status": "passed", "error": None}

    for name in _TOPOLOGY_WITNESSES:
        if name not in output_mesh.arrays or name not in dynamics_mesh.arrays:
            raise MeshValidationError(f"precision overlay lacks topology witness {name}")
        if not np.array_equal(output_mesh.arrays[name], dynamics_mesh.arrays[name]):
            raise MeshValidationError(
                f"precision overlay grid/static topology differs at {name}"
            )

    grid_radius = float(output_mesh.attrs.get("sphere_radius", np.nan))
    static_radius = float(dynamics_mesh.attrs.get("sphere_radius", np.nan))
    if (
        not np.isfinite(grid_radius)
        or not np.isfinite(static_radius)
        or grid_radius <= 0.0
        or static_radius <= 0.0
    ):
        raise MeshValidationError("precision overlay requires positive finite sphere radii")
    scale = static_radius / grid_radius
    for entity in ("Cell", "Edge", "Vertex"):
        for component in ("x", "y", "z"):
            name = f"{component}{entity}"
            source = np.asarray(output_mesh.arrays[name])
            if source.dtype != np.dtype(np.float64):
                raise MeshValidationError(f"official grid {name} is not binary64")
            dynamics_mesh.arrays[name] = np.ascontiguousarray(source * scale)
            dynamics_mesh.variable_sources[name] = (
                "grid_binary64_scaled_to_static_sphere_radius"
            )

        latitude_name = f"lat{entity}"
        longitude_name = f"lon{entity}"
        latitude = np.asarray(output_mesh.arrays[latitude_name])
        longitude = np.asarray(output_mesh.arrays[longitude_name])
        if latitude.dtype != np.dtype(np.float64) or longitude.dtype != np.dtype(np.float64):
            raise MeshValidationError(
                f"official grid {latitude_name}/{longitude_name} is not binary64"
            )
        output_mesh.arrays[latitude_name] = np.ascontiguousarray(latitude.copy())
        output_mesh.arrays[longitude_name] = np.ascontiguousarray(longitude.copy())
        dynamics_mesh.arrays[latitude_name] = np.ascontiguousarray(latitude.copy())
        dynamics_mesh.arrays[longitude_name] = np.ascontiguousarray(longitude.copy())
        dynamics_mesh.variable_sources[latitude_name] = "grid_binary64"
        dynamics_mesh.variable_sources[longitude_name] = output_mesh.variable_sources[
            longitude_name
        ]

    dynamics_mesh.validate()
    # Longitude reconciliation already strictly validated output_mesh.  Since
    # then its only replacements are byte-identical latitude/longitude copies;
    # all scaled geometry and static overlays belong to dynamics_mesh instead.
    evidence = {
        "schema": "mpas-port.precision-preserving-mesh-pair/v1",
        "grid_name": grid.name,
        "grid_sha256": longitude_reconciliation["input_grid_sha256"],
        "static_name": static.name,
        "static_sha256": _file_digest(static),
        "raw_static_overlay": raw_overlay,
        "topology_grid_static_bit_identical": True,
        "grid_sphere_radius": grid_radius,
        "static_sphere_radius": static_radius,
        "cartesian_scale": scale,
        "longitude_reconciliation": longitude_reconciliation,
        "longitude_normalization": {
            f"lon{entity}": longitude_reconciliation["entities"][entity][
                "canonicalization"
            ]
            for entity in ("Cell", "Edge", "Vertex")
        },
        "changed_array_inventory": list(_GEOMETRY_COORDINATES),
        "merged_validation_without_tolerance_change": "passed",
    }
    dynamics_mesh.provenance["precision_preserving_mesh_pair"] = evidence
    output_mesh.provenance["precision_preserving_mesh_pair"] = evidence
    return dynamics_mesh, output_mesh, evidence


__all__ = [
    "VERTEX_SLOT_CONVENTIONS",
    "closure_tolerance",
    "vertex_slot_convention",
    "GRID_ROTATE_CONVERT_XL_EPS",
    "GRID_ROTATE_CORRECTED_ATAN2_ATOL",
    "GRID_ROTATE_DEFAULT_REAL_PI",
    "GRID_ROTATE_LATITUDE_MAX_ULP_ERROR",
    "GRID_ROTATE_PI_DELTA",
    "GRID_ROTATE_PRODUCER_MAX_ULP_ERROR",
    "GRID_ROTATE_SOURCE_HEAD",
    "GRID_ROTATE_SOURCE_SHA256",
    "LONGITUDE_TRIG_EQUIVALENCE_ATOL",
    "Mesh",
    "MeshValidationError",
    "REGIONAL_BOUNDARY_MASK_NAMES",
    "REGIONAL_BOUNDARY_ZONE_WIDTH",
    "REGIONAL_SENTINEL_ARRAYS",
    "load_precision_preserving_mesh_pair",
    "normalize_longitudes",
    "reconcile_grid_rotate_longitudes",
    "regional_boundary_mask_digest",
    "spherical_arc_tolerance",
    "COORDINATE_REPRESENTATIONS",
    "COORDINATE_REPRESENTATION_ATTR",
    "COORDINATE_ORIGIN_ATTR",
    "validate_longitude_normalization",
]
