"""MPAS-A v8.4.2 height-based hybrid vertical coordinate authority.

Logical floating fields use Fortran order ``(vertical_level, horizontal_entity)``.
The scalar implementation follows ``src/core_init_atmosphere/mpas_init_atm_cases.F``
from the pinned MPAS-A v8.4.2 source.  It deliberately keeps the native loop
ordering for terrain and coordinate-surface smoothing so a compiled native
oracle can referee the result.

Two details are load-bearing:

* terrain smoothing skips a cell when the current field is exactly zero, as the
  native first and second passes do; coordinate-surface smoothing does *not*
  use that guard;
* the native ``do k=2,kz-1`` surface loop maps to Python
  ``range(1, first_height_level)``.  Stopping at ``first_height_level - 1``
  omits one native interface.

A LIMITED-AREA MESH IS REFUSED, EXCEPT AS A HALO.  The authority is a
closed-sphere one: a regional mesh's outermost ring has neighbours that were
cut away, and a coordinate surface smoothed there is smoothed against nothing,
so its values are not the global ones.  ``regional_halo=True`` admits such a
mesh only as a WORKING REGION that a caller cuts down afterwards to a region
lying far enough inside it that the cut-away neighbours cannot reach it
(:func:`woof.hex.mesh_point.generate_point`, the halo route).  The arithmetic
is untouched: a rim edge reads its one cell on both sides, so its own ``zxu``
is zero and it is discarded with the rim.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import subprocess
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigurationRefusal

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]

#: Edges per pass of the vectorized zb/zb3 loop.  Purely a working-set bound --
#: the per-chunk temporaries are ``(nVertLevels, chunk)`` float64, so 32,768
#: edges at 55 levels is about 14 MiB an array.  Every term is a pure function
#: of one edge, so the width changes no result bit; the equivalence suite
#: asserts that at widths 1, 7 and one whole mesh.
EDGE_METRIC_CHUNK_EDGES = 32_768


def materialize_rust_artifact(request: dict[str, Any]) -> dict[str, Any]:
    """Marshal the constructed contract to the engine's Rust container codec."""
    from .rust_geometry import geometry_engine

    engine, _ = geometry_engine()
    result = subprocess.run(
        [str(engine), "--protocol", "hex-init-artifact-v1"],
        input=json.dumps(request, allow_nan=False).encode("utf-8"),
        capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip()
                         or f"artifact bridge exited {result.returncode}")
    receipt = json.loads(result.stdout)
    if receipt.get("schema") != "hex-init-artifact-v1":
        raise ValueError("artifact bridge returned a different protocol schema")
    return receipt


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


@dataclass(frozen=True, slots=True)
class VerticalGrid:
    """Vertical coordinate and terrain metric fields consumed by the dycore."""

    zw: FloatArray
    dzw: FloatArray
    rdzw: FloatArray
    zu: FloatArray
    dzu: FloatArray
    rdzu: FloatArray
    rdzwp: FloatArray
    rdzwm: FloatArray
    fzp: FloatArray
    fzm: FloatArray
    ah: FloatArray
    hx: FloatArray
    zgrid: FloatArray
    zz: FloatArray
    zxu: FloatArray
    dss: FloatArray
    cf1: float
    cf2: float
    cf3: float
    first_height_level: int

    @property
    def n_vert_levels(self) -> int:
        return int(self.dzw.size)


@dataclass(frozen=True, slots=True)
class VerticalEdgeMetrics:
    """Edge-oriented omega-diagnosis metrics in logical Fortran order.

    Both arrays have shape ``(nVertLevels + 1, 2, nEdges)``.  MPAS computes
    rows 1..nVertLevels and leaves the top interface row at exact +0.
    """

    zb: FloatArray
    zb3: FloatArray
    theta_adv_order: int


def _refuse(knob: str, value: object, reason: str, remedy: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, remedy)


@dataclass(frozen=True, slots=True)
class _CellNeighborPlan:
    """One ordered gather per cell-edge slot for the native neighbor sum.

    The slot loop below is the same arithmetic the scalar transcription did,
    reordered slot-major instead of cell-major.  Every cell still accumulates
    its slots in native order 0, 1, ... nEdgesOnCell-1, so the floating-point
    addition sequence per cell is unchanged and the result is bitwise what the
    scalar loop produced.

    WHY IT EXISTS: on a 55-level parent
    :func:`build_vertical_grid` calls the neighbor sum 3,161 times -- the count
    is ``sum(nsm + level + 1 for level in 1..nVertLevels-1)``, so the surface
    smoother is quadratic in the level count -- and every one of those calls
    re-read the same five connectivity arrays and recomputed the same
    ``dvEdge/dcEdge`` ratios.  Measured on the 112,676-cell parent of
    2026-08-27, that loop was 86.2 % of the whole init stage.
    """

    owners: tuple[IntArray | None, ...]
    sources: tuple[IntArray, ...]
    weights: tuple[FloatArray, ...]


def _build_cell_neighbor_plan(mesh: object) -> _CellNeighborPlan:
    """Precompute the slot-major gather the neighbor sum repeats every call."""

    n_edges_on_cell = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    edges_on_cell = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    cells_on_cell = _mesh_array(mesh, "cellsOnCell").astype(np.int64, copy=False)
    dv_edge = _mesh_array(mesh, "dvEdge")
    dc_edge = _mesh_array(mesh, "dcEdge")
    n_cells = int(n_edges_on_cell.size)
    used_slots = int(n_edges_on_cell.max(initial=0))
    owners: list[IntArray | None] = []
    sources: list[IntArray] = []
    weights: list[FloatArray] = []
    for slot in range(used_slots):
        rows = np.flatnonzero(n_edges_on_cell > slot)
        # Every cell owns this slot: index with the whole array instead of a
        # copy of arange, which is the common case on a Goldberg mesh.
        full = rows.size == n_cells
        index = slice(None) if full else rows
        edge = edges_on_cell[index, slot]
        neighbor = cells_on_cell[index, slot]
        centre = np.arange(n_cells, dtype=np.int64) if full else rows
        owners.append(None if full else rows)
        sources.append(np.where(neighbor < 0, centre, neighbor))
        weights.append(dv_edge[edge] / dc_edge[edge])
    return _CellNeighborPlan(tuple(owners), tuple(sources), tuple(weights))


def _laplacian_without_area(
    mesh: object,
    field: FloatArray,
    *,
    skip_zero_center: bool,
    plan: _CellNeighborPlan | None = None,
) -> FloatArray:
    """Native weighted neighbor sum, including the regional garbage-cell rule."""

    if plan is None:
        plan = _build_cell_neighbor_plan(mesh)
    out = np.zeros_like(field)
    zero = field.dtype.type(0.0)
    for owners, sources, weights in zip(plan.owners, plan.sources, plan.weights):
        if owners is None:
            if skip_zero_center:
                active = field != zero
                np.add(
                    out,
                    weights * (field[sources] - field),
                    out=out,
                    where=active,
                )
            else:
                out += weights * (field[sources] - field)
            continue
        if skip_zero_center:
            keep = field[owners] != zero
            owners = owners[keep]
            sources = sources[keep]
            weights = weights[keep]
        centre = field[owners]
        out[owners] += weights * (field[sources] - centre)
    return out



#: The request fields that are arrays.  They stay numpy arrays in the request
#: dict; hex-vertical-v2 frames them as raw little-endian words and v1 turns
#: them into JSON lists.  Integer fields go as <i4/<i8 and the rest as <f4/<f8,
#: which the engine widens to the f64 a JSON number would have parsed to.
_VERTICAL_ARRAYS = ("terrain", "counts", "edges_on_cell", "cells_on_cell", "cells_on_edge",
                    "dc", "dv", "area", "deriv", "zgrid")
_VERTICAL_INTEGER_ARRAYS = frozenset({"counts", "edges_on_cell", "cells_on_cell", "cells_on_edge"})


def _rust_vertical_request(mesh: object, terrain: FloatArray, *, mode: str) -> dict:
    """Marshal arrays; all coordinate calculations run in the engine's rw-mpas crate."""
    if terrain.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("Rust vertical bridge requires float32 or float64")
    return dict(
        mode=mode, width=terrain.dtype.itemsize, terrain=np.asarray(terrain),
        counts=_mesh_array(mesh, "nEdgesOnCell"),
        edges_on_cell=_mesh_array(mesh, "edgesOnCell"),
        cells_on_cell=_mesh_array(mesh, "cellsOnCell"),
        cells_on_edge=_mesh_array(mesh, "cellsOnEdge"),
        dc=_mesh_array(mesh, "dcEdge"), dv=_mesh_array(mesh, "dvEdge"),
        area=_mesh_array(mesh, "areaCell"), levels=3, top=30000.0,
        scheme="tc", specified=None, projection="linear_interpolation",
        terrain_passes=0, smooth=False, surface_passes=0, minimum_fraction=0.3,
        hybrid=False, transition=30000.0, damping=0.0, damping_start=12000.0,
        regional_halo=True, order=2, deriv=np.zeros(0), zgrid=np.zeros(0),
    )


@lru_cache(maxsize=8)
def _check_vertical_engine(path: str, size: int, modified_ns: int) -> None:
    from pathlib import Path
    from .engines import EngineRefusal

    if b"rw_mpas_geometry --protocol hex-vertical-v1" not in Path(path).read_bytes():
        raise EngineRefusal(
            f"{path} lacks hex-vertical-v1; the older bridge cannot compute the "
            "v8.4.2 coordinate surface and omega metric arithmetic. Rebuild rw_mpas_geometry."
        )


@lru_cache(maxsize=8)
def _vertical_engine_speaks_v2(path: str, size: int, modified_ns: int) -> bool:
    from pathlib import Path
    return b"rw_mpas_geometry --protocol hex-vertical-v2" in Path(path).read_bytes()


def _as_lists(request: dict) -> dict:
    return {key: (value.tolist() if key in _VERTICAL_ARRAYS and isinstance(value, np.ndarray) else value)
            for key, value in request.items()}


def _run_rust_vertical(request: dict) -> bytes:
    """hex-vertical-v1: the whole request as JSON, the whole answer as bytes."""
    from .rust_geometry import geometry_engine

    engine, _ = geometry_engine()
    stat = engine.stat()
    _check_vertical_engine(str(engine), stat.st_size, stat.st_mtime_ns)
    result = subprocess.run([str(engine), "--protocol", "hex-vertical-v1"],
                            input=json.dumps(_as_lists(request), allow_nan=False).encode(),
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _v2_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if name in _VERTICAL_INTEGER_ARRAYS:
        wire = "<i4" if array.dtype == np.dtype("int32") else "<i8"
    else:
        wire = "<f4" if array.dtype == np.dtype("float32") else "<f8"
    selected = np.dtype(wire)
    # Keep conversions on the caller's thread to retain NumPy's error policy.
    if array.dtype != selected:
        return np.ascontiguousarray(array, dtype=selected)
    from .host_parallel import contiguous_copy

    return contiguous_copy(array, dtype=selected)


def _write_all(stream, view: memoryview) -> None:
    while len(view):
        written = stream.write(view)
        view = view[written:]


def _read_into(stream, view: memoryview) -> int:
    got = 0
    while got < len(view):
        count = stream.readinto(view[got:])
        if not count:
            break
        got += count
    return got


def _run_rust_vertical_v2(engine, request: dict, dtype: np.dtype, shapes: list[tuple],
                          trailer: int) -> tuple[list[FloatArray], bytes]:
    """hex-vertical-v2: arrays go and come back as raw words, written straight from
    and read straight into numpy memory.  At 30 km the v1 edge request was 1.65 GB
    of JSON text encoded on one thread here and parsed on one thread in the engine."""
    import tempfile

    arrays = {name: _v2_array(name, request[name]) for name in _VERTICAL_ARRAYS}
    header = {key: value for key, value in request.items() if key not in _VERTICAL_ARRAYS}
    header["arrays"] = [dict(name=name, dtype=array.dtype.str, shape=list(array.shape))
                        for name, array in arrays.items()]
    head = json.dumps(header, allow_nan=False).encode()
    outputs = [np.empty(shape, dtype=dtype.newbyteorder("<")) for shape in shapes]
    tail = bytearray(trailer)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([str(engine), "--protocol", "hex-vertical-v2"], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=errors, bufsize=0)
        complete = True
        try:
            try:
                _write_all(process.stdin, memoryview(b"HEXVRT2\0" + len(head).to_bytes(8, "little") + head))
                for array in arrays.values():
                    _write_all(process.stdin, memoryview(array).cast("B"))
            except BrokenPipeError:
                complete = False  # the engine refused early; its stderr says why
            finally:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            for array in outputs:
                view = memoryview(array).cast("B")
                complete = _read_into(process.stdout, view) == len(view) and complete
            complete = _read_into(process.stdout, memoryview(tail)) == trailer and complete
            extra = process.stdout.read(1)
        finally:
            process.stdout.close()
            code = process.wait()
        if code or not complete or extra:
            errors.seek(0)
            message = errors.read().decode("utf-8", errors="replace").strip()
            raise ValueError(message or f"vertical bridge exited {code} with an incomplete answer")
    return outputs, bytes(tail)


def _vertical_call(request: dict, dtype: np.dtype, shapes: list[tuple],
                   trailer: int = 0) -> tuple[list[FloatArray], bytes]:
    """Arrays of ``shapes`` plus ``trailer`` raw bytes; v2 framing when the engine has it."""
    from .rust_geometry import geometry_engine

    engine, _ = geometry_engine()
    stat = engine.stat()
    _check_vertical_engine(str(engine), stat.st_size, stat.st_mtime_ns)
    if _vertical_engine_speaks_v2(str(engine), stat.st_size, stat.st_mtime_ns):
        return _run_rust_vertical_v2(engine, request, dtype, shapes, trailer)
    raw = _run_rust_vertical(request)
    body = raw[:len(raw) - trailer] if trailer else raw
    return _decode_vertical(body, dtype, shapes), raw[len(raw) - trailer:] if trailer else b""


def _decode_vertical(raw: bytes, dtype: np.dtype, shapes: list[tuple]) -> list[FloatArray]:
    expected = sum(int(np.prod(shape)) for shape in shapes) * dtype.itemsize
    if len(raw) != expected:
        raise ValueError(f"vertical bridge returned {len(raw)} bytes; expected {expected}")
    values = np.frombuffer(raw, dtype=dtype.newbyteorder("<"))
    arrays, offset = [], 0
    for shape in shapes:
        count = int(np.prod(shape))
        arrays.append(values[offset:offset + count].reshape(shape).copy())
        offset += count
    return arrays


@lru_cache(maxsize=8)
def _check_compression_engine(path: str, size: int, modified_ns: int) -> None:
    from pathlib import Path
    from .engines import EngineRefusal
    if b"rw_mpas_geometry --protocol hex-advection-compression-v1" not in Path(path).read_bytes():
        raise EngineRefusal(f"{path} lacks native advection compression; rebuild rw_mpas_geometry")


def compress_native_advection(counts, neighbors, owners, deriv, dc, dv, *, garbage=-1):
    """Marshal native atmospheric-core compression through the Rust bridge.

    MPAS v8.4.2 core_atmosphere/mpas_atm_core.F:1391-1534. Inputs and
    returned cell IDs use zero-based mesh indexing; coefficients are RKIND.
    """
    from .rust_geometry import geometry_engine
    deriv = np.asarray(deriv)
    if deriv.dtype not in (np.dtype("float32"), np.dtype("float64")) or deriv.ndim != 3:
        raise TypeError("deriv must have floating shape (nEdges,2,nCoefficients)")
    nc = len(counts)
    request = dict(width=deriv.dtype.itemsize, counts=np.asarray(counts).tolist(),
        neighbors=np.where(np.asarray(neighbors) == garbage, -1, neighbors).tolist(),
        owners=np.where(np.asarray(owners) == garbage, -1, owners).tolist(),
        dc=np.asarray(dc).tolist(), dv=np.asarray(dv).tolist(), stencil=deriv.shape[2], deriv=deriv.tolist())
    engine, _ = geometry_engine()
    stat = engine.stat()
    _check_compression_engine(str(engine), stat.st_size, stat.st_mtime_ns)
    result = subprocess.run([str(engine), "--protocol", "hex-advection-compression-v1"],
        input=json.dumps(request, allow_nan=False).encode(), capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    ne, _, stride = deriv.shape
    split = 2 * ne * stride * deriv.dtype.itemsize
    coef, third = _decode_vertical(result.stdout[:split], deriv.dtype, [(ne, stride), (ne, stride)])
    integer = np.frombuffer(result.stdout[split:], dtype="<i8")
    if len(integer) != ne + ne * stride:
        raise ValueError("native compression returned an incomplete cell stencil")
    n_adv = integer[:ne].copy()
    cells = integer[ne:].reshape(ne, stride).copy() - 1
    cells[cells == nc] = garbage
    return coef, third, n_adv, cells


def edge_dc_squared_over_twelve(dc_edge: FloatArray, dtype: np.dtype) -> FloatArray:
    """Native integer square; compatibility helper, outside the init data path."""
    return (dc_edge * dc_edge / dtype.type(12.0)).astype(dtype, copy=False)

def smooth_terrain(
    mesh: object,
    terrain: FloatArray,
    passes: int = 1,
    *,
    plan: _CellNeighborPlan | None = None,
) -> FloatArray:
    """Apply the native two-pass fourth-order terrain smoother."""

    if passes < 0:
        _refuse(
            "nsmterrain",
            passes,
            "terrain smoothing passes cannot be negative",
            "nsmterrain=1",
        )
    request = _rust_vertical_request(mesh, np.asarray(terrain), mode="terrain")
    request["terrain_passes"] = passes
    raw = _run_rust_vertical(request)
    return _decode_vertical(raw, np.asarray(terrain).dtype, [(np.asarray(terrain).size,)])[0]


def _validate_parameters(
    *,
    n_vert_levels: int,
    ztop: float,
    terrain_smoothing_passes: int,
    surface_smoothing_passes: int,
    minimum_layer_fraction: float,
    hybrid_coordinate: bool,
    hybrid_transition_height: float,
    xnutr: float,
    damping_start: float,
) -> None:
    if n_vert_levels < 3:
        _refuse(
            "nVertLevels",
            n_vert_levels,
            "the lower-boundary extrapolation coefficients require at least three layers",
            "nVertLevels>=3",
        )
    if not np.isfinite(ztop) or ztop <= 0.0:
        _refuse("ztop", ztop, "model-top height must be finite and positive", "ztop=30000.0")
    if terrain_smoothing_passes < 0:
        _refuse("nsmterrain", terrain_smoothing_passes, "passes cannot be negative", "nsmterrain>=0")
    if surface_smoothing_passes < 0:
        _refuse("nsm", surface_smoothing_passes, "passes cannot be negative", "nsm>=0")
    if not np.isfinite(minimum_layer_fraction) or not 0.0 < minimum_layer_fraction < 1.0:
        _refuse(
            "dzmin",
            minimum_layer_fraction,
            "the accepted physical layer fraction must lie strictly between zero and one",
            "0 < dzmin < 1",
        )
    if hybrid_coordinate and (
        not np.isfinite(hybrid_transition_height) or hybrid_transition_height <= 0.0
    ):
        _refuse(
            "hybrid_top_z",
            hybrid_transition_height,
            "a hybrid coordinate needs a finite positive transition height",
            "hybrid_top_z=30000.0",
        )
    if not np.isfinite(xnutr):
        _refuse("xnutr", xnutr, "the damping amplitude must be finite", "xnutr=0.0")
    if not np.isfinite(damping_start):
        _refuse("zd", damping_start, "the damping start must be finite", "zd=22000.0")
    if xnutr != 0.0 and not damping_start < ztop:
        _refuse(
            "zd",
            damping_start,
            "nonzero upper-level damping with zd>=ztop divides by a zero or negative layer depth",
            "zd < ztop or xnutr=0",
        )


def build_vertical_grid(
    mesh: object,
    terrain: FloatArray,
    *,
    n_vert_levels: int = 55,
    ztop: float = 30_000.0,
    scheme: Literal["tc", "legacy", "specified"] = "tc",
    specified_zw: FloatArray | None = None,
    interface_projection: Literal["linear_interpolation", "layer_integral"] = "linear_interpolation",
    terrain_smoothing_passes: int = 1,
    smooth_surfaces: bool = True,
    surface_smoothing_passes: int = 30,
    minimum_layer_fraction: float = 0.3,
    hybrid_coordinate: bool = True,
    hybrid_transition_height: float = 30_000.0,
    xnutr: float = 0.0,
    damping_start: float = 22_000.0,
    regional_halo: bool = False,
    boundary_met_path: str | None = None,
    producer: Mapping[str, Any] | None = None,
) -> VerticalGrid:
    """Build all native vertical-coordinate fields for a closed/global mesh.

    ``regional_halo=True`` admits a limited-area mesh as a working halo whose
    rim the caller discards (see the module note); without it a limited-area
    mesh is refused by name.
    """

    if producer is not None:
        request = _producer_vertical_request(mesh, terrain, n_vert_levels,
            top=ztop, scheme=scheme, specified=specified_zw,
            projection=interface_projection, terrain_passes=terrain_smoothing_passes,
            smooth=smooth_surfaces, surface_passes=surface_smoothing_passes,
            minimum_fraction=minimum_layer_fraction, hybrid=hybrid_coordinate,
            transition=hybrid_transition_height, damping=xnutr,
            damping_start=damping_start, regional_halo=regional_halo, order=3)
        if boundary_met_path is not None:
            request["boundary"] = dict(
                met_path=str(boundary_met_path),
                lat=_mesh_array(mesh, "latCell").tolist(),
                lon=_mesh_array(mesh, "lonCell").tolist(),
                mask=_mesh_array(mesh, "bdyMaskCell").tolist(),
            )
        return _producer_vertical_grid(producer, request)

    _validate_parameters(
        n_vert_levels=n_vert_levels,
        ztop=ztop,
        terrain_smoothing_passes=terrain_smoothing_passes,
        surface_smoothing_passes=surface_smoothing_passes,
        minimum_layer_fraction=minimum_layer_fraction,
        hybrid_coordinate=hybrid_coordinate,
        hybrid_transition_height=hybrid_transition_height,
        xnutr=xnutr,
        damping_start=damping_start,
    )
    if interface_projection not in ("linear_interpolation", "layer_integral"):
        _refuse(
            "interface_projection",
            interface_projection,
            "only the two pinned MPAS projection branches exist",
            "interface_projection='linear_interpolation'",
        )

    ter = np.asarray(terrain)
    if ter.ndim != 1:
        raise ValueError(f"terrain must be one-dimensional over nCells, got {ter.shape}")
    n_cells = int(_mesh_array(mesh, "areaCell").size)
    if ter.shape != (n_cells,):
        raise ValueError(f"terrain shape {ter.shape} does not match nCells={n_cells}")
    if ter.dtype.kind != "f":
        ter = ter.astype(np.float64)
    if not np.all(np.isfinite(ter)):
        raise ValueError("terrain contains non-finite values")
    dtype = ter.dtype

    if not regional_halo and np.any(_mesh_array(mesh, "cellsOnEdge") < 0):
        _refuse(
            "regional_boundary_metrics", False,
            "this constructor does not invent exterior state; limited-area metrics require a caller-owned working halo",
            "regional_halo=True with a working halo whose rim is discarded",
        )
    request = _rust_vertical_request(mesh, ter, mode="grid")
    request.update(
        levels=n_vert_levels, top=ztop, scheme=scheme,
        specified=None if specified_zw is None else np.asarray(specified_zw).tolist(),
        projection=interface_projection, terrain_passes=terrain_smoothing_passes,
        smooth=smooth_surfaces, surface_passes=surface_smoothing_passes,
        minimum_fraction=minimum_layer_fraction, hybrid=hybrid_coordinate,
        transition=hybrid_transition_height, damping=xnutr, damping_start=damping_start,
        regional_halo=regional_halo,
    )
    if boundary_met_path is not None:
        request["boundary"] = dict(
            met_path=str(boundary_met_path),
            lat=_mesh_array(mesh, "latCell").tolist(),
            lon=_mesh_array(mesh, "lonCell").tolist(),
            mask=_mesh_array(mesh, "bdyMaskCell").tolist(),
        )
    dtype = ter.dtype
    n, nc, ne = n_vert_levels, n_cells, len(request["dc"])
    shapes = [(n + 1,), *[(n,)] * 9, (n + 1,),
              (n + 1, nc), (n + 1, nc), (n, nc), (n, ne), (n, nc), (3,)]
    arrays, kz = _vertical_call(request, dtype, shapes, trailer=8)
    # This is public metadata, whereas the native smoothing loop uses kz=nz
    # even when the hybrid transition has no interface in the model domain.
    first_height_level = int.from_bytes(kz, "little")
    if not hybrid_coordinate or not np.any(arrays[0] >= hybrid_transition_height):
        first_height_level = n + 1
    result = VerticalGrid(*arrays[:-1], *(float(x) for x in arrays[-1]), first_height_level)
    validate_vertical_grid(result, n_cells=nc, n_edges=ne)
    return result


def _logical_deriv_two(mesh: object, n_edges: int) -> FloatArray:
    raw = np.asarray(_mesh_array(mesh, "deriv_two"))
    if raw.dtype.kind != "f" or raw.ndim != 3:
        raise TypeError(f"deriv_two must be a three-dimensional floating array, got {raw.shape} {raw.dtype}")
    if raw.shape[2] == n_edges and raw.shape[1] == 2:
        logical = raw
    elif raw.shape[0] == n_edges and raw.shape[1] == 2:
        logical = np.transpose(raw, (2, 1, 0))
    elif raw.shape[1] == n_edges and raw.shape[0] == 2:
        logical = np.transpose(raw, (2, 0, 1))
    else:
        raise ValueError(
            "deriv_two orientation is unsupported: expected logical "
            f"(nCoeff,2,{n_edges}) or native ({n_edges},2,nCoeff), got {raw.shape}"
        )
    if logical.shape[0] < 2 or not np.all(np.isfinite(logical)):
        raise ValueError("deriv_two is too short or contains non-finite values")
    return np.asarray(logical)


def build_edge_vertical_metrics(
    mesh: object,
    vertical: VerticalGrid,
    *,
    theta_adv_order: int,
    producer: Mapping[str, Any] | None = None,
) -> VerticalEdgeMetrics:
    """Build ``zb`` and ``zb3`` exactly where native init does.

    This is the missing piece that prevented the existing vertical-grid port
    from replacing a native init capsule.  ``theta_adv_order`` 2 uses the edge
    average; orders 3/4 use ``deriv_two`` and only order 3 retains the odd
    correction in ``zb3``.
    """

    if producer is not None:
        request = _producer_vertical_request(mesh, _mesh_array(mesh, "ter"),
            vertical.n_vert_levels, mode="edge", order=theta_adv_order,
            zgrid=vertical.zgrid, regional_halo=True)
        raw = _run_vertical_producer(producer, request)
        ne = len(request["dc"])
        n = vertical.n_vert_levels
        expected = 2 * (n + 1) * 2 * ne * 4
        if len(raw) != expected:
            raise ValueError(f"vertical producer returned {len(raw)} edge bytes, expected {expected}")
        words = np.frombuffer(raw, dtype="<f4")
        shape = (n + 1, 2, ne)
        split = (n + 1) * 2 * ne
        return VerticalEdgeMetrics(words[:split].reshape(shape), words[split:].reshape(shape), theta_adv_order)

    if theta_adv_order not in (2, 3, 4):
        _refuse(
            "theta_adv_order",
            theta_adv_order,
            "the pinned init source implements only orders 2, 3, and 4",
            "theta_adv_order=3",
        )
    dtype = vertical.zgrid.dtype
    nc = vertical.zgrid.shape[1]
    request = _rust_vertical_request(mesh, np.zeros(nc, dtype=dtype), mode="edge")
    ne = len(request["dc"])
    request.update(levels=vertical.n_vert_levels, order=theta_adv_order,
                   zgrid=vertical.zgrid, regional_halo=True)
    if theta_adv_order != 2:
        deriv = _logical_deriv_two(mesh, ne)
        request["deriv"] = np.transpose(deriv, (2, 1, 0))
    shapes = [(vertical.n_vert_levels + 1, 2, ne)] * 2
    (zb, zb3), _ = _vertical_call(request, dtype, shapes)
    if not np.all(np.isfinite(zb)) or not np.all(np.isfinite(zb3)):
        raise ValueError("constructed zb/zb3 contain non-finite values")
    return VerticalEdgeMetrics(zb=zb, zb3=zb3, theta_adv_order=theta_adv_order)


def runtime_vertical_vectors(vertical: VerticalGrid) -> dict[str, FloatArray]:
    """Return native-file vectors with unused lower slots materialized as +0."""

    vectors: dict[str, FloatArray] = {}
    for name in ("dzu", "rdzu", "rdzwp", "rdzwm", "fzm", "fzp"):
        value = np.array(getattr(vertical, name), copy=True)
        value[0] = value.dtype.type(0.0)
        vectors[name] = value
    vectors["rdzw"] = np.array(vertical.rdzw, copy=True)
    return vectors


def validate_vertical_grid(
    vertical: VerticalGrid,
    *,
    n_cells: int | None = None,
    n_edges: int | None = None,
) -> dict[str, float | int]:
    """Independent structural and physical invariant gate."""

    nlev = vertical.n_vert_levels
    nc = int(vertical.zgrid.shape[1])
    ne = int(vertical.zxu.shape[1])
    if n_cells is not None and nc != n_cells:
        raise ValueError(f"vertical nCells={nc} != expected {n_cells}")
    if n_edges is not None and ne != n_edges:
        raise ValueError(f"vertical nEdges={ne} != expected {n_edges}")
    expected = {
        "zw": (nlev + 1,),
        "dzw": (nlev,),
        "rdzw": (nlev,),
        "zu": (nlev,),
        "dzu": (nlev,),
        "rdzu": (nlev,),
        "rdzwp": (nlev,),
        "rdzwm": (nlev,),
        "fzp": (nlev,),
        "fzm": (nlev,),
        "ah": (nlev + 1,),
        "hx": (nlev + 1, nc),
        "zgrid": (nlev + 1, nc),
        "zz": (nlev, nc),
        "zxu": (nlev, ne),
        "dss": (nlev, nc),
    }
    for name, shape in expected.items():
        value = np.asarray(getattr(vertical, name))
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
    for name in ("zw", "dzw", "rdzw", "zu", "ah", "hx", "zgrid", "zz", "zxu", "dss"):
        if not np.all(np.isfinite(np.asarray(getattr(vertical, name)))):
            raise ValueError(f"{name} contains non-finite values")
    for name in ("dzu", "rdzu", "rdzwp", "rdzwm", "fzp", "fzm"):
        value = np.asarray(getattr(vertical, name))
        if not np.all(np.isfinite(value[1:])):
            raise ValueError(f"{name} contains a non-finite active coefficient")
    dz = np.diff(vertical.zgrid, axis=0)
    if np.any(dz <= 0.0):
        raise ValueError("zgrid has a non-positive physical layer")
    if not np.all(np.diff(vertical.zw) > 0.0):
        raise ValueError("zw is not strictly increasing")
    if not np.all(vertical.zz > 0.0):
        raise ValueError("zz must be strictly positive")
    return {
        "n_vert_levels": nlev,
        "n_cells": nc,
        "n_edges": ne,
        "minimum_physical_layer_m": float(np.min(dz)),
        "maximum_physical_layer_m": float(np.max(dz)),
        "minimum_zz": float(np.min(vertical.zz)),
        "maximum_zz": float(np.max(vertical.zz)),
    }


def _producer_vertical_request(mesh: object, terrain: FloatArray, levels: int,
    *, mode: str = "grid", zgrid: FloatArray | None = None, **options: Any) -> dict[str, Any]:
    """Serialize the mesh ABI; all terrain/interpolation arithmetic stays in Rust.

    RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4448-4905
    at 1d6e370e. Producer selection comes from the resolved table row.
    """
    required = dict(top=0.0, scheme="tc", specified=None,
        projection="linear_interpolation", terrain_passes=0, smooth=False,
        surface_passes=0, minimum_fraction=0.0, hybrid=False, transition=0.0,
        damping=0.0, damping_start=0.0, regional_halo=False, order=3)
    required.update(options)
    deriv = _logical_deriv_two(mesh, int(_mesh_array(mesh, "dcEdge").size))
    request = dict(mode=mode, width=4, levels=levels,
        terrain=np.asarray(terrain).tolist(),
        counts=_mesh_array(mesh, "nEdgesOnCell").tolist(),
        edges_on_cell=_mesh_array(mesh, "edgesOnCell").tolist(),
        cells_on_cell=_mesh_array(mesh, "cellsOnCell").tolist(),
        cells_on_edge=_mesh_array(mesh, "cellsOnEdge").tolist(),
        dc=_mesh_array(mesh, "dcEdge").tolist(), dv=_mesh_array(mesh, "dvEdge").tolist(),
        area=_mesh_array(mesh, "areaCell").tolist(),
        deriv=np.transpose(deriv, (2, 1, 0)).tolist(),
        zgrid=[] if zgrid is None else np.asarray(zgrid).tolist(), **required)
    if request["specified"] is not None:
        request["specified"] = np.asarray(request["specified"]).tolist()
    return request


def _run_vertical_producer(producer: Mapping[str, Any], request: dict[str, Any]) -> bytes:
    """Run the table-selected internal worker with its declared JSON job ABI."""
    import json
    from pathlib import Path
    import subprocess
    executable = Path(producer["executable"])
    probe = subprocess.run([str(executable), "--abi"], capture_output=True, check=True, text=True)
    if probe.stdout.strip() != producer["abi"]:
        raise ValueError("vertical producer ABI differs from its resolved table row")
    if "vertical" not in producer["stages"]:
        raise ValueError("resolved producer does not implement the vertical stage")
    result = subprocess.run([str(executable)],
        input=json.dumps(dict(stage="vertical", config=request)).encode("utf-8"),
        capture_output=True)
    if result.returncode:
        raise ValueError("vertical producer failed: " + result.stderr.decode("utf-8", "replace"))
    return result.stdout


def _producer_vertical_grid(producer: Mapping[str, Any], request: dict[str, Any]) -> VerticalGrid:
    """Decode stored little-endian words into logical Fortran array views."""
    import struct
    raw = _run_vertical_producer(producer, request)
    n, nc, ne = request["levels"], len(request["terrain"]), len(request["dc"])
    shapes = [(n + 1,), *[(n,)] * 9, (n + 1,), (n + 1, nc),
              (n + 1, nc), (n, nc), (n, ne), (n, nc), (3,)]
    expected = sum(int(np.prod(shape)) for shape in shapes) * 4 + 8
    if len(raw) != expected:
        raise ValueError(f"vertical producer returned {len(raw)} grid bytes, expected {expected}")
    words = np.frombuffer(raw[:-8], dtype="<f4")
    arrays, offset = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        arrays.append(words[offset:offset + size].reshape(shape))
        offset += size
    cf = arrays.pop()
    first_height = struct.unpack("<Q", raw[-8:])[0]
    return VerticalGrid(*arrays, float(cf[0]), float(cf[1]), float(cf[2]), first_height)
