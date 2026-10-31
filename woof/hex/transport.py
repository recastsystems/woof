"""Scalar MPAS tracer transport CPU authority.

The transcription authority is the frozen MPAS-Model v8.2.3 source tree:

* ``src/operators/mpas_tracer_advection_helpers.F``
* ``src/operators/mpas_tracer_advection_std.F``
* ``src/operators/mpas_tracer_advection_mono.F``
* ``src/core_atmosphere/dynamics/mpas_atm_time_integration.F``
  (``atm_advance_scalars_work`` and ``atm_advance_scalars_mono_work``)

Arrays use Python MPAS order: entities are rows, connectivity is zero based,
and unused connectivity slots contain ``-1``.  Tracer fields have shape
``(nTracers, nVertLevels, nCells)``; a single tracer may omit the first axis.
All public calculations return a new array unless ``in_place=True`` is
explicitly requested.  Float32 and float64 tracer inputs remain in that
precision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import ConfigurationRefusal
from .prep_cache import cached


FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


@dataclass(frozen=True, slots=True)
class AdvectionCoefficients:
    """Compressed Skamarock--Gassmann horizontal-flux stencil.

    Coefficients are stored edge first, matching the dimensions seen through
    Python/netCDF rather than Fortran's reversed declaration order.  The
    third-order coefficients are *not* coupled to ``config_coef_3rd_order``;
    callers apply that coefficient at the flux evaluation site.
    """

    adv_coefs: FloatArray
    adv_coefs_3rd: FloatArray
    n_adv_cells_for_edge: NDArray[np.int64]
    adv_cells_for_edge: NDArray[np.int64]
    horizontal_order: int
    high_order_advection_mask: NDArray[np.int8] | None = None

    @property
    def nAdvCellsForEdge(self) -> NDArray[np.int64]:
        return self.n_adv_cells_for_edge

    @property
    def advCellsForEdge(self) -> NDArray[np.int64]:
        return self.adv_cells_for_edge

    @property
    def highOrderAdvectionMask(self) -> NDArray[np.int8] | None:
        return self.high_order_advection_mask


@dataclass(frozen=True, slots=True)
class ScalarTransportResult:
    """State returned by the atmosphere scalar-integration authority."""

    scalars: FloatArray
    density: FloatArray


@dataclass(frozen=True, slots=True)
class TracerAdvectionOptions:
    """Immutable replacement for the Fortran modules' saved init state."""

    horizontal_order: int
    vertical_order: int
    coefficient_3rd_order: float
    positive_dzdk: bool
    check_monotonicity: bool
    n_halos: int | None = None


__all__ = [
    "AdvectionCoefficients",
    "ScalarTransportResult",
    "TracerAdvectionOptions",
    "advance_scalar_transport",
    "advance_scalars",
    "advance_scalars_monotonic",
    "atm_advance_scalars_mono_work",
    "atm_advance_scalars_work",
    "atmosphere_vertical_flux_3",
    "build_advection_coefficients",
    "compute_advection_coefficients",
    "initialize_deriv_two",
    "monotonic_tracer_advection_tendency",
    "monotonic_tracer_tendency",
    "mpas_initialize_deriv_two",
    "mpas_tracer_advection_coefficients",
    "mpas_tracer_advection_mono_tend",
    "mpas_tracer_advection_mono_init",
    "mpas_tracer_advection_std_tend",
    "mpas_tracer_advection_std_init",
    "mpas_tracer_advection_vflux3",
    "mpas_tracer_advection_vflux4",
    "standard_tracer_advection_tendency",
    "standard_tracer_tendency",
    "vflux3",
    "vflux4",
    "vertical_flux_3",
    "vertical_flux_4",
]


def _mesh_value(mesh: Any, name: str, default: Any = None) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    if isinstance(mesh, Mapping) and name in mesh:
        return mesh[name]
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    dimensions = getattr(mesh, "dimensions", None)
    if isinstance(dimensions, Mapping) and name in dimensions:
        return dimensions[name]
    attrs = getattr(mesh, "attrs", None)
    if isinstance(attrs, Mapping) and name in attrs:
        return attrs[name]
    return default


def _mesh_array(mesh: Any, name: str) -> NDArray[Any]:
    value = _mesh_value(mesh, name, None)
    if value is None:
        raise AttributeError(f"mesh does not expose required array {name!r}")
    return np.asarray(value)


def _float_array(value: ArrayLike, name: str, dtype: np.dtype[Any] | None = None) -> FloatArray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must be numeric")
        array = array.astype(np.float64)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _source_inverse_array(
    value: ArrayLike,
    name: str,
    dtype: np.dtype[Any],
) -> FloatArray:
    """Load an already-rounded MPAS inverse without changing its RKIND."""

    array = np.asarray(value)
    if array.dtype != dtype:
        raise TypeError(f"{name} dtype {array.dtype} must match state dtype {dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _rows(array: ArrayLike, rows: int, name: str) -> NDArray[Any]:
    value = np.asarray(array)
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if value.shape[0] == rows:
        return value
    if value.shape[1] == rows:
        return value.T
    raise ValueError(f"{name} has shape {value.shape}; expected {rows} rows")


def _truth(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("ascii", errors="ignore")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "true", "t", "1"}:
            return True
        if normalized in {"no", "false", "f", "0"}:
            return False
    return bool(value)


def _validate_order(value: int, knob: str) -> int:
    order = int(value)
    if order not in (2, 3, 4):
        raise ConfigurationRefusal(
            knob,
            value,
            "the frozen MPAS tracer stencils admit only orders 2, 3, and 4",
            f"{knob}=3",
        )
    return order


def _validate_third_order_coefficient(value: float) -> None:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ConfigurationRefusal(
            "config_coef_3rd_order",
            value,
            "the frozen Registry.xml range is 0 through 1",
            "config_coef_3rd_order=0.25",
        )


def mpas_tracer_advection_std_init(
    horiz_adv_order: int,
    vert_adv_order: int,
    coef_3rd_order_in: float,
    dzdk_positive: bool,
    check_monotonicity: bool,
) -> TracerAdvectionOptions:
    """Validate and return the saved state from std.F:273-309.

    Returning state rather than changing module globals makes concurrent CPU
    authority calls deterministic.  Invalid orders refuse instead of taking
    the frozen routine's warning-and-substitute branch.
    """

    horizontal = _validate_order(horiz_adv_order, "config_scalar_adv_order")
    vertical = _validate_order(vert_adv_order, "config_scalar_vadv_order")
    _validate_third_order_coefficient(coef_3rd_order_in)
    effective = float(coef_3rd_order_in) if horizontal == 3 else 0.0
    return TracerAdvectionOptions(
        horizontal, vertical, effective, bool(dzdk_positive), bool(check_monotonicity)
    )


def mpas_tracer_advection_mono_init(
    nHalos: int,
    horiz_adv_order: int,
    vert_adv_order: int,
    coef_3rd_order_in: float,
    dzdk_positive: bool,
    check_monotonicity: bool,
) -> TracerAdvectionOptions:
    """Validate and return the saved state from mono.F:465-508."""

    if nHalos < 3:
        raise ConfigurationRefusal(
            "nHalos",
            nHalos,
            "the frozen monotonic FCT requires at least three halo rows",
            "nHalos=3",
        )
    standard = mpas_tracer_advection_std_init(
        horiz_adv_order,
        vert_adv_order,
        coef_3rd_order_in,
        dzdk_positive,
        check_monotonicity,
    )
    return TracerAdvectionOptions(
        standard.horizontal_order,
        standard.vertical_order,
        standard.coefficient_3rd_order,
        standard.positive_dzdk,
        standard.check_monotonicity,
        int(nHalos),
    )


def _fortran_sign_one(value: np.floating[Any], dtype: np.dtype[Any]) -> np.floating[Any]:
    return np.copysign(dtype.type(1.0), value)


def _as_tracers(value: ArrayLike, name: str) -> tuple[FloatArray, bool]:
    array = _float_array(value, name)
    if array.ndim == 2:
        return array[np.newaxis, ...], True
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (nLevels,nCells) or (nTracers,nLevels,nCells)")
    return array, False


def _restore_tracers(value: FloatArray, squeezed: bool) -> FloatArray:
    return value[0] if squeezed else value


def _topology(
    mesh: Any, *, allow_regional_sentinels: bool = False
) -> tuple[int, int, NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    n_cells = counts.size
    edges_on_cell = _rows(_mesh_array(mesh, "edgesOnCell"), n_cells, "edgesOnCell").astype(
        np.int64, copy=False
    )
    cells_on_cell = _rows(_mesh_array(mesh, "cellsOnCell"), n_cells, "cellsOnCell").astype(
        np.int64, copy=False
    )
    raw_cells_on_edge = np.asarray(_mesh_array(mesh, "cellsOnEdge"))
    if raw_cells_on_edge.ndim != 2:
        raise ValueError("cellsOnEdge must be two-dimensional")
    if raw_cells_on_edge.shape[1] == 2:
        cells_on_edge = raw_cells_on_edge.astype(np.int64, copy=False)
    elif raw_cells_on_edge.shape[0] == 2:
        cells_on_edge = raw_cells_on_edge.T.astype(np.int64, copy=False)
    else:
        raise ValueError("cellsOnEdge must have exactly two cells per edge")
    n_edges = cells_on_edge.shape[0]
    if counts.shape != (n_cells,) or np.any(counts < 0) or np.any(counts > edges_on_cell.shape[1]):
        raise ValueError("nEdgesOnCell is inconsistent with edgesOnCell")
    if cells_on_cell.shape[1] < edges_on_cell.shape[1]:
        raise ValueError("cellsOnCell has fewer slots than edgesOnCell")
    if allow_regional_sentinels:
        regional_garbage_index = n_cells
        # Regional branch: the culled mesh stores 0 (loaded as a negative
        # sentinel) in valid connectivity slots of ring-7 rows only.  Native
        # MPAS maps those entries to the garbage element nCells+1 at read
        # time; the caller-provided garbage index reproduces that mapping,
        # and the mask guards below this layer keep any gather through a
        # garbage index on an element whose value is never consumed.
        cells_on_cell = np.where(
            cells_on_cell < 0, regional_garbage_index, cells_on_cell
        )
        cells_on_edge = np.where(
            cells_on_edge < 0, regional_garbage_index, cells_on_edge
        )
    # Check active slots in row batches. The old loop dispatched three NumPy
    # reductions per cell under the GIL (1.1 s at 60 km in a cold preflight).
    # Return the first bad row from each slice, then preserve the old row and
    # edge-before-neighbor refusal order, including when both errors occur.
    width = edges_on_cell.shape[1]
    def check_rows(lo, hi):
        active = np.arange(width)[None, :] < counts[lo:hi, None]
        edges = edges_on_cell[lo:hi]
        bad_edge = np.any(active & ((edges < 0) | (edges >= n_edges)), axis=1)
        bad_neighbor = np.any(active & (cells_on_cell[lo:hi, :width] < 0), axis=1)
        bad = np.flatnonzero(bad_edge | bad_neighbor)
        return None if bad.size == 0 else bool(bad_edge[bad[0]])

    for bad_edge in _map_prep_rows(n_cells, check_rows):
        if bad_edge is None:
            continue
        if bad_edge:
            raise ValueError("active edgesOnCell entry is out of range")
        else:
            raise ConfigurationRefusal(
                "config_apply_lbcs",
                False,
                "active regional neighbor sentinels need lateral-boundary driving state",
                "config_apply_lbcs=True with bdy_mask_cell/bdy_mask_edge and an "
                "admitted lbc source (the regional_v841 runtime supplies all three)",
            )
    if np.any(cells_on_edge < 0):
        raise ConfigurationRefusal(
            "config_apply_lbcs",
            False,
            "an active cellsOnEdge sentinel needs the regional lateral-boundary branch",
            "config_apply_lbcs=True with bdy_mask_cell/bdy_mask_edge and an "
            "admitted lbc source (the regional_v841 runtime supplies all three)",
        )
    limit = n_cells + 1 if allow_regional_sentinels else n_cells
    if np.any(cells_on_edge >= limit):
        raise ValueError("cellsOnEdge contains an out-of-range cell")
    return n_cells, n_edges, counts, edges_on_cell, cells_on_cell, cells_on_edge


def _operator_edge_signs(
    counts: NDArray[np.int64],
    edges_on_cell: NDArray[np.int64],
    cells_on_edge: NDArray[np.int64],
) -> NDArray[np.int8]:
    """Signs at mono/std lines 142-151: first cell -1, second +1."""

    # Shape follows edgesOnCell, including canonical zero-valued padding.
    result = np.zeros((counts.size, edges_on_cell.shape[1]), dtype=np.int8)
    for cell in range(counts.size):
        for slot in range(int(counts[cell])):
            edge = int(edges_on_cell[cell, slot])
            result[cell, slot] = -1 if cells_on_edge[edge, 0] == cell else 1
    return result


def _normalize_edge_signs(
    value: ArrayLike | None,
    counts: NDArray[np.int64],
    edges_on_cell: NDArray[np.int64],
    cells_on_edge: NDArray[np.int64],
    dtype: np.dtype[Any],
) -> FloatArray:
    if value is None:
        return _operator_edge_signs(counts, edges_on_cell, cells_on_edge).astype(dtype)
    return _rows(value, counts.size, "edge_sign_on_cell").astype(dtype, copy=False)


def vertical_flux_4(
    q_im2: ArrayLike,
    q_im1: ArrayLike,
    q_i: ArrayLike,
    q_ip1: ArrayLike,
    w: ArrayLike,
) -> FloatArray:
    """Fourth-order vertical flux from helpers.F:44-51."""

    arrays = [_float_array(v, name) for v, name in zip(
        (q_im2, q_im1, q_i, q_ip1, w), ("q_im2", "q_im1", "q_i", "q_ip1", "w")
    )]
    dtype = np.result_type(*(array.dtype for array in arrays))
    q0, q1, q2, q3, velocity = (array.astype(dtype, copy=False) for array in arrays)
    return velocity * (dtype.type(7.0) * (q2 + q1) - (q3 + q0)) / dtype.type(12.0)


def vertical_flux_3(
    q_im2: ArrayLike,
    q_im1: ArrayLike,
    q_i: ArrayLike,
    q_ip1: ArrayLike,
    w: ArrayLike,
    coefficient: float,
) -> FloatArray:
    """Third-order helper flux, including its source **minus** correction.

    Frozen authority: ``mpas_tracer_advection_helpers.F:64-73``.
    """

    base = vertical_flux_4(q_im2, q_im1, q_i, q_ip1, w)
    dtype = base.dtype
    q0 = _float_array(q_im2, "q_im2", dtype)
    q1 = _float_array(q_im1, "q_im1", dtype)
    q2 = _float_array(q_i, "q_i", dtype)
    q3 = _float_array(q_ip1, "q_ip1", dtype)
    velocity = _float_array(w, "w", dtype)
    correction = dtype.type(coefficient) * np.abs(velocity) * (
        (q3 - q0) - dtype.type(3.0) * (q2 - q1)
    ) / dtype.type(12.0)
    return base - correction


def atmosphere_vertical_flux_3(
    q_im2: ArrayLike,
    q_im1: ArrayLike,
    q_i: ArrayLike,
    q_ip1: ArrayLike,
    w: ArrayLike,
    coefficient: float,
) -> FloatArray:
    """Atmosphere time-integration flux3, whose correction has a plus sign.

    Frozen authority: ``mpas_atm_time_integration.F:3108-3113`` and
    ``:3566-3571``.  This intentionally differs from :func:`vertical_flux_3`.
    """

    base = vertical_flux_4(q_im2, q_im1, q_i, q_ip1, w)
    dtype = base.dtype
    q0 = _float_array(q_im2, "q_im2", dtype)
    q1 = _float_array(q_im1, "q_im1", dtype)
    q2 = _float_array(q_i, "q_i", dtype)
    q3 = _float_array(q_ip1, "q_ip1", dtype)
    velocity = _float_array(w, "w", dtype)
    correction = dtype.type(coefficient) * np.abs(velocity) * (
        (q3 - q0) - dtype.type(3.0) * (q2 - q1)
    ) / dtype.type(12.0)
    return base + correction


mpas_tracer_advection_vflux4 = vertical_flux_4
mpas_tracer_advection_vflux3 = vertical_flux_3


def _normalize_deriv_two(value: ArrayLike, n_edges: int) -> FloatArray:
    deriv = _float_array(value, "deriv_two")
    if deriv.ndim != 3:
        raise ValueError("deriv_two must be three-dimensional")
    if deriv.shape[0] == n_edges and deriv.shape[1] == 2:
        return deriv
    if deriv.shape[2] == n_edges and deriv.shape[1] == 2:
        return deriv.transpose(2, 1, 0)
    raise ValueError(
        f"deriv_two has shape {deriv.shape}; expected (nEdges,2,stencil)"
    )


@cached("advection-coefficients", salt=lambda: __import__("woof.hex.rust_hostprep",fromlist=["_engine_digest"])._engine_digest())
def build_advection_coefficients(
    mesh: Any,
    deriv_two: ArrayLike | None = None,
    *,
    config_scalar_adv_order: int = 3,
    n_vert_levels: int | None = None,
    boundary_cell: ArrayLike | None = None,
    source_order_v841: bool = False,
    allow_regional_sentinels: bool = False,
) -> AdvectionCoefficients:
    """Compress cell second-derivative stencils onto each edge.

    This is helpers.F:87-302 with the published Python/netCDF array order.
    The legacy stencil is sorted by ``indexToCellID``. The v8.4.1 path
    preserves native ``atm_adv_coef_compression`` insertion order. On meshes
    without global IDs, zero-based cell indices are
    the deterministic IDs.

    On a regional cull (``allow_regional_sentinels=True``) the ring-7
    stored-0 connectivity maps to the garbage index exactly as native
    ``atm_adv_coef_compression`` sees it: the garbage cell contributes no
    neighbours (``nEdgesOnCell`` of the garbage element is 0) and may itself
    appear in the stencil of a ring-6/7 edge, whose flux no updated cell
    ever reads (the regional edge guards skip those edges entirely).
    """

    order = _validate_order(config_scalar_adv_order, "config_scalar_adv_order")
    n_cells, n_edges, counts, _, cells_on_cell, cells_on_edge = _topology(
        mesh, allow_regional_sentinels=allow_regional_sentinels
    )
    raw_deriv = _mesh_array(mesh, "deriv_two") if deriv_two is None else deriv_two
    deriv = _normalize_deriv_two(raw_deriv, n_edges)
    dtype = deriv.dtype
    dc_edge = _float_array(_mesh_array(mesh, "dcEdge"), "dcEdge", dtype)
    dv_edge = _float_array(_mesh_array(mesh, "dvEdge"), "dvEdge", dtype)
    if dc_edge.shape != (n_edges,) or dv_edge.shape != (n_edges,):
        raise ValueError("dcEdge and dvEdge must have one value per edge")
    ids_value = _mesh_value(mesh, "indexToCellID", None)
    ids = np.arange(n_cells, dtype=np.int64) if ids_value is None else np.asarray(ids_value, dtype=np.int64)
    if ids.shape != (n_cells,) or np.unique(ids).size != n_cells:
        raise ValueError("indexToCellID must contain one unique ID per cell")

    width = deriv.shape[2]
    from .rust_hostprep import compress_advection
    adv,adv3,n_adv,adv_cells = compress_advection(counts=counts,edges=_rows(_mesh_array(mesh,"edgesOnCell"),n_cells,"edgesOnCell"),
        cells_on_cell=cells_on_cell[:,:_rows(_mesh_array(mesh,"edgesOnCell"),n_cells,"edgesOnCell").shape[1]],
        cells_on_edge=cells_on_edge,ids=ids,deriv=deriv,dc=dc_edge,dv=dv_edge,
        source_order=source_order_v841,garbage=n_cells if allow_regional_sentinels else -1)
    if source_order_v841:
        # atm_adv_coef_compression preserves insertion order: the two owners,
        # cell1's neighbors, then cell2's previously unseen neighbors.
        # The bridge sorts by global ID. Reorder its bit-identical coefficients
        # before any flux sum, without changing coefficient construction.
        adv, adv3, adv_cells = _native_advection_order(
            counts, cells_on_cell, cells_on_edge, adv, adv3, n_adv, adv_cells,
            garbage=n_cells if allow_regional_sentinels else -1,
        )
        adv, adv3 = _native_advection_compression(
            counts, cells_on_cell, cells_on_edge, deriv, dc_edge, dv_edge,
            n_adv, adv_cells,
        )

    mask: NDArray[np.int8] | None = None
    if boundary_cell is not None:
        boundary = np.asarray(boundary_cell)
        if boundary.ndim != 2:
            raise ValueError("boundary_cell must have shape (nLevels,nCells)")
        if boundary.shape[1] != n_cells and boundary.shape[0] == n_cells:
            boundary = boundary.T
        if boundary.shape[1] != n_cells:
            raise ValueError("boundary_cell must have one column per cell")
        n_vert_levels = boundary.shape[0] if n_vert_levels is None else n_vert_levels
        if boundary.shape[0] != n_vert_levels:
            raise ValueError("boundary_cell level count disagrees with n_vert_levels")
        mask = np.zeros((n_vert_levels, n_edges), dtype=np.int8)
        if order != 2:
            for edge in range(n_edges):
                cell1, cell2 = cells_on_edge[edge]
                for level in range(n_vert_levels):
                    mask[level, edge] = 0 if (boundary[level, cell1] == 1 or boundary[level, cell2] == 1) else 1
    elif n_vert_levels is not None:
        if n_vert_levels < 1:
            raise ValueError("n_vert_levels must be positive")
        mask = np.zeros((n_vert_levels, n_edges), dtype=np.int8)
        if order != 2:
            mask.fill(1)

    return AdvectionCoefficients(adv, adv3, n_adv, adv_cells, order, mask)


mpas_tracer_advection_coefficients = build_advection_coefficients


def _sphere_arc_length(a: FloatArray, b: FloatArray, dtype: np.dtype[Any]) -> np.floating[Any]:
    delta = b - a
    radius = np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    chord = np.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
    return radius * dtype.type(2.0) * np.arcsin(chord / (dtype.type(2.0) * radius))


def _sphere_angle(a: FloatArray, b: FloatArray, c: FloatArray, dtype: np.dtype[Any]) -> np.floating[Any]:
    side_a = _sphere_arc_length(b, c, dtype)
    side_b = _sphere_arc_length(a, c, dtype)
    side_c = _sphere_arc_length(a, b, dtype)
    cross = np.cross(b - a, c - a)
    semiperimeter = dtype.type(0.5) * (side_a + side_b + side_c)
    ratio = (
        np.sin(semiperimeter - side_b) * np.sin(semiperimeter - side_c)
    ) / (np.sin(side_b) * np.sin(side_c))
    sine = np.sqrt(np.minimum(dtype.type(1.0), np.maximum(dtype.type(0.0), ratio)))
    angle = dtype.type(2.0) * np.arcsin(
        np.maximum(dtype.type(-1.0), np.minimum(dtype.type(1.0), sine))
    )
    return angle if np.dot(cross, a) >= 0 else -angle


def _arc_bisect(a: FloatArray, b: FloatArray, dtype: np.dtype[Any]) -> FloatArray:
    radius = np.sqrt(np.dot(a, a))
    midpoint = dtype.type(0.5) * (a + b)
    distance = np.sqrt(np.dot(midpoint, midpoint))
    if distance == 0:
        raise ValueError("verticesOnEdge contains diametrically opposite vertices")
    return radius * midpoint / distance


def initialize_deriv_two(
    mesh: Any,
    *,
    stencil_width: int | None = None,
    allow_regional_sentinels: bool = False,
) -> FloatArray:
    """Initialize the quadratic-fit directional second derivatives.

    This ports helpers.F:315-722 and the called weighted least-squares formula
    in ``mpas_geometry_utils.F:239-296``.  Result shape is
    ``(nEdges, 2, stencil_width)`` and defaults to the frozen atmosphere's
    ``FIFTEEN`` width (or a larger required width).
    """

    n_cells, n_edges, counts, edges_on_cell, cells_on_cell, cells_on_edge = _topology(
        mesh, allow_regional_sentinels=allow_regional_sentinels
    )
    x_cell = _float_array(_mesh_array(mesh, "xCell"), "xCell")
    dtype = x_cell.dtype
    y_cell = _float_array(_mesh_array(mesh, "yCell"), "yCell", dtype)
    z_cell = _float_array(_mesh_array(mesh, "zCell"), "zCell", dtype)
    x_vertex = _float_array(_mesh_array(mesh, "xVertex"), "xVertex", dtype)
    y_vertex = _float_array(_mesh_array(mesh, "yVertex"), "yVertex", dtype)
    z_vertex = _float_array(_mesh_array(mesh, "zVertex"), "zVertex", dtype)
    vertices_on_edge = _rows(_mesh_array(mesh, "verticesOnEdge"), n_edges, "verticesOnEdge").astype(np.int64, copy=False)
    angle_edge = _float_array(_mesh_array(mesh, "angleEdge"), "angleEdge", dtype)
    dc_edge = _float_array(_mesh_array(mesh, "dcEdge"), "dcEdge", dtype)
    if x_cell.shape != (n_cells,) or angle_edge.shape != (n_edges,):
        raise ValueError("cell/edge geometry dimensions disagree with connectivity")
    required_width = int(counts.max(initial=0)) + 1
    width = max(15, required_width) if stencil_width is None else int(stencil_width)
    if width < required_width:
        raise ValueError(f"stencil_width={width} is smaller than required {required_width}")
    result = np.zeros((n_edges, 2, width), dtype=dtype)
    on_sphere = _truth(_mesh_value(mesh, "on_a_sphere", True), True)
    radius = dtype.type(_mesh_value(mesh, "sphere_radius", 1.0))
    if on_sphere and radius <= 0:
        raise ValueError("sphere_radius must be positive")
    pi = dtype.type(2.0) * np.arcsin(dtype.type(1.0))
    cell_xyz = np.stack((x_cell, y_cell, z_cell), axis=-1)
    vertex_xyz = np.stack((x_vertex, y_vertex, z_vertex), axis=-1)

    for cell in range(n_cells):
        count = int(counts[cell])
        cell_list = np.empty(count + 1, dtype=np.int64)
        cell_list[0] = cell
        cell_list[1:] = cells_on_cell[cell, :count]
        points = cell_xyz[cell_list] / radius if on_sphere else cell_xyz[cell_list]
        xp = np.empty(count, dtype=dtype)
        yp = np.empty(count, dtype=dtype)
        theta_t = np.empty(count, dtype=dtype)
        theta_edge = np.empty(count, dtype=dtype)

        if on_sphere:
            if points[0, 2] == dtype.type(1.0):
                theta_abs = pi / dtype.type(2.0)
            else:
                north = np.array((0.0, 0.0, 1.0), dtype=dtype)
                theta_abs = pi / dtype.type(2.0) - _sphere_angle(points[0], points[1], north, dtype)
            theta_vertex = np.empty(count, dtype=dtype)
            distance = np.empty(count, dtype=dtype)
            for slot in range(count):
                next_slot = (slot + 1) % count
                theta_vertex[slot] = _sphere_angle(points[0], points[slot + 1], points[next_slot + 1], dtype)
                distance[slot] = radius * _sphere_arc_length(points[0], points[slot + 1], dtype)
            theta_t[0] = theta_abs
            for slot in range(1, count):
                theta_t[slot] = theta_t[slot - 1] + theta_vertex[slot - 1]
            for slot in range(count):
                xp[slot] = np.cos(theta_t[slot]) * distance[slot]
                yp[slot] = np.sin(theta_t[slot]) * distance[slot]
        else:
            for slot in range(count):
                edge = int(edges_on_cell[cell, slot])
                angle = angle_edge[edge]
                if cell != cells_on_edge[edge, 0]:
                    angle = angle - pi
                theta_t[slot] = angle
                xp[slot] = dc_edge[edge] * np.cos(angle)
                yp[slot] = dc_edge[edge] * np.sin(angle)

        rows = count + 1
        design = np.zeros((rows, 6), dtype=dtype)
        design[0, 0] = dtype.type(1.0)
        for row in range(1, rows):
            x = xp[row - 1]
            y = yp[row - 1]
            design[row, 0] = dtype.type(1.0)
            design[row, 1] = x
            design[row, 2] = y
            design[row, 3] = x * x
            design[row, 4] = x * y
            design[row, 5] = y * y
        normal_matrix = design.T @ design
        try:
            inverse_fit = np.linalg.solve(normal_matrix, design.T).astype(dtype, copy=False)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"quadratic derivative fit is singular at cell {cell}") from error

        for slot in range(count):
            edge = int(edges_on_cell[cell, slot])
            if on_sphere:
                vertices = vertices_on_edge[edge]
                first = vertex_xyz[vertices[0]] / radius
                second = vertex_xyz[vertices[1]] / radius
                midpoint = _arc_bisect(first, second, dtype)
                theta_edge[slot] = _sphere_angle(
                    points[0], points[slot + 1], midpoint, dtype
                ) + theta_t[slot]
                angle = theta_edge[slot]
            else:
                angle = theta_t[slot]
            cosine = np.cos(angle)
            sine = np.sin(angle)
            cosine_sine = cosine * sine
            cosine_squared = cosine * cosine
            sine_squared = sine * sine
            side = 0 if cells_on_edge[edge, 0] == cell else 1
            for stencil in range(rows):
                result[edge, side, stencil] = (
                    dtype.type(2.0) * cosine_squared * inverse_fit[3, stencil]
                    + dtype.type(2.0) * cosine_sine * inverse_fit[4, stencil]
                    + dtype.type(2.0) * sine_squared * inverse_fit[5, stencil]
                )
    return result


mpas_initialize_deriv_two = initialize_deriv_two


def _resolve_coefficients(
    coefficients: AdvectionCoefficients | None,
    adv_coefs: ArrayLike | None,
    adv_coefs_3rd: ArrayLike | None,
    n_adv_cells_for_edge: ArrayLike | None,
    adv_cells_for_edge: ArrayLike | None,
    n_edges: int,
    dtype: np.dtype[Any],
    horizontal_order: int,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64], NDArray[np.int64]]:
    if coefficients is not None:
        if any(value is not None for value in (adv_coefs, adv_coefs_3rd, n_adv_cells_for_edge, adv_cells_for_edge)):
            raise ValueError("pass coefficients or separate coefficient arrays, not both")
        adv_coefs = coefficients.adv_coefs
        adv_coefs_3rd = coefficients.adv_coefs_3rd
        n_adv_cells_for_edge = coefficients.n_adv_cells_for_edge
        adv_cells_for_edge = coefficients.adv_cells_for_edge
        if coefficients.horizontal_order != horizontal_order:
            raise ValueError(
                "coefficients.horizontal_order disagrees with config_scalar_adv_order"
            )
    missing = [
        name
        for name, value in (
            ("adv_coefs", adv_coefs),
            ("adv_coefs_3rd", adv_coefs_3rd),
            ("n_adv_cells_for_edge", n_adv_cells_for_edge),
            ("adv_cells_for_edge", adv_cells_for_edge),
        )
        if value is None
    ]
    if missing:
        raise ValueError("missing horizontal coefficient inputs: " + ", ".join(missing))
    adv = _rows(adv_coefs, n_edges, "adv_coefs").astype(dtype, copy=False)
    adv3 = _rows(adv_coefs_3rd, n_edges, "adv_coefs_3rd").astype(dtype, copy=False)
    n_adv = np.asarray(n_adv_cells_for_edge, dtype=np.int64)
    cells = _rows(adv_cells_for_edge, n_edges, "adv_cells_for_edge").astype(np.int64, copy=False)
    if adv.shape != adv3.shape or cells.shape != adv.shape or n_adv.shape != (n_edges,):
        raise ValueError("horizontal advection coefficient shapes disagree")
    if np.any(n_adv < 0) or np.any(n_adv > adv.shape[1]):
        raise ValueError("n_adv_cells_for_edge is outside coefficient width")
    return adv, adv3, n_adv, cells


def _level_limits(
    n_levels: int,
    n_cells: int,
    n_edges: int,
    max_level_cell: ArrayLike | None,
    max_level_edge_top: ArrayLike | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    cell_levels = (
        np.full(n_cells, n_levels, dtype=np.int64)
        if max_level_cell is None
        else np.asarray(max_level_cell, dtype=np.int64)
    )
    edge_levels = (
        np.full(n_edges, n_levels, dtype=np.int64)
        if max_level_edge_top is None
        else np.asarray(max_level_edge_top, dtype=np.int64)
    )
    if cell_levels.shape != (n_cells,) or np.any(cell_levels < 1) or np.any(cell_levels > n_levels):
        raise ValueError("max_level_cell must contain active-level counts in [1,nLevels]")
    if edge_levels.shape != (n_edges,) or np.any(edge_levels < 0) or np.any(edge_levels > n_levels):
        raise ValueError("max_level_edge_top must contain counts in [0,nLevels]")
    return cell_levels, edge_levels


def _high_order_mask(
    supplied: ArrayLike | None,
    coefficients: AdvectionCoefficients | None,
    n_levels: int,
    n_edges: int,
    horizontal_order: int,
) -> NDArray[np.int8]:
    value = supplied
    if value is None and coefficients is not None:
        value = coefficients.high_order_advection_mask
    if value is None:
        return np.full((n_levels, n_edges), 0 if horizontal_order == 2 else 1, dtype=np.int8)
    mask = np.asarray(value)
    if mask.shape == (n_edges, n_levels):
        mask = mask.T
    if mask.shape != (n_levels, n_edges) or np.any((mask != 0) & (mask != 1)):
        raise ValueError("high_order_advection_mask must be a zero/one (nLevels,nEdges) array")
    if horizontal_order == 2 and np.any(mask):
        raise ConfigurationRefusal(
            "config_scalar_adv_order",
            horizontal_order,
            "a nonzero highOrderAdvectionMask requests a high-order stencil",
            "config_scalar_adv_order=3 or highOrderAdvectionMask=0",
        )
    return mask.astype(np.int8, copy=False)


def _operator_vertical_flux(
    tracer: FloatArray,
    velocity: FloatArray,
    vertical_cell_size: FloatArray,
    cell_levels: NDArray[np.int64],
    order: int,
    coefficient: float,
) -> FloatArray:
    n_levels, n_cells = tracer.shape
    dtype = tracer.dtype
    output = np.zeros((n_levels + 1, n_cells), dtype=dtype)
    for cell in range(n_cells):
        top = int(cell_levels[cell])
        if top < 2:
            continue
        for interface in range(1, top):
            if interface == 1 or interface == top - 1 or order == 2:
                denominator = vertical_cell_size[interface, cell] + vertical_cell_size[interface - 1, cell]
                weight_upper = vertical_cell_size[interface - 1, cell] / denominator
                weight_lower = vertical_cell_size[interface, cell] / denominator
                output[interface, cell] = velocity[interface, cell] * (
                    weight_upper * tracer[interface, cell]
                    + weight_lower * tracer[interface - 1, cell]
                )
            elif order == 3:
                output[interface, cell] = vertical_flux_3(
                    tracer[interface - 2, cell],
                    tracer[interface - 1, cell],
                    tracer[interface, cell],
                    tracer[interface + 1, cell],
                    velocity[interface, cell],
                    coefficient,
                )
            else:
                output[interface, cell] = vertical_flux_4(
                    tracer[interface - 2, cell],
                    tracer[interface - 1, cell],
                    tracer[interface, cell],
                    tracer[interface + 1, cell],
                    velocity[interface, cell],
                )
    return output


def _operator_horizontal_flux(
    tracer: FloatArray,
    normal_flux: FloatArray,
    dv_edge: FloatArray,
    cells_on_edge: NDArray[np.int64],
    adv: FloatArray,
    adv3: FloatArray,
    n_adv: NDArray[np.int64],
    adv_cells: NDArray[np.int64],
    cell_levels: NDArray[np.int64],
    edge_levels: NDArray[np.int64],
    mask: NDArray[np.int8],
    coefficient: float,
) -> FloatArray:
    n_levels, n_edges = normal_flux.shape
    dtype = tracer.dtype
    output = np.zeros((n_levels, n_edges), dtype=dtype)
    half = dtype.type(0.5)
    coef = dtype.type(coefficient)
    # std/mono source order: edge, centered levels, advection cell, levels.
    for edge in range(n_edges):
        cell1, cell2 = cells_on_edge[edge]
        for level in range(int(edge_levels[edge])):
            if mask[level, edge] == 0:
                weight = dv_edge[edge] * half * normal_flux[level, edge]
                output[level, edge] = output[level, edge] + weight * (
                    tracer[level, cell1] + tracer[level, cell2]
                )
        for slot in range(int(n_adv[edge])):
            cell = int(adv_cells[edge, slot])
            if cell < 0 or cell >= tracer.shape[1]:
                raise ValueError("active adv_cells_for_edge entry is out of range")
            for level in range(int(cell_levels[cell])):
                if mask[level, edge] != 0:
                    weight = adv[edge, slot] + coef * _fortran_sign_one(
                        normal_flux[level, edge], dtype
                    ) * adv3[edge, slot]
                    weight = normal_flux[level, edge] * weight
                    output[level, edge] = output[level, edge] + weight * tracer[level, cell]
    return output


def standard_tracer_advection_tendency(
    mesh: Any,
    tracers: ArrayLike,
    normal_thickness_flux: ArrayLike,
    w: ArrayLike,
    vertical_cell_size: ArrayLike,
    *,
    coefficients: AdvectionCoefficients | None = None,
    adv_coefs: ArrayLike | None = None,
    adv_coefs_3rd: ArrayLike | None = None,
    n_adv_cells_for_edge: ArrayLike | None = None,
    adv_cells_for_edge: ArrayLike | None = None,
    config_scalar_adv_order: int = 3,
    config_scalar_vadv_order: int = 3,
    config_coef_3rd_order: float = 0.25,
    tendency: ArrayLike | None = None,
    max_level_cell: ArrayLike | None = None,
    max_level_edge_top: ArrayLike | None = None,
    high_order_advection_mask: ArrayLike | None = None,
    vertical_divergence_factor: ArrayLike | None = None,
    edge_sign_on_cell: ArrayLike | None = None,
    n_cells_solve: int | None = None,
    in_place: bool = False,
) -> FloatArray:
    """Port ``mpas_tracer_advection_std_tend`` without implicit mutation."""

    horizontal_order = _validate_order(config_scalar_adv_order, "config_scalar_adv_order")
    vertical_order = _validate_order(config_scalar_vadv_order, "config_scalar_vadv_order")
    _validate_third_order_coefficient(config_coef_3rd_order)
    q, squeezed = _as_tracers(tracers, "tracers")
    dtype = q.dtype
    n_tracers, n_levels, q_cells = q.shape
    n_cells, n_edges, counts, edges_on_cell, _, cells_on_edge = _topology(mesh)
    if q_cells != n_cells:
        raise ValueError("tracers cell dimension disagrees with mesh")
    normal = _float_array(normal_thickness_flux, "normal_thickness_flux", dtype)
    vertical_velocity = _float_array(w, "w", dtype)
    vertical_size = _float_array(vertical_cell_size, "vertical_cell_size", dtype)
    if normal.shape != (n_levels, n_edges):
        raise ValueError("normal_thickness_flux must have shape (nLevels,nEdges)")
    if vertical_velocity.shape != (n_levels + 1, n_cells):
        raise ValueError("w must have shape (nLevels+1,nCells)")
    if vertical_size.shape != (n_levels, n_cells):
        raise ValueError("vertical_cell_size must have shape (nLevels,nCells)")
    adv, adv3, n_adv, adv_cells = _resolve_coefficients(
        coefficients, adv_coefs, adv_coefs_3rd, n_adv_cells_for_edge,
        adv_cells_for_edge, n_edges, dtype, horizontal_order
    )
    cell_levels, edge_levels = _level_limits(
        n_levels, n_cells, n_edges, max_level_cell, max_level_edge_top
    )
    mask = _high_order_mask(
        high_order_advection_mask, coefficients, n_levels, n_edges, horizontal_order
    )
    dv_edge = _float_array(_mesh_array(mesh, "dvEdge"), "dvEdge", dtype)
    area_cell = _float_array(_mesh_array(mesh, "areaCell"), "areaCell", dtype)
    signs = _normalize_edge_signs(
        edge_sign_on_cell, counts, edges_on_cell, cells_on_edge, dtype
    )
    vfactor = (
        np.ones(n_levels, dtype=dtype)
        if vertical_divergence_factor is None
        else _float_array(vertical_divergence_factor, "vertical_divergence_factor", dtype)
    )
    if vfactor.shape != (n_levels,):
        raise ValueError("vertical_divergence_factor must have shape (nLevels,)")
    solve_count = n_cells if n_cells_solve is None else int(n_cells_solve)
    if solve_count < 0 or solve_count > n_cells:
        raise ValueError("n_cells_solve is outside [0,nCells]")

    if tendency is None:
        output = np.zeros_like(q)
    else:
        supplied, supplied_squeezed = _as_tracers(tendency, "tendency")
        if supplied.shape != q.shape or supplied_squeezed != squeezed:
            raise ValueError("tendency shape must equal tracers shape")
        if supplied.dtype != dtype:
            supplied = supplied.astype(dtype)
        output = supplied if in_place else supplied.copy()

    coefficient = config_coef_3rd_order if horizontal_order == 3 else 0.0
    vertical_coefficient = config_coef_3rd_order if vertical_order == 3 else 0.0
    for tracer_index in range(n_tracers):
        current = q[tracer_index]
        vertical_flux = _operator_vertical_flux(
            current, vertical_velocity, vertical_size, cell_levels,
            vertical_order, vertical_coefficient
        )
        horizontal_flux = _operator_horizontal_flux(
            current, normal, dv_edge, cells_on_edge, adv, adv3, n_adv,
            adv_cells, cell_levels, edge_levels, mask, coefficient
        )
        for cell in range(n_cells):
            inverse_area = dtype.type(1.0) / area_cell[cell]
            for slot in range(int(counts[cell])):
                edge = int(edges_on_cell[cell, slot])
                for level in range(int(edge_levels[edge])):
                    output[tracer_index, level, cell] = (
                        output[tracer_index, level, cell]
                        + signs[cell, slot] * horizontal_flux[level, edge] * inverse_area
                    )
        for cell in range(solve_count):
            for level in range(int(cell_levels[cell])):
                output[tracer_index, level, cell] = (
                    output[tracer_index, level, cell]
                    + vfactor[level]
                    * (vertical_flux[level + 1, cell] - vertical_flux[level, cell])
                )
    return _restore_tracers(output, squeezed)


def monotonic_tracer_advection_tendency(
    mesh: Any,
    tracers: ArrayLike,
    normal_thickness_flux: ArrayLike,
    w: ArrayLike,
    layer_thickness: ArrayLike,
    vertical_cell_size: ArrayLike,
    tend_layer_thickness: ArrayLike,
    dt: float,
    *,
    coefficients: AdvectionCoefficients | None = None,
    adv_coefs: ArrayLike | None = None,
    adv_coefs_3rd: ArrayLike | None = None,
    n_adv_cells_for_edge: ArrayLike | None = None,
    adv_cells_for_edge: ArrayLike | None = None,
    config_scalar_adv_order: int = 3,
    config_scalar_vadv_order: int = 3,
    config_coef_3rd_order: float = 0.25,
    dzdk_positive: bool = True,
    check_monotonicity: bool = False,
    n_halos: int = 3,
    tendency: ArrayLike | None = None,
    max_level_cell: ArrayLike | None = None,
    max_level_edge_top: ArrayLike | None = None,
    high_order_advection_mask: ArrayLike | None = None,
    vertical_divergence_factor: ArrayLike | None = None,
    edge_sign_on_cell: ArrayLike | None = None,
    n_cells_solve: int | None = None,
    in_place: bool = False,
    allow_regional_sentinels: bool = False,
) -> FloatArray:
    """Port the helpers-based FCT routine ``mpas_tracer_advection_mono_tend``."""

    if n_halos < 3:
        raise ConfigurationRefusal(
            "nHalos",
            n_halos,
            "the frozen monotonic FCT exchanges three halo rows",
            "nHalos=3",
        )
    horizontal_order = _validate_order(config_scalar_adv_order, "config_scalar_adv_order")
    vertical_order = _validate_order(config_scalar_vadv_order, "config_scalar_vadv_order")
    _validate_third_order_coefficient(config_coef_3rd_order)
    q, squeezed = _as_tracers(tracers, "tracers")
    dtype = q.dtype
    n_tracers, n_levels, q_cells = q.shape
    n_cells, n_edges, counts, edges_on_cell, cells_on_cell, cells_on_edge = _topology(
        mesh, allow_regional_sentinels=allow_regional_sentinels
    )
    if q_cells != n_cells:
        raise ValueError("tracers cell dimension disagrees with mesh")
    normal = _float_array(normal_thickness_flux, "normal_thickness_flux", dtype)
    vertical_velocity = _float_array(w, "w", dtype)
    thickness = _float_array(layer_thickness, "layer_thickness", dtype)
    vertical_size = _float_array(vertical_cell_size, "vertical_cell_size", dtype)
    thickness_tendency = _float_array(tend_layer_thickness, "tend_layer_thickness", dtype)
    expected_cell = (n_levels, n_cells)
    if normal.shape != (n_levels, n_edges):
        raise ValueError("normal_thickness_flux must have shape (nLevels,nEdges)")
    if vertical_velocity.shape != (n_levels + 1, n_cells):
        raise ValueError("w must have shape (nLevels+1,nCells)")
    for name, value in (("layer_thickness", thickness), ("vertical_cell_size", vertical_size), ("tend_layer_thickness", thickness_tendency)):
        if value.shape != expected_cell:
            raise ValueError(f"{name} must have shape (nLevels,nCells)")
    step = dtype.type(dt)
    new_thickness = thickness + step * thickness_tendency
    if np.any(new_thickness <= 0):
        raise ValueError("layerThickness + dt*tend_layerThickness must stay positive")
    inverse_new_thickness = dtype.type(1.0) / new_thickness
    adv, adv3, n_adv, adv_cells = _resolve_coefficients(
        coefficients, adv_coefs, adv_coefs_3rd, n_adv_cells_for_edge,
        adv_cells_for_edge, n_edges, dtype, horizontal_order
    )
    cell_levels, edge_levels = _level_limits(
        n_levels, n_cells, n_edges, max_level_cell, max_level_edge_top
    )
    mask = _high_order_mask(
        high_order_advection_mask, coefficients, n_levels, n_edges, horizontal_order
    )
    dv_edge = _float_array(_mesh_array(mesh, "dvEdge"), "dvEdge", dtype)
    area_cell = _float_array(_mesh_array(mesh, "areaCell"), "areaCell", dtype)
    signs = _normalize_edge_signs(
        edge_sign_on_cell, counts, edges_on_cell, cells_on_edge, dtype
    )
    vfactor = (
        np.ones(n_levels, dtype=dtype)
        if vertical_divergence_factor is None
        else _float_array(vertical_divergence_factor, "vertical_divergence_factor", dtype)
    )
    if vfactor.shape != (n_levels,):
        raise ValueError("vertical_divergence_factor must have shape (nLevels,)")
    solve_count = n_cells if n_cells_solve is None else int(n_cells_solve)
    if solve_count < 0 or solve_count > n_cells:
        raise ValueError("n_cells_solve is outside [0,nCells]")
    if tendency is None:
        output = np.zeros_like(q)
    else:
        supplied, supplied_squeezed = _as_tracers(tendency, "tendency")
        if supplied.shape != q.shape or supplied_squeezed != squeezed:
            raise ValueError("tendency shape must equal tracers shape")
        if supplied.dtype != dtype:
            supplied = supplied.astype(dtype)
        output = supplied if in_place else supplied.copy()
    eps = dtype.type(1.0e-10)
    zero = dtype.type(0.0)
    one = dtype.type(1.0)
    horizontal_coefficient = config_coef_3rd_order if horizontal_order == 3 else 0.0
    vertical_coefficient = config_coef_3rd_order if vertical_order == 3 else 0.0

    for tracer_index in range(n_tracers):
        current = q[tracer_index]
        vertical_flux = _operator_vertical_flux(
            current, vertical_velocity, vertical_size, cell_levels,
            vertical_order, vertical_coefficient
        )
        horizontal_flux = _operator_horizontal_flux(
            current, normal, dv_edge, cells_on_edge, adv, adv3, n_adv,
            adv_cells, cell_levels, edge_levels, mask, horizontal_coefficient
        )
        upwind_tendency = np.zeros((n_levels, n_cells), dtype=dtype)
        flux_incoming = np.zeros((n_levels, n_cells), dtype=dtype)
        flux_outgoing = np.zeros((n_levels, n_cells), dtype=dtype)
        tracer_min = np.zeros((n_levels, n_cells), dtype=dtype)
        tracer_max = np.zeros((n_levels, n_cells), dtype=dtype)

        for cell in range(n_cells):
            top = int(cell_levels[cell])
            for level in range(top):
                lower = max(0, level - 1)
                upper = min(top, level + 2)
                tracer_min[level, cell] = np.min(current[lower:upper, cell])
                tracer_max[level, cell] = np.max(current[lower:upper, cell])
            for slot in range(int(counts[cell])):
                neighbor = int(cells_on_cell[cell, slot])
                shared_top = min(top, int(cell_levels[neighbor]))
                for level in range(shared_top):
                    tracer_max[level, cell] = max(tracer_max[level, cell], current[level, neighbor])
                    tracer_min[level, cell] = min(tracer_min[level, cell], current[level, neighbor])

            for interface in range(1, top):
                velocity = vertical_velocity[interface, cell]
                if dzdk_positive:
                    flux_upwind = max(zero, velocity) * current[interface - 1, cell] + min(zero, velocity) * current[interface, cell]
                else:
                    flux_upwind = min(zero, velocity) * current[interface - 1, cell] + max(zero, velocity) * current[interface, cell]
                upwind_tendency[interface - 1, cell] = upwind_tendency[interface - 1, cell] + flux_upwind
                upwind_tendency[interface, cell] = upwind_tendency[interface, cell] - flux_upwind
                vertical_flux[interface, cell] = vertical_flux[interface, cell] - flux_upwind
            for level in range(top):
                if dzdk_positive:
                    flux_incoming[level, cell] = -(
                        min(zero, vertical_flux[level + 1, cell])
                        - max(zero, vertical_flux[level, cell])
                    )
                    flux_outgoing[level, cell] = -(
                        max(zero, vertical_flux[level + 1, cell])
                        - min(zero, vertical_flux[level, cell])
                    )
                else:
                    flux_incoming[level, cell] = (
                        max(zero, vertical_flux[level + 1, cell])
                        - min(zero, vertical_flux[level, cell])
                    )
                    flux_outgoing[level, cell] = (
                        min(zero, vertical_flux[level + 1, cell])
                        - max(zero, vertical_flux[level, cell])
                    )

        # Remove horizontal upwind flux from the high-order flux.
        for edge in range(n_edges):
            cell1, cell2 = cells_on_edge[edge]
            for level in range(int(edge_levels[edge])):
                velocity = normal[level, edge]
                flux_upwind = dv_edge[edge] * (
                    max(zero, velocity) * current[level, cell1]
                    + min(zero, velocity) * current[level, cell2]
                )
                horizontal_flux[level, edge] = horizontal_flux[level, edge] - flux_upwind

        for cell in range(n_cells):
            inverse_area = one / area_cell[cell]
            for slot in range(int(counts[cell])):
                edge = int(edges_on_cell[cell, slot])
                cell1, cell2 = cells_on_edge[edge]
                for level in range(int(edge_levels[edge])):
                    velocity = normal[level, edge]
                    flux_upwind = dv_edge[edge] * (
                        max(zero, velocity) * current[level, cell1]
                        + min(zero, velocity) * current[level, cell2]
                    )
                    signed_residual = signs[cell, slot] * horizontal_flux[level, edge]
                    upwind_tendency[level, cell] = (
                        upwind_tendency[level, cell]
                        + signs[cell, slot] * flux_upwind * inverse_area
                    )
                    flux_outgoing[level, cell] = flux_outgoing[level, cell] + min(zero, signed_residual) * inverse_area
                    flux_incoming[level, cell] = flux_incoming[level, cell] + max(zero, signed_residual) * inverse_area

        for cell in range(n_cells):
            for level in range(int(cell_levels[cell])):
                tracer_min_new = (
                    current[level, cell] * thickness[level, cell]
                    + step * (upwind_tendency[level, cell] + flux_outgoing[level, cell])
                ) * inverse_new_thickness[level, cell]
                tracer_max_new = (
                    current[level, cell] * thickness[level, cell]
                    + step * (upwind_tendency[level, cell] + flux_incoming[level, cell])
                ) * inverse_new_thickness[level, cell]
                tracer_upwind_new = (
                    current[level, cell] * thickness[level, cell]
                    + step * upwind_tendency[level, cell]
                ) * inverse_new_thickness[level, cell]
                scale_factor = (tracer_max[level, cell] - tracer_upwind_new) / (
                    tracer_max_new - tracer_upwind_new + eps
                )
                flux_incoming[level, cell] = min(one, max(zero, scale_factor))
                scale_factor = (tracer_upwind_new - tracer_min[level, cell]) / (
                    tracer_upwind_new - tracer_min_new + eps
                )
                flux_outgoing[level, cell] = min(one, max(zero, scale_factor))

        for edge in range(n_edges):
            cell1, cell2 = cells_on_edge[edge]
            for level in range(int(edge_levels[edge])):
                flux = horizontal_flux[level, edge]
                flux = max(zero, flux) * min(
                    flux_outgoing[level, cell1], flux_incoming[level, cell2]
                ) + min(zero, flux) * min(
                    flux_incoming[level, cell1], flux_outgoing[level, cell2]
                )
                horizontal_flux[level, edge] = flux
        for cell in range(solve_count):
            for interface in range(1, int(cell_levels[cell])):
                flux = vertical_flux[interface, cell]
                if dzdk_positive:
                    flux = max(zero, flux) * min(
                        flux_outgoing[interface - 1, cell], flux_incoming[interface, cell]
                    ) + min(zero, flux) * min(
                        flux_outgoing[interface, cell], flux_incoming[interface - 1, cell]
                    )
                else:
                    flux = max(zero, flux) * min(
                        flux_outgoing[interface, cell], flux_incoming[interface - 1, cell]
                    ) + min(zero, flux) * min(
                        flux_outgoing[interface - 1, cell], flux_incoming[interface, cell]
                    )
                vertical_flux[interface, cell] = flux

        check_state = np.zeros((n_levels, n_cells), dtype=dtype) if check_monotonicity else None
        for cell in range(n_cells):
            inverse_area = one / area_cell[cell]
            for slot in range(int(counts[cell])):
                edge = int(edges_on_cell[cell, slot])
                for level in range(int(edge_levels[edge])):
                    contribution = signs[cell, slot] * horizontal_flux[level, edge] * inverse_area
                    output[tracer_index, level, cell] = output[tracer_index, level, cell] + contribution
                    if check_state is not None:
                        check_state[level, cell] = check_state[level, cell] + contribution
        for cell in range(solve_count):
            for level in range(int(cell_levels[cell])):
                contribution = (
                    vfactor[level] * (vertical_flux[level + 1, cell] - vertical_flux[level, cell])
                    + upwind_tendency[level, cell]
                )
                output[tracer_index, level, cell] = output[tracer_index, level, cell] + contribution
                if check_state is not None:
                    check_state[level, cell] = (
                        check_state[level, cell] + contribution
                    )
                    updated = (
                        current[level, cell] * thickness[level, cell]
                        + step * check_state[level, cell]
                    ) * inverse_new_thickness[level, cell]
                    if updated < tracer_min[level, cell] - eps or updated > tracer_max[level, cell] + eps:
                        warnings.warn(
                            f"monotonicity check failed for tracer {tracer_index}, level {level}, cell {cell}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
    return _restore_tracers(output, squeezed)


def mpas_tracer_advection_std_tend(
    tracers: ArrayLike,
    adv_coefs: ArrayLike,
    adv_coefs_3rd: ArrayLike,
    nAdvCellsForEdge: ArrayLike,
    advCellsForEdge: ArrayLike,
    normalThicknessFlux: ArrayLike,
    w: ArrayLike,
    layerThickness: ArrayLike,
    verticalCellSize: ArrayLike,
    dt: float,
    meshPool: Any,
    tend_layerThickness: ArrayLike,
    tend: ArrayLike | None = None,
    **kwargs: Any,
) -> FloatArray:
    """Fortran-named adapter for ``mpas_tracer_advection_std_tend``.

    ``layerThickness``, ``dt``, and ``tend_layerThickness`` are intentionally
    accepted because they are present in the frozen interface, although that
    standard routine does not read them.
    """

    del layerThickness, dt, tend_layerThickness
    return standard_tracer_advection_tendency(
        meshPool,
        tracers,
        normalThicknessFlux,
        w,
        verticalCellSize,
        adv_coefs=adv_coefs,
        adv_coefs_3rd=adv_coefs_3rd,
        n_adv_cells_for_edge=nAdvCellsForEdge,
        adv_cells_for_edge=advCellsForEdge,
        tendency=tend,
        **kwargs,
    )


def mpas_tracer_advection_mono_tend(
    tracers: ArrayLike,
    adv_coefs: ArrayLike,
    adv_coefs_3rd: ArrayLike,
    nAdvCellsForEdge: ArrayLike,
    advCellsForEdge: ArrayLike,
    normalThicknessFlux: ArrayLike,
    w: ArrayLike,
    layerThickness: ArrayLike,
    verticalCellSize: ArrayLike,
    dt: float,
    meshPool: Any,
    tend_layerThickness: ArrayLike,
    tend: ArrayLike | None = None,
    **kwargs: Any,
) -> FloatArray:
    """Fortran-named adapter for ``mpas_tracer_advection_mono_tend``."""

    return monotonic_tracer_advection_tendency(
        meshPool,
        tracers,
        normalThicknessFlux,
        w,
        layerThickness,
        verticalCellSize,
        tend_layerThickness,
        dt,
        adv_coefs=adv_coefs,
        adv_coefs_3rd=adv_coefs_3rd,
        n_adv_cells_for_edge=nAdvCellsForEdge,
        adv_cells_for_edge=advCellsForEdge,
        tendency=tend,
        **kwargs,
    )


def _atmosphere_vertical_flux(
    tracer: FloatArray,
    velocity: FloatArray,
    fnm: FloatArray,
    fnp: FloatArray,
    order: int,
    coefficient: float,
) -> FloatArray:
    """Time-integration vertical flux with zero top and bottom boundaries."""

    n_levels, n_cells = tracer.shape
    dtype = tracer.dtype
    output = np.zeros((n_levels + 1, n_cells), dtype=dtype)
    for cell in range(n_cells):
        for interface in range(1, n_levels):
            if interface == 1 or interface == n_levels - 1 or order == 2:
                output[interface, cell] = velocity[interface, cell] * (
                    fnm[interface] * tracer[interface, cell]
                    + fnp[interface] * tracer[interface - 1, cell]
                )
            elif order == 3:
                output[interface, cell] = atmosphere_vertical_flux_3(
                    tracer[interface - 2, cell],
                    tracer[interface - 1, cell],
                    tracer[interface, cell],
                    tracer[interface + 1, cell],
                    velocity[interface, cell],
                    coefficient,
                )
            else:
                output[interface, cell] = vertical_flux_4(
                    tracer[interface - 2, cell],
                    tracer[interface - 1, cell],
                    tracer[interface, cell],
                    tracer[interface + 1, cell],
                    velocity[interface, cell],
                )
    return output


#: mpas_atm_boundaries.F:36-37, re-declared here so the transport module's
#: regional edge conditions read as the native expressions they transcribe.
_N_SPEC_ZONE = 2
_N_RELAX_ZONE = 5


def _regional_edge_flux_mode(
    config_apply_lbcs: bool,
    bdy_mask_edge: NDArray[np.int64] | None,
    edge: int,
) -> str:
    """The three-way edge split of atm_advance_scalars_work F:4764-4842.

    ``full``: ``(.not.config_apply_lbcs) .or. (bdyMaskEdge(iEdge) <
    nRelaxZone-1)``: the complete high-order stencil.
    ``upwind``: ``config_apply_lbcs .and. (bdyMaskEdge >= nRelaxZone-1)
    .and. (bdyMaskEdge <= nRelaxZone)``: the first-order downgrade at
    mask-4/5 edges.
    ``skip``, every other masked edge (rings 6-7): the native scratch is
    left unwritten, and no updated cell (mask <= nRelaxZone) ever reads it.
    """

    if bdy_mask_edge is None:
        return "full"
    mask = int(bdy_mask_edge[edge])
    if (not config_apply_lbcs) or (mask < _N_RELAX_ZONE - 1):
        return "full"
    if config_apply_lbcs and _N_RELAX_ZONE - 1 <= mask <= _N_RELAX_ZONE:
        return "upwind"
    return "skip"


def _atmosphere_horizontal_edge_values(
    tracers: FloatArray,
    velocity: FloatArray,
    dv_edge: FloatArray,
    cells_on_edge: NDArray[np.int64],
    adv: FloatArray,
    adv3: FloatArray,
    n_adv: NDArray[np.int64],
    adv_cells: NDArray[np.int64],
    order: int,
    coefficient: float,
    monotonic_source_flux: bool = False,
    config_apply_lbcs: bool = False,
    bdy_mask_edge: NDArray[np.int64] | None = None,
) -> FloatArray:
    """Scalar value times edge length, before multiplication by ``ruAvg``."""

    n_tracers, n_levels, _ = tracers.shape
    n_edges = cells_on_edge.shape[0]
    dtype = tracers.dtype
    result = np.zeros((n_tracers, n_levels, n_edges), dtype=dtype)
    half = dtype.type(0.5)
    coef = dtype.type(coefficient if order == 3 else 0.0)
    if order == 2:
        cell1, cell2 = cells_on_edge.T
        result[:] = dv_edge * half * (tracers[:, :, cell1] + tracers[:, :, cell2])
        if monotonic_source_flux:
            result *= velocity
        return result

    active = np.ones(n_edges, dtype=bool)
    if bdy_mask_edge is not None and config_apply_lbcs and not monotonic_source_flux:
        # The monotonic path applies the regional downgrade later, to fluxes.
        active = bdy_mask_edge < _N_RELAX_ZONE - 1
        edge = np.flatnonzero((bdy_mask_edge >= _N_RELAX_ZONE - 1)
                              & (bdy_mask_edge <= _N_RELAX_ZONE))
        cell1, cell2 = cells_on_edge[edge].T
        direction = np.copysign(half, velocity[:, edge])
        positive = dv_edge[edge] * np.abs(direction + half)
        negative = dv_edge[edge] * np.abs(direction - half)
        result[:, :, edge] = (positive * tracers[:, :, cell1]
                              + negative * tracers[:, :, cell2])

    if monotonic_source_flux:
        edge = np.flatnonzero(n_adv == 10)
        if edge.size:
            sign = np.copysign(dtype.type(1.0), velocity[:, edge])
            # Keep the literal ten-term association, including its first term
            # without an initial +0, and multiply velocity after the whole sum.
            value = (adv[edge, 0] + coef * sign * adv3[edge, 0]) * tracers[:, :, adv_cells[edge, 0]]
            for slot in range(1, 10):
                value = value + (adv[edge, slot] + coef * sign * adv3[edge, slot]) * tracers[:, :, adv_cells[edge, slot]]
            result[:, :, edge] = velocity[:, edge] * value
        active &= n_adv != 10

    # Preserve donor-slot accumulation order. The generic monotonic source
    # multiplies velocity into each weight before multiplying the tracer.
    for slot in range(adv_cells.shape[1]):
        edge = np.flatnonzero(active & (n_adv > slot))
        cell = adv_cells[edge, slot]
        if np.any((cell < 0) | (cell >= tracers.shape[2])):
            raise ValueError("active adv_cells_for_edge entry is out of range")
        sign = np.copysign(dtype.type(1.0), velocity[:, edge])
        weight = adv[edge, slot] + coef * sign * adv3[edge, slot]
        if monotonic_source_flux:
            weight = velocity[:, edge] * weight
        result[:, :, edge] += weight * tracers[:, :, cell]
    return result


def _atmosphere_inputs(
    mesh: Any,
    scalar_old: ArrayLike,
    scalar_stage: ArrayLike,
    rho_zz_old: ArrayLike,
    rho_zz_new: ArrayLike,
    uh_avg: ArrayLike,
    ww_avg: ArrayLike,
    fzm: ArrayLike | None,
    fzp: ArrayLike | None,
    rdzw: ArrayLike | None,
    allow_regional_sentinels: bool = False,
) -> tuple[
    FloatArray,
    FloatArray,
    bool,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    int,
    int,
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    old, squeezed = _as_tracers(scalar_old, "scalar_old")
    stage, stage_squeezed = _as_tracers(scalar_stage, "scalar_stage")
    if old.shape != stage.shape or squeezed != stage_squeezed:
        raise ValueError("scalar_old and scalar_stage shapes must agree")
    dtype = old.dtype
    stage = stage.astype(dtype, copy=False)
    _, n_levels, q_cells = old.shape
    n_cells, n_edges, counts, edges_on_cell, cells_on_cell, cells_on_edge = _topology(
        mesh, allow_regional_sentinels=allow_regional_sentinels
    )
    if q_cells != n_cells:
        raise ValueError("scalar cell dimension disagrees with mesh")
    rho_old = _float_array(rho_zz_old, "rho_zz_old", dtype)
    rho_new = _float_array(rho_zz_new, "rho_zz_new", dtype)
    horizontal_velocity = _float_array(uh_avg, "uh_avg", dtype)
    vertical_velocity = _float_array(ww_avg, "ww_avg", dtype)
    fnm = _float_array(_mesh_array(mesh, "fzm") if fzm is None else fzm, "fzm", dtype)
    fnp = _float_array(_mesh_array(mesh, "fzp") if fzp is None else fzp, "fzp", dtype)
    rdnw = _float_array(_mesh_array(mesh, "rdzw") if rdzw is None else rdzw, "rdzw", dtype)
    if rho_old.shape != (n_levels, n_cells) or rho_new.shape != rho_old.shape:
        raise ValueError("rho_zz_old/new must have shape (nLevels,nCells)")
    if horizontal_velocity.shape != (n_levels, n_edges):
        raise ValueError("uh_avg must have shape (nLevels,nEdges)")
    if vertical_velocity.shape != (n_levels + 1, n_cells):
        raise ValueError("ww_avg must have shape (nLevels+1,nCells)")
    if fnm.shape != (n_levels,) or fnp.shape != (n_levels,) or rdnw.shape != (n_levels,):
        raise ValueError("fzm, fzp, and rdzw must have shape (nLevels,)")
    return (
        old,
        stage,
        squeezed,
        rho_old,
        rho_new,
        horizontal_velocity,
        vertical_velocity,
        fnm,
        fnp,
        rdnw,
        n_cells,
        n_edges,
        counts,
        edges_on_cell,
        cells_on_cell,
        cells_on_edge,
    )


def _admit_lateral_boundaries(
    config_apply_lbcs: bool,
    bdy_mask_cell: ArrayLike | None,
    bdy_mask_edge: ArrayLike | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]] | None:
    """Admit or refuse the regional transport branch, by name.

    ``config_apply_lbcs=True`` is admitted exactly when both boundary masks
    are supplied: the caller that owns them (the regional_v841 runtime) also
    owns the driving scalars that ``atm_bdy_adjust_scalars``/``set_scalars``
    consume after this routine, so masks-present is the honest proxy for
    "real LBC state is loaded".  Masks without ``config_apply_lbcs`` refuse
    the same way ``mpas_atm_bdy_checks`` refuses a masked mesh with
    ``config_apply_lbcs=false``.
    """

    if config_apply_lbcs:
        if bdy_mask_cell is None or bdy_mask_edge is None:
            raise ConfigurationRefusal(
                "config_apply_lbcs",
                config_apply_lbcs,
                "the regional specified/relaxation-zone scalar update needs "
                "the 7-ring boundary masks of the culled mesh",
                "bdy_mask_cell=... and bdy_mask_edge=... from the regional "
                "mesh (regional_v841.derive_regional_masks)",
            )
        return (
            np.asarray(bdy_mask_cell, dtype=np.int64),
            np.asarray(bdy_mask_edge, dtype=np.int64),
        )
    if bdy_mask_cell is not None or bdy_mask_edge is not None:
        raise ConfigurationRefusal(
            "config_apply_lbcs",
            config_apply_lbcs,
            "boundary masks were supplied but the limited-area branch is off; "
            "silently ignoring them would run a regional mesh as global",
            "config_apply_lbcs=True for limited-area transport",
        )
    return None


def advance_scalars(
    mesh: Any,
    scalar_old: ArrayLike,
    scalar_stage: ArrayLike,
    rho_zz_old: ArrayLike,
    rho_zz_new: ArrayLike,
    uh_avg: ArrayLike,
    ww_avg: ArrayLike,
    dt: float,
    *,
    coefficients: AdvectionCoefficients | None = None,
    adv_coefs: ArrayLike | None = None,
    adv_coefs_3rd: ArrayLike | None = None,
    n_adv_cells_for_edge: ArrayLike | None = None,
    adv_cells_for_edge: ArrayLike | None = None,
    scalar_tendency: ArrayLike | None = None,
    fzm: ArrayLike | None = None,
    fzp: ArrayLike | None = None,
    rdzw: ArrayLike | None = None,
    rk_step: int = 3,
    config_time_integration_order: int = 3,
    config_scalar_adv_order: int = 3,
    config_scalar_vadv_order: int = 3,
    config_coef_3rd_order: float = 0.25,
    config_apply_lbcs: bool = False,
    bdy_mask_cell: ArrayLike | None = None,
    bdy_mask_edge: ArrayLike | None = None,
    advance_density: bool = False,
    inv_area_cell: ArrayLike | None = None,
    in_place: bool = False,
) -> ScalarTransportResult:
    """Unrestricted atmosphere scalar update from lines 3049-3330.

    Orders 2/3/4 use the corresponding complete horizontal and vertical
    stencils.  The frozen atmosphere routine itself instantiates order three;
    exposing all Registry-admitted orders here prevents an order knob from
    being silently ignored.

    With ``config_apply_lbcs=True`` and both boundary masks supplied, the
    regional branch of ``atm_advance_scalars_work`` is active: mask-4/5
    edges take the first-order upwind downgrade, mask>=6 edge scratch is
    unwritten, and specified-zone cells (mask > nRelaxZone) are not updated
    here (F:4861), they take driving values at the end of the step.
    """

    masks = _admit_lateral_boundaries(
        config_apply_lbcs, bdy_mask_cell, bdy_mask_edge
    )
    horizontal_order = _validate_order(config_scalar_adv_order, "config_scalar_adv_order")
    vertical_order = _validate_order(config_scalar_vadv_order, "config_scalar_vadv_order")
    _validate_third_order_coefficient(config_coef_3rd_order)
    if config_time_integration_order not in (2, 3):
        raise ConfigurationRefusal(
            "config_time_integration_order",
            config_time_integration_order,
            "the frozen scalar density interpolation has RK2 and RK3 weights only",
            "config_time_integration_order=3",
        )
    if rk_step not in (1, 2, 3):
        raise ConfigurationRefusal(
            "rk_step", rk_step, "the frozen Runge--Kutta transport has three stages", "rk_step=3"
        )
    (
        old, stage, squeezed, rho_old, rho_new, horizontal_velocity,
        vertical_velocity, fnm, fnp, rdnw, n_cells, n_edges, counts,
        edges_on_cell, _, cells_on_edge,
    ) = _atmosphere_inputs(
        mesh, scalar_old, scalar_stage, rho_zz_old, rho_zz_new, uh_avg,
        ww_avg, fzm, fzp, rdzw,
        allow_regional_sentinels=masks is not None,
    )
    mask_cell_array = None if masks is None else masks[0]
    mask_edge_array = None if masks is None else masks[1]
    dtype = old.dtype
    step = dtype.type(dt)
    if not np.isfinite(step) or step < 0:
        raise ValueError("dt must be finite and non-negative")
    adv, adv3, n_adv, adv_cells = _resolve_coefficients(
        coefficients, adv_coefs, adv_coefs_3rd, n_adv_cells_for_edge,
        adv_cells_for_edge, n_edges, dtype, horizontal_order
    )
    dv_edge = _float_array(_mesh_array(mesh, "dvEdge"), "dvEdge", dtype)
    area_cell = _float_array(_mesh_array(mesh, "areaCell"), "areaCell", dtype)
    source_inverse_area = None
    if inv_area_cell is not None:
        source_inverse_area = _source_inverse_array(
            inv_area_cell, "inv_area_cell", dtype
        )
        if source_inverse_area.shape != area_cell.shape:
            raise ValueError("inv_area_cell must have one value per cell")
    signs = _operator_edge_signs(counts, edges_on_cell, cells_on_edge).astype(dtype)
    if scalar_tendency is None:
        source = np.zeros_like(old)
    else:
        source, source_squeezed = _as_tracers(scalar_tendency, "scalar_tendency")
        if source.shape != old.shape or source_squeezed != squeezed:
            raise ValueError("scalar_tendency shape must equal scalar_old shape")
        source = source.astype(dtype, copy=False)

    if not advance_density:
        weight_new = dtype.type(1.0)
    elif rk_step == 1:
        weight_new = dtype.type(1.0 / (3.0 if config_time_integration_order == 3 else 2.0))
    elif rk_step == 2:
        weight_new = dtype.type(0.5)
    else:
        weight_new = dtype.type(1.0)
    target_density = (dtype.type(1.0) - weight_new) * rho_old + weight_new * rho_new
    if np.any(target_density <= 0):
        raise ValueError("time-interpolated rho_zz must stay positive")

    edge_values = _atmosphere_horizontal_edge_values(
        stage, horizontal_velocity, dv_edge, cells_on_edge, adv, adv3,
        n_adv, adv_cells, horizontal_order, config_coef_3rd_order,
        config_apply_lbcs=config_apply_lbcs,
        bdy_mask_edge=mask_edge_array,
    )
    output = stage if in_place else stage.copy()
    n_tracers, n_levels, _ = old.shape
    for scalar in range(n_tracers):
        vertical_flux = _atmosphere_vertical_flux(
            stage[scalar], vertical_velocity, fnm, fnp, vertical_order,
            config_coef_3rd_order if vertical_order == 3 else 0.0
        )
        for cell in range(n_cells):
            if (
                mask_cell_array is not None
                and int(mask_cell_array[cell]) > _N_RELAX_ZONE
            ):
                # F:4861: the specified zone is not updated in this routine;
                # it takes driving values via atm_bdy_set_scalars at the end
                # of the full timestep.
                continue
            if source_inverse_area is None:
                tendency_column = source[scalar, :, cell].copy()
                for slot in range(int(counts[cell])):
                    edge = int(edges_on_cell[cell, slot])
                    for level in range(n_levels):
                        tendency_column[level] = tendency_column[level] + (
                            signs[cell, slot]
                            * horizontal_velocity[level, edge]
                            * edge_values[scalar, level, edge]
                            / area_cell[cell]
                        )
            else:
                tendency_column = np.zeros(n_levels, dtype=dtype)
                for slot in range(int(counts[cell])):
                    edge = int(edges_on_cell[cell, slot])
                    for level in range(n_levels):
                        tendency_column[level] = tendency_column[level] + (
                            signs[cell, slot]
                            * horizontal_velocity[level, edge]
                            * edge_values[scalar, level, edge]
                        )
                tendency_column = (
                    tendency_column * source_inverse_area[cell]
                    + source[scalar, :, cell]
                )
            for level in range(n_levels):
                numerator = (
                    old[scalar, level, cell] * rho_old[level, cell]
                    + step
                    * (
                        tendency_column[level]
                        - rdnw[level]
                        * (vertical_flux[level + 1, cell] - vertical_flux[level, cell])
                    )
                )
                if source_inverse_area is None:
                    output[scalar, level, cell] = (
                        numerator / target_density[level, cell]
                    )
                else:
                    rho_zz_new_inv = dtype.type(1.0) / target_density[
                        level, cell
                    ]
                    output[scalar, level, cell] = numerator * rho_zz_new_inv
    return ScalarTransportResult(_restore_tracers(output, squeezed), target_density.copy())


def advance_scalars_monotonic(
    mesh: Any,
    scalar_old: ArrayLike,
    scalar_stage: ArrayLike,
    rho_zz_old: ArrayLike,
    rho_zz_new: ArrayLike,
    uh_avg: ArrayLike,
    ww_avg: ArrayLike,
    dt: float,
    *,
    coefficients: AdvectionCoefficients | None = None,
    adv_coefs: ArrayLike | None = None,
    adv_coefs_3rd: ArrayLike | None = None,
    n_adv_cells_for_edge: ArrayLike | None = None,
    adv_cells_for_edge: ArrayLike | None = None,
    scalar_tendency: ArrayLike | None = None,
    fzm: ArrayLike | None = None,
    fzp: ArrayLike | None = None,
    rdzw: ArrayLike | None = None,
    config_scalar_adv_order: int = 3,
    config_scalar_vadv_order: int = 3,
    config_coef_3rd_order: float = 0.25,
    config_apply_lbcs: bool = False,
    bdy_mask_cell: ArrayLike | None = None,
    bdy_mask_edge: ArrayLike | None = None,
    advance_density: bool = False,
    n_halos: int = 3,
    inv_area_cell: ArrayLike | None = None,
    in_place: bool = False,
) -> ScalarTransportResult:
    """Zalesak FCT atmosphere update from lines 3487-4211.

    It includes the low-order positive update, incoming/outgoing scale factors,
    horizontal and vertical antidiffusive-flux rescaling, density reintegration
    for split transport, and the source's final roundoff-level nonnegative
    clamp.

    The regional branch transcribes atm_advance_scalars_mono_work's
    limited-area quirks REPLICATED, not fixed, because the dycore pin law
    makes native bytes the definition of correct:

    * the mask-4/5 edge condition ``config_apply_lbcs .and. (bdyMaskEdge ==
      nRelaxZone) .or. (bdyMaskEdge == nRelaxZone-1)`` (F:5541/F:5654) keeps
      its Fortran operator precedence -- ``.and.`` binds tighter than
      ``.or.`` -- so the mask-4 half fires regardless of config_apply_lbcs;
    * the copy-back at F:5771 admits only ``bdyMaskCell <= nSpecZone``
      (rings 0-2), EXCLUDING relaxation rings 3-5, which therefore keep
      their pre-transport values until atm_bdy_adjust_scalars nudges them;
    * garbage-index gathers on ring-7 rows read the explicitly zeroed
      garbage column of the scratch arrays (advance_scalars wrapper,
      atm_srk3:2985-3007), reproduced here by a zero-padded column.

    No packaged compiled-MPAS oracle exercises monotonic regional transport
    (the CANDIDATE-REGIONAL-DRY record pins config_monotonic=false), so this
    branch is a documented transcription whose byte proof waits for a mono
    regional reference mint.
    """

    masks = _admit_lateral_boundaries(
        config_apply_lbcs, bdy_mask_cell, bdy_mask_edge
    )
    if n_halos < 3:
        raise ConfigurationRefusal(
            "nHalos", n_halos, "the monotonic transport state and limiter need three halo rows", "nHalos=3"
        )
    horizontal_order = _validate_order(config_scalar_adv_order, "config_scalar_adv_order")
    vertical_order = _validate_order(config_scalar_vadv_order, "config_scalar_vadv_order")
    _validate_third_order_coefficient(config_coef_3rd_order)
    (
        old, stage, squeezed, rho_old, rho_new, horizontal_velocity,
        vertical_velocity, fnm, fnp, rdnw, n_cells, n_edges, counts,
        edges_on_cell, cells_on_cell, cells_on_edge,
    ) = _atmosphere_inputs(
        mesh, scalar_old, scalar_stage, rho_zz_old, rho_zz_new, uh_avg,
        ww_avg, fzm, fzp, rdzw,
        allow_regional_sentinels=masks is not None,
    )
    dtype = old.dtype
    mask_cell_array = None if masks is None else masks[0]
    mask_edge_array = None if masks is None else masks[1]
    unpadded_stage = stage
    if masks is not None:
        # The garbage column: sentinel gathers on ring-7 rows read it, and
        # the native wrapper zeroes exactly this column of every scratch
        # array (atm_srk3:2985-3007).  rho pads are 1.0 so no dead-lane
        # division can trap; no real column consumes them.
        def _pad_cells(array, value=0.0):
            pad_shape = array.shape[:-1] + (1,)
            return np.concatenate(
                [array, np.full(pad_shape, value, dtype=array.dtype)], axis=-1
            )

        old = _pad_cells(old)
        stage = _pad_cells(stage)
        rho_old = _pad_cells(rho_old, 1.0)
        rho_new = _pad_cells(rho_new, 1.0)
        vertical_velocity = _pad_cells(vertical_velocity)
    step = dtype.type(dt)
    if not np.isfinite(step) or step < 0:
        raise ValueError("dt must be finite and non-negative")
    adv, adv3, n_adv, adv_cells = _resolve_coefficients(
        coefficients, adv_coefs, adv_coefs_3rd, n_adv_cells_for_edge,
        adv_cells_for_edge, n_edges, dtype, horizontal_order
    )
    dv_edge = _float_array(_mesh_array(mesh, "dvEdge"), "dvEdge", dtype)
    area_cell = _float_array(_mesh_array(mesh, "areaCell"), "areaCell", dtype)
    source_inverse_area = None
    if inv_area_cell is not None:
        source_inverse_area = _source_inverse_array(
            inv_area_cell, "inv_area_cell", dtype
        )
        if source_inverse_area.shape != area_cell.shape:
            raise ValueError("inv_area_cell must have one value per cell")
    signs = _operator_edge_signs(counts, edges_on_cell, cells_on_edge).astype(dtype)
    divergence_signs = -signs
    n_tracers, n_levels, _ = old.shape
    if scalar_tendency is None:
        source = np.zeros_like(old)
    else:
        source, source_squeezed = _as_tracers(scalar_tendency, "scalar_tendency")
        expected_shape = (
            old.shape if masks is None else old.shape[:-1] + (n_cells,)
        )
        if source.shape != expected_shape or source_squeezed != squeezed:
            raise ValueError("scalar_tendency shape must equal scalar_old shape")
        source = source.astype(dtype, copy=False)
        if masks is not None:
            source = np.concatenate(
                [source, np.zeros(source.shape[:-1] + (1,), dtype=dtype)],
                axis=-1,
            )
    if np.any(rho_old <= 0):
        raise ValueError("rho_zz_old must be positive")
    source_updated_old = old + step * source / rho_old[np.newaxis, :, :]

    if advance_density:
        target_density = rho_old.copy()
        for cell in range(n_cells):
            for level in range(n_levels):
                horizontal_divergence = dtype.type(0.0)
                for slot in range(int(counts[cell])):
                    edge = int(edges_on_cell[cell, slot])
                    contribution = (
                        signs[cell, slot]
                        * horizontal_velocity[level, edge]
                        * dv_edge[edge]
                    )
                    if source_inverse_area is None:
                        contribution = contribution / area_cell[cell]
                    else:
                        contribution = contribution * source_inverse_area[cell]
                    horizontal_divergence = horizontal_divergence + contribution
                target_density[level, cell] = rho_old[level, cell] + step * (
                    horizontal_divergence
                    - rdnw[level]
                    * (vertical_velocity[level + 1, cell] - vertical_velocity[level, cell])
                )
    else:
        target_density = rho_new.copy()
    if np.any(target_density <= 0):
        raise ValueError("limited transport target density must stay positive")

    high_edge_values = _atmosphere_horizontal_edge_values(
        stage, horizontal_velocity, dv_edge, cells_on_edge, adv, adv3,
        n_adv, adv_cells, horizontal_order, config_coef_3rd_order,
        monotonic_source_flux=source_inverse_area is not None,
    )
    output = stage if in_place else stage.copy()
    zero = dtype.type(0.0)
    one = dtype.type(1.0)
    eps = dtype.type(1.0e-20)

    for scalar in range(n_tracers):
        old_scalar = source_updated_old[scalar]
        stage_scalar = stage[scalar]
        high_vertical = _atmosphere_vertical_flux(
            stage_scalar, vertical_velocity, fnm, fnp, vertical_order,
            config_coef_3rd_order if vertical_order == 3 else 0.0
        )
        minimum = np.empty((n_levels, n_cells), dtype=dtype)
        maximum = np.empty((n_levels, n_cells), dtype=dtype)
        for cell in range(n_cells):
            for level in range(n_levels):
                lower = max(0, level - 1)
                upper = min(n_levels, level + 2)
                minimum[level, cell] = np.min(old_scalar[lower:upper, cell])
                maximum[level, cell] = np.max(old_scalar[lower:upper, cell])
            for slot in range(int(counts[cell])):
                neighbor = int(cells_on_cell[cell, slot])
                for level in range(n_levels):
                    minimum[level, cell] = min(minimum[level, cell], old_scalar[level, neighbor])
                    maximum[level, cell] = max(maximum[level, cell], old_scalar[level, neighbor])

        # Low-order vertical update and time-scaled antidiffusive flux.
        mass = old_scalar * rho_old
        vertical_residual = step * high_vertical
        # Regional gathers at ring-7 edges read the scale factors of the
        # garbage cell; the native scratch is allocated (nCells+1) with the
        # garbage column zeroed (atm_srk3:2985-2999), reproduced by the pad.
        scale_columns = n_cells if mask_cell_array is None else n_cells + 1
        scale_in = np.zeros((n_levels, scale_columns), dtype=dtype)
        scale_out = np.zeros((n_levels, scale_columns), dtype=dtype)
        # v8.4.x restructured this update around flux_upwind_arr: every level
        # first loses its upper interface's upwind flux, then gains its lower
        # one (mpas_atm_time_integration.F v8.4.2, the two k loops after
        # flux_upwind_arr).  v8.2.3 walks interfaces and so adds before it
        # subtracts.  The stored-inverse selector is the v8.4.1 source lane.
        v841_source_order = inv_area_cell is not None
        for cell in range(n_cells):
            if v841_source_order:
                flux_column = [zero] * (n_levels + 1)
                for interface in range(1, n_levels):
                    velocity = vertical_velocity[interface, cell]
                    flux_column[interface] = step * (
                        max(zero, velocity) * old_scalar[interface - 1, cell]
                        + min(zero, velocity) * old_scalar[interface, cell]
                    )
                for level in range(n_levels - 1):
                    mass[level, cell] = mass[level, cell] - flux_column[level + 1] * rdnw[level]
                for level in range(1, n_levels):
                    mass[level, cell] = mass[level, cell] + flux_column[level] * rdnw[level]
                    vertical_residual[level, cell] = vertical_residual[level, cell] - flux_column[level]
            else:
                for interface in range(1, n_levels):
                    velocity = vertical_velocity[interface, cell]
                    flux_upwind = step * (
                        max(zero, velocity) * old_scalar[interface - 1, cell]
                        + min(zero, velocity) * old_scalar[interface, cell]
                    )
                    mass[interface - 1, cell] = mass[interface - 1, cell] - flux_upwind * rdnw[interface - 1]
                    mass[interface, cell] = mass[interface, cell] + flux_upwind * rdnw[interface]
                    vertical_residual[interface, cell] = vertical_residual[interface, cell] - flux_upwind
            for level in range(n_levels):
                scale_in[level, cell] = -rdnw[level] * (
                    min(zero, vertical_residual[level + 1, cell])
                    - max(zero, vertical_residual[level, cell])
                )
                scale_out[level, cell] = -rdnw[level] * (
                    max(zero, vertical_residual[level + 1, cell])
                    - min(zero, vertical_residual[level, cell])
                )

        high_flux = (
            high_edge_values[scalar]
            if source_inverse_area is not None
            else high_edge_values[scalar] * horizontal_velocity
        )
        upwind_flux = np.zeros((n_levels, n_edges), dtype=dtype)
        horizontal_residual = np.zeros((n_levels, n_edges), dtype=dtype)
        for edge in range(n_edges):
            cell1, cell2 = cells_on_edge[edge]
            for level in range(n_levels):
                velocity = horizontal_velocity[level, edge]
                upwind_flux[level, edge] = dv_edge[edge] * step * (
                    max(zero, velocity) * old_scalar[level, cell1]
                    + min(zero, velocity) * old_scalar[level, cell2]
                )
                horizontal_residual[level, edge] = step * high_flux[level, edge] - upwind_flux[level, edge]
            if mask_edge_array is not None:
                edge_mask = int(mask_edge_array[edge])
                # F:5541/F:5654 verbatim, Fortran precedence REPLICATED:
                # (config_apply_lbcs .and. mask==nRelaxZone) .or.
                # (mask==nRelaxZone-1) -- the mask-4 half fires regardless
                # of config_apply_lbcs, exactly as compiled.
                if (config_apply_lbcs and edge_mask == _N_RELAX_ZONE) or (
                    edge_mask == _N_RELAX_ZONE - 1
                ):
                    for level in range(n_levels):
                        horizontal_residual[level, edge] = zero

        for cell in range(n_cells):
            inverse_area = (
                one / area_cell[cell]
                if source_inverse_area is None
                else source_inverse_area[cell]
            )
            for slot in range(int(counts[cell])):
                edge = int(edges_on_cell[cell, slot])
                for level in range(n_levels):
                    mass[level, cell] = mass[level, cell] + (
                        signs[cell, slot] * upwind_flux[level, edge] * inverse_area
                    )
                    signed_residual = divergence_signs[cell, slot] * horizontal_residual[level, edge]
                    scale_out[level, cell] = scale_out[level, cell] - max(zero, signed_residual) * inverse_area
                    scale_in[level, cell] = scale_in[level, cell] - min(zero, signed_residual) * inverse_area

        for cell in range(n_cells):
            for level in range(n_levels):
                scale_factor = (
                    maximum[level, cell] * target_density[level, cell] - mass[level, cell]
                ) / (scale_in[level, cell] + eps)
                scale_in[level, cell] = min(one, max(zero, scale_factor))
                scale_factor = (
                    minimum[level, cell] * target_density[level, cell] - mass[level, cell]
                ) / (scale_out[level, cell] - eps)
                scale_out[level, cell] = min(one, max(zero, scale_factor))

        for edge in range(n_edges):
            cell1, cell2 = cells_on_edge[edge]
            for level in range(n_levels):
                flux = horizontal_residual[level, edge]
                horizontal_residual[level, edge] = (
                    max(zero, flux) * min(scale_out[level, cell1], scale_in[level, cell2])
                    + min(zero, flux) * min(scale_in[level, cell1], scale_out[level, cell2])
                )
        for cell in range(n_cells):
            for interface in range(1, n_levels):
                flux = vertical_residual[interface, cell]
                vertical_residual[interface, cell] = (
                    max(zero, flux)
                    * min(scale_out[interface - 1, cell], scale_in[interface, cell])
                    + min(zero, flux)
                    * min(scale_out[interface, cell], scale_in[interface - 1, cell])
                )

        for cell in range(n_cells):
            inverse_area = (
                one / area_cell[cell]
                if source_inverse_area is None
                else source_inverse_area[cell]
            )
            for slot in range(int(counts[cell])):
                edge = int(edges_on_cell[cell, slot])
                for level in range(n_levels):
                    mass[level, cell] = mass[level, cell] + (
                        signs[cell, slot] * horizontal_residual[level, edge] * inverse_area
                    )
            for level in range(n_levels):
                mass[level, cell] = mass[level, cell] - rdnw[level] * (
                    vertical_residual[level + 1, cell] - vertical_residual[level, cell]
                )
                output[scalar, level, cell] = max(
                    zero, mass[level, cell] / target_density[level, cell]
                )
    if masks is not None:
        assert mask_cell_array is not None
        # F:5771 REPLICATED, not fixed: the copy-back admits only
        # bdyMaskCell <= nSpecZone (rings 0-2).  Relaxation rings 3-5 and
        # the specified zone keep their pre-transport values here; rings
        # 2-5 are nudged by atm_bdy_adjust_scalars and rings 6-7 are
        # overwritten by atm_bdy_set_scalars afterward.
        output = output[:, :, :n_cells]
        kept = np.flatnonzero(np.asarray(mask_cell_array) > _N_SPEC_ZONE)
        output[:, :, kept] = unpadded_stage[:, :, kept]
        target_density = target_density[:, :n_cells]
    return ScalarTransportResult(_restore_tracers(output, squeezed), target_density.copy())


def advance_scalar_transport(
    mesh: Any,
    scalar_old: ArrayLike,
    scalar_stage: ArrayLike,
    rho_zz_old: ArrayLike,
    rho_zz_new: ArrayLike,
    uh_avg: ArrayLike,
    ww_avg: ArrayLike,
    dt: float,
    *,
    rk_step: int = 3,
    config_scalar_advection: bool = True,
    config_monotonic: bool = True,
    config_positive_definite: bool = False,
    config_split_dynamics_transport: bool = False,
    config_time_integration_order: int = 3,
    **kwargs: Any,
) -> ScalarTransportResult:
    """Dispatch exactly as ``advance_scalars`` lines 1499 and 1544-1577."""

    stage, squeezed = _as_tracers(scalar_stage, "scalar_stage")
    rho_new = _float_array(rho_zz_new, "rho_zz_new", stage.dtype)
    if config_time_integration_order not in (2, 3):
        raise ConfigurationRefusal(
            "config_time_integration_order",
            config_time_integration_order,
            "the frozen scalar transport dispatcher admits RK2 or RK3",
            "config_time_integration_order=3",
        )
    if not config_scalar_advection:
        return ScalarTransportResult(_restore_tracers(stage.copy(), squeezed), rho_new.copy())
    if rk_step >= 3 and (config_monotonic or config_positive_definite):
        return advance_scalars_monotonic(
            mesh,
            scalar_old,
            scalar_stage,
            rho_zz_old,
            rho_zz_new,
            uh_avg,
            ww_avg,
            dt,
            advance_density=config_split_dynamics_transport,
            **kwargs,
        )
    return advance_scalars(
        mesh,
        scalar_old,
        scalar_stage,
        rho_zz_old,
        rho_zz_new,
        uh_avg,
        ww_avg,
        dt,
        rk_step=rk_step,
        config_time_integration_order=config_time_integration_order,
        advance_density=config_split_dynamics_transport,
        **kwargs,
    )


# Readable aliases plus the exact work-routine names used by the frozen source.
compute_advection_coefficients = build_advection_coefficients
standard_tracer_tendency = standard_tracer_advection_tendency
monotonic_tracer_tendency = monotonic_tracer_advection_tendency
vflux3 = vertical_flux_3
vflux4 = vertical_flux_4
atm_advance_scalars_work = advance_scalars
atm_advance_scalars_mono_work = advance_scalars_monotonic


def _map_prep_rows(n_rows, function):
    """Bounded NumPy row workers; results and errors remain in slice order."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, int(os.environ.get("WOOF_HEX_HOSTPREP_THREADS", "4")))
    workers = min(workers, max(1, n_rows // 16384))
    if workers == 1:
        return [function(0, n_rows)]
    error_mode = np.geterr()
    def run(index):
        with np.errstate(**error_mode):
            return function(n_rows * index // workers, n_rows * (index + 1) // workers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-advection-prep") as pool:
        return list(pool.map(run, range(workers)))


def _native_advection_order(counts, neighbors, owners, adv, adv3, n_adv, cells, *, garbage):
    def rows(lo, hi):
        return _native_advection_order_rows(counts, neighbors, owners[lo:hi],
            adv[lo:hi], adv3[lo:hi], n_adv[lo:hi], cells[lo:hi], garbage=garbage)
    try:
        chunks = _map_prep_rows(cells.shape[0], rows)
    except ValueError:
        # A malformed stencil can fail at different stages in two slices.
        # Replay only this refusal path to retain the original stage priority.
        return rows(0, cells.shape[0])
    return tuple(np.concatenate(parts, axis=0) for parts in zip(*chunks)) if len(chunks) > 1 else chunks[0]


def _native_advection_order_rows(counts, neighbors, owners, adv, adv3, n_adv, cells, *, garbage):
    """Vectorize native insertion order over edges, retaining every coefficient."""
    ne, width = cells.shape
    active_edge = np.any((owners >= 0) & (owners < counts.size), axis=1)
    ordered = cells.copy()
    ordered[active_edge] = -1
    filled = n_adv.copy()
    rows = np.arange(ne)
    ordered[active_edge, :2] = owners[active_edge]
    filled[active_edge] = 2
    for side in (0, 1):
        owner = owners[:, side]
        valid_owner = (owner >= 0) & (owner < counts.size)
        safe_owner = np.where(valid_owner, owner, 0)
        owner_count = np.where(valid_owner, counts[safe_owner], 0)
        for slot in range(neighbors.shape[1]):
            candidate = neighbors[safe_owner, slot]
            add = active_edge & (slot < owner_count) & ~np.any(ordered == candidate[:, None], axis=1)
            if np.any(filled[add] >= width):
                raise ValueError("native advection stencil exceeds coefficient width")
            ordered[rows[add], filled[add]] = candidate[add]
            filled[add] += 1
    if not np.array_equal(filled, n_adv):
        raise ValueError("native and bridge advection stencil counts differ")
    permutation = np.zeros_like(cells)
    for slot in range(width):
        active = active_edge & (slot < n_adv)
        matches = cells == ordered[:, slot, None]
        if np.any(active & ~np.any(matches, axis=1)):
            raise ValueError("native advection cell is absent from bridge stencil")
        permutation[:, slot] = np.argmax(matches, axis=1)
    # Inactive slots remain zeros and garbage IDs, as returned by the bridge.
    active = active_edge[:, None] & (np.arange(width)[None, :] < n_adv[:, None])
    return (
        np.where(active, np.take_along_axis(adv, permutation, axis=1), adv),
        np.where(active, np.take_along_axis(adv3, permutation, axis=1), adv3),
        np.where(active, ordered, cells),
    )


def _compress_advection_coefficients_serial(counts,cells_on_cell,cells_on_edge,ids,deriv,dc_edge,dv_edge,*,source_order_v841,allow_regional_sentinels):
    """Unchanged independent scalar authority for exactness checks."""
    n_cells=counts.size; n_edges=cells_on_edge.shape[0]; width=deriv.shape[2]; dtype=deriv.dtype
    adv = np.zeros((n_edges, width), dtype=dtype)
    adv3 = np.zeros((n_edges, width), dtype=dtype)
    adv_cells = np.full((n_edges, width), -1, dtype=np.int64)
    n_adv = np.zeros(n_edges, dtype=np.int64)
    half = dtype.type(0.5)
    twelve = dtype.type(12.0)

    garbage = n_cells if allow_regional_sentinels else -1

    def _cell_count(cell: int) -> int:
        # nEdgesOnCell of the garbage element is 0 by pool allocation.
        return 0 if cell == garbage else int(counts[cell])

    def _cell_id(cell: int) -> int:
        # The garbage element orders last: culled meshes carry contiguous
        # 1..N indexToCellID, so any value beyond them is stable.
        return n_cells + 1 if cell == garbage else int(ids[cell])

    for edge in range(n_edges):
        cell1 = int(cells_on_edge[edge, 0])
        cell2 = int(cells_on_edge[edge, 1])
        unique = {cell1, cell2}
        for slot in range(_cell_count(cell1)):
            unique.add(int(cells_on_cell[cell1, slot]))
        for slot in range(_cell_count(cell2)):
            unique.add(int(cells_on_cell[cell2, slot]))
        if source_order_v841:
            ordered = [cell1, cell2]
            for cell in (cell1, cell2):
                for slot in range(_cell_count(cell)):
                    neighbor = int(cells_on_cell[cell, slot])
                    if neighbor not in ordered:
                        ordered.append(neighbor)
        else:
            ordered = sorted(unique, key=_cell_id)
        if len(ordered) > width:
            raise ValueError(
                f"edge {edge} needs {len(ordered)} advection cells but deriv_two has width {width}"
            )
        n_adv[edge] = len(ordered)
        adv_cells[edge, : len(ordered)] = ordered
        position = {cell: slot for slot, cell in enumerate(ordered)}

        target = position[cell1]
        adv[edge, target] = adv[edge, target] + deriv[edge, 0, 0]
        adv3[edge, target] = adv3[edge, target] + deriv[edge, 0, 0]
        for neighbor_slot in range(_cell_count(cell1)):
            target = position[int(cells_on_cell[cell1, neighbor_slot])]
            value = deriv[edge, 0, neighbor_slot + 1]
            adv[edge, target] = adv[edge, target] + value
            adv3[edge, target] = adv3[edge, target] + value

        target = position[cell2]
        adv[edge, target] = adv[edge, target] + deriv[edge, 1, 0]
        adv3[edge, target] = adv3[edge, target] - deriv[edge, 1, 0]
        for neighbor_slot in range(_cell_count(cell2)):
            target = position[int(cells_on_cell[cell2, neighbor_slot])]
            value = deriv[edge, 1, neighbor_slot + 1]
            adv[edge, target] = adv[edge, target] + value
            adv3[edge, target] = adv3[edge, target] - value

        if source_order_v841:
            dc_squared = dc_edge[edge] * dc_edge[edge]
            for slot in range(len(ordered)):
                adv[edge, slot] = (
                    -dc_squared * adv[edge, slot] / twelve
                )
                adv3[edge, slot] = (
                    -dc_squared * adv3[edge, slot] / twelve
                )
        else:
            scale = -(dc_edge[edge] * dc_edge[edge]) / twelve
            for slot in range(len(ordered)):
                adv[edge, slot] = scale * adv[edge, slot]
                adv3[edge, slot] = scale * adv3[edge, slot]
        adv[edge, position[cell1]] = adv[edge, position[cell1]] + half
        adv[edge, position[cell2]] = adv[edge, position[cell2]] + half
        for slot in range(len(ordered)):
            adv[edge, slot] = dv_edge[edge] * adv[edge, slot]
            adv3[edge, slot] = dv_edge[edge] * adv3[edge, slot]

    return adv,adv3,n_adv,adv_cells



def _native_advection_compression(counts, neighbors, owners, deriv, dc, dv, n_adv, cells):
    def rows(lo, hi):
        return _native_advection_compression_rows(counts, neighbors, owners[lo:hi],
            deriv[lo:hi], dc[lo:hi], dv[lo:hi], n_adv[lo:hi], cells[lo:hi])
    chunks = _map_prep_rows(cells.shape[0], rows)
    return tuple(np.concatenate(parts, axis=0) for parts in zip(*chunks)) if len(chunks) > 1 else chunks[0]


def _native_advection_compression_rows(counts, neighbors, owners, deriv, dc, dv, n_adv, cells):
    """Match core.F:1475-1529 across edges, retaining native rounding points."""
    ne, width = cells.shape
    rows = np.arange(ne)
    dtype = deriv.dtype
    adv = np.zeros((ne, width), dtype=dtype)
    adv3 = np.zeros_like(adv)
    active_edge = np.any((owners >= 0) & (owners < counts.size), axis=1)
    for side in (0, 1):
        owner = owners[:, side]
        valid = (owner >= 0) & (owner < counts.size)
        safe = np.where(valid, owner, 0)
        count = np.where(valid, counts[safe], 0)
        for slot in range(neighbors.shape[1] + 1):
            cell = owner if slot == 0 else neighbors[safe, slot - 1]
            used = active_edge if slot == 0 else active_edge & (slot <= count)
            position = np.argmax(cells == cell[:, None], axis=1)
            rr, pp = rows[used], position[used]
            value = deriv[used, side, slot]
            adv[rr, pp] = adv[rr, pp] + value
            adv3[rr, pp] = adv3[rr, pp] + value if side == 0 else adv3[rr, pp] - value
    active = active_edge[:, None] & (np.arange(width)[None, :] < n_adv[:, None])
    # Integer exponent two in native is one rounded multiply, not a library power.
    dc_squared = dc * dc
    adv = np.where(active, (-dc_squared[:, None] * adv) / dtype.type(12.0), dtype.type(0.0))
    adv3 = np.where(active, (-dc_squared[:, None] * adv3) / dtype.type(12.0), dtype.type(0.0))
    for side in (0, 1):
        position = np.argmax(cells == owners[:, side, None], axis=1)
        rr, pp = rows[active_edge], position[active_edge]
        adv[rr, pp] = adv[rr, pp] + dtype.type(0.5)
    return adv * dv[:, None], adv3 * dv[:, None]
