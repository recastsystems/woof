"""C-grid diagnostics used by the MPAS atmospheric tendency calculation.

Direct authority for frozen
``src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5491-5799``.
The Hollingsworth kinetic-energy construction is enabled exactly as the source's
compile-time ``hollingsworth=.true.`` setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .acoustic import edge_signs_on_cells
from .errors import ConfigurationRefusal

FloatArray = NDArray[np.floating[Any]]


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


def _source_inverse(
    value: FloatArray,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> FloatArray:
    inverse = np.asarray(value)
    if inverse.dtype != dtype:
        raise TypeError(f"{name} dtype {inverse.dtype} != RKIND {dtype}")
    if inverse.shape != shape:
        raise ValueError(f"{name} shape {inverse.shape} != {shape}")
    if not np.all(np.isfinite(inverse)):
        raise ValueError(f"{name} must be finite")
    return inverse


def edge_signs_on_vertices(mesh: object) -> FloatArray:
    """Transcribe ``mpas_atm_core.F:1172-1184`` for 0-based connectivity."""
    edges = _mesh_array(mesh, "edgesOnVertex").astype(np.int64, copy=False)
    vertices = _mesh_array(mesh, "verticesOnEdge").astype(np.int64, copy=False)
    # Whole-array form of the per-vertex, per-slot loop (the same 1.0 / -1.0 / 0.0 words). THE COST THIS
    # REMOVES (RRFS hrrrv5 mv3km, V6000, 2026-10-28): the Python loop ran ~0.5 s per dycore build for 413k
    # vertices on one core (about 5 s on NOAA's conus3km).  An edge index outside verticesOnEdge raises
    # IndexError, as the loop did.
    active = edges >= 0
    upper = vertices[np.where(active, edges, 0), 1]
    owner = np.arange(edges.shape[0], dtype=np.int64)[:, None]
    return np.where(active, np.where(upper == owner, 1.0, -1.0), 0.0).astype(np.float64, copy=False)


def kite_indices_for_cells(mesh: object) -> NDArray[np.int64]:
    """Transcribe ``kiteForCell`` construction at ``mpas_atm_core.F:1204-1213``."""
    counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    vertices_on_cell = _mesh_array(mesh, "verticesOnCell").astype(np.int64, copy=False)
    cells_on_vertex = _mesh_array(mesh, "cellsOnVertex").astype(np.int64, copy=False)
    result = np.full(vertices_on_cell.shape, -1, dtype=np.int64)
    for cell in range(counts.size):
        for slot in range(int(counts[cell])):
            vertex = int(vertices_on_cell[cell, slot])
            matches = np.flatnonzero(cells_on_vertex[vertex] == cell)
            if matches.size != 1:
                raise ValueError(
                    f"cell {cell} occurs {matches.size} times at vertex {vertex}; expected once"
                )
            result[cell, slot] = int(matches[0])
    return result


@dataclass(frozen=True, slots=True)
class SolveDiagnostics:
    h_edge: FloatArray
    tangential_velocity: FloatArray
    vorticity: FloatArray
    divergence: FloatArray
    kinetic_energy: FloatArray
    pv_edge: FloatArray
    pv_vertex: FloatArray
    pv_cell: FloatArray
    grad_pv_normal: FloatArray
    grad_pv_tangential: FloatArray


def compute_solve_diagnostics(
    mesh: object,
    normal_velocity: FloatArray,
    layer_thickness: FloatArray,
    *,
    dt: float,
    apvm_upwinding: float = 0.0,
    rk_step: int | None = None,
    cached_tangential_velocity: FloatArray | None = None,
    f_vertex: FloatArray | None = None,
    inv_area_cell: FloatArray | None = None,
    inv_area_triangle: FloatArray | None = None,
    inv_dc_edge: FloatArray | None = None,
    inv_dv_edge: FloatArray | None = None,
) -> SolveDiagnostics:
    """Compute MPAS vorticity, divergence, KE, tangential velocity, and PV.

    ``normal_velocity`` and ``layer_thickness`` have shapes ``(level, edge)``
    and ``(level, cell)`` respectively.  When an RK step other than 3 is
    supplied, frozen MPAS reuses its cached tangential velocity; this API
    refuses if the caller omits that state instead of silently recomputing it.
    """
    u = np.asarray(normal_velocity)
    h = np.asarray(layer_thickness)
    if u.ndim != 2 or h.ndim != 2 or u.dtype.kind != "f":
        raise ValueError("normal_velocity and layer_thickness must be floating 2-D arrays")
    nlev, nedges = u.shape
    ncells = h.shape[1]
    if h.shape[0] != nlev:
        raise ValueError("velocity and thickness vertical dimensions differ")
    nvertices = _mesh_array(mesh, "areaTriangle").size
    if _mesh_array(mesh, "areaCell").size != ncells or _mesh_array(mesh, "dcEdge").size != nedges:
        raise ValueError("field entity dimensions do not match mesh")
    dtype = u.dtype

    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    vertices_on_edge = _mesh_array(mesh, "verticesOnEdge").astype(np.int64, copy=False)
    edges_on_vertex = _mesh_array(mesh, "edgesOnVertex").astype(np.int64, copy=False)
    edges_on_cell = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    edges_on_edge = _mesh_array(mesh, "edgesOnEdge").astype(np.int64, copy=False)
    vertices_on_cell = _mesh_array(mesh, "verticesOnCell").astype(np.int64, copy=False)
    counts_cell = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    counts_edge = _mesh_array(mesh, "nEdgesOnEdge").astype(np.int64, copy=False)
    dc_edge = _mesh_array(mesh, "dcEdge").astype(dtype, copy=False)
    dv_edge = _mesh_array(mesh, "dvEdge").astype(dtype, copy=False)
    area_cell = _mesh_array(mesh, "areaCell").astype(dtype, copy=False)
    area_triangle = _mesh_array(mesh, "areaTriangle").astype(dtype, copy=False)
    source_inv_area_cell = None
    if inv_area_cell is not None:
        source_inv_area_cell = _source_inverse(
            inv_area_cell, "inv_area_cell", dtype, area_cell.shape
        )
    source_inv_area_triangle = None
    if inv_area_triangle is not None:
        source_inv_area_triangle = _source_inverse(
            inv_area_triangle,
            "inv_area_triangle",
            dtype,
            area_triangle.shape,
        )
    source_inv_dc_edge = None
    if inv_dc_edge is not None:
        source_inv_dc_edge = _source_inverse(
            inv_dc_edge, "inv_dc_edge", dtype, dc_edge.shape
        )
    source_inv_dv_edge = None
    if inv_dv_edge is not None:
        source_inv_dv_edge = _source_inverse(
            inv_dv_edge, "inv_dv_edge", dtype, dv_edge.shape
        )
    weights = _mesh_array(mesh, "weightsOnEdge").astype(dtype, copy=False)
    kite_areas = _mesh_array(mesh, "kiteAreasOnVertex").astype(dtype, copy=False)
    cell_signs = edge_signs_on_cells(mesh).astype(dtype, copy=False)
    vertex_signs = edge_signs_on_vertices(mesh).astype(dtype, copy=False)
    kite_for_cell = kite_indices_for_cells(mesh)

    h_edge = dtype.type(0.5) * (h[:, cells_on_edge[:, 0]] + h[:, cells_on_edge[:, 1]])
    ke_edge = dc_edge[None, :] * dv_edge[None, :] * u**2

    # Independent entities are batched; each neighbor slot is accumulated in
    # the original order, including the literal three-edge Hollingsworth sum.
    vorticity = np.zeros((nlev, nvertices), dtype=dtype)
    for slot in range(edges_on_vertex.shape[1]):
        vertex = np.flatnonzero(edges_on_vertex[:, slot] >= 0)
        edge = edges_on_vertex[vertex, slot]
        vorticity[:, vertex] += vertex_signs[vertex, slot] * dc_edge[edge] * u[:, edge]
    if source_inv_area_triangle is None:
        vorticity /= area_triangle
    else:
        vorticity *= source_inv_area_triangle

    divergence = np.zeros((nlev, ncells), dtype=dtype)
    kinetic_energy = np.zeros((nlev, ncells), dtype=dtype)
    for slot in range(edges_on_cell.shape[1]):
        cell = np.flatnonzero(counts_cell > slot)
        edge = edges_on_cell[cell, slot]
        divergence[:, cell] += cell_signs[cell, slot] * dv_edge[edge] * u[:, edge]
        kinetic_energy[:, cell] += dtype.type(0.25) * ke_edge[:, edge]
    if source_inv_area_cell is None:
        divergence /= area_cell
        kinetic_energy /= area_cell
    else:
        divergence *= source_inv_area_cell
        kinetic_energy *= source_inv_area_cell

    ke_vertex = np.zeros((nlev, nvertices), dtype=dtype)
    if source_inv_area_triangle is None:
        for vertex in range(nvertices):
            valid_edges = edges_on_vertex[vertex]
            valid_edges = valid_edges[valid_edges >= 0]
            ke_vertex[:, vertex] = (
                dtype.type(0.25) * np.sum(ke_edge[:, valid_edges], axis=1)
                / area_triangle[vertex]
            )
    else:
        if np.any(np.sum(edges_on_vertex >= 0, axis=1) != 3):
            raise ValueError("v8.4.1 Hollingsworth KE requires exactly three edges per vertex")
        edge0, edge1, edge2 = edges_on_vertex[edges_on_vertex >= 0].reshape(nvertices, 3).T
        r = dtype.type(0.25) * source_inv_area_triangle
        ke_vertex = (ke_edge[:, edge0] + ke_edge[:, edge1] + ke_edge[:, edge2]) * r
    ke_fact = dtype.type(0.625)
    kinetic_energy *= ke_fact
    for slot in range(edges_on_cell.shape[1]):
        cell = np.flatnonzero(counts_cell > slot)
        vertex = vertices_on_cell[cell, slot]
        kite_slot = kite_for_cell[cell, slot]
        correction = ((dtype.type(1.0) - ke_fact) * kite_areas[vertex, kite_slot]
                      * ke_vertex[:, vertex])
        if source_inv_area_cell is None:
            kinetic_energy[:, cell] += correction / area_cell[cell]
        else:
            kinetic_energy[:, cell] += correction * source_inv_area_cell[cell]

    reconstruct_v = rk_step is None or rk_step == 3
    if reconstruct_v:
        tangential = np.zeros_like(u)
        for slot in range(edges_on_edge.shape[1]):
            edge = np.flatnonzero(counts_edge > slot)
            neighbor_edge = edges_on_edge[edge, slot]
            tangential[:, edge] += weights[edge, slot] * u[:, neighbor_edge]
    else:
        if cached_tangential_velocity is None:
            raise ConfigurationRefusal(
                "tangential_velocity_cache",
                None,
                f"rk_step={rk_step} reuses v from the previous reconstruction",
                "cached_tangential_velocity=<previous v> or rk_step=3",
            )
        tangential = np.asarray(cached_tangential_velocity).copy()
        if tangential.shape != u.shape:
            raise ValueError(f"cached tangential velocity shape {tangential.shape} != {u.shape}")

    if f_vertex is None:
        try:
            f_vertex = _mesh_array(mesh, "fVertex")
        except AttributeError:
            f_vertex = np.zeros(nvertices, dtype=dtype)
    f_vertex = np.asarray(f_vertex, dtype=dtype)
    if f_vertex.shape != (nvertices,):
        raise ValueError(f"f_vertex shape {f_vertex.shape} != ({nvertices},)")
    pv_vertex = f_vertex[None, :] + vorticity
    pv_edge = dtype.type(0.5) * (
        pv_vertex[:, vertices_on_edge[:, 0]] + pv_vertex[:, vertices_on_edge[:, 1]]
    )
    pv_cell = np.zeros((nlev, ncells), dtype=dtype)
    grad_pv_normal = np.zeros_like(u)
    grad_pv_tangential = np.zeros_like(u)

    if apvm_upwinding > 0.0:
        for slot in range(edges_on_cell.shape[1]):
            cell = np.flatnonzero(counts_cell > slot)
            vertex = vertices_on_cell[cell, slot]
            kite_slot = kite_for_cell[cell, slot]
            contribution = kite_areas[vertex, kite_slot] * pv_vertex[:, vertex]
            if source_inv_area_cell is None:
                pv_cell[:, cell] += contribution / area_cell[cell]
            else:
                pv_cell[:, cell] += contribution * source_inv_area_cell[cell]
        scale = dtype.type(apvm_upwinding) * dtype.type(dt)
        vertex0, vertex1 = vertices_on_edge.T
        cell0, cell1 = cells_on_edge.T
        tangential_difference = pv_vertex[:, vertex1] - pv_vertex[:, vertex0]
        normal_difference = pv_cell[:, cell1] - pv_cell[:, cell0]
        if source_inv_dv_edge is None:
            grad_pv_tangential[:] = tangential_difference / dv_edge
        else:
            grad_pv_tangential[:] = tangential_difference * source_inv_dv_edge
        if source_inv_dc_edge is None:
            grad_pv_normal[:] = normal_difference / dc_edge
        else:
            grad_pv_normal[:] = normal_difference * source_inv_dc_edge
        pv_edge -= scale * (tangential * grad_pv_tangential + u * grad_pv_normal)
    elif apvm_upwinding < 0.0:
        raise ConfigurationRefusal(
            "config_apvm_upwinding",
            apvm_upwinding,
            "the frozen source only activates APVM for positive coefficients",
            "config_apvm_upwinding=0.0",
        )

    return SolveDiagnostics(
        h_edge=h_edge,
        tangential_velocity=tangential,
        vorticity=vorticity,
        divergence=divergence,
        kinetic_energy=kinetic_energy,
        pv_edge=pv_edge,
        pv_vertex=pv_vertex,
        pv_cell=pv_cell,
        grad_pv_normal=grad_pv_normal,
        grad_pv_tangential=grad_pv_tangential,
    )
