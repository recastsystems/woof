"""Frozen MPAS-A terrain metrics and physical/coupled state conversion.

This module is the scalar CPU authority for the terrain-following pieces that
connect public MPAS state to the dry dycore's coupled variables.  It transcribes
the frozen MPAS-A v8.2.3 source at:

* ``mpas_atm_core.F:1137-1203`` (``edgesOnCell_sign``, ``zb_cell`` and
  ``zb3_cell`` construction);
* ``mpas_atm_core.F:1419-1437`` (``config_coef_3rd_order`` coupling);
* ``mpas_atm_time_integration.F:5887-5930`` (physical state to
  ``rho_zz``, ``theta_m``, ``ru`` and ``rw``);
* ``mpas_atm_time_integration.F:2793-2903`` (``ru``/``rw`` recovery);
* ``mpas_atm_time_integration.F:2089-2112`` (vertical-velocity tendency
  conversion); and
* ``mpas_atm_core.F:887-936`` (public ``rho``/``theta`` diagnostics).

Logical floating fields use Fortran's logical order ``(level, entity)``.
``array_layout="netcdf"`` accepts the native on-disk MPAS order, with an
optional leading Time dimension, and native one-based connectivity.  Results
are always returned in logical order.  This convention is an authority API,
not a GPU memory-layout decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


ArrayLayout = Literal["logical", "netcdf"]
FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


@dataclass(frozen=True, slots=True)
class TerrainCoupling:
    """Canonical topology and cell-oriented terrain metrics.

    Connectivity is zero-based with ``-1`` padding.  The metric arrays have
    shape ``(nVertLevels + 1, nCells, maxEdges)``.  ``zb3_cell`` has already
    been multiplied by ``config_coef_3rd_order``, exactly where the frozen core
    performs that coupling.
    """

    n_edges_on_cell: IntArray
    edges_on_cell: IntArray
    cells_on_edge: IntArray
    edges_on_cell_sign: FloatArray
    zb_cell: FloatArray
    zb3_cell: FloatArray
    config_coef_3rd_order: float

    @property
    def n_cells(self) -> int:
        return int(self.edges_on_cell.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.cells_on_edge.shape[0])

    @property
    def n_vert_levels(self) -> int:
        return int(self.zb_cell.shape[0] - 1)


@dataclass(frozen=True, slots=True)
class CoupledState:
    """Dry-dycore coupled variables, all in logical Fortran order."""

    rho_zz: FloatArray
    theta_m: FloatArray
    ru: FloatArray
    rw: FloatArray


@dataclass(frozen=True, slots=True)
class RecoveredVelocity:
    """Public normal and physical vertical velocity in logical order."""

    u: FloatArray
    w: FloatArray


@dataclass(frozen=True, slots=True)
class PhysicalState:
    """Public MPAS ``rho``, ``theta``, ``u`` and ``w`` in logical order."""

    rho: FloatArray
    theta: FloatArray
    u: FloatArray
    w: FloatArray


def _layout(value: str) -> ArrayLayout:
    if value not in ("logical", "netcdf"):
        raise ValueError("array_layout must be 'logical' or 'netcdf'")
    return value  # type: ignore[return-value]


def _float_array(name: str, value: object, *, ndim: int | None = None) -> FloatArray:
    if np.ma.isMaskedArray(value):
        if np.any(np.ma.getmaskarray(value)):
            raise ValueError(f"{name} contains masked values")
        value = np.ma.getdata(value)
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must have dtype float32 or float64, got {array.dtype}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    from .host_parallel import all_finite

    if not all_finite(array):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _integer_array(name: str, value: object, *, ndim: int) -> IntArray:
    if np.ma.isMaskedArray(value):
        if np.any(np.ma.getmaskarray(value)):
            raise ValueError(f"{name} contains masked values")
        value = np.ma.getdata(value)
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must have an integer dtype, got {array.dtype}")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    return array


def _same_dtype(reference_name: str, reference: FloatArray, **arrays: FloatArray) -> None:
    for name, array in arrays.items():
        if array.dtype != reference.dtype:
            raise TypeError(
                f"{name} dtype {array.dtype} differs from {reference_name} dtype "
                f"{reference.dtype}; frozen RKIND fields cannot be mixed"
            )


def _logical_field(
    name: str,
    value: object,
    *,
    layout: ArrayLayout,
    entities: int,
    levels: int,
    time_index: int,
) -> FloatArray:
    array = _float_array(name, value)
    if layout == "logical":
        expected = (levels, entities)
        if array.shape != expected:
            raise ValueError(f"{name} logical shape {array.shape} != {expected}")
        return np.array(array, copy=True, order="K")

    expected = (entities, levels)
    if array.ndim == 3:
        if array.shape[1:] != expected:
            raise ValueError(
                f"{name} native shape {array.shape} does not end in {expected}"
            )
        if not -array.shape[0] <= time_index < array.shape[0]:
            raise IndexError(
                f"time_index {time_index} is outside {name}'s Time dimension "
                f"of length {array.shape[0]}"
            )
        array = array[time_index]
    elif array.shape != expected:
        raise ValueError(f"{name} native shape {array.shape} != {expected}")
    return np.array(array.T, copy=True, order="K")


def _logical_vector(name: str, value: object, *, length: int) -> FloatArray:
    array = _float_array(name, value, ndim=1)
    if array.shape != (length,):
        raise ValueError(f"{name} shape {array.shape} != ({length},)")
    return np.array(array, copy=True)


def build_terrain_coupling(
    *,
    n_edges_on_cell: object,
    edges_on_cell: object,
    cells_on_edge: object,
    zb: object,
    zb3: object,
    config_coef_3rd_order: float,
    array_layout: ArrayLayout = "logical",
    allow_regional_sentinels: bool = False,
) -> TerrainCoupling:
    """Build the frozen cell-oriented slope metrics.

    With ``array_layout="logical"``, connectivity is canonical zero-based and
    ``zb``/``zb3`` have shape ``(nVertLevels+1, 2, nEdges)``.  With
    ``array_layout="netcdf"``, connectivity is native one-based and the metric
    shape is ``(nEdges, 2, nVertLevels+1)``.
    """

    layout = _layout(array_layout)
    counts_source = _integer_array("n_edges_on_cell", n_edges_on_cell, ndim=1)
    eoc_source = _integer_array("edges_on_cell", edges_on_cell, ndim=2)
    coe_source = _integer_array("cells_on_edge", cells_on_edge, ndim=2)
    raw_zb = _float_array("zb", zb, ndim=3)
    raw_zb3 = _float_array("zb3", zb3, ndim=3)
    _same_dtype("zb", raw_zb, zb3=raw_zb3)

    if not np.isfinite(config_coef_3rd_order):
        raise ValueError("config_coef_3rd_order must be finite")
    if eoc_source.shape[0] != counts_source.size:
        raise ValueError("edges_on_cell and n_edges_on_cell nCells differ")
    if coe_source.shape[1] != 2:
        raise ValueError("cells_on_edge must have shape (nEdges, 2)")

    n_cells, max_edges = eoc_source.shape
    n_edges = coe_source.shape[0]
    counts = np.asarray(counts_source, dtype=np.int64).copy()
    if np.any(counts < 0) or np.any(counts > max_edges):
        raise ValueError("n_edges_on_cell entries must be in [0, maxEdges]")

    if layout == "netcdf":
        if raw_zb.shape[:2] != (n_edges, 2):
            raise ValueError(
                f"zb native shape {raw_zb.shape} must start with ({n_edges}, 2)"
            )
        if raw_zb3.shape != raw_zb.shape:
            raise ValueError("zb3 shape differs from zb")
        logical_zb = np.transpose(raw_zb, (2, 1, 0))
        logical_zb3 = np.transpose(raw_zb3, (2, 1, 0))
        eoc = np.asarray(eoc_source, dtype=np.int64) - 1
        coe = np.asarray(coe_source, dtype=np.int64) - 1
    else:
        if raw_zb.shape[1:] != (2, n_edges):
            raise ValueError(
                f"zb logical shape {raw_zb.shape} must end in (2, {n_edges})"
            )
        if raw_zb3.shape != raw_zb.shape:
            raise ValueError("zb3 shape differs from zb")
        logical_zb = raw_zb
        logical_zb3 = raw_zb3
        eoc = np.asarray(eoc_source, dtype=np.int64).copy()
        coe = np.asarray(coe_source, dtype=np.int64).copy()

    if allow_regional_sentinels:
        # Ring-7 one-cell boundary edges store 0 (loaded as a negative
        # sentinel) in one cellsOnEdge slot; native MPAS maps it to the
        # garbage cell nCells+1 at read time.  The side test below still
        # finds the present cell in the other slot, which is all zb_cell
        # coupling needs.
        coe = np.where(coe < 0, n_cells, coe)
        if np.any(coe > n_cells):
            raise ValueError("cells_on_edge contains an out-of-range cell")
    elif np.any(coe < 0) or np.any(coe >= n_cells):
        raise ValueError("cells_on_edge contains a missing or out-of-range cell")
    if np.any(coe[:, 0] == coe[:, 1]):
        raise ValueError("cells_on_edge contains an edge with the same cell twice")

    dtype = raw_zb.dtype
    coefficient = dtype.type(config_coef_3rd_order)
    from .rust_hostprep import terrain_metrics

    canonical_eoc, signs, zb_cell, zb3_cell = terrain_metrics(
        counts=counts, edges=eoc, cells_on_edge=coe, zb=logical_zb,
        zb3=logical_zb3, coefficient=coefficient,
    )

    return TerrainCoupling(
        n_edges_on_cell=counts,
        edges_on_cell=canonical_eoc,
        cells_on_edge=np.asarray(coe, dtype=np.int64).copy(),
        edges_on_cell_sign=signs,
        zb_cell=zb_cell,
        zb3_cell=zb3_cell,
        config_coef_3rd_order=float(config_coef_3rd_order),
    )


def _state_inputs(
    *,
    rho: object,
    theta: object,
    qv: object,
    u: object,
    w: object,
    zz: object,
    fzm: object,
    fzp: object,
    coupling: TerrainCoupling,
    array_layout: ArrayLayout,
    time_index: int,
) -> tuple[FloatArray, ...]:
    layout = _layout(array_layout)
    nlev = coupling.n_vert_levels
    ncells = coupling.n_cells
    nedges = coupling.n_edges
    fields = (
        _logical_field(
            "rho", rho, layout=layout, entities=ncells, levels=nlev, time_index=time_index
        ),
        _logical_field(
            "theta", theta, layout=layout, entities=ncells, levels=nlev, time_index=time_index
        ),
        _logical_field(
            "qv", qv, layout=layout, entities=ncells, levels=nlev, time_index=time_index
        ),
        _logical_field(
            "u", u, layout=layout, entities=nedges, levels=nlev, time_index=time_index
        ),
        _logical_field(
            "w", w, layout=layout, entities=ncells, levels=nlev + 1, time_index=time_index
        ),
        _logical_field(
            "zz", zz, layout=layout, entities=ncells, levels=nlev, time_index=time_index
        ),
        _logical_vector("fzm", fzm, length=nlev),
        _logical_vector("fzp", fzp, length=nlev),
    )
    names = ("theta", "qv", "u", "w", "zz", "fzm", "fzp")
    _same_dtype("rho", fields[0], **dict(zip(names, fields[1:], strict=True)))
    if coupling.zb_cell.dtype != fields[0].dtype:
        raise TypeError(
            f"terrain dtype {coupling.zb_cell.dtype} differs from state dtype {fields[0].dtype}"
        )
    return fields


def initialize_coupled_state(
    *,
    rho: object,
    theta: object,
    qv: object,
    u: object,
    w: object,
    zz: object,
    fzm: object,
    fzp: object,
    coupling: TerrainCoupling,
    array_layout: ArrayLayout = "logical",
    time_index: int = 0,
    water_vapor_gas_constant: float = 461.6,
    dry_air_gas_constant: float = 287.0,
) -> CoupledState:
    """Convert public MPAS fields to the frozen coupled dry-dynamics state."""

    rho_l, theta_l, qv_l, u_l, w_l, zz_l, fzm_l, fzp_l = _state_inputs(
        rho=rho,
        theta=theta,
        qv=qv,
        u=u,
        w=w,
        zz=zz,
        fzm=fzm,
        fzp=fzp,
        coupling=coupling,
        array_layout=array_layout,
        time_index=time_index,
    )
    dtype = rho_l.dtype
    if not np.isfinite(water_vapor_gas_constant) or not np.isfinite(dry_air_gas_constant):
        raise ValueError("gas constants must be finite")
    if dry_air_gas_constant == 0.0:
        raise ValueError("dry_air_gas_constant must be nonzero")
    if np.any(rho_l <= dtype.type(0.0)):
        raise ValueError("rho must be strictly positive")
    if np.any(zz_l <= dtype.type(0.0)):
        raise ValueError("zz must be strictly positive")

    one = dtype.type(1.0)
    half = dtype.type(0.5)
    rvord = dtype.type(water_vapor_gas_constant) / dtype.type(dry_air_gas_constant)
    rho_zz = rho_l / zz_l
    theta_m = theta_l * (one + rvord * qv_l)
    nlev, ncells = rho_zz.shape
    nedges = coupling.n_edges

    ru = np.empty((nlev, nedges), dtype=dtype)
    for edge in range(nedges):
        cell1 = int(coupling.cells_on_edge[edge, 0])
        cell2 = int(coupling.cells_on_edge[edge, 1])
        for level in range(nlev):
            ru[level, edge] = (
                half
                * u_l[level, edge]
                * (rho_zz[level, cell1] + rho_zz[level, cell2])
            )

    # Frozen initialization explicitly fixes both omega boundaries to zero.
    rw = np.zeros((nlev + 1, ncells), dtype=dtype)
    for cell in range(ncells):
        for level in range(1, nlev):
            interp_rho = (
                fzp_l[level] * rho_zz[level - 1, cell]
                + fzm_l[level] * rho_zz[level, cell]
            )
            interp_zz = (
                fzp_l[level] * zz_l[level - 1, cell]
                + fzm_l[level] * zz_l[level, cell]
            )
            rw[level, cell] = w_l[level, cell] * interp_rho * interp_zz

    for cell in range(ncells):
        for slot in range(int(coupling.n_edges_on_cell[cell])):
            edge = int(coupling.edges_on_cell[cell, slot])
            cell_sign = coupling.edges_on_cell_sign[cell, slot]
            for level in range(1, nlev):
                flux = (
                    fzm_l[level] * ru[level, edge]
                    + fzp_l[level] * ru[level - 1, edge]
                )
                interp_zz = (
                    fzp_l[level] * zz_l[level - 1, cell]
                    + fzm_l[level] * zz_l[level, cell]
                )
                upwind = np.copysign(one, flux)
                rw[level, cell] = rw[level, cell] - (
                    cell_sign
                    * (
                        coupling.zb_cell[level, cell, slot]
                        + upwind * coupling.zb3_cell[level, cell, slot]
                    )
                    * flux
                    * interp_zz
                )

    return CoupledState(rho_zz=rho_zz, theta_m=theta_m, ru=ru, rw=rw)


def _coupled_recovery_inputs(
    *,
    rho_zz: object,
    ru: object,
    rw: object,
    zz: object,
    fzm: object,
    fzp: object,
    coupling: TerrainCoupling,
    array_layout: ArrayLayout,
    time_index: int,
) -> tuple[FloatArray, ...]:
    layout = _layout(array_layout)
    nlev = coupling.n_vert_levels
    ncells = coupling.n_cells
    nedges = coupling.n_edges
    fields = (
        _logical_field(
            "rho_zz",
            rho_zz,
            layout=layout,
            entities=ncells,
            levels=nlev,
            time_index=time_index,
        ),
        _logical_field(
            "ru", ru, layout=layout, entities=nedges, levels=nlev, time_index=time_index
        ),
        _logical_field(
            "rw",
            rw,
            layout=layout,
            entities=ncells,
            levels=nlev + 1,
            time_index=time_index,
        ),
        _logical_field(
            "zz", zz, layout=layout, entities=ncells, levels=nlev, time_index=time_index
        ),
        _logical_vector("fzm", fzm, length=nlev),
        _logical_vector("fzp", fzp, length=nlev),
    )
    _same_dtype(
        "rho_zz",
        fields[0],
        ru=fields[1],
        rw=fields[2],
        zz=fields[3],
        fzm=fields[4],
        fzp=fields[5],
    )
    if coupling.zb_cell.dtype != fields[0].dtype:
        raise TypeError(
            f"terrain dtype {coupling.zb_cell.dtype} differs from state dtype {fields[0].dtype}"
        )
    if np.any(fields[0] <= fields[0].dtype.type(0.0)):
        raise ValueError("rho_zz must be strictly positive")
    if np.any(fields[3] <= fields[3].dtype.type(0.0)):
        raise ValueError("zz must be strictly positive")
    return fields


def recover_velocities(
    *,
    rho_zz: object,
    ru: object,
    rw: object,
    zz: object,
    fzm: object,
    fzp: object,
    cf1: float,
    cf2: float,
    cf3: float,
    coupling: TerrainCoupling,
    array_layout: ArrayLayout = "logical",
    time_index: int = 0,
) -> RecoveredVelocity:
    """Recover exact public ``u``/``w`` from coupled ``ru``/``rw`` fields."""

    rho_l, ru_l, rw_l, zz_l, fzm_l, fzp_l = _coupled_recovery_inputs(
        rho_zz=rho_zz,
        ru=ru,
        rw=rw,
        zz=zz,
        fzm=fzm,
        fzp=fzp,
        coupling=coupling,
        array_layout=array_layout,
        time_index=time_index,
    )
    dtype = rho_l.dtype
    if coupling.n_vert_levels < 3:
        raise ValueError("bottom cf1/cf2/cf3 recovery requires at least three levels")
    if not all(np.isfinite(value) for value in (cf1, cf2, cf3)):
        raise ValueError("cf1, cf2 and cf3 must be finite")
    cf1_l, cf2_l, cf3_l = (
        dtype.type(cf1),
        dtype.type(cf2),
        dtype.type(cf3),
    )
    two = dtype.type(2.0)
    one = dtype.type(1.0)
    nlev, ncells = rho_l.shape
    nedges = coupling.n_edges

    cell1, cell2 = coupling.cells_on_edge.T
    denominator = rho_l[:, cell1] + rho_l[:, cell2]
    invalid = np.argwhere(denominator.T == dtype.type(0.0))
    if invalid.size:
        edge, level = invalid[0]
        raise FloatingPointError(f"zero edge-density denominator at level {level}, edge {edge}")
    u = two * ru_l / denominator
    w = np.zeros((nlev + 1, ncells), dtype=dtype)
    interp_zz = fzm_l[1:nlev, None] * zz_l[1:nlev] + fzp_l[1:nlev, None] * zz_l[:nlev-1]
    invalid = np.argwhere(interp_zz.T == dtype.type(0.0))
    if invalid.size:
        cell, level = invalid[0]
        raise FloatingPointError(f"zero interface zz denominator at level {level+1}, cell {cell}")
    w[1:nlev] = rw_l[1:nlev] / interp_zz
    for slot in range(coupling.edges_on_cell.shape[1]):
        cell = np.flatnonzero(coupling.n_edges_on_cell > slot)
        edge = coupling.edges_on_cell[cell, slot]
        cell_sign = coupling.edges_on_cell_sign[cell, slot]
        flux = cf1_l * ru_l[0, edge] + cf2_l * ru_l[1, edge] + cf3_l * ru_l[2, edge]
        w[0, cell] = w[0, cell] + cell_sign * (coupling.zb_cell[0, cell, slot]
            + np.copysign(one, flux) * coupling.zb3_cell[0, cell, slot]) * flux
        flux = fzm_l[1:nlev, None] * ru_l[1:nlev, edge] + fzp_l[1:nlev, None] * ru_l[:nlev-1, edge]
        w[1:nlev, cell] = w[1:nlev, cell] + cell_sign * (coupling.zb_cell[1:nlev, cell, slot]
            + np.copysign(one, flux) * coupling.zb3_cell[1:nlev, cell, slot]) * flux
    bottom_density = cf1_l * rho_l[0] + cf2_l * rho_l[1] + cf3_l * rho_l[2]
    interp_rho = fzm_l[1:nlev, None] * rho_l[1:nlev] + fzp_l[1:nlev, None] * rho_l[:nlev-1]
    # Preserve the scalar path's cell-first failure ordering.
    zeros = np.concatenate(((bottom_density == dtype.type(0.0))[None, :], interp_rho == dtype.type(0.0)))
    invalid = np.argwhere(zeros.T)
    if invalid.size:
        cell, level = invalid[0]
        if level == 0:
            raise FloatingPointError(f"zero bottom density denominator at cell {cell}")
        raise FloatingPointError(f"zero interface density denominator at level {level}, cell {cell}")
    w[0] = w[0] / bottom_density
    w[1:nlev] = w[1:nlev] / interp_rho

    return RecoveredVelocity(u=u, w=w)


def recover_physical_state(
    *,
    rho_zz: object,
    theta_m: object,
    qv: object,
    ru: object,
    rw: object,
    zz: object,
    fzm: object,
    fzp: object,
    cf1: float,
    cf2: float,
    cf3: float,
    coupling: TerrainCoupling,
    array_layout: ArrayLayout = "logical",
    time_index: int = 0,
    water_vapor_gas_constant: float = 461.6,
    dry_air_gas_constant: float = 287.0,
) -> PhysicalState:
    """Recover public ``rho``, ``theta``, ``u`` and ``w`` diagnostics."""

    layout = _layout(array_layout)
    nlev = coupling.n_vert_levels
    ncells = coupling.n_cells
    rho_l = _logical_field(
        "rho_zz",
        rho_zz,
        layout=layout,
        entities=ncells,
        levels=nlev,
        time_index=time_index,
    )
    theta_l = _logical_field(
        "theta_m",
        theta_m,
        layout=layout,
        entities=ncells,
        levels=nlev,
        time_index=time_index,
    )
    qv_l = _logical_field(
        "qv", qv, layout=layout, entities=ncells, levels=nlev, time_index=time_index
    )
    zz_l = _logical_field(
        "zz", zz, layout=layout, entities=ncells, levels=nlev, time_index=time_index
    )
    _same_dtype("rho_zz", rho_l, theta_m=theta_l, qv=qv_l, zz=zz_l)
    if not np.isfinite(water_vapor_gas_constant) or not np.isfinite(dry_air_gas_constant):
        raise ValueError("gas constants must be finite")
    if dry_air_gas_constant == 0.0:
        raise ValueError("dry_air_gas_constant must be nonzero")

    velocity = recover_velocities(
        rho_zz=rho_zz,
        ru=ru,
        rw=rw,
        zz=zz,
        fzm=fzm,
        fzp=fzp,
        cf1=cf1,
        cf2=cf2,
        cf3=cf3,
        coupling=coupling,
        array_layout=layout,
        time_index=time_index,
    )
    dtype = rho_l.dtype
    one = dtype.type(1.0)
    rvord = dtype.type(water_vapor_gas_constant) / dtype.type(dry_air_gas_constant)
    moisture_factor = one + rvord * qv_l
    if np.any(moisture_factor == dtype.type(0.0)):
        raise FloatingPointError("1 + rvord*qv is zero")
    rho_public = rho_l * zz_l
    theta_public = theta_l / moisture_factor
    return PhysicalState(
        rho=rho_public,
        theta=theta_public,
        u=velocity.u,
        w=velocity.w,
    )


def convert_w_tendency_to_omega(
    *,
    w_tendency: object,
    u_tendency: object,
    zz: object,
    fzm: object,
    fzp: object,
    coupling: TerrainCoupling,
    array_layout: ArrayLayout = "logical",
    time_index: int = 0,
    boundary_mask_cell: object | None = None,
    relaxation_zone: int = 0,
) -> FloatArray:
    """Convert physical-w tendency to omega tendency with frozen slope fluxes."""

    layout = _layout(array_layout)
    nlev = coupling.n_vert_levels
    ncells = coupling.n_cells
    nedges = coupling.n_edges
    w_l = _logical_field(
        "w_tendency",
        w_tendency,
        layout=layout,
        entities=ncells,
        levels=nlev + 1,
        time_index=time_index,
    )
    u_l = _logical_field(
        "u_tendency",
        u_tendency,
        layout=layout,
        entities=nedges,
        levels=nlev,
        time_index=time_index,
    )
    zz_l = _logical_field(
        "zz", zz, layout=layout, entities=ncells, levels=nlev, time_index=time_index
    )
    fzm_l = _logical_vector("fzm", fzm, length=nlev)
    fzp_l = _logical_vector("fzp", fzp, length=nlev)
    _same_dtype(
        "w_tendency", w_l, u_tendency=u_l, zz=zz_l, fzm=fzm_l, fzp=fzp_l
    )
    if coupling.zb_cell.dtype != w_l.dtype:
        raise TypeError(
            f"terrain dtype {coupling.zb_cell.dtype} differs from tendency dtype {w_l.dtype}"
        )
    if boundary_mask_cell is None:
        mask = np.zeros(ncells, dtype=np.int64)
    else:
        mask_source = _integer_array("boundary_mask_cell", boundary_mask_cell, ndim=1)
        if mask_source.shape != (ncells,):
            raise ValueError(
                f"boundary_mask_cell shape {mask_source.shape} != ({ncells},)"
            )
        mask = np.asarray(mask_source, dtype=np.int64)
    if not isinstance(relaxation_zone, (int, np.integer)):
        raise TypeError("relaxation_zone must be an integer")

    out = w_l.copy()
    one = out.dtype.type(1.0)
    for cell in range(ncells):
        if mask[cell] > relaxation_zone:
            continue
        for slot in range(int(coupling.n_edges_on_cell[cell])):
            edge = int(coupling.edges_on_cell[cell, slot])
            for level in range(1, nlev):
                flux = coupling.edges_on_cell_sign[cell, slot] * (
                    fzm_l[level] * u_l[level, edge]
                    + fzp_l[level] * u_l[level - 1, edge]
                )
                out[level, cell] = out[level, cell] - (
                    coupling.zb_cell[level, cell, slot]
                    + np.copysign(one, u_l[level, edge])
                    * coupling.zb3_cell[level, cell, slot]
                ) * flux
        for level in range(1, nlev):
            out[level, cell] = (
                fzm_l[level] * zz_l[level, cell]
                + fzp_l[level] * zz_l[level - 1, cell]
            ) * out[level, cell]
    return out


__all__ = [
    "ArrayLayout",
    "CoupledState",
    "PhysicalState",
    "RecoveredVelocity",
    "TerrainCoupling",
    "build_terrain_coupling",
    "convert_w_tendency_to_omega",
    "initialize_coupled_state",
    "recover_physical_state",
    "recover_velocities",
]
