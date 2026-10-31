"""Vertically implicit split-explicit acoustic authority.

This module transcribes the self-contained ``*_work`` routines from frozen
``src/core_atmosphere/dynamics/mpas_atm_time_integration.F``:

* vertical implicit coefficients: lines 1818-1936;
* small-step omega tendency conversion: lines 2030-2111;
* forward/backward acoustic step and column solve: lines 2253-2523.

Arrays retain the Fortran logical order ``(level, entity)``.  MPI halo and
regional exchange orchestration lives outside these scalar authority routines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

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


def _require_same_shape(reference: FloatArray, **arrays: FloatArray) -> None:
    for name, value in arrays.items():
        if np.shape(value) != np.shape(reference):
            raise ValueError(f"{name} shape {np.shape(value)} != {np.shape(reference)}")


@dataclass(frozen=True, slots=True)
class VerticalImplicitCoefficients:
    cofwr: FloatArray
    cofwz: FloatArray
    coftz: FloatArray
    cofwt: FloatArray
    cofrz: FloatArray
    a_tri: FloatArray
    b_tri: FloatArray
    c_tri: FloatArray
    alpha_tri: FloatArray
    gamma_tri: FloatArray


def compute_vertical_implicit_coefficients(
    *,
    dts: float,
    epssm: float,
    zz: FloatArray,
    cqw: FloatArray,
    pressure: FloatArray,
    theta: FloatArray,
    rho_base: FloatArray,
    rho_theta_base: FloatArray,
    pressure_base: FloatArray,
    rho_theta_perturbation: FloatArray,
    qtot: FloatArray,
    rdzw: FloatArray,
    fzm: FloatArray,
    fzp: FloatArray,
    rdzu: FloatArray,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
) -> VerticalImplicitCoefficients:
    """Compute the frozen MPAS tridiagonal acoustic coefficients.

    ``qtot`` is explicit here.  The v8.2.3 source's commented scalar loop is
    replaced by its compiled ``qtot(k,iCell)`` macro at line 1882; making the
    value an argument prevents hidden constituent state in the authority API.
    """
    zz = np.asarray(zz)
    if zz.ndim != 2 or zz.dtype.kind != "f":
        raise ValueError("zz must be a floating (nVertLevels, nCells) array")
    dtype = zz.dtype
    arrays = {
        "cqw": np.asarray(cqw),
        "pressure": np.asarray(pressure),
        "theta": np.asarray(theta),
        "rho_base": np.asarray(rho_base),
        "rho_theta_base": np.asarray(rho_theta_base),
        "pressure_base": np.asarray(pressure_base),
        "rho_theta_perturbation": np.asarray(rho_theta_perturbation),
        "qtot": np.asarray(qtot),
    }
    _require_same_shape(zz, **arrays)
    nlev, ncells = zz.shape
    for name, metric in {
        "rdzw": rdzw,
        "fzm": fzm,
        "fzp": fzp,
        "rdzu": rdzu,
    }.items():
        if np.shape(metric) != (nlev,):
            raise ValueError(f"{name} shape {np.shape(metric)} != ({nlev},)")
    if cp == rgas:
        raise ValueError("cp-rgas must be nonzero")
    if np.any(1.0 + arrays["qtot"] == 0.0):
        raise ValueError("1+qtot is zero")

    dtseps = dtype.type(0.5 * dts * (1.0 + epssm))
    rcv = dtype.type(rgas / (cp - rgas))
    c2 = dtype.type(cp) * rcv
    rdzw = np.asarray(rdzw, dtype=dtype)
    fzm = np.asarray(fzm, dtype=dtype)
    fzp = np.asarray(fzp, dtype=dtype)
    rdzu = np.asarray(rdzu, dtype=dtype)
    cofrz = dtseps * rdzw

    cofwr = np.zeros_like(zz)
    cofwz = np.zeros_like(zz)
    coftz = np.zeros((nlev + 1, ncells), dtype=dtype)
    cofwt = np.zeros_like(zz)
    for level in range(1, nlev):
        interp_zz = fzm[level] * zz[level] + fzp[level] * zz[level - 1]
        cofwr[level] = dtype.type(0.5) * dtseps * dtype.type(gravity) * interp_zz
        interp_pressure = fzm[level] * arrays["pressure"][level] + fzp[level] * arrays["pressure"][level - 1]
        cofwz[level] = (
            dtseps
            * c2
            * interp_zz
            * rdzu[level]
            * arrays["cqw"][level]
            * interp_pressure
        )
        coftz[level] = dtseps * (
            fzm[level] * arrays["theta"][level] + fzp[level] * arrays["theta"][level - 1]
        )
    cofwt[:] = (
        dtype.type(0.5)
        * dtseps
        * rcv
        * zz
        * dtype.type(gravity)
        * arrays["rho_base"]
        / (dtype.type(1.0) + arrays["qtot"])
        * arrays["pressure"]
        / (
            (arrays["rho_theta_base"] + arrays["rho_theta_perturbation"])
            * arrays["pressure_base"]
        )
    )

    a_tri = np.zeros_like(zz)
    b_tri = np.ones_like(zz)
    c_tri = np.zeros_like(zz)
    alpha_tri = np.zeros_like(zz)
    gamma_tri = np.zeros_like(zz)
    for level in range(1, nlev):
        a_tri[level] = (
            -cofwz[level] * coftz[level - 1] * rdzw[level - 1] * zz[level - 1]
            + cofwr[level] * cofrz[level - 1]
            - cofwt[level - 1] * coftz[level - 1] * rdzw[level - 1]
        )
        b_tri[level] = (
            dtype.type(1.0)
            + cofwz[level]
            * coftz[level]
            * (rdzw[level] * zz[level] + rdzw[level - 1] * zz[level - 1])
            - coftz[level]
            * (cofwt[level] * rdzw[level] - cofwt[level - 1] * rdzw[level - 1])
            + cofwr[level] * (cofrz[level] - cofrz[level - 1])
        )
        c_tri[level] = (
            -cofwz[level] * coftz[level + 1] * rdzw[level] * zz[level]
            - cofwr[level] * cofrz[level]
            + cofwt[level] * coftz[level + 1] * rdzw[level]
        )
        denominator = b_tri[level] - a_tri[level] * gamma_tri[level - 1]
        if np.any(denominator == 0.0):
            raise FloatingPointError(f"singular vertical acoustic system at level {level}")
        alpha_tri[level] = np.reciprocal(denominator)
        gamma_tri[level] = c_tri[level] * alpha_tri[level]

    return VerticalImplicitCoefficients(
        cofwr=cofwr,
        cofwz=cofwz,
        coftz=coftz,
        cofwt=cofwt,
        cofrz=cofrz,
        a_tri=a_tri,
        b_tri=b_tri,
        c_tri=c_tri,
        alpha_tri=alpha_tri,
        gamma_tri=gamma_tri,
    )


def solve_vertical_system(rhs: FloatArray, coefficients: VerticalImplicitCoefficients) -> FloatArray:
    """Apply MPAS's upward/downward Thomas sweep (Fortran 2475-2486)."""
    result = np.asarray(rhs).copy()
    if result.ndim != 2:
        raise ValueError("rhs must have shape (nVertLevels+1, nCells)")
    nlev = coefficients.a_tri.shape[0]
    if result.shape != (nlev + 1, coefficients.a_tri.shape[1]):
        raise ValueError(
            f"rhs shape {result.shape} != ({nlev + 1}, {coefficients.a_tri.shape[1]})"
        )
    for cell in range(result.shape[1]):
        for level in range(1, nlev):
            result[level, cell] = (
                result[level, cell]
                - coefficients.a_tri[level, cell] * result[level - 1, cell]
            ) * coefficients.alpha_tri[level, cell]
        for level in range(nlev - 1, -1, -1):
            result[level, cell] -= (
                coefficients.gamma_tri[level, cell] * result[level + 1, cell]
            )
    return result


def edge_signs_on_cells(mesh: object) -> FloatArray:
    """Build ``edgesOnCell_sign`` exactly as ``mpas_atm_core.F:1186-1202``."""
    counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    signs = np.zeros(edges.shape, dtype=np.float64)
    # Whole-array form of the per-slot loop (2.5 s of Python per call on the
    # 20 km mesh, rented box 2026-10-13): the slots below nEdgesOnCell of each cell,
    # a missing (negative) edge left at zero, +1 where the cell is the edge's
    # first endpoint and -1 otherwise.  A count past the row width or an edge
    # past cellsOnEdge raises IndexError, as the loop's indexing did.
    if counts.size and int(counts.max()) > edges.shape[1]:
        raise IndexError("nEdgesOnCell names more slots than edgesOnCell holds")
    used = np.arange(edges.shape[1])[None, :] < counts[:, None]
    cell, slot = np.nonzero(used)
    edge = edges[cell, slot]
    present = edge >= 0
    cell, slot, edge = cell[present], slot[present], edge[present]
    signs[cell, slot] = np.where(cells[edge, 0] == cell, 1.0, -1.0)
    return signs


def convert_w_tendency_to_omega(
    mesh: object,
    w_tendency: FloatArray,
    u_tendency: FloatArray,
    *,
    fzm: FloatArray,
    fzp: FloatArray,
    zz: FloatArray,
    zb_cell: FloatArray,
    zb3_cell: FloatArray,
    boundary_mask_cell: NDArray[np.integer[Any]] | None = None,
    relaxation_zone: int = 0,
) -> FloatArray:
    """Transcribe ``atm_set_smlstep_pert_variables_work`` lines 2088-2110."""
    out = np.asarray(w_tendency).copy()
    u_tendency = np.asarray(u_tendency)
    nlev, nedges = u_tendency.shape
    ncells = _mesh_array(mesh, "areaCell").size
    if out.shape != (nlev + 1, ncells):
        raise ValueError(f"w_tendency shape {out.shape} != ({nlev + 1}, {ncells})")
    counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    signs = edge_signs_on_cells(mesh).astype(out.dtype, copy=False)
    if boundary_mask_cell is None:
        boundary_mask_cell = np.zeros(ncells, dtype=np.int32)
    active = np.asarray(boundary_mask_cell) <= relaxation_zone
    for slot in range(edges.shape[1]):
        cells = np.flatnonzero(active & (counts > slot) & (edges[:, slot] >= 0) & (edges[:, slot] < nedges))
        edge = edges[cells, slot]
        flux = signs[cells, slot] * (fzm[1:nlev, None] * u_tendency[1:nlev, edge]
                                    + fzp[1:nlev, None] * u_tendency[:nlev-1, edge])
        upwind_sign = np.copysign(out.dtype.type(1.0), u_tendency[1:nlev, edge])
        out[1:nlev, cells] -= (zb_cell[1:nlev, cells, slot] + upwind_sign * zb3_cell[1:nlev, cells, slot]) * flux
    cells = np.flatnonzero(active)
    out[1:nlev, cells] *= fzm[1:nlev, None] * zz[1:nlev, cells] + fzp[1:nlev, None] * zz[:nlev-1, cells]
    return out


@dataclass(slots=True)
class AcousticStepState:
    ru_p: FloatArray
    rw_p: FloatArray
    rtheta_pp: FloatArray
    rtheta_pp_old: FloatArray
    rho_pp: FloatArray
    ru_avg: FloatArray
    ww_avg: FloatArray

    def copy(self) -> "AcousticStepState":
        return AcousticStepState(**{name: np.asarray(getattr(self, name)).copy() for name in self.__slots__})


@dataclass(frozen=True, slots=True)
class AcousticStepForcing:
    rho_zz: FloatArray
    theta_m: FloatArray
    zz: FloatArray
    exner: FloatArray
    cqu: FloatArray
    zxu: FloatArray
    dss: FloatArray
    tend_ru: FloatArray
    tend_rho: FloatArray
    tend_rt: FloatArray
    tend_rw: FloatArray
    w: FloatArray
    rw: FloatArray
    rw_save: FloatArray


def advance_acoustic_step(
    mesh: object,
    state: AcousticStepState,
    forcing: AcousticStepForcing,
    coefficients: VerticalImplicitCoefficients,
    *,
    dts: float,
    small_step: int,
    epssm: float,
    fzm: FloatArray,
    fzp: FloatArray,
    rdzw: FloatArray,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
    specified_zone_edge: FloatArray | None = None,
    specified_zone_cell: FloatArray | None = None,
) -> AcousticStepState:
    """Advance one forward/backward acoustic substep on a closed mesh.

    This keeps the source loop order and returns a new state.  The caller owns
    halo exchanges between substeps, matching the split between
    ``atm_advance_acoustic_step`` and its ``_work`` routine.
    """
    if small_step < 1:
        raise ConfigurationRefusal(
            "small_step", small_step, "MPAS acoustic substeps are numbered from one", "small_step=1"
        )
    out = state.copy()
    nlev, nedges = out.ru_p.shape
    ncells = out.rho_pp.shape[1]
    if out.rw_p.shape != (nlev + 1, ncells):
        raise ValueError("rw_p must have shape (nVertLevels+1, nCells)")
    if specified_zone_edge is None:
        specified_zone_edge = np.zeros(nedges, dtype=out.ru_p.dtype)
    if specified_zone_cell is None:
        specified_zone_cell = np.zeros(ncells, dtype=out.rho_pp.dtype)
    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    edges_on_cell = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    n_edges_on_cell = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    dc_edge = _mesh_array(mesh, "dcEdge").astype(out.ru_p.dtype, copy=False)
    dv_edge = _mesh_array(mesh, "dvEdge").astype(out.ru_p.dtype, copy=False)
    area_cell = _mesh_array(mesh, "areaCell").astype(out.rho_pp.dtype, copy=False)
    signs = edge_signs_on_cells(mesh).astype(out.rho_pp.dtype, copy=False)
    c2 = out.rho_pp.dtype.type(cp * rgas / (cp - rgas))
    resm = out.rho_pp.dtype.type((1.0 - epssm) / (1.0 + epssm))

    if small_step != 1:
        for edge in range(nedges):
            cell0, cell1 = cells_on_edge[edge]
            if cell0 < 0 or cell1 < 0:
                raise ConfigurationRefusal(
                    "regional_boundary_state",
                    f"edge {edge}",
                    "the acoustic pressure gradient needs both edge cells",
                    "explicit lateral boundary cells before this substep",
                )
            pgrad = (
                (out.rtheta_pp[:, cell1] - out.rtheta_pp[:, cell0]) / dc_edge[edge]
            ) / (out.rho_pp.dtype.type(0.5) * (forcing.zz[:, cell1] + forcing.zz[:, cell0]))
            pgrad = (
                forcing.cqu[:, edge]
                * out.rho_pp.dtype.type(0.5)
                * c2
                * (forcing.exner[:, cell0] + forcing.exner[:, cell1])
                * pgrad
            )
            pgrad += (
                out.rho_pp.dtype.type(0.5 * gravity)
                * forcing.zxu[:, edge]
                * (out.rho_pp[:, cell0] + out.rho_pp[:, cell1])
            )
            out.ru_p[:, edge] += out.ru_p.dtype.type(dts) * (
                forcing.tend_ru[:, edge]
                - (out.ru_p.dtype.type(1.0) - specified_zone_edge[edge]) * pgrad
            )
            out.ru_avg[:, edge] += out.ru_p[:, edge]
    else:
        out.ru_p[:] = out.ru_p.dtype.type(dts) * forcing.tend_ru
        out.ru_avg[:] = out.ru_p

    if small_step == 1:
        out.rtheta_pp_old.fill(0.0)
    else:
        out.rtheta_pp_old[:] = out.rtheta_pp

    for cell in range(ncells):
        if small_step == 1:
            out.ww_avg[:, cell] = 0.0
            out.rho_pp[:, cell] = 0.0
            out.rtheta_pp[:, cell] = 0.0
            out.rw_p[:, cell] = 0.0

        if specified_zone_cell[cell] != 0.0:
            out.rho_pp[:, cell] += out.rho_pp.dtype.type(dts) * forcing.tend_rho[:, cell]
            out.rtheta_pp[:, cell] += out.rtheta_pp.dtype.type(dts) * forcing.tend_rt[:, cell]
            out.rw_p[:-1, cell] += out.rw_p.dtype.type(dts) * forcing.tend_rw[:-1, cell]
            out.ww_avg[:-1, cell] += (
                out.ww_avg.dtype.type(0.5 * (1.0 + epssm)) * out.rw_p[:-1, cell]
            )
            continue

        rs = np.zeros(nlev, dtype=out.rho_pp.dtype)
        ts = np.zeros(nlev, dtype=out.rtheta_pp.dtype)
        for slot in range(int(n_edges_on_cell[cell])):
            edge = int(edges_on_cell[cell, slot])
            cell0, cell1 = cells_on_edge[edge]
            flux = (
                signs[cell, slot]
                * out.rho_pp.dtype.type(dts)
                * dv_edge[edge]
                * out.ru_p[:, edge]
                / area_cell[cell]
            )
            rs -= flux
            ts -= flux * out.rtheta_pp.dtype.type(0.5) * (
                forcing.theta_m[:, cell1] + forcing.theta_m[:, cell0]
            )

        rs += (
            out.rho_pp[:, cell]
            + out.rho_pp.dtype.type(dts) * forcing.tend_rho[:, cell]
            - coefficients.cofrz * resm * np.diff(out.rw_p[:, cell])
        )
        ts += (
            out.rtheta_pp[:, cell]
            + out.rtheta_pp.dtype.type(dts) * forcing.tend_rt[:, cell]
            - resm
            * rdzw
            * (
                coefficients.coftz[1:, cell] * out.rw_p[1:, cell]
                - coefficients.coftz[:-1, cell] * out.rw_p[:-1, cell]
            )
        )
        out.ww_avg[1:nlev, cell] += (
            out.ww_avg.dtype.type(0.5 * (1.0 - epssm)) * out.rw_p[1:nlev, cell]
        )
        for level in range(1, nlev):
            out.rw_p[level, cell] += (
                out.rw_p.dtype.type(dts) * forcing.tend_rw[level, cell]
                - coefficients.cofwz[level, cell]
                * (
                    forcing.zz[level, cell] * ts[level]
                    - forcing.zz[level - 1, cell] * ts[level - 1]
                    + resm
                    * (
                        forcing.zz[level, cell] * out.rtheta_pp[level, cell]
                        - forcing.zz[level - 1, cell] * out.rtheta_pp[level - 1, cell]
                    )
                )
                - coefficients.cofwr[level, cell]
                * (
                    rs[level]
                    + rs[level - 1]
                    + resm * (out.rho_pp[level, cell] + out.rho_pp[level - 1, cell])
                )
                + coefficients.cofwt[level, cell]
                * (ts[level] + resm * out.rtheta_pp[level, cell])
                + coefficients.cofwt[level - 1, cell]
                * (ts[level - 1] + resm * out.rtheta_pp[level - 1, cell])
            )

        solved = solve_vertical_system(out.rw_p[:, cell : cell + 1], coefficients_for_cell(coefficients, cell))
        out.rw_p[:, cell] = solved[:, 0]
        for level in range(1, nlev):
            delta_saved = forcing.rw_save[level, cell] - forcing.rw[level, cell]
            damping = forcing.dss[level, cell]
            density_at_interface = (
                fzm[level] * forcing.zz[level, cell]
                + fzp[level] * forcing.zz[level - 1, cell]
            ) * (
                fzm[level] * forcing.rho_zz[level, cell]
                + fzp[level] * forcing.rho_zz[level - 1, cell]
            )
            out.rw_p[level, cell] = (
                out.rw_p[level, cell]
                + delta_saved
                - out.rw_p.dtype.type(dts) * damping * density_at_interface * forcing.w[level, cell]
            ) / (out.rw_p.dtype.type(1.0) + out.rw_p.dtype.type(dts) * damping) - delta_saved
        out.ww_avg[1:nlev, cell] += (
            out.ww_avg.dtype.type(0.5 * (1.0 + epssm)) * out.rw_p[1:nlev, cell]
        )
        out.rho_pp[:, cell] = rs - coefficients.cofrz * np.diff(out.rw_p[:, cell])
        out.rtheta_pp[:, cell] = ts - rdzw * (
            coefficients.coftz[1:, cell] * out.rw_p[1:, cell]
            - coefficients.coftz[:-1, cell] * out.rw_p[:-1, cell]
        )
    return out


def coefficients_for_cell(
    coefficients: VerticalImplicitCoefficients, cell: int
) -> VerticalImplicitCoefficients:
    """Return a one-cell view for the scalar column authority."""
    values: dict[str, FloatArray] = {}
    for name in coefficients.__slots__:
        value = np.asarray(getattr(coefficients, name))
        values[name] = value if value.ndim == 1 else value[:, cell : cell + 1]
    return VerticalImplicitCoefficients(**values)

