"""Frozen MPAS-A v8.2.3 dry horizontal filters and divergence damping.

This module is the level-first CPU authority for the filter branches exercised
by the frozen Jablonowski--Williamson control run.  Logical array order is
``(nVertLevels, horizontal_entity)``; vertical velocity/momentum has
``nVertLevels + 1`` entries.  The implementation deliberately retains the
Fortran stencil accumulation order.

Source map
----------
* mesh/length scaling: ``mpas_atm_core.F:177-207,1079-1134``;
* Smagorinsky coefficient: ``mpas_atm_time_integration.F:4640-4701``;
* momentum filter: ``mpas_atm_time_integration.F:4822-4923``;
* vertical-momentum filter: ``mpas_atm_time_integration.F:5071-5133``;
* theta filter: ``mpas_atm_time_integration.F:5250-5304``; and
* 3-D divergence damping: ``mpas_atm_time_integration.F:2409-2417,2532-2601``.

The three horizontal Euler tendencies are computed on RK stage one and must be
saved unchanged for stages two and three.  :func:`apply_saved_euler_mixing`
performs the dry additions at frozen lines 5009-5015, 5197-5202, and
5386-5392; it does not pretend that these filter increments include the other
pressure-gradient/buoyancy terms stored in MPAS's full ``tend_*_euler`` pools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import rkind_libm
from .native_powf import powf_native_gnu
from .errors import ConfigurationRefusal
from .horizontal import edge_sign_on_cell, edge_sign_on_vertex


FloatArray = NDArray[np.floating[Any]]


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


def _optional_mesh_array(mesh: object, name: str) -> NDArray[Any] | None:
    try:
        return _mesh_array(mesh, name)
    except AttributeError:
        return None


def _float_field(name: str, value: object, *, ndim: int) -> FloatArray:
    field = np.asarray(value)
    if field.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {field.shape}")
    if field.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must have dtype float32 or float64")
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{name} must contain only finite values")
    return field


def _same_dtype(reference: FloatArray, name: str, value: object, *, ndim: int) -> FloatArray:
    field = _float_field(name, value, ndim=ndim)
    if field.dtype != reference.dtype:
        raise TypeError(
            f"{name} dtype {field.dtype} does not match authority dtype {reference.dtype}"
        )
    return field


def _finite_scalar(knob: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ConfigurationRefusal(
            knob,
            value,
            "the frozen real-valued option must be finite",
            f"a finite {knob}",
        )
    return scalar


@dataclass(frozen=True, slots=True)
class MixingConfig:
    """Exact works-or-refuses contract for the frozen JW dry filter branch."""

    config_horiz_mixing: str = "2d_smagorinsky"
    config_len_disp: float = 0.0
    config_visc4_2dsmag: float = 0.05
    config_smagorinsky_coef: float = 0.125
    config_del4u_div_factor: float = 10.0
    config_h_ScaleWithMesh: bool = True
    config_mpas_cam_coef: float = 0.0
    config_smdiv: float = 0.1

    def validate(self) -> None:
        if self.config_horiz_mixing != "2d_smagorinsky":
            raise ConfigurationRefusal(
                "config_horiz_mixing",
                self.config_horiz_mixing,
                "this authority admits the frozen JW 2-D Smagorinsky branch",
                "config_horiz_mixing='2d_smagorinsky'",
            )

        length = _finite_scalar("config_len_disp", self.config_len_disp)
        if length < 0.0:
            raise ConfigurationRefusal(
                "config_len_disp",
                self.config_len_disp,
                "a negative filter length is not an MPAS-admitted value",
                "config_len_disp>=0 (zero selects nominalMinDc)",
            )

        visc4 = _finite_scalar("config_visc4_2dsmag", self.config_visc4_2dsmag)
        if visc4 < 0.0:
            raise ConfigurationRefusal(
                "config_visc4_2dsmag",
                self.config_visc4_2dsmag,
                "the Registry admits only non-negative fourth-order scaling",
                "config_visc4_2dsmag>=0",
            )

        smag = _finite_scalar(
            "config_smagorinsky_coef", self.config_smagorinsky_coef
        )
        if smag < 0.0:
            raise ConfigurationRefusal(
                "config_smagorinsky_coef",
                self.config_smagorinsky_coef,
                "the admitted Smagorinsky coefficient is non-negative",
                "config_smagorinsky_coef>=0",
            )

        div_factor = _finite_scalar(
            "config_del4u_div_factor", self.config_del4u_div_factor
        )
        if div_factor <= 0.0:
            raise ConfigurationRefusal(
                "config_del4u_div_factor",
                self.config_del4u_div_factor,
                "the Registry requires a positive divergent hyperdiffusion factor",
                "config_del4u_div_factor>0",
            )

        cam = _finite_scalar("config_mpas_cam_coef", self.config_mpas_cam_coef)
        if cam != 0.0:
            raise ConfigurationRefusal(
                "config_mpas_cam_coef",
                self.config_mpas_cam_coef,
                "the JW branch has no CAM-SE upper-level coefficient floor",
                "config_mpas_cam_coef=0.0",
            )

        smdiv = _finite_scalar("config_smdiv", self.config_smdiv)
        if smdiv < 0.0:
            raise ConfigurationRefusal(
                "config_smdiv",
                self.config_smdiv,
                "the 3-D divergence damping coefficient must be non-negative",
                "config_smdiv>=0",
            )

        if not isinstance(self.config_h_ScaleWithMesh, (bool, np.bool_)):
            raise ConfigurationRefusal(
                "config_h_ScaleWithMesh",
                self.config_h_ScaleWithMesh,
                "the Registry option is logical",
                "config_h_ScaleWithMesh=True or False",
            )


@dataclass(frozen=True, slots=True)
class MeshMixingScaling:
    del2: FloatArray
    del4: FloatArray


@dataclass(frozen=True, slots=True)
class SmagorinskyCoefficients:
    kdiff: FloatArray
    h_mom_eddy_visc4: np.floating[Any]
    h_theta_eddy_visc4: np.floating[Any]
    config_len_disp: np.floating[Any]


@dataclass(frozen=True, slots=True)
class MomentumFilterResult:
    tendency: FloatArray
    delsq_u: FloatArray
    delsq_divergence: FloatArray
    delsq_vorticity: FloatArray


@dataclass(frozen=True, slots=True)
class CellFilterResult:
    tendency: FloatArray
    laplacian: FloatArray


@dataclass(frozen=True, slots=True)
class DryMixingTendencies:
    """RK-stage-one horizontal increments that must be retained for RK2/RK3."""

    kdiff: FloatArray
    h_mom_eddy_visc4: np.floating[Any]
    h_theta_eddy_visc4: np.floating[Any]
    tend_u_euler: FloatArray
    tend_w_euler: FloatArray
    tend_theta_euler: FloatArray
    delsq_u: FloatArray
    delsq_divergence: FloatArray
    delsq_vorticity: FloatArray
    delsq_w: FloatArray
    delsq_theta: FloatArray


@dataclass(frozen=True, slots=True)
class AppliedMixingTendencies:
    u: FloatArray
    w: FloatArray
    theta: FloatArray


def resolve_config_len_disp(
    mesh: object,
    config_len_disp: float = 0.0,
    *,
    dtype: np.dtype[Any] | type[Any] = np.float64,
) -> np.floating[Any]:
    """Resolve zero ``config_len_disp`` from ``nominalMinDc``.

    This is the frozen selection in ``mpas_atm_core.F:177-207``.  The
    published x1.2562 static file supplies ``nominalMinDc=480000 m``.
    """

    out_dtype = np.dtype(dtype)
    if out_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("mixing authority dtype must be float32 or float64")
    configured = _finite_scalar("config_len_disp", config_len_disp)
    if configured > 0.0:
        return out_dtype.type(configured)
    if configured < 0.0:
        raise ConfigurationRefusal(
            "config_len_disp",
            config_len_disp,
            "a negative filter length is not admitted",
            "config_len_disp>=0 (zero selects nominalMinDc)",
        )

    nominal_raw = _mesh_array(mesh, "nominalMinDc")
    if nominal_raw.size != 1:
        raise ValueError("mesh nominalMinDc must be scalar")
    nominal = float(nominal_raw.reshape(-1)[0])
    if not np.isfinite(nominal) or nominal <= 0.0:
        raise ConfigurationRefusal(
            "config_len_disp",
            config_len_disp,
            "both config_len_disp and mesh nominalMinDc are non-positive",
            "config_len_disp>0 or a mesh with nominalMinDc>0",
        )
    return out_dtype.type(nominal)


def compute_mesh_mixing_scaling(
    mesh: object,
    *,
    config_h_ScaleWithMesh: bool = True,
    dtype: np.dtype[Any] | type[Any] = np.float64,
) -> MeshMixingScaling:
    """Compute ``meshScalingDel2/Del4`` from frozen core lines 1103-1115."""

    out_dtype = np.dtype(dtype)
    if out_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("mixing authority dtype must be float32 or float64")
    if not isinstance(config_h_ScaleWithMesh, (bool, np.bool_)):
        raise ConfigurationRefusal(
            "config_h_ScaleWithMesh",
            config_h_ScaleWithMesh,
            "the Registry option is logical",
            "config_h_ScaleWithMesh=True or False",
        )
    cells = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("cellsOnEdge must have shape (nEdges, 2)")
    density = np.asarray(_mesh_array(mesh, "meshDensity"), dtype=out_dtype)
    if density.ndim != 1 or np.any(~np.isfinite(density)) or np.any(density <= 0.0):
        raise ValueError("meshDensity must be a finite positive cell field")
    n_cells = density.size
    if np.any(cells < 0):
        # Regional ring-7 one-cell edges: native gathers meshDensity at the
        # garbage cell, whose pool allocation is 0.0
        # (atm_compute_mesh_scaling, mpas_atm_core.F:1152-1160); the
        # resulting edge scaling is inert because ring-7 edges never carry a
        # mixing coefficient.
        cells = np.where(cells < 0, n_cells, cells)
        density = np.concatenate(
            [density, np.zeros(1, dtype=out_dtype)]
        )
    if np.any((cells < 0) | (cells >= density.size)):
        raise ValueError("cellsOnEdge contains an invalid cell")

    del2 = np.ones(cells.shape[0], dtype=out_dtype)
    del4 = np.ones(cells.shape[0], dtype=out_dtype)
    if config_h_ScaleWithMesh:
        half = out_dtype.type(0.5)
        quarter = out_dtype.type(0.25)
        three_quarters = out_dtype.type(0.75)
        # GNU libm powf, as gfortran's RKIND ** (see regional_v841.compute_mesh_scaling_regional).
        mean_density = half * (density[cells[:, 0]] + density[cells[:, 1]])
        del2 = (out_dtype.type(1.0) / powf_native_gnu(mean_density, quarter)).astype(out_dtype)
        del4 = (out_dtype.type(1.0) / powf_native_gnu(mean_density, three_quarters)).astype(out_dtype)
    return MeshMixingScaling(del2=del2, del4=del4)


def compute_smagorinsky_coefficients(
    mesh: object,
    normal_velocity: object,
    tangential_velocity: object,
    *,
    dt: float,
    config: MixingConfig | None = None,
) -> SmagorinskyCoefficients:
    """Compute ``kdiff`` and the constant del4 coefficients (lines 4640-4683)."""

    cfg = MixingConfig() if config is None else config
    cfg.validate()
    u = _float_field("normal_velocity", normal_velocity, ndim=2)
    v = _same_dtype(u, "tangential_velocity", tangential_velocity, ndim=2)
    if v.shape != u.shape:
        raise ValueError("normal_velocity and tangential_velocity shapes differ")
    timestep = _finite_scalar("config_dt", dt)
    if timestep <= 0.0:
        raise ConfigurationRefusal(
            "config_dt",
            dt,
            "the Smagorinsky stability ceiling requires a positive timestep",
            "config_dt>0",
        )

    edges = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    defc_a = np.asarray(_mesh_array(mesh, "defc_a"), dtype=u.dtype)
    defc_b = np.asarray(_mesh_array(mesh, "defc_b"), dtype=u.dtype)
    if edges.ndim != 2 or counts.shape != (edges.shape[0],):
        raise ValueError("edgesOnCell/nEdgesOnCell shapes are inconsistent")
    if defc_a.shape != edges.shape or defc_b.shape != edges.shape:
        raise ValueError("defc_a and defc_b must have shape edgesOnCell")
    used = np.arange(edges.shape[1])[None, :] < counts[:, None]
    if np.any((edges[used] < 0) | (edges[used] >= u.shape[1])):
        raise ValueError("edgesOnCell contains an invalid used edge")

    dtype = u.dtype
    length = resolve_config_len_disp(mesh, cfg.config_len_disp, dtype=dtype)
    cs = dtype.type(cfg.config_smagorinsky_coef)
    strain_scale = (cs * length) * (cs * length)
    ceiling = (
        dtype.type(0.01) * length * length / dtype.type(timestep)
    )
    kdiff = np.zeros((u.shape[0], edges.shape[0]), dtype=dtype)
    for cell in range(edges.shape[0]):
        d_diag = np.zeros(u.shape[0], dtype=dtype)
        d_off_diag = np.zeros(u.shape[0], dtype=dtype)
        for slot in range(int(counts[cell])):
            edge = int(edges[cell, slot])
            d_diag += defc_a[cell, slot] * u[:, edge] - defc_b[cell, slot] * v[:, edge]
            d_off_diag += defc_b[cell, slot] * u[:, edge] + defc_a[cell, slot] * v[:, edge]
        strain = np.sqrt(d_diag * d_diag + d_off_diag * d_off_diag)
        kdiff[:, cell] = np.minimum(strain_scale * strain, ceiling)

    visc4 = dtype.type(cfg.config_visc4_2dsmag)
    h4 = visc4 * length * length * length
    return SmagorinskyCoefficients(
        kdiff=kdiff,
        h_mom_eddy_visc4=h4,
        h_theta_eddy_visc4=h4,
        config_len_disp=length,
    )


def momentum_horizontal_filter_tendency(
    mesh: object,
    *,
    rho_edge: object,
    divergence: object,
    vorticity: object,
    kdiff: object,
    h_mom_eddy_visc4: float,
    config_del4u_div_factor: float = 10.0,
    mesh_scaling_del2: object | None = None,
    mesh_scaling_del4: object | None = None,
    initial_tendency: object | None = None,
    native_stored_inverse: bool = False,
) -> MomentumFilterResult:
    """Return the Euler momentum-filter increment from frozen lines 4828-4923."""

    rho = _float_field("rho_edge", rho_edge, ndim=2)
    div = _same_dtype(rho, "divergence", divergence, ndim=2)
    vort = _same_dtype(rho, "vorticity", vorticity, ndim=2)
    diffusivity = _same_dtype(rho, "kdiff", kdiff, ndim=2)
    nlev, nedges = rho.shape
    cells = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    vertices = np.asarray(_mesh_array(mesh, "verticesOnEdge"), dtype=np.int64)
    if cells.shape != (nedges, 2) or vertices.shape != (nedges, 2):
        raise ValueError("edge connectivity does not match rho_edge")
    if div.shape != diffusivity.shape:
        raise ValueError("divergence and kdiff cell shapes differ")
    if div.shape[0] != nlev or vort.shape[0] != nlev:
        raise ValueError("momentum filter vertical dimensions differ")
    if np.any((cells < 0) | (cells >= div.shape[1])):
        raise ValueError("cellsOnEdge contains an invalid cell")
    if np.any((vertices < 0) | (vertices >= vort.shape[1])):
        raise ValueError("verticesOnEdge contains an invalid vertex")

    dtype = rho.dtype
    dc = np.asarray(_mesh_array(mesh, "dcEdge"), dtype=dtype)
    dv = np.asarray(_mesh_array(mesh, "dvEdge"), dtype=dtype)
    if dc.shape != (nedges,) or dv.shape != (nedges,) or np.any(dc <= 0.0) or np.any(dv <= 0.0):
        raise ValueError("dcEdge/dvEdge must be positive edge fields")
    inv_dc = np.reciprocal(dc)
    inv_dv_limited = np.minimum(np.reciprocal(dv), dtype.type(4.0) * inv_dc)

    if mesh_scaling_del2 is None or mesh_scaling_del4 is None:
        scaling = compute_mesh_mixing_scaling(mesh, dtype=dtype)
        if mesh_scaling_del2 is None:
            mesh_scaling_del2 = scaling.del2
        if mesh_scaling_del4 is None:
            mesh_scaling_del4 = scaling.del4
    del2 = np.asarray(mesh_scaling_del2, dtype=dtype)
    del4 = np.asarray(mesh_scaling_del4, dtype=dtype)
    if del2.shape != (nedges,) or del4.shape != (nedges,):
        raise ValueError("mesh mixing scale fields must have shape (nEdges,)")

    h4_value = _finite_scalar("config_visc4_2dsmag", h_mom_eddy_visc4)
    if h4_value < 0.0:
        raise ConfigurationRefusal(
            "config_visc4_2dsmag",
            h_mom_eddy_visc4,
            "the derived fourth-order viscosity cannot be negative",
            "config_visc4_2dsmag>=0",
        )
    div_factor = _finite_scalar(
        "config_del4u_div_factor", config_del4u_div_factor
    )
    if div_factor <= 0.0:
        raise ConfigurationRefusal(
            "config_del4u_div_factor",
            config_del4u_div_factor,
            "the divergent fourth-order factor must be positive",
            "config_del4u_div_factor>0",
        )

    delsq_u = np.zeros_like(rho)
    tendency = np.zeros_like(rho) if initial_tendency is None else _same_dtype(
        rho, "initial_tendency", initial_tendency, ndim=2
    ).copy()
    if tendency.shape != rho.shape:
        raise ValueError("initial_tendency must match rho_edge")
    half = dtype.type(0.5)
    for edge, (cell0, cell1) in enumerate(cells):
        vertex0, vertex1 = vertices[edge]
        diffusion = (
            (div[:, cell1] - div[:, cell0]) * inv_dc[edge]
            - (vort[:, vertex1] - vort[:, vertex0]) * inv_dv_limited[edge]
        )
        delsq_u[:, edge] += diffusion
        kdiff_u = half * (diffusivity[:, cell0] + diffusivity[:, cell1])
        tendency[:, edge] += rho[:, edge] * kdiff_u * diffusion * del2[edge]

    ncells = div.shape[1]
    nvertices = vort.shape[1]
    delsq_divergence = np.zeros((nlev, ncells), dtype=dtype)
    delsq_vorticity = np.zeros((nlev, nvertices), dtype=dtype)
    h4 = dtype.type(h4_value)
    if h4 > dtype.type(0.0):
        edges_on_vertex = np.asarray(_mesh_array(mesh, "edgesOnVertex"), dtype=np.int64)
        edges_on_cell = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
        counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
        if edges_on_vertex.shape[0] != nvertices or edges_on_cell.shape[0] != ncells:
            raise ValueError("cell/vertex edge stencils do not match diagnostic fields")
        signs_vertex = edge_sign_on_vertex(mesh, dtype=dtype)
        signs_cell = edge_sign_on_cell(mesh, dtype=dtype)
        area_vertex = np.asarray(_mesh_array(mesh, "areaTriangle"), dtype=dtype)
        area_cell = np.asarray(_mesh_array(mesh, "areaCell"), dtype=dtype)
        if np.any(area_vertex <= 0.0) or np.any(area_cell <= 0.0):
            raise ValueError("MPAS cell and triangle areas must be positive")

        for vertex in range(nvertices):
            for slot in range(edges_on_vertex.shape[1]):
                edge = int(edges_on_vertex[vertex, slot])
                if edge < 0:
                    continue
                factor = ((dtype.type(1.0) / area_vertex[vertex]) * dc[edge] * signs_vertex[vertex, slot]
                          if native_stored_inverse else signs_vertex[vertex, slot] * dc[edge] / area_vertex[vertex])
                delsq_vorticity[:, vertex] += factor * delsq_u[:, edge]
        for cell in range(ncells):
            for slot in range(int(counts[cell])):
                edge = int(edges_on_cell[cell, slot])
                factor = ((dtype.type(1.0) / area_cell[cell]) * dv[edge] * signs_cell[cell, slot]
                          if native_stored_inverse else signs_cell[cell, slot] * dv[edge] / area_cell[cell])
                delsq_divergence[:, cell] += factor * delsq_u[:, edge]

        factor4 = dtype.type(div_factor)
        for edge, (cell0, cell1) in enumerate(cells):
            vertex0, vertex1 = vertices[edge]
            scale = del4[edge] * h4
            r_dc = scale * factor4 * inv_dc[edge]
            r_dv = scale * inv_dv_limited[edge]
            diffusion = rho[:, edge] * (
                (delsq_divergence[:, cell1] - delsq_divergence[:, cell0]) * r_dc
                - (delsq_vorticity[:, vertex1] - delsq_vorticity[:, vertex0]) * r_dv
            )
            tendency[:, edge] -= diffusion

    return MomentumFilterResult(
        tendency=tendency,
        delsq_u=delsq_u,
        delsq_divergence=delsq_divergence,
        delsq_vorticity=delsq_vorticity,
    )


def vertical_momentum_horizontal_filter_tendency(
    mesh: object,
    *,
    vertical_velocity: object,
    rho_edge: object,
    kdiff: object,
    h_mom_eddy_visc4: float,
    mesh_scaling_del2: object | None = None,
    mesh_scaling_del4: object | None = None,
    native_stored_inverse: bool = False,
) -> CellFilterResult:
    """Return horizontal ``w`` filtering from frozen lines 5076-5133."""

    w = _float_field("vertical_velocity", vertical_velocity, ndim=2)
    rho = _same_dtype(w, "rho_edge", rho_edge, ndim=2)
    diffusivity = _same_dtype(w, "kdiff", kdiff, ndim=2)
    nlev = rho.shape[0]
    ncells = diffusivity.shape[1]
    if w.shape != (nlev + 1, ncells) or diffusivity.shape[0] != nlev:
        raise ValueError("w/rho_edge/kdiff dimensions are inconsistent")
    cells = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    if cells.shape != (rho.shape[1], 2):
        raise ValueError("cellsOnEdge does not match rho_edge")
    dc = np.asarray(_mesh_array(mesh, "dcEdge"), dtype=w.dtype)
    dv = np.asarray(_mesh_array(mesh, "dvEdge"), dtype=w.dtype)
    area = np.asarray(_mesh_array(mesh, "areaCell"), dtype=w.dtype)
    edges = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    signs = edge_sign_on_cell(mesh, dtype=w.dtype)
    if np.any(dc <= 0.0) or np.any(area <= 0.0):
        raise ValueError("dcEdge and areaCell must be positive")

    if mesh_scaling_del2 is None or mesh_scaling_del4 is None:
        scaling = compute_mesh_mixing_scaling(mesh, dtype=w.dtype)
        if mesh_scaling_del2 is None:
            mesh_scaling_del2 = scaling.del2
        if mesh_scaling_del4 is None:
            mesh_scaling_del4 = scaling.del4
    del2 = np.asarray(mesh_scaling_del2, dtype=w.dtype)
    del4 = np.asarray(mesh_scaling_del4, dtype=w.dtype)
    if del2.shape != (rho.shape[1],) or del4.shape != (rho.shape[1],):
        raise ValueError("mesh mixing scale fields must have shape (nEdges,)")

    h4_value = _finite_scalar("config_visc4_2dsmag", h_mom_eddy_visc4)
    if h4_value < 0.0:
        raise ConfigurationRefusal(
            "config_visc4_2dsmag",
            h_mom_eddy_visc4,
            "the derived fourth-order viscosity cannot be negative",
            "config_visc4_2dsmag>=0",
        )

    dtype = w.dtype
    half = dtype.type(0.5)
    quarter = dtype.type(0.25)
    delsq_w = np.zeros((nlev, ncells), dtype=dtype)
    tendency = np.zeros_like(w)
    for cell in range(ncells):
        inv_area = dtype.type(1.0) / area[cell]
        for slot in range(int(counts[cell])):
            edge = int(edges[cell, slot])
            cell0, cell1 = cells[edge]
            edge_sign = (
                half * inv_area * signs[cell, slot] * dv[edge] * (dtype.type(1.0) / dc[edge])
                if native_stored_inverse else half * inv_area * signs[cell, slot] * dv[edge] / dc[edge]
            )
            for level in range(1, nlev):
                flux = (
                    edge_sign
                    * (rho[level, edge] + rho[level - 1, edge])
                    * (w[level, cell1] - w[level, cell0])
                )
                delsq_w[level, cell] += flux
                sum_k = (
                    diffusivity[level, cell0]
                    + diffusivity[level, cell1]
                    + diffusivity[level - 1, cell0]
                    + diffusivity[level - 1, cell1]
                )
                increment = (flux * del2[edge] * quarter * sum_k
                             if native_stored_inverse else flux * del2[edge] * (quarter * sum_k))
                tendency[level, cell] += increment

    h4 = dtype.type(h4_value)
    if h4 > dtype.type(0.0):
        for cell in range(ncells):
            scale = h4 * (dtype.type(1.0) / area[cell]) if native_stored_inverse else h4 / area[cell]
            for slot in range(int(counts[cell])):
                edge = int(edges[cell, slot])
                cell0, cell1 = cells[edge]
                edge_sign = (
                    del4[edge]
                    * scale
                    * dv[edge]
                    * signs[cell, slot]
                    * (dtype.type(1.0) / dc[edge])
                    if native_stored_inverse else del4[edge] * scale * dv[edge] * signs[cell, slot] / dc[edge]
                )
                for level in range(1, nlev):
                    tendency[level, cell] -= edge_sign * (
                        delsq_w[level, cell1] - delsq_w[level, cell0]
                    )
    return CellFilterResult(tendency=tendency, laplacian=delsq_w)


def theta_horizontal_filter_tendency(
    mesh: object,
    *,
    theta_m: object,
    rho_edge: object,
    kdiff: object,
    h_theta_eddy_visc4: float,
    mesh_scaling_del2: object | None = None,
    mesh_scaling_del4: object | None = None,
    native_stored_inverse: bool = False,
) -> CellFilterResult:
    """Return horizontal theta filtering from frozen lines 5255-5304."""

    theta = _float_field("theta_m", theta_m, ndim=2)
    rho = _same_dtype(theta, "rho_edge", rho_edge, ndim=2)
    diffusivity = _same_dtype(theta, "kdiff", kdiff, ndim=2)
    if theta.shape != diffusivity.shape or rho.shape[0] != theta.shape[0]:
        raise ValueError("theta_m/rho_edge/kdiff dimensions are inconsistent")
    nlev, ncells = theta.shape
    cells = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    if cells.shape != (rho.shape[1], 2):
        raise ValueError("cellsOnEdge does not match rho_edge")
    dc = np.asarray(_mesh_array(mesh, "dcEdge"), dtype=theta.dtype)
    dv = np.asarray(_mesh_array(mesh, "dvEdge"), dtype=theta.dtype)
    area = np.asarray(_mesh_array(mesh, "areaCell"), dtype=theta.dtype)
    edges = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    signs = edge_sign_on_cell(mesh, dtype=theta.dtype)
    if np.any(dc <= 0.0) or np.any(area <= 0.0):
        raise ValueError("dcEdge and areaCell must be positive")

    if mesh_scaling_del2 is None or mesh_scaling_del4 is None:
        scaling = compute_mesh_mixing_scaling(mesh, dtype=theta.dtype)
        if mesh_scaling_del2 is None:
            mesh_scaling_del2 = scaling.del2
        if mesh_scaling_del4 is None:
            mesh_scaling_del4 = scaling.del4
    del2 = np.asarray(mesh_scaling_del2, dtype=theta.dtype)
    del4 = np.asarray(mesh_scaling_del4, dtype=theta.dtype)
    if del2.shape != (rho.shape[1],) or del4.shape != (rho.shape[1],):
        raise ValueError("mesh mixing scale fields must have shape (nEdges,)")

    h4_value = _finite_scalar("config_visc4_2dsmag", h_theta_eddy_visc4)
    if h4_value < 0.0:
        raise ConfigurationRefusal(
            "config_visc4_2dsmag",
            h_theta_eddy_visc4,
            "the derived fourth-order viscosity cannot be negative",
            "config_visc4_2dsmag>=0",
        )

    dtype = theta.dtype
    half = dtype.type(0.5)
    delsq_theta = np.zeros_like(theta)
    tendency = np.zeros_like(theta)
    for cell in range(ncells):
        inv_area = dtype.type(1.0) / area[cell]
        for slot in range(int(counts[cell])):
            edge = int(edges[cell, slot])
            cell0, cell1 = cells[edge]
            edge_sign = (inv_area * signs[cell, slot] * dv[edge] * (dtype.type(1.0) / dc[edge])
                         if native_stored_inverse else inv_area * signs[cell, slot] * dv[edge] / dc[edge])
            for level in range(nlev):
                flux = (
                    edge_sign
                    * (theta[level, cell1] - theta[level, cell0])
                    * rho[level, edge]
                )
                delsq_theta[level, cell] += flux
                tendency[level, cell] += (
                    flux
                    * half
                    * (diffusivity[level, cell0] + diffusivity[level, cell1])
                    * del2[edge]
                )

    h4 = dtype.type(h4_value)
    if h4 > dtype.type(0.0):
        for cell in range(ncells):
            scale = h4 * (dtype.type(1.0) / area[cell]) if native_stored_inverse else h4 / area[cell]
            for slot in range(int(counts[cell])):
                edge = int(edges[cell, slot])
                cell0, cell1 = cells[edge]
                edge_sign = (
                    del4[edge]
                    * scale
                    * dv[edge]
                    * signs[cell, slot]
                    * (dtype.type(1.0) / dc[edge])
                    if native_stored_inverse else del4[edge] * scale * dv[edge] * signs[cell, slot] / dc[edge]
                )
                for level in range(nlev):
                    tendency[level, cell] -= edge_sign * (
                        delsq_theta[level, cell1] - delsq_theta[level, cell0]
                    )
    return CellFilterResult(tendency=tendency, laplacian=delsq_theta)


def compute_dry_mixing_tendencies(
    mesh: object,
    *,
    normal_velocity: object,
    tangential_velocity: object,
    vertical_velocity: object,
    theta_m: object,
    rho_edge: object,
    divergence: object,
    vorticity: object,
    dt: float,
    config: MixingConfig | None = None,
) -> DryMixingTendencies:
    """Compute every horizontal filter increment used by the frozen JW branch."""

    cfg = MixingConfig() if config is None else config
    cfg.validate()
    coefficients = compute_smagorinsky_coefficients(
        mesh,
        normal_velocity,
        tangential_velocity,
        dt=dt,
        config=cfg,
    )
    dtype = coefficients.kdiff.dtype
    scaling = compute_mesh_mixing_scaling(
        mesh,
        config_h_ScaleWithMesh=cfg.config_h_ScaleWithMesh,
        dtype=dtype,
    )
    momentum = momentum_horizontal_filter_tendency(
        mesh,
        rho_edge=rho_edge,
        divergence=divergence,
        vorticity=vorticity,
        kdiff=coefficients.kdiff,
        h_mom_eddy_visc4=coefficients.h_mom_eddy_visc4,
        config_del4u_div_factor=cfg.config_del4u_div_factor,
        mesh_scaling_del2=scaling.del2,
        mesh_scaling_del4=scaling.del4,
    )
    w_filter = vertical_momentum_horizontal_filter_tendency(
        mesh,
        vertical_velocity=vertical_velocity,
        rho_edge=rho_edge,
        kdiff=coefficients.kdiff,
        h_mom_eddy_visc4=coefficients.h_mom_eddy_visc4,
        mesh_scaling_del2=scaling.del2,
        mesh_scaling_del4=scaling.del4,
    )
    theta_filter = theta_horizontal_filter_tendency(
        mesh,
        theta_m=theta_m,
        rho_edge=rho_edge,
        kdiff=coefficients.kdiff,
        h_theta_eddy_visc4=coefficients.h_theta_eddy_visc4,
        mesh_scaling_del2=scaling.del2,
        mesh_scaling_del4=scaling.del4,
    )
    return DryMixingTendencies(
        kdiff=coefficients.kdiff,
        h_mom_eddy_visc4=coefficients.h_mom_eddy_visc4,
        h_theta_eddy_visc4=coefficients.h_theta_eddy_visc4,
        tend_u_euler=momentum.tendency,
        tend_w_euler=w_filter.tendency,
        tend_theta_euler=theta_filter.tendency,
        delsq_u=momentum.delsq_u,
        delsq_divergence=momentum.delsq_divergence,
        delsq_vorticity=momentum.delsq_vorticity,
        delsq_w=w_filter.laplacian,
        delsq_theta=theta_filter.laplacian,
    )


def apply_saved_euler_mixing(
    tendency_u: object,
    tendency_w: object,
    tendency_theta: object,
    saved: DryMixingTendencies,
) -> AppliedMixingTendencies:
    """Add saved RK1 filter increments to one RK stage's dry tendencies."""

    u = _float_field("tendency_u", tendency_u, ndim=2)
    w = _same_dtype(u, "tendency_w", tendency_w, ndim=2)
    theta = _same_dtype(u, "tendency_theta", tendency_theta, ndim=2)
    if u.shape != saved.tend_u_euler.shape:
        raise ValueError("tendency_u shape does not match saved tend_u_euler")
    if w.shape != saved.tend_w_euler.shape:
        raise ValueError("tendency_w shape does not match saved tend_w_euler")
    if theta.shape != saved.tend_theta_euler.shape:
        raise ValueError("tendency_theta shape does not match saved tend_theta_euler")
    if (
        saved.tend_u_euler.dtype != u.dtype
        or saved.tend_w_euler.dtype != u.dtype
        or saved.tend_theta_euler.dtype != u.dtype
    ):
        raise TypeError("saved Euler tendency dtype does not match stage tendency dtype")
    return AppliedMixingTendencies(
        u=u + saved.tend_u_euler,
        w=w + saved.tend_w_euler,
        theta=theta + saved.tend_theta_euler,
    )


def capture_rtheta_pp_old(rtheta_pp: object, *, small_step: int) -> FloatArray:
    """Capture the divergence-damping reference at frozen lines 2409-2417."""

    current = _float_field("rtheta_pp", rtheta_pp, ndim=2)
    if small_step < 1:
        raise ValueError("small_step is one-based and must be positive")
    if small_step == 1:
        return np.zeros_like(current)
    return current.copy()


def divergence_damping_3d(
    mesh: object,
    rho_u_perturbation: object,
    theta_m: object,
    rtheta_pp: object,
    rtheta_pp_old: object,
    *,
    dts: float,
    config_smdiv: float = 0.1,
    config_len_disp: float = 0.0,
    spec_zone_mask_edge: object | None = None,
    n_cells_solve: int | None = None,
    native_stored_inverse: bool = False,
) -> FloatArray:
    """Apply the scaled 3-D divergence damping update (lines 2532-2601).

    The returned array is a copy of ``rho_u_perturbation``.  The caller must
    capture ``rtheta_pp_old`` immediately before each acoustic update using
    :func:`capture_rtheta_pp_old`, then call this routine after that update and
    after the frozen ``rtheta_pp`` halo exchange.
    """

    ru_p = _float_field("rho_u_perturbation", rho_u_perturbation, ndim=2)
    theta = _same_dtype(ru_p, "theta_m", theta_m, ndim=2)
    current = _same_dtype(ru_p, "rtheta_pp", rtheta_pp, ndim=2)
    previous = _same_dtype(ru_p, "rtheta_pp_old", rtheta_pp_old, ndim=2)
    if theta.shape != current.shape or current.shape != previous.shape:
        raise ValueError("theta_m/rtheta_pp/rtheta_pp_old cell shapes differ")
    if ru_p.shape[0] != current.shape[0]:
        raise ValueError("rho_u_perturbation vertical dimension differs")
    timestep = float(dts)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("dts must be finite and positive")
    smdiv = _finite_scalar("config_smdiv", config_smdiv)
    if smdiv < 0.0:
        raise ConfigurationRefusal(
            "config_smdiv",
            config_smdiv,
            "the divergence damping coefficient must be non-negative",
            "config_smdiv>=0",
        )

    cells = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    if cells.shape != (ru_p.shape[1], 2):
        raise ValueError("cellsOnEdge does not match rho_u_perturbation")
    ncells = current.shape[1]
    if np.any((cells < 0) | (cells >= ncells)):
        raise ValueError("cellsOnEdge contains an invalid cell")
    solve_count = ncells if n_cells_solve is None else int(n_cells_solve)
    if solve_count < 0 or solve_count > ncells:
        raise ValueError("n_cells_solve must lie in [0, nCells]")

    if spec_zone_mask_edge is None:
        mesh_mask = _optional_mesh_array(mesh, "specZoneMaskEdge")
        if mesh_mask is None:
            mask = np.zeros(ru_p.shape[1], dtype=ru_p.dtype)
        else:
            mask = np.asarray(mesh_mask, dtype=ru_p.dtype)
    else:
        mask = np.asarray(spec_zone_mask_edge, dtype=ru_p.dtype)
    if mask.shape != (ru_p.shape[1],) or np.any(~np.isfinite(mask)):
        raise ValueError("specZoneMaskEdge must be a finite edge field")

    dtype = ru_p.dtype
    length = resolve_config_len_disp(mesh, config_len_disp, dtype=dtype)
    base_coefficient = dtype.type(2.0) * dtype.type(smdiv) * length
    # Native time_integration.F:4153-4154 stores rdts before multiplying.
    coefficient = (
        base_coefficient * (dtype.type(1.0) / dtype.type(timestep))
        if native_stored_inverse else base_coefficient / dtype.type(timestep)
    )
    one = dtype.type(1.0)
    out = ru_p.copy()
    delta = current - previous
    for edge, (cell0, cell1) in enumerate(cells):
        if cell0 >= solve_count and cell1 >= solve_count:
            continue
        denominator = theta[:, cell0] + theta[:, cell1]
        if np.any(denominator == dtype.type(0.0)):
            raise FloatingPointError(
                f"zero theta_m pair denominator on divergence-damped edge {edge}"
            )
        # divCell2-divCell1 with divCell=-(rtheta_pp-rtheta_pp_old).
        divergence_difference = delta[:, cell0] - delta[:, cell1]
        out[:, edge] += (
            coefficient
            * divergence_difference
            * (one - mask[edge])
            / denominator
        )
    return out


__all__ = [
    "AppliedMixingTendencies",
    "CellFilterResult",
    "DryMixingTendencies",
    "MeshMixingScaling",
    "MixingConfig",
    "MomentumFilterResult",
    "SmagorinskyCoefficients",
    "apply_saved_euler_mixing",
    "capture_rtheta_pp_old",
    "compute_dry_mixing_tendencies",
    "compute_mesh_mixing_scaling",
    "compute_smagorinsky_coefficients",
    "divergence_damping_3d",
    "momentum_horizontal_filter_tendency",
    "resolve_config_len_disp",
    "theta_horizontal_filter_tendency",
    "vertical_momentum_horizontal_filter_tendency",
]
