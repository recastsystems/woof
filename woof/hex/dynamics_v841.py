"""Additive MPAS-A v8.4.1 dry dynamics surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset
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


def precomputed_mesh_inverse_v841(
    mesh: object,
    field_name: str,
    dtype: np.dtype[Any],
) -> FloatArray:
    """Reproduce an RKIND mesh inverse initialized before the dycore step.

    MPAS stores ``invAreaCell``/``invAreaTriangle``/``invDcEdge``/
    ``invDvEdge`` in RKIND.  Geometry may arrive from NetCDF in a wider type,
    so source order is cast to RKIND first and reciprocal second.
    """

    value = _mesh_array(mesh, field_name).astype(dtype, copy=False)
    if value.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if np.any(value == dtype.type(0.0)):
        raise ValueError(f"{field_name} must be nonzero")
    return np.reciprocal(value)


@dataclass(frozen=True, slots=True)
class V841ReferenceWindProfiles:
    """Released one-dimensional ``u_init`` and ``v_init`` reference winds."""

    u_init: FloatArray
    v_init: FloatArray

    def validate(
        self,
        n_vert_levels: int,
        dtype: np.dtype[Any] | None = None,
    ) -> None:
        for name in self.__slots__:
            value = np.asarray(getattr(self, name))
            if value.shape != (n_vert_levels,):
                raise ConfigurationRefusal(
                    name,
                    f"shape={value.shape}",
                    "v8.4.1 perturbation Coriolis requires one reference value per level",
                    f"{name} with shape ({n_vert_levels},)",
                )
            if value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
                raise TypeError(f"{name} must use float32 or float64 RKIND storage")
            if dtype is not None and value.dtype != dtype:
                raise TypeError(f"{name} dtype {value.dtype} != state dtype {dtype}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        if np.asarray(self.u_init).dtype != np.asarray(self.v_init).dtype:
            raise TypeError("u_init and v_init must share one RKIND dtype")


def _native_profile(variable: Any, name: str) -> FloatArray:
    value = np.asarray(variable[:])
    if value.ndim == 1:
        return value.copy()
    squeezed = np.squeeze(value)
    if squeezed.ndim != 1:
        raise ConfigurationRefusal(
            name,
            f"shape={value.shape}",
            "the native reference wind must have only singleton record dimensions",
            f"{name}(nVertLevels) or {name}(Time,nVertLevels) with one Time record",
        )
    return np.asarray(squeezed).copy()


def load_v841_reference_wind_profiles(
    path: str | Path,
    *,
    n_vert_levels: int | None = None,
) -> V841ReferenceWindProfiles:
    """Load required v8.4.1 reference winds, refusing missing native fields."""

    source = Path(path).expanduser().resolve(strict=True)
    with Dataset(source, "r") as dataset:
        missing = [name for name in ("u_init", "v_init") if name not in dataset.variables]
        if missing:
            name = missing[0]
            raise ConfigurationRefusal(
                name,
                None,
                "v8.4.1 perturbation Coriolis consumes the initialized reference profile",
                f"a native file containing {name}(nVertLevels)",
            )
        result = V841ReferenceWindProfiles(
            u_init=_native_profile(dataset.variables["u_init"], "u_init"),
            v_init=_native_profile(dataset.variables["v_init"], "v_init"),
        )
    levels = result.u_init.size if n_vert_levels is None else n_vert_levels
    result.validate(levels)
    return result


def vector_invariant_momentum_tendency_v841(
    mesh: object,
    *,
    normal_velocity: FloatArray,
    rho_edge: FloatArray,
    pv_edge: FloatArray,
    kinetic_energy: FloatArray,
    horizontal_divergence: FloatArray,
    reference_wind: V841ReferenceWindProfiles,
    initial_tendency: FloatArray | None = None,
) -> FloatArray:
    """Add the released unconditional perturbation-Coriolis subtraction.

    The second source-order neighbor loop deliberately uses the neighbor
    edge's angle and the target edge's Coriolis parameter.
    """

    u = np.asarray(normal_velocity)
    if u.ndim != 2 or u.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("normal_velocity must be a float32/float64 (level, edge) array")
    if np.shape(rho_edge) != u.shape or np.shape(pv_edge) != u.shape:
        raise ValueError("rho_edge and pv_edge must match velocity")
    reference_wind.validate(u.shape[0], u.dtype)
    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    edges_on_edge = _mesh_array(mesh, "edgesOnEdge").astype(np.int64, copy=False)
    counts = _mesh_array(mesh, "nEdgesOnEdge").astype(np.int64, copy=False)
    weights = _mesh_array(mesh, "weightsOnEdge").astype(u.dtype, copy=False)
    angle_edge = _mesh_array(mesh, "angleEdge").astype(u.dtype, copy=False)
    f_edge = _mesh_array(mesh, "fEdge").astype(u.dtype, copy=False)
    inv_dc = precomputed_mesh_inverse_v841(mesh, "dcEdge", u.dtype)
    u_init = np.asarray(reference_wind.u_init)
    v_init = np.asarray(reference_wind.v_init)
    half = u.dtype.type(0.5)
    result = np.zeros_like(u)
    if initial_tendency is not None and (np.shape(initial_tendency) != u.shape or np.asarray(initial_tendency).dtype != u.dtype):
        raise ValueError("initial_tendency must match velocity shape and dtype")
    # Elements are independent; retain the source's neighbor-slot accumulation
    # order while evaluating every edge and level together. No parallel sum
    # or reassociation is allowed here.
    q = np.zeros_like(u)
    # NumPy's scalar and SIMD trigonometric kernels can round differently.
    # Evaluate each angle once with the original scalar kernel, rather than
    # once per level and neighbor, including for nonzero reference winds.
    cos_angle = np.asarray([np.cos(angle) for angle in angle_edge], dtype=u.dtype)
    sin_angle = np.asarray([np.sin(angle) for angle in angle_edge], dtype=u.dtype)
    for slot in range(edges_on_edge.shape[1]):
        active = np.flatnonzero(counts > slot)
        neighbor = edges_on_edge[active, slot]
        work_pv = half * (pv_edge[:, active] + pv_edge[:, neighbor])
        q[:, active] = q[:, active] + weights[active, slot] * u[:, neighbor] * work_pv
    for slot in range(edges_on_edge.shape[1]):
        active = np.flatnonzero(counts > slot)
        neighbor = edges_on_edge[active, slot]
        reference_u = (u_init[:, None] * cos_angle[neighbor]
                       - v_init[:, None] * sin_angle[neighbor])
        q[:, active] = q[:, active] - weights[active, slot] * reference_u * f_edge[active]
    cell0, cell1 = cells.T
    horizontal = rho_edge * (q - (kinetic_energy[:, cell1] - kinetic_energy[:, cell0]) * inv_dc)
    if initial_tendency is not None:
        horizontal = initial_tendency + horizontal
    result[:] = horizontal - u * half * (horizontal_divergence[:, cell0] + horizontal_divergence[:, cell1])
    return result


__all__ = [
    "V841ReferenceWindProfiles",
    "load_v841_reference_wind_profiles",
    "precomputed_mesh_inverse_v841",
    "vector_invariant_momentum_tendency_v841",
]
