"""Fused CUDA state recovery for the admitted MPAS logical array layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..cuda_fp32 import CUDA_FTZ_HELPERS
from .containers import DeviceAtmosphere, terrain_slot_tables
from .runtime import KernelCache, KernelTiming, launch_checked, launch_timed


RECOVERY_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
template <typename T> __device__ __forceinline__ T mp_pow(T x, T y);
template <> __device__ __forceinline__ float mp_pow<float>(float x, float y) { return powf(x, y); }
template <> __device__ __forceinline__ double mp_pow<double>(double x, double y) { return pow(x, y); }
template <typename T> __device__ __forceinline__ T mp_add(T x, T y) { return x + y; }
template <> __device__ __forceinline__ float mp_add<float>(float x, float y) { return mpas_add(x, y); }
template <typename T> __device__ __forceinline__ T mp_mul(T x, T y) { return x * y; }
template <> __device__ __forceinline__ float mp_mul<float>(float x, float y) { return mpas_mul(x, y); }
template <typename T> __device__ __forceinline__ T mp_div(T x, T y) { return x / y; }
template <> __device__ __forceinline__ float mp_div<float>(float x, float y) { return mpas_div(x, y); }
template <typename T> __device__ __forceinline__ T mp_sign(T x);
template <> __device__ __forceinline__ float mp_sign<float>(float x) { return mpas_copysign(1.0f, x); }
template <> __device__ __forceinline__ double mp_sign<double>(double x) { return copysign(1.0, x); }

template <typename T>
__device__ __forceinline__ void pressure_point(
    const T rho, const T rtheta, const T rho_base, const T rtheta_base,
    const T exner_base, const T zz, const T rgas, const T cp, const T p0,
    T &theta, T &exner, T &pressure, T &rho_p, T &rtheta_p, T &pressure_p) {
  theta = rtheta / rho;
  const T argument = zz * (rgas / p0) * rtheta;
  exner = mp_pow<T>(argument, rgas / (cp - rgas));
  pressure = zz * rgas * rtheta * exner;
  rho_p = rho - rho_base;
  rtheta_p = rtheta - rtheta_base;
  pressure_p = zz * rgas * (
      exner * rtheta_p + rtheta_base * (exner - exner_base));
}

/* Every kernel here walks a flat (level, owner) element index, owner
   fastest, in a grid-stride loop; each element runs the former per-owner
   loop body for its level unchanged (the terrain w column keeps its
   ascending edge-slot accumulation per level), so the launch geometry is
   not in the arithmetic.  recover_state launches one thread per element. */
#define RECOVERY_ELEMENT_LOOP(total) \
    for (int element = blockDim.x * blockIdx.x + threadIdx.x; \
         element < (total); element += gridDim.x * blockDim.x)

#define DECLARE_PRESSURE_KERNEL(NAME, T) \
extern "C" __global__ void NAME( \
    const T* rho, const T* rtheta, const T* rho_base, const T* rtheta_base, \
    const T* exner_base, const T* zz, T* theta, T* exner, T* pressure, \
    T* rho_p, T* rtheta_p, T* pressure_p, T rgas, T cp, T p0, \
    int nlev, int ncells) { \
  RECOVERY_ELEMENT_LOOP(nlev * ncells) { \
    const int index = element; \
    pressure_point<T>( \
        rho[index], rtheta[index], rho_base[index], rtheta_base[index], \
        exner_base[index], zz[index], rgas, cp, p0, theta[index], exner[index], \
        pressure[index], rho_p[index], rtheta_p[index], pressure_p[index]); \
  } \
}

DECLARE_PRESSURE_KERNEL(recover_pressure_f32, float)
DECLARE_PRESSURE_KERNEL(recover_pressure_f64, double)

#define DECLARE_EDGE_KERNEL(NAME, T) \
extern "C" __global__ void NAME( \
    const T* rho, const T* ru, const int* cells_on_edge, T* u, \
    int nlev, int ncells, int nedges) { \
  RECOVERY_ELEMENT_LOOP(nlev * nedges) { \
    const int edge = element % nedges; \
    const int level = element / nedges; \
    const int index = level * nedges + edge; \
    const int cell0 = cells_on_edge[2 * edge]; \
    const int cell1 = cells_on_edge[2 * edge + 1]; \
    const T denominator = mp_add<T>( \
        rho[level * ncells + cell0], rho[level * ncells + cell1]); \
    u[index] = mp_div<T>(mp_mul<T>(T(2), ru[index]), denominator); \
  } \
}

DECLARE_EDGE_KERNEL(recover_edge_velocity_f32, float)
DECLARE_EDGE_KERNEL(recover_edge_velocity_f64, double)

#define DECLARE_FLAT_W_KERNEL(NAME, T) \
extern "C" __global__ void NAME( \
    const T* rho, const T* rw, const T* zz, const T* fzm, const T* fzp, \
    T* w, int nlev, int ncells) { \
  RECOVERY_ELEMENT_LOOP((nlev + 1) * ncells) { \
    const int cell = element % ncells; \
    const int level = element / ncells; \
    const int index = level * ncells + cell; \
    if (level == 0 || level == nlev) { \
      w[index] = T(0); \
    } else { \
      const T metric = mp_add<T>( \
          mp_mul<T>(fzm[level], zz[level * ncells + cell]), \
          mp_mul<T>(fzp[level], zz[(level - 1) * ncells + cell])); \
      const T density = mp_add<T>( \
          mp_mul<T>(fzm[level], rho[level * ncells + cell]), \
          mp_mul<T>(fzp[level], rho[(level - 1) * ncells + cell])); \
      w[index] = mp_div<T>(mp_div<T>(rw[index], metric), density); \
    } \
  } \
}

DECLARE_FLAT_W_KERNEL(recover_flat_w_f32, float)
DECLARE_FLAT_W_KERNEL(recover_flat_w_f64, double)

template <typename T>
__device__ __forceinline__ void terrain_w_element(
    const int cell, const int level, const T* rho, const T* ru, const T* rw,
    const T* zz, const T* fzm, const T* fzp, const int* edges_on_cell,
    const int* n_edges_on_cell, const T* edge_sign_on_cell,
    const T* zb_cell, const T* zb3_cell, T* w, int nlev, int ncells,
    int nedges, int max_edges, T cf1, T cf2, T cf3) {
  /* The per-cell column, one level at a time: the value at (level, cell)
     starts where the column loop started it, takes the edge-slot
     contributions in the same ascending slot order, and is divided by the
     same interpolated density last, so each level's bits are the column
     loop's bits for that level. */
  const int index = level * ncells + cell;
  if (level == nlev) {
    w[index] = T(0);
    return;
  }
  const int slots = n_edges_on_cell[cell];
  if (level == 0) {
    T value = T(0);
    for (int slot = 0; slot < slots; ++slot) {
      const int edge = edges_on_cell[cell * max_edges + slot];
      const T cell_sign = edge_sign_on_cell[cell * max_edges + slot];
      const T flux = mp_add<T>(mp_add<T>(
          mp_mul<T>(cf1, ru[edge]), mp_mul<T>(cf2, ru[nedges + edge])),
          mp_mul<T>(cf3, ru[2 * nedges + edge]));
      const int metric_index = cell * max_edges + slot;
      value = mp_add<T>(value, mp_mul<T>(mp_mul<T>(cell_sign,
          mp_add<T>(zb_cell[metric_index],
              mp_mul<T>(mp_sign<T>(flux), zb3_cell[metric_index]))), flux));
    }
    const T bottom_density = mp_add<T>(mp_add<T>(
        mp_mul<T>(cf1, rho[cell]), mp_mul<T>(cf2, rho[ncells + cell])),
        mp_mul<T>(cf3, rho[2 * ncells + cell]));
    w[index] = mp_div<T>(value, bottom_density);
    return;
  }
  const T metric = mp_add<T>(
      mp_mul<T>(fzm[level], zz[level * ncells + cell]),
      mp_mul<T>(fzp[level], zz[(level - 1) * ncells + cell]));
  T value = mp_div<T>(rw[index], metric);
  for (int slot = 0; slot < slots; ++slot) {
    const int edge = edges_on_cell[cell * max_edges + slot];
    const T cell_sign = edge_sign_on_cell[cell * max_edges + slot];
    const T flux = mp_add<T>(
        mp_mul<T>(fzm[level], ru[level * nedges + edge]),
        mp_mul<T>(fzp[level], ru[(level - 1) * nedges + edge]));
    const int metric_index = (level * ncells + cell) * max_edges + slot;
    value = mp_add<T>(value,
        mp_mul<T>(mp_mul<T>(cell_sign,
            mp_add<T>(zb_cell[metric_index],
                mp_mul<T>(mp_sign<T>(flux), zb3_cell[metric_index]))),
            flux));
  }
  const T density = mp_add<T>(
      mp_mul<T>(fzm[level], rho[level * ncells + cell]),
      mp_mul<T>(fzp[level], rho[(level - 1) * ncells + cell]));
  w[index] = mp_div<T>(value, density);
}

#define DECLARE_TERRAIN_W_KERNEL(NAME, T) \
extern "C" __global__ void NAME( \
    const T* rho, const T* ru, const T* rw, const T* zz, const T* fzm, \
    const T* fzp, const int* edges_on_cell, const int* n_edges_on_cell, \
    const T* edge_sign_on_cell, const T* zb_cell, const T* zb3_cell, T* w, \
    int nlev, int ncells, int nedges, int max_edges, T cf1, T cf2, T cf3) { \
  RECOVERY_ELEMENT_LOOP((nlev + 1) * ncells) { \
    terrain_w_element<T>( \
        element % ncells, element / ncells, rho, ru, rw, zz, fzm, fzp, \
        edges_on_cell, n_edges_on_cell, edge_sign_on_cell, zb_cell, \
        zb3_cell, w, nlev, ncells, nedges, max_edges, cf1, cf2, cf3); \
  } \
}

DECLARE_TERRAIN_W_KERNEL(recover_terrain_w_f32, float)
DECLARE_TERRAIN_W_KERNEL(recover_terrain_w_f64, double)
"""


@dataclass(frozen=True, slots=True)
class DeviceRecoveredState:
    theta_m: Any
    exner: Any
    pressure: Any
    density_perturbation: Any
    rho_theta_perturbation: Any
    pressure_perturbation: Any
    normal_velocity: Any
    vertical_velocity: Any
    #: ``None`` per kernel on the step path (``timing_repeats=0``); a
    #: :class:`KernelTiming` per kernel when a caller asked to be timed.
    timings: dict[str, KernelTiming | None]

    def to_host(self) -> dict[str, np.ndarray]:
        import cupy as cp

        fields = {}
        for name in (
            "theta_m",
            "exner",
            "pressure",
            "density_perturbation",
            "rho_theta_perturbation",
            "pressure_perturbation",
            "normal_velocity",
            "vertical_velocity",
        ):
            value = getattr(self, name)
            if value is None:
                raise ValueError(
                    "recover_state(include_pressure=False) did not compute "
                    f"{name}; to_host() would publish a field set that silently "
                    "omits the pressure diagnostics"
                )
            fields[name] = cp.asnumpy(value)
        return fields


def _suffix(dtype: np.dtype[Any]) -> str:
    selected = np.dtype(dtype)
    if selected == np.dtype(np.float32):
        return "f32"
    if selected == np.dtype(np.float64):
        return "f64"
    raise TypeError("CUDA recovery supports only float32 or float64")


def recover_state(
    atmosphere: DeviceAtmosphere,
    *,
    cache: KernelCache | None = None,
    rgas: float = 287.0,
    cp: float = 1004.5,
    reference_pressure: float = 100_000.0,
    warmup: int = 0,
    timing_repeats: int = 0,
    include_pressure: bool = True,
) -> DeviceRecoveredState:
    """Recover pressure diagnostics and physical u/w without leaving device.

    ``timing_repeats=0`` (the default, and the step path) launches each
    kernel exactly once through :func:`launch_checked`: no event, no
    synchronize, ``timings`` holds ``None`` per kernel.  A positive count is
    the benchmark's timed path through :func:`launch_timed`, which launches
    each kernel ``1 + warmup + timing_repeats`` times and closes the first
    and the last ``timing_repeats`` with an event synchronize.  The second
    launch recomputes the same outputs from the same inputs, so the two
    paths publish the same bytes; what differed was 36 host stalls and
    10.7 ms of repeated device work per forecast step
    (``evidence/hex-perf-profile-2026-09-13``).

    ``include_pressure=False`` computes only ``normal_velocity`` and
    ``vertical_velocity`` and leaves the six cell-centred pressure diagnostics
    ``None``.  The pressure kernel writes nothing but those six arrays, so a
    caller that already holds its own theta/exner/rho_p/rtheta_p/pressure_p --
    ``CudaDeviceDriver._recover_candidate`` does, from ``recover_cells_f32`` --
    was allocating and filling six whole cell fields per RK stage and dropping
    every one of them on return.
    """

    import cupy as cupy

    state = atmosphere.state
    mesh = atmosphere.mesh
    vertical = atmosphere.vertical
    reference = atmosphere.reference
    if atmosphere.terrain is not None and int(vertical.n_vert_levels) < 3:
        raise ValueError(
            "terrain recovery requires nVertLevels>=3 for cf1/cf2/cf3"
        )
    dtype = np.dtype(state.dtype)
    if any(
        np.dtype(item.dtype) != dtype
        for item in (mesh, vertical, reference, atmosphere.saved)
    ):
        raise TypeError("CUDA atmosphere containers must share one floating precision")
    if mesh.index_dtype != np.dtype(np.int32):
        raise TypeError("CUDA recovery kernels require DeviceMesh index_dtype=int32")
    selected_cache = KernelCache() if cache is None else cache
    suffix = _suffix(dtype)
    pressure_name = f"recover_pressure_{suffix}"
    edge_name = f"recover_edge_velocity_{suffix}"
    vertical_name = (
        f"recover_terrain_w_{suffix}"
        if atmosphere.terrain is not None
        else f"recover_flat_w_{suffix}"
    )
    # The whole source module is compiled and cached under one module_key, so
    # naming all three kernels here keeps the cache key -- and the loaded cubin
    # image -- byte-identical in both modes.
    kernels = selected_cache.raw_kernels(
        (pressure_name, edge_name, vertical_name),
        RECOVERY_CUDA_SOURCE,
        module_key="hexcore.cuda_backend.recovery",
    )
    theta = exner = pressure = None
    density_p = rtheta_p = pressure_p = None
    pressure_timing = None
    # Allocation ORDER through the default (include_pressure=True) path is
    # unchanged: the frozen phase-1 longwave seam is sensitive to device pool
    # contents, so the six pressure arrays keep their place ahead of the two
    # velocity arrays.
    if include_pressure:
        theta = cupy.empty_like(state.rho)
        exner = cupy.empty_like(state.rho)
        pressure = cupy.empty_like(state.rho)
        density_p = cupy.empty_like(state.rho)
        rtheta_p = cupy.empty_like(state.rho)
        pressure_p = cupy.empty_like(state.rho)
    normal = cupy.empty_like(state.rho_u)
    vertical_velocity = cupy.empty_like(state.rho_w)
    block = (256,)
    scalar = np.float32 if dtype == np.dtype(np.float32) else np.float64
    if timing_repeats < 0:
        raise ValueError("timing_repeats must be zero (untimed) or positive")

    def launch(kernel: Any, grid: tuple[int, ...], args: tuple[Any, ...]) -> KernelTiming | None:
        if timing_repeats == 0:
            launch_checked(kernel, grid, block, args)
            return None
        return launch_timed(kernel, grid, block, args, warmup=warmup, repeats=timing_repeats)

    if include_pressure:
        pressure_timing = launch(
            kernels[pressure_name],
            ((int(vertical.n_vert_levels) * mesh.n_cells + block[0] - 1)
             // block[0],),
            (
                state.rho,
                state.rho_theta,
                reference.rho_base,
                reference.rho_theta_base,
                reference.exner_base,
                vertical.zz,
                theta,
                exner,
                pressure,
                density_p,
                rtheta_p,
                pressure_p,
                scalar(rgas),
                scalar(cp),
                scalar(reference_pressure),
                np.int32(vertical.n_vert_levels),
                np.int32(mesh.n_cells),
            ),
        )
    edge_timing = launch(
        kernels[edge_name],
        ((int(vertical.n_vert_levels) * mesh.n_edges + block[0] - 1)
         // block[0],),
        (
            state.rho,
            state.rho_u,
            mesh.cells_on_edge,
            normal,
            np.int32(vertical.n_vert_levels),
            np.int32(mesh.n_cells),
            np.int32(mesh.n_edges),
        ),
    )
    if atmosphere.terrain is None:
        vertical_args = (
            state.rho,
            state.rho_w,
            vertical.zz,
            vertical.fzm,
            vertical.fzp,
            vertical_velocity,
            np.int32(vertical.n_vert_levels),
            np.int32(mesh.n_cells),
        )
        vertical_count = (int(vertical.n_vert_levels) + 1) * mesh.n_cells
    else:
        terrain = atmosphere.terrain
        # The kernel indexes edgesOnCell, its sign and zb_cell with one slot
        # stride; a compacted zb_cell is read through the matching tables.
        slot_edges, slot_signs, slot_stride = terrain_slot_tables(mesh, terrain)
        vertical_args = (
            state.rho,
            state.rho_u,
            state.rho_w,
            vertical.zz,
            vertical.fzm,
            vertical.fzp,
            slot_edges,
            mesh.n_edges_on_cell,
            slot_signs,
            terrain.zb_cell,
            terrain.zb3_cell,
            vertical_velocity,
            np.int32(vertical.n_vert_levels),
            np.int32(mesh.n_cells),
            np.int32(mesh.n_edges),
            np.int32(slot_stride),
            scalar(vertical.cf1),
            scalar(vertical.cf2),
            scalar(vertical.cf3),
        )
        vertical_count = (int(vertical.n_vert_levels) + 1) * mesh.n_cells
    vertical_timing = launch(
        kernels[vertical_name],
        ((vertical_count + block[0] - 1) // block[0],),
        vertical_args,
    )
    return DeviceRecoveredState(
        theta_m=theta,
        exner=exner,
        pressure=pressure,
        density_perturbation=density_p,
        rho_theta_perturbation=rtheta_p,
        pressure_perturbation=pressure_p,
        normal_velocity=normal,
        vertical_velocity=vertical_velocity,
        timings=(
            {
                "pressure": pressure_timing,
                "normal_velocity": edge_timing,
                "vertical_velocity": vertical_timing,
            }
            if include_pressure
            else {
                "normal_velocity": edge_timing,
                "vertical_velocity": vertical_timing,
            }
        ),
    )


__all__ = [
    "DeviceRecoveredState",
    "RECOVERY_CUDA_SOURCE",
    "recover_state",
]
