"""Additive v8.4.1 CUDA dynamics kernels and split-flux reductions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .cuda_backend.containers import require_resident_array
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_v841 import CudaV841Context


_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
/* Level-independent kernels walk a flat (level, owner) element index, owner
   fastest, in a grid-stride loop; each element runs the former per-owner
   loop body for its level unchanged, so the launch geometry is not in the
   arithmetic.  The forecast launches one thread per element. */
#define DYNAMICS_ELEMENT_LOOP(total) \
    for (int element = blockDim.x * blockIdx.x + threadIdx.x; \
         element < (total); element += gridDim.x * blockDim.x)

extern "C" __global__ void reference_wind_edge_v841_f32(
    const int nlev, const int nedges, const float *angle_edge,
    const float *u_init, const float *v_init, float *reference_u)
{
    /* The perturbation-Coriolis reference wind at every (level, edge):
       u_init(k)*cos(angleEdge) - v_init(k)*sin(angleEdge), the expression
       vector_momentum_v841_f32 evaluated for every neighbour of every edge
       at every level and every RK stage, now evaluated once per run in this
       translation unit -- the same cosf/sinf, the same operands, the same
       operation order -- and gathered by the momentum kernel. */
    DYNAMICS_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        reference_u[E2(k, edge, nedges)] = mpas_sub(
            mpas_mul(u_init[k], cosf(angle_edge[edge])),
            mpas_mul(v_init[k], sinf(angle_edge[edge])));
    }
}

extern "C" __global__ void vector_momentum_v841_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges2,
    const float *u, const float *rho_edge, const float *pv_edge,
    const float *kinetic, const float *mass_divergence,
    const int *cells_on_edge, const int *edges_on_edge,
    const int *n_edges_on_edge, const float *weights_on_edge,
    const float *inv_dc_edge, const float *reference_u, const float *f_edge,
    const float *initial_tendency, const int has_initial,
    float *result)
{
    DYNAMICS_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const int index = E2(k, edge, nedges);
        float q = 0.0f;
        for (int slot = 0; slot < n_edges_on_edge[edge]; ++slot) {
            const int offset = edge * max_edges2 + slot;
            const int neighbor = edges_on_edge[offset];
            const float work_pv = mpas_mul(0.5f, mpas_add(
                pv_edge[index], pv_edge[E2(k, neighbor, nedges)]));
            q = mpas_add(q, mpas_mul(mpas_mul(
                weights_on_edge[offset], u[E2(k, neighbor, nedges)]), work_pv));
        }
        for (int slot = 0; slot < n_edges_on_edge[edge]; ++slot) {
            const int offset = edge * max_edges2 + slot;
            const int neighbor = edges_on_edge[offset];
            q = mpas_sub(q, mpas_mul(mpas_mul(
                weights_on_edge[offset], reference_u[E2(k, neighbor, nedges)]),
                f_edge[edge]));
        }
        const float gradient = mpas_mul(mpas_sub(
            kinetic[C2(k, c1, ncells)], kinetic[C2(k, c0, ncells)]),
            inv_dc_edge[edge]);
        const float horizontal = mpas_mul(rho_edge[index], mpas_sub(q, gradient));
        result[index] = mpas_sub(has_initial ? mpas_add(initial_tendency[index], horizontal) : horizontal,
            mpas_mul(mpas_mul(u[index], 0.5f),
                mpas_add(mass_divergence[C2(k, c0, ncells)],
                    mass_divergence[C2(k, c1, ncells)])));
    }
}

#ifndef HEX_HOIST_EOE
#define HEX_HOIST_EOE 10
#endif
extern "C" __global__ void vector_momentum_hoist_v841_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges2,
    const float *u, const float *rho_edge, const float *pv_edge,
    const float *kinetic, const float *mass_divergence,
    const int *cells_on_edge, const int *edges_on_edge,
    const int *n_edges_on_edge, const float *weights_on_edge,
    const float *inv_dc_edge, const float *reference_u, const float *f_edge,
    const float *initial_tendency, const int has_initial,
    float *result, const int lpt)
{
    /* One thread per edge walks every level.  The edge's stencil -- its
       neighbour edges, TRiSK weights, f, 1/dc and two cells -- is read once
       instead of once per level; every level runs the
       vector_momentum_v841_f32 element body unchanged (same operands, slot
       order and helper calls), so each output carries the same bits. */
    const int hex_rows = nlev;
    const int hex_chunks = (hex_rows + lpt - 1) / lpt;
    const int hex_element = blockDim.x * blockIdx.x + threadIdx.x;
    if (hex_element >= nedges * hex_chunks) return;
    const int edge = hex_element % nedges;
    const int hex_k0 = (hex_element / nedges) * lpt;
    const int hex_k1 = min(hex_rows, hex_k0 + lpt);
    const int count = n_edges_on_edge[edge];
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    if (count > HEX_HOIST_EOE) {
        for (int k = hex_k0; k < hex_k1; ++k) {
            const int index = E2(k, edge, nedges);
            float q = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int offset = edge * max_edges2 + slot;
                const int neighbor = edges_on_edge[offset];
                const float work_pv = mpas_mul(0.5f, mpas_add(
                    pv_edge[index], pv_edge[E2(k, neighbor, nedges)]));
                q = mpas_add(q, mpas_mul(mpas_mul(
                    weights_on_edge[offset], u[E2(k, neighbor, nedges)]), work_pv));
            }
            for (int slot = 0; slot < count; ++slot) {
                const int offset = edge * max_edges2 + slot;
                const int neighbor = edges_on_edge[offset];
                q = mpas_sub(q, mpas_mul(mpas_mul(
                    weights_on_edge[offset], reference_u[E2(k, neighbor, nedges)]),
                    f_edge[edge]));
            }
            const float gradient = mpas_mul(mpas_sub(
                kinetic[C2(k, c1, ncells)], kinetic[C2(k, c0, ncells)]),
                inv_dc_edge[edge]);
            const float horizontal = mpas_mul(rho_edge[index], mpas_sub(q, gradient));
            result[index] = mpas_sub(has_initial ? mpas_add(initial_tendency[index], horizontal) : horizontal,
                mpas_mul(mpas_mul(u[index], 0.5f),
                    mpas_add(mass_divergence[C2(k, c0, ncells)],
                        mass_divergence[C2(k, c1, ncells)])));
        }
        return;
    }
    int nb[HEX_HOIST_EOE];
    float wt[HEX_HOIST_EOE];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_EOE; ++slot) {
        nb[slot] = slot < count ? edges_on_edge[edge * max_edges2 + slot] : edge;
        wt[slot] = slot < count ? weights_on_edge[edge * max_edges2 + slot] : 0.0f;
    }
    const float fe = f_edge[edge];
    const float inv_dc = inv_dc_edge[edge];
    for (int k = hex_k0; k < hex_k1; ++k) {
        const int index = E2(k, edge, nedges);
        const float pv_self = pv_edge[index];
        float q = 0.0f;
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_EOE; ++slot) {
            if (slot < count) {
                const float work_pv = mpas_mul(0.5f, mpas_add(
                    pv_self, pv_edge[E2(k, nb[slot], nedges)]));
                q = mpas_add(q, mpas_mul(mpas_mul(
                    wt[slot], u[E2(k, nb[slot], nedges)]), work_pv));
            }
        }
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_EOE; ++slot) {
            if (slot < count) {
                q = mpas_sub(q, mpas_mul(mpas_mul(
                    wt[slot], reference_u[E2(k, nb[slot], nedges)]), fe));
            }
        }
        const float gradient = mpas_mul(mpas_sub(
            kinetic[C2(k, c1, ncells)], kinetic[C2(k, c0, ncells)]), inv_dc);
        const float horizontal = mpas_mul(rho_edge[index], mpas_sub(q, gradient));
        result[index] = mpas_sub(has_initial ? mpas_add(initial_tendency[index], horizontal) : horizontal,
            mpas_mul(mpas_mul(u[index], 0.5f),
                mpas_add(mass_divergence[C2(k, c0, ncells)],
                    mass_divergence[C2(k, c1, ncells)])));
    }
}

extern "C" __global__ void theta_finish_v841_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzw,
    const float *edge_flux, const float *vertical_flux, float *result)
{
    DYNAMICS_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = C2(k, cell, ncells);
        float value = 0.0f;
        for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
            const int edge = edges_on_cell[CES(cell, slot, max_edges)];
            value = mpas_sub(value, mpas_mul(
                acoustic_sign[CES(cell, slot, max_edges)],
                edge_flux[E2(k, edge, nedges)]));
        }
        value = mpas_mul(value, inv_area_cell[cell]);
        value = mpas_sub(value, mpas_mul(rdzw[k], mpas_sub(
            vertical_flux[C2(k + 1, cell, ncells)], vertical_flux[index])));
        result[index] = value;
    }
}

// Native F:6814-6874 accumulates all advective fluxes, then all corrections.
// Combining the two per edge changes cancellation before the cell sum.
extern "C" __global__ void theta_finish_corrected_v841_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzw,
    const float *edge_flux, const float *vertical_flux,
    const float *ru, const float *ru_saved, const float *theta_saved,
    const float *dv_edge, const int *cells_on_edge, float *result)
{
    DYNAMICS_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = C2(k, cell, ncells);
        float value = 0.0f;
        for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
            const int edge = edges_on_cell[CES(cell, slot, max_edges)];
            value = mpas_sub(value, mpas_mul(
                acoustic_sign[CES(cell, slot, max_edges)],
                edge_flux[E2(k, edge, nedges)]));
        }
        for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
            const int edge = edges_on_cell[CES(cell, slot, max_edges)];
            const int c0 = cells_on_edge[2 * edge];
            const int c1 = cells_on_edge[2 * edge + 1];
            const int eindex = E2(k, edge, nedges);
            const float flux = mpas_mul(mpas_mul(mpas_mul(mpas_mul(
                acoustic_sign[CES(cell, slot, max_edges)], dv_edge[edge]),
                mpas_sub(ru_saved[eindex], ru[eindex])), 0.5f),
                mpas_add(theta_saved[C2(k, c1, ncells)],
                    theta_saved[C2(k, c0, ncells)]));
            value = mpas_sub(value, flux);
        }
        value = mpas_mul(value, inv_area_cell[cell]);
        result[index] = mpas_sub(value, mpas_mul(rdzw[k], mpas_sub(
            vertical_flux[C2(k + 1, cell, ncells)], vertical_flux[index])));
    }
}

// Reuse topology for four levels without changing the per-output operation order.
extern "C" __global__ void theta_finish_corrected_chunk_v841_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzw,
    const float *edge_flux, const float *vertical_flux,
    const float *ru, const float *ru_saved, const float *theta_saved,
    const float *dv_edge, const int *cells_on_edge, float *result, const int lpt)
{
    const int chunks = (nlev + lpt - 1) / lpt;
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= ncells * chunks) return;
    const int cell = element % ncells;
    const int k0 = (element / ncells) * lpt;
    const int k1 = min(nlev, k0 + lpt);
    const int count = n_edges_on_cell[cell];
    const bool hoisted = count <= 10;
    int ed[10], ca[10], cb[10];
    float sg[10], sd[10];
#pragma unroll
    for (int slot = 0; slot < 10; ++slot) {
        const bool live = hoisted && slot < count;
        const int edge = live ? edges_on_cell[CES(cell, slot, max_edges)] : 0;
        ed[slot] = edge;
        ca[slot] = live ? cells_on_edge[2 * edge] : 0;
        cb[slot] = live ? cells_on_edge[2 * edge + 1] : 0;
        sg[slot] = live ? acoustic_sign[CES(cell, slot, max_edges)] : 0.0f;
        sd[slot] = live ? mpas_mul(sg[slot], dv_edge[edge]) : 0.0f;
    }
    for (int k = k0; k < k1; ++k) {
        const int index = C2(k, cell, ncells);
        float value = 0.0f;
        if (hoisted) {
#pragma unroll
            for (int slot = 0; slot < 10; ++slot) {
                if (slot < count) {
                    value = mpas_sub(value, mpas_mul(sg[slot],
                        edge_flux[E2(k, ed[slot], nedges)]));
                }
            }
            // Preserve the two separate sums and every FP32 helper operation.
#pragma unroll
            for (int slot = 0; slot < 10; ++slot) {
                if (slot < count) {
                    const int eindex = E2(k, ed[slot], nedges);
                    const float flux = mpas_mul(mpas_mul(mpas_mul(sd[slot],
                        mpas_sub(ru_saved[eindex], ru[eindex])), 0.5f),
                        mpas_add(theta_saved[C2(k, cb[slot], ncells)],
                            theta_saved[C2(k, ca[slot], ncells)]));
                    value = mpas_sub(value, flux);
                }
            }
        } else {
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                value = mpas_sub(value, mpas_mul(
                    acoustic_sign[CES(cell, slot, max_edges)],
                    edge_flux[E2(k, edge, nedges)]));
            }
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const int c0 = cells_on_edge[2 * edge];
                const int c1 = cells_on_edge[2 * edge + 1];
                const int eindex = E2(k, edge, nedges);
                const float flux = mpas_mul(mpas_mul(mpas_mul(mpas_mul(
                    acoustic_sign[CES(cell, slot, max_edges)], dv_edge[edge]),
                    mpas_sub(ru_saved[eindex], ru[eindex])), 0.5f),
                    mpas_add(theta_saved[C2(k, c1, ncells)],
                        theta_saved[C2(k, c0, ncells)]));
                value = mpas_sub(value, flux);
            }
        }
        value = mpas_mul(value, inv_area_cell[cell]);
        result[index] = mpas_sub(value, mpas_mul(rdzw[k], mpas_sub(
            vertical_flux[C2(k + 1, cell, ncells)], vertical_flux[index])));
    }
}

extern "C" __global__ void w_finish_v841_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzu,
    const float *edge_flux, const float *vertical_flux, float *result)
{
    DYNAMICS_ELEMENT_LOOP((nlev + 1) * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = C2(k, cell, ncells);
        if (k == 0 || k == nlev) {
            result[index] = 0.0f;
        } else {
            float value = 0.0f;
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                value = mpas_sub(value, mpas_mul(
                    acoustic_sign[CES(cell, slot, max_edges)],
                    edge_flux[E2(k, edge, nedges)]));
            }
            value = mpas_mul(value, inv_area_cell[cell]);
            value = mpas_sub(value, mpas_mul(rdzu[k], mpas_sub(
                vertical_flux[C2(k + 1, cell, ncells)], vertical_flux[index])));
            result[index] = value;
        }
    }
}

extern "C" __global__ void enforce_rw_endpoints_v841_f32(
    const int nlev, const int ncells, float *rw)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    rw[C2(0, cell, ncells)] = 0.0f;
    rw[C2(nlev, cell, ncells)] = 0.0f;
}

extern "C" __global__ void validate_positive_state_v841_f32(
    const int nvalues, const float *rho, const float *rho_theta, int *invalid)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nvalues) return;
    const float density = rho[index];
    const float theta_mass = rho_theta[index];
    if (!(isfinite(density) && density > 0.0f
            && isfinite(theta_mass) && theta_mass > 0.0f)) {
        atomicExch(invalid, 1);
    }
}

extern "C" __global__ void validate_recovered_v841_f32(
    const int nlev, const int ncells, const int nedges,
    const float *rho, const float *rho_theta, const float *rho_u,
    const float *rho_w, const float *theta, const float *exner,
    const float *rho_p, const float *rtheta_p, const float *pressure_p,
    const float *normal_velocity, const float *vertical_velocity, int *invalid)
{
    // Level-local checks. Keep every recovery checkpoint and its sticky flag.
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < nlev * ncells) {
        if (!(isfinite(rho[index]) && rho[index] > 0.0f
                && isfinite(rho_theta[index]) && rho_theta[index] > 0.0f
                && isfinite(theta[index]) && theta[index] > 0.0f
                && isfinite(exner[index]) && exner[index] > 0.0f
                && isfinite(rho_p[index]) && isfinite(rtheta_p[index])
                && isfinite(pressure_p[index]))) {
            atomicExch(invalid, 1);
        }
    }
    if (index < (nlev + 1) * ncells) {
        if (!(isfinite(rho_w[index]) && isfinite(vertical_velocity[index])))
            atomicExch(invalid, 1);
    }
    if (index < nlev * nedges) {
        if (!(isfinite(rho_u[index]) && isfinite(normal_velocity[index])))
            atomicExch(invalid, 1);
    }
}

extern "C" __global__ void validate_finite_array_v841_f32(
    const int nvalues, const float *values, int *invalid)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < nvalues && !isfinite(values[index])) atomicExch(invalid, 1);
}

extern "C" __global__ void split_flux_first_v841_f32(
    const int nlevels, const int nowners, const float *current, float *accumulator)
{
    DYNAMICS_ELEMENT_LOOP(nlevels * nowners) {
        accumulator[element] = current[element];
    }
}

extern "C" __global__ void split_flux_add_v841_f32(
    const int nlevels, const int nowners, const float *current, float *accumulator)
{
    DYNAMICS_ELEMENT_LOOP(nlevels * nowners) {
        accumulator[element] = mpas_add(current[element], accumulator[element]);
    }
}

extern "C" __global__ void split_flux_finish_v841_f32(
    const int nlevels, const int nowners, const float reciprocal,
    const float *accumulator, float *average)
{
    DYNAMICS_ELEMENT_LOOP(nlevels * nowners) {
        average[element] = mpas_mul(accumulator[element], reciprocal);
    }
}

#ifndef HEX_HOIST_CELL
#define HEX_HOIST_CELL 10
#endif
/* Chunked, stencil-hoisted forms of theta_finish_v841_f32 and
   w_finish_v841_f32: one thread per (level chunk, cell); rows 0 and nlev of
   w are the element kernel's zero rows. */
extern "C" __global__ void finish_chunk_v841_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdz,
    const float *edge_flux, const float *vertical_flux, float *result,
    const int w_rows, const int lpt)
{
    const int hex_rows = w_rows ? nlev + 1 : nlev;
    const int hex_chunks = (hex_rows + lpt - 1) / lpt;
    const int hex_element = blockDim.x * blockIdx.x + threadIdx.x;
    if (hex_element >= ncells * hex_chunks) return;
    const int cell = hex_element % ncells;
    const int hex_k0 = (hex_element / ncells) * lpt;
    const int hex_k1 = min(hex_rows, hex_k0 + lpt);
    const int count = n_edges_on_cell[cell];
    const bool hoisted = count <= HEX_HOIST_CELL;
    int ed[HEX_HOIST_CELL];
    float sg[HEX_HOIST_CELL];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
        const bool live = hoisted && slot < count;
        ed[slot] = live ? edges_on_cell[CES(cell, slot, max_edges)] : 0;
        sg[slot] = live ? acoustic_sign[CES(cell, slot, max_edges)] : 0.0f;
    }
    const float inv_area = inv_area_cell[cell];
    for (int k = hex_k0; k < hex_k1; ++k) {
        const int index = C2(k, cell, ncells);
        if (w_rows && (k == 0 || k == nlev)) {
            result[index] = 0.0f;
            continue;
        }
        float value = 0.0f;
        if (hoisted) {
#pragma unroll
            for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
                if (slot < count) {
                    value = mpas_sub(value, mpas_mul(sg[slot],
                        edge_flux[E2(k, ed[slot], nedges)]));
                }
            }
        } else {
            for (int slot = 0; slot < count; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                value = mpas_sub(value, mpas_mul(
                    acoustic_sign[CES(cell, slot, max_edges)],
                    edge_flux[E2(k, edge, nedges)]));
            }
        }
        value = mpas_mul(value, inv_area);
        value = mpas_sub(value, mpas_mul(rdz[k], mpas_sub(
            vertical_flux[C2(k + 1, cell, ncells)], vertical_flux[index])));
        result[index] = value;
    }
}
"""

_CACHE: Any | None = None
_KERNELS: dict[tuple[Any, str], Any] = {}


def _cupy() -> Any:
    from .cuda_backend import require_cuda

    require_cuda(min_compute=(12, 0))
    import cupy as cp

    return cp


def _kernel(name: str, kernel_cache: Any | None = None, dimensions=()) -> Any:
    global _CACHE
    from .cuda_backend import KernelCache, require_cuda

    selected = kernel_cache
    if selected is None:
        if _CACHE is None:
            _CACHE = KernelCache(capability=require_cuda(min_compute=(12, 0)))
        selected = _CACHE
    # Unspecialized LTS interface taps replace kernels through this key.
    key = (selected, name, dimensions) if dimensions else (selected, name)
    if key not in _KERNELS:
        if dimensions:
            from .cuda_shape import resolve
            _KERNELS[key] = resolve(selected, name, _CUDA_SOURCE,
                                    "hexcore.cuda_dynamics_v841", dimensions)
        else:
            _KERNELS[key] = selected.raw_kernel(
                name, _CUDA_SOURCE, module_key="hexcore.cuda_dynamics_v841")
    return _KERNELS[key]


def _launch(name: str, count: int, args: tuple[Any, ...], cache: Any | None) -> None:
    if count < 1:
        return
    threads = 128
    dimensions = tuple(zip(("nlev", "ncells", "nedges", "max_edges2"), args[:4])) if name == "vector_momentum_hoist_v841_f32" else ()
    _kernel(name, cache, dimensions)(
        ((int(count) + threads - 1) // threads,),
        (threads,),
        args,
    )


def _shape(value: Any, name: str) -> tuple[int, int]:
    array = require_resident_array(name, value, dtype=np.float32)
    if array.ndim != 2 or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous (level,owner) float32")
    return int(array.shape[0]), int(array.shape[1])


def vector_momentum_tendency_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    *,
    normal_velocity: Any,
    rho_edge: Any,
    pv_edge: Any,
    kinetic_energy: Any,
    horizontal_divergence: Any,
    initial_tendency: Any | None = None,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    nlev, nedges = _shape(normal_velocity, "normal_velocity")
    ncells = int(mesh.n_cells)
    if nedges != int(mesh.n_edges):
        raise ValueError("normal_velocity edge extent differs from mesh")
    mesh.validate()
    context.validate(
        n_vert_levels=nlev,
        n_cells=ncells,
        n_edges=nedges,
        n_vertices=int(mesh.n_vertices),
    )
    for name, value, shape in (
        ("rho_edge", rho_edge, (nlev, nedges)),
        ("pv_edge", pv_edge, (nlev, nedges)),
        ("kinetic_energy", kinetic_energy, (nlev, ncells)),
        ("horizontal_divergence", horizontal_divergence, (nlev, ncells)),
    ):
        require_resident_array(name, value, dtype=np.float32, shape=shape)
    out = cp.empty_like(normal_velocity)
    if initial_tendency is not None:
        require_resident_array("initial_tendency", initial_tendency, dtype=np.float32, shape=(nlev, nedges))
    reference_u = reference_wind_edge_cuda_v841(
        mesh, context, n_vert_levels=nlev, kernel_cache=kernel_cache
    )
    arguments = (
        np.int32(nlev), np.int32(ncells), np.int32(nedges),
        np.int32(mesh.max_edges2), normal_velocity, rho_edge, pv_edge,
        kinetic_energy, horizontal_divergence, mesh.cells_on_edge,
        mesh.edges_on_edge, mesh.n_edges_on_edge, mesh.weights_on_edge,
        context.inv_dc_edge, reference_u, mesh.f_edge,
        normal_velocity if initial_tendency is None else initial_tendency,
        np.int32(initial_tendency is not None),
    )
    from . import cuda_edge_hoist
    if not cuda_edge_hoist.enabled(nedges):
        _launch("vector_momentum_v841_f32", nlev * nedges, (*arguments, out), kernel_cache)
        return out
    lpt = cuda_edge_hoist.levels_per_thread(
        "vector_momentum_hoist_v841_f32", nedges, nlev, lambda value: _launch(
            "vector_momentum_hoist_v841_f32", nedges * cuda_edge_hoist.chunks(nlev, value),
            (*arguments, out, np.int32(value)), kernel_cache), _cupy())
    _launch("vector_momentum_hoist_v841_f32", nedges * cuda_edge_hoist.chunks(nlev, lpt),
            (*arguments, out, np.int32(lpt)), kernel_cache)
    if cuda_edge_hoist.check_wanted("vector_momentum_hoist_v841_f32", cp):
        element = cp.empty_like(out)
        _launch("vector_momentum_v841_f32", nlev * nedges, (*arguments, element), kernel_cache)
        cuda_edge_hoist.compare("vector_momentum_hoist_v841_f32", out, element, cp)
    return out


_REFERENCE_WIND: dict[tuple[Any, ...], tuple[Any, ...]] = {}


def reference_wind_edge_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    *,
    n_vert_levels: int,
    kernel_cache: Any | None = None,
) -> Any:
    """The (level, edge) perturbation-Coriolis reference wind, once per run.

    ``u_init(k)*cos(angleEdge(e)) - v_init(k)*sin(angleEdge(e))`` depends on
    nothing that moves during a forecast, and the momentum kernel evaluated
    it for every neighbour of every edge at every level on every RK stage:
    ten full-precision ``cosf``/``sinf`` pairs per element per stage.  It is
    computed once here by a kernel of the same translation unit -- the same
    ``cosf``/``sinf`` implementation, the same float32 operands, the same
    operation order -- so the gathered value is the value the kernel used to
    compute in place, bit for bit.

    The cache is keyed by the identity of the three device arrays and holds
    a reference to each, so an entry can only ever be served for the exact
    arrays it was computed from: a freed array's address cannot be reused
    under a live key.
    """

    cp = _cupy()
    nlev = int(n_vert_levels)
    nedges = int(mesh.n_edges)
    angle_edge = mesh.angle_edge
    key = (
        int(angle_edge.data.ptr), int(context.u_init.data.ptr),
        int(context.v_init.data.ptr), nlev, nedges, id(kernel_cache),
    )
    hit = _REFERENCE_WIND.get(key)
    if hit is not None and hit[0] is angle_edge and hit[1] is context.u_init \
            and hit[2] is context.v_init:
        return hit[3]
    require_resident_array("angle_edge", angle_edge, dtype=np.float32, shape=(nedges,))
    require_resident_array("u_init", context.u_init, dtype=np.float32, shape=(nlev,))
    require_resident_array("v_init", context.v_init, dtype=np.float32, shape=(nlev,))
    reference_u = cp.empty((nlev, nedges), dtype=cp.float32)
    _launch(
        "reference_wind_edge_v841_f32",
        nlev * nedges,
        (
            np.int32(nlev), np.int32(nedges), angle_edge,
            context.u_init, context.v_init, reference_u,
        ),
        kernel_cache,
    )
    _REFERENCE_WIND[key] = (angle_edge, context.u_init, context.v_init, reference_u)
    return reference_u


def theta_finish_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    *,
    edge_flux: Any,
    vertical_flux: Any,
    rdzw: Any,
    n_vert_levels: int,
    correction: tuple[Any, Any, Any] | None = None,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    nlev = int(n_vert_levels)
    ncells = int(mesh.n_cells)
    nedges = int(mesh.n_edges)
    mesh.validate()
    context.validate(
        n_vert_levels=nlev,
        n_cells=ncells,
        n_edges=nedges,
        n_vertices=int(mesh.n_vertices),
    )
    require_resident_array(
        "edge_flux", edge_flux, dtype=np.float32, shape=(nlev, nedges)
    )
    require_resident_array(
        "vertical_flux", vertical_flux,
        dtype=np.float32, shape=(nlev + 1, ncells),
    )
    require_resident_array("rdzw", rdzw, dtype=np.float32, shape=(nlev,))
    out = cp.empty((nlev, ncells), dtype=cp.float32)
    inputs = (
        np.int32(nlev), np.int32(ncells),
        np.int32(nedges), np.int32(mesh.max_edges),
        mesh.n_edges_on_cell, mesh.edges_on_cell, mesh.edge_sign_on_cell,
        context.inv_area_cell, rdzw, edge_flux, vertical_flux,
    )
    if correction is not None:
        ru, ru_saved, theta_saved = correction
        for name, value, shape in (
            ("ru", ru, (nlev, nedges)),
            ("ru_saved", ru_saved, (nlev, nedges)),
            ("theta_saved", theta_saved, (nlev, ncells)),
        ):
            require_resident_array(name, value, dtype=np.float32, shape=shape)
        from . import cuda_edge_hoist
        corrected_inputs = (*inputs, ru, ru_saved, theta_saved, mesh.dv_edge,
                            mesh.cells_on_edge)
        if cuda_edge_hoist.enabled(nedges):
            # Fixed geometry: no first-call tuning or synchronization here.
            lpt = 4
            _launch("theta_finish_corrected_chunk_v841_f32",
                    ncells * cuda_edge_hoist.chunks(nlev, lpt),
                    (*corrected_inputs, out, np.int32(lpt)), kernel_cache)
            if cuda_edge_hoist.check_wanted("theta_finish_corrected_chunk_v841_f32", cp):
                element = cp.empty_like(out)
                _launch("theta_finish_corrected_v841_f32", nlev * ncells,
                        (*corrected_inputs, element), kernel_cache)
                cuda_edge_hoist.compare("theta_finish_corrected_chunk_v841_f32",
                                        out, element, cp)
        else:
            _launch("theta_finish_corrected_v841_f32", nlev * ncells,
                    (*corrected_inputs, out), kernel_cache)
        return out
    from . import cuda_edge_hoist
    if not cuda_edge_hoist.enabled(nedges):
        _launch("theta_finish_v841_f32", nlev * ncells, (*inputs, out), kernel_cache)
        return out
    rows = nlev + 0
    lpt = cuda_edge_hoist.levels_per_thread(
        "theta_finish_chunk_v841_f32", ncells, rows, lambda value: _launch(
            "finish_chunk_v841_f32", ncells * cuda_edge_hoist.chunks(rows, value),
            (*inputs, out, np.int32(0), np.int32(value)), kernel_cache), cp)
    _launch("finish_chunk_v841_f32", ncells * cuda_edge_hoist.chunks(rows, lpt),
            (*inputs, out, np.int32(0), np.int32(lpt)), kernel_cache)
    if cuda_edge_hoist.check_wanted("theta_finish_chunk_v841_f32", cp):
        element = cp.empty_like(out)
        _launch("theta_finish_v841_f32", nlev * ncells, (*inputs, element), kernel_cache)
        cuda_edge_hoist.compare("theta_finish_chunk_v841_f32", out, element, cp)
    return out


def w_finish_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    *,
    edge_flux: Any,
    vertical_flux: Any,
    rdzu: Any,
    n_vert_levels: int,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    nlev = int(n_vert_levels)
    ncells = int(mesh.n_cells)
    nedges = int(mesh.n_edges)
    mesh.validate()
    context.validate(
        n_vert_levels=nlev,
        n_cells=ncells,
        n_edges=nedges,
        n_vertices=int(mesh.n_vertices),
    )
    require_resident_array(
        "edge_flux", edge_flux, dtype=np.float32, shape=(nlev + 1, nedges)
    )
    require_resident_array(
        "vertical_flux", vertical_flux,
        dtype=np.float32, shape=(nlev + 1, ncells),
    )
    require_resident_array("rdzu", rdzu, dtype=np.float32, shape=(nlev,))
    out = cp.empty((nlev + 1, ncells), dtype=cp.float32)
    inputs = (
        np.int32(nlev), np.int32(ncells),
        np.int32(nedges), np.int32(mesh.max_edges),
        mesh.n_edges_on_cell, mesh.edges_on_cell, mesh.edge_sign_on_cell,
        context.inv_area_cell, rdzu, edge_flux, vertical_flux,
    )
    from . import cuda_edge_hoist
    if not cuda_edge_hoist.enabled(nedges):
        _launch("w_finish_v841_f32", (nlev + 1) * ncells, (*inputs, out), kernel_cache)
        return out
    rows = nlev + 1
    lpt = cuda_edge_hoist.levels_per_thread(
        "w_finish_chunk_v841_f32", ncells, rows, lambda value: _launch(
            "finish_chunk_v841_f32", ncells * cuda_edge_hoist.chunks(rows, value),
            (*inputs, out, np.int32(1), np.int32(value)), kernel_cache), cp)
    _launch("finish_chunk_v841_f32", ncells * cuda_edge_hoist.chunks(rows, lpt),
            (*inputs, out, np.int32(1), np.int32(lpt)), kernel_cache)
    if cuda_edge_hoist.check_wanted("w_finish_chunk_v841_f32", cp):
        element = cp.empty_like(out)
        _launch("w_finish_v841_f32", (nlev + 1) * ncells, (*inputs, element), kernel_cache)
        cuda_edge_hoist.compare("w_finish_chunk_v841_f32", out, element, cp)
    return out


def enforce_rw_endpoints_cuda_v841(
    rho_w: Any,
    *,
    kernel_cache: Any | None = None,
) -> Any:
    nlevels, ncells = _shape(rho_w, "rho_w")
    if nlevels < 2:
        raise ValueError("rho_w must contain bottom and top interfaces")
    _launch(
        "enforce_rw_endpoints_v841_f32",
        ncells,
        (np.int32(nlevels - 1), np.int32(ncells), rho_w),
        kernel_cache,
    )
    return rho_w


def validate_positive_state_cuda_v841(
    rho: Any,
    rho_theta: Any,
    *,
    invalid_flag: Any | None = None,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    nlev, ncells = _shape(rho, "rho")
    require_resident_array(
        "rho_theta", rho_theta, dtype=np.float32, shape=(nlev, ncells)
    )
    owns_flag = invalid_flag is None
    invalid = (
        cp.zeros((1,), dtype=cp.int32)
        if owns_flag
        else require_resident_array(
            "invalid_flag", invalid_flag, dtype=np.int32, shape=(1,)
        )
    )
    _launch(
        "validate_positive_state_v841_f32",
        nlev * ncells,
        (
            np.int32(nlev * ncells), rho, rho_theta, invalid,
        ),
        kernel_cache,
    )
    if owns_flag and int(cp.asnumpy(invalid)[0]) != 0:
        raise FloatingPointError(
            "v8.4.1 CUDA rho and rho_theta must remain finite and positive"
        )
    return invalid


def validate_recovered_state_cuda_v841(
    state: Any,
    saved: Any,
    *,
    invalid_flag: Any,
    kernel_cache: Any | None = None,
) -> Any:
    nlev, ncells = _shape(state.rho, "state.rho")
    nedges = int(state.rho_u.shape[1])
    requirements = {
        "state.rho_theta": (state.rho_theta, (nlev, ncells)),
        "state.rho_u": (state.rho_u, (nlev, nedges)),
        "state.rho_w": (state.rho_w, (nlev + 1, ncells)),
        "saved.theta_m": (saved.theta_m, (nlev, ncells)),
        "saved.exner": (saved.exner, (nlev, ncells)),
        "saved.density_perturbation": (
            saved.density_perturbation, (nlev, ncells)
        ),
        "saved.rho_theta_perturbation": (
            saved.rho_theta_perturbation, (nlev, ncells)
        ),
        "saved.pressure_perturbation": (
            saved.pressure_perturbation, (nlev, ncells)
        ),
        "saved.normal_velocity": (saved.normal_velocity, (nlev, nedges)),
        "saved.vertical_velocity": (
            saved.vertical_velocity, (nlev + 1, ncells)
        ),
    }
    for name, (value, shape) in requirements.items():
        require_resident_array(name, value, dtype=np.float32, shape=shape)
    flag = require_resident_array(
        "invalid_flag", invalid_flag, dtype=np.int32, shape=(1,)
    )
    _launch(
        "validate_recovered_v841_f32",
        max((nlev + 1) * ncells, nlev * nedges),
        (
            np.int32(nlev), np.int32(ncells), np.int32(nedges),
            state.rho, state.rho_theta, state.rho_u, state.rho_w,
            saved.theta_m, saved.exner, saved.density_perturbation,
            saved.rho_theta_perturbation, saved.pressure_perturbation,
            saved.normal_velocity, saved.vertical_velocity, flag,
        ),
        kernel_cache,
    )
    return flag


def accumulate_finite_array_cuda_v841(
    values: Any,
    *,
    invalid_flag: Any,
    kernel_cache: Any | None = None,
) -> Any:
    array = require_resident_array("values", values, dtype=np.float32)
    flag = require_resident_array(
        "invalid_flag", invalid_flag, dtype=np.int32, shape=(1,)
    )
    _launch(
        "validate_finite_array_v841_f32",
        int(array.size),
        (np.int32(array.size), array, flag),
        kernel_cache,
    )
    return flag


def accumulate_split_flux_cuda_v841(
    current: Any,
    accumulator: Any | None,
    *,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    nlevels, nowners = _shape(current, "current_split_flux")
    if accumulator is None:
        result = cp.empty_like(current)
        kernel = "split_flux_first_v841_f32"
    else:
        accumulator_levels, accumulator_owners = _shape(
            accumulator, "split_flux_accumulator"
        )
        if (accumulator_levels, accumulator_owners) != (nlevels, nowners):
            raise ValueError("split flux accumulator shape mismatch")
        result = accumulator
        kernel = "split_flux_add_v841_f32"
    _launch(
        kernel,
        nlevels * nowners,
        (np.int32(nlevels), np.int32(nowners), current, result),
        kernel_cache,
    )
    return result


def finish_split_flux_cuda_v841(
    accumulator: Any,
    split_count: int,
    *,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    if not isinstance(split_count, (int, np.integer)) or isinstance(
        split_count, (bool, np.bool_)
    ) or int(split_count) < 1:
        raise ValueError("split_count must be a positive integer")
    nlevels, nowners = _shape(accumulator, "split_flux_accumulator")
    result = cp.empty_like(accumulator)
    reciprocal = np.float32(1.0) / np.float32(split_count)
    _launch(
        "split_flux_finish_v841_f32",
        nlevels * nowners,
        (
            np.int32(nlevels), np.int32(nowners), reciprocal,
            accumulator, result,
        ),
        kernel_cache,
    )
    return result


__all__ = [
    "accumulate_split_flux_cuda_v841",
    "accumulate_finite_array_cuda_v841",
    "enforce_rw_endpoints_cuda_v841",
    "finish_split_flux_cuda_v841",
    "theta_finish_cuda_v841",
    "validate_positive_state_cuda_v841",
    "validate_recovered_state_cuda_v841",
    "vector_momentum_tendency_cuda_v841",
    "w_finish_cuda_v841",
]
