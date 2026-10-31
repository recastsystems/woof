"""RRFS deformation and dry-mixing subsystem.

Source: RRFSx/MPAS-Model MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
The stored-inverse diagnostics and non-LES filters retain the shared source arithmetic;
the fork deformation uses defc_a/defc_b and left-associated accumulations."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_horizontal import (
    CudaDryMixingTendencies,
    CudaHorizontal,
    CudaSolveDiagnostics,
)
from .cuda_v841 import CudaV841Context


_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
/* Level-independent kernels walk a flat (level, owner) element index, owner
   fastest, in a grid-stride loop; each element runs the former per-owner
   loop body for its level unchanged, so the launch geometry is not in the
   arithmetic.  The forecast launches one thread per element. */
#define HORIZONTAL_ELEMENT_LOOP(total) \
    for (int element = blockDim.x * blockIdx.x + threadIdx.x; \
         element < (total); element += gridDim.x * blockDim.x)
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
__device__ __forceinline__ int lidx_v841(int level, int entity, int count) {
    return level * count + entity;
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
__device__ __forceinline__ float cell_sign_v841(
    int cell, int edge, const int *cells_on_edge)
{
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
__device__ __forceinline__ float vertex_sign_v841(
    int vertex, int edge, const int *vertices_on_edge)
{
    return vertices_on_edge[2 * edge] == vertex ? -1.0f : 1.0f;
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
__device__ __forceinline__ int kite_slot_v841(
    int vertex, int cell, const int *cells_on_vertex, int vertex_degree)
{
    for (int slot = 0; slot < vertex_degree; ++slot) {
        if (cells_on_vertex[vertex * vertex_degree + slot] == cell) return slot;
    }
    return -1;
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:6506-6942 at 1d6e370e.
extern "C" __global__ void vertex_diagnostics_v841_f32(
    const float *normal_velocity, const float *ke_edge,
    const int *edges_on_vertex, const int *vertices_on_edge,
    const float *dc_edge, const float *inv_area_triangle,
    const float *f_vertex, const int nlev, const int nedges,
    const int nvertices, const int vertex_degree,
    float *vorticity, float *ke_vertex, float *pv_vertex)
{
    HORIZONTAL_ELEMENT_LOOP(nlev * nvertices) {
        const int vertex = element % nvertices;
        const int k = element / nvertices;
        float vort = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex * vertex_degree + slot];
            if (edge < 0) continue;
            vort = mpas_add(vort, mpas_mul(mpas_mul(
                vertex_sign_v841(vertex, edge, vertices_on_edge), dc_edge[edge]),
                normal_velocity[lidx_v841(k, edge, nedges)]));
            ke = mpas_add(ke, ke_edge[lidx_v841(k, edge, nedges)]);
        }
        vort = mpas_mul(vort, inv_area_triangle[vertex]);
        const float r = mpas_mul(0.25f, inv_area_triangle[vertex]);
        ke = mpas_mul(ke, r);
        const int index = lidx_v841(k, vertex, nvertices);
        vorticity[index] = vort;
        ke_vertex[index] = ke;
        pv_vertex[index] = mpas_add(f_vertex[vertex], vort);
    }
}

#ifndef HEX_HOIST_EOE
#define HEX_HOIST_EOE 10
#endif
/* Thread-per-edge form of tangential_velocity_f32 (cuda_horizontal.py): the
   edge's neighbour indices and TRiSK weights are read once into registers
   and the thread walks the levels; each level runs the element body
   unchanged (same slot order, same helper calls), so each value carries the
   same bits. */
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:6506-6942 at 1d6e370e.
extern "C" __global__ void tangential_velocity_hoist_v841_f32(
    const float* normal_velocity,
    const int* edges_on_edge,
    const int* n_edges_on_edge,
    const float* weights_on_edge,
    const int nlev,
    const int nedges,
    const int max_edges2,
    float* tangential_velocity, const int lpt)
{
    const int hex_rows = nlev;
    const int hex_chunks = (hex_rows + lpt - 1) / lpt;
    const int hex_element = blockDim.x * blockIdx.x + threadIdx.x;
    if (hex_element >= nedges * hex_chunks) return;
    const int edge = hex_element % nedges;
    const int hex_k0 = (hex_element / nedges) * lpt;
    const int hex_k1 = min(hex_rows, hex_k0 + lpt);
    const int count = n_edges_on_edge[edge];
    if (count > HEX_HOIST_EOE) {
        for (int k = hex_k0; k < hex_k1; ++k) {
            float value = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int offset = edge * max_edges2 + slot;
                const int neighbor = edges_on_edge[offset];
                value = mpas_add(value, mpas_mul(weights_on_edge[offset],
                    normal_velocity[lidx_v841(k, neighbor, nedges)]));
            }
            tangential_velocity[lidx_v841(k, edge, nedges)] = value;
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
    for (int k = hex_k0; k < hex_k1; ++k) {
        float value = 0.0f;
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_EOE; ++slot) {
            if (slot < count) {
                value = mpas_add(value, mpas_mul(wt[slot],
                    normal_velocity[lidx_v841(k, nb[slot], nedges)]));
            }
        }
        tangential_velocity[lidx_v841(k, edge, nedges)] = value;
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:6506-6942 at 1d6e370e.
extern "C" __global__ void cell_diagnostics_v841_f32(
    const float *normal_velocity, const float *ke_edge,
    const float *ke_vertex, const int *edges_on_cell,
    const int *n_edges_on_cell, const int *cells_on_edge,
    const int *vertices_on_cell, const int *cells_on_vertex,
    const float *dv_edge, const float *inv_area_cell,
    const float *kite_areas, const int nlev, const int ncells,
    const int nedges, const int nvertices, const int max_edges,
    const int vertex_degree, float *divergence, float *kinetic_energy)
{
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        const float inv_area = inv_area_cell[cell];
        const float ke_fact = 0.625f;
        float div = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            div = mpas_add(div, mpas_mul(mpas_mul(
                cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                normal_velocity[lidx_v841(k, edge, nedges)]));
            ke = mpas_add(ke, mpas_mul(
                0.25f, ke_edge[lidx_v841(k, edge, nedges)]));
        }
        div = mpas_mul(div, inv_area);
        ke = mpas_mul(ke, inv_area);
        ke = mpas_mul(ke, ke_fact);
        for (int slot = 0; slot < count; ++slot) {
            const int vertex = vertices_on_cell[cell * max_edges + slot];
            const int kite = kite_slot_v841(
                vertex, cell, cells_on_vertex, vertex_degree);
            if (kite >= 0) {
                float correction = mpas_mul(mpas_sub(1.0f, ke_fact),
                    kite_areas[vertex * vertex_degree + kite]);
                correction = mpas_mul(correction,
                    ke_vertex[lidx_v841(k, vertex, nvertices)]);
                correction = mpas_mul(correction, inv_area);
                ke = mpas_add(ke, correction);
            }
        }
        const int index = lidx_v841(k, cell, ncells);
        divergence[index] = div;
        kinetic_energy[index] = ke;
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void laplacian_divergence_v841_f32(
    const float* delsq_u,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* dv_edge,
    const float* inv_area_cell,
    const int nlev,
    const int ncells,
    const int nedges,
    const int max_edges,
    float* delsq_divergence
) {
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            value = mpas_add(value, mpas_mul(mpas_mul(mpas_mul(inv_area_cell[cell], dv_edge[edge]),
                cell_sign_v841(cell, edge, cells_on_edge)), delsq_u[lidx_v841(k, edge, nedges)]));
        }
        delsq_divergence[lidx_v841(k, cell, ncells)] = value;
    }
}


// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void laplacian_vorticity_v841_f32(
    const float* delsq_u,
    const int* edges_on_vertex,
    const int* vertices_on_edge,
    const float* dc_edge,
    const float* inv_area_triangle,
    const int nlev,
    const int nedges,
    const int nvertices,
    const int vertex_degree,
    float* delsq_vorticity
) {
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * nvertices) return;
    const int vertex = element % nvertices;
    const int k = element / nvertices;
    { // Each level reads saved fields and writes its own output word.
        float value = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex * vertex_degree + slot];
            if (edge < 0) continue;
            value = mpas_add(value, mpas_mul(mpas_mul(mpas_mul(inv_area_triangle[vertex], dc_edge[edge]),
                vertex_sign_v841(vertex, edge, vertices_on_edge)), delsq_u[lidx_v841(k, edge, nedges)]));
        }
        delsq_vorticity[lidx_v841(k, vertex, nvertices)] = value;
    }
}


// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void theta_filter_lap2_v841_f32(
    const float* theta_m,
    const float* rho_edge,
    const float* kdiff,
    const float* del2,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* inv_dc_edge,
    const float* dv_edge,
    const float* inv_area_cell,
    const int nlev,
    const int ncells,
    const int nedges,
    const int max_edges,
    float* delsq_theta,
    float* tendency
) {
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float lap = 0.0f;
        float tend = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_mul(mpas_mul(mpas_mul(inv_area_cell[cell],
                cell_sign_v841(cell, edge, cells_on_edge)), dv_edge[edge]),
                inv_dc_edge[edge]);
            const float flux = mpas_mul(mpas_mul(factor, mpas_sub(
                theta_m[lidx_v841(k, cell1, ncells)],
                theta_m[lidx_v841(k, cell0, ncells)])),
                rho_edge[lidx_v841(k, edge, nedges)]);
            lap = mpas_add(lap, flux);
            tend = mpas_add(tend, mpas_mul(mpas_mul(mpas_mul(
                flux, 0.5f), mpas_add(
                    kdiff[lidx_v841(k, cell0, ncells)],
                    kdiff[lidx_v841(k, cell1, ncells)])), del2[edge]));
        }
        const int tid = lidx_v841(k, cell, ncells);
        delsq_theta[tid] = lap;
        tendency[tid] = tend;
    }
}


// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void theta_filter_lap4_v841_f32(
    const float* delsq_theta,
    const float* del4,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* inv_dc_edge,
    const float* dv_edge,
    const float* inv_area_cell,
    const float h4,
    const int nlev,
    const int ncells,
    const int max_edges,
    float* tendency
) {
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        const int tid = lidx_v841(k, cell, ncells);
        float tend = tendency[tid];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_mul(mpas_mul(mpas_mul(mpas_mul(del4[edge],
                mpas_mul(h4, inv_area_cell[cell])), dv_edge[edge]),
                cell_sign_v841(cell, edge, cells_on_edge)), inv_dc_edge[edge]);
            tend = mpas_sub(tend, mpas_mul(factor, mpas_sub(
                delsq_theta[lidx_v841(k, cell1, ncells)],
                delsq_theta[lidx_v841(k, cell0, ncells)])));
        }
        tendency[tid] = tend;
    }
}


// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void w_filter_lap2_v841_f32(
    const float* vertical_velocity,
    const float* rho_edge,
    const float* kdiff,
    const float* del2,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* inv_dc_edge,
    const float* dv_edge,
    const float* inv_area_cell,
    const int nlev,
    const int ncells,
    const int nedges,
    const int max_edges,
    float* delsq_w,
    float* tendency
) {
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        const int tid = lidx_v841(k, cell, ncells);
        if (k == 0) {
            delsq_w[tid] = 0.0f;
            tendency[tid] = 0.0f;
            continue;
        }
        float lap = 0.0f;
        float tend = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f,
                inv_area_cell[cell]),
                cell_sign_v841(cell, edge, cells_on_edge)), dv_edge[edge]),
                inv_dc_edge[edge]);
            const float flux = mpas_mul(mpas_mul(factor, mpas_add(
                rho_edge[lidx_v841(k, edge, nedges)],
                rho_edge[lidx_v841(k - 1, edge, nedges)])), mpas_sub(
                    vertical_velocity[lidx_v841(k, cell1, ncells)],
                    vertical_velocity[lidx_v841(k, cell0, ncells)]));
            lap = mpas_add(lap, flux);
            const float sum_k = mpas_add(mpas_add(mpas_add(
                kdiff[lidx_v841(k, cell0, ncells)],
                kdiff[lidx_v841(k, cell1, ncells)]),
                kdiff[lidx_v841(k - 1, cell0, ncells)]),
                kdiff[lidx_v841(k - 1, cell1, ncells)]);
            tend = mpas_add(tend, mpas_mul(mpas_mul(mpas_mul(
                flux, del2[edge]), 0.25f), sum_k));
        }
        delsq_w[tid] = lap;
        tendency[tid] = tend;
    }
}


// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void w_filter_lap4_v841_f32(
    const float* delsq_w,
    const float* del4,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* inv_dc_edge,
    const float* dv_edge,
    const float* inv_area_cell,
    const float h4,
    const int nlev,
    const int ncells,
    const int max_edges,
    float* tendency
) {
    /* Rows 1..nlev-1 only, as the per-cell loop ran them. */
    HORIZONTAL_ELEMENT_LOOP((nlev - 1) * ncells) {
        const int cell = element % ncells;
        const int k = 1 + element / ncells;
        const int count = n_edges_on_cell[cell];
        const int tid = lidx_v841(k, cell, ncells);
        float tend = tendency[tid];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_mul(mpas_mul(mpas_mul(mpas_mul(del4[edge],
                mpas_mul(h4, inv_area_cell[cell])), dv_edge[edge]),
                cell_sign_v841(cell, edge, cells_on_edge)), inv_dc_edge[edge]);
            tend = mpas_sub(tend, mpas_mul(factor, mpas_sub(
                delsq_w[lidx_v841(k, cell1, ncells)],
                delsq_w[lidx_v841(k, cell0, ncells)])));
        }
        tendency[tid] = tend;
    }
}


// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void momentum_filter_lap4_v841_f32(
    const float* rho_edge,
    const float* delsq_divergence,
    const float* delsq_vorticity,
    const float* del4,
    const int* cells_on_edge,
    const int* vertices_on_edge,
    const float* inv_dc_edge,
    const float* inv_dv_edge,
    const float h4,
    const float div_factor,
    const int nlev,
    const int ncells,
    const int nedges,
    const int nvertices,
    float* tendency
) {
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * nedges) return;
    const int edge = element % nedges;
    const int k = element / nedges;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    const int vertex0 = vertices_on_edge[2 * edge];
    const int vertex1 = vertices_on_edge[2 * edge + 1];
    const float inv_dc = inv_dc_edge[edge];
    const float inv_dv = fminf(inv_dv_edge[edge], 4.0f * inv_dc);
    const float scale = mpas_mul(del4[edge], h4);
    // Native prepares these rounded geometry coefficients outside the level loop.
    const float r_dc = mpas_mul(mpas_mul(scale, div_factor), inv_dc);
    const float r_dv = mpas_mul(scale, inv_dv);
    { // Each level reads saved fields and writes its own output word.
        const int tid = lidx_v841(k, edge, nedges);
        const float diffusion = mpas_mul(rho_edge[tid], mpas_sub(
            mpas_mul(mpas_sub(
                delsq_divergence[lidx_v841(k, cell1, ncells)],
                delsq_divergence[lidx_v841(k, cell0, ncells)]), r_dc),
            mpas_mul(mpas_sub(
                delsq_vorticity[lidx_v841(k, vertex1, nvertices)],
                delsq_vorticity[lidx_v841(k, vertex0, nvertices)]), r_dv)));
        tendency[tid] = mpas_sub(tendency[tid], diffusion);
    }
}


// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void momentum_filter_seed_v841_f32(
    const float* rho_edge, const float* divergence, const float* vorticity,
    const float* kdiff, const float* del2, const int* cells_on_edge,
    const int* vertices_on_edge, const float* inv_dc_edge, const float* inv_dv_edge,
    const int nlev, const int ncells, const int nedges, const int nvertices,
    float* delsq_u, float* tendency)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * nedges) return;
    const int edge = element % nedges;
    const int k = element / nedges;
    const int c0 = cells_on_edge[2*edge], c1 = cells_on_edge[2*edge+1];
    const int v0 = vertices_on_edge[2*edge], v1 = vertices_on_edge[2*edge+1];
    const float inv_dc = inv_dc_edge[edge];
    const float inv_dv = mpas_min(inv_dv_edge[edge], mpas_mul(4.0f, inv_dc));
    { // Each level reads saved fields and writes its own output word.
        const int tid=lidx_v841(k,edge,nedges);
        const float lap=mpas_sub(mpas_mul(mpas_sub(divergence[lidx_v841(k,c1,ncells)],divergence[lidx_v841(k,c0,ncells)]),inv_dc),mpas_mul(mpas_sub(vorticity[lidx_v841(k,v1,nvertices)],vorticity[lidx_v841(k,v0,nvertices)]),inv_dv));
        const float kedge=mpas_mul(0.5f,mpas_add(kdiff[lidx_v841(k,c0,ncells)],kdiff[lidx_v841(k,c1,ncells)]));
        delsq_u[tid]=lap;
        // Native adds del2 directly to the saved pressure-gradient pool,
        // then subtracts del4 from that already-rounded sum (F:721,803).
        tendency[tid]=mpas_add(tendency[tid],mpas_mul(mpas_mul(mpas_mul(rho_edge[tid],kedge),lap),del2[edge]));
    }
}

/* Fork mpas_atm_time_integration.F:5411-5444 at 1d6e370e. */
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void smagorinsky_rrfs_f32(
    const float *u, const float *v, const int *edges, const int *counts,
    const float *defc_a, const float *defc_b, const float strain_scale,
    const float ceiling, const int nl, const int nc, const int ne,
    const int me, float *kdiff)
{
    HORIZONTAL_ELEMENT_LOOP(nl * nc) {
        const int c = element % nc, k = element / nc;
        float diag = 0.0f, off = 0.0f;
        for (int j = 0; j < counts[c]; ++j) {
            const int o = c * me + j, e = edges[o];
            const float un = u[k * ne + e], vn = v[k * ne + e];
            diag = mpas_sub(mpas_add(diag, mpas_mul(defc_a[o], un)), mpas_mul(defc_b[o], vn));
            off = mpas_add(mpas_add(off, mpas_mul(defc_b[o], un)), mpas_mul(defc_a[o], vn));
        }
        kdiff[k * nc + c] = mpas_min(mpas_mul(strain_scale,
            mpas_sqrt(mpas_add(mpas_mul(diag,diag),mpas_mul(off,off)))),ceiling);
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:6506-6942 at 1d6e370e.
extern "C" __global__ void pv_cell_v841_f32(
    const float *pv_vertex, const int *vertices_on_cell,
    const int *n_edges_on_cell, const int *cells_on_vertex,
    const float *kite_areas, const float *inv_area_cell,
    const int nlev, const int ncells, const int nvertices,
    const int max_edges, const int vertex_degree, float *pv_cell)
{
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int vertex = vertices_on_cell[cell * max_edges + slot];
            const int kite = kite_slot_v841(
                vertex, cell, cells_on_vertex, vertex_degree);
            if (kite >= 0) {
                float contribution = mpas_mul(
                    kite_areas[vertex * vertex_degree + kite],
                    pv_vertex[lidx_v841(k, vertex, nvertices)]);
                contribution = mpas_mul(contribution, inv_area_cell[cell]);
                value = mpas_add(value, contribution);
            }
        }
        pv_cell[lidx_v841(k, cell, ncells)] = value;
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:6506-6942 at 1d6e370e.
extern "C" __global__ void pv_apvm_v841_f32(
    const float *normal_velocity, const float *tangential_velocity,
    const float *pv_vertex, const float *pv_cell,
    const int *vertices_on_edge, const int *cells_on_edge,
    const float *inv_dc_edge, const float *inv_dv_edge, const float scale,
    const int nlev, const int ncells, const int nedges, const int nvertices,
    float *pv_edge, float *grad_pv_normal, float *grad_pv_tangential)
{
    HORIZONTAL_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int vertex0 = vertices_on_edge[2 * edge];
        const int vertex1 = vertices_on_edge[2 * edge + 1];
        const int cell0 = cells_on_edge[2 * edge];
        const int cell1 = cells_on_edge[2 * edge + 1];
        const int index = lidx_v841(k, edge, nedges);
        const float grad_t = mpas_mul(mpas_sub(
            pv_vertex[lidx_v841(k, vertex1, nvertices)],
            pv_vertex[lidx_v841(k, vertex0, nvertices)]), inv_dv_edge[edge]);
        const float grad_n = mpas_mul(mpas_sub(
            pv_cell[lidx_v841(k, cell1, ncells)],
            pv_cell[lidx_v841(k, cell0, ncells)]), inv_dc_edge[edge]);
        grad_pv_tangential[index] = grad_t;
        grad_pv_normal[index] = grad_n;
        pv_edge[index] = mpas_sub(pv_edge[index], mpas_mul(scale, mpas_add(
            mpas_mul(tangential_velocity[index], grad_t),
            mpas_mul(normal_velocity[index], grad_n))));
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void mass_flux_divergence_v841_f32(
    const float *rho_u, const int *edges_on_cell,
    const int *n_edges_on_cell, const int *cells_on_edge,
    const float *dv_edge, const float *inv_area_cell,
    const int nlev, const int ncells, const int nedges,
    const int max_edges, float *mass_divergence)
{
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            value = mpas_add(value, mpas_mul(mpas_mul(
                cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                rho_u[lidx_v841(k, edge, nedges)]));
        }
        mass_divergence[lidx_v841(k, cell, ncells)] = mpas_mul(
            value, inv_area_cell[cell]);
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void pressure_gradient_v841_f32(
    const float *pressure_p, const float *dpdz, const float *cqu,
    const float *zz, const float *zxu, const int *cells_on_edge,
    const float *inv_dc_edge, const int nlev, const int ncells,
    const int nedges, float *tendency)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * nedges) return;
    const int edge = element % nedges;
    const int k = element / nedges;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    { // Each level reads saved fields and writes its own output word.
        const int index = lidx_v841(k, edge, nedges);
        float normal = mpas_mul(mpas_sub(
            pressure_p[lidx_v841(k, cell1, ncells)],
            pressure_p[lidx_v841(k, cell0, ncells)]), inv_dc_edge[edge]);
        normal = mpas_div(normal, mpas_mul(0.5f, mpas_add(
            zz[lidx_v841(k, cell1, ncells)],
            zz[lidx_v841(k, cell0, ncells)])));
        const float terrain = mpas_mul(mpas_mul(0.5f, zxu[index]), mpas_add(
            dpdz[lidx_v841(k, cell0, ncells)],
            dpdz[lidx_v841(k, cell1, ncells)]));
        tendency[index] = mpas_mul(mpas_sub(0.0f, cqu[index]),
            mpas_sub(normal, terrain));
    }
}

#ifndef HEX_HOIST_CELL
#define HEX_HOIST_CELL 10
#endif
#define HEX_CELL_CHUNK(rows) \
    const int hex_rows = (rows); \
    const int hex_chunks = (hex_rows + lpt - 1) / lpt; \
    const int hex_element = blockDim.x * blockIdx.x + threadIdx.x; \
    if (hex_element >= ncells * hex_chunks) return; \
    const int cell = hex_element % ncells; \
    const int hex_k0 = (hex_element / ncells) * lpt; \
    const int hex_k1 = min(hex_rows, hex_k0 + lpt);

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:6506-6942 at 1d6e370e.
extern "C" __global__ void cell_diagnostics_chunk_v841_f32(
    const float *normal_velocity, const float *ke_edge,
    const float *ke_vertex, const int *edges_on_cell,
    const int *n_edges_on_cell, const int *cells_on_edge,
    const int *vertices_on_cell, const int *cells_on_vertex,
    const float *dv_edge, const float *inv_area_cell,
    const float *kite_areas, const int nlev, const int ncells,
    const int nedges, const int nvertices, const int max_edges,
    const int vertex_degree, float *divergence, float *kinetic_energy,
    const int lpt)
{
    HEX_CELL_CHUNK(nlev)
    const int count = n_edges_on_cell[cell];
    const float inv_area = inv_area_cell[cell];
    const float ke_fact = 0.625f;
    if (count > HEX_HOIST_CELL) {
        for (int k = hex_k0; k < hex_k1; ++k) {
            float div = 0.0f;
            float ke = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int edge = edges_on_cell[cell * max_edges + slot];
                div = mpas_add(div, mpas_mul(mpas_mul(
                    cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                    normal_velocity[lidx_v841(k, edge, nedges)]));
                ke = mpas_add(ke, mpas_mul(
                    0.25f, ke_edge[lidx_v841(k, edge, nedges)]));
            }
            div = mpas_mul(div, inv_area);
            ke = mpas_mul(ke, inv_area);
            ke = mpas_mul(ke, ke_fact);
            for (int slot = 0; slot < count; ++slot) {
                const int vertex = vertices_on_cell[cell * max_edges + slot];
                const int kite = kite_slot_v841(
                    vertex, cell, cells_on_vertex, vertex_degree);
                if (kite >= 0) {
                    float correction = mpas_mul(mpas_sub(1.0f, ke_fact),
                        kite_areas[vertex * vertex_degree + kite]);
                    correction = mpas_mul(correction,
                        ke_vertex[lidx_v841(k, vertex, nvertices)]);
                    correction = mpas_mul(correction, inv_area);
                    ke = mpas_add(ke, correction);
                }
            }
            const int index = lidx_v841(k, cell, ncells);
            divergence[index] = div;
            kinetic_energy[index] = ke;
        }
        return;
    }
    int ed[HEX_HOIST_CELL];
    float sdv[HEX_HOIST_CELL];
    int vx[HEX_HOIST_CELL];
    float kc[HEX_HOIST_CELL];
    bool kv[HEX_HOIST_CELL];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
        const bool live = slot < count;
        const int edge = live ? edges_on_cell[cell * max_edges + slot] : 0;
        ed[slot] = edge;
        sdv[slot] = live ? mpas_mul(cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]) : 0.0f;
        const int vertex = live ? vertices_on_cell[cell * max_edges + slot] : 0;
        vx[slot] = vertex;
        const int kite = live ? kite_slot_v841(vertex, cell, cells_on_vertex, vertex_degree) : -1;
        kv[slot] = kite >= 0;
        kc[slot] = kite >= 0 ? mpas_mul(mpas_sub(1.0f, ke_fact),
            kite_areas[vertex * vertex_degree + kite]) : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        float div = 0.0f;
        float ke = 0.0f;
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
            if (slot < count) {
                div = mpas_add(div, mpas_mul(sdv[slot],
                    normal_velocity[lidx_v841(k, ed[slot], nedges)]));
                ke = mpas_add(ke, mpas_mul(
                    0.25f, ke_edge[lidx_v841(k, ed[slot], nedges)]));
            }
        }
        div = mpas_mul(div, inv_area);
        ke = mpas_mul(ke, inv_area);
        ke = mpas_mul(ke, ke_fact);
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
            if (slot < count && kv[slot]) {
                float correction = mpas_mul(kc[slot],
                    ke_vertex[lidx_v841(k, vx[slot], nvertices)]);
                correction = mpas_mul(correction, inv_area);
                ke = mpas_add(ke, correction);
            }
        }
        const int index = lidx_v841(k, cell, ncells);
        divergence[index] = div;
        kinetic_energy[index] = ke;
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:6506-6942 at 1d6e370e.
extern "C" __global__ void pv_cell_chunk_v841_f32(
    const float *pv_vertex, const int *vertices_on_cell,
    const int *n_edges_on_cell, const int *cells_on_vertex,
    const float *kite_areas, const float *inv_area_cell,
    const int nlev, const int ncells, const int nvertices,
    const int max_edges, const int vertex_degree, float *pv_cell,
    const int lpt)
{
    HEX_CELL_CHUNK(nlev)
    const int count = n_edges_on_cell[cell];
    const float inv_area = inv_area_cell[cell];
    if (count > HEX_HOIST_CELL) {
        for (int k = hex_k0; k < hex_k1; ++k) {
            float value = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int vertex = vertices_on_cell[cell * max_edges + slot];
                const int kite = kite_slot_v841(
                    vertex, cell, cells_on_vertex, vertex_degree);
                if (kite >= 0) {
                    float contribution = mpas_mul(
                        kite_areas[vertex * vertex_degree + kite],
                        pv_vertex[lidx_v841(k, vertex, nvertices)]);
                    contribution = mpas_mul(contribution, inv_area);
                    value = mpas_add(value, contribution);
                }
            }
            pv_cell[lidx_v841(k, cell, ncells)] = value;
        }
        return;
    }
    int vx[HEX_HOIST_CELL];
    float ka[HEX_HOIST_CELL];
    bool kv[HEX_HOIST_CELL];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
        const bool live = slot < count;
        const int vertex = live ? vertices_on_cell[cell * max_edges + slot] : 0;
        vx[slot] = vertex;
        const int kite = live ? kite_slot_v841(vertex, cell, cells_on_vertex, vertex_degree) : -1;
        kv[slot] = kite >= 0;
        ka[slot] = kite >= 0 ? kite_areas[vertex * vertex_degree + kite] : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        float value = 0.0f;
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
            if (slot < count && kv[slot]) {
                float contribution = mpas_mul(ka[slot],
                    pv_vertex[lidx_v841(k, vx[slot], nvertices)]);
                contribution = mpas_mul(contribution, inv_area);
                value = mpas_add(value, contribution);
            }
        }
        pv_cell[lidx_v841(k, cell, ncells)] = value;
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void mass_flux_divergence_chunk_v841_f32(
    const float *rho_u, const int *edges_on_cell,
    const int *n_edges_on_cell, const int *cells_on_edge,
    const float *dv_edge, const float *inv_area_cell,
    const int nlev, const int ncells, const int nedges,
    const int max_edges, float *mass_divergence, const int lpt)
{
    HEX_CELL_CHUNK(nlev)
    const int count = n_edges_on_cell[cell];
    const float inv_area = inv_area_cell[cell];
    if (count > HEX_HOIST_CELL) {
        for (int k = hex_k0; k < hex_k1; ++k) {
            float value = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int edge = edges_on_cell[cell * max_edges + slot];
                value = mpas_add(value, mpas_mul(mpas_mul(
                    cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                    rho_u[lidx_v841(k, edge, nedges)]));
            }
            mass_divergence[lidx_v841(k, cell, ncells)] = mpas_mul(value, inv_area);
        }
        return;
    }
    int ed[HEX_HOIST_CELL];
    float sdv[HEX_HOIST_CELL];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
        const bool live = slot < count;
        const int edge = live ? edges_on_cell[cell * max_edges + slot] : 0;
        ed[slot] = edge;
        sdv[slot] = live ? mpas_mul(cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]) : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        float value = 0.0f;
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
            if (slot < count) {
                value = mpas_add(value, mpas_mul(sdv[slot],
                    rho_u[lidx_v841(k, ed[slot], nedges)]));
            }
        }
        mass_divergence[lidx_v841(k, cell, ncells)] = mpas_mul(value, inv_area);
    }
}
"""


class CudaHorizontalRRFS(CudaHorizontal):
    """Fork stored-inverse operators, deformation mixing and diagnostics."""

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6942 at 1d6e370e.
    def __init__(
        self,
        mesh: Any,
        n_vert_levels: int,
        context: CudaV841Context,
        *,
        kernel_cache: Any | None = None,
    ) -> None:
        from .cuda_compile_rrfs import RRFSKernelCache
        kernel_cache=RRFSKernelCache() if kernel_cache is None else kernel_cache
        super().__init__(mesh, n_vert_levels, kernel_cache=kernel_cache)
        mesh.validate()
        if self.vertex_degree != 3:
            raise ValueError("v8.4.1 CUDA requires MPAS vertexDegree=3")
        self.v841 = context
        context.validate(
            n_vert_levels=self.nlev,
            n_cells=self.ncells,
            n_edges=self.nedges,
            n_vertices=self.nvertices,
        )
        self._v841_kernels: dict[str, Any] = {}
        self._deformation_v841: dict[str, Any] | None = None

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6942 at 1d6e370e.
    def _kernel(self,name):
        """Shared stencil arithmetic; GNU mesh-scaling power, strict NVRTC."""
        from .cuda_horizontal import _CUDA_SOURCE as shared
        from .native_powf import CUDA_NATIVE_POWF
        source=CUDA_NATIVE_POWF+shared.replace('powf(mean_density,','mpas_native_powf_positive(mean_density,')
        # Generic cache delegation, not a static AOT kernel declaration.
        # The runtime compile manifest authenticates the caller's full source.
        compile_kernel = self.kernel_cache.raw_kernel
        return compile_kernel(name,source,module_key='hexcore.cuda_horizontal_shared_rrfs')

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6942 at 1d6e370e.
    def _kernel_v841(self, name: str) -> Any:
        from .cuda_compile_rrfs import kernel
        return kernel(_CUDA_SOURCE,"hexcore.cuda_horizontal_rrfs",name,self.kernel_cache)

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6942 at 1d6e370e.
    def _launch_v841(self, name: str, total: int, args: tuple[Any, ...]) -> None:
        if total < 1:
            return
        threads = 128
        blocks = (int(total) + threads - 1) // threads
        self._kernel_v841(name)((blocks,), (threads,), args)

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6942 at 1d6e370e.
    def mass_flux_divergence(self, rho_u: Any) -> Any:
        rho_u = self._edge_field("rho_u", rho_u)
        out = self.cp.empty((self.nlev, self.ncells), dtype=self.cp.float32)
        arguments = (
            rho_u,
            self.mesh.edges_on_cell,
            self.mesh.n_edges_on_cell,
            self.mesh.cells_on_edge,
            self.mesh.dv_edge,
            self.v841.inv_area_cell,
            np.int32(self.nlev),
            np.int32(self.ncells),
            np.int32(self.nedges),
            np.int32(self.max_edges),
        )
        self._launch_cell_chunked("mass_flux_divergence_chunk_v841_f32",
                                  "mass_flux_divergence_v841_f32", arguments, (out,))
        return out

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6942 at 1d6e370e.
    def _launch_cell_chunked(self, chunked: str, element: str, inputs: tuple[Any, ...],
                             outputs: tuple[Any, ...]) -> None:
        """Chunked hoisted form when selected (bits identical), else the element kernel."""
        from . import cuda_edge_hoist
        cp = self.cp
        if not cuda_edge_hoist.enabled(self.nedges):
            self._launch_v841(element, self.nlev * self.ncells, (*inputs, *outputs))
            return
        lpt = cuda_edge_hoist.levels_per_thread(
            chunked, self.ncells, self.nlev, lambda value: self._launch_v841(
                chunked, self.ncells * cuda_edge_hoist.chunks(self.nlev, value),
                (*inputs, *outputs, np.int32(value))), cp)
        self._launch_v841(chunked, self.ncells * cuda_edge_hoist.chunks(self.nlev, lpt),
                          (*inputs, *outputs, np.int32(lpt)))
        if cuda_edge_hoist.check_wanted(chunked, cp):
            reference = tuple(cp.empty_like(value) for value in outputs)
            self._launch_v841(element, self.nlev * self.ncells, (*inputs, *reference))
            for value, expected in zip(outputs, reference):
                cuda_edge_hoist.compare(chunked, value, expected, cp)

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6942 at 1d6e370e.
    def solve_diagnostics(
        self,
        rho: Any,
        rho_u: Any,
        *,
        dt: float,
        apvm_upwinding: float = 0.0,
        normal_velocity: Any | None = None,
        cached_tangential_velocity: Any | None = None,
        rk_step: int | None = 3,
    ) -> CudaSolveDiagnostics:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(apvm_upwinding) or apvm_upwinding < 0.0:
            raise ValueError("apvm_upwinding must be finite and non-negative")
        cp = self.cp
        normal, rho_edge, ke_edge = self._edge_state(rho, rho_u, normal_velocity)
        reconstruct_v = rk_step is None or rk_step == 3
        if reconstruct_v:
            tangential = cp.empty_like(normal)
            tangential_arguments = (
                normal,
                self.mesh.edges_on_edge,
                self.mesh.n_edges_on_edge,
                self.mesh.weights_on_edge,
                np.int32(self.nlev),
                np.int32(self.nedges),
                np.int32(self.max_edges2),
            )
            from . import cuda_edge_hoist
            if cuda_edge_hoist.enabled(self.nedges):
                lpt = cuda_edge_hoist.levels_per_thread(
                    "tangential_velocity_hoist_v841_f32", self.nedges, self.nlev,
                    lambda value: self._launch_v841(
                        "tangential_velocity_hoist_v841_f32",
                        self.nedges * cuda_edge_hoist.chunks(self.nlev, value),
                        (*tangential_arguments, tangential, np.int32(value))), cp)
                self._launch_v841("tangential_velocity_hoist_v841_f32",
                                  self.nedges * cuda_edge_hoist.chunks(self.nlev, lpt),
                                  (*tangential_arguments, tangential, np.int32(lpt)))
                if cuda_edge_hoist.check_wanted("tangential_velocity_hoist_v841_f32", cp):
                    element = cp.empty_like(tangential)
                    self._launch("tangential_velocity_f32", self.nedges, (*tangential_arguments, element))
                    cuda_edge_hoist.compare("tangential_velocity_hoist_v841_f32", tangential, element, cp)
            else:
                self._launch("tangential_velocity_f32", self.nedges, (*tangential_arguments, tangential))
        else:
            if cached_tangential_velocity is None:
                raise ValueError("RK1/RK2 diagnostics require cached tangential velocity")
            # Bound, not copied: every consumer of the tangential velocity in
            # this tree takes it as a const float* -- the APVM kernel, the
            # v8.4.1 mixing operator and the vertex diagnostics all read it and
            # write their own outputs -- so the RK1/RK2 reuse of the RK3 field
            # needs a reference, not a private nlev x nEdges image per stage.
            tangential = self._edge_field(
                "cached_tangential_velocity", cached_tangential_velocity
            )

        vorticity = cp.empty((self.nlev, self.nvertices), dtype=cp.float32)
        ke_vertex = cp.empty_like(vorticity)
        pv_vertex = cp.empty_like(vorticity)
        self._launch_v841(
            "vertex_diagnostics_v841_f32",
            self.nlev * self.nvertices,
            (
                normal,
                ke_edge,
                self.mesh.edges_on_vertex,
                self.mesh.vertices_on_edge,
                self.mesh.dc_edge,
                self.v841.inv_area_triangle,
                self.mesh.f_vertex,
                np.int32(self.nlev),
                np.int32(self.nedges),
                np.int32(self.nvertices),
                np.int32(self.vertex_degree),
                vorticity,
                ke_vertex,
                pv_vertex,
            ),
        )
        divergence = cp.empty((self.nlev, self.ncells), dtype=cp.float32)
        kinetic_energy = cp.empty_like(divergence)
        self._launch_cell_chunked(
            "cell_diagnostics_chunk_v841_f32", "cell_diagnostics_v841_f32",
            (
                normal,
                ke_edge,
                ke_vertex,
                self.mesh.edges_on_cell,
                self.mesh.n_edges_on_cell,
                self.mesh.cells_on_edge,
                self.mesh.vertices_on_cell,
                self.mesh.cells_on_vertex,
                self.mesh.dv_edge,
                self.v841.inv_area_cell,
                self.mesh.kite_areas_on_vertex,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.nvertices),
                np.int32(self.max_edges),
                np.int32(self.vertex_degree),
            ),
            (divergence, kinetic_energy),
        )
        pv_edge = cp.empty_like(normal)
        self._launch(
            "pv_edge_base_f32",
            self.nedges,
            (
                pv_vertex,
                self.mesh.vertices_on_edge,
                np.int32(self.nlev),
                np.int32(self.nedges),
                np.int32(self.nvertices),
                pv_edge,
            ),
        )
        pv_cell = cp.zeros((self.nlev, self.ncells), dtype=cp.float32)
        grad_normal = cp.zeros_like(normal)
        grad_tangential = cp.zeros_like(normal)
        if apvm_upwinding > 0.0:
            self._launch_cell_chunked(
                "pv_cell_chunk_v841_f32", "pv_cell_v841_f32",
                (
                    pv_vertex,
                    self.mesh.vertices_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_vertex,
                    self.mesh.kite_areas_on_vertex,
                    self.v841.inv_area_cell,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.nvertices),
                    np.int32(self.max_edges),
                    np.int32(self.vertex_degree),
                ),
                (pv_cell,),
            )
            scale = np.float32(apvm_upwinding) * np.float32(dt)
            self._launch_v841(
                "pv_apvm_v841_f32",
                self.nlev * self.nedges,
                (
                    normal,
                    tangential,
                    pv_vertex,
                    pv_cell,
                    self.mesh.vertices_on_edge,
                    self.mesh.cells_on_edge,
                    self.v841.inv_dc_edge,
                    self.v841.inv_dv_edge,
                    scale,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.nedges),
                    np.int32(self.nvertices),
                    pv_edge,
                    grad_normal,
                    grad_tangential,
                ),
            )
        return CudaSolveDiagnostics(
            h_edge=rho_edge,
            normal_velocity=normal,
            tangential_velocity=tangential,
            vorticity=vorticity,
            divergence=divergence,
            kinetic_energy=kinetic_energy,
            pv_edge=pv_edge,
            pv_vertex=pv_vertex,
            pv_cell=pv_cell,
            grad_pv_normal=grad_normal,
            grad_pv_tangential=grad_tangential,
        )

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6942 at 1d6e370e.
    def attach_deformation_weights_v841(self, weights: Any) -> dict[str, Any]:
        """Upload host float32 v8.4.1 deformation weights onto the device.

        ``weights`` is a :class:`woof.hex.mixing_v841.DeformationWeightsV841`
        computed by the CPU authority; the device copies are byte-identical
        H2D transfers (no device arithmetic touches them at upload).
        """

        cp = self.cp
        shape = (self.ncells, self.max_edges)
        arrays = {}
        receipts: dict[str, Any] = {}
        for name in ("coef_c2", "coef_s2", "coef_cs"):
            host = np.ascontiguousarray(
                np.asarray(getattr(weights, name), dtype=np.float32)
            )
            if host.shape != shape:
                raise ValueError(f"{name} shape {host.shape} != {shape}")
            if not np.all(np.isfinite(host)):
                raise ValueError(f"{name} contains non-finite values")
            arrays[name] = cp.asarray(host)
            receipts[name] = {
                "shape": list(shape),
                "dtype": "float32",
                "nonzero": int(np.count_nonzero(host)),
                "sha256": hashlib.sha256(
                    host.tobytes(order="C")
                ).hexdigest(),
            }
        self._deformation_v841 = arrays
        return receipts

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6942 at 1d6e370e.
    def compute_dry_mixing_tendencies_v841(
        self,
        normal_velocity: Any,
        tangential_velocity: Any,
        vertical_velocity: Any,
        theta_m: Any,
        rho_edge: Any,
        divergence: Any,
        vorticity: Any,
        *,
        dt: float,
        config: Any | None = None,
        cam_damping_levels: int = 4,
        n_cells_solve: int | None = None,
        initial_u_tendency: Any | None = None,
        mesh_scaling_del2: Any | None = None,
        mesh_scaling_del4: Any | None = None,
    ) -> CudaDryMixingTendencies:
        """Fork mpas_atm_time_integration.F:5393-6409 at 1d6e370e: dry mixing."""

        from .mixing_v841 import V841MixingConfig
        from .mixing_rrfs import RRFSMixingConfig

        cfg = RRFSMixingConfig() if config is None else config
        if not isinstance(cfg, V841MixingConfig):
            raise TypeError("config must be a mixing_v841.V841MixingConfig")
        cfg.validate()
        timestep = float(dt)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("dt must be finite and positive")
        normal = self._edge_field("normal_velocity", normal_velocity)
        tangential = self._edge_field(
            "tangential_velocity", tangential_velocity
        )
        vertical = self._interface_field("vertical_velocity", vertical_velocity)
        theta = self._cell_field("theta_m", theta_m)
        edge_rho = self._edge_field("rho_edge", rho_edge)
        div = self._cell_field("divergence", divergence)
        vort = self._field(
            "vorticity", vorticity, (self.nlev, self.nvertices)
        )
        cp = self.cp

        configured_length = float(cfg.config_len_disp)
        length = (
            float(self.mesh.nominal_min_dc)
            if configured_length == 0.0
            else configured_length
        )
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(
                "config_len_disp or DeviceMesh.nominal_min_dc must be positive"
            )
        # Host-side scalar staging matches the float32 CPU authority bit for
        # bit: (c_s*len)**2, invDt = 1/dt, ceiling = (0.01*len**2)*invDt,
        # h4 = visc4*len**3 (dissipation-models lines 189-190, 199-200;
        # time-integration line 6323).
        length32 = np.float32(length)
        cs = np.float32(cfg.config_smagorinsky_coef)
        strain_scale = np.float32((cs * length32) * (cs * length32))
        inv_dt = np.float32(np.float32(1.0) / np.float32(timestep))
        ceiling = np.float32(
            (np.float32(0.01) * (length32 * length32)) * inv_dt
        )
        visc4 = np.float32(cfg.config_visc4_2dsmag)
        h4 = np.float32(visc4 * ((length32 * length32) * length32))
        theta_h4 = h4
        if cfg.config_horiz_mixing != '2d_smagorinsky':
            fixed=cfg.config_horiz_mixing=='2d_fixed'
            h4=np.float32(getattr(cfg,'config_h_mom_eddy_visc4',0.0) if fixed else 0.0)
            theta_h4=np.float32(getattr(cfg,'config_h_theta_eddy_visc4',0.0) if fixed else 0.0)

        if (mesh_scaling_del2 is None) != (mesh_scaling_del4 is None):
            raise ValueError('native mesh scaling requires both del2 and del4 pools')
        del2 = cp.empty((self.nedges,), dtype=cp.float32) if mesh_scaling_del2 is None else mesh_scaling_del2
        del4 = cp.empty_like(del2) if mesh_scaling_del4 is None else mesh_scaling_del4
        if mesh_scaling_del2 is None:
            self._launch(
            "mixing_scaling_f32",
            self.nedges,
            (
                self.mesh.cells_on_edge,
                self.mesh.mesh_density,
                np.int32(self.nedges),
                np.int32(bool(cfg.config_h_ScaleWithMesh)),
                del2,
                del4,
            ),
            )

        self.rrfs_scaling_del2=del2
        self.rrfs_scaling_del4=del4
        kdiff = cp.empty((self.nlev, self.ncells), dtype=cp.float32)
        if cfg.config_horiz_mixing=='2d_smagorinsky':
            self._launch_v841(
                "smagorinsky_rrfs_f32",
                self.nlev * self.ncells,
                (
                    normal,
                    tangential,
                    self.mesh.edges_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.defc_a,
                    self.mesh.defc_b,
                    strain_scale,
                    ceiling,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.nedges),
                    np.int32(self.max_edges),
                    kdiff,
                ),
            )
        else:
            kdiff.fill(np.float32(getattr(cfg,'config_h_theta_eddy_visc2',0.0)
                if cfg.config_horiz_mixing=='2d_fixed' else 0.0))

        if cfg.config_mpas_cam_coef > 0:
            from .cuda_upper_damping_v841 import apply_cam_floor_cuda
            apply_cam_floor_cuda(kdiff, length=length, coefficient=cfg.config_mpas_cam_coef,
                levels=cam_damping_levels, solved=n_cells_solve, cache=self.kernel_cache)
        delsq_u = cp.empty_like(normal)
        tend_u = cp.zeros_like(normal) if initial_u_tendency is None else cp.array(
            self._edge_field("initial_u_tendency", initial_u_tendency), copy=True
        )
        launch_lap2 = self._launch_v841
        launch_lap2(
            "momentum_filter_seed_v841_f32",
            self.nlev * self.nedges,
            (
                edge_rho,
                div,
                vort,
                kdiff,
                del2,
                self.mesh.cells_on_edge,
                self.mesh.vertices_on_edge,
                self.v841.inv_dc_edge,
                self.v841.inv_dv_edge,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.nvertices),
                delsq_u,
                tend_u,
            ),
        )
        delsq_div = cp.zeros((self.nlev, self.ncells), dtype=cp.float32)
        delsq_vort = cp.zeros((self.nlev, self.nvertices), dtype=cp.float32)
        if h4 > np.float32(0.0):
            self._launch_v841(
                "laplacian_divergence_v841_f32",
                self.nlev * self.ncells,
                (
                    delsq_u,
                    self.mesh.edges_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_edge,
                    self.mesh.dv_edge,
                    self.v841.inv_area_cell,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.nedges),
                    np.int32(self.max_edges),
                    delsq_div,
                ),
            )
            self._launch_v841(
                "laplacian_vorticity_v841_f32",
                self.nlev * self.nvertices,
                (
                    delsq_u,
                    self.mesh.edges_on_vertex,
                    self.mesh.vertices_on_edge,
                    self.mesh.dc_edge,
                    self.v841.inv_area_triangle,
                    np.int32(self.nlev),
                    np.int32(self.nedges),
                    np.int32(self.nvertices),
                    np.int32(self.vertex_degree),
                    delsq_vort,
                ),
            )
            self._launch_v841(
                "momentum_filter_lap4_v841_f32",
                self.nlev * self.nedges,
                (
                    edge_rho,
                    delsq_div,
                    delsq_vort,
                    del4,
                    self.mesh.cells_on_edge,
                    self.mesh.vertices_on_edge,
                    self.v841.inv_dc_edge,
                    self.v841.inv_dv_edge,
                    h4,
                    np.float32(cfg.config_del4u_div_factor),
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.nedges),
                    np.int32(self.nvertices),
                    tend_u,
                ),
            )

        delsq_theta = cp.empty_like(theta)
        tend_theta = cp.empty_like(theta)
        self._launch_v841(
            "theta_filter_lap2_v841_f32",
            self.nlev * self.ncells,
            (
                theta,
                edge_rho,
                kdiff,
                del2,
                self.mesh.edges_on_cell,
                self.mesh.n_edges_on_cell,
                self.mesh.cells_on_edge,
                self.v841.inv_dc_edge,
                self.mesh.dv_edge,
                self.v841.inv_area_cell,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.max_edges),
                delsq_theta,
                tend_theta,
            ),
        )
        if theta_h4 > np.float32(0.0):
            self._launch_v841(
                "theta_filter_lap4_v841_f32",
                self.nlev * self.ncells,
                (
                    delsq_theta,
                    del4,
                    self.mesh.edges_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_edge,
                    self.v841.inv_dc_edge,
                    self.mesh.dv_edge,
                    self.v841.inv_area_cell,
                    theta_h4,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.max_edges),
                    tend_theta,
                ),
            )

        delsq_w = cp.empty((self.nlev, self.ncells), dtype=cp.float32)
        tend_w = cp.zeros((self.nlev + 1, self.ncells), dtype=cp.float32)
        self._launch_v841(
            "w_filter_lap2_v841_f32",
            self.nlev * self.ncells,
            (
                vertical,
                edge_rho,
                kdiff,
                del2,
                self.mesh.edges_on_cell,
                self.mesh.n_edges_on_cell,
                self.mesh.cells_on_edge,
                self.v841.inv_dc_edge,
                self.mesh.dv_edge,
                self.v841.inv_area_cell,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.max_edges),
                delsq_w,
                tend_w,
            ),
        )
        if h4 > np.float32(0.0):
            self._launch_v841(
                "w_filter_lap4_v841_f32",
                (self.nlev - 1) * self.ncells,
                (
                    delsq_w,
                    del4,
                    self.mesh.edges_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_edge,
                    self.v841.inv_dc_edge,
                    self.mesh.dv_edge,
                    self.v841.inv_area_cell,
                    h4,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.max_edges),
                    tend_w,
                ),
            )

        return CudaDryMixingTendencies(
            kdiff=kdiff,
            h_mom_eddy_visc4=h4,
            h_theta_eddy_visc4=theta_h4,
            tend_u_euler=tend_u,
            tend_w_euler=tend_w,
            tend_theta_euler=tend_theta,
            delsq_u=delsq_u,
            delsq_divergence=delsq_div,
            delsq_vorticity=delsq_vort,
            delsq_w=delsq_w,
            delsq_theta=delsq_theta,
        )

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6942 at 1d6e370e.
    def pressure_gradient_euler_tendency(
        self,
        pressure_perturbation: Any,
        dpdz: Any,
        cqu: Any,
        zz: Any,
        zxu: Any,
    ) -> Any:
        pressure = self._cell_field("pressure_perturbation", pressure_perturbation)
        dpdz = self._cell_field("dpdz", dpdz)
        cqu = self._edge_field("cqu", cqu)
        zz = self._cell_field("zz", zz)
        zxu = self._edge_field("zxu", zxu)
        out = self.cp.empty_like(cqu)
        self._launch_v841(
            "pressure_gradient_v841_f32",
            self.nlev * self.nedges,
            (
                pressure,
                dpdz,
                cqu,
                zz,
                zxu,
                self.mesh.cells_on_edge,
                self.v841.inv_dc_edge,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                out,
            ),
        )
        return out


__all__ = ["CudaHorizontalRRFS"]
