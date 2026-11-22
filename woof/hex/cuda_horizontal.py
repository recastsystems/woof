"""Explicit CUDA C++ horizontal MPAS-A kernels for the RTX 5090 lane.

This module is deliberately a large coherent device-resident port, not a
collection of CuPy expressions.  Every numerical operator is a ``RawKernel``
compiled through :mod:`woof.hex.cuda_backend`; inputs and outputs remain
CuPy arrays in the CPU authority's logical level-major order:
``index = level * nEntity + entity``.

Every kernel launches one thread per horizontal topology owner (cell, edge, or
vertex) and walks levels in ascending order inside that thread.  Cell stencil
kernels are therefore literally thread-per-cell and retain ascending MPAS
connectivity-slot accumulation.

The float32 kernels cover frozen MPAS-A v8.2.3 horizontal mass/density fluxes,
C-grid diagnostics, pressure gradient, the complete JW Smagorinsky filter
bundle, and acoustic divergence damping.  Stencil slots are accumulated in
ascending MPAS order without atomics or warp reductions so CPU/GPU comparison
has a defensible rounding envelope.  ``--fmad=false`` is intentional for the
first authority pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cuda_backend import DeviceMesh, KernelCache, require_cuda
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .native_powf import CUDA_NATIVE_POWF


CUDA_HORIZONTAL_EVIDENCE = "cuda-horizontal-f32-implemented-unverified-whole-step"
_BLOCK_SIZE = 256


@dataclass(frozen=True, slots=True)
class CudaRecoveredEdges:
    normal_velocity: Any
    rho_edge: Any


@dataclass(frozen=True, slots=True)
class CudaSolveDiagnostics:
    h_edge: Any
    normal_velocity: Any
    tangential_velocity: Any
    vorticity: Any
    divergence: Any
    kinetic_energy: Any
    pv_edge: Any
    pv_vertex: Any
    pv_cell: Any
    grad_pv_normal: Any
    grad_pv_tangential: Any


@dataclass(frozen=True, slots=True)
class CudaDryMixingTendencies:
    kdiff: Any
    h_mom_eddy_visc4: np.float32
    h_theta_eddy_visc4: np.float32
    tend_u_euler: Any
    tend_w_euler: Any
    tend_theta_euler: Any
    delsq_u: Any
    delsq_divergence: Any
    delsq_vorticity: Any
    delsq_w: Any
    delsq_theta: Any


_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""

/* Level-independent kernels walk a flat (level, owner) element index, owner
   fastest, in a grid-stride loop; each element runs the former per-owner
   loop body for its level unchanged, so the launch geometry is not in the
   arithmetic.  The forecast launches one thread per element. */
#define HORIZONTAL_ELEMENT_LOOP(total) \
    for (int element = blockDim.x * blockIdx.x + threadIdx.x; \
         element < (total); element += gridDim.x * blockDim.x)
__device__ __forceinline__ int lidx(const int k, const int i, const int n) {
    return k * n + i;
}

__device__ __forceinline__ float cell_sign(
    const int cell, const int edge, const int* cells_on_edge
) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}

__device__ __forceinline__ float vertex_sign(
    const int vertex, const int edge, const int* vertices_on_edge
) {
    return vertices_on_edge[2 * edge + 1] == vertex ? 1.0f : -1.0f;
}

__device__ __forceinline__ int kite_slot(
    const int vertex,
    const int cell,
    const int* cells_on_vertex,
    const int vertex_degree
) {
    for (int slot = 0; slot < vertex_degree; ++slot) {
        if (cells_on_vertex[vertex * vertex_degree + slot] == cell) return slot;
    }
    return -1;
}

extern "C" __global__ void recover_edge_f32(
    const float* rho,
    const float* rho_u,
    const float* dc_edge,
    const float* dv_edge,
    const int* cells_on_edge,
    const int nlev,
    const int ncells,
    const int nedges,
    float* normal_velocity,
    float* rho_edge,
    float* ke_edge
) {
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, edge, nedges);
        const float edge_rho = mpas_mul(0.5f, mpas_add(
            rho[lidx(k, cell0, ncells)], rho[lidx(k, cell1, ncells)]
        ));
        const float u = mpas_div(rho_u[tid], edge_rho);
        rho_edge[tid] = edge_rho;
        normal_velocity[tid] = u;
        // Native MPAS: efac*u**2. Round the square before the area product.
        ke_edge[tid] = mpas_mul(mpas_mul(
            dc_edge[edge], dv_edge[edge]), mpas_mul(u, u));
    }
}

extern "C" __global__ void edge_fields_from_saved_u_f32(
    const float* rho,
    const float* normal_velocity,
    const float* dc_edge,
    const float* dv_edge,
    const int* cells_on_edge,
    const int nlev,
    const int ncells,
    const int nedges,
    float* rho_edge,
    float* ke_edge
) {
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, edge, nedges);
        const float edge_rho = mpas_mul(0.5f, mpas_add(
            rho[lidx(k, cell0, ncells)], rho[lidx(k, cell1, ncells)]
        ));
        const float u = normal_velocity[tid];
        rho_edge[tid] = edge_rho;
        // Native MPAS: efac*u**2. Round the square before the area product.
        ke_edge[tid] = mpas_mul(mpas_mul(
            dc_edge[edge], dv_edge[edge]), mpas_mul(u, u));
    }
}

extern "C" __global__ void tangential_velocity_f32(
    const float* normal_velocity,
    const int* edges_on_edge,
    const int* n_edges_on_edge,
    const float* weights_on_edge,
    const int nlev,
    const int nedges,
    const int max_edges2,
    float* tangential_velocity
) {
    HORIZONTAL_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int count = n_edges_on_edge[edge];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int offset = edge * max_edges2 + slot;
            const int neighbor = edges_on_edge[offset];
            value = mpas_add(value, mpas_mul(weights_on_edge[offset],
                normal_velocity[lidx(k, neighbor, nedges)]));
        }
        tangential_velocity[lidx(k, edge, nedges)] = value;
    }
}

extern "C" __global__ void vertex_diagnostics_f32(
    const float* normal_velocity,
    const float* ke_edge,
    const int* edges_on_vertex,
    const int* vertices_on_edge,
    const float* dc_edge,
    const float* area_triangle,
    const float* f_vertex,
    const int nlev,
    const int nedges,
    const int nvertices,
    const int vertex_degree,
    float* vorticity,
    float* ke_vertex,
    float* pv_vertex
) {
    const int vertex = blockDim.x * blockIdx.x + threadIdx.x;
    if (vertex >= nvertices) return;
    for (int k = 0; k < nlev; ++k) {
        float vort = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex * vertex_degree + slot];
            if (edge < 0) continue;
            vort = mpas_add(vort, mpas_mul(mpas_mul(
                vertex_sign(vertex, edge, vertices_on_edge), dc_edge[edge]),
                normal_velocity[lidx(k, edge, nedges)]));
            ke = mpas_add(ke, mpas_mul(
                0.25f, ke_edge[lidx(k, edge, nedges)]));
        }
        vort = mpas_div(vort, area_triangle[vertex]);
        ke = mpas_div(ke, area_triangle[vertex]);
        const int tid = lidx(k, vertex, nvertices);
        vorticity[tid] = vort;
        ke_vertex[tid] = ke;
        pv_vertex[tid] = mpas_add(f_vertex[vertex], vort);
    }
}

extern "C" __global__ void cell_diagnostics_f32(
    const float* normal_velocity,
    const float* ke_edge,
    const float* ke_vertex,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const int* vertices_on_cell,
    const int* cells_on_vertex,
    const float* dv_edge,
    const float* area_cell,
    const float* kite_areas,
    const int nlev,
    const int ncells,
    const int nedges,
    const int nvertices,
    const int max_edges,
    const int vertex_degree,
    float* divergence,
    float* kinetic_energy
) {
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int count = n_edges_on_cell[cell];
    const float inv_area = 1.0f / area_cell[cell];
    for (int k = 0; k < nlev; ++k) {
        float div = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            div = mpas_add(div, mpas_mul(mpas_mul(
                cell_sign(cell, edge, cells_on_edge), dv_edge[edge]),
                normal_velocity[lidx(k, edge, nedges)]));
            ke = mpas_add(ke, mpas_mul(
                0.25f, ke_edge[lidx(k, edge, nedges)]));
        }
        div = mpas_mul(div, inv_area);
        ke = mpas_mul(ke, inv_area);
        ke = mpas_mul(ke, 0.625f);
        for (int slot = 0; slot < count; ++slot) {
            const int vertex = vertices_on_cell[cell * max_edges + slot];
            const int kite = kite_slot(
                vertex, cell, cells_on_vertex, vertex_degree
            );
            if (kite >= 0) {
                ke = mpas_add(ke, mpas_mul(mpas_mul(mpas_mul(
                    0.375f, kite_areas[vertex * vertex_degree + kite]),
                    ke_vertex[lidx(k, vertex, nvertices)]), inv_area));
            }
        }
        const int tid = lidx(k, cell, ncells);
        divergence[tid] = div;
        kinetic_energy[tid] = ke;
    }
}

extern "C" __global__ void pv_edge_base_f32(
    const float* pv_vertex,
    const int* vertices_on_edge,
    const int nlev,
    const int nedges,
    const int nvertices,
    float* pv_edge
) {
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int vertex0 = vertices_on_edge[2 * edge];
    const int vertex1 = vertices_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        pv_edge[lidx(k, edge, nedges)] = mpas_mul(0.5f, mpas_add(
            pv_vertex[lidx(k, vertex0, nvertices)],
            pv_vertex[lidx(k, vertex1, nvertices)]));
    }
}

extern "C" __global__ void pv_cell_f32(
    const float* pv_vertex,
    const int* vertices_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_vertex,
    const float* kite_areas,
    const float* area_cell,
    const int nlev,
    const int ncells,
    const int nvertices,
    const int max_edges,
    const int vertex_degree,
    float* pv_cell
) {
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int count = n_edges_on_cell[cell];
    for (int k = 0; k < nlev; ++k) {
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int vertex = vertices_on_cell[cell * max_edges + slot];
            const int kite = kite_slot(
                vertex, cell, cells_on_vertex, vertex_degree
            );
            if (kite >= 0) {
                value = mpas_add(value, mpas_div(mpas_mul(
                    kite_areas[vertex * vertex_degree + kite],
                    pv_vertex[lidx(k, vertex, nvertices)]), area_cell[cell]));
            }
        }
        pv_cell[lidx(k, cell, ncells)] = value;
    }
}

extern "C" __global__ void pv_apvm_f32(
    const float* normal_velocity,
    const float* tangential_velocity,
    const float* pv_vertex,
    const float* pv_cell,
    const int* vertices_on_edge,
    const int* cells_on_edge,
    const float* dc_edge,
    const float* dv_edge,
    const float scale,
    const int nlev,
    const int ncells,
    const int nedges,
    const int nvertices,
    float* pv_edge,
    float* grad_pv_normal,
    float* grad_pv_tangential
) {
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int vertex0 = vertices_on_edge[2 * edge];
    const int vertex1 = vertices_on_edge[2 * edge + 1];
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, edge, nedges);
        const float grad_t = mpas_div(mpas_sub(
            pv_vertex[lidx(k, vertex1, nvertices)],
            pv_vertex[lidx(k, vertex0, nvertices)]), dv_edge[edge]);
        const float grad_n = mpas_div(mpas_sub(
            pv_cell[lidx(k, cell1, ncells)],
            pv_cell[lidx(k, cell0, ncells)]), dc_edge[edge]);
        grad_pv_tangential[tid] = grad_t;
        grad_pv_normal[tid] = grad_n;
        pv_edge[tid] = mpas_sub(pv_edge[tid], mpas_mul(scale, mpas_add(
            mpas_mul(tangential_velocity[tid], grad_t),
            mpas_mul(normal_velocity[tid], grad_n))));
    }
}

extern "C" __global__ void mass_flux_divergence_f32(
    const float* rho_u,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* dv_edge,
    const float* area_cell,
    const int nlev,
    const int ncells,
    const int nedges,
    const int max_edges,
    float* mass_divergence
) {
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int count = n_edges_on_cell[cell];
    for (int k = 0; k < nlev; ++k) {
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            value = mpas_add(value, mpas_mul(mpas_mul(
                cell_sign(cell, edge, cells_on_edge), dv_edge[edge]),
                rho_u[lidx(k, edge, nedges)]));
        }
        mass_divergence[lidx(k, cell, ncells)] = mpas_div(
            value, area_cell[cell]);
    }
}

extern "C" __global__ void density_tendency_f32(
    const float* mass_divergence,
    const float* rho_w,
    const float* rdzw,
    const float* physics_tendency,
    const int has_physics,
    const int nlev,
    const int ncells,
    float* tendency
) {
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, cell, ncells);
        float value = mpas_sub(mpas_sub(0.0f, mass_divergence[tid]),
            mpas_mul(rdzw[k], mpas_sub(
                rho_w[lidx(k + 1, cell, ncells)],
                rho_w[lidx(k, cell, ncells)])));
        if (has_physics) value = mpas_add(value, physics_tendency[tid]);
        tendency[tid] = value;
    }
}

extern "C" __global__ void pressure_gradient_f32(
    const float* pressure_p,
    const float* dpdz,
    const float* cqu,
    const float* zz,
    const float* zxu,
    const int* cells_on_edge,
    const float* dc_edge,
    const int nlev,
    const int ncells,
    const int nedges,
    float* tendency
) {
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, edge, nedges);
        const float normal = mpas_div(mpas_div(mpas_sub(
            pressure_p[lidx(k, cell1, ncells)],
            pressure_p[lidx(k, cell0, ncells)]), dc_edge[edge]),
            mpas_mul(0.5f, mpas_add(
                zz[lidx(k, cell1, ncells)],
                zz[lidx(k, cell0, ncells)])));
        const float terrain = mpas_mul(mpas_mul(0.5f, zxu[tid]), mpas_add(
            dpdz[lidx(k, cell0, ncells)],
            dpdz[lidx(k, cell1, ncells)]));
        tendency[tid] = mpas_mul(mpas_sub(0.0f, cqu[tid]),
            mpas_sub(normal, terrain));
    }
}

extern "C" __global__ void mixing_scaling_f32(
    const int* cells_on_edge,
    const float* mesh_density,
    const int nedges,
    const int scale_with_mesh,
    float* del2,
    float* del4
) {
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    if (!scale_with_mesh) {
        del2[edge] = 1.0f;
        del4[edge] = 1.0f;
        return;
    }
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    const float mean_density = 0.5f * (mesh_density[cell0] + mesh_density[cell1]);
    del2[edge] = 1.0f / powf(mean_density, 0.25f);
    del4[edge] = 1.0f / powf(mean_density, 0.75f);
}

extern "C" __global__ void smagorinsky_f32(
    const float* normal_velocity,
    const float* tangential_velocity,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const float* defc_a,
    const float* defc_b,
    const float strain_scale,
    const float ceiling,
    const int nlev,
    const int ncells,
    const int nedges,
    const int max_edges,
    float* kdiff
) {
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int count = n_edges_on_cell[cell];
    for (int k = 0; k < nlev; ++k) {
        float diagonal = 0.0f;
        float off_diagonal = 0.0f;
        bool ftz_sensitive = false;
        for (int slot = 0; slot < count; ++slot) {
            const int offset = cell * max_edges + slot;
            const int edge = edges_on_cell[offset];
            const float u = normal_velocity[lidx(k, edge, nedges)];
            const float v = tangential_velocity[lidx(k, edge, nedges)];
            const float a = defc_a[offset];
            const float b = defc_b[offset];
            const unsigned int umag = mpas_f32_magnitude_bits(u);
            const unsigned int vmag = mpas_f32_magnitude_bits(v);
            const unsigned int amag = mpas_f32_magnitude_bits(a);
            const unsigned int bmag = mpas_f32_magnitude_bits(b);
            ftz_sensitive = ftz_sensitive
                || (umag != 0u && (umag >> 23) <= 63u)
                || (vmag != 0u && (vmag >> 23) <= 63u)
                || (amag != 0u && (amag >> 23) == 0u)
                || (bmag != 0u && (bmag >> 23) == 0u);
            diagonal += a * u - b * v;
            off_diagonal += b * u + a * v;
        }
        if (ftz_sensitive) {
            diagonal = 0.0f;
            off_diagonal = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int offset = cell * max_edges + slot;
                const int edge = edges_on_cell[offset];
                const float u = normal_velocity[lidx(k, edge, nedges)];
                const float v = tangential_velocity[lidx(k, edge, nedges)];
                const float a = defc_a[offset];
                const float b = defc_b[offset];
                diagonal = mpas_add(diagonal,
                    mpas_sub(mpas_mul(a, u), mpas_mul(b, v)));
                off_diagonal = mpas_add(off_diagonal,
                    mpas_add(mpas_mul(b, u), mpas_mul(a, v)));
            }
            const float strain = mpas_sqrt(mpas_add(
                mpas_mul(diagonal, diagonal),
                mpas_mul(off_diagonal, off_diagonal)));
            kdiff[lidx(k, cell, ncells)] = mpas_min(
                mpas_mul(strain_scale, strain), ceiling);
        } else {
            const float strain = sqrtf(
                diagonal * diagonal + off_diagonal * off_diagonal);
            kdiff[lidx(k, cell, ncells)] = fminf(
                strain_scale * strain, ceiling);
        }
    }
}

extern "C" __global__ void momentum_filter_lap2_f32(
    const float* rho_edge,
    const float* divergence,
    const float* vorticity,
    const float* kdiff,
    const float* del2,
    const int* cells_on_edge,
    const int* vertices_on_edge,
    const float* dc_edge,
    const float* dv_edge,
    const int nlev,
    const int ncells,
    const int nedges,
    const int nvertices,
    float* delsq_u,
    float* tendency
) {
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    const int vertex0 = vertices_on_edge[2 * edge];
    const int vertex1 = vertices_on_edge[2 * edge + 1];
    const float inv_dc = 1.0f / dc_edge[edge];
    const float inv_dv = fminf(1.0f / dv_edge[edge], 4.0f * inv_dc);
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, edge, nedges);
        const float lap = mpas_sub(mpas_mul(mpas_sub(
            divergence[lidx(k, cell1, ncells)],
            divergence[lidx(k, cell0, ncells)]), inv_dc),
            mpas_mul(mpas_sub(vorticity[lidx(k, vertex1, nvertices)],
                vorticity[lidx(k, vertex0, nvertices)]), inv_dv));
        const float kedge = mpas_mul(0.5f, mpas_add(
            kdiff[lidx(k, cell0, ncells)],
            kdiff[lidx(k, cell1, ncells)]));
        delsq_u[tid] = lap;
        tendency[tid] = mpas_mul(mpas_mul(mpas_mul(
            rho_edge[tid], kedge), lap), del2[edge]);
    }
}

extern "C" __global__ void laplacian_divergence_f32(
    const float* delsq_u,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* dv_edge,
    const float* area_cell,
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
            value = mpas_add(value, mpas_mul(mpas_div(mpas_mul(
                cell_sign(cell, edge, cells_on_edge), dv_edge[edge]),
                area_cell[cell]), delsq_u[lidx(k, edge, nedges)]));
        }
        delsq_divergence[lidx(k, cell, ncells)] = value;
    }
}

extern "C" __global__ void laplacian_vorticity_f32(
    const float* delsq_u,
    const int* edges_on_vertex,
    const int* vertices_on_edge,
    const float* dc_edge,
    const float* area_triangle,
    const int nlev,
    const int nedges,
    const int nvertices,
    const int vertex_degree,
    float* delsq_vorticity
) {
    const int vertex = blockDim.x * blockIdx.x + threadIdx.x;
    if (vertex >= nvertices) return;
    for (int k = 0; k < nlev; ++k) {
        float value = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex * vertex_degree + slot];
            if (edge < 0) continue;
            value = mpas_add(value, mpas_mul(mpas_div(mpas_mul(
                vertex_sign(vertex, edge, vertices_on_edge), dc_edge[edge]),
                area_triangle[vertex]), delsq_u[lidx(k, edge, nedges)]));
        }
        delsq_vorticity[lidx(k, vertex, nvertices)] = value;
    }
}

extern "C" __global__ void momentum_filter_lap4_f32(
    const float* rho_edge,
    const float* delsq_divergence,
    const float* delsq_vorticity,
    const float* del4,
    const int* cells_on_edge,
    const int* vertices_on_edge,
    const float* dc_edge,
    const float* dv_edge,
    const float h4,
    const float div_factor,
    const int nlev,
    const int ncells,
    const int nedges,
    const int nvertices,
    float* tendency
) {
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    const int vertex0 = vertices_on_edge[2 * edge];
    const int vertex1 = vertices_on_edge[2 * edge + 1];
    const float inv_dc = 1.0f / dc_edge[edge];
    const float inv_dv = fminf(1.0f / dv_edge[edge], 4.0f * inv_dc);
    const float scale = mpas_mul(del4[edge], h4);
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, edge, nedges);
        const float diffusion = mpas_mul(rho_edge[tid], mpas_sub(
            mpas_mul(mpas_mul(mpas_mul(mpas_sub(
                delsq_divergence[lidx(k, cell1, ncells)],
                delsq_divergence[lidx(k, cell0, ncells)]), scale),
                div_factor), inv_dc),
            mpas_mul(mpas_mul(mpas_sub(
                delsq_vorticity[lidx(k, vertex1, nvertices)],
                delsq_vorticity[lidx(k, vertex0, nvertices)]), scale),
                inv_dv)));
        tendency[tid] = mpas_sub(tendency[tid], diffusion);
    }
}

extern "C" __global__ void theta_filter_lap2_f32(
    const float* theta_m,
    const float* rho_edge,
    const float* kdiff,
    const float* del2,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* dc_edge,
    const float* dv_edge,
    const float* area_cell,
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
            const float factor = mpas_div(mpas_mul(
                cell_sign(cell, edge, cells_on_edge), dv_edge[edge]),
                mpas_mul(area_cell[cell], dc_edge[edge]));
            const float flux = mpas_mul(mpas_mul(factor, mpas_sub(
                theta_m[lidx(k, cell1, ncells)],
                theta_m[lidx(k, cell0, ncells)])),
                rho_edge[lidx(k, edge, nedges)]);
            lap = mpas_add(lap, flux);
            tend = mpas_add(tend, mpas_mul(mpas_mul(mpas_mul(
                flux, 0.5f), mpas_add(
                    kdiff[lidx(k, cell0, ncells)],
                    kdiff[lidx(k, cell1, ncells)])), del2[edge]));
        }
        const int tid = lidx(k, cell, ncells);
        delsq_theta[tid] = lap;
        tendency[tid] = tend;
    }
}

extern "C" __global__ void theta_filter_lap4_f32(
    const float* delsq_theta,
    const float* del4,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* dc_edge,
    const float* dv_edge,
    const float* area_cell,
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
        const int tid = lidx(k, cell, ncells);
        float tend = tendency[tid];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_div(mpas_mul(mpas_mul(mpas_div(
                mpas_mul(del4[edge], h4), area_cell[cell]), dv_edge[edge]),
                cell_sign(cell, edge, cells_on_edge)), dc_edge[edge]);
            tend = mpas_sub(tend, mpas_mul(factor, mpas_sub(
                delsq_theta[lidx(k, cell1, ncells)],
                delsq_theta[lidx(k, cell0, ncells)])));
        }
        tendency[tid] = tend;
    }
}

extern "C" __global__ void w_filter_lap2_f32(
    const float* vertical_velocity,
    const float* rho_edge,
    const float* kdiff,
    const float* del2,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* dc_edge,
    const float* dv_edge,
    const float* area_cell,
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
        const int tid = lidx(k, cell, ncells);
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
            const float factor = mpas_div(mpas_mul(mpas_mul(
                mpas_div(0.5f, area_cell[cell]),
                cell_sign(cell, edge, cells_on_edge)), dv_edge[edge]),
                dc_edge[edge]);
            const float flux = mpas_mul(mpas_mul(factor, mpas_add(
                rho_edge[lidx(k, edge, nedges)],
                rho_edge[lidx(k - 1, edge, nedges)])), mpas_sub(
                    vertical_velocity[lidx(k, cell1, ncells)],
                    vertical_velocity[lidx(k, cell0, ncells)]));
            lap = mpas_add(lap, flux);
            const float kedge = mpas_mul(0.25f, mpas_add(mpas_add(
                kdiff[lidx(k, cell0, ncells)],
                kdiff[lidx(k, cell1, ncells)]), mpas_add(
                    kdiff[lidx(k - 1, cell0, ncells)],
                    kdiff[lidx(k - 1, cell1, ncells)])));
            tend = mpas_add(tend, mpas_mul(mpas_mul(
                flux, del2[edge]), kedge));
        }
        delsq_w[tid] = lap;
        tendency[tid] = tend;
    }
}

extern "C" __global__ void w_filter_lap4_f32(
    const float* delsq_w,
    const float* del4,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* dc_edge,
    const float* dv_edge,
    const float* area_cell,
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
        const int tid = lidx(k, cell, ncells);
        float tend = tendency[tid];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_div(mpas_mul(mpas_mul(mpas_div(
                mpas_mul(del4[edge], h4), area_cell[cell]), dv_edge[edge]),
                cell_sign(cell, edge, cells_on_edge)), dc_edge[edge]);
            tend = mpas_sub(tend, mpas_mul(factor, mpas_sub(
                delsq_w[lidx(k, cell1, ncells)],
                delsq_w[lidx(k, cell0, ncells)])));
        }
        tendency[tid] = tend;
    }
}

extern "C" __global__ void divergence_damping_f32(
    const float* theta_m,
    const float* rtheta_pp,
    const float* rtheta_pp_old,
    const int* cells_on_edge,
    const float* spec_zone_mask_edge,
    const float coefficient,
    const int n_cells_solve,
    const int nlev,
    const int ncells,
    const int nedges,
    float* rho_u_perturbation
) {
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    if (cell0 >= n_cells_solve && cell1 >= n_cells_solve) return;
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, edge, nedges);
        const float delta0 = mpas_sub(
            rtheta_pp[lidx(k, cell0, ncells)],
            rtheta_pp_old[lidx(k, cell0, ncells)]);
        const float delta1 = mpas_sub(
            rtheta_pp[lidx(k, cell1, ncells)],
            rtheta_pp_old[lidx(k, cell1, ncells)]);
        const float denominator = mpas_add(
            theta_m[lidx(k, cell0, ncells)],
            theta_m[lidx(k, cell1, ncells)]);
        rho_u_perturbation[tid] = mpas_add(rho_u_perturbation[tid],
            mpas_div(mpas_mul(mpas_mul(coefficient,
                mpas_sub(delta0, delta1)), mpas_sub(
                    1.0f, spec_zone_mask_edge[edge])), denominator));
    }
}
"""



# mixing_scaling_f32 evaluates meshScalingDel2/Del4 = 1/((d1+d2)/2)**0.25 and **0.75 in RKIND
# (mpas_atm_core.F atm_compute_mesh_scaling); gfortran calls GNU libm powf there. CUDA's powf differs from
# it in the last bit for some densities, and on a variable-resolution mesh that moves every mixed edge
# (storm box 2024-05-21 15Z against native 44c0849 strict: 986,078 history words differ after one step; 0
# with config_h_ScaleWithMesh false on both sides). Only that kernel calls powf on mean_density; the other
# kernels compile from _CUDA_SOURCE unchanged.
_SCALING_GNU_SOURCE = CUDA_NATIVE_POWF + _CUDA_SOURCE.replace(
    "powf(mean_density,", "mpas_native_powf_positive(mean_density,")

class CudaHorizontal:
    """Cached float32 RawKernel launcher over one resident :class:`DeviceMesh`."""

    def __init__(
        self,
        mesh: DeviceMesh,
        nlev: int,
        *,
        kernel_cache: KernelCache | None = None,
    ) -> None:
        capability = require_cuda(min_compute=(12, 0))
        import cupy as cp

        if not isinstance(mesh, DeviceMesh):
            raise TypeError("mesh must be a cuda_backend.DeviceMesh")
        if nlev < 1:
            raise ValueError("nlev must be positive")
        if np.dtype(mesh.dtype) != np.dtype(np.float32):
            raise TypeError("CudaHorizontal currently admits DeviceMesh dtype=float32")
        if np.dtype(mesh.index_dtype) != np.dtype(np.int32):
            raise TypeError("CudaHorizontal currently admits DeviceMesh index_dtype=int32")
        self.cp = cp
        self.mesh = mesh
        self.nlev = int(nlev)
        self.ncells = int(mesh.n_cells)
        self.nedges = int(mesh.n_edges)
        self.nvertices = int(mesh.n_vertices)
        self.max_edges = int(mesh.max_edges)
        self.max_edges2 = int(mesh.max_edges2)
        self.vertex_degree = int(mesh.vertex_degree)
        self.kernel_cache = (
            KernelCache(capability=capability)
            if kernel_cache is None
            else kernel_cache
        )
        self._kernels: dict[str, Any] = {}

    def _kernel(self, name: str) -> Any:
        kernel = self._kernels.get(name)
        if kernel is None:
            if name == "mixing_scaling_f32":
                kernel = self.kernel_cache.raw_kernel(
                    name,
                    _SCALING_GNU_SOURCE,
                    module_key="hexcore.cuda_horizontal_gnu_scaling",
                )
            else:
                kernel = self.kernel_cache.raw_kernel(
                    name,
                    _CUDA_SOURCE,
                    module_key="hexcore.cuda_horizontal",
                )
            self._kernels[name] = kernel
        return kernel

    def _launch(self, name: str, total: int, args: tuple[Any, ...]) -> None:
        if total < 1:
            return
        blocks = (int(total) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        self._kernel(name)((blocks,), (_BLOCK_SIZE,), args)

    def _field(
        self,
        name: str,
        value: Any,
        shape: tuple[int, ...],
    ) -> Any:
        cp = self.cp
        if not isinstance(value, cp.ndarray):
            raise TypeError(f"{name} must already be a resident cupy.ndarray")
        if value.dtype != cp.float32:
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous level-major storage")
        return value

    def _cell_field(self, name: str, value: Any) -> Any:
        return self._field(name, value, (self.nlev, self.ncells))

    def _edge_field(self, name: str, value: Any) -> Any:
        return self._field(name, value, (self.nlev, self.nedges))

    def _interface_field(self, name: str, value: Any) -> Any:
        return self._field(name, value, (self.nlev + 1, self.ncells))

    def _edge_state(
        self,
        rho: Any,
        rho_u: Any,
        normal_velocity: Any | None,
    ) -> tuple[Any, Any, Any]:
        cp = self.cp
        rho = self._cell_field("rho", rho)
        rho_u = self._edge_field("rho_u", rho_u)
        rho_edge = cp.empty_like(rho_u)
        ke_edge = cp.empty_like(rho_u)
        common = (
            self.mesh.dc_edge,
            self.mesh.dv_edge,
            self.mesh.cells_on_edge,
            np.int32(self.nlev),
            np.int32(self.ncells),
            np.int32(self.nedges),
        )
        if normal_velocity is None:
            normal = cp.empty_like(rho_u)
            self._launch(
                "recover_edge_f32",
                self.nedges,
                (rho, rho_u, *common, normal, rho_edge, ke_edge),
            )
        else:
            normal = self._edge_field("normal_velocity", normal_velocity)
            self._launch(
                "edge_fields_from_saved_u_f32",
                self.nedges,
                (rho, normal, *common, rho_edge, ke_edge),
            )
        return normal, rho_edge, ke_edge

    def recover_edge_fields(self, rho: Any, rho_u: Any) -> CudaRecoveredEdges:
        """Recover normal velocity and edge density without leaving the GPU."""

        normal, rho_edge, _ = self._edge_state(rho, rho_u, None)
        return CudaRecoveredEdges(normal_velocity=normal, rho_edge=rho_edge)

    def mass_flux_divergence(self, rho_u: Any) -> Any:
        rho_u = self._edge_field("rho_u", rho_u)
        out = self.cp.empty((self.nlev, self.ncells), dtype=self.cp.float32)
        self._launch(
            "mass_flux_divergence_f32",
            self.ncells,
            (
                rho_u,
                self.mesh.edges_on_cell,
                self.mesh.n_edges_on_cell,
                self.mesh.cells_on_edge,
                self.mesh.dv_edge,
                self.mesh.area_cell,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.max_edges),
                out,
            ),
        )
        return out

    def density_tendency(
        self,
        rho_u: Any,
        rho_w: Any,
        rdzw: Any,
        physics_tendency: Any | None = None,
    ) -> Any:
        mass_divergence = self.mass_flux_divergence(rho_u)
        rho_w = self._interface_field("rho_w", rho_w)
        rdzw = self._field("rdzw", rdzw, (self.nlev,))
        if physics_tendency is None:
            physics = mass_divergence
            has_physics = np.int32(0)
        else:
            physics = self._cell_field("physics_tendency", physics_tendency)
            has_physics = np.int32(1)
        out = self.cp.empty_like(mass_divergence)
        self._launch(
            "density_tendency_f32",
            self.ncells,
            (
                mass_divergence,
                rho_w,
                rdzw,
                physics,
                has_physics,
                np.int32(self.nlev),
                np.int32(self.ncells),
                out,
            ),
        )
        return out

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
        """Compute the full frozen C-grid diagnostic bundle on device."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(apvm_upwinding) or apvm_upwinding < 0.0:
            raise ValueError("apvm_upwinding must be finite and non-negative")
        cp = self.cp
        normal, rho_edge, ke_edge = self._edge_state(
            rho, rho_u, normal_velocity
        )
        reconstruct_v = rk_step is None or rk_step == 3
        if reconstruct_v:
            tangential = cp.empty_like(normal)
            self._launch(
                "tangential_velocity_f32",
                self.nlev * self.nedges,
                (
                    normal,
                    self.mesh.edges_on_edge,
                    self.mesh.n_edges_on_edge,
                    self.mesh.weights_on_edge,
                    np.int32(self.nlev),
                    np.int32(self.nedges),
                    np.int32(self.max_edges2),
                    tangential,
                ),
            )
        else:
            if cached_tangential_velocity is None:
                raise ValueError("RK1/RK2 diagnostics require cached tangential velocity")
            # Bound, not copied: every consumer takes the tangential velocity as
            # a const float*, so the RK1/RK2 reuse of the RK3 field needs a
            # reference, not a private nlev x nEdges image per stage.
            tangential = self._edge_field(
                "cached_tangential_velocity", cached_tangential_velocity
            )

        vorticity = cp.empty((self.nlev, self.nvertices), dtype=cp.float32)
        ke_vertex = cp.empty_like(vorticity)
        pv_vertex = cp.empty_like(vorticity)
        self._launch(
            "vertex_diagnostics_f32",
            self.nvertices,
            (
                normal,
                ke_edge,
                self.mesh.edges_on_vertex,
                self.mesh.vertices_on_edge,
                self.mesh.dc_edge,
                self.mesh.area_triangle,
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
        self._launch(
            "cell_diagnostics_f32",
            self.ncells,
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
                self.mesh.area_cell,
                self.mesh.kite_areas_on_vertex,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.nvertices),
                np.int32(self.max_edges),
                np.int32(self.vertex_degree),
                divergence,
                kinetic_energy,
            ),
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
            self._launch(
                "pv_cell_f32",
                self.ncells,
                (
                    pv_vertex,
                    self.mesh.vertices_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_vertex,
                    self.mesh.kite_areas_on_vertex,
                    self.mesh.area_cell,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.nvertices),
                    np.int32(self.max_edges),
                    np.int32(self.vertex_degree),
                    pv_cell,
                ),
            )
            self._launch(
                "pv_apvm_f32",
                self.nedges,
                (
                    normal,
                    tangential,
                    pv_vertex,
                    pv_cell,
                    self.mesh.vertices_on_edge,
                    self.mesh.cells_on_edge,
                    self.mesh.dc_edge,
                    self.mesh.dv_edge,
                    np.float32(apvm_upwinding * dt),
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
        self._launch(
            "pressure_gradient_f32",
            self.nedges,
            (
                pressure,
                dpdz,
                cqu,
                zz,
                zxu,
                self.mesh.cells_on_edge,
                self.mesh.dc_edge,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                out,
            ),
        )
        return out

    def compute_dry_mixing_tendencies(
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
    ) -> CudaDryMixingTendencies:
        """Compute the complete frozen-JW 2-D Smagorinsky branch on device.

        ``config`` is the existing :class:`woof.hex.mixing.MixingConfig`
        authority.  The CUDA lane deliberately accepts the same knobs and
        refuses every non-JW branch before launching a kernel.
        """

        from .mixing import MixingConfig

        cfg = MixingConfig() if config is None else config
        if not isinstance(cfg, MixingConfig):
            raise TypeError("config must be a mixing.MixingConfig")
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
        length32 = np.float32(length)
        cs = np.float32(cfg.config_smagorinsky_coef)
        strain_scale = np.float32((cs * length32) * (cs * length32))
        ceiling = np.float32(
            np.float32(0.01) * length32 * length32 / np.float32(timestep)
        )
        visc4 = np.float32(cfg.config_visc4_2dsmag)
        h4 = np.float32(visc4 * length32 * length32 * length32)

        del2 = cp.empty((self.nedges,), dtype=cp.float32)
        del4 = cp.empty_like(del2)
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

        kdiff = cp.empty((self.nlev, self.ncells), dtype=cp.float32)
        self._launch(
            "smagorinsky_f32",
            self.ncells,
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

        delsq_u = cp.empty_like(normal)
        tend_u = cp.empty_like(normal)
        self._launch(
            "momentum_filter_lap2_f32",
            self.nedges,
            (
                edge_rho,
                div,
                vort,
                kdiff,
                del2,
                self.mesh.cells_on_edge,
                self.mesh.vertices_on_edge,
                self.mesh.dc_edge,
                self.mesh.dv_edge,
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
            self._launch(
                "laplacian_divergence_f32",
                self.nlev * self.ncells,
                (
                    delsq_u,
                    self.mesh.edges_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_edge,
                    self.mesh.dv_edge,
                    self.mesh.area_cell,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.nedges),
                    np.int32(self.max_edges),
                    delsq_div,
                ),
            )
            self._launch(
                "laplacian_vorticity_f32",
                self.nvertices,
                (
                    delsq_u,
                    self.mesh.edges_on_vertex,
                    self.mesh.vertices_on_edge,
                    self.mesh.dc_edge,
                    self.mesh.area_triangle,
                    np.int32(self.nlev),
                    np.int32(self.nedges),
                    np.int32(self.nvertices),
                    np.int32(self.vertex_degree),
                    delsq_vort,
                ),
            )
            self._launch(
                "momentum_filter_lap4_f32",
                self.nedges,
                (
                    edge_rho,
                    delsq_div,
                    delsq_vort,
                    del4,
                    self.mesh.cells_on_edge,
                    self.mesh.vertices_on_edge,
                    self.mesh.dc_edge,
                    self.mesh.dv_edge,
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
        self._launch(
            "theta_filter_lap2_f32",
            self.nlev * self.ncells,
            (
                theta,
                edge_rho,
                kdiff,
                del2,
                self.mesh.edges_on_cell,
                self.mesh.n_edges_on_cell,
                self.mesh.cells_on_edge,
                self.mesh.dc_edge,
                self.mesh.dv_edge,
                self.mesh.area_cell,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.max_edges),
                delsq_theta,
                tend_theta,
            ),
        )
        if h4 > np.float32(0.0):
            self._launch(
                "theta_filter_lap4_f32",
                self.nlev * self.ncells,
                (
                    delsq_theta,
                    del4,
                    self.mesh.edges_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_edge,
                    self.mesh.dc_edge,
                    self.mesh.dv_edge,
                    self.mesh.area_cell,
                    h4,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.max_edges),
                    tend_theta,
                ),
            )

        delsq_w = cp.empty((self.nlev, self.ncells), dtype=cp.float32)
        tend_w = cp.zeros((self.nlev + 1, self.ncells), dtype=cp.float32)
        self._launch(
            "w_filter_lap2_f32",
            self.nlev * self.ncells,
            (
                vertical,
                edge_rho,
                kdiff,
                del2,
                self.mesh.edges_on_cell,
                self.mesh.n_edges_on_cell,
                self.mesh.cells_on_edge,
                self.mesh.dc_edge,
                self.mesh.dv_edge,
                self.mesh.area_cell,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.max_edges),
                delsq_w,
                tend_w,
            ),
        )
        if h4 > np.float32(0.0):
            self._launch(
                "w_filter_lap4_f32",
                (self.nlev - 1) * self.ncells,
                (
                    delsq_w,
                    del4,
                    self.mesh.edges_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_edge,
                    self.mesh.dc_edge,
                    self.mesh.dv_edge,
                    self.mesh.area_cell,
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
            h_theta_eddy_visc4=h4,
            tend_u_euler=tend_u,
            tend_w_euler=tend_w,
            tend_theta_euler=tend_theta,
            delsq_u=delsq_u,
            delsq_divergence=delsq_div,
            delsq_vorticity=delsq_vort,
            delsq_w=delsq_w,
            delsq_theta=delsq_theta,
        )

    def capture_rtheta_pp_old(self, rtheta_pp: Any, *, small_step: int) -> Any:
        """Capture the resident pre-acoustic damping reference."""

        current = self._cell_field("rtheta_pp", rtheta_pp)
        step = int(small_step)
        if step < 1:
            raise ValueError("small_step is one-based and must be positive")
        if step == 1:
            return self.cp.zeros_like(current)
        return current.copy()

    def divergence_damping_3d(
        self,
        rho_u_perturbation: Any,
        theta_m: Any,
        rtheta_pp: Any,
        rtheta_pp_old: Any,
        *,
        dts: float,
        config_smdiv: float = 0.1,
        config_len_disp: float = 0.0,
        spec_zone_mask_edge: Any | None = None,
        n_cells_solve: int | None = None,
        in_place: bool = False,
        native_stored_inverse: bool = False,
        _partition_edges: Any | None = None,
    ) -> Any:
        """Apply the frozen post-acoustic 3-D divergence damping update."""

        timestep = float(dts)
        smdiv = float(config_smdiv)
        configured_length = float(config_len_disp)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("dts must be finite and positive")
        if not np.isfinite(smdiv) or smdiv < 0.0:
            raise ValueError("config_smdiv must be finite and non-negative")
        if not np.isfinite(configured_length) or configured_length < 0.0:
            raise ValueError("config_len_disp must be finite and non-negative")
        length = (
            float(self.mesh.nominal_min_dc)
            if configured_length == 0.0
            else configured_length
        )
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(
                "config_len_disp or DeviceMesh.nominal_min_dc must be positive"
            )
        solve_count = self.ncells if n_cells_solve is None else int(n_cells_solve)
        if solve_count < 0 or solve_count > self.ncells:
            raise ValueError("n_cells_solve must lie in [0, nCells]")

        ru = self._edge_field("rho_u_perturbation", rho_u_perturbation)
        theta = self._cell_field("theta_m", theta_m)
        current = self._cell_field("rtheta_pp", rtheta_pp)
        previous = self._cell_field("rtheta_pp_old", rtheta_pp_old)
        if spec_zone_mask_edge is None:
            mask = self.mesh.spec_zone_mask_edge
        else:
            mask = self._field(
                "spec_zone_mask_edge",
                spec_zone_mask_edge,
                (self.nedges,),
            )
        if not isinstance(in_place, (bool, np.bool_)):
            raise TypeError("in_place must be boolean")
        out = ru if bool(in_place) else ru.copy()
        base_coefficient = np.float32(2.0) * np.float32(smdiv) * np.float32(length)
        coefficient = np.float32(
            base_coefficient * (np.float32(1.0) / np.float32(timestep))
            if native_stored_inverse else base_coefficient / np.float32(timestep)
        )
        arguments = (
                theta,
                current,
                previous,
                self.mesh.cells_on_edge,
                mask,
                coefficient,
                np.int32(solve_count),
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                out,
        )
        if _partition_edges is None:
            self._launch("divergence_damping_f32", self.nedges, arguments)
        else:
            from .partition_overlap_v841 import launch_selected
            launch_selected(self, self.nedges, arguments, _partition_edges)
        return out


__all__ = [
    "CUDA_HORIZONTAL_EVIDENCE",
    "CudaDryMixingTendencies",
    "CudaHorizontal",
    "CudaRecoveredEdges",
    "CudaSolveDiagnostics",
]
