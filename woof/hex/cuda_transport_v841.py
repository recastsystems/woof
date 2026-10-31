"""MPAS-A v8.4.1 CUDA atmosphere scalar transport.

The v8.2.3 translation unit remains untouched.  This additive module retains
the released stored-inverse and reduction associations needed by the native
split-three outer scalar RK.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import cuda_transport as _v823
from .cuda_transport import CudaAdvectionCoefficients, CudaScalarTransportResult
from .cuda_v841 import CudaV841Context
from .errors import ConfigurationRefusal


_CUDA_SOURCE = _v823._CUDA_SOURCE + r"""
extern "C" __global__ void fct_pool_density_v841(
    const int nlev, const int ncells, const int cell, float *density)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < nlev) density[k * ncells + cell] = 1.0f;
}
extern "C" __global__ void fct_boundary_residual_v841(
    const int ntracers, const int nlev, const int nedges,
    const int *bdy_mask_edge, float *residual)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= ntracers * nlev * nedges) return;
    const int mask = bdy_mask_edge[index % nedges];
    if (mask == 4 || mask == 5) residual[index] = 0.0f;
}
extern "C" __global__ void fct_copy_back_v841(
    const int ntracers, const int nlev, const int ncells,
    const int regional, const int *bdy_mask_cell,
    const float *stage, float *output)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= ntracers * nlev * ncells) return;
    if (regional && bdy_mask_cell[index % ncells] > 2)
        output[index] = stage[index];
    else output[index] = fmaxf(0.0f, output[index]);
}
/* atm_advance_scalars_mono_work (mpas_atm_time_integration.F, v8.4.2
   lines 5342-5352): every level first loses its UPPER interface's upwind
   flux (k = 1..nVertLevels-1), and only then gains its LOWER interface's
   (k = 2..nVertLevels).  The v8.2.3 kernel walks interfaces upward and so
   adds the lower flux before subtracting the upper one; (m+a)-b and (m-b)+a
   round differently.  Invisible while every transported scalar was zero (the
   dry identity cases); measured on the first moist step: 43,513 qv words. */
extern "C" __global__ void fct_vertical_low_order_v841(
    const int ntracers, const int nlev, const int ncells,
    const float dt, const float *source_old, const float *rho_old,
    const float *vertical_velocity, const float *rdzw,
    const float *high_vertical,
    float *mass, float *vertical_residual, float *scale_in, float *scale_out)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = ntracers * ncells;
    if (index >= total) return;
    const int cell = index % ncells;
    const int tracer = index / ncells;
    for (int k = 0; k < nlev; ++k) {
        const int q = Q3(tracer, k, cell, nlev, ncells);
        mass[q] = mpas_mul(source_old[q], rho_old[C2(k, cell, ncells)]);
    }
    for (int interface = 0; interface <= nlev; ++interface) {
        const int qi = QI3(tracer, interface, cell, nlev, ncells);
        vertical_residual[qi] = mpas_mul(dt, high_vertical[qi]);
    }
    /* Pass 1: level k loses interface k+1's upwind flux. */
    for (int k = 0; k + 1 < nlev; ++k) {
        const int interface = k + 1;
        const float velocity = vertical_velocity[C2(interface, cell, ncells)];
        const float flux = mpas_mul(dt, mpas_add(
            mpas_mul(plus(velocity), source_old[Q3(
                tracer, interface - 1, cell, nlev, ncells)]),
            mpas_mul(minus(velocity), source_old[Q3(
                tracer, interface, cell, nlev, ncells)])));
        const int q = Q3(tracer, k, cell, nlev, ncells);
        mass[q] = mpas_sub(mass[q], mpas_mul(flux, rdzw[k]));
    }
    /* Pass 2: level k gains interface k's flux; the residual is formed. */
    for (int k = 1; k < nlev; ++k) {
        const float velocity = vertical_velocity[C2(k, cell, ncells)];
        const float flux = mpas_mul(dt, mpas_add(
            mpas_mul(plus(velocity), source_old[Q3(
                tracer, k - 1, cell, nlev, ncells)]),
            mpas_mul(minus(velocity), source_old[Q3(
                tracer, k, cell, nlev, ncells)])));
        const int q = Q3(tracer, k, cell, nlev, ncells);
        mass[q] = mpas_add(mass[q], mpas_mul(flux, rdzw[k]));
        const int residual_q = QI3(tracer, k, cell, nlev, ncells);
        vertical_residual[residual_q] = mpas_sub(
            vertical_residual[residual_q], flux);
    }
    for (int k = 0; k < nlev; ++k) {
        const float upper = vertical_residual[QI3(tracer, k + 1, cell, nlev, ncells)];
        const float lower = vertical_residual[QI3(tracer, k, cell, nlev, ncells)];
        const int q = Q3(tracer, k, cell, nlev, ncells);
        scale_in[q] = mpas_mul(-rdzw[k],
            mpas_sub(minus(upper), plus(lower)));
        scale_out[q] = mpas_mul(-rdzw[k],
            mpas_sub(plus(upper), minus(lower)));
    }
}
extern "C" __global__ void transport_interpolate_target_v841(
    const int nvalues, const float weight,
    const float *rho_old, const float *rho_new, float *target)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nvalues) return;
    target[index] = mpas_add(
        mpas_mul(mpas_sub(1.0f, weight), rho_old[index]),
        mpas_mul(weight, rho_new[index]));
}

extern "C" __global__ void validate_density_v841(
    const int nvalues, const float *density, int *invalid)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nvalues) return;
    const float value = density[index];
    if (!(isfinite(value) && value > 0.0f)) atomicExch(invalid, 1);
}

extern "C" __global__ void validate_transport_indices_v841(
    const int ncells, const int nedges, const int max_edges,
    const int width, const int *n_edges_on_cell, const int *edges_on_cell,
    const int *cells_on_cell, const int *cells_on_edge,
    const int *n_adv, const int *adv_cells, int *invalid)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < ncells) {
        const int count = n_edges_on_cell[index];
        if (count < 0 || count > max_edges) atomicExch(invalid, 1);
        const int safe_count = count < 0 ? 0 : (count > max_edges ? max_edges : count);
        for (int slot = 0; slot < safe_count; ++slot) {
            const int edge = edges_on_cell[CES(index, slot, max_edges)];
            const int cell = cells_on_cell[CES(index, slot, max_edges)];
            if (edge < 0 || edge >= nedges || cell < 0 || cell >= ncells)
                atomicExch(invalid, 1);
        }
    }
    if (index < nedges) {
        const int c0 = cells_on_edge[2 * index];
        const int c1 = cells_on_edge[2 * index + 1];
        if (c0 < 0 || c0 >= ncells || c1 < 0 || c1 >= ncells)
            atomicExch(invalid, 1);
        const int count = n_adv[index];
        if (count < 0 || count > width) atomicExch(invalid, 1);
        const int safe_count = count < 0 ? 0 : (count > width ? width : count);
        for (int slot = 0; slot < safe_count; ++slot) {
            const int cell = adv_cells[ADV(index, slot, width)];
            if (cell < 0 || cell >= ncells) atomicExch(invalid, 1);
        }
    }
}

extern "C" __global__ void transport_standard_finish_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges, const float dt,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell,
    const float *velocity, const float *edge_values,
    const float *vertical_flux, const float *rdzw,
    const float *old, const float *rho_old, const float *target_density,
    const float *source, float *output)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            float horizontal = 0.0f;
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                horizontal = mpas_add(horizontal, mpas_mul(mpas_mul(
                    sign, velocity[E2(k, edge, nedges)]),
                    edge_values[QE3(tracer, k, edge, nlev, nedges)]));
            }
            const float tendency = mpas_add(
                mpas_mul(horizontal, inv_area_cell[cell]), source[index]);
            const float vertical = mpas_mul(rdzw[k], mpas_sub(
                vertical_flux[QI3(tracer, k + 1, cell, nlev, ncells)],
                vertical_flux[QI3(tracer, k, cell, nlev, ncells)]));
            const float numerator = mpas_add(
                mpas_mul(old[index], rho_old[C2(k, cell, ncells)]),
                mpas_mul(dt, mpas_sub(tendency, vertical)));
            const float rho_new_inv = mpas_div(
                1.0f, target_density[C2(k, cell, ncells)]);
            output[index] = mpas_mul(numerator, rho_new_inv);
        }
    }
}

extern "C" __global__ void transport_target_density_v841(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int advance_density, const float dt,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *dv_edge,
    const float *inv_area_cell, const float *velocity,
    const float *vertical_velocity, const float *rdzw,
    const float *rho_old, const float *rho_new, float *target_density)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int index = C2(k, cell, ncells);
        if (!advance_density) {
            target_density[index] = rho_new[index];
        } else {
            float horizontal = 0.0f;
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                float contribution = mpas_mul(
                    sign, velocity[E2(k, edge, nedges)]);
                contribution = mpas_mul(contribution, dv_edge[edge]);
                contribution = mpas_mul(contribution, inv_area_cell[cell]);
                horizontal = mpas_add(horizontal, contribution);
            }
            target_density[index] = mpas_add(rho_old[index], mpas_mul(dt,
                mpas_sub(horizontal, mpas_mul(rdzw[k], mpas_sub(
                    vertical_velocity[C2(k + 1, cell, ncells)],
                    vertical_velocity[index])))));
        }
    }
}

extern "C" __global__ void transport_edge_values_mono_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int width, const float coefficient,
    const float *stage, const float *velocity,
    const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells, float *edge_values)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int count = n_adv[edge];
    if (count == 10) {
        for (int k = 0; k < nlev; ++k) {
            const float vel = velocity[E2(k, edge, nedges)];
            const float sign = mpas_copysign(1.0f, vel);
            for (int tracer = 0; tracer < ntracers; ++tracer) {
                const int first_cell = adv_cells[ADV(edge, 0, width)];
                float weight = mpas_add(adv[ADV(edge, 0, width)],
                    mpas_mul(mpas_mul(coefficient, sign),
                        adv3[ADV(edge, 0, width)]));
                float value = mpas_mul(weight,
                    stage[Q3(tracer, k, first_cell, nlev, ncells)]);
                for (int slot = 1; slot < 10; ++slot) {
                    const int cell = adv_cells[ADV(edge, slot, width)];
                    weight = mpas_add(adv[ADV(edge, slot, width)],
                        mpas_mul(mpas_mul(coefficient, sign),
                            adv3[ADV(edge, slot, width)]));
                    value = mpas_add(value, mpas_mul(weight,
                        stage[Q3(tracer, k, cell, nlev, ncells)]));
                }
                edge_values[QE3(tracer, k, edge, nlev, nedges)] =
                    mpas_mul(vel, value);
            }
        }
    } else {
        for (int tracer = 0; tracer < ntracers; ++tracer) {
            for (int k = 0; k < nlev; ++k) {
                edge_values[QE3(tracer, k, edge, nlev, nedges)] = 0.0f;
            }
        }
        for (int slot = 0; slot < count; ++slot) {
            const int cell = adv_cells[ADV(edge, slot, width)];
            for (int k = 0; k < nlev; ++k) {
                const float vel = velocity[E2(k, edge, nedges)];
                const float sign = mpas_copysign(1.0f, vel);
                float scalar_weight = mpas_add(adv[ADV(edge, slot, width)],
                    mpas_mul(mpas_mul(coefficient, sign),
                        adv3[ADV(edge, slot, width)]));
                scalar_weight = mpas_mul(vel, scalar_weight);
                for (int tracer = 0; tracer < ntracers; ++tracer) {
                    const int index = QE3(tracer, k, edge, nlev, nedges);
                    edge_values[index] = mpas_add(edge_values[index],
                        mpas_mul(scalar_weight,
                            stage[Q3(tracer, k, cell, nlev, ncells)]));
                }
            }
        }
    }
}

extern "C" __global__ void fct_edge_residual_v841(
    const int ntracers, const int nlev, const int ncells, const int nedges,
    const float dt, const float *source_old, const float *velocity,
    const float *dv_edge, const int *cells_on_edge,
    const float *high_flux, float *upwind_flux, float *horizontal_residual)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = QE3(tracer, k, edge, nlev, nedges);
            const float vel = velocity[E2(k, edge, nedges)];
            const float upwind = mpas_mul(mpas_mul(dv_edge[edge], dt),
                mpas_add(mpas_mul(plus(vel), source_old[Q3(
                    tracer, k, c0, nlev, ncells)]),
                    mpas_mul(minus(vel), source_old[Q3(
                    tracer, k, c1, nlev, ncells)])));
            upwind_flux[index] = upwind;
            horizontal_residual[index] = mpas_sub(
                mpas_mul(dt, high_flux[index]), upwind);
        }
    }
}

extern "C" __global__ void fct_horizontal_low_order_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell,
    const float *upwind_flux, const float *horizontal_residual,
    float *mass, float *scale_in, float *scale_out)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const float inverse_area = inv_area_cell[cell];
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            float cell_mass = mass[index];
            float incoming = scale_in[index];
            float outgoing = scale_out[index];
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                const int qe = QE3(tracer, k, edge, nlev, nedges);
                cell_mass = mpas_add(cell_mass, mpas_mul(
                    mpas_mul(sign, upwind_flux[qe]), inverse_area));
                const float signed_residual = mpas_mul(
                    -sign, horizontal_residual[qe]);
                outgoing = mpas_sub(outgoing,
                    mpas_mul(plus(signed_residual), inverse_area));
                incoming = mpas_sub(incoming,
                    mpas_mul(minus(signed_residual), inverse_area));
            }
            mass[index] = cell_mass;
            scale_in[index] = incoming;
            scale_out[index] = outgoing;
        }
    }
}

extern "C" __global__ void fct_finish_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzw,
    const float *horizontal_residual, const float *vertical_residual,
    const float *target_density, const float *mass, float *output)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const float inverse_area = inv_area_cell[cell];
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            float value = mass[index];
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                value = mpas_add(value, mpas_mul(mpas_mul(sign,
                    horizontal_residual[QE3(
                        tracer, k, edge, nlev, nedges)]), inverse_area));
            }
            value = mpas_sub(value, mpas_mul(rdzw[k], mpas_sub(
                vertical_residual[QI3(
                    tracer, k + 1, cell, nlev, ncells)],
                vertical_residual[QI3(
                    tracer, k, cell, nlev, ncells)])));
            output[index] = mpas_max(0.0f,
                mpas_div(value, target_density[C2(k, cell, ncells)]));
        }
    }
}

#ifndef HEX_HOIST_ADV
#define HEX_HOIST_ADV 10
#endif
#ifndef HEX_HOIST_CELL
#define HEX_HOIST_CELL 10
#endif
/* Thread per (tracer, edge) form of transport_edge_values: the edge's
   advection stencil is read once into registers and the thread walks the
   levels.  Each (tracer, level) runs the element body unchanged, including
   the subnormal-sensitivity test and its exact re-evaluation, so every edge
   value carries the same bits. */
extern "C" __global__ void transport_edge_values_hoist_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int width, const float coefficient,
    const float *stage, const float *velocity,
    const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells,
    float *edge_values, const int lpt)
{
    const int hex_chunks = (nlev + lpt - 1) / lpt;
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= ntracers * nedges * hex_chunks) return;
    const int edge = element % nedges;
    const int tracer = (element / nedges) % ntracers;
    const int hex_k0 = (element / nedges / ntracers) * lpt;
    const int hex_k1 = min(nlev, hex_k0 + lpt);
    const int count = n_adv[edge];
    int cl[HEX_HOIST_ADV];
    float b[HEX_HOIST_ADV];
    float t[HEX_HOIST_ADV];
    const bool hoisted = count <= HEX_HOIST_ADV;
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_ADV; ++slot) {
        const bool live = hoisted && slot < count;
        cl[slot] = live ? adv_cells[ADV(edge, slot, width)] : 0;
        b[slot] = live ? adv[ADV(edge, slot, width)] : 0.0f;
        t[slot] = live ? adv3[ADV(edge, slot, width)] : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        float value = 0.0f;
        const float velocity_sign = mpas_copysign(
            1.0f, velocity[E2(k, edge, nedges)]);
        bool ftz_sensitive = false;
        if (hoisted) {
#pragma unroll
            for (int slot = 0; slot < HEX_HOIST_ADV; ++slot) {
                if (slot < count) {
                    const float base_weight = b[slot];
                    const float third_weight = t[slot];
                    const float stage_value = stage[Q3(
                        tracer, k, cl[slot], nlev, ncells)];
                    const float correction = coefficient * velocity_sign
                        * third_weight;
                    const float weight = base_weight + correction;
                    const float contribution = weight * stage_value;
                    const float next = value + contribution;
#if MPAS_FTZ_FALLBACK_ENABLED
                    const unsigned int coefficient_mag =
                        mpas_f32_magnitude_bits(coefficient);
                    const unsigned int third_mag =
                        mpas_f32_magnitude_bits(third_weight);
                    const unsigned int correction_mag =
                        mpas_f32_magnitude_bits(correction);
                    const unsigned int base_mag =
                        mpas_f32_magnitude_bits(base_weight);
                    const unsigned int weight_mag =
                        mpas_f32_magnitude_bits(weight);
                    const unsigned int stage_mag =
                        mpas_f32_magnitude_bits(stage_value);
                    const unsigned int contribution_mag =
                        mpas_f32_magnitude_bits(contribution);
                    const unsigned int value_mag =
                        mpas_f32_magnitude_bits(value);
                    const unsigned int next_mag = mpas_f32_magnitude_bits(next);
                    ftz_sensitive = ftz_sensitive
                        || (base_mag != 0u && (base_mag >> 23) == 0u)
                        || (third_mag != 0u && (third_mag >> 23) == 0u)
                        || (stage_mag != 0u && (stage_mag >> 23) == 0u)
                        || (correction_mag == 0u && coefficient_mag != 0u
                            && third_mag != 0u)
                        || (weight_mag == 0u && base_mag != correction_mag
                            && base_mag != 0u && correction_mag != 0u)
                        || (contribution_mag == 0u && weight_mag != 0u
                            && stage_mag != 0u)
                        || (next_mag == 0u && value_mag != contribution_mag
                            && value_mag != 0u && contribution_mag != 0u);
#endif
                    value = next;
                }
            }
        } else {
            for (int slot = 0; slot < count; ++slot) {
                const int cell = adv_cells[ADV(edge, slot, width)];
                const float base_weight = adv[ADV(edge, slot, width)];
                const float third_weight = adv3[ADV(edge, slot, width)];
                const float stage_value = stage[Q3(
                    tracer, k, cell, nlev, ncells)];
                const float correction = coefficient * velocity_sign
                    * third_weight;
                const float weight = base_weight + correction;
                const float contribution = weight * stage_value;
                const float next = value + contribution;
#if MPAS_FTZ_FALLBACK_ENABLED
                const unsigned int coefficient_mag =
                    mpas_f32_magnitude_bits(coefficient);
                const unsigned int third_mag =
                    mpas_f32_magnitude_bits(third_weight);
                const unsigned int correction_mag =
                    mpas_f32_magnitude_bits(correction);
                const unsigned int base_mag =
                    mpas_f32_magnitude_bits(base_weight);
                const unsigned int weight_mag =
                    mpas_f32_magnitude_bits(weight);
                const unsigned int stage_mag =
                    mpas_f32_magnitude_bits(stage_value);
                const unsigned int contribution_mag =
                    mpas_f32_magnitude_bits(contribution);
                const unsigned int value_mag =
                    mpas_f32_magnitude_bits(value);
                const unsigned int next_mag = mpas_f32_magnitude_bits(next);
                ftz_sensitive = ftz_sensitive
                    || (base_mag != 0u && (base_mag >> 23) == 0u)
                    || (third_mag != 0u && (third_mag >> 23) == 0u)
                    || (stage_mag != 0u && (stage_mag >> 23) == 0u)
                    || (correction_mag == 0u && coefficient_mag != 0u
                        && third_mag != 0u)
                    || (weight_mag == 0u && base_mag != correction_mag
                        && base_mag != 0u && correction_mag != 0u)
                    || (contribution_mag == 0u && weight_mag != 0u
                        && stage_mag != 0u)
                    || (next_mag == 0u && value_mag != contribution_mag
                        && value_mag != 0u && contribution_mag != 0u);
#endif
                value = next;
            }
        }
        if (ftz_sensitive) {
            value = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int cell = adv_cells[ADV(edge, slot, width)];
                const float weight = mpas_add(
                    adv[ADV(edge, slot, width)],
                    mpas_mul(mpas_mul(coefficient, velocity_sign),
                        adv3[ADV(edge, slot, width)]));
                value = mpas_add(value, mpas_mul(weight,
                    stage[Q3(tracer, k, cell, nlev, ncells)]));
            }
        }
        edge_values[QE3(tracer, k, edge, nlev, nedges)] = value;
    }
}

/* Element form of transport_standard_finish_v841: one thread per
   (level, cell), cell fastest, walking the tracers.  The per-cell loop of
   the original left most of the card idle (one thread per cell, every level
   and tracer serial).  Each (tracer, level, cell) runs the original body
   unchanged; the tracer-independent sign*velocity product of each slot is
   the same helper call on the same operands, formed once per slot. */
extern "C" __global__ void transport_standard_finish_elem_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges, const float dt,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell,
    const float *velocity, const float *edge_values,
    const float *vertical_flux, const float *rdzw,
    const float *old, const float *rho_old, const float *target_density,
    const float *source, float *output)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    const int count = n_edges_on_cell[cell];
    const bool hoisted = count <= HEX_HOIST_CELL;
    int ed[HEX_HOIST_CELL];
    float sv[HEX_HOIST_CELL];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
        const bool live = hoisted && slot < count;
        const int edge = live ? edges_on_cell[CES(cell, slot, max_edges)] : 0;
        ed[slot] = edge;
        sv[slot] = live ? mpas_mul(-acoustic_sign[CES(cell, slot, max_edges)],
            velocity[E2(k, edge, nedges)]) : 0.0f;
    }
    const float inv_area = inv_area_cell[cell];
    const float rdzw_k = rdzw[k];
    const float rho_old_k = rho_old[C2(k, cell, ncells)];
    const float rho_new_inv = mpas_div(1.0f, target_density[C2(k, cell, ncells)]);
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        const int index = Q3(tracer, k, cell, nlev, ncells);
        float horizontal = 0.0f;
        if (hoisted) {
#pragma unroll
            for (int slot = 0; slot < HEX_HOIST_CELL; ++slot) {
                if (slot < count) {
                    horizontal = mpas_add(horizontal, mpas_mul(sv[slot],
                        edge_values[QE3(tracer, k, ed[slot], nlev, nedges)]));
                }
            }
        } else {
            for (int slot = 0; slot < count; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                horizontal = mpas_add(horizontal, mpas_mul(mpas_mul(
                    sign, velocity[E2(k, edge, nedges)]),
                    edge_values[QE3(tracer, k, edge, nlev, nedges)]));
            }
        }
        const float tendency = mpas_add(
            mpas_mul(horizontal, inv_area), source[index]);
        const float vertical = mpas_mul(rdzw_k, mpas_sub(
            vertical_flux[QI3(tracer, k + 1, cell, nlev, ncells)],
            vertical_flux[QI3(tracer, k, cell, nlev, ncells)]));
        const float numerator = mpas_add(
            mpas_mul(old[index], rho_old_k),
            mpas_mul(dt, mpas_sub(tendency, vertical)));
        output[index] = mpas_mul(numerator, rho_new_inv);
    }
}

/* Element forms of the monotonic (FCT) scalar-transport kernels.  The
   originals run one thread per cell or edge with every tracer and level
   serial in the thread (about 340k threads per 15 km rank, one wave); these
   run one thread per (tracer, level, owner), owner fastest.  No original
   carries a value across tracers or levels in its loops (each element reads
   inputs and writes only its own index), so each element runs the original
   body unchanged and carries the same bits. */
#define HEX_TKO(total_owners) \
    const int owner = element % (total_owners); \
    const int rest = element / (total_owners); \
    const int k = rest % nlev; \
    const int tracer = rest / nlev;

extern "C" __global__ void transport_target_density_elem_v841(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int advance_density, const float dt,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *dv_edge,
    const float *inv_area_cell, const float *velocity,
    const float *vertical_velocity, const float *rdzw,
    const float *rho_old, const float *rho_new, float *target_density)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    const int index = C2(k, cell, ncells);
    if (!advance_density) {
        target_density[index] = rho_new[index];
    } else {
        float horizontal = 0.0f;
        for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
            const int edge = edges_on_cell[CES(cell, slot, max_edges)];
            const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
            float contribution = mpas_mul(
                sign, velocity[E2(k, edge, nedges)]);
            contribution = mpas_mul(contribution, dv_edge[edge]);
            contribution = mpas_mul(contribution, inv_area_cell[cell]);
            horizontal = mpas_add(horizontal, contribution);
        }
        target_density[index] = mpas_add(rho_old[index], mpas_mul(dt,
            mpas_sub(horizontal, mpas_mul(rdzw[k], mpas_sub(
                vertical_velocity[C2(k + 1, cell, ncells)],
                vertical_velocity[index])))));
    }
}

extern "C" __global__ void transport_edge_values_mono_hoist_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int width, const float coefficient,
    const float *stage, const float *velocity,
    const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells, float *edge_values, const int lpt)
{
    /* One thread per (tracer, edge); the edge's stencil is read once into
       registers and the thread walks the levels.  Both branches of the
       original keep their own association: the ten-cell branch weights the
       sum and multiplies by the velocity last; the other branch multiplies
       each weight by the velocity first and accumulates from 0.0f in slot
       order, exactly as the original's in-memory accumulation did. */
    const int hex_chunks = (nlev + lpt - 1) / lpt;
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= ntracers * nedges * hex_chunks) return;
    const int edge = element % nedges;
    const int tracer = (element / nedges) % ntracers;
    const int hex_k0 = (element / nedges / ntracers) * lpt;
    const int hex_k1 = min(nlev, hex_k0 + lpt);
    const int count = n_adv[edge];
    const bool hoisted = count <= HEX_HOIST_ADV;
    int cl[HEX_HOIST_ADV];
    float b[HEX_HOIST_ADV];
    float t[HEX_HOIST_ADV];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_ADV; ++slot) {
        const bool live = hoisted && slot < count;
        cl[slot] = live ? adv_cells[ADV(edge, slot, width)] : 0;
        b[slot] = live ? adv[ADV(edge, slot, width)] : 0.0f;
        t[slot] = live ? adv3[ADV(edge, slot, width)] : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        const float vel = velocity[E2(k, edge, nedges)];
        const float sign = mpas_copysign(1.0f, vel);
        float result;
        if (count == 10 && hoisted) {
            float weight = mpas_add(b[0],
                mpas_mul(mpas_mul(coefficient, sign), t[0]));
            float value = mpas_mul(weight,
                stage[Q3(tracer, k, cl[0], nlev, ncells)]);
#pragma unroll
            for (int slot = 1; slot < 10; ++slot) {
                weight = mpas_add(b[slot],
                    mpas_mul(mpas_mul(coefficient, sign), t[slot]));
                value = mpas_add(value, mpas_mul(weight,
                    stage[Q3(tracer, k, cl[slot], nlev, ncells)]));
            }
            result = mpas_mul(vel, value);
        } else if (count == 10) {
            const int first_cell = adv_cells[ADV(edge, 0, width)];
            float weight = mpas_add(adv[ADV(edge, 0, width)],
                mpas_mul(mpas_mul(coefficient, sign),
                    adv3[ADV(edge, 0, width)]));
            float value = mpas_mul(weight,
                stage[Q3(tracer, k, first_cell, nlev, ncells)]);
            for (int slot = 1; slot < 10; ++slot) {
                const int cell = adv_cells[ADV(edge, slot, width)];
                weight = mpas_add(adv[ADV(edge, slot, width)],
                    mpas_mul(mpas_mul(coefficient, sign),
                        adv3[ADV(edge, slot, width)]));
                value = mpas_add(value, mpas_mul(weight,
                    stage[Q3(tracer, k, cell, nlev, ncells)]));
            }
            result = mpas_mul(vel, value);
        } else if (hoisted) {
            float value = 0.0f;
#pragma unroll
            for (int slot = 0; slot < HEX_HOIST_ADV; ++slot) {
                if (slot < count) {
                    float scalar_weight = mpas_add(b[slot],
                        mpas_mul(mpas_mul(coefficient, sign), t[slot]));
                    scalar_weight = mpas_mul(vel, scalar_weight);
                    value = mpas_add(value, mpas_mul(scalar_weight,
                        stage[Q3(tracer, k, cl[slot], nlev, ncells)]));
                }
            }
            result = value;
        } else {
            float value = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int cell = adv_cells[ADV(edge, slot, width)];
                float scalar_weight = mpas_add(adv[ADV(edge, slot, width)],
                    mpas_mul(mpas_mul(coefficient, sign),
                        adv3[ADV(edge, slot, width)]));
                scalar_weight = mpas_mul(vel, scalar_weight);
                value = mpas_add(value, mpas_mul(scalar_weight,
                    stage[Q3(tracer, k, cell, nlev, ncells)]));
            }
            result = value;
        }
        edge_values[QE3(tracer, k, edge, nlev, nedges)] = result;
    }
}

extern "C" __global__ void fct_minmax_source_elem_v841(
    const int ntracers, const int nlev, const int ncells,
    const int max_edges, const float dt,
    const int *n_edges_on_cell, const int *cells_on_cell,
    const float *old, const float *source, const float *rho_old,
    float *source_old, float *minimum, float *maximum)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= ntracers * nlev * ncells) return;
    HEX_TKO(ncells)
    const int cell = owner;
    const int index = Q3(tracer, k, cell, nlev, ncells);
    const float current = mpas_add(old[index], mpas_div(mpas_mul(dt, source[index]), rho_old[C2(k, cell, ncells)]));
    source_old[index] = current;
    float low = current;
    float high = current;
    if (k > 0) {
        const float value = mpas_add(old[Q3(
            tracer, k - 1, cell, nlev, ncells)], mpas_div(mpas_mul(dt, source[Q3(
                tracer, k - 1, cell, nlev, ncells)]), rho_old[C2(k - 1, cell, ncells)]));
        low = mpas_min(low, value);
        high = mpas_max(high, value);
    }
    if (k + 1 < nlev) {
        const float value = mpas_add(old[Q3(
            tracer, k + 1, cell, nlev, ncells)], mpas_div(mpas_mul(dt, source[Q3(
                tracer, k + 1, cell, nlev, ncells)]), rho_old[C2(k + 1, cell, ncells)]));
        low = mpas_min(low, value);
        high = mpas_max(high, value);
    }
    for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
        const int neighbor = cells_on_cell[CES(cell, slot, max_edges)];
        const float value = mpas_add(old[Q3(
            tracer, k, neighbor, nlev, ncells)], mpas_div(mpas_mul(dt, source[Q3(tracer, k, neighbor, nlev, ncells)]), rho_old[C2(k, neighbor, ncells)]));
        low = mpas_min(low, value);
        high = mpas_max(high, value);
    }
    minimum[index] = low;
    maximum[index] = high;
}

extern "C" __global__ void fct_edge_residual_elem_v841(
    const int ntracers, const int nlev, const int ncells, const int nedges,
    const float dt, const float *source_old, const float *velocity,
    const float *dv_edge, const int *cells_on_edge,
    const float *high_flux, float *upwind_flux, float *horizontal_residual)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= ntracers * nlev * nedges) return;
    HEX_TKO(nedges)
    const int edge = owner;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    const int index = QE3(tracer, k, edge, nlev, nedges);
    const float vel = velocity[E2(k, edge, nedges)];
    const float upwind = mpas_mul(mpas_mul(dv_edge[edge], dt),
        mpas_add(mpas_mul(plus(vel), source_old[Q3(
            tracer, k, c0, nlev, ncells)]),
            mpas_mul(minus(vel), source_old[Q3(
            tracer, k, c1, nlev, ncells)])));
    upwind_flux[index] = upwind;
    horizontal_residual[index] = mpas_sub(
        mpas_mul(dt, high_flux[index]), upwind);
}

extern "C" __global__ void fct_horizontal_low_order_elem_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell,
    const float *upwind_flux, const float *horizontal_residual,
    float *mass, float *scale_in, float *scale_out)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= ntracers * nlev * ncells) return;
    HEX_TKO(ncells)
    const int cell = owner;
    const float inverse_area = inv_area_cell[cell];
    const int index = Q3(tracer, k, cell, nlev, ncells);
    float cell_mass = mass[index];
    float incoming = scale_in[index];
    float outgoing = scale_out[index];
    for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
        const int edge = edges_on_cell[CES(cell, slot, max_edges)];
        const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
        const int qe = QE3(tracer, k, edge, nlev, nedges);
        cell_mass = mpas_add(cell_mass, mpas_mul(
            mpas_mul(sign, upwind_flux[qe]), inverse_area));
        const float signed_residual = mpas_mul(
            -sign, horizontal_residual[qe]);
        outgoing = mpas_sub(outgoing,
            mpas_mul(plus(signed_residual), inverse_area));
        incoming = mpas_sub(incoming,
            mpas_mul(minus(signed_residual), inverse_area));
    }
    mass[index] = cell_mass;
    scale_in[index] = incoming;
    scale_out[index] = outgoing;
}

extern "C" __global__ void fct_scale_elem_v841(
    const int ntracers, const int nlev, const int ncells,
    const float *minimum, const float *maximum, const float *target_density,
    const float *mass, float *scale_in, float *scale_out)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= ntracers * nlev * ncells) return;
    HEX_TKO(ncells)
    const int cell = owner;
    const int index = Q3(tracer, k, cell, nlev, ncells);
    const float density = target_density[C2(k, cell, ncells)];
    const float eps = 1.0e-20f;
    const float incoming = mpas_div(mpas_sub(
        mpas_mul(maximum[index], density), mass[index]),
        mpas_add(scale_in[index], eps));
    const float outgoing = mpas_div(mpas_sub(
        mpas_mul(minimum[index], density), mass[index]),
        mpas_sub(scale_out[index], eps));
    scale_in[index] = mpas_min(1.0f, mpas_max(0.0f, incoming));
    scale_out[index] = mpas_min(1.0f, mpas_max(0.0f, outgoing));
}

/* The scale factors depend only on this element's completed low-order
   accumulation. Keep that accumulation in registers through scaling. The
   first 14 arguments retain the low-order ABI for the LTS flux hooks. */
extern "C" __global__ void fct_horizontal_low_order_scale_elem_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell,
    const float *upwind_flux, const float *horizontal_residual,
    float *mass, float *scale_in, float *scale_out,
    const float *minimum, const float *maximum, const float *target_density)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= ntracers * nlev * ncells) return;
    HEX_TKO(ncells)
    const int cell = owner;
    const float inverse_area = inv_area_cell[cell];
    const int index = Q3(tracer, k, cell, nlev, ncells);
    float cell_mass = mass[index];
    float incoming = scale_in[index];
    float outgoing = scale_out[index];
    for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
        const int edge = edges_on_cell[CES(cell, slot, max_edges)];
        const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
        const int qe = QE3(tracer, k, edge, nlev, nedges);
        cell_mass = mpas_add(cell_mass, mpas_mul(
            mpas_mul(sign, upwind_flux[qe]), inverse_area));
        const float signed_residual = mpas_mul(
            -sign, horizontal_residual[qe]);
        outgoing = mpas_sub(outgoing,
            mpas_mul(plus(signed_residual), inverse_area));
        incoming = mpas_sub(incoming,
            mpas_mul(minus(signed_residual), inverse_area));
    }
    mass[index] = cell_mass;
    const float density = target_density[C2(k, cell, ncells)];
    const float eps = 1.0e-20f;
    const float scaled_incoming = mpas_div(mpas_sub(
        mpas_mul(maximum[index], density), cell_mass),
        mpas_add(incoming, eps));
    const float scaled_outgoing = mpas_div(mpas_sub(
        mpas_mul(minimum[index], density), cell_mass),
        mpas_sub(outgoing, eps));
    scale_in[index] = mpas_min(1.0f, mpas_max(0.0f, scaled_incoming));
    scale_out[index] = mpas_min(1.0f, mpas_max(0.0f, scaled_outgoing));
}

extern "C" __global__ void fct_limit_horizontal_elem_v841(
    const int ntracers, const int nlev, const int ncells, const int nedges,
    const int *cells_on_edge, const float *scale_in, const float *scale_out,
    float *residual)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= ntracers * nlev * nedges) return;
    HEX_TKO(nedges)
    const int edge = owner;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    const int index = QE3(tracer, k, edge, nlev, nedges);
    const float flux = residual[index];
    residual[index] = mpas_add(mpas_mul(plus(flux), mpas_min(
            scale_out[Q3(tracer, k, c0, nlev, ncells)],
            scale_in[Q3(tracer, k, c1, nlev, ncells)])),
        mpas_mul(minus(flux), mpas_min(
            scale_in[Q3(tracer, k, c0, nlev, ncells)],
            scale_out[Q3(tracer, k, c1, nlev, ncells)])));
}

extern "C" __global__ void fct_limit_vertical_elem_v841(
    const int ntracers, const int nlev, const int ncells,
    const float *scale_in, const float *scale_out, float *residual)
{
    /* interface 1..nlev-1 of every (tracer, cell); interfaces 0 and nlev
       are untouched, as in the original loop. */
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= ntracers * (nlev - 1) * ncells) return;
    const int cell = element % ncells;
    const int rest = element / ncells;
    const int interface = 1 + rest % (nlev - 1);
    const int tracer = rest / (nlev - 1);
    const int qi = QI3(tracer, interface, cell, nlev, ncells);
    const float flux = residual[qi];
    residual[qi] = mpas_add(mpas_mul(plus(flux), mpas_min(
            scale_out[Q3(
                tracer, interface - 1, cell, nlev, ncells)],
            scale_in[Q3(tracer, interface, cell, nlev, ncells)])),
        mpas_mul(minus(flux), mpas_min(
            scale_out[Q3(tracer, interface, cell, nlev, ncells)],
            scale_in[Q3(
                tracer, interface - 1, cell, nlev, ncells)])));
}

extern "C" __global__ void fct_finish_elem_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzw,
    const float *horizontal_residual, const float *vertical_residual,
    const float *target_density, const float *mass, float *output)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= ntracers * nlev * ncells) return;
    HEX_TKO(ncells)
    const int cell = owner;
    const float inverse_area = inv_area_cell[cell];
    const int index = Q3(tracer, k, cell, nlev, ncells);
    float value = mass[index];
    for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
        const int edge = edges_on_cell[CES(cell, slot, max_edges)];
        const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
        value = mpas_add(value, mpas_mul(mpas_mul(sign,
            horizontal_residual[QE3(
                tracer, k, edge, nlev, nedges)]), inverse_area));
    }
    value = mpas_sub(value, mpas_mul(rdzw[k], mpas_sub(
        vertical_residual[QI3(
            tracer, k + 1, cell, nlev, ncells)],
        vertical_residual[QI3(
            tracer, k, cell, nlev, ncells)])));
    output[index] = mpas_max(0.0f,
        mpas_div(value, target_density[C2(k, cell, ncells)]));
}
"""

_CACHE: Any | None = None
_KERNELS: dict[tuple[Any, str] | tuple[Any, str, tuple], Any] = {}


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
    # LTS taps replace unspecialized kernels through the shared two-part key.
    # Only shape-specialized kernels need dimensions in their identity.
    key = (selected, name, dimensions) if dimensions else (selected, name)
    if key not in _KERNELS:
        if dimensions:
            from .cuda_shape import resolve
            _KERNELS[key] = resolve(selected, name, _CUDA_SOURCE,
                                    "hexcore.cuda_transport_v841", dimensions)
        else:
            _KERNELS[key] = selected.raw_kernel(
                name, _CUDA_SOURCE, module_key="hexcore.cuda_transport_v841")
    return _KERNELS[key]


def _launch(
    name: str,
    count: int,
    arguments: tuple[Any, ...],
    kernel_cache: Any | None,
) -> None:
    threads = 128
    dimensions = tuple(zip(("ntracers", "nlev", "ncells", "nedges", "width"), arguments[:5])) if name == "transport_edge_values_hoist_v841" else ()
    _kernel(name, kernel_cache, dimensions)(
        ((int(count) + threads - 1) // threads,), (threads,), arguments
    )


def _launch_checked(new: str, new_count: int, old: str, old_count: int,
                    inputs: tuple[Any, ...], outputs: tuple[Any, ...],
                    kernel_cache: Any | None, *, in_place: bool = False) -> None:
    """Launch the element form; under WOOF_HEX_EDGE_HOIST_CHECK also run the
    original on copies (in-place outputs) or fresh outputs and compare words."""
    from . import cuda_edge_hoist
    cp = _cupy()
    check = cuda_edge_hoist.check_wanted(new, cp)
    reference = None
    if check:
        reference = tuple(cp.array(value, copy=True) if in_place else cp.empty_like(value)
                          for value in outputs)
    _launch(new, new_count, (*inputs, *outputs), kernel_cache)
    if check:
        _launch(old, old_count, (*inputs, *reference), kernel_cache)
        for value, expected in zip(outputs, reference):
            cuda_edge_hoist.compare(new, value, expected, cp)


def _common_inputs(
    mesh: Any,
    context: CudaV841Context,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
) -> tuple[Any, ...]:
    values = _v823._inputs(
        mesh,
        scalar_old,
        scalar_stage,
        rho_zz_old,
        rho_zz_new,
        uh_avg,
        ww_avg,
        fzm,
        fzp,
        rdzw,
    )
    nlev, ncells, nedges = values[11], values[12], values[13]
    context.validate(
        n_vert_levels=nlev,
        n_cells=ncells,
        n_edges=nedges,
        n_vertices=int(mesh.n_vertices),
    )
    for name, value in (("fzm", values[7]), ("fzp", values[8]), ("rdzw", values[9])):
        if tuple(value.shape) != (nlev,):
            raise ValueError(f"{name} must have shape ({nlev},)")
    return values


def _check_density(
    density: Any,
    *,
    name: str,
    count: int,
    invalid_flag: Any | None,
    kernel_cache: Any | None,
) -> Any:
    cp = _cupy()
    owns_flag = invalid_flag is None
    invalid = (
        cp.zeros((1,), dtype=cp.int32)
        if owns_flag
        else _v823._i32(invalid_flag)
    )
    if tuple(invalid.shape) != (1,):
        raise ValueError("validation_flag must have shape (1,)")
    _launch(
        "validate_density_v841",
        count,
        (np.int32(count), density, invalid),
        kernel_cache,
    )
    if owns_flag and int(cp.asnumpy(invalid)[0]) != 0:
        raise ValueError(f"{name} must remain finite and strictly positive")
    return invalid


def _validate_geometry_and_coefficients(
    mesh: Any,
    coefficients: CudaAdvectionCoefficients,
    *,
    ncells: int,
    nedges: int,
    indices_prevalidated: bool,
    kernel_cache: Any | None,
) -> tuple[Any, ...]:
    cp = _cupy()
    coefficients.validate()
    coefficient_shape = tuple(coefficients.adv_coefs.shape)
    if len(coefficient_shape) != 2 or coefficient_shape[0] != nedges:
        raise ValueError("advection coefficient edge extent differs from mesh")
    width = int(coefficient_shape[1])
    geometry = _v823._geometry(mesh)
    counts, edges, neighbors, cells_on_edge, signs, dv, area, max_edges = geometry
    expected = {
        "n_edges_on_cell": (counts, (ncells,)),
        "edges_on_cell": (edges, (ncells, max_edges)),
        "cells_on_cell": (neighbors, (ncells, max_edges)),
        "cells_on_edge": (cells_on_edge, (nedges, 2)),
        "edge_sign_on_cell": (signs, (ncells, max_edges)),
        "dv_edge": (dv, (nedges,)),
        "area_cell": (area, (ncells,)),
    }
    for name, (value, shape) in expected.items():
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
    if not indices_prevalidated:
        invalid = cp.zeros((1,), dtype=cp.int32)
        _launch(
            "validate_transport_indices_v841",
            max(ncells, nedges),
            (
                np.int32(ncells), np.int32(nedges), np.int32(max_edges),
                np.int32(width), counts, edges, neighbors, cells_on_edge,
                coefficients.n_adv_cells_for_edge,
                coefficients.adv_cells_for_edge, invalid,
            ),
            kernel_cache,
        )
        if int(cp.asnumpy(invalid)[0]) != 0:
            raise ValueError(
                "active transport connectivity or stencil index is out of range"
            )
    return geometry


def _standard_high_fluxes(
    stage: Any,
    velocity: Any,
    vertical_velocity: Any,
    coefficients: CudaAdvectionCoefficients,
    fzm: Any,
    fzp: Any,
    coefficient: float,
    ntracers: int,
    nlev: int,
    ncells: int,
    nedges: int,
    kernel_cache: Any | None,
) -> tuple[Any, Any]:
    cp = _cupy()
    edge = cp.empty((ntracers, nlev, nedges), dtype=cp.float32)
    vertical = cp.empty((ntracers, nlev + 1, ncells), dtype=cp.float32)
    width = int(coefficients.adv_coefs.shape[1])
    edge_arguments = (
        np.int32(ntracers), np.int32(nlev), np.int32(ncells),
        np.int32(nedges), np.int32(width), np.float32(coefficient),
        stage, velocity, coefficients.adv_coefs,
        coefficients.adv_coefs_3rd,
        coefficients.n_adv_cells_for_edge,
        coefficients.adv_cells_for_edge,
    )
    from . import cuda_edge_hoist
    if cuda_edge_hoist.enabled(nedges):
        lpt = cuda_edge_hoist.levels_per_thread(
            "transport_edge_values_hoist_v841", ntracers * nedges, nlev, lambda value: _launch(
                "transport_edge_values_hoist_v841", ntracers * nedges * cuda_edge_hoist.chunks(nlev, value),
                (*edge_arguments, edge, np.int32(value)), kernel_cache), cp)
        _launch("transport_edge_values_hoist_v841", ntracers * nedges * cuda_edge_hoist.chunks(nlev, lpt),
                (*edge_arguments, edge, np.int32(lpt)), kernel_cache)
        if cuda_edge_hoist.check_wanted("transport_edge_values_hoist_v841", cp):
            element = cp.empty_like(edge)
            _launch("transport_edge_values", nedges, (*edge_arguments, element), kernel_cache)
            cuda_edge_hoist.compare("transport_edge_values_hoist_v841", edge, element, cp)
    else:
        _launch("transport_edge_values", nedges, (*edge_arguments, edge), kernel_cache)
    _launch(
        "transport_vertical_flux",
        ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.float32(coefficient), stage, vertical_velocity, fzm, fzp,
            vertical,
        ),
        kernel_cache,
    )
    return edge, vertical


def _source(
    old: Any,
    scalar_tendency: Any | None,
    cp: Any,
    *,
    squeezed: bool,
) -> Any:
    if scalar_tendency is None:
        return cp.zeros_like(old)
    source = _v823._f32(scalar_tendency, "scalar_tendency")
    expected_ndim = 2 if squeezed else 3
    if source.ndim != expected_ndim:
        raise ValueError("scalar_tendency carrier rank differs from scalar_old")
    if squeezed:
        source = source[None, ...]
    if source.shape != old.shape:
        raise ValueError("scalar_tendency shape differs from scalar_old")
    return source


def advance_scalars_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    dt: float,
    *,
    coefficients: CudaAdvectionCoefficients,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    rk_step: int,
    config_time_integration_order: int,
    config_scalar_adv_order: int,
    config_scalar_vadv_order: int,
    config_coef_3rd_order: float,
    scalar_tendency: Any | None = None,
    advance_density: bool = False,
    validation_flag: Any | None = None,
    indices_prevalidated: bool = False,
    rho_old_prevalidated: bool = False,
    kernel_cache: Any | None = None,
    halo_exchange: Any | None = None,
    config_apply_lbcs: bool = False,
    bdy_mask_cell: Any | None = None,
    bdy_mask_edge: Any | None = None,
    garbage_cell: int | None = None,
) -> CudaScalarTransportResult:
    # ``halo_exchange`` is accepted for signature parity with the monotonic
    # lane: RK stages 1/2 need no mid-function round -- the executor's D round
    # at stage entry covers every ring<=2 read in this function.
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    _v823._validate_orders(
        config_scalar_adv_order, config_scalar_vadv_order,
        config_coef_3rd_order,
    )
    if rk_step not in (1, 2, 3) or config_time_integration_order not in (2, 3):
        raise ConfigurationRefusal(
            "config_time_integration_order", config_time_integration_order,
            "the v8.4.1 scalar path has three stages at order 2 or 3", "config_time_integration_order=2 or 3",
        )
    values = _common_inputs(
        mesh, context, scalar_old, scalar_stage, rho_zz_old, rho_zz_new,
        uh_avg, ww_avg, fzm, fzp, rdzw,
    )
    (
        old, stage, squeezed, rho0, rho1, velocity, vertical_velocity,
        fnm, fnp, rdnw, ntracers, nlev, ncells, nedges,
    ) = values
    cp = _cupy()
    source = _source(old, scalar_tendency, cp, squeezed=squeezed)
    counts, edges, _neighbors, _cells, signs, _dv, _area, max_edges = (
        _validate_geometry_and_coefficients(
            mesh, coefficients, ncells=ncells, nedges=nedges,
            indices_prevalidated=indices_prevalidated,
            kernel_cache=kernel_cache,
        )
    )
    if not rho_old_prevalidated:
        _check_density(
            rho0, name="rho_zz_old", count=nlev * ncells,
            invalid_flag=None, kernel_cache=kernel_cache,
        )
    weight = np.float32(
        1.0 / 3.0 if advance_density and rk_step == 1 and config_time_integration_order == 3
        else 0.5 if advance_density and rk_step == 1
        else 0.5 if advance_density and rk_step == 2
        else 1.0
    )
    target = cp.empty_like(rho0)
    _launch(
        "transport_interpolate_target_v841",
        nlev * ncells,
        (
            np.int32(nlev * ncells), weight, rho0, rho1, target,
        ),
        kernel_cache,
    )
    _check_density(
        target, name="target_density", count=nlev * ncells,
        invalid_flag=validation_flag, kernel_cache=kernel_cache,
    )
    output = cp.empty_like(old)
    # Census item 4, the unlimited stages: species per pass, as the limiter
    # (limiter_species_per_pass).  The edge and vertical flux blocks were
    # (species, level, edge) and (species, level+1, cell) at once.
    per_pass = limiter_species_per_pass(
        ntracers, nlev=nlev, ncells=ncells, nedges=nedges,
        available_bytes=_workspace_available(cp),
    )
    passes, copies = _species_work(cp, old, stage, source, per_pass, halo_exchange)
    from . import cuda_edge_hoist
    for first, last in passes:
        count = last - first
        edge_values, vertical_flux = _standard_high_fluxes(
            stage[first:last], velocity, vertical_velocity, coefficients, fnm, fnp,
            config_coef_3rd_order, count, nlev, ncells, nedges, kernel_cache,
        )
        finish_arguments = (
            np.int32(count), np.int32(nlev), np.int32(ncells),
            np.int32(nedges), np.int32(max_edges), np.float32(dt), counts,
            edges, signs, context.inv_area_cell, velocity, edge_values,
            vertical_flux, rdnw, old[first:last], rho0, target, source[first:last],
        )
        output_pass = output[first:last]
        if cuda_edge_hoist.enabled(nedges):
            _launch("transport_standard_finish_elem_v841", nlev * ncells, (*finish_arguments, output_pass), kernel_cache)
            if cuda_edge_hoist.check_wanted("transport_standard_finish_elem_v841", cp):
                element = cp.empty_like(output_pass)
                _launch("transport_standard_finish_v841", ncells, (*finish_arguments, element), kernel_cache)
                cuda_edge_hoist.compare("transport_standard_finish_elem_v841", output_pass, element, cp)
        else:
            _launch("transport_standard_finish_v841", ncells, (*finish_arguments, output_pass), kernel_cache)
        del edge_values, vertical_flux, finish_arguments, output_pass
    _copy_zero_species(output, copies)
    return CudaScalarTransportResult(output[0] if squeezed else output, target)


def advance_scalars_monotonic_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    dt: float,
    *,
    coefficients: CudaAdvectionCoefficients,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    config_scalar_adv_order: int,
    config_scalar_vadv_order: int,
    config_coef_3rd_order: float,
    scalar_tendency: Any | None = None,
    advance_density: bool = False,
    n_halos: int = 3,
    validation_flag: Any | None = None,
    indices_prevalidated: bool = False,
    rho_old_prevalidated: bool = False,
    kernel_cache: Any | None = None,
    halo_exchange: Any | None = None,
    config_apply_lbcs: bool = False,
    bdy_mask_cell: Any | None = None,
    bdy_mask_edge: Any | None = None,
    garbage_cell: int | None = None,
    bind_density_pool: Any | None = None,
) -> CudaScalarTransportResult:
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    _v823._validate_orders(
        config_scalar_adv_order, config_scalar_vadv_order,
        config_coef_3rd_order,
    )
    if n_halos < 3:
        raise ConfigurationRefusal("nHalos", n_halos, "FCT requires three halos", "nHalos=3")
    values = _common_inputs(
        mesh, context, scalar_old, scalar_stage, rho_zz_old, rho_zz_new,
        uh_avg, ww_avg, fzm, fzp, rdzw,
    )
    (
        old, stage, squeezed, rho0, rho1, velocity, vertical_velocity,
        fnm, fnp, rdnw, ntracers, nlev, ncells, nedges,
    ) = values
    cp = _cupy()
    if garbage_cell is not None:
        if (not config_apply_lbcs or type(garbage_cell) is not int
            or garbage_cell != ncells - 1):
            raise ValueError("a regional FCT garbage cell must name an allocated pool element")
        # Native excludes the allocation sentinel from its loops. The
        # rectangular device launches use unit density in that inert scratch
        # column so source/rho is defined, then restore its stage value.
        rho0 = cp.array(rho0, copy=True)
        if bind_density_pool is not None:
            bind_density_pool(rho0)
        _launch("fct_pool_density_v841", nlev,
            (np.int32(nlev), np.int32(ncells), np.int32(garbage_cell), rho0), kernel_cache)
    source = (None if scalar_tendency is None
              else _source(old, scalar_tendency, cp, squeezed=squeezed))
    counts, edges, neighbors, cells_on_edge, signs, dv, _area, max_edges = (
        _validate_geometry_and_coefficients(
            mesh, coefficients, ncells=ncells, nedges=nedges,
            indices_prevalidated=indices_prevalidated,
            kernel_cache=kernel_cache,
        )
    )
    if not rho_old_prevalidated:
        _check_density(
            rho0, name="rho_zz_old", count=nlev * ncells,
            invalid_flag=None, kernel_cache=kernel_cache,
        )
    target = cp.empty_like(rho0)
    if garbage_cell is not None and bind_density_pool is not None:
        bind_density_pool(target)
    from . import cuda_edge_hoist
    element_fct = cuda_edge_hoist.enabled(nedges)
    target_inputs = (
        np.int32(nlev), np.int32(ncells), np.int32(nedges),
        np.int32(max_edges), np.int32(bool(advance_density)), np.float32(dt),
        counts, edges, signs, dv, context.inv_area_cell, velocity,
        vertical_velocity, rdnw, rho0, rho1,
    )
    if element_fct:
        _launch_checked("transport_target_density_elem_v841", nlev * ncells,
                        "transport_target_density_v841", ncells,
                        target_inputs, (target,), kernel_cache)
    else:
        _launch("transport_target_density_v841", ncells, (*target_inputs, target), kernel_cache)
    if garbage_cell is not None:
        _launch("fct_pool_density_v841", nlev,
            (np.int32(nlev), np.int32(ncells), np.int32(garbage_cell), target), kernel_cache)
    _check_density(
        target, name="target_density", count=nlev * ncells,
        invalid_flag=validation_flag, kernel_cache=kernel_cache,
    )
    width = int(coefficients.adv_coefs.shape[1])
    # Census item 4 (the 2026-10-14 memory census, ranked opportunities):
    # the limiter runs a few species per pass.  Every FCT kernel indexes
    # tracer-major (tracer, level, entity) and never reads across tracers, so
    # a pass over the contiguous slice [first, last) launches the same body on
    # the same words, in the same order per element, as the all-species launch
    # did; only the extent of the work arrays and the number of launches
    # change.  The target density above is species independent and is formed
    # once.  The breakage it prevents: the all-species work arrays were the
    # ordinary-step device peak of the default route (23.7 KB per column at
    # x1.40962; 11,580 MiB live at the 30 km step that ran a 96 GB card out of
    # memory, see _monotonic_species_pass).
    output = cp.empty_like(old)
    per_pass = limiter_species_per_pass(
        ntracers, halo_exchange, nlev=nlev, ncells=ncells, nedges=nedges,
        available_bytes=_workspace_available(cp),
    )
    passes, copies = _species_work(cp, old, stage, source, per_pass, halo_exchange)
    parts = len(passes)
    # An exchanger whose interface tables act inside the limiter (the one-card
    # active LTS class exchanger: packets, donor guard and reserves per tracer)
    # is told which contiguous species each pass carries.
    begin_species_pass = getattr(halo_exchange, "begin_species_pass", None)
    for part, (first, last) in enumerate(passes):
        if begin_species_pass is not None:
            begin_species_pass(first, last)
        _monotonic_species_pass(
            cp, old[first:last], stage[first:last],
            None if source is None else source[first:last],
            output[first:last], last - first,
            nlev=nlev, ncells=ncells, nedges=nedges, max_edges=max_edges,
            width=width, dt=dt, config_coef_3rd_order=config_coef_3rd_order,
            coefficients=coefficients, context=context, velocity=velocity,
            vertical_velocity=vertical_velocity, fnm=fnm, fnp=fnp, rdnw=rdnw,
            rho0=rho0, target=target, counts=counts, edges=edges,
            neighbors=neighbors, cells_on_edge=cells_on_edge, signs=signs,
            dv=dv, element_fct=element_fct, halo_exchange=halo_exchange,
            config_apply_lbcs=config_apply_lbcs, bdy_mask_cell=bdy_mask_cell,
            bdy_mask_edge=bdy_mask_edge, kernel_cache=kernel_cache,
            part=part, parts=parts,
        )
    _copy_zero_species(output, copies)
    return CudaScalarTransportResult(output[0] if squeezed else output, target)


#: Shape-less schedule callers retain one species per pass. Forecast calls
#: batch within the scratch budget below; the per-element arithmetic is unchanged.
LIMITER_SPECIES_PER_PASS = 1
TRANSPORT_WORKSPACE_BYTES = 2 * 1024**3


def transport_workspace_bytes(nlev: int, ncells: int, nedges: int) -> int:
    """Conservative one-species peak, including 12.5% allocator headroom.

    The FCT residual stage holds three edge blocks, six cell blocks and one
    interface block. Earlier stages hold fewer bytes on atmosphere meshes.
    Include the alternate peak explicitly for unusual cell/edge ratios.
    Caller-owned old/stage/source/output/target fields are already allocated.
    """
    cell = int(nlev) * int(ncells)
    edge = int(nlev) * int(nedges)
    interface = (int(nlev) + 1) * int(ncells)
    words = max(3 * edge + 6 * cell + interface, edge + 6 * cell + 2 * interface)
    return max(1, (4 * words * 9 + 7) // 8)


def _workspace_available(cp: Any) -> int:
    """Idle allocator blocks plus driver-free bytes, with 25% left in reserve."""
    free, _total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    if cp.cuda.get_allocator() == pool.malloc:
        free += pool.free_bytes()
    return int(free) * 3 // 4


def limiter_species_per_pass(
    ntracers: int, halo_exchange: Any | None = None, *,
    nlev: int | None = None, ncells: int | None = None,
    nedges: int | None = None, available_bytes: int | None = None,
) -> int:
    """How many species one scalar-transport pass carries.

    On a multi-card rank the limiter exchanges its scale factors (round E)
    once per pass, so the passes are the round-E slots the rank's exchanger
    declares (``fct_scale_parts``, the run of E in its schedule;
    partition_executor_v841.expected_round_sequence).  An exchanger that
    declares one, or none (the multi-card LTS block exchangers, whose round E
    is one peer message per stage), keeps every species in one pass.  An
    exchanger whose round E receives nothing declares ``fct_scale_parts =
    None`` (the one-card active LTS class exchanger): the route default below
    applies, exactly as with no exchanger.
    """

    ntracers = int(ntracers)
    if ntracers <= 0:
        return 1
    declared = (1 if halo_exchange is None
                else getattr(halo_exchange, "fct_scale_parts", 1))
    if halo_exchange is not None and declared is not None:
        parts = int(declared or 1)
        if parts <= 1:
            return ntracers
        per_pass = -(-ntracers // parts)
        if -(-ntracers // per_pass) != parts:
            raise ValueError(
                f"the halo exchanger declares {parts} round-E parts, which "
                f"{ntracers} species cannot be split into"
            )
        return per_pass
    value = os.environ.get("WOOF_HEX_LIMITER_SPECIES_PER_PASS")
    if value is None and all(v is not None for v in (nlev, ncells, nedges)):
        budget = int(os.environ.get("WOOF_HEX_TRANSPORT_WORKSPACE_BYTES",
                                    TRANSPORT_WORKSPACE_BYTES))
        if budget <= 0:
            raise ValueError("WOOF_HEX_TRANSPORT_WORKSPACE_BYTES must be positive")
        if available_bytes is not None:
            budget = min(budget, max(0, int(available_bytes)))
        # One species is the historical minimum, even if the estimate exceeds
        # the remaining memory. Never widen an already memory-bound pass.
        per_pass = max(1, budget // transport_workspace_bytes(nlev, ncells, nedges))
    else:
        per_pass = LIMITER_SPECIES_PER_PASS if value is None else int(value)
    if per_pass <= 0:
        return ntracers
    return min(per_pass, ntracers)



_ZERO_CLASSIFIER: Any | None = None
_ZERO_LEVEL_REDUCER: Any | None = None


def _species_work(cp: Any, old: Any, stage: Any, source: Any | None,
                  per_pass: int, halo_exchange: Any | None):
    """Batch contiguous species, computing repeated positive-zero inputs once.

    Classify all three inputs on the device with a fused integer OR reduction.
    Only one uint32 per species returns to the host. Reclassify on every call:
    a physics source or stage update immediately reactivates a dormant tracer.
    We reuse an actual computed result, rather than fill zero, because signed
    zeros and non-finite flow/coefficient arithmetic must remain identical.
    Rank/LTS exchanges have ordered per-species side effects, and CUDA graphs
    cannot synchronize for the classification, so neither route deduplicates.
    """
    ntracers = int(old.shape[0])
    whole = [(first, min(first + per_pass, ntracers))
             for first in range(0, ntracers, per_pass)]
    mode = os.environ.get("WOOF_HEX_TRANSPORT_ZERO_REUSE", "1")
    if mode not in ("0", "1"):
        raise ValueError("WOOF_HEX_TRANSPORT_ZERO_REUSE must be 0 or 1")
    if (mode == "0" or ntracers < 2 or halo_exchange is not None
            or cp.cuda.get_current_stream().is_capturing()):
        return whole, []
    global _ZERO_CLASSIFIER, _ZERO_LEVEL_REDUCER
    if _ZERO_CLASSIFIER is None:
        _ZERO_CLASSIFIER = cp.ReductionKernel(
            "uint32 old, uint32 stage, uint32 source",
            "uint32 out", "old | stage | source", "a | b", "out = a", "0",
            "hex_transport_species_words",
        )
    if _ZERO_LEVEL_REDUCER is None:
        _ZERO_LEVEL_REDUCER = cp.ReductionKernel(
            "uint32 partial", "uint32 out", "partial", "a | b", "out = a", "0",
            "hex_transport_species_levels",
        )
    # A joint (level, cell) reduction launches only one block per tracer.
    # On the 167254 x 55 mesh that left 11 blocks on a 170-SM card and
    # cost 22.45 ms per step. Reduce cells independently at every level,
    # then reduce the tiny level vector. Integer OR is associative even
    # for signed-zero, NaN payload and subnormal bit patterns.
    partial = _ZERO_CLASSIFIER(
        old.view(cp.uint32), stage.view(cp.uint32),
        (old if source is None else source).view(cp.uint32), axis=(2,),
    )
    words = _ZERO_LEVEL_REDUCER(partial, axis=(1,))
    return species_work_plan(cp.asnumpy(words), per_pass)


def species_work_plan(words: Any, per_pass: int):
    """Plan contiguous launch slices and (source, first, last) copy spans."""
    words = np.asarray(words, dtype=np.uint32)
    if words.ndim != 1 or per_pass < 1:
        raise ValueError("species words must be a vector and pass width positive")
    zero = np.flatnonzero(words == 0)
    active = np.ones(words.size, dtype=bool)
    active[zero[1:]] = False
    passes, copies = [], []
    first = 0
    while first < words.size:
        last = first + 1
        while last < words.size and active[last] == active[first]:
            last += 1
        if active[first]:
            passes.extend((i, min(i + per_pass, last))
                          for i in range(first, last, per_pass))
        else:
            copies.append((int(zero[0]), first, last))
        first = last
    return passes, copies


def _copy_zero_species(output: Any, copies: Any) -> None:
    for reference, first, last in copies:
        output[first:last] = output[reference]

def _monotonic_species_pass(
    cp: Any, old: Any, stage: Any, source: Any | None, output: Any, ntracers: int, *,
    nlev: int, ncells: int, nedges: int, max_edges: int, width: int, dt: float,
    config_coef_3rd_order: float, coefficients: CudaAdvectionCoefficients,
    context: CudaV841Context, velocity: Any, vertical_velocity: Any, fnm: Any,
    fnp: Any, rdnw: Any, rho0: Any, target: Any, counts: Any, edges: Any,
    neighbors: Any, cells_on_edge: Any, signs: Any, dv: Any, element_fct: bool,
    halo_exchange: Any | None, config_apply_lbcs: bool, bdy_mask_cell: Any | None,
    bdy_mask_edge: Any | None, kernel_cache: Any | None, part: int = 0, parts: int = 1,
) -> None:
    """Flux-corrected transport of ``ntracers`` contiguous species.

    ``old``, ``stage``, ``source`` and ``output`` are this pass's slices of
    the caller's (species, level, cell) blocks; ``output`` is written in place.
    A ``source`` of None is the zero tendency.
    """

    if source is None:
        source = cp.zeros_like(old)
    high_edge = cp.empty((ntracers, nlev, nedges), dtype=cp.float32)
    mono_inputs = (
        np.int32(ntracers), np.int32(nlev), np.int32(ncells),
        np.int32(nedges), np.int32(width),
        np.float32(config_coef_3rd_order), stage, velocity,
        coefficients.adv_coefs, coefficients.adv_coefs_3rd,
        coefficients.n_adv_cells_for_edge, coefficients.adv_cells_for_edge,
    )
    if element_fct:
        from . import cuda_edge_hoist
        lpt = cuda_edge_hoist.levels_per_thread(
            "transport_edge_values_mono_hoist_v841", ntracers * nedges, nlev, lambda value: _launch(
                "transport_edge_values_mono_hoist_v841",
                ntracers * nedges * cuda_edge_hoist.chunks(nlev, value),
                (*mono_inputs, high_edge, np.int32(value)), kernel_cache), cp)
        _launch("transport_edge_values_mono_hoist_v841", ntracers * nedges * cuda_edge_hoist.chunks(nlev, lpt),
                (*mono_inputs, high_edge, np.int32(lpt)), kernel_cache)
        if cuda_edge_hoist.check_wanted("transport_edge_values_mono_hoist_v841", cp):
            element = cp.empty_like(high_edge)
            _launch("transport_edge_values_mono_v841", nedges, (*mono_inputs, element), kernel_cache)
            cuda_edge_hoist.compare("transport_edge_values_mono_hoist_v841", high_edge, element, cp)
    else:
        _launch("transport_edge_values_mono_v841", nedges, (*mono_inputs, high_edge), kernel_cache)
    del mono_inputs
    high_vertical = cp.empty(
        (ntracers, nlev + 1, ncells), dtype=cp.float32
    )
    _launch(
        "transport_vertical_flux",
        ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.float32(config_coef_3rd_order), stage, vertical_velocity,
            fnm, fnp, high_vertical,
        ),
        kernel_cache,
    )
    source_old = cp.empty_like(old)
    minimum = cp.empty_like(old)
    maximum = cp.empty_like(old)
    minmax_inputs = (
        np.int32(ntracers), np.int32(nlev), np.int32(ncells),
        np.int32(max_edges), np.float32(dt), counts, neighbors, old,
        source, rho0,
    )
    if element_fct:
        _launch_checked("fct_minmax_source_elem_v841", ntracers * nlev * ncells,
                        "fct_minmax_source", ncells, minmax_inputs,
                        (source_old, minimum, maximum), kernel_cache)
    else:
        _launch("fct_minmax_source", ncells, (*minmax_inputs, source_old, minimum, maximum), kernel_cache)
    # THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-11, 8-card RTX PRO 6000 host, one 96 GB card,
    # 30 km x1.655362, six species, bare MPAS-A defaults): the limiter held
    # every one of its work arrays to the end of the call -- three
    # (species, level, edge) arrays of 2.6 GB and ten (species, level, cell)
    # arrays of 0.87 GB, 16.5 GB at once -- and the run stopped at step 18
    # with "Out of memory allocating 2,595,225,600 bytes" at the `upwind`
    # allocation below.  Each array is released here as soon as the last
    # kernel that reads it has been launched.  Launches are ordered on one
    # stream, so a block handed back to the pool is rewritten only after
    # its reader ran; no launch, argument or arithmetic line moves.
    # Merged with the per-element launches (2026-10-11): the launch input
    # tuples also hold these arrays, so each tuple is released with them.
    del source, minmax_inputs
    mass = cp.empty_like(old)
    vertical_residual = cp.empty_like(high_vertical)
    scale_in = cp.empty_like(old)
    scale_out = cp.empty_like(old)
    _launch(
        "fct_vertical_low_order_v841", ntracers * ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.float32(dt), source_old, rho0, vertical_velocity, rdnw,
            high_vertical, mass, vertical_residual, scale_in, scale_out,
        ), kernel_cache,
    )
    del high_vertical
    upwind = cp.empty_like(high_edge)
    horizontal_residual = cp.empty_like(high_edge)
    residual_inputs = (
        np.int32(ntracers), np.int32(nlev), np.int32(ncells),
        np.int32(nedges), np.float32(dt), source_old, velocity, dv,
        cells_on_edge, high_edge,
    )
    if element_fct:
        _launch_checked("fct_edge_residual_elem_v841", ntracers * nlev * nedges,
                        "fct_edge_residual_v841", nedges, residual_inputs,
                        (upwind, horizontal_residual), kernel_cache)
    else:
        _launch("fct_edge_residual_v841", nedges, (*residual_inputs, upwind, horizontal_residual), kernel_cache)
    del high_edge, source_old, residual_inputs
    if config_apply_lbcs:
        if bdy_mask_cell is None or bdy_mask_edge is None:
            raise ValueError("regional FCT requires both boundary masks")
        bdy_mask_cell = _v823._i32(bdy_mask_cell)
        bdy_mask_edge = _v823._i32(bdy_mask_edge)
        if tuple(bdy_mask_cell.shape) != (ncells,) or tuple(bdy_mask_edge.shape) != (nedges,):
            raise ValueError("regional FCT masks must include every pool element")
        _launch("fct_boundary_residual_v841", ntracers * nlev * nedges,
            (np.int32(ntracers), np.int32(nlev), np.int32(nedges),
             bdy_mask_edge, horizontal_residual), kernel_cache)
    low_inputs = (
        np.int32(ntracers), np.int32(nlev), np.int32(ncells),
        np.int32(nedges), np.int32(max_edges), counts, edges, signs,
        context.inv_area_cell, upwind, horizontal_residual,
    )
    scale_inputs = (
        np.int32(ntracers), np.int32(nlev), np.int32(ncells), minimum,
        maximum, target, mass,
    )
    if element_fct:
        check = cuda_edge_hoist.check_wanted("fct_horizontal_low_order_scale_elem_v841", cp)
        if check:
            reference = tuple(cp.array(value, copy=True) for value in (mass, scale_in, scale_out))
        _launch("fct_horizontal_low_order_scale_elem_v841", ntracers * nlev * ncells,
                (*low_inputs, mass, scale_in, scale_out, minimum, maximum, target), kernel_cache)
        if check:
            _launch("fct_horizontal_low_order_v841", ncells, (*low_inputs, *reference), kernel_cache)
            _launch("fct_scale", ncells, (*scale_inputs[:-1], *reference), kernel_cache)
            for value, expected in zip((mass, scale_in, scale_out), reference):
                cuda_edge_hoist.compare("fct_horizontal_low_order_scale_elem_v841", value, expected, cp)
            del reference
    else:
        _launch("fct_horizontal_low_order_v841", ncells, (*low_inputs, mass, scale_in, scale_out), kernel_cache)
        _launch("fct_scale", ncells, (*scale_inputs, scale_in, scale_out), kernel_cache)
    del upwind, low_inputs, minimum, maximum, scale_inputs
    if halo_exchange is not None:
        # Round E (two-rank executor): fct_limit_horizontal reads BOTH endpoint
        # cells' scale factors at owned edges, so ring-1/2 scale factors must be
        # owner truth first (a boundary cell's own factors read ring-2-edge
        # residuals whose input cones leave the K=2 halo).
        if parts == 1:
            halo_exchange.round_fct_scale(scale_in, scale_out)
        else:
            halo_exchange.round_fct_scale(scale_in, scale_out, part=part, parts=parts)
    limit_h_inputs = (
        np.int32(ntracers), np.int32(nlev), np.int32(ncells),
        np.int32(nedges), cells_on_edge, scale_in, scale_out,
    )
    limit_v_inputs = (
        np.int32(ntracers), np.int32(nlev), np.int32(ncells), scale_in,
        scale_out,
    )
    if element_fct:
        _launch_checked("fct_limit_horizontal_elem_v841", ntracers * nlev * nedges,
                        "fct_limit_horizontal", nedges, limit_h_inputs,
                        (horizontal_residual,), kernel_cache, in_place=True)
        _launch_checked("fct_limit_vertical_elem_v841", ntracers * (nlev - 1) * ncells,
                        "fct_limit_vertical", ncells, limit_v_inputs,
                        (vertical_residual,), kernel_cache, in_place=True)
    else:
        _launch("fct_limit_horizontal", nedges, (*limit_h_inputs, horizontal_residual), kernel_cache)
        _launch("fct_limit_vertical", ncells, (*limit_v_inputs, vertical_residual), kernel_cache)
    del scale_in, scale_out, limit_h_inputs, limit_v_inputs
    finish_inputs = (
        np.int32(ntracers), np.int32(nlev), np.int32(ncells),
        np.int32(nedges), np.int32(max_edges), counts, edges, signs,
        context.inv_area_cell, rdnw, horizontal_residual,
        vertical_residual, target, mass,
    )
    if element_fct:
        _launch_checked("fct_finish_elem_v841", ntracers * nlev * ncells,
                        "fct_finish_v841", ncells, finish_inputs,
                        (output,), kernel_cache)
    else:
        _launch("fct_finish_v841", ncells, (*finish_inputs, output), kernel_cache)
    if config_apply_lbcs:
        _launch("fct_copy_back_v841", ntracers * nlev * ncells,
            (np.int32(ntracers), np.int32(nlev), np.int32(ncells),
             np.int32(1), bdy_mask_cell, stage, output), kernel_cache)


def advance_scalar_transport_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    dt: float,
    *,
    coefficients: CudaAdvectionCoefficients,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    rk_step: int,
    config_scalar_advection: bool,
    config_monotonic: bool,
    config_positive_definite: bool,
    config_split_dynamics_transport: bool,
    config_time_integration_order: int,
    kernel_cache: Any | None = None,
    **kwargs: Any,
) -> CudaScalarTransportResult:
    if rk_step not in (1, 2, 3):
        raise ConfigurationRefusal(
            "rk_step", rk_step, "the v8.4.1 CUDA scalar RK has three stages",
            "rk_step=1, 2, or 3",
        )
    if config_time_integration_order not in (2, 3):
        raise ConfigurationRefusal(
            "config_time_integration_order", config_time_integration_order,
            "the v8.4.1 scalar path has three stages at order 2 or 3",
            "config_time_integration_order=2 or 3",
        )
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    cp = _cupy()
    stage = _v823._f32(scalar_stage, "scalar_stage")
    rho_new = _v823._f32(rho_zz_new, "rho_zz_new")
    if not config_scalar_advection:
        return CudaScalarTransportResult(
            cp.array(stage, copy=True), cp.array(rho_new, copy=True)
        )
    common = dict(
        coefficients=coefficients,
        fzm=fzm,
        fzp=fzp,
        rdzw=rdzw,
        advance_density=config_split_dynamics_transport,
        kernel_cache=kernel_cache,
        **kwargs,
    )
    if rk_step >= 3 and (config_monotonic or config_positive_definite):
        return advance_scalars_monotonic_cuda_v841(
            mesh, context, scalar_old, scalar_stage, rho_zz_old, rho_zz_new,
            uh_avg, ww_avg, dt, **common,
        )
    return advance_scalars_cuda_v841(
        mesh, context, scalar_old, scalar_stage, rho_zz_old, rho_zz_new,
        uh_avg, ww_avg, dt, rk_step=rk_step,
        config_time_integration_order=config_time_integration_order,
        **common,
    )


__all__ = [
    "advance_scalar_transport_cuda_v841",
    "advance_scalars_cuda_v841",
    "advance_scalars_monotonic_cuda_v841",
]
