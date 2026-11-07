"""CUDA regional (limited-area) kernels for MPAS-A v8.4.1.

Every kernel here mirrors exactly one function of the CPU authority lane
(:mod:`woof.hex.regional_v841`, plus the regional branches of
:mod:`woof.hex.acoustic_v841` and :mod:`woof.hex.transport`), which is
itself a transcription of the pinned v8.4.1 native sources.  The CPU lane is
this module's expected-bits oracle: each kernel is proved by running the host
function and the device kernel on the same inputs and comparing the raw
float32 bit patterns (``tools/run_cuda_regional_contract.py``).

**Why a separate translation unit.**  The regional acoustic and transport
branches differ from the global ones by a mask, and folding the mask into the
existing ``woof.hex.cuda_acoustic_v841`` / ``woof.hex.cuda_transport_v841``
translation units would change their source SHA-256 and therefore every
compile-manifest digest, FTZ audit count and archived proof receipt that names
them.  The regional variants therefore live here under their own module key
with their own kernel names, exactly as the v8.4.1 kernels were made additive
beside the frozen v8.2.3 ones.  The global lane's bits are untouched by
construction, not by inspection.

**The garbage element.**  Native MPAS allocates every array with one extra
element per dimension and maps absent neighbours to it at read time; a culled
regional grid stores 0 (loaded here as a negative sentinel) in the
absent-neighbour slots of ring-7 rows.  Measured on the reference x1 cull
(2,971 cells): sentinels appear in active slots of exactly five arrays --
``cellsOnEdge`` (407 rows), ``cellsOnCell`` (407 entries), ``edgesOnEdge``
(1,521 entries), ``edgesOnVertex`` (206 rows) and ``cellsOnVertex`` (407
rows) -- all of them ring-7 rows.  Kernels here take the sentinel explicitly
and reproduce the native garbage-element value rather than clamping to a real
index, because a clamp changes a sum (``x + garbage`` becomes ``x + x``) and
that difference reaches a live relaxation edge through the fourth-order
filters.

**Element-parallel launch geometry (2026-09-14).**  Every kernel whose
levels are independent -- all of them but the vertically implicit column
solve -- indexes a flat element ``(tracer, level, owner)`` with the owner
fastest, through the ``REGIONAL_ELEMENT_LOOP`` grid-stride macro.  The body
each element runs is the former per-owner loop body for that level, with
its edge-slot accumulation in the same ascending order, so the value written
for an element does not depend on which thread computed it or how many
threads the launch had.  The forecast launches one thread per element (the
relaxation-zone filters go from 10 and 28 blocks to hundreds, the
edge-parallel kernels from under one wave of the card to many); the
contract deck keeps launching one thread per owner and its bits are the
proof that the geometry is not in the arithmetic.  Measured on the point
cull (``docs/hex-point-hrrr.md``, the performance section).

**No literal divisors.**  MEASURED on this stack (RTX 5070 Ti, NVRTC through
CuPy 14.2.0, the port's own ``--std=c++17 --fmad=false`` options): NVRTC
rewrites ``x / <float literal>`` into ``x * (1/<literal>)``, which is a
different result whenever the reciprocal is inexact.  ``mpas_div(x, 5.0f)``
returned ``0x3B83126F`` where the correctly-rounded float32 quotient -- and
the CPU authority -- is ``0x3B83126E``; ``mpas_div(x, five)`` with ``five``
a kernel argument returns the correctly-rounded value, and so does
``__fdiv_rn`` either way.  The hardwired relaxation-zone denominator
``nRelaxZone`` is therefore passed as a runtime ``const float`` argument
rather than written into the source, and this module holds no
literal-divisor division.  The contract deck caught this as a one-ulp
divergence in four kernels at once; the same hazard exists anywhere in the
port a literal divisor is written, which is a finding for the shared
translation units rather than a change this lane makes to them.
"""

from __future__ import annotations

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import hashlib
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_backend.containers import TransferStats, require_resident_array
from .errors import ConfigurationRefusal
from .lbc import LbcAdmissionError, LbcFile, LbcInventory, LbcPool
from .regional_v841 import (
    N_RELAX_ZONE,
    N_SPEC_ZONE,
    RVORD_F32,
    RegionalMasks,
    compute_mesh_scaling_regional,
    derive_regional_masks,
    regional_bdy_checks,
    rkind_interval_seconds,
    scalar_index_for_field,
)


#: NVRTC module key for this translation unit.
MODULE_KEY = "hexcore.cuda_regional_v841"

# Direct NVRTC avoids CuPy RawModule's appended FTZ override. Native RKIND
# arithmetic retains subnormals; the interpolation regression measures them
# against mpas_atm_boundaries.F:631-642.
REGIONAL_CUDA_OPTIONS = (
    "--std=c++17", "--fmad=false", "--ftz=false",
    "--prec-div=true", "--prec-sqrt=true",
)


_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
#define RC2(k,c,nc) ((k)*(nc) + (c))
#define RE2(k,e,ne) ((k)*(ne) + (e))
#define RQ3(t,k,c,nlev,nc) (((t)*(nlev) + (k))*(nc) + (c))
#define RQE3(t,k,e,nlev,ne) (((t)*(nlev) + (k))*(ne) + (e))
#define RQI3(t,k,c,nlev,nc) (((t)*((nlev)+1) + (k))*(nc) + (c))
#define RCES(c,s,me) ((c)*(me) + (s))
#define RADV(e,s,w) ((e)*(w) + (s))

/* Every level-independent kernel below walks a flat element index with the
   owner (cell, edge or zone slot) fastest and the level, then the tracer,
   above it, in a grid-stride loop: any grid size computes every element, so
   the contract deck's one-thread-per-owner launch and the forecast's
   one-thread-per-element launch run the same per-element arithmetic. */
#define REGIONAL_ELEMENT_LOOP(total) \
    for (int element = blockDim.x * blockIdx.x + threadIdx.x; \
         element < (total); element += gridDim.x * blockDim.x)

/* ------------------------------------------------------------------ */
/* the lateral-boundary pool: derivation, tendency, time interpolation  */
/* ------------------------------------------------------------------ */

/* mpas_atm_update_bdy_tend, mpas_atm_boundaries.F:217-262, transcribed by
   RegionalDrivingState._derive.  lbc_rho_zz = lbc_rho/zz; lbc_rho_edge is
   the half-sum of the two adjacent cells' rho_zz and is written ONLY where
   both native indices are positive. mpas_block_creator.F:1490-1496 maps
   missing neighbors to the zero-valued garbage cell; lbc_ru =
   lbc_u*lbc_rho_edge; lbc_rtheta_m = lbc_theta*lbc_rho_zz*(1 + rvord*qv),
   left to right in float32. */
extern "C" __global__ void regional_lbc_derive_v841(
    const int nvalues, const float rvord, const float *zz,
    const float *lbc_rho, const float *lbc_theta, const float *lbc_qv,
    float *rho_zz, float *rtheta_m)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nvalues) return;
    const float r = mpas_div(lbc_rho[i], zz[i]);
    rho_zz[i] = r;
    rtheta_m[i] = mpas_mul(mpas_mul(lbc_theta[i], r),
        mpas_add(1.0f, mpas_mul(rvord, lbc_qv[i])));
}

/* Second half of the derivation: the edge average and lbc_ru.  Split from
   the cell half because rho_zz must be complete before any edge reads it. */
extern "C" __global__ void regional_lbc_rho_edge_v841(
    const int nlev, const int ncells, const int nedges,
    const int *cells_on_edge, const float *rho_zz, const float *lbc_u,
    float *rho_edge, float *ru)
{
    REGIONAL_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const int present = (c0 >= 0 && c1 >= 0);
        const int i = RE2(k, edge, nedges);
        float value = rho_edge[i];
        if (present) {
            const float r0 = c0 < ncells ? rho_zz[RC2(k, c0, ncells)] : 0.0f;
            const float r1 = c1 < ncells ? rho_zz[RC2(k, c1, ncells)] : 0.0f;
            value = mpas_mul(0.5f, mpas_add(r0, r1));
            rho_edge[i] = value;
        }
        ru[i] = mpas_mul(lbc_u[i], value);
    }
}

/* mpas_atm_boundaries.F:265-309: every boundary tendency is
   (new - old) * (1/interval), with the reciprocal formed once on the host
   exactly as the native REAL(RKIND) division does. */
extern "C" __global__ void regional_lbc_tendency_v841(
    const int nvalues, const float inv_dt,
    const float *new_state, const float *old_state, float *tend)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nvalues) return;
    tend[index] = mpas_mul(
        mpas_sub(new_state[index], old_state[index]), inv_dt);
}

/* mpas_atm_get_bdy_state (mpas_atm_boundaries.F:491-551): the driving state
   at an in-step offset is state(interval end) - dt_remaining*tendency, with
   dt_remaining formed on the host in float32 as
   float32(seconds(end - step_start)) - delta_t. */
extern "C" __global__ void regional_lbc_state_at_v841(
    const int nvalues, const float dt_remaining,
    const float *state, const float *tend, float *out)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nvalues) return;
    out[index] = mpas_sub(state[index], mpas_mul(dt_remaining, tend[index]));
}

/* ------------------------------------------------------------------ */
/* atm_bdy_adjust_dynamics_speczone_tend (F:7906-7967)                  */
/* ------------------------------------------------------------------ */

extern "C" __global__ void regional_speczone_tend_cell_v841(
    const int nlev, const int ncells, const int nspec,
    const int *spec_cells, const float *rho_driving_tend,
    const float *rt_driving_tend,
    float *tend_rho, float *tend_rt, float *tend_rw)
{
    REGIONAL_ELEMENT_LOOP(nlev * nspec) {
        const int slot = element % nspec;
        const int k = element / nspec;
        const int cell = spec_cells[slot];
        const int i = RC2(k, cell, ncells);
        tend_rho[i] = rho_driving_tend[i];
        tend_rt[i] = rt_driving_tend[i];
        /* F:7948 zeroes k=1..nVertLevels of the omega tendency; the top
           interface row (index nlev) is untouched. */
        tend_rw[i] = 0.0f;
    }
}

extern "C" __global__ void regional_speczone_tend_edge_v841(
    const int nlev, const int nedges, const int nspec,
    const int *spec_edges, const float *ru_driving_tend, float *tend_ru)
{
    REGIONAL_ELEMENT_LOOP(nlev * nspec) {
        const int slot = element % nspec;
        const int k = element / nspec;
        const int edge = spec_edges[slot];
        const int i = RE2(k, edge, nedges);
        tend_ru[i] = ru_driving_tend[i];
    }
}

/* ------------------------------------------------------------------ */
/* atm_bdy_adjust_dynamics_relaxzone_tend (F:7971-8198)                 */
/* ------------------------------------------------------------------ */

/* F:8054-8065 and F:8067-8076: the Rayleigh half.  The coefficient is
   ((bdyMask - 1)/nRelaxZone) / (50*dt*meshScalingRegional), grouped exactly
   as the CPU authority groups it (a/b/c is (a/b)/c). */
extern "C" __global__ void regional_relaxzone_rayleigh_cell_v841(
    const int nlev, const int ncells, const int nrelax, const float fifty_dt,
    const float relax_zone,
    const int *relax_cells, const int *bdy_mask_cell,
    const float *mesh_scaling_cell, const float *rho_zz, const float *theta_m,
    const float *rho_driving, const float *rt_driving,
    float *tend_rho, float *tend_rt)
{
    REGIONAL_ELEMENT_LOOP(nlev * nrelax) {
        const int slot = element % nrelax;
        const int k = element / nrelax;
        const int cell = relax_cells[slot];
        const float mask = (float)bdy_mask_cell[cell];
        const float coef = mpas_div(
            mpas_div(mpas_sub(mask, 1.0f), relax_zone),
            mpas_mul(fifty_dt, mesh_scaling_cell[cell]));
        const int i = RC2(k, cell, ncells);
        tend_rho[i] = mpas_sub(tend_rho[i], mpas_mul(
            coef, mpas_sub(rho_zz[i], rho_driving[i])));
        tend_rt[i] = mpas_sub(tend_rt[i], mpas_mul(
            coef, mpas_sub(mpas_mul(rho_zz[i], theta_m[i]), rt_driving[i])));
    }
}

extern "C" __global__ void regional_relaxzone_rayleigh_edge_v841(
    const int nlev, const int nedges, const int nrelax, const float fifty_dt,
    const float relax_zone,
    const int *relax_edges, const int *bdy_mask_edge,
    const float *mesh_scaling_edge, const float *ru,
    const float *ru_driving, float *tend_ru)
{
    REGIONAL_ELEMENT_LOOP(nlev * nrelax) {
        const int slot = element % nrelax;
        const int k = element / nrelax;
        const int edge = relax_edges[slot];
        const float mask = (float)bdy_mask_edge[edge];
        const float coef = mpas_div(
            mpas_div(mpas_sub(mask, 1.0f), relax_zone),
            mpas_mul(fifty_dt, mesh_scaling_edge[edge]));
        const int i = RE2(k, edge, nedges);
        tend_ru[i] = mpas_sub(tend_ru[i], mpas_mul(
            coef, mpas_sub(ru[i], ru_driving[i])));
    }
}

/* F:8080-8109: the dimensionless horizontal Laplacian filter on the
   deviation from the driving state, accumulated edge by edge in ascending
   slot order (the native loop order). */
extern "C" __global__ void regional_relaxzone_filter_cell_v841(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int nrelax, const float ten_dt, const float relax_zone,
    const int *relax_cells, const int *bdy_mask_cell,
    const float *mesh_scaling_cell, const int *n_edges_on_cell,
    const int *edges_on_cell, const int *cells_on_edge,
    const float *edge_sign_on_cell, const float *dv_edge,
    const float *inv_dc_edge, const float *rho_zz, const float *theta_m,
    const float *rho_driving, const float *rt_driving,
    float *tend_rho, float *tend_rt)
{
    REGIONAL_ELEMENT_LOOP(nlev * nrelax) {
        const int slot = element % nrelax;
        const int k = element / nrelax;
        const int cell = relax_cells[slot];
        const float mask = (float)bdy_mask_cell[cell];
        const float filter_coef = mpas_div(
            mpas_div(mpas_sub(mask, 1.0f), relax_zone),
            mpas_mul(ten_dt, mesh_scaling_cell[cell]));
        const int count = n_edges_on_cell[cell];
        const int i = RC2(k, cell, ncells);
        float rt = tend_rt[i];
        float rho = tend_rho[i];
        for (int s = 0; s < count; ++s) {
            const int edge = edges_on_cell[RCES(cell, s, max_edges)];
            float edge_sign = mpas_mul(
                edge_sign_on_cell[RCES(cell, s, max_edges)], dv_edge[edge]);
            edge_sign = mpas_mul(edge_sign, inv_dc_edge[edge]);
            edge_sign = mpas_mul(edge_sign, filter_coef);
            const int cell1 = cells_on_edge[2 * edge];
            const int cell2 = cells_on_edge[2 * edge + 1];
            /* A relaxation cell (mask 2..5) has every neighbour present; a
               sentinel here would mean the mask derivation disagrees with
               the connectivity, so the lane refuses through the caller's
               admission rather than reading out of bounds. */
            const int i1 = RC2(k, cell1, ncells);
            const int i2 = RC2(k, cell2, ncells);
            rt = mpas_add(rt, mpas_mul(edge_sign, mpas_sub(
                mpas_sub(mpas_mul(rho_zz[i2], theta_m[i2]), rt_driving[i2]),
                mpas_sub(mpas_mul(rho_zz[i1], theta_m[i1]), rt_driving[i1]))));
            rho = mpas_add(rho, mpas_mul(edge_sign, mpas_sub(
                mpas_sub(rho_zz[i2], rho_driving[i2]),
                mpas_sub(rho_zz[i1], rho_driving[i1]))));
        }
        tend_rt[i] = rt;
        tend_rho[i] = rho;
    }
}

/* F:8111-8194: the u filter.  filter_coef carries dcEdge**2 -- the Fortran
   integer exponent, i.e. the exact product, not a libm pow -- and the
   divergence half is scaled by config_relax_zone_divdamp_coef while the
   vorticity half uses r_dv = min(invDvEdge, 4*invDcEdge). */
extern "C" __global__ void regional_relaxzone_filter_edge_v841(
    const int nlev, const int ncells, const int nedges, const int nvertices,
    const int max_edges, const int vertex_degree, const int nrelax,
    const float ten_dt, const float divdamp, const float relax_zone,
    const int *relax_edges, const int *bdy_mask_edge,
    const float *mesh_scaling_edge, const int *n_edges_on_cell,
    const int *edges_on_cell, const int *cells_on_edge,
    const int *vertices_on_edge, const int *edges_on_vertex,
    const float *edge_sign_on_cell, const float *edge_sign_on_vertex,
    const float *dc_edge, const float *dv_edge, const float *inv_dc_edge,
    const float *inv_dv_edge, const float *inv_area_cell,
    const float *inv_area_triangle, const float *ru,
    const float *ru_driving, float *tend_ru)
{
    REGIONAL_ELEMENT_LOOP(nlev * nrelax) {
        const int slot = element % nrelax;
        const int k = element / nrelax;
        const int edge = relax_edges[slot];
        const float mask = (float)bdy_mask_edge[edge];
        const float dc = dc_edge[edge];
        const float dc_squared = mpas_mul(dc, dc);
        const float filter_coef = mpas_div(
            mpas_div(mpas_mul(dc_squared, mpas_sub(mask, 1.0f)),
                     relax_zone),
            mpas_mul(ten_dt, mesh_scaling_edge[edge]));
        const int cell1 = cells_on_edge[2 * edge];
        const int cell2 = cells_on_edge[2 * edge + 1];
        const int vertex1 = vertices_on_edge[2 * edge];
        const int vertex2 = vertices_on_edge[2 * edge + 1];
        const float r_dc = inv_dc_edge[edge];
        const float r_dv = mpas_min(inv_dv_edge[edge],
            mpas_mul(4.0f, inv_dc_edge[edge]));
        const int count1 = n_edges_on_cell[cell1];
        const int count2 = n_edges_on_cell[cell2];
        const float inv_area1 = inv_area_cell[cell1];
        const float inv_area2 = inv_area_cell[cell2];
        float divergence1 = 0.0f;
        float divergence2 = 0.0f;
        float vorticity1 = 0.0f;
        float vorticity2 = 0.0f;
        for (int s = 0; s < count1; ++s) {
            const int ed = edges_on_cell[RCES(cell1, s, max_edges)];
            float sign = mpas_mul(inv_area1, dv_edge[ed]);
            sign = mpas_mul(sign, edge_sign_on_cell[RCES(cell1, s, max_edges)]);
            const int j = RE2(k, ed, nedges);
            divergence1 = mpas_add(divergence1, mpas_mul(
                sign, mpas_sub(ru[j], ru_driving[j])));
        }
        for (int s = 0; s < count2; ++s) {
            const int ed = edges_on_cell[RCES(cell2, s, max_edges)];
            float sign = mpas_mul(inv_area2, dv_edge[ed]);
            sign = mpas_mul(sign, edge_sign_on_cell[RCES(cell2, s, max_edges)]);
            const int j = RE2(k, ed, nedges);
            divergence2 = mpas_add(divergence2, mpas_mul(
                sign, mpas_sub(ru[j], ru_driving[j])));
        }
        for (int s = 0; s < vertex_degree; ++s) {
            const int ed = edges_on_vertex[vertex1 * vertex_degree + s];
            float sign = mpas_mul(inv_area_triangle[vertex1], dc_edge[ed]);
            sign = mpas_mul(sign,
                edge_sign_on_vertex[vertex1 * vertex_degree + s]);
            const int j = RE2(k, ed, nedges);
            vorticity1 = mpas_add(vorticity1, mpas_mul(
                sign, mpas_sub(ru[j], ru_driving[j])));
        }
        for (int s = 0; s < vertex_degree; ++s) {
            const int ed = edges_on_vertex[vertex2 * vertex_degree + s];
            float sign = mpas_mul(inv_area_triangle[vertex2], dc_edge[ed]);
            sign = mpas_mul(sign,
                edge_sign_on_vertex[vertex2 * vertex_degree + s]);
            const int j = RE2(k, ed, nedges);
            vorticity2 = mpas_add(vorticity2, mpas_mul(
                sign, mpas_sub(ru[j], ru_driving[j])));
        }
        const int i = RE2(k, edge, nedges);
        const float divergence_part = mpas_mul(
            mpas_mul(divdamp, mpas_sub(divergence2, divergence1)), r_dc);
        const float vorticity_part = mpas_mul(
            mpas_sub(vorticity2, vorticity1), r_dv);
        tend_ru[i] = mpas_add(tend_ru[i], mpas_mul(
            filter_coef, mpas_sub(divergence_part, vorticity_part)));
    }
}

/* ------------------------------------------------------------------ */
/* the specified-zone overwrites                                        */
/* ------------------------------------------------------------------ */

/* atm_srk3:2442-2485 -- recover "will not have set outermost edge velocities
   correctly", so u and ru take the interpolated driving states. */
extern "C" __global__ void regional_speczone_u_ru_v841(
    const int nlev, const int nedges, const int nspec, const int *spec_edges,
    const float *u_driving, const float *ru_driving,
    float *normal_velocity, float *rho_u)
{
    REGIONAL_ELEMENT_LOOP(nlev * nspec) {
        const int slot = element % nspec;
        const int k = element / nspec;
        const int edge = spec_edges[slot];
        const int i = RE2(k, edge, nedges);
        normal_velocity[i] = u_driving[i];
        rho_u[i] = ru_driving[i];
    }
}

/* atm_zero_gradient_w_bdy_work (F:7868-7902) plus its context: the whole
   specified-zone w column, endpoints included, is identically zero. */
extern "C" __global__ void regional_zero_speczone_w_v841(
    const int nrows, const int ncells, const int nspec,
    const int *spec_cells, float *w)
{
    REGIONAL_ELEMENT_LOOP(nrows * nspec) {
        const int slot = element % nspec;
        const int k = element / nspec;
        const int cell = spec_cells[slot];
        w[RC2(k, cell, ncells)] = 0.0f;
    }
}

/* atm_bdy_reset_speczone_values (F:8201-8244) plus the F:8238 perturbation
   assignment the driver carries: theta_m := rt_driving/rho_driving,
   rho_theta := rt_driving, rtheta_p := rt_driving - rtheta_base. */
extern "C" __global__ void regional_reset_speczone_values_v841(
    const int nlev, const int ncells, const int nspec, const int *spec_cells,
    const float *rt_driving, const float *rho_driving,
    const float *rho_theta_base,
    float *theta_m, float *rho_theta, float *rho_theta_perturbation)
{
    REGIONAL_ELEMENT_LOOP(nlev * nspec) {
        const int slot = element % nspec;
        const int k = element / nspec;
        const int cell = spec_cells[slot];
        const int i = RC2(k, cell, ncells);
        theta_m[i] = mpas_div(rt_driving[i], rho_driving[i]);
        rho_theta[i] = rt_driving[i];
        rho_theta_perturbation[i] = mpas_sub(
            rt_driving[i], rho_theta_base[i]);
    }
}

/* ------------------------------------------------------------------ */
/* the scalar boundary stages                                           */
/* ------------------------------------------------------------------ */

/* atm_bdy_adjust_scalars_work (F:8305-8416), phase one: the relaxation-zone
   filter and Rayleigh terms and the specified-zone assignment both land in
   temporary storage. */
extern "C" __global__ void regional_bdy_adjust_scalars_compute_v841(
    const int ntracers, const int nlev, const int ncells, const int nedges,
    const int max_edges, const int nrelax, const int nspec,
    const float ten_dt, const float dt_rk, const float relax_zone,
    const int *relax_cells, const int *spec_cells, const int *bdy_mask_cell,
    const float *mesh_scaling_cell, const int *n_edges_on_cell,
    const int *edges_on_cell, const int *cells_on_edge,
    const float *edge_sign_on_cell, const float *dv_edge,
    const float *inv_dc_edge, const float *scalars_new,
    const float *scalars_driving, float *updates)
{
    const int nowners = nrelax + nspec;
    REGIONAL_ELEMENT_LOOP(ntracers * nlev * nowners) {
        const int slot = element % nowners;
        const int rest = element / nowners;
        const int k = rest % nlev;
        const int t = rest / nlev;
        if (slot < nrelax) {
            const int cell = relax_cells[slot];
            const float mask = (float)bdy_mask_cell[cell];
            const float filter_coef = mpas_div(
                mpas_div(mpas_mul(dt_rk, mpas_sub(mask, 1.0f)),
                         relax_zone),
                mpas_mul(ten_dt, mesh_scaling_cell[cell]));
            const float rayleigh_coef = mpas_div(filter_coef, relax_zone);
            const int count = n_edges_on_cell[cell];
            const int i = RQ3(t, k, cell, nlev, ncells);
            float column = scalars_new[i];
            for (int s = 0; s < count; ++s) {
                const int edge = edges_on_cell[RCES(cell, s, max_edges)];
                float edge_sign = mpas_mul(
                    edge_sign_on_cell[RCES(cell, s, max_edges)],
                    dv_edge[edge]);
                edge_sign = mpas_mul(edge_sign, inv_dc_edge[edge]);
                edge_sign = mpas_mul(edge_sign, filter_coef);
                const int cell1 = cells_on_edge[2 * edge];
                const int cell2 = cells_on_edge[2 * edge + 1];
                const int i1 = RQ3(t, k, cell1, nlev, ncells);
                const int i2 = RQ3(t, k, cell2, nlev, ncells);
                column = mpas_add(column, mpas_mul(edge_sign, mpas_sub(
                    mpas_sub(scalars_new[i2], scalars_driving[i2]),
                    mpas_sub(scalars_new[i1], scalars_driving[i1]))));
            }
            column = mpas_sub(column, mpas_mul(rayleigh_coef,
                mpas_sub(scalars_new[i], scalars_driving[i])));
            updates[i] = column;
        } else {
            const int cell = spec_cells[slot - nrelax];
            const int i = RQ3(t, k, cell, nlev, ncells);
            updates[i] = scalars_driving[i];
        }
    }
}

/* Phase two, F:8399-8412: the copy-back admits every cell with
   bdyMaskCell > 1 -- ring 1 is never nudged. */
extern "C" __global__ void regional_bdy_adjust_scalars_copyback_v841(
    const int ntracers, const int nlev, const int ncells, const int nnudged,
    const int *nudged_cells, const float *updates, float *scalars_new)
{
    REGIONAL_ELEMENT_LOOP(ntracers * nlev * nnudged) {
        const int slot = element % nnudged;
        const int rest = element / nnudged;
        const int k = rest % nlev;
        const int t = rest / nlev;
        const int cell = nudged_cells[slot];
        const int i = RQ3(t, k, cell, nlev, ncells);
        scalars_new[i] = updates[i];
    }
}

/* atm_bdy_set_scalars_work (F:8462-8505): the specified zone takes the
   driving values outright. */
extern "C" __global__ void regional_bdy_set_scalars_v841(
    const int ntracers, const int nlev, const int ncells, const int nspec,
    const int *spec_cells, const float *scalars_driving, float *scalars_new)
{
    REGIONAL_ELEMENT_LOOP(ntracers * nlev * nspec) {
        const int slot = element % nspec;
        const int rest = element / nspec;
        const int k = rest % nlev;
        const int t = rest / nlev;
        const int cell = spec_cells[slot];
        const int i = RQ3(t, k, cell, nlev, ncells);
        scalars_new[i] = scalars_driving[i];
    }
}

/* atm_srk3:2798-2800: the unconditional end-of-step clamp every DO_PHYSICS
   build performs regardless of the physics suite. */
extern "C" __global__ void regional_clamp_negative_scalars_v841(
    const int nvalues, float *scalars)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nvalues) return;
    // mpas_atm_time_integration.F:2799-2800: WHERE preserves -0 and NaNs.
    // Decode through the shared IEEE helper so negative subnormals also
    // compare correctly on the certified FTZ CUDA route.
    if (mpas_f32_to_f64_ieee(scalars[index]) < 0.0) scalars[index] = 0.0f;
}

/* ------------------------------------------------------------------ */
/* the regional acoustic branches                                       */
/* ------------------------------------------------------------------ */

/* atm_advance_acoustic_step_work F:3909: ru_p += dts*(tend_ru -
   (1 - specZoneMaskEdge)*pgrad).  A ring-7 one-cell edge has no second cell
   at all; native gathers the garbage element and multiplies the result by
   exactly zero, so the update is the driving tendency alone.  ``invalid`` is
   raised if such an edge is NOT in the specified zone, which is the device
   form of the CPU lane's named refusal. */
extern "C" __global__ void acoustic_ru_regional_v841(
    const int nlev, const int nedges, const int ncells,
    const int small_step, const float dts,
    const float gravity, const float rgas, const float cp,
    const int *cells_on_edge, const float *inv_dc_edge,
    const float *spec_zone_mask_edge,
    const float *zz, const float *exner, const float *cqu,
    const float *zxu, const float *tend_ru,
    const float *rho_pp, const float *rtheta_pp,
    float *ru_p, float *ru_avg, int *invalid)
{
    REGIONAL_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const float half = 0.5f;
        const float one = 1.0f;
        const float rcv = mpas_div(rgas, mpas_sub(cp, rgas));
        const float csquared = mpas_mul(cp, rcv);
        const float spec = spec_zone_mask_edge[edge];
        const int index = RE2(k, edge, nedges);
        if (small_step == 1) {
            const float value = mpas_mul(dts, tend_ru[index]);
            ru_p[index] = value;
            ru_avg[index] = value;
            continue;
        }
        if (c0 < 0 || c1 < 0) {
            if (spec != one) atomicExch(invalid, 1);
            ru_p[index] = mpas_add(ru_p[index],
                mpas_mul(dts, tend_ru[index]));
            ru_avg[index] = mpas_add(ru_avg[index], ru_p[index]);
            continue;
        }
        const float normal = mpas_mul(mpas_sub(
            rtheta_pp[RC2(k, c1, ncells)],
            rtheta_pp[RC2(k, c0, ncells)]), inv_dc_edge[edge]);
        const float normalized = mpas_div(normal, mpas_mul(half, mpas_add(
            zz[RC2(k, c1, ncells)], zz[RC2(k, c0, ncells)])));
        float pgrad = mpas_mul(cqu[index], half);
        pgrad = mpas_mul(pgrad, csquared);
        pgrad = mpas_mul(pgrad, mpas_add(exner[RC2(k, c0, ncells)],
            exner[RC2(k, c1, ncells)]));
        pgrad = mpas_mul(pgrad, normalized);
        const float terrain = mpas_mul(mpas_mul(mpas_mul(
            half, zxu[index]), gravity), mpas_add(
                rho_pp[RC2(k, c0, ncells)],
                rho_pp[RC2(k, c1, ncells)]));
        pgrad = mpas_add(pgrad, terrain);
        ru_p[index] = mpas_add(ru_p[index], mpas_mul(dts, mpas_sub(
            tend_ru[index], mpas_mul(mpas_sub(one, spec), pgrad))));
        ru_avg[index] = mpas_add(ru_avg[index], ru_p[index]);
    }
}

/* F:4093-4103: a specified-zone cell forms no rs/ts fluxes.  The scratch is
   written zero rather than left uninitialized so two runs of the same
   configuration produce the same bytes in every buffer, read or not. */
extern "C" __global__ void acoustic_rs_ts_regional_v841(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const float dts, const int *n_edges_on_cell, const int *edges_on_cell,
    const int *cells_on_edge, const float *edge_sign_on_cell,
    const float *dv_edge, const float *inv_area_cell,
    const float *spec_zone_mask_cell,
    const float *theta_m, const float *rdzw,
    const float *cofrz, const float *coftz, const float *ewm,
    const float *ru_p, const float *rw_p,
    const float *rho_pp, const float *rtheta_pp,
    const float *tend_rho, const float *tend_rt,
    float *rs, float *ts)
{
    REGIONAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = RC2(k, cell, ncells);
        if (spec_zone_mask_cell[cell] != 0.0f) {
            rs[index] = 0.0f;
            ts[index] = 0.0f;
            continue;
        }
        const int count = n_edges_on_cell[cell];
        float r = 0.0f;
        float t = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[RCES(cell, slot, max_edges)];
            const int c0 = cells_on_edge[2 * edge];
            const int c1 = cells_on_edge[2 * edge + 1];
            float flux = mpas_mul(
                edge_sign_on_cell[RCES(cell, slot, max_edges)], dts);
            flux = mpas_mul(flux, dv_edge[edge]);
            flux = mpas_mul(flux, ru_p[RE2(k, edge, nedges)]);
            flux = mpas_mul(flux, inv_area_cell[cell]);
            r = mpas_sub(r, flux);
            t = mpas_sub(t, mpas_mul(mpas_mul(flux, 0.5f), mpas_add(
                theta_m[RC2(k, c1, ncells)],
                theta_m[RC2(k, c0, ncells)])));
        }
        r = mpas_add(mpas_add(rho_pp[index], mpas_mul(dts, tend_rho[index])),
            r);
        r = mpas_sub(r, mpas_mul(mpas_mul(dts, cofrz[k]), mpas_sub(
            mpas_mul(ewm[k + 1], rw_p[RC2(k + 1, cell, ncells)]),
            mpas_mul(ewm[k], rw_p[index]))));
        t = mpas_add(mpas_add(rtheta_pp[index], mpas_mul(dts, tend_rt[index])),
            t);
        t = mpas_sub(t, mpas_mul(mpas_mul(dts, rdzw[k]), mpas_sub(
            mpas_mul(mpas_mul(ewm[k + 1], coftz[RC2(k + 1, cell, ncells)]),
                rw_p[RC2(k + 1, cell, ncells)]),
            mpas_mul(mpas_mul(ewm[k], coftz[index]), rw_p[index]))));
        rs[index] = r;
        ts[index] = t;
    }
}

/* F:4093-4103 again, the other half: no vertically implicit solve in the
   specified zone.  The driving tendencies integrate forward over rows
   k=1..nVertLevels only, and wwAvg accumulates ewp-weighted rw_p over the
   same rows -- including row 1, unlike the interior branch. */
extern "C" __global__ void acoustic_column_solve_regional_v841(
    const int nlev, const int ncells, const float dts,
    const float *zz, const float *rho_zz,
    const float *fzm, const float *fzp, const float *rdzw,
    const float *dss, const float *w, const float *rw, const float *rw_save,
    const float *tend_rw, const float *tend_rho, const float *tend_rt,
    const float *rs, const float *ts,
    const float *spec_zone_mask_cell,
    const float *cofwr, const float *cofwz, const float *coftz,
    const float *cofwt, const float *cofrz,
    const float *a_tri, const float *alpha_tri, const float *gamma_tri,
    const float *etp, const float *etm, const float *ewp, const float *ewm,
    float *rw_p, float *rho_pp, float *rtheta_pp, float *ww_avg)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    if (spec_zone_mask_cell[cell] != 0.0f) {
        for (int k = 0; k < nlev; ++k) {
            const int i = RC2(k, cell, ncells);
            rho_pp[i] = mpas_add(rho_pp[i], mpas_mul(dts, tend_rho[i]));
            rtheta_pp[i] = mpas_add(rtheta_pp[i], mpas_mul(dts, tend_rt[i]));
            rw_p[i] = mpas_add(rw_p[i], mpas_mul(dts, tend_rw[i]));
            ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(ewp[k], rw_p[i]));
        }
        return;
    }
    const float dts2 = mpas_mul(dts, dts);
    for (int k = 1; k < nlev; ++k) {
        const int i = RC2(k, cell, ncells);
        const int im = RC2(k - 1, cell, ncells);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(ewm[k], rw_p[i]));

        const float theta_implicit = mpas_sub(
            mpas_mul(mpas_mul(etp[k], zz[i]), ts[i]),
            mpas_mul(mpas_mul(etp[k - 1], zz[im]), ts[im]));
        const float theta_explicit = mpas_sub(
            mpas_mul(mpas_mul(etm[k], zz[i]), rtheta_pp[i]),
            mpas_mul(mpas_mul(etm[k - 1], zz[im]), rtheta_pp[im]));
        const float density_implicit = mpas_add(
            mpas_mul(etp[k], rs[i]), mpas_mul(etp[k - 1], rs[im]));
        const float density_explicit = mpas_add(
            mpas_mul(etm[k], rho_pp[i]),
            mpas_mul(etm[k - 1], rho_pp[im]));
        const float heat_upper = mpas_add(
            mpas_mul(etp[k], ts[i]), mpas_mul(etm[k], rtheta_pp[i]));
        const float heat_lower = mpas_add(
            mpas_mul(etp[k - 1], ts[im]),
            mpas_mul(etm[k - 1], rtheta_pp[im]));

        float inner = mpas_sub(tend_rw[i], mpas_mul(cofwz[i],
            mpas_add(theta_implicit, theta_explicit)));
        inner = mpas_sub(inner, mpas_mul(cofwr[i],
            mpas_add(density_implicit, density_explicit)));
        inner = mpas_add(inner, mpas_mul(cofwt[i], heat_upper));
        inner = mpas_add(inner, mpas_mul(cofwt[im], heat_lower));
        rw_p[i] = mpas_add(rw_p[i], mpas_mul(dts, inner));
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = RC2(k, cell, ncells);
        const int im = RC2(k - 1, cell, ncells);
        rw_p[i] = mpas_mul(mpas_sub(rw_p[i], mpas_mul(mpas_mul(
            dts2, a_tri[i]), rw_p[im])), alpha_tri[i]);
    }
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = RC2(k, cell, ncells);
        rw_p[i] = mpas_sub(rw_p[i], mpas_mul(
            gamma_tri[i], rw_p[RC2(k + 1, cell, ncells)]));
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = RC2(k, cell, ncells);
        const int im = RC2(k - 1, cell, ncells);
        const float delta_saved = mpas_sub(rw_save[i], rw[i]);
        /* MPAS v8.4.2 mpas_atm_time_integration.F:4059-4063:
           the damping product is evaluated from left to right. */
        const float zz_interface = mpas_add(
            mpas_mul(fzm[k], zz[i]), mpas_mul(fzp[k], zz[im]));
        const float rho_interface = mpas_add(
            mpas_mul(fzm[k], rho_zz[i]), mpas_mul(fzp[k], rho_zz[im]));
        float value = mpas_add(rw_p[i], delta_saved);
        value = mpas_sub(value, mpas_mul(mpas_mul(mpas_mul(mpas_mul(
            dts, dss[i]), zz_interface), rho_interface), w[i]));
        value = mpas_div(value, mpas_add(1.0f, mpas_mul(dts, dss[i])));
        rw_p[i] = mpas_sub(value, delta_saved);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(ewp[k], rw_p[i]));
    }
    for (int k = 0; k < nlev; ++k) {
        const int i = RC2(k, cell, ncells);
        rho_pp[i] = mpas_sub(rs[i], mpas_mul(mpas_mul(dts, cofrz[k]),
            mpas_sub(mpas_mul(ewp[k + 1], rw_p[RC2(k + 1, cell, ncells)]),
                mpas_mul(ewp[k], rw_p[i]))));
        rtheta_pp[i] = mpas_sub(ts[i], mpas_mul(mpas_mul(dts, rdzw[k]),
            mpas_sub(mpas_mul(mpas_mul(ewp[k + 1],
                    coftz[RC2(k + 1, cell, ncells)]),
                rw_p[RC2(k + 1, cell, ncells)]),
                mpas_mul(mpas_mul(ewp[k], coftz[i]), rw_p[i]))));
    }
}

/* acoustic_column_solve_regional_v841, split at its level-independent loops:
   the shape acoustic_column_{pre,sweep,post_w,post_rt}_v841 gave the global
   substep.  F:4093-4103 order is kept element by element.
   - pre: the specified-zone branch reads and writes only its own
     (level, cell), so it runs whole here, one thread per element; the
     interior's first loop (k = 1..nlev-1) writes rw_p and ww_avg at its own
     level and reads rho_pp/rtheta_pp, which nothing in this pass writes for
     an interior column.
   - sweep: the two tridiagonal recurrences, serial per column in native
     order.  The value just stored is carried in a register instead of being
     reloaded; the loads the recurrence does not depend on are issued four
     levels ahead.
   - post_w: the damping loop, one thread per (level, cell), k = 1..nlev-1.
   - post_rt: the rho/theta update, which reads rw_p at k and k+1, after
     post_w has finished every level.
   Every value is the same helper sequence on the same operands, so the four
   kernels write the bits the one kernel writes. */
extern "C" __global__ void acoustic_column_pre_regional_v841(
    const int nlev, const int ncells, const float dts,
    const float *zz, const float *tend_rw, const float *tend_rho,
    const float *tend_rt, const float *rs, const float *ts,
    const float *spec_zone_mask_cell,
    const float *cofwr, const float *cofwz, const float *cofwt,
    const float *etp, const float *etm, const float *ewp, const float *ewm,
    float *rw_p, float *rho_pp, float *rtheta_pp, float *ww_avg)
{
    REGIONAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int i = RC2(k, cell, ncells);
        if (spec_zone_mask_cell[cell] != 0.0f) {
            rho_pp[i] = mpas_add(rho_pp[i], mpas_mul(dts, tend_rho[i]));
            rtheta_pp[i] = mpas_add(rtheta_pp[i], mpas_mul(dts, tend_rt[i]));
            rw_p[i] = mpas_add(rw_p[i], mpas_mul(dts, tend_rw[i]));
            ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(ewp[k], rw_p[i]));
            continue;
        }
        if (k == 0) continue;
        const int im = RC2(k - 1, cell, ncells);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(ewm[k], rw_p[i]));

        const float theta_implicit = mpas_sub(
            mpas_mul(mpas_mul(etp[k], zz[i]), ts[i]),
            mpas_mul(mpas_mul(etp[k - 1], zz[im]), ts[im]));
        const float theta_explicit = mpas_sub(
            mpas_mul(mpas_mul(etm[k], zz[i]), rtheta_pp[i]),
            mpas_mul(mpas_mul(etm[k - 1], zz[im]), rtheta_pp[im]));
        const float density_implicit = mpas_add(
            mpas_mul(etp[k], rs[i]), mpas_mul(etp[k - 1], rs[im]));
        const float density_explicit = mpas_add(
            mpas_mul(etm[k], rho_pp[i]),
            mpas_mul(etm[k - 1], rho_pp[im]));
        const float heat_upper = mpas_add(
            mpas_mul(etp[k], ts[i]), mpas_mul(etm[k], rtheta_pp[i]));
        const float heat_lower = mpas_add(
            mpas_mul(etp[k - 1], ts[im]),
            mpas_mul(etm[k - 1], rtheta_pp[im]));

        float inner = mpas_sub(tend_rw[i], mpas_mul(cofwz[i],
            mpas_add(theta_implicit, theta_explicit)));
        inner = mpas_sub(inner, mpas_mul(cofwr[i],
            mpas_add(density_implicit, density_explicit)));
        inner = mpas_add(inner, mpas_mul(cofwt[i], heat_upper));
        inner = mpas_add(inner, mpas_mul(cofwt[im], heat_lower));
        rw_p[i] = mpas_add(rw_p[i], mpas_mul(dts, inner));
    }
}

extern "C" __global__ void acoustic_column_sweep_regional_v841(
    const int nlev, const int ncells, const float dts,
    const float *spec_zone_mask_cell,
    const float *a_tri, const float *alpha_tri, const float *gamma_tri,
    float *rw_p)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    if (spec_zone_mask_cell[cell] != 0.0f) return;
    const float dts2 = mpas_mul(dts, dts);
    float previous = rw_p[RC2(0, cell, ncells)];
    for (int k = 1; k < nlev; k += 4) {
        float value[4], lower[4], inverse[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k + j < nlev) {
                const int i = RC2(k + j, cell, ncells);
                value[j] = rw_p[i];
                lower[j] = a_tri[i];
                inverse[j] = alpha_tri[i];
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k + j < nlev) {
                previous = mpas_mul(mpas_sub(value[j], mpas_mul(mpas_mul(
                    dts2, lower[j]), previous)), inverse[j]);
                rw_p[RC2(k + j, cell, ncells)] = previous;
            }
        }
    }
    previous = rw_p[RC2(nlev, cell, ncells)];
    for (int k = nlev - 1; k >= 0; k -= 4) {
        float value[4], upper[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k - j >= 0) {
                const int i = RC2(k - j, cell, ncells);
                value[j] = rw_p[i];
                upper[j] = gamma_tri[i];
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k - j >= 0) {
                previous = mpas_sub(value[j], mpas_mul(upper[j], previous));
                rw_p[RC2(k - j, cell, ncells)] = previous;
            }
        }
    }
}

extern "C" __global__ void acoustic_column_post_w_regional_v841(
    const int nlev, const int ncells, const float dts,
    const float *zz, const float *rho_zz,
    const float *fzm, const float *fzp,
    const float *dss, const float *w, const float *rw, const float *rw_save,
    const float *spec_zone_mask_cell,
    const float *ewp, float *rw_p, float *ww_avg)
{
    REGIONAL_ELEMENT_LOOP((nlev - 1) * ncells) {
        const int cell = element % ncells;
        const int k = 1 + element / ncells;
        if (spec_zone_mask_cell[cell] != 0.0f) continue;
        const int i = RC2(k, cell, ncells);
        const int im = RC2(k - 1, cell, ncells);
        const float delta_saved = mpas_sub(rw_save[i], rw[i]);
        /* MPAS v8.4.2 mpas_atm_time_integration.F:4059-4063:
           the damping product is evaluated from left to right. */
        const float zz_interface = mpas_add(
            mpas_mul(fzm[k], zz[i]), mpas_mul(fzp[k], zz[im]));
        const float rho_interface = mpas_add(
            mpas_mul(fzm[k], rho_zz[i]), mpas_mul(fzp[k], rho_zz[im]));
        float value = mpas_add(rw_p[i], delta_saved);
        value = mpas_sub(value, mpas_mul(mpas_mul(mpas_mul(mpas_mul(
            dts, dss[i]), zz_interface), rho_interface), w[i]));
        value = mpas_div(value, mpas_add(1.0f, mpas_mul(dts, dss[i])));
        rw_p[i] = mpas_sub(value, delta_saved);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(ewp[k], rw_p[i]));
    }
}

extern "C" __global__ void acoustic_column_post_rt_regional_v841(
    const int nlev, const int ncells, const float dts,
    const float *rdzw, const float *rs, const float *ts,
    const float *spec_zone_mask_cell,
    const float *coftz, const float *cofrz, const float *ewp,
    const float *rw_p, float *rho_pp, float *rtheta_pp)
{
    REGIONAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        if (spec_zone_mask_cell[cell] != 0.0f) continue;
        const int i = RC2(k, cell, ncells);
        rho_pp[i] = mpas_sub(rs[i], mpas_mul(mpas_mul(dts, cofrz[k]),
            mpas_sub(mpas_mul(ewp[k + 1], rw_p[RC2(k + 1, cell, ncells)]),
                mpas_mul(ewp[k], rw_p[i]))));
        rtheta_pp[i] = mpas_sub(ts[i], mpas_mul(mpas_mul(dts, rdzw[k]),
            mpas_sub(mpas_mul(mpas_mul(ewp[k + 1],
                    coftz[RC2(k + 1, cell, ncells)]),
                rw_p[RC2(k + 1, cell, ncells)]),
                mpas_mul(mpas_mul(ewp[k], coftz[i]), rw_p[i]))));
    }
}

/* ------------------------------------------------------------------ */
/* the regional scalar-transport branches                               */
/* ------------------------------------------------------------------ */

/* atm_advance_scalars_work F:4764-4842, the three-way edge split:
   full high-order stencil below mask nRelaxZone-1, first-order upwind at
   masks nRelaxZone-1 and nRelaxZone, and no scratch write at all above --
   no updated cell reads those edges. */
extern "C" __global__ void transport_edge_values_regional_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int width, const float coefficient,
    const int *bdy_mask_edge, const float *dv_edge,
    const int *cells_on_edge,
    const float *stage, const float *velocity,
    const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells, float *edge_values)
{
    REGIONAL_ELEMENT_LOOP(ntracers * nlev * nedges) {
        const int edge = element % nedges;
        const int rest = element / nedges;
        const int k = rest % nlev;
        const int tracer = rest / nlev;
        const int mask = bdy_mask_edge[edge];
        if (mask > REGIONAL_N_RELAX_ZONE) {
            /* Native leaves the scratch unwritten here (F:4764-4842) and no
               updated cell reads it: an edge above nRelaxZone touches only
               specified-zone cells, which this routine does not update.  A
               device scratch buffer is uninitialized memory, so the lane is
               written zero instead of left alone -- unread either way, but
               byte-reproducible between two runs of the same configuration,
               which "left alone" is not. */
            edge_values[RQE3(tracer, k, edge, nlev, nedges)] = 0.0f;
            continue;
        }
        if (mask >= REGIONAL_N_RELAX_ZONE - 1) {
            const int cell1 = cells_on_edge[2 * edge];
            const int cell2 = cells_on_edge[2 * edge + 1];
            const float vel = velocity[RE2(k, edge, nedges)];
            const float direction = mpas_copysign(0.5f, vel);
            const float u_positive = mpas_mul(
                dv_edge[edge], mpas_abs(mpas_add(direction, 0.5f)));
            const float u_negative = mpas_mul(
                dv_edge[edge], mpas_abs(mpas_sub(direction, 0.5f)));
            edge_values[RQE3(tracer, k, edge, nlev, nedges)] = mpas_add(
                mpas_mul(u_positive,
                    stage[RQ3(tracer, k, cell1, nlev, ncells)]),
                mpas_mul(u_negative,
                    stage[RQ3(tracer, k, cell2, nlev, ncells)]));
            continue;
        }
        /* The full branch is the released transport_edge_values body verbatim,
           guarded fallback included, so a non-boundary edge produces the same
           bits as the global translation unit.  The contract deck proves that
           claim rather than asserting it. */
        float value = 0.0f;
        const float velocity_sign = mpas_copysign(
            1.0f, velocity[RE2(k, edge, nedges)]);
        bool ftz_sensitive = false;
        for (int slot = 0; slot < n_adv[edge]; ++slot) {
            const int cell = adv_cells[RADV(edge, slot, width)];
            const float base_weight = adv[RADV(edge, slot, width)];
            const float third_weight = adv3[RADV(edge, slot, width)];
            const float stage_value = stage[RQ3(
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
        if (ftz_sensitive) {
            value = 0.0f;
            for (int slot = 0; slot < n_adv[edge]; ++slot) {
                const int cell = adv_cells[RADV(edge, slot, width)];
                const float weight = mpas_add(
                    adv[RADV(edge, slot, width)],
                    mpas_mul(mpas_mul(coefficient, velocity_sign),
                        adv3[RADV(edge, slot, width)]));
                value = mpas_add(value, mpas_mul(weight,
                    stage[RQ3(tracer, k, cell, nlev, ncells)]));
            }
        }
        edge_values[RQE3(tracer, k, edge, nlev, nedges)] = value;
    }
}

/* atm_advance_scalars_work F:4861: the specified zone is not updated by the
   transport routine at all; it takes driving values at the end of the step.
   Everything else is the released standard finish. */
extern "C" __global__ void transport_standard_finish_regional_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges, const float dt,
    const int *bdy_mask_cell,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell,
    const float *velocity, const float *edge_values,
    const float *vertical_flux, const float *rdzw,
    const float *old, const float *rho_old, const float *target_density,
    const float *source, const float *stage, float *output)
{
    REGIONAL_ELEMENT_LOOP(ntracers * nlev * ncells) {
        const int cell = element % ncells;
        const int rest = element / ncells;
        const int k = rest % nlev;
        const int tracer = rest / nlev;
        const int keep = bdy_mask_cell[cell] > REGIONAL_N_RELAX_ZONE;
        const int index = RQ3(tracer, k, cell, nlev, ncells);
        if (keep) {
            output[index] = stage[index];
            continue;
        }
        float horizontal = 0.0f;
        for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
            const int edge = edges_on_cell[RCES(cell, slot, max_edges)];
            const float sign = -acoustic_sign[RCES(cell, slot, max_edges)];
            horizontal = mpas_add(horizontal, mpas_mul(mpas_mul(
                sign, velocity[RE2(k, edge, nedges)]),
                edge_values[RQE3(tracer, k, edge, nlev, nedges)]));
        }
        const float tendency = mpas_add(
            mpas_mul(horizontal, inv_area_cell[cell]), source[index]);
        const float vertical = mpas_mul(rdzw[k], mpas_sub(
            vertical_flux[RQI3(tracer, k + 1, cell, nlev, ncells)],
            vertical_flux[RQI3(tracer, k, cell, nlev, ncells)]));
        const float numerator = mpas_add(
            mpas_mul(old[index], rho_old[RC2(k, cell, ncells)]),
            mpas_mul(dt, mpas_sub(tendency, vertical)));
        const float rho_new_inv = mpas_div(
            1.0f, target_density[RC2(k, cell, ncells)]);
        output[index] = mpas_mul(numerator, rho_new_inv);
    }
}
"""


def _slot_major_rs_ts(source: str) -> str:
    """``acoustic_rs_ts_regional_v841`` reading ``edgesOnCell`` and
    ``edgesOnCell_sign`` slot-major (``[slot][cell]``), derived from the
    kernel's own text so nothing but the two table reads can differ.

    Consecutive threads hold consecutive cells, so a slot's table words are
    one contiguous run instead of a ``maxEdges``-strided gather; every
    arithmetic statement, operand and slot order is the cell-major kernel's.
    The cell-major kernel stays: the LTS gather and the fusion derive from it.
    """

    name = "acoustic_rs_ts_regional_v841"
    marker = f'extern "C" __global__ void {name}('
    start = source.index(marker)
    first = source.index(")\n{", start) + 2
    depth = 1
    last = first + 1
    while depth:
        depth += (source[last] == "{") - (source[last] == "}")
        last += 1
    text = source[start:last]
    replacements = (
        (f"void {name}(", "void acoustic_rs_ts_slot_regional_v841("),
        ("edges_on_cell[RCES(cell, slot, max_edges)]",
         "edges_on_cell[RSLOT(slot, cell, ncells)]"),
        ("edge_sign_on_cell[RCES(cell, slot, max_edges)]",
         "edge_sign_on_cell[RSLOT(slot, cell, ncells)]"),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"{name}: expected one {old!r}, found {text.count(old)}")
        text = text.replace(old, new)
    return (
        "\n/* Derived by cuda_regional_v841._slot_major_rs_ts: the kernel above"
        "\n   with its two edgesOnCell table reads slot-major. */"
        "\n#define RSLOT(s,c,nc) ((s)*(nc) + (c))\n" + text + "\n"
    )


_CUDA_SOURCE = _CUDA_SOURCE + _slot_major_rs_ts(_CUDA_SOURCE)


def _regional_source() -> str:
    """The translation unit with its zone constants bound to the CPU lane.

    The two constants are emitted from :mod:`woof.hex.regional_v841` rather
    than typed into the CUDA text, so the device zone geometry cannot drift
    from the host authority's without the source digest changing.
    """

    prologue = (
        f"#define REGIONAL_N_SPEC_ZONE {int(N_SPEC_ZONE)}\n"
        f"#define REGIONAL_N_RELAX_ZONE {int(N_RELAX_ZONE)}\n"
    )
    return prologue + _CUDA_SOURCE


CUDA_REGIONAL_SOURCE = _regional_source()

#: Every entrypoint this translation unit publishes, in the order the
#: contract deck exercises them.  The deck refuses if the module resolves a
#: kernel this list does not name, so a kernel cannot be added without a
#: deck.
REGIONAL_KERNELS: tuple[str, ...] = (
    "regional_lbc_derive_v841",
    "regional_lbc_rho_edge_v841",
    "regional_lbc_tendency_v841",
    "regional_lbc_state_at_v841",
    "regional_speczone_tend_cell_v841",
    "regional_speczone_tend_edge_v841",
    "regional_relaxzone_rayleigh_cell_v841",
    "regional_relaxzone_rayleigh_edge_v841",
    "regional_relaxzone_filter_cell_v841",
    "regional_relaxzone_filter_edge_v841",
    "regional_speczone_u_ru_v841",
    "regional_zero_speczone_w_v841",
    "regional_reset_speczone_values_v841",
    "regional_bdy_adjust_scalars_compute_v841",
    "regional_bdy_adjust_scalars_copyback_v841",
    "regional_bdy_set_scalars_v841",
    "regional_clamp_negative_scalars_v841",
    "acoustic_ru_regional_v841",
    "acoustic_rs_ts_regional_v841",
    "acoustic_rs_ts_slot_regional_v841",
    "acoustic_column_solve_regional_v841",
    "acoustic_column_pre_regional_v841",
    "acoustic_column_sweep_regional_v841",
    "acoustic_column_post_w_regional_v841",
    "acoustic_column_post_rt_regional_v841",
    "transport_edge_values_regional_v841",
    "transport_standard_finish_regional_v841",
)


def _cp() -> Any:
    import cupy as cp

    return cp


class RegionalDeviceRefusal(ConfigurationRefusal):
    """A regional device configuration is refused by name."""


def _f32(name: str, value: Any) -> Any:
    return require_resident_array(name, value, dtype=np.float32)


def _i32(name: str, value: Any) -> Any:
    return require_resident_array(name, value, dtype=np.int32)


@dataclass(slots=True)
class DeviceRegionalMasks:
    """The 7-ring masks and their derived element lists, resident.

    Every array is the host authority's own
    (:func:`woof.hex.regional_v841.derive_regional_masks`) uploaded
    unchanged: no device arithmetic touches them, so the zone geometry the
    kernels see is the zone geometry the CPU oracle used, by construction.
    """

    bdy_mask_cell: Any
    bdy_mask_edge: Any
    spec_zone_mask_cell: Any
    spec_zone_mask_edge: Any
    spec_cells: Any
    spec_edges: Any
    relax_cells: Any
    relax_edges: Any
    nudged_cells: Any
    n_cells: int
    n_edges: int
    h2d: TransferStats

    @classmethod
    def from_host(cls, masks: RegionalMasks) -> "DeviceRegionalMasks":
        import time

        cp = _cp()
        started = time.perf_counter()
        payload = {
            "bdy_mask_cell": np.ascontiguousarray(
                masks.bdy_mask_cell, dtype=np.int32
            ),
            "bdy_mask_edge": np.ascontiguousarray(
                masks.bdy_mask_edge, dtype=np.int32
            ),
            "spec_zone_mask_cell": np.ascontiguousarray(
                masks.spec_zone_mask_cell, dtype=np.float32
            ),
            "spec_zone_mask_edge": np.ascontiguousarray(
                masks.spec_zone_mask_edge, dtype=np.float32
            ),
            "spec_cells": np.ascontiguousarray(masks.spec_cells, np.int32),
            "spec_edges": np.ascontiguousarray(masks.spec_edges, np.int32),
            "relax_cells": np.ascontiguousarray(masks.relax_cells, np.int32),
            "relax_edges": np.ascontiguousarray(masks.relax_edges, np.int32),
            "nudged_cells": np.ascontiguousarray(masks.nudged_cells, np.int32),
        }
        device = {name: cp.asarray(value) for name, value in payload.items()}
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        return cls(
            **device,
            n_cells=int(masks.n_cells),
            n_edges=int(masks.n_edges),
            h2d=TransferStats(
                int(sum(int(v.nbytes) for v in device.values())), elapsed
            ),
        )

    def validate(self) -> None:
        _i32("regional.bdy_mask_cell", self.bdy_mask_cell)
        _i32("regional.bdy_mask_edge", self.bdy_mask_edge)
        _f32("regional.spec_zone_mask_cell", self.spec_zone_mask_cell)
        _f32("regional.spec_zone_mask_edge", self.spec_zone_mask_edge)
        for name in (
            "spec_cells",
            "spec_edges",
            "relax_cells",
            "relax_edges",
            "nudged_cells",
        ):
            _i32(f"regional.{name}", getattr(self, name))
        if tuple(self.bdy_mask_cell.shape) != (self.n_cells,):
            raise ValueError("regional.bdy_mask_cell shape differs from nCells")
        if tuple(self.bdy_mask_edge.shape) != (self.n_edges,):
            raise ValueError("regional.bdy_mask_edge shape differs from nEdges")

    @property
    def n_spec_cells(self) -> int:
        return int(self.spec_cells.size)

    @property
    def n_spec_edges(self) -> int:
        return int(self.spec_edges.size)

    @property
    def n_relax_cells(self) -> int:
        return int(self.relax_cells.size)

    @property
    def n_relax_edges(self) -> int:
        return int(self.relax_edges.size)

    @property
    def n_nudged_cells(self) -> int:
        return int(self.nudged_cells.size)


@lru_cache(maxsize=None)
def _direct_regional_module(device_id: int, platform_sha256: str) -> tuple[Any, dict[str, Any]]:
    """Load direct NVRTC output and retain its compiler evidence per device.

    The platform digest binds the shared cache's compile contract. It is part
    of the resident-module key even though NVRTC itself does not consume it.
    """
    from cupy.cuda import compiler
    from .cuda_backend.compile_contract import NvrtcCompileCapture

    cp = _cp()
    with cp.cuda.Device(device_id):
        with NvrtcCompileCapture() as capture:
            binary, _ = compiler.compile_using_nvrtc(
                CUDA_REGIONAL_SOURCE, options=_hex_numerics.options(REGIONAL_CUDA_OPTIONS),
                arch=_compilation_target(device_id, xp=cp),
            )
        module = cp.cuda.function.Module()
        module.load(binary)
    evidence = capture.evidence(hashlib.sha256(CUDA_REGIONAL_SOURCE.encode()).hexdigest())
    return module, evidence


class CudaRegionalKernels:
    """One NVRTC translation unit of regional kernels, bound to one cache."""

    def __init__(self, kernel_cache: Any) -> None:
        self.cache = kernel_cache
        self._kernels: dict[str, Any] = {}

    def kernel(self, name: str) -> Any:
        if name not in REGIONAL_KERNELS:
            raise KeyError(
                f"{name!r} is not a declared regional kernel; the contract "
                f"deck covers {len(REGIONAL_KERNELS)} entrypoints and a "
                "kernel with no deck has no expected bits to be checked "
                "against"
            )
        result = self._kernels.get(name)
        if result is None:
            started = time.perf_counter()
            manifest = self.cache._compile_manifest
            compiled_key = "direct-nvrtc:" + self.cache._module_key(
                CUDA_REGIONAL_SOURCE, _hex_numerics.options(REGIONAL_CUDA_OPTIONS)
            )
            manifest.declare_module(
                MODULE_KEY, source=CUDA_REGIONAL_SOURCE,
                requested_options=_hex_numerics.options(REGIONAL_CUDA_OPTIONS), module_cache_key=compiled_key,
            )
            module, evidence = _direct_regional_module(
                self.cache.capability.device_id, manifest.fingerprint_sha256,
            )
            manifest.record_effective_compile(MODULE_KEY, evidence)
            manifest.record_kernel(MODULE_KEY, name)
            kernel_key = f"{compiled_key}:{name}"
            result = module.get_function(name)
            self.cache._modules[compiled_key] = module
            self.cache._kernels[kernel_key] = result
            self.cache._compile_seconds[kernel_key] = time.perf_counter() - started
            # Preserve the forecast's regional post-launch observer.
            result = self.cache._observe(kernel_key, result, name, MODULE_KEY)
            self._kernels[name] = result
        return result

    def launch(self, name: str, count: int, args: tuple[Any, ...]) -> None:
        if int(count) < 1:
            return
        threads = 128
        blocks = (int(count) + threads - 1) // threads
        self.kernel(name)((blocks,), (threads,), args)


# ---------------------------------------------------------------------------
# the resident lateral-boundary pool
# ---------------------------------------------------------------------------

#: The pool fields carried on the device, and whether each is a cell field,
#: an edge field or an interface (nVertLevels+1) field.  Mirrors
#: ``regional_v841._DRIVING_FIELDS``.
DRIVING_FIELD_KIND: Mapping[str, str] = {
    "u": "edge",
    "ru": "edge",
    "rho_edge": "edge",
    "w": "interface",
    "rho": "cell",
    "rho_zz": "cell",
    "theta": "cell",
    "rtheta_m": "cell",
}


class DeviceRegionalDrivingState:
    """The two-level LBC pool with its derived coupled fields, on device.

    File decoding stays on the host (the reader is
    :mod:`woof.hex.lbc`); everything after the read is device arithmetic:
    the four derived coupled fields, the interval tendencies, and the in-step
    time interpolation every regional stage consumes.  The host never
    computes a driving value the kernels use.
    """

    def __init__(
        self,
        inventory: LbcInventory,
        masks: DeviceRegionalMasks,
        *,
        cells_on_edge: Any,
        zz: Any,
        n_vert_levels: int,
        kernels: CudaRegionalKernels,
        scalar_names: Sequence[str] = ("lbc_qv",),
    ) -> None:
        cp = _cp()
        self._pool = LbcPool(inventory, scalar_names=scalar_names, compute_tendencies=False)
        self._kernels = kernels
        self._masks = masks
        self._scalar_names = tuple(scalar_names)
        if not self._scalar_names:
            raise ConfigurationRefusal(
                "scalar_names",
                (),
                "the lbc_scalars var_array always carries at least qv",
                "scalar_names=('lbc_qv', ...)",
            )
        self._cells_on_edge = _i32("regional.cells_on_edge", cells_on_edge)
        self._zz = _f32("regional.zz", zz)
        self.nlev = int(n_vert_levels)
        self.ncells = int(masks.n_cells)
        self.nedges = int(masks.n_edges)
        # MPAS-A v8.4.2 mpas_block_creator.F:1490-1496 reindexes absent
        # file neighbors before mpas_atm_boundaries.F:234-247 gathers them.
        self._cells_on_edge = cp.where(
            self._cells_on_edge < 0, np.int32(self.ncells), self._cells_on_edge
        )
        self._cell_shape = (self.nlev, self.ncells)
        self._edge_shape = (self.nlev, self.nedges)
        self._scalar_shape = (
            len(self._scalar_names),
            self.nlev,
            self.ncells,
        )
        self._state: dict[str, Any] | None = None
        self._tend: dict[str, Any] | None = None
        self._state_scalars: Any | None = None
        self._tend_scalars: Any | None = None
        # Initial pool allocation before the first boundary admission.
        self._rho_edge_zero = cp.zeros(self._edge_shape, dtype=cp.float32)

    @property
    def interval_end(self) -> datetime:
        return self._pool.interval_end

    @property
    def scalar_names(self) -> tuple[str, ...]:
        return self._scalar_names

    def _missing_scalar(self, index: int) -> Any:
        """Native allocated/recycled scalar storage, entirely on the device.

        MPAS-A v8.4.2 gen_inc.c:615-620; mpas_field_routines.F:383-395,
        3335-3347; mpas_io.F:858-878. A missing read retains its time-level
        allocation, which holds the preceding tendency after rotation.
        """
        cp = _cp()
        if self._tend_scalars is None:
            return cp.zeros(self._cell_shape, dtype=cp.float32)
        return cp.ascontiguousarray(self._tend_scalars[index])

    def _derive(self, admitted: LbcFile) -> tuple[dict[str, Any], Any]:
        cp = _cp()

        def upload(name: str) -> Any:
            slab = np.ascontiguousarray(
                np.asarray(admitted.fields[name], dtype=np.float32).T
            )
            return cp.asarray(slab)

        u = upload("lbc_u")
        w = upload("lbc_w")
        rho = upload("lbc_rho")
        theta = upload("lbc_theta")
        scalars = cp.stack(
            [upload(name) if name in admitted.fields else self._missing_scalar(j)
             for j,name in enumerate(self._scalar_names)], axis=0
        )
        qv = scalars[0]
        rho_zz = cp.empty(self._cell_shape, dtype=cp.float32)
        rtheta_m = cp.empty(self._cell_shape, dtype=cp.float32)
        self._kernels.launch(
            "regional_lbc_derive_v841",
            self.nlev * self.ncells,
            (
                np.int32(self.nlev * self.ncells),
                np.float32(RVORD_F32),
                self._zz,
                rho,
                theta,
                cp.ascontiguousarray(qv),
                rho_zz,
                rtheta_m,
            ),
        )
        rho_edge = cp.array(self._rho_edge_zero, copy=True)
        ru = cp.empty(self._edge_shape, dtype=cp.float32)
        self._kernels.launch(
            "regional_lbc_rho_edge_v841",
            self.nlev * self.nedges,
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                self._cells_on_edge,
                rho_zz,
                u,
                rho_edge,
                ru,
            ),
        )
        state = {
            "u": u,
            "ru": ru,
            "rho_edge": rho_edge,
            "w": w,
            "rho": rho,
            "rho_zz": rho_zz,
            "theta": theta,
            "rtheta_m": rtheta_m,
        }
        return state, scalars

    def start(self, when: datetime) -> None:
        admitted = self._pool.start(when)
        self._state, self._state_scalars = self._derive(admitted)
        self._tend = None
        self._tend_scalars = None

    def advance(self, when: datetime | None = None) -> None:
        if self._state is None or self._state_scalars is None:
            raise LbcAdmissionError(
                "advance was called on a driving state that never started"
            )
        cp = _cp()
        old_state = self._state
        old_scalars = self._state_scalars
        old_end = self._pool.interval_end
        admitted = self._pool.advance(when)
        new_state, new_scalars = self._derive(admitted)
        # mpas_atm_boundaries.F:265-272 -- the interval in whole days plus
        # seconds, in REAL(RKIND), then one reciprocal.
        delta = admitted.valid_time - old_end
        dt = rkind_interval_seconds(delta)
        inv_dt = np.float32(np.float32(1.0) / dt)
        tend: dict[str, Any] = {}
        for name in DRIVING_FIELD_KIND:
            out = cp.empty_like(new_state[name])
            self._kernels.launch(
                "regional_lbc_tendency_v841",
                int(out.size),
                (
                    np.int32(out.size),
                    inv_dt,
                    new_state[name],
                    old_state[name],
                    out,
                ),
            )
            tend[name] = out
        tend_scalars = cp.empty_like(new_scalars)
        self._kernels.launch(
            "regional_lbc_tendency_v841",
            int(tend_scalars.size),
            (
                np.int32(tend_scalars.size),
                inv_dt,
                new_scalars,
                old_scalars,
                tend_scalars,
            ),
        )
        self._tend = tend
        self._tend_scalars = tend_scalars
        self._state = new_state
        self._state_scalars = new_scalars

    def _require_ready(self) -> None:
        if self._tend is None or self._tend_scalars is None:
            raise LbcAdmissionError(
                "the regional driving state holds no complete interval; "
                "start then advance must both run before boundary "
                "tendencies or interpolated state exist"
            )

    def tendency(self, name: str) -> Any:
        self._require_ready()
        assert self._tend is not None
        if name == "scalars":
            return self._tend_scalars
        scalar_index = scalar_index_for_field(name, self._scalar_names)
        if name not in DRIVING_FIELD_KIND and scalar_index is not None:
            return self._tend_scalars[scalar_index]
        if name not in DRIVING_FIELD_KIND:
            raise ConfigurationRefusal(
                "field",
                name,
                "not a driving field of the regional lbc pool",
                f"one of {sorted(DRIVING_FIELD_KIND)} or 'scalars'",
            )
        return self._tend[name]

    def remaining_seconds(
        self, step_start: datetime, delta_t: np.float32
    ) -> np.float32:
        """``dt_remaining`` of ``mpas_atm_get_bdy_state`` (F:491-551)."""

        delta = self._pool.interval_end - step_start
        remaining = rkind_interval_seconds(delta)
        return np.float32(remaining - np.float32(delta_t))

    def state_at(
        self, name: str, step_start: datetime, delta_t: np.float32, out: Any = None
    ) -> Any:
        """Device-side ``mpas_atm_get_bdy_state``: state - dt*tend."""

        self._require_ready()
        assert self._state is not None and self._tend is not None
        cp = _cp()
        dt = self.remaining_seconds(step_start, delta_t)
        if name == "scalars":
            state = self._state_scalars
            tend = self._tend_scalars
        else:
            scalar_index = scalar_index_for_field(name, self._scalar_names)
            if name not in DRIVING_FIELD_KIND and scalar_index is not None:
                state = self._state_scalars[scalar_index]
                tend = self._tend_scalars[scalar_index]
            else:
                if name not in DRIVING_FIELD_KIND:
                    raise ConfigurationRefusal(
                        "field",
                        name,
                        "not a driving field of the regional lbc pool",
                        f"one of {sorted(DRIVING_FIELD_KIND)} or 'scalars'",
                    )
                state = self._state[name]
                tend = self._tend[name]
        target = cp.empty_like(state) if out is None else out
        self._kernels.launch(
            "regional_lbc_state_at_v841",
            int(state.size),
            (np.int32(state.size), dt, state, tend, target),
        )
        return target


class DeviceRegionalRuntime:
    """Everything the CUDA whole-step driver needs for the regional branch."""

    def __init__(
        self,
        mesh: object,
        *,
        lbc_paths: Sequence[str],
        start_time: datetime,
        config_h_scale_with_mesh: bool,
        zz_host: Any,
        zz_device: Any,
        cells_on_edge_device: Any,
        n_vert_levels: int,
        kernel_cache: Any,
        config_apply_lbcs: bool = True,
        config_relax_zone_divdamp_coef: float = 6.0,
        scalar_names: Sequence[str] = ("lbc_qv",),
    ) -> None:
        cp = _cp()
        dtype = np.dtype(np.float32)
        self.masks_host = derive_regional_masks(mesh, dtype)
        regional_bdy_checks(
            self.masks_host,
            config_apply_lbcs=bool(config_apply_lbcs),
            lbc_input_interval_valid=True,
        )
        self.masks = DeviceRegionalMasks.from_host(self.masks_host)
        self.masks.validate()
        scaling_cell, scaling_edge = compute_mesh_scaling_regional(
            mesh, dtype, config_h_scale_with_mesh=config_h_scale_with_mesh
        )
        self.mesh_scaling_cell_host = scaling_cell
        self.mesh_scaling_edge_host = scaling_edge
        self.mesh_scaling_cell = cp.asarray(
            np.ascontiguousarray(scaling_cell, dtype=np.float32)
        )
        self.mesh_scaling_edge = cp.asarray(
            np.ascontiguousarray(scaling_edge, dtype=np.float32)
        )
        self.config_relax_zone_divdamp_coef = float(
            config_relax_zone_divdamp_coef
        )
        self.kernels = CudaRegionalKernels(kernel_cache)
        self.zz_host = np.asarray(zz_host, dtype=np.float32)
        self.driving = DeviceRegionalDrivingState(
            LbcInventory(lbc_paths),
            self.masks,
            cells_on_edge=cells_on_edge_device,
            zz=zz_device,
            n_vert_levels=n_vert_levels,
            kernels=self.kernels,
            scalar_names=scalar_names,
        )
        self.clock = start_time
        self._started = False

    def ensure_interval(self) -> None:
        """The ``lbc_in`` read cadence of ``atm_core_run`` (F:735-781)."""

        if not self._started:
            self.driving.start(self.clock)
            self.driving.advance(self.clock)
            self._started = True
            return
        if self.clock >= self.driving.interval_end:
            self.driving.advance(self.clock)

    def advance_clock(self, seconds: float) -> None:
        from datetime import timedelta

        self.clock = self.clock + timedelta(seconds=seconds)


__all__ = [
    "CUDA_REGIONAL_SOURCE",
    "CudaRegionalKernels",
    "DRIVING_FIELD_KIND",
    "DeviceRegionalDrivingState",
    "DeviceRegionalMasks",
    "DeviceRegionalRuntime",
    "MODULE_KEY",
    "REGIONAL_KERNELS",
    "RegionalDeviceRefusal",
]
