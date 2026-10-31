"""RRFS fork scalar-offcentred acoustic subsystem.

Source: RRFSx/MPAS-Model MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2366,2646-2984
at 1d6e370e7b43a332074809f170532a216c2212fb. Coefficients include dts;
these are not interchangeable with v8.4.2's unscaled coefficients.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics
from typing import Any
import numpy as np
import weakref
from .cuda_acoustic import (CudaAcousticForcing, CudaAcousticState,
    CudaVerticalImplicitCoefficients, _float32, _int32, _launch_1d, _mesh_value)
from .cuda_fp32 import CUDA_FTZ_HELPERS

_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
extern "C" __global__ void acoustic_cofrz_rrfs(
    const int nlev, const float dts, const float epssm,
    const float *rdzw, float *cofrz)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < nlev) cofrz[k] = mpas_mul(
        mpas_mul(mpas_mul(0.5f, dts), mpas_add(1.0f, epssm)), rdzw[k]);
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
extern "C" __global__ void acoustic_coefficients_rrfs(
    const int nlev, const int ncells,
    const float dts, const float epssm,
    const float gravity, const float rgas, const float cp,
    const float *zz, const float *cqw, const float *pressure,
    const float *theta, const float *rho_base,
    const float *rtheta_base, const float *pressure_base,
    const float *rtheta_p, const float *qtot,
    const float *rdzw, const float *fzm, const float *fzp,
    const float *rdzu, const int nsolve,
    float *cofwr, float *cofwz, float *coftz, float *cofwt,
    float *cofrz, float *a_tri, float *b_tri, float *c_tri,
    float *alpha_tri, float *gamma_tri)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    const float dtseps = 0.5f * dts * (1.0f + epssm);
    const float rcv = rgas / (cp - rgas);
    const float csquared = cp * rcv;

    {
        const int i = C2(k, cell, ncells);
        cofwt[i] = 0.5f * dtseps * rcv * zz[i] * gravity * rho_base[i]
            / (1.0f + qtot[i]) * pressure[i]
            / ((rtheta_base[i] + rtheta_p[i]) * pressure_base[i]);
        a_tri[i] = 0.0f;
        b_tri[i] = 1.0f;
        c_tri[i] = 0.0f;
        alpha_tri[i] = 0.0f;
        gamma_tri[i] = 0.0f;
        coftz[C2(k, cell, ncells)] = 0.0f;
    }
    if (k == 0) coftz[C2(nlev, cell, ncells)] = 0.0f;

    if (k > 0) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float interp_zz = fzm[k] * zz[i] + fzp[k] * zz[im];
        cofwr[i] = 0.5f * dtseps * gravity * interp_zz;
        const float interp_p = fzm[k] * pressure[i] + fzp[k] * pressure[im];
        cofwz[i] = dtseps * csquared * interp_zz * rdzu[k]
            * cqw[i] * interp_p;
        coftz[i] = dtseps * (fzm[k] * theta[i] + fzp[k] * theta[im]);
    }
}

extern "C" __global__ void acoustic_matrix_rrfs(
    const int nlev, const int ncells,
    const float dts, const float epssm,
    const float gravity, const float rgas, const float cp,
    const float *zz, const float *cqw, const float *pressure,
    const float *theta, const float *rho_base,
    const float *rtheta_base, const float *pressure_base,
    const float *rtheta_p, const float *qtot,
    const float *rdzw, const float *fzm, const float *fzp,
    const float *rdzu, const int nsolve,
    float *cofwr, float *cofwz, float *coftz, float *cofwt,
    float *cofrz, float *a_tri, float *b_tri, float *c_tri,
    float *alpha_tri, float *gamma_tri)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    if (k > 0) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const int ip = C2(k + 1, cell, ncells);
        a_tri[i] = -cofwz[i] * coftz[im] * rdzw[k - 1] * zz[im]
            + cofwr[i] * cofrz[k - 1]
            - cofwt[im] * coftz[im] * rdzw[k - 1];
        b_tri[i] = 1.0f
            + cofwz[i] * (coftz[i] * rdzw[k] * zz[i]
                + coftz[i] * rdzw[k - 1] * zz[im])
            - coftz[i] * (cofwt[i] * rdzw[k]
                - cofwt[im] * rdzw[k - 1])
            + cofwr[i] * (cofrz[k] - cofrz[k - 1]);
        c_tri[i] = -cofwz[i] * coftz[ip] * rdzw[k] * zz[i]
            - cofwr[i] * cofrz[k]
            + cofwt[i] * coftz[ip] * rdzw[k];
    }
}

extern "C" __global__ void acoustic_factor_rrfs(
    const int nlev, const int ncells,
    const float dts, const float epssm,
    const float gravity, const float rgas, const float cp,
    const float *zz, const float *cqw, const float *pressure,
    const float *theta, const float *rho_base,
    const float *rtheta_base, const float *pressure_base,
    const float *rtheta_p, const float *qtot,
    const float *rdzw, const float *fzm, const float *fzp,
    const float *rdzu, const int nsolve,
    float *cofwr, float *cofwz, float *coftz, float *cofwt,
    float *cofrz, float *a_tri, float *b_tri, float *c_tri,
    float *alpha_tri, float *gamma_tri)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= nsolve) return;
    // Forward elimination is a vertical recurrence. Keep its order.
    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float denominator = b_tri[i] - a_tri[i] * gamma_tri[im];
        alpha_tri[i] = 1.0f / denominator;
        gamma_tri[i] = c_tri[i] * alpha_tri[i];
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
extern "C" __global__ void acoustic_ru_rrfs(
    const int nlev, const int nedges, const int ncells,
    const int small_step, const float dts, const int edge_start, const int edge_end,
    const float gravity, const float rgas, const float cp,
    const int *cells_on_edge, const float *inv_dc_edge,
    const float *zz, const float *exner, const float *cqu,
    const float *zxu, const float *tend_ru, const float *spec_edge, const int nsolve,
    const float *rho_pp, const float *rtheta_pp,
    float *ru_p, float *ru_avg)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * nedges) return;
    const int edge = element % nedges;
    const int k = element / nedges;
    if (edge < edge_start || edge >= edge_end) return;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    if (c0 >= nsolve && c1 >= nsolve) return;
    const float half = 0.5f;
    const float rcv = mpas_div(rgas, mpas_sub(cp, rgas));
    const float csquared = mpas_mul(cp, rcv);
    {
        const int index = E2(k, edge, nedges);
        if (small_step == 1) {
            const float value = mpas_mul(dts, tend_ru[index]);
            ru_p[index] = value;
            ru_avg[index] = value;
        } else {
            const float normal = mpas_mul(mpas_sub(
                rtheta_pp[C2(k, c1, ncells)],
                rtheta_pp[C2(k, c0, ncells)]), inv_dc_edge[edge]);
            const float normalized = mpas_div(normal, mpas_mul(half, mpas_add(
                zz[C2(k, c1, ncells)], zz[C2(k, c0, ncells)])));
            float pgrad = mpas_mul(cqu[index], half);
            pgrad = mpas_mul(pgrad, csquared);
            pgrad = mpas_mul(pgrad, mpas_add(exner[C2(k, c0, ncells)],
                exner[C2(k, c1, ncells)]));
            pgrad = mpas_mul(pgrad, normalized);
            const float terrain = mpas_mul(mpas_mul(mpas_mul(
                half, zxu[index]), gravity), mpas_add(
                    rho_pp[C2(k, c0, ncells)],
                    rho_pp[C2(k, c1, ncells)]));
            pgrad = mpas_add(pgrad, terrain);
            ru_p[index] = mpas_add(ru_p[index], mpas_mul(
                dts, mpas_sub(tend_ru[index], mpas_mul(mpas_sub(1.0f, spec_edge[edge]), pgrad))));
            ru_avg[index] = mpas_add(ru_avg[index], ru_p[index]);
        }
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
extern "C" __global__ void acoustic_prepare_rrfs(
    const int nlev, const int ncells, const int small_step, const int nsolve,
    const int cell_start, const int cell_end,
    float *rw_p, float *rtheta_pp, float *rtheta_pp_old,
    float *rho_pp, float *ww_avg)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= (nlev + 1) * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= ncells) return;
    if (k < nlev) {
        const int index = C2(k, cell, ncells);
        if (small_step == 1) {
            if (cell >= cell_start && cell < cell_end) rtheta_pp_old[index] = 0.0f;
            if (cell < nsolve) {
                rtheta_pp[index] = 0.0f;
                rho_pp[index] = 0.0f;
            }
        } else {
            if (cell >= cell_start && cell < cell_end) rtheta_pp_old[index] = rtheta_pp[index];
        }
    }
    if (small_step == 1 && cell < nsolve) {
         {
            const int index = C2(k, cell, ncells);
            rw_p[index] = 0.0f;
            ww_avg[index] = 0.0f;
        }
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
extern "C" __global__ void acoustic_rs_ts_rrfs(
    const int nlev, const int ncells, const int nedges,
    const int max_edges, const float dts, const float epssm,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const int *cells_on_edge, const float *edge_sign_on_cell,
    const float *dv_edge, const float *inv_area_cell, const float *spec_cell, const int nsolve,
    const float *theta_m, const float *rdzw,
    const float *cofrz, const float *coftz,
    const float *ru_p, const float *rw_p,
    const float *rho_pp, const float *rtheta_pp,
    const float *tend_rho, const float *tend_rt,
    float *rs, float *ts)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    if (spec_cell[cell] != 0.0f) {
         {
            rs[C2(k, cell, ncells)] = tend_rho[C2(k, cell, ncells)];
            ts[C2(k, cell, ncells)] = tend_rt[C2(k, cell, ncells)];
        }
        return;
    }
    const int count = n_edges_on_cell[cell];
    {
        const int index = C2(k, cell, ncells);
        float r = 0.0f;
        float t = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[slot * ncells + cell];
            const int c0 = cells_on_edge[2 * edge];
            const int c1 = cells_on_edge[2 * edge + 1];
            const float flux = mpas_mul(mpas_mul(mpas_mul(mpas_mul(
                edge_sign_on_cell[slot * ncells + cell], dts),
                dv_edge[edge]), ru_p[E2(k, edge, nedges)]), inv_area_cell[cell]);
            r = mpas_sub(r, flux);
            t = mpas_sub(t, mpas_mul(mpas_mul(flux, 0.5f), mpas_add(
                theta_m[C2(k, c1, ncells)],
                theta_m[C2(k, c0, ncells)])));
        }
        const float resm = mpas_div(mpas_sub(1.0f, epssm), mpas_add(1.0f, epssm));
        r = mpas_sub(mpas_add(mpas_add(rho_pp[index], mpas_mul(dts, tend_rho[index])), r),
            mpas_mul(mpas_mul(cofrz[k], resm), mpas_sub(
                rw_p[C2(k + 1, cell, ncells)], rw_p[index])));
        t = mpas_sub(mpas_add(mpas_add(rtheta_pp[index], mpas_mul(dts, tend_rt[index])), t),
            mpas_mul(mpas_mul(resm, rdzw[k]), mpas_sub(
                mpas_mul(coftz[C2(k + 1, cell, ncells)], rw_p[C2(k + 1, cell, ncells)]),
                mpas_mul(coftz[index], rw_p[index]))));
        rs[index] = r;
        ts[index] = t;
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
// Split follows cuda_acoustic_v841 pre/sweep/post_w/post_rt ordering.
extern "C" __global__ void acoustic_column_pre_rrfs(
    const int nlev, const int ncells,
    const float dts, const float epssm, const int nsolve, const float *spec_cell,
    const float *zz, const float *rho_zz,
    const float *fzm, const float *fzp, const float *rdzw,
    const float *dss, const float *w,
    const float *rw, const float *rw_save,
    const float *tend_rw, const float *rs, const float *ts,
    const float *cofwr, const float *cofwz, const float *coftz,
    const float *cofwt, const float *cofrz,
    const float *a_tri, const float *alpha_tri, const float *gamma_tri,
    float *rw_p, float *rho_pp, float *rtheta_pp, float *ww_avg)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    if (spec_cell[cell] != 0.0f) {
        {
            const int i = C2(k, cell, ncells);
            rho_pp[i] = mpas_add(rho_pp[i], mpas_mul(dts, rs[i]));
            rtheta_pp[i] = mpas_add(rtheta_pp[i], mpas_mul(dts, ts[i]));
            rw_p[i] = mpas_add(rw_p[i], mpas_mul(dts, tend_rw[i]));
            ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(mpas_mul(0.5f, mpas_add(1.0f, epssm)), rw_p[i]));
        }
        return;
    }
    if (k == 0) return;
    const float resm = (1.0f - epssm) / (1.0f + epssm);
    {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(
            mpas_mul(0.5f, mpas_sub(1.0f, epssm)), rw_p[i]));
        rw_p[i] = mpas_add(mpas_add(mpas_sub(mpas_sub(mpas_add(
            rw_p[i], mpas_mul(dts, tend_rw[i])), mpas_mul(cofwz[i],
                mpas_add(mpas_sub(mpas_mul(zz[i], ts[i]),
                    mpas_mul(zz[im], ts[im])), mpas_mul(resm,
                        mpas_sub(mpas_mul(zz[i], rtheta_pp[i]),
                            mpas_mul(zz[im], rtheta_pp[im])))))),
            mpas_mul(cofwr[i], mpas_add(mpas_add(rs[i], rs[im]),
                mpas_mul(resm, mpas_add(rho_pp[i], rho_pp[im]))))),
            mpas_mul(cofwt[i], mpas_add(ts[i],
                mpas_mul(resm, rtheta_pp[i])))),
            mpas_mul(cofwt[im], mpas_add(ts[im],
                mpas_mul(resm, rtheta_pp[im]))));
    }
}

extern "C" __global__ void acoustic_column_solve_rrfs(
    const int nlev, const int ncells,
    const float dts, const float epssm, const int nsolve, const float *spec_cell,
    const float *zz, const float *rho_zz,
    const float *fzm, const float *fzp, const float *rdzw,
    const float *dss, const float *w,
    const float *rw, const float *rw_save,
    const float *tend_rw, const float *rs, const float *ts,
    const float *cofwr, const float *cofwz, const float *coftz,
    const float *cofwt, const float *cofrz,
    const float *a_tri, const float *alpha_tri, const float *gamma_tri,
    float *rw_p, float *rho_pp, float *rtheta_pp, float *ww_avg)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= nsolve || spec_cell[cell] != 0.0f) return;

    // Load four levels before evaluating their dependent recurrence. Each
    // column still visits every k in the original order; previous is exactly
    // the value just stored to rw_p, held in a register instead of reloaded.
    float previous = rw_p[cell];
    for (int k = 1; k < nlev; k += 4) {
        float value[4], lower[4], inverse[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k + j < nlev) {
                const int i = C2(k + j, cell, ncells);
                value[j] = rw_p[i];
                lower[j] = a_tri[i];
                inverse[j] = alpha_tri[i];
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k + j < nlev) {
                previous = mpas_mul(mpas_sub(value[j],
                    mpas_mul(lower[j], previous)), inverse[j]);
                rw_p[C2(k + j, cell, ncells)] = previous;
            }
        }
    }
    previous = rw_p[C2(nlev, cell, ncells)];
    for (int k = nlev - 1; k >= 0; k -= 4) {
        float value[4], upper[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k - j >= 0) {
                const int i = C2(k - j, cell, ncells);
                value[j] = rw_p[i];
                upper[j] = gamma_tri[i];
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (k - j >= 0) {
                previous = mpas_sub(value[j], mpas_mul(upper[j], previous));
                rw_p[C2(k - j, cell, ncells)] = previous;
            }
        }
    }
}

extern "C" __global__ void acoustic_column_post_w_rrfs(
    const int nlev, const int ncells,
    const float dts, const float epssm, const int nsolve, const float *spec_cell,
    const float *zz, const float *rho_zz,
    const float *fzm, const float *fzp, const float *rdzw,
    const float *dss, const float *w,
    const float *rw, const float *rw_save,
    const float *tend_rw, const float *rs, const float *ts,
    const float *cofwr, const float *cofwz, const float *coftz,
    const float *cofwt, const float *cofrz,
    const float *a_tri, const float *alpha_tri, const float *gamma_tri,
    float *rw_p, float *rho_pp, float *rtheta_pp, float *ww_avg)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    if (k == 0 || spec_cell[cell] != 0.0f) return;
    {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float delta_saved = mpas_sub(rw_save[i], rw[i]);
        float damping = mpas_mul(dts, dss[i]);
        damping = mpas_mul(damping, mpas_add(mpas_mul(fzm[k], zz[i]), mpas_mul(fzp[k], zz[im])));
        damping = mpas_mul(damping, mpas_add(mpas_mul(fzm[k], rho_zz[i]), mpas_mul(fzp[k], rho_zz[im])));
        damping = mpas_mul(damping, w[i]);
        rw_p[i] = mpas_sub(mpas_div(mpas_sub(mpas_add(rw_p[i], delta_saved), damping),
            mpas_add(1.0f, mpas_mul(dts, dss[i]))), delta_saved);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(mpas_mul(
            0.5f, mpas_add(1.0f, epssm)), rw_p[i]));
    }
}

extern "C" __global__ void acoustic_column_post_rt_rrfs(
    const int nlev, const int ncells,
    const float dts, const float epssm, const int nsolve, const float *spec_cell,
    const float *zz, const float *rho_zz,
    const float *fzm, const float *fzp, const float *rdzw,
    const float *dss, const float *w,
    const float *rw, const float *rw_save,
    const float *tend_rw, const float *rs, const float *ts,
    const float *cofwr, const float *cofwz, const float *coftz,
    const float *cofwt, const float *cofrz,
    const float *a_tri, const float *alpha_tri, const float *gamma_tri,
    float *rw_p, float *rho_pp, float *rtheta_pp, float *ww_avg)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    if (cell >= nsolve) return;
    if (spec_cell[cell] != 0.0f) return;
    {
        const int i = C2(k, cell, ncells);
        rho_pp[i] = mpas_sub(rs[i], mpas_mul(cofrz[k], mpas_sub(
            rw_p[C2(k + 1, cell, ncells)], rw_p[i])));
        rtheta_pp[i] = mpas_sub(ts[i], mpas_mul(rdzw[k], mpas_sub(
            mpas_mul(coftz[C2(k + 1, cell, ncells)],
                rw_p[C2(k + 1, cell, ncells)]),
            mpas_mul(coftz[i], rw_p[i]))));
    }
}
"""

COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
_MODULES: dict[tuple[int, str], Any] = {}


# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
def _cupy() -> Any:
    from .cuda_backend import require_cuda
    require_cuda(min_compute=(12, 0))
    import cupy as cp
    return cp


# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
def _module(source: str, key: str) -> Any:
    """Load direct NVRTC output, preserving native subnormal arithmetic."""
    from cupy.cuda import compiler, function
    cp_module = _cupy()
    cache_key = (cp_module.cuda.Device().id, key)
    if cache_key not in _MODULES:
        image, _ = compiler.compile_using_nvrtc(source, options=_hex_numerics.options(COMPILE_OPTIONS), cache_in_memory=True)
        module = function.Module()
        module.load(image)
        _MODULES[cache_key] = module
    return _MODULES[cache_key]


# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
def _kernel(name: str, kernel_cache: Any | None = None) -> Any:
    # RRFS cannot use RawModule: CuPy appends -ftz=true on that route.
    from .cuda_compile_rrfs import kernel
    return kernel(_CUDA_SOURCE,"hexcore.cuda_acoustic_rrfs",name,kernel_cache)


# Resident mesh tables are immutable. A new mesh supplies new arrays, whose
# weak references also retire their cached layouts. Separate producer streams
# prevent another stream reading an unfinished asynchronous transpose.
_CELL_STENCILS: dict[tuple[int, ...], tuple[Any, Any]] = {}


def _cell_stencil_layout(edges, signs, ncells):
    """Cache slot-major geometry, including the native dummy cell row."""
    cp = _cupy()
    arrays = (edges, signs)
    key = (int(cp.cuda.Device().id), int(cp.cuda.get_current_stream().ptr),
           int(ncells), *(id(a) for a in arrays))
    cached = _CELL_STENCILS.get(key)
    if cached is not None and all(ref() is a for ref, a in zip(cached[0], arrays)):
        return cached[1]
    layout = []
    for array in arrays:
        if array.shape[0] == ncells:
            layout.append(cp.ascontiguousarray(array.T))
        elif array.shape[0] == ncells - 1:
            table = cp.zeros((array.shape[1], ncells), dtype=array.dtype)
            table[:, :-1] = array.T
            layout.append(table)
        else:
            raise ValueError('native stencil extent differs from physical plus dummy')
    layout = tuple(layout)
    def release(_):
        _CELL_STENCILS.pop(key, None)
    _CELL_STENCILS[key] = (tuple(weakref.ref(a, release) for a in arrays), layout)
    return layout


def _launch_columns(kernel: Any, count: int, arguments: tuple[Any, ...]) -> None:
    # One warp per block exposes more independent columns on small meshes.
    threads = 32
    kernel(((count + threads - 1) // threads,), (threads,), arguments)


# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
def compute_vertical_implicit_coefficients_cuda_rrfs(
    *,
    dts: float,
    cell_solve_count: int | None = None,
    epssm: float,
    zz: Any,
    cqw: Any,
    pressure: Any,
    theta: Any,
    rho_base: Any,
    rho_theta_base: Any,
    pressure_base: Any,
    rho_theta_perturbation: Any,
    qtot: Any,
    rdzw: Any,
    fzm: Any,
    fzp: Any,
    rdzu: Any,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
    kernel_cache: Any | None = None,
    initial_coefficients: CudaVerticalImplicitCoefficients | None = None,
) -> CudaVerticalImplicitCoefficients:
    """Fork mpas_atm_time_integration.F:2225-2366 at 1d6e370e: scaled coefficients."""

    cp_module = _cupy()
    zz = _float32(zz, "zz")
    if zz.ndim != 2:
        raise ValueError("zz must have shape (nVertLevels,nCells)")
    nlev, ncells = map(int, zz.shape)
    nsolve = ncells if cell_solve_count is None else int(cell_solve_count)
    if not 0 <= nsolve <= ncells:
        raise ValueError("cell_solve_count outside resident cells")
    cell_shape = (nlev, ncells)
    inputs: dict[str, Any] = {}
    for name, value in {
        "cqw": cqw,
        "pressure": pressure,
        "theta": theta,
        "rho_base": rho_base,
        "rho_theta_base": rho_theta_base,
        "pressure_base": pressure_base,
        "rho_theta_perturbation": rho_theta_perturbation,
        "qtot": qtot,
    }.items():
        array = _float32(value, name)
        if array.shape != cell_shape:
            raise ValueError(f"{name} shape {array.shape} != {cell_shape}")
        inputs[name] = array
    vertical: dict[str, Any] = {}
    for name, value in {"rdzw": rdzw, "fzm": fzm, "fzp": fzp, "rdzu": rdzu}.items():
        array = _float32(value, name)
        if array.shape != (nlev,):
            raise ValueError(f"{name} shape {array.shape} != ({nlev},)")
        vertical[name] = array

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
    def zero() -> Any:
        return cp_module.zeros(cell_shape, dtype=cp_module.float32)

    coftz = cp_module.zeros((nlev + 1, ncells), dtype=cp_module.float32)
    cofrz = cp_module.empty((nlev,), dtype=cp_module.float32)
    result = CudaVerticalImplicitCoefficients(
        cofwr=zero(),
        cofwz=zero(),
        coftz=coftz,
        cofwt=zero(),
        cofrz=cofrz,
        a_tri=zero(),
        b_tri=zero(),
        c_tri=zero(),
        alpha_tri=zero(),
        gamma_tri=zero(),
    )
    # Fork 2297-2307 leaves cofwr/cofwz at level 1 untouched. All cell
    # coefficient arrays also retain entry storage outside the solve range.
    if initial_coefficients is not None:
        for name in ("cofwr", "cofwz", "coftz", "cofwt", "a_tri", "alpha_tri", "gamma_tri"):
            entry = _float32(getattr(initial_coefficients, name), name)
            target = getattr(result, name)
            if entry.shape != target.shape:
                raise ValueError(f"entry coefficient {name} shape differs from resident cells")
            target[...] = entry
    _launch_1d(
        _kernel("acoustic_cofrz_rrfs", kernel_cache),
        nlev,
        (np.int32(nlev), np.float32(dts), np.float32(epssm), vertical["rdzw"], cofrz),
    )
    coefficient_args = (
        np.int32(nlev),
        np.int32(ncells),
        np.float32(dts),
        np.float32(epssm),
        np.float32(gravity),
        np.float32(rgas),
        np.float32(cp),
        zz,
        inputs["cqw"],
        inputs["pressure"],
        inputs["theta"],
        inputs["rho_base"],
        inputs["rho_theta_base"],
        inputs["pressure_base"],
        inputs["rho_theta_perturbation"],
        inputs["qtot"],
        vertical["rdzw"],
        vertical["fzm"],
        vertical["fzp"],
        vertical["rdzu"],
        np.int32(nsolve),
        result.cofwr,
        result.cofwz,
        result.coftz,
        result.cofwt,
        result.cofrz,
        result.a_tri,
        result.b_tri,
        result.c_tri,
        result.alpha_tri,
        result.gamma_tri,
    )
    # Each pass completes before its neighboring-level consumers run.
    _launch_1d(_kernel("acoustic_coefficients_rrfs", kernel_cache),
               nlev * ncells, coefficient_args)
    _launch_1d(_kernel("acoustic_matrix_rrfs", kernel_cache),
               nlev * ncells, coefficient_args)
    _launch_columns(_kernel("acoustic_factor_rrfs", kernel_cache),
                    ncells, coefficient_args)
    return result


# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2984 at 1d6e370e.
def advance_acoustic_step_cuda_rrfs(
    mesh: Any,
    state: CudaAcousticState,
    forcing: CudaAcousticForcing,
    coefficients: CudaVerticalImplicitCoefficients,
    *,
    dts: float,
    context: Any,
    cell_solve_count: int | None = None,
    edge_start: int = 0,
    edge_end: int | None = None,
    cell_start: int = 0,
    cell_end: int | None = None,
    spec_zone_mask_cell: Any | None = None,
    spec_zone_mask_edge: Any | None = None,
    small_step: int,
    epssm: float,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
    in_place: bool = True,
    kernel_cache: Any | None = None,
) -> CudaAcousticState:
    """Fork mpas_atm_time_integration.F:2646-2984 at 1d6e370e: global/LAM substep."""

    if small_step < 1:
        raise ValueError("small_step must be at least one")
    out = state if in_place else state.copy()
    nlev, nedges = map(int, out.ru_p.shape)
    ncells = int(out.rho_pp.shape[1])
    edge_end = nedges if edge_end is None else int(edge_end)
    cell_end = ncells if cell_end is None else int(cell_end)
    if not 0 <= cell_start <= cell_end <= ncells:
        raise ValueError("acoustic cell interval outside resident cells")
    if not 0 <= edge_start <= edge_end <= nedges:
        raise ValueError("acoustic edge interval outside resident edges")
    if out.rho_pp.shape != (nlev, ncells) or out.rw_p.shape != (nlev + 1, ncells):
        raise ValueError("acoustic state shapes are inconsistent")
    nsolve = ncells if cell_solve_count is None else int(cell_solve_count)
    if not 0 <= nsolve <= ncells:
        raise ValueError("cell_solve_count outside resident cells")
    cp_module = _cupy()
    spec_cell = (cp_module.zeros(ncells, dtype=cp_module.float32) if spec_zone_mask_cell is None
                 else _float32(spec_zone_mask_cell, "spec_zone_mask_cell"))
    spec_edge = (cp_module.zeros(nedges, dtype=cp_module.float32) if spec_zone_mask_edge is None
                 else _float32(spec_zone_mask_edge, "spec_zone_mask_edge"))
    if spec_cell.shape != (ncells,) or spec_edge.shape != (nedges,):
        raise ValueError("specified-zone masks differ from mesh")
    cells_on_edge = _int32(_mesh_value(mesh, "cells_on_edge"), "cells_on_edge")
    edges_on_cell = _int32(_mesh_value(mesh, "edges_on_cell"), "edges_on_cell")
    counts = _int32(_mesh_value(mesh, "n_edges_on_cell"), "n_edges_on_cell")
    signs = _float32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell")
    dv_edge = _float32(_mesh_value(mesh, "dv_edge"), "dv_edge")
    inv_dc_edge = _float32(context.inv_dc_edge, "inv_dc_edge")
    inv_area_cell = _float32(context.inv_area_cell, "inv_area_cell")
    max_edges = int(edges_on_cell.shape[1])
    edges_on_cell, signs = _cell_stencil_layout(edges_on_cell, signs, ncells)
    fzm = _float32(fzm, "fzm")
    fzp = _float32(fzp, "fzp")
    rdzw = _float32(rdzw, "rdzw")

    _launch_1d(
        _kernel("acoustic_ru_rrfs", kernel_cache),
        nlev * nedges,
        (
            np.int32(nlev),
            np.int32(nedges),
            np.int32(ncells),
            np.int32(small_step),
            np.float32(dts),
            np.int32(edge_start), np.int32(edge_end),
            np.float32(gravity),
            np.float32(rgas),
            np.float32(cp),
            cells_on_edge,
            inv_dc_edge,
            _float32(forcing.zz, "forcing.zz"),
            _float32(forcing.exner, "forcing.exner"),
            _float32(forcing.cqu, "forcing.cqu"),
            _float32(forcing.zxu, "forcing.zxu"),
            _float32(forcing.tend_ru, "forcing.tend_ru"),
            spec_edge, np.int32(nsolve),
            out.rho_pp,
            out.rtheta_pp,
            out.ru_p,
            out.ru_avg,
        ),
    )
    _launch_1d(
        _kernel("acoustic_prepare_rrfs", kernel_cache),
        (nlev + 1) * ncells,
        (
            np.int32(nlev),
            np.int32(ncells),
            np.int32(small_step), np.int32(nsolve),
            np.int32(cell_start), np.int32(cell_end),
            out.rw_p,
            out.rtheta_pp,
            out.rtheta_pp_old,
            out.rho_pp,
            out.ww_avg,
        ),
    )
    cp = _cupy()
    rs = cp.empty((nlev, ncells), dtype=cp.float32)
    ts = cp.empty_like(rs)
    _launch_1d(
        _kernel("acoustic_rs_ts_rrfs", kernel_cache),
        nlev * ncells,
        (
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nedges),
            np.int32(max_edges),
            np.float32(dts),
            np.float32(epssm),
            counts,
            edges_on_cell,
            cells_on_edge,
            signs,
            dv_edge,
            inv_area_cell, spec_cell, np.int32(nsolve),
            _float32(forcing.theta_m, "forcing.theta_m"),
            rdzw,
            coefficients.cofrz,
            coefficients.coftz,
            out.ru_p,
            out.rw_p,
            out.rho_pp,
            out.rtheta_pp,
            _float32(forcing.tend_rho, "forcing.tend_rho"),
            _float32(forcing.tend_rt, "forcing.tend_rt"),
            rs,
            ts,
        ),
    )
    solve_arguments = (
        np.int32(nlev),
        np.int32(ncells),
        np.float32(dts),
        np.float32(epssm), np.int32(nsolve), spec_cell,
        _float32(forcing.zz, "forcing.zz"),
        _float32(forcing.rho_zz, "forcing.rho_zz"),
        fzm,
        fzp,
        rdzw,
        _float32(forcing.dss, "forcing.dss"),
        _float32(forcing.w, "forcing.w"),
        _float32(forcing.rw, "forcing.rw"),
        _float32(forcing.rw_save, "forcing.rw_save"),
        _float32(forcing.tend_rw, "forcing.tend_rw"),
        rs,
        ts,
        coefficients.cofwr,
        coefficients.cofwz,
        coefficients.coftz,
        coefficients.cofwt,
        coefficients.cofrz,
        coefficients.a_tri,
        coefficients.alpha_tri,
        coefficients.gamma_tri,
        out.rw_p,
        out.rho_pp,
        out.rtheta_pp,
        out.ww_avg,
    )
    # Separate launches supply the inter-level read barriers around the sweep.
    _launch_1d(_kernel("acoustic_column_pre_rrfs", kernel_cache), nlev * ncells, solve_arguments)
    _launch_1d(_kernel("acoustic_column_solve_rrfs", kernel_cache), ncells, solve_arguments)
    _launch_1d(_kernel("acoustic_column_post_w_rrfs", kernel_cache), nlev * ncells, solve_arguments)
    _launch_1d(_kernel("acoustic_column_post_rt_rrfs", kernel_cache), nlev * ncells, solve_arguments)
    return out

