"""RRFS MYJ PBL and surface layer at fork 1d6e370e.

Authority: MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:1-1784
and MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:1-1302.
Common WRF arithmetic blocks are reused from the pinned WOOF kernels;
fork precision, constants, terrain accumulation and shelter winds are separate.
Arrays use level-major (levels, columns) contiguous single precision storage.
PINT includes the top interface. LOWLYR is native one-based indexing.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from importlib.resources import files
import numpy as np

FORK_PIN = "1d6e370e7b43a332074809f170532a216c2212fb"
CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
PBL_COLUMNS = ("dz", "u", "v", "t", "th", "exner", "qv", "qc", "qi", "pmid")
PBL_SURFACE = ("ht", "lowlyr", "psfc", "ustar", "tsk", "chklowq", "xland", "sice", "snow", "akhs", "akms", "elflx", "uz0", "vz0")
PBL_STATE = ("thz0", "qz0", "qsfc", "ct")
PBL_OUTPUT_COLUMNS = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten", "el_myj", "exch_h")
PBL_OUTPUT_SURFACE = ("pblh", "kpbl", "mixht")
SFC_COLUMNS = ("dz", "tke")
SFC_SURFACE = ("u1", "v1", "t1", "th1", "qv1", "qc1", "p1", "psfc", "tsk", "xland", "mavail", "z0base", "ht", "lowlyr")
SFC_STATE = ("ustar", "znt", "thz0", "qz0", "uz0", "vz0", "qsfc", "akhs", "akms")
SFC_OUTPUTS = ("rmol", "ct", "pblh", "rib", "chs", "chs2", "cqs2", "hfx", "qfx", "flx_lh", "flhc", "flqc", "qgh", "cpm", "u10", "v10", "t02", "th02", "tshltr", "th10", "q02", "qshltr", "q10", "pshltr")

INIT_SOURCE = r'''
// Keep divisors in constant memory so every target retains division.
#ifndef HEX_DIVISOR_40000000_DEFINED
#define HEX_DIVISOR_40000000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_40000000 = 2.0f;
#else
static const float hex_divisor_40000000 = 2.0f;
#endif
#endif
#ifndef HEX_DIVISOR_461C4000_DEFINED
#define HEX_DIVISOR_461C4000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_461c4000 = 10000.0f;
#else
static const float hex_divisor_461c4000 = 10000.0f;
#endif
#endif

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:351-351
extern "C" __global__ void myj_clear_el(float* memory, int first, int active,
    int columns, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) memory[(size_t)(i/active)*columns+first+i%active]=0.0f;
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:232-303
extern "C" __global__ void myj_lowest(const int* lowlyr, const float* pint,
    const float* u,const float* v,const float* t,const float* th,
    const float* qv,const float* qc,const float* pmid,float* psfc,
    float* u1,float* v1,float* t1,float* th1,float* qv1,float* qc1,float* p1,int n) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=n)return;
    size_t k=(size_t)(lowlyr[c]-1)*n+c;
    psfc[c]=pint[k];u1[c]=u[k];v1[c]=v[k];t1[c]=t[k];th1[c]=th[k];
    qv1[c]=qv[k];qc1[c]=qc[k];p1[c]=pmid[k];
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1666-1676
extern "C" __global__ void myj_driver_tke_init(float* tke,int count) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<count)tke[c]=0.2f;
}

// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:1066-1295
extern "C" __global__ void myjsfc_tables(float* psim1, float* psih1,
    float* psim2, float* psih2, float* scalars) {
    if (blockIdx.x || threadIdx.x) return;
    float dzeta = (1.0f - (-5.0f)) / hex_divisor_461c4000;
    float zeta = -5.0f, ztmax = -5.0f;
    for (int k = 0; k < 10001; ++k) {
        float pm, ph;
        if (zeta < 0.0f) {
            float x = sqrtf(sqrtf(1.0f - 16.0f * zeta));
            pm = -2.0f * gfk_log((x + 1.0f) / hex_divisor_40000000)
                 - gfk_log((x * x + 1.0f) / hex_divisor_40000000)
                 + 2.0f * glibc_atanf(x) - (3.1415926f / hex_divisor_40000000);
            ph = -2.0f * gfk_log((x * x + 1.0f) / hex_divisor_40000000);
        } else {
            pm = 0.7f * zeta + 0.75f * zeta * (6.0f - 0.35f * zeta)
                 * gfk_exp_fma(-0.35f * zeta);
            ph = pm;
        }
        psim1[k] = pm; psih1[k] = ph; psim2[k] = pm; psih2[k] = ph;
        if (k == 10000) ztmax = zeta;
        zeta = zeta + dzeta;
    }
    scalars[0] = dzeta; scalars[1] = dzeta;
    scalars[2] = -5.0f; scalars[3] = ztmax - 1.0e-6f;
    scalars[4] = -5.0f; scalars[5] = ztmax - 1.0e-6f;
    scalars[6] = 1.0f; scalars[7] = 1.0f;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:1120-1177
extern "C" __global__ void myjsfc_init(int* lowlyr, float* ustar, int n, int restart) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    lowlyr[c] = 1;
    if (!restart) ustar[c] = 0.1f;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:1737-1777
extern "C" __global__ void myjpbl_init(float* tke, float* du, float* dv,
    float* dth, float* dq, float* exch, int count, int restart) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= count || restart) return;
    tke[c] = 0.2f;
    du[c] = 0.0f; dv[c] = 0.0f; dth[c] = 0.0f; dq[c] = 0.0f; exch[c] = 0.0f;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:1567-1652
// Private VDIFX is retained even though MYJPBL has no call to it at this pin.
__device__ void myj_vdifx(float dt, int lmh, float rkhs, float ct,
    float* dust1, float* dust2, const float* rkh, const float* z, const float* rho) {
    float cm[MYJ_LEVELS], cr[MYJ_LEVELS], dtoz[MYJ_LEVELS];
    float rs1[MYJ_LEVELS], rs2[MYJ_LEVELS];
    for (int k = 0; k < lmh - 1; ++k) {
        dtoz[k] = dt / (z[k] - z[k + 1]);
        cr[k] = -dtoz[k] * rkh[k];
    }
    cm[0] = dtoz[0] * rkh[0] + rho[0];
    rs1[0] = dust1[0] * rho[0]; rs2[0] = dust2[0] * rho[0];
    for (int k = 1; k < lmh - 1; ++k) {
        float cf = -dtoz[k] * rkh[k - 1] / cm[k - 1];
        cm[k] = -cr[k - 1] * cf + (rkh[k - 1] + rkh[k]) * dtoz[k] + rho[k];
        rs1[k] = -rs1[k - 1] * cf + dust1[k] * rho[k];
        rs2[k] = -rs2[k - 1] * cf + dust2[k] * rho[k];
    }
    int k = lmh - 1;
    float ds = dt / (z[k] - z[k + 1]);
    float cf = -ds * rkh[k - 1] / cm[k - 1];
    float cmdb = -cr[k - 1] * cf + rkh[k - 1] * ds + rho[k];
    dust1[k] = (-rs1[k - 1] * cf + dust1[k] * rho[k]) / cmdb;
    dust2[k] = (-rs2[k - 1] * cf + dust2[k] * rho[k]) / cmdb;
    for (--k; k >= 0; --k) {
        float rcml = 1.0f / cm[k];
        dust1[k] = (-cr[k] * dust1[k + 1] + rs1[k]) * rcml;
        dust2[k] = (-cr[k] * dust2[k + 1] + rs2[k]) * rcml;
    }
}
'''

PBL_SOURCE = r'''
// Keep divisors in constant memory so every target retains division.
#ifndef HEX_DIVISOR_40000000_DEFINED
#define HEX_DIVISOR_40000000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_40000000 = 2.0f;
#else
static const float hex_divisor_40000000 = 2.0f;
#endif
#endif
#ifndef HEX_DIVISOR_40400000_DEFINED
#define HEX_DIVISOR_40400000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_40400000 = 3.0f;
#else
static const float hex_divisor_40400000 = 3.0f;
#endif
#endif
#ifndef HEX_DIVISOR_43888000_DEFINED
#define HEX_DIVISOR_43888000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_43888000 = 273.0f;
#else
static const float hex_divisor_43888000 = 273.0f;
#endif
#endif
#ifndef HEX_DIVISOR_438F8000_DEFINED
#define HEX_DIVISOR_438F8000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_438f8000 = 287.0f;
#else
static const float hex_divisor_438f8000 = 287.0f;
#endif
#endif

// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:141-784
#ifndef MYJ_KMAX
#define MYJ_KMAX MYJ_LEVELS
#endif

// MPAS-Model-rrfs/src/framework/mpas_constants.F:46-51
#define MYJP_G       9.80616f
#define MYJP_RD      287.0f
#define MYJP_CP      (7.0f * 287.0f / hex_divisor_40000000)
#define MYJP_XLV     2.5e6f
#define MYJP_XLS     2.85e6f
#define MYJP_P608    (461.6f / hex_divisor_438f8000 - 1.0f)
#define MYJP_PQ0     379.90516f
#define MYJP_A2      17.2693882f
#define MYJP_A3      273.16f
#define MYJP_A4      35.86f
#define MYJP_EPSQ2   0.2f

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:38-136
#define PBL_VKARMAN  0.4f
#define PBL_CAPA     (MYJP_RD / MYJP_CP)
#define PBL_RLIVWV   (MYJP_XLS / MYJP_XLV)
#define PBL_ELOCP    (2.72e6f / MYJP_CP)
#define PBL_EPS1     1.0e-12f
#define PBL_EPS2     0.0f
#define PBL_EPSL     0.32f
#define PBL_EPSRU    1.0e-7f
#define PBL_EPSRS    1.0e-7f
#define PBL_EPSTRB   1.0e-24f
#define PBL_FH       1.01f
#define PBL_ALPH     0.30f
#define PBL_BETA     (1.0f / hex_divisor_43888000)
#define PBL_EL0MAX   1000.0f
#define PBL_EL0MIN   1.0f
#define PBL_ELFC     (0.23f * 0.5f)
#define PBL_A1       0.659888514560862645f
#define PBL_A2X      0.6574209922667784586f
#define PBL_B1       11.87799326209552761f
#define PBL_B2       7.226971804046074028f
#define PBL_C1       0.000830955950095854396f
#define PBL_ELZ0     0.0f
#define PBL_ESQ      5.0f
#define PBL_SEAFC    0.98f
#define PBL_PQ0SEA   (MYJP_PQ0 * PBL_SEAFC)
#define PBL_BTG      (PBL_BETA * MYJP_G)
#define PBL_RB1      (1.0f / PBL_B1)
#define PBL_ADNH (9.0f*PBL_A1*PBL_A2X*PBL_A2X*(12.0f*PBL_A1+3.0f*PBL_B2)*PBL_BTG*PBL_BTG)
#define PBL_ADNM (18.0f*PBL_A1*PBL_A1*PBL_A2X*(PBL_B2-3.0f*PBL_A2X)*PBL_BTG)
#define PBL_ANMH (-9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_BTG*PBL_BTG)
#define PBL_ANMM (-3.0f*PBL_A1*PBL_A2X*(3.0f*PBL_A2X+3.0f*PBL_B2*PBL_C1+18.0f*PBL_A1*PBL_C1-PBL_B2)*PBL_BTG)
#define PBL_BDNH (3.0f*PBL_A2X*(7.0f*PBL_A1+PBL_B2)*PBL_BTG)
#define PBL_BDNM (6.0f*PBL_A1*PBL_A1)
#define PBL_BEQH (PBL_A2X*PBL_B1*PBL_BTG+3.0f*PBL_A2X*(7.0f*PBL_A1+PBL_B2)*PBL_BTG)
#define PBL_BEQM (-PBL_A1*PBL_B1*(1.0f-3.0f*PBL_C1)+6.0f*PBL_A1*PBL_A1)
#define PBL_BNMH (-PBL_A2X*PBL_BTG)
#define PBL_BNMM (PBL_A1*(1.0f-3.0f*PBL_C1))
#define PBL_BSHH (9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_BTG)
#define PBL_BSHM (18.0f*PBL_A1*PBL_A1*PBL_A2X*PBL_C1)
#define PBL_BSMH (-3.0f*PBL_A1*PBL_A2X*(3.0f*PBL_A2X+3.0f*PBL_B2*PBL_C1+12.0f*PBL_A1*PBL_C1-PBL_B2)*PBL_BTG)
#define PBL_CESH PBL_A2X
#define PBL_CESM (PBL_A1*(1.0f-3.0f*PBL_C1)) // free term in the equilibrium equation for (L/Q)**2
#define PBL_AEQH (9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_B1*PBL_BTG*PBL_BTG \
                  + 9.0f*PBL_A1*PBL_A2X*PBL_A2X*(12.0f*PBL_A1+3.0f*PBL_B2)*PBL_BTG*PBL_BTG)
#define PBL_AEQM (3.0f*PBL_A1*PBL_A2X*PBL_B1*(3.0f*PBL_A2X+3.0f*PBL_B2*PBL_C1+18.0f*PBL_A1*PBL_C1-PBL_B2)*PBL_BTG \
                  + 18.0f*PBL_A1*PBL_A1*PBL_A2X*(PBL_B2-3.0f*PBL_A2X)*PBL_BTG) // forbidden turbulence area / near isotropy
#define PBL_REQU  (-PBL_AEQH/PBL_AEQM)
#define PBL_EPSGH 1.0e-9f
#define PBL_EPSGM (PBL_REQU*PBL_EPSGH)
#define PBL_UBRYL ((18.0f*PBL_REQU*PBL_A1*PBL_A1*PBL_A2X*PBL_B2*PBL_C1*PBL_BTG \
                    + 9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_B2*PBL_BTG*PBL_BTG) \
                   / (PBL_REQU*PBL_ADNM+PBL_ADNH))
#define PBL_UBRY  ((1.0f+PBL_EPSRS)*PBL_UBRYL)
#define PBL_UBRY3 (3.0f*PBL_UBRY)
#define PBL_AUBH  (27.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_B2*PBL_BTG*PBL_BTG - PBL_ADNH*PBL_UBRY3)
#define PBL_AUBM  (54.0f*PBL_A1*PBL_A1*PBL_A2X*PBL_B2*PBL_C1*PBL_BTG - PBL_ADNM*PBL_UBRY3)
#define PBL_BUBH  ((9.0f*PBL_A1*PBL_A2X+3.0f*PBL_A2X*PBL_B2)*PBL_BTG - PBL_BDNH*PBL_UBRY3)
#define PBL_BUBM  (18.0f*PBL_A1*PBL_A1*PBL_C1 - PBL_BDNM*PBL_UBRY3)
#define PBL_CUBR  (1.0f - PBL_UBRY3)
#define PBL_RCUBR (1.0f/PBL_CUBR)

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:141-784 at 1d6e370e
__device__ void myjpbl_apply(
    // full bottom-up columns, (nz, n)
    const real* __restrict__ dz_a,
    const real* __restrict__ u_a, const real* __restrict__ v_a,
    const real* __restrict__ t_a, const real* __restrict__ th_a,
    const real* __restrict__ exner_a, const real* __restrict__ qv_a,
    const real* __restrict__ qc_a, const real* __restrict__ qi_a,
    const real* __restrict__ p_a,
    real* __restrict__ tke_a,            // TKE_MYJ, inout (nz, n)
    // surface (n)
    const real* __restrict__ ht_a,
    const int* __restrict__ lowlyr_a,
    const real* __restrict__ psfc_a, const real* __restrict__ ust_a,
    const real* __restrict__ tsk_a, const real* __restrict__ chklowq_a,
    const real* __restrict__ xland_a, const real* __restrict__ sice_a,
    const real* __restrict__ snow_a, const real* __restrict__ akhs_a,
    const real* __restrict__ akms_a, const real* __restrict__ elflx_a,
    const real* __restrict__ uz0_a, const real* __restrict__ vz0_a,
    // inout surface (n)
    real* __restrict__ thz0_a, real* __restrict__ qz0_a,
    real* __restrict__ qsfc_a, real* __restrict__ ct_a,
    // outputs, bottom-up (nz, n)
    real* __restrict__ rublten_a, real* __restrict__ rvblten_a,
    real* __restrict__ rthblten_a, real* __restrict__ rqvblten_a,
    real* __restrict__ rqcblten_a, real* __restrict__ rqiblten_a,
    real* __restrict__ el_myj_a, real* __restrict__ exch_h_a,
    // outputs (n)
    real* __restrict__ pblh_a, int* __restrict__ kpbl_a,
    real* __restrict__ mixht_a,
    real dtturbl, int flqi, int nz, int n,
    int col, real* carry_el, real* carry_akm, real* carry_akh, int momentum_lmh)
{
    size_t st = (size_t)n;
#define MYJ_UP(a, kup) a[(size_t)(kup) * st + col]

    const int lmh = nz - lowlyr_a[col] + 1;
    const int nzm = lmh - 1;
    real rdtturbl = 1.0f / dtturbl;
    real dtdif = dtturbl;

    // Persistent MYJ-layout columns.
    real zh[MYJ_KMAX + 1];
    real uk[MYJ_KMAX], vk[MYJ_KMAX], tk[MYJ_KMAX];
    real the[MYJ_KMAX], qk[MYJ_KMAX], cwm[MYJ_KMAX], qcik[MYJ_KMAX];
    real q2[MYJ_KMAX], rhok[MYJ_KMAX];
    real gm[MYJ_KMAX], gh[MYJ_KMAX], el[MYJ_KMAX];
    real akm[MYJ_KMAX], akh[MYJ_KMAX];
    // Scratch, reused between phases (documented at each use).
    real s1[MYJ_KMAX], s2[MYJ_KMAX], s3[MYJ_KMAX], s4[MYJ_KMAX];

    // ---- flip on load --------------------------------------
    zh[nz] = ht_a[col];  // Native absolute heights retain terrain rounding.
    for (int m = nz - 1; m >= 0; --m) {
        zh[m] = zh[m + 1] + MYJ_UP(dz_a, nz - 1 - m);
    }
    for (int m = 0; m < nz; ++m) {
        int kup = nz - 1 - m;                  // KFLIP
        uk[m] = MYJ_UP(u_a, kup);
        vk[m] = MYJ_UP(v_a, kup);
        tk[m] = MYJ_UP(t_a, kup);
        real thx = MYJ_UP(th_a, kup);
        real ratiomx = MYJ_UP(qv_a, kup);
        qk[m] = ratiomx / (1.0f + ratiomx);
        real cw = MYJ_UP(qc_a, kup);
        qcik[m] = flqi ? MYJ_UP(qi_a, kup) : 0.0f;
        if (flqi) cw = cw + qcik[m];
        cwm[m] = cw;
        the[m] = (cw * (-PBL_ELOCP / tk[m]) + 1.0f) * thx;
        q2[m] = 2.0f * MYJ_UP(tke_a, kup);
    }

    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:787-1006
    // ================= MIXLEN ================================
    int lpbl = lmh;
    for (int kf = lmh - 1; kf >= 1; --kf) {    // K=LMH-1,1,-1
        if (q2[kf - 1] <= MYJP_EPSQ2 * PBL_FH) { lpbl = kf; goto mixlen_110; }
    }
    lpbl = 1;
mixlen_110:
    {
        real pblh = zh[lpbl] - zh[lmh]; // Z(LPBL+1)-Z(LMH+1)
        pblh_a[col] = pblh;
    }
    // DTH lives in s1; q1 lives in s2.
    for (int m = 0; m < nzm; ++m) s1[m] = the[m] - the[m + 1];
    {
        real ct = ct_a[col];
        for (int kf = lmh - 2; kf >= 1; --kf) {
            if (s1[kf - 1] > 0.0f && s1[kf] <= 0.0f) {
                s1[kf - 1] = s1[kf - 1] + ct;
                break;
            }
        }
        ct_a[col] = 0.0f;
    }
    for (int m = 0; m < nzm; ++m) {
        real rdz = 2.0f / (zh[m] - zh[m + 2]);
        real gml = ((uk[m] - uk[m + 1]) * (uk[m] - uk[m + 1])
                    + (vk[m] - vk[m + 1]) * (vk[m] - vk[m + 1])) * rdz * rdz;
        gm[m] = fmaxf(gml, PBL_EPSGM);
        real tem = (tk[m] + tk[m + 1]) * 0.5f;
        real thm = (the[m] + the[m + 1]) * 0.5f;
        real a = thm * MYJP_P608;
        real b = (PBL_ELOCP / tem - 1.0f - MYJP_P608) * thm;
        real ghl = (s1[m] * ((qk[m] + qk[m + 1] + cwm[m] + cwm[m + 1])
                             * (0.5f * MYJP_P608) + 1.0f)
                    + (qk[m] - qk[m + 1] + cwm[m] - cwm[m + 1]) * a
                    + (cwm[m] - cwm[m + 1]) * b) * rdz;
        if (fabsf(ghl) <= PBL_EPSGH) ghl = PBL_EPSGH;
        gh[m] = ghl;
    }
    // ELM lives in akm, REL in akh: both are dead until DIFCOF.
    int lmxl = lmh;
    for (int m = 0; m < nzm; ++m) {
        real gml = gm[m], ghl = gh[m], eloq2x;
        if (ghl >= PBL_EPSGH) {
            if (gml / ghl <= PBL_REQU) {
                akm[m] = PBL_EPSL;
                lmxl = m + 1;
                continue;
            }
            double aubr = (double)fminf((PBL_AUBM * gml + PBL_AUBH * ghl) * ghl, 1.0e-18f);
            real bubr = PBL_BUBM * gml + PBL_BUBH * ghl;
            // RRFS MIXLEN: BUBR*BUBR rounds in real32 before widening.
            double bubrsq = (double)(bubr * bubr);
            double aubrcubr = aubr * (double)PBL_CUBR;
            double qol2st = ((double)(-0.5f * bubr)
                           + sqrt(bubrsq * (double)0.25f - aubrcubr))
                          * (double)PBL_RCUBR;
            eloq2x = 1.0f / fmaxf(PBL_EPSGH, (float)qol2st);
        } else {
            real aden = (PBL_ADNM * gml + PBL_ADNH * ghl) * ghl;
            real bden = PBL_BDNM * gml + PBL_BDNH * ghl;
            real qol2un = -0.5f * bden + sqrtf(bden * bden * 0.25f - aden);
            eloq2x = 1.0f / (qol2un + PBL_EPSRU);
        }
        akm[m] = fmaxf(sqrtf(eloq2x * q2[m]), PBL_EPSL);
    }
    if (akm[lmh - 2] == PBL_EPSL) lmxl = lmh;
    mixht_a[col] = zh[lmxl - 1] - zh[lmh];     // BLMX
    for (int m = 0; m < lmh; ++m) s2[m] = 0.0f;
    for (int m = lpbl - 1; m < lmh; ++m) s2[m] = sqrtf(q2[m]);
    {
        real szq = 0.0f, sq = 0.0f;
        for (int m = 0; m < nzm; ++m) {
            real qdzl = (s2[m] + s2[m + 1]) * (zh[m + 1] - zh[m + 2]);
            szq = (zh[m + 1] + zh[m + 2] - zh[lmh] - zh[lmh]) * qdzl + szq;
            sq = qdzl + sq;
        }
        real el0 = fminf(PBL_ALPH * szq * 0.5f / sq, PBL_EL0MAX);
        el0 = fmaxf(el0, PBL_EL0MIN);
        int lpblm = max(lpbl - 1, 1);
        for (int m = 0; m < lpblm; ++m) { // above the PBL top
            el[m] = fminf((zh[m] - zh[m + 2]) * PBL_ELFC, akm[m]);
            akh[m] = el[m] / akm[m];           // REL
        }
        if (lpbl < lmh) { // inside the PBL
            for (int m = lpbl - 1; m < lmh - 1; ++m) {
                real vkrmz = (zh[m + 1] - zh[lmh]) * PBL_VKARMAN;
                el[m] = fminf(vkrmz / (vkrmz / el0 + 1.0f), akm[m]);
                akh[m] = el[m] / akm[m];
            }
        }
        for (int m = lpbl; m < lmh - 2; ++m) { // K=LPBL+1,LMH-2
            real srel = fminf(((akh[m - 1] + akh[m + 1]) * 0.5f + akh[m])
                              * 0.5f, akh[m]);
            el[m] = fmaxf(srel * akm[m], PBL_EPSL);
        }
    }

    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:1010-1211
    // ================= PRODQ2 ===============================
    {
        real ustar = ust_a[col];
        for (int m = 0; m < lmh - 1; ++m) {
            real gml = gm[m], ghl = gh[m];
            real aequ = (PBL_AEQM * gml + PBL_AEQH * ghl) * ghl;
            real bequ = PBL_BEQM * gml + PBL_BEQH * ghl;
            real eqol2 = -0.5f * bequ + sqrtf(bequ * bequ * 0.25f - aequ);
            if ((gml + ghl * ghl <= PBL_EPSTRB)
                || (ghl >= PBL_EPSGH && gml / ghl <= PBL_REQU)
                || (eqol2 <= PBL_EPS2)) {
                q2[m] = MYJP_EPSQ2;
                el[m] = PBL_EPSL;
                continue;
            }
            real anum = (PBL_ANMM * gml + PBL_ANMH * ghl) * ghl;
            real bnum = PBL_BNMM * gml + PBL_BNMH * ghl;
            real aden = (PBL_ADNM * gml + PBL_ADNH * ghl) * ghl;
            real bden = PBL_BDNM * gml + PBL_BDNH * ghl;
            real cden = 1.0f;
            real arhs = -(anum * bden - bnum * aden) * 2.0f;
            real brhs = -anum * 4.0f;
            real crhs = -bnum * 2.0f;
            real dloq1 = el[m] / sqrtf(q2[m]);
            real eloq21 = 1.0f / eqol2;
            real eloq11 = sqrtf(eloq21);
            real eloq31 = eloq21 * eloq11;
            real eloq41 = eloq21 * eloq21;
            real eloq51 = eloq21 * eloq31;
            real rden1 = 1.0f / (aden * eloq41 + bden * eloq21 + cden);
            real rhsp1 = (arhs * eloq51 + brhs * eloq31 + crhs * eloq11)
                         * rden1 * rden1;
            real eloq12 = eloq11 + (dloq1 - eloq11) * gfk_exp_fma(rhsp1 * dtturbl);
            eloq12 = fmaxf(eloq12, PBL_EPS1);
            real eloq22 = eloq12 * eloq12;
            real eloq32 = eloq22 * eloq12;
            real eloq42 = eloq22 * eloq22;
            real eloq52 = eloq22 * eloq32;
            real rden2 = 1.0f / (aden * eloq42 + bden * eloq22 + cden);
            real rhs2 = -(anum * eloq42 + bnum * eloq22) * rden2 + PBL_RB1;
            real rhsp2 = (arhs * eloq52 + brhs * eloq32 + crhs * eloq12)
                         * rden2 * rden2;
            real rhst2 = rhs2 / rhsp2;
            real eloq13 = eloq12 - rhst2
                          + (rhst2 + dloq1 - eloq12) * gfk_exp_fma(rhsp2 * dtturbl);
            eloq13 = fmaxf(eloq13, PBL_EPS1);
            if (eloq13 > PBL_EPS1) {
                q2[m] = el[m] * el[m] / (eloq13 * eloq13);
                q2[m] = fmaxf(q2[m], MYJP_EPSQ2);
                if (q2[m] == MYJP_EPSQ2) el[m] = PBL_EPSL;
            } else {
                q2[m] = MYJP_EPSQ2;
                el[m] = PBL_EPSL;
            }
        } // lower boundary, left-to-right (B1**(2./3.)*USTAR)*USTAR.
        q2[lmh - 1] = fmaxf(gfk_pow(PBL_B1, 2.0f / hex_divisor_40400000) * ustar * ustar,
                            MYJP_EPSQ2);
    }
    kpbl_a[col] = nz - lpbl + 1; // KPBL=KTE-LPBL+1

    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:1215-1369
    // ================= DIFCOF ==============================
    for (int m = 0; m < lmh - 1; ++m) {
        real ell = el[m];
        real eloq2 = ell * ell / q2[m];
        real eloq4 = eloq2 * eloq2;
        real gml = gm[m], ghl = gh[m];
        real aden = (PBL_ADNM * gml + PBL_ADNH * ghl) * ghl;
        real bden = PBL_BDNM * gml + PBL_BDNH * ghl;
        real cden = 1.0f;
        real besm = PBL_BSMH * ghl;
        real besh = PBL_BSHM * gml + PBL_BSHH * ghl;
        real rden = 1.0f / (aden * eloq4 + bden * eloq2 + cden);
        real esm = (besm * eloq2 + PBL_CESM) * rden;
        real esh = (besh * eloq2 + PBL_CESH) * rden;
        real rdz = 2.0f / (zh[m] - zh[m + 2]);
        real q1l = sqrtf(q2[m]);
        real elqdz = ell * q1l * rdz;
        akm[m] = elqdz * esm;
        akh[m] = elqdz * esh;
    }
    // MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:439-463
    // GNU retains unassigned scratch words across the native I loop. Save
    // the setup values before heat/momentum density scaling overwrites locals.
    // A first shortened column has native uninitialized scratch; no zero fill
    // or acceptance exemption is introduced for that source defect.
    if (carry_el) {
        for (int m = 0; m < nzm; ++m) {
            carry_el[m] = el[m];
            carry_akm[m] = akm[m];
            carry_akh[m] = akh[m];
        }
        for (int m = nzm; m < nz - 1; ++m) {
            el[m] = carry_el[m];
            akm[m] = carry_akm[m];
            akh[m] = carry_akh[m];
        }
    }
    // EXCH_H publication, written in WRF-up order.
    for (int kup = 0; kup < nz - 1; ++kup) {
        int kflip = nz - 1 - kup;              // Fortran KFLIP=KTE-K
        real deltaz = 0.5f * (zh[kflip - 1] - zh[kflip + 1]);
        MYJ_UP(exch_h_a, kup) = akh[kflip - 1] * deltaz;
    }
    // EXCH_H(KTE) is INOUT and remains untouched.           // AKH is undefined at KTE

    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:1373-1450
    // ================= VDIFQ ===============================
    // s1=DTOZ, s2=AKQ, s3=CR, s4=CM.  AKQ is still read by the bottom-level
    // block, so RSQ2 cannot share it; rhok is not loaded until the heat
    // half, so it carries RSQ2 here.
    {
        const real esqhf = 0.5f * PBL_ESQ;
        int nq = lmh - 2;
        for (int m = 0; m < nq; ++m) {
            s1[m] = (dtdif + dtdif) / (zh[m] - zh[m + 2]);
            s2[m] = sqrtf((q2[m] + q2[m + 1]) * 0.5f) * (el[m] + el[m + 1])
                    * esqhf / (zh[m + 1] - zh[m + 2]);
            s3[m] = -s1[m] * s2[m];
        }
        s4[0] = s1[0] * s2[0] + 1.0f;
        rhok[0] = q2[0];
        for (int m = 1; m < nq; ++m) {
            real cf = -s1[m] * s2[m - 1] / s4[m - 1];
            s4[m] = -s3[m - 1] * cf + (s2[m - 1] + s2[m]) * s1[m] + 1.0f;
            rhok[m] = -rhok[m - 1] * cf + q2[m];
        }
        real dtozs = (dtdif + dtdif) / (zh[lmh - 2] - zh[lmh]);
        real akqs = sqrtf((q2[lmh - 2] + q2[lmh - 1]) * 0.5f)
                    * (el[lmh - 2] + PBL_ELZ0) * esqhf
                    / (zh[lmh - 1] - zh[lmh]);
        real cf = -dtozs * s2[nq - 1] / s4[nq - 1];
        q2[lmh - 2] = (dtozs * akqs * q2[lmh - 1] - rhok[nq - 1] * cf
                       + q2[lmh - 2])
                      / ((s2[nq - 1] + akqs) * dtozs - s3[nq - 1] * cf + 1.0f);
        for (int m = nq - 1; m >= 0; --m) {
            q2[m] = (-s3[m] * q2[m + 1] + rhok[m]) / s4[m];
        }
    }
    // Save the new TKE and mixing length.
    for (int kup = 0; kup < nz; ++kup) {
        int kflip = nz - kup;                  // Fortran KFLIP=KTE+1-K
        q2[kflip - 1] = fmaxf(q2[kflip - 1], MYJP_EPSQ2);
        MYJ_UP(tke_a, kup) = 0.5f * q2[kflip - 1];
        MYJ_UP(el_myj_a, kup) = (kflip < nz) ? el[kflip - 1] : 0.0f;
    }

    // ================= main_integration, heat/moisture =======
    real psfc = psfc_a[col];
    real thsk = tsk_a[col] * gfk_pow(1.0e5f / psfc, PBL_CAPA);
    for (int m = 0; m < nz; ++m) {
        real pkm = MYJ_UP(p_a, nz - 1 - m);
        rhok[m] = pkm / (MYJP_RD * tk[m]
                         * (1.0f + MYJP_P608 * qk[m] - cwm[m]));
    }
    // AKHK overwrites AKH in place; AKH is not read again.
    for (int m = 0; m < nz - 1; ++m) {
        akh[m] = akh[m] * 0.5f * (rhok[m] + rhok[m + 1]);
    }
    real seamask = xland_a[col] - 1.0f;
    real thz0 = (1.0f - seamask) * thsk + seamask * thz0_a[col];
    thz0_a[col] = thz0;
    real akhs_dens = akhs_a[col] * rhok[lmh - 1];
    real qsfc = qsfc_a[col];
    if (seamask < 0.5f) {
        real qfc1 = MYJP_XLV * chklowq_a[col] * akhs_dens;
        if (snow_a[col] > 0.0f || sice_a[col] > 0.5f) qfc1 = qfc1 * PBL_RLIVWV;
        if (qfc1 > 0.0f) qsfc = qk[lmh - 1] + elflx_a[col] / qfc1;
    } else {
        real exnsfc = gfk_pow(1.0e5f / psfc, PBL_CAPA);
        qsfc = PBL_PQ0SEA / psfc
               * gfk_exp_fma(MYJP_A2 * (thsk - MYJP_A3 * exnsfc)
                      / (thsk - MYJP_A4 * exnsfc));
    }
    qsfc_a[col] = qsfc;
    real qz0 = (1.0f - seamask) * qsfc + seamask * qz0_a[col];
    qz0_a[col] = qz0;

    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:1454-1563
    // ---- VDIFH, coefficients once, then species by species.
    // s1=DTOZ, s3=CR, s4=CM, s2=RKHZ (the RKCT operand), rs in "the"'s row?
    // No -- the species rows ARE the unknowns.  RSS rides gm, which is dead
    // after DIFCOF.
    {
        // SPECIES(3,K) is QCW ALONE  while CWMK is the TOTAL
        // condensate.  Both rode ``cwm`` up to here because RHOK
        // and MIXLEN want the total; the row that gets MIXED is the
        // cloud-water one, so reload it now that the total's last reader is
        // behind us.
        for (int m = 0; m < nz; ++m) cwm[m] = MYJ_UP(qc_a, nz - 1 - m);
        int nh = lmh - 1;
        for (int m = 0; m < nh; ++m) {
            s1[m] = dtdif / (zh[m] - zh[m + 1]);
            s3[m] = -s1[m] * akh[m];
            s2[m] = ((m + 1) < lpbl) ? 0.0f : akh[m] * (zh[m] - zh[m + 2]);
        }
        s4[0] = s1[0] * akh[0] + rhok[0];
        for (int m = 1; m < nh; ++m) {
            real cf = -s1[m] * akh[m - 1] / s4[m - 1];
            s4[m] = -s3[m - 1] * cf + (akh[m - 1] + akh[m]) * s1[m] + rhok[m];
        }
        real dtozs = dtdif / (zh[lmh - 1] - zh[lmh]);
        real rkhh = akh[nh - 1];
        real cfb = -dtozs * rkhh / s4[nh - 1];
        real cmb = s3[nh - 1] * cfb;
        int nspec = flqi ? 4 : 3;
        real ct_species0 = 0.0f;    // CTS(1)=CT, zeroed by MIXLEN
        for (int s = 0; s < nspec; ++s) {
            real* var = (s == 0) ? the : (s == 1) ? qk
                        : (s == 2) ? cwm : qcik;
            // SZ0/CLOW/CTS: only rows 1 and 2 are nonzero.
            real sz0 = (s == 0) ? thz0 : (s == 1) ? qz0 : 0.0f;
            real clow = (s == 0) ? 1.0f : (s == 1) ? chklowq_a[col] : 0.0f;
            real cts = (s == 0) ? ct_species0 : 0.0f;
            // Forward sweep, RSS in gm (dead since DIFCOF).
            gm[0] = -((s2[0] * cts) * 0.5f) * s1[0] + var[0] * rhok[0];
            for (int m = 1; m < nh; ++m) {
                real cf = -s1[m] * akh[m - 1] / s4[m - 1];
                real rkct_prev = (s2[m - 1] * cts) * 0.5f;
                real rkct_cur = (s2[m] * cts) * 0.5f;
                gm[m] = -gm[m - 1] * cf + (rkct_prev - rkct_cur) * s1[m]
                        + var[m] * rhok[m];
            }
            real rkss = akhs_dens * clow;
            real cmsb = -cmb + (rkhh + rkss) * dtozs + rhok[lmh - 1];
            real rssb = -gm[nh - 1] * cfb + ((s2[nh - 1] * cts) * 0.5f) * dtozs
                        + var[lmh - 1] * rhok[lmh - 1];
            var[lmh - 1] = (dtozs * rkss * sz0 + rssb) / cmsb;
            for (int m = nh - 1; m >= 0; --m) {
                // RCML=1./CM(K) then MULTIPLY: a float32
                // reciprocal-then-multiply is not a float32 divide.
                real rcml = 1.0f / s4[m];
                var[m] = (-s3[m] * var[m + 1] + gm[m]) * rcml;
            }
        }
    }
    // Primary variable tendencies.  CWMK is rebuilt from the
    // MIXED species exactly as the native species loop does, and it is the total that
    // enters THNEW while RQCBLTEN sees the cloud-water row alone.
    for (int kup = 0; kup < nz; ++kup) {
        int m = nz - 1 - kup;
        real qci = flqi ? qcik[m] : 0.0f;
        real cwmk = cwm[m] + qci;
        real ape = 1.0f / MYJ_UP(exner_a, kup);
        real thold = MYJ_UP(th_a, kup);
        real thnew = the[m] + cwmk * PBL_ELOCP * ape;
        MYJ_UP(rthblten_a, kup) = (thnew - thold) * rdtturbl;
        real qvup = MYJ_UP(qv_a, kup);
        real qold = qvup / (1.0f + qvup);
        real dqdt = (qk[m] - qold) * rdtturbl;
        MYJ_UP(rqvblten_a, kup) = dqdt / ((1.0f - qk[m]) * (1.0f - qk[m]));
        MYJ_UP(rqcblten_a, kup) = (cwm[m] - MYJ_UP(qc_a, kup)) * rdtturbl;
        if (flqi) MYJ_UP(rqiblten_a, kup) = (qci - MYJ_UP(qi_a, kup)) * rdtturbl;
    }

    // ================= main_integration, momentum ============
    {
        for (int m = 0; m < nz - 1; ++m) {
            akm[m] = akm[m] * (rhok[m] + rhok[m + 1]) * 0.5f;
        }
        real akms_dens = akms_a[col] * rhok[lmh - 1];
        real uz0 = uz0_a[col], vz0 = vz0_a[col];
        int nv = momentum_lmh - 1;
        for (int m = 0; m < nv; ++m) {
            s1[m] = dtdif / (zh[m] - zh[m + 1]);
            s3[m] = -s1[m] * akm[m];
        }
        s4[0] = s1[0] * akm[0] + rhok[0];
        for (int m = 1; m < nv; ++m) {
            real cf = -s1[m] * akm[m - 1] / s4[m - 1];
            s4[m] = -s3[m - 1] * cf + (akm[m - 1] + akm[m]) * s1[m] + rhok[m];
        }
        real dtozs = dtdif / (zh[momentum_lmh - 1] - zh[momentum_lmh]);
        real rkmh = akm[nv - 1];
        real cfb = -dtozs * rkmh / s4[nv - 1];
        real rcmvb = 1.0f / ((rkmh + akms_dens) * dtozs - s3[nv - 1] * cfb
                             + rhok[momentum_lmh - 1]);
        real dtozak = dtozs * akms_dens;
        for (int comp = 0; comp < 2; ++comp) {
            real* var = comp ? vk : uk;
            real sz0 = comp ? vz0 : uz0;
            gm[0] = var[0] * rhok[0];                            // RSU/RSV
            for (int m = 1; m < nv; ++m) {
                real cf = -s1[m] * akm[m - 1] / s4[m - 1];
                gm[m] = -gm[m - 1] * cf + var[m] * rhok[m];
            }
            var[momentum_lmh - 1] = (dtozak * sz0 - gm[nv - 1] * cfb
                            + var[momentum_lmh - 1] * rhok[momentum_lmh - 1]) * rcmvb;
            for (int m = nv - 1; m >= 0; --m) {
                real rcml = 1.0f / s4[m]; // as VDIFH
                var[m] = (-s3[m] * var[m + 1] + gm[m]) * rcml;
            }
        }
        for (int kup = 0; kup < nz; ++kup) {
            int m = nz - 1 - kup;
            MYJ_UP(rublten_a, kup) = (uk[m] - MYJ_UP(u_a, kup)) * rdtturbl;
            MYJ_UP(rvblten_a, kup) = (vk[m] - MYJ_UP(v_a, kup)) * rdtturbl;
        }
    }
#undef MYJ_UP
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:596-610,730-752
extern "C" __global__ void myj_row_depth(const int* lowlyr, int* depth, int nz, int n) {
    if (blockIdx.x || threadIdx.x) return;
    int coupled = 0;
    for (int i = 0; i < n; ++i) if (lowlyr[i] != 1) coupled = 1;
    depth[0] = coupled;
    depth[1] = nz - lowlyr[n - 1] + 1;
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:356-782
extern "C" __global__ void myjpbl_column(
    // full bottom-up columns, (nz, n)
    const real* __restrict__ dz_a,
    const real* __restrict__ u_a, const real* __restrict__ v_a,
    const real* __restrict__ t_a, const real* __restrict__ th_a,
    const real* __restrict__ exner_a, const real* __restrict__ qv_a,
    const real* __restrict__ qc_a, const real* __restrict__ qi_a,
    const real* __restrict__ p_a,
    real* __restrict__ tke_a,            // TKE_MYJ, inout (nz, n)
    // surface (n)
    const real* __restrict__ ht_a,
    const int* __restrict__ lowlyr_a,
    const real* __restrict__ psfc_a, const real* __restrict__ ust_a,
    const real* __restrict__ tsk_a, const real* __restrict__ chklowq_a,
    const real* __restrict__ xland_a, const real* __restrict__ sice_a,
    const real* __restrict__ snow_a, const real* __restrict__ akhs_a,
    const real* __restrict__ akms_a, const real* __restrict__ elflx_a,
    const real* __restrict__ uz0_a, const real* __restrict__ vz0_a,
    // inout surface (n)
    real* __restrict__ thz0_a, real* __restrict__ qz0_a,
    real* __restrict__ qsfc_a, real* __restrict__ ct_a,
    // outputs, bottom-up (nz, n)
    real* __restrict__ rublten_a, real* __restrict__ rvblten_a,
    real* __restrict__ rthblten_a, real* __restrict__ rqvblten_a,
    real* __restrict__ rqcblten_a, real* __restrict__ rqiblten_a,
    real* __restrict__ el_myj_a, real* __restrict__ exch_h_a,
    // outputs (n)
    real* __restrict__ pblh_a, int* __restrict__ kpbl_a,
    real* __restrict__ mixht_a,
    real dtturbl, int flqi, int nz, int n,
    const int* depth)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n || depth[0]) return;
    myjpbl_apply(dz_a, u_a, v_a, t_a, th_a, exner_a, qv_a, qc_a, qi_a, p_a, tke_a, ht_a, lowlyr_a, psfc_a, ust_a, tsk_a, chklowq_a, xland_a, sice_a, snow_a, akhs_a, akms_a, elflx_a, uz0_a, vz0_a, thz0_a, qz0_a, qsfc_a, ct_a, rublten_a, rvblten_a, rthblten_a, rqvblten_a, rqcblten_a, rqiblten_a, el_myj_a, exch_h_a, pblh_a, kpbl_a, mixht_a, dtturbl, flqi, nz, n,
                 col, nullptr, nullptr, nullptr, depth[1]);
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:356-492,596-610,730-782
extern "C" __global__ void myjpbl_row(
    // full bottom-up columns, (nz, n)
    const real* __restrict__ dz_a,
    const real* __restrict__ u_a, const real* __restrict__ v_a,
    const real* __restrict__ t_a, const real* __restrict__ th_a,
    const real* __restrict__ exner_a, const real* __restrict__ qv_a,
    const real* __restrict__ qc_a, const real* __restrict__ qi_a,
    const real* __restrict__ p_a,
    real* __restrict__ tke_a,            // TKE_MYJ, inout (nz, n)
    // surface (n)
    const real* __restrict__ ht_a,
    const int* __restrict__ lowlyr_a,
    const real* __restrict__ psfc_a, const real* __restrict__ ust_a,
    const real* __restrict__ tsk_a, const real* __restrict__ chklowq_a,
    const real* __restrict__ xland_a, const real* __restrict__ sice_a,
    const real* __restrict__ snow_a, const real* __restrict__ akhs_a,
    const real* __restrict__ akms_a, const real* __restrict__ elflx_a,
    const real* __restrict__ uz0_a, const real* __restrict__ vz0_a,
    // inout surface (n)
    real* __restrict__ thz0_a, real* __restrict__ qz0_a,
    real* __restrict__ qsfc_a, real* __restrict__ ct_a,
    // outputs, bottom-up (nz, n)
    real* __restrict__ rublten_a, real* __restrict__ rvblten_a,
    real* __restrict__ rthblten_a, real* __restrict__ rqvblten_a,
    real* __restrict__ rqcblten_a, real* __restrict__ rqiblten_a,
    real* __restrict__ el_myj_a, real* __restrict__ exch_h_a,
    // outputs (n)
    real* __restrict__ pblh_a, int* __restrict__ kpbl_a,
    real* __restrict__ mixht_a,
    real dtturbl, int flqi, int nz, int n,
    const int* depth)
{
    if (blockIdx.x || threadIdx.x || !depth[0]) return;
    real carry_el[MYJ_KMAX], carry_akm[MYJ_KMAX], carry_akh[MYJ_KMAX];
    for (int col = 0; col < n; ++col) {
        myjpbl_apply(dz_a, u_a, v_a, t_a, th_a, exner_a, qv_a, qc_a, qi_a, p_a, tke_a, ht_a, lowlyr_a, psfc_a, ust_a, tsk_a, chklowq_a, xland_a, sice_a, snow_a, akhs_a, akms_a, elflx_a, uz0_a, vz0_a, thz0_a, qz0_a, qsfc_a, ct_a, rublten_a, rvblten_a, rthblten_a, rqvblten_a, rqcblten_a, rqiblten_a, el_myj_a, exch_h_a, pblh_a, kpbl_a, mixht_a, dtturbl, flqi, nz, n,
                     col, carry_el, carry_akm, carry_akh, depth[1]);
    }
}

'''

SFC_SOURCE = r'''
// Keep divisors in constant memory so every target retains division.
#ifndef HEX_DIVISOR_40000000_DEFINED
#define HEX_DIVISOR_40000000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_40000000 = 2.0f;
#else
static const float hex_divisor_40000000 = 2.0f;
#endif
#endif
#ifndef HEX_DIVISOR_40400000_DEFINED
#define HEX_DIVISOR_40400000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_40400000 = 3.0f;
#else
static const float hex_divisor_40400000 = 3.0f;
#endif
#endif
#ifndef HEX_DIVISOR_438F8000_DEFINED
#define HEX_DIVISOR_438F8000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_438f8000 = 287.0f;
#else
static const float hex_divisor_438f8000 = 287.0f;
#endif
#endif

// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:76-365,369-1062
#ifndef MYJSFC_KZTM
#define MYJSFC_KZTM 10001
#endif
#define MYJSFC_KZTM2 (MYJSFC_KZTM - 2)

// MPAS-Model-rrfs/src/framework/mpas_constants.F:46-51
#define MYJ_G        9.80616f
#define MYJ_RD       287.0f
#define MYJ_RV       461.6f
#define MYJ_CP       (7.0f * 287.0f / hex_divisor_40000000)
#define MYJ_XLV      2.5e6f
#define MYJ_P608     (461.6f / hex_divisor_438f8000 - 1.0f)
#define MYJ_PQ0      379.90516f
#define MYJ_A2       17.2693882f
#define MYJ_A3       273.16f
#define MYJ_A4       35.86f
#define MYJ_EPSQ2    0.2f
#define MYJ_P1000MB  1.0e5f

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:35-57
#define SFC_ITRMX    5
#define SFC_VKARMAN  0.4f
#define SFC_CAPA     (MYJ_RD / MYJ_CP)
#define SFC_ELOCP    (2.72e6f / MYJ_CP)
#define SFC_RCAP     (1.0f / SFC_CAPA)
#define SFC_GOCP02   (MYJ_G / MYJ_CP * 2.0f)
#define SFC_GOCP10   (MYJ_G / MYJ_CP * 10.0f)
#define SFC_EPSU2    1.0e-6f
#define SFC_EPSUST   1.0e-9f
#define SFC_EPSZT    1.0e-28f
#define SFC_A2S      17.2693882f
#define SFC_A3S      273.16f
#define SFC_A4S      35.86f
#define SFC_SEAFC    0.98f
#define SFC_PQ0SEA   (MYJ_PQ0 * SFC_SEAFC)
#define SFC_EXCML    0.0001f
#define SFC_EXCMS    0.0001f
#define SFC_GLKBR    10.0f
#define SFC_GLKBS    30.0f
#define SFC_QVISC    2.1e-5f
#define SFC_RIC      0.505f
#define SFC_SMALL    0.35f
#define SFC_SQPR     0.84f
#define SFC_SQSC     0.84f
#define SFC_SQVISC   258.2f
#define SFC_TVISC    2.1e-5f
#define SFC_USTC     0.7f
#define SFC_USTR     0.225f
#define SFC_VISC     1.5e-5f
#define SFC_FH       1.01f
#define SFC_WWST     1.2f
#define SFC_ZTFC     1.0f
#define SFC_CZIV     (SFC_SMALL * SFC_GLKBS)
#define SFC_GRRS     (SFC_GLKBR / SFC_GLKBS)
#define SFC_RTVISC   (1.0f / SFC_TVISC)
#define SFC_RVISC    (1.0f / SFC_VISC)
#define SFC_ZQRZT    (SFC_SQSC / SFC_SQPR)
#define SFC_FZQ1     (SFC_RTVISC * SFC_QVISC * SFC_ZQRZT)
#define SFC_FZQ2     (SFC_RTVISC * SFC_QVISC * SFC_ZQRZT)
#define SFC_FZT1     (SFC_RVISC * SFC_TVISC * SFC_SQPR)
#define SFC_FZT2     (SFC_CZIV * SFC_GRRS * SFC_TVISC * SFC_SQPR)
#define SFC_FZU1     (SFC_CZIV * SFC_VISC)
#define SFC_RQVISC   (1.0f / SFC_QVISC)
#define SFC_USTFC    (0.018f / MYJ_G)
#define SFC_WWST2    (SFC_WWST * SFC_WWST)

// Interpolate the native similarity tables.
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:631-651 at 1d6e370e
__device__ static real myjsfc_table(const real* __restrict__ psi,
                                    real zeta, real ztmin, real dzeta)
{
    real rz = (zeta - ztmin) / dzeta;
    int k = (int)rz;
    real rdzt = rz - (real)k;
    k = min(k, MYJSFC_KZTM2);
    k = max(k, 0);
    return (psi[k + 1] - psi[k]) * rdzt + psi[k];
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:76-1062 at 1d6e370e
extern "C" __global__ void myjsfc_column(
    // full columns, gpuwm bottom-up (nz, n)
    const real* __restrict__ dz_a,
    const real* __restrict__ tke_a,
    // lowest-model-level state (n)
    const real* __restrict__ u1_a, const real* __restrict__ v1_a,
    const real* __restrict__ t1_a, const real* __restrict__ th1_a,
    const real* __restrict__ qv1_a, const real* __restrict__ qc1_a,
    const real* __restrict__ p1_a, const real* __restrict__ psfc_a,
    // static surface (n)
    const real* __restrict__ tsk_a, const real* __restrict__ xland_a,
    const real* __restrict__ mavail_a, const real* __restrict__ z0base_a,
    const real* __restrict__ ht_a,
    const int* __restrict__ lowlyr_a,
    // WRF INOUT state (n)
    real* __restrict__ ust_a, real* __restrict__ znt_a,
    real* __restrict__ thz0_a, real* __restrict__ qz0_a,
    real* __restrict__ uz0_a, real* __restrict__ vz0_a,
    real* __restrict__ qsfc_a, real* __restrict__ akhs_a,
    real* __restrict__ akms_a,
    // outputs (n)
    real* __restrict__ rmol_a, real* __restrict__ ct_a,
    real* __restrict__ pblh_a, real* __restrict__ rib_a,
    real* __restrict__ chs_a, real* __restrict__ chs2_a,
    real* __restrict__ cqs2_a, real* __restrict__ hfx_a,
    real* __restrict__ qfx_a, real* __restrict__ lh_a,
    real* __restrict__ flhc_a, real* __restrict__ flqc_a,
    real* __restrict__ qgh_a, real* __restrict__ cpm_a,
    real* __restrict__ u10_a, real* __restrict__ v10_a,
    real* __restrict__ t2_a, real* __restrict__ th2_a,
    real* __restrict__ tshltr_a, real* __restrict__ th10_a,
    real* __restrict__ q2_a, real* __restrict__ qshltr_a,
    real* __restrict__ q10_a, real* __restrict__ pshltr_a,
    // similarity tables (MYJSFC_KZTM each) + their scalars
    const real* __restrict__ psim1, const real* __restrict__ psih1,
    const real* __restrict__ psim2, const real* __restrict__ psih2,
    real dzeta1, real dzeta2, real ztmin1, real ztmax1,
    real ztmin2, real ztmax2, real fh01, real fh02,
    real p1000mb, int itimestep, int nz, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    size_t st = (size_t)n;

    int ntsd = itimestep;
    real ust = ust_a[col];
    if (ntsd == 1) ust = 0.1f; // (ARW branch)
    real seamask = xland_a[col] - 1.0f;
    real psfc = psfc_a[col];
    real thsk = tsk_a[col] / gfk_pow(__fdiv_rn(psfc, p1000mb), SFC_CAPA);

    // PBLH scan: lpbl is the myj (top-down, 1-based) index of
    // the first weak-TKE level above the lowest layer;
    // PBLH = ZHK(LPBL) - ZHK(LMH+1).
    //
    // The interface heights are WRF's: ZINT(I,KTE+1,J)=HT(I,J) and
    // ZINT(I,K,J)=ZINT(I,K+1,J)+DZ(I,KFLIP,J), so they are
    // heights above SEA LEVEL and PBLH and ZSL are differences of
    // them.  HT cancels in real arithmetic, not in float32, and the
    // oracle's terrain columns grade the difference.  The accumulation
    // stays inside ONE column: element (0, col) of the (nz, ny, nx) array is
    // dz_a[col] (dz_a[0] was once every thread's seed, wrong by
    // dz[0][0] - dz[0][col] wherever terrain thins the lowest layer).
    int lowest = lowlyr_a[col] - 1;
    int lmh = nz - lowest;
    int lpbl = lmh;
    real zground = ht_a[col];
    for (int iz = 0; iz < lowest; ++iz) zground = zground + dz_a[(size_t)iz * st + col];             // ZHK(LMH+1) = ZINT(KTE+1) = HT
    real zlow = zground + dz_a[(size_t)lowest * st + col];      // ZINT(KTE): top of the lowest layer
    real zcum = zlow;
    real pblh = 0.0f;
    bool found = false;
    for (int iz = lowest + 1; iz < nz && !found; ++iz) {
        zcum = zcum + dz_a[(size_t)iz * st + col];
        if (2.0f * tke_a[(size_t)iz * st + col] <= MYJ_EPSQ2 * SFC_FH) {
            lpbl = nz - iz;
            pblh = zcum - zground;        // ZHK(LPBL) - ZHK(LMH+1)
            found = true;
        }
    }
    if (!found) { lpbl = 1; pblh = zcum - zground; }
    (void)lpbl;

    // Lowest-layer state.
    real plow = p1_a[col];
    real tlow = t1_a[col];
    real thlow = th1_a[col];
    real qv1 = qv1_a[col];
    real qlow = qv1 / (1.0f + qv1);
    real cwmlow = qc1_a[col];
    real thelow = (cwmlow * (-SFC_ELOCP / tlow) + 1.0f) * thlow;
    real ulow = u1_a[col];
    real vlow = v1_a[col];
    real zsl = (zlow - zground) * 0.5f;   // (ZHK(LMH)-ZHK(LMH+1))*0.5
    real apesfc = gfk_pow(__fdiv_rn(psfc, p1000mb), SFC_CAPA);
    real tz0 = (ntsd == 1) ? tsk_a[col] : thz0_a[col] * apesfc;

    // ---- SFCDIF ----
    real qs = qsfc_a[col];
    real thz0 = thz0_a[col];
    real qz0 = qz0_a[col];
    real uz0 = uz0_a[col];
    real vz0 = vz0_a[col];
    real z0 = znt_a[col];
    real akhs = akhs_a[col];
    real akms = akms_a[col];
    real z0base = z0base_a[col];
    real mavail = mavail_a[col];

    real rdz = 1.0f / zsl;
    real cxchl = SFC_EXCML * rdz;
    real cxchs = SFC_EXCMS * rdz;
    real btgx = MYJ_G / thlow;
    real elfc = SFC_VKARMAN * btgx;
    real btgh = (pblh > 1000.0f) ? btgx * pblh : btgx * 1000.0f;

    real wstar2 = 0.0f;
    real rlmo = 0.0f;
    real rib = 0.0f, du2 = SFC_EPSU2, dthv = 0.0f;
    real zu = z0, zt = z0;
    real psmz = 0.0f, pshz = 0.0f, ustark = ust * SFC_VKARMAN;

    if (seamask > 0.5f) {
        // ---- sea points ----
        for (int itr = 0; itr < SFC_ITRMX; ++itr) {
            z0 = fmaxf(SFC_USTFC * ust * ust, 1.59e-5f);
            if (ust < SFC_USTC) {
                if (ust < SFC_USTR) {
                    if (ntsd == 1) {
                        akms = cxchs;
                        akhs = cxchs;
                        qs = qlow;
                    }
                    zu = SFC_FZU1 * sqrtf(sqrtf(z0 * ust * SFC_RVISC)) / ust;
                    real wght = akms * zu * SFC_RVISC;
                    real rwgh = wght / (wght + 1.0f);
                    uz0 = (ulow * rwgh + uz0) * 0.5f;
                    vz0 = (vlow * rwgh + vz0) * 0.5f;
                    zt = SFC_FZT1 * zu;
                    real zq = SFC_FZQ1 * zt;
                    real wghtt = akhs * zt * SFC_RTVISC;
                    real wghtq = akhs * zq * SFC_RQVISC;
                    if (ntsd > 1) {
                        thz0 = ((wghtt * thlow + thsk) / (wghtt + 1.0f)
                                + thz0) * 0.5f;
                        qz0 = ((wghtq * qlow + qs) / (wghtq + 1.0f)
                               + qz0) * 0.5f;
                    } else {
                        thz0 = (wghtt * thlow + thsk) / (wghtt + 1.0f);
                        qz0 = (wghtq * qlow + qs) / (wghtq + 1.0f);
                    }
                }
                if (ust >= SFC_USTR && ust < SFC_USTC) {
                    zu = z0;
                    uz0 = 0.0f;
                    vz0 = 0.0f;
                    zt = SFC_FZT2 * sqrtf(sqrtf(z0 * ust * SFC_RVISC)) / ust;
                    real zq = SFC_FZQ2 * zt;
                    real wghtt = akhs * zt * SFC_RTVISC;
                    real wghtq = akhs * zq * SFC_RQVISC;
                    if (ntsd > 1) {
                        thz0 = ((wghtt * thlow + thsk) / (wghtt + 1.0f)
                                + thz0) * 0.5f;
                        qz0 = ((wghtq * qlow + qs) / (wghtq + 1.0f)
                               + qz0) * 0.5f;
                    } else {
                        thz0 = (wghtt * thlow + thsk) / (wghtt + 1.0f);
                        qz0 = (wghtq * qlow + qs) / (wghtq + 1.0f);
                    }
                }
            } else {
                zu = z0;
                uz0 = 0.0f;
                vz0 = 0.0f;
                zt = z0;
                thz0 = thsk;
                qz0 = qs;
            }
            real tem = (tlow + tz0) * 0.5f;
            real thm = (thelow + thz0) * 0.5f;
            real a = thm * MYJ_P608;
            real b = (SFC_ELOCP / tem - 1.0f - MYJ_P608) * thm;
            dthv = ((thelow - thz0)
                    * ((qlow + qz0 + cwmlow) * (0.5f * MYJ_P608) + 1.0f)
                    + (qlow - qz0 + cwmlow) * a + cwmlow * b);
            du2 = fmaxf((ulow - uz0) * (ulow - uz0)
                        + (vlow - vz0) * (vlow - vz0), SFC_EPSU2);
            rib = btgx * dthv * zsl / du2;
            real zslu = zsl + zu;
            real zslt = zsl + zt;
            real rzsu = zslu / zu;
            real rzst = zslt / zt;
            real rlogu = gfk_log(rzsu);
            real rlogt = gfk_log(rzst);
            rlmo = elfc * akhs * dthv / (ust * ust * ust);
            real zetalu = zslu * rlmo;
            real zetalt = zslt * rlmo;
            real zetau = zu * rlmo;
            real zetat = zt * rlmo;
            zetalu = fminf(fmaxf(zetalu, ztmin1), ztmax1);
            zetalt = fminf(fmaxf(zetalt, ztmin1), ztmax1);
            zetau = fminf(fmaxf(zetau, ztmin1 / rzsu), ztmax1 / rzsu);
            zetat = fminf(fmaxf(zetat, ztmin1 / rzst), ztmax1 / rzst);
            psmz = myjsfc_table(psim1, zetau, ztmin1, dzeta1);
            real psmzl = myjsfc_table(psim1, zetalu, ztmin1, dzeta1);
            real simm = psmzl - psmz + rlogu;
            pshz = myjsfc_table(psih1, zetat, ztmin1, dzeta1);
            real pshzl = myjsfc_table(psih1, zetalt, ztmin1, dzeta1);
            real simh = (pshzl - pshz + rlogt) * fh01;
            ustark = ust * SFC_VKARMAN;
            akms = fmaxf(ustark / simm, cxchs);
            akhs = fmaxf(ustark / simh, cxchs);
            if (dthv <= 0.0f) {
                wstar2 = SFC_WWST2
                    * gfk_pow(fabsf(btgh * akhs * dthv), 2.0f / hex_divisor_40400000);
            } else {
                wstar2 = 0.0f;
            }
            ust = fmaxf(sqrtf(akms * sqrtf(du2 + wstar2)), SFC_EPSUST);
        }
    } else {
        // ---- land points ----
        if (ntsd == 1) qs = qlow;
        zu = z0;
        uz0 = 0.0f;
        vz0 = 0.0f;
        zt = zu * SFC_ZTFC;
        thz0 = thsk;
        qz0 = qs;
        real tem = (tlow + tz0) * 0.5f;
        real thm = (thelow + thz0) * 0.5f;
        real a = thm * MYJ_P608;
        real b = (SFC_ELOCP / tem - 1.0f - MYJ_P608) * thm;
        dthv = ((thelow - thz0)
                * ((qlow + qz0 + cwmlow) * (0.5f * MYJ_P608) + 1.0f)
                + (qlow - qz0 + cwmlow) * a + cwmlow * b);
        du2 = fmaxf((ulow - uz0) * (ulow - uz0)
                    + (vlow - vz0) * (vlow - vz0), SFC_EPSU2);
        rib = btgx * dthv * zsl / du2;
        real zslu = zsl + zu;
        real rzsu = zslu / zu;
        real rlogu = gfk_log(rzsu);
        real zslt = zsl + zu;       // u,v and t are at the same level
        const real czil = 0.1f; // (Chen-Zhang block commented out)
        real zilfc = -czil * SFC_VKARMAN * SFC_SQVISC;
        const real czetmax = 10.0f;
        real zzil;
        if (dthv > 0.0f) {
            if (rib < SFC_RIC) {
                zzil = zilfc * (1.0f + (__fdiv_rn(rib, SFC_RIC)) * (__fdiv_rn(rib, SFC_RIC))
                                * czetmax);
            } else {
                zzil = zilfc * (1.0f + czetmax);
            }
        } else {
            zzil = zilfc;
        }
        for (int itr = 0; itr < SFC_ITRMX; ++itr) {
            zt = fmaxf(gfk_exp_fma(zzil * sqrtf(ust * z0base)) * z0base,
                       SFC_EPSZT);
            real rzst = zslt / zt;
            real rlogt = gfk_log(rzst);
            rlmo = elfc * akhs * dthv / (ust * ust * ust);
            real zetalu = zslu * rlmo;
            real zetalt = zslt * rlmo;
            real zetau = zu * rlmo;
            real zetat = zt * rlmo;
            zetalu = fminf(fmaxf(zetalu, ztmin2), ztmax2);
            zetalt = fminf(fmaxf(zetalt, ztmin2), ztmax2);
            zetau = fminf(fmaxf(zetau, ztmin2 / rzsu), ztmax2 / rzsu);
            zetat = fminf(fmaxf(zetat, ztmin2 / rzst), ztmax2 / rzst);
            psmz = myjsfc_table(psim2, zetau, ztmin2, dzeta2);
            real psmzl = myjsfc_table(psim2, zetalu, ztmin2, dzeta2);
            real simm = psmzl - psmz + rlogu;
            pshz = myjsfc_table(psih2, zetat, ztmin2, dzeta2);
            real pshzl = myjsfc_table(psih2, zetalt, ztmin2, dzeta2);
            real simh = (pshzl - pshz + rlogt) * fh02;
            ustark = ust * SFC_VKARMAN;
            akms = fmaxf(ustark / simm, cxchl);
            akhs = fmaxf(ustark / simh, cxchl);
            if (dthv <= 0.0f) {
                wstar2 = SFC_WWST2
                    * gfk_pow(fabsf(btgh * akhs * dthv), 2.0f / hex_divisor_40400000);
            } else {
                wstar2 = 0.0f;
            }
            ust = fmaxf(sqrtf(akms * sqrtf(du2 + wstar2)), SFC_EPSUST);
        }
    }
    real ct = 0.0f; // countergradient commented out

    // ---- 2m / 10m diagnostics ----
    real umflx = akms * (ulow - uz0);
    real vmflx = akms * (vlow - vz0);
    real hsflx = akhs * (thlow - thz0);
    real hlflx = akhs * (qlow - qz0);
    real zu10 = zu + 10.0f;
    real zt02 = zt + 2.0f;
    real zt10 = zt + 10.0f;
    real rlnu10 = gfk_log(zu10 / zu);
    real rlnt02 = gfk_log(zt02 / zt);
    real rlnt10 = gfk_log(zt10 / zt);
    real ztau10 = zu10 * rlmo;
    real ztat02 = zt02 * rlmo;
    real ztat10 = zt10 * rlmo;
    real akms10, akhs02, akhs10;
    if (seamask > 0.5f) {
        ztau10 = fminf(fmaxf(ztau10, ztmin1), ztmax1);
        ztat02 = fminf(fmaxf(ztat02, ztmin1), ztmax1);
        ztat10 = fminf(fmaxf(ztat10, ztmin1), ztmax1);
        real psm10 = myjsfc_table(psim1, ztau10, ztmin1, dzeta1);
        real simm10 = psm10 - psmz + rlnu10;
        real psh02 = myjsfc_table(psih1, ztat02, ztmin1, dzeta1);
        real simh02 = (psh02 - pshz + rlnt02) * fh01;
        real psh10 = myjsfc_table(psih1, ztat10, ztmin1, dzeta1);
        real simh10 = (psh10 - pshz + rlnt10) * fh01;
        akms10 = fmaxf(ustark / simm10, cxchs);
        akhs02 = fmaxf(ustark / simh02, cxchs);
        akhs10 = fmaxf(ustark / simh10, cxchs);
    } else {
        ztau10 = fminf(fmaxf(ztau10, ztmin2), ztmax2);
        ztat02 = fminf(fmaxf(ztat02, ztmin2), ztmax2);
        ztat10 = fminf(fmaxf(ztat10, ztmin2), ztmax2);
        real psm10 = myjsfc_table(psim2, ztau10, ztmin2, dzeta2);
        real simm10 = psm10 - psmz + rlnu10;
        real psh02 = myjsfc_table(psih2, ztat02, ztmin2, dzeta2);
        real simh02 = (psh02 - pshz + rlnt02) * fh02;
        real psh10 = myjsfc_table(psih2, ztat10, ztmin2, dzeta2);
        real simh10 = (psh10 - pshz + rlnt10) * fh02;
        akms10 = fmaxf(ustark / simm10, cxchl);
        akhs02 = fmaxf(ustark / simh02, cxchl);
        akhs10 = fmaxf(ustark / simh10, cxchl);
    }
    real u10 = umflx / akms10 + uz0;
    real v10 = vmflx / akms10 + vz0;
    real th02 = hsflx / akhs02 + thz0;
    if ((thlow > thz0 && (th02 < thz0 || th02 > thlow))
        || (thlow < thz0 && (th02 > thz0 || th02 < thlow))) {
        th02 = thz0 + 2.0f * rdz * (thlow - thz0);
    }
    real th10 = hsflx / akhs10 + thz0;
    if ((thlow > thz0 && (th10 < thz0 || th10 > thlow))
        || (thlow < thz0 && (th10 > thz0 || th10 < thlow))) {
        th10 = thz0 + 10.0f * rdz * (thlow - thz0);
    }
    real q02 = hlflx / akhs02 + qz0;
    real q10 = hlflx / akhs10 + qz0;
    real pshltr = psfc * gfk_exp_fma(-0.068283f / tlow);
    real u10e = u10;
    real v10e = v10;
    if (seamask < 0.5f) {
        // Equivalent z0 shelter winds over land.
        real zuuz = fminf(zu * 0.50f, 0.18f);
        real zu_l = fmaxf(zu * 0.35f, zuuz);
        real zu10_l = zu_l + 10.0f;
        real rzsu_l = zu10_l / zu_l;
        real rlnu10_l = gfk_log(rzsu_l);
        real ztau10_l = zu10_l * rlmo;
        ztau10_l = fminf(fmaxf(ztau10_l, ztmin2), ztmax2);
        real psm10 = myjsfc_table(psim2, ztau10_l, ztmin2, dzeta2);
        real simm10 = psm10 - psmz + rlnu10_l;
        real ekms10 = fmaxf(ustark / simm10, cxchl);
        u10e = umflx / ekms10 + uz0;
        v10e = vmflx / ekms10 + vz0;
    }
    // ---- WRF driver arrays ----
    real rlow = plow / (MYJ_RD * tlow);
    real chs = akhs;
    real chs2 = akhs02;
    real cqs2 = akhs02;
    real hfx = -rlow * MYJ_CP * hsflx;
    real qfx = -rlow * hlflx * mavail;
    real flx_lh = MYJ_XLV * qfx;
    real flhc = rlow * MYJ_CP * akhs;
    real flqc = rlow * akhs * mavail;
    real qgh = ((1.0f - seamask) * MYJ_PQ0 + seamask * SFC_PQ0SEA) / plow
        * gfk_exp_fma(SFC_A2S * (tlow - SFC_A3S) / (tlow - SFC_A4S));
    qgh = qgh / (1.0f - qgh);              // convert to mixing ratio
    real cpm = MYJ_CP * (1.0f + 0.8f * qlow);
    if (seamask > 0.5f) {
        real tskl = tsk_a[col];
        qs = SFC_PQ0SEA / psfc
            * gfk_exp_fma(SFC_A2S * (tskl - SFC_A3S) / (tskl - SFC_A4S));
        qs = qs / (1.0f - qs);
    }
    // MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:338-361
    real tshltr = th02;      // SFCDIF's TH02 dummy is MYJSFC's TSHLTR
    real qshltr = q02;       // SFCDIF's Q02 dummy is MYJSFC's QSHLTR
    real rapa = apesfc;
    real th02p = tshltr;
    real th10p = th10;
    real th2_grid = tshltr;
    real rapa02 = rapa - SFC_GOCP02 / th02p;
    real rapa10 = rapa - SFC_GOCP10 / th10p;
    real t02p = th02p * rapa02;
    real t10p = th10p * rapa10;
    real t2_grid = th2_grid * apesfc;
    real p02p = gfk_pow(rapa02, SFC_RCAP) * p1000mb;
    real p10p = gfk_pow(rapa10, SFC_RCAP) * p1000mb;
    real qs02 = MYJ_PQ0 / p02p
        * gfk_exp_fma(MYJ_A2 * (t02p - MYJ_A3) / (t02p - MYJ_A4));
    real qs10 = MYJ_PQ0 / p10p
        * gfk_exp_fma(MYJ_A2 * (t10p - MYJ_A3) / (t10p - MYJ_A4));
    if (qshltr > qs02) qshltr = qs02;
    if (q10 > qs10) q10 = qs10;
    real q02_grid = qshltr / (1.0f - qshltr);

    // ---- writeback ----
    ust_a[col] = ust;
    znt_a[col] = z0;
    thz0_a[col] = thz0;
    qz0_a[col] = qz0;
    uz0_a[col] = uz0;
    vz0_a[col] = vz0;
    qsfc_a[col] = qs;
    akhs_a[col] = akhs;
    akms_a[col] = akms;
    rmol_a[col] = rlmo;
    ct_a[col] = ct;
    pblh_a[col] = pblh;
    rib_a[col] = rib;
    chs_a[col] = chs;
    chs2_a[col] = chs2;
    cqs2_a[col] = cqs2;
    hfx_a[col] = hfx;
    qfx_a[col] = qfx;
    lh_a[col] = flx_lh;
    flhc_a[col] = flhc;
    flqc_a[col] = flqc;
    qgh_a[col] = qgh;
    cpm_a[col] = cpm;
    u10_a[col] = u10e;
    v10_a[col] = v10e;
    t2_a[col] = t2_grid;
    th2_a[col] = th2_grid;
    tshltr_a[col] = tshltr;
    th10_a[col] = th10;
    q2_a[col] = q02_grid;
    qshltr_a[col] = qshltr;
    q10_a[col] = q10;
    pshltr_a[col] = pshltr;
}

'''


def shared_math_source():
    """Shared math binding; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:1216-1257."""
    root = files('woof.core.kernels')
    return '\n'.join(root.joinpath(name).read_text(encoding='utf-8') for name in
                     ('glibc_flt32.cuh', 'flt32_expf_fma.cuh', 'glibc_trig_flt32.cuh'))


def cuda_source(levels):
    """Compile-time column extent; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:238-258."""
    return (f'#define MYJ_LEVELS {int(levels)}\ntypedef float real;\n'
            + shared_math_source() + INIT_SOURCE + PBL_SOURCE + SFC_SOURCE)


@lru_cache(maxsize=None)
def _module(device_id, levels):
    """Direct NVRTC preserves subnormals; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:653-664."""
    import cupy as cp
    from cupy.cuda import compiler
    with cp.cuda.Device(device_id):
        binary, _ = compiler.compile_using_nvrtc(
            cuda_source(levels), options=_hex_numerics.options(CUDA_OPTIONS),
            arch=_compilation_target(device_id, xp=cp))
        module = cp.cuda.function.Module()
        module.load(binary)
        return module


def _function(name, levels):
    """Resolve entry; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:141-784."""
    import cupy as cp
    return _module(cp.cuda.Device().id, levels).get_function(name)


def _launch(name, levels, count, arguments):
    """Launch complete columns; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:356-784."""
    _function(name, levels)(((count + 63) // 64,), (64,), arguments)


def _check(array, shape, name, integer=False):
    """Validate native storage; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:166-200."""
    dtype = np.dtype('int32' if integer else 'float32')
    if array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous:
        raise ValueError(f'{name} requires contiguous {dtype} storage with shape {shape}')


def clear_mixing_length(memory, *, first_column=0, column_count=None):
    """Full-memory clear; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:351-351."""
    levels, columns = memory.shape
    _check(memory, (levels, columns), 'el_myj memory')
    active = columns-first_column if column_count is None else column_count
    if not 0 <= first_column <= columns or not 0 <= active <= columns-first_column:
        raise ValueError('EL_MYJ active columns escape native memory storage')
    if active:
        _launch('myj_clear_el', 1, levels*active,
                (memory, np.int32(first_column), np.int32(active), np.int32(columns), np.int32(levels*active)))


def launch_pbl(columns, surface, state, tke, outputs, *, dt, stepbl=1, flqi=True):
    """MYJPBL; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:141-784 at 1d6e370e.

    Optional rain/snow/graupel inputs do not enter active mixing at this pin
    (lines 288-299). Their tendencies remain untouched. EXCH_H at KTE and
    RQIBLTEN when QCI is absent also retain their incoming words.
    """
    nz, n = tke.shape
    _check(tke, (nz, n), 'tke')
    arrays = [columns[k] if k != 'qi' or flqi else columns['qc'] for k in PBL_COLUMNS]
    for name, array in zip(PBL_COLUMNS, arrays):
        _check(array, (nz, n), name)
    for name in PBL_SURFACE:
        _check(surface[name], (n,), name, name == 'lowlyr')
    for name in PBL_STATE:
        _check(state[name], (n,), name)
    for name in PBL_OUTPUT_COLUMNS:
        _check(outputs[name], (nz, n), name)
    for name in PBL_OUTPUT_SURFACE:
        _check(outputs[name], (n,), name, name == 'kpbl')
    import cupy as cp
    depth = cp.empty(2, dtype=cp.int32)
    _launch('myj_row_depth', nz, 1, (surface['lowlyr'], depth, np.int32(nz), np.int32(n)))
    arguments = (tuple(arrays) + (tke,)
            + tuple(surface[k] for k in PBL_SURFACE)
            + tuple(state[k] for k in PBL_STATE)
            + tuple(outputs[k] for k in PBL_OUTPUT_COLUMNS + PBL_OUTPUT_SURFACE)
            + (np.float32(dt) * np.float32(stepbl), np.int32(flqi), np.int32(nz), np.int32(n), depth))
    _launch('myjpbl_column', nz, n, arguments)
    _launch('myjpbl_row', nz, 1, arguments)


@lru_cache(maxsize=None)
def device_tables(device_id):
    """MYJSFCINIT tables; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:1183-1295."""
    import cupy as cp
    with cp.cuda.Device(device_id):
        arrays = tuple(cp.empty(10001, dtype=cp.float32) for _ in range(4))
        scalars = cp.empty(8, dtype=cp.float32)
        _function('myjsfc_tables', 1)((1,), (1,), arrays + (scalars,))
        # Eight control scalars are copied for by-value kernel arguments.
        return arrays, tuple(scalars.get())


def launch_surface(columns, surface, state, outputs, *, itimestep, p1000mb=100000.0):
    """MYJSFC/SFCDIF; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:76-1062."""
    import cupy as cp
    nz, n = columns['dz'].shape
    for name in SFC_COLUMNS:
        _check(columns[name], (nz, n), name)
    for name in SFC_SURFACE:
        _check(surface[name], (n,), name, name == 'lowlyr')
    for name in SFC_STATE:
        _check(state[name], (n,), name)
    for name in SFC_OUTPUTS:
        _check(outputs[name], (n,), name)
    tables, scalars = device_tables(cp.cuda.Device().id)
    _launch('myjsfc_column', nz, n,
            tuple(columns[k] for k in SFC_COLUMNS)
            + tuple(surface[k] for k in SFC_SURFACE)
            + tuple(state[k] for k in SFC_STATE)
            + tuple(outputs[k] for k in SFC_OUTPUTS)
            + tables + scalars + (np.float32(p1000mb), np.int32(itimestep), np.int32(nz), np.int32(n)))


def init_surface(lowlyr, ustar, *, restart=False):
    """MPAS branch MYJSFCINIT; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:1066-1177."""
    n = int(ustar.size)
    _check(lowlyr, (n,), 'lowlyr', True)
    _check(ustar, (n,), 'ustar')
    _launch('myjsfc_init', 1, n, (lowlyr, ustar, np.int32(n), np.int32(restart)))


def init_pbl(tke, outputs, *, restart=False):
    """MYJPBLINIT; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:1737-1777."""
    nz, n = tke.shape
    names = ('rublten', 'rvblten', 'rthblten', 'rqvblten', 'exch_h')
    for name in names:
        _check(outputs[name], (nz, n), name)
    _launch('myjpbl_init', nz, nz*n, (tke,) + tuple(outputs[k] for k in names)
            + (np.int32(nz*n), np.int32(restart)))


def register(registry, pbl_callback, surface_callback):
    """Bind r4 slot callbacks; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:336-337."""
    registry.register('pbl', 'bl_myj', 'hex-cuda', pbl_callback)
    registry.register('surface_layer', 'sf_myj', 'hex-cuda', surface_callback)


def _native_lowest(arguments, workspace=None):
    """Pack lowest values on-device; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:232-303."""
    import cupy as cp
    nz, n = arguments['dz'].shape
    _check(arguments['lowlyr'], (n,), 'lowlyr', True)
    _check(arguments['pint'], (nz+1,n), 'pint')
    for key in ('u','v','t','th','qv','qc','pmid'):
        _check(arguments[key], (nz,n), key)
    result = {} if workspace is None else workspace
    for key in ('psfc', 'u1', 'v1', 't1', 'th1', 'qv1', 'qc1', 'p1'):
        if key not in result:
            result[key] = cp.empty(n, dtype=cp.float32)
        _check(result[key], (n,), key)
    _launch('myj_lowest', 1, n,
            tuple(arguments[k] for k in ('lowlyr','pint','u','v','t','th','qv','qc','pmid'))
            + tuple(result[k] for k in ('psfc','u1','v1','t1','th1','qv1','qc1','p1'))
            + (np.int32(n),))
    return result


def pbl_from_native(arguments, *, initflag=2, workspace=None):
    """MYJ driver slice; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1666-1753.

    Arguments use MYJPBL names and active, bottom-up (KTE, columns) device
    arrays; PINT retains its KTE+1 interface. Shared driver packing and
    tendency accumulation belong to r4. Unmixed optional species are untouched.
    """
    a = dict(arguments)
    a['qc'] = a['qcw']
    a['qi'] = a.get('qci')
    lowest = _native_lowest(a, workspace)
    a['psfc'] = lowest['psfc']
    if initflag == 1:
        tke = a['tke_myj']
        _check(tke,a['dz'].shape,'tke_myj')
        _launch('myj_driver_tke_init', tke.shape[0], tke.size,
                (tke, np.int32(tke.size)))
    launch_pbl(a, a, a, a['tke_myj'], a, dt=a['dt'], stepbl=a.get('stepbl',1),
               flqi=a['qi'] is not None)


def surface_from_native(arguments, *, workspace=None):
    """Surface driver call; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1464-1514.

    Native argument names and active (KTE, columns) device arrays. The r4
    driver invokes this once per native call, including its fractional-seaice
    second call, and retains the shared ZNT/PBLH aliasing and blending order.
    """
    a = dict(arguments)
    a.update(_native_lowest(a, workspace))
    a['tke'] = a['q2']
    launch_surface(a,a,a,a,itimestep=a['itimestep'],p1000mb=a['p1000mb'])


def surface_driver_from_native(arguments, *, sea_arguments=None, workspace=None):
    """MYJ driver pair; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1464-1514.

    Supply sea_arguments exactly when config_frac_seaice is true. The two
    native calls share ZNT and PBLH; all supplied sea state/output arrays are
    separate. The common sfclayer_to_MPAS accumulation belongs to r4.
    """
    surface_from_native(arguments, workspace=workspace)
    if sea_arguments is not None:
        second = dict(arguments, **sea_arguments)
        second['znt'] = arguments['znt']
        second['pblh'] = arguments['pblh']
        surface_from_native(second, workspace=workspace)


def pbl_driver_from_native(arguments, *, initflag, el_myj_memory, workspace=None):
    """MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:351-351
    and MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1666-1753.

    el_myj_memory is (full_memory, first_column, column_count), including
    inactive vertical storage. Native pool packing and accumulation belong
    to the shared driver; this entry owns the scheme's complete memory clear.
    """
    memory, first, count = el_myj_memory
    clear_mixing_length(memory, first_column=first, column_count=count)
    pbl_from_native(arguments, initflag=initflag, workspace=workspace)


def register_native_callbacks(registry, *, pbl_arguments, surface_arguments, after_call):
    """MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1666-1753
    and MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1464-1514.

    The driver supplies native-named device arguments. The PBL packet adds
    initflag and el_myj_memory; the surface packet is (main, sea), with sea
    None when fractional sea ice is disabled. after_call performs the native
    accumulation once after the whole slot, including both surface calls.
    """
    def pbl(call):
        """MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1666-1753."""
        arguments = dict(pbl_arguments(call))
        initflag = arguments.pop('initflag')
        memory = arguments.pop('el_myj_memory')
        pbl_driver_from_native(arguments, initflag=initflag, el_myj_memory=memory)
        after_call('driver_pbl', call)

    def surface(call):
        """MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1464-1514."""
        main, sea = surface_arguments(call)
        surface_driver_from_native(main, sea_arguments=sea)
        after_call('driver_sfclayer', call)

    register(registry, pbl, surface)


# MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1728-1753.
PBL_NATIVE_ARRAYS = {
    'ht': 'ht_p', 'pmid': 'pres_hyd_p', 'pint': 'pres2_hyd_p',
    'th': 'th_p', 't': 't_p', 'exner': 'pi_p', 'qv': 'qv_p',
    'qcw': 'qc_p', 'qci': 'qi_p', 'qcs': 'qs_p', 'qcr': 'qr_p',
    'qcg': 'qg_p', 'u': 'u_p', 'v': 'v_p', 'rho': 'rho_p',
    'tsk': 'tsk_p', 'qsfc': 'qsfc_p', 'chklowq': 'chlowq_p',
    'thz0': 'thz0_p', 'qz0': 'qz0_p', 'uz0': 'uz0_p', 'vz0': 'vz0_p',
    'lowlyr': 'lowlyr_p', 'xland': 'xland_p', 'sice': 'xice_p',
    'snow': 'snow_p', 'tke_myj': 'tkepbl_p', 'exch_h': 'kzh_p',
    'ustar': 'ust_p', 'znt': 'znt_p', 'el_myj': 'elpbl_p',
    'pblh': 'hpbl_p', 'kpbl': 'kpbl_p', 'ct': 'ct_p',
    'akhs': 'akhs_p', 'akms': 'akms_p', 'elflx': 'lh_p', 'mixht': 'mixht_p',
    'rublten': 'rublten_p', 'rvblten': 'rvblten_p', 'rthblten': 'rthblten_p',
    'rqvblten': 'rqvblten_p', 'rqcblten': 'rqcblten_p',
    'rqiblten': 'rqiblten_p', 'dz': 'dz_p',
}
# MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1464-1514.
SURFACE_NATIVE_ARRAYS = {
    'ht': 'ht_p', 'dz': 'dz_p', 'pmid': 'pres_hyd_p', 'pint': 'pres2_hyd_p',
    'th': 'th_p', 't': 't_p', 'qv': 'qv_p', 'qc': 'qc_p',
    'u': 'u_p', 'v': 'v_p', 'q2': 'tkepbl_p', 'tsk': 'tsk_p',
    'qsfc': 'qsfc_p', 'thz0': 'thz0_p', 'qz0': 'qz0_p',
    'uz0': 'uz0_p', 'vz0': 'vz0_p', 'lowlyr': 'lowlyr_p',
    'xland': 'xland_p', 'ivgtyp': 'ivgtyp_p', 'ustar': 'ust_p',
    'znt': 'znt_p', 'z0base': 'z0_p', 'pblh': 'hpbl_p', 'mavail': 'mavail_p',
    'rmol': 'rmol_p', 'akhs': 'akhs_p', 'akms': 'akms_p', 'rib': 'br_p',
    'chs': 'chs_p', 'chs2': 'chs2_p', 'cqs2': 'cqs2_p',
    'hfx': 'hfx_p', 'qfx': 'qfx_p', 'flx_lh': 'lh_p',
    'flhc': 'flhc_p', 'flqc': 'flqc_p', 'qgh': 'qgh_p', 'cpm': 'cpm_p',
    'ct': 'ct_p', 'u10': 'u10_p', 'v10': 'v10_p',
    't02': 't2m_p', 'th02': 'th2m_p', 'tshltr': 'tshltr_p',
    'th10': 'th10_p', 'q02': 'q2_p', 'qshltr': 'qshltr_p',
    'q10': 'q10_p', 'pshltr': 'pshltr_p',
}
SURFACE_SEA_ARRAYS = {
    key: value[:-2] + '_sea'
    for key, value in SURFACE_NATIVE_ARRAYS.items()
    if key in (*SFC_STATE, *SFC_OUTPUTS, 'tsk') and key not in ('znt', 'pblh')
}


def run_prepared(prepared, metadata, *, slot, options, itimestep,
                 lower=None, initflag_override=None):
    """Complete native allocations to modules and back, without pool conversion.

    MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1666-1753
    and MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1464-1514.
    The native driver owns one-based active vertical indexing; allocation
    tails and inactive columns survive the device packing and copyback.
    """
    import cupy as cp
    its, ite, kts, kte, jts, jte = (
        int(metadata[key]) for key in ('its', 'ite', 'kts', 'kte', 'jts', 'jte'))
    if kts != 1:
        raise ValueError('MYJ lowest-level device indexing requires native KTS=1')
    if slot not in ('pbl', 'surface_layer'):
        raise ValueError('MYJ prepared callback requires PBL or surface-layer slot')
    bindings = PBL_NATIVE_ARRAYS if slot == 'pbl' else SURFACE_NATIVE_ARRAYS
    changed = (PBL_STATE + PBL_OUTPUT_COLUMNS + PBL_OUTPUT_SURFACE + ('tke_myj',)
               if slot == 'pbl' else SFC_STATE + SFC_OUTPUTS)

    def origin(name):
        return lower[name] if lower is not None else (int(metadata['ims']),
            int(metadata['kms']) if prepared[name].ndim == 3 else int(metadata['jms']),
            int(metadata['jms']))[:prepared[name].ndim]

    def view(name, j, *, interface=False):
        value, lo = prepared[name], origin(name)
        cells = slice(its - lo[0], ite - lo[0] + 1)
        if value.ndim == 3:
            return value[cells, kts-lo[1]:kte-lo[1]+1+int(interface), j-lo[2]].T
        if value.ndim == 2:
            return value[cells, j-lo[1]]
        raise ValueError('MYJ native column allocation must have rank two or three: ' + name)

    restart = bool(options.get('config_do_restart', False))
    cycling = bool(options.get('config_do_DAcycling', False))
    initflag = int(not (restart or itimestep > 1 or (slot == 'surface_layer' and cycling)))
    if initflag_override is not None:
        initflag = int(initflag_override)
    for j in range(jts, jte + 1):
        def pack(mapping):
            return {key: cp.ascontiguousarray(view(name, j, interface=key == 'pint'))
                    for key, name in mapping.items() if name in prepared}

        arguments = pack(bindings)
        if slot == 'pbl':
            arguments.update(dt=metadata['dt_pbl'], stepbl=metadata.get('stepbl', 1))
            if 'rqiblten' not in arguments:
                arguments['rqiblten'] = cp.empty_like(arguments['th'])
            elname = bindings['el_myj']
            lo = origin(elname)
            memory = cp.ascontiguousarray(prepared[elname][:, :, j-lo[2]].T)
            pbl_driver_from_native(arguments, initflag=initflag,
                el_myj_memory=(memory, its-lo[0], ite-its+1))
            prepared[elname][:, :, j-lo[2]] = memory.T
        else:
            arguments.update(itimestep=initflag, p1000mb=metadata['p1000mb'])
            sea = pack(SURFACE_SEA_ARRAYS) if options.get('config_frac_seaice', False) else None
            surface_driver_from_native(arguments, sea_arguments=sea)
            if sea is not None:
                for key, name in SURFACE_SEA_ARRAYS.items():
                    if key in changed:
                        view(name, j)[...] = sea[key]
        for key in changed:
            if key in arguments and bindings[key] in prepared:
                if key == 'rqiblten' and 'qci' not in arguments:
                    continue
                view(bindings[key], j)[...] = arguments[key]
    return prepared


def bind_native_pools(provider, call):
    """Hydrate full native ABI without truncating allocation or species tails.

    MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:461-1283,1666-1753
    and MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:434-1052,1464-1514.
    The shared r4 conversions own entry packing and final pool accumulation.
    """
    from . import driver_rrfs as driver
    native, cp = provider.native, provider.cp
    if call.slot not in ('pbl', 'surface_layer'):
        raise ValueError('MYJ native callback received a different physics slot')
    metadata = {}
    for name in ('ids', 'ide', 'jds', 'jde', 'kds', 'kde', 'ims', 'ime',
                 'jms', 'jme', 'kms', 'kme', 'jts', 'jte', 'kts', 'kte',
                 'dt_pbl', 'do_chemistry', 'num_chem', 'xice_threshold'):
        value = native.get_variable(name)
        if value is None:
            raise RuntimeError('Native MYJ scalar is absent: ' + name)
        metadata[name] = value[0].item()
    metadata.update(its=provider.session.arguments.its, ite=provider.session.arguments.ite,
                    p1000mb=native.get_constant('p0'))
    if any(metadata[name] != 1 for name in ('ims', 'kms', 'jms')):
        raise RuntimeError('r4 MYJ pool conversion requires one-based native memory origins')
    options = dict(provider.session.layout.configs)
    options.update(call.options)
    selector, expected = ('config_pbl_scheme', 'bl_myj') if call.slot == 'pbl' else (
        'config_sfclayer_scheme', 'sf_myj')
    if options[selector].strip() != expected:
        raise ValueError('MYJ callback received a different native selector')
    dimensions = {}
    if call.slot == 'pbl':
        allocations = set(PBL_NATIVE_ARRAYS.values()) | {
            'hfx_p', 'qfx_p', 'wspd_p', 'uoce_p', 'voce_p', 'rthraten_p', 'kzm_p', 'kzq_p'}
        diag = {'hfx', 'hpbl', 'qfx', 'ust', 'wspd', 'znt', 'kpbl', 'kzh', 'kzm', 'kzq',
                'lowlyr', 'chlowq', 'lh', 'thz0', 'qz0', 'uz0', 'vz0',
                'akms', 'akhs', 'qsfc', 'ct', 'mixht', 'tke_pbl', 'el_pbl'}
        tend = set(driver._PBL_COMMON_TEND) | {'rthratenlw', 'rthratensw'}
        names = {'diag_physics': diag, 'tend_physics': tend,
                 'sfc_input': {'xland', 'skintemp'}, 'mesh': {'zgrid'}}
        if metadata['do_chemistry']:
            allocations.update(('chem_p', 'ddvel_p', 'tend_chem_settle_p'))
            names['state'] = {'scalars'}
            dimensions['chemistry_start'] = provider.session.layout.indices['state', 'chemistry_start']
            if int(options['drydep_opt']) != 0:
                diag.add('ddvel')
                tend.add('tend_chem_settle')
            if options['config_mynn_enh_mix']:
                allocations.update(('frp_out_p', 'aero_emis_for_enhmix_p'))
                if options['config_smoke_scheme'].strip() == 'on':
                    diag.add('frp_out')
                if (options['config_anthro_scheme'].strip() != 'off'
                        or options['config_rwc_scheme'].strip() != 'off'):
                    diag.add('aero_emis_for_enhmix')
    else:
        fields = (*driver._SURFACE_INOUT, *driver._SURFACE_OUTPUT,
                  *driver._MYJ_SURFACE_INOUT, *driver._MYJ_SURFACE_OUTPUT)
        allocations = set(SURFACE_NATIVE_ARRAYS.values()) | {
            field + '_p' for field in fields} | {'dx_p', 'xice_p'}
        names = {'diag_physics': set(fields) | {'hpbl', 'mavail', 'z0', 'lowlyr', 'tke_pbl'},
                 'sfc_input': {'skintemp', 'xland', 'xice'},
                 'mesh': {'meshDensity', 'zgrid'}}
        if options['config_frac_seaice']:
            allocations.update(field + '_sea' for field in (*fields, 'mavail', 'tsk', 'xland'))
            allocations.update(('regime_hold', 'sst_p'))
            names['sfc_input'].add('sst')
    prepared, lower = {}, {}
    for name in sorted(allocations):
        value = native.get_variable(name)
        if value is not None:
            host, bounds = value
            prepared[name] = cp.asarray(host, order='C')
            lower[name] = bounds
    pools = {pool: {name: cp.asarray(native.get_field(pool, name, level=1), order='C')
                    for name in sorted(fields)} for pool, fields in names.items()}
    if not hasattr(provider, '_rrfs_myj_common_kernels'):
        provider._rrfs_myj_common_kernels = driver.StrictPhysicsKernelCache()
    return {'myj_native': {'native': native, 'prepared': prepared, 'lower': lower,
            'pools': pools, 'metadata': metadata, 'options': options,
            'dimensions': dimensions, 'kernels': provider._rrfs_myj_common_kernels}}


def run_slot(call):
    """One full slot transaction, including fractional-sea-ice pair and writeback.

    MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:461-1283,1666-1759
    and MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:434-1052,1464-1526.
    """
    import cupy as cp
    from . import driver_rrfs as driver
    context = call.pools['myj_native']
    metadata, options = context['metadata'], context['options']
    tile = {name: int(metadata[name]) for name in ('its', 'ite', 'jts', 'jte', 'kts', 'kte')}
    common = dict(pools=context['pools'], options=options, **tile)
    if call.slot == 'pbl':
        common.update(flags=metadata, dimensions=context['dimensions'])
        prepared, _ = driver.pbl_from_mpas(**common, prepared=context['prepared'],
                                         kernel_cache=context['kernels'])
    else:
        common.update(xice_threshold=metadata['xice_threshold'])
        prepared = driver.surface_layer_from_mpas(**common, prepared=context['prepared'],
                                                 kernel_cache=context['kernels'])
    run_prepared(prepared, metadata, slot=call.slot, options=options,
                 itimestep=call.itimestep, lower=context['lower'])
    if call.slot == 'pbl':
        pools = driver.pbl_to_mpas(**common, prepared=prepared)
    else:
        del common['kts'], common['kte']
        pools = {'diag_physics': driver.surface_layer_to_mpas(
            **common, prepared=prepared, kernel_cache=context['kernels'])}
    cp.cuda.get_current_stream().synchronize()
    native = context['native']
    for name, value in prepared.items():
        native.put_variable(name, value, lower=context['lower'][name])
    for pool, fields in pools.items():
        for name, value in fields.items():
            native.put_field(pool, name, value, level=1, nlevels=native._fields[pool, name][2])
    return prepared
