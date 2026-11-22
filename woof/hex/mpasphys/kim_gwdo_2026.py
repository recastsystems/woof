"""KIM orographic GWD (MMM-physics 20260319-WRFv4.8.0), complete resident column operator.

Authority: NCAR/MMM-physics bl_gwdo.F90 at tag 20260319-WRFv4.8.0 (63c806deef6a, commit 6c3ce3a3be04,
"SFCLAY+GWDO", Song-You Hong; BSD-3-Clause), the scheme WRF v4.8.0 ships as gwd_opt=1 (#2285). The MPAS
driver slice is MPAS-Model PR #1409 (GWDO2026): mpas_atmphys_driver_gwdo.F / module_bl_gwdo.F pass
``omax = elvmax`` (static field, metres above sea level), ``dx_factor = config_gwdo_factor`` (default 2.0)
and ``if_nonhyd = config_gwdo_nonhyd`` (default true) to bl_gwdo_run; everything else in the wrapper is the
v8.4.2 one ysu_gwdo_v842 ports (sina=0/cosa=1, dx = config_len_disp/meshDensity**0.25, dry hydrostatic
pressures, additive PBL tendencies, rubldiff/rvbldiff).

Selector ``config_gwdo_scheme = 'bl_kim_gwdo'`` (the PR #1409 name); hex knob ``gwdo_version = 'kim2026'``.
The pinned MPAS-A v8.4.2 YSU drag stays ``bl_ysu_gwdo`` (``gwdo_version = 'v842'``) in ysu_gwdo_v842, untouched.

Upstream changes carried (survey UPSTREAM-SCHEMES-2026-11-08/ugwp-gwdo.md K1-K12): omax/dx_factor/if_nonhyd
inputs; cg = 1; dx_eff = dx_factor*dx; Xu et al. (2024) nonhydrostatic factor; reference level capped at
kgwdmax = kte/2; omax-based blocking layer (komax, kbomax); no drag for omax <= 50 m and for bnv2 < 0 below
the reference level; convexity clamp [1, 10]; ol/olp floor 1e-5; per-level wind-reversal limiter; mesosphere
density limiter (p <= 7.5 hPa).  Port note carried: komax is clamped at the model top before zl(komax) is
read (native reads past kte when omax >= zl(kte); unreachable on real meshes, identical everywhere else).
"""
from __future__ import annotations

from woof.hex import numerics as _hex_numerics

from collections.abc import Mapping, MutableMapping
from functools import lru_cache
from hashlib import sha256
import math
from typing import Any

import numpy as np

from ..cuda_fp32 import CUDA_FTZ_HELPERS
from ..cuda_backend.containers import require_resident_array
from . import ysu_gwdo_v842 as _ysu

SCHEME = "kim_gwdo"
NAMELIST_OPTION = ("config_gwdo_scheme", "bl_kim_gwdo")
GWDO_VERSION = "kim2026"
BACKEND = "hex-cuda"
SOURCE_SHA256 = {
    # NCAR/MMM-physics 63c806deef6a bl_gwdo.F90 (raw.githubusercontent.com, fetched 2026-11-09).
    "physics_mmm/bl_gwdo.F90": "d6211b838422ff3fe9ddadf1466a2da985feb21ffbb0e8eedd4eb412ee9812f0",
}
LAYER_INPUTS = _ysu.LAYER_INPUTS
STATIC_INPUTS = (*_ysu.STATIC_INPUTS, "elvmax")
LAYER_OUTPUTS = _ysu.LAYER_OUTPUTS
SURFACE_OUTPUTS = _ysu.SURFACE_OUTPUTS
COMPILE_OPTIONS = _ysu.COMPILE_OPTIONS
#: MPAS PR #1409 Registry defaults (config_gwdo_factor, config_gwdo_nonhyd); WRF v4.8.0 Registry
#: gwd_dx_factor = 2., gwd_if_nonhyd = .true.
DEFAULT_DX_FACTOR = 2.0
DEFAULT_IF_NONHYD = True

bl_gwdo_init = _ysu.bl_gwdo_init
bl_gwdo_finalize = _ysu.bl_gwdo_finalize


_KERNEL_BODY = r"""

#define GWDO_NLEV @NLEV@

// gfortran's real(4) atan2 is glibc atan2f, correctly rounded since glibc 2.41 (CORE-MATH). Rounding the
// double atan2 to float reproduces it (0 of 400,000 random pairs and 196 signed-zero/inf/subnormal edge
// pairs differ against host glibc 2.43, verification host, 2026-11-09). td_atan2f (WOOF's float atan2) differs by
// 1 ULP on ~9% of random pairs, which flips the wind-rose sector at nint boundaries (:347).
__device__ __forceinline__ float kim_atan2f(const float y, const float x)
{
    return (float)atan2((double)y, (double)x);
}

// MMM-physics 63c806d bl_gwdo.F90:38-680 (bl_gwdo_run); MPAS PR #1409 driver slice.
extern "C" __global__ void bl_kim_gwdo_2026_f32(
    const int nlev, const int ncells, const float dt,
    const float len_disp, const float gravity, const float cp,
    const float rd, const float fv, const float pi,
    const float dx_factor, const int if_nonhyd,
    const float *u, const float *v, const float *temperature,
    const float *qv, const float *p_mass_dry_hyd,
    const float *p_interface_dry_hyd, const float *exner,
    const float *zmid, const float *rublten_in,
    const float *rvblten_in, const float *mesh_density,
    const float *var2d, const float *oc1, const float *oa1,
    const float *oa2, const float *oa3, const float *oa4,
    const float *ol1, const float *ol2, const float *ol3,
    const float *ol4, const float *omax_in,
    float *rublten_out, float *rvblten_out,
    float *dtaux3d, float *dtauy3d, float *dusfcg,
    float *dvsfcg, float *rubldiff, float *rvbldiff,
    int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    if (nlev != GWDO_NLEV) {
        atomicExch(invalid, 1);
        return;
    }
    // bl_gwdo.F90:139-162 parameters.
    const float qmin = 1.0e-30f, ric = 0.25f, velmin = 1.0f, rimin = -100.0f;
    const float bnv2min = 1.0e-5f, efacmin = 0.0f, efacmax = 10.0f, frc = 1.0f;
    const float frmax = 10.0f, ce = 0.8f, cg = 1.0f, hmt_min = 50.0f;
    const float oc_min = 1.0f, oc_max = 10.0f, olmin = 1.0e-5f;
    const float odmin = 0.1f, odmax = 10.0f;
    const float pcutoff = 7.5e2f, pcutoff_den = 0.01f;
    const int kgwdmin = 2;
    const int kte = nlev;
    const int kgwdmax = kte / 2;   // :213

    const float md = mesh_density[cell];
    const float var = var2d[cell];
    const float omax = omax_in[cell];
    const float oa_input[4] = {oa1[cell], oa2[cell], oa3[cell], oa4[cell]};
    const float ol_input[4] = {ol1[cell], ol2[cell], ol3[cell], ol4[cell]};

    float vtj[GWDO_NLEV];
    float vtk[GWDO_NLEV];
    float rho[GWDO_NLEV];
    float del[GWDO_NLEV];
    float u1[GWDO_NLEV];
    float v1[GWDO_NLEV];
    float usqj[GWDO_NLEV];
    float bnv2[GWDO_NLEV];
    float velco[GWDO_NLEV - 1];
    float taup[GWDO_NLEV + 1];
    float taufb[GWDO_NLEV + 1];
    float taud[GWDO_NLEV];
    float dtfac[GWDO_NLEV];
    for (int k = 0; k < nlev; ++k) {
        usqj[k] = 0.0f;
        bnv2[k] = 0.0f;
        taud[k] = 0.0f;
        dtfac[k] = 1.0f;
    }
    for (int k = 0; k <= nlev; ++k) {
        taup[k] = 0.0f;
        taufb[k] = 0.0f;
    }

    // MPAS driver dx (mpas_atmphys_driver_gwdo.F:533); bl_gwdo.F90:224-236.
    const float dx = mpas_div(len_disp, gfk_pow(md, 0.25f));
    const float area = mpas_mul(dx, dx);
    const float diagonal = mpas_sqrt(
        mpas_add(mpas_mul(dx, dx), mpas_mul(dx, dx)));
    const float dxy4[4] = {dx, dx, diagonal, diagonal};
    const float dxy4p[4] = {dx, dx, diagonal, diagonal};
    const float dx_eff = mpas_mul(mpas_mul(dx_factor, mpas_add(dx, dx)), 0.5f);

    for (int k = 0; k < nlev; ++k) {   // :252-264
        const int i = k * ncells + cell;
        const int ip = (k + 1) * ncells + cell;
        const float virtual_factor = mpas_add(1.0f, mpas_mul(fv, qv[i]));
        vtj[k] = mpas_mul(temperature[i], virtual_factor);
        vtk[k] = mpas_div(vtj[k], exner[i]);
        rho[k] = mpas_div(
            mpas_mul(mpas_div(1.0f, rd), p_mass_dry_hyd[i]), vtj[k]);
        del[k] = mpas_sub(
            p_interface_dry_hyd[i], p_interface_dry_hyd[ip]);
        u1[k] = mpas_sub(mpas_mul(u[i], 1.0f), mpas_mul(v[i], 0.0f));
        v1[k] = mpas_add(mpas_mul(u[i], 0.0f), mpas_mul(v[i], 1.0f));
    }
#define ZL(kf) zmid[((kf) - 1) * ncells + cell]

    // :266-285 reference level.
    const float zref = mpas_mul(2.0f, var);
    int kref = 0;
    for (int kf = 2; kf <= kte; ++kf) {
        if (kref == 0 && mpas_sub(ZL(kf), ZL(1)) >= zref) kref = kf + 1;
    }
    kref = kref < kgwdmax ? kref : kgwdmax;
    kref = kref > kgwdmin ? kref : kgwdmin;

    // :287-317 omax levels (MSL heights in MPAS/WRF).
    int komax = 0;
    for (int kf = 2; kf <= kte; ++kf) {
        if (komax == 0 && ZL(kf) >= omax) komax = kf + 1;
    }
    komax = komax > kref ? komax : kref;
    const float omax_top = mpas_add(omax, zref);
    int kbomax = 0;
    for (int kf = 2; kf <= kte; ++kf) {
        if (kbomax == 0 && ZL(kf) >= omax_top) kbomax = kf + 1;
    }
    // Port note (survey): native reads zl(komax) at :563 with komax up to kte+1;
    // clamp the read index only (every k <= komax comparison is unchanged).
    const int komax_read = komax < kte ? komax : kte;

    const float delks = mpas_div(1.0f, mpas_sub(p_interface_dry_hyd[cell],
        p_interface_dry_hyd[(kref - 1) * ncells + cell]));   // :319-322
    const float delks1 = mpas_div(1.0f, mpas_sub(p_mass_dry_hyd[cell],
        p_mass_dry_hyd[(kref - 1) * ncells + cell]));

    float ubar = 0.0f;
    float vbar = 0.0f;
    float rhobar = 0.0f;
    for (int kf = 1; kf <= kgwdmax; ++kf) {   // :326-336
        if (kf < kref) {
            const int k = kf - 1;
            const float rcsks = mpas_mul(del[k], delks);
            const float rdelks = mpas_mul(del[k], delks);
            ubar = mpas_add(ubar, mpas_mul(rcsks, u1[k]));
            vbar = mpas_add(vbar, mpas_mul(rcsks, v1[k]));
            rhobar = mpas_add(rhobar, mpas_mul(rdelks, rho[k]));
        }
    }

    // :343-383 wind direction and directional statistics.
    const int nwdir[8] = {6, 7, 5, 8, 2, 3, 1, 4};
    const float fdir = mpas_div(8.0f, mpas_mul(2.0f, pi));
    const float wdir = mpas_add(kim_atan2f(ubar, vbar), pi);
    const int nearest = td_nint(mpas_mul(fdir, wdir));
    const int nwd = nwdir[nearest % 8];
    const int direction = (nwd - 1) % 4;
    const int sign = 1 - 2 * ((nwd - 1) / 4);
    const float oa = mpas_mul((float)sign, oa_input[direction]);
    const float ol = mpas_max(ol_input[direction], olmin);
    const float oc = mpas_min(mpas_max(oc1[cell], oc_min), oc_max);
    const float ol_perp[4] = {
        ol_input[1], ol_input[0], ol_input[3], ol_input[2]};
    const float olp = mpas_max(ol_perp[direction], olmin);
    float od = mpas_div(olp, ol);
    od = mpas_min(od, odmax);
    od = mpas_max(od, odmin);
    const float dxy = dxy4[direction];
    const float dxyp = dxy4p[direction];

    for (int k = 0; k < nlev - 1; ++k) {   // :387-399
        const int i = k * ncells + cell;
        const int ia = i + ncells;
        const float ti = mpas_div(
            2.0f, mpas_add(temperature[i], temperature[ia]));
        const float rdz = mpas_div(1.0f, mpas_sub(zmid[ia], zmid[i]));
        const float du = mpas_sub(u1[k], u1[k + 1]);
        const float dv = mpas_sub(v1[k], v1[k + 1]);
        const float dw2 = mpas_add(mpas_mul(du, du), mpas_mul(dv, dv));
        const float shr2 = mpas_mul(
            mpas_mul(mpas_max(dw2, velmin), rdz), rdz);
        const float stability = mpas_add(
            mpas_div(gravity, cp), mpas_mul(rdz, mpas_sub(vtj[k + 1], vtj[k])));
        const float bvf2 = mpas_mul(mpas_mul(gravity, stability), ti);
        usqj[k] = mpas_max(mpas_div(bvf2, shr2), rimin);
        bnv2[k] = mpas_div(
            mpas_mul(mpas_mul(mpas_mul(2.0f, gravity), rdz),
                mpas_sub(vtk[k + 1], vtk[k])),
            mpas_add(vtk[k + 1], vtk[k]));
    }

    float ulow = mpas_sqrt(mpas_add(mpas_mul(ubar, ubar), mpas_mul(vbar, vbar)));   // :403-406
    ulow = mpas_max(ulow, 1.0f);
    const float rulow = mpas_div(1.0f, ulow);
    for (int k = 0; k < nlev - 1; ++k) {   // :410-419
        float value = mpas_mul(0.5f, mpas_add(
            mpas_mul(mpas_add(u1[k], u1[k + 1]), ubar),
            mpas_mul(mpas_add(v1[k], v1[k + 1]), vbar)));
        value = mpas_mul(value, rulow);
        if (value < velmin && value > 0.0f) value = velmin;
        velco[k] = value;
    }

    bool ldrag = omax <= hmt_min;   // :416-419 (no drag over low terrain)
    ldrag = ldrag || var <= 0.0f;
    ldrag = ldrag || velco[0] <= 0.0f;
    ldrag = ldrag || ulow == 1.0f;
    for (int kf = kgwdmin; kf <= kgwdmax; ++kf) {   // :424-429
        if (kf < kref) ldrag = ldrag || velco[kf - 1] <= 0.0f || bnv2[kf - 1] < 0.0f;
    }

    const float wtkbj = mpas_mul(   // :433-437
        mpas_sub(p_mass_dry_hyd[cell], p_mass_dry_hyd[ncells + cell]), delks1);
    bnv2[0] = mpas_mul(wtkbj, bnv2[0]);
    usqj[0] = mpas_mul(wtkbj, usqj[0]);
    for (int kf = kgwdmin; kf <= kgwdmax; ++kf) {   // :439-447
        if (kf < kref) {
            const int k = kf - 1;
            const float rdelks = mpas_mul(
                mpas_sub(p_mass_dry_hyd[k * ncells + cell],
                    p_mass_dry_hyd[(k + 1) * ncells + cell]), delks1);
            bnv2[0] = mpas_add(bnv2[0], mpas_mul(bnv2[k], rdelks));
            usqj[0] = mpas_add(usqj[0], mpas_mul(usqj[k], rdelks));
        }
    }
    ldrag = ldrag || bnv2[0] <= 0.0f;   // :450
    for (int kf = kgwdmin; kf <= kgwdmax; ++kf) {   // :455-459
        if (kf < kref) usqj[kf - 1] = usqj[0];
    }

    float bnv = 0.0f;
    float fr = 0.0f;
    float xn = 0.0f;
    float yn = 0.0f;
    if (!ldrag) {   // :461-469
        bnv = mpas_sqrt(bnv2[0]);
        fr = mpas_mul(mpas_mul(mpas_mul(bnv, rulow), var), od);
        fr = mpas_min(fr, frmax);
        xn = mpas_mul(ubar, rulow);
        yn = mpas_mul(vbar, rulow);
    }

    float coefm = 0.0f;
    float taub = 0.0f;
    if (!ldrag) {   // :473-494
        float efac = gfk_pow(mpas_add(oa, 2.0f), mpas_div(mpas_mul(ce, fr), frc));
        efac = mpas_min(mpas_max(efac, efacmin), efacmax);
        coefm = gfk_pow(mpas_add(1.0f, ol), mpas_add(oa, 1.0f));
        const float xlinv = mpas_div(coefm, dx_eff);
        const float tem = mpas_mul(mpas_mul(fr, fr), oc);
        const float gfac = mpas_div(tem, mpas_add(tem, cg));
        float value = mpas_mul(rhobar, efac);
        value = mpas_mul(value, xlinv);
        value = mpas_mul(value, gfac);
        value = mpas_mul(value, ulow);
        value = mpas_mul(value, ulow);
        value = mpas_mul(value, ulow);
        taub = mpas_div(value, bnv);
        if (if_nonhyd) {
            // nhd = -9/8 Fr^2 + exp(-2/Fr) (-5/4/Fr^2 - 1/(2Fr) + 5/4 + 9/4 Fr + 9/8 Fr^2)
            const float fr2 = mpas_mul(fr, fr);
            float poly = -mpas_div(1.25f, fr2);
            poly = mpas_sub(poly, mpas_div(0.5f, fr));
            poly = mpas_add(poly, 1.25f);
            poly = mpas_add(poly, mpas_mul(2.25f, fr));
            poly = mpas_add(poly, mpas_mul(1.125f, fr2));
            const float nhd_effect = mpas_add(-mpas_mul(1.125f, fr2),
                mpas_mul(gfk_exp(-mpas_div(2.0f, fr)), poly));
            taub = mpas_mul(taub, mpas_add(1.0f, nhd_effect));
        }
    }

    for (int kf = 1; kf <= kgwdmax; ++kf) {   // :498-502
        if (kf <= kref) taup[kf - 1] = taub;
    }

    bool icrilv = false;
    float brvf = 0.0f;
    for (int kf = kgwdmin; kf <= kte - 1; ++kf) {   // :504-545
        const int k = kf - 1;
        const int kp1 = k + 1;
        if (kf >= kref) {
            icrilv = icrilv || usqj[k] < ric || velco[k] <= 0.0f;
            brvf = mpas_max(bnv2[k], bnv2min);
            brvf = mpas_sqrt(brvf);
        }
        if (kf >= kref && !ldrag && !icrilv && taup[k] > 0.0f) {
            const float temv = mpas_div(1.0f, velco[k]);
            float tem1 = mpas_div(coefm, dxy);
            tem1 = mpas_mul(tem1, mpas_add(rho[kp1], rho[k]));
            tem1 = mpas_mul(tem1, brvf);
            tem1 = mpas_mul(tem1, velco[k]);
            tem1 = mpas_mul(tem1, 0.5f);
            float hd = mpas_sqrt(mpas_div(taup[k], tem1));
            const float fro = mpas_mul(mpas_mul(brvf, hd), temv);
            const float tem2 = mpas_sqrt(usqj[k]);
            const float tem = mpas_add(1.0f, mpas_mul(tem2, fro));
            const float rim = mpas_div(
                mpas_mul(usqj[k], mpas_sub(1.0f, fro)), mpas_mul(tem, tem));
            if (rim <= ric) {
                const float temc = mpas_add(2.0f, mpas_div(1.0f, tem2));
                hd = mpas_div(
                    mpas_mul(velco[k],
                        mpas_sub(mpas_mul(2.0f, mpas_sqrt(temc)), temc)),
                    brvf);
                taup[kp1] = mpas_mul(mpas_mul(tem1, hd), hd);
            } else {
                taup[kp1] = taup[k];
            }
        }
    }

    if (!ldrag) {   // :553-588 flow blocking from omax
        int kblk = 0;
        float fbdpe = 0.0f;
        float zblk = 0.0f;
        for (int kf = kte; kf >= kgwdmin; --kf) {
            if (kblk == 0 && kf <= kbomax) {
                const int k = kf - 1;
                float contribution = mpas_mul(
                    bnv2[k], mpas_sub(ZL(komax_read), ZL(kf)));
                contribution = mpas_mul(contribution, del[k]);
                contribution = mpas_div(contribution, gravity);
                contribution = mpas_div(contribution, rho[k]);
                fbdpe = mpas_add(fbdpe, contribution);
                const float fbdke = mpas_mul(0.5f,
                    mpas_add(mpas_mul(u1[k], u1[k]), mpas_mul(v1[k], v1[k])));
                if (fbdpe >= fbdke && kf <= komax) {
                    kblk = kf;
                    zblk = mpas_sub(ZL(kblk), ZL(1));
                }
            }
        }
        if (kblk != 0) {
            const float fbdcd = mpas_max(
                mpas_sub(2.0f, mpas_div(1.0f, od)), 0.0f);
            float value = mpas_mul(0.5f, rhobar);
            value = mpas_mul(value, coefm);
            value = mpas_div(value, area);
            value = mpas_mul(value, fbdcd);
            value = mpas_mul(value, dxyp);
            value = mpas_mul(value, olp);
            value = mpas_mul(value, zblk);
            value = mpas_mul(value, mpas_mul(ulow, ulow));
            taufb[0] = value;
            const float tautem = mpas_div(taufb[0], (float)(kblk - 1));
            for (int kf = 2; kf <= kblk; ++kf)
                taufb[kf - 1] = mpas_sub(taufb[kf - 2], tautem);
            for (int k = 0; k <= nlev; ++k)
                taup[k] = mpas_add(taup[k], taufb[k]);
        }
    }
#undef ZL

    for (int k = 0; k < nlev; ++k) {   // :592-596
        taud[k] = mpas_div(mpas_mul(mpas_mul(1.0f,
            mpas_sub(taup[k + 1], taup[k])), gravity), del[k]);
    }
    for (int kf = 1; kf <= kgwdmax - 1; ++kf) {   // :600-606 per-level reversal limiter
        const int k = kf - 1;
        if (mpas_abs(taud[k]) >= qmin) {
            dtfac[k] = mpas_min(dtfac[k],
                mpas_abs(mpas_div(velco[k], mpas_mul(dt, taud[k]))));
        }
    }
    for (int kf = kgwdmax; kf <= kte - 1; ++kf) {   // :610-619 mesosphere limiter
        const int k = kf - 1;
        if (mpas_abs(taud[k]) >= qmin && p_mass_dry_hyd[k * ncells + cell] <= pcutoff) {
            const float denfac = mpas_min(mpas_div(rho[k], pcutoff_den), 1.0f);
            dtfac[k] = mpas_min(dtfac[k], mpas_mul(denfac,
                mpas_abs(mpas_div(velco[k], mpas_mul(dt, taud[k])))));
        }
    }
    dtfac[kte - 1] = dtfac[kte - 2];   // :620

    float surface_u = 0.0f;
    float surface_v = 0.0f;
    for (int k = 0; k < nlev; ++k) {   // :627-639 and :648-662
        const int i = k * ncells + cell;
        taud[k] = mpas_mul(taud[k], dtfac[k]);
        const float du = mpas_mul(taud[k], xn);
        const float dv = mpas_mul(taud[k], yn);
        dtaux3d[i] = mpas_add(mpas_mul(du, 1.0f), mpas_mul(dv, 0.0f));
        dtauy3d[i] = mpas_add(mpas_mul(-du, 0.0f), mpas_mul(dv, 1.0f));
        surface_u = mpas_add(surface_u, mpas_mul(du, del[k]));
        surface_v = mpas_add(surface_v, mpas_mul(dv, del[k]));
        rublten_out[i] = mpas_add(mpas_add(rublten_in[i], mpas_mul(du, 1.0f)), mpas_mul(dv, 0.0f));
        rvblten_out[i] = mpas_add(mpas_sub(rvblten_in[i], mpas_mul(du, 0.0f)), mpas_mul(dv, 1.0f));
        rubldiff[i] = mpas_sub(rublten_out[i], rublten_in[i]);
        rvbldiff[i] = mpas_sub(rvblten_out[i], rvblten_in[i]);
    }
    const float stress_u = mpas_mul(mpas_div(-1.0f, gravity), surface_u);
    const float stress_v = mpas_mul(mpas_div(-1.0f, gravity), surface_v);
    dusfcg[cell] = mpas_add(mpas_mul(stress_u, 1.0f), mpas_mul(stress_v, 0.0f));
    dvsfcg[cell] = mpas_add(mpas_mul(-stress_u, 0.0f), mpas_mul(stress_v, 1.0f));
}

"""


@lru_cache(maxsize=16)
def kernel_source(nlev: int, native_dx: bool = False) -> str:
    """MMM 63c806d bl_gwdo.F90:38-680 for one compiled level count (local column scratch only)."""
    if not isinstance(nlev, int) or nlev < 4:
        # kgwdmax = kte/2 must be >= kgwdmin = 2 for the native reference-level clamp (:281).
        raise ValueError("KIM GWDO needs at least four levels (bl_gwdo.F90:213,281)")
    helpers = CUDA_FTZ_HELPERS
    for operation, intrinsic in (("+", "__fadd_rn"), ("-", "__fsub_rn"), ("*", "__fmul_rn"), ("/", "__fdiv_rn")):
        helpers = helpers.replace(f"const float ordinary = left {operation} right;",
                                  f"const float ordinary = {intrinsic}(left, right);")
    helpers = helpers.replace("return sqrtf(value);", "return __fsqrt_rn(value);")
    body = _KERNEL_BODY.replace("@NLEV@", str(nlev))
    if native_dx:
        body = body.replace("mpas_div(len_disp, gfk_pow(md, 0.25f))", "md")
    return _ysu._device_math() + "\n" + helpers + body


@lru_cache(maxsize=None)
def _module(nlev: int, device_id: int, native_dx: bool = False) -> Any:
    """Compile MMM 63c806d bl_gwdo_run (one entry per card and variant; see ysu_gwdo_v842._module)."""
    import cupy as cp
    from cupy.cuda.compiler import compile_using_nvrtc
    with cp.cuda.Device(device_id):
        binary, _ = compile_using_nvrtc(kernel_source(nlev, native_dx), _hex_numerics.options(COMPILE_OPTIONS),
                                        filename="kim_gwdo_2026.cu", cache_in_memory=True)
        module = cp.cuda.function.Module()
        module.load(binary.encode() if isinstance(binary, str) else binary)
    return module


def run_bl_kim_gwdo_cuda_2026(*, dt_seconds: float, config_len_disp: float | None = None,
                              constants: Mapping[str, float] | None = None,
                              dx_p: Any = None, dx_factor: float = DEFAULT_DX_FACTOR,
                              if_nonhyd: bool = DEFAULT_IF_NONHYD,
                              **arrays: Any) -> dict[str, Any]:
    """MPAS PR #1409 driver_gwdo bl_kim_gwdo branch; resident arrays in, resident arrays out."""
    import cupy as cp
    if dx_p is None and config_len_disp is None:
        raise ValueError("resolved len_disp required for the mesh-density division")
    dt = np.float32(dt_seconds)
    length = np.float32(1 if dx_p is not None else config_len_disp)
    factor = np.float32(dx_factor)
    if not math.isfinite(float(dt)) or dt <= 0 or not math.isfinite(float(length)) or length <= 0:
        raise ValueError("positive finite dt and resolved len_disp required to avoid division by zero")
    if not math.isfinite(float(factor)) or factor <= 0:
        raise ValueError("config_gwdo_factor must be finite and positive: it divides the base stress (bl_gwdo.F90:478)")
    shape = tuple(arrays["u_p"].shape)
    if len(shape) != 2 or shape[0] < 4 or shape[1] < 1:
        raise ValueError("KIM GWDO needs [level>=4,column>=1] arrays (bl_gwdo.F90:213,281)")
    nlev, ncells = shape
    for name in LAYER_INPUTS:
        require_resident_array(name, arrays[name], dtype=np.float32, shape=shape)
    require_resident_array("pres2_hydd_p", arrays["pres2_hydd_p"], dtype=np.float32, shape=(nlev + 1, ncells))
    if dx_p is not None:
        require_resident_array("dx_p", dx_p, dtype=np.float32, shape=(ncells,))
    for name in STATIC_INPUTS[1:] if dx_p is not None else STATIC_INPUTS:
        require_resident_array(name, arrays[name], dtype=np.float32, shape=(ncells,))
    c = dict(gravity=np.float32(9.80616), cp=np.float32(1004.5), rd=np.float32(287),
             fv=np.float32(np.float32(461.6) / np.float32(287) - np.float32(1)),
             pi=np.float32(3.141592653589793))
    if constants is not None:
        if set(constants) != set(c):
            raise ValueError(f"constants must contain {tuple(c)}")
        c = {name: np.float32(constants[name]) for name in c}
    output = {name: cp.empty(shape, dtype=cp.float32) for name in LAYER_OUTPUTS}
    output.update({name: cp.empty(ncells, dtype=cp.float32) for name in SURFACE_OUTPUTS})
    invalid = cp.zeros(1, dtype=cp.int32)
    dynamic = ("u_p", "v_p", "t_p", "qv_p", "pres_hydd_p", "pres2_hydd_p", "pi_p", "zmid_p", "rublten", "rvblten")
    args = (np.int32(nlev), np.int32(ncells), dt, length, *c.values(), factor, np.int32(bool(if_nonhyd)),
            *(arrays[name] for name in dynamic),
            dx_p if dx_p is not None else arrays["meshDensity"],
            *(arrays[name] for name in STATIC_INPUTS[1:]),
            *(output[name] for name in ("rublten", "rvblten", "dtaux3d", "dtauy3d", "dusfcg", "dvsfcg",
                                        "rubldiff", "rvbldiff")), invalid)
    module = _module(nlev, int(cp.cuda.runtime.getDevice()), dx_p is not None)
    module.get_function("bl_kim_gwdo_2026_f32")(((ncells + 63) // 64,), (64,), args)
    from .deferred_checks_v842 import defer
    defer(invalid, "KIM GWDO compiled level count differs from its launch", error=RuntimeError)
    return output


def gwdo_from_MPAS(prepared: Mapping[str, Any], mesh: Mapping[str, Any],
                   sfc_input: Mapping[str, Any], tend_physics: Mapping[str, Any]) -> dict[str, Any]:
    """MPAS PR #1409 gwdo_from_MPAS bl_kim_gwdo branch: the v8.4.2 carriers plus elvmax."""
    if "elvmax" not in sfc_input:
        raise KeyError(
            "bl_kim_gwdo reads the static field elvmax (maximum sub-grid orography, m MSL; MPAS PR #1408); "
            "without it every cell is no-drag (bl_gwdo.F90:416). Build the static with "
            "rw_mpas_static --gwd-recipe kim2026, or select config_gwdo_scheme='bl_ysu_gwdo' (gwdo_version='v842')")
    result = _ysu.gwdo_from_MPAS(prepared, mesh, sfc_input, tend_physics)
    result["elvmax"] = sfc_input["elvmax"]
    return result


gwdo_to_MPAS = _ysu.gwdo_to_MPAS


def driver_gwdo(*, prepared: Mapping[str, Any], mesh: Mapping[str, Any],
                sfc_input: Mapping[str, Any], tend_physics: MutableMapping[str, Any],
                diag_physics: MutableMapping[str, Any], dt_pbl: float,
                config_len_disp: float, constants: Mapping[str, float] | None = None,
                dx_factor: float = DEFAULT_DX_FACTOR, if_nonhyd: bool = DEFAULT_IF_NONHYD,
                kernel_artifact: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """MPAS PR #1409 driver_gwdo, bl_kim_gwdo branch; whole driver slice."""
    del kernel_artifact  # the AOT artifact route is the v842 kernel's; this one compiles in-process
    inputs = gwdo_from_MPAS(prepared, mesh, sfc_input, tend_physics)
    result = run_bl_kim_gwdo_cuda_2026(dt_seconds=dt_pbl, config_len_disp=config_len_disp,
                                       constants=constants, dx_factor=dx_factor, if_nonhyd=if_nonhyd,
                                       **inputs)
    gwdo_to_MPAS(result, diag_physics, tend_physics)
    _ysu._discard_native_profile_inputs(prepared)
    return result


def bind_forecast(provider: Any, call: Any) -> dict[str, Any]:
    """MPAS PR #1409 driver_gwdo: the v8.4.2 binding plus config_gwdo_factor / config_gwdo_nonhyd."""
    bound = _ysu.bind_forecast(provider, call)
    configs = provider["configs"]
    bound["dx_factor"] = float(configs.get("config_gwdo_factor", DEFAULT_DX_FACTOR))
    bound["if_nonhyd"] = bool(configs.get("config_gwdo_nonhyd", DEFAULT_IF_NONHYD))
    return bound


def forecast_callback(call: Any) -> dict[str, Any]:
    """MPAS PR #1409 driver_gwdo bl_kim_gwdo branch; whole slot."""
    return driver_gwdo(**call.pools)


def register_backend(registry: Any, *, arguments: Any) -> None:
    """Register the hex-cuda bl_kim_gwdo row beside bl_ysu_gwdo (the pin)."""
    if not callable(arguments):
        raise TypeError("KIM GWDO registration needs the forecast argument provider")

    def launch(call: Any) -> dict[str, Any]:
        return driver_gwdo(**arguments(call))

    registry.register("gwdo", "bl_kim_gwdo", BACKEND, launch)


def kernel_sha256(nlev: int) -> str:
    return sha256(kernel_source(nlev).encode()).hexdigest()


#: MPAS-Model PR #1409 Registry.xml hunks (src/core_atmosphere/Registry.xml:521,2279,2452-2462,3791-3795).
REGISTRY_OPTIONS = (
    ("config_gwdo_factor", "real", "2.0", "-", "Effective grid length ratio in kim_gwdo scheme",
     "Non-negative real values"),
    ("config_gwdo_nonhyd", "logical", "true", "-", "Logical index for nonhydrostatic effect in kim_gwdo scheme",
     ".true. or .false."),
)
ELVMAX_ATTRIBUTES = {"name": "elvmax", "type": "real", "dimensions": "nCells", "units": "m",
                     "description": "elevation maximum over a grid cell"}


def prepare_registry(root: Any, config: Any) -> bool:
    """Apply the MPAS-Model PR #1409 Registry hunks to a parsed Registry when the run selects bl_kim_gwdo.

    elvmax joins sfc_input and the stream that reads ol4 (the invariant stream, which input and restart
    include), and config_gwdo_factor / config_gwdo_nonhyd join the physics record. A run on any other drag
    leaves the Registry untouched, so its pools, streams and history attributes are word for word what they
    were (the v8.4.2 pin). Returns True when it changed the Registry."""
    import xml.etree.ElementTree as ET
    selectors = getattr(config, "selectors", None) or {}
    if selectors.get("gwdo") != NAMELIST_OPTION[1]:
        return False

    def after(parent: Any, name: str, element: Any) -> None:
        children = list(parent)
        index = next(i for i, child in enumerate(children) if child.get("name") == name)
        parent.insert(index + 1, element)

    structs = [s for s in root.iter("var_struct")
               if any(v.tag == "var" and v.get("name") == "ol4" and v.get("type") for v in s)]
    if len(structs) != 1:
        raise ValueError(f"the Registry defines ol4 in {len(structs)} var_structs; PR #1409 adds elvmax beside it "
                         "in sfc_input")
    if not any(v.get("name") == "elvmax" for v in structs[0].iter("var")):
        after(structs[0], "ol4", ET.Element("var", dict(ELVMAX_ATTRIBUTES)))
    streams = [s for s in root.iter("stream")
               if any(v.tag == "var" and v.get("name") == "ol4" and not v.get("type") for v in s)]
    if not streams:
        raise ValueError("no Registry stream reads ol4; PR #1409 reads elvmax with it (the invariant/input stream)")
    for stream in streams:
        if not any(v.get("name") == "elvmax" for v in stream):
            after(stream, "ol4", ET.Element("var", {"name": "elvmax"}))
    records = [r for r in root.iter("nml_record") if any(o.get("name") == "config_gwdo_scheme" for o in r)]
    if len(records) != 1:
        raise ValueError("the Registry has no single nml_record holding config_gwdo_scheme")
    record = records[0]
    for option in record:
        if option.get("name") == "config_gwdo_scheme" and "bl_kim_gwdo" not in (option.get("possible_values") or ""):
            option.set("possible_values", (option.get("possible_values") or "") + ",`bl_kim_gwdo'")
    present = {o.get("name") for o in record}
    for name, kind, default, units, description, values in REGISTRY_OPTIONS:
        if name not in present:
            record.append(ET.Element("nml_option", {"name": name, "type": kind, "default_value": default,
                                                    "in_defaults": "false", "units": units,
                                                    "description": description, "possible_values": values}))
    return True
