"""MPAS-A v8.4.2 YSU orographic GWD, complete resident column operator.

Authority: physics_mmm/bl_gwdo.F90:20-649; physics_wrf/module_bl_gwdo.F:15-237;
mpas_atmphys_driver_gwdo.F:285-552,555-689,692-834. These three files are
SHA-identical to the v8.4.1 authority. The whole existing MPAS CUDA algorithm
is reused, with strict GNU arithmetic corrections measured by the standalone
pristine-Fortran oracle. WOOF and cuda_gwdo_v841 remain untouched.

Carriers are resident RKIND float32 C [level,column], surface to top. Driver
aliases (including dry hydrostatic pressures) follow MPAS, with sina=0/cosa=1,
dx=config_len_disp/meshDensity**0.25, additive PBL tendencies and rounded
rubldiff/rvbldiff. Thermal tendency is preserved by gwdo_to_MPAS. No tables or
persistent state. The namelist selector is bl_ysu_gwdo; the integration-owned
seam binds run_bl_ysu_gwdo_cuda_v842 as the hex-cuda implementation.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from collections.abc import Mapping, MutableMapping
from functools import lru_cache
from hashlib import sha256
from importlib import resources
import math
from typing import Any

import numpy as np

from ..cuda_fp32 import CUDA_FTZ_HELPERS
from ..cuda_backend.containers import require_resident_array

SCHEME = "ysu_gwdo"
NAMELIST_OPTION = ("config_gwdo_scheme", "bl_ysu_gwdo")
BACKEND = "hex-cuda"
SOURCE_SHA256 = {
    "physics_mmm/bl_gwdo.F90": "c294ef2eb9da1ee9ce6594b97adea7280eb18760e0c5576ab7149c35bb7c3c3b",
    "physics_wrf/module_bl_gwdo.F": "9784f201716e026fc66c531ca0012f9dfc3098112bb7b25b3da1ec8168aa52e6",
    "mpas_atmphys_driver_gwdo.F": "ec76abad1cd841b3a15cb63df494060488cea1026c532a628e61e1d71871fb54",
}
LAYER_INPUTS = ("u_p", "v_p", "t_p", "qv_p", "pres_hydd_p", "pi_p", "zmid_p", "rublten", "rvblten")
STATIC_INPUTS = ("meshDensity", "var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4")
LAYER_OUTPUTS = ("rublten", "rvblten", "dtaux3d", "dtauy3d", "rubldiff", "rvbldiff")
SURFACE_OUTPUTS = ("dusfcg", "dvsfcg")
_NATIVE_PROFILE_INPUTS = (*LAYER_INPUTS[:-2], "pres2_hydd_p")
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")


def bl_gwdo_init() -> tuple[str, int]:
    """MPAS physics_mmm/bl_gwdo.F90:20-33; no scheme state to allocate."""
    return "bl_gwdo_init OK", 0


def bl_gwdo_finalize() -> tuple[str, int]:
    """MPAS physics_mmm/bl_gwdo.F90:40-53; no scheme state to release."""
    return "bl_gwdo_finalize OK", 0


def gwdo_from_MPAS(prepared: Mapping[str, Any], mesh: Mapping[str, Any],
                   sfc_input: Mapping[str, Any], tend_physics: Mapping[str, Any]) -> dict[str, Any]:
    """MPAS mpas_atmphys_driver_gwdo.F:285-365,520-550 and :761-781.

    Native [level,column] arrays already have the kernel's resident layout;
    wrapper [column,level,1] copies are fused into the same column kernel.
    kpbl and incoming diagnostics are overwritten/dead in module_bl_gwdo.
    """
    # Native prepared globals extend through kme; YSU runs kts:kte and the
    # Registry tendencies provide that physical extent (:520-550,761-781).
    # Retain the extra interface pressure, excluding mass-level ghosts.
    nlev = tend_physics["rublten"].shape[0]
    result = {name: prepared[name][:nlev, :] for name in LAYER_INPUTS[:-2]}
    result["pres2_hydd_p"] = prepared["pres2_hydd_p"][:nlev + 1, :]
    result.update({name: tend_physics[name] for name in ("rublten", "rvblten")})
    result["meshDensity"] = mesh["meshDensity"]
    result.update({name: sfc_input[name] for name in STATIC_INPUTS[1:]})
    return result


def gwdo_to_MPAS(result: Mapping[str, Any], diag_physics: MutableMapping[str, Any],
                 tend_physics: MutableMapping[str, Any]) -> None:
    """MPAS mpas_atmphys_driver_gwdo.F:555-600,668-689.

    Commit device candidates after validation. Native rthblten is unchanged.
    rubldiff/rvbldiff are computed before overwriting the PBL tendencies.
    """
    for name in ("rublten", "rvblten"):
        tend_physics[name][...] = result[name]
    for name in ("dtaux3d", "dtauy3d", "rubldiff", "rvbldiff", *SURFACE_OUTPUTS):
        diag_physics[name][...] = result[name]


@lru_cache(maxsize=1)
def _device_math() -> str:
    """MPAS bl_gwdo.F90:337-355; mpas_atmphys_driver_gwdo.F:533.

    Reuse shared glibc device maths; do not transcribe elementary functions.

    glibc_flt32.cuh supplies gfk_pow (measured against host glibc 2.43).
    terrain_drag.cu's shared prefix supplies its atan2f and Fortran NINT.
    """
    kernels = resources.files("woof").joinpath("core/kernels")
    libm = kernels.joinpath("glibc_flt32.cuh").read_text(encoding="utf-8")
    terrain = kernels.joinpath("terrain_drag.cu").read_text(encoding="utf-8")
    prefix = terrain.split('// --------------------------------------------------------------------------', 1)[0]
    return libm + "\n" + prefix


_KERNEL_BODY = r"""

#define GWDO_NLEV @NLEV@

// MPAS physics_mmm/bl_gwdo.F90:55-643; mpas_atmphys_driver_gwdo.F:761-781.
extern "C" __global__ void bl_ysu_gwdo_v842_f32(
    const int nlev, const int ncells, const float dt,
    const float len_disp, const float gravity, const float cp,
    const float rd, const float fv, const float pi,
    const float *u, const float *v, const float *temperature,
    const float *qv, const float *p_mass_dry_hyd,
    const float *p_interface_dry_hyd, const float *exner,
    const float *zmid, const float *rublten_in,
    const float *rvblten_in, const float *mesh_density,
    const float *var2d, const float *oc1, const float *oa1,
    const float *oa2, const float *oa3, const float *oa4,
    const float *ol1, const float *ol2, const float *ol3,
    const float *ol4, float *rublten_out, float *rvblten_out,
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

    // Native bl_gwdo.F90:218-643 performs these operations without a
    // humidity/geometry/finite-value gate. Preserve its arithmetic even for
    // values that produce nonfinite outputs; do not invent a forecast abort.
    const float md = mesh_density[cell];
    const float oa_input[4] = {oa1[cell], oa2[cell], oa3[cell], oa4[cell]};
    const float ol_input[4] = {ol1[cell], ol2[cell], ol3[cell], ol4[cell]};

    float vtj[GWDO_NLEV];
    float vtk[GWDO_NLEV];
    float rho[GWDO_NLEV];
    float delta_p[GWDO_NLEV];
    float u1[GWDO_NLEV];
    float v1[GWDO_NLEV];
    float usqj[GWDO_NLEV];
    float bnv2[GWDO_NLEV];
    float velco[GWDO_NLEV - 1];
    float taup[GWDO_NLEV + 1];
    float taufb[GWDO_NLEV + 1];
    float taud[GWDO_NLEV];
    for (int k = 0; k < nlev; ++k) {
        usqj[k] = 0.0f;
        bnv2[k] = 0.0f;
        taud[k] = 0.0f;
    }
    for (int k = 0; k <= nlev; ++k) {
        taup[k] = 0.0f;
        taufb[k] = 0.0f;
    }

    const float dx = mpas_div(len_disp, gfk_pow(md, 0.25f));
    const float diagonal = mpas_sqrt(
        mpas_add(mpas_mul(dx, dx), mpas_mul(dx, dx)));
    const float dxy4[4] = {dx, dx, diagonal, diagonal};
    const float dxy4p[4] = {dx, dx, diagonal, diagonal};
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        const int ip = (k + 1) * ncells + cell;
        const float virtual_factor = mpas_add(1.0f, mpas_mul(fv, qv[i]));
        vtj[k] = mpas_mul(temperature[i], virtual_factor);
        vtk[k] = mpas_div(vtj[k], exner[i]);
        rho[k] = mpas_div(
            mpas_mul(mpas_div(1.0f, rd), p_mass_dry_hyd[i]), vtj[k]);
        delta_p[k] = mpas_sub(
            p_interface_dry_hyd[i], p_interface_dry_hyd[ip]);
        u1[k] = mpas_sub(mpas_mul(u[i], 1.0f), mpas_mul(v[i], 0.0f));
        v1[k] = mpas_add(mpas_mul(u[i], 0.0f), mpas_mul(v[i], 1.0f));
    }

    const float zlowtop = mpas_mul(2.0f, var2d[cell]);
    int klowtop = 0;
    for (int k = 1; k < nlev; ++k) {
        const float height = mpas_sub(
            zmid[k * ncells + cell], zmid[cell]);
        if (zlowtop > 0.0f && klowtop == 0 && height >= zlowtop)
            klowtop = k + 2;
    }
    int kbl = klowtop;
    if (kbl > nlev) kbl = nlev;
    if (kbl < 2) kbl = 2;
    const float delks = mpas_div(
        1.0f, mpas_sub(p_interface_dry_hyd[cell],
            p_interface_dry_hyd[(kbl - 1) * ncells + cell]));
    const float delks1 = mpas_div(
        1.0f, mpas_sub(p_mass_dry_hyd[cell],
            p_mass_dry_hyd[(kbl - 1) * ncells + cell]));
    float ubar = 0.0f;
    float vbar = 0.0f;
    float rhobar = 0.0f;
    for (int k = 0; k < nlev; ++k) {
        if (k + 1 < kbl) {
            const float weight = mpas_mul(delta_p[k], delks);
            ubar = mpas_add(ubar, mpas_mul(weight, u1[k]));
            vbar = mpas_add(vbar, mpas_mul(weight, v1[k]));
            rhobar = mpas_add(rhobar, mpas_mul(weight, rho[k]));
        }
    }

    const int nwdir[8] = {6, 7, 5, 8, 2, 3, 1, 4};
    const float fdir = mpas_div(8.0f, mpas_mul(2.0f, pi));
    const float wdir = mpas_add(td_atan2f(ubar, vbar), pi);
    const int nearest = td_nint(mpas_mul(fdir, wdir));
    const int nwd = nwdir[nearest % 8];
    const int direction = (nwd - 1) % 4;
    const int sign = 1 - 2 * ((nwd - 1) / 4);
    const float oa = mpas_mul((float)sign, oa_input[direction]);
    const float ol = ol_input[direction];
    const float ol_perp[4] = {
        ol_input[1], ol_input[0], ol_input[3], ol_input[2]};
    const float olp = ol_perp[direction];
    float od = mpas_div(olp, mpas_max(ol, 1.0e-5f));
    od = mpas_max(mpas_min(od, 10.0f), 0.1f);
    const float dxy = dxy4[direction];
    const float dxyp = dxy4p[direction];

    for (int k = 0; k < nlev - 1; ++k) {
        const int i = k * ncells + cell;
        const int ia = i + ncells;
        const float ti = mpas_div(
            2.0f, mpas_add(temperature[i], temperature[ia]));
        const float rdz = mpas_div(1.0f, mpas_sub(zmid[ia], zmid[i]));
        const float du = mpas_sub(u1[k], u1[k + 1]);
        const float dv = mpas_sub(v1[k], v1[k + 1]);
        const float dw2 = mpas_add(mpas_mul(du, du), mpas_mul(dv, dv));
        const float shr2 = mpas_mul(
            mpas_mul(mpas_max(dw2, 1.0f), rdz), rdz);
        const float stability = mpas_add(
            mpas_div(gravity, cp), mpas_mul(rdz, mpas_sub(vtj[k + 1], vtj[k])));
        const float bvf2 = mpas_mul(mpas_mul(gravity, stability), ti);
        usqj[k] = mpas_max(mpas_div(bvf2, shr2), -100.0f);
        bnv2[k] = mpas_div(
            mpas_mul(mpas_mul(mpas_mul(2.0f, gravity), rdz),
                mpas_sub(vtk[k + 1], vtk[k])),
            mpas_add(vtk[k + 1], vtk[k]));
    }

    float ulow = mpas_sqrt(mpas_add(mpas_mul(ubar, ubar), mpas_mul(vbar, vbar)));
    ulow = mpas_max(ulow, 1.0f);
    const float rulow = mpas_div(1.0f, ulow);
    for (int k = 0; k < nlev - 1; ++k) {
        float aligned = mpas_add(
            mpas_mul(mpas_add(u1[k], u1[k + 1]), ubar),
            mpas_mul(mpas_add(v1[k], v1[k + 1]), vbar));
        velco[k] = mpas_mul(mpas_mul(0.5f, aligned), rulow);
        if (velco[k] < 1.0f && velco[k] > 0.0f) velco[k] = 1.0f;
    }
    bool ldrag = velco[0] <= 0.0f;
    for (int kf = 2; kf <= nlev; ++kf) {
        if (kf < kbl && velco[kf - 1] <= 0.0f) ldrag = true;
    }

    const float first_weight = mpas_mul(
        mpas_sub(p_mass_dry_hyd[cell], p_mass_dry_hyd[ncells + cell]),
        delks1);
    bnv2[0] = mpas_mul(first_weight, bnv2[0]);
    usqj[0] = mpas_mul(first_weight, usqj[0]);
    for (int kf = 2; kf <= nlev; ++kf) {
        if (kf < kbl) {
            const int k = kf - 1;
            const float pressure_weight = mpas_mul(
                mpas_sub(p_mass_dry_hyd[k * ncells + cell],
                    p_mass_dry_hyd[(k + 1) * ncells + cell]), delks1);
            bnv2[0] = mpas_add(bnv2[0], mpas_mul(bnv2[k], pressure_weight));
            usqj[0] = mpas_add(usqj[0], mpas_mul(usqj[k], pressure_weight));
        }
    }
    ldrag = ldrag || bnv2[0] <= 0.0f || ulow == 1.0f || var2d[cell] <= 0.0f;
    for (int kf = 2; kf <= nlev; ++kf) {
        if (kf < kbl) usqj[kf - 1] = usqj[0];
    }

    float bnv = 0.0f;
    float fr = 0.0f;
    float xn = 0.0f;
    float yn = 0.0f;
    if (!ldrag) {
        bnv = mpas_sqrt(bnv2[0]);
        fr = mpas_mul(mpas_mul(mpas_mul(bnv, rulow), var2d[cell]), od);
        fr = mpas_min(fr, 10.0f);
        xn = mpas_mul(ubar, rulow);
        yn = mpas_mul(vbar, rulow);
    }

    float coefm = 0.0f;
    float taub = 0.0f;
    if (!ldrag) {
        float efact = gfk_pow(mpas_add(oa, 2.0f), mpas_mul(0.8f, fr));
        efact = mpas_min(mpas_max(efact, 0.0f), 10.0f);
        coefm = gfk_pow(mpas_add(1.0f, ol), mpas_add(oa, 1.0f));
        const float xlinv = mpas_div(coefm, dx);
        const float tem = mpas_mul(mpas_mul(fr, fr), oc1[cell]);
        const float gfobnv = mpas_div(
            tem, mpas_mul(mpas_add(tem, 0.5f), bnv));
        taub = mpas_mul(xlinv, rhobar);
        taub = mpas_mul(taub, ulow);
        taub = mpas_mul(taub, ulow);
        taub = mpas_mul(taub, ulow);
        taub = mpas_mul(mpas_mul(taub, gfobnv), efact);
    }
    for (int kf = 1; kf <= nlev; ++kf) {
        if (kf <= kbl) taup[kf - 1] = taub;
    }

    bool icrilv = false;
    for (int kf = 2; kf < nlev; ++kf) {
        const int k = kf - 1;
        const int kp1 = k + 1;
        float brvf = 0.0f;
        if (kf >= kbl) {
            icrilv = icrilv || usqj[k] < 0.25f || velco[k] <= 0.0f;
            brvf = mpas_sqrt(mpas_max(bnv2[k], 1.0e-5f));
        }
        if (kf >= kbl && !ldrag && !icrilv && taup[k] > 0.0f) {
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
            if (rim <= 0.25f) {
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

    if (!ldrag) {
        int kblk = 0;
        float fbdpe = 0.0f;
        float zblk = 0.0f;
        for (int kf = nlev; kf >= 2; --kf) {
            if (kblk == 0 && kf <= kbl) {
                const int k = kf - 1;
                float contribution = mpas_mul(
                    bnv2[k], mpas_sub(zmid[(kbl - 1) * ncells + cell],
                        zmid[k * ncells + cell]));
                contribution = mpas_mul(contribution, delta_p[k]);
                contribution = mpas_div(contribution, gravity);
                contribution = mpas_div(contribution, rho[k]);
                fbdpe = mpas_add(fbdpe, contribution);
                const float fbdke = mpas_mul(0.5f,
                    mpas_add(mpas_mul(u1[k], u1[k]), mpas_mul(v1[k], v1[k])));
                if (fbdpe >= fbdke) {
                    kblk = kf < kbl ? kf : kbl;
                    zblk = mpas_sub(zmid[(kblk - 1) * ncells + cell], zmid[cell]);
                }
            }
        }
        if (kblk != 0) {
            const float fbdcd = mpas_max(
                mpas_sub(2.0f, mpas_div(1.0f, od)), 0.0f);
            float value = mpas_mul(0.5f, rhobar);
            value = mpas_mul(value, coefm);
            value = mpas_div(value, mpas_mul(dx, dx));
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

    for (int k = 0; k < nlev; ++k) {
        taud[k] = mpas_div(
            mpas_mul(mpas_sub(taup[k + 1], taup[k]), gravity), delta_p[k]);
    }
    float dtfac = 1.0f;
    for (int kf = 1; kf < nlev; ++kf) {
        const int k = kf - 1;
        if (kf <= kbl && taud[k] != 0.0f) {
            const float limiter = mpas_abs(
                mpas_div(velco[k], mpas_mul(dt, taud[k])));
            dtfac = mpas_min(dtfac, limiter);
        }
    }

    float surface_u = 0.0f;
    float surface_v = 0.0f;
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        taud[k] = mpas_mul(taud[k], dtfac);
        const float du = mpas_mul(taud[k], xn);
        const float dv = mpas_mul(taud[k], yn);
        dtaux3d[i] = mpas_add(mpas_mul(du, 1.0f), mpas_mul(dv, 0.0f));
        dtauy3d[i] = mpas_add(mpas_mul(-du, 0.0f), mpas_mul(dv, 1.0f));
        surface_u = mpas_add(surface_u, mpas_mul(du, delta_p[k]));
        surface_v = mpas_add(surface_v, mpas_mul(dv, delta_p[k]));
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
    """MPAS physics_mmm/bl_gwdo.F90:55-643, with :531 specialization.

    A compiled level count sizes local column scratch; it imposes no mesh or
    level-count preset. NLEV>=2 prevents the native scheme's level-2 accesses
    from reading outside a column. The final driver diagnostics are :680-685.
    """
    if not isinstance(nlev, int) or nlev < 2:
        raise ValueError("YSU GWDO needs at least two levels (bl_gwdo.F90:418)")
    helpers = CUDA_FTZ_HELPERS
    for operation, intrinsic in (("+", "__fadd_rn"), ("-", "__fsub_rn"), ("*", "__fmul_rn"), ("/", "__fdiv_rn")):
        helpers = helpers.replace(f"const float ordinary = left {operation} right;",
                                  f"const float ordinary = {intrinsic}(left, right);")
    helpers = helpers.replace("return sqrtf(value);", "return __fsqrt_rn(value);")
    body = _KERNEL_BODY.replace("@NLEV@", str(nlev))
    if native_dx:
        # MPAS module_bl_gwdo.F:177,205: wrapper receives already resolved dx.
        # The same resident pointer slot carries dx instead of meshDensity.
        body = body.replace("mpas_div(len_disp, gfk_pow(md, 0.25f))", "md")
    return _device_math() + "\n" + helpers + body


# Unbounded on purpose (one entry per card and variant): a bounded per-card cache thrashes once ranks x
# variants exceed it, and every rank then rebuilds its module each step (thompson_v842._module, 8 cards,
# 2026-10-28: 1.28 s/step against 0.25 s with the cache fixed).
@lru_cache(maxsize=None)
def _module(nlev: int, device_id: int, native_dx: bool = False) -> Any:
    """Compile MPAS physics_mmm/bl_gwdo.F90:55-643 with contraction disabled."""
    import cupy as cp
    from cupy.cuda.compiler import compile_using_nvrtc
    # CuPy RawKernel appends -ftz=true after caller options. Direct NVRTC
    # preserves strict IEEE/subnormals; same path as shared terrain_drag.py.
    with cp.cuda.Device(device_id):
        binary, _ = compile_using_nvrtc(kernel_source(nlev, native_dx), _hex_numerics.options(COMPILE_OPTIONS),
                                        filename="ysu_gwdo_v842.cu", cache_in_memory=True)
        module = cp.cuda.function.Module()
        module.load(binary.encode() if isinstance(binary, str) else binary)
    return module


# Unbounded on purpose (one entry per card and variant): a bounded per-card cache thrashes once ranks x
# variants exceed it, and every rank then rebuilds its module each step (thompson_v842._module, 8 cards,
# 2026-10-28: 1.28 s/step against 0.25 s with the cache fixed).
@lru_cache(maxsize=None)
def _artifact_module(nlev: int, device_id: int, native_dx: bool,
                     source_hash: str, binary_hash: str,
                     options: tuple[str, ...], binary: bytes) -> Any:
    """Load the compiled MPAS physics_mmm/bl_gwdo.F90:55-643 operator.

    A build-queue caller can compile without occupying the GPU. Bind the
    artifact to the complete current source, strict options and device target;
    loading and execution still belong inside the GPU queue.
    """
    if source_hash != sha256(kernel_source(nlev, native_dx).encode()).hexdigest():
        raise ValueError("YSU compiled artifact does not match the current CUDA source")
    if not isinstance(binary, bytes) or sha256(binary).hexdigest() != binary_hash:
        raise ValueError("YSU compiled artifact binary hash mismatch")
    # Both refusals above need no device; CuPy is imported only to load it
    # (the public CPU tier has no CuPy and checks the refusals).
    import cupy as cp
    with cp.cuda.Device(device_id):
        architecture = cp.cuda.Device(device_id).compute_capability
        if options != (*_hex_numerics.options(COMPILE_OPTIONS), f"--gpu-architecture=sm_{architecture}"):
            raise ValueError("YSU compiled artifact needs strict options and the current device target")
        module = cp.cuda.function.Module()
        module.load(binary)
    return module


def run_bl_ysu_gwdo_cuda_v842(*, dt_seconds: float, config_len_disp: float | None = None,
                              constants: Mapping[str, float] | None = None,
                              dx_p: Any = None,
                              kernel_artifact: Mapping[str, Any] | None = None,
                              **arrays: Any) -> dict[str, Any]:
    """MPAS mpas_atmphys_driver_gwdo.F:692-834, YSU branch :761-781.

    config_len_disp is the already resolved MPAS length (the manager resolves
    nonpositive namelist values). Scalars/constants narrow to RKIND once.
    dx_p optionally supplies the native wrapper's resolved resident dx, without
    inverting or repeating the driver's mesh-density calculation. Returns
    resident arrays, preserving native arithmetic without a column-value gate.
    """
    import cupy as cp
    if dx_p is None and config_len_disp is None:
        raise ValueError("resolved len_disp required for the mesh-density division")
    dt = np.float32(dt_seconds)
    length = np.float32(1 if dx_p is not None else config_len_disp)
    if not math.isfinite(float(dt)) or dt <= 0 or not math.isfinite(float(length)) or length <= 0:
        raise ValueError("positive finite dt and resolved len_disp required to avoid division by zero")
    shape = tuple(arrays["u_p"].shape)
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 1:
        raise ValueError("YSU GWDO needs [level>=2,column>=1] arrays (bl_gwdo.F90:418)")
    nlev, ncells = shape
    for name in LAYER_INPUTS:
        require_resident_array(name, arrays[name], dtype=np.float32, shape=shape)
    require_resident_array("pres2_hydd_p", arrays["pres2_hydd_p"], dtype=np.float32,
                           shape=(nlev+1, ncells))
    if dx_p is not None:
        require_resident_array("dx_p", dx_p, dtype=np.float32, shape=(ncells,))
    for name in STATIC_INPUTS[1:] if dx_p is not None else STATIC_INPUTS:
        require_resident_array(name, arrays[name], dtype=np.float32, shape=(ncells,))
    # MPAS src/core_atmosphere/physics/mpas_atmphys_constants.F:37-66;
    # caller may pass native constants.
    c = dict(gravity=np.float32(9.80616), cp=np.float32(1004.5), rd=np.float32(287),
             fv=np.float32(np.float32(461.6)/np.float32(287)-np.float32(1)),
             pi=np.float32(3.141592653589793))
    if constants is not None:
        if set(constants) != set(c):
            raise ValueError(f"constants must contain {tuple(c)}")
        c = {name: np.float32(constants[name]) for name in c}
    if not all(math.isfinite(float(value)) and value > 0 for value in c.values()):
        raise ValueError("positive finite physics constants required to avoid division by zero")
    output = {name: cp.empty(shape, dtype=cp.float32) for name in LAYER_OUTPUTS}
    output.update({name: cp.empty(ncells, dtype=cp.float32) for name in SURFACE_OUTPUTS})
    invalid = cp.zeros(1, dtype=cp.int32)
    dynamic = ("u_p", "v_p", "t_p", "qv_p", "pres_hydd_p", "pres2_hydd_p", "pi_p", "zmid_p", "rublten", "rvblten")
    args = (np.int32(nlev), np.int32(ncells), dt, length, *c.values(),
            *(arrays[name] for name in dynamic),
            dx_p if dx_p is not None else arrays["meshDensity"],
            *(arrays[name] for name in STATIC_INPUTS[1:]),
            *(output[name] for name in ("rublten", "rvblten", "dtaux3d", "dtauy3d", "dusfcg", "dvsfcg", "rubldiff", "rvbldiff")), invalid)
    device_id = int(cp.cuda.runtime.getDevice())
    if kernel_artifact is None:
        module = _module(nlev, device_id, dx_p is not None)
    else:
        module = _artifact_module(nlev, device_id, dx_p is not None,
                                  kernel_artifact["source_sha256"],
                                  kernel_artifact["binary_sha256"],
                                  tuple(kernel_artifact["options"]), kernel_artifact["binary"])
    kernel = module.get_function("bl_ysu_gwdo_v842_f32")
    kernel(((ncells+63)//64,), (64,), args)
    # Read off the step path inside a forecast (deferred_checks_v842, U-MPASPHYS-1).
    from .deferred_checks_v842 import defer
    defer(invalid, "YSU GWDO compiled level count differs from its launch", error=RuntimeError)
    return output


def _discard_native_profile_inputs(prepared):
    """module_bl_gwdo.F:94-109; these eight profiles are INTENT(IN).

    The CUDA kernel consumes const pointers and writes separate outputs.
    Native storage, including halos and interface levels, stays authoritative.
    Ordinary mapping callers retain their existing behavior.
    """
    from .device_pools_v842 import _PreparedViews, NativeDevicePoolViews
    if isinstance(prepared, _PreparedViews) and isinstance(prepared.owner, NativeDevicePoolViews):
        for name in _NATIVE_PROFILE_INPUTS:
            prepared.owner.buffers.pop(('prepared', name), None)


def driver_gwdo(*, prepared: Mapping[str, Any], mesh: Mapping[str, Any],
                 sfc_input: Mapping[str, Any], tend_physics: MutableMapping[str, Any],
                 diag_physics: MutableMapping[str, Any], dt_pbl: float,
                 config_len_disp: float, constants: Mapping[str, float] | None = None,
                 kernel_artifact: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """MPAS mpas_atmphys_driver_gwdo.F:692-834; complete YSU driver slice."""
    inputs = gwdo_from_MPAS(prepared, mesh, sfc_input, tend_physics)
    result = run_bl_ysu_gwdo_cuda_v842(dt_seconds=dt_pbl, config_len_disp=config_len_disp,
                                     constants=constants, kernel_artifact=kernel_artifact, **inputs)
    gwdo_to_MPAS(result, diag_physics, tend_physics)
    _discard_native_profile_inputs(prepared)
    return result


def register_backend(registry: Any, *, arguments: Any) -> None:
    """MPAS mpas_atmphys_driver.F:333-341; mpas_atmphys_driver_gwdo.F:761-781.

    Bind the published f4 SlotCall registry contract. The forecast adapter's
    argument provider supplies the resident named pools and resolved length
    consumed by driver_gwdo, including native dt_pbl; it owns framework
    handles, scheme intervals and solve/halo views.
    Only this scheme's hex-cuda row is registered; suite selection is f4's.
    """
    if not callable(arguments):
        raise TypeError("YSU GWDO registration needs the forecast argument provider")

    def launch(call: Any) -> dict[str, Any]:
        """MPAS mpas_atmphys_driver_gwdo.F:761-781; manager owns dt_pbl."""
        return driver_gwdo(**arguments(call))

    registry.register("gwdo", "bl_ysu_gwdo", BACKEND, launch)


def bind_forecast(provider: Any, call: Any) -> dict[str, Any]:
    """MPAS mpas_atmphys_driver_gwdo.F:285-362,520-550,761-781.

    Bind the foundation device-pool transaction. Common preparation owns
    hydrostatic profiles and the core's resolved config_len_disp; native
    manager storage supplies dt_pbl, independently of the dynamical dt.
    The transaction retains halo words when these solve-cell views commit.
    """
    return dict(prepared=provider["prepared"], mesh=provider["mesh"],
                sfc_input=provider["sfc_input"],
                tend_physics=provider["tend_physics"],
                diag_physics=provider["diag_physics"],
                dt_pbl=provider["scalar_state"]["dt_pbl"],
                config_len_disp=provider["configs"]["config_len_disp"],
                constants={name: provider["constants"][native_name]
                           for name, native_name in (("gravity", "gravity"),
                                                     ("cp", "cp"), ("rd", "r_d"),
                                                     ("fv", "ep_1"), ("pi", "pii"))})


def forecast_callback(call: Any) -> dict[str, Any]:
    """MPAS mpas_atmphys_driver_gwdo.F:692-834,761-781; whole YSU slot."""
    return driver_gwdo(**call.pools)
