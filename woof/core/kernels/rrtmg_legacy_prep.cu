#ifndef RP_CLOUD_FORM
#define RP_CLOUD_FORM 0
#endif
// Device prep twin of rrtmg_legacy_prep.py, WRF v4.6.1 option 4.
// Third-party algorithm: Copyright (c) 2020 Atmospheric and Environmental
// Research. BSD-3-Clause; see licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt.
// Compiled by compile_using_nvrtc with --ftz=false, never RawModule.
// Every FP32 arithmetic step rounds once, without contraction. Inline PTX
// has no .ftz modifier, including correctly rounded division: cloud-path
// numerators CAN be subnormal, so the SW normal-operand invariant does not
// apply here. div.rn.f32 handles gradual underflow without double rounding.
// Comparisons depend on the unflushed compile route, tested before use.
// NumPy 2.5.3 AVX2 min/max: propagate NaNs; ties return the SECOND operand,
// including opposite signed zeros. The live test checks these semantics.
// One thread per column. All scratch profiles are global, no local arrays.

__device__ __forceinline__ float rp_add(float a, float b) {
    float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float rp_sub(float a, float b) {
    float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float rp_mul(float a, float b) {
    float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float rp_div(float a, float b) {
    float r; asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
#define RP_AD(a,b) rp_add((a),(b))
#define RP_SU(a,b) rp_sub((a),(b))
#define RP_MU(a,b) rp_mul((a),(b))
#define RP_DV(a,b) rp_div((a),(b))
#ifdef RP_RADIUS_MICRONS
#define RP_RADIUS(x) (x)
#else
#define RP_RADIUS(x) RP_MU((x), 1.e6f)
#endif
__device__ __forceinline__ float rp_max(float a, float b) {
    return isnan(a) ? a : (isnan(b) ? b : (a > b ? a : b));
}
__device__ __forceinline__ float rp_min(float a, float b) {
    return isnan(a) ? a : (isnan(b) ? b : (a < b ? a : b));
}
__device__ float rp_clip(float x) { return rp_min(1.0f, rp_max(0.0f, x)); }
__device__ float rp_rei(float t, const float* retab, int limit) {
    int j = (int)RP_SU(t, 179.0f);
    j = j < 1 ? 1 : (j > limit ? limit : j);
    float corr = RP_SU(t, (float)((int)t));
    return RP_AD(RP_MU(retab[j-1], RP_SU(1.0f, corr)), RP_MU(retab[j], corr));
}
__device__ float rp_varint(float p, const float* pp, const float* tp) {
    // searchsorted(-PPROF, -p, side=right), clamped to [1,59].
    int j = 1;
    while (j < 59 && !(pp[j] < p)) ++j;
    float w = RP_DV(RP_SU(p, pp[j-1]), RP_SU(pp[j], pp[j-1]));
    float v = RP_AD(RP_MU(w, RP_SU(tp[j], tp[j-1])), tp[j-1]);
    return pp[59] < p ? v : tp[59];
}
__device__ float rp_pos(float x) { return x > 0.0f ? x : 0.0f; }
__device__ float rp_o3(float p, float p1, const float* ow, const float* ph) {
    float acc = 0.0f;
    for (int j=0; j<31; ++j) {
        float pb1 = rp_pos(RP_SU(p, ph[j]));
        float pb2 = rp_pos(RP_SU(p, ph[j+1]));
        float pt1 = rp_pos(RP_SU(p1, ph[j]));
        float pt2 = rp_pos(RP_SU(p1, ph[j+1]));
        float w = RP_AD(RP_SU(RP_SU(pb2, pb1), pt2), pt1);
        acc = RP_AD(acc, RP_MU(w, ow[j]));
    }
    return RP_DV(acc, RP_SU(p, p1));
}
__device__ float rp_path(float q, float pd, float g) {
    return RP_MU(RP_DV(RP_MU(RP_MU(q, pd), 100.0f), g), 1000.0f);
}

extern "C" __global__ void rp_probe(const float* x, float* y) {
    if (threadIdx.x || blockIdx.x) return;
    y[0] = RP_MU(x[0], x[1]);
    y[1] = x[2] > 0.0f ? 1.0f : 0.0f;
    y[2] = RP_DV(x[2], x[3]);
    y[3] = rp_max(x[4], x[5]);
    y[4] = rp_min(x[4], x[5]);
}

extern "C" __global__ void rp_prep(
    int ncol, int nk, int nl, int sw, int icloud, int warm,
    int fqc, int fqr, int fqi, int fqs, int hc, int hi, int hs,
    int inflg, int iceflg, int o3input, int ferrier, int cam, float g,
    const float* p, const float* pw, const float* t, const float* tw,
    const float* dz, const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* cf, const float* oz,
    const float* rc, const float* ri, const float* rs,
    const float* cam_lradius, const float* cam_iradius,
    const float* land, const float* ice, const float* snow,
    const float* retab, const float* pp, const float* tp,
    const float* ow, const float* ph,
    float* pl, float* tl, float* pa, float* ta, float* hg,
    float* hv, float* ov, float* o31, float* pd,
    float* cl, float* cw, float* iw, float* snw,
    float* rel, float* rei, float* res) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    int a = c*nk, b = c*nl, e = c*(nl+1), w = c*(nk+1);
    float dzsum = 0.0f;
    for (int k=0; k<=nk; ++k) {
        pl[e+k] = RP_DV(pw[w+k], 100.0f);
        tl[e+k] = tw[w+k];
    }
    for (int k=0; k<nk; ++k) {
        pa[b+k] = RP_DV(p[a+k], 100.0f);
        ta[b+k] = t[a+k];
        pd[a+k] = RP_SU(pl[e+k], pl[e+k+1]);
        float v = rp_max(0.0f, qv[a+k]);
        v = rp_max(v, 1.e-12f);
        hv[b+k] = RP_MU(v, 1.607793f);
        hg[b+k] = RP_AD(dzsum, RP_MU(0.5f, dz[a+k]));
        dzsum = RP_AD(dzsum, dz[a+k]);
    }
    float lastdz = dz[a+nk-1];
    if (sw) {
        pa[b+nk] = RP_MU(0.5f, pl[e+nk]);
        ta[b+nk] = RP_AD(tl[e+nk], 0.0f);
        pl[e+nk+1] = 1.e-5f;
        tl[e+nk+1] = RP_AD(tl[e+nk], 0.0f);
        hv[b+nk] = hv[b+nk-1];
        hg[b+nk] = RP_AD(dzsum, RP_MU(0.5f, lastdz));
    } else {
        // The final interface is overwritten with zero below. Only the
        // preceding nl-nk-1 decrements must stay positive. Ordinary WRF
        // tops retain their prescribed 4-hPa spacing.
        float dp = 4.0f;
        if (pl[e+nk] <= RP_MU(4.0f, (float)(nl-nk-1)))
            dp = rp_min(4.0f, RP_DV(pl[e+nk], (float)(nl-nk)));
        for (int k=nk; k<nl; ++k) {
            pl[e+k+1] = RP_SU(pl[e+k], dp);
            pa[b+k] = RP_MU(0.5f, RP_AD(pl[e+k], pl[e+k+1]));
            hv[b+k] = hv[b+nk-1];
            hg[b+k] = RP_AD(dzsum, RP_MU(0.5f, lastdz));
            dzsum = RP_AD(dzsum, lastdz);
        }
        pl[e+nl] = 0.0f;
        pa[b+nl-1] = RP_MU(0.5f, RP_AD(pl[e+nl-1], pl[e+nl]));
        float anchor = RP_SU(tl[e+nk-1], rp_varint(pl[e+nk-1], pp, tp));
        for (int k=nk; k<=nl; ++k)
            tl[e+k] = RP_AD(rp_varint(pl[e+k], pp, tp), anchor);
        for (int k=nk-1; k<nl; ++k)
            ta[b+k] = RP_MU(0.5f, RP_AD(tl[e+k+1], tl[e+k]));
    }
    float obase = 0.0f;
    if (o3input == 2)
        obase = RP_SU(oz[a+nk-1], RP_MU(rp_o3(pl[e+nk-1], pl[e+nk], ow, ph), 0.603461f));
    for (int k=0; k<nl; ++k) {
        float clim = RP_MU(rp_o3(pl[e+k], pl[e+k+1], ow, ph), 0.603461f);
        float v = RP_AD(obase, clim);
        ov[b+k] = o3input == 2 ? (k<nk ? oz[a+k] : (v<=0.0f ? clim : v)) : clim;
        if (k<nk) o31[a+k] = ov[b+k];
        cl[b+k] = 0.0f; cw[b+k] = 0.0f; iw[b+k] = 0.0f; snw[b+k] = 0.0f;
        rel[b+k] = 10.0f; rei[b+k] = 10.0f; res[b+k] = 10.0f;
        if (k>=nk) continue;
        int j = a+k;
        float qcl=0.0f, qrain=0.0f, qice=0.0f, qsnow=0.0f;
        if (icloud) {
            cl[b+k] = cf[j];
            if (fqc) qcl = rp_max(0.0f, qc[j]);
            if (fqr) qrain = rp_max(0.0f, qr[j]);
            if (!fqi && !warm && t[j]<273.15f) {
                qice=qcl; qsnow=qrain; qcl=0.0f; qrain=0.0f;
            }
            if (fqi) qice = rp_max(0.0f, qi[j]);
            if (fqs) qsnow = rp_max(0.0f, qs[j]);
            // ETAMPNEW carries total ice in QS, without a QI species.
            // WRF 4.7.1 module_ra_rrtmg_lw/sw split it 10/90 for optics.
            if (ferrier == 2) {
                qice = rp_max(0.0f, RP_MU(0.1f, qs[j]));
                qsnow = RP_MU(0.9f, qs[j]);
                qcl = rp_max(0.0f, qc[j]);
            }
        }
        // WRF FER_MP_HIRES / FER_MP_HIRES_ADVECT remap is outside ICLOUD
        // (module_ra_rrtmg_lw.F:12135, module_ra_rrtmg_sw.F:10736).
        if (ferrier == 1) {
            qice = rp_max(0.0f, qi[j]);
            qsnow = 0.0f;
            qcl = rp_max(0.0f, qc[j]);
        }
        float rec=0.0f, ric=0.0f, rsc=0.0f;
        if (icloud) {
            rec = hc ? rp_max(2.5f, RP_RADIUS(rc[j])) : 5.0f;
            float xdiff = RP_SU(land[c], 1.5f);
            if (hc && xdiff>0.0f && rec<=2.5f && cf[j]>0.0f) rec=(sw && RP_CLOUD_FORM) ? 9.6f : 10.5f;
            if (hc && xdiff<0.0f && rec<=2.5f && cf[j]>0.0f) rec=(sw && RP_CLOUD_FORM) ? 5.4f : 7.5f;
            ric = hi ? rp_max(5.0f, RP_RADIUS(ri[j])) : 10.0f;
            if (hi && ric<=5.0f && cf[j]>0.0f)
                ric = rp_max(rp_rei(t[j], retab, 75), 5.0f);
            rsc = hs ? rp_max(10.0f, RP_RADIUS(rs[j])) : 10.0f;
            if (!hs && hi && hc) {
                rsc = rp_max(10.0f, RP_RADIUS(ri[j]));
                qsnow=qi[j]; qice=0.0f; ric=10.0f;
            }
        }
        float den = rp_max(0.01f, cl[b+k]);
        cw[b+k] = RP_DV(rp_path(qcl, pd[j], g), den);
        float iq = iceflg>=4 ? qice : RP_AD(qice, qsnow);
        iw[b+k] = RP_DV(rp_path(iq, pd[j], g), den);
        if (iceflg==5) {
            float smf=RP_CLOUD_FORM ? 1.0f : 0.99f;
            if (rsc>130.0f) {
                float q=RP_DV(130.0f, rsc);
                smf=rp_min(smf, RP_MU(q,q)); rsc=130.0f;
            }
            snw[b+k] = RP_DV(rp_path(RP_MU(qsnow,smf), pd[j], g), den);
        }
        float rliq = RP_AD(8.0f, RP_MU(6.0f, rp_clip(RP_MU(RP_SU(273.16f, ta[b+k]), 0.05f))));
        float wt = rp_clip(RP_MU(RP_MU(0.001f, snow[c]), 10.0f));
        rliq = RP_AD(rliq, RP_MU(RP_SU(14.0f,rliq), wt));
        wt = rp_clip(RP_SU(1.0f, RP_SU(2.0f,land[c])));
        rliq = RP_AD(rliq, RP_MU(RP_SU(14.0f,rliq), wt));
        rliq = RP_AD(rliq, RP_MU(RP_SU(14.0f,rliq), rp_clip(ice[c])));
        float rice = rp_rei(ta[b+k], retab, 94);
        if (inflg>=3) rliq=rec;
        if (iceflg>=4) rice=ric;
        if (iceflg==3) rice=rp_min(140.0f, RP_MU(rice,1.0315f));
        // WRF module_ra_rrtmg_lw.F:12607-12625 and sw.F:11156-11175.
        // CAM overrides the fallback after its Fu diameter conversion.
        if (cam) {
            rice = (qice > 1.e-20f || qsnow > 1.e-20f) ? cam_iradius[j] : 25.0f;
            rice = rp_max(5.0f, rp_min(140.0f, rice));
            rliq = qcl > 1.e-20f ? cam_lradius[j] : 10.0f;
            rliq = rp_max(2.5f, rp_min(60.0f, rliq));
        }
        rel[b+k]=rliq; rei[b+k]=rice;
        if (inflg==5) res[b+k]=rsc;
    }
}

extern "C" __global__ void rp_scon(int n, const float* sol, const float* obsc, float* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) out[i]=RP_MU(sol[i], RP_SU(1.0f,obsc[i]));
}

extern "C" __global__ void rp_day(int n, const float* coszen, unsigned char* night) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) night[i]=coszen[i]<=0.0f;
}

// WRF 4.6.1 public-domain module_radiation_driver.F:3676-3749.
// QI alone takes the zero arm. No clipping before the float32 sum.
extern "C" __global__ void rp_cldfra2(
    int n, int fqc, int fqi, const float* qc, const float* qi, float* cf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float condensate;
    if (fqi && fqc) condensate = RP_AD(qc[i], qi[i]);
    else if (fqc) condensate = qc[i];
    else { cf[i] = 0.0f; return; }
    cf[i] = condensate > 1.0e-6f ? 1.0f : 0.0f;
}
