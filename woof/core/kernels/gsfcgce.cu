// gpuwm/core/kernels/gsfcgce.cu
//
// Goddard GCE three-ice bulk microphysics, WRF mp_physics = 97.
// Transcription authority: WRF v4.6.1 phys/module_mp_gsfcgce.F (pinned
// d66e442, sha256 43e841f7...): gsfcgce (:90-349), fall_flux (:352-845),
// saticel_s (:1238-2908) and refl10cm_gsfc (:2942-3115).  consat_s
// (:957-1236) and fall_flux's gamma/prefactor block are evaluated once on
// the host into the float32 table this kernel reads
// (gpuwm/core/gsfcgce_constants.py, slot order = the enum below).
//
// One CUDA thread owns one (j, i) column.  gpuwm arrays are (k, j, i), x
// fastest; Fortran level k = 1..kte is index k - 1 here.  Every expression
// keeps WRF's float32 expressions. Strict proof builds disable multiply-add
// contraction; normal forecasts use the engine's fast contraction policy.
// In both modes
// every quotient is __fdiv_rn so no build may turn a constant divisor into a
// reciprocal multiply.  REAL powers call WOOF's own float32 gfk_pow, EXP
// calls gfk_exp (glibc_flt32.cuh); integer powers are the multiply chains
// gfortran -O2 emits for them (x**2 = x*x, x**3 = x*(x*x),
// x**5 = (x*x)*(x*(x*x))).
//
// Two native undefined reads are repaired (gsfcgce_constants.py
// DECLARED_DIVERGENCES): the warm-only branch (gsfcgce_2ice = 3)
// computes the transport coefficients
// its rain evaporation reads, and the ice+snow branch (gsfcgce_2ice = 1)
// hands the reflectivity a zero graupel column.

#define GQ(a, k) (a)[I3((k), j, i, ny, nx)]
#define FDV(a, b) __fdiv_rn((a), (b))
#define FSQ(a) __fsqrt_rn(a)

enum GceSlot {
    S_FF_PI, S_FF_CONSTA, S_FF_CONSTC, S_FF_GAMBP4, S_FF_GAMDP4,
    S_FF_GAM4PT5, S_FF_GAM4BBAR,
    S_AL, S_CP, S_C358, S_C610, S_C149, S_C879, S_C172, S_C409, S_C76,
    S_C218, S_C580, S_C141, S_ALV, S_ALF, S_ALS, S_T0, S_T00, S_AVC, S_AFC,
    S_ASC,
    S_AG, S_BG, S_AS, S_BS, S_AW, S_BW, S_BGH, S_BGQ, S_BSH, S_BSQ, S_BWH,
    S_BWQ, S_TNW, S_TNS, S_TNG, S_ROQS, S_ROQG, S_ROQR, S_ZRC, S_ZGC, S_ZSC,
    S_VRC, S_VGC, S_VSC,
    S_RN1, S_BND1, S_RN2, S_BND2, S_RN3, S_RN4, S_RN5, S_RN6, S_RN7, S_RN8,
    S_RN9, S_RN10, S_RN101, S_RN10A, S_RN11, S_RN11A, S_RN12, S_RN14,
    S_RN15, S_RN15A, S_RN16, S_RN17, S_RN17A, S_RN17B, S_RN17C, S_RN18,
    S_RN18A, S_RN19, S_RN19A, S_RN19B, S_RN20, S_RN20A, S_RN20B, S_BND3,
    S_RN21, S_RN22, S_RN23, S_RN23A, S_RN23B, S_RN25, S_RN30A, S_RN30B,
    S_RN30C, S_RN31, S_BETA, S_RN32, S_RN10B, S_RN10C, S_RNN191, S_RNN192,
    S_RN30, S_RNN30A, S_RN33, S_RN331, S_RN332,
    S_NSCALAR
};
#define T_RN12A (S_NSCALAR + 0)
#define T_RN12B (S_NSCALAR + 31)
#define T_RN13  (S_NSCALAR + 62)
#define T_RN25A (S_NSCALAR + 93)
#define T_AA1   (S_NSCALAR + 124)
#define T_AA2   (S_NSCALAR + 155)

// gfortran -O2 integer powers (constant exponent, multiply chains).
__device__ __forceinline__ float gce_p2(float x) { return x * x; }
__device__ __forceinline__ float gce_p3(float x) { return x * (x * x); }
__device__ __forceinline__ float gce_p5(float x) {
    float x2 = x * x;
    return x2 * (x * x2);
}
// gfortran MAX/MIN of REALs (MAX_EXPR / MIN_EXPR, maxss/minss on x86).
__device__ __forceinline__ float gce_max(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float gce_min(float a, float b) { return a < b ? a : b; }

// ---- fall_flux: one species, WRF's sub-stepped upwind flux ---------------
// species 0 rain, 1 snow, 2 graupel/hail, 3 cloud ice.  The terminal
// velocity is stored during the scan and reused by the flux pass, as in
// WRF's vtr/vts/vtg/vti arrays. One device workspace is reused across the
// four species. No contents survive a call or enter a checkpoint.
struct GceFall {
    float pi, rhowater, rhosnow, grav, consta, constc, gambp4, gamdp4,
          gam4pt5, gam4bbar;
    int ihail;
};

__device__ __forceinline__ float gce_vt(const GceFall& f, int species,
                                        float q, float rho, float pres)
{
    const float rhoe_s = 1.29f;
    if (species == 0) {
        float sqrhoz = FSQ(FDV(rhoe_s, rho));
        float tmp1 = FSQ(FDV(FDV(f.pi * f.rhowater * 8.0e6f, rho), q));
        tmp1 = FSQ(tmp1);
        float v = FDV(f.consta * f.gambp4 * sqrhoz, gfk_pow(tmp1, 0.8f));
        return FDV(v, 6.0f);
    }
    if (species == 1) {
        float sqrhoz = FSQ(FDV(rhoe_s, rho));
        float tmp1 = FSQ(FDV(FDV(f.pi * f.rhosnow * 1.6e7f, rho), q));
        tmp1 = FSQ(tmp1);
        float v = FDV(f.constc * f.gamdp4 * sqrhoz, gfk_pow(tmp1, 0.11f));
        return FDV(v, 6.0f);
    }
    if (species == 2) {
        if (f.ihail == 1) {
            float tmp1 = FSQ(FDV(FDV(f.pi * 917.0f * 2.0e5f, rho), q));
            tmp1 = FSQ(tmp1);
            float term0 = FSQ(FDV(FDV(FDV(4.0f * f.grav * 917.0f, 3.0f), rho),
                                  0.6f));
            float v = f.gam4pt5 * term0 * FSQ(FDV(1.0f, tmp1));
            return FDV(v, 6.0f);
        }
        float tmp1 = FSQ(FDV(FDV(f.pi * 400.0f * 4.0e6f, rho), q));
        tmp1 = FSQ(tmp1);
        tmp1 = gfk_pow(tmp1, 0.37f);
        tmp1 = FDV(1.0f, tmp1);
        float term0 = FDV(19.3f * f.gam4bbar, 6.0f);
        return term0 * tmp1 * gfk_pow(FDV(1.0e5f, pres), 0.4f);
    }
    return 3.29f * gfk_pow(rho * q, 0.16f);
}

__device__ float gce_sediment(const GceFall& f, int species, float* q,
                              const float* rho, const float* pres,
                              const float* zz, const float* dzw, float topo,
                              float dtb, float* velocity, int nz, int ny,
                              int nx, int j, int i)
{
    float ppt = 0.0f;
    float t_del_tv = 0.0f;
    float del_tv = dtb;
    bool notlast = true;
    while (notlast) {
        int min_q = nz - 1;
        int max_q = -1;
        for (int k = 0; k < nz - 1; ++k) {
            float qk = GQ(q, k);
            if (qk > 1.0e-8f) {
                min_q = min(min_q, k);
                max_q = max(max_q, k);
                float vt = gce_vt(f, species, qk, GQ(rho, k), GQ(pres, k));
                GQ(velocity, k) = vt;
                float dz = (k == 0) ? GQ(zz, k) - topo : GQ(zz, k) - GQ(zz, k - 1);
                del_tv = gce_min(del_tv, FDV(0.9f * dz, vt));
            } else {
                GQ(velocity, k) = 0.0f;
            }
        }
        if (max_q >= min_q) {
            t_del_tv = t_del_tv + del_tv;
            if (t_del_tv >= dtb) {
                notlast = false;
                del_tv = dtb + del_tv - t_del_tv;
            }
            float fluxin = 0.0f;
            for (int k = max_q; k >= min_q; --k) {
                float qk = GQ(q, k);
                float rk = GQ(rho, k);
                float vt = GQ(velocity, k);
                float fluxout = rk * vt * qk;
                float flux = FDV(FDV(fluxin - fluxout, rk), GQ(dzw, k));
                qk = qk + del_tv * flux;
                qk = gce_max(0.0f, qk);
                GQ(q, k) = qk;
                fluxin = fluxout;
            }
            if (min_q == 0) {
                ppt = ppt + fluxin * del_tv;
            } else {
                int km = min_q - 1;
                GQ(q, km) = GQ(q, km) + FDV(FDV(del_tv * fluxin, GQ(rho, km)),
                                            GQ(dzw, km));
            }
        } else {
            notlast = false;
        }
    }
    return ppt;
}

// ---- saticel_s: one level of one column ------------------------------------

extern "C" __global__ void gsfcgce_column(
    real* __restrict__ th, real* __restrict__ qv, real* __restrict__ qc,
    real* __restrict__ qr, real* __restrict__ qi, real* __restrict__ qs,
    real* __restrict__ qg,
    const real* __restrict__ rho_mks, const real* __restrict__ pii,
    const real* __restrict__ p_mks, const real* __restrict__ zz,
    const real* __restrict__ dz8w, const real* __restrict__ ht,
    real* __restrict__ rainnc, real* __restrict__ rainncv,
    real* __restrict__ snownc, real* __restrict__ snowncv,
    real* __restrict__ graupelnc, real* __restrict__ graupelncv,
    real* __restrict__ sr, real* __restrict__ velocity,
    const float* __restrict__ C,
    const float dt, const float grav, const float rhowater,
    const float rhosnow, const int ihail, const int ice2,
    const int nz, const int ny, const int nx)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    const int j = col / nx;
    const int i = col - j * nx;

    // ================= fall_flux (module_mp_gsfcgce.F:449-829) ============
    {
        GceFall f;
        f.pi = C[S_FF_PI];
        f.rhowater = rhowater;
        f.rhosnow = rhosnow;
        f.grav = grav;
        f.consta = C[S_FF_CONSTA];
        f.constc = C[S_FF_CONSTC];
        f.gambp4 = C[S_FF_GAMBP4];
        f.gamdp4 = C[S_FF_GAMDP4];
        f.gam4pt5 = C[S_FF_GAM4PT5];
        f.gam4bbar = C[S_FF_GAM4BBAR];
        f.ihail = ihail;
        const float topo = ht[I3(0, j, i, 1, nx)];
        float pptrain = gce_sediment(f, 0, qr, rho_mks, p_mks, zz, dz8w,
                                     topo, dt, velocity, nz, ny, nx, j, i);
        float pptsnow = gce_sediment(f, 1, qs, rho_mks, p_mks, zz, dz8w,
                                     topo, dt, velocity, nz, ny, nx, j, i);
        float pptgraul = 0.0f;
        // WRF loads and sediments graupel only for ice2 == 0 (:494, :647).
        // Retain its deterministic ice2 == 2 behavior, including zero fallout.
        if (ice2 == 0) {
            pptgraul = gce_sediment(f, 2, qg, rho_mks, p_mks, zz, dz8w,
                                    topo, dt, velocity, nz, ny, nx, j, i);
        }
        float pptice = gce_sediment(f, 3, qi, rho_mks, p_mks, zz, dz8w,
                                    topo, dt, velocity, nz, ny, nx, j, i);
        const size_t s = I3(0, j, i, 1, nx);
        snowncv[s] = pptsnow;
        snownc[s] = snownc[s] + pptsnow;
        graupelncv[s] = pptgraul;
        graupelnc[s] = graupelnc[s] + pptgraul;
        float vsum = pptrain + pptsnow + pptgraul + pptice;
        rainncv[s] = vsum;
        rainnc[s] = rainnc[s] + pptrain + pptsnow + pptgraul + pptice;
        sr[s] = 0.0f;
        if (vsum > 0.0f) sr[s] = FDV(pptsnow + pptgraul + pptice, vsum);
    }

    // ================= saticel_s (module_mp_gsfcgce.F:1531-2861) ==========
    const float d2t = dt;
    const float cmin = 1.e-19f, cmin1 = 1.e-20f, cmin2 = 1.e-12f;
    const float cn0 = 1.e-6f;
    const float tb0 = 0.0f, qb0 = 0.0f;
    float r2ig = 1.0f, r2is = 1.0f;
    if (ice2 == 1) { r2ig = 0.0f; r2is = 1.0f; }
    if (ice2 == 2) { r2ig = 1.0f; r2is = 0.0f; }
    const bool iwarm = (ice2 == 3);

    const float t0 = C[S_T0], t00 = C[S_T00];
    const float bw = C[S_BW], bs = C[S_BS], bg = C[S_BG];
    const float bw3 = bw + 3.0f, bs3 = bs + 3.0f, bg3 = bg + 3.0f;
    const float bsh5 = 2.5f + C[S_BSH], bgh5 = 2.5f + C[S_BGH],
                bwh5 = 2.5f + C[S_BWH];
    const float bw6 = bw + 6.0f;
    const float beta = C[S_BETA];
    const float betah = 0.5f * beta;
    const float r10t = C[S_RN10] * d2t;
    const float r11at = C[S_RN11A] * d2t;
    const float r19bt = C[S_RN19B] * d2t;
    const float r20t = -C[S_RN20] * d2t;
    const float r23t = -C[S_RN23] * d2t;
    const float r25a = C[S_RN25];
    const float ami100 = 1.51e-7f, ami40 = 2.41e-8f;
    const float c358 = C[S_C358], c172 = C[S_C172], c409 = C[S_C409];
    const float c76 = C[S_C76], c218 = C[S_C218], c580 = C[S_C580];
    const float c610 = C[S_C610];
    const float zrc = C[S_ZRC], zsc = C[S_ZSC], zgc = C[S_ZGC];
    const float bwq = C[S_BWQ], bsq = C[S_BSQ], bgq = C[S_BGQ];
    const float alf = C[S_ALF], rn17b = C[S_RN17B], rn17c = C[S_RN17C];

    // fv(k) = sqrt(rho(2)/rho(k)) in cgs (:1557-1563).
    const float rho2 = GQ(rho_mks, 1) * 0.001f;

    for (int k = 0; k < nz; ++k) {
        const float rhok = GQ(rho_mks, k) * 0.001f;
        const float p0k = GQ(p_mks, k) * 10.0f;
        const float pik = GQ(pii, k);
        const float fvk = FSQ(FDV(rho2, rhok));

        const float rp0 = FDV(3.799052e3f, p0k);
        const float pi0 = pik;
        const float pir = FDV(1.0f, pik);
        const float pr0 = FDV(1.0f, p0k);
        const float r00 = rhok;
        const float rr0 = FDV(1.0f, rhok);
        const float rrs = FSQ(rr0);
        const float rrq = FSQ(rrs);
        const float fv0 = fvk;
        const float fvs = FSQ(fvk);
        const float zrr = 1.e5f * zrc * rrq;
        const float zsr = 1.e5f * zsc * rrq;
        const float zgr = 1.e5f * zgc * rrq;
        const float cp409 = c409 * pi0;
        const float cv409 = c409 * C[S_AVC];
        const float cp580 = c580 * pi0;
        const float cs580 = c580 * C[S_ASC];
        const float alvr = r00 * C[S_ALV];
        const float afcp = C[S_AFC] * pir;
        const float avcp = C[S_AVC] * pir;
        const float ascp = C[S_ASC] * pir;
        const float vrcf = C[S_VRC] * fv0;
        const float vscf = C[S_VSC] * fv0;
        const float vgcf = C[S_VGC] * fv0;
        const float vgcr = C[S_VGC] * rrs;
        const float dwvp = C[S_C879] * pr0;
        const float r3f = C[S_RN3] * fv0;
        const float r4f = C[S_RN4] * fv0;
        const float r5f = C[S_RN5] * fv0;
        const float r6f = C[S_RN6] * fv0;
        const float r7r = C[S_RN7] * rr0;
        const float r8r = C[S_RN8] * rr0;
        const float r9r = C[S_RN9] * rr0;
        const float r101f = C[S_RN101] * fvs;
        const float r10ar = C[S_RN10A] * r00;
        const float r11rt = C[S_RN11] * rr0 * d2t;
        const float r12r = C[S_RN12] * r00;
        const float r14r = C[S_RN14] * rrs;
        const float r14f = C[S_RN14] * fv0;
        const float r15r = C[S_RN15] * rrs;
        const float r15ar = C[S_RN15A] * rrs;
        const float r15af = C[S_RN15A] * fv0;
        const float r16r = C[S_RN16] * rr0;
        const float r17r = C[S_RN17] * rr0;
        const float r17aq = C[S_RN17A] * rrq;
        const float r17as = C[S_RN17A] * fvs;
        const float r18r = C[S_RN18] * rr0;
        const float r19rt = C[S_RN19] * rr0 * d2t;
        const float r19aq = C[S_RN19A] * rrq;
        const float r19as = C[S_RN19A] * fvs;
        const float r20bq = C[S_RN20B] * rrq;
        const float r20bs = C[S_RN20B] * fvs;
        const float r22f = C[S_RN22] * fv0;
        const float r23af = C[S_RN23A] * fvs;
        const float r23br = C[S_RN23B] * r00;
        const float r25rt = C[S_RN25] * rr0 * d2t;
        const float r32rt = C[S_RN32] * d2t * rrs;

        float pt = GQ(th, k);
        float qvk = GQ(qv, k);
        float qck = GQ(qc, k);
        float qrk = GQ(qr, k);
        float qik = GQ(qi, k);
        float qsk = GQ(qs, k);
        float qgk = GQ(qg, k);
        if (qck <= cmin1) qck = 0.0f;
        if (qrk <= cmin1) qrk = 0.0f;
        if (qik <= cmin1) qik = 0.0f;
        if (qsk <= cmin1) qsk = 0.0f;
        if (qgk <= cmin1) qgk = 0.0f;
        float tair = (pt + tb0) * pi0;
        float tairc = tair - t0;
        float zr = zrr, zs = zsr, zg = zgr;
        float vr = 0.0f, vs = 0.0f, vg = 0.0f;
        float y1, y2, y3, y4, y5, dd, dd1, dm, rsub1, rtair, qsw, qsi, ssi,
              ssw, esi, cnd, dep, ern;
        float scv, tca, dwv;

        if (iwarm) {
            // ---------------- warm rain only (:1772-1868) ----------------
            qik = 0.0f; qsk = 0.0f; qgk = 0.0f;
            if (qrk > cmin1) {
                dd = r00 * qrk;
                y1 = gfk_pow(dd, 0.25f);
                zr = FDV(zrc, y1);
                vr = gce_max(vrcf * gfk_pow(dd, bwq), 0.0f);
            }
            // DECLARED DIVERGENCE: WRF's warm branch reads scv/tca/dwv
            // uninitialised (:1859-1860); these are the ice branch's
            // statements (:1912-1915) at the same point.
            y1 = FDV(C[S_C149] * gfk_pow(tair, 1.5f), tair + 120.0f);
            dwv = dwvp * gfk_pow(tair, 1.81f);
            tca = C[S_C141] * y1;
            scv = FDV(1.0f, gfk_pow(rr0 * y1, .1666667f)
                            * gfk_pow(dwv, .3333333f));
            float pracw = 0.0f, praut = 0.0f;
            pracw = FDV(r22f * qck, gfk_pow(zr, bw3));
            y1 = qck - C[S_BND3];
            if (y1 > 0.0f) {
                praut = FDV(r00 * y1 * y1, 1.2e-4f + FDV(C[S_RN21], y1));
            }
            y1 = FDV(qck, d2t);
            praut = gce_min(y1, praut);
            pracw = gce_min(y1, pracw);
            y1 = (praut + pracw) * d2t;
            if (qck < y1 && y1 >= cmin2) {
                y2 = FDV(qck, y1 + cmin2);
                praut = praut * y2;
                pracw = pracw * y2;
                qck = 0.0f;
            } else {
                qck = qck - y1;
            }
            float pr = (praut + pracw) * d2t;
            qrk = qrk + pr;
            // saturation adjustment (:1834-1845)
            cnd = 0.0f;
            tair = (pt + tb0) * pi0;
            y1 = FDV(1.0f, tair - c358);
            qsw = rp0 * gfk_exp(c172 - c409 * y1);
            dd = cp409 * y1 * y1;
            dm = qvk + qb0 - qsw;
            cnd = FDV(dm, 1.0f + avcp * dd * qsw);
            cnd = gce_max(-qck, cnd);
            pt = pt + avcp * cnd;
            qvk = qvk - cnd;
            qck = qck + cnd;
            // rain evaporation (:1849-1868)
            ern = 0.0f;
            if (qrk > 0.0f) {
                tair = (pt + tb0) * pi0;
                rtair = FDV(1.0f, tair - c358);
                qsw = rp0 * gfk_exp(c172 - c409 * rtair);
                ssw = FDV(qvk + qb0, qsw) - 1.0f;
                dm = qvk + qb0 - qsw;
                rsub1 = cv409 * qsw * rtair * rtair;
                dd1 = gce_max(FDV(-dm, 1.0f + rsub1), 0.0f);
                y1 = FDV(0.78f, gce_p2(zr))
                   + FDV(r23af * scv, gfk_pow(zr, bwh5));
                y2 = FDV(r23br, tca * gce_p2(tair)) + FDV(1.0f, dwv * qsw);
                ern = FDV(r23t * ssw * y1, y2);
                ern = gce_min(gce_min(dd1, qrk), gce_max(ern, 0.0f));
                pt = pt - avcp * ern;
                qvk = qvk + ern;
                qrk = qrk - ern;
            }
        } else {
            // ---------------- ice and warm rain (:1877-2745) -------------
            if (qrk > cmin1) {
                dd = r00 * qrk;
                y1 = gfk_pow(dd, 0.25f);
                zr = FDV(zrc, y1);
                vr = gce_max(vrcf * gfk_pow(dd, bwq), 0.0f);
            }
            if (qsk > cmin1) {
                dd = r00 * qsk;
                y1 = gfk_pow(dd, 0.25f);
                zs = FDV(zsc, y1);
                vs = gce_max(vscf * gfk_pow(dd, bsq), 0.0f);
            }
            if (qgk > cmin1) {
                dd = r00 * qgk;
                y1 = gfk_pow(dd, 0.25f);
                zg = FDV(zgc, y1);
                if (ihail == 1) vg = gce_max(vgcr * gfk_pow(dd, bgq), 0.0f);
                else            vg = gce_max(vgcf * gfk_pow(dd, bgq), 0.0f);
            }
            if (qrk <= cmin2) vr = 0.0f;
            if (qsk <= cmin2) vs = 0.0f;
            if (qgk <= cmin2) vg = 0.0f;

            y1 = FDV(C[S_C149] * gfk_pow(tair, 1.5f), tair + 120.0f);
            dwv = dwvp * gfk_pow(tair, 1.81f);
            tca = C[S_C141] * y1;
            scv = FDV(1.0f, gfk_pow(rr0 * y1, .1666667f)
                            * gfk_pow(dwv, .3333333f));

            // psaut psaci psacw praci piacr (:1926-1948)
            float psaut = 0.0f, psaci = 0.0f, praci = 0.0f, piacr = 0.0f,
                  psacw = 0.0f, qsacw = 0.0f;
            dd = FDV(1.0f, gfk_pow(zs, bs3));
            if (tair < t0) {
                esi = gfk_exp(.025f * tairc);
                psaut = r2is * gce_max(C[S_RN1] * esi * (qik - C[S_BND1]), 0.0f);
                psaci = r2is * r3f * esi * qik * dd;
                psacw = r2is * 0.5f * r4f * qck * dd;
                praci = FDV(r2is * r5f * qik, gfk_pow(zr, bw3));
                piacr = r2is * r6f * qik * gfk_pow(zr, -bw6);
            } else {
                qsacw = r2is * r4f * qck * dd;
            }
            // praut pracw (:1953-1958)
            float pracw = FDV(r22f * qck, gfk_pow(zr, bw3));
            float praut = 0.0f;
            y1 = qck - C[S_BND3];
            if (y1 > 0.0f) {
                praut = FDV(r00 * y1 * y1, 1.2e-4f + FDV(C[S_RN21], y1));
            }
            // psfw psfi pidep (:1963-2008)
            float psfw = 0.0f, psfi = 0.0f, pidep = 0.0f;
            if (tair < t0 && qik > cmin) {
                y1 = gce_max(gce_min(tairc, -1.0f), -31.0f);
                int it = (int)fabsf(y1);
                y1 = C[T_RN12A + it - 1];
                y2 = C[T_RN12B + it - 1];
                psfw = r2is * gce_max(d2t * y1 * (y2 + r12r * qck) * qik, 0.0f);
                rtair = FDV(1.0f, tair - c76);
                y2 = gfk_exp(c218 - c580 * rtair);
                qsi = rp0 * y2;
                esi = c610 * y2;
                ssi = FDV(qvk + qb0, qsi) - 1.0f;
                float r_nci = gce_min(1.e-6f * gfk_exp(-(.46f * tairc)), 1.0f);
                dm = gce_max(qvk + qb0 - qsi, 0.0f);
                rsub1 = cs580 * qsi * rtair * rtair;
                y3 = FDV(1.0f, tair);
                dd = y3 * (C[S_RN30A] * y3 - C[S_RN30B])
                   + FDV(C[S_RN30C] * tair, esi);
                y1 = FDV(206.18f * ssi, dd);
                pidep = y1 * FSQ(FDV(r_nci * qik, r00));
                dep = FDV(FDV(dm, 1.0f + rsub1), d2t);
                if (dm > cmin2) {
                    float a2 = 1.0f;
                    if (pidep > dep && pidep > cmin2) {
                        a2 = FDV(dep, pidep);
                        pidep = dep;
                    }
                    psfi = FDV(r2is * a2 * .5f * qik * y1,
                               FSQ(ami100) - FSQ(ami40));
                } else {
                    pidep = 0.0f;
                    psfi = 0.0f;
                }
            }
            // pgacs dgacw dgacr (:2015-2069)
            float ee1 = (qck + qrk < 1.e-4f) ? .01f : 1.0f;
            const float ee2 = .09f;
            float egs = ee1 * gfk_exp(ee2 * tairc);
            if (tair >= t0) egs = 1.0f;
            y1 = fabsf(vg - vs);
            y2 = zs * zg;
            y3 = FDV(5.0f, y2);
            y4 = .08f * y3 * y3;
            y5 = .05f * y3 * y4;
            dd = y1 * (FDV(y3, gce_p5(zs)) + FDV(y4, gce_p3(zs)) + FDV(y5, zs));
            float pgacs = r2ig * r2is * r9r * egs * dd;
            float dgacs = (ihail == 1) ? pgacs : 0.0f;
            float wgacs = r2ig * r2is * r9r * dd;
            y1 = FDV(1.0f, gfk_pow(zg, bg3));
            float dgacw;
            if (ihail == 1) dgacw = r2ig * gce_max(r14r * qck * y1, 0.0f);
            else            dgacw = r2ig * gce_max(r14f * qck * y1, 0.0f);
            float qgacw = dgacw;
            y1 = fabsf(vg - vr);
            y2 = zr * zg;
            y3 = FDV(5.0f, y2);
            y4 = .08f * y3 * y3;
            y5 = .05f * y3 * y4;
            dd = r16r * y1 * (FDV(y3, gce_p5(zr)) + FDV(y4, gce_p3(zr))
                              + FDV(y5, zr));
            float dgacr = r2ig * gce_max(dd, 0.0f);
            float qgacr = dgacr;
            if (tair >= t0) {
                dgacs = 0.0f; wgacs = 0.0f; dgacw = 0.0f; dgacr = 0.0f;
            } else {
                pgacs = 0.0f; qgacw = 0.0f; qgacr = 0.0f;
            }
            // dgaci wgaci pgwet (:2075-2109)
            float dgaci = 0.0f, wgaci = 0.0f, pgwet = 0.0f;
            if (tair < t0) {
                y1 = FDV(qik, gfk_pow(zg, bg3));
                if (ihail == 1) {
                    dgaci = r2ig * r15r * y1;
                    wgaci = r2ig * r15ar * y1;
                } else {
                    dgaci = 0.0f;
                    wgaci = r2ig * r15af * y1;
                }
                if (tairc >= -50.0f) {
                    y1 = FDV(1.0f, alf + rn17c * tairc);
                    if (ihail == 1)
                        y3 = FDV(.78f, gce_p2(zg)) + FDV(r17aq * scv, gfk_pow(zg, bgh5));
                    else
                        y3 = FDV(.78f, gce_p2(zg)) + FDV(r17as * scv, gfk_pow(zg, bgh5));
                    y4 = alvr * dwv * (rp0 - (qvk + qb0)) - tca * tairc;
                    dd = y1 * (r17r * y4 * y3
                               + (wgaci + wgacs) * (alf + rn17b * tairc));
                    pgwet = r2ig * gce_max(dd, 0.0f);
                }
            }
            // negative cloud water (:2118-2142)
            y1 = FDV(qck, d2t);
            psacw = gce_min(y1, psacw);
            praut = gce_min(y1, praut);
            pracw = gce_min(y1, pracw);
            psfw = gce_min(y1, psfw);
            dgacw = gce_min(y1, dgacw);
            qsacw = gce_min(y1, qsacw);
            qgacw = gce_min(y1, qgacw);
            y1 = (psacw + praut + pracw + psfw + dgacw + qsacw + qgacw) * d2t;
            qck = qck - y1;
            if (qck < 0.0f) {
                float a1 = 1.0f;
                if (y1 != 0.0f) a1 = FDV(qck, y1) + 1.0f;
                psacw = psacw * a1; praut = praut * a1; pracw = pracw * a1;
                psfw = psfw * a1; dgacw = dgacw * a1; qsacw = qsacw * a1;
                qgacw = qgacw * a1;
                qck = 0.0f;
            }
            // shed (:2147-2157)
            float wgacr = pgwet - dgacw - wgaci - wgacs;
            y2 = dgacw + dgaci + dgacr + dgacs;
            if (pgwet >= y2) {
                wgacr = 0.0f; wgaci = 0.0f; wgacs = 0.0f;
            } else {
                dgacr = 0.0f; dgaci = 0.0f; dgacs = 0.0f;
            }
            // negative cloud ice (:2160-2182)
            y1 = FDV(qik, d2t);
            psaut = gce_min(y1, psaut);
            psaci = gce_min(y1, psaci);
            praci = gce_min(y1, praci);
            psfi = gce_min(y1, psfi);
            dgaci = gce_min(y1, dgaci);
            wgaci = gce_min(y1, wgaci);
            y2 = (psaut + psaci + praci + psfi + dgaci + wgaci) * d2t;
            qik = qik - y2 + pidep * d2t;
            if (qik < 0.0f) {
                float a2 = 1.0f;
                if (y2 != 0.0f) a2 = FDV(qik, y2) + 1.0f;
                psaut = psaut * a2; psaci = psaci * a2; praci = praci * a2;
                psfi = psfi * a2; dgaci = dgaci * a2; wgaci = wgaci * a2;
                qik = 0.0f;
            }
            float dlt3 = 0.0f, dlt2 = 0.0f;
            if (tair < t0) {
                if (qrk < 1.e-4f) { dlt3 = 1.0f; dlt2 = 1.0f; }
                if (qsk >= 1.e-4f) dlt2 = 0.0f;
            }
            if (ice2 == 1) { dlt3 = 1.0f; dlt2 = 1.0f; }
            float pr = (qsacw + praut + pracw + qgacw) * d2t;
            float ps = (psaut + psaci + psacw + psfw + psfi + dlt3 * praci) * d2t;
            float pg = ((1.0f - dlt3) * praci + dgaci + wgaci + dgacw) * d2t;
            // pracs psacr (:2229-2245)
            y1 = fabsf(vr - vs);
            y2 = zr * zs;
            y3 = FDV(5.0f, y2);
            y4 = .08f * y3 * y3;
            y5 = .05f * y3 * y4;
            float pracs = r2ig * r2is * r7r * y1
                        * (FDV(y3, gce_p5(zs)) + FDV(y4, gce_p3(zs)) + FDV(y5, zs));
            float psacr = r2is * r8r * y1
                        * (FDV(y3, gce_p5(zr)) + FDV(y4, gce_p3(zr)) + FDV(y5, zr));
            float qsacr = psacr;
            if (tair >= t0) { pracs = 0.0f; psacr = 0.0f; }
            else            { qsacr = 0.0f; }
            // pgaut pgfr (:2250-2266)
            float pgaut = 0.0f, pgfr = 0.0f;
            if (tair < t0) {
                y2 = gfk_exp(C[S_RN18A] * (t0 - tair));
                float temp = FDV(1.0f, zr);
                temp = temp * temp * temp * temp * temp * temp * temp;
                pgfr = r2ig * gce_max(r18r * (y2 - 1.0f) * temp, 0.0f);
            }
            // negative rain and snow (:2276-2332)
            y1 = FDV(qrk, d2t);
            y2 = FDV(-qgk, d2t);
            piacr = gce_min(y1, piacr);
            dgacr = gce_min(y1, dgacr);
            wgacr = gce_min(y1, wgacr);
            wgacr = gce_max(y2, wgacr);
            psacr = gce_min(y1, psacr);
            pgfr = gce_min(y1, pgfr);
            float del = 0.0f;
            if (wgacr < 0.0f) del = 1.0f;
            y1 = (piacr + dgacr + (1.0f - del) * wgacr + psacr + pgfr) * d2t;
            qrk = qrk + pr - y1 - del * wgacr * d2t;
            if (qrk < 0.0f) {
                float a1 = 1.0f;
                if (y1 != 0.0f) a1 = FDV(qrk, y1) + 1.0f;
                piacr = piacr * a1;
                dgacr = dgacr * a1;
                if (wgacr > 0.0f) wgacr = wgacr * a1;
                pgfr = pgfr * a1;
                psacr = psacr * a1;
                qrk = 0.0f;
            }
            float prn = d2t * ((1.0f - dlt3) * piacr + dgacr + wgacr
                               + (1.0f - dlt2) * psacr + pgfr);
            ps = ps + d2t * (dlt3 * piacr + dlt2 * psacr);
            pracs = (1.0f - dlt2) * pracs;
            y1 = FDV(qsk, d2t);
            pgacs = gce_min(y1, pgacs);
            dgacs = gce_min(y1, dgacs);
            wgacs = gce_min(y1, wgacs);
            pgaut = gce_min(y1, pgaut);
            pracs = gce_min(y1, pracs);
            float psn = d2t * (pgacs + dgacs + wgacs + pgaut + pracs);
            qsk = qsk + ps - psn;
            if (qsk < 0.0f) {
                float a2 = 1.0f;
                if (psn != 0.0f) a2 = FDV(qsk, psn) + 1.0f;
                pgacs = pgacs * a2; dgacs = dgacs * a2; wgacs = wgacs * a2;
                pgaut = pgaut * a2; pracs = pracs * a2; psn = psn * a2;
                qsk = 0.0f;
            }
            y2 = d2t * (psacw + psfw + dgacw + piacr + dgacr + wgacr + psacr
                        + pgfr);
            pt = pt + afcp * y2;
            qgk = qgk + pg + prn + psn;
            // psmlt pgmlt (:2342-2368)
            float psmlt = 0.0f, pgmlt = 0.0f;
            tair = (pt + tb0) * pi0;
            if (tair >= t0) {
                tairc = tair - t0;
                y1 = tca * tairc - alvr * dwv * (rp0 - (qvk + qb0));
                y2 = FDV(.78f, gce_p2(zs)) + FDV(r101f * scv, gfk_pow(zs, bsh5));
                dd = r11rt * y1 * y2 + r11at * tairc * (qsacw + qsacr);
                psmlt = r2is * gce_max(0.0f, gce_min(dd, qsk));
                if (ihail == 1)
                    y3 = FDV(.78f, gce_p2(zg)) + FDV(r19aq * scv, gfk_pow(zg, bgh5));
                else
                    y3 = FDV(.78f, gce_p2(zg)) + FDV(r19as * scv, gfk_pow(zg, bgh5));
                dd1 = r19rt * y1 * y3 + r19bt * tairc * (qgacw + qgacr);
                pgmlt = r2ig * gce_max(0.0f, gce_min(dd1, qgk));
                pt = pt - afcp * (psmlt + pgmlt);
                qrk = qrk + psmlt + pgmlt;
                qsk = qsk - psmlt;
                qgk = qgk - pgmlt;
            }
            // pihom pidw pimlt (:2374-2405)
            if (qck <= cmin1) qck = 0.0f;
            if (qik <= cmin1) qik = 0.0f;
            tair = (pt + tb0) * pi0;
            float pihom = (tair <= t00) ? qck : 0.0f;
            float pimlt = (tair >= t0) ? qik : 0.0f;
            float pidw = 0.0f;
            if (tair < t0 && tair > t00) {
                tairc = tair - t0;
                y1 = gce_max(gce_min(tairc, -1.0f), -31.0f);
                int it = (int)fabsf(y1);
                y2 = C[T_AA1 + it - 1];
                y3 = C[T_AA2 + it - 1];
                y4 = gfk_exp(fabsf(beta * tairc));
                y5 = gfk_pow(FDV(r00 * qik, r25a * y4), y3);
                pidw = gce_min(r25rt * y2 * y4 * y5, qck);
            }
            y1 = pihom - pimlt + pidw;
            pt = pt + afcp * y1 + ascp * pidep * d2t;
            qvk = qvk - pidep * d2t;
            qck = qck - y1;
            qik = qik + y1;
            // pint, itaobraun = 1 (:2415-2466)
            float pint = 0.0f;
            tair = (pt + tb0) * pi0;
            if (tair < t0) {
                if (qik <= cmin2) qik = 0.0f;
                tairc = tair - t0;
                rtair = FDV(1.0f, tair - c76);
                y2 = gfk_exp(c218 - c580 * rtair);
                qsi = rp0 * y2;
                esi = c610 * y2;
                ssi = FDV(qvk + qb0, qsi) - 1.0f;
                const float ami50 = 3.76e-8f;
                y1 = FDV(1.0f, tair);
                float tairccri = tairc;
                if (tairccri <= -30.0f) tairccri = -30.0f;
                y2 = gfk_exp(betah * tairccri);
                y3 = FSQ(qik);
                dd = y1 * (C[S_RN10A] * y1 - C[S_RN10B])
                   + FDV(C[S_RN10C] * tair, esi);
                pidep = gce_max(FDV(r32rt * ssi * y2 * y3, dd), 0.0f);
                float r_nci = gce_min(cn0 * gfk_exp(beta * tairc), 1.0f);
                dd = gce_max(FDV(1.e-9f * r_nci, r00)
                             - FDV(qik * 1.e-9f, ami50), 0.0f);
                dm = gce_max(qvk + qb0 - qsi, 0.0f);
                rsub1 = cs580 * qsi * rtair * rtair;
                dep = FDV(dm, 1.0f + rsub1);
                pint = gce_max(gce_min(dd, dm), 0.0f);
                pint = gce_min(pint + pidep, dep);
                if (pint <= cmin2) pint = 0.0f;
                pt = pt + ascp * pint;
                qvk = qvk - pint;
                qik = qik + pint;
            }
            // saturation adjustment, new_ice_sat = 2 (:2623-2655)
            dep = 0.0f;
            cnd = 0.0f;
            tair = (pt + tb0) * pi0;
            if (tair >= 253.16f) {
                y1 = FDV(1.0f, tair - c358);
                qsw = rp0 * gfk_exp(c172 - c409 * y1);
                dd = cp409 * y1 * y1;
                dm = qvk + qb0 - qsw;
                cnd = FDV(dm, 1.0f + avcp * dd * qsw);
                cnd = gce_max(-qck, cnd);
                pt = pt + avcp * cnd;
                qvk = qvk - cnd;
                qck = qck + cnd;
            }
            if (tair <= 258.16f) {
                y2 = FDV(1.0f, tair - c76);
                qsi = rp0 * gfk_exp(c218 - c580 * y2);
                dd1 = cp580 * y2 * y2;
                dep = FDV(qvk + qb0 - qsi, 1.0f + ascp * dd1 * qsi);
                dep = gce_max(-qik, dep);
                pt = pt + ascp * dep;
                qvk = qvk - dep;
                qik = qik + dep;
            }
            // psdep pssub pgsub (:2665-2722)
            float psdep = 0.0f, pssub = 0.0f, pgsub = 0.0f;
            tair = (pt + tb0) * pi0;
            if (tair < t0) {
                if (qsk < cmin1) qsk = 0.0f;
                if (qgk < cmin1) qgk = 0.0f;
                rtair = FDV(1.0f, tair - c76);
                qsi = rp0 * gfk_exp(c218 - c580 * rtair);
                ssi = FDV(qvk + qb0, qsi) - 1.0f;
                y1 = FDV(r10ar, tca * gce_p2(tair)) + FDV(1.0f, dwv * qsi);
                y2 = FDV(.78f, gce_p2(zs)) + FDV(r101f * scv, gfk_pow(zs, bsh5));
                psdep = FDV(r10t * ssi * y2, y1);
                pssub = psdep;
                psdep = r2is * gce_max(psdep, 0.0f);
                pssub = r2is * gce_max(-qsk, gce_min(pssub, 0.0f));
                if (ihail == 1)
                    y2 = FDV(.78f, gce_p2(zg)) + FDV(r20bq * scv, gfk_pow(zg, bgh5));
                else
                    y2 = FDV(.78f, gce_p2(zg)) + FDV(r20bs * scv, gfk_pow(zg, bgh5));
                pgsub = FDV(r2ig * r20t * ssi * y2, y1);
                dm = qvk + qb0 - qsi;
                rsub1 = cs580 * qsi * rtair * rtair;
                y1 = FDV(dm, 1.0f + rsub1);
                psdep = r2is * gce_min(psdep, gce_max(y1, 0.0f));
                y2 = gce_min(y1, 0.0f);
                pssub = r2is * gce_max(pssub, y2);
                dd = gce_max(-y2 - qsk, 0.0f);
                pgsub = r2ig * gce_min(gce_min(dd, qgk), gce_max(pgsub, 0.0f));
                float dlt1 = (qck + qik > 1.e-5f) ? 1.0f : 0.0f;
                psdep = dlt1 * psdep;
                pssub = (1.0f - dlt1) * pssub;
                pgsub = (1.0f - dlt1) * pgsub;
                pt = pt + ascp * (psdep + pssub - pgsub);
                qvk = qvk + pgsub - pssub - psdep;
                qsk = qsk + psdep + pssub;
                qgk = qgk - pgsub;
            }
            // rain evaporation (:2726-2745)
            ern = 0.0f;
            if (qrk > 0.0f) {
                tair = (pt + tb0) * pi0;
                rtair = FDV(1.0f, tair - c358);
                qsw = rp0 * gfk_exp(c172 - c409 * rtair);
                ssw = FDV(qvk + qb0, qsw) - 1.0f;
                dm = qvk + qb0 - qsw;
                rsub1 = cv409 * qsw * rtair * rtair;
                dd1 = gce_max(FDV(-dm, 1.0f + rsub1), 0.0f);
                y1 = FDV(.78f, gce_p2(zr)) + FDV(r23af * scv, gfk_pow(zr, bwh5));
                y2 = FDV(r23br, tca * gce_p2(tair)) + FDV(1.0f, dwv * qsw);
                ern = FDV(r23t * ssw * y1, y2);
                ern = gce_min(gce_min(dd1, qrk), gce_max(ern, 0.0f));
                pt = pt - avcp * ern;
                qvk = qvk + ern;
                qrk = qrk - ern;
            }
        }
        if (qck <= cmin1) qck = 0.0f;
        if (qrk <= cmin1) qrk = 0.0f;
        if (qik <= cmin1) qik = 0.0f;
        if (qsk <= cmin1) qsk = 0.0f;
        if (qgk <= cmin1) qgk = 0.0f;
        GQ(th, k) = pt;
        GQ(qv, k) = qvk;
        GQ(qc, k) = qck;
        GQ(qr, k) = qrk;
        GQ(qi, k) = qik;
        GQ(qs, k) = qsk;
        GQ(qg, k) = qgk;
    }
}
