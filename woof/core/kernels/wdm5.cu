// WRF v4.6.1 WDM5, mp_physics=14. Public-domain WRF source.
// Pin d66e442fccc04111067e29274c9f9eaccc3cef28, phys/module_mp_wdm5.F.
// Five mass classes, no graupel, with CCN, cloud and rain numbers.
// One CUDA thread owns a vertical column in (k,j,i) layout, i fastest.
// The new shared headers retain stock WRF operation order and coefficients.
// GPUWM_NUMERICS=strict disables FMA for native replay; default permits FMA.
// See docs/wdm5_oracle_known_deltas.md for source and proof limits.

#ifndef WDM5_KMAX
#define WDM5_KMAX 64
#endif

// wdm5init's xlv1 = cl - cpv (:1666), the float32 word 0x4512799A.
#define WDM5_XLV1        2343.60010f
// 1.e20/16. folded at compile time (:1055), 0x5EAD78EC.
#define WDM5_1E20O16     6.25000013e+18f

// Saturation mixing ratio over water and over ice in the single-exponential
// expansion of the loop start and the post-update refresh (:582-606,
// :1487-1497): psat*exp(log(tr)*xa + xb*(1-tr)), then the 0.99p cap, the
// ep2 conversion and the qmin floor.
struct Wdm5Sat {
    float qs1, qs2;
};

__device__ __forceinline__ float wdm5_sat_single_exp(float logtr, float tr,
                                                      float aa, float bb,
                                                      float p) {
    float qs = 610.78f * gfk_exp(logtr * aa + bb * (1.0f - tr));
    qs = fminf(qs, 0.99f * p);
    qs = (287.0f / 461.6f) * qs / (p - qs);
    return fmaxf(qs, 1.0e-15f);
}

__device__ __forceinline__ Wdm5Sat wdm5_saturation(float t, float p) {
    const float ttp = 273.15f + 0.01f;                 // t0c + 0.01 (:573)
    const float xa = -(1846.4f - 4190.0f) / 461.6f;    // -dldt/rv
    const float xb = xa + 2.5e6f / (461.6f * ttp);
    const float xai = -(1846.4f - 2106.0f) / 461.6f;   // -dldti/rv
    const float xbi = xai + 2.85e6f / (461.6f * ttp);
    float tr = ttp / t;
    float logtr = gfk_log(tr);
    float xal = (t < ttp) ? xai : xa;
    float xbl = (t < ttp) ? xbi : xb;
    Wdm5Sat s;
    s.qs1 = wdm5_sat_single_exp(logtr, tr, xa, xb, p);
    s.qs2 = wdm5_sat_single_exp(logtr, tr, xal, xbl, p);
    return s;
}

// nislfv_rain_plm(id = 2, iter = 1) for snow (:765-782): one arrival-
// velocity iteration on slope_snow's speed of the arrival concentration,
// mean of departure and arrival winds; then qs = max(den*qs'/den, 0) on
// every layer, as the caller does after the routine returns.
template <int KMAX>
__device__ float wdm5_sediment_snow(
    float *qs, const float *works, const float *den, const float *denfac,
    const float *dz, const float *temp, int nz, float dt)
{
    float qq[KMAX], ww[KMAX], wd[KMAX], qn[KMAX];
    float zi[KMAX + 1], za[KMAX + 1], dza[KMAX + 1], qa[KMAX + 1];
    float allold = 0.0f;
    for (int k = 0; k < nz; ++k) {
        qq[k] = den[k] * qs[k];
        allold = allold + qq[k];
    }
    float precip = 0.0f;
    if (allold > 0.0f) {
        for (int k = 0; k < nz; ++k) {
            ww[k] = works[k];
            wd[k] = ww[k];
        }
        wdm_plm_arrival<KMAX>(qq, nullptr, ww, dz, nz, dt, zi, za, dza, qa,
                              nullptr);
        for (int k = 0; k < nz; ++k) {
            float q = qa[k] / den[k];
            float wa = wdm_snow_slope(q, den[k], denfac[k], temp[k]).vt;
            ww[k] = 0.5f * (wd[k] + wa);
        }
        wdm_plm_arrival<KMAX>(qq, nullptr, ww, dz, nz, dt, zi, za, dza, qa,
                              nullptr);
        precip = wdm_plm_remap<KMAX>(qa, zi, za, dza, nz, qn);
        for (int k = 0; k < nz; ++k) qq[k] = qn[k];
    }
    for (int k = 0; k < nz; ++k)
        qs[k] = fmaxf(qq[k] / den[k], 0.0f);
    return precip;
}

// The diffusivity/ventilation statement functions spelled inline (:804-809,
// :1021-1035).  A = viscos(t,den) = 1.496e-6*(t*sqrt(t))/(t+120)/den.
__device__ __forceinline__ float wdm5_viscos(float t, float den) {
    return 1.496e-6f * (t * sqrtf(t)) / (t + 120.0f) / den;
}

// venfac as the melting block writes it (:804-809).
__device__ __forceinline__ float wdm5_venfac_melt(float t, float p,
                                                  float den) {
    float a = wdm5_viscos(t, den);
    float b = 8.794e-5f * WDM_EXPLOG(t, 1.81f) / p;
    return WDM_EXPLOG(a / b, 0.3333333f) / sqrtf(wdm5_viscos(t, den))
         * sqrtf(sqrtf(1.28f / den));
}

// diffac (:1021-1029) for latent heat xl and saturation qsat.
__device__ __forceinline__ float wdm5_diffac(float xl, float t, float p,
                                             float den, float qsat) {
    return den * xl * xl * (t + 120.0f) * den
         / (1.414e3f * (1.496e-6f * (t * sqrtf(t))) * den
            * (461.6f * t * t))
         + p / (qsat * (8.794e-5f * WDM_EXPLOG(t, 1.81f)));
}

// venfac as the process block writes it (:1031-1035).
__device__ __forceinline__ float wdm5_venfac(float t, float p, float den) {
    float x = 1.496e-6f * (t * sqrtf(t)) * p
            / ((t + 120.0f) * den * (8.794e-5f * WDM_EXPLOG(t, 1.81f)));
    return gfk_exp(0.3333333f * gfk_log(x)) * sqrtf(sqrtf(1.28f / den))
         / sqrtf(1.496e-6f * (t * sqrtf(t)) / ((t + 120.0f) * den));
}

__device__ unsigned int wdm5_column_impl(
    float *theta, float *qv_g, float *qc_g, float *qi_g, float *qr_g,
    float *qs_g, float *nn_g, float *nc_g, float *nr_g,
    const float *den_g, const float *p, const float *pii,
    const float *dz_g,
    float *rainnc, float *rainncv, float *snownc, float *snowncv, float *sr,
    float *effc, float *effi, float *effs,
    float delt, int nz, int stride, int col)
{
    if (!isfinite(delt) || delt <= 0.0f) return 1u;
    // loops = max(nint(delt/dtcldcr),1) (:539): nint rounds half away.
    int loops = wdm_checked_steps(roundf(__fdiv_rn(delt, 120.0f)));
    if (loops < 0) return 2u;
    float dtcld = delt / (float)loops;                 // :540
    if (delt <= 120.0f) dtcld = delt;                  // :541
    int mstep = 1;
    unsigned int count_flags = 0u;

    float t[WDM5_KMAX], qv[WDM5_KMAX], qc[WDM5_KMAX], qi[WDM5_KMAX];
    float qr[WDM5_KMAX], qs[WDM5_KMAX];
    float nn[WDM5_KMAX], nc[WDM5_KMAX], nr[WDM5_KMAX];
    float den[WDM5_KMAX], delz[WDM5_KMAX], cpm[WDM5_KMAX], xl[WDM5_KMAX];
    float denfac[WDM5_KMAX];
    float qs1[WDM5_KMAX], qs2[WDM5_KMAX], rh1[WDM5_KMAX], rh2[WDM5_KMAX];
    float rslopec3_ls[WDM5_KMAX], xni_ls[WDM5_KMAX];
    float work1r[WDM5_KMAX], workn[WDM5_KMAX], works[WDM5_KMAX];
    float rslope3r_k[WDM5_KMAX], work1c[WDM5_KMAX];
    float collapse_slope[WDM5_KMAX];

    for (int k = 0; k < nz; ++k) {
        int id = col + k * stride;
        t[k] = theta[id] * pii[id];                    // :229
        den[k] = den_g[id]; delz[k] = dz_g[id];
        if (!isfinite(den[k]) || den[k] <= 0.0f || !isfinite(delz[k])
            || delz[k] <= 0.0f) count_flags |= 1u;
        // padding 0 for negative values generated by dynamics (:500-510).
        qv[k] = qv_g[id];
        qc[k] = fmaxf(qc_g[id], 0.0f); qr[k] = fmaxf(qr_g[id], 0.0f);
        qi[k] = fmaxf(qi_g[id], 0.0f); qs[k] = fmaxf(qs_g[id], 0.0f);
        nn[k] = fmaxf(nn_g[id], 0.0f); nc[k] = fmaxf(nc_g[id], 0.0f);
        nr[k] = fmaxf(nr_g[id], 0.0f);
        // cpmcal/xlcal once at scheme entry (:516-523), never refreshed.
        float vap = fmaxf(qv[k], 1.0e-15f);
        cpm[k] = 1004.5f * (1.0f - vap) + vap * 1846.4f;
        xl[k] = 2.5e6f - WDM5_XLV1 * (t[k] - 273.15f);
    }
    if (count_flags) return count_flags;

    rainncv[col] = 0.0f; snowncv[col] = 0.0f; sr[col] = 0.0f;   // :528-534

    for (int loop = 0; loop < loops; ++loop) {
        // ---- loop-start state (:559-693) --------------------------------
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            denfac[k] = wdm_denfac(den[k]);
            Wdm5Sat s = wdm5_saturation(t[k], p[id]);
            qs1[k] = s.qs1; qs2[k] = s.qs2;
            rh1[k] = fmaxf(qv[k] / qs1[k], 1.0e-15f);
            rh2[k] = fmaxf(qv[k] / qs2[k], 1.0e-15f);
            // rslopec3 of the loop-start cloud feeds pihtf/nihtf (:931);
            // xni of the loop-start ice feeds the ice fall speed (:843)
            // and nimlt (:907).
            rslopec3_ls[k] = wdm_cloud_slope(qc[k], nc[k], den[k]).r3;
            xni_ls[k] = wdm_xni(den[k], qi[k]);
            // slope_wdm5 at the loop start (:694-702).
            WdmRainSlope rr = wdm_rain_slope(qr[k], nr[k], den[k], denfac[k]);
            work1r[k] = rr.vt / delz[k];
            workn[k] = rr.vtn / delz[k];
            works[k] = wdm_snow_slope(qs[k], den[k], denfac[k], t[k]).vt;
        }

        // Native WRF NINT(max(v_mass,v_number)*dtcld + .5), per column.
        for (int k = 0; k < nz; ++k) {
            int steps = wdm_checked_steps(roundf(fmaxf(work1r[k], workn[k])
                                                   * dtcld + 0.5f));
            if (steps < 0) return 2u;
            if (steps > mstep) mstep = steps;
        }
        // ---- stock WRF rain mass and number fallout
        // on the geometric schedule (section 5; replaces :709-763) -------
        float fall_r = 0.0f;                           // fall(i,1,1)
        for (int n = 0; n < mstep; ++n) {
            float above_mass = 0.0f, above_number = 0.0f;
            for (int k = nz - 1; k >= 0; --k) {
                float mass = den[k] * qr[k] * work1r[k] / (float)mstep;
                float number = nr[k] * workn[k] / (float)mstep;
                if (k == nz - 1) {
                    qr[k] = fmaxf(qr[k] - mass * dtcld / den[k], 0.0f);
                    nr[k] = fmaxf(nr[k] - number * dtcld, 0.0f);
                } else {
                    qr[k] = fmaxf(qr[k] - (mass - above_mass * delz[k+1]
                               / delz[k]) * dtcld / den[k], 0.0f);
                    nr[k] = fmaxf(nr[k] - (number - above_number * delz[k+1]
                               / delz[k]) * dtcld, 0.0f);
                }
                above_mass = mass; above_number = number;
            }
            fall_r = fall_r + above_mass;
            for (int k = nz - 1; k >= 0; --k) {        // slope_rain refresh
                WdmRainSlope rr = wdm_rain_slope(qr[k], nr[k], den[k],
                                                 denfac[k]);
                work1r[k] = rr.vt / delz[k];
                workn[k] = rr.vtn / delz[k];
            }
        }

        // ---- snow semi-Lagrangian fallout (:765-782) --------------------
        float delqrs2 = wdm5_sediment_snow<WDM5_KMAX>(
            qs, works, den, denfac, delz, t, nz, dtcld);
        float fall_s = delqrs2 / delz[0] / dtcld;     // :781

        // ---- psmlt/nsmlt on the post-sedimentation slopes (:783-834) ----
        for (int k = nz - 1; k >= 0; --k) {
            int id = col + k * stride;
            WdmRainSlope rr = wdm_rain_slope(qr[k], nr[k], den[k], denfac[k]);
            WdmSlope ss = wdm_snow_slope(qs[k], den[k], denfac[k], t[k]);
            rslope3r_k[k] = rr.r3;                     // pgfrz reads it (:953)
            float n0sfac = wdm_n0sfac(t[k]);
            if (t[k] > 273.15f && qs[k] > 0.0f) {
                const float xlf = 3.5e5f;
                float work2 = wdm5_venfac_melt(t[k], p[id], den[k]);
                float coeres = ss.r2 * sqrtf(ss.r * ss.rb);
                float psmlt = __fdiv_rn(
                    __fdiv_rn(1.414e3f * wdm5_viscos(t[k], den[k]) * den[k],
                              xlf) * (273.15f - t[k]) * WDM_PI, 2.0f)
                    * n0sfac * (WDM_PRECS1 * ss.r2
                                + WDM_PRECS2 * work2 * coeres) / den[k];
                // Section 7: the full minor interval, not dtcld/mstep.
                psmlt = fminf(fmaxf(psmlt * dtcld / (float)mstep, -qs[k] / (float)mstep), 0.0f);
                if (qs[k] > 1.0e-9f) {                 // nsmlt (:825-828)
                    float sfac = ss.r * 2.0e6f * n0sfac * (float)mstep / qs[k];
                    nr[k] = nr[k] - sfac * psmlt;
                }
                qs[k] = qs[k] + psmlt;
                qr[k] = qr[k] - psmlt;
                t[k] = t[k] + __fdiv_rn(xlf, cpm[k]) * psmlt;
            }
        }

        // ---- cloud-ice fallout on the loop-start xni (:838-866) ---------
        for (int k = nz - 1; k >= 0; --k)
            work1c[k] = wdm_ice_fall_speed(qi[k], den[k], xni_ls[k]);
        float delqi = wdm_sediment_ice<WDM5_KMAX>(qi, work1c, den, delz, nz,
                                                  dtcld);
        float fall_c = delqi / delz[0] / dtcld;       // :865

        // ---- surface precipitation (:871-890) ---------------------------
        {
            float fallsum = fall_r + fall_s + fall_c;
            float fallsum_qsi = fall_s + fall_c;
            if (fallsum > 0.0f) {
                float step = __fdiv_rn(fallsum * delz[0], 1000.0f) * dtcld
                           * 1000.0f;
                rainncv[col] = step + rainncv[col];
                rainnc[col] = step + rainnc[col];
            }
            if (fallsum_qsi > 0.0f) {
                float step = __fdiv_rn(fallsum_qsi * delz[0], 1000.0f) * dtcld
                           * 1000.0f;
                snowncv[col] = step + snowncv[col];
                snownc[col] = step + snownc[col];
            }
            if (fallsum > 0.0f)
                sr[col] = snowncv[col] / (rainncv[col] + 1.0e-12f);
        }

        // ---- instantaneous melting and freezing (:896-976) --------------
        for (int k = 0; k < nz; ++k) {
            float supcol = 273.15f - t[k];
            float xlf = 2.85e6f - xl[k];
            if (supcol < 0.0f) xlf = 3.5e5f;
            if (supcol < 0.0f && qi[k] > 0.0f) {       // pimlt/nimlt
                qc[k] = qc[k] + qi[k];
                nc[k] = nc[k] + xni_ls[k];
                t[k] = t[k] - xlf / cpm[k] * qi[k];
                qi[k] = 0.0f;
            }
            if (supcol > 40.0f && qc[k] > 0.0f) {      // pihmf/nihmf
                qi[k] = qi[k] + qc[k];
                if (nc[k] > 0.0f) nc[k] = 0.0f;
                t[k] = t[k] + xlf / cpm[k] * qc[k];
                qc[k] = 0.0f;
            }
            if (supcol > 0.0f && qc[k] > 0.0f) {       // pihtf/nihtf
                float supcolt = fminf(supcol, 70.0f);
                float pfrzdtc = fminf(
                    __fdiv_rn(WDM_PI * WDM_PI * 100.0f
                              * (gfk_exp(0.66f * supcolt) - 1.0f) * 1000.0f
                              / den[k] * nc[k] * rslopec3_ls[k]
                              * rslopec3_ls[k], 18.0f) * dtcld, qc[k]);
                if (nc[k] > 1.0e1f) {
                    float nfrzdtc = fminf(
                        __fdiv_rn(WDM_PI * 100.0f
                                  * (gfk_exp(0.66f * supcolt) - 1.0f) * nc[k]
                                  * rslopec3_ls[k], 6.0f) * dtcld, nc[k]);
                    nc[k] = nc[k] - nfrzdtc;
                }
                qi[k] = qi[k] + pfrzdtc;
                t[k] = t[k] + xlf / cpm[k] * pfrzdtc;
                qc[k] = qc[k] - pfrzdtc;
            }
            if (supcol > 0.0f && qr[k] > 0.0f) {       // psfrz/nsfrz
                float supcolt = fminf(supcol, 70.0f);
                float pfrzdtr = fminf(
                    140.0f * (WDM_PI * WDM_PI) * 100.0f * nr[k] * 1000.0f
                    / den[k] * (gfk_exp(0.66f * supcolt) - 1.0f)
                    * rslope3r_k[k] * rslope3r_k[k] * dtcld, qr[k]);
                if (nr[k] > 1.0e-2f) {
                    float nfrzdtr = fminf(
                        4.0f * WDM_PI * 100.0f * nr[k]
                        * (gfk_exp(0.66f * supcolt) - 1.0f) * rslope3r_k[k]
                        * dtcld, nr[k]);
                    nr[k] = nr[k] - nfrzdtr;
                }
                qs[k] = qs[k] + pfrzdtr;
                t[k] = t[k] + xlf / cpm[k] * pfrzdtr;
                qr[k] = qr[k] - pfrzdtr;
            }
            nc[k] = fmaxf(nc[k], 0.0f);                // :971-976
            nr[k] = fmaxf(nr[k], 0.0f);
        }

        // ---- warm and cold process rates, conservation and the update
        // (:980-1472).  Every loop of the span is pointwise, so one pass per
        // level keeps each cell's statement order. ------------------------
        const float rdtcld = 1.0f / dtcld;             // :1163
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            // slope_wdm5 on the post-freezing state (:980-988).
            WdmRainSlope rr = wdm_rain_slope(qr[k], nr[k], den[k], denfac[k]);
            collapse_slope[k] = rr.r;
            WdmSlope ss = wdm_snow_slope(qs[k], den[k], denfac[k], t[k]);
            float avedia2 = rr.r * WDM_CBRT24;         // :995
            WdmCloudSlope cs = wdm_cloud_slope(qc[k], nc[k], den[k]);
            float avedia1 = cs.r;                      // :1009
            float work1w = wdm5_diffac(xl[k], t[k], p[id], den[k], qs1[k]);
            float work1i = wdm5_diffac(2.85e6f, t[k], p[id], den[k], qs2[k]);
            float work2 = wdm5_venfac(t[k], p[id], den[k]);

            // ==== warm rain (:1047-1148) ====
            float supsat = fmaxf(qv[k], 1.0e-15f) - qs1[k];
            float satdt = supsat / dtcld;
            float praut = 0.0f, nraut = 0.0f;
            float lencon = 2.7e-2f * den[k] * qc[k]
                         * (WDM5_1E20O16 * cs.r2 * cs.r2 - 0.4f);
            float lenconcr = fmaxf(1.2f * lencon, 1.0e-9f);
            if (avedia1 > 15.0e-6f) {                  // praut/nraut
                float taucon = 3.7f / den[k] / qc[k]
                             / (0.5e6f * cs.r - 7.5f);
                taucon = fmaxf(taucon, 1.0e-12f);
                praut = lencon / (taucon * den[k]);
                praut = fminf(fmaxf(praut, 0.0f), qc[k] / dtcld);
                nraut = 3.5e9f * den[k] * praut;
                if (qr[k] > lenconcr)
                    nraut = nr[k] / qr[k] * praut;
                nraut = fminf(nraut, nc[k] / dtcld);
            }
            WdmWarmNumber w = wdm_warm_collection(
                qc[k], qr[k], nc[k], nr[k], den[k], dtcld, lenconcr,
                avedia1, avedia2, cs.r3, rr.r3);
            float pracw = w.pracw, nracw = w.nracw;
            float nccol = w.nccol, nrcol = w.nrcol;
            float prevp = wdm_rain_evaporation(qr[k], rh1[k], work1w, work2,
                                               satdt, dtcld, rr,
                                               &nn[k], &nr[k]);

            // ==== cold rain (:1163-1299) ====
            float psdep = 0.0f, psaut = 0.0f, psaci = 0.0f, psacw = 0.0f;
            float pigen = 0.0f, pidep = 0.0f, psevp = 0.0f;
            float nsacw = 0.0f, nseml = 0.0f;
            float supcol = 273.15f - t[k];
            float n0sfac = wdm_n0sfac(t[k]);
            supsat = fmaxf(qv[k], 1.0e-15f) - qs2[k];
            satdt = supsat / dtcld;
            int ifsat = 0;
            float xni = wdm_xni(den[k], qi[k]);
            float eacrs = gfk_exp(0.07f * (-supcol));
            if (supcol > 0.0f) {
                if (qs[k] > 1.0e-9f && qi[k] > 1.0e-15f) {   // psaci
                    float xmi = den[k] * qi[k] / xni;
                    float diameter = fminf(11.9f * sqrtf(xmi), 500.0e-6f);
                    float vt2i = 1.49e4f * gfk_pow(diameter, 1.31f);
                    float vt2s = WDM_PVTS * ss.rb * denfac[k];
                    float acrfac = 2.0f * ss.r3 + 2.0f * diameter * ss.r2
                                 + diameter * diameter * ss.r;
                    psaci = __fdiv_rn(WDM_PI * qi[k] * eacrs * 2.0e6f
                                      * n0sfac * fabsf(vt2s - vt2i) * acrfac,
                                      4.0f);
                }
            }
            if (qs[k] > 1.0e-9f && qc[k] > 1.0e-15f)         // psacw
                psacw = fminf(WDM_PACRC * n0sfac * ss.r3 * ss.rb * qc[k]
                              * denfac[k], qc[k] * rdtcld);
            if (qs[k] > 1.0e-9f && nc[k] > 1.0e1f)           // nsacw
                nsacw = fminf(WDM_PACRC * n0sfac * ss.r3 * ss.rb * nc[k]
                              * denfac[k], nc[k] / dtcld);
            if (supcol <= 0.0f) {                            // nseml
                const float xlf = 3.5e5f;
                if (qs[k] > 1.0e-9f) {
                    float sfac = ss.r * 2.0e6f * n0sfac / qs[k];
                    nseml = -sfac * fminf(fmaxf(__fdiv_rn(4190.0f * supcol
                                                          * psacw, xlf),
                                                -qs[k] / dtcld), 0.0f);
                }
            }
            if (supcol > 0.0f) {
                if (qi[k] > 0.0f && ifsat != 1) {            // pidep
                    float xmi = den[k] * qi[k] / xni;
                    float diameter = 11.9f * sqrtf(xmi);
                    pidep = 4.0f * diameter * xni * (rh2[k] - 1.0f) / work1i;
                    float supice = satdt - prevp;
                    if (pidep < 0.0f) {
                        pidep = fmaxf(fmaxf(pidep, satdt * 0.5f), supice);
                        pidep = fmaxf(pidep, -qi[k] * rdtcld);
                    } else {
                        pidep = fminf(fminf(pidep, satdt * 0.5f), supice);
                    }
                    if (fabsf(prevp + pidep) >= fabsf(satdt)) ifsat = 1;
                }
                if (qs[k] > 0.0f && ifsat != 1) {            // psdep
                    float coeres = ss.r2 * sqrtf(ss.r * ss.rb);
                    psdep = (rh2[k] - 1.0f) * n0sfac
                          * (WDM_PRECS1 * ss.r2 + WDM_PRECS2 * work2 * coeres)
                          / work1i;
                    float supice = satdt - prevp - pidep;
                    if (psdep < 0.0f) {
                        psdep = fmaxf(psdep, -qs[k] * rdtcld);
                        psdep = fmaxf(fmaxf(psdep, satdt * 0.5f), supice);
                    } else {
                        psdep = fminf(fminf(psdep, satdt * 0.5f), supice);
                    }
                    if (fabsf(prevp + pidep + psdep) >= fabsf(satdt))
                        ifsat = 1;
                }
                if (supsat > 0.0f && ifsat != 1) {           // pigen
                    float supice = satdt - prevp - pidep - psdep;
                    float xni0 = 1.0e3f * gfk_exp(0.1f * supcol);
                    float roqi0 = 4.92e-11f * WDM_EXPLOG(xni0, 1.33f);
                    pigen = fmaxf(0.0f, (roqi0 / den[k] - fmaxf(qi[k], 0.0f))
                                        * rdtcld);
                    pigen = fminf(fminf(pigen, satdt), supice);
                }
                if (qi[k] > 0.0f) {                          // psaut
                    float qimax = WDM_ROQIMAX / den[k];
                    psaut = fmaxf(0.0f, (qi[k] - qimax) * rdtcld);
                }
            }
            if (supcol < 0.0f) {                             // psevp
                if (qs[k] > 0.0f && rh1[k] < 1.0f)
                    psevp = psdep * work1i / work1w;
                psevp = fminf(fmaxf(psevp, -qs[k] * rdtcld), 0.0f);
            }

            // ==== conservation and the update (:1306-1472) ====
#define WDM5_LIMIT(value_, source_, ...) do { \
    float _value = (value_); \
    float _source = (source_) * dtcld; \
    if (_source > _value) { float factor = _value / _source; __VA_ARGS__; } \
} while (0)
            if (t[k] <= 273.15f) {
                WDM5_LIMIT(fmaxf(1.0e-15f, qc[k]), praut + pracw + psacw,
                           praut *= factor; pracw *= factor;
                           psacw *= factor;);
                WDM5_LIMIT(fmaxf(1.0e-15f, qi[k]),
                           psaut + psaci - pigen - pidep,
                           psaut *= factor; psaci *= factor;
                           pigen *= factor; pidep *= factor;);
                WDM5_LIMIT(fmaxf(1.0e-15f, qr[k]), -praut - pracw - prevp,
                           praut *= factor; pracw *= factor;
                           prevp *= factor;);
                WDM5_LIMIT(fmaxf(1.0e-15f, qs[k]),
                           -psdep - psaut - psaci - psacw,
                           psdep *= factor; psaut *= factor;
                           psaci *= factor; psacw *= factor;);
                WDM5_LIMIT(fmaxf(1.0e1f, nc[k]),
                           +nraut + nccol + nracw + nsacw,
                           nraut *= factor; nccol *= factor;
                           nracw *= factor; nsacw *= factor;);
                WDM5_LIMIT(fmaxf(1.0e-2f, nr[k]), -nraut + nrcol,
                           nraut *= factor; nrcol *= factor;);
                float work2k = -(prevp + psdep + pigen + pidep);
                qv[k] = qv[k] + work2k * dtcld;
                qc[k] = fmaxf(qc[k] - (praut + pracw + psacw) * dtcld, 0.0f);
                qr[k] = fmaxf(qr[k] + (praut + pracw + prevp) * dtcld, 0.0f);
                qi[k] = fmaxf(qi[k] - (psaut + psaci - pigen - pidep) * dtcld,
                              0.0f);
                qs[k] = fmaxf(qs[k] + (psdep + psaut + psaci + psacw) * dtcld,
                              0.0f);
                nc[k] = fmaxf(nc[k] + (-nraut - nccol - nracw - nsacw)
                                      * dtcld, 0.0f);
                nr[k] = fmaxf(nr[k] + (nraut - nrcol) * dtcld, 0.0f);
                float xlf = 2.85e6f - xl[k];
                float xlwork2 = -2.85e6f * (psdep + pidep + pigen)
                              - xl[k] * prevp - xlf * psacw;
                t[k] = t[k] - xlwork2 / cpm[k] * dtcld;
            } else {
                WDM5_LIMIT(fmaxf(1.0e-15f, qc[k]), praut + pracw + psacw,
                           praut *= factor; pracw *= factor;
                           psacw *= factor;);
                WDM5_LIMIT(fmaxf(1.0e-15f, qr[k]),
                           -praut - pracw - prevp - psacw,
                           praut *= factor; pracw *= factor;
                           prevp *= factor; psacw *= factor;);
                WDM5_LIMIT(fmaxf(1.0e-9f, qs[k]), -psevp,
                           psevp *= factor;);
                WDM5_LIMIT(fmaxf(1.0e1f, nc[k]),
                           +nraut + nccol + nracw + nsacw,
                           nraut *= factor; nccol *= factor;
                           nracw *= factor; nsacw *= factor;);
                WDM5_LIMIT(fmaxf(1.0e-2f, nr[k]), -nraut - nseml + nrcol,
                           nraut *= factor; nseml *= factor;
                           nrcol *= factor;);
                float work2k = -(prevp + psevp);
                qv[k] = qv[k] + work2k * dtcld;
                qc[k] = fmaxf(qc[k] - (praut + pracw + psacw) * dtcld, 0.0f);
                qr[k] = fmaxf(qr[k] + (praut + pracw + prevp + psacw) * dtcld,
                              0.0f);
                qs[k] = fmaxf(qs[k] + psevp * dtcld, 0.0f);
                nc[k] = fmaxf(nc[k] + (-nraut - nccol - nracw - nsacw)
                                      * dtcld, 0.0f);
                nr[k] = fmaxf(nr[k] + (nraut + nseml - nrcol) * dtcld, 0.0f);
                float xlwork2 = -xl[k] * (prevp + psevp);
                t[k] = t[k] - xlwork2 / cpm[k] * dtcld;
            }
#undef WDM5_LIMIT
        }

        // ---- post-update: qs1/rh1 refresh (:1477-1497), the small-drop
        // collapse using the incoming rain slope (:1499-1523), CCN
        // activation and condensation (:1525-1575), padding (:1580-1607);
        // every loop pointwise. -------------------------------------------
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            Wdm5Sat s = wdm5_saturation(t[k], p[id]);
            qs1[k] = s.qs1;
            rh1[k] = fmaxf(qv[k] / qs1[k], 1.0e-15f);
            WdmRainSlope rr = wdm_rain_slope(qr[k], nr[k], den[k], denfac[k]);
            wdm_small_drop_collapse(collapse_slope[k], &qc[k], &nc[k], &qr[k], &nr[k]);
            wdm_activation_condensation(rh1[k], den[k], p[id], xl[k], cpm[k],
                                        dtcld, &t[k], &qv[k], &qc[k],
                                        &nn[k], &nc[k]);
            wdm_number_limits(den[k], &qc[k], &qi[k], &qr[k], &nc[k], &nr[k]);
        }
    }

    // ---- writeback, then effectRad_wdm5 (:2298-2396) on t1d = th*pii and
    // the driver's clamp (:290-313), in gpuwm's micron convention: WRF's
    // RRTMG forms microns as re*1.E6 in REAL, so the stored word is that one
    // float32 product of WRF's metres. ------------------------------------
    for (int k = 0; k < nz; ++k) {
        int id = col + k * stride;
        theta[id] = t[k] / pii[id];
        qv_g[id] = qv[k]; qc_g[id] = qc[k]; qi_g[id] = qi[k];
        qr_g[id] = qr[k]; qs_g[id] = qs[k];
        nn_g[id] = nn[k]; nc_g[id] = nc[k]; nr_g[id] = nr[k];

        float t1d = theta[id] * pii[id];
        float re_qc = 2.49e-6f, re_qi = 4.99e-6f, re_qs = 9.99e-6f;
        float rqc = fmaxf(1.0e-12f, qc[k] * den[k]);
        float rnc = fmaxf(1.0e-6f, nc[k] * den[k]);
        if (rqc > 1.0e-12f && rnc > 1.0e-6f) {
            float lamc = 2.0f * WDM_CDM2
                       * gfk_pow(WDM_PIDNC * nc[k] / rqc, 0.333333343f);
            re_qc = fmaxf(2.51e-6f, fminf((float)(1.0 / (double)lamc),
                                          50.0e-6f));
        }
        float rqi = fmaxf(1.0e-12f, qi[k] * den[k]);
        float ni = wdm_xni(den[k], qi[k]);
        float rni = fmaxf(1.0e-6f, ni * den[k]);
        if (rqi > 1.0e-12f && rni > 1.0e-6f) {
            float diai = 11.9f * sqrtf(rqi / ni);
            re_qi = fmaxf(10.01e-6f, fminf(0.75f * 0.163f * diai, 125.0e-6f));
        }
        float rqs = fmaxf(1.0e-12f, qs[k] * den[k]);
        if (rqs > 1.0e-12f) {
            float n0sfac = wdm_n0sfac(t1d);
            float lamdas = sqrtf(sqrtf(WDM_PIDN0S * n0sfac / rqs));
            re_qs = fmaxf(25.0e-6f, fminf(0.5f * (1.0f / lamdas), 999.0e-6f));
        }
        effc[id] = fmaxf(2.49e-6f, fminf(re_qc, 50.0e-6f)) * 1.0e6f;
        effi[id] = fmaxf(4.99e-6f, fminf(re_qi, 125.0e-6f)) * 1.0e6f;
        effs[id] = fmaxf(9.99e-6f, fminf(re_qs, 999.0e-6f)) * 1.0e6f;
    }
    return 0u;
}

#ifndef WDM5_CPU_MIRROR
extern "C" __global__ void wdm5_column(
    float *theta, float *qv, float *qc, float *qi, float *qr, float *qs,
    float *nn, float *nc, float *nr,
    const float *den, const float *p, const float *pii, const float *dz,
    float *rainnc, float *rainncv, float *snownc, float *snowncv, float *sr,
    float *effc, float *effi, float *effs,
    float delt, int nz, int ny, int nx, unsigned int *count_status)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = ny * nx;
    if (col >= stride || nz > WDM5_KMAX) return;
    unsigned int flags = wdm5_column_impl(
        theta, qv, qc, qi, qr, qs, nn, nc, nr, den, p, pii, dz,
        rainnc, rainncv, snownc, snowncv, sr, effc, effi, effs,
        delt, nz, stride, col);
    if (flags) atomicOr(count_status, flags);
}

// refl10cm_wdm5 (:2156-2294) and the driver's MAX(-35., dBZ) (:284), one
// thread per column.  The routine's REAL arithmetic stays float32 and its
// DOUBLE PRECISION arithmetic binary64, statement by statement; the
// melting-snow backscatter reuses wdm6_refl.cu's module_mp_radar machinery
// (refl_rayleigh_soak_wetgraupel), which the loader prepends to this unit.
// radar_init's REAL products after wdm5init's xam/xbm/xmu (:1730-1740) are
// the words WRF leaves in the radar module (tests/data/oracles/wdm5): rain
// xam_r = pi*denr/6, xbm 3, xmu 0, so xcre = (4,1,4,7), xcrg = (6,1,6,720),
// xorg2 = 1, xobmr = 1/3; snow xam_s = pi*dens/6 with the same tuples.
// radar_init's WGAMMA(7.) is the word 720.000061 (0x44340001), not 720.
// ``tables`` is gpuwm.core.refl's packed [xxDs | xdts | xxDg | xdtg |
// simpson] (radar_init's bins, graded against WRF's words).
#define WDM5_XAM_R   523.598816f      // 0x4402E653
#define WDM5_XAM_S   52.3598785f      // 0x42517084
#define WDM5_XOBMR   0.333333343f     // 0x3EAAAAAB
#define WDM5_XOCMS   0.267300904f     // 0x3E88DBAA
#define WDM5_XCRG4   720.000061f      // 0x44340001 (xcrg(4) == xcsg(4))
#define WDM5_REFL_NRBINS 50

extern "C" __global__ void refl10cm_wdm5_column(
    const float *qv1d, const float *qr1d, const float *nr1d,
    const float *qs1d, const float *t1d, const float *p1d,
    const double *tables, const double k_w, const double m_w0_re,
    const double m_w0_im, const double m_i0_re, const double m_i0_im,
    float *refl, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = ny * nx;
    if (col >= stride || nz > WDM5_KMAX) return;
    const double *xxds = tables;
    const double *xdts = tables + WDM5_REFL_NRBINS;
    const double *simpson = tables + 4 * WDM5_REFL_NRBINS;
    const cd mw = cd_make(m_w0_re, m_w0_im);
    const cd mi = cd_make(m_i0_re, m_i0_im);
    float temp[WDM5_KMAX], rr[WDM5_KMAX], nr[WDM5_KMAX], rs[WDM5_KMAX];
    double ilamr[WDM5_KMAX], ilams[WDM5_KMAX];
    double n0_r[WDM5_KMAX], n0_s[WDM5_KMAX];
    bool l_qr[WDM5_KMAX], l_qs[WDM5_KMAX];
    float ze_rain[WDM5_KMAX], ze_snow[WDM5_KMAX];

    for (int k = 0; k < nz; ++k) {                     // :2195-2226
        int id = col + k * stride;
        temp[k] = t1d[id];
        float temp_c = fminf(-0.001f, temp[k] - 273.15f);
        float qv = fmaxf(1.0e-10f, qv1d[id]);
        float pres = p1d[id];
        float rho = 0.622f * pres / (287.0f * temp[k] * (qv + 0.622f));
        n0_r[k] = 0.0; n0_s[k] = 0.0; ilamr[k] = 0.0; ilams[k] = 0.0;
        if (qr1d[id] > 1.0e-9f) {
            rr[k] = qr1d[id] * rho;
            nr[k] = nr1d[id] * rho;
            double lamr = (double)gfk_pow(
                WDM5_XAM_R * 6.0f * 1.0f * nr[k] / rr[k], WDM5_XOBMR);
            ilamr[k] = 1.0 / lamr;
            n0_r[k] = (double)(nr[k] * 1.0f) * pow(lamr, 1.0);
            l_qr[k] = true;
        } else {
            rr[k] = 1.0e-12f;
            nr[k] = 1.0e-12f;
            l_qr[k] = false;
        }
        if (qs1d[id] > 1.0e-9f) {
            rs[k] = qs1d[id] * rho;
            n0_s[k] = (double)fminf(1.0e11f,
                                    2.0e6f * gfk_exp(-(0.12f * temp_c)));
            double lams = pow((double)(WDM5_XAM_S * 6.0f) * n0_s[k]
                              / (double)rs[k], 0.25);
            ilams[k] = 1.0 / lams;
            l_qs[k] = true;
        } else {
            rs[k] = 1.0e-12f;
            l_qs[k] = false;
        }
    }

    bool melti = false;                                // :2231-2240
    int k_0 = 0;
    for (int k = nz - 2; k >= 0; --k) {
        if (temp[k] > 273.15f && l_qr[k] && l_qs[k + 1]) {
            k_0 = max(k + 1, k_0);
            melti = true;
            break;
        }
    }

    for (int k = 0; k < nz; ++k) {                     // :2248-2255
        ze_rain[k] = 1.0e-22f;
        ze_snow[k] = 1.0e-22f;
        // Section 9: numberless rain (nr = 0, so lamr = 0, ilamr = +Inf and
        // N0_r = 0) keeps the no-rain floor instead of 0*Inf = NaN.
        if (l_qr[k])
            ze_rain[k] = (float)(n0_r[k] * (double)WDM5_XCRG4
                                 * pow(ilamr[k], 7.0));
        if (l_qs[k]) {
            // Round the saved REAL coefficient products in WRF order.
            float c = __fmul_rn(0.189247310f, __fdiv_rn(6.0f, WDM_PI));
            c = __fmul_rn(c, __fdiv_rn(6.0f, WDM_PI));
            c = __fmul_rn(c, __fdiv_rn(WDM5_XAM_S, 900.0f));
            c = __fmul_rn(c, __fdiv_rn(WDM5_XAM_S, 900.0f));
            ze_snow[k] = (float)((double)c * n0_s[k] * (double)WDM5_XCRG4
                                 * pow(ilams[k], 7.0));
        }
    }

    if (melti && k_0 >= 1) {                           // :2266-2287
        for (int k = k_0 - 1; k >= 0; --k) {
            if (l_qs[k] && l_qs[k_0]) {
                double fmelt_s = fmax(0.005, fmin(1.0 - (double)(rs[k] / rs[k_0]),
                                                  0.99));
                double eta = 0.0;
                double lams = 1.0 / ilams[k];
                for (int n = 0; n < WDM5_REFL_NRBINS; ++n) {
                    double x = (double)WDM5_XAM_S * pow(xxds[n], 3.0);
                    double cback = refl_rayleigh_soak_wetgraupel(
                        x, (double)WDM5_XOCMS, (double)WDM5_XOBMR, fmelt_s,
                        // module_mp_radar.F:335 folds (1./3.) in REAL,
                        // then promotes that float32 word to binary64.
                        RMELT_OUTSIDE, mw, mi, (double)0.333333343f);
                    double f_d = n0_s[k] * pow(xxds[n], 0.0)
                               * exp(-lams * xxds[n]);
                    eta = eta + f_d * cback * simpson[n] * xdts[n];
                }
                ze_snow[k] = (float)(RLAMDA4 / (RPI5 * k_w) * eta);
            }
        }
    }

    for (int k = nz - 1; k >= 0; --k) {                // :2289-2291, :284
        int id = col + k * stride;
        float dbz = (float)(10.0 * log10((double)(ze_rain[k] + ze_snow[k])
                                         * 1.0e18));
        refl[id] = fmaxf(-35.0f, dbz);
    }
}
#endif
