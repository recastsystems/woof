// Stock WRF v4.6.1 warm-rain, number and remap arithmetic for WDM5.
// Public-domain phys/module_mp_wdm5.F, pin d66e442fccc04111067e29274c9f9eaccc3cef28.
// WDM6 does not include this header, preserving its established arithmetic.
// Coefficients are native FP32 words; strict numerics disable FMA.

#ifndef GPUWM_WDM_COMMON_CUH
#define GPUWM_WDM_COMMON_CUH

// wdm5init (W5:1657-1743) == wdm6init words, gfortran 13.3 / glibc 2.39.
#define WDM_PI           3.14159274f      // 4.*atan(1.)        0x40490FDB
#define WDM_PIDNC        523.598816f      // pi*denr/6.         0x4402E653
#define WDM_PIDNR        12566.3711f      // 4.*pi*denr         0x4644597C
#define WDM_PVTR         2998.47632f      // avtr*g5pbr/24.     0x453B679F
#define WDM_PVTRN        1410.88000f      // avtr*g2pbr         0x44B05C29
#define WDM_PRECR1       9.80176926f      // 2.*pi*1.56         0x411CD40C
#define WDM_PRECR2       299.268738f      // :precr2            0x4395A266
#define WDM_PVTS         20.0516376f      // avts*g4pbs/6.      0x41A069C1
#define WDM_PIDN0S       628318528.0f     // pi*dens*n0s        0x4E15CD85
#define WDM_PACRC        55442112.0f      // :pacrc             0x4C537EB0
#define WDM_PRECS1       5200000.0f       // 4.*n0s*.65         0x4A9EB100
#define WDM_PRECS2       18681944.0f      // :precs2            0x4B8E882C
#define WDM_ROQIMAX      8.12500366e-05f  // 2.08e22*dimax**8   0x38AA64C8
#define WDM_RSLOPECMAX   1.99999999e-06f  //                    0x360637BD
#define WDM_RSLOPEC2MAX  3.99999998e-12f  //                    0x2C8CBCCC
#define WDM_RSLOPEC3MAX  7.99999954e-18f  //                    0x231392EE
#define WDM_RSLOPERMAX   1.99999995e-05f  //                    0x37A7C5AC
#define WDM_RSLOPERBMAX  1.74110086e-04f  //                    0x39369152
#define WDM_RSLOPER2MAX  3.99999978e-10f  //                    0x2FDBE6FE
#define WDM_RSLOPER3MAX  7.99999918e-15f  //                    0x28101D7C
#define WDM_RSLOPESMAX   9.99999975e-06f  //                    0x3727C5AC
#define WDM_RSLOPESBMAX  8.91250931e-03f  //                    0x3C1205C6
#define WDM_RSLOPES2MAX  9.99999944e-11f  //                    0x2EDBE6FE
#define WDM_RSLOPES3MAX  9.99999898e-16f  //                    0x26901D7C
// effectRad's rgmma(5./3.), the module's own truncated product.
#define WDM_CDM2         0.902613938f     //                    0x3F6711B5
// Constant expressions gfortran folds at compile time (MPFR, one rounding):
// (actr*1.E-6)**3 (W5:1536, W6:1962) and (24.)**(.3333333) (W5:995).
#define WDM_ACTR3        3.37500030e-18f  //                    0x227907F4
#define WDM_CBRT24       2.88449907f      //                    0x40389BA2
// The module's "A**B => exp(log(A)*(B))" spelling (W5:479), which is not
// pow's word, so it is evaluated as written.
#define WDM_EXPLOG(x, c) gfk_exp(gfk_log(x) * (c))

struct WdmColumnMath {
    __device__ static float pi() { return WDM_PI; }
    __device__ static float precr1() { return WDM_PRECR1; }
    __device__ static float precr2() { return WDM_PRECR2; }
    __device__ static float pidnr() { return WDM_PIDNR; }
    __device__ static float pidnc() { return WDM_PIDNC; }
    __device__ static float activation_divide(float a,float b) { return __fdiv_rn(a,b); }
    __device__ static float actr3() { return WDM_ACTR3; }
    __device__ static float exp(float x) { return gfk_exp(x); }
    __device__ static float pow(float x,float y) { return gfk_pow(x,y); }
    __device__ static float explog(float x,float y) { return WDM_EXPLOG(x,y); }
    __device__ static float divide(float a,float b) { return __fdiv_rn(a,b); }
    __device__ static float number(float den,float q,float lam,float pidn) {
        return __fdiv_rn(den*q*(lam*lam*lam),pidn);
    }
};


// --------------------------------------------------------------------------
// Size distributions (slope_wdm5 W5:1745-1812, slope_rain W5:1814-1859,
// slope_snow W5:1861-1907; the W6 routines are the same words).
// --------------------------------------------------------------------------

// denfac (W5:559-565): VREC then *den0 then VSQRT -- sqrt((1/den)*den0),
// two roundings before the root, not den0/den.
__device__ __forceinline__ float wdm_denfac(float den) {
    return sqrtf((1.0f / den) * 1.28f);
}

struct WdmSlope {
    float r, rb, r2, r3, vt;
};

struct WdmRainSlope {
    float r, rb, r2, r3, vt, vtn;
};

// Rain: gamma mu=1 PSD with the prognostic intercept,
// lamdar = exp(log((pidnr*nr)/(q*den))*.33333333), rslope capped at 1e-3.
__device__ __forceinline__ WdmRainSlope wdm_rain_slope(
    float q, float nr, float den, float denfac) {
    WdmRainSlope s;
    if (q <= 1.0e-9f || nr <= 1.0e-2f) {
        s.r = WDM_RSLOPERMAX; s.rb = WDM_RSLOPERBMAX;
        s.r2 = WDM_RSLOPER2MAX; s.r3 = WDM_RSLOPER3MAX;
    } else {
        float lamdar = WDM_EXPLOG((WDM_PIDNR * nr) / (q * den), 0.33333333f);
        s.r = fminf(1.0f / lamdar, 1.0e-3f);
        s.rb = gfk_pow(s.r, 0.8f); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = WDM_PVTR * s.rb * denfac;
    s.vtn = WDM_PVTRN * s.rb * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    if (nr <= 0.0f) s.vtn = 0.0f;
    return s;
}

// The snow intercept factor n0sfac = max(min(exp(alpha*supcol),
// n0smax/n0s), 1.) with supcol = t0c - t.
__device__ __forceinline__ float wdm_n0sfac(float temp) {
    return fmaxf(fminf(gfk_exp(0.12f * (273.15f - temp)), 5.0e4f), 1.0f);
}

// Snow: fixed n0s with the supercooling intercept factor.
__device__ __forceinline__ WdmSlope wdm_snow_slope(
    float q, float den, float denfac, float temp) {
    WdmSlope s;
    float n0sfac = wdm_n0sfac(temp);
    if (q <= 1.0e-9f) {
        s.r = WDM_RSLOPESMAX; s.rb = WDM_RSLOPESBMAX;
        s.r2 = WDM_RSLOPES2MAX; s.r3 = WDM_RSLOPES3MAX;
    } else {
        s.r = 1.0f / sqrtf(sqrtf(WDM_PIDN0S * n0sfac / (q * den)));
        s.rb = gfk_pow(s.r, 0.41f); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = WDM_PVTS * s.rb * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

// Cloud droplet slope (W5:675-683, :996-1004): rslopec = 1/lamdac with
// lamdac = exp(log((pidnc*nc)/(qc*den))*.33333333).
struct WdmCloudSlope {
    float r, r2, r3;
};

__device__ __forceinline__ WdmCloudSlope wdm_cloud_slope(
    float qc, float nc, float den) {
    WdmCloudSlope s;
    if (qc <= 1.0e-15f || nc <= 1.0e1f) {
        s.r = WDM_RSLOPECMAX; s.r2 = WDM_RSLOPEC2MAX; s.r3 = WDM_RSLOPEC3MAX;
    } else {
        s.r = 1.0f / WDM_EXPLOG((WDM_PIDNC * nc) / (qc * den), 0.33333333f);
        s.r2 = s.r * s.r;
        s.r3 = s.r2 * s.r;
    }
    return s;
}

// Ice crystal number, HDC 5c (W5:689-691): min(max(5.38e7*(den*max(qi,
// qmin))**0.75, 1e3), 1e6) spelled sqrt(sqrt(x*x*x)).
__device__ __forceinline__ float wdm_xni(float den, float qi) {
    float temp = den * fmaxf(qi, 1.0e-15f);
    temp = sqrtf(sqrtf(temp * temp * temp));
    return fminf(fmaxf(5.38e7f * temp, 1.0e3f), 1.0e6f);
}

// Saturation over water, the two-exponential fpvs expansion of the
// post-update condensation (W5:1550-1554, W6 the same).
__device__ __forceinline__ float wdm_qsat_water_two_exp(float t, float p) {
    const float ttp = 273.15f + 0.01f;
    const float xa = -(1846.4f - 4190.0f) / 461.6f;
    const float xb = xa + 2.5e6f / (461.6f * ttp);
    float tr = ttp / t;
    float qs = 610.78f * WDM_EXPLOG(tr, xa) * gfk_exp(xb * (1.0f - tr));
    qs = fminf(qs, 0.99f * p);
    qs = (287.0f / 461.6f) * qs / (p - qs);
    return fmaxf(qs, 1.0e-15f);
}

// --------------------------------------------------------------------------
// Warm-rain number sources and sinks (W5:1073-1146 == W6:1184-1257).
// --------------------------------------------------------------------------


__device__ __forceinline__ WdmWarmNumber wdm_warm_collection(
    float qc,float qr,float nc,float nr,float den,float dtcld,
    float lenconcr,float avedia1,float avedia2,float rslopec3,float rslope3r) {
    return wdm_warm_collection_shared<WdmColumnMath>(qc,qr,nc,nr,den,dtcld,
        lenconcr,avedia1,avedia2,rslopec3,rslope3r);
}

// Native rain evaporation and condensation, WRF WDM5:1127-1146.
// Number is returned to CCN where native WRF performs that operation.
__device__ __forceinline__ float wdm_rain_evaporation(
    float qr,float rh1,float work1w,float work2,float satdt,float dtcld,
    const WdmRainSlope &rr,float *nn,float *nr) {
    return wdm_rain_evaporation_shared<WdmColumnMath>(qr,rh1,work1w,work2,
        satdt,dtcld,rr,nn,nr);
}

// Nrevp_s/Prevp_s (W5:1501-1523 == W6:1926-1948): every raindrop below
// di82 mean-volume diameter returns to cloud, number and mass.
__device__ __forceinline__ void wdm_small_drop_collapse(
    float rslope_r,float *qc,float *nc,float *qr,float *nr) {
    wdm_small_drop_collapse_shared(WDM_CBRT24,rslope_r,qc,nc,qr,nr);
}

// CCN activation, cloud condensation and the ncevp number return
// (W5:1532-1573 == W6:1951-1998).  ``qs1`` is recomputed inside, after the
// activation's temperature update, exactly where the Fortran does it.
__device__ __forceinline__ void wdm_activation_condensation(
    float rh1,float den,float p,float xl,float cpm,float dtcld,
    float *t,float *qv,float *qc,float *nn,float *nc) {
    wdm_activate_shared<WdmColumnMath>(rh1,den,p,xl,cpm,dtcld,t,qv,qc,nn,nc);
    float qs1 = wdm_qsat_water_two_exp(*t,p);
    wdm_condense_shared(qs1,xl,cpm,dtcld,t,qv,qc,nn,nc);
}

// Padding for small values with the number-adjusting slope clamps
// (W5:1580-1606 == W6:2005-2030).
__device__ __forceinline__ void wdm_number_limits(
    float den,float *qc,float *qi,float *qr,float *nc,float *nr) {
    wdm_number_limits_shared<WdmColumnMath>(den,qc,qi,qr,nc,nr);
}

// --------------------------------------------------------------------------
// Semi-Lagrangian monotone-PLM sedimentation (nislfv_rain_plm W5:1909-2152,
// nislfv_rain_plmr/plm6 in W6, the same words).  The per-thread scratch is
// sized by the including unit's own level tier, hence the template.
// --------------------------------------------------------------------------

// Arrival-point construction: third-order interface velocity, the
// raingroup-top and diffusivity CFL limits, arrival coordinates and the
// deformed density concentrations.  Arrays with nz+1 entries are
// interfaces; qq2 may be null for a single species.
template <int KMAX>
__device__ __forceinline__ void wdm_plm_arrival(
    const float *qq, const float *qq2, const float *ww, const float *dz,
    int nz, float dt, float *zi, float *za, float *dza,
    float *qa, float *qa2)
{
    float wi[KMAX + 1];
    zi[0] = 0.0f;
    for (int k = 0; k < nz; ++k) zi[k + 1] = zi[k] + dz[k];

    // 2nd-order interface interpolation, then overwritten by the active
    // 3rd-order one (W5:1979-1993), exactly as the Fortran computes both.
    wi[0] = ww[0];
    wi[nz] = ww[nz - 1];
    for (int k = 1; k < nz; ++k)
        wi[k] = (ww[k] * dz[k - 1] + ww[k - 1] * dz[k])
              / (dz[k - 1] + dz[k]);
    wi[0] = ww[0];
    if (nz > 1) wi[1] = 0.5f * (ww[1] + ww[0]);
    for (int k = 2; k < nz - 1; ++k)
        wi[k] = (9.0f / 16.0f) * (ww[k] + ww[k - 1])
              - (1.0f / 16.0f) * (ww[k + 1] + ww[k - 2]);
    if (nz > 1) wi[nz - 1] = 0.5f * (ww[nz - 1] + ww[nz - 2]);
    wi[nz] = ww[nz - 1];

    for (int k = 1; k < nz; ++k)                   // W5:1996-1998
        if (ww[k] == 0.0f) wi[k] = ww[k - 1];
    for (int k = nz - 1; k >= 0; --k) {            // W5:2001-2007
        float decfl = (wi[k + 1] - wi[k]) * dt / dz[k];
        if (decfl > 0.05f) wi[k] = wi[k + 1] - 0.05f * dz[k] / dt;
    }
    for (int k = 0; k <= nz; ++k) za[k] = zi[k] - wi[k] * dt;
    for (int k = 0; k < nz; ++k) dza[k] = za[k + 1] - za[k];
    dza[nz] = zi[nz] - za[nz];
    for (int k = 0; k < nz; ++k) {
        qa[k] = qq[k] * dz[k] / dza[k];
        if (qq2) qa2[k] = qq2[k] * dz[k] / dza[k];
    }
    qa[nz] = 0.0f;
    if (qq2) qa2[nz] = 0.0f;
}

// Monotone PLM interface reconstruction, remap to the regular grid, and the
// rain-out integral (W5:2049-2144).  Returns the fallen mass per area.
template <int KMAX>
__device__ __forceinline__ float wdm_plm_remap(
    const float *qa, const float *zi, const float *za, const float *dza,
    int nz, float *qn)
{
    float qmi[KMAX + 1], qpi[KMAX + 1];
    for (int k = 1; k < nz; ++k) {
        float dip = (qa[k + 1] - qa[k]) / (dza[k + 1] + dza[k]);
        float dim = (qa[k] - qa[k - 1]) / (dza[k - 1] + dza[k]);
        if (dip * dim <= 0.0f) {
            qmi[k] = qa[k]; qpi[k] = qa[k];
        } else {
            qpi[k] = qa[k] + 0.5f * (dip + dim) * dza[k];
            qmi[k] = 2.0f * qa[k] - qpi[k];
            if (qpi[k] < 0.0f || qmi[k] < 0.0f) {
                qpi[k] = qa[k]; qmi[k] = qa[k];
            }
        }
    }
    qpi[0] = qa[0]; qmi[0] = qa[0];
    qpi[nz] = qa[nz]; qmi[nz] = qa[nz];
    for (int k = 0; k < nz; ++k) qn[k] = 0.0f;

    int kb = 0, kt = 0;
    for (int k = 0; k < nz; ++k) {
        kb = (kb > 0) ? kb - 1 : 0;
        kt = (kt > 0) ? kt - 1 : 0;
        if (zi[k] >= za[nz]) break;
        for (int kk = kb; kk < nz; ++kk) {
            if (zi[k] <= za[kk + 1]) { kb = kk; break; }
        }
        for (int kk = kt; kk < nz; ++kk) {
            if (zi[k + 1] <= za[kk]) { kt = kk; break; }
        }
        // Native WRF decrements the search result unconditionally.
        --kt;
        if (kt == kb) {
            // tl2 = tl*tl and th2 = th*th are their own roundings.
            float tl = (zi[k] - za[kb]) / dza[kb];
            float th = (zi[k + 1] - za[kb]) / dza[kb];
            float tl2 = tl * tl;
            float th2 = th * th;
            float qqd = 0.5f * (qpi[kb] - qmi[kb]);
            float qqh = qqd * th2 + qmi[kb] * th;
            float qql = qqd * tl2 + qmi[kb] * tl;
            qn[k] = (qqh - qql) / (th - tl);
        } else if (kt > kb) {
            float tl = (zi[k] - za[kb]) / dza[kb];
            float tl2 = tl * tl;
            float qqd = 0.5f * (qpi[kb] - qmi[kb]);
            float qql = qqd * tl2 + qmi[kb] * tl;
            float zsum = (1.0f - tl) * dza[kb];
            float qsum = (qa[kb] - qql) * dza[kb];
            for (int m = kb + 1; m < kt; ++m) {
                zsum = zsum + dza[m]; qsum = qsum + qa[m] * dza[m];
            }
            float th = (zi[k + 1] - za[kt]) / dza[kt];
            float th2 = th * th;
            float qqdt = 0.5f * (qpi[kt] - qmi[kt]);
            float dqh = qqdt * th2 + qmi[kt] * th;
            zsum = zsum + th * dza[kt]; qsum = qsum + dqh * dza[kt];
            qn[k] = qsum / zsum;
        }
    }

    float precip = 0.0f;
    for (int k = 0; k < nz; ++k) {
        if (za[k] < 0.0f && za[k + 1] < 0.0f) {
            precip = precip + qa[k] * dza[k];
        } else if (za[k] < 0.0f && za[k + 1] >= 0.0f) {
            precip = precip + qa[k] * (0.0f - za[k]);
            break;
        } else {
            break;
        }
    }
    return precip;
}

// Cloud-ice fall speed, HDC 5a (W5:838-848): 1.49e4*D**1.31 spelled
// exp(log(D)*1.31), on the diameter max(min(dicon*sqrt(xmi), dimax), 1e-25).
__device__ __forceinline__ float wdm_ice_fall_speed(float qi, float den,
                                                    float xni) {
    if (qi <= 0.0f) return 0.0f;
    float xmi = den * qi / xni;
    float diameter = fmaxf(fminf(11.9f * sqrtf(xmi), 500.0e-6f), 1.0e-25f);
    return 1.49e4f * WDM_EXPLOG(diameter, 1.31f);
}

// Cloud-ice fallout: the caller's den*qi, nislfv with iter = 0 (departure
// wind), then qi = max(den*qi'/den, 0) (W5:852-866).  The division back
// runs on every layer even when the column carries no ice, as the
// Fortran's caller does after nislfv returns early.
template <int KMAX>
__device__ __forceinline__ float wdm_sediment_ice(
    float *qi, const float *ww, const float *den, const float *dz, int nz,
    float dt)
{
    float qq[KMAX], qn[KMAX];
    float zi[KMAX + 1], za[KMAX + 1];
    float dza[KMAX + 1], qa[KMAX + 1];
    float allold = 0.0f;
    for (int k = 0; k < nz; ++k) {
        qq[k] = den[k] * qi[k];
        allold = allold + qq[k];
    }
    float precip = 0.0f;
    if (allold > 0.0f) {
        wdm_plm_arrival<KMAX>(qq, nullptr, ww, dz, nz, dt, zi, za, dza, qa,
                              nullptr);
        precip = wdm_plm_remap<KMAX>(qa, zi, za, dza, nz, qn);
        for (int k = 0; k < nz; ++k) qq[k] = qn[k];
    }
    for (int k = 0; k < nz; ++k)
        qi[k] = fmaxf(qq[k] / den[k], 0.0f);
    return precip;
}

#endif  // GPUWM_WDM_COMMON_CUH
