// WRF461 aero refl10cm_hm: REAL powers round before widening.
__device__ double ma_rayleigh_soak(
    double x_g, double a_geo, double b_geo, double fmelt,
    double meltratio_outside, cd m_w, cd m_i, double pi5, double lamda4)
{
    double fm = fmin(fmax(fmelt, 0.0), 1.0);                   // :290
    double mra = fmin(fmax(meltratio_outside, 0.0), 1.0);      // :292
    mra = mra + (1.0 - mra) * fm;                              // :299
    double x_w = x_g * fm;                                     // :301
    double d_g = a_geo * pow(x_g, b_geo);                      // :303
    if (d_g < 1.0e-12) return 0.0;                             // :305/:354-356
    double vg = (RPI / 6.0) * ((d_g * d_g) * d_g);                   // :307
    double rhog = fmin(fmax(x_g / vg, 10.0), 900.0);           // :308
    vg = x_g / rhog;                                           // :309
    double grenz = 1.0 - rhog / 1000.0;                        // :311
    double volg;
    if (mra <= grenz) {                                        // :313-317
        volg = vg * (1.0 - mra * fm);
    } else {                                                   // :319-331
        double fmgrenz = (900.0 - rhog)
                       / (mra * 900.0 - rhog + 900.0 * rhog / 1000.0);
        volg = (fm <= fmgrenz) ? (1.0 - mra * fm) * vg
                               : (x_g - x_w) / 900.0 + x_w / 1000.0;
    }
    double d_large = pow(6.0 / RPI * volg, (double)(1.0f / 3.0f));         // :335
    double volice = (x_g - x_w) / (volg * 900.0);              // :336
    double volwater = x_w / (1000.0 * volg);                   // :337
    double volair = 1.0 - volice - volwater;                   // :338
    cd m_core = refl_m_mix_nested(cd_make(1.0, 0.0), m_i, m_w,
                                  volair, volice, volwater);   // :342
    cd m2 = cd_mul(m_core, m_core);
    double kfac = cd_abs(cd_div(cd_sub(m2, cd_make(1.0, 0.0)),
                                cd_add(m2, cd_make(2.0, 0.0))));
    double d2 = d_large * d_large; double d4 = d2 * d2; double d6 = d2 * d4;
    return kfac * kfac * pi5 * d6 / lamda4;             // :351-352
}

// WRF461 aero refl10cm_hm: preserve REAL intermediates and init words.
extern "C" __global__ void refl10cm_morrison_aero_column(
    const real* __restrict__ qv, const real* __restrict__ qr,
    const real* __restrict__ nr, const real* __restrict__ qs,
    const real* __restrict__ ns, const real* __restrict__ qg,
    const real* __restrict__ ng, const real* __restrict__ t,
    const real* __restrict__ p, const double* __restrict__ tables,
    const double k_w, const double m_w0_re, const double m_w0_im,
    const double m_i0_re, const double m_i0_im,
    const double* __restrict__ rc,
    real* __restrict__ refl, const int nz, const int ny, const int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;

    const double xam_g = rc[2];
    const float xam_r32 = (float)rc[0], xam_s32 = (float)rc[1], xam_g32 = (float)rc[2];
    const float gamma4 = (float)rc[11], gamma7 = (float)rc[12];
    // radar_init bin tables, packed [xxDs | xdts | xxDg | xdtg | simpson].
    const double* xxds = tables;
    const double* xdts = tables + REFL_NRBINS;
    const double* xxdg = tables + 2 * REFL_NRBINS;
    const double* xdtg = tables + 3 * REFL_NRBINS;
    const double* simpson = tables + 4 * REFL_NRBINS;
    const cd m_w_0 = cd_make(m_w0_re, m_w0_im);
    const cd m_i_0 = cd_make(m_i0_re, m_i0_im);
    // radar_init soak geometry factors (module_mp_radar.F:177-183):
    // xocms = (1/xam_s)**(1/xbm_s), xocmg likewise.
    const double xobm = rc[6];
    const double xocms = rc[9];
    const double xocmg = rc[10];

    float rr[REFL_KMAX], rs[REFL_KMAX], rg[REFL_KMAX];
    double ilamr[REFL_KMAX], ilams[REFL_KMAX], ilamg[REFL_KMAX];
    double n0r[REFL_KMAX], n0s[REFL_KMAX], n0g[REFL_KMAX];
    bool lqr[REFL_KMAX], lqs[REFL_KMAX], lqg[REFL_KMAX];
    bool invalid_moment[REFL_KMAX];
    float zer[REFL_KMAX], zes[REFL_KMAX], zeg[REFL_KMAX];

    // Load the column and diagnose PSD slopes/intercepts from the
    // scheme's prognostic moments (:4536-4584).  Species densities keep
    // the Fortran's REAL storage; slope/intercept locals are its
    // DOUBLE PRECISION.
    for (int k = 0; k < nz; ++k) {
        size_t idx = I3(k, j, i, ny, nx);
        float temp = t[idx];
        float qvk = fmaxf(1.0e-10f, qv[idx]);           // :4540
        float rho = __fdiv_rn(0.622f * p[idx], (287.0f * temp) * (qvk + 0.622f));             // :4542
        rr[k] = 1.0e-12f; rs[k] = 1.0e-12f; rg[k] = 1.0e-12f;
        lqr[k] = false; lqs[k] = false; lqg[k] = false;
        invalid_moment[k] = false;
        ilamr[k] = 0.0; ilams[k] = 0.0; ilamg[k] = 0.0;        // unset in
        n0r[k] = 0.0; n0s[k] = 0.0; n0g[k] = 0.0;              // WRF too
        if (qr[idx] > 1.0e-9f) {                               // :4544-4556
            rr[k] = qr[idx] * rho;
            lqr[k] = true;
            if (!(nr[idx] > 0.0f) || !isfinite((double)nr[idx])) {
                invalid_moment[k] = true;
            } else {
                float nrk = nr[idx] * rho;
                double lamr = (double)r_pow(__fdiv_rn((xam_r32 * gamma4) * nrk, rr[k]), (float)xobm);
                ilamr[k] = 1.0 / lamr;
                n0r[k] = (double)nrk * lamr;  // xorg2 * lamr**xcre(2)
            }
        }
        if (qs[idx] > 1.0e-9f) {                               // :4558-4570
            rs[k] = qs[idx] * rho;
            lqs[k] = true;
            if (!(ns[idx] > 0.0f) || !isfinite((double)ns[idx])) {
                invalid_moment[k] = true;
            } else {
                float nsk = ns[idx] * rho;
                double lams = (double)r_pow(__fdiv_rn((xam_s32 * gamma4) * nsk, rs[k]), (float)xobm);
                ilams[k] = 1.0 / lams;
                n0s[k] = (double)nsk * lams;
            }
        }
        if (qg[idx] > 1.0e-9f) {                               // :4572-4584
            rg[k] = qg[idx] * rho;
            lqg[k] = true;
            if (!(ng[idx] > 0.0f) || !isfinite((double)ng[idx])) {
                invalid_moment[k] = true;
            } else {
                float ngk = ng[idx] * rho;
                double lamg = (double)r_pow(__fdiv_rn((xam_g32 * gamma4) * ngk, rg[k]), (float)xobm);
                ilamg[k] = 1.0 / lamg;
                n0g[k] = (double)ngk * lamg;
            }
        }
    }

    // Highest warm rainy level with frozen species just above; k_0 is
    // the level above it (:4586-4599).
    bool melti = false;
    int k_0 = 0;
    for (int k = nz - 2; k >= 0; --k) {
        if ((double)t[I3(k, j, i, ny, nx)] > 273.15 && lqr[k]
            && (lqs[k + 1] || lqg[k + 1])) {
            k_0 = max(k + 1, k_0);
            melti = true;
            break;
        }
    }

    // Rayleigh integrals; dry snow/graupel carry the
    // (0.176/0.93)*(6/pi)^2*(am/900)^2 adjustment (:4601-4620).
    for (int k = 0; k < nz; ++k) {
        zer[k] = 1.0e-22f; zes[k] = 1.0e-22f; zeg[k] = 1.0e-22f;
        if (lqr[k])
            zer[k] = (float)(n0r[k] * (double)gamma7 * pow(ilamr[k], 7.0));
        if (lqs[k])
            zes[k] = (float)((double)(((__fdiv_rn(0.176f, 0.93f) * __fdiv_rn(6.0f, 3.141592653589793f)) * __fdiv_rn(6.0f, 3.141592653589793f)) * __fdiv_rn(xam_s32, 900.0f) * __fdiv_rn(xam_s32, 900.0f))
                             * n0s[k] * (double)gamma7 * pow(ilams[k], 7.0));
        if (lqg[k])
            zeg[k] = (float)((double)(((__fdiv_rn(0.176f, 0.93f) * __fdiv_rn(6.0f, 3.141592653589793f)) * __fdiv_rn(6.0f, 3.141592653589793f)) * __fdiv_rn(xam_g32, 900.0f) * __fdiv_rn(xam_g32, 900.0f))
                             * n0g[k] * (double)gamma7 * pow(ilamg[k], 7.0));
    }

    // Melting snow/graupel: 50-bin Simpson integration of the soaked
    // particle backscatter below k_0, meltwater fraction from the mass
    // ratio against the k_0 reference level (:4626-4667).
    if (melti && k_0 >= 1) {
        for (int k = k_0 - 1; k >= 0; --k) {
            if (lqs[k] && lqs[k_0]) {                          // :4629-4645
                double fmelt_s = fmax(0.005,
                    fmin(1.0 - (double)__fdiv_rn(rs[k], rs[k_0]), 0.99));
                double eta = 0.0;
                double lams = 1.0 / ilams[k];
                for (int n = 0; n < REFL_NRBINS; ++n) {
                    double x = (double)xam_s32 * pow(xxds[n], 3.0);
                    double cback = ma_rayleigh_soak(
                        x, xocms, xobm, fmelt_s, RMELT_OUTSIDE,
                        m_w_0, m_i_0, rc[13], rc[14]);
                    double f_d = n0s[k] * exp(-lams * xxds[n]); // xmu_s = 0
                    eta += f_d * cback * simpson[n] * xdts[n];
                }
                zes[k] = (float)(rc[14] / (rc[13] * k_w) * eta);
            }
            if (lqg[k] && lqg[k_0]) {                          // :4648-4665
                double fmelt_g = fmax(0.005,
                    fmin(1.0 - (double)__fdiv_rn(rg[k], rg[k_0]), 0.99));
                double eta = 0.0;
                double lamg = 1.0 / ilamg[k];
                for (int n = 0; n < REFL_NRBINS; ++n) {
                    double x = xam_g * pow(xxdg[n], 3.0);
                    double cback = ma_rayleigh_soak(
                        x, xocmg, xobm, fmelt_g, RMELT_OUTSIDE,
                        m_w_0, m_i_0, rc[13], rc[14]);
                    double f_d = n0g[k] * exp(-lamg * xxdg[n]); // xmu_g = 0
                    eta += f_d * cback * simpson[n] * xdtg[n];
                }
                zeg[k] = (float)(rc[14] / (rc[13] * k_w) * eta);
            }
        }
    }

    // dBZ from the REAL ze sum (:4670-4671) with the wrapper's floor
    // (:916).
    for (int k = 0; k < nz; ++k) {
        float zsum = zer[k] + zes[k] + zeg[k];
        double dbz = 10.0 * log10((double)zsum * 1.0e18);
        // WRF has no guard for an active mass with a zero/negative/nonfinite
        // number moment (:4544-4584); its slope arithmetic is invalid before
        // the wrapper floor.  Preserve that invalidity explicitly instead of
        // letting CUDA fmaxf turn NaN into meteorological clear air.
        refl[I3(k, j, i, ny, nx)] = (invalid_moment[k]
            ? nanf("") : fmaxf(-35.0f, (float)dbz));
    }
}
