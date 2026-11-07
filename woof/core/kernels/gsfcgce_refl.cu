// gpuwm/core/kernels/gsfcgce_refl.cu
//
// Goddard GCE (mp_physics = 97) radar reflectivity, WRF do_radar_ref = 1.
// Transcription authority: WRF v4.6.1 phys/module_mp_gsfcgce.F
// refl10cm_gsfc (:2942-3115) and the post-call column it is handed
// (:2866-2904). The complex mixing rules come from the allow-listed shared
// refl.cu header. The GCE backscatter below retains native WRF's REAL
// exponent and material constants. The shared refl.cu stays byte-frozen.
//
// The m(D) inputs and radar_init's gamma moments arrive as the float32
// values WRF holds (gsfcgce_constants.radar_parameters): xam_r = 3.14159 *
// rhowater/6, xam_s = 3.14159*rhosnow/6, xam_g from rhohail 917 or
// rhograul 400, fixed intercepts 8e6 / 1.6e7 / 2e5 (hail) or 4e6.  Species
// densities and the ze arrays are float32 and the slope/backscatter chain
// binary64, as in the Fortran's REAL and DOUBLE PRECISION locals.
//
// The column passed in is WRF's (:2876-2895) with one repair: under
// gsfcgce_2ice = 1 WRF leaves the graupel column uninitialised; the
// launcher passes zero there (gsfcgce_constants.DECLARED_DIVERGENCES).

// Native module_mp_radar.F:313-403. Reuse its complex mixing operations,
// retaining the native REAL exponent in (1./3.) and the caller's REAL PI5.
// The shared Morrison helper deliberately has its own numerical contract.
__device__ double gce_native_backscatter(double mass, double a, double b,
                                        double fm, cd mw, cd mi, double pi5)
{
    double mra = 0.9 + (1.0 - 0.9)*fm;
    double water = mass*fm;
    double dg = a*pow(mass,b);
    if (dg < 1.e-12) return 0.0;
    double vg = RPI/6.0*(dg*dg*dg);
    double density = fmax(fmin(mass/vg,900.0),10.0);
    vg = mass/density;
    double volume;
    if (mra <= 1.0-density/1000.0) {
        volume = vg*(1.0-mra*fm);
    } else {
        double limit = (900.0-density)/(mra*900.0-density+900.0*density/1000.0);
        volume = fm <= limit ? (1.0-mra*fm)*vg
                             : (mass-water)/900.0+water/1000.0;
    }
    double diameter = pow(6.0/RPI*volume,(double)(1.0f/3.0f));
    double icefrac = (mass-water)/(volume*900.0);
    double waterfrac = water/(1000.0*volume);
    cd core = refl_m_mix_nested(cd_make(1.0,0.0),mi,mw,
                                1.0-icefrac-waterfrac,icefrac,waterfrac);
    cd squared = cd_mul(core,core);
    double factor = cd_abs(cd_div(cd_sub(squared,cd_make(1.0,0.0)),
                                 cd_add(squared,cd_make(2.0,0.0))));
    double d2 = diameter*diameter;
    const double wavelength = (double)0.10f;
    const double wavelength4 = wavelength*wavelength*wavelength*wavelength;
    return factor*factor*pi5*(d2*d2*d2)/wavelength4;
}

extern "C" __global__ void gsfcgce_refl_column(
    const real* __restrict__ th, const real* __restrict__ pii,
    const real* __restrict__ p, const real* __restrict__ qv,
    const real* __restrict__ qr, const real* __restrict__ qs,
    const real* __restrict__ qg, const double* __restrict__ tables,
    const double k_w, const double m_w0_re, const double m_w0_im,
    const double m_i0_re, const double m_i0_im, const double pi5,
    const float xam_r, const float xam_s, const float xam_g,
    const float xcg3, const float xcg4, const float xcre1, const float xcre4,
    const float xocms, const float xocmg, const float n0_g,
    const int use_qs, const int use_qg, real* __restrict__ refl,
    const int nz, const int ny, const int nx)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    const int j = col / nx, i = col - j * nx;
    const double* xxds = tables;
    const double* xdts = tables + REFL_NRBINS;
    const double* xxdg = tables + 2 * REFL_NRBINS;
    const double* xdtg = tables + 3 * REFL_NRBINS;
    const double* simpson = tables + 4 * REFL_NRBINS;
    const cd mw = cd_make(m_w0_re, m_w0_im);
    const cd mi = cd_make(m_i0_re, m_i0_im);
    const double n0_r = 8.0e6, n0_s = 1.6e7;
    const double expo = (double)__fdiv_rn(1.0f, xcre1);
    float temp[REFL_KMAX], rr[REFL_KMAX], rs[REFL_KMAX], rg[REFL_KMAX];
    double ilamr[REFL_KMAX], ilams[REFL_KMAX], ilamg[REFL_KMAX];
    bool lqr[REFL_KMAX], lqs[REFL_KMAX], lqg[REFL_KMAX];
    float zer[REFL_KMAX], zes[REFL_KMAX], zeg[REFL_KMAX];
    for (int k = 0; k < nz; ++k) {
        const size_t idx = I3(k, j, i, ny, nx);
        temp[k] = th[idx] * pii[idx];
        const float qvk = fmaxf(1.e-10f, qv[idx]);
        const float pres = p[idx];
        const float rho = __fdiv_rn(0.622f * pres,
                                    287.0f * temp[k] * (qvk + 0.622f));
        const float qrk = qr[idx];
        const float qsk = use_qs ? qs[idx] : 0.0f;
        const float qgk = use_qg ? qg[idx] : 0.0f;
        lqr[k] = lqs[k] = lqg[k] = false;
        rr[k] = rs[k] = rg[k] = 1.e-12f;
        if (qrk > 1.e-9f) {
            rr[k] = qrk * rho;
            double lam = pow((double)(xam_r * xcg3) * n0_r / (double)rr[k], expo);
            ilamr[k] = 1.0 / lam;
            lqr[k] = true;
        }
        if (qsk > 1.e-9f) {
            rs[k] = qsk * rho;
            double lam = pow((double)(xam_s * xcg3) * n0_s / (double)rs[k], expo);
            ilams[k] = 1.0 / lam;
            lqs[k] = true;
        }
        if (qgk > 1.e-9f) {
            rg[k] = qgk * rho;
            double lam = pow((double)(xam_g * xcg3) * (double)n0_g
                             / (double)rg[k], expo);
            ilamg[k] = 1.0 / lam;
            lqg[k] = true;
        }
    }
    bool melti = false;
    int k0 = 0;
    for (int k = nz - 2; k >= 0; --k) {
        if (temp[k] > 273.15f && lqr[k] && (lqs[k + 1] || lqg[k + 1])) {
            k0 = max(k + 1, k0);
            melti = true;
            break;
        }
    }
    // (0.176/0.93)*(6/PIx)*(6/PIx) and the (xam/900)**2 factors are REAL.
    const float pix = 3.1415926536f;
    const float fac = __fdiv_rn(0.176f, 0.93f) * __fdiv_rn(6.0f, pix)
                    * __fdiv_rn(6.0f, pix);
    const float xs9 = __fdiv_rn(xam_s, 900.0f), xg9 = __fdiv_rn(xam_g, 900.0f);
    const float pre_s = fac * xs9 * xs9;
    const float pre_g = fac * xg9 * xg9;
    for (int k = 0; k < nz; ++k) {
        zer[k] = zes[k] = zeg[k] = 1.e-22f;
        if (lqr[k]) zer[k] = (float)(n0_r * (double)xcg4
                                     * pow(ilamr[k], (double)xcre4));
        if (lqs[k]) zes[k] = (float)((double)pre_s * n0_s * (double)xcg4
                                     * pow(ilams[k], (double)xcre4));
        if (lqg[k]) zeg[k] = (float)((double)pre_g * (double)n0_g
                                     * (double)xcg4
                                     * pow(ilamg[k], (double)xcre4));
    }
    if (melti && k0 >= 1) {
        for (int k = k0 - 1; k >= 0; --k) {
            if (lqs[k] && lqs[k0]) {
                double fm = fmax(0.005, fmin(1.0 - (double)__fdiv_rn(rs[k], rs[k0]), 0.99));
                double eta = 0.0, lam = 1.0 / ilams[k];
                for (int n = 0; n < REFL_NRBINS; ++n) {
                    double x = (double)xam_s * pow(xxds[n], 3.0);
                    double cb = gce_native_backscatter(
                        x, (double)xocms, (double)__fdiv_rn(1.0f, 3.0f), fm,
                        mw, mi, pi5);
                    double f_d = n0_s * exp(-lam * xxds[n]);
                    eta = eta + f_d * cb * simpson[n] * xdts[n];
                }
                const double w = (double)0.10f;
                zes[k] = (float)((w*w*w*w) / (pi5 * k_w) * eta);
            }
            if (lqg[k] && lqg[k0]) {
                double fm = fmax(0.005, fmin(1.0 - (double)__fdiv_rn(rg[k], rg[k0]), 0.99));
                double eta = 0.0, lam = 1.0 / ilamg[k];
                for (int n = 0; n < REFL_NRBINS; ++n) {
                    double x = (double)xam_g * pow(xxdg[n], 3.0);
                    double cb = gce_native_backscatter(
                        x, (double)xocmg, (double)__fdiv_rn(1.0f, 3.0f), fm,
                        mw, mi, pi5);
                    double f_d = (double)n0_g * exp(-lam * xxdg[n]);
                    eta = eta + f_d * cb * simpson[n] * xdtg[n];
                }
                const double w = (double)0.10f;
                zeg[k] = (float)((w*w*w*w) / (pi5 * k_w) * eta);
            }
        }
    }
    for (int k = 0; k < nz; ++k) {
        const float zsum = zer[k] + zes[k] + zeg[k];
        const float dbz = (float)(10.0 * log10((double)zsum * 1.0e18));
        refl[I3(k, j, i, ny, nx)] = fmaxf(-35.0f, dbz);
    }
}
