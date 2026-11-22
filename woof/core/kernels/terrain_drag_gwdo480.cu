// gwd_opt = 1 as WRF v4.8.0 runs it: the KIM orographic drag revised by
// Hong et al. (2025) and Koo and Hong (2025), with the nonhydrostatic factor
// of Xu et al. (2024).  phys/module_bl_gwdo.F (gwdo) calling
// phys/physics_mmm/bl_gwdo.F90 (bl_gwdo_run), MMM-physics 550b5b49 as
// WRF v4.8.0 (06d4240a) carries it.
//
// This translation unit is compiled AFTER terrain_drag.cu, in its own module
// (gpuwm.core.terrain_drag:gwdo480): it reuses that file's td_min, td_max,
// td_atan2f and td_nint, and leaves the frozen module (the WRF v4.6.1/4.7.1
// gwdo_column, selected by gwdo_version = "wrf_471") byte for byte as it
// was.
//
// What changed from the 4.6.1/4.7.1 scheme (bl_gwdo.F90 4.7.1 -> 4.8.0):
//   * the reference level kref (was kbl) is capped at kgwdmax = kte/2 and
//     found with no 2*var > 0 guard; every low-level loop stops at kgwdmax;
//   * ELVMAX (omax, the maximum sub-grid orographic height above sea level)
//     sets two new levels: komax, the first level at or above omax (never
//     below kref), and kbomax, the first at or above omax + 2*var; flow
//     blocking accumulates from kbomax down and may stop only at or below
//     komax, measuring the potential energy against zl(komax);
//   * no drag where omax <= 50 m, or where N^2 < 0 below the reference level;
//   * ol and olp are floored at olmin before use; CON is clamped to [1, 10];
//   * base stress: cg 0.5 -> 1, the grid length cleff = dx -> dx_eff =
//     dx_factor*(dx+dy)/2 (gwd_dx_factor, default 2), and, under
//     gwd_if_nonhyd (default true), times (1 + the Xu et al. factor);
//   * the wave-breaking test no longer carries the always-true oa/kp1 guard;
//   * the wind-reversal limiter is per level (dtfac(k)) on every level below
//     kgwdmax with |taud| >= 1e-30, plus a density-scaled limiter on levels
//     at or above kgwdmax where p <= 7.5 hPa; the top level takes the level
//     below's factor.
//
// Every statement follows WRF's, in WRF's order of evaluation, under the
// conventions of terrain_drag.cu (-fmad=false, --ftz=false, glibc powf and
// expf as gfk_pow and gfk_exp, Fortran real powers through powf, integer
// powers as products).  The oracle is tools/gwdo_kim_wrf480_oracle
// (tests/test_gwdo_kim_wrf480_parity.py).

// --------------------------------------------------------------------------
// gwd_opt = 1, WRF v4.8.0: bl_gwdo_run through module_bl_gwdo.F::gwdo.
// --------------------------------------------------------------------------
// As in gwdo_column, per-level quantities WRF stores are recomputed where
// they are read, from the same words by the same statement.  usqj below the
// reference level and the averaged usqj(1) are written by WRF and never read
// (the stress loop starts at k >= kref >= kgwdmin = 2), so they are not
// formed; the averaged bnv2(1) is read only for the drag-off test and bnv.
//
// One case is not reproduced: WRF reads zl(komax) with komax = kte + 1, one
// level past the column, when ELVMAX is at or above the second-highest mass
// level's height (bl_gwdo.F90:293-301 then :563).  No terrain on Earth
// reaches a model top; the kernel reads zl(kte) there instead of memory
// outside the column.
extern "C" __global__
void gwdo480_column(const real *u, const real *v, const real *t,
                    const real *qv, const real *p, const real *p_interface,
                    const real *pi, const real *z,
                    const real *var, const real *oc1, const real *oa4,
                    const real *ol4, const real *omax_in, const real *sina,
                    const real *cosa, real *du, real *dv,
                    real *dtaux3d, real *dtauy3d, real *dusfcg, real *dvsfcg,
                    real dxmeter, real deltim, real dx_factor, int if_nonhyd,
                    real g_, real cp_, real rd_, real fv_, real pi_,
                    int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int st = ny * nx;
    if (col >= st || nz > TD_KMAX) return;
    const int kgwdmin = 2, mdir = 8;
    const real qmin = 1.0e-30f, ric = 0.25f, velmin = 1.0f, rimin = -100.0f;
    const real bnv2min = 1.0e-5f, efacmin = 0.0f, efacmax = 10.0f;
    const real frc = 1.0f, frmax = 10.0f, ce = 0.8f, cg = 1.0f;
    const real hmt_min = 50.0f, oc_min = 1.0f, oc_max = 10.0f;
    const real olmin = 1.0e-5f, odmin = 0.1f, odmax = 10.0f;
    const real pcutoff = 7.5e2f, pcutoff_den = 0.01f;
    const int kte = nz;                    // Fortran kts = 1 .. kte = nz
    const int kgwdmax = kte / 2;
    const int nwdir[8] = {6, 7, 5, 8, 2, 3, 1, 4};
    real ca = cosa[col], sa = sina[col];

    // Fortran-indexed accessors (k = 1..nz).
    #define Q(k) ((size_t)((k) - 1) * st + col)
    #define U1(k) (u[Q(k)] * ca - v[Q(k)] * sa)
    #define V1(k) (u[Q(k)] * sa + v[Q(k)] * ca)
    #define VTJ(k) (t[Q(k)] * (1.0f + fv_ * qv[Q(k)]))
    #define VTK(k) (VTJ(k) / pi[Q(k)])
    #define RHO(k) (1.0f / rd_ * p[Q(k)] / VTJ(k))
    #define PRSI(k) (p_interface[(size_t)((k) - 1) * st + col])
    #define PRSL(k) (p[Q(k)])
    #define DEL(k) (PRSI(k) - PRSI((k) + 1))
    #define ZL(k) (z[Q(k)])

    const real fdir = (real)mdir / (2.0f * pi_);
    // :222-234
    real delx = dxmeter, dely = dxmeter;
    real area = delx * dely;
    real dxy4[4], dxy4p[4];
    dxy4[0] = delx;
    dxy4[1] = dely;
    dxy4[2] = sqrtf(gfk_pow(delx, 2.0f) + gfk_pow(dely, 2.0f));
    dxy4[3] = dxy4[2];
    dxy4p[0] = dxy4[1];
    dxy4p[1] = dxy4[0];
    dxy4p[2] = dxy4[3];
    dxy4p[3] = dxy4[2];
    real dx_eff = dx_factor * (delx + dely) * 0.5f;
    real varc = var[col];
    real omax = omax_in[col];

    // :263-282 the reference level: the first level whose height above the
    // lowest reaches 2*var, then clamped to [kgwdmin, kgwdmax].
    real zref = 2.0f * varc;
    int kref = 0;
    for (int k = 2; k <= kte; ++k) {
        if (ZL(k) - ZL(1) >= zref) { kref = k + 1; break; }
    }
    kref = max(min(kref, kgwdmax), kgwdmin);
    // :287-302 komax: the first level at or above omax (heights above sea
    // level in WRF), never below kref.
    int komax = 0;
    for (int k = 2; k <= kte; ++k) {
        if (ZL(k) >= omax) { komax = k + 1; break; }
    }
    komax = max(komax, kref);
    // :306-317 kbomax: where flow blocking starts accumulating.
    int kbomax = 0;
    for (int k = 2; k <= kte; ++k) {
        if (ZL(k) >= (omax + zref)) { kbomax = k + 1; break; }
    }
    real delks = 1.0f / (PRSI(1) - PRSI(kref));
    real delks1 = 1.0f / (PRSL(1) - PRSL(kref));
    // :326-336 low-level means below the reference level.
    real ubar = 0.0f, vbar = 0.0f, rhobar = 0.0f;
    for (int k = 1; k <= kgwdmax; ++k) {
        if (k < kref) {
            real rcsks = DEL(k) * delks;
            real rdelks = DEL(k) * delks;
            ubar = ubar + rcsks * U1(k);
            vbar = vbar + rcsks * V1(k);
            rhobar = rhobar + rdelks * RHO(k);
        }
    }
    // :343-377 wind sector, orographic statistics along and across it.
    real oa, ol, oc, olp, od, dxy, dxyp;
    {
        real oa4c[4], ol4c[4];
        for (int m = 0; m < 4; ++m) {
            oa4c[m] = oa4[(size_t)m * st + col];
            ol4c[m] = ol4[(size_t)m * st + col];
        }
        real wdir = td_atan2f(ubar, vbar) + pi_;
        int idir = td_nint(fdir * wdir) % mdir + 1;
        int nwd = nwdir[idir - 1];
        int m = (nwd - 1) % 4;
        oa = (real)(1 - 2 * ((nwd - 1) / 4)) * oa4c[m];
        ol = td_max(ol4c[m], olmin);
        oc = td_min(td_max(oc1[col], oc_min), oc_max);
        real ol4p[4] = {ol4c[1], ol4c[0], ol4c[3], ol4c[2]};
        olp = td_max(ol4p[m], olmin);
        od = olp / ol;
        od = td_min(od, odmax);
        od = td_max(od, odmin);
        dxy = dxy4[m];
        dxyp = dxy4p[m];
    }

    // :381-393 usqj(k), bnv2(k) for k = 1..kte-1; bnv2(kte) keeps its 0.
    #define TI(k) (2.0f / (t[Q(k)] + t[Q((k) + 1)]))
    #define RDZ(k) (1.0f / (ZL((k) + 1) - ZL(k)))
    auto usqj_at = [&](int k) -> real {
        real ti = TI(k);
        real rdz = RDZ(k);
        real tem1 = U1(k) - U1(k + 1);
        real tem2 = V1(k) - V1(k + 1);
        real dw2 = tem1 * tem1 + tem2 * tem2;
        real shr2 = td_max(dw2, velmin) * rdz * rdz;
        real bvf2 = g_ * (g_ / cp_ + rdz * (VTJ(k + 1) - VTJ(k))) * ti;
        return td_max(bvf2 / shr2, rimin);
    };
    auto bnv2_at = [&](int k) -> real {
        if (k >= kte) return 0.0f;
        real rdz = RDZ(k);
        return 2.0f * g_ * rdz * (VTK(k + 1) - VTK(k))
               / (VTK(k + 1) + VTK(k));
    };

    // :397-413
    real ulow = td_max(sqrtf(ubar * ubar + vbar * vbar), 1.0f);
    real rulow = 1.0f / ulow;
    auto velco_at = [&](int k) -> real {
        real vc = 0.5f * ((U1(k) + U1(k + 1)) * ubar
                          + (V1(k) + V1(k + 1)) * vbar);
        vc = vc * rulow;
        if (vc < velmin && vc > 0.0f) vc = velmin;
        return vc;
    };

    // :415-429 drag-off tests.
    bool ldrag = omax <= hmt_min;
    ldrag = ldrag || varc <= 0.0f;
    ldrag = ldrag || velco_at(1) <= 0.0f;
    ldrag = ldrag || ulow == 1.0f;
    for (int k = kgwdmin; k <= kgwdmax; ++k)
        if (k < kref)
            ldrag = ldrag || velco_at(k) <= 0.0f || bnv2_at(k) < 0.0f;
    // :433-451 the low-level weighted N^2.
    real wtkbj = (PRSL(1) - PRSL(2)) * delks1;
    real bnv2_1 = wtkbj * bnv2_at(1);
    for (int k = kgwdmin; k <= kgwdmax; ++k) {
        if (k < kref) {
            real rdelks = (PRSL(k) - PRSL(k + 1)) * delks1;
            bnv2_1 = bnv2_1 + bnv2_at(k) * rdelks;
        }
    }
    ldrag = ldrag || bnv2_1 <= 0.0f;

    // :461-469
    real bnv = 0.0f, fr = 0.0f, xn = 0.0f, yn = 0.0f;
    if (!ldrag) {
        bnv = sqrtf(bnv2_1);
        fr = bnv * rulow * varc * od;
        fr = td_min(fr, frmax);
        xn = ubar * rulow;
        yn = vbar * rulow;
    }
    // :473-494 the base-level stress.
    real taub, coefm = 0.0f;
    if (!ldrag) {
        real efac = gfk_pow(oa + 2.0f, ce * fr / frc);
        efac = td_min(td_max(efac, efacmin), efacmax);
        coefm = gfk_pow(1.0f + ol, oa + 1.0f);
        real xlinv = coefm / dx_eff;
        real tem = fr * fr * oc;
        real gfac = tem / (tem + cg);
        taub = rhobar * efac * xlinv * gfac * ulow * ulow * ulow / bnv;
        if (if_nonhyd) {
            // -9./8.*tem + exp(-2./fr)*(-5./4./tem-0.5/fr+5./4.+9./4.*fr
            //                           +9./8.*tem): the literal quotients
            // are exact (1.125, 1.25, 2.25) and fold as gfortran folds them.
            tem = fr * fr;
            real nhd_effect = -(1.125f * tem)
                + gfk_exp(-(2.0f / fr))
                  * ((((-(1.25f / tem) - 0.5f / fr) + 1.25f) + 2.25f * fr)
                     + 1.125f * tem);
            taub = taub * (1.0f + nhd_effect);
        }
    } else {
        taub = 0.0f;
        xn = 0.0f;
        yn = 0.0f;
    }

    // :498-544 vertical structure of the stress.
    real taup[TD_KMAX + 1];
    for (int k = 1; k <= kte + 1; ++k) taup[k - 1] = 0.0f;
    for (int k = 1; k <= kgwdmax; ++k)
        if (k <= kref) taup[k - 1] = taub;
    bool icrilv = false;
    real brvf = 0.0f;
    for (int k = kgwdmin; k <= kte - 1; ++k) {
        int kp1 = k + 1;
        if (k >= kref) {
            real usqj = usqj_at(k);
            icrilv = icrilv || (usqj < ric) || (velco_at(k) <= 0.0f);
            brvf = td_max(bnv2_at(k), bnv2min);
            brvf = sqrtf(brvf);
        }
        if (k >= kref && !ldrag) {
            if (!icrilv && taup[k - 1] > 0.0f) {
                real velco = velco_at(k);
                real usqj = usqj_at(k);
                real temv = 1.0f / velco;
                real tem1 = coefm / dxy * (RHO(kp1) + RHO(k)) * brvf * velco
                            * 0.5f;
                real hd = sqrtf(taup[k - 1] / tem1);
                real fro = brvf * hd * temv;
                real tem2 = sqrtf(usqj);
                real tem = 1.0f + tem2 * fro;
                real rim = usqj * (1.0f - fro) / (tem * tem);
                if (rim <= ric) {
                    real temc = 2.0f + 1.0f / tem2;
                    hd = velco * (2.0f * sqrtf(temc) - temc) / brvf;
                    taup[kp1 - 1] = tem1 * hd * hd;
                } else {
                    taup[kp1 - 1] = taup[k - 1];
                }
            }
        }
    }
    // :553-592 flow blocking from ELVMAX, summed into the stress.
    if (!ldrag) {
        int kblk = 0;
        real fbdpe = 0.0f, fbdke = 0.0f, zblk = 0.0f;
        const real zl_omax = ZL(min(komax, kte));
        for (int k = kte; k >= kgwdmin; --k) {
            if (kblk == 0 && k <= kbomax) {
                fbdpe = fbdpe + bnv2_at(k) * (zl_omax - ZL(k)) * DEL(k) / g_
                        / RHO(k);
                fbdke = 0.5f * (gfk_pow(U1(k), 2.0f) + gfk_pow(V1(k), 2.0f));
                if (fbdpe >= fbdke && k <= komax) {
                    kblk = k;
                    zblk = ZL(kblk) - ZL(1);
                }
            }
        }
        if (kblk != 0) {
            real fbdcd = td_max(2.0f - 1.0f / od, 0.0f);
            real taufb = 0.5f * rhobar * coefm / area * fbdcd * dxyp * olp
                         * zblk * (ulow * ulow);
            real tautem = taufb / (real)(kblk - 1);
            taup[0] = taup[0] + taufb;
            for (int k = 2; k <= kblk; ++k) {
                taufb = taufb - tautem;
                taup[k - 1] = taup[k - 1] + taufb;
            }
            // taufb is 0 above kblk: taup there gains + 0.
            for (int k = kblk + 1; k <= kte + 1; ++k)
                taup[k - 1] = taup[k - 1] + 0.0f;
        }
    }
    // :596-623 -(g) d(tau)/dp and the per-level limiters.
    auto taud_at = [&](int k) -> real {
        return 1.0f * (taup[k] - taup[k - 1]) * g_ / DEL(k);
    };
    auto dtfac_at = [&](int k) -> real {
        real f = 1.0f;
        if (k <= kgwdmax - 1) {
            real td = taud_at(k);
            if (fabsf(td) >= qmin)
                f = td_min(f, fabsf(velco_at(k) / (deltim * td)));
        } else if (k <= kte - 1) {
            real td = taud_at(k);
            if (fabsf(td) >= qmin && PRSL(k) <= pcutoff) {
                real denfac = td_min(RHO(k) / pcutoff_den, 1.0f);
                f = td_min(f, denfac * fabsf(velco_at(k) / (deltim * td)));
            }
        }
        return f;
    };
    const real dtfac_top = dtfac_at(kte - 1);
    real dusfc = 0.0f, dvsfc = 0.0f;
    for (int k = 1; k <= kte; ++k) {
        real td = taud_at(k) * (k == kte ? dtfac_top : dtfac_at(k));
        real dtaux = td * xn;
        real dtauy = td * yn;
        dusfc = dusfc + dtaux * DEL(k);
        dvsfc = dvsfc + dtauy * DEL(k);
        size_t q = Q(k);
        // module_bl_gwdo.F hands bl_gwdo_run rublten and copies it back;
        // bl_gwdo.F90 rotates the drag to the model grid and adds it.
        du[q] = du[q] + dtaux * ca + dtauy * sa;
        dv[q] = dv[q] - dtaux * sa + dtauy * ca;
        if (dtaux3d != nullptr) {
            dtaux3d[q] = dtaux * ca + dtauy * sa;
            dtauy3d[q] = -dtaux * sa + dtauy * ca;
        }
    }
    dusfc = (-1.0f / g_) * dusfc;
    dvsfc = (-1.0f / g_) * dvsfc;
    if (dusfcg != nullptr) {
        dusfcg[col] = dusfc * ca + dvsfc * sa;
        dvsfcg[col] = -dusfc * sa + dvsfc * ca;
    }
    #undef Q
    #undef U1
    #undef V1
    #undef VTJ
    #undef VTK
    #undef RHO
    #undef PRSI
    #undef PRSL
    #undef DEL
    #undef ZL
    #undef TI
    #undef RDZ
}
