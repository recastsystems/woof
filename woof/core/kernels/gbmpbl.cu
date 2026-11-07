// Grenier-Bretherton-McCaa moist TKE PBL (bl_pbl_physics=12), WRF v4.6.1.
//
// Transcribed from phys/module_bl_gbmpbl.F (sha256 92d31829...a59a4cf); the
// :NNNN numbers below are that file's lines.  Graded word for word against
// WRF's own compiled object by tools/gbmpbl_wrf461_oracle and
// tests/test_gbmpbl_wrf461_parity.py.
//
// One CUDA thread owns one column and runs the whole GBMPBL wrapper on it:
// GBM2D twice (:251-280), the first pass's TKE kept as the new TKE_PBL and
// the two passes' momentum tendencies averaged, the second pass's
// everything else, then RTHBLTEN divided by the Exner function (:283-287).
// GBM2D's column arrays live in a global workspace interleaved across the
// warp (the Shin-Hong layout), indexed 1-based and top-down like the
// Fortran: level 1 is the model top, half level kte the lowest mass level,
// full level kte+1 the surface.  Engine arrays are bottom-up (nz, ny, nx);
// kr = kte+1-k is WRF's flip (:460, :476).
//
// Arithmetic.  WRF compiles phys/ with -O2 -fno-fast-math -ffp-contract=off
// -fno-tree-vectorize; nothing is contracted and every statement rounds in
// float32 in source order, so this file spells every expression with the
// Fortran's left-to-right association. Strict numerics compiles with
// -fmad=false --ftz=false; the fast default uses the common compiler policy.
// EXP, LOG and
// every REAL**REAL go through WOOF's own float32 routines gfk_exp, gfk_log
// and gfk_pow; integer powers are multiply chains (x**2 -> x*x, x**3 ->
// (x*x)*x, gfortran's own lowering); SQRT is correctly rounded on both
// sides.  The module is default real: nothing is promoted.
//
// Declared divergences, each a WRF read of an undefined value:
//   * kpbl2dx (GBM2D :415) is never initialised but pblhgt tests its sign
//     (:1028, :1039, :1047) and negates it when negative.  Here it starts at
//     0 in every column, which is the >= 0 arm WRF takes for every column
//     after the first in a tile and for the first whenever the stack word
//     happens to be non-negative (it was, in every oracle call).
//   * tridag's pivot array is gam(nmax=100) (:849-851); WRF overruns it for
//     kte > 100.  Here it is a column-sized workspace slot.  Identical
//     for every column WRF can run.
// roots (:1058-1089) has no forecast caller in WRF. It is retained below
// and checked independently against the source routine.

#define GBWS_LANES 32
#define GBWS_SLOTS 49

struct GbCol {
    real *p;
    __device__ __forceinline__ real &operator[](int k) const {
        return p[(size_t)k * (size_t)GBWS_LANES];
    }
};
struct GbColI {
    real *p;
    __device__ __forceinline__ int get(int k) const {
        return __float_as_int(p[(size_t)k * (size_t)GBWS_LANES]);
    }
    __device__ __forceinline__ void set(int k, int v) const {
        p[(size_t)k * (size_t)GBWS_LANES] = __int_as_float(v);
    }
};
#define GBWS_AT(base, slot, kp) \
    ((base) + (size_t)(slot) * (size_t)(kp) * (size_t)GBWS_LANES)

// Scheme constants are loaded from the hash-checked GBM table asset.

// Bottom-up engine field, WRF level kr (1-based).
#define GG(a, kr) a[(size_t)((kr) - 1) * (size_t)n + (size_t)col]

struct GbWork {
    // full levels (1..kte+1), zqx to kte+2
    GbCol zqx, kth, kzm, rhoxfl, tke, rrhoxfl, bbls, nsquar, bouyan, rdza,
          svs, presfl, exnerfl, shear, rexnerfl, rcldb, epop, sm, sh;
    // half levels (1..kte)
    GbCol ux, vx, thx, qx, thvx, zax, qix, kethl, thlx, tvx, rttenx, preshl,
          qcx, qwx, rrhoxhl, rhoxhl, exnerhl, rexnerhl, rdzq;
    // tridiagonal solver
    GbCol aimp, bimp, cimp, uimp1, rimp1, uimp2, rimp2, gam;
    // pblhgt layer bookkeeping (integers stored by bit pattern)
    GbColI ktop, kbot, ktop_save;
};

__device__ __forceinline__ GbWork gb_work(real *base, int kp) {
    GbWork w;
    int s = 0;
    GbCol *cols[] = {&w.zqx, &w.kth, &w.kzm, &w.rhoxfl, &w.tke, &w.rrhoxfl,
                     &w.bbls, &w.nsquar, &w.bouyan, &w.rdza, &w.svs,
                     &w.presfl, &w.exnerfl, &w.shear, &w.rexnerfl, &w.rcldb,
                     &w.epop, &w.sm, &w.sh, &w.ux, &w.vx, &w.thx, &w.qx,
                     &w.thvx, &w.zax, &w.qix, &w.kethl, &w.thlx, &w.tvx,
                     &w.rttenx, &w.preshl, &w.qcx, &w.qwx, &w.rrhoxhl,
                     &w.rhoxhl, &w.exnerhl, &w.rexnerhl, &w.rdzq, &w.aimp,
                     &w.bimp, &w.cimp, &w.uimp1, &w.rimp1, &w.uimp2,
                     &w.rimp2, &w.gam};
    static_assert(sizeof(cols) / sizeof(cols[0]) + 3 == GBWS_SLOTS,
                  "GBWS_SLOTS must count every workspace column");
    for (int i = 0; i < (int)(sizeof(cols) / sizeof(cols[0])); ++i)
        cols[i]->p = GBWS_AT(base, s++, kp);
    w.ktop.p = GBWS_AT(base, s++, kp);
    w.kbot.p = GBWS_AT(base, s++, kp);
    w.ktop_save.p = GBWS_AT(base, s++, kp);
    return w;
}

// tridag (:843-863).
__device__ void gb_tridag(const GbCol &a, const GbCol &b, const GbCol &c,
                          const GbCol &r, const GbCol &u, const GbCol &gam,
                          int nn) {
    real rbet = GB_LITERAL_7 / b[1];
    u[1] = r[1] * rbet;
    for (int j = 2; j <= nn; ++j) {
        gam[j] = c[j - 1] * rbet;
        rbet = GB_LITERAL_7 / (b[j] - a[j] * gam[j]);
        u[j] = (r[j] - a[j] * u[j - 1]) * rbet;
    }
    for (int j = nn - 1; j >= 1; --j)
        u[j] = u[j] - gam[j + 1] * u[j + 1];
}

// n2 (:1091-1131): buoyancy jump at the flux levels.
__device__ void gb_n2(const GbCol &thlx, const GbCol &qwx, const GbWork &w,
                      real cpoxlv, real xlvocp, real svp1pa, int kte) {
    for (int k = 2; k <= kte; ++k) {
        real templ = thlx[k] * w.exnerfl[k];
        real rvls = ((GB_LITERAL_11 * svp1pa)
                     * gfk_exp((GB_SVP2 * (templ - GB_SVPT0))
                               / (templ - GB_SVP3))) * w.epop[k];
        real temps = templ + (qwx[k] - rvls)
            / (cpoxlv + ((GB_EP_2 * GB_XLV) * rvls)
                        / (GB_R_D * (templ * templ)));
        rvls = ((GB_LITERAL_11 * svp1pa)
                * gfk_exp((GB_SVP2 * (temps - GB_SVPT0))
                          / (temps - GB_SVP3))) * w.epop[k];
        w.rcldb[k] = fmaxf(qwx[k] - rvls, GB_LITERAL_2);
        real tempv = (templ + xlvocp * w.rcldb[k])
            * ((GB_LITERAL_7 + GB_EP_1 * (qwx[k] - w.rcldb[k])) - w.rcldb[k]);
        real tvbl = tempv * w.rexnerfl[k];
        templ = thlx[k - 1] * w.exnerfl[k];
        rvls = ((GB_LITERAL_11 * svp1pa)
                * gfk_exp((GB_SVP2 * (templ - GB_SVPT0))
                          / (templ - GB_SVP3))) * w.epop[k];
        temps = templ + (qwx[k - 1] - rvls)
            / (cpoxlv + ((GB_EP_2 * GB_XLV) * rvls)
                        / (GB_R_D * (templ * templ)));
        rvls = ((GB_LITERAL_11 * svp1pa)
                * gfk_exp((GB_SVP2 * (temps - GB_SVPT0))
                          / (temps - GB_SVP3))) * w.epop[k];
        real rcld = fmaxf(qwx[k - 1] - rvls, GB_LITERAL_2);
        tempv = (templ + xlvocp * rcld)
            * ((GB_LITERAL_7 + GB_EP_1 * (qwx[k - 1] - rcld)) - rcld);
        real tvab = tempv * w.rexnerfl[k];
        real thvxfl = GB_LITERAL_5 * (tvab + tvbl);
        real dtvdz = (tvab - tvbl) * w.rdza[k];
        w.nsquar[k] = (GB_G / thvxfl) * dtvdz;
    }
    w.nsquar[1] = w.nsquar[2];
}

// The inversion-jump entrainment closure pblhgt (:977-984) and my
// (:1194-1198) share, returning biga.
__device__ __forceinline__ real gb_biga(const GbWork &w, real xlvocp, int k) {
    real delthvl = (w.thlx[k - 2] + (w.thx[k - 2] * GB_EP_1) * w.qwx[k - 2])
        - (w.thlx[k] + (w.thx[k] * GB_EP_1) * w.qwx[k]);
    real elambda = ((xlvocp * w.rcldb[k]) * w.rexnerhl[k])
        / fmaxf(delthvl, GB_LITERAL_3);
    real bige = GB_LITERAL_6 * elambda;
    return GB_AONE * (GB_LITERAL_7 + GB_ATWO * bige);
}

// pblhgt (:865-1055).  Returns iconv; writes bbls, ktop, kbot, pblx and
// kpbl2dx.  The two `goto 2745` restarts re-enter the layer loop from ibeg
// with the merged layer table, exactly as the Fortran DO statement label
// does.
__device__ int gb_pblhgt(const GbWork &w, real xlvocp, int kte, real *pblx,
                         int *kpbl2dx) {
    int iconv = 0, istabl = 1;
    for (int k = 2; k <= kte + 1; ++k) {
        if (w.nsquar[k] <= GB_LITERAL_2) {
            if (istabl == 1) {
                iconv = iconv + 1;
                w.ktop.set(iconv, k);
            }
            istabl = 0;
            w.kbot.set(iconv, k);
        } else {
            istabl = 1;
            w.bbls[k] = fminf(GB_RSTBL * sqrtf(w.tke[k] / w.nsquar[k]),
                              GB_KARMAN * w.zqx[k]);
        }
    }
    for (int l = 1; l <= iconv; ++l) w.ktop_save.set(l, w.ktop.get(l));
    if (iconv >= 1) {
        int ibeg = 1;
restart_2745:
        for (int ilay = ibeg; ilay <= iconv; ++ilay) {
            real blinf = GB_XFR * (w.zqx[w.ktop.get(ilay) - 1]
                                   - w.zqx[w.kbot.get(ilay) + 1]);
            real rnnll = GB_LITERAL_2, tkeavg = GB_LITERAL_2;
            int nlev = w.kbot.get(ilay) - w.ktop.get(ilay) + 1;
            const int k0 = w.ktop.get(ilay), k1 = w.kbot.get(ilay);
            for (int k = k0; k <= k1; ++k) {
                w.bbls[k] = fminf(blinf, GB_KARMAN * w.zqx[k]);
                rnnll = rnnll + (w.nsquar[k] * w.bbls[k]) * w.bbls[k];
                tkeavg = tkeavg + w.tke[k] / (real)nlev;
            }
            // First extend up (:936-969).
            const int kstart = w.ktop.get(ilay) - 1;
            for (int k = kstart; k >= 2; --k) {
                w.ktop.set(ilay, k);
                w.bbls[k] = fminf(GB_KARMAN * w.zqx[k], blinf);
                real trnnll = (w.nsquar[k] * w.bbls[k]) * w.bbls[k];
                if (trnnll * (real)nlev >= -(GB_LITERAL_5 * rnnll)) break;  // 2746
                if (ilay > 1) {
                    if (w.ktop.get(ilay) == w.kbot.get(ilay - 1)) {
                        ibeg = ilay - 1;
                        w.ktop.set(ibeg, w.ktop_save.get(ibeg));
                        w.kbot.set(ibeg, w.kbot.get(ibeg + 1));
                        iconv = iconv - 1;
                        for (int it = ibeg + 1; it <= iconv; ++it) {
                            w.ktop.set(it, w.ktop.get(it + 1));
                            w.kbot.set(it, w.kbot.get(it + 1));
                            w.ktop_save.set(it, w.ktop_save.get(it + 1));
                        }
                        goto restart_2745;
                    }
                }
                rnnll = rnnll + trnnll;
                nlev = nlev + 1;
            }
            // 2746: radiative/entrainment contribution (:971-985).
            {
                const int k = w.ktop.get(ilay);
                real radnnll = GB_LITERAL_2;
                if (w.qcx[k] > GB_LITERAL_12)
                    radnnll = (w.rttenx[k] * (w.presfl[k + 1] - w.presfl[k]))
                        / ((w.rhoxfl[k] * w.thvx[k]) * w.exnerfl[k]);
                real entnnll = GB_LITERAL_2;
                if (k >= 3) {
                    real biga = gb_biga(w, xlvocp, k);
                    entnnll = (biga * sqrtf((tkeavg * tkeavg) * tkeavg))
                        / w.bbls[k];
                }
                if (tkeavg > GB_LITERAL_2)
                    rnnll = rnnll + fminf(GB_LITERAL_2, (w.bbls[k] / sqrtf(tkeavg))
                                                * (radnnll + entnnll));
            }
            // Now extend down (:987-1007).
            const int kdown = w.kbot.get(ilay) + 1;
            for (int k = kdown; k <= kte + 1; ++k) {
                real tbbls = fminf(GB_KARMAN * w.zqx[k], blinf);
                real trnnll = (w.nsquar[k] * tbbls) * tbbls;
                if (trnnll * (real)nlev >= -(GB_LITERAL_5 * rnnll)) break;  // 2747
                w.kbot.set(ilay, k);
                if (ilay < iconv && w.kbot.get(ilay) == w.ktop.get(ilay + 1)) {
                    w.ktop.set(ilay, w.ktop_save.get(ilay));
                    w.kbot.set(ilay, w.kbot.get(ilay + 1));
                    iconv = iconv - 1;
                    for (int it = ilay + 1; it <= iconv; ++it) {
                        w.ktop.set(it, w.ktop.get(it + 1));
                        w.kbot.set(it, w.kbot.get(it + 1));
                        w.ktop_save.set(it, w.ktop_save.get(it + 1));
                    }
                    goto restart_2745;
                }
                rnnll = rnnll + trnnll;
                w.bbls[k] = tbbls;
                nlev = nlev + 1;
            }
        }
        for (int ilay = 1; ilay <= iconv; ++ilay) {
            real blinf = GB_XFR * (w.zqx[w.ktop.get(ilay) - 1]
                                   - w.zqx[w.kbot.get(ilay) + 1]);
            const int k0 = w.ktop.get(ilay), k1 = w.kbot.get(ilay);
            for (int k = k0; k <= k1; ++k)
                w.bbls[k] = fminf(GB_KARMAN * w.zqx[k], blinf);
        }
    }
    int kmix2dx;
    if (iconv > 0) {
        if (w.kbot.get(iconv) == kte + 1) {
            kmix2dx = w.ktop.get(iconv);
            if (*kpbl2dx >= 0) {
                *kpbl2dx = (iconv > 1) ? w.ktop.get(iconv - 1) : kmix2dx;
            } else {
                *kpbl2dx = -*kpbl2dx;
            }
        } else {
            kmix2dx = kte;
            *kpbl2dx = (*kpbl2dx >= 0) ? w.ktop.get(iconv) : -*kpbl2dx;
        }
    } else {
        kmix2dx = kte;
        *kpbl2dx = (*kpbl2dx >= 0) ? kmix2dx : -*kpbl2dx;
    }
    *pblx = w.zqx[kmix2dx];
    return iconv;
}

// my (:1133-1210): diffusivities for momentum, thetal/qw and TKE.
__device__ void gb_my(const GbWork &w, real xlvocp, int iconv, int kte) {
    const real a1ob1 = GB_A1 / GB_B1;
    w.kzm[kte + 1] = GB_LITERAL_2;
    w.kth[kte + 1] = GB_LITERAL_2;
    w.kzm[1] = GB_LITERAL_2;
    w.kth[1] = GB_LITERAL_2;
    w.sm[kte + 1] = GB_LITERAL_7;
    w.sh[kte + 1] = GB_LITERAL_7;
    w.sm[1] = GB_LITERAL_7;
    w.sh[1] = GB_LITERAL_7;
    for (int k = kte; k >= 2; --k) {
        real gh = -(((w.bbls[k] * w.bbls[k]) * w.nsquar[k])
                    / (GB_LITERAL_14 * w.tke[k] + GB_LITERAL_13));
        gh = fminf(gh, GB_LITERAL_1);
        gh = fmaxf(gh, -GB_LITERAL_4);
        w.sm[k] = (GB_A1 * (((GB_LITERAL_7 - GB_LITERAL_15 * GB_C1) - GB_LITERAL_18 * a1ob1)
                            - ((GB_LITERAL_15 * GB_A2) * gh)
                              * ((GB_B2 - GB_LITERAL_15 * GB_A2) * (GB_LITERAL_7 - GB_LITERAL_18 * a1ob1)
                                 - (GB_LITERAL_15 * GB_C1) * (GB_B2 + GB_LITERAL_18 * GB_A1))))
            / ((GB_LITERAL_7 - ((GB_LITERAL_15 * GB_A2) * gh) * (GB_LITERAL_18 * GB_A1 + GB_B2))
               * (GB_LITERAL_7 - ((GB_LITERAL_19 * GB_A1) * GB_A2) * gh));
        w.sh[k] = (GB_A2 * (GB_LITERAL_7 - GB_LITERAL_18 * a1ob1))
            / (GB_LITERAL_7 - ((GB_LITERAL_15 * GB_A2) * gh) * (GB_LITERAL_18 * GB_A1 + GB_B2));
        w.kzm[k] = fminf((w.bbls[k] * sqrtf(GB_LITERAL_14 * w.tke[k])) * w.sm[k],
                         GB_KMAX);
        w.kth[k] = fminf((w.bbls[k] * sqrtf(GB_LITERAL_14 * w.tke[k])) * w.sh[k],
                         GB_KMAX);
        w.kethl[k] = GB_NUK * sqrtf(w.kzm[k] * w.kzm[k + 1]);
        w.kethl[k] = fminf(w.kethl[k], GB_KMAX);
    }
    for (int ilay = 1; ilay <= iconv; ++ilay) {
        const int k = w.ktop.get(ilay);
        if (w.nsquar[k] >= GB_N2MIN) {
            w.kethl[k] = GB_NUK * w.kzm[k + 1];
            if (k >= 3) {
                w.kethl[k - 1] = GB_LITERAL_2;
                real biga = gb_biga(w, xlvocp, k);
                w.kth[k] = fminf(w.kth[k],
                    ((biga * sqrtf((w.tke[k] * w.tke[k]) * w.tke[k]))
                     / w.nsquar[k]) / fmaxf(w.bbls[k], w.bbls[k + 1]));
                w.kzm[k] = fminf(w.kzm[k], (w.kth[k] / w.sh[k + 1])
                                           * w.sm[k + 1]);
            }
        }
    }
    w.kethl[1] = w.kethl[2];
    w.kethl[kte] = (GB_NUK * GB_LITERAL_5) * w.kzm[kte];
}

// Implicit diffusion coefficients of the half-level solves (:579-592,
// :670-683, :717-730): kdiff is kth for scalars and ice, kzm for momentum.
__device__ __forceinline__ void gb_halfcoef(const GbWork &w, const GbCol &kd,
                                            real dt2, int kte) {
    for (int k = 1; k <= kte; ++k) {
        w.aimp[k] = (k == 1) ? GB_LITERAL_2
            : -(((((w.rhoxfl[k] * w.rrhoxhl[k]) * kd[k]) * dt2) * w.rdzq[k])
                * w.rdza[k]);
        w.cimp[k] = (k == kte) ? GB_LITERAL_2
            : -(((((w.rhoxfl[k + 1] * w.rrhoxhl[k]) * kd[k + 1]) * dt2)
                 * w.rdzq[k]) * w.rdza[k + 1]);
        w.bimp[k] = (GB_LITERAL_7 - w.aimp[k]) - w.cimp[k];
    }
}

// GBM2D (:292-840) for one column.  Pass 1 reads TKE from tke_in and
// writes the new TKE and its momentum tendencies; pass 2 reads the pass-1
// TKE back and writes every published output.
__device__ void gb_gbm2d(int pass, size_t col, size_t n, int kte,
                         const GbWork &w,
                         const real *u, const real *v, const real *t,
                         const real *qv, const real *qc, const real *qi,
                         const real *p, const real *pii, const real *dz8w,
                         const real *rthraten, const real *tke_in,
                         real psfc, real gz1oz0, real psim, real xland,
                         real hfxx, real qfxx, real tskx,
                         real *rublten, real *rvblten, real *rthblten,
                         real *rqvblten, real *rqcblten, real *rqiblten,
                         real *exch_m, real *exch_h, real *exch_tke,
                         real *el_pbl, real *tke_out,
                         real *pblh, real *ust, int *kpbl, real dt) {
    // ---0--- preliminaries (:437-443).
    const real rdt = GB_LITERAL_7 / dt;
    const real dt2 = dt;
    const real svp1pa = GB_SVP1 * GB_LITERAL_10;
    const real cpoxlv = GB_CP / GB_XLV;
    const real xlvocp = GB_XLV / GB_CP;
    const real r1cp = GB_LITERAL_7 / GB_CP;
    const real rczero = GB_LITERAL_7 / GB_CZERO;
    int kpbl2dx = 0;  // declared divergence, see the header
    real pblx;

    // ---1--- copy in, save initial values, derivatives (:456-537).
    w.zqx[kte + 1] = GB_LITERAL_2;
    w.zqx[kte + 2] = GB_LITERAL_2;
    w.tke[kte + 1] = GB_LITERAL_2;
    const real *tke_src = (pass == 1) ? tke_in : tke_out;
    for (int k = kte; k >= 1; --k) {
        const int kr = kte + 1 - k;
        if (pass == 1) {
            w.zqx[k] = w.zqx[k + 1] + GG(dz8w, kr);
            w.zax[k] = GB_LITERAL_5 * (w.zqx[k] + w.zqx[k + 1]);
        }
        w.tke[k] = GG(tke_src, kr);
        w.ux[k] = GG(u, kr);
        w.vx[k] = GG(v, kr);
        w.qx[k] = GG(qv, kr);
        w.qcx[k] = GG(qc, kr);
        w.qix[k] = GG(qi, kr);
        w.rttenx[k] = GG(rthraten, kr) * GG(pii, kr);   // t2dten_ra, :243
    }
    real tvcon = GB_LITERAL_2;
    for (int k = 1; k <= kte; ++k) {
        const int kr = kte + 1 - k;
        // Pressure and geometry are immutable between WRF's two GBM2D
        // passes. Keep their first-pass words, including strict pow/log/exp,
        // rather than evaluate the same transcendental functions twice.
        if (pass == 1) {
            w.preshl[k] = GG(p, kr);
            const real dzq = w.zqx[k] - w.zqx[k + 1];
            w.rdzq[k] = GB_LITERAL_7 / dzq;
            w.exnerhl[k] = gfk_pow(w.preshl[k] / GB_PREF, GB_RCP);
            w.rexnerhl[k] = GB_LITERAL_7 / w.exnerhl[k];
        }
        const real tx = GG(t, kr);
        w.thx[k] = tx * w.rexnerhl[k];
        w.qwx[k] = w.qx[k] + w.qcx[k];
        tvcon = (GB_LITERAL_7 + GB_EP_1 * w.qx[k]) - w.qcx[k];
        w.tvx[k] = tx * tvcon;
        w.thvx[k] = w.thx[k] * tvcon;
        w.thlx[k] = w.thx[k] - (xlvocp * w.rexnerhl[k]) * w.qcx[k];
        w.rhoxhl[k] = w.preshl[k] / (GB_R_D * w.tvx[k]);
        w.rrhoxhl[k] = GB_LITERAL_7 / w.rhoxhl[k];
    }
    w.presfl[1] = GB_LITERAL_2;
    for (int k = 2; k <= kte; ++k) {
        const int kr = kte + 1 - k;
        if (pass == 1) {
            w.presfl[k] = gfk_exp(GB_LITERAL_5 * (gfk_log(GG(p, kr + 1))
                                          + gfk_log(GG(p, kr))));
            w.epop[k] = GB_EP_2 / w.presfl[k];
            const real dza = w.zax[k - 1] - w.zax[k];
            w.rdza[k] = GB_LITERAL_7 / dza;
            w.exnerfl[k] = gfk_pow(w.presfl[k] / GB_PREF, GB_RCP);
            w.rexnerfl[k] = GB_LITERAL_7 / w.exnerfl[k];
        }
        const real fracz = (w.zqx[k] - w.zax[k]) * w.rdza[k];
        w.rhoxfl[k] = w.rhoxhl[k] + (w.rhoxhl[k - 1] - w.rhoxhl[k]) * fracz;
        w.rrhoxfl[k] = GB_LITERAL_7 / w.rhoxfl[k];
        const real dudz = (w.ux[k - 1] - w.ux[k]) * w.rdza[k];
        const real dvdz = (w.vx[k - 1] - w.vx[k]) * w.rdza[k];
        w.svs[k] = dudz * dudz + dvdz * dvdz;
    }
    w.presfl[kte + 1] = psfc;
    w.rexnerfl[kte + 1] = gfk_pow(GB_PREF / w.presfl[kte + 1], GB_RCP);
    w.exnerfl[kte + 1] = GB_LITERAL_7 / w.rexnerfl[kte + 1];
    const real q0s = GB_EP_2
        / (w.presfl[kte + 1]
           / ((GB_LITERAL_11 * svp1pa)
              * gfk_exp((GB_SVP2 * (tskx - GB_SVPT0)) / (tskx - GB_SVP3)))
           - GB_LITERAL_7);
    const real thgb = tskx * w.rexnerfl[kte + 1];
    const real rhoxsf = w.presfl[kte + 1] / (GB_R_D * w.tvx[kte]);
    const real wspd1 = sqrtf(w.ux[kte] * w.ux[kte] + w.vx[kte] * w.vx[kte])
        + GB_LITERAL_9;
    const real thv0 = thgb * (GB_LITERAL_7 + GB_EP_1 * q0s);
    const real dthv = w.thvx[kte] - thv0;
    real ustx = (wspd1 * GB_VK) / (gz1oz0 - psim);
    const real ustxm = fmaxf(ustx, GB_LITERAL_3);
    if ((xland - GB_LITERAL_8) < GB_LITERAL_2) ustx = ustxm;

    // ---4--- buoyancy jump at flux levels (:547-549).
    gb_n2(w.thlx, w.qwx, w, cpoxlv, xlvocp, svp1pa, kte);
    w.nsquar[kte + 1] = ((GB_G / w.thvx[kte]) * dthv) / w.zax[kte];

    // ---5--- PBL height and length scale (:554-562).
    const int iconv = gb_pblhgt(w, xlvocp, kte, &pblx, &kpbl2dx);

    // ---6/7--- diffusivities and implicit thetal/qw diffusion (:567-637).
    for (int ip = 1; ip <= 2; ++ip) {
        gb_my(w, xlvocp, iconv, kte);
        gb_halfcoef(w, w.kth, dt2, kte);
        for (int k = 1; k <= kte; ++k) {
            if (k == kte) {
                w.rimp2[k] = w.qwx[k]
                    + ((dt2 * qfxx) * w.rrhoxhl[k]) * w.rdzq[kte];
                w.rimp1[k] = w.thlx[k]
                    + ((((dt2 * hfxx) * w.rrhoxhl[k]) * r1cp) * w.rdzq[kte])
                      * w.rexnerhl[kte];
            } else {
                w.rimp2[k] = w.qwx[k];
                w.rimp1[k] = w.thlx[k];
            }
        }
        gb_tridag(w.aimp, w.bimp, w.cimp, w.rimp1, w.uimp1, w.gam, kte);
        gb_tridag(w.aimp, w.bimp, w.cimp, w.rimp2, w.uimp2, w.gam, kte);
        gb_n2(w.uimp1, w.uimp2, w, cpoxlv, xlvocp, svp1pa, kte);
        const real thx_t = w.uimp1[kte] + (xlvocp * w.rexnerhl[kte])
            * w.qcx[kte];
        const real thvx_t = thx_t * tvcon;
        const real dthv_t = thvx_t - thv0;
        w.nsquar[kte + 1] = ((GB_G / thvx_t) * dthv_t) / w.zax[kte];
    }

    // Update the scalars (:643-664).
    for (int k = 1; k <= kte; ++k) {
        w.thlx[k] = w.uimp1[k];
        w.qwx[k] = w.uimp2[k];
        const real templ = w.thlx[k] * w.exnerhl[k];
        real temps = templ;
        real rvls = GB_EP_2
            / (w.preshl[k] / ((GB_LITERAL_11 * svp1pa)
                              * gfk_exp((GB_SVP2 * (temps - GB_SVPT0))
                                        / (temps - GB_SVP3))) - GB_LITERAL_7);
        for (int it = 1; it <= 3; ++it) {
            temps = temps
                + ((((templ - temps) * GB_CP) / GB_XLV + w.qwx[k]) - rvls)
                  / (GB_CP / GB_XLV
                     + ((((GB_EP_2 * GB_XLV) * rvls) / GB_R_D) / temps)
                       / temps);
            rvls = GB_EP_2
                / (w.preshl[k] / ((GB_LITERAL_11 * svp1pa)
                                  * gfk_exp((GB_SVP2 * (temps - GB_SVPT0))
                                            / (temps - GB_SVP3))) - GB_LITERAL_7);
        }
        w.qcx[k] = fmaxf(w.qwx[k] - rvls, GB_LITERAL_2);
        w.qx[k] = w.qwx[k] - w.qcx[k];
        w.thx[k] = (templ + (w.qcx[k] * GB_XLV) / GB_CP) / w.exnerhl[k];
        w.thvx[k] = w.thx[k] * ((GB_LITERAL_7 + GB_EP_1 * w.qx[k]) - w.qcx[k]);
    }

    // ---8--- implicit diffusion of momentum (:670-711).
    gb_halfcoef(w, w.kzm, dt2, kte);
    for (int k = 1; k <= kte; ++k) {
        if (k == kte) {
            const real uflxp = -(((ustx * ustx) * w.ux[kte]) / wspd1);
            const real vflxp = -(((ustx * ustx) * w.vx[kte]) / wspd1);
            w.rimp1[k] = w.ux[k]
                + ((dt2 * uflxp) * (rhoxsf * w.rrhoxhl[k])) * w.rdzq[kte];
            w.rimp2[k] = w.vx[k]
                + ((dt2 * vflxp) * (rhoxsf * w.rrhoxhl[k])) * w.rdzq[kte];
        } else {
            w.rimp1[k] = w.ux[k];
            w.rimp2[k] = w.vx[k];
        }
    }
    gb_tridag(w.aimp, w.bimp, w.cimp, w.rimp1, w.uimp1, w.gam, kte);
    gb_tridag(w.aimp, w.bimp, w.cimp, w.rimp2, w.uimp2, w.gam, kte);
    for (int k = 1; k <= kte; ++k) {
        w.ux[k] = w.uimp1[k];
        w.vx[k] = w.uimp2[k];
        if (k >= 2) {
            const real dudz = (w.ux[k - 1] - w.ux[k]) * w.rdza[k];
            const real dvdz = (w.vx[k - 1] - w.vx[k]) * w.rdza[k];
            w.svs[k] = dudz * dudz + dvdz * dvdz;
        }
    }

    // ---9--- implicit diffusion of cloud ice (:717-737).
    gb_halfcoef(w, w.kth, dt2, kte);
    for (int k = 1; k <= kte; ++k) w.rimp1[k] = w.qix[k];
    gb_tridag(w.aimp, w.bimp, w.cimp, w.rimp1, w.qix, w.gam, kte);

    // ---10--- prediction of TKE (:750-813).
    for (int k = 2; k <= kte; ++k) {
        w.bouyan[k] = -(w.kth[k] * w.nsquar[k]);
        w.shear[k] = w.kzm[k] * w.svs[k];
    }
    for (int ilay = 1; ilay <= iconv; ++ilay) {
        const int k = w.ktop.get(ilay);
        if (w.qcx[k] > GB_LITERAL_12 && k > 1)
            w.bouyan[k] = w.bouyan[k]
                - ((((w.rttenx[k] * (w.presfl[k + 1] - w.presfl[k]))
                     * w.rrhoxfl[k]) * w.rexnerfl[k]) / w.thvx[k]);
    }
    w.tke[1] = GB_LITERAL_2;
    w.bbls[1] = GB_LITERAL_2;
    w.tke[kte + 1] = (GB_LITERAL_16 * ustx) * ustx;
    for (int k = 2; k <= kte; ++k) {
        w.aimp[k - 1] = (k == 2) ? GB_LITERAL_2
            : -(((((w.rhoxhl[k - 1] * w.rrhoxfl[k]) * w.kethl[k - 1]) * dt2)
                 * w.rdzq[k - 1]) * w.rdza[k]);
        if (k == kte) {
            w.cimp[k - 1] = GB_LITERAL_2;
            if (w.bbls[k] > GB_LITERAL_2) {
                w.bimp[k - 1] = ((GB_LITERAL_7 - w.aimp[k - 1]) - w.cimp[k - 1])
                    + dt2 * ((sqrtf(w.tke[k]) * rczero) / w.bbls[k]
                             + (((w.rhoxhl[k] * w.rrhoxfl[k]) * w.kethl[k])
                                * w.rdzq[k]) * w.rdza[k]);
            } else {
                w.bimp[k - 1] = ((GB_LITERAL_7 - w.aimp[k - 1]) - w.cimp[k - 1])
                    + ((((dt2 * w.rhoxhl[k]) * w.rrhoxfl[k]) * w.kethl[k])
                       * w.rdzq[k]) * w.rdza[k];
            }
        } else {
            w.cimp[k - 1] = -(((((w.rhoxhl[k] * w.rrhoxfl[k]) * w.kethl[k])
                                * dt2) * w.rdzq[k]) * w.rdza[k]);
            const real tbbls = fmaxf(w.bbls[k], w.bbls[k + 1]);
            if (tbbls > GB_LITERAL_2) {
                w.bimp[k - 1] = ((GB_LITERAL_7 - w.aimp[k - 1]) - w.cimp[k - 1])
                    + ((dt2 * sqrtf(w.tke[k])) * rczero) / tbbls;
            } else {
                w.bimp[k - 1] = (GB_LITERAL_7 - w.aimp[k - 1]) - w.cimp[k - 1];
            }
        }
        if (k == kte) {
            w.rimp1[k - 1] = w.tke[k]
                + dt2 * ((w.shear[k] + w.bouyan[k])
                         + (((w.tke[kte + 1] * (w.rhoxhl[k] * w.rrhoxfl[k]))
                             * w.kethl[k]) * w.rdzq[k]) * w.rdza[k]);
        } else {
            w.rimp1[k - 1] = w.tke[k] + dt2 * (w.shear[k] + w.bouyan[k]);
        }
    }
    gb_tridag(w.aimp, w.bimp, w.cimp, w.rimp1, w.uimp1, w.gam, kte - 1);
    for (int k = 2; k <= kte; ++k)
        w.tke[k] = fmaxf(w.uimp1[k - 1], GB_LITERAL_0);

    // ---11--- tendencies for output to the model (:819-836).
    for (int k = 1; k <= kte; ++k) {
        const int kr = kte + 1 - k;
        const real ut = (w.ux[k] - GG(u, kr)) * rdt;
        const real vt = (w.vx[k] - GG(v, kr)) * rdt;
        if (pass == 1) {
            GG(rublten, kr) = ut;
            GG(rvblten, kr) = vt;
            GG(tke_out, kr) = w.tke[k];
            continue;
        }
        GG(rublten, kr) = GB_LITERAL_5 * (ut + GG(rublten, kr));
        GG(rvblten, kr) = GB_LITERAL_5 * (vt + GG(rvblten, kr));
        const real thxs = GG(t, kr) * w.rexnerhl[k];
        const real t2dten = ((w.thx[k] - thxs) * w.exnerhl[k]) * rdt;
        GG(rthblten, kr) = t2dten / GG(pii, kr);
        GG(rqvblten, kr) = (w.qx[k] - GG(qv, kr)) * rdt;
        GG(rqiblten, kr) = (w.qix[k] - GG(qi, kr)) * rdt;
        GG(rqcblten, kr) = (w.qcx[k] - GG(qc, kr)) * rdt;
        GG(exch_m, kr) = w.kzm[k];
        GG(exch_h, kr) = w.kth[k];
        GG(exch_tke, kr) = w.kethl[k];
        GG(el_pbl, kr) = w.bbls[k];
    }
    if (pass == 2) {
        pblh[col] = pblx;
        ust[col] = ustx;
        kpbl[col] = kte + 1 - kpbl2dx;
    }
}

extern "C" __global__
void gbmpbl_column(const real *u, const real *v, const real *t,
                   const real *qv, const real *qc, const real *qi,
                   const real *p, const real *pii, const real *dz8w,
                   const real *rthraten, const real *tke_in,
                   const real *psfc, const real *gz1oz0, const real *psim,
                   const real *xland, const real *hfx, const real *qfx,
                   const real *tsk,
                   real *rublten, real *rvblten, real *rthblten,
                   real *rqvblten, real *rqcblten, real *rqiblten,
                   real *exch_m, real *exch_h, real *exch_tke,
                   real *el_pbl, real *tke_out,
                   real *pblh, real *ust, int *kpbl,
                   real dt, int nz, long long ncol, real *ws, int kp,
                   long long col0) {
    const long long local = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long colll = col0 + local;
    if (colll >= ncol) return;
    const size_t col = (size_t)colll;
    const size_t n = (size_t)ncol;
    real *base = ws + (size_t)blockIdx.x * (size_t)GBWS_SLOTS * (size_t)kp
        * (size_t)GBWS_LANES + (size_t)threadIdx.x;
    const GbWork w = gb_work(base, kp);
    for (int pass = 1; pass <= 2; ++pass)
        gb_gbm2d(pass, col, n, nz, w, u, v, t, qv, qc, qi, p, pii, dz8w,
                 rthraten, tke_in, psfc[col], gz1oz0[col], psim[col],
                 xland[col], hfx[col], qfx[col], tsk[col],
                 rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten,
                 exch_m, exch_h, exch_tke, el_pbl, tke_out,
                 pblh, ust, kpbl, dt);
}

// roots (:1058-1089), retained even though GBM2D never calls it.
__device__ void gb_roots(real a, real b, real c, real &r1, real &r2) {
    if (a == GB_LITERAL_2) {
        r1 = b == GB_LITERAL_2 ? GB_ROOT_FAILURE : -c / b;
        r2 = r1;
    } else if (b == GB_LITERAL_2) {
        r1 = a * c > GB_LITERAL_2 ? GB_ROOT_FAILURE : sqrtf(-c / a);
        r2 = -r1;
    } else if (b * b - (GB_LITERAL_17 * a) * c < GB_LITERAL_2) {
        r1 = GB_ROOT_FAILURE;
        r2 = -r1;
    } else {
        real q = -GB_LITERAL_5 * (b + copysignf(GB_LITERAL_7, b)
                         * sqrtf(b * b - (GB_LITERAL_17 * a) * c));
        r1 = q / a;
        r2 = c / q;
    }
}
extern "C" __global__ void gbmpbl_roots_probe(
        const real *abc, real *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gb_roots(abc[3*i], abc[3*i+1], abc[3*i+2], out[2*i], out[2*i+1]);
}
