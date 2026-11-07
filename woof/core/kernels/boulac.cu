// Bougeault-Lacarrere TKE PBL (WRF bl_pbl_physics = 8), one thread per
// bottom-up gpuwm column.
//
// Transcribed from WRF v4.6.1 phys/module_bl_boulac.F (byte-identical to
// v4.7.1's; sha256 68285f14...d871): BOULAC's column loop with PBL_HEIGHT,
// BOULAC1D (DISSIP_BOUGEAULT, LENGTH_BOUGEAULT, CDTUR_BOUGEAULT,
// TKE_BOUGEAULT with SHEAR and BUOY) and the implicit DIFF/INVERT solves, in
// WRF's operation order.  Index m here is Fortran iz-1 (kts = 1, kte = nz);
// the Fortran's kte+1 slots are named where they are read.
//
// Arithmetic.  Every float32 operation is pinned with FADD/FSUB/FMUL/FDIV/
// FSQRT, so NVRTC cannot contract a multiply-add the gfortran reference
// (-O2 -fno-fast-math -ffp-contract=off) did not make, and the expression
// trees follow Fortran's left-to-right evaluation.  gfortran folds
// x**2. to x*x; every other real exponent here is a powf call, and those use
// WOOF's own float32 pow (gfk_pow): the PBL-height convective velocity
// (**(1./3.)) and the rural drag's wind speed (**.5, NOT sqrt -- gfortran
// keeps it a pow call without -funsafe-math-optimizations).  The t2/w2
// variance profiles are computed by WRF but read only under igamma = 2/3,
// and BOULAC hard-codes igamma = 1, so they are not formed here.  MAX/MIN
// follow gfortran's MAX(a,b) = a if (a > b or a is NaN) else b, MIN the
// mirror. Strict replay disables FTZ and FMA; forecasts permit fast math.
//
// THE ONE DECLARED DIVERGENCE.  WRF's rural lower boundary for momentum is
// the implicit drag a_u(1) = -ust*ust/dz(1)/|V(1)| (module_bl_boulac.F:396-
// 397).  With ust*ust == 0 on a calm lowest level that is 0/0 and the whole
// U and V columns, their tendencies and fluxes go NaN.  The port writes the
// limit of the drag there, -0. (no stress, no drag); every finite WRF word
// is unchanged because the guard fires only where WRF divides zero by zero.
// ust > 0 on a calm level is WRF's -inf drag, kept: the level-1 wind then
// stays exactly zero, which is finite and what WRF computes.
//
// Carried state: TKE (INOUT, floored at temin), DLG_BEP (INOUT under
// flag_bep: WRF writes the urban-blended ground distance back), PBLH
// (rewritten).  EL_PBL is DLK; EXCH_H and EXCH_M are both EXCH on the mass
// levels.  idiff = 1 skips the five tendency solves exactly as WRF does,
// and leaves the tendency and flux outputs untouched.

// Fast forecast arithmetic permits contraction and CUDA libm. Division uses
// the hardware float reciprocal path: the six serial tridiagonal solves
// otherwise pay for correctly rounded division at every recurrence. Exact replay
// opts into the pinned operations and host-compatible pow implementation.
#ifndef GPUWM_WRF_EXACT
#undef FADD
#undef FSUB
#undef FMUL
#undef FDIV
#undef FSQRT
#define FADD(a,b) ((a) + (b))
#define FSUB(a,b) ((a) - (b))
#define FMUL(a,b) ((a) * (b))
#define FDIV(a,b) __fdividef((a), (b))
#define FSQRT(a) sqrtf(a)
#define gfk_pow(a,b) powf((a), (b))
#endif

#ifndef BOULAC_KMAX
#define BOULAC_KMAX 128
#endif

// module_bl_boulac.F:28-29, parameter(ceps_b=1/1.4,ck_b=0.4,temin=0.0001)
#define BL_CEPS_B 0.71428573131561279f
#define BL_CK_B 0.40000000596046448f
#define BL_TEMIN 9.9999997473787516e-05f
// module_bl_boulac.F:189, the fixed reference state th_0 = 300.
#define BL_TH0 300.0f
// module_bl_boulac.F:291, (1./3.) folded to float32
#define BL_THIRD 0.33333334326744080f
// module_bl_boulac.F:978, thv = th*(1.+0.61*q)
#define BL_EP 0.61000001430511475f
// module_bl_boulac.F:986, max(0.01, ...)
#define BL_DTHV_MIN 0.0099999997764825821f
// module_bl_boulac.F:666, :669, :899, exch and dissipation length floors
#define BL_EXCH_MIN 0.10000000149011612f

__device__ __forceinline__ real bl_max(real a, real b)
{
    return (a > b || a != a) ? a : b;
}

__device__ __forceinline__ real bl_min(real a, real b)
{
    return (a < b || a != a) ? a : b;
}

// module_bl_boulac.F:839-874 INVERT on rows 0..nz-1: backward elimination of
// the upper diagonal, forward substitution, then x = c/a2.
__device__ void bl_invert(int nz, const real* a1, real* a2, const real* a3,
                          real* cc, real* x)
{
    for (int m = nz - 2; m >= 0; --m) {
        cc[m] = FSUB(cc[m], FDIV(FMUL(a3[m], cc[m + 1]), a2[m + 1]));
        a2[m] = FSUB(a2[m], FDIV(FMUL(a3[m], a1[m + 1]), a2[m + 1]));
    }
    for (int m = 1; m < nz; ++m)
        cc[m] = FSUB(cc[m], FDIV(FMUL(a1[m], cc[m - 1]), a2[m - 1]));
    for (int m = 0; m < nz; ++m)
        x[m] = FDIV(cc[m], a2[m]);
}

extern "C" __global__ void boulac_column(
    const real* __restrict__ dz_a, const real* __restrict__ u_a,
    const real* __restrict__ v_a, const real* __restrict__ th_a,
    const real* __restrict__ rho_a, const real* __restrict__ qv_a,
    const real* __restrict__ qc_a,
    real* __restrict__ tke_a,
    const real* __restrict__ hfx_a, const real* __restrict__ qfx_a,
    const real* __restrict__ ust_a, const real* __restrict__ frc_a,
    real* __restrict__ pblh_a,
    real* __restrict__ rublten_a, real* __restrict__ rvblten_a,
    real* __restrict__ rthblten_a, real* __restrict__ rqvblten_a,
    real* __restrict__ rqcblten_a,
    real* __restrict__ el_a, real* __restrict__ exch_h_a,
    real* __restrict__ exch_m_a,
    real* __restrict__ wu_a, real* __restrict__ wv_a,
    real* __restrict__ wt_a, real* __restrict__ wq_a,
    const real* __restrict__ a_u_bep, const real* __restrict__ a_v_bep,
    const real* __restrict__ a_t_bep, const real* __restrict__ a_q_bep,
    const real* __restrict__ a_e_bep,
    const real* __restrict__ b_u_bep, const real* __restrict__ b_v_bep,
    const real* __restrict__ b_t_bep, const real* __restrict__ b_q_bep,
    const real* __restrict__ b_e_bep,
    real* __restrict__ dlg_bep, const real* __restrict__ dl_u_bep,
    const real* __restrict__ vl_bep, const real* __restrict__ sf_bep,
    real dt, real cp, real g, int flag_bep, int idiff, int nz, int n)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    if (nz > BOULAC_KMAX || nz < 3) return;
    const size_t st = (size_t)n;
#define BL_AT(a, m) a[(size_t)(m) * st + col]

    real z[BOULAC_KMAX + 1];      // faces, z(kts) = 0
    real dz[BOULAC_KMAX];
    real th[BOULAC_KMAX];
    real rho[BOULAC_KMAX];
    real rhoz[BOULAC_KMAX];       // faces kts..kte (kte+1 is never read)
    real te[BOULAC_KMAX];
    real dls[BOULAC_KMAX];
    real exch[BOULAC_KMAX];       // faces kts..kte; exch(kte+1) = 0.1
    real gam[BOULAC_KMAX];        // faces kts..kte; gamma(kte+1) = 0
    real cddz[BOULAC_KMAX];       // faces kts..kte (kte+1 is never read)
    real a1[BOULAC_KMAX], a2b[BOULAC_KMAX], a2[BOULAC_KMAX];
    real a3[BOULAC_KMAX], cc[BOULAC_KMAX], co[BOULAC_KMAX];

    // ---- BOULAC :241-260, the 1-D copies and the face densities --------
    z[0] = 0.0f;
    for (int m = 0; m < nz; ++m) {
        dz[m] = BL_AT(dz_a, m);
        th[m] = BL_AT(th_a, m);
        rho[m] = BL_AT(rho_a, m);
        te[m] = BL_AT(tke_a, m);
        z[m + 1] = FADD(z[m], dz[m]);
    }
    rhoz[0] = rho[0];
    for (int m = 1; m < nz; ++m)
        rhoz[m] = FDIV(FADD(FMUL(rho[m], dz[m - 1]), FMUL(rho[m - 1], dz[m])),
                       FADD(dz[m - 1], dz[m]));

    const real frc = flag_bep ? frc_a[col] : 0.0f;
    const real one_m_frc = FSUB(1.0f, frc);

    // ---- PBL_HEIGHT :958-992 --------------------------------------------
    real pblh = 0.0f;
    {
        const real thv0 = FMUL(th[0], FADD(1.0f, FMUL(BL_EP, BL_AT(qv_a, 0))));
        const real thsfc = FADD(thv0, 0.5f);
        real thv_prev = thv0;
        real zc_prev = FADD(z[0], FDIV(dz[0], 2.0f));
        for (int m = 1; m < nz; ++m) {
            const real thv = FMUL(th[m], FADD(1.0f, FMUL(BL_EP, BL_AT(qv_a, m))));
            const real zc = FADD(z[m], FDIV(dz[m], 2.0f));
            if (pblh == 0.0f && thv > thsfc) {
                pblh = FADD(zc_prev,
                            FMUL(FDIV(FSUB(thsfc, thv_prev),
                                      bl_max(BL_DTHV_MIN, FSUB(thv, thv_prev))),
                                 FSUB(zc, zc_prev)));
            }
            thv_prev = thv;
            zc_prev = zc;
        }
    }
    pblh_a[col] = pblh;

    // ---- wstar and the Troen-Mahrt counter-gradient term :290-318 -------
    const real hfx = hfx_a[col];
    const real wts = bl_max(0.0f, FDIV(FDIV(hfx, rho[0]), cp));
    const real wstar = gfk_pow(FDIV(FMUL(FMUL(g, wts), pblh), BL_TH0), BL_THIRD);
    gam[0] = 0.0f;
    for (int m = 1; m < nz; ++m) {
        if (z[m] <= FMUL(1.0f, pblh) && wts > 0.0f)
            gam[m] = FDIV(FDIV(FMUL(10.0f, wts), wstar), pblh);
        else
            gam[m] = 0.0f;
    }

    // ---- BOULAC1D :457-551 ----------------------------------------------
    // DISSIP_BOUGEAULT :556-619 and LENGTH_BOUGEAULT :623-642, per level:
    // the up/down parcel lengths, then their dissipation (geometric) and
    // mixing (minimum) combinations with the ground distance and, under BEP,
    // the building length scale.
    const real beta = FDIV(g, BL_TH0);
    for (int iz = 0; iz < nz; ++iz) {
        const real tei = te[iz];
        const real thi = th[iz];
        // upward
        real zup = 0.0f, zup_inf = 0.0f, zzz = 0.0f;
        real dlu = FSUB(FSUB(z[nz], z[iz]), FDIV(dz[iz], 2.0f));
        for (int izz = iz; izz <= nz - 2; ++izz) {
            const real dzt = FDIV(FADD(dz[izz + 1], dz[izz]), 2.0f);
            zup = FSUB(zup, FMUL(FMUL(beta, thi), dzt));
            zup = FADD(zup, FDIV(FMUL(FMUL(beta, FADD(th[izz + 1], th[izz])),
                                      dzt), 2.0f));
            zzz = FADD(zzz, dzt);
            if (tei < zup && tei >= zup_inf) {
                const real bbb = FDIV(FSUB(th[izz + 1], th[izz]), dzt);
                real tl;
                if (bbb != 0.0f) {
                    const real bd = FMUL(beta, FSUB(th[izz], thi));
                    const real disc = FADD(FMUL(bd, bd),
                                           FMUL(FMUL(FMUL(2.0f, bbb), beta),
                                                FSUB(tei, zup_inf)));
                    tl = FDIV(FDIV(FADD(-bd, FSQRT(bl_max(0.0f, disc))), bbb),
                              beta);
                } else if (th[izz] != thi) {
                    tl = FDIV(FSUB(tei, zup_inf),
                              FMUL(beta, FSUB(th[izz], thi)));
                } else {
                    tl = 0.0f;
                }
                dlu = FADD(FSUB(zzz, dzt), tl);
            }
            zup_inf = zup;
        }
        // downward
        real zdo = 0.0f, zdo_sup = 0.0f;
        zzz = 0.0f;
        real dld = FADD(z[iz], FDIV(dz[iz], 2.0f));
        for (int izz = iz; izz >= 1; --izz) {
            const real dzt = FDIV(FADD(dz[izz - 1], dz[izz]), 2.0f);
            zdo = FADD(zdo, FMUL(FMUL(beta, thi), dzt));
            zdo = FSUB(zdo, FDIV(FMUL(FMUL(beta, FADD(th[izz - 1], th[izz])),
                                      dzt), 2.0f));
            zzz = FADD(zzz, dzt);
            if (tei < zdo && tei >= zdo_sup) {
                const real bbb = FDIV(FSUB(th[izz], th[izz - 1]), dzt);
                real tl;
                if (bbb != 0.0f) {
                    const real bd = FMUL(beta, FSUB(th[izz], thi));
                    const real disc = FADD(FMUL(bd, bd),
                                           FMUL(FMUL(FMUL(2.0f, bbb), beta),
                                                FSUB(tei, zdo_sup)));
                    tl = FDIV(FDIV(FADD(bd, FSQRT(bl_max(0.0f, disc))), bbb),
                              beta);
                } else if (th[izz] != thi) {
                    tl = FDIV(FSUB(tei, zdo_sup),
                              FMUL(beta, FSUB(th[izz], thi)));
                } else {
                    tl = 0.0f;
                }
                dld = FADD(FSUB(zzz, dzt), tl);
            }
            zdo_sup = zdo;
        }
        // BOULAC :261-284, the ground distance and building length scale
        const real zmid = FDIV(FADD(z[iz], z[iz + 1]), 2.0f);
        real dlg, dl_u;
        if (flag_bep) {
            dlg = FADD(FMUL(zmid, one_m_frc), FMUL(BL_AT(dlg_bep, iz), frc));
            dl_u = BL_AT(dl_u_bep, iz);
            if (one_m_frc < 1.0f) dl_u = FDIV(dl_u, frc);
            BL_AT(dlg_bep, iz) = dlg;      // :357-359 write-back
        } else {
            dlg = zmid;
            dl_u = 0.0f;
        }
        // LENGTH_BOUGEAULT
        dld = bl_min(dld, dlg);
        real dlsi = FSQRT(FMUL(dlu, dld));
        real dlki = bl_min(dlu, dld);
        if (dl_u > 0.0f) {
            dlsi = FDIV(1.0f, FADD(FDIV(1.0f, dlsi), FDIV(1.0f, dl_u)));
            dlki = FDIV(1.0f, FADD(FDIV(1.0f, dlki), FDIV(1.0f, dl_u)));
        }
        dls[iz] = dlsi;
        co[iz] = dlki;                     // DLK, held until CDTUR reads it
        BL_AT(el_a, iz) = dlki;
    }

    // CDTUR_BOUGEAULT :648-672
    exch[0] = 0.0f;
    for (int m = 1; m < nz; ++m) {
        const real wsum = FADD(dz[m], dz[m - 1]);
        const real te_m = FDIV(FADD(FMUL(te[m - 1], dz[m]),
                                    FMUL(te[m], dz[m - 1])), wsum);
        const real dlk_m = FDIV(FADD(FMUL(co[m - 1], dz[m]),
                                     FMUL(co[m], dz[m - 1])), wsum);
        exch[m] = bl_max(FMUL(FMUL(BL_CK_B, dlk_m), FSQRT(te_m)), BL_EXCH_MIN);
    }
    for (int m = 0; m < nz; ++m) {
        BL_AT(exch_h_a, m) = exch[m];
        BL_AT(exch_m_a, m) = exch[m];
    }

    // DIFF :716-744, the matrix every solve of this column shares: the face
    // conductances and the operator rows without their source terms.  The
    // top row (kte) is the identity (izf = 1).
    cddz[0] = FDIV(FMUL(FMUL(flag_bep ? BL_AT(sf_bep, 0) : 1.0f, rhoz[0]),
                        exch[0]), dz[0]);
    for (int m = 1; m < nz; ++m) {
        const real sfm = flag_bep ? BL_AT(sf_bep, m) : 1.0f;
        cddz[m] = FDIV(FMUL(FMUL(FMUL(2.0f, sfm), rhoz[m]), exch[m]),
                       FADD(dz[m], dz[m - 1]));
    }
    for (int m = 0; m <= nz - 2; ++m) {
        const real vlm = flag_bep ? BL_AT(vl_bep, m) : 1.0f;
        const real dzv = FMUL(vlm, dz[m]);
        a1[m] = -FDIV(FDIV(FMUL(cddz[m], dt), dzv), rho[m]);
        a2b[m] = FADD(1.0f, FDIV(FDIV(FMUL(dt, FADD(cddz[m], cddz[m + 1])),
                                      dzv), rho[m]));
        a3[m] = -FDIV(FDIV(FMUL(cddz[m + 1], dt), dzv), rho[m]);
    }
    a1[nz - 1] = 0.0f;
    a2b[nz - 1] = 1.0f;
    a3[nz - 1] = 0.0f;

    // TKE_BOUGEAULT :879-909 with SHEAR :805-834 and BUOY :775-800, built
    // straight into the TKE rows: a_e = a_e_bep + td, b_e = b_e_bep + sh + bu
    // (sh and bu scaled by sf; zero at kts and kte).
    for (int m = 0; m <= nz - 2; ++m) {
        real sh = 0.0f, bu = 0.0f;
        if (m >= 1) {
            const real dzm1 = dz[m - 1], dz0 = dz[m], dzp1 = dz[m + 1];
            const real u0 = BL_AT(u_a, m), um = BL_AT(u_a, m - 1),
                       up = BL_AT(u_a, m + 1);
            const real v0 = BL_AT(v_a, m), vm = BL_AT(v_a, m - 1),
                       vp = BL_AT(v_a, m + 1);
            const real u2 = FDIV(FADD(FMUL(dzp1, u0), FMUL(dz0, up)),
                                 FADD(dz0, dzp1));
            const real u1 = FDIV(FADD(FMUL(dz0, um), FMUL(dzm1, u0)),
                                 FADD(dzm1, dz0));
            const real v2 = FDIV(FADD(FMUL(dzp1, v0), FMUL(dz0, vp)),
                                 FADD(dz0, dzp1));
            const real v1 = FDIV(FADD(FMUL(dz0, vm), FMUL(dzm1, v0)),
                                 FADD(dzm1, dz0));
            const real cdm_s = FMUL(0.5f, FADD(exch[m], exch[m + 1]));
            const real du = FDIV(FSUB(u2, u1), dz0);
            const real dv = FDIV(FSUB(v2, v1), dz0);
            sh = FMUL(cdm_s, FADD(FMUL(du, du), FMUL(dv, dv)));

            const real dtdz1 = FDIV(FMUL(2.0f, FSUB(th[m], th[m - 1])),
                                    FADD(dzm1, dz0));
            const real dtdz2 = FDIV(FMUL(2.0f, FSUB(th[m + 1], th[m])),
                                    FADD(dzp1, dz0));
            const real dtmdz = FMUL(0.5f, FADD(dtdz1, dtdz2));
            const real cdm_b = FMUL(0.5f, FADD(exch[m + 1], exch[m]));
            const real gammam = FMUL(0.5f, FADD(gam[m + 1], gam[m]));
            bu = -FDIV(FMUL(FMUL(cdm_b, FSUB(dtmdz, gammam)), g), BL_TH0);
        }
        const real te1 = bl_max(te[m], BL_TEMIN);
        const real dl1 = bl_max(dls[m], BL_EXCH_MIN);
        const real td = -FDIV(FMUL(BL_CEPS_B, FSQRT(te1)), dl1);
        const real sfm = flag_bep ? BL_AT(sf_bep, m) : 1.0f;
        sh = FMUL(sh, sfm);
        bu = FMUL(bu, sfm);
        const real ae = FADD(flag_bep ? BL_AT(a_e_bep, m) : 0.0f, td);
        const real be = FADD(FADD(flag_bep ? BL_AT(b_e_bep, m) : 0.0f, sh), bu);
        a2[m] = FSUB(a2b[m], FMUL(ae, dt));
        cc[m] = FADD(te[m], FMUL(be, dt));
    }
    a2[nz - 1] = 1.0f;
    cc[nz - 1] = te[nz - 1];
    bl_invert(nz, a1, a2, a3, cc, te);
    // BOULAC1D :546-548, the TKE floor
    for (int m = 0; m < nz; ++m) {
        real t = te[m];
        if (t < BL_TEMIN) t = BL_TEMIN;
        BL_AT(tke_a, m) = t;
    }

    if (idiff == 1) return;

    // ---- the five tendency solves :368-442 ------------------------------
    const real dz1 = dz[0], rho1 = rho[0];
    // rural lower boundary :394-397 (the declared 0/0 guard for a_u, a_v)
    real a_uv1 = 0.0f, b_t1 = 0.0f, b_q1 = 0.0f;
    if (!flag_bep) {
        b_t1 = FDIV(FDIV(FDIV(hfx, dz1), rho1), cp);
        b_q1 = FDIV(FDIV(qfx_a[col], dz1), rho1);
        const real ust = ust_a[col];
        const real stress = FMUL(ust, ust);
        if (stress == 0.0f) {
            a_uv1 = -0.0f;
        } else {
            const real u1v = BL_AT(u_a, 0), v1v = BL_AT(v_a, 0);
            const real speed = gfk_pow(FADD(FMUL(u1v, u1v), FMUL(v1v, v1v)),
                                       0.5f);
            a_uv1 = -FDIV(FDIV(stress, dz1), speed);
        }
    }

    for (int var = 0; var < 5; ++var) {
        const real* x_a;
        const real* aa_bep = 0;
        const real* bb_bep = 0;
        real* ten_a;
        real* flux_a = 0;
        switch (var) {
        case 0: x_a = u_a; aa_bep = a_u_bep; bb_bep = b_u_bep;
                ten_a = rublten_a; flux_a = wu_a; break;
        case 1: x_a = v_a; aa_bep = a_v_bep; bb_bep = b_v_bep;
                ten_a = rvblten_a; flux_a = wv_a; break;
        case 2: x_a = th_a; aa_bep = a_t_bep; bb_bep = b_t_bep;
                ten_a = rthblten_a; flux_a = wt_a; break;
        case 3: x_a = qv_a; aa_bep = a_q_bep; bb_bep = b_q_bep;
                ten_a = rqvblten_a; flux_a = wq_a; break;
        default: x_a = qc_a; ten_a = rqcblten_a; break;
        }
        for (int m = 0; m <= nz - 2; ++m) {
            real aa = 0.0f, bb = 0.0f;
            if (var < 4) {
                if (flag_bep) {
                    aa = BL_AT(aa_bep, m);
                    bb = BL_AT(bb_bep, m);
                } else if (m == 0) {
                    aa = (var < 2) ? a_uv1 : 0.0f;
                    bb = (var == 2) ? b_t1 : (var == 3) ? b_q1 : 0.0f;
                }
            }
            // :404-408, the counter-gradient flux divergence joins b_t
            if (var == 2 && m >= 1 && z[m] <= FMUL(1.0f, pblh) && wts > 0.0f) {
                const real exch_up = (m + 1 < nz) ? exch[m + 1] : BL_EXCH_MIN;
                const real gam_up = (m + 1 < nz) ? gam[m + 1] : 0.0f;
                bb = FSUB(bb, FDIV(FSUB(FMUL(exch_up, gam_up),
                                        FMUL(exch[m], gam[m])), dz[m]));
            }
            const real xm = BL_AT(x_a, m);
            co[m] = xm;
            a2[m] = FSUB(a2b[m], FMUL(aa, dt));
            cc[m] = FADD(xm, FMUL(bb, dt));
        }
        co[nz - 1] = BL_AT(x_a, nz - 1);
        a2[nz - 1] = 1.0f;
        cc[nz - 1] = co[nz - 1];
        bl_invert(nz, a1, a2, a3, cc, co);
        for (int m = 0; m < nz; ++m)
            BL_AT(ten_a, m) = FDIV(FSUB(co[m], BL_AT(x_a, m)), dt);
        if (flux_a) {
            BL_AT(flux_a, 0) = 0.0f;
            for (int m = 1; m < nz; ++m)
                BL_AT(flux_a, m) = FDIV(-FMUL(cddz[m], FSUB(co[m], co[m - 1])),
                                        rho[m]);
        }
    }
#undef BL_AT
}
