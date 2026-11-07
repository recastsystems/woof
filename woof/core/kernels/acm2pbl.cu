/* ACM2 (Pleim) PBL, bl_pbl_physics = 7: WRF v4.6.1 phys/module_bl_acm.F.
 *
 * One CUDA thread per mass column.  Transcribed in WRF operation order:
 *
 *   ACM2D   :245-559  micromet scaling, PBL height, NOCONV, tendencies
 *   EDDYX   :641-811  boundary-scaling and local Kz
 *   ACM     :817-1162 scalar mixing (theta, qv, qc, qi), sub-stepped
 *   ACMM    :1166-1461 momentum mixing (u, v), sub-stepped
 *   MATRIX  :1466-1558 bordered band solver (free-convection columns)
 *   TRI     :1563-1627 Thomas solver (every other column)
 *
 * ACMINIT (:565-635) only zeroes the tendency arrays on a cold start; the
 * engine's tendency buffers are born zero, so it has no device half.  WRF's
 * WRF_CHEM arms are not compiled in WRF's own non-chem build and are not
 * here.
 *
 * STRICT ARITHMETIC. Every operation is spelled as a rounded intrinsic
 * (FADD/FSUB/FMUL/FDIV/FSQRT from glibc_flt32.cuh) in the association
 * gfortran gives the Fortran expression: left to right for operators of
 * equal precedence, unary minus over the whole product term.  The module
 * compiles with --fmad=false and --ftz=false (gpuwm/core/acm2pbl.py), and
 * the intrinsics hold the order even where a build would contract. Fast
 * numerics redefines these macros as ordinary expressions before this
 * file and allows contraction. Both modes preserve subnormal words.
 *   * Integer powers are gfortran's multiply chains: x**2 = x*x,
 *     x**3 = (x*x)*x, x**4 = (x*x)*(x*x).
 *   * (1-ZF/PBL)**P with the PARAMETER P = 2.0 is folded by gfortran -O2 to
 *     t*t (tools/acm2_wrf461_oracle/build.sh counts the five powf call
 *     sites that remain in WRF's object).
 *   * The real exponents 0.333333, 0.33333, -0.25 and 0.3333 go through
 *     WOOF's own float32 powf, gfk_pow.
 *
 * WRF'S FATAL.  When the bulk Richardson number never reaches RIC the
 * scheme calls wrf_error_fatal (:464-470) and WRF stops.  The kernel sets
 * fatal bit in `status` and returns. Standalone calls name the column;
 * model calls accumulate status on the device until the next ledger drain.
 * A column whose virtual potential temperature stays below the
 * source-layer value up to the model top (KMIX = kte) always reaches that
 * fatal when that level is above the source layer, and WRF reads
 * THETAV(kte+1) past its array on the way (:439); the kernel stops there
 * instead of reading past the column.
 *
 * DECLARED DIVERGENCE (WRF reads past its arrays, result undefined).  When
 * the PBL top lands in the top layer of a free-convection column
 * (KCBL = KLPBL = kte), ACM/ACMM read DZH(kte+1) and VCI(kte+1) and write
 * MDWN(kte+1) one element past arrays dimensioned kts:kte (:978, :1081,
 * :1300, :1392).  The written value is 0 and the read terms are multiplied
 * by it; the kernel carries MDWN(kte+1) = DZH(kte+1) = VCI(kte+1) = 0, the
 * defined value of that product for any finite word WRF's memory held.
 */

/* COLUMN STORAGE.  Every column array lives in a global workspace the
 * launcher sizes to the columns in flight (gpuwm.core.physics_inventory.
 * acm2_workspace_floats), interleaved by lane within a block the way CUDA
 * lays local memory across a warp: slot s, level k of lane l sits at
 * ((block*ACM2WS_SLOTS + s)*(kte+2) + k)*ACM2_BLOCK + l.  The per-thread
 * local frame therefore holds scalars only, and the arrays are 1-based
 * like WRF's with room for the kte+1 element (see DECLARED DIVERGENCE). */
#ifndef ACM2_BLOCK
#define ACM2_BLOCK 32
#endif
#ifndef ACM2WS_SLOTS
#define ACM2WS_SLOTS 27
#endif

struct AcmCol {
    float* p;
    __device__ __forceinline__ float& operator[](int k) const {
        return p[(long long)k * ACM2_BLOCK];
    }
};

/* module_bl_acm.F parameters. */

/* status bits */
#define ACM2_STATUS_RIB_NEVER_EXCEEDS_RIC 1u

/* --------------------------------------------------------------------- */
/* TRI (:1563-1627).  l = sub, d = diagonal, u = super, b = rhs, x = out. */
template <int NSP>
__device__ void acm2_tri(AcmCol l, AcmCol d, AcmCol u, const AcmCol* b,
                         const AcmCol* x, AcmCol gam, int kl)
{
    float bet = FDIV(1.0f, d[1]);
    for (int v = 0; v < NSP; ++v)
        x[v][1] = FMUL(bet, b[v][1]);
    for (int k = 2; k <= kl; ++k) {
        gam[k] = FMUL(bet, u[k - 1]);
        bet = FDIV(1.0f, FSUB(d[k], FMUL(l[k], gam[k])));
        for (int v = 0; v < NSP; ++v)
            x[v][k] = FMUL(bet, FSUB(b[v][k], FMUL(l[k], x[v][k - 1])));
    }
    for (int k = kl - 1; k >= 1; --k)
        for (int v = 0; v < NSP; ++v)
            x[v][k] = FSUB(x[v][k], FMUL(gam[k + 1], x[v][k + 1]));
}

/* MATRIX (:1466-1558).  WRF forms the dense lower factor L(KL,KL) first,
 * then UII/RUII, then forward and back substitution.  Every entry of L is a
 * pure function of a, b, c, e, so the kernel forms row I of L while it
 * forward-substitutes row I, in WRF's J order; the values are WRF's words.
 * den[j] is B(J) - L(J,J-1)*E(J-1), the denominator WRF evaluates inside
 * the L recurrence and again as UII(J).  d is overwritten by y (row I of
 * d is read only to start row I of y). */
template <int NSP>
__device__ void acm2_matrix(AcmCol a, AcmCol b, AcmCol c, const AcmCol* d,
                            AcmCol e, const AcmCol* x, AcmCol den,
                            AcmCol ruii, int kl)
{
    ruii[1] = FDIV(1.0f, b[1]);
    for (int i = 2; i <= kl; ++i) {
        float lij = FDIV(a[i], b[1]);                 /* L(I,1) */
        float sum[NSP];
#pragma unroll
        for (int v = 0; v < NSP; ++v)
            sum[v] = FSUB(d[v][i], FMUL(lij, d[v][1]));
        for (int j = 2; j <= i - 1; ++j) {
            const float aij = (i == j + 1) ? c[i] : 0.0f;
            lij = FDIV(FSUB(aij, FMUL(lij, e[j - 1])), den[j]);
#pragma unroll
            for (int v = 0; v < NSP; ++v)
                sum[v] = FSUB(sum[v], FMUL(lij, d[v][j]));
        }
#pragma unroll
        for (int v = 0; v < NSP; ++v)
            d[v][i] = sum[v];                         /* Y(V,I) */
        /* lij is now L(I,I-1) */
        den[i] = FSUB(b[i], FMUL(lij, e[i - 1]));     /* UII(I) */
        ruii[i] = FDIV(1.0f, den[i]);
    }
    for (int v = 0; v < NSP; ++v)
        x[v][kl] = FMUL(d[v][kl], ruii[kl]);
    for (int i = kl - 1; i >= 1; --i)
        for (int v = 0; v < NSP; ++v)
            x[v][i] = FMUL(FSUB(d[v][i], FMUL(e[i], x[v][i + 1])), ruii[i]);
}

#ifdef ACM2_FAST
/* MATRIX's original system has a tridiagonal tail (rows 2..kl) and
 * one extra first column A. Solve that tail for Y and Z:
 * T Y = D_tail, T Z = -A_tail, then X_tail = Y + Z X(1).
 * The first row gives X(1) = (D(1)-E(1)Y(2))/(B(1)+E(1)Z(2)).
 * This is the same implicit operator, with O(kl*NSP) work instead of
 * explicitly eliminating its dense lower factor. The strict path above
 * preserves WRF's rounding order. No coefficient or substep is omitted.
 * For row 2, A(2) already includes its subdiagonal diffusion contribution;
 * C(2) is not a second term (as in WRF MATRIX's L(2,1)=A(2)/B(1)). */
template <int NSP>
__device__ void acm2_bordered(AcmCol a, AcmCol b, AcmCol c, const AcmCol* d,
                            AcmCol e, const AcmCol* x, AcmCol z,
                            AcmCol inv, int kl)
{
    inv[2] = 1.0f / b[2];
    z[2] = -a[2] * inv[2];
    for (int v = 0; v < NSP; ++v) x[v][2] = d[v][2] * inv[2];
    for (int k = 3; k <= kl; ++k) {
        const float gamma = inv[k-1] * e[k-1];
        inv[k] = 1.0f / (b[k] - c[k] * gamma);
        z[k] = (-a[k] - c[k] * z[k-1]) * inv[k];
        for (int v = 0; v < NSP; ++v)
            x[v][k] = (d[v][k] - c[k] * x[v][k-1]) * inv[k];
    }
    for (int k = kl-1; k >= 2; --k) {
        const float gamma = inv[k] * e[k];
        z[k] -= gamma * z[k+1];
        for (int v = 0; v < NSP; ++v) x[v][k] -= gamma * x[v][k+1];
    }
    const float first_inv = 1.0f / (b[1] + e[1] * z[2]);
    for (int v = 0; v < NSP; ++v) {
        const float first = (d[v][1] - e[1] * x[v][2]) * first_inv;
        x[v][1] = first;
        for (int k = 2; k <= kl; ++k) x[v][k] += z[k] * first;
    }
}
#endif

/* ACM (:817-1162) when MOM is false, ACMM (:1166-1461) when true.  The two
 * are the same algorithm on different species; the only arithmetic that
 * differs is the order of the factors in BI(1)'s eddy term (:1064-1067
 * against :1375-1378) and the surface fluxes fs, which the caller forms.
 * eddy is EDDYZ (ACM) or EDDYZM (ACMM) and is scaled in place for
 * K < KCBL on free-convection columns, as WRF does; vci is the species
 * column and becomes the mixed column. */
template <int NSP, bool MOM>
__device__ void acm2_mix(float dtpbl, int noconv, int klpbl, float pbl,
                         float mol, AcmCol zf, AcmCol dzh, AcmCol dzhi,
                         AcmCol dzfi, AcmCol eddy, const float* fs,
                         const AcmCol* vci, const AcmCol* di, AcmCol mbarks,
                         AcmCol mdwn, AcmCol ai, AcmCol bi, AcmCol ci,
                         AcmCol ei, AcmCol xplus, AcmCol xminus, AcmCol w1,
                         AcmCol w2, int kl)
{
    const int klm = kl - 1;
    float dtlim = dtpbl;
    int kcbl = 1;
    float fsacm = 0.0f;

    if (noconv == 1) {
        kcbl = klpbl;
        const float hovl = -FDIV(pbl, mol);
        fsacm = FDIV(1.0f,
                     FADD(1.0f, FDIV(gfk_pow(FDIV(ACM_KARMAN, hovl), 0.3333f),
                                     FMUL(0.72f, ACM_KARMAN))));
        const float meddy = FDIV(FMUL(eddy[1], dzfi[1]), FSUB(pbl, zf[1]));
        const float mbar = FMUL(meddy, fsacm);
        for (int k = 1; k <= kcbl - 1; ++k)
            eddy[k] = FMUL(eddy[k], FSUB(1.0f, fsacm));
        for (int k = 2; k <= kcbl; ++k) {
            mbarks[k] = mbar;
            mdwn[k] = FMUL(FMUL(mbar, FSUB(pbl, zf[k - 1])), dzhi[k]);
        }
        mbarks[1] = mbar;
        mbarks[kcbl] = mdwn[kcbl];
        mdwn[kcbl + 1] = 0.0f;
    }

    for (int k = 1; k <= klm; ++k) {
        const float ekz = FMUL(FMUL(eddy[k], dzfi[k]), dzhi[k]);
        dtlim = fminf(FDIV(0.75f, ekz), dtlim);
    }
    if (noconv == 1) {
        const float rz = FMUL(FSUB(zf[kcbl], zf[1]), dzhi[1]);
        dtlim = fminf(FDIV(ACM_XX, FMUL(mbarks[1], rz)), dtlim);
    }

    const int nlp = (int)FADD(FDIV(dtpbl, dtlim), 1.0f);
    const float dts = FDIV(dtpbl, (float)nlp);
    const float omc = 1.0f - ACM_CRANKP;              /* (1.0 - CRANKP) */

    // The coefficients depend only on the frozen eddy profile and substep
    // length. WRF recomputes identical words on each substep; form them once.
    {
        for (int k = 1; k <= kl; ++k) {
            ai[k] = 0.0f; bi[k] = 0.0f; ci[k] = 0.0f; ei[k] = 0.0f;
        }
        for (int k = 2; k <= kcbl; ++k) {
            ei[k - 1] = -FMUL(FMUL(FMUL(FMUL(ACM_CRANKP, mdwn[k]), dts),
                                   dzh[k]), dzhi[k - 1]);
            bi[k] = FADD(1.0f, FMUL(FMUL(ACM_CRANKP, mdwn[k]), dts));
            ai[k] = -FMUL(FMUL(ACM_CRANKP, mbarks[k]), dts);
        }
        ei[1] = FSUB(ei[1], FMUL(FMUL(FMUL(FMUL(eddy[1], ACM_CRANKP),
                                           dzhi[1]), dzfi[1]), dts));
        ai[2] = FSUB(ai[2], FMUL(FMUL(FMUL(FMUL(eddy[1], ACM_CRANKP),
                                           dzhi[2]), dzfi[1]), dts));
        for (int k = kcbl + 1; k <= kl; ++k)
            bi[k] = 1.0f;
        for (int k = 2; k <= kl; ++k) {
            xplus[k] = FMUL(FMUL(FMUL(eddy[k], dzhi[k]), dzfi[k]), dts);
            xminus[k] = FMUL(FMUL(FMUL(eddy[k - 1], dzhi[k]), dzfi[k - 1]),
                             dts);
            ci[k] = -FMUL(xminus[k], ACM_CRANKP);
            ei[k] = FSUB(ei[k], FMUL(xplus[k], ACM_CRANKP));
            bi[k] = FADD(FADD(bi[k], FMUL(xplus[k], ACM_CRANKP)),
                         FMUL(xminus[k], ACM_CRANKP));
        }
        /* the eddy term of BI(1): ACM :1065/:1067 multiplies
         * EDDYZ*CRANKP*DZHI*DZFI*DTS, ACMM :1376/:1378
         * EDDYZM*DZHI*DZFI*CRANKP*DTS */
        const float eterm = MOM
            ? FMUL(FMUL(FMUL(FMUL(eddy[1], dzhi[1]), dzfi[1]), ACM_CRANKP), dts)
            : FMUL(FMUL(FMUL(FMUL(eddy[1], ACM_CRANKP), dzhi[1]), dzfi[1]), dts);
        if (noconv == 1) {
            bi[1] = FADD(FADD(1.0f,
                              FMUL(FMUL(FMUL(FMUL(ACM_CRANKP, mbarks[1]),
                                             FSUB(pbl, zf[1])), dts),
                                   dzhi[1])),
                         eterm);
        } else {
            bi[1] = FADD(1.0f, eterm);
        }

    }
    for (int nl = 1; nl <= nlp; ++nl) {
        for (int k = 1; k <= kl; ++k)
            for (int l = 0; l < NSP; ++l)
                di[l][k] = 0.0f;

        for (int k = 2; k <= kcbl; ++k) {
            for (int l = 0; l < NSP; ++l) {
                const float delc = FMUL(dts,
                    FADD(FSUB(FMUL(mbarks[k], vci[l][1]),
                              FMUL(mdwn[k], vci[l][k])),
                         FMUL(FMUL(FMUL(dzh[k + 1], dzhi[k]), mdwn[k + 1]),
                              vci[l][k + 1])));
                di[l][k] = FADD(vci[l][k], FMUL(omc, delc));
            }
        }
        for (int k = kcbl + 1; k <= kl; ++k)
            for (int l = 0; l < NSP; ++l)
                di[l][k] = vci[l][k];

        for (int k = 2; k <= kl; ++k) {
            if (k == kl) {
                for (int l = 0; l < NSP; ++l)
                    di[l][k] = FSUB(di[l][k],
                                    FMUL(FMUL(omc, xminus[k]),
                                         FSUB(vci[l][k], vci[l][k - 1])));
            } else {
                for (int l = 0; l < NSP; ++l)
                    di[l][k] = FSUB(
                        FADD(di[l][k], FMUL(FMUL(omc, xplus[k]),
                                            FSUB(vci[l][k + 1], vci[l][k]))),
                        FMUL(FMUL(omc, xminus[k]),
                             FSUB(vci[l][k], vci[l][k - 1])));
            }
        }

        if (noconv == 1) {
            for (int l = 0; l < NSP; ++l) {
                const float inner = FSUB(
                    FMUL(FMUL(mbarks[1], FSUB(pbl, zf[1])), vci[l][1]),
                    FMUL(FMUL(mdwn[2], vci[l][2]), dzh[2]));
                di[l][1] = FADD(vci[l][1],
                                FMUL(FMUL(FSUB(fs[l], FMUL(omc, inner)),
                                          dzhi[1]), dts));
            }
        } else {
            for (int l = 0; l < NSP; ++l)
                di[l][1] = FADD(vci[l][1], FMUL(FMUL(fs[l], dzhi[1]), dts));
        }
        for (int l = 0; l < NSP; ++l)
            di[l][1] = FADD(di[l][1],
                            FMUL(FMUL(FMUL(FMUL(FMUL(omc, eddy[1]), dzhi[1]),
                                           dzfi[1]), dts),
                                 FSUB(vci[l][2], vci[l][1])));

        if (noconv == 1)
#ifdef ACM2_FAST
            acm2_bordered<NSP>(ai, bi, ci, di, ei, vci, w1, w2, kl);
#else
            acm2_matrix<NSP>(ai, bi, ci, di, ei, vci, w1, w2, kl);
#endif
        else
            acm2_tri<NSP>(ci, bi, ei, di, vci, w1, kl);
    }
}

extern "C" __global__ void acm2_pbl_column(
    const float* __restrict__ us, const float* __restrict__ vs,
    const float* __restrict__ theta, const float* __restrict__ tt,
    const float* __restrict__ qvs, const float* __restrict__ qcs,
    const float* __restrict__ qis, const float* __restrict__ densx,
    const float* __restrict__ dzf,
    const float* __restrict__ ust_in, const float* __restrict__ hfx,
    const float* __restrict__ qfx, const float* __restrict__ wspd,
    float* __restrict__ pblh, float* __restrict__ regime,
    float* __restrict__ rmol, int* __restrict__ kpbl,
    float* __restrict__ exch_h, float* __restrict__ exch_m,
    float* __restrict__ utnp, float* __restrict__ vtnp,
    float* __restrict__ ttnp, float* __restrict__ qvtnp,
    float* __restrict__ qctnp, float* __restrict__ qitnp,
    unsigned int* __restrict__ status, int* __restrict__ first_fatal,
    float* __restrict__ ws, int col0, int ncol, int kte, float dtpbl,
    int xtime, float ep1, float g, float rd, float cpd)
{
    const int lane = threadIdx.x;
    const int col = col0 + blockIdx.x * ACM2_BLOCK + lane;
    if (col >= ncol) return;
    const long long st = ncol;
#define Q(k) ((long long)((k) - 1) * st + col)

    const long long np = kte + 2;
    float* const base = ws + (long long)blockIdx.x * ACM2WS_SLOTS * np
                        * ACM2_BLOCK + lane;
#define SLOT(s) AcmCol{base + (long long)(s) * np * ACM2_BLOCK}
    const AcmCol zf = SLOT(0), za = SLOT(1), dzh = SLOT(2), dzhi = SLOT(3),
        dzfi = SLOT(4), thetav = SLOT(5), rib = SLOT(6), eddyz = SLOT(7),
        eddyzm = SLOT(8), mbarks = SLOT(9), mdwn = SLOT(10), ai = SLOT(11),
        bi = SLOT(12), ci = SLOT(13), ei = SLOT(14), xplus = SLOT(15),
        xminus = SLOT(16), w1 = SLOT(17), w2 = SLOT(18);
    const AcmCol vci[4] = {SLOT(19), SLOT(20), SLOT(21), SLOT(22)};
    const AcmCol di[4] = {SLOT(23), SLOT(24), SLOT(25), SLOT(26)};
#undef SLOT

    const float ust = ust_in[col];
    const float rdt = FDIV(1.0f, dtpbl);              /* ACMPBL :177 */

    /* ---- micromet scaling (:349-366) ---- */
    const float qv1 = qvs[Q(1)], th1s = theta[Q(1)], dens1 = densx[Q(1)];
    const float us1 = us[Q(1)], vs1 = vs[Q(1)];
    const float cpair = FMUL(cpd, FADD(1.0f, FMUL(0.84f, qv1)));
    const float tmpfx = FDIV(hfx[col], FMUL(cpair, dens1));
    const float tmpvtcon = FADD(1.0f, FMUL(ep1, qv1));
    const float ws1 = FSQRT(FADD(FMUL(us1, us1), FMUL(vs1, vs1)));
    const float tst = -FDIV(tmpfx, ust);
    const float qst = -FDIV(qfx[col], FMUL(ust, dens1));
    const float ustm = FDIV(FMUL(ust, ws1), wspd[col]);
    const float thv1 = FMUL(tmpvtcon, th1s);
    float tstv = FADD(FMUL(tst, tmpvtcon), FMUL(FMUL(thv1, ep1), qst));
    if (fabsf(tstv) < 1.0e-6f)
        tstv = copysignf(1.0e-6f, tstv);
    const float mol = FDIV(FMUL(thv1, FMUL(ust, ust)),
                           FMUL(FMUL(ACM_KARMAN, g), tstv));
    const float rmolv = FDIV(1.0f, mol);
    const float pbl_in = pblh[col];
    const float wst = FMUL(ust, gfk_pow(FDIV(pbl_in,
                                             FMUL(ACM_KARMAN, fabsf(mol))),
                                        0.333333f));

    /* ---- heights (:372-401) ---- */
    zf[0] = 0.0f;
    int klpbl = 1;
    for (int k = 1; k <= kte; ++k) {
        zf[k] = FADD(dzf[Q(k)], zf[k - 1]);
        za[k] = FMUL(0.5f, FADD(zf[k], zf[k - 1]));
        dzh[k] = FSUB(zf[k], zf[k - 1]);
        dzhi[k] = FDIV(1.0f, dzh[k]);
    }
    for (int k = 1; k <= kte - 1; ++k)
        dzfi[k] = FDIV(1.0f, FSUB(za[k + 1], za[k]));
    dzfi[kte] = dzfi[kte - 1];
    for (int k = 1; k <= kte; ++k)
        thetav[k] = FMUL(theta[Q(k)], FADD(1.0f, FMUL(ep1, qvs[Q(k)])));
    dzh[kte + 1] = 0.0f;                               /* see header */

    /* ---- PBL height, Richardson number = RIC (:405-498) ---- */
    int ksrc = 1;
    for (int k = 1; k <= kte; ++k) {
        rib[k] = 0.0f;
        ksrc = k;
        if (zf[k] > 30.0f) break;
    }
    float th1 = 0.0f, zh1 = 0.0f, uh1 = 0.0f, vh1 = 0.0f;
    for (int k = 1; k <= ksrc; ++k) {
        th1 = FADD(th1, thetav[k]);
        zh1 = FADD(zh1, za[k]);
        uh1 = FADD(uh1, us[Q(k)]);
        vh1 = FADD(vh1, vs[Q(k)]);
    }
    const float rksrc = (float)ksrc;
    th1 = FDIV(th1, rksrc);
    zh1 = FDIV(zh1, rksrc);
    uh1 = FDIV(uh1, rksrc);
    vh1 = FDIV(vh1, rksrc);
    if (mol < 0.0f && xtime > 1) {
        const float wss = gfk_pow(
            FADD(FMUL(FMUL(ust, ust), ust),
                 FMUL(0.6f, FMUL(FMUL(wst, wst), wst))), 0.33333f);
        const float tconv = -FDIV(FMUL(FMUL(8.5f, ust), tstv), wss);
        th1 = FADD(th1, tconv);
    }
    int kmix = ksrc;
    for (int k = ksrc; k <= kte; ++k)
        if (FSUB(thetav[k], th1) < 0.0f) kmix = k;
    if (kmix > ksrc && kmix == kte) {
        /* WRF reaches its RIB fatal on this column after reading
         * THETAV(kte+1): DTMP < 0 at kte makes RIB(kte) <= 0 < RIC, and
         * kte is the only level the search then visits (see header). */
        atomicOr(status, ACM2_STATUS_RIB_NEVER_EXCEEDS_RIC);
        atomicMin(first_fatal, col);
        return;
    }
    float zmix, umix, vmix;
    if (kmix > ksrc) {
        const float fintt = FDIV(FSUB(th1, thetav[kmix]),
                                 FSUB(thetav[kmix + 1], thetav[kmix]));
        zmix = FADD(FMUL(fintt, FSUB(za[kmix + 1], za[kmix])), za[kmix]);
        umix = FADD(FMUL(fintt, FSUB(us[Q(kmix + 1)], us[Q(kmix)])),
                    us[Q(kmix)]);
        vmix = FADD(FMUL(fintt, FSUB(vs[Q(kmix + 1)], vs[Q(kmix)])),
                    vs[Q(kmix)]);
    } else {
        zmix = zh1; umix = uh1; vmix = vh1;
    }
    int kpblh = -1;
    for (int k = kmix; k <= kte; ++k) {
        const float dtmp = FSUB(thetav[k], th1);
        const float tog = FDIV(FMUL(0.5f, FADD(thetav[k], th1)), g);
        const float uk = us[Q(k)], vk = vs[Q(k)];
        float wssq;
        if (kmix == ksrc) {
            wssq = FADD(FMUL(uk, uk), FMUL(vk, vk));
            wssq = FADD(wssq, FMUL(FMUL(100.0f, ust), ust));
        } else {
            const float du = FSUB(uk, umix), dv = FSUB(vk, vmix);
            wssq = FADD(FMUL(du, du), FMUL(dv, dv));
        }
        wssq = fmaxf(wssq, 0.1f);
        rib[k] = FDIV(FMUL(fabsf(FSUB(za[k], zmix)), dtmp),
                      FMUL(tog, wssq));
        if (rib[k] >= ACM_RIC) { kpblh = k; break; }
    }
    if (kpblh < 0) {
        atomicOr(status, ACM2_STATUS_RIB_NEVER_EXCEEDS_RIC);
        atomicMin(first_fatal, col);
        return;
    }
    float pbl;
    if (kpblh > ksrc) {
        float fint = FDIV(FSUB(ACM_RIC, rib[kpblh - 1]),
                          FSUB(rib[kpblh], rib[kpblh - 1]));
        int kpblht;
        if (fint > 0.5f) {
            kpblht = kpblh;
            fint = FSUB(fint, 0.5f);
        } else {
            kpblht = kpblh - 1;
            fint = FADD(fint, 0.5f);
        }
        pbl = FADD(FMUL(fint, FSUB(zf[kpblht], zf[kpblht - 1])),
                   zf[kpblht - 1]);
        klpbl = kpblht;
    } else {
        klpbl = ksrc;
        pbl = za[ksrc];
    }

    /* ---- free convection (:500-509) ---- */
    int noconv = 0;
    if (FDIV(pbl, mol) < -0.02f && klpbl > 3 && thetav[1] > thetav[2]
            && xtime > 1) {
        noconv = 1;
        regime[col] = 4.0f;
    }

    /* ---- EDDYX (:641-811) ---- */
    for (int k = 1; k <= kte - 1; ++k) {
        float edyz = 0.0f, edyzm = 0.0f;
        const float dzfk = FSUB(za[k + 1], za[k]);
        const float kzo = ACM_EDYZ0;
        if (zf[k] < pbl) {
            const float zovl = FDIV(zf[k], mol);
            float wt, wm;
            if (zovl < 0.0f) {
                const float zs = (zf[k] < FMUL(0.1f, pbl))
                    ? zovl : FDIV(FMUL(0.1f, pbl), mol);
                const float phih = FDIV(1.0f,
                                        FSQRT(FSUB(1.0f, FMUL(ACM_GAMH, zs))));
                const float phim = gfk_pow(FSUB(1.0f, FMUL(ACM_GAMM, zs)),
                                           -0.25f);
                wt = FDIV(ust, phih);
                wm = FDIV(ust, phim);
            } else if (zovl < 1.0f) {
                const float phih = FADD(1.0f, FMUL(ACM_BETAH, zovl));
                wt = FDIV(ust, phih);
                wm = wt;
            } else {
                const float phih = FADD(ACM_BETAH, zovl);
                wt = FDIV(ust, phih);
                wm = wt;
            }
            const float t = FSUB(1.0f, FDIV(zf[k], pbl));
            const float zfunc = FMUL(zf[k], FMUL(t, t));
            edyz = FMUL(FMUL(ACM_KARMAN, wt), zfunc);
            edyzm = FMUL(FMUL(ACM_KARMAN, wm), zfunc);
        }
        const float du = FSUB(us[Q(k + 1)], us[Q(k)]);
        const float dv = FSUB(vs[Q(k + 1)], vs[Q(k)]);
        const float ss = FADD(FDIV(FADD(FMUL(du, du), FMUL(dv, dv)),
                                   FMUL(dzfk, dzfk)), 1.0e-9f);
        const float goth = FDIV(FMUL(2.0f, g), FADD(thetav[k + 1], thetav[k]));
        float ri = FDIV(FMUL(goth, FSUB(thetav[k + 1], thetav[k])),
                        FMUL(dzfk, ss));
        if (FADD(qcs[Q(k)], qis[Q(k)]) > 0.01E-3f
                || FADD(qcs[Q(k + 1)], qis[Q(k + 1)]) > 0.01E-3f) {
            const float qmean = FMUL(0.5f, FADD(qvs[Q(k)], qvs[Q(k + 1)]));
            const float tmean = FMUL(0.5f, FADD(tt[Q(k)], tt[Q(k + 1)]));
            const float xlv = FMUL(FSUB(2.501f,
                                        FMUL(0.00237f, FSUB(tmean, 273.15f))),
                                   1.E6f);
            const float alph = FDIV(FDIV(FMUL(xlv, qmean), rd), tmean);
            const float chi = FDIV(FDIV(FDIV(FDIV(FMUL(FMUL(xlv, xlv), qmean),
                                                  cpair), ACM_RV), tmean),
                                   tmean);
            ri = FMUL(FADD(1.0f, alph),
                      FSUB(ri, FMUL(FDIV(FDIV(FDIV(FMUL(g, g), ss), tmean),
                                         cpair),
                                    FDIV(FSUB(chi, alph), FADD(1.0f, chi)))));
        }
        const float zk = FMUL(0.4f, zf[k]);
        const float sq = FDIV(FMUL(zk, ACM_RLAM), FADD(ACM_RLAM, zk));
        const float sql = FMUL(sq, sq);
        float ez, ezm;
        if (ri >= 0.0f) {
            const float ri2 = FMUL(ri, ri);
            const float fh = FADD(
                FDIV(1.0f, FADD(FADD(FADD(1.0f, FMUL(10.0f, ri)),
                                     FMUL(50.0f, ri2)),
                                FMUL(5000.0f, FMUL(ri2, ri2)))),
                0.0012f);
            const float fm = FADD(FMUL(ACM_PR, fh), 0.00104f);
            ez = FADD(kzo, FMUL(FMUL(FSQRT(ss), fh), sql));
            ezm = FADD(kzo, FMUL(FMUL(FSQRT(ss), fm), sql));
        } else {
            ez = FADD(kzo, FMUL(FSQRT(FMUL(ss, FSUB(1.0f, FMUL(25.0f, ri)))),
                                sql));
            ezm = FMUL(ez, ACM_PR);
        }
        if (edyz > ez) {
            ez = edyz;
            ezm = fminf(edyzm, FMUL(edyz, 0.8f));
        }
        ez = fminf(1000.0f, ez);
        ez = fmaxf(kzo, ez);
        ezm = fminf(1000.0f, ezm);
        ezm = fmaxf(kzo, ezm);
        eddyz[k] = ez;
        eddyzm[k] = ezm;
    }
    eddyz[kte] = 0.0f;
    eddyzm[kte] = 0.0f;

    /* ---- ACM: theta, qv, qc, qi (:517-527, :997-1148) ---- */
    for (int k = 1; k <= kte; ++k) {
        vci[0][k] = theta[Q(k)];
        vci[1][k] = qvs[Q(k)];
        vci[2][k] = qcs[Q(k)];
        vci[3][k] = qis[Q(k)];
    }
    for (int l = 0; l < 4; ++l) vci[l][kte + 1] = 0.0f;   /* see header */
    mdwn[kte + 1] = 0.0f;
    {
        const float fs[4] = {-FMUL(ust, tst), -FMUL(ust, qst), 0.0f, 0.0f};
        acm2_mix<4, false>(dtpbl, noconv, klpbl, pbl, mol, zf, dzh, dzhi,
                           dzfi, eddyz, fs, vci, di, mbarks, mdwn, ai, bi,
                           ci, ei, xplus, xminus, w1, w2, kte);
    }
    /* tendencies (:548-557): UTNP starts at 0. and gains the increment */
    for (int k = 1; k <= kte; ++k) {
        ttnp[Q(k)] = FADD(0.0f, FMUL(FSUB(vci[0][k], theta[Q(k)]), rdt));
        qvtnp[Q(k)] = FADD(0.0f, FMUL(FSUB(vci[1][k], qvs[Q(k)]), rdt));
        qctnp[Q(k)] = FADD(0.0f, FMUL(FSUB(vci[2][k], qcs[Q(k)]), rdt));
        qitnp[Q(k)] = FADD(0.0f, FMUL(FSUB(vci[3][k], qis[Q(k)]), rdt));
    }

    /* ---- ACMM: u, v (:528-535, :1320-1458) ---- */
    for (int k = 1; k <= kte; ++k) {
        vci[0][k] = us[Q(k)];
        vci[1][k] = vs[Q(k)];
    }
    vci[0][kte + 1] = 0.0f;
    vci[1][kte + 1] = 0.0f;
    mdwn[kte + 1] = 0.0f;
    {
        const float fm = -FMUL(ustm, ustm);
        const float wspd1 = FADD(FSQRT(FADD(FMUL(us1, us1), FMUL(vs1, vs1))),
                                 1.E-9f);
        const float fs[2] = {FDIV(FMUL(fm, us1), wspd1),
                             FDIV(FMUL(fm, vs1), wspd1)};
        acm2_mix<2, true>(dtpbl, noconv, klpbl, pbl, mol, zf, dzh, dzhi,
                          dzfi, eddyzm, fs, vci, di, mbarks, mdwn, ai, bi,
                          ci, ei, xplus, xminus, w1, w2, kte);
    }
    for (int k = 1; k <= kte; ++k) {
        utnp[Q(k)] = FADD(0.0f, FMUL(FSUB(vci[0][k], us[Q(k)]), rdt));
        vtnp[Q(k)] = FADD(0.0f, FMUL(FSUB(vci[1][k], vs[Q(k)]), rdt));
    }

    /* ---- exch_h/exch_m (:539-544), carried 2-D ---- */
    for (int k = 1; k <= kte - 1; ++k) {
        exch_h[Q(k)] = eddyz[k];
        exch_m[Q(k)] = eddyzm[k];
    }
    pblh[col] = pbl;
    kpbl[col] = klpbl;
    rmol[col] = rmolv;
    /* Fold validation into the producing thread. The model health ledger
     * consumes this word on-device, without a per-step host branch. */
    unsigned int invalid = 0;
    for (int k = 1; k <= kte; ++k) {
        if (!isfinite(utnp[Q(k)])) invalid |= 1u << 1;
        if (!isfinite(vtnp[Q(k)])) invalid |= 1u << 2;
        if (!isfinite(ttnp[Q(k)])) invalid |= 1u << 3;
        if (!isfinite(qvtnp[Q(k)])) invalid |= 1u << 4;
        if (!isfinite(qctnp[Q(k)])) invalid |= 1u << 5;
        if (!isfinite(qitnp[Q(k)])) invalid |= 1u << 6;
        if (k < kte) {
            if (!isfinite(exch_h[Q(k)])) invalid |= 1u << 10;
            if (!isfinite(exch_m[Q(k)])) invalid |= 1u << 11;
        }
    }
    if (!isfinite(pblh[col])) invalid |= 1u << 7;
    if (!isfinite(rmol[col])) invalid |= 1u << 8;
    if (!isfinite(regime[col])) invalid |= 1u << 9;
    if (invalid) atomicOr(status, invalid);
#undef Q
}
