/* Betts-Miller-Janjic cumulus (cu_physics = 2), one CUDA thread per column.
 *
 * WRF v4.6.1 transcription anchors, all phys/module_cu_bmj.F:
 *     13-69     module parameters (DSP*, EFI*, PONE, PSH, TREL, table bounds)
 *     76-387    BMJDRV  -- the per-column driver: flip, mixing ratio to
 *               specific humidity, BMJ land mask, the bmj_rad_feedback
 *               cloud fraction and condensate (Slingo 1987, Poisson profile)
 *     391-1730  BMJ     -- the column scheme (max-buoyancy search, deep
 *               reference profiles with the cloud-efficiency iteration,
 *               shallow reference profiles)
 *     1734-1810 TTBLEX  -- moist-adiabat temperature from TTBL/TTBLQ
 *     1813-2077 BMJINIT -- the PTBL/TTBL/TTBLQ lookup tables
 *     2081-2189 SPLINE  -- Janjic's natural cubic spline for the tables
 *
 * Arithmetic is spelled in rounded intrinsics (FADD/FSUB/FMUL/FDIV from the
 * shared header) in the Fortran's own left-to-right order, so neither the
 * default nor the strict (GPUWM_WRF_EXACT=1) compile can contract a
 * multiply-add: both arms run this one kernel and round identically.
 * EXP, LOG, real ** real and GAMMA call WOOF's own float32 routines
 * gfk_exp / gfk_log / gfk_pow / gfk_tgamma.  gfortran folds only the real
 * exponents 2.0, -1.0 and 1.0 and integer exponents into multiplies; every
 * other real exponent here (CAPA, 0.5, JPR) is a pow call, and the oracle's
 * `nm -u` (tools/bmj_wrf461_oracle/libm-report.txt) lists exactly expf,
 * logf, powf and tgammaf -- no sqrtf, so PWCOL**GAM is NOT a square root.
 * Constant intrinsic calls with constant arguments (LOG(CRMX/CRMN),
 * ACOS(-1.)) are folded by gfortran at compile time, correctly rounded;
 * they appear below as their folded float32 words.
 *
 * DECLARED DIVERGENCE (owner ruling, the Grell-Freitas precedent,
 * docs/gf_gamma_known_delta.md): GAMMA(JPR+1.) in the bmj_rad_feedback
 * cloud fraction (:308) is WOOF's correctly rounded gfk_tgamma, not the C
 * library's tgammaf that gfortran binds the intrinsic to.  The oracle is run
 * twice -- stock, and with a correctly rounded tgammaf interposed -- and this
 * kernel is graded at 0 ULP against the second on every field; the first
 * measures the divergence, which can only reach CCLDFRA, QCCONV, QICONV.
 *
 * LAYOUT.  Engine columns are bottom-up and level-major, a[k * ncol + i].
 * BMJ counts DOWNWARD from the model top; the driver's KFLIP = KTE+1-K is
 * applied on every read (Fortran L = 1..nz maps to engine level nz-L).
 * LOWLYR is 1 (BMJINIT, :1880-1884), so LMH = KTE = nz.
 *
 * The column arrays live in a global workspace (the kf.cu / gf.cu shape),
 * slot-major and lane-interleaved: element L of slot s for tile column t is
 * ws[(s * nz + L - 1) * ntile + t], so a warp's 32 lanes touch 32
 * consecutive floats.  Function-scope column arrays would put ~20 * nz
 * floats in the per-thread local frame, which CUDA reserves for every
 * resident thread on the card.
 */

#define BMJ_ITB   76
#define BMJ_JTB   134
#define BMJ_ITBQ  152
#define BMJ_JTBQ  440

/* The table buffer, in BMJINIT's declaration and dump order (Fortran
 * column-major inside each array): PTBL(IQ,IT) at (IT-1)*ITB + IQ-1,
 * TTBL(ITH,IP) at (IP-1)*JTB + ITH-1, TTBLQ(ITH,IP) at (IP-1)*JTBQ + ITH-1. */
#define BMJ_OFF_QS0    0
#define BMJ_OFF_SQS    (BMJ_OFF_QS0 + BMJ_JTB)
#define BMJ_OFF_PTBL   (BMJ_OFF_SQS + BMJ_JTB)
#define BMJ_OFF_THE0   (BMJ_OFF_PTBL + BMJ_ITB * BMJ_JTB)
#define BMJ_OFF_STHE   (BMJ_OFF_THE0 + BMJ_ITB)
#define BMJ_OFF_TTBL   (BMJ_OFF_STHE + BMJ_ITB)
#define BMJ_OFF_THE0Q  (BMJ_OFF_TTBL + BMJ_JTB * BMJ_ITB)
#define BMJ_OFF_STHEQ  (BMJ_OFF_THE0Q + BMJ_ITBQ)
#define BMJ_OFF_TTBLQ  (BMJ_OFF_STHEQ + BMJ_ITBQ)
#define BMJ_TABLE_FLOATS (BMJ_OFF_TTBLQ + BMJ_JTBQ * BMJ_ITBQ)

/* ---- share/module_model_constants.F (the values the driver passes) ---- */
#define BMJ_RD     287.0f
#define BMJ_CPD    1004.5f            /* 7.*r_d/2., exact in float32 */
#define BMJ_XLV    2.5e6f
#define BMJ_XLS    2.85e6f
#define BMJ_GRAV   9.81f
#define BMJ_SVPT0  273.15f
#define BMJ_PQ0    379.90516f
#define BMJ_A2     17.2693882f
#define BMJ_A3     273.16f
#define BMJ_A4     35.86f
#define BMJ_EPSQ   1.e-12f

/* ---- module_cu_bmj.F:13-46 parameters (plain literals) ---------------- */
#define BMJ_EFIFC   5.0f
#define BMJ_EFIMN   0.20f
#define BMJ_EFMNT   0.70f
#define BMJ_ELIWV   2.683e6f
#define BMJ_EPSDN   1.05f
#define BMJ_EPSDT   0.0f
#define BMJ_EPSNTP  0.0001f
#define BMJ_EPSPR   1.e-7f
#define BMJ_FSL     0.85f
#define BMJ_FSS     0.85f
#define BMJ_CRMN    0.14f
#define BMJ_PFRZ    15000.0f
#define BMJ_PNO     1000.0f
#define BMJ_PONE    2500.0f
#define BMJ_PQM     20000.0f
#define BMJ_PSH     20000.0f
#define BMJ_PSHU    45000.0f
#define BMJ_RHLSC   0.00f
#define BMJ_RHHSC   1.10f
#define BMJ_ROW     1.e3f
#define BMJ_STABDF  0.90f
#define BMJ_STABDS  0.90f
#define BMJ_STABS   1.0f
#define BMJ_DTSHAL  (-1.0f)
#define BMJ_TREL    2400.0f
#define BMJ_PL      2500.0f
#define BMJ_PLQ     70000.0f
#define BMJ_PH      105000.0f
#define BMJ_THL     210.0f
#define BMJ_THH     365.0f
#define BMJ_THHQ    325.0f
#define BMJ_ELEVFC  0.6f
#define BMJ_STEFI   1.0f
/* DSPBFL=-3875.*FR etc. with FR = 1.00 exactly. */
#define BMJ_DSPBFL  (-3875.0f)
#define BMJ_DSP0FL  (-5875.0f)
#define BMJ_DSPTFL  (-1875.0f)
#define BMJ_DSPBFS  (-3875.0f)
#define BMJ_DSP0FS  (-5875.0f)
#define BMJ_DSPTFS  (-1875.0f)

/* Folded intrinsic constants (gfortran evaluates these at compile time,
 * correctly rounded, even at -O0):
 *   LOG(CRMX/CRMN): CRMX/CRMN = 607.142883f (0x4417c925); its correctly
 *   rounded log is 6.40876436f (0x40cd1499).  The exact value sits 0.13 ULP
 *   above a float32 rounding midpoint, so a logf that is not correctly
 *   rounded could return the neighbour; the folded word is used.
 *   ACOS(-1.) = 0x40490fdb. */
#define BMJ_LOG_CRMX_CRMN_BITS 0x40cd1499u
#define BMJ_ACOS_M1  3.14159274101257324219f   /* 0x40490fdb */

/* Module-level PARAMETER expressions, computed as gfortran folds them: one
 * float32 rounding per operation, left to right. */
struct BmjParams {
    float rendp_unused;
    float rdp, rdpq, rdq, rdth, rdthe, rdtheq, rsfcp, avgefi, peps;
    float dspbsl, dsp0sl, dsptsl, dspbss, dsp0ss, dsptss;
    float slopbl, slop0l, sloptl, slopbs, slop0s, slopts, slopst, slope;
    float d608;
};

__device__ __forceinline__ BmjParams bmj_params()
{
    BmjParams p;
    p.rendp_unused = 0.0f;
    /* RDP=(ITB-1.)/(PH-PL) etc., :64-67 */
    p.rdp    = FDIV(FSUB((float)BMJ_ITB, 1.0f), FSUB(BMJ_PH, BMJ_PL));
    p.rdpq   = FDIV(FSUB((float)BMJ_ITBQ, 1.0f), FSUB(BMJ_PH, BMJ_PLQ));
    p.rdq    = (float)(BMJ_ITB - 1);
    p.rdth   = FDIV(FSUB((float)BMJ_JTB, 1.0f), FSUB(BMJ_THH, BMJ_THL));
    p.rdthe  = FSUB((float)BMJ_JTB, 1.0f);
    p.rdtheq = FSUB((float)BMJ_JTBQ, 1.0f);
    p.rsfcp  = FDIV(1.0f, 101300.0f);
    p.avgefi = FMUL(FADD(BMJ_EFIMN, 1.0f), 0.5f);              /* :69 */
    p.peps   = FDIV(1.0f, 2.5f);                                /* :20 */
    /* :474-488 */
    p.dspbsl = FMUL(BMJ_DSPBFL, BMJ_FSL);
    p.dsp0sl = FMUL(BMJ_DSP0FL, BMJ_FSL);
    p.dsptsl = FMUL(BMJ_DSPTFL, BMJ_FSL);
    p.dspbss = FMUL(BMJ_DSPBFS, BMJ_FSS);
    p.dsp0ss = FMUL(BMJ_DSP0FS, BMJ_FSS);
    p.dsptss = FMUL(BMJ_DSPTFS, BMJ_FSS);
    float one_m_efimn = FSUB(1.0f, BMJ_EFIMN);
    p.slopbl = FDIV(FSUB(BMJ_DSPBFL, p.dspbsl), one_m_efimn);
    p.slop0l = FDIV(FSUB(BMJ_DSP0FL, p.dsp0sl), one_m_efimn);
    p.sloptl = FDIV(FSUB(BMJ_DSPTFL, p.dsptsl), one_m_efimn);
    p.slopbs = FDIV(FSUB(BMJ_DSPBFS, p.dspbss), one_m_efimn);
    p.slop0s = FDIV(FSUB(BMJ_DSP0FS, p.dsp0ss), one_m_efimn);
    p.slopts = FDIV(FSUB(BMJ_DSPTFS, p.dsptss), one_m_efimn);
    p.slopst = FDIV(FSUB(BMJ_STABDF, BMJ_STABDS), one_m_efimn);
    p.slope  = FDIV(FSUB(1.0f, BMJ_EFMNT), one_m_efimn);
    /* EP_1 = R_v/R_d - 1. (module_model_constants.F:80) */
    p.d608   = FSUB(FDIV(461.6f, 287.0f), 1.0f);
    return p;
}

/* ===================================================================== */
/* SPLINE, :2081-2189.  Arrays are 1-based in the Fortran; here x[k-1].   */
/* ===================================================================== */
__device__ void bmj_spline(int nold, const float* xold, const float* yold,
                           float* y2, int nnew, const float* xnew,
                           float* ynew, float* p, float* q)
{
    int k, k1, k2, kold = 0;
    float ak = 0.0f, bk = 0.0f, ck = 0.0f, den, dx, dxc, dxl, dxr;
    float dydxl, dydxr, rdx, rtdxc, x, xk, xsq, y2k, y2kp1;
    int noldm1 = nold - 1;

    dxl = FSUB(xold[1], xold[0]);
    dxr = FSUB(xold[2], xold[1]);
    dydxl = FDIV(FSUB(yold[1], yold[0]), dxl);
    dydxr = FDIV(FSUB(yold[2], yold[1]), dxr);
    rtdxc = FDIV(0.5f, FADD(dxl, dxr));
    /* P(1)= RTDXC*(6.*(DYDXR-DYDXL)-DXL*Y2(1)) */
    p[0] = FMUL(rtdxc, FSUB(FMUL(6.0f, FSUB(dydxr, dydxl)), FMUL(dxl, y2[0])));
    q[0] = FMUL(-rtdxc, dxr);
    if (nold != 3) {
        k = 3;
        do {                                           /* label 100 */
            dxl = dxr;
            dydxl = dydxr;
            dxr = FSUB(xold[k], xold[k - 1]);
            dydxr = FDIV(FSUB(yold[k], yold[k - 1]), dxr);
            dxc = FADD(dxl, dxr);
            den = FDIV(1.0f, FADD(FADD(FMUL(dxl, q[k - 3]), dxc), dxc));
            p[k - 2] = FMUL(den, FSUB(FMUL(6.0f, FSUB(dydxr, dydxl)),
                                      FMUL(dxl, p[k - 3])));
            q[k - 2] = FMUL(-den, dxr);
            k = k + 1;
        } while (k < nold);
    }
    k = noldm1;                                        /* label 150 */
    do {                                               /* label 200 */
        y2[k - 1] = FADD(p[k - 2], FMUL(q[k - 2], y2[k]));
        k = k - 1;
    } while (k > 1);
    k1 = 1;
    do {                                               /* label 300 */
        xk = xnew[k1 - 1];
        bool found = false;
        for (k2 = 2; k2 <= nold; ++k2) {
            if (xold[k2 - 1] > xk) { kold = k2 - 1; found = true; break; }
        }
        if (!found) {
            ynew[k1 - 1] = yold[nold - 1];
        } else {
            /* 450: IF(K1==1)GO TO 500; IF(K==KOLD)GO TO 550 */
            if (k1 == 1 || k != kold) {
                k = kold;                                /* label 500 */
                y2k = y2[k - 1];
                y2kp1 = y2[k];
                dx = FSUB(xold[k], xold[k - 1]);
                rdx = FDIV(1.0f, dx);
                ak = FMUL(FMUL(0.1666667f, rdx), FSUB(y2kp1, y2k));
                bk = FMUL(0.5f, y2k);
                ck = FSUB(FMUL(rdx, FSUB(yold[k], yold[k - 1])),
                          FMUL(FMUL(0.1666667f, dx),
                               FADD(FADD(y2kp1, y2k), y2k)));
            }
            x = FSUB(xk, xold[k - 1]);                   /* label 550 */
            xsq = FMUL(x, x);
            ynew[k1 - 1] = FADD(FADD(FADD(FMUL(FMUL(ak, xsq), x),
                                          FMUL(bk, xsq)),
                                     FMUL(ck, x)), yold[k - 1]);
        }
        k1 = k1 + 1;                                   /* label 600 */
    } while (k1 <= nnew);
}

/* ===================================================================== */
/* BMJINIT tables, :1886-2075.  One block per table family:              */
/*   blockIdx.x == 0: PTBL/QS0/SQS, one thread per KTH (1..JTB)           */
/*   blockIdx.x == 1: TTBL/THE0/STHE, one thread per KP (1..ITB)          */
/*   blockIdx.x == 2: TTBLQ/THE0Q/STHEQ, one thread per KP (1..ITBQ)      */
/* scratch: 7 * JTBQ floats per thread, global, (block, thread)-indexed.  */
/* ===================================================================== */
extern "C" __global__ void bmj_init_tables(float* __restrict__ tab,
                                           float* __restrict__ scratch,
                                           float cp, float rd)
{
    const int fam = blockIdx.x;
    const int tid = threadIdx.x;
    const float eps = 1.e-9f;
    float* s = scratch + ((size_t)fam * blockDim.x + tid) * (7 * BMJ_JTBQ);
    float* xold = s;
    float* yold = s + BMJ_JTBQ;
    float* y2 = s + 2 * BMJ_JTBQ;
    float* xnew = s + 3 * BMJ_JTBQ;
    float* ynew = s + 4 * BMJ_JTBQ;
    float* pp = s + 5 * BMJ_JTBQ;
    float* qq = s + 6 * BMJ_JTBQ;
    const float rd_cp = FDIV(rd, cp);
    const float elocp = FDIV(BMJ_ELIWV, cp);

    if (fam == 0) {
        /* COARSE LOOK-UP TABLE FOR SATURATION POINT, :1889-1949 */
        const int kthm = BMJ_JTB, kpm = BMJ_ITB;
        const int kpm1 = kpm - 1;
        if (tid >= kthm) return;
        const int kth = tid + 1;
        float dth = FDIV(FSUB(BMJ_THH, BMJ_THL), (float)(kthm - 1));
        float dp = FDIV(FSUB(BMJ_PH, BMJ_PL), (float)(kpm - 1));
        float th = FSUB(BMJ_THL, dth);
        for (int j = 1; j <= kth; ++j) th = FADD(th, dth);
        float p = FSUB(BMJ_PL, dp);
        for (int kp = 1; kp <= kpm; ++kp) {
            p = FADD(p, dp);
            float ape = gfk_pow(FDIV(100000.0f, p), rd_cp);
            float denom = FSUB(th, FMUL(BMJ_A4, ape));
            if (denom > eps) {
                xold[kp - 1] = FMUL(FDIV(BMJ_PQ0, p),
                    gfk_exp(FDIV(FMUL(BMJ_A2, FSUB(th, FMUL(BMJ_A3, ape))), denom)));
            } else {
                xold[kp - 1] = 0.0f;
            }
            yold[kp - 1] = p;                           /* POLD */
        }
        float qs0k = xold[0];
        float sqsk = FSUB(xold[kpm - 1], xold[0]);
        xold[0] = 0.0f;
        xold[kpm - 1] = 1.0f;
        for (int kp = 2; kp <= kpm1; ++kp) {
            xold[kp - 1] = FDIV(FSUB(xold[kp - 1], qs0k), sqsk);
            if (FSUB(xold[kp - 1], xold[kp - 2]) < eps)
                xold[kp - 1] = FADD(xold[kp - 2], eps);
        }
        tab[BMJ_OFF_QS0 + kth - 1] = qs0k;
        tab[BMJ_OFF_SQS + kth - 1] = sqsk;
        xnew[0] = 0.0f;
        xnew[kpm - 1] = 1.0f;
        float dqs = FDIV(1.0f, (float)(kpm - 1));
        for (int kp = 2; kp <= kpm1; ++kp) xnew[kp - 1] = FADD(xnew[kp - 2], dqs);
        y2[0] = 0.0f;
        y2[kpm - 1] = 0.0f;
        bmj_spline(kpm, xold, yold, y2, kpm, xnew, ynew, pp, qq);
        for (int kp = 1; kp <= kpm; ++kp)
            tab[BMJ_OFF_PTBL + (kth - 1) * BMJ_ITB + kp - 1] = ynew[kp - 1];
        return;
    }

    /* T(P) FROM CONSTANT THE, coarse (:1953-2005) and fine (:2011-2075) */
    const bool fine = (fam == 2);
    const int kthm = fine ? BMJ_JTBQ : BMJ_JTB;
    const int kpm = fine ? BMJ_ITBQ : BMJ_ITB;
    const int kthm1 = kthm - 1;
    if (tid >= kpm) return;
    const int kp = tid + 1;
    const float pbase = fine ? BMJ_PLQ : BMJ_PL;
    const float thtop = fine ? BMJ_THHQ : BMJ_THH;
    float dth = FDIV(FSUB(thtop, BMJ_THL), (float)(kthm - 1));
    float dp = FDIV(FSUB(BMJ_PH, pbase), (float)(kpm - 1));
    float p = FSUB(pbase, dp);
    for (int j = 1; j <= kp; ++j) p = FADD(p, dp);
    float th = FSUB(BMJ_THL, dth);
    for (int kth = 1; kth <= kthm; ++kth) {
        th = FADD(th, dth);
        float ape = gfk_pow(FDIV(1.e5f, p), rd_cp);
        float denom = FSUB(th, FMUL(BMJ_A4, ape));
        float qs;
        if (denom > eps) {
            qs = FMUL(FDIV(BMJ_PQ0, p),
                      gfk_exp(FDIV(FMUL(BMJ_A2, FSUB(th, FMUL(BMJ_A3, ape))), denom)));
        } else {
            qs = 0.0f;
        }
        float told = FDIV(th, ape);
        yold[kth - 1] = told;                           /* TOLD */
        xold[kth - 1] = FMUL(th, gfk_exp(FDIV(FMUL(elocp, qs), told)));  /* THEOLD */
    }
    float the0k = xold[0];
    float sthek = FSUB(xold[kthm - 1], xold[0]);
    xold[0] = 0.0f;
    xold[kthm - 1] = 1.0f;
    for (int kth = 2; kth <= kthm1; ++kth) {
        xold[kth - 1] = FDIV(FSUB(xold[kth - 1], the0k), sthek);
        if (FSUB(xold[kth - 1], xold[kth - 2]) < eps)
            xold[kth - 1] = FADD(xold[kth - 2], eps);
    }
    if (fine) {
        tab[BMJ_OFF_THE0Q + kp - 1] = the0k;
        tab[BMJ_OFF_STHEQ + kp - 1] = sthek;
    } else {
        tab[BMJ_OFF_THE0 + kp - 1] = the0k;
        tab[BMJ_OFF_STHE + kp - 1] = sthek;
    }
    xnew[0] = 0.0f;
    xnew[kthm - 1] = 1.0f;
    float dthe = FDIV(1.0f, (float)(kthm - 1));
    for (int kth = 2; kth <= kthm1; ++kth) xnew[kth - 1] = FADD(xnew[kth - 2], dthe);
    y2[0] = 0.0f;
    y2[kthm - 1] = 0.0f;
    bmj_spline(kthm, xold, yold, y2, kthm, xnew, ynew, pp, qq);
    if (fine) {
        for (int kth = 1; kth <= kthm; ++kth)
            tab[BMJ_OFF_TTBLQ + (kp - 1) * BMJ_JTBQ + kth - 1] = ynew[kth - 1];
    } else {
        for (int kth = 1; kth <= kthm; ++kth)
            tab[BMJ_OFF_TTBL + (kp - 1) * BMJ_JTB + kth - 1] = ynew[kth - 1];
    }
}

/* ===================================================================== */
/* TTBLEX, :1734-1810                                                     */
/* ===================================================================== */
__device__ __forceinline__ float bmj_ttblex(
    const float* __restrict__ tab, bool fine, float prsmid, float thesp,
    const BmjParams& c)
{
    const int itbx = fine ? BMJ_ITBQ : BMJ_ITB;
    const int jtbx = fine ? BMJ_JTBQ : BMJ_JTB;
    const float plx = fine ? BMJ_PLQ : BMJ_PL;
    const float rdpx = fine ? c.rdpq : c.rdp;
    const float rdthex = fine ? c.rdtheq : c.rdthe;
    const float* sthe = tab + (fine ? BMJ_OFF_STHEQ : BMJ_OFF_STHE);
    const float* the0 = tab + (fine ? BMJ_OFF_THE0Q : BMJ_OFF_THE0);
    const float* ttbl = tab + (fine ? BMJ_OFF_TTBLQ : BMJ_OFF_TTBL);

    float pk = prsmid;
    float tpk = FMUL(FSUB(pk, plx), rdpx);
    float qq = FSUB(tpk, truncf(tpk));
    int iptb = (int)tpk + 1;
    if (iptb < 1) { iptb = 1; qq = 0.0f; }
    if (iptb >= itbx) { iptb = itbx - 1; qq = 0.0f; }
    float bthe00k = the0[iptb - 1];
    float sthe00k = sthe[iptb - 1];
    float bthe10k = the0[iptb];
    float sthe10k = sthe[iptb];
    float bthk = FADD(FMUL(FSUB(bthe10k, bthe00k), qq), bthe00k);
    float sthk = FADD(FMUL(FSUB(sthe10k, sthe00k), qq), sthe00k);
    float tthk = FMUL(FDIV(FSUB(thesp, bthk), sthk), rdthex);
    float pp = FSUB(tthk, truncf(tthk));
    int ithtb = (int)tthk + 1;
    if (ithtb < 1) { ithtb = 1; pp = 0.0f; }
    if (ithtb >= jtbx) { ithtb = jtbx - 1; pp = 0.0f; }
    float t00k = ttbl[(iptb - 1) * jtbx + ithtb - 1];
    float t10k = ttbl[(iptb - 1) * jtbx + ithtb];
    float t01k = ttbl[iptb * jtbx + ithtb - 1];
    float t11k = ttbl[iptb * jtbx + ithtb];
    /* (T00K+(T10K-T00K)*PP+(T01K-T00K)*QQ+(T00K-T10K-T01K+T11K)*PP*QQ) */
    return FADD(FADD(FADD(t00k, FMUL(FSUB(t10k, t00k), pp)),
                     FMUL(FSUB(t01k, t00k), qq)),
                FMUL(FMUL(FADD(FSUB(FSUB(t00k, t10k), t01k), t11k), pp), qq));
}

/* Workspace slots (each nz floats per column). */
enum {
    S_Q = 0, S_DPRS, S_APE, S_CPE, S_DTV, S_THES, S_CPECNV, S_DTVCNV,
    S_THESCNV, S_DIFT, S_DIFQ, S_TREFK, S_QREFK, S_PSK, S_APESK, S_THERK,
    S_QSATK, S_THVREF, S_DQDT, S_DTDT, BMJ_NSLOTS
};
/* After BMJ returns, the driver's JPR and PVPR (:139) reuse two slots BMJ
 * no longer reads; the driver's own arrays are (KTS:KTE) bottom-up, so they
 * are indexed by the WRF K, not the BMJ L. */
#define S_JPR  S_CPECNV
#define S_PVPR S_DTVCNV

/* bmj_path codes (diagnostic output, one int per column): which exit of BMJ
 * the column took.  They name module_cu_bmj.F branches, nothing more. */
#define BMJ_PATH_QUICK_EXIT     0   /* :906 thin cloud / no CAPE        */
#define BMJ_PATH_DEEP           1   /* :1206 deep convection            */
#define BMJ_PATH_SHALLOW        2   /* :1707 shallow relaxation applied */
#define BMJ_PATH_SH_THIN        3   /* :1453 shallow cloud too thin     */
#define BMJ_PATH_SH_DST         4   /* :1601 entropy change not < 0     */
#define BMJ_PATH_SH_ISOTHERMAL  5   /* :1616                            */
#define BMJ_PATH_SH_DQREF       6   /* :1631 humidity rises with height */
#define BMJ_PATH_SH_DRY         7   /* :1654 too dry                    */
#define BMJ_PATH_SH_MOIST       8   /* :1664 too moist                  */
#define BMJ_PATH_SH_SLOPE       9   /* :1694 impossible slope           */
#define BMJ_PATH_DEEP_TO_SHALLOW 16 /* added when the deep arm fell back */

extern "C" __global__ void bmj_column(
    /* bottom-up, level-major a[k*ncol+i] (nz levels; pint nz+1) */
    const float* __restrict__ t_a, const float* __restrict__ qv_a,
    const float* __restrict__ pmid_a, const float* __restrict__ pi_a,
    const float* __restrict__ rho_a, const float* __restrict__ dz8w_a,
    const float* __restrict__ pint_a,
    const float* __restrict__ xland_a,          /* (ncol) */
    float* __restrict__ cldefi_a,               /* (ncol) inout, carried */
    const float* __restrict__ tab,              /* BMJINIT tables */
    float* __restrict__ ws,                     /* BMJ_NSLOTS*nz*ntile */
    /* outputs */
    float* __restrict__ rthcuten_a, float* __restrict__ rqvcuten_a,
    float* __restrict__ ccldfra_a, float* __restrict__ qcconv_a,
    float* __restrict__ qiconv_a,
    float* __restrict__ raincv_a, float* __restrict__ pratec_a,
    float* __restrict__ cutop_a, float* __restrict__ cubot_a,
    float* __restrict__ convcld_a, int* __restrict__ path_a,
    float dt, int stepcu, int rad_feedback,
    int col0, int ntile, int ncol, int nz)
{
    const int tcol = blockIdx.x * blockDim.x + threadIdx.x;
    if (tcol >= ntile) return;
    const int i = col0 + tcol;
    if (i >= ncol) return;
    const size_t st = (size_t)ncol;
#define LEV(a, k)   a[(size_t)(k) * st + i]           /* engine k, 0-based */
#define W(s, L)     ws[((size_t)(s) * nz + (L) - 1) * (size_t)ntile + tcol]
#define TT(L)       LEV(t_a, nz - (L))                 /* BMJ T(L)      */
#define PRS(L)      LEV(pmid_a, nz - (L))              /* BMJ PRSMID(L) */

    const BmjParams c = bmj_params();
    const float cp = BMJ_CPD, r = BMJ_RD, elwv = BMJ_XLV, g = BMJ_GRAV;
    const float tfrz = BMJ_SVPT0, d608 = c.d608;
    (void)BMJ_XLS;

    /* ------------------------- BMJDRV, :170-238 ------------------------ */
    const float dtcnvc = FMUL(dt, (float)stepcu);
    for (int k = 1; k <= nz; ++k) {
        W(S_DQDT, k) = 0.0f;
        W(S_DTDT, k) = 0.0f;
        LEV(qcconv_a, k - 1) = 0.0f;
        LEV(qiconv_a, k - 1) = 0.0f;
        LEV(ccldfra_a, k - 1) = 0.0f;
    }
    float dqcol = 0.0f, pwcol = 0.0f, pcpcol = 0.0f, dqcolmin = 0.0f;
    float raincv = 0.0f, pratec = 0.0f, convcld = 0.0f;
    const float psfc = LEV(pint_a, 0);                 /* PINT(I,LOWLYR=1) */
    const float sm = FSUB(xland_a[i], 1.0f);           /* LANDMASK */
    for (int k = 1; k <= nz; ++k) {
        int kflip = nz + 1 - k;
        float qvk = LEV(qv_a, kflip - 1);
        W(S_Q, k) = fmaxf(BMJ_EPSQ, FDIV(qvk, FADD(1.0f, qvk)));
        W(S_DPRS, k) = FMUL(FMUL(LEV(rho_a, kflip - 1), g), LEV(dz8w_a, kflip - 1));
    }
    const int lmh = nz;                                /* KTE+1-LOWLYR */
    float cldefi = cldefi_a[i];
    int lbot, ltop, path = BMJ_PATH_QUICK_EXIT;

    /* =========================== BMJ, :391 ============================= */
    {
    int iq, iqtb, it, itrefi, ittb, ittbk, kb, l, l0, l0m1, lb, lbm1, lcor;
    int lqm, ltp1, ltsh, lbotcnv, ltopcnv, lmid, iter;
    float apekl, apekxx, apekxy, apes, apests, avrgt, avrgtl, bq, bqk;
    float bqs00k, bqs10k, capa, cup, den, dentpy, depmin, depth, depwl, dhdt;
    float diftl, difql, dp, dpkl, dpmix, dpot, dqref, drheat, dsp, dsp0k;
    float dspbk, dsptk, dsq, dst, dstq, dthem, dtdp, efi, fefi, fptk, hcorr;
    float otsum, p, p00k, p01k, p10k, p11k, part1, part2, part3, pbot;
    float pbotfc, pk0, pkb, pkl, pkt, pkxxxx, pkxxxy, plmh, pelevfc, pbtmx;
    float plo, potsum, pp1, ppk, preck, prwd, pdqd, psp, psum, ptop, ptpk;
    float pup, qbt, qkl, qnew, qotsum, qq1, qqk, qrfkl, qrftp, qsp, qsum, qup;
    float rdp0t, rdpsum, rdtcnvc, rotsum, rtbar, rhavg, sm1, smix, sq, sqk;
    float sqs00k, sqs10k, stabdl, sumde, sumdp, sumdt, tauk, tauksc, tcorr;
    float thbt, therkx, therky, thesp, thskl, thtpk, thvmkl, tkl, tnew, tq;
    float tqk, trefkx, trfkl, trmlo, trmup, tsp, tth, tthk, tup;
    float capecnv, pspcnv, thbtcnv, capetrigr, cape, tlev2, qsat1, qsat2;
    float rhshmax, a23m4l, cprlg, elocp, rcp, qwat;
    bool deep = false, shallow = false;
    (void)deep;

    capa = FDIV(r, cp);
    cprlg = FDIV(cp, FMUL(FMUL(BMJ_ROW, g), elwv));
    elocp = FDIV(BMJ_ELIWV, cp);
    rcp = FDIV(1.0f, cp);
    a23m4l = FMUL(FMUL(BMJ_A2, FSUB(BMJ_A3, BMJ_A4)), elwv);
    rdtcnvc = FDIV(1.0f, dtcnvc);
    depmin = FMUL(FMUL(BMJ_PSH, psfc), c.rsfcp);
    tauk = FDIV(dtcnvc, BMJ_TREL);
    tauksc = FDIV(dtcnvc, FMUL(1.0f, BMJ_TREL));

    lbot = lmh;
    dqcol = 0.0f; pwcol = 0.0f; pcpcol = 0.0f; dqcolmin = 0.0f;
    for (l = 1; l <= lmh; ++l) {
        apests = PRS(l);
        W(S_APE, l) = gfk_pow(FDIV(1.e5f, apests), capa);
        W(S_CPECNV, l) = 0.0f;
        W(S_DTVCNV, l) = 0.0f;
        W(S_THESCNV, l) = 0.0f;
    }
    /* -------------- SEARCH FOR MAXIMUM BUOYANCY LEVEL, :533 ----------- */
    plmh = PRS(lmh);
    pelevfc = FMUL(plmh, BMJ_ELEVFC);
    pbtmx = FSUB(PRS(lmh), BMJ_PONE);
    capecnv = 0.0f; pspcnv = 0.0f; thbtcnv = 0.0f;
    lbotcnv = lbot; ltopcnv = lbot;
    psp = 0.0f; thbt = 0.0f; pbot = 0.0f; ptop = 0.0f; ltop = lmh;

    for (kb = lmh; kb >= 1; --kb) {                    /* max_buoy_loop */
        pkl = PRS(kb);
        if (pkl < pelevfc) break;
        lbot = lmh;
        ltop = lmh;
        qbt = W(S_Q, kb);
        thbt = FMUL(TT(kb), W(S_APE, kb));
        tth = FMUL(FSUB(thbt, BMJ_THL), c.rdth);
        qq1 = FSUB(tth, truncf(tth));
        ittb = (int)tth + 1;
        if (ittb < 1) { ittb = 1; qq1 = 0.0f; }
        else if (ittb >= BMJ_JTB) { ittb = BMJ_JTB - 1; qq1 = 0.0f; }
        ittbk = ittb;
        bqs00k = tab[BMJ_OFF_QS0 + ittbk - 1];
        sqs00k = tab[BMJ_OFF_SQS + ittbk - 1];
        bqs10k = tab[BMJ_OFF_QS0 + ittbk];
        sqs10k = tab[BMJ_OFF_SQS + ittbk];
        bq = FADD(FMUL(FSUB(bqs10k, bqs00k), qq1), bqs00k);
        sq = FADD(FMUL(FSUB(sqs10k, sqs00k), qq1), sqs00k);
        tq = FMUL(FDIV(FSUB(qbt, bq), sq), c.rdq);
        pp1 = FSUB(tq, truncf(tq));
        iqtb = (int)tq + 1;
        if (iqtb < 1) { iqtb = 1; pp1 = 0.0f; }
        else if (iqtb >= BMJ_ITB) { iqtb = BMJ_ITB - 1; pp1 = 0.0f; }
        iq = iqtb;
        it = ittb;
#define PTBL(IQ, IT) tab[BMJ_OFF_PTBL + ((IT) - 1) * BMJ_ITB + (IQ) - 1]
        p00k = PTBL(iq, it);
        p10k = PTBL(iq + 1, it);
        p01k = PTBL(iq, it + 1);
        p11k = PTBL(iq + 1, it + 1);
        psp = FADD(FADD(FADD(p00k, FMUL(FSUB(p10k, p00k), pp1)),
                        FMUL(FSUB(p01k, p00k), qq1)),
                   FMUL(FMUL(FADD(FSUB(FSUB(p00k, p10k), p01k), p11k), pp1), qq1));
        apes = gfk_pow(FDIV(1.e5f, psp), capa);
        thesp = FMUL(thbt, gfk_exp(FDIV(FMUL(FMUL(elocp, qbt), apes), thbt)));
        /* CHOOSE CLOUD BASE AS MODEL LEVEL JUST BELOW PSP, :613 */
        for (l = 1; l <= lmh - 1; ++l) {
            p = PRS(l);
            if (p < psp && p >= BMJ_PQM) lbot = l + 1;
        }
        pbot = PRS(lbot);
        if (pbot >= pbtmx || lbot >= lmh) {
            for (l = 1; l <= lmh - 1; ++l) {
                p = PRS(l);
                if (p < pbtmx) lbot = l;
            }
            pbot = PRS(lbot);
        }
        /* CLOUD TOP COMPUTATION, :634 */
        ltop = lbot;
        ptop = pbot;
        for (l = lmh; l >= 1; --l) W(S_THES, l) = thesp;
        /* FIRST ENTROPY CHECK, :719 */
        for (l = 1; l <= lmh; ++l) { W(S_CPE, l) = 0.0f; W(S_DTV, l) = 0.0f; }
        dentpy = 0.0f;
        l = kb;
        plo = PRS(l);
        trmlo = 0.0f;
        capetrigr = FDIV(FMUL(-0.0f, BMJ_PONE), TT(lbot));   /* DTPtrigr/T(LBOT) */
        /* Below cloud base, :736 */
        if (kb > lbot) {
            bool jumped = false;
            for (l = kb - 1; l >= lbot + 1; --l) {
                pup = PRS(l);
                tup = FDIV(thbt, W(S_APE, l));
                dp = FSUB(plo, pup);
                {
                    float tql = FMUL(TT(l), FADD(FMUL(W(S_Q, l), 0.608f), 1.0f));
                    trmup = FDIV(FMUL(FSUB(FMUL(tup, FADD(FMUL(qbt, 0.608f), 1.0f)), tql), 0.5f), tql);
                }
                W(S_DTV, l) = FADD(trmlo, trmup);
                dentpy = FADD(FMUL(W(S_DTV, l), dp), dentpy);
                W(S_CPE, l) = dentpy;
                if (dentpy < capetrigr) { jumped = true; break; }
                plo = pup;
                trmlo = trmup;
            }
            if (jumped) goto L170;
        } else {
            l = lbot + 1;
            plo = PRS(l);
            tup = FDIV(thbt, W(S_APE, l));
            {
                float tql = FMUL(TT(l), FADD(FMUL(W(S_Q, l), 0.608f), 1.0f));
                trmlo = FDIV(FMUL(FSUB(FMUL(tup, FADD(FMUL(qbt, 0.608f), 1.0f)), tql), 0.5f), tql);
            }
        }
        /* At cloud base, :762 */
        l = lbot;
        pup = psp;
        tup = FDIV(thbt, apes);
        tsp = FADD(FMUL(FDIV(FSUB(TT(l + 1), TT(l)), FSUB(plo, pbot)), FSUB(pup, pbot)), TT(l));
        qsp = FADD(FMUL(FDIV(FSUB(W(S_Q, l + 1), W(S_Q, l)), FSUB(plo, pbot)), FSUB(pup, pbot)), W(S_Q, l));
        dp = FSUB(plo, pup);
        {
            float tqs = FMUL(tsp, FADD(FMUL(qsp, 0.608f), 1.0f));
            trmup = FDIV(FMUL(FSUB(FMUL(tup, FADD(FMUL(qbt, 0.608f), 1.0f)), tqs), 0.5f), tqs);
        }
        W(S_DTV, l) = FADD(trmlo, trmup);
        dentpy = FADD(FMUL(W(S_DTV, l), dp), dentpy);
        W(S_CPE, l) = dentpy;
        W(S_DTV, l) = FMUL(W(S_DTV, l), dp);
        plo = pup;
        trmlo = trmup;
        pup = PRS(l);
        tup = bmj_ttblex(tab, !(pup < BMJ_PLQ), pup, W(S_THES, l), c);
        qup = FMUL(FDIV(BMJ_PQ0, pup), gfk_exp(FDIV(FMUL(BMJ_A2, FSUB(tup, BMJ_A3)), FSUB(tup, BMJ_A4))));
        qwat = FSUB(qbt, qup);
        dp = FSUB(plo, pup);
        {
            float tql = FMUL(TT(l), FADD(FMUL(W(S_Q, l), 0.608f), 1.0f));
            trmup = FDIV(FMUL(FSUB(FMUL(tup, FSUB(FADD(FMUL(qup, 0.608f), 1.0f), qwat)), tql), 0.5f), tql);
        }
        dentpy = FADD(FMUL(FADD(trmlo, trmup), dp), dentpy);
        W(S_CPE, l) = dentpy;
        W(S_DTV, l) = FDIV(FADD(W(S_DTV, l), FMUL(FADD(trmlo, trmup), dp)),
                           FSUB(PRS(lbot + 1), PRS(lbot)));
        if (dentpy < capetrigr) goto L170;
        plo = pup;
        trmlo = trmup;
        /* In cloud above cloud base, :810 */
        for (l = lbot - 1; l >= 1; --l) {
            pup = PRS(l);
            tup = bmj_ttblex(tab, !(pup < BMJ_PLQ), pup, W(S_THES, l), c);
            qup = FMUL(FDIV(BMJ_PQ0, pup), gfk_exp(FDIV(FMUL(BMJ_A2, FSUB(tup, BMJ_A3)), FSUB(tup, BMJ_A4))));
            qwat = FSUB(qbt, qup);
            dp = FSUB(plo, pup);
            {
                float tql = FMUL(TT(l), FADD(FMUL(W(S_Q, l), 0.608f), 1.0f));
                trmup = FDIV(FMUL(FSUB(FMUL(tup, FSUB(FADD(FMUL(qup, 0.608f), 1.0f), qwat)), tql), 0.5f), tql);
            }
            W(S_DTV, l) = FADD(trmlo, trmup);
            dentpy = FADD(FMUL(W(S_DTV, l), dp), dentpy);
            W(S_CPE, l) = dentpy;
            if (dentpy < capetrigr) goto L170;
            plo = pup;
            trmlo = trmup;
        }
L170:
        ltp1 = kb;
        cape = 0.0f;
        for (l = kb; l >= 1; --l) {
            if (W(S_CPE, l) < capetrigr) {
                break;
            } else if (W(S_CPE, l) > cape) {
                ltp1 = l;
                cape = W(S_CPE, l);
            }
        }
        ltop = min(ltp1, lbot);
        if (cape > capecnv) {
            capecnv = cape;
            pspcnv = psp;
            thbtcnv = thbt;
            lbotcnv = lbot;
            ltopcnv = ltop;
            for (l = lmh; l >= 1; --l) {
                W(S_CPECNV, l) = W(S_CPE, l);
                W(S_DTVCNV, l) = W(S_DTV, l);
                W(S_THESCNV, l) = W(S_THES, l);
            }
        }
    }                                                   /* max_buoy_loop */

    /* ------------------- MAXIMUM INSTABILITY, :886 --------------------- */
    if (capecnv > 0.0f) {
        psp = pspcnv;
        thbt = thbtcnv;
        lbot = lbotcnv;
        ltop = ltopcnv;
        pbot = PRS(lbot);
        ptop = PRS(ltop);
        for (l = lmh; l >= 1; --l) {
            W(S_CPE, l) = W(S_CPECNV, l);
            W(S_DTV, l) = W(S_DTVCNV, l);
            W(S_THES, l) = W(S_THESCNV, l);
        }
    }
    /* Quick exit if cloud is too thin or no CAPE is present, :906 */
    if (ptop > FSUB(pbot, BMJ_PNO) || ltop > lbot - 2 || capecnv <= 0.0f) {
        lbot = 0;
        ltop = nz;                                      /* KTE */
        pbot = PRS(lmh);
        ptop = pbot;
        cldefi = FADD(FMUL(c.avgefi, sm), FMUL(BMJ_STEFI, FSUB(1.0f, sm)));
        path = BMJ_PATH_QUICK_EXIT;
        goto L800;
    }
    depth = FSUB(pbot, ptop);
    if (depth >= depmin) {
        deep = true;
    } else {
        shallow = true;
        goto L600;
    }

    /* =================== DEEP CONVECTION, :935 ========================= */
    lb = lbot;
    efi = cldefi;
    for (l = 1; l <= lmh; ++l) {
        W(S_DIFT, l) = 0.0f;
        W(S_DIFQ, l) = 0.0f;
        tkl = TT(l);
        W(S_TREFK, l) = tkl;
        qkl = W(S_Q, l);
        W(S_QREFK, l) = qkl;
        pkl = PRS(l);
        W(S_PSK, l) = pkl;
        apekl = W(S_APE, l);
        {
            float tref = bmj_ttblex(tab, !(pkl < BMJ_PLQ), pkl, W(S_THES, l), c);
            W(S_THERK, l) = FMUL(tref, apekl);
        }
    }
    /* TK=T, QK=Q, PK=PRSMID, APEK=APE, EL=ELWV: aliases, never written. */
    ltp1 = ltop + 1;
    lbm1 = lb - 1;
    pkb = PRS(lb);
    pkt = PRS(ltop);
    stabdl = FADD(FMUL(FSUB(efi, BMJ_EFIMN), c.slopst), BMJ_STABDS);
    /* TEMPERATURE REFERENCE PROFILE BELOW FREEZING LEVEL, :992 */
    l0 = lb;
    pk0 = PRS(lb);
    trefkx = W(S_TREFK, lb);
    therkx = W(S_THERK, lb);
    apekxx = W(S_APE, lb);
    therky = W(S_THERK, lbm1);
    apekxy = W(S_APE, lbm1);
    {
        bool frozen = false;
        for (l = lbm1; l >= ltop; --l) {
            if (TT(l + 1) < tfrz) { frozen = true; break; }
            trefkx = FDIV(FADD(FMUL(FSUB(therky, therkx), stabdl), FMUL(trefkx, apekxx)), apekxy);
            W(S_TREFK, l) = trefkx;
            apekxx = apekxy;
            therkx = therky;
            /* APEK(L-1)/THERK(L-1): at L = 1 the Fortran reads index 0, one
             * past the array start; the value is dead (the loop ends), so the
             * port does not read it. */
            if (l > 1) {
                apekxy = W(S_APE, l - 1);
                therky = W(S_THERK, l - 1);
            }
            l0 = l;
            pk0 = PRS(l0);
        }
        if (frozen) {
            /* TEMPERATURE REFERENCE PROFILE ABOVE FREEZING LEVEL, :1021 */
            l0m1 = l0 - 1;
            rdp0t = FDIV(1.0f, FSUB(pk0, pkt));
            dthem = FSUB(W(S_THERK, l0), FMUL(W(S_TREFK, l0), W(S_APE, l0)));
            for (l = ltop; l <= l0m1; ++l) {
                W(S_TREFK, l) = FDIV(FSUB(W(S_THERK, l),
                                          FMUL(FMUL(FSUB(PRS(l), pkt), dthem), rdp0t)),
                                     W(S_APE, l));
            }
        }
    }
    /* DEEP CONVECTION REFERENCE HUMIDITY PROFILE, :1037 */
    depwl = FSUB(pkb, pk0);
    depth = FMUL(FMUL(BMJ_PFRZ, psfc), c.rsfcp);
    sm1 = FSUB(1.0f, sm);
    pbotfc = 1.0f;
    for (itrefi = 1; itrefi <= 3; ++itrefi) {           /* cloud_efficiency */
        float de = FSUB(efi, BMJ_EFIMN);
        dspbk = FADD(FMUL(FADD(FMUL(de, c.slopbs), FMUL(c.dspbss, pbotfc)), sm),
                     FMUL(FADD(FMUL(de, c.slopbl), FMUL(c.dspbsl, pbotfc)), sm1));
        dsp0k = FADD(FMUL(FADD(FMUL(de, c.slop0s), FMUL(c.dsp0ss, pbotfc)), sm),
                     FMUL(FADD(FMUL(de, c.slop0l), FMUL(c.dsp0sl, pbotfc)), sm1));
        dsptk = FADD(FMUL(FADD(FMUL(de, c.slopts), FMUL(c.dsptss, pbotfc)), sm),
                     FMUL(FADD(FMUL(de, c.sloptl), FMUL(c.dsptsl, pbotfc)), sm1));
        for (l = ltop; l <= lb; ++l) {
            float pkl2 = PRS(l);
            if (depwl >= depth) {
                if (l < l0) {
                    dsp = FDIV(FADD(FMUL(FSUB(pk0, pkl2), dsptk), FMUL(FSUB(pkl2, pkt), dsp0k)), FSUB(pk0, pkt));
                } else {
                    dsp = FDIV(FADD(FMUL(FSUB(pkb, pkl2), dsp0k), FMUL(FSUB(pkl2, pk0), dspbk)), FSUB(pkb, pk0));
                }
            } else {
                dsp = dsp0k;
                if (l < l0) {
                    dsp = FDIV(FADD(FMUL(FSUB(pk0, pkl2), dsptk), FMUL(FSUB(pkl2, pkt), dsp0k)), FSUB(pk0, pkt));
                }
            }
            W(S_PSK, l) = FADD(pkl2, dsp);
            W(S_APESK, l) = gfk_pow(FDIV(1.e5f, W(S_PSK, l)), capa);
            if (pkl2 > BMJ_PQM) {
                float thsk = FMUL(W(S_TREFK, l), W(S_APE, l));
                float apesk = W(S_APESK, l);
                W(S_QREFK, l) = FMUL(FDIV(BMJ_PQ0, W(S_PSK, l)),
                    gfk_exp(FDIV(FMUL(BMJ_A2, FSUB(thsk, FMUL(BMJ_A3, apesk))),
                                 FSUB(thsk, FMUL(BMJ_A4, apesk)))));
            } else {
                W(S_QREFK, l) = W(S_Q, l);
            }
        }
        /* ENTHALPY CONSERVATION INTEGRAL, :1112 */
        for (iter = 1; iter <= 2; ++iter) {
            sumde = 0.0f;
            sumdp = 0.0f;
            dhdt = 0.0f;
            for (l = ltop; l <= lb; ++l) {
                float dprs = W(S_DPRS, l);
                sumde = FADD(FMUL(FADD(FMUL(FSUB(TT(l), W(S_TREFK, l)), cp),
                                       FMUL(FSUB(W(S_Q, l), W(S_QREFK, l)), elwv)), dprs), sumde);
                float x = FSUB(FDIV(FMUL(W(S_TREFK, l), W(S_APE, l)), W(S_APESK, l)), BMJ_A4);
                dhdt = FADD(FMUL(FADD(FDIV(FMUL(W(S_QREFK, l), a23m4l), FMUL(x, x)), cp), dprs), dhdt);
                sumdp = FADD(sumdp, dprs);
            }
            hcorr = FDIV(sumde, FSUB(sumdp, W(S_DPRS, ltop)));
            dhdt = FDIV(dhdt, FSUB(sumdp, W(S_DPRS, ltop)));
            lcor = ltop + 1;
            lqm = 1;
            for (l = 1; l <= lb; ++l) {
                if (PRS(l) <= BMJ_PQM) lqm = l;
            }
            if (lcor <= lqm) {
                for (l = lcor; l <= lqm; ++l)
                    W(S_TREFK, l) = FADD(W(S_TREFK, l), FMUL(hcorr, rcp));
                lcor = lqm + 1;
            }
            for (l = lcor; l <= lb; ++l) {
                W(S_TREFK, l) = FADD(FDIV(hcorr, dhdt), W(S_TREFK, l));
                thskl = FMUL(W(S_TREFK, l), W(S_APE, l));
                float apesk = W(S_APESK, l);
                W(S_QREFK, l) = FMUL(FDIV(BMJ_PQ0, W(S_PSK, l)),
                    gfk_exp(FDIV(FMUL(BMJ_A2, FSUB(thskl, FMUL(BMJ_A3, apesk))),
                                 FSUB(thskl, FMUL(BMJ_A4, apesk)))));
            }
        }
        /* HEATING, MOISTENING, PRECIPITATION, :1161 */
        avrgt = 0.0f; preck = 0.0f; pdqd = 0.0f; prwd = 0.0f; dsq = 0.0f; dst = 0.0f;
        for (l = ltop; l <= lb; ++l) {
            float dprs = W(S_DPRS, l);
            tkl = TT(l);
            diftl = FMUL(FSUB(W(S_TREFK, l), tkl), tauk);
            difql = FMUL(FSUB(W(S_QREFK, l), W(S_Q, l)), tauk);
            avrgtl = FADD(FADD(tkl, tkl), diftl);
            dpot = FDIV(dprs, avrgtl);
            dst = FADD(FMUL(diftl, dpot), dst);
            dsq = FADD(FMUL(FMUL(difql, elwv), dpot), dsq);
            avrgt = FADD(FMUL(avrgtl, dprs), avrgt);
            preck = FADD(FMUL(diftl, dprs), preck);
            pdqd = FADD(FMUL(FSUB(W(S_Q, l), W(S_QREFK, l)), dprs), pdqd);
            prwd = FADD(FMUL(W(S_Q, l), dprs), prwd);
            W(S_DIFT, l) = diftl;
            W(S_DIFQ, l) = difql;
        }
        dst = FMUL(FADD(dst, dst), cp);
        dsq = FADD(dsq, dsq);
        dentpy = FADD(dst, dsq);
        avrgt = FDIV(avrgt, FADD(sumdp, sumdp));
        drheat = FDIV(FMUL(FADD(FMUL(preck, sm), FMUL(fmaxf(1.e-7f, preck), FSUB(1.0f, sm))), cp), avrgt);
        drheat = fmaxf(drheat, 1.e-20f);
        efi = FDIV(FMUL(BMJ_EFIFC, dentpy), drheat);
        efi = fminf(efi, 1.0f);
        efi = fmaxf(efi, BMJ_EFIMN);
    }                                                   /* cloud_efficiency */

    if (dentpy >= BMJ_EPSNTP && preck > BMJ_EPSPR) {
        /* DEEP CONVECTION, :1206 */
        cldefi = efi;
        fefi = FADD(BMJ_EFMNT, FMUL(c.slope, FSUB(efi, BMJ_EFIMN)));
        fefi = FDIV(FMUL(FSUB(dentpy, BMJ_EPSNTP), fefi), dentpy);
        preck = FMUL(preck, fefi);
        cup = FMUL(preck, cprlg);
        pcpcol = cup;
        dqcol = FDIV(pdqd, g);
        pwcol = FDIV(prwd, g);
        dqcolmin = FDIV(FMUL(BMJ_CRMN, BMJ_TREL), FMUL(fefi, 86400.0f));
        for (l = ltop; l <= lb; ++l) {
            W(S_DTDT, l) = FMUL(FMUL(W(S_DIFT, l), fefi), rdtcnvc);
            W(S_DQDT, l) = FMUL(FMUL(W(S_DIFQ, l), fefi), rdtcnvc);
        }
        path = BMJ_PATH_DEEP;
    } else {
        /* REDUCE THE CLOUD TOP -> shallow, :1244 */
        cldefi = FADD(FMUL(BMJ_EFIMN, sm), FMUL(BMJ_STEFI, FSUB(1.0f, sm)));
        ptpk = fmaxf(BMJ_PSHU, FSUB(PRS(lbot), depmin));
        for (l = 1; l <= lmh; ++l) {
            if (PRS(l) <= ptpk) ltop = l + 1;
        }
        ltp1 = lbot;
        for (l = lbot - 1; l >= ltop; --l) {
            if (W(S_DTV, l) > 0.0f) ltp1 = l; else break;
        }
        ltop = min(ltp1, lbot);
        ptop = PRS(ltop);
        shallow = true;
        deep = false;
        path = BMJ_PATH_DEEP_TO_SHALLOW;
    }

L600:
    if (!shallow) goto L800;

    /* ===================== SHALLOW CONVECTION, :1378 =================== */
    for (l = 1; l <= lmh; ++l) {
        tkl = TT(l);
        W(S_TREFK, l) = tkl;
        qkl = W(S_Q, l);
        W(S_QREFK, l) = qkl;
        pkl = PRS(l);
        W(S_QSATK, l) = FMUL(FDIV(BMJ_PQ0, pkl),
            gfk_exp(FDIV(FMUL(BMJ_A2, FSUB(tkl, BMJ_A3)), FSUB(tkl, BMJ_A4))));
        apekl = W(S_APE, l);
        thvmkl = FMUL(FMUL(tkl, apekl), FADD(FMUL(qkl, d608), 1.0f));
        W(S_THVREF, l) = thvmkl;
    }
    /* Raise cloud top if avg RH>RHSHmax and CAPE>0, :1405 */
    {
        float pkb2 = PRS(lbot);
        tlev2 = FMUL(TT(lbot), gfk_pow(FDIV(FSUB(pkb2, BMJ_PONE), pkb2), capa));
        qsat1 = FMUL(FDIV(BMJ_PQ0, pkb2), gfk_exp(FDIV(FMUL(BMJ_A2, FSUB(TT(lbot), BMJ_A3)), FSUB(TT(lbot), BMJ_A4))));
        qsat2 = FMUL(FDIV(BMJ_PQ0, FSUB(pkb2, BMJ_PONE)), gfk_exp(FDIV(FMUL(BMJ_A2, FSUB(tlev2, BMJ_A3)), FSUB(tlev2, BMJ_A4))));
    }
    rhshmax = FDIV(qsat2, qsat1);
    sumdp = 0.0f;
    rhavg = 0.0f;
    for (l = lbot; l >= ltop; --l) {
        rhavg = FADD(rhavg, FDIV(FMUL(W(S_DPRS, l), W(S_Q, l)), W(S_QSATK, l)));
        sumdp = FADD(sumdp, W(S_DPRS, l));
    }
    if (FDIV(rhavg, sumdp) > rhshmax) {
        ltsh = ltop;
        for (l = ltop - 1; l >= 1; --l) {
            rhavg = FADD(rhavg, FDIV(FMUL(W(S_DPRS, l), W(S_Q, l)), W(S_QSATK, l)));
            sumdp = FADD(sumdp, W(S_DPRS, l));
            if (W(S_CPE, l) > 0.0f) ltsh = l; else break;
            if (FDIV(rhavg, sumdp) <= rhshmax) break;
            if (PRS(l) <= BMJ_PSHU) break;
        }
        ltop = ltsh;
    }
    /* SHALLOW CLOUD TOP, :1436 */
    lbm1 = lbot - 1;
    ptpk = ptop;
    ltp1 = ltop - 1;
    depth = FSUB(pbot, ptop);
    /* LTP1 = 0 would make the Fortran read T(0) below; it needs LTOP = 1,
     * a shallow top at the model's top layer, which PSHU (450 hPa) rules
     * out on any column whose top is above 450 hPa.  The clamp only keeps
     * the device read inside the column. */
    if (ltp1 < 1) ltp1 = 1;
    if (ptop > FSUB(pbot, BMJ_PNO) || ltop > lbot - 2) {
        lbot = 0;
        ltop = nz;
        ptop = pbot;
        path |= BMJ_PATH_SH_THIN;
        goto L800;
    }
    /* SCALING POTENTIAL TEMPERATURE & TABLE INDEX AT TOP, :1463 */
    thtpk = FMUL(TT(ltp1), W(S_APE, ltp1));
    tthk = FMUL(FSUB(thtpk, BMJ_THL), c.rdth);
    qqk = FSUB(tthk, truncf(tthk));
    it = (int)tthk + 1;
    if (it < 1) { it = 1; qqk = 0.0f; }
    if (it >= BMJ_JTB) { it = BMJ_JTB - 1; qqk = 0.0f; }
    bqs00k = tab[BMJ_OFF_QS0 + it - 1];
    sqs00k = tab[BMJ_OFF_SQS + it - 1];
    bqs10k = tab[BMJ_OFF_QS0 + it];
    sqs10k = tab[BMJ_OFF_SQS + it];
    bqk = FADD(FMUL(FSUB(bqs10k, bqs00k), qqk), bqs00k);
    sqk = FADD(FMUL(FSUB(sqs10k, sqs00k), qqk), sqs00k);
    tqk = FMUL(FDIV(FSUB(W(S_Q, ltp1), bqk), sqk), c.rdq);
    ppk = FSUB(tqk, truncf(tqk));
    iq = (int)tqk + 1;
    if (iq < 1) { iq = 1; ppk = 0.0f; }
    if (iq >= BMJ_ITB) { iq = BMJ_ITB - 1; ppk = 0.0f; }
    /* CLOUD TOP SATURATION POINT PRESSURE, :1509 */
    part1 = FMUL(FSUB(PTBL(iq + 1, it), PTBL(iq, it)), ppk);
    part2 = FMUL(FSUB(PTBL(iq, it + 1), PTBL(iq, it)), qqk);
    part3 = FMUL(FMUL(FADD(FSUB(FSUB(PTBL(iq, it), PTBL(iq + 1, it)), PTBL(iq, it + 1)),
                           PTBL(iq + 1, it + 1)), ppk), qqk);
    ptpk = FADD(FADD(FADD(PTBL(iq, it), part1), part2), part3);
    dpmix = FSUB(ptpk, psp);
    if (fabsf(dpmix) < 3000.0f) dpmix = -3000.0f;
    /* TEMPERATURE PROFILE SLOPE, :1520 */
    smix = FMUL(FDIV(FSUB(thtpk, thbt), dpmix), BMJ_STABS);
    trefkx = W(S_TREFK, lbot + 1);
    pkxxxx = PRS(lbot + 1);
    pkxxxy = PRS(lbot);
    apekxx = W(S_APE, lbot + 1);
    apekxy = W(S_APE, lbot);
    lmid = (int)FMUL(0.5f, (float)(lbot + ltop));
    for (l = lbot; l >= ltop; --l) {
        trefkx = FDIV(FADD(FMUL(FSUB(pkxxxy, pkxxxx), smix), FMUL(trefkx, apekxx)), apekxy);
        W(S_TREFK, l) = trefkx;
        if (l <= lmid) W(S_TREFK, l) = fmaxf(W(S_TREFK, l), FADD(TT(l), BMJ_DTSHAL));
        apekxx = apekxy;
        pkxxxx = pkxxxy;
        /* APEK(L-1)/PK(L-1) at L = 1 would be index 0; dead, not read. */
        if (l > 1) {
            apekxy = W(S_APE, l - 1);
            pkxxxy = PRS(l - 1);
        }
    }
    /* TEMPERATURE REFERENCE PROFILE CORRECTION, :1543 */
    sumdt = 0.0f;
    sumdp = 0.0f;
    for (l = ltop; l <= lbot; ++l) {
        sumdt = FADD(FMUL(FSUB(TT(l), W(S_TREFK, l)), W(S_DPRS, l)), sumdt);
        sumdp = FADD(sumdp, W(S_DPRS, l));
    }
    rdpsum = FDIV(1.0f, sumdp);
#define FPK(L) W(S_THERK, L)          /* FPK aliases the deep arm's THERK */
    FPK(lbot) = W(S_TREFK, lbot);
    tcorr = FMUL(sumdt, rdpsum);
    for (l = ltop; l <= lbot; ++l) {
        trfkl = FADD(W(S_TREFK, l), tcorr);
        W(S_TREFK, l) = trfkl;
        FPK(l) = trfkl;
    }
    /* HUMIDITY PROFILE EQUATIONS, :1564 */
    psum = 0.0f; qsum = 0.0f; potsum = 0.0f; qotsum = 0.0f; otsum = 0.0f; dst = 0.0f;
    fptk = FPK(ltop);
    for (l = ltop; l <= lbot; ++l) {
        float dprs = W(S_DPRS, l);
        float qk = W(S_Q, l);
        float tk = TT(l);
        dpkl = FSUB(FPK(l), fptk);
        psum = FADD(FMUL(dpkl, dprs), psum);
        qsum = FADD(FMUL(qk, dprs), qsum);
        rtbar = FDIV(2.0f, FADD(W(S_TREFK, l), tk));
        otsum = FADD(FMUL(dprs, rtbar), otsum);
        potsum = FADD(FMUL(FMUL(dpkl, rtbar), dprs), potsum);
        qotsum = FADD(FMUL(FMUL(qk, rtbar), dprs), qotsum);
        dst = FADD(FDIV(FMUL(FMUL(FSUB(W(S_TREFK, l), tk), rtbar), dprs), elwv), dst);
    }
    psum = FMUL(psum, rdpsum);
    qsum = FMUL(qsum, rdpsum);
    rotsum = FDIV(1.0f, otsum);
    potsum = FMUL(potsum, rotsum);
    qotsum = FMUL(qotsum, rotsum);
    dst = FMUL(FMUL(dst, rotsum), cp);
    /* ENSURE POSITIVE ENTROPY CHANGE, :1601 */
    if (dst > 0.0f) {
        lbot = 0; ltop = nz; ptop = pbot;
        path |= BMJ_PATH_SH_DST;
        goto L800;
    } else {
        dstq = FMUL(dst, BMJ_EPSDN);
    }
    /* CHECK FOR ISOTHERMAL ATMOSPHERE, :1614 */
    den = FSUB(potsum, psum);
    if (FDIV(-den, psum) < 5.e-5f) {
        lbot = 0; ltop = nz; ptop = pbot;
        path |= BMJ_PATH_SH_ISOTHERMAL;
        goto L800;
    } else {
        dqref = FDIV(FSUB(FSUB(qotsum, dstq), qsum), den);
    }
    if (dqref < 0.0f) {
        lbot = 0; ltop = nz; ptop = pbot;
        path |= BMJ_PATH_SH_DQREF;
        goto L800;
    }
    qrftp = FSUB(qsum, FMUL(dqref, psum));
    for (l = ltop; l <= lbot; ++l) {
        float qk = W(S_Q, l);
        float tk = TT(l);
        qrfkl = FADD(FMUL(FSUB(FPK(l), fptk), dqref), qrftp);
        tnew = FADD(FMUL(FSUB(W(S_TREFK, l), tk), tauksc), tk);
        W(S_QSATK, l) = FMUL(FDIV(BMJ_PQ0, PRS(l)),
            gfk_exp(FDIV(FMUL(BMJ_A2, FSUB(tnew, BMJ_A3)), FSUB(tnew, BMJ_A4))));
        qnew = FADD(FMUL(FSUB(qrfkl, qk), tauksc), qk);
        if (qnew < FMUL(W(S_QSATK, l), BMJ_RHLSC)) {
            lbot = 0; ltop = nz; ptop = pbot;
            path |= BMJ_PATH_SH_DRY;
            goto L800;
        }
        if (qnew > FMUL(W(S_QSATK, l), BMJ_RHHSC)) {
            lbot = 0; ltop = nz; ptop = pbot;
            path |= BMJ_PATH_SH_MOIST;
            goto L800;
        }
        W(S_THVREF, l) = FMUL(FMUL(W(S_TREFK, l), W(S_APE, l)), FADD(FMUL(qrfkl, d608), 1.0f));
        W(S_QREFK, l) = qrfkl;
    }
    /* ELIMINATE IMPOSSIBLE SLOPES (BETTS, DTHETA/DQ), :1691 */
    for (l = ltop; l <= lbot; ++l) {
        /* THVREF(L-1) at L = LTOP = 1 would read index 0; the shallow top
         * is bounded below PSHU (450 hPa) so LTOP >= 2 on any column whose
         * model top is above 450 hPa, which the engine's vertical contract
         * guarantees; the guard keeps the read in bounds regardless. */
        float thvm1 = (l > 1) ? W(S_THVREF, l - 1) : W(S_THVREF, l);
        float pm1 = (l > 1) ? PRS(l - 1) : PRS(l);
        dtdp = FDIV(FSUB(thvm1, W(S_THVREF, l)), FSUB(PRS(l), pm1));
        if (dtdp < BMJ_EPSDT) {
            lbot = 0; ltop = nz; ptop = pbot;
            path |= BMJ_PATH_SH_SLOPE;
            goto L800;
        }
    }
    /* RELAXATION TOWARD REFERENCE PROFILES, :1707 */
    for (l = ltop; l <= lbot; ++l) {
        W(S_DTDT, l) = FMUL(FMUL(FSUB(W(S_TREFK, l), TT(l)), tauksc), rdtcnvc);
        W(S_DQDT, l) = FMUL(FMUL(FSUB(W(S_QREFK, l), W(S_Q, l)), tauksc), rdtcnvc);
    }
    path |= BMJ_PATH_SHALLOW;
#undef FPK
#undef PTBL
L800:;
    (void)ltp1; (void)lbm1; (void)depth; (void)ptpk; (void)ptop; (void)p;
    }
    /* ===================== end BMJ, back in BMJDRV, :243 ================ */
    for (int k = 1; k <= nz; ++k) {
        int kflip = nz + 1 - k;
        LEV(rthcuten_a, k - 1) = FDIV(W(S_DTDT, kflip), LEV(pi_a, k - 1));
        float omq = FSUB(1.0f, W(S_Q, kflip));
        LEV(rqvcuten_a, k - 1) = FDIV(W(S_DQDT, kflip), FMUL(omq, omq));
    }
    raincv = FDIV(FMUL(pcpcol, 1.e3f), (float)stepcu);
    pratec = FDIV(FMUL(pcpcol, 1.e3f), FMUL((float)stepcu, dt));
    const float cutop = (float)(nz + 1 - ltop);
    const float cubot = (float)(nz + 1 - lbot);

    if (rad_feedback) {
        if (dqcol > dqcolmin) {
            /* :273-345, WRF K is bottom-up 1..nz here (engine k = K-1) */
            int ttop = __float2int_rn(cutop);
            int bbot = __float2int_rn(cubot);
            float cumx, mpvpr, factl, qcis, mcol;
            float prrt = FDIV(FMUL(pratec, 86400.0f), BMJ_CRMN);
            /* RRP=0.8/(LOG(CRMX/CRMN)), the LOG folded at compile time. */
            const float rrp = FDIV(0.8f, __uint_as_float(BMJ_LOG_CRMX_CRMN_BITS));
            /* PRRT<CRMX/CRMN: the quotient is folded, correctly rounded. */
            if (prrt < FDIV(85.0f, BMJ_CRMN)) {
                cumx = FMUL(rrp, gfk_log(prrt));
            } else {
                cumx = 0.8f;
            }
            /* PAVG=1./(PEPS*3.)**2 */
            const float pe3 = FMUL(c.peps, 3.0f);
            const float pavg = FDIV(1.0f, FMUL(pe3, pe3));
            const float rpeps = FDIV(1.0f, c.peps);
            const float pmt = LEV(pmid_a, ttop - 1);
            const float pmb = LEV(pmid_a, bbot - 1);
            for (int k = 1; k <= nz; ++k) {
                if (k >= bbot && k <= ttop) {
                    W(S_JPR, k) = FMUL(rpeps, FDIV(FSUB(LEV(pmid_a, k - 1), pmt), FSUB(pmb, pmt)));
                } else {
                    W(S_JPR, k) = 0.0f;
                }
            }
            for (int k = 1; k <= nz; ++k) W(S_PVPR, k) = 0.0f;
            if (W(S_JPR, bbot) < 17.0f) {
                for (int k = bbot; k <= ttop; ++k) {
                    float jpr = W(S_JPR, k);
                    /* DECLARED DIVERGENCE: gfk_tgamma is correctly rounded */
                    W(S_PVPR, k) = FDIV(gfk_pow(pavg, jpr), gfk_tgamma(FADD(jpr, 1.0f)));
                }
            } else {
                for (int k = bbot; k <= ttop; ++k) {
                    float jpr = W(S_JPR, k);
                    factl = FADD(FADD(FSUB(FMUL(jpr, gfk_log(jpr)), jpr),
                                      FMUL(0.5f, gfk_log(FMUL(FMUL(2.0f, jpr), BMJ_ACOS_M1)))),
                                 gfk_log(FADD(FADD(1.0f, FDIV(1.0f, FMUL(12.0f, jpr))),
                                              FDIV(1.0f, FMUL(288.0f, FMUL(jpr, jpr))))));
                    W(S_PVPR, k) = gfk_exp(FSUB(FMUL(jpr, gfk_log(FMUL(1.0f, pavg))), factl));
                }
            }
            mpvpr = -3.40282347e38f;                    /* MAXVAL start */
            for (int k = 1; k <= nz; ++k) mpvpr = fmaxf(mpvpr, W(S_PVPR, k));
            for (int k = bbot; k <= ttop; ++k) W(S_PVPR, k) = FDIV(W(S_PVPR, k), mpvpr);
            for (int k = 1; k <= nz; ++k) LEV(ccldfra_a, k - 1) = FMUL(cumx, W(S_PVPR, k));
            qcis = 0.0f;
            mcol = 0.0f;
            for (int k = (int)cubot; k <= (int)cutop; ++k) {
                int kflip = nz + 1 - k;
                mcol = FADD(FMUL(FMUL(FMUL(LEV(rho_a, k - 1), LEV(dz8w_a, k - 1)), W(S_Q, kflip)),
                                 LEV(ccldfra_a, k - 1)), mcol);
            }
            convcld = FMUL(gfk_pow(pwcol, 0.5f), gfk_pow(FSUB(dqcol, dqcolmin), 0.5f));
            qcis = FDIV(convcld, mcol);
            for (int k = 1; k <= nz; ++k) {
                int kflip = nz + 1 - k;
                float qk = W(S_Q, kflip);
                float val = FDIV(FMUL(FMUL(qcis, qk), LEV(ccldfra_a, k - 1)), FSUB(1.0f, qk));
                if (LEV(t_a, k - 1) >= tfrz) {      /* TCOL(KFLIP) = T(I,K,J) */
                    LEV(qiconv_a, k - 1) = 0.0f;
                    LEV(qcconv_a, k - 1) = val;
                } else {
                    LEV(qiconv_a, k - 1) = val;
                    LEV(qcconv_a, k - 1) = 0.0f;
                }
            }
            (void)factl;
        }
    }
    raincv_a[i] = raincv;
    pratec_a[i] = pratec;
    cutop_a[i] = cutop;
    cubot_a[i] = cubot;
    convcld_a[i] = convcld;
    cldefi_a[i] = cldefi;
    path_a[i] = path;
#undef LEV
#undef W
#undef TT
#undef PRS
}

// WRF module_radiation_driver.F:1583-1587. Temporary radiation copies only.
extern "C" __global__ void column_cumulus_cloud_merge(
    float* qc, float* qi, float* fraction,
    const float* conv_qc, const float* conv_qi, const float* conv_fraction,
    const long long n)
{
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    qc[k] = FADD(qc[k], conv_qc[k]);
    qi[k] = FADD(qi[k], conv_qi[k]);
    if (fraction) {
        const float a = fraction[k], b = conv_fraction[k];
        const float upper = fminf(a, b);
        const float lower = fmaxf(0.0f, FSUB(FADD(a, b), 1.0f));
        fraction[k] = FSUB(FADD(a, b), FMUL(0.5f, FADD(upper, lower)));
    }
}
