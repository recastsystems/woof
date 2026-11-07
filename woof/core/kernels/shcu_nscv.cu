/* NCEP shallow convection, NSCV (shcu_physics = 4), one CUDA thread per
 * mass column.
 *
 * WRF v4.6.1 transcription anchors (phys/module_shcu_nscv.F):
 *   8-266      shcu_nscv: the wrapper -- dot, prsll/prsii, zii/zll, del,
 *              the icps test against RAINCV, pratesh = rain*1000/dt,
 *              HBOT/HTOP, and the tendency identities
 *   272-1369   nscv2d: the column scheme
 *   1373-1413  fpvs: saturation vapour pressure, analytic (no table)
 * Graded word for word against WRF's own scalar build of that file:
 * tools/shcu_nscv_wrf461_oracle, tests/test_shcu_nscv_parity.py.
 *
 * ==========================================================================
 * EVERY ARITHMETIC OPERATION IS A ROUNDED INTRINSIC
 * ==========================================================================
 * __fadd_rn/__fsub_rn/__fmul_rn/__fdiv_rn/__fsqrt_rn are never contracted
 * into an FMA, so the default and strict (GPUWM_WRF_EXACT=1) compiles run
 * the same operations in the same order from this one source.  The order
 * is Fortran's: left to right inside a parenthesis level, `a*b/c*d` as
 * ((a*b)/c)*d, and gfortran evaluates the module with -ffp-contract=off.
 * Integer exponents are multiply chains (`x**2` is x*x, `x**3` is (x*x)*x);
 * REAL exponents -- fpvs's tr**xa and wstar's ptem**h1 -- are powf, and
 * EXP is expf: WOOF's own float32 routines gfk_pow and gfk_exp.  SQRT is
 * the correctly rounded square root on both sides.
 *
 * The parameters nscv2d folds at compile time (c0+c1, tcrf, the cincr
 * half-span) are evaluated here at run time with the same rounded
 * operations; tools/shcu_nscv_wrf461_oracle/fold_probe.F90 records that
 * the folded and run-time values are the same words.
 *
 * ==========================================================================
 * WHAT IS DELIBERATELY NOT WRF
 * ==========================================================================
 * 1. HBOT/HTOP of a column that never enters the scheme.  shcu_nscv copies
 *    its local kbot/ktop into HBOT/HTOP for every column (:219-223), but
 *    nscv2d assigns them only to columns that enter (:450-453): a column
 *    with icps = 1 or a non-positive surface buoyancy flux hands WRF's
 *    HBOT/HTOP uninitialised stack.  Here such a column leaves HBOT/HTOP as
 *    they arrived -- the deep scheme's values, which is what GFS's shallow
 *    convection (where kbot/ktop are inout) does.  Columns that enter get
 *    WRF's values exactly, including kte+1 / 0 for a column that enters and
 *    does not trigger.
 * 2. Dead stores are not transcribed: delhbar/delqbar/deltbar/delubar/
 *    delvbar, the evaporation's dellaq/deltv/delq updates and dtmin/dtmax
 *    are written in nscv2d and read by nothing.
 * 3. A column whose kmax would pass kte (the layer below the model top
 *    still above 0.6*psfc) or whose kbm reaches kte indexes past WRF's own
 *    arrays (to(:,kte+1), zi(:,kte+1)).  No real model top gets there; the
 *    kernel refuses the column (no convection, status bit 1) instead of
 *    reading outside the column, and gpuwm/core/shcu_nscv.py raises on the
 *    bit, naming the breakage.
 *
 * ==========================================================================
 * LAYOUT
 * ==========================================================================
 * Column arrays are level-major, a[(k-1) * ncol + c] for 1-based k, so a
 * warp reads 32 consecutive floats per level.  The scheme's own column
 * arrays live in a caller-sized global workspace, lane-interleaved exactly
 * like kf.cu's (one region per warp, element (s, k) at
 * base + (s*nz + k-1)*NSCV_LANES + lane), and the launcher sizes it to the
 * columns in flight, not to the grid.
 */

#define NSCV_LANES 32
/* Workspace slot ids.  Literal because NVRTC has no __COUNTER__;
 * tests/test_shcu_nscv_parity.py re-reads them and fails on a duplicate or
 * an id past NSCV_SLOTS. */
#define NSCV_SLOTS 22
#define NS_ZL     0
#define NS_TO     1
#define NS_QO     2
#define NS_UO     3
#define NS_VO     4
#define NS_QESO   5
#define NS_HEO    6
#define NS_HESO   7
#define NS_ETA    8
#define NS_HCKO   9
#define NS_QCKO   10
#define NS_UCKO   11
#define NS_VCKO   12
#define NS_DBYO   13
#define NS_PWO    14
#define NS_DELLAL 15
#define NS_T1     16
#define NS_Q1     17
#define NS_U1     18
#define NS_V1     19
#define NS_QC2    20
#define NS_QI2    21

#define NSCV_STATUS_TOP_TOO_LOW 1
#define NSCV_STATUS_NONFINITE   2

/* Tunables and parameters of nscv2d (:414-418, :479-500).  Literals,
 * exactly as the Fortran spells them. */
#define NSCV_C0       0.002f
#define NSCV_C1       5.e-4f
#define NSCV_CINCRMAX 180.0f
#define NSCV_CINCRMIN 120.0f
#define NSCV_DTHK     25.0f
#define NSCV_H1       0.33333333f
#define NSCV_TF       233.16f
#define NSCV_TCR      263.16f
#define NSCV_CLAM     0.3f
#define NSCV_AAFAC    0.1f
#define NSCV_BETAW    0.03f
#define NSCV_EVFACT   0.3f
#define NSCV_EVFACTL  0.3f
#define NSCV_T0C      273.15f
#define NSCV_W1L (-8.e-3f)
#define NSCV_W2L (-4.e-2f)
#define NSCV_W3L (-5.e-3f)
#define NSCV_W4L (-5.e-4f)
#define NSCV_W1S (-2.e-4f)
#define NSCV_W2S (-2.e-3f)
#define NSCV_W3S (-1.e-3f)
#define NSCV_W4S (-2.e-5f)

#define NADD(a, b) __fadd_rn((a), (b))
#define NSUB(a, b) __fsub_rn((a), (b))
#define NMUL(a, b) __fmul_rn((a), (b))
#define NDIV(a, b) __fdiv_rn((a), (b))

/* The constants the shallow-cumulus driver hands the scheme
 * (module_shallowcu_driver.F:584-586), already float32, plus fpvs's
 * per-call constants, which depend only on them (:1381-1387). */
struct NscvConst {
    float cp, cliq, cvap, g, hvap, rd, rv, fv, cice, hsub, psat;
    float ttp, xa, xb, xai, xbi;
    float el2orc, eps, fact1, fact2;
};

__device__ __forceinline__ NscvConst nscv_constants(
        float cp, float cliq, float cpv, float g, float xlv, float r_d,
        float r_v, float ep_1, float cice, float xls, float psat) {
    NscvConst c;
    c.cp = cp; c.cliq = cliq; c.cvap = cpv; c.g = g; c.hvap = xlv;
    c.rd = r_d; c.rv = r_v; c.fv = ep_1; c.cice = cice; c.hsub = xls;
    c.psat = psat;
    /* fpvs :1381-1387.  t0c is nscv2d's t0c_ = 273.15 (:422). */
    c.ttp = NADD(NSCV_T0C, 0.01f);
    float dldt = NSUB(c.cvap, c.cliq);
    c.xa = NDIV(-dldt, c.rv);
    c.xb = NADD(c.xa, NDIV(c.hvap, NMUL(c.rv, c.ttp)));
    float dldti = NSUB(c.cvap, c.cice);
    c.xai = NDIV(-dldti, c.rv);
    c.xbi = NADD(c.xai, NDIV(c.hsub, NMUL(c.rv, c.ttp)));
    /* nscv2d :488-491 */
    c.el2orc = NDIV(NMUL(c.hvap, c.hvap), NMUL(c.rv, c.cp));
    c.eps = NDIV(c.rd, c.rv);
    c.fact1 = NDIV(NSUB(c.cvap, c.cliq), c.rv);
    c.fact2 = NSUB(NDIV(c.hvap, c.rv), NMUL(c.fact1, NSCV_T0C));
    return c;
}

/* psat*(tr**x_a)*exp(x_b*(1.-tr)) */
__device__ __forceinline__ float nscv_fpvs_arm(float tr, float x_a,
                                               float x_b, float psat) {
    return NMUL(NMUL(psat, gfk_pow(tr, x_a)),
                gfk_exp(NMUL(x_b, NSUB(1.0f, tr))));
}

/* fpvs (:1373-1413) with ice = 1, the only value nscv2d passes. */
__device__ float nscv_fpvs(float t, const NscvConst& c) {
    const bool ice = t < c.ttp;
    float tr = NDIV(c.ttp, t);
    float f = ice ? nscv_fpvs_arm(tr, c.xai, c.xbi, c.psat)
                  : nscv_fpvs_arm(tr, c.xa, c.xb, c.psat);
    if (t < 180.0f) {
        tr = NDIV(c.ttp, 180.0f);
        f = ice ? nscv_fpvs_arm(tr, c.xai, c.xbi, c.psat)
                : nscv_fpvs_arm(tr, c.xa, c.xb, c.psat);
    }
    if (t >= 330.0f) {
        tr = NDIV(c.ttp, 330.0f);
        f = ice ? nscv_fpvs_arm(tr, c.xai, c.xbi, c.psat)
                : nscv_fpvs_arm(tr, c.xa, c.xb, c.psat);
    }
    return f;
}

/* qeso from fpvs at pressure pp (hPa): 0.01*fpvs, then
 * eps*qeso/(pp + (eps-1.)*qeso), floored at 1.e-8 (:595-598 and its
 * repeats at :662-665, :1217-1220, :1259-1263). */
__device__ __forceinline__ float nscv_qes(float t, float pp,
                                          const NscvConst& c) {
    float es = NMUL(0.01f, nscv_fpvs(t, c));
    float q = NDIV(NMUL(c.eps, es),
                   NADD(pp, NMUL(NSUB(c.eps, 1.0f), es)));
    return fmaxf(q, 1.e-8f);
}

extern "C" __global__ void shcu_nscv_column(
        const float* __restrict__ t3d, const float* __restrict__ qv3d,
        const float* __restrict__ qc3d, const float* __restrict__ qi3d,
        const float* __restrict__ p3d, const float* __restrict__ p3di,
        const float* __restrict__ pi3d, const float* __restrict__ rho3d,
        const float* __restrict__ u3d, const float* __restrict__ v3d,
        const float* __restrict__ dz8w, const float* __restrict__ w,
        const float* __restrict__ raincv, const float* __restrict__ xland,
        const float* __restrict__ hpbl, const float* __restrict__ hfx,
        const float* __restrict__ qfx,
        float* __restrict__ rthshten, float* __restrict__ rqvshten,
        float* __restrict__ rqcshten, float* __restrict__ rqishten,
        float* __restrict__ rushten, float* __restrict__ rvshten,
        float* __restrict__ pratesh, float* __restrict__ hbot,
        float* __restrict__ htop, int* __restrict__ status,
        float* __restrict__ workspace,
        float dt, float pgcon, int ncloud, int f_qc, int f_qi,
        float cp, float cliq, float cpv, float g, float xlv, float r_d,
        float r_v, float ep_1, float cice, float xls, float psat,
        int nz, int ncol, int col0) {
    const int tile_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = col0 + tile_index;
    if (col >= ncol) return;

    /* This lane's workspace region (kf.cu's KFWS_LANE_BASE shape). */
    float* ws = workspace
        + ((size_t)blockIdx.x * (blockDim.x / NSCV_LANES)
           + threadIdx.x / NSCV_LANES) * NSCV_SLOTS * (size_t)nz * NSCV_LANES
        + threadIdx.x % NSCV_LANES;
#define WS(s, k) ws[((size_t)(s) * (size_t)nz + (size_t)((k) - 1)) \
                    * NSCV_LANES]
#define IN(a, k) (a)[(size_t)((k) - 1) * (size_t)ncol + (size_t)col]

    const NscvConst c = nscv_constants(cp, cliq, cpv, g, xlv, r_d, r_v,
                                       ep_1, cice, xls, psat);
    const int kte = nz;
    const int km1 = kte - 1;
    const float delt = dt;
    const float rdelt = NDIV(1.0f, delt);

    /* ---- shcu_nscv wrapper (:152-198) ---------------------------------- */
    /* prsll = p3d*0.001; prsii = p3di*0.001; del = prsll*g/r_d*dz8w/t3d */
#define PRSL(k) NMUL(IN(p3d, k), 0.001f)
#define PRSI(k) NMUL(IN(p3di, k), 0.001f)
#define PCB(k) NMUL(PRSL(k), 10.0f)      /* nscv2d's p, hPa (:567) */
#define DEL(k) NDIV(NMUL(NDIV(NMUL(PRSL(k), c.g), c.rd), IN(dz8w, k)), \
                    IN(t3d, k))
    const int icps = IN(raincv, 1) > 1.e-30f ? 1 : 0;
    /* raincv is 2-D: IN(raincv, 1) reads raincv[col]. */

    /* ---- nscv2d: surface buoyancy flux (:428-442) ---------------------- */
    const float t1_1 = IN(t3d, 1);
    const float q1_1 = IN(qv3d, 1);
    const float thx1 = NDIV(t1_1, IN(pi3d, 1));
    const float tvcon = NADD(1.0f, NMUL(c.fv, q1_1));
    const float rhox = NDIV(NMUL(PRSL(1), 1.e3f),
                            NMUL(NMUL(c.rd, t1_1), tvcon));
    const float hfx_c = hfx[col], qfx_c = qfx[col], hpbl_c = hpbl[col];
    const float sflx = NADD(NDIV(NDIV(hfx_c, rhox), c.cp),
                            NMUL(NMUL(NDIV(qfx_c, rhox), c.fv), thx1));

    bool cnv = !(icps == 1) && !(sflx <= 0.0f);
    float rain = 0.0f;
    int kbot = kte + 1, ktop = 0;       /* assigned only if cnv (:450-453) */
    const bool entered = cnv;
    int kmax = kte, kbm = kte;
    int kb = kte, kbcon = kte, ktcon = 1;

    if (cnv) {
        /* kbm / kmax (:505-519) */
        const float psfc_k = PRSI(1);
        for (int k = 1; k <= kte; ++k) {
            const float pk = PRSL(k);
            if (pk > NMUL(psfc_k, 0.70f)) kbm = k + 1;
            if (pk > NMUL(psfc_k, 0.60f)) kmax = k + 1;
        }
        kbm = min(kbm, kmax);
        if (kmax > kte || kbm >= kte) {
            /* Deliberate divergence 3 (header). */
            atomicOr(status, NSCV_STATUS_TOP_TOO_LOW);
            cnv = false;
        }
    }

    if (cnv) {
        /* zii/zll (:166-180) */
        float zii = 0.0f;
        for (int k = 1; k <= kte; ++k) {
            const float znext = NADD(zii, IN(dz8w, k));
            WS(NS_ZL, k) = NMUL(0.5f, NADD(zii, znext));
            zii = znext;
        }
    }
    /* zi(k) = 0.5*(zl(k-1)+zl(k)), k >= 2 (:524-528);
     * xlamue(k) = clam/zi(k+1), k <= km1, xlamue(kte) = xlamue(km1). */
#define ZL(k) WS(NS_ZL, k)
#define ZI(k) NMUL(0.5f, NADD(ZL((k) - 1), ZL(k)))
#define XLAMUE(k) NDIV(NSCV_CLAM, ZI(min((k), km1) + 1))

    int kpbl = 1;
    if (cnv) {
        bool flg = true;
        for (int k = 2; k <= km1; ++k) {
            if (flg && ZL(k) <= hpbl_c) kpbl = k;
            else flg = false;
        }
        kpbl = min(kpbl, kbm);

        /* initialise (:564-582) and the working copies t1/q1/u1/v1/qc2/qi2
         * (:182-193), which change only at k <= kmax. */
        for (int k = 1; k <= kmax; ++k) {
            WS(NS_ETA, k) = 1.0f;
            WS(NS_HCKO, k) = 0.0f; WS(NS_QCKO, k) = 0.0f;
            WS(NS_UCKO, k) = 0.0f; WS(NS_VCKO, k) = 0.0f;
            WS(NS_DBYO, k) = 0.0f; WS(NS_PWO, k) = 0.0f;
            WS(NS_DELLAL, k) = 0.0f;
            const float tk = IN(t3d, k), qk = IN(qv3d, k);
            const float uk = IN(u3d, k), vk = IN(v3d, k);
            WS(NS_TO, k) = tk; WS(NS_QO, k) = qk;
            WS(NS_UO, k) = NMUL(uk, 1.0f); WS(NS_VO, k) = NMUL(vk, 1.0f);
            WS(NS_T1, k) = tk; WS(NS_Q1, k) = qk;
            WS(NS_U1, k) = uk; WS(NS_V1, k) = vk;
            WS(NS_QC2, k) = IN(qc3d, k); WS(NS_QI2, k) = IN(qi3d, k);
        }
        /* qeso and the qo floor (:592-603), moist static energy (:607-615) */
        for (int k = 1; k <= kmax; ++k) {
            const float tok = WS(NS_TO, k);
            const float qes = nscv_qes(tok, PCB(k), c);
            WS(NS_QESO, k) = qes;
            const float qok = fmaxf(WS(NS_QO, k), 1.e-10f);
            WS(NS_QO, k) = qok;
            const float tem = NADD(NMUL(c.g, ZL(k)), NMUL(c.cp, tok));
            WS(NS_HEO, k) = NADD(tem, NMUL(c.hvap, qok));
            WS(NS_HESO, k) = NADD(tem, NMUL(c.hvap, qes));
        }
        /* updraft source level (:620-636) */
        float hmax = WS(NS_HEO, 1);
        kb = 1;
        for (int k = 2; k <= kte; ++k) {
            if (k <= kpbl) {
                const float h = WS(NS_HEO, k);
                if (h > hmax) { kb = k; hmax = h; }
            }
        }
        /* layer means (:638-657): ascending, so to(k+1)/qo(k+1)/qeso(k+1)
         * are still the level values when level k reads them. */
        for (int k = 1; k <= km1; ++k) {
            if (k <= kmax - 1) {
                const float dz = NMUL(0.5f, NSUB(ZL(k + 1), ZL(k)));
                const float pk1 = PCB(k + 1);
                const float dp = NMUL(0.5f, NSUB(pk1, PCB(k)));
                const float to1 = WS(NS_TO, k + 1);
                const float es = NMUL(0.01f, nscv_fpvs(to1, c));
                const float pprime = NADD(pk1, NMUL(NSUB(c.eps, 1.0f), es));
                const float qs = NDIV(NMUL(c.eps, es), pprime);
                const float dqsdp = -NDIV(qs, pprime);
                const float to1sq = NMUL(to1, to1);
                const float desdt = NMUL(es, NADD(NDIV(c.fact1, to1),
                                                  NDIV(c.fact2, to1sq)));
                const float dqsdt = NDIV(NMUL(NMUL(qs, pk1), desdt),
                                         NMUL(es, pprime));
                const float gamma = NDIV(NMUL(c.el2orc, WS(NS_QESO, k + 1)),
                                         to1sq);
                const float dtl = NDIV(
                    NADD(NMUL(c.g, dz), NMUL(NMUL(c.hvap, dqsdp), dp)),
                    NMUL(c.cp, NADD(1.0f, gamma)));
                const float dq = NADD(NMUL(dqsdt, dtl), NMUL(dqsdp, dp));
                WS(NS_TO, k) = NADD(to1, dtl);
                WS(NS_QO, k) = NADD(WS(NS_QO, k + 1), dq);
            }
        }
        /* (:659-676); po(k) = .5*(p(k)+p(k+1)) */
        for (int k = 1; k <= km1; ++k) {
            if (k <= kmax - 1) {
                const float po = NMUL(0.5f, NADD(PCB(k), PCB(k + 1)));
                const float tok = WS(NS_TO, k);
                const float qes = nscv_qes(tok, po, c);
                WS(NS_QESO, k) = qes;
                const float qok = fmaxf(WS(NS_QO, k), 1.e-10f);
                WS(NS_QO, k) = qok;
                const float base = NADD(
                    NMUL(NMUL(0.5f, c.g), NADD(ZL(k), ZL(k + 1))),
                    NMUL(c.cp, tok));
                WS(NS_HEO, k) = NADD(base, NMUL(c.hvap, qok));
                WS(NS_HESO, k) = NADD(base, NMUL(c.hvap, qes));
                WS(NS_UO, k) = NMUL(0.5f, NADD(WS(NS_UO, k), WS(NS_UO, k + 1)));
                WS(NS_VO, k) = NMUL(0.5f, NADD(WS(NS_VO, k), WS(NS_VO, k + 1)));
            }
        }
        /* level of free convection as cloud base (:680-700) */
        kbcon = kmax;
        {
            const float hkb = WS(NS_HEO, kb);
            for (int k = 2; k <= km1; ++k) {
                if (k < kbm && k > kb && hkb > WS(NS_HESO, k)) {
                    kbcon = k;
                    break;
                }
            }
        }
        if (kbcon == kmax) cnv = false;
    }

    float xlamud = 0.0f;
    if (cnv) {
        /* critical convective inhibition (:711-749) */
        const float rho_b = IN(rho3d, kbcon);
        const float dot_b = -NMUL(NMUL(NMUL(5.0e-4f, c.g), rho_b),
                                  NADD(IN(w, kbcon), IN(w, kbcon + 1)));
        const float pdot = NMUL(10.0f, dot_b);
        float w3, w4;
        if (xland[col] == 1.0f) { w3 = NSCV_W3L; w4 = NSCV_W4L; }
        else                    { w3 = NSCV_W3S; w4 = NSCV_W4S; }
        float ptem;
        if (pdot <= w4) {
            ptem = NDIV(NSUB(pdot, w4), NSUB(w3, w4));
        } else if (pdot >= -w4) {
            ptem = -NDIV(NADD(pdot, w4), NSUB(w4, w3));
        } else {
            ptem = 0.0f;
        }
        ptem = fmaxf(ptem, -1.0f);
        ptem = fminf(ptem, 1.0f);
        ptem = NSUB(1.0f, ptem);
        const float ptem1 = NMUL(0.5f, NSUB(NSCV_CINCRMAX, NSCV_CINCRMIN));
        const float cincr = NSUB(NSCV_CINCRMAX, NMUL(ptem, ptem1));
        const float tem1 = NSUB(PCB(kb), PCB(kbcon));
        if (tem1 > cincr) cnv = false;
    }

    int kbcon1 = kmax;
    if (cnv) {
        xlamud = XLAMUE(kbcon);                               /* :760-764 */
        /* subcloud mass flux (:768-778), descending */
        for (int k = km1; k >= 1; --k) {
            if (k < kbcon && k >= kb) {
                const float dz = NSUB(ZI(k + 2), ZI(k + 1));
                const float ptem = NSUB(
                    NMUL(0.5f, NADD(XLAMUE(k), XLAMUE(k + 1))), xlamud);
                WS(NS_ETA, k) = NDIV(WS(NS_ETA, k + 1),
                                     NADD(1.0f, NMUL(ptem, dz)));
            }
        }
        /* above cloud base (:782-792) */
        for (int k = 2; k <= km1; ++k) {
            if (k > kbcon && k < kmax) {
                const float dz = NSUB(ZI(k + 1), ZI(k));
                const float ptem = NSUB(
                    NMUL(0.5f, NADD(XLAMUE(k), XLAMUE(k - 1))), xlamud);
                WS(NS_ETA, k) = NMUL(WS(NS_ETA, k - 1),
                                     NADD(1.0f, NMUL(ptem, dz)));
            }
        }
        /* updraft properties (:796-825) */
        WS(NS_HCKO, kb) = WS(NS_HEO, kb);
        WS(NS_UCKO, kb) = WS(NS_UO, kb);
        WS(NS_VCKO, kb) = WS(NS_VO, kb);
        for (int k = 2; k <= km1; ++k) {
            if (k > kb && k < kmax) {
                const float dz = NSUB(ZI(k + 1), ZI(k));
                const float tem = NMUL(
                    NMUL(0.5f, NADD(XLAMUE(k), XLAMUE(k - 1))), dz);
                const float tem1 = NMUL(NMUL(0.5f, xlamud), dz);
                const float factor = NSUB(NADD(1.0f, tem), tem1);
                const float ptem = NADD(NMUL(0.5f, tem), pgcon);
                const float ptem1 = NSUB(NMUL(0.5f, tem), pgcon);
                const float om = NSUB(1.0f, tem1);
                const float hc = NDIV(
                    NADD(NMUL(om, WS(NS_HCKO, k - 1)),
                         NMUL(NMUL(tem, 0.5f),
                              NADD(WS(NS_HEO, k), WS(NS_HEO, k - 1)))),
                    factor);
                WS(NS_HCKO, k) = hc;
                WS(NS_UCKO, k) = NDIV(
                    NADD(NADD(NMUL(om, WS(NS_UCKO, k - 1)),
                              NMUL(ptem, WS(NS_UO, k))),
                         NMUL(ptem1, WS(NS_UO, k - 1))), factor);
                WS(NS_VCKO, k) = NDIV(
                    NADD(NADD(NMUL(om, WS(NS_VCKO, k - 1)),
                              NMUL(ptem, WS(NS_VO, k))),
                         NMUL(ptem1, WS(NS_VO, k - 1))), factor);
                WS(NS_DBYO, k) = NSUB(hc, WS(NS_HESO, k));
            }
        }
        /* dry layers below cloud base (:830-859) */
        for (int k = 2; k <= km1; ++k) {
            if (k < kbm && k >= kbcon && WS(NS_DBYO, k) > 0.0f) {
                kbcon1 = k;
                break;
            }
        }
        if (kbcon1 == kmax) cnv = false;
        if (cnv) {
            const float tem = NSUB(PCB(kbcon), PCB(kbcon1));
            if (tem > NSCV_DTHK) cnv = false;
        }
    }

    float aa1 = 0.0f, xmbmax = 0.0f;
    if (cnv) {
        /* first-guess cloud top (:870-884) */
        ktcon = kbm;
        for (int k = 2; k <= km1; ++k) {
            if (k < kbm && k > kbcon1 && WS(NS_DBYO, k) < 0.0f) {
                ktcon = k;
                break;
            }
        }
        /* upper limit of the cloud-base mass flux (:888-894) */
        xmbmax = NDIV(NMUL(1000.0f, DEL(kbcon)), NMUL(c.g, delt));
        /* cloud moisture and precipitation (:898-939) */
        aa1 = 0.0f;
        WS(NS_QCKO, kb) = WS(NS_QO, kb);
        const float c0c1 = NADD(NSCV_C0, NSCV_C1);
        for (int k = 2; k <= km1; ++k) {
            if (k > kb && k < ktcon) {
                const float dz = NSUB(ZI(k + 1), ZI(k));
                const float tok = WS(NS_TO, k);
                const float qes = WS(NS_QESO, k);
                const float gamma = NDIV(NMUL(c.el2orc, qes),
                                         NMUL(tok, tok));
                const float qrch = NADD(qes, NDIV(
                    NMUL(gamma, WS(NS_DBYO, k)),
                    NMUL(c.hvap, NADD(1.0f, gamma))));
                const float tem = NMUL(
                    NMUL(0.5f, NADD(XLAMUE(k), XLAMUE(k - 1))), dz);
                const float tem1 = NMUL(NMUL(0.5f, xlamud), dz);
                const float factor = NSUB(NADD(1.0f, tem), tem1);
                float qc = NDIV(
                    NADD(NMUL(NSUB(1.0f, tem1), WS(NS_QCKO, k - 1)),
                         NMUL(NMUL(tem, 0.5f),
                              NADD(WS(NS_QO, k), WS(NS_QO, k - 1)))),
                    factor);
                const float etak = WS(NS_ETA, k);
                const float dq = NMUL(etak, NSUB(qc, qrch));
                if (k >= kbcon && dq > 0.0f) {
                    const float etah = NMUL(0.5f,
                                            NADD(etak, WS(NS_ETA, k - 1)));
                    float qlk;
                    if (ncloud > 0) {
                        const float dp = NMUL(1000.0f, DEL(k));
                        qlk = NDIV(dq, NADD(etak, NMUL(NMUL(etah, c0c1), dz)));
                        WS(NS_DELLAL, k) = NDIV(
                            NMUL(NMUL(NMUL(NMUL(etah, NSCV_C1), dz), qlk),
                                 c.g), dp);
                    } else {
                        qlk = NDIV(dq, NADD(etak,
                                            NMUL(NMUL(etah, NSCV_C0), dz)));
                    }
                    aa1 = NSUB(aa1, NMUL(NMUL(dz, c.g), qlk));
                    qc = NADD(qlk, qrch);
                    WS(NS_PWO, k) = NMUL(NMUL(NMUL(etah, NSCV_C0), dz), qlk);
                }
                WS(NS_QCKO, k) = qc;
            }
        }
        /* cloud work function (:943-957) */
        for (int k = 2; k <= km1; ++k) {
            if (k >= kbcon && k < ktcon) {
                const float dz1 = NSUB(ZL(k + 1), ZL(k));
                const float tok = WS(NS_TO, k);
                const float qes = WS(NS_QESO, k);
                const float gamma = NDIV(NMUL(c.el2orc, qes),
                                         NMUL(tok, tok));
                const float rfact = NADD(1.0f, NDIV(
                    NMUL(NMUL(NMUL(c.fv, c.cp), gamma), tok), c.hvap));
                aa1 = NADD(aa1, NMUL(NDIV(NMUL(
                    NMUL(dz1, NDIV(c.g, NMUL(c.cp, tok))), WS(NS_DBYO, k)),
                    NADD(1.0f, gamma)), rfact));
                aa1 = NADD(aa1, NMUL(NMUL(NMUL(dz1, c.g), c.fv),
                                     fmaxf(0.0f, NSUB(qes, WS(NS_QO, k)))));
            }
        }
        if (aa1 <= 0.0f) cnv = false;
    }

    float qlko_ktcon = 0.0f, edt = 0.0f;
    if (cnv) {
        /* overshooting (:973-1002) */
        aa1 = NMUL(NSCV_AAFAC, aa1);
        int ktcon1 = kbm;
        for (int k = 2; k <= km1; ++k) {
            if (k >= ktcon && k < kbm) {
                const float dz1 = NSUB(ZL(k + 1), ZL(k));
                const float tok = WS(NS_TO, k);
                const float gamma = NDIV(NMUL(c.el2orc, WS(NS_QESO, k)),
                                         NMUL(tok, tok));
                const float rfact = NADD(1.0f, NDIV(
                    NMUL(NMUL(NMUL(c.fv, c.cp), gamma), tok), c.hvap));
                aa1 = NADD(aa1, NMUL(NDIV(NMUL(
                    NMUL(dz1, NDIV(c.g, NMUL(c.cp, tok))), WS(NS_DBYO, k)),
                    NADD(1.0f, gamma)), rfact));
                if (aa1 < 0.0f) {
                    ktcon1 = k;
                    break;
                }
            }
        }
        /* cloud moisture in the overshooting layers (:1007-1039) */
        const float c0c1 = NADD(NSCV_C0, NSCV_C1);
        for (int k = 2; k <= km1; ++k) {
            if (k >= ktcon && k < ktcon1) {
                const float dz = NSUB(ZI(k + 1), ZI(k));
                const float tok = WS(NS_TO, k);
                const float qes = WS(NS_QESO, k);
                const float gamma = NDIV(NMUL(c.el2orc, qes),
                                         NMUL(tok, tok));
                const float qrch = NADD(qes, NDIV(
                    NMUL(gamma, WS(NS_DBYO, k)),
                    NMUL(c.hvap, NADD(1.0f, gamma))));
                const float tem = NMUL(
                    NMUL(0.5f, NADD(XLAMUE(k), XLAMUE(k - 1))), dz);
                const float tem1 = NMUL(NMUL(0.5f, xlamud), dz);
                const float factor = NSUB(NADD(1.0f, tem), tem1);
                float qc = NDIV(
                    NADD(NMUL(NSUB(1.0f, tem1), WS(NS_QCKO, k - 1)),
                         NMUL(NMUL(tem, 0.5f),
                              NADD(WS(NS_QO, k), WS(NS_QO, k - 1)))),
                    factor);
                const float etak = WS(NS_ETA, k);
                const float dq = NMUL(etak, NSUB(qc, qrch));
                if (dq > 0.0f) {
                    const float etah = NMUL(0.5f,
                                            NADD(etak, WS(NS_ETA, k - 1)));
                    float qlk;
                    if (ncloud > 0) {
                        const float dp = NMUL(1000.0f, DEL(k));
                        qlk = NDIV(dq, NADD(etak, NMUL(NMUL(etah, c0c1), dz)));
                        WS(NS_DELLAL, k) = NDIV(
                            NMUL(NMUL(NMUL(NMUL(etah, NSCV_C1), dz), qlk),
                                 c.g), dp);
                    } else {
                        qlk = NDIV(dq, NADD(etak,
                                            NMUL(NMUL(etah, NSCV_C0), dz)));
                    }
                    qc = NADD(qlk, qrch);
                    WS(NS_PWO, k) = NMUL(NMUL(NMUL(etah, NSCV_C0), dz), qlk);
                }
                WS(NS_QCKO, k) = qc;
            }
        }
        /* exchange ktcon with ktcon1 (:1043-1049) */
        ktcon = ktcon1;

        /* liquid/vapour separation at cloud top (:1053-1074) */
        if (ncloud > 0) {
            const int k = ktcon - 1;
            const float tok = WS(NS_TO, k);
            const float qes = WS(NS_QESO, k);
            const float gamma = NDIV(NMUL(c.el2orc, qes), NMUL(tok, tok));
            const float qrch = NADD(qes, NDIV(
                NMUL(gamma, WS(NS_DBYO, k)),
                NMUL(c.hvap, NADD(1.0f, gamma))));
            const float dq = NSUB(WS(NS_QCKO, k), qrch);
            if (dq > 0.0f) {
                qlko_ktcon = dq;
                WS(NS_QCKO, k) = qrch;
            }
        }
        /* precipitation efficiency from wind shear (:1078-1106) */
        float vshear = 0.0f;
        for (int k = 2; k <= kte; ++k) {
            if (k > kb && k <= ktcon) {
                const float du = NSUB(WS(NS_UO, k), WS(NS_UO, k - 1));
                const float dv = NSUB(WS(NS_VO, k), WS(NS_VO, k - 1));
                vshear = NADD(vshear, __fsqrt_rn(NADD(NMUL(du, du),
                                                      NMUL(dv, dv))));
            }
        }
        vshear = NDIV(NMUL(1.e3f, vshear), NSUB(ZI(ktcon + 1), ZI(kb + 1)));
        const float vs2 = NMUL(vshear, vshear);
        const float e1 = NSUB(NADD(NSUB(1.591f, NMUL(0.639f, vshear)),
                                   NMUL(0.0953f, vs2)),
                              NMUL(0.00496f, NMUL(vs2, vshear)));
        edt = NSUB(1.0f, e1);
        edt = fminf(edt, 0.9f);
        edt = fmaxf(edt, 0.0f);
    }

    if (cnv) {
        /* cloud-base mass flux, Grant (2001) (:1203-1212); t1 is still the
         * input temperature here. */
        const float ptem = NDIV(NMUL(NMUL(c.g, sflx), hpbl_c), t1_1);
        const float wstar = gfk_pow(ptem, NSCV_H1);
        const float po_b = NMUL(0.5f, NADD(PCB(kbcon), PCB(kbcon + 1)));
        const float tem = NDIV(NMUL(po_b, 100.0f),
                               NMUL(c.rd, IN(t3d, kbcon)));
        float xmb = NMUL(NMUL(NSCV_BETAW, tem), wstar);
        xmb = fminf(xmb, xmbmax);
        const float dt2 = delt;

        /* qeso at the input temperature, every level <= kmax
         * (:1214-1223) -- the evaporation below reads it at k <= kb. */
        for (int k = 1; k <= kmax; ++k)
            WS(NS_QESO, k) = nscv_qes(IN(t3d, k), PCB(k), c);

        /* the cloud's change to the environment, applied (:1111-1198 and
         * :1234-1253).  dellah/dellaq/dellau/dellav start at zero and take
         * one value each, so they are formed where they are applied. */
        for (int k = 2; k <= kte; ++k) {
            if (k > kb && k <= ktcon) {
                const float dp = NMUL(1000.0f, DEL(k));
                float dh, dqv, du, dv;
                const float eta1 = WS(NS_ETA, k - 1);
                if (k < ktcon) {
                    const float dz = NSUB(ZI(k + 1), ZI(k));
                    const float etak = WS(NS_ETA, k);
                    const float tem = NMUL(0.5f, NADD(XLAMUE(k),
                                                      XLAMUE(k - 1)));
                    const float tem1 = xlamud;
                    const float h1v = WS(NS_HEO, k), h3v = WS(NS_HEO, k - 1);
                    const float q1v = WS(NS_QO, k), q3v = WS(NS_QO, k - 1);
                    const float u1v = WS(NS_UO, k), u3v = WS(NS_UO, k - 1);
                    const float v1v = WS(NS_VO, k), v3v = WS(NS_VO, k - 1);
                    const float h2v = NMUL(0.5f, NADD(h1v, h3v));
                    const float q2v = NMUL(0.5f, NADD(q1v, q3v));
                    const float u2v = NMUL(0.5f, NADD(u1v, u3v));
                    const float v2v = NMUL(0.5f, NADD(v1v, v3v));
                    const float te = NMUL(tem, eta1);
                    const float t1e = NMUL(NMUL(tem1, eta1), 0.5f);
                    dh = NDIV(NMUL(NADD(NSUB(NSUB(NMUL(etak, h1v),
                                                   NMUL(eta1, h3v)),
                                              NMUL(NMUL(te, h2v), dz)),
                                         NMUL(NMUL(t1e, NADD(WS(NS_HCKO, k),
                                                             WS(NS_HCKO, k - 1))),
                                              dz)),
                                    c.g), dp);
                    dqv = NDIV(NMUL(NADD(NSUB(NSUB(NMUL(etak, q1v),
                                                    NMUL(eta1, q3v)),
                                               NMUL(NMUL(te, q2v), dz)),
                                          NMUL(NMUL(t1e, NADD(WS(NS_QCKO, k),
                                                              WS(NS_QCKO, k - 1))),
                                               dz)),
                                     c.g), dp);
                    du = NDIV(NMUL(NSUB(NADD(NSUB(NSUB(NMUL(etak, u1v),
                                                        NMUL(eta1, u3v)),
                                                   NMUL(NMUL(te, u2v), dz)),
                                              NMUL(NMUL(t1e, NADD(WS(NS_UCKO, k),
                                                                  WS(NS_UCKO, k - 1))),
                                                   dz)),
                                         NMUL(NMUL(pgcon, eta1),
                                              NSUB(u1v, u3v))),
                                    c.g), dp);
                    dv = NDIV(NMUL(NSUB(NADD(NSUB(NSUB(NMUL(etak, v1v),
                                                        NMUL(eta1, v3v)),
                                                   NMUL(NMUL(te, v2v), dz)),
                                              NMUL(NMUL(t1e, NADD(WS(NS_VCKO, k),
                                                                  WS(NS_VCKO, k - 1))),
                                                   dz)),
                                         NMUL(NMUL(pgcon, eta1),
                                              NSUB(v1v, v3v))),
                                    c.g), dp);
                    /* dellah = 0 + increment: the delta loop ACCUMULATES
                     * onto its zero initialisation (:1147-1167). */
                    dh = NADD(0.0f, dh); dqv = NADD(0.0f, dqv);
                    du = NADD(0.0f, du); dv = NADD(0.0f, dv);
                } else {
                    /* cloud top (:1176-1191): ASSIGNED, not accumulated */
                    dh = NDIV(NMUL(NMUL(eta1, NSUB(WS(NS_HCKO, k - 1),
                                                   WS(NS_HEO, k - 1))),
                                   c.g), dp);
                    dqv = NDIV(NMUL(NMUL(eta1, NSUB(WS(NS_QCKO, k - 1),
                                                    WS(NS_QO, k - 1))),
                                    c.g), dp);
                    du = NDIV(NMUL(NMUL(eta1, NSUB(WS(NS_UCKO, k - 1),
                                                   WS(NS_UO, k - 1))),
                                   c.g), dp);
                    dv = NDIV(NMUL(NMUL(eta1, NSUB(WS(NS_VCKO, k - 1),
                                                   WS(NS_VO, k - 1))),
                                   c.g), dp);
                    /* cloud water at the top (:1195-1196) */
                    WS(NS_DELLAL, k) = NDIV(NMUL(NMUL(eta1, qlko_ktcon),
                                                 c.g), dp);
                }
                const float dellat = NDIV(NSUB(dh, NMUL(c.hvap, dqv)), c.cp);
                WS(NS_T1, k) = NADD(WS(NS_T1, k),
                                    NMUL(NMUL(dellat, xmb), dt2));
                WS(NS_Q1, k) = NADD(WS(NS_Q1, k),
                                    NMUL(NMUL(dqv, xmb), dt2));
                WS(NS_U1, k) = NADD(WS(NS_U1, k),
                                    NMUL(NMUL(NMUL(du, xmb), dt2), 1.0f));
                WS(NS_V1, k) = NADD(WS(NS_V1, k),
                                    NMUL(NMUL(NMUL(dv, xmb), dt2), 1.0f));
            }
        }
        /* qeso at the updated temperature inside the cloud (:1255-1267) */
        for (int k = 2; k <= kte; ++k) {
            if (k > kb && k <= ktcon)
                WS(NS_QESO, k) = nscv_qes(WS(NS_T1, k), PCB(k), c);
        }
        /* total rain (:1276-1284), descending */
        float rntot = 0.0f;
        for (int k = kte; k >= 1; --k) {
            if (k < ktcon && k > kb)
                rntot = NADD(rntot, NMUL(NMUL(NMUL(WS(NS_PWO, k), xmb),
                                              0.001f), dt2));
        }
        /* evaporating rain (:1288-1335), descending */
        bool flg = true;
        float delqev = 0.0f, delq2 = 0.0f;
        const float evef_factor = xland[col] == 1.0f ? NSCV_EVFACTL
                                                     : NSCV_EVFACT;
        for (int k = kte; k >= 1; --k) {
            if (k > kmax) continue;
            float qevap = 0.0f;
            if (k < ktcon && k > kb)
                rain = NADD(rain, NMUL(NMUL(NMUL(WS(NS_PWO, k), xmb),
                                            0.001f), dt2));
            if (flg && k < ktcon) {
                const float evef = NMUL(edt, evef_factor);
                const float t1k = WS(NS_T1, k);
                const float qes = WS(NS_QESO, k);
                const float qcond = NDIV(
                    NMUL(evef, NSUB(WS(NS_Q1, k), qes)),
                    NADD(1.0f, NDIV(NMUL(c.el2orc, qes), NMUL(t1k, t1k))));
                const float dp = NMUL(1000.0f, DEL(k));
                if (rain > 0.0f && qcond < 0.0f) {
                    qevap = NMUL(-qcond, NSUB(1.0f, gfk_exp(
                        -NMUL(0.32f, __fsqrt_rn(NMUL(dt2, rain))))));
                    qevap = fminf(qevap, NDIV(NMUL(NMUL(rain, 1000.0f), c.g),
                                              dp));
                    delq2 = NADD(delqev, NDIV(NMUL(NMUL(0.001f, qevap), dp),
                                              c.g));
                }
                if (rain > 0.0f && qcond < 0.0f && delq2 > rntot) {
                    qevap = NDIV(NMUL(NMUL(1000.0f, c.g),
                                      NSUB(rntot, delqev)), dp);
                    flg = false;
                }
                if (rain > 0.0f && qevap > 0.0f) {
                    const float tem = NDIV(NMUL(0.001f, dp), c.g);
                    const float tem1 = NMUL(qevap, tem);
                    if (tem1 > rain) {
                        qevap = NDIV(rain, tem);
                        rain = 0.0f;
                    } else {
                        rain = NSUB(rain, tem1);
                    }
                    WS(NS_Q1, k) = NADD(WS(NS_Q1, k), qevap);
                    WS(NS_T1, k) = NSUB(t1k, NMUL(NDIV(c.hvap, c.cp), qevap));
                    delqev = NADD(delqev, NDIV(NMUL(NMUL(0.001f, dp), qevap),
                                               c.g));
                }
            }
        }
        if (rain < 0.0f || !flg) rain = 0.0f;              /* :1339 */
        ktop = ktcon;
        kbot = kbcon;
        /* detrained cloud condensate (:1348-1367) */
        if (ncloud > 0) {
            const float tcrf = NDIV(1.0f, NSUB(NSCV_TCR, NSCV_TF));
            for (int k = 1; k <= km1; ++k) {
                if (k >= kbcon && k <= ktcon) {
                    const float tem = NMUL(NMUL(WS(NS_DELLAL, k), xmb), dt2);
                    const float tem1 = fmaxf(0.0f, fminf(1.0f, NMUL(
                        NSUB(NSCV_TCR, WS(NS_T1, k)), tcrf)));
                    if (ncloud >= 2) {
                        WS(NS_QI2, k) = NADD(WS(NS_QI2, k), NMUL(tem, tem1));
                        WS(NS_QC2, k) = NADD(WS(NS_QC2, k),
                                             NMUL(tem, NSUB(1.0f, tem1)));
                    } else {
                        WS(NS_QC2, k) = NADD(WS(NS_QC2, k), tem);
                    }
                }
            }
        }
    }

    /* ---- shcu_nscv wrapper outputs (:219-261) --------------------------- */
    pratesh[col] = NDIV(NMUL(rain, 1000.0f), dt);
    if (entered) {
        hbot[col] = (float)kbot;
        htop[col] = (float)ktop;
    }
    /* The working copies changed only where the column convected, and only
     * at k <= kmax; everywhere else t1 == t3d and every identity is +0. */
    const int kwork = cnv ? kmax : 0;
    bool finite = isfinite(pratesh[col]);
    for (int k = 1; k <= kte; ++k) {
        const size_t at = (size_t)(k - 1) * (size_t)ncol + (size_t)col;
        if (k <= kwork) {
            const float th = NMUL(NDIV(NSUB(WS(NS_T1, k), t3d[at]),
                                       pi3d[at]), rdelt);
            const float qv = NMUL(NSUB(WS(NS_Q1, k), qv3d[at]), rdelt);
            const float uu = NMUL(NSUB(WS(NS_U1, k), u3d[at]), rdelt);
            const float vv = NMUL(NSUB(WS(NS_V1, k), v3d[at]), rdelt);
            const float qi = f_qi ? NMUL(NSUB(WS(NS_QI2, k), qi3d[at]), rdelt)
                                  : 0.0f;
            const float qc = f_qc ? NMUL(NSUB(WS(NS_QC2, k), qc3d[at]), rdelt)
                                  : 0.0f;
            finite = finite && isfinite(th) && isfinite(qv) && isfinite(uu)
                     && isfinite(vv) && isfinite(qi) && isfinite(qc);
            rthshten[at] = th; rqvshten[at] = qv;
            rushten[at] = uu; rvshten[at] = vv;
            rqishten[at] = qi; rqcshten[at] = qc;
        } else {
            rthshten[at] = 0.0f; rqvshten[at] = 0.0f;
            rushten[at] = 0.0f; rvshten[at] = 0.0f;
            rqishten[at] = 0.0f; rqcshten[at] = 0.0f;
        }
    }
    if (!finite) atomicOr(status, NSCV_STATUS_NONFINITE);
#undef WS
#undef IN
#undef PRSL
#undef PRSI
#undef PCB
#undef DEL
#undef ZL
#undef ZI
#undef XLAMUE
}
