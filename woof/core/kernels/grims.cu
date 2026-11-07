// GRIMS shallow convection (WRF shcu_physics = 3), one column per thread.
//
// Transcribed from WRF v4.6.1 phys/module_shcu_grims.F: grims (:43-212),
// grims2d (:216-574), scv_tri_diagonal_grims (:578-652),
// phys_moist_adiabat_pa (:656-829) and the table functions ftdp, ftlcl,
// fthe, fpvs_pa (:833-936), in WRF's operation order.  Every + - * / is
// pinned with the FADD/FSUB/FMUL/FDIV macros (__f*_rn, never contracted),
// and the two real-exponent powers the column evaluates at run time --
// (p8w/p1000mb)**rcp and (...)**pfac with pfac = 3.0, which gfortran does
// NOT fold (nm -u on WRF's object: powf) -- call WOOF's own float32 powf,
// gfk_pow.  So the default build and the GPUWM_WRF_EXACT=1 build compute
// the same words through the shared unflushed NVRTC route.  Graded word for word against WRF's own compiled
// module_shcu_grims.o by tests/test_grims_wrf461_parity.py.
//
// The five grimsinit tables are inputs (gpuwm/data/grims/grims_tables.npz,
// built by tools/generate_grims_tables.py and checked word for word
// against the tables WRF's own grimsinit builds).
//
// ONE COLUMN PER THREAD IS WRF'S RESULT, NOT AN APPROXIMATION OF IT.
// grims2d packs a tile row's triggered columns into n2 and solves one
// tridiagonal band k1..k2 (min kbot .. max ktop) for all of them.  Every
// row of a column outside its own kbot..ktop has diagonal exactly 1 and
// off-diagonals and right-hand-side increments of signed zero
// (xkzh = 0), so for finite inputs the forward and back sweeps pass those
// rows through unchanged and the band's extent never reaches a column's
// words.  The oracle calls WRF both ways (all columns in one row, and one
// column per call) and the two agree on every word of the fixture.
//
// DECLARED DIVERGENCE (never bit-exact to a WRF bug): grims2d fills its
// height array z(i,k) for k <= levshc only (:367-387) and then reads
// z(i,kpbl) (:428).  A PBL top above sigma 0.6 reads uninitialised stack
// memory in WRF.  Here it reads the layer height zl(kpbl) the array
// stands for; the oracle grades those columns against WRF built with that
// one fill added (tools/grims_wrf461_oracle/build.sh, patched/).
//
// No per-thread local frame: the column's scalars live in registers, the
// swept right-hand sides in the output planes, and the normalised upper
// diagonal in dynamic shared memory, levshc words per thread
// (gpuwm/core/grims.py sizes the block to it).

#define GR_NXTDP 5001
#define GR_NXTHE 241
#define GR_NYTHE 151
#define GR_NXMA 151
#define GR_NYMA 121
#define GR_NXSVP 7501

// Order of the twelve interpolation constants in the `cst` argument; the
// same order gpuwm/core/grims.py and tools/generate_grims_tables.py use.
#define GR_C1XMA 0
#define GR_C2XMA 1
#define GR_C1YMA 2
#define GR_C2YMA 3
#define GR_C1XPVS 4
#define GR_C2XPVS 5
#define GR_C1XTDP 6
#define GR_C2XTDP 7
#define GR_C1XTHE 8
#define GR_C2XTHE 9
#define GR_C1YTHE 10
#define GR_C2YTHE 11

// Fortran MIN/MAX on finite operands.
__device__ __forceinline__ float gr_min(float a, float b) { return (b < a) ? b : a; }
__device__ __forceinline__ float gr_max(float a, float b) { return (b > a) ? b : a; }

// tb(jx) + (xj-jx)*(tb(jx+1)-tb(jx)), jx one-based.
__device__ __forceinline__ float gr_lookup1(const float *tb, float xj, int jx)
{
    float lo = tb[jx - 1], hi = tb[jx];
    return FADD(lo, FMUL(FSUB(xj, (float)jx), FSUB(hi, lo)));
}

// The bilinear form of fthe (:904-906) and of the inlined ftma (:808-813);
// `tb` is Fortran tb(jx, jy), i.e. row jy of length nx.
__device__ __forceinline__ float gr_bilinear(const float *tb, int nx,
                                             float xj, float yj, int jx, int jy)
{
    float fx = FSUB(xj, (float)jx);
    const float *r0 = tb + (size_t)(jy - 1) * nx;
    const float *r1 = r0 + nx;
    float ftx1 = FADD(r0[jx - 1], FMUL(fx, FSUB(r0[jx], r0[jx - 1])));
    float ftx2 = FADD(r1[jx - 1], FMUL(fx, FSUB(r1[jx], r1[jx - 1])));
    return FADD(ftx1, FMUL(FSUB(yj, (float)jy), FSUB(ftx2, ftx1)));
}

// fpvs_pa (:916-936): saturation vapour pressure in Pa.
__device__ __forceinline__ float gr_fpvs_pa(const float *tbpvs, const float *cst, float t)
{
    float xj = gr_min(gr_max(FADD(cst[GR_C1XPVS], FMUL(cst[GR_C2XPVS], t)), 1.0f),
                      (float)GR_NXSVP);
    int jx = (int)gr_min(xj, (float)GR_NXSVP - 1.0f);
    return FMUL(gr_lookup1(tbpvs, xj, jx), 1.e3f);
}

// heo (:383) and heso (:384) share this sum.
__device__ __forceinline__ float gr_mse(float g, float z, float cp, float t,
                                        float xlv, float q)
{
    return FADD(FADD(FMUL(g, z), FMUL(cp, t)), FMUL(xlv, q));
}

// qeso2 (:380-382).
__device__ __forceinline__ float gr_qeso(const float *tbpvs, const float *cst,
                                         float t, float pik, float eps, float epsm1)
{
    float e = gr_fpvs_pa(tbpvs, cst, t);
    e = FDIV(FMUL(eps, e), FADD(pik, FMUL(epsm1, e)));
    return gr_max(e, 1.E-8f);
}

extern "C" __global__
void grims_column(const float *__restrict__ qv, const float *__restrict__ t,
                  const float *__restrict__ p, const float *__restrict__ pi,
                  const float *__restrict__ p8w, const float *__restrict__ zw,
                  const float *__restrict__ wstar, const float *__restrict__ hpbl,
                  const float *__restrict__ delta, const float *__restrict__ raincv,
                  const int *__restrict__ kpbl,
                  const float *__restrict__ tbtdp, const float *__restrict__ tbthe,
                  const float *__restrict__ tbtma, const float *__restrict__ tbqma,
                  const float *__restrict__ tbpvs, const float *__restrict__ cst,
                  float *__restrict__ rthshten, float *__restrict__ rqvshten,
                  int *__restrict__ kbot_out, int *__restrict__ ktop_out,
                  float dt, float g, float xlv, float rd, float rv, float rcp,
                  float p1000mb, int levshc, int nz, int ncol)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    // Fortran k (1-based) -> field index.
#define AT(a, k) (a)[(size_t)((k) - 1) * (size_t)ncol + (size_t)col]

    // module_shcu_grims's own PARAMETER cp: grims passes cp=cp (:195) and
    // grims has no cp dummy, so the column uses 1004.6, not the driver's.
    const float cp = 1.0046e+3f;
    const float pfac = 3.0f, xkzmax = 50.0f, xkzmin = 0.001f;
    const float zdiffcr1 = 1000.0f, zdiffcr2 = 1000.0f, wsfac = 1.47f;
    const float rdelt = FDIV(1.0f, dt);

    int kp = kpbl[col];
    bool active = false;
    // --- trigger: moist static instability below sigma 0.6 (:338-351) ---
    if (!(raincv[col] > 1.e-30f) && kp >= 2) {
        for (int k = 1; k <= levshc - 1; ++k) {
            float eldq = FMUL(xlv, FSUB(AT(qv, k), AT(qv, k + 1)));
            float cpdt = FMUL(cp, FSUB(AT(t, k), AT(t, k + 1)));
            float rtdls = FMUL(FMUL(FMUL(FDIV(FSUB(AT(p, k), AT(p, k + 1)),
                                              AT(p8w, k + 1)), rd), 0.5f),
                               FADD(AT(t, k), AT(t, k + 1)));
            float dmse = FSUB(FADD(eldq, cpdt), rtdls);
            active = active || (dmse > 0.0f);
        }
    }
    if (wstar[col] < 0.001f) active = false;

    int kbot = 0, ktop = 0;
    const float eps = FDIV(rd, rv);
    const float epsm1 = FSUB(eps, 1.0f);

    if (active) {
        // zl (:181) on the fly.
#define ZL(k) FMUL(0.5f, FADD(AT(zw, k), AT(zw, (k) + 1)))
#define HEO(k) gr_mse(g, ZL(k), cp, AT(t, k), xlv, AT(qv, k))
        // --- parcel origin: largest heo at or below kpbl (:391-411) ------
        float hmax = HEO(1);
        int kb = 1, kbcon = levshc;
        for (int k = 2; k <= levshc - 1; ++k) {
            float h = HEO(k);
            if (h > hmax && k <= kp) { kb = k; hmax = h; }
        }
        // --- first level the parcel is saturated against (:415-424) -----
        for (int k = 2; k <= levshc - 1; ++k) {
            if (k > kb) {
                float heso = gr_mse(g, ZL(k), cp, AT(t, k), xlv,
                                    gr_qeso(tbpvs, cst, AT(t, k), AT(p, k), eps, epsm1));
                if (hmax > heso) { kbcon = k; break; }
            }
        }
        // hmax == heo(kb) here: kb only ever moved to where hmax was set.
        // :428 -- see the DECLARED DIVERGENCE above for kpbl > levshc.
        float zdiff1 = FSUB(ZL(kp < nz ? kp : nz), ZL(kb));
        float zdiff2 = FSUB(ZL(kbcon), ZL(kb));
        if (zdiff1 > zdiffcr1 || zdiff1 < 0.0f) active = false;
        if (zdiff2 > zdiffcr2) active = false;

        if (active) {
            // --- phys_moist_adiabat_pa, klev = levshc-1, k1 = k2 = 2 -----
            // Only the bottom interface Exner function is read (prsik(i,1)).
            float prsik1 = gfk_pow(FDIV(AT(p8w, 1), p1000mb), rcp);
            const float ftv = FSUB(FDIV(rv, rd), 1.0f);
            float slkma = 0.0f, thema = 0.0f;
            {
                const int k = 2;
                float q = AT(qv, k), tk = AT(t, k);
                float pv = FDIV(FMUL(FMUL(AT(p, k), 1.e-3f), q),
                                FSUB(eps, FMUL(epsm1, q)));
                // ftdp (:833-855)
                float xj = gr_min(gr_max(FADD(cst[GR_C1XTDP], FMUL(cst[GR_C2XTDP], pv)), 1.0f),
                                  (float)GR_NXTDP);
                int jx = (int)gr_min(xj, (float)GR_NXTDP - 1.0f);
                float tdpd = FSUB(tk, gr_lookup1(tbtdp, xj, jx));
                float tlcl, slklcl;
                if (tdpd > 0.0f) {
                    // ftlcl (:859-872)
                    const float clcl1 = 0.954442e+0f, clcl2 = 0.967772e-3f;
                    const float clcl3 = -0.710321e-3f, clcl4 = -0.270742e-5f;
                    tlcl = FSUB(tk, FMUL(tdpd, FADD(FADD(clcl1, FMUL(clcl2, tk)),
                                                    FMUL(tdpd, FADD(clcl3, FMUL(clcl4, tk))))));
                    slklcl = FDIV(FMUL(FDIV(AT(pi, k), prsik1), tlcl), tk);
                } else {
                    tlcl = tk;
                    slklcl = FDIV(AT(pi, k), prsik1);
                }
                // fthe (:876-912)
                float pk = FMUL(slklcl, prsik1);
                float xt = gr_min(FADD(cst[GR_C1XTHE], FMUL(cst[GR_C2XTHE], tlcl)), (float)GR_NXTHE);
                float yt = gr_min(FADD(cst[GR_C1YTHE], FMUL(cst[GR_C2YTHE], pk)), (float)GR_NYTHE);
                float thelcl = 0.0f;
                if (xt >= 1.0f && yt >= 1.0f) {
                    int jxt = (int)gr_min(xt, (float)GR_NXTHE - 1.0f);
                    int jyt = (int)gr_min(yt, (float)GR_NYTHE - 1.0f);
                    thelcl = gr_bilinear(tbthe, GR_NXTHE, xt, yt, jxt, jyt);
                }
                if (thelcl > thema) { slkma = slklcl; thema = thelcl; }
            }
            int ktop_m = 0;
            for (int k = 2; k <= levshc - 1; ++k) {
                if (FDIV(AT(pi, k), prsik1) <= slkma) {
                    float xj = gr_min(gr_max(FADD(cst[GR_C1XMA], FMUL(cst[GR_C2XMA], thema)), 1.0f),
                                      (float)GR_NXMA);
                    float yj = gr_min(gr_max(FADD(cst[GR_C1YMA], FMUL(cst[GR_C2YMA], AT(pi, k))), 1.0f),
                                      (float)GR_NYMA);
                    int jx = (int)gr_min(xj, (float)GR_NXMA - 1.0f);
                    int jy = (int)gr_min(yj, (float)GR_NYMA - 1.0f);
                    float tma = gr_bilinear(tbtma, GR_NXMA, xj, yj, jx, jy);
                    float qma = gr_bilinear(tbqma, GR_NXMA, xj, yj, jx, jy);
                    float tvcld = FMUL(tma, FADD(1.0f, FMUL(ftv, qma)));
                    float tvenv = FMUL(AT(t, k), FADD(1.0f, FMUL(ftv, AT(qv, k))));
                    if (tvcld > tvenv) ktop_m = k;   // max over ascending k
                }
            }
            // --- cloud base and top (:444-484) ---------------------------
            kbot = kp > 2 ? kp : 2;
            kbot = kbot < levshc ? kbot : levshc;
            ktop = ktop_m + 1;
            float hmin = HEO(kbot);
            int lmin = levshc;
            for (int k = kbot; k <= (ktop < levshc ? ktop : levshc); ++k) {
                float h = HEO(k);
                if (h < hmin) { lmin = k + 1; hmin = h; }
            }
            ktop = ktop < lmin ? ktop : lmin;
            if (ktop - kbot <= 1) active = false;
        }
#undef HEO
        if (active) {
            // --- diffusion properties (:488-509) -------------------------
            float rh2 = FDIV(AT(qv, kbot - 1),
                             gr_qeso(tbpvs, cst, AT(t, kbot - 1), AT(p, kbot - 1), eps, epsm1));
            float zbot = ZL(kbot);
            float depth = FSUB(ZL(ktop), zbot);
            const float xkzfac = FMUL(FMUL(FMUL(rh2, wsfac), wstar[col]), delta[col]);
            const float twodt = FMUL(2.0f, dt);
            const float hb = hpbl[col];
            // --- the implicit system and its forward sweep, fused --------
            // WRF assembles every interface first (:518-553) and then
            // sweeps (scv_tri_diagonal_grims :623-643).  Row r's sweep reads
            // only interfaces r-1 and r, so one ascending pass does both
            // with every value formed by the same operations in the same
            // order: au(k) = g/rtdls on the UNMODIFIED temperatures, row
            // k's t2 = (t - dtodsu(k-1)*dsdz1(k-1)) + dtodsl(k)*dsdz1(k),
            // ad(k) = (1 - al(k-1)) - au(k).  The normalised upper
            // diagonal waits in shared memory for the back sweep and the
            // swept right-hand sides wait in the output planes, so the
            // kernel holds no per-thread local frame.
            extern __shared__ float gr_au_shared[];
#define AUP(k) gr_au_shared[(size_t)((k) - kbot) * blockDim.x + threadIdx.x]
            float pend_t = AT(t, kbot);
            float pend_ad = 1.0f;
            float al_prev = 0.0f, aup_prev = 0.0f, qp_prev = 0.0f, tp_prev = 0.0f;
            for (int k = kbot; k < ktop; ++k) {
                float rtdls = FMUL(FMUL(FMUL(FDIV(FSUB(AT(p, k), AT(p, k + 1)),
                                                  AT(p8w, k + 1)), rd), 0.5f),
                                   FADD(AT(t, k), AT(t, k + 1)));
                float au_raw = FDIV(g, rtdls);
                float dtodsl = FDIV(twodt, FSUB(AT(p8w, k), AT(p8w, k + 1)));
                float dtodsu = FDIV(twodt, FSUB(AT(p8w, k + 1), AT(p8w, k + 2)));
                float dsig = FSUB(AT(p, k), AT(p, k + 1));
                float height = FSUB(ZL(k), zbot);
                float frac = FDIV(FADD(height, hb), FADD(depth, hb));
                float xkzh = gr_min(gr_max(FMUL(xkzfac, gfk_pow(FSUB(1.0f, frac), pfac)),
                                           xkzmin), xkzmax);
                float dsdz1 = FDIV(FMUL(FMUL(FMUL(xkzh, dsig), au_raw), g), cp);
                float dsdz2 = FMUL(FMUL(FMUL(xkzh, dsig), au_raw), au_raw);
                float au_k = -FMUL(dtodsl, dsdz2);
                float al_k = -FMUL(dtodsu, dsdz2);
                float ad_k = FSUB(pend_ad, au_k);
                float t2_k = FADD(pend_t, FMUL(dtodsl, dsdz1));
                float q2_k = AT(qv, k);
                float fk, aup, qp, tp;
                if (k == kbot) {
                    fk = FDIV(1.0f, ad_k);
                    aup = FMUL(fk, au_k);
                    qp = FMUL(fk, q2_k);
                    tp = FMUL(fk, t2_k);
                } else {
                    fk = FDIV(1.0f, FSUB(ad_k, FMUL(al_prev, aup_prev)));
                    aup = FMUL(fk, au_k);
                    qp = FMUL(fk, FSUB(q2_k, FMUL(al_prev, qp_prev)));
                    tp = FMUL(fk, FSUB(t2_k, FMUL(al_prev, tp_prev)));
                }
                AUP(k) = aup;
                AT(rthshten, k) = tp;
                AT(rqvshten, k) = qp;
                al_prev = al_k;
                aup_prev = aup;
                qp_prev = qp;
                tp_prev = tp;
                pend_ad = FSUB(1.0f, al_k);
                pend_t = FSUB(AT(t, k + 1), FMUL(dtodsu, dsdz1));
            }
            // Row ktop closes the sweep (:639-643) ...
            float fk = FDIV(1.0f, FSUB(pend_ad, FMUL(al_prev, aup_prev)));
            float qn = FMUL(fk, FSUB(AT(qv, ktop), FMUL(al_prev, qp_prev)));
            float tn = FMUL(fk, FSUB(pend_t, FMUL(al_prev, tp_prev)));
            // ... and the back sweep (:644-649) writes the tendencies
            // (:201-208) as each row's solution lands.
            AT(rthshten, ktop) = FMUL(FDIV(FSUB(tn, AT(t, ktop)), AT(pi, ktop)), rdelt);
            AT(rqvshten, ktop) = FMUL(FSUB(qn, AT(qv, ktop)), rdelt);
            for (int k = ktop - 1; k >= kbot; --k) {
                qn = FSUB(AT(rqvshten, k), FMUL(AUP(k), qn));
                tn = FSUB(AT(rthshten, k), FMUL(AUP(k), tn));
                AT(rthshten, k) = FMUL(FDIV(FSUB(tn, AT(t, k)), AT(pi, k)), rdelt);
                AT(rqvshten, k) = FMUL(FSUB(qn, AT(qv, k)), rdelt);
            }
#undef AUP
        }
#undef ZL
    }

    // --- tendencies (:201-208) of the rows the solve did not reach: an
    // unchanged temperature and humidity give exactly +0. -----------------
    for (int k = 1; k <= nz; ++k) {
        if (!(active && k >= kbot && k <= ktop)) {
            AT(rthshten, k) = 0.0f;
            AT(rqvshten, k) = 0.0f;
        }
    }
    if (kbot_out != nullptr) kbot_out[col] = active ? kbot : 0;
    if (ktop_out != nullptr) ktop_out[col] = active ? ktop : 0;
#undef AT
}
