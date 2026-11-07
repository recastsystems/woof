// The K-epsilon PBL (bl_pbl_physics = 17), WRF v4.7.1 phys/module_bl_keps.F
// (Zonato et al. 2022): prognostic TKE, dissipation rate and turbulent
// potential energy (TPE), with a Prandtl number from the surface layer and
// a counter-gradient heat flux from the TPE.  One thread owns one column
// for the whole call: every routine of the module is sequential in the
// vertical (the length scales walk the column, the diffusion solves are
// tridiagonal).
//
// Strict transcription rules. The current authority is the unmodified
// WRF 4.7.1 real run captured by tools/keps_wrf471_oracle. The earlier
// tools/keps_wrf461_oracle synthetic fixtures are historical only.
//
// * The referee is WRF's compiled scalar build (gfortran 13.3, -O2
//   -fno-fast-math -ffp-contract=off -fno-tree-vectorize).  There every real
//   power with exponent 2.0, 1.0 or -1.0 is folded (x*x, x, 1/x) and every
//   other real exponent is a powf call; integer exponents are multiply
//   chains.  An -O0 build calls powf for x**2. as well, which is not
//   correctly rounded, and so differs from the binary WRF users run.
// * One rounding per Fortran operator, in Fortran's left-to-right order
//   (FADD/FSUB/FMUL/FDIV/FSQRT, the round-to-nearest intrinsics), loaded
//   with -fmad=false --ftz=false so no product is fused and no subnormal is
//   flushed.
// * powf and expf are WOOF's float32 routines (gfk_pow, gfk_exp); tanhf is
//   WOOF's float32 tanh (mynn_tanhf, the MYNN unit's, reused rather than
//   copied).  sqrtf is correctly rounded on both sides.
// * MAX(a, b) and MIN(a, b) are gfortran's lowering at -O2, maxss/minss with
//   the Fortran operands in order: the first argument when it compares
//   strictly greater (smaller), else the second.
// * Constants WRF's compiler folds are folded once, here, as hex words
//   (gpuwm/data/keps/tables/constants.cuh).
//
// Native undefined variables have deterministic GPU initial values.
// The current native replay does not patch or initialize either variable:
//   D1 GET_PBLH's maximum-TKE scan compares against `ttke`, a never-assigned
//      variable (module_bl_keps.F:822-823); here it compares
//      against the TKE computed on the line above, as the MYNN original does.
//   D2 the cloud-top radiative sum `radsum` is accumulated without being set
//      (:1065); here it starts at zero.
//   The first-level y-momentum flux reads the x wind (:363), as WRF does.
//   D4 z/L = br*psim*psim/psih is 0/0 in the surface layer's neutral regime
//      (psim = psih = 0 exactly); here the neutral regime's own z/L = 0 goes
//      into the same clamps, which is the answer the -O2 build reaches
//      through maxss's NaN rule.
//
// WHAT THE SCHEME RECEIVES AND NEVER READS (the pinned source, proven by
// the oracle's flag_bep arm): TSK, MOL, FRC_URB2D, FLAG_BEP and the eleven
// BEP source-term arrays.  The driver sets the urban surface and volume
// fractions to one and the urban fraction to zero (:287-291), so under BEP
// the scheme feels the city only through the grid-blended HFX, QFX and UST
// the urban coupling writes.  WQ (WQ_TUR) is INTENT(OUT) and never assigned.
//
// Layout: inputs (nz, ncol) with the column fastest, WRF's k = 1 first;
// p8w (nz+1, ncol); surface fields (ncol).  The column arrays live in a
// global workspace, KE_SLOTS rows of (nz+2) words per thread, interleaved by
// thread so a warp's accesses coalesce; the grid is a fixed tile of threads
// that strides over the columns.

// Parameters are loaded from the pinned keps/tables asset.

// Strict replay follows WRF's scalar operator order. Runtime uses CUDA math
// and permits contraction. These overrides affect this new unit only.
#ifdef GPUWM_WRF_EXACT
#define KE_POW gfk_pow
#define KE_EXP gfk_exp
#define KE_TANH mynn_tanhf
#else
#undef FADD
#undef FSUB
#undef FMUL
#undef FDIV
#undef FSQRT
#define FADD(a,b) ((a)+(b))
#define FSUB(a,b) ((a)-(b))
#define FMUL(a,b) ((a)*(b))
#define FDIV(a,b) __fdividef((a),(b))
#define FSQRT(a) sqrtf(a)
__device__ __forceinline__ float ke_fast_pow(float x, float power) {
    if (power == 0.5f) return sqrtf(x);
    if (power == -0.5f) return rsqrtf(x);
    if (power == -0.25f) return rsqrtf(sqrtf(x));
    if (power == 1.5f) return x * sqrtf(x);
    if (power == 3.0f) return (x * x) * x;
    return __powf(x, power);
}
#define KE_POW ke_fast_pow
#define KE_EXP __expf
#define KE_TANH tanhf
#endif

// Workspace rows (1-based levels 1..nz+1 inside each row).
enum {
    KS_Z = 0, KS_DZ, KS_RHO, KS_RHOZ, KS_U, KS_V, KS_TH, KS_Q, KS_QC,
    KS_TE, KS_D, KS_TPE, KS_PRA, KS_SH, KS_BU, KS_NSQ, KS_DTMDZ, KS_RI,
    KS_PRST, KS_EXCH, KS_EXCHH, KS_DLS, KS_AD, KS_BT, KS_ATPE, KS_BTPE,
    KS_CDDZ, KS_A1, KS_A2, KS_A3, KS_C, KS_AUX,
    KE_SLOTS
};

struct KeCol {
    float *base;          // ws + thread
    size_t stride;        // threads in the tile
    int nzp;              // nz + 2
    __device__ __forceinline__ float &at(int slot, int k) const {
        return base[((size_t)slot * nzp + k) * stride];
    }
};
#define W(slot, k) col.at((slot), (k))

__device__ __forceinline__ float ke_max(float a, float b) { return (a > b) ? a : b; }
__device__ __forceinline__ float ke_min(float a, float b) { return (a < b) ? a : b; }

// diff (module_bl_keps.F:547-624) with sf = vl = 1 (the driver's values),
// followed by invert (:701-723).  `co` is updated in place.  The flux
// output fc is never read by keps1D and is not formed.  aa/bb are the
// implicit/explicit source rows (slot, or -1 for WRF's zero arrays, with
// `bb1` the explicit term at level 1 only, for the surface rows b_q/a_u).
// cd is read from slot `cdslot`, divided by `cddiv` when cddiv != 1 (the
// exch/sigma_eps actual argument of the dissipation solve).
#ifndef GPUWM_WRF_EXACT
// Assemble each row when the descending solve consumes it. The face
// coefficient above the row and eliminated values stay in registers.
// Independent fields share factors while retaining their own sources.
// Normalize each pivot once; the ascending substitution needs no division.
template<int N>
__device__ void ke_diff_batch(const KeCol &col, int nz, int iz1, float dt,
                             int cdslot, bool cd_sigma, int aaslot, float aa1,
                             const int (&slots)[N], const int (&bbs)[N],
                             const float (&bb1)[N])
{
    #define KE_BATCH_CD(k) (cd_sigma ? FDIV(W(cdslot, (k)), KE_SIGEPS) : W(cdslot, (k)))
    float next_cd = FDIV(FMUL(FMUL(2.0f, W(KS_RHOZ, nz)), KE_BATCH_CD(nz)),
                            FADD(W(KS_DZ, nz), W(KS_DZ, nz - 1)));
    float next_a1 = 0.0f;
    float next_c[N];
    #pragma unroll
    for (int j = 0; j < N; ++j) next_c[j] = W(slots[j], nz);
    W(KS_A1, nz) = 0.0f;
    for (int in = nz - 1; in >= 1; --in) {
        const float c0 = in == 1
            ? FDIV(FMUL(W(KS_RHOZ, 1), KE_BATCH_CD(1)), W(KS_DZ, 1))
            : FDIV(FMUL(FMUL(2.0f, W(KS_RHOZ, in)), KE_BATCH_CD(in)),
                   FADD(W(KS_DZ, in), W(KS_DZ, in - 1)));
        float a1 = 0.0f, a2 = 1.0f, a3 = 0.0f;
        if (in >= iz1) {
            const float dzv = W(KS_DZ, in), rh = W(KS_RHO, in);
            const float aa = aaslot >= 0 ? W(aaslot, in) : (in == 1 ? aa1 : 0.0f);
            a1 = -FDIV(FDIV(FMUL(c0, dt), dzv), rh);
            a2 = FSUB(FADD(1.0f, FDIV(FDIV(FMUL(dt, FADD(c0, next_cd)), dzv), rh)),
                      FMUL(aa, dt));
            a3 = -FDIV(FDIV(FMUL(next_cd, dt), dzv), rh);
        }
        const float inverse_pivot = FDIV(1.0f, FSUB(a2, FMUL(a3, next_a1)));
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            float rhs = W(slots[j], in);
            if (in >= iz1) {
                const float bb = bbs[j] >= 0 ? W(bbs[j], in) : (in == 1 ? bb1[j] : 0.0f);
                rhs = FADD(rhs, FMUL(bb, dt));
            }
            next_c[j] = FMUL(FSUB(rhs, FMUL(a3, next_c[j])), inverse_pivot);
            W(slots[j], in) = next_c[j];
        }
        next_a1 = FMUL(a1, inverse_pivot);
        next_cd = c0;
        W(KS_A1, in) = next_a1;
    }
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        next_c[j] = W(slots[j], 1);
    }
    for (int in = 2; in <= nz; ++in) {
        const float a1 = W(KS_A1, in);
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            next_c[j] = FSUB(W(slots[j], in), FMUL(a1, next_c[j]));
            W(slots[j], in) = next_c[j];
        }
    }
    #undef KE_BATCH_CD
}
#endif

__device__ void ke_diff(const KeCol &col, int nz, int iz1, float dt, int coslot,
                        int cdslot, bool cd_sigma, int aaslot, float aa1,
                        int bbslot, float bb1)
{
#ifndef GPUWM_WRF_EXACT
    const int slots[] = {coslot}, sources[] = {bbslot};
    const float surface[] = {bb1};
    ke_diff_batch(col, nz, iz1, dt, cdslot, cd_sigma, aaslot, aa1,
                  slots, sources, surface);
#else
    #define KE_CD(k) (cd_sigma ? FDIV(W(cdslot, (k)), KE_SIGEPS) : W(cdslot, (k)))
    // cddz(kts) = sf*rhoz*cd/dz; cddz(iz) = 2.*sf*rhoz*cd/(dz(iz)+dz(iz-1))
    W(KS_CDDZ, 1) = FDIV(FMUL(W(KS_RHOZ, 1), KE_CD(1)), W(KS_DZ, 1));
    for (int iz = 2; iz <= nz; ++iz)
        W(KS_CDDZ, iz) = FDIV(FMUL(FMUL(2.0f, W(KS_RHOZ, iz)), KE_CD(iz)),
                              FADD(W(KS_DZ, iz), W(KS_DZ, iz - 1)));
    // cddz(kte+1) feeds only the identity row kte and is not formed.
    for (int iz = 1; iz <= iz1 - 1; ++iz) {
        W(KS_A1, iz) = 0.0f; W(KS_A2, iz) = 1.0f; W(KS_A3, iz) = 0.0f;
        W(KS_C, iz) = W(coslot, iz);
    }
    for (int iz = iz1; iz <= nz - 1; ++iz) {
        const float dzv = W(KS_DZ, iz);          // vl(iz)*dz(iz), vl = 1
        const float rh = W(KS_RHO, iz);
        const float c0 = W(KS_CDDZ, iz), c1 = W(KS_CDDZ, iz + 1);
        const float aa = aaslot >= 0 ? W(aaslot, iz) : (iz == 1 ? aa1 : 0.0f);
        const float bb = bbslot >= 0 ? W(bbslot, iz) : (iz == 1 ? bb1 : 0.0f);
        W(KS_A1, iz) = -FDIV(FDIV(FMUL(c0, dt), dzv), rh);
        W(KS_A2, iz) = FSUB(FADD(1.0f, FDIV(FDIV(FMUL(dt, FADD(c0, c1)), dzv), rh)),
                            FMUL(aa, dt));
        W(KS_A3, iz) = -FDIV(FDIV(FMUL(c1, dt), dzv), rh);
        W(KS_C, iz) = FADD(W(coslot, iz), FMUL(bb, dt));
    }
    W(KS_A1, nz) = 0.0f; W(KS_A2, nz) = 1.0f; W(KS_A3, nz) = 0.0f;
    W(KS_C, nz) = W(coslot, nz);
    // invert
    for (int in = nz - 1; in >= 1; --in) {
        const float a2n = W(KS_A2, in + 1);
        W(KS_C, in) = FSUB(W(KS_C, in), FDIV(FMUL(W(KS_A3, in), W(KS_C, in + 1)), a2n));
        W(KS_A2, in) = FSUB(W(KS_A2, in), FDIV(FMUL(W(KS_A3, in), W(KS_A1, in + 1)), a2n));
    }
    for (int in = 2; in <= nz; ++in)
        W(KS_C, in) = FSUB(W(KS_C, in), FDIV(FMUL(W(KS_A1, in), W(KS_C, in - 1)),
                                             W(KS_A2, in - 1)));
    for (int in = 1; in <= nz; ++in)
        W(coslot, in) = FDIV(W(KS_C, in), W(KS_A2, in));
    #undef KE_CD
#endif
}



// GET_PBLH (module_bl_keps.F:775-879, with D1).  zw = z faces, the TKE is the
// clamped carried state, theta and q the call's inputs.  Returns zi and
// writes iz_pbl.  The two searches carry k <= kte guards WRF does not
// have: they only bind when the whole column lies below 500 m or the
// minimum theta-v sits within two levels of the top, where WRF reads past
// its arrays.
__device__ float ke_get_pblh(const KeCol &col, int nz, int &iz_pbl)
{
    for (int iz = 1; iz <= nz; ++iz)
        W(KS_AUX, iz) = FMUL(W(KS_TH, iz), FADD(1.0f, FMUL(0.61f, W(KS_Q, iz))));
    int k = 2, kthv = 1, ktke = 1;
    float maxtke = 0.0f, minthv = 9.e9f;
    while (k <= nz && W(KS_Z, k) <= 500.0f) {
        const float tke = ke_max(W(KS_TE, k), 0.0f);
        if (maxtke < tke) { maxtke = tke; ktke = k; }
        if (minthv > W(KS_AUX, k)) { minthv = W(KS_AUX, k); kthv = k; }
        k = k + 1;
    }
    float tkeeps = FDIV(maxtke, 20.0f);
    tkeeps = ke_max(tkeeps, KE_TKEEPS_LO);
    tkeeps = ke_min(tkeeps, KE_TKEEPS_HI);
    float zi = 0.0f;
    k = kthv + 1;
    const float thr = FADD(minthv, 1.5f);
    while (zi == 0.0f) {
        if (k > nz - 1) { zi = W(KS_Z, 2); break; }
        if (W(KS_AUX, k) >= thr) {
            const float fr = ke_min(FDIV(FSUB(W(KS_AUX, k), thr),
                                         ke_max(FSUB(W(KS_AUX, k), W(KS_AUX, k - 1)), 1e-6f)),
                                    1.0f);
            zi = FSUB(W(KS_Z, k), FMUL(W(KS_DZ, k - 1), fr));
        }
        k = k + 1;
        if (k == nz - 1) zi = W(KS_Z, 2);       // WRF's exit safeguard
    }
    float pblh_tke = 0.0f;
    k = ktke + 1;
    while (pblh_tke == 0.0f) {
        if (k > nz - 1) { pblh_tke = W(KS_Z, 2); break; }
        const float tke = ke_max(FDIV(W(KS_TE, k), 2.0f), 0.0f);
        const float tkem1 = ke_max(FDIV(W(KS_TE, k - 1), 2.0f), 0.0f);
        if (tke <= tkeeps) {
            pblh_tke = FSUB(W(KS_Z, k),
                            FMUL(W(KS_DZ, k - 1),
                                 ke_min(FDIV(FSUB(tkeeps, tke),
                                             ke_max(FSUB(tkem1, tke), 1e-6f)), 1.0f)));
            pblh_tke = ke_max(pblh_tke, W(KS_Z, 2));
        }
        k = k + 1;
        if (k == nz - 1) pblh_tke = W(KS_Z, 2);
    }
    const float wt = FADD(FMUL(0.5f, KE_TANH(FDIV(FSUB(zi, 200.0f), 400.0f))), 0.5f);
    zi = FADD(FMUL(pblh_tke, FSUB(1.0f, wt)), FMUL(zi, wt));
    iz_pbl = 1;
    for (k = 1; k <= nz; ++k)
        if (W(KS_Z, k) <= zi) iz_pbl = iz_pbl + 1;
    return zi;
}

// keps1D (module_bl_keps.F:387-517) and every routine it calls, in its call
// order.  On entry the workspace holds the driver's column; on return u, v,
// th, q, qc, te, d, tpe, Pra and exch are the scheme's results.
__device__ void ke_keps1d(const KeCol &col, int nz, int c, int ncol, float dt,
                          float ust, float hfx, float qfx, float sflux,
                          float cp, float g, float b_ric, float z0, float psim,
                          float psih, float pblh, int iz_pbl, float a_u1,
                          float b_t1, float b_q1, const float *__restrict__ pi,
                          const float *__restrict__ raten,
                          const float *__restrict__ p8w)
{
    #define GIN(a, k) (a)[(size_t)((k) - 1) * ncol + c]
    const float gth0 = FDIV(g, KE_TH0);
    // ---- shear (:676-696) ----------------------------------------------
    W(KS_SH, 1) = 0.0f;
    for (int iz = 2; iz <= nz - 1; ++iz) {
        const float dzm = W(KS_DZ, iz - 1), dz0 = W(KS_DZ, iz), dzp = W(KS_DZ, iz + 1);
        const float u2 = FDIV(FADD(FMUL(dzp, W(KS_U, iz)), FMUL(dz0, W(KS_U, iz + 1))),
                              FADD(dz0, dzp));
        const float u1 = FDIV(FADD(FMUL(dz0, W(KS_U, iz - 1)), FMUL(dzm, W(KS_U, iz))),
                              FADD(dzm, dz0));
        const float v2 = FDIV(FADD(FMUL(dzp, W(KS_V, iz)), FMUL(dz0, W(KS_V, iz + 1))),
                              FADD(dz0, dzp));
        const float v1 = FDIV(FADD(FMUL(dz0, W(KS_V, iz - 1)), FMUL(dzm, W(KS_V, iz))),
                              FADD(dzm, dz0));
        const float du = FDIV(FSUB(u2, u1), dz0), dv = FDIV(FSUB(v2, v1), dz0);
        W(KS_SH, iz) = FADD(FMUL(du, du), FMUL(dv, dv));
    }
    W(KS_SH, nz) = 0.0f;
    // ---- c_nsquare (:653-671) --------------------------------------------
    const float dz1h = FDIV(W(KS_DZ, 1), 2.0f);
    W(KS_DTMDZ, 1) = FDIV(FDIV(-FDIV(sflux, ust), KE_VK), FSUB(dz1h, z0));
    W(KS_NSQ, 1) = -FMUL(gth0, W(KS_DTMDZ, 1));
    for (int iz = 2; iz <= nz - 1; ++iz) {
        const float d1 = FDIV(FMUL(2.0f, FSUB(W(KS_TH, iz), W(KS_TH, iz - 1))),
                              FADD(W(KS_DZ, iz - 1), W(KS_DZ, iz)));
        const float d2 = FDIV(FMUL(2.0f, FSUB(W(KS_TH, iz + 1), W(KS_TH, iz))),
                              FADD(W(KS_DZ, iz + 1), W(KS_DZ, iz)));
        W(KS_DTMDZ, iz) = FMUL(0.5f, FADD(d1, d2));
        W(KS_NSQ, iz) = -FDIV(FMUL(W(KS_DTMDZ, iz), g), KE_TH0);
    }
    W(KS_NSQ, nz) = 0.0f;
    // dtmdz(kte) is never set in WRF; it reaches only a_tpe/b_tpe(kte), which
    // the TPE solve's identity row never reads.
    W(KS_DTMDZ, nz) = 0.0f;
    // ---- surface_bl_pra_ri (:1022-1126, with D2 and D4) --------------------
    W(KS_RI, 1) = b_ric;
    for (int iz = 2; iz <= nz; ++iz) {
        float ri = ke_min(FDIV(-W(KS_NSQ, iz), W(KS_SH, iz)), 100.0f);
        if (W(KS_SH, iz) < 1e-15f)
            ri = (-W(KS_NSQ, iz) > 0.0f) ? 10.0f : -1.0f;
        W(KS_RI, iz) = ri;
    }
    W(KS_RI, nz) = W(KS_RI, nz - 1);
    float radsum = 0.0f;                                      // D2
    if (iz_pbl >= 2) {
        for (int iz = 1; iz <= iz_pbl - 1; ++iz) {
            float radflux = FMUL(GIN(raten, iz), GIN(pi, iz));
            radflux = FMUL(FDIV(FMUL(radflux, cp), g),
                           FSUB(GIN(p8w, iz), GIN(p8w, iz + 1)));
            if (radflux < 0.0f) radsum = FADD(fabsf(radflux), radsum);
        }
    }
    radsum = FDIV(FDIV(ke_max(radsum, 0.0f), W(KS_RHO, iz_pbl - 1)), cp);
    const bool sfcflg = !(b_ric >= 0.0f);
    const float zl1 = dz1h;
    const float zz = FDIV(FADD(zl1, z0), zl1);
    float zol1 = (psih != 0.0f)                                // D4
        ? ke_max(FDIV(FMUL(FMUL(b_ric, psim), psim), psih), KE_RIMIN)
        : ke_max(0.0f, KE_RIMIN);
    if (sfcflg) zol1 = FMUL(zz, ke_min(zol1, -KE_ZFMIN));
    else zol1 = FMUL(zz, ke_max(zol1, KE_ZFMIN));
    const float hol1 = FMUL(FDIV(FMUL(zol1, pblh), zl1), KE_SFCFRAC);
    float phim, phih, phieps, phim_sl, phih_sl, wstar3, wstar3_2;
    if (sfcflg) {
        const float x = FSUB(1.0f, FMUL(KE_APHI16, zol1));
        phim = KE_POW(x, -0.25f);
        phih = KE_POW(x, -0.5f);
        phieps = FSUB(1.0f, zol1);
        const float y = FSUB(1.0f, FMUL(KE_APHI16, hol1));
        phim_sl = KE_POW(y, -0.25f);
        phih_sl = KE_POW(y, -0.5f);
        const float bfx0 = ke_max(sflux, 0.0f);
        const float govrth = FDIV(g, W(KS_TH, 1));
        const float govrthv = FDIV(g, FMUL(W(KS_TH, iz_pbl - 1),
                                           FADD(1.0f, FMUL(0.606271777f, W(KS_Q, iz_pbl - 1)))));
        wstar3 = FMUL(FMUL(govrth, bfx0), pblh);
        wstar3_2 = FMUL(FMUL(govrthv, radsum), pblh);
    } else {
        phim_sl = FADD(1.0f, FMUL(KE_APHI5, hol1));
        phih_sl = phim_sl;
        phim = FADD(1.0f, FMUL(KE_APHI5, zol1));
        phieps = KE_POW(FADD(1.0f, FMUL(2.5f, KE_POW(zol1, 0.6f))), 1.5f);
        phih = phim;
        wstar3 = 0.0f;
        wstar3_2 = 0.0f;
    }
    // Pra(kte+1) is formed by WRF from z(kte+2), one word past the column,
    // and read by nothing; the loop stops at kte.  WRF's iz_pbl > 1 test
    // always holds (z(kts) = 0 <= pblh), so Pra=4. never survives.
    for (int iz = 1; iz <= nz; ++iz) {
        float pra;
        if (sfcflg) {
            const float t = ke_max(FSUB(W(KS_Z, iz + 1), FMUL(KE_SFCFRAC, pblh)), 0.0f);
            const float prnumfac = -FDIV(FMUL(3.0f, FMUL(t, t)), FMUL(pblh, pblh));
            const float w = FADD(wstar3, wstar3_2);
            const float u3 = FMUL(FMUL(ust, ust), ust);
            const float prfac2 = FDIV(FDIV(FMUL(15.9f, w), u3),
                                      FADD(1.0f, FDIV(FMUL(KE_4VK, w), u3)));
            float prnum0 = FADD(FDIV(phih_sl, phim_sl), KE_CONPR);
            prnum0 = FDIV(prnum0, FADD(1.0f, FMUL(FMUL(prfac2, KE_VK), KE_SFCFRAC)));
            prnum0 = ke_max(ke_min(prnum0, KE_PRMAX), KE_PRMIN);
            pra = FADD(1.0f, FMUL(FSUB(prnum0, 1.0f), KE_EXP(prnumfac)));
        } else {
            pra = ke_max(1.0f, ke_min(4.0f, FADD(1.0f, FMUL(3.0f, W(KS_RI, iz)))));
        }
        W(KS_PRA, iz) = pra;
    }
    W(KS_DTMDZ, 1) = -FDIV(FMUL(FDIV(FDIV(sflux, ust), KE_VK), phih), FSUB(dz1h, z0));
    W(KS_PRST, 1) = 0.0f;
    for (int iz = 2; iz <= nz; ++iz)
        W(KS_PRST, iz) = FDIV(FADD(FMUL(W(KS_PRA, iz - 1), W(KS_DZ, iz)),
                                   FMUL(W(KS_PRA, iz), W(KS_DZ, iz - 1))),
                              FADD(W(KS_DZ, iz), W(KS_DZ, iz - 1)));
    W(KS_PRST, nz + 1) = W(KS_PRST, nz);
    // ---- buoy (:629-648) ---------------------------------------------------
    W(KS_BU, 1) = 0.0f;
    for (int iz = 2; iz <= nz - 1; ++iz) {
        const float phi_bu = FDIV(FMUL(FMUL(FDIV(FMUL(gth0, W(KS_TPE, iz)), W(KS_TE, iz)),
                                            W(KS_PRA, iz)), KE_CTHETA), KE_CMU);
        const float d1 = FDIV(FMUL(2.0f, FSUB(W(KS_TH, iz), W(KS_TH, iz - 1))),
                              FADD(W(KS_DZ, iz - 1), W(KS_DZ, iz)));
        const float d2 = FDIV(FMUL(2.0f, FSUB(W(KS_TH, iz + 1), W(KS_TH, iz))),
                              FADD(W(KS_DZ, iz + 1), W(KS_DZ, iz)));
        const float dtm = FMUL(0.5f, FADD(d1, d2));
        W(KS_BU, iz) = FADD(-FDIV(FMUL(dtm, g), KE_TH0), FMUL(gth0, phi_bu));
    }
    W(KS_BU, nz) = 0.0f;
    // ---- cdtur_ke (:522-541) -----------------------------------------------
    W(KS_EXCH, 1) = 0.0f;
    W(KS_EXCHH, 1) = 0.0f;
    for (int iz = 2; iz <= nz; ++iz) {
        const float tm = W(KS_TE, iz - 1), t0 = W(KS_TE, iz);
        const float e1 = FMUL(FDIV(FMUL(KE_CMU, FMUL(tm, tm)), W(KS_D, iz - 1)), W(KS_DZ, iz - 1));
        const float e2 = FMUL(FDIV(FMUL(KE_CMU, FMUL(t0, t0)), W(KS_D, iz)), W(KS_DZ, iz));
        float ex = FDIV(FADD(e1, e2), FADD(W(KS_DZ, iz), W(KS_DZ, iz - 1)));
        ex = ke_max(ex, 0.09f);
        W(KS_EXCH, iz) = ex;
        W(KS_EXCHH, iz) = FDIV(ex, W(KS_PRST, iz));
    }
    W(KS_EXCH, nz + 1) = W(KS_EXCH, nz);
    W(KS_EXCHH, nz + 1) = W(KS_EXCHH, nz);
    // ---- length_bougeault (:949-1017) --------------------------------------
    for (int iz = 1; iz <= nz; ++iz) {
        const float te = W(KS_TE, iz), thi = W(KS_TH, iz);
        // WRF consumes DLS only in the rare dissipation repair below
        // (thetaphi_alphabeta:915-917). The input TE/D words are unchanged
        // between these routines. Skip the quadratic search when its
        // result has no consumer; this changes no state or output word.
        const float diss = W(KS_D, iz);
        if (iz == 1 || !(FDIV(fabsf(te), fabsf(diss)) > 1e4f && diss <= 1e-6f)) {
            W(KS_DLS, iz) = 0.0f;
            continue;
        }
        const float beta = gth0;
        float zup = 0.0f, zzz = 0.0f, zup_inf = 0.0f;
        float dlu = FSUB(FSUB(W(KS_Z, nz + 1), W(KS_Z, iz)), FDIV(W(KS_DZ, iz), 2.0f));
        for (int izz = iz; izz <= nz - 1; ++izz) {
            const float dzt = FDIV(FADD(W(KS_DZ, izz + 1), W(KS_DZ, izz)), 2.0f);
            zup = FSUB(zup, FMUL(FMUL(beta, thi), dzt));
            zup = FADD(zup, FDIV(FMUL(FMUL(beta, FADD(W(KS_TH, izz + 1), W(KS_TH, izz))), dzt), 2.0f));
            zzz = FADD(zzz, dzt);
            if (te < zup && te >= zup_inf) {
                const float bbb = FDIV(FSUB(W(KS_TH, izz + 1), W(KS_TH, izz)), dzt);
                float tl;
                if (bbb != 0.0f) {
                    const float q = FMUL(beta, FSUB(W(KS_TH, izz), thi));
                    tl = FDIV(FDIV(FADD(-q, FSQRT(ke_max(0.0f, FADD(FMUL(q, q),
                              FMUL(FMUL(FMUL(2.0f, bbb), beta), FSUB(te, zup_inf)))))), bbb), beta);
                } else if (W(KS_TH, izz) != thi) {
                    tl = FDIV(FSUB(te, zup_inf), FMUL(beta, FSUB(W(KS_TH, izz), thi)));
                } else {
                    tl = 0.0f;
                }
                dlu = FADD(FSUB(zzz, dzt), tl);
            }
            zup_inf = zup;
        }
        float zdo = 0.0f, zdo_sup = 0.0f;
        float dld = FADD(W(KS_Z, iz), FDIV(W(KS_DZ, iz), 2.0f));
        zzz = 0.0f;
        for (int izz = iz; izz >= 2; --izz) {
            const float dzt = FDIV(FADD(W(KS_DZ, izz - 1), W(KS_DZ, izz)), 2.0f);
            zdo = FADD(zdo, FMUL(FMUL(beta, thi), dzt));
            zdo = FSUB(zdo, FDIV(FMUL(FMUL(beta, FADD(W(KS_TH, izz - 1), W(KS_TH, izz))), dzt), 2.0f));
            zzz = FADD(zzz, dzt);
            if (te < zdo && te >= zdo_sup) {
                const float bbb = FDIV(FSUB(W(KS_TH, izz), W(KS_TH, izz - 1)), dzt);
                float tl;
                if (bbb != 0.0f) {
                    const float q = FMUL(beta, FSUB(W(KS_TH, izz), thi));
                    tl = FDIV(FDIV(FADD(q, FSQRT(ke_max(0.0f, FADD(FMUL(q, q),
                              FMUL(FMUL(FMUL(2.0f, bbb), beta), FSUB(te, zdo_sup)))))), bbb), beta);
                } else if (W(KS_TH, izz) != thi) {
                    tl = FDIV(FSUB(te, zdo_sup), FMUL(beta, FSUB(W(KS_TH, izz), thi)));
                } else {
                    tl = 0.0f;
                }
                dld = FADD(FSUB(zzz, dzt), tl);
            }
            zdo_sup = zdo;
        }
        dld = ke_min(dld, FADD(W(KS_Z, iz), FDIV(W(KS_DZ, iz), 2.0f)));
        W(KS_DLS, iz) = FSQRT(FMUL(dlu, dld));
    }
    // ---- thetaphi_alphabeta (:883-944) -------------------------------------
    for (int iz = 2; iz <= nz; ++iz) {
        float te = W(KS_TE, iz), d = W(KS_D, iz);
        if (FDIV(fabsf(te), fabsf(d)) > 1e4f && d <= 1e-6f)
            d = FDIV(FMUL(KE_1O1P4, KE_POW(te, 1.5f)), W(KS_DLS, iz));
        const float theta = FDIV(ke_max(KE_TEMIN, te), ke_max(KE_DMIN, d));
        // max(temin,te)**alpha with alpha = 1: folded to the base by WRF's -O2.
        const float phi = FMUL(KE_POW(ke_max(KE_DMIN, d), KE_BETA), ke_max(KE_TEMIN, te));
        const float sh = W(KS_SH, iz), bu = W(KS_BU, iz), pra = W(KS_PRA, iz);
        float C = FMUL(KE_CMU, FADD(FMUL(KE_C1M1, sh), FDIV(FMUL(KE_C3M1, bu), pra)));
        const float A = FMUL(KE_CMU, FADD(sh, FDIV(bu, pra)));
        const float B = FMUL(KE_CMU, FADD(FMUL(KE_C1, sh), FDIV(FMUL(KE_C3, bu), pra)));
        float theta_new;
        if (C < 0.0f) {
            C = -C;
            const float sC = KE_POW(C, 0.5f);
            const float th_ = KE_TANH(FMUL(FMUL(sC, dt), KE_SQ_C2M1));
            const float num = FMUL(KE_SQ_C2M1, FADD(FSUB(FMUL(C, FMUL(theta, theta)), KE_C2), 1.0f));
            const float den = FMUL(FMUL(C, theta), FADD(KE_SQ_C2M1, FMUL(FMUL(sC, theta), th_)));
            theta_new = FADD(FDIV(KE_C2M1, FMUL(C, theta)), FDIV(num, den));
        } else if (C == 0.0f) {
            theta_new = FADD(theta, FMUL(KE_C2M1, dt));
        } else {
            const float a1 = KE_POW(FDIV(KE_C2M1, C), 0.5f);
            const float t = KE_TANH(FMUL(dt, KE_POW(FMUL(C, KE_C2M1), 0.5f)));
            const float b1 = KE_POW(FDIV(C, KE_C2M1), 0.5f);
            theta_new = FDIV(FMUL(a1, FADD(t, FMUL(theta, b1))),
                             FADD(FMUL(FMUL(theta, t), b1), 1.0f));
        }
        theta_new = ke_max(1.0f, theta_new);
        // alpha*A with alpha = 1 is A exactly.
        const float phi_new = FMUL(phi, KE_EXP(FMUL(FMUL(theta_new, dt),
                                                     FADD(A, FMUL(KE_BETA, B)))));
        te = KE_POW(FMUL(phi_new, KE_POW(theta_new, KE_BETA)), KE_INV_AB);
        W(KS_TE, iz) = te;
        W(KS_D, iz) = FDIV(te, theta_new);
    }
    for (int iz = 1; iz <= nz; ++iz) {
        float ad = 0.0f;                     // the driver's a_d1D = 0.
        const float ri = W(KS_RI, iz);
        if (ri > 0.0f)
            ad = FADD(ad, FMUL(FMUL(KE_C4, ke_min(1.0f, FSQRT(FDIV(ri, KE_C5)))),
                               FSQRT(ke_max(0.0f, -W(KS_NSQ, iz)))));
        W(KS_AD, iz) = ad;
    }
    {
        const float d1 = W(KS_D, 1);
        W(KS_D, 1) = FMUL(FMUL(FDIV(FDIV(KE_POW(ust, 3.0f), KE_VK), FSUB(dz1h, z0)), phieps),
                          FADD(1.0f, FMUL(FMUL(W(KS_AD, 1), d1), dt)));
        W(KS_TE, 1) = FMUL(FDIV(FMUL(ust, ust), KE_SQRT_CMU), FSQRT(FDIV(phieps, phim)));
    }
    // ---- diffusion of TKE and dissipation ----------------------------------
    ke_diff(col, nz, 2, dt, KS_TE, KS_EXCH, false, -1, 0.0f, -1, 0.0f);
    ke_diff(col, nz, 2, dt, KS_D, KS_EXCH, true, KS_AD, 0.0f, -1, 0.0f);
    // ---- terms_tpe (:1131-1147) --------------------------------------------
    for (int iz = 1; iz <= nz - 1; ++iz) {       // row kte is the solve's identity
        const float te = W(KS_TE, iz), d = W(KS_D, iz), tp = W(KS_TPE, iz);
        const float pra = W(KS_PRA, iz), dtm = W(KS_DTMDZ, iz);
        const float x = FMUL(FDIV(FMUL(FDIV(KE_CMU, pra), FMUL(te, te)), d), dtm);
        const float awt = FMUL(FDIV(FDIV(FDIV(1.0f, te), tp), 2.0f), FMUL(x, x));
        const float coeff = ke_max(0.55f, FMUL(1.5f, FADD(1.0f, awt)));
        W(KS_ATPE, iz) = FSUB(-FMUL(FDIV(FMUL(FMUL(gth0, KE_CTHETA), te), d), dtm),
                              FDIV(FMUL(coeff, d), te));
        W(KS_BTPE, iz) = FDIV(FMUL(FDIV(FMUL(KE_CMU, FMUL(te, te)), d), FMUL(dtm, dtm)), pra);
    }
    W(KS_ATPE, nz) = 0.0f;
    W(KS_BTPE, nz) = 0.0f;
    // ---- calc_countergradient (:1151-1176) -----------------------------------
    for (int iz = 1; iz <= nz; ++iz)
        W(KS_AUX, iz) = FDIV(FMUL(FMUL(FMUL(gth0, W(KS_TE, iz)), W(KS_TPE, iz)), KE_CTHETA),
                             W(KS_D, iz));
    W(KS_BT, 1) = FSUB(W(KS_BT, 1), 0.0f);        // dphidz(kts) = 0
    for (int iz = 2; iz <= nz - 1; ++iz) {
        const float p1 = FDIV(FMUL(2.0f, FSUB(W(KS_AUX, iz), W(KS_AUX, iz - 1))),
                              FADD(W(KS_DZ, iz - 1), W(KS_DZ, iz)));
        const float p2 = FDIV(FMUL(2.0f, FSUB(W(KS_AUX, iz + 1), W(KS_AUX, iz))),
                              FADD(W(KS_DZ, iz + 1), W(KS_DZ, iz)));
        W(KS_BT, iz) = FSUB(W(KS_BT, iz), FMUL(0.5f, FADD(p1, p2)));
    }
    W(KS_BT, nz) = FSUB(W(KS_BT, nz), 0.0f);      // dphidz(kte) = 0
    // ---- remaining solves, in keps1D's order --------------------------------
    ke_diff(col, nz, 1, dt, KS_TPE, KS_EXCH, false, KS_ATPE, 0.0f, KS_BTPE, 0.0f);
#ifdef GPUWM_WRF_EXACT
    ke_diff(col, nz, 1, dt, KS_U, KS_EXCH, false, -1, a_u1, -1, 0.0f);
    ke_diff(col, nz, 1, dt, KS_V, KS_EXCH, false, -1, a_u1, -1, 0.0f);
    ke_diff(col, nz, 1, dt, KS_TH, KS_EXCHH, false, -1, 0.0f, KS_BT, 0.0f);
    ke_diff(col, nz, 1, dt, KS_Q, KS_EXCHH, false, -1, 0.0f, -1, b_q1);
    ke_diff(col, nz, 1, dt, KS_QC, KS_EXCHH, false, -1, 0.0f, -1, 0.0f);
#else
    const int wind_slots[] = {KS_U, KS_V};
    const int wind_sources[] = {-1, -1};
    const float wind_surface[] = {0.0f, 0.0f};
    ke_diff_batch(col, nz, 1, dt, KS_EXCH, false, -1, a_u1,
                  wind_slots, wind_sources, wind_surface);
    const int scalar_slots[] = {KS_TH, KS_Q, KS_QC};
    const int scalar_sources[] = {KS_BT, -1, -1};
    const float scalar_surface[] = {0.0f, b_q1, 0.0f};
    ke_diff_batch(col, nz, 1, dt, KS_EXCHH, false, -1, 0.0f,
                  scalar_slots, scalar_sources, scalar_surface);
#endif
    for (int iz = 1; iz <= nz; ++iz) {
        W(KS_TE, iz) = ke_min(ke_max(KE_TEMIN, W(KS_TE, iz)), KE_TEMAX);
        W(KS_D, iz) = ke_min(ke_max(KE_DMIN, W(KS_D, iz)), KE_DMAX);
        W(KS_TPE, iz) = ke_min(ke_max(KE_TPEMIN, W(KS_TPE, iz)), KE_TPEMAX);
    }
    #undef GIN
    (void)hfx; (void)qfx; (void)b_t1;
}

// keps (module_bl_keps.F:13-378), one column per loop trip.
extern "C" __global__ void keps_columns(
    const float *__restrict__ u_in, const float *__restrict__ v_in,
    const float *__restrict__ th_in, const float *__restrict__ qv_in,
    const float *__restrict__ qc_in, const float *__restrict__ rho_in,
    const float *__restrict__ dz8w, const float *__restrict__ pi,
    const float *__restrict__ raten, const float *__restrict__ p8w,
    const float *__restrict__ ust_in, const float *__restrict__ hfx_in,
    const float *__restrict__ qfx_in, const float *__restrict__ br_in,
    const float *__restrict__ znt_in, const float *__restrict__ psim_in,
    const float *__restrict__ psih_in,
    float *tke_pbl, float *diss_pbl, float *tpe_pbl, float *pr_pbl,
    float *tke_adv, float *diss_adv, float *tpe_adv, float *pblh,
    float *rublten, float *rvblten, float *rthblten, float *rqvblten,
    float *rqcblten, float *wu, float *wv, float *wt, float *exch_h,
#if defined(KEPS_NZ) && !defined(GPUWM_WRF_EXACT)
    float *exch_m, float *ws, int nz_runtime, int ncol, float dt, int xtime,
#else
    float *exch_m, float *ws, int nz, int ncol, float dt, int xtime,
#endif
    float cp, float g)
{
#if defined(KEPS_NZ) && !defined(GPUWM_WRF_EXACT)
    const int nz = KEPS_NZ;
#endif
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile = gridDim.x * blockDim.x;
    KeCol col;
#ifdef GPUWM_WRF_EXACT
    col.base = ws + t;
    col.stride = (size_t)tile;
#else
    col.base = ws + (size_t)(t / 32) * KE_SLOTS * (nz + 2) * 32 + t % 32;
    col.stride = 32;
#endif
    col.nzp = nz + 2;
    for (int c = t; c < ncol; c += tile) {
        #define G(a, k) (a)[(size_t)((k) - 1) * ncol + c]
        const float ust = ust_in[c], hfx = hfx_in[c], qfx = qfx_in[c];
        const float rho1 = G(rho_in, 1), th1 = G(th_in, 1);
        // sflux = hfx/rho/cp + qfx/rho*(461./287.-1)*th
        const float sflux = FADD(FDIV(FDIV(hfx, rho1), cp),
                                 FMUL(FMUL(FDIV(qfx, rho1), KE_RVRD_M1), th1));
        W(KS_Z, 1) = 0.0f;
        for (int iz = 1; iz <= nz; ++iz) {
            if (xtime <= 2) {
                G(tke_pbl, iz) = KE_TEMIN; G(diss_pbl, iz) = KE_DMIN;
                G(tpe_pbl, iz) = KE_TPEMIN; G(tke_adv, iz) = KE_TEMIN;
                G(diss_adv, iz) = KE_DMIN; G(tpe_adv, iz) = KE_TPEMIN;
                G(pr_pbl, iz) = 1.0f;
            }
            W(KS_U, iz) = G(u_in, iz); W(KS_V, iz) = G(v_in, iz);
            W(KS_TH, iz) = G(th_in, iz); W(KS_Q, iz) = G(qv_in, iz);
            W(KS_QC, iz) = G(qc_in, iz); W(KS_RHO, iz) = G(rho_in, iz);
            W(KS_DZ, iz) = G(dz8w, iz);
            W(KS_PRA, iz) = G(pr_pbl, iz);
            W(KS_Z, iz + 1) = FADD(W(KS_Z, iz), W(KS_DZ, iz));
            W(KS_TE, iz) = ke_min(KE_TEMAX, ke_max(KE_TEMIN, G(tke_pbl, iz)));
            W(KS_D, iz) = ke_min(KE_DMAX, ke_max(KE_DMIN, G(diss_pbl, iz)));
            W(KS_TPE, iz) = ke_min(KE_TPEMAX, ke_max(KE_TPEMIN, G(tpe_pbl, iz)));
        }
        W(KS_RHOZ, 1) = W(KS_RHO, 1);
        for (int iz = 2; iz <= nz; ++iz)
            W(KS_RHOZ, iz) = FDIV(FADD(FMUL(W(KS_RHO, iz), W(KS_DZ, iz - 1)),
                                       FMUL(W(KS_RHO, iz - 1), W(KS_DZ, iz))),
                                  FADD(W(KS_DZ, iz - 1), W(KS_DZ, iz)));
        W(KS_RHOZ, nz + 1) = W(KS_RHO, nz);
        int iz_pbl;
        const float zi = ke_get_pblh(col, nz, iz_pbl);
        pblh[c] = zi;
        // The driver's source rows: only level 1 is nonzero.
        const float dz1 = W(KS_DZ, 1);
        for (int iz = 1; iz <= nz; ++iz) W(KS_BT, iz) = 0.0f;
        W(KS_BT, 1) = FDIV(FDIV(FDIV(hfx, dz1), rho1), cp);
        const float b_q1 = FDIV(FDIV(qfx, dz1), rho1);
        const float mu = ke_max(1.E-10f, fabsf(W(KS_U, 1)));
        const float mv = ke_max(1.E-10f, fabsf(W(KS_V, 1)));
        const float a_u1 = -FDIV(FDIV(FMUL(ust, ust), dz1),
                                 KE_POW(FADD(FMUL(mu, mu), FMUL(mv, mv)), 0.5f));
        const bool active = xtime > 2;
        if (active)
            ke_keps1d(col, nz, c, ncol, dt, ust, hfx, qfx, sflux, cp, g, br_in[c],
                      znt_in[c], psim_in[c], psih_in[c], zi, iz_pbl, a_u1,
                      W(KS_BT, 1), b_q1, pi, raten, p8w);
#ifdef GPUWM_WRF_EXACT
        for (int iz = 1; iz <= nz; ++iz) {
            G(tke_pbl, iz) = W(KS_TE, iz);
            G(diss_pbl, iz) = W(KS_D, iz);
            G(tpe_pbl, iz) = W(KS_TPE, iz);
            G(pr_pbl, iz) = W(KS_PRA, iz);
            // exch1D stays the zero it was initialised to when keps1D does
            // not run (xtime <= 2).
            const float ex = active ? W(KS_EXCH, iz) : 0.0f;
            G(exch_h, iz) = FDIV(ex, W(KS_PRA, iz));
            G(exch_m, iz) = ex;
            G(rthblten, iz) = FADD(0.0f, FDIV(FSUB(W(KS_TH, iz), G(th_in, iz)), dt));
            G(rqvblten, iz) = FADD(0.0f, FDIV(FSUB(W(KS_Q, iz), G(qv_in, iz)), dt));
            G(rqcblten, iz) = FADD(0.0f, FDIV(FSUB(W(KS_QC, iz), G(qc_in, iz)), dt));
            G(rublten, iz) = FADD(0.0f, FDIV(FSUB(W(KS_U, iz), G(u_in, iz)), dt));
            G(rvblten, iz) = FADD(0.0f, FDIV(FSUB(W(KS_V, iz), G(v_in, iz)), dt));
        }
        const float uu = FMUL(ust, ust);
        float wtk = FSUB(FDIV(FDIV(hfx, rho1), cp),
                         FMUL(FDIV(FMUL(rho1, FSUB(W(KS_TH, 1), th1)), dt), dz1));
        float wuk = FSUB(-uu, FMUL(FDIV(FMUL(rho1, FSUB(W(KS_U, 1), G(u_in, 1))), dt), dz1));
        float wvk = FSUB(-uu, FMUL(FDIV(FMUL(rho1, FSUB(W(KS_U, 1), G(u_in, 1))), dt), dz1)); // WRF :363
        G(wt, 1) = wtk; G(wu, 1) = wuk; G(wv, 1) = wvk;
        for (int iz = 2; iz <= nz; ++iz) {
            const float r = W(KS_RHO, iz), dz = W(KS_DZ, iz);
            wtk = FSUB(wtk, FMUL(FDIV(FMUL(r, FSUB(W(KS_TH, iz), G(th_in, iz))), dt), dz));
            wuk = FSUB(wuk, FMUL(FDIV(FMUL(r, FSUB(W(KS_U, iz), G(u_in, iz))), dt), dz));
            wvk = FSUB(wvk, FMUL(FDIV(FMUL(r, FSUB(W(KS_V, iz), G(v_in, iz))), dt), dz));
            G(wt, iz) = wtk; G(wu, iz) = wuk; G(wv, iz) = wvk;
        }
#else
    // Reuse the field deltas for tendencies and accumulated native fluxes.
    float wtk = 0.0f, wuk = 0.0f, wvk = 0.0f;
    const float uu = FMUL(ust, ust);
    for (int iz = 1; iz <= nz; ++iz) {
        const float th = W(KS_TH, iz), u = W(KS_U, iz), v = W(KS_V, iz);
        const float dth = FSUB(th, G(th_in, iz));
        const float du = FSUB(u, G(u_in, iz)), dv = FSUB(v, G(v_in, iz));
        const float r = W(KS_RHO, iz), dz = W(KS_DZ, iz);
        G(tke_pbl, iz) = W(KS_TE, iz);
        G(diss_pbl, iz) = W(KS_D, iz);
        G(tpe_pbl, iz) = W(KS_TPE, iz);
        G(pr_pbl, iz) = W(KS_PRA, iz);
        const float ex = active ? W(KS_EXCH, iz) : 0.0f;
        G(exch_h, iz) = FDIV(ex, W(KS_PRA, iz));
        G(exch_m, iz) = ex;
        G(rthblten, iz) = FADD(0.0f, FDIV(dth, dt));
        G(rqvblten, iz) = FADD(0.0f, FDIV(FSUB(W(KS_Q, iz), G(qv_in, iz)), dt));
        G(rqcblten, iz) = FADD(0.0f, FDIV(FSUB(W(KS_QC, iz), G(qc_in, iz)), dt));
        G(rublten, iz) = FADD(0.0f, FDIV(du, dt));
        G(rvblten, iz) = FADD(0.0f, FDIV(dv, dt));
        if (iz == 1) {
            wtk = FSUB(FDIV(FDIV(hfx, rho1), cp), FMUL(FDIV(FMUL(rho1, dth), dt), dz1));
            wuk = FSUB(-uu, FMUL(FDIV(FMUL(rho1, du), dt), dz1));
            wvk = wuk; // Native first-level y flux reads u, as before.
        } else {
            wtk = FSUB(wtk, FMUL(FDIV(FMUL(r, dth), dt), dz));
            wuk = FSUB(wuk, FMUL(FDIV(FMUL(r, du), dt), dz));
            wvk = FSUB(wvk, FMUL(FDIV(FMUL(r, dv), dt), dz));
        }
        G(wt, iz) = wtk; G(wu, iz) = wuk; G(wv, iz) = wvk;
    }
#endif
        #undef G
    }
}
