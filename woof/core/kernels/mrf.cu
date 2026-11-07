// MRF planetary-boundary-layer column scheme (bl_pbl_physics = 99).
//
// Transcribed from WRF v4.6.1 phys/module_bl_mrf.F (sha256 a1306d00...,
// the Hong-Pan 1996 nonlocal PBL: MRF wrapper, MRF2D column, TRIDI2).
// One CUDA thread owns one complete column.  The arithmetic is MRF's own,
// statement by statement and in WRF's operation order; it shares no code
// path with ysu.cu (YSU descends from this scheme but reorders most of
// these expressions), only the workspace placement pattern.
//
// INDEXING.  MRF is an MM5 transcription and indexes its column TOP-DOWN:
// K = 1 is the model top, K = KL = nz the lowest mass level, and the
// tridiagonal systems are assembled BOTTOM-UP in KK = kme-K.  The kernel
// keeps both conventions verbatim (workspace arrays are 1-based in K or
// KK, slot 0 unused) so every line below can be read against the Fortran;
// engine fields are (nz, ny, nx) bottom-up, reached through MRF_AT(field,K),
// which is module_bl_mrf.F's NK = kme-K.
//
// EXPONENTS (tools/mrf_wrf461_oracle/build-receipt: WRF's own -O2 object).
// (P1000mb/(PL*1000.))**ROVCP, (1.-APHI16*HOL1)**(-1./4.) and **(-1./2.)
// stay powf calls in WRF as built and call gfk_pow here, WOOF's own float32
// pow.  ZFAC**PFAC (PFAC = 2.0) is folded to ZFAC*ZFAC by gfortran at -O1
// and above, and every integer exponent is a multiply; they are multiplies
// here.  SQRT is a correctly rounded sqrtss in WRF and __fsqrt_rn here.
//
// DEAD STORES WRF COMPUTES AND NEVER READS, not evaluated here: THGB/DTG
// (the MOL line that read them is commented out, :782), E1/QGH (:548-549,
// QGH is never read), PSIQ (:773), TX, DZQ and SCR4 above level KL.  They
// are why TSK and ZNT reach WRF's MRF but not this kernel: neither touches
// an output.  REGIME and ZOL are INOUT in WRF and never assigned (the whole
// stability-class block :685-760 is commented out), so the scheme leaves
// them exactly as the surface layer wrote them; so does the caller here.
//
// DIVISIONS go through __fdiv_rn and square roots through __fsqrt_rn, so
// neither the strict nor the default compile can turn a division by a
// literal into a reciprocal multiply (A146) or relax a root.  Under
// GPUWM_WRF_EXACT=1 (--fmad=false --ftz=false) the result is WRF's words;
// fast numerics permits contraction at the shared compiler boundary.
// Flushing remains disabled in both modes through the no-FTZ loader.
//
// DECLARED DIVERGENCE: KPBL.  WRF stores KPBL2D = KPBL0 (:1284), MRF's
// TOP-DOWN index, into the same KPBL field every other PBL scheme fills
// with a bottom-up level number: a stable column reports KPBL = kte (the
// model top) and a 3 km convective PBL reports a level near the bottom.
// Every consumer of KPBL reads it bottom-up.  This kernel publishes the
// same level bottom-up, kpbl = nz + 1 - KPBL0; the oracle comparison maps
// WRF's word through that bijection and is exact.  Nothing inside the
// scheme reads KPBL1D, so no tendency moves.

#define MRF_BLOCK 32
#define MRFWS_SLOTS 11
#define MRFWS_LANES 32

// Coefficients come from the pinned gpuwm/data/mrf/constants.cuh asset.

//: A column array in the lane-interleaved workspace (the ysu.cu layout:
//: element K of slot s for lane t at block_base + (s*wskp + K)*LANES + t,
//: so a warp reading arr[K] touches 32 consecutive floats).
struct MrfCol {
    real *p;
    __device__ __forceinline__ real &operator[](int k) const {
        return p[(size_t)k * (size_t)MRFWS_LANES];
    }
};
#define MRFWS_AT(base, idx, kp) \
    (MrfCol{(base) + (size_t)(idx) * (size_t)(kp) * (size_t)MRFWS_LANES})
#define MRFWS_LANE_BASE(ws, kp) \
    ((ws) + (size_t)blockIdx.x * (size_t)MRFWS_SLOTS * (size_t)(kp) \
     * (size_t)MRFWS_LANES + (size_t)threadIdx.x)

// Fortran MIN/MAX as gfortran -O2 lowers them on x86-64 SSE (minss/maxss
// with the FIRST argument in the destination): the first argument is kept
// only when it compares strictly beyond the second, so a NaN in either
// position and a signed-zero tie both return the SECOND.
__device__ __forceinline__ real mrf_max(real a, real b) { return a > b ? a : b; }
__device__ __forceinline__ real mrf_min(real a, real b) { return a < b ? a : b; }

// BR .GT. 0.0 (:808) decided on the word, so a default (FTZ/DAZ) compile
// still takes WRF's branch for a subnormal BR; NaN compares false, as in
// Fortran.  Identical to `br > 0.0f` whenever subnormals are honoured.
__device__ __forceinline__ bool mrf_positive(real x) {
    return x == x && __float_as_int(x) > 0;
}

// TRIDI2 (module_bl_mrf.F:1291-1347), called as
//   CALL TRIDI2(AL,AD,AU,A1,A2,AU,A1,A2,its,ite,kts,kte)
// so CU aliases AU, R1 aliases A1 and R2 aliases A2, and CL is declared
// (its:ite, kts+1:kte+1): CL(K) is the caller's AL(K-1).
__device__ void mrf_tridi2(MrfCol al, MrfCol ad, MrfCol au, MrfCol a1,
                           MrfCol a2, int n) {
    real fk = __fdiv_rn(1.0f, ad[1]);
    au[1] = fk * au[1];
    a1[1] = fk * a1[1];
    a2[1] = fk * a2[1];
    for (int k = 2; k <= n - 1; ++k) {
        const real cl = al[k - 1];
        fk = __fdiv_rn(1.0f, ad[k] - cl * au[k - 1]);
        au[k] = fk * au[k];
        a1[k] = fk * (a1[k] - cl * a1[k - 1]);
        a2[k] = fk * (a2[k] - cl * a2[k - 1]);
    }
    {
        const real cl = al[n - 1];
        fk = __fdiv_rn(1.0f, ad[n] - cl * au[n - 1]);
        a1[n] = fk * (a1[n] - cl * a1[n - 1]);
        a2[n] = fk * (a2[n] - cl * a2[n - 1]);
    }
    for (int k = n - 1; k >= 1; --k) {
        a1[k] = a1[k] - au[k] * a1[k + 1];
        a2[k] = a2[k] - au[k] * a2[k + 1];
    }
}

extern "C" __global__ void mrf_column(
    const real *__restrict__ u, const real *__restrict__ v,
    const real *__restrict__ t, const real *__restrict__ qv,
    const real *__restrict__ qc, const real *__restrict__ qi,
    const real *__restrict__ p, const real *__restrict__ pi3d,
    const real *__restrict__ dz8w, const real *__restrict__ z,
    const real *__restrict__ psfc, real *__restrict__ ust,
    const real *__restrict__ psim, const real *__restrict__ psih,
    const real *__restrict__ xland, const real *__restrict__ hfx,
    const real *__restrict__ qfx, const real *__restrict__ gz1oz0,
    const real *__restrict__ wspd, const real *__restrict__ br,
    real dt, int flag_qi,
    real *__restrict__ rublten, real *__restrict__ rvblten,
    real *__restrict__ rthblten, real *__restrict__ rqvblten,
    real *__restrict__ rqcblten, real *__restrict__ rqiblten,
    real *__restrict__ pblh, real *__restrict__ hol_out,
    int *__restrict__ kpbl_out, int *__restrict__ kpbl_native_out,
    int nz, int ny, int nx, real *__restrict__ ws, int wskp, int col0)
{
    const int col = col0 + blockIdx.x * blockDim.x + threadIdx.x;
    const int ncol = ny * nx;
    if (col >= ncol) return;
    const size_t st = (size_t)ncol;
    // NK = kme - K (module_bl_mrf.F:201), kme = nz + 1, zero-based.
#define MRF_AT(f, K) ((f)[(size_t)(nz - (K)) * st + (size_t)col])

    real *lane = MRFWS_LANE_BASE(ws, wskp);
    MrfCol zq = MRFWS_AT(lane, 0, wskp), za = MRFWS_AT(lane, 1, wskp);
    MrfCol dza = MRFWS_AT(lane, 2, wskp), thvx = MRFWS_AT(lane, 3, wskp);
    MrfCol xkzm = MRFWS_AT(lane, 4, wskp), xkzh = MRFWS_AT(lane, 5, wskp);
    MrfCol au = MRFWS_AT(lane, 6, wskp), al = MRFWS_AT(lane, 7, wskp);
    MrfCol ad = MRFWS_AT(lane, 8, wskp), a1 = MRFWS_AT(lane, 9, wskp);
    MrfCol a2 = MRFWS_AT(lane, 10, wskp);

    const int KL = nz, KLM = nz - 1, KLP1 = nz + 1, KLPBL = 1;
    const real brc = br[col];
    const real xl = xland[col];
    const bool water = (xl - 1.5f) >= 0.0f;

    // :494 PS(I)=PSFCPA(I)/1000.
    const real ps = __fdiv_rn(psfc[col], 1000.0f);

    // :517-545 THX, THVX, SCR4 (only SCR4(KL) and THX(KL) are ever read).
    real thx_kl = 0.0f, scr4_kl = 0.0f;
    for (int K = 1; K <= nz; ++K) {
        const real pl = __fdiv_rn(MRF_AT(p, K), 1000.0f);
        const real scr3 = MRF_AT(t, K);
        const real thcon = gfk_pow(__fdiv_rn(MRF_P1000MB, pl * 1000.0f),
                                   MRF_ROVCP);
        const real thx = scr3 * thcon;
        const real tvcon = 1.0f + MRF_EP1 * MRF_AT(qv, K);
        thvx[K] = thx * tvcon;
        if (K == KL) { thx_kl = thx; scr4_kl = scr3 * tvcon; }
    }
    // :550 CPM(I)=CP*(1.+0.8*QX(I,KL))
    const real qx_kl = MRF_AT(qv, KL);
    const real cpm = MRF_CP * (1.0f + 0.8f * qx_kl);

    // :582-606 heights of full and half levels above ground.
    zq[KLP1] = 0.0f;
    const real rhox = __fdiv_rn(ps * 1000.0f, MRF_RD * scr4_kl);
    for (int KK = 1; KK <= nz; ++KK) {
        const int K = KLP1 - KK;
        zq[K] = MRF_AT(dz8w, K) + zq[K + 1];
    }
    for (int K = 1; K <= nz; ++K)
        za[K] = 0.5f * (zq[K] + zq[K + 1]);
    for (int K = 1; K <= nz - 1; ++K)
        dza[K] = za[K] - za[K + 1];
    // :611 GOVRTH(I)=G/THX(I,KL)
    const real govrth = __fdiv_rn(MRF_G, thx_kl);

    // :765-783 frictional velocity.
    real ustc;
    {
        const real psix = gz1oz0[col] - psim[col];
        ustc = __fdiv_rn(MRF_KARMAN * wspd[col], psix);
        const real ustm = mrf_max(ustc, 0.1f);
        if (!water) ustc = ustm;
    }
    ust[col] = ustc;
    // :786 WSPD1
    const real ux_kl = MRF_AT(u, KL), vx_kl = MRF_AT(v, KL);
    const real wspd1 = __fsqrt_rn(ux_kl * ux_kl + vx_kl * vx_kl) + 1.e-9f;

    // :795-811 preliminary variables.
    const real dt4 = 2.0f * dt;
    const real rdt = __fdiv_rn(1.0f, dt4);
    real hgamt = 0.0f, hgamq = 0.0f, wscale = 0.0f;
    int kpbl = KL, kpbl0 = KL;
    real pbl = zq[KL], pbl0 = zq[KL];
    bool pblflg = true;
    const bool sfcflg = !mrf_positive(brc);
    const real zl1 = za[KL];
    real thermal = thvx[KL];
    const real thvx_kl = thvx[KL];

    // :815-842 first guess of the PBL height.
    real brup = brc, brdn = 0.0f;
    {
        bool stable = false;
        for (int K = KLM; K >= KLPBL && !stable; --K) {
            brdn = brup;
            const real ux = MRF_AT(u, K), vx = MRF_AT(v, K);
            const real spdk2 = mrf_max(ux * ux + vx * vx, 1.0f);
            brup = __fdiv_rn((thvx[K] - thermal)
                             * __fdiv_rn(MRF_G * za[K], thvx_kl), spdk2);
            kpbl = K;
            stable = brup > MRF_BRCR;
        }
        const int K = kpbl;
        real brint;
        if (brdn >= MRF_BRCR) brint = 0.0f;
        else if (brup <= MRF_BRCR) brint = 1.0f;
        else brint = __fdiv_rn(MRF_BRCR - brdn, brup - brdn);
        pbl = za[K + 1] + brint * (za[K] - za[K + 1]);
        if (pbl < zq[kpbl + 1]) kpbl = kpbl + 1;
    }

    // :844-866 stability functions and the velocity scale.
    real phim, phih, hol;
    {
        const real fm = gz1oz0[col] - psim[col];
        const real fh = gz1oz0[col] - psih[col];
        hol = mrf_max(__fdiv_rn(brc * fm * fm, fh), MRF_RIMIN);
        if (sfcflg) hol = mrf_min(hol, -MRF_ZFMIN);
        else hol = mrf_max(hol, MRF_ZFMIN);
        const real hol1 = __fdiv_rn(hol * pbl, zl1) * MRF_SFCFRAC;
        hol = -__fdiv_rn(hol * pbl, zl1);
        if (sfcflg) {
            phim = gfk_pow(1.0f - MRF_APHI16 * hol1, -0.25f);
            phih = gfk_pow(1.0f - MRF_APHI16 * hol1, -0.5f);
        } else {
            phim = 1.0f + MRF_APHI5 * hol1;
            phih = phim;
        }
        wscale = __fdiv_rn(ustc, phim);
        wscale = mrf_min(wscale, ustc * MRF_APHI16);
        wscale = mrf_max(wscale, __fdiv_rn(ustc, MRF_APHI5));
    }

    // :871-885 surface variables for the unstable PBL height.
    if (sfcflg) {
        const real gamfac = __fdiv_rn(__fdiv_rn(MRF_CFAC, rhox), wscale);
        hgamt = mrf_min(__fdiv_rn(gamfac * hfx[col], cpm), MRF_GAMCRT);
        hgamq = mrf_min(gamfac * qfx[col], MRF_GAMCRQ);
        if (water) hgamq = 0.0f;
        real vpert = hgamt + MRF_EP1 * thx_kl * hgamq;
        vpert = mrf_min(vpert, MRF_GAMCRT);
        thermal = thermal + mrf_max(vpert, 0.0f);
        hgamt = mrf_max(hgamt, 0.0f);
        hgamq = mrf_max(hgamq, 0.0f);
    } else {
        pblflg = false;
    }
    if (pblflg) {
        kpbl = KL;
        pbl = zq[KL];
    }

    // :896-928 PBL height enhanced by the thermal.
    if (pblflg) {
        brup = brc;
        bool stable = false;
        for (int K = KLM; K >= KLPBL && !stable; --K) {
            brdn = brup;
            const real ux = MRF_AT(u, K), vx = MRF_AT(v, K);
            const real spdk2 = mrf_max(ux * ux + vx * vx, 1.0f);
            brup = __fdiv_rn((thvx[K] - thermal)
                             * __fdiv_rn(MRF_G * za[K], thvx_kl), spdk2);
            kpbl = K;
            stable = brup > MRF_BRCR;
        }
        const int K = kpbl;
        real brint;
        if (brdn >= MRF_BRCR) brint = 0.0f;
        else if (brup <= MRF_BRCR) brint = 1.0f;
        else brint = __fdiv_rn(MRF_BRCR - brdn, brup - brdn);
        pbl = za[K + 1] + brint * (za[K] - za[K + 1]);
        if (pbl < zq[kpbl + 1]) kpbl = kpbl + 1;
        if (kpbl <= 1) pblflg = false;
    }

    // :932-965 diagnostic PBL height with BRCR effectively zero (PBL0).
    if (pblflg) {
        brup = brc;
        bool stable = false;
        for (int K = KLM; K >= KLPBL && !stable; --K) {
            brdn = brup;
            const real ux = MRF_AT(u, K), vx = MRF_AT(v, K);
            const real spdk2 = mrf_max(ux * ux + vx * vx, 1.0f);
            brup = __fdiv_rn((thvx[K] - thermal)
                             * __fdiv_rn(MRF_G * za[K], thvx_kl), spdk2);
            kpbl0 = K;
            stable = brup > 0.0f;
        }
        const int K = kpbl0;
        real brint;
        if (brdn >= 0.0f) brint = 0.0f;
        else if (brup <= 0.0f) brint = 1.0f;
        else brint = __fdiv_rn(0.0f - brdn, brup - brdn);
        pbl0 = za[K + 1] + brint * (za[K] - za[K + 1]);
        if (pbl0 < zq[kpbl0 + 1]) kpbl0 = kpbl0 + 1;
        if (kpbl0 <= 1) pblflg = false;
    }

    // :970-986 diffusion coefficients below the PBL.
    for (int K = nz; K >= KLPBL; --K) {
        if (kpbl < K) {
            real prnum = __fdiv_rn(phih, phim)
                         + MRF_CFAC * MRF_KARMAN * MRF_SFCFRAC;
            prnum = mrf_min(prnum, MRF_PRMAX);
            prnum = mrf_max(prnum, MRF_PRMIN);
            const real zfac = mrf_max(
                1.0f - __fdiv_rn(zq[K] - zl1, pbl - zl1), MRF_ZFMIN);
            const real xkzo = MRF_CKZ * dza[K - 1];
            real km = xkzo + wscale * MRF_KARMAN * zq[K] * (zfac * zfac);
            real kh = __fdiv_rn(km, prnum);
            km = mrf_min(km, MRF_XKZMAX);
            km = mrf_max(km, MRF_XKZMIN);
            kh = mrf_min(kh, MRF_XKZMAX);
            kh = mrf_max(kh, MRF_XKZMIN);
            xkzm[K] = km;
            xkzh[K] = kh;
        }
    }

    // :990-1032 diffusion coefficients over the PBL (free atmosphere).
    for (int K = KLPBL + 1; K <= nz; ++K) {
        const real xkzo = MRF_CKZ * dza[K - 1];
        if (K <= kpbl) {
            const real du = MRF_AT(u, K - 1) - MRF_AT(u, K);
            const real dv = MRF_AT(v, K - 1) - MRF_AT(v, K);
            const real ss = __fdiv_rn(du * du + dv * dv,
                                      dza[K - 1] * dza[K - 1]) + 1.e-9f;
            real ri = __fdiv_rn(govrth * (thvx[K - 1] - thvx[K]),
                                ss * dza[K - 1]);
            const real qik = flag_qi ? MRF_AT(qi, K) : 0.0f;
            const real qikm = flag_qi ? MRF_AT(qi, K - 1) : 0.0f;
            if ((MRF_AT(qc, K) + qik) > MRF_QCLOUD
                    && (MRF_AT(qc, K - 1) + qikm) > MRF_QCLOUD) {
                const real qmean = 0.5f * (MRF_AT(qv, K) + MRF_AT(qv, K - 1));
                const real tmean = 0.5f * (MRF_AT(t, K) + MRF_AT(t, K - 1));
                const real alph = __fdiv_rn(
                    __fdiv_rn(MRF_XLV * qmean, MRF_RD), tmean);
                const real chi = __fdiv_rn(__fdiv_rn(__fdiv_rn(__fdiv_rn(
                    MRF_XLV * MRF_XLV * qmean, MRF_CP), MRF_RV), tmean), tmean);
                ri = (1.0f + alph)
                     * (ri - __fdiv_rn(__fdiv_rn(__fdiv_rn(
                            MRF_G * MRF_G, ss), tmean), MRF_CP)
                         * __fdiv_rn(chi - alph, 1.0f + chi));
            }
            const real zk = MRF_KARMAN * zq[K];
            const real rl = __fdiv_rn(zk * MRF_RLAM, MRF_RLAM + zk);
            const real rl2 = rl * rl;
            const real dk = rl2 * __fsqrt_rn(ss);
            real km, kh;
            if (ri < 0.0f) {
                const real sri = __fsqrt_rn(-ri);
                km = xkzo + dk * (1.0f + __fdiv_rn(8.0f * (-ri),
                                                  1.0f + 1.746f * sri));
                kh = xkzo + dk * (1.0f + __fdiv_rn(8.0f * (-ri),
                                                  1.0f + 1.286f * sri));
            } else {
                const real d = 1.0f + 5.0f * ri;
                kh = xkzo + __fdiv_rn(dk, d * d);
                real prnum = 1.0f + 2.1f * ri;
                prnum = mrf_min(prnum, MRF_PRMAX);
                km = (kh - xkzo) * prnum + xkzo;
            }
            km = mrf_min(km, MRF_XKZMAX);
            km = mrf_max(km, MRF_XKZMIN);
            kh = mrf_min(kh, MRF_XKZMAX);
            kh = mrf_max(kh, MRF_XKZMIN);
            xkzm[K] = km;
            xkzh[K] = kh;
        }
    }

    const real goc = __fdiv_rn(MRF_G, MRF_CP);

    // :1046-1077 heat and moisture matrix.
    ad[1] = 1.0f;
    a1[1] = MRF_AT(t, KL)
            + __fdiv_rn(__fdiv_rn(hfx[col], rhox * cpm), zq[KL]) * dt4;
    a2[1] = qx_kl + __fdiv_rn(__fdiv_rn(qfx[col], rhox), zq[KL]) * dt4;
    for (int K = nz; K >= KLPBL + 1; --K) {
        const int KK = KLP1 - K;
        const real dtodsd = __fdiv_rn(dt4, MRF_AT(dz8w, K));
        const real dtodsu = __fdiv_rn(dt4, MRF_AT(dz8w, K - 1));
        const real dsig = -(MRF_AT(z, K) - MRF_AT(z, K - 1));
        const real rdz = __fdiv_rn(1.0f, dza[K - 1]);
        real dsdzt;
        if (pblflg && kpbl < K) {
            dsdzt = dsig * xkzh[K] * rdz * (goc - __fdiv_rn(hgamt, pbl));
            const real dsdzq = dsig * xkzh[K] * rdz
                               * (-__fdiv_rn(hgamq, pbl));
            a2[KK] = a2[KK] + dtodsd * dsdzq;
            a2[KK + 1] = MRF_AT(qv, K - 1) - dtodsu * dsdzq;
        } else {
            dsdzt = dsig * xkzh[K] * rdz * goc;
            a2[KK + 1] = MRF_AT(qv, K - 1);
        }
        const real dsdz2 = dsig * xkzh[K] * rdz * rdz;
        au[KK] = -(dtodsd * dsdz2);
        al[KK] = -(dtodsu * dsdz2);
        ad[KK] = ad[KK] - au[KK];
        ad[KK + 1] = 1.0f - al[KK];
        a1[KK] = a1[KK] + dtodsd * dsdzt;
        a1[KK + 1] = MRF_AT(t, K - 1) - dtodsu * dsdzt;
    }
    mrf_tridi2(al, ad, au, a1, a2, nz);
    // :1086-1094 heat and moisture tendencies; the wrapper divides the
    // temperature tendency by PI3D (:233-237).
    for (int K = nz; K >= 1; --K) {
        const int KK = KLP1 - K;
        const real ttend = (a1[KK] - MRF_AT(t, K)) * rdt;
        const real qtend = (a2[KK] - MRF_AT(qv, K)) * rdt;
        MRF_AT(rthblten, K) = __fdiv_rn(0.0f + ttend, MRF_AT(pi3d, K));
        MRF_AT(rqvblten, K) = 0.0f + qtend;
    }

    // :1098-1132 momentum matrix.
    {
        const real r = __fdiv_rn(wspd1, wspd[col]);
        ad[1] = 1.0f;
        a1[1] = ux_kl - __fdiv_rn(__fdiv_rn(ux_kl, wspd1) * ustc * ustc,
                                  zq[KL]) * dt4 * (r * r);
        a2[1] = vx_kl - __fdiv_rn(__fdiv_rn(vx_kl, wspd1) * ustc * ustc,
                                  zq[KL]) * dt4 * (r * r);
    }
    for (int K = nz; K >= KLPBL + 1; --K) {
        const int KK = KLP1 - K;
        const real dtodsd = __fdiv_rn(dt4, MRF_AT(dz8w, K));
        const real dtodsu = __fdiv_rn(dt4, MRF_AT(dz8w, K - 1));
        const real dsig = -(MRF_AT(z, K) - MRF_AT(z, K - 1));
        const real rdz = __fdiv_rn(1.0f, dza[K - 1]);
        const real dsdz2 = dsig * xkzm[K] * rdz * rdz;
        au[KK] = -(dtodsd * dsdz2);
        al[KK] = -(dtodsu * dsdz2);
        ad[KK] = ad[KK] - au[KK];
        ad[KK + 1] = 1.0f - al[KK];
        a1[KK + 1] = MRF_AT(u, K - 1);
        a2[KK + 1] = MRF_AT(v, K - 1);
    }
    mrf_tridi2(al, ad, au, a1, a2, nz);
    for (int K = nz; K >= 1; --K) {
        const int KK = KLP1 - K;
        MRF_AT(rublten, K) = 0.0f + (a1[KK] - MRF_AT(u, K)) * rdt;
        MRF_AT(rvblten, K) = 0.0f + (a2[KK] - MRF_AT(v, K)) * rdt;
    }

    // :1153-1201 cloud water and ice matrix.
    ad[1] = 1.0f;
    a1[1] = MRF_AT(qc, KL);
    a2[1] = flag_qi ? MRF_AT(qi, KL) : 0.0f;
    for (int K = nz; K >= KLPBL + 1; --K) {
        const int KK = KLP1 - K;
        const real dtodsd = __fdiv_rn(dt4, MRF_AT(dz8w, K));
        const real dtodsu = __fdiv_rn(dt4, MRF_AT(dz8w, K - 1));
        const real dsig = -(MRF_AT(z, K) - MRF_AT(z, K - 1));
        const real rdz = __fdiv_rn(1.0f, dza[K - 1]);
        a1[KK + 1] = MRF_AT(qc, K - 1);
        a2[KK + 1] = flag_qi ? MRF_AT(qi, K - 1) : 0.0f;
        const real dsdz2 = dsig * xkzh[K] * rdz * rdz;
        au[KK] = -(dtodsd * dsdz2);
        al[KK] = -(dtodsu * dsdz2);
        ad[KK] = ad[KK] - au[KK];
        ad[KK + 1] = 1.0f - al[KK];
    }
    mrf_tridi2(al, ad, au, a1, a2, nz);
    for (int K = nz; K >= 1; --K) {
        const int KK = KLP1 - K;
        MRF_AT(rqcblten, K) = 0.0f + (a1[KK] - MRF_AT(qc, K)) * rdt;
        // :1259-1266 RQIBLTEN only under FLAG_QI.
        if (flag_qi)
            MRF_AT(rqiblten, K) = 0.0f + (a2[KK] - MRF_AT(qi, K)) * rdt;
    }

    // :1283-1286 KPBL1D = KPBL0, PBL = PBL0; HOL from :855.
    pblh[col] = pbl0;
    hol_out[col] = hol;
    kpbl_out[col] = nz + 1 - kpbl0;
    kpbl_native_out[col] = kpbl0;
#undef MRF_AT
}
