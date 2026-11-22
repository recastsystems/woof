// Shin-Hong scale-aware PBL (bl_pbl_physics=11), WRF v4.8.0 generation.
//
// Transcribed statement for statement from WRF v4.8.0
// phys/physics_mmm/bl_shinhong.F90 (MMM physics 550b5b49, sha256 7d258836...)
// called through phys/module_bl_shinhong.F (sha256 c7bfe363...), the
// wrapper module_pbl_driver.F calls.  The :NNNN line numbers below are
// bl_shinhong.F90's.  The unit is compiled with kernels/shinhong.cu
// prepended (kernels/__init__.py header map): the five partition functions,
// the tridiagonal factor/solve pair, mixlen, vdifq, the float32 tanhf and the
// column workspace layout are the 4.6.1 unit's own words, because 4.8.0 left
// them unchanged (pu/pq/pthnl/pthl/ptke, mixlen and vdifq differ only in
// declarations; tridin_ysu moved its LU band to a local, the same
// arithmetic).  What 4.8.0 changed is written here:
//
//   * the nonlocal heat flux: an LES-type mass-flux profile
//     (shinhong_nonlocal_flux, default .true.) or YSU's counter-gradient
//     term (.false.); the 4.6.1 rigs/enlfrac2 entrainment fraction and the
//     double pthnl scaling of mf are gone, pthnl now scales the assembled
//     term once (:1217-1224);
//   * optional stratocumulus-top downdraft mixing (shinhong_scu_mixing,
//     default .false.): a second PBL-top search on liquid-ice potential
//     temperature and a radiatively driven entrainment (:794-861);
//   * optional kinetic-energy dissipation heating
//     (shinhong_ke_dissipation, default .false.) (:1529-1543);
//   * the scale-aware length is dx2d (= dx, module_physics_init.F
//     compute_2d_dx_area), not sqrt(dx*dy);
//   * the surface drag splits by the bulk Richardson number,
//     ad(1) = 1 + tau0*hvalue + ctopo*tau0*(1-hvalue) (:1424-1428), and the
//     sea-surface current enters the first-level wind (:1430-1431);
//   * the TKE chain always runs (if_tke_diag = .true. parameter), with new
//     floors: q2xk >= 1e-4 (:1588), the hgame denominator >= 0.01 (:1305),
//     prodq2's q2l >= epsq2l and disel >= 0.01 (:2344, :2367); efxpbl's
//     q2xk index is bounded (:1625-1628);
//   * the wrapper publishes exch_m as well as exch_h.
//
// Arithmetic spelling, as in shinhong.cu and measured against the referee
// object (25 powf, 6 expf, 1 tanhf call sites in bl_shinhong.o): x**2 and
// x**2. fold to multiplies; every other REAL power is glibc's powf
// (gfk_pow); EXP is gfk_exp; TANH is sh_tanhf in the strict build.
// MAX/MIN follow the referee's -O2 code, sh8_maxc/sh8_minc below.
//
// Declared divergences, both upstream reads of words WRF never defines:
//   * scu mixing, kpbl == kte with a cloudy top: :805 reads thlix/thx/qvx/
//     qcxl at level kpbl+1 = kte+1, past the column; this port reads kte.
//   * scu mixing, a column the cloud-top search makes pblflg without
//     sflux > 0: :1189-1196 reads amf2/bmf2 left over from an earlier column
//     of the tile (uninitialised on the first); this port uses 0.
// Neither arm is reachable with shinhong_scu_mixing = .false., the default.

#define SH8_NONLOCAL 1
#define SH8_SCU 2
#define SH8_DISSIP 4

// MAX/MIN as the referee's gfortran 13 -O2 code evaluates them, measured in
// the referee container on NaN and signed-zero operands.  Against a
// literal or PARAMETER bound c, MAX(x,c) returns x when x > c or x is NaN,
// else c, and MIN(x,c) returns c when c < x, else x: a NaN propagates, a
// tie keeps c for MAX and x for MIN (the operand order in the source does
// not matter).  Between two variables the compiler picks the operand order
// per site, so only ordered, non-tied operands have one answer; those sites
// use the plain compare.
__device__ __forceinline__ real sh8_maxc(real x, real c) {
    return (x > c || x != x) ? x : c;
}
__device__ __forceinline__ real sh8_minc(real x, real c) {
    return (c < x) ? c : x;
}
__device__ __forceinline__ real sh8_max(real a, real b) { return a > b ? a : b; }
__device__ __forceinline__ real sh8_min(real a, real b) { return a < b ? a : b; }

// prodq2 (:2312-2416), one column.  4.8.0 floors q2l on entry and disel.
__device__ void sh8_prodq2(real dtturbl, real ustar,
                           const real *u, const real *v, const real *theta,
                           const real *exner,
                           int st, int col, ShColC thvx,
                           ShCol q2, ShColC el, ShColC z,
                           ShColC akm, ShColC akh,
                           real hgamu, real hgamv, real hgamq, real delxy,
                           real hpbl, bool pblflg, int kpbl,
                           ShColC mf, ShColC ufxpbl,
                           ShColC vfxpbl, ShColC qfxpbl, int kte) {
    const real g_qnse = 9.81f;
    const real c0 = 0.55f, ceps = 16.6f;
    const real epsq2l = 0.01f;
    const real ep1 = RV / RD - 1.0f;
    const real rc02 = 2.0f / (c0 * c0);
    for (int k = 2; k <= kte; ++k) {
        real deltaz = 0.5f * (z[k + 1] - z[k - 1]);
        real q2l = sh8_maxc(q2[k], epsq2l);       // :2344 (new floor)
        real suk = (SH_UX(k) - SH_UX(k - 1)) / deltaz;
        real svk = (SH_VX(k) - SH_VX(k - 1)) / deltaz;
        real gthvk = (thvx[k] - thvx[k - 1]) / deltaz;
        real govrthvk = g_qnse / (0.5f * (thvx[k] + thvx[k - 1]));
        real akml = akm[k];
        real akhl = akh[k];
        real thm = (SH_THX(k) + SH_THX(k - 1)) * 0.5f;
        real pru, prv;
        if (pblflg && k <= kpbl) {
            pru = (akml * (suk - hgamu / hpbl - ufxpbl[k])) * suk;
            prv = (akml * (svk - hgamv / hpbl - vfxpbl[k])) * svk;
        } else {
            pru = akml * suk * suk;
            prv = akml * svk * svk;
        }
        real pr = pru + prv;
        real bpr;
        if (pblflg && k <= kpbl) {
            bpr = (akhl * (gthvk - mf[k]
                           - (hgamq / hpbl + qfxpbl[k]) * ep1 * thm))
                * govrthvk;
        } else {
            bpr = akhl * gthvk * govrthvk;
        }
        real disel = sh8_min(delxy, ceps * el[k]);
        disel = sh8_maxc(disel, 0.01f);           // :2367 (new floor)
        real dis = gfk_pow(q2l, 1.5f) / disel;
        q2l = q2l + 2.0f * (pr - bpr - dis) * dtturbl;
        q2[k] = sh8_maxc(q2l, epsq2l);
    }
    q2[1] = sh8_maxc(rc02 * ustar * ustar, epsq2l);
}

template <bool TOPO>
__device__ __forceinline__
void shinhong480_column_body(
        const real *u, const real *v, const real *theta,
        const real *qv, const real *qc, const real *qi,
        const real *p, const real *p_interface,
        const real *exner, const real *dz, const real *tke,
        const real *rthraten,
        const real *psfc, const real *znt, const real *ust,
        const real *hfx, const real *qfx, const real *wspd,
        const real *br, const real *psim, const real *psih,
        const real *xland, const real *u10, const real *v10,
        const real *corf, const real *uoce, const real *voce,
        real *du, real *dv, real *dtheta, real *dqv, real *dqc,
        real *dqi, real *exch_h, real *exch_m, real *tke_out, real *el,
        real *hpbl_out, int *kpbl_out, real *wstar_out,
        real *delta_out, real *u10_out, real *v10_out,
        real dt, real dx, int flags,
        int nz, int ny, int nx, real *ws, int wskp, int col0,
        const real *ctopo, const real *ctopo2) {
    int col = col0 + blockDim.x * blockIdx.x + threadIdx.x;
    int st = ny * nx;
    if (col >= st || nz > SHINHONG_KMAX) return;
    real *wsb = SHWS_LANE_BASE(ws, wskp);
    const int kte = nz;
    const int klpbl = kte;
    const bool if_nonlocal = (flags & SH8_NONLOCAL) != 0;
    const bool if_scu = (flags & SH8_SCU) != 0;
    const bool if_dissip = (flags & SH8_DISSIP) != 0;

    // Parameters (:79-130).
    const real xkzminm = 0.1f, xkzminh = 0.01f, xkzmax = 1000.0f;
    const real rimin = -100.0f, rlam = 30.0f, prmin = 0.25f, prmax = 4.0f;
    const real brcr_ub = 0.0f, brcr_sb = 0.25f, cori = 1.0e-4f;
    const real afac = 6.8f, bfac = 6.8f;
    const real phifac = 8.0f, sfcfrac = 0.1f;
    const real d1 = 0.02f, d2 = 0.05f, d3 = 0.001f;
    const real h1 = 0.333333f, h2 = 0.666666f;
    const real zfmin = 1.0e-8f, aphi5 = 5.0f, aphi16 = 16.0f;
    const real tmin = 1.0e-2f, gamcrt = 3.0f, gamcrq = 2.0e-3f;
    const real qci_cr = 0.01e-3f;
    const real mltop = 1.0f, sfcfracn1 = 0.075f, entfracn1 = 0.87f;
    const real fent = 2.0f, cent = -0.2f, nlfrac = 0.7f;
    const real a11 = 1.0f, a12 = -1.15f, scu_ent_max = 0.4f;
    const real epsq2l = 0.01f, c_1 = 1.0f, gamcre = 0.224f;
    // The driver's constants (module_model_constants, float32 forming).
    const real karman = 0.4f, ep1 = RV / RD - 1.0f, ep2 = RD / RV;
    const real rovcp = RD / CP;
    // BEP off (flag_bep false): bepswitch = 0, a_* = b_* = 0, sfk = vlk = 1
    // (:395-410).  The arithmetic that reads them is spelled as WRF spells it.
    const real bepswitch = 0.0f, bep0 = 0.0f, bep1 = 1.0f;

    ShCol thvx = SHWS_AT(wsb, 12, wskp);
    ShCol zq = SHWS_AT(wsb, 13, wskp);
    ShCol za = SHWS_AT(wsb, 14, wskp);
    ShCol delp = SHWS_AT(wsb, 15, wskp);
    ShCol dza = SHWS_AT(wsb, 16, wskp);
    ShCol q2x = SHWS_AT(wsb, 17, wskp);
    ShCol hgame2d = SHWS_AT(wsb, 18, wskp);
    ShCol tflux_e = SHWS_AT(wsb, 19, wskp);
    ShCol qflux_e = SHWS_AT(wsb, 20, wskp);
    ShCol tvflux_e = SHWS_AT(wsb, 21, wskp);
    ShCol mf = SHWS_AT(wsb, 22, wskp);
    ShCol entfacmf = SHWS_AT(wsb, 23, wskp);
    ShCol entfac = SHWS_AT(wsb, 24, wskp);
    ShCol xkzm = SHWS_AT(wsb, 25, wskp);
    ShCol xkzh = SHWS_AT(wsb, 26, wskp);
    ShCol xkzq = SHWS_AT(wsb, 27, wskp);
    ShCol xkzml = SHWS_AT(wsb, 28, wskp);
    ShCol xkzhl = SHWS_AT(wsb, 29, wskp);
    ShCol zfacent = SHWS_AT(wsb, 30, wskp);
    ShCol al = SHWS_AT(wsb, 31, wskp);
    ShCol ad = SHWS_AT(wsb, 32, wskp);
    ShCol au = SHWS_AT(wsb, 33, wskp);
    ShCol fkk = SHWS_AT(wsb, 34, wskp);
    ShCol lau = SHWS_AT(wsb, 35, wskp);
    ShCol r1 = SHWS_AT(wsb, 36, wskp);
    ShCol r2 = SHWS_AT(wsb, 37, wskp);
    ShCol r3 = SHWS_AT(wsb, 38, wskp);
    for (int k = 0; k < wskp; ++k) {
        thvx[k] = zq[k] = za[k] = delp[k] = dza[k] = 0.0f;
        q2x[k] = hgame2d[k] = 0.0f;
        tflux_e[k] = qflux_e[k] = tvflux_e[k] = 0.0f;
        mf[k] = entfacmf[k] = entfac[k] = 0.0f;
        xkzm[k] = xkzh[k] = xkzq[k] = xkzml[k] = xkzhl[k] = zfacent[k] = 0.0f;
        al[k] = ad[k] = au[k] = fkk[k] = lau[k] = 0.0f;
        r1[k] = r2[k] = r3[k] = 0.0f;
    }

    const real psfcv = psfc[col], zntv = znt[col], ustv = ust[col];
    const real hfxv = hfx[col], qfxv = qfx[col], wspdv = wspd[col];
    const real brv = br[col], psimv = psim[col], psihv = psih[col];
    const real xlandv = xland[col], corfv = corf[col];
    const real u10v = u10[col], v10v = v10[col];
    const real uoxl = uoce ? uoce[col] : 0.0f;   // :376-386
    const real voxl = voce ? voce[col] : 0.0f;
    const real ctopov = TOPO ? ctopo[col] : 1.0f;
    const real ctopo2v = TOPO ? ctopo2[col] : 1.0f;

    const real cont = CP / G;                    // :343
    const real conw = 1.0f / G;
    const real conwrc = conw * sqrtf(1.0f);      // conw*sqrt(rcl), rcl = 1
    const real conpr = bfac * karman * sfcfrac;

    // thx, thlix, qci (:360-368); thvx (:369-374).  thlix (scu only) is
    // formed where it is read, from the same words.
#define SH8_THLIX(k) __fdiv_rn(SH_TX(k) - XLV * SH_QCX(k) / CP \
                               - 2.834e6f * SH_QIX(k) / CP, SH_PI2DX(k))
#define SH8_QCI(k) (SH_QCX(k) + SH_QIX(k))
    for (int k = 1; k <= kte; ++k) {
        real tvcon = 1.0f + ep1 * SH_QVX(k);
        thvx[k] = SH_THX(k) * tvcon;
    }
    real tvcon = 1.0f + ep1 * SH_QVX(1);         // :387-391
    real rhox = psfcv / (RD * SH_TX(1) * tvcon);
    real govrth = G / SH_THX(1);
#define SH8_RHOX2(k) (SH_P2DX(k) / (RD * SH_TX(k) * (1.0f + ep1 * SH_QVX(k))))
    const real dxy = dx;                         // :392-396 dxy = dxmeter

    zq[1] = 0.0f;                                // :411-420
    for (int k = 1; k <= kte; ++k)
        zq[k + 1] = SH_DZ8W(k) + zq[k];
    for (int k = 1; k <= kte; ++k) {
        za[k] = 0.5f * (zq[k] + zq[k + 1]);
        delp[k] = p_interface[(size_t)(k - 1) * st + col]
                - p_interface[(size_t)k * st + col];
    }
    dza[1] = za[1];
    for (int k = 2; k <= kte; ++k)
        dza[k] = za[k] - za[k - 1];

    // :440-449 zero exch_h/exch_m; levels 2..kte are rewritten below
    // (exch_hx(k+1) = xkzh(k), k = 1..kte-1), so only level 1 is stored.
    SH_G(exch_h, 1) = 0.0f;
    SH_G(exch_m, 1) = 0.0f;
    // :450-453 wspd1 = sqrt((u-uox)^2 + (v-vox)^2) + 1e-9
    real wspd1 = sqrtf((SH_UX(1) - uoxl) * (SH_UX(1) - uoxl)
                       + (SH_VX(1) - voxl) * (SH_VX(1) - voxl)) + 1.0e-9f;
    real dt2 = 2.0f * dt;
    real rdt = 1.0f / dt2;

    real hgamu = 0.0f, hgamv = 0.0f, delta = 0.0f, delta_sh = 0.0f;
    real wstar3 = 0.0f, wstar3_2 = 0.0f, hfxpbl_sh = 0.0f;
    real efxpbl = 0.0f, epshol = 0.0f, deltaoh = 0.0f, entfrac = 0.0f;
    real cslen = 0.0f;
    for (int k = 1; k <= kte; ++k)               // :500-504
        q2x[k] = 2.0f * SH_TKEX(k);
    // :505-513 el_pbl = 0; the always-on TKE chain rewrites k >= 2.
    SH_G(el, 1) = 0.0f;

    real we = 0.0f, hgamt = 0.0f, hgamq = 0.0f, wscale = 0.0f;   // :534-549
    int kpbl = 1;
    real hpbl = zq[1];
    real zl1 = za[1];
    real thermal = thvx[1];
    real thermalli = SH8_THLIX(1);
    bool pblflg = true;
    bool sfcflg = true;
    real sflux = __fdiv_rn(hfxv / rhox, CP) + qfxv / rhox * ep1 * SH_THX(1);
    // br > 0 compared in double from the word's bits, shinhong.cu's sh_f2d.
    if (sh_f2d(brv) > 0.0)
        sfcflg = false;

    // ---- first guess of the PBL height (:550-575) ----
    bool stable = false;
    real brup = brv;
    real brcr = brcr_ub;
    real brdn = 0.0f;
    for (int k = 2; k <= klpbl; ++k) {
        if (!stable) {
            brdn = brup;
            real spdk2 = sh8_maxc(SH_UX(k) * SH_UX(k) + SH_VX(k) * SH_VX(k),
                                 1.0f);
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2;
            kpbl = k;
            stable = brup > brcr;
        }
    }
    {
        int k = kpbl;
        real brint;
        if (brdn >= brcr) brint = 0.0f;
        else if (brup <= brcr) brint = 1.0f;
        else brint = (brcr - brdn) / (brup - brdn);
        hpbl = za[k - 1] + brint * (za[k] - za[k - 1]);
        if (hpbl < zq[2]) kpbl = 1;
        if (kpbl <= 1) pblflg = false;
    }

    // ---- surface-layer scales (:576-603) ----
    real zol1 = sh8_maxc(brv * psimv * psimv / psihv, rimin);
    if (sfcflg) zol1 = sh8_minc(zol1, -zfmin);
    else zol1 = sh8_maxc(zol1, zfmin);
    real hol1 = zol1 * hpbl / zl1 * sfcfrac;
    epshol = hol1;
    real phim, phih, wstar;
    if (sfcflg) {
        phim = gfk_pow(1.0f - aphi16 * hol1, -1.0f / 4.0f);
        phih = gfk_pow(1.0f - aphi16 * hol1, -1.0f / 2.0f);
        real bfx0 = sh8_maxc(sflux, 0.0f);
        wstar3 = govrth * bfx0 * hpbl;
        wstar = gfk_pow(wstar3, h1);
    } else {
        phim = 1.0f + aphi5 * hol1;
        phih = phim;
        wstar = 0.0f;
        wstar3 = 0.0f;
    }
    real ust3 = gfk_pow(ustv, 3.0f);
    wscale = gfk_pow(ust3 + phifac * karman * wstar3 * 0.5f, h1);
    wscale = sh8_min(wscale, ustv * aphi16);
    wscale = sh8_max(wscale, __fdiv_rn(ustv, aphi5));

    // ---- countergradient terms (:604-622) ----
    if (sfcflg && sflux > 0.0f) {
        real gamfac = bfac / rhox / wscale;
        hgamt = sh8_minc(__fdiv_rn(gamfac * hfxv, CP), gamcrt);
        hgamq = sh8_minc(gamfac * qfxv, gamcrq);
        real vpert = __fdiv_rn((hgamt + ep1 * SH_THX(1) * hgamq), bfac) * afac;
        thermal = thermal + sh8_maxc(vpert, 0.0f)
                * sh8_minc(za[1] / (sfcfrac * hpbl), 1.0f);
        thermalli = thermalli + sh8_maxc(vpert, 0.0f)
                  * sh8_minc(za[1] / (sfcfrac * hpbl), 1.0f);
        hgamt = sh8_maxc(hgamt, 0.0f);
        hgamq = sh8_maxc(hgamq, 0.0f);
        real brint = -15.9f * ustv * ustv / wspdv * wstar3
                   / gfk_pow(wscale, 4.0f);
        hgamu = brint * SH_UX(1);
        hgamv = brint * SH_VX(1);
    } else {
        pblflg = false;
    }

    // ---- enhance the PBL height by the thermal excess (:623-646) ----
    if (pblflg) {
        kpbl = 1;
        hpbl = zq[1];
        stable = false;
        brup = brv;
        brcr = brcr_ub;
    }
    for (int k = 2; k <= klpbl; ++k) {
        if (!stable && pblflg) {
            brdn = brup;
            real spdk2 = sh8_maxc(SH_UX(k) * SH_UX(k) + SH_VX(k) * SH_VX(k),
                                 1.0f);
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2;
            kpbl = k;
            stable = brup > brcr;
        }
    }
    // ---- stratocumulus-top search (:647-668) ----
    if (if_scu) {
        int kcld = kpbl;
        bool definebrup = false;
        real thlix1 = SH8_THLIX(1);
        for (int k = 2; k <= klpbl; ++k) {
            if (k >= kcld) {
                real spdk2 = sh8_maxc(SH_UX(k) * SH_UX(k)
                                     + SH_VX(k) * SH_VX(k), 1.0f);
                real bruptmp = (SH8_THLIX(k) - thermalli)
                             * (G * za[k] / thlix1) / spdk2;
                stable = bruptmp >= brcr;
                if (definebrup) {
                    kpbl = k;
                    brup = bruptmp;
                    definebrup = false;
                }
                if (!stable) {
                    brdn = bruptmp;
                    definebrup = true;
                    pblflg = true;
                }
            }
        }
    }
    if (pblflg) {                                // :669-691
        int k = kpbl;
        real brint;
        if (brdn >= brcr) brint = 0.0f;
        else if (brup <= brcr) brint = 1.0f;
        else brint = (brcr - brdn) / (brup - brdn);
        hpbl = za[k - 1] + brint * (za[k] - za[k - 1]);
        if (hpbl < zq[2]) kpbl = 1;
        if (kpbl <= 1) pblflg = false;
        real csfac;
        if (wstar != 0.0f) {
            real uwst = fabsf(ustv / wstar - 0.5f);
            real uwstx = -80.0f * uwst + 14.0f;
#ifdef GPUWM_WRF_EXACT
            csfac = 0.5f * (sh_tanhf(uwstx) + 3.0f);
#else
            csfac = 0.5f * (__double2float_rn(tanh((double)uwstx)) + 3.0f);
#endif
        } else {
            csfac = 1.0f;
        }
        cslen = csfac * hpbl;
    }

    // ---- stable boundary layer (:692-741) ----
    if (!sfcflg && hpbl < zq[2]) {
        brup = brv;
        stable = false;
    } else {
        stable = true;
    }
    real brcr_sbro = 0.0f;
    if (!stable && (xlandv - 1.5f) >= 0.0f) {
        real wspd10 = u10v * u10v + v10v * v10v;
        wspd10 = sqrtf(wspd10);
        real ross = wspd10 / (cori * zntv);
        brcr_sbro = sh8_minc(0.16f * gfk_pow(1.0e-7f * ross, -0.18f), 0.3f);
    }
    if (!stable) {
        if ((xlandv - 1.5f) >= 0.0f) brcr = brcr_sbro;
        else brcr = brcr_sb;
    }
    for (int k = 2; k <= klpbl; ++k) {
        if (!stable) {
            brdn = brup;
            real spdk2 = sh8_maxc(SH_UX(k) * SH_UX(k) + SH_VX(k) * SH_VX(k),
                                 1.0f);
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2;
            kpbl = k;
            stable = brup > brcr;
        }
    }
    if (!sfcflg && hpbl < zq[2]) {
        int k = kpbl;
        real brint;
        if (brdn >= brcr) brint = 0.0f;
        else if (brup <= brcr) brint = 1.0f;
        else brint = (brcr - brdn) / (brup - brdn);
        hpbl = za[k - 1] + brint * (za[k] - za[k - 1]);
        if (hpbl < zq[2]) kpbl = 1;
        if (kpbl <= 1) pblflg = false;
    }

    // ---- scale dependency (:742-751) ----
    real pu1 = sh_pu(dxy, cslen);
    real pq1 = sh_pq(dxy, cslen);
    if (pblflg) {
        hgamu = hgamu * pu1;
        hgamv = hgamv * pu1;
        hgamq = hgamq * pq1;
    }

    // ---- entrainment (:752-850) ----
    bool cloudflg = false;
    real prpbl = 0.0f, wm2 = 0.0f, bfxpbl = 0.0f, dthvx = 0.0f;
    real hfxpbl = 0.0f, qfxpbl = 0.0f, ufxpbl = 0.0f, vfxpbl = 0.0f;
    int kcloud = 0;                              // k = kpbl-1 of a cloudy top
    if (pblflg) {
        int k = kpbl - 1;
        real wm3 = wstar3 + 5.0f * ust3;
        wm2 = gfk_pow(wm3, h2);
        bfxpbl = __fdiv_rn(-0.15f * thvx[1], G) * wm3 / hpbl;
        dthvx = sh8_maxc(thvx[k + 1] - thvx[k], tmin);
        we = sh8_max(bfxpbl / dthvx, -sqrtf(wm2));
        if (SH8_QCI(k) > qci_cr && if_scu) {
            if (kpbl >= 2) {
                cloudflg = true;
                kcloud = k;
                const int k2 = min(k + 2, kte);  // declared: WRF reads kte+1
                const real pk1 = p_interface[(size_t)k * st + col];   // p2di(k+1)
                real tlix = SH8_THLIX(k) * gfk_pow(pk1 / 100000.0f, rovcp);
                real rvls = 100.0f * 6.112f
                          * gfk_exp(17.67f * (tlix - 273.16f) / (tlix - 29.65f))
                          * (ep2 / pk1);
                real temps = tlix + ((SH_QVX(k) + SH_QCX(k)) - rvls)
                           / (CP / XLV + ep2 * XLV * rvls / (RD * (tlix * tlix)));
                rvls = 100.0f * 6.112f
                     * gfk_exp(17.67f * (temps - 273.15f) / (temps - 29.65f))
                     * (ep2 / pk1);
                real rcldb = sh8_maxc((SH_QVX(k) + SH_QCX(k)) - rvls, 0.0f);
                dthvx = (SH8_THLIX(k2) + SH_THX(k2) * ep1
                         * (SH_QVX(k2) + SH_QCX(k2)))
                      - (SH8_THLIX(k) + SH_THX(k) * ep1
                         * (SH_QVX(k) + SH_QCX(k)));
                dthvx = sh8_maxc(dthvx, 0.1f);
                real evp_fac = XLV / CP * rcldb / (SH_PI2DX(k) * dthvx);
                real ent_eff = sh8_minc(0.2f * 8.0f * evp_fac, scu_ent_max);
                real radsum = 0.0f;
                for (int kk = 1; kk <= k; ++kk) {
                    real radflx = SH_G(rthraten, kk) * SH_PI2DX(kk);
                    radflx = radflx * CP / G
                           * (p_interface[(size_t)(kk - 1) * st + col]
                              - p_interface[(size_t)kk * st + col]);
                    if (radflx < 0.0f) radsum = fabsf(radflx) + radsum;
                }
                radsum = sh8_maxc(radsum, 0.0f);
                real bfx0 = sh8_maxc(sflux, 0.0f);
                wm3 = (govrth * bfx0 * hpbl) + 5.0f * ust3;
                wm2 = gfk_pow(wm3, h2);
                bfxpbl = __fdiv_rn(-0.15f * thvx[1], G) * wm3 / hpbl;
                dthvx = sh8_maxc(thvx[k + 1] - thvx[k], tmin);
                we = sh8_max(bfxpbl / dthvx, -sqrtf(wm2));
                real rhox2k = SH8_RHOX2(k);
                bfx0 = sh8_maxc(__fdiv_rn(radsum / rhox2k, CP), 0.0f);
                wm3 = (G / thvx[k] * bfx0 * hpbl);
                wm2 = wm2 + gfk_pow(wm3, h2);
                bfxpbl = -(ent_eff * bfx0);
                dthvx = sh8_maxc(thvx[k + 1] - thvx[k], 0.1f);
                we = we + sh8_max(bfxpbl / dthvx, -sqrtf(gfk_pow(wm3, h2)));
                wstar3_2 = (G / thvx[k] * bfx0 * hpbl);
                wscale = gfk_pow(ust3 + phifac * karman * (wstar3 + wstar3_2)
                                 * 0.5f, h1);
                wscale = sh8_min(wscale, ustv * aphi16);
                wscale = sh8_max(wscale, __fdiv_rn(ustv, aphi5));
                real gamfac = bfac / rhox / wscale;
                hgamt = sh8_minc(__fdiv_rn(gamfac * hfxv, CP), gamcrt);
                hgamq = sh8_minc(gamfac * qfxv, gamcrq);
                gamfac = bfac / rhox2k / wscale;
                real hgamt2 = sh8_minc(__fdiv_rn(gamfac * radsum, CP), gamcrt);
                hgamt = sh8_maxc(hgamt, 0.0f) + sh8_maxc(hgamt2, 0.0f);
                real brint = -15.9f * ustv * ustv / wspdv * (wstar3 + wstar3_2)
                           / gfk_pow(wscale, 4.0f);
                hgamu = brint * SH_UX(1);
                hgamv = brint * SH_VX(1);
            }
        }
        prpbl = 1.0f;
        real dthx = sh8_maxc(SH_THX(k + 1) - SH_THX(k), tmin);
        real dqx = sh8_minc(SH_QVX(k + 1) - SH_QVX(k), 0.0f);
        hfxpbl = we * dthx;
        pq1 = sh_pq(dxy, cslen);
        qfxpbl = we * dqx * pq1;
        pu1 = sh_pu(dxy, cslen);
        real dux = SH_UX(k + 1) - SH_UX(k);
        real dvx = SH_VX(k + 1) - SH_VX(k);
        if (dux > tmin)
            ufxpbl = sh8_max(prpbl * we * dux * pu1, -(ustv * ustv));
        else if (dux < -tmin)
            ufxpbl = sh8_min(prpbl * we * dux * pu1, ustv * ustv);
        else
            ufxpbl = 0.0f;
        if (dvx > tmin)
            vfxpbl = sh8_max(prpbl * we * dvx * pu1, -(ustv * ustv));
        else if (dvx < -tmin)
            vfxpbl = sh8_min(prpbl * we * dvx * pu1, ustv * ustv);
        else
            vfxpbl = 0.0f;
        if (if_nonlocal) {
            delta_sh = (mltop - entfracn1) * hpbl;
            deltaoh = delta_sh / hpbl;
            entfrac = cent * fent * sh8_minc(wm3 / wstar3, 2.0f);
        }
        real delb = govrth * d3 * hpbl;
        delta = sh8_minc(d1 * hpbl + d2 * wm2 / delb, 100.0f);
    }

    for (int k = 1; k <= klpbl; ++k) {           // :851-863
        if (if_nonlocal && pblflg) {
            real t = (zq[k + 1] - hpbl) / delta_sh;
            entfacmf[k] = sqrtf(t * t);
        }
        if (pblflg && k >= kpbl) {
            real t = (zq[k + 1] - hpbl) / delta;
            entfac[k] = t * t;
        } else {
            entfac[k] = 1.0e30f;
        }
    }

    // ---- diffusion coefficients below the PBL top (:864-905) ----
    for (int k = 1; k <= klpbl; ++k) {
        if (k < kpbl) {
            real zfac = sh8_minc(sh8_maxc(
                1.0f - (zq[k + 1] - zl1) / (hpbl - zl1), zfmin), 1.0f);
            real zfac2 = 1.0f - zfac;
            zfacent[k] = gfk_pow(1.0f - zfac, 3.0f);
            real wscalek = gfk_pow(
                ust3 + phifac * karman * wstar3 * (1.0f - zfac), h1);
            real wscalek2 = gfk_pow(phifac * karman * wstar3_2 * zfac, h1);
            real prfac, prfac2, prnumfac;
            if (sfcflg) {
                prfac = conpr;
                prfac2 = 15.9f * (wstar3 + wstar3_2) / ust3
                       / (1.0f + 4.0f * karman * (wstar3 + wstar3_2) / ust3);
                real m = sh8_maxc(zq[k + 1] - sfcfrac * hpbl, 0.0f);
                prnumfac = -3.0f * (m * m) / (hpbl * hpbl);
            } else {
                prfac = 0.0f;
                prfac2 = 0.0f;
                prnumfac = 0.0f;
                real phim8z = 1.0f + aphi5 * zol1 * zq[k + 1] / zl1;
                wscalek = ustv / phim8z;
                wscalek = sh8_maxc(wscalek, 0.001f);
            }
            real prnum0 = phih / phim + prfac;
            prnum0 = sh8_maxc(sh8_minc(prnum0, prmax), prmin);
            xkzm[k] = wscalek * karman * zq[k + 1] * (zfac * zfac)
                    + wscalek2 * karman * (hpbl - zq[k + 1]) * (zfac2 * zfac2);
            if (k == kpbl - 1 && cloudflg && we < 0.0f)
                xkzm[k] = 0.0f;
            real prnum = 1.0f + (prnum0 - 1.0f) * gfk_exp(prnumfac);
            xkzq[k] = xkzm[k] / prnum * gfk_pow(zfac, 0.0f);   // **(pfac_q-pfac)
            prnum0 = prnum0 / (1.0f + prfac2 * karman * sfcfrac);
            prnum = 1.0f + (prnum0 - 1.0f) * gfk_exp(prnumfac);
            xkzh[k] = xkzm[k] / prnum;
            xkzm[k] = xkzm[k] + xkzminm;
            xkzh[k] = xkzh[k] + xkzminh;
            xkzq[k] = xkzq[k] + xkzminh;
            xkzm[k] = sh8_minc(xkzm[k], xkzmax);
            xkzh[k] = sh8_minc(xkzh[k], xkzmax);
            xkzq[k] = sh8_minc(xkzq[k], xkzmax);
        }
    }

    // ---- free atmosphere (:906-952) ----
    for (int k = 1; k <= kte - 1; ++k) {
        if (k >= kpbl) {
            real duk = SH_UX(k + 1) - SH_UX(k);
            real dvk = SH_VX(k + 1) - SH_VX(k);
            real ss = (duk * duk + dvk * dvk) / (dza[k + 1] * dza[k + 1])
                    + 1.0e-9f;
            real govrthv = G / (0.5f * (thvx[k + 1] + thvx[k]));
            real ri = govrthv * (thvx[k + 1] - thvx[k]) / (ss * dza[k + 1]);
            if (SH8_QCI(k) > qci_cr && SH8_QCI(k + 1) > qci_cr) {
                real qmean = 0.5f * (SH_QVX(k) + SH_QVX(k + 1));
                real tmean = 0.5f * (SH_TX(k) + SH_TX(k + 1));
                real alpha = __fdiv_rn(XLV * qmean, RD) / tmean;
                real chi = __fdiv_rn(__fdiv_rn(XLV * XLV * qmean, CP), RV)
                         / tmean / tmean;
                ri = (1.0f + alpha) * (
                    ri - __fdiv_rn(G * G / ss / tmean, CP)
                    * ((chi - alpha) / (1.0f + chi)));
            }
            real zk = karman * zq[k + 1];
            real rlamdz = sh8_minc(sh8_maxc(0.1f * dza[k + 1], rlam), 300.0f);
            rlamdz = sh8_min(dza[k + 1], rlamdz);
            real tt = zk * rlamdz / (rlamdz + zk);
            real rl2 = tt * tt;
            real dk = rl2 * sqrtf(ss);
            if (ri < 0.0f) {
                ri = sh8_maxc(ri, rimin);
                real sri = sqrtf(-ri);
                xkzm[k] = dk * (1.0f + 8.0f * (-ri) / (1.0f + 1.746f * sri));
                xkzh[k] = dk * (1.0f + 8.0f * (-ri) / (1.0f + 1.286f * sri));
            } else {
                tt = 1.0f + 5.0f * ri;
                xkzh[k] = dk / (tt * tt);
                real prnum = 1.0f + 2.1f * ri;
                prnum = sh8_minc(prnum, prmax);
                xkzm[k] = xkzh[k] * prnum;
            }
            xkzm[k] = xkzm[k] + xkzminm;
            xkzh[k] = xkzh[k] + xkzminh;
            xkzm[k] = sh8_minc(xkzm[k], xkzmax);
            xkzh[k] = sh8_minc(xkzh[k], xkzmax);
            xkzml[k] = xkzm[k];
            xkzhl[k] = xkzh[k];
        }
    }

    // ---- LES-type nonlocal heat flux profile (:953-985) ----
    if (if_nonlocal) {
        real mlfrac = mltop - deltaoh;
        real zfacmf1 = sh8_minc(sh8_maxc(zq[2] / hpbl, zfmin), 1.0f);
        real sfcfracn = sh8_maxc(zfacmf1, sfcfracn1);
        real sflux0 = (a11 + a12 * sfcfracn) * sflux;
        real snlflux0 = nlfrac * sflux0;
        real amf1 = snlflux0 / sfcfracn;
        real amf2 = 0.0f, bmf2 = 0.0f, amf3;    // declared: see header
        if (pblflg && sflux > 0.0f) {
            amf2 = -(snlflux0 / (mlfrac - sfcfracn));
            bmf2 = -(mlfrac * amf2);
            amf3 = snlflux0 * entfrac / deltaoh;
        } else {
            amf3 = 0.0f;
        }
        real bmf3 = -(amf3 * mlfrac);
        hfxpbl_sh = amf3 + bmf3;
        for (int k = 1; k <= klpbl; ++k) {
            real zfacmf = sh8_maxc(zq[k + 1] / hpbl, zfmin);
            if (pblflg && k < kpbl) {
                if (zfacmf <= sfcfracn)
                    mf[k] = amf1 * zfacmf;
                else if (zfacmf <= mlfrac)
                    mf[k] = amf2 * zfacmf + bmf2;
                mf[k] = mf[k] + hfxpbl_sh * gfk_exp(-entfacmf[k]);
            }
        }
    }

    // ---- heat (:986-1060) ----
    ad[1] = 1.0f;
    r1[1] = SH_THX(1) - 300.0f
          + __fdiv_rn((1.0f - bepswitch) * hfxv, cont) / delp[1] * dt2;
    for (int k = 1; k <= kte - 1; ++k) {
        real dtodsd = bep1 * dt2 / delp[k];
        real dtodsu = bep1 * dt2 / delp[k + 1];
        real dsig = SH_P2DX(k) - SH_P2DX(k + 1);
        real rdz = 1.0f / dza[k + 1];
        real tem1 = dsig * xkzh[k] * rdz;
        if (pblflg && k < kpbl) {
            real pth1 = sh_pthnl(dxy, cslen);
            real hvalue;
            if (if_nonlocal)
                hvalue = mf[k] / xkzh[k];
            else
                hvalue = hgamt / hpbl + hfxpbl * zfacent[k] / xkzh[k];
            real dsdzt = tem1 * (-(hvalue * pth1));
            r1[k] = r1[k] + dtodsd * dsdzt;
            r1[k + 1] = SH_THX(k + 1) - 300.0f - dtodsu * dsdzt;
        } else if (pblflg && k >= kpbl && entfac[k] < 4.6f) {
            xkzh[k] = -(we * dza[kpbl] * gfk_exp(-entfac[k]));
            xkzh[k] = sqrtf(xkzh[k] * xkzhl[k]);
            xkzh[k] = sh8_maxc(xkzh[k], xkzminh);
            xkzh[k] = sh8_minc(xkzh[k], xkzmax);
            r1[k + 1] = SH_THX(k + 1) - 300.0f;
        } else {
            r1[k + 1] = SH_THX(k + 1) - 300.0f;
        }
        tem1 = dsig * xkzh[k] * rdz;
        real dsdz2 = tem1 * rdz;
        au[k] = -(dtodsd * dsdz2 / bep1);
        al[k] = -(dtodsu * dsdz2 / bep1);
        real zfacdx = 0.2f * hpbl / zq[k + 1];
        real delxy_l = dxy * sh8_maxc(zfacdx, 1.0f);
        real pthl1 = sh_pthl(delxy_l, hpbl);
        if (pblflg && k < kpbl) {
            au[k] = au[k] * pthl1;
            al[k] = al[k] * pthl1;
        }
        ad[k] = ad[k] - au[k];
        ad[k + 1] = 1.0f - al[k];
        SH_G(exch_h, k + 1) = xkzh[k];
    }
    // :1037-1042 ad = ad - a_t*dt2, f1 = f1 + b_t*dt2 with BEP off: the
    // subtraction of +0 is the identity on every word and is not spelled;
    // the addition maps a -0 right-hand side to +0 and is.
    for (int k = 1; k <= kte; ++k)
        r1[k] = r1[k] + bep0 * dt2;
    sh_tridi_factor(al, ad, au, fkk, lau, kte);  // tridin_ysu :1048
    sh_tridi_solve(al, fkk, lau, r1, kte);
    // ttnp, in temperature units until the wrapper divides it by pi after
    // the dissipation term (module_bl_shinhong.F:461); held in the TKE
    // chain's q2xk slot, which is free until then.
    ShCol ttnp = SHWS_AT(wsb, 39, wskp);
    for (int k = kte; k >= 1; --k) {             // :1049-1060
        real ttend = (r1[k] - SH_THX(k) + 300.0f) * rdt * SH_PI2DX(k);
        ttnp[k] = 0.0f + ttend;
        if (k == kte)
            tflux_e[k] = ttend * SH_DZ8W(k);
        else
            tflux_e[k] = tflux_e[k + 1] + ttend * SH_DZ8W(k);
    }

    // ---- moisture (:1061-1150) ----
    for (int k = 1; k <= kte - 1; ++k)
        if (k >= kpbl) xkzq[k] = xkzh[k];
    for (int k = 0; k < wskp; ++k) {
        al[k] = ad[k] = au[k] = 0.0f;
        r1[k] = r2[k] = r3[k] = 0.0f;
    }
    ad[1] = 1.0f;
    r1[1] = SH_QVX(1) + (1.0f - bepswitch) * qfxv * G / delp[1] * dt2;
    for (int k = 1; k <= kte - 1; ++k) {
        real dtodsd = bep1 * dt2 / delp[k];
        real dtodsu = bep1 * dt2 / delp[k + 1];
        real dsig = SH_P2DX(k) - SH_P2DX(k + 1);
        real rdz = 1.0f / dza[k + 1];
        real tem1 = dsig * xkzq[k] * rdz;
        if (pblflg && k < kpbl) {
            real dsdzq = tem1 * (-(qfxpbl * zfacent[k] / xkzq[k]));
            r1[k] = r1[k] + dtodsd * dsdzq;
            r1[k + 1] = SH_QVX(k + 1) - dtodsu * dsdzq;
        } else if (pblflg && k >= kpbl && entfac[k] < 4.6f) {
            xkzq[k] = -(we * dza[kpbl] * gfk_exp(-entfac[k]));
            xkzq[k] = sqrtf(xkzq[k] * xkzhl[k]);
            xkzq[k] = sh8_maxc(xkzq[k], xkzminh);
            xkzq[k] = sh8_minc(xkzq[k], xkzmax);
            r1[k + 1] = SH_QVX(k + 1);
        } else {
            r1[k + 1] = SH_QVX(k + 1);
        }
        tem1 = dsig * xkzq[k] * rdz;
        real dsdz2 = tem1 * rdz;
        au[k] = -(dtodsd * dsdz2 / bep1);
        al[k] = -(dtodsu * dsdz2 / bep1);
        real zfacdx = 0.2f * hpbl / zq[k + 1];
        real delxy_l = dxy * sh8_maxc(zfacdx, 1.0f);
        real pq1_l = sh_pq(delxy_l, hpbl);
        if (pblflg && k < kpbl) {
            au[k] = au[k] * pq1_l;
            al[k] = al[k] * pq1_l;
        }
        ad[k] = ad[k] - au[k];
        ad[k + 1] = 1.0f - al[k];
    }
    // adv = ad - a_q*dt2 for vapour (:1112); cloud water and ice reuse ad
    // (:1166, :1182), the same words with BEP off, so one factorization
    // serves all three right-hand sides (tridin_ysu recomputes it per call
    // with the identical operation sequence).
    for (int k = 1; k <= kte; ++k) {
        r1[k] = r1[k] + bep0 * dt2;
        r2[k] = SH_QCX(k);
        r3[k] = SH_QIX(k);
    }
    sh_tridi_factor(al, ad, au, fkk, lau, kte);
    sh_tridi_solve(al, fkk, lau, r1, kte);
    sh_tridi_solve(al, fkk, lau, r2, kte);
    sh_tridi_solve(al, fkk, lau, r3, kte);
    for (int k = kte; k >= 1; --k) {             // :1124-1137
        real qtend = (r1[k] - SH_QVX(k)) * rdt;
        SH_G(dqv, k) = qtend + 0.0f;
        if (k == kte)
            qflux_e[k] = qtend * SH_DZ8W(k);
        else
            qflux_e[k] = qflux_e[k + 1] + qtend * SH_DZ8W(k);
        tvflux_e[k] = tflux_e[k] + qflux_e[k] * ep1 * SH_THX(k);
    }
    for (int k = 1; k <= kte; ++k) {             // :1138-1150
        if (pblflg && k < kpbl) {
            real hgame_c = c_1 * 0.2f * 2.5f * (G / thvx[k]) * wstar
                         / sh8_maxc(0.25f * (q2x[k + 1] + q2x[k]), 0.01f);
            hgame_c = sh8_minc(hgame_c, gamcre);
            if (k == kte)
                hgame2d[k] = hgame_c * 0.5f * tvflux_e[k] * hpbl;
            else
                hgame2d[k] = hgame_c * 0.5f * (tvflux_e[k] + tvflux_e[k + 1])
                           * hpbl;
            hgame2d[k] = sh8_maxc(hgame2d[k], 0.0f);
        }
    }
    for (int k = kte; k >= 1; --k)               // :1151-1166
        SH_G(dqc, k) = (r2[k] - SH_QCX(k)) * rdt + 0.0f;
    for (int k = kte; k >= 1; --k)               // :1167-1182
        SH_G(dqi, k) = (r3[k] - SH_QIX(k)) * rdt + 0.0f;

    // ---- momentum (:1199-1295) ----
    for (int k = 0; k < wskp; ++k) {
        al[k] = ad[k] = au[k] = 0.0f;
        r1[k] = r2[k] = r3[k] = 0.0f;
    }
    {
        real s = wspd1 / wspdv;
        real tau0 = ustv * ustv / wspd1 * rhox * G / delp[1] * dt2 * (s * s);
        real hvalue = sh8_minc(sh8_maxc(1.0f - brv, 0.0f), 1.0f);
        ad[1] = 1.0f + tau0 * hvalue + ctopov * tau0 * (1.0f - hvalue);
        ad[1] = ad[1] - bepswitch * bep0
              * (tau0 * hvalue + ctopov * tau0 * (1.0f - hvalue));
        r1[1] = SH_UX(1) + uoxl * tau0;
        r2[1] = SH_VX(1) + voxl * tau0;
    }
    for (int k = 1; k <= kte - 1; ++k) {
        real dtodsd = bep1 * dt2 / delp[k];
        real dtodsu = bep1 * dt2 / delp[k + 1];
        real dsig = SH_P2DX(k) - SH_P2DX(k + 1);
        real rdz = 1.0f / dza[k + 1];
        real tem1 = dsig * xkzm[k] * rdz;
        if (pblflg && k < kpbl) {
            real dsdzu = tem1 * (-(hgamu / hpbl) - ufxpbl * zfacent[k] / xkzm[k]);
            real dsdzv = tem1 * (-(hgamv / hpbl) - vfxpbl * zfacent[k] / xkzm[k]);
            r1[k] = r1[k] + dtodsd * dsdzu;
            r1[k + 1] = SH_UX(k + 1) - dtodsu * dsdzu;
            r2[k] = r2[k] + dtodsd * dsdzv;
            r2[k + 1] = SH_VX(k + 1) - dtodsu * dsdzv;
        } else if (pblflg && k >= kpbl && entfac[k] < 4.6f) {
            xkzm[k] = prpbl * xkzh[k];
            xkzm[k] = sqrtf(xkzm[k] * xkzml[k]);
            xkzm[k] = sh8_maxc(xkzm[k], xkzminm);
            xkzm[k] = sh8_minc(xkzm[k], xkzmax);
            r1[k + 1] = SH_UX(k + 1);
            r2[k + 1] = SH_VX(k + 1);
        } else {
            r1[k + 1] = SH_UX(k + 1);
            r2[k + 1] = SH_VX(k + 1);
        }
        tem1 = dsig * xkzm[k] * rdz;
        real dsdz2 = tem1 * rdz;
        au[k] = -(dtodsd * dsdz2 / bep1);
        al[k] = -(dtodsu * dsdz2 / bep1);
        real zfacdx = 0.2f * hpbl / zq[k + 1];
        real delxy_l = dxy * sh8_maxc(zfacdx, 1.0f);
        real pu1_l = sh_pu(delxy_l, hpbl);
        if (pblflg && k < kpbl) {
            au[k] = au[k] * pu1_l;
            al[k] = al[k] * pu1_l;
        }
        ad[k] = ad[k] - au[k];
        ad[k + 1] = 1.0f - al[k];
        SH_G(exch_m, k + 1) = xkzm[k];
    }
    // ad1 = ad (:1272-1276), then each less its BEP term; equal words with
    // BEP off, so tridi2n's two factorizations are this one.
    for (int k = 1; k <= kte; ++k) {
        r1[k] = r1[k] + bep0 * dt2;
        r2[k] = r2[k] + bep0 * dt2;
    }
    sh_tridi_factor(al, ad, au, fkk, lau, kte);
    sh_tridi_solve(al, fkk, lau, r1, kte);
    sh_tridi_solve(al, fkk, lau, r2, kte);
    // r3 holds dishxi(kts..kte+1), zeroed above (:503).
    real uflux = 0.0f, vflux = 0.0f;
    for (int k = kte; k >= 1; --k) {             // :1296-1318
        real utend = (r1[k] - SH_UX(k)) * rdt;
        real vtend = (r2[k] - SH_VX(k)) * rdt;
        SH_G(du, k) = 0.0f + utend;
        SH_G(dv, k) = 0.0f + vtend;
        if (if_dissip) {
            if (k == kte) {
                uflux = utend * conwrc * delp[k];
                vflux = vtend * conwrc * delp[k];
            } else {
                uflux = uflux + utend * delp[k] * conwrc;
                vflux = vflux + vtend * delp[k] * conwrc;
            }
            if (k != 1) {
                real dp = SH_P2DX(k) - SH_P2DX(k - 1);
                real d = G * (uflux * (r1[k] - r1[k - 1]) / dp
                              + vflux * (r2[k] - r2[k - 1]) / dp);
                r3[k] = sh8_maxc(d, 0.0f);
            }
        }
    }
    for (int k = kte; k >= 1; --k) {             // :1319-1327, wrapper :461
        real t = ttnp[k];
        if (if_dissip) {
            real dishx = (r3[k] + r3[k + 1]) * 0.5f / (CP - RD);
            t = t + dishx;
        }
        SH_G(dtheta, k) = t / SH_PI2DX(k);
    }

    // ---- 10 m wind (:1331-1336) ----
    u10_out[col] = __fadd_rn(__fmul_rn(ctopo2v, u10v),
                             __fmul_rn(__fsub_rn(1.0f, ctopo2v), SH_UX(1)));
    v10_out[col] = __fadd_rn(__fmul_rn(ctopo2v, v10v),
                             __fmul_rn(__fsub_rn(1.0f, ctopo2v), SH_VX(1)));

    // ---- SGS TKE (:1337-1468), always on ----
    {
        ShCol q2xk = SHWS_AT(wsb, 39, wskp);
        ShCol ptke1 = SHWS_AT(wsb, 40, wskp);
        ShCol akmk = SHWS_AT(wsb, 41, wskp);
        ShCol akhk = SHWS_AT(wsb, 42, wskp);
        ShCol mfk = SHWS_AT(wsb, 43, wskp);
        ShCol ufxpblk = SHWS_AT(wsb, 44, wskp);
        ShCol vfxpblk = SHWS_AT(wsb, 45, wskp);
        ShCol qfxpblk = SHWS_AT(wsb, 46, wskp);
        ShCol s2k = SHWS_AT(wsb, 47, wskp);
        ShCol rigk = SHWS_AT(wsb, 48, wskp);
        ShCol elk = SHWS_AT(wsb, 49, wskp);
        for (int k = 0; k < wskp; ++k) {
            q2xk[k] = akmk[k] = akhk[k] = mfk[k] = 0.0f;
            ufxpblk[k] = vfxpblk[k] = qfxpblk[k] = 0.0f;
            s2k[k] = rigk[k] = elk[k] = 0.0f;
            ptke1[k] = 1.0f;
        }
        for (int k = 1; k <= kte; ++k)
            q2xk[k] = sh8_maxc(q2x[k], 0.0001f);
        for (int k = 1; k <= kte - 1; ++k) {
            if (pblflg && k <= kpbl) {
                real zfacdx = 0.2f * hpbl / za[k];
                real delxy_l = dxy * sh8_maxc(zfacdx, 1.0f);
                ptke1[k + 1] = sh_ptke(delxy_l, hpbl);
            }
        }
        for (int k = 2; k <= kte; ++k) {
            akmk[k] = xkzm[k - 1];
            akhk[k] = xkzh[k - 1];
            mfk[k] = mf[k - 1] / xkzh[k - 1];
            ufxpblk[k] = ufxpbl * zfacent[k - 1] / xkzm[k - 1];
            vfxpblk[k] = vfxpbl * zfacent[k - 1] / xkzm[k - 1];
            qfxpblk[k] = qfxpbl * zfacent[k - 1] / xkzq[k - 1];
        }
        if (pblflg) {                            // :1418-1422, bounded
            int k = min(kpbl - 1, klpbl - 2);
            real dex = 0.25f * (q2xk[k + 2] - q2xk[k]);
            efxpbl = we * dex;
        }
        sh_mixlen(u, v, theta, exner, qv, qc, st, col,
                  q2xk, zq, ustv, corfv, epshol, hpbl, kpbl, pblflg,
                  hgamu, hgamv, hgamq, mfk, ufxpblk, vfxpblk, qfxpblk, kte,
                  s2k, rigk, elk, wsb, wskp);
        sh8_prodq2(dt, ustv, u, v, theta, exner, st, col, thvx,
                   q2xk, elk, zq, akmk, akhk,
                   hgamu, hgamv, hgamq, dxy, hpbl, pblflg, kpbl,
                   mfk, ufxpblk, vfxpblk, qfxpblk, kte);
        sh_vdifq(dt, q2xk, zq, akhk, ptke1, hgame2d, hpbl, pblflg, kpbl,
                 efxpbl, kte, wsb, wskp);
        for (int k = 1; k <= kte; ++k) {
            // The referee's code returns the floor for a NaN q2xk here
            // (measured: the ust = 0 probes, whose prfac2 0/0 reaches q2
            // through hgame), unlike the propagating constant-bound sites.
            q2x[k] = sh8_max(q2xk[k], epsq2l);
            SH_G(tke_out, k) = 0.5f * q2x[k];
            if (k != 1)
                SH_G(el, k) = elk[k];
        }
    }

    hpbl_out[col] = hpbl;
    kpbl_out[col] = kpbl;
    wstar_out[col] = wstar;
    delta_out[col] = delta;
    (void)kcloud;
}
#undef SH8_THLIX
#undef SH8_QCI
#undef SH8_RHOX2

#define SH8_PARAMS const real *u, const real *v, const real *theta, \
    const real *qv, const real *qc, const real *qi, const real *p, \
    const real *p_interface, const real *exner, const real *dz, \
    const real *tke, const real *rthraten, const real *psfc, const real *znt, \
    const real *ust, const real *hfx, const real *qfx, const real *wspd, \
    const real *br, const real *psim, const real *psih, const real *xland, \
    const real *u10, const real *v10, const real *corf, const real *uoce, \
    const real *voce, real *du, real *dv, real *dtheta, real *dqv, \
    real *dqc, real *dqi, real *exch_h, real *exch_m, real *tke_out, \
    real *el, real *hpbl_out, int *kpbl_out, real *wstar_out, \
    real *delta_out, real *u10_out, real *v10_out, real dt, real dx, \
    int flags, int nz, int ny, int nx, real *ws, int wskp, int col0
#define SH8_ARGS u, v, theta, qv, qc, qi, p, p_interface, exner, dz, tke, \
    rthraten, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland, u10, \
    v10, corf, uoce, voce, du, dv, dtheta, dqv, dqc, dqi, exch_h, exch_m, \
    tke_out, el, hpbl_out, kpbl_out, wstar_out, delta_out, u10_out, \
    v10_out, dt, dx, flags, nz, ny, nx, ws, wskp, col0

// topo_wind = 0: ctopo = ctopo2 = 1, WRF's arrays when the option is off.
extern "C" __global__
void shinhong480_column(SH8_PARAMS) {
    shinhong480_column_body<false>(SH8_ARGS, nullptr, nullptr);
}

// topo_wind = 1 or 2.
extern "C" __global__
void shinhong480_column_topo(SH8_PARAMS, const real *ctopo,
                             const real *ctopo2) {
    shinhong480_column_body<true>(SH8_ARGS, ctopo, ctopo2);
}

// Non-finite census of the launcher's float outputs, one bit per output in
// gpuwm/core/shinhong.py _SH480_VALIDATED order.  Read-only; it sits in this
// unit because no other module owns this output layout.
extern "C" __global__
void shinhong480_validate_outputs(
        const real *du, const real *dv, const real *dtheta,
        const real *dqv, const real *dqc, const real *dqi,
        const real *exch_h, const real *exch_m, const real *tke,
        const real *el, const real *hpbl, const real *wstar,
        const real *delta, const real *u10, const real *v10,
        unsigned int *status, long long count_3d, long long count_2d) {
    long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int invalid = 0;
    if (index < count_3d) {
        if (!isfinite(du[index])) invalid |= 1u << 0;
        if (!isfinite(dv[index])) invalid |= 1u << 1;
        if (!isfinite(dtheta[index])) invalid |= 1u << 2;
        if (!isfinite(dqv[index])) invalid |= 1u << 3;
        if (!isfinite(dqc[index])) invalid |= 1u << 4;
        if (!isfinite(dqi[index])) invalid |= 1u << 5;
        if (!isfinite(exch_h[index])) invalid |= 1u << 6;
        if (!isfinite(exch_m[index])) invalid |= 1u << 7;
        if (!isfinite(tke[index])) invalid |= 1u << 8;
        if (!isfinite(el[index])) invalid |= 1u << 9;
    }
    if (index < count_2d) {
        if (!isfinite(hpbl[index])) invalid |= 1u << 10;
        if (!isfinite(wstar[index])) invalid |= 1u << 11;
        if (!isfinite(delta[index])) invalid |= 1u << 12;
        if (!isfinite(u10[index])) invalid |= 1u << 13;
        if (!isfinite(v10[index])) invalid |= 1u << 14;
    }
    if (invalid != 0) atomicOr(status, invalid);
}
