// gpuwm/core/kernels/ssib.cu -- SSiB land surface (sf_surface_physics = 8).
//
// A transcription of WRF v4.6.1 phys/module_sf_ssib.F (d66e442) and of the
// CASE (SSIBSCHEME) arm of phys/module_surface_driver.F (:4156-4345), one
// thread per column, in WRF's operation order.  Routine names follow the
// Fortran (ss_<name>); every array keeps Fortran's 1-based index so a line
// here reads against its line there.
//
// Arithmetic.  float32 throughout, except where the Fortran mixes in a
// DOUBLE PRECISION literal: gfortran then evaluates that operation in
// binary64 and rounds on assignment, and so does this file (COMPACT,
// GETMET, SDSOL, SNOW_1ST, TPROPTY).  EXP/ALOG/REAL**REAL call WOOF's own
// float32 routines (gfk_exp/gfk_log/gfk_pow); the binary64 EXP and ** call
// WOOF's own binary64 routines.  SQRT is IEEE sqrt.  Integer powers are
// multiply chains, as gfortran expands them.  Strict arithmetic
// (GPUWM_WRF_EXACT=1) compiles this file with --fmad=false; the default arm
// permits contraction. The default arm also repairs the declared sea-ice units.
//
// Locals.  Every local starts at zero.  That is gfortran's
// -finit-local-zero, which the column oracle is built with: SSIB reads two
// automatic locals before writing them (UPDAT1's SNOWW(2) under the
// multi-layer pack, TEMRS2's ICASE family when the top layer's ice
// fraction exceeds 1), and WRF itself has no defined value there.
//
// State layout.  Field-major: st[k*ncol + col].  The enum below is the
// order of gpuwm.core.ssib.STATE (tests/test_ssib_layout.py holds them
// equal).

// ---------------------------------------------------------------- layout
enum { C_XLAND = 0, C_IVGTYP, C_XLAT, C_MMINLU, C_RA_SW, C_DTBL, NC_SSIB };
enum { F_RAINNCV = 0, F_RAINCV, F_GLW, F_COSZEN, F_SWDOWN,
       F_SWVISDIR, F_SWVISDIF, F_SWNIRDIR, F_SWNIRDIF,
       F_DZ8W1, F_U1, F_V1, F_QV1, F_T1, F_P1, F_PSFC,
       F_HFX_SEA, F_QFX_SEA, F_LH_SEA, F_TSK_SEA,
       F_XICE, F_XICE_SAVE, F_JULIAN, F_T2_SFCLAY, NF_SSIB };
enum { S_SMOIS1 = 0, S_SMOIS2, S_SMOIS3, S_TSLB1, S_TSLB2, S_TSLB3,
       S_SNOW, S_CANWAT, S_SFCRUNOFF, S_SFCEVP,
       S_TSK, S_T2, S_Q2, S_ALBEDO, S_UST, S_HFX, S_LH, S_GRDFLX, S_QFX,
       S_ZNT, S_GZ1OZ0, S_BR, S_U10, S_V10, S_SNOWH, S_TSK_SAVE,
       S_SSIB_BR, S_SSIB_FM, S_SSIB_FH, S_SSIB_CM, S_SSIBXDD,
       S_SSIB_LHF, S_SSIB_SHF, S_SSIB_GHF, S_SSIB_EGS, S_SSIB_ECI,
       S_SSIB_ECT, S_SSIB_EGI, S_SSIB_EGT, S_SSIB_SDN, S_SSIB_SUP,
       S_SSIB_LDN, S_SSIB_LUP, S_SSIB_WAT, S_SSIB_SHC, S_SSIB_SHG,
       S_SSIB_LAI, S_SSIB_VCF, S_SSIB_Z00, S_SSIB_VEG,
       S_ALSWVISDIR, S_ALSWVISDIF, S_ALSWNIRDIR, S_ALSWNIRDIF,
       S_SWVISDIR, S_SWVISDIF, S_SWNIRDIR, S_SWNIRDIF,
       S_ISNOW, S_SWE, S_SNOWDEN, S_SNOWDEPTH, S_TKAIR,
       S_LAYER1,               // 12 per layer x 4: dzo wo tssn tssno bwo bto
       NS_SSIB = S_LAYER1 + 48 };   // cto fio flo bio blo ho

// Error bits a column can raise (the host refuses the step and names them).
#define SSIB_ERR_VEGMAP   1   // itype outside 1..13 (module_sf_ssib.F:1022-1026)
#define SSIB_ERR_MAP      2   // MMINLU not a map SSIB accepts (:1009-1019)
#define SSIB_ERR_UDUM0    4   // TEMRS2 'udum0 is WRONG' STOP (:5218-5221)
#define SSIB_ERR_ICASE    8   // TEMRS2 top-layer fi > 1: ICASE never set

// ---------------------------------------------------------------- tables
// The 12-type SSiB vegetation DATA tables (module_sf_ssib.F:41-879), plus
// SSIB's local category tables, flattened in Fortran (column-major) order.
// Filled from gpuwm/data/ssib/ssib_veg_tables.json, whose SHA-256 the host
// pins (gpuwm.core.ssib.SSIB_TABLE_SHA256).
#define T_TRAN0    0      // (13,2,3,2)
#define T_REF0     156    // (13,2,3,2)
#define T_GREEN0   312    // (13,12,2)
#define T_VCOVER0  624    // (13,12,2)
#define T_ZLT0     936    // (13,12,2)
#define T_RSTPAR0  1248   // (13,2,3)
#define T_Z000     1326   // (13,12)
#define T_D0       1482
#define T_Z20      1638
#define T_Z10      1794
#define T_RDC0     1950
#define T_RBC0     2106
#define T_DEPTH0   2262   // (13,3)
#define T_SOREF0   2301   // (13,3)
#define T_CHIL0    2340   // (13,2)
#define T_TOPT0    2366
#define T_TL0      2392
#define T_TU0      2418
#define T_DEFAC0   2444
#define T_PH10     2470
#define T_PH20     2496
#define T_ROOTD0   2522
#define T_BEE0     2548   // (13)
#define T_PHSAT0   2561
#define T_POROS0   2574
#define T_SATCO0   2587
#define T_SLOPE0   2600
#define T_IVUSGS   2613   // (24) SSIB :1001-1003
#define T_IVMODIS  2637   // (20) SSIB :1005-1007
#define T_TDDEPTH  2657   // (13) SSIB :996-997 (= SSIB_SEAICE :1506-1507)
#define NTAB_SSIB  2670
__constant__ float ssib_tab[NTAB_SSIB];

#define TAB4(off, it, a, b, c, na, nb) \
    ssib_tab[(off) + ((it)-1) + 13*(((a)-1) + (na)*(((b)-1) + (nb)*((c)-1)))]
#define TAB3(off, it, a, b, na) \
    ssib_tab[(off) + ((it)-1) + 13*(((a)-1) + (na)*((b)-1))]
#define TAB2(off, it, a) ssib_tab[(off) + ((it)-1) + 13*((a)-1)]
#define TAB1(off, it) ssib_tab[(off) + ((it)-1)]

// ---------------------------------------------------------------- math
__device__ __forceinline__ float ss_exp(float x) { return gfk_exp(x); }
__device__ __forceinline__ float ss_log(float x) { return gfk_log(x); }
__device__ __forceinline__ float ss_pow(float x, float y) { return gfk_pow(x, y); }
__device__ __forceinline__ double ss_dexp(double x) { return glibc_exp(x); }
__device__ __forceinline__ double ss_dpow(double x, double y) { return glibc_pow(x, y); }
__device__ __forceinline__ float ss_max(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float ss_min(float a, float b) { return (b < a) ? b : a; }
__device__ __forceinline__ float ss_sign(float a, float b) { return copysignf(fabsf(a), b); }

// ---------------------------------------------------------------- constants
// Module parameters, :10-38.  Products of literals are formed with the
// rounded single-precision operations gfortran folds them with.
#define CPAIR    1004.6f
#define STEFAN   __fmul_rn(5.669f, 10E-9f)
#define GRAV     9.81f
#define VKC      0.4f
#define PIE      3.14159265f
#define TIMCON   __fdiv_rn(PIE, 86400.f)
#define CLAI     __fmul_rn(__fmul_rn(4.2f, 1000.f), 0.2f)
#define CW       __fmul_rn(__fmul_rn(4.2f, 1000.f), 1000.f)
#define TF       273.16f
#define GASR     287.05f
#define HLAT     2.52E6f
#define SNOMEL   __fmul_rn(370518.5f, 1000.f)
#define SSISNOW  0.04f
#define FLMIN    0.03f
#define FLMAX    0.10f
#define DZMIN    0.002f
#define WOMIN    0.0004f
#define CL       4212.7f
#define DLM      333500.0f          // 3.335d5 assigned to a REAL parameter
#define RHOWATER 1000.0f
#define DICE     920.0f
#define DKSATSNOW 0.01f
#define SNODEP_CR 0.07f
#define NLAY 3                      // N
#define N1 4

struct SsibSnow {   // SSIB's local snow arrays, 1-based, N2 = 4
    float ss[5], sso[5], porosity[5], h[5], ho[5], bi[5], bio[5], dz[5],
          dzo[5], bw[5], bwo[5], bl[5], blo[5], tssn[5], tssno[5], w[5],
          wo[5], wf[5], fi[5], fio[5], fl[5], flo[5], dmlt[5], dmlto[5],
          bt[5], bto[5], s[5], so[5], ct[5], cto[5], dliqvol[5], dicevol[5],
          qk[5], pdzdtc[5], dmass[5], dsol[5], dhp[5], thk[5];
};

// ================================================================ CROPS
__device__ void ss_crops(float xlat, float day, float *chil, float *zlt,
                         float *green, float *xcover)
{
    const float wlai[10] = {0.f, 1.0f, 2.0f, 6.0f, 4.0f, 3.0f, 1.0f, 0.01f, 0.01f, 1.0f};
    const float wgrn[10] = {0.f, 0.6f, 0.9f, 0.8f, 0.5f, 0.2f, 0.1f, 0.01f, 0.01f, 0.6f};
    const int ihead = 3, iend = 9;
    const float dend = 244.f;
    float phenst[10] = {0.f};
    float rday = 0.f, phi = 0.f, aphi = 0.f, flip = 0.f, demg = 0.f;
    float chilw = 0.f, tlai = 0.f, grlf = 0.f, top = 0.f, bot = 0.f;
    float diff1 = 0.f, diff2 = 0.f, perc = 0.f, zltgmx = 0.f;
    int ns = 0, icond = 0;
    (void)zltgmx;
    if (xlat < 0.f) {
        rday = day + 184.f;
        if (rday > 365.f) rday = rday - 365.f;
    } else {
        rday = day;
    }
    phi = xlat;
    aphi = fabsf(phi);
    if (aphi > 55.f) phi = ss_sign(55.f, phi);
    if (aphi < 20.f) phi = ss_sign(20.f, phi);
    flip = 0.0f;
    phenst[2] = 4.50f * fabsf(phi) - 64.0f + flip;
    phenst[3] = 4.74f * fabsf(phi) - 46.2f + flip;
    phenst[4] = 4.86f * fabsf(phi) - 30.8f + flip;
    phenst[5] = 4.55f * fabsf(phi) - 3.0f + flip;
    phenst[6] = 4.35f * fabsf(phi) + 11.5f + flip;
    phenst[7] = phenst[6] + 3.0f;
    demg = fabsf(5.21f * fabsf(phi) - 0.3f);
    phenst[1] = phenst[2] - demg;
    phenst[9] = phenst[1];
    phenst[8] = phenst[9] - 5.0f;
    for (ns = 1; ns <= 9; ++ns) {
        if (phenst[ns] < 0.0f) phenst[ns] = phenst[ns] + 365.f;
        if (phenst[ns] > 365.f) phenst[ns] = phenst[ns] - 365.f;
    }
    chilw = -0.02f;
    tlai = 0.5f;
    grlf = 0.6f;
    for (ns = 1; ns <= 8; ++ns) {
        top = phenst[ns + 1];
        bot = phenst[ns];
        diff1 = top - bot;
        diff2 = rday - bot;
        if (rday >= bot && rday <= top) goto L40;
        if (bot < top) goto L50;
        icond = 0;
        if (rday >= bot && rday <= 365.f) icond = 1;
        if (rday >= 0.0f && rday <= top) icond = 2;
        if (icond == 0) goto L50;
        if (icond == 2) goto L35;
        diff1 = 365.f - bot + top;
        diff2 = rday - bot;
        goto L40;
L35:
        diff1 = 365.f - bot + top;
        diff2 = 365.f - bot + rday;
L40:
        if ((rday > phenst[ihead]) && (rday <= dend)) {
            tlai = wlai[ihead];
            grlf = wgrn[ihead];
            goto L77;
        }
        if ((rday > dend) && (rday <= phenst[iend])) {
            diff1 = phenst[iend] - dend;
            diff2 = rday - dend;
            perc = diff2 / diff1;
            tlai = perc * (wlai[iend] - wlai[ihead]) + wlai[ihead];
            grlf = perc * (wgrn[iend] - wgrn[ihead]) + wgrn[ihead];
            goto L77;
        }
        perc = diff2 / diff1;
        tlai = perc * (wlai[ns + 1] - wlai[ns]) + wlai[ns];
        grlf = perc * (wgrn[ns + 1] - wgrn[ns]) + wgrn[ns];
L77:
        goto L95;
L50:
        ;
    }
    // A completed DO leaves its index at the last value plus one.
L95:
    xcover[1] = 0.90f * (1.0f - ss_exp(-tlai));
    if (ns != 1 && ns != 2) chilw = -0.2f;
    zlt[1] = tlai;
    green[1] = grlf;
    chil[1] = chilw;
}

// ================================================================ ROOT1
__device__ void ss_root1(float phsat, float bee, const float *www, float *phsoil)
{
    for (int il = 1; il <= 3; ++il)
        phsoil[il] = phsat * ss_pow(ss_max(0.05f, www[il]), (-bee));
}

// ================================================================ STOMA1
__device__ void ss_stoma1(const float *green, const float *vcover,
                          const float *chil, const float *zlt, const float *par,
                          const float *pd, float extk[3][4][3], float sunang,
                          float *rst, float rstpar[3][4], float ctlpa)
{
    float at = 0.f, xabc = 0.f, gamma = 0.f, power1 = 0.f, power2 = 0.f;
    float aa = 0.f, bb = 0.f, zat = 0.f, pow1 = 0.f, pow2 = 0.f, zk = 0.f;
    float pw = 0.f, ekat = 0.f, avflux = 0.f, rho4 = 0.f;
    for (int iveg = 1; iveg <= 2; ++iveg) {
        at = zlt[iveg] / vcover[iveg];
        if (sunang <= 0.02f) {
            xabc = rstpar[iveg][1] / rstpar[iveg][2] + rstpar[iveg][3];
            rst[iveg] = 0.5f / xabc * at;
            if (rst[iveg] < 0.f) rst[iveg] = 0.00001f;
            goto L1010;
        }
        gamma = (rstpar[iveg][1] + rstpar[iveg][2] * rstpar[iveg][3]) /
                rstpar[iveg][3];
        power1 = ss_min(50.f, at * extk[iveg][1][1]);
        power2 = ss_min(50.f, at * extk[iveg][1][2]);
        aa = 0.5f - 0.633f * chil[iveg] - 0.33f * chil[iveg] * chil[iveg];
        bb = 0.877f * (1.f - 2.f * aa);
        zat = ss_log((ss_exp(-power1) + 1.f) / 2.f) * pd[iveg]
              / (power1 / at);
        zat = zat + ss_log((ss_exp(-power2) + 1.f) / 2.f)
              * (1.f - pd[iveg]) / (power2 / at);
        pow1 = ss_min(50.f, (power1 * zat / at));
        pow2 = ss_min(50.f, (power2 * zat / at));
        zk = 1.f / zat * ss_log(pd[iveg] * ss_exp(pow1)
                                + (1.f - pd[iveg]) * ss_exp(pow2));
        pw = ss_min(50.f, zk * at);
        ekat = ss_exp(pw);
        avflux = par[iveg] * (pd[iveg] / sunang * (aa + bb * sunang)
                 + (1.f - pd[iveg]) * (bb / 3.f + aa * 1.5f
                 + bb / 4.f * PIE));
        rho4 = gamma / avflux;
        rst[iveg] = rstpar[iveg][2] / gamma * ss_log((rho4 * ekat + 1.f) /
                    (rho4 + 1.f));
        rst[iveg] = rst[iveg] - ss_log((rho4 + 1.f / ekat) /
                    (rho4 + 1.f));
        rst[iveg] = rst[iveg] / (zk * rstpar[iveg][3]);
L1010:
        rst[iveg] = 1.f / (rst[iveg] * green[iveg] + 0.0000001f);
    }
    rst[1] = rst[1] * ctlpa;
}

// ================================================================ VEGOUT
__device__ void ss_vegout(float xtran[3][4][3], float xref[3][4][3],
                          float *xgreen, float *xvcover, float *xchil,
                          float xrstpar[3][4], float *xtopt, float *xtl,
                          float *xtu, float *xdefac, float *xph1, float *xph2,
                          float *xzlt, float &xz0, float &xdd, float &xz2,
                          float &xz1, float &xrdc, float &xrbc, float *xrootd,
                          float *xsoref, float &xbee, float &xphsat,
                          float &xporos, float &xsatco, float &xslope,
                          float *xdepth, int month, int itype)
{
    for (int iw = 1; iw <= 3; ++iw) {
        xtran[1][iw][1] = TAB4(T_TRAN0, itype, 1, iw, 1, 2, 3);
        xtran[1][iw][2] = TAB4(T_TRAN0, itype, 1, iw, 2, 2, 3);
        xtran[2][iw][1] = TAB4(T_TRAN0, itype, 2, iw, 1, 2, 3);
        xtran[2][iw][2] = TAB4(T_TRAN0, itype, 2, iw, 2, 2, 3);
        xref[1][iw][1] = TAB4(T_REF0, itype, 1, iw, 1, 2, 3);
        xref[1][iw][2] = TAB4(T_REF0, itype, 1, iw, 2, 2, 3);
        xref[2][iw][1] = TAB4(T_REF0, itype, 2, iw, 1, 2, 3);
        xref[2][iw][2] = TAB4(T_REF0, itype, 2, iw, 2, 2, 3);
        xrstpar[1][iw] = TAB3(T_RSTPAR0, itype, 1, iw, 2);
        xrstpar[2][iw] = TAB3(T_RSTPAR0, itype, 2, iw, 2);
        xsoref[iw] = TAB2(T_SOREF0, itype, iw);
    }
    for (int iv = 1; iv <= 2; ++iv) {
        xchil[iv] = TAB2(T_CHIL0, itype, iv);
        xtopt[iv] = TAB2(T_TOPT0, itype, iv);
        xtl[iv] = TAB2(T_TL0, itype, iv);
        xtu[iv] = TAB2(T_TU0, itype, iv);
        xdefac[iv] = TAB2(T_DEFAC0, itype, iv);
        xph1[iv] = TAB2(T_PH10, itype, iv);
        xph2[iv] = TAB2(T_PH20, itype, iv);
        xrootd[iv] = TAB2(T_ROOTD0, itype, iv);
        xzlt[iv] = TAB3(T_ZLT0, itype, month, iv, 12);
        xgreen[iv] = TAB3(T_GREEN0, itype, month, iv, 12);
        xvcover[iv] = TAB3(T_VCOVER0, itype, month, iv, 12);
    }
    for (int idep = 1; idep <= 3; ++idep) xdepth[idep] = TAB2(T_DEPTH0, itype, idep);
    xbee = TAB1(T_BEE0, itype);
    xphsat = TAB1(T_PHSAT0, itype);
    xsatco = TAB1(T_SATCO0, itype);
    xporos = TAB1(T_POROS0, itype);
    xslope = TAB1(T_SLOPE0, itype);
    xz2 = TAB2(T_Z20, itype, month);
    xz1 = TAB2(T_Z10, itype, month);
    xz0 = TAB2(T_Z000, itype, month);
    xdd = TAB2(T_D0, itype, month);
    xrbc = TAB2(T_RBC0, itype, month);
    xrdc = TAB2(T_RDC0, itype, month);
}

// ================================================================ COMBO
__device__ void ss_combo(float ddz2, float &dzp, float &dzm, float &wp,
                         float &wm, float &hp, float &hm, float &tp,
                         float &tm, float &bwp, float &bwm, float &bip,
                         float &bim, float &blp, float &blm, float &btp,
                         float &btm, float &fip, float &fim, float &flp,
                         float &flm, float &ctp, float &ctm, float &dliqvolp,
                         float &dliqvolm, float &dicevolp, float &dicevolm)
{
    float ratio = 0.f, dmlt = 0.f;
    ratio = ddz2 / dzm;
    dzp = dzp + ratio * dzm;
    wp = wp + ratio * wm;
    hp = hp + ratio * hm;
    bwp = wp * RHOWATER / dzp;
    btp = bwp;
    ctp = (1.9e6f) * (bwp / 920.0f);
    dmlt = wp * RHOWATER * DLM;
    if (hp >= (-1.0f) * dmlt) {
        tp = 273.16f;
        fip = (-1.0f) * hp / dmlt;
        flp = 1.0f - fip;
        blp = bwp * flp;
        bip = bwp * fip;
        dliqvolp = blp / RHOWATER;
        dicevolp = bip / DICE;
    } else {
        flp = 0.0f;
        fip = 1.0f;
        tp = (hp + dmlt) / (ctp * dzp) + 273.16f;
        bip = bwp;
        blp = 0.0f;
        dliqvolp = 0.0f;
        dicevolp = bip / DICE;
    }
    dzm = dzm - ratio * dzm;
    wm = wm - ratio * wm;
    hm = hm - ratio * hm;
}

// ================================================================ COMPACT
__device__ void ss_compact(float bi, float t, float bl, float overburden,
                           float &pdzdt, float ss, float dice)
{
    // data c2,c3,c4,c5/23d-3,2.777d-6,0.04,2.0/  dm/150/  eta0/0.9d6/
    const float c2 = 23e-3f, c3 = 2.777e-6f, c4 = 0.04f, c5 = 2.0f;
    const float dm = 150.f, eta0 = 0.9e6f;
    float ddz1 = 0.f, ddz2 = 0.f, ddz3 = 0.f;
    if (bi >= dice || ss >= 1.f) return;
    ddz1 = -c3 * ss_exp(-c4 * (273.15f - t));
    // exp(-46.0d-3*(bi-dm)): binary64, rounded on assignment.
    if (bi > dm) ddz1 = (float)((double)ddz1 * ss_dexp(-46.0e-3 * (double)(bi - dm)));
    if (bl > 0.01f) ddz1 = ddz1 * c5;
    ddz2 = -overburden * ss_exp(-0.08f * (273.15f - t) - c2 * bi) / eta0;
    ddz3 = 0.0f;
    pdzdt = ddz1 + ddz2 + ddz3;
}

// ================================================================ GETMET
__device__ void ss_getmet(int &iptype, float prcp_total, float tair,
                          float &prcp_s, float &prcp_w, float &fi_fall,
                          float &fl_fall, float &bi_fall, float &bl_fall)
{
    if (prcp_total > 0.f) {
        if (iptype == 2) {
            prcp_s = prcp_total;
            prcp_w = 0.0f;
        } else if (iptype == 1) {
            prcp_w = prcp_total;
            prcp_s = 0.0f;
            fl_fall = 1.0f;
            fi_fall = 0.f;
            bl_fall = 1000.0f;
            bi_fall = 0.0f;
        }
    } else {
        prcp_w = 0.0f;
        prcp_s = 0.0f;
        iptype = 0;
        return;
    }
    if (iptype != 1) {
        if (tair > 275.15f) {
            bi_fall = 189.f;
        } else if (tair > 258.16f) {
            // 50+1.7*(TAIR-258.16)**1.5d0 in binary64
            bi_fall = (float)(50.0 + (double)1.7f *
                              ss_dpow((double)(tair - 258.16f), 1.5));
        } else {
            bi_fall = 50.f;
        }
        fl_fall = 0.f;
        fi_fall = 1.0f;
        bl_fall = 0.0f;
    }
}

// ================================================================ INTERCS
__device__ void ss_intercs(float dtt, const float *vcover, const float *zlai,
                           float tm, float &tc, float &tgs, float *capac,
                           float *www, float ppc, float ppl, float &roff,
                           const float *zdepth, float poros, float &ccx,
                           float &cg, float satco, const float *satcap,
                           float &spwet, float extk[3][4][3], int isnow,
                           float &p0, float &csoil, float &dzsoil,
                           float &chisl, float &smelt)
{
    const float pc11 = 20.f, pc12 = .206E-8f, pc21 = 0.0001f, pc22 = 0.9999f;
    const float bp = 20.f;
    float capacp[3] = {0.f}, snowp[3] = {0.f}, snoww[3] = {0.f};
    float ap = 0.f, cp = 0.f, totalp = 0.f, thru = 0.f, fpi = 0.f;
    float theta = 0.f, difsl = 0.f, rocs = 0.f, d1 = 0.f;
    float thalas = 0.f, oceans = 0.f, polar = 0.f, spwet1 = 0.f, ts = 0.f;
    float specht = 0.f, xsc = 0.f, zload = 0.f, tti = 0.f, xs = 0.f;
    float arg = 0.f, tex = 0.f, pinf = 0.f, equdep = 0.f, roffo = 0.f;
    float diff = 0.f, ccp = 0.f, cct = 0.f, tsd = 0.f, freeze = 0.f;
    float tta = 0.f, ttb = 0.f, cca = 0.f, ccb = 0.f, ccc = 0.f, zmelt = 0.f;
    int iveg = 0;
    ap = pc21;
    cp = pc22;
    totalp = ppc + ppl;
    if (totalp < 1.E-8f) goto L6000;
    ap = ppc / totalp * pc11 + ppl / totalp * pc21;
    cp = ppc / totalp * pc12 + ppl / totalp * pc22;
L6000:
    roff = 0.f;
    thru = 0.f;
    fpi = 0.f;
    theta = www[1] * poros;
    chisl = (9.8E-4f + 1.2E-3f * theta) / (1.1f - 0.4f * theta);
    chisl = chisl * 4.186E2f;
    difsl = 5.E-7f;
    rocs = chisl / difsl;
    d1 = sqrtf(difsl * 86400.0f);
    csoil = rocs * d1 / sqrtf(PIE) / 2.0f;
    dzsoil = d1 / sqrtf(PIE) / 2.0f;
    thalas = 0.f;
    oceans = 0.f;
    polar = 0.f;
    csoil = csoil * (1.0f - thalas) + 10.E10f * oceans + polar * 3.6f * 4.2E4f;
    p0 = totalp * 0.001f;
    for (iveg = 1; iveg <= 2; ++iveg) {
        spwet1 = ss_min(0.05f, capac[iveg]) * CW;
        ts = tc;
        specht = zlai[1] * CLAI;
        if (iveg == 1) goto L1100;
        ts = tgs;
        specht = csoil;
L1100:
        xsc = ss_max(0.f, capac[iveg] - satcap[iveg]);
        if (iveg == 2 && ts <= TF) goto L1170;
        capac[iveg] = capac[iveg] - xsc;
        roff = roff + xsc;
L1170:
        capacp[iveg] = 0.f;
        snowp[iveg] = 0.f;
        if (ts > TF) capacp[iveg] = capac[iveg];
        if (ts <= TF) snowp[iveg] = capac[iveg];
        capac[iveg] = capacp[iveg];
        snoww[iveg] = snowp[iveg];
        zload = capac[iveg] + snoww[iveg];
        fpi = (1.f - ss_exp(-extk[iveg][3][1] * zlai[iveg] / vcover[iveg]))
              * vcover[iveg];
        tti = p0 * (1.f - fpi);
        xs = 1.f;
        if (p0 < 1.E-9f) goto L1150;
        arg = (satcap[iveg] - zload) / (p0 * fpi * ap) - cp / ap;
        if (arg < 1.E-9f) goto L1150;
        xs = -1.f / bp * ss_log(arg);
        xs = ss_min(xs, 1.f);
        xs = ss_max(xs, 0.f);
L1150:
        tex = p0 * fpi * (ap / bp * (1.f - ss_exp(-bp * xs)) + cp * xs) -
              (satcap[iveg] - zload) * xs;
        tex = ss_max(tex, 0.f);
        thru = tti + tex;
        if (iveg == 2 && tgs <= TF) thru = 0.f;
        pinf = p0 - thru;
        if (tm > TF) capac[iveg] = capac[iveg] + pinf;
        if (tm <= TF) snoww[iveg] = snoww[iveg] + pinf;
        if (iveg == 1) goto L1300;
        if (tm > TF) goto L1200;
        snoww[iveg] = snowp[iveg] + p0;
        thru = 0.f;
        goto L1300;
L1200:
        equdep = satco * dtt;
        xs = 1.f;
        if (thru < 1.E-9f) goto L1250;
        arg = equdep / (thru * ap) - cp / ap;
        if (arg < 1.E-9f) goto L1250;
        xs = -1.f / bp * ss_log(arg);
        xs = ss_min(xs, 1.f);
        xs = ss_max(xs, 0.f);
L1250:
        roffo = thru * (ap / bp * (1.f - ss_exp(-bp * xs)) + cp * xs)
                - equdep * xs;
        roffo = ss_max(roffo, 0.f);
        roff = roff + roffo;
        www[1] = www[1] + (thru - roffo) / (poros * zdepth[1]);
L1300:
        diff = (capac[iveg] + snoww[iveg] - capacp[iveg] - snowp[iveg]) * CW;
        ccp = specht + spwet1;
        cct = specht + spwet1 + diff;
        tsd = (ts * ccp + tm * diff) / cct;
        freeze = 0.f;
        if (ts > TF && tm > TF) goto L2000;
        if (ts <= TF && tm <= TF) goto L2000;
        tta = ts;
        ttb = tm;
        cca = ccp;
        ccb = diff;
        if (tsd > TF) goto L2100;
        ccc = capacp[iveg] * SNOMEL;
        if (ts < tm) ccc = diff * SNOMEL / CW;
        tsd = (tta * cca + ttb * ccb + ccc) / cct;
        freeze = (TF * cct - (tta * cca + ttb * ccb));
        freeze = (ss_min(ccc, freeze)) / SNOMEL;
        if (tsd > TF) tsd = TF - 0.1f;
        goto L2000;
L2100:
        ccc = -snoww[iveg] * SNOMEL;
        if (ts > tm) ccc = -diff * SNOMEL / CW;
        tsd = (tta * cca + ttb * ccb + ccc) / cct;
        freeze = (TF * cct - (tta * cca + ttb * ccb));
        freeze = (ss_max(ccc, freeze)) / SNOMEL;
        if (tsd <= TF) tsd = TF - 0.1f;
L2000:
        smelt = freeze;
        snoww[iveg] = snoww[iveg] + freeze;
        capac[iveg] = capac[iveg] - freeze;
        if (iveg == 1) tc = tsd;
        if (iveg == 2) tgs = tsd;
        if (snoww[iveg] < 0.0000001f) goto L3000;
        zmelt = capac[iveg];
        capac[iveg] = 0.f;
        www[1] = www[1] + zmelt / (poros * zdepth[1]);
L3000:
        capac[iveg] = capac[iveg] + snoww[iveg];
        snoww[iveg] = 0.f;
        p0 = thru;
        if (isnow == 0) goto L1001;
    }
L1001:
    ccx = zlai[1] * CLAI + capac[1] * CW;
    spwet = ss_min(0.05f, capac[2]) * CW;
    cg = (csoil + spwet);
}

// ================================================================ INTERC
__device__ void ss_interc(float dtt, const float *vcover, const float *zlt,
                          float tm, float &tc, float &tgs, float *capac,
                          float *www, float ppc, float ppl, float &roff,
                          const float *zdepth, float poros, float &ccx,
                          float &cg, float satco, const float *satcap,
                          float &spwet, float extk[3][4][3], float &rnoffs,
                          float &filtr, float &smelt)
{
    const float pc11 = 20.f, pc12 = .206E-8f, pc21 = 0.0001f, pc22 = 0.9999f;
    const float bp = 20.f;
    float capacp[3] = {0.f}, snowp[3] = {0.f}, snoww[3] = {0.f};
    float ap = 0.f, cp = 0.f, totalp = 0.f, thru = 0.f, fpi = 0.f;
    float theta = 0.f, chisl = 0.f, difsl = 0.f, rocs = 0.f, d1 = 0.f;
    float csoil = 0.f, thalas = 0.f, oceans = 0.f, polar = 0.f, p0 = 0.f;
    float spwet1 = 0.f, ts = 0.f, specht = 0.f, xsc = 0.f, zload = 0.f;
    float tti = 0.f, xs = 0.f, arg = 0.f, tex = 0.f, pinf = 0.f;
    float equdep = 0.f, roffo = 0.f, diff = 0.f, ccp = 0.f, cct = 0.f;
    float tsd = 0.f, freeze = 0.f, tta = 0.f, ttb = 0.f, cca = 0.f;
    float ccb = 0.f, ccc = 0.f, zmelt = 0.f;
    int iveg = 0;
    ap = pc21;
    cp = pc22;
    totalp = ppc + ppl;
    if (totalp < 1.E-8f) goto L6000;
    ap = ppc / totalp * pc11 + ppl / totalp * pc21;
    cp = ppc / totalp * pc12 + ppl / totalp * pc22;
L6000:
    roff = 0.f;
    thru = 0.f;
    fpi = 0.f;
    theta = www[1] * poros;
    chisl = (9.8E-4f + 1.2E-3f * theta) / (1.1f - 0.4f * theta);
    chisl = chisl * 4.186E2f;
    difsl = 5.E-7f;
    rocs = chisl / difsl;
    d1 = sqrtf(difsl * 86400.0f);
    csoil = rocs * d1 / sqrtf(PIE) / 2.0f;
    thalas = 0.f;
    oceans = 0.f;
    polar = 0.f;
    csoil = csoil * (1.0f - thalas) + 10.E10f * oceans + polar * 3.6f * 4.2E4f;
    p0 = totalp * 0.001f;
    for (iveg = 1; iveg <= 2; ++iveg) {
        spwet1 = ss_min(0.05f, capac[iveg]) * CW;
        ts = tc;
        specht = zlt[1] * CLAI;
        if (iveg == 1) goto L1100;
        ts = tgs;
        specht = csoil;
L1100:
        xsc = ss_max(0.f, capac[iveg] - satcap[iveg]);
        if (iveg == 2 && ts <= TF) goto L1170;
        capac[iveg] = capac[iveg] - xsc;
        roff = roff + xsc;
        rnoffs = xsc * 1000.f + rnoffs;
L1170:
        capacp[iveg] = 0.f;
        snowp[iveg] = 0.f;
        if (ts > TF) capacp[iveg] = capac[iveg];
        if (ts <= TF) snowp[iveg] = capac[iveg];
        capac[iveg] = capacp[iveg];
        snoww[iveg] = snowp[iveg];
        zload = capac[iveg] + snoww[iveg];
        fpi = (1.f - ss_exp(-extk[iveg][3][1] * zlt[iveg] / vcover[iveg]))
              * vcover[iveg];
        tti = p0 * (1.f - fpi);
        xs = 1.f;
        if (p0 < 1.E-9f) goto L1150;
        arg = (satcap[iveg] - zload) / (p0 * fpi * ap) - cp / ap;
        if (arg < 1.E-9f) goto L1150;
        xs = -1.f / bp * ss_log(arg);
        xs = ss_min(xs, 1.f);
        xs = ss_max(xs, 0.f);
L1150:
        tex = p0 * fpi * (ap / bp * (1.f - ss_exp(-bp * xs)) + cp * xs) -
              (satcap[iveg] - zload) * xs;
        tex = ss_max(tex, 0.f);
        thru = tti + tex;
        if (iveg == 2 && tgs <= TF) thru = 0.f;
        pinf = p0 - thru;
        if (tm > TF) capac[iveg] = capac[iveg] + pinf;
        if (tm <= TF) snoww[iveg] = snoww[iveg] + pinf;
        if (iveg == 1) goto L1300;
        if (tm > TF) goto L1200;
        snoww[iveg] = snowp[iveg] + p0;
        thru = 0.f;
        goto L1300;
L1200:
        equdep = satco * dtt;
        xs = 1.f;
        if (thru < 1.E-9f) goto L1250;
        arg = equdep / (thru * ap) - cp / ap;
        if (arg < 1.E-9f) goto L1250;
        xs = -1.f / bp * ss_log(arg);
        xs = ss_min(xs, 1.f);
        xs = ss_max(xs, 0.f);
L1250:
        roffo = thru * (ap / bp * (1.f - ss_exp(-bp * xs)) + cp * xs)
                - equdep * xs;
        roffo = ss_max(roffo, 0.f);
        roff = roff + roffo;
        rnoffs = rnoffs + roffo * 1000.f;
        filtr = filtr + (thru - roffo);
        www[1] = www[1] + (thru - roffo) / (poros * zdepth[1]);
L1300:
        diff = (capac[iveg] + snoww[iveg] - capacp[iveg] - snowp[iveg]) * CW;
        ccp = specht + spwet1;
        cct = specht + spwet1 + diff;
        tsd = (ts * ccp + tm * diff) / cct;
        freeze = 0.f;
        if (ts > TF && tm > TF) goto L2000;
        if (ts <= TF && tm <= TF) goto L2000;
        tta = ts;
        ttb = tm;
        cca = ccp;
        ccb = diff;
        if (tsd > TF) goto L2100;
        ccc = capacp[iveg] * SNOMEL;
        if (ts < tm) ccc = diff * SNOMEL / CW;
        tsd = (tta * cca + ttb * ccb + ccc) / cct;
        freeze = (TF * cct - (tta * cca + ttb * ccb));
        freeze = (ss_min(ccc, freeze)) / SNOMEL;
        if (tsd > TF) tsd = TF - 0.1f;
        goto L2000;
L2100:
        ccc = -snoww[iveg] * SNOMEL;
        if (ts > tm) ccc = -diff * SNOMEL / CW;
        tsd = (tta * cca + ttb * ccb + ccc) / cct;
        freeze = (TF * cct - (tta * cca + ttb * ccb));
        freeze = (ss_max(ccc, freeze)) / SNOMEL;
        if (tsd <= TF) tsd = TF - 0.1f;
L2000:
        smelt = freeze;
        snoww[iveg] = snoww[iveg] + freeze;
        capac[iveg] = capac[iveg] - freeze;
        if (iveg == 1) tc = tsd;
        if (iveg == 2) tgs = tsd;
        if (snoww[iveg] < 0.0000001f) goto L3000;
        zmelt = 0.f;
        zmelt = capac[iveg];
        capac[iveg] = 0.f;
        www[1] = www[1] + zmelt / (poros * zdepth[1]);
        filtr = filtr + zmelt;
L3000:
        capac[iveg] = capac[iveg] + snoww[iveg];
        snoww[iveg] = 0.f;
        freeze = 0.0f;
        p0 = thru;
    }
    ccx = zlt[1] * CLAI + capac[1] * CW;
    spwet = ss_min(0.05f, capac[2]) * CW;
    cg = (csoil + spwet);
}

// ================================================================ LAYER1
__device__ void ss_layer1(float csoil, float tgs, float dzsoil, const float *h,
                          const float *w, float snowdepth, float &swe,
                          float &stemp)
{
    float snh = 0.f, dmlto = 0.f, scv = 0.f;
    (void)dzsoil;
    swe = w[1] + w[2] + w[3];
    snh = h[1] + h[2] + h[3] + csoil * (tgs - 273.16f);
    dmlto = swe * DLM * RHOWATER;
    scv = 1.9e+6f * (swe / snowdepth) / DICE;
    if (snh > 0.0f) {
        stemp = snh / (swe * 4.18f * 1.0e6f + csoil) + 273.16f;   // 10**6. folds to 1e6
    } else if (snh > -dmlto) {
        stemp = 273.16f;
    } else {
        stemp = (snh + dmlto) / (scv * snowdepth + csoil) + 273.16f;
    }
}

// ================================================================ LAYERN
__device__ void ss_layern(float tg, float snow_we, float snow_depth,
                          float *dz0, float *bw0, float *w0, float *bt0,
                          float *ct0, float *fl0, float *fi0, float *h0,
                          float *bl0, float *bi0, float *dliqv0, float *dicev0,
                          float *tssn0, float *dmlt0)
{
    int i = 0;
    if (snow_depth > 0.05f && snow_depth <= 0.06f) {
        dz0[1] = 0.02f;
        dz0[2] = 0.02f;
        dz0[3] = snow_depth - dz0[1] - dz0[2];
    } else if (snow_depth > 0.06f && snow_depth <= 0.08f) {
        dz0[3] = 0.02f;
        dz0[2] = 0.02f;
        dz0[1] = snow_depth - dz0[3] - dz0[2];
    } else if (snow_depth > 0.08f && snow_depth <= 0.62f) {
        dz0[3] = 0.02f;
        dz0[2] = (snow_depth - dz0[3]) * 0.33333333f;
        dz0[1] = (snow_depth - dz0[3]) * 0.66666667f;
    } else if (snow_depth > 0.62f) {
        dz0[3] = 0.02f;
        dz0[2] = 0.20f;
        dz0[1] = snow_depth - dz0[3] - dz0[2];
    }
    for (i = 1; i <= NLAY; ++i) {
        tssn0[i] = tg;
        bw0[i] = snow_we * RHOWATER / snow_depth;
    }
    for (i = 1; i <= NLAY; ++i) {
        w0[i] = (bw0[i] * dz0[i]) / RHOWATER;
        bt0[i] = bw0[i];
        ct0[i] = (bw0[i] / 920.0f) * 1.9e+6f;
        if (tssn0[i] == 273.16f) {
            fl0[i] = FLMIN;
            fi0[i] = 1.0f - FLMIN;
            h0[i] = (-1.0f) * w0[i] * fi0[i] * DLM * RHOWATER;
            bl0[i] = bw0[i] * fl0[i];
            bi0[i] = bw0[i] * fi0[i];
            dliqv0[i] = bl0[i] / RHOWATER;
            dicev0[i] = bi0[i] / DICE;
        } else if (tssn0[i] < 273.16f) {
            fl0[i] = 0.0f;
            fi0[i] = 1.0f;
            dmlt0[i] = w0[i] * DLM * RHOWATER;
            h0[i] = (tssn0[i] - 273.16f) * ct0[i] * dz0[i] - dmlt0[i];
            bl0[i] = 0.0f;
            bi0[i] = bw0[i];
            dliqv0[i] = 0.0f;
            dicev0[i] = bi0[i] / DICE;
        }
    }
}

// ================================================================ MODNODE
__device__ void ss_modnode(float &snowdepth, SsibSnow &a)
{
    float dz1 = 0.f, dz2 = 0.f, dz3 = 0.f, ddz3 = 0.f, sum12 = 0.f, ddz2 = 0.f;
    (void)dz1;
    if (snowdepth <= 0.06f) {
        dz1 = 0.02f;
        dz2 = 0.02f;
        dz3 = snowdepth - (dz2 + dz1);
    } else if (snowdepth > 0.06f) {
        dz3 = 0.02f;
    }
    ddz3 = dz3 - a.dzo[3];
    if (ddz3 > 0.0f) {
        ddz3 = ss_min(ddz3, a.dzo[2]);
        ss_combo(ddz3, a.dzo[3], a.dzo[2], a.wo[3], a.wo[2], a.ho[3], a.ho[2],
                 a.tssno[3], a.tssno[2], a.bwo[3], a.bwo[2], a.bio[3], a.bio[2],
                 a.blo[3], a.blo[2], a.bto[3], a.bto[2], a.fio[3], a.fio[2],
                 a.flo[3], a.flo[2], a.cto[3], a.cto[2], a.dliqvol[3],
                 a.dliqvol[2], a.dicevol[3], a.dicevol[2]);
    } else {
        ddz3 = -ddz3;
        ss_combo(ddz3, a.dzo[2], a.dzo[3], a.wo[2], a.wo[3], a.ho[2], a.ho[3],
                 a.tssno[2], a.tssno[3], a.bwo[2], a.bwo[3], a.bio[2], a.bio[3],
                 a.blo[2], a.blo[3], a.bto[2], a.bto[3], a.fio[2], a.fio[3],
                 a.flo[2], a.flo[3], a.cto[2], a.cto[3], a.dliqvol[2],
                 a.dliqvol[3], a.dicevol[2], a.dicevol[3]);
    }
    sum12 = a.dzo[1] + a.dzo[2];
    if (snowdepth <= 0.06f) {
        dz2 = 0.5f * sum12;
    } else if (snowdepth > 0.06f && snowdepth <= 0.08f) {
        dz2 = 0.02f;
    } else if (snowdepth > 0.08f && snowdepth <= 0.62f) {
        dz2 = 0.33333333f * sum12;
    } else if (snowdepth > 0.62f) {
        dz2 = 0.20f;
    }
    ddz2 = dz2 - a.dzo[2];
    if (ddz2 > 0.0f) {
        ss_combo(ddz2, a.dzo[2], a.dzo[1], a.wo[2], a.wo[1], a.ho[2], a.ho[1],
                 a.tssno[2], a.tssno[1], a.bwo[2], a.bwo[1], a.bio[2], a.bio[1],
                 a.blo[2], a.blo[1], a.bto[2], a.bto[1], a.fio[2], a.fio[1],
                 a.flo[2], a.flo[1], a.cto[2], a.cto[1], a.dliqvol[2],
                 a.dliqvol[1], a.dicevol[2], a.dicevol[1]);
    } else {
        ddz2 = -ddz2;
        ss_combo(ddz2, a.dzo[1], a.dzo[2], a.wo[1], a.wo[2], a.ho[1], a.ho[2],
                 a.tssno[1], a.tssno[2], a.bwo[1], a.bwo[2], a.bio[1], a.bio[2],
                 a.blo[1], a.blo[2], a.bto[1], a.bto[2], a.fio[1], a.fio[2],
                 a.flo[1], a.flo[2], a.cto[1], a.cto[2], a.dliqvol[1],
                 a.dliqvol[2], a.dicevol[1], a.dicevol[2]);
    }
    snowdepth = a.dzo[1] + a.dzo[2] + a.dzo[3];
}

// ================================================================ NEWSNOW
__device__ void ss_newsnow(float prcp, float bifall, float blfall, float flfall,
                           float tkair, float &dzo, float &wo, float &bwo,
                           float &cto, float &ho, float &dmlto, float &fio,
                           float &flo, float &bio, float &blo, float &dliqvol,
                           float &dicevol, float &tssno, float &wf)
{
    float dzfall = 0.f, dum = 0.f;
    dzfall = prcp * RHOWATER / bifall;
    dzo = dzo + dzfall;
    wo = wo + prcp;
    bwo = (wo * RHOWATER) / dzo;
    cto = 1.9e+6f * (bwo / 920.0f);
    dum = (tkair - 273.16f) * cto * dzfall
          - (1.0f - flfall) * (blfall + bifall) * DLM * dzfall;
    ho = ho + dum;
    dmlto = wo * RHOWATER * DLM;
    if (ho >= -dmlto) {
        tssno = 273.16f;
        fio = -ho / dmlto;
        flo = 1.0f - fio;
        blo = bwo * flo;
        bio = bwo * fio;
        dliqvol = blo / RHOWATER;
        dicevol = bio / DICE;
    } else {
        fio = 1.0f;
        flo = 0.0f;
        bio = bwo;
        blo = 0.0f;
        dliqvol = 0.0f;
        dicevol = bio / DICE;
        wf = 0.0f;
        tssno = (ho + dmlto) / (cto * dzo) + 273.16f;
    }
}

// ================================================================ NEWTON
__device__ void ss_newton(float &a1, float &y, float finc, int &nox,
                          int nonpos, int &iwolk, int l, float *zinc,
                          float *a2, float *y1, float *iter)
{
    // IWALK and NEX are REAL(3) locals; only element L is ever read, and it
    // is set from the arguments on entry.  ITER (the caller's ITEX) is REAL.
    const float cons = 1.0f;
    float iwalk[4] = {0.f}, nex[4] = {0.f};
    float ertol = 0.f, a = 0.f;
    ertol = 0.05f * finc;
    iwalk[l] = (float)iwolk;
    nex[l] = (float)nox;
    if (iter[l] >= 490.f) goto L160;
    if (ertol < 0.00000001f) ertol = 0.000001f;
    if (fabsf(y) <= ertol) goto L150;
    if ((fabsf(y - y1[l])) <= 0.01f * ertol && iwalk[l] == 0.f) goto L8;
    if (fabsf(y1[l]) > ertol) goto L1;
    a2[l] = a1;
    a1 = a1 - y;
    nex[l] = 0.f;
    y1[l] = y;
    iter[l] = 1.f;
    if (iwalk[l] == 3.f) goto L101;
    iwalk[l] = 0.f;
    goto L101;
L1:
    iter[l] = iter[l] + 1.f;
    if (iter[l] == 10.f) iwalk[l] = 1.f;
    if (iwalk[l] != 0.f) goto L2;
    if (fabsf(y) > ertol) goto L3;
    nex[l] = 1.f;
    goto L150;
L3:
    a = a1 - y * (a1 - a2[l]) / (y - y1[l]);
    if (fabsf(a - a1) > (10.0f * finc))
        a = a1 + 10.0f * finc * ss_sign(cons, (a - a1));
    a2[l] = a1;
    a1 = a;
    y1[l] = y;
    goto L101;
L2:
    if (iwalk[l] == 2.f) goto L4;
    if (iwalk[l] == 3.f) goto L6;
    if (ss_sign(cons, y) == ss_sign(cons, y1[l])) goto L3;
    zinc[l] = (a1 - a2[l]) / 4.0f;
    a1 = a2[l] + zinc[l];
    iwalk[l] = 2.f;
    nex[l] = 0.f;
    goto L101;
L4:
    if (ss_sign(cons, y) == ss_sign(cons, y1[l])) goto L5;
    zinc[l] = -zinc[l] / 4.0f;
    a2[l] = a1;
    a1 = a1 + zinc[l];
    nex[l] = 0.f;
    y1[l] = y;
    goto L101;
L5:
    a2[l] = a1;
    a1 = a1 + zinc[l];
    y1[l] = y;
    nex[l] = 0.f;
    goto L101;
L6:
    if (ss_sign(cons, y) == ss_sign(cons, y1[l])) goto L7;
    iwalk[l] = 1.f;
    goto L2;
L7:
    a2[l] = a1;
    a1 = a1 + finc;
    y1[l] = y;
    nex[l] = 0.f;
    goto L101;
L8:
    a1 = a1 + finc * 2.0f;
    nex[l] = 0.f;
    goto L101;
L160:
L150:
    nex[l] = 1.f;
    zinc[l] = 0.0f;
    iter[l] = 0.f;
    iwalk[l] = 0.f;
    y1[l] = 0.0f;
    y = 0.0f;
    a2[l] = 0.0f;
L101:
    if (nonpos == 1 && a1 < 0.0f) a1 = a2[l] / 2.0f;
    nox = (int)nex[l];
    iwolk = (int)iwalk[l];
}

// ================================================================ OLD
__device__ void ss_old(SsibSnow &a)
{
    for (int i = 1; i <= NLAY; ++i) {
        a.tssno[i] = a.tssn[i];
        a.bwo[i] = a.bw[i];
        a.blo[i] = a.bl[i];
        a.bio[i] = a.bi[i];
        a.ho[i] = a.h[i];
        a.flo[i] = a.fl[i];
        a.fio[i] = a.fi[i];
        a.wo[i] = a.w[i];
        a.dzo[i] = a.dz[i];
        a.sso[i] = a.ss[i];
        a.cto[i] = a.ct[i];
        a.bto[i] = a.bt[i];
        a.dmlto[i] = a.dmlt[i];
    }
}

// ================================================================ RADAB
__device__ void ss_radab(float tran[3][4][3], float ref[3][4][3],
                         const float *green, const float *vcover,
                         const float *chil, const float *zlai, float z2,
                         float z1, const float *soref, float tc, float tgs,
                         float *satcap, float extk[3][4][3],
                         float radfac[3][3][3], float &thermk, float *radt,
                         float *par, float *pd, float albedo[3][4][3],
                         float salb[3][3], float &tgeff, float sunang,
                         float xadj, const float *capac, float radn[4][3],
                         float &zlwup, float frac[3][3], int isnow,
                         float snowden, float snowdepth, float swdown,
                         float &xalbedo, float &scov2, int isice,
                         float &fsdown, float &fldown, float &fsup, float &flup)
{
    float tranc1[3] = {0.f}, tranc2[3] = {0.f}, tranc3[3] = {0.f};
    float radsav[13] = {0.f};
    float f = 0.f, xref1 = 0.f, xref2 = 0.f, fmelt = 0.f, depcov = 0.f;
    float scov = 0.f, reff1 = 0.f, reff2 = 0.f, tran1 = 0.f, tran2 = 0.f;
    float scat = 0.f, chiv = 0.f, aa = 0.f, bb = 0.f, proj = 0.f;
    float extkb = 0.f, zmew = 0.f, acss = 0.f, upscat = 0.f, betao = 0.f;
    float be = 0.f, ce = 0.f, bot = 0.f, de = 0.f, fe = 0.f, cce = 0.f;
    float ffe = 0.f, tore = 0.f, sige = 0.f, psi = 0.f, sdep = 0.f;
    float fac = 0.f, zat = 0.f, power1 = 0.f, power2 = 0.f, epsi = 0.f;
    float ek = 0.f, rosb = 0.f, rosd = 0.f, ge = 0.f, f1 = 0.f, zp = 0.f;
    float den = 0.f, alpha = 0.f, beta = 0.f, gamma = 0.f, delta = 0.f;
    float zmk = 0.f, xx = 0.f, xy = 0.f, ssum = 0.f, x0 = 0.f, x1 = 0.f;
    float x2 = 0.f, x3 = 0.f, swup = 0.f, tc4 = 0.f, tg4 = 0.f, zkat = 0.f;
    float fac1 = 0.f, fac2 = 0.f, closs = 0.f, gloss = 0.f, p1f = 0.f;
    float p2f = 0.f, swcan = 0.f, swgnd = 0.f, p1 = 0.f, p2 = 0.f;
    int iveg = 0, iwave = 0, irad = 0;
    (void)p2f; (void)swcan; (void)swgnd;
    f = ss_max(sunang, 0.01746f);
    xref1 = 1.05f;
    xref2 = 0.20f;
    fmelt = 1.f;
    if (fabsf(TF - tgs) < 0.5f) fmelt = 0.6f;
    satcap[1] = zlai[1] * 0.0001f;
    satcap[2] = zlai[2] * 0.0001f;
    if (isnow == 0) {
        depcov = ss_max(0.f, (snowdepth - z1));
    } else {
        depcov = ss_max(0.f, (capac[2] * snowden - z1));
    }
    depcov = ss_min(depcov, (z2 - z1) * 0.95f);
    satcap[1] = satcap[1] * (1.f - depcov / (z2 - z1));
    for (iveg = 1; iveg <= 2; ++iveg)
        for (iwave = 1; iwave <= 3; ++iwave)
            for (irad = 1; irad <= 2; ++irad) albedo[iveg][iwave][irad] = 0.f;
    for (iwave = 1; iwave <= 2; ++iwave) {
        for (iveg = 2; iveg >= 1; --iveg) {
            scov = 0.f;
            if (iveg == 2) goto L100;
            if (tc <= TF) scov = ss_min(0.5f, capac[1] / satcap[1]);
L100:
            reff1 = (1.f - scov) * ref[iveg][iwave][1] + scov * (xref1 -
                    (float)iwave * xref2) * fmelt;
            reff2 = (1.f - scov) * ref[iveg][iwave][2] + scov * (xref1 -
                    (float)iwave * xref2) * fmelt;
            tran1 = tran[iveg][iwave][1] * (1.f - scov)
                    + scov * (1.f - (xref1 - (float)iwave * xref2) * fmelt)
                    * tran[iveg][iwave][1];
            tran2 = tran[iveg][iwave][2] * (1.f - scov)
                    + scov * (1.f - (xref1 - (float)iwave * xref2) * fmelt) * 0.9f
                    * tran[iveg][iwave][2];
            scat = green[iveg] * (tran1 + reff1) + (1.f - green[iveg]) *
                   (tran2 + reff2);
            chiv = chil[iveg];
            if (fabsf(chiv) <= 0.01f) chiv = 0.01f;
            aa = 0.5f - 0.633f * chiv - 0.33f * chiv * chiv;
            bb = 0.877f * (1.f - 2.f * aa);
            proj = aa + bb * f;
            extkb = (aa + bb * f) / f;
            zmew = 1.f / bb * (1.f - aa / bb * ss_log((aa + bb) / aa));
            acss = scat / 2.f * proj / (proj + f * bb);
            acss = acss * (1.f - f * aa / (proj + f * bb) * ss_log((proj
                   + f * bb + f * aa) / (f * aa)));
            extk[iveg][iwave][1] = proj / f * sqrtf(1.f - scat);
            extk[iveg][iwave][2] = 1.f / zmew * sqrtf(1.f - scat);
            extk[iveg][3][1] = aa + bb;
            extk[iveg][3][2] = 1.f / zmew;
            upscat = green[iveg] * tran1 + (1.f - green[iveg]) * tran2;
            {
                float q = ((1.f - chiv) / 2.f);
                upscat = 0.5f * (scat + (scat - 2.f * upscat) * (q * q));
            }
            betao = (1.f + zmew * extkb) / (scat * zmew * extkb) * acss;
            be = 1.f - scat + upscat;
            ce = upscat;
            {
                float q = (zmew * extkb);
                bot = q * q + (ce * ce - be * be);
            }
            if (fabsf(bot) > 1.E-10f) goto L200;
            scat = scat * 0.98f;
            be = 1.f - scat + upscat;
            {
                float q = (zmew * extkb);
                bot = q * q + (ce * ce - be * be);
            }
L200:
            de = scat * zmew * extkb * betao;
            fe = scat * zmew * extkb * (1.f - betao);
            cce = de * be - zmew * de * extkb + ce * fe;
            ffe = be * fe + zmew * fe * extkb + ce * de;
            tore = -cce / bot;
            sige = -ffe / bot;
            psi = sqrtf(be * be - ce * ce) / zmew;
            if (isnow == 0) {
                sdep = snowdepth;
            } else {
                sdep = capac[2] * snowden;
            }
            fac = (sdep - z1) / (z2 - z1);
            fac = ss_max(0.f, fac);
            fac = ss_min(0.99f, fac);
            zat = zlai[iveg] / vcover[iveg];
            if (iveg == 1) zat = zat * (1.f - fac);
            power1 = ss_min(psi * zat, 50.f);
            power2 = ss_min(extkb * zat, 50.f);
            epsi = ss_exp(-power1);
            ek = ss_exp(-power2);
            rosb = soref[iwave];
            rosd = soref[iwave];
            if (iveg == 2) goto L300;
            rosb = albedo[2][iwave][1];
            rosd = albedo[2][iwave][2];
L300:
            ge = rosb / rosd;
            f1 = be - ce / rosd;
            zp = zmew * psi;
            den = (be + zp) * (f1 - zp) / epsi -
                  (be - zp) * (f1 + zp) * epsi;
            alpha = ce * (f1 - zp) / epsi / den;
            beta = -ce * (f1 + zp) * epsi / den;
            f1 = be - ce * rosd;
            den = (f1 + zp) / epsi - (f1 - zp) * epsi;
            gamma = (f1 + zp) / epsi / den;
            delta = -(f1 - zp) * epsi / den;
            albedo[iveg][iwave][2] = alpha + beta;
            if (iveg == 1) goto L400;
            scov2 = 0.f;
            if (tgs <= TF) scov2 = ss_min(1.f, capac[2] / 0.004f);
            if (isice == 1) scov2 = 1.f;
            albedo[2][iwave][2] =
                rosd * (1.f - vcover[2]) + albedo[2][iwave][2] * vcover[2];
            albedo[2][iwave][2] =
                (1.f - scov2) * albedo[2][iwave][2] + scov2 *
                (xref1 - (float)iwave * xref2) *
                fmelt;
L400:
            tranc2[iwave] = gamma * epsi + delta / epsi;
            f1 = be - ce / rosd;
            zmk = zmew * extkb;
            den = (be + zp) * (f1 - zp) / epsi -
                  (be - zp) * (f1 + zp) * epsi;
            alpha = (de - tore * (be + zmk)) * (f1 - zp) / epsi -
                    (be - zp) * (de - ce * ge - tore * (f1 + zmk)) * ek;
            alpha = alpha / den;
            beta = (be + zp) * (de - ce * ge - tore * (f1 + zmk)) * ek -
                   (de - tore * (be + zmk)) * (f1 + zp) * epsi;
            beta = beta / den;
            f1 = be - ce * rosd;
            den = (f1 + zp) / epsi - (f1 - zp) * epsi;
            gamma = -sige * (f1 + zp) / epsi -
                    (fe + ce * ge * rosd + sige * (zmk - f1)) * ek;
            gamma = gamma / den;
            delta = (ce * ge * rosd + fe + sige * (zmk - f1)) * ek
                    + sige * (f1 - zp) * epsi;
            delta = delta / den;
            albedo[iveg][iwave][1] = tore + alpha + beta;
            if (iveg == 1) goto L500;
            albedo[2][iwave][1] = rosb * (1.f - vcover[2])
                                  + albedo[2][iwave][1] * vcover[2];
            albedo[2][iwave][1] = (1.f - scov2) * albedo[2][iwave][1] +
                                  scov2 * (xref1 - (float)iwave * xref2) * fmelt;
L500:
            tranc1[iwave] = ek;
            tranc3[iwave] = sige * ek + gamma * epsi + delta / epsi;
        }
        radfac[2][iwave][1] = (1.f - vcover[1]) * (1.f - albedo[2][iwave][1])
                + vcover[1] * (tranc1[iwave] * (1.f - albedo[2][iwave][1])
                + tranc3[iwave] * (1.f - albedo[2][iwave][2]));
        radfac[2][iwave][2] = (1.f - vcover[1]) * (1.f - albedo[2][iwave][2])
                + vcover[1] * tranc2[iwave] * (1.f - albedo[2][iwave][2]);
        radfac[1][iwave][1] = vcover[1] * ((1.f - albedo[1][iwave][1])
                - tranc1[iwave] * (1.f - albedo[2][iwave][1])
                - tranc3[iwave] * (1.f - albedo[2][iwave][2]));
        radfac[1][iwave][2] = vcover[1] * ((1.f - albedo[1][iwave][2])
                - tranc2[iwave] * (1.f - albedo[2][iwave][2]));
        for (irad = 1; irad <= 2; ++irad)
            salb[iwave][irad] = (1.f - vcover[1]) * albedo[2][iwave][irad] +
                                vcover[1] * albedo[1][iwave][irad];
        if (iwave == 2) goto L600;
        radsav[1] = 1.f - vcover[1]
                    + vcover[1] * (tranc1[iwave] + tranc3[iwave]);
        radsav[2] = 1.f - vcover[1] + vcover[1] * tranc2[iwave];
L600:
        ;
    }
    if (xadj == 0.f) goto L730;
    xx = radfac[1][1][2] + radsav[2];
    xy = radfac[1][1][1] + radsav[1];
    ssum = salb[1][1] * frac[1][1] + salb[1][2] * frac[1][2] +
           salb[2][1] * frac[2][1] + salb[2][2] * frac[2][2];
    for (iwave = 1; iwave <= 2; ++iwave) {
        salb[iwave][2] = salb[iwave][2] + xadj * salb[iwave][2] / ssum;
        x0 = 1.f - salb[iwave][2];
        x1 = radfac[1][iwave][2] + radfac[2][iwave][2];
        x2 = radfac[1][iwave][2] / x1;
        x3 = radfac[2][iwave][2] / x1;
        radfac[1][iwave][2] = x0 * x2;
        radfac[2][iwave][2] = x0 * x3;
    }
    for (iwave = 1; iwave <= 2; ++iwave) {
        salb[iwave][1] = salb[iwave][1] + xadj * salb[iwave][1] / ssum;
        x0 = 1.f - salb[iwave][1];
        x1 = radfac[1][iwave][1] + radfac[2][iwave][1];
        x2 = radfac[1][iwave][1] / x1;
        x3 = radfac[2][iwave][1] / x1;
        radfac[1][iwave][1] = x0 * x2;
        radfac[2][iwave][1] = x0 * x3;
        radsav[1] = xy - radfac[1][1][1];
        radsav[2] = xx - radfac[1][1][2];
    }
L730:
    swup = radn[1][1] * salb[1][1] + radn[1][2] * salb[1][2]
         + radn[2][1] * salb[2][1] + radn[2][2] * salb[2][2];
    if ((swdown > 0.01f) && (swup > 0.01f)) {
        xalbedo = swup / swdown;
        if (xalbedo > 1.f) {
            swup = 0.f;
            xalbedo = 999.f;
        }
    } else {
        swup = 0.0f;
        xalbedo = .1f;
    }
    tc4 = tc * tc * tc * tc;
    tg4 = tgs * tgs * tgs * tgs;
    zkat = extk[1][3][2] * zlai[1] / vcover[1];
    zkat = ss_min(50.f, zkat);
    zkat = ss_max(1.E-5f, zkat);
    thermk = ss_exp(-zkat);
    fac1 = vcover[1] * (1.f - thermk);
    fac2 = 1.f;
    closs = 2.f * fac1 * STEFAN * tc4;
    closs = closs - fac2 * fac1 * STEFAN * tg4;
    gloss = fac2 * STEFAN * tg4;
    gloss = gloss - fac1 * fac2 * STEFAN * tc4;
    zlwup = fac1 * STEFAN * tc4 + (1.f - fac1) * fac2 * STEFAN * tg4;
    tgeff = sqrtf(sqrtf((zlwup / STEFAN)));
    p1f = radsav[1];
    p2f = radsav[2];
    radt[1] = 0.f;
    radt[2] = 0.f;
    for (iveg = 1; iveg <= 2; ++iveg)
        for (iwave = 1; iwave <= 2; ++iwave)
            for (irad = 1; irad <= 2; ++irad)
                radt[iveg] = radt[iveg] + radfac[iveg][iwave][irad] * radn[iwave][irad];
    fsdown = radn[1][1] + radn[1][2] + radn[2][1] + radn[2][2];
    fsup = fsdown - radt[1] - radt[2];
    radt[1] = radt[1] + radn[3][2] * vcover[1] * (1.f - thermk)
              - closs;
    radt[2] = radt[2] + radn[3][2] * (1.f - vcover[1] * (1.f - thermk))
              - gloss;
    fldown = radn[3][2];
    flup = closs + gloss;
    par[1] = radn[1][1] + radn[1][2] + 0.001f;
    pd[1] = (radn[1][1] + 0.001f) / par[1];
    p1 = p1f * radn[1][1] + 0.001f;
    p2 = radsav[2] * radn[1][2];
    par[2] = p1 + p2;
    pd[2] = p1 / par[2];
}

// ================================================================ RADAB_ICE
__device__ void ss_radab_ice(float tran[3][4][3], float ref[3][4][3],
                             const float *green, const float *vcover,
                             const float *chil, const float *zlt, float z2,
                             float z1, const float *soref, float tc, float tgs,
                             float *satcap, float extk[3][4][3], float &closs,
                             float &gloss, float &thermk, float &p1f,
                             float &p2f, float *radt, float *par, float *pd,
                             float salb[3][3], float albedo[3][4][3],
                             float &tgeff, float sunang, float xadj,
                             const float *capac, float radn[4][3], float &bedo,
                             float &zlwup, float radfrac[3][3], float swdown,
                             float &scov2, int isice, float &fsdown,
                             float &fldown, float &fsup, float &flup)
{
    float tranc1[3] = {0.f}, tranc2[3] = {0.f}, tranc3[3] = {0.f};
    float radfac[3][3][3] = {{{0.f}}};
    float radsav[13] = {0.f};
    float f = 0.f, fmelt = 0.f, depcov = 0.f;
    float scov = 0.f, reff1 = 0.f, reff2 = 0.f, tran1 = 0.f, tran2 = 0.f;
    float scat = 0.f, chiv = 0.f, aa = 0.f, bb = 0.f, proj = 0.f;
    float extkb = 0.f, zmew = 0.f, acss = 0.f, upscat = 0.f, betao = 0.f;
    float be = 0.f, ce = 0.f, bot = 0.f, de = 0.f, fe = 0.f, cce = 0.f;
    float ffe = 0.f, tore = 0.f, sige = 0.f, psi = 0.f, sdep = 0.f;
    float fac = 0.f, zat = 0.f, power1 = 0.f, power2 = 0.f, epsi = 0.f;
    float ek = 0.f, rosb = 0.f, rosd = 0.f, ge = 0.f, f1 = 0.f, zp = 0.f;
    float den = 0.f, alpha = 0.f, beta = 0.f, gamma = 0.f, delta = 0.f;
    float zmk = 0.f, xx = 0.f, xy = 0.f, ssum = 0.f, x0 = 0.f, x1 = 0.f;
    float x2 = 0.f, x3 = 0.f, sibsu = 0.f, tc4 = 0.f, tg4 = 0.f, zkat = 0.f;
    float fac1 = 0.f, fac2 = 0.f, p1 = 0.f, p2 = 0.f;
    int iveg = 0, iwave = 0, irad = 0, ivdum = 0;
    f = ss_max(sunang, 0.01746f);
    fmelt = 1.f;
    if (fabsf(TF - tgs) < 0.5f) fmelt = 0.6f;
    satcap[1] = zlt[1] * 0.0001f;
    satcap[2] = zlt[2] * 0.0001f;
    depcov = ss_max(0.f, (capac[2] * 5.f - z1));
    depcov = ss_min(depcov, (z2 - z1) * 0.95f);
    satcap[1] = satcap[1] * (1.f - depcov / (z2 - z1));
    for (iveg = 1; iveg <= 2; ++iveg)
        for (iwave = 1; iwave <= 3; ++iwave)
            for (irad = 1; irad <= 2; ++irad) albedo[iveg][iwave][irad] = 0.f;
    for (iwave = 1; iwave <= 2; ++iwave) {
        for (ivdum = 1; ivdum <= 2; ++ivdum) {
            if (ivdum == 1) iveg = 2;
            if (ivdum == 2) iveg = 1;
            scov = 0.f;
            if (iveg == 2) goto L100;
            if (tc <= TF) scov = ss_min(0.5f, capac[1] / satcap[1]);
L100:
            reff1 = (1.f - scov) * ref[iveg][iwave][1] + scov * (1.2f -
                    (float)iwave * 0.4f) * fmelt;
            reff2 = (1.f - scov) * ref[iveg][iwave][2] + scov * (1.2f -
                    (float)iwave * 0.4f) * fmelt;
            tran1 = tran[iveg][iwave][1] * (1.f - scov)
                    + scov * (1.f - (1.2f - (float)iwave * 0.4f) * fmelt)
                    * tran[iveg][iwave][1];
            tran2 = tran[iveg][iwave][2] * (1.f - scov)
                    + scov * (1.f - (1.2f - (float)iwave * 0.4f) * fmelt) * 0.9f
                    * tran[iveg][iwave][2];
            scat = green[iveg] * (tran1 + reff1) + (1.f - green[iveg]) *
                   (tran2 + reff2);
            chiv = chil[iveg];
            if (fabsf(chiv) <= 0.01f) chiv = 0.01f;
            aa = 0.5f - 0.633f * chiv - 0.33f * chiv * chiv;
            bb = 0.877f * (1.f - 2.f * aa);
            proj = aa + bb * f;
            extkb = (aa + bb * f) / f;
            zmew = 1.f / bb * (1.f - aa / bb * ss_log((aa + bb) / aa));
            acss = scat / 2.f * proj / (proj + f * bb);
            acss = acss * (1.f - f * aa / (proj + f * bb) * ss_log((proj
                   + f * bb + f * aa) / (f * aa)));
            extk[iveg][iwave][1] = proj / f * sqrtf(1.f - scat);
            extk[iveg][iwave][2] = 1.f / zmew * sqrtf(1.f - scat);
            extk[iveg][3][1] = aa + bb;
            extk[iveg][3][2] = 1.f / zmew;
            upscat = green[iveg] * tran1 + (1.f - green[iveg]) * tran2;
            {
                float q = ((1.f - chiv) / 2.f);
                upscat = 0.5f * (scat + (scat - 2.f * upscat) * (q * q));
            }
            betao = (1.f + zmew * extkb) / (scat * zmew * extkb) * acss;
            be = 1.f - scat + upscat;
            ce = upscat;
            {
                float q = (zmew * extkb);
                bot = q * q + (ce * ce - be * be);
            }
            if (fabsf(bot) > 1.E-10f) goto L200;
            scat = scat * 0.98f;
            be = 1.f - scat + upscat;
            {
                float q = (zmew * extkb);
                bot = q * q + (ce * ce - be * be);
            }
L200:
            de = scat * zmew * extkb * betao;
            fe = scat * zmew * extkb * (1.f - betao);
            cce = de * be - zmew * de * extkb + ce * fe;
            ffe = be * fe + zmew * fe * extkb + ce * de;
            tore = -cce / bot;
            sige = -ffe / bot;
            psi = sqrtf(be * be - ce * ce) / zmew;
            sdep = capac[2] * 5.f;
            fac = (sdep - z1) / (z2 - z1);
            fac = ss_max(0.f, fac);
            fac = ss_min(0.99f, fac);
            zat = zlt[iveg] / vcover[iveg];
            if (iveg == 1) zat = zat * (1.f - fac);
            power1 = ss_min(psi * zat, 50.f);
            power2 = ss_min(extkb * zat, 50.f);
            epsi = ss_exp(-power1);
            ek = ss_exp(-power2);
            rosb = soref[iwave];
            rosd = soref[iwave];
            if (iveg == 2) goto L300;
            rosb = albedo[2][iwave][1];
            rosd = albedo[2][iwave][2];
L300:
            ge = rosb / rosd;
            f1 = be - ce / rosd;
            zp = zmew * psi;
            den = (be + zp) * (f1 - zp) / epsi -
                  (be - zp) * (f1 + zp) * epsi;
            alpha = ce * (f1 - zp) / epsi / den;
            beta = -ce * (f1 + zp) * epsi / den;
            f1 = be - ce * rosd;
            den = (f1 + zp) / epsi - (f1 - zp) * epsi;
            gamma = (f1 + zp) / epsi / den;
            delta = -(f1 - zp) * epsi / den;
            albedo[iveg][iwave][2] = alpha + beta;
            if (iveg == 1) goto L400;
            scov2 = 0.f;
            if (isice == 1) scov2 = 1.f;
            if (tgs <= TF) scov2 = ss_min(1.f, capac[2] / 0.004f);
            albedo[2][iwave][2] = rosd * (1.f - vcover[2]) + albedo[2][iwave][2] * vcover[2];
            albedo[2][iwave][2] =
                (1.f - scov2) * albedo[2][iwave][2] + scov2 *
                (1.2f - (float)iwave * 0.4f) * fmelt;
L400:
            tranc2[iwave] = gamma * epsi + delta / epsi;
            f1 = be - ce / rosd;
            zmk = zmew * extkb;
            den = (be + zp) * (f1 - zp) / epsi -
                  (be - zp) * (f1 + zp) * epsi;
            alpha = (de - tore * (be + zmk)) * (f1 - zp) / epsi -
                    (be - zp) * (de - ce * ge - tore * (f1 + zmk)) * ek;
            alpha = alpha / den;
            beta = (be + zp) * (de - ce * ge - tore * (f1 + zmk)) * ek -
                   (de - tore * (be + zmk)) * (f1 + zp) * epsi;
            beta = beta / den;
            f1 = be - ce * rosd;
            den = (f1 + zp) / epsi - (f1 - zp) * epsi;
            gamma = -sige * (f1 + zp) / epsi -
                    (fe + ce * ge * rosd + sige * (zmk - f1)) * ek;
            gamma = gamma / den;
            delta = (ce * ge * rosd + fe + sige * (zmk - f1)) * ek
                    + sige * (f1 - zp) * epsi;
            delta = delta / den;
            albedo[iveg][iwave][1] = tore + alpha + beta;
            if (iveg == 1) goto L500;
            albedo[2][iwave][1] = rosb * (1.f - vcover[2])
                                  + albedo[2][iwave][1] * vcover[2];
            albedo[2][iwave][1] = (1.f - scov2) * albedo[2][iwave][1] +
                                  scov2 * (1.2f - (float)iwave * 0.4f) * fmelt;
L500:
            tranc1[iwave] = ek;
            tranc3[iwave] = sige * ek + gamma * epsi + delta / epsi;
        }
        radfac[2][iwave][1] = (1.f - vcover[1]) * (1.f - albedo[2][iwave][1])
                + vcover[1] * (tranc1[iwave] * (1.f - albedo[2][iwave][1])
                + tranc3[iwave] * (1.f - albedo[2][iwave][2]));
        radfac[2][iwave][2] = (1.f - vcover[1]) * (1.f - albedo[2][iwave][2])
                + vcover[1] * tranc2[iwave] * (1.f - albedo[2][iwave][2]);
        radfac[1][iwave][1] = vcover[1] * ((1.f - albedo[1][iwave][1])
                - tranc1[iwave] * (1.f - albedo[2][iwave][1])
                - tranc3[iwave] * (1.f - albedo[2][iwave][2]));
        radfac[1][iwave][2] = vcover[1] * ((1.f - albedo[1][iwave][2])
                - tranc2[iwave] * (1.f - albedo[2][iwave][2]));
        for (irad = 1; irad <= 2; ++irad)
            salb[iwave][irad] = (1.f - vcover[1]) * albedo[2][iwave][irad] +
                                vcover[1] * albedo[1][iwave][irad];
        if (iwave == 2) goto L600;
        radsav[1] = 1.f - vcover[1]
                    + vcover[1] * (tranc1[iwave] + tranc3[iwave]);
        radsav[2] = 1.f - vcover[1] + vcover[1] * tranc2[iwave];
L600:
        ;
    }
    if (xadj == 0.f) goto L730;
    // XADJ is hardwired 0 by the only caller (SSIB_SEAICE :1510); the
    // adjustment block (:3757-3797, whose failure exits are STOP 999) is
    // never reached.
    (void)xx; (void)xy; (void)ssum; (void)x0; (void)x1; (void)x2; (void)x3;
L730:
    sibsu = radn[1][1] * salb[1][1] + radn[1][2] * salb[1][2]
          + radn[2][1] * salb[2][1] + radn[2][2] * salb[2][2];
    if ((swdown > 0.01f) && (sibsu > 0.01f)) {
        bedo = sibsu / swdown;
        if (bedo > 1.f) {
            sibsu = 0.f;
            bedo = .1f;
        }
    } else {
        sibsu = 0.0f;
        bedo = .1f;
    }
    tc4 = tc * tc * tc * tc;
    tg4 = tgs * tgs * tgs * tgs;
    zkat = extk[1][3][2] * zlt[1] / vcover[1];
    zkat = ss_min(50.f, zkat);
    zkat = ss_max(1.E-5f, zkat);
    thermk = ss_exp(-zkat);
    fac1 = vcover[1] * (1.f - thermk);
    fac2 = 1.f;
    closs = 2.f * fac1 * STEFAN * tc4;
    closs = closs - fac2 * fac1 * STEFAN * tg4;
    gloss = fac2 * STEFAN * tg4;
    gloss = gloss - fac1 * fac2 * STEFAN * tc4;
    zlwup = fac1 * STEFAN * tc4 + (1.f - fac1) * fac2 * STEFAN * tg4;
    tgeff = sqrtf(sqrtf((zlwup / STEFAN)));
    p1f = radsav[1];
    p2f = radsav[2];
    radt[1] = 0.f;
    radt[2] = 0.f;
    for (iveg = 1; iveg <= 2; ++iveg)
        for (iwave = 1; iwave <= 2; ++iwave)
            for (irad = 1; irad <= 2; ++irad)
                radt[iveg] = radt[iveg] + radfac[iveg][iwave][irad] * radn[iwave][irad];
    fsdown = radn[1][1] + radn[1][2] + radn[2][1] + radn[2][2];
    fsup = fsdown - radt[1] - radt[2];
    radt[1] = radt[1] + radn[3][2] * vcover[1] * (1.f - thermk)
              - closs;
    radt[2] = radt[2] + radn[3][2] * (1.f - vcover[1] * (1.f - thermk))
              - gloss;
    fldown = radn[3][2];
    flup = closs + gloss;
    par[1] = radn[1][1] + radn[1][2] + 0.001f;
    pd[1] = (radn[1][1] + 0.001f) / par[1];
    p1 = p1f * radn[1][1] + 0.001f;
    p2 = p2f * radn[1][2];
    par[2] = p1 + p2;
    pd[2] = p1 / par[2];
}

// ================================================================ RASIT5
__device__ void ss_rasit5(float trib, float &ctni, float &cuni, float &ra,
                          float &z2, float z0, float &d, float zzwind,
                          float umm1, float rhoa, float tmm, float &u2,
                          float &ustar, float &drag, float ta, float bps,
                          float &rib, float &cu, float &ct, float &uv10)
{
    float g2 = 0.f, g3 = 0.f, z22 = 0.f, zl = 0.f, zwind = 0.f, tm = 0.f;
    float umm = 0.f, xct1 = 0.f, xct2 = 0.f, xctu2 = 0.f, um = 0.f;
    float ustarn = 0.f, ventn = 0.f, thm = 0.f, thvgm = 0.f, grib = 0.f;
    float grzl = 0.f, grz2 = 0.f, fvv = 0.f, ftt = 0.f, cui = 0.f, cti = 0.f;
    float rzl = 0.f, rz2 = 0.f, raf = 0.f, uest = 0.f, cuni10 = 0.f;
    (void)ventn;
    g2 = 0.75f;
    g3 = 0.75f;
    z22 = z2;
    zl = z2 + 11.785f * z0;
    zwind = zzwind;
    tm = tmm;
    umm = umm1;
    if (zwind <= d || zl <= d) d = ss_min(zwind, zl) - 0.1f;
    z2 = d + z0;
    cuni = ss_log((zwind - d) / z0) / VKC;
    if (zl < zwind) {
        xct1 = ss_log((zwind - d) / (zl - d));
        xct2 = ss_log((zl - d) / (z2 - d));
        xctu2 = ss_log((zl - d) / (z22 - d));
        ctni = (xct1 + g3 * xct2) / VKC;
    } else {
        xct2 = ss_log((zwind - d) / (z2 - d));
        xctu2 = ss_log((zwind - d) / (z22 - d));
        ctni = g3 * xct2 / VKC;
    }
    um = ss_max(umm, 2.f);
    ustarn = um / cuni;
    ventn = rhoa / ctni * ustarn;
    if (zl < zwind) {
        u2 = um - 1.f / VKC * ustarn * (xct1 + g2 * xctu2);
    } else {
        u2 = um - 1.f / VKC * ustarn * g2 * xctu2;
    }
    if (u2 < 0.01f) u2 = 0.01f;
    thm = tm * bps;
    thvgm = trib - thm;
    if (ta == 0.f) thvgm = 0.f;
    {
        float q = (um - u2);
        rib = -thvgm * GRAV * (zwind - d) / (thm * (q * q));
    }
    rib = ss_max(-10.E0f, rib);
    rib = ss_min(0.1643E0f, rib);
    // Statement functions FS(X)=66.85*X, FT(X)=0.904*X, FV(X)=0.315*X.
    if (rib < 0.0f) {
        grib = +rib;
        grzl = +rib * (zl - d) / (zwind - d);
        grz2 = +rib * (z2 - d) / (zwind - d);
        fvv = 0.315f * grib;
        if (zl < zwind) {
            ftt = 0.904f * grib + (g3 - 1.f) * (0.904f * grzl) - g3 * (0.904f * grz2);
        } else {
            ftt = g3 * ((0.904f * grib) - (0.904f * grz2));
        }
        cui = cuni + fvv;
        cti = ctni + ftt;
    } else {
        rzl = rib / (zwind - d) * (zl - d);
        rz2 = rib / (zwind - d) * (z2 - d);
        fvv = 66.85f * rib;
        if (zl < zwind) {
            ftt = 66.85f * rib + (g3 - 1.f) * (66.85f * rzl) - g3 * (66.85f * rz2);
        } else {
            ftt = g3 * ((66.85f * rib) - (66.85f * rz2));
        }
        cui = cuni + fvv;
        cti = ctni + ftt;
    }
    cu = 1.f / cui;
    ustar = um * cu;
    raf = cti / ustar;
    if (raf < 0.80f) {
        raf = 0.80f;
        cti = raf * ustar;
    }
    ct = 1.f / cti;
    ra = raf;
    uest = ustar;
    drag = rhoa * uest * uest;
    z2 = z22;
    cuni10 = ss_log((z2 + 10.f - d) / z0) / VKC;
    uv10 = ustar * (cuni10 + fvv);
}

// ================================================================ SDSOL
__device__ void ss_sdsol(float *dsol, const float *dmass, int n, float solar,
                         float &solsoil)
{
    float fext[5] = {0.f};
    const float gsize = 5.e-4f, bext = 400.0f, cv = 3.795e-3f, depth = 30.f;
    float tmass = 0.f, tsolt = 0.f, tsolb = 0.f;
    int i = 0, j = 0;
    for (i = 1; i <= n; ++i) fext[i] = 0.0f;
    tmass = 0.0f;
    for (i = 1; i <= n; ++i) {
        j = n + 1 - i;
        tmass = tmass + dmass[j];
        if (tmass > depth) goto L30;
        fext[j] = ss_exp(-cv * dmass[j] / sqrtf(gsize));
        // exp(-bext*2d-3)*fext(n): binary64, rounded on assignment
        if (j == n) fext[n] = (float)(ss_dexp(-(double)bext * 2e-3) * (double)fext[n]);
    }
L30:
    tsolt = solar;
    for (i = 1; i <= n; ++i) {
        j = n + 1 - i;
        if ((double)tsolt <= 0.0) {
            dsol[j] = 0.0f;
            tsolb = 0.0f;
        } else {
            tsolb = tsolt * fext[j];
            dsol[j] = tsolt - tsolb;
            tsolt = tsolb;
        }
    }
    solsoil = tsolb;
}

// ================================================================ SET0
__device__ void ss_set0(SsibSnow &a)
{
    for (int i = NLAY + 1; i <= N1; ++i) {
        a.tssno[i] = 0.0f; a.bwo[i] = 0.0f; a.blo[i] = 0.0f; a.bio[i] = 0.0f;
        a.ho[i] = 0.0f; a.flo[i] = 0.0f; a.fio[i] = 0.0f; a.wo[i] = 0.0f;
        a.dzo[i] = 0.0f; a.sso[i] = 0.0f; a.cto[i] = 0.0f; a.bto[i] = 0.0f;
        a.dmlto[i] = 0.0f;
    }
    for (int i = 1; i <= N1; ++i) {
        a.wf[i] = 0.0f;
        a.dhp[i] = 0.0f;
    }
}

// ================================================================ SNOW_1ST
__device__ void ss_snow_1st(float dtt, float tm, float solar, float prcpw,
                            float prcps, SsibSnow &a, float &snroff,
                            float &hroff, float &snowdepth, float &solsoil,
                            float bifall, float blfall, float flfall)
{
    float tkair = 0.f, prcp = 0.f, overburden = 0.f, dzot = 0.f, dummy = 0.f;
    int i = 0;
    tkair = tm;
    prcp = prcpw + prcps;
    snroff = 0.0f;
    hroff = 0.0f;
    if (prcpw > 0.0f) {
        a.wf[NLAY + 1] = ss_min(prcpw, DKSATSNOW * dtt);
        a.dhp[NLAY + 1] = (a.wf[NLAY + 1] / dtt) * CL * RHOWATER * (tkair - 273.16f);
        snroff = snroff + (prcpw - a.wf[NLAY + 1]);
        hroff = hroff + (prcpw - a.wf[NLAY + 1]) * CL * RHOWATER * (tkair - 273.16f);
    } else if (prcps > 0.0f) {
        a.wf[NLAY + 1] = 0.0f;
        a.dhp[NLAY + 1] = 0.0f;
        ss_newsnow(prcp, bifall, blfall, flfall, tkair,
                   a.dzo[NLAY], a.wo[NLAY], a.bwo[NLAY], a.cto[NLAY], a.ho[NLAY],
                   a.dmlto[NLAY], a.fio[NLAY], a.flo[NLAY], a.bio[NLAY],
                   a.blo[NLAY], a.dliqvol[NLAY], a.dicevol[NLAY], a.tssno[NLAY],
                   a.wf[NLAY]);
    }
    for (i = NLAY; i >= 1; --i) {
        a.dicevol[i] = a.bio[i] / DICE;
        a.dliqvol[i] = a.blo[i] / RHOWATER;
        a.porosity[i] = 1.0f - a.dicevol[i];
        a.porosity[i] = ss_min(a.porosity[i], 1.0f);
        a.porosity[i] = ss_max(a.porosity[i], 0.0f);
        a.so[i] = SSISNOW;
        if (a.porosity[i] != 0.0f) a.so[i] = a.dliqvol[i] / a.porosity[i];
        a.sso[i] = (a.so[i] - SSISNOW) / (1.0f - SSISNOW);
    }
    overburden = 0.0f;
    for (i = NLAY; i >= 1; --i) {
        overburden = overburden + a.wo[i] * RHOWATER;
        ss_compact(a.bio[i], a.tssno[i], a.blo[i], overburden, a.pdzdtc[i],
                   a.sso[i], DICE);
    }
    for (i = 1; i <= NLAY; ++i) {
        if ((a.sso[i] < 1.0f && a.porosity[i] > 0.0f)) {
            // dzo(i)*(1d0+pdzdtc(i)*dtt): the product is REAL, the sum binary64
            dzot = (float)((double)a.dzo[i] * (1.0 + (double)(a.pdzdtc[i] * dtt)));
            a.dzo[i] = ss_max(dzot, DZMIN);
            if (a.wo[i] > WOMIN) {
                a.bwo[i] = (a.wo[i] * RHOWATER) / a.dzo[i];
                if (a.bwo[i] > 920.0f) {
                    a.bwo[i] = 920.0f;
                    a.dzo[i] = (a.wo[i] * RHOWATER) / a.bwo[i];
                }
            }
            a.blo[i] = a.bwo[i] * a.flo[i];
            a.bio[i] = a.bwo[i] * a.fio[i];
            a.bto[i] = a.bwo[i];
        }
        a.dicevol[i] = a.bio[i] / DICE;
        a.dliqvol[i] = a.blo[i] / RHOWATER;
        dummy = a.dliqvol[i] + a.dicevol[i];
        if (dummy > 1.0f) {
            a.dliqvol[i] = 1.0f - a.dicevol[i];
            a.blo[i] = a.dliqvol[i] * RHOWATER;
            a.bwo[i] = a.blo[i] + a.bio[i];
            a.dzo[i] = (a.wo[i] * RHOWATER) / a.bwo[i];
        }
        a.cto[i] = (a.bwo[i] / 920.0f) * 1.9e+6f;
        a.porosity[i] = 1.0f - a.dicevol[i];
        if (a.porosity[i] > 1.0f) a.porosity[i] = 1.0f;
        if (a.porosity[i] < 0.0f) a.porosity[i] = 0.0f;
        if (a.porosity[i] > 0.0f) {
            a.so[i] = a.dliqvol[i] / a.porosity[i];
        } else {
            a.so[i] = SSISNOW;
        }
        if (a.so[i] > SSISNOW) {
            a.sso[i] = (a.so[i] - SSISNOW) / (1.0f - SSISNOW);
        } else {
            a.sso[i] = 0.0f;
        }
        a.dmass[i] = a.bto[i] * a.dzo[i];
    }
    snowdepth = a.dzo[1] + a.dzo[2] + a.dzo[3];
    if ((double)solar > 0.0) {
        ss_sdsol(a.dsol, a.dmass, NLAY, solar, solsoil);
    } else {
        for (i = 1; i <= NLAY; ++i) a.dsol[i] = 0.0f;
        solsoil = 0.0f;
    }
}

// ================================================================ SNRESULT
__device__ void ss_snresult(float dtt, int i, int icase, float &wfsoil,
                            float tsoil, float b1, float b2, float fff,
                            float *delth, float *www, const float *zdepth,
                            float poros, SsibSnow &a, float &snroff,
                            float &hroff, float &qsoil)
{
    const float bwe = 200.0f;
    float hx = 0.f, dum = 0.f, flm = 0.f, slwet = 0.f, xnodalmelt = 0.f;
    float ddz3 = 0.f;
    hx = 0.0f;
    if (icase == 1) {
        a.fi[i] = 1.0f;
        a.fl[i] = 0.0f;
        a.dz[i] = a.dzo[i];
        a.bw[i] = (a.w[i] * RHOWATER) / a.dz[i];
        if ((a.w[i] / a.dz[i]) < 0.05f || (a.w[i] / a.dz[i]) > (DICE / 1000.0f)) {
            a.bw[i] = a.bwo[i];
            a.dz[i] = (a.w[i] * RHOWATER) / a.bw[i];
        }
        a.bi[i] = a.bw[i];
        a.bl[i] = 0.0f;
        a.bt[i] = a.bw[i];
        a.wf[i] = 0.0f;
        if (i == 1) wfsoil = 0.0f;
        a.dliqvol[i] = 0.0f;
        a.dicevol[i] = a.bi[i] / DICE;
        a.ct[i] = (a.bw[i] / 920.0f) * 1.9e+6f;
        if (i == NLAY) {
            a.h[i] = a.ct[i] * a.dz[i] * (a.tssn[i] - 273.16f) - RHOWATER * DLM * a.w[NLAY] * a.fi[NLAY];
        } else {
            a.tssn[i] = (a.ho[i] + a.ct[i] * a.dz[i] * 273.16f + b1 * dtt
                         + RHOWATER * DLM * a.w[i])
                        / (a.ct[i] * a.dz[i] - b2 * dtt);
            a.h[i] = a.ho[i] + (b1 + b2 * a.tssn[i]) * dtt;
        }
        if (a.tssn[i] > 273.16f) {
            // 'Warning: Snow Temp above freezing' (a message only)
            a.tssn[i] = 273.16f;
        }
    } else if (icase == 2) {
        a.fl[i] = 1.0f - a.fi[i];
        a.tssn[i] = 273.16f;
        a.wf[i] = 0.0f;
        if (a.bwo[i] >= bwe) {
            if (a.fl[i] > FLMIN) {
                a.wf[i] = a.w[i] - (a.fi[i] / (1.0f - FLMIN)) * a.w[i];
                a.w[i] = (a.fi[i] / (1.0f - FLMIN)) * a.w[i];
                dum = a.wf[i];
                a.fl[i] = FLMIN;
                a.fi[i] = 1.0f - a.fl[i];
            }
        } else {
            flm = FLMIN + (FLMAX - FLMIN) * ((bwe - a.bwo[i]) / bwe);
            if (a.fl[i] > flm) {
                a.wf[i] = a.w[i] - (a.fi[i] / (1.0f - flm)) * a.w[i];
                a.w[i] = (a.fi[i] / (1.0f - flm)) * a.w[i];
                dum = a.wf[i];
                a.fl[i] = flm;
                a.fi[i] = 1.0f - a.fl[i];
            }
        }
        if (a.wf[i] > 0.0f) {
            if (i != 1) {
                a.wf[i] = ss_min(dum, DKSATSNOW * dtt);
                snroff = snroff + (dum - a.wf[i]);
                hroff = hroff + (dum - a.wf[i]) * CL * RHOWATER * (a.tssn[i] - 273.16f);
            } else {
                if (www[1] >= 1.0f) {
                    snroff = snroff + a.wf[i];
                    wfsoil = 0.0f;
                } else {
                    slwet = www[1] * poros * zdepth[1];
                    www[1] = (slwet + a.wf[i]) / (poros * zdepth[1]);
                    if (www[1] > 1.0f) {
                        snroff = snroff + (www[1] - 1.0f) * poros * zdepth[1];
                        www[1] = 1.0f;
                    }
                    wfsoil = 0.0f;
                }
                hroff = hroff + a.wf[i] * CL * RHOWATER * (a.tssn[i] - 273.16f);
            }
        }
        xnodalmelt = a.bio[i] * a.dzo[i] - a.w[i] * RHOWATER * a.fi[i];
        if (xnodalmelt > 0.0f && a.bio[i] * a.dzo[i] > 0.0f
            && (a.bio[i] < 250.0f || (i == NLAY && a.bio[i] < 400.0f))) {
            ddz3 = -xnodalmelt / (a.bio[i] * a.dzo[i]);
            a.dz[i] = a.dzo[i] * (1.0f + ddz3);
        } else {
            a.dz[i] = a.dzo[i];
        }
        a.bw[i] = (a.w[i] * RHOWATER) / a.dz[i];
        if ((a.w[i] / a.dz[i]) < 0.05f || (a.w[i] / a.dz[i]) > (DICE / 1000.0f)) {
            a.bw[i] = a.bwo[i];
            a.dz[i] = (a.w[i] * RHOWATER) / a.bw[i];
        }
        a.bi[i] = a.bw[i] * a.fi[i];
        a.bl[i] = a.bw[i] - a.bi[i];
        a.bt[i] = a.bw[i];
        a.ct[i] = (a.bw[i] / 920.0f) * 1.9e+6f;
        a.dliqvol[i] = a.bl[i] / RHOWATER;
        a.dicevol[i] = a.bi[i] / DICE;
        a.h[i] = (-1.0f) * a.w[i] * a.fi[i] * DLM * RHOWATER;
    } else if (icase == 3) {
        a.fl[i] = 1.0f;
        a.fi[i] = 0.0f;
        a.wf[i] = a.w[i];
        dum = a.wf[i];
        a.dz[i] = 10e-15f;
        a.w[i] = 10e-15f;
        a.bw[i] = RHOWATER;
        a.bl[i] = a.bw[i];
        a.bi[i] = 0.0f;
        a.dliqvol[i] = 1.0f;
        a.dicevol[i] = 0.0f;
        a.ct[i] = (a.bw[i] / 920.0f) * 1.9e+6f;
        a.tssn[i] = 273.16f;
        a.h[i] = 0.0f;
        if (i == NLAY) {
            if (i == 1) {
                hx = (-1.0f) * a.w[i] * fff * DLM * RHOWATER / dtt;
                snroff = a.wf[1] + snroff;
                wfsoil = 0.0f;
            } else {
                a.wf[i] = ss_min(dum, DKSATSNOW * dtt);
                snroff = snroff + (dum - a.wf[i]);
                hroff = hroff + (dum - a.wf[i]) * CL * RHOWATER * (a.tssn[i] - 273.16f);
                delth[NLAY - 1] = (-1.0f) * a.w[i] * fff * DLM * RHOWATER / dtt;
            }
        } else {
            if (i == 1) {
                hx = a.ho[i] / dtt + b1 + b2 * a.tssn[i];
                if (www[1] >= 1.0f) {
                    snroff = snroff + a.wf[i];
                    wfsoil = 0.0f;
                } else {
                    slwet = www[1] * poros * zdepth[1];
                    www[1] = (slwet + a.wf[i]) / (poros * zdepth[1]);
                    if (www[1] > 1.0f) {
                        snroff = snroff + (www[1] - 1.0f) * poros * zdepth[1];
                        www[1] = 1.0f;
                    }
                    wfsoil = 0.0f;
                }
            } else {
                a.wf[i] = ss_min(dum, DKSATSNOW * dtt);
                snroff = snroff + (dum - a.wf[i]);
                hroff = hroff + (dum - a.wf[i]) * CL * RHOWATER * (a.tssn[i] - 273.16f);
                delth[i - 1] = a.ho[i] / dtt + b1 + b2 * a.tssn[i];
            }
        }
    }
    if (i == 1) qsoil = a.qk[1] * (a.tssn[1] - tsoil) + hx;
}

// ================================================================ STRES1
__device__ void ss_stres1(int ifirst, float *rstm, const float *rootp,
                          float rstfac[3][5], float *rst, float tc, float etc,
                          float rb, float tgs, float etgs, float rd,
                          const float *tu, const float *tl, const float *topt,
                          float ea, const float *defac, const float *ph1,
                          const float *ph2, int nroot, const float *zdepth,
                          const float *phsoil, const float *rootd,
                          const float *vcover, float drop)
{
    float xdrr[4] = {0.f}, dep[4] = {0.f};
    float tv = 0.f, etv = 0.f, rair = 0.f, d2 = 0.f, d1 = 0.f, xrot = 0.f;
    float xdr = 0.f, epot = 0.f, ftpd = 0.f;
    int iveg = 0, i = 0;
    (void)rair; (void)epot;
    for (iveg = 1; iveg <= 2; ++iveg) {
        tv = tc;
        etv = etc;
        rair = rb * 2.f;
        if (iveg == 1) goto L100;
        tv = tgs;
        etv = etgs;
        rair = rd;
L100:
        tv = ss_min((tu[iveg] - 0.1f), tv);
        tv = ss_max((tl[iveg] + 0.1f), tv);
        if (ifirst == 0) goto L200;
        rstm[iveg] = rst[iveg];
        d2 = (tu[iveg] - topt[iveg]) / (topt[iveg] - tl[iveg]);
        d1 = 1.f / ((topt[iveg] - tl[iveg]) *
             ss_exp(ss_log(tu[iveg] - topt[iveg]) * d2));
        rstfac[iveg][3] = d1 * (tv - tl[iveg]) * ss_exp(ss_log(tu[iveg] - tv) * d2);
        if (rstfac[iveg][3] < 0.f) rstfac[iveg][3] = 0.f;
        if (rstfac[iveg][3] > 1.f) rstfac[iveg][3] = 1.f;
        xdrr[1] = -phsoil[1];
        xdrr[2] = -phsoil[2];
        xdrr[3] = -phsoil[3];
        if (xdrr[1] <= 0.001f) xdrr[1] = 0.001f;
        if (xdrr[2] <= 0.001f) xdrr[2] = 0.001f;
        if (xdrr[3] <= 0.001f) xdrr[3] = 0.001f;
        xdrr[1] = ss_log(xdrr[1]);
        xdrr[2] = ss_log(xdrr[2]);
        xdrr[3] = ss_log(xdrr[3]);
        if (nroot == 1) {
            xrot = rootd[1];
            for (i = 1; i <= 3; ++i) dep[i] = 0.f;
            for (i = 1; i <= 3; ++i) {
                dep[i] = ss_min(zdepth[i], xrot);
                xrot = xrot - zdepth[i];
                if (xrot <= 0.f) break;
            }
            xdr = (xdrr[1] * dep[1] + xdrr[2] * dep[2] + xdrr[3] * dep[3]) / rootd[1];
        } else {
            xdr = xdrr[1] * rootp[1] + xdrr[2] * rootp[2] + xdrr[3] * rootp[3];
        }
        rstfac[iveg][2] = 1.f - ss_exp(-ph1[iveg] * (ph2[iveg] - xdr));
        if (rstfac[iveg][2] > 1.f) rstfac[iveg][2] = 1.f;
        if (rstfac[iveg][2] < 0.f) rstfac[iveg][2] = 0.f;
L200:
        rst[iveg] = rstm[iveg];
        epot = etv - ea;
        epot = ss_max(0.0001f, (etv - ea));
        rstfac[iveg][1] = 1.f / (1.f + defac[iveg] * drop);
        if (rstfac[iveg][1] < 0.f) rstfac[iveg][1] = 0.f;
        if (rstfac[iveg][1] > 1.f) rstfac[iveg][1] = 1.f;
        ftpd = rstfac[iveg][1] * rstfac[iveg][2] * rstfac[iveg][3];
        rstfac[iveg][4] = ss_max(ftpd, 0.00001f);
        rst[iveg] = rst[iveg] / rstfac[iveg][4] / vcover[iveg];
        rst[iveg] = ss_min(rst[iveg], 100000.f);
    }
}

// Saturation vapour pressure statement functions of TEMRS1/TEMRS2.
__device__ __forceinline__ float ss_e(float x)
{
    return ss_exp(21.18123f - 5418.f / x) / .622f;
}
__device__ __forceinline__ float ss_ge(float x)
{
    return ss_exp(21.18123f - 5418.f / x) * 5418.f / (x * x) / .622f;
}

// The dummies TEMRS1 and TEMRS2 share, bundled so both read alike.  Every
// pointer is SSIB's own local array (1-based); nothing is aliased.
struct SsibTem {
    float dtt, tm, qm, em, psurf;
    float *www, *capac, *satcap, *zdepth, *vcover, *zlai, *defac, *tu, *tl,
          *topt, *rootd, *phsoil, *rootp, *ph1, *ph2, *rst, *radt;
    float (*rstfac)[5];
    float (*albedo)[4][3];
    float bee, phsat, poros, z2, thermk, rhoair, zwind, um, ccx, cg, bps;
    int nroot, isnow;
};

// ================================================================ TEMRS1
__device__ void ss_temrs1(SsibTem &p, float &tc, float &tgs, float &td,
                          float &ta, float &dtc, float &dtg, float &ra,
                          float &d, float &z0, float &rdc, float &rbc,
                          float &ect, float &eci, float &egt, float &egi,
                          float &egs, float &hc, float &hg, float &ec,
                          float &eg, float &ea, float &chf, float &shf,
                          float &zlwup, float &ustar, float &drag,
                          float snowden, float &rib, float &cu, float &xct,
                          float &flup, float &uv10)
{
    float zinc[4] = {0.f}, a2[4] = {0.f}, y1[4] = {0.f}, itex[4] = {0.f};
    float rstm[3] = {0.f};
    float etc = 0.f, etgs = 0.f, getc = 0.f, getgs = 0.f, psy = 0.f, rcp = 0.f;
    float wc = 0.f, wg = 0.f, fac = 0.f, rsoil = 0.f, psit = 0.f, argg = 0.f;
    float hr = 0.f, resd = 0.f, resz0 = 0.f, resrdc = 0.f, resrbc = 0.f;
    float resv2 = 0.f, sdep = 0.f, tgen = 0.f, tcen = 0.f, fc = 0.f, fg = 0.f;
    float trib = 0.f, ht = 0.f, ctni = 0.f, cuni = 0.f, u2 = 0.f, rb = 0.f;
    float tgta = 0.f, temdif = 0.f, fih = 0.f, rd = 0.f, d1 = 0.f, rcc = 0.f;
    float coc = 0.f, rg = 0.f, rsurf = 0.f, cog1 = 0.f, cog2 = 0.f, d2 = 0.f;
    float top = 0.f, drop = 0.f, tc3 = 0.f, tg3 = 0.f, fac1 = 0.f, fac2 = 0.f;
    float rncdtc = 0.f, rncdtg = 0.f, rngdtg = 0.f, rngdtc = 0.f, finc = 0.f;
    float hcdtc = 0.f, hcdtg = 0.f, hgdtg = 0.f, hgdtc = 0.f, hrr = 0.f;
    float deadtc = 0.f, deadtg = 0.f, ecdtc = 0.f, ecdtg = 0.f, egdtg = 0.f;
    float egdtc = 0.f, ccodtc = 0.f, ccodtg = 0.f, ccorhs = 0.f, gcodtg = 0.f;
    float gcodtc = 0.f, gcorhs = 0.f, denom = 0.f, ecpot = 0.f, ecidif = 0.f;
    float egpot = 0.f, egidif = 0.f, taen = 0.f, hend = 0.f, y = 0.f;
    float coct = 0.f, cogt = 0.f, cogs1 = 0.f, cogs2 = 0.f, egsmax = 0.f;
    float egiadd = 0.f, ecf = 0.f, egf = 0.f, dewc = 0.f, dewg = 0.f;
    int ifirst = 0, icount = 0, i = 0, nox = 0, nonpos = 0, iwalk = 0, lx = 0;
    (void)ht; (void)hend; (void)ctni; (void)cuni;
    etc = ss_e(tc);
    etgs = ss_e(tgs);
    getc = ss_ge(tc);
    getgs = ss_ge(tgs);
    psy = CPAIR / HLAT * p.psurf / 100.f / .622f;
    rcp = p.rhoair * CPAIR;
    wc = ss_min(1.f, p.capac[1] / p.satcap[1]);
    wg = ss_min(1.f, p.capac[2] / p.satcap[2]);
    fac = ss_min(p.www[1], 0.99f);
    fac = ss_max(fac, 0.02f);
    rsoil = 101840.f * (1.f - ss_pow(fac, 0.0027f));
    psit = p.phsat * ss_pow(fac, (-p.bee));
    argg = ss_max(-10.f, (psit * GRAV / 461.5f / tgs));
    hr = ss_exp(argg);
    resd = d;
    resz0 = z0;
    resrdc = rdc;
    resrbc = rbc;
    resv2 = p.vcover[2];
    if (tgs > TF) goto L100;
    sdep = p.capac[2] * snowden;
    sdep = ss_min(sdep, (p.z2 * 0.95f));
    d = p.z2 - (p.z2 - d) / p.z2 * (p.z2 - sdep);
    z0 = z0 / (p.z2 - resd) * (p.z2 - d);
    rdc = rdc * (p.z2 - sdep) / p.z2;
    rbc = rbc * p.z2 / (p.z2 - sdep);
    p.vcover[2] = 1.f;
    wg = ss_min(1.f, p.capac[2] / 0.004f);
    p.rst[2] = rsoil;
L100:
    ifirst = 1;
    icount = 0;
    tgen = tgs;
    tcen = tc;
    fc = 1.f;
    fg = 1.f;
    trib = ta;
    ea = p.em;
    ht = 0.f;
L1000:
    icount = icount + 1;
    ss_rasit5(trib, ctni, cuni, ra, p.z2, z0, d, p.zwind, p.um, p.rhoair,
              p.tm, u2, ustar, drag, ta, p.bps, rib, cu, xct, uv10);
    if (ifirst == 1) {
        rb = 1.0f / (sqrtf(u2) / rbc + p.zlai[1] * .004f);
        tgta = tgs - ta;
        temdif = (tgta + sqrtf(tgta * tgta)) / 2.f + 0.1f;
        fih = sqrtf(1.f + 9.f * GRAV * temdif * p.z2 / tgs / (u2 * u2));
        rd = rdc / u2 / fih;
    }
    d1 = 1.f / ra + 1.f / rb + 1.f / rd;
    ta = (tgs / rd + tc / rb + p.tm / ra * p.bps) / d1;
    ht = (ta - p.tm) * rcp / ra;
    rcc = p.rst[1] * fc + 2.f * rb;
    coc = (1.f - wc) / rcc + wc / (2.f * rb);
    rg = p.rst[2] * fg;
    rsurf = rsoil * fg;
    cog1 = p.vcover[2] * (1.f - wg) / (rg + rd) + (1.f - p.vcover[2]) / (rsurf + rd) * hr
           + p.vcover[2] / (rsurf + rd + 44.f) * hr;
    cog2 = p.vcover[2] * (1.f - wg) / (rg + rd) + (1.f - p.vcover[2]) / (rsurf + rd)
           + p.vcover[2] / (rsurf + rd + 44.f);
    cog1 = cog1 + wg / rd * p.vcover[2];
    cog2 = cog2 + wg / rd * p.vcover[2];
    d2 = 1.f / ra + coc + cog2;
    top = coc * etc + cog1 * etgs + p.em / ra;
    ea = top / d2;
    drop = ss_max(0.f, (ss_e(ta) - ea));
    ss_stres1(ifirst, rstm, p.rootp, p.rstfac, p.rst, tc, etc, rb, tgs, etgs,
              rd, p.tu, p.tl, p.topt, ea, p.defac, p.ph1, p.ph2, p.nroot,
              p.zdepth, p.phsoil, p.rootd, p.vcover, drop);
    ifirst = 0;
    trib = ta;
    if (icount <= 4) goto L1000;
    tc3 = tc * tc * tc;
    tg3 = tgs * tgs * tgs;
    fac1 = (1.f - p.albedo[1][3][2]) * (1.f - p.thermk) * p.vcover[1];
    fac2 = 1.f - p.albedo[2][3][2];
    rncdtc = -2.f * 4.f * fac1 * STEFAN * tc3;
    rncdtg = 4.f * fac1 * fac2 * STEFAN * tg3;
    rngdtg = -4.f * fac2 * STEFAN * tg3;
    rngdtc = 4.f * fac1 * fac2 * STEFAN * tc3;
    if (ea > etc) fc = 0.f;
    if (ea > etgs) fg = 0.f;
    i = 0;
    nox = 0;
    nonpos = 1;
    iwalk = 0;
    lx = 2;
    finc = 1.f;
    itex[lx] = 0.f;
    zinc[lx] = 0.f;
    a2[lx] = 0.f;
    y1[lx] = 0.f;
L2000:
    ss_rasit5(trib, ctni, cuni, ra, p.z2, z0, d, p.zwind, p.um, p.rhoair,
              p.tm, u2, ustar, drag, ta, p.bps, rib, cu, xct, uv10);
    rcp = p.rhoair * CPAIR;
    d1 = 1.f / ra + 1.f / rb + 1.f / rd;
    ta = (tgs / rd + tc / rb + p.tm / ra * p.bps) / d1;
    hc = rcp * (tc - ta) / rb * p.dtt;
    hg = rcp * (tgs - ta) / rd * p.dtt;
    hcdtc = rcp / rb * (1.f / ra + 1.f / rd) / d1;
    hcdtg = -rcp / (rb * rd) / d1;
    hgdtg = rcp / rd * (1.f / ra + 1.f / rb) / d1;
    hgdtc = -rcp / (rd * rb) / d1;
    hrr = hr;
    if (fg < .5f) hrr = 1.f;
    rcc = p.rst[1] * fc + 2.f * rb;
    coc = (1.f - wc) / rcc + wc / (2.f * rb);
    rg = p.rst[2] * fg;
    if (p.isnow == 0) {
        rsurf = rsoil;
    } else {
        rsurf = rsoil * fg;
    }
    cog1 = p.vcover[2] * (1.f - wg) / (rg + rd) + (1.f - p.vcover[2]) / (rsurf + rd) * hrr
           + p.vcover[2] / (rsurf + rd + 44.f) * hrr;
    cog2 = p.vcover[2] * (1.f - wg) / (rg + rd) + (1.f - p.vcover[2]) / (rsurf + rd)
           + p.vcover[2] / (rsurf + rd + 44.f);
    cog1 = cog1 + wg / rd * p.vcover[2];
    cog2 = cog2 + wg / rd * p.vcover[2];
    d2 = 1.f / ra + coc + cog2;
    top = coc * etc + cog1 * etgs + p.em / ra;
    ea = top / d2;
    ec = (etc - ea) * coc * rcp / psy * p.dtt;
    eg = (etgs * cog1 - ea * cog2) * rcp / psy * p.dtt;
    deadtc = getc * coc / d2;
    deadtg = getgs * cog1 / d2;
    ecdtc = (getc - deadtc) * coc * rcp / psy;
    ecdtg = -deadtg * coc * rcp / psy;
    egdtg = (getgs * cog1 - deadtg * cog2) * rcp / psy;
    egdtc = -deadtc * cog2 * rcp / psy;
    ccodtc = p.ccx / p.dtt - rncdtc + hcdtc + ecdtc;
    ccodtg = -rncdtg + hcdtg + ecdtg;
    ccorhs = p.radt[1] - (hc + ec) / p.dtt;
    gcodtg = p.cg / p.dtt + TIMCON * p.cg * 2.f - rngdtg + hgdtg + egdtg;
    gcodtc = -rngdtc + hgdtc + egdtc;
    gcorhs = p.radt[2] - TIMCON * p.cg * 2.f * (tgs - td) - (hg + eg) / p.dtt;
    denom = ccodtc * gcodtg - ccodtg * gcodtc;
    dtc = (ccorhs * gcodtg - ccodtg * gcorhs) / denom;
    dtg = (ccodtc * gcorhs - ccorhs * gcodtc) / denom;
    ecpot = ((etc - ea) + (getc - deadtc) * dtc - deadtg * dtg);
    eci = ecpot * wc / (2.f * rb) * rcp / psy * p.dtt;
    ecidif = ss_max(0.0f, (eci - p.capac[1] * 1.E3f * HLAT));
    eci = ss_min(eci, (p.capac[1] * 1.E3f * HLAT));
    egpot = ((etgs - ea) + (getgs - deadtg) * dtg - deadtc * dtc);
    egi = egpot * p.vcover[2] * wg / rd * rcp / psy * p.dtt;
    egidif = ss_max(0.0f, (egi - p.capac[2] * 1.E3f * HLAT));
    egi = ss_min(egi, (p.capac[2] * 1.E3f * HLAT));
    tgen = tgs + dtg;
    tcen = tc + dtc;
    d1 = 1.f / ra + 1.f / rb + 1.f / rd;
    taen = (tgen / rd + tcen / rb + p.tm / ra * p.bps) / d1;
    hend = (taen - p.tm) * rcp / ra + (ecidif + egidif) / p.dtt;
    y = trib - taen;
    i = i + 1;
    ht = hend;
    if (i > 20) goto L200;
    ss_newton(trib, y, finc, nox, nonpos, iwalk, lx, zinc, a2, y1, itex);
    if (nox != 1) goto L2000;
L200:
    hrr = hr;
    if (fg < .5f) hrr = 1.f;
    rsurf = rsoil * fg;
    coct = (1.f - wc) / rcc;
    cogt = p.vcover[2] * (1.f - wg) / (rg + rd);
    cogs1 = (1.f - p.vcover[2]) / (rd + rsurf) * hrr
            + p.vcover[2] / (rd + rsurf + 44.f) * hrr;
    cogs2 = cogs1 / hrr;
    ect = ecpot * coct * rcp / psy * p.dtt;
    egt = egpot * cogt * rcp / psy * p.dtt;
    egs = (etgs + getgs * dtg) * cogs1
          - (ea + deadtg * dtg + deadtc * dtc) * cogs2;
    egs = egs * rcp / psy * p.dtt;
    egsmax = p.www[1] / 2.f * p.zdepth[1] * p.poros * HLAT * 1000.f;
    egiadd = ss_max(0.f, egs - egsmax);
    egs = ss_min(egs, egsmax);
    egidif = egidif + egiadd;
    hc = hc + (hcdtc * dtc + hcdtg * dtg) * p.dtt + ecidif;
    hg = hg + (hgdtc * dtc + hgdtg * dtg) * p.dtt + egidif;
    ecf = ss_sign(1.f, ecpot);
    egf = ss_sign(1.f, egpot);
    dewc = fc * 2.f - 1.f;
    dewg = fg * 2.f - 1.f;
    if (dewc * ecf > 0.0f) goto L300;
    hc = hc + eci + ect;
    eci = 0.f;
    ect = 0.f;
L300:
    if (dewg * egf > 0.0f) goto L400;
    hg = hg + egs + egi + egt;
    egs = 0.f;
    egi = 0.f;
    egt = 0.f;
L400:
    ec = eci + ect;
    eg = egt + egs + egi;
    tc = tcen;
    tgs = tgen;
    ta = taen;
    ea = ea + deadtc * dtc + deadtg * dtg;
    p.radt[1] = p.radt[1] + rncdtc * dtc + rncdtg * dtg;
    p.radt[2] = p.radt[2] + rngdtc * dtc + rngdtg * dtg;
    flup = flup - (rncdtc + rngdtc) * dtc - (rncdtg + rngdtg) * dtg;
    chf = p.ccx / p.dtt * dtc;
    shf = p.cg / p.dtt * dtg + TIMCON * p.cg * 2.f * (tgs - td);
    zlwup = zlwup - rncdtc * dtc / 2.f
            - rngdtg * dtg * (1.f - p.vcover[1] * (1.f - p.thermk));
    if (tgs > TF) goto L500;
    egs = eg - egi;
    egt = 0.f;
L500:
    p.vcover[2] = resv2;
    d = resd;
    z0 = resz0;
    rdc = resrdc;
    rbc = resrbc;
}

// ================================================================ TPROPTY
__device__ void ss_tpropty(float thksoil, SsibSnow &a, float tkair,
                           float dzsoil)
{
    (void)tkair;
    for (int i = 1; i <= NLAY; ++i) {
        const float thkice = 2.290f;     // 2.290d0 assigned to a REAL
        const float thkair = 2.30e-2f;   // 2.30d-2 assigned to a REAL
        // thkair+(7.75d-5*bwo+1.105d-6*bwo*bwo)*(thkice-thkair)+0.1:
        // binary64 from the first d-literal on (thkice-thkair is REAL),
        // rounded on assignment.
        double bw = (double)a.bwo[i];
        double t = 7.75e-5 * bw + 1.105e-6 * bw * bw;
        t = t * (double)(thkice - thkair);
        a.thk[i] = (float)((double)thkair + t + (double)0.1f);
    }
    for (int i = 2; i <= NLAY; ++i)
        a.qk[i] = 2.0f * a.thk[i] * a.thk[i - 1] /
                  (a.thk[i] * a.dzo[i - 1] + a.thk[i - 1] * a.dzo[i]);
    a.qk[1] = 2.0f * a.thk[1] * thksoil / (a.thk[1] * dzsoil + thksoil * a.dzo[1]);
}

// ================================================================ TEMRS2
// Called only with ISNOW = 0 (SSIB :1230-1257); the ISNOW /= 0 arms are
// transcribed as written all the same.
__device__ void ss_temrs2(SsibTem &p, float &tc, float &tgs, float &td,
                          float &ta, float &dtc, float &dtg, float &ra,
                          float &d, float &z0, float &rdc, float &rbc,
                          float &ect, float &eci, float &egt, float &egi,
                          float &egs, float &hc, float &hg, float &ec,
                          float &eg, float &ea, float &chf, float &shf,
                          float &zlwup, float &ustar, float &drag,
                          float chisl, float tsoil, float solsoil,
                          float csoil, float &wfsoil, SsibSnow &a, float &swe,
                          float &snowden, float &snowdepth, float tkair,
                          float &snroff, float dzsoil, float &rib, float &cu,
                          float &xct, float &flup, float &uv10, int &err)
{
    // DELTH carries DATA (so SAVE) in WRF, but every element a call sets is
    // consumed and zeroed later in the same call (SNRESULT sets DELTH(k) at
    // layer k+1; layer k reads and zeroes it), so it is zero on every entry.
    float work[5] = {0.f}, work1[5] = {0.f}, delth[21] = {0.f};
    float zinc[4] = {0.f}, a2[4] = {0.f}, y1[4] = {0.f}, itex[4] = {0.f};
    float rstm[3] = {0.f};
    float etc = 0.f, etgs = 0.f, getc = 0.f, getgs = 0.f, psy = 0.f, rcp = 0.f;
    float wc = 0.f, wg = 0.f, fac = 0.f, rsoil = 0.f, psit = 0.f, argg = 0.f;
    float hr = 0.f, resd = 0.f, resz0 = 0.f, resrdc = 0.f, resrbc = 0.f;
    float resv2 = 0.f, sdep = 0.f, tgen = 0.f, tcen = 0.f, fc = 0.f, fg = 0.f;
    float trib = 0.f, ht = 0.f, ctni = 0.f, cuni = 0.f, u2 = 0.f, rb = 0.f;
    float tgta = 0.f, temdif = 0.f, fih = 0.f, rd = 0.f, d1 = 0.f, rcc = 0.f;
    float coc = 0.f, rg = 0.f, rsurf = 0.f, cog1 = 0.f, cog2 = 0.f, d2 = 0.f;
    float top = 0.f, drop = 0.f, tc3 = 0.f, tg3 = 0.f, fac1 = 0.f, fac2 = 0.f;
    float rncdtc = 0.f, rncdtg = 0.f, rngdtg = 0.f, rngdtc = 0.f, finc = 0.f;
    float hx = 0.f, raddwn = 0.f, rng = 0.f, udum0 = 0.f, uuu = 0.f;
    float hroff = 0.f, b1 = 0.f, b2 = 0.f, fff = 0.f;
    float hcdtc = 0.f, hcdtg = 0.f, hgdtg = 0.f, hgdtc = 0.f, hrr = 0.f;
    float deadtc = 0.f, deadtg = 0.f, ecdtc = 0.f, ecdtg = 0.f, egdtg = 0.f;
    float egdtc = 0.f, ccodtc = 0.f, ccodtg = 0.f, ccorhs = 0.f, gcodtg = 0.f;
    float gcodtc = 0.f, gcorhs = 0.f, gcorhs1 = 0.f, denom = 0.f;
    float tgsnew = 0.f, dh_dtt_dtg = 0.f, tonm1 = 0.f, qkn = 0.f;
    float ecpot = 0.f, ecidif = 0.f, egpot = 0.f, egidif = 0.f, taen = 0.f;
    float hend = 0.f, y = 0.f, snofac = 0.f, egidw = 0.f, dzold = 0.f;
    float coct = 0.f, cogt = 0.f, cogs1 = 0.f, cogs2 = 0.f, egsmax = 0.f;
    float egiadd = 0.f, ecf = 0.f, egf = 0.f, dewc = 0.f, dewg = 0.f;
    float tgsold = 0.f, atmp = 0.f, btmp = 0.f, ctmp = 0.f, qsoil = 0.f;
    int ifirst = 0, icount = 0, ii = 0, nox = 0, nonpos = 0, iwalk = 0;
    int lx = 0, nk = 0, ik = 0, icase = 0;
    (void)ht; (void)hend; (void)ctni; (void)cuni; (void)hx; (void)icount;
    etc = ss_e(tc);
    etgs = ss_e(tgs);
    getc = ss_ge(tc);
    getgs = ss_ge(tgs);
    psy = CPAIR / HLAT * p.psurf / 100.f / .622f;
    rcp = p.rhoair * CPAIR;
    wc = ss_min(1.f, p.capac[1] / p.satcap[1]);
    if (p.isnow == 0) {
        wg = 1.0f;
    } else {
        wg = ss_min(1.f, p.capac[2] / p.satcap[2]);
    }
    fac = ss_min(p.www[1], 0.99f);
    fac = ss_max(fac, 0.02f);
    if (p.isnow == 0) {
        rsoil = 10000000000.f;
    } else {
        rsoil = 101840.f * (1.f - ss_pow(fac, 0.0027f));
    }
    psit = p.phsat * ss_pow(fac, (-p.bee));
    argg = ss_max(-10.f, (psit * GRAV / 461.5f / tgs));
    hr = ss_exp(argg);
    resd = d;
    resz0 = z0;
    resrdc = rdc;
    resrbc = rbc;
    resv2 = p.vcover[2];
    if (tgs > TF) goto L100;
    if (p.isnow == 0) {
        sdep = snowdepth;
    } else {
        sdep = p.capac[2] * snowden;
    }
    sdep = ss_min(sdep, (p.z2 * 0.95f));
    d = p.z2 - (p.z2 - d) / p.z2 * (p.z2 - sdep);
    z0 = z0 / (p.z2 - resd) * (p.z2 - d);
    rdc = rdc * (p.z2 - sdep) / p.z2;
    rbc = rbc * p.z2 / (p.z2 - sdep);
    p.vcover[2] = 1.f;
    if (p.isnow == 0) {
        wg = 1.0f;
    } else {
        wg = ss_min(1.f, p.capac[2] / 0.004f);
    }
    p.rst[2] = rsoil;
L100:
    ifirst = 1;
    icount = 0;
    tgen = tgs;
    tcen = tc;
    fc = 1.f;
    fg = 1.f;
    trib = ta;
    ea = p.em;
    ht = 0.f;
L1000:
    icount = icount + 1;
    ss_rasit5(trib, ctni, cuni, ra, p.z2, z0, d, p.zwind, p.um, p.rhoair,
              p.tm, u2, ustar, drag, ta, p.bps, rib, cu, xct, uv10);
    if (ifirst == 1) {
        rb = 1.0f / (sqrtf(u2) / rbc + p.zlai[1] * .004f);
        tgta = tgs - ta;
        temdif = (tgta + sqrtf(tgta * tgta)) / 2.f + 0.1f;
        fih = sqrtf(1.f + 9.f * GRAV * temdif * p.z2 / tgs / (u2 * u2));
        rd = rdc / u2 / fih;
    }
    d1 = 1.f / ra + 1.f / rb + 1.f / rd;
    ta = (tgs / rd + tc / rb + p.tm / ra * p.bps) / d1;
    ht = (ta - p.tm) * rcp / ra;
    rcc = p.rst[1] * fc + 2.f * rb;
    coc = (1.f - wc) / rcc + wc / (2.f * rb);
    rg = p.rst[2] * fg;
    if (p.isnow == 0) {
        rsurf = rsoil;
    } else {
        rsurf = rsoil * fg;
    }
    cog1 = p.vcover[2] * (1.f - wg) / (rg + rd) + (1.f - p.vcover[2]) / (rsurf + rd) * hr
           + p.vcover[2] / (rsurf + rd + 44.f) * hr;
    cog2 = p.vcover[2] * (1.f - wg) / (rg + rd) + (1.f - p.vcover[2]) / (rsurf + rd)
           + p.vcover[2] / (rsurf + rd + 44.f);
    cog1 = cog1 + wg / rd * p.vcover[2];
    cog2 = cog2 + wg / rd * p.vcover[2];
    d2 = 1.f / ra + coc + cog2;
    top = coc * etc + cog1 * etgs + p.em / ra;
    ea = top / d2;
    drop = ss_max(0.f, (ss_e(ta) - ea));
    ss_stres1(ifirst, rstm, p.rootp, p.rstfac, p.rst, tc, etc, rb, tgs, etgs,
              rd, p.tu, p.tl, p.topt, ea, p.defac, p.ph1, p.ph2, p.nroot,
              p.zdepth, p.phsoil, p.rootd, p.vcover, drop);
    ifirst = 0;
    trib = ta;
    if (icount <= 4) goto L1000;
    tc3 = tc * tc * tc;
    tg3 = tgs * tgs * tgs;
    fac1 = (1.f - p.albedo[1][3][2]) * (1.f - p.thermk) * p.vcover[1];
    fac2 = 1.f - p.albedo[2][3][2];
    rncdtc = -2.f * 4.f * fac1 * STEFAN * tc3;
    rncdtg = 4.f * fac1 * fac2 * STEFAN * tg3;
    rngdtg = -4.f * fac2 * STEFAN * tg3;
    rngdtc = 4.f * fac1 * fac2 * STEFAN * tc3;
    if (ea > etc) fc = 0.f;
    if (ea > etgs) fg = 0.f;
    ii = 0;
    nox = 0;
    nonpos = 1;
    iwalk = 0;
    lx = 2;
    finc = 1.f;
    itex[lx] = 0.f;
    zinc[lx] = 0.f;
    a2[lx] = 0.f;
    y1[lx] = 0.f;
    if (p.isnow == 0) {
        ss_tpropty(chisl, a, tkair, dzsoil);
        a.tssn[NLAY + 1] = tkair;
        icount = 0;
        for (int i = 1; i <= NLAY; ++i) {
            work[i] = a.tssno[i];
            work1[i] = a.dliqvol[i];
        }
        hx = 0.0f;
        nk = NLAY;
    } else {
        nk = 1;
    }
    raddwn = solsoil;
    raddwn = raddwn + a.dsol[1] + a.dsol[2];
    rng = p.radt[2] - raddwn;
    p.radt[2] = rng;
    for (ik = nk; ik >= 1; --ik) {
        if (p.isnow != 0) goto L2000;
        if ((a.sso[ik] < 1.0f && a.porosity[ik] > 0.0f)) {
            udum0 = a.dzo[ik] * (a.porosity[ik] - work1[ik]);
            if (udum0 < 0.0f) {
                // ' udum0 is WRONG in thermal.f' then STOP: WRF ends the run.
                err |= SSIB_ERR_UDUM0;
            }
            if (a.wf[ik + 1] > udum0) {
                uuu = udum0;
                snroff = snroff + (a.wf[ik + 1] - udum0);
                hroff = hroff + (a.wf[ik + 1] - udum0) * CL * RHOWATER * (a.tssn[ik + 1] - 273.16f);
            } else {
                uuu = a.wf[ik + 1];
            }
            a.dhp[ik + 1] = (uuu * CL * RHOWATER * (a.tssn[ik + 1] - 273.16f)) / p.dtt;
            a.w[ik] = a.wo[ik] + uuu;
            a.bwo[ik] = RHOWATER * a.w[ik] / a.dzo[ik];
            a.cto[ik] = (a.bwo[ik] / 920.0f) * 1.9e+6f;
            a.dmlto[ik] = a.w[ik] * DLM * RHOWATER;
            if (a.ho[ik] < -a.dmlto[ik]) {
                a.fio[ik] = 1.0f;
                a.flo[ik] = 0.0f;
                a.tssno[ik] = (a.ho[ik] + a.dmlto[ik]) / (a.cto[ik] * a.dzo[ik]) + 273.16f;
            } else {
                a.tssno[ik] = 273.16f;
                a.fio[ik] = -a.ho[ik] / a.dmlto[ik];
                a.flo[ik] = 1.0f - a.fio[ik];
            }
            a.blo[ik] = a.bwo[ik] * a.flo[ik];
            a.bio[ik] = a.bwo[ik] * a.fio[ik];
            a.dliqvol[ik] = a.blo[ik] / RHOWATER;
            a.dicevol[ik] = a.bio[ik] / DICE;
        } else {
            a.w[ik] = a.wo[ik];
            snroff = snroff + a.wf[ik + 1];
            hroff = hroff + a.wf[ik + 1] * CL * RHOWATER * (a.tssn[ik + 1] - 273.16f);
            a.dhp[ik + 1] = 0.0f;
        }
        tgs = a.tssno[nk];
        if (ik < nk) {
            if (ik != 1) {
                b1 = a.dsol[ik] + delth[ik]
                     + a.qk[ik + 1] * (a.tssn[ik + 1] - a.tssno[ik]) + a.qk[ik] * work[ik - 1];
            } else {
                b1 = a.dsol[ik] + delth[ik]
                     + a.qk[ik + 1] * (a.tssn[ik + 1] - a.tssno[ik]) + a.qk[ik] * tsoil;
            }
            b2 = -a.qk[ik];
            delth[ik] = 0.0f;
        }
        a.dmlt[ik] = a.w[ik] * DLM * RHOWATER;
        if (ik < nk && ik >= 1) {
            fff = -(a.ho[ik] + (b1 + b2 * 273.16f) * p.dtt)
                  / (RHOWATER * DLM * a.w[ik]);
            if (fff > 0.0f && fff <= 1.0f) {
                icase = 2;
                a.fi[ik] = fff;
            } else if (fff > 1.0f) {
                icase = 1;
            } else if (fff <= 0.0f) {
                icase = 3;
            }
        }
        if (ik < nk) goto L3000;
L2000:
        ss_rasit5(trib, ctni, cuni, ra, p.z2, z0, d, p.zwind, p.um, p.rhoair,
                  p.tm, u2, ustar, drag, ta, p.bps, rib, cu, xct, uv10);
        rcp = p.rhoair * CPAIR;
        d1 = 1.f / ra + 1.f / rb + 1.f / rd;
        ta = (tgs / rd + tc / rb + p.tm / ra * p.bps) / d1;
        hc = rcp * (tc - ta) / rb * p.dtt;
        hg = rcp * (tgs - ta) / rd * p.dtt;
        hcdtc = rcp / rb * (1.f / ra + 1.f / rd) / d1;
        hcdtg = -rcp / (rb * rd) / d1;
        hgdtg = rcp / rd * (1.f / ra + 1.f / rb) / d1;
        hgdtc = -rcp / (rd * rb) / d1;
        hrr = hr;
        if (fg < .5f) hrr = 1.f;
        rcc = p.rst[1] * fc + 2.f * rb;
        coc = (1.f - wc) / rcc + wc / (2.f * rb);
        rg = p.rst[2] * fg;
        if (p.isnow == 0) {
            rsurf = rsoil;
        } else {
            rsurf = rsoil * fg;
        }
        cog1 = p.vcover[2] * (1.f - wg) / (rg + rd) + (1.f - p.vcover[2]) / (rsurf + rd) * hrr
               + p.vcover[2] / (rsurf + rd + 44.f) * hrr;
        cog2 = p.vcover[2] * (1.f - wg) / (rg + rd) + (1.f - p.vcover[2]) / (rsurf + rd)
               + p.vcover[2] / (rsurf + rd + 44.f);
        cog1 = cog1 + wg / rd * p.vcover[2];
        cog2 = cog2 + wg / rd * p.vcover[2];
        d2 = 1.f / ra + coc + cog2;
        top = coc * etc + cog1 * etgs + p.em / ra;
        ea = top / d2;
        ec = (etc - ea) * coc * rcp / psy * p.dtt;
        eg = (etgs * cog1 - ea * cog2) * rcp / psy * p.dtt;
        deadtc = getc * coc / d2;
        deadtg = getgs * cog1 / d2;
        ecdtc = (getc - deadtc) * coc * rcp / psy;
        ecdtg = -deadtg * coc * rcp / psy;
        egdtg = (getgs * cog1 - deadtg * cog2) * rcp / psy;
        egdtc = -deadtc * cog2 * rcp / psy;
        ccodtc = p.ccx / p.dtt - rncdtc + hcdtc + ecdtc;
        ccodtg = -rncdtg + hcdtg + ecdtg;
        ccorhs = p.radt[1] - (hc + ec) / p.dtt;
        if (p.isnow == 0) {
            gcodtg = a.cto[NLAY] * a.dzo[NLAY] / p.dtt - rngdtg + hgdtg + egdtg + a.qk[NLAY];
        } else {
            gcodtg = p.cg / p.dtt + TIMCON * p.cg * 2.f - rngdtg + hgdtg + egdtg;
        }
        gcodtc = -rngdtc + hgdtc + egdtc;
        if (p.isnow != 0) {
            gcorhs = p.radt[2] - TIMCON * p.cg * 2.f * (tgs - td) - (hg + eg) / p.dtt;
        } else {
            a.fi[NLAY] = 1.0f;
            gcorhs1 = a.ho[NLAY] / p.dtt + rng - (hg + eg) / p.dtt + a.dhp[NLAY + 1]
                      - a.qk[NLAY] * (tgs - a.tssno[NLAY - 1])
                      - a.cto[NLAY] * a.dzo[NLAY] * (a.tssno[NLAY] - 273.16f) / p.dtt;
            gcorhs = gcorhs1 + RHOWATER * DLM * a.w[NLAY] * a.fi[NLAY] / p.dtt;
        }
        denom = ccodtc * gcodtg - ccodtg * gcodtc;
        dtc = (ccorhs * gcodtg - ccodtg * gcorhs) / denom;
        dtg = (ccodtc * gcorhs - ccorhs * gcodtc) / denom;
        if (p.isnow == 0) {
            if ((tgs + dtg) <= 273.16f) {
                tgsnew = (tgs + dtg);
                icase = 1;
                a.h[nk] = (tgsnew - 273.16f) * a.cto[NLAY] * a.dzo[NLAY] - a.fi[nk] * a.w[nk] * DLM * RHOWATER;
                dh_dtt_dtg = (a.h[nk] - a.ho[nk]) / p.dtt;
                tonm1 = a.tssno[nk - 1];
                qkn = a.qk[NLAY];
            } else {
                dtg = 273.16f - tgs;
                dtc = (ccorhs - ccodtg * dtg) / ccodtc;
                a.fi[nk] = (gcodtc * dtc + gcodtg * dtg - gcorhs1) / (RHOWATER * DLM * a.w[NLAY]) * p.dtt;
                if (a.fi[nk] >= 0.0f && a.fi[nk] <= 1.0f) {
                    a.h[nk] = -a.fi[NLAY] * a.w[NLAY] * DLM * RHOWATER;
                    dh_dtt_dtg = (a.h[nk] - a.ho[nk]) / p.dtt;
                    icase = 2;
                    tonm1 = a.tssno[nk - 1];
                    qkn = a.qk[NLAY];
                } else if (a.fi[nk] < 0.f) {
                    a.h[nk] = -a.fi[nk] * a.w[nk] * DLM * RHOWATER;
                    dh_dtt_dtg = (a.h[nk] - a.ho[nk]) / p.dtt;
                    icase = 3;
                    tonm1 = a.tssno[nk - 1];
                    qkn = a.qk[NLAY];
                    fff = a.fi[nk];
                    a.fi[nk] = 0.0f;
                } else {
                    // fi > 1 (or NaN): no branch sets ICASE, H(NK),
                    // Dh_DTT_DTG, tonm1 or qkn; WRF reads whatever the stack
                    // held.  Recorded, and left at the zero-initialised (or
                    // last) values the oracle's -finit-local-zero gives.
                    err |= SSIB_ERR_ICASE;
                }
            }
        }
        ecpot = ((etc - ea) + (getc - deadtc) * dtc - deadtg * dtg);
        eci = ecpot * wc / (2.f * rb) * rcp / psy * p.dtt;
        ecidif = ss_max(0.0f, (eci - p.capac[1] * 1.E3f * HLAT));
        eci = ss_min(eci, (p.capac[1] * 1.E3f * HLAT));
        egpot = ((etgs - ea) + (getgs - deadtg) * dtg - deadtc * dtc);
        egi = egpot * p.vcover[2] * wg / rd * rcp / psy * p.dtt;
        egidif = ss_max(0.0f, (egi - p.capac[2] * 1.E3f * HLAT));
        egi = ss_min(egi, (p.capac[2] * 1.E3f * HLAT));
        tgen = tgs + dtg;
        tcen = tc + dtc;
        d1 = 1.f / ra + 1.f / rb + 1.f / rd;
        taen = (tgen / rd + tcen / rb + p.tm / ra * p.bps) / d1;
        hend = (taen - p.tm) * rcp / ra + (ecidif + egidif) / p.dtt;
        y = trib - taen;
        ii = ii + 1;
        ht = hend;
        if (ii > 20) goto L200;
        ss_newton(trib, y, finc, nox, nonpos, iwalk, lx, zinc, a2, y1, itex);
        if (nox != 1) goto L2000;
L200:
L3000:
        if (p.isnow == 0) {
            if (icase == 1 && ik == nk) {
                a.tssn[nk] = tgs + dtg;
            }
            if (ik == nk) {
                snofac = HLAT / (HLAT + SNOMEL / 1000.f);
                egidw = egi * snofac / HLAT / 1000.f;
                a.w[NLAY] = a.w[NLAY] - egidw;
                swe = swe - egidw;
                dzold = a.dzo[NLAY];
                a.dzo[NLAY] = a.dzo[NLAY] - egidw * RHOWATER / a.bwo[NLAY];
                a.ho[NLAY] = a.ho[NLAY] * a.dzo[NLAY] / dzold;
                p.capac[2] = swe;
            }
            ss_snresult(p.dtt, ik, icase, wfsoil, tsoil, b1, b2, fff, delth,
                        p.www, p.zdepth, p.poros, a, snroff, hroff, qsoil);
        }
    }
    snowdepth = a.dzo[1] + a.dzo[2] + a.dzo[3];
    if (p.isnow == 0) {
        swe = a.w[1] + a.w[2] + a.w[3];
        p.capac[2] = swe;
        if ((a.dz[1] + a.dz[2] + a.dz[3]) != 0.0f) {
            snowden = (a.bw[1] * a.dz[1] + a.bw[2] * a.dz[2] + a.bw[3] * a.dz[3])
                      / (a.dz[1] + a.dz[2] + a.dz[3]);
            snowden = 1000.f / snowden;
        }
    }
    hrr = hr;
    if (fg < .5f) hrr = 1.f;
    if (p.isnow == 0) {
        rsurf = rsoil;
    } else {
        rsurf = rsoil * fg;
    }
    coct = (1.f - wc) / rcc;
    cogt = p.vcover[2] * (1.f - wg) / (rg + rd);
    cogs1 = (1.f - p.vcover[2]) / (rd + rsurf) * hrr
            + p.vcover[2] / (rd + rsurf + 44.f) * hrr;
    cogs2 = cogs1 / hrr;
    ect = ecpot * coct * rcp / psy * p.dtt;
    egt = egpot * cogt * rcp / psy * p.dtt;
    egs = (etgs + getgs * dtg) * cogs1
          - (ea + deadtg * dtg + deadtc * dtc) * cogs2;
    egs = egs * rcp / psy * p.dtt;
    if (p.isnow == 0) egs = 0.0f;
    egsmax = p.www[1] / 2.f * p.zdepth[1] * p.poros * HLAT * 1000.f;
    egiadd = ss_max(0.f, egs - egsmax);
    egs = ss_min(egs, egsmax);
    egidif = egidif + egiadd;
    hc = hc + (hcdtc * dtc + hcdtg * dtg) * p.dtt + ecidif;
    hg = hg + (hgdtc * dtc + hgdtg * dtg) * p.dtt + egidif;
    if (p.isnow == 0) goto L401;
    ecf = ss_sign(1.f, ecpot);
    egf = ss_sign(1.f, egpot);
    dewc = fc * 2.f - 1.f;
    dewg = fg * 2.f - 1.f;
    if (dewc * ecf > 0.0f) goto L300;
    hc = hc + eci + ect;
    eci = 0.f;
    ect = 0.f;
L300:
    if (dewg * egf > 0.0f) goto L400;
    hg = hg + egs + egi + egt;
    egs = 0.f;
    egi = 0.f;
    egt = 0.f;
L400:
L401:
    ec = eci + ect;
    eg = egt + egs + egi;
    tgsold = tgs;
    tc = tcen;
    tgs = tgen;
    if (p.isnow == 0) a.tssn[NLAY] = tgs;
    ta = taen;
    ea = ea + deadtc * dtc + deadtg * dtg;
    p.radt[1] = p.radt[1] + rncdtc * dtc + rncdtg * dtg;
    p.radt[2] = p.radt[2] + rngdtc * dtc + rngdtg * dtg;
    flup = flup - (rncdtc + rngdtc) * dtc - (rncdtg + rngdtg) * dtg;
    chf = p.ccx / p.dtt * dtc;
    if (p.isnow == 0) {
        shf = dh_dtt_dtg - a.dhp[NLAY + 1] + qkn * (tgsold - tonm1)
              + qkn * dtg;
    } else {
        shf = p.cg / p.dtt * dtg + TIMCON * p.cg * 2.f * (tgs - td);
    }
    zlwup = zlwup - rncdtc * dtc / 2.f
            - rngdtg * dtg * (1.f - p.vcover[1] * (1.f - p.thermk));
    if (tgs > TF) goto L500;
    egs = eg - egi;
    egt = 0.f;
L500:
    p.vcover[2] = resv2;
    d = resd;
    z0 = resz0;
    rdc = resrdc;
    rbc = resrbc;
    if (p.isnow == 0) {
        tgs = tsoil;
        atmp = (qsoil + solsoil) / csoil;
        btmp = __fdiv_rn(__fmul_rn(2.f, 3.1416f), 86400.f);
        // (365.*3.1416)**0.5 is a compile-time constant in WRF: the
        // correctly rounded square root of the rounded product.
        ctmp = csoil * btmp / csoil / __fsqrt_rn(__fmul_rn(365.f, 3.1416f));
        tgs = (tsoil + atmp * p.dtt + btmp * p.dtt * td / (1.f + ctmp * p.dtt)) /
              (1.f + btmp * p.dtt * (1.f - ctmp * p.dtt / (1.f + ctmp * p.dtt)));
        td = (ctmp * p.dtt * tgs + td) / (1.f + ctmp * p.dtt);
    }
}

// ================================================================ RUN2
// The soil-water redistribution block UPDAT1 (:5894-6058) and UPDAT1_ICE
// (:6273-6444) share; ice=1 selects UPDAT1_ICE's two differences (the
// PILPS accumulators, which feed no output, and the operand order of the
// WWW(3) > 1 runoff term).
__device__ void ss_run2(float dtt, float td, float tgs, float poros,
                        float bee, float satco, float slope, float phsat,
                        const float *zdepth, float *www, float &roff,
                        float snoww2, int ice)
{
    float temw[4] = {0.f}, temwp[4] = {0.f}, temwpp[4] = {0.f};
    float aaa[3] = {0.f}, bbb[3] = {0.f}, ccc[3] = {0.f}, qqq[3] = {0.f};
    float pows = 0.f, q3g = 0.f, wmax = 0.f, pmax = 0.f, wmin = 0.f;
    float pmin = 0.f, dpdw = 0.f, rsame = 0.f, avk = 0.f, div = 0.f;
    float avkmin = 0.f, avkmax = 0.f, tsnow = 0.f, areas = 0.f, tgg = 0.f;
    float ts = 0.f, props = 0.f, dpdwdz = 0.f, denom = 0.f, rdenom = 0.f;
    float qmax = 0.f, qmin = 0.f, excess = 0.f, deficit = 0.f;
    int i = 0;
    for (i = 1; i <= 3; ++i) {
        temw[i] = ss_max(0.03f, www[i]);
        temwp[i] = ss_pow(temw[i], (-bee));
        temwpp[i] = ss_pow(ss_min(1.f, temw[i]), (2.f * bee + 3.f));
    }
    pows = 2.f * bee + 2.f;
    q3g = ss_pow(temw[3], (-pows)) + satco / zdepth[3] / poros * slope * pows * dtt;
    q3g = ss_pow(q3g, (1.f / pows));
    q3g = -(1.f / q3g - www[3]) * poros * zdepth[3] / dtt;
    q3g = ss_max(0.f, q3g);
    q3g = ss_min(q3g, www[3] * poros * zdepth[3] / dtt);
    q3g = q3g + 0.002f * poros * zdepth[3] * 0.5f / 86400.f * www[3];
    wmax = ss_max(ss_max(ss_max(www[1], www[2]), www[3]), 0.05f);
    wmax = ss_min(wmax, 1.f);
    pmax = ss_pow(wmax, (-bee));
    wmin = ss_pow((pmax - 2.f / (phsat * (zdepth[1] + 2.f * zdepth[2] + zdepth[3]))),
                  (-(1.f / bee)));
    wmin = ss_min(ss_min(ss_min(www[1], www[2]), www[3]), wmin);
    wmin = ss_max(wmin, 0.02f);
    pmin = ss_pow(wmin, (-bee));
    dpdw = phsat * (pmax - pmin) / (wmax - wmin);
    for (i = 1; i <= 2; ++i) {
        rsame = 0.f;
        avk = temwp[i] * temwpp[i] - temwp[i + 1] * temwpp[i + 1];
        div = temwp[i + 1] - temwp[i];
        if (fabsf(div) < 1.E-6f) rsame = 1.f;
        avk = satco * avk / ((1.f + 3.f / bee) * div + rsame);
        avkmin = satco * ss_min(temwpp[i], temwpp[i + 1]);
        avkmax = satco * ss_max(temwpp[i], temwpp[i + 1]) * 1.01f;
        avk = ss_max(avk, avkmin);
        avk = ss_min(avk, avkmax);
        tsnow = ss_min(TF - 0.01f, tgs);
        areas = ss_min(0.999f, 13.2f * snoww2);
        tgg = tsnow * areas + tgs * (1.f - areas);
        ts = tgg * (float)(2 - i) + td * (float)(i - 1);
        props = (ts - (TF - 10.f)) / 10.f;
        props = ss_max(0.05f, ss_min(1.0f, props));
        avk = avk * props;
        q3g = q3g * props;
        dpdwdz = dpdw * 2.f / (zdepth[i] + zdepth[i + 1]);
        aaa[i] = 1.f + avk * dpdwdz * (1.f / zdepth[i] + 1.f / zdepth[i + 1])
                 * dtt / poros;
        bbb[i] = -avk * dpdwdz * 1.f / zdepth[2] * dtt / poros;
        ccc[i] = avk * (dpdwdz * (www[i] - www[i + 1]) + 1.f +
                 (float)(i - 1) * dpdwdz * q3g * 1.f / zdepth[3] * dtt / poros);
    }
    denom = (aaa[1] * aaa[2] - bbb[1] * bbb[2]);
    rdenom = 0.f;
    if (fabsf(denom) < 1.E-6f) rdenom = 1.f;
    rdenom = (1.f - rdenom) / (denom + rdenom);
    qqq[1] = (aaa[2] * ccc[1] - bbb[1] * ccc[2]) * rdenom;
    qqq[2] = (aaa[1] * ccc[2] - bbb[2] * ccc[1]) * rdenom;
    www[3] = www[3] - q3g * dtt / (poros * zdepth[3]);
    roff = roff + q3g * dtt;
    for (i = 1; i <= 2; ++i) {
        qmax = www[i] * (poros * zdepth[i] / dtt);
        qmin = -www[i + 1] * (poros * zdepth[i + 1] / dtt);
        qqq[i] = ss_min(qqq[i], qmax);
        qqq[i] = ss_max(qqq[i], qmin);
        www[i] = www[i] - qqq[i] / (poros * zdepth[i] / dtt);
        www[i + 1] = www[i + 1] + qqq[i] / (poros * zdepth[i + 1] / dtt);
    }
    for (i = 1; i <= 3; ++i) {
        excess = ss_max(0.f, (www[i] - 1.f));
        www[i] = www[i] - excess;
        roff = roff + excess * poros * zdepth[i];
    }
    for (i = 1; i <= 2; ++i) {
        deficit = ss_max(0.f, (1.E-12f - www[i]));
        www[i] = www[i] + deficit;
        www[i + 1] = www[i + 1] - deficit * zdepth[i] / zdepth[i + 1];
    }
    www[3] = ss_max(www[3], 1.E-12f);
    if (www[1] > 1.f) {
        www[2] = www[2] + (www[1] - 1.f) * zdepth[1] / zdepth[2];
        www[1] = 1.f;
    }
    if (www[2] > 1.f) {
        www[3] = www[3] + (www[2] - 1.f) * zdepth[2] / zdepth[3];
        www[2] = 1.f;
    }
    if (www[3] > 1.f) {
        if (ice) roff = roff + (www[3] - 1.f) * poros * zdepth[3];
        else     roff = roff + (www[3] - 1.f) * zdepth[3] * poros;
        www[3] = 1.f;
    }
}

// ================================================================ UPDAT1
__device__ void ss_updat1(float dtt, float &tc, float &tgs, float &td,
                          float *capac, float dtc, float dtg, float &ect,
                          float &eci, float &egt, float &egi, float egs,
                          float eg, float hc, float hg, float &hflux,
                          float &etmass, float &roff, int nroot,
                          const float *rootd, const float *rootp, float poros,
                          float bee, float satco, float slope, float phsat,
                          const float *zdepth, float *www, float ccx, float cg,
                          float chf, float shf, int isnow, float wfsoil,
                          float swe, float &smelt)
{
    float ef[4] = {0.f}, snoww[3] = {0.f};
    float snofac = 0.f, facks = 0.f, ecmass = 0.f, egmass = 0.f, zmelt = 0.f;
    float cct = 0.f, ts = 0.f, dts = 0.f, flux = 0.f, tta = 0.f, ttb = 0.f;
    float dtf = 0.f, dtime1 = 0.f, hf = 0.f, fcap = 0.f, spwet = 0.f;
    float dtime2 = 0.f, dtf2 = 0.f, tn = 0.f, change = 0.f, fluxef = 0.f;
    float absoil = 0.f, totdep = 0.f, eft = 0.f;
    int iveg = 0, il = 0;
    (void)eg; (void)change;
    snofac = HLAT / (HLAT + SNOMEL / 1000.f);
    facks = 1.f;
    if ((tc - dtc) <= TF) facks = snofac;
    if ((ect + eci) > 0.f) goto L100;
    eci = ect + eci;
    ect = 0.f;
    facks = 1.f / facks;
L100:
    capac[1] = capac[1] - eci * facks / HLAT / 1000.f;
    ecmass = (ect + eci * facks) / HLAT;
    if (isnow == 0) facks = snofac;
    if (isnow == 0) goto L201;
    facks = 1.f;
    if ((tgs - dtg) <= TF) facks = snofac;
    if ((egt + egi) > 0.f) goto L200;
    egi = egt + egi;
    egt = 0.f;
    facks = 1.f / facks;
L200:
    capac[2] = capac[2] - egi * facks / HLAT / 1000.f;
L201:
    egmass = (egt + egs + egi * facks) / HLAT;
    etmass = ecmass + egmass;
    hflux = (hc + hg) / dtt;
    for (iveg = 1; iveg <= 2; ++iveg) {
        if (capac[iveg] > 0.000001f) continue;
        www[1] = www[1] + capac[iveg] / (poros * zdepth[1]);
        capac[iveg] = 0.f;
    }
    for (iveg = 1; iveg <= 2; ++iveg) {
        if (isnow == 0 && iveg == 2) {
            zmelt = wfsoil;
            www[1] = www[1] + zmelt / (poros * zdepth[1]);
            capac[2] = swe;
            continue;
        }
        cct = ccx;
        ts = tc;
        dts = dtc;
        flux = chf;
        if (iveg == 1) goto L110;
        cct = cg;
        ts = tgs;
        dts = dtg;
        flux = cct * dtg / dtt;
L110:
        tta = ts - dts;
        ttb = ts;
        snoww[iveg] = 0.f;
        if (tta <= TF) snoww[iveg] = capac[iveg];
        capac[iveg] = capac[iveg] - snoww[iveg];
        if (tta > TF && ttb > TF) goto L120;
        if (tta <= TF && ttb <= TF) goto L120;
        dtf = TF - tta;
        dtime1 = cct * dtf / flux;
        hf = flux * (dtt - dtime1);
        fcap = -capac[iveg] * SNOMEL;
        spwet = ss_min(5.f, snoww[iveg]);
        if (dts > 0.f) fcap = spwet * SNOMEL;
        dtime2 = fcap / flux;
        dtf2 = flux * (dtt - dtime1 - dtime2) / cct;
        tn = TF + dtf2;
        ts = TF - 0.1f;
        if (fabsf(hf) >= fabsf(fcap)) ts = tn;
        change = hf;
        if (fabsf(change) >= fabsf(fcap)) change = fcap;
        change = change / SNOMEL;
        if (change > 0.0f) smelt = change + smelt;
        snoww[iveg] = snoww[iveg] - change;
        capac[iveg] = capac[iveg] + change;
        if (iveg == 1) tc = ts;
        if (iveg == 2) tgs = ts;
        if (snoww[iveg] < 0.00001f) goto L120;
        zmelt = capac[iveg];
        www[1] = www[1] + zmelt / (poros * zdepth[1]);
        capac[iveg] = 0.f;
L120:
        capac[iveg] = capac[iveg] + snoww[iveg];
    }
    if (isnow != 0) {
        fluxef = shf - cct * dtg / dtt;
        td = td + fluxef / (cg * 2.f * __fsqrt_rn(__fmul_rn(PIE, 365.f))) * dtt;
    }
    change = 0.0f;
    www[1] = www[1] - egs / HLAT / 1000.f / (poros * zdepth[1]);
    for (iveg = 1; iveg <= 2; ++iveg) {
        if (iveg == 1) absoil = ect / HLAT / 1000.f;
        if (iveg == 2) absoil = egt / HLAT / 1000.f;
        if (nroot == 1) {
            ef[2] = 0.f;
            ef[3] = 0.f;
            totdep = zdepth[1];
            for (il = 2; il <= 3; ++il) {
                totdep = totdep + zdepth[il];
                if (rootd[iveg] < totdep) goto L400;
                ef[il] = zdepth[il] / rootd[iveg];
                continue;
L400:
                ef[il] = rootd[iveg] - totdep + zdepth[il];
                ef[il] = ef[il] / rootd[iveg];
                break;
            }
            eft = ef[2] + ef[3];
            eft = ss_max(eft, 0.1E-5f);
            ef[2] = ef[2] / eft;
            ef[3] = ef[3] / eft;
            for (il = 2; il <= 3; ++il)
                www[il] = www[il] - absoil * ef[il] / (poros * zdepth[il]);
        } else {
            ef[1] = rootp[1];
            ef[2] = rootp[2];
            ef[3] = rootp[3];
            for (il = 1; il <= 3; ++il)
                www[il] = www[il] - absoil * ef[il] / (poros * zdepth[il]);
        }
    }
    for (il = 1; il <= 2; ++il) {
        if (www[il] > 0.f) continue;
        www[il + 1] = www[il + 1] + www[il] * zdepth[il] / zdepth[il + 1];
        www[il] = 0.f;
    }
    // SNOWW(2) is never assigned under the multi-layer pack (ISNOW = 0 takes
    // the IVEG = 2 branch above); WRF reads an uninitialised local there.
    // Both arms read zero, as the oracle's -finit-local-zero does: in this
    // mode TGS is already the soil temperature under the pack.
    ss_run2(dtt, td, tgs, poros, bee, satco, slope, phsat, zdepth, www, roff,
            snoww[2], 0);
}

// ================================================================ UPDAT1_ICE
__device__ void ss_updat1_ice(float dtt, float &tc, float &tgs, float &td,
                              float *capac, float dtc, float dtg, float &ect,
                              float &eci, float &egt, float &egi, float egs,
                              float hc, float hg, float &hflux, float &etmass,
                              float &roff, int nroot, const float *rootd,
                              const float *rootp, float poros, float bee,
                              float satco, float slope, float phsat,
                              const float *zdepth, float *www, float ccx,
                              float cg, float chf, float shf, float &smelt)
{
    float ef[4] = {0.f}, snoww[3] = {0.f};
    float snofac = 0.f, facks = 0.f, ecmass = 0.f, egmass = 0.f, zmelt = 0.f;
    float cct = 0.f, ts = 0.f, dts = 0.f, flux = 0.f, tta = 0.f, ttb = 0.f;
    float dtf = 0.f, dtime1 = 0.f, hf = 0.f, fcap = 0.f, spwet = 0.f;
    float dtime2 = 0.f, dtf2 = 0.f, tn = 0.f, change = 0.f, fluxef = 0.f;
    float absoil = 0.f, totdep = 0.f, eft = 0.f;
    int iveg = 0, il = 0;
    (void)change;
    snofac = HLAT / (HLAT + SNOMEL / 1000.f);
    facks = 1.f;
    if ((tc - dtc) <= TF) facks = snofac;
    if ((ect + eci) > 0.f) goto L100;
    eci = ect + eci;
    ect = 0.f;
    facks = 1.f / facks;
L100:
    capac[1] = capac[1] - eci * facks / HLAT / 1000.f;
    ecmass = (ect + eci * facks) / HLAT;
    facks = 1.f;
    if ((tgs - dtg) <= TF) facks = snofac;
    if ((egt + egi) > 0.f) goto L200;
    egi = egt + egi;
    egt = 0.f;
    facks = 1.f / facks;
L200:
    capac[2] = capac[2] - egi * facks / HLAT / 1000.f;
    egmass = (egt + egs + egi * facks) / HLAT;
    etmass = ecmass + egmass;
    hflux = (hc + hg);
    for (iveg = 1; iveg <= 2; ++iveg) {
        if (capac[iveg] > 0.000001f) continue;
        www[1] = www[1] + capac[iveg] / (poros * zdepth[1]);
        capac[iveg] = 0.f;
    }
    for (iveg = 1; iveg <= 2; ++iveg) {
        cct = ccx;
        ts = tc;
        dts = dtc;
        flux = chf;
        if (iveg == 1) goto L7100;
        cct = cg;
        ts = tgs;
        dts = dtg;
        flux = cct * dtg / dtt;
L7100:
        tta = ts - dts;
        ttb = ts;
        snoww[iveg] = 0.f;
        if (tta <= TF) snoww[iveg] = capac[iveg];
        capac[iveg] = capac[iveg] - snoww[iveg];
        if (tta > TF && ttb > TF) goto L7200;
        if (tta <= TF && ttb <= TF) goto L7200;
        dtf = TF - tta;
        dtime1 = cct * dtf / flux;
        hf = flux * (dtt - dtime1);
        fcap = -capac[iveg] * SNOMEL;
        spwet = ss_min(5.f, snoww[iveg]);
        if (dts > 0.f) fcap = spwet * SNOMEL;
        dtime2 = fcap / flux;
        dtf2 = flux * (dtt - dtime1 - dtime2) / cct;
        tn = TF + dtf2;
        ts = TF - 0.1f;
        if (fabsf(hf) >= fabsf(fcap)) ts = tn;
        change = hf;
        if (fabsf(change) >= fabsf(fcap)) change = fcap;
        change = change / SNOMEL;
        if (change > 0.0f) smelt = change + smelt;
        snoww[iveg] = snoww[iveg] - change;
        capac[iveg] = capac[iveg] + change;
        if (iveg == 1) tc = ts;
        if (iveg == 2) tgs = ts;
        if (snoww[iveg] < 0.00001f) goto L7200;
        zmelt = 0.f;
        zmelt = capac[iveg];
        www[1] = www[1] + zmelt / (poros * zdepth[1]);
        capac[iveg] = 0.f;
L7200:
        capac[iveg] = capac[iveg] + snoww[iveg];
    }
    fluxef = shf - cct * dtg / dtt;
    td = td + fluxef / (cg * 2.f * __fsqrt_rn(__fmul_rn(PIE, 365.f))) * dtt;
    change = 0.0f;
    www[1] = www[1] - egs / HLAT / 1000.f / (poros * zdepth[1]);
    for (iveg = 1; iveg <= 2; ++iveg) {
        if (iveg == 1) absoil = ect / HLAT / 1000.f;
        if (iveg == 2) absoil = egt / HLAT / 1000.f;
        if (nroot == 1) {
            ef[2] = 0.f;
            ef[3] = 0.f;
            totdep = zdepth[1];
            for (il = 2; il <= 3; ++il) {
                totdep = totdep + zdepth[il];
                if (rootd[iveg] < totdep) goto L400;
                ef[il] = zdepth[il] / rootd[iveg];
                continue;
L400:
                ef[il] = rootd[iveg] - totdep + zdepth[il];
                ef[il] = ef[il] / rootd[iveg];
                break;
            }
            eft = ef[2] + ef[3];
            eft = ss_max(eft, 0.1E-5f);
            ef[2] = ef[2] / eft;
            ef[3] = ef[3] / eft;
            for (il = 2; il <= 3; ++il)
                www[il] = www[il] - absoil * ef[il] / (poros * zdepth[il]);
        } else {
            ef[1] = rootp[1];
            ef[2] = rootp[2];
            ef[3] = rootp[3];
            for (il = 1; il <= 3; ++il)
                www[il] = www[il] - absoil * ef[il] / (poros * zdepth[il]);
        }
    }
    for (il = 1; il <= 2; ++il) {
        if (www[il] > 0.f) continue;
        www[il + 1] = www[il + 1] + www[il] * zdepth[il] / zdepth[il + 1];
        www[il] = 0.f;
    }
    ss_run2(dtt, td, tgs, poros, bee, satco, slope, phsat, zdepth, www, roff,
            snoww[2], 1);
}

// Default-arm fixes (each a declared divergence from WRF; the host passes
// none under GPUWM_WRF_EXACT=1, all of them otherwise).
#define SSIB_FIX_SEAICE_SNOW_UNITS 1   // SSIB_SEAICE returns metres (:1732)
#define SSIB_FIX_T2_CANOPY_AIR     2   // driver overwrites T2 with TSK (:4289)
#define SSIB_FIXES_ALL             3

// One column's state, by name (field-major in global memory).
struct SsibCol {
    float s[NS_SSIB];
};
#define CV(name) c.s[S_##name]

// Radiation-split prologue shared by SSIB (:1149-1196) and SSIB_SEAICE
// (:1614-1662).
__device__ void ss_radn(int sw_physics, float sunang, float swdown1,
                        float &cloud, float &radfrac11, float &radfrac12,
                        float &radfrac21, float &radfrac22,
                        float radfrac[3][3], float &swdown, float rlwdown,
                        float radn[4][3])
{
    float difrat = 0.f, vnrat = 0.f;
    if (sw_physics == 3 || sw_physics == 4) {
        radfrac11 = ss_max(radfrac11, 0.025f);
        radfrac12 = ss_max(radfrac12, 0.025f);
        radfrac21 = ss_max(radfrac21, 0.025f);
        radfrac22 = ss_max(radfrac22, 0.025f);
        swdown = radfrac11 + radfrac12 + radfrac21 + radfrac22;
        radfrac[1][1] = radfrac11 / swdown;
        radfrac[1][2] = radfrac12 / swdown;
        radfrac[2][1] = radfrac21 / swdown;
        radfrac[2][2] = radfrac22 / swdown;
    } else {
        swdown = ss_max(swdown1, 0.1f);
        cloud = ss_max(cloud, 0.0f);
        cloud = ss_min(cloud, 1.0f);
        cloud = ss_max(0.58f, cloud);
        difrat = 0.0604f / (sunang - 0.0223f) + 0.0683f;
        if (difrat < 0.0f) difrat = 0.0f;
        if (difrat > 1.0f) difrat = 1.0f;
        difrat = difrat + (1.0f - difrat) * cloud;
        vnrat = (580.0f - cloud * 464.0f) / ((580.0f - cloud * 499.0f)
                + (580.0f - cloud * 464.0f));
        radfrac[1][1] = (1.0f - difrat) * vnrat;
        radfrac[1][2] = difrat * vnrat;
        radfrac[2][1] = (1.0f - difrat) * (1.0f - vnrat);
        radfrac[2][2] = difrat * (1.0f - vnrat);
    }
    radn[1][1] = radfrac[1][1] * swdown;
    radn[1][2] = radfrac[1][2] * swdown;
    radn[2][1] = radfrac[2][1] * swdown;
    radn[2][2] = radfrac[2][2] * swdown;
    radn[3][1] = 0.f;
    radn[3][2] = rlwdown;
}

__device__ int ss_month(float day, float zlat)
{
    const int idays[13] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366};
    int imonth = 1, iday = (int)day, mon_cor = 0;
    for (int i = 1; i <= 12; ++i) {
        if (iday <= idays[i]) {
            imonth = i;
            break;
        }
    }
    if (zlat < 0.0f) {
        mon_cor = imonth + 6;
        if (mon_cor > 12) mon_cor = mon_cor - 12;
    } else {
        mon_cor = imonth;
    }
    return mon_cor;
}

// ================================================================ SSIB
// Arguments as bound by the driver (:4262-4284); `c` holds every INOUT.
__device__ void ss_ssib(SsibCol &c, int itime, float ddtt, float zlat,
                        float sunangle, float ppl, float ppc, float rlwdown,
                        float zwind2, float umm, float vmm, float qm, float tm,
                        float pm, float psur, int ivgtyp, float swdown1,
                        float day, float &cloud, float &ta, float &uv10,
                        int sw_physics, int mminlu, int &err)
{
    float capac[3] = {0.f}, satcap[3] = {0.f}, green[3] = {0.f};
    float vcover[3] = {0.f}, zlt[3] = {0.f}, chil[3] = {0.f}, topt[3] = {0.f};
    float tl[3] = {0.f}, tu[3] = {0.f}, defac[3] = {0.f}, ph1[3] = {0.f};
    float ph2[3] = {0.f}, rst[3] = {0.f}, rootd[3] = {0.f}, radt[3] = {0.f};
    float par[3] = {0.f}, pd[3] = {0.f};
    float www[4] = {0.f}, soref[4] = {0.f}, zdepth[4] = {0.f};
    float rootp[4] = {0.f}, phsoil[4] = {0.f};
    float radfrac[3][3] = {{0.f}}, salb[3][3] = {{0.f}}, rstpar[3][4] = {{0.f}};
    float rstfac[3][5] = {{0.f}}, radn[4][3] = {{0.f}};
    float tran[3][4][3] = {{{0.f}}}, ref[3][4][3] = {{{0.f}}};
    float albedo[3][4][3] = {{{0.f}}}, extk[3][4][3] = {{{0.f}}};
    float radfac[3][3][3] = {{{0.f}}};
    SsibSnow sn = {};
    float xadj = 0.f, ctlpa = 0.f, wfsoil = 0.f, zwind = 0.f, psurf = 0.f;
    float dtt = 0.f, z0 = 0.f, xdd = 0.f, z2 = 0.f, z1 = 0.f, rdc = 0.f;
    float rbc = 0.f, bee = 0.f, phsat = 0.f, poros = 0.f, satco = 0.f;
    float slope = 0.f, stlev1 = 0.f, stlev2 = 0.f, depth = 0.f, um = 0.f;
    float rhoair = 0.f, akappa = 0.f, bps = 0.f, tsoil = 0.f, em = 0.f;
    float ea = 0.f, sunang = 0.f, swdown = 0.f, thermk = 0.f, tgeff = 0.f;
    float zlwup = 0.f, snocv = 0.f, fsdown = 0.f, fldown = 0.f, fsup = 0.f;
    float flup = 0.f, ccx = 0.f, cg = 0.f, spwet = 0.f, p0 = 0.f;
    float csoil = 0.f, dzsoil = 0.f, chisl = 0.f, smelt = 0.f, prcp = 0.f;
    float prcps = 0.f, prcpw = 0.f, fifall = 0.f, flfall = 0.f;
    float bifall = 0.f, blfall = 0.f, solar = 0.f, snroff = 0.f, hroff = 0.f;
    float solsoil = 0.f, dtc = 0.f, dtg = 0.f, ra = 0.f, ect = 0.f;
    float eci = 0.f, egt = 0.f, egi = 0.f, egs = 0.f, hc = 0.f, hg = 0.f;
    float ec = 0.f, eg = 0.f, chf = 0.f, shf = 0.f, ustar = 0.f, drag = 0.f;
    float cu = 0.f, xct = 0.f, hflux = 0.f, etmass = 0.f, stemp = 0.f;
    float soilm = 0.f, cm = 0.f, fm = 0.f, fh = 0.f;
    float evapsoil = 0.f, evapwc = 0.f, evapdc = 0.f, evapsn = 0.f;
    float evapgx = 0.f, elaten = 0.f, xhsflx = 0.f, rib = 0.f;
    int itype = 0, intg = 0, nroot = 0, mon_cor = 0, iptype = 0, isnow = 0;
    (void)fifall; (void)cm; (void)hflux;
    float &tc = CV(TSLB1), &tgs = CV(TSLB2), &td = CV(TSLB3);
    float &tkair = CV(TKAIR);

    if (mminlu == 1) {
        itype = ivgtyp;
    } else if (mminlu == 2) {
        if (ivgtyp < 1 || ivgtyp > 24) { err |= SSIB_ERR_VEGMAP; return; }
        itype = (int)TAB1(T_IVUSGS, ivgtyp);
    } else if (mminlu == 3 || mminlu == 4) {
        if (ivgtyp < 1 || ivgtyp > 20) { err |= SSIB_ERR_VEGMAP; return; }
        itype = (int)TAB1(T_IVMODIS, ivgtyp);
    } else {
        err |= SSIB_ERR_MAP;
        return;
    }
    if (itype <= 0 || itype > 13) {
        err |= SSIB_ERR_VEGMAP;
        return;
    }
    intg = 1;
    xadj = 0.f;
    ctlpa = 1.f;
    nroot = 1;
    wfsoil = 0.f;
    zwind = zwind2 * 0.5f;
    mon_cor = ss_month(day, zlat);
    if (itime == 1) ta = tc;
    psurf = psur * 0.01f;
    dtt = ddtt * (float)intg;
    ss_vegout(tran, ref, green, vcover, chil, rstpar, topt, tl, tu, defac,
              ph1, ph2, zlt, z0, xdd, z2, z1, rdc, rbc, rootd, soref, bee,
              phsat, poros, satco, slope, zdepth, mon_cor, itype);
    if (itype == 12) ss_crops(zlat, day, chil, zlt, green, vcover);
    if (itime == 1) {
        stlev1 = 0.05f;
        stlev2 = 1.05f;
        depth = TAB1(T_TDDEPTH, itype);
        if (depth > stlev1 && depth <= stlev2) {
            td = ((stlev2 - depth) * tgs + (depth - stlev1) * td)
                 / (stlev2 - stlev1);
        } else if (depth > stlev2) {
            td = ((depth - stlev1) * td - (depth - stlev2) * tgs)
                 / (stlev2 - stlev1);
        }
    }
    www[1] = CV(SMOIS1) / poros;
    www[2] = CV(SMOIS2) / poros;
    www[3] = CV(SMOIS3) / poros;
    CV(SNOW) = CV(SNOW) / 1000.f;
    CV(CANWAT) = CV(CANWAT) / 1000.f;
    // CONVDIM(0): the 4 x 12 carried layer variables into the arrays.
    for (int k = 1; k <= 4; ++k) {
        const int o = S_LAYER1 + 12 * (k - 1);
        sn.dzo[k] = c.s[o + 0];   sn.wo[k] = c.s[o + 1];
        sn.tssn[k] = c.s[o + 2];  sn.tssno[k] = c.s[o + 3];
        sn.bwo[k] = c.s[o + 4];   sn.bto[k] = c.s[o + 5];
        sn.cto[k] = c.s[o + 6];   sn.fio[k] = c.s[o + 7];
        sn.flo[k] = c.s[o + 8];   sn.bio[k] = c.s[o + 9];
        sn.blo[k] = c.s[o + 10];  sn.ho[k] = c.s[o + 11];
    }
    isnow = (int)CV(ISNOW);
    if (itime == 1) {
        isnow = 1;
        CV(SNOWDEN) = 3.75f;
        CV(SWE) = CV(SNOW);
        CV(SNOWDEPTH) = CV(SWE) * CV(SNOWDEN);
    }
    capac[1] = CV(CANWAT);
    capac[2] = CV(SNOW);
    if (itime == 1) {
        if (CV(SNOW) > 0.f) {
            capac[1] = zlt[1] * 0.0001f;
            tc = ss_min(tc, TF - 0.01f);
            tgs = ss_min(tgs, TF - 0.01f);
        }
    }
    um = sqrtf(umm * umm + vmm * vmm);
    rhoair = 100.f / GASR * (psurf + 0.01f * pm) / (tm + ta);
    akappa = GASR / CPAIR;
    bps = 1.0f / ss_exp(akappa * ss_log(0.01f * pm / psurf));
    if (isnow == 0) {
        tsoil = tgs;
        tgs = sn.tssno[NLAY];
        capac[2] = CV(SWE);
        iptype = 2;
        if (tm >= TF) iptype = 1;
    }
    em = (psurf * qm) / 0.6220f;
    if (itime == 1) ea = em;
    sunang = ss_max(sunangle, 0.01746f);
    ss_radn(sw_physics, sunang, swdown1, cloud, CV(SWVISDIR), CV(SWVISDIF),
            CV(SWNIRDIR), CV(SWNIRDIF), radfrac, swdown, rlwdown, radn);
    ss_radab(tran, ref, green, vcover, chil, zlt, z2, z1, soref, tc, tgs,
             satcap, extk, radfac, thermk, radt, par, pd, albedo, salb, tgeff,
             sunang, xadj, capac, radn, zlwup, radfrac, isnow, CV(SNOWDEN),
             CV(SNOWDEPTH), swdown, CV(ALBEDO), snocv, 0, fsdown, fldown,
             fsup, flup);
    ss_root1(phsat, bee, www, phsoil);
    ss_stoma1(green, vcover, chil, zlt, par, pd, extk, sunang, rst, rstpar,
              ctlpa);
    ss_intercs(dtt, vcover, zlt, tm, tc, tgs, capac, www, ppc, ppl,
               CV(SFCRUNOFF), zdepth, poros, ccx, cg, satco, satcap, spwet,
               extk, isnow, p0, csoil, dzsoil, chisl, smelt);
    ss_set0(sn);
    SsibTem p;
    p.dtt = dtt; p.tm = tm; p.qm = qm; p.em = em; p.psurf = psur;
    p.www = www; p.capac = capac; p.satcap = satcap; p.zdepth = zdepth;
    p.vcover = vcover; p.zlai = zlt; p.defac = defac; p.tu = tu; p.tl = tl;
    p.topt = topt; p.rootd = rootd; p.phsoil = phsoil; p.rootp = rootp;
    p.ph1 = ph1; p.ph2 = ph2; p.rst = rst; p.radt = radt; p.rstfac = rstfac;
    p.albedo = albedo; p.bee = bee; p.phsat = phsat; p.poros = poros;
    p.z2 = z2; p.thermk = thermk; p.rhoair = rhoair; p.zwind = zwind;
    p.um = um; p.ccx = ccx; p.cg = cg; p.bps = bps; p.nroot = nroot;
    p.isnow = isnow;
    if (isnow == 0) {
        prcp = p0;
        tkair = tm;
        ss_getmet(iptype, prcp, tkair, prcps, prcpw, fifall, flfall, bifall,
                  blfall);
        solar = 0.f;
        for (int iveg = 2; iveg <= 2; ++iveg)
            for (int iwave = 1; iwave <= 2; ++iwave)
                for (int irad = 1; irad <= 2; ++irad)
                    solar = solar + radfac[iveg][iwave][irad] * radn[iwave][irad];
        ss_snow_1st(dtt, tm, solar, prcpw, prcps, sn, snroff, hroff,
                    CV(SNOWDEPTH), solsoil, bifall, blfall, flfall);
        ss_temrs2(p, tc, tgs, td, ta, dtc, dtg, ra, xdd, z0, rdc, rbc, ect,
                  eci, egt, egi, egs, hc, hg, ec, eg, ea, chf, shf, zlwup,
                  ustar, drag, chisl, tsoil, solsoil, csoil, wfsoil, sn,
                  CV(SWE), CV(SNOWDEN), CV(SNOWDEPTH), tkair, snroff, dzsoil,
                  rib, cu, xct, flup, uv10, err);
        ss_old(sn);
    } else {
        ss_temrs1(p, tc, tgs, td, ta, dtc, dtg, ra, xdd, z0, rdc, rbc, ect,
                  eci, egt, egi, egs, hc, hg, ec, eg, ea, chf, shf, zlwup,
                  ustar, drag, CV(SNOWDEN), rib, cu, xct, flup, uv10);
        CV(SWE) = capac[2];
        CV(SNOWDEPTH) = CV(SWE) * CV(SNOWDEN);
        snroff = 0.f;
    }
    ss_updat1(dtt, tc, tgs, td, capac, dtc, dtg, ect, eci, egt, egi, egs, eg,
              hc, hg, hflux, etmass, CV(SFCRUNOFF), 1, rootd, rootp, poros,
              bee, satco, slope, phsat, zdepth, www, ccx, cg, chf, shf, isnow,
              wfsoil, CV(SWE), smelt);
    if (isnow == 0) {
        capac[2] = CV(SWE);
        if (CV(SNOWDEPTH) < SNODEP_CR) {
            isnow = 1;
            ss_layer1(csoil, tgs, dzsoil, sn.h, sn.w, CV(SNOWDEPTH), CV(SWE),
                      stemp);
        } else {
            isnow = 0;
            ss_modnode(CV(SNOWDEPTH), sn);
        }
    } else if (isnow > 0) {
        if (capac[2] * CV(SNOWDEN) > SNODEP_CR) {
            CV(SWE) = capac[2];
            CV(SNOWDEPTH) = capac[2] * CV(SNOWDEN);
            isnow = 0;
            ss_layern(tgs, CV(SWE), CV(SNOWDEPTH), sn.dzo, sn.bwo, sn.wo,
                      sn.bto, sn.cto, sn.flo, sn.fio, sn.ho, sn.blo, sn.bio,
                      sn.dliqvol, sn.dicevol, sn.tssno, sn.dmlto);
        } else {
            isnow = 1;
        }
    }
    CV(SFCRUNOFF) = CV(SFCRUNOFF) + snroff;
    soilm = (www[1] * zdepth[1] + www[2] * zdepth[2] + www[3] * zdepth[3]) * poros;
    CV(Q2) = 0.622f * ea / (psurf - ea);
    CV(SSIB_CM) = (ustar * ustar) / (um * um);
    fm = VKC / cu;
    fh = VKC / xct;
    evapsoil = egs / dtt;
    evapwc = eci / dtt;
    evapdc = ect / dtt;
    evapsn = egi / dtt;
    evapgx = egt / dtt;
    elaten = evapsoil + evapwc + evapdc + evapsn + evapgx;
    CV(LH) = elaten;
    CV(QFX) = elaten / HLAT;
    CV(GRDFLX) = chf + shf;
    xhsflx = (hc + hg) / dtt;
    CV(HFX) = xhsflx;
    // CONVDIM(1)
    for (int k = 1; k <= 4; ++k) {
        const int o = S_LAYER1 + 12 * (k - 1);
        c.s[o + 0] = sn.dzo[k];   c.s[o + 1] = sn.wo[k];
        c.s[o + 2] = sn.tssn[k];  c.s[o + 3] = sn.tssno[k];
        c.s[o + 4] = sn.bwo[k];   c.s[o + 5] = sn.bto[k];
        c.s[o + 6] = sn.cto[k];   c.s[o + 7] = sn.fio[k];
        c.s[o + 8] = sn.flo[k];   c.s[o + 9] = sn.bio[k];
        c.s[o + 10] = sn.blo[k];  c.s[o + 11] = sn.ho[k];
    }
    CV(SMOIS1) = www[1] * poros;
    CV(SMOIS2) = www[2] * poros;
    CV(SMOIS3) = www[3] * poros;
    CV(SNOW) = capac[2];
    CV(CANWAT) = capac[1];
    CV(SNOW) = CV(SNOW) * 1000.f;
    CV(CANWAT) = CV(CANWAT) * 1000.f;
    CV(ALSWVISDIR) = salb[1][1];
    CV(ALSWVISDIF) = salb[1][2];
    CV(ALSWNIRDIR) = salb[2][1];
    CV(ALSWNIRDIF) = salb[2][2];
    CV(TSK) = tgeff;
    CV(UST) = ustar;
    CV(SSIB_BR) = rib;
    CV(SSIB_FM) = fm;
    CV(SSIB_FH) = fh;
    CV(SSIB_LHF) = elaten;
    CV(SSIB_SHF) = xhsflx;
    CV(SSIB_GHF) = chf + shf;
    CV(SSIB_EGS) = evapsoil;
    CV(SSIB_ECI) = evapwc;
    CV(SSIB_ECT) = evapdc;
    CV(SSIB_EGI) = evapsn;
    CV(SSIB_EGT) = evapgx;
    CV(SSIB_SDN) = fsdown;
    CV(SSIB_SUP) = fsup;
    CV(SSIB_LDN) = fldown;
    CV(SSIB_LUP) = flup;
    CV(SSIB_WAT) = soilm;
    CV(SSIB_SHC) = hc / dtt;
    CV(SSIB_SHG) = hg / dtt;
    CV(SSIB_LAI) = zlt[1];
    CV(SSIB_VCF) = vcover[1];
    CV(SSIB_Z00) = z0;
    CV(SSIB_VEG) = (float)itype;
    CV(SSIBXDD) = xdd;
    CV(ISNOW) = (float)isnow;
}

// ================================================================ SSIB_SEAICE
__device__ void ss_ssib_seaice(SsibCol &c, int itime, float ddtt, float zlat,
                               float sunangle, float ppl, float ppc,
                               float rlwdown, float zwind2, float yice,
                               float umm, float vmm, float qm, float tm,
                               float pm, float psur, float swdown1, float day,
                               float &cloud, float &ta, float &uv10,
                               int sw_physics, float ice_threshold, int fixes)
{
    float capac[3] = {0.f}, satcap[3] = {0.f}, green[3] = {0.f};
    float vcover[3] = {0.f}, zlt[3] = {0.f}, chil[3] = {0.f}, topt[3] = {0.f};
    float tl[3] = {0.f}, tu[3] = {0.f}, defac[3] = {0.f}, ph1[3] = {0.f};
    float ph2[3] = {0.f}, rst[3] = {0.f}, rootd[3] = {0.f}, radt[3] = {0.f};
    float par[3] = {0.f}, pd[3] = {0.f};
    float www[4] = {0.f}, soref[4] = {0.f}, zdepth[4] = {0.f};
    float rootp[4] = {0.f}, phsoil[4] = {0.f};
    float radfrac[3][3] = {{0.f}}, salb[3][3] = {{0.f}}, rstpar[3][4] = {{0.f}};
    float rstfac[3][5] = {{0.f}}, radn[4][3] = {{0.f}};
    float tran[3][4][3] = {{{0.f}}}, ref[3][4][3] = {{{0.f}}};
    float albedo[3][4][3] = {{{0.f}}}, extk[3][4][3] = {{{0.f}}};
    float xadj = 0.f, ctlpa = 0.f, zwind = 0.f, psurf = 0.f, dtt = 0.f;
    float z0 = 0.f, xdd = 0.f, z2 = 0.f, z1 = 0.f, rdc = 0.f, rbc = 0.f;
    float bee = 0.f, phsat = 0.f, poros = 0.f, satco = 0.f, slope = 0.f;
    float stlev1 = 0.f, stlev2 = 0.f, depth = 0.f, snowden = 0.f;
    float xadia = 0.f, xx = 0.f, um = 0.f, rhoair = 0.f, akappa = 0.f;
    float bps = 0.f, em = 0.f, ea = 0.f, sunang = 0.f, swdown = 0.f;
    float closs = 0.f, gloss = 0.f, thermk = 0.f, p1f = 0.f, p2f = 0.f;
    float tgeff = 0.f, zlwup = 0.f, snocv = 0.f, fsdown = 0.f, fldown = 0.f;
    float fsup = 0.f, flup = 0.f, porosave = 0.f, ccx = 0.f, cg = 0.f;
    float spwet = 0.f, rnoffs = 0.f, filtr = 0.f, smelt = 0.f, dtc = 0.f;
    float dtg = 0.f, ra = 0.f, ect = 0.f, eci = 0.f, egt = 0.f, egi = 0.f;
    float egs = 0.f, hc = 0.f, hg = 0.f, ec = 0.f, eg = 0.f, chf = 0.f;
    float shf = 0.f, ustar = 0.f, drag = 0.f, rib = 0.f, cu = 0.f, xct = 0.f;
    float hflux = 0.f, etmass = 0.f, soilm = 0.f, evap = 0.f, fm = 0.f;
    float fh = 0.f, elaten = 0.f, xhsflx = 0.f;
    int itype = 0, intg = 0, nroot = 0, mon_cor = 0, isnow = 0;
    (void)xx; (void)hflux; (void)closs; (void)gloss; (void)p2f;
    float &tc = CV(TSLB1), &tgs = CV(TSLB2), &td = CV(TSLB3);
    xadj = 0.f;
    ctlpa = 1.f;
    nroot = 1;
    intg = 1;
    itype = 13;
    zwind = zwind2 * 0.5f;
    mon_cor = ss_month(day, zlat);
    if (itime == 1) ta = tc;
    psurf = psur * 0.01f;
    dtt = ddtt * (float)intg;
    ss_vegout(tran, ref, green, vcover, chil, rstpar, topt, tl, tu, defac,
              ph1, ph2, zlt, z0, xdd, z2, z1, rdc, rbc, rootd, soref, bee,
              phsat, poros, satco, slope, zdepth, mon_cor, itype);
    if (itime == 1) {
        stlev1 = 0.05f;
        stlev2 = 1.05f;
        depth = TAB1(T_TDDEPTH, itype);
        if (depth > stlev1 && depth <= stlev2) {
            td = ((stlev2 - depth) * tgs + (depth - stlev1) * td)
                 / (stlev2 - stlev1);
        } else if (depth > stlev2) {
            td = ((depth - stlev1) * td - (depth - stlev2) * tgs)
                 / (stlev2 - stlev1);
        }
    }
    www[1] = 1.f;
    www[2] = 1.f;
    www[3] = 1.f;
    CV(SNOW) = CV(SNOW) / 1000.f;
    CV(CANWAT) = CV(CANWAT) / 1000.f;
    capac[1] = CV(CANWAT);
    capac[2] = CV(SNOW);
    snowden = 3.75f;
    isnow = 1;
    if (itime == 1) {
        ta = tgs;
        capac[1] = 0.f;
        capac[2] = 0.f;
        if (CV(SNOW) > 0.f) capac[1] = zlt[1] * 0.0001f;
        tc = ss_min(tc, 273.15f);
        tgs = ss_min(tgs, 273.15f);
        td = ss_min(td, 272.50f);
    } else {
        if (yice < ice_threshold) {
            capac[1] = 0.f;
            capac[2] = 0.f;
            xadia = ss_exp(GASR / CPAIR * ss_log(psur / pm));
            xx = ss_min(tm * xadia, 273.15f);
            tc = ss_min(tm * xadia, 273.15f);
            tgs = ss_min(tm * xadia, 273.15f);
            if (td == 0.f) td = 272.5f;
            td = ss_min(td, 272.5f);
        }
    }
    um = sqrtf(umm * umm + vmm * vmm);
    rhoair = 100.f / GASR * (psurf + 0.01f * pm) / (tm + ta);
    akappa = GASR / CPAIR;
    bps = 1.0f / ss_exp(akappa * ss_log(0.01f * pm / psurf));
    em = (psurf * qm) / 0.6220f;
    if (itime == 1) ea = em;
    sunang = ss_max(sunangle, 0.01746f);
    ss_radn(sw_physics, sunang, swdown1, cloud, CV(SWVISDIR), CV(SWVISDIF),
            CV(SWNIRDIR), CV(SWNIRDIF), radfrac, swdown, rlwdown, radn);
    ss_radab_ice(tran, ref, green, vcover, chil, zlt, z2, z1, soref, tc, tgs,
                 satcap, extk, closs, gloss, thermk, p1f, p2f, radt, par, pd,
                 salb, albedo, tgeff, sunang, xadj, capac, radn, CV(ALBEDO),
                 zlwup, radfrac, swdown, snocv, 1, fsdown, fldown, fsup, flup);
    ss_root1(phsat, bee, www, phsoil);
    ss_stoma1(green, vcover, chil, zlt, par, pd, extk, sunang, rst, rstpar,
              ctlpa);
    porosave = poros;
    poros = 0.95f;
    ss_interc(dtt, vcover, zlt, tm, tc, tgs, capac, www, ppc, ppl,
              CV(SFCRUNOFF), zdepth, poros, ccx, cg, satco, satcap, spwet,
              extk, rnoffs, filtr, smelt);
    SsibTem p;
    p.dtt = dtt; p.tm = tm; p.qm = qm; p.em = em; p.psurf = psur;
    p.www = www; p.capac = capac; p.satcap = satcap; p.zdepth = zdepth;
    p.vcover = vcover; p.zlai = zlt; p.defac = defac; p.tu = tu; p.tl = tl;
    p.topt = topt; p.rootd = rootd; p.phsoil = phsoil; p.rootp = rootp;
    p.ph1 = ph1; p.ph2 = ph2; p.rst = rst; p.radt = radt; p.rstfac = rstfac;
    p.albedo = albedo; p.bee = bee; p.phsat = phsat; p.poros = poros;
    p.z2 = z2; p.thermk = thermk; p.rhoair = rhoair; p.zwind = zwind;
    p.um = um; p.ccx = ccx; p.cg = cg; p.bps = bps; p.nroot = nroot;
    p.isnow = isnow;
    ss_temrs1(p, tc, tgs, td, ta, dtc, dtg, ra, xdd, z0, rdc, rbc, ect, eci,
              egt, egi, egs, hc, hg, ec, eg, ea, chf, shf, zlwup, ustar, drag,
              snowden, rib, cu, xct, flup, uv10);
    ss_updat1_ice(dtt, tc, tgs, td, capac, dtc, dtg, ect, eci, egt, egi, egs,
                  hc, hg, hflux, etmass, CV(SFCRUNOFF), nroot, rootd, rootp,
                  poros, bee, satco, slope, phsat, zdepth, www, ccx, cg, chf,
                  shf, smelt);
    poros = porosave;
    td = ss_min(td, 273.15f);
    tc = ss_min(tc, 273.15f);
    tgs = ss_min(tgs, 273.15f);
    soilm = (www[1] * zdepth[1] + www[2] * zdepth[2] + www[3] * zdepth[3]) * poros;
    CV(Q2) = 0.622f * ea / (psurf - ea);
    evap = etmass * HLAT;
    CV(SSIB_CM) = (ustar * ustar) / (um * um);
    fm = VKC / cu;
    fh = VKC / xct;
    elaten = evap / dtt;
    CV(LH) = elaten;
    CV(QFX) = elaten / HLAT;
    CV(GRDFLX) = chf + shf;
    xhsflx = (hc + hg) / dtt;
    CV(HFX) = xhsflx;
    CV(SMOIS1) = www[1] * poros;
    CV(SMOIS2) = www[2] * poros;
    CV(SMOIS3) = www[3] * poros;
    CV(SNOW) = capac[2];
    CV(CANWAT) = capac[1];
    if (fixes & SSIB_FIX_SEAICE_SNOW_UNITS) {
        // DIVERGENCE (default arm): SSIB_SEAICE divides SNOA and SNOB by
        // 1000 on entry (:1576-1577) but never multiplies them back, as SSIB
        // does at :1374-1375, so every sea-ice step hands the grid metres of
        // water as kg m-2 and the next step divides again.
        CV(SNOW) = CV(SNOW) * 1000.f;
        CV(CANWAT) = CV(CANWAT) * 1000.f;
    }
    CV(ALSWVISDIR) = salb[1][1];
    CV(ALSWVISDIF) = salb[1][2];
    CV(ALSWNIRDIR) = salb[2][1];
    CV(ALSWNIRDIF) = salb[2][2];
    CV(TSK) = tgeff;
    CV(UST) = ustar;
    CV(SSIB_BR) = rib;
    CV(SSIB_FM) = fm;
    CV(SSIB_FH) = fh;
    CV(SSIB_LHF) = elaten;
    CV(SSIB_SHF) = xhsflx;
    CV(SSIB_GHF) = chf + shf;
    CV(SSIB_SDN) = fsdown;
    CV(SSIB_SUP) = fsup;
    CV(SSIB_LDN) = fldown;
    CV(SSIB_LUP) = flup;
    CV(SSIB_WAT) = soilm;
    CV(SSIB_Z00) = z0;
    CV(SSIB_VEG) = (float)itype;
}

// ================================================================ driver
#ifdef SSIB_DIRECT_FORCING
__device__ __forceinline__ float ss_field(
    const unsigned long long* fields, int k, int col, int nx)
{
    const unsigned long long* d = fields + 4*k;
    const char* p = (const char*)d[0] + (col/nx)*d[1] + (col%nx)*d[2];
    return d[3] ? (float)*(const int*)p : *(const float*)p;
}

__device__ __forceinline__ float ss_forcing(
    const unsigned long long* fields, int k, int col, int nx, int ncol,
    const float* cloud, int cloud_present, const float* dz, const float* u,
    const float* v, const float* qv, const float* temperature,
    const float* pressure, float julian)
{
    if (k >= NF_SSIB) return cloud_present ? cloud[(size_t)(k-NF_SSIB)*ncol+col] : 0.f;
    if (k == F_DZ8W1) return dz[col];
    if (k == F_U1) return u[col];
    if (k == F_V1) return v[col];
    if (k == F_QV1) return qv[col];
    if (k == F_T1) return temperature[col];
    if (k == F_P1) return pressure[col];
    if (k == F_JULIAN) return julian;
    return ss_field(fields, NS_SSIB+k, col, nx);
}
#endif
// CASE (SSIBSCHEME), module_surface_driver.F:4156-4345, FRACTIONAL_SEAICE=1
// (so XICE_THRESHOLD = 0.02 and isisfc is true under sf_sfclay_physics=1).
extern "C" __global__ void ssib_surface_step(
#ifdef SSIB_DIRECT_FIELDS
    const float* __restrict__ cst, const unsigned long long* __restrict__ st,
#else
    const float* __restrict__ cst, float* __restrict__ st,
#endif
    const float* __restrict__ frc, int ncol, int nz, int itimestep,
    int fixes, int* __restrict__ err_out
#ifdef SSIB_DIRECT_FIELDS
    , int nx
#endif
#ifdef SSIB_DIRECT_FORCING
    , const float* __restrict__ dz, const float* __restrict__ u,
    const float* __restrict__ v, const float* __restrict__ qv,
    const float* __restrict__ temperature, const float* __restrict__ pressure,
    int cloud_present, float julian, float dt,
    const unsigned long long* __restrict__ finish_fields, int finish
#endif
    )
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
#ifdef SSIB_DIRECT_FORCING
#define CST(k) ((k) == C_XLAND ? ss_field(st, NS_SSIB+NF_SSIB, col, nx) : \
                (k) == C_IVGTYP ? ss_field(st, NS_SSIB+NF_SSIB+1, col, nx) : \
                (k) == C_DTBL ? dt : cst[(size_t)(k)*ncol+col])
#define FRC(k) ss_forcing(st, k, col, nx, ncol, frc, cloud_present, dz, u, v, qv, temperature, pressure, julian)
#else
#define CST(k) cst[(size_t)(k) * ncol + col]
#define FRC(k) frc[(size_t)(k) * ncol + col]
#endif
    SsibCol c;
#ifdef SSIB_DIRECT_FIELDS
    for (int k = 0; k < NS_SSIB; ++k) {
        const unsigned long long* d = st + 4*k;
        const char* p = (const char*)d[0] + (col/nx)*d[1] + (col%nx)*d[2];
        c.s[k] = d[3] ? (float)*(const int*)p : *(const float*)p;
    }
#else
    for (int k = 0; k < NS_SSIB; ++k) c.s[k] = st[(size_t)k * ncol + col];
#endif
    int err = 0;
    const float xland = CST(C_XLAND);
    const int ivgtyp = (int)CST(C_IVGTYP);
    const float xlat = CST(C_XLAT);
    const int mminlu = (int)CST(C_MMINLU);
    const int ra_sw_physics = (int)CST(C_RA_SW);
    const float dtbl = CST(C_DTBL);
    const float xice = FRC(F_XICE);
    const float xice_save = FRC(F_XICE_SAVE);
    const float julian_in = FRC(F_JULIAN);
    const float u1 = FRC(F_U1), v1 = FRC(F_V1);
    const float xice_threshold = 0.02f;
    float day = 0.f, cloudfrac = 0.f, uv10 = 0.f, zdiff = 0.f;
    float t2 = FRC(F_T2_SFCLAY);
    CV(SWVISDIR) = FRC(F_SWVISDIR);
    CV(SWVISDIF) = FRC(F_SWVISDIF);
    CV(SWNIRDIR) = FRC(F_SWNIRDIR);
    CV(SWNIRDIF) = FRC(F_SWNIRDIF);
    if ((xice >= xice_threshold) && (xice <= 1.f)) {
        CV(ALBEDO) = (CV(ALBEDO) - (1.f - xice) * 0.08f) / xice;
    }
    day = (float)((int)(julian_in + 0.01f)) + 1.f;
    if (xland < 1.5f) {
        cloudfrac = 0.f;
        for (int k = 0; k < nz; ++k)
            cloudfrac = ss_max(cloudfrac, ss_min(FRC(NF_SSIB + k), 1.0f));
        if (xice >= xice_threshold) {
            ss_ssib_seaice(c, itimestep, dtbl, xlat, FRC(F_COSZEN),
                           FRC(F_RAINNCV), FRC(F_RAINCV), FRC(F_GLW),
                           FRC(F_DZ8W1), xice_save, u1, v1, FRC(F_QV1),
                           FRC(F_T1), FRC(F_P1), FRC(F_PSFC), FRC(F_SWDOWN),
                           day, cloudfrac, t2, uv10, ra_sw_physics,
                           xice_threshold, fixes);
        } else {
            ss_ssib(c, itimestep, dtbl, xlat, FRC(F_COSZEN), FRC(F_RAINNCV),
                    FRC(F_RAINCV), FRC(F_GLW), FRC(F_DZ8W1), u1, v1,
                    FRC(F_QV1), FRC(F_T1), FRC(F_P1), FRC(F_PSFC), ivgtyp,
                    FRC(F_SWDOWN), day, cloudfrac, t2, uv10, ra_sw_physics,
                    mminlu, err);
        }
        if (!(err & (SSIB_ERR_VEGMAP | SSIB_ERR_MAP))) {
            CV(BR) = CV(SSIB_BR);
            CV(ZNT) = CV(SSIB_Z00);
            CV(SFCEVP) = CV(SFCEVP) + CV(QFX) * dtbl;
            // DIVERGENCE (default arm): WRF overwrites T2 with TSK here
            // ("t2(i,j) = tsk(i,j)  !keep this").  The default keeps SSIB's
            // own canopy-air temperature TA, which SSIB returned through
            // this very argument, beside its canopy-air humidity Q2.
            if (!(fixes & SSIB_FIX_T2_CANOPY_AIR)) t2 = CV(TSK);
            if (itimestep != 1) {
                zdiff = (0.5f * FRC(F_DZ8W1)) - CV(SSIBXDD);
                if (zdiff <= CV(ZNT)) zdiff = CV(ZNT) + 0.2f;
                CV(GZ1OZ0) = ss_log(zdiff / CV(ZNT));
            }
            if (xice >= xice_threshold) {
                CV(SNOWH) = 0.0f;
            } else {
                CV(SNOWH) = CV(SNOWDEPTH);
            }
            CV(U10) = uv10 * u1 / sqrtf(u1 * u1 + v1 * v1);
            CV(V10) = uv10 * v1 / sqrtf(u1 * u1 + v1 * v1);
        }
    }
    // T2 is the surface layer's on water and SSIB's (or TSK) on land.
    CV(T2) = t2;
    if ((xice >= xice_threshold) && (xice <= 1.0f)) {
        CV(ALBEDO) = (CV(ALBEDO) * xice) + ((1.0f - xice) * 0.08f);
    }
    if ((xice >= xice_threshold) && (xice <= 1.0f)) {
        CV(HFX) = (CV(HFX) * xice) + ((1.0f - xice) * FRC(F_HFX_SEA));
        CV(QFX) = (CV(QFX) * xice) + ((1.0f - xice) * FRC(F_QFX_SEA));
        CV(LH) = (CV(LH) * xice) + ((1.0f - xice) * FRC(F_LH_SEA));
        CV(TSK_SAVE) = CV(TSK);
        CV(TSK) = (CV(TSK) * xice) + ((1.0f - xice) * FRC(F_TSK_SEA));
    }
#ifdef SSIB_DIRECT_FIELDS
    for (int k = 0; k < NS_SSIB; ++k) {
        const unsigned long long* d = st + 4*k;
        char* p = (char*)d[0] + (col/nx)*d[1] + (col%nx)*d[2];
        if (d[3]) *(int*)p = (int)c.s[k]; else *(float*)p = c.s[k];
    }
#else
    for (int k = 0; k < NS_SSIB; ++k) st[(size_t)k * ncol + col] = c.s[k];
#endif
#ifdef SSIB_DIRECT_FORCING
    if (finish) {
        // Every thread owns one column. Finish only after its forcing was
        // consumed and its carried fields were written, as the old scatter.
        const unsigned long long* d = finish_fields;
        char* saved = (char*)d[0] + (col/nx)*d[1] + (col%nx)*d[2];
        *(float*)saved = ss_field(finish_fields, 1, col, nx);
        for (int k = 2; k < 10; ++k) {
            d = finish_fields + 4*k;
            char* p = (char*)d[0] + (col/nx)*d[1] + (col%nx)*d[2];
            *(float*)p = 0.f;
        }
        // Only fatal bits enter the shared status. AND commutes with the
        // other columns' fatal ORs, so no final device launch is needed.
        err &= 7;
        if (col == 0) atomicAnd(err_out, 7);
    }
#endif
    if (err) atomicOr(err_out, err);
#undef CST
#undef FRC
}
