// Fast math is the production default. Strict replay selects the pinned
// scalar WRF arithmetic through GPUWM_QNSE_STRICT.
#ifndef GPUWM_QNSE_STRICT
#undef FMUL
#define FMUL(a,b) ((a)*(b))
#define gfk_exp __expf
#define gfk_exp_fma __expf
#define gfk_log __logf
#define gfk_pow __powf
#endif

// QNSE surface layer, sf_sfclay_physics=4, WRF v4.6.1.
//
// Device port of the byte-frozen phys/module_sf_qnsesfc.F (QNSESFC +
// SFCDIF).  The :NNN line anchors below are that Fortran file's.  One thread
// owns one surface column: the full dz/tke columns feed the PBLH scan
// (:258-275), everything else is lowest-model-level state.  The
// PSIM1/PSIH1/PSIM2/PSIH2 tables QNSESFCINIT builds (:1125-1279) are
// frozen assets loaded once through Rust NetCDF by
// gpuwm/core/myjsfc_tables.build_qnse_psi_tables and passed as device arrays.
//
// QNSE is the Eta surface layer with Sukoriansky's quasi-normal scale
// elimination functions; the launcher, runner and published field set are
// the Eta layer's (gpuwm/core/myjsfc.py).  What differs from
// module_sf_myjsfc.F, line by line, is what this file carries:
//   * PBLH threshold EPSQ2L=0.01, not EPSQ2=0.2 (:23, :260).
//   * THSK, APESFC and P02P/P10P use PSFC*1.E-5 and *1.E5, not P1000mb
//     (:221, :288, :327-328): a different float32 rounding.
//   * TZ0 = THZ0*APESFC on every call; MYJ seeds TSK on the first (:289).
//   * ZTFC=0.1 (:38): land ZT starts at 0.1*Z0 and the Sukoriansky floor
//     ZT=MAX(ZT,ZU*ZTFC) follows the Zilitinkevich fix (:684).
//   * The stable Zilitinkevich factor is TOPOTERM=MAX(TOPOFAC*HT**2,3), so
//     the ground height HT enters the land branch (:665-670); MYJ's RIB
//     modification with CZETMAX is absent.
//   * SIMH and the 2 m / 10 m SIMH carry PRT0=0.72, not FH01/FH02=1 (:589).
//   * Over land THZ0 is re-diagnosed from the INCOMING HFX on every
//     iteration, THZ0=HFX/(AKHS*RLOW*CP)+THLOW (:744); HFX is INOUT.
//   * No 2 m / 10 m theta bracketing (MYJ's :960-975 is absent).
//   * HFX/QFX are computed under SCM_FORCE_FLUX present and 0 (:983-988),
//     which is what the ARW driver passes (module_first_rk_step_part1.F:654,
//     Registry default 0).
//   * Over water QS=QLOW+QFX/(RLOW*AKHS), then QS/(1-QS) (:1003-1006); MYJ
//     uses the saturation value at TSK.
//   * Tables: ZTMAX=2.3 and the QNSE stable and unstable-floor functions.
//
// CONFORMANCE: graded word for word against the WRF v4.6.1 column oracle
// (tools/qnsesfc_wrf461_oracle: the byte-unmodified module under gfortran,
// WRF's own -O2 phys flags) by tests/test_qnsesfc_wrf461_parity.py.  What it
// takes:
//   * ZINT(KTE+1)=HT (:168): the interface heights start at the terrain
//     height, and PBLH and ZSL are differences of them (:275, :290).
//   * LOG is gfk_log, EXP gfk_exp_fma and REAL**REAL gfk_pow -- WOOF's own
//     float32 routines (glibc_flt32.cuh, flt32_expf_fma.cuh), not CUDA's.
//     Integer powers are the multiply chains WRF's optimised build emits:
//     USTAR**3 = USTAR*(USTAR*USTAR); the tables' (ZETA-0.5)**5 is
//     X2*(X*X2) on the host.
//   * GPUWM_WRF_EXACT=1 selects scalar WRF math and no contraction.
//     Production uses CUDA elementary functions and permits contraction.
//
// Not ported, on purpose: the debug PRINT at I=170,J=20 (:315-317, no field
// effect); TZ0's update inside the land iteration (:745), whose value
// SFCDIF never reads again and QNSESFC discards; WSTAR (:793), computed and
// never used.

#define QNSE_KZTM 10001
#define QNSE_KZTM2 (QNSE_KZTM - 2)

// ---- share/module_model_constants.F ----
#define QNSE_G        9.81f
#define QNSE_RD       287.0f
#define QNSE_CP       (7.0f * 287.0f / 2.0f)
#define QNSE_XLV      2.5e6f
#define QNSE_P608     (461.6f / 287.0f - 1.0f)
#define QNSE_PQ0      379.90516f
#define QNSE_A2       17.2693882f
#define QNSE_A3       273.16f
#define QNSE_A4       35.86f

// ---- module_sf_qnsesfc.F:21-57 ----
#define QNSE_ITRMX    5
#define QNSE_EPSQ2L   0.01f
#define QNSE_VKARMAN  0.4f
#define QNSE_CAPA     (QNSE_RD / QNSE_CP)
#define QNSE_ELOCP    (2.72e6f / QNSE_CP)
#define QNSE_RCAP     (1.0f / QNSE_CAPA)
#define QNSE_GOCP02   (QNSE_G / QNSE_CP * 2.0f)
#define QNSE_GOCP10   (QNSE_G / QNSE_CP * 10.0f)
#define QNSE_EPSU2    1.0e-6f
#define QNSE_EPSUST   1.0e-9f
#define QNSE_EPSZT    1.0e-28f
#define QNSE_A2S      17.2693882f
#define QNSE_A3S      273.16f
#define QNSE_A4S      35.86f
#define QNSE_SEAFC    0.98f
#define QNSE_PQ0SEA   (QNSE_PQ0 * QNSE_SEAFC)
#define QNSE_CZIL     0.1f
#define QNSE_EXCML    0.0001f
#define QNSE_EXCMS    0.0001f
#define QNSE_GLKBR    10.0f
#define QNSE_GLKBS    30.0f
#define QNSE_QVISC    2.1e-5f
#define QNSE_SMALL    0.35f
#define QNSE_SQPR     0.84f
#define QNSE_SQSC     0.84f
#define QNSE_SQVISC   258.2f
#define QNSE_TVISC    2.1e-5f
#define QNSE_USTC     0.7f
#define QNSE_USTR     0.225f
#define QNSE_VISC     1.5e-5f
#define QNSE_FH       1.01f
#define QNSE_WWST     1.2f
#define QNSE_ZTFC     0.1f
#define QNSE_TOPOFAC  9.0e-6f
#define QNSE_CZIV     (QNSE_SMALL * QNSE_GLKBS)
#define QNSE_GRRS     (QNSE_GLKBR / QNSE_GLKBS)
#define QNSE_RTVISC   (1.0f / QNSE_TVISC)
#define QNSE_RVISC    (1.0f / QNSE_VISC)
#define QNSE_ZQRZT    (QNSE_SQSC / QNSE_SQPR)
#define QNSE_FZQ1     (QNSE_RTVISC * QNSE_QVISC * QNSE_ZQRZT)
#define QNSE_FZQ2     (QNSE_RTVISC * QNSE_QVISC * QNSE_ZQRZT)
#define QNSE_FZT1     (QNSE_RVISC * QNSE_TVISC * QNSE_SQPR)
#define QNSE_FZT2     (QNSE_CZIV * QNSE_GRRS * QNSE_TVISC * QNSE_SQPR)
#define QNSE_FZU1     (QNSE_CZIV * QNSE_VISC)
#define QNSE_PRT0     0.72f
#define QNSE_RQVISC   (1.0f / QNSE_QVISC)
#define QNSE_USTFC    (0.018f / QNSE_G)
#define QNSE_WWST2    (QNSE_WWST * QNSE_WWST)
#define QNSE_ZILFC    (-(QNSE_CZIL * QNSE_VKARMAN * QNSE_SQVISC))

// One PSIM/PSIH table lookup (module_sf_qnsesfc.F:561-566 pattern).
__device__ static real qnsesfc_table(const real* __restrict__ psi,
                                     real zeta, real ztmin, real dzeta)
{
    real rz = (zeta - ztmin) / dzeta;
    int k = (int)rz;
    real rdzt = rz - (real)k;
    k = min(k, QNSE_KZTM2);
    k = max(k, 0);
    return (psi[k + 1] - psi[k]) * rdzt + psi[k];
}

extern "C" __global__ void qnsesfc_column(
    // full columns, gpuwm bottom-up (nz, n)
    const real* __restrict__ dz_a,
    const real* __restrict__ tke_a,
    // lowest-model-level state (n)
    const real* __restrict__ u1_a, const real* __restrict__ v1_a,
    const real* __restrict__ t1_a, const real* __restrict__ th1_a,
    const real* __restrict__ qv1_a, const real* __restrict__ qc1_a,
    const real* __restrict__ p1_a, const real* __restrict__ psfc_a,
    // static surface (n)
    const real* __restrict__ tsk_a, const real* __restrict__ xland_a,
    const real* __restrict__ mavail_a, const real* __restrict__ z0base_a,
    const real* __restrict__ ht_a,
    // WRF INOUT state (n)
    real* __restrict__ ust_a, real* __restrict__ znt_a,
    real* __restrict__ thz0_a, real* __restrict__ qz0_a,
    real* __restrict__ uz0_a, real* __restrict__ vz0_a,
    real* __restrict__ qsfc_a, real* __restrict__ akhs_a,
    real* __restrict__ akms_a,
    // outputs (n); hfx is ALSO read: the land branch takes the incoming
    // (land-surface model's) HFX (:744)
    real* __restrict__ rmol_a, real* __restrict__ ct_a,
    real* __restrict__ pblh_a, real* __restrict__ rib_a,
    real* __restrict__ chs_a, real* __restrict__ chs2_a,
    real* __restrict__ cqs2_a, real* __restrict__ hfx_a,
    real* __restrict__ qfx_a, real* __restrict__ lh_a,
    real* __restrict__ flhc_a, real* __restrict__ flqc_a,
    real* __restrict__ qgh_a, real* __restrict__ cpm_a,
    real* __restrict__ u10_a, real* __restrict__ v10_a,
    real* __restrict__ t2_a, real* __restrict__ th2_a,
    real* __restrict__ tshltr_a, real* __restrict__ th10_a,
    real* __restrict__ q2_a, real* __restrict__ qshltr_a,
    real* __restrict__ q10_a, real* __restrict__ pshltr_a,
    real* __restrict__ u10e_a, real* __restrict__ v10e_a,
    // QNSESFCINIT tables (QNSE_KZTM each) + their scalars
    const real* __restrict__ psim1, const real* __restrict__ psih1,
    const real* __restrict__ psim2, const real* __restrict__ psih2,
    real dzeta1, real dzeta2, real ztmin1, real ztmax1,
    real ztmin2, real ztmax2,
    int itimestep, int nz, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    size_t st = (size_t)n;

    int ntsd = itimestep;
    real ust = ust_a[col];
    if (ntsd == 1) ust = 0.1f;            // :191-201
    real psfc = psfc_a[col];
    real thsk = tsk_a[col] / gfk_pow(psfc * 1.0e-5f, QNSE_CAPA);   // :224
    real seamask = xland_a[col] - 1.0f;   // :228

    // PBLH scan (:258-275).  ZINT(I,KTE+1,J)=HT(I,J) (:168) and
    // ZINT(I,K,J)=ZINT(I,K+1,J)+DZ(I,KFLIP,J) (:180-187): heights above sea
    // level, accumulated inside ONE column; PBLH=ZHK(LPBL)-ZHK(LMH+1).
    real zground = ht_a[col];             // ZHK(LMH+1) = HT
    real zlow = zground + dz_a[col];      // ZHK(LMH)
    real zcum = zlow;
    real pblh = 0.0f;
    bool found = false;
    for (int iz = 1; iz < nz && !found; ++iz) {
        zcum = zcum + dz_a[(size_t)iz * st + col];
        if (2.0f * tke_a[(size_t)iz * st + col] <= QNSE_EPSQ2L * QNSE_FH) {
            pblh = zcum - zground;
            found = true;
        }
    }
    if (!found) pblh = zcum - zground;    // LPBL = 1

    // Lowest-layer state (:282-292).
    real plow = p1_a[col];
    real tlow = t1_a[col];
    real thlow = th1_a[col];
    real qv1 = qv1_a[col];
    real qlow = qv1 / (1.0f + qv1);
    real cwmlow = qc1_a[col];
    real thelow = (cwmlow * (-QNSE_ELOCP / tlow) + 1.0f) * thlow;
    real ulow = u1_a[col];
    real vlow = v1_a[col];
    real zsl = (zlow - zground) * 0.5f;
    real apesfc = gfk_pow(psfc * 1.0e-5f, QNSE_CAPA);
    real thz0 = thz0_a[col];
    real tz0 = thz0 * apesfc;             // :292

    // ---- SFCDIF (:350-1009) ----
    real qs = qsfc_a[col];
    real qz0 = qz0_a[col];
    real uz0 = uz0_a[col];
    real vz0 = vz0_a[col];
    real z0 = znt_a[col];
    real akhs = akhs_a[col];
    real akms = akms_a[col];
    real z0base = z0base_a[col];
    real mavail = mavail_a[col];
    real hfx_in = hfx_a[col];

    real rdz = 1.0f / zsl;
    real cxchl = QNSE_EXCML * rdz;
    real cxchs = QNSE_EXCMS * rdz;
    real btgx = QNSE_G / thlow;
    real elfc = QNSE_VKARMAN * btgx;
    real btgh = (pblh > 1000.0f) ? btgx * pblh : btgx * 1000.0f;
    real rlow = plow / (QNSE_RD * tlow);  // :425

    real wstar2 = 0.0f;
    real rlmo = 0.0f;
    real rib = 0.0f, du2 = QNSE_EPSU2, dthv = 0.0f;
    real zu = z0, zt = z0;
    real psmz = 0.0f, pshz = 0.0f, ustark = ust * QNSE_VKARMAN;

    if (seamask > 0.5f) {
        // ---- sea points (:437-613) ----
        for (int itr = 0; itr < QNSE_ITRMX; ++itr) {
            z0 = fmaxf(QNSE_USTFC * ust * ust, 1.59e-5f);
            if (ust < QNSE_USTC) {
                if (ust < QNSE_USTR) {
                    if (ntsd == 1) {
                        akms = cxchs;
                        akhs = cxchs;
                        qs = qlow;
                    }
                    zu = QNSE_FZU1 * sqrtf(sqrtf(z0 * ust * QNSE_RVISC)) / ust;
                    real wght = akms * zu * QNSE_RVISC;
                    real rwgh = wght / (wght + 1.0f);
                    uz0 = (ulow * rwgh + uz0) * 0.5f;
                    vz0 = (vlow * rwgh + vz0) * 0.5f;
                    zt = QNSE_FZT1 * zu;
                    real zq = QNSE_FZQ1 * zt;
                    real wghtt = akhs * zt * QNSE_RTVISC;
                    real wghtq = akhs * zq * QNSE_RQVISC;
                    if (ntsd > 1) {
                        thz0 = ((wghtt * thlow + thsk) / (wghtt + 1.0f)
                                + thz0) * 0.5f;
                        qz0 = ((wghtq * qlow + qs) / (wghtq + 1.0f)
                               + qz0) * 0.5f;
                    } else {
                        thz0 = (wghtt * thlow + thsk) / (wghtt + 1.0f);
                        qz0 = (wghtq * qlow + qs) / (wghtq + 1.0f);
                    }
                }
                if (ust >= QNSE_USTR && ust < QNSE_USTC) {
                    zu = z0;
                    uz0 = 0.0f;
                    vz0 = 0.0f;
                    zt = QNSE_FZT2 * sqrtf(sqrtf(z0 * ust * QNSE_RVISC)) / ust;
                    real zq = QNSE_FZQ2 * zt;
                    real wghtt = akhs * zt * QNSE_RTVISC;
                    real wghtq = akhs * zq * QNSE_RQVISC;
                    if (ntsd > 1) {
                        thz0 = ((wghtt * thlow + thsk) / (wghtt + 1.0f)
                                + thz0) * 0.5f;
                        qz0 = ((wghtq * qlow + qs) / (wghtq + 1.0f)
                               + qz0) * 0.5f;
                    } else {
                        thz0 = (wghtt * thlow + thsk) / (wghtt + 1.0f);
                        qz0 = (wghtq * qlow + qs) / (wghtq + 1.0f);
                    }
                }
            } else {
                zu = z0;
                uz0 = 0.0f;
                vz0 = 0.0f;
                zt = z0;
                thz0 = thsk;
                qz0 = qs;
            }
            real tem = (tlow + tz0) * 0.5f;
            real thm = (thelow + thz0) * 0.5f;
            real a = thm * QNSE_P608;
            real b = (QNSE_ELOCP / tem - 1.0f - QNSE_P608) * thm;
            dthv = ((thelow - thz0)
                    * ((qlow + qz0 + cwmlow) * (0.5f * QNSE_P608) + 1.0f)
                    + (qlow - qz0 + cwmlow) * a + cwmlow * b);
            du2 = fmaxf((ulow - uz0) * (ulow - uz0)
                        + (vlow - vz0) * (vlow - vz0), QNSE_EPSU2);
            rib = btgx * dthv * zsl / du2;
            real zslu = zsl + zu;
            real zslt = zsl + zt;
            real rzsu = zslu / zu;
            real rzst = zslt / zt;
            real rlogu = gfk_log(rzsu);
            real rlogt = gfk_log(rzst);
            rlmo = elfc * akhs * dthv / (ust * (ust * ust));
            real zetalu = zslu * rlmo;
            real zetalt = zslt * rlmo;
            real zetau = zu * rlmo;
            real zetat = zt * rlmo;
            zetalu = fminf(fmaxf(zetalu, ztmin1), ztmax1);
            zetalt = fminf(fmaxf(zetalt, ztmin1), ztmax1);
            zetau = fminf(fmaxf(zetau, ztmin1 / rzsu), ztmax1 / rzsu);
            zetat = fminf(fmaxf(zetat, ztmin1 / rzst), ztmax1 / rzst);
            psmz = qnsesfc_table(psim1, zetau, ztmin1, dzeta1);
            real psmzl = qnsesfc_table(psim1, zetalu, ztmin1, dzeta1);
            real simm = psmzl - psmz + rlogu;
            pshz = qnsesfc_table(psih1, zetat, ztmin1, dzeta1);
            real pshzl = qnsesfc_table(psih1, zetalt, ztmin1, dzeta1);
            real simh = (pshzl - pshz + rlogt) * QNSE_PRT0;
            ustark = ust * QNSE_VKARMAN;
            akms = fmaxf(ustark / simm, cxchs);
            akhs = fmaxf(ustark / simh, cxchs);
            if (dthv <= 0.0f) {
                wstar2 = QNSE_WWST2
                    * gfk_pow(fabsf(btgh * akhs * dthv), 2.0f / 3.0f);
            } else {
                wstar2 = 0.0f;
            }
            ust = fmaxf(sqrtf(akms * sqrtf(du2 + wstar2)), QNSE_EPSUST);
        }
    } else {
        // ---- land points (:621-763) ----
        if (ntsd == 1) qs = qlow;
        zu = z0;
        uz0 = 0.0f;
        vz0 = 0.0f;
        zt = zu * QNSE_ZTFC;
        thz0 = thsk;
        qz0 = qs;
        real tem = (tlow + tz0) * 0.5f;
        real thm = (thelow + thz0) * 0.5f;
        real a = thm * QNSE_P608;
        real b = (QNSE_ELOCP / tem - 1.0f - QNSE_P608) * thm;
        dthv = ((thelow - thz0)
                * ((qlow + qz0 + cwmlow) * (0.5f * QNSE_P608) + 1.0f)
                + (qlow - qz0 + cwmlow) * a + cwmlow * b);
        du2 = fmaxf((ulow - uz0) * (ulow - uz0)
                    + (vlow - vz0) * (vlow - vz0), QNSE_EPSU2);
        rib = btgx * dthv * zsl / du2;
        real zslu = zsl + zu;
        real rzsu = zslu / zu;
        real rlogu = gfk_log(rzsu);
        real zslt = zsl + zu;             // u,v and t at the same level (:663)
        // Topography in the stable Zilitinkevich factor (:668-675):
        // ZSFC is ZHK(LMH+1), the terrain height HT.
        real topoterm = QNSE_TOPOFAC * (zground * zground);
        topoterm = fmaxf(topoterm, 3.0f);
        real zzil = (dthv > 0.0f) ? QNSE_ZILFC * topoterm : QNSE_ZILFC;
        for (int itr = 0; itr < QNSE_ITRMX; ++itr) {
            zt = fmaxf(gfk_exp_fma(zzil * sqrtf(ust * z0base)) * z0base,
                       QNSE_EPSZT);                       // :686
            zt = fmaxf(zt, zu * QNSE_ZTFC);               // :687
            real rzst = zslt / zt;
            real rlogt = gfk_log(rzst);
            rlmo = elfc * akhs * dthv / (ust * (ust * ust));
            real zetalu = zslu * rlmo;
            real zetalt = zslt * rlmo;
            real zetau = zu * rlmo;
            real zetat = zt * rlmo;
            zetalu = fminf(fmaxf(zetalu, ztmin2), ztmax2);
            zetalt = fminf(fmaxf(zetalt, ztmin2), ztmax2);
            zetau = fminf(fmaxf(zetau, ztmin2 / rzsu), ztmax2 / rzsu);
            zetat = fminf(fmaxf(zetat, ztmin2 / rzst), ztmax2 / rzst);
            psmz = qnsesfc_table(psim2, zetau, ztmin2, dzeta2);
            real psmzl = qnsesfc_table(psim2, zetalu, ztmin2, dzeta2);
            real simm = psmzl - psmz + rlogu;
            pshz = qnsesfc_table(psih2, zetat, ztmin2, dzeta2);
            real pshzl = qnsesfc_table(psih2, zetalt, ztmin2, dzeta2);
            real simh = (pshzl - pshz + rlogt) * QNSE_PRT0;
            ustark = ust * QNSE_VKARMAN;
            akms = fmaxf(ustark / simm, cxchl);
            akhs = fmaxf(ustark / simh, cxchl);
            thz0 = hfx_in / (akhs * rlow * QNSE_CP) + thlow;   // :747
            if (dthv <= 0.0f) {
                wstar2 = QNSE_WWST2
                    * gfk_pow(fabsf(btgh * akhs * dthv), 2.0f / 3.0f);
            } else {
                wstar2 = 0.0f;
            }
            ust = fmaxf(sqrtf(akms * sqrtf(du2 + wstar2)), QNSE_EPSUST);
        }
    }

    // ---- 2 m / 10 m diagnostics (:793-925) ----
    real umflx = akms * (ulow - uz0);
    real vmflx = akms * (vlow - vz0);
    real hsflx = akhs * (thlow - thz0);
    real hlflx = akhs * (qlow - qz0);
    real zu10 = zu + 10.0f;
    real zt02 = zt + 2.0f;
    real zt10 = zt + 10.0f;
    real rlnu10 = gfk_log(zu10 / zu);
    real rlnt02 = gfk_log(zt02 / zt);
    real rlnt10 = gfk_log(zt10 / zt);
    real ztau10 = zu10 * rlmo;
    real ztat02 = zt02 * rlmo;
    real ztat10 = zt10 * rlmo;
    real akms10, akhs02, akhs10;
    if (seamask > 0.5f) {
        ztau10 = fminf(fmaxf(ztau10, ztmin1), ztmax1);
        ztat02 = fminf(fmaxf(ztat02, ztmin1), ztmax1);
        ztat10 = fminf(fmaxf(ztat10, ztmin1), ztmax1);
        real psm10 = qnsesfc_table(psim1, ztau10, ztmin1, dzeta1);
        real simm10 = psm10 - psmz + rlnu10;
        real psh02 = qnsesfc_table(psih1, ztat02, ztmin1, dzeta1);
        real simh02 = (psh02 - pshz + rlnt02) * QNSE_PRT0;
        real psh10 = qnsesfc_table(psih1, ztat10, ztmin1, dzeta1);
        real simh10 = (psh10 - pshz + rlnt10) * QNSE_PRT0;
        akms10 = fmaxf(ustark / simm10, cxchs);
        akhs02 = fmaxf(ustark / simh02, cxchs);
        akhs10 = fmaxf(ustark / simh10, cxchs);
    } else {
        ztau10 = fminf(fmaxf(ztau10, ztmin2), ztmax2);
        ztat02 = fminf(fmaxf(ztat02, ztmin2), ztmax2);
        ztat10 = fminf(fmaxf(ztat10, ztmin2), ztmax2);
        real psm10 = qnsesfc_table(psim2, ztau10, ztmin2, dzeta2);
        real simm10 = psm10 - psmz + rlnu10;
        real psh02 = qnsesfc_table(psih2, ztat02, ztmin2, dzeta2);
        real simh02 = (psh02 - pshz + rlnt02) * QNSE_PRT0;
        real psh10 = qnsesfc_table(psih2, ztat10, ztmin2, dzeta2);
        real simh10 = (psh10 - pshz + rlnt10) * QNSE_PRT0;
        akms10 = fmaxf(ustark / simm10, cxchl);
        akhs02 = fmaxf(ustark / simh02, cxchl);
        akhs10 = fmaxf(ustark / simh10, cxchl);
    }
    real u10 = umflx / akms10 + uz0;
    real v10 = vmflx / akms10 + vz0;
    real th02 = hsflx / akhs02 + thz0;
    real th10 = hsflx / akhs10 + thz0;
    real q02 = hlflx / akhs02 + qz0;
    real q10 = hlflx / akhs10 + qz0;
    real pshltr = psfc * gfk_exp_fma(-0.068283f / tlow);
    real u10e = u10;
    real v10e = v10;
    if (seamask < 0.5f) {
        // "Equivalent" z0 shelter winds over land (:934-966).
        real zuuz = fminf(zu * 0.50f, 0.18f);
        real zu_l = fmaxf(zu * 0.35f, zuuz);
        real zu10_l = zu_l + 10.0f;
        real rzsu_l = zu10_l / zu_l;
        real rlnu10_l = gfk_log(rzsu_l);
        real ztau10_l = zu10_l * rlmo;
        ztau10_l = fminf(fmaxf(ztau10_l, ztmin2), ztmax2);
        real psm10 = qnsesfc_table(psim2, ztau10_l, ztmin2, dzeta2);
        real simm10 = psm10 - psmz + rlnu10_l;
        real ekms10 = fmaxf(ustark / simm10, cxchl);
        u10e = umflx / ekms10 + uz0;
        v10e = vmflx / ekms10 + vz0;
    }
    // ---- WRF driver arrays (:975-1006) ----
    real chs = akhs;
    real chs2 = akhs02;
    real cqs2 = akhs02;
    real hfx = -rlow * QNSE_CP * hsflx;
    real qfx = -rlow * hlflx * mavail;
    real flx_lh = QNSE_XLV * qfx;
    real flhc = rlow * QNSE_CP * akhs;
    real flqc = rlow * akhs * mavail;
    real qgh = ((1.0f - seamask) * QNSE_PQ0 + seamask * QNSE_PQ0SEA) / plow
        * gfk_exp_fma(QNSE_A2S * (tlow - QNSE_A3S) / (tlow - QNSE_A4S));
    qgh = qgh / (1.0f - qgh);
    real cpm = QNSE_CP * (1.0f + 0.8f * qlow);
    if (seamask > 0.5f) {
        qs = qlow + qfx / (rlow * akhs);
        qs = qs / (1.0f - qs);
    }
    // ---- shelter supersaturation removal (QNSESFC :314-338) ----
    real tshltr = th02;      // SFCDIF's TH02 dummy is QNSESFC's TSHLTR
    real qshltr = q02;       // SFCDIF's Q02 dummy is QNSESFC's QSHLTR
    real rapa = apesfc;
    real th02p = tshltr;
    real th10p = th10;
    real th2_grid = tshltr;
    real rapa02 = rapa - QNSE_GOCP02 / th02p;
    real rapa10 = rapa - QNSE_GOCP10 / th10p;
    real t02p = th02p * rapa02;
    real t10p = th10p * rapa10;
    real t2_grid = th2_grid * apesfc;
    real p02p = gfk_pow(rapa02, QNSE_RCAP) * 1.0e5f;
    real p10p = gfk_pow(rapa10, QNSE_RCAP) * 1.0e5f;
    real qs02 = QNSE_PQ0 / p02p
        * gfk_exp_fma(QNSE_A2 * (t02p - QNSE_A3) / (t02p - QNSE_A4));
    real qs10 = QNSE_PQ0 / p10p
        * gfk_exp_fma(QNSE_A2 * (t10p - QNSE_A3) / (t10p - QNSE_A4));
    if (qshltr > qs02) qshltr = qs02;
    if (q10 > qs10) q10 = qs10;
    real q02_grid = qshltr / (1.0f - qshltr);

    // ---- writeback ----
    ust_a[col] = ust;
    znt_a[col] = z0;
    thz0_a[col] = thz0;
    qz0_a[col] = qz0;
    uz0_a[col] = uz0;
    vz0_a[col] = vz0;
    qsfc_a[col] = qs;
    akhs_a[col] = akhs;
    akms_a[col] = akms;
    rmol_a[col] = rlmo;
    ct_a[col] = 0.0f;                     // :206-209, :781
    pblh_a[col] = pblh;
    rib_a[col] = rib;
    chs_a[col] = chs;
    chs2_a[col] = chs2;
    cqs2_a[col] = cqs2;
    hfx_a[col] = hfx;
    qfx_a[col] = qfx;
    lh_a[col] = flx_lh;
    flhc_a[col] = flhc;
    flqc_a[col] = flqc;
    qgh_a[col] = qgh;
    cpm_a[col] = cpm;
    u10_a[col] = u10;
    v10_a[col] = v10;
    t2_a[col] = t2_grid;
    th2_a[col] = th2_grid;
    tshltr_a[col] = tshltr;
    th10_a[col] = th10;
    q2_a[col] = q02_grid;
    qshltr_a[col] = qshltr;
    q10_a[col] = q10;
    pshltr_a[col] = pshltr;
    u10e_a[col] = u10e;
    v10e_a[col] = v10e;
}
