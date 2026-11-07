// Fast math is the production default. Strict replay selects the pinned
// scalar WRF arithmetic through GPUWM_QNSE_STRICT.
#ifndef GPUWM_QNSE_STRICT
#undef FMUL
#define FMUL(a,b) ((a)*(b))
#define gfk_exp expf
#define gfk_exp_fma expf
#define gfk_log logf
#define gfk_pow powf
#endif

// QNSE PBL (bl_pbl_physics=4) and its EDMF shallow mass flux (mfshconv=1),
// WRF v4.6.1 phys/module_bl_qnsepbl.F and phys/module_bl_mfshconvpbl.F.
//
// Two kernels, one thread per bottom-up gpuwm column, launched in WRF's
// order (module_pbl_driver.F:1515-1557):
//
//   mfshconv_column  MFSHCONVPBL: Bougeault-Lacarrere length, the dry and
//                    cloudy updraft, MF_TURB and the convective cloud.  It
//                    ASSIGNS RTHBLTEN/RQVBLTEN/RQCBLTEN/RUBLTEN/RVBLTEN, as
//                    the Fortran does (:253-265).
//   qnsepbl_column   QNSEPBL: MIXLEN, DIFCOF, PRODQ2, VDIFQ, VDIFH, VDIFV.
//                    It ADDS its tendencies to whatever the slot holds
//                    (:543-545, :649-650): the mass-flux rates when
//                    mfshconv=1, the driver's zeroed rows otherwise.
//
// Graded word for word against tools/qnse_wrf461_oracle (gfortran 13 at
// WRF's -O2 -fno-fast-math -ffp-contract=off flags) by
// tests/test_qnse_wrf461_parity.py under GPUWM_WRF_EXACT=1. Strict
// products are FMUL, with scalar WRF elementary functions and no FMA.
// Production uses CUDA elementary functions and permits contraction.
// At -O2
// gfortran folds a real exponent of 2.0 to x*x (pow-fold-probe.txt), so
// ZKIC**2. is spelled as a product here.
//
// Interface heights are seeded with the terrain height exactly as WRF
// seeds them (ZINT(I,KTE+1,J)=HT(I,J), :273), unlike myjpbl.cu: the
// float32 rounding of the differences depends on the offset.
//
// Constants are the words gfortran folded (oracle consts.txt), pinned as
// hex literals; C0**3 is one MPFR rounding there, not a product chain.
//
// LAYOUT.  Inputs and outputs are (nz, n) bottom-up planes.  Inside QNSE,
// m is the zero-based QNSE index (Fortran K = m+1, counted DOWN from the
// top), zh[m] is ZHK(m+1) and zh[nz] = ZHK(LMH+1) = HT.  Inside the mass
// flux, k is the zero-based WRF level (Fortran K = k+1, counted UP).
//
// WRF's tile-wide COUNT(GTEST) controls the updraft level loop. The
// production launcher first reduces each column's last active level on
// the GPU, then repeats the updraft through that tile maximum. Unmasked
// thermodynamic and detrainment diagnostics above an individual plume
// therefore follow native WRF too. The historical one-column fixture is
// replayed with tile_mode=0; runtime uses the complete device domain.
//
// GTEST_LOCAL_LCL2 (:730-731) is read at :997 on paths that never assign
// it.  This kernel starts it false, as every other logical of the routine
// starts.  The oracle is built with -finit-logical=false and its
// -finit-logical=true arm reproduces every word too: on those paths
// ZPART_DRY is still 1, so the cloudy increment it gates is
// (1-ZPART_DRY)*... = +0.
//
// Unused by WRF and not computed here: COMPUTE_FUNCTION_THERMO_MF's
// AMOIST/ATHETA, MFSHCONVPBL_CORE's PTH/ZCPH/ZEXN (overwritten at
// :465-469 before any use), COMPUTE_MF_CLOUD's cloud fraction, BL89's
// first ZLM loop (:2730-2736, overwritten at :2738), COMPUTE_UPDRAFT's
// PW_UP.  BL89's mixing length and the theta-v flux reach no prognostic
// field in WRF (QNSEPBL never reads LM_BL89, and its WTHV_MF term is
// commented out at :948); the kernel writes them only when diag != 0
// (the parity test asks for them).

#ifndef QNSE_KMAX
#define QNSE_KMAX 128
#endif

// ---- share/module_model_constants.F, as folded (oracle consts.txt) ----
#define QN_R_D      0x1.1fp+8f          // R_D
#define QN_CP       0x1.f64p+9f         // CP = 7*R_D/2
#define QN_G        0x1.39eb86p+3f      // G = 9.81
#define QN_XLV      0x1.312dp+21f       // XLV
#define QN_P608     0x1.377b48p-1f      // P608 = R_V/R_D-1
#define QN_A2       0x1.144f6ap+4f      // A2
#define QN_A3       0x1.1128f6p+8f      // A3
#define QN_A4       0x1.1ee148p+5f      // A4

// ---- module_bl_qnsepbl.F:25-135 ----
#define QN_EPSQ2L   0x1.47ae14p-7f      // EPSQ2L = 0.01
#define QN_C0       0x1.19999ap-1f      // C0 = 0.55
#define QN_CEPS     0x1.54bc6cp-3f      // CEPS = C0**3
#define QN_BLCKDR   0x1.9ce076p-8f      // BLCKDR
#define QN_CN       0x1.8p-1f           // CN
#define QN_AM1      0x1.0p+3f           // AM1
#define QN_AM2      0x1.266666p+1f      // AM2
#define QN_AM3      0x1.18p+5f          // AM3
#define QN_AH1      0x1.666666p+0f      // AH1
#define QN_AH2      -0x1.47ae14p-7f     // AH2
#define QN_AH3      0x1.4a3d7p+0f       // AH3
#define QN_AH4      0x1.3851ecp+1f      // AH4
#define QN_AH5      0x1.3cccccp+4f      // AH5
#define QN_ARIMIN   0x1.041894p-3f      // ARIMIN
#define QN_BM1      0x1.70a3d8p+1f      // BM1
#define QN_BM2      0x1.0p+4f           // BM2
#define QN_BH1      0x1.ccccccp+1f      // BH1
#define QN_BH2      0x1.0p+4f           // BH2
#define QN_BH3      0x1.68p+9f          // BH3
#define QN_EPSKM    0x1.0624dep-10f     // EPSKM
#define QN_CAPA     0x1.24924ap-2f      // CAPA = R_D/CP
#define QN_RLIVWV   0x1.23d70ap+0f      // RLIVWV = XLS/XLV
#define QN_ELOCP    0x1.527a14p+11f     // ELOCP = 2.72E6/CP
#define QN_EPS1     0x1.197998p-40f     // EPS1
#define QN_EPSL     0x1.47ae14p-2f      // EPSL
#define QN_EPSRU    0x1.ad7f2ap-24f     // EPSRU
#define QN_FH       0x1.028f5cp+0f      // FH
#define QN_ALPH     0x1.333334p-2f      // ALPH
#define QN_EL0MAX   0x1.f4p+9f          // EL0MAX
#define QN_EL0MIN   0x1.0p+0f           // EL0MIN
#define QN_ELFC     0x1.d70a3ep-4f      // ELFC = 0.23*0.5
#define QN_ESQ      0x1.4p+2f           // ESQ
#define QN_ELZ0     0x0.0p+0f           // ELZ0
#define QN_PQ0SEA   0x1.744e9cp+8f      // PQ0SEA = PQ0*SEAFC
#define QN_REQU     0x1.23af8p-4f       // REQU
#define QN_EPSGH    0x1.12e0bep-30f     // EPSGH
#define QN_EPSGM    0x1.3931eap-34f     // EPSGM
#define QN_AUBH     0x1.e50f2p-6f       // AUBH
#define QN_AUBM     -0x1.a9b742p-2f     // AUBM
#define QN_BUBH     0x1.29de6ap-2f      // BUBH
#define QN_BUBM     -0x1.1e6944p+0f     // BUBM
#define QN_CUBR     0x1.237adp-1f       // CUBR
#define QN_RCUBR    0x1.c1ad8cp+0f      // RCUBR
#define QN_ADNH     0x1.91d84cp-4f      // ADNH
#define QN_ADNM     0x1.f22ce2p-1f      // ADNM
#define QN_BDNH     0x1.adda5ap-1f      // BDNH
#define QN_BDNM     0x1.4e6d84p+1f      // BDNM
#define QN_VKARMAN  0x1.99999ap-2f      // VKARMAN
#define QN_RC02     0x1.a723f6p+2f      // RC02 = 2.0/(C0*C0) (:932)
#define QN_HALF_P608 0x1.377b48p-2f     // 0.5*P608 (:766)

// ---- module_bl_mfshconvpbl.F:8-62 ----
#define MF_XG       0x1.39d014p+3f      // XG = 9.80665
#define MF_XP00     0x1.86ap+16f        // XP00
#define MF_XCPD     0x1.f64p+9f         // XCPD = 7*XRD/2
#define MF_XCPV     0x1.cd999ap+10f     // XCPV = 4*XRV
#define MF_XCL      0x1.07ap+12f        // XCL
#define MF_XCI      0x1.074p+11f        // XCI
#define MF_XTT      0x1.1128f6p+8f      // XTT
#define MF_XLVTT    0x1.3146p+21f       // XLVTT
#define MF_XLSTT    0x1.5a022p+21f      // XLSTT
#define MF_XGAMW    0x1.48d16ap+2f      // XGAMW
#define MF_XBETAW   0x1.aa51d4p+12f     // XBETAW
#define MF_XALPW    0x1.e1ad7cp+5f      // XALPW
#define MF_XLINI    0x1.47ae14p-2f      // XLINI
#define MF_XALP_PERT 0x1.333334p-2f     // XALP_PERT
#define MF_XABUO    0x1.0p+0f           // XABUO
#define MF_XBENTR   0x1.0p+0f           // XBENTR
#define MF_XBDETR   0x0.0p+0f           // XBDETR
#define MF_XCMF     0x1.0a3d7p-4f       // XCMF
#define MF_XENTR_DRY 0x1.19999ap-1f     // XENTR_DRY
#define MF_XDETR_DRY 0x1.4p+3f          // XDETR_DRY
#define MF_XDETR_LUP 0x1.0p+0f          // XDETR_LUP
#define MF_XKCF_HALF 0x1.6p+0f          // 0.5*XKCF_MF (:2531-2534)
#define MF_XFRAC_UP_MAX 0x1.51eb86p-2f  // XFRAC_UP_MAX
#define MF_RDOCPD   0x1.24924ap-2f      // XRD/XCPD
#define MF_RDORV    0x1.3e5612p-1f      // XRD/XRV
#define MF_RVORD    0x1.9bbda4p+0f      // XRV/XRD
#define MF_RDORV_M1 -0x1.8353dcp-2f     // XRD/XRV-1
#define MF_CPV_CL   -0x1.287334p+11f    // XCPV-XCL
#define MF_CPV_CI   -0x1.039998p+8f     // XCPV-XCI
#define MF_C2DRY    0x1.ccccccp-1f      // 2*(XABUO-XBENTR*XENTR_DRY)
#define MF_C05DRY   0x1.1c71c8p+0f      // 0.5/(XABUO-XBENTR*XENTR_DRY)
#define MF_C05BUO   0x1.0p-1f           // 0.5/XABUO
#define MF_C2BUO    0x1.0p+1f           // 2.*(XABUO)
#define MF_CMFCLOUD 0x1.c1e1ccp-8f      // XENTR_MF*XG/XCRAD_MF
#define MF_TWO_THIRDS 0x1.555556p-1f    // 2./3.
#define MF_ONE_THIRD  0x1.555556p-2f    // 1./3.
#define MF_MTWO_THIRDS -0x1.555556p-1f  // -2./3.

// ===================================================================
// MFSHCONVPBL helpers
// ===================================================================

// TH_R_FROM_THL_RT_1D/_2D (module_bl_mfshconvpbl.F:1847-2044): the two
// are the same arithmetic on one element.  20 fixed iterations, warm
// (T > 273.15) arm only; the cold arm returns rt as vapour.
__device__ void mf_th_r_from_thl_rt(float pp, float pthl, float prt,
                                    float* pth_o, float* prv_o,
                                    float* prl_o)
{
    const float zcoef = 0.8f;
    float prl = 0.0f, pri = 0.0f;
    float prv = prt, pth = pthl;
    float zexn = gfk_pow(pp / MF_XP00, MF_RDOCPD);
    for (int ii = 0; ii < 20; ++ii) {
#ifndef GPUWM_QNSE_STRICT
        const unsigned int old_th = __float_as_uint(pth);
        const unsigned int old_rv = __float_as_uint(prv);
        const unsigned int old_rl = __float_as_uint(prl);
        const unsigned int old_ri = __float_as_uint(pri);
#endif
        float zt = FMUL(pth, zexn);
        if (zt > 273.15f) {
            float zfoes = gfk_exp((MF_XALPW - MF_XBETAW / zt)
                                  - FMUL(MF_XGAMW, gfk_log(zt)));
            float zqvsat = (FMUL(MF_RDORV, zfoes) / pp)
                           / (1.0f + FMUL(MF_RDORV_M1, zfoes) / pp);
            float zrvsat = FMUL(FMUL(1.0f - zcoef, zqvsat), 1.0f + prt)
                           + FMUL(zcoef, prv);
            float pfrac_ice = 0.0f;
            float zrltemp = fmaxf(0.0f, prv - zrvsat);
            prv = prv - zrltemp;
            prl = (prl + pri) + zrltemp;
            pri = FMUL(pfrac_ice, prl);
            prl = FMUL(1.0f - pfrac_ice, prt - prv);
            float zcph = ((MF_XCPD + FMUL(MF_XCPV, prv)) + FMUL(MF_XCL, prl))
                         + FMUL(MF_XCI, pri);
            float zlovcpexn = (MF_XLVTT + FMUL(MF_CPV_CL, zt - MF_XTT))
                              / FMUL(zcph, zexn);
            float zloscpexn = (MF_XLSTT + FMUL(MF_CPV_CI, zt - MF_XTT))
                              / FMUL(zcph, zexn);
            pth = (pthl + FMUL(zlovcpexn, prl)) + FMUL(zloscpexn, pri);
        } else {
            prl = 0.0f;
            pri = 0.0f;
            prv = prt;
            pth = pthl;
        }
#ifndef GPUWM_QNSE_STRICT
        // The recurrence has no iteration-dependent inputs. Once all its
        // carried words repeat, the remaining native iterations repeat too.
        // Compare bits (including signed zero), without a tolerance. Keep
        // the strict replay's fixed iteration count and compiled code intact.
        if (__float_as_uint(pth) == old_th && __float_as_uint(prv) == old_rv
                && __float_as_uint(prl) == old_rl
                && __float_as_uint(pri) == old_ri) break;
#endif
    }
    *pth_o = pth;
    *prv_o = prv;
    *prl_o = prl;
}

// COMPUTE_BL89_ML (:476-619) for one column.  pdzz/ptkem/pvpt are the
// column's (IKU) rows; g_kk = PG_O_THVREF2D(:,KK).  The loop's ZTESTM is
// the tile-wide "any column still has energy"; for one column it is that
// column's own ZTEST0, which is what a one-column tile computes.
__device__ float mf_bl89_ml(const float* pdzz, const float* ptkem,
                            float g_kk, const float* pvpt,
                            const float* zdeltvpt, const float* zhlvpt,
                            int kk, int iku, bool up)
{
    const int ike = iku - 1;           // Fortran IKE = SIZE-1 (1-based)
    float zinte = ptkem[kk];
    float plwork = 0.0f;
    float ztestm = 1.0f;
    float zth = pvpt[kk];
    if (up) {
        for (int jkk = kk + 1; jkk <= ike - 1; ++jkk) {   // JKK=KK+1,IKE
            if (!(ztestm > 0.0f)) continue;
            ztestm = 0.0f;
            float ztest0 = 0.5f + copysignf(0.5f, zinte);
            float zpote = FMUL(FMUL(ztest0, FMUL(g_kk, zhlvpt[jkk] - zth)),
                               pdzz[jkk]);
            float ztest = 0.5f + copysignf(0.5f, zinte - zpote);
            ztestm = ztestm + ztest0;
            float zlwork1 = pdzz[jkk];
            float x = FMUL(g_kk, pvpt[jkk - 1] - zth);
            float inner = FMUL(x, x)
                          + FMUL(FMUL(FMUL(2.0f, zinte), g_kk), zdeltvpt[jkk])
                            / pdzz[jkk];
            float zlwork2 = (-x + sqrtf(fabsf(inner)))
                            / (FMUL(g_kk, zdeltvpt[jkk]) / pdzz[jkk]);
            plwork = plwork + FMUL(ztest0, FMUL(ztest, zlwork1)
                                           + FMUL(1.0f - ztest, zlwork2));
            zinte = zinte - zpote;
        }
    } else {
        for (int jkk = kk; jkk >= 0; --jkk) {             // JKK=KK,IKB,-1
            if (!(ztestm > 0.0f)) continue;
            ztestm = 0.0f;
            float ztest0 = 0.5f + copysignf(0.5f, zinte);
            float zpote = -FMUL(FMUL(ztest0, FMUL(g_kk, zhlvpt[jkk] - zth)),
                                pdzz[jkk]);
            float ztest = 0.5f + copysignf(0.5f, zinte - zpote);
            ztestm = ztestm + ztest0;
            float zlwork1 = pdzz[jkk];
            float x = FMUL(g_kk, pvpt[jkk] - zth);
            float inner = FMUL(x, x)
                          + FMUL(FMUL(FMUL(2.0f, zinte), g_kk), zdeltvpt[jkk])
                            / pdzz[jkk];
            float zlwork2 = (x + sqrtf(fabsf(inner)))
                            / (FMUL(g_kk, zdeltvpt[jkk]) / pdzz[jkk]);
            plwork = plwork + FMUL(ztest0, FMUL(ztest, zlwork1)
                                           + FMUL(1.0f - ztest, zlwork2));
            zinte = zinte - zpote;
        }
    }
    return plwork;
}

__device__ void mf_bl89_tables(const float* pvpt, int iku, float* zdeltvpt,
                               float* zhlvpt)
{
    for (int k = 1; k < iku; ++k) zdeltvpt[k] = pvpt[k] - pvpt[k - 1];
    zdeltvpt[0] = 0.0f;
    for (int k = 0; k < iku; ++k) {
        if (fabsf(zdeltvpt[k]) < 1.0e-10f) zdeltvpt[k] = 1.0e-10f;
    }
    for (int k = 1; k < iku; ++k) zhlvpt[k] = FMUL(0.5f, pvpt[k] + pvpt[k - 1]);
    zhlvpt[0] = pvpt[0];
}

// MZM (:1811-1841): flux-level mean, PMZM(1)=PA(2).
#define MF_MZM(a, k) ((k) == 0 ? (a)[1] : FMUL(0.5f, (a)[k] + (a)[(k) - 1]))

// TRIDIAG_MASSFLUX (:2116-2272) at PIMPL = 0 (MFSHCONVPBL_CORE :236), the
// explicit arm.  The PIMPL terms are still formed, so a signed zero they
// would add is added.
__device__ void mf_tridiag_explicit(const float* pvarm, const float* pf,
                                    const float* emf, float ptstep,
                                    const float* pdzz, const float* prhodj,
                                    int iku, float* pvarp)
{
    const float pimpl = 0.0f;
    const int ike = iku - 2;           // zero-based IKE
    // ZRHODJ_DFDT_O_DZ(k) = MZM(PRHODJ)(k)*(-PEMF(k))/PDZZ(k)
#define ZMZMR(k) MF_MZM(prhodj, k)
#define ZRD(k) (FMUL(ZMZMR(k), -emf[k]) / pdzz[k])
#define PIMT(k, v) FMUL(FMUL(FMUL(ZRD(k), 0.5f), pimpl), (v))
    {
        float zy = ((((FMUL(prhodj[0], pvarm[0]) / ptstep
                       - FMUL(ZMZMR(1), pf[1]) / pdzz[1])
                      + FMUL(ZMZMR(0), pf[0]) / pdzz[0])
                     + PIMT(1, pvarm[1]))
                    + PIMT(1, pvarm[0]));
        pvarp[0] = FMUL(zy, ptstep) / prhodj[0];
    }
    for (int k = 1; k <= ike - 1; ++k) {
        float zy = ((((((FMUL(prhodj[k], pvarm[k]) / ptstep
                         - FMUL(ZMZMR(k + 1), pf[k + 1]) / pdzz[k + 1])
                        + FMUL(ZMZMR(k), pf[k]) / pdzz[k])
                       + PIMT(k + 1, pvarm[k + 1]))
                      + PIMT(k + 1, pvarm[k]))
                     - PIMT(k, pvarm[k]))
                    - PIMT(k, pvarm[k - 1]));
        pvarp[k] = FMUL(zy, ptstep) / prhodj[k];
    }
    {
        int k = ike;
        float zy = ((((FMUL(prhodj[k], pvarm[k]) / ptstep
                       - FMUL(ZMZMR(k + 1), pf[k + 1]) / pdzz[k + 1])
                      + FMUL(ZMZMR(k), pf[k]) / pdzz[k])
                     - PIMT(k, pvarm[k]))
                    - PIMT(k, pvarm[k - 1]));
        pvarp[k] = FMUL(zy, ptstep) / prhodj[k];
    }
    pvarp[ike + 1] = pvarp[ike];
#undef PIMT
#undef ZRD
#undef ZMZMR
}

// COMPUTE_ENTR_DETR (:624-1011) for one column at zero-based mass level kk.
// otest/otestlcl are the column's GTEST/GTESTLCL; pentr/pdetr are
// PENTR(:,KK)/PDETR(:,KK) on entry and exit; pbuo is PBUO_INTEG.
__device__ void mf_entr_detr(bool* otest, bool otestlcl, int kk,
                             const float* ppabsm, const float* pzz,
                             const float* pdzz, const float* pthvm,
                             float pthlm, float prtm, const float* w2,
                             float pthl_up, float prt_up, float plup,
                             float* pentr, float* pdetr, float* pbuo)
{
    float zg_kk = MF_XG / pthvm[kk];
    float zcoeff_qsat = 0.0f;
    float zrc_ord = 0.0f;
    float zpart_dry = 1.0f;
    bool lcl = false, lcl2 = false;
    float zdz_half = (pzz[kk + 1] - pzz[kk]) / 2.0f;
    float zdz_stop = zdz_half;
    float zkic = 0.1f;
    float zmixthl = FMUL(zkic, pthlm) + FMUL(1.0f - zkic, pthl_up);
    float zmixrt = FMUL(zkic, prtm) + FMUL(1.0f - zkic, prt_up);
    float zcopre_m = (kk == 0) ? 0.0f
                     : ((ppabsm[kk] - ppabsm[kk - 1]) / pdzz[kk]);
    float zcopre_p = ((ppabsm[kk + 1] - ppabsm[kk]) / pdzz[kk + 1]);
    float zpre_m = (kk == 0) ? ppabsm[kk]
                   : FMUL(FMUL(zcopre_m, 0.5f), pzz[kk] - pzz[kk - 1])
                     + ppabsm[kk - 1];
    float zpre_p = FMUL(FMUL(zcopre_p, 0.5f), pzz[kk + 1] - pzz[kk])
                   + ppabsm[kk];
    float zthmix, zrvmix, zrcmix;
    mf_th_r_from_thl_rt(ppabsm[kk], zmixthl, zmixrt, &zthmix, &zrvmix, &zrcmix);
    float zthvmix_1 = FMUL(zthmix, 1.0f + FMUL(MF_RVORD, zrvmix))
                      / (1.0f + zmixrt);
    mf_th_r_from_thl_rt(zpre_p, zmixthl, zmixrt, &zthmix, &zrvmix, &zrcmix);
    float zthvmix_2 = FMUL(zthmix, 1.0f + FMUL(MF_RVORD, zrvmix))
                      / (1.0f + zmixrt);
    mf_th_r_from_thl_rt(zpre_m, pthl_up, prt_up, &zthmix, &zrvmix, &zrcmix);
    float zthv_up_f1 = FMUL(zthmix, 1.0f + FMUL(MF_RVORD, zrvmix))
                       / (1.0f + prt_up);
    mf_th_r_from_thl_rt(ppabsm[kk], pthl_up, prt_up, &zthmix, &zrvmix, &zrcmix);
    float zthv_up = FMUL(zthmix, 1.0f + FMUL(MF_RVORD, zrvmix))
                    / (1.0f + prt_up);
    float zthmix_f2, zrvmix_f2, zrcmix_f2;
    mf_th_r_from_thl_rt(zpre_p, pthl_up, prt_up, &zthmix_f2, &zrvmix_f2,
                        &zrcmix_f2);
    float zthv_up_f2 = FMUL(zthmix_f2, 1.0f + FMUL(MF_RVORD, zrvmix_f2))
                       / (1.0f + prt_up);

    // :845-855
    if ((zrcmix > 0.0f) && !otestlcl) {
        zcoeff_qsat = (zrcmix_f2 - zrcmix) / zdz_half;
        zrc_ord = zrcmix - FMUL(zcoeff_qsat, zdz_half);
        zdz_stop = -(zrc_ord / zcoeff_qsat);
        zpart_dry = zdz_stop / (pzz[kk + 1] - pzz[kk]);
        lcl = true;
    }
    float zcoeff_m = (kk == 0) ? 0.0f : ((pthvm[kk] - pthvm[kk - 1]) / pdzz[kk]);
    float zcoeff_p = ((pthvm[kk + 1] - pthvm[kk]) / pdzz[kk + 1]);
    float zcothvu_m = (zthv_up - zthv_up_f1) / zdz_half;
    float zcothvu_p = (zthv_up_f2 - zthv_up) / zdz_half;
    float zthv_m = (kk == 0) ? pthvm[kk]
                   : FMUL(FMUL(zcoeff_m, 0.5f), pzz[kk] - pzz[kk - 1])
                     + pthvm[kk - 1];
    float zthv_p = FMUL(FMUL(zcoeff_p, 0.5f), pzz[kk + 1] - pzz[kk]) + zthv_m;
    // :878-880
    *pbuo = FMUL(FMUL(zg_kk, zdz_half),
                 (FMUL(FMUL(0.5f, zcothvu_m - zcoeff_m), zdz_half) - zthv_m)
                 + zthv_up_f1);
    float zbuo_integ = 0.0f;
    float zw2_half = 0.0f;
    if (*otest && !otestlcl) {                                   // :884-910
        *pentr = 0.0f;
        *pdetr = 0.0f;
        zbuo_integ = FMUL(FMUL(zg_kk, zdz_stop),
                          (FMUL(FMUL(0.5f, -zcoeff_m), zdz_stop) - zthv_m)
                          + zthv_up_f1);
        if (zbuo_integ >= 0.0f) {
            *pentr = FMUL(MF_C05DRY,
                          gfk_log(1.0f + FMUL(MF_C2DRY / w2[kk], zbuo_integ)));
            *pdetr = 0.0f;
            zw2_half = w2[kk] + FMUL(MF_C2DRY, zbuo_integ);
        } else {
            *pentr = 0.0f;
            *pdetr = FMUL(MF_C05BUO,
                          gfk_log(1.0f + FMUL(MF_C2BUO / w2[kk],
                                              fmaxf(0.0f, -zbuo_integ))));
            zw2_half = w2[kk] + FMUL(MF_C2BUO, zbuo_integ);
        }
    }
    zdz_stop = zdz_half;
    // :917-919
    *pbuo = *pbuo + FMUL(FMUL(zg_kk, zdz_half),
                         (FMUL(FMUL(0.5f, zcothvu_p - zcoeff_p), zdz_half)
                          - pthvm[kk]) + zthv_up);
    // :922-936
    if (((zrcmix_f2 > 0.0f) && (zrcmix <= 0.0f)) && !otestlcl && !lcl) {
        float pt = FMUL(zthmix_f2, gfk_pow(ppabsm[kk + 1] / MF_XP00, MF_RDOCPD));
        float zqvsat = gfk_exp((MF_XALPW - MF_XBETAW / pt)
                               - FMUL(MF_XGAMW, gfk_log(pt)));
        zqvsat = (FMUL(MF_RDORV, zqvsat) / ppabsm[kk + 1])
                 / (1.0f + FMUL(MF_RDORV_M1, zqvsat) / ppabsm[kk + 1]);
        zcoeff_qsat = ((prt_up - zqvsat) - zrcmix)
                      / FMUL(0.5f, pzz[kk + 2] - pzz[kk + 1]);
        zrc_ord = zrcmix_f2 - FMUL(zcoeff_qsat, zdz_half);
        zdz_stop = -(zrc_ord / zcoeff_qsat);
        zpart_dry = 0.5f + zdz_stop / (pzz[kk + 1] - pzz[kk]);
        lcl2 = true;
    }
    // :939-968
    if ((*otest && !otestlcl) && !lcl) {
        zbuo_integ = FMUL(FMUL(zg_kk, zdz_stop),
                          (FMUL(FMUL(0.5f, -zcoeff_p), zdz_stop) - pthvm[kk])
                          + zthv_up);
        if (zw2_half > 0.0f) {
            if (zbuo_integ >= 0.0f) {
                *pentr = *pentr + FMUL(MF_C05DRY,
                    gfk_log(1.0f + FMUL(MF_C2DRY / zw2_half, zbuo_integ)));
            } else {
                *pdetr = *pdetr + FMUL(MF_C05BUO,
                    gfk_log(1.0f + FMUL(MF_C2BUO / zw2_half,
                                        fmaxf(-zbuo_integ, 0.0f))));
            }
        } else {
            *otest = false;
        }
    }
    // :969-971
    *pentr = FMUL(MF_XENTR_DRY, *pentr) / (pzz[kk + 1] - pzz[kk]);
    *pdetr = FMUL(MF_XDETR_DRY, *pdetr) / (pzz[kk + 1] - pzz[kk]);
    *pdetr = fmaxf(FMUL(zpart_dry, MF_XDETR_LUP)
                   / (plup - FMUL(0.5f, pzz[kk] + pzz[kk + 1])), *pdetr);
    // :979-991
    if (*otest && otestlcl) {
        zkic = FMUL(fmaxf(0.0f, zthv_up - pthvm[kk]), zkic)
               / ((zthv_up - zthvmix_1) + 1.0e-10f);
        zkic = fmaxf(0.0f, fminf(1.0f, zkic));
        float zepsi = FMUL(zkic, zkic);
        float zdelta = FMUL(1.0f - zkic, 1.0f - zkic);
        float zepsi_cloud = fminf(zdelta, zepsi);
        *pentr = FMUL(MF_CMFCLOUD, zepsi_cloud);
        *pdetr = FMUL(MF_CMFCLOUD, zdelta);
    }
    // :997-1007
    if ((*otest && !otestlcl) && (lcl || lcl2)) {
        zkic = FMUL(fmaxf(0.0f, zthv_up_f2 - zthv_p), zkic)
               / ((zthv_up_f2 - zthvmix_2) + 1.0e-10f);
        zkic = fmaxf(0.0f, fminf(1.0f, zkic));
        float zepsi = FMUL(zkic, zkic);
        float zdelta = FMUL(1.0f - zkic, 1.0f - zkic);
        float zepsi_cloud = fminf(zdelta, zepsi);
        *pentr = *pentr + FMUL(FMUL(1.0f - zpart_dry, MF_CMFCLOUD), zepsi_cloud);
        *pdetr = *pdetr + FMUL(FMUL(1.0f - zpart_dry, MF_CMFCLOUD), zdelta);
    }
}

extern "C" __global__ void mfshconv_column(
    // (nz, n) bottom-up inputs
    const float* __restrict__ dz_a, const float* __restrict__ rho_a,
    const float* __restrict__ p_a, const float* __restrict__ th_a,
    const float* __restrict__ exner_a, const float* __restrict__ qv_a,
    const float* __restrict__ u_a, const float* __restrict__ v_a,
    const float* __restrict__ tke_a,
    // (n) surface inputs
    const float* __restrict__ hfx_a, const float* __restrict__ qfx_a,
    // (nz, n) carried state: RC_MF, read then rewritten
    float* __restrict__ rc_mf_a,
    // (nz, n) tendencies, ASSIGNED (module_bl_mfshconvpbl.F:256-260)
    float* __restrict__ rublten_a, float* __restrict__ rvblten_a,
    float* __restrict__ rthblten_a, float* __restrict__ rqvblten_a,
    float* __restrict__ rqcblten_a,
    // (nz, n) EDKF outputs (:271-282)
    float* __restrict__ massflux_a, float* __restrict__ entr_a,
    float* __restrict__ detr_a, float* __restrict__ thl_up_a,
    float* __restrict__ thv_up_a, float* __restrict__ rt_up_a,
    float* __restrict__ rv_up_a, float* __restrict__ rc_up_a,
    float* __restrict__ u_up_a, float* __restrict__ v_up_a,
    float* __restrict__ frac_up_a,
    // (nz, n) diagnostics WRF hands to QNSEPBL but QNSEPBL never uses;
    // written only when diag != 0 (the BL89 length is O(nz^2) per column).
    float* __restrict__ wthv_a, float* __restrict__ lm_bl89_a,
    float ptstep, int diag, int nz, int n, int* tile_top, int tile_mode)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    if (nz > QNSE_KMAX || nz < 4) return;      // launcher refuses these
    size_t st = (size_t)n;
#define MF_AT(a, k) a[(size_t)(k) * st + col]
    const int iku = nz;

    float pzz[QNSE_KMAX], pzzm[QNSE_KMAX], pdzz[QNSE_KMAX];
    float pthm[QNSE_KMAX], ptkem[QNSE_KMAX], prm1[QNSE_KMAX], prm2[QNSE_KMAX];
    float pum[QNSE_KMAX], pvm[QNSE_KMAX], prhodref[QNSE_KMAX];
    float pexnm[QNSE_KMAX], ppabsm[QNSE_KMAX], prhodj[QNSE_KMAX];
    float zthlm[QNSE_KMAX], zrtm[QNSE_KMAX], zthvm[QNSE_KMAX];
    float thl_up[QNSE_KMAX], rt_up[QNSE_KMAX], rv_up[QNSE_KMAX];
    float rc_up[QNSE_KMAX], thv_up[QNSE_KMAX], w2[QNSE_KMAX];
    float emf[QNSE_KMAX], entr[QNSE_KMAX], detr[QNSE_KMAX], frac[QNSE_KMAX];
    float zthlm_f[QNSE_KMAX], zrtm_f[QNSE_KMAX], zthvm_f[QNSE_KMAX];
    float s1[QNSE_KMAX], s2[QNSE_KMAX], s3[QNSE_KMAX];

    // ---- MFSHCONVPBL wrapper, :170-227 ----------------------------------
    pzz[0] = 0.0f;
    for (int k = 0; k < iku; ++k) {
        pthm[k] = MF_AT(th_a, k);
        ptkem[k] = MF_AT(tke_a, k);
        float qv = MF_AT(qv_a, k);
        float rcmf = MF_AT(rc_mf_a, k);
        prm1[k] = qv - rcmf;
        prm2[k] = rcmf;
        pum[k] = MF_AT(u_a, k);
        pvm[k] = MF_AT(v_a, k);
        prhodref[k] = MF_AT(rho_a, k) / (1.0f + qv);
        pexnm[k] = MF_AT(exner_a, k);
        ppabsm[k] = MF_AT(p_a, k);
        float dzk = MF_AT(dz_a, k);
        if (k != iku - 1) {
            pzz[k + 1] = pzz[k] + dzk;
            pzzm[k] = FMUL(0.5f, pzz[k + 1] + pzz[k]);
        } else {
            pzzm[k] = pzz[k] + FMUL(0.5f, MF_AT(dz_a, k - 1));
        }
        pdzz[k] = (k == 0) ? FMUL(2.0f, pzzm[k]) : (pzzm[k] - pzzm[k - 1]);
        prhodj[k] = FMUL(prhodref[k], dzk);
    }
    {   // fill the kte+1 level (:215-224)
        int t = iku - 1, b = iku - 2;
        pthm[t] = pthm[b]; ptkem[t] = ptkem[b]; prm1[t] = prm1[b];
        prm2[t] = prm2[b]; pum[t] = pum[b]; pvm[t] = pvm[b];
        prhodref[t] = prhodref[b]; pexnm[t] = pexnm[b];
        ppabsm[t] = ppabsm[b]; prhodj[t] = prhodj[b];
    }
    // :226-227 (PRHODREF(KTS) is never overwritten by the copy above)
    const float psfth = hfx_a[col] / FMUL(prhodref[0], MF_XCPD);
    const float psfrv = qfx_a[col] / prhodref[0];

    // ---- MFSHCONVPBL_CORE, :391-418 -------------------------------------
    for (int k = 0; k < iku; ++k) {
        // COMPUTE_FUNCTION_THERMO_MF (:1080-1103, :1194): KRR=2, KRRL=1
        float zcp = MF_XCPD;
        zcp = zcp + FMUL(MF_XCPV, prm1[k]);
        zcp = zcp + FMUL(MF_XCL, prm2[k]);
        float pt = FMUL(pthm[k], pexnm[k]);
        float lvocpexn = (MF_XLVTT + FMUL(MF_CPV_CL, pt - MF_XTT)) / zcp;
        lvocpexn = lvocpexn / pexnm[k];
        // THL_RT_FROM_TH_R_MF (:2101-2103)
        zrtm[k] = prm1[k] + prm2[k];
        zthlm[k] = pthm[k] - FMUL(lvocpexn, prm2[k]);
        // :413
        zthvm[k] = FMUL(pthm[k], (1.0f + FMUL(MF_RVORD, prm1[k]))
                                 / (1.0f + zrtm[k]));
    }
    if (diag && tile_mode != 1) {
        // BL89 (:2599-2762).  s1 = ZVPT, s2 = ZG_O_THVREF, s3/zthlm_f scratch.
        for (int k = 0; k < iku; ++k) {
            float zthvref = MF_XG / zthvm[k];
            s2[k] = MF_XG / zthvref;
            float zsum = 0.0f;
            zsum = zsum + prm1[k];
            zsum = zsum + prm2[k];
            s1[k] = FMUL(zthlm[k], 1.0f + FMUL(MF_RVORD, prm1[k]))
                    / (1.0f + zsum);
        }
        float* zdelt = s3;
        float* zhl = zthlm_f;              // reused before the updraft fills it
        mf_bl89_tables(s1, iku, zdelt, zhl);
        float lm[QNSE_KMAX];
        const int ike0 = iku - 2;          // zero-based IKE
        for (int jk = 0; jk <= ike0; ++jk) {
            float dn = mf_bl89_ml(pdzz, ptkem, s2[jk], s1, zdelt, zhl, jk,
                                  iku, false);
            float zlmdn = fminf(dn, FMUL(0.5f, pzz[jk] + pzz[jk + 1]) - pzz[0]);
            float zlmup = mf_bl89_ml(pdzz, ptkem, s2[jk], s1, zdelt, zhl, jk,
                                     iku, true);
            float a = gfk_pow(fmaxf(zlmdn, 1.0e-10f), MF_MTWO_THIRDS);
            float b = gfk_pow(fmaxf(zlmup, 1.0e-10f), MF_MTWO_THIRDS);
            float zlm = gfk_pow(FMUL(0.5f, a + b), -1.5f);
            lm[jk] = fmaxf(zlm, MF_XLINI);
        }
        lm[ike0] = lm[ike0 - 1];
        lm[ike0 + 1] = lm[ike0 - 1];
        for (int k = 0; k < iku; ++k) MF_AT(lm_bl89_a, k) = lm[k];
    }

    // ---- COMPUTE_UPDRAFT (:1202-1619), KK = IKB = 1 ---------------------
    // Flux-level environment (:1352-1377).  s1 = ZTKEM_F, s2 = ZRHO_F,
    // s3 = ZPRES_F; ZTHM_F/ZRVM_F/ZUM_F/ZVM_F are formed where read.
    float zthm_f0 = pthm[0];
    float zrvm_f0 = prm1[0];
    for (int k = 0; k < iku; ++k) {
        if (k == 0) {
            zthlm_f[k] = zthlm[0]; zrtm_f[k] = zrtm[0]; s1[k] = ptkem[0];
            s3[k] = ppabsm[0]; s2[k] = prhodref[0];
            zthvm_f[k] = FMUL(zthm_f0, (1.0f + FMUL(MF_RVORD, zrvm_f0))
                                       / (1.0f + zrtm_f[k]));
            thl_up[k] = zthlm_f[k];
            rt_up[k] = zrtm_f[k];
            MF_AT(u_up_a, k) = pum[0];
            MF_AT(v_up_a, k) = pvm[0];
        } else {
            float zthm_f = FMUL(0.5f, pthm[k - 1] + pthm[k]);
            float zrvm_f = FMUL(0.5f, prm1[k - 1] + prm1[k]);
            zthlm_f[k] = FMUL(0.5f, zthlm[k - 1] + zthlm[k]);
            zrtm_f[k] = FMUL(0.5f, zrtm[k - 1] + zrtm[k]);
            s1[k] = FMUL(0.5f, ptkem[k - 1] + ptkem[k]);
            s3[k] = FMUL(0.5f, ppabsm[k - 1] + ppabsm[k]);
            s2[k] = FMUL(0.5f, prhodref[k - 1] + prhodref[k]);
            zthvm_f[k] = FMUL(zthm_f, (1.0f + FMUL(MF_RVORD, zrvm_f))
                                      / (1.0f + zrtm_f[k]));
            thl_up[k] = zthlm_f[k];
            rt_up[k] = zrtm_f[k];
            MF_AT(u_up_a, k) = FMUL(0.5f, pum[k - 1] + pum[k]);
            MF_AT(v_up_a, k) = FMUL(0.5f, pvm[k - 1] + pvm[k]);
        }
        rv_up[k] = 0.0f; rc_up[k] = 0.0f; w2[k] = 0.0f; emf[k] = 0.0f;
        detr[k] = 0.0f; entr[k] = 0.0f; frac[k] = 0.0f;
        thv_up[k] = zthvm_f[k];
    }
    // ZTHVM at mass levels with the updraft's own ZRVORD (:1377)
    for (int k = 0; k < iku; ++k) {
        zthvm[k] = FMUL(pthm[k], (1.0f + FMUL(MF_RVORD, prm1[k]))
                                 / (1.0f + zrtm[k]));
    }
    int kklcl = iku - 2, kketl = iku - 2, kkctl = iku - 2;   // IKE
    (void)kketl;
    const float ztmax = 2.0f, zrmax = 1.0e-3f;
    // :1396-1401
    thl_up[0] = zthlm_f[0] + fmaxf(0.0f, fminf(ztmax,
                    FMUL(psfth / sqrtf(s1[0]), MF_XALP_PERT)));
    rt_up[0] = zrtm_f[0] + fmaxf(0.0f, fminf(zrmax,
                    FMUL(psfrv / sqrtf(s1[0]), MF_XALP_PERT)));
    w2[0] = fmaxf(0.0001f, FMUL(MF_TWO_THIRDS, s1[0]));
    {
        float th_up, rl;
        mf_th_r_from_thl_rt(s3[0], thl_up[0], rt_up[0], &th_up, &rv_up[0], &rl);
        rc_up[0] = rl;
        thv_up[0] = FMUL(th_up, (1.0f + FMUL(MF_RVORD, rv_up[0]))
                                / (1.0f + rt_up[0]));
    }
    // ZG_O_THVREF = XG/ZTHVM_F, held in pzzm (dead since PDZZ was built).
    for (int k = 0; k < iku; ++k) pzzm[k] = MF_XG / zthvm_f[k];
    // L_up from the ground with ZTKEM_F(KK) = 0 (:1417-1421)
    float zlup;
    {
        s1[0] = 0.0f;
        // ZDELTVPT/ZHLVPT of ZTHVM_F, in emf/entr rows (still all zero
        // and re-zeroed right after).
        mf_bl89_tables(zthvm_f, iku, emf, entr);
        zlup = mf_bl89_ml(pdzz, s1, pzzm[0], zthvm_f, emf, entr, 0, iku, true);
        zlup = fmaxf(zlup, 1.0e-10f);
        for (int k = 0; k < iku; ++k) { emf[k] = 0.0f; entr[k] = 0.0f; }
    }
    float zwthvsurf = FMUL(zthvm_f[0] / zthm_f0, psfth)
                      + FMUL(FMUL(0.61f, zthm_f0), psfrv);
    bool gtest;
    if (zwthvsurf > 0.0f) {
        emf[0] = FMUL(FMUL(MF_XCMF, s2[0]),
                      gfk_pow(FMUL(FMUL(pzzm[0], zwthvsurf), zlup), MF_ONE_THIRD));
        frac[0] = fminf(emf[0] / FMUL(sqrtf(w2[0]), s2[0]), MF_XFRAC_UP_MAX);
        float x = emf[0] / FMUL(frac[0], s2[0]);
        w2[0] = FMUL(x, x);
        gtest = true;
    } else {
        emf[0] = 0.0f;
        gtest = false;
    }
    bool gtestlcl = false, gtestetl = false;
    float zbuo_integ = 0.0f;
    float zmix2 = 0.0f, zmix3 = 0.0f;
    const int ike0 = iku - 2;
    int processed = 0;
    for (int jk = 0; jk <= ike0 - 1; ++jk) {             // JK=KK,IKE-1
        // Native COUNT(GTEST) spans the complete horizontal WRF tile.
        // Pass 1 obtains its last active level without changing carried
        // state; pass 2 repeats the updraft through that device maximum.
        if (tile_mode == 2 ? jk >= *tile_top : !gtest) continue;
        processed = jk + 1;
        if (gtest && rc_up[jk] > 0.0f && !gtestlcl) {             // :1462-1467
            kklcl = jk;
            gtestlcl = true;
        }
        mf_entr_detr(&gtest, gtestlcl, jk, ppabsm, pzz, pdzz, zthvm,
                     zthlm[jk], zrtm[jk], w2, thl_up[jk], rt_up[jk], zlup,
                     &entr[jk], &detr[jk], &zbuo_integ);
        if (jk == 0) detr[jk] = 0.0f;                    // :1479-1481
        if (gtest) {                                     // :1485-1488
            float zmix1 = FMUL(FMUL(0.5f, pzz[jk + 1] - pzz[jk]),
                               entr[jk] - detr[jk]);
            emf[jk + 1] = FMUL(emf[jk], gfk_exp(FMUL(2.0f, zmix1)));
        }
        if (gtest && (emf[jk + 1] <= 0.0f)) {            // :1493-1497
            emf[jk + 1] = 0.0f;
            gtest = false;
            kkctl = jk + 1;
        }
        if (gtest) {                                     // :1501-1509
            zmix2 = FMUL(pzz[jk + 1] - pzz[jk], entr[jk]);
            zmix3 = FMUL(pzz[jk + 1] - pzz[jk], detr[jk]);
            thl_up[jk + 1] = (FMUL(thl_up[jk], 1.0f - FMUL(0.5f, zmix2))
                              + FMUL(zthlm[jk], zmix2))
                             / (1.0f + FMUL(0.5f, zmix2));
            rt_up[jk + 1] = (FMUL(rt_up[jk], 1.0f - FMUL(0.5f, zmix2))
                             + FMUL(zrtm[jk], zmix2))
                            / (1.0f + FMUL(0.5f, zmix2));
        }
        float th_up1, rl1;                               // :1528-1530
        mf_th_r_from_thl_rt(s3[jk + 1], thl_up[jk + 1], rt_up[jk + 1],
                            &th_up1, &rv_up[jk + 1], &rl1);
        rc_up[jk + 1] = rl1;
        if (gtest) {                                     // :1534-1549
            thv_up[jk + 1] = FMUL(th_up1, (1.0f + FMUL(MF_RVORD, rv_up[jk + 1]))
                                          / (1.0f + rt_up[jk + 1]));
            if (!gtestlcl) {
                if (zbuo_integ > 0.0f)
                    w2[jk + 1] = w2[jk] + FMUL(MF_C2DRY, zbuo_integ);
                if (zbuo_integ <= 0.0f)
                    w2[jk + 1] = w2[jk] + FMUL(MF_C2BUO, zbuo_integ);
            }
            if (gtestlcl) {
                float s = FMUL(MF_XBDETR, zmix3) + FMUL(MF_XBENTR, zmix2);
                w2[jk + 1] = FMUL(w2[jk], 1.0f - s) / (1.0f + s)
                             + FMUL(MF_C2BUO, zbuo_integ) / (1.0f + s);
            }
        }
        gtestetl = false;                                // :1553-1557
        if (gtest && (zbuo_integ <= 0.0f)) {
            kketl = jk + 1;
            gtestetl = true;
        }
        if (gtest && ((w2[jk + 1] <= 0.0f) || (emf[jk + 1] <= 0.0f))) {
            w2[jk + 1] = 0.0f;                           // :1560-1571
            emf[jk + 1] = 0.0f;
            gtest = false;
            thl_up[jk + 1] = zthlm_f[jk + 1];
            rt_up[jk + 1] = zrtm_f[jk + 1];
            rc_up[jk + 1] = 0.0f;
            rv_up[jk + 1] = 0.0f;
            thv_up[jk + 1] = zthvm_f[jk + 1];
            frac[jk + 1] = 0.0f;
            kkctl = jk + 1;
        }
        if (gtest) {                                     // :1574-1581
            frac[jk + 1] = emf[jk + 1] / FMUL(sqrtf(w2[jk + 1]), s2[jk + 1]);
            frac[jk + 1] = fminf(MF_XFRAC_UP_MAX, frac[jk + 1]);
        }
        if ((gtest && gtestetl) && gtestlcl)             // :1586-1588
            frac[jk + 1] = fminf(frac[jk + 1], frac[jk]);
        emf[jk + 1] = FMUL(FMUL(frac[jk + 1], sqrtf(w2[jk + 1])), s2[jk + 1]);
    }
    if (tile_mode == 1) {
        const unsigned mask = __activemask();
        const int lane = threadIdx.x & 31;
        for (int offset = 16; offset; offset >>= 1) {
            int other = __shfl_down_sync(mask, processed, offset);
            if (lane + offset < 32 && (mask & (1u << (lane + offset))))
                processed = max(processed, other);
        }
        if (lane == 0) atomicMax(tile_top, processed);
        return;
    }
    emf[0] = 0.0f;                                       // :1599-1601
    emf[iku - 1] = 0.0f;
    frac[iku - 1] = 0.0f;
    {
        float zdepth = pzz[kkctl] - pzz[kklcl];          // :1604-1616
        float zcoef = 1.0f - (zdepth - 3000.0f) / 1000.0f;
        zcoef = fminf(fmaxf(zcoef, 0.0f), 1.0f);
        if (gtestlcl && (zdepth > 3000.0f)) {
            for (int k = 0; k < iku; ++k) {
                emf[k] = FMUL(emf[k], zcoef);
                frac[k] = FMUL(frac[k], zcoef);
            }
        }
    }

    // ---- MF_TURB (:1624-1809), OMIXUV = .FALSE., PIMPL = 0 --------------
    // WRF core divides mass flux by dry density before MF_TURB (:429-437).
    // s1 = flux, s2 = ZVARS, s3 = ZEMF_O_RHODREF.
    for (int k = 0; k < iku; ++k) s3[k] = emf[k] / prhodref[k];
    for (int k = 0; k < iku; ++k)
        s1[k] = FMUL(s3[k], thl_up[k] - MF_MZM(zthlm, k));
    mf_tridiag_explicit(zthlm, s1, s3, ptstep, pdzz, prhodj, iku, s2);
    for (int k = 0; k < iku; ++k)
        MF_AT(rthblten_a, k) = (s2[k] - zthlm[k]) / ptstep;
    for (int k = 0; k < iku; ++k)
        s1[k] = FMUL(s3[k], rt_up[k] - MF_MZM(zrtm, k));
    mf_tridiag_explicit(zrtm, s1, s3, ptstep, pdzz, prhodj, iku, s2);
    for (int k = 0; k < iku; ++k)
        MF_AT(rqvblten_a, k) = (s2[k] - zrtm[k]) / ptstep;
    if (diag) {
        for (int k = 0; k < iku; ++k)
            MF_AT(wthv_a, k) = FMUL(FMUL(9.81f / zthvm[k], s3[k]),
                                    thv_up[k] - MF_MZM(zthvm, k));
    }

    // ---- COMPUTE_MF_CLOUD (:2444-2542) and the outputs (:253-286) -------
    for (int k = 0; k < iku; ++k) {
        float rcmf = 0.0f;
        if (k >= kklcl && k <= ike0) {
            rcmf = FMUL(MF_XKCF_HALF, FMUL(frac[k], rc_up[k])
                                      + FMUL(frac[k + 1], rc_up[k + 1]));
        }
        MF_AT(rc_mf_a, k) = rcmf;
        MF_AT(rqcblten_a, k) = 0.0f;
        MF_AT(rublten_a, k) = 0.0f;
        MF_AT(rvblten_a, k) = 0.0f;
        MF_AT(massflux_a, k) = emf[k];
        MF_AT(entr_a, k) = entr[k];
        MF_AT(detr_a, k) = detr[k];
        MF_AT(thl_up_a, k) = thl_up[k];
        MF_AT(thv_up_a, k) = thv_up[k];
        MF_AT(rt_up_a, k) = rt_up[k];
        MF_AT(rv_up_a, k) = rv_up[k];
        MF_AT(rc_up_a, k) = rc_up[k];
        MF_AT(frac_up_a, k) = frac[k];
    }
#undef MF_AT
}

// ===================================================================
// QNSEPBL (module_bl_qnsepbl.F:142-660)
// ===================================================================
extern "C" __global__ void qnsepbl_column(
    // (nz, n) bottom-up inputs
    const float* __restrict__ dz_a,
    const float* __restrict__ u_a, const float* __restrict__ v_a,
    const float* __restrict__ t_a, const float* __restrict__ th_a,
    const float* __restrict__ exner_a, const float* __restrict__ qv_a,
    const float* __restrict__ qc_a, const float* __restrict__ p_a,
    float* __restrict__ tke_a,              // TKE_PBL, inout (nz, n)
    // (n) surface inputs
    const float* __restrict__ ht_a, const float* __restrict__ psfc_a,
    const float* __restrict__ ust_a, const float* __restrict__ chklowq_a,
    const float* __restrict__ xland_a, const float* __restrict__ sice_a,
    const float* __restrict__ snow_a, const float* __restrict__ akhs_a,
    const float* __restrict__ akms_a, const float* __restrict__ elflx_a,
    const float* __restrict__ hfx_a, const float* __restrict__ uz0_a,
    const float* __restrict__ vz0_a, const float* __restrict__ corf_a,
    // (n) inout surface state
    float* __restrict__ thz0_a, float* __restrict__ qz0_a,
    float* __restrict__ qsfc_a, float* __restrict__ ct_a,
    // (nz, n) tendencies, ACCUMULATED (:543-545, :649-650)
    float* __restrict__ rublten_a, float* __restrict__ rvblten_a,
    float* __restrict__ rthblten_a, float* __restrict__ rqvblten_a,
    float* __restrict__ rqcblten_a,
    // (nz, n) outputs: EL_MYJ written whole; EXCH_H/EXCH_M levels 1..nz-1
    // (level 0 is INOUT and left as it was, :393-400)
    float* __restrict__ el_a, float* __restrict__ exch_h_a,
    float* __restrict__ exch_m_a,
    // (n) outputs
    float* __restrict__ pblh_a, int* __restrict__ kpbl_a,
    float dtturbl, int nz, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    if (nz > QNSE_KMAX || nz < 4) return;      // launcher refuses these
    size_t st = (size_t)n;
#define QN_UP(a, kup) a[(size_t)(kup) * st + col]

    const int lmh = nz;
    const int nzm = nz - 1;
    const float rdtturbl = 1.0f / dtturbl;
    const float dtdif = dtturbl;

    float zh[QNSE_KMAX + 1];
    float uk[QNSE_KMAX], vk[QNSE_KMAX], tk[QNSE_KMAX];
    float the[QNSE_KMAX], qk[QNSE_KMAX], cwm[QNSE_KMAX];
    float q2[QNSE_KMAX], rhok[QNSE_KMAX];
    float s2v[QNSE_KMAX], ri[QNSE_KMAX], en2[QNSE_KMAX], el[QNSE_KMAX];
    float akm[QNSE_KMAX], akh[QNSE_KMAX];
    float s1[QNSE_KMAX], s2[QNSE_KMAX], s3[QNSE_KMAX], s4[QNSE_KMAX];

    // ---- setup_integration (:271-350) -----------------------------------
    zh[nz] = ht_a[col];
    for (int m = nz - 1; m >= 0; --m) zh[m] = zh[m + 1] + QN_UP(dz_a, nz - 1 - m);
    for (int m = 0; m < nz; ++m) {
        int kup = nz - 1 - m;
        float tx = QN_UP(t_a, kup);
        tk[m] = tx;
        float cw = QN_UP(qc_a, kup);
        the[m] = FMUL(FMUL(cw, -QN_ELOCP / tx) + 1.0f, QN_UP(th_a, kup));
        float ratiomx = QN_UP(qv_a, kup);
        qk[m] = ratiomx / (1.0f + ratiomx);
        cwm[m] = cw;
        uk[m] = QN_UP(u_a, kup);
        vk[m] = QN_UP(v_a, kup);
        q2[m] = FMUL(2.0f, QN_UP(tke_a, kup));
    }

    // ================= MIXLEN (:665-882) ================================
    int lpbl = lmh;
    for (int kf = lmh - 1; kf >= 1; --kf) {
        if (q2[kf - 1] <= FMUL(QN_EPSQ2L, QN_FH)) { lpbl = kf; goto mixlen_110; }
    }
    lpbl = 1;
mixlen_110:
    pblh_a[col] = zh[lpbl - 1] - zh[lmh];            // :732 Z(LPBL)-Z(LMH+1)
    // DTH in s1 (:739-748)
    for (int m = 0; m < nzm; ++m) s1[m] = the[m] - the[m + 1];
    {
        float ct = ct_a[col];
        for (int kf = lmh - 2; kf >= 1; --kf) {
            if (s1[kf - 1] > 0.0f && s1[kf] <= 0.0f) {
                s1[kf - 1] = s1[kf - 1] + ct;
                break;
            }
        }
        ct_a[col] = 0.0f;                            // :750
    }
    for (int m = 0; m < nzm; ++m) {                  // :754-776
        float rdz = 2.0f / (zh[m] - zh[m + 2]);
        float du = uk[m] - uk[m + 1], dv = vk[m] - vk[m + 1];
        float s2l = FMUL(FMUL(FMUL(du, du) + FMUL(dv, dv), rdz), rdz);
        s2l = fmaxf(s2l, QN_EPSGM);
        s2v[m] = s2l;
        float tem = FMUL(tk[m] + tk[m + 1], 0.5f);
        float thm = FMUL(the[m] + the[m + 1], 0.5f);
        float a = FMUL(thm, QN_P608);
        float b = FMUL(((QN_ELOCP / tem) - 1.0f) - QN_P608, thm);
        float ghl = FMUL((FMUL(s1[m], FMUL(((qk[m] + qk[m + 1]) + cwm[m]) + cwm[m + 1],
                                           QN_HALF_P608) + 1.0f)
                          + FMUL(((qk[m] - qk[m + 1]) + cwm[m]) - cwm[m + 1], a))
                         + FMUL(cwm[m] - cwm[m + 1], b), rdz);
        if (fabsf(ghl) <= QN_EPSGH) ghl = QN_EPSGH;
        en2[m] = FMUL(ghl, QN_G) / thm;
        s3[m] = ghl;                                 // GH
        ri[m] = en2[m] / s2l;
    }
    // ELM in akm, REL in akh (both dead until DIFCOF)  (:782-806)
    int lmxl = lmh;
    for (int m = 0; m < nzm; ++m) {
        float s2l = s2v[m], ghl = s3[m], eloq2x;
        if (ghl >= QN_EPSGH) {
            if (s2l / ghl <= QN_REQU) {
                akm[m] = QN_EPSL;
                lmxl = m + 1;
                continue;
            }
            float aubr = FMUL(FMUL(QN_AUBM, s2l) + FMUL(QN_AUBH, ghl), ghl);
            float bubr = FMUL(QN_BUBM, s2l) + FMUL(QN_BUBH, ghl);
            float qol2st = FMUL(FMUL(-0.5f, bubr)
                                + sqrtf(FMUL(FMUL(bubr, bubr), 0.25f)
                                        - FMUL(aubr, QN_CUBR)), QN_RCUBR);
            eloq2x = 1.0f / qol2st;
        } else {
            float aden = FMUL(FMUL(QN_ADNM, s2l) + FMUL(QN_ADNH, ghl), ghl);
            float bden = FMUL(QN_BDNM, s2l) + FMUL(QN_BDNH, ghl);
            float qol2un = FMUL(-0.5f, bden)
                           + sqrtf(FMUL(FMUL(bden, bden), 0.25f) - aden);
            eloq2x = 1.0f / (qol2un + QN_EPSRU);
        }
        akm[m] = fmaxf(sqrtf(FMUL(eloq2x, q2[m])), QN_EPSL);
    }
    if (akm[lmh - 2] == QN_EPSL) lmxl = lmh;         // :808 (BLMX unused)
    (void)lmxl;
    for (int m = 0; m < lmh; ++m) s4[m] = 0.0f;      // Q1 (:735-737)
    for (int m = lpbl - 1; m < lmh; ++m) s4[m] = sqrtf(q2[m]);
    {
        float szq = 0.0f, sq = 0.0f;                 // :821-828
        for (int m = 0; m < nzm; ++m) {
            float qdzl = FMUL(s4[m] + s4[m + 1], zh[m + 1] - zh[m + 2]);
            szq = FMUL(((zh[m + 1] + zh[m + 2]) - zh[lmh]) - zh[lmh], qdzl) + szq;
            sq = qdzl + sq;
        }
        float el0 = fminf(FMUL(FMUL(QN_ALPH, szq), 0.5f) / sq, QN_EL0MAX);
        el0 = fmaxf(el0, QN_EL0MIN);
        int lpblm = max(lpbl - 1, 1);
        for (int m = 0; m < lpblm; ++m) {            // :843-846
            el[m] = fminf(FMUL(zh[m] - zh[m + 2], QN_ELFC), akm[m]);
            akh[m] = el[m] / akm[m];
        }
        if (lpbl < lmh) {                            // :852-858
            for (int m = lpbl - 1; m < lmh - 1; ++m) {
                float vkrmz = FMUL(zh[m + 1] - zh[lmh], QN_VKARMAN);
                el[m] = fminf(vkrmz / (vkrmz / el0 + 1.0f), akm[m]);
                akh[m] = el[m] / akm[m];
            }
        }
        for (int m = lpbl; m < lmh - 2; ++m) {       // :860-863
            float srel = fminf(FMUL(FMUL(akh[m - 1] + akh[m + 1], 0.5f) + akh[m],
                                    0.5f), akh[m]);
            el[m] = fmaxf(FMUL(srel, akm[m]), QN_EPSL);
        }
    }
    {   // stable-case QNSE length (:869-879)
        float f = fmaxf(corf_a[col], QN_EPS1);
        float rlambda = f / FMUL(QN_BLCKDR, ust_a[col]);
        for (int m = 0; m < nzm; ++m) {
            if (en2[m] >= 0.0f) {
                float vkrmz = FMUL(zh[m + 1] - zh[lmh], QN_VKARMAN);
                float rlb = rlambda + 1.0f / vkrmz;
                float rln = sqrtf(FMUL(2.0f, en2[m]) / q2[m]) / QN_CN;
                el[m] = 1.0f / (rlb + rln);
            }
        }
    }
    kpbl_a[col] = nz - lpbl + 1;                     // :376

    // ================= DIFCOF (:973-1061) ================================
    for (int m = 0; m < nzm; ++m) {
        float ell = el[m];
        float q1l = sqrtf(FMUL(0.5f, q2[m]));
        float ril = ri[m];
        float ak0 = FMUL(FMUL(QN_C0, ell), q1l);
        float alpham, alphah;
        if (ril <= 0.0f) {
            float aril = fminf(fabsf(ril), FMUL(2.0f, QN_ARIMIN));
            float aril2 = FMUL(aril, aril);
            float aril4 = FMUL(aril2, aril2);
            alpham = (1.0f + FMUL(QN_BM1, aril)) + FMUL(QN_BM2, aril2);
            alphah = ((QN_AH1 + FMUL(QN_BH1, aril)) + FMUL(QN_BH2, aril2))
                     + FMUL(QN_BH3, aril4);
        } else {
            float ril2 = FMUL(ril, ril);
            alpham = (1.0f + FMUL(QN_AM1, ril2))
                     / ((1.0f + FMUL(QN_AM2, ril)) + FMUL(QN_AM3, ril2));
            alphah = ((QN_AH1 + FMUL(QN_AH2, ril)) + FMUL(QN_AH3, ril2))
                     / ((1.0f + FMUL(QN_AH4, ril)) + FMUL(QN_AH5, ril2));
        }
        float rdz = 2.0f / (zh[m] - zh[m + 2]);
        float ak0dz = FMUL(ak0, rdz);
        float akmin = FMUL(QN_EPSKM, rdz);
        akm[m] = fmaxf(FMUL(ak0dz, alpham), akmin);
        akh[m] = fmaxf(FMUL(ak0dz, alphah), akmin);
    }
    // EXCH_H/EXCH_M (:393-400), WRF-up order, level 0 untouched.
    for (int kup = 1; kup < nz; ++kup) {
        int m = nz - kup - 1;                       // KFLIP-1, KFLIP=KTE-K
        float deltaz = FMUL(0.5f, zh[m] - zh[m + 2]);
        QN_UP(exch_h_a, kup) = FMUL(akh[m], deltaz);
        QN_UP(exch_m_a, kup) = FMUL(akm[m], deltaz);
    }

    // ================= PRODQ2 (:886-968) =================================
    for (int m = 0; m < lmh - 1; ++m) {
        float s2l = s2v[m];
        float q2l = q2[m];
        float deltaz = FMUL(0.5f, zh[m] - zh[m + 2]);
        float akml = FMUL(akm[m], deltaz);
        float akhl = FMUL(akh[m], deltaz);
        float en2l = FMUL(ri[m], s2l);
        float pr = FMUL(akml, s2l);
        float bpr = FMUL(akhl, en2l);
        float dis = FMUL(QN_CEPS, gfk_pow(FMUL(0.5f, q2l), 1.5f)) / el[m];
        q2l = q2l + FMUL(FMUL(2.0f, (pr - bpr) - dis), dtturbl);
        q2[m] = fmaxf(q2l, QN_EPSQ2L);
    }
    q2[lmh - 1] = fmaxf(FMUL(FMUL(QN_RC02, ust_a[col]), ust_a[col]), QN_EPSQ2L);

    // ================= VDIFQ (:1066-1142) ================================
    // s1=DTOZ, s2=AKQ, s3=CR, s4=CM, rhok=RSQ2 (rhok is not loaded yet).
    {
        const float esqhf = FMUL(0.5f, QN_ESQ);
        int nq = lmh - 2;
        for (int m = 0; m < nq; ++m) {
            s1[m] = (dtdif + dtdif) / (zh[m] - zh[m + 2]);
            s2[m] = FMUL(FMUL(sqrtf(FMUL(q2[m] + q2[m + 1], 0.5f)), el[m] + el[m + 1]),
                         esqhf) / (zh[m + 1] - zh[m + 2]);
            s3[m] = -FMUL(s1[m], s2[m]);
        }
        s4[0] = FMUL(s1[0], s2[0]) + 1.0f;
        rhok[0] = q2[0];
        for (int m = 1; m < nq; ++m) {
            float cf = -FMUL(s1[m], s2[m - 1]) / s4[m - 1];
            s4[m] = (-FMUL(s3[m - 1], cf) + FMUL(s2[m - 1] + s2[m], s1[m])) + 1.0f;
            rhok[m] = -FMUL(rhok[m - 1], cf) + q2[m];
        }
        float dtozs = (dtdif + dtdif) / (zh[lmh - 2] - zh[lmh]);
        float akqs = FMUL(FMUL(sqrtf(FMUL(q2[lmh - 2] + q2[lmh - 1], 0.5f)),
                               el[lmh - 2] + QN_ELZ0), esqhf)
                     / (zh[lmh - 1] - zh[lmh]);
        float cf = -FMUL(dtozs, s2[nq - 1]) / s4[nq - 1];
        q2[lmh - 2] = ((FMUL(FMUL(dtozs, akqs), q2[lmh - 1]) - FMUL(rhok[nq - 1], cf))
                       + q2[lmh - 2])
                      / ((FMUL(s2[nq - 1] + akqs, dtozs) - FMUL(s3[nq - 1], cf)) + 1.0f);
        for (int m = nq - 1; m >= 0; --m) {
            q2[m] = (-FMUL(s3[m], q2[m + 1]) + rhok[m]) / s4[m];
        }
    }
    // Save the new TKE and mixing length (:426-431).
    for (int kup = 0; kup < nz; ++kup) {
        int m = nz - 1 - kup;
        q2[m] = fmaxf(q2[m], QN_EPSQ2L);
        QN_UP(tke_a, kup) = FMUL(0.5f, q2[m]);
        QN_UP(el_a, kup) = (m + 1 != nz) ? el[m] : 0.0f;
    }

    // ================= THSK (:442-451) ===================================
    const float psfc = psfc_a[col];
    float thsk;
    {
        float rlow = psfc / FMUL(QN_R_D, QN_UP(t_a, 0));
        thsk = hfx_a[col] / FMUL(FMUL(akhs_a[col], rlow), QN_CP) + QN_UP(th_a, 0);
    }

    // ================= main_integration, heat/moisture (:458-546) ========
    for (int m = 0; m < nz; ++m) {
        int kup = nz - 1 - m;
        // THEK, QK, CWMK reload the setup values (:467-476): the[] still
        // holds them (VDIFQ did not touch it); qk/cwm likewise.
        rhok[m] = QN_UP(p_a, kup)
                  / FMUL(FMUL(QN_R_D, QN_UP(t_a, kup)),
                         (1.0f + FMUL(QN_P608, qk[m])) - cwm[m]);
    }
    // AKHK (:482-484) in place over AKH; AKH is not read again.
    for (int m = 0; m < nz - 1; ++m)
        akh[m] = FMUL(FMUL(akh[m], 0.5f), rhok[m] + rhok[m + 1]);
    float seamask = xland_a[col] - 1.0f;
    float thz0 = FMUL(1.0f - seamask, thsk) + FMUL(seamask, thz0_a[col]);
    thz0_a[col] = thz0;
    float akhs_dens = FMUL(akhs_a[col], rhok[nz - 1]);
    float qsfc = qsfc_a[col];
    if (seamask < 0.5f) {
        float qfc1 = FMUL(FMUL(QN_XLV, chklowq_a[col]), akhs_dens);
        if (snow_a[col] > 0.0f || sice_a[col] > 0.5f) qfc1 = FMUL(qfc1, QN_RLIVWV);
        if (qfc1 > 0.0f) qsfc = qk[nz - 1] + elflx_a[col] / qfc1;
    } else {
        float exnsfc = gfk_pow(1.0e5f / psfc, QN_CAPA);
        qsfc = FMUL(QN_PQ0SEA / psfc,
                    gfk_exp(FMUL(QN_A2, thsk - FMUL(QN_A3, exnsfc))
                            / (thsk - FMUL(QN_A4, exnsfc))));
    }
    qsfc_a[col] = qsfc;
    float qz0 = FMUL(1.0f - seamask, qsfc) + FMUL(seamask, qz0_a[col]);
    qz0_a[col] = qz0;

    // ---- VDIFH (:1147-1240).  CT was zeroed by MIXLEN, so CTHF = 0 and
    // RKCT is 0*... -- formed anyway.  s1=DTOZ, s3=CR, s4=CM, s2=RKCT;
    // the three right-hand sides ride gm-free rows: RST in en2, RSQ in
    // ri, RSC in s2v (all dead since PRODQ2).
    {
        const float cthf = FMUL(0.5f, 0.0f);         // CT after MIXLEN
        const int nh = lmh - 1;
        const float chklowq = chklowq_a[col];
        for (int m = 0; m < nh; ++m) {
            s1[m] = dtdif / (zh[m] - zh[m + 1]);
            s3[m] = -FMUL(s1[m], akh[m]);
            s2[m] = FMUL(FMUL(akh[m], zh[m] - zh[m + 2]), cthf);
        }
        s4[0] = FMUL(s1[0], akh[0]) + rhok[0];
        en2[0] = -FMUL(s2[0], s1[0]) + FMUL(the[0], rhok[0]);
        ri[0] = FMUL(qk[0], rhok[0]);
        s2v[0] = FMUL(cwm[0], rhok[0]);
        for (int m = 1; m < nh; ++m) {
            float dtozl = s1[m];
            float cf = -FMUL(dtozl, akh[m - 1]) / s4[m - 1];
            s4[m] = (-FMUL(s3[m - 1], cf) + FMUL(akh[m - 1] + akh[m], dtozl)) + rhok[m];
            en2[m] = (-FMUL(en2[m - 1], cf) + FMUL(s2[m - 1] - s2[m], dtozl))
                     + FMUL(the[m], rhok[m]);
            ri[m] = -FMUL(ri[m - 1], cf) + FMUL(qk[m], rhok[m]);
            s2v[m] = -FMUL(s2v[m - 1], cf) + FMUL(cwm[m], rhok[m]);
        }
        float dtozs = dtdif / (zh[lmh - 1] - zh[lmh]);
        float rkhh = akh[nh - 1];
        float cf = -FMUL(dtozs, rkhh) / s4[nh - 1];
        float rkqs = FMUL(akhs_dens, chklowq);
        float cmb = FMUL(s3[nh - 1], cf);
        float rho_l = rhok[lmh - 1];
        float cmtb = (-cmb + FMUL(rkhh + akhs_dens, dtozs)) + rho_l;
        float cmqb = (-cmb + FMUL(rkhh + rkqs, dtozs)) + rho_l;
        float cmcb = (-cmb + FMUL(rkhh, dtozs)) + rho_l;
        float rstb = (-FMUL(en2[nh - 1], cf) + FMUL(s2[nh - 1], dtozs))
                     + FMUL(the[lmh - 1], rho_l);
        float rsqb = -FMUL(ri[nh - 1], cf) + FMUL(qk[lmh - 1], rho_l);
        float rscb = -FMUL(s2v[nh - 1], cf) + FMUL(cwm[lmh - 1], rho_l);
        the[lmh - 1] = (FMUL(FMUL(dtozs, akhs_dens), thz0) + rstb) / cmtb;
        qk[lmh - 1] = (FMUL(FMUL(dtozs, rkqs), qz0) + rsqb) / cmqb;
        cwm[lmh - 1] = rscb / cmcb;
        for (int m = nh - 1; m >= 0; --m) {
            float rcml = 1.0f / s4[m];
            the[m] = FMUL(-FMUL(s3[m], the[m + 1]) + en2[m], rcml);
            qk[m] = FMUL(-FMUL(s3[m], qk[m + 1]) + ri[m], rcml);
            cwm[m] = FMUL(-FMUL(s3[m], cwm[m + 1]) + s2v[m], rcml);
        }
    }
    // Tendencies (:534-546), accumulated onto the slot.
    for (int kup = 0; kup < nz; ++kup) {
        int m = nz - 1 - kup;
        float ape = 1.0f / QN_UP(exner_a, kup);
        float thold = QN_UP(th_a, kup);
        float thnew = the[m] + FMUL(FMUL(cwm[m], QN_ELOCP), ape);
        float dtdt = FMUL(thnew - thold, rdtturbl);
        float qvup = QN_UP(qv_a, kup);
        float qold = qvup / (1.0f + qvup);
        float dqdt = FMUL(qk[m] - qold, rdtturbl);
        float dcdt = FMUL(cwm[m] - QN_UP(qc_a, kup), rdtturbl);
        QN_UP(rthblten_a, kup) = dtdt + QN_UP(rthblten_a, kup);
        QN_UP(rqvblten_a, kup) = dqdt / FMUL(1.0f - qk[m], 1.0f - qk[m])
                                 + QN_UP(rqvblten_a, kup);
        QN_UP(rqcblten_a, kup) = dcdt + QN_UP(rqcblten_a, kup);
    }

    // ================= main_integration, momentum (:608-653) =============
    {
        for (int m = 0; m < nz - 1; ++m)
            akm[m] = FMUL(FMUL(akm[m], rhok[m] + rhok[m + 1]), 0.5f);
        float akms_dens = FMUL(akms_a[col], rhok[nz - 1]);
        float uz0 = uz0_a[col], vz0 = vz0_a[col];
        int nv = lmh - 1;
        for (int m = 0; m < nv; ++m) {
            s1[m] = dtdif / (zh[m] - zh[m + 1]);
            s3[m] = -FMUL(s1[m], akm[m]);
        }
        s4[0] = FMUL(s1[0], akm[0]) + rhok[0];
        for (int m = 1; m < nv; ++m) {
            float cf = -FMUL(s1[m], akm[m - 1]) / s4[m - 1];
            s4[m] = (-FMUL(s3[m - 1], cf) + FMUL(akm[m - 1] + akm[m], s1[m])) + rhok[m];
        }
        float dtozs = dtdif / (zh[lmh - 1] - zh[lmh]);
        float rkmh = akm[nv - 1];
        float cfb = -FMUL(dtozs, rkmh) / s4[nv - 1];
        float rcmvb = 1.0f / ((FMUL(rkmh + akms_dens, dtozs) - FMUL(s3[nv - 1], cfb))
                              + rhok[lmh - 1]);
        float dtozak = FMUL(dtozs, akms_dens);
        for (int comp = 0; comp < 2; ++comp) {
            float* var = comp ? vk : uk;
            float sz0 = comp ? vz0 : uz0;
            s2[0] = FMUL(var[0], rhok[0]);           // RSU/RSV
            for (int m = 1; m < nv; ++m) {
                float cf = -FMUL(s1[m], akm[m - 1]) / s4[m - 1];
                s2[m] = -FMUL(s2[m - 1], cf) + FMUL(var[m], rhok[m]);
            }
            var[lmh - 1] = FMUL((FMUL(dtozak, sz0) - FMUL(s2[nv - 1], cfb))
                                + FMUL(var[lmh - 1], rhok[lmh - 1]), rcmvb);
            for (int m = nv - 1; m >= 0; --m) {
                float rcml = 1.0f / s4[m];
                var[m] = FMUL(-FMUL(s3[m], var[m + 1]) + s2[m], rcml);
            }
        }
        for (int kup = 0; kup < nz; ++kup) {
            int m = nz - 1 - kup;
            QN_UP(rublten_a, kup) = FMUL(uk[m] - QN_UP(u_a, kup), rdtturbl)
                                    + QN_UP(rublten_a, kup);
            QN_UP(rvblten_a, kup) = FMUL(vk[m] - QN_UP(v_a, kup), rdtturbl)
                                    + QN_UP(rvblten_a, kup);
        }
    }
#undef QN_UP
}
