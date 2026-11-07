// TEMF surface layer, sf_sfclay_physics=10, WRF v4.6.1.
//
// Transcribed in WRF operation order from the byte-pinned
// phys/module_sf_temfsfclay.F (TEMFSFCLAY1D, :154-384; sha256 70f2ed6b...).
// The :NNN anchors below are that file's.  One thread owns one surface
// column; the scheme reads only the lowest model level.
//
// Arithmetic: every binary32 operation is pinned with the round-to-nearest
// intrinsics (FADD/FSUB/FMUL/FDIV/FSQRT), so the kernel is never contracted
// under any --fmad setting, and the default and strict (GPUWM_WRF_EXACT=1)
// builds run this one source.  exp, log and real powers are WOOF's own
// binary32 routines gfk_exp/gfk_log/gfk_pow; LOG10 is WOOF's own correctly
// rounded woof_log10f (log10_flt32.cuh).  gfortran lowering: X**2. folds to
// X*X; X**1.5 and X**(1./3.) are calls (measured in the oracle's
// undefined-symbol receipt), here gfk_pow.  WRF's parameters are derived
// below from the module's own literals in the module's own float32
// operations, exactly as gfortran folds them.
//
// Graded by tests/test_temf_sfclay_wrf461_parity.py against
// tools/temf_sfclay_wrf461_oracle (gfortran 13.3, the WRF scalar build's own
// object): the parity tests require 0 ULP outside declared divergences.
//
// Declared divergences from WRF v4.6.1 (each in the registry row):
//  * MAVAIL = 0.  WRF forms FLQC = (...)*MAVAIL (:350) and then divides by
//    MAVAIL for EXCH_TEMF and CHS (:351-352): 0/0, a NaN that CHS carries
//    into the land-surface model.  Here EXCH_TEMF and CHS take the limit
//    the formula has, the coefficient before the MAVAIL factor.  Where
//    MAVAIL != 0 the WRF expression is evaluated word for word.
//  * LOG10.  WRF's log10 word is the C library's binary32 log10f, which is
//    not correctly rounded (it misses on 5.7% of the words in [1,10));
//    WOOF's is.  Where the two words differ, every output downstream of the
//    gradients differs; the oracle's arm C (WRF with a correctly rounded
//    log10f) is the referee those columns are graded 0 ULP against.
//
// Not reproduced, by design: the PRINT at :278 (a diagnostic, no state),
// ANG = ATAN2(V,U) (:330, read by nothing), KM/KH (:320-323) and DQTDZ
// (:249), read by nothing in this routine.

#define TEMF_KARMAN TC_KARMAN

extern "C" __global__ void temf_sfclay_column(
    // lowest-model-level state (n)
    const real* __restrict__ u1_a, const real* __restrict__ v1_a,
    const real* __restrict__ th1_a, const real* __restrict__ qv1_a,
    const real* __restrict__ p1_a, const real* __restrict__ pi1_a,
    const real* __restrict__ rho1_a, const real* __restrict__ z1_a,
    // surface (n)
    const real* __restrict__ ht_a, const real* __restrict__ psfc_a,
    const real* __restrict__ znt_a, const real* __restrict__ mavail_a,
    const real* __restrict__ tsk_a, const real* __restrict__ fcor_a,
    const real* __restrict__ hd_a,
    // carried (n): TE_TEMF at the first level (floored in place), HFX (read
    // for w*, then rewritten), WM_TEMF (two-step average)
    real* __restrict__ te1_a, real* __restrict__ hfx_a,
    real* __restrict__ wm_a,
    // outputs (n)
    real* __restrict__ ust_a, real* __restrict__ qsfc_a,
    real* __restrict__ chs_a, real* __restrict__ chs2_a,
    real* __restrict__ cqs2_a, real* __restrict__ cpm_a,
    real* __restrict__ qgh_a, real* __restrict__ qfx_a,
    real* __restrict__ lh_a, real* __restrict__ flhc_a,
    real* __restrict__ flqc_a, real* __restrict__ u10_a,
    real* __restrict__ v10_a, real* __restrict__ th2_a,
    real* __restrict__ t2_a, real* __restrict__ q2_a,
    real* __restrict__ exch_a,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // ---- TE model constants, :196-212, as gfortran folds them ----
    const real TEmin = TC_TEMIN;
    const real ftau0 = TC_FTAU0;
    const real fth0 = TC_FTH0;
    const real Cf = TC_CF;
    const real CN = 2.0f;
    const real Ceps = TC_CEPS;
    const real Cphi = Ceps;
    // PrT0 = Cphi/Ceps * ftau0**2 / 2. / fth0**2
    const real PrT0 = FDIV(FDIV(FMUL(FDIV(Cphi, Ceps), FMUL(ftau0, ftau0)),
                                2.0f), FMUL(fth0, fth0));
    // Cphi**2.*ftau0**2./(2.*Ceps**2.*fth0**2.)  (:291) and its -2. twin (:297)
    const real ri_num = FMUL(FMUL(Cphi, Cphi), FMUL(ftau0, ftau0));
    const real ri_den = FMUL(FMUL(2.0f, FMUL(Ceps, Ceps)), FMUL(fth0, fth0));
    const real ri_stable = FDIV(ri_num, ri_den);
    const real ri_unstable = FDIV(ri_num, -ri_den);

    const real u1 = u1_a[i], v1 = v1_a[i], th1 = th1_a[i], qv1 = qv1_a[i];
    const real p1 = p1_a[i], pi1 = pi1_a[i], rho = rho1_a[i], z = z1_a[i];
    const real zsrf = ht_a[i], psfc = psfc_a[i], znt = znt_a[i];
    const real mavail = mavail_a[i], tsk = tsk_a[i], fcor = fcor_a[i];
    const real hd = hd_a[i];
    real te = te1_a[i];
    const real hfx_prev = hfx_a[i];
    const real wm_prev = wm_a[i];

    // :235-238 surface saturated q and q in air at the surface
    real e1 = FMUL(SVP1, gfk_exp(FDIV(FMUL(SVP2, FSUB(tsk, SVPT0)),
                                      FSUB(tsk, SVP3))));
    const real qsfc = FDIV(FMUL(EP2, e1), FSUB(FDIV(psfc, 1000.0f), e1));
    const real qsfc_air = FMUL(qsfc, mavail);
    const real tsk_pi = FDIV(tsk, pi1);
    const real thetav = FMUL(tsk_pi, FADD(1.0f, FMUL(0.608f, qsfc_air)));
    const real z0t = znt;                                   // :241
    // :244-245 heights at the turbulence and mass levels
    const real zt = FDIV(FSUB(FSUB(z, zsrf), znt), 2.0f);
    const real zm = FSUB(z, zsrf);
    // :248-251 gradients at the first level
    const real l10t = woof_log10f(FDIV(zm, z0t));
    const real l10m = woof_log10f(FDIV(zm, znt));
    const real dthdz = FDIV(FSUB(th1, tsk_pi), FMUL(zt, l10t));
    // dqtdz (:249) feeds nothing this routine writes.
    const real dudz = FDIV(u1, FMUL(zt, l10m));
    const real dvdz = FDIV(v1, FMUL(zt, l10m));

    // :256 TE_TEMF may arrive uninitialised; floored in place
    if (te < TEmin) te = TEmin;

    // :258-262
    real wstr = 0.0f;
    if (hfx_prev > 0.0f) {
        wstr = gfk_pow(FMUL(FDIV(FMUL(G, hd), thetav),
                            FDIV(hfx_prev, FMUL(rho, CP))),
                       FDIV(1.0f, 3.0f));
    }

    // :272-300 stability parameters
    const real N2 = FMUL(FDIV(G, thetav), dthdz);
    const real S = FSQRT(FADD(FMUL(dudz, dudz), FMUL(dvdz, dvdz)));
    real Ri = FDIV(N2, FMUL(S, S));
    if (S < 1e-15f) {
        Ri = (N2 >= 0.0f) ? 0.2f : -1.0f;
    }
    if (Ri > 0.2f) Ri = 0.2f;
    const real beta = FDIV(G, thetav);
    real ratio, ftau, fth;
    if (Ri > 0.0f) {
        ratio = FDIV(Ri, FADD(ri_stable, FMUL(3.0f, Ri)));
        const real stab = FADD(1.0f, FMUL(4.0f, Ri));
        ftau = FMUL(ftau0, FADD(FDIV(0.75f, stab), 0.25f));
        fth = FDIV(fth0, stab);
        // TE2 (:295) feeds only KM, which this routine never writes.
    } else {
        ratio = FDIV(Ri, FADD(ri_unstable, FMUL(2.0f, Ri)));
        ftau = ftau0;
        fth = fth0;
    }
    const real TKE = FMUL(te, FSUB(1.0f, ratio));
    const real ustrtilde = FSQRT(FMUL(ftau, TKE));
    // :302-310 length scale
    real linv = FADD(FDIV(FDIV(1.0f, TEMF_KARMAN), zt),
                     FDIV(fabsf(fcor), FMUL(Cf, ustrtilde)));
    if (N2 > 0.0f) {
        linv = FADD(linv, FDIV(FSQRT(N2), FMUL(CN, ustrtilde)));
    }
    real leps = FDIV(1.0f, linv);
    // :313-314 lepsmin = 0; gfortran's MAX keeps the second argument when
    // the first is not greater
    leps = (leps > 0.0f) ? leps : 0.0f;

    // :329 surface fluxes
    const real logm = gfk_log(FDIV(zm, znt));
    const real half_w = FMUL(0.5f, wstr);
    const real ust = FDIV(FDIV(FMUL(FMUL(FSQRT(FDIV(ftau, ftau0)),
        FSQRT(FADD(FADD(FMUL(u1, u1), FMUL(v1, v1)), FMUL(half_w, half_w)))),
        leps), logm), zt);
    // :333-340
    real wm = FDIV(FADD(ust, wm_prev), 2.0f);
    const real wm_out = (wm > 1e-2f) ? wm : 1e-2f;

    // :349-352 exchange coefficients
    const real logt = gfk_log(FDIV(zm, z0t));
    const real flhc = FDIV(FDIV(FDIV(FMUL(FMUL(FDIV(FMUL(FMUL(rho, CP), fth),
                                                    fth0), wm), leps),
                                      PrT0), logt), zt);
    const real flqc_dry = FDIV(FDIV(FDIV(FMUL(FMUL(FDIV(FMUL(rho, fth), fth0),
                                                   wm), leps), PrT0), logt),
                               zt);
    const real flqc = FMUL(flqc_dry, mavail);
    real exch, chs;
    if (mavail != 0.0f) {
        exch = FDIV(flqc, mavail);
        chs = FDIV(FDIV(flqc, rho), mavail);
    } else {
        // declared divergence: WRF's 0/0 replaced by the limit
        exch = flqc_dry;
        chs = FDIV(flqc_dry, rho);
    }

    // :358-362 surface heat and moisture fluxes
    const real hfx = FMUL(flhc, FSUB(tsk, FMUL(th1, pi1)));
    real qfx = FMUL(flqc, FSUB(qsfc, qv1));
    qfx = (qfx > 0.0f) ? qfx : 0.0f;
    const real lh = FMUL(XLV, qfx);

    // :367-371 10 m winds, 2 m temperature and humidity
    const real log10m = gfk_log(FDIV(10.0f, znt));
    const real log2t = gfk_log(FDIV(2.0f, z0t));
    const real log2m = gfk_log(FDIV(2.0f, znt));
    const real u10 = FDIV(FMUL(u1, log10m), logm);
    const real v10 = FDIV(FMUL(v1, log10m), logm);
    const real t2 = FMUL(FADD(tsk_pi, FDIV(FMUL(FSUB(th1, tsk_pi), log2t),
                                           logt)), pi1);
    const real th2 = FDIV(t2, pi1);
    const real q2 = FADD(qsfc_air, FDIV(FMUL(FSUB(qv1, qsfc_air), log2m), logm));
    // :374-375
    const real coef = FMUL(FMUL(FDIV(fth, fth0), wm), leps);
    const real chs2 = FDIV(FDIV(FDIV(coef, PrT0), log2t), zt);
    const real cqs2 = FDIV(FDIV(FDIV(coef, PrT0), log2m), zt);

    // :378-380 qgh at the first-level temperature, cpm
    const real t1 = FMUL(th1, pi1);
    e1 = FMUL(SVP1, gfk_exp(FDIV(FMUL(SVP2, FSUB(t1, SVPT0)),
                                 FSUB(t1, SVP3))));
    const real qgh = FDIV(FMUL(EP2, e1), FSUB(FDIV(p1, 1000.0f), e1));
    const real cpm = FMUL(CP, FADD(1.0f, FMUL(0.8f, qv1)));

    te1_a[i] = te;
    hfx_a[i] = hfx;
    wm_a[i] = wm_out;
    ust_a[i] = ust;
    qsfc_a[i] = qsfc;
    chs_a[i] = chs;
    chs2_a[i] = chs2;
    cqs2_a[i] = cqs2;
    cpm_a[i] = cpm;
    qgh_a[i] = qgh;
    qfx_a[i] = qfx;
    lh_a[i] = lh;
    flhc_a[i] = flhc;
    flqc_a[i] = flqc;
    u10_a[i] = u10;
    v10_a[i] = v10;
    th2_a[i] = th2;
    t2_a[i] = t2;
    q2_a[i] = q2;
    exch_a[i] = exch;
}
