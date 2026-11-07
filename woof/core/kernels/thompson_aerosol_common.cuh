// ======================================================================
// THIRD-PARTY NOTICE.  This file carries values derived from third-party
// work.  ArWen distributes it under the Apache License 2.0; the notice
// below belongs to that upstream and is kept here because the upstream
// kept it.  Full text in the licenses/ directory.
//
//   The THOMPSON_AA_CC* tables below are values produced by WRF's REAL(4)
//   WGAMMA/GAMMLN pair, whose Lanczos coefficients come from Numerical
//   Recipes via WRF v4.6.1 phys/module_mp_thompson.F, which preserves the
//   notice:
//
//       (C) Copr. 1986-92 Numerical Recipes Software 2.02
//
//   See licenses/NOTICE-Numerical-Recipes.txt.
// ======================================================================
// gpuwm/core/kernels/thompson_aerosol_common.cuh
//
// Shared __device__ helpers for WRF v4.6.1 aerosol-aware Thompson
// (mp_physics=28).  Numerical authority is
// WRF v4.6.1 phys/module_mp_thompson.F, commit
// d66e442fccc04111067e29274c9f9eaccc3cef28, zero local modifications.  Every
// line number in this file refers to that source.
//
// ---------------------------------------------------------------------------
// HOW THIS FILE IS USED
// ---------------------------------------------------------------------------
// There is no #include path under cupy.RawModule, so this header is PREPENDED
// textually by gpuwm/core/kernels/__init__.py::_EXTRA_HEADERS to the six
// aerosol translation units:
//     thompson_aerosol_probe, thompson_aerosol_state, thompson_aerosol_sat,
//     thompson_aerosol_cold,  thompson_aerosol_warm,  thompson_aerosol_sed
// Every other module -- above all thompson.cu -- assembles a BYTE-IDENTICAL
// source string to what it assembled before this header existed.  Do not add
// this header to any other module, and do not #include it from a .cu file.
//
// ===========================================================================
// PUBLISHED SHARED SIGNATURES -- cold.cu, warm.cu and sed.cu MUST CALL THESE
// ===========================================================================
//
// These used to be duplicated, per-network, with DIFFERENT BODIES.  That is
// exactly how the two halves of a scheme drift apart: separate
// cupy.RawModule translation units mean nvrtc never sees the conflict, so the
// drift is silent.  Each now has ONE definition, here, and the .cu files carry
// none.
//
// ---------------------------------------------------------------------------
// WHAT ENFORCES THAT, AND WHAT DOES NOT
// ---------------------------------------------------------------------------
// THE ENFORCEMENT MECHANISM IS THE SOURCE SCAN, NOT THE COMPILER.
// tests/test_thompson_aerosol_device_helpers.py::
// test_shared_helpers_are_defined_exactly_once_and_only_in_the_header greps
// every gpuwm/core/kernels/thompson_aerosol_*.cu for a DEFINITION of each name
// below and fails, naming file and line, if one survives.  That test is the
// contract.  Run it; do not rely on a build failure.
//
// This file used to claim instead that "a surviving local copy will FAIL TO
// COMPILE with a redefinition error -- that error is the enforcement
// mechanism".  THAT CLAIM WAS FALSE FOR ONE OF THE FOUR AND THE FALSE HALF IS
// THE DANGEROUS HALF:
//
//   * thompson_aa_bound_rain_number, thompson_aa_bound_ice_number and
//     thompson_aa_decade_index_double have shared signatures IDENTICAL to the
//     local copies they replaced, so a survivor really is a hard nvrtc
//     "function has already been defined" error.  VERIFIED by compiling the
//     assembled source string for thompson_aerosol_cold and
//     thompson_aerosol_warm with a local thompson_aa_decade_index_double
//     still present: both fail at the local definition, citing the header's
//     line as the previous one.
//
//   * thompson_aa_entry_rain_distribution DOES NOT.  Its shared form takes
//     SEVEN parameters where the deleted local copies took six (they emitted
//     no rain_intercept_n0).  C++ treats a different parameter list as an
//     OVERLOAD, not a redefinition: the translation unit compiles CLEANLY,
//     and every six-argument call site keeps resolving to the LOCAL copy --
//     the plain-powf one that sat ~2.7e-7 away from the oracle -- while the
//     header's contraction-pinned definition sits there unused.  An agent
//     built exactly that variant and nvrtc said nothing.  There is no
//     compiler diagnostic for this; only the source scan catches it.
//
// So: delete the local copy.  Do not rename it, and do not assume the build
// would have told you.
//
//  bool thompson_aa_entry_rain_distribution(
//           float rain_per_kg, float rain_number_per_kg, float density,
//           float* rain_number, double* rain_lambda, float* rain_mvd,
//           double* rain_intercept_n0)
//        module_mp_thompson.F:1878-1898 (entry bound) THEN :2144-2150 (the
//        y-intercept pass).  Returns L_qr, i.e. rain_per_kg > R1.
//        rain_number [m^-3], rain_lambda [m^-1], rain_mvd [m],
//        rain_intercept_n0 = nr*org2*lamr**cre(2) = nr*lamr (cre(2)=mu_r+1=1,
//        org2=1/WGAMMA(1)=1).  THIS IS THE CONTRACTION-PINNED FORM: every
//        product/quotient is thompson_aa_mul/div and every power is
//        thompson_aa_powf.  warm.cu:81-92 records lamr / mvd_r / N0_r and
//        every rate built on them at <= 7e-16 relative over 12348 Fortran-
//        oracle rows -- float32-exact, the residual being the CSV round trip
//        -- where the earlier plain-powf form sat at ~2.7e-7.  Two properties
//        callers depend on:
//          * the :2146-2150 pass ALWAYS re-forms lamr from the bounded nr,
//            for every level, rain or not -- WRF's loop at :2145 has no
//            L_qr guard -- so rain_lambda/rain_mvd/rain_intercept_n0 are
//            defined even when the return value is false (rr=R1, nr=R2);
//          * prr_rcw / pnc_rcw / pna_rca / pnd_rcd read N0_r and
//            (lamr+fv_r)**(-cre(9)) directly, so they MUST use
//            rain_intercept_n0 and rain_lambda from here, never a lambda
//            re-derived from the clamped mvd.
//
//  void thompson_aa_bound_rain_number(float rain_mass, float density,
//                                     float* rain_number_per_kg)
//        module_mp_thompson.F:4032-4046 == thompson.cu:2574-2598.  Terminal
//        size bound; rain_mass [kg m^-3], rain_number_per_kg in/out [kg^-1].
//
//  void thompson_aa_bound_ice_number(float ice_mass, float density,
//                                    float* ice_number_per_kg)
//        module_mp_thompson.F:4029-4039 == thompson.cu:3719-3743.  Terminal
//        size bound; idempotent, so a fused and a terminal application agree.
//
//  int  thompson_aa_decade_index_double(double value, int first_exponent,
//                                       int table_size)
//        thompson.cu:3084-3105.  The DOUBLE form of the base-ten
//        mantissa/decade bin, for the rain and graupel y-intercept lookups,
//        which WRF keeps in DOUBLE PRECISION (N0_r/N0_g, :1587) even though
//        the state and the base-ten scale are default REAL.  Returned
//        ZERO-BASED.  Callers: the idx_r lookup at (first_exponent 6,
//        table_size 37) and the idx_g lookup at (2, 37).
//        PROMOTED FROM cold.cu:206-223 AND warm.cu:272-289, which carried
//        byte-identical copies.  The float sibling thompson_aa_decade_index
//        was already shared; only the double form was not, and two copies of
//        one helper in two translation units is precisely the drift vector
//        that produced the entry_rain_distribution divergence recorded above.
//        The body here is those copies verbatim, and
//        test_promoted_decade_index_double_is_bitwise_identical_to_the_local_
//        copies compiles a golden byte-copy of them alongside this definition
//        in ONE translation unit and asserts bitwise equality on every input,
//        so the promotion cannot have changed a result.
//
//  int  thompson_aa_nu_c_working(float nc_m3_after_rediagnosis)
//        module_mp_thompson.F:2170.  See the nu_c STAGING RULE below.  This
//        is the nu_c every rate in the warm and cold networks must use.
//
// ---------------------------------------------------------------------------
// THE nu_c STAGING RULE  (:1832 entry nu_c  vs  :2170 working nu_c)
// ---------------------------------------------------------------------------
// WRF computes nu_c TWICE from two DIFFERENT droplet numbers:
//
//   :1832  nu_c = MIN(15, NINT(1000.E6/nc(k)) + 2)   <- nc(k) from :1829,
//          the PRE-rediagnosis nc = MAX(2, MIN(nc1d*rho, Nt_c_max)).  Used
//          ONLY to build lamc at :1833 and to run the :1834-1838 droplet-size
//          clamp.  thompson_aa_cloud_dist returns it as nu_c_entry_out.
//
//   :1840  nc(k) is REDIAGNOSED from the clamped lamc.  This is what
//          thompson_aa_cloud_dist RETURNS.
//
//   :2170  nu_c = MIN(15, NINT(1000.E6/nc(k)) + 2)   <- recomputed from the
//          POST-rediagnosis nc(k).  THIS is the nu_c that feeds lamc (:2173),
//          mvd_c (:2174), Dc_g (:2181) and pnr_wau (:2192).
//
// Whenever the :1834-1838 size clamp engages -- thin cloud edges, which the
// cold network reaches at qc > 1e-12 -- the two DIFFER.  MEASURED on an
// RTX 5090 over a 70-point (nc_entry, rc) grid at rho = 1: 13 of 70 states
// diverge, every one of them at rc <= 1e-7.  Worked examples, nc in m^-3:
//     nc_entry=1e9,   rc=1e-8 -> nc = 5.4590136e7, nu_c_entry=3  WRF 15
//     nc_entry=1e8,   rc=1e-8 -> nc = 2.8654908e7, nu_c_entry=12 WRF 15
//     nc_entry=1.5e9, rc=1e-7 -> nc = 5.4590140e8, nu_c_entry=3  WRF 4
// Between nu_c = 3 and nu_c = 15 the individual gamma columns move by more
// than TEN orders of magnitude -- ccg(2,n) by 8.9e12x, ocg1(n) by 2.2e11x --
// and the PRODUCTS the rates are built from move by large finite factors:
// ccg(2,n)*ocg1(n) by 40.8x, so lamc by 3.4x and mvd_c with it;
// ccg(3,n)*ocg2(n) by 15.8x for Dc_g; and pnr_wau, being linear in nu_c,
// rescales by 5.  No tolerance absorbs that.  It is a physics error, not a
// rounding one.
//
//   CORRECT:  nc_m3 = thompson_aa_cloud_dist(rc, nc_per_kg, rho, &nu_c_entry,
//                                            &lamc_entry);
//             const int nu_c = thompson_aa_nu_c_working(nc_m3);
//   WRONG:    use nu_c_entry for anything after :1838.
//
// tests/test_thompson_aerosol_device_helpers.py asserts the divergence over
// the full 70-point grid through thompson_aa_probe_nu_c_staging, so a kernel
// that reuses the entry value can be caught without running a network.
//
// ---------------------------------------------------------------------------
// THE CENTRAL POINT: nc IS PROGNOSTIC
// ---------------------------------------------------------------------------
// mp=8 freezes cloud droplet number at Nt_c = 100e6 m^-3.  thompson.cu
// therefore hardcodes 100.0e6f (12 sites), cloud_number_bin = 65 (3 sites)
// and the gamma ratios 2730.0f / 272.0f (7 sites).  In mp=28 all three are
// live functions of nc.  Every value you compute must trace to nc through
// these helpers, never to a literal.  Two HARD IDENTITY GATES prove the
// generalized forms reduce to what mp=8 froze:
//     thompson_aa_droplet_bin(100.0e6f) == 65
//     thompson_aa_in_bin(1000.0f)       == 27
// They are asserted in tests/test_thompson_aerosol_device_helpers.py.
//
// ---------------------------------------------------------------------------
// GAMMA PARITY, AND A PRE-EXISTING mp=8 DEVIATION THAT MUST NOT BE "FIXED"
// ---------------------------------------------------------------------------
// The THOMPSON_AA_CC* tables below are WRF's REAL(4) WGAMMA/GAMMLN Lanczos
// series (5325-5377) evaluated in float32 by
// gpuwm/core/thompson_aerosol_contract.py and emitted here as exact
// round-tripping float literals.  math.lgamma is NOT equivalent: it gives
// Gamma(16) = 1.30767441e12 where WRF gives 1.30767389e12, and
// 1.30767389e12f is the literal already embedded in thompson.cu:2083.
//
// At nu_c = 12 (i.e. nc = Nt_c) the runtime products are
//     ccg(2,12)*ocg1(12) = 2729.9973    (not 2730)
//     ccg(5,12)*ocg2(12) =  272.00012   (not 272)
// thompson.cu hardcodes 2730.0f at :882, :999, :4005, :4128, :4680 and
// 272.0f at :888, :1006, so mp=8 carries a small deviation at those five/two
// sites.  (thompson.cu:343 is NOT one of them: calc_effectRad genuinely uses
// WRF's exact-integer g_ratio PARAMETER, for which 2730 is correct.)
// mp=28 is RIGHT -- it computes from the series via THOMPSON_AA_CCG*/OCG*.
// mp=8 stays frozen and slightly wrong.  DO NOT reconcile them.
//
// ---------------------------------------------------------------------------
// PUBLISHED HELPER API   (signature | units | valid range | WRF authority)
// ---------------------------------------------------------------------------
//
// -- arithmetic primitives (USE THESE in any new aerosol kernel) ------------
//  float thompson_aa_add/sub/mul/div(float a, float b)
//        __fadd_rn/__fsub_rn/__fmul_rn/__fdiv_rn.  nvrtc defaults to
//        --fmad=true; build_aero.sh compiles the oracle with plain
//        `gfortran -O2` on baseline x86-64, which has NO fma instruction, so
//        every REAL(4) multiply and add in WRF is separately rounded.
//        Leaving contraction on cost 1-2 float digits in activ_ncloud,
//        iceKoop and Eff_aero; pinning it made all three BIT-EXACT against
//        the Fortran probe.  Prefer these wherever a fixture disagrees in the
//        last digits.
//
//  float thompson_aa_expf/logf_cr(float x), thompson_aa_powf(float,
//        float)
//        Correctly-rounded float32 EXP/LOG/**, evaluated in double and
//        rounded ONCE.  gfortran lowers REAL(4) EXP/LOG/** to glibc, which is
//        correctly rounded; CUDA's expf/logf/powf carry up to ~2 ulp.  Every
//        operand and every stored result is still float32.
//
// -- rounding / indexing ----------------------------------------------------
//  int   thompson_aa_nint(float x)
//        Fortran NINT: round half AWAY FROM ZERO.  CUDA __float2int_rn
//        rounds half to even and is NOT a substitute.
//
//  int   thompson_aa_nu_c(float nc_m3)
//        nc_m3 [m^-3], must be > 0 (callers clamp to [2, 1.999e9] first).
//        Returns the cloud shape parameter in [2, 15].  :2170, :1832.
//        This is the integer that indexes every THOMPSON_AA_CC*/OCG* table.
//        WHICH nc YOU PASS IS PHYSICS, NOT PLUMBING -- see the nu_c STAGING
//        RULE above and prefer thompson_aa_nu_c_working at every :2170 site.
//
//  int   thompson_aa_nu_c_working(float nc_m3_after_rediagnosis)
//        :2170.  thompson_aa_nu_c under a name that records WHICH nc is
//        legal to pass: the value thompson_aa_cloud_dist RETURNS, never the
//        nu_c_entry_out it writes.
//
//  int   thompson_aa_droplet_bin(float nc_m3)
//        Zero-based idx_n into tnc_wev's third axis, range [0, 99].
//        :3447-3448.  Computed in DOUBLE precision because WRF uses DLOG of
//        a DOUBLE t_Nc(1); nic1 is the TRUNCATED integer 7, not 7.926.
//        GATE: thompson_aa_droplet_bin(100.0e6f) == 65.
//
//  int   thompson_aa_decade_index(float value, int first_exponent,
//                                 int table_size)
//        Zero-based base-ten mantissa/decade bin.  value > 0.  Transcribed
//        from thompson.cu:3063-3082 (itself :2282-2307); the identical
//        pattern appears at :2581-2589 for idx_IN.
//
//  int   thompson_aa_decade_index_double(double value, int first_exponent,
//                                        int table_size)
//        The DOUBLE spelling of the same rule, thompson.cu:3084-3105, for
//        WRF's DOUBLE PRECISION rain and graupel y-intercepts (N0_r/N0_g,
//        :1587).  Promoted out of cold.cu and warm.cu in wave 4; see the
//        PUBLISHED SHARED SIGNATURES block.  Two spellings exist because WRF
//        has two, not because the rule differs.
//
//  int   thompson_aa_in_bin(float xni)
//        Zero-based idx_IN into the freezeH2O tables, range [0, 54].
//        xni [m^-3] is the ice-nuclei number from thompson_ice_demott.
//        :2579-2591.  GATE: thompson_aa_in_bin(1000.0f) == 27, which is the
//        nuclei_bin thompson.cu:3936 hardcodes.
//
// -- saturation and snow-moment fits (transcribed from WRF, NOT from mp=8) --
//  float thompson_rslf(float p_pa, float t_k)      :5378-5413
//  float thompson_rsif(float p_pa, float t_k)      :5414-5446
//  float thompson_field_a(float tc, float moment)  :2069-2075 (sa, :358-359)
//  float thompson_field_b(float tc, float moment)  :2076-2079 (sb, :361-362)
//        ALL FOUR now diverge from thompson.cu's plain chains BY DESIGN.  The
//        two saturation fits are contraction-pinned (see "THE SHARED FITS"
//        above thompson_aa_add); the two snow-moment fits additionally
//        restore WRF's LEFT-TO-RIGHT operator association, which the copied
//        mp=8 form broke by hoisting tc*tc and moment*moment, and evaluate
//        `10.0**loga_` correctly rounded.  Measured bit-exact against
//        gfortran -O2 on 253 states and against the real calc_effectRad on
//        360 more; the numbers and the two defects are recorded above
//        thompson_field_a.
//
// -- aerosol physics --------------------------------------------------------
//  float thompson_activ_ncloud(float Tt, float Ww, float NCCN,
//                              const double* tnccn_act)
//        Tt [K], Ww [m s^-1], NCCN [m^-3] (water-friendly aerosol).
//        tnccn_act is WP-01's float64 FORTRAN-ORDER (7,9,7,5,4) device
//        array.  Returns activated droplet number [m^-3].  :5178-5253.
//        NEAREST-NEIGHBOUR (not interpolated) in temperature; bilinear in
//        log(N) and log(w); aerosol radius index l=3 and kappa index m=2 are
//        hardcoded by WRF.  Both clamps use ASYMMETRIC epsilons.
//        TOLERANCE NOTE: k is a nearest 10 K bin and i/j are bracket
//        indices, so the result is a STEP function of state near a bin edge.
//
//  float thompson_ice_demott(float tempc, float rho, float nifa_m3)
//        tempc [degC, < 0], rho [kg m^-3], nifa_m3 [m^-3].  Returns ice
//        nuclei number [m^-3].  :5448-5518.  NEGATIVE FINDING: in v4.6.1 the
//        Phillips (2008) branch (5474-5505) is entirely commented out, so
//        this is a PURE function of (tempc, rho, nifa) -- do not pass
//        qv/qvs/qvsi and do not port the commented code.  WRF's two call
//        sites (:2574, :2623) differ only in the dead qv argument and return
//        the identical value; evaluate ONCE per level.
//
//  float thompson_ice_koop(float temp_k, float qv, float qvs,
//                          float nwfa_m3, float dt)
//        Homogeneous freezing of deliquesced haze [m^-3].  :5521-5546.
//        log_J capped at 20; result capped at 1000.e3.  Gated by WRF at
//        :2634-2637 on temp < 238 K, ssati >= 0.4 and ns+ni <= 999.e3 --
//        the caller applies that gate, not this helper.
//
//  float thompson_eff_aero(float D, float Da, float visc, float rhoa,
//                          float temp_k, int species)
//        Aerosol collection efficiency of a collector drop/crystal, Slinn
//        (1983) via Wang et al (2010).  D [m] collector diameter, Da [m]
//        aerosol diameter, visc [kg m^-1 s^-1], rhoa [kg m^-3], temp_k [K].
//        species is THOMPSON_AA_SPECIES_RAIN / _SNOW / _GRAUPEL.  Result is
//        clamped to [1e-5, 1.0].  :4965-5001.  The graupel fall speed uses
//        av_g(idx_bg1)=442.0, bv_g(idx_bg1)=0.89, which is what
//        thompson_init installs when ng is absent (:462-465) -- i.e. the
//        non-hail-aware mp=28 configuration this port targets.
//
//  float thompson_aa_snow_number(float smob, float smoc)
//        WRF's EXPLICIT two-gamma snow number ns(k) [m^-3], :2083-2088.
//        smob is the bm_s-th (== 2nd) snow moment, smoc the (bm_s+1)-th.
//        This is NOT smo0 and is not interchangeable with it.  It is needed
//        only by the Koop homogeneous-freezing gate at :2634, where it enters
//        an ADDITIVE threshold test (xni = ns+ni+... <= 999.e3), so an error
//        here does not perturb a rate, it flips a branch.
//        No WRF procedure exposes it -- Kap0/Kap1/Lam0/Lam1 (:114-117),
//        mu_s (:113) and csg/cse are all PRIVATE -- so the gate is a
//        `gfortran -O2` transcription of :2029-2088 built with build_aero.sh's
//        own flags, over 391 states (23 temperatures x 17 snow contents,
//        ns from 1.5e-1 to 2.6e+08 m^-3).  MEASURED BIT-EXACT on all 391.
//        CALLER NOTE: WRF sets ns(k) = 0 at :1790 and only overwrites it
//        inside the `if (.not. L_qs(k)) CYCLE` loop at :2026-2088, so at a
//        snow-free level the Koop gate sees ns = 0, not a stale value.
//
// -- droplet distribution ---------------------------------------------------
//  float thompson_aa_cloud_dist(float rc, float nc_per_kg, float rho,
//                               int* nu_c_entry_out, double* lamc_entry_out)
//        :1826-1842.  rc [kg m^-3] cloud water CONTENT (qc*rho),
//        nc_per_kg [kg^-1] entry droplet number, rho [kg m^-3].
//        RETURNS the rediagnosed nc [m^-3] (:1840) after the D0c / 2*D0r size
//        clamp.  The two out-parameters are WRF's ENTRY-STAGE values from
//        :1832-1838 and are NOT usable downstream: nu_c_entry_out is computed
//        from the PRE-rediagnosis nc and lamc_entry_out is the clamped lambda
//        that produced the return value.  Everything after :1838 recomputes
//        both from the RETURNED nc -- see the nu_c STAGING RULE above.
//        Caller must have rc > R1.
//
//  int   thompson_aa_inu_c_effrad(float nc_m3)
//        calc_effectRad's THREE-branch shape selector, :5637-5643.  It is
//        deliberately different from thompson_aa_nu_c: nc < 100 -> 15,
//        nc > 1e10 -> 2 (dead code, the preceding Nt_c_max clamp forbids
//        it), else MIN(15, NINT(1000e6/nc)+2).
//
//  float thompson_aa_eff_rad_cloud(float rc, float nc_m3)   :5636-5646
//  float thompson_aa_eff_rad_ice(float ri, float ni)        :5650-5656
//  float thompson_aa_eff_rad_snow(float rs, float t_k)      :5658-5694
//        Effective radii in METRES, already carrying calc_effectRad's own
//        clamps.  mp_gt_driver's second clamp (:1475-1477) and gpuwm's
//        metre->micron convention are the CALLER's job (see
//        thompson.cu:373-381 for the mp=8 precedent).
//
// -- WRF's terminal clamps (:1805-1806, :3217, :3486, :3979-3981) -----------
//  float thompson_aa_clamp_nc(float nc_m3)     -> [2, 1.999e9]
//  float thompson_aa_clamp_nwfa(float nwfa_m3) -> [11.1e6, 9999e6]
//  float thompson_aa_clamp_nifa(float nifa_m3) -> [5.0e3, 9999e6]
//
// -- tables (index 0 unused so device code reads like Fortran) --------------
//  THOMPSON_AA_CCE1..CCE5[16], THOMPSON_AA_CCG1..CCG5[16],
//  THOMPSON_AA_OCG1[16], THOMPSON_AA_OCG2[16]   -- :671-685, WGAMMA-FP32.
//  THOMPSON_AA_G_RATIO[16]                      -- :5611-5613, EXACT
//        integers (n+1)(n+2)(n+3).  Used ONLY by calc_effectRad.  It is NOT
//        interchangeable with CCG2*OCG1 (2730 vs 2729.9973).
//
#pragma once

// ---------------------------------------------------------------------------
// Scalar parameters.  Each is the module_mp_thompson.F REAL(4) value; where
// WRF forms a compile-time product the float32 result is written out.
// ---------------------------------------------------------------------------
#define THOMPSON_AA_PI            3.1415926536f   // :67
#define THOMPSON_AA_R_DRY         287.04f         // module_model_constants R
// DELIBERATELY ABSENT: Nt_c (:88, 100.0e6).  It is the single literal this
// port must never read as a droplet number -- mp=8 freezes nc at it, mp=28
// carries nc prognostically -- and a #define here would make it visible to
// all six aerosol translation units for the benefit of no production kernel.
// The identity gates that prove the generalized forms reduce to what mp=8
// froze consume NT_C from gpuwm/core/thompson_aerosol_contract.py:104, on the
// host, where it cannot be reached by device code.
#define THOMPSON_AA_NT_C_MAX      1999.0e6f       // :89
#define THOMPSON_AA_NC_FLOOR      2.0f            // :1830, :3217, :3486
#define THOMPSON_AA_NWFA_FLOOR    11.1e6f         // :1805
#define THOMPSON_AA_NIFA_FLOOR    5.0e3f          // :1806  (naIN1*0.01)
#define THOMPSON_AA_AERO_CEIL     9999.0e6f       // :1805-1806, :3979-3981
#define THOMPSON_AA_R1            1.0e-12f        // :183
#define THOMPSON_AA_R2            1.0e-6f         // :184
#define THOMPSON_AA_AM_R          5.235988159e+02f  // :128 PI*rho_w/6
#define THOMPSON_AA_BM_R          3.0f            // :129
#define THOMPSON_AA_OBMR          3.333333433e-01f  // :721 1./bm_r in REAL(4)
#define THOMPSON_AA_AM_I          4.660029297e+02f  // :137 PI*rho_i/6
#define THOMPSON_AA_OBMI          3.333333433e-01f  // :703 1./bm_i
#define THOMPSON_AA_CIG2          6.0f            // :695 WGAMMA(4) exactly 6
#define THOMPSON_AA_OIG1          1.0f            // :701 1./WGAMMA(1)
#define THOMPSON_AA_MU_I          0.0f            // :105
#define THOMPSON_AA_D0C           1.0e-6f         // :224
#define THOMPSON_AA_D0R           50.0e-6f        // :225
#define THOMPSON_AA_AV_C          0.316946e8f     // :163
#define THOMPSON_AA_BV_C          2.0f            // :164
#define THOMPSON_AA_AM_S          0.069f          // :130
#define THOMPSON_AA_OAMS          1.449275398e+01f  // :747 1./am_s
#define THOMPSON_AA_BM_S          2.0f            // :131
#define THOMPSON_AA_AV_S          40.0f           // :146
#define THOMPSON_AA_BV_S          0.55f           // :147
#define THOMPSON_AA_AV_G          442.0f          // :149 av_g_old == av_g(5)
#define THOMPSON_AA_BV_G          0.89f           // :150 bv_g_old == bv_g(5)
#define THOMPSON_AA_MU_S          0.6357f         // :113
#define THOMPSON_AA_KAP0          490.6f          // :114
#define THOMPSON_AA_KAP1          17.46f          // :115
#define THOMPSON_AA_LAM0          20.78f          // :116
#define THOMPSON_AA_LAM1          3.29f           // :117
#define THOMPSON_AA_CSE15         1.635699987e+00f  // :741 mu_s + 1.
#define THOMPSON_AA_CSG15         8.980315328e-01f  // WGAMMA(cse(15))
#define THOMPSON_AA_R_UNI         8.314f          // :206
#define THOMPSON_AA_AR_VOLUME     6.544983879e-17f  // :213 4./3.*PI*(2.5e-6)**3
#define THOMPSON_AA_RHO_NOT0      1.292283773e+00f  // :5470 101325/(287.05*273.15)
#define THOMPSON_AA_RE_QC_BG      2.49e-6f        // module_model_constants
#define THOMPSON_AA_RE_QI_BG      4.99e-6f
#define THOMPSON_AA_RE_QS_BG      9.99e-6f

// Table extents, :231-252.
#define THOMPSON_AA_NBC       100
#define THOMPSON_AA_NTB_C     37
#define THOMPSON_AA_NTB_IN    55
#define THOMPSON_AA_NTB_ARC   7
#define THOMPSON_AA_NTB_ARW   9
#define THOMPSON_AA_NTB_ART   7
#define THOMPSON_AA_NTB_ARR   5
#define THOMPSON_AA_NTB_ARK   4

// activ_ncloud:5229-5230.  One-based and hardcoded by WRF: mean aerosol
// radius 0.04 um and hygroscopicity kappa 0.4.  Four fifths of the shipped
// CCN_ACTIVATE.BIN is therefore never read.
#define THOMPSON_AA_ACTIV_L   3
#define THOMPSON_AA_ACTIV_M   2

// Eff_aero collector species selector (:4974-4980).  WRF passes CHARACTER*1.
#define THOMPSON_AA_SPECIES_RAIN     0
#define THOMPSON_AA_SPECIES_SNOW     1
#define THOMPSON_AA_SPECIES_GRAUPEL  2
// The hail-aware scheme (mp_physics=38) keeps the Heymsfield pair in slot
// idx_bg1 = 5 (av_g = 143.204224, bv_g = 0.640961647, :152-158), so its
// Eff_aero graupel fall speed av_g(idx_bg1)*D**bv_g(idx_bg1) (:4981) is not
// the 442/0.89 mp=8/28 install at :463-464.
#define THOMPSON_AA_SPECIES_GRAUPEL_HAIL  3

// t_Nc(1), :720-731.  DOUBLE PRECISION in WRF and DLOG'd there, so the
// droplet-bin index must be formed in double.  nic1 is declared INTEGER at
// :246 and assigned a DOUBLE at :896, so Fortran TRUNCATES 7.926303892 to 7.
#define THOMPSON_AA_T_NC_1    1040843.9118849806
#define THOMPSON_AA_NIC1      7

// ---------------------------------------------------------------------------
// Gamma moment tables, module_mp_thompson.F:671-685, one column per integer
// nu_c in 1..15.  Emitted from gpuwm.core.thompson_aerosol_contract's
// transcription of WRF's REAL(4) WGAMMA/GAMMLN; every literal round-trips
// exactly through float32.  Element 0 is an unused zero.
//     cce(1,n) = n + 1                cce(2,n) = bm_r + n + 1
//     cce(3,n) = bm_r + n + 4         cce(4,n) = n + bv_c + 1
//     cce(5,n) = bm_r + n + bv_c + 1  ccg(r,n) = WGAMMA(cce(r,n))
//     ocg1(n)  = 1./ccg(1,n)          ocg2(n)  = 1./ccg(2,n)
// ---------------------------------------------------------------------------
__constant__ float THOMPSON_AA_CCE1[16] = {
    0.000000000e+00f, 2.000000000e+00f, 3.000000000e+00f, 4.000000000e+00f,
    5.000000000e+00f, 6.000000000e+00f, 7.000000000e+00f, 8.000000000e+00f,
    9.000000000e+00f, 1.000000000e+01f, 1.100000000e+01f, 1.200000000e+01f,
    1.300000000e+01f, 1.400000000e+01f, 1.500000000e+01f, 1.600000000e+01f};

__constant__ float THOMPSON_AA_CCE2[16] = {
    0.000000000e+00f, 5.000000000e+00f, 6.000000000e+00f, 7.000000000e+00f,
    8.000000000e+00f, 9.000000000e+00f, 1.000000000e+01f, 1.100000000e+01f,
    1.200000000e+01f, 1.300000000e+01f, 1.400000000e+01f, 1.500000000e+01f,
    1.600000000e+01f, 1.700000000e+01f, 1.800000000e+01f, 1.900000000e+01f};

__constant__ float THOMPSON_AA_CCE3[16] = {
    0.000000000e+00f, 8.000000000e+00f, 9.000000000e+00f, 1.000000000e+01f,
    1.100000000e+01f, 1.200000000e+01f, 1.300000000e+01f, 1.400000000e+01f,
    1.500000000e+01f, 1.600000000e+01f, 1.700000000e+01f, 1.800000000e+01f,
    1.900000000e+01f, 2.000000000e+01f, 2.100000000e+01f, 2.200000000e+01f};

__constant__ float THOMPSON_AA_CCE4[16] = {
    0.000000000e+00f, 4.000000000e+00f, 5.000000000e+00f, 6.000000000e+00f,
    7.000000000e+00f, 8.000000000e+00f, 9.000000000e+00f, 1.000000000e+01f,
    1.100000000e+01f, 1.200000000e+01f, 1.300000000e+01f, 1.400000000e+01f,
    1.500000000e+01f, 1.600000000e+01f, 1.700000000e+01f, 1.800000000e+01f};

__constant__ float THOMPSON_AA_CCE5[16] = {
    0.000000000e+00f, 7.000000000e+00f, 8.000000000e+00f, 9.000000000e+00f,
    1.000000000e+01f, 1.100000000e+01f, 1.200000000e+01f, 1.300000000e+01f,
    1.400000000e+01f, 1.500000000e+01f, 1.600000000e+01f, 1.700000000e+01f,
    1.800000000e+01f, 1.900000000e+01f, 2.000000000e+01f, 2.100000000e+01f};

__constant__ float THOMPSON_AA_CCG1[16] = {
    0.000000000e+00f, 1.000000000e+00f, 2.000000000e+00f, 6.000000000e+00f,
    2.400000000e+01f, 1.200000076e+02f, 7.200000610e+02f, 5.040001953e+03f,
    4.031999609e+04f, 3.628799688e+05f, 3.628801750e+06f, 3.991680000e+07f,
    4.790018560e+08f, 6.227022336e+09f, 8.717829734e+10f, 1.307673887e+12f};

__constant__ float THOMPSON_AA_CCG2[16] = {
    0.000000000e+00f, 2.400000000e+01f, 1.200000076e+02f, 7.200000610e+02f,
    5.040001953e+03f, 4.031999609e+04f, 3.628799688e+05f, 3.628801750e+06f,
    3.991680000e+07f, 4.790018560e+08f, 6.227022336e+09f, 8.717829734e+10f,
    1.307673887e+12f, 2.092278219e+13f, 3.556874482e+14f, 6.402383731e+15f};

__constant__ float THOMPSON_AA_CCG3[16] = {
    0.000000000e+00f, 5.040001953e+03f, 4.031999609e+04f, 3.628799688e+05f,
    3.628801750e+06f, 3.991680000e+07f, 4.790018560e+08f, 6.227022336e+09f,
    8.717829734e+10f, 1.307673887e+12f, 2.092278219e+13f, 3.556874482e+14f,
    6.402383731e+15f, 1.216452850e+17f, 2.432903398e+18f, 5.109091445e+19f};

__constant__ float THOMPSON_AA_CCG4[16] = {
    0.000000000e+00f, 6.000000000e+00f, 2.400000000e+01f, 1.200000076e+02f,
    7.200000610e+02f, 5.040001953e+03f, 4.031999609e+04f, 3.628799688e+05f,
    3.628801750e+06f, 3.991680000e+07f, 4.790018560e+08f, 6.227022336e+09f,
    8.717829734e+10f, 1.307673887e+12f, 2.092278219e+13f, 3.556874482e+14f};

__constant__ float THOMPSON_AA_CCG5[16] = {
    0.000000000e+00f, 7.200000610e+02f, 5.040001953e+03f, 4.031999609e+04f,
    3.628799688e+05f, 3.628801750e+06f, 3.991680000e+07f, 4.790018560e+08f,
    6.227022336e+09f, 8.717829734e+10f, 1.307673887e+12f, 2.092278219e+13f,
    3.556874482e+14f, 6.402383731e+15f, 1.216452850e+17f, 2.432903398e+18f};

__constant__ float THOMPSON_AA_OCG1[16] = {
    0.000000000e+00f, 1.000000000e+00f, 5.000000000e-01f, 1.666666716e-01f,
    4.166666791e-02f, 8.333332837e-03f, 1.388888806e-03f, 1.984126284e-04f,
    2.480158946e-05f, 2.755732112e-06f, 2.755730577e-07f, 2.505210794e-08f,
    2.087674478e-09f, 1.605904021e-10f, 1.147074449e-11f, 7.647166320e-13f};

__constant__ float THOMPSON_AA_OCG2[16] = {
    0.000000000e+00f, 4.166666791e-02f, 8.333332837e-03f, 1.388888806e-03f,
    1.984126284e-04f, 2.480158946e-05f, 2.755732112e-06f, 2.755730577e-07f,
    2.505210794e-08f, 2.087674478e-09f, 1.605904021e-10f, 1.147074449e-11f,
    7.647166320e-13f, 4.779478950e-14f, 2.811457147e-15f, 1.561918299e-16f};

// calc_effectRad:5611-5613.  EXACT integers (n+1)(n+2)(n+3) for n = 2..16,
// written out verbatim in WRF as a PARAMETER.  Do NOT substitute
// CCG2[n]*OCG1[n]: at n=12 the runtime product is 2729.9973, not 2730.
__constant__ float THOMPSON_AA_G_RATIO[16] = {
    0.000000000e+00f, 2.400000000e+01f, 6.000000000e+01f, 1.200000000e+02f,
    2.100000000e+02f, 3.360000000e+02f, 5.040000000e+02f, 7.200000000e+02f,
    9.900000000e+02f, 1.320000000e+03f, 1.716000000e+03f, 2.184000000e+03f,
    2.730000000e+03f, 3.360000000e+03f, 4.080000000e+03f, 4.896000000e+03f};

// activ_ncloud axis values, module_mp_thompson.F:335-344.  ta_Ra and ta_Ka
// are not needed on device because l and m are hardcoded.
__constant__ float THOMPSON_AA_TA_NA[THOMPSON_AA_NTB_ARC] = {
    10.0f, 31.6f, 100.0f, 316.0f, 1000.0f, 3160.0f, 10000.0f};

__constant__ float THOMPSON_AA_TA_WW[THOMPSON_AA_NTB_ARW] = {
    0.01f, 0.0316f, 0.1f, 0.316f, 1.0f, 3.16f, 10.0f, 31.6f, 100.0f};

// Snow moment fit coefficients sa/sb, module_mp_thompson.F:167-181.  These
// are byte-identical to thompson.cu's thompson_sa/thompson_sb; they are
// duplicated, not shared, because thompson.cu is byte-frozen.
__constant__ float THOMPSON_AA_SA[10] = {
    5.065339f, -0.062659f, -3.032362f, 0.029469f, -0.000285f,
    0.31255f, 0.000204f, 0.003199f, 0.0f, -0.015952f};

__constant__ float THOMPSON_AA_SB[10] = {
    0.476221f, -0.015896f, 0.165977f, 0.007468f, -0.000141f,
    0.060366f, 0.000079f, 0.000594f, 0.0f, -0.003577f};


// ---------------------------------------------------------------------------
// Rounding.
// ---------------------------------------------------------------------------

// Fortran NINT rounds half AWAY FROM ZERO.  CUDA's __float2int_rn rounds half
// to even and would select a different bin at an exact .5 boundary.  Every
// index in this header goes through here or through floorf(x+0.5f) where the
// argument is provably positive.
__device__ __forceinline__ int thompson_aa_nint(float x)
{
    return (int)(x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f));
}

__device__ __forceinline__ int thompson_aa_nint_double(double x)
{
    return (int)(x >= 0.0 ? floor(x + 0.5) : ceil(x - 0.5));
}


// WRF's EXP/LOG/LOG10/** are WOOF's own libm words: thompson_aerosol_libm.cuh,
// prepended to every mp=28 unit ahead of this header.



// ---------------------------------------------------------------------------
// Contraction-pinned float32 arithmetic.
// ---------------------------------------------------------------------------
//
// nvrtc defaults to --fmad=true and will fuse a*b+c into one FMA with a
// single rounding.  build_aero.sh compiles the oracle with plain
// `gfortran -O2` and no -march, i.e. baseline x86-64 with no FMA
// instruction, so every REAL(4) multiply and add in WRF is separately
// rounded.  Measured: with contraction left on, activ_ncloud, iceKoop and
// Eff_aero disagree with the Fortran probe in the last one or two float
// digits; with every operation pinned they agree BIT-EXACTLY on all 1320 /
// 480 / 48 probe rows.
//
// Pinning it here rather than with a -fmad=false compile option keeps
// gpuwm/core/kernels/__init__.py's change to the _EXTRA_HEADERS dict alone,
// and gives the five aerosol kernel packages the property automatically.
// Repo precedent: noahmp_bareflux.cu, noahmp_thermal.cu.
//
// These are applied to the aerosol-only helpers (activ_ncloud, iceDeMott,
// iceKoop, Eff_aero), which have no mp=8 counterpart, AND -- deliberately --
// to thompson_rslf/thompson_rsif, which do.
//
// THE SHARED FITS: WHY mp=28 DIVERGES FROM ITS mp=8 SIBLING HERE.
// This header originally kept thompson_rslf/rsif as verbatim copies of
// thompson.cu's plain Horner chains, so that the aerosol port and the
// model-validated mp=8 port would evaluate the shared fits identically.
// That is the wrong tie-break, and mp=28 is where it becomes visible.
//
// The authority is WRF, not ArWen's mp=8.  nvrtc contracts the plain chain
// into FMAs; the Fortran WRF the oracle is built from has no FMA instruction
// at all, so the contracted device fit lands ONE float32 ulp low -- measured
// at 23 of 24 levels of the aero-nc-sed entry column.  In mp=8 that is a
// sub-atol mass error and nothing notices.  In mp=28 it is not a rate
// perturbation at all: module_mp_thompson.F:3400 opens the ENTIRE
// condensation and CCN-activation block on `ssatw(k) .gt. eps` with
// eps = 1.E-15 (:185), so one ulp FLIPS A BRANCH.  The call then condenses
// water at every cloud-free level of a saturated column and activates
// droplets from aerosol that WRF never touches.  Because activation is a
// ONE-WAY SINK, the error does not average out -- it destroyed 33-56% of a
// column's CCN in a single 10-50 s step.
//
// So these two fits are pinned and mp=8's are not.  The two ports therefore
// evaluate RSLF/RSIF differently BY DESIGN: mp=28 matches WRF, mp=8 stays
// frozen and byte-identical to its validated trajectory.  Do not "restore
// consistency" by unpinning these -- consistency with a sibling is not the
// goal, agreement with WRF is.  thompson.cu carries the same unpinned chain
// at :48-58 and is a pre-existing ArWen-wide deviation from WRF that this
// port merely made observable; correcting it there is a separate decision
// about a model-validated trajectory and is NOT in this port's scope.
//
// THE SAME TIE-BREAK NOW APPLIES TO thompson_field_a/thompson_field_b.  They
// used to keep thompson.cu's plain form on the grounds that nothing had
// measured them.  They have now been measured, against a `gfortran -O2`
// transcription of module_mp_thompson.F:2069-2079 AND against the real
// (PUBLIC) calc_effectRad, and the copied mp=8 form was wrong by up to
// 3.267395e-06 on a_ and 5.870704e-06 on the effs_m it feeds -- from a broken
// operator association as much as from FMA and pow.  Both are now WRF's form
// and both are bit-exact.  See the block above thompson_field_a for the
// numbers.  mp=8 keeps thompson.cu:81-107 frozen; mp=28 matches WRF.
__device__ __forceinline__ float thompson_aa_add(float a, float b)
{
    return __fadd_rn(a, b);
}

__device__ __forceinline__ float thompson_aa_sub(float a, float b)
{
    return __fsub_rn(a, b);
}

__device__ __forceinline__ float thompson_aa_mul(float a, float b)
{
    return __fmul_rn(a, b);
}

__device__ __forceinline__ float thompson_aa_div(float a, float b)
{
    return __fdiv_rn(a, b);
}

// WRF paired collision transfer, module_mp_thompson.F:2945-2954.
// Restore the delivered mp28 helper with its REAL rounding and signed zero.
__device__ __forceinline__ void thompson_aa_reenforce_pair(double* a,
                                                        double* b)
{
    const float ratio = (float)fmin(fabs(*a), fabs(*b));
    *a = (double)thompson_aa_mul(ratio, copysignf(1.0f, (float)*a));
    *b = -*a;
}


// ---------------------------------------------------------------------------
// WRF's terminal clamps.  :1805-1806 (entry), :3217/:3486 (working refresh),
// :3979-3981 (terminal apply).  Reproduced here so five packages cannot each
// invent a slightly different one.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float thompson_aa_clamp_nc(float nc_m3)
{
    return fmaxf(THOMPSON_AA_NC_FLOOR, fminf(nc_m3, THOMPSON_AA_NT_C_MAX));
}

__device__ __forceinline__ float thompson_aa_clamp_nwfa(float nwfa_m3)
{
    return fmaxf(THOMPSON_AA_NWFA_FLOOR,
                 fminf(THOMPSON_AA_AERO_CEIL, nwfa_m3));
}

__device__ __forceinline__ float thompson_aa_clamp_nifa(float nifa_m3)
{
    return fmaxf(THOMPSON_AA_NIFA_FLOOR,
                 fminf(THOMPSON_AA_AERO_CEIL, nifa_m3));
}


// ---------------------------------------------------------------------------
// nu_c and the lookup indices mp=8 froze into constants.
// ---------------------------------------------------------------------------

// module_mp_thompson.F:1832, :2171, :3002, :3658.
//     nu_c = MIN(15, NINT(1000.E6/nc(k)) + 2)
// nc is positive at every call site (callers clamp to >= 2 first), so
// floorf(x+0.5f) is the faithful NINT.  The result is in [2, 15]; at
// nc = Nt_c = 100e6 it is exactly 12, which is the value mp=8 froze.
__device__ __forceinline__ int thompson_aa_nu_c(float nc_m3)
{
    return min(15, (int)floorf(1000.0e6f / nc_m3 + 0.5f) + 2);
}

// module_mp_thompson.F:2170, the WORKING nu_c.  Identical arithmetic to
// thompson_aa_nu_c; the separate name exists so that every call site records
// WHICH nc it passed.  WRF recomputes nu_c here from the POST-rediagnosis
// nc(k) assigned at :1840, NOT from the entry nc(k) of :1829 that produced
// thompson_aa_cloud_dist's nu_c_entry_out.  See the nu_c STAGING RULE at the
// top of this header; the two differ wherever the :1834-1838 droplet-size
// clamp engages, and the gamma columns they select move by more than ten
// orders of magnitude between them.
__device__ __forceinline__ int thompson_aa_nu_c_working(
    float nc_m3_after_rediagnosis)
{
    return thompson_aa_nu_c(nc_m3_after_rediagnosis);
}

// module_mp_thompson.F:3447-3448.
//     idx_n = NINT(1.0 + FLOAT(nbc) * DLOG(nc(k)/t_Nc(1)) / nic1)
//     idx_n = MAX(1, MIN(idx_n, nbc))
// t_Nc is DOUBLE PRECISION and nic1 is the truncated INTEGER 7, so the whole
// expression is double.  Returned ZERO-BASED for C indexing.
// HARD GATE: thompson_aa_droplet_bin(100.0e6f) == 65.
__device__ __forceinline__ int thompson_aa_droplet_bin(float nc_m3)
{
    const double raw = 1.0
        + 100.0 * thompson_aa_log((double)nc_m3 / THOMPSON_AA_T_NC_1)
          / (double)THOMPSON_AA_NIC1;
    const int one_based = thompson_aa_nint_double(raw);
    return max(1, min(one_based, THOMPSON_AA_NBC)) - 1;
}

// Transcribed from thompson.cu:3063-3082 (mp=8's copy of :2282-2307), which
// is the same NINT(log10)-then-truncate-the-mantissa pattern WRF uses for
// idx_c, idx_r and idx_IN.  Returned ZERO-BASED.
__device__ __forceinline__ int thompson_aa_decade_index(
    float value, int first_exponent, int table_size)
{
    const int center = (int)roundf(thompson_aa_log10f(value));
    int exponent = center;
    for (int candidate = center - 1; candidate <= center + 1; ++candidate) {
        const float scale = thompson_aa_powf(10.0f, (float)candidate);
        const float mantissa = value / scale;
        if (mantissa >= 1.0f && mantissa < 10.0f) {
            exponent = candidate;
            break;
        }
    }
    const float scale = thompson_aa_powf(10.0f, (float)exponent);
    const int digit = (int)(value / scale);
    const int one_based = digit + 9 * (exponent - first_exponent);
    return max(0, min(one_based - 1, table_size - 1));
}

// The DOUBLE form, thompson.cu:3084-3105.  WRF keeps the rain and graupel
// y-intercepts in DOUBLE PRECISION (N0_r, N0_g at :1587) and indexes
// t_Nor/t_Nog with them, so the mantissa split has to be done in double; the
// base-ten SCALE stays default REAL, which is why `scale` below is a float
// and the division that uses it is widened rather than the other way round.
//
// PROMOTED, wave 4.  cold.cu:206-223 and warm.cu:272-289 each carried this
// body, byte-identical to each other and to what is written here (verified by
// diff before the move).  There was no divergence yet -- and that is the
// point: thompson_aa_entry_rain_distribution had none either, right up until
// it did.  The two local copies MUST NOW BE DELETED; until they are, those
// two translation units fail to compile with
//     error: function "thompson_aa_decade_index_double" has already been
//     defined (previous definition at line 23)
// which is the correct and intended outcome for a helper whose shared
// signature matches the local one exactly.  MEASURED: with both local copies
// removed all six aerosol modules compile, and the full 19-fixture G3 table
// is BIT-IDENTICAL to the pre-promotion tree.  Do not rename either copy, and
// do not delete this definition to make the build go green.
__device__ __forceinline__ int thompson_aa_decade_index_double(
    double value, int first_exponent, int table_size)
{
    const int center = (int)round(thompson_aa_log10(value));
    int exponent = center;
    for (int candidate = center - 1; candidate <= center + 1; ++candidate) {
        const float scale = thompson_aa_powf(10.0f, (float)candidate);
        const double mantissa = value / (double)scale;
        if (mantissa >= 1.0 && mantissa < 10.0) {
            exponent = candidate;
            break;
        }
    }
    const float scale = thompson_aa_powf(10.0f, (float)exponent);
    const int digit = (int)(value / (double)scale);
    const int one_based = digit + 9 * (exponent - first_exponent);
    return max(0, min(one_based - 1, table_size - 1));
}

// module_mp_thompson.F:2579-2591.
//     if (xni .gt. Nt_IN(1)) then     ! Nt_IN(1) = 1.0
//        idx_IN = INT(xni/10.**n) + 10*(n-niin2) - (n-niin2)
//        idx_IN = MAX(1, MIN(idx_IN, ntb_IN))
//     else
//        idx_IN = 1
// niin2 = NINT(ALOG10(Nt_IN(1))) = 0 (:828), and
// 10*(n-niin2) - (n-niin2) == 9*(n-niin2), i.e. first_exponent = 0.
// Returned ZERO-BASED.
// HARD GATE: thompson_aa_in_bin(1000.0f) == 27, the nuclei_bin thompson.cu
// hardcodes at :3936 for its fixed 1-per-litre default.
__device__ __forceinline__ int thompson_aa_in_bin(float xni)
{
    if (xni > 1.0f) {
        return thompson_aa_decade_index(xni, 0, THOMPSON_AA_NTB_IN);
    }
    return 0;
}


// ---------------------------------------------------------------------------
// Saturation vapour mixing ratios and the snow-moment power-law fits.
// Transcribed (not shared) from module_mp_thompson.F; byte-for-byte the same
// arithmetic thompson.cu:18-77 performs, because thompson.cu may not be
// edited and there is no #include path under RawModule.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float thompson_rslf(float pressure, float temp)
{
    // module_mp_thompson.F:5378-5413.  Preserve WRF's default-REAL Horner
    // order exactly.
    const float c0 = 0.611583699e3f;
    const float c1 = 0.444606896e2f;
    const float c2 = 0.143177157e1f;
    const float c3 = 0.264224321e-1f;
    const float c4 = 0.299291081e-3f;
    const float c5 = 0.203154182e-5f;
    const float c6 = 0.702620698e-8f;
    const float c7 = 0.379534310e-11f;
    const float c8 = -0.321582393e-13f;
    const float x = fmaxf(-80.0f, temp - 273.16f);
    // CONTRACTION-PINNED, and deliberately NOT identical to thompson.cu's
    // plain chain.  See the "shared fits" note above thompson_aa_add.
    float esl = thompson_aa_add(c0, thompson_aa_mul(x, thompson_aa_add(c1, thompson_aa_mul(x, thompson_aa_add(c2, thompson_aa_mul(x, thompson_aa_add(c3, thompson_aa_mul(x, thompson_aa_add(c4, thompson_aa_mul(x, thompson_aa_add(c5, thompson_aa_mul(x, thompson_aa_add(c6, thompson_aa_mul(x, thompson_aa_add(c7, thompson_aa_mul(x, c8))))))))))))))));
    esl = fminf(esl, pressure * 0.15f);
    return 0.622f * esl / (pressure - esl);
}

__device__ __forceinline__ float thompson_rsif(float pressure, float temp)
{
    // module_mp_thompson.F:5414-5446.
    const float c0 = 0.609868993e3f;
    const float c1 = 0.499320233e2f;
    const float c2 = 0.184672631e1f;
    const float c3 = 0.402737184e-1f;
    const float c4 = 0.565392987e-3f;
    const float c5 = 0.521693933e-5f;
    const float c6 = 0.307839583e-7f;
    const float c7 = 0.105785160e-9f;
    const float c8 = 0.161444444e-12f;
    const float x = fmaxf(-80.0f, temp - 273.16f);
    // CONTRACTION-PINNED, as thompson_rslf.  Same reasoning.
    float esi = thompson_aa_add(c0, thompson_aa_mul(x, thompson_aa_add(c1, thompson_aa_mul(x, thompson_aa_add(c2, thompson_aa_mul(x, thompson_aa_add(c3, thompson_aa_mul(x, thompson_aa_add(c4, thompson_aa_mul(x, thompson_aa_add(c5, thompson_aa_mul(x, thompson_aa_add(c6, thompson_aa_mul(x, thompson_aa_add(c7, thompson_aa_mul(x, c8))))))))))))))));
    esi = fminf(esi, pressure * 0.15f);
    return 0.622f * esi / fmaxf(1.0e-4f, pressure - esi);
}

// THE FIELD ET AL (2005) SNOW-MOMENT FITS.  WRF has no `field_a` procedure:
// the ten-term chain is written out INLINE at :2036-2046, :2050-2053,
// :2056-2064, :2069-2079, :2091-2100, :2103-2112, :2116-2125, :3332-3350,
// :4447-4470, :5670-5680 and :5684-5693, always with the SAME operator tree
// and only the moment symbol changing (bm_s, cse(1), cse(13), cse(14),
// cse(16), cse(17), or a literal 1. / absent factor).  These two helpers are
// that tree with the moment as an argument.
//
// TWO DEFECTS WERE MEASURED HERE AND FIXED, AND THE FORM BELOW IS NOT
// thompson.cu's.  The earlier body was a verbatim copy of thompson.cu:81-107
// and it disagreed with WRF in two independent ways:
//
//   1. ASSOCIATION.  WRF writes `sa(5)*tc0*tc0`, which Fortran evaluates
//      LEFT TO RIGHT as (sa5*tc0)*tc0.  The copied form hoisted `tc2 = tc*tc`
//      and computed sa5*(tc*tc).  In float32 with no FMA those are different
//      numbers.  The same mismatch applied to sa(6)*m*m, sa(7)*tc0*tc0*m,
//      sa(8)*tc0*m*m, sa(9)*tc0**3 and sa(10)*m**3 -- six of the ten terms.
//   2. TRANSCENDENTAL.  `a_ = 10.0**loga_` is REAL(4)**REAL(4), which
//      gfortran lowers to glibc's correctly-rounded powf; CUDA's powf carries
//      several ulp.
//
// Plus nvrtc's default FMA contraction across the nine additions, which the
// baseline-x86-64 `gfortran -O2` the oracle is built with cannot do.
//
// MEASURED on an RTX 5090 against a `gfortran -O2 -ffree-form` transcription
// of :2069-2079 (same compiler and flags as
// tools/thompson_wrf461_oracle/build_aero.sh, i.e. no FMA instruction),
// over 253 states = 23 temperatures from -0.1 to -70 C x 11 moments covering
// every one the mp=28 kernels ask for (0, 1, 1.775, 2.55, 3):
//     a_  old form: 118/253 exact, max 3.267395e-06 relative
//     a_  this form: 253/253 BIT-EXACT
//     b_  old form: 180/253 exact, max 4.411423e-07 relative
//     b_  this form: 253/253 BIT-EXACT
// Restoring only the correctly-rounded pow, without the association fix,
// changes nothing (126/253, still 3.267e-06): the association is the larger
// error and both had to go.
//
// AND AGAINST A TRUE WRF ORACLE, not a transcription: calc_effectRad is
// PUBLIC, so its snow branch (:5658-5695) can be called directly over a
// 24-temperature x 15-snow-mass grid.  360 rows, 299 of them strictly inside
// the [5.01, 999] um clamp:
//     old fits: 138/360 exact, max 5.870704e-06 relative
//     these fits + a correctly-rounded smo2**b_: 360/360 BIT-EXACT
// The committed gpuwm/data/thompson/oracle-aero/probe-effectrad.csv could not
// see this: all 14 of its rows carry t = 285 K and qs = 2e-4, so its effs_m
// column is ONE state repeated fourteen times.
//
// This is the same tie-break the "THE SHARED FITS" note above thompson_aa_add
// records for thompson_rslf/thompson_rsif: the authority is WRF, not ArWen's
// mp=8 port.  thompson.cu keeps its plain chain and stays byte-frozen; mp=28
// matches WRF.  Do not "restore consistency" with thompson.cu:81-107.
__device__ __forceinline__ float thompson_aa_field_loga(float tc0, float mom)
{
    // module_mp_thompson.F:2069-2074, term for term, left to right.
    float v = THOMPSON_AA_SA[0];
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_SA[1], tc0));
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_SA[2], mom));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[3], tc0), mom));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[4], tc0), tc0));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[5], mom), mom));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[6], tc0), tc0), mom));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[7], tc0), mom), mom));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[8], tc0), tc0), tc0));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[9], mom), mom), mom));
    return v;
}

__device__ __forceinline__ float thompson_field_a(float tc, float moment)
{
    // :2075  a_ = 10.0**loga_   (REAL(4)**REAL(4) -> glibc powf)
    return thompson_aa_powf(10.0f, thompson_aa_field_loga(tc, moment));
}

__device__ __forceinline__ float thompson_field_b(float tc, float moment)
{
    // module_mp_thompson.F:2076-2079, term for term, left to right.
    float v = THOMPSON_AA_SB[0];
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_SB[1], tc));
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_SB[2], moment));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[3], tc), moment));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[4], tc), tc));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[5], moment), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[6], tc), tc), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[7], tc), moment), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[8], tc), tc), tc));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[9], moment), moment), moment));
    return v;
}


// ---------------------------------------------------------------------------
// CCN activation, module_mp_thompson.F:5178-5253.
// ---------------------------------------------------------------------------
//
// tnccn_act is WP-01's float64 Fortran-order (7,9,7,5,4) device array.  The
// Fortran linear index of tnccn_act(i,j,k,l,m) with one-based subscripts is
//     (i-1) + 7*((j-1) + 9*((k-1) + 7*((l-1) + 5*(m-1))))
// Values are cast to float on load; every blend operation is float, matching
// WRF's REAL(KIND=R4SIZE) table (:393) and default-REAL arithmetic.
__device__ __forceinline__ float thompson_activ_ncloud(
    float Tt, float Ww, float NCCN, const double* __restrict__ tnccn_act)
{
    // Asymmetric epsilons: -1.0/+1.0 on the CCN axis, -1.0/+0.001 on the
    // updraft axis.  Reproduce literally; they are not a symmetric guard.
    float n_local = thompson_aa_mul(NCCN, 1.0e-6f);
    if (n_local >= THOMPSON_AA_TA_NA[THOMPSON_AA_NTB_ARC - 1]) {
        n_local = THOMPSON_AA_TA_NA[THOMPSON_AA_NTB_ARC - 1] - 1.0f;
    } else if (n_local <= THOMPSON_AA_TA_NA[0]) {
        n_local = THOMPSON_AA_TA_NA[0] + 1.0f;
    }
    // Fortran DO/GOTO bracket search, i.e. a LINEAR SCAN, not a bisection.
    // WRF's fallthrough leaves n == ntb_arc+1 and would index one past the
    // end; the clamps above make that unreachable for any finite input
    // (11 -> i=2, 9999 -> i=7), so the initializer is pinned at ntb_arc to
    // degrade in bounds rather than read out of bounds on a NaN.
    int i = THOMPSON_AA_NTB_ARC;
    for (int n = 2; n <= THOMPSON_AA_NTB_ARC; ++n) {
        if (n_local >= THOMPSON_AA_TA_NA[n - 2]
                && n_local < THOMPSON_AA_TA_NA[n - 1]) {
            i = n;
            break;
        }
    }
    const float x1 = thompson_aa_logf(THOMPSON_AA_TA_NA[i - 2]);
    const float x2 = thompson_aa_logf(THOMPSON_AA_TA_NA[i - 1]);

    float w_local = Ww;
    if (w_local >= THOMPSON_AA_TA_WW[THOMPSON_AA_NTB_ARW - 1]) {
        w_local = THOMPSON_AA_TA_WW[THOMPSON_AA_NTB_ARW - 1] - 1.0f;
    } else if (w_local <= THOMPSON_AA_TA_WW[0]) {
        w_local = THOMPSON_AA_TA_WW[0] + 0.001f;
    }
    int j = THOMPSON_AA_NTB_ARW;
    for (int n = 2; n <= THOMPSON_AA_NTB_ARW; ++n) {
        if (w_local >= THOMPSON_AA_TA_WW[n - 2]
                && w_local < THOMPSON_AA_TA_WW[n - 1]) {
            j = n;
            break;
        }
    }
    const float y1 = thompson_aa_logf(THOMPSON_AA_TA_WW[j - 2]);
    const float y2 = thompson_aa_logf(THOMPSON_AA_TA_WW[j - 1]);

    // NEAREST-NEIGHBOUR in temperature over a 10 K grid.  There is no
    // interpolation here, so activated number is a STEP function of T.
    const int k = max(1, min(
        thompson_aa_nint(
            thompson_aa_mul(thompson_aa_sub(Tt, 243.15f), 0.1f)) + 1,
        THOMPSON_AA_NTB_ART));

    const int l = THOMPSON_AA_ACTIV_L;
    const int m = THOMPSON_AA_ACTIV_M;
    const int tail = (k - 1)
        + THOMPSON_AA_NTB_ART * ((l - 1) + THOMPSON_AA_NTB_ARR * (m - 1));
    const int base = THOMPSON_AA_NTB_ARC * THOMPSON_AA_NTB_ARW * tail;
    const int row_lo = base + THOMPSON_AA_NTB_ARC * (j - 2);
    const int row_hi = base + THOMPSON_AA_NTB_ARC * (j - 1);

    const float A = (float)tnccn_act[row_lo + (i - 2)];
    const float B = (float)tnccn_act[row_lo + (i - 1)];
    const float C = (float)tnccn_act[row_hi + (i - 1)];
    const float D = (float)tnccn_act[row_hi + (i - 2)];

    const float nx = thompson_aa_logf(n_local);
    const float wy = thompson_aa_logf(w_local);
    const float t = thompson_aa_div(thompson_aa_sub(nx, x1),
                                    thompson_aa_sub(x2, x1));
    const float u = thompson_aa_div(thompson_aa_sub(wy, y1),
                                    thompson_aa_sub(y2, y1));
    const float t1 = thompson_aa_sub(1.0f, t);
    const float u1 = thompson_aa_sub(1.0f, u);
    // Fortran's left-to-right sum of four separately rounded products.
    float fraction = thompson_aa_mul(thompson_aa_mul(t1, u1), A);
    fraction = thompson_aa_add(
        fraction, thompson_aa_mul(thompson_aa_mul(t, u1), B));
    fraction = thompson_aa_add(
        fraction, thompson_aa_mul(thompson_aa_mul(t, u), C));
    fraction = thompson_aa_add(
        fraction, thompson_aa_mul(thompson_aa_mul(t1, u), D));
    return thompson_aa_mul(NCCN, fraction);
}

__device__ __forceinline__ float thompson_aa_activ_ncloud(
    float Tt, float Ww, float NCCN, const double* __restrict__ tnccn_act)
{
    return thompson_activ_ncloud(Tt, Ww, NCCN, tnccn_act);
}


// ---------------------------------------------------------------------------
// DeMott (2010) ice nucleation, module_mp_thompson.F:5448-5518.
// ---------------------------------------------------------------------------
//
// NEGATIVE FINDING, verified in v4.6.1: the Phillips (2008) branch at
// 5474-5505 is entirely commented out, together with the satw/sati/siw
// diagnostics that fed it.  What remains is unconditional and depends only
// on (tempc, rho, nifa); qv, qvs and qvsi are dead formal arguments and are
// deliberately NOT parameters here.
__device__ __forceinline__ float thompson_ice_demott(
    float tempc, float rho, float nifa_m3)
{
    // nifa_cc = MAX(0.5, nifa*RHO_NOT0*1.E-6/rho)
    const float nifa_cc = fmaxf(
        0.5f,
        thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(nifa_m3, THOMPSON_AA_RHO_NOT0), 1.0e-6f),
            rho));
    // xni = (5.94e-5*(-tempc)**3.33) * (nifa_cc**((-0.0264*tempc)+0.0033))
    const float exponent = thompson_aa_add(
        thompson_aa_mul(-0.0264f, tempc), 0.0033f);
    float xni = thompson_aa_mul(
        thompson_aa_mul(5.94e-5f, thompson_aa_powf(-tempc, 3.33f)),
        thompson_aa_powf(nifa_cc, exponent));
    // xni = xni*rho/RHO_NOT0 * 1000.
    xni = thompson_aa_mul(
        thompson_aa_div(thompson_aa_mul(xni, rho), THOMPSON_AA_RHO_NOT0),
        1000.0f);
    return fmaxf(0.0f, xni);
}

__device__ __forceinline__ float thompson_aa_ice_demott(
    float tempc, float rho, float nifa_m3)
{
    return thompson_ice_demott(tempc, rho, nifa_m3);
}


// ---------------------------------------------------------------------------
// Koop et al (2001) homogeneous haze freezing, module_mp_thompson.F:5521-5546.
// ---------------------------------------------------------------------------
//
// The caller owns WRF's gate at :2634-2637 (is_aerosol_aware .AND. homogIce
// .AND. ns+ni <= 999.e3 .AND. temp < 238 .AND. ssati >= 0.4).  This helper
// evaluates the rate unconditionally.
__device__ __forceinline__ float thompson_ice_koop(
    float temp, float qv, float qvs, float naero, float dt)
{
    const float satw = thompson_aa_div(qv, qvs);
    // mu_diff = 210368.0 + (131.438*temp) - (3.32373E6/temp)
    //           - (41729.1*alog(temp))
    float mu_diff = thompson_aa_add(
        210368.0f, thompson_aa_mul(131.438f, temp));
    mu_diff = thompson_aa_sub(mu_diff, thompson_aa_div(3.32373e6f, temp));
    mu_diff = thompson_aa_sub(
        mu_diff, thompson_aa_mul(41729.1f, thompson_aa_logf(temp)));
    const float a_w_i = thompson_aa_expf(
        thompson_aa_div(mu_diff,
                        thompson_aa_mul(THOMPSON_AA_R_UNI, temp)));
    const float d = thompson_aa_sub(satw, a_w_i);
    // log_J_rate = -906.7 + 8502*d - 26924*d*d + 29180*d*d*d, with Fortran's
    // left-to-right products (26924.0*d)*d and ((29180.0*d)*d)*d.
    float log_J_rate = thompson_aa_add(
        -906.7f, thompson_aa_mul(8502.0f, d));
    log_J_rate = thompson_aa_sub(
        log_J_rate, thompson_aa_mul(thompson_aa_mul(26924.0f, d), d));
    log_J_rate = thompson_aa_add(
        log_J_rate,
        thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(29180.0f, d), d), d));
    log_J_rate = fminf(20.0f, log_J_rate);
    const float J_rate = thompson_aa_powf(10.0f, log_J_rate);  // cm-3 s-1
    // `1. - exp(-x)` with x ~ 1e-14: prob_h is quantized to multiples of
    // 2^-24 here, so this subtraction is where WRF's own REAL(4) evaluation
    // becomes ulp-sensitive.  See thompson_aa_expf above.
    const float koop_arg = thompson_aa_mul(
        thompson_aa_mul(-J_rate, THOMPSON_AA_AR_VOLUME), dt);
    const float prob_h = fminf(
        thompson_aa_sub(1.0f, thompson_aa_expf(koop_arg)), 1.0f);
    float xni = 0.0f;
    if (prob_h > 0.0f) {
        xni = fminf(thompson_aa_mul(prob_h, naero), 1000.0e3f);
    }
    return fmaxf(0.0f, xni);
}

__device__ __forceinline__ float thompson_aa_ice_koop(
    float temp, float qv, float qvs, float naero, float dt)
{
    return thompson_ice_koop(temp, qv, qvs, naero, dt);
}


// ---------------------------------------------------------------------------
// Aerosol collection efficiency, module_mp_thompson.F:4965-5001.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float thompson_eff_aero(
    float D, float Da, float visc, float rhoa, float Temp, int species)
{
    const float boltzman = 1.3806503e-23f;
    const float meanPath = 0.0256e-6f;

    float vt = 1.0f;
    if (species == THOMPSON_AA_SPECIES_RAIN) {
        // -0.1021 + 4.932E3*D - 0.9551E6*D*D + 0.07934E9*D*D*D
        //         - 0.002362E12*D*D*D*D, all products left-associated.
        vt = thompson_aa_add(-0.1021f, thompson_aa_mul(4.932e3f, D));
        vt = thompson_aa_sub(
            vt, thompson_aa_mul(thompson_aa_mul(0.9551e6f, D), D));
        vt = thompson_aa_add(
            vt, thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(0.07934e9f, D), D), D));
        vt = thompson_aa_sub(
            vt, thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(thompson_aa_mul(0.002362e12f, D), D), D),
                D));
    } else if (species == THOMPSON_AA_SPECIES_SNOW) {
        vt = thompson_aa_mul(
            THOMPSON_AA_AV_S, thompson_aa_powf(D, THOMPSON_AA_BV_S));
    } else if (species == THOMPSON_AA_SPECIES_GRAUPEL) {
        vt = thompson_aa_mul(
            THOMPSON_AA_AV_G, thompson_aa_powf(D, THOMPSON_AA_BV_G));
    } else if (species == THOMPSON_AA_SPECIES_GRAUPEL_HAIL) {
        // av_g(5)*D**bv_g(5) with the Heymsfield slot-5 pair (literals as
        // THOMPSON_AA_HAIL_AV_G[5] / THOMPSON_AA_HAIL_BV_G below), through
        // WOOF's own float32 powf.
        vt = thompson_aa_mul(1.432042236e+02f,
                             gfk_pow(D, 6.409616470e-01f));
    }

    // Cc = 1. + 2.*meanPath/Da * (1.257 + 0.4*exp(-0.55*Da/meanPath))
    const float slip = thompson_aa_add(
        1.257f,
        thompson_aa_mul(
            0.4f,
            thompson_aa_expf(
                thompson_aa_div(thompson_aa_mul(-0.55f, Da), meanPath))));
    const float Cc = thompson_aa_add(
        1.0f,
        thompson_aa_mul(
            thompson_aa_div(thompson_aa_mul(2.0f, meanPath), Da), slip));
    // diff = boltzman*Temp*Cc/(3.*PI*visc*Da)
    const float diff = thompson_aa_div(
        thompson_aa_mul(thompson_aa_mul(boltzman, Temp), Cc),
        thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(3.0f, THOMPSON_AA_PI), visc),
            Da));

    // Re = 0.5*rhoa*D*vt/visc ;  Sc = visc/(rhoa*diff)
    const float Re = thompson_aa_div(
        thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(0.5f, rhoa), D), vt),
        visc);
    const float Sc = thompson_aa_div(visc, thompson_aa_mul(rhoa, diff));

    // St = Da*Da*vt*1000./(9.*visc*D)
    const float St = thompson_aa_div(
        thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(Da, Da), vt), 1000.0f),
        thompson_aa_mul(thompson_aa_mul(9.0f, visc), D));
    const float aval = thompson_aa_add(
        1.0f, thompson_aa_logf(thompson_aa_add(1.0f, Re)));
    // St2 = (1.2 + 1./12.*aval)/(1.+aval)
    const float St2 = thompson_aa_div(
        thompson_aa_add(1.2f, thompson_aa_mul(1.0f / 12.0f, aval)),
        thompson_aa_add(1.0f, aval));

    const float sqrt_re = sqrtf(Re);
    // 1. + 0.4*SQRT(Re)*Sc**0.3333 + 0.16*SQRT(Re)*SQRT(Sc)
    float brownian = thompson_aa_add(
        1.0f,
        thompson_aa_mul(thompson_aa_mul(0.4f, sqrt_re),
                        thompson_aa_powf(Sc, 0.3333f)));
    brownian = thompson_aa_add(
        brownian,
        thompson_aa_mul(thompson_aa_mul(0.16f, sqrt_re), sqrtf(Sc)));
    const float term1 = thompson_aa_mul(
        thompson_aa_div(4.0f, thompson_aa_mul(Re, Sc)), brownian);
    // 4.*Da/D * (0.02 + Da/D*(1.+2.*SQRT(Re)))
    const float ratio = thompson_aa_div(Da, D);
    const float term2 = thompson_aa_mul(
        thompson_aa_div(thompson_aa_mul(4.0f, Da), D),
        thompson_aa_add(
            0.02f,
            thompson_aa_mul(
                ratio,
                thompson_aa_add(1.0f, thompson_aa_mul(2.0f, sqrt_re)))));
    float Eff = thompson_aa_add(term1, term2);

    if (St > St2) {
        const float excess = thompson_aa_sub(St, St2);
        Eff = thompson_aa_add(
            Eff,
            thompson_aa_powf(
                thompson_aa_div(excess,
                                thompson_aa_add(excess, 0.666667f)),
                1.5f));
    }
    return fmaxf(1.0e-5f, fminf(Eff, 1.0f));
}

__device__ __forceinline__ float thompson_aa_eff_aero(
    float D, float Da, float visc, float rhoa, float Temp, int species)
{
    return thompson_eff_aero(D, Da, visc, rhoa, Temp, species);
}


// ---------------------------------------------------------------------------
// Explicit two-gamma snow number, module_mp_thompson.F:2081-2088.
// ---------------------------------------------------------------------------
//
// This is NOT smo0 (the zeroth power-law moment).  It is the analytic
// integral of WRF's bimodal snow distribution and is the ns(k) the Koop
// homogeneous-freezing gate at :2634 tests.  smob is the bm_s-th moment
// (rs*oams, since bm_s == 2 exactly) and smoc the (bm_s+1)-th.
//
// Every operand is REAL(4) in WRF -- ns, M0, Mrat, slam1 and slam2 are all
// declared at :1609-1610 -- so this is float32 throughout, and Fortran's
// left-to-right evaluation of the equal-precedence `*` and `/` chain makes
// the second term ((((Mrat*Kap1)*M0**mu_s)*csg(15))/slam2**cse(15)).
//
// CONTRACTION-PINNED and correctly-rounded, for the same reason every other
// helper here is -- every operation below is __fdiv_rn / __fmul_rn /
// __fadd_rn and both powers are thompson_aa_powf.  (If you are reading
// this because someone told you the body "uses plain *, /, + and CUDA's
// powf": it does not, and has not since wave 4.  Check the body, not the
// claim.)
//
// GATED AGAINST COMPILED WRF, NOT AGAINST A HOST TRANSCRIPTION.  The earlier
// version of this note quoted 391 states from a `gfortran -O2 -ffree-form`
// TRANSCRIPTION of :2029-2088.  Both halves of that are now superseded.  The
// reference is a PUBLIC probe subroutine whose body is a verbatim copy of
// :2028-2088, compiled into the real module_mp_thompson (so Kap0/Kap1,
// Lam0/Lam1, mu_s, csg(15) and cse(15) are thompson_init's own PRIVATE
// values, not retyped constants), and the sweep is 3721 states = 61
// temperatures from 273.05 K to 201.05 K x 61 log-spaced snow contents from
// 1e-12 to 1e-2 kg m^-3, with ns spanning 1.460e-01 to 2.797e+08 m^-3:
//
//     this form, fed WRF's own smoc:   3719/3721 BIT-EXACT
//         the two survivors are (268.25 K, rs = 6.8129234e-06) and
//         (265.84998 K, rs = 4.641592e-05), each exactly ONE float32 ulp,
//         max 6.923721e-08 relative.  Rebuilding the helper with plain CUDA
//         powf reproduces the SAME 3719/3721, so they are not a powf choice;
//         they are the double-rounding limit of `(float)pow(double,double)`
//         against glibc's singly-rounded powf.
//     the composite the callers run, a_ * smob**b_ then ns:
//         plain powf   smoc 3489/3721 (1.663435e-07)  ns 3526/3721 (5.512328e-07)
//         powf_cr      smoc 3717/3721 (1.063558e-07)  ns 3716/3721 (1.344195e-07)
//     thompson_field_a / thompson_field_b are BIT-EXACT on all 3721, so the
//     whole of that difference is the power.  cold.cu and warm.cu both spell
//     the composite with thompson_aa_powf for exactly this reason.
//
// THE HELPER WAS NOT THE ICE-KOOP PROBLEM.  Feeding it WRF's own smoc it was
// already good to 1.2e-07; feeding it the smoc the OLD thompson_field_a /
// thompson_field_b produced it was wrong by up to 1.490356e-05, because the
// composite a_*smo2**b_ inherited the 3.3e-06 fit error and ns is quartic in
// smob/smoc.  The defect was upstream, in the fits, not here.
//
// Gates: tests/test_thompson_aerosol_cold_gpu.py::
// test_two_gamma_snow_number_matches_the_wrf_fortran_integral (92 Fortran
// states, EQUALITY) and ::test_two_gamma_snow_number_survivors_are_exactly_
// one_ulp (the two states above, pinned so the limit cannot silently grow).
__device__ __forceinline__ float thompson_aa_snow_number(
    float smob, float smoc)
{
    const float M0 = thompson_aa_div(smob, smoc);                 // :2083
    const float Mrat = thompson_aa_mul(                           // :2084
        thompson_aa_mul(thompson_aa_mul(smob, M0), M0), M0);
    const float slam1 = thompson_aa_mul(M0, THOMPSON_AA_LAM0);    // :2085
    const float slam2 = thompson_aa_mul(M0, THOMPSON_AA_LAM1);    // :2086
    // :2087-2088
    const float first = thompson_aa_div(
        thompson_aa_mul(Mrat, THOMPSON_AA_KAP0), slam1);
    const float second = thompson_aa_div(
        thompson_aa_mul(
            thompson_aa_mul(
                thompson_aa_mul(Mrat, THOMPSON_AA_KAP1),
                thompson_aa_powf(M0, THOMPSON_AA_MU_S)),
            THOMPSON_AA_CSG15),
        thompson_aa_powf(slam2, THOMPSON_AA_CSE15));
    return thompson_aa_add(first, second);
}


// ---------------------------------------------------------------------------
// Cloud droplet distribution, module_mp_thompson.F:1826-1842.
// ---------------------------------------------------------------------------
//
// Reproduces WRF's entry diagnosis exactly, including its type mixing: the
// first lambda is a REAL power (powf) widened to DOUBLE, the size clamps are
// REAL divisions widened to DOUBLE, and the rediagnosis is a DOUBLE pow.
// rc must already satisfy rc > R1; the caller owns that branch.
//
// CORRECTLY ROUNDED AND CONTRACTION-PINNED, and both halves were MEASURED.
// :1833's `**obmr` is REAL(4)**REAL(4) (nc, am_r, ccg, ocg1, rc and obmr are
// all default REAL), which gfortran lowers to glibc's correctly-rounded
// powf; the float32 products and quotients around it are separately rounded
// on a baseline-x86-64 build with no FMA instruction, and :1841 then CUBES
// the resulting lambda.  MEASURED against a PUBLIC probe subroutine holding
// a verbatim copy of :1826-1842, compiled into the same module_mp_thompson
// object the column oracle uses (so ccg/ocg1/cce/ocg2 are thompson_init's
// own values), over 975 states = 13 qc from 1e-9 to 1e-2 kg kg^-1 x 15 nc
// from 1e6 to 2e9 kg^-1 x 5 densities from 0.2 to 1.3, which spans the whole
// nu_c ladder and both :1835/:1837 size clamps:
//     plain powf, unpinned  (what shipped before)  nc  791/975 exact,
//                                                  max 3.632162e-07 relative
//                                                  lamc 929/975 exact,
//                                                  max 1.108940e-07
//     thompson_aa_powf only                     nc  831/975 exact,
//                                                  max 1.893911e-07
//                                                  lamc 975/975 BIT-EXACT
//     powf_cr + every float32 chain pinned         nc  975/975 BIT-EXACT
//                                                  lamc 975/975 BIT-EXACT
// nu_c was already exact in all three.  Both changes are needed: the power
// alone fixes lamc but not the :1840 rediagnosis, whose REAL(4) prefactor
// ccg(1,nu_c)*ocg2(nu_c)*rc/am_r is rounded to float32 in Fortran before it
// meets the DOUBLE lamc**bm_r and was being widened here.
__device__ __forceinline__ float thompson_aa_cloud_dist(
    float rc, float nc_per_kg, float rho,
    int* nu_c_entry_out, double* lamc_entry_out)
{
    // :1830  nc(k) = MAX(2., MIN(nc1d(k)*rho(k), Nt_c_max))
    float nc = thompson_aa_clamp_nc(thompson_aa_mul(nc_per_kg, rho));
    const int nu_c = thompson_aa_nu_c(nc);
    // :1833  lamc = (nc(k)*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc(k))**obmr
    double lamc = (double)thompson_aa_powf(
        thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(nc, THOMPSON_AA_AM_R),
                                THOMPSON_AA_CCG2[nu_c]),
                THOMPSON_AA_OCG1[nu_c]),
            rc),
        THOMPSON_AA_OBMR);
    // :1834  xDc = (bm_r + nu_c + 1.) / lamc -- a REAL numerator (exact small
    // integers) over the DOUBLE lambda, the quotient narrowed back to REAL.
    const float xDc = (float)((double)thompson_aa_add(
        thompson_aa_add(THOMPSON_AA_BM_R, (float)nu_c), 1.0f) / lamc);
    // :1836 / :1838  a REAL(4) quotient ASSIGNED to the DOUBLE lamc.  D0c and
    // D0r*2. are not exact in binary, so the rounding is part of the answer.
    if (xDc < THOMPSON_AA_D0C) {
        lamc = (double)thompson_aa_div(THOMPSON_AA_CCE2[nu_c],
                                       THOMPSON_AA_D0C);
    } else if (xDc > THOMPSON_AA_D0R * 2.0f) {
        lamc = (double)thompson_aa_div(THOMPSON_AA_CCE2[nu_c],
                                       THOMPSON_AA_D0R * 2.0f);
    }
    // :1840-1841  MIN(DBLE(Nt_c_max), ccg(1,nu_c)*ocg2(nu_c)*rc(k)/am_r
    //                                 * lamc**bm_r)
    nc = (float)fmin(
        (double)THOMPSON_AA_NT_C_MAX,
        (double)thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_CCG1[nu_c],
                                THOMPSON_AA_OCG2[nu_c]),
                rc),
            THOMPSON_AA_AM_R)
            * thompson_aa_pow(lamc, (double)THOMPSON_AA_BM_R));
    // ENTRY-STAGE values only (:1832-1838).  Everything downstream of :1838
    // recomputes nu_c from the RETURNED nc via thompson_aa_nu_c_working.
    if (nu_c_entry_out != nullptr) *nu_c_entry_out = nu_c;
    if (lamc_entry_out != nullptr) *lamc_entry_out = lamc;
    return nc;
}


// ---------------------------------------------------------------------------
// SHARED NETWORK HELPERS.  One definition each; cold.cu, warm.cu and sed.cu
// carry NONE.  See the PUBLISHED SHARED SIGNATURES block at the top of this
// header for the contract, INCLUDING its correction of what enforces it: a
// surviving local copy of thompson_aa_entry_rain_distribution is NOT a
// redefinition error, because the shared form takes seven parameters and the
// deleted local ones took six, which C++ resolves as an overload in favour of
// the local copy while nvrtc says nothing.  The source scan in
// tests/test_thompson_aerosol_device_helpers.py is the enforcement mechanism.
// ---------------------------------------------------------------------------

// module_mp_thompson.F:4032-4046 == thompson.cu:2574-2598.  Terminal rain
// number size bound.  This is the mp=8 arithmetic verbatim; it carries no nc
// dependence, so mp=8 and mp=28 agree here by construction.
//
// THOMPSON_AA_AM_R is bit-identical to the `3.1415926536f*1000.0f/6.0f`
// product form the deleted cold.cu copy spelled out, and THOMPSON_AA_R1 /
// THOMPSON_AA_R2 are the same float32 values as its 1.0e-12f / 1.0e-6f
// literals.  test_mass_coefficient_constants_equal_the_product_forms proves
// the first of those ON DEVICE so the substitution cannot rot.
__device__ __forceinline__ void thompson_aa_bound_rain_number(
    float rain_mass, float density, float* rain_number_per_kg)
{
    if (rain_mass <= THOMPSON_AA_R1) {
        *rain_number_per_kg = 0.0f;
        return;
    }
    const float am_r = THOMPSON_AA_AM_R;
    float rain_number = fmaxf(THOMPSON_AA_R2, *rain_number_per_kg * density);
    float lambda = thompson_aa_powf(am_r * 6.0f * rain_number / rain_mass,
                        1.0f / 3.0f);
    float mvd = 3.672f / lambda;
    if (mvd > 2.5e-3f) {
        mvd = 2.5e-3f;
    } else if (mvd < 37.5e-6f) {
        mvd = 37.5e-6f;
    } else {
        return;
    }
    lambda = 3.672f / mvd;
    rain_number = __fdiv_rn((1.0f / 6.0f) * rain_mass, am_r)
        * lambda * lambda * lambda;
    *rain_number_per_kg = rain_number / density;
}

// module_mp_thompson.F:4029-4039 == thompson.cu:3719-3743.  Terminal ice
// number size bound.  Idempotent, which is what lets the sedimentation kernel
// keep mp=8's fused placement while WP-04's terminal state kernel applies the
// same bound again.  THOMPSON_AA_AM_I is bit-identical to the
// `3.1415926536f*890.0f/6.0f` product form the deleted copies spelled out.
__device__ __forceinline__ void thompson_aa_bound_ice_number(
    float ice_mass, float density, float* ice_number_per_kg)
{
    if (ice_mass <= THOMPSON_AA_R1) {
        *ice_number_per_kg = 0.0f;
        return;
    }
    const float am_i = THOMPSON_AA_AM_I;
    float ice_number = fmaxf(THOMPSON_AA_R2, *ice_number_per_kg * density);
    double lambda = (double)thompson_aa_powf(
        am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f);
    const float diameter = (float)(4.0 / lambda);
    if (diameter < 5.0e-6f) {
        lambda = 4.0 / 5.0e-6;
        ice_number = fminf(999.0e3f,
            __fdiv_rn((1.0f / 6.0f) * ice_mass, am_i)
            * (float)(lambda * lambda * lambda));
    } else if (diameter > 300.0e-6f) {
        lambda = 4.0 / 300.0e-6;
        ice_number = __fdiv_rn((1.0f / 6.0f) * ice_mass, am_i)
            * (float)(lambda * lambda * lambda);
    }
    *ice_number_per_kg = fminf(ice_number, 999.0e3f) / density;
}

// ---------------------------------------------------------------------------
// THE SOURCE-STAGE ICE AND RAIN BALANCES IN WRF'S TENDENCY FORM, for the
// accumulator path (qrten/nrten/qiten/niten non-null).  module_mp_thompson.F
// :3033-3055 and :3070-3091 do not touch the state: they rewrite niten /
// nrten (or qrten) so that X1d + Xten*DT lands inside the size bounds, and
// the terminal apply (:4023-4053) applies the result once.  The in-place
// bounds above (thompson_aa_bound_ice_number / _rain_number) are the same
// physics with the state as the carrier; they round differently, and they
// put the terminal size bound before the :3956 freeze where WRF puts it
// after.  Every product, quotient and sum below is REAL(4) and pinned as
// gfortran -O2 (no FMA) forms it; lami/lamr are DOUBLE (:1597-1599), the
// powers of REAL bases are correctly rounded (thompson_aa_powf) and the
// DOUBLE**REAL(3.0) is pow(x, 3.0), as the graupel balance in the cold
// network already writes it.
//
// rho/orho are the entry density and its REAL reciprocal (:1802, :2959),
// odts = 1./DT, and the *ten in/out are per kilogram per second.
__device__ __forceinline__ void thompson_aa_ice_balance_tendency(
    float qi1d, float ni1d, float qiten, float* __restrict__ niten,
    float rho, float orho, float odts, float dt)
{
    // :3036-3037.
    const float xri = fmaxf(THOMPSON_AA_R1, thompson_aa_mul(
        thompson_aa_add(qi1d, thompson_aa_mul(qiten, dt)), rho));
    float xni = fmaxf(THOMPSON_AA_R2, thompson_aa_mul(
        thompson_aa_add(ni1d, thompson_aa_mul(*niten, dt)), rho));
    if (xri > THOMPSON_AA_R1) {
        // :3039-3041.  cig(2) = 6 and oig1 = 1 exactly.
        double lami = (double)thompson_aa_powf(
            thompson_aa_div(thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_AM_I, THOMPSON_AA_CIG2),
                THOMPSON_AA_OIG1), xni), xri),
            THOMPSON_AA_OBMI);
        const double ilami = 1.0 / lami;
        // (bm_i + mu_i + 1.) is REAL 4.0; the product with DOUBLE ilami is
        // DOUBLE, rounded on assignment to REAL xDi.
        const float xdi = (float)(4.0 * ilami);
        // cig(1)*oig2*xri/am_i, REAL, with cig(1) = 1 and oig2 = 1/6.
        if (xdi < 5.0e-6f) {
            // :3043-3045.  cie(2)/5.E-6 is a REAL quotient widened to DOUBLE.
            lami = (double)thompson_aa_div(4.0f, 5.0e-6f);
            xni = (float)fmin(999.0e3, (double)thompson_aa_div(
                thompson_aa_mul(thompson_aa_mul(1.0f, 1.0f / 6.0f), xri),
                THOMPSON_AA_AM_I) * thompson_aa_pow(lami, 3.0));
            *niten = thompson_aa_mul(thompson_aa_mul(thompson_aa_sub(
                xni, thompson_aa_mul(ni1d, rho)), odts), orho);
        } else if (xdi > 300.0e-6f) {
            // :3047-3049.
            lami = (double)thompson_aa_div(4.0f, 300.0e-6f);
            xni = (float)((double)thompson_aa_div(
                thompson_aa_mul(thompson_aa_mul(1.0f, 1.0f / 6.0f), xri),
                THOMPSON_AA_AM_I) * thompson_aa_pow(lami, 3.0));
            *niten = thompson_aa_mul(thompson_aa_mul(thompson_aa_sub(
                xni, thompson_aa_mul(ni1d, rho)), odts), orho);
        }
    } else {
        // :3051.
        *niten = thompson_aa_mul(-ni1d, odts);
    }
    // :3053-3055, the 999 per litre ceiling on the result.
    const float xni_after = fmaxf(0.0f, thompson_aa_mul(
        thompson_aa_add(ni1d, thompson_aa_mul(*niten, dt)), rho));
    if (xni_after > 999.0e3f) {
        *niten = thompson_aa_mul(thompson_aa_mul(thompson_aa_sub(
            999.0e3f, thompson_aa_mul(ni1d, rho)), odts), orho);
    }
}

__device__ __forceinline__ void thompson_aa_rain_balance_tendency(
    float qr1d, float nr1d, float* __restrict__ qrten,
    float* __restrict__ nrten, float rho, float orho, float odts, float dt)
{
    // :3072-3073.
    const float xrr = fmaxf(THOMPSON_AA_R1, thompson_aa_mul(
        thompson_aa_add(qr1d, thompson_aa_mul(*qrten, dt)), rho));
    const float xnr = fmaxf(THOMPSON_AA_R2, thompson_aa_mul(
        thompson_aa_add(nr1d, thompson_aa_mul(*nrten, dt)), rho));
    if (xrr > THOMPSON_AA_R1) {
        // :3075-3076.  crg(3) = 6 and org2 = 1 exactly.
        double lamr = (double)thompson_aa_powf(
            thompson_aa_div(thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_AM_R, 6.0f), 1.0f), xnr), xrr),
            THOMPSON_AA_OBMR);
        // (3.0 + mu_r + 0.672) is REAL; over DOUBLE lamr, rounded to REAL.
        const float mvd_num = thompson_aa_add(
            thompson_aa_add(3.0f, 0.0f), 0.672f);
        float mvd_r = (float)((double)mvd_num / lamr);
        const float d0r_low = thompson_aa_mul(THOMPSON_AA_D0R, 0.75f);
        bool bounded = false;
        if (mvd_r > 2.5e-3f) {
            mvd_r = 2.5e-3f;
            bounded = true;
        } else if (mvd_r < d0r_low) {
            mvd_r = d0r_low;
            bounded = true;
        }
        if (bounded) {
            // :3079-3082 / :3084-3087.  lamr is a REAL quotient widened;
            // crg(2)*org3*xrr is REAL (crg(2) = 1, org3 = 1/6), times the
            // DOUBLE power, over am_r in DOUBLE, rounded to REAL xnr.
            lamr = (double)thompson_aa_div(mvd_num, mvd_r);
            const float xnr_bounded = (float)(
                (double)thompson_aa_mul(thompson_aa_mul(1.0f, 1.0f / 6.0f),
                                        xrr)
                * thompson_aa_pow(lamr, 3.0) / (double)THOMPSON_AA_AM_R);
            *nrten = thompson_aa_mul(thompson_aa_mul(thompson_aa_sub(
                xnr_bounded, thompson_aa_mul(nr1d, rho)), odts), orho);
        }
    } else {
        // :3089-3090.
        *qrten = thompson_aa_mul(-qr1d, odts);
        *nrten = thompson_aa_mul(-nr1d, odts);
    }
}

// module_mp_thompson.F:1878-1898 (the bounded LOCAL rain distribution WRF
// diagnoses at entry, deliberately distinct from the prognostic nr1d)
// followed by :2144-2150 (the y-intercept pass, which RE-DERIVES lamr from
// the bounded nr rather than reusing the clamped lambda).  thompson.cu:
// 2607-2645 skips that round trip; mp=28 reproduces it because prr_rcw,
// pnc_rcw, pna_rca and pnd_rcd all read N0_r and (lamr+fv_r)^-cre(9)
// directly.
//
// WRF's :2145 loop has NO L_qr guard, so :2146-2150 runs at every level.
// When rain_per_kg <= R1 the entry block leaves rr = R1 and nr = R2 (:1892-
// 1897) and the y-intercept pass still forms lamr, mvd_r and N0_r from those
// sentinels; this function therefore writes all four outputs unconditionally
// and returns L_qr as its value.
//
// crg(3) = WGAMMA(bm_r+mu_r+1) = WGAMMA(4) = 6 exactly, org2 = 1/WGAMMA(1) =
// 1 exactly, cre(2) = mu_r+1 = 1 exactly, so N0_r = nr*lamr.  crg(2)*org3 =
// WGAMMA(1)/WGAMMA(4) = 1/6.
//
// CONTRACTION-PINNED, and every power is thompson_aa_powf: build_aero.sh
// compiles the oracle with plain `gfortran -O2` on baseline x86-64, which has
// no FMA instruction and lowers REAL(4)** to glibc's correctly-rounded powf,
// while nvrtc defaults to --fmad=true and CUDA's powf carries ~2 ulp.
// MEASURED (warm.cu:81-92, RTX 5090, 12348 oracle rows): with plain powf,
// lamr / N0_r / prr_rcw / pna_rca / pnd_rcd all sit at ~2.7e-7 relative;
// pinned they are BIT-EXACT.  Do not "simplify" this back to powf.
__device__ __forceinline__ bool thompson_aa_entry_rain_distribution(
    float rain_per_kg, float rain_number_per_kg, float density,
    float* rain_number, double* rain_lambda, float* rain_mvd,
    double* rain_intercept_n0)
{
    const float am_r = THOMPSON_AA_AM_R;
    const float obmr = THOMPSON_AA_OBMR;
    // (3.0 + mu_r + 0.672) is a REAL sum, 3.6719999313354492, not the
    // DOUBLE literal 3.672.  Every mvd_r = (...)/lamr below divides that
    // REAL value, widened, by the DOUBLE lamr, and rounds to REAL; with the
    // DOUBLE literal mvd_r moved by one float32 unit at a share of levels,
    // and Ef_rr = 1 - EXP(2300*(mvd_r - 1950e-6)) with it (pnr_rcr).
    const float mvd_num = thompson_aa_add(thompson_aa_add(3.0f, 0.0f), 0.672f);
    // crg(2)*org3 = WGAMMA(mu_r+1)/WGAMMA(bm_r+mu_r+1) = 1/6 exactly.
    const float crg2_org3 = thompson_aa_mul(1.0f, 1.0f / 6.0f);
    float rr;
    float nr;
    bool active;
    if (rain_per_kg > THOMPSON_AA_R1) {
        active = true;
        rr = thompson_aa_mul(rain_per_kg, density);
        nr = fmaxf(THOMPSON_AA_R2,
                   thompson_aa_mul(rain_number_per_kg, density));
        if (nr <= THOMPSON_AA_R2) {
            // :1883-1885.  lamr is a REAL quotient widened; nr is
            // ((crg(2)*org3)*rr) REAL times lamr**bm_r, which gfortran
            // lowers to the DOUBLE pow(lamr, 3.0) (only exponents 2, 1 and
            // -1 fold), over am_r in DOUBLE, rounded to REAL.
            const double lam = (double)thompson_aa_div(mvd_num, 1.0e-3f);
            nr = (float)((double)thompson_aa_mul(crg2_org3, rr)
                         * thompson_aa_pow(lam, 3.0) / (double)am_r);
        }
        double lamr = (double)thompson_aa_powf(
            thompson_aa_div(
                thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(am_r, 6.0f), 1.0f), nr), rr),
            obmr);
        float mvd = (float)((double)mvd_num / lamr);
        const float d0r_low = thompson_aa_mul(THOMPSON_AA_D0R, 0.75f);
        if (mvd > 2.5e-3f) {
            // :1891-1893, the same forms.
            mvd = 2.5e-3f;
            lamr = (double)thompson_aa_div(mvd_num, mvd);
            nr = (float)((double)thompson_aa_mul(crg2_org3, rr)
                         * thompson_aa_pow(lamr, 3.0) / (double)am_r);
        } else if (mvd < d0r_low) {
            // :1895-1897.
            mvd = d0r_low;
            lamr = (double)thompson_aa_div(mvd_num, mvd);
            nr = (float)((double)thompson_aa_mul(crg2_org3, rr)
                         * thompson_aa_pow(lamr, 3.0) / (double)am_r);
        }
    } else {
        active = false;
        rr = THOMPSON_AA_R1;
        nr = THOMPSON_AA_R2;
    }

    // :2146-2150, executed for every level in WRF, rain or not.
    const double lamr = (double)thompson_aa_powf(
        thompson_aa_div(
            thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(am_r, 6.0f), 1.0f), nr), rr),
        obmr);
    *rain_number = nr;
    *rain_lambda = lamr;
    *rain_mvd = (float)((double)mvd_num / lamr);
    // N0_r = (nr*org2)*lamr**cre(2): org2 = 1, and cre(2) = 1 is a runtime
    // REAL, so pow(lamr, 1.0), which is lamr exactly.
    *rain_intercept_n0 = (double)thompson_aa_mul(nr, 1.0f) * lamr;
    return active;
}

// ilamr(k) = 1./lamr (:2148), and every rate that reads the rain slope
// after :2150 re-forms it as lamr = 1./ilamr(k) (:2198, :2213, :2322,
// :2723), which is not always the :2147 lamr: the double reciprocal moves
// it by one binary64 unit at a share of levels, and (lamr+fv_r)**(-cre(9))
// follows.  rain_lambda above is the :2147 value; this is what the
// collection rates must use.
__device__ __forceinline__ double thompson_aa_rain_lambda_reformed(
    double rain_lambda)
{
    const double ilamr = 1.0 / rain_lambda;
    return 1.0 / ilamr;
}


// ---------------------------------------------------------------------------
// calc_effectRad, module_mp_thompson.F:5594-5699.
// ---------------------------------------------------------------------------

// :5637-5643.  NOT the same selector as thompson_aa_nu_c.  The nc > 1.0e10
// branch is DEAD CODE in v4.6.1 because :5626 clamps nc to Nt_c_max first;
// it is transcribed anyway so a future caller cannot reintroduce it wrongly.
__device__ __forceinline__ int thompson_aa_inu_c_effrad(float nc_m3)
{
    if (nc_m3 < 100.0f) return 15;
    if (nc_m3 > 1.0e10f) return 2;
    return min(15, (int)floorf(1000.0e6f / nc_m3 + 0.5f) + 2);
}

// EVERY `**` IN calc_effectRad IS A REAL(4)**REAL(4), which gfortran lowers
// to glibc's correctly-rounded powf, where CUDA's powf carries several ulp.
// MEASURED against the REAL calc_effectRad -- it is PUBLIC, so it can be
// CALLED rather than transcribed -- over grids far wider than the committed
// probe-effectrad.csv, whose 14 rows all share one t, one qc and one qs and
// therefore pin only the nc ladder:
//
//     effs_m, 360 rows (24 T x 15 qs), 299 strictly inside the clamp
//         old fits + plain powf: 138/360 exact, max 5.870704e-06 relative
//         new fits + plain powf: 342/360 exact, max 1.947693e-07 relative
//         new fits + powf_cr:    360/360 BIT-EXACT      <- SHIPPED
//     effc_m, 378 rows (6 T x 7 qc x 9 nc), 198 strictly inside the clamp
//         plain powf: 373/378 exact, max 1.042121e-07
//         powf_cr:    378/378 BIT-EXACT                 <- SHIPPED
//     effi_m, 378 rows (6 T x 9 qi x 7 ni), 228 strictly inside the clamp
//         plain powf: 374/378 exact, max 6.293743e-08
//         powf_cr:    378/378 BIT-EXACT                 <- SHIPPED
//
// ALL THREE ARE WRF-EXACT, AND THE ICE BRANCH DELIBERATELY DIVERGES FROM
// mp=8 BY ONE ULP.  RECORD, DO NOT REVERT:
//
//   On the aero-reduces-to-classic after-column, thompson.cu and this header
//   disagree on effi at k = 23 and k = 24 by 6.567156e-08 relative -- one
//   float32 ulp.  MEASURED at those two levels, in metres:
//       WRF fixture effi_m   2.9473000e-05      2.9043753e-05
//       mp=28 powf_cr        2.9473000e-05      2.9043753e-05   <- matches
//       mp=8 thompson.cu     2.9473001e-05      2.9043755e-05   <- 1 ulp off
//   i.e. mp=8 is the one that disagrees with WRF there.  Reproducing that
//   disagreement was the only thing plain powf bought, and
//   tests/test_thompson_aerosol_state_gpu.py::
//   test_effective_radius_is_bitwise_against_every_oracle_after_column
//   demands the opposite.
//
//   WP-04's mp=8 identity test,
//   test_effective_radius_reduces_to_the_frozen_mp8_kernel_at_nt_c, has been
//   restated by its owner to EXPECT this: it still requires effc and effs
//   bitwise against thompson.cu, requires the effi divergence to be exactly
//   levels [22, 23] and at most one ulp, and requires mp=28 -- not mp=8 -- to
//   match the Fortran column.  Both tests are green.  If you put CUDA's powf
//   back you will break it from the other side.
//
//   MP28_PORT_SPEC.md's named hard identity gates are
//   thompson_aa_droplet_bin(100.0e6f) == 65 and
//   thompson_aa_in_bin(1000.0f) == 27.  Both are green and both are asserted
//   in tests/test_thompson_aerosol_device_helpers.py.
//
//   This is the same tie-break as "THE SHARED FITS" above thompson_aa_add and
//   as MP28_PORT_SPEC.md's gamma-deviation finding: mp=8 stays frozen and
//   slightly wrong, mp=28 is right.
// tests/test_thompson_aerosol_device_helpers.py::
// test_effect_rad_cloud_and_ice_match_the_real_calc_effect_rad and
// test_effect_rad_cloud_and_ice_diverge_from_mp8_by_one_ulp_on_purpose pin
// both halves of that so neither can rot.

// rc [kg m^-3] = MAX(R1, qc*rho); nc_m3 = MAX(2, MIN(nc*rho, Nt_c_max)).
// Caller must have already applied those, and must skip levels with
// rc <= R1 or nc <= R2.  Returns metres.  :5644-5646.
//
// ---------------------------------------------------------------------------
// CONTRACTION-PINNED AS WELL AS CORRECTLY ROUNDED, AND THE PINS MOVE NOTHING
// TODAY.  SAID PLAINLY BECAUSE IT IS THE POINT.
// ---------------------------------------------------------------------------
// Every operand at :5646, :5654, :5663 and :5694 is REAL(4), so each float32
// product and quotient is separately rounded in the gfortran -O2 baseline-
// x86-64 oracle.  Here each of those chains ends in `(double)<float expr>`,
// and nvrtc is free to evaluate such a chain in DOUBLE and skip the float32
// roundings -- WP-04 MEASURED exactly that happening in
// thompson_aa_state_finalize's :4019 prefactor, where it cost 2 to 3.5
// float32 ulps of droplet number and took 38 of 456 states off a Fortran-
// faithful host transcription.  A named `float` local does not stop it; only
// __fmul_rn / __fdiv_rn do.
//
// MEASURED HERE, against the REAL calc_effectRad (it is PUBLIC, so it is
// CALLED, not transcribed) over 960 states = 40 columns x 24 levels sweeping
// temperature 233-260 K, qc 1e-7..1e-4, nc 3e6..3e9 kg^-1, qi/ni and qs each
// over four decades:
//     unpinned float chains (what shipped)  effc 960/960, effi 958/960
//                                           (max 1.094592e-07), effs 960/960
//     every float32 chain pinned            effc 960/960, effi 958/960
//                                           (max 1.094592e-07), effs 960/960
// i.e. IDENTICAL.  The pins are not a correction; they turn "nvrtc happens
// not to widen this today" into a property of the source, which is what lets
// thompson_aerosol_state.cu delete its three private copies of these
// functions and call these instead.  The two surviving effi states are the
// double-rounding limit of thompson_aa_powf -- `(float)pow(double,double)`
// rounds twice where glibc's powf rounds once -- not a chain-association
// defect; plain CUDA powf does not fix them either.

__device__ __forceinline__ float thompson_aa_eff_rad_cloud(
    float rc, float nc_m3)
{
    const int inu_c = thompson_aa_inu_c_effrad(nc_m3);
    // :5646  lamc = (nc(k)*am_r*g_ratio(inu_c)/rc(k))**obmr.  CORRECTLY
    // ROUNDED: WRF's `**` is REAL(4)**REAL(4) -> glibc powf.  See the block
    // above for what this costs and why it is right.
    const double lamc = (double)thompson_aa_powf(
        thompson_aa_div(
            thompson_aa_mul(thompson_aa_mul(nc_m3, THOMPSON_AA_AM_R),
                            THOMPSON_AA_G_RATIO[inu_c]),
            rc),
        THOMPSON_AA_OBMR);
    // :5647  SNGL(0.5D0 * DBLE(3.+inu_c)/lamc).  `3.+inu_c` is a REAL plus an
    // INTEGER, i.e. a REAL add of exact small integers, widened only after.
    const float diagnosed = (float)(
        0.5 * (double)thompson_aa_add(3.0f, (float)inu_c) / lamc);
    return fmaxf(2.51e-6f, fminf(diagnosed, 50.0e-6f));
}

// ri [kg m^-3] = MAX(R1, qi*rho); ni [m^-3] = MAX(R2, ni*rho).  :5654-5655.
__device__ __forceinline__ float thompson_aa_eff_rad_ice(float ri, float ni)
{
    // :5654  lami = (am_i*cig(2)*oig1*ni(k)/ri(k))**obmi.  CORRECTLY ROUNDED.
    // THIS is the one that costs the mp=8 effective-radius identity test, at
    // aero-reduces-to-classic's top levels, by one float32 ulp -- because
    // mp=8 is 1 ulp off WRF there and mp=28 is not.  See the block above.
    const double lami = (double)thompson_aa_powf(
        thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(THOMPSON_AA_AM_I, THOMPSON_AA_CIG2),
                    THOMPSON_AA_OIG1),
                ni),
            ri),
        THOMPSON_AA_OBMI);
    // :5655  SNGL(0.5D0 * DBLE(3.+mu_i)/lami); mu_i is REAL (:105).
    const float diagnosed = (float)(
        0.5 * (double)thompson_aa_add(3.0f, THOMPSON_AA_MU_I) / lami);
    return fmaxf(2.51e-6f, fminf(diagnosed, 125.0e-6f));
}

// rs [kg m^-3] = MAX(R1, qs*rho).  bm_s is exactly 2, so WRF's reference
// second moment IS smob and the bm_s.ne.2 branch (:5665-5679) is dead.
// :5661-5695.
__device__ __forceinline__ float thompson_aa_eff_rad_snow(float rs, float t_k)
{
    const float tc0 = fminf(-0.1f, thompson_aa_sub(t_k, 273.15f));   // :5662
    const float smob = thompson_aa_mul(rs, THOMPSON_AA_OAMS);        // :5663
    const float smo2 = smob;                                         // :5668
    const float moment = 3.0f;                // cse(1) = bm_s + 1
    // :5694  smoc = a_ * smo2**b_
    const float smoc = thompson_aa_mul(
        thompson_field_a(tc0, moment),
        thompson_aa_powf(smo2, thompson_field_b(tc0, moment)));
    // :5695  0.5*(smoc/smob)
    const float diagnosed = thompson_aa_mul(
        0.5f, thompson_aa_div(smoc, smob));
    return fmaxf(5.01e-6f, fminf(diagnosed, 999.0e-6f));
}


// ===========================================================================
// THE OPERATIONAL WRF 3.9 FORK'S VARIANTS (RunConfig.thompson_version =
// "wrf_39_noaa").
// ===========================================================================
//
// Source: NOAA-EMC/HRRR tag v4.1.21, sorc/hrrr_wrfarw.fd/WRFV3.9/phys/
// module_mp_thompson.F (SHA-256 4d600111...).  Bare `fork :NNNN` is a line of
// that file.  The .cu units select these under THOMPSON_AA_WRF39, an integer
// define kernels/__init__.py inserts AFTER this header, so the header itself
// carries no #if: every fork form here is a separate function or constant the
// fork arms call, and under wrf_461 none of them is referenced.

// Ice number ceiling, fork :1816, :2879, :2890, :3733 (499.D3 where v4.6.1
// has 999.D3) and the size the fork gives ice that arrives with no number,
// fork :1815 (25 microns where v4.6.1 has 5).
#define THOMPSON_AA_WRF39_NI_MAX        499.0e3f
#define THOMPSON_AA_WRF39_NI_SEED_D     25.0e-6
// D0s = 200.E-6 (fork :205) and D0g = 250.E-6 (fork :206).
#define THOMPSON_AA_WRF39_D0S           200.0e-6f
#define THOMPSON_AA_WRF39_D0G           250.0e-6f
// rho_g = 500 (fork :72), so am_g = PI*rho_g/6 (fork :124), rounded once as
// the Fortran PARAMETER is.
#define THOMPSON_AA_WRF39_RHO_G         500.0f
// gonv_min / gonv_max, fork :115-116.
#define THOMPSON_AA_WRF39_GONV_MIN      1.0e4
#define THOMPSON_AA_WRF39_GONV_MAX      3.0e6
// r_s(1) = r_g(1) = 1.e-5 (fork :271-284), 28 entries each (fork :219-221),
// and the graupel intercept axis N0g_exp(1) = 1.e4, 28 entries (fork :288).
#define THOMPSON_AA_WRF39_R_SG_FIRST    1.0e-5f
#define THOMPSON_AA_WRF39_NTB_SG        28
#define THOMPSON_AA_WRF39_SG_EXP0       (-5)
#define THOMPSON_AA_WRF39_NG1_EXP0      4

__device__ __forceinline__ float thompson_aa_wrf39_am_g()
{
    return __fdiv_rn(THOMPSON_AA_PI * THOMPSON_AA_WRF39_RHO_G, 6.0f);
}

// thompson_aa_bound_ice_number with the fork's 499.D3 ceiling (fork
// :2873-2891, the source-stage ice mass/number balance).  Same body
// otherwise: the 5 and 300 micron size clamps are the fork's too.
__device__ __forceinline__ void thompson_aa_wrf39_bound_ice_number(
    float ice_mass, float density, float* ice_number_per_kg)
{
    if (ice_mass <= THOMPSON_AA_R1) {
        *ice_number_per_kg = 0.0f;
        return;
    }
    const float am_i = THOMPSON_AA_AM_I;
    float ice_number = fmaxf(THOMPSON_AA_R2, *ice_number_per_kg * density);
    double lambda = (double)thompson_aa_powf(
        am_i * 6.0f * ice_number / ice_mass, 0.33333334326744080f);
    const float diameter = (float)(4.0 / lambda);
    if (diameter < 5.0e-6f) {
        lambda = 4.0 / 5.0e-6;
        ice_number = fminf(THOMPSON_AA_WRF39_NI_MAX,
            __fdiv_rn(0.16666667163372040f * ice_mass, am_i)
            * (float)(lambda * lambda * lambda));
    } else if (diameter > 300.0e-6f) {
        lambda = 4.0 / 300.0e-6;
        ice_number = __fdiv_rn(0.16666667163372040f * ice_mass, am_i)
            * (float)(lambda * lambda * lambda);
    }
    *ice_number_per_kg = fminf(ice_number, THOMPSON_AA_WRF39_NI_MAX)
        / density;
}

// iceDeMott without v4.6.1's 0.5 per cc floor on the ice-friendly aerosol
// (fork :5129, `nifa_cc = nifa*RHO_NOT0*1.E-6/rho`; v4.6.1 :5505 wraps it in
// MAX(0.5, ...)).  Otherwise thompson_ice_demott verbatim.
__device__ __forceinline__ float thompson_aa_wrf39_ice_demott(
    float tempc, float rho, float nifa_m3)
{
    const float nifa_cc = thompson_aa_div(
        thompson_aa_mul(
            thompson_aa_mul(nifa_m3, THOMPSON_AA_RHO_NOT0), 1.0e-6f),
        rho);
    const float exponent = thompson_aa_add(
        thompson_aa_mul(-0.0264f, tempc), 0.0033f);
    float xni = thompson_aa_mul(
        thompson_aa_mul(5.94e-5f, thompson_aa_powf(-tempc, 3.33f)),
        thompson_aa_powf(nifa_cc, exponent));
    xni = thompson_aa_mul(
        thompson_aa_div(thompson_aa_mul(xni, rho), THOMPSON_AA_RHO_NOT0),
        1000.0f);
    return fmaxf(0.0f, xni);
}

// The fork's graupel slope from its column intercept (fork :2051-2054,
// :3130-3133, :5506-5509):
//   lam_exp = (N0_exp*am_g*cgg(1)/rg)**oge1
//   lamg    = lam_exp * (cgg(3)*ogg2*ogg1)**obmg
//   N0_g    = N0_exp/(cgg(2)*lam_exp) * lamg**cge(2)
// N0_exp is DOUBLE PRECISION (fork :1597) but always a REAL value (10.**zans1
// is REAL, gonv_min/max are REAL), so it is carried in float32 exactly.
// cgg(1) = cgg(3) = WGAMMA(4) = 6, ogg1 = 1/6, ogg2 = cgg(2) = cge(2) = 1 and
// oge1 = 0.25; the REAL(4) product cgg(3)*ogg2*ogg1 rounds to exactly 1, so
// lamg is lam_exp.
__device__ __forceinline__ void thompson_aa_wrf39_graupel_slope(
    float n0_exp, float rg, double* lamg, double* ilamg, double* n0_g)
{
    const float am_g = thompson_aa_wrf39_am_g();
    const double lam_exp = thompson_aa_pow(
        (double)n0_exp * (double)am_g * 6.0 / (double)rg, 0.25);
    *lamg = lam_exp;
    *ilamg = 1.0 / lam_exp;
    *n0_g = (double)n0_exp / lam_exp * lam_exp;
}

// The fork's collision-table graupel intercept, rebuilt from ilamg (fork
// :2264-2267): lam_exp = lamg*(cgg(3)*ogg2*ogg1)**bm_g = lamg, and
// N0_exp = ogg1*rg/am_g * lam_exp**cge(1) with ogg1*rg/am_g REAL(4).
__device__ __forceinline__ double thompson_aa_wrf39_table_intercept(
    float rg, double ilamg)
{
    const double lamg = 1.0 / ilamg;
    const float prefix = __fdiv_rn(
        thompson_aa_mul(0.16666667163372040f, rg), thompson_aa_wrf39_am_g());
    return (double)prefix * thompson_aa_pow(lamg, 4.0);
}

// calc_effectRad's fork floors: ice 5.01 microns (fork :5275, v4.6.1 2.51),
// snow 10 microns (fork :5315, v4.6.1 5.01).  Bodies otherwise those above.
__device__ __forceinline__ float thompson_aa_wrf39_eff_rad_ice(
    float ri, float ni)
{
    const float v461 = thompson_aa_eff_rad_ice(ri, ni);
    // thompson_aa_eff_rad_ice returns MAX(2.51e-6, MIN(d, 125e-6)); the fork
    // form is MAX(5.01e-6, MIN(d, 125e-6)), which is MAX(5.01e-6, that).
    return fmaxf(5.01e-6f, v461);
}

__device__ __forceinline__ float thompson_aa_wrf39_eff_rad_snow(
    float rs, float t_k)
{
    return fmaxf(10.0e-6f, thompson_aa_eff_rad_snow(rs, t_k));
}


// ===========================================================================
// HAIL-AWARE THOMPSON (mp_physics = 38): the variable-density graupel
// ===========================================================================
//
// WRF v4.6.1 runs ONE module for mp=8, 28 and 38.  What makes it hail aware
// is a single fact, the presence of the optional graupel number at
// thompson_init (:459-466), and that fact changes exactly these things:
//
//   * the graupel number ng (QNGRAUPEL) and graupel volume qb (QVGRAUPEL,
//     m3 per kg) are prognostic state, read at :1260-1264 and written back at
//     :1342-1347, instead of being diagnosed from qg at every call;
//   * the bulk graupel density rho_g = qg/qb selects one of NINE density bins
//     idx_bg in 1..9 (:1920-1923, :3272-3274, :4065-4067) where mp=8/28 sit
//     on idx_bg1 = 5 always (:1950, :2848, :3284-3286, :4068);
//   * av_g/bv_g keep the Heymsfield Best-Reynolds values in all nine slots;
//     mp=8/28 overwrite slot 5 with av_g_old = 442, bv_g_old = 0.89
//     (:463-464).  So every graupel fall-speed moment changes, and since
//     bv_g is the SAME in all nine slots (0.640961647, :155-158), cge(n,m)
//     and cgg(n,m) are one column repeated nine times;
//   * the rain-graupel collection tables carry nine density slabs,
//     qr_acr_qg_mp38V1.dat (:4117-4125, :4192-4273);
//   * graupel falls with a density-dependent speed from Heymsfield's
//     Best-number relation (:3750-3753) instead of av_g_old/bv_g_old.
//
// The literals below are WRF's own REAL(4) values, read back from the
// compiled module after a hail-aware thompson_init (an instrumented copy of
// module_mp_thompson.F that writes rho_g/am_g/av_g/bv_g/cge/cgg/oamg/ocmg/
// oge1/ogg1/ogg2/ogg3/obmg/Sc3/t1_qg_sd/t1_qg_me in Z8.8 and stops; recorded
// in tools/thompson_wrf461_oracle/README-HAIL.md).  Every decimal literal
// round-trips to exactly that bit pattern.  Index 0 is an unused pad so the
// device code reads idx_bg with WRF's one-based value.
#define THOMPSON_AA_HAIL_NRHG       9
#define THOMPSON_AA_HAIL_IDX_BG1    5           // :79
#define THOMPSON_AA_HAIL_RHO_I      890.0f      // :72
#define THOMPSON_AA_HAIL_BV_G       6.409616470e-01f   // :155-158, every slot
#define THOMPSON_AA_HAIL_A_COEFF    0.47244157f        // :159
#define THOMPSON_AA_HAIL_B_COEFF    0.54698726f        // :160
#define THOMPSON_AA_HAIL_OGG1       1.666666716e-01f   // 1./cgg(1,1)
#define THOMPSON_AA_HAIL_OGG2       1.000000000e+00f   // 1./cgg(2,1)
#define THOMPSON_AA_HAIL_OGG3       1.666666716e-01f   // 1./cgg(3,1)
#define THOMPSON_AA_HAIL_OBMG       3.333333433e-01f   // 1./bm_g
#define THOMPSON_AA_HAIL_SC3        8.581680655e-01f   // Sc**(1./3.), :663
#define THOMPSON_AA_HAIL_T1_QG_SD   8.600000143e-01f   // :812
#define THOMPSON_AA_HAIL_T1_QG_ME   1.617826092e-05f   // :816
#define THOMPSON_AA_HAIL_OLFUS      (1.0f / 334000.0f) // :219, lsub-lvap0
#define THOMPSON_AA_HAIL_C_CUBE     0.5f               // :167
// (3.0 + mu_g + 0.672) as WRF forms it in REAL(4) (:1926).
#define THOMPSON_AA_HAIL_MVD_NUM    ((3.0f + 0.0f) + 0.672f)

__constant__ float THOMPSON_AA_HAIL_RHO_G[10] = {
    0.0f, 50.0f, 100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f,
    800.0f};
// am_g(m) = PI*rho_g(m)/6., :132-135.
__constant__ float THOMPSON_AA_HAIL_AM_G[10] = {
    0.0f, 2.617993927e+01f, 5.235987854e+01f, 1.047197571e+02f,
    1.570796356e+02f, 2.094395142e+02f, 2.617994080e+02f, 3.141592712e+02f,
    3.665191650e+02f, 4.188790283e+02f};
// av_g(m), Heymsfield Best-Reynolds, :151-154 (slot 5 NOT overwritten).
__constant__ float THOMPSON_AA_HAIL_AV_G[10] = {
    0.0f, 4.591738129e+01f, 6.708673859e+01f, 9.801584625e+01f,
    1.223533783e+02f, 1.432042236e+02f, 1.617947235e+02f, 1.787621155e+02f,
    1.944887848e+02f, 2.092258759e+02f};
// cge(n,m) and cgg(n,m) = WGAMMA(cge(n,m)), n = 1..12, :751-769.  One
// column: bv_g is identical in all nine slots under hail awareness.
__constant__ float THOMPSON_AA_HAIL_CGE[13] = {
    0.0f, 4.000000000e+00f, 1.000000000e+00f, 4.000000000e+00f,
    7.000000000e+00f, 7.640961647e+00f, 4.640961647e+00f, 3.140961647e+00f,
    1.640961647e+00f, 3.640961647e+00f, 2.000000000e+00f, 2.820480824e+00f,
    2.500000000e+00f};
__constant__ float THOMPSON_AA_HAIL_CGG[13] = {
    0.0f, 6.000000000e+00f, 1.000000000e+00f, 6.000000000e+00f,
    7.200000610e+02f, 2.465598145e+03f, 1.418176556e+01f, 2.286627531e+00f,
    8.987805247e-01f, 3.895060301e+00f, 1.000000000e+00f, 1.705754399e+00f,
    1.329340339e+00f};

// WRF's REAL(4) `x**y` with a REAL exponent is the C library powf, which
// WOOF's own float32 libm (glibc_flt32.cuh, prepended to the aerosol units)
// reproduces bit for bit.  The hail-aware statements call it directly; the
// inherited mp=28 statements keep their thompson_aa_powf_cr, whose measured
// residual is recorded where it is used.
__device__ __forceinline__ float thompson_aa_hail_powf(float x, float y)
{
    return gfk_pow(x, y);
}

// NINT(q/rb*0.01)+1 clamped to 1..NRHG: the bulk-density bin, :1923, :3274,
// :4067, :5791.  `ratio_scaled` is WRF's REAL (q/rb)*0.01 (or
// ((rg/rho)/rb)*0.01 at :3274), formed by the caller in WRF's own order.
// NOTE for readers: rho_g(1) = 50 is reachable only below the 50 kg m-3
// clamp, i.e. never -- NINT maps 50..149 to slot 2 (100 kg m-3).  That is
// WRF's binning, kept: it is a property of the scheme's design, not an
// arithmetic defect, and the nine-slab tables are built for exactly it.
__device__ __forceinline__ int thompson_aa_hail_density_bin(
    float ratio_scaled)
{
    return max(1, min(thompson_aa_nint(ratio_scaled) + 1,
                      THOMPSON_AA_HAIL_NRHG));
}

// ng = cgg(2,1)*ogg3*rg*lamg**bm_g / am_g(idx_bg), :1927/:1934/:1938/:3148.
// REAL(4) prefix, DOUBLE lamg**bm_g and division, one rounding to REAL.
__device__ __forceinline__ float thompson_aa_hail_number_from_lambda(
    float rg, double lamg, int idx_bg)
{
    return (float)(
        (double)thompson_aa_mul(
            thompson_aa_mul(THOMPSON_AA_HAIL_CGG[2], THOMPSON_AA_HAIL_OGG3),
            rg)
        * thompson_aa_pow(lamg, 3.0) / (double)THOMPSON_AA_HAIL_AM_G[idx_bg]);
}

// lamg = (am_g(idx_bg)*cgg(3,1)*ogg2*ng/rg)**obmg, REAL(4) throughout and
// widened into the DOUBLE lamg (:1929, :2136, :3126, :3374, :4069).
__device__ __forceinline__ double thompson_aa_hail_lambda(
    float ng, float rg, int idx_bg)
{
    return (double)thompson_aa_hail_powf(
        thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(THOMPSON_AA_HAIL_AM_G[idx_bg],
                                    THOMPSON_AA_HAIL_CGG[3]),
                    THOMPSON_AA_HAIL_OGG2),
                ng),
            rg),
        THOMPSON_AA_HAIL_OBMG);
}

// The hail-aware graupel distribution of one level at call entry.
struct ThompsonAaHailGraupel {
    bool l_qg;       // qg1d > R1, :1915
    int idx_bg;      // :1923 (idx_bg1 where L_qg fails, :1944)
    float rg;        // qg1d*rho, or R1
    float ng;        // per m3 after the :1924-1939 seed and size clamps
    float rb;        // clamped volume (m3 per kg), :1920-1921, :1947
    float qb_entry;  // qb1d as :1922/:1943 rewrite it
    float mvd_g;
    double lamg;     // :2136, every level
    double ilamg;    // :2137
    double n0_g;     // :2138
};

// module_mp_thompson.F:1915-1949 (hail-aware arm) then the unguarded
// kte->kts slope loop :2135-2139.  `rho` is the entry density of :1802.
__device__ __forceinline__ void thompson_aa_hail_graupel_entry(
    float qg1d, float ng1d, float qb1d, float rho,
    ThompsonAaHailGraupel* g)
{
    float ng = THOMPSON_AA_R2;
    int idx = THOMPSON_AA_HAIL_IDX_BG1;
    g->mvd_g = 0.0f;
    if (qg1d > THOMPSON_AA_R1) {
        g->l_qg = true;
        g->rg = thompson_aa_mul(qg1d, rho);
        ng = fmaxf(THOMPSON_AA_R2, thompson_aa_mul(ng1d, rho));
        float rb = fmaxf(thompson_aa_div(qg1d, THOMPSON_AA_HAIL_RHO_G[9]),
                         qb1d);
        rb = fminf(thompson_aa_div(qg1d, THOMPSON_AA_HAIL_RHO_G[1]), rb);
        g->rb = rb;
        g->qb_entry = rb;
        idx = thompson_aa_hail_density_bin(
            thompson_aa_mul(thompson_aa_div(qg1d, rb), 0.01f));
        double lamg;
        if (ng <= THOMPSON_AA_R2) {
            g->mvd_g = 1.5e-3f;
            lamg = (double)thompson_aa_div(THOMPSON_AA_HAIL_MVD_NUM,
                                           g->mvd_g);
            ng = thompson_aa_hail_number_from_lambda(g->rg, lamg, idx);
        }
        lamg = thompson_aa_hail_lambda(ng, g->rg, idx);
        g->mvd_g = (float)((double)THOMPSON_AA_HAIL_MVD_NUM / lamg);
        if (g->mvd_g > 25.4e-3f) {
            g->mvd_g = 25.4e-3f;
            lamg = (double)thompson_aa_div(THOMPSON_AA_HAIL_MVD_NUM,
                                           g->mvd_g);
            ng = thompson_aa_hail_number_from_lambda(g->rg, lamg, idx);
        } else if (g->mvd_g < THOMPSON_AA_D0R) {
            g->mvd_g = THOMPSON_AA_D0R;
            lamg = (double)thompson_aa_div(THOMPSON_AA_HAIL_MVD_NUM,
                                           g->mvd_g);
            ng = thompson_aa_hail_number_from_lambda(g->rg, lamg, idx);
        }
    } else {
        g->l_qg = false;
        g->rg = THOMPSON_AA_R1;
        g->rb = thompson_aa_div(thompson_aa_div(THOMPSON_AA_R1, rho),
                                THOMPSON_AA_HAIL_RHO_G[9]);
        g->qb_entry = 0.0f;
    }
    g->ng = ng;
    g->idx_bg = idx;
    // :2135-2139, on the post-clamp ng and the entry idx_bg, every level.
    g->lamg = thompson_aa_hail_lambda(ng, g->rg, idx);
    g->ilamg = 1.0 / g->lamg;
    // N0_g = ng*ogg2*lamg**cge(2,1); cge(2,1) = mu_g+1 = 1 exactly.
    g->n0_g = (double)thompson_aa_mul(ng, THOMPSON_AA_HAIL_OGG2) * g->lamg;
}

// Graupel collecting cloud water and scavenging aerosol, :2414-2439 and
// :2460-2471, in WRF's REAL/DOUBLE mixture, for one level.  Shared by the
// cold and warm networks so the two halves of the hail-aware column cannot
// see different graupel.
struct ThompsonAaHailCollect {
    double prg_gcw;
    double pnc_gcw;
    double pbg_gcw;
    double pna_gca;
    double pnd_gcd;
};

__device__ __forceinline__ void thompson_aa_hail_graupel_collect(
    const ThompsonAaHailGraupel* g,
    bool l_qc, float mvd_c, float rc, float nc, float nwfa, float nifa,
    float rho, float temp_k, float twet, float rhof, float visco,
    float odts, ThompsonAaHailCollect* out)
{
    out->prg_gcw = 0.0;
    out->pnc_gcw = 0.0;
    out->pbg_gcw = 0.0;
    out->pna_gca = 0.0;
    out->pnd_gcd = 0.0;
    const int idx = g->idx_bg;
    const float av = THOMPSON_AA_HAIL_AV_G[idx];
    const float tempc = thompson_aa_sub(temp_k, 273.15f);
    // t1_qg_qc = PI*.25*av_g(idx_bg)*cgg(9,idx_bg), :2432 / :2463.
    const float t1_qg_qc = thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_PI, 0.25f), av),
        THOMPSON_AA_HAIL_CGG[9]);
    const double fall9 = thompson_aa_pow(g->ilamg, (double)THOMPSON_AA_HAIL_CGE[9]);
    // xDg = (bm_g + mu_g + 1.)*ilamg: a REAL 4.0 times the DOUBLE slope.
    const float xdg = (float)(4.0 * g->ilamg);
    if (l_qc && mvd_c > THOMPSON_AA_D0C
            && g->rg >= 1.0e-6f && mvd_c > THOMPSON_AA_D0C) {
        // vtg = rhof*av_g*cgg(6)*ogg3 * ilamg**bv_g, :2417.
        const float vtg = (float)(
            (double)thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(rhof, av),
                                THOMPSON_AA_HAIL_CGG[6]),
                THOMPSON_AA_HAIL_OGG3)
            * thompson_aa_pow(g->ilamg, (double)THOMPSON_AA_HAIL_BV_G));
        // stoke_g = mvd_c*mvd_c*vtg*rho_w/(9.*visco*xDg), :2418.
        const float stoke_g = thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(mvd_c, mvd_c), vtg),
                1000.0f),
            thompson_aa_mul(thompson_aa_mul(9.0f, visco), xdg));
        // Cober and List (1993) rime density, :2420-2422.
        float const_ri = thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(-1.0f, thompson_aa_mul(mvd_c, 0.5e6f)),
                vtg),
            fminf(-0.1f, tempc));
        const_ri = fmaxf(0.1f, fminf(const_ri, 10.0f));
        const float rime_dens = thompson_aa_mul(
            thompson_aa_sub(
                thompson_aa_add(0.051f, thompson_aa_mul(0.114f, const_ri)),
                thompson_aa_mul(thompson_aa_mul(0.0055f, const_ri),
                                const_ri)),
            1000.0f);
        float ef_gw = 0.0f;
        if (stoke_g >= 0.4f && stoke_g <= 10.0f) {
            ef_gw = thompson_aa_mul(
                0.55f, (float)thompson_aa_log10((double)thompson_aa_mul(2.51f, stoke_g)));
        } else if (stoke_g > 10.0f) {
            ef_gw = 0.77f;
        }
        if (twet > 273.15f) ef_gw = thompson_aa_mul(ef_gw, 0.1f);
        out->prg_gcw = (double)thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(rhof, t1_qg_qc), ef_gw), rc)
            * g->n0_g * fall9;
        const double pnc = (double)thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(rhof, t1_qg_qc), ef_gw), nc)
            * g->n0_g * fall9;
        out->pnc_gcw = fmin((double)thompson_aa_mul(nc, odts), pnc);
        if (twet < 273.15f) out->pbg_gcw = out->prg_gcw / (double)rime_dens;
    }
    if (g->rg > 1.0e-6f) {
        const float ef_ccn = thompson_eff_aero(
            xdg, 0.04e-6f, visco, rho, temp_k,
            THOMPSON_AA_SPECIES_GRAUPEL_HAIL);
        out->pna_gca = fmin(
            (double)thompson_aa_mul(nwfa, odts),
            (double)thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(rhof, t1_qg_qc), ef_ccn),
                nwfa) * g->n0_g * fall9);
        const float ef_in = thompson_eff_aero(
            xdg, 0.8e-6f, visco, rho, temp_k,
            THOMPSON_AA_SPECIES_GRAUPEL_HAIL);
        out->pnd_gcd = fmin(
            (double)thompson_aa_mul(nifa, odts),
            (double)thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(rhof, t1_qg_qc), ef_in),
                nifa) * g->n0_g * fall9);
    }
}

// The rain-graupel table's graupel intercept bin, :2366-2377: N0_exp is
// re-formed from 1./ilamg(k) with the density bin's am_g.
// (cgg(3,1)*ogg2*ogg1)**bm_g is REAL 1.0, so lam_exp is lamg.
__device__ __forceinline__ int thompson_aa_hail_intercept_bin(
    const ThompsonAaHailGraupel* g)
{
    const double lamg = 1.0 / g->ilamg;
    const float scale = thompson_aa_hail_powf(
        thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_HAIL_CGG[3],
                                        THOMPSON_AA_HAIL_OGG2),
                        THOMPSON_AA_HAIL_OGG1),
        3.0f);
    const double lam_exp = lamg * (double)scale;
    const double n0_exp = (double)thompson_aa_div(
        thompson_aa_mul(THOMPSON_AA_HAIL_OGG1, g->rg),
        THOMPSON_AA_HAIL_AM_G[g->idx_bg])
        * thompson_aa_pow(lam_exp, (double)THOMPSON_AA_HAIL_CGE[1]);
    return thompson_aa_decade_index_double(n0_exp, 2, 37);
}

// Flat index into the nine-slab (37, 37, 9, 37, 37) Fortran-order
// rain-graupel tables (tcg_racg ... tnr_gacr), all indices zero-based.
__device__ __forceinline__ size_t thompson_aa_hail_rain_graupel_index(
    int graupel_intercept_bin, int graupel_mass_bin, int idx_bg,
    int rain_intercept_bin, int rain_mass_bin)
{
    return (size_t)graupel_intercept_bin
        + (size_t)37 * ((size_t)graupel_mass_bin
        + (size_t)37 * ((size_t)(idx_bg - 1)
        + (size_t)THOMPSON_AA_HAIL_NRHG * ((size_t)rain_intercept_bin
        + (size_t)37 * (size_t)rain_mass_bin)));
}

// The graupel mass/number balance of the hail-aware column, :3118-3161, in
// WRF's tendency form.  qg1d/ng1d/qb1d are the ENTRY values as :1922/:1943
// rewrote them, rho/orho the entry density, idx_bg the entry bin.  Updates
// qgten/ngten/qbten in place exactly as WRF does: the number tendency is
// re-formed where the median volume diameter leaves [D0r, 25.4 mm], and the
// whole call's graupel is removed where the post-source content is at or
// below R1.  (xrb, :3122-3123, feeds only a debug print and is not formed.)
__device__ __forceinline__ void thompson_aa_hail_graupel_balance(
    float qg1d, float ng1d, float qb1d, int idx_bg, float rho, float orho,
    float odts, float dt, float* qgten, float* ngten, float* qbten)
{
    const float xrg = fmaxf(THOMPSON_AA_R1, thompson_aa_mul(
        thompson_aa_add(qg1d, thompson_aa_mul(*qgten, dt)), rho));
    const float xng = fmaxf(THOMPSON_AA_R2, thompson_aa_mul(
        thompson_aa_add(ng1d, thompson_aa_mul(*ngten, dt)), rho));
    if (xrg > THOMPSON_AA_R1) {
        double lamg = thompson_aa_hail_lambda(xng, xrg, idx_bg);
        const float mvd_g = (float)((double)THOMPSON_AA_HAIL_MVD_NUM / lamg);
        bool bounded = false;
        float mvd = mvd_g;
        if (mvd_g > 25.4e-3f) {
            mvd = 25.4e-3f;
            bounded = true;
        } else if (mvd_g < THOMPSON_AA_D0R) {
            mvd = THOMPSON_AA_D0R;
            bounded = true;
        }
        if (bounded) {
            lamg = (double)thompson_aa_div(THOMPSON_AA_HAIL_MVD_NUM, mvd);
            const float xng_b = thompson_aa_hail_number_from_lambda(
                xrg, lamg, idx_bg);
            *ngten = thompson_aa_mul(thompson_aa_mul(
                thompson_aa_sub(xng_b, thompson_aa_mul(ng1d, rho)), odts),
                orho);
        }
    } else {
        *qgten = thompson_aa_mul(-qg1d, odts);
        *ngten = thompson_aa_mul(-ng1d, odts);
        *qbten = thompson_aa_mul(-qb1d, odts);
    }
}

// diffu(k) = 2.11E-5*(temp/273.15)**1.94 * (101325./pres), :1991 / :3199,
// REAL(4) left to right through WOOF's float32 powf.
__device__ __forceinline__ float thompson_aa_hail_diffu(float temp_k,
                                                        float pres)
{
    return thompson_aa_mul(
        thompson_aa_mul(2.11e-5f,
                        gfk_pow(thompson_aa_div(temp_k, 273.15f), 1.94f)),
        thompson_aa_div(101325.0f, pres));
}

// tcond(k) = (5.69 + 0.0168*tempc)*1.0E-5 * 418.936, :2000 / :3207.
__device__ __forceinline__ float thompson_aa_hail_tcond(float tempc)
{
    return thompson_aa_mul(
        thompson_aa_mul(thompson_aa_add(5.69f, thompson_aa_mul(0.0168f,
                                                               tempc)),
                        1.0e-5f),
        418.936f);
}

// The Srivastava & Coen deposition/sublimation prefactor t1_subl,
// :2384-2400, with WRF's grouping (rvs*otemp first, the three rvs_pp terms
// summed left to right, alphsc divided through in source order).  Returns
// t1_subl; *rvs_out receives rvs = rho*qvsi.
__device__ __forceinline__ float thompson_aa_hail_t1_subl(
    float temp_k, float rho, float qvsi, float diffu, float tcond,
    float ssati, float* rvs_out)
{
    const float lsub = 2.834e6f;
    const float orv = 1.0f / 461.5f;
    const float otemp = thompson_aa_div(1.0f, temp_k);
    const float rvs = thompson_aa_mul(rho, qvsi);
    const float tt = thompson_aa_sub(
        thompson_aa_mul(thompson_aa_mul(lsub, otemp), orv), 1.0f);
    const float rvs_p = thompson_aa_mul(thompson_aa_mul(rvs, otemp), tt);
    const float pp_a = thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(otemp, tt), otemp), tt);
    float pp_b = thompson_aa_mul(2.0f, lsub);
    pp_b = thompson_aa_mul(pp_b, otemp);
    pp_b = thompson_aa_mul(pp_b, otemp);
    pp_b = thompson_aa_mul(pp_b, otemp);
    pp_b = -thompson_aa_mul(pp_b, orv);
    const float pp_c = thompson_aa_mul(otemp, otemp);
    const float rvs_pp = thompson_aa_mul(
        rvs, thompson_aa_add(thompson_aa_add(pp_a, pp_b), pp_c));
    const float gamsc = thompson_aa_mul(
        thompson_aa_div(thompson_aa_mul(lsub, diffu), tcond), rvs_p);
    const float gr = thompson_aa_div(gamsc, thompson_aa_add(1.0f, gamsc));
    float alphsc = thompson_aa_mul(0.5f, gr);
    alphsc = thompson_aa_mul(alphsc, gr);
    alphsc = thompson_aa_mul(alphsc, rvs_pp);
    alphsc = thompson_aa_div(alphsc, rvs_p);
    alphsc = thompson_aa_mul(alphsc, rvs);
    alphsc = thompson_aa_div(alphsc, rvs_p);
    alphsc = fmaxf(1.0e-9f, alphsc);
    float xsat = ssati;
    if (fabsf(xsat) < 1.0e-9f) xsat = 0.0f;
    const float ax = thompson_aa_mul(alphsc, xsat);
    float t2 = thompson_aa_mul(2.0f, alphsc);
    t2 = thompson_aa_mul(t2, alphsc);
    t2 = thompson_aa_mul(t2, xsat);
    t2 = thompson_aa_mul(t2, xsat);
    float t3 = thompson_aa_mul(5.0f, alphsc);
    t3 = thompson_aa_mul(t3, alphsc);
    t3 = thompson_aa_mul(t3, alphsc);
    t3 = thompson_aa_mul(t3, xsat);
    t3 = thompson_aa_mul(t3, xsat);
    t3 = thompson_aa_mul(t3, xsat);
    float paren = thompson_aa_sub(1.0f, ax);
    paren = thompson_aa_add(paren, t2);
    paren = thompson_aa_sub(paren, t3);
    *rvs_out = rvs;
    return thompson_aa_div(
        thompson_aa_mul(thompson_aa_mul(4.0f, THOMPSON_AA_PI), paren),
        thompson_aa_add(1.0f, gamsc));

}

// THE PER-LEVEL SOURCE STAGE IN WRF'S OWN ARITHMETIC ORDER
// (module_mp_thompson.F:1798-2151 entry, :2157-2234 warm loop, :2239-2848
// frozen block, :2856-2952 conservation).
// ===========================================================================
//
// THE BREAKAGE THIS PREVENTS.  The cold network was assembled from the
// classic (mp=8) kernel's algebra: products regrouped (rvs*(otemp*x) for
// WRF's (rvs*otemp)*x, a*(x*x) for (a*x)*x), DOUBLE literals where WRF has
// REAL(4) constants (1.0e-12 for xm0i, 2.0e-12 for 2.*xm0i, 3.89 for
// cge(9,5), 720 for crg(8)), caps formed as DBLE(x)*DBLE(odts) where WRF
// rounds x*odts in REAL first, the graupel distribution re-diagnosed from
// mp=8's intercept fit instead of from WRF's ng, the conservation limiters
// in DOUBLE where WRF's sump/ratio are REAL, the paired rain/graupel
// transfer kept in DOUBLE where WRF rounds it through the REAL ratio, and
// 1./ilamr, NINT/INT and 10.**n spelled differently.  The host CPU
// comparison against WRF v4.6.1 (tools/thompson_aerosol_column_oracle/
// host_bitwise.py) found 52 of the 64 process rates differing, on up to
// every active cell: 6585 differing rate cells below 0 C.
//
// Everything below is written statement for statement from the Fortran:
// every REAL(4) operation is one pinned float32 operation (no contraction
// in any arithmetic mode), REAL op DOUBLE widens the REAL operand, a REAL
// variable assigned a DOUBLE expression rounds once, x**y with a REAL
// exponent calls WOOF's own powf/pow word (gfortran folds only the
// exponents 2.0, 1.0 and -1.0; WRF's exponents here are runtime REAL
// variables or other constants, so they reach the library), REAL**INTEGER
// is libgcc's __powisf2 (thompson_aa_powi_f), and NINT is lroundf/lround.
// WRF names are kept as field names so each line can be checked against
// the source.

__device__ __forceinline__ float aaf_fm(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float aaf_fa(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float aaf_fs(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float aaf_fd(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ double aaf_dm(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double aaf_da(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ double aaf_ds(double a, double b) { return __dsub_rn(a, b); }
__device__ __forceinline__ double aaf_dd(double a, double b) { return __ddiv_rn(a, b); }

// REAL(4)**INTEGER as gfortran lowers it with a non-constant exponent: a
// call to libgcc's __powisf2, square-and-multiply in float32 and one
// reciprocal for a negative exponent (libgcc2.c).
__device__ __forceinline__ float thompson_aa_powi_f(float x, int m)
{
    unsigned int n = m < 0 ? 0u - (unsigned int)m : (unsigned int)m;
    float y = (n % 2u) ? x : 1.0f;
    while (n >>= 1) {
        x = aaf_fm(x, x);
        if (n % 2u) y = aaf_fm(y, x);
    }
    return m < 0 ? aaf_fd(1.0f, y) : y;
}

// The decade-mantissa table index WRF writes out at :2263-2272 (rc), :2282
// (ri), :2296 (ni), :2311 (rr), :2340 (rs), :2355 (rg) and :2580 (xni):
//     nic = NINT(ALOG10(v))
//     do nn = nic-1, nic+1
//        n = nn
//        if ((v/10.**nn).ge.1.0 .and. (v/10.**nn).lt.10.0) goto 141
//     enddo
// 141 idx = INT(v/10.**n) + 10*(n-n2) - (n-n2);  MAX(1, MIN(idx, ntb))
// n2 = NINT(ALOG10(first table entry)).  When no candidate matches, n is
// the last one tried (nic+1).  Returned ONE-BASED.
__device__ __forceinline__ int thompson_aa_wrf_table_index(
    float v, int n2, int ntb)
{
    const int nic = (int)roundf(thompson_aa_log10f(v));
    int n = nic + 1;
    for (int nn = nic - 1; nn <= nic + 1; ++nn) {
        const float m = aaf_fd(v, thompson_aa_powi_f(10.0f, nn));
        if (m >= 1.0f && m < 10.0f) { n = nn; break; }
    }
    const int idx = (int)aaf_fd(v, thompson_aa_powi_f(10.0f, n))
        + 10 * (n - n2) - (n - n2);
    return max(1, min(idx, ntb));
}

// The DOUBLE PRECISION N0_exp form at :2325-2333 and :2369-2377:
// nir = NINT(DLOG10(N0_exp)); N0_exp/10.**nn is DOUBLE over a REAL(4)
// __powisf2 result.  Returned ONE-BASED.
__device__ __forceinline__ int thompson_aa_wrf_table_index_d(
    double v, int n2, int ntb)
{
    const int nic = (int)round(thompson_aa_log10(v));
    int n = nic + 1;
    for (int nn = nic - 1; nn <= nic + 1; ++nn) {
        const double m = aaf_dd(v, (double)thompson_aa_powi_f(10.0f, nn));
        if (m >= 1.0 && m < 10.0) { n = nn; break; }
    }
    const int idx = (int)aaf_dd(v, (double)thompson_aa_powi_f(10.0f, n))
        + 10 * (n - n2) - (n - n2);
    return max(1, min(idx, ntb));
}

// WRF's REAL(4) constants: PARAMETERs folded as gfortran folds them, and
// the init-time values thompson_init forms (:663-817), as the instrumented
// WRF v4.6.1 build writes them out (the literals below are those values).
#define AAF_PI          3.1415926536f
#define AAF_T0          273.15f
#define AAF_R1          1.0e-12f
#define AAF_R2          1.0e-6f
#define AAF_EPS         1.0e-15f
#define AAF_XM0I        1.0e-12f
#define AAF_D0C         1.0e-6f
#define AAF_D0R         50.0e-6f
#define AAF_D0S         300.0e-6f
#define AAF_HGFR        235.16f
#define AAF_LSUB        2.834e6f
#define AAF_LVAP0       2.5e6f
#define AAF_ORV         (1.0f / 461.5f)               // oRv
#define AAF_OLFUS       (1.0f / (2.834e6f - 2.5e6f))  // olfus
#define AAF_RHO_NOT     (101325.0f / (287.05f * 298.0f))
#define AAF_AM_R        (AAF_PI * 1000.0f / 6.0f)
#define AAF_AM_I        (AAF_PI * 890.0f / 6.0f)
#define AAF_AM_G5       (AAF_PI * 400.0f / 6.0f)      // am_g(idx_bg1)
#define AAF_OAMS        14.492753982543945f           // 1./am_s
#define AAF_OBM         0.3333333432674408f           // obmr = obmi = obmg
#define AAF_MVD_NUM     ((3.0f + 0.0f) + 0.672f)      // (3.0 + mu + 0.672)
#define AAF_D0I         1.289843385166023e-05f        // (xm0i/am_i)**(1./bm_i)
#define AAF_O6          0.1666666716337204f           // oig2 = org3 = ogg3
#define AAF_T1_QR_QC    22873.9375f                   // PI*.25*av_r*crg(9)
#define AAF_T2_QR_QI    1437212032.0f                 // PI*.25*am_r*av_r*crg(8)
#define AAF_T1_QS_QC    31.41592788696289f            // PI*.25*av_s
#define AAF_T1_QS_SD    0.86f
#define AAF_T2_QS_SD    1.5197088718414307f           // 0.28*Sc3*SQRT(av_s)
#define AAF_T1_QS_ME    4.853478912991704e-06f        // PI*4.*C_sqrd*olfus*0.86
#define AAF_T2_QS_ME    8.576598702347837e-06f
#define AAF_T1_QG_SD    0.86f                         // 0.86*cgg(10,1)
#define AAF_T1_QG_ME    1.6178260921151377e-05f
#define AAF_SC3         0.8581680655479431f           // Sc**(1./3.)
#define AAF_AV_G5       442.0f                        // av_g(idx_bg1)
#define AAF_BV_G5       0.8899999856948853f           // bv_g(idx_bg1)
#define AAF_CGE9_5      3.8899998664855957f           // cge(9,5)
#define AAF_CGE11_5     2.944999933242798f            // cge(11,5)
#define AAF_CGE10_1     2.0f                          // cge(10,1)
#define AAF_CGG6_5      20.36322784423828f            // cgg(6,5)
#define AAF_CGG9_5      5.234762668609619f            // cgg(9,5)
#define AAF_CGG11_5     1.9021706581115723f           // cgg(11,5)
#define AAF_CSE1        3.0f
#define AAF_CSE13       2.549999952316284f            // bv_s + 2.
#define AAF_CSE16       1.774999976158142f            // 1.+(1.+bv_s)/2.
// Dr(1), Dr(nbr), Ds(1), Ds(nbs) as thompson_init builds them in DOUBLE
// from the REAL(4) D0r / D0s (:850-872).
#define AAF_DR1         5.116464832797255e-05
#define AAF_DRN         0.004886186104161873
#define AAF_DS1         0.0003063661781989868
#define AAF_DSN         0.019584408175394835

// One model level of mp_thompson, under WRF's names.  REAL fields are
// float, DOUBLE PRECISION fields double.
struct ThompsonAaLevel {
    // :1798-1951
    float temp, qv, pres, rho, nwfa, nifa;
    float rc, nc, ri, ni, rr, nr, rs, rg, ng;
    bool L_qc, L_qi, L_qr, L_qs, L_qg;
    // :1971-2013
    float tempc, rhof, rhof2, qvs, qvsi, delQvs, satw, sati, ssatw, ssati;
    float diffu, visco, ocp, vsc2, lvap, tcond, twet;
    // :2025-2129
    float smob, smo0, smo1, smoc, smoe, smof, ns;
    // :2135-2151
    double ilamg, N0_g, ilamr, N0_r;
    float mvd_r, mvd_c;
    int nu_c;
    // the frozen block's per-level scalars, :2242-2777
    float orho, xDs, rvs, rvs_p, rvs_pp, gamsc, alphsc, t1_subl, rate_max;
    float xni, xnc, xDi, xmi, oxmi, C_snow, tf, r_frac, g_frac, vts;
    float const_Ri, rime_dens, vtg, stoke_g, xDg, Ef_sw, Ef_gw, vts_boost;
    double lami, ilami;
    int idx_tc, idx_t, idx_c, idx_n, idx_i, idx_i1, idx_r, idx_r1, idx_s;
    int idx_g, idx_g1, idx_IN;
    // the process rates, :1545-1575
    double pnc_wau, pnc_rcw, pnc_scw, pnc_gcw;
    double pna_rca, pna_sca, pna_gca, pnd_rcd, pnd_scd, pnd_gcd;
    double prr_wau, prr_rcw, prr_rcs, prr_rcg, prr_sml, prr_gml, prr_rci;
    double pnr_wau, pnr_rcs, pnr_rcg, pnr_rci, pnr_sml, pnr_gml, pnr_rcr,
           pnr_rfz;
    double pri_inu, pni_inu, pri_ihm, pni_ihm, pri_wfz, pni_wfz, pri_rfz,
           pni_rfz, pri_ide, pni_ide, pri_rci, pni_rci, pni_sci, pni_iau,
           pri_iha, pni_iha;
    double prs_iau, prs_sci, prs_rcs, prs_scw, prs_sde, prs_ihm, prs_ide;
    double prg_scw, prg_rfz, prg_gde, prg_gcw, prg_rci, prg_rcs, prg_rcg,
           prg_ihm;
    double png_rcs, png_rcg, png_scw, png_gde;
};

// :1798-1951, one level.  ng1d is the driver's diagnosed graupel number per
// kilogram (:1267-1281).
__device__ __forceinline__ void thompson_aa_wrf_entry(
    ThompsonAaLevel* L, float t1d, float p1d, float qv1d, float qc1d,
    float nc1d, float qi1d, float ni1d, float qr1d, float nr1d, float qs1d,
    float qg1d, float ng1d, float nwfa1d, float nifa1d)
{
    L->temp = t1d;
    L->qv = fmaxf(1.0e-10f, qv1d);
    L->pres = p1d;
    // :1802  rho = 0.622*pres/(R*temp*(qv+0.622))
    L->rho = aaf_fd(aaf_fm(0.622f, p1d),
                    aaf_fm(aaf_fm(287.04f, t1d), aaf_fa(L->qv, 0.622f)));
    // :1805-1806 (aer_init_opt < 2); naIN1*0.01 is 5000.
    L->nwfa = fmaxf(11.1e6f, fminf(9999.0e6f, aaf_fm(nwfa1d, L->rho)));
    L->nifa = fmaxf(0.5e6f * 0.01f, fminf(9999.0e6f, aaf_fm(nifa1d, L->rho)));

    // :1827-1849
    if (qc1d > AAF_R1) {
        L->rc = aaf_fm(qc1d, L->rho);
        L->nc = thompson_aa_cloud_dist(L->rc, nc1d, L->rho, nullptr,
                                       nullptr);
        L->L_qc = true;
    } else {
        L->rc = AAF_R1;
        L->nc = 2.0f;
        L->L_qc = false;
    }

    // :1851-1876.  cig(1)*oig2 = 1*oig2; am_i*cig(2)*oig1 = am_i*6*1;
    // cie(2)/5.E-6 and cie(2)/300.E-6 are REAL quotients.
    if (qi1d > AAF_R1) {
        L->ri = aaf_fm(qi1d, L->rho);
        L->ni = fmaxf(AAF_R2, aaf_fm(ni1d, L->rho));
        const float pre = aaf_fd(aaf_fm(aaf_fm(1.0f, AAF_O6), L->ri),
                                 AAF_AM_I);
        if (L->ni <= AAF_R2) {
            const double lami = (double)aaf_fd(4.0f, 5.0e-6f);
            L->ni = (float)fmin(999.0e3, aaf_dm((double)pre,
                                                thompson_aa_pow(lami, 3.0)));
        }
        L->L_qi = true;
        double lami = (double)thompson_aa_powf(
            aaf_fd(aaf_fm(aaf_fm(aaf_fm(AAF_AM_I, 6.0f), 1.0f), L->ni),
                   L->ri), AAF_OBM);
        const double ilami = aaf_dd(1.0, lami);
        const float xDi = (float)aaf_dm(4.0, ilami);
        if (xDi < 5.0e-6f) {
            lami = (double)aaf_fd(4.0f, 5.0e-6f);
            L->ni = (float)fmin(999.0e3, aaf_dm((double)pre,
                                                thompson_aa_pow(lami, 3.0)));
        } else if (xDi > 300.0e-6f) {
            lami = (double)aaf_fd(4.0f, 300.0e-6f);
            L->ni = (float)aaf_dm((double)pre, thompson_aa_pow(lami, 3.0));
        }
    } else {
        L->ri = AAF_R1;
        L->ni = AAF_R2;
        L->L_qi = false;
    }

    // :1878-1905.  crg(2)*org3 = 1*org3; am_r*crg(3)*org2 = am_r*6*1.
    if (qr1d > AAF_R1) {
        L->rr = aaf_fm(qr1d, L->rho);
        L->nr = fmaxf(AAF_R2, aaf_fm(nr1d, L->rho));
        const float pre = aaf_fm(aaf_fm(1.0f, AAF_O6), L->rr);
        if (L->nr <= AAF_R2) {
            const double lamr = (double)aaf_fd(AAF_MVD_NUM, 1.0e-3f);
            L->nr = (float)aaf_dd(aaf_dm((double)pre,
                                         thompson_aa_pow(lamr, 3.0)),
                                  (double)AAF_AM_R);
        }
        L->L_qr = true;
        const double lamr = (double)thompson_aa_powf(
            aaf_fd(aaf_fm(aaf_fm(aaf_fm(AAF_AM_R, 6.0f), 1.0f), L->nr),
                   L->rr), AAF_OBM);
        const float mvd_r = (float)aaf_dd((double)AAF_MVD_NUM, lamr);
        float bound = 0.0f;
        if (mvd_r > 2.5e-3f) {
            bound = 2.5e-3f;
        } else if (mvd_r < AAF_D0R * 0.75f) {
            bound = AAF_D0R * 0.75f;
        }
        if (bound != 0.0f) {
            const double lb = (double)aaf_fd(AAF_MVD_NUM, bound);
            L->nr = (float)aaf_dd(aaf_dm((double)pre,
                                         thompson_aa_pow(lb, 3.0)),
                                  (double)AAF_AM_R);
        }
    } else {
        L->rr = AAF_R1;
        L->nr = AAF_R2;
        L->L_qr = false;
    }

    // :1906-1914
    if (qs1d > AAF_R1) {
        L->rs = aaf_fm(qs1d, L->rho);
        L->L_qs = true;
    } else {
        L->rs = AAF_R1;
        L->L_qs = false;
    }

    // :1915-1949, not hail aware: idx_bg = idx_bg1 = 5.
    // cgg(2,1)*ogg3 = 1*ogg3; am_g*cgg(3,1)*ogg2 = am_g*6*1.
    if (qg1d > AAF_R1) {
        L->L_qg = true;
        L->rg = aaf_fm(qg1d, L->rho);
        L->ng = fmaxf(AAF_R2, aaf_fm(ng1d, L->rho));
        const float pre = aaf_fm(aaf_fm(1.0f, AAF_O6), L->rg);
        if (L->ng <= AAF_R2) {
            const double lamg = (double)aaf_fd(AAF_MVD_NUM, 1.5e-3f);
            L->ng = (float)aaf_dd(aaf_dm((double)pre,
                                         thompson_aa_pow(lamg, 3.0)),
                                  (double)AAF_AM_G5);
        }
        const double lamg = (double)thompson_aa_powf(
            aaf_fd(aaf_fm(aaf_fm(aaf_fm(AAF_AM_G5, 6.0f), 1.0f), L->ng),
                   L->rg), AAF_OBM);
        const float mvd_g = (float)aaf_dd((double)AAF_MVD_NUM, lamg);
        float bound = 0.0f;
        if (mvd_g > 25.4e-3f) {
            bound = 25.4e-3f;
        } else if (mvd_g < AAF_D0R) {
            bound = AAF_D0R;
        }
        if (bound != 0.0f) {
            const double lb = (double)aaf_fd(AAF_MVD_NUM, bound);
            L->ng = (float)aaf_dd(aaf_dm((double)pre,
                                         thompson_aa_pow(lb, 3.0)),
                                  (double)AAF_AM_G5);
        }
    } else {
        L->rg = AAF_R1;
        L->ng = AAF_R2;
        L->L_qg = false;
    }
}

// :1971-2002 for one level, then the snow moments (:2025-2129) and the
// graupel and rain intercepts (:2135-2151).  twet is the caller's
// (:2004-2013; at a level below 0 C only its comparison with T_0 is ever
// read, and there it agrees with temp's).
__device__ __forceinline__ void thompson_aa_wrf_level_state(
    ThompsonAaLevel* L, float twet)
{
    const float tempc = aaf_fs(L->temp, 273.15f);
    L->tempc = tempc;
    L->rhof = sqrtf(aaf_fd(AAF_RHO_NOT, L->rho));
    L->rhof2 = sqrtf(L->rhof);
    L->qvs = thompson_rslf(L->pres, L->temp);
    L->delQvs = fmaxf(0.0f, aaf_fs(thompson_rslf(L->pres, 273.15f), L->qv));
    L->qvsi = tempc <= 0.0f ? thompson_rsif(L->pres, L->temp) : L->qvs;
    L->satw = aaf_fd(L->qv, L->qvs);
    L->sati = aaf_fd(L->qv, L->qvsi);
    L->ssatw = aaf_fs(L->satw, 1.0f);
    L->ssati = aaf_fs(L->sati, 1.0f);
    if (fabsf(L->ssatw) < AAF_EPS) L->ssatw = 0.0f;
    if (fabsf(L->ssati) < AAF_EPS) L->ssati = 0.0f;
    // :1991  2.11E-5*(temp/273.15)**1.94 * (101325./pres)
    L->diffu = aaf_fm(aaf_fm(2.11e-5f, thompson_aa_powf(
        aaf_fd(L->temp, 273.15f), 1.94f)), aaf_fd(101325.0f, L->pres));
    if (tempc >= 0.0f) {
        L->visco = aaf_fm(aaf_fa(1.718f, aaf_fm(0.0049f, tempc)), 1.0e-5f);
    } else {
        L->visco = aaf_fm(aaf_fs(aaf_fa(1.718f, aaf_fm(0.0049f, tempc)),
                                 aaf_fm(aaf_fm(1.2e-5f, tempc), tempc)),
                          1.0e-5f);
    }
    L->ocp = aaf_fd(1.0f, aaf_fm(1004.0f, aaf_fa(1.0f,
                                                 aaf_fm(0.887f, L->qv))));
    L->vsc2 = sqrtf(aaf_fd(L->rho, L->visco));
    L->lvap = aaf_fa(AAF_LVAP0, aaf_fm(2106.0f - 4218.0f, tempc));
    L->tcond = aaf_fm(aaf_fm(aaf_fa(5.69f, aaf_fm(0.0168f, tempc)), 1.0e-5f),
                      418.936f);
    L->twet = twet;

    // :2027-2129.  bm_s = 2, so smo2 = smob.
    L->smob = L->smo0 = L->smo1 = L->smoc = L->smoe = L->smof = 0.0f;
    L->ns = 0.0f;
    if (L->L_qs) {
        const float tc0 = fminf(-0.1f, tempc);
        L->smob = aaf_fm(L->rs, AAF_OAMS);
        L->smo0 = aaf_fm(thompson_field_a(tc0, 0.0f), thompson_aa_powf(
            L->smob, thompson_field_b(tc0, 0.0f)));
        L->smo1 = aaf_fm(thompson_field_a(tc0, 1.0f), thompson_aa_powf(
            L->smob, thompson_field_b(tc0, 1.0f)));
        L->smoc = aaf_fm(thompson_field_a(tc0, AAF_CSE1), thompson_aa_powf(
            L->smob, thompson_field_b(tc0, AAF_CSE1)));
        L->ns = thompson_aa_snow_number(L->smob, L->smoc);
        L->smoe = aaf_fm(thompson_field_a(tc0, AAF_CSE13), thompson_aa_powf(
            L->smob, thompson_field_b(tc0, AAF_CSE13)));
        L->smof = aaf_fm(thompson_field_a(tc0, AAF_CSE16), thompson_aa_powf(
            L->smob, thompson_field_b(tc0, AAF_CSE16)));
    }

    // :2135-2139, every level.  N0_g = ng*ogg2*lamg**cge(2,1), with ogg2 = 1
    // and cge(2,1) = 1 (the library's pow(x, 1.0) is x).
    {
        const double lamg = (double)thompson_aa_powf(
            aaf_fd(aaf_fm(aaf_fm(aaf_fm(AAF_AM_G5, 6.0f), 1.0f), L->ng),
                   L->rg), AAF_OBM);
        L->ilamg = aaf_dd(1.0, lamg);
        L->N0_g = aaf_dm((double)aaf_fm(L->ng, 1.0f), lamg);
    }
    // :2146-2151, every level.  N0_r = nr*org2*lamr**cre(2), org2 = 1,
    // cre(2) = 1.
    {
        const double lamr = (double)thompson_aa_powf(
            aaf_fd(aaf_fm(aaf_fm(aaf_fm(AAF_AM_R, 6.0f), 1.0f), L->nr),
                   L->rr), AAF_OBM);
        L->ilamr = aaf_dd(1.0, lamr);
        L->mvd_r = (float)aaf_dd((double)AAF_MVD_NUM, lamr);
        L->N0_r = aaf_dm((double)aaf_fm(L->nr, 1.0f), lamr);
    }
}

// Zero every rate (:1667-1769).
__device__ __forceinline__ void thompson_aa_wrf_zero_rates(ThompsonAaLevel* L)
{
    L->pnc_wau = L->pnc_rcw = L->pnc_scw = L->pnc_gcw = 0.0;
    L->pna_rca = L->pna_sca = L->pna_gca = 0.0;
    L->pnd_rcd = L->pnd_scd = L->pnd_gcd = 0.0;
    L->prr_wau = L->prr_rcw = L->prr_rcs = L->prr_rcg = L->prr_sml = 0.0;
    L->prr_gml = L->prr_rci = 0.0;
    L->pnr_wau = L->pnr_rcs = L->pnr_rcg = L->pnr_rci = L->pnr_sml = 0.0;
    L->pnr_gml = L->pnr_rcr = L->pnr_rfz = 0.0;
    L->pri_inu = L->pni_inu = L->pri_ihm = L->pni_ihm = L->pri_wfz = 0.0;
    L->pni_wfz = L->pri_rfz = L->pni_rfz = L->pri_ide = L->pni_ide = 0.0;
    L->pri_rci = L->pni_rci = L->pni_sci = L->pni_iau = L->pri_iha = 0.0;
    L->pni_iha = 0.0;
    L->prs_iau = L->prs_sci = L->prs_rcs = L->prs_scw = L->prs_sde = 0.0;
    L->prs_ihm = L->prs_ide = 0.0;
    L->prg_scw = L->prg_rfz = L->prg_gde = L->prg_gcw = L->prg_rci = 0.0;
    L->prg_rcs = L->prg_rcg = L->prg_ihm = 0.0;
    L->png_rcs = L->png_rcg = L->png_scw = L->png_gde = 0.0;
    L->vts_boost = 1.0f;
    L->xDi = L->xmi = L->oxmi = L->C_snow = L->tf = L->r_frac = 0.0f;
    L->g_frac = L->vts = L->const_Ri = L->rime_dens = L->vtg = 0.0f;
    L->stoke_g = L->xDg = L->Ef_sw = L->Ef_gw = L->xni = L->xnc = 0.0f;
    L->rate_max = 0.0f;
    L->lami = L->ilami = 0.0;
    L->idx_IN = 0;
}

// (lamr + fv_r)**(-cre(n)) with lamr = 1./ilamr(k), as :2198-2203 forms it.
__device__ __forceinline__ double aaf_rain_kernel(double ilamr, double expo)
{
    return thompson_aa_pow(aaf_da(aaf_dd(1.0, ilamr), 195.0), expo);
}

// :2157-2234, the warm-rain loop, one level.
__device__ __forceinline__ void thompson_aa_wrf_warm_loop(
    ThompsonAaLevel* L, float odts, const double* __restrict__ t_Efrw)
{
    // :2161-2167
    if (L->L_qr && L->mvd_r > AAF_D0R) {
        const float Ef_rr = aaf_fs(1.0f, thompson_aa_expf(
            aaf_fm(2300.0f, aaf_fs(L->mvd_r, 1950.0e-6f))));
        L->pnr_rcr = (double)aaf_fm(aaf_fm(aaf_fm(Ef_rr, 2.0f), L->nr),
                                    L->rr);
    }
    // :2169-2176
    L->mvd_c = AAF_D0C;
    float xDc = 0.0f;
    double lamc = 0.0;
    int nu_c = 0;
    if (L->L_qc) {
        nu_c = min(15, (int)roundf(aaf_fd(1000.0e6f, L->nc)) + 2);
        xDc = fmaxf(AAF_D0C * 1.0e6f, aaf_fm(thompson_aa_powf(
            aaf_fd(L->rc, aaf_fm(AAF_AM_R, L->nc)), AAF_OBM), 1.0e6f));
        lamc = (double)thompson_aa_powf(
            aaf_fd(aaf_fm(aaf_fm(aaf_fm(L->nc, AAF_AM_R),
                                 THOMPSON_AA_CCG2[nu_c]),
                          THOMPSON_AA_OCG1[nu_c]), L->rc), AAF_OBM);
        const float num = aaf_fa(aaf_fa(3.0f, (float)nu_c), 0.672f);
        L->mvd_c = (float)aaf_dd((double)num, lamc);
        L->mvd_c = fmaxf(AAF_D0C, fminf(L->mvd_c, AAF_D0R));
    }
    L->nu_c = nu_c;
    // :2180-2194, Berry and Reinhardt.
    if (L->rc > 0.01e-3f) {
        const float Dc_g = (float)aaf_dm(aaf_dd((double)thompson_aa_powf(
            aaf_fm(THOMPSON_AA_CCG3[nu_c], THOMPSON_AA_OCG2[nu_c]), AAF_OBM),
            lamc), 1.0e6);
        const float x3 = aaf_fm(aaf_fm(xDc, xDc), xDc);
        const float arg = aaf_fs(
            aaf_fm(aaf_fm(aaf_fm(x3, Dc_g), Dc_g), Dc_g),
            aaf_fm(aaf_fm(aaf_fm(x3, xDc), xDc), xDc));
        const float Dc_b = thompson_aa_powf(fmaxf(0.0f, arg), 1.0f / 6.0f);
        const float z = aaf_fs(aaf_fm(aaf_fm(aaf_fm(aaf_fm(6.25e-6f, xDc),
                                                    Dc_b), Dc_b), Dc_b),
                               0.4f);
        const float zeta1 = aaf_fm(0.5f, aaf_fa(z, fabsf(z)));
        const float zeta = aaf_fm(aaf_fm(0.027f, L->rc), zeta1);
        const float td = aaf_fs(aaf_fm(0.5f, Dc_b), 7.5f);
        const float taud = aaf_fa(aaf_fm(0.5f, aaf_fa(td, fabsf(td))), AAF_R1);
        const float tau = aaf_fd(3.72f, aaf_fm(L->rc, taud));
        L->prr_wau = (double)aaf_fd(zeta, tau);
        L->prr_wau = fmin((double)aaf_fm(L->rc, odts), L->prr_wau);
        L->pnr_wau = aaf_dd(L->prr_wau, (double)aaf_fm(aaf_fm(aaf_fm(
            aaf_fm(aaf_fm(AAF_AM_R, (float)nu_c), 10.0f), AAF_D0R), AAF_D0R),
            AAF_D0R));
        L->pnc_wau = fmin((double)aaf_fm(L->nc, odts),
                          aaf_dd(L->prr_wau, (double)aaf_fm(aaf_fm(
                              aaf_fm(AAF_AM_R, L->mvd_c), L->mvd_c),
                              L->mvd_c)));
    }
    // :2197-2208
    if (L->L_qr && L->mvd_r > AAF_D0R && L->mvd_c > AAF_D0C) {
        int idx = 1 + (int)aaf_dd(aaf_dm(100.0, thompson_aa_log(
            aaf_dd((double)L->mvd_r, AAF_DR1))),
            thompson_aa_log(aaf_dd(AAF_DRN, AAF_DR1)));
        idx = min(idx, 100);
        const int jdx = (int)aaf_fm(L->mvd_c, 1.0e6f);
        const float Ef_rw = (float)t_Efrw[(idx - 1) + 100 * (jdx - 1)];
        const double kern = aaf_rain_kernel(L->ilamr, -4.0);
        L->prr_rcw = aaf_dm(aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
            L->rhof, AAF_T1_QR_QC), Ef_rw), L->rc), L->N0_r), kern);
        L->prr_rcw = fmin((double)aaf_fm(L->rc, odts), L->prr_rcw);
        L->pnc_rcw = aaf_dm(aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
            L->rhof, AAF_T1_QR_QC), Ef_rw), L->nc), L->N0_r), kern);
        L->pnc_rcw = fmin((double)aaf_fm(L->nc, odts), L->pnc_rcw);
    }
    // :2211-2222
    if (L->L_qr && L->mvd_r > AAF_D0R) {
        const double kern = aaf_rain_kernel(L->ilamr, -4.0);
        float Ef_ra = thompson_eff_aero(L->mvd_r, 0.04e-6f, L->visco,
                                        L->rho, L->temp,
                                        THOMPSON_AA_SPECIES_RAIN);
        L->pna_rca = aaf_dm(aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
            L->rhof, AAF_T1_QR_QC), Ef_ra), L->nwfa), L->N0_r), kern);
        L->pna_rca = fmin((double)aaf_fm(L->nwfa, odts), L->pna_rca);
        Ef_ra = thompson_eff_aero(L->mvd_r, 0.8e-6f, L->visco, L->rho,
                                  L->temp, THOMPSON_AA_SPECIES_RAIN);
        L->pnd_rcd = aaf_dm(aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
            L->rhof, AAF_T1_QR_QC), Ef_ra), L->nifa), L->N0_r), kern);
        L->pnd_rcd = fmin((double)aaf_fm(L->nifa, odts), L->pnd_rcd);
    }
}

// The lami/ilami/xDi/xmi/oxmi block WRF writes twice (:2647-2651 and
// :2711-2715).
__device__ __forceinline__ void aaf_ice_size(ThompsonAaLevel* L)
{
    L->lami = (double)thompson_aa_powf(
        aaf_fd(aaf_fm(aaf_fm(aaf_fm(AAF_AM_I, 6.0f), 1.0f), L->ni), L->ri),
        AAF_OBM);
    L->ilami = aaf_dd(1.0, L->lami);
    L->xDi = (float)fmax((double)AAF_D0I, aaf_dm(4.0, L->ilami));
    L->xmi = aaf_fm(AAF_AM_I, thompson_aa_powf(L->xDi, 3.0f));
    L->oxmi = aaf_fd(1.0f, L->xmi);
}

// The tables the frozen block reads, float64 in Fortran order.
struct ThompsonAaFrozenTables {
    const double* tpi_ide;   const double* tps_iaus;  const double* tni_iaus;
    const double* tcs_racs1; const double* tmr_racs1;
    const double* tcs_racs2; const double* tmr_racs2;
    const double* tcr_sacr1; const double* tms_sacr1;
    const double* tcr_sacr2; const double* tms_sacr2;
    const double* tnr_racs1; const double* tnr_racs2;
    const double* tnr_sacr1; const double* tnr_sacr2;
    const double* tcg_racg;  const double* tmr_racg;  const double* tcr_gacr;
    const double* tnr_racg;  const double* tnr_gacr;
    const double* tpi_qrfz;  const double* tni_qrfz;
    const double* tpg_qrfz;  const double* tnr_qrfz;
    const double* tpi_qcfz;  const double* tni_qcfz;
    const double* t_Efsw;
};

// THE RAIN-GRAUPEL TABLE INDEX.  WRF allocates tcg_racg..tnr_gacr as
// (ntb_g1, ntb_g, dimNRHG, ntb_r1, ntb_r) with dimNRHG = NRHG1 = 1 when the
// scheme is not hail aware (:465, :607-615), builds its one density slab
// for rho_g(idx_bg1) (:4123), and then reads it at :2527-2545 with
// idx_bg(k) = idx_bg1 = 5 (:1950, :2848): an out-of-bounds subscript on a
// dimension of extent 1, which gfortran without bounds checking turns into
// a read 4*37*37 words further on (idx_r1+4, and past the end of the array
// for the last rain bins).  WOOF does not reproduce an out-of-bounds read:
// it reads the one slab the table holds, the collision rates WRF built for
// this graupel density.  This is a DECLARED divergence from WRF v4.6.1
// wherever rain meets graupel (prg_rcg, prr_rcg, pnr_rcg, png_rcg and
// everything downstream of them).
__device__ __forceinline__ size_t aaf_racg_index(int idx_g1, int idx_g,
                                                 int idx_r1, int idx_r)
{
    return (size_t)(idx_g1 - 1) + (size_t)37 * ((size_t)(idx_g - 1)
        + (size_t)37 * ((size_t)(idx_r1 - 1)
        + (size_t)37 * (size_t)(idx_r - 1)));
}

// :2242-2848, the frozen-species block for one level, both temperature
// branches.  dt is dtsave; odts = 1./dtsave.
__device__ __forceinline__ void thompson_aa_wrf_frozen_block(
    ThompsonAaLevel* L, float dt, float odts,
    const ThompsonAaFrozenTables* T)
{
    const float tempc = L->tempc;
    L->vts_boost = 1.0f;
    L->orho = aaf_fd(1.0f, L->rho);

    L->xDs = L->L_qs ? aaf_fd(L->smoc, L->smob) : 0.0f;

    // :2255-2259.  INT truncates toward zero.
    L->idx_tc = max(1, min((int)roundf(-tempc), 45));
    int idx_t = (int)aaf_fd(aaf_fs(tempc, 2.5f), 5.0f) - 1;
    idx_t = max(1, -idx_t);
    L->idx_t = min(idx_t, 9);

    // :2263-2275.  r_c(1) = 1.e-6, nic2 = -6.
    L->idx_c = L->rc > 1.0e-6f
        ? thompson_aa_wrf_table_index(L->rc, -6, 37) : 1;
    // :2278-2279.  NINT(1.0 + FLOAT(nbc)*DLOG(nc/t_Nc(1))/nic1), nic1 = 7.
    {
        const double raw = aaf_da(1.0, aaf_dd(aaf_dm(100.0, thompson_aa_log(
            aaf_dd((double)L->nc, THOMPSON_AA_T_NC_1))), 7.0));
        L->idx_n = max(1, min((int)round(raw), 100));
    }
    // :2282-2308.  r_i(1) = 1.e-10 (nii2 = -10), Nt_i(1) = 1 (nii3 = 0).
    L->idx_i = L->ri > 1.0e-10f
        ? thompson_aa_wrf_table_index(L->ri, -10, 64) : 1;
    L->idx_i1 = L->ni > 1.0f
        ? thompson_aa_wrf_table_index(L->ni, 0, 55) : 1;
    // :2311-2337.  r_r(1) = 1.e-6 (nir2 = -6), N0r_exp(1) = 1.e6 (nir3 = 6).
    // lam_exp = lamr*(crg(3)*org2*org1)**bm_r: the REAL base 6*1*(1/6)
    // rounds to 1.0, powf(1, 3) = 1.
    if (L->rr > 1.0e-6f) {
        L->idx_r = thompson_aa_wrf_table_index(L->rr, -6, 37);
        const double lamr = aaf_dd(1.0, L->ilamr);
        const double lam_exp = aaf_dm(lamr, (double)thompson_aa_powf(
            aaf_fm(aaf_fm(6.0f, 1.0f), AAF_O6), 3.0f));
        const double N0_exp = aaf_dm((double)aaf_fd(aaf_fm(AAF_O6, L->rr),
                                                    AAF_AM_R),
                                     thompson_aa_pow(lam_exp, 4.0));
        L->idx_r1 = thompson_aa_wrf_table_index_d(N0_exp, 6, 37);
    } else {
        L->idx_r = 1;
        L->idx_r1 = 37;
    }
    // :2340-2352.  r_s(1) = 1.e-6.
    L->idx_s = L->rs > 1.0e-6f
        ? thompson_aa_wrf_table_index(L->rs, -6, 37) : 1;
    // :2355-2381.  r_g(1) = 1.e-6 (nig2 = -6), N0g_exp(1) = 1.e2 (nig3 = 2).
    if (L->rg > 1.0e-6f) {
        L->idx_g = thompson_aa_wrf_table_index(L->rg, -6, 37);
        const double lamg = aaf_dd(1.0, L->ilamg);
        const double lam_exp = aaf_dm(lamg, (double)thompson_aa_powf(
            aaf_fm(aaf_fm(6.0f, 1.0f), AAF_O6), 3.0f));
        const double N0_exp = aaf_dm((double)aaf_fd(aaf_fm(AAF_O6, L->rg),
                                                    AAF_AM_G5),
                                     thompson_aa_pow(lam_exp, 4.0));
        L->idx_g1 = thompson_aa_wrf_table_index_d(N0_exp, 2, 37);
    } else {
        L->idx_g = 1;
        L->idx_g1 = 37;
    }

    // :2384-2400, the deposition/sublimation prefactor.
    const float otemp = aaf_fd(1.0f, L->temp);
    L->rvs = aaf_fm(L->rho, L->qvsi);
    const float X = aaf_fs(aaf_fm(aaf_fm(AAF_LSUB, otemp), AAF_ORV), 1.0f);
    L->rvs_p = aaf_fm(aaf_fm(L->rvs, otemp), X);
    L->rvs_pp = aaf_fm(L->rvs, aaf_fa(aaf_fa(
        aaf_fm(aaf_fm(aaf_fm(otemp, X), otemp), X),
        -aaf_fm(aaf_fm(aaf_fm(aaf_fm(2.0f * AAF_LSUB, otemp), otemp), otemp),
                AAF_ORV)),
        aaf_fm(otemp, otemp)));
    L->gamsc = aaf_fm(aaf_fd(aaf_fm(AAF_LSUB, L->diffu), L->tcond), L->rvs_p);
    {
        const float g1 = aaf_fd(L->gamsc, aaf_fa(1.0f, L->gamsc));
        L->alphsc = aaf_fd(aaf_fm(aaf_fd(aaf_fm(aaf_fm(aaf_fm(0.5f, g1), g1),
                                                L->rvs_pp), L->rvs_p),
                                  L->rvs), L->rvs_p);
    }
    L->alphsc = fmaxf(1.0e-9f, L->alphsc);
    float xsat = L->ssati;
    if (fabsf(xsat) < 1.0e-9f) xsat = 0.0f;
    {
        const float a = L->alphsc;
        float p = aaf_fs(1.0f, aaf_fm(a, xsat));
        p = aaf_fa(p, aaf_fm(aaf_fm(aaf_fm(aaf_fm(2.0f, a), a), xsat), xsat));
        p = aaf_fs(p, aaf_fm(aaf_fm(aaf_fm(aaf_fm(aaf_fm(aaf_fm(5.0f, a), a),
                                                  a), xsat), xsat), xsat));
        L->t1_subl = aaf_fd(aaf_fm(4.0f * AAF_PI, p), aaf_fa(1.0f, L->gamsc));
    }

    // :2403-2440, snow and graupel collecting cloud water.
    const float t1_qg_qc = aaf_fm(aaf_fm(AAF_PI * 0.25f, AAF_AV_G5),
                                  AAF_CGG9_5);
    if (L->L_qc && L->mvd_c > AAF_D0C) {
        if (L->xDs > AAF_D0S) {
            int idx = 1 + (int)aaf_dd(aaf_dm(100.0, thompson_aa_log(
                aaf_dd((double)L->xDs, AAF_DS1))),
                thompson_aa_log(aaf_dd(AAF_DSN, AAF_DS1)));
            idx = min(idx, 100);
            const int jdx = (int)aaf_fm(L->mvd_c, 1.0e6f);
            L->Ef_sw = (float)T->t_Efsw[(idx - 1) + 100 * (jdx - 1)];
            L->prs_scw = (double)aaf_fm(aaf_fm(aaf_fm(aaf_fm(
                L->rhof, AAF_T1_QS_QC), L->Ef_sw), L->rc), L->smoe);
            L->prs_scw = fmin((double)aaf_fm(L->rc, odts), L->prs_scw);
            L->pnc_scw = (double)aaf_fm(aaf_fm(aaf_fm(aaf_fm(
                L->rhof, AAF_T1_QS_QC), L->Ef_sw), L->nc), L->smoe);
            L->pnc_scw = fmin((double)aaf_fm(L->nc, odts), L->pnc_scw);
        }
        if (L->rg >= 1.0e-6f && L->mvd_c > AAF_D0C) {
            L->xDg = (float)aaf_dm(4.0, L->ilamg);
            L->vtg = (float)aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
                L->rhof, AAF_AV_G5), AAF_CGG6_5), AAF_O6),
                thompson_aa_pow(L->ilamg, (double)AAF_BV_G5));
            L->stoke_g = aaf_fd(aaf_fm(aaf_fm(aaf_fm(L->mvd_c, L->mvd_c),
                                              L->vtg), 1000.0f),
                                aaf_fm(aaf_fm(9.0f, L->visco), L->xDg));
            float Ef_gw = 0.0f;
            if (L->stoke_g >= 0.4f && L->stoke_g <= 10.0f) {
                Ef_gw = aaf_fm(0.55f, thompson_aa_log10f(
                    aaf_fm(2.51f, L->stoke_g)));
            } else if (L->stoke_g > 10.0f) {
                Ef_gw = 0.77f;
            }
            if (L->twet > AAF_T0) Ef_gw = aaf_fm(Ef_gw, 0.1f);
            L->Ef_gw = Ef_gw;
            const double kern = thompson_aa_pow(L->ilamg, (double)AAF_CGE9_5);
            L->prg_gcw = aaf_dm(aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
                L->rhof, t1_qg_qc), Ef_gw), L->rc), L->N0_g), kern);
            L->pnc_gcw = aaf_dm(aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
                L->rhof, t1_qg_qc), Ef_gw), L->nc), L->N0_g), kern);
            L->pnc_gcw = fmin((double)aaf_fm(L->nc, odts), L->pnc_gcw);
        }
    }

    // :2443-2481, snow and graupel collecting aerosols.
    if (L->rs > 1.0e-6f) {
        float Ef_sa = thompson_eff_aero(L->xDs, 0.04e-6f, L->visco, L->rho,
                                        L->temp, THOMPSON_AA_SPECIES_SNOW);
        L->pna_sca = (double)aaf_fm(aaf_fm(aaf_fm(aaf_fm(
            L->rhof, AAF_T1_QS_QC), Ef_sa), L->nwfa), L->smoe);
        L->pna_sca = fmin((double)aaf_fm(L->nwfa, odts), L->pna_sca);
        Ef_sa = thompson_eff_aero(L->xDs, 0.8e-6f, L->visco, L->rho,
                                  L->temp, THOMPSON_AA_SPECIES_SNOW);
        L->pnd_scd = (double)aaf_fm(aaf_fm(aaf_fm(aaf_fm(
            L->rhof, AAF_T1_QS_QC), Ef_sa), L->nifa), L->smoe);
        L->pnd_scd = fmin((double)aaf_fm(L->nifa, odts), L->pnd_scd);
    }
    if (L->rg > 1.0e-6f) {
        L->xDg = (float)aaf_dm(4.0, L->ilamg);
        float Ef_ga = thompson_eff_aero(L->xDg, 0.04e-6f, L->visco, L->rho,
                                        L->temp,
                                        THOMPSON_AA_SPECIES_GRAUPEL);
        const double kern = thompson_aa_pow(L->ilamg, (double)AAF_CGE9_5);
        L->pna_gca = aaf_dm(aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
            L->rhof, t1_qg_qc), Ef_ga), L->nwfa), L->N0_g), kern);
        L->pna_gca = fmin((double)aaf_fm(L->nwfa, odts), L->pna_gca);
        Ef_ga = thompson_eff_aero(L->xDg, 0.8e-6f, L->visco, L->rho,
                                  L->temp, THOMPSON_AA_SPECIES_GRAUPEL);
        L->pnd_gcd = aaf_dm(aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
            L->rhof, t1_qg_qc), Ef_ga), L->nifa), L->N0_g), kern);
        L->pnd_gcd = fmin((double)aaf_fm(L->nifa, odts), L->pnd_gcd);
    }

    // :2486-2548, rain collecting snow and graupel.
    if (L->rr >= 1.0e-6f) {
        if (L->rs >= 1.0e-6f) {
            const size_t i = (size_t)(L->idx_s - 1) + (size_t)37
                * ((size_t)(L->idx_t - 1) + (size_t)9
                * ((size_t)(L->idx_r1 - 1) + (size_t)37
                * (size_t)(L->idx_r - 1)));
            if (L->twet < AAF_T0) {
                L->prr_rcs = -(T->tmr_racs2[i] + T->tcr_sacr2[i]
                               + T->tmr_racs1[i] + T->tcr_sacr1[i]);
                L->prs_rcs = T->tmr_racs2[i] + T->tcr_sacr2[i]
                    - T->tcs_racs1[i] - T->tms_sacr1[i];
                L->prg_rcs = T->tmr_racs1[i] + T->tcr_sacr1[i]
                    + T->tcs_racs1[i] + T->tms_sacr1[i];
                L->prr_rcs = fmax((double)aaf_fm(-L->rr, odts), L->prr_rcs);
                L->prs_rcs = fmax((double)aaf_fm(-L->rs, odts), L->prs_rcs);
                L->prg_rcs = fmin((double)aaf_fm(aaf_fa(L->rr, L->rs), odts),
                                  L->prg_rcs);
                L->pnr_rcs = T->tnr_racs1[i] + T->tnr_racs2[i]
                    + T->tnr_sacr1[i] + T->tnr_sacr2[i];
                L->pnr_rcs = fmin((double)aaf_fm(L->nr, odts), L->pnr_rcs);
                L->png_rcs = L->pnr_rcs;
            } else {
                L->prs_rcs = -T->tcs_racs1[i] - T->tms_sacr1[i]
                    + T->tmr_racs2[i] + T->tcr_sacr2[i];
                L->prs_rcs = fmax((double)aaf_fm(-L->rs, odts), L->prs_rcs);
                L->prr_rcs = -L->prs_rcs;
            }
        }
        if (L->rg >= 1.0e-6f) {
            const size_t i = aaf_racg_index(L->idx_g1, L->idx_g, L->idx_r1,
                                            L->idx_r);
            if (L->twet < AAF_T0) {
                L->prg_rcg = T->tmr_racg[i] + T->tcr_gacr[i];
                L->prg_rcg = fmin((double)aaf_fm(L->rr, odts), L->prg_rcg);
                L->prr_rcg = -L->prg_rcg;
                L->pnr_rcg = T->tnr_racg[i] + T->tnr_gacr[i];
                L->pnr_rcg = fmin((double)aaf_fm(L->nr, odts), L->pnr_rcg);
            } else {
                L->prr_rcg = T->tcg_racg[i];
                L->prr_rcg = fmin((double)aaf_fm(L->rg, odts), L->prr_rcg);
                L->prg_rcg = -L->prr_rcg;
                L->png_rcg = T->tnr_racg[i];
                L->png_rcg = fmin((double)aaf_fm(L->ng, odts), L->png_rcg);
                L->pnr_rcg = -1.5 * T->tnr_gacr[i];
            }
        }
    }

    if (L->temp < AAF_T0) {
        // ---- :2554-2777, below 0 C ----------------------------------------
        L->vts_boost = 1.0f;
        L->rate_max = aaf_fm(aaf_fm(aaf_fm(aaf_fs(L->qv, L->qvsi), L->rho),
                                    odts), 0.999f);
        // :2573-2577
        L->xni = thompson_ice_demott(tempc, L->rho, L->nifa);
        // :2580-2592.  Nt_IN(1) = 1, niIN2 = 0.
        L->idx_IN = L->xni > 1.0f
            ? thompson_aa_wrf_table_index(L->xni, 0, 55) : 1;

        // :2595-2605, freezing of rain.
        {
            const size_t i = (size_t)(L->idx_r - 1) + (size_t)37
                * ((size_t)(L->idx_r1 - 1) + (size_t)37
                * ((size_t)(L->idx_tc - 1) + (size_t)45
                * (size_t)(L->idx_IN - 1)));
            if (L->rr > 1.0e-6f) {
                L->prg_rfz = aaf_dm(T->tpg_qrfz[i], (double)odts);
                L->pri_rfz = aaf_dm(T->tpi_qrfz[i], (double)odts);
                L->pni_rfz = aaf_dm(T->tni_qrfz[i], (double)odts);
                L->pnr_rfz = aaf_dm(T->tnr_qrfz[i], (double)odts);
                L->pnr_rfz = fmin((double)aaf_fm(L->nr, odts), L->pnr_rfz);
            } else if (L->rr > AAF_R1 && L->temp < AAF_HGFR) {
                L->pri_rfz = (double)aaf_fm(L->rr, odts);
                L->pni_rfz = (double)aaf_fm(L->nr, odts);
            }
        }
        // :2607-2616, freezing of cloud water.
        {
            const size_t i = (size_t)(L->idx_c - 1) + (size_t)37
                * ((size_t)(L->idx_n - 1) + (size_t)100
                * ((size_t)(L->idx_tc - 1) + (size_t)45
                * (size_t)(L->idx_IN - 1)));
            if (L->rc > 1.0e-6f) {
                L->pri_wfz = aaf_dm(T->tpi_qcfz[i], (double)odts);
                L->pri_wfz = fmin((double)aaf_fm(L->rc, odts), L->pri_wfz);
                L->pni_wfz = aaf_dm(T->tni_qcfz[i], (double)odts);
                L->pni_wfz = fmin(fmin((double)aaf_fm(L->nc, odts),
                                       aaf_dd(L->pri_wfz,
                                              (double)(2.0f * AAF_XM0I))),
                                  L->pni_wfz);
            } else if (L->rc > AAF_R1 && L->temp < AAF_HGFR) {
                L->pri_wfz = (double)aaf_fm(L->rc, odts);
                L->pni_wfz = (double)aaf_fm(L->nc, odts);
            }
        }
        // :2620-2631, deposition nucleation.
        if (L->ssati >= 0.25f
                || (L->ssatw > AAF_EPS && L->temp < 253.15f)) {
            L->xnc = thompson_ice_demott(tempc, L->rho, L->nifa);
            L->xni = (float)aaf_da((double)L->ni, aaf_dm(
                aaf_da(L->pni_rfz, L->pni_wfz), (double)dt));
            const float d = aaf_fs(L->xnc, L->xni);
            L->pni_inu = (double)aaf_fm(aaf_fm(0.5f, aaf_fa(d, fabsf(d))),
                                        odts);
            L->pri_inu = fmin((double)L->rate_max,
                              aaf_dm((double)AAF_XM0I, L->pni_inu));
            L->pni_inu = aaf_dd(L->pri_inu, (double)AAF_XM0I);
        }
        // :2634-2641, Koop haze freezing.
        L->xni = (float)aaf_da((double)aaf_fa(L->ns, L->ni), aaf_dm(
            aaf_da(aaf_da(L->pni_rfz, L->pni_wfz), L->pni_inu), (double)dt));
        if (L->xni <= 999.0e3f && L->temp < 238.0f && L->ssati >= 0.4f) {
            L->xnc = thompson_ice_koop(L->temp, L->qv, L->qvs, L->nwfa, dt);
            L->pni_iha = (double)aaf_fm(L->xnc, odts);
            L->pri_iha = fmin((double)L->rate_max,
                              aaf_dm((double)(AAF_XM0I * 0.1f), L->pni_iha));
            L->pni_iha = aaf_dd(L->pri_iha, (double)(AAF_XM0I * 0.1f));
        }

        // :2646-2679, cloud ice deposition and ice-to-snow.  C_cube = 0.5,
        // oig1 = cig(5) = 1.
        if (L->L_qi) {
            aaf_ice_size(L);
            L->pri_ide = aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(aaf_fm(aaf_fm(
                aaf_fm(aaf_fm(0.5f, L->t1_subl), L->diffu), L->ssati),
                L->rvs), 1.0f), 1.0f), L->ni), L->ilami);
            if (L->pri_ide < 0.0) {
                L->pri_ide = fmax(fmax((double)aaf_fm(-L->ri, odts),
                                       L->pri_ide), (double)L->rate_max);
                L->pni_ide = aaf_dm(L->pri_ide, (double)L->oxmi);
                L->pni_ide = fmax((double)aaf_fm(-L->ni, odts), L->pni_ide);
            } else {
                L->pri_ide = fmin(L->pri_ide, (double)L->rate_max);
                const double t = T->tpi_ide[(L->idx_i - 1)
                                            + 64 * (L->idx_i1 - 1)];
                L->prs_ide = aaf_dm(aaf_ds(1.0, t), L->pri_ide);
                L->pri_ide = aaf_dm(t, L->pri_ide);
            }
            if (L->idx_i == 64 || L->xDi > 5.0f * AAF_D0S) {
                L->prs_iau = (double)aaf_fm(aaf_fm(L->ri, 0.99f), odts);
                L->pni_iau = (double)aaf_fm(aaf_fm(L->ni, 0.95f), odts);
            } else if (L->xDi < 0.1f * AAF_D0S) {
                L->prs_iau = 0.0;
                L->pni_iau = 0.0;
            } else {
                const size_t i = (size_t)(L->idx_i - 1)
                    + (size_t)64 * (size_t)(L->idx_i1 - 1);
                L->prs_iau = aaf_dm(T->tps_iaus[i], (double)odts);
                L->prs_iau = fmin((double)aaf_fm(aaf_fm(L->ri, 0.99f), odts),
                                  L->prs_iau);
                L->pni_iau = aaf_dm(T->tni_iaus[i], (double)odts);
                L->pni_iau = fmin((double)aaf_fm(aaf_fm(L->ni, 0.95f), odts),
                                  L->pni_iau);
            }
        }

        // :2683-2694, snow deposition/sublimation.  C_sqrd = 0.15,
        // C_cube = 0.5.
        if (L->L_qs) {
            L->C_snow = aaf_fa(0.15f, aaf_fd(aaf_fm(aaf_fa(tempc, 1.5f),
                                                    0.5f - 0.15f),
                                             -30.0f + 1.5f));
            L->C_snow = fmaxf(0.15f, fminf(L->C_snow, 0.5f));
            L->prs_sde = (double)aaf_fm(aaf_fm(aaf_fm(aaf_fm(aaf_fm(
                L->C_snow, L->t1_subl), L->diffu), L->ssati), L->rvs),
                aaf_fa(aaf_fm(AAF_T1_QS_SD, L->smo1),
                       aaf_fm(aaf_fm(aaf_fm(AAF_T2_QS_SD, L->rhof2), L->vsc2),
                              L->smof)));
            if (L->prs_sde < 0.0) {
                L->prs_sde = fmax(fmax((double)aaf_fm(-L->rs, odts),
                                       L->prs_sde), (double)L->rate_max);
            } else {
                L->prs_sde = fmin(L->prs_sde, (double)L->rate_max);
            }
        }

        // :2696-2707, graupel deposition/sublimation.
        if (L->L_qg && L->ssati < -AAF_EPS) {
            const float t2_qg_sd = aaf_fm(aaf_fm(aaf_fm(0.28f, AAF_SC3),
                                                 sqrtf(AAF_AV_G5)),
                                          AAF_CGG11_5);
            L->prg_gde = aaf_dm(aaf_dm(
                (double)aaf_fm(aaf_fm(aaf_fm(aaf_fm(0.5f, L->t1_subl),
                                             L->diffu), L->ssati), L->rvs),
                L->N0_g),
                aaf_da(aaf_dm((double)AAF_T1_QG_SD, thompson_aa_pow(
                           L->ilamg, (double)AAF_CGE10_1)),
                       aaf_dm((double)aaf_fm(aaf_fm(t2_qg_sd, L->vsc2),
                                             L->rhof2),
                              thompson_aa_pow(L->ilamg,
                                              (double)AAF_CGE11_5))));
            if (L->prg_gde < 0.0) {
                L->prg_gde = fmax(fmax((double)aaf_fm(-L->rg, odts),
                                       L->prg_gde), (double)L->rate_max);
                L->png_gde = aaf_dd(aaf_dm(L->prg_gde, (double)L->ng),
                                    (double)L->rg);
            } else {
                L->prg_gde = fmin(L->prg_gde, (double)L->rate_max);
            }
        }

        // :2710-2736, snow and rain collecting cloud ice.  t1_qs_qi =
        // t1_qs_qc, Ef_si = 0.05, t1_qr_qi = t1_qr_qc, Ef_ri = 0.95.
        if (L->L_qi) {
            aaf_ice_size(L);
            if (L->rs >= 1.0e-6f) {
                L->prs_sci = (double)aaf_fm(aaf_fm(aaf_fm(aaf_fm(
                    AAF_T1_QS_QC, L->rhof), 0.05f), L->ri), L->smoe);
                L->pni_sci = aaf_dm(L->prs_sci, (double)L->oxmi);
            }
            if (L->rr >= 1.0e-6f && L->mvd_r > aaf_fm(4.0f, L->xDi)) {
                const double k4 = aaf_rain_kernel(L->ilamr, -4.0);
                L->pri_rci = aaf_dm(aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
                    L->rhof, AAF_T1_QR_QC), 0.95f), L->ri), L->N0_r), k4);
                L->pnr_rci = aaf_dm(aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
                    L->rhof, AAF_T1_QR_QC), 0.95f), L->ni), L->N0_r), k4);
                L->pnr_rci = fmin((double)aaf_fm(L->nr, odts), L->pnr_rci);
                L->pni_rci = aaf_dm(L->pri_rci, (double)L->oxmi);
                L->prr_rci = aaf_dm(aaf_dm((double)aaf_fm(aaf_fm(aaf_fm(
                    L->rhof, AAF_T2_QR_QI), 0.95f), L->ni), L->N0_r),
                    aaf_rain_kernel(L->ilamr, -7.0));
                L->prr_rci = fmin((double)aaf_fm(L->rr, odts), L->prr_rci);
                L->prg_rci = aaf_da(L->pri_rci, L->prr_rci);
            }
        }

        // :2739-2752, Hallett-Mossop.
        if (L->prg_gcw > (double)AAF_EPS && tempc > -8.0f) {
            float tf = 0.0f;
            if (tempc >= -5.0f && tempc < -3.0f) {
                tf = aaf_fm(0.5f, aaf_fs(-3.0f, tempc));
            } else if (tempc > -8.0f && tempc < -5.0f) {
                tf = aaf_fm(0.33333333f, aaf_fa(8.0f, tempc));
            }
            L->tf = tf;
            L->pni_ihm = aaf_dm((double)aaf_fm(3.5e8f, tf), L->prg_gcw);
            L->pri_ihm = aaf_dm((double)AAF_XM0I, L->pni_ihm);
            L->prs_ihm = aaf_dm(aaf_dd(L->prs_scw,
                                       aaf_da(L->prs_scw, L->prg_gcw)),
                                L->pri_ihm);
            L->prg_ihm = aaf_dm(aaf_dd(L->prg_gcw,
                                       aaf_da(L->prs_scw, L->prg_gcw)),
                                L->pri_ihm);
        }

        // :2758-2777, rimed snow to graupel.
        if (L->prs_scw > aaf_dm(2.0, L->prs_sde)
                && L->prs_sde > (double)AAF_EPS) {
            L->r_frac = (float)fmin(30.0, aaf_dd(L->prs_scw, L->prs_sde));
            float g_frac = fminf(0.95f, aaf_fa(0.15f, aaf_fm(
                aaf_fs(L->r_frac, 2.0f), 0.028f)));
            L->vts_boost = fminf(1.5f, aaf_fa(1.1f, aaf_fm(
                aaf_fs(L->r_frac, 2.0f), 0.014f)));
            L->prg_scw = aaf_dm((double)g_frac, L->prs_scw);
            L->png_scw = aaf_dd(aaf_dm(L->prg_scw, (double)L->smo0),
                                (double)L->rs);
            L->vts = aaf_fm(aaf_fm(40.0f, thompson_aa_powf(L->xDs, 0.55f)),
                            thompson_aa_expf(-aaf_fm(100.0f, L->xDs)));
            float cri = -aaf_fd(aaf_fm(aaf_fm(1.0f, aaf_fm(L->mvd_c, 0.5e6f)),
                                       L->vts),
                                fminf(-0.1f, tempc));
            cri = fmaxf(0.1f, fminf(cri, 10.0f));
            L->const_Ri = cri;
            L->rime_dens = aaf_fm(aaf_fs(aaf_fa(0.051f, aaf_fm(0.114f, cri)),
                                         aaf_fm(aaf_fm(0.0055f, cri), cri)),
                                  1000.0f);
            if (L->rime_dens < 150.0f) {
                g_frac = 0.0f;
                L->prg_scw = 0.0;
                L->png_scw = 0.0;
            }
            L->g_frac = g_frac;
            L->prs_scw = aaf_dm((double)aaf_fs(1.0f, g_frac), L->prs_scw);
        }
    } else {
        // ---- :2781-2844, at or above 0 C ----------------------------------
        if (L->L_qs) {
            L->prr_sml = (double)aaf_fm(aaf_fs(aaf_fm(tempc, L->tcond),
                aaf_fm(aaf_fm(AAF_LVAP0, L->diffu), L->delQvs)),
                aaf_fa(aaf_fm(AAF_T1_QS_ME, L->smo1),
                       aaf_fm(aaf_fm(aaf_fm(AAF_T2_QS_ME, L->rhof2), L->vsc2),
                              L->smof)));
            if (L->prr_sml > 0.0) {
                L->prr_sml = aaf_da(L->prr_sml, aaf_dm(
                    (double)aaf_fm(4218.0f * AAF_OLFUS,
                                   aaf_fs(L->twet, AAF_T0)),
                    aaf_da(L->prr_rcs, L->prs_scw)));
            }
            L->prr_sml = fmin((double)aaf_fm(L->rs, odts),
                              fmax(0.0, L->prr_sml));
            if (L->prr_sml > 0.0) {
                L->pnr_sml = aaf_dm(aaf_dm((double)aaf_fd(L->smo0, L->rs),
                                           L->prr_sml),
                                    (double)thompson_aa_powf(10.0f, aaf_fm(
                                        -0.25f, aaf_fs(L->twet, AAF_T0))));
            } else if (L->ssati < 0.0f) {
                L->prs_sde = (double)aaf_fm(aaf_fm(aaf_fm(aaf_fm(aaf_fm(
                    0.15f, L->t1_subl), L->diffu), L->ssati), L->rvs),
                    aaf_fa(aaf_fm(AAF_T1_QS_SD, L->smo1),
                           aaf_fm(aaf_fm(aaf_fm(AAF_T2_QS_SD, L->rhof2),
                                         L->vsc2), L->smof)));
                L->prs_sde = fmax((double)aaf_fm(-L->rs, odts), L->prs_sde);
            }
        }
        if (L->L_qg) {
            double N0_melt = L->N0_g;
            if (aaf_fm(L->rg, L->ng) < 1.0e-4f) {
                const double lamg = aaf_dd(1.0, L->ilamg);
                N0_melt = aaf_dm((double)aaf_fm(aaf_fd(1.0e-4f, L->rg), 1.0f),
                                 thompson_aa_pow(lamg, 1.0));
            }
            const float t2_qg_me = aaf_fm(aaf_fm(aaf_fm(aaf_fm(aaf_fm(aaf_fm(
                aaf_fm(AAF_PI, 4.0f), 0.5f), AAF_OLFUS), 0.28f), AAF_SC3),
                sqrtf(AAF_AV_G5)), AAF_CGG11_5);
            L->prr_gml = aaf_dm(aaf_dm(
                (double)aaf_fs(aaf_fm(tempc, L->tcond),
                               aaf_fm(aaf_fm(AAF_LVAP0, L->diffu), L->delQvs)),
                N0_melt),
                aaf_da(aaf_dm((double)AAF_T1_QG_ME, thompson_aa_pow(
                           L->ilamg, (double)AAF_CGE10_1)),
                       aaf_dm((double)aaf_fm(aaf_fm(t2_qg_me, L->rhof2),
                                             L->vsc2),
                              thompson_aa_pow(L->ilamg,
                                              (double)AAF_CGE11_5))));
            L->prr_gml = fmin((double)aaf_fm(L->rg, odts),
                              fmax(0.0, L->prr_gml));
            if (L->prr_gml > 0.0) {
                L->pnr_gml = aaf_dm(aaf_dd(aaf_dm(L->prr_gml, (double)L->ng),
                                           (double)L->rg),
                                    (double)thompson_aa_powf(10.0f, aaf_fm(
                                        -0.33f, aaf_fs(L->twet, AAF_T0))));
            } else if (L->ssati < 0.0f) {
                const float t2_qg_sd = aaf_fm(aaf_fm(aaf_fm(0.28f, AAF_SC3),
                                                     sqrtf(AAF_AV_G5)),
                                              AAF_CGG11_5);
                L->prg_gde = aaf_dm(aaf_dm(
                    (double)aaf_fm(aaf_fm(aaf_fm(aaf_fm(0.5f, L->t1_subl),
                                                 L->diffu), L->ssati), L->rvs),
                    L->N0_g),
                    aaf_da(aaf_dm((double)AAF_T1_QG_SD, thompson_aa_pow(
                               L->ilamg, (double)AAF_CGE10_1)),
                           aaf_dm((double)aaf_fm(aaf_fm(t2_qg_sd, L->vsc2),
                                                 L->rhof2),
                                  thompson_aa_pow(L->ilamg,
                                                  (double)AAF_CGE11_5))));
                L->prg_gde = fmax((double)aaf_fm(-L->rg, odts), L->prg_gde);
                L->png_gde = aaf_dd(aaf_dm(L->prg_gde, (double)L->ng),
                                    (double)L->rg);
            }
        }
        if (dt > 120.0f) {
            L->prr_rcw = aaf_da(aaf_da(L->prr_rcw, L->prs_scw), L->prg_gcw);
            L->prs_scw = 0.0;
            L->prg_gcw = 0.0;
        }
    }
}

// :2856-2952, the conservation limiters for one level.  sump, rate_max and
// ratio are REAL (:1615): each DOUBLE sum is rounded once, compared and
// divided in float32, and the REAL ratio widens to rescale the DOUBLE rates.
__device__ __forceinline__ void thompson_aa_wrf_conserve(
    ThompsonAaLevel* L, float odts)
{
    float sump, rate_max, ratio;
    // :2862-2875
    sump = (float)aaf_da(aaf_da(aaf_da(aaf_da(aaf_da(L->pri_inu, L->pri_ide),
                                              L->prs_ide), L->prs_sde),
                                L->prg_gde), L->pri_iha);
    rate_max = aaf_fm(aaf_fm(aaf_fm(aaf_fs(L->qv, L->qvsi), L->rho), odts),
                      0.999f);
    if ((sump > AAF_EPS && sump > rate_max)
            || (sump < -AAF_EPS && sump < rate_max)) {
        ratio = aaf_fd(rate_max, sump);
        L->pri_inu = aaf_dm(L->pri_inu, (double)ratio);
        L->pri_ide = aaf_dm(L->pri_ide, (double)ratio);
        L->pni_ide = aaf_dm(L->pni_ide, (double)ratio);
        L->prs_ide = aaf_dm(L->prs_ide, (double)ratio);
        L->prs_sde = aaf_dm(L->prs_sde, (double)ratio);
        L->prg_gde = aaf_dm(L->prg_gde, (double)ratio);
        L->pri_iha = aaf_dm(L->pri_iha, (double)ratio);
    }
    // :2878-2889
    sump = (float)aaf_ds(aaf_ds(aaf_ds(aaf_ds(aaf_ds(-L->prr_wau, L->pri_wfz),
                                              L->prr_rcw), L->prs_scw),
                                L->prg_scw), L->prg_gcw);
    rate_max = aaf_fm(-L->rc, odts);
    if (sump < rate_max && L->L_qc) {
        ratio = aaf_fd(rate_max, sump);
        L->prr_wau = aaf_dm(L->prr_wau, (double)ratio);
        L->pri_wfz = aaf_dm(L->pri_wfz, (double)ratio);
        L->prr_rcw = aaf_dm(L->prr_rcw, (double)ratio);
        L->prs_scw = aaf_dm(L->prs_scw, (double)ratio);
        L->prg_scw = aaf_dm(L->prg_scw, (double)ratio);
        L->prg_gcw = aaf_dm(L->prg_gcw, (double)ratio);
    }
    // :2892-2901
    sump = (float)aaf_ds(aaf_ds(aaf_ds(L->pri_ide, L->prs_iau), L->prs_sci),
                         L->pri_rci);
    rate_max = aaf_fm(-L->ri, odts);
    if (sump < rate_max && L->L_qi) {
        ratio = aaf_fd(rate_max, sump);
        L->pri_ide = aaf_dm(L->pri_ide, (double)ratio);
        L->prs_iau = aaf_dm(L->prs_iau, (double)ratio);
        L->prs_sci = aaf_dm(L->prs_sci, (double)ratio);
        L->pri_rci = aaf_dm(L->pri_rci, (double)ratio);
    }
    // :2904-2914
    sump = (float)aaf_da(aaf_da(aaf_ds(aaf_ds(-L->prg_rfz, L->pri_rfz),
                                       L->prr_rci), L->prr_rcs), L->prr_rcg);
    rate_max = aaf_fm(-L->rr, odts);
    if (sump < rate_max && L->L_qr) {
        ratio = aaf_fd(rate_max, sump);
        L->prg_rfz = aaf_dm(L->prg_rfz, (double)ratio);
        L->pri_rfz = aaf_dm(L->pri_rfz, (double)ratio);
        L->prr_rci = aaf_dm(L->prr_rci, (double)ratio);
        L->prr_rcs = aaf_dm(L->prr_rcs, (double)ratio);
        L->prr_rcg = aaf_dm(L->prr_rcg, (double)ratio);
    }
    // :2917-2926
    sump = (float)aaf_da(aaf_ds(aaf_ds(L->prs_sde, L->prs_ihm), L->prr_sml),
                         L->prs_rcs);
    rate_max = aaf_fm(-L->rs, odts);
    if (sump < rate_max && L->L_qs) {
        ratio = aaf_fd(rate_max, sump);
        L->prs_sde = aaf_dm(L->prs_sde, (double)ratio);
        L->prs_ihm = aaf_dm(L->prs_ihm, (double)ratio);
        L->prr_sml = aaf_dm(L->prr_sml, (double)ratio);
        L->prs_rcs = aaf_dm(L->prs_rcs, (double)ratio);
    }
    // :2929-2938
    sump = (float)aaf_da(aaf_ds(aaf_ds(L->prg_gde, L->prg_ihm), L->prr_gml),
                         L->prg_rcg);
    rate_max = aaf_fm(-L->rg, odts);
    if (sump < rate_max && L->L_qg) {
        ratio = aaf_fd(rate_max, sump);
        L->prg_gde = aaf_dm(L->prg_gde, (double)ratio);
        L->prg_ihm = aaf_dm(L->prg_ihm, (double)ratio);
        L->prr_gml = aaf_dm(L->prr_gml, (double)ratio);
        L->prg_rcg = aaf_dm(L->prg_rcg, (double)ratio);
    }
    // :2942-2950.  ratio is REAL: MIN(ABS(),ABS()) of two DOUBLEs rounds
    // once, and ratio*SIGN(1.0, SNGL(x)) is a REAL product.
    L->pri_ihm = aaf_da(L->prs_ihm, L->prg_ihm);
    ratio = (float)fmin(fabs(L->prr_rcg), fabs(L->prg_rcg));
    L->prr_rcg = (double)aaf_fm(ratio, copysignf(1.0f, (float)L->prr_rcg));
    L->prg_rcg = -L->prr_rcg;
    if (L->twet > AAF_T0) {
        ratio = (float)fmin(fabs(L->prr_rcs), fabs(L->prs_rcs));
        L->prr_rcs = (double)aaf_fm(ratio,
                                    copysignf(1.0f, (float)L->prr_rcs));
        L->prs_rcs = -L->prr_rcs;
    }
}


// :2982 and :3164-3179, then :3189.  qvten and tten are REAL accumulators
// (:1668-1669 zero them): each source stage adds its DOUBLE right-hand side
// and rounds once.  Given both (the production adapter's v4.6.1 path), the
// vapour stays WRF's read-only qv1d, and the temperature becomes the TAU+1
// t1d + DT*tten; the condensation and the rain evaporation add their own
// terms to the same accumulators and re-form qv1d + DT*qvten and
// t1d + DT*tten as WRF does (:3189, :3479-3483, :3563-3566).  Null (the unit
// gates' in-place form): the level's own tendencies are applied in place.
__device__ __forceinline__ void thompson_aa_wrf_apply_vapor_heat(
    int idx, float t1d, float* __restrict__ qv,
    float* __restrict__ temperature, float* __restrict__ qvten,
    float* __restrict__ tten, double qv_rate, double t_rate, float dt)
{
    if (tten != nullptr) {
        qvten[idx] = (float)aaf_da((double)qvten[idx], qv_rate);
        tten[idx] = (float)aaf_da((double)tten[idx], t_rate);
        temperature[idx] = aaf_fa(t1d, aaf_fm(dt, tten[idx]));
    } else {
        temperature[idx] = aaf_fa(t1d, aaf_fm(dt, (float)t_rate));
        qv[idx] = aaf_fa(qv[idx], aaf_fm(dt, (float)qv_rate));
    }
}


// ---------------------------------------------------------------------------
// WRF's wet-bulb temperature twet (:2004-2013) and its Bolton (1980)
// helpers theta_e (:6032), t_lcl (:6066), t_dew (:6089), theta_wetb (:6111)
// and compT_fr_The (:6149), every REAL(4) operation pinned.  twet selects
// the melting branch of rain collecting snow and graupel and scales the
// collision-enhanced melting and the melted-drop number (:2488, :2526,
// :2787, :2793, :2825).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float aaf_theta_e(float pp, float tt, float w,
                                             float tlc)
{
    const float rr = aaf_fa(w, 1.0e-8f);
    const float power = aaf_fm(0.2854f, aaf_fs(1.0f, aaf_fm(0.28f, rr)));
    const float xx = aaf_fm(tt, thompson_aa_powf(aaf_fd(100000.0f, pp),
                                                 power));
    const float p1 = aaf_fs(aaf_fd(3.376f, tlc), 0.00254f);
    const float p2 = aaf_fm(aaf_fm(rr, 1000.0f),
                            aaf_fa(1.0f, aaf_fm(0.81f, rr)));
    return aaf_fm(xx, thompson_aa_expf(aaf_fm(p1, p2)));
}

__device__ __forceinline__ float aaf_t_lcl(float tt, float tttd)
{
    const float denom = aaf_fa(aaf_fd(1.0f, aaf_fs(tttd, 56.0f)),
                               aaf_fd(thompson_aa_logf(aaf_fd(tt, tttd)),
                                      800.0f));
    return aaf_fa(aaf_fd(1.0f, denom), 56.0f);
}

__device__ __forceinline__ float aaf_t_dew(float p, float w)
{
    const float rr = aaf_fa(w, 1.0e-8f);
    const float es = aaf_fd(aaf_fm(p, rr), aaf_fa(0.622f, rr));
    const float esln = thompson_aa_logf(es);
    return aaf_fd(aaf_fs(aaf_fm(35.86f, esln), 4947.2325f),
                  aaf_fs(esln, 23.6837f));
}

// c and d are real*8 arrays DATA-initialised from default-REAL literals,
// so each coefficient is the float32 literal widened; answer is REAL.
__device__ __forceinline__ float aaf_theta_wetb(float thetae)
{
    const double c[7] = {
        (double)-1.00922292e-10f, (double)-1.47945344e-8f,
        (double)-1.7303757e-6f, (double)-0.00012709f,
        (double)1.15849867e-6f, (double)-3.518296861e-9f,
        (double)3.5741522e-12f};
    const double d[7] = {
        (double)0.0f, (double)-3.5223513e-10f, (double)-5.7250807e-8f,
        (double)-5.83975422e-6f, (double)4.72445163e-8f,
        (double)-1.13402845e-10f, (double)8.729580402e-14f};
    const float x = fminf(475.0f, thetae);
    const double* k = x <= 335.5f ? c : d;
    const double xd = (double)x;
    const double a = aaf_da(k[0], aaf_dm(xd, aaf_da(k[1], aaf_dm(xd,
        aaf_da(k[2], aaf_dm(xd, aaf_da(k[3], aaf_dm(xd, aaf_da(k[4],
        aaf_dm(xd, aaf_da(k[5], aaf_dm(xd, k[6]))))))))))));
    return aaf_fa((float)a, 273.15f);
}

__device__ __forceinline__ float aaf_compT_fr_The(float thelcl, float pres)
{
    float guess = aaf_fm(aaf_fs(thelcl, aaf_fm(0.5f, thompson_aa_powf(
        fmaxf(aaf_fs(thelcl, 270.0f), 0.0f), 1.05f))),
        thompson_aa_powf(aaf_fd(pres, 100000.0f), 0.2f));
    for (int iter = 1; iter <= 100; ++iter) {
        const float w1 = thompson_rslf(pres, guess);
        const float w2 = thompson_rslf(pres, aaf_fa(guess, 1.0f));
        const float tenu = aaf_theta_e(pres, guess, w1, guess);
        const float tenup = aaf_theta_e(pres, aaf_fa(guess, 1.0f), w2,
                                        aaf_fa(guess, 1.0f));
        // WRF divides unguarded; a zero denominator there is garbage, not a
        // result, and WOOF leaves the iteration instead (never measured to
        // occur on the column oracle).
        const float den = aaf_fs(tenup, tenu);
        if (den == 0.0f) break;
        const float cor = aaf_fd(aaf_fs(thelcl, tenu), den);
        guess = aaf_fa(guess, cor);
        if (cor < 0.01f && -cor < 0.01f) return guess;
    }
    return aaf_fm(aaf_theta_wetb(thelcl),
                  thompson_aa_powf(aaf_fd(pres, 100000.0f), 0.286f));
}

// :2006-2011 for a level at or below k_melting.  satw = qv/qvs with qv the
// floored working vapour (:1800, :1984).
__device__ __forceinline__ float thompson_aa_wrf_twet(float pres, float temp,
                                                      float qv)
{
    const float satw = aaf_fd(qv, thompson_rslf(pres, temp));
    if (!(satw < 0.999f)) return temp;
    const float dew_t = fminf(aaf_fs(temp, 0.001f), aaf_t_dew(pres, qv));
    const float Tlcl = aaf_t_lcl(temp, dew_t);
    const float The = aaf_theta_e(pres, temp, qv, Tlcl);
    return fminf(temp, aaf_compT_fr_The(The, pres));
}


// ---------------------------------------------------------------------------
// Snow and classic-graupel constants, the air density and the classic
// graupel number, shared by the v4.6.1 snow and graupel fallout
// (thompson_aerosol_sed.cu), the graupel-number entry and exit
// (thompson_aerosol_state.cu) and the reflectivity.  The REAL(4) gamma
// moments were read back from the oracle build's thompson_init, which forms
// them as WGAMMA(cse(k)) at runtime (:727-776); two of them are not the
// integers they approximate: crg(4) = cgg(4,1) = 720.000061, not 720.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float thompson_aa_air_density(
    float pressure, float temperature, float qv)
{
    // 0.622*pres(k)/(R*temp(k)*(qv(k)+0.622)), qv already floored.
    return thompson_aa_div(thompson_aa_mul(0.622f, pressure),
        thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_R_DRY, temperature),
                        thompson_aa_add(qv, 0.622f)));
}

// WRF's RHO_NOT, 101325./(287.05*298.0) folded in REAL(4).
#define THOMPSON_AA_RHO_NOT 1.18452108f

// Snow (:113-117, :146-148, :727-749): csg(k) = WGAMMA(cse(k)) as REAL(4),
// read back from the oracle build's thompson_init.
#define THOMPSON_AA_CSE1  3.0f
#define THOMPSON_AA_CSG1  2.0f
#define THOMPSON_AA_CSE4  3.54999995f
#define THOMPSON_AA_CSG4  3.51325202f
#define THOMPSON_AA_CSE7  3.63569999f
#define THOMPSON_AA_CSG7  3.87160635f
#define THOMPSON_AA_CSE10 4.18569994f
#define THOMPSON_AA_CSG10 7.61279917f
#define THOMPSON_AA_FV_S  100.0f

// Classic graupel, idx_bg1 = 5 (rho_g = 400): am_g(5), cgg(k,5) and the
// cge/ogg family, as thompson_init leaves them in REAL(4).
#define THOMPSON_AA_AM_G5   209.439514f
#define THOMPSON_AA_CGG1    6.0f
#define THOMPSON_AA_CGG2    1.0f
#define THOMPSON_AA_CGG3    6.0f
#define THOMPSON_AA_CGG4    720.000061f
#define THOMPSON_AA_CGE4    7.0f
#define THOMPSON_AA_CGG6    20.3632278f
#define THOMPSON_AA_CGG7    2.94954014f
#define THOMPSON_AA_CGG12   1.32934034f
#define THOMPSON_AA_OGG1    0.166666672f
#define THOMPSON_AA_OGG2    1.0f
#define THOMPSON_AA_OGG3    0.166666672f
#define THOMPSON_AA_OGE1    0.25f
#define THOMPSON_AA_OBMG    0.333333343f
#define THOMPSON_AA_BM_G    3.0f

// :3289-3297, the classic (not hail-aware) graupel number WRF diagnoses
// from the working content rg(k) for the fallout; ng1d is not read.  N0_exp,
// lam_exp and lamg are DOUBLE PRECISION, ygra1 and zans1 REAL.
__device__ __forceinline__ float thompson_aa_classic_graupel_number_m3(
    float rg)
{
    const float ygra1 = thompson_aa_log10f(fmaxf(1.0e-9f, rg));
    float zans1 = thompson_aa_add(3.0f, thompson_aa_mul(
        2.0f / 7.0f, thompson_aa_add(ygra1, 8.0f)));
    zans1 = fmaxf(2.0f, fminf(zans1, 6.0f));
    const double n0_exp = (double)thompson_aa_powf(10.0f, zans1);
    const double lam_exp = thompson_aa_pow(__ddiv_rn(__dmul_rn(__dmul_rn(
        n0_exp, (double)THOMPSON_AA_AM_G5), (double)THOMPSON_AA_CGG1),
        (double)rg), (double)THOMPSON_AA_OGE1);
    const double lamg = __dmul_rn(lam_exp, (double)thompson_aa_powf(
        thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_CGG3, THOMPSON_AA_OGG2),
                        THOMPSON_AA_OGG1), THOMPSON_AA_OBMG));
    return (float)__ddiv_rn(__dmul_rn((double)thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_CGG2, THOMPSON_AA_OGG3), rg),
        thompson_aa_pow(lamg, (double)THOMPSON_AA_BM_G)),
        (double)THOMPSON_AA_AM_G5);
}
