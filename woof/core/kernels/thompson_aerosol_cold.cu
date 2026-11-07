#if defined(THOMPSON_AA_HAIL)
#define pow thompson_aa_pow
#define exp thompson_aa_exp
#define log thompson_aa_log
#define log10 thompson_aa_log10
#define powf thompson_aa_powf
#define expf thompson_aa_expf
#define logf thompson_aa_logf
#define log10f thompson_aa_log10f
#endif
// gpuwm/core/kernels/thompson_aerosol_cold.cu
//
// WP-06 -- aerosol-aware (mp_physics=28) COLD network for WRF v4.6.1
// Thompson.  One kernel, thompson_aa_cold_network, reproducing the whole
// sub-freezing source group of module_mp_thompson.F:2239-3190 with
// is_aerosol_aware = .TRUE., dustyIce = .TRUE., homogIce = .TRUE.,
// is_hail_aware = .FALSE., wif_input_opt = 0.
//
// Numerical authority: WRF v4.6.1 phys/module_mp_thompson.F,
// commit d66e442fccc04111067e29274c9f9eaccc3cef28.  Every bare line number
// below refers to that file.  Structural authority for everything that is
// NOT aerosol-dependent: gpuwm/core/kernels/thompson.cu's
// thompson_frozen_vapor_cloud_network (:6514-7586), which is byte-frozen and
// is transcribed -- not shared -- because cupy.RawModule has no #include
// path.  Shared device helpers come from thompson_aerosol_common.cuh, which
// gpuwm/core/kernels/__init__.py prepends to this translation unit.
//
// ---------------------------------------------------------------------------
// THE v4.6.1 BODY IS thompson_aa_cold_level_wrf461 (2026-10-07)
// ---------------------------------------------------------------------------
// For the v4.6.1 generation the kernel below hands every sub-freezing level
// to thompson_aa_cold_level_wrf461, which runs WRF's own statements in
// WRF's own arithmetic order through the ThompsonAaLevel transcription in
// thompson_aerosol_common.cuh (entry state, warm-rain loop, frozen-species
// block, conservation limiters, tendencies).  Measured on the host CPU
// against WRF v4.6.1 over the column oracle's 153 columns: every one of the
// 60 rates the source networks own is bit-identical on every level below
// 0 C, except where rain meets graupel, where WRF reads its rain-graupel
// tables out of bounds and WOOF reads the slab they hold (aaf_racg_index;
// a declared divergence).  The kernel body after the dispatch, and the
// measurements quoted in the rest of this header, now describe the WRF 3.9
// fork generation (THOMPSON_AA_WRF39) only.
//
// ---------------------------------------------------------------------------
// WHAT CHANGES RELATIVE TO mp=8, AND WHY EACH ONE MATTERS
// ---------------------------------------------------------------------------
// 1. nc is PROGNOSTIC.  thompson.cu freezes the droplet number at
//    Nt_c = 100e6 m^-3 (`const float cloud_number = 100.0e6f`, :6928).  Here
//    it is rediagnosed per cell from the entry nc array through
//    thompson_aa_cloud_dist (:1826-1842), and every consumer -- the Bigg
//    freezing number cap, the droplet-bin table index, the mean volume
//    diameter, and the five cloud-number sinks -- reads that value.
//
// 1a. THE nu_c STAGING.  WRF computes nu_c at :1832 from the PRE-rediagnosis
//    nc and AGAIN at :2170 from the POST-rediagnosis nc(k) of :1840.  Only
//    the second is the working shape parameter; it feeds lamc (:2173),
//    mvd_c (:2174), Dc_g (:2181) and pnr_wau (:2191).  This kernel used to
//    take nu_c from thompson_aa_cloud_dist's ENTRY out-parameter, which is
//    the first one.  Whenever the :1834-1838 droplet-size clamp engages the
//    two differ -- measured 689 of 5400 states on a (nc, rc) grid at rho = 1
//    -- and between nu_c = 3 and nu_c = 15 the gamma products the rates are
//    built from move by 40.8x (lamc) and 15.8x (Dc_g) while pnr_wau, being
//    linear in nu_c, rescales by 5.  See thompson_aerosol_common.cuh's
//    "nu_c STAGING RULE" block.
//
//    OBSERVABILITY, MEASURED, because it explains why the committed fixtures
//    could not have caught this: in EVERY divergent state mvd_c lands on a
//    clamp bound.  That is structural, not lucky -- if the entry clamp
//    engaged then lamc = cce(2,nu)/D0c or cce(2,nu)/(2*D0r), so
//    mvd_c = (3.672+nu)/(4+nu) times D0c or 2*D0r, which is below D0c in the
//    first case and above D0r in the second for every nu in [2,15].  The
//    only observables of the staging are therefore Dc_g and pnr_wau (and
//    pnc_wau through prr_wau), all gated on rc > 0.01e-3, and the lowest
//    divergent rc that clears that gate is 3.4365e-2 kg m^-3 (re-measured in
//    wave 4 over a 60 x 80 (nc, qc) sweep: 1259 of 4800 states diverge and
//    EVERY one of them has prr_wau > 0).  No committed fixture goes there.
//
//    WAVE 4 CLOSES THE REGRESSION GAP THAT LEFT.  Wave 3's gate was a PROBE
//    comparison, and the probe/production agreement test compared ncten --
//    which on every divergent state is pinned to the nc*odts cap at :2192
//    and therefore cannot tell the two stages apart.  So nothing failed if
//    the PRODUCTION kernel regressed.  It does now:
//    test_production_kernel_uses_the_working_stage_nu_c drives THIS kernel
//    into that regime with no rain, ice, snow or graupel and compares its
//    own qr/nr/ncten against WRF's :2168-2193.  nr is the discriminator,
//    because prr_wau is pinned to rc*odts by the MIN at :2190 across the
//    whole regime while pnr_wau = prr_wau/(am_r*nu_c*10*D0r**3) stays
//    LINEAR in nu_c.  MEASURED with the working stage reverted to the entry
//    stage: qr and ncten do not move at all, and nr is wrong by 25.0% to
//    80.0% on all 36 divergent rows.
//
// 2. idx_IN IS LIVE.  thompson.cu hardcodes `const int nuclei_bin = 27` at
//    :6820 (rain freezing) and :7008 (cloud freezing), which is WRF's
//    non-aerosol default of 1 ice nucleus per litre.  Under mp=28 the index
//    is thompson_aa_in_bin(iceDeMott(...)) (:2579-2591).  This is the FIRST
//    time gpuwm reads any freezeH2O slice other than 27; 54 of the 55 IN
//    slices have never been exercised by any previous test.
//
// 3. idx_n IS LIVE.  thompson.cu hardcodes `const int cloud_number_bin = 65`
//    at :7005.  Here it is thompson_aa_droplet_bin(nc_work) (:3447-3448).
//    thompson_aa_droplet_bin(100.0e6f) == 65 is a hard identity gate in
//    tests/test_thompson_aerosol_device_helpers.py, so at nc = Nt_c this
//    reduces to what mp=8 froze.
//
// 4. iceDeMott REPLACES Cooper, it does not supplement it.  WRF:2622-2626
//    is an if/else on `dustyIce .AND. is_aerosol_aware`; the Cooper
//    expression MIN(250.E3, TNO*EXP(ATO*(T_0-temp))) that thompson.cu
//    evaluates at :6645-6646 and :7189-7200 is unreachable in mp=28 and
//    appears NOWHERE in this file.  Both of WRF's iceDeMott call sites
//    (:2574 and :2623) differ only in dead formal arguments and return the
//    identical value, so it is evaluated ONCE per cell here.
//
// 5. KOOP HOMOGENEOUS HAZE FREEZING (:2633-2643) is entirely new; classic
//    Thompson has no counterpart.  Its gate needs ns(k), WRF's EXPLICIT
//    two-gamma snow-number integral at :2081-2088 -- NOT smo0, and not
//    gpuwm's mp=8 snow number closure.  ns(k)'s ONLY use in the entire
//    routine is this gate.  pri_iha joins the shared frozen-vapor limiter
//    sum (:2862-2876); pni_iha is deliberately NOT rescaled with it, the
//    same held-number quirk WRF already applies to pni_inu.
//
// 6. FIVE CLOUD-NUMBER SINKS feed ncten: pni_wfz (Bigg immersion freezing),
//    pnc_scw (snow collecting cloud water, :2410-2411), pnc_gcw (graupel
//    collecting cloud water, :2435-2437), pnc_wau (autoconversion,
//    :2192-2193) and pnc_rcw (rain collecting cloud water, :2205-2207).
//    Each collection term is MIN'd against nc*odts individually.
//
// 7. SIX AEROSOL SCAVENGING RATES feed nwfaten/nifaten: pna_sca and pnd_scd
//    (snow, :2444-2450), pna_gca and pnd_gcd (graupel, :2461-2471) and
//    pna_rca and pnd_rcd (rain, :2212-2221), with Eff_aero collection
//    diameters 0.04 um for water-friendly aerosol and 0.8 um for
//    ice-friendly aerosol.  WRF caps each against the FULL available aerosol
//    and deliberately lets the sum overshoot, relying on the terminal floors
//    at :3979-3981; do NOT add a shared limiter across collectors.
//
// 8. TWO AEROSOL SINKS FROM NUCLEATION: nifaten -= pni_inu*orho
//    (:2974-2976, because dustyIce is a PARAMETER .true.) and
//    nwfaten -= pni_iha*orho (:2964-2965 -- Koop consumes WATER-friendly
//    aerosol, not ice nuclei).  CRITICAL NEGATIVE FINDING: Bigg immersion
//    freezing of cloud droplets and raindrops consumes NO aerosol at all.
//    nifa influences it only by shifting the freezing table's temperature
//    index through T_adjust = MAX(-3, MIN(3-LOG10(Nt_IN(m)), 3)) at :4701,
//    i.e. through idx_IN.  Adding a nifa sink for pni_wfz or pni_rfz
//    because it "seems physical" is not in WRF and would deplete ice nuclei
//    roughly an order of magnitude too fast in any glaciating cloud.
//
// ---------------------------------------------------------------------------
// ACCUMULATOR CONTRACT
// ---------------------------------------------------------------------------
// nc/nwfa/nifa state are READ-ONLY entry state for the whole mp=28 call.
// This kernel writes ONLY into the three per-kilogram scratch accumulators
// ncten / nwfaten / nifaten (which the adapter zeroes at entry) with `+=`,
// and a terminal WP-04 kernel applies them once with WRF's clamps
// (:3972-4021).
//
// WHY THE ACCUMULATORS ARE NOT READ BACK HERE.  At the point in WRF where
// this network runs, nwfaten/nifaten/ncten are still identically zero: they
// are first assigned in the tendency loop at :2964-3009, which is AFTER
// every process rate in :2239-2850 has been diagnosed.  The working
// per-m^3 values WRF uses inside the cold network are therefore the ENTRY
// clamps at :1805-1806 alone, and nc(k) is the entry rediagnosis at
// :1826-1842 -- not (nc1d + ncten*dt).  Reading the accumulators here would
// make the answer depend on whether the adapter happens to launch the warm
// network first, which WRF's single-pass structure never does.
//
// ---------------------------------------------------------------------------
// THE TEMPERATURE SPLIT, AND WHY IT IS EXHAUSTIVE
// ---------------------------------------------------------------------------
// iiwarm is a PARAMETER .false. (:59) and mu_g/idx_bg are fixed, so WRF runs
// ONE column loop with two temperature branches inside it.  Only the block
// opened at :2554 is guarded by `if (temp(k).lt.T_0)`.  Everything at
// :2156-2234 (the warm-rain loop) and everything at :2402-2478 (snow/graupel
// collection of cloud water and of aerosol) runs at EVERY level.
//
// gpuwm splits that one loop across two kernels by ENTRY temperature:
//   * this kernel returns at once for temperature >= 273.15 f, and it runs
//     BEFORE any kernel has moved the temperature field, so its gate sees
//     the entry column;
//   * thompson_aerosol_warm.cu returns unless its held entry-temperature
//     mask (cp.greater_equal(temperature, 273.15), seeded before the first
//     source launch) is set.
// The two masks are exact complements of one another over every cell, with
// 273.15 K itself owned by the warm side.  Each of the ten always-run rates
// therefore appears in BOTH translation units and is evaluated EXACTLY ONCE
// per cell.  A gap or an overlap at 273.15 K is the obvious way this goes
// wrong; tests/test_thompson_aerosol_cold_gpu.py pins it at 273.15 and at
// both bracketing floats.
//
// HISTORICAL NOTE, kept because the reasoning was wrong in an instructive
// way.  This file originally omitted pnc_wau, pnc_rcw, pna_rca and pnd_rcd
// on the grounds that they were "WP-07's by the port's exclusive
// file-ownership rule".  File ownership is about who may EDIT a file, not
// about which cells a kernel is responsible for.  WP-07's kernel does own
// those four rates -- for warm cells.  Omitting them here did not delegate
// them, it deleted them for the entire sub-freezing half of the domain,
// while this kernel went on computing their MASS partners prr_wau and
// prr_rcw.  Cloud mass left without its droplet number and rain wet
// scavenging of CCN/IN was absent below freezing.
//
// EVERY level of all six committed aerosol fixtures is sub-freezing (230,
// 240 or 260 K), so WP-07's kernel returns immediately on all of them: the
// four rates reached nobody at all, and the wave-2 reference table had been
// assembled with them subtracted out, so the suite agreed with the defect.
//
// ---------------------------------------------------------------------------
// MEASURED AGREEMENT WITH WRF v4.6.1 (RTX 5090, CuPy 14.1.1, nvrtc c++17)
// ---------------------------------------------------------------------------
// A. Six committed column fixtures, gpuwm/data/thompson/oracle-aero/, max
//    relative difference over levels carrying signal:
//      aero-scav-frozen      nwfa, nifa                        BIT-EXACT
//      aero-ice-demott-dep   nifa                              BIT-EXACT
//      aero-ice-koop         nifa BIT-EXACT, nwfa              1.78e-07
//      aero-ice-demott-idxin qi 1.20e-07 ni 6.60e-08 ncten 3.96e-08
//                            nwfaten 4.22e-08 nifaten 5.24e-08
//      aero-cloud-freeze-nc  qi 2.01e-08 ni 4.03e-08 ncten 4.33e-08
//      aero-cold-overlap     qi 9.95e-08 ni 1.52e-07 ncten 3.43e-07
//                            nwfaten 2.55e-07 nifaten 5.16e-07
//    aero-ice-demott-idxin is the acceptance fixture: it is the only fixture
//    in the port that pins an idx_IN other than 27, i.e. an entire freezeH2O
//    axis gpuwm had never read.
//
// B. 11340 sub-freezing states from tools/thompson_wrf461_oracle/
//    probe_cold_warm_loop_aero.F90 (COMMITTED; build_aero_probes.sh rebuilds
//    it and it reproduces its pinned SHA-256), which links the same compiled
//    module_mp_thompson.o and evaluates :1826-1842 and :2144-2232 verbatim:
//      nu_c (both stages), nc_m3, mvd_c, mvd_r                 EXACT
//      prr_wau, pnr_wau, pnc_wau                               EXACT
//      pnc_rcw, pna_rca, pnd_rcd                   <= 7.6e-16 (CSV round trip)
//    THE LAST FIVE USED TO BE THE UPSTREAM RESIDUAL AND ARE NOW GONE.  This
//    block used to read
//      mvd_c 1.23e-07, nc_m3 3.20e-07, pnc_rcw 3.25e-07
//      pnc_wau 1.97e-06, prr_wau / pnr_wau 2.31e-06
//    and attributed the last line, correctly, to thompson_aa_cloud_dist's
//    CUDA powf in thompson_aerosol_common.cuh, amplified by the Dc_b
//    cancellation at :2182 on the rc = 0.15 kg m^-3 end of the ladder.
//    Contraction-pinning everything this file owns had already taken prr_wau
//    from 1.03e-05 to 2.31e-06; making the shared droplet diagnosis
//    correctly-rounded AND contraction-pinned took the remaining 2.31e-06 to
//    zero.  Nothing in this file changed for it.
//
// C. WAVE 4 -- AGAINST WRF'S OWN POST-SOURCE TENDENCIES, not against a
//    reduction of them.  A scratch build of the WP-06 oracle, outside the
//    tree, compiles a COPY of module_mp_thompson.F carrying only added
//    `write` statements (plus three added PUBLIC probe subroutines whose
//    bodies are verbatim transcriptions of statements already in the file),
//    links the same stub_wrf.o / module_mp_radar.o / run_column_aero.o with
//    the same flags and the same four assets, and REPRODUCES ALL 38
//    COMMITTED aerosol fixture CSVs BYTE FOR BYTE.  Its instrumentation
//    publishes qiten/niten/qrten/nrten as WRF has them immediately after the
//    :3021-3092 tendency loop -- i.e. exactly what this kernel is
//    responsible for, before condensation and before any fallout.  Driving
//    this kernel on each fixture's OWN entry column, max relative
//    difference over the levels carrying signal:
//      aero-ice-koop          qi 1.152e-07   ni 8.482e-08
//      aero-cold-overlap      qi 9.950e-08   ni 1.515e-07   qr 8.795e-08
//      aero-ice-demott-idxin  qi 1.203e-07   ni 6.599e-08   qr 1.304e-08
//      aero-ice-demott-dep    qi 1.209e-07   ni 5.907e-08
//      aero-cloud-freeze-nc   qi 2.013e-08   ni 4.026e-08   qr 4.290e-08
//    (nr agrees EXACTLY wherever it moves at all; the apparent 4e-4 in a
//    finite-difference reconstruction of nrten is the reconstruction's own
//    cancellation, not a state difference.)
//
//    WHAT THAT SETTLES.  Two adapter-level G3 fixtures were open against
//    this package.  Neither is this kernel:
//
//    * aero-ice-koop, qi 1.612e-03 / ni 1.764e-03.  The whole residual is
//      ONE 2^-24 quantum of iceKoop's prob_h.  :5539 forms
//      prob_h = 1. - exp(-J*V*DT) with the exponential just below 1, so the
//      subtraction is exact and prob_h can only be an integer multiple of
//      2^-24; at this fixture's level 15 the integer is 561, so one step of
//      that grid is 1/561 = 1.78e-3 of the answer.  log_J_rate is a cubic in
//      delta_aw whose four terms (-906.7, +2.63e3, -2.59e3, +8.7e2) sum to
//      about 10.7, giving d(log10 J)/d(satw) ~ 222, so ONE float32 ulp of
//      pres (9.7e-8 relative) is enough to cross a quantum boundary.
//      tests/test_thompson_aerosol_adapter.py's entry-state reconstruction
//      perturbs p by exactly +1 ulp at that level in order to find a float32
//      theta reproducing temp_k exactly, and that costs 595.97 of 334348.6
//      crystals m^-3 = 1.7825e-3.  MEASURED, and NOT repairable inside the
//      harness: over +-256 ulps of p there is NO pressure at that level that
//      both admits an exact theta and leaves RSLF/RSIF bit-identical, and
//      driving the adapter with the exact p (accepting a 1-ulp temperature
//      instead) makes it WORSE, 3.08e-03.  Gated by
//      test_ice_koop_is_quantised_and_one_pressure_ulp_moves_it_one_quantum.
//
//    * aero-cold-overlap, the terminal droplet number.  Cloud water is
//      consumed to WRF's own R1 floor at these levels (which this port
//      reproduces BIT-EXACTLY), so :4019's rediagnosis collapses onto the
//      MAX(2./rho(k), ...) floor at :3976 and the whole residual is the
//      choice of rho.  WRF's rho(k) in that loop is the TAU+1 density -- the
//      unconditional :3193 refresh, then per level :3490 and :3572 -- and
//      thompson_aa_state_finalize is handed the ENTRY density of :1802.
//      RE-MEASURED on the regenerated fixtures with an instrumented pristine
//      WRF that reproduces all 22 committed column CSVs byte for byte: the
//      two densities differ by up to 7.4672e-03 (this fixture), and
//      recomputing :3976-4021 INSIDE WRF with the entry density substituted
//      moves nc1d at exactly one of 504 fixture levels -- aero-cold-overlap
//      k=5, 1.833336115 -> 1.826136470 kg^-1, 3.9271e-03 relative.
//      (It used to be quoted here as level 7 / 5.73e-05; that level's qc no
//      longer lands on the floor after the harness repaired its own pii.)
//      That is WP-04's kernel and WP-09's call site, not this one; the fix
//      is one added launch of thompson_aa_tau1_density immediately after
//      rain evaporation, which reproduces WRF's terminal rho BITWISE at 501
//      of those 504 levels.  Reported as an integration request.
//
// D. WAVE 4 -- ns(k), the least Fortran-verified helper in the shared
//    header, now gated against compiled WRF.  thompson_aa_snow_number is
//    BIT-EXACT against module_mp_thompson.F:2081-2088 on 3719 of 3721
//    states of a 61-temperature (273.05 to 201.05 K) x 61-snow-content
//    (1e-12 to 1e-2 kg m^-3) sweep run through the module's own
//    csg(15)/cse(15), the other 2 being ONE float32 ulp (6.924e-08) and
//    reproducing identically when the helper is rebuilt fully
//    contraction-pinned with correctly-rounded powers, i.e. not repairable.
//    The COMPOSITE path this kernel actually uses
//    (rho -> smob -> thompson_field_a/b -> smoc -> ns) is BIT-EXACT against
//    WRF's own ns(k) at all 16 snow-bearing levels of the committed
//    fixtures.  ns is NOT interchangeable with smo0: over that sweep
//    ns/smo0 runs from 0.83 to 1226, and
//    test_two_gamma_snow_number_and_not_smo0_decides_the_koop_gate makes the
//    difference PRODUCTION-observable by putting two cells either side of
//    the 999e3 gate where the two closures disagree.


// ---------------------------------------------------------------------------
// Helpers this translation unit consumes from thompson_aerosol_common.cuh.
// thompson.cu is byte-frozen and there is no #include path under
// cupy.RawModule, so anything shared with the other four aerosol kernels
// lives ONCE in the header the loader prepends, never as a local duplicate.
// ---------------------------------------------------------------------------

// thompson_aa_decade_index_double (thompson.cu:3084-3105, the DOUBLE-precision
// base-ten mantissa/decade bin used by the rain and graupel y-intercept
// lookups) USED TO BE DUPLICATED HERE, byte-identically with warm.cu's copy.
// It now lives in thompson_aerosol_common.cuh; see that file's PUBLISHED
// SHARED SIGNATURES block.  Do not reintroduce a local copy -- the shared
// signature is identical, so nvrtc rejects the survivor outright with
// "function has already been defined", which is exactly the diagnostic this
// consolidation exists to guarantee.
//
// thompson_aa_bound_rain_number, thompson_aa_bound_ice_number and
// thompson_aa_entry_rain_distribution USED TO BE DUPLICATED HERE with bodies
// that had already drifted from warm.cu's copies (no rain_intercept_n0, no
// contraction pinning, no unconditional :2146-2150 lamr re-derivation).
// Separate cupy.RawModule translation units mean nvrtc never sees such a
// conflict, so the drift is silent.  All three now live ONCE in
// thompson_aerosol_common.cuh; see its PUBLISHED SHARED SIGNATURES block.
// Do not reintroduce a local copy -- for the two `void` helpers that is a
// hard nvrtc redefinition error, but for the rain distribution the shared
// form takes SEVEN parameters and a six-parameter local copy would quietly
// OVERLOAD rather than collide.  That is exactly the failure mode this
// consolidation exists to remove.


// ---------------------------------------------------------------------------
// ONE LEVEL BELOW 0 C, WRF v4.6.1, IN WRF'S ARITHMETIC ORDER.
// ---------------------------------------------------------------------------
//
// The entry state (:1798-1951), the per-level thermodynamics and moments
// (:1971-2151), the warm-rain loop (:2157-2234), the frozen-species block
// (:2242-2848), the conservation limiters (:2856-2952) and the tendencies
// (:2958-3183) for a level whose ENTRY temperature is below 273.15 K, through
// the shared transcription in thompson_aerosol_common.cuh
// (ThompsonAaLevel).  See that block's head for the defects this replaced.
//
// What this writes, and in what form, is the production contract the
// adapter (gpuwm/core/microphysics_aerosol.py) relies on: the per-kilogram
// accumulators ncten/nwfaten/nifaten and, on the v4.6.1 path, qcten, qrten,
// nrten, qiten and niten receive WRF's tendencies (WRF's REAL accumulator
// plus a DOUBLE right-hand side, rounded once); qs, qg, the graupel-number
// shadow (ng1d), qv and the temperature are left at WRF's TAU+1 values
// X1d + Xten*DT (:3189-3192, :3258, :3270); vts_boost is WRF's.
__device__ __forceinline__ void thompson_aa_cold_level_wrf461(
    int idx,
    float* __restrict__ qi, float* __restrict__ ni, float* __restrict__ qs,
    float* __restrict__ qg, float* __restrict__ qr, float* __restrict__ nr,
    float* __restrict__ qc, float* __restrict__ temperature,
    const float* __restrict__ pressure, float* __restrict__ qv,
    const float* __restrict__ nc_entry, const float* __restrict__ nwfa_entry,
    const float* __restrict__ nifa_entry,
    float* __restrict__ ncten, float* __restrict__ nwfaten,
    float* __restrict__ nifaten,
    float* __restrict__ graupel_number_shadow,
    float* __restrict__ snow_velocity_boost,
    const ThompsonAaFrozenTables* T,
    const double* __restrict__ t_Efrw,
    float* __restrict__ qcten, float* __restrict__ qrten,
    float* __restrict__ nrten, float* __restrict__ qiten,
    float* __restrict__ niten, float* __restrict__ qvten_acc,
    float* __restrict__ tten_acc, float* __restrict__ qsten_acc,
    float* __restrict__ qgten_acc, float* __restrict__ ngten_acc, float dt)
{
    const bool accumulate_rain_ice = niten != nullptr;
    const float odts = aaf_fd(1.0f, dt);

    // The entry state as WRF zeroes it (:1844-1945): a mixing ratio at or
    // below R1 is 0 from here on, with its number.
    const float qc1d = qc[idx] > AAF_R1 ? qc[idx] : 0.0f;
    const float qi1d = qi[idx] > AAF_R1 ? qi[idx] : 0.0f;
    const float ni1d = qi[idx] > AAF_R1 ? ni[idx] : 0.0f;
    const float qr1d = qr[idx] > AAF_R1 ? qr[idx] : 0.0f;
    const float nr1d = qr[idx] > AAF_R1 ? nr[idx] : 0.0f;
    const float qs1d = qs[idx] > AAF_R1 ? qs[idx] : 0.0f;
    const float qg1d = qg[idx] > AAF_R1 ? qg[idx] : 0.0f;
    const float ng1d = qg[idx] > AAF_R1 ? graupel_number_shadow[idx] : 0.0f;

    ThompsonAaLevel L;
    thompson_aa_wrf_zero_rates(&L);
    thompson_aa_wrf_entry(&L, temperature[idx], pressure[idx], qv[idx],
                          qc[idx], nc_entry[idx], qi[idx], ni[idx], qr[idx],
                          nr[idx], qs[idx], qg[idx],
                          graupel_number_shadow[idx], nwfa_entry[idx],
                          nifa_entry[idx]);
    // Below 0 C twet is only ever compared with T_0, and twet <= temp.
    thompson_aa_wrf_level_state(&L, L.temp);
    thompson_aa_wrf_warm_loop(&L, odts, t_Efrw);
    thompson_aa_wrf_frozen_block(&L, dt, odts, T);
    snow_velocity_boost[idx] = L.vts_boost;
    // HOST READBACK ANCHOR: the frozen block is complete, pre-limiter.
    thompson_aa_wrf_conserve(&L, odts);
    // HOST READBACK ANCHOR: every rate final.

    const float orho = L.orho;
    const float lfus2 = aaf_fs(AAF_LSUB, L.lvap);
    const double dorho = (double)orho;

    // :2964-2978.
    nwfaten[idx] = (float)aaf_ds((double)nwfaten[idx], aaf_dm(aaf_da(aaf_da(
        aaf_da(L.pna_rca, L.pna_sca), L.pna_gca), L.pni_iha), dorho));
    nifaten[idx] = (float)aaf_ds((double)nifaten[idx], aaf_dm(aaf_da(aaf_da(
        L.pnd_rcd, L.pnd_scd), L.pnd_gcd), dorho));
    nifaten[idx] = (float)aaf_ds((double)nifaten[idx],
                                 aaf_dm(L.pni_inu, dorho));
    // :2982-2984.
    const double qv_rate = aaf_dm(aaf_ds(aaf_ds(aaf_ds(aaf_ds(aaf_ds(
        -L.pri_inu, L.pri_iha), L.pri_ide), L.prs_ide), L.prs_sde),
        L.prg_gde), dorho);
    // :2987-2995.
    const double qc_rate = aaf_dm(aaf_ds(aaf_ds(aaf_ds(aaf_ds(aaf_ds(
        -L.prr_wau, L.pri_wfz), L.prr_rcw), L.prs_scw), L.prg_scw),
        L.prg_gcw), dorho);
    if (qcten != nullptr) {
        qcten[idx] = (float)aaf_da((double)qcten[idx], qc_rate);
    } else {
        qc[idx] = fmaxf(0.0f, aaf_fa(qc1d, aaf_fm((float)qc_rate, dt)));
    }
    ncten[idx] = (float)aaf_da((double)ncten[idx], aaf_dm(aaf_ds(aaf_ds(
        aaf_ds(aaf_ds(-L.pnc_wau, L.pnc_rcw), L.pni_wfz), L.pnc_scw),
        L.pnc_gcw), dorho));
    // :3022-3031.
    const double qi_rate = aaf_dm(aaf_ds(aaf_ds(aaf_ds(aaf_da(aaf_da(aaf_da(
        aaf_da(aaf_da(L.pri_inu, L.pri_iha), L.pri_ihm), L.pri_wfz),
        L.pri_rfz), L.pri_ide), L.prs_iau), L.prs_sci), L.pri_rci), dorho);
    const double ni_rate = aaf_dm(aaf_ds(aaf_ds(aaf_ds(aaf_da(aaf_da(aaf_da(
        aaf_da(aaf_da(L.pni_inu, L.pni_iha), L.pni_ihm), L.pni_wfz),
        L.pni_rfz), L.pni_ide), L.pni_iau), L.pni_sci), L.pni_rci), dorho);
    // :3058-3068.
    const double qr_rate = aaf_dm(aaf_ds(aaf_ds(aaf_ds(aaf_da(aaf_da(aaf_da(
        aaf_da(aaf_da(L.prr_wau, L.prr_rcw), L.prr_sml), L.prr_gml),
        L.prr_rcs), L.prr_rcg), L.prg_rfz), L.pri_rfz), L.prr_rci), dorho);
    const double nr_rate = aaf_dm(aaf_ds(aaf_da(aaf_da(L.pnr_wau, L.pnr_sml),
        L.pnr_gml), aaf_da(aaf_da(aaf_da(aaf_da(aaf_da(L.pnr_rfz, L.pnr_rcr),
        L.pnr_rcg), L.pnr_rcs), L.pnr_rci), L.pni_rfz)), dorho);
    if (accumulate_rain_ice) {
        qiten[idx] = (float)aaf_da((double)qiten[idx], qi_rate);
        niten[idx] = (float)aaf_da((double)niten[idx], ni_rate);
        float ice_number_tendency = niten[idx];
        thompson_aa_ice_balance_tendency(qi1d, ni1d, qiten[idx],
            &ice_number_tendency, L.rho, orho, odts, dt);
        niten[idx] = ice_number_tendency;
        qrten[idx] = (float)aaf_da((double)qrten[idx], qr_rate);
        nrten[idx] = (float)aaf_da((double)nrten[idx], nr_rate);
        float rain_mass_tendency = qrten[idx];
        float rain_number_tendency = nrten[idx];
        thompson_aa_rain_balance_tendency(qr1d, nr1d, &rain_mass_tendency,
            &rain_number_tendency, L.rho, orho, odts, dt);
        qrten[idx] = rain_mass_tendency;
        nrten[idx] = rain_number_tendency;
    } else {
        // The unit gates' in-place form: the same tendencies through the
        // same balances, applied to the state here.
        float qiten_l = (float)qi_rate;
        float niten_l = (float)ni_rate;
        thompson_aa_ice_balance_tendency(qi1d, ni1d, qiten_l, &niten_l,
                                         L.rho, orho, odts, dt);
        float qrten_l = (float)qr_rate;
        float nrten_l = (float)nr_rate;
        thompson_aa_rain_balance_tendency(qr1d, nr1d, &qrten_l, &nrten_l,
                                          L.rho, orho, odts, dt);
        qi[idx] = fmaxf(0.0f, aaf_fa(qi1d, aaf_fm(qiten_l, dt)));
        ni[idx] = fmaxf(0.0f, aaf_fa(ni1d, aaf_fm(niten_l, dt)));
        qr[idx] = fmaxf(0.0f, aaf_fa(qr1d, aaf_fm(qrten_l, dt)));
        nr[idx] = fmaxf(0.0f, aaf_fa(nr1d, aaf_fm(nrten_l, dt)));
    }
    // :3094-3097.  qsten, qgten and ngten are REAL accumulators (zero at
    // :1668-1677): REAL + DOUBLE, rounded once.
    const bool accumulate_frozen = qsten_acc != nullptr;
    const float qsten = (float)aaf_da(
        (double)(accumulate_frozen ? qsten_acc[idx] : 0.0f),
        aaf_dm(aaf_ds(aaf_ds(aaf_da(aaf_da(aaf_da(
        aaf_da(aaf_da(L.prs_iau, L.prs_sde), L.prs_sci), L.prs_scw),
        L.prs_rcs), L.prs_ide), L.prs_ihm), L.prr_sml), dorho));
    // :3100-3109.
    float qgten = (float)aaf_da(
        (double)(accumulate_frozen ? qgten_acc[idx] : 0.0f),
        aaf_dm(aaf_ds(aaf_ds(aaf_da(aaf_da(aaf_da(aaf_da(
        aaf_da(aaf_da(L.prg_scw, L.prg_rfz), L.prg_gde), L.prg_rcg),
        L.prg_gcw), L.prg_rci), L.prg_rcs), L.prg_ihm), L.prr_gml), dorho));
    float ngten = (float)aaf_da(
        (double)(accumulate_frozen ? ngten_acc[idx] : 0.0f),
        aaf_dm(aaf_ds(aaf_da(aaf_da(aaf_da(aaf_ds(aaf_da(
        L.png_scw, L.pnr_rfz), L.png_rcg), L.pnr_rci), L.png_rcs),
        L.png_gde), L.pnr_gml), dorho));
    // :3120-3161, the graupel mass/number balance (not hail aware).
    {
        const float xrg = fmaxf(AAF_R1, aaf_fm(aaf_fa(qg1d, aaf_fm(qgten, dt)),
                                              L.rho));
        const float xng = fmaxf(AAF_R2, aaf_fm(aaf_fa(ng1d, aaf_fm(ngten, dt)),
                                              L.rho));
        if (xrg > AAF_R1) {
            const double lamg = (double)thompson_aa_powf(
                aaf_fd(aaf_fm(aaf_fm(aaf_fm(AAF_AM_G5, 6.0f), 1.0f), xng),
                       xrg), AAF_OBM);
            const float mvd_g = (float)aaf_dd((double)AAF_MVD_NUM, lamg);
            float bound = 0.0f;
            if (mvd_g > 25.4e-3f) {
                bound = 25.4e-3f;
            } else if (mvd_g < AAF_D0R) {
                bound = AAF_D0R;
            }
            if (bound != 0.0f) {
                const double lb = (double)aaf_fd(AAF_MVD_NUM, bound);
                const float xng_b = (float)aaf_dd(aaf_dm(
                    (double)aaf_fm(aaf_fm(1.0f, AAF_O6), xrg),
                    thompson_aa_pow(lb, 3.0)), (double)AAF_AM_G5);
                ngten = aaf_fm(aaf_fm(aaf_fs(xng_b, aaf_fm(ng1d, L.rho)),
                                      odts), orho);
            }
        } else {
            qgten = aaf_fm(-qg1d, odts);
            ngten = aaf_fm(-ng1d, odts);
        }
    }
    if (accumulate_frozen) {
        // WRF applies qsten, qgten and ngten once, at :4054-4059, after the
        // snow and graupel fallout have added their own tendencies to the
        // same REAL accumulators; the fallout forms its working content as
        // qs1d + qsten*DT (:3257) and qg1d + qgten*DT (:3283).  Applying the
        // source tendency here and the fallout's on top of it rounded twice
        // where WRF rounds once.
        qsten_acc[idx] = qsten;
        qgten_acc[idx] = qgten;
        ngten_acc[idx] = ngten;
    } else {
        qs[idx] = aaf_fa(qs1d, aaf_fm(qsten, dt));
        qg[idx] = aaf_fa(qg1d, aaf_fm(qgten, dt));
        graupel_number_shadow[idx] = aaf_fa(ng1d, aaf_fm(ngten, dt));
    }

    // :3164-3174 and :3189-3192.  IFDRY = 0.
    const double t_rate = aaf_dm(aaf_dm(aaf_da(
        aaf_dm((double)aaf_fm(AAF_LSUB, L.ocp), aaf_da(aaf_da(aaf_da(aaf_da(
            aaf_da(L.pri_inu, L.pri_ide), L.prs_ide), L.prs_sde), L.prg_gde),
            L.pri_iha)),
        aaf_dm((double)aaf_fm(lfus2, L.ocp), aaf_da(aaf_da(aaf_da(aaf_da(
            aaf_da(aaf_da(aaf_da(aaf_da(aaf_da(L.pri_wfz, L.pri_rfz),
            L.prg_rfz), L.prs_scw), L.prg_scw), L.prg_gcw), L.prg_rcs),
            L.prs_rcs), L.prr_rci), L.prg_rcg))), dorho), 1.0);
    thompson_aa_wrf_apply_vapor_heat(idx, L.temp, qv, temperature,
                                     qvten_acc, tten_acc, qv_rate, t_rate,
                                     dt);
}


// ---------------------------------------------------------------------------
// The aerosol-aware cold network.
// ---------------------------------------------------------------------------
//
// ABI note: mp=8's kernel carries an include_cold_rain / include_cold_cloud /
// include_snow_rime_conversion / track_graupel_number flag matrix so a
// focused unit gate and the production path can share one translation unit.
// mp=28's production path ALWAYS has cloud water and always includes the
// complete cold-rain group, so those flags are dropped: this kernel is the
// full-group kernel unconditionally.  That is a real scope reduction, not an
// omission -- the disabled branches were unreachable under mp=28.
extern "C" __global__ void thompson_aa_cold_network(
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ qc,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    // Entry aerosol/droplet state, per kilogram.  READ-ONLY for the whole
    // mp=28 call (:1803-1830).
    const float* __restrict__ nc_entry,
    const float* __restrict__ nwfa_entry,
    const float* __restrict__ nifa_entry,
    // Per-kilogram tendency accumulators, applied once by WP-04's terminal
    // kernel.  This kernel only adds to them.
    float* __restrict__ ncten,
    float* __restrict__ nwfaten,
    float* __restrict__ nifaten,
    float* __restrict__ graupel_number_shadow,
    float* __restrict__ snow_velocity_boost,
    const double* __restrict__ ice_deposition_partition,
    const double* __restrict__ ice_to_snow_mass,
    const double* __restrict__ ice_to_snow_number,
    const double* __restrict__ tcs_racs1,
    const double* __restrict__ tmr_racs1,
    const double* __restrict__ tcs_racs2,
    const double* __restrict__ tmr_racs2,
    const double* __restrict__ tcr_sacr1,
    const double* __restrict__ tms_sacr1,
    const double* __restrict__ tcr_sacr2,
    const double* __restrict__ tms_sacr2,
    const double* __restrict__ tnr_racs1,
    const double* __restrict__ tnr_racs2,
    const double* __restrict__ tnr_sacr1,
    const double* __restrict__ tnr_sacr2,
    const double* __restrict__ tcg_racg,
    const double* __restrict__ tmr_racg,
    const double* __restrict__ tcr_gacr,
    const double* __restrict__ tnr_racg,
    const double* __restrict__ tnr_gacr,
    const double* __restrict__ rain_to_ice_mass,
    const double* __restrict__ rain_to_ice_number,
    const double* __restrict__ rain_to_graupel_mass,
    const double* __restrict__ rain_to_graupel_number,
    const double* __restrict__ rain_cloud_efficiency,
    const double* __restrict__ cloud_to_ice_mass,
    const double* __restrict__ cloud_to_ice_number,
    // t_Efsw, snow collecting cloud water (:2407).  Read by the v4.6.1
    // generation only.
    const double* __restrict__ snow_cloud_efficiency,
    // WRF's qcten (per kilogram per second).  Null: the cloud sink is
    // applied to qc in place, the unit gates' form.  Non-null (the
    // production adapter): qc is the read-only entry cloud and the sink is
    // ADDED here, to be applied once at :3975 with every later cloud
    // tendency (THE CLOUD-WATER ACCUMULATOR, microphysics_aerosol.py).
    float* __restrict__ qcten,
    // WRF's qrten / nrten / qiten / niten, likewise: all four null keeps the
    // in-place rain and ice apply and its in-place size bounds; all four
    // given (the production adapter's v4.6.1 generation) adds the sources to
    // them and runs the :3033-3055 / :3070-3091 balances in tendency form,
    // leaving qr, nr, qi and ni the read-only entry state.
    float* __restrict__ qrten,
    float* __restrict__ nrten,
    float* __restrict__ qiten,
    float* __restrict__ niten,
    // WRF's qvten / tten (REAL, per second), both given or both null.
    // Given (the production adapter's v4.6.1 generation), the sources are
    // ADDED to them in WRF's own form (:2982 and :3164-3172), qv stays the
    // read-only entry vapour qv1d, and temperature becomes WRF's TAU+1
    // t1d + DT*tten (:3189).  The condensation and the rain evaporation
    // then re-form qv1d + DT*qvten and t1d + DT*tten exactly as WRF does.
    float* __restrict__ qvten,
    float* __restrict__ tten,
    // WRF's qsten / qgten / ngten (REAL, per second), all three given or
    // all null.  Given (the production adapter's v4.6.1 generation), the
    // sources are ADDED to them and qs, qg and the private graupel number
    // stay the read-only entry state; the snow and graupel fallout add
    // their own tendencies and apply the sums once (:4054-4059).
    float* __restrict__ qsten,
    float* __restrict__ qgten,
    float* __restrict__ ngten,
    const float* __restrict__ graupel_volume,
    float* __restrict__ qbten,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const bool accumulate_rain_ice = niten != nullptr;
#if defined(THOMPSON_AA_HAIL)
    // The hail-aware column is always the accumulator form: its graupel
    // number and volume only exist as WRF's ngten/qbten.
    if (qgten == nullptr || ngten == nullptr || qbten == nullptr
            || graupel_volume == nullptr || !accumulate_rain_ice) return;
#else
    (void)graupel_volume;
    (void)qgten;
    (void)ngten;
    (void)qbten;
#endif

    // WRF resets vts_boost on every cold-source call (:2243) before
    // diagnosing the deposition-conditioned rimed-snow conversion.  Warm and
    // empty cells must also receive the neutral value because the later
    // column sedimentation kernel consumes the complete field.
    snow_velocity_boost[idx] = 1.0f;
    if (temperature[idx] >= 273.15f) {
        // :3033-3055, the cloud ice mass/number balance, runs at EVERY level
        // of a column with microphysics, not only below 0 C: ice that sits
        // at or above 0 C with no number (advection and analyses both leave
        // it) gets the 300 micron number from the balance's size clamp, and
        // ice with a number outside 5..300 microns is brought inside it.  The
        // warm network owns no ice, so without this such ice reached the
        // fallout with no number and the ice fallout gave it 5 micron
        // crystals instead: up to 70 percent of the final ice number off
        // WRF v4.6.1 on saved real-data analysis columns
        // (tools/thompson_real_column_parity).  No source process changes
        // ice here, so the balance reads the entry ice at the entry density
        // (:1802), exactly as :3036-3037 do.
        if (accumulate_rain_ice) {
            // The same balance in WRF's tendency form, at every level, on
            // the entry ice (no source changed it here) at the entry
            // density.
            const float qv_entry = fmaxf(1.0e-10f, qv[idx]);
            const float rho_entry = 0.622f * pressure[idx]
                / (287.04f * temperature[idx] * (qv_entry + 0.622f));
            float ice_number_tendency = niten[idx];
            thompson_aa_ice_balance_tendency(
                qi[idx], ni[idx], qiten[idx], &ice_number_tendency,
                rho_entry, 1.0f / rho_entry, 1.0f / dt, dt);
            niten[idx] = ice_number_tendency;
        } else if (qi[idx] > 0.0f || ni[idx] > 0.0f) {
            const float qv_entry = fmaxf(1.0e-10f, qv[idx]);
            const float rho_entry = 0.622f * pressure[idx]
                / (287.04f * temperature[idx] * (qv_entry + 0.622f));
#if defined(THOMPSON_AA_WRF39)
            thompson_aa_wrf39_bound_ice_number(qi[idx] * rho_entry,
                                               rho_entry, &ni[idx]);
#else
            thompson_aa_bound_ice_number(qi[idx] * rho_entry, rho_entry,
                                         &ni[idx]);
#endif
        }
        return;
    }

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float qvsi = thompson_rsif(pressure[idx], temp0);
    const float qvs = thompson_rslf(pressure[idx], temp0);
    float ssati = qv0 / qvsi - 1.0f;
    float ssatw = qv0 / qvs - 1.0f;
    if (fabsf(ssati) < 1.0e-15f) ssati = 0.0f;
    if (fabsf(ssatw) < 1.0e-15f) ssatw = 0.0f;
    // :2620-2621.  Koop's own gate (ssati >= 0.4) is strictly stronger than
    // this one, so the fast exit below cannot skip a productive Koop cell.
    const bool nucleation_active = ssati >= 0.25f
        || (ssatw > 1.0e-15f && temp0 < 253.15f);
    if (qi[idx] <= 1.0e-12f && qs[idx] <= 1.0e-12f
            && qg[idx] <= 1.0e-12f && qr[idx] <= 1.0e-12f
            && qc[idx] <= 1.0e-12f && !nucleation_active) {
#if defined(THOMPSON_AA_HAIL)
        // A clear level in a wet column still reaches WRF's mass/number
        // balances (:3054, :3088-3091, :3157-3159). Preserve their -0
        // tendencies when the source network has nothing to evaluate.
        const float odts = 1.0f / dt;
        niten[idx] = thompson_aa_mul(-ni[idx], odts);
        qrten[idx] = thompson_aa_mul(-qr[idx], odts);
        nrten[idx] = thompson_aa_mul(-nr[idx], odts);
        qgten[idx] = thompson_aa_mul(-qg[idx], odts);
        ngten[idx] = thompson_aa_mul(-graupel_number_shadow[idx], odts);
        qbten[idx] = thompson_aa_mul(-graupel_volume[idx], odts);
#endif
        return;
    }

#if !defined(THOMPSON_AA_WRF39) && !defined(THOMPSON_AA_HAIL)
    {
        ThompsonAaFrozenTables T;
        T.tpi_ide = ice_deposition_partition;
        T.tps_iaus = ice_to_snow_mass;
        T.tni_iaus = ice_to_snow_number;
        T.tcs_racs1 = tcs_racs1; T.tmr_racs1 = tmr_racs1;
        T.tcs_racs2 = tcs_racs2; T.tmr_racs2 = tmr_racs2;
        T.tcr_sacr1 = tcr_sacr1; T.tms_sacr1 = tms_sacr1;
        T.tcr_sacr2 = tcr_sacr2; T.tms_sacr2 = tms_sacr2;
        T.tnr_racs1 = tnr_racs1; T.tnr_racs2 = tnr_racs2;
        T.tnr_sacr1 = tnr_sacr1; T.tnr_sacr2 = tnr_sacr2;
        T.tcg_racg = tcg_racg; T.tmr_racg = tmr_racg;
        T.tcr_gacr = tcr_gacr; T.tnr_racg = tnr_racg;
        T.tnr_gacr = tnr_gacr;
        T.tpi_qrfz = rain_to_ice_mass;
        T.tni_qrfz = rain_to_ice_number;
        T.tpg_qrfz = rain_to_graupel_mass;
        T.tnr_qrfz = rain_to_graupel_number;
        T.tpi_qcfz = cloud_to_ice_mass;
        T.tni_qcfz = cloud_to_ice_number;
        T.t_Efsw = snow_cloud_efficiency;
        thompson_aa_cold_level_wrf461(
            idx, qi, ni, qs, qg, qr, nr, qc, temperature, pressure, qv,
            nc_entry, nwfa_entry, nifa_entry, ncten, nwfaten, nifaten,
            graupel_number_shadow, snow_velocity_boost, &T,
            rain_cloud_efficiency, qcten, qrten, nrten, qiten, niten,
            qvten, tten, qsten, qgten, ngten, dt);
        return;
    }
#endif
    // Below: the WRF 3.9 fork generation (THOMPSON_AA_WRF39) only.
    (void)snow_cloud_efficiency;

    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    // odts = 1./dtsave, :1649.  WRF writes every rate cap as X*odts, never
    // X/dt; with dt = 10 s the two differ because 1/10 is inexact in binary,
    // and the caps are exactly where a MIN can pick a different branch.
    const float inverse_dt = 1.0f / dt;
    const double inverse_dt_d = (double)inverse_dt;
    const float tempc = temp0 - 273.15f;
    const float inverse_temp = 1.0f / temp0;
    const float diffusivity = 2.11e-5f
        * thompson_aa_powf(__fdiv_rn(temp0, 273.15f), 1.94f)
        * (101325.0f / pressure[idx]);
    const float viscosity = (1.718f + 0.0049f * tempc
        - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float inverse_cp = 1.0f / (1004.0f * (1.0f + 0.887f * qv0));
    const float saturated_density = rho * qvsi;
    const float slope_term = inverse_temp
        * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f);
    const float saturation_slope = saturated_density *
#if defined(THOMPSON_AA_HAIL)
        inverse_temp * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f);
#else
        slope_term;
#endif
    const float saturation_curvature = saturated_density
#if defined(THOMPSON_AA_HAIL)
        // :2387-2389: preserve the REAL product chain rather than squaring
        // the rounded first derivative factor.
        * (inverse_temp * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f)
           * inverse_temp * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f)
#else
        * (slope_term * slope_term
#endif
           - 2.0f * 2.834e6f * inverse_temp * inverse_temp
             * inverse_temp * (1.0f / 461.5f)
           + inverse_temp * inverse_temp);
    const float gamma = 2.834e6f * diffusivity / conductivity
        * saturation_slope;
    const float gamma_ratio = gamma / (1.0f + gamma);
    const float alpha = fmaxf(1.0e-9f,
        0.5f * gamma_ratio * gamma_ratio
        * saturation_curvature / saturation_slope
        * saturated_density / saturation_slope);
    float xsat = ssati;
    if (fabsf(xsat) < 1.0e-9f) xsat = 0.0f;
    const float xsat2 = xsat * xsat;
    const float alpha2 = alpha * alpha;
    const float vapor_geometry = 4.0f * 3.1415926536f
        * (1.0f - alpha * xsat
#if defined(THOMPSON_AA_HAIL)
           + 2.0f * alpha * alpha * xsat * xsat
           - 5.0f * alpha * alpha * alpha * xsat * xsat * xsat)
#else
           + 2.0f * alpha2 * xsat2
           - 5.0f * alpha2 * alpha * xsat2 * xsat)
#endif
        / (1.0f + gamma);
    // rate_max, :2557 and :2864.
    const float vapor_limit = (qv0 - qvsi) * rho * inverse_dt * 0.999f;
#if defined(THOMPSON_AA_WRF39)
    // fork :2701: the summed deposition limiter's rate_max carries no
    // density, `(qv(k)-qvsi(k))*odts*0.999`, while each term's own limit
    // (fork :2427) does (audit T16).
    const float vapor_limit_sum = (qv0 - qvsi) * inverse_dt * 0.999f;
#else
    const float vapor_limit_sum = vapor_limit;
#endif

    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    // rhof(k) = SQRT(RHO_NOT/rho(k)), :1974.  thompson.cu:6600 writes the
    // algebraically equal RHO_NOT*orho; the two differ by up to an ulp of
    // rhof, which propagates into every collection rate.  mp=28 uses WRF's
    // own division so this kernel and the Fortran oracle agree to the last
    // float bit, exactly as thompson_aerosol_warm.cu already does.
    const float density_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) / rho);

    // -------------------------------------------------------------------
    // ENTRY AEROSOL AND DROPLET STATE, module_mp_thompson.F:1803-1842.
    // -------------------------------------------------------------------
    // aer_init_opt < 2 (the v1 scope pin), so both aerosol species take the
    // clamped branch at :1805-1806.
    const float nwfa_work = thompson_aa_clamp_nwfa(nwfa_entry[idx] * rho);
    const float nifa_work = thompson_aa_clamp_nifa(nifa_entry[idx] * rho);

    // L_qc and the entry droplet rediagnosis, :1827-1848.  When qc is at or
    // below R1 WRF sets rc = R1 and nc = 2.
    const bool has_cloud = qc[idx] > 1.0e-12f;
    const float cloud_mass = has_cloud ? qc[idx] * rho : 1.0e-12f;
    // ---- THE nu_c STAGING RULE, :1832 vs :2170 -------------------------
    // WRF computes nu_c TWICE from two DIFFERENT droplet numbers.  :1832
    // uses the PRE-rediagnosis nc of :1829 and exists only to build the
    // entry lamc and run the :1834-1838 droplet-size clamp.  :1840 then
    // REDIAGNOSES nc from that clamped lamc, and :2170 recomputes nu_c from
    // the rediagnosed value.  The :2170 answer is the one that feeds lamc
    // (:2173), mvd_c (:2174), Dc_g (:2181) and pnr_wau (:2191) -- i.e.
    // everything below.  Reusing the entry answer stays finite, stays
    // stable and is grossly wrong wherever the size clamp engages.
    //
    // Both sentinels are 0, never 12: 12 is exactly the value mp=8 froze
    // for Nt_c = 100e6, and a frozen droplet-shape constant sitting in a
    // live kernel is one dropped guard away from becoming the answer.
    // THOMPSON_AA_CCG2[0] and friends are the deliberate zero pad, so a
    // dropped guard produces a division by zero rather than mp=8 physics.
    int nu_c_entry = 0;             // :1832, diagnostic only
    int nu_c = 0;                   // :2170, THE working shape parameter
    double entry_lamc = 0.0;
    float nc_work = THOMPSON_AA_NC_FLOOR;
    if (has_cloud) {
        nc_work = thompson_aa_cloud_dist(
            cloud_mass, nc_entry[idx], rho, &nu_c_entry, &entry_lamc);
        nu_c = thompson_aa_nu_c_working(nc_work);
    }
    // entry_lamc is WRF's clamped entry lamc and nu_c_entry the shape
    // parameter that produced it.  NEITHER is usable after :1838; they are
    // captured so a reader can see which stage is which, and so the
    // readback probe at the bottom of this file can publish both.
    (void)entry_lamc;
    (void)nu_c_entry;

    // Cloud lambda and mean volume diameter, :2173-2175.  thompson.cu:
    // 6939-6944 folds nu_c = 12 into the literals 1.30767389e12f /
    // 2.08767448e-9f and the numerator 15.672f (which IS 3.0 + 12 + 0.672);
    // all three become live.  Written exactly as
    // thompson_aerosol_warm.cu:404-427, including WRF's type mixing: the
    // power is REAL(4) and is WIDENED into the DOUBLE PRECISION lamc
    // (declared at :1597), so mvd_c is a double division rounded once to
    // REAL on assignment -- not a float division.
    double cloud_lambda = 0.0;
    float cloud_mvd = 1.0e-6f;
    if (has_cloud) {
        cloud_lambda = (double)thompson_aa_powf(
            thompson_aa_div(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(nc_work, am_r),
                        THOMPSON_AA_CCG2[nu_c]),
                    THOMPSON_AA_OCG1[nu_c]),
                cloud_mass),
            THOMPSON_AA_OBMR);
        const float mvd = (float)(
            (double)((3.0f + (float)nu_c) + 0.672f) / cloud_lambda);
#if defined(THOMPSON_AA_WRF39)
        // fork :2096 carries no 1..50 micron hold on mvd_c (audit T20).
        cloud_mvd = mvd;
#else
        cloud_mvd = fmaxf(THOMPSON_AA_D0C,
                          fminf(mvd, THOMPSON_AA_D0R));
#endif
    }

    // -------------------------------------------------------------------
    // iceDeMott, evaluated ONCE (:2574 and :2623 return the same value).
    // -------------------------------------------------------------------
#if defined(THOMPSON_AA_WRF39)
    const float xni_demott = thompson_aa_wrf39_ice_demott(
        tempc, rho, nifa_work);
#else
    const float xni_demott = thompson_ice_demott(tempc, rho, nifa_work);
#endif
    // :2579-2591.  Replaces thompson.cu's hardcoded `nuclei_bin = 27`.
    const int nuclei_bin = thompson_aa_in_bin(xni_demott);
    // :3447-3448.  Replaces thompson.cu's hardcoded `cloud_number_bin = 65`.
    const int cloud_number_bin = thompson_aa_droplet_bin(nc_work);

    // -------------------------------------------------------------------
    // Snow moments, :2027-2101.  Hoisted out of thompson.cu's two separate
    // riming/deposition blocks because the aerosol scavenging terms need
    // smoe and xDs even when there is no cloud water.  Every expression is
    // transcribed unchanged, so the hoist cannot move a bit.
    // -------------------------------------------------------------------
    const float snow_mass = qs[idx] * rho;
    const bool has_snow = qs[idx] > 1.0e-12f;
    const float snow_tc0 = fminf(-0.1f, tempc);
    float smob = 0.0f;
    float smoc = 0.0f;
    float smo0 = 0.0f;
    float smoe = 0.0f;
    float snow_diameter = 0.0f;     // xDs
    float snow_number_ns = 0.0f;    // ns(k), :2081-2088
    if (has_snow) {
        smob = snow_mass * (1.0f / 0.069f);          // rs*oams, bm_s == 2
        // :2054 / :2067 / :2080 / :2101 are all `a_ * smo2**b_` with a_, b_
        // and smo2 REAL(4), which gfortran lowers to glibc's CORRECTLY
        // ROUNDED powf where CUDA's carries several ulp.  MEASURED against a
        // PUBLIC probe holding a verbatim copy of :2028-2088, compiled into
        // the same module_mp_thompson object the column oracle uses, over
        // 3721 states (61 temperatures 273.05 K -> 201.05 K x 61 log-spaced
        // snow contents 1e-12 -> 1e-2 kg m^-3), feeding it WRF's OWN a_ / b_
        // so the fits are not under test:
        //     smoc  plain powf  3489/3721 exact, max 1.663435e-07 relative
        //     smoc  powf_cr     3717/3721 exact, max 1.063558e-07
        //     ns    plain powf  3526/3721 exact, max 5.512328e-07
        //     ns    powf_cr     3716/3721 exact, max 1.344195e-07
        // thompson_field_a / thompson_field_b are BIT-EXACT on all 3721, so
        // the whole of the improvement is the power.  The four survivors are
        // not a fit defect and not repairable here: thompson_aa_powf is
        // `(float)thompson_aa_pow((double)x,(double)y)`, which DOUBLE-ROUNDS where
        // glibc's powf rounds once, and at four of these states the double
        // result straddles a float32 midpoint.  Reported, not hidden.
        smoc = thompson_field_a(snow_tc0, 3.0f)
            * thompson_aa_powf(smob, thompson_field_b(snow_tc0, 3.0f));
        smo0 = thompson_field_a(snow_tc0, 0.0f)
            * thompson_aa_powf(smob, thompson_field_b(snow_tc0, 0.0f));
        smoe = thompson_field_a(snow_tc0, 2.55f)
            * thompson_aa_powf(smob, thompson_field_b(snow_tc0, 2.55f));
        snow_diameter = smoc / smob;
        // THE explicit two-gamma integral.  Not smo0, not interchangeable
        // with it, and used by exactly one gate in the whole routine.
        snow_number_ns = thompson_aa_snow_number(smob, smoc);
    }

    // -------------------------------------------------------------------
    // Graupel distribution, hoisted for the same reason.  Transcribed from
    // thompson.cu:7112-7135 (the riming block's reconstruction of WRF's
    // N0_g(k)/ilamg(k), which gpuwm forms from the diagnosed intercept
    // because it carries no prognostic graupel number).
    // -------------------------------------------------------------------
    const float graupel_mass = qg[idx] * rho;
#if defined(THOMPSON_AA_WRF39)
    const float am_g = thompson_aa_wrf39_am_g();
#else
    const float am_g = pi * 400.0f / 6.0f;
#endif
    double graupel_lambda = 1.0;
    double graupel_ilam = 1.0;
    double graupel_intercept = 0.0;   // N0_g
    float graupel_diameter = 0.0f;    // xDg = (bm_g+mu_g+1)*ilamg
    // t1_qg_qc = PI*.25*av_g*cgg(9,1), :2432.  cgg(9,1) = 5.23476267.
    const float t1_qg_qc = pi * 0.25f * 442.0f * 5.23476267f;
    // t1_qs_qc = PI*.25*av_s, :794.
    const float t1_qs_qc = pi * 0.25f * 40.0f;
#if defined(THOMPSON_AA_WRF39)
    // fork :2031-2054: the slope from the column intercept the entry pass
    // (thompson_aa_wrf39_graupel_intercept) left in graupel_number_shadow,
    // on every level with L_qg; every consumer below gates on r_g(1).
    if (qg[idx] > THOMPSON_AA_R1) {
        thompson_aa_wrf39_graupel_slope(
            graupel_number_shadow[idx], graupel_mass,
            &graupel_lambda, &graupel_ilam, &graupel_intercept);
        graupel_diameter = (float)(4.0 * graupel_ilam);
    }
    if (false) {
#else
    if (graupel_mass >= 1.0e-6f) {
#endif
        const float intercept_power = fmaxf(2.0f, fminf(6.0f,
            3.0f + (2.0f / 7.0f)
                * (thompson_aa_log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f)));
        const float intercept_guess = thompson_aa_powf(10.0f, intercept_power);
        double lambda = (double)thompson_aa_powf(
            intercept_guess * am_g * 6.0f / graupel_mass, 0.25f);
        const float mvd = (float)(3.672 / lambda);
        if (mvd > 25.4e-3f) {
            lambda = 3.672 / 25.4e-3;
        } else if (mvd < 50.0e-6f) {
            lambda = 3.672 / 50.0e-6;
        }
        graupel_lambda = lambda;
        graupel_ilam = 1.0 / lambda;
        const float graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass, am_g)
            * (float)(lambda * lambda * lambda);
        graupel_intercept = (double)graupel_number * lambda;
        graupel_diameter = (float)(4.0 * graupel_ilam);
    }
#if defined(THOMPSON_AA_HAIL)
    // The hail-aware graupel: WRF's prognostic ng1d and qb1d, the density
    // bin idx_bg, and the slope/intercept of :2135-2139 (shared with the
    // warm network through thompson_aerosol_common.cuh).  The diagnosed
    // intercept above is mp=8/28's and is not read on this build.
    ThompsonAaHailGraupel hail;
    thompson_aa_hail_graupel_entry(qg[idx], graupel_number_shadow[idx],
                                   graupel_volume[idx], rho, &hail);
    graupel_lambda = hail.lamg;
    graupel_ilam = hail.ilamg;
    graupel_intercept = hail.n0_g;
    graupel_diameter = (float)(4.0 * hail.ilamg);
    // pbg_* and the hail-only number sources, each captured where WRF
    // computes it, i.e. BEFORE the conservation passes rescale the mass
    // rates (:2535, :2605, :2734, :2511, :2438, :2775; png_* :2509, :2703,
    // :2764).
    double hail_pbg_rcg = 0.0;
    double hail_pbg_scw = 0.0;
    double hail_png_gde = 0.0;
    // Graupel riming and aerosol scavenging, :2414-2439 and :2460-2471.  A
    // sub-freezing level has twet <= temp < T_0 (:2004-2013), so pbg_gcw is
    // live and the Ef_gw*0.1 melting factor is not.
    ThompsonAaHailCollect hail_collect;
    thompson_aa_hail_graupel_collect(
        &hail, has_cloud, cloud_mvd, cloud_mass, nc_work, nwfa_work,
        nifa_work, rho, temp0, temp0, density_factor, viscosity,
        inverse_dt, &hail_collect);
    const double hail_pbg_gcw = hail_collect.pbg_gcw;
#endif

    // -------------------------------------------------------------------
    // Cooper is GONE.  In mp=8 the ice-nucleation target is
    //     MIN(250.0e3f, 5.0f*expf(0.304f*(273.15f - T)))
    // (thompson.cu:6645-6646 and :7189-7200).  Under mp=28 WRF's if/else at
    // :2622-2626 selects iceDeMott instead, so that expression must not
    // appear anywhere in this file.
    // -------------------------------------------------------------------

    double ice_rate = 0.0;
    double ice_to_snow_rate = 0.0;
    double ice_number_rate = 0.0;
    double autoconversion_rate = 0.0;
    double autoconversion_number_rate = 0.0;
    double rain_ice_ice_rate = 0.0;
    double rain_ice_rain_rate = 0.0;
    double rain_ice_ice_number_rate = 0.0;
    double rain_ice_rain_number_rate = 0.0;
    double rain_ice_graupel_rate = 0.0;
    double rain_self_number_rate = 0.0;
    float ice_particle_mass = 1.0e-12f;
    const float ice_mass = qi[idx] * rho;
    // ni(k), :1859-1875.  WRF's entry ice number carries the 5..300 micron
    // mean-diameter clamps below and is R2 = 1e-6 when there is no ice.
    float ice_number = fmaxf(1.0e-6f, ni[idx] * rho);
    const float rain_mass = qr[idx] * rho;
    float rain_number;
    double rain_lambda;
    float rain_mvd;
    // N0_r, :2150.  The shared helper always runs WRF's :2146-2150
    // y-intercept pass, which RE-DERIVES lamr from the bounded nr instead of
    // reusing the clamped lambda, and publishes N0_r = nr*org2*lamr**cre(2).
    // prr_rcw, pnc_rcw, pna_rca and pnd_rcd all read N0_r and
    // (lamr + fv_r)**(-cre(9)) directly, so they must consume these two and
    // never a lambda re-derived from the clamped mvd.
    double rain_intercept_n0;
    const bool rain_active = thompson_aa_entry_rain_distribution(
        qr[idx], nr[idx], rho, &rain_number, &rain_lambda, &rain_mvd,
        &rain_intercept_n0);
    if (rain_active) {
        if (rain_mvd > 50.0e-6f) {
            const float efficiency = 1.0f
                - thompson_aa_expf(2300.0f * (rain_mvd - 1950.0e-6f));
            rain_self_number_rate = (double)(
                efficiency * 2.0f * rain_number * rain_mass);
        }
    }
    if (qi[idx] > 1.0e-12f) {
        const float am_i = pi * 890.0f / 6.0f;
        // :1855-1858.  Ice that arrives with no number (ni*rho <= R2) is
        // given 5-micron crystals BEFORE the size clamps below.  Without it
        // the R2 floor makes the mean diameter enormous, the 300-micron clamp
        // fires instead, and the ice sees a far smaller number: measured on
        // saved real-data columns, deposition, ice autoconversion and snow
        // collecting ice then differ from WRF v4.6.1 at 1,500 to 2,500
        // levels of every 19,600-column frame (advection and analyses both
        // leave ice mass without number).
#if defined(THOMPSON_AA_WRF39)
        // fork :1814-1817: 25 micron crystals, at most 499.D3.
        if (ice_number <= 1.0e-6f) {
            const double lambda25 = 4.0 / THOMPSON_AA_WRF39_NI_SEED_D;
            ice_number = fminf(THOMPSON_AA_WRF39_NI_MAX,
                __fdiv_rn(0.16666667163372040f * ice_mass, am_i)
                * (float)(lambda25 * lambda25 * lambda25));
        }
#elif defined(THOMPSON_AA_HAIL)
        if (ice_number <= 1.0e-6f) {
            const double lambda5 = (double)thompson_aa_div(4.0f, 5.0e-6f);
            const float pre = thompson_aa_div((1.0f / 6.0f) * ice_mass, am_i);
            ice_number = (float)fmin(999.0e3,
                (double)pre * (lambda5 * lambda5 * lambda5));
        }
#else
        if (ice_number <= 1.0e-6f) {
            const double lambda5 = 4.0 / 5.0e-6;
            ice_number = fminf(999.0e3f,
                __fdiv_rn((1.0f / 6.0f) * ice_mass, am_i)
                * (float)(lambda5 * lambda5 * lambda5));
        }
#endif
        double lambda = (double)thompson_aa_powf(
            am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f);
        double inverse_lambda = 1.0 / lambda;
        float mean_diameter = (float)(4.0 * inverse_lambda);
#if defined(THOMPSON_AA_HAIL)
        // :1865-1869: cie(2)/D is a REAL quotient, then lami is DOUBLE;
        // ni is rounded only after pre*lami**3. :2647 re-forms lami from
        // that bounded ni before evaluating the ice deposition rates.
        if (mean_diameter < 5.0e-6f || mean_diameter > 300.0e-6f) {
            const bool minimum = mean_diameter < 5.0e-6f;
            lambda = (double)thompson_aa_div(4.0f,
                minimum ? 5.0e-6f : 300.0e-6f);
            const float pre = thompson_aa_div((1.0f / 6.0f) * ice_mass, am_i);
            const double bounded = (double)pre * (lambda * lambda * lambda);
            ice_number = (float)(minimum ? fmin(999.0e3, bounded) : bounded);
        }
        lambda = (double)thompson_aa_powf(
            am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f);
        inverse_lambda = 1.0 / lambda;
        mean_diameter = (float)(4.0 * inverse_lambda);
#else
        if (mean_diameter < 5.0e-6f) {
            lambda = 4.0 / 5.0e-6;
            inverse_lambda = 1.0 / lambda;
#if defined(THOMPSON_AA_WRF39)
            ice_number = fminf(THOMPSON_AA_WRF39_NI_MAX,     // fork :1824
#else
            ice_number = fminf(999.0e3f,
#endif
                __fdiv_rn((1.0f / 6.0f) * ice_mass, am_i)
                * (float)(lambda * lambda * lambda));
            mean_diameter = 5.0e-6f;
        } else if (mean_diameter > 300.0e-6f) {
            lambda = 4.0 / 300.0e-6;
            inverse_lambda = 1.0 / lambda;
            ice_number = __fdiv_rn((1.0f / 6.0f) * ice_mass, am_i)
                * (float)(lambda * lambda * lambda);
            mean_diameter = 300.0e-6f;
        }
#endif
        // :2649/:2713, xDi = MAX(D0i, (bm_i+mu_i+1)*ilami) with D0i =
        // (xm0i/am_i)**(1/bm_i), about 12.9 microns.  xmi, the particle mass
        // that turns collected ice mass into a number (pni_sci, pni_rci), and
        // the rain-collects-ice gate mvd_r > 4*xDi both read the FLOORED
        // diameter.  The deposition branch below already floors its own copy.
        const float floored_ice_diameter = fmaxf(
            thompson_aa_powf(1.0e-12f / am_i, 1.0f / 3.0f), mean_diameter);
        ice_particle_mass = am_i *
#if defined(THOMPSON_AA_HAIL)
            thompson_aa_powf(floored_ice_diameter, 3.0f);
#else
            floored_ice_diameter * floored_ice_diameter * floored_ice_diameter;
#endif
        const int mass_bin = ice_mass > 1.0e-10f
            ? thompson_aa_decade_index(ice_mass, -10, 64) : 0;
        const int number_bin = ice_number > 1.0f
            ? thompson_aa_decade_index(ice_number, 0, 55) : 0;
        double total_rate = (double)(
            0.5f * vapor_geometry * diffusivity * ssati
            * saturated_density * ice_number) * inverse_lambda;
        if (total_rate > 0.0) {
            total_rate = fmin(total_rate, (double)vapor_limit);
            const double ice_fraction = ice_deposition_partition[
                mass_bin + 64 * number_bin];
            ice_rate = ice_fraction * total_rate;
            ice_to_snow_rate = (1.0 - ice_fraction) * total_rate;
        } else {
            total_rate = fmax((double)(-ice_mass * inverse_dt), total_rate);
            total_rate = fmax(total_rate, (double)vapor_limit);
            const float minimum_diameter = thompson_aa_powf(1.0e-12f / am_i, 1.0f / 3.0f);
            const float particle_diameter = fmaxf(
                minimum_diameter, mean_diameter);
            const float particle_mass = am_i *
#if defined(THOMPSON_AA_HAIL)
                thompson_aa_powf(particle_diameter, 3.0f);
#else
                particle_diameter * particle_diameter * particle_diameter;
#endif
#if defined(THOMPSON_AA_HAIL)
            // WRF :2651/:2657 forms oxmi in REAL before DOUBLE*REAL.
            ice_number_rate = total_rate * (double)thompson_aa_div(1.0f, particle_mass);
#else
            ice_number_rate = total_rate / (double)particle_mass;
#endif
            ice_number_rate = fmax(
                (double)(-ice_number * inverse_dt), ice_number_rate);
            ice_rate = total_rate;
        }

        const size_t table_idx = (size_t)mass_bin
            + (size_t)64 * (size_t)number_bin;
#if defined(THOMPSON_AA_WRF39)
        // fork :2538-2543, with D0s = 200 microns.
        const float iau_upper = thompson_aa_mul(5.0f, THOMPSON_AA_WRF39_D0S);
        const float iau_lower = thompson_aa_mul(0.1f, THOMPSON_AA_WRF39_D0S);
#else
        const float iau_upper = 1500.0e-6f;
        const float iau_lower = 30.0e-6f;
#endif
#if defined(THOMPSON_AA_HAIL)
        const double ice_autoconversion_bound = (double)(ice_mass * 0.99f * inverse_dt);
        const double number_autoconversion_bound = (double)(ice_number * 0.95f * inverse_dt);
#else
        const double ice_autoconversion_bound = (double)(ice_mass * 0.99f) * (double)inverse_dt;
        const double number_autoconversion_bound = (double)(ice_number * 0.95f) * (double)inverse_dt;
#endif
        if (mass_bin == 63 || mean_diameter > iau_upper) {
            autoconversion_rate = ice_autoconversion_bound;
            autoconversion_number_rate = number_autoconversion_bound;
        } else if (mean_diameter >= iau_lower) {
            autoconversion_rate = fmin(
                ice_autoconversion_bound,
                ice_to_snow_mass[table_idx] * (double)inverse_dt);
            autoconversion_number_rate = fmin(
                number_autoconversion_bound,
                ice_to_snow_number[table_idx] * (double)inverse_dt);
        }

        if (rain_mass >= 1.0e-6f
                && rain_mvd > 4.0f * floored_ice_diameter) {
            const double rain_intercept = (double)rain_number * rain_lambda;
            const double shifted_lambda = rain_lambda + 195.0;
            const float collection_efficiency = 0.95f;
            const float mass_prefactor = pi * 0.25f * 4854.0f * 6.0f;
            const float rain_mass_prefactor =
                pi * 0.25f * am_r * 4854.0f * 720.0f;
            rain_ice_ice_rate = (double)(
                density_factor * mass_prefactor
                * collection_efficiency * ice_mass)
                * rain_intercept * thompson_aa_pow(shifted_lambda, -4.0);
            rain_ice_rain_rate = (double)(
                density_factor * rain_mass_prefactor
                * collection_efficiency * ice_number)
                * rain_intercept * thompson_aa_pow(shifted_lambda, -7.0);
            rain_ice_rain_rate = fmin(
                rain_ice_rain_rate, (double)rain_mass * (double)inverse_dt);
            rain_ice_ice_number_rate =
                rain_ice_ice_rate
#if defined(THOMPSON_AA_HAIL)
                * (double)thompson_aa_div(1.0f, ice_particle_mass);
#else
                / (double)ice_particle_mass;
#endif
            rain_ice_rain_number_rate = (double)(
                density_factor * mass_prefactor
                * collection_efficiency * ice_number)
                * rain_intercept * thompson_aa_pow(shifted_lambda, -4.0);
#if !defined(THOMPSON_AA_WRF39)
            rain_ice_rain_number_rate = fmin(
                rain_ice_rain_number_rate,
                (double)rain_number * (double)inverse_dt);
#endif
            rain_ice_graupel_rate = rain_ice_ice_rate + rain_ice_rain_rate;
        }
    }

    // -------------------------------------------------------------------
    // Cold-rain group, :2492-2553 and :2594-2603.
    // -------------------------------------------------------------------
    (void)tcg_racg;
    double freeze_ice_rate = 0.0;
    double freeze_ice_number_rate = 0.0;
    double freeze_graupel_rate = 0.0;
    double freeze_graupel_number_rate = 0.0;
    // pnr_rfz, the rain-number sink of the freezing: the table's tnr_qrfz
    // (= freeze_graupel_number_rate) or nr*odts below HGFR.
    double freeze_rain_number_rate = 0.0;
    double rain_snow_rain_rate = 0.0;
    double rain_snow_category_rate = 0.0;
    double rain_snow_graupel_rate = 0.0;
    double rain_snow_number_rate = 0.0;
    double rain_graupel_rain_rate = 0.0;
    double rain_graupel_graupel_rate = 0.0;
    double rain_graupel_number_rate = 0.0;
    if (rain_active) {
        const int rain_mass_bin = thompson_aa_decade_index(rain_mass, -6, 37);
        const double table_rain_intercept =
            (double)(__fdiv_rn((1.0f / 6.0f) * rain_mass, am_r))
            * rain_lambda * rain_lambda * rain_lambda * rain_lambda;
        const int rain_intercept_bin =
            thompson_aa_decade_index_double(table_rain_intercept, 6, 37);

        // Bigg raindrop freezing, :2594-2603.  idx_IN IS LIVE HERE.
        if (rain_mass > 1.0e-6f) {
            const int temp_bin = max(0, min((int)roundf(-tempc) - 1, 44));
            const size_t table_idx = (size_t)rain_mass_bin
                + (size_t)37 * ((size_t)rain_intercept_bin
                + (size_t)37 * ((size_t)temp_bin
                + (size_t)45 * (size_t)nuclei_bin));
            freeze_ice_rate = rain_to_ice_mass[table_idx] * inverse_dt_d;
            freeze_ice_number_rate =
                rain_to_ice_number[table_idx] * inverse_dt_d;
            freeze_graupel_rate =
                rain_to_graupel_mass[table_idx] * inverse_dt_d;
            freeze_graupel_number_rate = fmin(
                (double)rain_number,
                rain_to_graupel_number[table_idx]) * inverse_dt_d;
            freeze_rain_number_rate = freeze_graupel_number_rate;
        } else if (rain_mass > 1.0e-12f && temp0 < 235.16f) {
            freeze_ice_rate = (double)thompson_aa_mul(rain_mass, inverse_dt);
            freeze_ice_number_rate =
                (double)thompson_aa_mul(rain_number, inverse_dt);
            freeze_rain_number_rate = freeze_ice_number_rate;
        }

#if defined(THOMPSON_AA_WRF39)
        // fork :2236-2247 and :2377: r_s(1) = 1.e-5 and 28 snow entries.
        if (rain_mass >= 1.0e-6f
                && snow_mass >= THOMPSON_AA_WRF39_R_SG_FIRST) {
            const int ntb_s = THOMPSON_AA_WRF39_NTB_SG;
            const int snow_bin = thompson_aa_decade_index(
                snow_mass, THOMPSON_AA_WRF39_SG_EXP0, ntb_s);
#else
        if (rain_mass >= 1.0e-6f && snow_mass >= 1.0e-6f) {
            const int ntb_s = 37;
            const int snow_bin = thompson_aa_decade_index(snow_mass, -6, 37);
#endif
            const int raw_temp_bin = (int)(__fdiv_rn((tempc - 2.5f), 5.0f)) - 1;
            const int temp_bin = min(9, max(1, -raw_temp_bin)) - 1;
            const size_t table_idx = (size_t)snow_bin
                + (size_t)ntb_s * ((size_t)temp_bin
                + (size_t)9 * ((size_t)rain_intercept_bin
                + (size_t)37 * (size_t)rain_mass_bin));
            rain_snow_rain_rate = -(
                tmr_racs2[table_idx] + tcr_sacr2[table_idx]
                + tmr_racs1[table_idx] + tcr_sacr1[table_idx]);
            rain_snow_category_rate =
                tmr_racs2[table_idx] + tcr_sacr2[table_idx]
                - tcs_racs1[table_idx] - tms_sacr1[table_idx];
            rain_snow_graupel_rate =
                tmr_racs1[table_idx] + tcr_sacr1[table_idx]
                + tcs_racs1[table_idx] + tms_sacr1[table_idx];
            rain_snow_number_rate =
                tnr_racs1[table_idx] + tnr_racs2[table_idx]
                + tnr_sacr1[table_idx] + tnr_sacr2[table_idx];
            rain_snow_rain_rate = fmax(
                (double)(-thompson_aa_mul(rain_mass, inverse_dt)),
                rain_snow_rain_rate);
            rain_snow_category_rate = fmax(
                (double)(-thompson_aa_mul(snow_mass, inverse_dt)),
                rain_snow_category_rate);
            rain_snow_graupel_rate = fmin(
                (double)thompson_aa_mul(rain_mass + snow_mass, inverse_dt),
                rain_snow_graupel_rate);
            rain_snow_number_rate = fmin(
                (double)thompson_aa_mul(rain_number, inverse_dt),
                rain_snow_number_rate);
        }

#if defined(THOMPSON_AA_WRF39)
        // fork :2249-2270 and :2401-2416: r_g(1) = 1.e-5, 28 entries on the
        // content and intercept axes (N0g_exp(1) = 1.e4), the intercept
        // rebuilt from ilamg, tables (28,28,37,37) with no density axis.
        if (rain_mass >= 1.0e-6f
                && graupel_mass >= THOMPSON_AA_WRF39_R_SG_FIRST) {
            const int ntb_g = THOMPSON_AA_WRF39_NTB_SG;
            const int graupel_mass_bin = thompson_aa_decade_index(
                graupel_mass, THOMPSON_AA_WRF39_SG_EXP0, ntb_g);
            const int graupel_intercept_bin =
                thompson_aa_decade_index_double(
                    thompson_aa_wrf39_table_intercept(
                        graupel_mass, graupel_ilam),
                    THOMPSON_AA_WRF39_NG1_EXP0, ntb_g);
            const size_t table_idx = (size_t)graupel_intercept_bin
                + (size_t)ntb_g * ((size_t)graupel_mass_bin
                + (size_t)ntb_g * ((size_t)rain_intercept_bin
                + (size_t)37 * (size_t)rain_mass_bin));
            {
                rain_graupel_graupel_rate =
                    tmr_racg[table_idx] + tcr_gacr[table_idx];
                rain_graupel_graupel_rate = fmin(
                    (double)thompson_aa_mul(rain_mass, inverse_dt),
                    rain_graupel_graupel_rate);
                rain_graupel_rain_rate = -rain_graupel_graupel_rate;
                rain_graupel_number_rate =
                    tnr_racg[table_idx] + tnr_gacr[table_idx];
                rain_graupel_number_rate = fmin(
                    (double)thompson_aa_mul(rain_number, inverse_dt),
                    rain_graupel_number_rate);
            }
        }
#elif defined(THOMPSON_AA_HAIL)
        // Rain collecting graupel, :2522-2547, below 0 C (twet <= temp <
        // T_0 here): the nine-slab tables at the level's density bin, the
        // graupel bins from rg (:2355-2381; idx_g = 1 and idx_g1 = ntb_g1
        // when rg sits exactly on r_g(1)), and pbg_rcg = prg_rcg/rho_i
        // (:2535) taken before any conservation pass rescales prg_rcg.
        if (rain_mass >= 1.0e-6f && hail.rg >= 1.0e-6f) {
            int graupel_mass_bin = 0;
            int graupel_intercept_bin = 36;
            if (hail.rg > 1.0e-6f) {
                graupel_mass_bin = thompson_aa_decade_index(hail.rg, -6, 37);
                graupel_intercept_bin = thompson_aa_hail_intercept_bin(&hail);
            }
            const size_t table_idx = thompson_aa_hail_rain_graupel_index(
                graupel_intercept_bin, graupel_mass_bin, hail.idx_bg,
                rain_intercept_bin, rain_mass_bin);
            rain_graupel_graupel_rate = fmin(
                (double)thompson_aa_mul(rain_mass, inverse_dt),
                tmr_racg[table_idx] + tcr_gacr[table_idx]);
            rain_graupel_rain_rate = -rain_graupel_graupel_rate;
            rain_graupel_number_rate = fmin(
                (double)thompson_aa_mul(rain_number, inverse_dt),
                tnr_racg[table_idx] + tnr_gacr[table_idx]);
            hail_pbg_rcg = rain_graupel_graupel_rate
                / (double)THOMPSON_AA_HAIL_RHO_I;
        }
#endif
    }

    // -------------------------------------------------------------------
    // Cold cloud-water group, :2179-2222 and :2402-2443 and :2607-2617.
    // -------------------------------------------------------------------
    double cloud_autoconversion_rate = 0.0;
    double cloud_autoconversion_number_rate = 0.0;
    double cloud_rain_accretion_rate = 0.0;
    double cloud_freezing_rate = 0.0;
    double cloud_freezing_number_rate = 0.0;
    double snow_riming_rate = 0.0;
    double graupel_riming_rate = 0.0;
    double hm_number_rate = 0.0;
    double hm_mass_rate = 0.0;
    double snow_hm_rate = 0.0;
    double graupel_hm_rate = 0.0;
    // NEW under mp=28: the two cloud-number collection sinks.
    double cloud_number_snow_rate = 0.0;      // pnc_scw, :2410-2411
    double cloud_number_graupel_rate = 0.0;   // pnc_gcw, :2435-2437
    // -------------------------------------------------------------------
    // THE FOUR :2156-2232 NUMBER/AEROSOL TERMS THAT ALSO RUN SUB-FREEZING.
    // -------------------------------------------------------------------
    // WRF's warm-rain loop opens at :2157 and closes at :2234.  It precedes
    // the `if (.not. iiwarm)` gate at :2239 AND the `if (temp(k).lt.T_0)`
    // guard at :2554, so every rate inside it is evaluated at EVERY level,
    // sub-freezing ones included.  This kernel already carries the MASS
    // members of that loop (prr_wau at :2189-2190 and prr_rcw at :2202-2204)
    // because they are cloud-water sinks the cold group has to compete with;
    // their NUMBER and AEROSOL partners were missing, so cloud mass left
    // without its droplet number and rain wet-scavenging of CCN/IN was
    // absent from the entire sub-freezing half of the domain.
    //
    // NO DOUBLE COUNTING.  thompson_aerosol_warm.cu computes the same four
    // terms and returns early unless its held entry-temperature mask
    // (T_entry >= 273.15 K, seeded before ANY source kernel runs) is set;
    // this kernel returns at :268 for temperature >= 273.15 f and runs
    // FIRST, before any kernel has moved the temperature field.  The two
    // masks are therefore exact complements over every cell, with 273.15 K
    // itself owned by the warm side.  tests/test_thompson_aerosol_cold_gpu.py
    // ::test_cold_and_warm_temperature_masks_are_exact_complements pins that
    // at the two floats bracketing 273.15 as well as at 273.15 itself.
    double cloud_number_autoconversion_rate = 0.0;  // pnc_wau, :2192-2193
    double cloud_number_rain_rate = 0.0;            // pnc_rcw, :2205-2207
    double nwfa_rain_rate = 0.0;                    // pna_rca, :2213-2216
    double nifa_rain_rate = 0.0;                    // pnd_rcd, :2218-2221
    {
        // Berry-Reinhardt (1974) warm-rain autoconversion, :2179-2194.
        //
        // CONTRACTION-PINNED, matching thompson_aerosol_warm.cu:395-445 term
        // for term so the two halves of the scheme cannot disagree about
        // prr_wau.  Dc_b subtracts two nearly equal sixth powers (:2182), so
        // a single fused multiply-add moves it by several float ulps.
        // build_aero.sh compiles the oracle with plain `gfortran -O2` for
        // baseline x86-64, which has NO fma instruction, so every WRF product
        // and sum in this chain is separately rounded, while nvrtc defaults
        // to --fmad=true.  MEASURED over 11340 sub-freezing Fortran-oracle
        // rows spanning rc up to 0.15 kg m^-3: the unpinned form sat at
        // 1.03e-5 relative on prr_wau / pnr_wau / pnc_wau; pinned, and with
        // lamc and Dc_g carrying WRF's REAL-into-DOUBLE type mixing, 2.31e-6.
        // What is left is thompson_aa_cloud_dist's CUDA powf upstream.
        if (cloud_mass > 0.01e-3f && has_cloud) {
            // xDc, :2172.  D0c*1.E6 == 1.0 exactly.
            const float xdc = fmaxf(1.0f,
                thompson_aa_mul(
                    thompson_aa_powf(
                        thompson_aa_div(cloud_mass,
                                        thompson_aa_mul(am_r, nc_work)),
                        THOMPSON_AA_OBMR),
                    1.0e6f));
            // Dc_g = ((ccg(3,nu_c)*ocg2(nu_c))**obmr / lamc) * 1e6, :2181.
            const float dcg = (float)(
                (double)thompson_aa_powf(
                    thompson_aa_mul(THOMPSON_AA_CCG3[nu_c],
                                    THOMPSON_AA_OCG2[nu_c]),
                    THOMPSON_AA_OBMR)
                / cloud_lambda * 1.0e6);
            // WRF writes both sixth powers left-associated and un-grouped
            // (:2182-2183); the grouping is observable in float32.
            const float xdc3 = thompson_aa_mul(
                thompson_aa_mul(xdc, xdc), xdc);
            const float dcb_arg = thompson_aa_sub(
                thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(xdc3, dcg), dcg), dcg),
                thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(xdc3, xdc), xdc), xdc));
            const float dcb = thompson_aa_powf(
                fmaxf(0.0f, dcb_arg), 1.0f / 6.0f);
            const float zeta_term = thompson_aa_sub(
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(6.25e-6f, xdc), dcb), dcb), dcb),
                0.4f);
            const float zeta1 = thompson_aa_mul(
                0.5f, thompson_aa_add(zeta_term, fabsf(zeta_term)));
            const float zeta = thompson_aa_mul(
                thompson_aa_mul(0.027f, cloud_mass), zeta1);
            const float tau_diameter = thompson_aa_sub(
                thompson_aa_mul(0.5f, dcb), 7.5f);
            const float taud = thompson_aa_add(
                thompson_aa_mul(
                    0.5f, thompson_aa_add(tau_diameter, fabsf(tau_diameter))),
                THOMPSON_AA_R1);
            const float tau = thompson_aa_div(
                3.72f, thompson_aa_mul(cloud_mass, taud));
            cloud_autoconversion_rate = fmin(
                (double)thompson_aa_mul(cloud_mass, inverse_dt),
                (double)thompson_aa_div(zeta, tau));
            // pnr_wau, :2191.  nu_c appears EXPLICITLY; thompson.cu:3241 and
            // :6973 fold it into the literal 12.0f.  WRF's denominator is
            // written un-grouped, so it is left-associated here.
            cloud_autoconversion_number_rate = cloud_autoconversion_rate
                / (double)thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(
#if defined(THOMPSON_AA_WRF39)
                            // fork :2111, am_r*nu_c*D0r*D0r*D0r.
                            thompson_aa_mul(am_r, (float)nu_c),
#else
                            thompson_aa_mul(
                                thompson_aa_mul(am_r, (float)nu_c), 10.0f),
#endif
                            THOMPSON_AA_D0R),
                        THOMPSON_AA_D0R),
                    THOMPSON_AA_D0R);
            // pnc_wau, :2192-2193.  The droplet number that leaves with the
            // autoconverted mass, sized by the CLAMPED mvd_c -- not by xDc
            // and not by Dc_g.  prr_wau here is already MIN'd at :2190.
            cloud_number_autoconversion_rate = fmin(
                (double)thompson_aa_mul(nc_work, inverse_dt),
                cloud_autoconversion_rate
                / (double)thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(am_r, cloud_mvd), cloud_mvd),
                    cloud_mvd));
        }

        // Rain collecting cloud water and aerosol, :2196-2222.
        //
        // All four rates share ONE kernel,
        //     rhof(k)*t1_qr_qc*Ef*X*N0_r(k)*((lamr+fv_r)**(-cre(9)))
        // with X in {rc, nc, nwfa, nifa} and Ef the matching efficiency.
        // t1_qr_qc = PI*.25*av_r*crg(9) (:786) and cre(9) = mu_r+bv_r+3 = 4
        // exactly, so the power is thompson_aa_pow(lamr+195, -4).  WRF's TYPES are
        // reproduced literally: rhof, t1_qr_qc, Ef and X are REAL and are
        // multiplied left-to-right in float32; N0_r and lamr are DOUBLE
        // PRECISION, so the trailing two factors are double.
        //
        // GATES.  :2197 needs L_qr .and. mvd_r > D0r .and. mvd_c > D0c;
        // :2211 needs only L_qr .and. mvd_r > D0r, so rain scavenges CCN and
        // IN out of clear supercooled air with no cloud droplets present at
        // all -- exactly the state the cold network used to leave untouched.
        // L_qr is the MIXING-RATIO test qr1d > R1 (:1878), not qr*rho > R1;
        // the shared entry helper returns precisely that as rain_active.
        if (rain_active && rain_mvd > 50.0e-6f) {
            const float coefficient = pi * 0.25f * 4854.0f * 6.0f;
            // WRF multiplies left to right: ((REAL prefix)*N0_r)*power,
            // and forms the power from lamr = 1./ilamr(k) (:2198, :2213),
            // not from the :2147 lamr.  Pre-multiplying N0_r by the power,
            // or using the :2147 lamr, each moved these four rates by a
            // binary64 unit.
            const double fall = thompson_aa_pow(
                thompson_aa_rain_lambda_reformed(rain_lambda) + 195.0, -4.0);
            const float prefactor = density_factor * coefficient;

            if (cloud_mvd > 1.0e-6f) {
                const double dr_first = 5.1164649614037726e-05;
                const double dr_last = 0.004886186104779057;
                int rain_bin = 1 + (int)(100.0
                    * thompson_aa_log((double)rain_mvd / dr_first)
                    / thompson_aa_log(dr_last / dr_first));
                rain_bin = min(rain_bin, 100);
#if defined(THOMPSON_AA_WRF39)
                const int cloud_bin = min(100, (int)(cloud_mvd * 1.0e6f));
#else
                const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
#endif
                const float efficiency = (float)rain_cloud_efficiency[
                    (rain_bin - 1) + 100 * (cloud_bin - 1)];
                cloud_rain_accretion_rate = fmin(
                    (double)(cloud_mass * inverse_dt),
                    (double)(prefactor * efficiency * cloud_mass)
                    * rain_intercept_n0 * fall);
                // pnc_rcw, :2205-2207.  Same kernel with nc in place of rc,
                // MIN'd against nc*odts.  WRF caps this one even though it
                // deliberately leaves prg_gcw raw.
                cloud_number_rain_rate = fmin(
                    (double)thompson_aa_mul(nc_work, inverse_dt),
                    (double)(prefactor * efficiency * nc_work)
                    * rain_intercept_n0 * fall);
            }

            // pna_rca (:2212-2216) and pnd_rcd (:2218-2221).  Each species is
            // capped against the FULL available aerosol on its own; WRF lets
            // the sum of all six scavenging rates overshoot and relies on the
            // terminal floors at :3979-3981, so no shared limiter here.  The
            // wif_input_opt == 2 black-carbon term (:2223-2230) is out of
            // scope for this port.
            const float ef_ccn = thompson_eff_aero(
                rain_mvd, 0.04e-6f, viscosity, rho, temp0,
                THOMPSON_AA_SPECIES_RAIN);
            nwfa_rain_rate = fmin(
                (double)thompson_aa_mul(nwfa_work, inverse_dt),
                (double)(prefactor * ef_ccn * nwfa_work)
                    * rain_intercept_n0 * fall);
            const float ef_in = thompson_eff_aero(
                rain_mvd, 0.8e-6f, viscosity, rho, temp0,
                THOMPSON_AA_SPECIES_RAIN);
            nifa_rain_rate = fmin(
                (double)thompson_aa_mul(nifa_work, inverse_dt),
                (double)(prefactor * ef_in * nifa_work)
                    * rain_intercept_n0 * fall);
        }

        // Bigg cloud-droplet freezing, :2607-2617.  BOTH table indices that
        // mp=8 freezes are live here: cloud_number_bin (idx_n) and
        // nuclei_bin (idx_IN).
        if (cloud_mass > 1.0e-6f) {
            const int mass_bin = thompson_aa_decade_index(cloud_mass, -6, 37);
            const int temp_bin = max(0, min((int)roundf(-tempc) - 1, 44));
            const size_t table_idx = (size_t)mass_bin
                + (size_t)37 * ((size_t)cloud_number_bin
                + (size_t)100 * ((size_t)temp_bin
                + (size_t)45 * (size_t)nuclei_bin));
            cloud_freezing_rate = fmin(
                (double)thompson_aa_mul(cloud_mass, inverse_dt),
                cloud_to_ice_mass[table_idx] * inverse_dt_d);
            cloud_freezing_number_rate = fmin(
                (double)thompson_aa_mul(nc_work, inverse_dt),
                fmin(cloud_freezing_rate / (2.0 * 1.0e-12),
                     cloud_to_ice_number[table_idx] * inverse_dt_d));
        } else if (has_cloud && cloud_mass > 1.0e-12f && temp0 < 235.16f) {
            cloud_freezing_rate =
                (double)thompson_aa_mul(cloud_mass, inverse_dt);
            // :2616 -- nc(k)*odts, not Nt_c*odts.
            cloud_freezing_number_rate =
                (double)thompson_aa_mul(nc_work, inverse_dt);
        }

        // Snow collecting cloud water, :2402-2412.
#if defined(THOMPSON_AA_WRF39)
        // fork :2300-2305 and :864: D0s = 200 microns, snow bins from D0s.
        const float snow_d0 = THOMPSON_AA_WRF39_D0S;
        const double snow_d0_d = (double)THOMPSON_AA_WRF39_D0S;
#else
        const float snow_d0 = 300.0e-6f;
        const double snow_d0_d = 300.0e-6;
#endif
        if (has_cloud && has_snow && cloud_mvd > 1.0e-6f
                && snow_diameter > snow_d0) {
            const double diameter_ratio = 0.02 / snow_d0_d;
            const double log_ratio = thompson_aa_log(diameter_ratio);
            const double first_snow_bin = snow_d0_d
                * thompson_aa_exp(0.5 / 100.0 * log_ratio);
            const double last_snow_bin = snow_d0_d
                * thompson_aa_exp(99.5 / 100.0 * log_ratio);
            int snow_bin = 1 + (int)(100.0
                * thompson_aa_log((double)snow_diameter / first_snow_bin)
                / thompson_aa_log(last_snow_bin / first_snow_bin));
            snow_bin = min(snow_bin, 100);
            const double table_snow_diameter = snow_d0_d
                * thompson_aa_exp(((double)snow_bin - 0.5) / 100.0 * log_ratio);
#if defined(THOMPSON_AA_WRF39)
            const int cloud_bin = min(100, (int)(cloud_mvd * 1.0e6f));
#else
            const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
#endif
            // Dc(j) exactly as table_Efsw sees it: Dc(1) = D0c*1.0d0 with
            // the REAL(4) D0c, then + 1.0D-6 per bin (:831-834).
            double table_cloud_diameter = (double)1.0e-6f;
            for (int n = 2; n <= cloud_bin; ++n) {
                table_cloud_diameter += 1.0e-6;
            }
            const double cloud_velocity = 1.19e4
                * (1.0e4 * table_cloud_diameter
                   * table_cloud_diameter * 0.25);
            const double snow_velocity = 40.0
                * thompson_aa_pow(table_snow_diameter, 0.55)
                * thompson_aa_exp(-100.0 * table_snow_diameter) - cloud_velocity;
            const double melted_snow_diameter = thompson_aa_pow(
                0.069 * table_snow_diameter * table_snow_diameter
                    / (pi * 1000.0 / 6.0), 1.0 / 3.0);
            const double diameter_fraction = table_cloud_diameter
                / melted_snow_diameter;
            float efficiency = 0.0f;
            if (diameter_fraction <= 0.25
                    // table_Efsw (:4936) tests Dc(j).lt.6.E-6 and
                    // vts.lt.1.E-3 against REAL(4) literals.  Dc is summed
                    // in double (:834), so Dc(6) sits below 6.E-6 and WRF's
                    // bin 6 is zero; a double 6.0e-6 kept it.
                    && table_cloud_diameter >= (double)6.0e-6f
                    && snow_velocity >= (double)1.0e-3f) {
                const double stokes_number = table_cloud_diameter
                    * table_cloud_diameter * snow_velocity * 1000.0
                    / (9.0 * 1.718e-5 * melted_snow_diameter);
                const double reynolds_number = 9.0 * stokes_number
                    / (diameter_fraction * diameter_fraction * 1000.0);
                const double log_reynolds = thompson_aa_log(reynolds_number);
                const double k0 = thompson_aa_exp(-0.1007 - 0.358 * log_reynolds
                    + 0.0261 * log_reynolds * log_reynolds);
                const double z = thompson_aa_log(stokes_number / (k0 + 1.0e-15));
                const double h = 0.1465 + 1.302 * z - 0.607 * z * z
                    + 0.293 * z * z * z;
                const double yc0 = 2.0 / 3.14159265358979323846 * atan(h);
                const double value = (yc0 + diameter_fraction)
                    * (yc0 + diameter_fraction)
                    / ((1.0 + diameter_fraction) * (1.0 + diameter_fraction));
                efficiency = fmaxf(0.0f, fminf((float)value, 0.95f));
            }
            snow_riming_rate = (double)(density_factor * t1_qs_qc
                * efficiency * cloud_mass * smoe);
            snow_riming_rate = fmin(
                snow_riming_rate,
                (double)thompson_aa_mul(cloud_mass, inverse_dt));
            // pnc_scw, :2410-2411.  Same coefficients, nc in place of rc.
            cloud_number_snow_rate = (double)(density_factor * t1_qs_qc
                * efficiency * nc_work * smoe);
            cloud_number_snow_rate = fmin(
                cloud_number_snow_rate, (double)nc_work * (double)inverse_dt);
        }

        // Graupel collecting cloud water, :2414-2443.
#if defined(THOMPSON_AA_WRF39)
        // fork :2315-2320: rg >= r_g(1) = 1.e-5, and only graupel whose
        // mean size exceeds D0g = 250 microns collects cloud water.
        if (has_cloud && graupel_mass >= THOMPSON_AA_WRF39_R_SG_FIRST
                && cloud_mvd > 1.0e-6f
                && graupel_diameter > THOMPSON_AA_WRF39_D0G) {
#elif defined(THOMPSON_AA_HAIL)
        graupel_riming_rate = hail_collect.prg_gcw;
        cloud_number_graupel_rate = hail_collect.pnc_gcw;
        if (false) {
#else
        if (has_cloud && graupel_mass >= 1.0e-6f && cloud_mvd > 1.0e-6f) {
#endif
            const float velocity = (float)(
                (double)(density_factor * 442.0f * 20.3632278f
                         * (1.0f / 6.0f))
                * thompson_aa_pow(graupel_ilam, 0.89));
            const float stokes_number = cloud_mvd * cloud_mvd
                * velocity * 1000.0f
                / (9.0f * viscosity * graupel_diameter);
            float efficiency = 0.0f;
            if (stokes_number >= 0.4f && stokes_number <= 10.0f) {
                efficiency = 0.55f * thompson_aa_log10f(2.51f * stokes_number);
            } else if (stokes_number > 10.0f) {
                efficiency = 0.77f;
            }
            graupel_riming_rate = (double)(density_factor * t1_qg_qc
                * efficiency * cloud_mass)
                * graupel_intercept * thompson_aa_pow(graupel_ilam, 3.89);
            // pnc_gcw, :2435-2437.  MIN'd against nc*odts; WRF leaves the
            // paired MASS rate prg_gcw raw until the joint cloud-water
            // conservation pass, and does NOT bound it here.
            cloud_number_graupel_rate = (double)(density_factor * t1_qg_qc
                * efficiency * nc_work)
                * graupel_intercept * thompson_aa_pow(graupel_ilam, 3.89);
            cloud_number_graupel_rate = fmin(
                cloud_number_graupel_rate,
                (double)nc_work * (double)inverse_dt);
        }

        // Hallett-Mossop, :2425-2440 in thompson.cu terms.  Held when the
        // later cloud-water cap rescales the riming mass sources.
        if (graupel_riming_rate > 1.0e-15 && tempc > -8.0f) {
            float factor = 0.0f;
            if (tempc >= -5.0f && tempc < -3.0f) {
                factor = 0.5f * (-3.0f - tempc);
            } else if (tempc > -8.0f && tempc < -5.0f) {
                factor = 0.33333333f * (8.0f + tempc);
            }
            hm_number_rate = 3.5e8 * (double)factor * graupel_riming_rate;
            hm_mass_rate = 1.0e-12 * hm_number_rate;
            const double total_riming = snow_riming_rate + graupel_riming_rate;
            if (total_riming > 0.0) {
                snow_hm_rate = snow_riming_rate / total_riming * hm_mass_rate;
                graupel_hm_rate =
                    graupel_riming_rate / total_riming * hm_mass_rate;
            }
        }
    }

    // -------------------------------------------------------------------
    // FROZEN AEROSOL WET SCAVENGING, :2444-2478.
    // -------------------------------------------------------------------
    // Gated on rs > r_s(1) and rg > r_g(1) alone -- NOT on cloud water and
    // NOT on the riming efficiencies.  Each rate is capped against the FULL
    // available aerosol independently; WRF deliberately allows the four
    // rates to over-deplete in sum and relies on the terminal floors.
    double nwfa_snow_rate = 0.0;      // pna_sca
    double nifa_snow_rate = 0.0;      // pnd_scd
    double nwfa_graupel_rate = 0.0;   // pna_gca
    double nifa_graupel_rate = 0.0;   // pnd_gcd
#if defined(THOMPSON_AA_WRF39)
    const float frozen_first = THOMPSON_AA_WRF39_R_SG_FIRST;   // fork :2335
#else
    const float frozen_first = 1.0e-6f;
#endif
    if (snow_mass > frozen_first) {
        const float ef_ccn = thompson_eff_aero(
            snow_diameter, 0.04e-6f, viscosity, rho, temp0,
            THOMPSON_AA_SPECIES_SNOW);
        nwfa_snow_rate = (double)(density_factor * t1_qs_qc * ef_ccn
                                  * nwfa_work * smoe);
        nwfa_snow_rate = fmin(nwfa_snow_rate,
                              (double)nwfa_work * (double)inverse_dt);

        const float ef_in = thompson_eff_aero(
            snow_diameter, 0.8e-6f, viscosity, rho, temp0,
            THOMPSON_AA_SPECIES_SNOW);
        nifa_snow_rate = (double)(density_factor * t1_qs_qc * ef_in
                                  * nifa_work * smoe);
        nifa_snow_rate = fmin(nifa_snow_rate,
                              (double)nifa_work * (double)inverse_dt);
    }
#if defined(THOMPSON_AA_HAIL)
    nwfa_graupel_rate = hail_collect.pna_gca;
    nifa_graupel_rate = hail_collect.pnd_gcd;
    if (false) {
#else
    if (graupel_mass > frozen_first) {
#endif
        const float ef_ccn = thompson_eff_aero(
            graupel_diameter, 0.04e-6f, viscosity, rho, temp0,
            THOMPSON_AA_SPECIES_GRAUPEL);
        nwfa_graupel_rate = (double)(density_factor * t1_qg_qc * ef_ccn
                                     * nwfa_work)
            * graupel_intercept * thompson_aa_pow(graupel_ilam, 3.89);
        nwfa_graupel_rate = fmin(nwfa_graupel_rate,
                                 (double)nwfa_work * (double)inverse_dt);

        const float ef_in = thompson_eff_aero(
            graupel_diameter, 0.8e-6f, viscosity, rho, temp0,
            THOMPSON_AA_SPECIES_GRAUPEL);
        nifa_graupel_rate = (double)(density_factor * t1_qg_qc * ef_in
                                     * nifa_work)
            * graupel_intercept * thompson_aa_pow(graupel_ilam, 3.89);
        nifa_graupel_rate = fmin(nifa_graupel_rate,
                                 (double)nifa_work * (double)inverse_dt);
    }

    // -------------------------------------------------------------------
    // DEPOSITION NUCLEATION, :2619-2631.  iceDeMott, not Cooper.
    // -------------------------------------------------------------------
    // The target population subtracts the crystals this same call already
    // created by Bigg freezing of rain AND cloud droplets
    // (xni = ni(k) + (pni_rfz + pni_wfz)*dtsave, :2628), so it must run
    // after both freezing blocks.
    double nucleation_rate = 0.0;         // pri_inu
    double nucleation_number_rate = 0.0;  // pni_inu
    if (nucleation_active) {
        const float existing = ice_number
            + (float)((freeze_ice_number_rate + cloud_freezing_number_rate)
                      * (double)dt);
        const float delta = xni_demott - existing;
        nucleation_number_rate =
            (double)(0.5f * (delta + fabsf(delta)) * inverse_dt);
        nucleation_rate = fmin(
            (double)vapor_limit, 1.0e-12 * nucleation_number_rate);
        nucleation_number_rate = nucleation_rate / 1.0e-12;
    }

    // -------------------------------------------------------------------
    // KOOP (2001) HOMOGENEOUS FREEZING OF DELIQUESCED HAZE, :2633-2643.
    // -------------------------------------------------------------------
    double koop_rate = 0.0;         // pri_iha
    double koop_number_rate = 0.0;  // pni_iha
    {
#if defined(THOMPSON_AA_WRF39)
        // fork :2505-2506: smo0 + ni, at most 500.E3.
        const float xni_test = smo0 + ice_number
            + (float)((freeze_ice_number_rate + cloud_freezing_number_rate
                       + nucleation_number_rate) * (double)dt);
        const float koop_gate = 500.0e3f;
#else
        const float xni_test = snow_number_ns + ice_number
            + (float)((freeze_ice_number_rate + cloud_freezing_number_rate
                       + nucleation_number_rate) * (double)dt);
        const float koop_gate = 999.0e3f;
#endif
        if (xni_test <= koop_gate && temp0 < 238.0f && ssati >= 0.4f) {
            const float xnc = thompson_ice_koop(
                temp0, qv0, qvs, nwfa_work, dt);
            koop_number_rate = (double)(xnc * inverse_dt);
            // xm0i*0.1 -- a homogeneously frozen haze droplet starts TEN
            // TIMES smaller than a deposition-nucleated crystal.
            koop_rate = fmin((double)vapor_limit, 1.0e-13 * koop_number_rate);
            koop_number_rate = koop_rate / 1.0e-13;
        }
    }

    // -------------------------------------------------------------------
    // Snow and graupel vapor exchange, :2688-2712.
    // -------------------------------------------------------------------
    double snow_rate = 0.0;
    double snow_collection_rate = 0.0;
    double snow_collection_number_rate = 0.0;
    if (has_snow) {
        const float rho_factor_sqrt = sqrtf(density_factor);
        const float viscosity_factor = sqrtf(rho / viscosity);
        // :2067 and :2114, both `a_ * smo2**b_` -- see the measurement note
        // above the smoc/smo0/smoe block for why these are powf_cr.
        const float snow_first_moment = thompson_field_a(snow_tc0, 1.0f)
            * thompson_aa_powf(smob,
                                  thompson_field_b(snow_tc0, 1.0f));
        const float deposition_moment = 1.0f + (1.0f + 0.55f) * 0.5f;
        const float snow_ventilation_moment =
            thompson_field_a(snow_tc0, deposition_moment)
            * thompson_aa_powf(
                smob, thompson_field_b(snow_tc0, deposition_moment));
        const float snow_capacitance = fmaxf(0.15f, fminf(
            0.15f + __fdiv_rn((tempc + 1.5f) * (0.5f - 0.15f), (-30.0f + 1.5f)), 0.5f));
        const float ventilation_coefficient = 0.28f
            * thompson_aa_powf(0.632f, 1.0f / 3.0f) * sqrtf(40.0f);
        const float moment_sum = 0.86f * snow_first_moment
            + ventilation_coefficient * rho_factor_sqrt
              * viscosity_factor * snow_ventilation_moment;
        snow_rate = (double)(
            snow_capacitance * vapor_geometry * diffusivity * ssati
            * saturated_density * moment_sum);
        if (snow_rate > 0.0) {
            snow_rate = fmin(snow_rate, (double)vapor_limit);
        } else {
            snow_rate = fmax((double)(-snow_mass * inverse_dt), snow_rate);
            snow_rate = fmax(snow_rate, (double)vapor_limit);
        }

        if (qi[idx] > 1.0e-12f && snow_mass >= 1.0e-6f) {
            snow_collection_rate = (double)(
                t1_qs_qc * density_factor * 0.05f * ice_mass * smoe);
            snow_collection_number_rate = snow_collection_rate
#if defined(THOMPSON_AA_HAIL)
                * (double)thompson_aa_div(1.0f, ice_particle_mass);
#else
                / (double)ice_particle_mass;
#endif
        }
    }

    // Deposition-conditioned rimed-snow to graupel conversion, :2754-2777.
    double snow_graupel_conversion_rate = 0.0;
    double snow_graupel_conversion_number_rate = 0.0;
    if (snow_riming_rate > 2.0 * snow_rate && snow_rate > 1.0e-15) {
        const float rime_ratio = (float)fmin(
            30.0, snow_riming_rate / snow_rate);
        float graupel_fraction = fminf(
            0.95f, 0.15f + (rime_ratio - 2.0f) * 0.028f);
#if defined(THOMPSON_AA_WRF39)
        snow_velocity_boost[idx] = fminf(
            1.5f, 1.1f + (rime_ratio - 2.0f) * 0.016f);       // fork :2629
#else
        snow_velocity_boost[idx] = fminf(
            1.5f, 1.1f + (rime_ratio - 2.0f) * 0.014f);
#endif
        snow_graupel_conversion_rate =
            (double)graupel_fraction * snow_riming_rate;
        snow_graupel_conversion_number_rate = snow_graupel_conversion_rate
            * (double)smo0 / (double)snow_mass;

#if defined(THOMPSON_AA_HAIL)
        // :2766-2775 in WRF's REAL(4) order through WOOF's float32 libm:
        // the rime density both gates the conversion (Jensen's 150 kg m-3)
        // and sets the volume the converted rime brings, pbg_scw =
        // prg_scw/(0.5*(rime_dens+rho_s)), with rho_s from :2248.
        const float snow_velocity = thompson_aa_mul(
            thompson_aa_mul(THOMPSON_AA_AV_S,
                            gfk_pow(snow_diameter, THOMPSON_AA_BV_S)),
            gfk_exp(-thompson_aa_mul(100.0f, snow_diameter)));
        float rime_parameter = thompson_aa_div(
            thompson_aa_mul(
                -thompson_aa_mul(cloud_mvd, 0.5e6f), snow_velocity),
            fminf(-0.1f, tempc));
        rime_parameter = fmaxf(0.1f, fminf(rime_parameter, 10.0f));
        const float rime_density = thompson_aa_mul(
            thompson_aa_sub(
                thompson_aa_add(0.051f,
                                thompson_aa_mul(0.114f, rime_parameter)),
                thompson_aa_mul(thompson_aa_mul(0.0055f, rime_parameter),
                                rime_parameter)),
            1000.0f);
        if (rime_density < 150.0f) {
            graupel_fraction = 0.0f;
            snow_graupel_conversion_rate = 0.0;
            snow_graupel_conversion_number_rate = 0.0;
        }
        const float rho_s = fmaxf(
            THOMPSON_AA_HAIL_RHO_G[1],
            fminf(thompson_aa_div(0.13f, snow_diameter),
                  THOMPSON_AA_HAIL_RHO_I - 100.0f));
        hail_pbg_scw = snow_graupel_conversion_rate / (double)thompson_aa_mul(
            0.5f, thompson_aa_add(rime_density, rho_s));
#elif !defined(THOMPSON_AA_WRF39)
        const float snow_velocity = 40.0f
            * thompson_aa_powf(snow_diameter, 0.55f)
            * thompson_aa_expf(-100.0f * snow_diameter);
        float rime_parameter = -(cloud_mvd * 0.5e6f) * snow_velocity
            / fminf(-0.1f, tempc);
        rime_parameter = fmaxf(0.1f, fminf(rime_parameter, 10.0f));
        const float rime_density = (0.051f + 0.114f * rime_parameter
            - 0.0055f * rime_parameter * rime_parameter) * 1000.0f;
        if (rime_density < 150.0f) {
            graupel_fraction = 0.0f;
            snow_graupel_conversion_rate = 0.0;
            snow_graupel_conversion_number_rate = 0.0;
        }
#endif  // the fork has no rime-density gate (fork :2625-2633)
        snow_riming_rate = (1.0 - (double)graupel_fraction) * snow_riming_rate;
    }

    double graupel_rate = 0.0;
#if defined(THOMPSON_AA_WRF39)
    // fork :2567-2575: prg_gde = C_cube*t1_subl*diffu*ssati*rvs * N0_g *
    // (t1_qg_sd*ilamg**cge(10) + t2_qg_sd*vsc2*rhof2*ilamg**cge(11)),
    // REAL prefactors, DOUBLE N0_g and ilamg; per-term rate_max with rho.
    if (qg[idx] > 1.0e-12f && ssati < -1.0e-15f) {
        const float rho_factor_sqrt = sqrtf(density_factor);
        const float viscosity_factor = sqrtf(rho / viscosity);
        const float t2_qg_sd = 0.28f * thompson_aa_powf(0.632f, 0.33333334326744080f)
            * sqrtf(442.0f) * 1.9021706581115723f;
        graupel_rate = (double)(0.5f * vapor_geometry * diffusivity * ssati
                                * saturated_density)
            * graupel_intercept
            * ((double)0.86f * thompson_aa_pow(graupel_ilam, 2.0)
               + (double)(t2_qg_sd * viscosity_factor * rho_factor_sqrt)
                 * thompson_aa_pow(graupel_ilam, (double)2.945f));
        if (graupel_rate < 0.0) {
            graupel_rate = fmax(fmax((double)(-graupel_mass * inverse_dt),
                                     graupel_rate), (double)vapor_limit);
        } else {
            graupel_rate = fmin(graupel_rate, (double)vapor_limit);
        }
    }
    if (false) {
#elif defined(THOMPSON_AA_HAIL)
    // Deposition/sublimation of the hail-aware graupel, :2696-2707, on the
    // level's density bin.  ssati < -eps, so the rate is a sublimation and
    // takes the number with it, png_gde = prg_gde*ng/rg.
    if (hail.l_qg && ssati < -1.0e-15f) {
        const float rhof2 = sqrtf(density_factor);
        const float vsc2 = sqrtf(thompson_aa_div(rho, viscosity));
        const float t2_qg_sd = thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(0.28f, THOMPSON_AA_HAIL_SC3),
                            sqrtf(THOMPSON_AA_HAIL_AV_G[hail.idx_bg])),
            THOMPSON_AA_HAIL_CGG[11]);
        const float prefix = thompson_aa_mul(thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_HAIL_C_CUBE,
                                            vapor_geometry),
                            diffusivity), ssati), saturated_density);
        graupel_rate = (double)prefix * hail.n0_g
            * ((double)THOMPSON_AA_HAIL_T1_QG_SD
                   * thompson_aa_pow(hail.ilamg, (double)THOMPSON_AA_HAIL_CGE[10])
               + (double)thompson_aa_mul(thompson_aa_mul(t2_qg_sd, vsc2),
                                         rhof2)
                   * thompson_aa_pow(hail.ilamg, (double)THOMPSON_AA_HAIL_CGE[11]));
        if (graupel_rate < 0.0) {
            graupel_rate = fmax(fmax(
                (double)thompson_aa_mul(-hail.rg, inverse_dt), graupel_rate),
                (double)vapor_limit);
            hail_png_gde = graupel_rate * (double)hail.ng / (double)hail.rg;
        } else {
            graupel_rate = fmin(graupel_rate, (double)vapor_limit);
        }
    }
    if (false) {
#else
    if (qg[idx] > 1.0e-12f && ssati < -1.0e-15f) {
#endif
        const float rho_factor_sqrt = sqrtf(density_factor);
        const float viscosity_factor = sqrtf(rho / viscosity);
        const float intercept_power = fmaxf(2.0f, fminf(
            3.0f + (2.0f / 7.0f)
                * (thompson_aa_log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f),
            6.0f));
        const float diagnosed_intercept = thompson_aa_powf(10.0f, intercept_power);
        float lambda = thompson_aa_powf(
            diagnosed_intercept * am_g * 6.0f / graupel_mass, 0.25f);
        float number_per_kg = __fdiv_rn((1.0f / 6.0f) * graupel_mass
            * thompson_aa_powf(lambda, 3.0f), am_g) / rho;
        number_per_kg = fmaxf(1.0e-6f, number_per_kg);
        float graupel_number = fmaxf(1.0e-6f, number_per_kg * rho);
        lambda = thompson_aa_powf(am_g * 6.0f * graupel_number / graupel_mass,
                      1.0f / 3.0f);
        float mvd = 3.672f / lambda;
        if (mvd > 25.4e-3f) {
            mvd = 25.4e-3f;
            lambda = 3.672f / mvd;
            graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass
                * thompson_aa_powf(lambda, 3.0f), am_g);
        } else if (mvd < 50.0e-6f) {
            mvd = 50.0e-6f;
            lambda = 3.672f / mvd;
            graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass
                * thompson_aa_powf(lambda, 3.0f), am_g);
        }
        const float inverse_lambda = 1.0f / lambda;
        const float intercept = graupel_number * lambda;
        const float ventilation_coefficient = 0.28f
            * thompson_aa_powf(0.632f, 1.0f / 3.0f)
            * sqrtf(442.0f) * 1.9021706581115723f;
        const float moment_sum = intercept * (
            0.86f * thompson_aa_powf(inverse_lambda, 2.0f)
            + ventilation_coefficient * viscosity_factor
              * rho_factor_sqrt * thompson_aa_powf(inverse_lambda, 2.945f));
        graupel_rate = (double)(
            0.5f * vapor_geometry * diffusivity * ssati
            * saturated_density * moment_sum);
        graupel_rate = fmax((double)(-graupel_mass * inverse_dt), graupel_rate);
        graupel_rate = fmax(graupel_rate, (double)vapor_limit);
    }
#if defined(THOMPSON_AA_HAIL)
    // pbg_rfz = prg_rfz/rho_i is formed at :2605, BEFORE the rain
    // conservation pass (:2904-2914) rescales prg_rfz; hold that value.
    const double hail_prg_rfz_held = freeze_graupel_rate;
#endif

    // -------------------------------------------------------------------
    // SHARED FROZEN-VAPOR LIMITER, :2862-2876.
    // -------------------------------------------------------------------
    // sump = pri_inu + pri_ide + prs_ide + prs_sde + prg_gde + pri_iha.
    // pri_iha JOINS the sum and IS rescaled; pni_iha and pni_inu are
    // deliberately left alone (WRF rescales only the six MASS members plus
    // pni_ide).
    double vapor_sum = nucleation_rate + ice_rate + ice_to_snow_rate
        + snow_rate + graupel_rate + koop_rate;
#if defined(THOMPSON_AA_HAIL)
    const float vapor_sum_real = (float)vapor_sum;
    if ((vapor_sum_real > 1.0e-15f && vapor_sum_real > vapor_limit_sum)
            || (vapor_sum_real < -1.0e-15f && vapor_sum_real < vapor_limit_sum)) {
        const double ratio = (double)thompson_aa_div(vapor_limit_sum, vapor_sum_real);
#else
    if ((vapor_sum > 1.0e-15 && vapor_sum > (double)vapor_limit_sum)
            || (vapor_sum < -1.0e-15
                && vapor_sum < (double)vapor_limit_sum)) {
        const double ratio = (double)vapor_limit_sum / vapor_sum;
#endif
        nucleation_rate *= ratio;
        ice_rate *= ratio;
        ice_to_snow_rate *= ratio;
        ice_number_rate *= ratio;
        snow_rate *= ratio;
        graupel_rate *= ratio;
        koop_rate *= ratio;
        vapor_sum = (double)vapor_limit_sum;
    }

    // Cloud-water conservation, :2878-2890.  The paired number tendencies
    // (pnc_scw, pnc_gcw, pni_wfz) and the H-M terms intentionally stay held
    // when these mass rates are rescaled.
    const double cloud_sum = -cloud_autoconversion_rate
        - cloud_freezing_rate - cloud_rain_accretion_rate
        - snow_riming_rate - snow_graupel_conversion_rate
        - graupel_riming_rate;
    // sump, rate_max and ratio are REAL (:1615; fork :1620 declares the
    // same), as for the rain limiter below: the DOUBLE sum is rounded to
    // REAL, compared in REAL against the REAL -rc*odts, and the REAL
    // quotient widens to rescale the DOUBLE rates.  This limiter fires where
    // the step drains the whole cloud, and :3975 then forms
    // qc1d + qcten*DT as the difference of two nearly equal numbers, so the
    // ratio's last bit decides between exactly zero and a one-ulp residue
    // of the entry cloud (with a floor droplet number and a radius).
    const float cloud_sum_real = (float)cloud_sum;
    const float cloud_limit_real = -cloud_mass * inverse_dt;
    if (cloud_sum_real < cloud_limit_real && has_cloud) {
        const double ratio = (double)thompson_aa_div(cloud_limit_real,
                                                     cloud_sum_real);
        cloud_autoconversion_rate *= ratio;
        cloud_freezing_rate *= ratio;
        cloud_rain_accretion_rate *= ratio;
        snow_riming_rate *= ratio;
        snow_graupel_conversion_rate *= ratio;
        graupel_riming_rate *= ratio;
    }

    // Cloud-ice conservation, :2892-2903.
    const double ice_limit = (double)(-ice_mass * inverse_dt);
    const double ice_sum = ice_rate - autoconversion_rate
        - snow_collection_rate - rain_ice_ice_rate;
#if defined(THOMPSON_AA_HAIL)
    const float ice_sum_real = (float)ice_sum;
    if (qi[idx] > 1.0e-12f && ice_sum_real < (float)ice_limit) {
        const double ratio = (double)thompson_aa_div((float)ice_limit, ice_sum_real);
#else
    if (qi[idx] > 1.0e-12f && ice_sum < (double)-1.0e-15f
            && ice_sum < ice_limit) {
        const double ratio = ice_limit / ice_sum;
#endif
        const double unbounded_ice_rate = ice_rate;
        ice_rate *= ratio;
        autoconversion_rate *= ratio;
        snow_collection_rate *= ratio;
        rain_ice_ice_rate *= ratio;
        vapor_sum += ice_rate - unbounded_ice_rate;
    }

    // Rain conservation, :2905-2917.
    const double rain_limit = (double)(-rain_mass * inverse_dt);
    const double rain_sum = -freeze_graupel_rate - freeze_ice_rate
        - rain_ice_rain_rate + rain_snow_rain_rate + rain_graupel_rain_rate;
#if !defined(THOMPSON_AA_WRF39)
    // WRF declares sump, rate_max and ratio REAL (:1615): the double sum is
    // rounded to REAL, compared in REAL, and the REAL ratio widens to
    // rescale the DOUBLE rates.  A double ratio is 3e-8 away, and at a level
    // the limiter drains (aero-cold-overlap level 6, 99.97 percent of the
    // rain) that is 2.48e-05 of what survives.
    const float rain_sum_real = (float)rain_sum;
    const float rain_limit_real = (float)rain_limit;
    if (rain_active && rain_sum_real < rain_limit_real) {
        const double ratio = (double)thompson_aa_div(rain_limit_real,
                                                     rain_sum_real);
#else
    if (rain_active && rain_sum < rain_limit) {
        const double ratio = rain_limit / rain_sum;
#endif
        freeze_graupel_rate *= ratio;
        freeze_ice_rate *= ratio;
        rain_ice_rain_rate *= ratio;
        rain_snow_rain_rate *= ratio;
        rain_graupel_rain_rate *= ratio;
    }

    // Snow conservation, :2919-2929.
    const double snow_limit = (double)(-snow_mass * inverse_dt);
    const double snow_sum = snow_rate - snow_hm_rate + rain_snow_category_rate;
#if defined(THOMPSON_AA_HAIL)
    if (has_snow && (float)snow_sum < (float)snow_limit) {
        const double ratio = (double)thompson_aa_div((float)snow_limit, (float)snow_sum);
#else
    if (has_snow && snow_sum < snow_limit) {
        const double ratio = snow_limit / snow_sum;
#endif
        const double unbounded_snow_vapor_rate = snow_rate;
        snow_rate *= ratio;
        snow_hm_rate *= ratio;
        rain_snow_category_rate *= ratio;
        vapor_sum += snow_rate - unbounded_snow_vapor_rate;
    }

    // Graupel conservation, :2931-2941.
    const double graupel_limit = (double)(-graupel_mass * inverse_dt);
    const double graupel_sum = graupel_rate - graupel_hm_rate
        + rain_graupel_graupel_rate;
#if defined(THOMPSON_AA_HAIL)
    if (qg[idx] > 1.0e-12f && (float)graupel_sum < (float)graupel_limit) {
        const double ratio = (double)thompson_aa_div((float)graupel_limit, (float)graupel_sum);
#else
    if (qg[idx] > 1.0e-12f && graupel_sum < graupel_limit) {
        const double ratio = graupel_limit / graupel_sum;
#endif
        const double unbounded_graupel_vapor_rate = graupel_rate;
        graupel_rate *= ratio;
        graupel_hm_rate *= ratio;
        rain_graupel_graupel_rate *= ratio;
        vapor_sum += graupel_rate - unbounded_graupel_vapor_rate;
    }

    // Blossey re-enforcement of the paired rain/graupel transfer,
    // :2945-2947: the REAL ratio rounds the paired magnitude to float32 and
    // the rain rate's own sign is kept (thompson_aerosol_common.cuh).  The
    // rain-snow pair is re-enforced only where twet > T_0, which a level
    // below 0 C never is.
    thompson_aa_reenforce_pair(&rain_graupel_rain_rate,
                               &rain_graupel_graupel_rate);
    hm_mass_rate = snow_hm_rate + graupel_hm_rate;

    const double rain_rate = cloud_autoconversion_rate
        + cloud_rain_accretion_rate
        - freeze_graupel_rate - freeze_ice_rate - rain_ice_rain_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
#if defined(THOMPSON_AA_WRF39)
    // fork :2901: pnr_rfz is the freezing's whole rain-number sink; the
    // drops the table freezes into ice (pni_rfz) are not removed again
    // (audit T7).
    const double rain_number_sink = rain_self_number_rate
        + freeze_rain_number_rate
        + rain_ice_rain_number_rate + rain_snow_number_rate
        + rain_graupel_number_rate;
#else
    (void)freeze_rain_number_rate;
    const double rain_number_sink = rain_self_number_rate
        + freeze_graupel_number_rate + freeze_ice_number_rate
        + rain_ice_rain_number_rate + rain_snow_number_rate
        + rain_graupel_number_rate;
#endif

    const float latent_vapor = 2.5e6f + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const double fusion_rate = cloud_freezing_rate
        + snow_riming_rate + snow_graupel_conversion_rate
        + graupel_riming_rate
        + freeze_ice_rate + freeze_graupel_rate
        + rain_ice_rain_rate + rain_snow_category_rate
        + rain_snow_graupel_rate + rain_graupel_graupel_rate;
    // qv1d + DT*qvten on the vapour as it ENTERED, not on the 1.E-10 floor
    // the rates read (:1800): WRF's running vapour is qv1d plus the summed
    // tendency and it is floored once, at the terminal apply (:3974), in a
    // column with microphysics (thompson_aa_state_finalize_with_columns).
    // Adding the source to the floored working value put 1.E-10 kg/kg of
    // vapour into every level that entered at zero and sublimated or
    // deposited: 292 levels of a saved real-data analysis state against WRF
    // v4.6.1's own Fortran.  Every reader floors it again for its working
    // value (:3192).  Pinned for the reason given at the qc apply below.
    const float post_source_qv = thompson_aa_sub(qv[idx],
        thompson_aa_mul((float)(vapor_sum * (double)orho), dt));
    const float post_source_temperature = temp0 + (float)(
        (double)(2.834e6f * inverse_cp) * vapor_sum
        * (double)orho * (double)dt
        + (double)(latent_fusion * inverse_cp) * fusion_rate
        * (double)orho * (double)dt);

    // -------------------------------------------------------------------
    // State update, :3020-3031 and :3096-3120.
    // -------------------------------------------------------------------
    //
    // CONTRACTION, AND WHY EVERY MULTIPLY-ADD BELOW IS PINNED (WP-13b).
    // WRF applies the accumulated tendencies at :3973-4023 in the form
    // `q1d(k) = q1d(k) + qten(k)*DT`, and the gfortran -O2 baseline-x86-64
    // build the oracle comes from has NO FMA instruction: qten*DT is rounded
    // to REAL(4) BEFORE the add.  Written plainly, nvrtc (--fmad=true)
    // contracts each of these into a single fma and never rounds the product,
    // so the two disagree by up to one ulp of the ENTRY value -- which is a
    // 1e-5 RELATIVE error wherever the level is nearly emptied, and that is
    // exactly where this port's surviving residuals live.
    //
    // MEASURED, wp08-freeze level 0: entry qc = 3.0000001424923540e-04, this
    // kernel's qc tendency -2.727245919231791e-05, dt = 10.  The rounded
    // product gives 2.7275411412119865e-05, which is WRF's own answer; the
    // fused form gives 2.7275422326056287e-05, which is what these lines
    // produced before WP-13b.  Same at aero-cold-overlap level 4:
    // 1.5486410120502114e-05 (WRF, and now this kernel) against
    // 1.5486406482523307e-05 (fused).
    //
    // thompson_aerosol_sat.cu:911-916 already pinned exactly this multiply-add
    // for the rain-evaporation apply, so the source networks were the
    // inconsistent half -- this is not a new tie-break, it is the same one as
    // "THE SHARED FITS" in thompson_aerosol_common.cuh: the authority is WRF,
    // not ArWen's mp=8 sibling.  Pinned by tests/
    // test_thompson_aerosol_adapter.py::
    // test_the_source_networks_round_the_tendency_before_adding_it, which
    // tests the PROPERTY (post - entry must round-trip through float32) and
    // not a constant.
    // Entry rain and graupel as WRF holds them after :1878-1905 and
    // :1915-1949 (zero where the mixing ratio is at or below R1), for the
    // source-stage removals at :3088-3091 and :3157-3159 below.
    const float qr_entry_wrf = qr[idx] > THOMPSON_AA_R1 ? qr[idx] : 0.0f;
    const float nr_entry_wrf = qr[idx] > THOMPSON_AA_R1 ? nr[idx] : 0.0f;
    const float qg_entry_wrf = qg[idx] > THOMPSON_AA_R1 ? qg[idx] : 0.0f;
    const float ng_entry_wrf = qg[idx] > THOMPSON_AA_R1
        ? graupel_number_shadow[idx] : 0.0f;
    // :3022-3031.  REAL + DOUBLE*orho, rounded once.
    const double ice_mass_rate = (nucleation_rate + koop_rate + hm_mass_rate
        + cloud_freezing_rate + freeze_ice_rate + ice_rate
        - autoconversion_rate - snow_collection_rate - rain_ice_ice_rate)
        * (double)orho;
    const double ice_number_rate_sum = (nucleation_number_rate
        + koop_number_rate + hm_number_rate + cloud_freezing_number_rate
        + freeze_ice_number_rate + ice_number_rate
        - autoconversion_number_rate - snow_collection_number_rate
        - rain_ice_ice_number_rate) * (double)orho;
    if (accumulate_rain_ice) {
        qiten[idx] = (float)((double)qiten[idx] + ice_mass_rate);
        niten[idx] = (float)((double)niten[idx] + ice_number_rate_sum);
    } else {
        qi[idx] = fmaxf(0.0f, thompson_aa_add(qi[idx],
            thompson_aa_mul((float)ice_mass_rate, dt)));
        ni[idx] = fmaxf(0.0f, thompson_aa_add(ni[idx],
            thompson_aa_mul((float)ice_number_rate_sum, dt)));
    }
#if defined(THOMPSON_AA_HAIL)
    // :3094-3097. Preserve the entry snow and accumulate REAL + DOUBLE.
    const double snow_tendency = ((((((autoconversion_rate + snow_rate)
        + snow_collection_rate) + snow_riming_rate)
        + rain_snow_category_rate) + ice_to_snow_rate) - snow_hm_rate);
    qsten[idx] = (float)((double)qsten[idx] + snow_tendency * (double)orho);
#else
    qs[idx] = fmaxf(0.0f, thompson_aa_add(qs[idx],
        thompson_aa_mul(
            (float)((ice_to_snow_rate + snow_rate + autoconversion_rate
                     + snow_collection_rate + snow_riming_rate
                     + rain_snow_category_rate - snow_hm_rate)
                    * (double)orho),
            dt)));
#endif
#if defined(THOMPSON_AA_HAIL)
    {
        // :3099-3116, the hail-aware graupel mass, number and volume
        // tendencies, each REAL + DOUBLE*orho rounded once, summed in WRF's
        // own order.  Below 0 C prr_gml, pnr_gml, png_rcg and pbg_gml are
        // zero (they are set only in the :2779 melting branch) and pbg_sml
        // is never set at all.  qg, ng and qb stay the read-only entry
        // state; the terminal apply (thompson_aa_hail_graupel_terminal) adds
        // these once with the fallout's.
        const double pbg_rfz = hail_prg_rfz_held
            / (double)THOMPSON_AA_HAIL_RHO_I;                     // :2605
        const double pbg_rci = rain_ice_graupel_rate
            / (double)THOMPSON_AA_HAIL_RHO_I;                     // :2734
        const double pbg_rcs = rain_snow_graupel_rate
            / (double)THOMPSON_AA_HAIL_RHO_I;                     // :2511
        float hail_qgten = (float)((double)qgten[idx]
            + (snow_graupel_conversion_rate + freeze_graupel_rate
               + graupel_rate + rain_graupel_graupel_rate
               + graupel_riming_rate + rain_ice_graupel_rate
               + rain_snow_graupel_rate - graupel_hm_rate)
              * (double)orho);
        float hail_ngten = (float)((double)ngten[idx]
            + (snow_graupel_conversion_number_rate
               + freeze_graupel_number_rate
               + rain_ice_rain_number_rate + rain_snow_number_rate
               + hail_png_gde)
              * (double)orho);
        float hail_qbten = (float)((double)qbten[idx]
            + (hail_pbg_scw + pbg_rfz + hail_pbg_gcw + pbg_rci + pbg_rcs
               + hail_pbg_rcg
               + (graupel_rate - graupel_hm_rate)
                 / (double)THOMPSON_AA_HAIL_RHO_G[hail.idx_bg])
              * (double)orho);
        // :3118-3161 on the entry graupel as :1915-1949 rewrote it.
        const float qg_entry = qg[idx] > THOMPSON_AA_R1 ? qg[idx] : 0.0f;
        const float ng_entry = qg[idx] > THOMPSON_AA_R1
            ? graupel_number_shadow[idx] : 0.0f;
        thompson_aa_hail_graupel_balance(
            qg_entry, ng_entry, hail.qb_entry, hail.idx_bg, rho, orho,
            inverse_dt, dt, &hail_qgten, &hail_ngten, &hail_qbten);
        qgten[idx] = hail_qgten;
        ngten[idx] = hail_ngten;
        qbten[idx] = hail_qbten;
    }
#else
    qg[idx] = fmaxf(0.0f, thompson_aa_add(qg[idx],
        thompson_aa_mul(
            (float)((graupel_rate + freeze_graupel_rate
                     + snow_graupel_conversion_rate
                     + graupel_riming_rate - graupel_hm_rate
                     + rain_ice_graupel_rate + rain_snow_graupel_rate
                     + rain_graupel_graupel_rate)
                    * (double)orho),
            dt)));
#endif
#if !defined(THOMPSON_AA_WRF39) && !defined(THOMPSON_AA_HAIL)
    {
        // ng1d, the classic wrapper's private, untransported graupel number.
        // Signs mirror :3107-3109.  mp=28 changes nothing here: calc_refl10cm
        // and the graupel sedimentation both read it exactly as mp=8 does.
        const float initial_number_per_kg = graupel_number_shadow[idx];
        // png_gde = prg_gde*ng(k)/rg(k) (:2703) wherever L_qg holds, i.e.
        // the ENTRY mixing ratio passed R1, with ng(k) = MAX(R2, ng1d*rho)
        // from the entry block.  Gating on the entry concentration and on the
        // post-source mass dropped it for graupel of 1.0 to 1.6e-12 kg/kg
        // aloft, 1,200 to 2,000 levels of every saved 19,600-column
        // real-data frame (inert there only because the balance below then
        // removes that graupel).
        const float initial_number = qg_entry_wrf > 0.0f
            ? fmaxf(1.0e-6f, initial_number_per_kg * rho) : 0.0f;
        const double graupel_vapor_number_rate = qg_entry_wrf > 0.0f
            ? graupel_rate * (double)initial_number / (double)graupel_mass
            : 0.0;
        const double number_rate = freeze_graupel_number_rate
            + snow_graupel_conversion_number_rate
            + rain_ice_rain_number_rate
            + rain_snow_number_rate + graupel_vapor_number_rate;
        graupel_number_shadow[idx] = thompson_aa_add(initial_number_per_kg,
            thompson_aa_mul((float)(number_rate * (double)orho), dt));
    }
#endif
    // On the accumulator path the sum is :3058-3061's, in WRF's order (the
    // melt terms are zero below 0 C); the in-place form keeps the order the
    // fork-generation results were measured with.  The two differ by a
    // rounding of the double sum, which the tendency carries to the state:
    // one ulp of qrten at aero-cold-overlap level 6, 2.48e-05 on its qr.
    const double rain_mass_rate = (accumulate_rain_ice
        ? cloud_autoconversion_rate + cloud_rain_accretion_rate
          + rain_snow_rain_rate + rain_graupel_rain_rate
          - freeze_graupel_rate - freeze_ice_rate - rain_ice_rain_rate
        : rain_rate) * (double)orho;
    const double rain_number_rate_sum =
        (cloud_autoconversion_number_rate - rain_number_sink) * (double)orho;
    if (accumulate_rain_ice) {
        // :3058-3067.
        qrten[idx] = (float)((double)qrten[idx] + rain_mass_rate);
        nrten[idx] = (float)((double)nrten[idx] + rain_number_rate_sum);
    } else {
        qr[idx] = fmaxf(0.0f, thompson_aa_add(qr[idx],
            thompson_aa_mul((float)rain_mass_rate, dt)));
        nr[idx] = fmaxf(0.0f, thompson_aa_add(nr[idx],
            thompson_aa_mul((float)rain_number_rate_sum, dt)));
    }
    // :3975, `qc1d(k) = qc1d(k) + qcten(k)*DT`.  See the contraction note at
    // the head of this block; this is the apply the two measured cells there
    // were taken from.
    const double cloud_sink_per_kg = (cloud_autoconversion_rate
        + cloud_freezing_rate + cloud_rain_accretion_rate + snow_riming_rate
        + snow_graupel_conversion_rate + graupel_riming_rate) * (double)orho;
    if (qcten != nullptr) {
        // :2987, `qcten(k) = qcten(k) + (-prr_wau(k) - ...)*orho`: a REAL
        // plus a DOUBLE expression, rounded once to REAL.
        qcten[idx] = (float)((double)qcten[idx] - cloud_sink_per_kg);
    } else {
        qc[idx] = fmaxf(0.0f, thompson_aa_sub(qc[idx],
            thompson_aa_mul((float)cloud_sink_per_kg, dt)));
    }
    if (accumulate_rain_ice) {
        // :3033-3055 and :3070-3091 in WRF's tendency form.
        float ice_number_tendency = niten[idx];
        thompson_aa_ice_balance_tendency(qi[idx], ni[idx], qiten[idx],
            &ice_number_tendency, rho, orho, inverse_dt, dt);
        niten[idx] = ice_number_tendency;
        float rain_mass_tendency = qrten[idx];
        float rain_number_tendency = nrten[idx];
        thompson_aa_rain_balance_tendency(qr[idx], nr[idx],
            &rain_mass_tendency, &rain_number_tendency, rho, orho,
            inverse_dt, dt);
        qrten[idx] = rain_mass_tendency;
        nrten[idx] = rain_number_tendency;
    } else {
#if defined(THOMPSON_AA_WRF39)
    thompson_aa_wrf39_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
#else
    thompson_aa_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
#endif
    thompson_aa_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
    }
    // :3067-3091.  Where the post-source rain concentration is at or below R1
    // WRF discards the call's rain sources and removes the entry rain:
    // qrten = -qr1d*odts, nrten = -nr1d*odts.  The number bound above zeroes
    // only the number; the mass has to go too.
    if (!accumulate_rain_ice
            && !(thompson_aa_mul(qr[idx], rho) > THOMPSON_AA_R1)) {
        qr[idx] = thompson_aa_add(qr_entry_wrf, thompson_aa_mul(
            thompson_aa_mul(-qr_entry_wrf, inverse_dt), dt));
        nr[idx] = thompson_aa_add(nr_entry_wrf, thompson_aa_mul(
            thompson_aa_mul(-nr_entry_wrf, inverse_dt), dt));
    }
    // :3118-3160, the graupel mass/number balance, which WRF runs for
    // non-hail-aware Thompson too.  At or below R1 the graupel and its
    // private number are removed; otherwise the private number is reset so
    // the median volume diameter stays in [D0r, 25.4 mm].  calc_refl10cm
    // reads that number: without the balance the saved reflectivity differs
    // from WRF v4.6.1 by up to 8.7 dB in 505 to 939 cells of every saved
    // forecast frame, and graupel at or below R1 as a concentration survives.
    // The fork has neither the number nor the balance; its graupel at or
    // below R1 is removed by the terminal apply (fork :3755).  The
    // hail-aware build ran its own balance in tendency form above.
#if !defined(THOMPSON_AA_WRF39) && !defined(THOMPSON_AA_HAIL)
    {
        const float xrg = thompson_aa_mul(qg[idx], rho);
        if (!(xrg > THOMPSON_AA_R1)) {
            qg[idx] = thompson_aa_add(qg_entry_wrf, thompson_aa_mul(
                thompson_aa_mul(-qg_entry_wrf, inverse_dt), dt));
            graupel_number_shadow[idx] = thompson_aa_add(ng_entry_wrf,
                thompson_aa_mul(thompson_aa_mul(-ng_entry_wrf, inverse_dt),
                                dt));
        } else {
            // REAL(4) throughout except lamg, which is DOUBLE (:1597).
            const float am_g = 3.1415926536f * 400.0f / 6.0f;
            const float mvd_numerator = 3.0f + 0.0f + 0.672f;
            const float shadow_m3 =
                thompson_aa_mul(graupel_number_shadow[idx], rho);
            const float xng = fmaxf(THOMPSON_AA_R2, shadow_m3);
            double lamg = (double)thompson_aa_powf(
                thompson_aa_div(thompson_aa_mul(thompson_aa_mul(am_g, 6.0f),
                                                xng), xrg),
                1.0f / 3.0f);
            const float mvd_g = (float)((double)mvd_numerator / lamg);
            bool bounded = false;
            if (mvd_g > 25.4e-3f) {
                lamg = (double)thompson_aa_div(mvd_numerator, 25.4e-3f);
                bounded = true;
            } else if (mvd_g < 50.0e-6f) {
                lamg = (double)thompson_aa_div(mvd_numerator, 50.0e-6f);
                bounded = true;
            }
            if (bounded) {
                const float xng_bounded = (float)(
                    (double)thompson_aa_mul(1.0f / 6.0f, xrg)
                    * thompson_aa_pow(lamg, 3.0) / (double)am_g);
                const float ngten = thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_sub(xng_bounded,
                                    thompson_aa_mul(ng_entry_wrf, rho)),
                    inverse_dt), thompson_aa_div(1.0f, rho));
                graupel_number_shadow[idx] = thompson_aa_add(
                    ng_entry_wrf, thompson_aa_mul(ngten, dt));
            }
        }
    }
#else
    (void)qg_entry_wrf;
    (void)ng_entry_wrf;
#endif
#if !defined(THOMPSON_AA_WRF39)
    if (tten != nullptr) {
        // :2982.  qvten += (-pri_inu - pri_iha - pri_ide - prs_ide - prs_sde
        // - prg_gde)*orho: that DOUBLE sum in that order, over the final
        // rates, REAL + DOUBLE rounded once.
        const double vapor_term = (((((-nucleation_rate) - koop_rate)
            - ice_rate) - ice_to_snow_rate) - snow_rate) - graupel_rate;
        // Every multiply-add below is pinned (__dmul_rn/__dadd_rn): the
        // gfortran build rounds each one, and nvrtc's default contraction
        // would fuse them.
        const float qvten_src = (float)__dadd_rn((double)qvten[idx],
            __dmul_rn(vapor_term, (double)orho));
        // :3164-3172, the temp < T_0 branch.  lsub*ocp and lfus2*ocp are
        // REAL products (lfus2 = lsub - lvap(k), :2960; lvap(k) from the
        // entry temperature, :1990), each times its DOUBLE sum in WRF's
        // term order, then *orho*(1-IFDRY), rounded once into tten.
        const float lvap = thompson_aa_add(
            2.5e6f, thompson_aa_mul(2106.0f - 4218.0f, tempc));
        const float lfus2 = thompson_aa_sub(2.834e6f, lvap);
        const double deposition_sum = ((((nucleation_rate + ice_rate)
            + ice_to_snow_rate) + snow_rate) + graupel_rate) + koop_rate;
        const double freezing_sum = ((((((((cloud_freezing_rate
            + freeze_ice_rate) + freeze_graupel_rate) + snow_riming_rate)
            + snow_graupel_conversion_rate) + graupel_riming_rate)
            + rain_snow_graupel_rate) + rain_snow_category_rate)
            + rain_ice_rain_rate) + rain_graupel_graupel_rate;
        const float tten_src = (float)__dadd_rn((double)tten[idx],
            __dmul_rn(__dadd_rn(
                __dmul_rn((double)thompson_aa_mul(2.834e6f, inverse_cp),
                          deposition_sum),
                __dmul_rn((double)thompson_aa_mul(lfus2, inverse_cp),
                          freezing_sum)),
                (double)orho));
        qvten[idx] = qvten_src;
        tten[idx] = tten_src;
        // :3189, temp(k) = t1d(k) + DT*tten(k).  qv stays qv1d.
        temperature[idx] = thompson_aa_add(
            temp0, thompson_aa_mul(dt, tten_src));
        (void)post_source_qv;
        (void)post_source_temperature;
    } else {
        qv[idx] = post_source_qv;
        temperature[idx] = post_source_temperature;
    }
#else
    (void)qvten;
    (void)tten;
    qv[idx] = post_source_qv;
    temperature[idx] = post_source_temperature;
#endif

    // -------------------------------------------------------------------
    // AEROSOL AND DROPLET NUMBER ACCUMULATORS, :2964-2994.
    // -------------------------------------------------------------------
    // WRF, verbatim and COMPLETE for a sub-freezing level:
    //   ncten   += (-pnc_wau - pnc_rcw - pni_wfz - pnc_scw - pnc_gcw)*orho
    //                                                        (:2993-2995)
    //   nwfaten -= (pna_rca + pna_sca + pna_gca + pni_iha)*orho (:2964-2965)
    //   nifaten -= (pnd_rcd + pnd_scd + pnd_gcd)*orho          (:2966-2967)
    //   nifaten -= pni_inu*orho     (:2974-2976, dustyIce is PARAMETER .true.)
    //
    // pnc_wau, pnc_rcw, pna_rca and pnd_rcd come from WRF's :2157-2234 loop,
    // which runs at EVERY level.  They used to be omitted here on the theory
    // that WP-07's warm kernel owned them; it does, but only for cells whose
    // ENTRY temperature was >= 273.15 K, and this kernel owns exactly the
    // complement.  Leaving them out therefore deleted them outright below
    // freezing rather than delegating them.
    //
    // pni_wfz is the value AFTER its own MIN chain at :2609-2612 but BEFORE
    // the cloud-water mass rescale, exactly as WRF uses it: the limiter at
    // :2880-2890 lists only the six MASS members.
    //
    // pni_iha and pni_inu are likewise the pre-vapor-limiter values.
    //
    // The sums are accumulated in DOUBLE and rounded once, matching WRF's
    // REAL accumulator fed by a DOUBLE right-hand side, and are grouped
    // left-to-right in WRF's own order.
    ncten[idx] = (float)((double)ncten[idx]
        + ((((-cloud_number_autoconversion_rate
              - cloud_number_rain_rate)
              - cloud_freezing_number_rate)
              - cloud_number_snow_rate)
              - cloud_number_graupel_rate) * (double)orho);
    nwfaten[idx] = (float)((double)nwfaten[idx]
        - (((nwfa_rain_rate + nwfa_snow_rate)
            + nwfa_graupel_rate) + koop_number_rate) * (double)orho);
    nifaten[idx] = (float)((double)nifaten[idx]
        - ((nifa_rain_rate + nifa_snow_rate)
           + nifa_graupel_rate) * (double)orho);
    nifaten[idx] = (float)((double)nifaten[idx]
        - nucleation_number_rate * (double)orho);
}


// ---------------------------------------------------------------------------
// READBACK PROBE.
// ---------------------------------------------------------------------------
//
// thompson_aa_cold_network's droplet-distribution staging and its four
// :2157-2234 number/aerosol rates have no end-to-end observable in the
// committed fixtures: every divergent nu_c state clamps mvd_c to D0c or D0r,
// and pnr_wau / Dc_g only switch on above rc = 0.01e-3.  A defect there is
// therefore invisible to a state-level comparison, which is exactly how the
// wrong-stage nu_c survived wave 2.  This kernel publishes the internals so
// the host can compare them against WRF v4.6.1 pointwise.
//
// It re-derives them from the SAME shared helpers the production kernel uses
// (thompson_aa_cloud_dist, thompson_aa_nu_c_working,
// thompson_aa_entry_rain_distribution, thompson_eff_aero) and reproduces the
// same expressions; tests/test_thompson_aerosol_cold_gpu.py additionally
// gates the probe against the production kernel's own ncten/nwfaten/nifaten
// on a shared column, so the two cannot drift apart silently.
extern "C" __global__ void thompson_aa_probe_cold_warm_loop(
    const float* __restrict__ qc,
    const float* __restrict__ nc_entry,
    const float* __restrict__ qr,
    const float* __restrict__ nr,
    const float* __restrict__ nwfa_entry,
    const float* __restrict__ nifa_entry,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const double* __restrict__ rain_cloud_efficiency,
    int* __restrict__ nu_c_entry_out,
    int* __restrict__ nu_c_working_out,
    float* __restrict__ nc_m3_out,
    float* __restrict__ mvd_c_out,
    float* __restrict__ mvd_r_out,
    double* __restrict__ pnc_wau_out,
    double* __restrict__ pnc_rcw_out,
    double* __restrict__ pna_rca_out,
    double* __restrict__ pnd_rcd_out,
    double* __restrict__ prr_wau_out,
    double* __restrict__ pnr_wau_out,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // The production level function's own entry state, per-level state and
    // warm-rain loop (thompson_aerosol_common.cuh), so the probe reads
    // exactly what thompson_aa_cold_network computes.
    ThompsonAaLevel L;
    thompson_aa_wrf_zero_rates(&L);
    thompson_aa_wrf_entry(&L, temperature[idx], pressure[idx], qv[idx],
                          qc[idx], nc_entry[idx], 0.0f, 0.0f, qr[idx],
                          nr[idx], 0.0f, 0.0f, 0.0f, nwfa_entry[idx],
                          nifa_entry[idx]);
    thompson_aa_wrf_level_state(&L, L.temp);
    thompson_aa_wrf_warm_loop(&L, aaf_fd(1.0f, dt), rain_cloud_efficiency);

    // :1832, the entry-stage shape parameter, diagnostic only.
    int nu_c_entry = 0;
    if (L.L_qc) {
        double entry_lamc = 0.0;
        (void)thompson_aa_cloud_dist(L.rc, nc_entry[idx], L.rho, &nu_c_entry,
                                     &entry_lamc);
    }

    nu_c_entry_out[idx] = nu_c_entry;
    nu_c_working_out[idx] = L.nu_c;
    nc_m3_out[idx] = L.L_qc ? L.nc : THOMPSON_AA_NC_FLOOR;
    mvd_c_out[idx] = L.mvd_c;
    mvd_r_out[idx] = L.mvd_r;
    pnc_wau_out[idx] = L.pnc_wau;
    pnc_rcw_out[idx] = L.pnc_rcw;
    pna_rca_out[idx] = L.pna_rca;
    pnd_rcd_out[idx] = L.pnd_rcd;
    prr_wau_out[idx] = L.prr_wau;
    pnr_wau_out[idx] = L.pnr_wau;
}
