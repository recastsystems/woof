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
// gpuwm/core/kernels/thompson_aerosol_warm.cu
//
// WP-07.  Aerosol-aware (mp_physics=28) WARM source network plus the
// standalone ncten balance limiter.
//
// Numerical authority is WRF v4.6.1 phys/module_mp_thompson.F
// (WRF v4.6.1, commit d66e442, zero local modifications).  Every bare line
// number below refers to that file.  Structural authority -- which rates are
// fused into one launch, the conservation order, the held-number quirks, the
// wet-bulb branch selection -- is gpuwm/core/kernels/thompson.cu's
// thompson_warm_frozen_source_network (:3105-3743), which this file COPIES.
// thompson.cu itself is byte-frozen and is not edited, and there is no
// #include path under cupy.RawModule, so its device helpers are duplicated
// here under thompson_aa_* names.  thompson_aerosol_common.cuh is prepended
// textually by gpuwm/core/kernels/__init__.py::_EXTRA_HEADERS.
//
// ---------------------------------------------------------------------------
// THE v4.6.1 BODY IS thompson_aa_warm_level_wrf461 (2026-10-07)
// ---------------------------------------------------------------------------
// For the v4.6.1 generation the warm source network hands every level at or
// above 0 C to thompson_aa_warm_level_wrf461: the same ThompsonAaLevel
// transcription of WRF's statements, in WRF's arithmetic order, that the
// cold network runs (thompson_aerosol_common.cuh), with WRF's wet-bulb
// temperature.  thompson_aa_warm_rain_rates and
// thompson_aa_frozen_collect_rates below remain the readback probes' and
// the WRF 3.9 fork generation's (THOMPSON_AA_WRF39) arithmetic; a change
// meant for the v4.6.1 warm-rain or frozen rates belongs in the shared
// ThompsonAaLevel functions (thompson_aa_wrf_warm_loop,
// thompson_aa_wrf_frozen_block, thompson_aa_wrf_conserve).
//
// ---------------------------------------------------------------------------
// WHAT CHANGES RELATIVE TO mp=8
// ---------------------------------------------------------------------------
// mp=8 freezes the droplet population: cloud_number = 100.0e6f, the gamma
// pair 1.30767389e12f / 2.08767448e-9f, the mvd numerator 15.672f, and the
// pnr_wau shape factor 12.0f are all compile-time constants at
// thompson.cu:3174-3187 and :3241.  Under mp=28 nc is prognostic, so
//
//   nu_c   = MIN(15, NINT(1000.E6/nc) + 2)            (:2171)   -> live 2..15
//   xDc    = MAX(D0c*1e6, (rc/(am_r*nc))**obmr * 1e6) (:2172)
//   lamc   = (nc*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc)**obmr(:2173)
//   mvd_c  = (3.0+nu_c+0.672)/lamc clamped [D0c,D0r]  (:2174-2175)
//   Dc_g   = ((ccg(3,nu_c)*ocg2(nu_c))**obmr/lamc)*1e6(:2181)
//   pnr_wau= prr_wau/(am_r*nu_c*10.*D0r**3)           (:2192)
//
// and every downstream consumer of mvd_c -- the t_Efrw cloud bin, the snow
// t_Efsw cloud bin, the graupel Stokes number -- becomes nc-driven for free.
// The literal 15.672f IS 3.0 + 12 + 0.672; the literal 12.0f in pnr_wau IS
// nu_c.  Both must trace to nc, never to a constant.
//
// NEW rate terms with no mp=8 counterpart, all written ONLY into the three
// scratch accumulators:
//   pnc_wau (:2192-2193), pnc_rcw (:2205-2206)      -> ncten
//   pnc_scw (:2411-2412), pnc_gcw (:2436-2437)      -> ncten
//   pna_rca (:2213-2216), pna_sca (:2444-2446),
//   pna_gca (:2462-2467)                            -> nwfaten
//   pnd_rcd (:2218-2221), pnd_scd (:2448-2450),
//   pnd_gcd (:2469-2471)                            -> nifaten
//
// ---------------------------------------------------------------------------
// ACCUMULATOR CONTRACT
// ---------------------------------------------------------------------------
// state nc/nwfa/nifa are READ-ONLY entry state for the whole mp=28 call.
// This kernel reads nc_entry/nwfa_entry/nifa_entry (per kg) and ADDS to
// ncten/nwfaten/nifaten (per kg per second), which a terminal kernel applies
// once with WRF's clamps (:3972-4021).  Nothing here writes the state arrays.
// WRF's own nc(k) inside the source phase is NOT (nc1d + ncten*dt): it is the
// entry value rediagnosed once at :1826-1842 and then held constant for the
// entire routine.  This kernel reproduces that literally via
// thompson_aa_cloud_dist, which is why nc_entry is the only droplet input.
//
// ---------------------------------------------------------------------------
// PACKAGE BOUNDARY (read before changing anything)
// ---------------------------------------------------------------------------
// iiwarm is a PARAMETER .false. (:59), so WRF's "frozen species" block at
// :2239+ executes at EVERY level, warm ones included.  ArWen's mp=8 split
// gives sub-freezing levels to the cold network and ambient-warm levels to
// this one, and thompson.cu's warm kernel accordingly carries prs_scw /
// prg_gcw for warm levels.  Their number and aerosol companions -- pnc_scw,
// pnc_gcw, pna_sca, pnd_scd, pna_gca, pnd_gcd -- therefore belong HERE for
// warm levels and in WP-06's kernel for cold levels.  WP-06's kernel returns
// early at temperature >= 273.15 (thompson.cu:6571), so there is no overlap;
// if that gate ever changes, these terms double-count silently.
//
// THE SAME IS TRUE ONE LOOP EARLIER, AND IT IS EASIER TO GET WRONG.  WRF's
// pnc_wau (:2192-2193), pnc_rcw (:2205-2207), pna_rca (:2213-2216) and
// pnd_rcd (:2218-2221) sit at :2160-2230, INSIDE the k-loop opened at :2156
// and BEFORE both the `.not. iiwarm` branch at :2239 and the
// `temp(k) .lt. T_0` guard at :2554, so WRF evaluates all four at EVERY
// level.  This kernel supplies them for levels whose ENTRY mask is set and
// WP-06's kernel supplies them for the complement.  The two gates are exact
// complements -- this one consumes `temperature >= 273.15` captured on the
// ENTRY temperature (gpuwm/core/microphysics.py:504 is the mp=8 precedent),
// WP-06's returns for `temperature >= 273.15f` -- so 273.15 K itself is
// WARM and every level is serviced exactly once.
//
// NEITHER OWNER CAN PROVE THAT ALONE.  It is proven jointly by
// test_every_level_across_the_freezing_seam_is_serviced_exactly_once and
// test_warm_section_number_sinks_are_evaluated_exactly_once_per_level in
// tests/test_thompson_aerosol_warm_gpu.py, which drive BOTH kernels over a
// 47-level sweep straddling 273.15 K (23 cold, 24 warm, the seam sampled
// exactly and at both float32 neighbours) and reconcile the three
// accumulators against this file's own ungated probe.  MEASURED on an
// RTX 5090: max |sum/reference - 1| = 1.6e-7 (ncten), 2.4e-7 (nwfaten),
// 1.9e-7 (nifaten).  A gap reads as 0 and a double count as 2.
//
// ---------------------------------------------------------------------------
// MEASURED AGREEMENT WITH WRF (RTX 5090, CuPy 14.1.1, nvrtc -std=c++17)
// ---------------------------------------------------------------------------
// Gated in tests/test_thompson_aerosol_warm_gpu.py against TWO Fortran
// oracles, both linking the SAME compiled module_mp_thompson.o build_aero.sh
// produces.
//
// (1) :2144-2232 and :2996-3019, over 12348 warm-rate and 11025 balance
//     states.  Max relative difference:
//
//   ncten balance, all 11025 rows, all six branches            EXACT
//   nu_c, nr_m3, nwfa_m3, nifa_m3                              EXACT
//   nc_m3, lamc, mvd_c, xDc, mvd_r, N0_r                       EXACT
//   prr_wau, pnc_wau, pnr_rcr                                  EXACT
//   lamr, prr_rcw, pnc_rcw, pna_rca, pnd_rcd                  <= 7e-16
//   pnr_wau                                                    1.2337e-07
//
//   THOSE MIDDLE THREE LINES USED TO READ
//     nc_m3, lamc, mvd_c, xDc, pnc_rcw                        <= 5e-7
//     prr_wau, pnr_wau, pnc_wau                               <= 1.6e-6
//   and this file attributed them, correctly, to thompson_aa_cloud_dist's
//   CUDA powf on an unpinned float32 chain in thompson_aerosol_common.cuh.
//   That helper is now correctly-rounded AND contraction-pinned, and the
//   residual it was carrying is gone.  Nothing in this file changed for it.
//   pnr_wau was the one survivor, and it was not rounding: its divisor
//   grouped the last two D0r factors first, where WRF's chain runs left to
//   right.  Repaired on lane/mp28fix-warm-network (thompson_aa_warm_rain_
//   rates), together with the collection rates' lamr = 1./ilamr(k).
//
// (2) :2402-2471 -- WRF's ALWAYS-RUN frozen collection block -- evaluated
//     ABOVE FREEZING, over 14400 states, by
//     tools/thompson_wrf461_oracle/probe_warm_frozen_aero.F90 (committed,
//     regenerable via build_probe_warm_frozen.sh).  This is the oracle the
//     six mp=28-only rates pnc_scw / pnc_gcw / pna_sca / pnd_scd / pna_gca /
//     pnd_gcd never had, and a melting layer is exactly the state they fire
//     in.  Max relative difference:
//
//   (RE-MEASURED on a fresh 14400-row regeneration of that oracle.)
//   xDs, smoe, ilamg, N0_g, xDg, vtg, Ef_sw, prs_scw,
//     pna_sca, pnd_scd                             BIT-EXACT 14400/14400
//   pna_gca                                                   3.301947e-16
//   pnd_gcd                                                   3.430941e-16
//   stoke_g                                                   2.202357e-07
//   Ef_gw                                                     2.293061e-07
//   prg_gcw                                                   2.812419e-07
//   pnc_scw                                                   3.199996e-07
//   pnc_gcw                                                   3.244877e-07
//   twet (Bolton iteration)                                   1.119592e-07
//   (previously quoted as <= 3.5e-16 / 4.4e-7 / 5.3e-7 / 1.3e-6 / 1.7e-6 /
//   1.2e-7; five of the eight tighten by 4-7x and _FROZEN_TOLERANCE moved
//   with them.)  Rebuilding this translation unit with EVERY remaining plain
//   powf replaced by thompson_aa_powf reproduces the identical eighteen
//   numbers, so none of the residual above is a power.
//
//     Restricted to the 13500 rows where mvd_c and nc_m3 are BIT-EXACT,
//     stoke_g and pnc_scw are bit-exact too and Ef_gw / prg_gcw / pnc_gcw
//     fall to 2.3e-7 / 2.8e-7 / 2.6e-7.  The remainder is
//     thompson_aa_cloud_dist's plain powf (WP-02, same residual as prr_wau
//     below), amplified because Ef_gw is 0.55*ALOG10(2.51*stoke_g), a
//     logarithm whose zero at stoke_g = 0.398 sits one part in a thousand
//     under the stoke_g >= 0.4 gate it is evaluated behind.  Ef_gw's
//     ABSOLUTE error never exceeds 6.8e-8, and 13702 of 14400 rows are
//     bit-exact.
//
//     The same program's REPLAY mode evaluates both blocks for a supplied
//     list of states; fed the entry columns of the nine relevant committed
//     fixtures (216 levels, warm and sub-freezing) it reports every rate of
//     :2144-2222 and :2402-2471 within 6.5e-6, with prr_wau bit-exact on 6
//     of aero-drop-evap's 7 rain-producing levels.
//
// The last two lines are NOT a property of this file.  They are entirely
// attributable to thompson_aa_cloud_dist in thompson_aerosol_common.cuh,
// which uses CUDA's powf (~2 ulp) where gfortran lowers REAL(4)**REAL(4) to
// glibc's correctly-rounded powf.  PROVEN by substitution: swapping those two
// powf calls for the header's own thompson_aa_powf makes nc_m3, lamc,
// mvd_c, xDc and prr_wau bit-exact on all 12348 rows.  This file deliberately
// does NOT carry a local copy of cloud_dist -- one droplet diagnosis, one
// owner -- so the residual is reported rather than worked around.
//
// Everything this file owns is contraction-pinned (thompson_aa_add/sub/mul/
// div) and uses thompson_aa_powf, because the oracle is compiled by
// `gfortran -O2` for baseline x86-64, which has NO fma instruction, while
// nvrtc defaults to --fmad=true.  That is not cosmetic: with contraction
// left on, `Nt_c_max - nc1d*rho` in the balance limiter keeps an exact
// residual where WRF gets exactly zero, and 80 of 11025 rows returned a
// spurious non-zero droplet tendency.
//
// ---------------------------------------------------------------------------
// THE BALANCE LIMITER RUNS EXACTLY ONCE
// ---------------------------------------------------------------------------
// thompson_aa_ncten_balance (:2996-3019) is a SEPARATE kernel on purpose.
// WRF applies it once per column after every ncten source; running it inside
// both the warm and the cold network would double-apply it, stay stable, and
// be wrong.  The adapter must launch it between the warm network and the
// saturation adjustment.
//
// The uniqueness is enforced two ways, both in
// tests/test_thompson_aerosol_warm_gpu.py.  Structurally: exactly one
// `extern "C" __global__ void thompson_aa_ncten_balance(` and exactly one
// `def launch_ncten_balance(` in the whole port.  Semantically, which is the
// stronger statement: this kernel's `ncten[idx] = tendency;` is the ONLY
// bare assignment to ncten in any aerosol translation unit -- every other
// write is a read-modify-write, because WRF's :2963-2994 accumulate while
// only :3007/:3011/:3014/:3018 back the tendency out against nc1d*rho.
//
// :3003 IS ASYMMETRIC AND THE ASYMMETRY IS LOAD-BEARING.  lamc is formed
// from the PROJECTED number xnc and the ENTRY cloud mass rc(k); the
// rediagnosis at :3006/:3010 then uses the POST-source mass xrc.  Because
// the clamp REPLACES lamc, rc's only job is to select the branch -- so a
// "consistency cleanup" that made both use xrc is observable exactly at the
// branch boundary and nowhere else.  test_the_limiter_selects_its_branch_
// from_rc_not_xrc pins one such state: rc = 0.6*rc_star clamps, xrc =
// 2.0*rc_star does not, and the cleanup would leave ncten untouched.

// ---------------------------------------------------------------------------
// Device helpers duplicated from thompson.cu (byte-frozen, no #include path).
// ---------------------------------------------------------------------------

// Bolton (1980) wet-bulb chain, thompson.cu:78-166.  WRF selects the
// rain/snow and rain/graupel collision direction from twet, not temp, so the
// iteration must be reproduced rather than approximated.
__device__ __forceinline__ float thompson_aa_theta_e(
    float pressure, float temperature, float mixing_ratio, float tlcl)
{
    const float rr = mixing_ratio + 1.0e-8f;
    const float power = 0.2854f * (1.0f - 0.28f * rr);
    const float dry_theta = temperature * thompson_aa_powf(100000.0f / pressure, power);
    const float p1 = 3.376f / tlcl - 0.00254f;
    const float p2 = rr * 1000.0f * (1.0f + 0.81f * rr);
    return dry_theta * thompson_aa_expf(p1 * p2);
}

__device__ __forceinline__ float thompson_aa_t_dew(
    float pressure, float mixing_ratio)
{
    const float rr = mixing_ratio + 1.0e-8f;
    const float esln = thompson_aa_logf(pressure * rr / (0.622f + rr));
    return (35.86f * esln - 4947.2325f) / (esln - 23.6837f);
}

__device__ __forceinline__ float thompson_aa_t_lcl(
    float temperature, float dewpoint)
{
    const float denominator = 1.0f / (dewpoint - 56.0f)
        + __fdiv_rn(thompson_aa_logf(temperature / dewpoint), 800.0f);
    return 1.0f / denominator + 56.0f;
}

__device__ __forceinline__ float thompson_aa_theta_wetb(float theta_e)
{
    const double c[7] = {
        -1.00922292e-10, -1.47945344e-8, -1.7303757e-6,
        -0.00012709, 1.15849867e-6, -3.518296861e-9,
        3.5741522e-12};
    const double d[7] = {
        0.0, -3.5223513e-10, -5.7250807e-8, -5.83975422e-6,
        4.72445163e-8, -1.13402845e-10, 8.729580402e-14};
    const float x = fminf(475.0f, theta_e);
    const double* coefficients = x <= 335.5f ? c : d;
    const double xd = (double)x;
    const double answer = coefficients[0] + xd * (
        coefficients[1] + xd * (coefficients[2] + xd * (
        coefficients[3] + xd * (coefficients[4] + xd * (
        coefficients[5] + xd * coefficients[6])))));
    return (float)(answer + 273.15);
}

__device__ __forceinline__ float thompson_aa_temperature_from_theta_e(
    float theta_e_lcl, float pressure)
{
    float guess = (theta_e_lcl - 0.5f
        * thompson_aa_powf(fmaxf(theta_e_lcl - 270.0f, 0.0f), 1.05f))
        * thompson_aa_powf(__fdiv_rn(pressure, 100000.0f), 0.2f);
    for (int iteration = 0; iteration < 100; ++iteration) {
        const float w1 = thompson_rslf(pressure, guess);
        const float w2 = thompson_rslf(pressure, guess + 1.0f);
        const float tenu = thompson_aa_theta_e(
            pressure, guess, w1, guess);
        const float tenup = thompson_aa_theta_e(
            pressure, guess + 1.0f, w2, guess + 1.0f);
        const float denominator = tenup - tenu;
        if (fabsf(denominator) < 1.0e-12f) break;
        const float correction = (theta_e_lcl - tenu) / denominator;
        guess += correction;
        if (fabsf(correction) < 0.01f) return guess;
    }
    return thompson_aa_theta_wetb(theta_e_lcl)
        * thompson_aa_powf(__fdiv_rn(pressure, 100000.0f), 0.286f);
}

__device__ __forceinline__ float thompson_aa_wet_bulb_temperature(
    float pressure, float temperature, float mixing_ratio)
{
    if (mixing_ratio / thompson_rslf(pressure, temperature) >= 0.999f) {
        return temperature;
    }
    const float dewpoint = fminf(
        temperature - 0.001f, thompson_aa_t_dew(pressure, mixing_ratio));
    const float tlcl = thompson_aa_t_lcl(temperature, dewpoint);
    const float theta_e_lcl = thompson_aa_theta_e(
        pressure, temperature, mixing_ratio, tlcl);
    return fminf(
        temperature,
        thompson_aa_temperature_from_theta_e(theta_e_lcl, pressure));
}

// ---------------------------------------------------------------------------
// SHARED HELPERS THIS FILE NO LONGER DEFINES
// ---------------------------------------------------------------------------
//
// thompson_aa_bound_rain_number (:4032-4046) and
// thompson_aa_entry_rain_distribution (:1878-1898 then :2144-2150) used to be
// defined HERE and, with different bodies, in thompson_aerosol_cold.cu.
// Separate cupy.RawModule translation units meant nvrtc never saw the
// conflict, so the two halves of the scheme drifted silently: the cold copy
// used plain powf and emitted no N0_r, this one was contraction-pinned and
// did.  Both now live once, in thompson_aerosol_common.cuh, which
// gpuwm/core/kernels/__init__.py::_EXTRA_HEADERS prepends to this translation
// unit; a surviving local copy is a hard nvrtc redefinition error.
//
// The promoted forms are THIS file's contraction-pinned bodies verbatim.
// PROVEN, not assumed: a single translation unit carrying the header's
// definitions alongside byte-copies of the pre-consolidation warm.cu bodies
// under *_legacy names agrees BITWISE on every output -- active, nr, lamr,
// mvd_r and N0_r for the distribution and the bounded number for the rain
// bound -- over 400,007 randomized (qr, nr, rho) states spanning qr in
// [1e-14, 1e-1], nr in [1e-8, 1e8] and rho in [0.05, 1.5], plus the exact R1
// and R2 edge values.  See
// test_promoted_header_helpers_are_bitwise_identical_to_the_local_copies in
// tests/test_thompson_aerosol_warm_gpu.py, which reruns that comparison
// against a golden copy of the deleted bodies so the header cannot drift away
// from the form this package's oracle numbers were measured on.

// thompson_aa_decade_index_double (thompson.cu:3083-3103) used to be defined
// HERE as well, byte-identical to thompson_aerosol_cold.cu's copy.  Wave 4
// promoted it into thompson_aerosol_common.cuh:799 for exactly the reason
// thompson_aa_entry_rain_distribution was promoted: separate cupy.RawModule
// translation units meant nvrtc could never see the two copies diverge.  A
// surviving local copy is now a hard nvrtc redefinition error, which is the
// intended enforcement.


// ---------------------------------------------------------------------------
// The nc-driven warm-rain rate set, module_mp_thompson.F:2144-2222.
// ---------------------------------------------------------------------------
//
// This is the physics WP-07 exists for, and it is factored into ONE device
// function so that thompson_aa_warm_source_network and the readback probe
// cannot drift apart: an oracle disagreement in the probe is by construction
// a disagreement in the network.
//
// Every quantity here traces to nc.  mp=8 froze the whole block into
// constants (thompson.cu:3174-3187, :3241): cloud_number 100.0e6f, the gamma
// pair 1.30767389e12f / 2.08767448e-9f, the mvd numerator 15.672f and the
// pnr_wau shape factor 12.0f.
//
// GATE NOTE.  WRF's L_qc(k) tests the MIXING RATIO against R1 (:1826), not
// the mass content rc = qc*rho.  thompson.cu gates the cloud-collection
// rates on rc > 1e-12 instead.  At low density (400 hPa here) the two
// disagree and WRF still evaluates prr_rcw; measured on the WP-07 Fortran
// oracle, 27 of 12348 rows.  mp=28 follows WRF.
struct ThompsonAaWarmRain {
    bool l_qc;
    bool rain_active;
    int nu_c;
    float nc_m3;
    float mvd_c;
    float xdc;
    float nwfa_m3;
    float nifa_m3;
    float nr_m3;
    float mvd_r;
    double lamc;
    double lamr;
    double n0_r;
    double prr_wau;
    double pnr_wau;
    double pnc_wau;
    double prr_rcw;
    double pnc_rcw;
    double pnr_rcr;
    double pna_rca;
    double pnd_rcd;
};

__device__ __forceinline__ void thompson_aa_warm_rain_rates(
    float qc_v, float nc_entry_v, float qr_v, float nr_entry_v,
    float nwfa_entry_v, float nifa_entry_v,
    float rho, float temp_k, float rho_factor, float inverse_dt,
    const double* __restrict__ t_efrw,
    ThompsonAaWarmRain* out)
{
    const float pi = THOMPSON_AA_PI;
    const float am_r = THOMPSON_AA_AM_R;
    const float tempc = temp_k - 273.15f;
    const float cloud_mass = fmaxf(0.0f, qc_v * rho);
    const float rain_mass = fmaxf(0.0f, qr_v * rho);

    out->l_qc = qc_v > THOMPSON_AA_R1;
    out->nu_c = 0;
    out->lamc = 0.0;
    out->xdc = 0.0f;
    out->mvd_c = THOMPSON_AA_D0C;          // WRF mvd_c(k) = D0c default
    out->nc_m3 = THOMPSON_AA_NC_FLOOR;     // WRF nc(k) = 2. if .not. L_qc
    out->prr_wau = 0.0;
    out->pnr_wau = 0.0;
    out->pnc_wau = 0.0;
    out->prr_rcw = 0.0;
    out->pnc_rcw = 0.0;
    out->pnr_rcr = 0.0;
    out->pna_rca = 0.0;
    out->pnd_rcd = 0.0;

    // Entry aerosol state, :1804-1806 (aer_init_opt = 0 branch).
    out->nwfa_m3 = thompson_aa_clamp_nwfa(nwfa_entry_v * rho);
    out->nifa_m3 = thompson_aa_clamp_nifa(nifa_entry_v * rho);

    // Entry droplet distribution :1826-1842, then the warm-rain block's own
    // re-selection of nu_c from the REDIAGNOSED nc, :2171-2175.
    //
    // THE nu_c STAGING RULE.  cloud_dist's out-parameters are WRF's ENTRY
    // values (:1832-1838), computed from the PRE-rediagnosis nc of :1829.
    // WRF throws them away after the :1834-1838 size clamp and recomputes
    // nu_c at :2170 from the nc it REDIAGNOSED at :1840, which is what
    // cloud_dist RETURNS.  Whenever the size clamp engages the two differ --
    // measured 13 of 70 states on the (nc_entry, rc) grid -- and between
    // nu_c = 3 and nu_c = 15 ccg(2,n)*ocg1(n) moves by 40.8x, so lamc,
    // mvd_c, Dc_g and pnr_wau are all grossly wrong if the entry value
    // leaks downstream.  thompson_aa_nu_c_working is thompson_aa_nu_c under
    // a name that records which nc is legal to pass; the entry values are
    // read into named locals ONLY so that the discard is explicit.
    if (out->l_qc) {
        int entry_nu_c = 0;
        double entry_lamc = 0.0;
        out->nc_m3 = thompson_aa_cloud_dist(
            cloud_mass, nc_entry_v, rho, &entry_nu_c, &entry_lamc);
        (void)entry_nu_c;    // :1832, deliberately NOT used after :1838
        (void)entry_lamc;    // :1833/:1836/:1838, likewise
        out->nu_c = thompson_aa_nu_c_working(out->nc_m3);
        out->xdc = fmaxf(
            THOMPSON_AA_D0C * 1.0e6f,
            thompson_aa_mul(
                thompson_aa_powf(
                    thompson_aa_div(cloud_mass,
                                    thompson_aa_mul(am_r, out->nc_m3)),
                    THOMPSON_AA_OBMR),
                1.0e6f));
        out->lamc = (double)thompson_aa_powf(
            thompson_aa_div(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(out->nc_m3, am_r),
                        THOMPSON_AA_CCG2[out->nu_c]),
                    THOMPSON_AA_OCG1[out->nu_c]),
                cloud_mass),
            THOMPSON_AA_OBMR);
        float mvd_c = (float)(
            (double)((3.0f + (float)out->nu_c) + 0.672f) / out->lamc);
#if defined(THOMPSON_AA_WRF39)
        out->mvd_c = mvd_c;              // fork :2096, no hold (audit T20)
#else
        out->mvd_c = fmaxf(
            THOMPSON_AA_D0C, fminf(mvd_c, THOMPSON_AA_D0R));
#endif
    }

    float rain_number;
    double rain_lambda;
    float rain_mvd;
    double rain_n0;
    out->rain_active = thompson_aa_entry_rain_distribution(
        qr_v, nr_entry_v, rho, &rain_number, &rain_lambda, &rain_mvd,
        &rain_n0);
    out->nr_m3 = rain_number;
    out->lamr = rain_lambda;
    out->mvd_r = rain_mvd;
    out->n0_r = rain_n0;

    // Berry & Reinhardt (1974) autoconversion, :2179-2194.
    //
    // Contraction-pinned.  Dc_b subtracts two nearly equal sixth powers, so
    // one fused multiply-add moves it by several float ulps and prr_wau by
    // ~1e-6 relative.  build_aero.sh compiles the oracle with plain
    // `gfortran -O2` on baseline x86-64, which has no FMA instruction, so
    // every WRF product and sum in this chain is separately rounded.
    if (cloud_mass > 0.01e-3f) {
        const int nu_c = out->nu_c;
        const float xdc = out->xdc;
        const float dcg = (float)(
            (double)thompson_aa_powf(
                thompson_aa_mul(THOMPSON_AA_CCG3[nu_c],
                                THOMPSON_AA_OCG2[nu_c]),
                THOMPSON_AA_OBMR)
            / out->lamc * 1.0e6);
        // WRF writes both products left-associated and un-grouped
        // (:2182-2183); the grouping is observable in float32.
        const float xdc3 = thompson_aa_mul(thompson_aa_mul(xdc, xdc), xdc);
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
        out->prr_wau = fmin(
            (double)(cloud_mass * inverse_dt),
            (double)thompson_aa_div(zeta, tau));
        // :2192.  The 12.0f at thompson.cu:3241 IS nu_c.
#if defined(THOMPSON_AA_WRF39)
        // fork :2111, am_r*nu_c*D0r*D0r*D0r (audit T6).
        out->pnr_wau = out->prr_wau
            / (double)thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(am_r, (float)nu_c), THOMPSON_AA_D0R),
                thompson_aa_mul(THOMPSON_AA_D0R, THOMPSON_AA_D0R));
#else
        // am_r*nu_c*10.*D0r*D0r*D0r is one REAL chain, left to right:
        // ((((am_r*nu_c)*10.)*D0r)*D0r)*D0r.  Grouping the last two D0r
        // factors first moved the divisor by one float32 unit and pnr_wau
        // with it on 111 of 365 active oracle cells.
        out->pnr_wau = out->prr_wau
            / (double)thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(
                            thompson_aa_mul(am_r, (float)nu_c), 10.0f),
                        THOMPSON_AA_D0R),
                    THOMPSON_AA_D0R),
                THOMPSON_AA_D0R);
#endif
        // :2193-2194.  NEW under mp=28: droplets consumed by autoconversion.
        out->pnc_wau = fmin(
            (double)(out->nc_m3 * inverse_dt),
            out->prr_wau / (double)thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(am_r, out->mvd_c), out->mvd_c),
                out->mvd_c));
    }

    if (!out->rain_active) return;

    // Seifert (1994) rain self-collection with Verlinde & Cotton break-up,
    // :2159-2166.
    if (rain_mvd > THOMPSON_AA_D0R) {
        const float efficiency = 1.0f
            - thompson_aa_expf(2300.0f * (rain_mvd - 1950.0e-6f));
        out->pnr_rcr = (double)(
            efficiency * 2.0f * rain_number * rain_mass);
    }
    if (!(rain_mvd > THOMPSON_AA_D0R)) return;

    // (lamr + fv_r)**(-cre(9)); cre(9) = mu_r + bv_r + 3 = 4 exactly, and
    // t1_qr_qc = PI*.25*av_r*crg(9) with crg(9) = WGAMMA(4) = 6 exactly.
    // :2198 / :2213: lamr = 1./ilamr(k), the reciprocal of the :2148
    // reciprocal, not the :2147 lamr itself.
    const double fall_kernel = thompson_aa_pow(
        thompson_aa_rain_lambda_reformed(rain_lambda) + 195.0, -4.0);
    const float t1_qr_qc = pi * 0.25f * 4854.0f * 6.0f;

    // Rain collecting cloud water, :2197-2208.
    if (out->l_qc && out->mvd_c > THOMPSON_AA_D0C) {
        const double first = 5.1164649614037726e-05;
        const double last = 0.004886186104779057;
        int rain_bin = 1 + (int)(100.0
            * thompson_aa_log((double)rain_mvd / first) / thompson_aa_log(last / first));
        rain_bin = min(rain_bin, 100);
#if defined(THOMPSON_AA_WRF39)
        const int cloud_bin = min(100, (int)(out->mvd_c * 1.0e6f));
#else
        const int cloud_bin = (int)(out->mvd_c * 1.0e6f);
#endif
        const float efficiency = (float)t_efrw[
            (rain_bin - 1) + 100 * (cloud_bin - 1)];
        out->prr_rcw = fmin(
            (double)(cloud_mass * inverse_dt),
            (double)(rho_factor * t1_qr_qc * efficiency * cloud_mass)
                * rain_n0 * fall_kernel);
        // :2205-2206.  NEW under mp=28.
        out->pnc_rcw = fmin(
            (double)(out->nc_m3 * inverse_dt),
            (double)(rho_factor * t1_qr_qc * efficiency * out->nc_m3)
                * rain_n0 * fall_kernel);
    }

    // Rain collecting aerosols, wet scavenging, :2211-2222.  WRF caps each
    // species against the FULL available aerosol; there is deliberately no
    // shared limiter across collector species.  The nbca (wif_input_opt==2)
    // branch at :2223-2231 is out of scope for this port.
    const float visco = (tempc >= 0.0f)
        ? (1.718f + 0.0049f * tempc) * 1.0e-5f
        : (1.718f + 0.0049f * tempc
           - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    const float ef_ccn = thompson_aa_eff_aero(
        rain_mvd, 0.04e-6f, visco, rho, temp_k,
        THOMPSON_AA_SPECIES_RAIN);
    out->pna_rca = fmin(
        (double)(out->nwfa_m3 * inverse_dt),
        (double)(rho_factor * t1_qr_qc * ef_ccn * out->nwfa_m3)
            * rain_n0 * fall_kernel);
    const float ef_in = thompson_aa_eff_aero(
        rain_mvd, 0.8e-6f, visco, rho, temp_k,
        THOMPSON_AA_SPECIES_RAIN);
    out->pnd_rcd = fmin(
        (double)(out->nifa_m3 * inverse_dt),
        (double)(rho_factor * t1_qr_qc * ef_in * out->nifa_m3)
            * rain_n0 * fall_kernel);
}

// ---------------------------------------------------------------------------
// WRF's REAL(4) GRAUPEL EXPONENTS.  THE DECIMAL LITERALS ARE NOT THEM.
// ---------------------------------------------------------------------------
//
// thompson_init stores bv_g in a REAL array and, because the optional ng
// argument is absent for both mp=8 and mp=28, writes bv_g_old = 0.89 into
// slot idx_bg1 = 5 (:149-150, :463-464, :79).  Every graupel exponent is then
// built from that REAL(4) value by REAL(4) arithmetic at :753-770:
//
//   bv_g(5)  = 0.89                     -> 0.8899999856948853
//   cge(6,5) = bm_g + mu_g + bv_g + 1.  -> 4.8899998664855957
//   cge(9,5) = mu_g + bv_g + 3.         -> 3.8899998664855957
//   cge(11,5)= 0.5*(bv_g + 5. + 2.*mu_g)-> 2.9449999332427979
//
// The Fortran oracle prints all four (WP07F_BVG / _CGE6 / _CGE9 / _CGE11) and
// tests/test_thompson_aerosol_warm_gpu.py compares the device values against
// those printed digits, so this block cannot drift into "close enough".
//
// WRITING thompson_aa_pow(ilamg, 3.89) INSTEAD IS A MEASURABLE ERROR, NOT A COSMETIC ONE.
// ilamg is below 1, so a LARGER exponent gives a SMALLER factor: the decimal
// 3.89 exceeds cge(9,5) by 1.335e-7 absolute, which lowers ilamg**cge(9) by
// |ln(ilamg)|*1.335e-7 -- 9.2e-7 at ilamg = 1e-3 and 1.5e-6 at ilamg = 1e-5,
// against a 2.0e-6 end-to-end fixture gate.  It scales prg_gcw, pnc_gcw,
// pna_gca and pnd_gcd, i.e. one cloud-water sink, one droplet sink and both
// graupel scavenging rates.  cge(11,5) does the same to the graupel melting
// and sublimation ventilation terms, which feed qr and RAINNC.
//
// mp=8's thompson.cu carries the decimal literals at every one of these
// sites.  It is byte-frozen and stays that way; mp=28 follows WRF.
#define THOMPSON_AA_WARM_BV_G   ((double)(0.0f + THOMPSON_AA_BV_G))
#define THOMPSON_AA_WARM_CGE6   ((double)((3.0f + 0.0f) \
                                          + THOMPSON_AA_BV_G + 1.0f))
#define THOMPSON_AA_WARM_CGE9   ((double)((0.0f + THOMPSON_AA_BV_G) + 3.0f))
#define THOMPSON_AA_WARM_CGE11  ((double)(0.5f * ((THOMPSON_AA_BV_G + 5.0f) \
                                                  + 2.0f * 0.0f)))
// WGAMMA(cge(n,5)); the oracle prints these as WP07F_CGG6 / _CGG9 / _CGG11.
#define THOMPSON_AA_WARM_CGG6   20.3632278f
#define THOMPSON_AA_WARM_CGG9   5.23476267f
#define THOMPSON_AA_WARM_CGG11  1.9021706581115723f
// (3.0 + mu_g + 0.672) as WRF forms it, in REAL(4): 3.6719999313354492, not
// the decimal 3.672 (:1926, :1930, :1934, :1937).  The _F form is the REAL(4)
// value used as a REAL(4) numerator; the _D form is the same bits widened for
// the REAL(4)/DOUBLE divisions at :1929 and :1938.
#define THOMPSON_AA_WARM_MVDG_F ((3.0f + 0.0f) + 0.672f)
#define THOMPSON_AA_WARM_MVDG_D ((double)THOMPSON_AA_WARM_MVDG_F)
#define THOMPSON_AA_D0S         300.0e-6f       // :226
// ogg3 = 1./cgg(3,1) with cgg(3,1) = WGAMMA(4) exactly 6 (oracle: WP07F_CGG3
// = 6.0, WP07F_OGG2 = 1.0, WP07F_OGG3 = 0.16666667163372040).
#define THOMPSON_AA_WARM_OGG3   (1.0f / 6.0f)


// ALOG10 lowers to glibc's log10f, which is correctly rounded; CUDA's
// log10f is ~2 ulp.  Ef_gw = 0.55*ALOG10(2.51*stoke_g) (:2426) is a
// LOGARITHM WITH A ZERO INSIDE the branch it is evaluated in -- 2.51*stoke_g
// reaches 1 at stoke_g = 0.398, one thousandth below the stoke_g >= 0.4 gate
// -- so a 2-ulp absolute error in the log10 is an unbounded RELATIVE error
// in Ef_gw and therefore in prg_gcw, pnc_gcw and the cloud-water
// competition they feed.  MEASURED on the 14400-row oracle: plain log10f
// leaves Ef_gw at 1.68e-6 and prg_gcw at 1.72e-6 relative; this form leaves
// them at 3.0e-7 and 5.2e-7.  Same idiom, same reason, as
// thompson_aerosol_common.cuh's thompson_aa_expf/logf_cr/powf_cr: the
// operand and the result are still float32, only the transcendental's
// internal rounding changes.
__device__ __forceinline__ float thompson_aa_log10f_cr(float x)
{
    // ALOG10 of a REAL(4) is the C library's log10f in the gfortran build;
    // thompson_aa_log10f is WOOF's own word for it (thompson_aerosol_libm.cuh).
    return thompson_aa_log10f(x);
}


// :1927 / :1932 / :1936.  ng(k) = cgg(2,1)*ogg3*rg(k)*lamg**bm_g / am_g.
// cgg(2,1) is exactly 1, so the REAL(4) prefix is ogg3*rg; the lamg**bm_g
// factor and the /am_g division happen in DOUBLE and the result lands back
// in a REAL(4).  mp=8's kernel folds the /am_g into the REAL(4) prefix
// instead, which rounds twice in float32 where WRF rounds once.
__device__ __forceinline__ float thompson_aa_graupel_number_from_lambda(
    float graupel_mass, double lamg, float am_g)
{
    return (float)(
        (double)thompson_aa_mul(THOMPSON_AA_WARM_OGG3, graupel_mass)
        * thompson_aa_pow(lamg, 3.0) / (double)am_g);
}


// ---------------------------------------------------------------------------
// WRF's ALWAYS-RUN frozen collection block, module_mp_thompson.F:2402-2471.
// ---------------------------------------------------------------------------
//
// iiwarm is a PARAMETER .false. (:59), so the frozen-species loop opened at
// :2239 carries NO temperature guard.  Snow and graupel collect cloud water
// and scavenge aerosol at ambient-warm levels too -- which is to say IN A
// MELTING LAYER, the most common mixed-phase state in a real forecast.  Six
// of the rates below are new in mp=28 and have no mp=8 counterpart whose
// validation they could inherit:
//
//   pnc_scw :2411-2412   pnc_gcw :2436-2437                    -> ncten
//   pna_sca :2444-2446   pna_gca :2462-2467                    -> nwfaten
//   pnd_scd :2448-2450   pnd_gcd :2468-2471                    -> nifaten
//
// The two MASS companions prs_scw (:2407-2410) and prg_gcw (:2433-2435) are
// returned alongside them because WRF diagnoses all eight from one snow and
// one graupel distribution; splitting them would let the number rate and the
// mass rate see different distributions.
//
// SHARED VERBATIM with thompson_aa_probe_warm_frozen_rates at the bottom of
// this file, so the Fortran oracle
// tools/thompson_wrf461_oracle/probe_warm_frozen_aero.F90 gates the network's
// own arithmetic rather than a second transcription of it.
//
// L_qs AND L_qg TEST THE MIXING RATIO, NOT THE MASS CONTENT.  WRF's gates
// are `qs1d(k) .gt. R1` (:1906) and `qg1d(k) .gt. R1` (:1915), and the
// species content is then rs = qs1d*rho or the placeholder R1 (:1911, :1945).
// thompson.cu tests qs*rho instead, which at rho != 1 disagrees over
// qs in (R1, R1/rho] -- a window up to 18 percent wide around 1e-12 kg/kg.
// This function follows WRF, and the oracle ladder carries qs = qg = 1.1e-12
// at rho ~ 0.86 so the choice is measured rather than asserted.
//
// THE GRAUPEL SLOPE IS DIAGNOSED EVEN WITH NO GRAUPEL.  :2135-2139 is its own
// unguarded kte->kts loop, so on a graupel-free level WRF still forms
// lamg / ilamg / N0_g from the rg = R1, ng = R2 placeholders, giving
// N0_g ~ 1.08e-3 rather than zero.  Nothing consumes it there -- every rate
// gate is rg >= r_g(1) = 1e-6 or L_qg -- but reproducing it is what lets the
// oracle comparison cover all 14400 rows instead of "all rows with graupel".
struct ThompsonAaFrozen {
    bool l_qs;
    bool l_qg;
    float rs;        // qs1d*rho, or the R1 placeholder (:1908, :1911)
    float rg;        // qg1d*rho, or the R1 placeholder (:1918, :1945)
    float smob;      // rs*oams == smo2, because bm_s is exactly 2 (:2033)
    float smo0;      // snow number, power-law moment (:2054)
    float smo1;      // 1st moment, melting (:2067)
    float smoc;      // bm_s+1 moment, diameter (:2080)
    float smoe;      // bv_s+2 moment, riming (:2101)
    float smof;      // 1+(1+bv_s)/2 moment, ventilation (:2114)
    float xds;       // smoc/smob (:2245)
    float ng_m3;
    float mvd_g;
    float xdg;
    float vtg;
    float stoke_g;
    float ef_sw;
    float ef_gw;
    double lamg;
    double ilamg;
    double n0_g;
    double prs_scw;
    double pnc_scw;
    double prg_gcw;
    double pnc_gcw;
    double pna_sca;
    double pnd_scd;
    double pna_gca;
    double pnd_gcd;
};

__device__ __forceinline__ void thompson_aa_frozen_collect_rates(
    float qs_v, float qg_v, float graupel_number_per_kg_v,
    bool l_qc, float cloud_mass, float cloud_number, float cloud_mvd,
    float nwfa_m3, float nifa_m3,
    float rho, float temp_k, float wet_bulb, float rho_factor,
    float inverse_dt, const double* __restrict__ snow_cloud_efficiency,
    ThompsonAaFrozen* out)
{
    const float pi = THOMPSON_AA_PI;
#if defined(THOMPSON_AA_WRF39)
    const float am_g = thompson_aa_wrf39_am_g();
#else
    const float am_g = pi * 400.0f / 6.0f;
#endif
    const float tempc = temp_k - 273.15f;
    const float t1_qs_qc = pi * 0.25f * THOMPSON_AA_AV_S;
    const float t1_qg_qc = pi * 0.25f * THOMPSON_AA_AV_G
        * THOMPSON_AA_WARM_CGG9;

    // :1906-1913 and :1915-1949.
    out->l_qs = qs_v > THOMPSON_AA_R1;
    out->l_qg = qg_v > THOMPSON_AA_R1;
    const float snow_mass = out->l_qs
        ? fmaxf(0.0f, qs_v * rho) : THOMPSON_AA_R1;
    const float graupel_mass = out->l_qg
        ? fmaxf(0.0f, qg_v * rho) : THOMPSON_AA_R1;
    out->rs = snow_mass;
    out->rg = graupel_mass;
    out->smob = 0.0f;
    out->smo0 = 0.0f;
    out->smo1 = 0.0f;
    out->smoc = 0.0f;
    out->smoe = 0.0f;
    out->smof = 0.0f;
    out->xds = 0.0f;
    out->ng_m3 = 0.0f;
    out->mvd_g = 0.0f;
    out->xdg = 0.0f;
    out->vtg = 0.0f;
    out->stoke_g = 0.0f;
    out->ef_sw = 0.0f;
    out->ef_gw = 0.0f;
    out->lamg = 1.0;
    out->ilamg = 1.0;
    out->n0_g = 0.0;
    out->prs_scw = 0.0;
    out->pnc_scw = 0.0;
    out->prg_gcw = 0.0;
    out->pnc_gcw = 0.0;
    out->pna_sca = 0.0;
    out->pnd_scd = 0.0;
    out->pna_gca = 0.0;
    out->pnd_gcd = 0.0;

    // --- snow moments, :2029-2114, and xDs at :2245 ----------------------
    if (out->l_qs) {
        const float tc0 = fminf(-0.1f, tempc);
        out->smob = snow_mass * (1.0f / 0.069f);
        out->smo0 = thompson_field_a(tc0, 0.0f)
            * thompson_aa_powf(out->smob, thompson_field_b(tc0, 0.0f));
        out->smo1 = thompson_field_a(tc0, 1.0f)
            * thompson_aa_powf(out->smob, thompson_field_b(tc0, 1.0f));
        out->smof = thompson_field_a(tc0, 1.775f)
            * thompson_aa_powf(out->smob, thompson_field_b(tc0, 1.775f));
        out->smoc = thompson_field_a(tc0, 3.0f)
            * thompson_aa_powf(out->smob, thompson_field_b(tc0, 3.0f));
        out->smoe = thompson_field_a(tc0, 2.55f)
            * thompson_aa_powf(out->smob, thompson_field_b(tc0, 2.55f));
        out->xds = out->smoc / out->smob;
    }

    // --- graupel distribution, :1915-1949 then :2135-2139 ----------------
#if defined(THOMPSON_AA_WRF39)
    // fork :2031-2054: the slope from the column intercept the entry pass
    // left in graupel_number_per_kg_v (an N0_exp, not a number), on every
    // level (rg = R1 placeholder where L_qg fails), as the fork's loop runs.
    thompson_aa_wrf39_graupel_slope(graupel_number_per_kg_v, graupel_mass,
                                    &out->lamg, &out->ilamg, &out->n0_g);
    out->ng_m3 = 0.0f;
    (void)am_g;
    if (false) {
        float ng = THOMPSON_AA_R2;
#else
    float ng = THOMPSON_AA_R2;              // :1946, the L_qg = .false. value
    if (out->l_qg) {
#endif
        ng = fmaxf(THOMPSON_AA_R2, graupel_number_per_kg_v * rho);
        double lamg;
        // :1924-1928.  A WHOLE BRANCH mp=8's kernel has no counterpart for.
        // When the incoming number is at (or under) R2, WRF DISCARDS it and
        // re-seeds the distribution from a 1.5 mm mean-volume diameter.
        // Without this, a graupel-bearing level whose number field is zero
        // gets lamg from ng = R2, which for rg ~ 1e-4 lands mvd_g near a
        // METRE, trips the 25.4 mm clamp, and ends up three decades away
        // from the distribution WRF actually integrates.
        if (ng <= THOMPSON_AA_R2) {
            out->mvd_g = 1.5e-3f;
            lamg = (double)(THOMPSON_AA_WARM_MVDG_F / out->mvd_g);
            ng = thompson_aa_graupel_number_from_lambda(
                graupel_mass, lamg, am_g);
        }
        lamg = (double)thompson_aa_powf(
            am_g * 6.0f * ng / graupel_mass, 1.0f / 3.0f);
        out->mvd_g = (float)(THOMPSON_AA_WARM_MVDG_D / lamg);
        if (out->mvd_g > 25.4e-3f) {
            out->mvd_g = 25.4e-3f;
            lamg = (double)(THOMPSON_AA_WARM_MVDG_F / out->mvd_g);
            ng = thompson_aa_graupel_number_from_lambda(
                graupel_mass, lamg, am_g);
        } else if (out->mvd_g < THOMPSON_AA_D0R) {
            out->mvd_g = THOMPSON_AA_D0R;
            lamg = (double)(THOMPSON_AA_WARM_MVDG_F / out->mvd_g);
            ng = thompson_aa_graupel_number_from_lambda(
                graupel_mass, lamg, am_g);
        }
    }
#if !defined(THOMPSON_AA_WRF39)
    out->ng_m3 = ng;
    // :2135-2139.  WRF's own kte->kts loop, UNGUARDED: it re-derives lamg
    // from the POST-clamp ng on every level, graupel-free ones included
    // (where it runs on the rg = R1 / ng = R2 placeholders).  Every
    // downstream consumer -- riming, scavenging, melting, sublimation, the
    // collision-table intercept bin -- reads ilamg(k) and N0_g(k) from THIS
    // derivation, not from the :1929 local.
    {
        const double lamg_final = (double)thompson_aa_powf(
            am_g * 6.0f * ng / graupel_mass, 1.0f / 3.0f);
        out->lamg = lamg_final;
        out->ilamg = 1.0 / lamg_final;
        out->n0_g = (double)ng * lamg_final;
    }
#endif

    // visco(k), :1991-1996.  The sub-freezing branch carries an extra
    // -1.2e-5*tempc**2 term.  thompson.cu's warm network drops it because
    // that kernel only ever sees tempc >= 0, where the two forms are
    // identical -- but WRF's frozen block is NOT temperature gated, and the
    // Stokes number that selects Ef_gw is proportional to 1/visco.  At
    // tempc = -33 K the unbranched form is 0.86 percent high, which moved
    // prg_gcw and pnc_gcw by 2.5e-2 on the aero-cold-overlap entry column.
    // Using WRF's branch here is INERT for every warm level and correct for
    // the rest.
    const float viscosity_aero = (tempc >= 0.0f)
        ? (1.718f + 0.0049f * tempc) * 1.0e-5f
        : (1.718f + 0.0049f * tempc
           - 1.2e-5f * tempc * tempc) * 1.0e-5f;

    // --- snow and graupel collecting cloud water, :2402-2440 -------------
#if defined(THOMPSON_AA_WRF39)
    // fork :205, :864: D0s = 200 microns and the snow bins built from it;
    // fork :271-284: r_s(1) = r_g(1) = 1.e-5.
    const float snow_d0 = THOMPSON_AA_WRF39_D0S;
    const double snow_d0_d = (double)THOMPSON_AA_WRF39_D0S;
    const float frozen_first = THOMPSON_AA_WRF39_R_SG_FIRST;
#else
    const float snow_d0 = THOMPSON_AA_D0S;
    const double snow_d0_d = 300.0e-6;
    const float frozen_first = THOMPSON_AA_R2;
#endif
    if (l_qc && cloud_mvd > THOMPSON_AA_D0C) {
        if (out->xds > snow_d0) {
            const double diameter_ratio = 0.02 / snow_d0_d;
            const double log_ratio = thompson_aa_log(diameter_ratio);
            const double first = snow_d0_d * thompson_aa_exp(0.5 / 100.0 * log_ratio);
            const double last = snow_d0_d * thompson_aa_exp(99.5 / 100.0 * log_ratio);
            int snow_bin = 1 + (int)(100.0
                * thompson_aa_log((double)out->xds / first) / thompson_aa_log(last / first));
            snow_bin = max(1, min(snow_bin, 100));
            const int cloud_bin = max(1, min(
                (int)(cloud_mvd * 1.0e6f), 100));
            out->ef_sw = (float)snow_cloud_efficiency[
                (snow_bin - 1) + 100 * (cloud_bin - 1)];
            out->prs_scw = fmin(
                (double)(cloud_mass * inverse_dt),
                (double)(rho_factor * t1_qs_qc
                    * out->ef_sw * cloud_mass * out->smoe));
            // :2411-2412.  NEW under mp=28.
            out->pnc_scw = fmin(
                (double)(cloud_number * inverse_dt),
                (double)(rho_factor * t1_qs_qc
                    * out->ef_sw * cloud_number * out->smoe));
        }
        // :2415.  Note >= here and > for the scavenging gate at :2460.
        if (graupel_mass >= frozen_first) {
            // xDg = (bm_g + mu_g + 1.)*ilamg; the REAL(4) sum is exactly 4.
            out->xdg = (float)(4.0 * out->ilamg);
            out->vtg = (float)(
                (double)(rho_factor * THOMPSON_AA_AV_G
                         * THOMPSON_AA_WARM_CGG6 * THOMPSON_AA_WARM_OGG3)
                * thompson_aa_pow(out->ilamg, THOMPSON_AA_WARM_BV_G));
            out->stoke_g = cloud_mvd * cloud_mvd * out->vtg * 1000.0f
                / (9.0f * viscosity_aero * out->xdg);
            float efficiency = 0.0f;
            if (out->stoke_g >= 0.4f && out->stoke_g <= 10.0f) {
                efficiency = thompson_aa_mul(
                    0.55f,
                    thompson_aa_log10f_cr(
                        thompson_aa_mul(2.51f, out->stoke_g)));
            } else if (out->stoke_g > 10.0f) {
                efficiency = 0.77f;
            }
#if defined(THOMPSON_AA_WRF39)
            // fork :2319: only graupel larger than D0g = 250 microns
            // collects cloud water, at full efficiency (audit T13).
            (void)wet_bulb;
            if (!(out->xdg > THOMPSON_AA_WRF39_D0G)) efficiency = 0.0f;
#else
            if (wet_bulb > 273.15f) efficiency *= 0.1f;
#endif
            out->ef_gw = efficiency;
            const double fall_kernel = thompson_aa_pow(
                out->ilamg, THOMPSON_AA_WARM_CGE9);
            // WRF deliberately does not cap prg_gcw by itself.  It diagnoses
            // the raw collection rate here and scales it together with every
            // other cloud-water sink in the later cloud conservation pass.
            // Pre-capping this term changes that competition, routing cloud
            // water away from graupel and toward rain.
            out->prg_gcw =
                (double)(rho_factor * t1_qg_qc * efficiency * cloud_mass)
                    * out->n0_g * fall_kernel;
            // :2436-2437.  pnc_gcw IS capped, unlike prg_gcw.
            out->pnc_gcw = fmin(
                (double)(cloud_number * inverse_dt),
                (double)(rho_factor * t1_qg_qc * efficiency * cloud_number)
                    * out->n0_g * fall_kernel);
        }
    }

    // --- snow collecting aerosols, wet scavenging, :2443-2450 ------------
    // Gate is rs > r_s(1) = 1e-6, NOT the D0s riming gate above, and it sits
    // OUTSIDE WRF's L_qc/mvd_c branch: aerosol is scavenged from a
    // cloud-free level too.
    if (snow_mass > frozen_first) {
        const float ef_ccn = thompson_aa_eff_aero(
            out->xds, 0.04e-6f, viscosity_aero, rho, temp_k,
            THOMPSON_AA_SPECIES_SNOW);
        out->pna_sca = fmin(
            (double)(nwfa_m3 * inverse_dt),
            (double)(rho_factor * t1_qs_qc * ef_ccn * nwfa_m3 * out->smoe));
        const float ef_in = thompson_aa_eff_aero(
            out->xds, 0.8e-6f, viscosity_aero, rho, temp_k,
            THOMPSON_AA_SPECIES_SNOW);
        out->pnd_scd = fmin(
            (double)(nifa_m3 * inverse_dt),
            (double)(rho_factor * t1_qs_qc * ef_in * nifa_m3 * out->smoe));
    }

    // --- graupel collecting aerosols, :2460-2471 -------------------------
    if (graupel_mass > frozen_first) {
        out->xdg = (float)(4.0 * out->ilamg);
        const double fall_kernel = thompson_aa_pow(out->ilamg, THOMPSON_AA_WARM_CGE9);
        const float ef_ccn = thompson_aa_eff_aero(
            out->xdg, 0.04e-6f, viscosity_aero, rho, temp_k,
            THOMPSON_AA_SPECIES_GRAUPEL);
        out->pna_gca = fmin(
            (double)(nwfa_m3 * inverse_dt),
            (double)(rho_factor * t1_qg_qc * ef_ccn * nwfa_m3)
                * out->n0_g * fall_kernel);
        const float ef_in = thompson_aa_eff_aero(
            out->xdg, 0.8e-6f, viscosity_aero, rho, temp_k,
            THOMPSON_AA_SPECIES_GRAUPEL);
        out->pnd_gcd = fmin(
            (double)(nifa_m3 * inverse_dt),
            (double)(rho_factor * t1_qg_qc * ef_in * nifa_m3)
                * out->n0_g * fall_kernel);
    }
}


// ---------------------------------------------------------------------------
// ONE LEVEL AT OR ABOVE 0 C, WRF v4.6.1, IN WRF'S ARITHMETIC ORDER.
// ---------------------------------------------------------------------------
//
// The same shared transcription as the cold network's level function
// (thompson_aerosol_cold.cu, thompson_aa_cold_level_wrf461; the
// ThompsonAaLevel block of thompson_aerosol_common.cuh), for a level whose
// ENTRY temperature is at or above 273.15 K: the entry state, the warm-rain
// loop, the frozen-species block's ambient-warm branch (snow and graupel
// riming and scavenging, rain collecting snow and graupel, melting,
// sublimation), the conservation limiters and the tendencies.  twet is WRF's
// wet-bulb temperature (:2004-2013).  No ice process runs here; the cloud
// ice balance of a warm level is the cold network's (it owns every level's
// ice).
//
// THE BREAKAGE THIS PREVENTS.  The previous warm-level body formed the
// melting rates as one REAL product where WRF multiplies two (prr_sml) or
// regrouped them (prr_gml), applied the collision enhancement through a
// DOUBLE 4218*olfus*(twet-T_0), took the melting/sublimation vapour limiter
// and the snow and graupel conservation in DOUBLE where WRF's sump and ratio
// are REAL, kept the paired rain/snow and rain/graupel transfers in DOUBLE
// where WRF rounds them through the REAL ratio, and formed the temperature
// change as one DOUBLE expression where WRF's tten is a REAL accumulator.
// The host CPU comparison against WRF v4.6.1 found prr_sml, pnr_sml,
// prr_gml, pnr_gml, prg_gde, png_gde, prs_sde, prr_rcs, prs_rcs, prr_rcg
// and prg_rcg differing on up to every active warm level.
__device__ __forceinline__ void thompson_aa_warm_level_wrf461(
    int idx,
    float* __restrict__ qc, float* __restrict__ qr, float* __restrict__ nr,
    float* __restrict__ qs, float* __restrict__ qg,
    float* __restrict__ graupel_number_per_kg,
    float* __restrict__ graupel_melt_marker,
    float* __restrict__ snow_melt_marker,
    float* __restrict__ temperature, const float* __restrict__ pressure,
    float* __restrict__ qv,
    const float* __restrict__ nc_entry, const float* __restrict__ nwfa_entry,
    const float* __restrict__ nifa_entry,
    float* __restrict__ ncten, float* __restrict__ nwfaten,
    float* __restrict__ nifaten,
    const double* __restrict__ t_Efrw, const ThompsonAaFrozenTables* T,
    float* __restrict__ qcten, float* __restrict__ qrten,
    float* __restrict__ nrten, float* __restrict__ qvten_acc,
    float* __restrict__ tten_acc, float* __restrict__ qsten_acc,
    float* __restrict__ qgten_acc, float* __restrict__ ngten_acc,
    bool twet_reformed, float dt)
{
    const float odts = aaf_fd(1.0f, dt);
    const float qc1d = qc[idx] > AAF_R1 ? qc[idx] : 0.0f;
    const float qr1d = qr[idx] > AAF_R1 ? qr[idx] : 0.0f;
    const float nr1d = qr[idx] > AAF_R1 ? nr[idx] : 0.0f;
    const float qs1d = qs[idx] > AAF_R1 ? qs[idx] : 0.0f;
    const float qg1d = qg[idx] > AAF_R1 ? qg[idx] : 0.0f;
    const float ng1d = qg[idx] > AAF_R1 ? graupel_number_per_kg[idx] : 0.0f;

    ThompsonAaLevel L;
    thompson_aa_wrf_zero_rates(&L);
    // No ice process runs on a warm level; ri/ni enter as WRF's R1/R2.
    thompson_aa_wrf_entry(&L, temperature[idx], pressure[idx], qv[idx],
                          qc[idx], nc_entry[idx], 0.0f, 0.0f, qr[idx],
                          nr[idx], qs[idx], qg[idx],
                          graupel_number_per_kg[idx], nwfa_entry[idx],
                          nifa_entry[idx]);
    // :2001-2012: twet(k) = temp(k), re-formed from the wet bulb only at
    // or below k_melting (thompson_aa_entry_warm_mask).
    thompson_aa_wrf_level_state(
        &L, twet_reformed ? thompson_aa_wrf_twet(L.pres, L.temp, L.qv)
                          : L.temp);
    thompson_aa_wrf_warm_loop(&L, odts, t_Efrw);
    thompson_aa_wrf_frozen_block(&L, dt, odts, T);
    // HOST READBACK ANCHOR: the warm level's frozen block is complete.
    thompson_aa_wrf_conserve(&L, odts);
    // HOST READBACK ANCHOR: every warm-level rate final.

    // The two held branch decisions the fallout reads.
    snow_melt_marker[idx] = L.prr_sml > 0.0 ? 1.0f : 0.0f;
    graupel_melt_marker[idx] = L.prr_gml > 0.0 ? 1.0f : 0.0f;

    const float orho = L.orho;
    const double dorho = (double)orho;
    nwfaten[idx] = (float)aaf_ds((double)nwfaten[idx], aaf_dm(aaf_da(aaf_da(
        aaf_da(L.pna_rca, L.pna_sca), L.pna_gca), L.pni_iha), dorho));
    nifaten[idx] = (float)aaf_ds((double)nifaten[idx], aaf_dm(aaf_da(aaf_da(
        L.pnd_rcd, L.pnd_scd), L.pnd_gcd), dorho));
    nifaten[idx] = (float)aaf_ds((double)nifaten[idx],
                                 aaf_dm(L.pni_inu, dorho));
    const double qv_rate = aaf_dm(aaf_ds(aaf_ds(aaf_ds(aaf_ds(aaf_ds(
        -L.pri_inu, L.pri_iha), L.pri_ide), L.prs_ide), L.prs_sde),
        L.prg_gde), dorho);
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
    const double qr_rate = aaf_dm(aaf_ds(aaf_ds(aaf_ds(aaf_da(aaf_da(aaf_da(
        aaf_da(aaf_da(L.prr_wau, L.prr_rcw), L.prr_sml), L.prr_gml),
        L.prr_rcs), L.prr_rcg), L.prg_rfz), L.pri_rfz), L.prr_rci), dorho);
    const double nr_rate = aaf_dm(aaf_ds(aaf_da(aaf_da(L.pnr_wau, L.pnr_sml),
        L.pnr_gml), aaf_da(aaf_da(aaf_da(aaf_da(aaf_da(L.pnr_rfz, L.pnr_rcr),
        L.pnr_rcg), L.pnr_rcs), L.pnr_rci), L.pni_rfz)), dorho);
    if (nrten != nullptr) {
        qrten[idx] = (float)aaf_da((double)qrten[idx], qr_rate);
        nrten[idx] = (float)aaf_da((double)nrten[idx], nr_rate);
        float rain_mass_tendency = qrten[idx];
        float rain_number_tendency = nrten[idx];
        thompson_aa_rain_balance_tendency(qr1d, nr1d, &rain_mass_tendency,
            &rain_number_tendency, L.rho, orho, odts, dt);
        qrten[idx] = rain_mass_tendency;
        nrten[idx] = rain_number_tendency;
    } else {
        float qrten_l = (float)qr_rate;
        float nrten_l = (float)nr_rate;
        thompson_aa_rain_balance_tendency(qr1d, nr1d, &qrten_l, &nrten_l,
                                          L.rho, orho, odts, dt);
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
        graupel_number_per_kg[idx] = aaf_fa(ng1d, aaf_fm(ngten, dt));
    }

    // :3176-3180, the at-or-above-0 C temperature tendency.  lfus = 334000.
    const double t_rate = aaf_dm(aaf_dm(aaf_da(
        aaf_dm((double)aaf_fm(334000.0f, L.ocp), aaf_ds(aaf_ds(aaf_ds(
            -L.prr_sml, L.prr_gml), L.prr_rcg), L.prr_rcs)),
        aaf_dm((double)aaf_fm(AAF_LSUB, L.ocp), aaf_da(L.prs_sde,
                                                     L.prg_gde))),
        dorho), 1.0);
    thompson_aa_wrf_apply_vapor_heat(idx, L.temp, qv, temperature,
                                     qvten_acc, tten_acc, qv_rate, t_rate,
                                     dt);
}


// ---------------------------------------------------------------------------
// The warm source network.
// ---------------------------------------------------------------------------

extern "C" __global__ void thompson_aa_warm_source_network(
    float* __restrict__ qc,
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ graupel_number_per_kg,
    float* __restrict__ graupel_melt_marker,
    float* __restrict__ snow_melt_marker,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    // Frozen mp=28 entry state (per kilogram).  Never written.
    const float* __restrict__ nc_entry,
    const float* __restrict__ nwfa_entry,
    const float* __restrict__ nifa_entry,
    // Shared scratch accumulators (per kilogram per second).  Read-modify-
    // write; zeroed once at adapter entry and applied once by the terminal
    // state kernel.
    float* __restrict__ ncten,
    float* __restrict__ nwfaten,
    float* __restrict__ nifaten,
    const double* __restrict__ rain_cloud_efficiency,
    const double* __restrict__ snow_cloud_efficiency,
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
    // WRF's qcten, exactly as in thompson_aa_cold_network: null applies the
    // cloud sink to qc in place, non-null adds it to the accumulator and
    // leaves qc the read-only entry cloud.
    float* __restrict__ qcten,
    // WRF's qrten / nrten: both null keeps the in-place rain apply and its
    // in-place size bound; both given adds the sources and runs the
    // :3070-3091 balance in tendency form (see thompson_aa_cold_network).
    float* __restrict__ qrten,
    float* __restrict__ nrten,
    // WRF's qvten / tten, both given or both null, exactly as in
    // thompson_aa_cold_network: given, the sources are added in WRF's form
    // (:2982, :3174-3179), qv stays qv1d and temperature becomes t1d +
    // DT*tten.
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
#if !defined(THOMPSON_AA_HAIL)
    (void)graupel_volume;
    (void)qgten;
    (void)ngten;
    (void)qbten;
#endif
    // WRF's cold source kernel already owns every sub-freezing level.  This
    // companion owns the remaining ambient-warm levels, while still using
    // WRF's wet-bulb temperature to select the inverse (melting) collision
    // branches.  All rates below read one immutable incoming state and are
    // applied only after the WRF-ordered category conservation passes.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    // 1: warm, twet re-formed; 2: warm at exactly 273.15 K above the
    // melting level, twet = temp (thompson_aa_entry_warm_mask); a plain
    // >= 273.15 mask (the fork's) carries 1 everywhere it is warm.
    const float entry_mark = graupel_melt_marker[idx];
    const bool entry_warm = entry_mark != 0.0f;
    // These in-call decisions must never retain a prior invocation's value.
    // The graupel buffer arrives seeded with the entry-temperature mask and
    // becomes the held prr_gml marker after that one read.
    graupel_melt_marker[idx] = 0.0f;
    snow_melt_marker[idx] = 0.0f;
    if (!entry_warm) return;

#if !defined(THOMPSON_AA_WRF39) && !defined(THOMPSON_AA_HAIL)
    {
        ThompsonAaFrozenTables T;
        T.tpi_ide = nullptr; T.tps_iaus = nullptr; T.tni_iaus = nullptr;
        T.tcs_racs1 = tcs_racs1; T.tmr_racs1 = tmr_racs1;
        T.tcs_racs2 = tcs_racs2; T.tmr_racs2 = tmr_racs2;
        T.tcr_sacr1 = tcr_sacr1; T.tms_sacr1 = tms_sacr1;
        T.tcr_sacr2 = tcr_sacr2; T.tms_sacr2 = tms_sacr2;
        T.tnr_racs1 = tnr_racs1; T.tnr_racs2 = tnr_racs2;
        T.tnr_sacr1 = tnr_sacr1; T.tnr_sacr2 = tnr_sacr2;
        T.tcg_racg = tcg_racg; T.tmr_racg = tmr_racg;
        T.tcr_gacr = tcr_gacr; T.tnr_racg = tnr_racg;
        T.tnr_gacr = tnr_gacr;
        // The freezing tables are read below 0 C only.
        T.tpi_qrfz = nullptr; T.tni_qrfz = nullptr;
        T.tpg_qrfz = nullptr; T.tnr_qrfz = nullptr;
        T.tpi_qcfz = nullptr; T.tni_qcfz = nullptr;
        T.t_Efsw = snow_cloud_efficiency;
        thompson_aa_warm_level_wrf461(
            idx, qc, qr, nr, qs, qg, graupel_number_per_kg,
            graupel_melt_marker, snow_melt_marker, temperature, pressure,
            qv, nc_entry, nwfa_entry, nifa_entry, ncten, nwfaten, nifaten,
            rain_cloud_efficiency, &T, qcten, qrten, nrten, qvten, tten,
            qsten, qgten, ngten, entry_mark != 2.0f, dt);
        return;
    }
#endif
    // Below: the WRF 3.9 fork generation (THOMPSON_AA_WRF39) only.

    const float temp0 = temperature[idx];
    const float pressure0 = pressure[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure0
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float inverse_dt = 1.0f / dt;
    const float tempc = temp0 - 273.15f;
    const float wet_bulb = thompson_aa_wet_bulb_temperature(
        pressure0, temp0, qv0);
    const float pi = THOMPSON_AA_PI;
    const float am_r = THOMPSON_AA_AM_R;
    // WRF: rhof(k) = SQRT(RHO_NOT/rho(k)) (:1974).  thompson.cu:3161 writes
    // the algebraically equal RHO_NOT*orho; mp=28 uses WRF's division so the
    // aerosol port and the Fortran oracle agree to the last float bit.
    const float rho_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) / rho);

    const float cloud_mass = fmaxf(0.0f, qc[idx] * rho);
    const float rain_mass = fmaxf(0.0f, qr[idx] * rho);

    // Entry distributions plus the whole nc-driven warm-rain rate set,
    // :1804-1842 and :2144-2222.  Shared verbatim with the readback probe
    // at the bottom of this file so the two cannot drift.
    ThompsonAaWarmRain warm;
    thompson_aa_warm_rain_rates(
        qc[idx], nc_entry[idx], qr[idx], nr[idx],
        nwfa_entry[idx], nifa_entry[idx],
        rho, temp0, rho_factor, inverse_dt, rain_cloud_efficiency, &warm);

    // WRF's L_qc(k) tests the MIXING RATIO against R1 (:1826), not the mass
    // content rc = qc*rho; see the note on thompson_aa_warm_rain_rates.
    const bool l_qc = warm.l_qc;
    const float cloud_number = warm.nc_m3;
    const float cloud_mvd = warm.mvd_c;
    const float nwfa_m3 = warm.nwfa_m3;
    const float nifa_m3 = warm.nifa_m3;
    const float rain_number = warm.nr_m3;
    const double rain_lambda = warm.lamr;
    // warm.mvd_r is deliberately NOT unpacked here: every rate that reads it
    // (pnr_rcr, the t_Efrw bin, Eff_aero) lives inside
    // thompson_aa_warm_rain_rates, and an unread local here would be a
    // standing invitation to re-derive a second, subtly different rain MVD.
    double autoconversion_rate = warm.prr_wau;
    double autoconversion_number_rate = warm.pnr_wau;
    double rain_cloud_rate = warm.prr_rcw;
    const double rain_self_number_rate = warm.pnr_rcr;
    const double cloud_autoconversion_number_sink = warm.pnc_wau;
    const double cloud_accretion_number_sink = warm.pnc_rcw;
    const double ccn_rain_scavenge = warm.pna_rca;
    const double in_rain_scavenge = warm.pnd_rcd;

    // WRF's frozen-species block (:2239+) is NOT temperature gated, so snow
    // and graupel still collect cloud water and still scavenge aerosol on an
    // ambient-warm level.  One shared diagnosis, gated against
    // tools/thompson_wrf461_oracle/probe_warm_frozen_aero.F90.
    ThompsonAaFrozen frozen;
    thompson_aa_frozen_collect_rates(
        qs[idx], qg[idx], graupel_number_per_kg[idx],
        l_qc, cloud_mass, cloud_number, cloud_mvd, nwfa_m3, nifa_m3,
        rho, temp0, wet_bulb, rho_factor, inverse_dt,
        snow_cloud_efficiency, &frozen);
#if defined(THOMPSON_AA_HAIL)
    // The hail-aware graupel replaces every graupel member of `frozen`:
    // the prognostic ng1d/qb1d distribution on its density bin (:1915-1949,
    // :2135-2139) and the riming/scavenging rates on it (:2414-2471), with
    // the Ef_gw*0.1 melting factor and the pbg_gcw rime volume both decided
    // by the wet-bulb temperature as WRF decides them.  visco(k) is the
    // tempc >= 0 branch of :1993.
    ThompsonAaHailGraupel hail;
    thompson_aa_hail_graupel_entry(qg[idx], graupel_number_per_kg[idx],
                                   graupel_volume[idx], rho, &hail);
    const float hail_visco = thompson_aa_mul(
        thompson_aa_add(1.718f, thompson_aa_mul(0.0049f, tempc)), 1.0e-5f);
    ThompsonAaHailCollect hail_collect;
    thompson_aa_hail_graupel_collect(
        &hail, l_qc, cloud_mvd, cloud_mass, cloud_number, nwfa_m3, nifa_m3,
        rho, temp0, wet_bulb, rho_factor, hail_visco, inverse_dt,
        &hail_collect);
    frozen.l_qg = hail.l_qg;
    frozen.rg = hail.rg;
    frozen.ng_m3 = hail.ng;
    frozen.lamg = hail.lamg;
    frozen.ilamg = hail.ilamg;
    frozen.n0_g = hail.n0_g;
    frozen.prg_gcw = hail_collect.prg_gcw;
    frozen.pnc_gcw = hail_collect.pnc_gcw;
    frozen.pna_gca = hail_collect.pna_gca;
    frozen.pnd_gcd = hail_collect.pnd_gcd;
    // pbg_rcg and the melt/sublimation volume and number terms, each
    // captured where WRF forms it (before the conservation passes).
    double hail_pbg_rcg = 0.0;
    double hail_pbg_gml = 0.0;
    double hail_png_rcg = 0.0;
    double hail_png_gde = 0.0;
#endif

    // WRF's rs(k) and rg(k): the mass content where L_qs/L_qg hold, and the
    // R1 placeholder otherwise (:1908-1911, :1918-1946).  Everything below
    // -- the collision-table content bins, the melting and sublimation
    // blocks, and the two species conservation limiters -- reads these,
    // exactly as :2486-2530, :2782-2830 and :2916-2937 do, and each of
    // those guards on L_qs/L_qg rather than on a content threshold.
    const bool l_qs = frozen.l_qs;
    const bool l_qg = frozen.l_qg;
    const float snow_mass = frozen.rs;
    const float graupel_mass = frozen.rg;
    const float graupel_number = frozen.ng_m3;
    const float snow_number = frozen.smo0;
    const float snow_first_moment = frozen.smo1;
    const float snow_ventilation_moment = frozen.smof;
    double snow_cloud_rate = frozen.prs_scw;
    double graupel_cloud_rate = frozen.prg_gcw;
    const double cloud_snow_number_sink = frozen.pnc_scw;      // pnc_scw
    const double cloud_graupel_number_sink = frozen.pnc_gcw;   // pnc_gcw
    const double ccn_snow_scavenge = frozen.pna_sca;           // pna_sca
    const double in_snow_scavenge = frozen.pnd_scd;            // pnd_scd
    const double ccn_graupel_scavenge = frozen.pna_gca;        // pna_gca
    const double in_graupel_scavenge = frozen.pnd_gcd;         // pnd_gcd

    // The ambient-warm block still selects each liquid/frozen collision
    // direction from wet-bulb temperature.  A dry layer can therefore enter
    // the freezing branch while the same call also evaluates ambient-warm
    // melt, exactly as mp_thompson does.
    double rain_snow_rain_rate = 0.0;
    double rain_snow_snow_rate = 0.0;
    double rain_snow_graupel_rate = 0.0;
    double rain_snow_number_rate = 0.0;
    double rain_graupel_rain_rate = 0.0;
    double rain_graupel_graupel_rate = 0.0;
    double rain_graupel_number_rate = 0.0;
    double rain_graupel_rain_number_rate = 0.0;
    if (rain_mass >= 1.0e-6f) {
        const int rain_bin = thompson_aa_decade_index(
            rain_mass, -6, 37);
        const double rain_intercept = (double)(__fdiv_rn((1.0f / 6.0f)
            * rain_mass, am_r)) * rain_lambda * rain_lambda
            * rain_lambda * rain_lambda;
        const int rain_intercept_bin =
            thompson_aa_decade_index_double(rain_intercept, 6, 37);
#if defined(THOMPSON_AA_WRF39)
        const int ntb_sg = THOMPSON_AA_WRF39_NTB_SG;
        const int sg_exp0 = THOMPSON_AA_WRF39_SG_EXP0;
        const float sg_first = THOMPSON_AA_WRF39_R_SG_FIRST;
        // fork :2365, :2402: the dry-bulb temperature picks the branch, and
        // this kernel owns only levels at or above 0 C (audit T3).
        const bool freezing_branch = false;
#else
        const int ntb_sg = 37;
        const int sg_exp0 = -6;
        const float sg_first = 1.0e-6f;
        const bool freezing_branch = wet_bulb < 273.15f;
#endif
        if (snow_mass >= sg_first) {
            const int snow_bin = thompson_aa_decade_index(
                snow_mass, sg_exp0, ntb_sg);
            const int raw_temp_bin = (int)(__fdiv_rn((tempc - 2.5f), 5.0f)) - 1;
            const int temp_bin = min(9, max(1, -raw_temp_bin)) - 1;
            const size_t table_idx = (size_t)snow_bin
                + (size_t)ntb_sg * ((size_t)temp_bin
                + (size_t)9 * ((size_t)rain_intercept_bin
                + (size_t)37 * (size_t)rain_bin));
            if (freezing_branch) {
                rain_snow_rain_rate = -(tmr_racs2[table_idx]
                    + tcr_sacr2[table_idx] + tmr_racs1[table_idx]
                    + tcr_sacr1[table_idx]);
                rain_snow_snow_rate = tmr_racs2[table_idx]
                    + tcr_sacr2[table_idx] - tcs_racs1[table_idx]
                    - tms_sacr1[table_idx];
                rain_snow_graupel_rate = tmr_racs1[table_idx]
                    + tcr_sacr1[table_idx] + tcs_racs1[table_idx]
                    + tms_sacr1[table_idx];
                rain_snow_rain_rate = fmax(
                    (double)(-rain_mass * inverse_dt),
                    rain_snow_rain_rate);
                rain_snow_snow_rate = fmax(
                    (double)(-snow_mass * inverse_dt),
                    rain_snow_snow_rate);
                rain_snow_graupel_rate = fmin(
                    (double)((rain_mass + snow_mass) * inverse_dt),
                    rain_snow_graupel_rate);
                rain_snow_number_rate = tnr_racs1[table_idx]
                    + tnr_racs2[table_idx] + tnr_sacr1[table_idx]
                    + tnr_sacr2[table_idx];
                rain_snow_number_rate = fmin(
                    (double)(rain_number * inverse_dt),
                    rain_snow_number_rate);
            } else {
                rain_snow_snow_rate = -tcs_racs1[table_idx]
                    - tms_sacr1[table_idx] + tmr_racs2[table_idx]
                    + tcr_sacr2[table_idx];
                rain_snow_snow_rate = fmax(
                    (double)(-snow_mass * inverse_dt),
                    rain_snow_snow_rate);
                rain_snow_rain_rate = -rain_snow_snow_rate;
#if defined(THOMPSON_AA_WRF39)
                // fork :2392-2395: rain number is removed in the melting
                // branch too (audit T7).
                rain_snow_number_rate = fmin(
                    (double)(rain_number * inverse_dt),
                    tnr_racs2[table_idx] + tnr_sacr2[table_idx]);
#endif
            }
        }
#if defined(THOMPSON_AA_WRF39)
        // fork :2249-2270 and :2401-2416: the fork's graupel axes, tables
        // (28,28,37,37) with no density axis; above 0 C always the melting
        // branch, with the break-up source 5*tnr_gacr (audit T7).
        if (graupel_mass >= sg_first) {
            const int graupel_mass_bin = thompson_aa_decade_index(
                graupel_mass, sg_exp0, ntb_sg);
            const int graupel_intercept_bin =
                thompson_aa_decade_index_double(
                    thompson_aa_wrf39_table_intercept(
                        graupel_mass, frozen.ilamg),
                    THOMPSON_AA_WRF39_NG1_EXP0, ntb_sg);
            const size_t table_idx = (size_t)graupel_intercept_bin
                + (size_t)ntb_sg * ((size_t)graupel_mass_bin
                + (size_t)ntb_sg * ((size_t)rain_intercept_bin
                + (size_t)37 * (size_t)rain_bin));
            rain_graupel_rain_rate = fmin(
                (double)(graupel_mass * inverse_dt), tcg_racg[table_idx]);
            rain_graupel_graupel_rate = -rain_graupel_rain_rate;
            rain_graupel_rain_number_rate = -5.0 * tnr_gacr[table_idx];
        }
#elif defined(THOMPSON_AA_HAIL)
        // :2522-2547 on the nine-slab tables at the level's density bin.
        // The branch is the wet-bulb's: below T_0 rain freezes onto the
        // graupel (pbg_rcg at ice density, :2535), above it the graupel
        // sheds into rain, losing number (png_rcg) and volume at its own
        // density (:2543), with the -1.5*tnr_gacr break-up (:2545).
        if (graupel_mass >= 1.0e-6f) {
            int graupel_mass_bin = 0;
            int graupel_intercept_bin = 36;
            if (graupel_mass > 1.0e-6f) {
                graupel_mass_bin = thompson_aa_decade_index(
                    graupel_mass, -6, 37);
                graupel_intercept_bin = thompson_aa_hail_intercept_bin(&hail);
            }
            const size_t table_idx = thompson_aa_hail_rain_graupel_index(
                graupel_intercept_bin, graupel_mass_bin, hail.idx_bg,
                rain_intercept_bin, rain_bin);
            if (wet_bulb < 273.15f) {
                rain_graupel_graupel_rate = fmin(
                    (double)thompson_aa_mul(rain_mass, inverse_dt),
                    tmr_racg[table_idx] + tcr_gacr[table_idx]);
                rain_graupel_rain_rate = -rain_graupel_graupel_rate;
                rain_graupel_rain_number_rate = fmin(
                    (double)thompson_aa_mul(rain_number, inverse_dt),
                    tnr_racg[table_idx] + tnr_gacr[table_idx]);
                hail_pbg_rcg = rain_graupel_graupel_rate
                    / (double)THOMPSON_AA_HAIL_RHO_I;
            } else {
                rain_graupel_rain_rate = fmin(
                    (double)thompson_aa_mul(graupel_mass, inverse_dt),
                    tcg_racg[table_idx]);
                rain_graupel_graupel_rate = -rain_graupel_rain_rate;
                rain_graupel_number_rate = fmin(
                    (double)thompson_aa_mul(graupel_number, inverse_dt),
                    tnr_racg[table_idx]);
                hail_png_rcg = rain_graupel_number_rate;
                hail_pbg_rcg = rain_graupel_graupel_rate
                    / (double)THOMPSON_AA_HAIL_RHO_G[hail.idx_bg];
                rain_graupel_rain_number_rate = -1.5 * tnr_gacr[table_idx];
            }
        }
#endif
    }

    const float diffusivity = 2.11e-5f
        * thompson_aa_powf(__fdiv_rn(temp0, 273.15f), 1.94f) * (101325.0f / pressure0);
    const float viscosity = (1.718f + 0.0049f * tempc) * 1.0e-5f;
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float rho_factor_sqrt = sqrtf(rho_factor);
    const float viscosity_factor = sqrtf(rho / viscosity);
    const float vapor_deficit = fmaxf(
        0.0f, thompson_rslf(pressure0, 273.15f) - qv0);
    const float inverse_fusion = 1.0f / 334000.0f;
    const float schmidt_cuberoot = thompson_aa_powf(0.632f, 1.0f / 3.0f);

    double snow_melt_rate = 0.0;
    double snow_melt_number_rate = 0.0;
    if (l_qs) {
        const float melt_moment =
            (pi * 4.0f * 0.15f * inverse_fusion * 0.86f)
                * snow_first_moment
            + (pi * 4.0f * 0.15f * inverse_fusion * 0.28f
               * schmidt_cuberoot * sqrtf(40.0f))
                * rho_factor_sqrt * viscosity_factor
                * snow_ventilation_moment;
        snow_melt_rate = (double)((tempc * conductivity
            - 2.5e6f * diffusivity * vapor_deficit) * melt_moment);
#if defined(THOMPSON_AA_WRF39)
        // fork :2639-2645: the collision enhancement on the dry-bulb
        // temperature and unconditional, then the melted-drop number at
        // 10**(-0.25*tempc), capped at the snow number.
        snow_melt_rate += (double)(4218.0f * inverse_fusion * tempc)
            * (rain_snow_rain_rate + snow_cloud_rate);
        snow_melt_rate = fmin(
            (double)(snow_mass * inverse_dt),
            fmax(0.0, snow_melt_rate));
        snow_melt_number_rate = (double)(snow_number / snow_mass)
            * snow_melt_rate
            * (double)thompson_aa_powf(10.0f, -0.25f * tempc);
        snow_melt_number_rate = fmin(
            (double)(snow_number * inverse_dt), snow_melt_number_rate);
#else
        if (snow_melt_rate > 0.0) {
            snow_melt_rate += 4218.0 * (double)inverse_fusion
                * (double)(wet_bulb - 273.15f)
                * (rain_snow_rain_rate + snow_cloud_rate);
        }
        snow_melt_rate = fmin(
            (double)(snow_mass * inverse_dt),
            fmax(0.0, snow_melt_rate));
        if (snow_melt_rate > 0.0) {
            snow_melt_number_rate =
                (double)snow_number / (double)snow_mass * snow_melt_rate
                * (double)thompson_aa_powf(
                    10.0f, -0.25f * (wet_bulb - 273.15f));
        }
#endif
    }

    double graupel_melt_rate = 0.0;
    double graupel_melt_number_rate = 0.0;
#if defined(THOMPSON_AA_HAIL)
    // Melting and sublimation of the hail-aware graupel, :2802-2834, in
    // WRF's REAL/DOUBLE order on the level's density bin.  The melted
    // volume leaves at the melt-scaled density (:2821-2823, so the bulk
    // density rises while it melts) and the drops shed carry the
    // wet-bulb-scaled number (:2825).  Sublimation (:2826-2833) needs
    // prr_gml <= 0 and ssati < 0 and carries no rate_max of its own; the
    // shared vapor limiter below scales it with the snow's.
    const float hail_diffu = thompson_aa_hail_diffu(temp0, pressure0);
    const float hail_tcond = thompson_aa_hail_tcond(tempc);
    const float hail_rhof2 = sqrtf(rho_factor);
    const float hail_vsc2 = sqrtf(thompson_aa_div(rho, hail_visco));
    double hail_prg_gde = 0.0;
    if (l_qg) {
        const float delqvs = fmaxf(0.0f, thompson_aa_sub(
            thompson_rslf(pressure0, 273.15f), qv0));
        double n0_melt = hail.n0_g;
        if (thompson_aa_mul(graupel_mass, graupel_number) < 1.0e-4f) {
            const double lamg = 1.0 / hail.ilamg;
            n0_melt = (double)thompson_aa_mul(
                thompson_aa_div(1.0e-4f, graupel_mass),
                THOMPSON_AA_HAIL_OGG2) * lamg;
        }
        float t2_qg_me = thompson_aa_mul(THOMPSON_AA_PI, 4.0f);
        t2_qg_me = thompson_aa_mul(t2_qg_me, THOMPSON_AA_HAIL_C_CUBE);
        t2_qg_me = thompson_aa_mul(t2_qg_me, THOMPSON_AA_HAIL_OLFUS);
        t2_qg_me = thompson_aa_mul(t2_qg_me, 0.28f);
        t2_qg_me = thompson_aa_mul(t2_qg_me, THOMPSON_AA_HAIL_SC3);
        t2_qg_me = thompson_aa_mul(
            t2_qg_me, sqrtf(THOMPSON_AA_HAIL_AV_G[hail.idx_bg]));
        t2_qg_me = thompson_aa_mul(t2_qg_me, THOMPSON_AA_HAIL_CGG[11]);
        const float melt_prefix = thompson_aa_sub(
            thompson_aa_mul(tempc, hail_tcond),
            thompson_aa_mul(thompson_aa_mul(2.5e6f, hail_diffu), delqvs));
        double prr_gml = (double)melt_prefix * n0_melt
            * ((double)THOMPSON_AA_HAIL_T1_QG_ME
                   * thompson_aa_pow(hail.ilamg, (double)THOMPSON_AA_HAIL_CGE[10])
               + (double)thompson_aa_mul(thompson_aa_mul(t2_qg_me,
                                                         hail_rhof2),
                                         hail_vsc2)
                   * thompson_aa_pow(hail.ilamg, (double)THOMPSON_AA_HAIL_CGE[11]));
        prr_gml = fmin((double)thompson_aa_mul(graupel_mass, inverse_dt),
                       fmax(0.0, prr_gml));
        graupel_melt_rate = prr_gml;
        if (prr_gml > 0.0) {
            const float melt_f = (float)fmax(
                (double)0.05f,
                fmin(prr_gml * (double)dt / (double)graupel_mass, 1.0));
            hail_pbg_gml = prr_gml / (double)fmaxf(fminf(
                thompson_aa_mul(melt_f, THOMPSON_AA_HAIL_RHO_G[hail.idx_bg]),
                1000.0f), 50.0f);
            graupel_melt_number_rate = prr_gml * (double)graupel_number
                / (double)graupel_mass
                * (double)gfk_pow(10.0f, thompson_aa_mul(
                    -0.33f, thompson_aa_sub(wet_bulb, 273.15f)));
        } else {
            float hail_ssati = thompson_aa_sub(
                thompson_aa_div(qv0, thompson_rslf(pressure0, temp0)), 1.0f);
            if (fabsf(hail_ssati) < 1.0e-15f) hail_ssati = 0.0f;
            if (hail_ssati < 0.0f) {
                float rvs = 0.0f;
                const float t1_subl = thompson_aa_hail_t1_subl(
                    temp0, rho, thompson_rslf(pressure0, temp0), hail_diffu,
                    hail_tcond, hail_ssati, &rvs);
                const float t2_qg_sd = thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(0.28f, THOMPSON_AA_HAIL_SC3),
                        sqrtf(THOMPSON_AA_HAIL_AV_G[hail.idx_bg])),
                    THOMPSON_AA_HAIL_CGG[11]);
                const float prefix = thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_HAIL_C_CUBE,
                                                    t1_subl),
                                    hail_diffu), hail_ssati), rvs);
                hail_prg_gde = (double)prefix * hail.n0_g
                    * ((double)THOMPSON_AA_HAIL_T1_QG_SD
                           * thompson_aa_pow(hail.ilamg,
                                 (double)THOMPSON_AA_HAIL_CGE[10])
                       + (double)thompson_aa_mul(
                             thompson_aa_mul(t2_qg_sd, hail_vsc2),
                             hail_rhof2)
                           * thompson_aa_pow(hail.ilamg,
                                 (double)THOMPSON_AA_HAIL_CGE[11]));
                hail_prg_gde = fmax(
                    (double)thompson_aa_mul(-graupel_mass, inverse_dt),
                    hail_prg_gde);
                hail_png_gde = hail_prg_gde * (double)graupel_number
                    / (double)graupel_mass;
            }
        }
    }
    if (false) {
#else
    if (l_qg) {
#endif
        const double inverse_lambda = frozen.ilamg;
        const double graupel_intercept = frozen.n0_g;
        double melt_intercept = graupel_intercept;
#if defined(THOMPSON_AA_WRF39)
        // fork :2657: no floor on the melting intercept.
        if (false) {
#else
        if (graupel_mass * graupel_number < 1.0e-4f) {
#endif
            // :2804-2806.  WRF re-forms lamg as 1./ilamg(k) here rather
            // than reusing its own local, and ogg2 is exactly 1 while
            // cge(2,1) is exactly 1 so lamg**cge(2,1) is lamg.
            melt_intercept = (double)(1.0e-4f / graupel_mass)
                * (1.0 / inverse_lambda);
        }
        const double melt_moment = melt_intercept * (
            (double)(pi * 4.0f * 0.5f * inverse_fusion * 0.86f)
                * inverse_lambda * inverse_lambda
            + (double)(pi * 4.0f * 0.5f * inverse_fusion * 0.28f
                       * schmidt_cuberoot * sqrtf(THOMPSON_AA_AV_G)
                       * THOMPSON_AA_WARM_CGG11)
                * (double)(rho_factor_sqrt * viscosity_factor)
                * thompson_aa_pow(inverse_lambda, THOMPSON_AA_WARM_CGE11));
        graupel_melt_rate = (double)(tempc * conductivity
            - 2.5e6f * diffusivity * vapor_deficit) * melt_moment;
        graupel_melt_rate = fmin(
            (double)(graupel_mass * inverse_dt),
            fmax(0.0, graupel_melt_rate));
#if defined(THOMPSON_AA_WRF39)
        // fork :2663-2664: N0_g*cgg(2)*ilamg**cge(2)/rg * prr_gml *
        // 10**(-0.5*tempc), the graupel number from the fork's intercept.
        graupel_melt_number_rate = graupel_intercept * inverse_lambda
            / (double)graupel_mass * graupel_melt_rate
            * (double)thompson_aa_powf(10.0f, -0.5f * tempc);
#else
        if (graupel_melt_rate > 0.0) {
            graupel_melt_number_rate = graupel_melt_rate
                * (double)graupel_number / (double)graupel_mass
                * (double)thompson_aa_powf(
                    10.0f, -0.33f * (wet_bulb - 273.15f));
        }
#endif
    }

    // Above freezing WRF sublimates snow/graupel only when the corresponding
    // melt rate is zero.  Both rates share one later vapor limiter, so they
    // must be diagnosed together rather than by two ordered launchers.
    const float qvsi = thompson_rslf(pressure0, temp0);
    float ssati = qv0 / qvsi - 1.0f;
    if (fabsf(ssati) < 1.0e-15f) ssati = 0.0f;
    double snow_vapor_rate = 0.0;
    double graupel_vapor_rate = 0.0;
    double graupel_vapor_number_rate = 0.0;
#if defined(THOMPSON_AA_WRF39)
    // fork :2648-2652, :2667-2672: snow and graupel sublimate whenever the
    // air is subsaturated, melting or not, snow with C_cube = 0.5.
    const bool snow_sublimates = l_qs;
    const bool graupel_sublimates = l_qg;
    const float snow_capacitance = 0.5f;
#else
    const bool snow_sublimates = l_qs && snow_melt_rate <= 0.0;
#if defined(THOMPSON_AA_HAIL)
    // The hail-aware graupel sublimation was formed above; it joins the
    // shared vapor limiter here.
    const bool graupel_sublimates = hail_prg_gde != 0.0;
#else
    const bool graupel_sublimates = l_qg && graupel_melt_rate <= 0.0;
#endif
    const float snow_capacitance = 0.15f;
#endif
    if (ssati < 0.0f && (snow_sublimates || graupel_sublimates)) {
        const float inverse_temp = 1.0f / temp0;
        const float saturated_density = rho * qvsi;
        const float slope_term = inverse_temp
            * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f);
        const float saturation_slope = saturated_density * slope_term;
        const float saturation_curvature = saturated_density
            * (slope_term * slope_term
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
        const float xsat2 = ssati * ssati;
        const float alpha2 = alpha * alpha;
        const float geometry = 4.0f * pi
            * (1.0f - alpha * ssati
               + 2.0f * alpha2 * xsat2
               - 5.0f * alpha2 * alpha * xsat2 * ssati)
            / (1.0f + gamma);
        if (snow_sublimates) {
            const float ventilation = 0.28f * schmidt_cuberoot
                * sqrtf(40.0f) * rho_factor_sqrt * viscosity_factor;
            snow_vapor_rate = (double)(snow_capacitance * geometry
                * diffusivity
                * ssati * saturated_density
                * (0.86f * snow_first_moment
                   + ventilation * snow_ventilation_moment));
            snow_vapor_rate = fmax(
                (double)(-snow_mass * inverse_dt), snow_vapor_rate);
        }
        if (graupel_sublimates) {
            const double inverse_lambda = frozen.ilamg;
            const double graupel_intercept = frozen.n0_g;
            const float ventilation = 0.28f * schmidt_cuberoot
                * sqrtf(THOMPSON_AA_AV_G) * THOMPSON_AA_WARM_CGG11
                * rho_factor_sqrt * viscosity_factor;
            graupel_vapor_rate = (double)(0.5f * geometry * diffusivity
                * ssati * saturated_density) * graupel_intercept
                * (0.86 * thompson_aa_pow(inverse_lambda, 2.0)
                   + (double)ventilation
                     * thompson_aa_pow(inverse_lambda, THOMPSON_AA_WARM_CGE11));
            graupel_vapor_rate = fmax(
                (double)(-graupel_mass * inverse_dt),
                graupel_vapor_rate);
            graupel_vapor_number_rate = graupel_vapor_rate
                * (double)graupel_number / (double)graupel_mass;
#if defined(THOMPSON_AA_HAIL)
            graupel_vapor_rate = hail_prg_gde;
            graupel_vapor_number_rate = hail_png_gde;
#endif
        }

        // :2862-2876, REAL sump / rate_max / ratio (:1615) as for the
        // limiters below.  At a level at or above 0 C only prs_sde and
        // prg_gde are live and both are sublimation (negative), so only
        // the sump < -eps branch can fire.
        const float vapor_sum_real = (float)(snow_vapor_rate
                                             + graupel_vapor_rate);
#if defined(THOMPSON_AA_WRF39)
        // fork :2701, no density (audit T16).
        const float vapor_limit_real = (qv0 - qvsi) * inverse_dt * 0.999f;
#else
        const float vapor_limit_real = (qv0 - qvsi) * rho * inverse_dt
            * 0.999f;
#endif
        if (vapor_sum_real < -1.0e-15f
                && vapor_sum_real < vapor_limit_real) {
            const double ratio = (double)thompson_aa_div(vapor_limit_real,
                                                         vapor_sum_real);
            snow_vapor_rate *= ratio;
            graupel_vapor_rate *= ratio;
        }
        // WRF deliberately leaves png_gde held when this shared vapor
        // limiter rescales prg_gde.
    }

    if (dt > 120.0f) {
        // WRF's adaptive-step safeguard converts warm cloud riming directly
        // to rain instead of leaving liquid mass on frozen categories.
        // :2841 (fork :2680) `prr_rcw(k)=prr_rcw(k)+prs_scw(k)+prg_gcw(k)`
        // associates left to right; `+=` would add the riming pair first.
        rain_cloud_rate = rain_cloud_rate + snow_cloud_rate
            + graupel_cloud_rate;
        snow_cloud_rate = 0.0;
        graupel_cloud_rate = 0.0;
    }

    // WRF conservation order: cloud, rain, snow, graupel, then restore the
    // paired collision transfers.  Number rates intentionally remain held
    // when a mass limiter scales its corresponding process -- verified at
    // module_mp_thompson.F:2880-2889, where pnc_wau / pnc_rcw / pnc_scw /
    // pnc_gcw are ABSENT from the rescale list.
    //
    // Cloud water, :2878-2890 (fork :2714-2726), as WRF writes it.  At a
    // level at or above 0 C pri_wfz and prg_scw are zero, so WRF's
    // sump = -prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw - prg_gcw is
    // the negated sum below, rounded identically.  sump, rate_max and ratio
    // are REAL (:1615): the DOUBLE sum is rounded to REAL, compared in REAL
    // against -rc*odts, and the REAL quotient widens to rescale the DOUBLE
    // rates; the gate is L_qc(k).  The cloud tendency (:2987) is then the
    // sum of the RESCALED rates, never the limit itself.  This limiter fires
    // where the step drains the whole cloud, and :3975 forms
    // qc1d + qcten*DT as the difference of two nearly equal numbers, so its
    // last bit decides between exactly zero and a one-ulp residue.  The
    // double ratio and the substituted limit moved the rain the drained
    // cloud feeds: 7 qr / nr cells of the mp=28 column oracle's edge column
    // with masses and no numbers at dt = 20 s, now bit-identical.
    double cloud_sink = autoconversion_rate + rain_cloud_rate
        + snow_cloud_rate + graupel_cloud_rate;
    const float cloud_sum_real = (float)(-cloud_sink);
    const float cloud_limit_real = -cloud_mass * inverse_dt;
    if (cloud_sum_real < cloud_limit_real && l_qc) {
        const double ratio = (double)thompson_aa_div(cloud_limit_real,
                                                     cloud_sum_real);
        autoconversion_rate *= ratio;
        rain_cloud_rate *= ratio;
        snow_cloud_rate *= ratio;
        graupel_cloud_rate *= ratio;
        cloud_sink = autoconversion_rate + rain_cloud_rate
            + snow_cloud_rate + graupel_cloud_rate;
    }

    // Rain, snow and graupel, :2905-2941 (fork :2740-2775), with the same
    // REAL sump / rate_max / ratio as the cloud above.  The sums keep WRF's
    // association: the terms that are zero at a level at or above 0 C
    // (prg_rfz, pri_rfz, prr_rci, prs_ihm, prg_ihm) drop out exactly, but
    // prs_sde - prr_sml + prs_rcs and prg_gde - prr_gml + prg_rcg are
    // summed left to right, the melt before the collection.  The rain gate
    // is L_qr(k), the entry mixing ratio above R1 (:1878), not rr > R1.
    const float rain_sum_real = (float)(rain_snow_rain_rate
                                        + rain_graupel_rain_rate);
    const float rain_limit_real = -rain_mass * inverse_dt;
    if (rain_sum_real < rain_limit_real && qr[idx] > THOMPSON_AA_R1) {
        const double ratio = (double)thompson_aa_div(rain_limit_real,
                                                     rain_sum_real);
        rain_snow_rain_rate *= ratio;
        rain_graupel_rain_rate *= ratio;
    }

    const float snow_sum_real = (float)(snow_vapor_rate - snow_melt_rate
                                        + rain_snow_snow_rate);
    const float snow_limit_real = -snow_mass * inverse_dt;
    if (snow_sum_real < snow_limit_real && l_qs) {
        const double ratio = (double)thompson_aa_div(snow_limit_real,
                                                     snow_sum_real);
        snow_vapor_rate *= ratio;
        snow_melt_rate *= ratio;
        rain_snow_snow_rate *= ratio;
    }
    const float graupel_sum_real = (float)(graupel_vapor_rate
        - graupel_melt_rate + rain_graupel_graupel_rate);
    const float graupel_limit_real = -graupel_mass * inverse_dt;
    if (graupel_sum_real < graupel_limit_real && l_qg) {
        const double ratio = (double)thompson_aa_div(graupel_limit_real,
                                                     graupel_sum_real);
        graupel_vapor_rate *= ratio;
        graupel_melt_rate *= ratio;
        rain_graupel_graupel_rate *= ratio;
    }
    // Blossey's re-enforcement, :2945-2954 (fork :2780-2787).  ratio is
    // REAL, so the paired magnitude MIN(ABS(a), ABS(b)) of two DOUBLE rates
    // is rounded to REAL before SIGN(1.0, SNGL(a)) gives it a's sign: both
    // members of each pair leave as float32 values.  Keeping the double
    // magnitude moved every level with a rain-graupel or rain-snow
    // collection by up to half a float32 ulp of that rate.
    thompson_aa_reenforce_pair(&rain_graupel_rain_rate,
                               &rain_graupel_graupel_rate);
#if defined(THOMPSON_AA_WRF39)
    if (temp0 > 273.15f) {
#else
    if (wet_bulb > 273.15f) {
#endif
        thompson_aa_reenforce_pair(&rain_snow_rain_rate,
                                   &rain_snow_snow_rate);
    }


    // Preserve WRF's two independent post-conservation branch decisions.
    // Snow fallout blends with the rain velocity only for actual same-call
    // snow melt; rain merely coexisting with snow is not sufficient.
    snow_melt_marker[idx] = snow_melt_rate > 0.0 ? 1.0f : 0.0f;
    graupel_melt_marker[idx] = graupel_melt_rate > 0.0 ? 1.0f : 0.0f;

    const double rain_rate = autoconversion_rate + rain_cloud_rate
        + snow_melt_rate + graupel_melt_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
#if defined(THOMPSON_AA_WRF39)
    const double rain_number_rate = autoconversion_number_rate
        + snow_melt_number_rate + graupel_melt_number_rate
        - rain_self_number_rate - rain_snow_number_rate
        - rain_graupel_rain_number_rate;
#else
    // :3064-3067: (pnr_wau + pnr_sml + pnr_gml) - (pnr_rfz + pnr_rcr +
    // pnr_rcg + pnr_rcs + pnr_rci + pni_rfz), the sinks summed first and
    // in that order (pnr_rfz, pnr_rci and pni_rfz are zero here).
    const double rain_number_rate = ((autoconversion_number_rate
        + snow_melt_number_rate) + graupel_melt_number_rate)
        - ((rain_self_number_rate + rain_graupel_rain_number_rate)
           + rain_snow_number_rate);
#endif
    const double snow_rate = snow_vapor_rate
        + snow_cloud_rate + rain_snow_snow_rate - snow_melt_rate;
    const double graupel_rate = graupel_vapor_rate + graupel_cloud_rate
        + rain_snow_graupel_rate + rain_graupel_graupel_rate
        - graupel_melt_rate;
    const double graupel_number_rate = graupel_vapor_number_rate
        + rain_snow_number_rate - rain_graupel_number_rate
        - graupel_melt_number_rate;

    // CONTRACTION, and why every multiply-add below is pinned (WP-13b).
    // WRF applies the accumulated tendencies at :3973-4023 in the form
    // `q1d(k) = q1d(k) + qten(k)*DT`, and the gfortran -O2 baseline-x86-64
    // the oracle is built from has NO FMA instruction, so qten*DT is rounded
    // to REAL(4) BEFORE the add.  Left plain, nvrtc (--fmad=true) contracts
    // each of these into a single fma and never rounds the product; the two
    // then disagree by up to one ulp of the ENTRY value, which is a 1e-5
    // relative error wherever the level is nearly emptied.
    // thompson_aerosol_sat.cu:911-916 already pins exactly this multiply-add
    // for the rain-evaporation apply, and thompson_aerosol_cold.cu pins the
    // cold half; leaving the warm half contracted was an internal
    // inconsistency, not a decision.  Same tie-break as "THE SHARED FITS" in
    // thompson_aerosol_common.cuh: the authority is WRF, not mp=8.
    // Entry rain and graupel as WRF holds them after :1878-1905 and
    // :1915-1949 (zero where the mixing ratio is at or below R1), for the
    // source-stage removals at :3088-3091 and :3157-3159 below.
    const float qr_entry_wrf = qr[idx] > THOMPSON_AA_R1 ? qr[idx] : 0.0f;
    const float nr_entry_wrf = qr[idx] > THOMPSON_AA_R1 ? nr[idx] : 0.0f;
    const float qg_entry_wrf = qg[idx] > THOMPSON_AA_R1 ? qg[idx] : 0.0f;
    const float ng_entry_wrf = qg[idx] > THOMPSON_AA_R1
        ? graupel_number_per_kg[idx] : 0.0f;
    if (qcten != nullptr) {
        // :2987, rounded once to REAL as WRF's REAL + DOUBLE sum is.
        qcten[idx] = (float)((double)qcten[idx]
                             - cloud_sink * (double)orho);
    } else {
        qc[idx] = fmaxf(0.0f, thompson_aa_sub(qc[idx],
            thompson_aa_mul((float)(cloud_sink * (double)orho), dt)));
    }
    const bool accumulate_rain = nrten != nullptr;
    if (accumulate_rain) {
        // :3058-3067.  REAL + DOUBLE*orho, rounded once.
        qrten[idx] = (float)((double)qrten[idx] + rain_rate * (double)orho);
        nrten[idx] = (float)((double)nrten[idx]
                             + rain_number_rate * (double)orho);
    } else {
        qr[idx] = fmaxf(0.0f, thompson_aa_add(qr[idx],
            thompson_aa_mul((float)(rain_rate * (double)orho), dt)));
        nr[idx] = fmaxf(0.0f, thompson_aa_add(nr[idx],
            thompson_aa_mul((float)(rain_number_rate * (double)orho), dt)));
    }
#if defined(THOMPSON_AA_HAIL)
    // :3094-3097, the ambient-warm snow source before sedimentation.
    qsten[idx] = (float)((double)qsten[idx] + snow_rate * (double)orho);
#else
    qs[idx] = fmaxf(0.0f, thompson_aa_add(qs[idx],
        thompson_aa_mul((float)(snow_rate * (double)orho), dt)));
#endif
#if defined(THOMPSON_AA_HAIL)
    {
        // :3099-3116 for an ambient-warm level, in WRF's order with its
        // zero members dropped (adding an exact 0.0 to a DOUBLE sum is the
        // identity): prg_scw, prg_rfz, prg_rci and prg_ihm are sub-freezing
        // terms, and pbg_sml is never set.  qg, ng and qb stay the read-only
        // entry state for the terminal apply.
        (void)graupel_rate;
        (void)graupel_number_rate;
        float hail_qgten = (float)((double)qgten[idx]
            + ((((graupel_vapor_rate + rain_graupel_graupel_rate)
                 + graupel_cloud_rate) + rain_snow_graupel_rate)
               - graupel_melt_rate) * (double)orho);
        float hail_ngten = (float)((double)ngten[idx]
            + (((-hail_png_rcg + rain_snow_number_rate) + hail_png_gde)
               - graupel_melt_number_rate) * (double)orho);
        const double pbg_rcs = rain_snow_graupel_rate
            / (double)THOMPSON_AA_HAIL_RHO_I;                     // :2511
        float hail_qbten = (float)((double)qbten[idx]
            + ((((hail_collect.pbg_gcw + pbg_rcs) + hail_pbg_rcg)
                - hail_pbg_gml)
               + graupel_vapor_rate
                 / (double)THOMPSON_AA_HAIL_RHO_G[hail.idx_bg])
              * (double)orho);
        const float qg_entry = qg[idx] > THOMPSON_AA_R1 ? qg[idx] : 0.0f;
        const float ng_entry = qg[idx] > THOMPSON_AA_R1
            ? graupel_number_per_kg[idx] : 0.0f;
        thompson_aa_hail_graupel_balance(
            qg_entry, ng_entry, hail.qb_entry, hail.idx_bg, rho, orho,
            inverse_dt, dt, &hail_qgten, &hail_ngten, &hail_qbten);
        qgten[idx] = hail_qgten;
        ngten[idx] = hail_ngten;
        qbten[idx] = hail_qbten;
    }
#else
    qg[idx] = fmaxf(0.0f, thompson_aa_add(qg[idx],
        thompson_aa_mul((float)(graupel_rate * (double)orho), dt)));
#endif
    // Keep classic ng1d raw through source and fallout tendencies.  WRF
    // diagnoses a separate qg-based number for fallout velocity and applies
    // the private moment's sole size bound only in the final state pass.
#if defined(THOMPSON_AA_WRF39) || defined(THOMPSON_AA_HAIL)
    (void)graupel_number_rate;
#else
    graupel_number_per_kg[idx] = thompson_aa_add(
        graupel_number_per_kg[idx],
        thompson_aa_mul((float)(graupel_number_rate * (double)orho), dt));
#endif
    const double vapor_rate = snow_vapor_rate + graupel_vapor_rate;
    // The entry vapour plus the tendency, unfloored: WRF floors the running
    // vapour once, at the terminal apply (:3974); see the cold network.
    // With qvten given, qv stays qv1d and the tendency is accumulated at
    // the end of this kernel instead.
    if (qvten == nullptr) {
        qv[idx] = thompson_aa_sub(qv[idx],
            thompson_aa_mul((float)(vapor_rate * (double)orho), dt));
    }
    if (accumulate_rain) {
        // :3070-3091 in WRF's tendency form.
        float rain_mass_tendency = qrten[idx];
        float rain_number_tendency = nrten[idx];
        thompson_aa_rain_balance_tendency(qr[idx], nr[idx],
            &rain_mass_tendency, &rain_number_tendency, rho, orho,
            inverse_dt, dt);
        qrten[idx] = rain_mass_tendency;
        nrten[idx] = rain_number_tendency;
    } else {
        thompson_aa_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
    }
    // :3067-3091.  Where the post-source rain concentration is at or below R1
    // WRF discards the call's rain sources and removes the entry rain:
    // qrten = -qr1d*odts, nrten = -nr1d*odts.  The number bound above zeroes
    // only the number; the mass has to go too.
    if (!accumulate_rain
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
#if defined(THOMPSON_AA_WRF39) || defined(THOMPSON_AA_HAIL)
    // (The hail-aware build ran its balance in tendency form above.)
    (void)qg_entry_wrf;
    (void)ng_entry_wrf;
    if (false)
#endif
    {
        const float xrg = thompson_aa_mul(qg[idx], rho);
        if (!(xrg > THOMPSON_AA_R1)) {
            qg[idx] = thompson_aa_add(qg_entry_wrf, thompson_aa_mul(
                thompson_aa_mul(-qg_entry_wrf, inverse_dt), dt));
            graupel_number_per_kg[idx] = thompson_aa_add(ng_entry_wrf,
                thompson_aa_mul(thompson_aa_mul(-ng_entry_wrf, inverse_dt),
                                dt));
        } else {
            // REAL(4) throughout except lamg, which is DOUBLE (:1597).
            const float am_g = 3.1415926536f * 400.0f / 6.0f;
            const float mvd_numerator = 3.0f + 0.0f + 0.672f;
            const float shadow_m3 =
                thompson_aa_mul(graupel_number_per_kg[idx], rho);
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
                graupel_number_per_kg[idx] = thompson_aa_add(
                    ng_entry_wrf, thompson_aa_mul(ngten, dt));
            }
        }
    }

    // --- the three mp=28 accumulators, :2963-2994 ------------------------
    // pni_wfz / pni_iha / pni_inu are identically zero on an ambient-warm
    // level (Bigg freezing, Koop haze freezing and deposition nucleation are
    // all sub-freezing processes), but the zero terms are written out so the
    // left-to-right summation order matches WRF's.
    // WRF's accumulators are REAL but every p* rate is DOUBLE PRECISION, so
    // `ncten(k) = ncten(k) + (...)*orho` promotes the running total to
    // double, adds, and rounds ONCE.  Rounding the increment to float first
    // and then adding is a different (and, across four accumulating
    // packages, cumulative) answer.
    const double pni_wfz = 0.0;
    const double pni_iha = 0.0;
    const double pni_inu = 0.0;
    ncten[idx] = (float)((double)ncten[idx]
        + ((((-cloud_autoconversion_number_sink
              - cloud_accretion_number_sink)
              - pni_wfz)
              - cloud_snow_number_sink)
              - cloud_graupel_number_sink) * (double)orho);
    nwfaten[idx] = (float)((double)nwfaten[idx]
        - (((ccn_rain_scavenge + ccn_snow_scavenge)
            + ccn_graupel_scavenge) + pni_iha) * (double)orho);
    nifaten[idx] = (float)((double)nifaten[idx]
        - ((in_rain_scavenge + in_snow_scavenge)
           + in_graupel_scavenge) * (double)orho);
    // dustyIce is a PARAMETER .true. (:61), so pni_inu is a nifa sink too.
    nifaten[idx] = (float)((double)nifaten[idx]
        - pni_inu * (double)orho);

    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
#if !defined(THOMPSON_AA_WRF39)
    if (tten != nullptr) {
        // :2982.  qvten += (-pri_inu - pri_iha - pri_ide - prs_ide - prs_sde
        // - prg_gde)*orho; only prs_sde and prg_gde live above freezing and
        // subtracting the zeros is exact.
        // Multiply-adds pinned, as in the cold network.
        qvten[idx] = (float)__dadd_rn((double)qvten[idx],
            __dmul_rn((-snow_vapor_rate) - graupel_vapor_rate,
                      (double)orho));
        // :3174-3179, the temp >= T_0 branch:
        //   tten += (lfus*ocp*(-prr_sml - prr_gml - prr_rcg - prr_rcs)
        //            + lsub*ocp*(prs_sde + prg_gde))*orho*(1-IFDRY)
        // lfus*ocp and lsub*ocp are REAL products (lfus = lsub - lvap0 =
        // 334000 exactly), each times its DOUBLE sum in WRF's term order.
        // The form this replaces (334000.0*fusion + 2.834e6*vapor in
        // DOUBLE, times ocp, orho and DT, added to the state) summed prr_rcs
        // before prr_rcg and never rounded the REAL tendency.
        const float ocp = thompson_aa_div(1.0f, thompson_aa_mul(1004.0f,
            thompson_aa_add(1.0f, thompson_aa_mul(0.887f, qv0))));
        const double melt_sum = (((-snow_melt_rate) - graupel_melt_rate)
            - rain_graupel_rain_rate) - rain_snow_rain_rate;
        const double vapor_sum = snow_vapor_rate + graupel_vapor_rate;
        const float tten_src = (float)__dadd_rn((double)tten[idx],
            __dmul_rn(__dadd_rn(
                __dmul_rn((double)thompson_aa_mul(334000.0f, ocp), melt_sum),
                __dmul_rn((double)thompson_aa_mul(2.834e6f, ocp),
                          vapor_sum)),
                (double)orho));
        tten[idx] = tten_src;
        // :3189, temp(k) = t1d(k) + DT*tten(k).
        temperature[idx] = thompson_aa_add(
            temp0, thompson_aa_mul(dt, tten_src));
        (void)inverse_cp;
        return;
    }
#else
    (void)qvten;
    (void)tten;
#endif
    const double fusion_rate = -snow_melt_rate - graupel_melt_rate
        - rain_snow_rain_rate - rain_graupel_rain_rate;
    temperature[idx] = temp0 + (float)(
        ((334000.0 * fusion_rate) + (2.834e6 * vapor_rate))
        * (double)inverse_cp * (double)orho * (double)dt);
}


// ---------------------------------------------------------------------------
// Cloud water mass/number balance, module_mp_thompson.F:2996-3019.
// ---------------------------------------------------------------------------
//
// THIS KERNEL RUNS EXACTLY ONCE PER mp=28 CALL, after every ncten source and
// before the saturation adjustment.  WRF applies it once per column; putting
// it inside both the warm and the cold network double-applies it, runs, stays
// numerically stable, and is silently wrong.
//
// Two subtleties reproduced literally:
//
//  1. lamc is formed from the PROJECTED number xnc but the ENTRY cloud mass
//     rc(k), NOT from xrc (verified at :3003).  A "consistency cleanup" that
//     swaps rc for xrc changes the answer wherever qcten != 0.
//  2. the rediagnosed xnc inside each clamp branch uses xrc (:3006, :3010),
//     and the resulting ncten is BACKED OUT against nc1d*rho -- it is not an
//     increment.
//
// qc_entry / nc_entry are the frozen entry state; qc_after is the current
// cloud mixing ratio, i.e. WRF's (qc1d + qcten*dtsave).  density is the ENTRY
// rho (:1802); it must not be recomputed from the mutated temperature and
// vapour this late in the call.
extern "C" __global__ void thompson_aa_ncten_balance(
    const float* __restrict__ qc_entry,
    const float* __restrict__ qc_after,
    const float* __restrict__ nc_entry,
    const float* __restrict__ density,
    float* __restrict__ ncten,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float rho = density[idx];
    const float orho = 1.0f / rho;
    const float odts = 1.0f / dt;
    const float am_r = THOMPSON_AA_AM_R;

    // :1844-1849.  A level whose entry cloud is at or below R1 has its
    // qc1d AND nc1d zeroed before any tendency is formed, and carries
    // rc(k) = R1.  Applying that rule here is idempotent.
    float nc1d;
    float rc;
    if (qc_entry[idx] > THOMPSON_AA_R1) {
        nc1d = nc_entry[idx];
        rc = qc_entry[idx] * rho;
    } else {
        nc1d = 0.0f;
        rc = THOMPSON_AA_R1;
    }

    // CONTRACTION-PINNED THROUGHOUT.  Every expression below is of the form
    // (a +/- b*c), which nvrtc (--fmad=true by default) fuses into a single
    // rounding.  gfortran -O2 on baseline x86-64 has no FMA instruction, so
    // WRF rounds the product first.  MEASURED: with contraction left on,
    // (Nt_c_max - nc1d*rho) keeps an exact residual of ~44 where WRF gets
    // exactly 0, and 80 of 11025 oracle rows return a spurious non-zero
    // tendency.  This is not cosmetic -- it is the difference between "the
    // droplet cap is inactive" and "the cap emits a tendency".
    float tendency = ncten[idx];
    const float xrc = fmaxf(
        THOMPSON_AA_R1, thompson_aa_mul(qc_after[idx], rho));
    float xnc = fmaxf(
        THOMPSON_AA_NC_FLOOR,
        thompson_aa_mul(
            thompson_aa_add(nc1d, thompson_aa_mul(tendency, dt)), rho));

    if (xrc > THOMPSON_AA_R1) {
        const int nu_c = thompson_aa_nu_c(xnc);
        double lamc = (double)thompson_aa_powf(
            thompson_aa_div(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(xnc, am_r),
                        THOMPSON_AA_CCG2[nu_c]),
                    THOMPSON_AA_OCG1[nu_c]),
                rc),
            THOMPSON_AA_OBMR);
        const float xdc = (float)(
            (double)((THOMPSON_AA_BM_R + (float)nu_c) + 1.0f) / lamc);
        bool clamped = false;
        if (xdc < THOMPSON_AA_D0C) {
            lamc = (double)thompson_aa_div(THOMPSON_AA_CCE2[nu_c], THOMPSON_AA_D0C);
            clamped = true;
        } else if (xdc > THOMPSON_AA_D0R * 2.0f) {
            lamc = (double)thompson_aa_div(THOMPSON_AA_CCE2[nu_c],
                                           THOMPSON_AA_D0R * 2.0f);
            clamped = true;
        }
        if (clamped) {
            // NOTE: xrc here, rc above.  Both are deliberate.  There is no
            // Nt_c_max MIN on this rediagnosis (contrast :1840).
            xnc = (float)(
                (double)thompson_aa_div(
                    thompson_aa_mul(
                        thompson_aa_mul(THOMPSON_AA_CCG1[nu_c],
                                        THOMPSON_AA_OCG2[nu_c]),
                        xrc),
                    am_r)
                * thompson_aa_pow(lamc, (double)THOMPSON_AA_BM_R));
            tendency = thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_sub(xnc, thompson_aa_mul(nc1d, rho)),
                    odts),
                orho);
        }
    } else {
        tendency = thompson_aa_mul(-nc1d, odts);
    }

    // :3016-3019.  The final guard floors at 0, not at 2.
    xnc = fmaxf(
        0.0f,
        thompson_aa_mul(
            thompson_aa_add(nc1d, thompson_aa_mul(tendency, dt)), rho));
    if (xnc > THOMPSON_AA_NT_C_MAX) {
        tendency = thompson_aa_mul(
            thompson_aa_mul(
                thompson_aa_sub(THOMPSON_AA_NT_C_MAX,
                                thompson_aa_mul(nc1d, rho)),
                odts),
            orho);
    }
    ncten[idx] = tendency;
}


// ---------------------------------------------------------------------------
// Probe kernel: the WP-07 rate set, per cell, with no state mutation.
// ---------------------------------------------------------------------------
//
// This exists so tests/test_thompson_aerosol_warm_gpu.py can gate every
// individual rate against a Fortran oracle built from the pristine
// module_mp_thompson.F, instead of only observing their summed effect on qc
// and qr.  It shares the SAME code path as the network kernel above for the
// droplet distribution and the rain distribution, so an error in either is
// visible in both.
extern "C" __global__ void thompson_aa_probe_warm_rates(
    const float* __restrict__ pressure,
    const float* __restrict__ temperature,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ nc_entry,
    const float* __restrict__ qr,
    const float* __restrict__ nr_entry,
    const float* __restrict__ nwfa_entry,
    const float* __restrict__ nifa_entry,
    const double* __restrict__ rain_cloud_efficiency,
    float* __restrict__ out_nc_m3,
    int* __restrict__ out_nu_c,
    double* __restrict__ out_lamc,
    float* __restrict__ out_mvd_c,
    float* __restrict__ out_xdc,
    float* __restrict__ out_nwfa_m3,
    float* __restrict__ out_nifa_m3,
    float* __restrict__ out_nr_m3,
    double* __restrict__ out_lamr,
    float* __restrict__ out_mvd_r,
    double* __restrict__ out_n0_r,
    double* __restrict__ out_prr_wau,
    double* __restrict__ out_pnr_wau,
    double* __restrict__ out_pnc_wau,
    double* __restrict__ out_prr_rcw,
    double* __restrict__ out_pnc_rcw,
    double* __restrict__ out_pnr_rcr,
    double* __restrict__ out_pna_rca,
    double* __restrict__ out_pnd_rcd,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float rho_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) / rho);

    ThompsonAaWarmRain warm;
    thompson_aa_warm_rain_rates(
        qc[idx], nc_entry[idx], qr[idx], nr_entry[idx],
        nwfa_entry[idx], nifa_entry[idx],
        rho, temp0, rho_factor, 1.0f / dt, rain_cloud_efficiency, &warm);

    out_nc_m3[idx] = warm.nc_m3;
    out_nu_c[idx] = warm.nu_c;
    out_lamc[idx] = warm.lamc;
    out_mvd_c[idx] = warm.mvd_c;
    out_xdc[idx] = warm.xdc;
    out_nwfa_m3[idx] = warm.nwfa_m3;
    out_nifa_m3[idx] = warm.nifa_m3;
    out_nr_m3[idx] = warm.nr_m3;
    // The slope the collection rates read: lamr = 1./ilamr(k) wherever
    // :2197 or :2211 runs (L_qr .and. mvd_r > D0r), the :2147 lamr
    // elsewhere -- the value WRF's lamr holds after :2222, which is what the
    // Fortran probe prints.
    out_lamr[idx] = (warm.rain_active && warm.mvd_r > THOMPSON_AA_D0R)
        ? thompson_aa_rain_lambda_reformed(warm.lamr) : warm.lamr;
    out_mvd_r[idx] = warm.mvd_r;
    out_n0_r[idx] = warm.n0_r;
    out_prr_wau[idx] = warm.prr_wau;
    out_pnr_wau[idx] = warm.pnr_wau;
    out_pnc_wau[idx] = warm.pnc_wau;
    out_prr_rcw[idx] = warm.prr_rcw;
    out_pnc_rcw[idx] = warm.pnc_rcw;
    out_pnr_rcr[idx] = warm.pnr_rcr;
    out_pna_rca[idx] = warm.pna_rca;
    out_pnd_rcd[idx] = warm.pnd_rcd;
}


// ---------------------------------------------------------------------------
// Probe kernel: WRF's always-run frozen collection block, per cell.
// ---------------------------------------------------------------------------
//
// This is the readback that closes the port's largest unmeasured claim.  The
// six mp=28-only rates pnc_scw / pnc_gcw / pna_sca / pnd_scd / pna_gca /
// pnd_gcd run at EVERY level because iiwarm is a PARAMETER .false. (:59), so
// on this side of the freezing seam they fire in exactly the state a real
// forecast spends its mixed-phase time in -- a melting layer -- and no
// committed column fixture reaches it.
//
// It calls thompson_aa_frozen_collect_rates, the SAME device function
// thompson_aa_warm_source_network calls, so a disagreement against
// tools/thompson_wrf461_oracle/probe_warm_frozen_aero.F90 is by construction
// a disagreement in the network.  The wet-bulb temperature is computed here
// exactly as the network computes it, because twet selects the graupel
// riming efficiency's 0.1 factor at :2431 and is therefore part of what is
// being gated.
extern "C" __global__ void thompson_aa_probe_warm_frozen_rates(
    const float* __restrict__ pressure,
    const float* __restrict__ temperature,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ nc_entry,
    const float* __restrict__ qs,
    const float* __restrict__ qg,
    const float* __restrict__ ng_entry,
    const float* __restrict__ nwfa_entry,
    const float* __restrict__ nifa_entry,
    const double* __restrict__ rain_cloud_efficiency,
    const double* __restrict__ snow_cloud_efficiency,
    float* __restrict__ out_rho,
    float* __restrict__ out_rhof,
    float* __restrict__ out_visco,
    float* __restrict__ out_twet,
    float* __restrict__ out_nc_m3,
    float* __restrict__ out_mvd_c,
    float* __restrict__ out_nwfa_m3,
    float* __restrict__ out_nifa_m3,
    float* __restrict__ out_xds,
    float* __restrict__ out_smoe,
    double* __restrict__ out_ilamg,
    double* __restrict__ out_n0_g,
    float* __restrict__ out_xdg,
    float* __restrict__ out_vtg,
    float* __restrict__ out_stoke_g,
    float* __restrict__ out_ef_sw,
    float* __restrict__ out_ef_gw,
    double* __restrict__ out_prs_scw,
    double* __restrict__ out_pnc_scw,
    double* __restrict__ out_prg_gcw,
    double* __restrict__ out_pnc_gcw,
    double* __restrict__ out_pna_sca,
    double* __restrict__ out_pnd_scd,
    double* __restrict__ out_pna_gca,
    double* __restrict__ out_pnd_gcd,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float temp0 = temperature[idx];
    const float pressure0 = pressure[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure0
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float inverse_dt = 1.0f / dt;
    const float tempc = temp0 - 273.15f;
    const float rho_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) / rho);
    const float wet_bulb = thompson_aa_wet_bulb_temperature(
        pressure0, temp0, qv0);
    const float viscosity = (tempc >= 0.0f)
        ? (1.718f + 0.0049f * tempc) * 1.0e-5f
        : (1.718f + 0.0049f * tempc
           - 1.2e-5f * tempc * tempc) * 1.0e-5f;

    // The droplet diagnosis the frozen block consumes is the warm block's
    // own (:1826-1842 then :2168-2175); reusing it here is what makes
    // mvd_c, nc and the aerosol clamps identical to the network's.
    ThompsonAaWarmRain warm;
    thompson_aa_warm_rain_rates(
        qc[idx], nc_entry[idx], 0.0f, 0.0f,
        nwfa_entry[idx], nifa_entry[idx],
        rho, temp0, rho_factor, inverse_dt, rain_cloud_efficiency, &warm);

    const float cloud_mass = fmaxf(0.0f, qc[idx] * rho);

    ThompsonAaFrozen frozen;
    thompson_aa_frozen_collect_rates(
        qs[idx], qg[idx], ng_entry[idx],
        warm.l_qc, cloud_mass, warm.nc_m3, warm.mvd_c,
        warm.nwfa_m3, warm.nifa_m3,
        rho, temp0, wet_bulb, rho_factor, inverse_dt,
        snow_cloud_efficiency, &frozen);

    out_rho[idx] = rho;
    out_rhof[idx] = rho_factor;
    out_visco[idx] = viscosity;
    out_twet[idx] = wet_bulb;
    out_nc_m3[idx] = warm.nc_m3;
    out_mvd_c[idx] = warm.mvd_c;
    out_nwfa_m3[idx] = warm.nwfa_m3;
    out_nifa_m3[idx] = warm.nifa_m3;
    out_xds[idx] = frozen.xds;
    out_smoe[idx] = frozen.smoe;
    out_ilamg[idx] = frozen.ilamg;
    out_n0_g[idx] = frozen.n0_g;
    out_xdg[idx] = frozen.xdg;
    out_vtg[idx] = frozen.vtg;
    out_stoke_g[idx] = frozen.stoke_g;
    out_ef_sw[idx] = frozen.ef_sw;
    out_ef_gw[idx] = frozen.ef_gw;
    out_prs_scw[idx] = frozen.prs_scw;
    out_pnc_scw[idx] = frozen.pnc_scw;
    out_prg_gcw[idx] = frozen.prg_gcw;
    out_pnc_gcw[idx] = frozen.pnc_gcw;
    out_pna_sca[idx] = frozen.pna_sca;
    out_pnd_scd[idx] = frozen.pnd_scd;
    out_pna_gca[idx] = frozen.pna_gca;
    out_pnd_gcd[idx] = frozen.pnd_gcd;
}


// ---------------------------------------------------------------------------
// Probe kernel: the REAL(4) graupel exponents this file derives.
// ---------------------------------------------------------------------------
//
// One thread, eight doubles.  It exists so the test can compare the device's
// cge(6,5) / cge(9,5) / cge(11,5) / bv_g(5) / (3.+mu_g+.672) against the
// digits the Fortran oracle PRINTS (WP07F_CGE6 / _CGE9 / _CGE11 / _BVG /
// _MVDGNUM), instead of against a second hand transcription of them.  A
// compiler that folded these in double instead of float32 would be caught
// here rather than as an 9e-7 drift in prg_gcw.
extern "C" __global__ void thompson_aa_probe_frozen_constants(
    double* __restrict__ out)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    out[0] = THOMPSON_AA_WARM_BV_G;
    out[1] = THOMPSON_AA_WARM_CGE6;
    out[2] = THOMPSON_AA_WARM_CGE9;
    out[3] = THOMPSON_AA_WARM_CGE11;
    out[4] = (double)THOMPSON_AA_WARM_CGG6;
    out[5] = (double)THOMPSON_AA_WARM_CGG9;
    out[6] = (double)THOMPSON_AA_WARM_CGG11;
    out[7] = THOMPSON_AA_WARM_MVDG_D;
    out[8] = (double)(THOMPSON_AA_PI * 400.0f / 6.0f);
    out[9] = (double)THOMPSON_AA_WARM_OGG3;
    out[10] = (double)(THOMPSON_AA_PI * 0.25f * THOMPSON_AA_AV_S);
    out[11] = (double)(THOMPSON_AA_PI * 0.25f * THOMPSON_AA_AV_G
                       * THOMPSON_AA_WARM_CGG9);
}
