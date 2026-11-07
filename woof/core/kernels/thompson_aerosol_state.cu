// gpuwm/core/kernels/thompson_aerosol_state.cu
//
// WP-04 -- aerosol state kernels for WRF v4.6.1 aerosol-aware Thompson
// (mp_physics=28).  Numerical authority is
// WRF v4.6.1 phys/module_mp_thompson.F, commit
// d66e442fccc04111067e29274c9f9eaccc3cef28, zero local modifications.  Every
// bare line number below refers to that file.
// WRF-derived arithmetic retains the notice in NOTICE and
// licenses/LICENSE-WRF-public-domain.txt. Optional output guards add no
// numerical transcription.
//
// This translation unit receives gpuwm/core/kernels/thompson_aerosol_common.cuh
// textually, prepended by gpuwm/core/kernels/__init__.py's _EXTRA_HEADERS
// allow-list.  Every nu_c, every gamma moment, every clamp and every lookup
// index comes from that header.  Nothing here re-derives them.
//
// ===========================================================================
// THE ACCUMULATOR CONTRACT -- this file owns it
// ===========================================================================
// WRF runs one monolithic column loop that
//   (a) freezes nc1d/nwfa1d/nifa1d as read-only ENTRY state at :1795-1842,
//   (b) accumulates ncten/nwfaten/nifaten (per KILOGRAM per second; every
//       increment site multiplies by orho -- :2964, :2975, :3008, :3833) in
//       regions separated by thousands of lines,
//   (c) applies them ONCE at :3972-4021 with a single set of clamps.
//
// ArWen's fused network launchers write state in place, so mp=28 must make
// that split explicit:
//
//   * state nc / nwfa / nifa are READ-ONLY entry state for the whole call.
//   * three device scratch accumulators ncten / nwfaten / nifaten are zeroed
//     at adapter entry and written by every aerosol kernel.
//   * thompson_aa_state_finalize below is the ONLY place the accumulators are
//     applied to state, and it carries WRF's ONLY set of clamps.
//   * any kernel needing a working per-m3 value RECOMPUTES it locally exactly
//     the way WRF does (thompson_aa_working_number /
//     thompson_aa_working_cloud below), instead of applying its own delta.
//
// Four other packages depend on there being exactly ONE clamp point, because
// WRF has exactly one.  Clamping four times is a silent physics change that no
// unit test downstream would flag.
//
// ===========================================================================
// TWO DISTINCT AEROSOL SNAPSHOTS -- the asymmetry is intentional
// ===========================================================================
//   :1805-1806  ENTRY   nwfa = MAX(11.1E6, MIN(9999.E6, nwfa1d*rho))
//                       nifa = MAX(naIN1*0.01, MIN(9999.E6, nifa1d*rho))
//               Feeds scavenging, iceDeMott and iceKoop.  BOTH bounds.
//
//   :3211       WORKING nwfa = MAX(11.1E6, (nwfa1d + nwfaten*DT)*rho)
//               Feeds activ_ncloud ONLY.  NO upper bound, and there is NO
//               nifa counterpart at all.  rho here is the TAU+1 density
//               recomputed at :3193 from the updated temp/qv, NOT the entry
//               rho of :1802.
//
// Conflating the two changes activated droplet number wherever scavenging was
// significant.  Reproduce, do not smooth.
//
// ===========================================================================
// WRF UNIT INCONSISTENCIES THAT ARE REPRODUCED LITERALLY
// ===========================================================================
// :3976  nc1d(k) = MAX(2./rho(k), MIN(nc1d(k) + ncten(k)*DT, Nt_c_max))
//        nc1d is PER KILOGRAM but is compared against the volumetric
//        Nt_c_max = 1999.E6 with no density conversion.  Do NOT divide by rho
//        to "fix" the upper bound; the lower bound IS converted and the upper
//        bound is not.  Fixture aero-nc-cap is what pins this.
// :3979-3982  nwfa1d/nifa1d, also per kilogram, clamped against the per-m3
//        constants 11.1E6 / 5.0E3 / 9999.E6.  Same treatment.
// :4020  the terminal droplet rediagnosis caps at DBLE(Nt_c_max)/rho(k),
//        i.e. the SAME constant but converted.  Three different conventions
//        in nine lines; all three are transcribed as written.
//
// ===========================================================================
// HEIGHT FIELD FOR thompson_init's PROFILE FILL  (resolved, see the .py)
// ===========================================================================
// WRF passes hgt=z_at_q (module_physics_init.F:4517-4544).  Despite the name,
// dyn_em/start_em.F:870-876 fills it as
//     z_at_q(i,k,j) = (grid%ph_2(i,k,j)+grid%phb(i,k,j))/g,  k = kts..kte
// and ph/phb are Z-STAGGERED in Registry.EM_COMMON:198-200, i.e. FULL (w)
// levels.  z_at_q is therefore the w-level height above SEA level, truncated
// to the lowest kte entries -- it is NOT the mass-level height.  hgt(i,1,j) is
// the terrain elevation, which is what the ABSOLUTE 1000 m / 2500 m h_01
// thresholds are testing.  ArWen's exact analogue is z8w[:nz] with
// z8w = (phb + php)/G, which gpuwm/core/microphysics.py::_apply_thompson
// already materializes.  Passing the mass-level height 0.5*(z8w[k]+z8w[k+1])
// instead would shift h_01 over terrain and reshape the whole CCN profile.
//
#define THOMPSON_AA_STATE_EPS 1.0e-15f   // :185, thompson_init's fill test

// Correctly-rounded float32 cosine, for thompson_init's h_01 branch only.
// gfortran lowers REAL(4) COS to glibc cosf (correctly rounded); CUDA's cosf
// carries ~2 ulp.  Same rationale as thompson_aa_expf in the shared header.
__device__ __forceinline__ float thompson_aa_cosf_cr(float x)
{
    return (float)cos((double)x);
}

// module_mp_thompson.F:1802, :3193, :5624 -- one definition, three sites.
// qv must already be MAX'd at 1.E-10 by the caller, exactly as :1801 does.
__device__ __forceinline__ float thompson_aa_density(
    float pressure, float temperature, float qv)
{
    return 0.622f * pressure / (287.04f * temperature * (qv + 0.622f));
}


// ---------------------------------------------------------------------------
// 1.  ENTRY SNAPSHOT -- module_mp_thompson.F:1795-1812, aer_init_opt < 2.
// ---------------------------------------------------------------------------
//
// The read-only per-m3 aerosol state for the whole call.  Written once at
// adapter entry, then never again; the scavenging, iceDeMott and iceKoop
// kernels read THIS, not state.nwfa/state.nifa and not the working refresh.
//
// aer_init_opt is pinned at 0 for this port (SCOPE PIN, spec "Strategy"), so
// only the .lt. 2 branch exists here.  wif_input_opt is pinned at 0, so nbca
// is identically zero and is not carried at all.
//
// rho_out is WRF's entry density (:1802).  It is an output rather than an
// input because :1801's MAX(1.E-10, qv1d) must be applied before the density,
// and having one kernel own that ordering is what stops five packages from
// each writing a slightly different rho.
extern "C" __global__ void thompson_aa_entry_snapshot(
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ nwfa,      // state, per kilogram, READ-ONLY
    const float* __restrict__ nifa,      // state, per kilogram, READ-ONLY
    float* __restrict__ rho_out,
    float* __restrict__ nwfa_entry_m3,
    float* __restrict__ nifa_entry_m3,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // :1801  qv(k) = MAX(1.E-10, qv1d(k))   -- the density uses the CLAMPED
    // vapour, so the clamp cannot be hoisted out of this kernel.
    const float qv_local = fmaxf(1.0e-10f, qv[idx]);
    const float rho = thompson_aa_density(pressure[idx], temperature[idx],
                                          qv_local);
    rho_out[idx] = rho;

    // :1805  nwfa(k) = MAX(11.1E6, MIN(9999.E6, nwfa1d(k)*rho(k)))
    // :1806  nifa(k) = MAX(naIN1*0.01, MIN(9999.E6, nifa1d(k)*rho(k)))
    //        naIN1*0.01 = 0.5E6*0.01 = 5.0E3 exactly.
    if (nwfa_entry_m3)
        nwfa_entry_m3[idx] = thompson_aa_clamp_nwfa(nwfa[idx] * rho);
    if (nifa_entry_m3)
        nifa_entry_m3[idx] = thompson_aa_clamp_nifa(nifa[idx] * rho);
}


// ---------------------------------------------------------------------------
// 1a. THE NO-MICROPHYSICS COLUMN -- module_mp_thompson.F:1646, :1827-1943,
//     :1990 and the early return at :2020.
// ---------------------------------------------------------------------------
//
// mp_thompson starts every column with no_micro = .true. (:1646) and clears
// it where any entry mixing ratio of cloud, ice, rain, snow or graupel
// exceeds R1 (:1827-1943) or where the entry air is supersaturated over ice,
// ssati > 0 (:1990, with qvsi = rsif at or below 0 C and rslf above,
// :1978-1982, and |ssati| < eps written as zero, :1989).  A column that
// keeps it returns at :2020, before the source loop: nothing after the
// entry block runs, neither a process nor the terminal apply
// (:3972-4082).  Such a column
// leaves mp_thompson with its entry rewrite and nothing else: vapour NOT
// floored at 1.E-10 (:3974 never runs, so a level at exactly zero stays at
// zero) and the aerosol numbers NOT clamped (:3979-3982).  Everywhere else
// :3974 floors every level's vapour at 1.E-10.
//
// This kernel writes 1.0 per column where the column has microphysics and
// 0.0 where WRF returns at :2020, from the ENTRY state (after the entry
// rewrite, before any process), one thread per column.  The terminal apply
// below reads it.  Without it the port floored no vapour anywhere and
// clamped every column's aerosol: on saved real-data frames against WRF
// v4.6.1's own Fortran, 1 to 71 vapour cells per 19,600-column frame at
// exactly 0 where WRF has 1.E-10, and the aerosol clamp applied in columns
// WRF leaves alone (up to 1.5 percent of nwfa at one analysis cell).
// The warm network's entry mask, with WRF's melting level in it
// (:1971-2013).  k_melting is the highest level whose entry tempc =
// temp - 273.15 is above zero, and twet(k) is re-formed from the Bolton
// wet-bulb only at k <= k_melting; everywhere else twet(k) = temp(k).  A
// level at exactly 273.15 K (tempc = 0) belongs to the warm network but
// lies above k_melting unless a warmer level sits at or above it, and there
// WRF keeps twet = temp.  Written per level: 0 below 273.15 K (the cold
// network's), 1 at or above it with twet re-formed, 2 at or above it with
// twet = temp.  Every warm level above 273.15 K is at or below k_melting.
//
// THE BREAKAGE THIS PREVENTS: the warm network re-formed twet at every warm
// level, so a column at exactly 273.15 K with no warmer level above it took
// the wet-bulb melting and collection branches WRF does not take there
// (the column oracle's isothermal 273.15 K, RH 0.001 columns, at every
// time step from 0.001 s to 300 s).
extern "C" __global__ void thompson_aa_entry_warm_mask(
    const float* __restrict__ temperature,
    float* __restrict__ warm_mask,
    int nz, int ncol)
{
    const int column = blockDim.x * blockIdx.x + threadIdx.x;
    if (column >= ncol) return;
    bool at_or_below_melting = false;
    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = (size_t)k * (size_t)ncol + (size_t)column;
        const float temp = temperature[idx];
        // :1977-1981, tempc .le. 0.0 or k_melting = MAX(k, k_melting).
        if (thompson_aa_sub(temp, 273.15f) > 0.0f) at_or_below_melting = true;
        warm_mask[idx] = temp >= 273.15f
            ? (at_or_below_melting ? 1.0f : 2.0f) : 0.0f;
    }
}

extern "C" __global__ void thompson_aa_micro_columns(
    const float* __restrict__ qc,
    const float* __restrict__ qi,
    const float* __restrict__ qr,
    const float* __restrict__ qs,
    const float* __restrict__ qg,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float* __restrict__ micro_columns,
    int nz, int ncol)
{
    const int column = blockDim.x * blockIdx.x + threadIdx.x;
    if (column >= ncol) return;
    float micro = 0.0f;
    for (int k = 0; k < nz; ++k) {
        const size_t idx = (size_t)k * (size_t)ncol + (size_t)column;
        // :1827, :1851, :1878, :1906, :1915.
        if (qc[idx] > THOMPSON_AA_R1 || qi[idx] > THOMPSON_AA_R1
                || qr[idx] > THOMPSON_AA_R1 || qs[idx] > THOMPSON_AA_R1
                || qg[idx] > THOMPSON_AA_R1) {
            micro = 1.0f;
            break;
        }
        // :1800, :1975-1990.  qv(k) = MAX(1.E-10, qv1d(k)); qvsi is rsif
        // where tempc <= 0 and qvs = rslf otherwise; sati = qv/qvsi and
        // ssati = sati - 1, each rounded to REAL(4).
        const float temp = temperature[idx];
        const float qv_local = fmaxf(1.0e-10f, qv[idx]);
        const float tempc = thompson_aa_sub(temp, 273.15f);
        const float qvsi = tempc <= 0.0f
            ? thompson_rsif(pressure[idx], temp)
            : thompson_rslf(pressure[idx], temp);
        float ssati = thompson_aa_sub(thompson_aa_div(qv_local, qvsi), 1.0f);
        if (fabsf(ssati) < THOMPSON_AA_STATE_EPS) ssati = 0.0f;
        if (ssati > 0.0f) {
            micro = 1.0f;
            break;
        }
    }
    micro_columns[column] = micro;
}


// ---------------------------------------------------------------------------
// 1b. ENTRY CLOUD-DROPLET DIAGNOSIS -- module_mp_thompson.F:1826-1842.
// ---------------------------------------------------------------------------
//
// The other half of the entry pack.  Kept in this file so that the entry
// state has exactly one owner; the warm, cold and saturation packages may
// either consume these outputs or call thompson_aa_cloud_dist from the shared
// header themselves -- either way the arithmetic is the header's, so the two
// halves of the scheme cannot drift.
//
// L_qc_out is WRF's L_qc(k) as int32 (1/0).  On the false branch WRF also
// ZEROES qc1d and nc1d in place (:1844-1845); that is done here too, which is
// why qc and nc are mutable.  rc_out is the CONTENT in kg m^-3 (never below
// R1), nc_entry_m3 the rediagnosed droplet number in m^-3.
extern "C" __global__ void thompson_aa_entry_cloud_number(
    float* __restrict__ qc,              // state, per kilogram, zeroed if <= R1
    float* __restrict__ nc,              // state, per kilogram, zeroed if <= R1
    const float* __restrict__ rho,
    float* __restrict__ rc_out,          // kg m^-3
    float* __restrict__ nc_entry_m3,     // m^-3
    int* __restrict__ nu_c_out,
    int* __restrict__ l_qc_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float rho_local = rho[idx];
    const float qc_local = qc[idx];

    if (qc_local > THOMPSON_AA_R1) {
        // Forecast sources diagnose this distribution inline.  With no
        // output requested this branch changes no prognostic field.
        if (!(rc_out || nc_entry_m3 || nu_c_out || l_qc_out)) return;
        // :1828-1841.  thompson_aa_cloud_dist carries WRF's type mixing:
        // a REAL power widened to DOUBLE, REAL size clamps, a DOUBLE
        // rediagnosis.  Do not re-derive it here.
        const float rc = qc_local * rho_local;
        int nu_c = 0;
        double lamc = 0.0;
        const float nc_m3 = thompson_aa_cloud_dist(rc, nc[idx], rho_local,
                                                   &nu_c, &lamc);
        if (rc_out) rc_out[idx] = rc;
        if (nc_entry_m3) nc_entry_m3[idx] = nc_m3;
        if (nu_c_out) nu_c_out[idx] = nu_c;
        if (l_qc_out) l_qc_out[idx] = 1;
    } else {
        // :1843-1848
        qc[idx] = 0.0f;
        nc[idx] = 0.0f;
        if (rc_out) rc_out[idx] = THOMPSON_AA_R1;
        if (nc_entry_m3) nc_entry_m3[idx] = THOMPSON_AA_NC_FLOOR;
        // nu_c is undefined on this branch in WRF (the whole level is
        // switched off by L_qc).  Publish the nc=2 value so a downstream
        // read of a switched-off level is deterministic rather than stale.
        if (nu_c_out) nu_c_out[idx] = thompson_aa_nu_c(THOMPSON_AA_NC_FLOOR);
        if (l_qc_out) l_qc_out[idx] = 0;
    }
}


// ---------------------------------------------------------------------------
// 2.  WORKING AEROSOL REFRESH -- module_mp_thompson.F:3211.
// ---------------------------------------------------------------------------
//
//     nwfa(k) = MAX(11.1E6, (nwfa1d(k) + nwfaten(k)*DT)*rho(k))
//
// This is the SECOND, DISTINCT snapshot, and it is consumed by activ_ncloud
// alone (:3416-3421).  Differences from the entry snapshot at :1805, all
// deliberate:
//   * no 9999.E6 ceiling,
//   * no nifa counterpart anywhere in the scheme,
//   * rho is the TAU+1 density recomputed at :3193 from the post-tendency
//     temp/qv, not the entry rho of :1802.
// The caller must pass that TAU+1 rho.  Use thompson_aa_tau1_density below to
// build it so the definition stays in one place.
//
// nwfa is the READ-ONLY entry per-kg state.  Nothing here writes state.
extern "C" __global__ void thompson_aa_working_number(
    const float* __restrict__ nwfa,      // state, per kilogram, READ-ONLY
    const float* __restrict__ nwfaten,   // accumulator, per kilogram per s
    const float* __restrict__ rho,       // TAU+1 density, :3193
    float dt,
    float* __restrict__ nwfa_work_m3,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float updated = thompson_aa_add(
        nwfa[idx], thompson_aa_mul(nwfaten[idx], dt));
    nwfa_work_m3[idx] = fmaxf(THOMPSON_AA_NWFA_FLOOR,
                              thompson_aa_mul(updated, rho[idx]));
}


// The Exner function mp_gt_driver receives as pii: WRF forms it in phy_prep
// (module_big_step_utilities_em.F) as pi_phy = (p_phy/p1000mb)**rcp, a
// REAL(4) power, i.e. the C library's powf on the gfortran oracle host.  The
// adapter formed it with CuPy's power, which is CUDA's powf: on the 0 ULP
// column oracle's 153 columns it differed from WRF's word at 1131 of 7497
// cells (1 ULP each), and every temperature, saturation and rate of the
// call inherited the difference.  thompson_aa_powf is WOOF's own word.
// rcp = r_d/cp = 287./1004.5 in REAL(4) (module_model_constants.F).
extern "C" __global__ void thompson_aa_exner(
    const float* __restrict__ pressure,      // Pa
    float* __restrict__ pii_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    pii_out[idx] = thompson_aa_powf(
        thompson_aa_div(pressure[idx], 100000.0f),
        thompson_aa_div(287.0f, 1004.5f));
}


// module_mp_thompson.F:3189-3193 -- the TAU+1 state the working refresh and
// the whole condensation region are evaluated on.
//     temp(k) = t1d(k) + DT*tten(k)
//     qv(k)   = MAX(1.E-10, qv1d(k) + DT*qvten(k))
//     rho(k)  = 0.622*pres(k)/(R*temp(k)*(qv(k)+0.622))
// Supplied here as a kernel so that the working refresh, the saturation
// package and the finalize kernel cannot disagree about which density they
// mean.  ArWen's networks write temperature and qv in place rather than
// carrying tten/qvten, so the caller passes the ALREADY-UPDATED TAU+1 fields
// and this kernel only re-applies :3192's vapour floor and forms the density.
// It is deliberately identical arithmetic to thompson_aa_entry_snapshot's
// density; only the inputs differ.
extern "C" __global__ void thompson_aa_tau1_density(
    const float* __restrict__ temperature,   // TAU+1
    const float* __restrict__ pressure,
    const float* __restrict__ qv,            // TAU+1, before the floor
    float* __restrict__ rho_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // :3192  qv(k) = MAX(1.E-10, qv1d(k) + DT*qvten(k))
    const float qv_local = fmaxf(1.0e-10f, qv[idx]);
    rho_out[idx] = thompson_aa_density(pressure[idx], temperature[idx],
                                       qv_local);
}


// ---------------------------------------------------------------------------
// 2b. WORKING CLOUD REFRESH -- module_mp_thompson.F:3213-3221 and :3484-3488.
// ---------------------------------------------------------------------------
//
//     if ((qc1d(k) + qcten(k)*DT) .gt. R1) then
//        rc(k) = (qc1d(k) + qcten(k)*DT)*rho(k)
//        nc(k) = MAX(2., MIN((nc1d(k)+ncten(k)*DT)*rho(k), Nt_c_max))
//        L_qc(k) = .true.
//     else
//        rc(k) = R1 ;  nc(k) = 2. ;  L_qc(k) = .false.
//
// Same shape as the entry diagnosis but WITHOUT the lamc/D0c/D0r rediagnosis:
// this one is a plain clamp of the accumulated value.  It runs twice in WRF,
// once before condensation (:3213) and once after the droplet-evaporation
// block (:3484), with the same code; the rho differs (:3193 vs :3490).
//
// Nothing here writes state.  qc/nc are the read-only entry per-kg fields.
extern "C" __global__ void thompson_aa_working_cloud(
    const float* __restrict__ qc,        // state, per kilogram, READ-ONLY
    const float* __restrict__ qcten,     // accumulator, per kilogram per s
    const float* __restrict__ nc,        // state, per kilogram, READ-ONLY
    const float* __restrict__ ncten,     // accumulator, per kilogram per s
    const float* __restrict__ rho,
    float dt,
    float* __restrict__ rc_work,         // kg m^-3
    float* __restrict__ nc_work_m3,      // m^-3
    int* __restrict__ l_qc_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float rho_local = rho[idx];
    const float qc_updated = thompson_aa_add(
        qc[idx], thompson_aa_mul(qcten[idx], dt));

    if (qc_updated > THOMPSON_AA_R1) {
        rc_work[idx] = thompson_aa_mul(qc_updated, rho_local);
        const float nc_updated = thompson_aa_add(
            nc[idx], thompson_aa_mul(ncten[idx], dt));
        nc_work_m3[idx] = thompson_aa_clamp_nc(
            thompson_aa_mul(nc_updated, rho_local));
        l_qc_out[idx] = 1;
    } else {
        rc_work[idx] = THOMPSON_AA_R1;
        nc_work_m3[idx] = THOMPSON_AA_NC_FLOOR;
        l_qc_out[idx] = 0;
    }
}


// ---------------------------------------------------------------------------
// 3.  TERMINAL APPLY AND CLAMP -- module_mp_thompson.F:3972-4021.
// ---------------------------------------------------------------------------
//
// THE single point at which the three accumulators reach state.  WRF's order
// is reproduced exactly:
//
//   (a) nc1d = MAX(2./rho, MIN(nc1d + ncten*DT, Nt_c_max))
//       -- per-kg value against the volumetric ceiling, see the unit note at
//          the top of this file.
//   (b) nwfa1d = MAX(11.1E6, MIN(9999.E6, nwfa1d + nwfaten*DT))
//       nifa1d = MAX(naIN1*0.01, MIN(9999.E6, nifa1d + nifaten*DT))
//       -- again per-kg values against per-m3 constants, and NOTE that unlike
//          the entry snapshot there is no *rho here at all.
//   (c) if qc1d <= R1 then qc1d = 0, nc1d = 0
//       else rediagnose nc1d through nu_c / lamc / D0c / 2*D0r.
//
// (c) uses the PER-KILOGRAM form
//       lamc = (am_r*ccg(2,nu_c)*ocg1(nu_c)*nc1d/qc1d)**obmr
// in which the densities cancel.  Do NOT algebraically simplify it to the
// entry form: the two differ in operand association
// (entry :1832 is nc*am_r*ccg2*ocg1/rc, terminal :4014 is am_r*ccg2*ocg1*nc/qc)
// and float32 is not associative.
//
// qc is mutable because WRF zeroes it on the (c) false branch.  Every output
// pointer MAY alias its corresponding input (each thread reads its own
// element before writing it), so the adapter can legally pass state.nc for
// both nc and nc_out.
//
// ---------------------------------------------------------------------------
// WHICH DENSITY.  IT IS NOT THE ENTRY DENSITY, AND THE DIFFERENCE IS MEASURED.
// ---------------------------------------------------------------------------
// `rho` MUST be WRF's TAU+1 density as the terminal loop finds it, NOT the
// :1802 entry density.  rho(k) is written in exactly four places in
// mp_thompson -- :1802 (entry), :3193 (the unconditional TAU+1 refresh before
// condensation), :3490 (inside the condensation block, per level) and :3572
// (inside the rain-evaporation block, per level) -- and nothing after :3574
// touches it, so what :3976, :4011, :4019 and :4020 read is whichever of the
// last three ran at that level.
//
// MEASURED, from a build of PRISTINE module_mp_thompson.F carrying only added
// `write` statements (it reproduces all 22 committed column fixtures BYTE FOR
// BYTE), over all 21 fixtures x 24 levels:
//     max |rho_terminal - rho_entry| / rho_terminal = 7.4672e-03
//         (aero-cold-overlap; also 4.1e-03 on aero-reduces-to-classic and
//          7.3e-04 on aero-cloud-freeze-nc)
// and, recomputing :3976-:4021 with the entry density substituted for the
// terminal one inside WRF itself, nc1d changes at exactly one of those 504
// levels -- aero-cold-overlap k=5, 1.833336115 -> 1.826136470 kg^-1, i.e.
//     3.9271e-03 relative, 1963x the 2.0e-06 end-to-end gate.
// The path is :3976's `2./rho(k)` floor; :4011's nu_c and :4020's
// Nt_c_max/rho(k) ceiling read the same rho and are integer-selector and
// ceiling respectively, so they are latent rather than quiet.
//
// HOW A CALLER GETS IT.  ArWen's `temperature` and `qv` arrays hold WRF's
// temp(k) / qv(k) exactly at ONE moment: immediately after the rain-
// evaporation launcher returns and before sedimentation, because WRF's
// :3569-3572 is the last statement that refreshes them and everything after
// :3574 accumulates into tten/qvten without writing temp/qv back.  MEASURED:
// launching thompson_aa_tau1_density there reproduces WRF's terminal rho
// BITWISE at 501 of those 504 levels, worst 1.2334e-07 (inherited float32
// noise on two fixtures), against 7.4672e-03 for the entry density.
// Recomputing it later -- e.g. at the finalize call itself -- is 5.39e-05
// off, because ArWen's temperature keeps absorbing the melt/freeze cleanup's
// tten while WRF's temp(k) snapshot does not.
//
// ---------------------------------------------------------------------------
// THE COLUMN EXIT AND THE VAPOUR FLOOR (:2020, :3974).
// ---------------------------------------------------------------------------
// thompson_aa_state_finalize_with_columns also takes the column flag
// thompson_aa_micro_columns wrote at entry and the vapour.  In a column WRF
// returned from at :2020 the terminal apply never ran: nc, nwfa and nifa
// leave as they entered (nc is already zero there: every level's cloud was
// at or below R1, and the entry diagnosis zeroed it), qc stays at the zero
// the entry rewrite left, and the vapour is not floored.  In every other
// column :3974 writes qv1d = MAX(1.E-10, qv1d + qvten*DT) at every level;
// the port's vapour already holds qv1d + qvten*DT, so the floor is the whole
// of it.  The plain entry point keeps the old contract for the unit gates
// that drive the terminal apply alone.
__device__ __forceinline__ void thompson_aa_state_finalize_impl(
    float* __restrict__ qc,
    const float* __restrict__ nc,
    const float* __restrict__ nwfa,
    const float* __restrict__ nifa,
    const float* __restrict__ ncten,
    const float* __restrict__ nwfaten,
    const float* __restrict__ nifaten,
    const float* __restrict__ rho,
    float dt,
    float* __restrict__ nc_out,
    float* __restrict__ nwfa_out,
    float* __restrict__ nifa_out,
    float* __restrict__ qv,
    const float* __restrict__ micro_columns,
    int ncol, int idx)
{
    if (micro_columns != nullptr) {
        if (micro_columns[idx % ncol] == 0.0f) {
            // :2020.  Each output may alias its input; write it anyway so a
            // non-aliased caller still receives the entry value.
            nc_out[idx] = nc[idx];
            nwfa_out[idx] = nwfa[idx];
            nifa_out[idx] = nifa[idx];
            return;
        }
        // :3974.
        qv[idx] = fmaxf(1.0e-10f, qv[idx]);
    }

    const float rho_local = rho[idx];

#if defined(THOMPSON_AA_WRF39)
    // fork :3692-3696.  No Nt_c_max on the droplet number (audit T20), and
    // the aerosol bounds carry the density: 11.1E6/rho and 9999.E6/rho on a
    // per-kilogram value, naIN1*0.01 unconverted (audit T18).
    float nc_new = fmaxf(
        THOMPSON_AA_NC_FLOOR / rho_local,
        thompson_aa_add(nc[idx], thompson_aa_mul(ncten[idx], dt)));
    nwfa_out[idx] = fmaxf(THOMPSON_AA_NWFA_FLOOR / rho_local,
        fminf(THOMPSON_AA_AERO_CEIL / rho_local,
              thompson_aa_add(nwfa[idx], thompson_aa_mul(nwfaten[idx], dt))));
    nifa_out[idx] = fmaxf(THOMPSON_AA_NIFA_FLOOR,
        fminf(THOMPSON_AA_AERO_CEIL / rho_local,
              thompson_aa_add(nifa[idx], thompson_aa_mul(nifaten[idx], dt))));
#else
    // (a) :3976
    float nc_new = fmaxf(
        THOMPSON_AA_NC_FLOOR / rho_local,
        fminf(thompson_aa_add(nc[idx], thompson_aa_mul(ncten[idx], dt)),
              THOMPSON_AA_NT_C_MAX));

    // (b) :3979-3982.  aer_init_opt < 2 branch; wif_input_opt = 0 so nbca is
    // identically zero and is not carried.
    nwfa_out[idx] = thompson_aa_clamp_nwfa(
        thompson_aa_add(nwfa[idx], thompson_aa_mul(nwfaten[idx], dt)));
    nifa_out[idx] = thompson_aa_clamp_nifa(
        thompson_aa_add(nifa[idx], thompson_aa_mul(nifaten[idx], dt)));
#endif

    // (c) :4008-4021
    const float qc_local = qc[idx];
    if (qc_local <= THOMPSON_AA_R1) {
        qc[idx] = 0.0f;
        nc_out[idx] = 0.0f;
        return;
    }

    const int nu_c = thompson_aa_nu_c(nc_new * rho_local);

    // -----------------------------------------------------------------------
    // EVERY float32 SUB-EXPRESSION BELOW THAT FEEDS A DOUBLE IS PINNED WITH
    // thompson_aa_mul / thompson_aa_div, AND THAT IS LOAD-BEARING.
    // -----------------------------------------------------------------------
    // MEASURED on an RTX 5090 with nvrtc from CUDA 12, options ("-std=c++17",)
    // -- exactly what gpuwm/core/kernels/__init__.py::load_module passes.  For
    // nu_c = 6, qc = 8.9721725e-06 and lamc = 2184250.25 the three spellings
    //     (double)(C1[u]*O2[u]*qc/AM_R) * pow(lamc, 3.0)
    //     float pref = C1[u]*O2[u]*qc/AM_R;  (double)pref * pow(lamc, 3.0)
    //     (double)__fdiv_rn(__fmul_rn(__fmul_rn(C1[u],O2[u]),qc),AM_R) * ...
    // give 3.5430368e+08, 3.5430368e+08 and 3.5430370e+08.  The first two are
    // bit-for-bit the value you get by evaluating the PREFACTOR IN DOUBLE:
    // nvrtc widens the float32 chain when its result is consumed by a double
    // expression, and a named `float` local does NOT stop it.  Only the
    // rounding intrinsics do.
    //
    // WRF has no such freedom.  :4019's ccg(1,nu_c), ocg2(nu_c), qc1d(k) and
    // am_r are all REAL(4), so the prefactor IS rounded to float32 before it
    // meets the DOUBLE lamc**bm_r; same at :4012 for the lambda base and at
    // :4015/:4017 where a REAL(4) quotient is assigned to the DOUBLE lamc.
    // Reproducing that is not pedantry: the terminal rediagnosis CUBES lamc,
    // so the skipped roundings surfaced as a 2.4e-07 to 4.2e-07 end-to-end
    // nc_per_kg residual on nearly every fixture -- 2 to 3.5 float32 ulps,
    // and 38 of 456 fixture states disagreed with a Fortran-faithful host
    // transcription.  With the pins the same comparison is BITWISE.

    // :4012  lamc = (am_r*ccg(2,nu_c)*ocg1(nu_c)*nc1d(k)/qc1d(k))**obmr
    //        REAL base to a REAL power, the REAL result widened to DOUBLE.
    //        thompson_aa_powf, not CUDA's powf: gfortran lowers REAL(4) **
    //        to glibc powf, which is correctly rounded where libdevice's powf
    //        is not, and :4019 then CUBES this lambda.
    double lamc = (double)thompson_aa_powf(
        thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(THOMPSON_AA_AM_R,
                                    THOMPSON_AA_CCG2[nu_c]),
                    THOMPSON_AA_OCG1[nu_c]),
                nc_new),
            qc_local),
        THOMPSON_AA_OBMR);
    // :4013  REAL xDc from a DOUBLE division.  (bm_r + nu_c + 1.) is an exact
    // small integer, so only the division needs care.
    const float xDc = (float)((double)(THOMPSON_AA_BM_R + (float)nu_c + 1.0f)
                              / lamc);
    // :4015 / :4017  cce(2,nu_c)/D0c is a REAL(4) quotient ASSIGNED to the
    // DOUBLE lamc.  D0c = 1.E-6 and D0r*2. = 1.E-4 are not exact in binary, so
    // evaluating these in double instead of float32 shifts lamc: at nu_c = 6,
    // 10.0f/1.0e-6f is 10000000.0 in float32 and 10000000.025 in double.
    if (xDc < THOMPSON_AA_D0C) {
        lamc = (double)thompson_aa_div(THOMPSON_AA_CCE2[nu_c],
                                       THOMPSON_AA_D0C);
    } else if (xDc > THOMPSON_AA_D0R * 2.0f) {
        lamc = (double)thompson_aa_div(THOMPSON_AA_CCE2[nu_c],
                                       THOMPSON_AA_D0R * 2.0f);
    }
    // :4019-4020  MIN(<double>, DBLE(Nt_c_max)/rho(k)) -- here the ceiling IS
    // density-converted, unlike (a).  The prefactor is REAL(4).
    nc_new = (float)fmin(
        (double)thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_CCG1[nu_c],
                                THOMPSON_AA_OCG2[nu_c]),
                qc_local),
            THOMPSON_AA_AM_R)
            * thompson_aa_pow(lamc, (double)THOMPSON_AA_BM_R),
        (double)THOMPSON_AA_NT_C_MAX / (double)rho_local);
    nc_out[idx] = nc_new;
}

extern "C" __global__ void thompson_aa_state_finalize(
    float* __restrict__ qc,              // final per-kg qc, zeroed if <= R1
    const float* __restrict__ nc,        // ENTRY per-kg nc, READ-ONLY
    const float* __restrict__ nwfa,      // ENTRY per-kg nwfa, READ-ONLY
    const float* __restrict__ nifa,      // ENTRY per-kg nifa, READ-ONLY
    const float* __restrict__ ncten,
    const float* __restrict__ nwfaten,
    const float* __restrict__ nifaten,
    // TAU+1 density as of :3972 -- :3193/:3490/:3572, whichever ran last at
    // this level.  NOT the :1802 entry density; see the block above.
    const float* __restrict__ rho,
    float dt,
    float* __restrict__ nc_out,
    float* __restrict__ nwfa_out,
    float* __restrict__ nifa_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    thompson_aa_state_finalize_impl(
        qc, nc, nwfa, nifa, ncten, nwfaten, nifaten, rho, dt,
        nc_out, nwfa_out, nifa_out, nullptr, nullptr, 1, idx);
}

// The production entry point: the terminal apply of a column with
// microphysics, including the :3974 vapour floor, and nothing at all in a
// column WRF returned from at :2020 (micro_columns[column] == 0).
extern "C" __global__ void thompson_aa_state_finalize_with_columns(
    float* __restrict__ qc,
    const float* __restrict__ nc,
    const float* __restrict__ nwfa,
    const float* __restrict__ nifa,
    const float* __restrict__ ncten,
    const float* __restrict__ nwfaten,
    const float* __restrict__ nifaten,
    const float* __restrict__ rho,
    float dt,
    float* __restrict__ nc_out,
    float* __restrict__ nwfa_out,
    float* __restrict__ nifa_out,
    float* __restrict__ qv,
    const float* __restrict__ micro_columns,
    int ncol, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    thompson_aa_state_finalize_impl(
        qc, nc, nwfa, nifa, ncten, nwfaten, nifaten, rho, dt,
        nc_out, nwfa_out, nifa_out, qv, micro_columns, ncol, idx);
}


// ---------------------------------------------------------------------------
// 3b. THE TERMINAL RAIN AND ICE APPLY -- module_mp_thompson.F:4023-4053.
// ---------------------------------------------------------------------------
//
// The v4.6.1 accumulator path's single application of qiten/niten and
// qrten/nrten, and the size bounds WRF puts here, after the :3943-3966 phase
// cleanup.  rho is the terminal density (see 3 above); a column WRF left at
// :2020 keeps its entry state.  WRF re-forms ni1d and nr1d from the slope at
// every level that keeps its mass, bounded or not, so this does too.
extern "C" __global__ void thompson_aa_terminal_rain_ice(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ qi,
    float* __restrict__ ni,
    const float* __restrict__ qrten,
    const float* __restrict__ nrten,
    const float* __restrict__ qiten,
    const float* __restrict__ niten,
    const float* __restrict__ rho,
    const float* __restrict__ micro_columns,
    float dt, int ncol, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    if (micro_columns[idx % ncol] == 0.0f) return;
    const float rho_local = rho[idx];

    // :4023-4039.
    const float qi_new = thompson_aa_add(qi[idx],
                                         thompson_aa_mul(qiten[idx], dt));
    float ni_new = fmaxf(thompson_aa_div(THOMPSON_AA_R2, rho_local),
        thompson_aa_add(ni[idx], thompson_aa_mul(niten[idx], dt)));
    if (qi_new <= THOMPSON_AA_R1) {
        qi[idx] = 0.0f;
        ni[idx] = 0.0f;
    } else {
        double lami = (double)thompson_aa_powf(
            thompson_aa_div(thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_AM_I, THOMPSON_AA_CIG2),
                THOMPSON_AA_OIG1), ni_new), qi_new),
            THOMPSON_AA_OBMI);
        const double ilami = 1.0 / lami;
        const float xdi = (float)(4.0 * ilami);
        if (xdi < 5.0e-6f) {
            lami = (double)thompson_aa_div(4.0f, 5.0e-6f);
        } else if (xdi > 300.0e-6f) {
            lami = (double)thompson_aa_div(4.0f, 300.0e-6f);
        }
        ni_new = (float)fmin(
            (double)thompson_aa_div(
                thompson_aa_mul(thompson_aa_mul(1.0f, 1.0f / 6.0f), qi_new),
                THOMPSON_AA_AM_I) * thompson_aa_pow(lami, 3.0),
            999.0e3 / (double)rho_local);
        qi[idx] = qi_new;
        ni[idx] = ni_new;
    }

    // :4040-4053.
    const float qr_new = thompson_aa_add(qr[idx],
                                         thompson_aa_mul(qrten[idx], dt));
    float nr_new = fmaxf(thompson_aa_div(THOMPSON_AA_R2, rho_local),
        thompson_aa_add(nr[idx], thompson_aa_mul(nrten[idx], dt)));
    if (qr_new <= THOMPSON_AA_R1) {
        qr[idx] = 0.0f;
        nr[idx] = 0.0f;
    } else {
        double lamr = (double)thompson_aa_powf(
            thompson_aa_div(thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_AM_R, 6.0f), 1.0f), nr_new),
                qr_new),
            THOMPSON_AA_OBMR);
        const float mvd_num = thompson_aa_add(
            thompson_aa_add(3.0f, 0.0f), 0.672f);
        float mvd_r = (float)((double)mvd_num / lamr);
        if (mvd_r > 2.5e-3f) {
            mvd_r = 2.5e-3f;
        } else if (mvd_r < thompson_aa_mul(THOMPSON_AA_D0R, 0.75f)) {
            mvd_r = thompson_aa_mul(THOMPSON_AA_D0R, 0.75f);
        }
        lamr = (double)thompson_aa_div(mvd_num, mvd_r);
        nr_new = (float)(
            (double)thompson_aa_mul(thompson_aa_mul(1.0f, 1.0f / 6.0f), qr_new)
            * thompson_aa_pow(lamr, 3.0) / (double)THOMPSON_AA_AM_R);
        qr[idx] = qr_new;
        nr[idx] = nr_new;
    }
}


// ---------------------------------------------------------------------------
// 4.  SURFACE AEROSOL EMISSION -- mp_gt_driver:1310-1327.
// ---------------------------------------------------------------------------
//
//     nwfa1d(kts) = nwfa1d(kts) + nwfa2d(i,j)*dt_in
//     nifa1d(kts) = nifa1d(kts) + nifa2d(i,j)*dt_in
//
// LOWEST MODEL LEVEL ONLY, and DELIBERATELY NOT CLAMPED.  This runs AFTER
// mp_thompson has returned, i.e. after thompson_aa_state_finalize has already
// applied its ceiling, so between this kernel and the next call's entry pack
// nwfa/nifa may legitimately exceed 9999.E6.  The ceiling reappears only at
// the next call's :1805-1806.
//
// Adding MIN(9999.E6, ...) here "for safety" would silently change the
// boundary-layer aerosol budget on EVERY step and would make the model
// diverge from WRF in exactly the regime the emission exists to represent.
// Do not add it.  Fixture aero-sfc-emit is what pins the arithmetic.
//
// nwfa2d / nifa2d are number tendencies, per kilogram per second (the comment
// at :1313-1315 says so explicitly; the field was redefined from a
// concentration to a tendency on 13 May 2013).
//
// Launched over (ny*nx) columns.  Field layout is C-contiguous (nz, ny, nx),
// so the lowest level of column (j,i) is element j*nx + i.
extern "C" __global__ void thompson_aa_surface_emission(
    float* __restrict__ nwfa,            // per kilogram, updated in place
    float* __restrict__ nifa,            // per kilogram, updated in place
    const float* __restrict__ nwfa2d,    // per kilogram per second
    const float* __restrict__ nifa2d,    // per kilogram per second
    float dt,
    int ncolumns)
{
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncolumns) return;

    nwfa[col] = thompson_aa_add(nwfa[col],
                                thompson_aa_mul(nwfa2d[col], dt));
    nifa[col] = thompson_aa_add(nifa[col],
                                thompson_aa_mul(nifa2d[col], dt));
}


// ---------------------------------------------------------------------------
// 5.  thompson_init's SYNTHETIC CCN / IN PROFILE -- thompson_init:493-551.
// ---------------------------------------------------------------------------
//
//     if (hgt(i,1,j) <= 1000.)      h_01 = 0.8
//     elseif (hgt(i,1,j) >= 2500.)  h_01 = 0.01
//     else                          h_01 = 0.8*cos(hgt(i,1,j)*0.001 - 1.0)
//     niCCN3 = -1.0*ALOG(naCCN1/naCCN0)/h_01
//     nwfa(i,1,j) = naCCN1 + naCCN0*exp(-((hgt(i,2,j)-hgt(i,1,j))/1000.)*niCCN3)
//     z1 = hgt(i,2,j)-hgt(i,1,j)
//     nwfa2d(i,j) = nwfa(i,1,j) * 0.000196 * (50./z1)
//     do k = 2, kte
//        nwfa(i,k,j) = naCCN1 + naCCN0*exp(-((hgt(i,k,j)-hgt(i,1,j))/1000.)*niCCN3)
//
// FOUR THINGS THAT ARE EASY TO GET WRONG AND ARE PINNED BY aero-init-profile:
//   * the k=1 level uses the LEVEL-2 height difference, not zero.  WRF is
//     deliberately not evaluating the profile at the surface.
//   * hgt is ABSOLUTE (above sea level) in the h_01 branch but the profile
//     itself uses the AGL difference hgt(k)-hgt(1).  Both, in one formula.
//   * nifa gets the identical shape with naIN0=1.5E6 / naIN1=0.5E6 but there
//     is NO 2-D flux: WRF never derives a nifa2d, and it stays exactly zero.
//   * nc is NEVER touched by thompson_init.  It stays 0 and is bootstrapped
//     by the first call's terminal rediagnosis.
//
// The CCN and IN fills are INDEPENDENT: WRF tests MAXVAL(nwfa) and MAXVAL(nifa)
// separately against eps=1.E-15, so a domain can get one and not the other.
// Those two domain-wide reductions are the launcher's job; this kernel takes
// the two decisions as flags.
//
// Launched over (ny*nx) columns; requires nz >= 2.
extern "C" __global__ void thompson_aa_init_profile(
    const float* __restrict__ hgt,       // (nz, ny, nx), w-level height ASL
    float* __restrict__ nwfa,
    float* __restrict__ nifa,
    float* __restrict__ nwfa2d,
    int fill_ccn,
    int fill_in,
    int nz, int ncolumns)
{
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncolumns) return;

    // CONTRACTION IS PINNED THROUGHOUT THIS KERNEL.  `naCCN1 + naCCN0*exp(x)`
    // is exactly the a*b+c shape nvrtc fuses into a single-rounded FMA, while
    // build_aero.sh compiles the oracle with plain `gfortran -O2` on baseline
    // x86-64, which has no FMA instruction and rounds the multiply and the
    // add separately.  MEASURED: with contraction left on, the nifa profile
    // over a 1500 m terrain column differs from the Fortran-equivalent host
    // transcription by 1 ulp (928957.75 vs 928957.625) at one level; with
    // every operation pinned the two agree exactly.  The same applies to
    // `hgt*0.001 - 1.0` in the h_01 branch.
    const float hgt1 = hgt[col];
    const float hgt2 = hgt[ncolumns + col];

    float h_01;
    if (hgt1 <= 1000.0f) {
        h_01 = 0.8f;
    } else if (hgt1 >= 2500.0f) {
        h_01 = 0.01f;
    } else {
        h_01 = thompson_aa_mul(
            0.8f,
            thompson_aa_cosf_cr(
                thompson_aa_sub(thompson_aa_mul(hgt1, 0.001f), 1.0f)));
    }

    const float z1 = thompson_aa_sub(hgt2, hgt1);

    if (fill_ccn != 0) {
        // naCCN1 = 50.0E6, naCCN0 = 300.0E6  (:96-97)
        const float niCCN3 = thompson_aa_div(
            thompson_aa_mul(
                -1.0f, thompson_aa_logf(thompson_aa_div(50.0e6f,
                                                           300.0e6f))),
            h_01);
        const float first = thompson_aa_add(
            50.0e6f,
            thompson_aa_mul(
                300.0e6f,
                thompson_aa_expf(thompson_aa_mul(
                    -thompson_aa_div(z1, 1000.0f), niCCN3))));
        nwfa[col] = first;
        nwfa2d[col] = thompson_aa_mul(
            thompson_aa_mul(first, 0.000196f),
            thompson_aa_div(50.0f, z1));
        for (int k = 1; k < nz; ++k) {
            const float dz = thompson_aa_sub(hgt[k * ncolumns + col], hgt1);
            nwfa[k * ncolumns + col] = thompson_aa_add(
                50.0e6f,
                thompson_aa_mul(
                    300.0e6f,
                    thompson_aa_expf(thompson_aa_mul(
                        -thompson_aa_div(dz, 1000.0f), niCCN3))));
        }
    }

    if (fill_in != 0) {
        // naIN1 = 0.5E6, naIN0 = 1.5E6  (:94-95).  No 2-D counterpart.
        const float niIN3 = thompson_aa_div(
            thompson_aa_mul(
                -1.0f, thompson_aa_logf(thompson_aa_div(0.5e6f, 1.5e6f))),
            h_01);
        nifa[col] = thompson_aa_add(
            0.5e6f,
            thompson_aa_mul(
                1.5e6f,
                thompson_aa_expf(thompson_aa_mul(
                    -thompson_aa_div(z1, 1000.0f), niIN3))));
        for (int k = 1; k < nz; ++k) {
            const float dz = thompson_aa_sub(hgt[k * ncolumns + col], hgt1);
            nifa[k * ncolumns + col] = thompson_aa_add(
                0.5e6f,
                thompson_aa_mul(
                    1.5e6f,
                    thompson_aa_expf(thompson_aa_mul(
                        -thompson_aa_div(dz, 1000.0f), niIN3))));
        }
    }
}


// ---------------------------------------------------------------------------
// 6.  EFFECTIVE RADIUS -- calc_effectRad:5594-5699.
// ---------------------------------------------------------------------------
//
// The cloud branch is the ONLY place in the whole scheme with a THREE-way
// shape selector (:5637-5643):
//     nc < 100.      -> inu_c = 15
//     nc > 1.E10     -> inu_c = 2       (dead: :5626 already capped nc)
//     otherwise      -> inu_c = MIN(15, NINT(1000.E6/nc) + 2)
// thompson_aa_nu_c is NOT a substitute; thompson_aa_inu_c_effrad is.
//
// It is also the only consumer of the EXACT-INTEGER g_ratio PARAMETER
// (:5611-5613) rather than the runtime ccg(2,n)*ocg1(n) product.  At nu_c=12
// those differ: 2730 exactly versus 2729.9973.  That is precisely why
// thompson.cu:343's hardcoded 2730.0f is CORRECT for mp=8's effective radius
// while its 2730.0f at :882/:999/:4005/:4128/:4680 is a 1e-6 deviation.
//
// OUTPUT CONVENTION: metres are converted to MICRONS as the very last
// operation, after mp_gt_driver's second clamp at :1476-1478, exactly as
// thompson.cu:373-381 does.  gpuwm's state contract for effc/effi/effs is
// microns; WRF's own driver does re*1.E6 on the way into radiation.
//
// ===========================================================================
// WHAT THIS PACKAGE MEASURED ABOUT calc_effectRad, AND WHERE THE FIX LIVES
// ===========================================================================
// The three branch helpers below come from thompson_aerosol_common.cuh.  When
// WP-04 first gated them they were verbatim transcriptions of
// thompson.cu:339-368 and the header said so ("thompson_field_a/
// thompson_field_b still keep thompson.cu's plain form; nothing has measured
// them").  Driving THIS kernel with the ORACLE's own post-step column -- so
// that nothing but calc_effectRad is under test and no upstream residual can
// be blamed -- measured two deviations from WRF v4.6.1:
//
//   1. THE sa/sb POLYNOMIAL ASSOCIATION.  :5684-5688 spells the 5th, 7th,
//      8th, 9th and 10th terms as sa(5)*tc0*tc0, sa(7)*tc0*tc0*cse(1),
//      sa(8)*tc0*cse(1)*cse(1), sa(9)*tc0*tc0*tc0, sa(10)*cse(1)**3, and
//      Fortran multiplication is LEFT-associative; thompson.cu's
//      thompson_field_a precomputes tc2 = tc*tc / moment2 = moment*moment and
//      forms sa[4]*tc2, sa[6]*(tc2*moment), sa[7]*(tc*moment2),
//      sa[8]*(tc2*tc), sa[9]*(moment2*moment).  float32 multiplication is not
//      associative, and :5689's a_ = 10.0**loga_ AMPLIFIES the difference by
//      ln(10).  MEASURED on oracle-aero/aero-scav-frozen's after-column
//      (T = 260 K exactly at every level, so the fixture's temperature
//      round-trip is exact and nothing else can be blamed): the old spelling
//      gives loga_ = -2.3101425 where gfortran -O2 gives -2.3101428, and
//      re_qs came out 4.0e-7 to 6.4e-7 HIGH at all 9 snowy levels.  On
//      aero-cold-overlap the same defect reached 2.305e-6 -- it BREACHED the
//      port's 2e-6 end-to-end gate.
//
//   2. CUDA powf.  gfortran lowers REAL(4) ** to glibc powf (<= ~0.5 ulp);
//      CUDA's powf carries several.  MEASURED across the 19 committed
//      after-columns: 7 cloud levels and 5 ice levels differed from the
//      oracle by exactly 1 ulp.  Independently: glibc powf and
//      thompson_aa_powf (evaluate in double, round once) agree on 19983 of
//      20000 random cube-root arguments and 19987 of 20000 random snow
//      (smob, b) pairs, so the correctly-rounded helper is the right stand-in
//      for the oracle's libm and CUDA's powf is not.
//
// Both are now repaired in thompson_aerosol_common.cuh -- WP-02's file, fixed
// concurrently by the package that owns it and reached independently from the
// cold network's snow moments -- so this file does NOT carry a private copy.
// A private copy would be a second place for the same arithmetic to drift.
// What this file owns is the composition and the receipts:
// tests/test_thompson_aerosol_state_gpu.py gates all three branches BITWISE
// against all 19 committed after-columns and against probe-effectrad.csv.
//
// CONSEQUENCE FOR THE mp=8 IDENTITY: at nc = Nt_c the two kernels no longer
// agree BITWISE on every level.  On aero-reduces-to-classic effc and effs are
// still bitwise equal and effi differs at 2 of 24 levels by exactly 1 float32
// ulp (6.567e-08) -- and at those two levels mp=28 is bitwise equal to the
// WRF oracle while frozen mp=8 is not.  That is the same tie-break the shared
// header records for thompson_rslf/thompson_rsif: the authority is WRF, not
// ArWen's mp=8, and thompson.cu stays byte-frozen.

// -------------------------------------------------------------------------
// THE THREE BRANCHES COME FROM THE SHARED HEADER.  THEY USED NOT TO.
// -------------------------------------------------------------------------
// This file used to carry private copies named thompson_aa_state_eff_rad_
// cloud / _ice / _snow, on two grounds that were true when they were written
// and are no longer true:
//
// (1) THE SHARED ONES' float32 CHAINS WERE UNPINNED, AND nvrtc WIDENS SUCH
//     CHAINS WHEN IT FEELS LIKE IT.  Every branch ends in `(double)<float
//     expression>`.  MEASURED, in this very file: thompson_aa_state_finalize's
//     :4019 prefactor was spelled exactly that way and nvrtc evaluated it in
//     DOUBLE, which cost 2 to 3.5 float32 ulps of droplet number on nearly
//     every fixture and took 38 of 456 states off a Fortran-faithful host
//     transcription.  A named `float` local does not stop it; only __fmul_rn
//     / __fdiv_rn do.  WRF's operands at :5646, :5654, :5663 and :5694 are
//     all REAL(4), so every one of those roundings is part of the answer.
//     (Receipt: tests/test_thompson_aerosol_state_gpu.py::
//     test_state_finalize_rounds_every_real4_subexpression_that_feeds_a_double
//     compiles the three spellings side by side and measures the split.)
//     thompson_aerosol_common.cuh NOW PINS ALL THREE with the identical
//     __fmul_rn / __fdiv_rn / __fadd_rn spelling these copies used, and its
//     own header block carries the 960-state measurement showing the pins are
//     bit-for-bit inert on today's toolchain and load-bearing as a guarantee.
//
// (2) THE ICE BRANCH'S POWER WAS A LIVE DISAGREEMENT.  :5654 is
//     REAL(4)**REAL(4), which gfortran lowers to glibc powf.  MEASURED over
//     the 19 committed after-columns: with CUDA's powf the ice branch misses
//     the Fortran by exactly 1 float32 ulp at several levels; with
//     thompson_aa_powf it is bitwise wherever the fixture's own
//     temperature is exact.  It also costs the mp=8 effective-radius
//     identity, and the shared header was flipped between the two spellings
//     twice while this package was measuring.  The header now pins
//     thompson_aa_powf there for good, with MP28_PORT_SPEC.md's tie-break
//     -- the authority is WRF, not ArWen's mp=8 -- written next to it.
//
// So the two copies are gone.  Two implementations of one WRF formula is how
// a port acquires a silent split, and the source scan in
// tests/test_thompson_aerosol_device_helpers.py::
// test_shared_helpers_are_defined_exactly_once_and_only_in_the_header is what
// keeps them gone.  thompson_field_a / thompson_field_b were already shared
// for the same reason.

// calc_effectRad's whole body for one level, :5624-5695.  Both kernels below
// share it so the micron and metre gates cannot drift apart.
__device__ __forceinline__ void thompson_aa_calc_effect_rad(
    float t_k, float p_pa, float qv_kg, float qc_kg, float nc_kg,
    float qi_kg, float ni_kg, float qs_kg,
    float* __restrict__ reqc, float* __restrict__ reqi,
    float* __restrict__ reqs)
{
    // :5624-5634.  Every constant is single precision because the WRF
    // declarations are default REAL; only the two lambdas are DOUBLE.
    const float rho = thompson_aa_density(p_pa, t_k, qv_kg);
    const float rc = fmaxf(THOMPSON_AA_R1, qc_kg * rho);
    // :5626 + :5627.  is_aerosol_aware is TRUE for mp=28, so :5627's
    // `nc(k) = Nt_c` override does NOT run and the prognostic value stands.
#if defined(THOMPSON_AA_WRF39)
    // fork :5246, nc(k) = MAX(R2, nc1d(k)*rho(k)): no 2 per m3 floor and no
    // Nt_c_max (audit T20).
    const float nc_m3 = fmaxf(THOMPSON_AA_R2, nc_kg * rho);
#else
    const float nc_m3 = thompson_aa_clamp_nc(nc_kg * rho);
#endif
    const float ri = fmaxf(THOMPSON_AA_R1, qi_kg * rho);
    const float ni_m3 = fmaxf(THOMPSON_AA_R2, ni_kg * rho);
    const float rs = fmaxf(THOMPSON_AA_R1, qs_kg * rho);

    // :5619-5621, the background values every level starts at.
    *reqc = THOMPSON_AA_RE_QC_BG;
    *reqi = THOMPSON_AA_RE_QI_BG;
    *reqs = THOMPSON_AA_RE_QS_BG;

    // :5636-5648 / :5651-5656 / :5659-5696.  WRF's column-wide has_qc /
    // has_qi / has_qs flags are exactly the disjunction of these per-level
    // tests, and every level that fails them CYCLEs, so a pointwise kernel
    // reproduces the column loop exactly.
    if (rc > THOMPSON_AA_R1 && nc_m3 > THOMPSON_AA_R2) {
        *reqc = thompson_aa_eff_rad_cloud(rc, nc_m3);
    }
#if defined(THOMPSON_AA_WRF39)
    // fork :5275 and :5315: ice at least 5.01 and snow at least 10 microns
    // (audit T19).
    if (ri > THOMPSON_AA_R1 && ni_m3 > THOMPSON_AA_R2) {
        *reqi = thompson_aa_wrf39_eff_rad_ice(ri, ni_m3);
    }
    if (rs > THOMPSON_AA_R1) {
        *reqs = thompson_aa_wrf39_eff_rad_snow(rs, t_k);
    }
#else
    if (ri > THOMPSON_AA_R1 && ni_m3 > THOMPSON_AA_R2) {
        *reqi = thompson_aa_eff_rad_ice(ri, ni_m3);
    }
    if (rs > THOMPSON_AA_R1) {
        *reqs = thompson_aa_eff_rad_snow(rs, t_k);
    }
#endif
}

extern "C" __global__ void thompson_aa_effective_radius(
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ nc,        // per kilogram
    const float* __restrict__ qi,
    const float* __restrict__ ni,        // per kilogram
    const float* __restrict__ qs,
    float* __restrict__ effc,
    float* __restrict__ effi,
    float* __restrict__ effs,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    float reqc, reqi, reqs;
    thompson_aa_calc_effect_rad(
        temperature[idx], pressure[idx], qv[idx], qc[idx], nc[idx],
        qi[idx], ni[idx], qs[idx], &reqc, &reqi, &reqs);

    // mp_gt_driver:1476-1478, then the metre->micron convention.
    effc[idx] = fmaxf(THOMPSON_AA_RE_QC_BG, fminf(reqc, 50.0e-6f)) * 1.0e6f;
    effi[idx] = fmaxf(THOMPSON_AA_RE_QI_BG, fminf(reqi, 125.0e-6f)) * 1.0e6f;
    effs[idx] = fmaxf(THOMPSON_AA_RE_QS_BG, fminf(reqs, 999.0e-6f)) * 1.0e6f;
}


// Metre-valued variant of the same computation, for gating directly against
// gpuwm/data/thompson/oracle-aero/probe-effectrad.csv, which is a call to
// calc_effectRad itself and therefore carries neither mp_gt_driver's second
// clamp nor gpuwm's micron convention.  Same arithmetic, different tail.
extern "C" __global__ void thompson_aa_effective_radius_metres(
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ nc,
    const float* __restrict__ qi,
    const float* __restrict__ ni,
    const float* __restrict__ qs,
    float* __restrict__ effc,
    float* __restrict__ effi,
    float* __restrict__ effs,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    thompson_aa_calc_effect_rad(
        temperature[idx], pressure[idx], qv[idx], qc[idx], nc[idx],
        qi[idx], ni[idx], qs[idx], &effc[idx], &effi[idx], &effs[idx]);
}


#if defined(THOMPSON_AA_WRF39)
// ---------------------------------------------------------------------------
// THE FORK'S GRAUPEL INTERCEPT, a column pass (RunConfig.thompson_version =
// "wrf_39_noaa", audit T1).
// ---------------------------------------------------------------------------
//
// fork :2031-2054 (at entry), :3110-3133 (after the sources, for the
// fallout) and :5486-5510 (calc_refl10cm).  With k_0 the highest level at or
// above 270.65 K, from the top down:
//   xslw1 = 4.01 + alog10(mvd_r)   above k_0 where rain has mvd_r > 100 um,
//           0.01                    elsewhere
//   ygra1 = 4.31 + alog10(max(5.E-5, rg))
//   zans1 = 3.1 + 100./(300.*xslw1*ygra1/(10./xslw1+1.+0.25*ygra1)
//                       + 30. + 10.*ygra1)
//   N0_exp = max(gonv_min, min(10.**zans1, gonv_max)), then the running
//   minimum from the top: N0 never increases downward.
// rand1 is zero (no stochastic block in the operational namelist, audit
// T26).  Every level is visited, graupel or not (rg = R1 placeholder), as
// the fork's loop is.  n0_out receives N0_exp per level; the consumers form
// the slope with thompson_aa_wrf39_graupel_slope.
//
// ``mode`` selects which of the three passes runs:
//   1  the entry block (fork :1878-1905): a rain level with no number is
//      seeded at a 1 mm mean volume diameter, mvd_r clamped to
//      [37.5 um, 2.5 mm], graupel present where qg > R1;
//   0  the post-source rebuild (fork :3013-3025): no seed, same clamps;
//   2  calc_refl10cm (fork :5391-5420): mvd_r = 3.672*ilamr unclamped,
//      graupel present where qg > R2.  This mode writes the graupel NUMBER
//      per kilogram, N0_g*ilamg/rho, in place of N0_exp: with it the shared
//      calc_refl10cm column (refl.cu) forms N0_g*720*ilamg**7*(am_g/900)**2
//      of the fork's distribution, because that Rayleigh moment is
//      rg**2/(36*ng) whatever density the kernel assumes.
// The density is formed from the temperature, pressure and vapour passed
// in, as each block does.
extern "C" __global__ void thompson_aa_wrf39_graupel_intercept(
    const float* __restrict__ qg,
    const float* __restrict__ qr,
    const float* __restrict__ nr,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float* __restrict__ n0_out,
    int mode, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    int k_0 = 0;
    for (int k = nz - 1; k >= 0; --k) {
        if (temperature[IDX3(k, j, i)] >= 270.65f) k_0 = max(k_0, k);
    }
    const float am_r = THOMPSON_AA_AM_R;
    const float mvd_numerator = (3.0f + 0.0f) + 0.672f;
    double n0_min = THOMPSON_AA_WRF39_GONV_MAX;
    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        const bool l_qr = qr[idx] > THOMPSON_AA_R1;
        float mvd_r = 0.0f;
        if (l_qr) {
            const float rr = qr[idx] * rho;
            float rain_number = fmaxf(THOMPSON_AA_R2, nr[idx] * rho);
            if (mode == 1 && rain_number <= THOMPSON_AA_R2) {
                // fork :1883-1887.
                const double lamr_seed = (double)__fdiv_rn(
                    mvd_numerator, 1.0e-3f);
                rain_number = (float)((double)thompson_aa_mul(
                    0.16666667163372040f, rr) * thompson_aa_pow(lamr_seed, 3.0)
                    / (double)am_r);
            }
            // lamr = (am_r*crg(3)*org2*nr/rr)**obmr, REAL into DOUBLE.
            const double lamr = (double)thompson_aa_powf(
                thompson_aa_div(
                    thompson_aa_mul(thompson_aa_mul(am_r, 6.0f), rain_number),
                    rr),
                THOMPSON_AA_OBMR);
            mvd_r = (float)((double)mvd_numerator / lamr);
            if (mode == 2) {
                // fork :5402, (3.0 + mu_r + 0.672) * ilamr(k), no clamp.
                mvd_r = (float)((double)mvd_numerator * (1.0 / lamr));
            } else if (mvd_r > 2.5e-3f) {
                mvd_r = 2.5e-3f;
            } else if (mvd_r < thompson_aa_mul(THOMPSON_AA_D0R, 0.75f)) {
                mvd_r = thompson_aa_mul(THOMPSON_AA_D0R, 0.75f);
            }
        }
        const float graupel_floor = mode == 2 ? THOMPSON_AA_R2
                                              : THOMPSON_AA_R1;
        const float rg = qg[idx] > graupel_floor
            ? qg[idx] * rho : THOMPSON_AA_R1;
        float xslw1 = 0.01f;
        if (k > k_0 && l_qr && mvd_r > 100.0e-6f) {
            xslw1 = thompson_aa_add(4.01f, thompson_aa_log10f(mvd_r));
        }
        const float ygra1 = thompson_aa_add(
            4.31f, thompson_aa_log10f(fmaxf(5.0e-5f, rg)));
        const float numerator = thompson_aa_mul(
            thompson_aa_mul(300.0f, xslw1), ygra1);
        const float denominator = thompson_aa_add(
            thompson_aa_add(thompson_aa_div(10.0f, xslw1), 1.0f),
            thompson_aa_mul(0.25f, ygra1));
        const float sum = thompson_aa_add(
            thompson_aa_add(thompson_aa_div(numerator, denominator), 30.0f),
            thompson_aa_mul(10.0f, ygra1));
        const float zans1 = thompson_aa_add(
            3.1f, thompson_aa_div(100.0f, sum));
        double n0_exp = (double)thompson_aa_powf(10.0f, zans1);
        n0_exp = fmax(THOMPSON_AA_WRF39_GONV_MIN,
                      fmin(n0_exp, THOMPSON_AA_WRF39_GONV_MAX));
        n0_min = fmin(n0_exp, n0_min);
        if (mode == 2) {
            double lamg, ilamg, n0_g;
            thompson_aa_wrf39_graupel_slope(
                (float)n0_min, rg, &lamg, &ilamg, &n0_g);
            n0_out[idx] = (float)(n0_g * ilamg / (double)rho);
        } else {
            n0_out[idx] = (float)n0_min;
        }
    }
}

// fork :3755, the graupel half of the terminal apply: graupel at or below
// R1 leaves as zero.  The fork carries no graupel number.
extern "C" __global__ void thompson_aa_wrf39_graupel_finalize(
    float* __restrict__ qg, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (qg[idx] <= THOMPSON_AA_R1) qg[idx] = 0.0f;
}
// fork :587-603: every domain start replaces the surface CCN emission
// with the analysed lowest-level number. Keep the source's REAL(4)
// log/power sequence rather than its algebraic 2e-4 approximation.
extern "C" __global__ void thompson_aa_wrf39_start_emission(
    const float* __restrict__ nwfa,
    float* __restrict__ nwfa2d, float dx, float dy, int columns)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= columns) return;
    const float spacing = sqrtf(thompson_aa_mul(dx, dy));
    const float relative_spacing = __fdiv_rn(spacing, 20000.0f);
    float scale = 0.875f;
    if (relative_spacing < 1.0f) {
        scale = thompson_aa_mul(thompson_aa_add(0.875f,
            thompson_aa_mul(0.125f,
                __fdiv_rn(thompson_aa_sub(20000.0f, spacing), 16000.0f))),
            relative_spacing);
    }
    // A non-positive analysed number has a defined zero emission.
    const float number = nwfa[idx];
    if (!(number > 0.0f)) { nwfa2d[idx] = 0.0f; return; }
    const float exponent = thompson_aa_sub(
        thompson_aa_log10f(thompson_aa_mul(number, 1.0e-6f)), 3.69897f);
    const float emission = thompson_aa_powf(10.0f, exponent);
    nwfa2d[idx] = thompson_aa_mul(
        thompson_aa_mul(emission, scale), 1.0e6f);
}
#endif  // THOMPSON_AA_WRF39


#if defined(THOMPSON_AA_HAIL)
// ---------------------------------------------------------------------------
// THE HAIL-AWARE GRAUPEL STATE (mp_physics=38): entry rewrite and terminal
// apply.  QNGRAUPEL (ng, # per kg) and QVGRAUPEL (qb, m3 per kg) are
// prognostic state here, read at mp_gt_driver:1260-1264 and written back at
// :1342-1347.
// ---------------------------------------------------------------------------

// :1915-1949's state side, before any process runs.  Where the graupel
// mixing ratio is at or below R1 WRF zeroes qg1d, ng1d and qb1d (:1941-1943,
// copied back into the state whether or not the column has microphysics);
// elsewhere it clamps the volume so the bulk density qg/qb lies in
// [rho_g(1), rho_g(NRHG)] = [50, 800] kg m-3 and writes the clamped volume
// back into qb1d (:1920-1922).  ng1d itself is not touched: the :1924-1939
// seed and size clamps act on the local ng(k) only.
extern "C" __global__ void thompson_aa_hail_graupel_entry(
    float* __restrict__ qg,
    float* __restrict__ ng,
    float* __restrict__ qb,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    const float qg1d = qg[idx];
    if (qg1d > THOMPSON_AA_R1) {
        float rb = fmaxf(thompson_aa_div(qg1d, THOMPSON_AA_HAIL_RHO_G[9]),
                         qb[idx]);
        rb = fminf(thompson_aa_div(qg1d, THOMPSON_AA_HAIL_RHO_G[1]), rb);
        qb[idx] = rb;
    } else {
        qg[idx] = 0.0f;
        ng[idx] = 0.0f;
        qb[idx] = 0.0f;
    }
}

// :4058-4078, the terminal apply of the graupel mass, number and volume on
// the terminal density rho(k) (`rho`), in a column WRF ran mp_thompson for
// (the :2020 early exit leaves the entry-rewritten state, which is all
// zeros there because every qg1d was at or below R1).
extern "C" __global__ void thompson_aa_hail_graupel_terminal(
    float* __restrict__ qg,
    float* __restrict__ ng,
    float* __restrict__ qb,
    const float* __restrict__ qgten,
    const float* __restrict__ ngten,
    const float* __restrict__ qbten,
    const float* __restrict__ rho,
    const float* __restrict__ micro_columns,
    float dt, int ncol, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    if (micro_columns[idx % ncol] == 0.0f) return;
#if defined(GPUWM_HAIL_ORACLE_MUTATION)
    // Test-only one-bit change of the prognostic graupel-number tendency.
    const_cast<float*>(ngten)[idx] = __uint_as_float(__float_as_uint(ngten[idx]) ^ 1u);
#endif
    const float qg_new = thompson_aa_add(qg[idx],
                                         thompson_aa_mul(qgten[idx], dt));
    float ng_new = fmaxf(thompson_aa_div(THOMPSON_AA_R2, rho[idx]),
        thompson_aa_add(ng[idx], thompson_aa_mul(ngten[idx], dt)));
    if (qg_new <= THOMPSON_AA_R1) {
        qg[idx] = 0.0f;
        ng[idx] = 0.0f;
        qb[idx] = 0.0f;
        return;
    }
    float qb_new = fmaxf(thompson_aa_div(qg_new, THOMPSON_AA_HAIL_RHO_G[9]),
        thompson_aa_add(qb[idx], thompson_aa_mul(qbten[idx], dt)));
    qb_new = fminf(thompson_aa_div(qg_new, THOMPSON_AA_HAIL_RHO_G[1]),
                   qb_new);
    const int idx_bg = thompson_aa_hail_density_bin(
        thompson_aa_mul(thompson_aa_div(qg_new, qb_new), 0.01f));
    double lamg = thompson_aa_hail_lambda(ng_new, qg_new, idx_bg);
    float mvd_g = (float)((double)THOMPSON_AA_HAIL_MVD_NUM / lamg);
    if (mvd_g > 25.4e-3f) {
        mvd_g = 25.4e-3f;
    } else if (mvd_g < THOMPSON_AA_D0R) {
        mvd_g = THOMPSON_AA_D0R;
    }
    // :4076, unconditional: the slope is re-formed from the bounded MVD.
    lamg = (double)thompson_aa_div(THOMPSON_AA_HAIL_MVD_NUM, mvd_g);
    ng_new = thompson_aa_hail_number_from_lambda(qg_new, lamg, idx_bg);
    qg[idx] = qg_new;
    ng[idx] = ng_new;
    qb[idx] = qb_new;
}
#endif  // THOMPSON_AA_HAIL

#if defined(THOMPSON_AA_HAIL)
// mp_gt_driver:1225 forms Exner with WRF's REAL constants.
extern "C" __global__ void thompson_aa_hail_exner(
    const float* pressure, const float* theta, float* pii, float* temperature, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    pii[idx] = gfk_pow(__fdiv_rn(pressure[idx], 100000.0f), __fdiv_rn(2.0f, 7.0f));
    temperature[idx] = __fmul_rn(theta[idx], pii[idx]);
}
#endif

// ===========================================================================
// THE CLASSIC GRAUPEL NUMBER AND THE 10 cm REFLECTIVITY, WRF v4.6.1.
// ===========================================================================
//
// mp=28's own transcriptions of mp_gt_driver :1266-1281 (the call's private
// ng1d), :4058-4077 (its terminal bound) and calc_refl10cm :5710-6028 with
// module_mp_radar.F :265-590 (the melting-snow backscatter).  The adapter
// used to call thompson.cu's classic kernels for all three; that unit is
// byte-frozen for mp=8 and its arithmetic is not WRF's: CUDA's powf, pow
// and log10 where gfortran calls the C library, the graupel slope taken
// straight from the intercept with a size clamp the driver does not apply,
// 720 where crg(4) = cgg(4,1) = 720.000061, binary64 slopes where WRF
// forms REAL(4) ones, PI5 and lamda4 in binary64 where WRF forms them from
// REAL(4) literals, and the complex quotients and the radar module's
// association order differ.  The 0 ULP column oracle measured the refl
// field differing at 3,492 of 7,497 cells.

// The radar module's state as radar_init leaves it, read back bit for bit
// from the oracle build (module_mp_radar.F :76-80): PI5 =
// 3.14159**5 in REAL(4) arithmetic, lamda4 = lamda_radar**4 with
// lamda_radar = 0.10 a REAL(4) literal, and m_w_0, m_i_0, K_w from the
// COMPLEX(4)-literal refractive-index formulas.  gpuwm.core.refl.radar_init
// computes those five in binary64 for the other schemes; the size bins,
// their widths and the Simpson weights it builds are WRF's bit for bit and
// arrive through the device table.
#define THOMPSON_AA_RADAR_PI5     0x1.3204ba0000000p+8
#define THOMPSON_AA_RADAR_LAMDA4  0x1.a36e305532621p-14
#define THOMPSON_AA_RADAR_K_W     0x1.de4b5a79d9de5p-1
#define THOMPSON_AA_RADAR_MW_RE   0x1.213d1a0000000p+3
#define THOMPSON_AA_RADAR_MW_IM   (-0x1.634d6c0000000p+0)
#define THOMPSON_AA_RADAR_MI_RE   0x1.c91dae74d3b24p+0
#define THOMPSON_AA_RADAR_MI_IM   (-0x1.1acb622ae704dp-13)
#define THOMPSON_AA_RADAR_NRBINS  50
// module_mp_radar.F :283, PIx.
#define THOMPSON_AA_RADAR_PIX     3.1415926535897932384626434

// Rain (:127-129, :697-721): crg(3) = 6, crg(4) = 720.000061, org2 = 1,
// cre(2) = 1, cre(4) = 7; snow cse(3) = 4 and ocms = oams**obms.
#define THOMPSON_AA_CRG3   6.0f
#define THOMPSON_AA_CRG4   720.000061f
#define THOMPSON_AA_ORG2   1.0f
#define THOMPSON_AA_CRE2   1.0f
#define THOMPSON_AA_CRE4   7.0f
#define THOMPSON_AA_CSE3   4.0f
#define THOMPSON_AA_OCMS   3.80693507f
#define THOMPSON_AA_OBMS   0.5f
#define THOMPSON_AA_CGE2   1.0f

// mp_gt_driver :1266-1281: the classic wrapper's ng1d, diagnosed from the
// entry graupel on the driver's own density (:1234, the RAW qv1d, no floor).
// N0_exp, lam_exp and lamg are the driver's DOUBLE PRECISION locals
// (:1137), ygra1 and zans1 REAL.  Zero where the graupel is at or below R1.
extern "C" __global__ void thompson_aa_graupel_number_init(
    const float* __restrict__ qg,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float* __restrict__ graupel_number,
    const int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const float q = qg[idx];
    if (!(q > THOMPSON_AA_R1)) {
        graupel_number[idx] = 0.0f;
        return;
    }
    const float rho = thompson_aa_air_density(
        pressure[idx], temperature[idx], qv[idx]);
    const float ygra1 = thompson_aa_log10f(
        fmaxf(1.0e-9f, thompson_aa_mul(q, rho)));
    float zans1 = thompson_aa_add(3.0f, thompson_aa_mul(
        2.0f / 7.0f, thompson_aa_add(ygra1, 8.0f)));
    zans1 = fmaxf(2.0f, fminf(zans1, 6.0f));
    const double n0_exp = (double)thompson_aa_powf(10.0f, zans1);
    const double lam_exp = thompson_aa_pow(__ddiv_rn(__dmul_rn(__dmul_rn(
        n0_exp, (double)THOMPSON_AA_AM_G5), (double)THOMPSON_AA_CGG1),
        (double)thompson_aa_mul(rho, q)), (double)THOMPSON_AA_OGE1);
    const double lamg = __dmul_rn(lam_exp, (double)thompson_aa_powf(
        thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_CGG3, THOMPSON_AA_OGG2),
                        THOMPSON_AA_OGG1), THOMPSON_AA_OBMG));
    const float number = (float)__ddiv_rn(__dmul_rn((double)thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_CGG2, THOMPSON_AA_OGG3),
                        rho), q),
        thompson_aa_pow(lamg, (double)THOMPSON_AA_BM_G)),
        (double)THOMPSON_AA_AM_G5);
    graupel_number[idx] = fmaxf(THOMPSON_AA_R2, thompson_aa_div(number, rho));
}

// :4058-4077 for the classic graupel, after every source and fallout
// tendency and the phase cleanup.  qg holds qg1d + qgten*DT and
// graupel_number ng1d + ngten*DT (both in place); rho is rho(k) as the
// terminal apply finds it (terminal_density, the adapter's :3572 density,
// formed before the cleanup moves the temperature).  At or below R1 the
// graupel and its number are zero; otherwise the number is rebuilt from the
// bounded median volume diameter, mvd_g REAL and lamg DOUBLE PRECISION.
extern "C" __global__ void thompson_aa_graupel_number_finalize(
    float* __restrict__ qg,
    float* __restrict__ graupel_number,
    const float* __restrict__ terminal_density,
    const int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const float rho = terminal_density[idx];
    const float q = qg[idx];
    float number = fmaxf(thompson_aa_div(THOMPSON_AA_R2, rho),
                         graupel_number[idx]);
    if (q <= THOMPSON_AA_R1) {
        qg[idx] = 0.0f;
        graupel_number[idx] = 0.0f;
        return;
    }
    double lamg = (double)thompson_aa_powf(thompson_aa_div(thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_AM_G5, THOMPSON_AA_CGG3),
                        THOMPSON_AA_OGG2), number), q), THOMPSON_AA_OBMG);
    // (3.0 + mu_g + 0.672), REAL, over the DOUBLE lamg.
    const float mvd_numerator = thompson_aa_add(
        thompson_aa_add(3.0f, 0.0f), 0.672f);
    float mvd = (float)__ddiv_rn((double)mvd_numerator, lamg);
    if (mvd > 25.4e-3f) {
        mvd = 25.4e-3f;
    } else if (mvd < THOMPSON_AA_D0R) {
        mvd = THOMPSON_AA_D0R;
    }
    lamg = (double)thompson_aa_div(mvd_numerator, mvd);
    number = (float)__ddiv_rn(__dmul_rn((double)thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_CGG2, THOMPSON_AA_OGG3), q),
        thompson_aa_pow(lamg, (double)THOMPSON_AA_BM_G)),
        (double)THOMPSON_AA_AM_G5);
    graupel_number[idx] = number;
}

// ---- COMPLEX*16 arithmetic as gfortran -O2 lowers it ----------------------
//
// Multiplication is inline, (ac - bd) + i(ad + bc), each product rounded
// (baseline x86-64, no FMA).  A REAL operand is applied component-wise.
// Division is GCC's inline wide-range (Smith) division for Fortran
// (-fcx-fortran-rules), not the naive quotient: measured equal to
// gfortran 13.3's quotient on 2e6 random pairs on the oracle host, as were
// z**2 = z*z and D**6 = (D*D*D)**2 (tools/thompson_aerosol_column_oracle).
// ABS, LOG and SQRT are the C library's cabs, clog and csqrt; their bodies
// below are those functions' operation orders on WOOF's own hypot, log,
// log1p and atan2 words.
struct thompson_aa_complex { double re, im; };

__device__ __forceinline__ thompson_aa_complex thompson_aa_cmake(
    double re, double im)
{
    thompson_aa_complex z; z.re = re; z.im = im; return z;
}

__device__ __forceinline__ thompson_aa_complex thompson_aa_cmul(
    thompson_aa_complex a, thompson_aa_complex b)
{
    return thompson_aa_cmake(
        __dsub_rn(__dmul_rn(a.re, b.re), __dmul_rn(a.im, b.im)),
        __dadd_rn(__dmul_rn(a.re, b.im), __dmul_rn(a.im, b.re)));
}

__device__ __forceinline__ thompson_aa_complex thompson_aa_cscale(
    double s, thompson_aa_complex a)
{
    return thompson_aa_cmake(__dmul_rn(s, a.re), __dmul_rn(s, a.im));
}

__device__ __forceinline__ thompson_aa_complex thompson_aa_csub(
    thompson_aa_complex a, thompson_aa_complex b)
{
    return thompson_aa_cmake(__dsub_rn(a.re, b.re), __dsub_rn(a.im, b.im));
}

__device__ __forceinline__ thompson_aa_complex thompson_aa_cadd(
    thompson_aa_complex a, thompson_aa_complex b)
{
    return thompson_aa_cmake(__dadd_rn(a.re, b.re), __dadd_rn(a.im, b.im));
}

__device__ __forceinline__ thompson_aa_complex thompson_aa_cdiv(
    thompson_aa_complex a, thompson_aa_complex b)
{
    if (fabs(b.re) < fabs(b.im)) {
        const double ratio = __ddiv_rn(b.re, b.im);
        const double div = __dadd_rn(__dmul_rn(b.re, ratio), b.im);
        const double tr = __dadd_rn(__dmul_rn(a.re, ratio), a.im);
        const double ti = __dsub_rn(__dmul_rn(a.im, ratio), a.re);
        return thompson_aa_cmake(__ddiv_rn(tr, div), __ddiv_rn(ti, div));
    }
    const double ratio = __ddiv_rn(b.im, b.re);
    const double div = __dadd_rn(__dmul_rn(b.im, ratio), b.re);
    const double tr = __dadd_rn(__dmul_rn(a.im, ratio), a.re);
    const double ti = __dsub_rn(a.im, __dmul_rn(a.re, ratio));
    return thompson_aa_cmake(__ddiv_rn(tr, div), __ddiv_rn(ti, div));
}

__device__ __forceinline__ double thompson_aa_cabs(thompson_aa_complex z)
{
    return thompson_aa_hypot(z.re, z.im);
}

// csqrt on finite arguments away from the overflow and underflow
// thresholds (refractive indices and their mixtures).
__device__ __forceinline__ thompson_aa_complex thompson_aa_csqrt(
    thompson_aa_complex z)
{
    if (z.im == 0.0) {
        if (z.re < 0.0) {
            return thompson_aa_cmake(0.0, copysign(__dsqrt_rn(-z.re), z.im));
        }
        return thompson_aa_cmake(fabs(__dsqrt_rn(z.re)), copysign(0.0, z.im));
    }
    if (z.re == 0.0) {
        const double r = __dsqrt_rn(__dmul_rn(0.5, fabs(z.im)));
        return thompson_aa_cmake(r, copysign(r, z.im));
    }
    const double d = thompson_aa_hypot(z.re, z.im);
    double r, s;
    if (z.re > 0.0) {
        r = __dsqrt_rn(__dmul_rn(0.5, __dadd_rn(d, z.re)));
        s = __dmul_rn(0.5, __ddiv_rn(z.im, r));
    } else {
        s = __dsqrt_rn(__dmul_rn(0.5, __dsub_rn(d, z.re)));
        r = fabs(__dmul_rn(0.5, __ddiv_rn(z.im, s)));
    }
    return thompson_aa_cmake(r, copysign(s, z.im));
}

// x*x + y*y - 1 for 0.5 <= x < 1, summed exactly from the split products
// in increasing magnitude and rounded once, as the C library's clog forms
// it near the unit circle.
__device__ __forceinline__ double thompson_aa_x2y2m1(double x, double y)
{
    double v[5];
    v[1] = __dmul_rn(x, x);
    v[0] = __fma_rn(x, x, -v[1]);
    v[3] = __dmul_rn(y, y);
    v[2] = __fma_rn(y, y, -v[3]);
    v[4] = -1.0;
    // Insertion sort by magnitude, then each partial sum carried exactly.
    for (int a = 1; a < 5; ++a) {
        const double t = v[a];
        int b = a - 1;
        while (b >= 0 && fabs(v[b]) > fabs(t)) { v[b + 1] = v[b]; --b; }
        v[b + 1] = t;
    }
    for (int n = 0; n <= 3; ++n) {
        const double hi = __dadd_rn(v[n + 1], v[n]);
        const double lo = __dadd_rn(__dsub_rn(v[n + 1], hi), v[n]);
        v[n + 1] = hi;
        v[n] = lo;
        for (int a = n + 2; a < 5; ++a) {
            const double t = v[a];
            int b = a - 1;
            while (b >= n + 1 && fabs(v[b]) > fabs(t)) {
                v[b + 1] = v[b];
                --b;
            }
            v[b + 1] = t;
        }
    }
    return __dadd_rn(__dadd_rn(__dadd_rn(__dadd_rn(v[4], v[3]), v[2]), v[1]),
                     v[0]);
}

// clog on finite nonzero arguments away from the overflow and underflow
// thresholds.
__device__ __forceinline__ thompson_aa_complex thompson_aa_clog(
    thompson_aa_complex z)
{
    double absx = fabs(z.re), absy = fabs(z.im);
    if (absx < absy) { const double t = absx; absx = absy; absy = t; }
    const double eps = 0x1p-52;
    double real;
    if (absx == 1.0) {
        real = __dmul_rn(thompson_aa_log1p(__dmul_rn(absy, absy)), 0.5);
    } else if (absx > 1.0 && absx < 2.0 && absy < 1.0) {
        double d2m1 = __dmul_rn(__dsub_rn(absx, 1.0), __dadd_rn(absx, 1.0));
        if (absy >= eps) d2m1 = __dadd_rn(d2m1, __dmul_rn(absy, absy));
        real = __dmul_rn(thompson_aa_log1p(d2m1), 0.5);
    } else if (absx < 1.0 && absx >= 0.5 && absy < __dmul_rn(eps, 0.5)) {
        const double d2m1 = __dmul_rn(__dsub_rn(absx, 1.0),
                                      __dadd_rn(absx, 1.0));
        real = __dmul_rn(thompson_aa_log1p(d2m1), 0.5);
    } else if (absx < 1.0 && absx >= 0.5
               && __dadd_rn(__dmul_rn(absx, absx), __dmul_rn(absy, absy))
                   >= 0.5) {
        real = __dmul_rn(thompson_aa_log1p(thompson_aa_x2y2m1(absx, absy)),
                         0.5);
    } else {
        real = thompson_aa_log(thompson_aa_hypot(absx, absy));
    }
    return thompson_aa_cmake(real, thompson_aa_atan2(z.im, z.re));
}

// m_complex_maxwellgarnett (:544-590) on radar_init's inclusion string
// 'spheroidal' (:106-118).  *error is set where WRF's volume check fails.
__device__ __forceinline__ thompson_aa_complex thompson_aa_maxwellgarnett(
    double vol1, double vol2, double vol3, thompson_aa_complex m1,
    thompson_aa_complex m2, thompson_aa_complex m3, int* error)
{
    if (fabs(__dsub_rn(__dadd_rn(__dadd_rn(vol1, vol2), vol3), 1.0)) > 1.0e-6) {
        *error = 1;
        return thompson_aa_cmake(-999.99, -999.99);
    }
    const thompson_aa_complex one = thompson_aa_cmake(1.0, 0.0);
    const thompson_aa_complex m1t = thompson_aa_cmul(m1, m1);
    const thompson_aa_complex m2t = thompson_aa_cmul(m2, m2);
    const thompson_aa_complex m3t = thompson_aa_cmul(m3, m3);
    // beta = 2.0d0*m1t/(mt-m1t) * (mt/(mt-m1t)*LOG(mt/m1t)-1.0d0)
    const thompson_aa_complex d2 = thompson_aa_csub(m2t, m1t);
    const thompson_aa_complex d3 = thompson_aa_csub(m3t, m1t);
    const thompson_aa_complex beta2 = thompson_aa_cmul(
        thompson_aa_cdiv(thompson_aa_cscale(2.0, m1t), d2),
        thompson_aa_csub(thompson_aa_cmul(thompson_aa_cdiv(m2t, d2),
            thompson_aa_clog(thompson_aa_cdiv(m2t, m1t))), one));
    const thompson_aa_complex beta3 = thompson_aa_cmul(
        thompson_aa_cdiv(thompson_aa_cscale(2.0, m1t), d3),
        thompson_aa_csub(thompson_aa_cmul(thompson_aa_cdiv(m3t, d3),
            thompson_aa_clog(thompson_aa_cdiv(m3t, m1t))), one));
    const double host = __dsub_rn(__dsub_rn(1.0, vol2), vol3);
    const thompson_aa_complex numerator = thompson_aa_cadd(thompson_aa_cadd(
        thompson_aa_cscale(host, m1t),
        thompson_aa_cmul(thompson_aa_cscale(vol2, beta2), m2t)),
        thompson_aa_cmul(thompson_aa_cscale(vol3, beta3), m3t));
    const thompson_aa_complex vb2 = thompson_aa_cscale(vol2, beta2);
    const thompson_aa_complex vb3 = thompson_aa_cscale(vol3, beta3);
    const thompson_aa_complex denominator = thompson_aa_cadd(
        thompson_aa_cmake(__dadd_rn(host, vb2.re), vb2.im), vb3);
    return thompson_aa_csqrt(thompson_aa_cdiv(numerator, denominator));
}

// rayleigh_soak_wetgraupel (:265-358) for snow, with get_m_mix_nested
// (:362-489) on radar_init's snow strings: host 'air', matrix 'water',
// hostmatrix 'icewater' -- the ice-in-water mixture, then that mixture as
// the matrix of an air inclusion.  C_back in m^2; zero where WRF's own
// volume check fails or the particle is smaller than 1d-12 m.
__device__ __forceinline__ double thompson_aa_rayleigh_soak_wetsnow(
    double x_g, double fmelt)
{
    const double a_geo = (double)THOMPSON_AA_OCMS;
    const double b_geo = (double)THOMPSON_AA_OBMS;
    const thompson_aa_complex m_w = thompson_aa_cmake(
        THOMPSON_AA_RADAR_MW_RE, THOMPSON_AA_RADAR_MW_IM);
    const thompson_aa_complex m_i = thompson_aa_cmake(
        THOMPSON_AA_RADAR_MI_RE, THOMPSON_AA_RADAR_MI_IM);
    const thompson_aa_complex m_air = thompson_aa_cmake(1.0, 0.0);
    const double fm = fmax(fmin(fmelt, 1.0), 0.0);
    double mra = fmax(fmin(0.9, 1.0), 0.0);
    mra = __dadd_rn(mra, __dmul_rn(__dsub_rn(1.0, mra), fm));
    const double x_w = __dmul_rn(x_g, fm);
    const double d_g = __dmul_rn(a_geo, thompson_aa_pow(x_g, b_geo));
    if (!(d_g >= 1.0e-12)) return 0.0;
    double vg = __dmul_rn(THOMPSON_AA_RADAR_PIX / 6.0,
                          __dmul_rn(__dmul_rn(d_g, d_g), d_g));
    const double rhog = fmax(fmin(__ddiv_rn(x_g, vg), 900.0), 10.0);
    vg = __ddiv_rn(x_g, rhog);
    const double grenz = __dsub_rn(1.0, __ddiv_rn(rhog, 1000.0));
    double volg;
    if (mra <= grenz) {
        volg = __dmul_rn(vg, __dsub_rn(1.0, __dmul_rn(mra, fm)));
    } else {
        const double fmgrenz = __ddiv_rn(__dsub_rn(900.0, rhog),
            __dadd_rn(__dsub_rn(__dmul_rn(mra, 900.0), rhog),
                      __ddiv_rn(__dmul_rn(900.0, rhog), 1000.0)));
        if (fm <= fmgrenz) {
            volg = __dmul_rn(__dsub_rn(1.0, __dmul_rn(mra, fm)), vg);
        } else {
            volg = __dadd_rn(__ddiv_rn(__dsub_rn(x_g, x_w), 900.0),
                             __ddiv_rn(x_w, 1000.0));
        }
    }
    // (6.0 / PIx * volg) ** (1./3.), the exponent a REAL(4) literal.
    const double d_large = thompson_aa_pow(
        __dmul_rn(6.0 / THOMPSON_AA_RADAR_PIX, volg),
        (double)(1.0f / 3.0f));
    const double volice = __ddiv_rn(__dsub_rn(x_g, x_w), __dmul_rn(volg, 900.0));
    const double volwater = __ddiv_rn(x_w, __dmul_rn(1000.0, volg));
    const double volair = __dsub_rn(__dsub_rn(1.0, volice), volwater);
    // get_m_mix_nested, host 'air'.
    int error = 0;
    const double vol1 = __ddiv_rn(volice,
                                  fmax(__dadd_rn(volice, volwater), 1.0e-10));
    const double vol2 = __dsub_rn(1.0, vol1);
    // get_m_mix matrix 'water': MG(volwater, volair, volice, m_w, m_a, m_i).
    const thompson_aa_complex mtmp = thompson_aa_maxwellgarnett(
        vol2, 0.0, vol1, m_w, m_air, m_i, &error);
    // hostmatrix 'icewater' -> get_m_mix matrix 'ice':
    // MG(volice, volair, volwater, m_i, m_a, m_w) on (mtmp, m_a, 2*m_a).
    const thompson_aa_complex m_core = thompson_aa_maxwellgarnett(
        __dsub_rn(1.0, volair), volair, 0.0, mtmp, m_air,
        thompson_aa_cscale(2.0, m_air), &error);
    if (error != 0) return 0.0;
    const thompson_aa_complex m2 = thompson_aa_cmul(m_core, m_core);
    const double kfac = thompson_aa_cabs(thompson_aa_cdiv(
        thompson_aa_cmake(__dsub_rn(m2.re, 1.0), m2.im),
        thompson_aa_cmake(__dadd_rn(m2.re, 2.0), m2.im)));
    const double d3 = __dmul_rn(__dmul_rn(d_large, d_large), d_large);
    return __ddiv_rn(__dmul_rn(__dmul_rn(__dmul_rn(kfac, kfac),
                                         THOMPSON_AA_RADAR_PI5),
                               __dmul_rn(d3, d3)),
                     THOMPSON_AA_RADAR_LAMDA4);
}

// calc_refl10cm (:5710-6028), then mp_gt_driver :1460-1462's MAX(-35., dBZ),
// one thread per column and nothing stored per level: the melting level
// k_0 is found first, every level's three Rayleigh sums are formed from the
// state, and rs(k_0) is recomputed where the melting snow needs it.
// ke_diag = kte.  melting = 0 leaves the melting-snow term out
// (gpuwm.core.refl.refl_melting_for: the MPAS column seam's default).  ng is
// the call's private graupel number after
// thompson_aa_graupel_number_finalize; tables is gpuwm.core.refl's device
// table (xxDs, xdts and the Simpson weights at offsets 0, 50 and 200).
__device__ __forceinline__ void thompson_aa_refl_level(
    float qv, float qr, float nr, float qs, float qg, float ng, float qb,
    float t, float p, float* rs, float* smob, float* smoc, float* ze_rain,
    float* ze_snow, float* ze_graupel, bool* l_qr, bool* l_qs, bool* l_qg)
{
    // :5745-5786.
    const float rho = thompson_aa_air_density(p, t, fmaxf(1.0e-10f, qv));
    *l_qr = qr > THOMPSON_AA_R1;
    *l_qs = qs > THOMPSON_AA_R2;
    *l_qg = qg > THOMPSON_AA_R2;
    *ze_rain = 1.0e-22f;
    *ze_snow = 1.0e-22f;
    *ze_graupel = 1.0e-22f;
    *rs = THOMPSON_AA_R1;
    *smob = 0.0f;
    *smoc = 0.0f;
    if (*l_qr) {
        const float rr = thompson_aa_mul(qr, rho);
        const float nr_m3 = fmaxf(THOMPSON_AA_R2, thompson_aa_mul(nr, rho));
        const double lamr = (double)thompson_aa_powf(thompson_aa_div(
            thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                THOMPSON_AA_AM_R, THOMPSON_AA_CRG3), THOMPSON_AA_ORG2),
                nr_m3), rr), THOMPSON_AA_OBMR);
        const double ilamr = __ddiv_rn(1.0, lamr);
        const double n0_r = __dmul_rn(
            (double)thompson_aa_mul(nr_m3, THOMPSON_AA_ORG2),
            thompson_aa_pow(lamr, (double)THOMPSON_AA_CRE2));
        // :5855, ze_rain = N0_r*crg(4)*ilamr**cre(4).
        *ze_rain = (float)__dmul_rn(__dmul_rn(n0_r, (double)THOMPSON_AA_CRG4),
            thompson_aa_pow(ilamr, (double)THOMPSON_AA_CRE4));
    }
    if (*l_qs) {
        // :5800-5841, bm_s = 2 so smo2 = smob.
        *rs = thompson_aa_mul(qs, rho);
        const float tc0 = fminf(-0.1f, thompson_aa_sub(t, 273.15f));
        *smob = thompson_aa_mul(*rs, THOMPSON_AA_OAMS);
        *smoc = thompson_aa_mul(thompson_field_a(tc0, THOMPSON_AA_CSE1),
            thompson_aa_powf(*smob, thompson_field_b(tc0, THOMPSON_AA_CSE1)));
        const float smoz = thompson_aa_mul(
            thompson_field_a(tc0, THOMPSON_AA_CSE3),
            thompson_aa_powf(*smob, thompson_field_b(tc0, THOMPSON_AA_CSE3)));
        // :5856-5857, the constant prefix folded in REAL(4) as gfortran
        // folds it.
        const float prefix = thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
            thompson_aa_mul(thompson_aa_div(0.176f, 0.93f),
                thompson_aa_div(6.0f, THOMPSON_AA_PI)),
            thompson_aa_div(6.0f, THOMPSON_AA_PI)),
            thompson_aa_div(THOMPSON_AA_AM_S, 900.0f)),
            thompson_aa_div(THOMPSON_AA_AM_S, 900.0f));
        *ze_snow = thompson_aa_mul(prefix, smoz);
    }
    if (*l_qg) {
        // :5787-5794 and :5847-5852 (idx_bg1).
#if defined(THOMPSON_AA_HAIL)
        float rb = fmaxf(thompson_aa_div(qg, THOMPSON_AA_HAIL_RHO_G[9]), qb);
        rb = fminf(thompson_aa_div(qg, THOMPSON_AA_HAIL_RHO_G[1]), rb);
        const int bin = thompson_aa_hail_density_bin(
            thompson_aa_mul(thompson_aa_div(qg, rb), 0.01f));
        const float am_g = THOMPSON_AA_HAIL_AM_G[bin];
#else
        const float am_g = THOMPSON_AA_AM_G5;
#endif
        const float rg = thompson_aa_mul(qg, rho);
        const float ng_m3 = fmaxf(THOMPSON_AA_R2, thompson_aa_mul(ng, rho));
        const double lamg = (double)thompson_aa_powf(thompson_aa_div(
            thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                am_g, THOMPSON_AA_CGG3), THOMPSON_AA_OGG2),
                ng_m3), rg), THOMPSON_AA_OBMG);
        const double ilamg = __ddiv_rn(1.0, lamg);
        const double n0_g = __dmul_rn(
            (double)thompson_aa_mul(ng_m3, THOMPSON_AA_OGG2),
            thompson_aa_pow(lamg, (double)THOMPSON_AA_CGE2));
        // :5858-5860: the REAL(4) prefix (constants folded, then the two
        // am_g/900 factors), then DOUBLE PRECISION with N0_g.
        const float am_g_900 = thompson_aa_div(am_g, 900.0f);
        const float prefix = thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
            thompson_aa_mul(thompson_aa_div(0.176f, 0.93f),
                thompson_aa_div(6.0f, THOMPSON_AA_PI)),
            thompson_aa_div(6.0f, THOMPSON_AA_PI)), am_g_900), am_g_900);
        *ze_graupel = (float)__dmul_rn(__dmul_rn(__dmul_rn(
            (double)prefix, n0_g), (double)THOMPSON_AA_CGG4),
            thompson_aa_pow(ilamg, (double)THOMPSON_AA_CGE4));
    }
}

extern "C" __global__ void thompson_aa_refl10cm(
    const float* __restrict__ qv,
    const float* __restrict__ qr,
    const float* __restrict__ nr,
    const float* __restrict__ qs,
    const float* __restrict__ qg,
    const float* __restrict__ ng,
    const float* __restrict__ qb,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const double* __restrict__ tables,
    float* __restrict__ refl,
    const int melting, const int nz, const int ny, const int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;
    const double* xxds = tables;
    const double* xdts = tables + THOMPSON_AA_RADAR_NRBINS;
    const double* simpson = tables + 4 * THOMPSON_AA_RADAR_NRBINS;

    // :5868-5878, k_0: the level above the highest above-freezing rain
    // level with snow or graupel above it.
    bool melti = false;
    int k_0 = 0;
    float rs_k0 = THOMPSON_AA_R1;
    bool l_qs_k0 = false;
    for (int k = nz - 2; melting && k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const size_t up = IDX3(k + 1, j, i);
        if (temperature[idx] > 273.15f && qr[idx] > THOMPSON_AA_R1
                && (qs[up] > THOMPSON_AA_R2 || qg[up] > THOMPSON_AA_R2)) {
            k_0 = max(k + 1, k_0);
            melti = true;
            break;
        }
    }
    if (melti) {
        float smob, smoc, zr, zs, zg;
        bool lr, lg;
        const size_t idx = IDX3(k_0, j, i);
        thompson_aa_refl_level(qv[idx], qr[idx], nr[idx], qs[idx], qg[idx],
            ng[idx], qb == nullptr ? 0.0f : qb[idx], temperature[idx], pressure[idx], &rs_k0, &smob, &smoc,
            &zr, &zs, &zg, &lr, &l_qs_k0, &lg);
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        float rs, smob, smoc, ze_rain, ze_snow, ze_graupel;
        bool l_qr, l_qs, l_qg;
        thompson_aa_refl_level(qv[idx], qr[idx], nr[idx], qs[idx], qg[idx],
            ng[idx], qb == nullptr ? 0.0f : qb[idx], temperature[idx], pressure[idx], &rs, &smob, &smoc,
            &ze_rain, &ze_snow, &ze_graupel, &l_qr, &l_qs, &l_qg);
        // :5873-5952, melting snow below k_0 (the graupel block is
        // commented out in v4.6.1).  k_0 >= 2 in Fortran's 1-based index.
        if (melti && k_0 >= 1 && k < k_0 && l_qs && l_qs_k0) {
            const double fmelt = fmax(0.05, fmin(
                __dsub_rn(1.0, (double)thompson_aa_div(rs, rs_k0)), 0.99));
            const float om3 = thompson_aa_div(1.0f, smoc);
            const float m0 = thompson_aa_mul(smob, om3);
            const float mrat = thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(smob, m0), m0), m0);
            const float slam1 = thompson_aa_mul(m0, THOMPSON_AA_LAM0);
            const float slam2 = thompson_aa_mul(m0, THOMPSON_AA_LAM1);
            double eta = 0.0;
            for (int n = 0; n < THOMPSON_AA_RADAR_NRBINS; ++n) {
                const double d = xxds[n];
                // x = am_s * xxDs(n)**bm_s, bm_s = 2.
                const double x = __dmul_rn((double)THOMPSON_AA_AM_S,
                                           __dmul_rn(d, d));
                const double cback = thompson_aa_rayleigh_soak_wetsnow(
                    x, fmelt);
                // f_d = Mrat*(Kap0*DEXP(-slam1*xxDs(n))
                //       + Kap1*(M0*xxDs(n))**mu_s * DEXP(-slam2*xxDs(n)))
                const double fd = __dmul_rn((double)mrat, __dadd_rn(
                    __dmul_rn((double)THOMPSON_AA_KAP0, thompson_aa_exp(
                        -__dmul_rn((double)slam1, d))),
                    __dmul_rn(__dmul_rn((double)THOMPSON_AA_KAP1,
                        thompson_aa_pow(__dmul_rn((double)m0, d),
                                        (double)THOMPSON_AA_MU_S)),
                        thompson_aa_exp(-__dmul_rn((double)slam2, d)))));
                eta = __dadd_rn(eta, __dmul_rn(__dmul_rn(
                    __dmul_rn(fd, cback), simpson[n]), xdts[n]));
            }
            // ze_snow(k) = SNGL(lamda4 / (pi5 * K_w) * eta)
            ze_snow = (float)__dmul_rn(__ddiv_rn(THOMPSON_AA_RADAR_LAMDA4,
                __dmul_rn(THOMPSON_AA_RADAR_PI5, THOMPSON_AA_RADAR_K_W)), eta);
        }
        // :5955-5957, dBZ = 10.*log10((ze_rain+ze_snow+ze_graupel)*1.d18).
        const float sum = thompson_aa_add(thompson_aa_add(ze_rain, ze_snow),
                                          ze_graupel);
        const float dbz = (float)__dmul_rn(10.0, thompson_aa_log10(
            __dmul_rn((double)sum, 1.0e18)));
        refl[idx] = fmaxf(-35.0f, dbz);
    }
}

#if defined(THOMPSON_AA_HAIL)
// WRF :4065-4070 density bin and distribution slope, output-only.
extern "C" __global__ void thompson_aa_hail_size(
    const float* qg, const float* ng, const float* qb,
    float* density, float* mvd_mm, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    if (qg[idx] <= THOMPSON_AA_R1) {
        density[idx] = 0.0f;
        mvd_mm[idx] = 0.0f;
        return;
    }
    const float rb = fminf(__fdiv_rn(qg[idx], 50.0f),
                          fmaxf(__fdiv_rn(qg[idx], 800.0f), qb[idx]));
    density[idx] = __fdiv_rn(qg[idx], rb);
    const int bin = thompson_aa_hail_density_bin(__fmul_rn(density[idx], 0.01f));
    const double lamg = thompson_aa_hail_lambda(ng[idx], qg[idx], bin);
    mvd_mm[idx] = __fmul_rn((float)__ddiv_rn((double)THOMPSON_AA_HAIL_MVD_NUM, lamg), 1000.0f);
}
#endif
#if defined(THOMPSON_AA_HAIL)
extern "C" __global__ void thompson_aa_hail_size_maximum(
    const float* mvd_mm, float* column_maximum, int nz, int ncol)
{
    const int column = blockDim.x * blockIdx.x + threadIdx.x;
    if (column >= ncol) return;
    float maximum = 0.0f;
    for (int k = 0; k < nz; ++k) {
        const float value = mvd_mm[k*ncol + column];
        if (!isfinite(value)) { column_maximum[column] = value; return; }
        maximum = fmaxf(maximum, value);
    }
    column_maximum[column] = maximum;
}
#endif
