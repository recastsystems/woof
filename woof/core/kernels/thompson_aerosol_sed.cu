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
// gpuwm/core/kernels/thompson_aerosol_sed.cu
//
// Aerosol-aware Thompson (mp_physics=28): number-weighted cloud-water
// sedimentation and the number-conserving final phase cleanup.
//
// Numerical authority is WRF v4.6.1 phys/module_mp_thompson.F
// (WRF v4.6.1, commit d66e442fccc04111067e29274c9f9eaccc3cef28, zero local
// modifications).  Every line number below refers to that file.
//
// gpuwm/core/kernels/thompson_aerosol_common.cuh is PREPENDED to this
// translation unit by gpuwm/core/kernels/__init__.py::_EXTRA_HEADERS.  Do not
// #include it and do not duplicate any helper it publishes.
//
// ---------------------------------------------------------------------------
// WHY THIS IS A SEPARATE TRANSLATION UNIT
// ---------------------------------------------------------------------------
// gpuwm/core/kernels/thompson.cu is byte-frozen: its compiled source string is
// the entire mp=8 numerics guarantee.  Its
// thompson_cloud_sediment_held_density_impl (thompson.cu:944-1042) is the
// structural template for the mass channel here, but mp=8 has NO cloud-number
// fallout at all, and it hardcodes
//     100.0e6f              (the constant Nt_c)
//     2730.0f == ccg(2,12)*ocg1(12)
//      272.0f == ccg(5,12)*ocg2(12)
// where mp=28 needs live per-cell nu_c-indexed gamma moments.  A shared
// implementation is impossible without editing thompson.cu, so the mass
// channel is transcribed here and mp=8 stays frozen.
//
// ---------------------------------------------------------------------------
// THE TWO THINGS THAT ARE EASY TO GET SILENTLY WRONG
// ---------------------------------------------------------------------------
// 1. NO SUBSTEPPING.  Rain (:3790-3820), ice (:3840-3870), snow (:3871-3902)
//    and graupel (:3903-3937) all wrap their apply loop in
//    `do n = 1, nstep` and scale every term by onstep(1..4).  CLOUD DOES NOT
//    (:3823-3838): one pass, no onstep factor, no k=kte export term and no
//    surface accumulation.  Copying a rain/ice launcher as a template and
//    keeping its substep loop is a silent rate error that no bounds check
//    would catch.
//
// 2. THE FLOOR IS 10, NOT 2.  :3835 is
//        nc(k) = MAX(10., nc(k) + (sed_n(k+1)-sed_n(k))*odzq*DT)
//    It is the ONLY use of 10 as a droplet-number floor anywhere in the
//    scheme; every other site floors at 2 (THOMPSON_AA_NC_FLOOR).
//
// A third, quieter trap: cloud mass and number leaving level kts are simply
// DISCARDED.  There is no `pptrain`-style accumulation for cloud, so a
// number-budget test must not expect closure.
//
// ---------------------------------------------------------------------------
// ACCUMULATOR CONTRACT
// ---------------------------------------------------------------------------
// state.nc is READ-ONLY entry state (nc1d) for the whole mp=28 call.  These
// kernels never write it.  They write the shared per-kilogram-per-second
// accumulator ncten, which a terminal kernel (WP-04) applies once with WRF's
// clamps at :3972-4021.  Any working per-m3 droplet number is recomputed
// locally the way WRF does at :3216/:3486:
//     nc = MAX(2, MIN((nc1d + ncten*DT)*rho, Nt_c_max))
// Cloud MASS is different: gpuwm applies mass tendencies in place, exactly as
// the frozen mp=8 pipeline does, so qc is read-modify-written here.
//
// ENTRY STATE IS NOT THE RAW STATE ARRAY.  :1844-1846 and :1870-1871 rewrite
// the caller's own column on the way in:
//     else                    ! qc1d(k) .le. R1        (and qi1d(k) .le. R1)
//        qc1d(k) = 0.0                                  qi1d(k) = 0.0
//        nc1d(k) = 0.0                                  ni1d(k) = 0.0
// so cloud_number_entry and ice_number_entry are state.nc / state.ni ZEROED
// wherever the matching condensate was absent at call entry.  Feeding the raw
// arrays instead produces a non-zero working droplet number in air that has no
// droplets -- bounded, finite, and wrong.
//
// ---------------------------------------------------------------------------
// FLOATING-POINT CONTRACTION
// ---------------------------------------------------------------------------
// nvrtc defaults to --fmad=true; the oracle is `gfortran -O2` on baseline
// x86-64 with no FMA instruction, so every REAL(4) multiply and add in WRF is
// separately rounded.  Every expression below that could contract into an FMA
// goes through thompson_aa_add/sub/mul/div.  Pure multiply/divide chains do
// not contract and are written plainly, but still respect Fortran's
// left-to-right association: WRF's
//     nc(k)*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc(k)
// is (((nc*am_r)*ccg2)*ocg1)/rc -- four separately rounded operations, NOT
// mp=8's fused nc*am_r*2730/rc.
//
// MEASURED (tests/test_thompson_aerosol_sed_gpu.py): with this pinning the
// kernel reproduces WRF's ncten, qcten, vtck and vtnck BIT-EXACTLY on the
// aero-nc-sed, aero-reduces-to-classic, aero-nc-cap, aero-warm-overlap and
// aero-cold-overlap columns dumped from an instrumented build of the pristine
// source.
//
// ---------------------------------------------------------------------------
// THE NUMBER CHANNEL HAS NO mp=8 EVIDENCE BEHIND IT, SO IT CARRIES ITS OWN
// ---------------------------------------------------------------------------
// Classic Thompson has no cloud-water number flux at all, so nothing in the
// model-validated mp=8 trajectory constrains vtnck.  If this kernel silently
// reused the MASS fall speed for the number -- the obvious copy-paste -- every
// bound would still hold, the column budget would still close against its own
// fluxes, and the scheme would simply drift nc with no error visible anywhere.
// Three gates in the test module close that hole, none of them fixture-bound:
//   * vtck/vtnck must equal (nu_c+5)(nu_c+4)/((nu_c+2)(nu_c+1)) at every
//     reachable nu_c.  That is an identity, not a measurement: :673-684 with
//     bm_r = 3 and bv_c = 2 makes ccg(5,n)*ocg2(n) = WGAMMA(n+6)/WGAMMA(n+4)
//     and ccg(4,n)*ocg1(n) = WGAMMA(n+3)/WGAMMA(n+1).  A copied mass velocity
//     reads 1.0 against true values from 1.397 to 2.8.
//   * both channels must reproduce (F[k+1]-F[k])*odzq*orho per level, bit for
//     bit, which catches a stray onstep factor in one channel only.
//   * on a uniform slab a copied velocity leaves the mean droplet mass
//     bit-unchanged; the real kernel drops it by ~40% in one 30 s step.
// Both mutations -- swapping CCG4/OCG1 for CCG5/OCG2, and halving the number
// divergence -- were injected and confirmed to fail those gates.
//
// ---------------------------------------------------------------------------
// SHARED HELPERS ARE NOT DEFINED HERE
// ---------------------------------------------------------------------------
// thompson_aa_bound_ice_number used to be duplicated in this file with am_i
// spelled as `3.1415926536f*890.0f/6.0f` where thompson_aerosol_common.cuh
// uses THOMPSON_AA_AM_I.  The two constants are bit-identical in float32 and
// the whole bound is now measured against an independent NumPy transcription
// of :4029-4039, so deleting the copy moved nothing.  It is deleted because
// two definitions in two cupy.RawModule translation units are how the halves
// of a scheme drift apart: nvrtc never diffs them.

#define THOMPSON_AA_KMAX_SHALLOW 64
#define THOMPSON_AA_KMAX_GENERIC 256

//! module_mp_thompson.F:3835.  The single site in the scheme that floors the
//! droplet number at 10 m^-3 instead of THOMPSON_AA_NC_FLOOR (2 m^-3).
#define THOMPSON_AA_NC_SED_FLOOR 10.0f

//! :3656.  Cloud fallout is suppressed in rising air.
#define THOMPSON_AA_SED_W_LIMIT 1.0e-1f

//! :3650.  Fallout depth search stops at 500 m above ground.
#define THOMPSON_AA_SED_HGT_AGL 500.0f


// ---------------------------------------------------------------------------
// Cloud-water sedimentation with the number channel, :3644-3666 + :3823-3838.
// ---------------------------------------------------------------------------
//
// Structural template: thompson.cu:944-1042
// (thompson_cloud_sediment_held_density_impl).  The gating is kept VERBATIM
// from mp=8, because WRF's is identical for both options:
//   * the ksed1(5) fallout-depth search is keyed on cloud MASS rc > R2 and
//     still breaks at 500 m AGL (:3646-3652).  Cloud NUMBER does not extend
//     the fallout depth.
//   * the per-level velocity gate is still rc > R1 .AND. w1d(k) < 1.E-1
//     (:3656).
//
// DENSITY RULE (two distinct densities, and using one for both is an
// invisible error):
//   * reference_density is WRF's HELD pre-adjustment rho -- the value rc and
//     nc were both formed on at :3216/:3486, before rho is refreshed at
//     :3489.  cloud_mass and cloud_number are built on it.
//   * density[k], recomputed here from the current temperature/pressure/qv,
//     is WRF's refreshed rho(k), and it is what converts the flux divergence
//     back into a tendency at :3831-3833.
//   * the rhof fall-speed factor (:3194, :3506, :3614) stays on the held
//     density UNLESS a post-source rain column caused the rain fall-speed
//     pass to refresh rhof for every level first; rain_active_columns carries
//     WRF's ANY(L_qr) for that.  This is mp=8's model, unchanged.
//
// Optional diagnostic outputs (any may be null) expose the exact intermediate
// columns the oracle comparison pins: vtck, vtnck, rc and nc.
template <int KMAX>
__device__ __forceinline__ void thompson_aa_cloud_sediment_impl(
    float* __restrict__ qc,
    const float* __restrict__ cloud_number_entry,
    float* __restrict__ cloud_number_tendency,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ rain_active_columns,
    const float* __restrict__ cloud_active_columns,
    const float* __restrict__ vertical_velocity,
    const float* __restrict__ dz,
    float* __restrict__ out_mass_velocity,
    float* __restrict__ out_number_velocity,
    float* __restrict__ out_cloud_mass,
    float* __restrict__ out_cloud_number,
    float* __restrict__ qcten,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    if (cloud_active_columns != nullptr
            && cloud_active_columns[column] == 0.0f) return;
    const int j = column / nx;
    const int i = column - j * nx;

#ifndef THOMPSON_NO_EXACT_SHORTCUTS
    // Diagnostic entry points retain the full working-column outputs.
    bool empty = out_mass_velocity == nullptr && out_number_velocity == nullptr
        && out_cloud_mass == nullptr && out_cloud_number == nullptr
        && dt >= 0.0f && dt <= 100000.0f;
    for (int k = 0; empty && k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qc_working = qcten != nullptr
            ? thompson_aa_add(qc[idx], thompson_aa_mul(qcten[idx], dt))
            : qc[idx];
        empty = qc_working >= -1.0f && qc_working <= THOMPSON_AA_R1
            && isfinite(cloud_number_entry[idx])
            && isfinite(cloud_number_tendency[idx])
            && reference_density[idx] >= 0.00001f
            && reference_density[idx] <= 100.0f
            && temperature[idx] >= 100.0f && temperature[idx] <= 400.0f
            && pressure[idx] >= 1.0f && pressure[idx] <= 120000.0f
            && qv[idx] >= -1.0f && qv[idx] <= 1.0f
            && dz[idx] >= 1.0f && dz[idx] <= 100000.0f;
    }
    if (empty) {
        // Only the bottom level receives the zero divergence at sediment_top=0.
        const size_t bottom = IDX3(0, j, i);
        cloud_number_tendency[bottom] = thompson_aa_add(
            cloud_number_tendency[bottom], 0.0f);
        // With the accumulator the zero divergence is added to qcten, and
        // qcten + 0 is qcten: nothing to write.
        if (qcten == nullptr) {
            for (int k = 0; k < nz; ++k) {
                const size_t idx = IDX3(k, j, i);
                qc[idx] = thompson_aa_add(qc[idx], thompson_aa_mul(0.0f, dt));
            }
        }
        return;
    }
#endif  // THOMPSON_NO_EXACT_SHORTCUTS

    float density[KMAX];
    float cloud_mass[KMAX];
    float cloud_number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX];
    float number_flux[KMAX];
    float qc_tendency[KMAX];
    float qc_initial[KMAX];

    // module_mp_thompson.F:192, rho_not = 101325.0/(287.05*298.0).  Written
    // as the same runtime expression thompson.cu:970 uses.
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const bool rain_refreshes_rhof = rain_active_columns != nullptr
        && rain_active_columns[column] != 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        density[k] = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        // With the accumulator, qc is qc1d and the tendency starts from
        // WRF's running qcten, so :3832 adds the fallout to it and :3975
        // applies the sum once.  Without it, qc already carries the earlier
        // tendencies and only the fallout's is accumulated here.
        qc_initial[k] = qc[idx];
        qc_tendency[k] = qcten != nullptr ? qcten[idx] : 0.0f;
        const float qc_working = qcten != nullptr
            ? thompson_aa_add(qc[idx], thompson_aa_mul(qcten[idx], dt))
            : qc[idx];
        const float held = reference_density[idx];
        // :3215-3216 / :3484 rc(k) = MAX(R1, (qc1d(k) + qcten(k)*DT)*rho(k))
        cloud_mass[k] = qc_working > THOMPSON_AA_R1
            ? qc_working * held : THOMPSON_AA_R1;
        // :3217 / :3486 nc(k) = MAX(2., MIN((nc1d(k)+ncten(k)*DT)*rho(k),
        //                                   Nt_c_max))
        //
        // The clamp form is applied UNCONDITIONALLY, unlike the mass, even
        // though :3222 assigns a bare 2.0 on the rc <= R1 side.  WRF's own
        // droplet number at this point is whichever of :3222 and :3486 ran
        // last for that level, and the difference is provably dead: sed_n is
        // vtnck*nc, and vtnck is left at zero for every level with
        // rc <= R1 (:3656).  Reproducing the clamp everywhere removes a
        // divergent branch and MEASURES equal to WRF on every level of every
        // aerosol fixture, including the levels the saturation adjustment
        // evaporated down to R1 with a cancellation residue in
        // nc1d + ncten*DT.
        cloud_number[k] = thompson_aa_clamp_nc(
            thompson_aa_mul(
                thompson_aa_add(
                    cloud_number_entry[idx],
                    thompson_aa_mul(cloud_number_tendency[idx], dt)),
                held));
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
    }

    // :3646-3652.  ksed1(:) = 1 at :3598, i.e. sediment_top starts at kts.
    int sediment_top = 0;
    float height_agl = 0.0f;
    for (int k = 0; k < nz - 1; ++k) {
        const size_t idx = IDX3(k, j, i);
        if (cloud_mass[k] > THOMPSON_AA_R2) sediment_top = k;
        height_agl += dz[idx];
        if (height_agl > THOMPSON_AA_SED_HGT_AGL) break;
    }

    // :3654-3665.
    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        if (cloud_mass[k] > THOMPSON_AA_R1
                && vertical_velocity[idx] < THOMPSON_AA_SED_W_LIMIT) {
            // nu_c = MIN(15, NINT(1000.E6/nc(k)) + 2)
            const int nu_c = thompson_aa_nu_c(cloud_number[k]);
            // lamc = (nc(k)*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc(k))**obmr
            const float lambda_arg = thompson_aa_div(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(cloud_number[k], THOMPSON_AA_AM_R),
                        THOMPSON_AA_CCG2[nu_c]),
                    THOMPSON_AA_OCG1[nu_c]),
                cloud_mass[k]);
            // lamc and ilamc are DOUBLE PRECISION in WRF (:1597-1598); the
            // power itself is a REAL**REAL, i.e. a single-precision powf,
            // widened afterwards.  thompson_aa_powf, not CUDA's powf:
            // gfortran lowers REAL**REAL to glibc's correctly-rounded powf
            // while CUDA's carries up to ~2 ulp, and MEASURED over the
            // nu_c = 3..15 ladder that costs up to 2.1e-7 relative on vtck
            // and 1.2e-5 on the resulting ncten.  With the correctly-rounded
            // form every level of every fixture is bit-exact.  (thompson.cu
            // uses the plain powf here; it is frozen, and this is one of the
            // places mp=28 is simply closer to WRF than mp=8 is.)
            const double lambda =
                (double)thompson_aa_powf(lambda_arg, THOMPSON_AA_OBMR);
            const double inverse_lambda = 1.0 / lambda;

            const float velocity_density = rain_refreshes_rhof
                ? density[k] : reference_density[idx];
            const float rhof = sqrtf(rho_not / velocity_density);

            // MASS: vtc = rhof(k)*av_c*ccg(5,nu_c)*ocg2(nu_c) * ilamc**bv_c
            const float mass_prefix = thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(rhof, THOMPSON_AA_AV_C),
                    THOMPSON_AA_CCG5[nu_c]),
                THOMPSON_AA_OCG2[nu_c]);
            // NUMBER (:3663, no mp=8 counterpart):
            //     vtc = rhof(k)*av_c*ccg(4,nu_c)*ocg1(nu_c) * ilamc**bv_c
            const float number_prefix = thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(rhof, THOMPSON_AA_AV_C),
                    THOMPSON_AA_CCG4[nu_c]),
                THOMPSON_AA_OCG1[nu_c]);
            // bv_c is exactly 2.0 (:164), and gfortran -O2 expands the
            // DOUBLE**REAL(2.0) to one exact multiply pair.
            mass_velocity[k] = (float)((double)mass_prefix
                * inverse_lambda * inverse_lambda);
            number_velocity[k] = (float)((double)number_prefix
                * inverse_lambda * inverse_lambda);
        }
    }

    // :3825-3828.  sed_c/sed_n are filled over the whole column, not just the
    // fallout depth, because the apply loop reads index k+1.
    for (int k = nz - 1; k >= 0; --k) {
        mass_flux[k] = mass_velocity[k] * cloud_mass[k];
        number_flux[k] = number_velocity[k] * cloud_number[k];
    }

    if (out_mass_velocity != nullptr || out_number_velocity != nullptr
            || out_cloud_mass != nullptr) {
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            if (out_mass_velocity != nullptr) {
                out_mass_velocity[idx] = mass_velocity[k];
            }
            if (out_number_velocity != nullptr) {
                out_number_velocity[idx] = number_velocity[k];
            }
            if (out_cloud_mass != nullptr) out_cloud_mass[idx] = cloud_mass[k];
        }
    }

    // :3829-3836.  SINGLE PASS.  No nstep loop, no onstep factor, no k=kte
    // export term, no surface accumulation.
    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float odzq = 1.0f / dz[idx];
        const float orho = 1.0f / density[k];
        const float mass_divergence = mass_flux[k + 1] - mass_flux[k];
        const float number_divergence = number_flux[k + 1] - number_flux[k];
        qc_tendency[k] = thompson_aa_add(
            qc_tendency[k],
            thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), orho));
        cloud_number_tendency[idx] = thompson_aa_add(
            cloud_number_tendency[idx],
            thompson_aa_mul(thompson_aa_mul(number_divergence, odzq), orho));
        cloud_mass[k] = fmaxf(
            THOMPSON_AA_R1,
            thompson_aa_add(
                cloud_mass[k],
                thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), dt)));
        cloud_number[k] = fmaxf(
            THOMPSON_AA_NC_SED_FLOOR,
            thompson_aa_add(
                cloud_number[k],
                thompson_aa_mul(thompson_aa_mul(number_divergence, odzq),
                                dt)));
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        // Cloud at or below R1 is carried to the phase cleanup, which
        // freezes any positive cloud below HGFR and adds melted ice to it
        // above 0 C before it removes what is left at or below R1, as WRF's
        // :3943-3966 and terminal :4007-4009 do.  Removing it here took it
        // out of both: cloud that WRF keeps at 1.6e-12 to 2.0e-12 kg/kg,
        // with its droplets, came back as zero on saved real-data columns
        // (tools/thompson_real_column_parity).
        if (qcten != nullptr) {
            qcten[idx] = qc_tendency[k];
        } else {
            qc[idx] = thompson_aa_add(
                qc_initial[k], thompson_aa_mul(qc_tendency[k], dt));
        }
        if (out_cloud_number != nullptr) out_cloud_number[idx] = cloud_number[k];
    }
}


#define THOMPSON_AA_CLOUD_SEDIMENT_PARAMETERS                             \
    float* __restrict__ qc,                                               \
    const float* __restrict__ cloud_number_entry,                         \
    float* __restrict__ cloud_number_tendency,                            \
    const float* __restrict__ temperature,                                \
    const float* __restrict__ pressure,                                   \
    const float* __restrict__ qv,                                         \
    const float* __restrict__ reference_density,                          \
    const float* __restrict__ vertical_velocity,                          \
    const float* __restrict__ dz,                                         \
    float dt, int nz, int ny, int nx

#define THOMPSON_AA_CLOUD_SEDIMENT_ARGUMENTS                              \
    qc, cloud_number_entry, cloud_number_tendency, temperature, pressure, \
    qv, reference_density, (const float*)0, (const float*)0,              \
    vertical_velocity, dz, (float*)0, (float*)0, (float*)0, (float*)0,    \
    (float*)0, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_cloud_sediment_64(
    THOMPSON_AA_CLOUD_SEDIMENT_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_CLOUD_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_cloud_sediment_256(
    THOMPSON_AA_CLOUD_SEDIMENT_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_CLOUD_SEDIMENT_ARGUMENTS);
}


#define THOMPSON_AA_CLOUD_SEDIMENT_RAIN_PARAMETERS                        \
    float* __restrict__ qc,                                               \
    const float* __restrict__ cloud_number_entry,                         \
    float* __restrict__ cloud_number_tendency,                            \
    const float* __restrict__ temperature,                                \
    const float* __restrict__ pressure,                                   \
    const float* __restrict__ qv,                                         \
    const float* __restrict__ reference_density,                          \
    const float* __restrict__ rain_active_columns,                        \
    const float* __restrict__ vertical_velocity,                          \
    const float* __restrict__ dz,                                         \
    float dt, int nz, int ny, int nx

#define THOMPSON_AA_CLOUD_SEDIMENT_RAIN_ARGUMENTS                         \
    qc, cloud_number_entry, cloud_number_tendency, temperature, pressure, \
    qv, reference_density, rain_active_columns, (const float*)0,          \
    vertical_velocity, dz, (float*)0, (float*)0, (float*)0, (float*)0,    \
    (float*)0, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_cloud_sediment_64_with_rain(
    THOMPSON_AA_CLOUD_SEDIMENT_RAIN_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_CLOUD_SEDIMENT_RAIN_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_cloud_sediment_256_with_rain(
    THOMPSON_AA_CLOUD_SEDIMENT_RAIN_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_CLOUD_SEDIMENT_RAIN_ARGUMENTS);
}


// The production entry points.  qcten is WRF's cloud-water accumulator
// (null: qc is read-modify-written in place, as the unit gates drive it).
#define THOMPSON_AA_CLOUD_SEDIMENT_MASKS_PARAMETERS                       \
    float* __restrict__ qc,                                               \
    const float* __restrict__ cloud_number_entry,                         \
    float* __restrict__ cloud_number_tendency,                            \
    const float* __restrict__ temperature,                                \
    const float* __restrict__ pressure,                                   \
    const float* __restrict__ qv,                                         \
    const float* __restrict__ reference_density,                          \
    const float* __restrict__ rain_active_columns,                        \
    const float* __restrict__ cloud_active_columns,                       \
    const float* __restrict__ vertical_velocity,                          \
    const float* __restrict__ dz,                                         \
    float* __restrict__ qcten,                                            \
    float dt, int nz, int ny, int nx

#define THOMPSON_AA_CLOUD_SEDIMENT_MASKS_ARGUMENTS                        \
    qc, cloud_number_entry, cloud_number_tendency, temperature, pressure, \
    qv, reference_density, rain_active_columns, cloud_active_columns,     \
    vertical_velocity, dz, (float*)0, (float*)0, (float*)0, (float*)0,    \
    qcten, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_cloud_sediment_64_with_masks(
    THOMPSON_AA_CLOUD_SEDIMENT_MASKS_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_CLOUD_SEDIMENT_MASKS_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_cloud_sediment_256_with_masks(
    THOMPSON_AA_CLOUD_SEDIMENT_MASKS_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_CLOUD_SEDIMENT_MASKS_ARGUMENTS);
}


// Diagnostic entry point.  Same physics, same code path, but it also writes
// vtck, vtnck, the working rc and the post-fallout nc so the oracle test can
// pin WRF's intermediate columns instead of only the endpoints.  It is not
// used by the forecast adapter.
#define THOMPSON_AA_CLOUD_SEDIMENT_DIAG_PARAMETERS                        \
    float* __restrict__ qc,                                               \
    const float* __restrict__ cloud_number_entry,                         \
    float* __restrict__ cloud_number_tendency,                            \
    const float* __restrict__ temperature,                                \
    const float* __restrict__ pressure,                                   \
    const float* __restrict__ qv,                                         \
    const float* __restrict__ reference_density,                          \
    const float* __restrict__ rain_active_columns,                        \
    const float* __restrict__ cloud_active_columns,                       \
    const float* __restrict__ vertical_velocity,                          \
    const float* __restrict__ dz,                                         \
    float* __restrict__ out_mass_velocity,                                \
    float* __restrict__ out_number_velocity,                              \
    float* __restrict__ out_cloud_mass,                                   \
    float* __restrict__ out_cloud_number,                                 \
    float dt, int nz, int ny, int nx

#define THOMPSON_AA_CLOUD_SEDIMENT_DIAG_ARGUMENTS                         \
    qc, cloud_number_entry, cloud_number_tendency, temperature, pressure, \
    qv, reference_density, rain_active_columns, cloud_active_columns,     \
    vertical_velocity, dz, out_mass_velocity, out_number_velocity,        \
    out_cloud_mass, out_cloud_number, (float*)0, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_cloud_sediment_64_diagnostic(
    THOMPSON_AA_CLOUD_SEDIMENT_DIAG_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_CLOUD_SEDIMENT_DIAG_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_cloud_sediment_256_diagnostic(
    THOMPSON_AA_CLOUD_SEDIMENT_DIAG_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_CLOUD_SEDIMENT_DIAG_ARGUMENTS);
}


// ---------------------------------------------------------------------------
// Terminal ice-number size bound, :4029-4039.
// ---------------------------------------------------------------------------
//
// NOT DEFINED HERE.  thompson_aa_bound_ice_number lives in
// thompson_aerosol_common.cuh and is prepended to this translation unit; see
// its PUBLISHED SHARED SIGNATURES block.  This file used to carry a local
// copy whose only textual difference was spelling am_i as the product
// `3.1415926536f*890.0f/6.0f` where the header uses THOMPSON_AA_AM_I
// (4.660029297e+02f).  The two constants are BIT-IDENTICAL in float32
// (test_am_i_product_form_is_bit_identical_to_the_header_constant proves it,
// and test_bound_ice_number_matches_an_independent_wrf_transcription proves
// the whole bound value-by-value against a NumPy transcription of :4029-4039
// that never touches the CUDA source), so removing the copy moved nothing.
//
// It is deleted because a second definition is exactly how the two halves of
// a scheme drift apart: separate cupy.RawModule translation units mean nvrtc
// never diffs them.  A re-added local copy is now a hard nvrtc redefinition
// error, and test_sed_defines_no_published_shared_helper catches a renamed
// one.
//
// ---------------------------------------------------------------------------
// Number-conserving final phase cleanup, :3943-3966.
// ---------------------------------------------------------------------------
//
// Structural template: thompson.cu:3745-3790 (thompson_final_phase_cleanup).
// The two instantaneous phase transfers run after every fallout tendency and
// before the terminal category bounds.  mp=8 exposes only the carried ICE
// number; mp=28 exposes the DROPLET number on both sides:
//
//   MELT   (temp > T_0 = 273.15, xri > 0), :3947-3953
//     qcten(k) = qcten(k) + xri*odt
//     ncten(k) = ncten(k) + ni1d(k)*odt      <-- the melted ice number
//                                                becomes DROPLET number.
//     qiten(k) = qiten(k) - xri*odt
//     niten(k) = -ni1d(k)*odt                <-- assignment, so the final ni
//                                                is exactly zero.
//   NOTE THE ARGUMENT: WRF credits ncten with ni1d(k), the ENTRY ice number,
//   NOT the current ni.  gpuwm applies ice-number tendencies in place, so the
//   entry value has to be carried in explicitly; passing the live ni here
//   would be a plausible-looking, silently wrong droplet source.
//
//   FREEZE (temp < HGFR = 235.16, xrc > 0), :3956-3965
//     xnc = nc1d(k) + ncten(k)*DT            <-- the TRUE running per-kg
//                                                droplet number: unclamped,
//                                                not multiplied by rho, and
//                                                read AFTER the melt branch.
//     niten(k) = niten(k) + xnc*odt
//     ncten(k) = ncten(k) - xnc*odt
//
// FLAGGED FOR THE RECORD, DELIBERATELY NOT FIXED: thompson.cu:3780 uses
// 100.0e6f/rho for that xnc, ignoring the accumulated ncten -- which is NOT
// zero even in mp=8, since mp=8 accumulates all five droplet sinks plus the
// balance limiter plus sedimentation into it and only discards it at
// mp_gt_driver's writeback.  That is a real pre-existing mp=8 deviation.
// Correcting thompson.cu would move a model-validated trajectory and confound
// the mp=28 gate, so it is recorded, not repaired.
//
// The two branches are written as WRF writes them -- two sequential IFs, not
// an IF/ELSE -- even though T_0 > HGFR makes them mutually exclusive, because
// the freeze branch deliberately reads the qc and ncten the melt branch just
// wrote.
extern "C" __global__ void thompson_aa_final_phase_cleanup(
    float* __restrict__ qc,
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ temperature,
    const float* __restrict__ cloud_number_entry,
    const float* __restrict__ ice_number_entry,
    float* __restrict__ cloud_number_tendency,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float* __restrict__ qcten,
    // WRF's qiten / niten (both given with qcten on the v4.6.1 accumulator
    // path): qi and ni are then the read-only qi1d / ni1d, the melt and the
    // freeze write :3949-3964's tendencies, and the ice size bound waits for
    // the terminal apply, AFTER the freeze, where WRF has it.
    float* __restrict__ qiten,
    float* __restrict__ niten,
    // WRF's tten and the ocp(k) / lvap(k) it multiplies here (all given on
    // the v4.6.1 accumulator path, with theta*exner = t1d): the melt and
    // the freeze add their latent heat to tten as REAL terms (:3953,
    // :3964) and the temperature is re-formed as t1d + tten*DT (:3973).
    // Null: the heat is applied to the temperature in place.
    float* __restrict__ tten,
    const float* __restrict__ heat_ocp,
    const float* __restrict__ heat_lvap,
    const float* __restrict__ theta,
    const float* __restrict__ exner,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const bool accumulate_ice = niten != nullptr;
    const bool accumulate_heat = tten != nullptr;

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float inverse_cp = 1.0f / (1004.0f * (1.0f + 0.887f * qv0));
    const float odt = 1.0f / dt;

    // :3946, `xri = MAX(0.0, qi1d(k) + qiten(k)*DT)`.
    const float qi_working = accumulate_ice
        ? thompson_aa_add(qi[idx], thompson_aa_mul(qiten[idx], dt)) : qi[idx];
    if (temp0 > 273.15f && qi_working > 0.0f) {
        const float transferred = fmaxf(0.0f, qi_working);
        if (qcten != nullptr) {
            // :3949, `qcten(k) = qcten(k) + xri*odt`.
            qcten[idx] = thompson_aa_add(qcten[idx],
                                         thompson_aa_mul(transferred, odt));
        } else {
            qc[idx] += transferred;
        }
        cloud_number_tendency[idx] = thompson_aa_add(
            cloud_number_tendency[idx],
            thompson_aa_mul(ice_number_entry[idx], odt));
        if (accumulate_ice) {
            // :3951-3952.  niten is ASSIGNED, so the terminal ni1d is
            // exactly ni1d - ni1d*odt*DT's rounding of zero.
            qiten[idx] = thompson_aa_sub(qiten[idx],
                                         thompson_aa_mul(transferred, odt));
            niten[idx] = thompson_aa_mul(-ice_number_entry[idx], odt);
        } else {
            qi[idx] = 0.0f;
            ni[idx] = 0.0f;
        }
        if (accumulate_heat) {
            // :3953, tten - lfus*ocp(k)*xri*odt*(1-IFDRY), REAL left to
            // right; lfus = lsub - lvap0 = 334000 exactly.
            tten[idx] = thompson_aa_sub(tten[idx], thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(334000.0f, heat_ocp[idx]),
                                transferred), odt));
        } else {
            temperature[idx] -= 334000.0f * inverse_cp * transferred;
        }
    }

    // :3955, `xrc = MAX(0.0, qc1d(k) + qcten(k)*DT)`, read after the melt.
    const float qc_working = qcten != nullptr
        ? thompson_aa_add(qc[idx], thompson_aa_mul(qcten[idx], dt))
        : qc[idx];
    if (temp0 < 235.16f && qc_working > 0.0f) {
        const float transferred = fmaxf(0.0f, qc_working);
        // lfus2 = lsub - lvap(k); lvap(k) = lvap0 + (2106.0 - 4218.0)*tempc.
        const float latent_vapor = 2.5e6f
            + (2106.0f - 4218.0f) * (temp0 - 273.15f);
        const float latent_fusion = 2.834e6f - latent_vapor;
        const float xnc = thompson_aa_add(
            cloud_number_entry[idx],
            thompson_aa_mul(cloud_number_tendency[idx], dt));
        if (qcten != nullptr) {
            // :3962, `qcten(k) = qcten(k) - xrc*odt`.
            qcten[idx] = thompson_aa_sub(qcten[idx],
                                         thompson_aa_mul(transferred, odt));
        } else {
            qc[idx] = 0.0f;
        }
        if (accumulate_ice) {
            // :3959-3960.
            qiten[idx] = thompson_aa_add(qiten[idx],
                                         thompson_aa_mul(transferred, odt));
            niten[idx] = thompson_aa_add(niten[idx],
                                         thompson_aa_mul(xnc, odt));
        } else {
            qi[idx] += transferred;
            ni[idx] += xnc;
        }
        cloud_number_tendency[idx] = thompson_aa_sub(
            cloud_number_tendency[idx], thompson_aa_mul(xnc, odt));
        if (accumulate_heat) {
            // :3957 lfus2 = lsub - lvap(k); :3964, tten +
            // lfus2*ocp(k)*xrc*odt*(1-IFDRY), REAL left to right.
            const float lfus2 = thompson_aa_sub(2.834e6f, heat_lvap[idx]);
            tten[idx] = thompson_aa_add(tten[idx], thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(lfus2, heat_ocp[idx]),
                                transferred), odt));
        } else {
            temperature[idx] += latent_fusion * inverse_cp * transferred;
        }
    }
    if (accumulate_heat) {
        // :3973, t1d(k) + tten(k)*DT with t1d = th*pii (mp_gt_driver).
        temperature[idx] = thompson_aa_add(
            thompson_aa_mul(theta[idx], exner[idx]),
            thompson_aa_mul(tten[idx], dt));
    }

    // :3990-3991 and :4025-4039, kept in mp=8's fused position.  Both are
    // idempotent, so WP-04's terminal state kernel may repeat them.  Droplet
    // number is deliberately absent: nc is entry state and is only ever
    // written by that terminal kernel.
    // With the accumulator, qc is still qc1d here; the terminal apply forms
    // qc1d + qcten*DT and zeroes it at or below R1 (:3975, :4007-4009).
    if (qcten == nullptr && qc[idx] <= THOMPSON_AA_R1) qc[idx] = 0.0f;
    if (accumulate_ice) return;
    if (qi[idx] <= THOMPSON_AA_R1) {
        qi[idx] = 0.0f;
        ni[idx] = 0.0f;
    } else if (qi[idx] * rho > THOMPSON_AA_R1) {
#if defined(THOMPSON_AA_WRF39)
        // fork :3733, 499.D3 (audit T8).
        thompson_aa_wrf39_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
#else
        thompson_aa_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
#endif
    } else {
        // :4025-4039 tests the MIXING RATIO and keeps both mass and number.
        // The shared bound tests the concentration (right for the source
        // stage at :3036-3055, wrong here) and zeroed the number of ice that
        // sediments into thin air aloft while keeping its mass: 12 to 72
        // levels of every saved 19,600-column real-data frame.  WRF's
        // per-kilogram form:
        const float am_i = THOMPSON_AA_AM_I;
        const float qi_local = qi[idx];
        const float ni_local = fmaxf(thompson_aa_div(THOMPSON_AA_R2, rho),
                                     ni[idx]);
        double lami = (double)thompson_aa_powf(
            thompson_aa_div(thompson_aa_mul(thompson_aa_mul(am_i, 6.0f),
                                            ni_local), qi_local),
            1.0f / 3.0f);
        const float xdi = (float)(4.0 * (1.0 / lami));
        if (xdi < 5.0e-6f) {
            lami = (double)thompson_aa_div(4.0f, 5.0e-6f);
        } else if (xdi > 300.0e-6f) {
            lami = (double)thompson_aa_div(4.0f, 300.0e-6f);
        }
#if defined(THOMPSON_AA_WRF39)
        // fork :3733, 499.D3/rho (audit T8).
        ni[idx] = (float)fmin(
            (double)thompson_aa_div(thompson_aa_mul(1.0f / 6.0f, qi_local),
                                    am_i) * thompson_aa_pow(lami, 3.0),
            (double)THOMPSON_AA_WRF39_NI_MAX / (double)rho);
#else
        ni[idx] = (float)fmin(
            (double)thompson_aa_div(thompson_aa_mul(1.0f / 6.0f, qi_local),
                                    am_i) * thompson_aa_pow(lami, 3.0),
            999.0e3 / (double)rho);
#endif
    }
}

// Level-parallel cloud fallout for columns of at most 64 levels, eight
// columns per block and one level per thread.  It keeps the column
// kernel's separate-rounding helpers, number floor, masks and serial
// height sum; the diagnostic entry points keep the column kernel.

// Device only: a block barrier has no meaning in the serial host
// build (tools/thompson_real_column_parity), which runs the column
// kernels instead (gpuwm/core/thompson.py LEVEL_PARALLEL_FALLOUT).
#ifdef __CUDACC_RTC__
extern "C" __global__ void thompson_aa_cloud_sediment_levels_64_with_masks(
    float* __restrict__ qc,
    const float* __restrict__ cloud_number_entry,
    float* __restrict__ cloud_number_tendency,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ rain_active_columns,
    const float* __restrict__ cloud_active_columns,
    const float* __restrict__ vertical_velocity,
    const float* __restrict__ dz,
    float* __restrict__ qcten,
    float dt, int nz, int ny, int nx)
{
    const int c = threadIdx.x;
    const int column = blockIdx.x * 8 + c;
    const int j = column / nx;
    const int i = column - j * nx;
    const int k = threadIdx.y;
    const bool live = column < ny * nx
        && (cloud_active_columns == nullptr || cloud_active_columns[column] != 0.0f);
    __shared__ float depth[64][8], mass[64][8], mass_flux[64][8], number_flux[64][8];
    __shared__ int top[8];
    float density, cloud_mass, cloud_number, mass_velocity, number_velocity;
    float qc_tendency, qc_initial;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const bool rain_refreshes_rhof = live && rain_active_columns != nullptr
        && rain_active_columns[column] != 0.0f;

    if (live && k < nz) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        density = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        // The column kernel's accumulator contract, level by level.
        qc_initial = qc[idx];
        qc_tendency = qcten != nullptr ? qcten[idx] : 0.0f;
        const float qc_working = qcten != nullptr
            ? thompson_aa_add(qc[idx], thompson_aa_mul(qcten[idx], dt))
            : qc[idx];
        const float held = reference_density[idx];
        cloud_mass = qc_working > THOMPSON_AA_R1
            ? qc_working * held : THOMPSON_AA_R1;
        cloud_number = thompson_aa_clamp_nc(
            thompson_aa_mul(
                thompson_aa_add(
                    cloud_number_entry[idx],
                    thompson_aa_mul(cloud_number_tendency[idx], dt)),
                held));
        mass_velocity = 0.0f;
        number_velocity = 0.0f;
        depth[k][c] = dz[idx];
        mass[k][c] = cloud_mass;
    }
    __syncthreads();
    if (k == 0) {
        int sediment_top = 0;
        float height_agl = 0.0f;
        if (live) for (int level = 0; level < nz - 1; ++level) {
            if (mass[level][c] > THOMPSON_AA_R2) sediment_top = level;
            height_agl += depth[level][c];
            if (height_agl > THOMPSON_AA_SED_HGT_AGL) break;
        }
        top[c] = sediment_top;
    }
    __syncthreads();
    const int sediment_top = top[c];
    if (live && k <= sediment_top) {
        const size_t idx = IDX3(k, j, i);
        if (cloud_mass > THOMPSON_AA_R1
                && vertical_velocity[idx] < THOMPSON_AA_SED_W_LIMIT) {
            const int nu_c = thompson_aa_nu_c(cloud_number);
            const float lambda_arg = thompson_aa_div(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(cloud_number, THOMPSON_AA_AM_R),
                        THOMPSON_AA_CCG2[nu_c]),
                    THOMPSON_AA_OCG1[nu_c]),
                cloud_mass);
            const double lambda =
                (double)thompson_aa_powf(lambda_arg, THOMPSON_AA_OBMR);
            const double inverse_lambda = 1.0 / lambda;

            const float velocity_density = rain_refreshes_rhof
                ? density : reference_density[idx];
            const float rhof = sqrtf(rho_not / velocity_density);
            const float mass_prefix = thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(rhof, THOMPSON_AA_AV_C),
                    THOMPSON_AA_CCG5[nu_c]),
                THOMPSON_AA_OCG2[nu_c]);
            const float number_prefix = thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(rhof, THOMPSON_AA_AV_C),
                    THOMPSON_AA_CCG4[nu_c]),
                THOMPSON_AA_OCG1[nu_c]);
            mass_velocity = (float)((double)mass_prefix
                * inverse_lambda * inverse_lambda);
            number_velocity = (float)((double)number_prefix
                * inverse_lambda * inverse_lambda);
        }
    }
    if (live && k < nz) {
        mass_flux[k][c] = mass_velocity * cloud_mass;
        number_flux[k][c] = number_velocity * cloud_number;
    }
    __syncthreads();
    if (live && k <= sediment_top) {
        const size_t idx = IDX3(k, j, i);
        const float odzq = 1.0f / dz[idx];
        const float orho = 1.0f / density;
        const float mass_divergence = mass_flux[k + 1][c] - mass_flux[k][c];
        const float number_divergence = number_flux[k + 1][c] - number_flux[k][c];
        qc_tendency = thompson_aa_add(
            qc_tendency,
            thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), orho));
        cloud_number_tendency[idx] = thompson_aa_add(
            cloud_number_tendency[idx],
            thompson_aa_mul(thompson_aa_mul(number_divergence, odzq), orho));
        cloud_mass = fmaxf(
            THOMPSON_AA_R1,
            thompson_aa_add(
                cloud_mass,
                thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), dt)));
        cloud_number = fmaxf(
            THOMPSON_AA_NC_SED_FLOOR,
            thompson_aa_add(
                cloud_number,
                thompson_aa_mul(thompson_aa_mul(number_divergence, odzq),
                                dt)));
    }
    if (live && k < nz) {
        const size_t idx = IDX3(k, j, i);
        if (qcten != nullptr) {
            qcten[idx] = qc_tendency;
        } else {
            qc[idx] = thompson_aa_add(qc_initial, thompson_aa_mul(qc_tendency, dt));
        }
    }
}
#endif  // __CUDACC_RTC__


// ---------------------------------------------------------------------------
// RAIN AND ICE FALLOUT IN WRF'S TENDENCY FORM (the v4.6.1 accumulator path).
// ---------------------------------------------------------------------------
//
// mp=28's own copies of the rain (:3611-3640, :3790-3812) and ice (:3664-
// 3698, :3838-3870) fallout, for the adapter's accumulator path: qr/nr/qi/ni
// arrive as WRF's read-only qr1d/nr1d/qi1d/ni1d, the working pair is formed
// exactly where WRF forms it, the fallout is ADDED to qrten/nrten/qiten/
// niten, and nothing is applied here.  The terminal apply and its size
// bounds (:4023-4053) run once, after the phase cleanup, in
// thompson_aa_terminal_rain_ice (thompson_aerosol_state.cu) -- the classic
// kernels fold them into the fallout, which put the ice size bound before
// the :3956 freeze where WRF puts it after.
//
// The classic kernels in thompson.cu are mp=8's, byte-frozen
// (tests/test_mp8_frozen.py), and keep their in-place form; that is why these
// live here.  The physics is theirs statement for statement.  The arithmetic
// is WRF's: every REAL product, quotient and sum is pinned against nvrtc's
// contraction (gfortran -O2 baseline x86-64 has no FMA), the REAL**REAL
// slopes are the correctly rounded powf glibc gives gfortran, and the
// substep factors associate as WRF writes them, odzq*DT*onstep and
// odzq*onstep*orho, where the classic kernels fold DT*onstep first.
//
// The working pair, rain (:3236-3255, :3568-3570).  reference_density is the
// rain evaporation's level-wise export: ZERO where :3236's L_qr failed (rr =
// R1, nr = R2), NEGATIVE where :3568-3570 rebuilt the pair from the :3490
// density -- MAX(R1, (qr1d + DT*qrten)*rho) and the same for nr -- and
// otherwise the :3193 TAU+1 density the :3237-3250 pair was formed on,
// including the mean-volume-diameter clamp that rebuilds nr (the tendencies
// are unchanged at a level the evaporation gate skipped, so re-forming that
// pair here gives the value WRF formed then).

__device__ __forceinline__ float thompson_aa_tau1_rain_number(float rr,
                                                             float nr_m3)
{
    // :3239-3250, the clamp the rain evaporation's own copy also applies.
    float nr_work = fmaxf(THOMPSON_AA_R2, nr_m3);
    const double lamr = (double)thompson_aa_powf(
        thompson_aa_div(thompson_aa_mul(thompson_aa_mul(
            thompson_aa_mul(THOMPSON_AA_AM_R, 6.0f), 1.0f), nr_work), rr),
        THOMPSON_AA_OBMR);
    const float mvd_num = thompson_aa_add(thompson_aa_add(3.0f, 0.0f), 0.672f);
    float mvd_r = (float)((double)mvd_num / lamr);
    const float d0r_low = thompson_aa_mul(THOMPSON_AA_D0R, 0.75f);
    const bool high = mvd_r > 2.5e-3f;
    const bool low = mvd_r < d0r_low;
    if (high || low) {
        mvd_r = high ? 2.5e-3f : d0r_low;
        const double lamr_bounded = (double)thompson_aa_div(mvd_num, mvd_r);
        nr_work = (float)(
            (double)thompson_aa_mul(thompson_aa_mul(1.0f, 1.0f / 6.0f), rr)
            * thompson_aa_pow(lamr_bounded, 3.0) / (double)THOMPSON_AA_AM_R);
    }
    return nr_work;
}

// :3236-3255 and :3568-3570, the working rain pair the rain fallout forms
// (see thompson_aa_rain_sediment_accumulate_impl).  `carried` is the rain
// evaporation's density export: zero where :3236's L_qr failed, negative
// where :3568-3570 rebuilt the pair on the :3490 density, positive (the
// :3193 density) otherwise.  Returns L_qr.
__device__ __forceinline__ bool thompson_aa_rain_fallout_pair(
    float qr1d, float nr1d, float qrten, float nrten, float carried,
    float dt, float* rr, float* nn)
{
    const float mass_working = thompson_aa_add(qr1d, thompson_aa_mul(qrten, dt));
    const float number_working = thompson_aa_add(
        nr1d, thompson_aa_mul(nrten, dt));
    if (carried == 0.0f) {
        *rr = THOMPSON_AA_R1;
        *nn = THOMPSON_AA_R2;
        return false;
    }
    if (carried < 0.0f) {
        // :3568 and :3570.
        *rr = fmaxf(THOMPSON_AA_R1, thompson_aa_mul(mass_working, -carried));
        *nn = fmaxf(THOMPSON_AA_R2, thompson_aa_mul(number_working, -carried));
        return true;
    }
    // :3237-3250.
    *rr = thompson_aa_mul(mass_working, carried);
    *nn = thompson_aa_tau1_rain_number(
        *rr, thompson_aa_mul(number_working, carried));
    return true;
}

// :3617, lamr = (am_r*crg(3)*org2*nr(k)/rr(k))**obmr, a REAL(4) powf held
// in DOUBLE PRECISION.
__device__ __forceinline__ double thompson_aa_rain_fallout_lamr(float rr,
                                                                float nn)
{
    return (double)thompson_aa_powf(
        thompson_aa_div(thompson_aa_mul(thompson_aa_mul(
            thompson_aa_mul(THOMPSON_AA_AM_R, 6.0f), 1.0f), nn), rr),
        THOMPSON_AA_OBMR);
}

// :3618-3619, vtr = rhof*av_r*crg(6)*org3 * lamr**cre(3)
// *((lamr+fv_r)**(-cre(6))), crg(6) = 24, cre(3) = 4, cre(6) = 5.
__device__ __forceinline__ float thompson_aa_rain_fallout_mass_speed(
    float rhof, double lamr)
{
    return (float)__dmul_rn(__dmul_rn((double)thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(rhof, 4854.0f), 24.0f), 1.0f / 6.0f),
        thompson_aa_pow(lamr, 4.0)),
        thompson_aa_pow(__dadd_rn(lamr, 195.0), -5.0));
}

template <int KMAX>
__device__ __forceinline__ void thompson_aa_rain_sediment_accumulate_impl(
    const float* __restrict__ qr1d,
    const float* __restrict__ nr1d,
    float* __restrict__ qrten,
    float* __restrict__ nrten,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    int accumulate_surface, float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float density[KMAX];
    float rain_mass[KMAX];
    float rain_number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX];
    float number_flux[KMAX];
    float mass_tendency[KMAX];
    float number_tendency[KMAX];

    const float rho_not = 101325.0f / (287.05f * 298.0f);
    bool any_rain = false;
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above_mass = 0.0f;
    float velocity_above_number = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density[k] = rho;
        mass_tendency[k] = qrten[idx];
        number_tendency[k] = nrten[idx];
        float rr, nn;
        if (thompson_aa_rain_fallout_pair(qr1d[idx], nr1d[idx], qrten[idx],
                nrten[idx], reference_density[idx], dt, &rr, &nn)) {
            any_rain = true;
        }
        rain_mass[k] = rr;
        rain_number[k] = nn;
        // :3612-3632.
        const float rhof = sqrtf(rho_not / rho);
        if (rr > THOMPSON_AA_R1) {
            const double lamr = thompson_aa_rain_fallout_lamr(rr, nn);
            // rhof*av_r*crg(6)*org3, crg(6) = 24, then the DOUBLE powers.
            mass_velocity[k] = thompson_aa_rain_fallout_mass_speed(rhof, lamr);
            // rhof*av_r*crg(7)/crg(12).
            number_velocity[k] = (float)((double)thompson_aa_div(
                thompson_aa_mul(thompson_aa_mul(rhof, 4854.0f), 3.3233511f),
                1.3293403f)
                * thompson_aa_pow(lamr, 2.5) * thompson_aa_pow(lamr + 195.0, -3.5));
        } else {
            mass_velocity[k] = velocity_above_mass;
            number_velocity[k] = velocity_above_number;
        }
        velocity_above_mass = mass_velocity[k];
        velocity_above_number = number_velocity[k];
        // :3634-3638.
        const float vmax = fmaxf(mass_velocity[k], number_velocity[k]);
        if (vmax > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = thompson_aa_div(dz[idx], vmax);
            nstep = max(nstep, (int)thompson_aa_add(
                thompson_aa_div(dt, delta_tp), 1.0f));
        }
    }

    float exported = 0.0f;
    if (any_rain) {
        // :3640-3641 and :3790.
        if (sediment_top == nz - 1) sediment_top = nz - 2;
        const float onstep = nstep > 0
            ? thompson_aa_div(1.0f, (float)nstep) : 1.0f;
        const int steps = (int)rintf(thompson_aa_div(1.0f, onstep));
        for (int step = 0; step < steps; ++step) {
            for (int k = nz - 1; k >= 0; --k) {
                mass_flux[k] = thompson_aa_mul(mass_velocity[k], rain_mass[k]);
                number_flux[k] = thompson_aa_mul(number_velocity[k],
                                                 rain_number[k]);
            }
            int k = nz - 1;
            size_t idx = IDX3(k, j, i);
            float odzq = thompson_aa_div(1.0f, dz[idx]);
            float orho = thompson_aa_div(1.0f, density[k]);
            mass_tendency[k] = thompson_aa_sub(mass_tendency[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    mass_flux[k], odzq), onstep), orho));
            number_tendency[k] = thompson_aa_sub(number_tendency[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    number_flux[k], odzq), onstep), orho));
            rain_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_sub(rain_mass[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    mass_flux[k], odzq), dt), onstep)));
            rain_number[k] = fmaxf(THOMPSON_AA_R2, thompson_aa_sub(
                rain_number[k], thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(number_flux[k], odzq), dt), onstep)));
            for (k = sediment_top; k >= 0; --k) {
                idx = IDX3(k, j, i);
                odzq = thompson_aa_div(1.0f, dz[idx]);
                orho = thompson_aa_div(1.0f, density[k]);
                const float mass_divergence =
                    thompson_aa_sub(mass_flux[k + 1], mass_flux[k]);
                const float number_divergence =
                    thompson_aa_sub(number_flux[k + 1], number_flux[k]);
                mass_tendency[k] = thompson_aa_add(mass_tendency[k],
                    thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                        mass_divergence, odzq), onstep), orho));
                number_tendency[k] = thompson_aa_add(number_tendency[k],
                    thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                        number_divergence, odzq), onstep), orho));
                rain_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_add(
                    rain_mass[k], thompson_aa_mul(thompson_aa_mul(
                        thompson_aa_mul(mass_divergence, odzq), dt),
                        onstep)));
                rain_number[k] = fmaxf(THOMPSON_AA_R2, thompson_aa_add(
                    rain_number[k], thompson_aa_mul(thompson_aa_mul(
                        thompson_aa_mul(number_divergence, odzq), dt),
                        onstep)));
            }
            // :3811-3812.
            if (rain_mass[0] > thompson_aa_mul(THOMPSON_AA_R1, 1000.0f)) {
                exported = thompson_aa_add(exported, thompson_aa_mul(
                    thompson_aa_mul(mass_flux[0], dt), onstep));
            }
        }
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            qrten[idx] = mass_tendency[k];
            nrten[idx] = number_tendency[k];
        }
    }
    // accumulate_surface 2: pptrain alone into rainncv, for
    // thompson_aa_surface_precipitation to add in mp_gt_driver's order.
    // 0 and 1: the classic launcher's bookkeeping.
    if (accumulate_surface == 2) {
        rainncv[column] = exported;
        return;
    }
    if (accumulate_surface) rainncv[column] += exported;
    else rainncv[column] = exported;
    rainnc[column] += exported;
}

#define THOMPSON_AA_RAIN_ACCUMULATE_PARAMETERS                            \
    const float* __restrict__ qr1d, const float* __restrict__ nr1d,     \
    float* __restrict__ qrten, float* __restrict__ nrten,               \
    const float* __restrict__ temperature,                              \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, int accumulate_surface, float dt,     \
    int nz, int ny, int nx

#define THOMPSON_AA_RAIN_ACCUMULATE_ARGUMENTS                             \
    qr1d, nr1d, qrten, nrten, temperature, pressure, qv,                 \
    reference_density, dz, rainnc, rainncv, accumulate_surface, dt,      \
    nz, ny, nx

extern "C" __global__ void thompson_aa_rain_sediment_accumulate_64(
    THOMPSON_AA_RAIN_ACCUMULATE_PARAMETERS)
{
    thompson_aa_rain_sediment_accumulate_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_RAIN_ACCUMULATE_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_rain_sediment_accumulate_256(
    THOMPSON_AA_RAIN_ACCUMULATE_PARAMETERS)
{
    thompson_aa_rain_sediment_accumulate_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_RAIN_ACCUMULATE_ARGUMENTS);
}


// Ice (:3664-3698, :3838-3870).  The working pair is :3226-3233's, on the
// held :3193 density: ri = (qi1d + qiten*DT)*rho where L_qi, else R1 / R2;
// nothing between the TAU+1 refresh and the ice fallout moves qiten or
// niten.  rhof is :3614's refresh from the current density in a column
// with any L_qr (rain_active_columns), and :3194's on the held density
// otherwise -- the cloud fallout's model.
template <int KMAX>
__device__ __forceinline__ void thompson_aa_ice_sediment_accumulate_impl(
    const float* __restrict__ qi1d,
    const float* __restrict__ ni1d,
    float* __restrict__ qiten,
    float* __restrict__ niten,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ rain_active_columns,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ snownc,
    float* __restrict__ snowncv,
    int export_only, float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float density[KMAX];
    float ice_mass[KMAX];
    float ice_number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX];
    float number_flux[KMAX];
    float mass_tendency[KMAX];
    float number_tendency[KMAX];

    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const bool rain_refreshes_rhof = rain_active_columns != nullptr
        && rain_active_columns[column] != 0.0f;
    bool any_ice = false;
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above_mass = 0.0f;
    float velocity_above_number = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density[k] = rho;
        mass_tendency[k] = qiten[idx];
        number_tendency[k] = niten[idx];
        const float held = reference_density[idx];
        const float mass_working = thompson_aa_add(
            qi1d[idx], thompson_aa_mul(qiten[idx], dt));
        float ri, nn;
        if (mass_working > THOMPSON_AA_R1) {
            ri = thompson_aa_mul(mass_working, held);
            nn = fmaxf(THOMPSON_AA_R2, thompson_aa_mul(thompson_aa_add(
                ni1d[idx], thompson_aa_mul(niten[idx], dt)), held));
            any_ice = true;
        } else {
            ri = THOMPSON_AA_R1;
            nn = THOMPSON_AA_R2;
        }
        ice_mass[k] = ri;
        ice_number[k] = nn;
        const float rhof = sqrtf(rho_not / (rain_refreshes_rhof ? rho : held));
        if (ri > THOMPSON_AA_R1) {
            // :3679-3686.  bv_i = 1, so ilami**bv_i is ilami.
            const double lami = (double)thompson_aa_powf(
                thompson_aa_div(thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(THOMPSON_AA_AM_I, THOMPSON_AA_CIG2),
                    THOMPSON_AA_OIG1), nn), ri),
                THOMPSON_AA_OBMI);
            const double ilami = 1.0 / lami;
            mass_velocity[k] = (float)((double)thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(rhof, 1493.9f), 24.0f),
                1.0f / 6.0f) * ilami);
            number_velocity[k] = (float)((double)thompson_aa_div(
                thompson_aa_mul(thompson_aa_mul(rhof, 1493.9f), 3.3233511f),
                1.3293403f) * ilami);
        } else {
            mass_velocity[k] = velocity_above_mass;
            number_velocity[k] = velocity_above_number;
        }
        velocity_above_mass = mass_velocity[k];
        velocity_above_number = number_velocity[k];
        // :3692-3696.
        if (mass_velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = thompson_aa_div(dz[idx], mass_velocity[k]);
            nstep = max(nstep, (int)thompson_aa_add(
                thompson_aa_div(dt, delta_tp), 1.0f));
        }
    }

    float exported = 0.0f;
    if (any_ice) {
        if (sediment_top == nz - 1) sediment_top = nz - 2;
        const float onstep = nstep > 0
            ? thompson_aa_div(1.0f, (float)nstep) : 1.0f;
        const int steps = (int)rintf(thompson_aa_div(1.0f, onstep));
        for (int step = 0; step < steps; ++step) {
            for (int k = nz - 1; k >= 0; --k) {
                mass_flux[k] = thompson_aa_mul(mass_velocity[k], ice_mass[k]);
                number_flux[k] = thompson_aa_mul(number_velocity[k],
                                                 ice_number[k]);
            }
            int k = nz - 1;
            size_t idx = IDX3(k, j, i);
            float odzq = thompson_aa_div(1.0f, dz[idx]);
            float orho = thompson_aa_div(1.0f, density[k]);
            mass_tendency[k] = thompson_aa_sub(mass_tendency[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    mass_flux[k], odzq), onstep), orho));
            number_tendency[k] = thompson_aa_sub(number_tendency[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    number_flux[k], odzq), onstep), orho));
            ice_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_sub(ice_mass[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    mass_flux[k], odzq), dt), onstep)));
            ice_number[k] = fmaxf(THOMPSON_AA_R2, thompson_aa_sub(
                ice_number[k], thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(number_flux[k], odzq), dt), onstep)));
            for (k = sediment_top; k >= 0; --k) {
                idx = IDX3(k, j, i);
                odzq = thompson_aa_div(1.0f, dz[idx]);
                orho = thompson_aa_div(1.0f, density[k]);
                const float mass_divergence =
                    thompson_aa_sub(mass_flux[k + 1], mass_flux[k]);
                const float number_divergence =
                    thompson_aa_sub(number_flux[k + 1], number_flux[k]);
                mass_tendency[k] = thompson_aa_add(mass_tendency[k],
                    thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                        mass_divergence, odzq), onstep), orho));
                number_tendency[k] = thompson_aa_add(number_tendency[k],
                    thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                        number_divergence, odzq), onstep), orho));
                ice_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_add(
                    ice_mass[k], thompson_aa_mul(thompson_aa_mul(
                        thompson_aa_mul(mass_divergence, odzq), dt),
                        onstep)));
                ice_number[k] = fmaxf(THOMPSON_AA_R2, thompson_aa_add(
                    ice_number[k], thompson_aa_mul(thompson_aa_mul(
                        thompson_aa_mul(number_divergence, odzq), dt),
                        onstep)));
            }
            // :3868-3869.
            if (ice_mass[0] > thompson_aa_mul(THOMPSON_AA_R1, 1000.0f)) {
                exported = thompson_aa_add(exported, thompson_aa_mul(
                    thompson_aa_mul(mass_flux[0], dt), onstep));
            }
        }
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            qiten[idx] = mass_tendency[k];
            niten[idx] = number_tendency[k];
        }
    }
    // export_only: pptice alone into snowncv, for
    // thompson_aa_surface_precipitation to add in mp_gt_driver's order;
    // otherwise the classic launcher's bookkeeping.
    if (export_only) {
        snowncv[column] = exported;
        return;
    }
    rainncv[column] = exported;
    snowncv[column] = exported;
    rainnc[column] += exported;
    snownc[column] += exported;
}

#define THOMPSON_AA_ICE_ACCUMULATE_PARAMETERS                             \
    const float* __restrict__ qi1d, const float* __restrict__ ni1d,     \
    float* __restrict__ qiten, float* __restrict__ niten,               \
    const float* __restrict__ temperature,                              \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ rain_active_columns,                      \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ snownc,            \
    float* __restrict__ snowncv, int export_only, float dt, int nz,     \
    int ny, int nx

#define THOMPSON_AA_ICE_ACCUMULATE_ARGUMENTS                              \
    qi1d, ni1d, qiten, niten, temperature, pressure, qv,                 \
    reference_density, rain_active_columns, dz, rainnc, rainncv, snownc, \
    snowncv, export_only, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_ice_sediment_accumulate_64(
    THOMPSON_AA_ICE_ACCUMULATE_PARAMETERS)
{
    thompson_aa_ice_sediment_accumulate_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_ICE_ACCUMULATE_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_ice_sediment_accumulate_256(
    THOMPSON_AA_ICE_ACCUMULATE_PARAMETERS)
{
    thompson_aa_ice_sediment_accumulate_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_ICE_ACCUMULATE_ARGUMENTS);
}


// ---------------------------------------------------------------------------
// SNOW AND GRAUPEL FALLOUT AND THE SURFACE TOTALS, WRF v4.6.1 (:3699-3780,
// :3871-3937, mp_gt_driver :1294-1308).
// ---------------------------------------------------------------------------
//
// mp=28's own passes.  They used to be thompson.cu's classic snow and
// graupel fallout, which is byte-frozen for mp=8 and differs from WRF's
// arithmetic in ways the 0 ULP column oracle measured
// (tools/thompson_aerosol_column_oracle): CUDA's powf/pow/log10f instead of
// WOOF's own words (the device-only share), the substep factor folded as
// odzq*(DT*onstep) where WRF writes (odzq*DT)*onstep, the graupel slope taken
// straight from the intercept with a size clamp WRF does not apply at
// :3289-3297 and :3373, rhof formed on the current density in columns where
// WRF's :3614 refresh never ran, and the four surface totals summed in the
// kernels' order instead of mp_gt_driver's.  Statement for statement:
//
//   rhof  :3614 SQRT(RHO_NOT/rho(k)) on the current density in a column
//         with any L_qr (the rain fall-speed pass refreshes every level),
//         otherwise :3194's on the :3193 TAU+1 density.  The rain
//         evaporation's density export carries L_qr (zero where it failed).
//   vtrk  the rain fall-speed pass's mass speed, :3612-3632, on WRF's rain
//         pair (thompson_aa_rain_fallout_pair): a level whose rr(k) is at or
//         below R1 inherits the speed from above, and a column with no L_qr
//         has none.
//   snow  :3257-3262 rs(k) on the :3193 density; :3313-3353 smob, smoc on
//         the :3188 temperature; :3699-3727 the fall speed; :3871-3902 the
//         apply with pptsnow above R1*1000.
//   graupel :3283-3303 rg(k) and the diagnosed ng(k) (not the call's
//         private ng1d); :3370-3376 lamg on that ng(k); :3740-3780 the mass
//         and number speeds; :3903-3937 the apply with pptgraul above
//         R1*1000.  ngten from the number channel is the private ng1d's
//         fallout tendency (graupel_number_shadow).
//
// The working mixing ratio a pass starts from is the in-place post-source
// value the networks leave in the state (qs1d + qsten*DT as the sources
// formed it); the pass adds its own fallout tendency times DT to it.  Every
// REAL product, quotient and sum is pinned against nvrtc's contraction, the
// DOUBLE PRECISION ones too, so strict and default arithmetic compile the
// same operations.

// :3597-3640's ANY(L_qr): the evaporation's density export is nonzero
// exactly where L_qr holds.
__device__ __forceinline__ bool thompson_aa_column_has_rain(
    const float* __restrict__ rain_density, int j, int i, int nz, int ny,
    int nx)
{
    for (int k = 0; k < nz; ++k) {
        if (rain_density[IDX3(k, j, i)] != 0.0f) return true;
    }
    return false;
}

// :3616-3632 for one level, top down: vtrk(k) on WRF's rain pair, or the
// speed from above.  `rain_rr` receives rr(k) for the melting-snow blend.
__device__ __forceinline__ float thompson_aa_rain_pass_speed(
    const float* __restrict__ qr1d, const float* __restrict__ nr1d,
    const float* __restrict__ qrten, const float* __restrict__ nrten,
    const float* __restrict__ rain_density, size_t idx, float rho, float dt,
    float speed_above, float* rain_rr)
{
    float rr, nn;
    thompson_aa_rain_fallout_pair(qr1d[idx], nr1d[idx], qrten[idx],
                                  nrten[idx], rain_density[idx], dt, &rr, &nn);
    *rain_rr = rr;
    if (!(rr > THOMPSON_AA_R1)) return speed_above;
    const float rhof = __fsqrt_rn(thompson_aa_div(THOMPSON_AA_RHO_NOT, rho));
    return thompson_aa_rain_fallout_mass_speed(
        rhof, thompson_aa_rain_fallout_lamr(rr, nn));
}

// One substep factor's worth of WRF's apply, :3879-3899 / :3910-3934:
// nstep = NINT(1./onstep) after onstep = 1./REAL(nstep) where nstep > 0.
__device__ __forceinline__ float thompson_aa_onstep(int nstep)
{
    return nstep > 0 ? thompson_aa_div(1.0f, (float)nstep) : 1.0f;
}

template <int KMAX>
__device__ __forceinline__ void thompson_aa_snow_sediment_impl(
    float* __restrict__ qs,
    const float* __restrict__ snow_melt_marker,
    const float* __restrict__ qr1d,
    const float* __restrict__ nr1d,
    const float* __restrict__ qrten,
    const float* __restrict__ nrten,
    const float* __restrict__ rain_density,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ reference_temperature,
    const float* __restrict__ velocity_boost,
    const float* __restrict__ dz,
    float* __restrict__ snow_precipitation,
    float* __restrict__ qsten,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float density[KMAX];
    float snow_mass[KMAX];
    float velocity[KMAX];
    float flux[KMAX];
    float tendency[KMAX];

    const bool any_rain = thompson_aa_column_has_rain(
        rain_density, j, i, nz, ny, nx);
    bool any_snow = false;
    int sediment_top = 0;
    int nstep = 0;
    float rain_speed = 0.0f;
    float speed_above = 0.0f;   // vtsk(kte+1) = 0, :3598-3606

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float rho = thompson_aa_air_density(
            pressure[idx], temperature[idx], fmaxf(1.0e-10f, qv[idx]));
        density[k] = rho;
        // qsten(k) as the sources left it (zero when the networks applied
        // their tendency in place).
        tendency[k] = qsten != nullptr ? qsten[idx] : 0.0f;
        const float held = reference_density[idx];

        // vtrk(k) and rr(k), :3612-3632; zero in a column without L_qr.
        float rain_rr = THOMPSON_AA_R1;
        if (any_rain) {
            rain_speed = thompson_aa_rain_pass_speed(
                qr1d, nr1d, qrten, nrten, rain_density, idx, rho, dt,
                rain_speed, &rain_rr);
        }

        // :3257-3262, on qs1d + qsten*DT.
        const float working = qsten != nullptr
            ? thompson_aa_add(qs[idx], thompson_aa_mul(qsten[idx], dt))
            : qs[idx];
        float rs = THOMPSON_AA_R1;
        if (working > THOMPSON_AA_R1) {
            rs = thompson_aa_mul(working, held);
            any_snow = true;
        }
        snow_mass[k] = rs;

        // :3699-3727.
        if (rs > THOMPSON_AA_R1) {
            const float tc0 = fminf(-0.1f, thompson_aa_sub(
                reference_temperature[idx], 273.15f));
            const float smob = thompson_aa_mul(rs, THOMPSON_AA_OAMS);
            const float smoc = thompson_aa_mul(
                thompson_field_a(tc0, THOMPSON_AA_CSE1),
                thompson_aa_powf(smob, thompson_field_b(tc0, THOMPSON_AA_CSE1)));
            const float xds = thompson_aa_div(smoc, smob);
            const float mrat = thompson_aa_div(1.0f, xds);
            float ils1 = thompson_aa_div(1.0f, thompson_aa_add(
                thompson_aa_mul(mrat, THOMPSON_AA_LAM0), THOMPSON_AA_FV_S));
            float ils2 = thompson_aa_div(1.0f, thompson_aa_add(
                thompson_aa_mul(mrat, THOMPSON_AA_LAM1), THOMPSON_AA_FV_S));
            const float mrat_mu = thompson_aa_powf(mrat, THOMPSON_AA_MU_S);
            const float t1 = thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_KAP0, THOMPSON_AA_CSG4),
                thompson_aa_powf(ils1, THOMPSON_AA_CSE4));
            const float t2 = thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_KAP1, mrat_mu), THOMPSON_AA_CSG10),
                thompson_aa_powf(ils2, THOMPSON_AA_CSE10));
            ils1 = thompson_aa_div(1.0f, thompson_aa_mul(mrat, THOMPSON_AA_LAM0));
            ils2 = thompson_aa_div(1.0f, thompson_aa_mul(mrat, THOMPSON_AA_LAM1));
            const float t3 = thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_KAP0, THOMPSON_AA_CSG1),
                thompson_aa_powf(ils1, THOMPSON_AA_CSE1));
            const float t4 = thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_KAP1, mrat_mu), THOMPSON_AA_CSG7),
                thompson_aa_powf(ils2, THOMPSON_AA_CSE7));
            const float rhof = __fsqrt_rn(thompson_aa_div(
                THOMPSON_AA_RHO_NOT, any_rain ? rho : held));
            const float vts = thompson_aa_div(thompson_aa_mul(
                thompson_aa_mul(rhof, THOMPSON_AA_AV_S),
                thompson_aa_add(t1, t2)), thompson_aa_add(t3, t4));
            if (snow_melt_marker[idx] != 0.0f) {
                // :3721-3723, vtsk = vts*SR + (1.-SR)*vtrk(k).
                const float sr = thompson_aa_div(rs, thompson_aa_add(rs, rain_rr));
                velocity[k] = thompson_aa_add(thompson_aa_mul(vts, sr),
                    thompson_aa_mul(thompson_aa_sub(1.0f, sr), rain_speed));
            } else {
                velocity[k] = thompson_aa_mul(vts, velocity_boost[idx]);
            }
        } else {
            velocity[k] = speed_above;
        }
        speed_above = velocity[k];

        // :3729-3733.
        if (velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = thompson_aa_div(dz[idx], velocity[k]);
            nstep = max(nstep, (int)thompson_aa_add(
                thompson_aa_div(dt, delta_tp), 1.0f));
        }
    }

    float precipitation = 0.0f;
    if (any_snow) {
        if (sediment_top == nz - 1) sediment_top = nz - 2;
        const float onstep = thompson_aa_onstep(nstep);
        const int steps = thompson_aa_nint(thompson_aa_div(1.0f, onstep));
        for (int step = 0; step < steps; ++step) {
            for (int k = nz - 1; k >= 0; --k) {
                flux[k] = thompson_aa_mul(velocity[k], snow_mass[k]);
            }
            int k = nz - 1;
            float odzq = thompson_aa_div(1.0f, dz[IDX3(k, j, i)]);
            float orho = thompson_aa_div(1.0f, density[k]);
            tendency[k] = thompson_aa_sub(tendency[k], thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(flux[k], odzq), onstep), orho));
            snow_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_sub(snow_mass[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    flux[k], odzq), dt), onstep)));
            for (k = sediment_top; k >= 0; --k) {
                odzq = thompson_aa_div(1.0f, dz[IDX3(k, j, i)]);
                orho = thompson_aa_div(1.0f, density[k]);
                const float divergence = thompson_aa_sub(flux[k + 1], flux[k]);
                tendency[k] = thompson_aa_add(tendency[k], thompson_aa_mul(
                    thompson_aa_mul(thompson_aa_mul(divergence, odzq), onstep),
                    orho));
                snow_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_add(
                    snow_mass[k], thompson_aa_mul(thompson_aa_mul(
                        thompson_aa_mul(divergence, odzq), dt), onstep)));
            }
            // :3900-3901.
            if (snow_mass[0] > thompson_aa_mul(THOMPSON_AA_R1, 1000.0f)) {
                precipitation = thompson_aa_add(precipitation, thompson_aa_mul(
                    thompson_aa_mul(flux[0], dt), onstep));
            }
        }
    }
    // :4054-4055, qs1d = qs1d + qsten*DT, zero at or below R1.  With the
    // accumulator qs is still qs1d and tendency[k] the whole qsten(k);
    // without it qs holds the sources' in-place result and tendency[k]
    // the fallout's part alone.
    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float updated = thompson_aa_add(
            qs[idx], thompson_aa_mul(tendency[k], dt));
        qs[idx] = updated <= THOMPSON_AA_R1 ? 0.0f : updated;
        if (qsten != nullptr) qsten[idx] = tendency[k];
    }
    snow_precipitation[column] = precipitation;
}

#ifdef __CUDACC_RTC__
// Level-parallel exact fallout. One block owns one column.
template <int KMAX>
__device__ __forceinline__ void thompson_aa_snow_sediment_levels_impl(
    float* __restrict__ qs,
    const float* __restrict__ snow_melt_marker,
    const float* __restrict__ qr1d,
    const float* __restrict__ nr1d,
    const float* __restrict__ qrten,
    const float* __restrict__ nrten,
    const float* __restrict__ rain_density,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ reference_temperature,
    const float* __restrict__ velocity_boost,
    const float* __restrict__ dz,
    float* __restrict__ snow_precipitation,
    float* __restrict__ qsten,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x;
    if (column >= ny * nx) return;
    const int lane = threadIdx.x;
    const int j = column / nx;
    const int i = column - j * nx;

    __shared__ float density[KMAX];
    __shared__ float snow_mass[KMAX];
    __shared__ float velocity[KMAX];
    __shared__ float flux[KMAX];
    __shared__ float tendency[KMAX];

    __shared__ int sediment_top;
    __shared__ int nstep;
    __shared__ bool any_rain;
    if (lane == 0) {
        any_rain = thompson_aa_column_has_rain(rain_density, j, i, nz, ny, nx);
        sediment_top = 0;
        nstep = 0;
    }
    __syncthreads();
    __shared__ bool any_snow;
    if (lane == 0) {
        any_snow = false;
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float working = qsten != nullptr
                ? thompson_aa_add(qs[idx], thompson_aa_mul(qsten[idx], dt)) : qs[idx];
            if (working > THOMPSON_AA_R1) any_snow = true;
        }
    }
    __syncthreads();
    if (!any_snow) {
        for (int k = lane; k < nz; k += blockDim.x) {
            const size_t idx = IDX3(k, j, i);
            const float updated = thompson_aa_add(qs[idx], thompson_aa_mul(
                qsten != nullptr ? qsten[idx] : 0.0f, dt));
            qs[idx] = updated <= THOMPSON_AA_R1 ? 0.0f : updated;
        }
        if (lane == 0) snow_precipitation[column] = 0.0f;
        return;
    }
    __shared__ float rain_speed_at[KMAX];
    __shared__ float rain_rr_at[KMAX];
    if (lane == 0) {
        bool blend = false;
        for (int k = 0; k < nz; ++k)
            blend |= snow_melt_marker[IDX3(k, j, i)] != 0.0f;
        float rain_speed = 0.0f;
        for (int k = nz - 1; k >= 0; --k) {
            const size_t idx = IDX3(k, j, i);
            float rain_rr = THOMPSON_AA_R1;
            if (any_rain && blend) {
                const float rho = thompson_aa_air_density(pressure[idx],
                    temperature[idx], fmaxf(1.0e-10f, qv[idx]));
                rain_speed = thompson_aa_rain_pass_speed(qr1d, nr1d, qrten,
                    nrten, rain_density, idx, rho, dt, rain_speed, &rain_rr);
            }
            rain_speed_at[k] = rain_speed;
            rain_rr_at[k] = rain_rr;
        }
    }
    __syncthreads();


    for (int k = lane; k < nz; k += blockDim.x) {
        const size_t idx = IDX3(k, j, i);
        const float rho = thompson_aa_air_density(
            pressure[idx], temperature[idx], fmaxf(1.0e-10f, qv[idx]));
        density[k] = rho;
        // qsten(k) as the sources left it (zero when the networks applied
        // their tendency in place).
        tendency[k] = qsten != nullptr ? qsten[idx] : 0.0f;
        const float held = reference_density[idx];

        const float rain_rr = rain_rr_at[k];
        const float rain_speed = rain_speed_at[k];

        // :3257-3262, on qs1d + qsten*DT.
        const float working = qsten != nullptr
            ? thompson_aa_add(qs[idx], thompson_aa_mul(qsten[idx], dt))
            : qs[idx];
        float rs = THOMPSON_AA_R1;
        if (working > THOMPSON_AA_R1) {
            rs = thompson_aa_mul(working, held);

        }
        snow_mass[k] = rs;

        // :3699-3727.
        if (rs > THOMPSON_AA_R1) {
            const float tc0 = fminf(-0.1f, thompson_aa_sub(
                reference_temperature[idx], 273.15f));
            const float smob = thompson_aa_mul(rs, THOMPSON_AA_OAMS);
            const float smoc = thompson_aa_mul(
                thompson_field_a(tc0, THOMPSON_AA_CSE1),
                thompson_aa_powf(smob, thompson_field_b(tc0, THOMPSON_AA_CSE1)));
            const float xds = thompson_aa_div(smoc, smob);
            const float mrat = thompson_aa_div(1.0f, xds);
            float ils1 = thompson_aa_div(1.0f, thompson_aa_add(
                thompson_aa_mul(mrat, THOMPSON_AA_LAM0), THOMPSON_AA_FV_S));
            float ils2 = thompson_aa_div(1.0f, thompson_aa_add(
                thompson_aa_mul(mrat, THOMPSON_AA_LAM1), THOMPSON_AA_FV_S));
            const float mrat_mu = thompson_aa_powf(mrat, THOMPSON_AA_MU_S);
            const float t1 = thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_KAP0, THOMPSON_AA_CSG4),
                thompson_aa_powf(ils1, THOMPSON_AA_CSE4));
            const float t2 = thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_KAP1, mrat_mu), THOMPSON_AA_CSG10),
                thompson_aa_powf(ils2, THOMPSON_AA_CSE10));
            ils1 = thompson_aa_div(1.0f, thompson_aa_mul(mrat, THOMPSON_AA_LAM0));
            ils2 = thompson_aa_div(1.0f, thompson_aa_mul(mrat, THOMPSON_AA_LAM1));
            const float t3 = thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_KAP0, THOMPSON_AA_CSG1),
                thompson_aa_powf(ils1, THOMPSON_AA_CSE1));
            const float t4 = thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_KAP1, mrat_mu), THOMPSON_AA_CSG7),
                thompson_aa_powf(ils2, THOMPSON_AA_CSE7));
            const float rhof = __fsqrt_rn(thompson_aa_div(
                THOMPSON_AA_RHO_NOT, any_rain ? rho : held));
            const float vts = thompson_aa_div(thompson_aa_mul(
                thompson_aa_mul(rhof, THOMPSON_AA_AV_S),
                thompson_aa_add(t1, t2)), thompson_aa_add(t3, t4));
            if (snow_melt_marker[idx] != 0.0f) {
                // :3721-3723, vtsk = vts*SR + (1.-SR)*vtrk(k).
                const float sr = thompson_aa_div(rs, thompson_aa_add(rs, rain_rr));
                velocity[k] = thompson_aa_add(thompson_aa_mul(vts, sr),
                    thompson_aa_mul(thompson_aa_sub(1.0f, sr), rain_speed));
            } else {
                velocity[k] = thompson_aa_mul(vts, velocity_boost[idx]);
            }
        } else {
            velocity[k] = -1.0f;
        }
    }
    __syncthreads();
    if (lane == 0) {
        float above = 0.0f;
        for (int k = nz - 1; k >= 0; --k) {
            if (!(snow_mass[k] > THOMPSON_AA_R1)) velocity[k] = above;
            above = velocity[k];
            if (velocity[k] > 1.0e-3f) {
                sediment_top = max(sediment_top, k);
                const float delta_tp = thompson_aa_div(dz[IDX3(k,j,i)], velocity[k]);
                nstep = max(nstep, (int)thompson_aa_add(thompson_aa_div(dt, delta_tp), 1.0f));
            }
        }
    }
    __syncthreads();
    float precipitation = 0.0f;
    if (any_snow) {
        if (lane == 0 && sediment_top == nz - 1) sediment_top = nz - 2;
        __syncthreads();
        const float onstep = thompson_aa_onstep(nstep);
        const int steps = thompson_aa_nint(thompson_aa_div(1.0f, onstep));
        for (int step = 0; step < steps; ++step) {
            for (int k = lane; k < nz; k += blockDim.x) {
                flux[k] = thompson_aa_mul(velocity[k], snow_mass[k]);
            }
            __syncthreads();
            int k = nz - 1;
            if (lane == nz - 1) {
            float odzq = thompson_aa_div(1.0f, dz[IDX3(k, j, i)]);
            float orho = thompson_aa_div(1.0f, density[k]);
            tendency[k] = thompson_aa_sub(tendency[k], thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(flux[k], odzq), onstep), orho));
            snow_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_sub(snow_mass[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    flux[k], odzq), dt), onstep)));
            }
            for (k = lane; k <= sediment_top; k += blockDim.x) {
                const float odzq = thompson_aa_div(1.0f, dz[IDX3(k, j, i)]);
                const float orho = thompson_aa_div(1.0f, density[k]);
                const float divergence = thompson_aa_sub(flux[k + 1], flux[k]);
                tendency[k] = thompson_aa_add(tendency[k], thompson_aa_mul(
                    thompson_aa_mul(thompson_aa_mul(divergence, odzq), onstep),
                    orho));
                snow_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_add(
                    snow_mass[k], thompson_aa_mul(thompson_aa_mul(
                        thompson_aa_mul(divergence, odzq), dt), onstep)));
            }
            __syncthreads();
            // :3900-3901.
            if (lane == 0 && snow_mass[0] > thompson_aa_mul(THOMPSON_AA_R1, 1000.0f)) {
                precipitation = thompson_aa_add(precipitation, thompson_aa_mul(
                    thompson_aa_mul(flux[0], dt), onstep));
            }
        }
    }
    // :4054-4055, qs1d = qs1d + qsten*DT, zero at or below R1.  With the
    // accumulator qs is still qs1d and tendency[k] the whole qsten(k);
    // without it qs holds the sources' in-place result and tendency[k]
    // the fallout's part alone.
    for (int k = lane; k < nz; k += blockDim.x) {
        const size_t idx = IDX3(k, j, i);
        const float updated = thompson_aa_add(
            qs[idx], thompson_aa_mul(tendency[k], dt));
        qs[idx] = updated <= THOMPSON_AA_R1 ? 0.0f : updated;
        if (qsten != nullptr) qsten[idx] = tendency[k];
    }
    if (lane == 0) snow_precipitation[column] = precipitation;
}

#endif  // __CUDACC_RTC__

#define THOMPSON_AA_SNOW_SEDIMENT_PARAMETERS                              \
    float* __restrict__ qs, const float* __restrict__ snow_melt_marker,  \
    const float* __restrict__ qr1d, const float* __restrict__ nr1d,      \
    const float* __restrict__ qrten, const float* __restrict__ nrten,    \
    const float* __restrict__ rain_density,                              \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ reference_temperature,                     \
    const float* __restrict__ velocity_boost,                            \
    const float* __restrict__ dz,                                        \
    float* __restrict__ snow_precipitation,                              \
    float* __restrict__ qsten, float dt, int nz, int ny, int nx

#define THOMPSON_AA_SNOW_SEDIMENT_ARGUMENTS                               \
    qs, snow_melt_marker, qr1d, nr1d, qrten, nrten, rain_density,        \
    temperature, pressure, qv, reference_density, reference_temperature, \
    velocity_boost, dz, snow_precipitation, qsten, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_snow_sediment_64(
    THOMPSON_AA_SNOW_SEDIMENT_PARAMETERS)
{
    thompson_aa_snow_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_SNOW_SEDIMENT_ARGUMENTS);
}

#ifdef __CUDACC_RTC__
extern "C" __global__ void thompson_aa_snow_sediment_levels_64(
    THOMPSON_AA_SNOW_SEDIMENT_PARAMETERS)
{
    thompson_aa_snow_sediment_levels_impl<64>(
        THOMPSON_AA_SNOW_SEDIMENT_ARGUMENTS);
}
#endif  // __CUDACC_RTC__
extern "C" __global__ void thompson_aa_snow_sediment_256(
    THOMPSON_AA_SNOW_SEDIMENT_PARAMETERS)
{
    thompson_aa_snow_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_SNOW_SEDIMENT_ARGUMENTS);
}

template <int KMAX>
__device__ __forceinline__ void thompson_aa_graupel_sediment_impl(
    float* __restrict__ qg,
    float* __restrict__ graupel_number,
    const float* __restrict__ rain_density,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ dz,
    const float* __restrict__ active_columns,
    float* __restrict__ graupel_precipitation,
    float* __restrict__ qgten,
    float* __restrict__ ngten,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;
    const bool accumulate = qgten != nullptr;
    // :3903, ANY(L_qg): entry graupel that survived the sources.
    if (active_columns[column] == 0.0f) {
        graupel_precipitation[column] = 0.0f;
        if (accumulate) {
            // No fallout: :4058-4059 applies the sources' qgten and ngten.
            for (int k = 0; k < nz; ++k) {
                const size_t idx = IDX3(k, j, i);
                qg[idx] = thompson_aa_add(qg[idx],
                                          thompson_aa_mul(qgten[idx], dt));
                graupel_number[idx] = thompson_aa_add(graupel_number[idx],
                    thompson_aa_mul(ngten[idx], dt));
            }
        }
        return;
    }

    float density[KMAX];
    float graupel_mass[KMAX];
    float number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX];
    float number_flux[KMAX];
    float mass_tendency[KMAX];
    float number_tendency[KMAX];

    const bool any_rain = thompson_aa_column_has_rain(
        rain_density, j, i, nz, ny, nx);
    int sediment_top = 0;
    int nstep = 0;
    float mass_above = 0.0f;     // vtgk(kte+1) = vtngk(kte+1) = 0
    float number_above = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float rho = thompson_aa_air_density(
            pressure[idx], temperature[idx], fmaxf(1.0e-10f, qv[idx]));
        density[k] = rho;
        // qgten(k) and ngten(k) as the sources left them (zero when the
        // networks applied their tendencies in place).
        mass_tendency[k] = accumulate ? qgten[idx] : 0.0f;
        number_tendency[k] = accumulate ? ngten[idx] : 0.0f;
        const float held = reference_density[idx];

        // :3288-3302, on qg1d + qgten*DT.
        const float working = accumulate
            ? thompson_aa_add(qg[idx], thompson_aa_mul(qgten[idx], dt))
            : qg[idx];
        float rg = THOMPSON_AA_R1;
        float ng = THOMPSON_AA_R2;
        if (working > THOMPSON_AA_R1) {
            rg = thompson_aa_mul(working, held);
            ng = thompson_aa_classic_graupel_number_m3(rg);
        }
        graupel_mass[k] = rg;
        number[k] = ng;

        // :3740-3767.
        if (rg > THOMPSON_AA_R1) {
            // :3373-3374.
            const double lamg = (double)thompson_aa_powf(thompson_aa_div(
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    THOMPSON_AA_AM_G5, THOMPSON_AA_CGG3), THOMPSON_AA_OGG2),
                    ng), rg), THOMPSON_AA_OBMG);
            const double ilamg = __ddiv_rn(1.0, lamg);
            const double fall = thompson_aa_pow(ilamg,
                                                (double)THOMPSON_AA_BV_G);
            const float rhof = __fsqrt_rn(thompson_aa_div(
                THOMPSON_AA_RHO_NOT, any_rain ? rho : held));
            const float rhof_afall = thompson_aa_mul(rhof, THOMPSON_AA_AV_G);
            mass_velocity[k] = (float)__dmul_rn((double)thompson_aa_mul(
                thompson_aa_mul(rhof_afall, THOMPSON_AA_CGG6),
                THOMPSON_AA_OGG3), fall);
            // mu_g = 0: rhof*afall*cgg(7)/cgg(12) * ilamg**bfall.
            number_velocity[k] = (float)__dmul_rn((double)thompson_aa_div(
                thompson_aa_mul(rhof_afall, THOMPSON_AA_CGG7),
                THOMPSON_AA_CGG12), fall);
        } else {
            mass_velocity[k] = mass_above;
            number_velocity[k] = number_above;
        }
        mass_above = mass_velocity[k];
        number_above = number_velocity[k];

        // :3769-3773.
        if (mass_velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = thompson_aa_div(dz[idx], mass_velocity[k]);
            nstep = max(nstep, (int)thompson_aa_add(
                thompson_aa_div(dt, delta_tp), 1.0f));
        }
    }

    if (sediment_top == nz - 1) sediment_top = nz - 2;
    const float onstep = thompson_aa_onstep(nstep);
    const int steps = thompson_aa_nint(thompson_aa_div(1.0f, onstep));
    float precipitation = 0.0f;
    for (int step = 0; step < steps; ++step) {
        for (int k = nz - 1; k >= 0; --k) {
            mass_flux[k] = thompson_aa_mul(mass_velocity[k], graupel_mass[k]);
            number_flux[k] = thompson_aa_mul(number_velocity[k], number[k]);
        }
        int k = nz - 1;
        float odzq = thompson_aa_div(1.0f, dz[IDX3(k, j, i)]);
        float orho = thompson_aa_div(1.0f, density[k]);
        mass_tendency[k] = thompson_aa_sub(mass_tendency[k], thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(mass_flux[k], odzq), onstep), orho));
        number_tendency[k] = thompson_aa_sub(number_tendency[k],
            thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                number_flux[k], odzq), onstep), orho));
        graupel_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_sub(
            graupel_mass[k], thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                mass_flux[k], odzq), dt), onstep)));
        number[k] = fmaxf(THOMPSON_AA_R2, thompson_aa_sub(number[k],
            thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                number_flux[k], odzq), dt), onstep)));
        for (k = sediment_top; k >= 0; --k) {
            odzq = thompson_aa_div(1.0f, dz[IDX3(k, j, i)]);
            orho = thompson_aa_div(1.0f, density[k]);
            const float mass_divergence =
                thompson_aa_sub(mass_flux[k + 1], mass_flux[k]);
            const float number_divergence =
                thompson_aa_sub(number_flux[k + 1], number_flux[k]);
            mass_tendency[k] = thompson_aa_add(mass_tendency[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    mass_divergence, odzq), onstep), orho));
            number_tendency[k] = thompson_aa_add(number_tendency[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    number_divergence, odzq), onstep), orho));
            graupel_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_add(
                graupel_mass[k], thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(mass_divergence, odzq), dt), onstep)));
            number[k] = fmaxf(THOMPSON_AA_R2, thompson_aa_add(number[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    number_divergence, odzq), dt), onstep)));
        }
        // :3935-3936.
        if (graupel_mass[0] > thompson_aa_mul(THOMPSON_AA_R1, 1000.0f)) {
            precipitation = thompson_aa_add(precipitation, thompson_aa_mul(
                thompson_aa_mul(mass_flux[0], dt), onstep));
        }
    }
    // :4058-4059 without the bounds, which the graupel-number finalize
    // applies after the phase cleanup, as WRF orders them.  With the
    // accumulators qg and the number are still qg1d and ng1d and the
    // tendencies the whole qgten(k) and ngten(k).
    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        qg[idx] = thompson_aa_add(qg[idx],
                                  thompson_aa_mul(mass_tendency[k], dt));
        graupel_number[idx] = thompson_aa_add(graupel_number[idx],
            thompson_aa_mul(number_tendency[k], dt));
        if (accumulate) {
            qgten[idx] = mass_tendency[k];
            ngten[idx] = number_tendency[k];
        }
    }
    graupel_precipitation[column] = precipitation;
}

#ifdef __CUDACC_RTC__
// Level-parallel exact fallout. One block owns one column.
template <int KMAX>
__device__ __forceinline__ void thompson_aa_graupel_sediment_levels_impl(
    float* __restrict__ qg,
    float* __restrict__ graupel_number,
    const float* __restrict__ rain_density,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ dz,
    const float* __restrict__ active_columns,
    float* __restrict__ graupel_precipitation,
    float* __restrict__ qgten,
    float* __restrict__ ngten,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x;
    if (column >= ny * nx) return;
    const int lane = threadIdx.x;
    const int j = column / nx;
    const int i = column - j * nx;
    const bool accumulate = qgten != nullptr;
    // :3903, ANY(L_qg): entry graupel that survived the sources.
    if (active_columns[column] == 0.0f) {
        if (lane == 0) graupel_precipitation[column] = 0.0f;
        if (accumulate) {
            // No fallout: :4058-4059 applies the sources' qgten and ngten.
            for (int k = lane; k < nz; k += blockDim.x) {
                const size_t idx = IDX3(k, j, i);
                qg[idx] = thompson_aa_add(qg[idx],
                                          thompson_aa_mul(qgten[idx], dt));
                graupel_number[idx] = thompson_aa_add(graupel_number[idx],
                    thompson_aa_mul(ngten[idx], dt));
            }
        }
        return;
    }

    __shared__ float density[KMAX];
    __shared__ float graupel_mass[KMAX];
    __shared__ float number[KMAX];
    __shared__ float mass_velocity[KMAX];
    __shared__ float number_velocity[KMAX];
    __shared__ float mass_flux[KMAX];
    __shared__ float number_flux[KMAX];
    __shared__ float mass_tendency[KMAX];
    __shared__ float number_tendency[KMAX];

    __shared__ int sediment_top;
    __shared__ int nstep;
    __shared__ bool any_rain;
    if (lane == 0) {
        any_rain = thompson_aa_column_has_rain(rain_density, j, i, nz, ny, nx);
        sediment_top = 0;
        nstep = 0;
    }
    __syncthreads();



    for (int k = lane; k < nz; k += blockDim.x) {
        const size_t idx = IDX3(k, j, i);
        const float rho = thompson_aa_air_density(
            pressure[idx], temperature[idx], fmaxf(1.0e-10f, qv[idx]));
        density[k] = rho;
        // qgten(k) and ngten(k) as the sources left them (zero when the
        // networks applied their tendencies in place).
        mass_tendency[k] = accumulate ? qgten[idx] : 0.0f;
        number_tendency[k] = accumulate ? ngten[idx] : 0.0f;
        const float held = reference_density[idx];

        // :3288-3302, on qg1d + qgten*DT.
        const float working = accumulate
            ? thompson_aa_add(qg[idx], thompson_aa_mul(qgten[idx], dt))
            : qg[idx];
        float rg = THOMPSON_AA_R1;
        float ng = THOMPSON_AA_R2;
        if (working > THOMPSON_AA_R1) {
            rg = thompson_aa_mul(working, held);
            ng = thompson_aa_classic_graupel_number_m3(rg);
        }
        graupel_mass[k] = rg;
        number[k] = ng;

        // :3740-3767.
        if (rg > THOMPSON_AA_R1) {
            // :3373-3374.
            const double lamg = (double)thompson_aa_powf(thompson_aa_div(
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    THOMPSON_AA_AM_G5, THOMPSON_AA_CGG3), THOMPSON_AA_OGG2),
                    ng), rg), THOMPSON_AA_OBMG);
            const double ilamg = __ddiv_rn(1.0, lamg);
            const double fall = thompson_aa_pow(ilamg,
                                                (double)THOMPSON_AA_BV_G);
            const float rhof = __fsqrt_rn(thompson_aa_div(
                THOMPSON_AA_RHO_NOT, any_rain ? rho : held));
            const float rhof_afall = thompson_aa_mul(rhof, THOMPSON_AA_AV_G);
            mass_velocity[k] = (float)__dmul_rn((double)thompson_aa_mul(
                thompson_aa_mul(rhof_afall, THOMPSON_AA_CGG6),
                THOMPSON_AA_OGG3), fall);
            // mu_g = 0: rhof*afall*cgg(7)/cgg(12) * ilamg**bfall.
            number_velocity[k] = (float)__dmul_rn((double)thompson_aa_div(
                thompson_aa_mul(rhof_afall, THOMPSON_AA_CGG7),
                THOMPSON_AA_CGG12), fall);
        } else {
            mass_velocity[k] = -1.0f;
            number_velocity[k] = -1.0f;
        }
    }
    __syncthreads();
    if (lane == 0) {
        float mass_above = 0.0f, number_above = 0.0f;
        for (int k = nz - 1; k >= 0; --k) {
            if (!(graupel_mass[k] > THOMPSON_AA_R1)) {
                mass_velocity[k] = mass_above;
                number_velocity[k] = number_above;
            }
            mass_above = mass_velocity[k];
            number_above = number_velocity[k];
            if (mass_velocity[k] > 1.0e-3f) {
                sediment_top = max(sediment_top, k);
                const float delta_tp = thompson_aa_div(dz[IDX3(k,j,i)], mass_velocity[k]);
                nstep = max(nstep, (int)thompson_aa_add(thompson_aa_div(dt, delta_tp), 1.0f));
            }
        }
        if (sediment_top == nz - 1) sediment_top = nz - 2;
    }
    __syncthreads();
    const float onstep = thompson_aa_onstep(nstep);
    const int steps = thompson_aa_nint(thompson_aa_div(1.0f, onstep));
    float precipitation = 0.0f;
    for (int step = 0; step < steps; ++step) {
        for (int k = lane; k < nz; k += blockDim.x) {
            mass_flux[k] = thompson_aa_mul(mass_velocity[k], graupel_mass[k]);
            number_flux[k] = thompson_aa_mul(number_velocity[k], number[k]);
        }
        __syncthreads();
        int k = nz - 1;
        if (lane == nz - 1) {
        float odzq = thompson_aa_div(1.0f, dz[IDX3(k, j, i)]);
        float orho = thompson_aa_div(1.0f, density[k]);
        mass_tendency[k] = thompson_aa_sub(mass_tendency[k], thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(mass_flux[k], odzq), onstep), orho));
        number_tendency[k] = thompson_aa_sub(number_tendency[k],
            thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                number_flux[k], odzq), onstep), orho));
        graupel_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_sub(
            graupel_mass[k], thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                mass_flux[k], odzq), dt), onstep)));
        number[k] = fmaxf(THOMPSON_AA_R2, thompson_aa_sub(number[k],
            thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                number_flux[k], odzq), dt), onstep)));
        }
        for (k = lane; k <= sediment_top; k += blockDim.x) {
            const float odzq = thompson_aa_div(1.0f, dz[IDX3(k, j, i)]);
            const float orho = thompson_aa_div(1.0f, density[k]);
            const float mass_divergence =
                thompson_aa_sub(mass_flux[k + 1], mass_flux[k]);
            const float number_divergence =
                thompson_aa_sub(number_flux[k + 1], number_flux[k]);
            mass_tendency[k] = thompson_aa_add(mass_tendency[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    mass_divergence, odzq), onstep), orho));
            number_tendency[k] = thompson_aa_add(number_tendency[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    number_divergence, odzq), onstep), orho));
            graupel_mass[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_add(
                graupel_mass[k], thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(mass_divergence, odzq), dt), onstep)));
            number[k] = fmaxf(THOMPSON_AA_R2, thompson_aa_add(number[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    number_divergence, odzq), dt), onstep)));
        }
        __syncthreads();
        // :3935-3936.
        if (lane == 0 && graupel_mass[0] > thompson_aa_mul(THOMPSON_AA_R1, 1000.0f)) {
            precipitation = thompson_aa_add(precipitation, thompson_aa_mul(
                thompson_aa_mul(mass_flux[0], dt), onstep));
        }
    }
    // :4058-4059 without the bounds, which the graupel-number finalize
    // applies after the phase cleanup, as WRF orders them.  With the
    // accumulators qg and the number are still qg1d and ng1d and the
    // tendencies the whole qgten(k) and ngten(k).
    for (int k = lane; k < nz; k += blockDim.x) {
        const size_t idx = IDX3(k, j, i);
        qg[idx] = thompson_aa_add(qg[idx],
                                  thompson_aa_mul(mass_tendency[k], dt));
        graupel_number[idx] = thompson_aa_add(graupel_number[idx],
            thompson_aa_mul(number_tendency[k], dt));
        if (accumulate) {
            qgten[idx] = mass_tendency[k];
            ngten[idx] = number_tendency[k];
        }
    }
    if (lane == 0) graupel_precipitation[column] = precipitation;
}

#endif  // __CUDACC_RTC__

#define THOMPSON_AA_GRAUPEL_SEDIMENT_PARAMETERS                           \
    float* __restrict__ qg, float* __restrict__ graupel_number,          \
    const float* __restrict__ rain_density,                              \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ dz,                                        \
    const float* __restrict__ active_columns,                            \
    float* __restrict__ graupel_precipitation,                           \
    float* __restrict__ qgten, float* __restrict__ ngten, float dt,      \
    int nz, int ny, int nx

#define THOMPSON_AA_GRAUPEL_SEDIMENT_ARGUMENTS                            \
    qg, graupel_number, rain_density, temperature, pressure, qv,         \
    reference_density, dz, active_columns, graupel_precipitation, qgten, \
    ngten, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_graupel_sediment_64(
    THOMPSON_AA_GRAUPEL_SEDIMENT_PARAMETERS)
{
    thompson_aa_graupel_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_GRAUPEL_SEDIMENT_ARGUMENTS);
}

#ifdef __CUDACC_RTC__
extern "C" __global__ void thompson_aa_graupel_sediment_levels_64(
    THOMPSON_AA_GRAUPEL_SEDIMENT_PARAMETERS)
{
    thompson_aa_graupel_sediment_levels_impl<64>(
        THOMPSON_AA_GRAUPEL_SEDIMENT_ARGUMENTS);
}
#endif  // __CUDACC_RTC__
extern "C" __global__ void thompson_aa_graupel_sediment_256(
    THOMPSON_AA_GRAUPEL_SEDIMENT_PARAMETERS)
{
    thompson_aa_graupel_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_GRAUPEL_SEDIMENT_ARGUMENTS);
}

// mp_gt_driver :1294-1308, in the driver's own order of addition:
//     RAINNCV = pptrain + pptsnow + pptgraul + pptice
//     RAINNC  = RAINNC + pptrain + pptsnow + pptgraul + pptice
//     SNOWNCV = pptsnow + pptice,  SNOWNC = SNOWNC + pptsnow + pptice
//     GRAUPELNCV = pptgraul,       GRAUPELNC = GRAUPELNC + pptgraul
//     SR = (pptsnow + pptgraul + pptice)/(RAINNCV + 1.e-12)
// The four fallout passes leave their column totals in the four arguments
// below (the rain pass in sr, the snow pass in rainncv, the ice pass in
// snowncv and the graupel pass in graupelncv); this kernel reads all four
// before it writes any.
extern "C" __global__ void thompson_aa_surface_precipitation(
    float* __restrict__ rainnc, float* __restrict__ rainncv,
    float* __restrict__ snownc, float* __restrict__ snowncv,
    float* __restrict__ graupelnc, float* __restrict__ graupelncv,
    float* __restrict__ sr, int ncol)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    const float pptrain = sr[column];
    const float pptsnow = rainncv[column];
    const float pptice = snowncv[column];
    const float pptgraul = graupelncv[column];
    const float total = thompson_aa_add(thompson_aa_add(
        thompson_aa_add(pptrain, pptsnow), pptgraul), pptice);
    rainncv[column] = total;
    rainnc[column] = thompson_aa_add(thompson_aa_add(thompson_aa_add(
        thompson_aa_add(rainnc[column], pptrain), pptsnow), pptgraul), pptice);
    snowncv[column] = thompson_aa_add(pptsnow, pptice);
    snownc[column] = thompson_aa_add(thompson_aa_add(snownc[column], pptsnow),
                                     pptice);
    graupelncv[column] = pptgraul;
    graupelnc[column] = thompson_aa_add(graupelnc[column], pptgraul);
    sr[column] = thompson_aa_div(thompson_aa_add(thompson_aa_add(
        pptsnow, pptgraul), pptice), thompson_aa_add(total, 1.0e-12f));
}


#if defined(THOMPSON_AA_WRF39)
// ---------------------------------------------------------------------------
// THE FORK'S ICE AND GRAUPEL FALLOUT (RunConfig.thompson_version =
// "wrf_39_noaa").
// ---------------------------------------------------------------------------
//
// thompson.cu's classic ice and graupel fallout (thompson_ice_sediment_impl,
// thompson_graupel_sediment_impl) carry v4.6.1's constants and its graupel
// number, and that unit stays byte-frozen, so the fork's two passes live
// here.  Each is the classic pass statement for statement with only the
// fork's own differences:
//
//   ice     av_i = 1847.5 (fork :140, audit T9), the 499.D3 number ceiling
//           (fork :2879, :3733, audit T8), surface ice counted above R1*10
//           (fork :3603, audit T21);
//   graupel the slope from the fork's column intercept (fork :3110-3133,
//           thompson_aa_wrf39_graupel_intercept on the post-source state,
//           audit T1) with am_g at rho_g = 500 (audit T11), no number and
//           no size clamp; above 0 C the fall speed is at least the rain's,
//           vtgk = MAX(vtg, vtrk) (fork :3500-3505, audit T15); surface
//           graupel counted above R1*10 (fork :3653, audit T21).
//
// The density decisions are the classic passes': the fallout state is
// formed with the held pre-adjustment density, the fall speed and the
// tendency with the current one.

template <int KMAX>
__device__ __forceinline__ void thompson_aa_wrf39_ice_sediment_impl(
    float* __restrict__ qi,
    float* __restrict__ ni,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ snownc,
    float* __restrict__ snowncv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float density[KMAX];
    float ice_mass[KMAX];
    float ice_number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX];
    float number_flux[KMAX];
    float qi_tendency[KMAX];
    float ni_tendency[KMAX];
    float qi_initial[KMAX];
    float ni_initial[KMAX];

    const float am_i = THOMPSON_AA_AM_I;
    const float oig2 = 0.16666667163372040f;
    const float rho_not = __fdiv_rn(101325.0f, 287.05f * 298.0f);
    const float av_i = 1847.5f;
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above_mass = 0.0f;
    float velocity_above_number = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density[k] = rho;
        qi_initial[k] = qi[idx];
        ni_initial[k] = ni[idx];
        qi_tendency[k] = 0.0f;
        ni_tendency[k] = 0.0f;
        const float ice_state_rho = reference_density == nullptr
            ? rho : reference_density[idx];
        if (qi[idx] > 1.0e-12f && qi[idx] * ice_state_rho > 1.0e-12f) {
            const float ri = qi[idx] * ice_state_rho;
            float nn = fmaxf(1.0e-6f, ni[idx] * ice_state_rho);
            if (nn <= 1.0e-6f) {
                const double lambda = 4.0 / 5.0e-6;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = fminf(THOMPSON_AA_WRF39_NI_MAX,
                           (float)((double)prefix * thompson_aa_pow(lambda, 3.0)));
            }
            const float lambda_arg = am_i * 6.0f * nn / ri;
            double lambda = (double)thompson_aa_powf(lambda_arg, 0.33333334326744080f);
            float diameter = (float)(4.0 / lambda);
            if (diameter < 5.0e-6f) {
                diameter = 5.0e-6f;
                lambda = 4.0 / (double)diameter;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = fminf(THOMPSON_AA_WRF39_NI_MAX,
                           (float)((double)prefix * thompson_aa_pow(lambda, 3.0)));
            } else if (diameter > 300.0e-6f) {
                diameter = 300.0e-6f;
                lambda = 4.0 / (double)diameter;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = (float)((double)prefix * thompson_aa_pow(lambda, 3.0));
            }
            ice_mass[k] = ri;
            ice_number[k] = nn;
            const float rhof = sqrtf(rho_not / rho);
            const double inverse_lambda = 1.0 / lambda;
            const float mass_prefix = rhof * av_i * 24.0f * oig2;
            mass_velocity[k] = (float)((double)mass_prefix * inverse_lambda);
            const float number_prefix = __fdiv_rn(rhof * av_i
                * 3.3233511f, 1.3293403f);
            number_velocity[k] = (float)((double)number_prefix
                                          * inverse_lambda);
        } else {
            const bool l_qi = qi[idx] > 1.0e-12f;
            ice_mass[k] = l_qi ? qi[idx] * ice_state_rho : 1.0e-12f;
            ice_number[k] = l_qi
                ? fmaxf(1.0e-6f, ni[idx] * ice_state_rho) : 1.0e-6f;
            mass_velocity[k] = velocity_above_mass;
            number_velocity[k] = velocity_above_number;
        }
        velocity_above_mass = mass_velocity[k];
        velocity_above_number = number_velocity[k];
        if (mass_velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = dz[idx] / mass_velocity[k];
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    if (sediment_top == nz - 1) sediment_top = nz - 2;
    nstep = max(nstep, 1);
    const float onstep = 1.0f / (float)nstep;
    const float dt_substep = dt * onstep;
    float exported = 0.0f;

    for (int step = 0; step < nstep; ++step) {
        for (int k = nz - 1; k >= 0; --k) {
            mass_flux[k] = mass_velocity[k] * ice_mass[k];
            number_flux[k] = number_velocity[k] * ice_number[k];
        }
        int k = nz - 1;
        size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density[k];
        qi_tendency[k] -= mass_flux[k] * inv_dz * onstep * inv_rho;
        ni_tendency[k] -= number_flux[k] * inv_dz * onstep * inv_rho;
        ice_mass[k] = fmaxf(1.0e-12f,
            ice_mass[k] - mass_flux[k] * inv_dz * dt_substep);
        ice_number[k] = fmaxf(1.0e-6f,
            ice_number[k] - number_flux[k] * inv_dz * dt_substep);
        for (k = sediment_top; k >= 0; --k) {
            idx = IDX3(k, j, i);
            inv_dz = 1.0f / dz[idx];
            inv_rho = 1.0f / density[k];
            const float mass_divergence = mass_flux[k + 1] - mass_flux[k];
            const float number_divergence =
                number_flux[k + 1] - number_flux[k];
            qi_tendency[k] += mass_divergence * inv_dz * onstep * inv_rho;
            ni_tendency[k] += number_divergence * inv_dz * onstep * inv_rho;
            ice_mass[k] = fmaxf(1.0e-12f,
                ice_mass[k] + mass_divergence * inv_dz * dt_substep);
            ice_number[k] = fmaxf(1.0e-6f,
                ice_number[k] + number_divergence * inv_dz * dt_substep);
        }
        // fork :3603, ri(kts) > R1*10.
        if (ice_mass[0] > 1.0e-11f) {
            exported += mass_flux[0] * dt_substep;
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qi_new = qi_initial[k] + qi_tendency[k] * dt;
        float ni_new = fmaxf(1.0e-6f / density[k],
                             ni_initial[k] + ni_tendency[k] * dt);
        if (qi_new <= 1.0e-12f) {
            qi[idx] = qi_new;
            ni[idx] = ni_initial[k] + ni_tendency[k] * dt;
            continue;
        }
        const float lambda_arg = am_i * 6.0f * ni_new / qi_new;
        double lambda = (double)thompson_aa_powf(lambda_arg, 0.33333334326744080f);
        float diameter = (float)(4.0 / lambda);
        if (diameter < 5.0e-6f) diameter = 5.0e-6f;
        else if (diameter > 300.0e-6f) diameter = 300.0e-6f;
        lambda = 4.0 / (double)diameter;
        const float prefix = __fdiv_rn(oig2 * qi_new, am_i);
        // fork :3733, 499.D3/rho.
        ni_new = fminf((float)((double)prefix * thompson_aa_pow(lambda, 3.0)),
                       THOMPSON_AA_WRF39_NI_MAX / density[k]);
        qi[idx] = qi_new;
        ni[idx] = ni_new;
    }
    rainncv[column] = exported;
    snowncv[column] = exported;
    rainnc[column] += exported;
    snownc[column] += exported;
}

#define THOMPSON_AA_WRF39_ICE_SEDIMENT_PARAMETERS                        \
    float* __restrict__ qi, float* __restrict__ ni,                      \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ snownc,             \
    float* __restrict__ snowncv, float dt, int nz, int ny, int nx

#define THOMPSON_AA_WRF39_ICE_SEDIMENT_ARGUMENTS                         \
    qi, ni, temperature, pressure, qv, reference_density, dz, rainnc,    \
    rainncv, snownc, snowncv, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_wrf39_ice_sediment_64(
    THOMPSON_AA_WRF39_ICE_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_ice_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_WRF39_ICE_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_wrf39_ice_sediment_256(
    THOMPSON_AA_WRF39_ICE_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_ice_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_WRF39_ICE_SEDIMENT_ARGUMENTS);
}

// graupel_intercept: N0_exp per level from thompson_aa_wrf39_graupel_
// intercept on the post-source state.  melt_rain_qr/nr and
// melt_rain_density are the rain fallout's own inputs (the rain evaporation
// writes the density with L_qr carried in its sign, as the classic snow
// pass reads it), so vtrk(k) is the rain pass's speed, inherited from above
// where a level has no rain.
template <int KMAX>
__device__ __forceinline__ void thompson_aa_wrf39_graupel_sediment_impl(
    float* __restrict__ qg,
    const float* __restrict__ graupel_intercept,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ melt_rain_qr,
    const float* __restrict__ melt_rain_nr,
    const float* __restrict__ melt_rain_density,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ graupelnc,
    float* __restrict__ graupelncv,
    const float* __restrict__ active_columns,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx || active_columns[column] == 0.0f) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float density[KMAX];
    float graupel_mass[KMAX];
    float mass_velocity[KMAX];
    float mass_flux[KMAX];
    float qg_tendency[KMAX];
    float qg_initial[KMAX];

    const float ogg3 = 0.16666667163372040f;
    const float rho_not = __fdiv_rn(101325.0f, 287.05f * 298.0f);
    // bv_g = 0.89 as the REAL(4) PARAMETER holds it, cgg(6) = WGAMMA(4.89).
    const double bv_g = (double)0.89f;
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above = 0.0f;
    float rain_velocity_above = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density[k] = rho;
        qg_initial[k] = qg[idx];
        qg_tendency[k] = 0.0f;

        // The rain pass's vtrk(k), as thompson.cu's snow fallout forms it.
        float rain_velocity = 0.0f;
        {
            const float carried = melt_rain_density[idx];
            const float rain_density = fabsf(carried);
            const bool l_qr = carried != 0.0f;
            const bool rewritten = carried < 0.0f;
            const float rain_rr = !l_qr ? 1.0e-12f
                : rewritten ? fmaxf(1.0e-12f, melt_rain_qr[idx] * rain_density)
                : melt_rain_qr[idx] * rain_density;
            if (rain_rr > 1.0e-12f) {
                const float am_r = THOMPSON_AA_AM_R;
                const float rain_number = fmaxf(
                    1.0e-6f, melt_rain_nr[idx] * rain_density);
                const double rain_lambda = (double)thompson_aa_powf(
                    am_r * 6.0f * rain_number / rain_rr,
                    0.33333334326744080f);
                const float rain_rhof = sqrtf(rho_not / rho);
                rain_velocity = (float)(
                    (double)(rain_rhof * 4854.0f * 24.0f * ogg3)
                    * thompson_aa_pow(rain_lambda, 4.0)
                    * thompson_aa_pow(rain_lambda + 195.0, -5.0));
            } else {
                rain_velocity = rain_velocity_above;
            }
            rain_velocity_above = rain_velocity;
        }

        if (qg[idx] > 1.0e-12f) {
            const float state_rho = reference_density == nullptr
                ? rho : reference_density[idx];
            const float rg = qg[idx] * state_rho;
            double lamg, ilamg, n0_g;
            thompson_aa_wrf39_graupel_slope(
                graupel_intercept[idx], rg, &lamg, &ilamg, &n0_g);
            const float rhof = sqrtf(rho_not / rho);
            // vtg = rhof*av_g*cgg(6)*ogg3 * ilamg**bv_g, fork :3500.
            const float prefix = rhof * 442.0f * 20.3632278f * ogg3;
            float velocity = (float)((double)prefix * thompson_aa_pow(ilamg, bv_g));
            // fork :3501-3505.
            if (temperature[idx] > 273.15f) {
                velocity = fmaxf(velocity, rain_velocity);
            }
            mass_velocity[k] = velocity;
            graupel_mass[k] = rg;
        } else {
            graupel_mass[k] = 1.0e-12f;
            mass_velocity[k] = velocity_above;
        }
        velocity_above = mass_velocity[k];

        if (mass_velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = dz[idx] / mass_velocity[k];
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    if (sediment_top == nz - 1) sediment_top = nz - 2;
    nstep = max(nstep, 1);
    const float onstep = 1.0f / (float)nstep;
    const float dt_substep = dt * onstep;
    float exported = 0.0f;

    for (int step = 0; step < nstep; ++step) {
        for (int k = nz - 1; k >= 0; --k) {
            mass_flux[k] = mass_velocity[k] * graupel_mass[k];
        }
        int k = nz - 1;
        size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density[k];
        qg_tendency[k] -= mass_flux[k] * inv_dz * onstep * inv_rho;
        graupel_mass[k] = fmaxf(1.0e-12f,
            graupel_mass[k] - mass_flux[k] * inv_dz * dt_substep);
        for (k = sediment_top; k >= 0; --k) {
            idx = IDX3(k, j, i);
            inv_dz = 1.0f / dz[idx];
            inv_rho = 1.0f / density[k];
            const float divergence = mass_flux[k + 1] - mass_flux[k];
            qg_tendency[k] += divergence * inv_dz * onstep * inv_rho;
            graupel_mass[k] = fmaxf(1.0e-12f,
                graupel_mass[k] + divergence * inv_dz * dt_substep);
        }
        // fork :3653, rg(kts) > R1*10.
        if (graupel_mass[0] > 1.0e-11f) {
            exported += mass_flux[0] * dt_substep;
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qg_new = qg_initial[k] + qg_tendency[k] * dt;
        qg[idx] = qg_new <= 1.0e-12f ? 0.0f : qg_new;
    }
    rainncv[column] += exported;
    graupelncv[column] += exported;
    rainnc[column] += exported;
    graupelnc[column] += exported;
}

#define THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_PARAMETERS                    \
    float* __restrict__ qg, const float* __restrict__ graupel_intercept, \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ melt_rain_qr,                              \
    const float* __restrict__ melt_rain_nr,                              \
    const float* __restrict__ melt_rain_density,                         \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ graupelnc,          \
    float* __restrict__ graupelncv,                                      \
    const float* __restrict__ active_columns,                            \
    float dt, int nz, int ny, int nx

#define THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_ARGUMENTS                     \
    qg, graupel_intercept, temperature, pressure, qv, reference_density, \
    melt_rain_qr, melt_rain_nr, melt_rain_density, dz, rainnc, rainncv,  \
    graupelnc, graupelncv, active_columns, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_wrf39_graupel_sediment_64(
    THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_graupel_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_wrf39_graupel_sediment_256(
    THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_graupel_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_ARGUMENTS);
}
#endif  // THOMPSON_AA_WRF39


#if defined(THOMPSON_AA_WRF39)
// ---------------------------------------------------------------------------
// THE FORK'S SNOW FALLOUT (RunConfig.thompson_version = "wrf_39_noaa").
// ---------------------------------------------------------------------------
//
// thompson.cu's thompson_snow_sediment_impl (the RAIN_PRESENCE arm the
// coupled adapter launches) statement for statement, with the fork's two
// differences:
//
//   surface snow is counted above R1*10 (fork :3628, audit T21);
//   the speed of snow on a level warmer than 0 C follows RunConfig.
//   thompson_fork_snow_fall, passed as ``singular_fall``:
//     0 ("blend", the default): WRF v4.6.1's rain-share blend where snow
//        melts, vtsk = vts*SR + (1-SR)*vtrk with SR = rs/(rs+rr) (:3722-
//        3724), the later upstream form the fork itself carries commented
//        out at fork :3475-3476 as an upstream bug fix;
//     1 ("wrf_39_noaa"): the fork's own expression (fork :3472-3478,
//        audit T4), above 0.1 C
//          vtsk = MAX(vts*vts_boost, vts*((vtrk - vts*vts_boost)/(T - T_0)))
//        and vts*vts_boost between 0 and 0.1 C, with vts_boost = 1.5 on
//        every level the source stage found at or above 0 C (fork :2151).
//        The divisor goes to zero just above +0.1 C, where 1 m/s snow
//        under 5 m/s rain falls at about 35 m/s: a defect kept by name for
//        the owner's ruling, never the default.
template <int KMAX>
__device__ __forceinline__ void thompson_aa_wrf39_snow_sediment_impl(
    float* __restrict__ qs,
    const float* __restrict__ snow_melt_marker,
    const float* __restrict__ melt_rain_qr,
    const float* __restrict__ melt_rain_nr,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ reference_temperature,
    const float* __restrict__ velocity_boost,
    const float* __restrict__ melt_rain_density,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ snownc,
    float* __restrict__ snowncv,
    int singular_fall, float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float density[KMAX];
    float snow_mass[KMAX];
    float mass_velocity[KMAX];
    float mass_flux[KMAX];
    float qs_tendency[KMAX];
    float qs_initial[KMAX];

    const float rho_not = __fdiv_rn(101325.0f, 287.05f * 298.0f);
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above = 0.0f;
    float rain_velocity_above = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density[k] = rho;
        qs_initial[k] = qs[idx];
        qs_tendency[k] = 0.0f;

        float rain_rr = 1.0e-12f;
        float rain_velocity = 0.0f;
        {
            const float carried = melt_rain_density[idx];
            const float rain_density = fabsf(carried);
            const bool l_qr = carried != 0.0f;
            const bool rewritten = carried < 0.0f;
            rain_rr = !l_qr ? 1.0e-12f
                : rewritten ? fmaxf(1.0e-12f, melt_rain_qr[idx] * rain_density)
                : melt_rain_qr[idx] * rain_density;
            if (rain_rr > 1.0e-12f) {
                const float am_r = THOMPSON_AA_AM_R;
                const float rain_number = fmaxf(
                    1.0e-6f, melt_rain_nr[idx] * rain_density);
                const double rain_lambda = (double)thompson_aa_powf(
                    am_r * 6.0f * rain_number / rain_rr,
                    0.33333334326744080f);
                const float rain_rhof = sqrtf(rho_not / rho);
                rain_velocity = (float)(
                    (double)(rain_rhof * 4854.0f * 24.0f
                             * 0.16666667163372040f)
                    * thompson_aa_pow(rain_lambda, 4.0)
                    * thompson_aa_pow(rain_lambda + 195.0, -5.0));
            } else {
                rain_velocity = rain_velocity_above;
            }
            rain_velocity_above = rain_velocity;
        }

        const float snow_state_rho = reference_density[idx];
        if (qs[idx] > 1.0e-12f && qs[idx] * snow_state_rho > 1.0e-12f) {
            const float rs = qs[idx] * snow_state_rho;
            const float smob = rs * THOMPSON_AA_OAMS;      // rs*oams
            const float tc0 = fminf(-0.1f,
                                    reference_temperature[idx] - 273.15f);
            const float moment = 3.0f;
            const float tc02 = tc0 * tc0;
            const float moment2 = moment * moment;
            const float loga = 5.065339f + -0.062659f * tc0
                + -3.032362f * moment + 0.029469f * tc0 * moment
                + -0.000285f * tc02 + 0.31255f * moment2
                + 0.000204f * tc02 * moment
                + 0.003199f * tc0 * moment2
                + 0.0f * tc02 * tc0
                + -0.015952f * moment2 * moment;
            const float exponent = 0.476221f + -0.015896f * tc0
                + 0.165977f * moment + 0.007468f * tc0 * moment
                + -0.000141f * tc02 + 0.060366f * moment2
                + 0.000079f * tc02 * moment
                + 0.000594f * tc0 * moment2
                + 0.0f * tc02 * tc0
                + -0.003577f * moment2 * moment;
            const float smoc = thompson_aa_powf(10.0f, loga) * thompson_aa_powf(smob, exponent);
            const float mean_ratio = smob / smoc;
            float ils1 = 1.0f / (mean_ratio * 20.78f + 100.0f);
            float ils2 = 1.0f / (mean_ratio * 3.29f + 100.0f);
            const float ratio_power = thompson_aa_powf(mean_ratio, 0.6357f);
            const float numerator1 = 490.6f * 3.51325202f
                * thompson_aa_powf(ils1, 3.55f);
            const float numerator2 = 17.46f * ratio_power * 7.61279917f
                * thompson_aa_powf(ils2, 4.1857f);
            ils1 = 1.0f / (mean_ratio * 20.78f);
            ils2 = 1.0f / (mean_ratio * 3.29f);
            const float denominator1 = 490.6f * 2.0f
                * thompson_aa_powf(ils1, 3.0f);
            const float denominator2 = 17.46f * ratio_power * 3.87160635f
                * thompson_aa_powf(ils2, 3.6357f);
            const float rhof = sqrtf(rho_not / rho);
            const float vts = rhof * 40.0f
                * (numerator1 + numerator2)
                / (denominator1 + denominator2);
            const float boost = velocity_boost[idx];
            float snow_velocity;
            const float temp_k = temperature[idx];
            if (singular_fall) {
                if (temp_k > 273.15f + 0.1f) {
                    snow_velocity = fmaxf(vts * boost,
                        vts * ((rain_velocity - vts * boost)
                               / (temp_k - 273.15f)));
                } else {
                    snow_velocity = vts * boost;
                }
            } else if (snow_melt_marker[idx] != 0.0f) {
                const float solid_fraction = rs / (rs + rain_rr);
                snow_velocity = vts * boost * solid_fraction
                    + rain_velocity * (1.0f - solid_fraction);
            } else {
                snow_velocity = vts * boost;
            }
            mass_velocity[k] = snow_velocity;
            snow_mass[k] = rs;
        } else {
            snow_mass[k] = qs[idx] > 1.0e-12f
                ? qs[idx] * snow_state_rho : 1.0e-12f;
            mass_velocity[k] = velocity_above;
        }
        velocity_above = mass_velocity[k];

        if (mass_velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = dz[idx] / mass_velocity[k];
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    if (sediment_top == nz - 1) sediment_top = nz - 2;
    nstep = max(nstep, 1);
    const float onstep = 1.0f / (float)nstep;
    const float dt_substep = dt * onstep;
    float exported = 0.0f;

    for (int step = 0; step < nstep; ++step) {
        for (int k = nz - 1; k >= 0; --k) {
            mass_flux[k] = mass_velocity[k] * snow_mass[k];
        }
        int k = nz - 1;
        size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density[k];
        qs_tendency[k] -= mass_flux[k] * inv_dz * onstep * inv_rho;
        snow_mass[k] = fmaxf(1.0e-12f,
            snow_mass[k] - mass_flux[k] * inv_dz * dt_substep);
        for (k = sediment_top; k >= 0; --k) {
            idx = IDX3(k, j, i);
            inv_dz = 1.0f / dz[idx];
            inv_rho = 1.0f / density[k];
            const float divergence = mass_flux[k + 1] - mass_flux[k];
            qs_tendency[k] += divergence * inv_dz * onstep * inv_rho;
            snow_mass[k] = fmaxf(1.0e-12f,
                snow_mass[k] + divergence * inv_dz * dt_substep);
        }
        // fork :3628, rs(kts) > R1*10.
        if (snow_mass[0] > 1.0e-11f) {
            exported += mass_flux[0] * dt_substep;
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qs_new = qs_initial[k] + qs_tendency[k] * dt;
        qs[idx] = qs_new <= 1.0e-12f ? 0.0f : qs_new;
    }
    rainncv[column] += exported;
    snowncv[column] += exported;
    rainnc[column] += exported;
    snownc[column] += exported;
}

#define THOMPSON_AA_WRF39_SNOW_SEDIMENT_PARAMETERS                       \
    float* __restrict__ qs, const float* __restrict__ snow_melt_marker,  \
    const float* __restrict__ melt_rain_qr,                              \
    const float* __restrict__ melt_rain_nr,                              \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ reference_temperature,                     \
    const float* __restrict__ velocity_boost,                            \
    const float* __restrict__ melt_rain_density,                         \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ snownc,             \
    float* __restrict__ snowncv, int singular_fall, float dt,            \
    int nz, int ny, int nx

#define THOMPSON_AA_WRF39_SNOW_SEDIMENT_ARGUMENTS                        \
    qs, snow_melt_marker, melt_rain_qr, melt_rain_nr, temperature,       \
    pressure, qv, reference_density, reference_temperature,              \
    velocity_boost, melt_rain_density, dz, rainnc, rainncv, snownc,      \
    snowncv, singular_fall, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_wrf39_snow_sediment_64(
    THOMPSON_AA_WRF39_SNOW_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_snow_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_WRF39_SNOW_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_wrf39_snow_sediment_256(
    THOMPSON_AA_WRF39_SNOW_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_snow_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_WRF39_SNOW_SEDIMENT_ARGUMENTS);
}

// The fork's vts_boost on the levels its source stage found at or above
// 0 C (fork :2151): 1.5 where the singular fall is selected, 1.0 (the
// v4.6.1 start value the cold network already wrote) otherwise.
extern "C" __global__ void thompson_aa_wrf39_warm_snow_boost(
    const float* __restrict__ entry_warm_mask,
    float* __restrict__ velocity_boost, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (entry_warm_mask[idx] != 0.0f) velocity_boost[idx] = 1.5f;
}
#endif  // THOMPSON_AA_WRF39



#if defined(THOMPSON_AA_WRF39)
// ---------------------------------------------------------------------------
// THE FORK'S RAIN FALLOUT (RunConfig.thompson_version = "wrf_39_noaa").
// ---------------------------------------------------------------------------
//
// thompson.cu's thompson_rain_sediment_impl<KMAX, true> (the rain-presence
// arm the coupled adapter launches, accumulating the surface totals)
// statement for statement, with the fork's one difference:
//
//   surface rain is counted above R1*10 (fork :3556, rr(kts) > R1*10),
//   where v4.6.1 counts it above R1*1000 (:3817).  Audit T21.
//
// thompson.cu stays byte-frozen (it is the mp=8 numerics guarantee), so the
// pass lives here beside the fork's ice, snow and graupel passes, which
// carry the same R1*10 test (fork :3603, :3628, :3653).  The empty-column
// shortcut is thompson.cu's: a column with no rain presence performs exactly
// the writes of the full sweep, and its export is zero under either
// threshold because every level then holds the R1 sentinel.
__device__ __forceinline__ bool thompson_aa_wrf39_empty_rain_column(
    const float* q, const float* temperature, const float* pressure,
    const float* qv, const float* dz, const float* presence, float dt,
    int column, int nz, int ny, int nx)
{
    if (!(dt >= 0.0f && dt <= 100000.0f)) return false;
    for (int k = 0; k < nz; ++k) {
        const size_t idx = (size_t)k * ny * nx + column;
        if (!(q[idx] >= -1.0f && q[idx] <= 1.0f)) return false;
        if (presence[idx] != 0.0f) return false;
        if (!(temperature[idx] >= 100.0f && temperature[idx] <= 400.0f
                && pressure[idx] >= 1.0f && pressure[idx] <= 120000.0f
                && qv[idx] >= -1.0f && qv[idx] <= 1.0f
                && dz[idx] >= 1.0f && dz[idx] <= 100000.0f)) return false;
    }
    return true;
}

template <int KMAX>
__device__ __forceinline__ void thompson_aa_wrf39_rain_sediment_impl(
    float* __restrict__ qr,
    float* __restrict__ nr,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    if (thompson_aa_wrf39_empty_rain_column(qr, temperature, pressure, qv, dz,
            reference_density, dt, column, nz, ny, nx)) {
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float qr_new = qr[idx] + 0.0f * dt;
            if (qr_new <= 1.0e-12f) {
                qr[idx] = 0.0f;
                nr[idx] = 0.0f;
                continue;
            }
            const float qvk = fmaxf(1.0e-10f, qv[idx]);
            const float rho = 0.622f * pressure[idx]
                / (287.04f * temperature[idx] * (qvk + 0.622f));
            const float am_r = 3.1415926536f * 1000.0f / 6.0f;
            const float org3 = 1.0f / 6.0f;
            float nr_new = fmaxf(1.0e-6f / rho, nr[idx] + 0.0f * dt);
            const float lambda_arg = am_r * 6.0f * nr_new / qr_new;
            double lambda = (double)thompson_aa_powf(lambda_arg, 1.0f / 3.0f);
            float mvd = (float)(3.672 / lambda);
            if (mvd > 2.5e-3f) mvd = 2.5e-3f;
            else if (mvd < 37.5e-6f) mvd = 37.5e-6f;
            lambda = 3.672 / (double)mvd;
            const float prefix = __fdiv_rn(org3 * qr_new, am_r);
            nr_new = (float)((double)prefix * thompson_aa_pow(lambda, 3.0));
            qr[idx] = qr_new;
            nr[idx] = nr_new;
        }
        rainncv[column] += 0.0f;
        rainnc[column] += 0.0f;
        return;
    }

    float density[KMAX];
    float rain_mass[KMAX];
    float rain_number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX];
    float number_flux[KMAX];
    float qr_tendency[KMAX];
    float nr_tendency[KMAX];
    float qr_initial[KMAX];
    float nr_initial[KMAX];

    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float org3 = 1.0f / 6.0f;
    const float rho_not = __fdiv_rn(101325.0f, 287.05f * 298.0f);
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above_mass = 0.0f;
    float velocity_above_number = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        // The reference density carries L_qr (zero where it failed) and the
        // :3568 rewrite (negative), exactly as for the v4.6.1 pass.
        const float rain_density_carried = reference_density[idx];
        const float rain_density = fabsf(rain_density_carried);
        density[k] = rho;
        qr_initial[k] = qr[idx];
        nr_initial[k] = nr[idx];
        qr_tendency[k] = 0.0f;
        nr_tendency[k] = 0.0f;

        const bool l_qr = rain_density_carried != 0.0f;
        const bool rewritten = rain_density_carried < 0.0f;
        const float rr = !l_qr ? 1.0e-12f
            : rewritten ? fmaxf(1.0e-12f, qr[idx] * rain_density)
            : qr[idx] * rain_density;
        if (rr > 1.0e-12f) {
            const float nn = fmaxf(1.0e-6f, nr[idx] * rain_density);
            const float lambda_arg = am_r * 6.0f * nn / rr;
            const double lambda = (double)thompson_aa_powf(lambda_arg, 1.0f / 3.0f);
            rain_mass[k] = rr;
            rain_number[k] = nn;

            const float rhof = sqrtf(rho_not / rho);
            const float mass_prefix = rhof * 4854.0f * 24.0f * org3;
            mass_velocity[k] = (float)((double)mass_prefix
                * thompson_aa_pow(lambda, 4.0) * thompson_aa_pow(lambda + 195.0, -5.0));
            const float number_prefix = __fdiv_rn(rhof * 4854.0f
                * 3.3233511f, 1.3293403f);
            number_velocity[k] = (float)((double)number_prefix
                * thompson_aa_pow(lambda, 2.5) * thompson_aa_pow(lambda + 195.0, -3.5));
        } else {
            rain_mass[k] = rr;
            rain_number[k] = l_qr
                ? fmaxf(1.0e-6f, nr[idx] * rain_density) : 1.0e-6f;
            mass_velocity[k] = velocity_above_mass;
            number_velocity[k] = velocity_above_number;
        }
        velocity_above_mass = mass_velocity[k];
        velocity_above_number = number_velocity[k];

        const float vmax = fmaxf(mass_velocity[k], number_velocity[k]);
        if (vmax > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = dz[idx] / vmax;
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    if (sediment_top == nz - 1) sediment_top = nz - 2;
    nstep = max(nstep, 1);
    const float onstep = 1.0f / (float)nstep;
    const float dt_substep = dt * onstep;
    float exported = 0.0f;

    for (int step = 0; step < nstep; ++step) {
        for (int k = nz - 1; k >= 0; --k) {
            mass_flux[k] = mass_velocity[k] * rain_mass[k];
            number_flux[k] = number_velocity[k] * rain_number[k];
        }

        int k = nz - 1;
        size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density[k];
        qr_tendency[k] -= mass_flux[k] * inv_dz * onstep * inv_rho;
        nr_tendency[k] -= number_flux[k] * inv_dz * onstep * inv_rho;
        rain_mass[k] = fmaxf(1.0e-12f,
            rain_mass[k] - mass_flux[k] * inv_dz * dt_substep);
        rain_number[k] = fmaxf(1.0e-6f,
            rain_number[k] - number_flux[k] * inv_dz * dt_substep);

        for (k = sediment_top; k >= 0; --k) {
            idx = IDX3(k, j, i);
            inv_dz = 1.0f / dz[idx];
            inv_rho = 1.0f / density[k];
            const float mass_divergence = mass_flux[k + 1] - mass_flux[k];
            const float number_divergence =
                number_flux[k + 1] - number_flux[k];
            qr_tendency[k] += mass_divergence * inv_dz * onstep * inv_rho;
            nr_tendency[k] += number_divergence * inv_dz * onstep * inv_rho;
            rain_mass[k] = fmaxf(1.0e-12f,
                rain_mass[k] + mass_divergence * inv_dz * dt_substep);
            rain_number[k] = fmaxf(1.0e-6f,
                rain_number[k] + number_divergence * inv_dz * dt_substep);
        }
        // fork :3556, rr(kts) > R1*10.
        if (rain_mass[0] > 1.0e-11f) {
            exported += mass_flux[0] * dt_substep;
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        float qr_new = qr_initial[k] + qr_tendency[k] * dt;
        float nr_new = fmaxf(1.0e-6f / density[k],
                             nr_initial[k] + nr_tendency[k] * dt);
        if (qr_new <= 1.0e-12f) {
            qr[idx] = 0.0f;
            nr[idx] = 0.0f;
            continue;
        }
        const float lambda_arg = am_r * 6.0f * nr_new / qr_new;
        double lambda = (double)thompson_aa_powf(lambda_arg, 1.0f / 3.0f);
        float mvd = (float)(3.672 / lambda);
        if (mvd > 2.5e-3f) mvd = 2.5e-3f;
        else if (mvd < 37.5e-6f) mvd = 37.5e-6f;
        lambda = 3.672 / (double)mvd;
        const float prefix = __fdiv_rn(org3 * qr_new, am_r);
        nr_new = (float)((double)prefix * thompson_aa_pow(lambda, 3.0));
        qr[idx] = qr_new;
        nr[idx] = nr_new;
    }
    rainncv[column] += exported;
    rainnc[column] += exported;
}

#define THOMPSON_AA_WRF39_RAIN_SEDIMENT_PARAMETERS                       \
    float* __restrict__ qr, float* __restrict__ nr,                      \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float dt, int nz, int ny, int nx

#define THOMPSON_AA_WRF39_RAIN_SEDIMENT_ARGUMENTS                        \
    qr, nr, temperature, pressure, qv, reference_density, dz, rainnc,    \
    rainncv, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_wrf39_rain_sediment_64(
    THOMPSON_AA_WRF39_RAIN_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_rain_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_WRF39_RAIN_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_wrf39_rain_sediment_256(
    THOMPSON_AA_WRF39_RAIN_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_rain_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_WRF39_RAIN_SEDIMENT_ARGUMENTS);
}
#endif  // THOMPSON_AA_WRF39


#if defined(THOMPSON_AA_HAIL)
// ---------------------------------------------------------------------------
// THE HAIL-AWARE GRAUPEL FALLOUT (mp_physics=38).
// ---------------------------------------------------------------------------
//
// One thread per column, WRF's own sequence for the graupel category:
//
//   :3266-3282  the TAU+1 graupel state from the source tendencies, on the
//               :3193 density (`reference_density`): rg, the prognostic
//               ng and the clamped volume rb, and its density bin idx_bg;
//   :3373-3377  the slope recomputed from that state;
//   :3744-3781  the density-dependent fall speed.  The bulk density is
//               rg/rho/rb with rho(k) the density as the fallout finds it
//               (rewritten at :3490/:3572 where condensation and rain
//               evaporation ran, i.e. re-formed here from the current
//               temperature and vapour), Heymsfield's Best-number relation
//               afall = a*((4*rho_g*9.8)/(3*rho))**b * visco**(1-2b), and
//               rhof(k) = SQRT(RHO_NOT/rho) from the :3193 density unless
//               the column carries rain (:3614 re-forms it from the current
//               one); visco(k) is the :3201 value, re-formed at :3510 on
//               levels where rain evaporated (`visco_temperature`);
//   :3902-3939  the substepped flux divergence of mass, number and volume
//               into qgten/ngten/qbten, and the surface graupel.
//
// WRF v4.6.1 :3908 carries rb as a volume mixing ratio.
// Its volume flux and tendency have no air-density factor.
template <int KMAX>
__device__ __forceinline__ void thompson_aa_hail_graupel_sediment_impl(
    const float* __restrict__ qg1d,
    const float* __restrict__ ng1d,
    const float* __restrict__ qb1d,
    float* __restrict__ qgten,
    float* __restrict__ ngten,
    float* __restrict__ qbten,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ rain_active_columns,
    const float* __restrict__ visco_temperature,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ graupelnc,
    float* __restrict__ graupelncv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float density[KMAX];
    float rg[KMAX];
    float ng[KMAX];
    float rb[KMAX];
    float vtgk[KMAX];
    float vtngk[KMAX];
    float sed_g[KMAX];
    float sed_n[KMAX];
    float sed_b[KMAX];
    float qg_tendency[KMAX];
    float ng_tendency[KMAX];
    float qb_tendency[KMAX];

    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const bool rain_refreshes_rhof = rain_active_columns != nullptr
        && rain_active_columns[column] != 0.0f;
    // bfall = 3*b_coeff - 1 (:1605), REAL; 1.0-2.0*b_coeff (:3753).
    const float bfall = thompson_aa_sub(
        thompson_aa_mul(3.0f, THOMPSON_AA_HAIL_B_COEFF), 1.0f);
    const float visco_power = thompson_aa_sub(
        1.0f, thompson_aa_mul(2.0f, THOMPSON_AA_HAIL_B_COEFF));
    bool any_graupel = false;
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above = 0.0f;
    float number_velocity_above = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = thompson_aa_div(
            thompson_aa_mul(0.622f, pressure[idx]),
            thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_R_DRY,
                                            temperature[idx]),
                            thompson_aa_add(qvk, 0.622f)));
        density[k] = rho;
        qg_tendency[k] = qgten[idx];
        ng_tendency[k] = ngten[idx];
        qb_tendency[k] = qbten[idx];
        const float held = reference_density[idx];
        // :3266-3282.
        const float qg_working = thompson_aa_add(
            qg1d[idx], thompson_aa_mul(qgten[idx], dt));
        int idx_bg = THOMPSON_AA_HAIL_IDX_BG1;
        if (qg_working > THOMPSON_AA_R1) {
            any_graupel = true;
            rg[k] = thompson_aa_mul(qg_working, held);
            ng[k] = fmaxf(THOMPSON_AA_R2, thompson_aa_mul(thompson_aa_add(
                ng1d[idx], thompson_aa_mul(ngten[idx], dt)), held));
            const float qg_back = thompson_aa_div(rg[k], held);
            float volume = fmaxf(
                thompson_aa_div(qg_back, THOMPSON_AA_HAIL_RHO_G[9]),
                thompson_aa_add(qb1d[idx], thompson_aa_mul(qbten[idx], dt)));
            volume = fminf(thompson_aa_div(qg_back,
                                           THOMPSON_AA_HAIL_RHO_G[1]),
                           volume);
            rb[k] = volume;
            idx_bg = thompson_aa_hail_density_bin(thompson_aa_mul(
                thompson_aa_div(qg_back, volume), 0.01f));
        } else {
            rg[k] = THOMPSON_AA_R1;
            ng[k] = THOMPSON_AA_R2;
            rb[k] = thompson_aa_div(thompson_aa_div(THOMPSON_AA_R1, held),
                                    THOMPSON_AA_HAIL_RHO_G[9]);
        }
        // :3373-3377 and :3744-3771.
        if (rg[k] > THOMPSON_AA_R1) {
            const double lamg = thompson_aa_hail_lambda(ng[k], rg[k], idx_bg);
            const double ilamg = 1.0 / lamg;
            const float rhof = sqrtf(thompson_aa_div(
                rho_not, rain_refreshes_rhof ? rho : held));
            const float tv = visco_temperature[idx];
            const float tempc = thompson_aa_sub(tv, 273.15f);
            const float visco = tempc >= 0.0f
                ? thompson_aa_mul(
                      thompson_aa_add(1.718f, thompson_aa_mul(0.0049f, tempc)),
                      1.0e-5f)
                : thompson_aa_mul(
                      thompson_aa_sub(
                          thompson_aa_add(1.718f,
                                          thompson_aa_mul(0.0049f, tempc)),
                          thompson_aa_mul(thompson_aa_mul(1.2e-5f, tempc),
                                          tempc)),
                      1.0e-5f);
            const float xrho_g = fmaxf(THOMPSON_AA_HAIL_RHO_G[1], fminf(
                thompson_aa_div(thompson_aa_div(rg[k], rho), rb[k]),
                THOMPSON_AA_HAIL_RHO_G[9]));
            float afall = thompson_aa_mul(
                THOMPSON_AA_HAIL_A_COEFF,
                gfk_pow(thompson_aa_div(
                            thompson_aa_mul(thompson_aa_mul(4.0f, xrho_g),
                                            9.8f),
                            thompson_aa_mul(3.0f, rho)),
                        THOMPSON_AA_HAIL_B_COEFF));
            afall = thompson_aa_mul(afall, gfk_pow(visco, visco_power));
            const double fall = thompson_aa_pow(ilamg, (double)bfall);
            vtgk[k] = (float)((double)thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(rhof, afall),
                                THOMPSON_AA_HAIL_CGG[6]),
                THOMPSON_AA_HAIL_OGG3) * fall);
            // mu_g = 0: the "less size sorting" number speed, :3763.
            vtngk[k] = (float)((double)thompson_aa_div(
                thompson_aa_mul(thompson_aa_mul(rhof, afall),
                                THOMPSON_AA_HAIL_CGG[7]),
                THOMPSON_AA_HAIL_CGG[12]) * fall);
        } else {
            vtgk[k] = velocity_above;
            vtngk[k] = number_velocity_above;
        }
        velocity_above = vtgk[k];
        number_velocity_above = vtngk[k];
        if (vtgk[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = thompson_aa_div(dz[idx], vtgk[k]);
            nstep = max(nstep, (int)thompson_aa_add(
                thompson_aa_div(dt, delta_tp), 1.0f));
        }
    }

    float exported = 0.0f;
    if (any_graupel) {
        if (sediment_top == nz - 1) sediment_top = nz - 2;
        const float onstep = nstep > 0
            ? thompson_aa_div(1.0f, (float)nstep) : 1.0f;
        const int steps = (int)rintf(thompson_aa_div(1.0f, onstep));
        for (int step = 0; step < steps; ++step) {
            for (int k = nz - 1; k >= 0; --k) {
                sed_g[k] = thompson_aa_mul(vtgk[k], rg[k]);
                sed_n[k] = thompson_aa_mul(vtngk[k], ng[k]);
                sed_b[k] = thompson_aa_mul(vtgk[k], rb[k]);
            }
            int k = nz - 1;
            size_t idx = IDX3(k, j, i);
            float odzq = thompson_aa_div(1.0f, dz[idx]);
            float orho = thompson_aa_div(1.0f, density[k]);
            qg_tendency[k] = thompson_aa_sub(qg_tendency[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    sed_g[k], odzq), onstep), orho));
            ng_tendency[k] = thompson_aa_sub(ng_tendency[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    sed_n[k], odzq), onstep), orho));
            qb_tendency[k] = thompson_aa_sub(qb_tendency[k],
                thompson_aa_mul(thompson_aa_mul(sed_b[k], odzq), onstep));
            rg[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_sub(rg[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    sed_g[k], odzq), dt), onstep)));
            ng[k] = fmaxf(THOMPSON_AA_R2, thompson_aa_sub(ng[k],
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    sed_n[k], odzq), dt), onstep)));
            rb[k] = fmaxf(
                thompson_aa_div(thompson_aa_div(THOMPSON_AA_R1, density[k]),
                                THOMPSON_AA_HAIL_RHO_G[9]),
                thompson_aa_sub(rb[k], thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(sed_b[k], odzq), dt), onstep)));
            for (k = sediment_top; k >= 0; --k) {
                idx = IDX3(k, j, i);
                odzq = thompson_aa_div(1.0f, dz[idx]);
                orho = thompson_aa_div(1.0f, density[k]);
                const float dg = thompson_aa_sub(sed_g[k + 1], sed_g[k]);
                const float dn = thompson_aa_sub(sed_n[k + 1], sed_n[k]);
                const float db = thompson_aa_sub(sed_b[k + 1], sed_b[k]);
                qg_tendency[k] = thompson_aa_add(qg_tendency[k],
                    thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                        dg, odzq), onstep), orho));
                ng_tendency[k] = thompson_aa_add(ng_tendency[k],
                    thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                        dn, odzq), onstep), orho));
                qb_tendency[k] = thompson_aa_add(qb_tendency[k],
                    thompson_aa_mul(thompson_aa_mul(db, odzq), onstep));
                rg[k] = fmaxf(THOMPSON_AA_R1, thompson_aa_add(rg[k],
                    thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                        dg, odzq), dt), onstep)));
                ng[k] = fmaxf(THOMPSON_AA_R2, thompson_aa_add(ng[k],
                    thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                        dn, odzq), dt), onstep)));
                rb[k] = fmaxf(
                    thompson_aa_div(thompson_aa_div(rg[k], density[k]),
                                    THOMPSON_AA_HAIL_RHO_G[9]),
                    thompson_aa_add(rb[k], thompson_aa_mul(thompson_aa_mul(
                        thompson_aa_mul(db, odzq), dt), onstep)));
            }
            // :3936-3937.
            if (rg[0] > thompson_aa_mul(THOMPSON_AA_R1, 1000.0f)) {
                exported = thompson_aa_add(exported, thompson_aa_mul(
                    thompson_aa_mul(sed_g[0], dt), onstep));
            }
        }
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            qgten[idx] = qg_tendency[k];
            ngten[idx] = ng_tendency[k];
            qbten[idx] = qb_tendency[k];
        }
    }
    // The shared adapter sums all four precipitation exports once.
    graupelncv[column] = exported;
}

#define THOMPSON_AA_HAIL_GRAUPEL_SEDIMENT_PARAMETERS                      \
    const float* __restrict__ qg1d, const float* __restrict__ ng1d,      \
    const float* __restrict__ qb1d, float* __restrict__ qgten,           \
    float* __restrict__ ngten, float* __restrict__ qbten,                \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ rain_active_columns,                       \
    const float* __restrict__ visco_temperature,                         \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ graupelnc,          \
    float* __restrict__ graupelncv, float dt, int nz, int ny, int nx

#define THOMPSON_AA_HAIL_GRAUPEL_SEDIMENT_ARGUMENTS                       \
    qg1d, ng1d, qb1d, qgten, ngten, qbten, temperature, pressure, qv,    \
    reference_density, rain_active_columns, visco_temperature, dz,       \
    rainnc, rainncv, graupelnc, graupelncv, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_hail_graupel_sediment_64(
    THOMPSON_AA_HAIL_GRAUPEL_SEDIMENT_PARAMETERS)
{
    thompson_aa_hail_graupel_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_HAIL_GRAUPEL_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_hail_graupel_sediment_256(
    THOMPSON_AA_HAIL_GRAUPEL_SEDIMENT_PARAMETERS)
{
    thompson_aa_hail_graupel_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_HAIL_GRAUPEL_SEDIMENT_ARGUMENTS);
}
#endif  // THOMPSON_AA_HAIL
