// Classic WRF v4.6.1 Thompson (mp_physics=8) CUDA slices.
//
// Numerical authority for thompson_effective_radius is
// phys/module_mp_thompson.F:5594-5699 plus the mp_gt_driver output clamps at
// :1462-1474.  Classic mp=8 fixes cloud droplet concentration to Nt_c; this
// kernel is intentionally not shared with aerosol-aware option 28.

__device__ __constant__ float thompson_sa[10] = {
    5.065339f, -0.062659f, -3.032362f, 0.029469f, -0.000285f,
    0.31255f, 0.000204f, 0.003199f, 0.0f, -0.015952f
};

__device__ __constant__ float thompson_sb[10] = {
    0.476221f, -0.015896f, 0.165977f, 0.007468f, -0.000141f,
    0.060366f, 0.000079f, 0.000594f, 0.0f, -0.003577f
};

__device__ __forceinline__ float thompson_field_a(float tc, float moment) {
    const float tc2 = tc * tc;
    const float moment2 = moment * moment;
    const float loga = thompson_sa[0] + thompson_sa[1] * tc
        + thompson_sa[2] * moment + thompson_sa[3] * tc * moment
        + thompson_sa[4] * tc2 + thompson_sa[5] * moment2
        + thompson_sa[6] * tc2 * moment
        + thompson_sa[7] * tc * moment2
        + thompson_sa[8] * tc2 * tc
        + thompson_sa[9] * moment2 * moment;
    return powf(10.0f, loga);
}

__device__ __forceinline__ float thompson_field_b(float tc, float moment) {
    const float tc2 = tc * tc;
    const float moment2 = moment * moment;
    return thompson_sb[0] + thompson_sb[1] * tc
        + thompson_sb[2] * moment + thompson_sb[3] * tc * moment
        + thompson_sb[4] * tc2 + thompson_sb[5] * moment2
        + thompson_sb[6] * tc2 * moment
        + thompson_sb[7] * tc * moment2
        + thompson_sb[8] * tc2 * tc
        + thompson_sb[9] * moment2 * moment;
}

__device__ __forceinline__ float thompson_rslf(float pressure, float temp) {
    // module_mp_thompson.F:RSLF.  Preserve WRF's default-REAL Horner order.
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
    float esl = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4
        + x * (c5 + x * (c6 + x * (c7 + x * c8)))))));
    esl = fminf(esl, pressure * 0.15f);
    return 0.622f * esl / (pressure - esl);
}

__device__ __forceinline__ float thompson_rsif(float pressure, float temp) {
    // module_mp_thompson.F:RSIF.  Preserve WRF's default-REAL Horner order.
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
    float esi = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4
        + x * (c5 + x * (c6 + x * (c7 + x * c8)))))));
    esi = fminf(esi, pressure * 0.15f);
    return 0.622f * esi / fmaxf(1.0e-4f, pressure - esi);
}

// Bolton (1980) wet-bulb chain used verbatim by WRF v4.6.1 Thompson before
// it selects the warm/cold rain-snow and rain-graupel collision branches.
// Keep these helpers in default-REAL arithmetic: the lookup-table branch can
// flip at 273.15 K, so replacing this with a generic psychrometric estimate
// is not an admissible shortcut.
__device__ __forceinline__ float thompson_theta_e(
    float pressure, float temperature, float mixing_ratio, float tlcl)
{
    const float rr = mixing_ratio + 1.0e-8f;
    const float power = 0.2854f * (1.0f - 0.28f * rr);
    const float dry_theta = temperature * powf(100000.0f / pressure, power);
    const float p1 = 3.376f / tlcl - 0.00254f;
    const float p2 = rr * 1000.0f * (1.0f + 0.81f * rr);
    return dry_theta * expf(p1 * p2);
}

__device__ __forceinline__ float thompson_t_dew(
    float pressure, float mixing_ratio)
{
    const float rr = mixing_ratio + 1.0e-8f;
    const float esln = logf(pressure * rr / (0.622f + rr));
    return (35.86f * esln - 4947.2325f) / (esln - 23.6837f);
}

__device__ __forceinline__ float thompson_t_lcl(
    float temperature, float dewpoint)
{
    const float denominator = 1.0f / (dewpoint - 56.0f)
        + __fdiv_rn(logf(temperature / dewpoint), 800.0f);
    return 1.0f / denominator + 56.0f;
}

__device__ __forceinline__ float thompson_theta_wetb(float theta_e)
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

__device__ __forceinline__ float thompson_temperature_from_theta_e(
    float theta_e_lcl, float pressure)
{
    float guess = (theta_e_lcl - 0.5f
        * powf(fmaxf(theta_e_lcl - 270.0f, 0.0f), 1.05f))
        * powf(__fdiv_rn(pressure, 100000.0f), 0.2f);
    for (int iteration = 0; iteration < 100; ++iteration) {
        const float w1 = thompson_rslf(pressure, guess);
        const float w2 = thompson_rslf(pressure, guess + 1.0f);
        const float tenu = thompson_theta_e(
            pressure, guess, w1, guess);
        const float tenup = thompson_theta_e(
            pressure, guess + 1.0f, w2, guess + 1.0f);
        const float denominator = tenup - tenu;
        if (fabsf(denominator) < 1.0e-12f) break;
        const float correction = (theta_e_lcl - tenu) / denominator;
        guess += correction;
        if (fabsf(correction) < 0.01f) return guess;
    }
    return thompson_theta_wetb(theta_e_lcl)
        * powf(__fdiv_rn(pressure, 100000.0f), 0.286f);
}

__device__ __forceinline__ float thompson_wet_bulb_temperature(
    float pressure, float temperature, float mixing_ratio)
{
    if (mixing_ratio / thompson_rslf(pressure, temperature) >= 0.999f) {
        return temperature;
    }
    const float dewpoint = fminf(
        temperature - 0.001f, thompson_t_dew(pressure, mixing_ratio));
    const float tlcl = thompson_t_lcl(temperature, dewpoint);
    const float theta_e_lcl = thompson_theta_e(
        pressure, temperature, mixing_ratio, tlcl);
    return fminf(
        temperature,
        thompson_temperature_from_theta_e(theta_e_lcl, pressure));
}

extern "C" __global__ void thompson_warm_saturation_adjust(
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ qc,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float temp0 = temperature[idx];
    const float qv0 = qv[idx];
    const float qc0 = qc[idx];
    if (temp0 <= 273.15f) return;

    const float qvs = thompson_rslf(pressure[idx], temp0);
    float ssatw = qv0 / qvs - 1.0f;
    if (fabsf(ssatw) < 1.0e-15f) ssatw = 0.0f;
    if (!(ssatw > 1.0e-15f
            || (ssatw < -1.0e-15f && qc0 > 1.0e-12f))) return;

    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float rc = (qc0 > 1.0e-12f)
        ? qc0 * rho : 1.0e-12f;
    const float tempc = temp0 - 273.15f;
    const float lvap = 2.5e6f + (2106.0f - 4218.0f) * tempc;
    const float ocp = 1.0f / (1004.0f * (1.0f + 0.887f * qv0));
    const float inv_temp = 1.0f / temp0;
    const float lvt2 = lvap * lvap * ocp * (1.0f / 461.5f)
        * inv_temp * inv_temp;

    float clap = (qv0 - qvs) / (1.0f + lvt2 * qvs);
    for (int iteration = 0; iteration < 3; ++iteration) {
        const float exponential = expf(lvt2 * clap);
        const float fcd = qvs * exponential - qv0 + clap;
        const float dfcd = qvs * lvt2 * exponential + 1.0f;
        clap -= fcd / dfcd;
    }

    // This admitted slice has zero incoming tendencies.  WRF's dt factors
    // cancel exactly in the state update, leaving the diagnosed phase change.
    if (rc + clap * rho <= 1.0e-12f) {
        clap = -rc / rho;
    }
    qv[idx] = fmaxf(1.0e-10f, qv0 - clap);
    qc[idx] = fmaxf(0.0f, qc0 + clap);
    temperature[idx] = temp0 + lvap * ocp * clap;
}

__device__ __forceinline__ void thompson_cloud_saturation_adjust_impl(
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ qc,
    float* __restrict__ reference_density,
    float* __restrict__ reference_temperature,
    float* __restrict__ condensation_marker,
    int idx,
    float* __restrict__ cloud_presence = nullptr)
{
    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float qc0 = qc[idx];
    if (condensation_marker != nullptr) condensation_marker[idx] = 0.0f;
    // :3215-3223 and :3485.  WRF's L_qc(k) as the cloud fallout's
    // ANY(L_qc) gate reads it (:3645): set from the post-source cloud, and
    // cleared below where the adjustment leaves rc(k) at R1.  Never set by
    // the adjustment, so cloud that only condensed this step does not open
    // the gate.  Written at every level, as 1 or 0.
    if (cloud_presence != nullptr) {
        cloud_presence[idx] = qc0 > 1.0e-12f ? 1.0f : 0.0f;
    }
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    if (reference_density != nullptr) reference_density[idx] = rho;
    if (reference_temperature != nullptr) {
        reference_temperature[idx] = temp0;
    }

    const float qvs = thompson_rslf(pressure[idx], temp0);
    float ssatw = qv0 / qvs - 1.0f;
    if (fabsf(ssatw) < 1.0e-15f) ssatw = 0.0f;
    if (!(ssatw > 1.0e-15f
            || (ssatw < -1.0e-15f && qc0 > 1.0e-12f))) return;

    const float rc = qc0 > 1.0e-12f ? qc0 * rho : 1.0e-12f;
    const float tempc = temp0 - 273.15f;
    const float lvap = 2.5e6f + (2106.0f - 4218.0f) * tempc;
    const float ocp = 1.0f / (1004.0f * (1.0f + 0.887f * qv0));
    const float inverse_temp = 1.0f / temp0;
    const float lvt2 = lvap * lvap * ocp * (1.0f / 461.5f)
        * inverse_temp * inverse_temp;

    float clap = (qv0 - qvs) / (1.0f + lvt2 * qvs);
    for (int iteration = 0; iteration < 3; ++iteration) {
        const float exponential = expf(lvt2 * clap);
        const float fcd = qvs * exponential - qv0 + clap;
        const float dfcd = qvs * lvt2 * exponential + 1.0f;
        clap -= fcd / dfcd;
    }
    if (rc + clap * rho <= 1.0e-12f) clap = -rc / rho;

    if (condensation_marker != nullptr) {
        condensation_marker[idx] = clap > 0.0f ? 1.0f : 0.0f;
    }
    // :3479 and :3488.  The vapour WRF carries is qv1d + DT*qvten, floored
    // only for the working value each block reads (every reader here floors
    // it again) and once at the terminal apply (:3974, in a column with
    // microphysics: thompson_final_phase_cleanup_with_columns).
    qv[idx] = qv[idx] - clap;
    qc[idx] = fmaxf(0.0f, qc0 + clap);
    // :3484-3485.  rc(k) = MAX(R1, (qc1d + DT*qcten)*rho(k)) on the density
    // before :3490 refreshes it, and L_qc(k) goes false where that is R1.
    if (cloud_presence != nullptr && !(qc[idx] * rho > 1.0e-12f)) {
        cloud_presence[idx] = 0.0f;
    }
    temperature[idx] = temp0 + lvap * ocp * clap;
}

extern "C" __global__ void thompson_cloud_saturation_adjust(
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ qc,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    thompson_cloud_saturation_adjust_impl(
        temperature, pressure, qv, qc, nullptr, nullptr, nullptr, idx);
}

extern "C" __global__ void thompson_cloud_saturation_adjust_with_density(
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ qc,
    float* __restrict__ reference_density,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    thompson_cloud_saturation_adjust_impl(
        temperature, pressure, qv, qc, reference_density, nullptr, nullptr, idx);
}

extern "C" __global__ void thompson_cloud_saturation_adjust_with_state(
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ qc,
    float* __restrict__ reference_density,
    float* __restrict__ reference_temperature,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    thompson_cloud_saturation_adjust_impl(
        temperature, pressure, qv, qc,
        reference_density, reference_temperature, nullptr, idx);
}

extern "C" __global__ void thompson_cloud_saturation_adjust_with_history(
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ qc,
    float* __restrict__ reference_density,
    float* __restrict__ reference_temperature,
    float* __restrict__ condensation_marker,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    thompson_cloud_saturation_adjust_impl(
        temperature, pressure, qv, qc, reference_density,
        reference_temperature, condensation_marker, idx);
}

// The production entry point: the history outputs plus WRF's L_qc(k) as the
// adjustment leaves it (:3485), which the cloud fallout's column gate reads
// (:3645).
extern "C" __global__ void thompson_cloud_saturation_adjust_with_presence(
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ qc,
    float* __restrict__ reference_density,
    float* __restrict__ reference_temperature,
    float* __restrict__ condensation_marker,
    float* __restrict__ cloud_presence,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    thompson_cloud_saturation_adjust_impl(
        temperature, pressure, qv, qc, reference_density,
        reference_temperature, condensation_marker, idx, cloud_presence);
}

extern "C" __global__ void thompson_effective_radius(
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
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

#ifndef THOMPSON_NO_EXACT_SHORTCUTS
    // Exact shortcut: with no cloud, ice or snow mass (either signed zero)
    // every concentration below takes its 1e-12 floor, whatever the density,
    // so each radius is its lower clamp.  These are the full path's writes.
    if (qc[idx] == 0.0f && qi[idx] == 0.0f && qs[idx] == 0.0f) {
        effc[idx] = fmaxf(2.49e-6f, fminf(2.49e-6f, 50.0e-6f)) * 1.0e6f;
        effi[idx] = fmaxf(4.99e-6f, fminf(4.99e-6f, 125.0e-6f)) * 1.0e6f;
        effs[idx] = fmaxf(9.99e-6f, fminf(9.99e-6f, 999.0e-6f)) * 1.0e6f;
        return;
    }
#endif  // THOMPSON_NO_EXACT_SHORTCUTS

    // All constants are explicitly single precision because the corresponding
    // WRF declarations and pre-lambda expressions are default REAL.  Only the
    // two lambda variables and their final divisions are double precision.
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temperature[idx] * (qv[idx] + 0.622f));
    const float rc = fmaxf(1.0e-12f, qc[idx] * rho);
    const float ri = fmaxf(1.0e-12f, qi[idx] * rho);
    const float ice_number = fmaxf(1.0e-6f, ni[idx] * rho);
    const float rs = fmaxf(1.0e-12f, qs[idx] * rho);

    float reqc = 2.49e-6f;
    float reqi = 4.99e-6f;
    float reqs = 9.99e-6f;

    if (rc > 1.0e-12f) {
        const float am_r = 3.1415926536f * 1000.0f / 6.0f;
        // Nt_c=100e6 -> inu_c=NINT(1000e6/Nt_c)+2=12, whose
        // gamma-ratio entry is 2730.
        const float lambda_arg = 100.0e6f * am_r * 2730.0f / rc;
        const double lamc = (double)powf(lambda_arg, 1.0f / 3.0f);
        const float diagnosed = (float)(0.5 * 15.0 / lamc);
        reqc = fmaxf(2.51e-6f, fminf(diagnosed, 50.0e-6f));
    }

    if (ri > 1.0e-12f && ice_number > 1.0e-6f) {
        const float am_i = 3.1415926536f * 890.0f / 6.0f;
        // mu_i=0, bm_i=3: cig(2)=Gamma(4)=6, oig1=1.
        const float lambda_arg = am_i * 6.0f * ice_number / ri;
        const double lami = (double)powf(lambda_arg, 1.0f / 3.0f);
        const float diagnosed = (float)(0.5 * 3.0 / lami);
        reqi = fmaxf(2.51e-6f, fminf(diagnosed, 125.0e-6f));
    }

    if (rs > 1.0e-12f) {
        const float tc = fminf(-0.1f, temperature[idx] - 273.15f);
        // bm_s is exactly 2, so WRF's reference second moment is smob.
        const float smob = rs * (1.0f / 0.069f);
        const float moment = 3.0f;  // cse(1)=bm_s+1
        const float a = thompson_field_a(tc, moment);
        const float b = thompson_field_b(tc, moment);
        const float smoc = a * powf(smob, b);
        const float diagnosed = 0.5f * (smoc / smob);
        reqs = fmaxf(5.01e-6f, fminf(diagnosed, 999.0e-6f));
    }

    // mp_gt_driver's radiation-facing clamps retain the model constants
    // (metres, exactly as WRF stores re_cloud/re_ice/re_snow).  gpuwm's
    // state contract for effc/effi/effs is the radiation-facing MICRON
    // convention (state.py, thompson_contract.py), so the writer applies
    // WRF's own driver-side metre->micron conversion re*1.E6
    // (module_ra_rrtmg_lw.F:12184,12203,12242) after the clamps.  The
    // multiply is the last operation, so the stored microns are bitwise
    // fl(metres * 1e6) of the WRF metre values.
    effc[idx] = fmaxf(2.49e-6f, fminf(reqc, 50.0e-6f)) * 1.0e6f;
    effi[idx] = fmaxf(4.99e-6f, fminf(reqi, 125.0e-6f)) * 1.0e6f;
    effs[idx] = fmaxf(9.99e-6f, fminf(reqs, 999.0e-6f)) * 1.0e6f;
}

#define THOMPSON_KMAX_SHALLOW 64
#define THOMPSON_KMAX_GENERIC 256

// Exact shortcuts.  A column or cell whose result is already known
// (no species above R1, so nothing falls and every terminal write is a
// floor or a zero-tendency update) takes a short path that performs
// exactly the writes of the full path, signed zeros included.  The bit
// tests compile this unit with THOMPSON_NO_EXACT_SHORTCUTS defined as
// their reference, which removes every shortcut and nothing else.
#ifndef THOMPSON_NO_EXACT_SHORTCUTS
// The no-fallout gate is conservative: an unusual environment (a
// non-finite number, a density or depth out of range) takes the full sweep.
__device__ __forceinline__ bool thompson_empty_column(
    const float* q, const float* number, const float* temperature,
    const float* pressure, const float* qv, const float* dz,
    const float* presence, float dt, int column, int nz, int ny, int nx)
{
    if (!(dt >= 0.0f && dt <= 100000.0f)) return false;
    for (int k = 0; k < nz; ++k) {
        const size_t idx = (size_t)k * ny * nx + column;
        if (!(q[idx] >= -1.0f
                && q[idx] <= (presence != nullptr ? 1.0f : 1.0e-12f))) return false;
        if (presence != nullptr && presence[idx] != 0.0f) return false;
        if (number != nullptr && !isfinite(number[idx])) return false;
        if (!(temperature[idx] >= 100.0f && temperature[idx] <= 400.0f
                && pressure[idx] >= 1.0f && pressure[idx] <= 120000.0f
                && qv[idx] >= -1.0f && qv[idx] <= 1.0f
                && dz[idx] >= 1.0f && dz[idx] <= 100000.0f)) return false;
    }
    return true;
}
#endif  // THOMPSON_NO_EXACT_SHORTCUTS

// DENSITY_CARRIES_LQR: the caller's reference density is zero exactly where
// WRF's L_qr failed at :3236 (the mp=28 rain evaporation writes it so, see
// thompson_aerosol_sat.cu).  The plain entry points keep the state's mixing
// ratio as the stand-in for L_qr.
template <int KMAX, bool DENSITY_CARRIES_LQR = false>
__device__ __forceinline__ void thompson_rain_sediment_impl(
    float* __restrict__ qr,
    float* __restrict__ nr,
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

#ifndef THOMPSON_NO_EXACT_SHORTCUTS
    // Preserve the terminal writes, including zero-tendency signed-zero changes.
    if (thompson_empty_column(qr, nullptr, temperature, pressure, qv, dz,
            DENSITY_CARRIES_LQR ? reference_density : nullptr, dt, column, nz, ny, nx)) {
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float qr_new = qr[idx] + 0.0f * dt;
            if (qr_new <= 1.0e-12f) {
                qr[idx] = 0.0f;
                nr[idx] = 0.0f;
                continue;
            }
            // Absence of a fall speed does not suppress the terminal MVD bound.
            const float qvk = fmaxf(1.0e-10f, qv[idx]);
            const float rho = 0.622f * pressure[idx]
                / (287.04f * temperature[idx] * (qvk + 0.622f));
            const float am_r = 3.1415926536f * 1000.0f / 6.0f;
            const float org3 = 1.0f / 6.0f;
            float nr_new = fmaxf(1.0e-6f / rho, nr[idx] + 0.0f * dt);
            const float lambda_arg = am_r * 6.0f * nr_new / qr_new;
            double lambda = (double)powf(lambda_arg, 1.0f / 3.0f);
            float mvd = (float)(3.672 / lambda);
            if (mvd > 2.5e-3f) mvd = 2.5e-3f;
            else if (mvd < 37.5e-6f) mvd = 37.5e-6f;
            lambda = 3.672 / (double)mvd;
            const float prefix = __fdiv_rn(org3 * qr_new, am_r);
            nr_new = (float)((double)prefix * pow(lambda, 3.0));
            qr[idx] = qr_new;
            nr[idx] = nr_new;
        }
        if (accumulate_surface) rainncv[column] += 0.0f;
        else rainncv[column] = 0.0f;
        rainnc[column] += 0.0f;
        return;
    }
#endif  // THOMPSON_NO_EXACT_SHORTCUTS

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
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above_mass = 0.0f;
    float velocity_above_number = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        const float rain_density_carried = reference_density == nullptr
            ? rho : reference_density[idx];
        const float rain_density = fabsf(rain_density_carried);
        density[k] = rho;
        qr_initial[k] = qr[idx];
        nr_initial[k] = nr[idx];
        qr_tendency[k] = 0.0f;
        nr_tendency[k] = 0.0f;

        // WRF's rain-presence branch at module_mp_thompson.F:3616 tests the
        // working rain MASS CONCENTRATION, rr(k) > R1, not a mixing ratio.
        // rr(k) is formed as q*rho at :3237 only inside the TAU+1 test on the
        // mixing ratio at :3236 (L_qr), and is floored to R1 at :3252 when that
        // test fails; the post-evaporation rewrite at :3568 is itself inside
        // the L_qr(k) gate opened at :3501, and FLOORS rr(k) at R1 where
        // :3237 did not.  All three are essential.
        // L_qr and the rewrite are statements about the POST-SOURCE rain and
        // the evaporation gate, which this kernel no longer sees, so the rain
        // evaporation carries them in the reference density: zero where L_qr
        // failed, negative where :3568 rewrote the pair, positive otherwise.
        // Where L_qr held, rr(k) is the level's own q*rho WHATEVER its mixing
        // ratio is now: rain evaporation can take the mixing ratio to R1 or
        // below while q*rho stays above it (rho > 1), and WRF then gives that
        // level its own fall speed; where it took the mass away entirely, the
        // floor still sends R1.
        // Testing the post-evaporation mixing ratio instead sent such a level
        // the faster speed from above (5.4 times the number outflow at
        // wp08-freeze level 1).  Testing the mixing ratio of the state was a
        // stand-in for L_qr: it fired this branch on levels where WRF's rr(k)
        // sits at or below R1 because rho < 1 (13 634 level-visits of a 2 h
        // 12 km forecast), and dropping it without a real L_qr fired on
        // 56 839 level-visits WRF leaves alone.  The entry points that are
        // not told the density carries L_qr keep the stand-in.
        const bool l_qr = DENSITY_CARRIES_LQR
            ? rain_density_carried != 0.0f : qr[idx] > 1.0e-12f;
        const bool rewritten = DENSITY_CARRIES_LQR
            && rain_density_carried < 0.0f;
        const float rr = !l_qr ? 1.0e-12f
            : rewritten ? fmaxf(1.0e-12f, qr[idx] * rain_density)
            : qr[idx] * rain_density;
        if (rr > 1.0e-12f) {
            const float nn = fmaxf(1.0e-6f, nr[idx] * rain_density);
            const float lambda_arg = am_r * 6.0f * nn / rr;
            // WRF bounds the rain MVD to [0.75*D0r, 2.5 mm] exactly once,
            // in the TAU+1 block (module_mp_thompson.F:3240-3250), where it
            // also rewrites nr(k) so the distribution stays self-consistent.
            // thompson_bound_rain_number carries that single bound across
            // gpuwm's split source kernels.  WRF then rewrites the rain pair
            // after evaporation with no bound at all (:3568-3570) and derives
            // the fall speed from the pair as it stands (:3616-3627); mvd_r
            // does not appear anywhere between :3251 and :3632.  Re-applying
            // the bound here is therefore an extra WRF does not have: it is
            // idempotent wherever nothing has moved the pair since the last
            // bound call, but it bites on evaporation-narrowed drop spectra,
            // where it inflated the fall speed by up to 2.888x.
            const double lambda = (double)powf(lambda_arg, 1.0f / 3.0f);
            rain_mass[k] = rr;
            rain_number[k] = nn;

            const float rhof = sqrtf(rho_not / rho);
            const float mass_prefix = rhof * 4854.0f * 24.0f * org3;
            mass_velocity[k] = (float)((double)mass_prefix
                * pow(lambda, 4.0) * pow(lambda + 195.0, -5.0));
            const float number_prefix = __fdiv_rn(rhof * 4854.0f
                * 3.3233511f, 1.3293403f);
            number_velocity[k] = (float)((double)number_prefix
                * pow(lambda, 2.5) * pow(lambda + 195.0, -3.5));
        } else {
            // At or below R1 as a concentration the level takes the fall
            // speed from above (:3630-3631) but sediments its OWN working
            // pair: rr(k) as formed at :3237 (R1 where the mixing ratio
            // failed, :3252), and nr(k) = MAX(R2, nr*rho).  The number is not
            // the R2 sentinel when rain evaporation has just taken the mass:
            // pnr_rev stops at 99 percent of the number (:3559), L_qr stays
            // set from :3236, and :3570 rewrites nr(k) from the survivor,
            // which then falls into the level below.  Sending R2 instead
            // moved the rain reflectivity under evaporating shafts by up to
            // 3.9 dB against WRF v4.6.1 on saved real-data columns.  Where L_qr
            // failed at :3236 both codes send the R1/R2 sentinels.
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
        if (rain_mass[0] > 1.0e-9f) {
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
        double lambda = (double)powf(lambda_arg, 1.0f / 3.0f);
        float mvd = (float)(3.672 / lambda);
        if (mvd > 2.5e-3f) mvd = 2.5e-3f;
        else if (mvd < 37.5e-6f) mvd = 37.5e-6f;
        lambda = 3.672 / (double)mvd;
        const float prefix = __fdiv_rn(org3 * qr_new, am_r);
        nr_new = (float)((double)prefix * pow(lambda, 3.0));
        qr[idx] = qr_new;
        nr[idx] = nr_new;
    }
    if (accumulate_surface) {
        rainncv[column] += exported;
    } else {
        rainncv[column] = exported;
    }
    rainnc[column] += exported;
}

#define THOMPSON_RAIN_SEDIMENT_PARAMETERS                                \
    float* __restrict__ qr, float* __restrict__ nr,                      \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, int accumulate_surface, float dt,      \
    int nz, int ny, int nx

extern "C" __global__ void thompson_rain_sediment_64(
    THOMPSON_RAIN_SEDIMENT_PARAMETERS)
{
    thompson_rain_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        qr, nr, temperature, pressure, qv, nullptr, dz, rainnc, rainncv,
        accumulate_surface, dt, nz, ny, nx);
}

extern "C" __global__ void thompson_rain_sediment_256(
    THOMPSON_RAIN_SEDIMENT_PARAMETERS)
{
    thompson_rain_sediment_impl<THOMPSON_KMAX_GENERIC>(
        qr, nr, temperature, pressure, qv, nullptr, dz, rainnc, rainncv,
        accumulate_surface, dt, nz, ny, nx);
}

#define THOMPSON_RAIN_SEDIMENT_DENSITY_PARAMETERS                        \
    float* __restrict__ qr, float* __restrict__ nr,                      \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, int accumulate_surface, float dt,      \
    int nz, int ny, int nx

extern "C" __global__ void thompson_rain_sediment_64_with_density(
    THOMPSON_RAIN_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_rain_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        qr, nr, temperature, pressure, qv, reference_density, dz,
        rainnc, rainncv, accumulate_surface, dt, nz, ny, nx);
}

extern "C" __global__ void thompson_rain_sediment_256_with_density(
    THOMPSON_RAIN_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_rain_sediment_impl<THOMPSON_KMAX_GENERIC>(
        qr, nr, temperature, pressure, qv, reference_density, dz,
        rainnc, rainncv, accumulate_surface, dt, nz, ny, nx);
}

// The reference density also carries WRF's L_qr (zero where it failed).
extern "C" __global__ void thompson_rain_sediment_64_with_presence(
    THOMPSON_RAIN_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_rain_sediment_impl<THOMPSON_KMAX_SHALLOW, true>(
        qr, nr, temperature, pressure, qv, reference_density, dz,
        rainnc, rainncv, accumulate_surface, dt, nz, ny, nx);
}

extern "C" __global__ void thompson_rain_sediment_256_with_presence(
    THOMPSON_RAIN_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_rain_sediment_impl<THOMPSON_KMAX_GENERIC, true>(
        qr, nr, temperature, pressure, qv, reference_density, dz,
        rainnc, rainncv, accumulate_surface, dt, nz, ny, nx);
}

template <int KMAX>
__device__ __forceinline__ void thompson_ice_sediment_impl(
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

#ifndef THOMPSON_NO_EXACT_SHORTCUTS
    // Preserve the terminal writes, including zero-tendency signed-zero changes.
    if (thompson_empty_column(qi, ni, temperature, pressure, qv, dz,
            nullptr, dt, column, nz, ny, nx)) {
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            qi[idx] = qi[idx] + 0.0f * dt;
            ni[idx] = ni[idx] + 0.0f * dt;
        }
        rainncv[column] = 0.0f;
        snowncv[column] = 0.0f;
        rainnc[column] += 0.0f;
        snownc[column] += 0.0f;
        return;
    }
#endif  // THOMPSON_NO_EXACT_SHORTCUTS

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

    const float pi = 3.1415926536f;
    const float am_i = pi * 890.0f / 6.0f;
    const float oig2 = 1.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
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

        // :3668, `if (ri(k).gt. R1)`: WRF opens the fall-speed branch on the
        // working ice CONCENTRATION.  Below it the level takes the speed from
        // above but keeps its own ri/ni, formed at :3230-3232 whenever the
        // MIXING RATIO passed.  Opening on the mixing ratio alone gave a fall
        // speed to ice WRF moves at the speed from above, and the sentinels
        // replaced mass the level really held: ice number, effective radius
        // and up to 13 dB of reflectivity differed from WRF v4.6.1 on saved
        // real-data columns.
        const float ice_state_rho = reference_density == nullptr
            ? rho : reference_density[idx];
        if (qi[idx] > 1.0e-12f && qi[idx] * ice_state_rho > 1.0e-12f) {
            // Ice mass/number were formed with the pre-evaporation density,
            // while WRF uses the updated environment for the fallspeed and
            // conversion back to mixing-ratio tendencies.
            const float state_rho = ice_state_rho;
            const float ri = qi[idx] * state_rho;
            float nn = fmaxf(1.0e-6f, ni[idx] * state_rho);
            if (nn <= 1.0e-6f) {
                const double lambda = 4.0 / 5.0e-6;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = fminf(999.0e3f,
                           (float)((double)prefix * pow(lambda, 3.0)));
            }
            const float lambda_arg = am_i * 6.0f * nn / ri;
            double lambda = (double)powf(lambda_arg, 1.0f / 3.0f);
            float diameter = (float)(4.0 / lambda);
            if (diameter < 5.0e-6f) {
                diameter = 5.0e-6f;
                lambda = 4.0 / (double)diameter;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = fminf(999.0e3f,
                           (float)((double)prefix * pow(lambda, 3.0)));
            } else if (diameter > 300.0e-6f) {
                diameter = 300.0e-6f;
                lambda = 4.0 / (double)diameter;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = (float)((double)prefix * pow(lambda, 3.0));
            }
            ice_mass[k] = ri;
            ice_number[k] = nn;

            const float rhof = sqrtf(rho_not / rho);
            const double inverse_lambda = 1.0 / lambda;
            const float mass_prefix = rhof * 1493.9f * 24.0f * oig2;
            mass_velocity[k] = (float)((double)mass_prefix * inverse_lambda);
            const float number_prefix = __fdiv_rn(rhof * 1493.9f
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
        if (ice_mass[0] > 1.0e-9f) {
            exported += mass_flux[0] * dt_substep;
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qi_new = qi_initial[k] + qi_tendency[k] * dt;
        float ni_new = fmaxf(1.0e-6f / density[k],
                             ni_initial[k] + ni_tendency[k] * dt);
        if (qi_new <= 1.0e-12f) {
            // :3943-3953 melts ANY positive ice above 0 C into cloud water,
            // and only the terminal apply after it (:4023-4027) removes ice
            // at or below R1.  So ice at or below R1 is carried to the phase
            // cleanup, which melts it or removes it there.  Removing it here
            // took it out of the melt: cloud that WRF keeps at 1.6e-12 to
            // 2.0e-12 kg/kg, with its droplets, came back as zero on saved
            // real-data columns (tools/thompson_real_column_parity).  The
            // number goes with it unfloored, ni1d + niten*DT: WRF's R2/rho
            // floor (:4022) belongs to ice the terminal apply keeps, and a
            // level that held no ice keeps no number.
            qi[idx] = qi_new;
            ni[idx] = ni_initial[k] + ni_tendency[k] * dt;
            continue;
        }
        const float lambda_arg = am_i * 6.0f * ni_new / qi_new;
        double lambda = (double)powf(lambda_arg, 1.0f / 3.0f);
        float diameter = (float)(4.0 / lambda);
        if (diameter < 5.0e-6f) diameter = 5.0e-6f;
        else if (diameter > 300.0e-6f) diameter = 300.0e-6f;
        lambda = 4.0 / (double)diameter;
        const float prefix = __fdiv_rn(oig2 * qi_new, am_i);
        ni_new = fminf((float)((double)prefix * pow(lambda, 3.0)),
                       999.0e3f / density[k]);
        qi[idx] = qi_new;
        ni[idx] = ni_new;
    }
    rainncv[column] = exported;
    snowncv[column] = exported;
    rainnc[column] += exported;
    snownc[column] += exported;
}

#define THOMPSON_ICE_SEDIMENT_PARAMETERS                                 \
    float* __restrict__ qi, float* __restrict__ ni,                      \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ snownc,             \
    float* __restrict__ snowncv, float dt, int nz, int ny, int nx

#define THOMPSON_ICE_SEDIMENT_ARGUMENTS                                  \
    qi, ni, temperature, pressure, qv, nullptr, dz, rainnc, rainncv,     \
    snownc,                                                               \
    snowncv, dt, nz, ny, nx

extern "C" __global__ void thompson_ice_sediment_64(
    THOMPSON_ICE_SEDIMENT_PARAMETERS)
{
    thompson_ice_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_ICE_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_ice_sediment_256(
    THOMPSON_ICE_SEDIMENT_PARAMETERS)
{
    thompson_ice_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_ICE_SEDIMENT_ARGUMENTS);
}

#define THOMPSON_ICE_SEDIMENT_DENSITY_PARAMETERS                         \
    float* __restrict__ qi, float* __restrict__ ni,                      \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ snownc,             \
    float* __restrict__ snowncv, float dt, int nz, int ny, int nx

#define THOMPSON_ICE_SEDIMENT_DENSITY_ARGUMENTS                          \
    qi, ni, temperature, pressure, qv, reference_density, dz, rainnc,    \
    rainncv, snownc, snowncv, dt, nz, ny, nx

extern "C" __global__ void thompson_ice_sediment_64_with_density(
    THOMPSON_ICE_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_ice_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_ICE_SEDIMENT_DENSITY_ARGUMENTS);
}

extern "C" __global__ void thompson_ice_sediment_256_with_density(
    THOMPSON_ICE_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_ice_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_ICE_SEDIMENT_DENSITY_ARGUMENTS);
}

template <int KMAX>
__device__ __forceinline__ void thompson_cloud_sediment_impl(
    float* __restrict__ qc,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ vertical_velocity,
    const float* __restrict__ dz,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

#ifndef THOMPSON_NO_EXACT_SHORTCUTS
    // Preserve the terminal writes, including zero-tendency signed-zero changes.
    if (thompson_empty_column(qc, nullptr, temperature, pressure, qv, dz,
            nullptr, dt, column, nz, ny, nx)) {
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            qc[idx] = 0.0f;
        }
        return;
    }
#endif  // THOMPSON_NO_EXACT_SHORTCUTS

    float density[KMAX];
    float cloud_mass[KMAX];
    float mass_velocity[KMAX];
    float mass_flux[KMAX];
    float qc_tendency[KMAX];
    float qc_initial[KMAX];
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        density[k] = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        qc_initial[k] = qc[idx];
        qc_tendency[k] = 0.0f;
        cloud_mass[k] = qc[idx] > 1.0e-12f
            ? qc[idx] * density[k] : 1.0e-12f;
        mass_velocity[k] = 0.0f;
    }

    int sediment_top = 0;
    float height_agl = 0.0f;
    for (int k = 0; k < nz - 1; ++k) {
        const size_t idx = IDX3(k, j, i);
        if (cloud_mass[k] > 1.0e-6f) sediment_top = k;
        height_agl += dz[idx];
        if (height_agl > 500.0f) break;
    }

    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        if (cloud_mass[k] > 1.0e-12f
                && vertical_velocity[idx] < 1.0e-1f) {
            const float lambda_arg = 100.0e6f * am_r * 2730.0f
                / cloud_mass[k];
            const double lambda =
                (double)powf(lambda_arg, 1.0f / 3.0f);
            const double inverse_lambda = 1.0 / lambda;
            const float prefix = sqrtf(rho_not / density[k])
                * 0.316946e8f * 272.0f;
            mass_velocity[k] = (float)((double)prefix
                * inverse_lambda * inverse_lambda);
        }
    }
    for (int k = nz - 1; k >= 0; --k) {
        mass_flux[k] = mass_velocity[k] * cloud_mass[k];
    }
    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float divergence = mass_flux[k + 1] - mass_flux[k];
        const float inv_dz = 1.0f / dz[idx];
        qc_tendency[k] += divergence * inv_dz / density[k];
        cloud_mass[k] = fmaxf(1.0e-12f,
            cloud_mass[k] + divergence * inv_dz * dt);
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qc_new = qc_initial[k] + qc_tendency[k] * dt;
        qc[idx] = qc_new <= 1.0e-12f ? 0.0f : qc_new;
    }
}

#define THOMPSON_CLOUD_SEDIMENT_PARAMETERS                               \
    float* __restrict__ qc, const float* __restrict__ temperature,      \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ vertical_velocity,                        \
    const float* __restrict__ dz, float dt, int nz, int ny, int nx

#define THOMPSON_CLOUD_SEDIMENT_ARGUMENTS                                \
    qc, temperature, pressure, qv, vertical_velocity, dz,                \
    dt, nz, ny, nx

extern "C" __global__ void thompson_cloud_sediment_64(
    THOMPSON_CLOUD_SEDIMENT_PARAMETERS)
{
    thompson_cloud_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_CLOUD_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_cloud_sediment_256(
    THOMPSON_CLOUD_SEDIMENT_PARAMETERS)
{
    thompson_cloud_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_CLOUD_SEDIMENT_ARGUMENTS);
}

// Coupled classic-Thompson cloud fallout needs two distinct densities.  WRF
// forms rc from the density immediately before cloud saturation adjustment,
// then converts sediment-flux divergence back to mixing-ratio tendency with
// the post-adjustment density.  Its rhof fall-speed factor also stays on the
// held density unless a post-source rain column causes the rain fall-speed
// pass to refresh rhof first.  Keep this implementation separate from the
// already admitted standalone kernel above so its generated code is frozen.
template <int KMAX>
__device__ __forceinline__ void thompson_cloud_sediment_held_density_impl(
    float* __restrict__ qc,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ rain_active_columns,
    const float* __restrict__ cloud_active_columns,
    const float* __restrict__ vertical_velocity,
    const float* __restrict__ dz,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    if (cloud_active_columns != nullptr
            && cloud_active_columns[column] == 0.0f) return;
    const int j = column / nx;
    const int i = column - j * nx;

#ifndef THOMPSON_NO_EXACT_SHORTCUTS
    // Preserve the terminal writes, including zero-tendency signed-zero changes.
    if (thompson_empty_column(qc, nullptr, temperature, pressure, qv, dz,
            nullptr, dt, column, nz, ny, nx)) {
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            qc[idx] = qc[idx] + 0.0f * dt;
        }
        return;
    }
#endif  // THOMPSON_NO_EXACT_SHORTCUTS

    float density[KMAX];
    float cloud_mass[KMAX];
    float mass_velocity[KMAX];
    float mass_flux[KMAX];
    float qc_tendency[KMAX];
    float qc_initial[KMAX];
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const bool rain_refreshes_rhof = rain_active_columns != nullptr
        && rain_active_columns[column] != 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        density[k] = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        qc_initial[k] = qc[idx];
        qc_tendency[k] = 0.0f;
        cloud_mass[k] = qc[idx] > 1.0e-12f
            ? qc[idx] * reference_density[idx] : 1.0e-12f;
        mass_velocity[k] = 0.0f;
    }

    int sediment_top = 0;
    float height_agl = 0.0f;
    for (int k = 0; k < nz - 1; ++k) {
        const size_t idx = IDX3(k, j, i);
        if (cloud_mass[k] > 1.0e-6f) sediment_top = k;
        height_agl += dz[idx];
        if (height_agl > 500.0f) break;
    }

    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        if (cloud_mass[k] > 1.0e-12f
                && vertical_velocity[idx] < 1.0e-1f) {
            const float lambda_arg = 100.0e6f * am_r * 2730.0f
                / cloud_mass[k];
            const double lambda =
                (double)powf(lambda_arg, 1.0f / 3.0f);
            const double inverse_lambda = 1.0 / lambda;
            const float velocity_density = rain_refreshes_rhof
                ? density[k] : reference_density[idx];
            const float prefix = sqrtf(rho_not / velocity_density)
                * 0.316946e8f * 272.0f;
            mass_velocity[k] = (float)((double)prefix
                * inverse_lambda * inverse_lambda);
        }
    }
    for (int k = nz - 1; k >= 0; --k) {
        mass_flux[k] = mass_velocity[k] * cloud_mass[k];
    }
    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float divergence = mass_flux[k + 1] - mass_flux[k];
        const float inv_dz = 1.0f / dz[idx];
        qc_tendency[k] += divergence * inv_dz / density[k];
        cloud_mass[k] = fmaxf(1.0e-12f,
            cloud_mass[k] + divergence * inv_dz * dt);
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        // Cloud at or below R1 is carried to the phase cleanup: WRF freezes
        // any positive cloud below HGFR and adds melted ice to it above 0 C
        // (:3943-3966) before the terminal apply removes what is left at or
        // below R1 (:4007-4009).  thompson_final_phase_cleanup removes it.
        qc[idx] = qc_initial[k] + qc_tendency[k] * dt;
    }
}

#define THOMPSON_CLOUD_SEDIMENT_DENSITY_PARAMETERS                       \
    float* __restrict__ qc, const float* __restrict__ temperature,       \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ vertical_velocity,                         \
    const float* __restrict__ dz, float dt, int nz, int ny, int nx

#define THOMPSON_CLOUD_SEDIMENT_DENSITY_ARGUMENTS                        \
    qc, temperature, pressure, qv, reference_density,                    \
    (const float*)0, (const float*)0, vertical_velocity, dz,              \
    dt, nz, ny, nx

extern "C" __global__ void thompson_cloud_sediment_64_with_density(
    THOMPSON_CLOUD_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_cloud_sediment_held_density_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_CLOUD_SEDIMENT_DENSITY_ARGUMENTS);
}

extern "C" __global__ void thompson_cloud_sediment_256_with_density(
    THOMPSON_CLOUD_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_cloud_sediment_held_density_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_CLOUD_SEDIMENT_DENSITY_ARGUMENTS);
}

#define THOMPSON_CLOUD_SEDIMENT_DENSITY_RAIN_PARAMETERS                  \
    float* __restrict__ qc, const float* __restrict__ temperature,       \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ rain_active_columns,                       \
    const float* __restrict__ vertical_velocity,                         \
    const float* __restrict__ dz, float dt, int nz, int ny, int nx

#define THOMPSON_CLOUD_SEDIMENT_DENSITY_RAIN_ARGUMENTS                   \
    qc, temperature, pressure, qv, reference_density,                    \
    rain_active_columns, (const float*)0, vertical_velocity, dz,          \
    dt, nz, ny, nx

extern "C" __global__ void
thompson_cloud_sediment_64_with_density_and_rain(
    THOMPSON_CLOUD_SEDIMENT_DENSITY_RAIN_PARAMETERS)
{
    thompson_cloud_sediment_held_density_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_CLOUD_SEDIMENT_DENSITY_RAIN_ARGUMENTS);
}

extern "C" __global__ void
thompson_cloud_sediment_256_with_density_and_rain(
    THOMPSON_CLOUD_SEDIMENT_DENSITY_RAIN_PARAMETERS)
{
    thompson_cloud_sediment_held_density_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_CLOUD_SEDIMENT_DENSITY_RAIN_ARGUMENTS);
}

#define THOMPSON_CLOUD_SEDIMENT_DENSITY_MASKS_PARAMETERS                 \
    float* __restrict__ qc, const float* __restrict__ temperature,       \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ rain_active_columns,                       \
    const float* __restrict__ cloud_active_columns,                      \
    const float* __restrict__ vertical_velocity,                         \
    const float* __restrict__ dz, float dt, int nz, int ny, int nx

#define THOMPSON_CLOUD_SEDIMENT_DENSITY_MASKS_ARGUMENTS                  \
    qc, temperature, pressure, qv, reference_density,                    \
    rain_active_columns, cloud_active_columns, vertical_velocity, dz,    \
    dt, nz, ny, nx

extern "C" __global__ void
thompson_cloud_sediment_64_with_density_and_masks(
    THOMPSON_CLOUD_SEDIMENT_DENSITY_MASKS_PARAMETERS)
{
    thompson_cloud_sediment_held_density_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_CLOUD_SEDIMENT_DENSITY_MASKS_ARGUMENTS);
}

extern "C" __global__ void
thompson_cloud_sediment_256_with_density_and_masks(
    THOMPSON_CLOUD_SEDIMENT_DENSITY_MASKS_PARAMETERS)
{
    thompson_cloud_sediment_held_density_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_CLOUD_SEDIMENT_DENSITY_MASKS_ARGUMENTS);
}

// melt_rain_density is the rain fallout's own reference density (the rain
// evaporation writes it), from which the melting-snow blend forms WRF's rr(k)
// and vtrk(k) exactly as thompson_rain_sediment_impl does; null keeps the
// snow's reference density as the stand-in.  RAIN_PRESENCE says that density
// carries L_qr and the :3568 rewrite (DENSITY_CARRIES_LQR there).
template <int KMAX, bool RAIN_PRESENCE = false>
__device__ __forceinline__ void thompson_snow_sediment_impl(
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
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ snownc,
    float* __restrict__ snowncv,
    int accumulate_surface, float dt, int nz, int ny, int nx,
    const float* __restrict__ melt_rain_density = nullptr)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

#ifndef THOMPSON_NO_EXACT_SHORTCUTS
    // Preserve the terminal writes, including zero-tendency signed-zero changes.
    if (thompson_empty_column(qs, nullptr, temperature, pressure, qv, dz,
            nullptr, dt, column, nz, ny, nx)) {
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            qs[idx] = 0.0f;
        }
        if (accumulate_surface) rainncv[column] += 0.0f;
        else rainncv[column] = 0.0f;
        if (accumulate_surface) snowncv[column] += 0.0f;
        else snowncv[column] = 0.0f;
        rainnc[column] += 0.0f;
        snownc[column] += 0.0f;
        return;
    }
#endif  // THOMPSON_NO_EXACT_SHORTCUTS

    float density[KMAX];
    float snow_mass[KMAX];
    float mass_velocity[KMAX];
    float mass_flux[KMAX];
    float qs_tendency[KMAX];
    float qs_initial[KMAX];

    const float rho_not = 101325.0f / (287.05f * 298.0f);
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

        // WRF's rain fall-speed pass (:3612-3634) as the melting-snow blend
        // reads it: rr(k) as the rain fallout forms it, and vtrk(k), which a
        // level whose rr(k) is at or below R1 inherits from the level above
        // (:3630) and which is zero in a column where no level has L_qr (the
        // pass never runs, :3597-3607).  Same arithmetic as
        // thompson_rain_sediment_impl's mass velocity.
        float rain_rr = 1.0e-12f;
        float rain_velocity = 0.0f;
        if (melt_rain_qr != (const float*)0) {
            const float rain_density_carried = melt_rain_density == nullptr
                ? (reference_density == nullptr ? rho : reference_density[idx])
                : melt_rain_density[idx];
            const float rain_density = fabsf(rain_density_carried);
            const bool l_qr = RAIN_PRESENCE
                ? rain_density_carried != 0.0f : melt_rain_qr[idx] > 1.0e-12f;
            const bool rewritten = RAIN_PRESENCE
                && rain_density_carried < 0.0f;
            rain_rr = !l_qr ? 1.0e-12f
                : rewritten ? fmaxf(1.0e-12f, melt_rain_qr[idx] * rain_density)
                : melt_rain_qr[idx] * rain_density;
            if (rain_rr > 1.0e-12f) {
                const float am_r = 3.1415926536f * 1000.0f / 6.0f;
                const float rain_number = fmaxf(
                    1.0e-6f, melt_rain_nr[idx] * rain_density);
                const double rain_lambda = (double)powf(
                    am_r * 6.0f * rain_number / rain_rr, 1.0f / 3.0f);
                const float rain_rhof = sqrtf(rho_not / rho);
                rain_velocity = (float)(
                    (double)(rain_rhof * 4854.0f * 24.0f * (1.0f / 6.0f))
                    * pow(rain_lambda, 4.0)
                    * pow(rain_lambda + 195.0, -5.0));
            } else {
                rain_velocity = rain_velocity_above;
            }
            rain_velocity_above = rain_velocity;
        }

        // :3699, `if (rs(k).gt. R1)`: the same concentration gate as ice.
        // Below it the level takes the speed from above and sediments its
        // own rs(k), formed at :3257 whenever the mixing ratio passed.
        const float snow_state_rho = reference_density == nullptr
            ? rho : reference_density[idx];
        if (qs[idx] > 1.0e-12f && qs[idx] * snow_state_rho > 1.0e-12f) {
            const float state_rho = snow_state_rho;
            const float rs = qs[idx] * state_rho;
            const float smob = rs * (1.0f / 0.069f);
            const float state_temperature = reference_temperature == nullptr
                ? temperature[idx] : reference_temperature[idx];
            const float tc0 = fminf(-0.1f, state_temperature - 273.15f);
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
            const float smoc = powf(10.0f, loga) * powf(smob, exponent);
            const float mean_ratio = smob / smoc;
            float ils1 = 1.0f / (mean_ratio * 20.78f + 100.0f);
            float ils2 = 1.0f / (mean_ratio * 3.29f + 100.0f);
            const float ratio_power = powf(mean_ratio, 0.6357f);
            const float numerator1 = 490.6f * 3.51325202f
                * powf(ils1, 3.55f);
            const float numerator2 = 17.46f * ratio_power * 7.61279917f
                * powf(ils2, 4.1857f);
            ils1 = 1.0f / (mean_ratio * 20.78f);
            ils2 = 1.0f / (mean_ratio * 3.29f);
            const float denominator1 = 490.6f * 2.0f
                * powf(ils1, 3.0f);
            const float denominator2 = 17.46f * ratio_power * 3.87160635f
                * powf(ils2, 3.6357f);
            const float rhof = sqrtf(rho_not / rho);
            float snow_velocity = rhof * 40.0f
                * (numerator1 + numerator2)
                / (denominator1 + denominator2);
            if (velocity_boost != (const float*)0) {
                snow_velocity *= velocity_boost[idx];
            }
            // :3722-3724: melting snow (prr_sml > 0) falls at its speed
            // blended with vtrk(k) by SR = rs/(rs+rr), whatever the level's
            // own rain: the rain speed a level without rain inherits from
            // above, R1 for its rr.  Blending only where the level itself
            // held rain gave melting snow its unblended speed there and moved
            // snow by up to 2.3e-9 kg/kg against WRF v4.6.1 on saved
            // real-data columns (tools/thompson_real_column_parity).
            if (snow_melt_marker != (const float*)0
                    && snow_melt_marker[idx] != 0.0f
                    && melt_rain_qr != (const float*)0) {
                const float solid_fraction = rs / (rs + rain_rr);
                snow_velocity = snow_velocity * solid_fraction
                    + rain_velocity * (1.0f - solid_fraction);
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
        if (snow_mass[0] > 1.0e-9f) {
            exported += mass_flux[0] * dt_substep;
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qs_new = qs_initial[k] + qs_tendency[k] * dt;
        qs[idx] = qs_new <= 1.0e-12f ? 0.0f : qs_new;
    }
    if (accumulate_surface) {
        rainncv[column] += exported;
        snowncv[column] += exported;
    } else {
        rainncv[column] = exported;
        snowncv[column] = exported;
    }
    rainnc[column] += exported;
    snownc[column] += exported;
}

#define THOMPSON_SNOW_SEDIMENT_PARAMETERS                                \
    float* __restrict__ qs, const float* __restrict__ temperature,       \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ snownc,             \
    float* __restrict__ snowncv, int accumulate_surface, float dt,       \
    int nz, int ny, int nx

#define THOMPSON_SNOW_SEDIMENT_ARGUMENTS                                  \
    qs, (const float*)0, (const float*)0, (const float*)0,                \
    temperature, pressure, qv,                                            \
    (const float*)0, (const float*)0, (const float*)0, dz, rainnc,         \
    rainncv, snownc,                                                       \
    snowncv, accumulate_surface, dt, nz, ny, nx

#define THOMPSON_SNOW_SEDIMENT_MELT_PARAMETERS                           \
    float* __restrict__ qs,                                               \
    const float* __restrict__ snow_melt_marker,                           \
    const float* __restrict__ melt_rain_qr,                               \
    const float* __restrict__ melt_rain_nr,                              \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ snownc,             \
    float* __restrict__ snowncv, int accumulate_surface, float dt,       \
    int nz, int ny, int nx

#define THOMPSON_SNOW_SEDIMENT_MELT_ARGUMENTS                            \
    qs, snow_melt_marker, melt_rain_qr, melt_rain_nr,                    \
    temperature, pressure, qv,                                           \
    (const float*)0, (const float*)0, (const float*)0, dz, rainnc,        \
    rainncv, snownc,                                                      \
    snowncv,                                                              \
    accumulate_surface, dt, nz, ny, nx

extern "C" __global__ void thompson_snow_sediment_64(
    THOMPSON_SNOW_SEDIMENT_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_SNOW_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_snow_sediment_256(
    THOMPSON_SNOW_SEDIMENT_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_SNOW_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_snow_sediment_64_with_melt_rain(
    THOMPSON_SNOW_SEDIMENT_MELT_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_SNOW_SEDIMENT_MELT_ARGUMENTS);
}

extern "C" __global__ void thompson_snow_sediment_256_with_melt_rain(
    THOMPSON_SNOW_SEDIMENT_MELT_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_SNOW_SEDIMENT_MELT_ARGUMENTS);
}

#define THOMPSON_SNOW_SEDIMENT_DENSITY_PARAMETERS                       \
    float* __restrict__ qs, const float* __restrict__ temperature,      \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ snownc,            \
    float* __restrict__ snowncv, int accumulate_surface, float dt,      \
    int nz, int ny, int nx

#define THOMPSON_SNOW_SEDIMENT_DENSITY_ARGUMENTS                        \
    qs, (const float*)0, (const float*)0, (const float*)0,              \
    temperature, pressure, qv,                                          \
    reference_density, (const float*)0, (const float*)0, dz, rainnc,    \
    rainncv, snownc,                                                     \
    snowncv,                                                             \
    accumulate_surface, dt, nz, ny, nx

#define THOMPSON_SNOW_SEDIMENT_MELT_DENSITY_PARAMETERS                  \
    float* __restrict__ qs,                                              \
    const float* __restrict__ snow_melt_marker,                          \
    const float* __restrict__ melt_rain_qr,                              \
    const float* __restrict__ melt_rain_nr,                             \
    const float* __restrict__ temperature,                              \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ snownc,            \
    float* __restrict__ snowncv, int accumulate_surface, float dt,      \
    int nz, int ny, int nx

#define THOMPSON_SNOW_SEDIMENT_MELT_DENSITY_ARGUMENTS                   \
    qs, snow_melt_marker, melt_rain_qr, melt_rain_nr,                   \
    temperature, pressure, qv,                                          \
    reference_density, (const float*)0, (const float*)0, dz, rainnc,    \
    rainncv, snownc,                                                     \
    snowncv,                                                             \
    accumulate_surface, dt, nz, ny, nx

extern "C" __global__ void thompson_snow_sediment_64_with_density(
    THOMPSON_SNOW_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_SNOW_SEDIMENT_DENSITY_ARGUMENTS);
}

extern "C" __global__ void thompson_snow_sediment_256_with_density(
    THOMPSON_SNOW_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_SNOW_SEDIMENT_DENSITY_ARGUMENTS);
}

extern "C" __global__ void
thompson_snow_sediment_64_with_melt_rain_and_density(
    THOMPSON_SNOW_SEDIMENT_MELT_DENSITY_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_SNOW_SEDIMENT_MELT_DENSITY_ARGUMENTS);
}

extern "C" __global__ void
thompson_snow_sediment_256_with_melt_rain_and_density(
    THOMPSON_SNOW_SEDIMENT_MELT_DENSITY_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_SNOW_SEDIMENT_MELT_DENSITY_ARGUMENTS);
}

#define THOMPSON_SNOW_SEDIMENT_STATE_PARAMETERS                         \
    float* __restrict__ qs, const float* __restrict__ temperature,      \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ reference_temperature,                    \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ snownc,            \
    float* __restrict__ snowncv, int accumulate_surface, float dt,      \
    int nz, int ny, int nx

#define THOMPSON_SNOW_SEDIMENT_STATE_ARGUMENTS                          \
    qs, (const float*)0, (const float*)0, (const float*)0,              \
    temperature, pressure, qv,                                          \
    reference_density, reference_temperature, (const float*)0, dz,      \
    rainnc, rainncv, snownc, snowncv, accumulate_surface, dt, nz, ny, nx

#define THOMPSON_SNOW_SEDIMENT_MELT_STATE_PARAMETERS                    \
    float* __restrict__ qs,                                              \
    const float* __restrict__ snow_melt_marker,                          \
    const float* __restrict__ melt_rain_qr,                              \
    const float* __restrict__ melt_rain_nr,                             \
    const float* __restrict__ temperature,                              \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ reference_temperature,                    \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ snownc,            \
    float* __restrict__ snowncv, int accumulate_surface, float dt,      \
    int nz, int ny, int nx

#define THOMPSON_SNOW_SEDIMENT_MELT_STATE_ARGUMENTS                     \
    qs, snow_melt_marker, melt_rain_qr, melt_rain_nr,                   \
    temperature, pressure, qv,                                          \
    reference_density, reference_temperature, (const float*)0, dz,      \
    rainnc, rainncv, snownc, snowncv, accumulate_surface, dt, nz, ny, nx

extern "C" __global__ void thompson_snow_sediment_64_with_state(
    THOMPSON_SNOW_SEDIMENT_STATE_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_SNOW_SEDIMENT_STATE_ARGUMENTS);
}

extern "C" __global__ void thompson_snow_sediment_256_with_state(
    THOMPSON_SNOW_SEDIMENT_STATE_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_SNOW_SEDIMENT_STATE_ARGUMENTS);
}

extern "C" __global__ void
thompson_snow_sediment_64_with_melt_rain_and_state(
    THOMPSON_SNOW_SEDIMENT_MELT_STATE_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_SNOW_SEDIMENT_MELT_STATE_ARGUMENTS);
}

extern "C" __global__ void
thompson_snow_sediment_256_with_melt_rain_and_state(
    THOMPSON_SNOW_SEDIMENT_MELT_STATE_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_SNOW_SEDIMENT_MELT_STATE_ARGUMENTS);
}

#define THOMPSON_SNOW_SEDIMENT_STATE_BOOST_PARAMETERS                   \
    float* __restrict__ qs, const float* __restrict__ temperature,      \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ reference_temperature,                    \
    const float* __restrict__ velocity_boost,                            \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ snownc,            \
    float* __restrict__ snowncv, int accumulate_surface, float dt,      \
    int nz, int ny, int nx

#define THOMPSON_SNOW_SEDIMENT_STATE_BOOST_ARGUMENTS                    \
    qs, (const float*)0, (const float*)0, (const float*)0,              \
    temperature, pressure, qv,                                          \
    reference_density, reference_temperature, velocity_boost, dz,      \
    rainnc, rainncv, snownc, snowncv, accumulate_surface, dt, nz, ny, nx

#define THOMPSON_SNOW_SEDIMENT_MELT_STATE_BOOST_PARAMETERS              \
    float* __restrict__ qs,                                              \
    const float* __restrict__ snow_melt_marker,                          \
    const float* __restrict__ melt_rain_qr,                              \
    const float* __restrict__ melt_rain_nr,                             \
    const float* __restrict__ temperature,                              \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ reference_temperature,                    \
    const float* __restrict__ velocity_boost,                           \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ snownc,            \
    float* __restrict__ snowncv, int accumulate_surface, float dt,      \
    int nz, int ny, int nx

#define THOMPSON_SNOW_SEDIMENT_MELT_STATE_BOOST_ARGUMENTS               \
    qs, snow_melt_marker, melt_rain_qr, melt_rain_nr,                   \
    temperature, pressure, qv,                                          \
    reference_density, reference_temperature, velocity_boost, dz,      \
    rainnc, rainncv, snownc, snowncv, accumulate_surface, dt, nz, ny, nx

extern "C" __global__ void thompson_snow_sediment_64_with_state_and_boost(
    THOMPSON_SNOW_SEDIMENT_STATE_BOOST_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_SNOW_SEDIMENT_STATE_BOOST_ARGUMENTS);
}

extern "C" __global__ void thompson_snow_sediment_256_with_state_and_boost(
    THOMPSON_SNOW_SEDIMENT_STATE_BOOST_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_SNOW_SEDIMENT_STATE_BOOST_ARGUMENTS);
}

extern "C" __global__ void
thompson_snow_sediment_64_with_melt_rain_and_state_and_boost(
    THOMPSON_SNOW_SEDIMENT_MELT_STATE_BOOST_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_SNOW_SEDIMENT_MELT_STATE_BOOST_ARGUMENTS);
}

extern "C" __global__ void
thompson_snow_sediment_256_with_melt_rain_and_state_and_boost(
    THOMPSON_SNOW_SEDIMENT_MELT_STATE_BOOST_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_SNOW_SEDIMENT_MELT_STATE_BOOST_ARGUMENTS);
}

// The production snow fallout: the held state, the boost and the melting
// blend with the rain fallout's own profile, formed from its reference
// density.  _and_presence when that density carries L_qr (mp=28).
#define THOMPSON_SNOW_SEDIMENT_RAIN_PROFILE_PARAMETERS                  \
    float* __restrict__ qs,                                              \
    const float* __restrict__ snow_melt_marker,                          \
    const float* __restrict__ melt_rain_qr,                              \
    const float* __restrict__ melt_rain_nr,                             \
    const float* __restrict__ melt_rain_density,                        \
    const float* __restrict__ temperature,                              \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ reference_temperature,                    \
    const float* __restrict__ velocity_boost,                           \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ snownc,            \
    float* __restrict__ snowncv, int accumulate_surface, float dt,      \
    int nz, int ny, int nx

#define THOMPSON_SNOW_SEDIMENT_RAIN_PROFILE_ARGUMENTS                   \
    qs, snow_melt_marker, melt_rain_qr, melt_rain_nr,                   \
    temperature, pressure, qv,                                          \
    reference_density, reference_temperature, velocity_boost, dz,      \
    rainnc, rainncv, snownc, snowncv, accumulate_surface, dt, nz, ny,  \
    nx, melt_rain_density

extern "C" __global__ void thompson_snow_sediment_64_with_rain_profile(
    THOMPSON_SNOW_SEDIMENT_RAIN_PROFILE_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_SNOW_SEDIMENT_RAIN_PROFILE_ARGUMENTS);
}

extern "C" __global__ void thompson_snow_sediment_256_with_rain_profile(
    THOMPSON_SNOW_SEDIMENT_RAIN_PROFILE_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_SNOW_SEDIMENT_RAIN_PROFILE_ARGUMENTS);
}

extern "C" __global__ void
thompson_snow_sediment_64_with_rain_profile_and_presence(
    THOMPSON_SNOW_SEDIMENT_RAIN_PROFILE_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_SHALLOW, true>(
        THOMPSON_SNOW_SEDIMENT_RAIN_PROFILE_ARGUMENTS);
}

extern "C" __global__ void
thompson_snow_sediment_256_with_rain_profile_and_presence(
    THOMPSON_SNOW_SEDIMENT_RAIN_PROFILE_PARAMETERS)
{
    thompson_snow_sediment_impl<THOMPSON_KMAX_GENERIC, true>(
        THOMPSON_SNOW_SEDIMENT_RAIN_PROFILE_ARGUMENTS);
}

extern "C" __global__ void thompson_graupel_fallout_column_mask(
    const float* __restrict__ entry_active,
    const float* __restrict__ qg,
    float* __restrict__ active_columns,
    int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float active = 0.0f;
    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        if (entry_active[idx] != 0.0f && qg[idx] > 1.0e-12f) {
            active = 1.0f;
            break;
        }
    }
    active_columns[column] = active;
}

extern "C" __global__ void thompson_hydrometeor_column_mask(
    const float* __restrict__ mixing_ratio,
    float* __restrict__ active_columns,
    int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float active = 0.0f;
    for (int k = 0; k < nz; ++k) {
        if (mixing_ratio[IDX3(k, j, i)] > 1.0e-12f) {
            active = 1.0f;
            break;
        }
    }
    active_columns[column] = active;
}

// WRF's no-microphysics column, module_mp_thompson.F:1646, :1827-1990 and
// the early return at :2020, for classic mp=8 (the aerosol twin is
// thompson_aa_micro_columns in thompson_aerosol_state.cu).  mp_thompson
// clears no_micro where any ENTRY cloud, ice, rain, snow or graupel mixing
// ratio exceeds R1 (:1827, :1851, :1878, :1906, :1915) or where the entry
// air is supersaturated over ice (:1990; qvsi = rsif at or below 0 C and
// qvs above, :1978-1982; |ssati| < eps is zero, :1989).  A column that keeps
// it returns before the source loop and never reaches the terminal apply,
// so its vapour is not floored at 1.E-10 (:3974).  1.0 where the column has
// microphysics, 0.0 where WRF returns; read by
// thompson_final_phase_cleanup_with_columns.  Take it on the entry state.
// One level's part of that test: true where the level alone clears
// no_micro.  A column has microphysics where any of its levels does.
__device__ __forceinline__ bool thompson_level_has_microphysics(
    const float* __restrict__ qc,
    const float* __restrict__ qi,
    const float* __restrict__ qr,
    const float* __restrict__ qs,
    const float* __restrict__ qg,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    size_t idx)
{
    if (qc[idx] > 1.0e-12f || qi[idx] > 1.0e-12f || qr[idx] > 1.0e-12f
            || qs[idx] > 1.0e-12f || qg[idx] > 1.0e-12f) {
        return true;
    }
    const float temp = temperature[idx];
    const float qv_local = fmaxf(1.0e-10f, qv[idx]);
    const float qvsi = temp - 273.15f <= 0.0f
        ? thompson_rsif(pressure[idx], temp)
        : thompson_rslf(pressure[idx], temp);
    float ssati = qv_local / qvsi - 1.0f;
    if (fabsf(ssati) < 1.0e-15f) ssati = 0.0f;
    return ssati > 0.0f;
}

extern "C" __global__ void thompson_microphysics_columns(
    const float* __restrict__ qc,
    const float* __restrict__ qi,
    const float* __restrict__ qr,
    const float* __restrict__ qs,
    const float* __restrict__ qg,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float* __restrict__ micro_columns,
    int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float micro = 0.0f;
    for (int k = 0; k < nz; ++k) {
        if (thompson_level_has_microphysics(qc, qi, qr, qs, qg, temperature,
                                            pressure, qv, IDX3(k, j, i))) {
            micro = 1.0f;
            break;
        }
    }
    micro_columns[column] = micro;
}

// Classic mp=8 does not carry qng in the Registry.  mp_gt_driver diagnoses a
// transient graupel number moment from qg at the beginning of every call,
// evolves that private moment through the source/sedimentation operators, and
// consumes it only for same-call fallout and REFL_10CM.  Initialize gpuwm's
// output-due shadow with the identical wrapper diagnosis (module_mp_thompson.F
// 1265-1281 and 1915-1939).  The shadow is never a transported model field.
__device__ __forceinline__ float thompson_classic_graupel_number(
    const float* __restrict__ qg,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    int idx)
{
    if (qg[idx] <= 1.0e-12f) return 0.0f;

    const float qvk = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temperature[idx] * (qvk + 0.622f));
    const float rg = qg[idx] * rho;
    const float pi = 3.1415926536f;
    const float am_g = pi * 400.0f / 6.0f;
    const float intercept_power = fmaxf(2.0f, fminf(
        3.0f + (2.0f / 7.0f) * (log10f(fmaxf(1.0e-9f, rg)) + 8.0f),
        6.0f));
    const float intercept = powf(10.0f, intercept_power);
    double lambda = (double)powf(
        intercept * am_g * 6.0f / rg, 0.25f);
    float number = (float)((double)(__fdiv_rn((1.0f / 6.0f) * rg, am_g))
        * lambda * lambda * lambda);

    float mvd = (float)(3.672 / lambda);
    if (mvd > 25.4e-3f) {
        lambda = 3.672 / 25.4e-3;
        number = (float)((double)(__fdiv_rn((1.0f / 6.0f) * rg, am_g))
            * lambda * lambda * lambda);
    } else if (mvd < 50.0e-6f) {
        lambda = 3.672 / 50.0e-6;
        number = (float)((double)(__fdiv_rn((1.0f / 6.0f) * rg, am_g))
            * lambda * lambda * lambda);
    }
    return number / rho;
}

extern "C" __global__ void thompson_classic_graupel_number_init(
    const float* __restrict__ qg,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float* __restrict__ graupel_number_per_kg,
    const int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    graupel_number_per_kg[idx] = thompson_classic_graupel_number(
        qg, temperature, pressure, qv, idx);
}

// mp_thompson applies the private ng1d tendency only after every source and
// sedimentation tendency is complete (module_mp_thompson.F:4059-4077).
// Keep the shadow raw between operators, then reproduce that single final
// lower bound / MVD reconstruction immediately before calc_refl10cm.
//
// It is also the last writer of the graupel MASS in every column, so it
// carries :4058-4063 for the mass as well: `if (qg1d(k) .le. R1)` writes
// qg1d = 0 with ng1d = 0.  The graupel fallout applies that zero only in the
// columns whose fallout runs (its ANY(L_qg) gate); a column whose graupel the
// sources took to a residue at or below R1 without any level keeping L_qg
// used to leave the residue in the state (188,969 cells of one saved
// 19,600-column history frame carry 0 < qg <= 1e-12), where every
// between-step reader and the history saw graupel WRF does not return.
extern "C" __global__ void thompson_classic_graupel_number_finalize(
    float* __restrict__ qg,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float* __restrict__ graupel_number_per_kg,
    const int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (qg[idx] <= 1.0e-12f) {
        qg[idx] = 0.0f;
        graupel_number_per_kg[idx] = 0.0f;
        return;
    }

    const float qvk = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temperature[idx] * (qvk + 0.622f));
    const float pi = 3.1415926536f;
    const float am_g = pi * 400.0f / 6.0f;
    float number_per_kg = fmaxf(
        1.0e-6f / rho, graupel_number_per_kg[idx]);
    double lambda = (double)powf(
        am_g * 6.0f * number_per_kg / qg[idx], 1.0f / 3.0f);
    const float mvd = (float)(3.672 / lambda);
    if (mvd > 25.4e-3f) {
        lambda = 3.672 / 25.4e-3;
        number_per_kg = (float)((double)(__fdiv_rn((1.0f / 6.0f)
            * qg[idx], am_g)) * lambda * lambda * lambda);
    } else if (mvd < 50.0e-6f) {
        lambda = 3.672 / 50.0e-6;
        number_per_kg = (float)((double)(__fdiv_rn((1.0f / 6.0f)
            * qg[idx], am_g)) * lambda * lambda * lambda);
    }
    graupel_number_per_kg[idx] = number_per_kg;
}

template <int KMAX, bool TRACK_NUMBER = false>
__device__ __forceinline__ void thompson_graupel_sediment_impl(
    float* __restrict__ qg,
    const float* __restrict__ graupel_number_per_kg,
    float* __restrict__ graupel_number_shadow,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ graupelnc,
    float* __restrict__ graupelncv,
    int accumulate_surface, float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

#ifndef THOMPSON_NO_EXACT_SHORTCUTS
    // Preserve the terminal writes, including zero-tendency signed-zero changes.
    if (thompson_empty_column(qg, TRACK_NUMBER ? graupel_number_shadow : nullptr, temperature, pressure, qv, dz,
            nullptr, dt, column, nz, ny, nx)) {
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            qg[idx] = 0.0f;
            if (TRACK_NUMBER) graupel_number_shadow[idx] =
                graupel_number_shadow[idx] + 0.0f * dt;
        }
        if (accumulate_surface) rainncv[column] += 0.0f;
        else rainncv[column] = 0.0f;
        if (accumulate_surface) graupelncv[column] += 0.0f;
        else graupelncv[column] = 0.0f;
        rainnc[column] += 0.0f;
        graupelnc[column] += 0.0f;
        return;
    }
#endif  // THOMPSON_NO_EXACT_SHORTCUTS

    float density[KMAX];
    float graupel_mass[KMAX];
    float mass_velocity[KMAX];
    float mass_flux[KMAX];
    float qg_tendency[KMAX];
    float qg_initial[KMAX];
    float diagnostic_number[KMAX];
    float number_velocity[KMAX];
    float number_flux[KMAX];
    float number_tendency[KMAX];
    float number_initial[KMAX];

    const float pi = 3.1415926536f;
    const float am_g = pi * 400.0f / 6.0f;
    const float ogg3 = 1.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above = 0.0f;
    float number_velocity_above = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density[k] = rho;
        qg_initial[k] = qg[idx];
        qg_tendency[k] = 0.0f;
        if (TRACK_NUMBER) {
            number_initial[k] = graupel_number_shadow[idx];
            number_tendency[k] = 0.0f;
        }

        if (qg[idx] > 1.0e-12f) {
            const float state_rho = reference_density == nullptr
                ? rho : reference_density[idx];
            const float rg = qg[idx] * state_rho;
            double lambda;
            if (graupel_number_per_kg != (const float*)0) {
                const float number = fmaxf(
                    1.0e-6f, graupel_number_per_kg[idx] * state_rho);
                lambda = (double)powf(
                    am_g * 6.0f * number / rg, 1.0f / 3.0f);
            } else {
                const float log_mass = log10f(fmaxf(1.0e-9f, rg));
                const float intercept_power = fmaxf(2.0f, fminf(
                    3.0f + (2.0f / 7.0f) * (log_mass + 8.0f), 6.0f));
                const float intercept = powf(10.0f, intercept_power);
                const float lambda_arg = intercept * am_g * 6.0f / rg;
                lambda = (double)powf(lambda_arg, 0.25f);
            }
            float mvd = (float)(3.672 / lambda);
            if (mvd > 25.4e-3f) {
                mvd = 25.4e-3f;
                lambda = 3.672 / (double)mvd;
            } else if (mvd < 50.0e-6f) {
                mvd = 50.0e-6f;
                lambda = 3.672 / (double)mvd;
            }
            const float rhof = sqrtf(rho_not / rho);
            const float prefix = rhof * 442.0f * 20.3632278f * ogg3;
            mass_velocity[k] = (float)((double)prefix
                * pow(1.0 / lambda, 0.89));
            graupel_mass[k] = rg;
            if (TRACK_NUMBER) {
                diagnostic_number[k] = (float)((double)(__fdiv_rn((1.0f / 6.0f)
                    * rg, am_g)) * lambda * lambda * lambda);
                const float number_prefix = rhof * 442.0f * 2.21880022f;
                number_velocity[k] = (float)((double)number_prefix
                    * pow(1.0 / lambda, 0.89));
            }
        } else {
            graupel_mass[k] = 1.0e-12f;
            mass_velocity[k] = velocity_above;
            if (TRACK_NUMBER) {
                diagnostic_number[k] = 1.0e-6f;
                number_velocity[k] = number_velocity_above;
            }
        }
        velocity_above = mass_velocity[k];
        if (TRACK_NUMBER) number_velocity_above = number_velocity[k];

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
            if (TRACK_NUMBER) {
                number_flux[k] = number_velocity[k] * diagnostic_number[k];
            }
        }

        int k = nz - 1;
        size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density[k];
        qg_tendency[k] -= mass_flux[k] * inv_dz * onstep * inv_rho;
        if (TRACK_NUMBER) {
            number_tendency[k] -=
                number_flux[k] * inv_dz * onstep * inv_rho;
        }
        graupel_mass[k] = fmaxf(1.0e-12f,
            graupel_mass[k] - mass_flux[k] * inv_dz * dt_substep);
        if (TRACK_NUMBER) {
            diagnostic_number[k] = fmaxf(1.0e-6f,
                diagnostic_number[k]
                    - number_flux[k] * inv_dz * dt_substep);
        }

        for (k = sediment_top; k >= 0; --k) {
            idx = IDX3(k, j, i);
            inv_dz = 1.0f / dz[idx];
            inv_rho = 1.0f / density[k];
            const float divergence = mass_flux[k + 1] - mass_flux[k];
            qg_tendency[k] += divergence * inv_dz * onstep * inv_rho;
            graupel_mass[k] = fmaxf(1.0e-12f,
                graupel_mass[k] + divergence * inv_dz * dt_substep);
            if (TRACK_NUMBER) {
                const float number_divergence =
                    number_flux[k + 1] - number_flux[k];
                number_tendency[k] +=
                    number_divergence * inv_dz * onstep * inv_rho;
                diagnostic_number[k] = fmaxf(1.0e-6f,
                    diagnostic_number[k]
                        + number_divergence * inv_dz * dt_substep);
            }
        }
        if (graupel_mass[0] > 1.0e-9f) {
            exported += mass_flux[0] * dt_substep;
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qg_new = qg_initial[k] + qg_tendency[k] * dt;
        qg[idx] = qg_new <= 1.0e-12f ? 0.0f : qg_new;
        if (TRACK_NUMBER) {
            graupel_number_shadow[idx] =
                number_initial[k] + number_tendency[k] * dt;
        }
    }
    if (accumulate_surface) {
        rainncv[column] += exported;
        graupelncv[column] += exported;
    } else {
        rainncv[column] = exported;
        graupelncv[column] = exported;
    }
    rainnc[column] += exported;
    graupelnc[column] += exported;
}

#define THOMPSON_GRAUPEL_SEDIMENT_PARAMETERS                             \
    float* __restrict__ qg, const float* __restrict__ temperature,       \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ graupelnc,          \
    float* __restrict__ graupelncv, int accumulate_surface, float dt,    \
    int nz, int ny, int nx

#define THOMPSON_GRAUPEL_SEDIMENT_ARGUMENTS                              \
    qg, (const float*)0, (float*)0, temperature, pressure, qv,            \
    (const float*)0, dz,                                                  \
    rainnc, rainncv, graupelnc, graupelncv, accumulate_surface, dt,     \
    nz, ny, nx

#define THOMPSON_GRAUPEL_SEDIMENT_NUMBER_PARAMETERS                      \
    float* __restrict__ qg,                                              \
    const float* __restrict__ graupel_number_per_kg,                     \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ graupelnc,          \
    float* __restrict__ graupelncv, int accumulate_surface, float dt,    \
    int nz, int ny, int nx

#define THOMPSON_GRAUPEL_SEDIMENT_NUMBER_ARGUMENTS                       \
    qg, graupel_number_per_kg, (float*)0, temperature, pressure, qv,     \
    (const float*)0, dz, rainnc, rainncv, graupelnc, graupelncv,        \
    accumulate_surface, dt, nz, ny, nx

extern "C" __global__ void thompson_graupel_sediment_64(
    THOMPSON_GRAUPEL_SEDIMENT_PARAMETERS)
{
    thompson_graupel_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_GRAUPEL_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_graupel_sediment_256(
    THOMPSON_GRAUPEL_SEDIMENT_PARAMETERS)
{
    thompson_graupel_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_GRAUPEL_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_graupel_sediment_64_with_number(
    THOMPSON_GRAUPEL_SEDIMENT_NUMBER_PARAMETERS)
{
    thompson_graupel_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_GRAUPEL_SEDIMENT_NUMBER_ARGUMENTS);
}

extern "C" __global__ void thompson_graupel_sediment_256_with_number(
    THOMPSON_GRAUPEL_SEDIMENT_NUMBER_PARAMETERS)
{
    thompson_graupel_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_GRAUPEL_SEDIMENT_NUMBER_ARGUMENTS);
}

#define THOMPSON_GRAUPEL_SEDIMENT_DENSITY_PARAMETERS                    \
    float* __restrict__ qg, const float* __restrict__ temperature,      \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ graupelnc,         \
    float* __restrict__ graupelncv, int accumulate_surface, float dt,   \
    int nz, int ny, int nx

#define THOMPSON_GRAUPEL_SEDIMENT_DENSITY_ARGUMENTS                     \
    qg, (const float*)0, (float*)0, temperature, pressure, qv,           \
    reference_density,                                                   \
    dz, rainnc, rainncv, graupelnc, graupelncv, accumulate_surface,    \
    dt, nz, ny, nx

#define THOMPSON_GRAUPEL_SEDIMENT_NUMBER_DENSITY_PARAMETERS             \
    float* __restrict__ qg,                                              \
    const float* __restrict__ graupel_number_per_kg,                    \
    const float* __restrict__ temperature,                              \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ graupelnc,         \
    float* __restrict__ graupelncv, int accumulate_surface, float dt,   \
    int nz, int ny, int nx

#define THOMPSON_GRAUPEL_SEDIMENT_NUMBER_DENSITY_ARGUMENTS              \
    qg, graupel_number_per_kg, (float*)0, temperature, pressure, qv,    \
    reference_density, dz, rainnc, rainncv, graupelnc, graupelncv,     \
    accumulate_surface, dt, nz, ny, nx

extern "C" __global__ void thompson_graupel_sediment_64_with_density(
    THOMPSON_GRAUPEL_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_graupel_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_GRAUPEL_SEDIMENT_DENSITY_ARGUMENTS);
}

extern "C" __global__ void thompson_graupel_sediment_256_with_density(
    THOMPSON_GRAUPEL_SEDIMENT_DENSITY_PARAMETERS)
{
    thompson_graupel_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_GRAUPEL_SEDIMENT_DENSITY_ARGUMENTS);
}

extern "C" __global__ void
thompson_graupel_sediment_64_with_number_and_density(
    THOMPSON_GRAUPEL_SEDIMENT_NUMBER_DENSITY_PARAMETERS)
{
    thompson_graupel_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_GRAUPEL_SEDIMENT_NUMBER_DENSITY_ARGUMENTS);
}

extern "C" __global__ void
thompson_graupel_sediment_256_with_number_and_density(
    THOMPSON_GRAUPEL_SEDIMENT_NUMBER_DENSITY_PARAMETERS)
{
    thompson_graupel_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_GRAUPEL_SEDIMENT_NUMBER_DENSITY_ARGUMENTS);
}

#define THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_PARAMETERS               \
    float* __restrict__ qg, const float* __restrict__ temperature,      \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ graupelnc,         \
    float* __restrict__ graupelncv,                                     \
    const float* __restrict__ active_columns,                           \
    int accumulate_surface, float dt, int nz, int ny, int nx

#define THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_ARGUMENTS                \
    qg, (const float*)0, (float*)0, temperature, pressure, qv,           \
    reference_density,                                                   \
    dz, rainnc, rainncv, graupelnc, graupelncv, accumulate_surface,    \
    dt, nz, ny, nx

extern "C" __global__ void
thompson_graupel_sediment_64_with_density_and_column_mask(
    THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_PARAMETERS)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx || active_columns[column] == 0.0f) return;
    thompson_graupel_sediment_impl<THOMPSON_KMAX_SHALLOW>(
        THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_ARGUMENTS);
}

extern "C" __global__ void
thompson_graupel_sediment_256_with_density_and_column_mask(
    THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_PARAMETERS)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx || active_columns[column] == 0.0f) return;
    thompson_graupel_sediment_impl<THOMPSON_KMAX_GENERIC>(
        THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_ARGUMENTS);
}

#define THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_SHADOW_PARAMETERS        \
    float* __restrict__ qg, const float* __restrict__ temperature,      \
    const float* __restrict__ pressure, const float* __restrict__ qv,   \
    const float* __restrict__ reference_density,                        \
    const float* __restrict__ dz, float* __restrict__ rainnc,           \
    float* __restrict__ rainncv, float* __restrict__ graupelnc,         \
    float* __restrict__ graupelncv,                                     \
    const float* __restrict__ active_columns,                           \
    float* __restrict__ graupel_number_shadow,                          \
    int accumulate_surface, float dt, int nz, int ny, int nx

#define THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_SHADOW_ARGUMENTS         \
    qg, (const float*)0, graupel_number_shadow, temperature, pressure,  \
    qv, reference_density, dz, rainnc, rainncv, graupelnc, graupelncv, \
    accumulate_surface, dt, nz, ny, nx

extern "C" __global__ void
thompson_graupel_sediment_64_with_density_and_column_mask_and_shadow(
    THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_SHADOW_PARAMETERS)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx || active_columns[column] == 0.0f) return;
    thompson_graupel_sediment_impl<THOMPSON_KMAX_SHALLOW, true>(
        THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_SHADOW_ARGUMENTS);
}

extern "C" __global__ void
thompson_graupel_sediment_256_with_density_and_column_mask_and_shadow(
    THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_SHADOW_PARAMETERS)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx || active_columns[column] == 0.0f) return;
    thompson_graupel_sediment_impl<THOMPSON_KMAX_GENERIC, true>(
        THOMPSON_GRAUPEL_SEDIMENT_DENSITY_MASK_SHADOW_ARGUMENTS);
}

extern "C" __global__ void thompson_warm_autoconversion(
    float* __restrict__ qc,
    float* __restrict__ qr,
    float* __restrict__ nr,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float qvk = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temperature[idx] * (qvk + 0.622f));
    const float rc = qc[idx] * rho;
    if (rc <= 0.01e-3f) return;

    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float cloud_number_per_kg = 100.0e6f / rho;
    const float cloud_number = cloud_number_per_kg * rho;
    const float gamma_mass = 1.30767389e12f;
    const float inverse_gamma_number = 2.08767448e-9f;
    const float gamma_higher = 6.40238373e15f;
    const float inverse_gamma_mass = 7.64716632e-13f;

    const float xdc = fmaxf(1.0f,
        powf(rc / (am_r * cloud_number), 1.0f / 3.0f) * 1.0e6f);
    const float lambda = powf(
        cloud_number * am_r * gamma_mass * inverse_gamma_number / rc,
        1.0f / 3.0f);
    const float dcg = powf(gamma_higher * inverse_gamma_mass,
                           1.0f / 3.0f) / lambda * 1.0e6f;
    const float xdc3 = xdc * xdc * xdc;
    const float dcg3 = dcg * dcg * dcg;
    const float dcb_arg = fmaxf(0.0f, xdc3 * dcg3 - xdc3 * xdc3);
    const float dcb = powf(dcb_arg, 1.0f / 6.0f);
    const float zeta_term = 6.25e-6f * xdc * dcb * dcb * dcb - 0.4f;
    const float zeta1 = 0.5f * (zeta_term + fabsf(zeta_term));
    const float zeta = 0.027f * rc * zeta1;
    const float tau_diameter = 0.5f * dcb - 7.5f;
    const float taud = 0.5f * (tau_diameter + fabsf(tau_diameter))
        + 1.0e-12f;
    const float tau = 3.72f / (rc * taud);
    const double rain_rate = fmin((double)(rc / dt),
                                  (double)(zeta / tau));
    const float number_denominator = am_r * 12.0f * 10.0f
        * 50.0e-6f * 50.0e-6f * 50.0e-6f;
    const double number_rate = rain_rate / (double)number_denominator;
    const float mass_tendency = (float)(rain_rate / (double)rho);
    const float number_tendency = (float)(number_rate / (double)rho);

    qc[idx] -= mass_tendency * dt;
    qr[idx] += mass_tendency * dt;
    nr[idx] += number_tendency * dt;
}

extern "C" __global__ void thompson_rain_self_collection(
    const float* __restrict__ qr,
    float* __restrict__ nr,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qr[idx] <= 1.0e-12f) return;

    const float qvk = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temperature[idx] * (qvk + 0.622f));
    const float rain_mass = qr[idx] * rho;
    const float rain_number = fmaxf(1.0e-6f, nr[idx] * rho);
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const float lambda = powf(
        am_r * 6.0f * rain_number / rain_mass, 1.0f / 3.0f);
    const float mvd = 3.672f / lambda;
    if (mvd <= 50.0e-6f) return;

    const float efficiency = 1.0f
        - expf(2300.0f * (mvd - 1950.0e-6f));
    const double collection_rate = (double)(
        efficiency * 2.0f * rain_number * rain_mass);
    const float number_tendency = (float)(collection_rate / (double)rho);
    nr[idx] -= number_tendency * dt;
}

__device__ __forceinline__ void thompson_rain_evaporation_impl(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ reference_density,
    float* __restrict__ reference_temperature,
    const float* __restrict__ graupel_melt_marker,
    const float* __restrict__ source_density,
    const float* __restrict__ condensation_marker,
    float dt, int size, bool density_carries_presence = false)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    // The held source marker and RHOF output are distinct allocations.  The
    // host wrapper rejects identity aliasing because both pointers are
    // restrict-qualified and remain simultaneously live in this kernel.
    const bool melting_graupel = graupel_melt_marker != nullptr
        && graupel_melt_marker[idx] != 0.0f;
    // Rain mass and number concentrations were formed before cloud
    // adjustment.  Only an executed rain evaporation branch refreshes them.
    const float rain_density = source_density == nullptr
        ? rho : source_density[idx];
    if (reference_density != nullptr) reference_density[idx] = rain_density;
    if (reference_temperature != nullptr) {
        reference_temperature[idx] = temp0;
    }
    if (qr[idx] <= 1.0e-12f) {
        // :3236, L_qr.  Where it fails WRF formed no working rain pair:
        // rr(k) = R1 and nr(k) = R2 (:3252-3253).  With
        // density_carries_presence a ZERO density tells the rain fallout
        // and the melting-snow blend exactly that (DENSITY_CARRIES_LQR).
        if (density_carries_presence && reference_density != nullptr) {
            reference_density[idx] = 0.0f;
        }
        return;
    }
    // WRF retains prw_vcd > 0 from cloud adjustment, even when a small
    // post-adjustment saturation residual has the opposite sign.
    if (condensation_marker != nullptr && condensation_marker[idx] != 0.0f) return;

    const float qvs = thompson_rslf(pressure[idx], temp0);
    float ssatw = qv0 / qvs - 1.0f;
    if (fabsf(ssatw) < 1.0e-15f) ssatw = 0.0f;
    if (ssatw >= -1.0e-15f) return;

    const float orho = 1.0f / rho;
    const float rr = qr[idx] * rain_density;
    const float rain_number = fmaxf(1.0e-6f, nr[idx] * rain_density);

    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const double lambda = (double)powf(
        am_r * 6.0f * rain_number / rr, 1.0f / 3.0f);
    const double inverse_lambda = 1.0 / lambda;
    const double intercept = (double)rain_number * lambda;

    const float tempc = temp0 - 273.15f;
    const float inverse_temp = 1.0f / temp0;
    const float rho_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) * orho);
    const float rho_factor_sqrt = sqrtf(rho_factor);
    const float diffusivity = 2.11e-5f
        * powf(__fdiv_rn(temp0, 273.15f), 1.94f)
        * (101325.0f / pressure[idx]);
    const float viscosity = tempc >= 0.0f
        ? (1.718f + 0.0049f * tempc) * 1.0e-5f
        : (1.718f + 0.0049f * tempc
           - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    const float viscosity_factor = sqrtf(rho / viscosity);
    const float latent_heat = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));

    const float saturated_density = rho * qvs;
    const float saturation_slope = saturated_density * inverse_temp
        * (latent_heat * inverse_temp * (1.0f / 461.5f) - 1.0f);
    const float slope_term = inverse_temp
        * (latent_heat * inverse_temp * (1.0f / 461.5f) - 1.0f);
    const float saturation_curvature = saturated_density
        * (slope_term * slope_term
           + (-2.0f * latent_heat * inverse_temp * inverse_temp
              * inverse_temp * (1.0f / 461.5f))
           + inverse_temp * inverse_temp);
    const float gamma = latent_heat * diffusivity / conductivity
        * saturation_slope;
    const float gamma_ratio = gamma / (1.0f + gamma);
    const float alpha = fmaxf(1.0e-9f,
        0.5f * gamma_ratio * gamma_ratio
        * saturation_curvature / saturation_slope
        * saturated_density / saturation_slope);
    const float xsat = fminf(-1.0e-9f, ssatw);
    const float xsat2 = xsat * xsat;
    const float alpha2 = alpha * alpha;
    const float evaporation_geometry = 2.0f * pi
        * (1.0f - alpha * xsat
           + 2.0f * alpha2 * xsat2
           - 5.0f * alpha2 * alpha * xsat2 * xsat)
        / (1.0f + gamma);

    const float schmidt_cuberoot = powf(0.632f, 1.0f / 3.0f);
    const float ventilation_two = 0.308f * schmidt_cuberoot
        * sqrtf(4854.0f) * 2.0f;
    const float rate_prefix = evaporation_geometry * diffusivity * (-ssatw);
    const double diffusion_term = 0.78 * pow(inverse_lambda, 2.0);
    const float ventilation_prefix = ventilation_two
        * viscosity_factor * rho_factor_sqrt;
    const double ventilation_term = (double)ventilation_prefix
        * pow(lambda + 0.5 * 195.0, -3.0);
    double evaporation_rate = (double)rate_prefix * intercept
        * (double)saturated_density
        * (diffusion_term + ventilation_term);

    const float inverse_dt = 1.0f / dt;
    if (qv0 / qvs < 0.95f && rr * orho <= 1.0e-8f) {
        evaporation_rate = (double)(rr * orho * inverse_dt);
    } else {
        const float maximum_rate = fminf(
            rr * orho * inverse_dt, (qvs - qv0) * inverse_dt);
        evaporation_rate = fmin(
            (double)maximum_rate, evaporation_rate * (double)orho);
        // WRF slows evaporation where graupel melted during this source
        // pass: the liquid remains coating a near-0-C particle instead of
        // immediately behaving as freely falling rain.
        if (melting_graupel) {
            const float evaporation_factor = fminf(
                1.0f, 0.01f + (0.99f - 0.01f) * (__fdiv_rn(tempc, 20.0f)));
            evaporation_rate *= (double)evaporation_factor;
        }
    }

    const double number_rate = fmin(
        (double)(rain_number * 0.99f * orho * inverse_dt),
        evaporation_rate * (double)rain_number / (double)rr);
    const float qr_tendency = (float)(-evaporation_rate);
    const float qv_tendency = (float)evaporation_rate;
    const float nr_tendency = (float)(-number_rate);
    const float thermal_factor = latent_heat * inverse_cp;
    const float temperature_tendency = (float)(
        -(double)thermal_factor * evaporation_rate);

    // :3568-3570 rebuild the working rain pair from this density, FLOORED
    // at R1 / R2 where :3237-3238 did not: with density_carries_presence
    // a NEGATIVE density marks the rewrite, its magnitude the density.
    if (reference_density != nullptr) {
        reference_density[idx] = density_carries_presence ? -rho : rho;
    }
    qr[idx] += qr_tendency * dt;
    // :3563 and :3569: the running vapour, unfloored (see the adjustment).
    qv[idx] = qv[idx] + qv_tendency * dt;
    nr[idx] += nr_tendency * dt;
    temperature[idx] = temp0 + temperature_tendency * dt;
}

extern "C" __global__ void thompson_rain_evaporation(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float dt, int size)
{
    thompson_rain_evaporation_impl(
        qr, nr, temperature, pressure, qv, nullptr, nullptr, nullptr,
        nullptr, nullptr, dt, size);
}

extern "C" __global__ void thompson_rain_evaporation_with_density(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ reference_density,
    float dt, int size)
{
    thompson_rain_evaporation_impl(
        qr, nr, temperature, pressure, qv, reference_density, nullptr,
        nullptr, nullptr, nullptr, dt, size);
}

extern "C" __global__ void thompson_rain_evaporation_with_state(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ reference_density,
    float* __restrict__ reference_temperature,
    float dt, int size)
{
    thompson_rain_evaporation_impl(
        qr, nr, temperature, pressure, qv, reference_density,
        reference_temperature, nullptr, nullptr, nullptr, dt, size);
}

extern "C" __global__ void
thompson_rain_evaporation_with_density_and_graupel_melt_marker(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ reference_density,
    const float* __restrict__ graupel_melt_marker,
    float dt, int size)
{
    thompson_rain_evaporation_impl(
        qr, nr, temperature, pressure, qv, reference_density, nullptr,
        graupel_melt_marker, nullptr, nullptr, dt, size);
}

extern "C" __global__ void thompson_rain_evaporation_with_density_history(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ reference_density,
    const float* __restrict__ graupel_melt_marker,
    const float* __restrict__ source_density,
    const float* __restrict__ condensation_marker,
    float dt, int size)
{
    thompson_rain_evaporation_impl(
        qr, nr, temperature, pressure, qv, reference_density, nullptr,
        graupel_melt_marker, source_density, condensation_marker, dt, size);
}

// The production entry point: the history form, and the reference density
// also carries WRF's L_qr (zero where :3236 failed) and the :3568 rewrite
// (negative), which the rain fallout's and the snow fallout's _presence
// entry points read, as thompson_aa_rain_evaporation writes them for mp=28.
extern "C" __global__ void
thompson_rain_evaporation_with_density_history_and_presence(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ reference_density,
    const float* __restrict__ graupel_melt_marker,
    const float* __restrict__ source_density,
    const float* __restrict__ condensation_marker,
    float dt, int size)
{
    thompson_rain_evaporation_impl(
        qr, nr, temperature, pressure, qv, reference_density, nullptr,
        graupel_melt_marker, source_density, condensation_marker, dt, size,
        true);
}

extern "C" __global__ void thompson_snow_sublimation(
    float* __restrict__ qs,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qs[idx] <= 1.0e-12f) return;

    const float temp0 = temperature[idx];
    if (temp0 >= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float qvsi = thompson_rsif(pressure[idx], temp0);
    float ssati = qv0 / qvsi - 1.0f;
    if (fabsf(ssati) < 1.0e-15f) ssati = 0.0f;
    if (ssati == 0.0f) return;

    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float snow_mass = qs[idx] * rho;
    const float tempc = temp0 - 273.15f;
    const float inverse_temp = 1.0f / temp0;
    const float rho_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) * orho);
    const float rho_factor_sqrt = sqrtf(rho_factor);
    const float diffusivity = 2.11e-5f
        * powf(__fdiv_rn(temp0, 273.15f), 1.94f)
        * (101325.0f / pressure[idx]);
    const float viscosity = (1.718f + 0.0049f * tempc
        - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    const float viscosity_factor = sqrtf(rho / viscosity);
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));

    const float saturated_density = rho * qvsi;
    const float saturation_slope = saturated_density * inverse_temp
        * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f);
    const float slope_term = inverse_temp
        * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f);
    const float saturation_curvature = saturated_density
        * (slope_term * slope_term
           + (-2.0f * 2.834e6f * inverse_temp * inverse_temp
              * inverse_temp * (1.0f / 461.5f))
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
    const float sublimation_geometry = 4.0f * 3.1415926536f
        * (1.0f - alpha * xsat
           + 2.0f * alpha2 * xsat2
           - 5.0f * alpha2 * alpha * xsat2 * xsat)
        / (1.0f + gamma);

    const float snow_second_moment = snow_mass * (1.0f / 0.069f);
    const float tc0 = fminf(-0.1f, tempc);
    const float snow_first_moment = thompson_field_a(tc0, 1.0f)
        * powf(snow_second_moment, thompson_field_b(tc0, 1.0f));
    const float deposition_moment = 1.0f + (1.0f + 0.55f) * 0.5f;
    const float snow_ventilation_moment =
        thompson_field_a(tc0, deposition_moment)
        * powf(snow_second_moment,
               thompson_field_b(tc0, deposition_moment));
    const float snow_capacitance = fmaxf(0.15f, fminf(
        0.15f + __fdiv_rn((tempc + 1.5f) * (0.5f - 0.15f), (-30.0f + 1.5f)),
        0.5f));
    const float schmidt_cuberoot = powf(0.632f, 1.0f / 3.0f);
    const float ventilation_coefficient = 0.28f * schmidt_cuberoot
        * sqrtf(40.0f);
    const float moment_sum = 0.86f * snow_first_moment
        + ventilation_coefficient * rho_factor_sqrt * viscosity_factor
          * snow_ventilation_moment;
    double snow_rate = (double)(
        snow_capacitance * sublimation_geometry * diffusivity * ssati
        * saturated_density * moment_sum);

    const float inverse_dt = 1.0f / dt;
    const float vapor_limit = (qv0 - qvsi) * rho * inverse_dt * 0.999f;
    if (snow_rate > 0.0) {
        snow_rate = fmin(snow_rate, (double)vapor_limit);
    } else {
        snow_rate = fmax((double)(-snow_mass * inverse_dt), snow_rate);
        snow_rate = fmax(snow_rate, (double)vapor_limit);
    }

    const float snow_tendency = (float)(snow_rate * (double)orho);
    const float vapor_tendency = -snow_tendency;
    const float thermal_factor = 2.834e6f * inverse_cp;
    const float temperature_tendency = (float)(
        (double)thermal_factor * snow_rate * (double)orho);

    qs[idx] += snow_tendency * dt;
    qv[idx] = fmaxf(1.0e-10f, qv0 + vapor_tendency * dt);
    temperature[idx] = temp0 + temperature_tendency * dt;
}

extern "C" __global__ void thompson_graupel_sublimation(
    float* __restrict__ qg,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qg[idx] <= 1.0e-12f) return;

    const float temp0 = temperature[idx];
    if (temp0 >= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float qvsi = thompson_rsif(pressure[idx], temp0);
    float ssati = qv0 / qvsi - 1.0f;
    if (fabsf(ssati) < 1.0e-15f) ssati = 0.0f;
    if (ssati >= -1.0e-15f) return;

    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float graupel_mass = qg[idx] * rho;
    const float tempc = temp0 - 273.15f;
    const float inverse_temp = 1.0f / temp0;
    const float rho_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) * orho);
    const float rho_factor_sqrt = sqrtf(rho_factor);
    const float diffusivity = 2.11e-5f
        * powf(__fdiv_rn(temp0, 273.15f), 1.94f)
        * (101325.0f / pressure[idx]);
    const float viscosity = (1.718f + 0.0049f * tempc
        - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    const float viscosity_factor = sqrtf(rho / viscosity);
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));

    const float saturated_density = rho * qvsi;
    const float saturation_slope = saturated_density * inverse_temp
        * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f);
    const float slope_term = inverse_temp
        * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f);
    const float saturation_curvature = saturated_density
        * (slope_term * slope_term
           + (-2.0f * 2.834e6f * inverse_temp * inverse_temp
              * inverse_temp * (1.0f / 461.5f))
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
    const float sublimation_geometry = 4.0f * 3.1415926536f
        * (1.0f - alpha * xsat
           + 2.0f * alpha2 * xsat2
           - 5.0f * alpha2 * alpha * xsat2 * xsat)
        / (1.0f + gamma);

    // mp_gt_driver diagnoses classic fixed-density graupel number from qg;
    // mp_thompson then reconstructs lambda and N0 from that rounded state.
    const float am_g = 3.1415926536f * 400.0f / 6.0f;
    const float log_mass = log10f(fmaxf(1.0e-9f, graupel_mass));
    const float intercept_power = fmaxf(2.0f, fminf(
        3.0f + (2.0f / 7.0f) * (log_mass + 8.0f), 6.0f));
    const float diagnosed_intercept = powf(10.0f, intercept_power);
    float lambda = powf(
        diagnosed_intercept * am_g * 6.0f / graupel_mass, 0.25f);
    float number_per_kg = __fdiv_rn((1.0f / 6.0f) * graupel_mass
        * powf(lambda, 3.0f), am_g) / rho;
    number_per_kg = fmaxf(1.0e-6f, number_per_kg);
    float graupel_number = fmaxf(1.0e-6f, number_per_kg * rho);
    lambda = powf(
        am_g * 6.0f * graupel_number / graupel_mass, 1.0f / 3.0f);
    float mvd = 3.672f / lambda;
    if (mvd > 25.4e-3f) {
        mvd = 25.4e-3f;
        lambda = 3.672f / mvd;
        graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass
            * powf(lambda, 3.0f), am_g);
    } else if (mvd < 50.0e-6f) {
        mvd = 50.0e-6f;
        lambda = 3.672f / mvd;
        graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass
            * powf(lambda, 3.0f), am_g);
    }
    const float inverse_lambda = 1.0f / lambda;
    const float graupel_intercept = graupel_number * lambda;
    const float schmidt_cuberoot = powf(0.632f, 1.0f / 3.0f);
    const float ventilation_coefficient = 0.28f * schmidt_cuberoot
        * sqrtf(442.0f) * 1.9021706581115723f;
    const float moment_sum = graupel_intercept * (
        0.86f * powf(inverse_lambda, 2.0f)
        + ventilation_coefficient * viscosity_factor * rho_factor_sqrt
          * powf(inverse_lambda, 2.945f));
    double graupel_rate = (double)(
        0.5f * sublimation_geometry * diffusivity * ssati
        * saturated_density * moment_sum);

    const float inverse_dt = 1.0f / dt;
    const float vapor_limit = (qv0 - qvsi) * rho * inverse_dt * 0.999f;
    graupel_rate = fmax((double)(-graupel_mass * inverse_dt),
                        graupel_rate);
    graupel_rate = fmax(graupel_rate, (double)vapor_limit);

    const float graupel_tendency = (float)(graupel_rate * (double)orho);
    const float vapor_tendency = -graupel_tendency;
    const float thermal_factor = 2.834e6f * inverse_cp;
    const float temperature_tendency = (float)(
        (double)thermal_factor * graupel_rate * (double)orho);

    qg[idx] += graupel_tendency * dt;
    qv[idx] = fmaxf(1.0e-10f, qv0 + vapor_tendency * dt);
    temperature[idx] = temp0 + temperature_tendency * dt;
}

__device__ __forceinline__ void thompson_bound_rain_number(
    float rain_mass, float density, float* rain_number_per_kg)
{
    if (rain_mass <= 1.0e-12f) {
        *rain_number_per_kg = 0.0f;
        return;
    }
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    float rain_number = fmaxf(1.0e-6f,
                               *rain_number_per_kg * density);
    float lambda = powf(am_r * 6.0f * rain_number / rain_mass,
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

// WRF diagnoses a bounded local rain distribution at mp_gt_driver entry
// before any source rate is evaluated (module_mp_thompson.F:1878-1898).
// That local number is deliberately distinct from the prognostic nr1d: the
// latter is corrected only by the post-source mass/number bound below.  In
// particular, qr > 0 with nr <= R2 uses a 1-mm local MVD.  Feeding the raw
// R2 floor to the breakup relation can otherwise create an enormous MVD,
// overflow exp(2300*(MVD-1.95 mm)), and turn a number sink into +Inf.
__device__ __forceinline__ bool thompson_prepare_entry_rain_distribution(
    float rain_per_kg, float rain_number_per_kg, float density,
    float* rain_number, double* rain_lambda, float* rain_mvd)
{
    *rain_number = 1.0e-6f;
    *rain_lambda = 1.0;
    *rain_mvd = 0.0f;
    if (rain_per_kg <= 1.0e-12f) return false;

    const float rain_mass = rain_per_kg * density;
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    float number = fmaxf(1.0e-6f, rain_number_per_kg * density);
    if (number <= 1.0e-6f) {
        const float lambda = 3.672f / 1.0e-3f;
        number = __fdiv_rn((1.0f / 6.0f) * rain_mass, am_r)
            * lambda * lambda * lambda;
    }

    float lambda = powf(am_r * 6.0f * number / rain_mass, 1.0f / 3.0f);
    float mvd = 3.672f / lambda;
    if (mvd > 2.5e-3f) {
        mvd = 2.5e-3f;
    } else if (mvd < 37.5e-6f) {
        mvd = 37.5e-6f;
    } else {
        *rain_number = number;
        *rain_lambda = (double)lambda;
        *rain_mvd = mvd;
        return true;
    }
    lambda = 3.672f / mvd;
    number = __fdiv_rn((1.0f / 6.0f) * rain_mass, am_r)
        * lambda * lambda * lambda;
    *rain_number = number;
    *rain_lambda = (double)lambda;
    *rain_mvd = mvd;
    return true;
}

extern "C" __global__ void thompson_snow_melting(
    float* __restrict__ qs,
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qs[idx] <= 1.0e-12f) return;

    const float temp0 = temperature[idx];
    if (temp0 <= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float qvs = thompson_rslf(pressure[idx], temp0);
    // This admitted slice pins WRF's saturated warm branch.  The general
    // wet-bulb path remains part of the full-driver admission.
    if (qv0 / qvs < 0.999f) return;

    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float snow_mass = qs[idx] * rho;
    const float tempc = temp0 - 273.15f;
    const float diffusivity = 2.11e-5f
        * powf(__fdiv_rn(temp0, 273.15f), 1.94f)
        * (101325.0f / pressure[idx]);
    const float viscosity = (1.718f + 0.0049f * tempc) * 1.0e-5f;
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float density_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) * orho);
    const float density_factor_sqrt = sqrtf(density_factor);
    const float viscosity_factor = sqrtf(rho / viscosity);
    const float vapor_deficit = fmaxf(
        0.0f, thompson_rslf(pressure[idx], 273.15f) - qv0);

    const float tc0 = fminf(-0.1f, tempc);
    const float snow_second_moment = snow_mass * (1.0f / 0.069f);
    const float snow_number = thompson_field_a(tc0, 0.0f)
        * powf(snow_second_moment, thompson_field_b(tc0, 0.0f));
    const float snow_first_moment = thompson_field_a(tc0, 1.0f)
        * powf(snow_second_moment, thompson_field_b(tc0, 1.0f));
    const float ventilation_moment = 1.0f + (1.0f + 0.55f) * 0.5f;
    const float snow_ventilation_moment =
        thompson_field_a(tc0, ventilation_moment)
        * powf(snow_second_moment,
               thompson_field_b(tc0, ventilation_moment));
    const float inverse_fusion = 1.0f / 334000.0f;
    const float schmidt_cuberoot = powf(0.632f, 1.0f / 3.0f);
    const float melt_moment =
        (3.1415926536f * 4.0f * 0.15f * inverse_fusion * 0.86f)
            * snow_first_moment
        + (3.1415926536f * 4.0f * 0.15f * inverse_fusion
           * 0.28f * schmidt_cuberoot * sqrtf(40.0f))
            * density_factor_sqrt * viscosity_factor
            * snow_ventilation_moment;
    const float energy = tempc * conductivity
        - 2.5e6f * diffusivity * vapor_deficit;
    double melt_rate = (double)(energy * melt_moment);
    melt_rate = fmin((double)(snow_mass / dt), fmax(0.0, melt_rate));
    if (melt_rate <= 0.0) return;

    const double number_rate = (double)(
        snow_number / snow_mass * (float)melt_rate
        * powf(10.0f, -0.25f * tempc));
    const float mass_tendency = (float)(melt_rate * (double)orho);
    const float number_tendency = (float)(number_rate * (double)orho);
    qs[idx] = fmaxf(0.0f, qs[idx] - mass_tendency * dt);
    qr[idx] = fmaxf(0.0f, qr[idx] + mass_tendency * dt);
    nr[idx] = fmaxf(0.0f, nr[idx] + number_tendency * dt);
    thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    temperature[idx] = temp0 - 334000.0f * inverse_cp
        * mass_tendency * dt;
}

extern "C" __global__ void thompson_graupel_melting(
    float* __restrict__ qg,
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ graupel_number_per_kg,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (qg[idx] <= 1.0e-12f) {
        graupel_number_per_kg[idx] = 0.0f;
        return;
    }

    const float temp0 = temperature[idx];
    if (temp0 <= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float qvs = thompson_rslf(pressure[idx], temp0);
    if (qv0 / qvs < 0.999f) return;

    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float graupel_mass = qg[idx] * rho;
    const float am_g = 3.1415926536f * 400.0f / 6.0f;
    const float log_mass = log10f(fmaxf(1.0e-9f, graupel_mass));
    const float intercept_power = fmaxf(2.0f, fminf(
        3.0f + (2.0f / 7.0f) * (log_mass + 8.0f), 6.0f));
    const float diagnosed_intercept = powf(10.0f, intercept_power);
    float lambda = powf(
        diagnosed_intercept * am_g * 6.0f / graupel_mass, 0.25f);
    float graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass
        * lambda * lambda * lambda, am_g);
    graupel_number = fmaxf(1.0e-6f, graupel_number);
    lambda = powf(am_g * 6.0f * graupel_number / graupel_mass,
                  1.0f / 3.0f);
    float mvd = 3.672f / lambda;
    if (mvd > 25.4e-3f) {
        mvd = 25.4e-3f;
        lambda = 3.672f / mvd;
        graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass
            * lambda * lambda * lambda, am_g);
    } else if (mvd < 50.0e-6f) {
        mvd = 50.0e-6f;
        lambda = 3.672f / mvd;
        graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass
            * lambda * lambda * lambda, am_g);
    }
    graupel_number_per_kg[idx] = graupel_number * orho;

    const float tempc = temp0 - 273.15f;
    const float diffusivity = 2.11e-5f
        * powf(__fdiv_rn(temp0, 273.15f), 1.94f)
        * (101325.0f / pressure[idx]);
    const float viscosity = (1.718f + 0.0049f * tempc) * 1.0e-5f;
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float density_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) * orho);
    const float density_factor_sqrt = sqrtf(density_factor);
    const float viscosity_factor = sqrtf(rho / viscosity);
    const float vapor_deficit = fmaxf(
        0.0f, thompson_rslf(pressure[idx], 273.15f) - qv0);
    const float inverse_lambda = 1.0f / lambda;
    const float graupel_intercept = graupel_number * lambda;
    float melt_intercept = graupel_intercept;
    if (graupel_mass * graupel_number < 1.0e-4f) {
        melt_intercept = (1.0e-4f / graupel_mass) * lambda;
    }
    const float inverse_fusion = 1.0f / 334000.0f;
    const float schmidt_cuberoot = powf(0.632f, 1.0f / 3.0f);
    const float melt_moment = melt_intercept * (
        (3.1415926536f * 4.0f * 0.5f * inverse_fusion * 0.86f)
            * inverse_lambda * inverse_lambda
        + (3.1415926536f * 4.0f * 0.5f * inverse_fusion
           * 0.28f * schmidt_cuberoot * sqrtf(442.0f)
           * 1.9021706581115723f)
            * density_factor_sqrt * viscosity_factor
            * powf(inverse_lambda, 2.945f));
    const float energy = tempc * conductivity
        - 2.5e6f * diffusivity * vapor_deficit;
    double melt_rate = (double)(energy * melt_moment);
    melt_rate = fmin((double)(graupel_mass / dt), fmax(0.0, melt_rate));
    if (melt_rate <= 0.0) return;

    const double number_rate = (double)(
        (float)melt_rate * graupel_number / graupel_mass
        * powf(10.0f, -0.33f * tempc));
    const float mass_tendency = (float)(melt_rate * (double)orho);
    const float number_tendency = (float)(number_rate * (double)orho);
    qg[idx] = fmaxf(0.0f, qg[idx] - mass_tendency * dt);
    qr[idx] = fmaxf(0.0f, qr[idx] + mass_tendency * dt);
    nr[idx] = fmaxf(0.0f, nr[idx] + number_tendency * dt);
    graupel_number_per_kg[idx] = fmaxf(
        0.0f, graupel_number_per_kg[idx] - number_tendency * dt);
    if (qg[idx] > 1.0e-12f) {
        const float remaining_mass = qg[idx] * rho;
        float remaining_number = fmaxf(
            1.0e-6f, graupel_number_per_kg[idx] * rho);
        float final_lambda = powf(
            am_g * 6.0f * remaining_number / remaining_mass,
            1.0f / 3.0f);
        float final_mvd = 3.672f / final_lambda;
        if (final_mvd > 25.4e-3f) {
            final_mvd = 25.4e-3f;
        } else if (final_mvd < 50.0e-6f) {
            final_mvd = 50.0e-6f;
        } else {
            final_mvd = 0.0f;
        }
        if (final_mvd > 0.0f) {
            final_lambda = 3.672f / final_mvd;
            remaining_number = __fdiv_rn((1.0f / 6.0f) * remaining_mass
                * final_lambda * final_lambda * final_lambda, am_g);
            graupel_number_per_kg[idx] = remaining_number * orho;
        }
    } else {
        qg[idx] = 0.0f;
        graupel_number_per_kg[idx] = 0.0f;
    }
    thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    temperature[idx] = temp0 - 334000.0f * inverse_cp
        * mass_tendency * dt;
}

extern "C" __global__ void thompson_warm_rain_collection(
    float* __restrict__ qc,
    float* __restrict__ qr,
    float* __restrict__ nr,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const double* __restrict__ rain_cloud_efficiency,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qr[idx] <= 1.0e-12f) return;

    const float qvk = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temperature[idx] * (qvk + 0.622f));
    const float rain_mass = qr[idx] * rho;
    const float rain_number = fmaxf(1.0e-6f, nr[idx] * rho);
    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float rain_lambda = powf(
        am_r * 6.0f * rain_number / rain_mass, 1.0f / 3.0f);
    const float rain_mvd = 3.672f / rain_lambda;

    double number_rate = 0.0;
    if (rain_mvd > 50.0e-6f) {
        const float efficiency = 1.0f
            - expf(2300.0f * (rain_mvd - 1950.0e-6f));
        number_rate = (double)(
            efficiency * 2.0f * rain_number * rain_mass);
    }

    double accretion_rate = 0.0;
    const float cloud_mass = qc[idx] * rho;
    if (cloud_mass > 1.0e-12f && rain_mvd > 50.0e-6f) {
        const float cloud_number_per_kg = 100.0e6f / rho;
        const float cloud_number = cloud_number_per_kg * rho;
        const float gamma_mass = 1.30767389e12f;
        const float inverse_gamma_number = 2.08767448e-9f;
        const float cloud_lambda = powf(
            cloud_number * am_r * gamma_mass * inverse_gamma_number
                / cloud_mass,
            1.0f / 3.0f);
        const float cloud_mvd = fmaxf(1.0e-6f, fminf(
            15.672f / cloud_lambda, 50.0e-6f));
        if (cloud_mvd > 1.0e-6f) {
            const double dr_first = 5.1164649614037726e-05;
            const double dr_last = 0.004886186104779057;
            int rain_bin = 1 + (int)(100.0
                * log((double)rain_mvd / dr_first)
                / log(dr_last / dr_first));
            rain_bin = min(rain_bin, 100);
            const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
            const float collision_efficiency = (float)
                rain_cloud_efficiency[(rain_bin - 1)
                                      + 100 * (cloud_bin - 1)];
            const float rhof = sqrtf(
                (101325.0f / (287.05f * 298.0f)) / rho);
            const float coefficient = pi * 0.25f * 4854.0f * 6.0f;
            const float intercept = rain_number * rain_lambda;
            const float rate = rhof * coefficient * collision_efficiency
                * cloud_mass * intercept
                * powf(rain_lambda + 195.0f, -4.0f);
            accretion_rate = fmin((double)(cloud_mass / dt),
                                  (double)rate);
        }
    }

    const float mass_tendency = (float)(accretion_rate / (double)rho);
    const float number_tendency = (float)(number_rate / (double)rho);
    qc[idx] -= mass_tendency * dt;
    qr[idx] += mass_tendency * dt;
    nr[idx] -= number_tendency * dt;
}

extern "C" __global__ void thompson_warm_process_network(
    float* __restrict__ qc,
    float* __restrict__ qr,
    float* __restrict__ nr,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const double* __restrict__ rain_cloud_efficiency,
    float dt, int size)
{
    // First fused slice of the production driver.  Every rate below is
    // diagnosed from the same incoming state, then WRF's shared cloud-water
    // source cap is applied once before any category is updated.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float qvk = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temperature[idx] * (qvk + 0.622f));
    const float orho = 1.0f / rho;
    const float cloud_mass = fmaxf(0.0f, qc[idx] * rho);
    const float rain_mass = fmaxf(0.0f, qr[idx] * rho);
    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;

    // Classic mp=8 diagnoses a fixed 100 cm^-3 cloud population.  The
    // following constants are WRF's nu_c=12 gamma moments.
    const float cloud_number_per_kg = 100.0e6f / rho;
    const float cloud_number = cloud_number_per_kg * rho;
    const float gamma_mass = 1.30767389e12f;
    const float inverse_gamma_number = 2.08767448e-9f;
    const float gamma_higher = 6.40238373e15f;
    const float inverse_gamma_mass = 7.64716632e-13f;

    float cloud_lambda = 0.0f;
    float cloud_mvd = 1.0e-6f;
    if (cloud_mass > 1.0e-12f) {
        cloud_lambda = powf(
            cloud_number * am_r * gamma_mass * inverse_gamma_number
                / cloud_mass,
            1.0f / 3.0f);
        cloud_mvd = fmaxf(1.0e-6f, fminf(
            15.672f / cloud_lambda, 50.0e-6f));
    }

    double autoconversion_rate = 0.0;
    double autoconversion_number_rate = 0.0;
    if (cloud_mass > 0.01e-3f) {
        const float xdc = fmaxf(1.0f,
            powf(cloud_mass / (am_r * cloud_number), 1.0f / 3.0f)
                * 1.0e6f);
        const float dcg = powf(gamma_higher * inverse_gamma_mass,
                               1.0f / 3.0f)
            / cloud_lambda * 1.0e6f;
        const float xdc3 = xdc * xdc * xdc;
        const float dcg3 = dcg * dcg * dcg;
        const float dcb_arg = fmaxf(
            0.0f, xdc3 * dcg3 - xdc3 * xdc3);
        const float dcb = powf(dcb_arg, 1.0f / 6.0f);
        const float zeta_term = 6.25e-6f * xdc * dcb * dcb * dcb - 0.4f;
        const float zeta1 = 0.5f * (zeta_term + fabsf(zeta_term));
        const float zeta = 0.027f * cloud_mass * zeta1;
        const float tau_diameter = 0.5f * dcb - 7.5f;
        const float taud = 0.5f * (
            tau_diameter + fabsf(tau_diameter)) + 1.0e-12f;
        const float tau = 3.72f / (cloud_mass * taud);
        autoconversion_rate = fmin(
            (double)(cloud_mass / dt), (double)(zeta / tau));
        autoconversion_number_rate = autoconversion_rate
            / (double)(am_r * 12.0f * 10.0f
                       * 50.0e-6f * 50.0e-6f * 50.0e-6f);
    }

    double self_collection_number_rate = 0.0;
    double accretion_rate = 0.0;
    if (rain_mass > 1.0e-12f) {
        const float rain_number = fmaxf(1.0e-6f, nr[idx] * rho);
        const float rain_lambda = powf(
            am_r * 6.0f * rain_number / rain_mass, 1.0f / 3.0f);
        const float rain_mvd = 3.672f / rain_lambda;

        if (rain_mvd > 50.0e-6f) {
            const float efficiency = 1.0f
                - expf(2300.0f * (rain_mvd - 1950.0e-6f));
            self_collection_number_rate = (double)(
                efficiency * 2.0f * rain_number * rain_mass);
        }

        if (cloud_mass > 1.0e-12f && rain_mvd > 50.0e-6f
                && cloud_mvd > 1.0e-6f) {
            const double dr_first = 5.1164649614037726e-05;
            const double dr_last = 0.004886186104779057;
            int rain_bin = 1 + (int)(100.0
                * log((double)rain_mvd / dr_first)
                / log(dr_last / dr_first));
            rain_bin = min(rain_bin, 100);
            const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
            const float collision_efficiency = (float)
                rain_cloud_efficiency[(rain_bin - 1)
                                      + 100 * (cloud_bin - 1)];
            const float rhof = sqrtf(
                (101325.0f / (287.05f * 298.0f)) / rho);
            const float coefficient = pi * 0.25f * 4854.0f * 6.0f;
            const float intercept = rain_number * rain_lambda;
            const float rate = rhof * coefficient * collision_efficiency
                * cloud_mass * intercept
                * powf(rain_lambda + 195.0f, -4.0f);
            accretion_rate = fmin(
                (double)(cloud_mass / dt), (double)rate);
        }
    }

    const double cloud_sink = autoconversion_rate + accretion_rate;
    const double cloud_rate_limit = (double)cloud_mass / (double)dt;
    if (cloud_sink > cloud_rate_limit && cloud_sink > 0.0) {
        const double ratio = cloud_rate_limit / cloud_sink;
        autoconversion_rate *= ratio;
        accretion_rate *= ratio;
        // WRF v4.6.1 intentionally does not rescale pnr_wau here.  The later
        // rain-number diameter bound consumes the original number rate.
    }

    const float mass_tendency = (float)(
        (autoconversion_rate + accretion_rate) * (double)orho);
    const float number_tendency = (float)(
        (autoconversion_number_rate - self_collection_number_rate)
        * (double)orho);
    qc[idx] = fmaxf(0.0f, qc[idx] - mass_tendency * dt);
    qr[idx] = fmaxf(0.0f, qr[idx] + mass_tendency * dt);
    nr[idx] = fmaxf(0.0f, nr[idx] + number_tendency * dt);
    thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
}

__device__ __forceinline__ int thompson_decade_table_index(
    float value, int first_exponent, int table_size)
{
    // module_mp_thompson.F:2282-2307.  WRF searches around NINT(log10(x))
    // before truncating the positive base-10 mantissa into each 1..9 bin.
    const int center = (int)roundf(log10f(value));
    int exponent = center;
    for (int candidate = center - 1; candidate <= center + 1; ++candidate) {
        const float scale = powf(10.0f, (float)candidate);
        const float mantissa = value / scale;
        if (mantissa >= 1.0f && mantissa < 10.0f) {
            exponent = candidate;
            break;
        }
    }
    const float scale = powf(10.0f, (float)exponent);
    const int digit = (int)(value / scale);
    const int one_based = digit + 9 * (exponent - first_exponent);
    return max(0, min(one_based - 1, table_size - 1));
}

__device__ __forceinline__ int thompson_decade_table_index_double(
    double value, int first_exponent, int table_size)
{
    // The rain-intercept lookup is DOUBLE PRECISION in WRF even though the
    // rain state and the base-ten scale are default REAL.
    const int center = (int)round(log10(value));
    int exponent = center;
    for (int candidate = center - 1; candidate <= center + 1; ++candidate) {
        const float scale = powf(10.0f, (float)candidate);
        const double mantissa = value / (double)scale;
        if (mantissa >= 1.0 && mantissa < 10.0) {
            exponent = candidate;
            break;
        }
    }
    const float scale = powf(10.0f, (float)exponent);
    const int digit = (int)(value / (double)scale);
    const int one_based = digit + 9 * (exponent - first_exponent);
    return max(0, min(one_based - 1, table_size - 1));
}

// THE RAIN-GRAUPEL TABLE INDEX (mp=8).  WRF allocates tcg_racg..tnr_gacr as
// (ntb_g1, ntb_g, dimNRHG, ntb_r1, ntb_r) with dimNRHG = NRHG1 = 1 when the
// scheme is not hail aware (module_mp_thompson.F:465, :607-615), builds its
// one density slab for rho_g(idx_bg1) (:4123), and then reads it at
// :2527-2545 with idx_bg(k) = idx_bg1 = 5 (:1950, :2848): an out-of-bounds
// subscript on a dimension of extent 1, which gfortran without bounds
// checking turns into a read 4*37*37 words further on (idx_r1+4, and past
// the end of the array for the last rain bins).  WOOF does not reproduce an
// out-of-bounds read: it reads the one slab the table holds, the collision
// rates WRF built for this graupel density.  This is a DECLARED divergence
// from WRF v4.6.1 wherever rain meets graupel (prg_rcg, prr_rcg, pnr_rcg,
// png_rcg and everything downstream of them), the same one mp=28 declares
// at aaf_racg_index (thompson_aerosol_common.cuh).  Bins are zero based.
__device__ __forceinline__ size_t thompson_racg_index(
    int graupel_intercept_bin, int graupel_mass_bin,
    int rain_intercept_bin, int rain_mass_bin)
{
    return (size_t)graupel_intercept_bin
        + (size_t)37 * ((size_t)graupel_mass_bin
        + (size_t)37 * ((size_t)rain_intercept_bin
        + (size_t)37 * (size_t)rain_mass_bin));
}

extern "C" __global__ void thompson_warm_frozen_source_network(
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
    float dt, int size)
{
    // WRF's cold source kernel already owns every sub-freezing level.  This
    // companion owns the remaining ambient-warm levels, while still using
    // WRF's wet-bulb temperature to select the inverse (melting) collision
    // branches.  All rates below read one immutable incoming state and are
    // applied only after the WRF-ordered category conservation passes.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const bool entry_warm = graupel_melt_marker[idx] != 0.0f;
    // These in-call decisions must never retain a prior invocation's value.
    // The graupel buffer arrives seeded with the entry-temperature mask and
    // becomes the held prr_gml marker after that one read.
    graupel_melt_marker[idx] = 0.0f;
    snow_melt_marker[idx] = 0.0f;
    if (!entry_warm) return;

    const float temp0 = temperature[idx];
    const float pressure0 = pressure[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure0
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float inverse_dt = 1.0f / dt;
    const float tempc = temp0 - 273.15f;
    const float wet_bulb = thompson_wet_bulb_temperature(
        pressure0, temp0, qv0);
    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float am_g = pi * 400.0f / 6.0f;
    const float rho_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) * orho);

    const float cloud_mass = fmaxf(0.0f, qc[idx] * rho);
    const float rain_mass = fmaxf(0.0f, qr[idx] * rho);
    const float snow_mass = fmaxf(0.0f, qs[idx] * rho);
    const float graupel_mass = fmaxf(0.0f, qg[idx] * rho);
    // Presence is WRF's L_qc / L_qs / L_qg / L_qr, the MIXING RATIO tests
    // of the entry block (:1827-1949), not the concentration q*rho: in thin
    // air (rho < 1) a mixing ratio just above R1 is a concentration at or
    // below it, and WRF still forms that species' distribution and rates
    // (:2029, :2169, :2194, :2783, :2803) and applies its conservation limit
    // (:2893-2934) there.  The mp=28 networks test the mixing ratio too.
    const bool l_qc = qc[idx] > 1.0e-12f;
    const bool l_qs = qs[idx] > 1.0e-12f;
    const bool l_qg = qg[idx] > 1.0e-12f;

    const float cloud_number = 100.0e6f;
    const float gamma_mass = 1.30767389e12f;
    const float inverse_gamma_number = 2.08767448e-9f;
    const float gamma_higher = 6.40238373e15f;
    const float inverse_gamma_mass = 7.64716632e-13f;
    float cloud_lambda = 0.0f;
    float cloud_mvd = 1.0e-6f;
    if (l_qc) {
        cloud_lambda = powf(
            cloud_number * am_r * gamma_mass * inverse_gamma_number
                / cloud_mass,
            1.0f / 3.0f);
        cloud_mvd = fmaxf(1.0e-6f, fminf(
            15.672f / cloud_lambda, 50.0e-6f));
    }

    float rain_number;
    double rain_lambda;
    float rain_mvd;
    const bool rain_active = thompson_prepare_entry_rain_distribution(
        qr[idx], nr[idx], rho, &rain_number, &rain_lambda, &rain_mvd);

    float graupel_number = l_qg
        ? fmaxf(1.0e-6f, graupel_number_per_kg[idx] * rho) : 0.0f;
    double graupel_lambda = l_qg
        ? (double)powf(am_g * 6.0f * graupel_number / graupel_mass,
                       1.0f / 3.0f)
        : 1.0;
    if (l_qg) {
        float diameter = (float)(3.672 / graupel_lambda);
        if (diameter > 25.4e-3f) {
            graupel_lambda = 3.672 / 25.4e-3;
        } else if (diameter < 50.0e-6f) {
            graupel_lambda = 3.672 / 50.0e-6;
        }
        graupel_number = (float)((double)(__fdiv_rn((1.0f / 6.0f)
            * graupel_mass, am_g)) * graupel_lambda * graupel_lambda
            * graupel_lambda);
    }

    // Berry-Reinhardt autoconversion, rain-cloud accretion, and Seifert
    // self-collection are simultaneous in WRF, including on frozen-bearing
    // warm levels.
    double autoconversion_rate = 0.0;
    double autoconversion_number_rate = 0.0;
    if (cloud_mass > 0.01e-3f) {
        const float xdc = fmaxf(1.0f,
            powf(__fdiv_rn(cloud_mass, am_r * cloud_number), 1.0f / 3.0f)
                * 1.0e6f);
        const float dcg = powf(
            gamma_higher * inverse_gamma_mass, 1.0f / 3.0f)
            / cloud_lambda * 1.0e6f;
        const float xdc3 = xdc * xdc * xdc;
        const float dcg3 = dcg * dcg * dcg;
        const float dcb = powf(fmaxf(
            0.0f, xdc3 * dcg3 - xdc3 * xdc3), 1.0f / 6.0f);
        const float zeta_term =
            6.25e-6f * xdc * dcb * dcb * dcb - 0.4f;
        const float zeta1 = 0.5f * (zeta_term + fabsf(zeta_term));
        const float zeta = 0.027f * cloud_mass * zeta1;
        const float tau_diameter = 0.5f * dcb - 7.5f;
        const float taud = 0.5f
            * (tau_diameter + fabsf(tau_diameter)) + 1.0e-12f;
        const float tau = 3.72f / (cloud_mass * taud);
        autoconversion_rate = fmin(
            (double)(cloud_mass * inverse_dt), (double)(zeta / tau));
        autoconversion_number_rate = autoconversion_rate
            / (double)(am_r * 12.0f * 10.0f
                       * 50.0e-6f * 50.0e-6f * 50.0e-6f);
    }

    double rain_self_number_rate = 0.0;
    double rain_cloud_rate = 0.0;
    if (rain_active) {
        if (rain_mvd > 50.0e-6f) {
            const float efficiency = 1.0f
                - expf(2300.0f * (rain_mvd - 1950.0e-6f));
            rain_self_number_rate = (double)(
                efficiency * 2.0f * rain_number * rain_mass);
        }
        if (l_qc && rain_mvd > 50.0e-6f
                && cloud_mvd > 1.0e-6f) {
            const double first = 5.1164649614037726e-05;
            const double last = 0.004886186104779057;
            int rain_bin = 1 + (int)(100.0
                * log((double)rain_mvd / first) / log(last / first));
            rain_bin = min(rain_bin, 100);
            const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
            const float efficiency = (float)rain_cloud_efficiency[
                (rain_bin - 1) + 100 * (cloud_bin - 1)];
            const float coefficient = pi * 0.25f * 4854.0f * 6.0f;
            const float intercept = rain_number * (float)rain_lambda;
            rain_cloud_rate = fmin(
                (double)(cloud_mass * inverse_dt),
                (double)(rho_factor * coefficient * efficiency
                    * cloud_mass * intercept
                    * powf((float)rain_lambda + 195.0f, -4.0f)));
        }
    }

    // Snow/graupel collection of cloud water is also outside WRF's
    // temperature branch.  It must therefore accompany the warm path; the
    // old adapter only executed its cold-kernel implementation.
    double snow_cloud_rate = 0.0;
    float snow_second_moment = 0.0f;
    float snow_number = 0.0f;
    float snow_first_moment = 0.0f;
    float snow_ventilation_moment = 0.0f;
    if (l_qs) {
        const float tc0 = fminf(-0.1f, tempc);
        snow_second_moment = snow_mass * (1.0f / 0.069f);
        snow_number = thompson_field_a(tc0, 0.0f)
            * powf(snow_second_moment, thompson_field_b(tc0, 0.0f));
        snow_first_moment = thompson_field_a(tc0, 1.0f)
            * powf(snow_second_moment, thompson_field_b(tc0, 1.0f));
        snow_ventilation_moment = thompson_field_a(tc0, 1.775f)
            * powf(snow_second_moment, thompson_field_b(tc0, 1.775f));
        const float snow_moment_3 = thompson_field_a(tc0, 3.0f)
            * powf(snow_second_moment, thompson_field_b(tc0, 3.0f));
        const float snow_diameter = snow_moment_3 / snow_second_moment;
        if (l_qc && cloud_mvd > 1.0e-6f
                && snow_diameter > 300.0e-6f) {
            const double diameter_ratio = 0.02 / 300.0e-6;
            const double log_ratio = log(diameter_ratio);
            const double first = 300.0e-6
                * exp(0.5 / 100.0 * log_ratio);
            const double last = 300.0e-6
                * exp(99.5 / 100.0 * log_ratio);
            int snow_bin = 1 + (int)(100.0
                * log((double)snow_diameter / first) / log(last / first));
            snow_bin = max(1, min(snow_bin, 100));
            const int cloud_bin = max(1, min(
                (int)(cloud_mvd * 1.0e6f), 100));
            const float efficiency = (float)snow_cloud_efficiency[
                (snow_bin - 1) + 100 * (cloud_bin - 1)];
            const float moment = thompson_field_a(tc0, 2.55f)
                * powf(snow_second_moment,
                       thompson_field_b(tc0, 2.55f));
            snow_cloud_rate = fmin(
                (double)(cloud_mass * inverse_dt),
                (double)(rho_factor * pi * 0.25f * 40.0f
                    * efficiency * cloud_mass * moment));
        }
    }

    double graupel_cloud_rate = 0.0;
    if (l_qc && cloud_mvd > 1.0e-6f
            && graupel_mass >= 1.0e-6f) {
        const double inverse_lambda = 1.0 / graupel_lambda;
        const double intercept = (double)graupel_number * graupel_lambda;
        const float viscosity = (1.718f + 0.0049f * tempc) * 1.0e-5f;
        const float diameter = (float)(4.0 * inverse_lambda);
        const float velocity = (float)(
            (double)(rho_factor * 442.0f * 20.3632278f * (1.0f / 6.0f))
            * pow(inverse_lambda, 0.89));
        const float stokes = cloud_mvd * cloud_mvd * velocity * 1000.0f
            / (9.0f * viscosity * diameter);
        float efficiency = 0.0f;
        if (stokes >= 0.4f && stokes <= 10.0f) {
            efficiency = 0.55f * log10f(2.51f * stokes);
        } else if (stokes > 10.0f) {
            efficiency = 0.77f;
        }
        if (wet_bulb > 273.15f) efficiency *= 0.1f;
        const float prefactor = pi * 0.25f * 442.0f * 5.23476267f;
        // WRF deliberately does not cap prg_gcw by itself.  It diagnoses
        // the raw collection rate here and scales it together with every
        // other cloud-water sink in the later cloud conservation pass.
        // Pre-capping this term changes that competition, routing cloud
        // water away from graupel and toward rain.
        graupel_cloud_rate =
            (double)(rho_factor * prefactor * efficiency * cloud_mass)
                * intercept * pow(inverse_lambda, 3.89);
    }

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
        const int rain_bin = thompson_decade_table_index(
            rain_mass, -6, 37);
        const double rain_intercept = (double)(__fdiv_rn((1.0f / 6.0f)
            * rain_mass, am_r)) * rain_lambda * rain_lambda
            * rain_lambda * rain_lambda;
        const int rain_intercept_bin =
            thompson_decade_table_index_double(rain_intercept, 6, 37);
        if (snow_mass >= 1.0e-6f) {
            const int snow_bin = thompson_decade_table_index(
                snow_mass, -6, 37);
            const int raw_temp_bin = (int)(__fdiv_rn((tempc - 2.5f), 5.0f)) - 1;
            const int temp_bin = min(9, max(1, -raw_temp_bin)) - 1;
            const size_t table_idx = (size_t)snow_bin
                + (size_t)37 * ((size_t)temp_bin
                + (size_t)9 * ((size_t)rain_intercept_bin
                + (size_t)37 * (size_t)rain_bin));
            if (wet_bulb < 273.15f) {
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
            }
        }
        if (graupel_mass >= 1.0e-6f) {
            const int graupel_mass_bin = thompson_decade_table_index(
                graupel_mass, -6, 37);
            const double intercept = (double)graupel_number
                * graupel_lambda;
            const int graupel_intercept_bin =
                thompson_decade_table_index_double(intercept, 2, 37);
            // One slab, never WRF's idx_bg1=5 subscript: thompson_racg_index.
            const size_t table_idx = thompson_racg_index(
                graupel_intercept_bin, graupel_mass_bin,
                rain_intercept_bin, rain_bin);
            if (wet_bulb < 273.15f) {
                rain_graupel_graupel_rate = fmin(
                    (double)(rain_mass * inverse_dt),
                    tmr_racg[table_idx] + tcr_gacr[table_idx]);
                rain_graupel_rain_rate =
                    -rain_graupel_graupel_rate;
                rain_graupel_rain_number_rate = fmin(
                    (double)(rain_number * inverse_dt),
                    tnr_racg[table_idx] + tnr_gacr[table_idx]);
            } else {
                rain_graupel_rain_rate = fmin(
                    (double)(graupel_mass * inverse_dt),
                    tcg_racg[table_idx]);
                rain_graupel_graupel_rate =
                    -rain_graupel_rain_rate;
                rain_graupel_number_rate = fmin(
                    (double)(graupel_number * inverse_dt),
                    tnr_racg[table_idx]);
                // pnr_rcg is deliberately negative in this inverse
                // branch: subtracting it adds breakup drops.
                rain_graupel_rain_number_rate =
                    -1.5 * tnr_gacr[table_idx];
            }
        }
    }

    const float diffusivity = 2.11e-5f
        * powf(__fdiv_rn(temp0, 273.15f), 1.94f) * (101325.0f / pressure0);
    const float viscosity = (1.718f + 0.0049f * tempc) * 1.0e-5f;
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float rho_factor_sqrt = sqrtf(rho_factor);
    const float viscosity_factor = sqrtf(rho / viscosity);
    const float vapor_deficit = fmaxf(
        0.0f, thompson_rslf(pressure0, 273.15f) - qv0);
    const float inverse_fusion = 1.0f / 334000.0f;
    const float schmidt_cuberoot = powf(0.632f, 1.0f / 3.0f);

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
                * (double)powf(
                    10.0f, -0.25f * (wet_bulb - 273.15f));
        }
    }

    double graupel_melt_rate = 0.0;
    double graupel_melt_number_rate = 0.0;
    if (l_qg) {
        const double inverse_lambda = 1.0 / graupel_lambda;
        const double graupel_intercept =
            (double)graupel_number * graupel_lambda;
        double melt_intercept = graupel_intercept;
        if (graupel_mass * graupel_number < 1.0e-4f) {
            melt_intercept = (double)(1.0e-4f / graupel_mass)
                * graupel_lambda;
        }
        const double melt_moment = melt_intercept * (
            (double)(pi * 4.0f * 0.5f * inverse_fusion * 0.86f)
                * inverse_lambda * inverse_lambda
            + (double)(pi * 4.0f * 0.5f * inverse_fusion * 0.28f
                       * schmidt_cuberoot * sqrtf(442.0f)
                       * 1.9021706581115723f)
                * (double)(rho_factor_sqrt * viscosity_factor)
                * pow(inverse_lambda, 2.945));
        graupel_melt_rate = (double)(tempc * conductivity
            - 2.5e6f * diffusivity * vapor_deficit) * melt_moment;
        graupel_melt_rate = fmin(
            (double)(graupel_mass * inverse_dt),
            fmax(0.0, graupel_melt_rate));
        if (graupel_melt_rate > 0.0) {
            graupel_melt_number_rate = graupel_melt_rate
                * (double)graupel_number / (double)graupel_mass
                * (double)powf(
                    10.0f, -0.33f * (wet_bulb - 273.15f));
        }
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
    if (ssati < 0.0f
            && (snow_melt_rate <= 0.0 || graupel_melt_rate <= 0.0)) {
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
        if (l_qs && snow_melt_rate <= 0.0) {
            const float ventilation = 0.28f * schmidt_cuberoot
                * sqrtf(40.0f) * rho_factor_sqrt * viscosity_factor;
            snow_vapor_rate = (double)(0.15f * geometry * diffusivity
                * ssati * saturated_density
                * (0.86f * snow_first_moment
                   + ventilation * snow_ventilation_moment));
            snow_vapor_rate = fmax(
                (double)(-snow_mass * inverse_dt), snow_vapor_rate);
        }
        if (l_qg && graupel_melt_rate <= 0.0) {
            const double inverse_lambda = 1.0 / graupel_lambda;
            const double graupel_intercept =
                (double)graupel_number * graupel_lambda;
            const float ventilation = 0.28f * schmidt_cuberoot
                * sqrtf(442.0f) * 1.9021706581115723f
                * rho_factor_sqrt * viscosity_factor;
            graupel_vapor_rate = (double)(0.5f * geometry * diffusivity
                * ssati * saturated_density) * graupel_intercept
                * (0.86 * pow(inverse_lambda, 2.0)
                   + (double)ventilation * pow(inverse_lambda, 2.945));
            graupel_vapor_rate = fmax(
                (double)(-graupel_mass * inverse_dt),
                graupel_vapor_rate);
            graupel_vapor_number_rate = graupel_vapor_rate
                * (double)graupel_number / (double)graupel_mass;
        }

        const double vapor_sum = snow_vapor_rate + graupel_vapor_rate;
        const double vapor_limit = (double)((qv0 - qvsi) * rho
            * inverse_dt * 0.999f);
        if (vapor_sum < -1.0e-15 && vapor_sum < vapor_limit) {
            const double ratio = vapor_limit / vapor_sum;
            snow_vapor_rate *= ratio;
            graupel_vapor_rate *= ratio;
            // WRF deliberately leaves png_gde held when this shared vapor
            // limiter rescales prg_gde.
        }
    }

    if (dt > 120.0f) {
        // WRF's adaptive-step safeguard converts warm cloud riming directly
        // to rain instead of leaving liquid mass on frozen categories.
        rain_cloud_rate += snow_cloud_rate + graupel_cloud_rate;
        snow_cloud_rate = 0.0;
        graupel_cloud_rate = 0.0;
    }

    // WRF conservation order: cloud, rain, snow, graupel, then restore the
    // paired collision transfers.  Number rates intentionally remain held
    // when a mass limiter scales its corresponding process.
    double cloud_sink = autoconversion_rate + rain_cloud_rate
        + snow_cloud_rate + graupel_cloud_rate;
    if (cloud_sink > (double)(cloud_mass * inverse_dt)
            && cloud_sink > 0.0) {
        const double ratio = (double)(cloud_mass * inverse_dt) / cloud_sink;
        autoconversion_rate *= ratio;
        rain_cloud_rate *= ratio;
        snow_cloud_rate *= ratio;
        graupel_cloud_rate *= ratio;
        cloud_sink = (double)(cloud_mass * inverse_dt);
    }

    double rain_sum = rain_snow_rain_rate + rain_graupel_rain_rate;
    if (rain_active
            && rain_sum < (double)(-rain_mass * inverse_dt)) {
        const double ratio =
            (double)(-rain_mass * inverse_dt) / rain_sum;
        rain_snow_rain_rate *= ratio;
        rain_graupel_rain_rate *= ratio;
    }

    double snow_sum = snow_vapor_rate
        + rain_snow_snow_rate - snow_melt_rate;
    if (l_qs
            && snow_sum < (double)(-snow_mass * inverse_dt)) {
        const double ratio = (double)(-snow_mass * inverse_dt) / snow_sum;
        snow_vapor_rate *= ratio;
        rain_snow_snow_rate *= ratio;
        snow_melt_rate *= ratio;
    }
    double graupel_sum =
        graupel_vapor_rate + rain_graupel_graupel_rate
        - graupel_melt_rate;
    if (l_qg
            && graupel_sum < (double)(-graupel_mass * inverse_dt)) {
        const double ratio =
            (double)(-graupel_mass * inverse_dt) / graupel_sum;
        graupel_vapor_rate *= ratio;
        rain_graupel_graupel_rate *= ratio;
        graupel_melt_rate *= ratio;
    }
    double paired;
    if (wet_bulb > 273.15f) {
        paired = fmin(
            fabs(rain_snow_rain_rate), fabs(rain_snow_snow_rate));
        rain_snow_rain_rate = copysign(paired, rain_snow_rain_rate);
        rain_snow_snow_rate = -rain_snow_rain_rate;
    }
    paired = fmin(
        fabs(rain_graupel_rain_rate),
        fabs(rain_graupel_graupel_rate));
    rain_graupel_rain_rate = copysign(
        paired, rain_graupel_rain_rate);
    rain_graupel_graupel_rate = -rain_graupel_rain_rate;

    // Preserve WRF's two independent post-conservation branch decisions.
    // Snow fallout blends with the rain velocity only for actual same-call
    // snow melt; rain merely coexisting with snow is not sufficient.
    snow_melt_marker[idx] = snow_melt_rate > 0.0 ? 1.0f : 0.0f;
    graupel_melt_marker[idx] = graupel_melt_rate > 0.0 ? 1.0f : 0.0f;

    const double rain_rate = autoconversion_rate + rain_cloud_rate
        + snow_melt_rate + graupel_melt_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
    const double rain_number_rate = autoconversion_number_rate
        + snow_melt_number_rate + graupel_melt_number_rate
        - rain_self_number_rate - rain_snow_number_rate
        - rain_graupel_rain_number_rate;
    const double snow_rate = snow_vapor_rate
        + snow_cloud_rate + rain_snow_snow_rate - snow_melt_rate;
    const double graupel_rate = graupel_vapor_rate + graupel_cloud_rate
        + rain_snow_graupel_rate + rain_graupel_graupel_rate
        - graupel_melt_rate;
    const double graupel_number_rate = graupel_vapor_number_rate
        + rain_snow_number_rate - rain_graupel_number_rate
        - graupel_melt_number_rate;

    // Entry rain and graupel as WRF holds them after :1878-1905 and
    // :1915-1949 (zero where the mixing ratio is at or below R1), for the
    // source-stage removals at :3088-3091 and :3157-3159 below.
    const float qr_entry_wrf = qr[idx] > 1.0e-12f ? qr[idx] : 0.0f;
    const float nr_entry_wrf = qr[idx] > 1.0e-12f ? nr[idx] : 0.0f;
    const float qg_entry_wrf = qg[idx] > 1.0e-12f ? qg[idx] : 0.0f;
    const float ng_entry_wrf = qg[idx] > 1.0e-12f
        ? graupel_number_per_kg[idx] : 0.0f;
    qc[idx] = fmaxf(0.0f,
        qc[idx] - (float)(cloud_sink * (double)orho) * dt);
    qr[idx] = fmaxf(0.0f,
        qr[idx] + (float)(rain_rate * (double)orho) * dt);
    nr[idx] = fmaxf(0.0f,
        nr[idx] + (float)(rain_number_rate * (double)orho) * dt);
    qs[idx] = fmaxf(0.0f,
        qs[idx] + (float)(snow_rate * (double)orho) * dt);
    qg[idx] = fmaxf(0.0f,
        qg[idx] + (float)(graupel_rate * (double)orho) * dt);
    // Keep classic ng1d raw through source and fallout tendencies.  WRF
    // diagnoses a separate qg-based number for fallout velocity and applies
    // the private moment's sole size bound only in the final state pass.
    graupel_number_per_kg[idx] +=
        (float)(graupel_number_rate * (double)orho) * dt;
    const double vapor_rate = snow_vapor_rate + graupel_vapor_rate;
    // The entry vapour plus the tendency, unfloored: WRF floors the running
    // vapour once, at the terminal apply (:3974).  Adding the tendency to the
    // floored working value put 1.E-10 kg/kg into every level that entered
    // at zero, and floored levels WRF returns untouched at :2020.
    qv[idx] = qv[idx] - (float)(vapor_rate * (double)orho) * dt;
    thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
    // :3067-3091.  Where the post-source rain concentration is at or below R1
    // WRF discards the call's rain sources and removes the entry rain:
    // qrten = -qr1d*odts, nrten = -nr1d*odts.  The number bound above zeroes
    // only the number; the mass has to go too.
    if (!(qr[idx] * rho > 1.0e-12f)) {
        qr[idx] = qr_entry_wrf + (-qr_entry_wrf * inverse_dt) * dt;
        nr[idx] = nr_entry_wrf + (-nr_entry_wrf * inverse_dt) * dt;
    }
    // :3118-3160, the graupel mass/number balance, which WRF runs for
    // non-hail-aware Thompson too.  At or below R1 the graupel and its
    // private number are removed; otherwise the private number is reset so
    // the median volume diameter stays in [D0r, 25.4 mm].  The fallout and
    // calc_refl10cm read that number (the mp=28 source networks carry the
    // same balance).
    {
        const float xrg = qg[idx] * rho;
        if (!(xrg > 1.0e-12f)) {
            qg[idx] = qg_entry_wrf + (-qg_entry_wrf * inverse_dt) * dt;
            graupel_number_per_kg[idx] =
                ng_entry_wrf + (-ng_entry_wrf * inverse_dt) * dt;
        } else {
            // REAL(4) throughout except lamg, which is DOUBLE (:1597).
            const float am_g = 3.1415926536f * 400.0f / 6.0f;
            const float mvd_numerator = 3.0f + 0.0f + 0.672f;
            const float xng = fmaxf(1.0e-6f, graupel_number_per_kg[idx] * rho);
            double lamg = (double)powf(
                am_g * 6.0f * xng / xrg, 1.0f / 3.0f);
            const float mvd_g = (float)((double)mvd_numerator / lamg);
            bool bounded = false;
            if (mvd_g > 25.4e-3f) {
                lamg = (double)(mvd_numerator / 25.4e-3f);
                bounded = true;
            } else if (mvd_g < 50.0e-6f) {
                lamg = (double)(mvd_numerator / 50.0e-6f);
                bounded = true;
            }
            if (bounded) {
                const float xng_bounded = (float)(
                    (double)((1.0f / 6.0f) * xrg) * pow(lamg, 3.0)
                    / (double)am_g);
                const float ngten = (xng_bounded - ng_entry_wrf * rho)
                    * inverse_dt * (1.0f / rho);
                graupel_number_per_kg[idx] = ng_entry_wrf + ngten * dt;
            }
        }
    }

    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    const double fusion_rate = -snow_melt_rate - graupel_melt_rate
        - rain_snow_rain_rate - rain_graupel_rain_rate;
    temperature[idx] = temp0 + (float)(
        ((334000.0 * fusion_rate) + (2.834e6 * vapor_rate))
        * (double)inverse_cp * (double)orho * (double)dt);
}

__device__ __forceinline__ void thompson_bound_ice_number(
    float ice_mass, float density, float* ice_number_per_kg)
{
    if (ice_mass <= 1.0e-12f) {
        *ice_number_per_kg = 0.0f;
        return;
    }
    const float am_i = 3.1415926536f * 890.0f / 6.0f;
    float ice_number = fmaxf(1.0e-6f,
                              *ice_number_per_kg * density);
    double lambda = (double)powf(
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

__device__ __forceinline__ void thompson_final_phase_cleanup_impl(
    float* __restrict__ qc,
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    const float* __restrict__ micro_columns,
    int ncol, int idx)
{
    // module_mp_thompson.F performs these two instantaneous transfers after
    // every fallout tendency and before the final category bounds.  Classic
    // mp=8 has fixed cloud number, so only the carried ice number is exposed.
    if (micro_columns != nullptr) {
        // :2020.  A column WRF left before the source loop reaches neither
        // the phase cleanup nor the terminal apply: it returns its entry
        // rewrite and nothing else, which for cloud and ice is zero at every
        // level (all of them entered at or below R1, :1844-1845,
        // :1870-1871), and its vapour unfloored.
        if (micro_columns[idx % ncol] == 0.0f) {
            if (qc[idx] <= 1.0e-12f) qc[idx] = 0.0f;
            if (qi[idx] <= 1.0e-12f) {
                qi[idx] = 0.0f;
                ni[idx] = 0.0f;
            }
            return;
        }
        // :3974.  Every level of a column with microphysics leaves with its
        // vapour floored at 1.E-10; the kernels before carry it unfloored.
        qv[idx] = fmaxf(1.0e-10f, qv[idx]);
    }

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));

    if (temp0 > 273.15f && qi[idx] > 0.0f) {
        const float transferred = fmaxf(0.0f, qi[idx]);
        qc[idx] += transferred;
        qi[idx] = 0.0f;
        ni[idx] = 0.0f;
        temperature[idx] -= 334000.0f * inverse_cp * transferred;
    } else if (temp0 < 235.16f && qc[idx] > 0.0f) {
        const float transferred = fmaxf(0.0f, qc[idx]);
        const float latent_vapor = 2.5e6f
            + (2106.0f - 4218.0f) * (temp0 - 273.15f);
        const float latent_fusion = 2.834e6f - latent_vapor;
        qc[idx] = 0.0f;
        qi[idx] += transferred;
        ni[idx] += 100.0e6f / rho;
        temperature[idx] += latent_fusion * inverse_cp * transferred;
    }

    if (qc[idx] <= 1.0e-12f) qc[idx] = 0.0f;
    if (qi[idx] <= 1.0e-12f) {
        qi[idx] = 0.0f;
        ni[idx] = 0.0f;
    } else if (qi[idx] * rho > 1.0e-12f) {
        thompson_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
    } else {
        // :4025-4039 tests the MIXING RATIO and keeps both mass and number.
        // The shared bound tests the concentration (right for the source
        // stage at :3036-3055, wrong here) and zeroed the number of ice that
        // sits in thin air aloft while keeping its mass.  WRF's per-kilogram
        // form (the mp=28 phase cleanup carries the same rule):
        const float am_i = 3.1415926536f * 890.0f / 6.0f;
        const float qi_local = qi[idx];
        const float ni_local = fmaxf(1.0e-6f / rho, ni[idx]);
        double lami = (double)powf(
            am_i * 6.0f * ni_local / qi_local, 1.0f / 3.0f);
        const float xdi = (float)(4.0 * (1.0 / lami));
        if (xdi < 5.0e-6f) {
            lami = (double)(4.0f / 5.0e-6f);
        } else if (xdi > 300.0e-6f) {
            lami = (double)(4.0f / 300.0e-6f);
        }
        ni[idx] = (float)fmin(
            (double)(__fdiv_rn((1.0f / 6.0f) * qi_local, am_i)) * pow(lami, 3.0),
            999.0e3 / (double)rho);
    }
}

extern "C" __global__ void thompson_final_phase_cleanup(
    float* __restrict__ qc,
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    thompson_final_phase_cleanup_impl(
        qc, qi, ni, temperature, pressure, (float*)qv, nullptr, 1, idx);
}

// The production entry point: the cleanup of a column with microphysics
// with WRF's terminal vapour floor (:3974), and only the entry rewrite's
// zeros in a column WRF returned from at :2020 (micro_columns == 0, from
// thompson_microphysics_columns).
extern "C" __global__ void thompson_final_phase_cleanup_with_columns(
    float* __restrict__ qc,
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    const float* __restrict__ micro_columns,
    int ncol, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    thompson_final_phase_cleanup_impl(
        qc, qi, ni, temperature, pressure, qv, micro_columns, ncol, idx);
}

extern "C" __global__ void thompson_rain_freezing(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ qg,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const double* __restrict__ rain_to_ice_mass,
    const double* __restrict__ rain_to_ice_number,
    const double* __restrict__ rain_to_graupel_mass,
    const double* __restrict__ rain_to_graupel_number,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qr[idx] <= 1.0e-12f) return;

    const float temp0 = temperature[idx];
    if (temp0 >= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float rain_mass0 = qr[idx] * rho;
    float rain_number0 = fmaxf(1.0e-6f, nr[idx] * rho);

    double ice_mass_amount = 0.0;
    double ice_number_amount = 0.0;
    double graupel_mass_amount = 0.0;
    double graupel_number_amount = 0.0;

    if (rain_mass0 > 1.0e-6f) {
        const int mass_bin = thompson_decade_table_index(
            rain_mass0, -6, 37);
        const float am_r = 3.1415926536f * 1000.0f / 6.0f;
        const double lambda = (double)powf(
            am_r * 6.0f * rain_number0 / rain_mass0, 1.0f / 3.0f);
        // crg(3)*org2*org1 is exactly one for classic mu_r=0, so
        // N0_exp reduces to this expression while retaining WRF's mixed
        // REAL/DOUBLE evaluation order.
        const float mass_coefficient = __fdiv_rn((1.0f / 6.0f)
            * rain_mass0, am_r);
        const double intercept = (double)mass_coefficient
            * lambda * lambda * lambda * lambda;
        const int intercept_bin = thompson_decade_table_index_double(
            intercept, 6, 37);
        const int temp_bin = max(0, min(
            (int)roundf(-(temp0 - 273.15f)) - 1, 44));
        // Classic non-aerosol Thompson fixes available ice nuclei to
        // 1000 m^-3, which maps to one-based bin 28.
        const int nuclei_bin = 27;
        const size_t table_idx = (size_t)mass_bin
            + (size_t)37 * ((size_t)intercept_bin
            + (size_t)37 * ((size_t)temp_bin
            + (size_t)45 * (size_t)nuclei_bin));
        ice_mass_amount = rain_to_ice_mass[table_idx];
        ice_number_amount = rain_to_ice_number[table_idx];
        graupel_mass_amount = rain_to_graupel_mass[table_idx];
        graupel_number_amount = fmin(
            (double)rain_number0, rain_to_graupel_number[table_idx]);

        const double total_mass = ice_mass_amount + graupel_mass_amount;
        if (total_mass > (double)rain_mass0) {
            const double ratio = (double)rain_mass0 / total_mass;
            ice_mass_amount *= ratio;
            graupel_mass_amount *= ratio;
        }
    } else if (rain_mass0 > 1.0e-12f && temp0 < 235.16f) {
        // WRF instantaneously converts sub-table supercooled rain to cloud
        // ice below its homogeneous-freezing threshold.
        ice_mass_amount = (double)rain_mass0;
        ice_number_amount = (double)rain_number0;
    } else {
        return;
    }

    const double total_mass_amount = ice_mass_amount
        + graupel_mass_amount;
    const double total_rain_number_loss = graupel_number_amount
        + ice_number_amount;
    const float rain_mass = fmaxf(
        0.0f, rain_mass0 - (float)total_mass_amount);
    const float rain_number = fmaxf(
        0.0f, rain_number0 - (float)total_rain_number_loss);

    qr[idx] = rain_mass * orho;
    nr[idx] = rain_number * orho;
    thompson_bound_rain_number(rain_mass, rho, &nr[idx]);

    qi[idx] = fmaxf(0.0f,
        qi[idx] + (float)(ice_mass_amount * (double)orho));
    ni[idx] = fmaxf(0.0f,
        ni[idx] + (float)(ice_number_amount * (double)orho));
    thompson_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
    qg[idx] = fmaxf(0.0f,
        qg[idx] + (float)(graupel_mass_amount * (double)orho));

    const float tempc = temp0 - 273.15f;
    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    const float frozen_tendency = (float)(
        total_mass_amount * (double)orho / (double)dt);
    temperature[idx] = temp0 + latent_fusion * inverse_cp
        * frozen_tendency * dt;
}

extern "C" __global__ void thompson_cloud_freezing(
    float* __restrict__ qc,
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const double* __restrict__ cloud_to_ice_mass,
    const double* __restrict__ cloud_to_ice_number,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qc[idx] <= 1.0e-12f) return;

    const float temp0 = temperature[idx];
    if (temp0 >= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float cloud_mass0 = qc[idx] * rho;

    double ice_mass_amount = 0.0;
    double ice_number_amount = 0.0;
    if (cloud_mass0 > 1.0e-6f) {
        const int mass_bin = thompson_decade_table_index(
            cloud_mass0, -6, 37);
        // Classic mp=8 fixes cloud number to 100e6 m^-3.  WRF stores nic1
        // as INTEGER, truncating its logarithmic span from 7.926 to 7; the
        // resulting legacy lookup is one-based bin 66 (not bin 59).
        const int cloud_number_bin = 65;
        const int temp_bin = max(0, min(
            (int)roundf(-(temp0 - 273.15f)) - 1, 44));
        const int nuclei_bin = 27;
        const size_t table_idx = (size_t)mass_bin
            + (size_t)37 * ((size_t)cloud_number_bin
            + (size_t)100 * ((size_t)temp_bin
            + (size_t)45 * (size_t)nuclei_bin));
        ice_mass_amount = fmin(
            (double)cloud_mass0, cloud_to_ice_mass[table_idx]);
        ice_number_amount = fmin(
            100.0e6,
            fmin(ice_mass_amount / (2.0 * 1.0e-12),
                 cloud_to_ice_number[table_idx]));
    } else if (cloud_mass0 > 1.0e-12f && temp0 < 235.16f) {
        ice_mass_amount = (double)cloud_mass0;
        ice_number_amount = 100.0e6;
    } else {
        return;
    }

    const float mass_tendency = (float)(
        ice_mass_amount * (double)orho / (double)dt);
    const float number_tendency = (float)(
        ice_number_amount * (double)orho / (double)dt);
    qc[idx] = fmaxf(0.0f, qc[idx] - mass_tendency * dt);
    qi[idx] = fmaxf(0.0f, qi[idx] + mass_tendency * dt);
    ni[idx] = fmaxf(0.0f, ni[idx] + number_tendency * dt);
    thompson_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);

    const float tempc = temp0 - 273.15f;
    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    temperature[idx] = temp0 + latent_fusion * inverse_cp
        * mass_tendency * dt;
}

extern "C" __global__ void thompson_graupel_cloud_riming(
    float* __restrict__ qc,
    float* __restrict__ qg,
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float dt, int size)
{
    // Classic WRF Thompson's cold graupel-cloud collection plus its
    // Hallett-Mossop rime-splinter branch.  This admitted slice deliberately
    // leaves warm/wet-growth collection to a later oracle milestone.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qc[idx] <= 1.0e-12f
            || qg[idx] <= 1.0e-12f) return;

    const float temp0 = temperature[idx];
    if (temp0 >= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float inverse_rho = 1.0f / rho;
    const float cloud_mass = qc[idx] * rho;
    const float graupel_mass = qg[idx] * rho;
    if (cloud_mass <= 1.0e-12f || graupel_mass <= 1.0e-12f) return;

    const float pi = 3.1415926536f;
    const float water_mass_coefficient = pi * 1000.0f / 6.0f;
    // Non-aerosol mp=8 fixes Nc=100e6 m^-3.  That selects nu_c=12,
    // for which Gamma(nu_c+4)/Gamma(nu_c+1)=2730.
    const float cloud_lambda = powf(
        100.0e6f * water_mass_coefficient * 2730.0f / cloud_mass,
        1.0f / 3.0f);
    float cloud_mvd = (15.672 / (double)cloud_lambda);
    cloud_mvd = fmaxf(1.0e-6f, fminf(cloud_mvd, 50.0e-6f));
    if (cloud_mvd <= 1.0e-6f) return;

    // Classic mp=8 diagnoses graupel number from mass with the Field-style
    // intercept and fixes bulk density to 400 kg m^-3 (idx_bg1=5).
    const float graupel_mass_coefficient = pi * 400.0f / 6.0f;
    const float intercept_power = fmaxf(2.0f, fminf(6.0f,
        3.0f + (2.0f / 7.0f)
            * (log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f)));
    const float intercept_guess = powf(10.0f, intercept_power);
    double graupel_lambda = (double)powf(
        intercept_guess * graupel_mass_coefficient * 6.0f
            / graupel_mass,
        0.25f);
    float graupel_mvd = (float)(3.672 / graupel_lambda);
    if (graupel_mvd > 25.4e-3f) {
        graupel_mvd = 25.4e-3f;
        graupel_lambda = 3.672 / 25.4e-3;
    } else if (graupel_mvd < 50.0e-6f) {
        graupel_mvd = 50.0e-6f;
        graupel_lambda = 3.672 / 50.0e-6;
    }
    const double inverse_lambda = 1.0 / graupel_lambda;
    const float graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass,
        graupel_mass_coefficient)
        * (float)(graupel_lambda * graupel_lambda * graupel_lambda);
    const double graupel_intercept = (double)graupel_number
        * graupel_lambda;

    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const float density_factor = sqrtf(rho_not / rho);
    const float tempc = temp0 - 273.15f;
    const float viscosity = (1.718f + 0.0049f * tempc
        - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    const float graupel_diameter = (float)(4.0 * inverse_lambda);
    const float graupel_velocity = (float)(
        (double)(density_factor * 442.0f * 20.3632278f * (1.0f / 6.0f))
        * pow(inverse_lambda, 0.89));
    const float stokes_number = cloud_mvd * cloud_mvd
        * graupel_velocity * 1000.0f
        / (9.0f * viscosity * graupel_diameter);
    float collection_efficiency;
    if (stokes_number < 0.4f) {
        collection_efficiency = 0.0f;
    } else if (stokes_number <= 10.0f) {
        collection_efficiency = 0.55f
            * log10f(2.51f * stokes_number);
    } else {
        collection_efficiency = 0.77f;
    }
    if (collection_efficiency <= 0.0f) return;

    const float collection_prefactor = pi * 0.25f
        * 442.0f * 5.23476267f;
    double riming_rate = (double)(density_factor * collection_prefactor
        * collection_efficiency * cloud_mass)
        * graupel_intercept * pow(inverse_lambda, 3.89);
    riming_rate = fmin(riming_rate, (double)cloud_mass / (double)dt);
    if (riming_rate <= 0.0) return;

    float hm_temperature_factor = 0.0f;
    if (tempc >= -5.0f && tempc < -3.0f) {
        hm_temperature_factor = 0.5f * (-3.0f - tempc);
    } else if (tempc > -8.0f && tempc < -5.0f) {
        hm_temperature_factor = 0.33333333f * (8.0f + tempc);
    }
    const double splinter_number_rate = 3.5e8
        * (double)hm_temperature_factor * riming_rate;
    const double splinter_mass_rate = 1.0e-12
        * splinter_number_rate;
    const double graupel_gain_rate = riming_rate - splinter_mass_rate;
    const float rimed_mixing = (float)(
        riming_rate * (double)inverse_rho * (double)dt);

    qc[idx] = fmaxf(0.0f, qc[idx] - rimed_mixing);
    qg[idx] = fmaxf(0.0f, qg[idx] + (float)(
        graupel_gain_rate * (double)inverse_rho * (double)dt));
    qi[idx] = fmaxf(0.0f, qi[idx] + (float)(
        splinter_mass_rate * (double)inverse_rho * (double)dt));
    ni[idx] = fmaxf(0.0f, ni[idx] + (float)(
        splinter_number_rate * (double)inverse_rho * (double)dt));
    thompson_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);

    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    temperature[idx] = temp0
        + latent_fusion * inverse_cp * rimed_mixing;
}

extern "C" __global__ void thompson_snow_cloud_riming(
    float* __restrict__ qc,
    float* __restrict__ qs,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float dt, int size)
{
    // Classic WRF Thompson snow-cloud collection (Wang-Ji efficiency).
    // This admitted slice excludes the separate deposition-conditioned
    // partial snow-to-graupel conversion branch.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qc[idx] <= 1.0e-12f
            || qs[idx] <= 1.0e-12f) return;

    const float temp0 = temperature[idx];
    if (temp0 >= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float inverse_rho = 1.0f / rho;
    const float cloud_mass = qc[idx] * rho;
    const float snow_mass = qs[idx] * rho;
    if (cloud_mass <= 1.0e-12f || snow_mass <= 1.0e-12f) return;

    const float pi = 3.1415926536f;
    const float water_mass_coefficient = pi * 1000.0f / 6.0f;
    const float cloud_lambda = powf(
        100.0e6f * water_mass_coefficient * 2730.0f / cloud_mass,
        1.0f / 3.0f);
    float cloud_mvd = (15.672 / (double)cloud_lambda);
    cloud_mvd = fmaxf(1.0e-6f, fminf(cloud_mvd, 50.0e-6f));
    if (cloud_mvd <= 1.0e-6f) return;

    const float snow_temperature = fminf(-0.1f, temp0 - 273.15f);
    const float snow_moment_2 = snow_mass * (1.0f / 0.069f);
    const float snow_moment_3 = thompson_field_a(
        snow_temperature, 3.0f) * powf(
            snow_moment_2, thompson_field_b(snow_temperature, 3.0f));
    const float snow_diameter = snow_moment_3 / snow_moment_2;
    if (snow_diameter <= 300.0e-6f) return;
    const float snow_riming_moment = thompson_field_a(
        snow_temperature, 2.55f) * powf(
            snow_moment_2, thompson_field_b(snow_temperature, 2.55f));

    // Reconstruct the exact logarithmic snow-bin center selected by WRF's
    // 100x100 t_Efsw lookup, then evaluate the same table-generation formula.
    const double diameter_ratio = 0.02 / 300.0e-6;
    const double log_ratio = log(diameter_ratio);
    const double first_snow_bin = 300.0e-6
        * exp(0.5 / 100.0 * log_ratio);
    const double last_snow_bin = 300.0e-6
        * exp(99.5 / 100.0 * log_ratio);
    int snow_bin = 1 + (int)(100.0
        * log((double)snow_diameter / first_snow_bin)
        / log(last_snow_bin / first_snow_bin));
    snow_bin = min(snow_bin, 100);
    const double table_snow_diameter = 300.0e-6
        * exp(((double)snow_bin - 0.5) / 100.0 * log_ratio);
    const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
    const double table_cloud_diameter = (double)cloud_bin * 1.0e-6;

    const double cloud_velocity = 1.19e4
        * (1.0e4 * table_cloud_diameter * table_cloud_diameter * 0.25);
    const double snow_velocity = 40.0 * pow(table_snow_diameter, 0.55)
        * exp(-100.0 * table_snow_diameter) - cloud_velocity;
    const double melted_snow_diameter = pow(
        0.069 * table_snow_diameter * table_snow_diameter
            / (3.1415926536 * 1000.0 / 6.0),
        1.0 / 3.0);
    const double diameter_fraction = table_cloud_diameter
        / melted_snow_diameter;
    float collection_efficiency = 0.0f;
    if (diameter_fraction <= 0.25
            && table_snow_diameter >= 300.0e-6
            && table_cloud_diameter >= (double)6.0e-6f
            && snow_velocity >= (double)1.0e-3f) {
        const double stokes_number = table_cloud_diameter
            * table_cloud_diameter * snow_velocity * 1000.0
            / (9.0 * 1.718e-5 * melted_snow_diameter);
        const double reynolds_number = 9.0 * stokes_number
            / (diameter_fraction * diameter_fraction * 1000.0);
        const double log_reynolds = log(reynolds_number);
        const double k0 = exp(-0.1007 - 0.358 * log_reynolds
            + 0.0261 * log_reynolds * log_reynolds);
        const double z = log(stokes_number / (k0 + 1.0e-15));
        const double h = 0.1465 + 1.302 * z - 0.607 * z * z
            + 0.293 * z * z * z;
        const double yc0 = 2.0 / 3.14159265358979323846 * atan(h);
        const double efficiency = (yc0 + diameter_fraction)
            * (yc0 + diameter_fraction)
            / ((1.0 + diameter_fraction)
               * (1.0 + diameter_fraction));
        collection_efficiency = fmaxf(
            0.0f, fminf((float)efficiency, 0.95f));
    }
    if (collection_efficiency <= 0.0f) return;

    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const float density_factor = sqrtf(rho_not / rho);
    const float collection_prefactor = pi * 0.25f * 40.0f;
    float riming_rate = density_factor * collection_prefactor
        * collection_efficiency * cloud_mass * snow_riming_moment;
    riming_rate = fminf(riming_rate, cloud_mass * (1.0f / dt));
    if (riming_rate <= 0.0f) return;

    const float riming_tendency = (float)(
        (double)riming_rate * (double)inverse_rho);
    const float rimed_mixing = riming_tendency * dt;
    qc[idx] = fmaxf(0.0f, qc[idx] - rimed_mixing);
    qs[idx] = fmaxf(0.0f, qs[idx] + rimed_mixing);

    const float tempc = temp0 - 273.15f;
    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    const float temperature_tendency = (float)(
        (double)(latent_fusion * inverse_cp)
        * (double)riming_rate * (double)inverse_rho);
    temperature[idx] = temp0 + dt * temperature_tendency;
}

extern "C" __global__ void thompson_cold_cloud_source_network(
    float* __restrict__ qc,
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const double* __restrict__ rain_cloud_efficiency,
    const double* __restrict__ cloud_to_ice_mass,
    const double* __restrict__ cloud_to_ice_number,
    float dt, int size)
{
    // Complete cold cloud-water source group.  Liquid autoconversion and
    // accretion, Bigg cloud freezing, snow riming, graupel riming, and
    // Hallett-Mossop splinters all read one incoming cloud state.  The shared
    // WRF cloud-water bound is applied once before any category update.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || temperature[idx] >= 273.15f) return;

    const float temp0 = temperature[idx];
    const float tempc = temp0 - 273.15f;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float cloud_mass = fmaxf(0.0f, qc[idx] * rho);
    const float rain_mass = fmaxf(0.0f, qr[idx] * rho);
    const float snow_mass = fmaxf(0.0f, qs[idx] * rho);
    const float graupel_mass = fmaxf(0.0f, qg[idx] * rho);
    if (cloud_mass <= 1.0e-12f && rain_mass <= 1.0e-12f) return;
    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float cloud_number = 100.0e6f;
    const float gamma_mass = 1.30767389e12f;
    const float inverse_gamma_number = 2.08767448e-9f;
    const float gamma_higher = 6.40238373e15f;
    const float inverse_gamma_mass = 7.64716632e-13f;

    float cloud_lambda = 0.0f;
    float cloud_mvd = 1.0e-6f;
    if (cloud_mass > 1.0e-12f) {
        cloud_lambda = powf(
            cloud_number * am_r * gamma_mass * inverse_gamma_number
                / cloud_mass,
            1.0f / 3.0f);
        cloud_mvd = fmaxf(1.0e-6f, fminf(
            15.672f / cloud_lambda, 50.0e-6f));
    }

    double autoconversion_rate = 0.0;
    double autoconversion_number_rate = 0.0;
    if (cloud_mass > 0.01e-3f) {
        const float xdc = fmaxf(1.0f,
            powf(__fdiv_rn(cloud_mass, am_r * cloud_number), 1.0f / 3.0f)
                * 1.0e6f);
        const float dcg = powf(
            gamma_higher * inverse_gamma_mass, 1.0f / 3.0f)
            / cloud_lambda * 1.0e6f;
        const float xdc3 = xdc * xdc * xdc;
        const float dcg3 = dcg * dcg * dcg;
        const float dcb = powf(fmaxf(
            0.0f, xdc3 * dcg3 - xdc3 * xdc3), 1.0f / 6.0f);
        const float zeta_term =
            6.25e-6f * xdc * dcb * dcb * dcb - 0.4f;
        const float zeta1 = 0.5f * (zeta_term + fabsf(zeta_term));
        const float zeta = 0.027f * cloud_mass * zeta1;
        const float tau_diameter = 0.5f * dcb - 7.5f;
        const float taud = 0.5f
            * (tau_diameter + fabsf(tau_diameter)) + 1.0e-12f;
        const float tau = 3.72f / (cloud_mass * taud);
        autoconversion_rate = fmin(
            (double)(cloud_mass / dt), (double)(zeta / tau));
        autoconversion_number_rate = autoconversion_rate
            / (double)(am_r * 12.0f * 10.0f
                       * 50.0e-6f * 50.0e-6f * 50.0e-6f);
    }

    double self_number_rate = 0.0;
    double accretion_rate = 0.0;
    if (rain_mass > 1.0e-12f) {
        const float rain_number = fmaxf(1.0e-6f, nr[idx] * rho);
        const float rain_lambda = powf(
            am_r * 6.0f * rain_number / rain_mass, 1.0f / 3.0f);
        const float rain_mvd = 3.672f / rain_lambda;
        if (rain_mvd > 50.0e-6f) {
            const float efficiency = 1.0f
                - expf(2300.0f * (rain_mvd - 1950.0e-6f));
            self_number_rate = (double)(
                efficiency * 2.0f * rain_number * rain_mass);
        }
        if (cloud_mass > 1.0e-12f && rain_mvd > 50.0e-6f
                && cloud_mvd > 1.0e-6f) {
            const double dr_first = 5.1164649614037726e-05;
            const double dr_last = 0.004886186104779057;
            int rain_bin = 1 + (int)(100.0
                * log((double)rain_mvd / dr_first)
                / log(dr_last / dr_first));
            rain_bin = min(rain_bin, 100);
            const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
            const float efficiency = (float)rain_cloud_efficiency[
                (rain_bin - 1) + 100 * (cloud_bin - 1)];
            const float density_factor = sqrtf(
                (101325.0f / (287.05f * 298.0f)) / rho);
            const float coefficient = pi * 0.25f * 4854.0f * 6.0f;
            const float intercept = rain_number * rain_lambda;
            const float rate = density_factor * coefficient * efficiency
                * cloud_mass * intercept
                * powf(rain_lambda + 195.0f, -4.0f);
            accretion_rate = fmin(
                (double)(cloud_mass / dt), (double)rate);
        }
    }

    double cloud_freezing_rate = 0.0;
    double cloud_freezing_number_rate = 0.0;
    if (cloud_mass > 1.0e-6f) {
        const int mass_bin = thompson_decade_table_index(
            cloud_mass, -6, 37);
        const int cloud_number_bin = 65;
        const int temp_bin = max(0, min(
            (int)roundf(-tempc) - 1, 44));
        const int nuclei_bin = 27;
        const size_t table_idx = (size_t)mass_bin
            + (size_t)37 * ((size_t)cloud_number_bin
            + (size_t)100 * ((size_t)temp_bin
            + (size_t)45 * (size_t)nuclei_bin));
        cloud_freezing_rate = fmin(
            (double)(cloud_mass / dt),
            cloud_to_ice_mass[table_idx] / (double)dt);
        cloud_freezing_number_rate = fmin(
            (double)(cloud_number / dt),
            fmin(cloud_freezing_rate / (2.0 * 1.0e-12),
                 cloud_to_ice_number[table_idx] / (double)dt));
    } else if (cloud_mass > 1.0e-12f && temp0 < 235.16f) {
        cloud_freezing_rate = (double)cloud_mass / (double)dt;
        cloud_freezing_number_rate =
            (double)cloud_number / (double)dt;
    }

    double snow_riming_rate = 0.0;
    if (cloud_mass > 1.0e-12f && snow_mass > 1.0e-12f
            && cloud_mvd > 1.0e-6f) {
        const float snow_temperature = fminf(-0.1f, tempc);
        const float snow_moment_2 = snow_mass * (1.0f / 0.069f);
        const float snow_moment_3 = thompson_field_a(
            snow_temperature, 3.0f) * powf(
                snow_moment_2,
                thompson_field_b(snow_temperature, 3.0f));
        const float snow_diameter = snow_moment_3 / snow_moment_2;
        if (snow_diameter > 300.0e-6f) {
            const float snow_riming_moment = thompson_field_a(
                snow_temperature, 2.55f) * powf(
                    snow_moment_2,
                    thompson_field_b(snow_temperature, 2.55f));
            const double diameter_ratio = 0.02 / 300.0e-6;
            const double log_ratio = log(diameter_ratio);
            const double first_snow_bin = 300.0e-6
                * exp(0.5 / 100.0 * log_ratio);
            const double last_snow_bin = 300.0e-6
                * exp(99.5 / 100.0 * log_ratio);
            int snow_bin = 1 + (int)(100.0
                * log((double)snow_diameter / first_snow_bin)
                / log(last_snow_bin / first_snow_bin));
            snow_bin = min(snow_bin, 100);
            const double table_snow_diameter = 300.0e-6
                * exp(((double)snow_bin - 0.5) / 100.0 * log_ratio);
            const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
            const double table_cloud_diameter =
                (double)cloud_bin * 1.0e-6;
            const double cloud_velocity = 1.19e4
                * (1.0e4 * table_cloud_diameter
                   * table_cloud_diameter * 0.25);
            const double snow_velocity = 40.0
                * pow(table_snow_diameter, 0.55)
                * exp(-100.0 * table_snow_diameter) - cloud_velocity;
            const double melted_snow_diameter = pow(
                0.069 * table_snow_diameter * table_snow_diameter
                    / (pi * 1000.0 / 6.0), 1.0 / 3.0);
            const double diameter_fraction = table_cloud_diameter
                / melted_snow_diameter;
            float efficiency = 0.0f;
            if (diameter_fraction <= 0.25
                    && table_cloud_diameter >= (double)6.0e-6f
                    && snow_velocity >= (double)1.0e-3f) {
                const double stokes_number = table_cloud_diameter
                    * table_cloud_diameter * snow_velocity * 1000.0
                    / (9.0 * 1.718e-5 * melted_snow_diameter);
                const double reynolds_number = 9.0 * stokes_number
                    / (diameter_fraction * diameter_fraction * 1000.0);
                const double log_reynolds = log(reynolds_number);
                const double k0 = exp(-0.1007 - 0.358 * log_reynolds
                    + 0.0261 * log_reynolds * log_reynolds);
                const double z = log(stokes_number / (k0 + 1.0e-15));
                const double h = 0.1465 + 1.302 * z - 0.607 * z * z
                    + 0.293 * z * z * z;
                const double yc0 = 2.0 / 3.14159265358979323846
                    * atan(h);
                const double value = (yc0 + diameter_fraction)
                    * (yc0 + diameter_fraction)
                    / ((1.0 + diameter_fraction)
                       * (1.0 + diameter_fraction));
                efficiency = fmaxf(0.0f, fminf((float)value, 0.95f));
            }
            const float density_factor = sqrtf(
                (101325.0f / (287.05f * 298.0f)) / rho);
            const float prefactor = pi * 0.25f * 40.0f;
            snow_riming_rate = (double)(density_factor * prefactor
                * efficiency * cloud_mass * snow_riming_moment);
            snow_riming_rate = fmin(
                snow_riming_rate, (double)cloud_mass / (double)dt);
        }
    }

    double graupel_riming_rate = 0.0;
    if (cloud_mass > 1.0e-12f && graupel_mass >= 1.0e-6f
            && cloud_mvd > 1.0e-6f) {
        const float am_g = pi * 400.0f / 6.0f;
        const float intercept_power = fmaxf(2.0f, fminf(6.0f,
            3.0f + (2.0f / 7.0f)
                * (log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f)));
        const float intercept_guess = powf(10.0f, intercept_power);
        double lambda = (double)powf(
            intercept_guess * am_g * 6.0f / graupel_mass, 0.25f);
        float mvd = (float)(3.672 / lambda);
        if (mvd > 25.4e-3f) {
            lambda = 3.672 / 25.4e-3;
        } else if (mvd < 50.0e-6f) {
            lambda = 3.672 / 50.0e-6;
        }
        const double inverse_lambda = 1.0 / lambda;
        const float graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass,
            am_g) * (float)(lambda * lambda * lambda);
        const double intercept = (double)graupel_number * lambda;
        const float density_factor = sqrtf(
            (101325.0f / (287.05f * 298.0f)) / rho);
        const float viscosity = (1.718f + 0.0049f * tempc
            - 1.2e-5f * tempc * tempc) * 1.0e-5f;
        const float diameter = (float)(4.0 * inverse_lambda);
        const float velocity = (float)(
            (double)(density_factor * 442.0f * 20.3632278f
                     * (1.0f / 6.0f))
            * pow(inverse_lambda, 0.89));
        const float stokes_number = cloud_mvd * cloud_mvd
            * velocity * 1000.0f / (9.0f * viscosity * diameter);
        float efficiency = 0.0f;
        if (stokes_number >= 0.4f && stokes_number <= 10.0f) {
            efficiency = 0.55f * log10f(2.51f * stokes_number);
        } else if (stokes_number > 10.0f) {
            efficiency = 0.77f;
        }
        const float prefactor = pi * 0.25f * 442.0f * 5.23476267f;
        graupel_riming_rate = (double)(density_factor * prefactor
            * efficiency * cloud_mass)
            * intercept * pow(inverse_lambda, 3.89);
        // WRF leaves prg_gcw raw until the joint cloud-water conservation
        // pass below; unlike the cloud-number tendency, the mass tendency
        // has no individual rc/dt cap.
    }

    // H-M rates are diagnosed before the cloud-water cap and intentionally
    // remain held when that later cap rescales riming mass rates.
    double hm_number_rate = 0.0;
    double hm_mass_rate = 0.0;
    double snow_hm_rate = 0.0;
    double graupel_hm_rate = 0.0;
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

    const double cloud_sink = autoconversion_rate + accretion_rate
        + cloud_freezing_rate + snow_riming_rate + graupel_riming_rate;
    const double cloud_limit = (double)cloud_mass / (double)dt;
    double cloud_ratio = 1.0;
    if (cloud_sink > cloud_limit && cloud_sink > 0.0) {
        cloud_ratio = cloud_limit / cloud_sink;
        autoconversion_rate *= cloud_ratio;
        accretion_rate *= cloud_ratio;
        cloud_freezing_rate *= cloud_ratio;
        snow_riming_rate *= cloud_ratio;
        graupel_riming_rate *= cloud_ratio;
    }

    const double bounded_cloud_sink = autoconversion_rate + accretion_rate
        + cloud_freezing_rate + snow_riming_rate + graupel_riming_rate;
    qc[idx] = fmaxf(0.0f, qc[idx]
        - (float)(bounded_cloud_sink * (double)orho) * dt);
    qr[idx] = fmaxf(0.0f, qr[idx]
        + (float)((autoconversion_rate + accretion_rate)
                  * (double)orho) * dt);
    nr[idx] = fmaxf(0.0f, nr[idx]
        + (float)((autoconversion_number_rate - self_number_rate)
                  * (double)orho) * dt);
    qi[idx] = fmaxf(0.0f, qi[idx]
        + (float)((cloud_freezing_rate + hm_mass_rate)
                  * (double)orho) * dt);
    ni[idx] = fmaxf(0.0f, ni[idx]
        + (float)((cloud_freezing_number_rate + hm_number_rate)
                  * (double)orho) * dt);
    qs[idx] = fmaxf(0.0f, qs[idx]
        + (float)((snow_riming_rate - snow_hm_rate)
                  * (double)orho) * dt);
    qg[idx] = fmaxf(0.0f, qg[idx]
        + (float)((graupel_riming_rate - graupel_hm_rate)
                  * (double)orho) * dt);
    thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
    thompson_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);

    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    const double freezing_rate = cloud_freezing_rate
        + snow_riming_rate + graupel_riming_rate;
    temperature[idx] = temp0 + (float)(
        (double)(latent_fusion * inverse_cp) * freezing_rate
        * (double)orho * (double)dt);
}

extern "C" __global__ void thompson_snow_rime_conversion(
    float* __restrict__ qc,
    float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ velocity_boost,
    float dt, int size)
{
    // WRF's deposition-conditioned partial conversion of rimed snow to
    // graupel.  Riming and snow-vapor exchange share the incoming state, so
    // they are evaluated together rather than chaining isolated launchers.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    velocity_boost[idx] = 1.0f;
    if (qs[idx] <= 1.0e-12f || temperature[idx] >= 273.15f) return;

    const float temp0 = temperature[idx];
    const float tempc = temp0 - 273.15f;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float snow_mass = qs[idx] * rho;
    const float snow_temperature = fminf(-0.1f, tempc);
    const float snow_second_moment = snow_mass * (1.0f / 0.069f);
    const float snow_first_moment = thompson_field_a(
        snow_temperature, 1.0f) * powf(
            snow_second_moment,
            thompson_field_b(snow_temperature, 1.0f));
    const float snow_third_moment = thompson_field_a(
        snow_temperature, 3.0f) * powf(
            snow_second_moment,
            thompson_field_b(snow_temperature, 3.0f));
    const float snow_diameter = snow_third_moment / snow_second_moment;
    const float snow_riming_moment = thompson_field_a(
        snow_temperature, 2.55f) * powf(
            snow_second_moment,
            thompson_field_b(snow_temperature, 2.55f));

    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const float density_factor = sqrtf(rho_not * orho);
    const float density_factor_sqrt = sqrtf(density_factor);
    const float viscosity = (1.718f + 0.0049f * tempc
        - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    const float viscosity_factor = sqrtf(rho / viscosity);
    const float diffusivity = 2.11e-5f
        * powf(__fdiv_rn(temp0, 273.15f), 1.94f)
        * (101325.0f / pressure[idx]);
    const float qvsi = thompson_rsif(pressure[idx], temp0);
    float ssati = qv0 / qvsi - 1.0f;
    if (fabsf(ssati) < 1.0e-15f) ssati = 0.0f;

    double deposition_rate = 0.0;
    if (ssati != 0.0f) {
        const float inverse_temp = 1.0f / temp0;
        const float conductivity = (5.69f + 0.0168f * tempc)
            * 1.0e-5f * 418.936f;
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
        float xsat = ssati;
        if (fabsf(xsat) < 1.0e-9f) xsat = 0.0f;
        const float xsat2 = xsat * xsat;
        const float alpha2 = alpha * alpha;
        const float geometry = 4.0f * 3.1415926536f
            * (1.0f - alpha * xsat
               + 2.0f * alpha2 * xsat2
               - 5.0f * alpha2 * alpha * xsat2 * xsat)
            / (1.0f + gamma);
        const float ventilation_moment = 1.0f + (1.0f + 0.55f) * 0.5f;
        const float snow_ventilation_moment = thompson_field_a(
            snow_temperature, ventilation_moment) * powf(
                snow_second_moment,
                thompson_field_b(snow_temperature, ventilation_moment));
        const float capacitance = fmaxf(0.15f, fminf(
            0.15f + __fdiv_rn((tempc + 1.5f) * (0.5f - 0.15f),
                (-30.0f + 1.5f)),
            0.5f));
        const float ventilation = 0.28f * powf(
            0.632f, 1.0f / 3.0f) * sqrtf(40.0f);
        const float moment_sum = 0.86f * snow_first_moment
            + ventilation * density_factor_sqrt * viscosity_factor
              * snow_ventilation_moment;
        deposition_rate = (double)(capacitance * geometry * diffusivity
            * ssati * saturated_density * moment_sum);
        const float inverse_dt = 1.0f / dt;
        const float vapor_limit = (qv0 - qvsi) * rho
            * inverse_dt * 0.999f;
        if (deposition_rate > 0.0) {
            deposition_rate = fmin(
                deposition_rate, (double)vapor_limit);
        } else {
            deposition_rate = fmax(
                (double)(-snow_mass * inverse_dt), deposition_rate);
            deposition_rate = fmax(
                deposition_rate, (double)vapor_limit);
        }
    }

    double riming_rate = 0.0;
    float cloud_mvd = 0.0f;
    const float cloud_mass = qc[idx] * rho;
    if (cloud_mass > 1.0e-12f && snow_diameter > 300.0e-6f) {
        const float am_r = 3.1415926536f * 1000.0f / 6.0f;
        const float cloud_lambda = powf(
            100.0e6f * am_r * 2730.0f / cloud_mass,
            1.0f / 3.0f);
        cloud_mvd = (15.672 / (double)cloud_lambda);
        cloud_mvd = fmaxf(1.0e-6f, fminf(cloud_mvd, 50.0e-6f));
        if (cloud_mvd > 1.0e-6f) {
            const double diameter_ratio = 0.02 / 300.0e-6;
            const double log_ratio = log(diameter_ratio);
            const double first_snow_bin = 300.0e-6
                * exp(0.5 / 100.0 * log_ratio);
            const double last_snow_bin = 300.0e-6
                * exp(99.5 / 100.0 * log_ratio);
            int snow_bin = 1 + (int)(100.0
                * log((double)snow_diameter / first_snow_bin)
                / log(last_snow_bin / first_snow_bin));
            snow_bin = min(snow_bin, 100);
            const double table_snow_diameter = 300.0e-6
                * exp(((double)snow_bin - 0.5) / 100.0 * log_ratio);
            const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
            const double table_cloud_diameter =
                (double)cloud_bin * 1.0e-6;
            const double cloud_velocity = 1.19e4
                * (1.0e4 * table_cloud_diameter
                   * table_cloud_diameter * 0.25);
            const double snow_velocity = 40.0
                * pow(table_snow_diameter, 0.55)
                * exp(-100.0 * table_snow_diameter) - cloud_velocity;
            const double melted_snow_diameter = pow(
                0.069 * table_snow_diameter * table_snow_diameter
                    / (3.1415926536 * 1000.0 / 6.0),
                1.0 / 3.0);
            const double diameter_fraction = table_cloud_diameter
                / melted_snow_diameter;
            float efficiency = 0.0f;
            if (diameter_fraction <= 0.25
                    && table_cloud_diameter >= (double)6.0e-6f
                    && snow_velocity >= (double)1.0e-3f) {
                const double stokes = table_cloud_diameter
                    * table_cloud_diameter * snow_velocity * 1000.0
                    / (9.0 * 1.718e-5 * melted_snow_diameter);
                const double reynolds = 9.0 * stokes
                    / (diameter_fraction * diameter_fraction * 1000.0);
                const double log_reynolds = log(reynolds);
                const double k0 = exp(-0.1007 - 0.358 * log_reynolds
                    + 0.0261 * log_reynolds * log_reynolds);
                const double z = log(stokes / (k0 + 1.0e-15));
                const double h = 0.1465 + 1.302 * z - 0.607 * z * z
                    + 0.293 * z * z * z;
                const double yc0 = 2.0 / 3.14159265358979323846
                    * atan(h);
                efficiency = fmaxf(0.0f, fminf((float)(
                    (yc0 + diameter_fraction)
                    * (yc0 + diameter_fraction)
                    / ((1.0 + diameter_fraction)
                       * (1.0 + diameter_fraction))), 0.95f));
            }
            riming_rate = (double)(density_factor * 3.1415926536f
                * 0.25f * 40.0f * efficiency * cloud_mass
                * snow_riming_moment);
            riming_rate = fmin(
                riming_rate, (double)(cloud_mass / dt));
        }
    }

    float graupel_fraction = 0.0f;
    if (riming_rate > 2.0 * deposition_rate
            && deposition_rate > 1.0e-15) {
        const float riming_ratio = (float)fmin(
            30.0, riming_rate / deposition_rate);
        graupel_fraction = fminf(
            0.95f, 0.15f + (riming_ratio - 2.0f) * 0.028f);
        velocity_boost[idx] = fminf(
            1.5f, 1.1f + (riming_ratio - 2.0f) * 0.014f);
        const float snow_velocity = 40.0f
            * powf(snow_diameter, 0.55f)
            * expf(-100.0f * snow_diameter);
        float rime_parameter = -(cloud_mvd * 0.5e6f) * snow_velocity
            / fminf(-0.1f, tempc);
        rime_parameter = fmaxf(0.1f, fminf(rime_parameter, 10.0f));
        const float rime_density = (0.051f + 0.114f * rime_parameter
            - 0.0055f * rime_parameter * rime_parameter) * 1000.0f;
        if (rime_density < 150.0f) graupel_fraction = 0.0f;
    }

    const double graupel_rate =
        (double)graupel_fraction * riming_rate;
    const double snow_riming_rate = riming_rate - graupel_rate;
    qc[idx] = fmaxf(0.0f, qc[idx]
        - (float)(riming_rate * (double)orho * (double)dt));
    qs[idx] = fmaxf(0.0f, qs[idx] + (float)(
        (snow_riming_rate + deposition_rate)
        * (double)orho * (double)dt));
    qg[idx] = fmaxf(0.0f, qg[idx] + (float)(
        graupel_rate * (double)orho * (double)dt));
    qv[idx] = fmaxf(1.0e-10f, qv0 - (float)(
        deposition_rate * (double)orho * (double)dt));
    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    temperature[idx] = temp0 + (float)(
        (double)inverse_cp * (double)orho * (double)dt
        * (2.834e6 * deposition_rate
           + (double)latent_fusion * riming_rate));
}

extern "C" __global__ void thompson_snow_ice_collection(
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ qs,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float dt, int size)
{
    // Classic WRF Thompson snow collection of cloud ice, with the small-ice
    // Wisner approximation used by the source process network.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qi[idx] <= 1.0e-12f
            || qs[idx] <= 1.0e-12f) return;

    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float temp0 = temperature[idx];
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float inverse_rho = 1.0f / rho;
    const float ice_mass = qi[idx] * rho;
    const float snow_mass = qs[idx] * rho;
    float ice_number = fmaxf(1.0e-6f, ni[idx] * rho);
    if (ice_mass <= 1.0e-12f || snow_mass <= 1.0e-12f) return;

    const float pi = 3.1415926536f;
    const float ice_mass_coefficient = pi * 890.0f / 6.0f;
    const double ice_lambda = (double)powf(
        ice_mass_coefficient * 6.0f * ice_number / ice_mass,
        1.0f / 3.0f);
    const float minimum_ice_diameter = powf(
        1.0e-12f / ice_mass_coefficient, 1.0f / 3.0f);
    const float ice_diameter = fmaxf(
        minimum_ice_diameter, (float)(4.0 / ice_lambda));
    const float particle_mass = ice_mass_coefficient
        * ice_diameter * ice_diameter * ice_diameter;

    const float snow_temperature = fminf(-0.1f, temp0 - 273.15f);
    const float snow_moment_2 = snow_mass * (1.0f / 0.069f);
    const float snow_riming_moment = thompson_field_a(
        snow_temperature, 2.55f) * powf(
            snow_moment_2, thompson_field_b(snow_temperature, 2.55f));
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const float density_factor = sqrtf(rho_not / rho);
    const float collection_prefactor = pi * 0.25f * 40.0f;
    double mass_rate = (double)(collection_prefactor * density_factor
        * 0.05f * ice_mass * snow_riming_moment);
    mass_rate = fmin(mass_rate, (double)ice_mass / (double)dt);
    if (mass_rate <= 0.0) return;
    const double number_rate = mass_rate * (double)(1.0f / particle_mass);

    const float mass_tendency = (float)(
        mass_rate * (double)inverse_rho);
    const float number_tendency = (float)(
        number_rate * (double)inverse_rho);
    qi[idx] = fmaxf(0.0f, qi[idx] - mass_tendency * dt);
    ni[idx] = fmaxf(0.0f, ni[idx] - number_tendency * dt);
    qs[idx] = fmaxf(0.0f, qs[idx] + mass_tendency * dt);
    thompson_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
}

extern "C" __global__ void thompson_rain_ice_collection(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ qg,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float dt, int size)
{
    // Classic WRF Thompson rain collection of cloud ice.  Rain
    // self-collection is evaluated from the same incoming distribution so
    // the two simultaneous number sinks retain WRF's process ordering.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qr[idx] <= 1.0e-12f
            || qi[idx] <= 1.0e-12f) return;

    const float temp0 = temperature[idx];
    if (temp0 >= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float rain_mass = qr[idx] * rho;
    const float ice_mass = qi[idx] * rho;
    const float rain_number = fmaxf(1.0e-6f, nr[idx] * rho);
    const float ice_number = fmaxf(1.0e-6f, ni[idx] * rho);
    if (rain_mass <= 1.0e-12f || ice_mass <= 1.0e-12f) return;

    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float am_i = pi * 890.0f / 6.0f;
    const double rain_lambda = (double)powf(
        am_r * 6.0f * rain_number / rain_mass, 1.0f / 3.0f);
    const float rain_mvd = (float)(3.672 / rain_lambda);

    const double ice_lambda = (double)powf(
        am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f);
    const float minimum_ice_diameter = powf(
        1.0e-12f / am_i, 1.0f / 3.0f);
    const float ice_diameter = fmaxf(
        minimum_ice_diameter, (float)(4.0 / ice_lambda));
    if (rain_mass < 1.0e-6f || rain_mvd <= 4.0f * ice_diameter) {
        // The rain/ice collision tables begin at r_r(1)=1e-6 kg m^-3,
        // while rain self-collection remains active below that threshold.
        if (rain_mvd > 50.0e-6f) {
            const float self_efficiency = 1.0f
                - expf(2300.0f * (rain_mvd - 1950.0e-6f));
            const double self_number_rate = (double)(
                self_efficiency * 2.0f * rain_number * rain_mass);
            nr[idx] = fmaxf(0.0f, nr[idx] - (float)(
                self_number_rate * (double)orho) * dt);
            thompson_bound_rain_number(rain_mass, rho, &nr[idx]);
        }
        return;
    }

    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const float density_factor = sqrtf(rho_not / rho);
    const double rain_intercept = (double)rain_number * rain_lambda;
    const double shifted_lambda = rain_lambda + 195.0;
    const float collection_efficiency = 0.95f;
    const float mass_prefactor = pi * 0.25f * 4854.0f * 6.0f;
    const float rain_mass_prefactor = pi * 0.25f * am_r
        * 4854.0f * 720.0f;

    double ice_mass_rate = (double)(density_factor * mass_prefactor
        * collection_efficiency * ice_mass) * rain_intercept
        * pow(shifted_lambda, -4.0);
    double rain_mass_rate = (double)(density_factor * rain_mass_prefactor
        * collection_efficiency * ice_number) * rain_intercept
        * pow(shifted_lambda, -7.0);
    ice_mass_rate = fmin(ice_mass_rate, (double)ice_mass / (double)dt);
    rain_mass_rate = fmin(
        rain_mass_rate, (double)rain_mass / (double)dt);

    const float ice_particle_mass = am_i
        * ice_diameter * ice_diameter * ice_diameter;
    const double ice_number_rate = ice_mass_rate
        / (double)ice_particle_mass;
    double collision_number_rate = (double)(density_factor
        * mass_prefactor * collection_efficiency * ice_number)
        * rain_intercept * pow(shifted_lambda, -4.0);
    collision_number_rate = fmin(
        collision_number_rate, (double)rain_number / (double)dt);

    double self_number_rate = 0.0;
    if (rain_mvd > 50.0e-6f) {
        const float self_efficiency = 1.0f
            - expf(2300.0f * (rain_mvd - 1950.0e-6f));
        self_number_rate = (double)(
            self_efficiency * 2.0f * rain_number * rain_mass);
    }

    const float ice_mass_tendency = (float)(ice_mass_rate * orho);
    const float rain_mass_tendency = (float)(rain_mass_rate * orho);
    const float ice_number_tendency = (float)(ice_number_rate * orho);
    const float rain_number_tendency = (float)(
        (collision_number_rate + self_number_rate) * (double)orho);
    qi[idx] = fmaxf(0.0f, qi[idx] - ice_mass_tendency * dt);
    ni[idx] = fmaxf(0.0f, ni[idx] - ice_number_tendency * dt);
    qr[idx] = fmaxf(0.0f, qr[idx] - rain_mass_tendency * dt);
    nr[idx] = fmaxf(0.0f, nr[idx] - rain_number_tendency * dt);
    qg[idx] = fmaxf(0.0f, qg[idx]
        + (ice_mass_tendency + rain_mass_tendency) * dt);
    thompson_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
    thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);

    const float tempc = temp0 - 273.15f;
    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    temperature[idx] = temp0 + latent_fusion * inverse_cp
        * rain_mass_tendency * dt;
}

extern "C" __global__ void thompson_rain_snow_collection(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
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
    float dt, int size)
{
    // Cold classic-WRF rain/snow collision table plus the simultaneous
    // Seifert rain-number sink diagnosed from the same incoming state.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qr[idx] <= 1.0e-12f) return;

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float rain_mass = qr[idx] * rho;
    const float rain_number = fmaxf(1.0e-6f, nr[idx] * rho);
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const double rain_lambda = (double)powf(
        am_r * 6.0f * rain_number / rain_mass, 1.0f / 3.0f);
    const float rain_mvd = (float)(3.672 / rain_lambda);

    double self_number_rate = 0.0;
    if (rain_mvd > 50.0e-6f) {
        const float efficiency = 1.0f
            - expf(2300.0f * (rain_mvd - 1950.0e-6f));
        self_number_rate = (double)(
            efficiency * 2.0f * rain_number * rain_mass);
    }

    double rain_rate = 0.0;
    double snow_rate = 0.0;
    double graupel_rate = 0.0;
    double collision_number_rate = 0.0;
    const float snow_mass = qs[idx] * rho;
    if (temp0 < 273.15f && rain_mass >= 1.0e-6f
            && snow_mass >= 1.0e-6f) {
        const int snow_bin = thompson_decade_table_index(
            snow_mass, -6, 37);
        const int rain_bin = thompson_decade_table_index(
            rain_mass, -6, 37);
        const float intercept_prefix = __fdiv_rn((1.0f / 6.0f)
            * rain_mass, am_r);
        const double intercept = (double)intercept_prefix
            * rain_lambda * rain_lambda * rain_lambda * rain_lambda;
        const int intercept_bin = thompson_decade_table_index_double(
            intercept, 6, 37);
        const float tempc = temp0 - 273.15f;
        const int raw_temp_bin = (int)(__fdiv_rn((tempc - 2.5f), 5.0f)) - 1;
        const int temp_bin = min(9, max(1, -raw_temp_bin)) - 1;
        const size_t table_idx = (size_t)snow_bin
            + (size_t)37 * ((size_t)temp_bin
            + (size_t)9 * ((size_t)intercept_bin
            + (size_t)37 * (size_t)rain_bin));

        rain_rate = -(tmr_racs2[table_idx] + tcr_sacr2[table_idx]
            + tmr_racs1[table_idx] + tcr_sacr1[table_idx]);
        snow_rate = tmr_racs2[table_idx] + tcr_sacr2[table_idx]
            - tcs_racs1[table_idx] - tms_sacr1[table_idx];
        graupel_rate = tmr_racs1[table_idx] + tcr_sacr1[table_idx]
            + tcs_racs1[table_idx] + tms_sacr1[table_idx];
        collision_number_rate = tnr_racs1[table_idx]
            + tnr_racs2[table_idx] + tnr_sacr1[table_idx]
            + tnr_sacr2[table_idx];
        rain_rate = fmax((double)(-rain_mass / dt), rain_rate);
        snow_rate = fmax((double)(-snow_mass / dt), snow_rate);
        graupel_rate = fmin(
            (double)((rain_mass + snow_mass) / dt), graupel_rate);
        collision_number_rate = fmin(
            (double)(rain_number / dt), collision_number_rate);
    }

    const float qr_tendency = (float)(rain_rate * (double)orho);
    const float qs_tendency = (float)(snow_rate * (double)orho);
    const float qg_tendency = (float)(graupel_rate * (double)orho);
    const float nr_tendency = (float)(
        -(collision_number_rate + self_number_rate) * (double)orho);
    qr[idx] = fmaxf(0.0f, qr[idx] + qr_tendency * dt);
    qs[idx] = fmaxf(0.0f, qs[idx] + qs_tendency * dt);
    qg[idx] = fmaxf(0.0f, qg[idx] + qg_tendency * dt);
    nr[idx] = fmaxf(0.0f, nr[idx] + nr_tendency * dt);
    thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);

    const float tempc = temp0 - 273.15f;
    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    temperature[idx] = temp0 + (float)(
        (double)(latent_fusion * inverse_cp)
        * (graupel_rate + snow_rate) * (double)orho * (double)dt);
}

extern "C" __global__ void thompson_rain_graupel_collection(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ qg,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const double* __restrict__ tcg_racg,
    const double* __restrict__ tmr_racg,
    const double* __restrict__ tcr_gacr,
    const double* __restrict__ tnr_racg,
    const double* __restrict__ tnr_gacr,
    float dt, int size)
{
    // Cold classic-WRF rain/graupel collision table plus the simultaneous
    // Seifert rain-number sink diagnosed from the same incoming state.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qr[idx] <= 1.0e-12f) return;

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float rain_mass = qr[idx] * rho;
    const float rain_number = fmaxf(1.0e-6f, nr[idx] * rho);
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const double rain_lambda = (double)powf(
        am_r * 6.0f * rain_number / rain_mass, 1.0f / 3.0f);
    const float rain_mvd = (float)(3.672 / rain_lambda);

    double self_number_rate = 0.0;
    if (rain_mvd > 50.0e-6f) {
        const float efficiency = 1.0f
            - expf(2300.0f * (rain_mvd - 1950.0e-6f));
        self_number_rate = (double)(
            efficiency * 2.0f * rain_number * rain_mass);
    }

    double graupel_rate = 0.0;
    double collision_number_rate = 0.0;
    const float graupel_mass = qg[idx] * rho;
    if (temp0 < 273.15f && rain_mass >= 1.0e-6f
            && graupel_mass >= 1.0e-6f) {
        const int graupel_mass_bin = thompson_decade_table_index(
            graupel_mass, -6, 37);
        const int rain_mass_bin = thompson_decade_table_index(
            rain_mass, -6, 37);
        const float am_g = 3.1415926536f * 400.0f / 6.0f;
        const float intercept_power = fmaxf(2.0f, fminf(6.0f,
            3.0f + (2.0f / 7.0f)
                * (log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f)));
        const double graupel_intercept = (double)powf(
            10.0f, intercept_power);
        const int graupel_intercept_bin =
            thompson_decade_table_index_double(
                graupel_intercept, 2, 37);
        const int rain_intercept_bin =
            thompson_decade_table_index_double(
                (double)(__fdiv_rn((1.0f / 6.0f) * rain_mass, am_r))
                    * rain_lambda * rain_lambda
                    * rain_lambda * rain_lambda,
                6, 37);

        // One slab, never WRF's idx_bg1=5 subscript: thompson_racg_index.
        const size_t table_idx = thompson_racg_index(
            graupel_intercept_bin, graupel_mass_bin,
            rain_intercept_bin, rain_mass_bin);
        graupel_rate = tmr_racg[table_idx] + tcr_gacr[table_idx];
        graupel_rate = fmin(
            (double)(rain_mass / dt), graupel_rate);
        collision_number_rate = tnr_racg[table_idx]
            + tnr_gacr[table_idx];
        collision_number_rate = fmin(
            (double)(rain_number / dt), collision_number_rate);
    }

    const float mass_tendency = (float)(graupel_rate * (double)orho);
    const float number_tendency = (float)(
        (collision_number_rate + self_number_rate) * (double)orho);
    qr[idx] = fmaxf(0.0f, qr[idx] - mass_tendency * dt);
    qg[idx] = fmaxf(0.0f, qg[idx] + mass_tendency * dt);
    nr[idx] = fmaxf(0.0f, nr[idx] - number_tendency * dt);
    thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);

    const float tempc = temp0 - 273.15f;
    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    temperature[idx] = temp0
        + latent_fusion * inverse_cp * mass_tendency * dt;
}

extern "C" __global__ void thompson_cold_rain_snow_graupel_network(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
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
    float dt, int size)
{
    // Production-order cold collision slice.  Both lookup families and
    // Seifert self-collection are diagnosed from one immutable incoming rain
    // state, then WRF's grouped rain limiter is applied exactly once.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qr[idx] <= 1.0e-12f) return;

    // tcg_racg is the warm rain/graupel table.  It remains part of the
    // canonical five-table bundle but is intentionally unused in this cold
    // network.
    (void)tcg_racg;

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float rain_mass = qr[idx] * rho;
    const float rain_number = fmaxf(1.0e-6f, nr[idx] * rho);
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const double rain_lambda = (double)powf(
        am_r * 6.0f * rain_number / rain_mass, 1.0f / 3.0f);
    const float rain_mvd = (float)(3.672 / rain_lambda);
    const int rain_mass_bin = thompson_decade_table_index(
        rain_mass, -6, 37);
    const double rain_intercept =
        (double)(__fdiv_rn((1.0f / 6.0f) * rain_mass, am_r))
        * rain_lambda * rain_lambda * rain_lambda * rain_lambda;
    const int rain_intercept_bin =
        thompson_decade_table_index_double(rain_intercept, 6, 37);

    double self_number_rate = 0.0;
    if (rain_mvd > 50.0e-6f) {
        const float efficiency = 1.0f
            - expf(2300.0f * (rain_mvd - 1950.0e-6f));
        self_number_rate = (double)(
            efficiency * 2.0f * rain_number * rain_mass);
    }

    double rain_snow_rain_rate = 0.0;
    double snow_rate = 0.0;
    double rain_snow_graupel_rate = 0.0;
    double rain_snow_number_rate = 0.0;
    const float snow_mass = qs[idx] * rho;
    if (temp0 < 273.15f && rain_mass >= 1.0e-6f
            && snow_mass >= 1.0e-6f) {
        const int snow_bin = thompson_decade_table_index(
            snow_mass, -6, 37);
        const float tempc = temp0 - 273.15f;
        const int raw_temp_bin = (int)(__fdiv_rn((tempc - 2.5f), 5.0f)) - 1;
        const int temp_bin = min(9, max(1, -raw_temp_bin)) - 1;
        const size_t table_idx = (size_t)snow_bin
            + (size_t)37 * ((size_t)temp_bin
            + (size_t)9 * ((size_t)rain_intercept_bin
            + (size_t)37 * (size_t)rain_mass_bin));

        rain_snow_rain_rate = -(
            tmr_racs2[table_idx] + tcr_sacr2[table_idx]
            + tmr_racs1[table_idx] + tcr_sacr1[table_idx]);
        snow_rate = tmr_racs2[table_idx] + tcr_sacr2[table_idx]
            - tcs_racs1[table_idx] - tms_sacr1[table_idx];
        rain_snow_graupel_rate =
            tmr_racs1[table_idx] + tcr_sacr1[table_idx]
            + tcs_racs1[table_idx] + tms_sacr1[table_idx];
        rain_snow_number_rate =
            tnr_racs1[table_idx] + tnr_racs2[table_idx]
            + tnr_sacr1[table_idx] + tnr_sacr2[table_idx];
        rain_snow_rain_rate = fmax(
            (double)(-rain_mass / dt), rain_snow_rain_rate);
        snow_rate = fmax((double)(-snow_mass / dt), snow_rate);
        rain_snow_graupel_rate = fmin(
            (double)((rain_mass + snow_mass) / dt),
            rain_snow_graupel_rate);
        rain_snow_number_rate = fmin(
            (double)(rain_number / dt), rain_snow_number_rate);
    }

    double rain_graupel_rain_rate = 0.0;
    double rain_graupel_graupel_rate = 0.0;
    double rain_graupel_number_rate = 0.0;
    const float graupel_mass = qg[idx] * rho;
    if (temp0 < 273.15f && rain_mass >= 1.0e-6f
            && graupel_mass >= 1.0e-6f) {
        const int graupel_mass_bin = thompson_decade_table_index(
            graupel_mass, -6, 37);
        const float intercept_power = fmaxf(2.0f, fminf(6.0f,
            3.0f + (2.0f / 7.0f)
                * (log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f)));
        const double graupel_intercept = (double)powf(
            10.0f, intercept_power);
        const int graupel_intercept_bin =
            thompson_decade_table_index_double(
                graupel_intercept, 2, 37);

        // One slab, never WRF's idx_bg1=5 subscript: thompson_racg_index.
        const size_t table_idx = thompson_racg_index(
            graupel_intercept_bin, graupel_mass_bin,
            rain_intercept_bin, rain_mass_bin);
        rain_graupel_graupel_rate =
            tmr_racg[table_idx] + tcr_gacr[table_idx];
        rain_graupel_graupel_rate = fmin(
            (double)(rain_mass / dt),
            rain_graupel_graupel_rate);
        rain_graupel_rain_rate = -rain_graupel_graupel_rate;
        rain_graupel_number_rate =
            tnr_racg[table_idx] + tnr_gacr[table_idx];
        rain_graupel_number_rate = fmin(
            (double)(rain_number / dt),
            rain_graupel_number_rate);
    }

    // WRF groups all rain-mass sinks before applying the species bound.  It
    // deliberately does not scale collision-number rates with this ratio.
    const double rain_sum =
        rain_snow_rain_rate + rain_graupel_rain_rate;
    const double rain_limit = (double)(-rain_mass / dt);
    if (rain_sum < rain_limit) {
        const double ratio = rain_limit / rain_sum;
        rain_snow_rain_rate *= ratio;
        rain_graupel_rain_rate *= ratio;
    }

    // The official driver re-enforces only the rain/graupel pair after the
    // grouped bound.  Cold rain/snow is a three-way split and is left as-is.
    const double paired_rate = fmin(
        fabs(rain_graupel_rain_rate),
        fabs(rain_graupel_graupel_rate));
    rain_graupel_rain_rate = -paired_rate;
    rain_graupel_graupel_rate = paired_rate;

    const double rain_rate =
        rain_snow_rain_rate + rain_graupel_rain_rate;
    const double graupel_rate =
        rain_snow_graupel_rate + rain_graupel_graupel_rate;
    const double number_sink = self_number_rate
        + rain_snow_number_rate + rain_graupel_number_rate;
    const float qr_tendency = (float)(rain_rate * (double)orho);
    const float qs_tendency = (float)(snow_rate * (double)orho);
    const float qg_tendency = (float)(graupel_rate * (double)orho);
    const float nr_tendency = (float)(-number_sink * (double)orho);
    qr[idx] = fmaxf(0.0f, qr[idx] + qr_tendency * dt);
    qs[idx] = fmaxf(0.0f, qs[idx] + qs_tendency * dt);
    qg[idx] = fmaxf(0.0f, qg[idx] + qg_tendency * dt);
    nr[idx] = fmaxf(0.0f, nr[idx] + nr_tendency * dt);
    thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);

    const float tempc = temp0 - 273.15f;
    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    temperature[idx] = temp0 + (float)(
        (double)(latent_fusion * inverse_cp)
        * (snow_rate + rain_snow_graupel_rate
           + rain_graupel_graupel_rate)
        * (double)orho * (double)dt);
}

extern "C" __global__ void thompson_cold_rain_source_network(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
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
    float dt, int size)
{
    // Complete classic cold-rain source group.  Table freezing, rain/ice,
    // rain/snow, rain/graupel, and Seifert self-collection all see the same
    // immutable incoming distribution before WRF's species bounds run.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qr[idx] <= 1.0e-12f) return;
    (void)tcg_racg;

    const float temp0 = temperature[idx];
    if (temp0 >= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float rain_mass = qr[idx] * rho;
    const float rain_number = fmaxf(1.0e-6f, nr[idx] * rho);
    const float ice_mass = qi[idx] * rho;
    const float ice_number = fmaxf(1.0e-6f, ni[idx] * rho);
    const float snow_mass = qs[idx] * rho;
    const float graupel_mass = qg[idx] * rho;
    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float am_i = pi * 890.0f / 6.0f;
    const double rain_lambda = (double)powf(
        am_r * 6.0f * rain_number / rain_mass, 1.0f / 3.0f);
    const float rain_mvd = (float)(3.672 / rain_lambda);
    const int rain_mass_bin = thompson_decade_table_index(
        rain_mass, -6, 37);
    const double rain_intercept =
        (double)(__fdiv_rn((1.0f / 6.0f) * rain_mass, am_r))
        * rain_lambda * rain_lambda * rain_lambda * rain_lambda;
    const int rain_intercept_bin =
        thompson_decade_table_index_double(rain_intercept, 6, 37);

    double self_number_rate = 0.0;
    if (rain_mvd > 50.0e-6f) {
        const float efficiency = 1.0f
            - expf(2300.0f * (rain_mvd - 1950.0e-6f));
        self_number_rate = (double)(
            efficiency * 2.0f * rain_number * rain_mass);
    }

    // Bigg table freezing.  WRF stores per-step amounts and converts them to
    // rates with odts; associated number rates are not rescaled later when
    // the grouped rain-mass bound activates.
    double freeze_ice_rate = 0.0;
    double freeze_ice_number_rate = 0.0;
    double freeze_graupel_rate = 0.0;
    double freeze_graupel_number_rate = 0.0;
    if (rain_mass > 1.0e-6f) {
        const int temp_bin = max(0, min(
            (int)roundf(-(temp0 - 273.15f)) - 1, 44));
        const int nuclei_bin = 27;
        const size_t table_idx = (size_t)rain_mass_bin
            + (size_t)37 * ((size_t)rain_intercept_bin
            + (size_t)37 * ((size_t)temp_bin
            + (size_t)45 * (size_t)nuclei_bin));
        freeze_ice_rate = rain_to_ice_mass[table_idx] / (double)dt;
        freeze_ice_number_rate =
            rain_to_ice_number[table_idx] / (double)dt;
        freeze_graupel_rate =
            rain_to_graupel_mass[table_idx] / (double)dt;
        freeze_graupel_number_rate = fmin(
            (double)rain_number,
            rain_to_graupel_number[table_idx]) / (double)dt;
    } else if (rain_mass > 1.0e-12f && temp0 < 235.16f) {
        freeze_ice_rate = (double)rain_mass / (double)dt;
        freeze_ice_number_rate = (double)rain_number / (double)dt;
    }

    // Rain collecting cloud ice.  Diagnose all mass and number channels
    // before applying the separate cloud-ice and rain source bounds.
    double rain_ice_ice_rate = 0.0;
    double rain_ice_rain_rate = 0.0;
    double rain_ice_ice_number_rate = 0.0;
    double rain_ice_rain_number_rate = 0.0;
    double rain_ice_graupel_rate = 0.0;
    if (ice_mass > 1.0e-12f) {
        const double ice_lambda = (double)powf(
            am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f);
        const float minimum_ice_diameter = powf(
            1.0e-12f / am_i, 1.0f / 3.0f);
        const float ice_diameter = fmaxf(
            minimum_ice_diameter, (float)(4.0 / ice_lambda));
        if (rain_mass >= 1.0e-6f
                && rain_mvd > 4.0f * ice_diameter) {
            const float rho_not = 101325.0f / (287.05f * 298.0f);
            const float density_factor = sqrtf(rho_not / rho);
            const double intercept = (double)rain_number * rain_lambda;
            const double shifted_lambda = rain_lambda + 195.0;
            const float collection_efficiency = 0.95f;
            const float mass_prefactor =
                pi * 0.25f * 4854.0f * 6.0f;
            const float rain_mass_prefactor =
                pi * 0.25f * am_r * 4854.0f * 720.0f;
            rain_ice_ice_rate = (double)(
                density_factor * mass_prefactor
                * collection_efficiency * ice_mass)
                * intercept * pow(shifted_lambda, -4.0);
            rain_ice_rain_rate = (double)(
                density_factor * rain_mass_prefactor
                * collection_efficiency * ice_number)
                * intercept * pow(shifted_lambda, -7.0);
            rain_ice_rain_rate = fmin(
                rain_ice_rain_rate,
                (double)rain_mass / (double)dt);
            const float particle_mass = am_i
                * ice_diameter * ice_diameter * ice_diameter;
            rain_ice_ice_number_rate =
                rain_ice_ice_rate / (double)particle_mass;
            rain_ice_rain_number_rate = (double)(
                density_factor * mass_prefactor
                * collection_efficiency * ice_number)
                * intercept * pow(shifted_lambda, -4.0);
            rain_ice_rain_number_rate = fmin(
                rain_ice_rain_number_rate,
                (double)rain_number / (double)dt);
            // WRF forms this source before its later species caps and does
            // not reconstruct it afterward.
            rain_ice_graupel_rate =
                rain_ice_ice_rate + rain_ice_rain_rate;
        }
    }
    // This slice has no other cloud-ice sink, so WRF's cloud-ice group bound
    // reduces to this one rate.  Its number and graupel channels stay held.
    rain_ice_ice_rate = fmin(
        rain_ice_ice_rate, (double)ice_mass / (double)dt);

    double rain_snow_rain_rate = 0.0;
    double snow_rate = 0.0;
    double rain_snow_graupel_rate = 0.0;
    double rain_snow_number_rate = 0.0;
    if (rain_mass >= 1.0e-6f && snow_mass >= 1.0e-6f) {
        const int snow_bin = thompson_decade_table_index(
            snow_mass, -6, 37);
        const float tempc = temp0 - 273.15f;
        const int raw_temp_bin = (int)(__fdiv_rn((tempc - 2.5f), 5.0f)) - 1;
        const int temp_bin = min(9, max(1, -raw_temp_bin)) - 1;
        const size_t table_idx = (size_t)snow_bin
            + (size_t)37 * ((size_t)temp_bin
            + (size_t)9 * ((size_t)rain_intercept_bin
            + (size_t)37 * (size_t)rain_mass_bin));
        rain_snow_rain_rate = -(
            tmr_racs2[table_idx] + tcr_sacr2[table_idx]
            + tmr_racs1[table_idx] + tcr_sacr1[table_idx]);
        snow_rate = tmr_racs2[table_idx] + tcr_sacr2[table_idx]
            - tcs_racs1[table_idx] - tms_sacr1[table_idx];
        rain_snow_graupel_rate =
            tmr_racs1[table_idx] + tcr_sacr1[table_idx]
            + tcs_racs1[table_idx] + tms_sacr1[table_idx];
        rain_snow_number_rate =
            tnr_racs1[table_idx] + tnr_racs2[table_idx]
            + tnr_sacr1[table_idx] + tnr_sacr2[table_idx];
        rain_snow_rain_rate = fmax(
            (double)(-rain_mass / dt), rain_snow_rain_rate);
        snow_rate = fmax((double)(-snow_mass / dt), snow_rate);
        rain_snow_graupel_rate = fmin(
            (double)((rain_mass + snow_mass) / dt),
            rain_snow_graupel_rate);
        rain_snow_number_rate = fmin(
            (double)(rain_number / dt), rain_snow_number_rate);
    }

    double rain_graupel_rain_rate = 0.0;
    double rain_graupel_graupel_rate = 0.0;
    double rain_graupel_number_rate = 0.0;
    if (rain_mass >= 1.0e-6f && graupel_mass >= 1.0e-6f) {
        const int graupel_mass_bin = thompson_decade_table_index(
            graupel_mass, -6, 37);
        const float intercept_power = fmaxf(2.0f, fminf(6.0f,
            3.0f + (2.0f / 7.0f)
                * (log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f)));
        const int graupel_intercept_bin =
            thompson_decade_table_index_double(
                (double)powf(10.0f, intercept_power), 2, 37);
        // One slab, never WRF's idx_bg1=5 subscript: thompson_racg_index.
        const size_t table_idx = thompson_racg_index(
            graupel_intercept_bin, graupel_mass_bin,
            rain_intercept_bin, rain_mass_bin);
        rain_graupel_graupel_rate =
            tmr_racg[table_idx] + tcr_gacr[table_idx];
        rain_graupel_graupel_rate = fmin(
            (double)(rain_mass / dt),
            rain_graupel_graupel_rate);
        rain_graupel_rain_rate = -rain_graupel_graupel_rate;
        rain_graupel_number_rate =
            tnr_racg[table_idx] + tnr_gacr[table_idx];
        rain_graupel_number_rate = fmin(
            (double)(rain_number / dt),
            rain_graupel_number_rate);
    }

    // Exact classic-WRF rain conservation group.  Only mass rates scale;
    // every diagnosed number channel remains on the incoming state.
    const double rain_sum = -freeze_graupel_rate - freeze_ice_rate
        - rain_ice_rain_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
    const double rain_limit = (double)(-rain_mass / dt);
    if (rain_sum < rain_limit) {
        const double ratio = rain_limit / rain_sum;
        freeze_graupel_rate *= ratio;
        freeze_ice_rate *= ratio;
        rain_ice_rain_rate *= ratio;
        rain_snow_rain_rate *= ratio;
        rain_graupel_rain_rate *= ratio;
    }

    const double paired_rate = fmin(
        fabs(rain_graupel_rain_rate),
        fabs(rain_graupel_graupel_rate));
    rain_graupel_rain_rate = -paired_rate;
    rain_graupel_graupel_rate = paired_rate;

    const double rain_rate = -freeze_graupel_rate - freeze_ice_rate
        - rain_ice_rain_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
    const double ice_rate = freeze_ice_rate - rain_ice_ice_rate;
    const double graupel_rate = freeze_graupel_rate
        + rain_ice_graupel_rate + rain_snow_graupel_rate
        + rain_graupel_graupel_rate;
    const double rain_number_sink = self_number_rate
        + freeze_graupel_number_rate + freeze_ice_number_rate
        + rain_ice_rain_number_rate + rain_snow_number_rate
        + rain_graupel_number_rate;
    const double ice_number_rate =
        freeze_ice_number_rate - rain_ice_ice_number_rate;

    qr[idx] = fmaxf(0.0f, qr[idx]
        + (float)(rain_rate * (double)orho) * dt);
    nr[idx] = fmaxf(0.0f, nr[idx]
        - (float)(rain_number_sink * (double)orho) * dt);
    qi[idx] = fmaxf(0.0f, qi[idx]
        + (float)(ice_rate * (double)orho) * dt);
    ni[idx] = fmaxf(0.0f, ni[idx]
        + (float)(ice_number_rate * (double)orho) * dt);
    qs[idx] = fmaxf(0.0f, qs[idx]
        + (float)(snow_rate * (double)orho) * dt);
    qg[idx] = fmaxf(0.0f, qg[idx]
        + (float)(graupel_rate * (double)orho) * dt);
    thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
    thompson_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);

    const float tempc = temp0 - 273.15f;
    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    const double freezing_rate = freeze_ice_rate + freeze_graupel_rate
        + rain_ice_rain_rate + snow_rate + rain_snow_graupel_rate
        + rain_graupel_graupel_rate;
    temperature[idx] = temp0 + (float)(
        (double)(latent_fusion * inverse_cp) * freezing_rate
        * (double)orho * (double)dt);
}

extern "C" __global__ void thompson_ice_deposition(
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ qs,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    const double* __restrict__ ice_deposition_partition,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qi[idx] <= 1.0e-12f) return;

    const float temp0 = temperature[idx];
    if (temp0 >= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float qvsi = thompson_rsif(pressure[idx], temp0);
    float ssati = qv0 / qvsi - 1.0f;
    if (fabsf(ssati) < 1.0e-15f) ssati = 0.0f;
    if (ssati == 0.0f) return;

    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float ice_mass = qi[idx] * rho;
    float ice_number = fmaxf(1.0e-6f, ni[idx] * rho);
    const float pi = 3.1415926536f;
    const float am_i = pi * 890.0f / 6.0f;
    double lambda = (double)powf(
        am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f);
    double inverse_lambda = 1.0 / lambda;
    float mean_diameter = (float)(4.0 * inverse_lambda);
    if (mean_diameter < 5.0e-6f) {
        lambda = 4.0 / 5.0e-6;
        inverse_lambda = 1.0 / lambda;
        ice_number = fminf(999.0e3f,
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

    const float tempc = temp0 - 273.15f;
    const float inverse_temp = 1.0f / temp0;
    const float diffusivity = 2.11e-5f
        * powf(__fdiv_rn(temp0, 273.15f), 1.94f)
        * (101325.0f / pressure[idx]);
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
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
    float xsat = ssati;
    if (fabsf(xsat) < 1.0e-9f) xsat = 0.0f;
    const float xsat2 = xsat * xsat;
    const float alpha2 = alpha * alpha;
    const float deposition_geometry = 4.0f * pi
        * (1.0f - alpha * xsat
           + 2.0f * alpha2 * xsat2
           - 5.0f * alpha2 * alpha * xsat2 * xsat)
        / (1.0f + gamma);

    double total_rate = (double)(
        0.5f * deposition_geometry * diffusivity * ssati
        * saturated_density * ice_number) * inverse_lambda;
    const float inverse_dt = 1.0f / dt;
    const float vapor_limit = (qv0 - qvsi) * rho * inverse_dt * 0.999f;
    double ice_rate;
    double snow_rate = 0.0;
    double number_rate = 0.0;
    if (total_rate > 0.0) {
        total_rate = fmin(total_rate, (double)vapor_limit);
        const int mass_bin = ice_mass > 1.0e-10f
            ? thompson_decade_table_index(ice_mass, -10, 64) : 0;
        const int number_bin = ice_number > 1.0f
            ? thompson_decade_table_index(ice_number, 0, 55) : 0;
        const double ice_fraction = ice_deposition_partition[
            mass_bin + 64 * number_bin];
        ice_rate = ice_fraction * total_rate;
        snow_rate = (1.0 - ice_fraction) * total_rate;
    } else {
        total_rate = fmax((double)(-ice_mass * inverse_dt), total_rate);
        total_rate = fmax(total_rate, (double)vapor_limit);
        const float minimum_diameter = powf(1.0e-12f / am_i,
                                             1.0f / 3.0f);
        const float particle_diameter = fmaxf(
            minimum_diameter, mean_diameter);
        const float particle_mass = am_i * particle_diameter
            * particle_diameter * particle_diameter;
        number_rate = total_rate / (double)particle_mass;
        number_rate = fmax((double)(-ice_number * inverse_dt), number_rate);
        ice_rate = total_rate;
    }

    const float qi_tendency = (float)(ice_rate * (double)orho);
    const float qs_tendency = (float)(snow_rate * (double)orho);
    const float ni_tendency = (float)(number_rate * (double)orho);
    float qi_new = qi[idx] + qi_tendency * dt;
    float ni_new = ni[idx] + ni_tendency * dt;
    const float final_ice_mass = fmaxf(1.0e-12f, qi_new * rho);
    float final_ice_number = fmaxf(1.0e-6f, ni_new * rho);
    if (final_ice_mass > 1.0e-12f) {
        double final_lambda = (double)powf(
            am_i * 6.0f * final_ice_number / final_ice_mass,
            1.0f / 3.0f);
        const float final_diameter = (float)(4.0 / final_lambda);
        if (final_diameter < 5.0e-6f) {
            final_lambda = 4.0 / 5.0e-6;
            final_ice_number = fminf(999.0e3f,
                __fdiv_rn((1.0f / 6.0f) * final_ice_mass, am_i)
                * (float)(final_lambda * final_lambda * final_lambda));
        } else if (final_diameter > 300.0e-6f) {
            final_lambda = 4.0 / 300.0e-6;
            final_ice_number = __fdiv_rn((1.0f / 6.0f) * final_ice_mass, am_i)
                * (float)(final_lambda * final_lambda * final_lambda);
        }
        final_ice_number = fminf(final_ice_number, 999.0e3f);
        ni_new = final_ice_number * orho;
    } else {
        qi_new = 0.0f;
        ni_new = 0.0f;
    }

    qi[idx] = fmaxf(0.0f, qi_new);
    ni[idx] = fmaxf(0.0f, ni_new);
    qs[idx] = fmaxf(0.0f, qs[idx] + qs_tendency * dt);
    qv[idx] = fmaxf(1.0e-10f,
        qv0 - (float)(total_rate * (double)orho) * dt);
    temperature[idx] = temp0 + (float)(
        (double)(2.834e6f * inverse_cp) * total_rate
        * (double)orho * (double)dt);
}

extern "C" __global__ void thompson_frozen_vapor_network(
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
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
    int include_cold_rain,
    float dt, int size)
{
    // Simultaneous classic cold-ice group.  Non-aerosol Cooper nucleation,
    // ice/snow/graupel vapor rates, ice autoconversion, snow collection of
    // cloud ice, and rain collection of cloud ice are all diagnosed from one
    // incoming state before WRF's species limiters run in driver order.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || temperature[idx] >= 273.15f) return;

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float qvsi = thompson_rsif(pressure[idx], temp0);
    const float qvs = thompson_rslf(pressure[idx], temp0);
    float ssati = qv0 / qvsi - 1.0f;
    float ssatw = qv0 / qvs - 1.0f;
    if (fabsf(ssati) < 1.0e-15f) ssati = 0.0f;
    if (fabsf(ssatw) < 1.0e-15f) ssatw = 0.0f;
    const bool nucleation_active = ssati >= 0.25f
        || (ssatw > 1.0e-15f && temp0 < 253.15f);
    if (qi[idx] <= 1.0e-12f && qs[idx] <= 1.0e-12f
            && qg[idx] <= 1.0e-12f && qr[idx] <= 1.0e-12f
            && !nucleation_active) return;

    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float inverse_dt = 1.0f / dt;
    const float tempc = temp0 - 273.15f;
    const float inverse_temp = 1.0f / temp0;
    const float diffusivity = 2.11e-5f
        * powf(__fdiv_rn(temp0, 273.15f), 1.94f)
        * (101325.0f / pressure[idx]);
    const float viscosity = (1.718f + 0.0049f * tempc
        - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
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
    float xsat = ssati;
    if (fabsf(xsat) < 1.0e-9f) xsat = 0.0f;
    const float xsat2 = xsat * xsat;
    const float alpha2 = alpha * alpha;
    const float vapor_geometry = 4.0f * 3.1415926536f
        * (1.0f - alpha * xsat
           + 2.0f * alpha2 * xsat2
           - 5.0f * alpha2 * alpha * xsat2 * xsat)
        / (1.0f + gamma);
    const float vapor_limit =
        (qv0 - qvsi) * rho * inverse_dt * 0.999f;

    // Classic non-aerosol Cooper (1986) deposition nucleation.  WRF first
    // constrains the nucleated mass by the per-process vapor allowance, then
    // includes that mass in the shared frozen-vapor cap below.  Its number
    // source is deliberately held when that later cap rescales mass.
    double nucleation_rate = 0.0;
    double nucleation_number_rate = 0.0;
    if (nucleation_active) {
        const float existing_ice_number = qi[idx] > 1.0e-12f
            ? fmaxf(1.0e-6f, ni[idx] * rho) : 0.0f;
        const float target_number = fminf(
            250.0e3f, 5.0f * expf(0.304f * (273.15f - temp0)));
        nucleation_number_rate = (double)(fmaxf(
            0.0f, target_number - existing_ice_number) * inverse_dt);
        nucleation_rate = fmin(
            (double)vapor_limit,
            1.0e-12 * nucleation_number_rate);
        nucleation_number_rate = nucleation_rate / 1.0e-12;
    }

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
    float ice_number = fmaxf(1.0e-6f, ni[idx] * rho);
    const float rain_mass = qr[idx] * rho;
    float rain_number;
    double rain_lambda;
    float rain_mvd;
    const bool rain_active = thompson_prepare_entry_rain_distribution(
        qr[idx], nr[idx], rho, &rain_number, &rain_lambda, &rain_mvd);
    if (rain_active) {
        if (rain_mvd > 50.0e-6f) {
            const float efficiency = 1.0f
                - expf(2300.0f * (rain_mvd - 1950.0e-6f));
            rain_self_number_rate = (double)(
                efficiency * 2.0f * rain_number * rain_mass);
        }
    }
    if (qi[idx] > 1.0e-12f) {
        const float am_i = 3.1415926536f * 890.0f / 6.0f;
        double lambda = (double)powf(
            am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f);
        double inverse_lambda = 1.0 / lambda;
        float mean_diameter = (float)(4.0 * inverse_lambda);
        if (mean_diameter < 5.0e-6f) {
            lambda = 4.0 / 5.0e-6;
            inverse_lambda = 1.0 / lambda;
            ice_number = fminf(999.0e3f,
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
        ice_particle_mass = am_i * mean_diameter * mean_diameter
            * mean_diameter;
        const int mass_bin = ice_mass > 1.0e-10f
            ? thompson_decade_table_index(ice_mass, -10, 64) : 0;
        const int number_bin = ice_number > 1.0f
            ? thompson_decade_table_index(ice_number, 0, 55) : 0;
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
            total_rate = fmax(
                (double)(-ice_mass * inverse_dt), total_rate);
            total_rate = fmax(total_rate, (double)vapor_limit);
            const float minimum_diameter = powf(
                1.0e-12f / am_i, 1.0f / 3.0f);
            const float particle_diameter = fmaxf(
                minimum_diameter, mean_diameter);
            const float particle_mass = am_i * particle_diameter
                * particle_diameter * particle_diameter;
            ice_number_rate = total_rate / (double)particle_mass;
            ice_number_rate = fmax(
                (double)(-ice_number * inverse_dt), ice_number_rate);
            ice_rate = total_rate;
        }

        // Lookup-table cloud-ice to snow autoconversion is diagnosed from
        // the same incoming ice distribution as deposition and collection.
        const size_t table_idx = (size_t)mass_bin
            + (size_t)64 * (size_t)number_bin;
        if (mass_bin == 63 || mean_diameter > 1500.0e-6f) {
            autoconversion_rate = (double)(ice_mass * 0.99f)
                * (double)inverse_dt;
            autoconversion_number_rate = (double)(ice_number * 0.95f)
                * (double)inverse_dt;
        } else if (mean_diameter >= 30.0e-6f) {
            autoconversion_rate = fmin(
                (double)(ice_mass * 0.99f) * (double)inverse_dt,
                ice_to_snow_mass[table_idx] * (double)inverse_dt);
            autoconversion_number_rate = fmin(
                (double)(ice_number * 0.95f) * (double)inverse_dt,
                ice_to_snow_number[table_idx] * (double)inverse_dt);
        }

        if (rain_mass >= 1.0e-6f
                && rain_mvd > 4.0f * mean_diameter) {
            const float density_factor = sqrtf(
                (101325.0f / (287.05f * 298.0f)) * orho);
            const double rain_intercept =
                (double)rain_number * rain_lambda;
            const double shifted_lambda = rain_lambda + 195.0;
            const float collection_efficiency = 0.95f;
            const float mass_prefactor =
                3.1415926536f * 0.25f * 4854.0f * 6.0f;
            const float am_r = 3.1415926536f * 1000.0f / 6.0f;
            const float rain_mass_prefactor =
                3.1415926536f * 0.25f * am_r * 4854.0f * 720.0f;
            rain_ice_ice_rate = (double)(
                density_factor * mass_prefactor
                * collection_efficiency * ice_mass)
                * rain_intercept * pow(shifted_lambda, -4.0);
            rain_ice_rain_rate = (double)(
                density_factor * rain_mass_prefactor
                * collection_efficiency * ice_number)
                * rain_intercept * pow(shifted_lambda, -7.0);
            rain_ice_rain_rate = fmin(
                rain_ice_rain_rate,
                (double)rain_mass * (double)inverse_dt);
            rain_ice_ice_number_rate =
                rain_ice_ice_rate / (double)ice_particle_mass;
            rain_ice_rain_number_rate = (double)(
                density_factor * mass_prefactor
                * collection_efficiency * ice_number)
                * rain_intercept * pow(shifted_lambda, -4.0);
            rain_ice_rain_number_rate = fmin(
                rain_ice_rain_number_rate,
                (double)rain_number * (double)inverse_dt);
            // WRF forms this paired graupel source before either later mass
            // cap and never reconstructs it from the bounded sink rates.
            rain_ice_graupel_rate =
                rain_ice_ice_rate + rain_ice_rain_rate;
        }
    }

    // Optional completion of the cold-rain source group. These rates share
    // the same immutable incoming rain/ice/snow/graupel distributions as the
    // vapor, autoconversion, and rain/ice rates above. The grouped species
    // caps are applied only after every member has been diagnosed.
    (void)tcg_racg;
    double freeze_ice_rate = 0.0;
    double freeze_ice_number_rate = 0.0;
    double freeze_graupel_rate = 0.0;
    double freeze_graupel_number_rate = 0.0;
    double rain_snow_rain_rate = 0.0;
    double rain_snow_category_rate = 0.0;
    double rain_snow_graupel_rate = 0.0;
    double rain_snow_number_rate = 0.0;
    double rain_graupel_rain_rate = 0.0;
    double rain_graupel_graupel_rate = 0.0;
    double rain_graupel_number_rate = 0.0;
    if (include_cold_rain && rain_active) {
        const float pi = 3.1415926536f;
        const float am_r = pi * 1000.0f / 6.0f;
        const int rain_mass_bin = thompson_decade_table_index(
            rain_mass, -6, 37);
        const double table_rain_intercept =
            (double)(__fdiv_rn((1.0f / 6.0f) * rain_mass, am_r))
            * rain_lambda * rain_lambda * rain_lambda * rain_lambda;
        const int rain_intercept_bin =
            thompson_decade_table_index_double(
                table_rain_intercept, 6, 37);

        // Bigg table freezing. WRF holds all associated number rates if the
        // later grouped rain-mass bound rescales these mass transfers.
        if (rain_mass > 1.0e-6f) {
            const int temp_bin = max(0, min(
                (int)roundf(-(temp0 - 273.15f)) - 1, 44));
            const int nuclei_bin = 27;
            const size_t table_idx = (size_t)rain_mass_bin
                + (size_t)37 * ((size_t)rain_intercept_bin
                + (size_t)37 * ((size_t)temp_bin
                + (size_t)45 * (size_t)nuclei_bin));
            freeze_ice_rate = rain_to_ice_mass[table_idx] / (double)dt;
            freeze_ice_number_rate =
                rain_to_ice_number[table_idx] / (double)dt;
            freeze_graupel_rate =
                rain_to_graupel_mass[table_idx] / (double)dt;
            freeze_graupel_number_rate = fmin(
                (double)rain_number,
                rain_to_graupel_number[table_idx]) / (double)dt;
        } else if (rain_mass > 1.0e-12f && temp0 < 235.16f) {
            freeze_ice_rate = (double)rain_mass / (double)dt;
            freeze_ice_number_rate = (double)rain_number / (double)dt;
        }

        if (rain_mass >= 1.0e-6f && qs[idx] * rho >= 1.0e-6f) {
            const float incoming_snow_mass = qs[idx] * rho;
            const int snow_bin = thompson_decade_table_index(
                incoming_snow_mass, -6, 37);
            const int raw_temp_bin =
                (int)(__fdiv_rn(((temp0 - 273.15f) - 2.5f), 5.0f)) - 1;
            const int temp_bin = min(9, max(1, -raw_temp_bin)) - 1;
            const size_t table_idx = (size_t)snow_bin
                + (size_t)37 * ((size_t)temp_bin
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
                (double)(-rain_mass / dt), rain_snow_rain_rate);
            rain_snow_category_rate = fmax(
                (double)(-incoming_snow_mass / dt),
                rain_snow_category_rate);
            rain_snow_graupel_rate = fmin(
                (double)((rain_mass + incoming_snow_mass) / dt),
                rain_snow_graupel_rate);
            rain_snow_number_rate = fmin(
                (double)(rain_number / dt), rain_snow_number_rate);
        }

        const float incoming_graupel_mass = qg[idx] * rho;
        if (rain_mass >= 1.0e-6f && incoming_graupel_mass >= 1.0e-6f) {
            const int graupel_mass_bin = thompson_decade_table_index(
                incoming_graupel_mass, -6, 37);
            const float intercept_power = fmaxf(2.0f, fminf(6.0f,
                3.0f + (2.0f / 7.0f)
                    * (log10f(fmaxf(1.0e-9f, incoming_graupel_mass))
                       + 8.0f)));
            const int graupel_intercept_bin =
                thompson_decade_table_index_double(
                    (double)powf(10.0f, intercept_power), 2, 37);
            // One slab, never WRF's idx_bg1=5 subscript: thompson_racg_index.
            const size_t table_idx = thompson_racg_index(
                graupel_intercept_bin, graupel_mass_bin,
                rain_intercept_bin, rain_mass_bin);
            rain_graupel_graupel_rate =
                tmr_racg[table_idx] + tcr_gacr[table_idx];
            rain_graupel_graupel_rate = fmin(
                (double)(rain_mass / dt),
                rain_graupel_graupel_rate);
            rain_graupel_rain_rate =
                -rain_graupel_graupel_rate;
            rain_graupel_number_rate =
                tnr_racg[table_idx] + tnr_gacr[table_idx];
            rain_graupel_number_rate = fmin(
                (double)(rain_number / dt),
                rain_graupel_number_rate);
        }
    }

    if (include_cold_rain && nucleation_active) {
        // Classic WRF diagnoses rain freezing before Cooper nucleation.  Its
        // target crystal population therefore subtracts ice crystals created
        // by pni_rfz in this same call (and, in the full cloud-water group,
        // pni_wfz as well).  The focused frozen-vapor gate predates this
        // cross-group interaction, so retain its ABI above and apply the
        // exact ordering only for the admitted complete cold-rain group.
        const float existing_ice_number = qi[idx] > 1.0e-12f
            ? fmaxf(1.0e-6f, ni[idx] * rho) : 0.0f;
        const float target_number = fminf(
            250.0e3f, 5.0f * expf(0.304f * (273.15f - temp0)));
        nucleation_number_rate = (double)fmaxf(
            0.0f,
            target_number - existing_ice_number
                - (float)(freeze_ice_number_rate * (double)dt))
            * (double)inverse_dt;
        nucleation_rate = fmin(
            (double)vapor_limit, 1.0e-12 * nucleation_number_rate);
        nucleation_number_rate = nucleation_rate / 1.0e-12;
    }

    double snow_rate = 0.0;
    double snow_collection_rate = 0.0;
    double snow_collection_number_rate = 0.0;
    const float snow_mass = qs[idx] * rho;
    if (qs[idx] > 1.0e-12f) {
        const float rho_factor = sqrtf(
            (101325.0f / (287.05f * 298.0f)) * orho);
        const float rho_factor_sqrt = sqrtf(rho_factor);
        const float viscosity_factor = sqrtf(rho / viscosity);
        const float snow_second_moment = snow_mass * (1.0f / 0.069f);
        const float tc0 = fminf(-0.1f, tempc);
        const float snow_first_moment = thompson_field_a(tc0, 1.0f)
            * powf(snow_second_moment, thompson_field_b(tc0, 1.0f));
        const float deposition_moment =
            1.0f + (1.0f + 0.55f) * 0.5f;
        const float snow_ventilation_moment =
            thompson_field_a(tc0, deposition_moment)
            * powf(snow_second_moment,
                   thompson_field_b(tc0, deposition_moment));
        const float snow_riming_moment = thompson_field_a(tc0, 2.55f)
            * powf(snow_second_moment,
                   thompson_field_b(tc0, 2.55f));
        const float snow_capacitance = fmaxf(0.15f, fminf(
            0.15f + __fdiv_rn((tempc + 1.5f) * (0.5f - 0.15f),
                (-30.0f + 1.5f)), 0.5f));
        const float ventilation_coefficient = 0.28f
            * powf(0.632f, 1.0f / 3.0f) * sqrtf(40.0f);
        const float moment_sum = 0.86f * snow_first_moment
            + ventilation_coefficient * rho_factor_sqrt
              * viscosity_factor * snow_ventilation_moment;
        snow_rate = (double)(
            snow_capacitance * vapor_geometry * diffusivity * ssati
            * saturated_density * moment_sum);
        if (snow_rate > 0.0) {
            snow_rate = fmin(snow_rate, (double)vapor_limit);
        } else {
            snow_rate = fmax(
                (double)(-snow_mass * inverse_dt), snow_rate);
            snow_rate = fmax(snow_rate, (double)vapor_limit);
        }

        // Snow collecting cloud ice.  WRF diagnoses this sink from the same
        // incoming ice and snow distributions as the vapor terms, then
        // includes its mass (but not number) rate in the cloud-ice cap.
        if (qi[idx] > 1.0e-12f && snow_mass >= 1.0e-6f) {
            const float density_factor = sqrtf(
                (101325.0f / (287.05f * 298.0f)) * orho);
            snow_collection_rate = (double)(
                (3.1415926536f * 0.25f * 40.0f) * density_factor
                * 0.05f * ice_mass * snow_riming_moment);
            snow_collection_number_rate = snow_collection_rate
                / (double)ice_particle_mass;
        }
    }

    double graupel_rate = 0.0;
    const float graupel_mass = qg[idx] * rho;
    if (qg[idx] > 1.0e-12f && ssati < -1.0e-15f) {
        const float rho_factor = sqrtf(
            (101325.0f / (287.05f * 298.0f)) * orho);
        const float rho_factor_sqrt = sqrtf(rho_factor);
        const float viscosity_factor = sqrtf(rho / viscosity);
        const float am_g = 3.1415926536f * 400.0f / 6.0f;
        const float intercept_power = fmaxf(2.0f, fminf(
            3.0f + (2.0f / 7.0f)
                * (log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f),
            6.0f));
        const float diagnosed_intercept = powf(
            10.0f, intercept_power);
        float lambda = powf(
            diagnosed_intercept * am_g * 6.0f / graupel_mass, 0.25f);
        float number_per_kg = __fdiv_rn((1.0f / 6.0f) * graupel_mass
            * powf(lambda, 3.0f), am_g) / rho;
        number_per_kg = fmaxf(1.0e-6f, number_per_kg);
        float graupel_number = fmaxf(
            1.0e-6f, number_per_kg * rho);
        lambda = powf(
            am_g * 6.0f * graupel_number / graupel_mass,
            1.0f / 3.0f);
        float mvd = 3.672f / lambda;
        if (mvd > 25.4e-3f) {
            mvd = 25.4e-3f;
            lambda = 3.672f / mvd;
            graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass
                * powf(lambda, 3.0f), am_g);
        } else if (mvd < 50.0e-6f) {
            mvd = 50.0e-6f;
            lambda = 3.672f / mvd;
            graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass
                * powf(lambda, 3.0f), am_g);
        }
        const float inverse_lambda = 1.0f / lambda;
        const float graupel_intercept = graupel_number * lambda;
        const float ventilation_coefficient = 0.28f
            * powf(0.632f, 1.0f / 3.0f)
            * sqrtf(442.0f) * 1.9021706581115723f;
        const float moment_sum = graupel_intercept * (
            0.86f * powf(inverse_lambda, 2.0f)
            + ventilation_coefficient * viscosity_factor
              * rho_factor_sqrt * powf(inverse_lambda, 2.945f));
        graupel_rate = (double)(
            0.5f * vapor_geometry * diffusivity * ssati
            * saturated_density * moment_sum);
        graupel_rate = fmax(
            (double)(-graupel_mass * inverse_dt), graupel_rate);
        graupel_rate = fmax(graupel_rate, (double)vapor_limit);
    }

    double vapor_sum = nucleation_rate + ice_rate + ice_to_snow_rate
        + snow_rate + graupel_rate;
    if ((vapor_sum > 1.0e-15 && vapor_sum > (double)vapor_limit)
            || (vapor_sum < -1.0e-15
                && vapor_sum < (double)vapor_limit)) {
        const double ratio = (double)vapor_limit / vapor_sum;
        nucleation_rate *= ratio;
        ice_rate *= ratio;
        ice_to_snow_rate *= ratio;
        ice_number_rate *= ratio;
        snow_rate *= ratio;
        graupel_rate *= ratio;
        vapor_sum = (double)vapor_limit;
    }

    const double ice_limit = (double)(-ice_mass * inverse_dt);
    const double ice_sum = ice_rate - autoconversion_rate
        - snow_collection_rate - rain_ice_ice_rate;
    if (qi[idx] > 1.0e-12f && ice_sum < (double)-1.0e-15f
            && ice_sum < ice_limit) {
        const double ratio = ice_limit / ice_sum;
        const double unbounded_ice_rate = ice_rate;
        ice_rate *= ratio;
        autoconversion_rate *= ratio;
        snow_collection_rate *= ratio;
        rain_ice_ice_rate *= ratio;
        vapor_sum += ice_rate - unbounded_ice_rate;
        // Classic WRF intentionally does not scale pni_ide, pni_iau, or
        // pni_sci, or pni_rci in this cloud-ice mass conservation pass.
    }

    // Preserve the already admitted focused cold-ice ABI exactly when the
    // full cold-rain table group is disabled. Besides keeping that numerical
    // contract stable, this makes the production extension an explicit gate
    // rather than a silent behavior change to existing callers.
    if (!include_cold_rain) {
        const double rain_limit = (double)(-rain_mass * inverse_dt);
        if (-rain_ice_rain_rate < rain_limit) {
            rain_ice_rain_rate *= rain_limit / (-rain_ice_rain_rate);
        }

        qi[idx] = fmaxf(0.0f, qi[idx]
            + (float)((nucleation_rate + ice_rate - autoconversion_rate
                       - snow_collection_rate - rain_ice_ice_rate)
                      * (double)orho) * dt);
        ni[idx] = fmaxf(0.0f, ni[idx]
            + (float)((nucleation_number_rate + ice_number_rate
                       - autoconversion_number_rate
                       - snow_collection_number_rate
                       - rain_ice_ice_number_rate)
                      * (double)orho) * dt);
        qs[idx] = fmaxf(0.0f, qs[idx]
            + (float)((ice_to_snow_rate + snow_rate + autoconversion_rate
                       + snow_collection_rate) * (double)orho) * dt);
        qg[idx] = fmaxf(0.0f, qg[idx]
            + (float)((graupel_rate + rain_ice_graupel_rate)
                      * (double)orho) * dt);
        qr[idx] = fmaxf(0.0f, qr[idx]
            - (float)(rain_ice_rain_rate * (double)orho) * dt);
        nr[idx] = fmaxf(0.0f, nr[idx]
            - (float)((rain_self_number_rate + rain_ice_rain_number_rate)
                      * (double)orho) * dt);
        thompson_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
        thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
        qv[idx] = fmaxf(1.0e-10f,
            qv0 - (float)(vapor_sum * (double)orho) * dt);
        const float latent_vapor = 2.5e6f
            + (2106.0f - 4218.0f) * tempc;
        const float latent_fusion = 2.834e6f - latent_vapor;
        temperature[idx] = temp0 + (float)(
            (double)(2.834e6f * inverse_cp) * vapor_sum
            * (double)orho * (double)dt
            + (double)(latent_fusion * inverse_cp) * rain_ice_rain_rate
            * (double)orho * (double)dt);
        return;
    }

    // WRF's rain bound sees every cold-rain mass transfer at once. Number
    // rates and the rain/ice paired graupel source intentionally remain held.
    const double rain_limit = (double)(-rain_mass * inverse_dt);
    const double rain_sum = -freeze_graupel_rate - freeze_ice_rate
        - rain_ice_rain_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
    if (rain_active && rain_sum < rain_limit) {
        const double ratio = rain_limit / rain_sum;
        freeze_graupel_rate *= ratio;
        freeze_ice_rate *= ratio;
        rain_ice_rain_rate *= ratio;
        rain_snow_rain_rate *= ratio;
        rain_graupel_rain_rate *= ratio;
    }

    // The following snow and graupel passes occur after rain conservation.
    // Only the WRF-listed members scale; paired sources outside each group
    // are deliberately not reconstructed here.
    const double snow_limit = (double)(-snow_mass * inverse_dt);
    const double snow_sum = snow_rate + rain_snow_category_rate;
    if (qs[idx] > 1.0e-12f && snow_sum < snow_limit) {
        const double ratio = snow_limit / snow_sum;
        const double unbounded_snow_vapor_rate = snow_rate;
        snow_rate *= ratio;
        rain_snow_category_rate *= ratio;
        vapor_sum += snow_rate - unbounded_snow_vapor_rate;
    }

    const double graupel_limit = (double)(-graupel_mass * inverse_dt);
    const double graupel_sum =
        graupel_rate + rain_graupel_graupel_rate;
    if (qg[idx] > 1.0e-12f && graupel_sum < graupel_limit) {
        const double ratio = graupel_limit / graupel_sum;
        const double unbounded_graupel_vapor_rate = graupel_rate;
        graupel_rate *= ratio;
        rain_graupel_graupel_rate *= ratio;
        vapor_sum += graupel_rate - unbounded_graupel_vapor_rate;
    }

    // Blossey re-enforcement follows every species cap. Rain/snow is not
    // paired in this cold branch; rain/graupel is restored symmetrically.
    const double paired_rate = fmin(
        fabs(rain_graupel_rain_rate),
        fabs(rain_graupel_graupel_rate));
    rain_graupel_rain_rate = -paired_rate;
    rain_graupel_graupel_rate = paired_rate;

    const double rain_rate = -freeze_graupel_rate - freeze_ice_rate
        - rain_ice_rain_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
    const double rain_number_sink = rain_self_number_rate
        + freeze_graupel_number_rate + freeze_ice_number_rate
        + rain_ice_rain_number_rate + rain_snow_number_rate
        + rain_graupel_number_rate;

    // WRF applies every source tendency to temperature and vapor before its
    // hydrometeor mass/number balance.  The diameter bounds therefore use
    // the post-source density, not the immutable density used to diagnose
    // the rates above.  This distinction is material when several latent-
    // heating and frozen-vapor terms overlap in one call.
    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const double fusion_rate = freeze_ice_rate + freeze_graupel_rate
        + rain_ice_rain_rate + rain_snow_category_rate
        + rain_snow_graupel_rate + rain_graupel_graupel_rate;
    const float post_source_qv = fmaxf(1.0e-10f,
        qv0 - (float)(vapor_sum * (double)orho) * dt);
    const float post_source_temperature = temp0 + (float)(
        (double)(2.834e6f * inverse_cp) * vapor_sum
        * (double)orho * (double)dt
        + (double)(latent_fusion * inverse_cp) * fusion_rate
        * (double)orho * (double)dt);
    const float post_source_density = 0.622f * pressure[idx]
        / (287.04f * post_source_temperature
           * (post_source_qv + 0.622f));

    qi[idx] = fmaxf(0.0f, qi[idx]
        + (float)((nucleation_rate + freeze_ice_rate + ice_rate
                   - autoconversion_rate
                   - snow_collection_rate - rain_ice_ice_rate)
                  * (double)orho) * dt);
    ni[idx] = fmaxf(0.0f, ni[idx]
        + (float)((nucleation_number_rate + freeze_ice_number_rate
                   + ice_number_rate
                   - autoconversion_number_rate
                   - snow_collection_number_rate
                   - rain_ice_ice_number_rate)
                  * (double)orho) * dt);
    qs[idx] = fmaxf(0.0f, qs[idx]
        + (float)((ice_to_snow_rate + snow_rate + autoconversion_rate
                   + snow_collection_rate + rain_snow_category_rate)
                  * (double)orho) * dt);
    qg[idx] = fmaxf(0.0f, qg[idx]
        + (float)((graupel_rate + freeze_graupel_rate
                   + rain_ice_graupel_rate + rain_snow_graupel_rate
                   + rain_graupel_graupel_rate)
                  * (double)orho) * dt);
    qr[idx] = fmaxf(0.0f, qr[idx]
        + (float)(rain_rate * (double)orho) * dt);
    nr[idx] = fmaxf(0.0f, nr[idx]
        - (float)(rain_number_sink * (double)orho) * dt);
    thompson_bound_ice_number(
        qi[idx] * post_source_density, post_source_density, &ni[idx]);
    thompson_bound_rain_number(
        qr[idx] * post_source_density, post_source_density, &nr[idx]);
    qv[idx] = post_source_qv;
    temperature[idx] = post_source_temperature;
}

// Keep the source expressions unchanged while admitting two 256-thread
// blocks within the SM register file. The production launch uses 64 threads.
#ifdef __CUDACC_RTC__
#define THOMPSON_COLD_NETWORK_BOUNDS __launch_bounds__(256, 2)
#else
#define THOMPSON_COLD_NETWORK_BOUNDS
#endif
extern "C" __global__ THOMPSON_COLD_NETWORK_BOUNDS void thompson_frozen_vapor_cloud_network(
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* qc,
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
    float* __restrict__ graupel_number_shadow,
    float* __restrict__ snow_velocity_boost,
    int track_graupel_number,
    int include_snow_rime_conversion,
    int include_cold_rain,
    int include_cold_cloud,
    float dt, int size)
{
    // Simultaneous classic cold-ice group.  Non-aerosol Cooper nucleation,
    // ice/snow/graupel vapor rates, ice autoconversion, snow collection of
    // cloud ice, and rain collection of cloud ice are all diagnosed from one
    // incoming state before WRF's species limiters run in driver order.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    // WRF resets vts_boost on every cold-source call before diagnosing the
    // deposition-conditioned rimed-snow conversion.  Warm/empty cells must
    // also receive the neutral value because the later column sedimentation
    // kernel consumes the complete field.
    if (include_snow_rime_conversion) snow_velocity_boost[idx] = 1.0f;
    if (temperature[idx] >= 273.15f) {
        // :3033-3055, the cloud ice mass/number balance, runs at EVERY level
        // of a column with microphysics, not only below 0 C: ice that sits
        // at or above 0 C with no number gets the 300 micron number from the
        // balance's size clamp, and ice with a number outside 5..300 microns
        // is brought inside it.  The warm network owns no ice, so without
        // this such ice reached the fallout with no number and the ice
        // fallout gave it 5 micron crystals instead.  No source process
        // changes ice here, so the balance reads the entry ice at the entry
        // density (:1802), exactly as :3036-3037 do.
        if (qi[idx] > 0.0f || ni[idx] > 0.0f) {
            const float qv_entry = fmaxf(1.0e-10f, qv[idx]);
            const float rho_entry = 0.622f * pressure[idx]
                / (287.04f * temperature[idx] * (qv_entry + 0.622f));
            thompson_bound_ice_number(qi[idx] * rho_entry, rho_entry,
                                      &ni[idx]);
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
    const bool nucleation_active = ssati >= 0.25f
        || (ssatw > 1.0e-15f && temp0 < 253.15f);
    if (qi[idx] <= 1.0e-12f && qs[idx] <= 1.0e-12f
            && qg[idx] <= 1.0e-12f && qr[idx] <= 1.0e-12f
            && (!include_cold_cloud || qc[idx] <= 1.0e-12f)
            && !nucleation_active) return;

    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float inverse_dt = 1.0f / dt;
    const float tempc = temp0 - 273.15f;
    const float inverse_temp = 1.0f / temp0;
    const float diffusivity = 2.11e-5f
        * powf(__fdiv_rn(temp0, 273.15f), 1.94f)
        * (101325.0f / pressure[idx]);
    const float viscosity = (1.718f + 0.0049f * tempc
        - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
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
    float xsat = ssati;
    if (fabsf(xsat) < 1.0e-9f) xsat = 0.0f;
    const float xsat2 = xsat * xsat;
    const float alpha2 = alpha * alpha;
    const float vapor_geometry = 4.0f * 3.1415926536f
        * (1.0f - alpha * xsat
           + 2.0f * alpha2 * xsat2
           - 5.0f * alpha2 * alpha * xsat2 * xsat)
        / (1.0f + gamma);
    const float vapor_limit =
        (qv0 - qvsi) * rho * inverse_dt * 0.999f;

    // Classic non-aerosol Cooper (1986) deposition nucleation.  WRF first
    // constrains the nucleated mass by the per-process vapor allowance, then
    // includes that mass in the shared frozen-vapor cap below.  Its number
    // source is deliberately held when that later cap rescales mass.
    double nucleation_rate = 0.0;
    double nucleation_number_rate = 0.0;
    if (nucleation_active) {
        const float existing_ice_number = qi[idx] > 1.0e-12f
            ? fmaxf(1.0e-6f, ni[idx] * rho) : 0.0f;
        const float target_number = fminf(
            250.0e3f, 5.0f * expf(0.304f * (273.15f - temp0)));
        nucleation_number_rate = (double)(fmaxf(
            0.0f, target_number - existing_ice_number) * inverse_dt);
        nucleation_rate = fmin(
            (double)vapor_limit,
            1.0e-12 * nucleation_number_rate);
        nucleation_number_rate = nucleation_rate / 1.0e-12;
    }

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
    float ice_number = fmaxf(1.0e-6f, ni[idx] * rho);
    const float rain_mass = qr[idx] * rho;
    float rain_number;
    double rain_lambda;
    float rain_mvd;
    const bool rain_active = thompson_prepare_entry_rain_distribution(
        qr[idx], nr[idx], rho, &rain_number, &rain_lambda, &rain_mvd);
    if (rain_active) {
        if (rain_mvd > 50.0e-6f) {
            const float efficiency = 1.0f
                - expf(2300.0f * (rain_mvd - 1950.0e-6f));
            rain_self_number_rate = (double)(
                efficiency * 2.0f * rain_number * rain_mass);
        }
    }
    if (qi[idx] > 1.0e-12f) {
        const float am_i = 3.1415926536f * 890.0f / 6.0f;
        // :1855-1858.  Ice that arrives with no number (ni*rho <= R2) is
        // given 5-micron crystals BEFORE the size clamps below.  Without it
        // the R2 floor makes the mean diameter enormous, the 300-micron clamp
        // fires instead, and the ice sees a far smaller number.  The
        // deposition, ice autoconversion, snow-collects-ice and
        // rain-collects-ice rates and the nucleation target below all read
        // this entry number (the mp=28 cold network carries the same rule).
        if (ice_number <= 1.0e-6f) {
            const double lambda5 = 4.0 / 5.0e-6;
            ice_number = fminf(999.0e3f,
                __fdiv_rn((1.0f / 6.0f) * ice_mass, am_i)
                * (float)(lambda5 * lambda5 * lambda5));
        }
        double lambda = (double)powf(
            am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f);
        double inverse_lambda = 1.0 / lambda;
        float mean_diameter = (float)(4.0 * inverse_lambda);
        if (mean_diameter < 5.0e-6f) {
            lambda = 4.0 / 5.0e-6;
            inverse_lambda = 1.0 / lambda;
            ice_number = fminf(999.0e3f,
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
        // :2649/:2713, xDi = MAX(D0i, (bm_i+mu_i+1)*ilami) with D0i =
        // (xm0i/am_i)**(1/bm_i), about 12.9 microns.  xmi, the particle mass
        // that turns collected ice mass into a number (pni_sci, pni_rci),
        // and the rain-collects-ice gate mvd_r > 4*xDi both read the FLOORED
        // diameter, as the deposition branch below already does for its own
        // copy (the mp=28 cold network carries the same rule).
        const float floored_ice_diameter = fmaxf(
            powf(1.0e-12f / am_i, 1.0f / 3.0f), mean_diameter);
        ice_particle_mass = am_i * floored_ice_diameter
            * floored_ice_diameter * floored_ice_diameter;
        const int mass_bin = ice_mass > 1.0e-10f
            ? thompson_decade_table_index(ice_mass, -10, 64) : 0;
        const int number_bin = ice_number > 1.0f
            ? thompson_decade_table_index(ice_number, 0, 55) : 0;
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
            total_rate = fmax(
                (double)(-ice_mass * inverse_dt), total_rate);
            total_rate = fmax(total_rate, (double)vapor_limit);
            const float minimum_diameter = powf(
                1.0e-12f / am_i, 1.0f / 3.0f);
            const float particle_diameter = fmaxf(
                minimum_diameter, mean_diameter);
            const float particle_mass = am_i * particle_diameter
                * particle_diameter * particle_diameter;
            ice_number_rate = total_rate / (double)particle_mass;
            ice_number_rate = fmax(
                (double)(-ice_number * inverse_dt), ice_number_rate);
            ice_rate = total_rate;
        }

        // Lookup-table cloud-ice to snow autoconversion is diagnosed from
        // the same incoming ice distribution as deposition and collection.
        const size_t table_idx = (size_t)mass_bin
            + (size_t)64 * (size_t)number_bin;
        if (mass_bin == 63 || mean_diameter > 1500.0e-6f) {
            autoconversion_rate = (double)(ice_mass * 0.99f)
                * (double)inverse_dt;
            autoconversion_number_rate = (double)(ice_number * 0.95f)
                * (double)inverse_dt;
        } else if (mean_diameter >= 30.0e-6f) {
            autoconversion_rate = fmin(
                (double)(ice_mass * 0.99f) * (double)inverse_dt,
                ice_to_snow_mass[table_idx] * (double)inverse_dt);
            autoconversion_number_rate = fmin(
                (double)(ice_number * 0.95f) * (double)inverse_dt,
                ice_to_snow_number[table_idx] * (double)inverse_dt);
        }

        if (rain_mass >= 1.0e-6f
                && rain_mvd > 4.0f * floored_ice_diameter) {
            const float density_factor = sqrtf(
                (101325.0f / (287.05f * 298.0f)) * orho);
            const double rain_intercept =
                (double)rain_number * rain_lambda;
            const double shifted_lambda = rain_lambda + 195.0;
            const float collection_efficiency = 0.95f;
            const float mass_prefactor =
                3.1415926536f * 0.25f * 4854.0f * 6.0f;
            const float am_r = 3.1415926536f * 1000.0f / 6.0f;
            const float rain_mass_prefactor =
                3.1415926536f * 0.25f * am_r * 4854.0f * 720.0f;
            rain_ice_ice_rate = (double)(
                density_factor * mass_prefactor
                * collection_efficiency * ice_mass)
                * rain_intercept * pow(shifted_lambda, -4.0);
            rain_ice_rain_rate = (double)(
                density_factor * rain_mass_prefactor
                * collection_efficiency * ice_number)
                * rain_intercept * pow(shifted_lambda, -7.0);
            rain_ice_rain_rate = fmin(
                rain_ice_rain_rate,
                (double)rain_mass * (double)inverse_dt);
            rain_ice_ice_number_rate =
                rain_ice_ice_rate / (double)ice_particle_mass;
            rain_ice_rain_number_rate = (double)(
                density_factor * mass_prefactor
                * collection_efficiency * ice_number)
                * rain_intercept * pow(shifted_lambda, -4.0);
            rain_ice_rain_number_rate = fmin(
                rain_ice_rain_number_rate,
                (double)rain_number * (double)inverse_dt);
            // WRF forms this paired graupel source before either later mass
            // cap and never reconstructs it from the bounded sink rates.
            rain_ice_graupel_rate =
                rain_ice_ice_rate + rain_ice_rain_rate;
        }
    }

    // Optional completion of the cold-rain source group. These rates share
    // the same immutable incoming rain/ice/snow/graupel distributions as the
    // vapor, autoconversion, and rain/ice rates above. The grouped species
    // caps are applied only after every member has been diagnosed.
    (void)tcg_racg;
    double freeze_ice_rate = 0.0;
    double freeze_ice_number_rate = 0.0;
    double freeze_graupel_rate = 0.0;
    double freeze_graupel_number_rate = 0.0;
    double rain_snow_rain_rate = 0.0;
    double rain_snow_category_rate = 0.0;
    double rain_snow_graupel_rate = 0.0;
    double rain_snow_number_rate = 0.0;
    double rain_graupel_rain_rate = 0.0;
    double rain_graupel_graupel_rate = 0.0;
    double rain_graupel_number_rate = 0.0;
    if (include_cold_rain && rain_active) {
        const float pi = 3.1415926536f;
        const float am_r = pi * 1000.0f / 6.0f;
        const int rain_mass_bin = thompson_decade_table_index(
            rain_mass, -6, 37);
        const double table_rain_intercept =
            (double)(__fdiv_rn((1.0f / 6.0f) * rain_mass, am_r))
            * rain_lambda * rain_lambda * rain_lambda * rain_lambda;
        const int rain_intercept_bin =
            thompson_decade_table_index_double(
                table_rain_intercept, 6, 37);

        // Bigg table freezing. WRF holds all associated number rates if the
        // later grouped rain-mass bound rescales these mass transfers.
        if (rain_mass > 1.0e-6f) {
            const int temp_bin = max(0, min(
                (int)roundf(-(temp0 - 273.15f)) - 1, 44));
            const int nuclei_bin = 27;
            const size_t table_idx = (size_t)rain_mass_bin
                + (size_t)37 * ((size_t)rain_intercept_bin
                + (size_t)37 * ((size_t)temp_bin
                + (size_t)45 * (size_t)nuclei_bin));
            freeze_ice_rate = rain_to_ice_mass[table_idx] / (double)dt;
            freeze_ice_number_rate =
                rain_to_ice_number[table_idx] / (double)dt;
            freeze_graupel_rate =
                rain_to_graupel_mass[table_idx] / (double)dt;
            freeze_graupel_number_rate = fmin(
                (double)rain_number,
                rain_to_graupel_number[table_idx]) / (double)dt;
        } else if (rain_mass > 1.0e-12f && temp0 < 235.16f) {
            freeze_ice_rate = (double)rain_mass / (double)dt;
            freeze_ice_number_rate = (double)rain_number / (double)dt;
        }

        if (rain_mass >= 1.0e-6f && qs[idx] * rho >= 1.0e-6f) {
            const float incoming_snow_mass = qs[idx] * rho;
            const int snow_bin = thompson_decade_table_index(
                incoming_snow_mass, -6, 37);
            const int raw_temp_bin =
                (int)(__fdiv_rn(((temp0 - 273.15f) - 2.5f), 5.0f)) - 1;
            const int temp_bin = min(9, max(1, -raw_temp_bin)) - 1;
            const size_t table_idx = (size_t)snow_bin
                + (size_t)37 * ((size_t)temp_bin
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
                (double)(-rain_mass / dt), rain_snow_rain_rate);
            rain_snow_category_rate = fmax(
                (double)(-incoming_snow_mass / dt),
                rain_snow_category_rate);
            rain_snow_graupel_rate = fmin(
                (double)((rain_mass + incoming_snow_mass) / dt),
                rain_snow_graupel_rate);
            rain_snow_number_rate = fmin(
                (double)(rain_number / dt), rain_snow_number_rate);
        }

        const float incoming_graupel_mass = qg[idx] * rho;
        if (rain_mass >= 1.0e-6f && incoming_graupel_mass >= 1.0e-6f) {
            const int graupel_mass_bin = thompson_decade_table_index(
                incoming_graupel_mass, -6, 37);
            const float intercept_power = fmaxf(2.0f, fminf(6.0f,
                3.0f + (2.0f / 7.0f)
                    * (log10f(fmaxf(1.0e-9f, incoming_graupel_mass))
                       + 8.0f)));
            const int graupel_intercept_bin =
                thompson_decade_table_index_double(
                    (double)powf(10.0f, intercept_power), 2, 37);
            // One slab, never WRF's idx_bg1=5 subscript: thompson_racg_index.
            const size_t table_idx = thompson_racg_index(
                graupel_intercept_bin, graupel_mass_bin,
                rain_intercept_bin, rain_mass_bin);
            rain_graupel_graupel_rate =
                tmr_racg[table_idx] + tcr_gacr[table_idx];
            rain_graupel_graupel_rate = fmin(
                (double)(rain_mass / dt),
                rain_graupel_graupel_rate);
            rain_graupel_rain_rate =
                -rain_graupel_graupel_rate;
            rain_graupel_number_rate =
                tnr_racg[table_idx] + tnr_gacr[table_idx];
            rain_graupel_number_rate = fmin(
                (double)(rain_number / dt),
                rain_graupel_number_rate);
        }
    }

    // Optional completion of the cold cloud-water source group.  These
    // rates deliberately read the same incoming state as every cold-rain
    // and frozen-vapor process above.  In particular, pni_wfz must exist
    // before Cooper nucleation is diagnosed below.
    const float cloud_mass = include_cold_cloud
        ? fmaxf(0.0f, qc[idx] * rho) : 0.0f;
    const float snow_mass = qs[idx] * rho;
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
    float cloud_mvd = 1.0e-6f;
    float snow_rime_moment_zero = 0.0f;
    float snow_rime_diameter = 0.0f;
    if (include_cold_cloud) {
        const float pi = 3.1415926536f;
        const float am_r = pi * 1000.0f / 6.0f;
        const float cloud_number = 100.0e6f;
        const float gamma_mass = 1.30767389e12f;
        const float inverse_gamma_number = 2.08767448e-9f;
        const float gamma_higher = 6.40238373e15f;
        const float inverse_gamma_mass = 7.64716632e-13f;

        // Cloud presence is WRF's L_qc, the MIXING RATIO test of the entry
        // block (:1827), not the concentration: in thin air a mixing ratio
        // just above R1 is a concentration at or below it, and WRF still
        // forms mvd_c there (:2169-2175) and lets rain, snow and graupel
        // collect that cloud (:2194, :2403, :2415).  The mp=28 networks test
        // the mixing ratio too.
        const bool l_qc = qc[idx] > 1.0e-12f;
        float cloud_lambda = 0.0f;
        if (l_qc) {
            cloud_lambda = powf(
                cloud_number * am_r * gamma_mass * inverse_gamma_number
                    / cloud_mass,
                1.0f / 3.0f);
            cloud_mvd = fmaxf(1.0e-6f, fminf(
                15.672f / cloud_lambda, 50.0e-6f));
        }

        // Berry-Reinhardt warm-rain autoconversion is part of the same
        // cloud-water conservation group even below freezing.
        if (cloud_mass > 0.01e-3f) {
            const float xdc = fmaxf(1.0f,
                powf(__fdiv_rn(cloud_mass, am_r * cloud_number), 1.0f / 3.0f)
                    * 1.0e6f);
            const float dcg = powf(
                gamma_higher * inverse_gamma_mass, 1.0f / 3.0f)
                / cloud_lambda * 1.0e6f;
            const float xdc3 = xdc * xdc * xdc;
            const float dcg3 = dcg * dcg * dcg;
            const float dcb = powf(fmaxf(
                0.0f, xdc3 * dcg3 - xdc3 * xdc3), 1.0f / 6.0f);
            const float zeta_term =
                6.25e-6f * xdc * dcb * dcb * dcb - 0.4f;
            const float zeta1 = 0.5f * (zeta_term + fabsf(zeta_term));
            const float zeta = 0.027f * cloud_mass * zeta1;
            const float tau_diameter = 0.5f * dcb - 7.5f;
            const float taud = 0.5f
                * (tau_diameter + fabsf(tau_diameter)) + 1.0e-12f;
            const float tau = 3.72f / (cloud_mass * taud);
            cloud_autoconversion_rate = fmin(
                (double)(cloud_mass / dt), (double)(zeta / tau));
            cloud_autoconversion_number_rate =
                cloud_autoconversion_rate
                / (double)(am_r * 12.0f * 10.0f
                           * 50.0e-6f * 50.0e-6f * 50.0e-6f);
        }

        // Rain self-collection was already diagnosed once above.  Only the
        // rain/cloud mass collection member is added here.
        // :2194, L_qr .and. mvd_r > D0r .and. mvd_c > D0c: rain presence
        // is the mixing ratio's too (rain_active, :1878).
        if (rain_active && l_qc
                && rain_mvd > 50.0e-6f && cloud_mvd > 1.0e-6f) {
            const double dr_first = 5.1164649614037726e-05;
            const double dr_last = 0.004886186104779057;
            int rain_bin = 1 + (int)(100.0
                * log((double)rain_mvd / dr_first)
                / log(dr_last / dr_first));
            rain_bin = min(rain_bin, 100);
            const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
            const float efficiency = (float)rain_cloud_efficiency[
                (rain_bin - 1) + 100 * (cloud_bin - 1)];
            const float density_factor = sqrtf(
                (101325.0f / (287.05f * 298.0f)) / rho);
            const float coefficient = pi * 0.25f * 4854.0f * 6.0f;
            const float intercept = rain_number * (float)rain_lambda;
            const float rate = density_factor * coefficient * efficiency
                * cloud_mass * intercept
                * powf((float)rain_lambda + 195.0f, -4.0f);
            cloud_rain_accretion_rate = fmin(
                (double)(cloud_mass / dt), (double)rate);
        }

        // Bigg cloud-drop freezing and its paired crystal source.
        if (cloud_mass > 1.0e-6f) {
            const int mass_bin = thompson_decade_table_index(
                cloud_mass, -6, 37);
            const int cloud_number_bin = 65;
            const int temp_bin = max(0, min(
                (int)roundf(-tempc) - 1, 44));
            const int nuclei_bin = 27;
            const size_t table_idx = (size_t)mass_bin
                + (size_t)37 * ((size_t)cloud_number_bin
                + (size_t)100 * ((size_t)temp_bin
                + (size_t)45 * (size_t)nuclei_bin));
            cloud_freezing_rate = fmin(
                (double)(cloud_mass / dt),
                cloud_to_ice_mass[table_idx] / (double)dt);
            cloud_freezing_number_rate = fmin(
                (double)(cloud_number / dt),
                fmin(cloud_freezing_rate / (2.0 * 1.0e-12),
                     cloud_to_ice_number[table_idx] / (double)dt));
        } else if (cloud_mass > 1.0e-12f && temp0 < 235.16f) {
            cloud_freezing_rate = (double)cloud_mass / (double)dt;
            cloud_freezing_number_rate =
                (double)cloud_number / (double)dt;
        }

        const float cloud_snow_mass = fmaxf(0.0f, qs[idx] * rho);
        // :2403 with :2245-2251: L_qc, mvd_c > D0c, and xDs > D0s, where
        // xDs is zero unless L_qs.
        if (l_qc && qs[idx] > 1.0e-12f
                && cloud_mvd > 1.0e-6f) {
            const float snow_temperature = fminf(-0.1f, tempc);
            const float snow_moment_2 = cloud_snow_mass * (1.0f / 0.069f);
            const float snow_moment_3 = thompson_field_a(
                snow_temperature, 3.0f) * powf(
                    snow_moment_2,
                    thompson_field_b(snow_temperature, 3.0f));
            snow_rime_moment_zero = thompson_field_a(
                snow_temperature, 0.0f) * powf(
                    snow_moment_2,
                    thompson_field_b(snow_temperature, 0.0f));
            snow_rime_diameter = snow_moment_3 / snow_moment_2;
            if (snow_rime_diameter > 300.0e-6f) {
                const float snow_riming_moment = thompson_field_a(
                    snow_temperature, 2.55f) * powf(
                        snow_moment_2,
                        thompson_field_b(snow_temperature, 2.55f));
                const double diameter_ratio = 0.02 / 300.0e-6;
                const double log_ratio = log(diameter_ratio);
                const double first_snow_bin = 300.0e-6
                    * exp(0.5 / 100.0 * log_ratio);
                const double last_snow_bin = 300.0e-6
                    * exp(99.5 / 100.0 * log_ratio);
                int snow_bin = 1 + (int)(100.0
                    * log((double)snow_rime_diameter / first_snow_bin)
                    / log(last_snow_bin / first_snow_bin));
                snow_bin = min(snow_bin, 100);
                const double table_snow_diameter = 300.0e-6
                    * exp(((double)snow_bin - 0.5) / 100.0 * log_ratio);
                const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
                const double table_cloud_diameter =
                    (double)cloud_bin * 1.0e-6;
                const double cloud_velocity = 1.19e4
                    * (1.0e4 * table_cloud_diameter
                       * table_cloud_diameter * 0.25);
                const double snow_velocity = 40.0
                    * pow(table_snow_diameter, 0.55)
                    * exp(-100.0 * table_snow_diameter) - cloud_velocity;
                const double melted_snow_diameter = pow(
                    0.069 * table_snow_diameter * table_snow_diameter
                        / (pi * 1000.0 / 6.0), 1.0 / 3.0);
                const double diameter_fraction = table_cloud_diameter
                    / melted_snow_diameter;
                float efficiency = 0.0f;
                if (diameter_fraction <= 0.25
                        && table_cloud_diameter >= (double)6.0e-6f
                        && snow_velocity >= (double)1.0e-3f) {
                    const double stokes_number = table_cloud_diameter
                        * table_cloud_diameter * snow_velocity * 1000.0
                        / (9.0 * 1.718e-5 * melted_snow_diameter);
                    const double reynolds_number = 9.0 * stokes_number
                        / (diameter_fraction * diameter_fraction * 1000.0);
                    const double log_reynolds = log(reynolds_number);
                    const double k0 = exp(-0.1007 - 0.358 * log_reynolds
                        + 0.0261 * log_reynolds * log_reynolds);
                    const double z = log(stokes_number / (k0 + 1.0e-15));
                    const double h = 0.1465 + 1.302 * z - 0.607 * z * z
                        + 0.293 * z * z * z;
                    const double yc0 = 2.0 / 3.14159265358979323846
                        * atan(h);
                    const double value = (yc0 + diameter_fraction)
                        * (yc0 + diameter_fraction)
                        / ((1.0 + diameter_fraction)
                           * (1.0 + diameter_fraction));
                    efficiency = fmaxf(
                        0.0f, fminf((float)value, 0.95f));
                }
                const float density_factor = sqrtf(
                    (101325.0f / (287.05f * 298.0f)) / rho);
                const float prefactor = pi * 0.25f * 40.0f;
                snow_riming_rate = (double)(density_factor * prefactor
                    * efficiency * cloud_mass * snow_riming_moment);
                snow_riming_rate = fmin(
                    snow_riming_rate, (double)cloud_mass / (double)dt);
            }
        }

        const float cloud_graupel_mass = fmaxf(0.0f, qg[idx] * rho);
        if (l_qc && cloud_graupel_mass >= 1.0e-6f
                && cloud_mvd > 1.0e-6f) {
            const float am_g = pi * 400.0f / 6.0f;
            const float intercept_power = fmaxf(2.0f, fminf(6.0f,
                3.0f + (2.0f / 7.0f)
                    * (log10f(fmaxf(1.0e-9f, cloud_graupel_mass))
                       + 8.0f)));
            const float intercept_guess = powf(10.0f, intercept_power);
            double lambda = (double)powf(
                intercept_guess * am_g * 6.0f / cloud_graupel_mass,
                0.25f);
            float mvd = (float)(3.672 / lambda);
            if (mvd > 25.4e-3f) {
                lambda = 3.672 / 25.4e-3;
            } else if (mvd < 50.0e-6f) {
                lambda = 3.672 / 50.0e-6;
            }
            const double inverse_lambda = 1.0 / lambda;
            const float graupel_number = __fdiv_rn((1.0f / 6.0f)
                * cloud_graupel_mass, am_g)
                * (float)(lambda * lambda * lambda);
            const double intercept = (double)graupel_number * lambda;
            const float density_factor = sqrtf(
                (101325.0f / (287.05f * 298.0f)) / rho);
            const float cloud_viscosity = (1.718f + 0.0049f * tempc
                - 1.2e-5f * tempc * tempc) * 1.0e-5f;
            const float diameter = (float)(4.0 * inverse_lambda);
            const float velocity = (float)(
                (double)(density_factor * 442.0f * 20.3632278f
                         * (1.0f / 6.0f))
                * pow(inverse_lambda, 0.89));
            const float stokes_number = cloud_mvd * cloud_mvd
                * velocity * 1000.0f
                / (9.0f * cloud_viscosity * diameter);
            float efficiency = 0.0f;
            if (stokes_number >= 0.4f && stokes_number <= 10.0f) {
                efficiency = 0.55f * log10f(2.51f * stokes_number);
            } else if (stokes_number > 10.0f) {
                efficiency = 0.77f;
            }
            const float prefactor = pi * 0.25f * 442.0f * 5.23476267f;
            graupel_riming_rate = (double)(density_factor * prefactor
                * efficiency * cloud_mass)
                * intercept * pow(inverse_lambda, 3.89);
            // WRF leaves prg_gcw raw until the joint cloud-water
            // conservation pass below.  An individual cloud-mass cap here
            // changes the relative allocation between graupel riming,
            // autoconversion, and rain accretion.
        }

        // H-M number/mass tendencies remain held when the later cloud-water
        // cap rescales the riming mass sources, matching classic WRF.
        if (graupel_riming_rate > 1.0e-15 && tempc > -8.0f) {
            float factor = 0.0f;
            if (tempc >= -5.0f && tempc < -3.0f) {
                factor = 0.5f * (-3.0f - tempc);
            } else if (tempc > -8.0f && tempc < -5.0f) {
                factor = 0.33333333f * (8.0f + tempc);
            }
            hm_number_rate =
                3.5e8 * (double)factor * graupel_riming_rate;
            hm_mass_rate = 1.0e-12 * hm_number_rate;
            const double total_riming =
                snow_riming_rate + graupel_riming_rate;
            if (total_riming > 0.0) {
                snow_hm_rate =
                    snow_riming_rate / total_riming * hm_mass_rate;
                graupel_hm_rate =
                    graupel_riming_rate / total_riming * hm_mass_rate;
            }
        }
    }

    if ((include_cold_rain || include_cold_cloud)
            && nucleation_active) {
        // Classic WRF diagnoses rain freezing before Cooper nucleation.  Its
        // target crystal population therefore subtracts ice crystals created
        // by pni_rfz in this same call (and, in the full cloud-water group,
        // pni_wfz as well).  The focused frozen-vapor gate predates this
        // cross-group interaction, so retain its ABI above and apply the
        // exact ordering only for the admitted complete cold-rain group.
        // xni = ni(k) + ... (:2627) reads WRF's ENTRY ice number ni(k):
        // R2 where there is no ice (:1873), and otherwise the number after
        // the :1855-1858 re-diagnosis and the 5..300 micron clamps of
        // :1860-1868, which is ice_number above.  The raw ni*rho made a
        // number-less ice level look empty and out-of-range ice look like
        // its unclamped number.
        const float existing_ice_number = ice_number;
        const float target_number = fminf(
            250.0e3f, 5.0f * expf(0.304f * (273.15f - temp0)));
        nucleation_number_rate = (double)fmaxf(
            0.0f,
            target_number - existing_ice_number
                - (float)((freeze_ice_number_rate
                           + cloud_freezing_number_rate) * (double)dt))
            * (double)inverse_dt;
        nucleation_rate = fmin(
            (double)vapor_limit, 1.0e-12 * nucleation_number_rate);
        nucleation_number_rate = nucleation_rate / 1.0e-12;
    }

    double snow_rate = 0.0;
    double snow_collection_rate = 0.0;
    double snow_collection_number_rate = 0.0;
    if (qs[idx] > 1.0e-12f) {
        const float rho_factor = sqrtf(
            (101325.0f / (287.05f * 298.0f)) * orho);
        const float rho_factor_sqrt = sqrtf(rho_factor);
        const float viscosity_factor = sqrtf(rho / viscosity);
        const float snow_second_moment = snow_mass * (1.0f / 0.069f);
        const float tc0 = fminf(-0.1f, tempc);
        const float snow_first_moment = thompson_field_a(tc0, 1.0f)
            * powf(snow_second_moment, thompson_field_b(tc0, 1.0f));
        const float deposition_moment =
            1.0f + (1.0f + 0.55f) * 0.5f;
        const float snow_ventilation_moment =
            thompson_field_a(tc0, deposition_moment)
            * powf(snow_second_moment,
                   thompson_field_b(tc0, deposition_moment));
        const float snow_riming_moment = thompson_field_a(tc0, 2.55f)
            * powf(snow_second_moment,
                   thompson_field_b(tc0, 2.55f));
        const float snow_capacitance = fmaxf(0.15f, fminf(
            0.15f + __fdiv_rn((tempc + 1.5f) * (0.5f - 0.15f),
                (-30.0f + 1.5f)), 0.5f));
        const float ventilation_coefficient = 0.28f
            * powf(0.632f, 1.0f / 3.0f) * sqrtf(40.0f);
        const float moment_sum = 0.86f * snow_first_moment
            + ventilation_coefficient * rho_factor_sqrt
              * viscosity_factor * snow_ventilation_moment;
        snow_rate = (double)(
            snow_capacitance * vapor_geometry * diffusivity * ssati
            * saturated_density * moment_sum);
        if (snow_rate > 0.0) {
            snow_rate = fmin(snow_rate, (double)vapor_limit);
        } else {
            snow_rate = fmax(
                (double)(-snow_mass * inverse_dt), snow_rate);
            snow_rate = fmax(snow_rate, (double)vapor_limit);
        }

        // Snow collecting cloud ice.  WRF diagnoses this sink from the same
        // incoming ice and snow distributions as the vapor terms, then
        // includes its mass (but not number) rate in the cloud-ice cap.
        if (qi[idx] > 1.0e-12f && snow_mass >= 1.0e-6f) {
            const float density_factor = sqrtf(
                (101325.0f / (287.05f * 298.0f)) * orho);
            snow_collection_rate = (double)(
                (3.1415926536f * 0.25f * 40.0f) * density_factor
                * 0.05f * ice_mass * snow_riming_moment);
            snow_collection_number_rate = snow_collection_rate
                / (double)ice_particle_mass;
        }
    }

    // WRF v4.6.1 module_mp_thompson.F:2754-2777. Diagnose the conversion
    // from the unbounded incoming-state snow deposition and riming rates,
    // before the later vapor/cloud conservation passes. The private number
    // source png_scw is intentionally held if the cloud-water mass cap later
    // rescales the paired prg_scw mass source.
    double snow_graupel_conversion_rate = 0.0;
    double snow_graupel_conversion_number_rate = 0.0;
    if (include_snow_rime_conversion
            && snow_riming_rate > 2.0 * snow_rate
            && snow_rate > 1.0e-15) {
        const float rime_ratio = (float)fmin(
            30.0, snow_riming_rate / snow_rate);
        float graupel_fraction = fminf(
            0.95f, 0.15f + (rime_ratio - 2.0f) * 0.028f);
        snow_velocity_boost[idx] = fminf(
            1.5f, 1.1f + (rime_ratio - 2.0f) * 0.014f);
        snow_graupel_conversion_rate =
            (double)graupel_fraction * snow_riming_rate;
        snow_graupel_conversion_number_rate =
            snow_graupel_conversion_rate
            * (double)snow_rime_moment_zero / (double)snow_mass;

        const float snow_velocity = 40.0f
            * powf(snow_rime_diameter, 0.55f)
            * expf(-100.0f * snow_rime_diameter);
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
        snow_riming_rate =
            (1.0 - (double)graupel_fraction) * snow_riming_rate;
    }

    double graupel_rate = 0.0;
    const float graupel_mass = qg[idx] * rho;
    if (qg[idx] > 1.0e-12f && ssati < -1.0e-15f) {
        const float rho_factor = sqrtf(
            (101325.0f / (287.05f * 298.0f)) * orho);
        const float rho_factor_sqrt = sqrtf(rho_factor);
        const float viscosity_factor = sqrtf(rho / viscosity);
        const float am_g = 3.1415926536f * 400.0f / 6.0f;
        const float intercept_power = fmaxf(2.0f, fminf(
            3.0f + (2.0f / 7.0f)
                * (log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f),
            6.0f));
        const float diagnosed_intercept = powf(
            10.0f, intercept_power);
        float lambda = powf(
            diagnosed_intercept * am_g * 6.0f / graupel_mass, 0.25f);
        float number_per_kg = __fdiv_rn((1.0f / 6.0f) * graupel_mass
            * powf(lambda, 3.0f), am_g) / rho;
        number_per_kg = fmaxf(1.0e-6f, number_per_kg);
        float graupel_number = fmaxf(
            1.0e-6f, number_per_kg * rho);
        lambda = powf(
            am_g * 6.0f * graupel_number / graupel_mass,
            1.0f / 3.0f);
        float mvd = 3.672f / lambda;
        if (mvd > 25.4e-3f) {
            mvd = 25.4e-3f;
            lambda = 3.672f / mvd;
            graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass
                * powf(lambda, 3.0f), am_g);
        } else if (mvd < 50.0e-6f) {
            mvd = 50.0e-6f;
            lambda = 3.672f / mvd;
            graupel_number = __fdiv_rn((1.0f / 6.0f) * graupel_mass
                * powf(lambda, 3.0f), am_g);
        }
        const float inverse_lambda = 1.0f / lambda;
        const float graupel_intercept = graupel_number * lambda;
        const float ventilation_coefficient = 0.28f
            * powf(0.632f, 1.0f / 3.0f)
            * sqrtf(442.0f) * 1.9021706581115723f;
        const float moment_sum = graupel_intercept * (
            0.86f * powf(inverse_lambda, 2.0f)
            + ventilation_coefficient * viscosity_factor
              * rho_factor_sqrt * powf(inverse_lambda, 2.945f));
        graupel_rate = (double)(
            0.5f * vapor_geometry * diffusivity * ssati
            * saturated_density * moment_sum);
        graupel_rate = fmax(
            (double)(-graupel_mass * inverse_dt), graupel_rate);
        graupel_rate = fmax(graupel_rate, (double)vapor_limit);
    }

    double vapor_sum = nucleation_rate + ice_rate + ice_to_snow_rate
        + snow_rate + graupel_rate;
    if ((vapor_sum > 1.0e-15 && vapor_sum > (double)vapor_limit)
            || (vapor_sum < -1.0e-15
                && vapor_sum < (double)vapor_limit)) {
        const double ratio = (double)vapor_limit / vapor_sum;
        nucleation_rate *= ratio;
        ice_rate *= ratio;
        ice_to_snow_rate *= ratio;
        ice_number_rate *= ratio;
        snow_rate *= ratio;
        graupel_rate *= ratio;
        vapor_sum = (double)vapor_limit;
    }

    // WRF conserves cloud water immediately after the shared frozen-vapor
    // pass.  The paired number and H-M tendencies intentionally remain held
    // when these mass rates are rescaled.
    const double cloud_limit = (double)(-cloud_mass * inverse_dt);
    const double cloud_sum = -cloud_autoconversion_rate
        - cloud_freezing_rate - cloud_rain_accretion_rate
        - snow_riming_rate - snow_graupel_conversion_rate
        - graupel_riming_rate;
    if (include_cold_cloud && qc[idx] > 1.0e-12f
            && cloud_sum < cloud_limit) {
        const double ratio = cloud_limit / cloud_sum;
        cloud_autoconversion_rate *= ratio;
        cloud_freezing_rate *= ratio;
        cloud_rain_accretion_rate *= ratio;
        snow_riming_rate *= ratio;
        snow_graupel_conversion_rate *= ratio;
        graupel_riming_rate *= ratio;
    }

    const double ice_limit = (double)(-ice_mass * inverse_dt);
    const double ice_sum = ice_rate - autoconversion_rate
        - snow_collection_rate - rain_ice_ice_rate;
    if (qi[idx] > 1.0e-12f && ice_sum < (double)-1.0e-15f
            && ice_sum < ice_limit) {
        const double ratio = ice_limit / ice_sum;
        const double unbounded_ice_rate = ice_rate;
        ice_rate *= ratio;
        autoconversion_rate *= ratio;
        snow_collection_rate *= ratio;
        rain_ice_ice_rate *= ratio;
        vapor_sum += ice_rate - unbounded_ice_rate;
        // Classic WRF intentionally does not scale pni_ide, pni_iau, or
        // pni_sci, or pni_rci in this cloud-ice mass conservation pass.
    }

    // Preserve the already admitted focused cold-ice ABI exactly when the
    // full cold-rain table group is disabled. Besides keeping that numerical
    // contract stable, this makes the production extension an explicit gate
    // rather than a silent behavior change to existing callers.
    if (!include_cold_rain && !include_cold_cloud) {
        const double rain_limit = (double)(-rain_mass * inverse_dt);
        if (-rain_ice_rain_rate < rain_limit) {
            rain_ice_rain_rate *= rain_limit / (-rain_ice_rain_rate);
        }

        qi[idx] = fmaxf(0.0f, qi[idx]
            + (float)((nucleation_rate + ice_rate - autoconversion_rate
                       - snow_collection_rate - rain_ice_ice_rate)
                      * (double)orho) * dt);
        ni[idx] = fmaxf(0.0f, ni[idx]
            + (float)((nucleation_number_rate + ice_number_rate
                       - autoconversion_number_rate
                       - snow_collection_number_rate
                       - rain_ice_ice_number_rate)
                      * (double)orho) * dt);
        qs[idx] = fmaxf(0.0f, qs[idx]
            + (float)((ice_to_snow_rate + snow_rate + autoconversion_rate
                       + snow_collection_rate) * (double)orho) * dt);
        qg[idx] = fmaxf(0.0f, qg[idx]
            + (float)((graupel_rate + rain_ice_graupel_rate)
                      * (double)orho) * dt);
        qr[idx] = fmaxf(0.0f, qr[idx]
            - (float)(rain_ice_rain_rate * (double)orho) * dt);
        nr[idx] = fmaxf(0.0f, nr[idx]
            - (float)((rain_self_number_rate + rain_ice_rain_number_rate)
                      * (double)orho) * dt);
        thompson_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
        thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
        qv[idx] = fmaxf(1.0e-10f,
            qv0 - (float)(vapor_sum * (double)orho) * dt);
        const float latent_vapor = 2.5e6f
            + (2106.0f - 4218.0f) * tempc;
        const float latent_fusion = 2.834e6f - latent_vapor;
        temperature[idx] = temp0 + (float)(
            (double)(2.834e6f * inverse_cp) * vapor_sum
            * (double)orho * (double)dt
            + (double)(latent_fusion * inverse_cp) * rain_ice_rain_rate
            * (double)orho * (double)dt);
        return;
    }

    // WRF's rain bound sees every cold-rain mass transfer at once. Number
    // rates and the rain/ice paired graupel source intentionally remain held.
    const double rain_limit = (double)(-rain_mass * inverse_dt);
    const double rain_sum = -freeze_graupel_rate - freeze_ice_rate
        - rain_ice_rain_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
    if (rain_active && rain_sum < rain_limit) {
        const double ratio = rain_limit / rain_sum;
        freeze_graupel_rate *= ratio;
        freeze_ice_rate *= ratio;
        rain_ice_rain_rate *= ratio;
        rain_snow_rain_rate *= ratio;
        rain_graupel_rain_rate *= ratio;
    }

    // The following snow and graupel passes occur after rain conservation.
    // Only the WRF-listed members scale; paired sources outside each group
    // are deliberately not reconstructed here.
    const double snow_limit = (double)(-snow_mass * inverse_dt);
    const double snow_sum = snow_rate - snow_hm_rate
        + rain_snow_category_rate;
    if (qs[idx] > 1.0e-12f && snow_sum < snow_limit) {
        const double ratio = snow_limit / snow_sum;
        const double unbounded_snow_vapor_rate = snow_rate;
        snow_rate *= ratio;
        snow_hm_rate *= ratio;
        rain_snow_category_rate *= ratio;
        vapor_sum += snow_rate - unbounded_snow_vapor_rate;
    }

    const double graupel_limit = (double)(-graupel_mass * inverse_dt);
    const double graupel_sum = graupel_rate - graupel_hm_rate
        + rain_graupel_graupel_rate;
    if (qg[idx] > 1.0e-12f && graupel_sum < graupel_limit) {
        const double ratio = graupel_limit / graupel_sum;
        const double unbounded_graupel_vapor_rate = graupel_rate;
        graupel_rate *= ratio;
        graupel_hm_rate *= ratio;
        rain_graupel_graupel_rate *= ratio;
        vapor_sum += graupel_rate - unbounded_graupel_vapor_rate;
    }

    // Blossey re-enforcement follows every species cap. Rain/snow is not
    // paired in this cold branch; rain/graupel is restored symmetrically.
    const double paired_rate = fmin(
        fabs(rain_graupel_rain_rate),
        fabs(rain_graupel_graupel_rate));
    rain_graupel_rain_rate = -paired_rate;
    rain_graupel_graupel_rate = paired_rate;
    hm_mass_rate = snow_hm_rate + graupel_hm_rate;

    const double rain_rate = cloud_autoconversion_rate
        + cloud_rain_accretion_rate
        - freeze_graupel_rate - freeze_ice_rate - rain_ice_rain_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
    const double rain_number_sink = rain_self_number_rate
        + freeze_graupel_number_rate + freeze_ice_number_rate
        + rain_ice_rain_number_rate + rain_snow_number_rate
        + rain_graupel_number_rate;

    // WRF applies the hydrometeor mass/number bounds while forming tendencies,
    // before it updates temperature, vapor, and density.  The bounds therefore
    // use the immutable incoming density, even though the later sedimentation
    // setup reconstructs category concentrations with post-source density.
    const float latent_vapor = 2.5e6f
        + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const double fusion_rate = cloud_freezing_rate
        + snow_riming_rate + snow_graupel_conversion_rate
        + graupel_riming_rate
        + freeze_ice_rate + freeze_graupel_rate
        + rain_ice_rain_rate + rain_snow_category_rate
        + rain_snow_graupel_rate + rain_graupel_graupel_rate;
    // qv1d + DT*qvten on the vapour as it ENTERED, not on the 1.E-10 floor
    // the rates read (:1800).  WRF floors the running vapour once, at the
    // terminal apply (:3974); every later reader floors it for its working
    // value (:3192).
    const float post_source_qv =
        qv[idx] - (float)(vapor_sum * (double)orho) * dt;
    const float post_source_temperature = temp0 + (float)(
        (double)(2.834e6f * inverse_cp) * vapor_sum
        * (double)orho * (double)dt
        + (double)(latent_fusion * inverse_cp) * fusion_rate
        * (double)orho * (double)dt);
    // Entry rain and graupel as WRF holds them after :1878-1905 and
    // :1915-1949 (zero where the mixing ratio is at or below R1), for the
    // source-stage removals at :3088-3091 and :3157-3159 below.
    const float qr_entry_wrf = qr[idx] > 1.0e-12f ? qr[idx] : 0.0f;
    const float nr_entry_wrf = qr[idx] > 1.0e-12f ? nr[idx] : 0.0f;
    const float qg_entry_wrf = qg[idx] > 1.0e-12f ? qg[idx] : 0.0f;
    const float ng_entry_wrf = qg[idx] > 1.0e-12f
        && track_graupel_number ? graupel_number_shadow[idx] : 0.0f;
    qi[idx] = fmaxf(0.0f, qi[idx]
        + (float)((nucleation_rate + hm_mass_rate
                   + cloud_freezing_rate + freeze_ice_rate + ice_rate
                   - autoconversion_rate
                   - snow_collection_rate - rain_ice_ice_rate)
                  * (double)orho) * dt);
    ni[idx] = fmaxf(0.0f, ni[idx]
        + (float)((nucleation_number_rate + hm_number_rate
                   + cloud_freezing_number_rate + freeze_ice_number_rate
                   + ice_number_rate
                   - autoconversion_number_rate
                   - snow_collection_number_rate
                   - rain_ice_ice_number_rate)
                  * (double)orho) * dt);
    qs[idx] = fmaxf(0.0f, qs[idx]
        + (float)((ice_to_snow_rate + snow_rate + autoconversion_rate
                   + snow_collection_rate + snow_riming_rate
                   + rain_snow_category_rate - snow_hm_rate)
                  * (double)orho) * dt);
    qg[idx] = fmaxf(0.0f, qg[idx]
        + (float)((graupel_rate + freeze_graupel_rate
                   + snow_graupel_conversion_rate
                   + graupel_riming_rate - graupel_hm_rate
                   + rain_ice_graupel_rate + rain_snow_graupel_rate
                   + rain_graupel_graupel_rate)
                  * (double)orho) * dt);
    if (track_graupel_number) {
        // The classic wrapper's private ng1d is not transported, but WRF
        // retains its source tendency for same-call number sedimentation and
        // reflectivity.  Every term below already exists in this simultaneous
        // source diagnosis.  Signs mirror module_mp_thompson.F:3107-3109.
        const float initial_number_per_kg = graupel_number_shadow[idx];
        // png_gde = prg_gde*ng(k)/rg(k) (:2703) wherever L_qg holds, i.e.
        // the ENTRY mixing ratio passed R1, with ng(k) = MAX(R2, ng1d*rho)
        // from the entry block.  Gating on the entry concentration and on
        // the post-source mass dropped it for graupel of 1e-12 kg/kg or so
        // aloft (the mp=28 cold network carries the same gate).
        const float initial_number = qg_entry_wrf > 0.0f
            ? fmaxf(1.0e-6f, initial_number_per_kg * rho) : 0.0f;
        const double graupel_vapor_number_rate = qg_entry_wrf > 0.0f
            ? graupel_rate * (double)initial_number / (double)graupel_mass
            : 0.0;
        // In the sub-freezing branch rain_graupel_number_rate is WRF's
        // pnr_rcg (a rain-number sink), not png_rcg.  png_rcg exists only in
        // the warm inverse-transfer branch and is identically zero here.
        const double number_rate = freeze_graupel_number_rate
            + snow_graupel_conversion_number_rate
            + rain_ice_rain_number_rate
            + rain_snow_number_rate + graupel_vapor_number_rate;
        graupel_number_shadow[idx] = initial_number_per_kg
            + (float)(number_rate * (double)orho) * dt;
    }
    qr[idx] = fmaxf(0.0f, qr[idx]
        + (float)(rain_rate * (double)orho) * dt);
    nr[idx] = fmaxf(0.0f, nr[idx]
        + (float)((cloud_autoconversion_number_rate - rain_number_sink)
                  * (double)orho) * dt);
    qc[idx] = fmaxf(0.0f, qc[idx]
        - (float)((cloud_autoconversion_rate + cloud_freezing_rate
                   + cloud_rain_accretion_rate + snow_riming_rate
                   + snow_graupel_conversion_rate
                   + graupel_riming_rate) * (double)orho) * dt);
    thompson_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
    thompson_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
    // :3067-3091.  Where the post-source rain concentration is at or below R1
    // WRF discards the call's rain sources and removes the entry rain:
    // qrten = -qr1d*odts, nrten = -nr1d*odts.  The number bound above zeroes
    // only the number; the mass has to go too.
    if (!(qr[idx] * rho > 1.0e-12f)) {
        qr[idx] = qr_entry_wrf + (-qr_entry_wrf * inverse_dt) * dt;
        nr[idx] = nr_entry_wrf + (-nr_entry_wrf * inverse_dt) * dt;
    }
    // :3118-3160, the graupel mass/number balance, which WRF runs for
    // non-hail-aware Thompson too.  At or below R1 the graupel and its
    // private number are removed; otherwise the private number is reset so
    // the median volume diameter stays in [D0r, 25.4 mm].  The fallout and
    // calc_refl10cm read that number (the mp=28 source networks carry the
    // same balance).
    {
        const float xrg = qg[idx] * rho;
        if (!(xrg > 1.0e-12f)) {
            qg[idx] = qg_entry_wrf + (-qg_entry_wrf * inverse_dt) * dt;
            if (track_graupel_number) {
                graupel_number_shadow[idx] =
                    ng_entry_wrf + (-ng_entry_wrf * inverse_dt) * dt;
            }
        } else if (track_graupel_number) {
            // REAL(4) throughout except lamg, which is DOUBLE (:1597).
            const float am_g = 3.1415926536f * 400.0f / 6.0f;
            const float mvd_numerator = 3.0f + 0.0f + 0.672f;
            const float xng = fmaxf(1.0e-6f, graupel_number_shadow[idx] * rho);
            double lamg = (double)powf(
                am_g * 6.0f * xng / xrg, 1.0f / 3.0f);
            const float mvd_g = (float)((double)mvd_numerator / lamg);
            bool bounded = false;
            if (mvd_g > 25.4e-3f) {
                lamg = (double)(mvd_numerator / 25.4e-3f);
                bounded = true;
            } else if (mvd_g < 50.0e-6f) {
                lamg = (double)(mvd_numerator / 50.0e-6f);
                bounded = true;
            }
            if (bounded) {
                const float xng_bounded = (float)(
                    (double)((1.0f / 6.0f) * xrg) * pow(lamg, 3.0)
                    / (double)am_g);
                const float ngten = (xng_bounded - ng_entry_wrf * rho)
                    * inverse_dt * (1.0f / rho);
                graupel_number_shadow[idx] = ng_entry_wrf + ngten * dt;
            }
        }
    }
    qv[idx] = post_source_qv;
    temperature[idx] = post_source_temperature;
}

extern "C" __global__ void thompson_ice_nucleation(
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float temp0 = temperature[idx];
    if (temp0 >= 273.15f) return;
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float qvsi = thompson_rsif(pressure[idx], temp0);
    const float qvs = thompson_rslf(pressure[idx], temp0);
    float ssati = qv0 / qvsi - 1.0f;
    float ssatw = qv0 / qvs - 1.0f;
    if (fabsf(ssati) < 1.0e-15f) ssati = 0.0f;
    if (fabsf(ssatw) < 1.0e-15f) ssatw = 0.0f;
    if (!(ssati >= 0.25f
            || (ssatw > 1.0e-15f && temp0 < 253.15f))) return;

    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float inverse_dt = 1.0f / dt;
    float ice_number = 1.0e-6f;
    if (qi[idx] > 1.0e-12f) {
        const float ice_mass = qi[idx] * rho;
        ice_number = fmaxf(1.0e-6f, ni[idx] * rho);
        const float am_i = 3.1415926536f * 890.0f / 6.0f;
        double lambda = (double)powf(
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
    }

    const float target_number = fminf(
        250.0e3f, 5.0f * expf(0.304f * (273.15f - temp0)));
    float number_rate = fmaxf(0.0f, target_number - ice_number)
        * inverse_dt;
    const float vapor_limit = (qv0 - qvsi) * rho * inverse_dt * 0.999f;
    float mass_rate = fminf(vapor_limit, 1.0e-12f * number_rate);
    number_rate = __fdiv_rn(mass_rate, 1.0e-12f);

    const float mass_tendency = mass_rate * orho;
    const float number_tendency = number_rate * orho;
    float qi_new = qi[idx] + mass_tendency * dt;
    float ni_new = ni[idx] + number_tendency * dt;
    const float am_i = 3.1415926536f * 890.0f / 6.0f;
    const float final_ice_mass = fmaxf(1.0e-12f, qi_new * rho);
    float final_ice_number = fmaxf(1.0e-6f, ni_new * rho);
    if (final_ice_mass > 1.0e-12f) {
        double lambda = (double)powf(
            am_i * 6.0f * final_ice_number / final_ice_mass,
            1.0f / 3.0f);
        const float diameter = (float)(4.0 / lambda);
        if (diameter < 5.0e-6f) {
            lambda = 4.0 / 5.0e-6;
            final_ice_number = fminf(999.0e3f,
                __fdiv_rn((1.0f / 6.0f) * final_ice_mass, am_i)
                * (float)(lambda * lambda * lambda));
        } else if (diameter > 300.0e-6f) {
            lambda = 4.0 / 300.0e-6;
            final_ice_number = __fdiv_rn((1.0f / 6.0f) * final_ice_mass, am_i)
                * (float)(lambda * lambda * lambda);
        }
        final_ice_number = fminf(final_ice_number, 999.0e3f);
        ni_new = final_ice_number * orho;
    } else {
        qi_new = 0.0f;
        ni_new = 0.0f;
    }

    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    qi[idx] = fmaxf(0.0f, qi_new);
    ni[idx] = fmaxf(0.0f, ni_new);
    qv[idx] = fmaxf(1.0e-10f, qv0 - mass_tendency * dt);
    temperature[idx] = temp0
        + 2.834e6f * inverse_cp * mass_tendency * dt;
}

extern "C" __global__ void thompson_ice_autoconversion(
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ qs,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const double* __restrict__ ice_to_snow_mass,
    const double* __restrict__ ice_to_snow_number,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || qi[idx] <= 1.0e-12f) return;

    const float qvk = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temperature[idx] * (qvk + 0.622f));
    const float ice_mass = qi[idx] * rho;
    const float ice_number = fmaxf(1.0e-6f, ni[idx] * rho);
    const int mass_bin = ice_mass > 1.0e-10f
        ? thompson_decade_table_index(ice_mass, -10, 64) : 0;
    const int number_bin = ice_number > 1.0f
        ? thompson_decade_table_index(ice_number, 0, 55) : 0;

    const float am_i = 3.1415926536f * 890.0f / 6.0f;
    const float lambda = powf(
        am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f);
    const double diameter = fmax(1.0e-6, 4.0 / (double)lambda);

    double mass_rate = 0.0;
    double number_rate = 0.0;
    const double inv_dt = 1.0 / (double)dt;
    if (mass_bin == 63 || diameter > 1500.0e-6) {
        mass_rate = (double)(ice_mass * 0.99f) * inv_dt;
        number_rate = (double)(ice_number * 0.95f) * inv_dt;
    } else if (diameter >= 30.0e-6) {
        const int table_idx = mass_bin + 64 * number_bin;
        mass_rate = fmin((double)(ice_mass * 0.99f) * inv_dt,
                         ice_to_snow_mass[table_idx] * inv_dt);
        number_rate = fmin((double)(ice_number * 0.95f) * inv_dt,
                           ice_to_snow_number[table_idx] * inv_dt);
    }

    const float mass_tendency = (float)(mass_rate / (double)rho);
    const float number_tendency = (float)(number_rate / (double)rho);
    qi[idx] -= mass_tendency * dt;
    qs[idx] += mass_tendency * dt;
    ni[idx] -= number_tendency * dt;
}

// ---------------------------------------------------------------------------
// The classic adapter's own arithmetic in four launches
// (gpuwm/core/microphysics.py:_apply_thompson through thompson.py's
// launch_adapter_*), where it was about forty CuPy operations and four small
// kernels.  Each cell does exactly what those did, in their order; a value
// that crossed a CuPy launch boundary is rounded where it crossed
// (__fadd_rn / __fmul_rn), so no product can contract into a later sum, and
// min / max keep CuPy's canonical NaN.  tests/test_thompson_speed_glue.py
// holds every output, special values included, to the former sequence bit
// for bit.

#define THOMPSON_CANONICAL_NAN __int_as_float(0x7fffffff)

// Before any process: theta (thb + thp) and temperature (theta times the
// Exner function the adapter computed with CuPy's power); the layer depths
// from the full-level geopotential, divided by gravity passed as a value,
// because NVRTC for sm_120 divides by a literal constant without IEEE
// rounding (0.08 percent of quotients off by one ULP on an RTX 5090) while a
// runtime divisor, like CuPy's, rounds correctly; the entry graupel mask and
// the entry-temperature
// warm marker; GRAUPELNCV reset; the pre-microphysics full theta parked in
// h_diabatic (save_pre_mp_theta);
// WRF's entry rewrite (:1844-1845, :1871-1872, :1900-1901, :1911,
// :1941-1942), which zeroes the mass (and number) of every species at or
// below R1 in every column; and the no-microphysics column flag cleared for
// thompson_adapter_entry to set.
extern "C" __global__ void thompson_adapter_prepare(
    const float* __restrict__ thb, const float* __restrict__ thp,
    const float* __restrict__ phb, const float* __restrict__ php,
    float* __restrict__ th, const float* __restrict__ pii,
    float* __restrict__ temperature, float* __restrict__ dz,
    float* __restrict__ saved_theta,
    float* __restrict__ qc, float* __restrict__ qi, float* __restrict__ ni,
    float* __restrict__ qr, float* __restrict__ nr, float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ entry_graupel, float* __restrict__ warm_entry,
    float* __restrict__ graupelncv, float* __restrict__ micro_columns,
    float gravity, int size, int columns, int thb_full, int phb_full
#ifdef GPUWM_USE_THETA_M
    , const float* __restrict__ qv
#endif
    )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const int k = idx / columns;
    const float thermal = __fadd_rn(thb[thb_full ? idx : k], thp[idx]);
#ifdef GPUWM_USE_THETA_M
    const float theta = __fdiv_rn(thermal,
        __fadd_rn(1.0f, __fmul_rn(RVOVRD, qv[idx])));
#else
    const float theta = thermal;
#endif
    const float temp = __fmul_rn(theta, pii[idx]);
    th[idx] = theta;
    temperature[idx] = temp;
    saved_theta[idx] = theta;
    const float lower = __fadd_rn(phb[phb_full ? idx : k], php[idx])
        / gravity;
    const float upper = __fadd_rn(phb[phb_full ? idx + columns : k + 1],
                                  php[idx + columns]) / gravity;
    dz[idx] = __fsub_rn(upper, lower);
    entry_graupel[idx] = qg[idx] > 1.0e-12f ? 1.0f : 0.0f;
    warm_entry[idx] = temp >= 273.15f ? 1.0f : 0.0f;
    if (idx < columns) {
        graupelncv[idx] = 0.0f;
        micro_columns[idx] = 0.0f;
    }
    if (!(qc[idx] > 1.0e-12f)) qc[idx] = 0.0f;
    if (!(qi[idx] > 1.0e-12f)) { qi[idx] = 0.0f; ni[idx] = 0.0f; }
    if (!(qr[idx] > 1.0e-12f)) { qr[idx] = 0.0f; nr[idx] = 0.0f; }
    if (!(qs[idx] > 1.0e-12f)) qs[idx] = 0.0f;
    if (!(qg[idx] > 1.0e-12f)) qg[idx] = 0.0f;
}

// On the rewritten entry state: the private graupel number
// (thompson_classic_graupel_number_init) and WRF's no-microphysics column
// flag (thompson_microphysics_columns), one thread per cell.  A level that
// clears the flag sets its column's; every writer writes the same 1.0, and
// the flag does not depend on which level is read first.
extern "C" __global__ void thompson_adapter_entry(
    const float* __restrict__ qc, const float* __restrict__ qi,
    const float* __restrict__ qr, const float* __restrict__ qs,
    const float* __restrict__ qg, const float* __restrict__ temperature,
    const float* __restrict__ pressure, const float* __restrict__ qv,
    float* __restrict__ graupel_number_per_kg,
    float* __restrict__ micro_columns, int size, int columns)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    graupel_number_per_kg[idx] = thompson_classic_graupel_number(
        qg, temperature, pressure, qv, idx);
    if (thompson_level_has_microphysics(qc, qi, qr, qs, qg, temperature,
                                        pressure, qv, idx)) {
        micro_columns[idx % columns] = 1.0f;
    }
}

// After the sources: the post-source rain column mask
// (thompson_hydrometeor_column_mask on qr) and the graupel fallout column
// mask (thompson_graupel_fallout_column_mask), one thread per column.
extern "C" __global__ void thompson_adapter_masks(
    const float* __restrict__ qr, const float* __restrict__ qg,
    const float* __restrict__ entry_graupel,
    float* __restrict__ rain_columns, float* __restrict__ graupel_columns,
    int nz, int columns)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;
    float rain = 0.0f;
    float graupel = 0.0f;
    for (int k = 0; k < nz && (rain == 0.0f || graupel == 0.0f); ++k) {
        const size_t idx = (size_t)k * columns + column;
        if (qr[idx] > 1.0e-12f) rain = 1.0f;
        if (entry_graupel[idx] != 0.0f && qg[idx] > 1.0e-12f) graupel = 1.0f;
    }
    rain_columns[column] = rain;
    graupel_columns[column] = graupel;
}

// Last: theta from temperature (T / Exner), moist_physics_finish (WRF
// moist_physics_finish_em, use_theta_m = 0: the increment over the saved
// theta clamped to +/- mp_tend_lim * dt, added to thp once and kept as
// h_diabatic = increment / dt; with no_mp_heating h_diabatic = 0), and the
// surface frozen fraction SR.
extern "C" __global__ void thompson_adapter_finish(
    const float* __restrict__ temperature, const float* __restrict__ pii,
    float* __restrict__ th, float* __restrict__ thp,
    float* __restrict__ h_diabatic,
    const float* __restrict__ rainncv, const float* __restrict__ snowncv,
    const float* __restrict__ graupelncv, float* __restrict__ sr,
    float limit, float dt, int no_mp_heating, int size, int columns)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const float theta = temperature[idx] / pii[idx];
    th[idx] = theta;
#ifndef GPUWM_USE_THETA_M
    if (!no_mp_heating) {
        float increment = __fsub_rn(theta, h_diabatic[idx]);
        increment = isnan(increment) || isnan(limit)
            ? THOMPSON_CANONICAL_NAN : min(limit, increment);
        increment = isnan(increment) || isnan(limit)
            ? THOMPSON_CANONICAL_NAN : max(-limit, increment);
        thp[idx] = __fadd_rn(thp[idx], increment);
        h_diabatic[idx] = increment / dt;
    } else {
        h_diabatic[idx] = 0.0f;
    }
#endif
    if (idx < columns) {
        const float frozen = __fadd_rn(snowncv[idx], graupelncv[idx]);
        const float ratio = frozen / rainncv[idx];
        const float bounded = isnan(ratio)
            ? THOMPSON_CANONICAL_NAN : min(1.0f, ratio);
        sr[idx] = rainncv[idx] > 1.0e-12f ? bounded : 0.0f;
    }
}

// Level-parallel sedimentation for columns of at most 64 levels.  Each
// block carries eight columns (thirty-two for graupel, compacted to the
// active ones) with one level per thread.  Every level keeps the column
// kernel's own expressions and order: fall speeds are computed per level,
// a level at or below R1 inherits the nearest defined speed above it, the
// substep count and sediment top are exact maxima, each substep publishes
// every flux before any level updates (the column kernel's Jacobi form),
// and the surface export is summed in substep order by the bottom level.
// Deeper columns and the other entry points keep the column kernels.

// Device only: a block barrier has no meaning in the serial host
// build (tools/thompson_real_column_parity), which runs the column
// kernels instead (gpuwm/core/thompson.py LEVEL_PARALLEL_FALLOUT).
#ifdef __CUDACC_RTC__
extern "C" __global__ void thompson_rain_sediment_levels_64_with_presence(
    float* __restrict__ qr,
    float* __restrict__ nr,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    int accumulate_surface, float dt, int nz, int ny, int nx)
{
    const int c = threadIdx.x;
    const int column = blockIdx.x * 8 + c;
    const int j = column / nx;
    const int i = column - j * nx;
    const int k = threadIdx.y;
    __shared__ float mass_flux[64][8], number_flux[64][8];
    __shared__ float defined_mass[64][8], defined_number[64][8];
    __shared__ int defined[64][8], steps[64][8], tops[64][8];
    __shared__ int column_nstep[8], column_top[8];
    float density;
    float rain_mass;
    float rain_number;
    float mass_velocity;
    float number_velocity;
    float qr_tendency;
    float nr_tendency;
    float qr_initial;
    float nr_initial;
    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float org3 = 1.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above_mass = 0.0f;
    float velocity_above_number = 0.0f;

    defined[k][c] = 0;
    if (column < ny * nx && k < nz) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        const float rain_density_carried = reference_density == nullptr
            ? rho : reference_density[idx];
        const float rain_density = fabsf(rain_density_carried);
        density = rho;
        qr_initial = qr[idx];
        nr_initial = nr[idx];
        qr_tendency = 0.0f;
        nr_tendency = 0.0f;
        const bool l_qr = true
            ? rain_density_carried != 0.0f : qr[idx] > 1.0e-12f;
        const bool rewritten = true
            && rain_density_carried < 0.0f;
        const float rr = !l_qr ? 1.0e-12f
            : rewritten ? fmaxf(1.0e-12f, qr[idx] * rain_density)
            : qr[idx] * rain_density;
        defined[k][c] = rr > 1.0e-12f;
        if (rr > 1.0e-12f) {
            const float nn = fmaxf(1.0e-6f, nr[idx] * rain_density);
            const float lambda_arg = am_r * 6.0f * nn / rr;
            const double lambda = (double)powf(lambda_arg, 1.0f / 3.0f);
            rain_mass = rr;
            rain_number = nn;

            const float rhof = sqrtf(rho_not / rho);
            const float mass_prefix = rhof * 4854.0f * 24.0f * org3;
            mass_velocity = (float)((double)mass_prefix
                * pow(lambda, 4.0) * pow(lambda + 195.0, -5.0));
            const float number_prefix = __fdiv_rn(rhof * 4854.0f
                * 3.3233511f, 1.3293403f);
            number_velocity = (float)((double)number_prefix
                * pow(lambda, 2.5) * pow(lambda + 195.0, -3.5));
        } else {
            rain_mass = rr;
            rain_number = l_qr
                ? fmaxf(1.0e-6f, nr[idx] * rain_density) : 1.0e-6f;
            mass_velocity = velocity_above_mass;
            number_velocity = velocity_above_number;
        }
    }
    defined_mass[k][c] = column < ny * nx && k < nz ? mass_velocity : 0.0f;
    defined_number[k][c] = column < ny * nx && k < nz ? number_velocity : 0.0f;
    __syncthreads();
    if (column < ny * nx && k < nz) {
        if (!defined[k][c]) {
            for (int above = k + 1; above < nz; ++above) {
                if (defined[above][c]) {
                    mass_velocity = defined_mass[above][c];
                    number_velocity = defined_number[above][c];
                    break;
                }
            }
        }
        const size_t idx = IDX3(k, j, i);
        const float vmax = fmaxf(mass_velocity, number_velocity);
        if (vmax > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = dz[idx] / vmax;
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    steps[k][c] = nstep;
    tops[k][c] = sediment_top;
    __syncthreads();
    if (k == 0) {
        int n = 0, t = 0;
        for (int lane = 0; lane < nz; ++lane) {
            n = max(n, steps[lane][c]);
            t = max(t, tops[lane][c]);
        }
        column_nstep[c] = max(n, 1);
        column_top[c] = t == nz - 1 ? nz - 2 : t;
    }
    __syncthreads();
    nstep = column_nstep[c];
    sediment_top = column_top[c];
    const float onstep = 1.0f / (float)nstep;
    const float dt_substep = dt * onstep;
    float exported = 0.0f;
    int block_nstep = 1;
    for (int col = 0; col < 8; ++col) block_nstep = max(block_nstep, column_nstep[col]);
    for (int step = 0; step < block_nstep; ++step) {
        if (column < ny * nx && k < nz && step < nstep) {
            mass_flux[k][c] = mass_velocity * rain_mass;
            number_flux[k][c] = number_velocity * rain_number;
        }
        __syncthreads();
        if (column < ny * nx && k == nz - 1 && step < nstep) {
        const size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density;
        qr_tendency -= mass_flux[k][c] * inv_dz * onstep * inv_rho;
        nr_tendency -= number_flux[k][c] * inv_dz * onstep * inv_rho;
        rain_mass = fmaxf(1.0e-12f,
            rain_mass - mass_flux[k][c] * inv_dz * dt_substep);
        rain_number = fmaxf(1.0e-6f,
            rain_number - number_flux[k][c] * inv_dz * dt_substep);

        }
        if (column < ny * nx && k <= sediment_top && step < nstep) {
            const size_t idx = IDX3(k, j, i);
            const float inv_dz = 1.0f / dz[idx];
            const float inv_rho = 1.0f / density;
            const float mass_divergence = mass_flux[k + 1][c] - mass_flux[k][c];
            const float number_divergence =
                number_flux[k + 1][c] - number_flux[k][c];
            qr_tendency += mass_divergence * inv_dz * onstep * inv_rho;
            nr_tendency += number_divergence * inv_dz * onstep * inv_rho;
            rain_mass = fmaxf(1.0e-12f,
                rain_mass + mass_divergence * inv_dz * dt_substep);
            rain_number = fmaxf(1.0e-6f,
                rain_number + number_divergence * inv_dz * dt_substep);
        }
        if (column < ny * nx && k == 0 && step < nstep && rain_mass > 1.0e-9f) {
            exported += mass_flux[0][c] * dt_substep;
        }
        __syncthreads();
    }
    if (column < ny * nx && k == 0) {
    if (accumulate_surface) {
        rainncv[column] += exported;
    } else {
        rainncv[column] = exported;
    }
    rainnc[column] += exported;
    }
    if (column < ny * nx && k < nz) {
        const size_t idx = IDX3(k, j, i);
        float qr_new = qr_initial + qr_tendency * dt;
        float nr_new = fmaxf(1.0e-6f / density,
                             nr_initial + nr_tendency * dt);
        if (qr_new <= 1.0e-12f) {
            qr[idx] = 0.0f;
            nr[idx] = 0.0f;
            return;
        }
        const float lambda_arg = am_r * 6.0f * nr_new / qr_new;
        double lambda = (double)powf(lambda_arg, 1.0f / 3.0f);
        float mvd = (float)(3.672 / lambda);
        if (mvd > 2.5e-3f) mvd = 2.5e-3f;
        else if (mvd < 37.5e-6f) mvd = 37.5e-6f;
        lambda = 3.672 / (double)mvd;
        const float prefix = __fdiv_rn(org3 * qr_new, am_r);
        nr_new = (float)((double)prefix * pow(lambda, 3.0));
        qr[idx] = qr_new;
        nr[idx] = nr_new;
    }
}
extern "C" __global__ void thompson_ice_sediment_levels_64_with_density(
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
    const int c = threadIdx.x;
    const int column = blockIdx.x * 8 + c;
    const int j = column / nx;
    const int i = column - j * nx;
    const int k = threadIdx.y;
    __shared__ float mass_flux[64][8], number_flux[64][8];
    __shared__ float defined_mass[64][8], defined_number[64][8];
    __shared__ int defined[64][8], steps[64][8], tops[64][8];
    __shared__ int column_nstep[8], column_top[8];
    float density;
    float ice_mass;
    float ice_number;
    float mass_velocity;
    float number_velocity;
    float qi_tendency;
    float ni_tendency;
    float qi_initial;
    float ni_initial;
    const float pi = 3.1415926536f;
    const float am_i = pi * 890.0f / 6.0f;
    const float oig2 = 1.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above_mass = 0.0f;
    float velocity_above_number = 0.0f;

    defined[k][c] = 0;
    if (column < ny * nx && k < nz) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density = rho;
        qi_initial = qi[idx];
        ni_initial = ni[idx];
        qi_tendency = 0.0f;
        ni_tendency = 0.0f;
        const float ice_state_rho = reference_density == nullptr
            ? rho : reference_density[idx];
        defined[k][c] = qi[idx] > 1.0e-12f && qi[idx] * ice_state_rho > 1.0e-12f;
        if (defined[k][c]) {
            const float state_rho = ice_state_rho;
            const float ri = qi[idx] * state_rho;
            float nn = fmaxf(1.0e-6f, ni[idx] * state_rho);
            if (nn <= 1.0e-6f) {
                const double lambda = 4.0 / 5.0e-6;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = fminf(999.0e3f,
                           (float)((double)prefix * pow(lambda, 3.0)));
            }
            const float lambda_arg = am_i * 6.0f * nn / ri;
            double lambda = (double)powf(lambda_arg, 1.0f / 3.0f);
            float diameter = (float)(4.0 / lambda);
            if (diameter < 5.0e-6f) {
                diameter = 5.0e-6f;
                lambda = 4.0 / (double)diameter;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = fminf(999.0e3f,
                           (float)((double)prefix * pow(lambda, 3.0)));
            } else if (diameter > 300.0e-6f) {
                diameter = 300.0e-6f;
                lambda = 4.0 / (double)diameter;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = (float)((double)prefix * pow(lambda, 3.0));
            }
            ice_mass = ri;
            ice_number = nn;

            const float rhof = sqrtf(rho_not / rho);
            const double inverse_lambda = 1.0 / lambda;
            const float mass_prefix = rhof * 1493.9f * 24.0f * oig2;
            mass_velocity = (float)((double)mass_prefix * inverse_lambda);
            const float number_prefix = __fdiv_rn(rhof * 1493.9f
                * 3.3233511f, 1.3293403f);
            number_velocity = (float)((double)number_prefix
                                          * inverse_lambda);
        } else {
            const bool l_qi = qi[idx] > 1.0e-12f;
            ice_mass = l_qi ? qi[idx] * ice_state_rho : 1.0e-12f;
            ice_number = l_qi
                ? fmaxf(1.0e-6f, ni[idx] * ice_state_rho) : 1.0e-6f;
            mass_velocity = velocity_above_mass;
            number_velocity = velocity_above_number;
        }
    }
    defined_mass[k][c] = column < ny * nx && k < nz ? mass_velocity : 0.0f;
    defined_number[k][c] = column < ny * nx && k < nz ? number_velocity : 0.0f;
    __syncthreads();
    if (column < ny * nx && k < nz) {
        if (!defined[k][c]) {
            for (int above = k + 1; above < nz; ++above) {
                if (defined[above][c]) {
                    mass_velocity = defined_mass[above][c];
                    number_velocity = defined_number[above][c];
                    break;
                }
            }
        }
        const size_t idx = IDX3(k, j, i);
        if (mass_velocity > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = dz[idx] / mass_velocity;
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    steps[k][c] = nstep;
    tops[k][c] = sediment_top;
    __syncthreads();
    if (k == 0) {
        int n = 0, t = 0;
        for (int lane = 0; lane < nz; ++lane) {
            n = max(n, steps[lane][c]);
            t = max(t, tops[lane][c]);
        }
        column_nstep[c] = max(n, 1);
        column_top[c] = t == nz - 1 ? nz - 2 : t;
    }
    __syncthreads();
    nstep = column_nstep[c];
    sediment_top = column_top[c];
    const float onstep = 1.0f / (float)nstep;
    const float dt_substep = dt * onstep;
    float exported = 0.0f;
    int block_nstep = 1;
    for (int col = 0; col < 8; ++col) block_nstep = max(block_nstep, column_nstep[col]);
    for (int step = 0; step < block_nstep; ++step) {
        if (column < ny * nx && k < nz && step < nstep) {
            mass_flux[k][c] = mass_velocity * ice_mass;
            number_flux[k][c] = number_velocity * ice_number;
        }
        __syncthreads();
        if (column < ny * nx && k == nz - 1 && step < nstep) {
        const size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density;
        qi_tendency -= mass_flux[k][c] * inv_dz * onstep * inv_rho;
        ni_tendency -= number_flux[k][c] * inv_dz * onstep * inv_rho;
        ice_mass = fmaxf(1.0e-12f,
            ice_mass - mass_flux[k][c] * inv_dz * dt_substep);
        ice_number = fmaxf(1.0e-6f,
            ice_number - number_flux[k][c] * inv_dz * dt_substep);

        }
        if (column < ny * nx && k <= sediment_top && step < nstep) {
            const size_t idx = IDX3(k, j, i);
            const float inv_dz = 1.0f / dz[idx];
            const float inv_rho = 1.0f / density;
            const float mass_divergence = mass_flux[k + 1][c] - mass_flux[k][c];
            const float number_divergence =
                number_flux[k + 1][c] - number_flux[k][c];
            qi_tendency += mass_divergence * inv_dz * onstep * inv_rho;
            ni_tendency += number_divergence * inv_dz * onstep * inv_rho;
            ice_mass = fmaxf(1.0e-12f,
                ice_mass + mass_divergence * inv_dz * dt_substep);
            ice_number = fmaxf(1.0e-6f,
                ice_number + number_divergence * inv_dz * dt_substep);
        }
        if (column < ny * nx && k == 0 && step < nstep && ice_mass > 1.0e-9f) {
            exported += mass_flux[0][c] * dt_substep;
        }
        __syncthreads();
    }
    if (column < ny * nx && k == 0) {
    rainncv[column] = exported;
    snowncv[column] = exported;
    rainnc[column] += exported;
    snownc[column] += exported;
    }
    if (column < ny * nx && k < nz) {
        const size_t idx = IDX3(k, j, i);
        const float qi_new = qi_initial + qi_tendency * dt;
        float ni_new = fmaxf(1.0e-6f / density,
                             ni_initial + ni_tendency * dt);
        if (qi_new <= 1.0e-12f) {
            qi[idx] = qi_new;
            ni[idx] = ni_initial + ni_tendency * dt;
            return;
        }
        const float lambda_arg = am_i * 6.0f * ni_new / qi_new;
        double lambda = (double)powf(lambda_arg, 1.0f / 3.0f);
        float diameter = (float)(4.0 / lambda);
        if (diameter < 5.0e-6f) diameter = 5.0e-6f;
        else if (diameter > 300.0e-6f) diameter = 300.0e-6f;
        lambda = 4.0 / (double)diameter;
        const float prefix = __fdiv_rn(oig2 * qi_new, am_i);
        ni_new = fminf((float)((double)prefix * pow(lambda, 3.0)),
                       999.0e3f / density);
        qi[idx] = qi_new;
        ni[idx] = ni_new;
    }
}

// Rain inheritance precedes the snow blend. Substep sums retain their order.
extern "C" __global__ void thompson_snow_sediment_levels_64_with_rain_profile_and_presence(
    float* __restrict__ qs,
    const float* __restrict__ snow_melt_marker,
    const float* __restrict__ melt_rain_qr,
    const float* __restrict__ melt_rain_nr,
    const float* __restrict__ melt_rain_density,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ reference_temperature,
    const float* __restrict__ velocity_boost,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ snownc,
    float* __restrict__ snowncv,
    int accumulate_surface, float dt, int nz, int ny, int nx)
{
    const int c = threadIdx.x;
    const int column = blockIdx.x * 8 + c;
    const int j = column / nx;
    const int i = column - j * nx;
    const int k = threadIdx.y;
    __shared__ float mass_flux[64][8], defined_mass[64][8], rain_speed[64][8];
    __shared__ int defined[64][8], rain_defined[64][8], steps[64][8], tops[64][8];
    __shared__ int column_nstep[8], column_top[8];
    float density;
    float snow_mass;
    float mass_velocity;
    float qs_tendency;
    float qs_initial;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above = 0.0f;
    float rain_velocity_above = 0.0f;

    float rain_rr = 1.0e-12f, rain_velocity = 0.0f;
    defined[k][c] = 0;
    rain_defined[k][c] = 0;
    if (column < ny * nx && k < nz) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density = rho;
        qs_initial = qs[idx];
        qs_tendency = 0.0f;
        rain_rr = 1.0e-12f;
        rain_velocity = 0.0f;
        if (melt_rain_qr != (const float*)0) {
            const float rain_density_carried = melt_rain_density == nullptr
                ? (reference_density == nullptr ? rho : reference_density[idx])
                : melt_rain_density[idx];
            const float rain_density = fabsf(rain_density_carried);
            const bool l_qr = true
                ? rain_density_carried != 0.0f : melt_rain_qr[idx] > 1.0e-12f;
            const bool rewritten = true
                && rain_density_carried < 0.0f;
            rain_rr = !l_qr ? 1.0e-12f
                : rewritten ? fmaxf(1.0e-12f, melt_rain_qr[idx] * rain_density)
                : melt_rain_qr[idx] * rain_density;
            if (rain_rr > 1.0e-12f) {
                const float am_r = 3.1415926536f * 1000.0f / 6.0f;
                const float rain_number = fmaxf(
                    1.0e-6f, melt_rain_nr[idx] * rain_density);
                const double rain_lambda = (double)powf(
                    am_r * 6.0f * rain_number / rain_rr, 1.0f / 3.0f);
                const float rain_rhof = sqrtf(rho_not / rho);
                rain_velocity = (float)(
                    (double)(rain_rhof * 4854.0f * 24.0f * (1.0f / 6.0f))
                    * pow(rain_lambda, 4.0)
                    * pow(rain_lambda + 195.0, -5.0));
            } else {
                rain_velocity = rain_velocity_above;
            }

        }
        rain_defined[k][c] = melt_rain_qr != nullptr && rain_rr > 1.0e-12f;
    }
    rain_speed[k][c] = rain_velocity;
    __syncthreads();
    if (column < ny * nx && k < nz) {
        if (!rain_defined[k][c]) {
            for (int above = k + 1; above < nz; ++above) {
                if (rain_defined[above][c]) {
                    rain_velocity = rain_speed[above][c];
                    break;
                }
            }
        }
        const float rho = density;
        const size_t idx = IDX3(k, j, i);
        const float snow_state_rho = reference_density == nullptr
            ? rho : reference_density[idx];
        defined[k][c] = qs[idx] > 1.0e-12f && qs[idx] * snow_state_rho > 1.0e-12f;
        if (defined[k][c]) {
            const float state_rho = snow_state_rho;
            const float rs = qs[idx] * state_rho;
            const float smob = rs * (1.0f / 0.069f);
            const float state_temperature = reference_temperature == nullptr
                ? temperature[idx] : reference_temperature[idx];
            const float tc0 = fminf(-0.1f, state_temperature - 273.15f);
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
            const float smoc = powf(10.0f, loga) * powf(smob, exponent);
            const float mean_ratio = smob / smoc;
            float ils1 = 1.0f / (mean_ratio * 20.78f + 100.0f);
            float ils2 = 1.0f / (mean_ratio * 3.29f + 100.0f);
            const float ratio_power = powf(mean_ratio, 0.6357f);
            const float numerator1 = 490.6f * 3.51325202f
                * powf(ils1, 3.55f);
            const float numerator2 = 17.46f * ratio_power * 7.61279917f
                * powf(ils2, 4.1857f);
            ils1 = 1.0f / (mean_ratio * 20.78f);
            ils2 = 1.0f / (mean_ratio * 3.29f);
            const float denominator1 = 490.6f * 2.0f
                * powf(ils1, 3.0f);
            const float denominator2 = 17.46f * ratio_power * 3.87160635f
                * powf(ils2, 3.6357f);
            const float rhof = sqrtf(rho_not / rho);
            float snow_velocity = rhof * 40.0f
                * (numerator1 + numerator2)
                / (denominator1 + denominator2);
            if (velocity_boost != (const float*)0) {
                snow_velocity *= velocity_boost[idx];
            }
            if (snow_melt_marker != (const float*)0
                    && snow_melt_marker[idx] != 0.0f
                    && melt_rain_qr != (const float*)0) {
                const float solid_fraction = rs / (rs + rain_rr);
                snow_velocity = snow_velocity * solid_fraction
                    + rain_velocity * (1.0f - solid_fraction);
            }
            mass_velocity = snow_velocity;
            snow_mass = rs;
        } else {
            snow_mass = qs[idx] > 1.0e-12f
                ? qs[idx] * snow_state_rho : 1.0e-12f;
            mass_velocity = velocity_above;
        }
    }
    defined_mass[k][c] = column < ny * nx && k < nz ? mass_velocity : 0.0f;
    __syncthreads();
    if (column < ny * nx && k < nz) {
        if (!defined[k][c]) {
            for (int above = k + 1; above < nz; ++above) {
                if (defined[above][c]) {
                    mass_velocity = defined_mass[above][c];
                    break;
                }
            }
        }
        const size_t idx = IDX3(k, j, i);
        if (mass_velocity > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = dz[idx] / mass_velocity;
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    steps[k][c] = nstep;
    tops[k][c] = sediment_top;
    __syncthreads();
    if (k == 0) {
        int n = 0, t = 0;
        for (int lane = 0; lane < nz; ++lane) {
            n = max(n, steps[lane][c]);
            t = max(t, tops[lane][c]);
        }
        column_nstep[c] = max(n, 1);
        column_top[c] = t == nz - 1 ? nz - 2 : t;
    }
    __syncthreads();
    nstep = column_nstep[c];
    sediment_top = column_top[c];
    const float onstep = 1.0f / (float)nstep;
    const float dt_substep = dt * onstep;
    float exported = 0.0f;
    int block_nstep = 1;
    for (int col = 0; col < 8; ++col) block_nstep = max(block_nstep, column_nstep[col]);
    for (int step = 0; step < block_nstep; ++step) {
        if (column < ny * nx && k < nz && step < nstep) mass_flux[k][c] = mass_velocity * snow_mass;
        __syncthreads();
        if (column < ny * nx && k == nz - 1 && step < nstep) {
        const size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density;
        qs_tendency -= mass_flux[k][c] * inv_dz * onstep * inv_rho;
        snow_mass = fmaxf(1.0e-12f,
            snow_mass - mass_flux[k][c] * inv_dz * dt_substep);

        }
        if (column < ny * nx && k <= sediment_top && step < nstep) {
            const size_t idx = IDX3(k, j, i);
            const float inv_dz = 1.0f / dz[idx];
            const float inv_rho = 1.0f / density;
            const float divergence = mass_flux[k + 1][c] - mass_flux[k][c];
            qs_tendency += divergence * inv_dz * onstep * inv_rho;
            snow_mass = fmaxf(1.0e-12f,
                snow_mass + divergence * inv_dz * dt_substep);
        }
        if (column < ny * nx && k == 0 && step < nstep && snow_mass > 1.0e-9f) {
            exported += mass_flux[0][c] * dt_substep;
        }
        __syncthreads();
    }
    if (column < ny * nx && k == 0) {
    if (accumulate_surface) {
        rainncv[column] += exported;
        snowncv[column] += exported;
    } else {
        rainncv[column] = exported;
        snowncv[column] = exported;
    }
    rainnc[column] += exported;
    snownc[column] += exported;
    }
    if (column < ny * nx && k < nz) {
        const size_t idx = IDX3(k, j, i);
        const float qs_new = qs_initial + qs_tendency * dt;
        qs[idx] = qs_new <= 1.0e-12f ? 0.0f : qs_new;
    }
}
extern "C" __global__ void thompson_graupel_sediment_levels_64_with_density_and_column_mask_and_shadow(
    float* __restrict__ qg,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ graupelnc,
    float* __restrict__ graupelncv,
    const float* __restrict__ active_columns,
    float* __restrict__ graupel_number_shadow,
    int accumulate_surface, float dt, int nz, int ny, int nx)
{
    const int c = threadIdx.x;
    const int k = threadIdx.y;
    const int lane = k * 8 + c;
    __shared__ int selected[32], selected_count, enabled[32];
    if (lane < 32) {
        const int candidate = blockIdx.x * 32 + lane;
        enabled[lane] = candidate < ny * nx && active_columns[candidate] != 0.0f;
    }
    __syncthreads();
    if (lane == 0) {
        int count = 0;
        for (int candidate = 0; candidate < 32; ++candidate) {
            if (enabled[candidate]) selected[count++] = candidate;
        }
        selected_count = count;
    }
    __syncthreads();
    if (selected_count == 0) return;
    for (int group = 0; group < selected_count; group += 8) {
    const int column = group + c < selected_count
        ? blockIdx.x * 32 + selected[group + c] : ny * nx;
    const int j = column / nx;
    const int i = column - j * nx;
    const bool live = column < ny * nx && active_columns[column] != 0.0f;
    const float* graupel_number_per_kg = nullptr;
    __shared__ float mass_flux[64][8], number_flux[64][8];
    __shared__ float defined_mass[64][8], defined_number[64][8];
    __shared__ int defined[64][8], steps[64][8], tops[64][8];
    __shared__ int column_nstep[8], column_top[8];
    float density;
    float graupel_mass;
    float diagnostic_number;
    float mass_velocity;
    float number_velocity;
    float qg_tendency;
    float number_tendency;
    float qg_initial;
    float number_initial;
    const float pi = 3.1415926536f;
    const float am_g = pi * 400.0f / 6.0f;
    const float ogg3 = 1.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above = 0.0f;
    float number_velocity_above = 0.0f;

    defined[k][c] = 0;
    if (live && k < nz) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density = rho;
        qg_initial = qg[idx];
        qg_tendency = 0.0f;
        if (true) {
            number_initial = graupel_number_shadow[idx];
            number_tendency = 0.0f;
        }

        defined[k][c] = qg[idx] > 1.0e-12f;
        if (defined[k][c]) {
            const float state_rho = reference_density == nullptr
                ? rho : reference_density[idx];
            const float rg = qg[idx] * state_rho;
            double lambda;
            if (graupel_number_per_kg != (const float*)0) {
                const float number = fmaxf(
                    1.0e-6f, graupel_number_per_kg[idx] * state_rho);
                lambda = (double)powf(
                    am_g * 6.0f * number / rg, 1.0f / 3.0f);
            } else {
                const float log_mass = log10f(fmaxf(1.0e-9f, rg));
                const float intercept_power = fmaxf(2.0f, fminf(
                    3.0f + (2.0f / 7.0f) * (log_mass + 8.0f), 6.0f));
                const float intercept = powf(10.0f, intercept_power);
                const float lambda_arg = intercept * am_g * 6.0f / rg;
                lambda = (double)powf(lambda_arg, 0.25f);
            }
            float mvd = (float)(3.672 / lambda);
            if (mvd > 25.4e-3f) {
                mvd = 25.4e-3f;
                lambda = 3.672 / (double)mvd;
            } else if (mvd < 50.0e-6f) {
                mvd = 50.0e-6f;
                lambda = 3.672 / (double)mvd;
            }
            const float rhof = sqrtf(rho_not / rho);
            const double velocity_power = pow(1.0 / lambda, 0.89);
            const float prefix = rhof * 442.0f * 20.3632278f * ogg3;
            mass_velocity = (float)((double)prefix
                * velocity_power);
            graupel_mass = rg;
            if (true) {
                diagnostic_number = (float)((double)(__fdiv_rn((1.0f / 6.0f)
                    * rg, am_g)) * lambda * lambda * lambda);
                const float number_prefix = rhof * 442.0f * 2.21880022f;
                number_velocity = (float)((double)number_prefix
                    * velocity_power);
            }
        } else {
            graupel_mass = 1.0e-12f;
            mass_velocity = velocity_above;
            if (true) {
                diagnostic_number = 1.0e-6f;
                number_velocity = number_velocity_above;
            }
        }
    }
    defined_mass[k][c] = live && k < nz ? mass_velocity : 0.0f;
    defined_number[k][c] = live && k < nz ? number_velocity : 0.0f;
    __syncthreads();
    if (live && k < nz) {
        if (!defined[k][c]) {
            for (int above = k + 1; above < nz; ++above) {
                if (defined[above][c]) {
                    mass_velocity = defined_mass[above][c];
                    number_velocity = defined_number[above][c];
                    break;
                }
            }
        }
        const size_t idx = IDX3(k, j, i);
        if (mass_velocity > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = dz[idx] / mass_velocity;
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    steps[k][c] = nstep;
    tops[k][c] = sediment_top;
    __syncthreads();
    if (k == 0) {
        int n = 0, t = 0;
        for (int lane = 0; lane < nz; ++lane) {
            n = max(n, steps[lane][c]);
            t = max(t, tops[lane][c]);
        }
        column_nstep[c] = max(n, 1);
        column_top[c] = t == nz - 1 ? nz - 2 : t;
    }
    __syncthreads();
    nstep = column_nstep[c];
    sediment_top = column_top[c];
    const float onstep = 1.0f / (float)nstep;
    const float dt_substep = dt * onstep;
    float exported = 0.0f;
    int block_nstep = 1;
    for (int col = 0; col < 8; ++col) block_nstep = max(block_nstep, column_nstep[col]);
    for (int step = 0; step < block_nstep; ++step) {
        if (live && k < nz && step < nstep) {
            mass_flux[k][c] = mass_velocity * graupel_mass;
            number_flux[k][c] = number_velocity * diagnostic_number;
        }
        __syncthreads();
        if (live && k == nz - 1 && step < nstep) {
        const size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density;
        qg_tendency -= mass_flux[k][c] * inv_dz * onstep * inv_rho;
        if (true) {
            number_tendency -=
                number_flux[k][c] * inv_dz * onstep * inv_rho;
        }
        graupel_mass = fmaxf(1.0e-12f,
            graupel_mass - mass_flux[k][c] * inv_dz * dt_substep);
        if (true) {
            diagnostic_number = fmaxf(1.0e-6f,
                diagnostic_number
                    - number_flux[k][c] * inv_dz * dt_substep);
        }

        }
        if (live && k <= sediment_top && step < nstep) {
            const size_t idx = IDX3(k, j, i);
            const float inv_dz = 1.0f / dz[idx];
            const float inv_rho = 1.0f / density;
            const float divergence = mass_flux[k + 1][c] - mass_flux[k][c];
            qg_tendency += divergence * inv_dz * onstep * inv_rho;
            graupel_mass = fmaxf(1.0e-12f,
                graupel_mass + divergence * inv_dz * dt_substep);
            if (true) {
                const float number_divergence =
                    number_flux[k + 1][c] - number_flux[k][c];
                number_tendency +=
                    number_divergence * inv_dz * onstep * inv_rho;
                diagnostic_number = fmaxf(1.0e-6f,
                    diagnostic_number
                        + number_divergence * inv_dz * dt_substep);
            }
        }
        if (live && k == 0 && step < nstep && graupel_mass > 1.0e-9f) {
            exported += mass_flux[0][c] * dt_substep;
        }
        __syncthreads();
    }
    if (live && k == 0) {
    if (accumulate_surface) {
        rainncv[column] += exported;
        graupelncv[column] += exported;
    } else {
        rainncv[column] = exported;
        graupelncv[column] = exported;
    }
    rainnc[column] += exported;
    graupelnc[column] += exported;
    }
    if (live && k < nz) {
        const size_t idx = IDX3(k, j, i);
        const float qg_new = qg_initial + qg_tendency * dt;
        qg[idx] = qg_new <= 1.0e-12f ? 0.0f : qg_new;
        if (true) {
            graupel_number_shadow[idx] =
                number_initial + number_tendency * dt;
        }
    }
    __syncthreads();
    }
}

// The height search retains its serial float additions. Level updates are Jacobi.
extern "C" __global__ void thompson_cloud_sediment_levels_64_with_density_and_masks(
    float* __restrict__ qc,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ rain_active_columns,
    const float* __restrict__ cloud_active_columns,
    const float* __restrict__ vertical_velocity,
    const float* __restrict__ dz,
    float dt, int nz, int ny, int nx)
{
    const int c = threadIdx.x;
    const int column = blockIdx.x * 8 + c;
    const int j = column / nx;
    const int i = column - j * nx;
    const int k = threadIdx.y;
    const bool live = column < ny * nx
        && (cloud_active_columns == nullptr || cloud_active_columns[column] != 0.0f);
    __shared__ float depth[64][8], mass[64][8], mass_flux[64][8];
    __shared__ int top[8];
    float density, cloud_mass, mass_velocity, qc_tendency, qc_initial;
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const bool rain_refreshes_rhof = live && rain_active_columns != nullptr
        && rain_active_columns[column] != 0.0f;

    if (live && k < nz) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        density = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        qc_initial = qc[idx];
        qc_tendency = 0.0f;
        cloud_mass = qc[idx] > 1.0e-12f
            ? qc[idx] * reference_density[idx] : 1.0e-12f;
        mass_velocity = 0.0f;
        depth[k][c] = dz[idx];
        mass[k][c] = cloud_mass;
    }
    __syncthreads();
    if (k == 0) {
        int sediment_top = 0;
        float height_agl = 0.0f;
        if (live) for (int level = 0; level < nz - 1; ++level) {
            if (mass[level][c] > 1.0e-6f) sediment_top = level;
            height_agl += depth[level][c];
            if (height_agl > 500.0f) break;
        }
        top[c] = sediment_top;
    }
    __syncthreads();
    const int sediment_top = top[c];
    if (live && k <= sediment_top) {
        const size_t idx = IDX3(k, j, i);
        if (cloud_mass > 1.0e-12f
                && vertical_velocity[idx] < 1.0e-1f) {
            const float lambda_arg = 100.0e6f * am_r * 2730.0f
                / cloud_mass;
            const double lambda =
                (double)powf(lambda_arg, 1.0f / 3.0f);
            const double inverse_lambda = 1.0 / lambda;
            const float velocity_density = rain_refreshes_rhof
                ? density : reference_density[idx];
            const float prefix = sqrtf(rho_not / velocity_density)
                * 0.316946e8f * 272.0f;
            mass_velocity = (float)((double)prefix
                * inverse_lambda * inverse_lambda);
        }
    }
    if (live && k < nz) mass_flux[k][c] = mass_velocity * cloud_mass;
    __syncthreads();
    if (live && k <= sediment_top) {
        const size_t idx = IDX3(k, j, i);
        const float divergence = mass_flux[k + 1][c] - mass_flux[k][c];
        const float inv_dz = 1.0f / dz[idx];
        qc_tendency += divergence * inv_dz / density;
        cloud_mass = fmaxf(1.0e-12f,
            cloud_mass + divergence * inv_dz * dt);
    }
    if (live && k < nz) {
        const size_t idx = IDX3(k, j, i);
        qc[idx] = qc_initial + qc_tendency * dt;
    }
}
#endif  // __CUDACC_RTC__
