// Default WRF v4.6.1 NSSL option-18 NUCOND production stage.
//
// Numerical authority: phys/module_mp_nssl_2mom.F:6052-6199 (QVEXCESS)
// and :9611-12215 (NUCOND), initialized with ipconc=5, rcond=2,
// irenuc=2, iqcinit=2, imaxsupopt=4, predicted CCN, and no WRF-Chem
// droplet override. Registry number/CCN inputs are #/kg dry air; the source
// routine's internal concentration form is reconstructed with dry density.

__device__ __forceinline__ int nssl2_sat_index(float temperature)
{
    int index = (int)(__fdiv_rn((temperature - 163.15f), 0.002f) + 1.5f);
    if (index < 1) index = 1;
    if (index > 1000001) index = 1000001;
    return index;
}

__device__ __forceinline__ float nssl2_sat_table(float temperature)
{
    const int index = nssl2_sat_index(temperature);
    // WRF builds TABQVS with two separately rounded REAL operations.
    // Prevent NVRTC from contracting the multiply-add: at large table
    // indices the fused result differs by one temperature ULP, which is
    // amplified in near-saturated cloud evaporation and predicted CCN.
    const float table_temperature = __fadd_rn(
        163.15f, __fmul_rn((float)(index - 1), 0.002f));
    return expf(
        17.2693882f * (table_temperature - 273.15f)
        / (table_temperature - 35.86f));
}

__device__ __forceinline__ float nssl2_qvs(
    float temperature, float pressure)
{
    return (380.0f / pressure) * nssl2_sat_table(temperature);
}

__device__ __forceinline__ float nssl2_dtabqvs(float temperature)
{
    const int index = nssl2_sat_index(temperature);
    const float table_temperature = __fadd_rn(
        163.15f, __fmul_rn((float)(index - 1), 0.002f));
    const float table = expf(
        17.2693882f * (table_temperature - 273.15f)
        / (table_temperature - 35.86f));
    const float offset = table_temperature - 35.86f;
    return (
        -17.2693882f * (-273.15f + table_temperature)
            / (offset * offset)
        + 17.2693882f / offset) * table;
}

__device__ __forceinline__ float nssl2_supersaturation_percent(
    float theta, float pressure, float exner, float vapor)
{
    const float saturation = nssl2_qvs(theta * exner, pressure);
    return 100.0f * (vapor / saturation - 1.0f);
}

// Two-iteration modified-Straka adjustment. QVEXCESS returns only the
// positive cloud increment; its private trial state is not scattered.
__device__ __forceinline__ float nssl2_qvexcess(
    float theta, float pressure, float exner, float vapor, float cloud,
    float latent_over_cp, float target_supersaturation_percent)
{
    float trial_theta_increment = 0.0f;
    float trial_vapor = fmaxf(vapor, 0.0f);
    float trial_cloud = fmaxf(cloud, 0.0f);
    const float condensation_factor =
        4098.0258f * latent_over_cp;

    for (int iteration = 0; iteration < 2; ++iteration) {
        const float temperature =
            (theta + trial_theta_increment) * exner;
        const float target_vapor =
            (1.0f + 0.01f * target_supersaturation_percent)
            * nssl2_qvs(temperature, pressure);
        float vapor_delta = trial_vapor - target_vapor;
        float cloud_delta = 0.0f;
        if (vapor_delta < 0.0f) {
            cloud_delta = fmaxf(vapor_delta, -trial_cloud);
        } else {
            const float temperature_offset = temperature - 35.86f;
            cloud_delta = vapor_delta /
                (1.0f + condensation_factor * target_vapor
                 / (temperature_offset * temperature_offset));
        }
        trial_theta_increment +=
            latent_over_cp * cloud_delta / exner;
        trial_vapor -= cloud_delta;
        trial_cloud += cloud_delta;
        trial_vapor = fmaxf(trial_vapor, 0.0f);
        trial_cloud = fmaxf(trial_cloud, 0.0f);
    }
    return fmaxf(0.0f, trial_cloud - cloud);
}

// WRF computes SSFILT for the complete column before NUCOND mutates any
// thermodynamic field.  Keep that ordering explicit so vertical neighbor
// checks are deterministic regardless of CUDA block scheduling.
extern "C" __global__ void nssl2_nucond_supersaturation(
    const float* __restrict__ full_theta,
    const float* __restrict__ pressure_pa,
    const float* __restrict__ exner,
    const float* __restrict__ qv,
    float* __restrict__ supersaturation_percent,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    supersaturation_percent[idx] = nssl2_supersaturation_percent(
        full_theta[idx], pressure_pa[idx], exner[idx], fmaxf(qv[idx], 0.0f));
}

extern "C" __global__ void nssl2_nucond_default(
    float* __restrict__ full_theta,
    const float* __restrict__ air_density,
    const float* __restrict__ pressure_pa,
    const float* __restrict__ exner,
    const float* __restrict__ w_interface,
    const float* __restrict__ initial_supersaturation_percent,
    float* __restrict__ qv,
    float* __restrict__ qc,
    float* __restrict__ qr,
    const float* __restrict__ qi,
    const float* __restrict__ qs,
    float* __restrict__ qndrop,
    float* __restrict__ qnr,
    const float* __restrict__ qni,
    const float* __restrict__ qns,
    float* __restrict__ qnn,
    float dt,
    int concentration_space,
    int n,
    int nz,
    int horizontal_size,
#ifdef NSSL2_VARIABLE_CCCN
    int predicted_ccn,
    float background_ccn_per_kg,
    float nssl_cccn)
#else
    int predicted_ccn)
#endif
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float rho = air_density[idx];
    const float pressure = pressure_pa[idx];
    const float exner_local = exner[idx];
    float theta = full_theta[idx];
    float vapor = qv[idx];
    float cloud = qc[idx];
    float rain = qr[idx];
    const float input_cloud_number = qndrop[idx];
    const float input_rain_number = qnr[idx];
    const float input_ccn = qnn[idx];
    const float input_number_scale = concentration_space ? 1.0f : rho;
    const float output_number_scale = concentration_space ? 1.0f : 1.0f / rho;
    float cloud_number = input_cloud_number * input_number_scale;
    float rain_number = input_rain_number * input_number_scale;
    float ccn_number = input_ccn * input_number_scale;

    const float qxmin_cloud = 1.0e-13f;
    const float qxmin_rain = 1.0e-12f;
    const float cxmin = 1.0e-8f;
    const float pi = 3.14159265358979323846f;
    const float cloud_min_mass =
        1000.0f * 0.523599f * (4.0e-6f * 4.0e-6f * 4.0e-6f);
    const float cloud_max_mass =
        1000.0f * 0.523599f * (120.0e-6f * 120.0e-6f * 120.0e-6f);
    const float cloud_five_micron_mass =
        1000.0f * 0.523599f * (10.0e-6f * 10.0e-6f * 10.0e-6f);
    const float cloud_twenty_micron_mass =
        1000.0f * 0.523599f * (40.0e-6f * 40.0e-6f * 40.0e-6f);
    const float rain_min_volume =
        0.523599f * (80.0e-6f * 80.0e-6f * 80.0e-6f);
    const float rain_max_volume =
        0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f);
#ifdef NSSL2_VARIABLE_CCCN
    const float qccn_concentration = rho * background_ccn_per_kg;
    // Keep the established fast default arithmetic. For an override,
    // follow WRF NUCOND: multiply the concentration before dividing.
    const float background_ccn = nssl_cccn == 0.5e9f
        ? rho * background_ccn_per_kg
        : __fdiv_rn(__fmul_rn(nssl_cccn, rho), 1.225f);
#else
    const float background_ccn = rho * 408163264.0f;
#endif
    const float input_background_ccn =
#ifdef NSSL2_VARIABLE_CCCN
        concentration_space ? qccn_concentration : background_ccn_per_kg;
#else
        concentration_space ? background_ccn : 408163264.0f;
#endif

    float temperature = theta * exner_local;
    float saturation = nssl2_qvs(temperature, pressure);
    float saturation_ratio = vapor / saturation;
    float supersaturation_percent =
        100.0f * (saturation_ratio - 1.0f);
    const bool gathered =
        (temperature > 233.15f || saturation_ratio > 1.08f)
        && (vapor > saturation || cloud > qxmin_cloud
            || rain > qxmin_rain);

    // Preserve a true inactive dry-cell no-op at this Registry-unit API.
    // WRF's surrounding mixconv round-trip is outside NUCOND itself.
    if (!gathered && cloud == 0.0f && rain == 0.0f
            && input_cloud_number == 0.0f
            && input_rain_number == 0.0f
            && input_ccn == input_background_ccn) {
        return;
    }

    if (gathered) {
        vapor = fmaxf(vapor, 0.0f);
        cloud = fmaxf(cloud, 0.0f);
        rain = fmaxf(rain, 0.0f);
        cloud_number = fmaxf(cloud_number, 0.0f);
        rain_number = fmaxf(rain_number, 0.0f);
        ccn_number = fmaxf(ccn_number, 0.0f);

        // WRF's nucleation pool `cnuc`, module_mp_nssl_2mom.F:10112-10131.
        // It is formed ONCE per grid point from the gather-time CCN and is
        // never refreshed as CCN deplete within the step.
        //
        // With predicted CCN the driver leaves renucfrac at its 0.0 default
        // (:346) and cnuc = Max(ccnc, cwnccn); that is the shipped default
        // lane, and its existing in-line spelling below is left untouched
        // so the mp18 trajectory does not move.
        //
        // With nssl_ccn_on=0 the driver sets renucfrac = 1.0 at :2555-2557,
        // which collapses :10116 to cnuc = ccnc -- the actual available
        // (diagnosed) CCN rather than the background floor -- and arms the
        // low-temperature limiter at :10120-10127 that WRF describes as a
        // "hack to reduce nucleation at low temp in updraft when ccn are
        // not predicted".  Both are transcribed here.
        float diagnostic_cnuc = 0.0f;
        if (!predicted_ccn) {
            diagnostic_cnuc = ccn_number;
            if (temperature < 265.0f) {
                // :10122 reads the RAW staggered element
                // w(igs(mgs),jgs,kgs(mgs)) -- the cell's own bottom face --
                // not a mass-level average.  That is not an oversight to
                // correct: wvel is not assigned until :10381, well after
                // this block, so the routine has no averaged w to read yet.
                const float cnuc_w = w_interface[idx];
                if (cloud > 10.0f * qxmin_cloud && cnuc_w > 2.0f) {
                    diagnostic_cnuc = 0.0f;
                } else {
                    diagnostic_cnuc = 0.1f * diagnostic_cnuc;
                }
            }
        }

        float cloud_mean_mass = cloud_min_mass;
        if (cloud_number > 1.0e6f) {
            cloud_mean_mass = fminf(
                cloud_max_mass,
                fmaxf(cloud_min_mass, rho * cloud / cloud_number));
        } else if (cloud > qxmin_cloud && cloud_number > cxmin) {
            cloud_mean_mass = fminf(
                cloud_max_mass,
                fmaxf(cloud_min_mass, rho * cloud / cloud_number));
            cloud_number = rho * cloud / cloud_mean_mass;
        } else if (cloud > qxmin_cloud) {
            cloud_number = fmaxf(
                cxmin, __fdiv_rn(rho * cloud, cloud_max_mass));
            cloud_mean_mass = fminf(
                cloud_max_mass,
                fmaxf(cloud_min_mass, rho * cloud / cloud_number));
        }
        const float cloud_diameter = powf(
            cloud_mean_mass * (6.0f / (pi * 1000.0f)), 1.0f / 3.0f);

        float rain_mean_volume = rain_min_volume;
        float rain_diameter = 1.0e-9f;
        if (rain > qxmin_rain) {
            rain_mean_volume =
                rho * rain / (1000.0f * fmaxf(1.0e-9f, rain_number));
            if (rain_mean_volume > rain_max_volume) {
                rain_mean_volume = rain_max_volume;
                rain_number = __fdiv_rn(rho * rain, (rain_max_volume * 1000.0f));
            } else if (rain_mean_volume < rain_min_volume) {
                rain_mean_volume = rain_min_volume;
                rain_number = __fdiv_rn(rho * rain, (rain_min_volume * 1000.0f));
            }
            // imurain=1, alphar=0: 6*xv/[pi*(3*2*1)] = xv/pi.
            rain_diameter = powf(__fdiv_rn(rain_mean_volume, pi), 1.0f / 3.0f);
        }

        const float bounded_temperature =
            fminf(313.15f, fmaxf(233.15f, temperature));
        const float latent_heat = 2500837.367f * powf(
            273.15f / bounded_temperature,
            0.167f + 3.67e-4f * bounded_temperature);
        const float latent_over_cp = latent_heat * (1.0f / 1004.0f);
        const float latent_over_cp_exner = latent_over_cp / exner_local;
        float cloud_increment = 0.0f;
        float rain_increment = 0.0f;

        if (supersaturation_percent <= 0.0f && cloud > 0.0f) {
            const float temperature_offset = temperature - 35.86f;
            const float evaporation_factor = 1.0f / (
                1.0f
                + 17.2693882f * (273.15f - 35.86f) * saturation
                    * latent_heat
                    / (1004.0f * temperature_offset * temperature_offset));
            const float evaporated = fminf(
                cloud, evaporation_factor * (saturation - vapor));
            if (cloud <= evaporated) {
                vapor += cloud;
                theta -= latent_over_cp_exner * cloud;
                ccn_number = fmaxf(
                    ccn_number,
#ifdef NSSL2_VARIABLE_CCCN
                    fminf(qccn_concentration, ccn_number + cloud_number));
#else
                    fminf(background_ccn, ccn_number + cloud_number));
#endif
                cloud = 0.0f;
                cloud_number = 0.0f;
            } else {
                const float old_cloud = cloud;
                vapor += evaporated;
                cloud -= evaporated;
                const float removed_number =
                    0.9f * evaporated * cloud_number / old_cloud;
                ccn_number = fmaxf(
                    ccn_number,
#ifdef NSSL2_VARIABLE_CCCN
                    fminf(qccn_concentration, ccn_number + removed_number));
#else
                    fminf(background_ccn, ccn_number + removed_number));
#endif
                cloud_number -= removed_number;
                theta -= latent_over_cp_exner * evaporated;
            }
        } else if (supersaturation_percent > 0.0f
                   && cloud > qxmin_cloud && cloud_number >= 1.0f) {
            const float dynamic_viscosity =
                1.832e-5f * (416.16f / (temperature + 120.0f))
                * powf(__fdiv_rn(temperature, 296.0f), 1.5f);
            const float thermal_conductivity =
                __fdiv_rn(2.43e-2f * dynamic_viscosity, 1.718e-5f);
            const float vapor_diffusivity =
                2.11e-5f * powf(__fdiv_rn(temperature, 273.15f), 1.94f)
                * (101325.0f / pressure);
            const float vapor_pressure =
                610.78f * nssl2_sat_table(temperature);
            const float ac1 = latent_heat * latent_heat /
                (thermal_conductivity * 461.5f
                 * temperature * temperature);
            const float bc = 461.5f * temperature /
                (vapor_diffusivity * vapor_pressure);
            const float resistance_inverse = 1.0f / (ac1 + bc);
            const float cloud_transfer = resistance_inverse
                * 4.0f * pi * 0.8929795026779175f
                * 0.5f * cloud_diameter * cloud_number / rho;

            float rain_transfer = 0.0f;
            if (rain > qxmin_rain && rain_number > 1.0e-9f) {
                const float kinematic_viscosity = dynamic_viscosity / rho;
                const float vapor_schmidt =
                    kinematic_viscosity / vapor_diffusivity;
                const float ventilation_scale =
                    powf(vapor_schmidt, 1.0f / 3.0f)
                    * powf(kinematic_viscosity, -0.5f);
                const float density_velocity = sqrtf(1.225f / rho);
                const float rain_ventilation =
                    0.78f
                    + 0.308f * 1.8273550271987915f
                        * ventilation_scale
                        * sqrtf(841.99666f * density_velocity)
                        * powf(rain_diameter, 0.9f);
                rain_transfer = resistance_inverse
                    * 4.0f * pi * rain_ventilation
                    * 0.5f * rain_diameter * rain_number / rho;
            }

            if (cloud_transfer > 0.0f && isfinite(cloud_transfer)) {
                float trial_vapor = vapor;
                float trial_saturation = saturation;
                float trial_temperature = temperature;
                float trial_ratio = trial_vapor / trial_saturation;
                float previous_ratio = trial_ratio;
                float previous_temperature = trial_temperature;
                float delta;
                if (fabsf(trial_ratio - 1.0f) > 1.0e-5f) {
                    delta = 0.5f * (trial_vapor - trial_saturation)
                        / (cloud_transfer * (trial_ratio - 1.0f));
                } else {
                    delta = 0.1f * dt;
                }
                const float dt_small = fminf(0.05f, 0.2f * delta);
                int native_steps = 2 * (int)floorf(
                    (dt - 4.0f * dt_small) / delta + 0.5f);
                if (native_steps < 5) native_steps = 5;
                const float dt_large =
                    (dt - 4.0f * dt_small) / (float)native_steps;

                int step_index = 1;
                int guard = 0;
                float elapsed = 0.0f;
                bool stop_adjustment = false;
                while (elapsed < dt && guard < 100000) {
                    ++guard;
                    float dt_condense =
                        step_index <= 4 ? dt_small : dt_large;
                    float midpoint_cloud;
                    float midpoint_rain;
                    float midpoint_temperature_change;
                    float midpoint_vapor;
                    float midpoint_saturation;
                    float midpoint_ratio;
                    while (true) {
                        midpoint_cloud =
                            -(trial_ratio - 1.0f)
                            * cloud_transfer * dt_condense;
                        midpoint_rain =
                            -(trial_ratio - 1.0f)
                            * rain_transfer * dt_condense;
                        midpoint_temperature_change =
                            -0.5f * latent_over_cp
                            * (midpoint_cloud + midpoint_rain);
                        const float midpoint_temperature =
                            trial_temperature
                            + midpoint_temperature_change;
                        const float midpoint_saturation_change =
                            midpoint_temperature_change
                            * (380.0f / pressure)
                            * nssl2_dtabqvs(midpoint_temperature);
                        midpoint_vapor = trial_vapor
                            + midpoint_cloud + midpoint_rain;
                        midpoint_saturation = trial_saturation
                            + midpoint_saturation_change;
                        midpoint_ratio =
                            midpoint_vapor / midpoint_saturation;
                        if (midpoint_ratio < 1.0f) {
                            dt_condense *= 0.5f;
                            if (dt_condense >= dt_small) continue;
                            stop_adjustment = true;
                        }
                        break;
                    }
                    if (stop_adjustment) break;

                    const float vapor_to_cloud =
                        -(midpoint_ratio - 1.0f)
                        * cloud_transfer * dt_condense;
                    const float vapor_to_rain =
                        -(midpoint_ratio - 1.0f)
                        * rain_transfer * dt_condense;
                    const float temperature_change =
                        -latent_over_cp
                        * (vapor_to_cloud + vapor_to_rain);
                    const float final_temperature =
                        trial_temperature + temperature_change;
                    const float saturation_change = temperature_change
                        * (380.0f / pressure)
                        * nssl2_dtabqvs(final_temperature);

                    trial_vapor += vapor_to_cloud + vapor_to_rain;
                    cloud_increment -= vapor_to_cloud;
                    rain_increment -= vapor_to_rain;
                    trial_saturation += saturation_change;
                    trial_ratio = trial_vapor / trial_saturation;
                    trial_temperature += temperature_change;
                    if (previous_temperature == trial_temperature
                            || previous_ratio == trial_ratio
                            || trial_ratio == 1.0f
                            || (step_index > 10
                                && trial_ratio < 1.0005f)) {
                        break;
                    }
                    previous_ratio = trial_ratio;
                    previous_temperature = trial_temperature;
                    elapsed += dt_condense;
                    ++step_index;
                }

                theta += latent_over_cp_exner
                    * (cloud_increment + rain_increment);
                vapor -= cloud_increment + rain_increment;
                cloud += cloud_increment;
                rain += rain_increment;
                temperature = theta * exner_local;
                saturation = nssl2_qvs(temperature, pressure);
            }

            // Default irenuc=2 continuation for an existing cloud.
            const int k = idx / horizontal_size;
            const float mass_level_w = 0.5f * (
                w_interface[idx] + w_interface[idx + horizontal_size]);
            bool admit_renucleation = cloud_increment > 0.0f
                && supersaturation_percent > 0.5f
                && supersaturation_percent < 238.0f
                && !(k == 0 && mass_level_w > 0.0f);
            if (admit_renucleation && k > 0 && k < nz - 2) {
                const float below_ss = initial_supersaturation_percent[
                    idx - horizontal_size];
                const float above_ss = initial_supersaturation_percent[
                    idx + horizontal_size];
                if (below_ss >= 238.0f || above_ss >= 238.0f) {
                    admit_renucleation = false;
                }
            }
            if (admit_renucleation) {
                const float nucleation_pool = predicted_ccn
                    ? fmaxf(ccn_number, background_ccn)
                    : diagnostic_cnuc;
                const float diagnosed_activated =
                    background_ccn - ccn_number;
                float activated = 23.984773635864258f
                    * powf(nucleation_pool, 0.7692307829856873f)
                    * powf(fmaxf(mass_level_w, 0.0f),
                           0.3461538553237915f);
                activated = fminf(activated, ccn_number);
                activated = fminf(
                    activated, __fdiv_rn(0.5f * cloud_increment, cloud_min_mass));
                activated = fminf(
                    activated,
                    fmaxf(0.0f, nucleation_pool - diagnosed_activated));
                activated = fmaxf(activated, 0.0f);
                cloud_number += activated;
                ccn_number = fmaxf(0.0f, ccn_number - activated);
            }
        } else if (supersaturation_percent > 0.0f) {
            // No pre-existing cloud: iqcinit=2 ordinary QVEXCESS path.
            float initial_cloud_increment = 0.0f;
            if (supersaturation_percent > 0.4f
                    && supersaturation_percent < 20.0f
                    && ccn_number > 0.05f * background_ccn) {
                initial_cloud_increment = nssl2_qvexcess(
                    theta, pressure, exner_local, vapor, cloud,
                    latent_over_cp, 0.4f);
            }
            theta += latent_over_cp_exner * initial_cloud_increment;
            vapor -= initial_cloud_increment;
            cloud += initial_cloud_increment;
            cloud_increment = initial_cloud_increment;
            temperature = theta * exner_local;
            saturation = nssl2_qvs(temperature, pressure);

            const float mass_level_w = 0.5f * (
                w_interface[idx] + w_interface[idx + horizontal_size]);
            if (initial_cloud_increment > qxmin_cloud
                    && mass_level_w > 0.0f) {
                const float activation_pool = predicted_ccn
                    ? fmaxf(ccn_number, background_ccn)
                    : diagnostic_cnuc;
                float activated = 23.984773635864258f
                    * powf(activation_pool, 0.7692307829856873f)
                    * powf(mass_level_w, 0.3461538553237915f);
                const float four_micron_mass =
                    1000.0f * (4.0f * pi / 3.0f)
                    * (4.0e-6f * 4.0e-6f * 4.0e-6f);
                activated = fminf(
                    background_ccn,
                    fmaxf(activated, __fdiv_rn(rho * cloud, four_micron_mass)));
                activated = fminf(activated, ccn_number);
                ccn_number = fmaxf(0.0f, ccn_number - activated);
                cloud_number = fmaxf(cloud_number, activated);
                cloud_number = fminf(
                    cloud_number, __fdiv_rn(rho * fmaxf(cloud, 0.0f),
                        cloud_min_mass));
            }
        }

        // Active default maxsupersat=1.9 / imaxsupopt=4 adjustment.
        temperature = theta * exner_local;
        saturation = nssl2_qvs(temperature, pressure);
        if (vapor > 1.9f * saturation) {
            const float vapor_excess = nssl2_qvexcess(
                theta, pressure, exner_local, vapor, cloud,
                latent_over_cp, 90.0f);
            if (vapor_excess > 0.0f) {
                theta += latent_over_cp_exner * vapor_excess;
                vapor -= vapor_excess;
                cloud += vapor_excess;
                const float activation_mass = fmaxf(
                    cloud_five_micron_mass,
                    fmaxf(cloud_twenty_micron_mass, cloud_mean_mass));
                float activated = fminf(
                    fmaxf(ccn_number, background_ccn),
                    rho * vapor_excess / activation_mass);
                ccn_number = fmaxf(0.0f, ccn_number - activated);
                cloud_number += activated;
            }
        }

        // Native final droplet mean-mass bound.
        if (cloud_number > cxmin && cloud > qxmin_cloud) {
            cloud_mean_mass = rho * cloud / cloud_number;
            if (cloud_mean_mass < cloud_min_mass
                    || cloud_mean_mass > cloud_max_mass) {
                cloud_mean_mass = fminf(
                    cloud_max_mass,
                    fmaxf(cloud_min_mass, cloud_mean_mass));
                cloud_number = rho * cloud / cloud_mean_mass;
            }
        }
        // NUCOND diagnoses/clamps the final rain mean volume but, in the
        // official rcond=2 source, does not scatter a reconstructed number.
        if (rain_number > 0.0f && rain > qxmin_rain) {
            rain_mean_volume = rho * rain / (1000.0f * rain_number);
            rain_mean_volume = fminf(
                rain_max_volume,
                fmaxf(rain_min_volume, rain_mean_volume));
        }
    }

    // All-domain warm-category cleanup from the tail of NUCOND.
    if (rain < qxmin_rain || rain_number <= 0.0f) {
        vapor += fmaxf(rain, 0.0f);
        rain = 0.0f;
        rain_number = 0.0f;
    }
    if (cloud <= qxmin_cloud || cloud_number <= 0.0f) {
        vapor += fmaxf(cloud, 0.0f);
        cloud = 0.0f;
        ccn_number += fmaxf(cloud_number, 0.0f);
        cloud_number = 0.0f;
        // WRF restores predicted CCN only when the cell is also free of
        // primary ice plus snow.  Omitting this frozen-water gate nudges CCN
        // in mixed-phase/anvil levels where official NUCOND leaves it alone.
        const float retained_ice =
            (qi[idx] > 1.0e-13f && qni[idx] > 0.0f) ? qi[idx] : 0.0f;
        const float retained_snow =
            (qs[idx] >= 1.0e-13f && qns[idx] > 0.0f) ? qs[idx] : 0.0f;
        if (ccn_number > 1.0f
                && retained_ice + retained_snow < 1.0e-13f) {
#ifdef NSSL2_VARIABLE_CCCN
            ccn_number = qccn_concentration
                - fmaxf(0.0f, qccn_concentration - ccn_number)
#else
            ccn_number = background_ccn
                - fmaxf(0.0f, background_ccn - ccn_number)
#endif
                    * expf(__fdiv_rn(-dt, 3600.0f));
        }
    }

    full_theta[idx] = theta;
    qv[idx] = fmaxf(vapor, 0.0f);
    qc[idx] = fmaxf(cloud, 0.0f);
    qr[idx] = fmaxf(rain, 0.0f);
    qndrop[idx] = fmaxf(cloud_number, 0.0f) * output_number_scale;
    qnr[idx] = fmaxf(rain_number, 0.0f) * output_number_scale;
    qnn[idx] = fmaxf(ccn_number, 0.0f) * output_number_scale;
}
