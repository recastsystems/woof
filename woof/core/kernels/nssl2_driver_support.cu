// :4748-4855 (fallout1d), and :4859-5118 (Method I+II correction).
// Default option 18 uses fixed Atlas rain velocities, infall=irfall=4,
// adaptive first-order upwind substeps, and number in volumetric #/m3 while
// inside the scheme.  Each CUDA thread owns one complete vertical column.
#define NSSL2_KMAX_SHALLOW 64
#define NSSL2_KMAX_GENERIC 256

// WRF v4.6.1 NSSL option-18 driver support.
//
// The Registry arrays are gathered once into a 16-field internal slab. Number
// and volume moments remain in concentration space across every sedimentation
// category, then one final kernel scatters them to Registry mixing ratios.
// Numerical authority: module_mp_nssl_2mom.F:2650-3059, :4242-5118,
// :5168-5513 (calcnfromq), and :5546-5739 (calcnfromcuten).
enum Nssl2DriverField {
    NSSL2_QV = 0,
    NSSL2_QC = 1,
    NSSL2_QR = 2,
    NSSL2_QI = 3,
    NSSL2_QS = 4,
    NSSL2_QG = 5,
    NSSL2_QH = 6,
    NSSL2_NC = 7,
    NSSL2_NR = 8,
    NSSL2_NI = 9,
    NSSL2_NS = 10,
    NSSL2_NG = 11,
    NSSL2_NH = 12,
    NSSL2_NN = 13,
    NSSL2_VG = 14,
    NSSL2_VH = 15,
    NSSL2_DRIVER_FIELD_COUNT = 16,
};

__device__ __forceinline__ float* nssl2_driver_field(
    float* state, int field, int n)
{
    return state + (size_t)field * (size_t)n;
}

extern "C" __global__ void nssl2_driver_gather_initialize(
    const float* __restrict__ air_density,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ qr,
    const float* __restrict__ qi,
    const float* __restrict__ qs,
    const float* __restrict__ qg,
    const float* __restrict__ qh,
    const float* __restrict__ qndrop,
    const float* __restrict__ qnr,
    const float* __restrict__ qni,
    const float* __restrict__ qns,
    const float* __restrict__ qng,
    const float* __restrict__ qnh,
    const float* __restrict__ qnn,
    const float* __restrict__ qvolg,
    const float* __restrict__ qvolh,
    const float* __restrict__ qrcuten,
    const float* __restrict__ qscuten,
    const float* __restrict__ qicuten,
    const float* __restrict__ qccuten,
    float* __restrict__ state,
    float dt,
    int first_step,
    int cu_used,
    int n,
#ifdef NSSL2_VARIABLE_CCCN
    int predicted_ccn, float background_ccn_per_kg)
#else
    int predicted_ccn)
#endif
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float rho = air_density[idx];
    const float cxmin = 1.0e-8f;
    const float qxmin_init = 1.0e-8f;
    const float qxmin_cloud = 1.0e-13f;
    const float qxmin_rain = 1.0e-12f;

    float vapor = qv[idx];
    float cloud = qc[idx];
    float rain = qr[idx];
    float ice = qi[idx];
    float snow = qs[idx];
    float graupel = qg[idx];
    float hail = qh[idx];

    // Exact driver denscale loop: all number and volume moments enter the
    // internal pipeline in concentration space once, before any processing.
    float cloud_number = qndrop[idx] * rho;
    float rain_number = qnr[idx] * rho;
    float ice_number = qni[idx] * rho;
    float snow_number = qns[idx] * rho;
    float graupel_number = qng[idx] * rho;
    float hail_number = qnh[idx] * rho;
    // module_mp_nssl_2mom.F:2719-2739 is the driver's CCN load.  With
    // predicted CCN (nssl_ccn_on=1 -> Registry package nssl_ccn_opt, so
    // f_qnn and therefore flag_ccn are true at :2464-2466) the prognostic
    // qnn field is loaded at :2727.  With nssl_ccn_on=0 the field does not
    // exist and WRF instead diagnoses the unactivated CCN every step from
    // the constant base concentration at :2734, taking the lccna==0 branch
    // because turn_on_ccna is gated on ccn_on==1 (:1403).  The subtraction
    // happens in per-mass units and only then is the whole slot scaled by
    // density at :2935-2944, so the product is formed after the difference.
    float ccn_number = predicted_ccn
        ? qnn[idx] * rho
#ifdef NSSL2_VARIABLE_CCCN
        : (background_ccn_per_kg - qndrop[idx]) * rho;
#else
        : (408163264.0f - qndrop[idx]) * rho;
#endif
    float graupel_volume = qvolg[idx] * rho;
    float hail_volume = qvolh[idx] * rho;

    if (first_step != 0) {
        if (cloud_number <= cxmin && cloud > qxmin_init) {
#ifdef NSSL2_VARIABLE_CCCN
            const float qccn = background_ccn_per_kg;
#else
            const float qccn = 408163264.0f;
#endif
            const float cwmas_inverse = 327479132160.0f;
            cloud_number = fminf(qccn, cloud * cwmas_inverse) * rho;
            ccn_number -= cloud_number;
        } else if (cloud <= qxmin_cloud
                   || (cloud_number <= cxmin && cloud <= qxmin_init)) {
            vapor += cloud;
            cloud_number = 0.0f;
            cloud = 0.0f;
        }

        if (ice_number <= cxmin && ice > qxmin_init) {
            const float xims = 4.7123910329460728e-10f;
            ice_number = __fdiv_rn(rho * ice, xims);
        } else if (ice <= qxmin_cloud
                   || (ice_number <= cxmin && ice <= qxmin_init)) {
            vapor += ice;
            ice_number = 0.0f;
            ice = 0.0f;
        }

        if (rain_number <= 0.1f * cxmin && rain > qxmin_init) {
            const float zrfac = 3.9788734806922577e-11f;
            const double lambda_inverse = pow(
                (double)rho * (double)rain * (double)zrfac, 0.25);
            rain_number = (float)(
                lambda_inverse * (double)8000000.0f
                * (double)20.0f / (double)20.0f);
        } else if (rain <= qxmin_rain
                   || (rain_number <= cxmin && rain <= qxmin_init)) {
            vapor += rain;
            rain_number = 0.0f;
            rain = 0.0f;
        }

        if (snow_number <= 0.1f * cxmin && snow > qxmin_init) {
            const float zsfac = 1.0610329281846020e-9f;
            const double lambda_inverse = pow(
                (double)rho * (double)snow * (double)zsfac, 0.25);
            snow_number = (float)(
                lambda_inverse * (double)3000000.0f
                * (double)6.0000004768371582f / (double)20.0f);
        } else if (snow <= qxmin_cloud
                   || (snow_number <= cxmin && snow <= qxmin_init)) {
            vapor += snow;
            snow_number = 0.0f;
            snow = 0.0f;
        }

        if (graupel_number <= 0.1f * cxmin && graupel > qxmin_init) {
            if (graupel_volume <= 0.0f) {
                // Historical WRF quirk: this assignment follows denscale and
                // therefore is intentionally not multiplied by air density.
                graupel_volume = __fdiv_rn(graupel, 700.0f);
            }
            const float zhfac = 2.2736419413860176e-9f;
            const float xgms = 9.8960235561662557e-9f;
            const double lambda_inverse = pow(
                (double)rho * (double)graupel * (double)zhfac, 0.25);
            const double intercept_number =
                lambda_inverse * (double)200000.0f;
            const double maximum_number =
                (double)rho * (double)graupel / (double)xgms;
            const double diagnosed = fmin(intercept_number, maximum_number);
            if (diagnosed > (double)cxmin) {
                graupel_number = (float)diagnosed;
            } else {
                graupel = 0.0f;
                graupel_number = 0.0f;
                graupel_volume = 0.0f;
            }
        } else if (graupel <= qxmin_rain
                   || (graupel_number <= cxmin
                       && graupel <= qxmin_init)) {
            vapor += graupel;
            graupel = 0.0f;
        }

        if (hail_number <= 0.1f * cxmin && hail > qxmin_init) {
            if (hail_volume <= 0.0f) {
                hail_volume = __fdiv_rn(hail, 900.0f);
            }
            const float zhlfac = 8.8419414012719244e-9f;
            const double lambda_inverse = pow(
                (double)rho * (double)hail * (double)zhlfac, 0.25);
            hail_number = (float)(
                lambda_inverse * (double)40000.0f
                * (double)8.75f / (double)20.0f);
        } else if (hail <= qxmin_rain
                   || (hail_number <= cxmin && hail <= qxmin_init)) {
            vapor += hail;
            hail = 0.0f;
        }
    }

    if (cu_used != 0) {
        // calcnfromcuten diagnoses number only. Graupel/hail branches are
        // commented out in WRF 4.6.1, while all four live KF rates are used.
        const float cloud_increment = dt * qccuten[idx];
        const float ice_increment = dt * qicuten[idx];
        const float rain_increment = dt * qrcuten[idx];
        const float snow_increment = dt * qscuten[idx];
        // WRF 4.6.1 loads qccuten/qicuten into the cloud/ice *mass* slots,
        // but calcnfromcuten gates those branches on the corresponding empty
        // ancuten number slots. Thus both rates are consumed yet diagnose no
        // number increment. Preserve that exact official-source behavior.
        (void)cloud_increment;
        (void)ice_increment;
        if (rain_increment > qxmin_rain) {
            const float zrfac = 3.9788734806922577e-11f;
            const double lambda_inverse = pow(
                (double)rho * (double)rain_increment * (double)zrfac,
                0.25);
            rain_number += (float)(
                lambda_inverse * (double)8000000.0f
                * (double)20.0f / (double)20.0f);
        }
        if (snow_increment > qxmin_cloud) {
            const float zsfac = 1.0610329281846020e-9f;
            const double lambda_inverse = pow(
                (double)rho * (double)snow_increment * (double)zsfac,
                0.25);
            snow_number += (float)(
                lambda_inverse * (double)3000000.0f
                * (double)6.0000004768371582f / (double)20.0f);
        }
    }

    nssl2_driver_field(state, NSSL2_QV, n)[idx] = vapor;
    nssl2_driver_field(state, NSSL2_QC, n)[idx] = cloud;
    nssl2_driver_field(state, NSSL2_QR, n)[idx] = rain;
    nssl2_driver_field(state, NSSL2_QI, n)[idx] = ice;
    nssl2_driver_field(state, NSSL2_QS, n)[idx] = snow;
    nssl2_driver_field(state, NSSL2_QG, n)[idx] = graupel;
    nssl2_driver_field(state, NSSL2_QH, n)[idx] = hail;
    nssl2_driver_field(state, NSSL2_NC, n)[idx] = cloud_number;
    nssl2_driver_field(state, NSSL2_NR, n)[idx] = rain_number;
    nssl2_driver_field(state, NSSL2_NI, n)[idx] = ice_number;
    nssl2_driver_field(state, NSSL2_NS, n)[idx] = snow_number;
    nssl2_driver_field(state, NSSL2_NG, n)[idx] = graupel_number;
    nssl2_driver_field(state, NSSL2_NH, n)[idx] = hail_number;
    nssl2_driver_field(state, NSSL2_NN, n)[idx] = ccn_number;
    nssl2_driver_field(state, NSSL2_VG, n)[idx] = graupel_volume;
    nssl2_driver_field(state, NSSL2_VH, n)[idx] = hail_volume;
}

// WRF v4.6.1 module_mp_nssl_2mom.F:4242-4734 (sediment1d) and
// :6211-6498/:7333-7340 (default two-moment cloud-droplet velocity).
// Cloud mass and number use the same Stokes velocity.  Number remains in
// concentration space (#/m3), while cloud mass is a dry-air mixing ratio.
template <int KMAX>
__device__ __forceinline__ void nssl2_cloud_sediment_impl(
    const float* __restrict__ air_density,
    const float* __restrict__ temperature_k,
    float* __restrict__ qc,
    float* __restrict__ qndrop,
    const float* __restrict__ dz,
    float* __restrict__ cloud_surface_export,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float cloud[KMAX];
    float number[KMAX];
    float velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];

    const float pi = 3.14159265358979323846f;
    // cwmasn/cwmasx: WRF's folded parameters (nssl2_wrf_math.cuh).
    const float minimum_mass = NSSL2_WRF_CWMASN;
    const float maximum_mass = NSSL2_WRF_CWMASX;
    const float mass_to_diameter = 6.0f / (pi * 1000.0f);
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        rho[k] = air_density[idx];
        cloud[k] = qc[idx];
        number[k] = qndrop[idx];
        velocity[k] = 0.0f;

        const float positive_cloud = fmaxf(cloud[k], 0.0f);
        if (positive_cloud > 1.0e-13f) {
            const float positive_number = fmaxf(number[k], 0.0f);
            const float effective_number = positive_number > 1.0e-8f
                ? positive_number
                : fmaxf(1.0e-8f,
                        __fdiv_rn(rho[k] * positive_cloud, maximum_mass));
            const float particle_mass = fminf(
                maximum_mass,
                fmaxf(minimum_mass,
                      positive_cloud * rho[k] / effective_number));
            const float diameter = gfk_pow(
                particle_mass * mass_to_diameter, 1.0f / 3.0f);
            const float radius = 0.5f * diameter;
            const float temperature = temperature_k[idx];
            const float viscosity = 1.832e-5f
                * (416.16f / (temperature + 120.0f))
                * gfk_pow(__fdiv_rn(temperature, 296.0f), 1.5f);
            if (viscosity > 0.0f) {
                velocity[k] = fminf(
                    70.0f,
                    (2.0f * 9.8f * 1000.0f * (radius * radius))
                        / (9.0f * viscosity));
            }
        }
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, velocity[k] * (1.0f / dz[idx]));
    }

    if (maximum_courant_rate == 0.0f) {
        cloud_surface_export[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            mass_flux[k] = cloud[k] * velocity[k] * rho[k];
            number_flux[k] = number[k] * velocity[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;

        surface_mean_flux += cloud[0] * velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            cloud[k] += dt_substep * (inverse_dz * (1.0f / rho[k]))
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        qc[idx] = cloud[k];
        qndrop[idx] = number[k];
    }
    cloud_surface_export[column] = dt * rho[0] * surface_mean_flux;
}

// One category runs in each block, so every row can reuse this storage.
__device__ __forceinline__ float* nssl2_sediment_shared_storage()
{
    __shared__ float storage[389];
    return storage;
}

__device__ __forceinline__ void nssl2_cloud_sediment_parallel_impl(
    const float* __restrict__ air_density,
    const float* __restrict__ temperature_k,
    float* __restrict__ qc,
    float* __restrict__ qndrop,
    const float* __restrict__ dz,
    float* __restrict__ cloud_surface_export,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x;
    const int k = threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho = 0.0f;
    float cloud = 0.0f;
    float number = 0.0f;
    float velocity = 0.0f;
    float* shared_sediment = nssl2_sediment_shared_storage();
    float* mass_flux = shared_sediment + 0;
    float* number_flux = shared_sediment + 65;

    const float pi = 3.14159265358979323846f;
    // cwmasn/cwmasx: WRF's folded parameters (nssl2_wrf_math.cuh).
    const float minimum_mass = NSSL2_WRF_CWMASN;
    const float maximum_mass = NSSL2_WRF_CWMASX;
    const float mass_to_diameter = 6.0f / (pi * 1000.0f);
    float maximum_courant_rate = 0.0f;

    if (k < nz) {
        const size_t idx = IDX3(k, j, i);
        rho = air_density[idx];
        cloud = qc[idx];
        number = qndrop[idx];
        velocity = 0.0f;

        const float positive_cloud = fmaxf(cloud, 0.0f);
        if (positive_cloud > 1.0e-13f) {
            const float positive_number = fmaxf(number, 0.0f);
            const float effective_number = positive_number > 1.0e-8f
                ? positive_number
                : fmaxf(1.0e-8f,
                        __fdiv_rn(rho * positive_cloud, maximum_mass));
            const float particle_mass = fminf(
                maximum_mass,
                fmaxf(minimum_mass,
                      positive_cloud * rho / effective_number));
            const float diameter = gfk_pow(
                particle_mass * mass_to_diameter, 1.0f / 3.0f);
            const float radius = 0.5f * diameter;
            const float temperature = temperature_k[idx];
            const float viscosity = 1.832e-5f
                * (416.16f / (temperature + 120.0f))
                * gfk_pow(__fdiv_rn(temperature, 296.0f), 1.5f);
            if (viscosity > 0.0f) {
                velocity = fminf(
                    70.0f,
                    (2.0f * 9.8f * 1000.0f * (radius * radius))
                        / (9.0f * viscosity));
            }
        }
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, velocity * (1.0f / dz[idx]));
    }

    float* rates = shared_sediment + 325;
    rates[k] = maximum_courant_rate;
    __syncthreads();
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (k < stride) rates[k] = fmaxf(rates[k], rates[k + stride]);
        __syncthreads();
    }
    maximum_courant_rate = rates[0];
    if (maximum_courant_rate == 0.0f) {
        if (k == 0) cloud_surface_export[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        if (k < nz) {
            mass_flux[k] = cloud * velocity * rho;
            number_flux[k] = number * velocity;
        }
        if (k == 0) {
            mass_flux[nz] = 0.0f;
            number_flux[nz] = 0.0f;
        }
        __syncthreads();

        if (k == 0) surface_mean_flux += cloud * velocity * dt_fraction;

        if (k < nz) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            cloud += dt_substep * (inverse_dz * (1.0f / rho))
                * (mass_flux[k + 1] - mass_flux[k]);
            number += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
        }
        __syncthreads();
    }

    if (k < nz) {
        const size_t idx = IDX3(k, j, i);
        qc[idx] = cloud;
        qndrop[idx] = number;
    }
    if (k == 0) cloud_surface_export[column] = dt * rho * surface_mean_flux;
}

#define NSSL2_CLOUD_SEDIMENT_PARAMETERS                                 \
    const float* __restrict__ air_density,                              \
    const float* __restrict__ temperature_k, float* __restrict__ qc,    \
    float* __restrict__ qndrop, const float* __restrict__ dz,           \
    float* __restrict__ cloud_surface_export,                           \
    float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_cloud_sediment_64(
    NSSL2_CLOUD_SEDIMENT_PARAMETERS)
{
    nssl2_cloud_sediment_impl<NSSL2_KMAX_SHALLOW>(
        air_density, temperature_k, qc, qndrop, dz,
        cloud_surface_export, dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_cloud_sediment_256(
    NSSL2_CLOUD_SEDIMENT_PARAMETERS)
{
    nssl2_cloud_sediment_impl<NSSL2_KMAX_GENERIC>(
        air_density, temperature_k, qc, qndrop, dz,
        cloud_surface_export, dt, nz, ny, nx);
}


// calczgr1d (module_mp_nssl_2mom.F:4860), REAL arithmetic:
//   xv = db*q/(xdn*N); clamp xv (N rebuilt from the clamp);
//   zx = g1*db**2*q*q/N; z = zx*(6./(pi*1000.))**2
#define NSSL2_SED_ZFAC ((6.0f / (NSSL2_WRF_PI * 1000.0f))                         * (6.0f / (NSSL2_WRF_PI * 1000.0f)))

__device__ __forceinline__ float nssl2_sediment_z(
    float q, float number, float rho, float xdn, float g1,
    float xvmn, float xvmx)
{
    if (!(q > 1.0e-12f) || !(number > 1.0e-15f)) return 0.0f;
    float xv = rho * q / (xdn * number);
    float chw = number;
    if (xv < xvmn || xv > xvmx) {
        xv = fminf(xvmx, fmaxf(xvmn, xv));
        chw = rho * q / (xv * xdn);
    }
    const float zx = g1 * (rho * rho) * q * q / chw;
    return zx * NSSL2_SED_ZFAC;
}

// calcnfromz1d (module_mp_nssl_2mom.F:4982), infall = 4, for a level whose
// advected reflectivity t0 is positive.  vr, nrx, g1, zx, chw and z are
// DOUBLE PRECISION there; xv and xdn are REAL; t0/((6./(pi*1000.))**2) is
// a REAL quotient.
__device__ __forceinline__ float nssl2_sediment_number_from_z(
    float q, float number, float rho, float xdn, float g1,
    float xvmn, float xvmx, float z_advected, float z_initial,
    float number_mass_weighted)
{
    float xv = rho * q / (xdn * number);
    double chw = (double)number;
    if (xv < xvmn || xv > xvmx) {
        xv = fminf(xvmx, fmaxf(xvmn, xv));
        chw = (double)(rho * q / (xv * xdn));
    }
    const double g1d = (double)g1;
    const double rho2 = (double)(rho * rho);
    const double qd = (double)q;
    double zx = __ddiv_rn(__dmul_rn(__dmul_rn(__dmul_rn(g1d, rho2), qd), qd), chw);
    const double z = __dmul_rn(zx, (double)NSSL2_SED_ZFAC);
    if (z > (double)z_advected && z > 0.0 && z_advected > z_initial) {
        zx = (double)__fdiv_rn(z_advected, NSSL2_SED_ZFAC);
        const double nrx = __ddiv_rn(
            __dmul_rn(__dmul_rn(__dmul_rn(g1d, rho2), qd), qd), zx);
        return fmaxf(fminf((float)nrx, number_mass_weighted), number);
    }
    return fmaxf(number_mass_weighted, number);
}

__device__ __forceinline__ float nssl2_rain_z(
    float q, float number, float rho)
{
    return nssl2_sediment_z(q, number, rho, 1000.0f, 20.0f,
                            NSSL2_WRF_XVRMN, NSSL2_WRF_XVRMX0);
}

template <int KMAX>
__device__ __forceinline__ void nssl2_rain_sediment_impl(
    const float* __restrict__ air_density,
    float* __restrict__ qr,
    float* __restrict__ qnr,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float rain[KMAX];
    float number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float z_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float z_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float z_initial[KMAX];
    float z_advected[KMAX];
    float number_mass_weighted[KMAX];

    const float pi = 3.14159265358979323846f;
    const float minimum_volume = NSSL2_WRF_XVRMN;
    const float configured_maximum_volume = NSSL2_WRF_XVRMX0;
    const float maximum_speed_volume =
        __fdiv_rn(configured_maximum_volume, 64.0f / 6.0f);
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        rho[k] = air_density[idx];
        rain[k] = qr[idx];
        number[k] = qnr[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
        z_velocity[k] = 0.0f;

        const float positive_rain = fmaxf(rain[k], 0.0f);
        if (positive_rain > 1.0e-12f) {
            const float local_number = fmaxf(number[k], 0.0f);
            float mean_volume = rho[k] * positive_rain
                / (1000.0f * fmaxf(1.0e-11f, local_number));
            if (mean_volume > maximum_speed_volume) {
                mean_volume = maximum_speed_volume;
            } else if (mean_volume < minimum_volume) {
                mean_volume = minimum_volume;
            }

            const float diameter = gfk_pow(
                __fdiv_rn((6.0f * NSSL2_WRF_PIINV) * mean_volume,
                          (3.0f * 2.0f * 1.0f)),
                1.0f / 3.0f);
            const float density_factor = sqrtf(
                1.225f * fminf(20.0f, 1.0f / rho[k]));
            const float speed_base = 1.0f + 516.575f * diameter;
            float vm = density_factor * 10.0f
                * (1.0f - gfk_pow(speed_base, -4.0f));
            float vn = density_factor * 10.0f
                * (1.0f - gfk_pow(speed_base, -1.0f));
            float vz = density_factor * 10.0f
                * (1.0f - gfk_pow(speed_base, -7.0f));
            if (vn > vm || (vm > vz && vz > 0.0f)) {
                vm = fmaxf(vm, vn);
                vz = fmaxf(vz, vm);
            }
            mass_velocity[k] = fminf(70.0f, fminf(150.0f, vm));
            number_velocity[k] = fminf(70.0f, fminf(150.0f, vn));
            z_velocity[k] = fminf(70.0f, fminf(150.0f, vz));
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, mass_velocity[k] * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, number_velocity[k] * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, z_velocity[k] * inverse_dz);
    }

    if (maximum_courant_rate == 0.0f) {
        rainncv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        // Diagnose the pre-fallout reflectivity moment and preserve the
        // pre-fallout number for the parallel mass-weighted correction.
        for (int k = 0; k < nz; ++k) {
            z_initial[k] = nssl2_rain_z(rain[k], number[k], rho[k]);
            z_advected[k] = z_initial[k];
            number_mass_weighted[k] = number[k];
            mass_flux[k] = rain[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            z_flux[k] = z_initial[k] * z_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        z_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux +=
            rain[0] * mass_velocity[0] * dt_fraction;

        // fallout1d computes every flux first, then updates every level.
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            rain[k] += dt_substep * (inverse_dz * (1.0f / rho[k]))
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            z_advected[k] += dt_substep * inverse_dz
                * (z_flux[k + 1] - z_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        // calcnfromz1d uses double temporaries for the inverse-Z number
        // reconstruction, but stores REAL(Nz) before its max/min correction.
        for (int k = 0; k < nz; ++k) {
            if (z_advected[k] > 0.0f) {
                number[k] = nssl2_sediment_number_from_z(
                    rain[k], number[k], rho[k], 1000.0f, 20.0f,
                    NSSL2_WRF_XVRMN, NSSL2_WRF_XVRMX0, z_advected[k],
                    z_initial[k], number_mass_weighted[k]);
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        qr[idx] = rain[k];
        qnr[idx] = number[k];
    }
    // WRF's surface reducer (module_mp_nssl_2mom.F:3008-3061) sums the four
    // categories' xfall before it multiplies by dtp*dn1, so this row carries
    // the raw xfall and nssl2_driver_reduce_precipitation applies WRF's
    // formula; a per-category dt*rho product cannot be summed back exactly.
    const float exported = surface_mean_flux;
    rainncv[column] = exported;
    rainnc[column] += exported;
}

template <bool ACCUMULATE = true>
__device__ __forceinline__ void nssl2_rain_sediment_parallel_impl(
    const float* __restrict__ air_density,
    float* __restrict__ qr,
    float* __restrict__ qnr,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x;
    const int k = threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho = 0.0f;
    float rain = 0.0f;
    float number = 0.0f;
    float mass_velocity = 0.0f;
    float number_velocity = 0.0f;
    float z_velocity = 0.0f;
    float* shared_sediment = nssl2_sediment_shared_storage();
    float* mass_flux = shared_sediment + 0;
    float* number_flux = shared_sediment + 65;
    float* z_flux = shared_sediment + 130;
    float* mass_number_flux = shared_sediment + 195;
    float z_initial = 0.0f;
    float z_advected = 0.0f;
    float number_mass_weighted = 0.0f;

    const float pi = 3.14159265358979323846f;
    const float minimum_volume = NSSL2_WRF_XVRMN;
    const float configured_maximum_volume = NSSL2_WRF_XVRMX0;
    const float maximum_speed_volume =
        __fdiv_rn(configured_maximum_volume, 64.0f / 6.0f);
    float maximum_courant_rate = 0.0f;

    if (k < nz) {
        const size_t idx = IDX3(k, j, i);
        rho = air_density[idx];
        rain = qr[idx];
        number = qnr[idx];
        mass_velocity = 0.0f;
        number_velocity = 0.0f;
        z_velocity = 0.0f;

        const float positive_rain = fmaxf(rain, 0.0f);
        if (positive_rain > 1.0e-12f) {
            const float local_number = fmaxf(number, 0.0f);
            float mean_volume = rho * positive_rain
                / (1000.0f * fmaxf(1.0e-11f, local_number));
            if (mean_volume > maximum_speed_volume) {
                mean_volume = maximum_speed_volume;
            } else if (mean_volume < minimum_volume) {
                mean_volume = minimum_volume;
            }

            const float diameter = gfk_pow(
                __fdiv_rn((6.0f * NSSL2_WRF_PIINV) * mean_volume,
                          (3.0f * 2.0f * 1.0f)),
                1.0f / 3.0f);
            const float density_factor = sqrtf(
                1.225f * fminf(20.0f, 1.0f / rho));
            const float speed_base = 1.0f + 516.575f * diameter;
            float vm = density_factor * 10.0f
                * (1.0f - gfk_pow(speed_base, -4.0f));
            float vn = density_factor * 10.0f
                * (1.0f - gfk_pow(speed_base, -1.0f));
            float vz = density_factor * 10.0f
                * (1.0f - gfk_pow(speed_base, -7.0f));
            if (vn > vm || (vm > vz && vz > 0.0f)) {
                vm = fmaxf(vm, vn);
                vz = fmaxf(vz, vm);
            }
            mass_velocity = fminf(70.0f, fminf(150.0f, vm));
            number_velocity = fminf(70.0f, fminf(150.0f, vn));
            z_velocity = fminf(70.0f, fminf(150.0f, vz));
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, mass_velocity * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, number_velocity * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, z_velocity * inverse_dz);
    }

    float* rates = shared_sediment + 325;
    rates[k] = maximum_courant_rate;
    __syncthreads();
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (k < stride) rates[k] = fmaxf(rates[k], rates[k + stride]);
        __syncthreads();
    }
    maximum_courant_rate = rates[0];
    if (maximum_courant_rate == 0.0f) {
        if (k == 0) rainncv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        // Diagnose the pre-fallout reflectivity moment and preserve the
        // pre-fallout number for the parallel mass-weighted correction.
        if (k < nz) {
            z_initial = nssl2_rain_z(rain, number, rho);
            z_advected = z_initial;
            number_mass_weighted = number;
            mass_flux[k] = rain * mass_velocity * rho;
            number_flux[k] = number * number_velocity;
            z_flux[k] = z_initial * z_velocity;
            mass_number_flux[k] = number * mass_velocity;
        }
        if (k == 0) {
            mass_flux[nz] = 0.0f;
            number_flux[nz] = 0.0f;
            z_flux[nz] = 0.0f;
            mass_number_flux[nz] = 0.0f;
        }
        __syncthreads();

        if (k == 0) surface_mean_flux +=
            rain * mass_velocity * dt_fraction;

        // fallout1d computes every flux first, then updates every level.
        if (k < nz) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            rain += dt_substep * (inverse_dz * (1.0f / rho))
                * (mass_flux[k + 1] - mass_flux[k]);
            number += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            z_advected += dt_substep * inverse_dz
                * (z_flux[k + 1] - z_flux[k]);
            number_mass_weighted += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        // calcnfromz1d uses double temporaries for the inverse-Z number
        // reconstruction, but stores REAL(Nz) before its max/min correction.
        if (k < nz) {
            if (z_advected > 0.0f) {
                number = nssl2_sediment_number_from_z(
                    rain, number, rho, 1000.0f, 20.0f,
                    NSSL2_WRF_XVRMN, NSSL2_WRF_XVRMX0, z_advected,
                    z_initial, number_mass_weighted);
            }
        }
        __syncthreads();
    }

    if (k < nz) {
        const size_t idx = IDX3(k, j, i);
        qr[idx] = rain;
        qnr[idx] = number;
    }
    if (k == 0) {
        // Raw xfall: see the reducer (WRF sums xfall before dtp*dn1).
        const float exported = surface_mean_flux;
        rainncv[column] = exported;
        if constexpr (ACCUMULATE) rainnc[column] += exported;
    }
}

#define NSSL2_RAIN_SEDIMENT_PARAMETERS                                  \
    const float* __restrict__ air_density, float* __restrict__ qr,       \
    float* __restrict__ qnr, const float* __restrict__ dz,               \
    float* __restrict__ rainnc, float* __restrict__ rainncv,             \
    float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_rain_sediment_64(
    NSSL2_RAIN_SEDIMENT_PARAMETERS)
{
    nssl2_rain_sediment_impl<NSSL2_KMAX_SHALLOW>(
        air_density, qr, qnr, dz, rainnc, rainncv, dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_rain_sediment_256(
    NSSL2_RAIN_SEDIMENT_PARAMETERS)
{
    nssl2_rain_sediment_impl<NSSL2_KMAX_GENERIC>(
        air_density, qr, qnr, dz, rainnc, rainncv, dt, nz, ny, nx);
}


// WRF v4.6.1 module_mp_nssl_2mom.F:4242-4734 (sediment1d) and
// :6625-6798/:7038-7118 (default two-moment snow distribution and Ferrier
// velocities).  Default option 18 uses fixed 100-kg/m3 snow density,
// isnowfall=2, infall=4, isfall=2, adaptive first-order upwind substeps, and
// the mass-weighted Method-II lower bound on the advected number moment.
template <int KMAX>
__device__ __forceinline__ void nssl2_snow_sediment_impl(
    const float* __restrict__ air_density,
    float* __restrict__ qs,
    float* __restrict__ qns,
    const float* __restrict__ dz,
    float* __restrict__ snownc,
    float* __restrict__ snowncv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float snow[KMAX];
    float number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float z_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float number_mass_weighted[KMAX];

    const float minimum_volume = NSSL2_WRF_XVSMN;
    const float maximum_volume = NSSL2_WRF_XVSMX;
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        rho[k] = air_density[idx];
        snow[k] = qs[idx];
        number[k] = qns[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
        z_velocity[k] = 0.0f;

        const float positive_snow = fmaxf(snow[k], 0.0f);
        if (positive_snow > 1.0e-13f) {
            const float local_number = fmaxf(number[k], 0.0f);
            float mean_volume = rho[k] * positive_snow
                / (100.0f * fmaxf(1.0e-9f, local_number));
            mean_volume = fminf(
                maximum_volume, fmaxf(minimum_volume, mean_volume));
            const float density_factor = sqrtf(
                1.225f * fminf(20.0f, 1.0f / rho[k]));
            const float size_factor = gfk_pow(mean_volume, 0.14f);
            mass_velocity[k] = fminf(
                70.0f, 11.9495f * density_factor * size_factor);
            number_velocity[k] = fminf(
                70.0f, 7.02909f * density_factor * size_factor);
            z_velocity[k] = fminf(
                70.0f, 13.3436f * density_factor * size_factor);
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, mass_velocity[k] * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, number_velocity[k] * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, z_velocity[k] * inverse_dz);
    }

    if (maximum_courant_rate == 0.0f) {
        snowncv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            mass_flux[k] = snow[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
            number_mass_weighted[k] = number[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux +=
            snow[0] * mass_velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            snow[k] += dt_substep * (inverse_dz * (1.0f / rho[k]))
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        for (int k = 0; k < nz; ++k) {
            number[k] = fmaxf(number[k], number_mass_weighted[k]);
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        qs[idx] = snow[k];
        qns[idx] = number[k];
    }
    // WRF's surface reducer (module_mp_nssl_2mom.F:3008-3061) sums the four
    // categories' xfall before it multiplies by dtp*dn1, so this row carries
    // the raw xfall and nssl2_driver_reduce_precipitation applies WRF's
    // formula; a per-category dt*rho product cannot be summed back exactly.
    const float exported = surface_mean_flux;
    snowncv[column] = exported;
    snownc[column] += exported;
}

template <bool ACCUMULATE = true>
__device__ __forceinline__ void nssl2_snow_sediment_parallel_impl(
    const float* __restrict__ air_density,
    float* __restrict__ qs,
    float* __restrict__ qns,
    const float* __restrict__ dz,
    float* __restrict__ snownc,
    float* __restrict__ snowncv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x;
    const int k = threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho = 0.0f;
    float snow = 0.0f;
    float number = 0.0f;
    float mass_velocity = 0.0f;
    float number_velocity = 0.0f;
    float z_velocity = 0.0f;
    float* shared_sediment = nssl2_sediment_shared_storage();
    float* mass_flux = shared_sediment + 0;
    float* number_flux = shared_sediment + 65;
    float* mass_number_flux = shared_sediment + 130;
    float number_mass_weighted = 0.0f;

    const float minimum_volume = NSSL2_WRF_XVSMN;
    const float maximum_volume = NSSL2_WRF_XVSMX;
    float maximum_courant_rate = 0.0f;

    if (k < nz) {
        const size_t idx = IDX3(k, j, i);
        rho = air_density[idx];
        snow = qs[idx];
        number = qns[idx];
        mass_velocity = 0.0f;
        number_velocity = 0.0f;
        z_velocity = 0.0f;

        const float positive_snow = fmaxf(snow, 0.0f);
        if (positive_snow > 1.0e-13f) {
            const float local_number = fmaxf(number, 0.0f);
            float mean_volume = rho * positive_snow
                / (100.0f * fmaxf(1.0e-9f, local_number));
            mean_volume = fminf(
                maximum_volume, fmaxf(minimum_volume, mean_volume));
            const float density_factor = sqrtf(
                1.225f * fminf(20.0f, 1.0f / rho));
            const float size_factor = gfk_pow(mean_volume, 0.14f);
            mass_velocity = fminf(
                70.0f, 11.9495f * density_factor * size_factor);
            number_velocity = fminf(
                70.0f, 7.02909f * density_factor * size_factor);
            z_velocity = fminf(
                70.0f, 13.3436f * density_factor * size_factor);
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, mass_velocity * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, number_velocity * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, z_velocity * inverse_dz);
    }

    float* rates = shared_sediment + 325;
    rates[k] = maximum_courant_rate;
    __syncthreads();
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (k < stride) rates[k] = fmaxf(rates[k], rates[k + stride]);
        __syncthreads();
    }
    maximum_courant_rate = rates[0];
    if (maximum_courant_rate == 0.0f) {
        if (k == 0) snowncv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        if (k < nz) {
            mass_flux[k] = snow * mass_velocity * rho;
            number_flux[k] = number * number_velocity;
            mass_number_flux[k] = number * mass_velocity;
            number_mass_weighted = number;
        }
        if (k == 0) {
            mass_flux[nz] = 0.0f;
            number_flux[nz] = 0.0f;
            mass_number_flux[nz] = 0.0f;
        }
        __syncthreads();

        if (k == 0) surface_mean_flux +=
            snow * mass_velocity * dt_fraction;

        if (k < nz) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            snow += dt_substep * (inverse_dz * (1.0f / rho))
                * (mass_flux[k + 1] - mass_flux[k]);
            number += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            number_mass_weighted += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        if (k < nz) {
            number = fmaxf(number, number_mass_weighted);
        }
        __syncthreads();
    }

    if (k < nz) {
        const size_t idx = IDX3(k, j, i);
        qs[idx] = snow;
        qns[idx] = number;
    }
    if (k == 0) {
        // Raw xfall: see the reducer (WRF sums xfall before dtp*dn1).
        const float exported = surface_mean_flux;
        snowncv[column] = exported;
        if constexpr (ACCUMULATE) snownc[column] += exported;
    }
}

#define NSSL2_SNOW_SEDIMENT_PARAMETERS                                  \
    const float* __restrict__ air_density, float* __restrict__ qs,       \
    float* __restrict__ qns, const float* __restrict__ dz,               \
    float* __restrict__ snownc, float* __restrict__ snowncv,             \
    float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_snow_sediment_64(
    NSSL2_SNOW_SEDIMENT_PARAMETERS)
{
    nssl2_snow_sediment_impl<NSSL2_KMAX_SHALLOW>(
        air_density, qs, qns, dz, snownc, snowncv, dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_snow_sediment_256(
    NSSL2_SNOW_SEDIMENT_PARAMETERS)
{
    nssl2_snow_sediment_impl<NSSL2_KMAX_GENERIC>(
        air_density, qs, qns, dz, snownc, snowncv, dt, nz, ny, nx);
}


// WRF v4.6.1 module_mp_nssl_2mom.F:4242-4734 and :6515-6632.
// Default option 18 uses predicted column-ice number, icefallopt=3's
// adjusted-Ferrier velocity, infall=4 adaptive upwind fallout, and the
// mass-weighted Method-II lower bound on the advected number moment.
template <int KMAX>
__device__ __forceinline__ void nssl2_ice_sediment_impl(
    const float* __restrict__ air_density,
    float* __restrict__ qi,
    float* __restrict__ qni,
    const float* __restrict__ dz,
    float* __restrict__ icenc,
    float* __restrict__ icencv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float ice[KMAX];
    float number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float number_mass_weighted[KMAX];

    const float minimum_mass = 6.88e-13f;
    const float maximum_mass = 1.0e-8f;
    const float gamma_1p18 = 0.922766923904419f;
    const float gamma_2p18 = 1.091937899589539f;
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        rho[k] = air_density[idx];
        ice[k] = qi[idx];
        number[k] = qni[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;

        const float positive_ice = fmaxf(ice[k], 0.0f);
        if (positive_ice > 1.0e-13f) {
            float local_number = fmaxf(number[k], 0.0f);
            local_number = fmaxf(
                local_number, __fdiv_rn(rho[k] * positive_ice, maximum_mass));
            local_number = fminf(
                local_number, __fdiv_rn(rho[k] * positive_ice, minimum_mass));
            const float particle_mass = fmaxf(
                rho[k] * positive_ice / local_number, minimum_mass);
            const float mean_volume = __fdiv_rn(particle_mass, 900.0f);
            const float density_factor = sqrtf(
                1.225f * fminf(20.0f, 1.0f / rho[k]));
            const float tmp = 47.6273f * density_factor
                / gfk_pow(1.0f / mean_volume, 0.18333f);
            number_velocity[k] = fminf(70.0f, tmp * gamma_1p18);
            mass_velocity[k] = fminf(70.0f, tmp * gamma_2p18);
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, mass_velocity[k] * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, number_velocity[k] * inverse_dz);
    }

    if (maximum_courant_rate == 0.0f) {
        icencv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            mass_flux[k] = ice[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
            number_mass_weighted[k] = number[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux += ice[0] * mass_velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            ice[k] += dt_substep * (inverse_dz * (1.0f / rho[k]))
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        for (int k = 0; k < nz; ++k) {
            number[k] = fmaxf(number[k], number_mass_weighted[k]);
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        qi[idx] = ice[k];
        qni[idx] = number[k];
    }
    const float exported = dt * rho[0] * surface_mean_flux;
    icencv[column] = exported;
    icenc[column] += exported;
}

template <bool ACCUMULATE = true>
__device__ __forceinline__ void nssl2_ice_sediment_parallel_impl(
    const float* __restrict__ air_density,
    float* __restrict__ qi,
    float* __restrict__ qni,
    const float* __restrict__ dz,
    float* __restrict__ icenc,
    float* __restrict__ icencv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x;
    const int k = threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho = 0.0f;
    float ice = 0.0f;
    float number = 0.0f;
    float mass_velocity = 0.0f;
    float number_velocity = 0.0f;
    float* shared_sediment = nssl2_sediment_shared_storage();
    float* mass_flux = shared_sediment + 0;
    float* number_flux = shared_sediment + 65;
    float* mass_number_flux = shared_sediment + 130;
    float number_mass_weighted = 0.0f;

    const float minimum_mass = 6.88e-13f;
    const float maximum_mass = 1.0e-8f;
    const float gamma_1p18 = 0.922766923904419f;
    const float gamma_2p18 = 1.091937899589539f;
    float maximum_courant_rate = 0.0f;

    if (k < nz) {
        const size_t idx = IDX3(k, j, i);
        rho = air_density[idx];
        ice = qi[idx];
        number = qni[idx];
        mass_velocity = 0.0f;
        number_velocity = 0.0f;

        const float positive_ice = fmaxf(ice, 0.0f);
        if (positive_ice > 1.0e-13f) {
            float local_number = fmaxf(number, 0.0f);
            local_number = fmaxf(
                local_number, __fdiv_rn(rho * positive_ice, maximum_mass));
            local_number = fminf(
                local_number, __fdiv_rn(rho * positive_ice, minimum_mass));
            const float particle_mass = fmaxf(
                rho * positive_ice / local_number, minimum_mass);
            const float mean_volume = __fdiv_rn(particle_mass, 900.0f);
            const float density_factor = sqrtf(
                1.225f * fminf(20.0f, 1.0f / rho));
            const float tmp = 47.6273f * density_factor
                / gfk_pow(1.0f / mean_volume, 0.18333f);
            number_velocity = fminf(70.0f, tmp * gamma_1p18);
            mass_velocity = fminf(70.0f, tmp * gamma_2p18);
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, mass_velocity * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, number_velocity * inverse_dz);
    }

    float* rates = shared_sediment + 325;
    rates[k] = maximum_courant_rate;
    __syncthreads();
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (k < stride) rates[k] = fmaxf(rates[k], rates[k + stride]);
        __syncthreads();
    }
    maximum_courant_rate = rates[0];
    if (maximum_courant_rate == 0.0f) {
        if (k == 0) icencv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        if (k < nz) {
            mass_flux[k] = ice * mass_velocity * rho;
            number_flux[k] = number * number_velocity;
            mass_number_flux[k] = number * mass_velocity;
            number_mass_weighted = number;
        }
        if (k == 0) {
            mass_flux[nz] = 0.0f;
            number_flux[nz] = 0.0f;
            mass_number_flux[nz] = 0.0f;
        }
        __syncthreads();

        if (k == 0) surface_mean_flux += ice * mass_velocity * dt_fraction;

        if (k < nz) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            ice += dt_substep * (inverse_dz * (1.0f / rho))
                * (mass_flux[k + 1] - mass_flux[k]);
            number += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            number_mass_weighted += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        if (k < nz) {
            number = fmaxf(number, number_mass_weighted);
        }
        __syncthreads();
    }

    if (k < nz) {
        const size_t idx = IDX3(k, j, i);
        qi[idx] = ice;
        qni[idx] = number;
    }
    if (k == 0) {
        const float exported = dt * rho * surface_mean_flux;
        icencv[column] = exported;
        if constexpr (ACCUMULATE) icenc[column] += exported;
    }
}

#define NSSL2_ICE_SEDIMENT_PARAMETERS                                  \
    const float* __restrict__ air_density, float* __restrict__ qi,       \
    float* __restrict__ qni, const float* __restrict__ dz,               \
    float* __restrict__ icenc, float* __restrict__ icencv,               \
    float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_ice_sediment_64(
    NSSL2_ICE_SEDIMENT_PARAMETERS)
{
    nssl2_ice_sediment_impl<NSSL2_KMAX_SHALLOW>(
        air_density, qi, qni, dz, icenc, icencv, dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_ice_sediment_256(
    NSSL2_ICE_SEDIMENT_PARAMETERS)
{
    nssl2_ice_sediment_impl<NSSL2_KMAX_GENERIC>(
        air_density, qi, qni, dz, icenc, icencv, dt, nz, ny, nx);
}


// WRF v4.6.1 module_mp_nssl_2mom.F:4242-5163 and :6799-7554.
// Default option 18 uses predicted graupel number and volume, variable
// 170--900-kg/m3 particle density, Milbrandt--Morrison (2013) terminal
// velocities (icdx=6), infall=4 adaptive upwind fallout, and the combined
// reflectivity/mass-weighted number correction.  The gamma lookup below
// reproduces WRF's 0.01-spaced, linearly interpolated double-precision table
// before the result is rounded back to REAL.
__device__ __forceinline__ float nssl2_gamma_lookup(float argument)
{
    return nssl2_gmoi_lookup(argument);
}

// Each block reuses WRF's gmoi values for the velocity indices.
__device__ __forceinline__ float nssl2_gamma_lookup_shared(
    float argument, const double* table, int shape)
{
    const int lower_index = nssl2_gmoi_index(argument);
    const int group = (lower_index - 150 - 100 * shape) / 300;
    const int slot = group * 20 + lower_index - (150 + 100 * shape + group * 300);
    return nssl2_gmoi_blend(argument, table[2 * slot], table[2 * slot + 1]);
}

template <bool CACHED>
__device__ __forceinline__ float nssl2_velocity_gamma_lookup(
    float argument, const double* table, int shape)
{
    if constexpr (CACHED) {
        return nssl2_gamma_lookup_shared(argument, table, shape);
    } else {
        return nssl2_gamma_lookup(argument);
    }
}

// calczgr1d / calcnfromz1d particle density for graupel and hail: the
// volume moment when it is positive, clamped to [hdnmn, 900] (hdnmn = 170
// for both categories there), else xdn0 (rho_qh = 500, rho_qhl = 900).
__device__ __forceinline__ float nssl2_dense_frozen_z_density(
    float q, float volume, float rho, bool hail)
{
    if (volume > 0.0f) {
        return fminf(900.0f, fmaxf(170.0f, rho * q / volume));
    }
    return hail ? 900.0f : 500.0f;
}

__device__ __forceinline__ float nssl2_dense_frozen_z(
    float q, float number, float volume, float rho, bool hail)
{
    if (!(q > 1.0e-12f) || !(number > 1.0e-15f)) return 0.0f;
    return nssl2_sediment_z(
        q, number, rho, nssl2_dense_frozen_z_density(q, volume, rho, hail),
        hail ? 8.75f : 20.0f,
        hail ? NSSL2_WRF_XVHLMN : NSSL2_WRF_XVHMN,
        hail ? NSSL2_WRF_XVHLMX : NSSL2_WRF_XVHMX0);
}

__device__ __forceinline__ float nssl2_dense_frozen_number_from_z(
    float q, float number, float volume, float rho, bool hail,
    float z_advected, float z_initial, float number_mass_weighted)
{
    return nssl2_sediment_number_from_z(
        q, number, rho, nssl2_dense_frozen_z_density(q, volume, rho, hail),
        hail ? 8.75f : 20.0f,
        hail ? NSSL2_WRF_XVHLMN : NSSL2_WRF_XVHMN,
        hail ? NSSL2_WRF_XVHLMX : NSSL2_WRF_XVHMX0,
        z_advected, z_initial, number_mass_weighted);
}

__device__ __forceinline__ void nssl2_graupel_mm_coefficients(
    float particle_density, float* coefficient, float* exponent)
{
    const float table_density[9] = {
        50.0f, 150.0f, 250.0f, 350.0f, 450.0f,
        550.0f, 650.0f, 750.0f, 850.0f};
    const float table_coefficient[9] = {
        62.923f, 94.122f, 114.74f, 131.21f, 145.26f,
        157.71f, 168.98f, 179.36f, 189.02f};
    const float table_exponent[9] = {
        0.67819f, 0.63789f, 0.62197f, 0.61240f, 0.60572f,
        0.60066f, 0.59663f, 0.59330f, 0.59048f};

    int index = (int)(__fdiv_rn((particle_density - 50.0f), 100.0f));
    index = max(0, min(8, index));
    if (index < 8) {
        const float fraction = fmaxf(
            0.0f, 0.01f * (particle_density - table_density[index]));
        *coefficient = table_coefficient[index]
            + fraction * (table_coefficient[index + 1]
                          - table_coefficient[index]);
        *exponent = table_exponent[index]
            + fraction * (table_exponent[index + 1]
                          - table_exponent[index]);
    } else {
        *coefficient = table_coefficient[index];
        *exponent = table_exponent[index];
    }
}

// ziegfall1d (module_mp_nssl_2mom.F:7595) + setvtz (:6211) for graupel
// (icdx = 6) and hail (icdxhl = 6), ipconc = 5, infdo = 2.
template <bool HAIL, bool CACHED>
__device__ __forceinline__ void nssl2_dense_frozen_velocities(
    float q_raw, float number, float volume, float rho,
    const double* cached_gamma, float* vm, float* vn, float* vz)
{
    *vm = 0.0f;
    *vn = 0.0f;
    *vz = 0.0f;
    const float q = fmaxf(q_raw, 0.0f);
    if (!(q > 1.0e-12f)) return;
    float xdn = HAIL ? 900.0f : 500.0f;
    const float vx = fmaxf(volume, 0.0f);
    if (vx > (rho * 1.0e-12f) * 1.0e-3f) {
        xdn = fminf(900.0f, fmaxf(HAIL ? 500.0f : 170.0f, rho * q / vx));
    }
    const float xvmn = HAIL ? NSSL2_WRF_XVHLMN : NSSL2_WRF_XVHMN;
    const float xvmx = HAIL ? NSSL2_WRF_XVHLMX : NSSL2_WRF_XVHMX0;
    float xv = rho * q / (xdn * fmaxf(1.0e-9f, number));
    if (xv < xvmn || xv > xvmx) xv = fminf(xvmx, fmaxf(xvmn, xv));
    // graupel: (xv*6.*pii)**(1./3.); hail: (xv*6./pi)**(1./3.)
    const float mass_diameter = HAIL
        ? gfk_pow(__fdiv_rn(xv * 6.0f, NSSL2_WRF_PI), 1.0f / 3.0f)
        : gfk_pow((xv * 6.0f) * NSSL2_WRF_PIINV, 1.0f / 3.0f);
    const float alpha = HAIL ? 1.0f : 0.0f;
    const float cwch = gfk_pow(
        (3.0f + alpha) * (2.0f + alpha) * (1.0f + alpha), -1.0f / 3.0f);
    const float xdia = cwch * mass_diameter;
    float aax;
    float bbx;
    nssl2_graupel_mm_coefficients(xdn, &aax, &bbx);
    const float rhovt = sqrtf(1.225f * fminf(20.0f, 1.0f / rho));
    const float power = gfk_pow(xdia, bbx);
    const int shape = HAIL ? 1 : 0;
    float v1 = rhovt * aax * power
        * nssl2_velocity_gamma_lookup<CACHED>(4.0f + alpha + bbx, cached_gamma, shape)
        / nssl2_gamma_lookup(4.0f + alpha);
    float v2 = rhovt * aax * power
        * nssl2_velocity_gamma_lookup<CACHED>(1.0f + alpha + bbx, cached_gamma, shape)
        / nssl2_gamma_lookup(1.0f + alpha);
    float v3 = rhovt * (aax * power
        * nssl2_velocity_gamma_lookup<CACHED>(7.0f + alpha + bbx, cached_gamma, shape))
        / nssl2_gamma_lookup(7.0f + alpha);
    if (v2 > v1 || (v1 > v3 && v3 > 0.0f)) {
        v1 = fmaxf(v1, v2);
        v3 = fmaxf(v3, v1);
    }
    *vm = fminf(70.0f, fminf(150.0f, v1));
    *vn = fminf(70.0f, fminf(150.0f, v2));
    *vz = fminf(70.0f, fminf(150.0f, v3));
}

template <int KMAX, bool HAIL, bool CACHED = false>
__device__ __forceinline__ void nssl2_dense_frozen_sediment_impl(
    const float* __restrict__ air_density,
    float* __restrict__ qx,
    float* __restrict__ qnx,
    float* __restrict__ qvolx,
    const float* __restrict__ dz,
    float* __restrict__ frozennc,
    float* __restrict__ frozenncv,
    float dt, int nz, int ny, int nx, const double* cached_gamma = nullptr)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float graupel[KMAX];
    float number[KMAX];
    float volume[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float z_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float volume_flux[KMAX + 1];
    float z_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float z_initial[KMAX];
    float z_advected[KMAX];
    float number_mass_weighted[KMAX];

    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        rho[k] = air_density[idx];
        graupel[k] = qx[idx];
        number[k] = qnx[idx];
        volume[k] = qvolx[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
        z_velocity[k] = 0.0f;

        nssl2_dense_frozen_velocities<HAIL, CACHED>(
            graupel[k], number[k], volume[k], rho[k], cached_gamma,
            &mass_velocity[k], &number_velocity[k], &z_velocity[k]);
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, mass_velocity[k] * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, number_velocity[k] * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, z_velocity[k] * inverse_dz);
    }

    if (maximum_courant_rate == 0.0f) {
        frozenncv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(
            2, (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            z_initial[k] = nssl2_dense_frozen_z(
                graupel[k], number[k], volume[k], rho[k], HAIL);
            z_advected[k] = z_initial[k];
            number_mass_weighted[k] = number[k];
            mass_flux[k] = graupel[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            volume_flux[k] = volume[k] * mass_velocity[k];
            z_flux[k] = z_initial[k] * z_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        volume_flux[nz] = 0.0f;
        z_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux +=
            graupel[0] * mass_velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            graupel[k] += dt_substep * (inverse_dz * (1.0f / rho[k]))
                * (mass_flux[k + 1] - mass_flux[k]);
            volume[k] += dt_substep * inverse_dz
                * (volume_flux[k + 1] - volume_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            z_advected[k] += dt_substep * inverse_dz
                * (z_flux[k + 1] - z_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        for (int k = 0; k < nz; ++k) {
            if (z_advected[k] > 0.0f) {
                number[k] = nssl2_dense_frozen_number_from_z(
                    graupel[k], number[k], volume[k], rho[k], HAIL,
                    z_advected[k], z_initial[k], number_mass_weighted[k]);
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        qx[idx] = graupel[k];
        qnx[idx] = number[k];
        qvolx[idx] = volume[k];
    }
    // WRF's surface reducer (module_mp_nssl_2mom.F:3008-3061) sums the four
    // categories' xfall before it multiplies by dtp*dn1, so this row carries
    // the raw xfall and nssl2_driver_reduce_precipitation applies WRF's
    // formula; a per-category dt*rho product cannot be summed back exactly.
    const float exported = surface_mean_flux;
    frozenncv[column] = exported;
    frozennc[column] += exported;
}

template <bool HAIL, bool ACCUMULATE = true>
__device__ __forceinline__ void nssl2_dense_frozen_sediment_parallel_cached(
    const float* __restrict__ air_density,
    float* __restrict__ qx,
    float* __restrict__ qnx,
    float* __restrict__ qvolx,
    const float* __restrict__ dz,
    float* __restrict__ frozennc,
    float* __restrict__ frozenncv,
    float dt, int nz, int ny, int nx, const double* cached_gamma)
{
    const int column = blockIdx.x;
    const int k = threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho = 0.0f;
    float graupel = 0.0f;
    float number = 0.0f;
    float volume = 0.0f;
    float mass_velocity = 0.0f;
    float number_velocity = 0.0f;
    float z_velocity = 0.0f;
    float* shared_sediment = nssl2_sediment_shared_storage();
    float* mass_flux = shared_sediment + 0;
    float* number_flux = shared_sediment + 65;
    float* volume_flux = shared_sediment + 130;
    float* z_flux = shared_sediment + 195;
    float* mass_number_flux = shared_sediment + 260;
    float z_initial = 0.0f;
    float z_advected = 0.0f;
    float number_mass_weighted = 0.0f;

    float maximum_courant_rate = 0.0f;

    if (k < nz) {
        const size_t idx = IDX3(k, j, i);
        rho = air_density[idx];
        graupel = qx[idx];
        number = qnx[idx];
        volume = qvolx[idx];
        mass_velocity = 0.0f;
        number_velocity = 0.0f;
        z_velocity = 0.0f;

        nssl2_dense_frozen_velocities<HAIL, true>(
            graupel, number, volume, rho, cached_gamma,
            &mass_velocity, &number_velocity, &z_velocity);
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, mass_velocity * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, number_velocity * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, z_velocity * inverse_dz);
    }

    float* rates = shared_sediment + 325;
    rates[k] = maximum_courant_rate;
    __syncthreads();
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (k < stride) rates[k] = fmaxf(rates[k], rates[k + stride]);
        __syncthreads();
    }
    maximum_courant_rate = rates[0];
    if (maximum_courant_rate == 0.0f) {
        if (k == 0) frozenncv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(
            2, (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        if (k < nz) {
            z_initial = nssl2_dense_frozen_z(
                graupel, number, volume, rho, HAIL);
            z_advected = z_initial;
            number_mass_weighted = number;
            mass_flux[k] = graupel * mass_velocity * rho;
            number_flux[k] = number * number_velocity;
            volume_flux[k] = volume * mass_velocity;
            z_flux[k] = z_initial * z_velocity;
            mass_number_flux[k] = number * mass_velocity;
        }
        if (k == 0) {
            mass_flux[nz] = 0.0f;
            number_flux[nz] = 0.0f;
            volume_flux[nz] = 0.0f;
            z_flux[nz] = 0.0f;
            mass_number_flux[nz] = 0.0f;
        }
        __syncthreads();

        if (k == 0) surface_mean_flux +=
            graupel * mass_velocity * dt_fraction;

        if (k < nz) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            graupel += dt_substep * (inverse_dz * (1.0f / rho))
                * (mass_flux[k + 1] - mass_flux[k]);
            volume += dt_substep * inverse_dz
                * (volume_flux[k + 1] - volume_flux[k]);
            number += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            z_advected += dt_substep * inverse_dz
                * (z_flux[k + 1] - z_flux[k]);
            number_mass_weighted += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        if (k < nz) {
            if (z_advected > 0.0f) {
                number = nssl2_dense_frozen_number_from_z(
                    graupel, number, volume, rho, HAIL,
                    z_advected, z_initial, number_mass_weighted);
            }
        }
        __syncthreads();
    }

    if (k < nz) {
        const size_t idx = IDX3(k, j, i);
        qx[idx] = graupel;
        qnx[idx] = number;
        qvolx[idx] = volume;
    }
    if (k == 0) {
        // Raw xfall: see the reducer (WRF sums xfall before dtp*dn1).
        const float exported = surface_mean_flux;
        frozenncv[column] = exported;
        if constexpr (ACCUMULATE) frozennc[column] += exported;
    }
}

#define NSSL2_GRAUPEL_SEDIMENT_PARAMETERS                              \
    const float* __restrict__ air_density, float* __restrict__ qg,      \
    float* __restrict__ qng, float* __restrict__ qvolg,                 \
    const float* __restrict__ dz, float* __restrict__ graupelnc,        \
    float* __restrict__ graupelncv, float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_graupel_sediment_64(
    NSSL2_GRAUPEL_SEDIMENT_PARAMETERS)
{
    nssl2_dense_frozen_sediment_impl<NSSL2_KMAX_SHALLOW, false>(
        air_density, qg, qng, qvolg, dz, graupelnc, graupelncv,
        dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_graupel_sediment_256(
    NSSL2_GRAUPEL_SEDIMENT_PARAMETERS)
{
    nssl2_dense_frozen_sediment_impl<NSSL2_KMAX_GENERIC, false>(
        air_density, qg, qng, qvolg, dz, graupelnc, graupelncv,
        dt, nz, ny, nx);
}

#define NSSL2_HAIL_SEDIMENT_PARAMETERS                                  \
    const float* __restrict__ air_density, float* __restrict__ qh,      \
    float* __restrict__ qnh, float* __restrict__ qvolh,                 \
    const float* __restrict__ dz, float* __restrict__ hailnc,           \
    float* __restrict__ hailncv, float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_hail_sediment_64(
    NSSL2_HAIL_SEDIMENT_PARAMETERS)
{
    nssl2_dense_frozen_sediment_impl<NSSL2_KMAX_SHALLOW, true>(
        air_density, qh, qnh, qvolh, dz, hailnc, hailncv,
        dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_hail_sediment_256(
    NSSL2_HAIL_SEDIMENT_PARAMETERS)
{
    nssl2_dense_frozen_sediment_impl<NSSL2_KMAX_GENERIC, true>(
        air_density, qh, qnh, qvolh, dz, hailnc, hailncv,
        dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_fill_velocity_gamma(double* table)
{
    const int shape = blockIdx.x;
    const int slot = threadIdx.x;
    if (slot >= 60) return;
    const int index = 150 + 100 * shape + (slot / 20) * 300 + slot % 20;
    table[shape * 120 + 2 * slot] = nssl2_gmoi(index);
    table[shape * 120 + 2 * slot + 1] = nssl2_gmoi(index + 1);
}


extern "C" __global__ void nssl2_graupel_sediment_cached_256(
    NSSL2_GRAUPEL_SEDIMENT_PARAMETERS, const double* cached_gamma)
{
    nssl2_dense_frozen_sediment_impl<NSSL2_KMAX_GENERIC, false, true>(
        air_density, qg, qng, qvolg, dz, graupelnc, graupelncv,
        dt, nz, ny, nx, cached_gamma);
}


extern "C" __global__ void nssl2_hail_sediment_cached_256(
    NSSL2_HAIL_SEDIMENT_PARAMETERS, const double* cached_gamma)
{
    nssl2_dense_frozen_sediment_impl<NSSL2_KMAX_GENERIC, true, true>(
        air_density, qh, qnh, qvolh, dz, hailnc, hailncv,
        dt, nz, ny, nx, cached_gamma);
}

// Category rows own disjoint moments and exports. Cloud writes the final
// ignored export, so the other rows omit its unused intermediate additions.
extern "C" __global__ void nssl2_sediment_all_parallel_64(
    const float* air_density, const float* temperature_k,
    float* state, const float* dz, float* ignored_accumulator,
    float* category_export, const double* cached_gamma,
    float dt, int nz, int ny, int nx)
{
    const size_t n = (size_t)nz * ny * nx;
    const size_t ncol = (size_t)ny * nx;
    const int category = blockIdx.y;
    if (category == 0) {
        nssl2_rain_sediment_parallel_impl<false>(
            air_density, state + NSSL2_QR * n, state + NSSL2_NR * n,
            dz, ignored_accumulator, category_export, dt, nz, ny, nx);
    } else if (category == 1) {
        nssl2_ice_sediment_parallel_impl<false>(
            air_density, state + NSSL2_QI * n, state + NSSL2_NI * n,
            dz, ignored_accumulator, category_export + ncol, dt, nz, ny, nx);
    } else if (category == 2) {
        nssl2_snow_sediment_parallel_impl<false>(
            air_density, state + NSSL2_QS * n, state + NSSL2_NS * n,
            dz, ignored_accumulator, category_export + 2 * ncol, dt, nz, ny, nx);
    } else if (category == 3) {
        nssl2_dense_frozen_sediment_parallel_cached<false, false>(
            air_density, state + NSSL2_QG * n, state + NSSL2_NG * n,
            state + NSSL2_VG * n, dz, ignored_accumulator,
            category_export + 3 * ncol, dt, nz, ny, nx, cached_gamma);
    } else if (category == 4) {
        nssl2_dense_frozen_sediment_parallel_cached<true, false>(
            air_density, state + NSSL2_QH * n, state + NSSL2_NH * n,
            state + NSSL2_VH * n, dz, ignored_accumulator,
            category_export + 4 * ncol, dt, nz, ny, nx, cached_gamma + 120);
    } else {
        nssl2_cloud_sediment_parallel_impl(
            air_density, temperature_k, state + NSSL2_QC * n,
            state + NSSL2_NC * n, dz, ignored_accumulator, dt, nz, ny, nx);
    }
}

extern "C" __global__ void nssl2_driver_reduce_precipitation(
    const float* __restrict__ category_export,
    const float* __restrict__ air_density,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ snownc,
    float* __restrict__ snowncv,
    float* __restrict__ graupelnc,
    float* __restrict__ graupelncv,
    float* __restrict__ hailnc,
    float* __restrict__ hailncv,
    float* __restrict__ sr,
    float dt,
    int ncol)
{
    const int column = blockDim.x * blockIdx.x + threadIdx.x;
    if (column >= ncol) return;

    // module_mp_nssl_2mom.F:2804-2807 and :3008-3061 (one microphysics loop,
    // HAILNC present).  category_export rows are rain, cloud ice (kg/m2,
    // not reduced: WRF's reducer omits cloud-ice fallout), snow, graupel,
    // hail; the four reduced rows carry WRF's surface xfall.  xdn0(lr) is a
    // runtime REAL, so x*1000./xdn0(lr) is a multiply and a divide.  With
    // hail off the hail row is +0 and every hail term vanishes exactly, as
    // WRF's lhl < 1 branch and its HAILNCV = 0 reset do.
    const float xr = category_export[column];
    const float xs = category_export[2 * ncol + column];
    const float xh = category_export[3 * ncol + column];
    const float xhl = category_export[4 * ncol + column];
    const float xdn0r = 1000.0f;
    const float dtrho = dt * air_density[column];   // dtp*dn1(ix,1,1)

    const float sum = ((xr + __fdiv_rn(xs * 1000.0f, xdn0r))
                       + __fdiv_rn(xh * 1000.0f, xdn0r))
                      + __fdiv_rn(xhl * 1000.0f, xdn0r);
    const float rain_v = 0.0f + dtrho * sum;
    const float snow_v = 0.0f + __fdiv_rn((dtrho * xs) * 1000.0f, xdn0r);
    const float graupel_v = 0.0f + __fdiv_rn((dtrho * xh) * 1000.0f, xdn0r);
    const float hail_v = __fdiv_rn((dtrho * xhl) * 1000.0f, xdn0r);

    rainncv[column] = rain_v;
    snowncv[column] = snow_v;
    graupelncv[column] = graupel_v;
    hailncv[column] = hail_v;
    rainnc[column] = rainnc[column] + rain_v;
    snownc[column] = snownc[column] + snow_v;
    hailnc[column] = hailnc[column] + hail_v;
    graupelnc[column] = graupelnc[column] + graupel_v;
    sr[column] = __fdiv_rn((snow_v + hail_v) + graupel_v, rain_v + 1.e-12f);
}

extern "C" __global__ void nssl2_driver_scatter(
    const float* __restrict__ air_density,
    const float* __restrict__ state,
    float* __restrict__ qv,
    float* __restrict__ qc,
    float* __restrict__ qr,
    float* __restrict__ qi,
    float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ qh,
    float* __restrict__ qndrop,
    float* __restrict__ qnr,
    float* __restrict__ qni,
    float* __restrict__ qns,
    float* __restrict__ qng,
    float* __restrict__ qnh,
    float* __restrict__ qnn,
    float* __restrict__ qvolg,
    float* __restrict__ qvolh,
    int n,
    int predicted_ccn)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    qv[idx] = state[(size_t)NSSL2_QV * n + idx];
    qc[idx] = state[(size_t)NSSL2_QC * n + idx];
    qr[idx] = state[(size_t)NSSL2_QR * n + idx];
    qi[idx] = state[(size_t)NSSL2_QI * n + idx];
    qs[idx] = state[(size_t)NSSL2_QS * n + idx];
    qg[idx] = state[(size_t)NSSL2_QG * n + idx];
    qh[idx] = state[(size_t)NSSL2_QH * n + idx];
    qndrop[idx] = state[(size_t)NSSL2_NC * n + idx] / air_density[idx];
    qnr[idx] = state[(size_t)NSSL2_NR * n + idx] / air_density[idx];
    qni[idx] = state[(size_t)NSSL2_NI * n + idx] / air_density[idx];
    qns[idx] = state[(size_t)NSSL2_NS * n + idx] / air_density[idx];
    qng[idx] = state[(size_t)NSSL2_NG * n + idx] / air_density[idx];
    qnh[idx] = state[(size_t)NSSL2_NH * n + idx] / air_density[idx];
    // module_mp_nssl_2mom.F:3283 writes the CCN field back only when
    // flag_ccn is set.  With nssl_ccn_on=0 the Registry never allocated
    // qnn, WRF's store is skipped, and the diagnostic value recomputed at
    // the next load is the only CCN the scheme ever sees.  Leaving the
    // caller's array untouched is what "the field does not exist" means at
    // this seam; the resolved mode also pins it in pinned_zero_fields().
    if (predicted_ccn) {
        qnn[idx] = state[(size_t)NSSL2_NN * n + idx] / air_density[idx];
    }
    qvolg[idx] = state[(size_t)NSSL2_VG * n + idx] / air_density[idx];
    qvolh[idx] = state[(size_t)NSSL2_VH * n + idx] / air_density[idx];
}
