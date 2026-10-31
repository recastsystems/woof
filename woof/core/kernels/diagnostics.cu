// gpuwm/core/kernels/diagnostics.cu
//
// Equation-of-state diagnostics (WRF v4.6.1 calc_p_rho_phi analogue,
// dyn_em/module_big_step_utilities_em.F:1025-1052 NONHYDROSTATIC branch,
// keyed on hypsometric_opt; general hybrid/terrain form).
// One thread per (i, j) column; loops over k recomputing, from the
// prognostic perturbation fields:
//   alt  total dry specific volume.
//        hypso = 1 (frozen gpuwm default): from the geopotential and the
//        hybrid column-mass increment:
//          alt[k] = -((ph[k+1] - ph[k]) * rdnw[k]) / (c1h[k]*mu + c2h[k])
//        (rdnw < 0 => alt > 0; c1h = 1, c2h = 0 reduces bitwise to the
//        Phase 1 form -dph*rdnw/mu).
//        hypso = 2 (WRF Registry default, F:1042-1051 verbatim): the
//        log-pressure hypsometric form on the reference dry pressures
//          pfu = c3f[k+1]*mu + c4f[k+1] + p_top
//          pfd = c3f[k  ]*mu + c4f[k  ] + p_top
//          phm = c3h[k  ]*mu + c4h[k  ] + p_top
//          al[k]  = dph/phm/log(pfd/pfu) - alb[k]
//          alt[k] = al[k] + alb[k]
//        with mu = mub + mu' the total dry column mass (WRF MUTS).
//
// TWO CANCELLATIONS, and the FP32 spelling that removes them.  Both
// differences above are formed from numbers far larger than themselves,
// so in FP32 the diagnosed alt degrades as 1/dz -- the direction every
// LES configuration moves.  Measured against the float64 mirror on a
// random 1 K state (relative error in p): opt 1, 2.1e-6 at nz=16 and
// 2.9e-5 at nz=160; opt 2, 4.3e-6 and 1.2e-4.
//
//   1. dph.  The layer thickness is ~368 J/kg over a 2400 m column at
//      nz=64 while the total geopotentials are ~2.4e4 J/kg, so summing
//      phb+php first spends ulp(2.4e4) on it.  This kernel differences
//      the base and the perturbation SEPARATELY -- both subtractions are
//      exact -- and adds dphb_resid[k], the float64 base thickness minus
//      that float32 base subtraction (gpuwm/core/state.py
//      set_base_geopotential).  The correction is a property of the
//      profile the kernel is reading, so a stale one is worth at most one
//      ulp of phb, never a wrong layer.
//   2. log(pfd/pfu), opt 2 only.  pfd and pfu are independently rounded
//      at ~1e5 Pa and their ratio is 1 + O(dz/H), so WRF's spelling puts
//      ulp(1e5) into a ~470 Pa difference: 1.7e-5 relative, which is the
//      DOMINANT error on the production path and is untouched by fixing
//      dph.  This kernel writes the identical quantity as
//      log1p((pfd - pfu)/pfu) with pfd - pfu = dc3f[k]*mu + dc4f[k] --
//      p_top cancels identically -- from float64-differenced
//      coefficients.  Measured at nz=160: 1.15e-4 -> 5.5e-7 in p
//      (4 seeds, 4.98e-7 to 5.50e-7).
//      This is a deliberate divergence from WRF's text (F:1046), of the
//      never-bit-exact-to-a-bug kind: same quantity, one spelling that
//      cancels and one that does not.  The dycore parity target is
//      MPAS-A in the hex port, not WRF here.
//   p    full pressure via the ideal-gas EOS on the moist potential
//        temperature theta_m = theta*(1 + Rv/Rd*qv) (ARW ch. 2; theta_m
//        reduces to theta bitwise when moist = 0 or qv = 0):
//          p = P0 * ((RD * th_m) / (P0 * alt))^GAMMA
//        (for hypso = 2 alt = al + alb, exactly WRF's moist_nonhydro
//        temp = r_d*t*qvf/(p0*(al+alb)), F:1061-1066, use_theta_m = 0).
//   al   perturbation specific volume alt - alb.
//
// Base-state profiles are 1-D columns (base3d = 0, flat terrain) or full
// per-column 3-D fields (base3d = 1, terrain); mub is always the (ny, nx)
// dry-mass field (a broadcast scalar when flat).  qv is read only when
// moist = 1 (dry launches pass a dummy pointer).  c3h/c4h/c3f/c4f and
// p_top are read only when hypso = 2 (opt-1 launches may pass zeros).
//
// (j0, i0, nyw, nxw) restrict the launch to a column WINDOW.  The kernel
// is column-local -- each thread reads and writes only its own (j, i) --
// so a windowed launch is bitwise the full launch on the window and a
// no-op off it.  The full-domain call passes (0, 0, ny, nx), for which
// the (j, i) decode below is arithmetic-identical to the pre-window
// `j = col / nx` form.  The consumer is the two-way-feedback finalize,
// which re-diagnoses only the parent columns the restriction touched.

// Prepared inputs use the same pinned libm words as the host path. The
// dedicated loader also preserves separate FP32 operations. Forecast and
// WRF-exact modules retain their original arithmetic.
#ifdef GPUWM_PREPARATION_EOS
#define EOS_LOG1P plm_log1pf
#define EOS_POW gfk_pow
#else
#define EOS_LOG1P log1pf
#define EOS_POW powf
#endif

#ifdef GPUWM_WRF_EXACT_D_DIAGNOSTICS
// WRF calc_p_rho_phi for one column, with WRF's grid%muts as an argument.
// calc_p_alpha passes mub+mu (start_em's spelling, and WRF's mut); after an
// acoustic loop WRF's muts is advance_mu_t's MUTS = MUT + MU'', which
// calc_p_alpha_carried_muts reads from the carrier the dycore keeps
// (combo-sweep round 3: 164 al words at the second RK stage).
static __device__ void wrf_p_rho_phi_column(
    const real* __restrict__ thp, const real* __restrict__ php,
    const real* __restrict__ mup, const real* __restrict__ phb,
    const real* __restrict__ alb, const real* __restrict__ pb,
    const real* __restrict__ rdnw, const real* __restrict__ c1h,
    const real* __restrict__ c2h, const real* __restrict__ c3h,
    const real* __restrict__ c4h, const real* __restrict__ c3f,
    const real* __restrict__ c4f, const real* __restrict__ qv,
    real p_top, int hypso, int moist, size_t kstr, size_t coff,
    int nz, int ny, int nx, int j, int i, const real muts,
    real* __restrict__ p, real* __restrict__ al, real* __restrict__ alt,
    real* __restrict__ p_perturbation)
{
    for (int k = 0; k < nz; ++k) {
        // Verification-only WRF v4.7.1 calc_p_rho_phi operation order.
        // This substage reads canonical t = theta - 300 from thp. Native
        // exact initialization keeps thb at 300; other initialization doors
        // must supply that same carrier before selecting this substage.
        // gfk_log/gfk_pow come from the conditional glibc_flt32 header.
        const size_t h = IDX3(k, j, i);
        const size_t b = (size_t)k * kstr + coff;
        const real dphp = __fsub_rn(php[IDX3(k + 1, j, i)], php[h]);
        real ap;
        if (hypso == 2) {
            const real pfu = __fadd_rn(
                __fadd_rn(__fmul_rn(c3f[k + 1], muts), c4f[k + 1]), p_top);
            const real pfd = __fadd_rn(
                __fadd_rn(__fmul_rn(c3f[k], muts), c4f[k]), p_top);
            const real phm = __fadd_rn(
                __fadd_rn(__fmul_rn(c3h[k], muts), c4h[k]), p_top);
            const real dph = __fsub_rn(
                __fadd_rn(dphp, phb[(size_t)(k + 1) * kstr + coff]), phb[b]);
            ap = __fsub_rn(
                __fdiv_rn(__fdiv_rn(dph, phm),
                          gfk_log(__fdiv_rn(pfd, pfu))), alb[b]);
        } else {
            const real inverse_mass = __fdiv_rn(
                -1.0f, __fadd_rn(__fmul_rn(c1h[k], muts), c2h[k]));
            const real base_mass = __fmul_rn(
                alb[b], __fmul_rn(c1h[k], mup[(size_t)j * nx + i]));
            ap = __fmul_rn(inverse_mass,
                __fadd_rn(base_mass, __fmul_rn(rdnw[k], dphp)));
        }
        const real a = __fadd_rn(ap, alb[b]);
        const real theta = __fadd_rn(300.0f, thp[h]);
        real numerator = __fmul_rn(RD, theta);
        if (moist) {
            const real qvf = __fadd_rn(1.0f, __fmul_rn(RVOVRD, qv[h]));
            numerator = __fmul_rn(numerator, qvf);
        }
        const real temperature_ratio = __fdiv_rn(numerator, __fmul_rn(P0, a));
        const real perturbation_pressure = __fsub_rn(
            __fmul_rn(gfk_pow(temperature_ratio, GAMMA), P0), pb[b]);
        alt[h] = a;
        al[h] = ap;
        p_perturbation[h] = perturbation_pressure;
        p[h] = __fadd_rn(perturbation_pressure, pb[b]);
    }
}

#endif
extern "C" __global__
void calc_p_alpha(const real* __restrict__ thp,   // (nz,   ny, nx) theta'
                  const real* __restrict__ php,   // (nz+1, ny, nx) phi'
                  const real* __restrict__ mup,   // (ny, nx)       mu'
                  const real* __restrict__ thb,   // (nz[,ny,nx])   base theta
                  const real* __restrict__ phb,   // (nz+1[,ny,nx]) base geopot.
                  const real* __restrict__ dphbr, // (nz[,ny,nx]) base-thickness
                                                  //   float64-minus-float32
                                                  //   residual
                  const real* __restrict__ alb,   // (nz[,ny,nx])   base alpha
#ifdef GPUWM_WRF_EXACT_D_DIAGNOSTICS
                  const real* __restrict__ pb,    // WRF pressure subtraction carrier
#endif
                  const real* __restrict__ rdnw,  // (nz,)  1/dnw (< 0)
                  const real* __restrict__ c1h,   // (nz,)  dB/deta
                  const real* __restrict__ c2h,   // (nz,)  (1-c1h)(p0-pt)
                  const real* __restrict__ c3h,   // (nz,)  B(eta) half levels
                  const real* __restrict__ c4h,   // (nz,)  (eta-B)(p0-pt)
                  const real* __restrict__ c3f,   // (nz+1,) full-level c3
                  const real* __restrict__ c4f,   // (nz+1,) full-level c4
                  const real* __restrict__ dc3f,  // (nz,) c3f[k] - c3f[k+1]
                  const real* __restrict__ dc4f,  // (nz,) c4f[k] - c4f[k+1]
                  const real* __restrict__ mub,   // (ny, nx) base column mass
                  const real* __restrict__ qv,    // (nz, ny, nx) vapor (moist)
                  real p_top, int hypso,
                  int moist, int base3d, int nz, int ny, int nx,
                  int j0, int i0, int nyw, int nxw,
                  real* __restrict__ p,           // (nz, ny, nx) full pressure
                  real* __restrict__ al,          // (nz, ny, nx) alpha'
#ifdef GPUWM_WRF_EXACT_D_DIAGNOSTICS
                  real* __restrict__ alt,         // (nz, ny, nx) total alpha
                  real* __restrict__ p_perturbation)
#else
                  real* __restrict__ alt)         // (nz, ny, nx) total alpha
#endif
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= nyw * nxw) return;
    int j = j0 + col / nxw;
    int i = i0 + (col - (col / nxw) * nxw);

    // Base-profile indexing: level stride and column offset collapse to
    // (1, 0) for flat 1-D columns.
    size_t kstr = base3d ? (size_t)ny * nx : 1;
    size_t coff = base3d ? (size_t)j * nx + i : 0;

#ifdef GPUWM_WRF_EXACT_D_DIAGNOSTICS
    wrf_p_rho_phi_column(thp, php, mup, phb, alb, pb, rdnw, c1h, c2h, c3h, c4h,
                         c3f, c4f, qv, p_top, hypso, moist, kstr, coff, nz,
                         ny, nx, j, i,
                         __fadd_rn(mub[(size_t)j * nx + i],
                                   mup[(size_t)j * nx + i]),
                         p, al, alt, p_perturbation);
#else
    real mu = mub[(size_t)j * nx + i] + mup[(size_t)j * nx + i];
    for (int k = 0; k < nz; ++k) {
        real th  = thb[k * kstr + coff] + thp[IDX3(k, j, i)];
        if (moist) th *= 1.0f + RVOVRD * qv[IDX3(k, j, i)];
        real dphb = (phb[(k + 1) * kstr + coff] - phb[k * kstr + coff])
                  + dphbr[k * kstr + coff];
        real dph = dphb
                 + (php[IDX3(k + 1, j, i)] - php[IDX3(k, j, i)]);
        real a, ap;
        if (hypso == 2) {
            real pfu = c3f[k + 1] * mu + c4f[k + 1] + p_top;
            real dpf = dc3f[k]    * mu + dc4f[k];   // = pfd - pfu exactly
            real phm = c3h[k]     * mu + c4h[k]     + p_top;
            ap = dph / phm / EOS_LOG1P(dpf / pfu) - alb[k * kstr + coff];
            a  = ap + alb[k * kstr + coff];
        } else {
            a  = -dph * rdnw[k] / (c1h[k] * mu + c2h[k]);
            ap = a - alb[k * kstr + coff];
        }
        alt[IDX3(k, j, i)] = a;
        al[IDX3(k, j, i)]  = ap;
        p[IDX3(k, j, i)]   = P0 * EOS_POW((RD * th) / (P0 * a), GAMMA);
    }
#endif
}
#ifdef GPUWM_WRF_EXACT_D_DIAGNOSTICS

// calc_p_alpha with WRF's grid%muts read from a carrier (ny, nx) instead of
// formed as mub+mu: the strict dycore's diagnosis after an acoustic loop.
extern "C" __global__
void calc_p_alpha_carried_muts(
    const real* __restrict__ thp, const real* __restrict__ php,
    const real* __restrict__ mup, const real* __restrict__ thb,
    const real* __restrict__ phb, const real* __restrict__ alb,
    const real* __restrict__ pb, const real* __restrict__ muts,
    const real* __restrict__ rdnw, const real* __restrict__ c1h,
    const real* __restrict__ c2h, const real* __restrict__ c3h,
    const real* __restrict__ c4h, const real* __restrict__ c3f,
    const real* __restrict__ c4f, const real* __restrict__ qv,
    real p_top, int hypso, int moist, int base3d, int nz, int ny, int nx,
    int j0, int i0, int nyw, int nxw,
    real* __restrict__ p, real* __restrict__ al, real* __restrict__ alt,
    real* __restrict__ p_perturbation)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= nyw * nxw) return;
    int j = j0 + col / nxw;
    int i = i0 + (col - (col / nxw) * nxw);
    size_t kstr = base3d ? (size_t)ny * nx : 1;
    size_t coff = base3d ? (size_t)j * nx + i : 0;
    wrf_p_rho_phi_column(thp, php, mup, phb, alb, pb, rdnw, c1h, c2h, c3h, c4h,
                         c3f, c4f, qv, p_top, hypso, moist, kstr, coff, nz,
                         ny, nx, j, i, muts[(size_t)j * nx + i],
                         p, al, alt, p_perturbation);
}
#endif
