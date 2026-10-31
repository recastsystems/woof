// Classic WRF v4.6.1 Thompson (mp_physics=8): a literal transcription of
// phys/module_mp_thompson.F (tag v4.6.1, d66e442f, SHA-256 fabf19e2...),
// mp_gt_driver :1205-1479, mp_thompson :1513-4082 (is_aerosol_aware and
// is_hail_aware false), calc_effectRad :5594-5699 and calc_refl10cm
// :5710-6028, with the numbers WRF's own thompson_init computes
// (thompson_classic_constants.cuh, dumped from the WRF build).
//
// Word-for-word discipline (graded against WRF's own Fortran by
// tools/thompson_real_column_parity/mp8_wrf_ulp.py):
// * every expression in gfortran's evaluation order and precision: REAL
//   operations in float, a DOUBLE PRECISION operand promotes the rest of
//   the expression from that point, assignment rounds; the optimized
//   GIMPLE of the oracle's own -O2 -fno-tree-vectorize build is the
//   reference for each expression's operation sequence;
// * Fortran x**r with a real exponent is the libm call gfortran makes
//   (powf/pow, glibc's: gfk_pow, glibc_pow), x**2.0 is x*x, an integer
//   exponent is the multiplication chain __builtin_powi expands to;
//   EXP/LOG/LOG10 are glibc's (gfk_exp, gfk_log, gfk_log10, glibc_exp,
//   glibc_log);
// * the unit compiles with --fmad=false (gpuwm/core/kernels/__init__.py
//   _NO_FMAD_MODULES): WRF's reference has no fused multiply-add;
// * WRF's structure: the source, condensation, evaporation and
//   sedimentation stages accumulate float tendencies and the state is
//   written once, X1d + Xten*DT, by the terminal apply (:3972-4078).
//
// * every division by a compile-time constant is __fdiv_rn/__ddiv_rn
//   (FDIV/DDIV): NVRTC compiles x / 273.15f as a multiply by the
//   reciprocal even under --prec-div=true, which is not the IEEE quotient
//   gfortran computes (measured on sm_120: 277 of the fixture's 2,058
//   temperatures).  A runtime divisor compiles to the IEEE division.
//
// What the card cannot reproduce: sm_120 flushes FP32 subnormal results
// (CuPy compiles with -ftz=true).  Those lanes are the only expected
// differences.

// Diagnostic readback on the device, compiled only when a measurement
// defines TC_DEVICE_RATES (the production source never does): the same
// HOST_RATE slots the host build publishes.
#if defined(__CUDACC_RTC__) && defined(TC_DEVICE_RATES)
__device__ double* tc_rate_buf;
__device__ long long tc_rate_cells;
__device__ int tc_rate_slots;
#define GPUWM_HOST_RATES 1
__device__ __forceinline__ void HOST_RATE(int slot, long long i, double v)
{
    if (tc_rate_buf != 0 && slot < tc_rate_slots && i < tc_rate_cells)
        tc_rate_buf[(long long)slot * tc_rate_cells + i] = v;
}
#endif

#define TC_R1 1.0e-12f
#define TC_R2 1.0e-6f
#define TC_EPS 1.0e-15f
#define TC_PI 3.1415926536f
#define TC_T0 273.15f
#define TC_HGFR 235.16f
#define TC_AM_R (TC_PI * 1000.0f / 6.0f)
#define TC_AM_I (TC_PI * 890.0f / 6.0f)
#define TC_AM_S 0.069f
#define TC_RHO_I 890.0f
#define TC_D0C 1.0e-6f
#define TC_D0R 50.0e-6f
#define TC_D0S 300.0e-6f
#define TC_NT_C 100.0e6f
#define TC_NT_C_MAX 1999.0e6f
#define TC_XM0I 1.0e-12f
#define TC_LSUB 2.834e6f
#define TC_LVAP0 2.5e6f
#define TC_LFUS (TC_LSUB - TC_LVAP0)
#define TC_OLFUS (1.0f / TC_LFUS)
#define TC_ORV (1.0f / 461.5f)
#define TC_R 287.04f
#define TC_CP 1004.0f
#define TC_RHO_NOT (101325.0f / (287.05f * 298.0f))
#define TC_C_CUBE 0.5f
#define TC_C_SQRD 0.15f
#define TC_AV_R 4854.0f
#define TC_FV_R 195.0f
#define TC_AV_S 40.0f
#define TC_BV_S 0.55f
#define TC_FV_S 100.0f
#define TC_AV_I 1493.9f
#define TC_AV_C 0.316946e8f
#define TC_EF_SI 0.05f
#define TC_EF_RI 0.95f
#define TC_TNO 5.0f
#define TC_ATO 0.304f
#define TC_MU_S 0.6357f
#define TC_KAP0 490.6f
#define TC_KAP1 17.46f
#define TC_LAM0 20.78f
#define TC_LAM1 3.29f
#define TC_IDX_BG1 5
#define TC_NBC 100
#define TC_NBR 100
#define TC_NBS 100

// Table dimensions (module_mp_thompson.F:231-252).
#define TC_NTB_C 37
#define TC_NTB_I 64
#define TC_NTB_R 37
#define TC_NTB_S 37
#define TC_NTB_G 37
#define TC_NTB_G1 37
#define TC_NTB_R1 37
#define TC_NTB_I1 55
#define TC_NTB_T 9
#define TC_NTB_IN 55

__device__ __constant__ float tc_sa[10] = {
    5.065339f, -0.062659f, -3.032362f, 0.029469f, -0.000285f,
    0.31255f, 0.000204f, 0.003199f, 0.0f, -0.015952f};
__device__ __constant__ float tc_sb[10] = {
    0.476221f, -0.015896f, 0.165977f, 0.007468f, -0.000141f,
    0.060366f, 0.000079f, 0.000594f, 0.0f, -0.003577f};
#define SA(i) (tc_sa[(i) - 1])
#define SB(i) (tc_sb[(i) - 1])

// rho_g (:77-78) and am_g = PI*rho_g/6 (:132-135); classic Thompson reads
// slot idx_bg1 = 5 (rho_g 400) everywhere and rho_g(1), rho_g(NRHG).
#define TC_RHO_G5 400.0f
#define TC_AM_G5 (TC_PI * TC_RHO_G5 / 6.0f)

// libgcc __powisf2: x**n for a runtime integer n (Fortran 10.**nn).
__device__ __forceinline__ float tc_powi(float x, int m)
{
    unsigned int n = m < 0 ? (unsigned int)(-m) : (unsigned int)m;
    float y = (n % 2u) ? x : 1.0f;
    while (n >>= 1) {
        x = x * x;
        if (n % 2u) y = y * x;
    }
    return m < 0 ? 1.0f / y : y;
}

// Fortran NINT on a float (__builtin_iroundf): half away from zero.
__device__ __forceinline__ int tc_nint(float x) { return (int)roundf(x); }
__device__ __forceinline__ int tc_nint_d(double x) { return (int)round(x); }

// :5383-5413 and :5418-5444.
__device__ __forceinline__ float tc_rslf(float p, float t)
{
    const float c0 = 0.611583699e3f, c1 = 0.444606896e2f,
        c2 = 0.143177157e1f, c3 = 0.264224321e-1f, c4 = 0.299291081e-3f,
        c5 = 0.203154182e-5f, c6 = 0.702620698e-8f, c7 = 0.379534310e-11f,
        c8 = -0.321582393e-13f;
    const float x = fmaxf(-80.0f, t - 273.16f);
    float esl = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5
        + x * (c6 + x * (c7 + x * c8)))))));
    esl = fminf(esl, p * 0.15f);
    return 0.622f * esl / (p - esl);
}

__device__ __forceinline__ float tc_rsif(float p, float t)
{
    const float c0 = 0.609868993e3f, c1 = 0.499320233e2f,
        c2 = 0.184672631e1f, c3 = 0.402737184e-1f, c4 = 0.565392987e-3f,
        c5 = 0.521693933e-5f, c6 = 0.307839583e-7f, c7 = 0.105785160e-9f,
        c8 = 0.161444444e-12f;
    const float x = fmaxf(-80.0f, t - 273.16f);
    float esi = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5
        + x * (c6 + x * (c7 + x * c8)))))));
    esi = fminf(esi, p * 0.15f);
    return 0.622f * esi / fmaxf(1.0e-4f, p - esi);
}

// :6040-6060 theta_e.
__device__ __forceinline__ float tc_theta_e(float pp, float tt, float w,
                                            float tlc)
{
    const float rr = w + 1.e-8f;
    const float power = 0.2854f * (1.0f - (0.28f * rr));
    const float xx = tt * gfk_pow(100000.0f / pp, power);
    const float p1 = (3.376f / tlc) - 0.00254f;
    const float p2 = (rr * 1000.0f) * (1.0f + 0.81f * rr);
    return xx * gfk_exp(p1 * p2);
}

// :6074-6085 t_lcl.
__device__ __forceinline__ float tc_t_lcl(float tt, float tttd)
{
    const float denom = (1.0f / (tttd - 56.0f))
        + FDIV(gfk_log(tt / tttd), 800.0f);
    return (1.0f / denom) + 56.0f;
}

// :6094-6107 t_dew.
__device__ __forceinline__ float tc_t_dew(float p, float w)
{
    const float rr = w + 1e-8f;
    const float es = p * rr / (.622f + rr);
    const float esln = gfk_log(es);
    return (35.86f * esln - 4947.2325f) / (esln - 23.6837f);
}

// :6116-6143 theta_wetb: real*8 data initialised from REAL literals.
__device__ __forceinline__ float tc_theta_wetb(float thetae)
{
    const double c0 = (double)-1.00922292e-10f, c1 = (double)-1.47945344e-8f,
        c2 = (double)-1.7303757e-6f, c3 = (double)-0.00012709f,
        c4 = (double)1.15849867e-6f, c5 = (double)-3.518296861e-9f,
        c6 = (double)3.5741522e-12f;
    const double d0 = (double)0.00000000f, d1 = (double)-3.5223513e-10f,
        d2 = (double)-5.7250807e-8f, d3 = (double)-5.83975422e-6f,
        d4 = (double)4.72445163e-8f, d5 = (double)-1.13402845e-10f,
        d6 = (double)8.729580402e-14f;
    const float x = fminf(475.0f, thetae);
    const double xd = (double)x;
    float answer;
    if (x <= 335.5f) {
        answer = (float)(c0 + xd * (c1 + xd * (c2 + xd * (c3 + xd * (c4
            + xd * (c5 + xd * c6))))));
    } else {
        answer = (float)(d0 + xd * (d1 + xd * (d2 + xd * (d3 + xd * (d4
            + xd * (d5 + xd * d6))))));
    }
    return answer + 273.15f;
}

// :6155-6187 compT_fr_The.
__device__ __forceinline__ float tc_compt_fr_the(float thelcl, float pres)
{
    float guess = (thelcl - 0.5f * gfk_pow(fmaxf(thelcl - 270.f, 0.f),
                                          1.05f))
        * gfk_pow(FDIV(pres, 100000.0f), .2f);
    const float epsilon = 0.01f;
    for (int iter = 1; iter <= 100; ++iter) {
        const float w1 = tc_rslf(pres, guess);
        const float w2 = tc_rslf(pres, guess + 1.f);
        const float tenu = tc_theta_e(pres, guess, w1, guess);
        const float tenup = tc_theta_e(pres, guess + 1.0f, w2, guess + 1.f);
        const float cor = (thelcl - tenu) / (tenup - tenu);
        guess = guess + cor;
        if ((cor < epsilon) && (-cor < epsilon)) return guess;
    }
    const float thwlcl = tc_theta_wetb(thelcl);
    return thwlcl * (gfk_pow(FDIV(pres, 100000.0f), 0.286f));
}

// The ten-term Field et al. (2005) polynomial in Fortran order
// (:2036-2127): sa(1) + sa(2)*tc0 + sa(3)*m + sa(4)*tc0*m + sa(5)*tc0*tc0
// + sa(6)*m*m + sa(7)*tc0*tc0*m + sa(8)*tc0*m*m + sa(9)*tc0*tc0*tc0
// + sa(10)*m*m*m, left to right.
__device__ __forceinline__ float tc_poly(const float* s, float tc0, float m)
{
    return s[0] + s[1] * tc0 + s[2] * m + s[3] * tc0 * m
        + s[4] * tc0 * tc0 + s[5] * m * m + s[6] * tc0 * tc0 * m
        + s[7] * tc0 * m * m + s[8] * tc0 * tc0 * tc0 + s[9] * m * m * m;
}

// Lookup-table decade index (:2263-2275 and its siblings): n is the first
// of nic-1..nic+1 with x/10**n in [1, 10); idx = INT(x/10**n) +
// 10*(n-n2) - (n-n2).  The search depends only on the comparison, so the
// starting guess (NINT(ALOG10(x))) is reproduced with glibc's log10f.
__device__ __forceinline__ int tc_decade(float x, int n2)
{
    const int nic = tc_nint(gfk_log10(x));
    int n = nic - 1;
    for (int nn = nic - 1; nn <= nic + 1; ++nn) {
        n = nn;
        const float q = x / tc_powi(10.0f, nn);
        if (q >= 1.0f && q < 10.0f) break;
    }
    return (int)(x / tc_powi(10.0f, n)) + 10 * (n - n2) - (n - n2);
}

// The same for a DOUBLE argument (N0_exp, :2325-2332): DLOG10 then the
// comparisons N0_exp/10.**nn in double (10.**nn is REAL, promoted).
__device__ __forceinline__ int tc_decade_d(double x, int n2)
{
    const int nic = tc_nint_d(log10(x));
    int n = nic - 1;
    for (int nn = nic - 1; nn <= nic + 1; ++nn) {
        n = nn;
        const double q = x / (double)tc_powi(10.0f, nn);
        if (q >= 1.0 && q < 10.0) break;
    }
    return (int)(x / (double)tc_powi(10.0f, n)) + 10 * (n - n2) - (n - n2);
}

// Flag bits carried per cell from the source kernel to the column kernel.
#define TC_LQC 1
#define TC_LQI 2
#define TC_LQR 4
#define TC_LQS 8
#define TC_LQG 16
#define TC_SML 32     // prr_sml(k) > 0 (:3722)
#define TC_DONE 64    // no_micro column: entry rewrite only
// WRF v4.6.1's rain-collecting-graupel read (:2524-2546, idx_bg1 = 5 on an
// extent-1 dimension) addressed past the end of the table allocation at this
// cell: undefined in WRF (heap contents).  The unit reads the slot the
// table holds there and marks the cell so a replay can count every such
// read (none may be claimed identical).
#define TC_RCG_PAST_END 128
// Words in one rain-graupel table: ntb_g1*ntb_g*dimNRHG*ntb_r1*ntb_r with
// dimNRHG = NRHG1 = 1 (:465, :607-615).
#define TC_RACG_WORDS ((size_t)37 * 37 * 37 * 37)
// WRF's subscript idx_bg(k) = idx_bg1 = 5 on that extent-1 dimension adds
// (5 - 1) strides of ntb_g1*ntb_g words.
#define TC_RACG_BG1_OFFSET ((size_t)4 * 37 * 37)

// ---------------------------------------------------------------------
// Kernel 1 (one thread per cell): mp_gt_driver's entry (:1221-1283) for
// the classic path and mp_thompson's per-level entry values needed by the
// column flags.  Writes t1d, the private graupel number ng1d (:1267-1282)
// and, per column, no_micro (:1827-1990) and k_melting (:1971-1983)
// through atomics that are order-independent (OR and MAX).
// ---------------------------------------------------------------------
extern "C" __global__ void tc_entry(
    const float* __restrict__ thb, const float* __restrict__ thp,
    const float* __restrict__ phb, const float* __restrict__ php,
    const float* __restrict__ p,
    const float* __restrict__ qv, const float* __restrict__ qc,
    const float* __restrict__ qi, const float* __restrict__ qr,
    const float* __restrict__ qs, const float* __restrict__ qg,
    float* __restrict__ th_save, float* __restrict__ pii,
    float* __restrict__ t1d, float* __restrict__ dz,
    float* __restrict__ ng1d,
    int* __restrict__ column_micro, int* __restrict__ column_kmelt,
    float p0, float rcp, float gravity,
    int size, int columns, int thb_full, int phb_full, int member_columns)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const int k = idx / columns;
    const int col = idx - k * columns;
    // A full base state is one (nz, members*ny, nx) field, or one shared
    // (nz, ny, nx) slab for every ensemble member (member_columns = ny*nx;
    // gpuwm/ensemble/batch_physics.py); the base read is the same word.
    const int bcol = col % member_columns;
    // phy_prep: th_phy, pi_phy = (p/p1000mb)**rcp, t_phy = th*pi; dz8w.
    const float theta = thb[thb_full ? k * member_columns + bcol : k]
        + thp[idx];
    const float pi_phy = gfk_pow(p[idx] / p0, rcp);
    const float temp = theta * pi_phy;
    th_save[idx] = theta;
    pii[idx] = pi_phy;
    t1d[idx] = temp;
    const float lower = (phb[phb_full ? k * member_columns + bcol : k]
                         + php[idx]) / gravity;
    const float upper = (phb[phb_full ? (k + 1) * member_columns + bcol
                             : k + 1]
                         + php[idx + columns]) / gravity;
    dz[idx] = upper - lower;

    // :1234 the driver's density (unfloored qv), :1267-1282 ng1d.
    const float rho_drv = 0.622f * p[idx] / (TC_R * temp * (qv[idx] + 0.622f));
    const float q_g = qg[idx];
    if (q_g > TC_R1) {
        const float ygra1 = gfk_log10(fmaxf(1.E-9f, q_g * rho_drv));
        float zans1 = 3.0f + 2.f / 7.f * (ygra1 + 8.f);
        zans1 = fmaxf(2.f, fminf(zans1, 6.f));
        const double n0_exp = (double)gfk_pow(10.f, zans1);
        const double lam_exp = glibc_pow(
            n0_exp * (double)TC_AM_G5 * (double)TC_CGG(1, 1)
                / (double)(rho_drv * q_g),
            (double)TC_oge1);
        const double lamg = lam_exp * (double)gfk_pow(
            TC_CGG(3, 1) * TC_ogg2 * TC_ogg1, TC_obmg);
        float ng = (float)(DDIV((double)(TC_CGG(2, 1) * TC_ogg3 * rho_drv * q_g)
                           * glibc_pow(lamg, 3.0), (double)TC_AM_G5));
        ng1d[idx] = fmaxf(TC_R2, ng / rho_drv);
    } else {
        ng1d[idx] = 0.0f;
    }

    // mp_thompson :1798-2002, the per-level parts of no_micro and
    // k_melting.
    const float qvl = fmaxf(1.E-10f, qv[idx]);
    const float rho = 0.622f * p[idx] / (TC_R * temp * (qvl + 0.622f));
    (void)rho;
    bool micro = qc[idx] > TC_R1 || qi[idx] > TC_R1 || qr[idx] > TC_R1
        || qs[idx] > TC_R1 || q_g > TC_R1;
    const float tempc = temp - 273.15f;
    const float qvs = tc_rslf(p[idx], temp);
    const float qvsi = tempc <= 0.0f ? tc_rsif(p[idx], temp) : qvs;
    float ssati = qvl / qvsi - 1.f;
    if (fabsf(ssati) < TC_EPS) ssati = 0.0f;
    if (ssati > 0.0f) micro = true;
    if (micro) atomicOr(&column_micro[col], 1);
    if (!(tempc <= 0.0f)) atomicMax(&column_kmelt[col], k + 1);
}

// Fortran-order flat indexes of the lookup tables (module_mp_thompson.F
// :375-392, classic dimNRHG = 1).
#define TC_IX_RS(is, it, ir1, ir) \
    ((size_t)((is) - 1) + (size_t)37 * ((size_t)((it) - 1) \
     + (size_t)9 * ((size_t)((ir1) - 1) + (size_t)37 * (size_t)((ir) - 1))))
#define TC_IX_IAU(ii, ii1) \
    ((size_t)((ii) - 1) + (size_t)64 * (size_t)((ii1) - 1))
#define TC_IX_QRFZ(ir, ir1, itc, iin) \
    ((size_t)((ir) - 1) + (size_t)37 * ((size_t)((ir1) - 1) \
     + (size_t)37 * ((size_t)((itc) - 1) + (size_t)45 * (size_t)((iin) - 1))))
#define TC_IX_QCFZ(ic, in_, itc, iin) \
    ((size_t)((ic) - 1) + (size_t)37 * ((size_t)((in_) - 1) \
     + (size_t)100 * ((size_t)((itc) - 1) + (size_t)45 * (size_t)((iin) - 1))))
#define TC_IX_EF(ib, ic) \
    ((size_t)((ib) - 1) + (size_t)100 * (size_t)((ic) - 1))

// Tendency and work slots (planes of `size` floats).
#define TC_TTEN 0
#define TC_QVTEN 1
#define TC_QCTEN 2
#define TC_NCTEN 3
#define TC_QITEN 4
#define TC_NITEN 5
#define TC_QRTEN 6
#define TC_NRTEN 7
#define TC_QSTEN 8
#define TC_QGTEN 9
#define TC_NGTEN 10
#define TC_NTEN 11
#define TC_W_RHO 0
#define TC_W_RHOF 1
#define TC_W_RC 2
#define TC_W_NC 3
#define TC_W_RI 4
#define TC_W_NI 5
#define TC_W_RR 6
#define TC_W_NR 7
#define TC_W_RS 8
#define TC_W_RG 9
#define TC_W_NG 10
#define TC_W_BOOST 11
#define TC_W_TEMP 12
#define TC_W_OCP 13
#define TC_W_LVAP 14
#define TC_W_SMOB 15
#define TC_W_SMOC 16
#define TC_W_NC1D 17
#define TC_NWORK 18

// ---------------------------------------------------------------------
// Kernel 2 (one thread per cell): mp_thompson :1798-3574 at one level.
// Writes the eleven float tendencies WRF carries into sedimentation and
// the per-level working state the column kernel reads.
// ---------------------------------------------------------------------
extern "C" __global__ void tc_sources(
    const float* __restrict__ p, const float* __restrict__ t1d,
    const float* __restrict__ qv1d, float* __restrict__ qc1d,
    float* __restrict__ qi1d, float* __restrict__ qr1d,
    float* __restrict__ qs1d, float* __restrict__ qg1d,
    float* __restrict__ ni1d, float* __restrict__ nr1d,
    float* __restrict__ ng1d,
    const int* __restrict__ column_micro,
    const int* __restrict__ column_kmelt,
    const double* __restrict__ tcs_racs1, const double* __restrict__ tmr_racs1,
    const double* __restrict__ tcs_racs2, const double* __restrict__ tmr_racs2,
    const double* __restrict__ tcr_sacr1, const double* __restrict__ tms_sacr1,
    const double* __restrict__ tcr_sacr2, const double* __restrict__ tms_sacr2,
    const double* __restrict__ tnr_racs1, const double* __restrict__ tnr_racs2,
    const double* __restrict__ tnr_sacr1, const double* __restrict__ tnr_sacr2,
    const double* __restrict__ tcg_racg, const double* __restrict__ tmr_racg,
    const double* __restrict__ tcr_gacr, const double* __restrict__ tnr_racg,
    const double* __restrict__ tnr_gacr,
    const double* __restrict__ tpi_qrfz, const double* __restrict__ tni_qrfz,
    const double* __restrict__ tpg_qrfz, const double* __restrict__ tnr_qrfz,
    const double* __restrict__ tpi_qcfz, const double* __restrict__ tni_qcfz,
    const double* __restrict__ tps_iaus, const double* __restrict__ tni_iaus,
    const double* __restrict__ tpi_ide,
    const double* __restrict__ t_efrw, const double* __restrict__ t_efsw,
    float* __restrict__ ten, float* __restrict__ work,
    int* __restrict__ flags, float dt, int size, int columns)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const int k = idx / columns;
    const int col = idx - k * columns;
    const int kf = k + 1;
    const bool no_micro = column_micro[col] == 0;
    const int k_melting = column_kmelt[col];
#define TEN(slot) ten[(size_t)(slot) * (size_t)size + idx]
#define WORK(slot) work[(size_t)(slot) * (size_t)size + idx]

    const float DT = dt;
    const float dtsave = dt;
    const float odt = 1.f / dt;
    const float odts = 1.f / dtsave;

    // Driver entry (:1234, :1250): nc1d = Nt_c/rho with the driver's
    // unfloored density.
    const float pres = p[idx];
    const float temp_in = t1d[idx];
    const float qv_in = qv1d[idx];
    const float rho_drv = 0.622f * pres / (TC_R * temp_in * (qv_in + 0.622f));
    float nc1d = TC_NT_C / rho_drv;

    // :1798-1951 the entry block.
    float temp = temp_in;
    float qv = fmaxf(1.E-10f, qv_in);
    float rho = 0.622f * pres / (TC_R * temp * (qv + 0.622f));
    float rc, nc, ri, ni, rr, nr, rs, rg, ng;
    bool L_qc, L_qi, L_qr, L_qs, L_qg;
    bool rcg_past_end = false;
    float mvd_r = 0.f, mvd_c = 0.f;
    int nu_c = 0;
    double lamc = 0.0, lami = 0.0, ilami = 0.0, lamr = 0.0, lamg = 0.0;
    float xDc = 0.f, xDi = 0.f;

    float qc_e = qc1d[idx], qi_e = qi1d[idx], qr_e = qr1d[idx];
    float qs_e = qs1d[idx], qg_e = qg1d[idx];
    float ni_e = ni1d[idx], nr_e = nr1d[idx];
    float ng_e = ng1d[idx];

    if (qc_e > TC_R1) {
        rc = qc_e * rho;
        nc = fmaxf(2.f, fminf(nc1d * rho, TC_NT_C_MAX));
        L_qc = true;
        nu_c = min(15, tc_nint(1000.E6f / nc) + 2);
        // lamc, xDc and the size balance (:1833-1841) feed nothing in the
        // classic scheme: :1842 sets nc = Nt_c.
        nc = TC_NT_C;
    } else {
        qc_e = 0.0f;
        nc1d = 0.0f;
        rc = TC_R1;
        nc = 2.f;
        L_qc = false;
    }
    WORK(TC_W_NC1D) = nc1d;

    if (qi_e > TC_R1) {
        ri = qi_e * rho;
        ni = fmaxf(TC_R2, ni_e * rho);
        if (ni <= TC_R2) {
            lami = (double)(FDIV(TC_CIE(2), 5.E-6f));
            ni = (float)fmin(999.E3, (double)FDIV(TC_CIG(1) * TC_oig2 * ri,
                 TC_AM_I) * glibc_pow(lami, 3.0));
        }
        L_qi = true;
        lami = (double)gfk_pow(TC_AM_I * TC_CIG(2) * TC_oig1 * ni / ri,
                               TC_obmi);
        ilami = 1. / lami;
        xDi = (float)((double)(3.0f + 0.0f + 1.f) * ilami);
        if (xDi < 5.E-6f) {
            lami = (double)(FDIV(TC_CIE(2), 5.E-6f));
            ni = (float)fmin(999.E3, (double)FDIV(TC_CIG(1) * TC_oig2 * ri,
                 TC_AM_I) * glibc_pow(lami, 3.0));
        } else if (xDi > 300.E-6f) {
            lami = (double)(FDIV(TC_CIE(2), 300.E-6f));
            ni = (float)((double)FDIV(TC_CIG(1) * TC_oig2 * ri, TC_AM_I)
                 * glibc_pow(lami, 3.0));
        }
    } else {
        qi_e = 0.0f;
        ni_e = 0.0f;
        ri = TC_R1;
        ni = TC_R2;
        L_qi = false;
    }

    if (qr_e > TC_R1) {
        rr = qr_e * rho;
        nr = fmaxf(TC_R2, nr_e * rho);
        if (nr <= TC_R2) {
            mvd_r = 1.0E-3f;
            lamr = (double)((3.0f + 0.0f + 0.672f) / mvd_r);
            nr = (float)(DDIV((double)(TC_CRG(2) * TC_org3 * rr)
                 * glibc_pow(lamr, 3.0), (double)TC_AM_R));
        }
        L_qr = true;
        lamr = (double)gfk_pow(TC_AM_R * TC_CRG(3) * TC_org2 * nr / rr,
                               TC_obmr);
        mvd_r = (float)((double)(3.0f + 0.0f + 0.672f) / lamr);
        if (mvd_r > 2.5E-3f) {
            mvd_r = 2.5E-3f;
            lamr = (double)((3.0f + 0.0f + 0.672f) / mvd_r);
            nr = (float)(DDIV((double)(TC_CRG(2) * TC_org3 * rr)
                 * glibc_pow(lamr, 3.0), (double)TC_AM_R));
        } else if (mvd_r < TC_D0R * 0.75f) {
            mvd_r = TC_D0R * 0.75f;
            lamr = (double)((3.0f + 0.0f + 0.672f) / mvd_r);
            nr = (float)(DDIV((double)(TC_CRG(2) * TC_org3 * rr)
                 * glibc_pow(lamr, 3.0), (double)TC_AM_R));
        }
    } else {
        qr_e = 0.0f;
        nr_e = 0.0f;
        rr = TC_R1;
        nr = TC_R2;
        L_qr = false;
    }
    if (qs_e > TC_R1) {
        rs = qs_e * rho;
        L_qs = true;
    } else {
        qs_e = 0.0f;
        rs = TC_R1;
        L_qs = false;
    }
    float mvd_g = 0.f;
    if (qg_e > TC_R1) {
        L_qg = true;
        rg = qg_e * rho;
        ng = fmaxf(TC_R2, ng_e * rho);
        if (ng <= TC_R2) {
            mvd_g = 1.5E-3f;
            lamg = (double)((3.0f + 0.0f + 0.672f) / mvd_g);
            ng = (float)(DDIV((double)(TC_CGG(2, 1) * TC_ogg3 * rg)
                 * glibc_pow(lamg, 3.0), (double)TC_AM_G5));
        }
        lamg = (double)gfk_pow(TC_AM_G5 * TC_CGG(3, 1) * TC_ogg2 * ng / rg,
                               TC_obmg);
        mvd_g = (float)((double)(3.0f + 0.0f + 0.672f) / lamg);
        if (mvd_g > 25.4E-3f) {
            mvd_g = 25.4E-3f;
            lamg = (double)((3.0f + 0.0f + 0.672f) / mvd_g);
            ng = (float)(DDIV((double)(TC_CGG(2, 1) * TC_ogg3 * rg)
                 * glibc_pow(lamg, 3.0), (double)TC_AM_G5));
        } else if (mvd_g < TC_D0R) {
            mvd_g = TC_D0R;
            lamg = (double)((3.0f + 0.0f + 0.672f) / mvd_g);
            ng = (float)(DDIV((double)(TC_CGG(2, 1) * TC_ogg3 * rg)
                 * glibc_pow(lamg, 3.0), (double)TC_AM_G5));
        }
    } else {
        qg_e = 0.0f;
        ng_e = 0.0f;
        rg = TC_R1;
        ng = TC_R2;
        L_qg = false;
    }
    // The entry rewrite reaches the state whether or not the column has
    // microphysics (the 1-D arrays are copied back, :1350-1357).
    qc1d[idx] = qc_e;
    qi1d[idx] = qi_e;
    ni1d[idx] = ni_e;
    qr1d[idx] = qr_e;
    nr1d[idx] = nr_e;
    qs1d[idx] = qs_e;
    qg1d[idx] = qg_e;
    ng1d[idx] = ng_e;

    if (no_micro) {
        flags[idx] = TC_DONE;
        for (int n = 0; n < TC_NTEN; ++n) TEN(n) = 0.0f;
        return;
    }

    // :1971-2013 thermodynamics.
    float tempc = temp - 273.15f;
    float rhof = sqrtf(TC_RHO_NOT / rho);
    float rhof2 = sqrtf(rhof);
    float qvs = tc_rslf(pres, temp);
    const float delQvs = fmaxf(0.0f, tc_rslf(pres, 273.15f) - qv);
    float qvsi;
    if (tempc <= 0.0f) {
        qvsi = tc_rsif(pres, temp);
    } else {
        qvsi = qvs;
    }
    const float satw = qv / qvs;
    const float sati = qv / qvsi;
    float ssatw = satw - 1.f;
    float ssati = sati - 1.f;
    if (fabsf(ssatw) < TC_EPS) ssatw = 0.0f;
    if (fabsf(ssati) < TC_EPS) ssati = 0.0f;
    float diffu = 2.11E-5f * gfk_pow(FDIV(temp, 273.15f), 1.94f)
        * (101325.f / pres);
    float visco;
    if (tempc >= 0.0f) {
        visco = (1.718f + 0.0049f * tempc) * 1.0E-5f;
    } else {
        visco = (1.718f + 0.0049f * tempc - 1.2E-5f * tempc * tempc)
            * 1.0E-5f;
    }
    float ocp = 1.f / (TC_CP * (1.f + 0.887f * qv));
    float vsc2 = sqrtf(rho / visco);
    float lvap = TC_LVAP0 + (2106.0f - 4218.0f) * tempc;
    float tcond = (5.69f + 0.0168f * tempc) * 1.0E-5f * 418.936f;
    float twet = temp;
    if (k_melting > 0 && kf <= k_melting && satw < 0.999f) {
        const float dew_t = fminf(temp - 0.001f, tc_t_dew(pres, qv));
        const float Tlcl = tc_t_lcl(temp, dew_t);
        const float The = tc_theta_e(pres, temp, qv, Tlcl);
        twet = fminf(temp, tc_compt_fr_the(The, pres));
    }

    // :2025-2129 snow moments.
    float smo0 = 0.f, smo1 = 0.f, smo2 = 0.f, smob = 0.f, smoc = 0.f;
    float smoe = 0.f, smof = 0.f;
    if (L_qs) {
        const float tc0 = fminf(-0.1f, temp - 273.15f);
        smob = rs * TC_oams;
        smo2 = smob;   // bm_s == 2 (:2033)
        float loga_ = SA(1) + SA(2) * tc0 + SA(5) * tc0 * tc0
            + SA(9) * tc0 * tc0 * tc0;
        float a_ = gfk_pow(10.0f, loga_);
        float b_ = SB(1) + SB(2) * tc0 + SB(5) * tc0 * tc0
            + SB(9) * tc0 * tc0 * tc0;
        smo0 = a_ * gfk_pow(smo2, b_);
        loga_ = SA(1) + SA(2) * tc0 + SA(3)
            + SA(4) * tc0 + SA(5) * tc0 * tc0
            + SA(6) + SA(7) * tc0 * tc0
            + SA(8) * tc0 + SA(9) * tc0 * tc0 * tc0
            + SA(10);
        a_ = gfk_pow(10.0f, loga_);
        b_ = SB(1) + SB(2) * tc0 + SB(3) + SB(4) * tc0
            + SB(5) * tc0 * tc0 + SB(6)
            + SB(7) * tc0 * tc0 + SB(8) * tc0
            + SB(9) * tc0 * tc0 * tc0 + SB(10);
        smo1 = a_ * gfk_pow(smo2, b_);
        a_ = gfk_pow(10.0f, tc_poly(tc_sa, tc0, TC_CSE(1)));
        b_ = tc_poly(tc_sb, tc0, TC_CSE(1));
        smoc = a_ * gfk_pow(smo2, b_);
        // ns(k) (:2083-2088) feeds only the aerosol-aware Koop branch.
        a_ = gfk_pow(10.0f, tc_poly(tc_sa, tc0, TC_CSE(13)));
        b_ = tc_poly(tc_sb, tc0, TC_CSE(13));
        smoe = a_ * gfk_pow(smo2, b_);
        a_ = gfk_pow(10.0f, tc_poly(tc_sa, tc0, TC_CSE(16)));
        b_ = tc_poly(tc_sb, tc0, TC_CSE(16));
        smof = a_ * gfk_pow(smo2, b_);
        // smog (:2117-2127) feeds no classic output.
    }

    // :2135-2139 graupel, :2146-2151 rain intercepts and slopes.
    double ilamg, N0_g, ilamr, N0_r;
    lamg = (double)gfk_pow(TC_AM_G5 * TC_CGG(3, 1) * TC_ogg2 * ng / rg,
                           TC_obmg);
    ilamg = 1. / lamg;
    N0_g = (double)(ng * TC_ogg2) * glibc_pow(lamg, (double)TC_CGE(2, 1));
    lamr = (double)gfk_pow(TC_AM_R * TC_CRG(3) * TC_org2 * nr / rr, TC_obmr);
    ilamr = 1. / lamr;
    mvd_r = (float)((double)(3.0f + 0.0f + 0.672f) / lamr);
    N0_r = (double)(nr * TC_org2) * glibc_pow(lamr, (double)TC_CRE(2));

    // Process rates (:1650-1777 zero them).
    double prw_vcd = 0., pnc_wcd = 0., pnc_wau = 0., pnc_rcw = 0.,
        pnc_scw = 0., pnc_gcw = 0.;
    double prr_wau = 0., prr_rcw = 0., prr_rcs = 0., prr_rcg = 0.,
        prr_sml = 0., prr_gml = 0., prr_rci = 0., prv_rev = 0.;
    double pnr_wau = 0., pnr_rcs = 0., pnr_rcg = 0., pnr_rci = 0.,
        pnr_sml = 0., pnr_gml = 0., pnr_rev = 0., pnr_rcr = 0., pnr_rfz = 0.;
    double pri_inu = 0., pni_inu = 0., pri_ihm = 0., pni_ihm = 0.,
        pri_wfz = 0., pni_wfz = 0., pri_rfz = 0., pni_rfz = 0., pri_ide = 0.,
        pni_ide = 0., pri_rci = 0., pni_rci = 0., pni_sci = 0., pni_iau = 0.,
        pri_iha = 0., pni_iha = 0.;
    double prs_iau = 0., prs_sci = 0., prs_rcs = 0., prs_scw = 0.,
        prs_sde = 0., prs_ihm = 0., prs_ide = 0.;
    double prg_scw = 0., prg_rfz = 0., prg_gde = 0., prg_gcw = 0.,
        prg_rci = 0., prg_rcs = 0., prg_rcg = 0., prg_ihm = 0.;
    double png_rcs = 0., png_rcg = 0., png_scw = 0., png_gde = 0.;

    // :2157-2234 warm rain.
    if (L_qr && mvd_r > TC_D0R) {
        const float Ef_rr = 1.0f - gfk_exp(2300.0f * (mvd_r - 1950.0E-6f));
        pnr_rcr = (double)(Ef_rr * 2.0f * nr * rr);
    }
    mvd_c = TC_D0C;
    if (L_qc) {
        nu_c = min(15, tc_nint(1000.E6f / nc) + 2);
        xDc = fmaxf(TC_D0C * 1.E6f,
                    (gfk_pow(FDIV(rc, (TC_AM_R * nc)), TC_obmr)) * 1.E6f);
        lamc = (double)gfk_pow(nc * TC_AM_R * TC_CCG(2, nu_c)
                               * TC_OCG1(nu_c) / rc, TC_obmr);
        mvd_c = (float)((double)(3.0f + (float)nu_c + 0.672f) / lamc);
        mvd_c = fmaxf(TC_D0C, fminf(mvd_c, TC_D0R));
    }
    if (rc > 0.01e-3f) {
        const float Dc_g = (float)((double)gfk_pow(
            TC_CCG(3, nu_c) * TC_OCG2(nu_c), TC_obmr) / lamc * 1.E6);
        const float Dc_b = gfk_pow(xDc * xDc * xDc * Dc_g * Dc_g * Dc_g
            - xDc * xDc * xDc * xDc * xDc * xDc, 1.f / 6.f);
        const float zeta1 = 0.5f * ((6.25E-6f * xDc * Dc_b * Dc_b * Dc_b
            - 0.4f) + fabsf(6.25E-6f * xDc * Dc_b * Dc_b * Dc_b - 0.4f));
        const float zeta = 0.027f * rc * zeta1;
        const float taud = 0.5f * ((0.5f * Dc_b - 7.5f)
            + fabsf(0.5f * Dc_b - 7.5f)) + TC_R1;
        const float tau = 3.72f / (rc * taud);
        prr_wau = (double)(zeta / tau);
        prr_wau = fmin((double)(rc * odts), prr_wau);
        pnr_wau = prr_wau / (double)(TC_AM_R * (float)nu_c * 10.f * TC_D0R
                                     * TC_D0R * TC_D0R);
        pnc_wau = fmin((double)(nc * odts), prr_wau
            / (double)(TC_AM_R * mvd_c * mvd_c * mvd_c));
    }
    if (L_qr && mvd_r > TC_D0R && mvd_c > TC_D0C) {
        lamr = 1. / ilamr;
        int ib = 1 + (int)((double)TC_NBR * glibc_log(DDIV((double)mvd_r,
            TC_DR(1))) / glibc_log(DDIV(TC_DR(TC_NBR), TC_DR(1))));
        ib = min(ib, TC_NBR);
        const float Ef_rw = (float)t_efrw[TC_IX_EF(ib, (int)(mvd_c * 1.E6f))];
        prr_rcw = (double)(rhof * TC_t1_qr_qc * Ef_rw * rc) * N0_r
            * glibc_pow(lamr + (double)TC_FV_R, (double)(-TC_CRE(9)));
        prr_rcw = fmin((double)(rc * odts), prr_rcw);
        pnc_rcw = (double)(rhof * TC_t1_qr_qc * Ef_rw * nc) * N0_r
            * glibc_pow(lamr + (double)TC_FV_R, (double)(-TC_CRE(9)));
        pnc_rcw = fmin((double)(nc * odts), pnc_rcw);
    }
    // :2210-2232 aerosol scavenging: aerosol tendencies only.

    // :2239-2850 frozen species.
    float vts_boost = 1.0f;
    {
        float xDs;
        if (L_qs) {
            xDs = smoc / smob;
        } else {
            xDs = 0.f;
        }
        tempc = temp - 273.15f;
        const int idx_tc = max(1, min(tc_nint(-tempc), 45));
        int idx_t = (int)FDIV((tempc - 2.5f), 5.f) - 1;
        idx_t = max(1, -idx_t);
        idx_t = min(idx_t, TC_NTB_T);

        int idx_c = 1;
        if (rc > 1.e-6f) {
            idx_c = tc_decade(rc, TC_nic2);
            idx_c = max(1, min(idx_c, TC_NTB_C));
        }
        int idx_n = tc_nint_d(1.0 + DDIV((double)(float)TC_NBC
            * glibc_log(DDIV((double)nc, TC_T_NC(1))), (double)TC_nic1));
        idx_n = max(1, min(idx_n, TC_NBC));
        int idx_i = 1;
        if (ri > 1.e-10f) {
            idx_i = tc_decade(ri, TC_nii2);
            idx_i = max(1, min(idx_i, TC_NTB_I));
        }
        int idx_i1 = 1;
        if (ni > 1.0f) {
            idx_i1 = tc_decade(ni, TC_nii3);
            idx_i1 = max(1, min(idx_i1, TC_NTB_I1));
        }
        int idx_r, idx_r1;
        if (rr > 1.e-6f) {
            idx_r = tc_decade(rr, TC_nir2);
            idx_r = max(1, min(idx_r, TC_NTB_R));
            lamr = 1. / ilamr;
            const double lam_exp = lamr * (double)gfk_pow(
                TC_CRG(3) * TC_org2 * TC_org1, 3.0f);
            const double N0_exp = (double)FDIV(TC_org1 * rr, TC_AM_R)
                * glibc_pow(lam_exp, (double)TC_CRE(1));
            idx_r1 = tc_decade_d(N0_exp, TC_nir3);
            idx_r1 = max(1, min(idx_r1, TC_NTB_R1));
        } else {
            idx_r = 1;
            idx_r1 = TC_NTB_R1;
        }
        int idx_s = 1;
        if (rs > 1.e-6f) {
            idx_s = tc_decade(rs, TC_nis2);
            idx_s = max(1, min(idx_s, TC_NTB_S));
        }
        int idx_g, idx_g1;
        if (rg > 1.e-6f) {
            idx_g = tc_decade(rg, TC_nig2);
            idx_g = max(1, min(idx_g, TC_NTB_G));
            lamg = 1. / ilamg;
            const double lam_exp = lamg * (double)gfk_pow(
                TC_CGG(3, 1) * TC_ogg2 * TC_ogg1, 3.0f);
            const double N0_exp = (double)FDIV(TC_ogg1 * rg, TC_AM_G5)
                * glibc_pow(lam_exp, (double)TC_CGE(1, 1));
            idx_g1 = tc_decade_d(N0_exp, TC_nig3);
            idx_g1 = max(1, min(idx_g1, TC_NTB_G1));
        } else {
            idx_g = 1;
            idx_g1 = TC_NTB_G1;
        }

        // :2384-2400 Srivastava & Coen prefactor.
        const float otemp = 1.f / temp;
        const float rvs = rho * qvsi;
        const float rvs_p = rvs * otemp * (TC_LSUB * otemp * TC_ORV - 1.f);
        const float rvs_pp = rvs * (otemp * (TC_LSUB * otemp * TC_ORV - 1.f)
            * otemp * (TC_LSUB * otemp * TC_ORV - 1.f)
            + (-2.f * TC_LSUB * otemp * otemp * otemp * TC_ORV)
            + otemp * otemp);
        const float gamsc = TC_LSUB * diffu / tcond * rvs_p;
        float alphsc = 0.5f * (gamsc / (1.f + gamsc)) * (gamsc / (1.f + gamsc))
            * rvs_pp / rvs_p * rvs / rvs_p;
        alphsc = fmaxf(1.E-9f, alphsc);
        float xsat = ssati;
        if (fabsf(xsat) < 1.E-9f) xsat = 0.f;
        const float t1_subl = 4.f * TC_PI * (1.0f - alphsc * xsat
            + 2.f * alphsc * alphsc * xsat * xsat
            - 5.f * alphsc * alphsc * alphsc * xsat * xsat * xsat)
            / (1.f + gamsc);

        // :2403-2440 snow and graupel collecting cloud water.
        if (L_qc && mvd_c > TC_D0C) {
            if (xDs > TC_D0S) {
                int ib = 1 + (int)((double)TC_NBS * glibc_log(DDIV((double)xDs,
                    TC_DS(1))) / glibc_log(DDIV(TC_DS(TC_NBS), TC_DS(1))));
                ib = min(ib, TC_NBS);
                const float Ef_sw = (float)t_efsw[TC_IX_EF(ib,
                                                   (int)(mvd_c * 1.E6f))];
                prs_scw = (double)(rhof * TC_t1_qs_qc * Ef_sw * rc * smoe);
                prs_scw = fmin((double)(rc * odts), prs_scw);
                pnc_scw = (double)(rhof * TC_t1_qs_qc * Ef_sw * nc * smoe);
                pnc_scw = fmin((double)(nc * odts), pnc_scw);
            }
            if (rg >= 1.e-6f && mvd_c > TC_D0C) {
                const float xDg = (float)((double)(3.0f + 0.0f + 1.f)
                                          * ilamg);
                const float vtg = (float)((double)(rhof * TC_AV_G(TC_IDX_BG1)
                    * TC_CGG(6, TC_IDX_BG1) * TC_ogg3)
                    * glibc_pow(ilamg, (double)TC_BV_G(TC_IDX_BG1)));
                const float stoke_g = mvd_c * mvd_c * vtg * 1000.0f
                    / (9.f * visco * xDg);
                float Ef_gw;
                if (stoke_g >= 0.4f && stoke_g <= 10.f) {
                    Ef_gw = 0.55f * gfk_log10(2.51f * stoke_g);
                } else if (stoke_g < 0.4f) {
                    Ef_gw = 0.0f;
                } else {
                    Ef_gw = 0.77f;
                }
                if (twet > TC_T0) Ef_gw = Ef_gw * 0.1f;
                const float t1_qg_qc = TC_PI * .25f * TC_AV_G(TC_IDX_BG1)
                    * TC_CGG(9, TC_IDX_BG1);
                prg_gcw = (double)(rhof * t1_qg_qc * Ef_gw * rc) * N0_g
                    * glibc_pow(ilamg, (double)TC_CGE(9, TC_IDX_BG1));
                pnc_gcw = (double)(rhof * t1_qg_qc * Ef_gw * nc) * N0_g
                    * glibc_pow(ilamg, (double)TC_CGE(9, TC_IDX_BG1));
                pnc_gcw = fmin((double)(nc * odts), pnc_gcw);
            }
        }

        // :2486-2548 rain collecting snow and graupel.
        if (rr >= 1.e-6f) {
            if (rs >= 1.e-6f) {
                const size_t ix = TC_IX_RS(idx_s, idx_t, idx_r1, idx_r);
                if (twet < TC_T0) {
                    prr_rcs = -(tmr_racs2[ix] + tcr_sacr2[ix]
                                + tmr_racs1[ix] + tcr_sacr1[ix]);
                    prs_rcs = tmr_racs2[ix] + tcr_sacr2[ix]
                        - tcs_racs1[ix] - tms_sacr1[ix];
                    prg_rcs = tmr_racs1[ix] + tcr_sacr1[ix]
                        + tcs_racs1[ix] + tms_sacr1[ix];
                    prr_rcs = fmax((double)(-rr * odts), prr_rcs);
                    prs_rcs = fmax((double)(-rs * odts), prs_rcs);
                    prg_rcs = fmin((double)((rr + rs) * odts), prg_rcs);
                    pnr_rcs = tnr_racs1[ix] + tnr_racs2[ix]
                        + tnr_sacr1[ix] + tnr_sacr2[ix];
                    pnr_rcs = fmin((double)(nr * odts), pnr_rcs);
                    png_rcs = pnr_rcs;
                } else {
                    prs_rcs = -tcs_racs1[ix] - tms_sacr1[ix]
                        + tmr_racs2[ix] + tcr_sacr2[ix];
                    prs_rcs = fmax((double)(-rs * odts), prs_rcs);
                    prr_rcs = -prs_rcs;
                }
            }
            if (rg >= 1.e-6f) {
                // :2524-2546, WRF v4.6.1's read, line for line.  WRF reads
                // these five tables at idx_bg(k) = idx_bg1 = 5, but for
                // mp=8 they are allocated with dimNRHG = 1 (:465,
                // :607-615) and qr_acr_qg fills that one slot
                // (:4192-4265).  gfortran (no bounds checking) addresses
                // index 5 on that extent-1 dimension four ntb_g1*ntb_g
                // strides on: the entry for drop intercept bin
                // idx_r1 + 4.  This unit is the WRF transcription
                // (GPUWM_WRF_EXACT=1, exact mode), so it reads that same
                // word.  Where the address passes the end of the
                // allocation (idx_r = ntb_r and idx_r1 >= ntb_r1 - 3) WRF
                // reads heap contents, which nothing can reproduce: the
                // unit reads the slot the table holds and marks the cell
                // TC_RCG_PAST_END.  The forecast's default adapter
                // (thompson.cu thompson_racg_index) keeps the repaired
                // read (tools/thompson_mp8_column_oracle/repair_wrf461.py,
                // rcg) as its declared divergence.
                const size_t ix_slot = (size_t)(idx_g1 - 1) + (size_t)37
                    * ((size_t)(idx_g - 1) + (size_t)37
                    * ((size_t)(idx_r1 - 1) + (size_t)37
                    * (size_t)(idx_r - 1)));
                size_t ix = ix_slot + TC_RACG_BG1_OFFSET;
                if (ix >= TC_RACG_WORDS) {
                    rcg_past_end = true;
                    ix = ix_slot;
                }
                if (twet < TC_T0) {
                    prg_rcg = tmr_racg[ix] + tcr_gacr[ix];
                    prg_rcg = fmin((double)(rr * odts), prg_rcg);
                    prr_rcg = -prg_rcg;
                    pnr_rcg = tnr_racg[ix] + tnr_gacr[ix];
                    pnr_rcg = fmin((double)(nr * odts), pnr_rcg);
                } else {
                    prr_rcg = tcg_racg[ix];
                    prr_rcg = fmin((double)(rg * odts), prr_rcg);
                    prg_rcg = -prr_rcg;
                    png_rcg = tnr_racg[ix];
                    png_rcg = fmin((double)(ng * odts), png_rcg);
                    pnr_rcg = -1.5 * tnr_gacr[ix];
                }
            }
        }

        if (temp < TC_T0) {
            // :2554-2777 below 0 C.
            vts_boost = 1.0f;
            const float rate_max = (qv - qvsi) * rho * odts * 0.999f;
            const float xni_in = 1.0f * 1000.f;
            int idx_IN = tc_decade(xni_in, TC_niIN2);
            idx_IN = max(1, min(idx_IN, TC_NTB_IN));
            if (rr > 1.e-6f) {
                const size_t ix = TC_IX_QRFZ(idx_r, idx_r1, idx_tc, idx_IN);
                prg_rfz = tpg_qrfz[ix] * (double)odts;
                pri_rfz = tpi_qrfz[ix] * (double)odts;
                pni_rfz = tni_qrfz[ix] * (double)odts;
                pnr_rfz = tnr_qrfz[ix] * (double)odts;
                pnr_rfz = fmin((double)(nr * odts), pnr_rfz);
            } else if (rr > TC_R1 && temp < TC_HGFR) {
                pri_rfz = (double)(rr * odts);
                pni_rfz = (double)(nr * odts);
            }
            if (rc > 1.e-6f) {
                const size_t ix = TC_IX_QCFZ(idx_c, idx_n, idx_tc, idx_IN);
                pri_wfz = tpi_qcfz[ix] * (double)odts;
                pri_wfz = fmin((double)(rc * odts), pri_wfz);
                pni_wfz = tni_qcfz[ix] * (double)odts;
                pni_wfz = fmin(fmin((double)(nc * odts),
                                    DDIV(pri_wfz, (double)(2.f * TC_XM0I))),
                               pni_wfz);
            } else if (rc > TC_R1 && temp < TC_HGFR) {
                pri_wfz = (double)(rc * odts);
                pni_wfz = (double)(nc * odts);
            }
            if ((ssati >= 0.25f) || (ssatw > TC_EPS && temp < 253.15f)) {
                const float xnc = fminf(250.E3f,
                    TC_TNO * gfk_exp(TC_ATO * (TC_T0 - temp)));
                const float xni = (float)((double)ni + (pni_rfz + pni_wfz)
                                          * (double)dtsave);
                pni_inu = (double)(0.5f * (xnc - xni + fabsf(xnc - xni))
                                   * odts);
                pri_inu = fmin((double)rate_max, (double)TC_XM0I * pni_inu);
                pni_inu = DDIV(pri_inu, (double)TC_XM0I);
            }
            // :2634-2641 Koop freezing: aerosol-aware only.

            if (L_qi) {
                lami = (double)gfk_pow(TC_AM_I * TC_CIG(2) * TC_oig1 * ni
                                       / ri, TC_obmi);
                ilami = 1. / lami;
                xDi = (float)fmax((double)TC_D0i,
                                  (double)(3.0f + 0.0f + 1.f) * ilami);
                const float xmi = TC_AM_I * gfk_pow(xDi, 3.0f);
                const float oxmi = 1.f / xmi;
                pri_ide = (double)(TC_C_CUBE * t1_subl * diffu * ssati * rvs
                    * TC_oig1 * TC_CIG(5) * ni) * ilami;
                if (pri_ide < 0.0) {
                    pri_ide = fmax(fmax((double)(-ri * odts), pri_ide),
                                   (double)rate_max);
                    pni_ide = pri_ide * (double)oxmi;
                    pni_ide = fmax((double)(-ni * odts), pni_ide);
                } else {
                    pri_ide = fmin(pri_ide, (double)rate_max);
                    const size_t ix = TC_IX_IAU(idx_i, idx_i1);
                    prs_ide = (1.0 - tpi_ide[ix]) * pri_ide;
                    pri_ide = tpi_ide[ix] * pri_ide;
                }
                if ((idx_i == TC_NTB_I) || (xDi > 5.0f * TC_D0S)) {
                    prs_iau = (double)(ri * .99f * odts);
                    pni_iau = (double)(ni * .95f * odts);
                } else if (xDi < 0.1f * TC_D0S) {
                    prs_iau = 0.;
                    pni_iau = 0.;
                } else {
                    const size_t ix = TC_IX_IAU(idx_i, idx_i1);
                    prs_iau = tps_iaus[ix] * (double)odts;
                    prs_iau = fmin((double)(ri * .99f * odts), prs_iau);
                    pni_iau = tni_iaus[ix] * (double)odts;
                    pni_iau = fmin((double)(ni * .95f * odts), pni_iau);
                }
            }

            if (L_qs) {
                float C_snow = TC_C_SQRD + FDIV((tempc + 1.5f)
                    * (TC_C_CUBE - TC_C_SQRD), (-30.f + 1.5f));
                C_snow = fmaxf(TC_C_SQRD, fminf(C_snow, TC_C_CUBE));
                prs_sde = (double)(C_snow * t1_subl * diffu * ssati * rvs
                    * (TC_t1_qs_sd * smo1
                       + TC_t2_qs_sd * rhof2 * vsc2 * smof));
                if (prs_sde < 0.) {
                    prs_sde = fmax(fmax((double)(-rs * odts), prs_sde),
                                   (double)rate_max);
                } else {
                    prs_sde = fmin(prs_sde, (double)rate_max);
                }
            }

            if (L_qg && ssati < -TC_EPS) {
                const float t2_qg_sd = 0.28f * TC_Sc3
                    * sqrtf(TC_AV_G(TC_IDX_BG1)) * TC_CGG(11, TC_IDX_BG1);
                prg_gde = (double)(TC_C_CUBE * t1_subl * diffu * ssati * rvs)
                    * N0_g * ((double)TC_t1_qg_sd
                              * glibc_pow(ilamg, (double)TC_CGE(10, 1))
                    + (double)(t2_qg_sd * vsc2 * rhof2)
                      * glibc_pow(ilamg, (double)TC_CGE(11, TC_IDX_BG1)));
                if (prg_gde < 0.) {
                    prg_gde = fmax(fmax((double)(-rg * odts), prg_gde),
                                   (double)rate_max);
                    png_gde = prg_gde * (double)ng / (double)rg;
                } else {
                    prg_gde = fmin(prg_gde, (double)rate_max);
                }
            }

            if (L_qi) {
                lami = (double)gfk_pow(TC_AM_I * TC_CIG(2) * TC_oig1 * ni
                                       / ri, TC_obmi);
                ilami = 1. / lami;
                xDi = (float)fmax((double)TC_D0i,
                                  (double)(3.0f + 0.0f + 1.f) * ilami);
                const float xmi = TC_AM_I * gfk_pow(xDi, 3.0f);
                const float oxmi = 1.f / xmi;
                if (rs >= 1.e-6f) {
                    prs_sci = (double)(TC_t1_qs_qi * rhof * TC_EF_SI * ri
                                       * smoe);
                    pni_sci = prs_sci * (double)oxmi;
                }
                if (rr >= 1.e-6f && mvd_r > 4.f * xDi) {
                    lamr = 1. / ilamr;
                    pri_rci = (double)(rhof * TC_t1_qr_qi * TC_EF_RI * ri)
                        * N0_r * glibc_pow(lamr + (double)TC_FV_R,
                                           (double)(-TC_CRE(9)));
                    pnr_rci = (double)(rhof * TC_t1_qr_qi * TC_EF_RI * ni)
                        * N0_r * glibc_pow(lamr + (double)TC_FV_R,
                                           (double)(-TC_CRE(9)));
                    pnr_rci = fmin((double)(nr * odts), pnr_rci);
                    pni_rci = pri_rci * (double)oxmi;
                    prr_rci = (double)(rhof * TC_t2_qr_qi * TC_EF_RI * ni)
                        * N0_r * glibc_pow(lamr + (double)TC_FV_R,
                                           (double)(-TC_CRE(8)));
                    prr_rci = fmin((double)(rr * odts), prr_rci);
                    prg_rci = pri_rci + prr_rci;
                }
            }

            if (prg_gcw > (double)TC_EPS && tempc > -8.0f) {
                float tf = 0.f;
                if (tempc >= -5.0f && tempc < -3.0f) {
                    tf = 0.5f * (-3.0f - tempc);
                } else if (tempc > -8.0f && tempc < -5.0f) {
                    tf = 0.33333333f * (8.0f + tempc);
                }
                pni_ihm = (double)(3.5E8f * tf) * prg_gcw;
                pri_ihm = (double)TC_XM0I * pni_ihm;
                prs_ihm = prs_scw / (prs_scw + prg_gcw) * pri_ihm;
                prg_ihm = prg_gcw / (prs_scw + prg_gcw) * pri_ihm;
            }

            if (prs_scw > 2.0 * prs_sde && prs_sde > (double)TC_EPS) {
                const float r_frac = (float)fmin(30.0, prs_scw / prs_sde);
                float g_frac = fminf(0.95f, 0.15f + (r_frac - 2.f) * .028f);
                vts_boost = fminf(1.5f, 1.1f + (r_frac - 2.f) * .014f);
                prg_scw = (double)g_frac * prs_scw;
                png_scw = prg_scw * (double)smo0 / (double)rs;
                const float vts = TC_AV_S * gfk_pow(xDs, TC_BV_S)
                    * gfk_exp(-TC_FV_S * xDs);
                float const_Ri = -1.f * (mvd_c * 0.5E6f) * vts
                    / fminf(-0.1f, tempc);
                const_Ri = fmaxf(0.1f, fminf(const_Ri, 10.f));
                const float rime_dens = (0.051f + 0.114f * const_Ri
                    - 0.0055f * const_Ri * const_Ri) * 1000.f;
                if (rime_dens < 150.f) {
                    g_frac = 0.f;
                    prg_scw = 0.;
                    png_scw = 0.;
                }
                prs_scw = (double)(1.f - g_frac) * prs_scw;
            }
        } else {
            // :2783-2844 at or above 0 C.
            if (L_qs) {
                prr_sml = (double)((tempc * tcond - TC_LVAP0 * diffu
                                    * delQvs)
                    * (TC_t1_qs_me * smo1
                       + TC_t2_qs_me * rhof2 * vsc2 * smof));
                if (prr_sml > 0.) {
                    prr_sml = prr_sml + (double)(4218.f * TC_OLFUS
                        * (twet - TC_T0)) * (prr_rcs + prs_scw);
                }
                prr_sml = fmin((double)(rs * odts), fmax(0., prr_sml));
                if (prr_sml > 0.0) {
                    pnr_sml = (double)(smo0 / rs) * prr_sml
                        * (double)gfk_pow(10.0f, -0.25f * (twet - TC_T0));
                } else if (ssati < 0.f) {
                    prs_sde = (double)(TC_C_SQRD * t1_subl * diffu * ssati
                        * rvs * (TC_t1_qs_sd * smo1
                                 + TC_t2_qs_sd * rhof2 * vsc2 * smof));
                    prs_sde = fmax((double)(-rs * odts), prs_sde);
                }
            }
            if (L_qg) {
                double N0_melt = N0_g;
                if ((rg * ng) < 1.E-4f) {
                    lamg = 1. / ilamg;
                    N0_melt = (double)((1.E-4f / rg) * TC_ogg2)
                        * glibc_pow(lamg, (double)TC_CGE(2, 1));
                }
                const float t2_qg_me = TC_PI * 4.f * TC_C_CUBE * TC_OLFUS
                    * 0.28f * TC_Sc3 * sqrtf(TC_AV_G(TC_IDX_BG1))
                    * TC_CGG(11, TC_IDX_BG1);
                prr_gml = (double)(tempc * tcond - TC_LVAP0 * diffu * delQvs)
                    * N0_melt * ((double)TC_t1_qg_me
                                 * glibc_pow(ilamg, (double)TC_CGE(10, 1))
                    + (double)(t2_qg_me * rhof2 * vsc2)
                      * glibc_pow(ilamg, (double)TC_CGE(11, TC_IDX_BG1)));
                prr_gml = fmin((double)(rg * odts), fmax(0., prr_gml));
                if (prr_gml > 0.0) {
                    pnr_gml = prr_gml * (double)ng / (double)rg
                        * (double)gfk_pow(10.0f, -0.33f * (twet - TC_T0));
                } else if (ssati < 0.f) {
                    const float t2_qg_sd = 0.28f * TC_Sc3
                        * sqrtf(TC_AV_G(TC_IDX_BG1)) * TC_CGG(11, TC_IDX_BG1);
                    prg_gde = (double)(TC_C_CUBE * t1_subl * diffu * ssati
                                       * rvs)
                        * N0_g * ((double)TC_t1_qg_sd
                                  * glibc_pow(ilamg, (double)TC_CGE(10, 1))
                        + (double)(t2_qg_sd * vsc2 * rhof2)
                          * glibc_pow(ilamg, (double)TC_CGE(11, TC_IDX_BG1)));
                    prg_gde = fmax((double)(-rg * odts), prg_gde);
                    png_gde = prg_gde * (double)ng / (double)rg;
                }
            }
            if (dt > 120.f) {
                prr_rcw = prr_rcw + prs_scw + prg_gcw;
                prs_scw = 0.;
                prg_gcw = 0.;
            }
        }
    }

    // :2856-2950 conservation limiters.  ratio and rate_max are REAL.
    {
        float sump = (float)(pri_inu + pri_ide + prs_ide + prs_sde
            + prg_gde + pri_iha);
        float rate_max = (qv - qvsi) * rho * odts * 0.999f;
        if ((sump > TC_EPS && sump > rate_max)
                || (sump < -TC_EPS && sump < rate_max)) {
            const float ratio = rate_max / sump;
            pri_inu = pri_inu * (double)ratio;
            pri_ide = pri_ide * (double)ratio;
            pni_ide = pni_ide * (double)ratio;
            prs_ide = prs_ide * (double)ratio;
            prs_sde = prs_sde * (double)ratio;
            prg_gde = prg_gde * (double)ratio;
            pri_iha = pri_iha * (double)ratio;
        }
        sump = (float)(-prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw
            - prg_gcw);
        rate_max = -rc * odts;
        if (sump < rate_max && L_qc) {
            const float ratio = rate_max / sump;
            prr_wau = prr_wau * (double)ratio;
            pri_wfz = pri_wfz * (double)ratio;
            prr_rcw = prr_rcw * (double)ratio;
            prs_scw = prs_scw * (double)ratio;
            prg_scw = prg_scw * (double)ratio;
            prg_gcw = prg_gcw * (double)ratio;
        }
        sump = (float)(pri_ide - prs_iau - prs_sci - pri_rci);
        rate_max = -ri * odts;
        if (sump < rate_max && L_qi) {
            const float ratio = rate_max / sump;
            pri_ide = pri_ide * (double)ratio;
            prs_iau = prs_iau * (double)ratio;
            prs_sci = prs_sci * (double)ratio;
            pri_rci = pri_rci * (double)ratio;
        }
        sump = (float)(-prg_rfz - pri_rfz - prr_rci + prr_rcs + prr_rcg);
        rate_max = -rr * odts;
        if (sump < rate_max && L_qr) {
            const float ratio = rate_max / sump;
            prg_rfz = prg_rfz * (double)ratio;
            pri_rfz = pri_rfz * (double)ratio;
            prr_rci = prr_rci * (double)ratio;
            prr_rcs = prr_rcs * (double)ratio;
            prr_rcg = prr_rcg * (double)ratio;
        }
        sump = (float)(prs_sde - prs_ihm - prr_sml + prs_rcs);
        rate_max = -rs * odts;
        if (sump < rate_max && L_qs) {
            const float ratio = rate_max / sump;
            prs_sde = prs_sde * (double)ratio;
            prs_ihm = prs_ihm * (double)ratio;
            prr_sml = prr_sml * (double)ratio;
            prs_rcs = prs_rcs * (double)ratio;
        }
        sump = (float)(prg_gde - prg_ihm - prr_gml + prg_rcg);
        rate_max = -rg * odts;
        if (sump < rate_max && L_qg) {
            const float ratio = rate_max / sump;
            prg_gde = prg_gde * (double)ratio;
            prg_ihm = prg_ihm * (double)ratio;
            prr_gml = prr_gml * (double)ratio;
            prg_rcg = prg_rcg * (double)ratio;
        }
        pri_ihm = prs_ihm + prg_ihm;
        float ratio = (float)fmin(fabs(prr_rcg), fabs(prg_rcg));
        prr_rcg = (double)(ratio * copysignf(1.0f, (float)prr_rcg));
        prg_rcg = -prr_rcg;
        if (twet > TC_T0) {
            ratio = (float)fmin(fabs(prr_rcs), fabs(prs_rcs));
            prr_rcs = (double)(ratio * copysignf(1.0f, (float)prr_rcs));
            prs_rcs = -prr_rcs;
        }
    }


#ifdef GPUWM_HOST_RATES
    // Host-only readback (tools/thompson_real_column_parity): every rate
    // as WRF's cp1 checkpoint publishes it, in RATES order.
    {
        const double r_[64] = {prw_vcd, pnc_wcd, pnc_wau, pnc_rcw, pnc_scw, pnc_gcw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, prr_wau, prr_rcw, prr_rcs, prr_rcg, prr_sml, prr_gml, prr_rci, prv_rev, pnr_wau, pnr_rcs, pnr_rcg, pnr_rci, pnr_sml, pnr_gml, pnr_rev, pnr_rcr, pnr_rfz, pri_inu, pni_inu, pri_ihm, pni_ihm, pri_wfz, pni_wfz, pri_rfz, pni_rfz, pri_ide, pni_ide, pri_rci, pni_rci, pni_sci, pni_iau, pri_iha, pni_iha, prs_iau, prs_sci, prs_rcs, prs_scw, prs_sde, prs_ihm, prs_ide, prg_scw, prg_rfz, prg_gde, prg_gcw, prg_rci, prg_rcs, prg_rcg, prg_ihm, png_scw, png_rcs, png_rcg, png_gde};
        for (int i_ = 0; i_ < 64; ++i_) HOST_RATE(i_, idx, r_[i_]);
    }
#endif
    // :2958-3183 tendencies (each starts from 0: :1650-1664).
    float t_tten, t_qvten, t_qcten, t_ncten, t_qiten, t_niten, t_qrten,
        t_nrten, t_qsten, t_qgten, t_ngten;
    {
        const float orho = 1.f / rho;
        const float lfus2 = TC_LSUB - lvap;
        t_qvten = (float)((double)0.f + (-pri_inu - pri_iha - pri_ide
            - prs_ide - prs_sde - prg_gde) * (double)orho);
        t_qcten = (float)((double)0.f + (-prr_wau - pri_wfz - prr_rcw
            - prs_scw - prg_scw - prg_gcw) * (double)orho);
        t_ncten = (float)((double)0.f + (-pnc_wau - pnc_rcw - pni_wfz
            - pnc_scw - pnc_gcw) * (double)orho);
        float xrc = fmaxf(TC_R1, (qc_e + t_qcten * dtsave) * rho);
        float xnc = fmaxf(2.f, (nc1d + t_ncten * dtsave) * rho);
        if (xrc > TC_R1) {
            nu_c = min(15, tc_nint(1000.E6f / xnc) + 2);
            lamc = (double)gfk_pow(xnc * TC_AM_R * TC_CCG(2, nu_c)
                                   * TC_OCG1(nu_c) / rc, TC_obmr);
            xDc = (float)((double)(3.0f + (float)nu_c + 1.f) / lamc);
            if (xDc < TC_D0C) {
                lamc = (double)(FDIV(TC_CCE(2, nu_c), TC_D0C));
                xnc = (float)((double)FDIV(TC_CCG(1, nu_c) * TC_OCG2(nu_c) * xrc,
                    TC_AM_R) * glibc_pow(lamc, 3.0));
                t_ncten = (xnc - nc1d * rho) * odts * orho;
            } else if (xDc > TC_D0R * 2.f) {
                lamc = (double)(FDIV(TC_CCE(2, nu_c), (TC_D0R * 2.f)));
                xnc = (float)((double)FDIV(TC_CCG(1, nu_c) * TC_OCG2(nu_c) * xrc,
                    TC_AM_R) * glibc_pow(lamc, 3.0));
                t_ncten = (xnc - nc1d * rho) * odts * orho;
            }
        } else {
            t_ncten = -nc1d * odts;
        }
        xnc = fmaxf(0.f, (nc1d + t_ncten * dtsave) * rho);
        if (xnc > TC_NT_C_MAX) {
            t_ncten = (TC_NT_C_MAX - nc1d * rho) * odts * orho;
        }

        t_qiten = (float)((double)0.f + (pri_inu + pri_iha + pri_ihm
            + pri_wfz + pri_rfz + pri_ide - prs_iau - prs_sci - pri_rci)
            * (double)orho);
        t_niten = (float)((double)0.f + (pni_inu + pni_iha + pni_ihm
            + pni_wfz + pni_rfz + pni_ide - pni_iau - pni_sci - pni_rci)
            * (double)orho);
        const float xri = fmaxf(TC_R1, (qi_e + t_qiten * dtsave) * rho);
        float xni = fmaxf(TC_R2, (ni_e + t_niten * dtsave) * rho);
        if (xri > TC_R1) {
            lami = (double)gfk_pow(TC_AM_I * TC_CIG(2) * TC_oig1 * xni / xri,
                                   TC_obmi);
            ilami = 1. / lami;
            xDi = (float)((double)(3.0f + 0.0f + 1.f) * ilami);
            if (xDi < 5.E-6f) {
                lami = (double)(FDIV(TC_CIE(2), 5.E-6f));
                xni = (float)fmin(999.E3, (double)FDIV(TC_CIG(1) * TC_oig2 * xri,
                    TC_AM_I) * glibc_pow(lami, 3.0));
                t_niten = (xni - ni_e * rho) * odts * orho;
            } else if (xDi > 300.E-6f) {
                lami = (double)(FDIV(TC_CIE(2), 300.E-6f));
                xni = (float)((double)FDIV(TC_CIG(1) * TC_oig2 * xri, TC_AM_I)
                    * glibc_pow(lami, 3.0));
                t_niten = (xni - ni_e * rho) * odts * orho;
            }
        } else {
            t_niten = -ni_e * odts;
        }
        xni = fmaxf(0.f, (ni_e + t_niten * dtsave) * rho);
        if (xni > 999.E3f) {
            t_niten = (999.E3f - ni_e * rho) * odts * orho;
        }

        t_qrten = (float)((double)0.f + (prr_wau + prr_rcw + prr_sml
            + prr_gml + prr_rcs + prr_rcg - prg_rfz - pri_rfz - prr_rci)
            * (double)orho);
        t_nrten = (float)((double)0.f + (pnr_wau + pnr_sml + pnr_gml
            - (pnr_rfz + pnr_rcr + pnr_rcg + pnr_rcs + pnr_rci + pni_rfz))
            * (double)orho);
        const float xrr = fmaxf(TC_R1, (qr_e + t_qrten * dtsave) * rho);
        float xnr = fmaxf(TC_R2, (nr_e + t_nrten * dtsave) * rho);
        if (xrr > TC_R1) {
            lamr = (double)gfk_pow(TC_AM_R * TC_CRG(3) * TC_org2 * xnr / xrr,
                                   TC_obmr);
            float mvd = (float)((double)(3.0f + 0.0f + 0.672f) / lamr);
            if (mvd > 2.5E-3f) {
                mvd = 2.5E-3f;
                lamr = (double)((3.0f + 0.0f + 0.672f) / mvd);
                xnr = (float)(DDIV((double)(TC_CRG(2) * TC_org3 * xrr)
                    * glibc_pow(lamr, 3.0), (double)TC_AM_R));
                t_nrten = (xnr - nr_e * rho) * odts * orho;
            } else if (mvd < TC_D0R * 0.75f) {
                mvd = TC_D0R * 0.75f;
                lamr = (double)((3.0f + 0.0f + 0.672f) / mvd);
                xnr = (float)(DDIV((double)(TC_CRG(2) * TC_org3 * xrr)
                    * glibc_pow(lamr, 3.0), (double)TC_AM_R));
                t_nrten = (xnr - nr_e * rho) * odts * orho;
            }
        } else {
            t_qrten = -qr_e * odts;
            t_nrten = -nr_e * odts;
        }

        t_qsten = (float)((double)0.f + (prs_iau + prs_sde + prs_sci
            + prs_scw + prs_rcs + prs_ide - prs_ihm - prr_sml)
            * (double)orho);
        t_qgten = (float)((double)0.f + (prg_scw + prg_rfz + prg_gde
            + prg_rcg + prg_gcw + prg_rci + prg_rcs - prg_ihm - prr_gml)
            * (double)orho);
        t_ngten = (float)((double)0.f + (png_scw + pnr_rfz - png_rcg
            + pnr_rci + png_rcs + png_gde - pnr_gml) * (double)orho);
        const float xrg = fmaxf(TC_R1, (qg_e + t_qgten * dtsave) * rho);
        float xng = fmaxf(TC_R2, (ng_e + t_ngten * dtsave) * rho);
        if (xrg > TC_R1) {
            lamg = (double)gfk_pow(TC_AM_G5 * TC_CGG(3, 1) * TC_ogg2 * xng
                                   / xrg, TC_obmg);
            float mvd = (float)((double)(3.0f + 0.0f + 0.672f) / lamg);
            if (mvd > 25.4E-3f) {
                mvd = 25.4E-3f;
                lamg = (double)((3.0f + 0.0f + 0.672f) / mvd);
                xng = (float)(DDIV((double)(TC_CGG(2, 1) * TC_ogg3 * xrg)
                    * glibc_pow(lamg, 3.0), (double)TC_AM_G5));
                t_ngten = (xng - ng_e * rho) * odts * orho;
            } else if (mvd < TC_D0R) {
                mvd = TC_D0R;
                lamg = (double)((3.0f + 0.0f + 0.672f) / mvd);
                xng = (float)(DDIV((double)(TC_CGG(2, 1) * TC_ogg3 * xrg)
                    * glibc_pow(lamg, 3.0), (double)TC_AM_G5));
                t_ngten = (xng - ng_e * rho) * odts * orho;
            }
        } else {
            t_qgten = -qg_e * odts;
            t_ngten = -ng_e * odts;
        }

        if (temp < TC_T0) {
            t_tten = (float)((double)0.f + ((double)(TC_LSUB * ocp)
                * (pri_inu + pri_ide + prs_ide + prs_sde + prg_gde + pri_iha)
                + (double)(lfus2 * ocp) * (pri_wfz + pri_rfz + prg_rfz
                    + prs_scw + prg_scw + prg_gcw + prg_rcs + prs_rcs
                    + prr_rci + prg_rcg)) * (double)orho * 1.0);
        } else {
            t_tten = (float)((double)0.f + ((double)(TC_LFUS * ocp)
                * (-prr_sml - prr_gml - prr_rcg - prr_rcs)
                + (double)(TC_LSUB * ocp) * (prs_sde + prg_gde))
                * (double)orho * 1.0);
        }
    }

    // :3188-3264 TAU+1 (classic graupel :3284-3304).
    temp = temp_in + DT * t_tten;
    float otemp = 1.f / temp;
    tempc = temp - 273.15f;
    qv = fmaxf(1.E-10f, qv_in + DT * t_qvten);
    rho = 0.622f * pres / (TC_R * temp * (qv + 0.622f));
    rhof = sqrtf(TC_RHO_NOT / rho);
    rhof2 = sqrtf(rhof);
    qvs = tc_rslf(pres, temp);
    ssatw = qv / qvs - 1.f;
    if (fabsf(ssatw) < TC_EPS) ssatw = 0.0f;
    diffu = 2.11E-5f * gfk_pow(FDIV(temp, 273.15f), 1.94f) * (101325.f / pres);
    if (tempc >= 0.0f) {
        visco = (1.718f + 0.0049f * tempc) * 1.0E-5f;
    } else {
        visco = (1.718f + 0.0049f * tempc - 1.2E-5f * tempc * tempc)
            * 1.0E-5f;
    }
    vsc2 = sqrtf(rho / visco);
    lvap = TC_LVAP0 + (2106.0f - 4218.0f) * tempc;
    tcond = (5.69f + 0.0168f * tempc) * 1.0E-5f * 418.936f;
    ocp = 1.f / (TC_CP * (1.f + 0.887f * qv));
    const float lvt2 = lvap * lvap * ocp * TC_ORV * otemp * otemp;

    if ((qc_e + t_qcten * DT) > TC_R1) {
        rc = (qc_e + t_qcten * DT) * rho;
        nc = TC_NT_C;
        L_qc = true;
    } else {
        rc = TC_R1;
        nc = 2.f;
        L_qc = false;
    }
    if ((qi_e + t_qiten * DT) > TC_R1) {
        ri = (qi_e + t_qiten * DT) * rho;
        ni = fmaxf(TC_R2, (ni_e + t_niten * DT) * rho);
        L_qi = true;
    } else {
        ri = TC_R1;
        ni = TC_R2;
        L_qi = false;
    }
    if ((qr_e + t_qrten * DT) > TC_R1) {
        rr = (qr_e + t_qrten * DT) * rho;
        nr = fmaxf(TC_R2, (nr_e + t_nrten * DT) * rho);
        L_qr = true;
        lamr = (double)gfk_pow(TC_AM_R * TC_CRG(3) * TC_org2 * nr / rr,
                               TC_obmr);
        mvd_r = (float)((double)(3.0f + 0.0f + 0.672f) / lamr);
        if (mvd_r > 2.5E-3f) {
            mvd_r = 2.5E-3f;
            lamr = (double)((3.0f + 0.0f + 0.672f) / mvd_r);
            nr = (float)(DDIV((double)(TC_CRG(2) * TC_org3 * rr)
                 * glibc_pow(lamr, 3.0), (double)TC_AM_R));
        } else if (mvd_r < TC_D0R * 0.75f) {
            mvd_r = TC_D0R * 0.75f;
            lamr = (double)((3.0f + 0.0f + 0.672f) / mvd_r);
            nr = (float)(DDIV((double)(TC_CRG(2) * TC_org3 * rr)
                 * glibc_pow(lamr, 3.0), (double)TC_AM_R));
        }
    } else {
        rr = TC_R1;
        nr = TC_R2;
        L_qr = false;
    }
    if ((qs_e + t_qsten * DT) > TC_R1) {
        rs = (qs_e + t_qsten * DT) * rho;
        L_qs = true;
    } else {
        rs = TC_R1;
        L_qs = false;
    }
    if ((qg_e + t_qgten * DT) > TC_R1) {
        rg = (qg_e + t_qgten * DT) * rho;
        const float ygra1 = gfk_log10(fmaxf(1.E-9f, rg));
        float zans1 = 3.0f + 2.f / 7.f * (ygra1 + 8.f);
        zans1 = fmaxf(2.f, fminf(zans1, 6.f));
        const double N0_exp = (double)gfk_pow(10.f, zans1);
        const double lam_exp = glibc_pow(N0_exp * (double)TC_AM_G5
            * (double)TC_CGG(1, 1) / (double)rg, (double)TC_oge1);
        lamg = lam_exp * (double)gfk_pow(TC_CGG(3, 1) * TC_ogg2 * TC_ogg1,
                                         TC_obmg);
        ng = (float)(DDIV((double)(TC_CGG(2, 1) * TC_ogg3 * rg)
             * glibc_pow(lamg, 3.0), (double)TC_AM_G5));
        // WRF v4.6.1's non-hail-aware branch here (:3287-3304), line for
        // line: it never sets L_qg(k) = .true.; it can only clear it.
        // Graupel the sources make at a level that had none at entry stays
        // unflagged, and when no level had graupel at entry the column's
        // graupel fallout (:3744, :3902, gated on ANY(L_qg)) is skipped for
        // the step.  Every other species, WRF's hail-aware branch (:3269)
        // and v4.4.2 (:2980-2986) set the flag, so this is a WRF defect;
        // this unit is the WRF transcription (GPUWM_WRF_EXACT=1, exact
        // mode) and keeps it.  The forecast's default adapter sets the flag
        // (tools/thompson_mp8_column_oracle/repair_wrf461.py, lqg).
    } else {
        rg = TC_R1;
        ng = TC_R2;
        L_qg = false;
    }

    // :3311-3378 snow moments and graupel slope after the update.
    smo2 = 0.f; smob = 0.f; smoc = 0.f;
    if (L_qs) {
        const float tc0 = fminf(-0.1f, temp - 273.15f);
        smob = rs * TC_oams;
        smo2 = smob;
        const float a_ = gfk_pow(10.0f, tc_poly(tc_sa, tc0, TC_CSE(1)));
        const float b_ = tc_poly(tc_sb, tc0, TC_CSE(1));
        smoc = a_ * gfk_pow(smo2, b_);
        // smod (:3356-3366) feeds no classic output.
    }
    lamg = (double)gfk_pow(TC_AM_G5 * TC_CGG(3, 1) * TC_ogg2 * ng / rg,
                           TC_obmg);
    ilamg = 1. / lamg;
    lamr = (double)gfk_pow(TC_AM_R * TC_CRG(3) * TC_org2 * nr / rr, TC_obmr);
    ilamr = 1. / lamr;
    mvd_r = (float)((double)(3.0f + 0.0f + 0.672f) / lamr);
    N0_r = (double)(nr * TC_org2) * glibc_pow(lamr, (double)TC_CRE(2));

    // :3399-3494 condensation and evaporation of cloud water.
    {
        const float orho = 1.f / rho;
        if ((ssatw > TC_EPS) || (ssatw < -TC_EPS && L_qc)) {
            float clap = (qv - qvs) / (1.f + lvt2 * qvs);
            for (int n = 1; n <= 3; ++n) {
                const float fcd = qvs * gfk_exp(lvt2 * clap) - qv + clap;
                const float dfcd = qvs * lvt2 * gfk_exp(lvt2 * clap) + 1.f;
                clap = clap - fcd / dfcd;
            }
            const float xrc = rc + clap * rho;
            if (xrc > TC_R1) {
                prw_vcd = (double)(clap * odt);
                if (clap > TC_EPS) {
                    const float xnc = TC_NT_C;
                    pnc_wcd = (double)(0.5f * (xnc - nc + fabsf(xnc - nc))
                                       * odts * orho);
                }
                // :3423 evaporation branch: aerosol-aware only.
            } else {
                prw_vcd = (double)(-rc * orho * odt);
                pnc_wcd = (double)(-nc * orho * odt);
            }
            t_qvten = (float)((double)t_qvten - prw_vcd);
            t_qcten = (float)((double)t_qcten + prw_vcd);
            t_ncten = (float)((double)t_ncten + pnc_wcd);
            t_tten = (float)((double)t_tten + (double)(lvap * ocp) * prw_vcd
                             * (double)(1 - 0));
            rc = fmaxf(TC_R1, (qc_e + DT * t_qcten) * rho);
            if (rc == TC_R1) L_qc = false;
            nc = TC_NT_C;
            qv = fmaxf(1.E-10f, qv_in + DT * t_qvten);
            temp = temp_in + DT * t_tten;
            rho = 0.622f * pres / (TC_R * temp * (qv + 0.622f));
            qvs = tc_rslf(pres, temp);
            ssatw = qv / qvs - 1.f;
        }
    }

    // :3500-3574 rain evaporation.
    if ((ssatw < -TC_EPS) && L_qr && (!(prw_vcd > 0.))) {
        tempc = temp - 273.15f;
        otemp = 1.f / temp;
        const float orho = 1.f / rho;
        rhof = sqrtf(TC_RHO_NOT * orho);
        rhof2 = sqrtf(rhof);
        diffu = 2.11E-5f * gfk_pow(FDIV(temp, 273.15f), 1.94f) * (101325.f / pres);
        if (tempc >= 0.0f) {
            visco = (1.718f + 0.0049f * tempc) * 1.0E-5f;
        } else {
            visco = (1.718f + 0.0049f * tempc - 1.2E-5f * tempc * tempc)
                * 1.0E-5f;
        }
        vsc2 = sqrtf(rho / visco);
        lvap = TC_LVAP0 + (2106.0f - 4218.0f) * tempc;
        tcond = (5.69f + 0.0168f * tempc) * 1.0E-5f * 418.936f;
        ocp = 1.f / (TC_CP * (1.f + 0.887f * qv));
        const float rvs = rho * qvs;
        const float rvs_p = rvs * otemp * (lvap * otemp * TC_ORV - 1.f);
        const float rvs_pp = rvs * (otemp * (lvap * otemp * TC_ORV - 1.f)
            * otemp * (lvap * otemp * TC_ORV - 1.f)
            + (-2.f * lvap * otemp * otemp * otemp * TC_ORV)
            + otemp * otemp);
        const float gamsc = lvap * diffu / tcond * rvs_p;
        float alphsc = 0.5f * (gamsc / (1.f + gamsc)) * (gamsc / (1.f + gamsc))
            * rvs_pp / rvs_p * rvs / rvs_p;
        alphsc = fmaxf(1.E-9f, alphsc);
        const float xsat = fminf(-1.E-9f, ssatw);
        const float t1_evap = 2.f * TC_PI * (1.0f - alphsc * xsat
            + 2.f * alphsc * alphsc * xsat * xsat
            - 5.f * alphsc * alphsc * alphsc * xsat * xsat * xsat)
            / (1.f + gamsc);
        lamr = 1. / ilamr;
        if (qv / qvs < 0.95f && rr * orho <= 1.E-8f) {
            prv_rev = (double)(rr * orho * odts);
        } else {
            prv_rev = (double)(t1_evap * diffu * (-ssatw)) * N0_r
                * (double)rvs * ((double)TC_t1_qr_ev
                    * glibc_pow(ilamr, (double)TC_CRE(10))
                + (double)(TC_t2_qr_ev * vsc2 * rhof2)
                  * glibc_pow(lamr + (double)(0.5f * TC_FV_R),
                              (double)(-TC_CRE(11))));
            const float rate_max = fminf((rr * orho * odts),
                                         (qvs - qv) * odts);
            prv_rev = fmin((double)rate_max, prv_rev * (double)orho);
            if (prr_gml > 0.0) {
                const float eva_factor = fminf(1.0f,
                    0.01f + (0.99f - 0.01f) * FDIV(tempc, 20.0f));
                prv_rev = prv_rev * (double)eva_factor;
            }
        }
        pnr_rev = fmin((double)(nr * 0.99f * orho * odts),
                       prv_rev * (double)nr / (double)rr);
        t_qrten = (float)((double)t_qrten - prv_rev);
        t_qvten = (float)((double)t_qvten + prv_rev);
        t_nrten = (float)((double)t_nrten - pnr_rev);
        t_tten = (float)((double)t_tten - (double)(lvap * ocp) * prv_rev
                         * (double)(1 - 0));
        rr = fmaxf(TC_R1, (qr_e + DT * t_qrten) * rho);
        qv = fmaxf(1.E-10f, qv_in + DT * t_qvten);
        nr = fmaxf(TC_R2, (nr_e + DT * t_nrten) * rho);
        temp = temp_in + DT * t_tten;
        rho = 0.622f * pres / (TC_R * temp * (qv + 0.622f));
    }


#ifdef GPUWM_HOST_RATES
    HOST_RATE(0, idx, prw_vcd);
    HOST_RATE(1, idx, pnc_wcd);
    HOST_RATE(19, idx, prv_rev);
    HOST_RATE(26, idx, pnr_rev);
#endif
    TEN(TC_TTEN) = t_tten;
    TEN(TC_QVTEN) = t_qvten;
    TEN(TC_QCTEN) = t_qcten;
    TEN(TC_NCTEN) = t_ncten;
    TEN(TC_QITEN) = t_qiten;
    TEN(TC_NITEN) = t_niten;
    TEN(TC_QRTEN) = t_qrten;
    TEN(TC_NRTEN) = t_nrten;
    TEN(TC_QSTEN) = t_qsten;
    TEN(TC_QGTEN) = t_qgten;
    TEN(TC_NGTEN) = t_ngten;
    WORK(TC_W_RHO) = rho;
    WORK(TC_W_RHOF) = rhof;
    WORK(TC_W_RC) = rc;
    WORK(TC_W_NC) = nc;
    WORK(TC_W_RI) = ri;
    WORK(TC_W_NI) = ni;
    WORK(TC_W_RR) = rr;
    WORK(TC_W_NR) = nr;
    WORK(TC_W_RS) = rs;
    WORK(TC_W_RG) = rg;
    WORK(TC_W_NG) = ng;
    WORK(TC_W_BOOST) = vts_boost;
    WORK(TC_W_TEMP) = temp;
    WORK(TC_W_OCP) = ocp;
    WORK(TC_W_LVAP) = lvap;
    WORK(TC_W_SMOB) = smob;
    WORK(TC_W_SMOC) = smoc;
    flags[idx] = (L_qc ? TC_LQC : 0) | (L_qi ? TC_LQI : 0)
        | (L_qr ? TC_LQR : 0) | (L_qs ? TC_LQS : 0) | (L_qg ? TC_LQG : 0)
        | (prr_sml > 0.0 ? TC_SML : 0)
        | (rcg_past_end ? TC_RCG_PAST_END : 0);
#undef TEN
#undef WORK
}

// Column scratch planes, each (nz + 1) levels: nine fall-speed profiles
// (:3599-3609) and five flux profiles.
#define TC_V_TR 0
#define TC_V_TNR 1
#define TC_V_TI 2
#define TC_V_TNI 3
#define TC_V_TS 4
#define TC_V_TG 5
#define TC_V_TNG 6
#define TC_V_TC 7
#define TC_V_TNC 8
#define TC_S_A 9
#define TC_S_N 10
#define TC_NCOLPLANES 11

// ---------------------------------------------------------------------
// Kernel 3 (one thread per column): mp_thompson :3596-4078 and the
// driver's copy-out (:1294-1358).
// ---------------------------------------------------------------------
extern "C" __global__ void tc_column(
    float* __restrict__ t1d, const float* __restrict__ pii,
    float* __restrict__ th,
    float* __restrict__ qv1d, float* __restrict__ qc1d,
    float* __restrict__ qi1d, float* __restrict__ qr1d,
    float* __restrict__ qs1d, float* __restrict__ qg1d,
    float* __restrict__ ni1d, float* __restrict__ nr1d,
    float* __restrict__ ng1d,
    const float* __restrict__ w, const float* __restrict__ dzq,
    float* __restrict__ ten, float* __restrict__ work,
    const int* __restrict__ flags,
    float* __restrict__ colbuf,
    float* __restrict__ rainnc, float* __restrict__ rainncv,
    float* __restrict__ snownc, float* __restrict__ snowncv,
    float* __restrict__ graupelnc, float* __restrict__ graupelncv,
    float* __restrict__ sr,
    float dt, int nz, int columns)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= columns) return;
    const int size = nz * columns;
#define AT(k) ((size_t)(k) * (size_t)columns + (size_t)col)
#define TEN(slot, k) ten[(size_t)(slot) * (size_t)size + AT(k)]
#define WORK(slot, k) work[(size_t)(slot) * (size_t)size + AT(k)]
#define CB(plane, k) colbuf[((size_t)(plane) * (size_t)(nz + 1) + (size_t)(k)) \
                            * (size_t)columns + (size_t)col]
    const float DT = dt;
    const float odt = 1.f / dt;
    const int kts = 0, kte = nz - 1;   // zero-based Fortran kts..kte

    float pptrain = 0.f, pptsnow = 0.f, pptgraul = 0.f, pptice = 0.f;
    const bool no_micro = (flags[AT(0)] & TC_DONE) != 0;

    if (!no_micro) {
        bool any_qr = false, any_qc = false, any_qi = false, any_qs = false,
            any_qg = false;
        for (int k = kts; k <= kte; ++k) {
            const int f = flags[AT(k)];
            any_qr |= (f & TC_LQR) != 0;
            any_qc |= (f & TC_LQC) != 0;
            any_qi |= (f & TC_LQI) != 0;
            any_qs |= (f & TC_LQS) != 0;
            any_qg |= (f & TC_LQG) != 0;
        }
        int nstep = 0;
        float onstep[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        int ksed1[5] = {kts, kts, kts, kts, kts};
        for (int k = kte + 1; k >= kts; --k) {
            for (int p = TC_V_TR; p <= TC_V_TNC; ++p) CB(p, k) = 0.f;
        }

        if (any_qr) {
            for (int k = kte; k >= kts; --k) {
                float vtr = 0.f;
                const float rho = WORK(TC_W_RHO, k);
                const float rhof = sqrtf(TC_RHO_NOT / rho);
                WORK(TC_W_RHOF, k) = rhof;
                const float rr = WORK(TC_W_RR, k);
                if (rr > TC_R1) {
                    const float nr = WORK(TC_W_NR, k);
                    const double lamr = (double)gfk_pow(
                        TC_AM_R * TC_CRG(3) * TC_org2 * nr / rr, TC_obmr);
                    vtr = (float)((double)(rhof * TC_AV_R * TC_CRG(6)
                                           * TC_org3)
                        * glibc_pow(lamr, (double)TC_CRE(3))
                        * glibc_pow(lamr + (double)TC_FV_R,
                                    (double)(-TC_CRE(6))));
                    CB(TC_V_TR, k) = vtr;
                    vtr = (float)((double)FDIV(rhof * TC_AV_R * TC_CRG(7),
                                           TC_CRG(12))
                        * glibc_pow(lamr, (double)TC_CRE(12))
                        * glibc_pow(lamr + (double)TC_FV_R,
                                    (double)(-TC_CRE(7))));
                    CB(TC_V_TNR, k) = vtr;
                } else {
                    CB(TC_V_TR, k) = CB(TC_V_TR, k + 1);
                    CB(TC_V_TNR, k) = CB(TC_V_TNR, k + 1);
                }
                const float vmax = fmaxf(CB(TC_V_TR, k), CB(TC_V_TNR, k));
                if (vmax > 1.E-3f) {
                    ksed1[0] = max(ksed1[0], k);
                    const float delta_tp = dzq[AT(k)] / vmax;
                    nstep = max(nstep, (int)(DT / delta_tp + 1.f));
                }
            }
            if (ksed1[0] == kte) ksed1[0] = kte - 1;
            if (nstep > 0) onstep[0] = 1.f / (float)nstep;
        }

        if (any_qc) {
            float hgt_agl = 0.f;
            for (int k = kts; k <= kte - 1; ++k) {
                if (WORK(TC_W_RC, k) > TC_R2) ksed1[4] = k;
                hgt_agl = hgt_agl + dzq[AT(k)];
                if (hgt_agl > 500.0f) break;
            }
            for (int k = ksed1[4]; k >= kts; --k) {
                const float rc = WORK(TC_W_RC, k);
                if (rc > TC_R1 && w[AT(k)] < 1.E-1f) {
                    const float nc = WORK(TC_W_NC, k);
                    const float rhof = WORK(TC_W_RHOF, k);
                    const int nu_c = min(15, tc_nint(1000.E6f / nc) + 2);
                    const double lamc = (double)gfk_pow(nc * TC_AM_R
                        * TC_CCG(2, nu_c) * TC_OCG1(nu_c) / rc, TC_obmr);
                    const double ilamc = 1. / lamc;
                    CB(TC_V_TC, k) = (float)((double)(rhof * TC_AV_C
                        * TC_CCG(5, nu_c) * TC_OCG2(nu_c)) * (ilamc * ilamc));
                    CB(TC_V_TNC, k) = (float)((double)(rhof * TC_AV_C
                        * TC_CCG(4, nu_c) * TC_OCG1(nu_c)) * (ilamc * ilamc));
                }
            }
        }

        if (any_qi) {
            nstep = 0;
            for (int k = kte; k >= kts; --k) {
                const float ri = WORK(TC_W_RI, k);
                if (ri > TC_R1) {
                    const float ni = WORK(TC_W_NI, k);
                    const float rhof = WORK(TC_W_RHOF, k);
                    const double lami = (double)gfk_pow(TC_AM_I * TC_CIG(2)
                        * TC_oig1 * ni / ri, TC_obmi);
                    const double ilami = 1. / lami;
                    CB(TC_V_TI, k) = (float)((double)(rhof * TC_AV_I
                        * TC_CIG(3) * TC_oig2) * ilami);
                    CB(TC_V_TNI, k) = (float)((double)FDIV(rhof * TC_AV_I
                        * TC_CIG(6), TC_CIG(7)) * ilami);
                } else {
                    CB(TC_V_TI, k) = CB(TC_V_TI, k + 1);
                    CB(TC_V_TNI, k) = CB(TC_V_TNI, k + 1);
                }
                if (CB(TC_V_TI, k) > 1.E-3f) {
                    ksed1[1] = max(ksed1[1], k);
                    const float delta_tp = dzq[AT(k)] / CB(TC_V_TI, k);
                    nstep = max(nstep, (int)(DT / delta_tp + 1.f));
                }
            }
            if (ksed1[1] == kte) ksed1[1] = kte - 1;
            if (nstep > 0) onstep[1] = 1.f / (float)nstep;
        }

        if (any_qs) {
            nstep = 0;
            for (int k = kte; k >= kts; --k) {
                const float rs = WORK(TC_W_RS, k);
                if (rs > TC_R1) {
                    const float rhof = WORK(TC_W_RHOF, k);
                    const float xDs = WORK(TC_W_SMOC, k) / WORK(TC_W_SMOB, k);
                    const float Mrat = 1.f / xDs;
                    float ils1 = 1.f / (Mrat * TC_LAM0 + TC_FV_S);
                    float ils2 = 1.f / (Mrat * TC_LAM1 + TC_FV_S);
                    const float t1_vts = TC_KAP0 * TC_CSG(4)
                        * gfk_pow(ils1, TC_CSE(4));
                    const float t2_vts = TC_KAP1 * gfk_pow(Mrat, TC_MU_S)
                        * TC_CSG(10) * gfk_pow(ils2, TC_CSE(10));
                    ils1 = 1.f / (Mrat * TC_LAM0);
                    ils2 = 1.f / (Mrat * TC_LAM1);
                    const float t3_vts = TC_KAP0 * TC_CSG(1)
                        * gfk_pow(ils1, TC_CSE(1));
                    const float t4_vts = TC_KAP1 * gfk_pow(Mrat, TC_MU_S)
                        * TC_CSG(7) * gfk_pow(ils2, TC_CSE(7));
                    const float vts = rhof * TC_AV_S * (t1_vts + t2_vts)
                        / (t3_vts + t4_vts);
                    if ((flags[AT(k)] & TC_SML) != 0) {
                        const float SR = rs / (rs + WORK(TC_W_RR, k));
                        CB(TC_V_TS, k) = vts * SR + (1.f - SR) * CB(TC_V_TR, k);
                    } else {
                        CB(TC_V_TS, k) = vts * WORK(TC_W_BOOST, k);
                    }
                } else {
                    CB(TC_V_TS, k) = CB(TC_V_TS, k + 1);
                }
                if (CB(TC_V_TS, k) > 1.E-3f) {
                    ksed1[2] = max(ksed1[2], k);
                    const float delta_tp = dzq[AT(k)] / CB(TC_V_TS, k);
                    nstep = max(nstep, (int)(DT / delta_tp + 1.f));
                }
            }
            if (ksed1[2] == kte) ksed1[2] = kte - 1;
            if (nstep > 0) onstep[2] = 1.f / (float)nstep;
        }

        if (any_qg) {
            nstep = 0;
            const float afall = 442.0f;    // av_g_old
            const float bfall = 0.89f;     // bv_g_old
            for (int k = kte; k >= kts; --k) {
                const float rg = WORK(TC_W_RG, k);
                if (rg > TC_R1) {
                    const float rhof = WORK(TC_W_RHOF, k);
                    // :3373-3375 on the TAU+1 graupel, which nothing
                    // after it changes before the fall speeds.
                    const double ilamg = 1. / (double)gfk_pow(
                        TC_AM_G5 * TC_CGG(3, 1) * TC_ogg2 * WORK(TC_W_NG, k)
                        / rg, TC_obmg);
                    CB(TC_V_TG, k) = (float)((double)(rhof * afall
                        * TC_CGG(6, TC_IDX_BG1) * TC_ogg3)
                        * glibc_pow(ilamg, (double)bfall));
                    CB(TC_V_TNG, k) = (float)((double)FDIV(rhof * afall
                        * TC_CGG(7, TC_IDX_BG1), TC_CGG(12, TC_IDX_BG1))
                        * glibc_pow(ilamg, (double)bfall));
                } else {
                    CB(TC_V_TG, k) = CB(TC_V_TG, k + 1);
                    CB(TC_V_TNG, k) = CB(TC_V_TNG, k + 1);
                }
                if (CB(TC_V_TG, k) > 1.E-3f) {
                    ksed1[3] = max(ksed1[3], k);
                    const float delta_tp = dzq[AT(k)] / CB(TC_V_TG, k);
                    nstep = max(nstep, (int)(DT / delta_tp + 1.f));
                }
            }
            if (ksed1[3] == kte) ksed1[3] = kte - 1;
            if (nstep > 0) onstep[3] = 1.f / (float)nstep;
        }

        // :3790-3820 rain.
        if (any_qr) {
            nstep = tc_nint(1.f / onstep[0]);
            for (int n = 1; n <= nstep; ++n) {
                for (int k = kte; k >= kts; --k) {
                    CB(TC_S_A, k) = CB(TC_V_TR, k) * WORK(TC_W_RR, k);
                    CB(TC_S_N, k) = CB(TC_V_TNR, k) * WORK(TC_W_NR, k);
                }
                int k = kte;
                float odzq = 1.f / dzq[AT(k)];
                float orho = 1.f / WORK(TC_W_RHO, k);
                TEN(TC_QRTEN, k) = TEN(TC_QRTEN, k)
                    - CB(TC_S_A, k) * odzq * onstep[0] * orho;
                TEN(TC_NRTEN, k) = TEN(TC_NRTEN, k)
                    - CB(TC_S_N, k) * odzq * onstep[0] * orho;
                WORK(TC_W_RR, k) = fmaxf(TC_R1, WORK(TC_W_RR, k)
                    - CB(TC_S_A, k) * odzq * DT * onstep[0]);
                WORK(TC_W_NR, k) = fmaxf(TC_R2, WORK(TC_W_NR, k)
                    - CB(TC_S_N, k) * odzq * DT * onstep[0]);
                for (k = ksed1[0]; k >= kts; --k) {
                    odzq = 1.f / dzq[AT(k)];
                    orho = 1.f / WORK(TC_W_RHO, k);
                    TEN(TC_QRTEN, k) = TEN(TC_QRTEN, k)
                        + (CB(TC_S_A, k + 1) - CB(TC_S_A, k))
                        * odzq * onstep[0] * orho;
                    TEN(TC_NRTEN, k) = TEN(TC_NRTEN, k)
                        + (CB(TC_S_N, k + 1) - CB(TC_S_N, k))
                        * odzq * onstep[0] * orho;
                    WORK(TC_W_RR, k) = fmaxf(TC_R1, WORK(TC_W_RR, k)
                        + (CB(TC_S_A, k + 1) - CB(TC_S_A, k))
                        * odzq * DT * onstep[0]);
                    WORK(TC_W_NR, k) = fmaxf(TC_R2, WORK(TC_W_NR, k)
                        + (CB(TC_S_N, k + 1) - CB(TC_S_N, k))
                        * odzq * DT * onstep[0]);
                }
                if (WORK(TC_W_RR, kts) > TC_R1 * 1000.f)
                    pptrain = pptrain + CB(TC_S_A, kts) * DT * onstep[0];
            }
        }

        // :3824-3837 cloud water (one step).
        if (any_qc) {
            for (int k = kte; k >= kts; --k) {
                CB(TC_S_A, k) = CB(TC_V_TC, k) * WORK(TC_W_RC, k);
                CB(TC_S_N, k) = CB(TC_V_TNC, k) * WORK(TC_W_NC, k);
            }
            for (int k = ksed1[4]; k >= kts; --k) {
                const float odzq = 1.f / dzq[AT(k)];
                const float orho = 1.f / WORK(TC_W_RHO, k);
                TEN(TC_QCTEN, k) = TEN(TC_QCTEN, k)
                    + (CB(TC_S_A, k + 1) - CB(TC_S_A, k)) * odzq * orho;
                TEN(TC_NCTEN, k) = TEN(TC_NCTEN, k)
                    + (CB(TC_S_N, k + 1) - CB(TC_S_N, k)) * odzq * orho;
                WORK(TC_W_RC, k) = fmaxf(TC_R1, WORK(TC_W_RC, k)
                    + (CB(TC_S_A, k + 1) - CB(TC_S_A, k)) * odzq * DT);
                WORK(TC_W_NC, k) = fmaxf(10.f, WORK(TC_W_NC, k)
                    + (CB(TC_S_N, k + 1) - CB(TC_S_N, k)) * odzq * DT);
            }
        }

        // :3841-3871 cloud ice.
        if (any_qi) {
            nstep = tc_nint(1.f / onstep[1]);
            for (int n = 1; n <= nstep; ++n) {
                for (int k = kte; k >= kts; --k) {
                    CB(TC_S_A, k) = CB(TC_V_TI, k) * WORK(TC_W_RI, k);
                    CB(TC_S_N, k) = CB(TC_V_TNI, k) * WORK(TC_W_NI, k);
                }
                int k = kte;
                float odzq = 1.f / dzq[AT(k)];
                float orho = 1.f / WORK(TC_W_RHO, k);
                TEN(TC_QITEN, k) = TEN(TC_QITEN, k)
                    - CB(TC_S_A, k) * odzq * onstep[1] * orho;
                TEN(TC_NITEN, k) = TEN(TC_NITEN, k)
                    - CB(TC_S_N, k) * odzq * onstep[1] * orho;
                WORK(TC_W_RI, k) = fmaxf(TC_R1, WORK(TC_W_RI, k)
                    - CB(TC_S_A, k) * odzq * DT * onstep[1]);
                WORK(TC_W_NI, k) = fmaxf(TC_R2, WORK(TC_W_NI, k)
                    - CB(TC_S_N, k) * odzq * DT * onstep[1]);
                for (k = ksed1[1]; k >= kts; --k) {
                    odzq = 1.f / dzq[AT(k)];
                    orho = 1.f / WORK(TC_W_RHO, k);
                    TEN(TC_QITEN, k) = TEN(TC_QITEN, k)
                        + (CB(TC_S_A, k + 1) - CB(TC_S_A, k))
                        * odzq * onstep[1] * orho;
                    TEN(TC_NITEN, k) = TEN(TC_NITEN, k)
                        + (CB(TC_S_N, k + 1) - CB(TC_S_N, k))
                        * odzq * onstep[1] * orho;
                    WORK(TC_W_RI, k) = fmaxf(TC_R1, WORK(TC_W_RI, k)
                        + (CB(TC_S_A, k + 1) - CB(TC_S_A, k))
                        * odzq * DT * onstep[1]);
                    WORK(TC_W_NI, k) = fmaxf(TC_R2, WORK(TC_W_NI, k)
                        + (CB(TC_S_N, k + 1) - CB(TC_S_N, k))
                        * odzq * DT * onstep[1]);
                }
                if (WORK(TC_W_RI, kts) > TC_R1 * 1000.f)
                    pptice = pptice + CB(TC_S_A, kts) * DT * onstep[1];
            }
        }

        // :3875-3898 snow.
        if (any_qs) {
            nstep = tc_nint(1.f / onstep[2]);
            for (int n = 1; n <= nstep; ++n) {
                for (int k = kte; k >= kts; --k) {
                    CB(TC_S_A, k) = CB(TC_V_TS, k) * WORK(TC_W_RS, k);
                }
                int k = kte;
                float odzq = 1.f / dzq[AT(k)];
                float orho = 1.f / WORK(TC_W_RHO, k);
                TEN(TC_QSTEN, k) = TEN(TC_QSTEN, k)
                    - CB(TC_S_A, k) * odzq * onstep[2] * orho;
                WORK(TC_W_RS, k) = fmaxf(TC_R1, WORK(TC_W_RS, k)
                    - CB(TC_S_A, k) * odzq * DT * onstep[2]);
                for (k = ksed1[2]; k >= kts; --k) {
                    odzq = 1.f / dzq[AT(k)];
                    orho = 1.f / WORK(TC_W_RHO, k);
                    TEN(TC_QSTEN, k) = TEN(TC_QSTEN, k)
                        + (CB(TC_S_A, k + 1) - CB(TC_S_A, k))
                        * odzq * onstep[2] * orho;
                    WORK(TC_W_RS, k) = fmaxf(TC_R1, WORK(TC_W_RS, k)
                        + (CB(TC_S_A, k + 1) - CB(TC_S_A, k))
                        * odzq * DT * onstep[2]);
                }
                if (WORK(TC_W_RS, kts) > TC_R1 * 1000.f)
                    pptsnow = pptsnow + CB(TC_S_A, kts) * DT * onstep[2];
            }
        }

        // :3902-3939 graupel (the volume rb feeds no classic output).
        if (any_qg) {
            nstep = tc_nint(1.f / onstep[3]);
            for (int n = 1; n <= nstep; ++n) {
                for (int k = kte; k >= kts; --k) {
                    CB(TC_S_A, k) = CB(TC_V_TG, k) * WORK(TC_W_RG, k);
                    CB(TC_S_N, k) = CB(TC_V_TNG, k) * WORK(TC_W_NG, k);
                }
                int k = kte;
                float odzq = 1.f / dzq[AT(k)];
                float orho = 1.f / WORK(TC_W_RHO, k);
                TEN(TC_QGTEN, k) = TEN(TC_QGTEN, k)
                    - CB(TC_S_A, k) * odzq * onstep[3] * orho;
                TEN(TC_NGTEN, k) = TEN(TC_NGTEN, k)
                    - CB(TC_S_N, k) * odzq * onstep[3] * orho;
                WORK(TC_W_RG, k) = fmaxf(TC_R1, WORK(TC_W_RG, k)
                    - CB(TC_S_A, k) * odzq * DT * onstep[3]);
                WORK(TC_W_NG, k) = fmaxf(TC_R2, WORK(TC_W_NG, k)
                    - CB(TC_S_N, k) * odzq * DT * onstep[3]);
                for (k = ksed1[3]; k >= kts; --k) {
                    odzq = 1.f / dzq[AT(k)];
                    orho = 1.f / WORK(TC_W_RHO, k);
                    TEN(TC_QGTEN, k) = TEN(TC_QGTEN, k)
                        + (CB(TC_S_A, k + 1) - CB(TC_S_A, k))
                        * odzq * onstep[3] * orho;
                    TEN(TC_NGTEN, k) = TEN(TC_NGTEN, k)
                        + (CB(TC_S_N, k + 1) - CB(TC_S_N, k))
                        * odzq * onstep[3] * orho;
                    WORK(TC_W_RG, k) = fmaxf(TC_R1, WORK(TC_W_RG, k)
                        + (CB(TC_S_A, k + 1) - CB(TC_S_A, k))
                        * odzq * DT * onstep[3]);
                    WORK(TC_W_NG, k) = fmaxf(TC_R2, WORK(TC_W_NG, k)
                        + (CB(TC_S_N, k + 1) - CB(TC_S_N, k))
                        * odzq * DT * onstep[3]);
                }
                if (WORK(TC_W_RG, kts) > TC_R1 * 1000.f)
                    pptgraul = pptgraul + CB(TC_S_A, kts) * DT * onstep[3];
            }
        }

        for (int k = kts; k <= kte; ++k) {
            const float temp = WORK(TC_W_TEMP, k);
            const float ocp = WORK(TC_W_OCP, k);
            const float rho = WORK(TC_W_RHO, k);
            const float nc1d = WORK(TC_W_NC1D, k);
            float tten = TEN(TC_TTEN, k);
            const float qvten = TEN(TC_QVTEN, k);
            float qcten = TEN(TC_QCTEN, k);
            float ncten = TEN(TC_NCTEN, k);
            float qiten = TEN(TC_QITEN, k);
            float niten = TEN(TC_NITEN, k);
            const float qrten = TEN(TC_QRTEN, k);
            const float nrten = TEN(TC_NRTEN, k);
            const float qsten = TEN(TC_QSTEN, k);
            const float qgten = TEN(TC_QGTEN, k);
            const float ngten = TEN(TC_NGTEN, k);
            const float qi0 = qi1d[AT(k)];
            const float ni0 = ni1d[AT(k)];
            const float qc0 = qc1d[AT(k)];

            // :3946-3966 phase cleanup.
            const float xri = fmaxf(0.0f, qi0 + qiten * DT);
            if ((temp > TC_T0) && (xri > 0.0f)) {
                qcten = qcten + xri * odt;
                ncten = ncten + ni0 * odt;
                qiten = qiten - xri * odt;
                niten = -ni0 * odt;
                tten = tten - TC_LFUS * ocp * xri * odt * (float)(1 - 0);
            }
            const float xrc = fmaxf(0.0f, qc0 + qcten * DT);
            if ((temp < TC_HGFR) && (xrc > 0.0f)) {
                const float lfus2 = TC_LSUB - WORK(TC_W_LVAP, k);
                const float xnc = nc1d + ncten * DT;
                qiten = qiten + xrc * odt;
                niten = niten + xnc * odt;
                qcten = qcten - xrc * odt;
                ncten = ncten - xnc * odt;
                tten = tten + lfus2 * ocp * xrc * odt * (float)(1 - 0);
            }

            // :3972-4078 terminal apply.
            t1d[AT(k)] = t1d[AT(k)] + tten * DT;
            qv1d[AT(k)] = fmaxf(1.E-10f, qv1d[AT(k)] + qvten * DT);
            float qc = qc0 + qcten * DT;
            if (qc <= TC_R1) qc = 0.0f;
            qc1d[AT(k)] = qc;
            // nc1d (:3976, :4007-4021) feeds no classic output.

            float qi = qi0 + qiten * DT;
            float ni = fmaxf(TC_R2 / rho, ni0 + niten * DT);
            if (qi <= TC_R1) {
                qi = 0.0f;
                ni = 0.0f;
            } else {
                double lami = (double)gfk_pow(TC_AM_I * TC_CIG(2) * TC_oig1
                                              * ni / qi, TC_obmi);
                const double ilami = 1. / lami;
                const float xDi = (float)((double)(3.0f + 0.0f + 1.f) * ilami);
                if (xDi < 5.E-6f) {
                    lami = (double)(FDIV(TC_CIE(2), 5.E-6f));
                } else if (xDi > 300.E-6f) {
                    lami = (double)(FDIV(TC_CIE(2), 300.E-6f));
                }
                ni = (float)fmin((double)FDIV(TC_CIG(1) * TC_oig2 * qi, TC_AM_I)
                                 * glibc_pow(lami, 3.0), 999.E3 / (double)rho);
            }
            qi1d[AT(k)] = qi;
            ni1d[AT(k)] = ni;

            float qr = qr1d[AT(k)] + qrten * DT;
            float nr = fmaxf(TC_R2 / rho, nr1d[AT(k)] + nrten * DT);
            if (qr <= TC_R1) {
                qr = 0.0f;
                nr = 0.0f;
            } else {
                double lamr = (double)gfk_pow(TC_AM_R * TC_CRG(3) * TC_org2
                                              * nr / qr, TC_obmr);
                float mvd = (float)((double)(3.0f + 0.0f + 0.672f) / lamr);
                if (mvd > 2.5E-3f) {
                    mvd = 2.5E-3f;
                } else if (mvd < TC_D0R * 0.75f) {
                    mvd = TC_D0R * 0.75f;
                }
                lamr = (double)((3.0f + 0.0f + 0.672f) / mvd);
                nr = (float)(DDIV((double)(TC_CRG(2) * TC_org3 * qr)
                             * glibc_pow(lamr, 3.0), (double)TC_AM_R));
            }
            qr1d[AT(k)] = qr;
            nr1d[AT(k)] = nr;

            float qs = qs1d[AT(k)] + qsten * DT;
            if (qs <= TC_R1) qs = 0.0f;
            qs1d[AT(k)] = qs;

            float qg = qg1d[AT(k)] + qgten * DT;
            float ng = fmaxf(TC_R2 / rho, ng1d[AT(k)] + ngten * DT);
            if (qg <= TC_R1) {
                qg = 0.0f;
                ng = 0.0f;
            } else {
                double lamg = (double)gfk_pow(TC_AM_G5 * TC_CGG(3, 1)
                                              * TC_ogg2 * ng / qg, TC_obmg);
                float mvd = (float)((double)(3.0f + 0.0f + 0.672f) / lamg);
                if (mvd > 25.4E-3f) {
                    mvd = 25.4E-3f;
                } else if (mvd < TC_D0R) {
                    mvd = TC_D0R;
                }
                lamg = (double)((3.0f + 0.0f + 0.672f) / mvd);
                ng = (float)(DDIV((double)(TC_CGG(2, 1) * TC_ogg3 * qg)
                             * glibc_pow(lamg, 3.0), (double)TC_AM_G5));
            }
            qg1d[AT(k)] = qg;
            ng1d[AT(k)] = ng;
        }
    }

    // :1294-1308 surface accumulations; :1358 theta.
    const float ncv = pptrain + pptsnow + pptgraul + pptice;
    rainncv[col] = ncv;
    rainnc[col] = rainnc[col] + pptrain + pptsnow + pptgraul + pptice;
    snowncv[col] = pptsnow + pptice;
    snownc[col] = snownc[col] + pptsnow + pptice;
    graupelncv[col] = pptgraul;
    graupelnc[col] = graupelnc[col] + pptgraul;
    sr[col] = (pptsnow + pptgraul + pptice) / (ncv + 1.e-12f);
    for (int k = kts; k <= kte; ++k) {
        th[AT(k)] = t1d[AT(k)] / pii[AT(k)];
    }
#undef AT
#undef TEN
#undef WORK
#undef CB
}

// WRF moist_physics_finish_em (use_theta_m = 0) on the microphysics
// theta: the increment over the pre-microphysics theta, clamped to
// +/- mp_tend_lim*dt, added to thp; h_diabatic = increment/dt.
extern "C" __global__ void tc_finish(
    const float* __restrict__ th, float* __restrict__ thp,
    float* __restrict__ h_diabatic, const float* __restrict__ th_save,
    float limit, float dt, int no_mp_heating, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (!no_mp_heating) {
        // NaN propagates as the canonical NaN, as thompson_adapter_finish.
        const float nan_ = __int_as_float(0x7fffffff);
        float increment = th[idx] - th_save[idx];
        increment = isnan(increment) || isnan(limit)
            ? nan_ : fminf(limit, increment);
        increment = isnan(increment) || isnan(limit)
            ? nan_ : fmaxf(-limit, increment);
        thp[idx] = thp[idx] + increment;
        h_diabatic[idx] = increment / dt;
    } else {
        h_diabatic[idx] = 0.0f;
    }
}

// :1466-1478 with calc_effectRad :5594-5699, classic (nc = Nt_c), and
// the driver's metre->micron conversion re*1.E6 after the clamps.
extern "C" __global__ void tc_effective_radius(
    const float* __restrict__ t1d, const float* __restrict__ p,
    const float* __restrict__ qv1d, const float* __restrict__ qc1d,
    const float* __restrict__ qi1d, const float* __restrict__ ni1d,
    const float* __restrict__ qs1d,
    float* __restrict__ effc, float* __restrict__ effi,
    float* __restrict__ effs, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const float RE_QC_BG = 2.49E-6f, RE_QI_BG = 4.99E-6f, RE_QS_BG = 9.99E-6f;
    float re_qc = RE_QC_BG, re_qi = RE_QI_BG, re_qs = RE_QS_BG;
    const float rho = 0.622f * p[idx]
        / (TC_R * t1d[idx] * (qv1d[idx] + 0.622f));
    const float rc = fmaxf(TC_R1, qc1d[idx] * rho);
    const float nc = TC_NT_C;
    if (!(rc <= TC_R1 || nc <= TC_R2)) {
        int inu_c;
        if (nc < 100.f) {
            inu_c = 15;
        } else if (nc > 1.E10f) {
            inu_c = 2;
        } else {
            inu_c = min(15, tc_nint(1000.E6f / nc) + 2);
        }
        const float g_ratio[15] = {24.f, 60.f, 120.f, 210.f, 336.f, 504.f,
            720.f, 990.f, 1320.f, 1716.f, 2184.f, 2730.f, 3360.f, 4080.f,
            4896.f};
        const double lamc = (double)gfk_pow(nc * TC_AM_R * g_ratio[inu_c - 1]
                                            / rc, TC_obmr);
        re_qc = fmaxf(2.51E-6f, fminf((float)(0.5 * (double)(3.f
            + (float)inu_c) / lamc), 50.E-6f));
    }
    const float ri = fmaxf(TC_R1, qi1d[idx] * rho);
    const float ni = fmaxf(TC_R2, ni1d[idx] * rho);
    if (!(ri <= TC_R1 || ni <= TC_R2)) {
        const double lami = (double)gfk_pow(TC_AM_I * TC_CIG(2) * TC_oig1
                                            * ni / ri, TC_obmi);
        re_qi = fmaxf(2.51E-6f, fminf((float)(0.5 * (double)(3.f + 0.0f)
            / lami), 125.E-6f));
    }
    const float rs = fmaxf(TC_R1, qs1d[idx] * rho);
    if (!(rs <= TC_R1)) {
        const float tc0 = fminf(-0.1f, t1d[idx] - 273.15f);
        const float smob = rs * TC_oams;
        const float smo2 = smob;
        const float a_ = gfk_pow(10.0f, tc_poly(tc_sa, tc0, TC_CSE(1)));
        const float b_ = tc_poly(tc_sb, tc0, TC_CSE(1));
        const float smoc = a_ * gfk_pow(smo2, b_);
        re_qs = fmaxf(5.01E-6f, fminf(0.5f * (smoc / smob), 999.E-6f));
    }
    effc[idx] = fmaxf(RE_QC_BG, fminf(re_qc, 50.E-6f)) * 1.0e6f;
    effi[idx] = fmaxf(RE_QI_BG, fminf(re_qi, 125.E-6f)) * 1.0e6f;
    effs[idx] = fmaxf(RE_QS_BG, fminf(re_qs, 999.E-6f)) * 1.0e6f;
}

// ---------------------------------------------------------------------
// calc_refl10cm (:5710-6028) and module_mp_radar's melting-snow
// scattering (rayleigh_soak_wetgraupel :285-354, get_m_mix_nested,
// get_m_mix, m_complex_maxwellgarnett), complex arithmetic as gfortran
// expands it (-fcx-fortran-rules): products in full, quotients by Smith's
// range-reduced division.
// ---------------------------------------------------------------------
struct tcz { double re, im; };

__device__ __forceinline__ tcz tcz_make(double re, double im)
{ tcz z; z.re = re; z.im = im; return z; }

__device__ __forceinline__ tcz tcz_mul(tcz a, tcz b)
{
    return tcz_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

// Real scalar times complex: gfortran promotes the scalar to (s, 0) and
// multiplies in full.
__device__ __forceinline__ tcz tcz_rmul(double s, tcz b)
{
    return tcz_make(s * b.re - 0.0 * b.im, s * b.im + 0.0 * b.re);
}

// z**2: (re*re - im*im, (re*im)*2).
__device__ __forceinline__ tcz tcz_sq(tcz a)
{
    return tcz_make(a.re * a.re - a.im * a.im, (a.re * a.im) * 2.0);
}

// GCC expand_complex_div_wide (Smith).
__device__ __forceinline__ tcz tcz_div(tcz a, tcz b)
{
    if (fabs(b.re) < fabs(b.im)) {
        const double ratio = b.re / b.im;
        const double div = b.im + b.re * ratio;
        return tcz_make((a.im + a.re * ratio) / div,
                        (a.im * ratio - a.re) / div);
    }
    const double ratio = b.im / b.re;
    const double div = b.re + b.im * ratio;
    return tcz_make((a.re + a.im * ratio) / div,
                    (a.im - a.re * ratio) / div);
}

__device__ __forceinline__ double tcz_abs(tcz a) { return hypot(a.re, a.im); }

__device__ __forceinline__ tcz tcz_log(tcz a)
{
    return tcz_make(glibc_log(hypot(a.re, a.im)), atan2(a.im, a.re));
}

__device__ __forceinline__ tcz tcz_sqrt(tcz a)
{
    if (a.re == 0.0 && a.im == 0.0) return tcz_make(0.0, a.im);
    const double m = hypot(a.re, a.im);
    if (a.re > 0.0) {
        const double r = sqrt(0.5 * (m + a.re));
        return tcz_make(r, a.im / (2.0 * r));
    }
    const double s = sqrt(0.5 * (m - a.re));
    return tcz_make(fabs(a.im) / (2.0 * s), copysign(s, a.im));
}

// m_complex_maxwellgarnett, inclusion 'spheroidal'; returns false where
// WRF sets error = 1 (volume fractions not summing to one).
__device__ bool tc_maxwellgarnett(double vol1, double vol2, double vol3,
                                  tcz m1, tcz m2, tcz m3, tcz* out)
{
    if (fabs(vol1 + vol2 + vol3 - 1.0) > 1e-6) return false;
    const tcz m1t = tcz_sq(m1), m2t = tcz_sq(m2), m3t = tcz_sq(m3);
    const tcz one = tcz_make(1.0, 0.0);
    // beta2 = 2*m1t/(m2t-m1t) * (m2t/(m2t-m1t)*LOG(m2t/m1t) - 1)
    tcz d2 = tcz_make(m2t.re - m1t.re, m2t.im - m1t.im);
    tcz l2 = tcz_log(tcz_div(m2t, m1t));
    tcz t2 = tcz_mul(tcz_div(m2t, d2), l2);
    t2 = tcz_make(t2.re - one.re, t2.im);
    const tcz beta2 = tcz_mul(tcz_div(tcz_rmul(2.0, m1t), d2), t2);
    tcz d3 = tcz_make(m3t.re - m1t.re, m3t.im - m1t.im);
    tcz l3 = tcz_log(tcz_div(m3t, m1t));
    tcz t3 = tcz_mul(tcz_div(m3t, d3), l3);
    t3 = tcz_make(t3.re - one.re, t3.im);
    const tcz beta3 = tcz_mul(tcz_div(tcz_rmul(2.0, m1t), d3), t3);
    const double w1 = 1.0 - vol2 - vol3;
    const tcz a = tcz_rmul(w1, m1t);
    const tcz b = tcz_mul(tcz_rmul(vol2, beta2), m2t);
    const tcz c = tcz_mul(tcz_rmul(vol3, beta3), m3t);
    const tcz num = tcz_make((a.re + b.re) + c.re, (a.im + b.im) + c.im);
    const tcz e = tcz_rmul(vol2, beta2);
    const tcz f = tcz_rmul(vol3, beta3);
    const tcz den = tcz_make((w1 + e.re) + f.re, (0.0 + e.im) + f.im);
    *out = tcz_sqrt(tcz_div(num, den));
    return true;
}

// rayleigh_soak_wetgraupel for snow (host 'air', matrix 'water',
// hostmatrix 'icewater', spheroidal inclusions, Maxwell-Garnett).
__device__ double tc_rayleigh_soak_wetgraupel(double x_g, double a_geo,
    double b_geo, double fmelt, double meltratio_outside)
{
    const tcz m_w = tcz_make(TC_M_W_0(1), TC_M_W_0(2));
    const tcz m_i = tcz_make(TC_M_I_0(1), TC_M_I_0(2));
    const tcz m_air = tcz_make(1.0, 0.0);
    const double PIx = 3.1415926535897932384626434;
    const double fm = fmax(fmin(fmelt, 1.0), 0.0);
    double mra = fmax(fmin(meltratio_outside, 1.0), 0.0);
    mra = mra + (1.0 - mra) * fm;
    const double x_w = x_g * fm;
    const double D_g = a_geo * glibc_pow(x_g, b_geo);
    if (!(D_g >= 1e-12)) return 0.0;
    double vg = (PIx / 6.) * (D_g * (D_g * D_g));
    const double rhog = fmax(fmin(x_g / vg, 900.0), 10.0);
    vg = x_g / rhog;
    const double meltratio_outside_grenz = 1.0 - DDIV(rhog, 1000.);
    double volg;
    if (mra <= meltratio_outside_grenz) {
        volg = vg * (1.0 - mra * fm);
    } else {
        const double fmgrenz = (900.0 - rhog)
            / (mra * 900.0 - rhog + DDIV(900.0 * rhog, 1000.));
        if (fm <= fmgrenz) {
            volg = (1.0 - mra * fm) * vg;
        } else {
            volg = DDIV((x_g - x_w), 900.0) + DDIV(x_w, 1000.);
        }
    }
    const double D_large = glibc_pow(6.0 / PIx * volg, (double)(1.f / 3.f));
    const double volice = (x_g - x_w) / (volg * 900.0);
    const double volwater = x_w / (1000. * volg);
    const double volair = 1.0 - volice - volwater;
    // get_m_mix_nested, host 'air'.
    const double vol1 = volice / fmax(volice + volwater, 1e-10);
    const double vol2 = 1.0 - vol1;
    tcz mtmp, m_core;
    if (!tc_maxwellgarnett(vol2, 0.0, vol1, m_w, m_air, m_i, &mtmp))
        return 0.0;
    if (!tc_maxwellgarnett(1.0 - volair, volair, 0.0, mtmp, m_air,
                           tcz_rmul(2.0, m_air), &m_core))
        return 0.0;
    const tcz m2 = tcz_sq(m_core);
    const tcz q = tcz_div(tcz_make(m2.re - 1.0, m2.im),
                          tcz_make(m2.re + 2.0, m2.im + 0.0));
    const double a = tcz_abs(q);
    const double d3 = D_large * D_large * D_large;
    return DDIV(a * a * TC_PI5 * (d3 * d3), TC_lamda4);
}

// WRF's own DOUBLE log10 (glibc 2.39 e_log10.c, fdlibm): y*log10_2lo +
// ivln10*log(x), then + y*log10_2hi.  Normal positive arguments only
// (dBZ's argument is at least 3e-4).
__device__ __forceinline__ double tc_log10d(double x)
{
    const double ivln10 = __longlong_as_double(0x3FDBCB7B1526E50ELL);
    const double log10_2hi = __longlong_as_double(0x3FD34413509F6000LL);
    const double log10_2lo = __longlong_as_double(0x3D59FEF311F12B36LL);
    long long hx = __double_as_longlong(x);
    const int k = (int)(hx >> 52) - 1023;
    const long long i = (long long)(((unsigned long long)(long long)k
                                     & 0x8000000000000000ULL) >> 63);
    hx = (hx & 0x000fffffffffffffLL) | ((0x3ffLL - i) << 52);
    const double y = (double)(k + i);
    const double xm = __longlong_as_double(hx);
    const double z = y * log10_2lo + ivln10 * glibc_log(xm);
    return z + y * log10_2hi;
}

extern "C" __global__ void tc_refl10cm(
    const float* __restrict__ qv1d, const float* __restrict__ qr1d,
    const float* __restrict__ nr1d, const float* __restrict__ qs1d,
    const float* __restrict__ qg1d, const float* __restrict__ ng1d,
    const float* __restrict__ t1d, const float* __restrict__ p1d,
    float* __restrict__ refl, int nz, int columns)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= columns) return;
#define AT(k) ((size_t)(k) * (size_t)columns + (size_t)col)
    const int kts = 0, kte = nz - 1, ke_diag = kte;
    // Column-local copies (:5758-5800), only the quantities used after.
    bool any_qs = false, any_qg = false;
    for (int k = kts; k <= kte; ++k) {
        any_qs |= qs1d[AT(k)] > TC_R2;
        any_qg |= qg1d[AT(k)] > TC_R2;
    }
    // :5883-5893 melting level.
    bool melti = false;
    int k_0 = kts;
    for (int k = kte - 1; k >= kts; --k) {
        const bool lqr = qr1d[AT(k)] > TC_R1;
        const bool lqs1 = qs1d[AT(k + 1)] > TC_R2;
        const bool lqg1 = qg1d[AT(k + 1)] > TC_R2;
        if ((t1d[AT(k)] > 273.15f) && lqr && (lqs1 || lqg1)) {
            k_0 = max(k + 1, k_0);
            melti = true;
            break;
        }
    }
    const int k_0loop = min(k_0, ke_diag + 1);
    // rs(k_0) for the melt fraction.
    float rs_k0 = TC_R1;
    {
        const float qv = fmaxf(1.E-10f, qv1d[AT(k_0)]);
        const float rho = 0.622f * p1d[AT(k_0)]
            / (TC_R * t1d[AT(k_0)] * (qv + 0.622f));
        if (qs1d[AT(k_0)] > TC_R2) rs_k0 = qs1d[AT(k_0)] * rho;
    }

    for (int k = kts; k <= ke_diag; ++k) {
        const float temp = t1d[AT(k)];
        const float qv = fmaxf(1.E-10f, qv1d[AT(k)]);
        const float pres = p1d[AT(k)];
        const float rho = 0.622f * pres / (TC_R * temp * (qv + 0.622f));
        float ze_rain = 1.e-22f, ze_snow = 1.e-22f, ze_graupel = 1.e-22f;
        if (qr1d[AT(k)] > TC_R1) {
            const float rr = qr1d[AT(k)] * rho;
            const float nr = fmaxf(TC_R2, nr1d[AT(k)] * rho);
            const double lamr = (double)gfk_pow(TC_AM_R * TC_CRG(3) * TC_org2
                                                * nr / rr, TC_obmr);
            const double ilamr = 1. / lamr;
            const double N0_r = (double)(nr * TC_org2)
                * glibc_pow(lamr, (double)TC_CRE(2));
            ze_rain = (float)(N0_r * (double)TC_CRG(4)
                              * glibc_pow(ilamr, (double)TC_CRE(4)));
        }
        float smob = 0.f, smoc = 0.f, smoz = 0.f;
        const bool lqs = qs1d[AT(k)] > TC_R2;
        float rs = TC_R1;
        if (lqs) rs = qs1d[AT(k)] * rho;
        if (any_qs && lqs) {
            const float tc0 = fminf(-0.1f, temp - 273.15f);
            smob = rs * TC_oams;
            const float smo2 = smob;
            float a_ = gfk_pow(10.0f, tc_poly(tc_sa, tc0, TC_CSE(1)));
            float b_ = tc_poly(tc_sb, tc0, TC_CSE(1));
            smoc = a_ * gfk_pow(smo2, b_);
            a_ = gfk_pow(10.0f, tc_poly(tc_sa, tc0, TC_CSE(3)));
            b_ = tc_poly(tc_sb, tc0, TC_CSE(3));
            smoz = a_ * gfk_pow(smo2, b_);
        }
        if (lqs) {
            ze_snow = (0.176f / 0.93f) * (6.0f / TC_PI) * (6.0f / TC_PI)
                * (TC_AM_S / 900.0f) * (TC_AM_S / 900.0f) * smoz;
        }
        if (qg1d[AT(k)] > TC_R2) {
            const float rg = qg1d[AT(k)] * rho;
            const float ng = fmaxf(TC_R2, ng1d[AT(k)] * rho);
            const double lamg = (double)gfk_pow(TC_AM_G5 * TC_CGG(3, 1)
                                                * TC_ogg2 * ng / rg, TC_obmg);
            const double ilamg = 1. / lamg;
            const double N0_g = (double)(ng * TC_ogg2)
                * glibc_pow(lamg, (double)TC_CGE(2, 1));
            ze_graupel = (float)((double)((0.176f / 0.93f) * (6.0f / TC_PI)
                * (6.0f / TC_PI) * (TC_AM_G5 / 900.0f) * (TC_AM_G5 / 900.0f))
                * N0_g * (double)TC_CGG(4, 1)
                * glibc_pow(ilamg, (double)TC_CGE(4, 1)));
        }
        // :5931-5977 melting snow below the melting level.
        if (melti && k_0 >= 1 && k <= k_0loop - 1 && lqs
                && qs1d[AT(k_0)] > TC_R2) {
            const double fmelt_s = fmax(0.05, fmin(1.0 - (double)(rs / rs_k0),
                                                   0.99));
            double eta = 0.;
            const float oM3 = 1.f / smoc;
            const float M0 = (smob * oM3);
            const float Mrat = smob * M0 * M0 * M0;
            const float slam1 = M0 * TC_LAM0;
            const float slam2 = M0 * TC_LAM1;
            for (int n = 1; n <= 50; ++n) {
                const double xs = TC_XXDS(n);
                const double x = (double)TC_AM_S * (xs * xs);
                const double CBACK = tc_rayleigh_soak_wetgraupel(
                    x, (double)TC_ocms, (double)TC_obms, fmelt_s, 0.9);
                const double f_d = (double)Mrat * ((double)TC_KAP0
                    * glibc_exp(-(double)slam1 * xs) + (double)TC_KAP1
                    * glibc_pow((double)M0 * xs, (double)TC_MU_S)
                    * glibc_exp(-(double)slam2 * xs));
                eta = eta + f_d * CBACK * TC_SIMPSON(n) * TC_XDTS(n);
            }
            ze_snow = (float)(TC_lamda4 / (TC_PI5 * TC_K_w) * eta);
        }
        const float dbz = (float)(10. * tc_log10d((double)(ze_rain + ze_snow
                                                           + ze_graupel)
                                                  * 1.e18));
        refl[AT(k)] = fmaxf(-35.f, dbz);
    }
#undef AT
}
