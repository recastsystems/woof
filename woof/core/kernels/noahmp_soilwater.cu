// gpuwm/core/kernels/noahmp_soilwater.cu
//
// CUDA half of the Noah-MP soil-water port.  Bitwise-equal to
// gpuwm/core/noahmp_soilwater.py and to the WRF v4.6.1 oracle
// (tree d66e442fccc04111067e29274c9f9eaccc3cef28,
//  sha256(phys/module_sf_noahmplsm.F) =
//  bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282).
//
// Covers CANWATER (6265-6394), SOILWATER (7234-7556), INFIL (7616-7712),
// SRT (7716-7846), SSTEP (7850-7973), and the two WDFCND forms (9153-9232)
// and ROSR12 (5534-5591) that they call.
//
// Three rules make bitwise agreement possible and none is optional:
//
// 1. Every float32 operation uses an explicit rounding intrinsic
//    (__fadd_rn/__fsub_rn/__fmul_rn/__fdiv_rn) and every float64 operation
//    inside glibc_expf/glibc_powf uses __dadd_rn/__dsub_rn/__dmul_rn/
//    __fma_rn.  That pins the hardware rounding mode AND makes nvcc's
//    contraction pass a no-op, so -fmad=true cannot fuse a site gfortran did
//    not fuse.
// 2. Every constant lives in __constant__ memory as a bit pattern.  ptxas
//    12.8's constant folder does not honour round-to-nearest-even on literal
//    arrays, so a table of FP32 literals can have its differences mis-folded
//    at compile time; __fsub_rn pins the hardware, not the folder.
// 3. EXP and ** are glibc calls in the oracle, so glibc_expf and glibc_powf
//    below transcribe glibc 2.39's own algorithms.  CUDA's expf, __expf,
//    exp2f and powf are all different functions and none can hold a
//    max_ulp-0 gate.
//
// A fourth thing this file must NOT do: vectorise SOILWATER's frozen-fraction
// loop the way gfortran does at WRF's own -O2.  There, gfortran calls glibc's
// libmvec _ZGVbN4v_expf, which is a different function from scalar expf and
// disagrees with it by 1 ULP on the fixture's slw_frozen case.  The fixture is
// built at -O0 for exactly that reason (see PROVENANCE-soilwater.md), and the
// per-layer loop here calls the scalar transcription four times.
//
// Pinned option identity: opt_run=3, opt_inf=1, opt_tdrn=0, opt_irr=0.
// Everything they kill is absent, not stubbed:
//   opt_run=3   GROUNDWATER, SHALLOWWATERTABLE, ZWTEQ, the VIC/XAJ/DVIC
//               runoff schemes, SOILWATER's OPT_RUN 1/2/4/5/6/7/8 blocks,
//               SRT's OPT_RUN 1/2/4/5 drainage forms, SSTEP's water table;
//   opt_inf=1   SRT's WDFCND2 loop (WDFCND2 stays live through INFIL);
//   opt_tdrn=0  TILE_DRAIN and TILE_HOOGHOUDT;
//   opt_irr=0   the irrigation routines.
//
// Two INTENT(OUT) aliasing hazards are reproduced, not tidied: SOILWATER's
// RUNSUB is read at 7549 before it is ever assigned, and INFIL's PDDUM/RUNSRF
// are assigned only inside IF (QINSUR > 0.0).  Both are passed in and returned.

#define NSOIL 4

// --------------------------------------------------------------------------
// Flat host<->device layout.  One row per case; slot meanings per leaf.
// --------------------------------------------------------------------------
#define P_SMCMAX  0  /* 4 */
#define P_SMCWLT  4  /* 4 */
#define P_BEXP    8  /* 4 */
#define P_DKSAT  12  /* 4 */
#define P_DWSAT  16  /* 4 */
#define P_KDT    20
#define P_FRZX   21
#define P_SLOPE  22
#define P_CH2OP  23
#define P_STRIDE 24

#define IN_STRIDE  40
#define OUT_STRIDE 32

// --------------------------------------------------------------------------
// glibc __exp2f_data / __powf_log2_data, as glibc stores them
// --------------------------------------------------------------------------
__constant__ unsigned long long C_EXP2F_TAB[32] = {
    0x3FF0000000000000ULL, 0x3FEFD9B0D3158574ULL, 0x3FEFB5586CF9890FULL, 0x3FEF9301D0125B51ULL,
    0x3FEF72B83C7D517BULL, 0x3FEF54873168B9AAULL, 0x3FEF387A6E756238ULL, 0x3FEF1E9DF51FDEE1ULL,
    0x3FEF06FE0A31B715ULL, 0x3FEEF1A7373AA9CBULL, 0x3FEEDEA64C123422ULL, 0x3FEECE086061892DULL,
    0x3FEEBFDAD5362A27ULL, 0x3FEEB42B569D4F82ULL, 0x3FEEAB07DD485429ULL, 0x3FEEA47EB03A5585ULL,
    0x3FEEA09E667F3BCDULL, 0x3FEE9F75E8EC5F74ULL, 0x3FEEA11473EB0187ULL, 0x3FEEA589994CCE13ULL,
    0x3FEEACE5422AA0DBULL, 0x3FEEB737B0CDC5E5ULL, 0x3FEEC49182A3F090ULL, 0x3FEED503B23E255DULL,
    0x3FEEE89F995AD3ADULL, 0x3FEEFF76F2FB5E47ULL, 0x3FEF199BDD85529CULL, 0x3FEF3720DCEF9069ULL,
    0x3FEF5818DCFBA487ULL, 0x3FEF7C97337B9B5FULL, 0x3FEFA4AFA2A490DAULL, 0x3FEFD0765B6E4540ULL,
};
// [0] shift_scaled, [1] shift, [2] invln2_scaled
__constant__ unsigned long long C_EXP2F_MISC[3] = {
    0x42E8000000000000ULL, 0x4338000000000000ULL, 0x40471547652B82FEULL,
};
__constant__ unsigned long long C_EXP2F_POLY[3] = {
    0x3FAC6AF84B912394ULL, 0x3FCEBFCE50FAC4F3ULL, 0x3FE62E42FF0C52D6ULL,
};
__constant__ unsigned long long C_EXP2F_POLY_SCALED[3] = {
    0x3EBC6AF84B912394ULL, 0x3F2EBFCE50FAC4F3ULL, 0x3F962E42FF0C52D6ULL,
};
__constant__ unsigned long long C_POWF_LOG2_TAB[32] = {
    0x3FF661EC79F8F3BEULL, 0xBFDEFEC65B963019ULL,
    0x3FF571ED4AAF883DULL, 0xBFDB0B6832D4FCA4ULL,
    0x3FF49539F0F010B0ULL, 0xBFD7418B0A1FB77BULL,
    0x3FF3C995B0B80385ULL, 0xBFD39DE91A6DCF7BULL,
    0x3FF30D190C8864A5ULL, 0xBFD01D9BF3F2B631ULL,
    0x3FF25E227B0B8EA0ULL, 0xBFC97C1D1B3B7AF0ULL,
    0x3FF1BB4A4A1A343FULL, 0xBFC2F9E393AF3C9FULL,
    0x3FF12358F08AE5BAULL, 0xBFB960CBBF788D5CULL,
    0x3FF0953F419900A7ULL, 0xBFAA6F9DB6475FCEULL,
    0x3FF0000000000000ULL, 0x0000000000000000ULL,
    0x3FEE608CFD9A47ACULL, 0x3FB338CA9F24F53DULL,
    0x3FECA4B31F026AA0ULL, 0x3FC476A9543891BAULL,
    0x3FEB2036576AFCE6ULL, 0x3FCE840B4AC4E4D2ULL,
    0x3FE9C2D163A1AA2DULL, 0x3FD40645F0C6651CULL,
    0x3FE886E6037841EDULL, 0x3FD88E9C2C1B9FF8ULL,
    0x3FE767DCF5534862ULL, 0x3FDCE0A44EB17BCCULL,
};
__constant__ unsigned long long C_POWF_LOG2_POLY[5] = {
    0x3FD27616C9496E0BULL, 0xBFD71969A075C67AULL, 0x3FDEC70A6CA7BADDULL,
    0xBFE7154748BEF6C8ULL, 0x3FF71547652AB82BULL,
};
// [0] expf overflow, [1] expf underflow, [2] powf overflow
__constant__ unsigned long long C_LIMITS[3] = {
    0x40562E42E0000000ULL, 0xC059FE3680000000ULL, 0x405FFFFFFFD1D571ULL,
};

// --------------------------------------------------------------------------
// Every float32 literal these routines use, as a bit pattern.  Sites that
// spell the same value in different Fortran statements get their own slot, so
// the mutation study can probe each independently.
// --------------------------------------------------------------------------
__constant__ unsigned int C_F32[28] = {
    0x00000000u, /*  0 K_ZERO      0.0                        */
    0x3F800000u, /*  1 K_ONE       1.0                        */
    0x40000000u, /*  2 K_TWO       2.0                        */
    0x4388947Bu, /*  3 K_TFRZ      273.16   :207              */
    0x4A193900u, /*  4 K_HVAP      2.5104e6 :209              */
    0x4A2D9580u, /*  5 K_HSUB      2.8440e6 :208              */
    0x48A2E400u, /*  6 K_HFUS      0.3336e6 :210              */
    0x4A7F9D80u, /*  7 K_CWAT      4.188e6  :211              */
    0x49FF9D80u, /*  8 K_CICE      2.094e6  :212              */
    0x447A0000u, /*  9 K_DENH2O    1000.0   :219              */
    0x44654000u, /* 10 K_DENICE    917.0    :220              */
    0x358637BDu, /* 11 K_1EM6_LIQ  1.0E-06  :6346 zeroing     */
    0x358637BDu, /* 12 K_1EM6_ICE  1.0E-6   :6355 zeroing     */
    0x358637BDu, /* 13 K_1EM6_SNO  1.0E-06  :6360 FWET floor  */
    0x358637BDu, /* 14 K_1EM6_LIQF 1.0E-06  :6362 FWET floor  */
    0x358637BDu, /* 15 K_1EM6_MELT 1.0E-6   :6372 melt test   */
    0x358637BDu, /* 16 K_1EM6_FRZ  1.0E-6   :6379 freeze test */
    0x40D33333u, /* 17 K_6P6       6.6      :6351             */
    0x3E8A3D71u, /* 18 K_0P27      0.27     :6351             */
    0x42380000u, /* 19 K_46        46.0     :6351             */
    0x3F2AC083u, /* 20 K_0P667     0.667    :6364  NB 0.667, not 2/3 */
    0x47A8C000u, /* 21 K_86400     86400.0  :7656             */
    0x3C23D70Au, /* 22 K_1EM2      1.0E-2   :7687             */
    0x38D1B717u, /* 23 K_1EM4_RSAT 1.0E-4   :7326             */
    0x38D1B717u, /* 24 K_1EM4_UP   1.0E-4   :7952             */
    0x38D1B717u, /* 25 K_1EM4_L1   1.0E-4   :7958             */
    0x38D1B717u, /* 26 K_1EM4_DOWN 1.0E-4   :7963/7967        */
    0x3C23D70Au, /* 27 K_WATMIN    0.01     :7551             */
};
#define K_ZERO      __uint_as_float(C_F32[0])
#define K_ONE       __uint_as_float(C_F32[1])
#define K_TWO       __uint_as_float(C_F32[2])
#define K_TFRZ      __uint_as_float(C_F32[3])
#define K_HVAP      __uint_as_float(C_F32[4])
#define K_HSUB      __uint_as_float(C_F32[5])
#define K_HFUS      __uint_as_float(C_F32[6])
#define K_CWAT      __uint_as_float(C_F32[7])
#define K_CICE      __uint_as_float(C_F32[8])
#define K_DENH2O    __uint_as_float(C_F32[9])
#define K_DENICE    __uint_as_float(C_F32[10])
#define K_1EM6_LIQ  __uint_as_float(C_F32[11])
#define K_1EM6_ICE  __uint_as_float(C_F32[12])
#define K_1EM6_SNO  __uint_as_float(C_F32[13])
#define K_1EM6_LIQF __uint_as_float(C_F32[14])
#define K_1EM6_MELT __uint_as_float(C_F32[15])
#define K_1EM6_FRZ  __uint_as_float(C_F32[16])
#define K_6P6       __uint_as_float(C_F32[17])
#define K_0P27      __uint_as_float(C_F32[18])
#define K_46        __uint_as_float(C_F32[19])
#define K_0P667     __uint_as_float(C_F32[20])
#define K_86400     __uint_as_float(C_F32[21])
#define K_1EM2      __uint_as_float(C_F32[22])
#define K_1EM4_RSAT __uint_as_float(C_F32[23])
#define K_1EM4_UP   __uint_as_float(C_F32[24])
#define K_1EM4_L1   __uint_as_float(C_F32[25])
#define K_1EM4_DOWN __uint_as_float(C_F32[26])
#define K_WATMIN    __uint_as_float(C_F32[27])

// SOILWATER's REAL, PARAMETER :: A = 4.0 at :7317, and the WDFCND/urban/
// unit-conversion literals.
__constant__ unsigned int C_F32B[11] = {
    0x40800000u, /* 0 K_A        4.0     :7317 */
    0x3F733333u, /* 1 K_0P95     0.95    :7361 */
    0x447A0000u, /* 2 K_1000_MLQ 1000.0  :7548 */
    0x447A0000u, /* 3 K_1000_RSR 1000.0  :7497 runsrf     */
    0x447A0000u, /* 4 K_1000_RST 1000.0  :7497 rsat       */
    0x447A0000u, /* 5 K_1000_QDR 1000.0  :7498 qdrain     */
    0x447A0000u, /* 6 K_1000_BAC 1000.0  :7573 back out   */
    0x40400000u, /* 7 K_THREE    3.0     :7440 NITER      */
    0x40400000u, /* 8 K_CVFRZ    3.0     :7683 CVFRZ       */
    0x40000000u, /* 9 K_FLOATK2  2.0     :7691 FLOAT(K), J=1 */
    0x3F800000u, /* 10 K_FLOATK1 1.0     :7691 FLOAT(K), J=2 */
};
#define K_A        __uint_as_float(C_F32B[0])
#define K_0P95     __uint_as_float(C_F32B[1])
#define K_1000_MLQ __uint_as_float(C_F32B[2])
#define K_1000_RSR __uint_as_float(C_F32B[3])
#define K_1000_RST __uint_as_float(C_F32B[4])
#define K_1000_QDR __uint_as_float(C_F32B[5])
#define K_1000_BAC __uint_as_float(C_F32B[6])
#define K_CVFRZ    __uint_as_float(C_F32B[8])
#define K_FLOATK2  __uint_as_float(C_F32B[9])
#define K_FLOATK1  __uint_as_float(C_F32B[10])

// WDFCND1/WDFCND2's own literals, module_sf_noahmplsm.F:9153-9232.
__constant__ unsigned int C_WDF[5] = {
    0x3C23D70Au, /* 0 W_0P01   0.01  :9177/:9217 FACTR floor */
    0x40000000u, /* 1 W_TWO    2.0   :9178/:9184/:9219/:9229 */
    0x40400000u, /* 2 W_THREE  3.0   :9184/:9229              */
    0x3D4CCCCDu, /* 3 W_0P05   0.05  :9216                    */
    0x43FA0000u, /* 4 W_500    500.0 :9224                    */
};
#define W_0P01  __uint_as_float(C_WDF[0])
#define W_TWO   __uint_as_float(C_WDF[1])
#define W_THREE __uint_as_float(C_WDF[2])
#define W_0P05  __uint_as_float(C_WDF[3])
#define W_500   __uint_as_float(C_WDF[4])

// --------------------------------------------------------------------------
// rounding-pinned primitives
// --------------------------------------------------------------------------
#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))

#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define DFMA(a, b, c) __fma_rn((a), (b), (c))

// Python's builtin max/min, which the CPU transcription uses, return the
// FIRST operand on a tie.  Fortran MAX/MIN may return either, and the values
// are equal, so this is the tie-break to mirror.
__device__ __forceinline__ float f_max(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float f_min(float a, float b) { return (b < a) ? b : a; }
__device__ __forceinline__ float f_abs(float a)
{
    return __uint_as_float(__float_as_uint(a) & 0x7FFFFFFFu);
}
__device__ __forceinline__ float f_neg(float a)
{
    return __uint_as_float(__float_as_uint(a) ^ 0x80000000u);
}

// Round a double to binary32, INCLUDING into the subnormal range.
//
// `__double2float_rn` flushes a subnormal result to zero on this toolchain --
// measured on sm_120 with CUDA 13.0, and asking for `--ftz=false` does not
// change it, because CuPy appends `-ftz=true` after the caller's options and
// the compiler honours the last occurrence.  The flush is the flag, not the
// SM: the conversion instruction has no flush of its own, so under `-ftz=true`
// the compiler emits an extra multiply after it to produce one, and built
// without the append the same conversion keeps the subnormal.  glibc's
// expf and powf do produce subnormals, gfortran at -O0 leaves MXCSR's FTZ/DAZ
// clear, and `gpuwm.core.noahmp_libm` -- verified against the live glibc 2.39
// over 1,106,247,680 inputs -- produces them too.  So the conversion as
// compiled here is a divergence from the authority in a narrow but reachable
// band:
// expf(x) for x in [-103.616, -87.337), and powf wherever y*log2(x) lands in
// the same place.  ENERGY's RHSUR (:2203) evaluates exp(PSI*GRAV/(RW*TG)) on
// very dry soil and can reach it.
//
// The correctly rounded subnormal is recovered exactly rather than
// approximated: a binary32 subnormal with mantissa bits `m` is exactly
// m * 2^-149, `y * 2^149` is an exact scaling in binary64 for any y in this
// band, and `rint` rounds to nearest with ties to even -- which is the same
// rounding the conversion owes.  m == 2^23 is the carry into the smallest
// normal and its bit pattern 0x00800000 is already that number, so the carry
// needs no special case.
__device__ float nmp_d2f_rn(double y)
{
    double a = fabs(y);
    if (a > 0.0 && a < 1.1754943508222875e-38) {   /* 0x1p-126 */
        double scaled = rint(a * 7.1362384635297994e+44);   /* 2^149 */
        unsigned int m = (unsigned int)scaled;
        unsigned int s = (__double_as_longlong(y) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(s | m);
    }
    return __double2float_rn(y);
}

// --------------------------------------------------------------------------
// glibc 2.39 expf  --  sysdeps/ieee754/flt-32/e_expf.c, FMA variant
// --------------------------------------------------------------------------
__device__ float glibc_expf(float x)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int abstop = (ix >> 20) & 0x7FFu;

    if (abstop >= 0x42Bu) {                       // top12(88.0f) & 0x7ff
        if (ix == 0xFF800000u) return 0.0f;       // -inf
        if (abstop >= 0x7F8u) return FADD(x, x);  // +-inf or NaN
        if ((double) x > __longlong_as_double(C_LIMITS[0]))
            return __uint_as_float(0x7F800000u);  // overflow
        if ((double) x < __longlong_as_double(C_LIMITS[1]))
            return 0.0f;                          // underflow
    }

    double shift = __longlong_as_double(C_EXP2F_MISC[1]);
    double xd = (double) x;
    double z = DMUL(__longlong_as_double(C_EXP2F_MISC[2]), xd);
    double kd = DADD(z, shift);
    unsigned long long ki = (unsigned long long) __double_as_longlong(kd);
    kd = DSUB(kd, shift);
    double r = DSUB(z, kd);

    unsigned long long t = C_EXP2F_TAB[ki & 31ULL];
    t += (ki << (52 - 5));
    double s = __longlong_as_double((long long) t);
    double c0 = __longlong_as_double(C_EXP2F_POLY_SCALED[0]);
    double c1 = __longlong_as_double(C_EXP2F_POLY_SCALED[1]);
    double c2 = __longlong_as_double(C_EXP2F_POLY_SCALED[2]);
    double z2 = DFMA(c0, r, c1);
    double r2 = DMUL(r, r);
    double y = DFMA(c2, r, 1.0);
    y = DFMA(z2, r2, y);
    y = DMUL(y, s);
    return nmp_d2f_rn(y);
}

// --------------------------------------------------------------------------
// glibc 2.39 powf  --  sysdeps/ieee754/flt-32/e_powf.c, FMA variant.
// Only the domain this group reaches is transcribed: a strictly positive
// normal base and a finite exponent.  Anything else returns NaN rather than a
// value this kernel cannot vouch for.
// --------------------------------------------------------------------------
__device__ double powf_log2_inline(unsigned int ix)
{
    unsigned int tmp = ix - 0x3F330000u;
    unsigned int i = (tmp >> 19) & 15u;
    unsigned int top = tmp & 0xFF800000u;
    unsigned int iz = ix - top;
    int k = ((int) top) >> 23;
    double invc = __longlong_as_double(C_POWF_LOG2_TAB[2 * i]);
    double logc = __longlong_as_double(C_POWF_LOG2_TAB[2 * i + 1]);
    double z = (double) __uint_as_float(iz);

    double r = DFMA(z, invc, -1.0);
    double y0 = DADD(logc, (double) k);

    double a0 = __longlong_as_double(C_POWF_LOG2_POLY[0]);
    double a1 = __longlong_as_double(C_POWF_LOG2_POLY[1]);
    double a2 = __longlong_as_double(C_POWF_LOG2_POLY[2]);
    double a3 = __longlong_as_double(C_POWF_LOG2_POLY[3]);
    double a4 = __longlong_as_double(C_POWF_LOG2_POLY[4]);

    double r2 = DMUL(r, r);
    double y = DFMA(a0, r, a1);
    double p = DFMA(a2, r, a3);
    double r4 = DMUL(r2, r2);
    double q = DFMA(a4, r, y0);
    q = DFMA(p, r2, q);
    y = DFMA(y, r4, q);
    return y;
}

__device__ float powf_exp2_inline(double xd, unsigned long long sign_bias)
{
    double shift = __longlong_as_double(C_EXP2F_MISC[0]);
    double kd = DADD(xd, shift);
    unsigned long long ki = (unsigned long long) __double_as_longlong(kd);
    kd = DSUB(kd, shift);
    double r = DSUB(xd, kd);

    unsigned long long t = C_EXP2F_TAB[ki & 31ULL];
    unsigned long long ski = ki + sign_bias;
    t += (ski << (52 - 5));
    double s = __longlong_as_double((long long) t);
    double c0 = __longlong_as_double(C_EXP2F_POLY[0]);
    double c1 = __longlong_as_double(C_EXP2F_POLY[1]);
    double c2 = __longlong_as_double(C_EXP2F_POLY[2]);
    double z = DFMA(c0, r, c1);
    double r2 = DMUL(r, r);
    double y = DFMA(c2, r, 1.0);
    y = DFMA(z, r2, y);
    y = DMUL(y, s);
    return nmp_d2f_rn(y);
}

__device__ float glibc_powf(float x, float y)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int iy = __float_as_uint(y);
    if (ix == 0x3F800000u) return 1.0f;           // 1**y, glibc's fast exit
    // glibc's zeroinfnan path, restricted to the one sub-case this group
    // reaches: +0 ** (finite, positive, non-integer) is +0.  CANWATER hits it
    // whenever FWET is exactly 0, which is every dry-canopy case.
    if (ix == 0u && (iy & 0x80000000u) == 0u && (iy & 0x7F800000u) != 0x7F800000u
        && iy != 0u)
        return 0.0f;
    if ((ix - 0x00800000u) >= 0x7F000000u || (((2u * iy) - 1u) >= 0xFEFFFFFFu))
        return __uint_as_float(0x7FC00000u);      // outside the ported domain

    double logx = powf_log2_inline(ix);
    double ylogx = DMUL((double) y, logx);
    unsigned long long ab = (unsigned long long) __double_as_longlong(ylogx);
    unsigned long long lim = ((unsigned long long) __double_as_longlong(126.0)) >> 47;
    if (((ab >> 47) & 0xFFFFULL) >= lim) {
        if (ylogx > __longlong_as_double(C_LIMITS[2]))
            return __uint_as_float(0x7F800000u);
        if (ylogx <= -150.0) return 0.0f;
    }
    return powf_exp2_inline(ylogx, 0ULL);
}

// --------------------------------------------------------------------------
// WDFCND1 -- module_sf_noahmplsm.F:9153-9188 (OPT_INF == 1)
// --------------------------------------------------------------------------
__device__ void d_wdfcnd1(float smc, float fcr, float smcmax, float bexp,
                          float dwsat, float dksat, float *wdf, float *wcnd)
{
    float factr = f_max(W_0P01, FDIV(smc, smcmax));                  // :9177
    float expon = FADD(bexp, W_TWO);                                 // :9178
    float w = FMUL(dwsat, glibc_powf(factr, expon));                 // :9179
    w = FMUL(w, FSUB(K_ONE, fcr));                                   // :9180
    expon = FADD(FMUL(W_TWO, bexp), W_THREE);                        // :9184
    float c = FMUL(dksat, glibc_powf(factr, expon));                 // :9185
    c = FMUL(c, FSUB(K_ONE, fcr));                                   // :9186
    *wdf = w;
    *wcnd = c;
}

// --------------------------------------------------------------------------
// WDFCND2 -- module_sf_noahmplsm.F:9192-9232
// --------------------------------------------------------------------------
__device__ void d_wdfcnd2(float smc, float sice, float smcmax, float bexp,
                          float dwsat, float dksat, float *wdf, float *wcnd)
{
    float factr1 = FDIV(W_0P05, smcmax);                             // :9216
    float factr2 = f_max(W_0P01, FDIV(smc, smcmax));                 // :9217
    factr1 = f_min(factr1, factr2);                                  // :9218
    float expon = FADD(bexp, W_TWO);                                 // :9219
    float w = FMUL(dwsat, glibc_powf(factr2, expon));                // :9220
    if (sice > K_ZERO) {                                             // :9222
        // (500*SICE)**3.0 has a REAL constant exponent, so gfortran routes it
        // through powf: it rounds once, not three times.
        float vkwgt = FDIV(K_ONE,
                           FADD(K_ONE, glibc_powf(FMUL(W_500, sice), W_THREE)));
        w = FADD(FMUL(vkwgt, w),
                 FMUL(FMUL(FSUB(K_ONE, vkwgt), dwsat),
                      glibc_powf(factr1, expon)));
    }
    expon = FADD(FMUL(W_TWO, bexp), W_THREE);                        // :9229
    *wcnd = FMUL(dksat, glibc_powf(factr2, expon));                  // :9230
    *wdf = w;
}

// --------------------------------------------------------------------------
// ROSR12 -- module_sf_noahmplsm.F:5534-5591, called with NTOP=1, NSNOW=0
// P <- ci, A <- ai, B <- bi, C <- ciin, D <- rhsttin, DELTA <- rhstt
// --------------------------------------------------------------------------
__device__ void d_rosr12(const float *a, const float *b, float *c,
                         const float *d, float *p, float *delta)
{
    c[NSOIL - 1] = K_ZERO;                                           // :5565
    p[0] = FDIV(f_neg(c[0]), b[0]);                                  // :5566
    delta[0] = FDIV(d[0], b[0]);                                     // :5570
    for (int k = 1; k < NSOIL; ++k) {                                // :5574
        float recip = FDIV(K_ONE, FADD(b[k], FMUL(a[k], p[k - 1])));
        p[k] = FMUL(f_neg(c[k]), recip);
        delta[k] = FMUL(FSUB(d[k], FMUL(a[k], delta[k - 1])), recip);
    }
    p[NSOIL - 1] = delta[NSOIL - 1];                                 // :5582
    for (int k = NSOIL - 2; k >= 0; --k) {                           // :5586
        p[k] = FADD(FMUL(p[k], p[k + 1]), delta[k]);
    }
}

// --------------------------------------------------------------------------
// CANWATER -- module_sf_noahmplsm.F:6265-6394
// in : ch2op dt fcev fctr elai esai fveg bdfall canliq canice tv frozen
// out: canliq canice tv cmc ecan etran fwet qsubc qfroc qfrzc qmeltc qevac qdewc
// --------------------------------------------------------------------------
__device__ void d_canwater(float ch2op, float dt, float fcev, float fctr,
                           float elai, float esai, float fveg, float bdfall,
                           int frozen, float canliq, float canice, float tv,
                           float *o)
{
    float lsai = FADD(elai, esai);
    float maxliq = FMUL(FMUL(fveg, ch2op), lsai);                    // :6323
    float etran, qevac, qdewc, qsubc, qfroc;

    if (!frozen) {                                                   // :6327
        etran = f_max(FDIV(fctr, K_HVAP), K_ZERO);
        qevac = f_max(FDIV(fcev, K_HVAP), K_ZERO);
        qdewc = f_abs(f_min(FDIV(fcev, K_HVAP), K_ZERO));
        qsubc = K_ZERO;
        qfroc = K_ZERO;
    } else {                                                         // :6333
        etran = f_max(FDIV(fctr, K_HSUB), K_ZERO);
        qevac = K_ZERO;
        qdewc = K_ZERO;
        qsubc = f_max(FDIV(fcev, K_HSUB), K_ZERO);
        qfroc = f_abs(f_min(FDIV(fcev, K_HSUB), K_ZERO));
    }

    // The strict native MIN at :6344 selects the second operand on a
    // signed-zero tie. Real WRF captures retain -0 here when FCEV is -0
    // and CANLIQ is +0; the generic first-on-tie helper loses that word.
    float canopy_rate = FDIV(canliq, dt);
    qevac = (canopy_rate < qevac) ? canopy_rate : qevac;            // :6344
    canliq = f_max(K_ZERO, FADD(canliq, FMUL(FSUB(qdewc, qevac), dt)));
    if (canliq <= K_1EM6_LIQ) canliq = K_ZERO;                       // :6346

    // FVEG * 6.6*(0.27+46.0/BDFALL) * (ELAI+ESAI) is a left-associative chain.
    float maxsno = FMUL(FMUL(FMUL(fveg, K_6P6),
                             FADD(K_0P27, FDIV(K_46, bdfall))), lsai);

    qsubc = f_min(FDIV(canice, dt), qsubc);                          // :6353
    canice = f_max(K_ZERO, FADD(canice, FMUL(FSUB(qfroc, qsubc), dt)));
    if (canice <= K_1EM6_ICE) canice = K_ZERO;                       // :6355

    float fwet;
    if (canice > K_ZERO && canice >= canliq) {                       // :6359
        fwet = FDIV(f_max(K_ZERO, canice), f_max(maxsno, K_1EM6_SNO));
    } else {
        fwet = FDIV(f_max(K_ZERO, canliq), f_max(maxliq, K_1EM6_LIQF));
    }
    fwet = glibc_powf(f_min(fwet, K_ONE), K_0P667);                  // :6364

    float qmeltc = K_ZERO, qfrzc = K_ZERO;
    float cmc = FADD(canliq, canice);                                // :6370

    if (canice > K_1EM6_MELT && tv > K_TFRZ) {                       // :6372
        qmeltc = f_min(FDIV(canice, dt),
                       FDIV(FDIV(FMUL(FMUL(FSUB(tv, K_TFRZ), K_CICE), canice),
                                 K_DENICE),
                            FMUL(dt, K_HFUS)));
        canice = f_max(K_ZERO, FSUB(canice, FMUL(qmeltc, dt)));
        canliq = f_max(K_ZERO, FSUB(cmc, canice));
        tv = FADD(FMUL(fwet, K_TFRZ), FMUL(FSUB(K_ONE, fwet), tv));
    }

    if (canliq > K_1EM6_FRZ && tv < K_TFRZ) {                        // :6379
        qfrzc = f_min(FDIV(canliq, dt),
                      FDIV(FDIV(FMUL(FMUL(FSUB(K_TFRZ, tv), K_CWAT), canliq),
                                K_DENH2O),
                           FMUL(dt, K_HFUS)));
        canliq = f_max(K_ZERO, FSUB(canliq, FMUL(qfrzc, dt)));
        canice = f_max(K_ZERO, FSUB(cmc, canliq));
        tv = FADD(FMUL(fwet, K_TFRZ), FMUL(FSUB(K_ONE, fwet), tv));
    }

    cmc = FADD(canliq, canice);                                      // :6388
    float ecan = FSUB(FSUB(FADD(qevac, qsubc), qdewc), qfroc);       // :6392

    o[0] = canliq; o[1] = canice; o[2] = tv;  o[3] = cmc;
    o[4] = ecan;   o[5] = etran;  o[6] = fwet;
    o[7] = qsubc;  o[8] = qfroc;  o[9] = qfrzc;
    o[10] = qmeltc; o[11] = qevac; o[12] = qdewc;
}

// --------------------------------------------------------------------------
// INFIL -- module_sf_noahmplsm.F:7616-7712
// pddum/runsrf are INTENT(OUT) but assigned only inside IF (QINSUR > 0.0), so
// they are in-out here.
// --------------------------------------------------------------------------
__device__ void d_infil(const float *par, float dt, const float *zsoil,
                        const float *sh2o, const float *sice, float sicemax,
                        float qinsur, float *pddum, float *runsrf)
{
    if (!(qinsur > K_ZERO)) return;                                  // :7655

    float dt1 = FDIV(dt, K_86400);                                   // :7656
    float smcav = FSUB(par[P_SMCMAX], par[P_SMCWLT]);                // :7657

    float dmax[NSOIL];
    dmax[0] = FMUL(f_neg(zsoil[0]), smcav);                          // :7661
    float dice = FMUL(f_neg(zsoil[0]), sice[0]);                     // :7662
    dmax[0] = FMUL(dmax[0],
                   FSUB(K_ONE, FDIV(FSUB(FADD(sh2o[0], sice[0]), par[P_SMCWLT]),
                                    smcav)));                        // :7663
    float dd = dmax[0];                                              // :7665

    for (int k = 1; k < NSOIL; ++k) {                                // :7667
        dice = FADD(dice, FMUL(FSUB(zsoil[k - 1], zsoil[k]), sice[k]));
        dmax[k] = FMUL(FSUB(zsoil[k - 1], zsoil[k]), smcav);
        dmax[k] = FMUL(dmax[k],
                       FSUB(K_ONE,
                            FDIV(FSUB(FADD(sh2o[k], sice[k]), par[P_SMCWLT + k]),
                                 smcav)));
        dd = FADD(dd, dmax[k]);
    }

    float val = FSUB(K_ONE, glibc_expf(f_neg(FMUL(par[P_KDT], dt1))));  // :7674
    float ddt = FMUL(dd, val);                                       // :7675
    float px = f_max(K_ZERO, FMUL(qinsur, dt));                      // :7676
    float infmax = FDIV(FMUL(px, FDIV(ddt, FADD(px, ddt))), dt);     // :7677

    float fcr = K_ONE;                                               // :7681
    if (dice > K_1EM2) {                                             // :7682
        // CVFRZ is an INTEGER PARAMETER = 3, so CVFRZ*FRZX is 3.0*FRZX.
        float acrt = FDIV(FMUL(K_CVFRZ, par[P_FRZX]), dice);         // :7683
        float ssum = K_ONE;                                          // :7684
        // IALP1 = CVFRZ-1 = 2, so J runs 1..2 and the inner K is 2 then 1.
        // ACRT**(CVFRZ-J) has an INTEGER exponent: gfortran expands it to
        // multiplications, never to powf.
        ssum = FADD(ssum, FDIV(FMUL(acrt, acrt), K_FLOATK2));        // J = 1
        ssum = FADD(ssum, FDIV(acrt, K_FLOATK1));                    // J = 2
        fcr = FSUB(K_ONE, FMUL(glibc_expf(f_neg(acrt)), ssum));      // :7693
    }

    infmax = FMUL(infmax, fcr);                                      // :7698

    float wdf, wcnd;
    d_wdfcnd2(sh2o[0], sicemax, par[P_SMCMAX], par[P_BEXP],
              par[P_DWSAT], par[P_DKSAT], &wdf, &wcnd);              // :7703
    infmax = f_max(infmax, wcnd);                                    // :7704
    infmax = f_min(infmax, FDIV(px, dt));                            // :7705

    *runsrf = f_max(K_ZERO, FSUB(qinsur, infmax));                   // :7707
    *pddum = FSUB(qinsur, *runsrf);                                  // :7708
}

// --------------------------------------------------------------------------
// SRT -- module_sf_noahmplsm.F:7716-7846, OPT_INF == 1 and OPT_RUN == 3
// --------------------------------------------------------------------------
__device__ void d_srt(const float *par, const float *zsoil, float pddum,
                      const float *etrani, float qseva, const float *smc,
                      const float *fcr, float *rhstt, float *ai, float *bi,
                      float *ci, float *qdrain, float *wcnd)
{
    float wdf[NSOIL], smx[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {                                // :7775
        d_wdfcnd1(smc[k], fcr[k], par[P_SMCMAX + k], par[P_BEXP + k],
                  par[P_DWSAT + k], par[P_DKSAT + k], &wdf[k], &wcnd[k]);
        smx[k] = smc[k];
    }

    float denom[NSOIL], ddz[NSOIL], dsmdz[NSOIL], wflux[NSOIL];
    for (int k = 0; k < NSOIL; ++k) ddz[k] = dsmdz[k] = K_ZERO;
    *qdrain = K_ZERO;

    for (int k = 0; k < NSOIL; ++k) {                                // :7792
        if (k == 0) {
            denom[k] = f_neg(zsoil[k]);
            float temp1 = f_neg(zsoil[k + 1]);
            ddz[k] = FDIV(K_TWO, temp1);
            dsmdz[k] = FDIV(FMUL(K_TWO, FSUB(smx[k], smx[k + 1])), temp1);
            wflux[k] = FADD(FADD(FSUB(FADD(FMUL(wdf[k], dsmdz[k]), wcnd[k]),
                                      pddum), etrani[k]), qseva);
        } else if (k < NSOIL - 1) {
            denom[k] = FSUB(zsoil[k - 1], zsoil[k]);
            float temp1 = FSUB(zsoil[k - 1], zsoil[k + 1]);
            ddz[k] = FDIV(K_TWO, temp1);
            dsmdz[k] = FDIV(FMUL(K_TWO, FSUB(smx[k], smx[k + 1])), temp1);
            wflux[k] = FADD(FSUB(FSUB(FADD(FMUL(wdf[k], dsmdz[k]), wcnd[k]),
                                      FMUL(wdf[k - 1], dsmdz[k - 1])),
                                 wcnd[k - 1]), etrani[k]);
        } else {
            denom[k] = FSUB(zsoil[k - 1], zsoil[k]);
            *qdrain = FMUL(par[P_SLOPE], wcnd[k]);                   // :7807
            wflux[k] = FADD(FADD(FSUB(f_neg(FMUL(wdf[k - 1], dsmdz[k - 1])),
                                      wcnd[k - 1]), etrani[k]), *qdrain);
        }
    }

    for (int k = 0; k < NSOIL; ++k) {                                // :7828
        if (k == 0) {
            ai[k] = K_ZERO;
            bi[k] = FDIV(FMUL(wdf[k], ddz[k]), denom[k]);
            ci[k] = f_neg(bi[k]);
        } else if (k < NSOIL - 1) {
            ai[k] = f_neg(FDIV(FMUL(wdf[k - 1], ddz[k - 1]), denom[k]));
            ci[k] = f_neg(FDIV(FMUL(wdf[k], ddz[k]), denom[k]));
            bi[k] = f_neg(FADD(ai[k], ci[k]));
        } else {
            ai[k] = f_neg(FDIV(FMUL(wdf[k - 1], ddz[k - 1]), denom[k]));
            ci[k] = K_ZERO;
            bi[k] = f_neg(FADD(ai[k], ci[k]));
        }
        rhstt[k] = FDIV(wflux[k], f_neg(denom[k]));                  // :7843
    }
}

// --------------------------------------------------------------------------
// SSTEP -- module_sf_noahmplsm.F:7850-7973, OPT_RUN == 3
// dz is DZSNSO(1:NSOIL); the snow slots are never read.
// --------------------------------------------------------------------------
__device__ float d_sstep(const float *par, float dt, const float *dz,
                         const float *sice, float *sh2o, float *ai, float *bi,
                         float *ci, float *rhstt, float *smc)
{
    float wplus = K_ZERO;                                            // :7894

    for (int k = 0; k < NSOIL; ++k) {                                // :7896
        rhstt[k] = FMUL(rhstt[k], dt);
        ai[k] = FMUL(ai[k], dt);
        bi[k] = FADD(K_ONE, FMUL(bi[k], dt));
        ci[k] = FMUL(ci[k], dt);
    }

    float rhsttin[NSOIL], ciin[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {                                // :7906
        rhsttin[k] = rhstt[k];
        ciin[k] = ci[k];
    }

    d_rosr12(ai, bi, ciin, rhsttin, ci, rhstt);                      // :7913

    for (int k = 0; k < NSOIL; ++k) sh2o[k] = FADD(sh2o[k], ci[k]);  // :7915

    // The OPT_RUN==5 block at 7923-7947 is dead.

    for (int k = NSOIL - 1; k >= 1; --k) {                           // :7951
        float epore = f_max(K_1EM4_UP, FSUB(par[P_SMCMAX + k], sice[k]));
        wplus = FMUL(f_max(FSUB(sh2o[k], epore), K_ZERO), dz[k]);
        sh2o[k] = f_min(epore, sh2o[k]);
        sh2o[k - 1] = FADD(sh2o[k - 1], FDIV(wplus, dz[k - 1]));
    }

    float epore = f_max(K_1EM4_L1, FSUB(par[P_SMCMAX], sice[0]));    // :7958
    wplus = FMUL(f_max(FSUB(sh2o[0], epore), K_ZERO), dz[0]);        // :7959
    sh2o[0] = f_min(epore, sh2o[0]);                                 // :7960

    if (wplus > K_ZERO) {                                            // :7962
        sh2o[1] = FADD(sh2o[1], FDIV(wplus, dz[1]));                 // :7963
        for (int k = 1; k < NSOIL - 1; ++k) {                        // :7964
            float e = f_max(K_1EM4_DOWN, FSUB(par[P_SMCMAX + k], sice[k]));
            wplus = FMUL(f_max(FSUB(sh2o[k], e), K_ZERO), dz[k]);
            sh2o[k] = f_min(e, sh2o[k]);
            sh2o[k + 1] = FADD(sh2o[k + 1], FDIV(wplus, dz[k + 1]));
        }
        float e = f_max(K_1EM4_DOWN,
                        FSUB(par[P_SMCMAX + NSOIL - 1], sice[NSOIL - 1]));
        wplus = FMUL(f_max(FSUB(sh2o[NSOIL - 1], e), K_ZERO), dz[NSOIL - 1]);
        sh2o[NSOIL - 1] = f_min(e, sh2o[NSOIL - 1]);                 // :7971
    }

    for (int k = 0; k < NSOIL; ++k) smc[k] = FADD(sh2o[k], sice[k]); // :7974
    return wplus;
}

// --------------------------------------------------------------------------
// SOILWATER -- module_sf_noahmplsm.F:7234-7556, OPT_RUN == 3
// --------------------------------------------------------------------------
__device__ void d_soilwater(const float *par, int urban, float dt,
                            const float *zsoil, const float *dz, float qinsur,
                            float qseva, const float *etrani, const float *sice,
                            float *sh2o, float *smc, float *runsub,
                            float *runsrf_o, float *qdrain_o, float *wcnd_o,
                            float *fcrmax_o)
{
    float runsrf = K_ZERO;                                           // :7318
    float pddum = K_ZERO;                                            // :7319
    float rsat = K_ZERO;                                             // :7320

    for (int k = 0; k < NSOIL; ++k) {                                // :7324
        float epore = f_max(K_1EM4_RSAT, FSUB(par[P_SMCMAX + k], sice[k]));
        rsat = FADD(rsat, FMUL(f_max(K_ZERO, FSUB(sh2o[k], epore)), dz[k]));
        sh2o[k] = f_min(epore, sh2o[k]);
    }

    // Scalar, one layer at a time.  gfortran vectorises this loop into
    // libmvec at -O2 and gets a different answer; see the header.
    float fcr[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {                                // :7333
        float fice = f_min(K_ONE, FDIV(sice[k], par[P_SMCMAX + k]));
        fcr[k] = FDIV(f_max(K_ZERO,
                            FSUB(glibc_expf(f_neg(FMUL(K_A, FSUB(K_ONE, fice)))),
                                 glibc_expf(f_neg(K_A)))),
                      FSUB(K_ONE, glibc_expf(f_neg(K_A))));
    }

    float sicemax = K_ZERO;                                          // :7341
    float fcrmax = K_ZERO;                                           // :7342
    for (int k = 0; k < NSOIL; ++k) {                                // :7344
        if (sice[k] > sicemax) sicemax = sice[k];
        if (fcr[k] > fcrmax) fcrmax = fcr[k];
    }

    if (urban) fcr[0] = K_0P95;                                      // :7361

    d_infil(par, dt, zsoil, sh2o, sice, sicemax, qinsur, &pddum, &runsrf);

    int niter = 3;                                                   // :7440
    if (FMUL(pddum, dt) > FMUL(dz[0], par[P_SMCMAX])) niter *= 2;    // :7443
    float dtfine = FDIV(dt, (float) niter);                          // :7449

    float qdrain_save = K_ZERO, runsrf_save = K_ZERO;                // :7453
    float qdrain = K_ZERO;
    float rhstt[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL];

    for (int it = 0; it < niter; ++it) {                             // :7456
        if (qinsur > K_ZERO) {                                       // :7457
            d_infil(par, dtfine, zsoil, sh2o, sice, sicemax, qinsur,
                    &pddum, &runsrf);
        }
        // SRT reads SMC, not SH2O, under OPT_INF == 1.
        d_srt(par, zsoil, pddum, etrani, qseva, smc, fcr,
              rhstt, ai, bi, ci, &qdrain, wcnd_o);
        float wplus = d_sstep(par, dtfine, dz, sice, sh2o, ai, bi, ci,
                              rhstt, smc);
        rsat = FADD(rsat, wplus);                                    // :7489
        qdrain_save = FADD(qdrain_save, qdrain);                     // :7490
        runsrf_save = FADD(runsrf_save, runsrf);                     // :7491
    }

    qdrain = FDIV(qdrain_save, (float) niter);                       // :7494
    runsrf = FDIV(runsrf_save, (float) niter);                       // :7495
    runsrf = FADD(FMUL(runsrf, K_1000_RSR), FDIV(FMUL(rsat, K_1000_RST), dt));
    qdrain = FMUL(qdrain, K_1000_QDR);                               // :7498

    float mliq[NSOIL];
    for (int k = 0; k < NSOIL; ++k)                                  // :7548
        mliq[k] = FMUL(FMUL(sh2o[k], dz[k]), K_1000_MLQ);

    for (int iz = 0; iz < NSOIL - 1; ++iz) {                         // :7552
        float xs = (mliq[iz] < K_ZERO) ? FSUB(K_WATMIN, mliq[iz]) : K_ZERO;
        mliq[iz] = FADD(mliq[iz], xs);
        mliq[iz + 1] = FSUB(mliq[iz + 1], xs);
    }
    {
        int iz = NSOIL - 1;                                          // :7562
        float xs = (mliq[iz] < K_WATMIN) ? FSUB(K_WATMIN, mliq[iz]) : K_ZERO;
        mliq[iz] = FADD(mliq[iz], xs);
        *runsub = FSUB(*runsub, FDIV(xs, dt));                       // :7569
    }
    for (int iz = 0; iz < NSOIL; ++iz)                               // :7572
        sh2o[iz] = FDIV(mliq[iz], FMUL(dz[iz], K_1000_BAC));

    // SMC is deliberately NOT recomputed: the WATMIN fixup rewrites SH2O and
    // never touches SMC, so on exit the two are inconsistent by exactly that
    // correction.  The oracle pins the difference.
    *runsrf_o = runsrf;
    *qdrain_o = qdrain;
    *fcrmax_o = fcrmax;
}

// ==========================================================================
// Host-facing kernels.  One thread per case; `par` is P_STRIDE floats per
// case, `fin` is IN_STRIDE, `fout` is OUT_STRIDE.
// ==========================================================================

extern "C" __global__ void k_canwater(const float *par, const float *fin,
                                      const int *iin, float *fout, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *p = par + (size_t) i * P_STRIDE;
    const float *x = fin + (size_t) i * IN_STRIDE;
    d_canwater(p[P_CH2OP], x[0], x[1], x[2], x[3], x[4], x[5], x[6],
               iin[i], x[7], x[8], x[9], fout + (size_t) i * OUT_STRIDE);
}

extern "C" __global__ void k_infil(const float *par, const float *fin,
                                   float *fout, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *p = par + (size_t) i * P_STRIDE;
    const float *x = fin + (size_t) i * IN_STRIDE;
    float pddum = x[15], runsrf = x[16];
    d_infil(p, x[0], x + 1, x + 5, x + 9, x[13], x[14], &pddum, &runsrf);
    float *o = fout + (size_t) i * OUT_STRIDE;
    o[0] = pddum;
    o[1] = runsrf;
}

extern "C" __global__ void k_srt(const float *par, const float *fin,
                                 float *fout, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *p = par + (size_t) i * P_STRIDE;
    const float *x = fin + (size_t) i * IN_STRIDE;
    float rhstt[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], wcnd[NSOIL], qdrain;
    d_srt(p, x + 1, x[0], x + 5, x[9], x + 10, x + 14,
          rhstt, ai, bi, ci, &qdrain, wcnd);
    float *o = fout + (size_t) i * OUT_STRIDE;
    for (int k = 0; k < NSOIL; ++k) {
        o[k] = rhstt[k];
        o[NSOIL + k] = ai[k];
        o[2 * NSOIL + k] = bi[k];
        o[3 * NSOIL + k] = ci[k];
        o[4 * NSOIL + k] = wcnd[k];
    }
    o[5 * NSOIL] = qdrain;
}

extern "C" __global__ void k_sstep(const float *par, const float *fin,
                                   float *fout, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *p = par + (size_t) i * P_STRIDE;
    const float *x = fin + (size_t) i * IN_STRIDE;
    float sh2o[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], rhstt[NSOIL],
          smc[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        sh2o[k] = x[5 + k];
        ai[k] = x[9 + k];
        bi[k] = x[13 + k];
        ci[k] = x[17 + k];
        rhstt[k] = x[21 + k];
    }
    float wplus = d_sstep(p, x[0], x + 1, x + 25, sh2o, ai, bi, ci, rhstt, smc);
    float *o = fout + (size_t) i * OUT_STRIDE;
    for (int k = 0; k < NSOIL; ++k) {
        o[k] = sh2o[k];
        o[NSOIL + k] = smc[k];
        o[2 * NSOIL + k] = ai[k];
        o[3 * NSOIL + k] = bi[k];
        o[4 * NSOIL + k] = ci[k];
        o[5 * NSOIL + k] = rhstt[k];
    }
    o[6 * NSOIL] = wplus;
}

extern "C" __global__ void k_soilwater(const float *par, const float *fin,
                                       const int *iin, float *fout, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *p = par + (size_t) i * P_STRIDE;
    const float *x = fin + (size_t) i * IN_STRIDE;
    float sh2o[NSOIL], smc[NSOIL], sice[NSOIL], wcnd[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        sh2o[k] = x[15 + k];
        smc[k] = x[19 + k];
        sice[k] = x[11 + k];
    }
    float runsub = x[23], runsrf, qdrain, fcrmax;
    d_soilwater(p, iin[i], x[0], x + 1, x + 5, x[9], x[10], x + 24, sice,
                sh2o, smc, &runsub, &runsrf, &qdrain, wcnd, &fcrmax);
    float *o = fout + (size_t) i * OUT_STRIDE;
    for (int k = 0; k < NSOIL; ++k) {
        o[k] = sh2o[k];
        o[NSOIL + k] = smc[k];
        o[2 * NSOIL + k] = wcnd[k];
    }
    o[3 * NSOIL] = runsrf;
    o[3 * NSOIL + 1] = qdrain;
    o[3 * NSOIL + 2] = runsub;
    o[3 * NSOIL + 3] = fcrmax;
}

// --------------------------------------------------------------------------
// libm sweeps, so the device transcriptions are gated over a domain far wider
// than the leaves reach.  A mis-folded __constant__ entry could otherwise hide
// inside a branch no case selects.
// --------------------------------------------------------------------------
extern "C" __global__ void k_expf_sweep(const float *x, float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = glibc_expf(x[i]);
}

extern "C" __global__ void k_powf_sweep(const float *x, const float *e,
                                        float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = glibc_powf(x[i], e[i]);
}
