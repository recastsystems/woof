// gpuwm/core/kernels/noahmp_water.cu
//
// CUDA half of the Noah-MP WATER assembly.  Bitwise-equal to
// gpuwm/core/noahmp_water.py and to the WRF v4.6.1 oracle
// (tree d66e442fccc04111067e29274c9f9eaccc3cef28,
//  sha256(phys/module_sf_noahmplsm.F) =
//  bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282).
//
// Covers WATER (5954-6261) and, through it, everything it calls: CANWATER,
// SOILWATER, INFIL, SRT, SSTEP, WDFCND1/2, ROSR12, and the snow chain
// SNOWFALL, COMPACT, COMBINE, DIVIDE, COMBO, SNOWH2O, SNOWWATER.
//
// Why this file repeats two other .cu files instead of including them
// -------------------------------------------------------------------
// CuPy's RawModule compiles from a string with no include path, so a kernel
// source has to be self-contained.  The two sections below are therefore
// *copies* of noahmp_soilwater.cu and noahmp_snow.cu, delimited by
// `>>> BEGIN imported section` / `<<< END imported section` markers.
//
// They are not maintained here.  tests/test_noahmp_water_cuda.py re-derives
// both sections from their source files by the documented transform and
// requires byte equality, so a change to either lane that this file has not
// picked up fails a test rather than silently forking a transcription that is
// already gated at max_ulp 0.
//
// The transform is: take noahmp_soilwater.cu from `#define NSOIL 4` to the
// start of its host-facing kernels, and noahmp_snow.cu from `#define NSNOW 3`
// to the start of its entry points; drop from the snow copy the four pieces
// the soil copy already provides (the __exp2f_data tables, the rounding-pinned
// FADD/DADD macros, f_max/f_min, and glibc_expf); drop the per-leaf host
// layout macros from both; and move the snow lane's private constant table
// into its own namespace, C_F32 -> C_SN_F32 and K_* -> SN_K_*.  Nothing else
// is touched, so every arithmetic site keeps the exact form its own lane's
// device gate already accepted.
//
// The three rules the two imported sections are built on hold here unchanged:
//
// 1. Every float32 operation uses an explicit rounding intrinsic
//    (__fadd_rn/__fsub_rn/__fmul_rn/__fdiv_rn) and every float64 operation
//    inside glibc_expf/glibc_powf uses __dadd_rn/__dsub_rn/__dmul_rn/
//    __fma_rn.  That pins the hardware rounding mode AND makes nvcc's
//    contraction pass a no-op, so -fmad=true cannot fuse a site gfortran did
//    not fuse.
// 2. Every constant lives in __constant__ memory as a bit pattern.  ptxas
//    12.8's constant folder does not honour round-to-nearest-even on literal
//    arrays.  WATER's own three constants (1000.0, 0.001 and WSLMAX) are in
//    C_W_F32 below for the same reason.
// 3. EXP and ** are glibc calls in the oracle, so glibc_expf and glibc_powf
//    transcribe glibc 2.39's own algorithms.  CUDA's expf, __expf, exp2f and
//    powf are all different functions and none can hold a max_ulp-0 gate.
//
// And the fourth thing this file must NOT do, which WATER inherits from
// SOILWATER: vectorise the frozen-fraction loop the way gfortran does at
// WRF's own -O2, where it calls glibc's libmvec _ZGVbN4v_expf.  The fixture
// is built at -O0 and build_water.sh fails on any _ZGV* symbol; see
// gpuwm/data/noahmp/oracle/PROVENANCE-water.md.
//
// Pinned option identity: opt_run=3, opt_inf=1, opt_tdrn=0, opt_irr=0,
// soiltstep=0.0 (so soil_update_steps=1 and calculate_soil=.true.).
// Everything they kill is absent, not stubbed:
//   opt_run=3   GROUNDWATER (6225-6231) and SHALLOWWATERTABLE (6242-6250);
//   opt_irr=0   FLOOD_IRRIGATION (6188-6193), MICRO_IRRIGATION (6196-6202) --
//               the kill runs through the caller, which cannot deliver a
//               positive IRAMTFI/IRAMTMI;
//   opt_tdrn=0  the tile drain, so QTLDRN is identically 0.0;
//   WRF_HYDRO   undefined, so the sfcheadrt term at 6174 does not exist.
//
// QIN (6052) and QDIS (6053) are INTENT(OUT) and no live statement writes
// either, so they are not outputs of this kernel.  The oracle drives both
// non-zero on every case and requires entry == exit; see the port docstring.

// >>> BEGIN imported section: noahmp_soilwater.cu
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
// <<< END imported section: noahmp_soilwater.cu

// >>> BEGIN imported section: noahmp_snow.cu (K_* -> SN_K_*)
#define NSNOW 3
#define NFULL 7 /* NSNOW + NSOIL */

// Flat host<->device layout.  Slot meanings are per leaf; see the comment on
// each kernel.  Kept uniform so one host harness drives all seven.
#define ST_SNOWH   0
#define ST_SNEQV   1
#define ST_SNICE   2  /* 3 slots, j = -2..0 */
#define ST_SNLIQ   5  /* 3 slots            */
#define ST_STC     8  /* 7 slots, j = -2..4 */
#define ST_ZSNSO  15  /* 7 slots            */
#define ST_DZSNSO 22  /* 7 slots            */
#define ST_SH2O   29  /* 4 slots, j = 1..4  */
#define ST_SICE   33  /* 4 slots            */
#define NSTATE    37

// --------------------------------------------------------------------------
// Every float32 constant these leaves use, as bit patterns.
// SN_K_C5 duplicates SN_K_TWO and SN_K_P2 duplicates SN_K_P20 on purpose: the Fortran
// writes them as separate literals in separate expressions, and separate
// slots let the mutation study probe each site independently.
// --------------------------------------------------------------------------
__constant__ unsigned int C_SN_F32[32] = {
    0x00000000u, /*  0 SN_K_ZERO      0.0        */
    0x3F800000u, /*  1 SN_K_ONE       1.0        */
    0x3F000000u, /*  2 SN_K_HALF      0.5        */
    0x40000000u, /*  3 SN_K_TWO       2.0        */
    0x4388947Bu, /*  4 SN_K_TFRZ      273.16     */
    0x48A2E400u, /*  5 SN_K_HFUS      0.3336e6   */
    0x4A7F9D80u, /*  6 SN_K_CWAT      4.188e6    */
    0x49FF9D80u, /*  7 SN_K_CICE      2.094e6    */
    0x447A0000u, /*  8 SN_K_DENH2O    1000.0     */
    0x44654000u, /*  9 SN_K_DENICE    917.0      */
    0x3CAC0831u, /* 10 SN_K_C2        21.0e-3    */
    0x3627C5ACu, /* 11 SN_K_C3        2.5e-6     */
    0x3D23D70Au, /* 12 SN_K_C4        0.04       */
    0x40000000u, /* 13 SN_K_C5        2.0        */
    0x42C80000u, /* 14 SN_K_DM        100.0      */
    0x49A25A80u, /* 15 SN_K_ETA0      1.33e6     */
    0xBD3C6A7Fu, /* 16 SN_K_NEG46EM3  -46.0e-3   */
    0xBDA3D70Au, /* 17 SN_K_NEG008    -0.08      */
    0x3A83126Fu, /* 18 SN_K_P001      0.001      */
    0x3C23D70Au, /* 19 SN_K_P01       0.01       */
    0x358637BDu, /* 20 SN_K_1EM6      1.0e-6     */
    0xBF000000u, /* 21 SN_K_NEGHALF   -0.5       */
    0x43FA0000u, /* 22 SN_K_500       500.0      */
    0x42480000u, /* 23 SN_K_50        50.0       */
    0x3CCCCCCDu, /* 24 SN_K_P025      0.025      */
    0x3DCCCCCDu, /* 25 SN_K_P1        0.1        */
    0x3D4CCCCDu, /* 26 SN_K_P05       0.05       */
    0x3E4CCCCDu, /* 27 SN_K_P20       0.20       */
    0x3E4CCCCDu, /* 28 SN_K_P2        0.2        */
    0x3ECCCCCDu, /* 29 SN_K_MAXLIQ    0.4        */
    0x322BCC77u, /* 30 SN_K_1EM8      1.0e-8     */
    0x459C4000u, /* 31 SN_K_5000      5000.0     */
};
#define SN_K_ZERO      __uint_as_float(C_SN_F32[0])
#define SN_K_ONE       __uint_as_float(C_SN_F32[1])
#define SN_K_HALF      __uint_as_float(C_SN_F32[2])
#define SN_K_TWO       __uint_as_float(C_SN_F32[3])
#define SN_K_TFRZ      __uint_as_float(C_SN_F32[4])
#define SN_K_HFUS      __uint_as_float(C_SN_F32[5])
#define SN_K_CWAT      __uint_as_float(C_SN_F32[6])
#define SN_K_CICE      __uint_as_float(C_SN_F32[7])
#define SN_K_DENH2O    __uint_as_float(C_SN_F32[8])
#define SN_K_DENICE    __uint_as_float(C_SN_F32[9])
#define SN_K_C2        __uint_as_float(C_SN_F32[10])
#define SN_K_C3        __uint_as_float(C_SN_F32[11])
#define SN_K_C4        __uint_as_float(C_SN_F32[12])
#define SN_K_C5        __uint_as_float(C_SN_F32[13])
#define SN_K_DM        __uint_as_float(C_SN_F32[14])
#define SN_K_ETA0      __uint_as_float(C_SN_F32[15])
#define SN_K_NEG46EM3  __uint_as_float(C_SN_F32[16])
#define SN_K_NEG008    __uint_as_float(C_SN_F32[17])
#define SN_K_P001      __uint_as_float(C_SN_F32[18])
#define SN_K_P01       __uint_as_float(C_SN_F32[19])
#define SN_K_1EM6      __uint_as_float(C_SN_F32[20])
#define SN_K_NEGHALF   __uint_as_float(C_SN_F32[21])
#define SN_K_500       __uint_as_float(C_SN_F32[22])
#define SN_K_50        __uint_as_float(C_SN_F32[23])
#define SN_K_P025      __uint_as_float(C_SN_F32[24])
#define SN_K_P1        __uint_as_float(C_SN_F32[25])
#define SN_K_P05       __uint_as_float(C_SN_F32[26])
#define SN_K_P20       __uint_as_float(C_SN_F32[27])
#define SN_K_P2        __uint_as_float(C_SN_F32[28])
#define SN_K_MAXLIQ    __uint_as_float(C_SN_F32[29])
#define SN_K_1EM8      __uint_as_float(C_SN_F32[30])
#define SN_K_5000      __uint_as_float(C_SN_F32[31])

// COMBINE's `DATA DZMIN /0.025, 0.025, 0.1/`.  A literal array of FP32
// constants is precisely the shape ptxas 12.8 mis-folds, so it lives here.
__constant__ unsigned int C_DZMIN[3] = { 0x3CCCCCCDu, 0x3CCCCCCDu, 0x3DCCCCCDu };
#define DZMIN(i) __uint_as_float(C_DZMIN[(i)])

#define QNAN __uint_as_float(0x7FC00000u)

// --------------------------------------------------------------------------
// column state, in WRF's index convention
// --------------------------------------------------------------------------
typedef struct {
    int   isnow;
    float snowh, sneqv;
    float snice[NSNOW];    /* j = -NSNOW+1 .. 0 */
    float snliq[NSNOW];
    float stc[NFULL];      /* j = -NSNOW+1 .. NSOIL */
    float zsnso[NFULL];
    float dzsnso[NFULL];
    float sh2o[NSOIL];     /* j = 1 .. NSOIL */
    float sice[NSOIL];
} Col;

#define SNICE(j)  c->snice[(j) + NSNOW - 1]
#define SNLIQ(j)  c->snliq[(j) + NSNOW - 1]
#define STC(j)    c->stc[(j) + NSNOW - 1]
#define ZSNSO(j)  c->zsnso[(j) + NSNOW - 1]
#define DZSNSO(j) c->dzsnso[(j) + NSNOW - 1]
#define SH2O(j)   c->sh2o[(j) - 1]
#define SICE(j)   c->sice[(j) - 1]

__device__ void col_load(Col *c, const float *in, int isnow)
{
    c->isnow = isnow;
    c->snowh = in[ST_SNOWH];
    c->sneqv = in[ST_SNEQV];
    for (int k = 0; k < NSNOW; ++k) { c->snice[k] = in[ST_SNICE + k];
                                      c->snliq[k] = in[ST_SNLIQ + k]; }
    for (int k = 0; k < NFULL; ++k) { c->stc[k]    = in[ST_STC + k];
                                      c->zsnso[k]  = in[ST_ZSNSO + k];
                                      c->dzsnso[k] = in[ST_DZSNSO + k]; }
    for (int k = 0; k < NSOIL; ++k) { c->sh2o[k] = in[ST_SH2O + k];
                                      c->sice[k] = in[ST_SICE + k]; }
}

__device__ void col_store(const Col *c, float *out, int *iout)
{
    *iout = c->isnow;
    out[ST_SNOWH] = c->snowh;
    out[ST_SNEQV] = c->sneqv;
    for (int k = 0; k < NSNOW; ++k) { out[ST_SNICE + k] = c->snice[k];
                                      out[ST_SNLIQ + k] = c->snliq[k]; }
    for (int k = 0; k < NFULL; ++k) { out[ST_STC + k]    = c->stc[k];
                                      out[ST_ZSNSO + k]  = c->zsnso[k];
                                      out[ST_DZSNSO + k] = c->dzsnso[k]; }
    for (int k = 0; k < NSOIL; ++k) { out[ST_SH2O + k] = c->sh2o[k];
                                      out[ST_SICE + k] = c->sice[k]; }
}

// ==========================================================================
// COMBO -- lines 6920-6970
// ==========================================================================
__device__ void combo(float *dz, float *wliq, float *wice, float *t,
                      float dz2, float wliq2, float wice2, float t2)
{
    float dzc = FADD(*dz, dz2);
    float wicec = FADD(*wice, wice2);
    float wliqc = FADD(*wliq, wliq2);
    float h = FADD(FMUL(FADD(FMUL(SN_K_CICE, *wice), FMUL(SN_K_CWAT, *wliq)),
                        FSUB(*t, SN_K_TFRZ)), FMUL(SN_K_HFUS, *wliq));
    float h2 = FADD(FMUL(FADD(FMUL(SN_K_CICE, wice2), FMUL(SN_K_CWAT, wliq2)),
                         FSUB(t2, SN_K_TFRZ)), FMUL(SN_K_HFUS, wliq2));
    float hc = FADD(h, h2);
    float tc;
    if (hc < SN_K_ZERO) {
        tc = FADD(SN_K_TFRZ, FDIV(hc, FADD(FMUL(SN_K_CICE, wicec), FMUL(SN_K_CWAT, wliqc))));
    } else if (hc <= FMUL(SN_K_HFUS, wliqc)) {
        tc = SN_K_TFRZ;
    } else {
        tc = FADD(SN_K_TFRZ, FDIV(FSUB(hc, FMUL(SN_K_HFUS, wliqc)),
                               FADD(FMUL(SN_K_CICE, wicec), FMUL(SN_K_CWAT, wliqc))));
    }
    *dz = dzc; *wice = wicec; *wliq = wliqc; *t = tc;
}

// ==========================================================================
// SNOWFALL -- lines 6539-6606
// ==========================================================================
__device__ void snowfall(Col *c, float dt, float qsnow, float snowhin, float sfctmp)
{
    int newnode = 0;

    if (c->isnow == 0 && qsnow > SN_K_ZERO) {
        c->snowh = FADD(c->snowh, FMUL(snowhin, dt));
        c->sneqv = FADD(c->sneqv, FMUL(qsnow, dt));
    }

    // C.He removed the QSNOW>0 condition so ISNOW can still be adjusted from
    // SNOWH alone when nothing is falling.
    if (c->isnow == 0 && c->snowh >= SN_K_P025) {
        c->isnow = -1;
        newnode = 1;
        DZSNSO(0) = c->snowh;
        c->snowh = SN_K_ZERO;
        STC(0) = f_min(SN_K_TFRZ, sfctmp);
        SNICE(0) = c->sneqv;
        SNLIQ(0) = SN_K_ZERO;
    }

    if (c->isnow < 0 && newnode == 0 && qsnow > SN_K_ZERO) {
        SNICE(c->isnow + 1) = FADD(SNICE(c->isnow + 1), FMUL(qsnow, dt));
        DZSNSO(c->isnow + 1) = FADD(DZSNSO(c->isnow + 1), FMUL(snowhin, dt));
    }
}

// ==========================================================================
// COMPACT -- lines 6974-7081
// ==========================================================================
__device__ void compact(Col *c, float dt, const int *imelt, const float *ficeold)
{
    float fice[NSNOW];
    float burden = SN_K_ZERO;

    for (int j = c->isnow + 1; j <= 0; ++j) {
        int k = j + NSNOW - 1;
        float wx = FADD(SNICE(j), SNLIQ(j));
        fice[k] = FDIV(SNICE(j), wx);
        float voidf = FSUB(SN_K_ONE,
            FDIV(FADD(FDIV(SNICE(j), SN_K_DENICE), FDIV(SNLIQ(j), SN_K_DENH2O)), DZSNSO(j)));

        if (voidf > SN_K_P001 && SNICE(j) > SN_K_P1) {
            float bi = FDIV(SNICE(j), DZSNSO(j));
            float td = f_max(SN_K_ZERO, FSUB(SN_K_TFRZ, STC(j)));
            float dexpf = glibc_expf(FMUL(-SN_K_C4, td));

            float ddz1 = FMUL(-SN_K_C3, dexpf);
            if (bi > SN_K_DM)
                ddz1 = FMUL(ddz1, glibc_expf(FMUL(SN_K_NEG46EM3, FSUB(bi, SN_K_DM))));

            if (SNLIQ(j) > FMUL(SN_K_P01, DZSNSO(j)))
                ddz1 = FMUL(ddz1, SN_K_C5);

            float ddz2 = FDIV(FMUL(-FADD(burden, FMUL(SN_K_HALF, wx)),
                                   glibc_expf(FSUB(FMUL(SN_K_NEG008, td), FMUL(SN_K_C2, bi)))),
                              SN_K_ETA0);

            float ddz3;
            if (imelt[k] == 1) {
                ddz3 = f_max(SN_K_ZERO, FDIV(FSUB(ficeold[k], fice[k]),
                                          f_max(SN_K_1EM6, ficeold[k])));
                ddz3 = FDIV(-ddz3, dt);
            } else {
                ddz3 = SN_K_ZERO;
            }

            float pdzdtc = FMUL(FADD(FADD(ddz1, ddz2), ddz3), dt);
            pdzdtc = f_max(SN_K_NEGHALF, pdzdtc);

            DZSNSO(j) = FMUL(DZSNSO(j), FADD(SN_K_ONE, pdzdtc));
            DZSNSO(j) = f_max(DZSNSO(j), FADD(FDIV(SNICE(j), SN_K_DENICE),
                                              FDIV(SNLIQ(j), SN_K_DENH2O)));
            // C.He: constrain snow density to 50~500 kg/m3
            DZSNSO(j) = f_min(f_max(DZSNSO(j), FDIV(FADD(SNICE(j), SNLIQ(j)), SN_K_500)),
                              FDIV(FADD(SNICE(j), SNLIQ(j)), SN_K_50));
        }

        burden = FADD(burden, wx);
    }
}

// ==========================================================================
// COMBINE -- lines 6610-6788
// ==========================================================================
__device__ void combine(Col *c, float *ponding1, float *ponding2)
{
    int isnow_old = c->isnow;

    for (int j = isnow_old + 1; j <= 0; ++j) {
        if (SNICE(j) <= SN_K_P1) {
            if (j != 0) {
                SNLIQ(j + 1) = FADD(SNLIQ(j + 1), SNLIQ(j));
                SNICE(j + 1) = FADD(SNICE(j + 1), SNICE(j));
                DZSNSO(j + 1) = FADD(DZSNSO(j + 1), DZSNSO(j));
            } else {
                if (c->isnow < -1) {
                    SNLIQ(j - 1) = FADD(SNLIQ(j - 1), SNLIQ(j));
                    SNICE(j - 1) = FADD(SNICE(j - 1), SNICE(j));
                    DZSNSO(j - 1) = FADD(DZSNSO(j - 1), DZSNSO(j));
                } else {
                    if (SNICE(j) >= SN_K_ZERO) {
                        *ponding1 = SNLIQ(j);
                        c->sneqv = SNICE(j);
                        c->snowh = DZSNSO(j);
                    } else { // SNICE over-sublimated earlier
                        *ponding1 = FADD(SNLIQ(j), SNICE(j));
                        if (*ponding1 < SN_K_ZERO) {
                            SICE(1) = FADD(SICE(1),
                                FDIV(*ponding1, FMUL(DZSNSO(1), SN_K_DENH2O)));
                            *ponding1 = SN_K_ZERO;
                        }
                        c->sneqv = SN_K_ZERO;
                        c->snowh = SN_K_ZERO;
                    }
                    SNLIQ(j) = SN_K_ZERO;
                    SNICE(j) = SN_K_ZERO;
                    DZSNSO(j) = SN_K_ZERO;
                }
            }

            if (j > c->isnow + 1 && c->isnow < -1) {
                for (int i = j; i >= c->isnow + 2; --i) {
                    STC(i) = STC(i - 1);
                    SNLIQ(i) = SNLIQ(i - 1);
                    SNICE(i) = SNICE(i - 1);
                    DZSNSO(i) = DZSNSO(i - 1);
                }
            }
            c->isnow = c->isnow + 1;
        }
    }

    if (SICE(1) < SN_K_ZERO) {
        SH2O(1) = FADD(SH2O(1), SICE(1));
        SICE(1) = SN_K_ZERO;
    }

    if (c->isnow == 0) return; // MB: get out if no longer multi-layer

    c->sneqv = SN_K_ZERO;
    c->snowh = SN_K_ZERO;
    float zwice = SN_K_ZERO;
    float zwliq = SN_K_ZERO;

    for (int j = c->isnow + 1; j <= 0; ++j) {
        // Fortran is `SNEQV = SNEQV + SNICE(J) + SNLIQ(J)`, which associates
        // left: (SNEQV + SNICE) + SNLIQ.  Summing the layer pair first is a
        // different rounding and costs a ULP.
        c->sneqv = FADD(FADD(c->sneqv, SNICE(j)), SNLIQ(j));
        c->snowh = FADD(c->snowh, DZSNSO(j));
        zwice = FADD(zwice, SNICE(j));
        zwliq = FADD(zwliq, SNLIQ(j));
    }

    if (c->snowh < SN_K_P025 && c->isnow < 0) {
        c->isnow = 0;
        c->sneqv = zwice;
        *ponding2 = zwliq;
        if (c->sneqv <= SN_K_ZERO) c->snowh = SN_K_ZERO;
    }

    if (c->isnow < -1) {
        isnow_old = c->isnow;
        int mssi = 1;

        for (int i = isnow_old + 1; i <= 0; ++i) {
            if (DZSNSO(i) < DZMIN(mssi - 1)) {
                int neibor;
                if (i == c->isnow + 1) {
                    neibor = i + 1;
                } else if (i == 0) {
                    neibor = i - 1;
                } else {
                    neibor = i + 1;
                    if (FADD(DZSNSO(i - 1), DZSNSO(i)) < FADD(DZSNSO(i + 1), DZSNSO(i)))
                        neibor = i - 1;
                }

                int j, l;
                if (neibor > i) { j = neibor; l = i; } else { j = i; l = neibor; }

                combo(&DZSNSO(j), &SNLIQ(j), &SNICE(j), &STC(j),
                      DZSNSO(l), SNLIQ(l), SNICE(l), STC(l));

                if (j - 1 > c->isnow + 1) {
                    for (int k = j - 1; k >= c->isnow + 2; --k) {
                        STC(k) = STC(k - 1);
                        SNICE(k) = SNICE(k - 1);
                        SNLIQ(k) = SNLIQ(k - 1);
                        DZSNSO(k) = DZSNSO(k - 1);
                    }
                }

                c->isnow = c->isnow + 1;
                if (c->isnow >= -1) break;
            } else {
                mssi = mssi + 1;
            }
        }
    }
}

// ==========================================================================
// DIVIDE -- lines 6792-6916
// ==========================================================================
__device__ void divide(Col *c)
{
    // WRF leaves the slots above ABS(ISNOW) undefined.  Poisoned with a quiet
    // NaN here so an accidental read shows up rather than reading a zero; the
    // oracle build proves no emitted value depends on them by re-running the
    // whole fixture under -finit-real=snan.
    float dz[NSNOW], swice[NSNOW], swliq[NSNOW], tsno[NSNOW];
    for (int k = 0; k < NSNOW; ++k) {
        dz[k] = QNAN; swice[k] = QNAN; swliq[k] = QNAN; tsno[k] = QNAN;
    }
#define DZ(i)    dz[(i) - 1]
#define SWICE(i) swice[(i) - 1]
#define SWLIQ(i) swliq[(i) - 1]
#define TSNO(i)  tsno[(i) - 1]

    int isn = c->isnow;
    for (int j = 1; j <= NSNOW; ++j) {
        if (j <= abs(isn)) {
            DZ(j)    = DZSNSO(j + isn);
            SWICE(j) = SNICE(j + isn);
            SWLIQ(j) = SNLIQ(j + isn);
            TSNO(j)  = STC(j + isn);
        }
    }

    int msno = abs(isn);
    float drr, propor, zwice, zwliq, dtdz;

    if (msno == 1) {
        if (DZ(1) > SN_K_P05) {
            msno = 2;
            DZ(1) = FDIV(DZ(1), SN_K_TWO);
            SWICE(1) = FDIV(SWICE(1), SN_K_TWO);
            SWLIQ(1) = FDIV(SWLIQ(1), SN_K_TWO);
            DZ(2) = DZ(1);
            SWICE(2) = SWICE(1);
            SWLIQ(2) = SWLIQ(1);
            TSNO(2) = TSNO(1);
        }
    }

    if (msno > 1) {
        if (DZ(1) > SN_K_P05) {
            drr = FSUB(DZ(1), SN_K_P05);
            propor = FDIV(drr, DZ(1));
            zwice = FMUL(propor, SWICE(1));
            zwliq = FMUL(propor, SWLIQ(1));
            propor = FDIV(SN_K_P05, DZ(1));
            SWICE(1) = FMUL(propor, SWICE(1));
            SWLIQ(1) = FMUL(propor, SWLIQ(1));
            DZ(1) = SN_K_P05;

            combo(&DZ(2), &SWLIQ(2), &SWICE(2), &TSNO(2),
                  drr, zwliq, zwice, TSNO(1));

            // MB raised this limit from 0.10 to 0.20.
            if (msno <= 2 && DZ(2) > SN_K_P20) {
                msno = 3;
                dtdz = FDIV(FSUB(TSNO(1), TSNO(2)), FDIV(FADD(DZ(1), DZ(2)), SN_K_TWO));
                DZ(2) = FDIV(DZ(2), SN_K_TWO);
                SWICE(2) = FDIV(SWICE(2), SN_K_TWO);
                SWLIQ(2) = FDIV(SWLIQ(2), SN_K_TWO);
                DZ(3) = DZ(2);
                SWICE(3) = SWICE(2);
                SWLIQ(3) = SWLIQ(2);
                TSNO(3) = FSUB(TSNO(2), FDIV(FMUL(dtdz, DZ(2)), SN_K_TWO));
                if (TSNO(3) >= SN_K_TFRZ) {
                    TSNO(3) = TSNO(2);
                } else {
                    TSNO(2) = FADD(TSNO(2), FDIV(FMUL(dtdz, DZ(2)), SN_K_TWO));
                }
            }
        }
    }

    if (msno > 2) {
        if (DZ(2) > SN_K_P2) {
            drr = FSUB(DZ(2), SN_K_P2);
            propor = FDIV(drr, DZ(2));
            zwice = FMUL(propor, SWICE(2));
            zwliq = FMUL(propor, SWLIQ(2));
            propor = FDIV(SN_K_P2, DZ(2));
            SWICE(2) = FMUL(propor, SWICE(2));
            SWLIQ(2) = FMUL(propor, SWLIQ(2));
            DZ(2) = SN_K_P2;
            combo(&DZ(3), &SWLIQ(3), &SWICE(3), &TSNO(3),
                  drr, zwliq, zwice, TSNO(2));
        }
    }

    c->isnow = -msno;

    for (int j = c->isnow + 1; j <= 0; ++j) {
        DZSNSO(j) = DZ(j - c->isnow);
        SNICE(j)  = SWICE(j - c->isnow);
        SNLIQ(j)  = SWLIQ(j - c->isnow);
        STC(j)    = TSNO(j - c->isnow);
    }
#undef DZ
#undef SWICE
#undef SWLIQ
#undef TSNO
}

// ==========================================================================
// SNOWH2O -- lines 7085-7230
// ==========================================================================
__device__ float snowh2o(Col *c, float dt, float qsnfro, float qsnsub, float qrain,
                         float ssi, float snow_ret_fac, float *ponding1, float *ponding2)
{
    float vol_liq[NSNOW], vol_ice[NSNOW], epore[NSNOW];
    for (int k = 0; k < NSNOW; ++k) { vol_liq[k] = SN_K_ZERO; vol_ice[k] = SN_K_ZERO;
                                      epore[k] = SN_K_ZERO; }

    // for the case when SNEQV becomes '0' after 'COMBINE'
    if (c->sneqv == SN_K_ZERO) {
        // Barlage: SH2O -> SICE in v3.6
        SICE(1) = FADD(SICE(1), FDIV(FMUL(FSUB(qsnfro, qsnsub), dt),
                                     FMUL(DZSNSO(1), SN_K_DENH2O)));
        if (SICE(1) < SN_K_ZERO) {
            SH2O(1) = FADD(SH2O(1), SICE(1));
            SICE(1) = SN_K_ZERO;
        }
    }

    // shallow snow without a layer: excess sublimation reduces soil water
    if (c->isnow == 0 && c->sneqv > SN_K_ZERO) {
        float temp = c->sneqv;
        c->sneqv = FADD(FSUB(c->sneqv, FMUL(qsnsub, dt)), FMUL(qsnfro, dt));
        float propor = FDIV(c->sneqv, temp);
        c->snowh = f_max(SN_K_ZERO, FMUL(propor, c->snowh));
        c->snowh = f_min(f_max(c->snowh, FDIV(c->sneqv, SN_K_500)),
                         FDIV(c->sneqv, SN_K_50));

        if (c->sneqv < SN_K_ZERO) {
            SICE(1) = FADD(SICE(1), FDIV(c->sneqv, FMUL(DZSNSO(1), SN_K_DENH2O)));
            c->sneqv = SN_K_ZERO;
            c->snowh = SN_K_ZERO;
        }
        if (SICE(1) < SN_K_ZERO) {
            SH2O(1) = FADD(SH2O(1), SICE(1));
            SICE(1) = SN_K_ZERO;
        }
    }

    if (c->snowh <= SN_K_1EM8 || c->sneqv <= SN_K_1EM6) {
        c->snowh = SN_K_ZERO;
        c->sneqv = SN_K_ZERO;
    }

    // for deep snow
    if (c->isnow < 0) {
        float wgdif = FADD(FSUB(SNICE(c->isnow + 1), FMUL(qsnsub, dt)),
                           FMUL(qsnfro, dt));
        SNICE(c->isnow + 1) = wgdif;
        if (wgdif < SN_K_1EM6 && c->isnow < 0)
            combine(c, ponding1, ponding2);
        // KWM: COMBINE can change ISNOW back to 0
        if (c->isnow < 0) {
            SNLIQ(c->isnow + 1) = FADD(SNLIQ(c->isnow + 1), FMUL(qrain, dt));
            SNLIQ(c->isnow + 1) = f_max(SN_K_ZERO, SNLIQ(c->isnow + 1));
        }
    }

    for (int j = c->isnow + 1; j <= 0; ++j) {
        int k = j + NSNOW - 1;
        vol_ice[k] = f_min(SN_K_ONE, FDIV(SNICE(j), FMUL(DZSNSO(j), SN_K_DENICE)));
        epore[k] = FSUB(SN_K_ONE, vol_ice[k]);
    }

    float qin = SN_K_ZERO;
    float qout = SN_K_ZERO;

    for (int j = c->isnow + 1; j <= 0; ++j) {
        int k = j + NSNOW - 1;
        SNLIQ(j) = FADD(SNLIQ(j), qin);
        vol_liq[k] = FDIV(SNLIQ(j), FMUL(DZSNSO(j), SN_K_DENH2O));
        qout = f_max(SN_K_ZERO, FMUL(FSUB(vol_liq[k], FMUL(ssi, epore[k])), DZSNSO(j)));
        if (j == 0) {
            qout = f_max(FMUL(FSUB(vol_liq[k], epore[k]), DZSNSO(j)),
                         FMUL(FMUL(snow_ret_fac, dt), qout));
        }
        qout = FMUL(qout, SN_K_DENH2O);
        SNLIQ(j) = FSUB(SNLIQ(j), qout);
        if (FDIV(SNLIQ(j), FADD(SNICE(j), SNLIQ(j))) > SN_K_MAXLIQ) {
            qout = FADD(qout, FSUB(SNLIQ(j),
                        FMUL(FDIV(SN_K_MAXLIQ, FSUB(SN_K_ONE, SN_K_MAXLIQ)), SNICE(j))));
            SNLIQ(j) = FMUL(FDIV(SN_K_MAXLIQ, FSUB(SN_K_ONE, SN_K_MAXLIQ)), SNICE(j));
        }
        qin = qout;
    }

    for (int j = c->isnow + 1; j <= 0; ++j) {
        DZSNSO(j) = f_max(DZSNSO(j), FADD(FDIV(SNLIQ(j), SN_K_DENH2O),
                                          FDIV(SNICE(j), SN_K_DENICE)));
    }

    return FDIV(qout, dt);   // QSNBOT, mm/s
}

// ==========================================================================
// SNOWWATER -- lines 6398-6535
// ==========================================================================
__device__ void snowwater(Col *c, float dt, const float *zsoil,
                          const int *imelt, const float *ficeold,
                          float sfctmp, float snowhin, float qsnow,
                          float qsnfro, float qsnsub, float qrain,
                          float ssi, float snow_ret_fac,
                          float *qsnbot, float *snoflow,
                          float *ponding1, float *ponding2)
{
    *snoflow = SN_K_ZERO;
    *ponding1 = SN_K_ZERO;
    *ponding2 = SN_K_ZERO;

    snowfall(c, dt, qsnow, snowhin, sfctmp);

    // MB: do each if block separately
    if (c->isnow < 0) compact(c, dt, imelt, ficeold);
    if (c->isnow < 0) combine(c, ponding1, ponding2);
    if (c->isnow < 0) divide(c);

    *qsnbot = snowh2o(c, dt, qsnfro, qsnsub, qrain, ssi, snow_ret_fac,
                      ponding1, ponding2);

    // set empty snow layers to zero
    for (int iz = -NSNOW + 1; iz <= c->isnow; ++iz) {
        SNICE(iz) = SN_K_ZERO;
        SNLIQ(iz) = SN_K_ZERO;
        STC(iz) = SN_K_ZERO;
        DZSNSO(iz) = SN_K_ZERO;
        ZSNSO(iz) = SN_K_ZERO;
    }

    // equilibrium state of snow in the glacier region
    if (c->sneqv > SN_K_5000) {
        float bdsnow = FDIV(SNICE(0), DZSNSO(0));
        *snoflow = FSUB(c->sneqv, SN_K_5000);
        SNICE(0) = FSUB(SNICE(0), *snoflow);
        DZSNSO(0) = FSUB(DZSNSO(0), FDIV(*snoflow, bdsnow));
        *snoflow = FDIV(*snoflow, dt);
    }

    // sum up snow mass for layered snow
    if (c->isnow < 0) {
        c->sneqv = SN_K_ZERO;
        for (int iz = c->isnow + 1; iz <= 0; ++iz)
            c->sneqv = FADD(FADD(c->sneqv, SNICE(iz)), SNLIQ(iz));
    }

    // Reset ZSNSO and layer thickness DZSNSO
    for (int iz = c->isnow + 1; iz <= 0; ++iz) DZSNSO(iz) = -DZSNSO(iz);

    DZSNSO(1) = zsoil[0];
    for (int iz = 2; iz <= NSOIL; ++iz) DZSNSO(iz) = FSUB(zsoil[iz - 1], zsoil[iz - 2]);

    ZSNSO(c->isnow + 1) = DZSNSO(c->isnow + 1);
    for (int iz = c->isnow + 2; iz <= NSOIL; ++iz)
        ZSNSO(iz) = FADD(ZSNSO(iz - 1), DZSNSO(iz));

    for (int iz = c->isnow + 1; iz <= NSOIL; ++iz) DZSNSO(iz) = -DZSNSO(iz);

    // C.He: update SNOWH for multi-layer snow
    if (c->isnow < 0) {
        c->snowh = SN_K_ZERO;
        for (int iz = c->isnow + 1; iz <= 0; ++iz)
            c->snowh = FADD(c->snowh, DZSNSO(iz));
    }
}
// <<< END imported section: noahmp_snow.cu

// ==========================================================================
// WATER's own constants and host layout.  Everything above this line is an
// imported section and is checked against its source file by
// tests/test_noahmp_water_cuda.py.
// ==========================================================================

__constant__ unsigned int C_W_F32[3] = {
    0x447A0000u, /* 1000.0  -- 6146 and 6212                              */
    0x3A83126Fu, /* 0.001   -- 6159, 6162, 6164, 6167, 6170               */
    0x459C4000u, /* 5000.0  -- WSLMAX, the PARAMETER at 6098              */
};
#define W_K_1000   __uint_as_float(C_W_F32[0])
#define W_K_MILLI  __uint_as_float(C_W_F32[1])
#define W_K_WSLMAX __uint_as_float(C_W_F32[2])

// par: the 24 soilwater slots, then SNOWH2O's two.
#define P_SSI      24
#define P_SRF      25
#define W_P_STRIDE 26

// iin: [0] isnow  [1] urban  [2] nroot  [3] ist
//      [4] frozen_canopy  [5] frozen_ground  [6..8] imelt(-2..0)
#define W_INT_STRIDE 9

// fin: [0..NSTATE-1] the snow/soil column, then
#define W_DT          (NSTATE +  0)
#define W_FCEV        (NSTATE +  1)
#define W_FCTR        (NSTATE +  2)
#define W_ELAI        (NSTATE +  3)
#define W_ESAI        (NSTATE +  4)
#define W_FVEG        (NSTATE +  5)
#define W_BDFALL      (NSTATE +  6)
#define W_SFCTMP      (NSTATE +  7)
#define W_QVAP        (NSTATE +  8)
#define W_QDEW        (NSTATE +  9)
#define W_QSNOW       (NSTATE + 10)
#define W_QRAIN       (NSTATE + 11)
#define W_SNOWHIN     (NSTATE + 12)
#define W_PONDING     (NSTATE + 13)
#define W_CANLIQ      (NSTATE + 14)
#define W_CANICE      (NSTATE + 15)
#define W_TV          (NSTATE + 16)
#define W_WSLAKE      (NSTATE + 17)
#define W_ACC_QINSUR  (NSTATE + 18)
#define W_ACC_QSEVA   (NSTATE + 19)
#define W_ZSOIL       (NSTATE + 20)  /* 4 */
#define W_BTRANI      (NSTATE + 24)  /* 4 */
#define W_ACC_ETRANI  (NSTATE + 28)  /* 4 */
#define W_SMC         (NSTATE + 32)  /* 4 */
#define W_FICEOLD     (NSTATE + 36)  /* 3 */
#define W_IN_STRIDE   (NSTATE + 39)

// fout: [0..NSTATE-1] the column, then
#define O_SMC         (NSTATE +  0)  /* 4 */
#define O_CANLIQ      (NSTATE +  4)
#define O_CANICE      (NSTATE +  5)
#define O_TV          (NSTATE +  6)
#define O_WSLAKE      (NSTATE +  7)
#define O_ACC_QINSUR  (NSTATE +  8)
#define O_ACC_QSEVA   (NSTATE +  9)
#define O_ACC_ETRANI  (NSTATE + 10)  /* 4 */
#define O_CMC         (NSTATE + 14)
#define O_ECAN        (NSTATE + 15)
#define O_ETRAN       (NSTATE + 16)
#define O_FWET        (NSTATE + 17)
#define O_RUNSRF      (NSTATE + 18)
#define O_RUNSUB      (NSTATE + 19)
#define O_QTLDRN      (NSTATE + 20)
#define O_PONDING1    (NSTATE + 21)
#define O_PONDING2    (NSTATE + 22)
#define O_QSNBOT      (NSTATE + 23)
#define O_QSNSUB      (NSTATE + 24)
#define O_QSNFRO      (NSTATE + 25)
#define O_QSUBC       (NSTATE + 26)
#define O_QFROC       (NSTATE + 27)
#define O_QFRZC       (NSTATE + 28)
#define O_QMELTC      (NSTATE + 29)
#define O_QEVAC       (NSTATE + 30)
#define O_QDEWC       (NSTATE + 31)
// The four locals the oracle's `probe` stage pins.  They never reach a WRF
// output, but they decide a branch, so the device is held to them too.
#define O_QSEVA       (NSTATE + 32)
#define O_QSDEW       (NSTATE + 33)
#define O_QINSUR      (NSTATE + 34)
#define O_SNOFLOW     (NSTATE + 35)
#define O_SOILDRAIN   (NSTATE + 36)
#define W_OUT_STRIDE  (NSTATE + 37)

// ==========================================================================
// WATER -- module_sf_noahmplsm.F:5954-6261
// ==========================================================================
__device__ void d_water(const float *par, int urban, int nroot, int ist,
                        Col *c, float *smc,
                        const int *imelt, const float *ficeold,
                        const float *zsoil, const float *btrani,
                        float dt, float fcev, float fctr, float elai,
                        float esai, float fveg, float bdfall,
                        int frozen_canopy, int frozen_ground, float sfctmp,
                        float qvap, float qdew, float qsnow, float qrain,
                        float snowhin, float ponding,
                        float canliq, float canice, float tv, float wslake,
                        float acc_qinsur, float acc_qseva, float *acc_etrani,
                        float *o)
{
    float etrani[NSOIL];
    for (int k = 0; k < NSOIL; ++k) etrani[k] = K_ZERO;               // :6107
    float snoflow = K_ZERO;                                          // :6108
    float runsub  = K_ZERO;                                          // :6109
    float runsrf  = K_ZERO;                                          // :6110
    float qinsur  = K_ZERO;                                          // :6111
    float qtldrn  = K_ZERO;                                          // :6112
    // Genuine bottom drainage, separate from exfiltration and snow runoff.
    // Diagnostic only; existing WATER calculations and outputs are unchanged.
    float soldrain = K_ZERO;

    // canopy-intercepted snowfall/rainfall, drips, and throughfall    :6116
    float cw[13];
    d_canwater(par[P_CH2OP], dt, fcev, fctr, elai, esai, fveg, bdfall,
               frozen_canopy, canliq, canice, tv, cw);
    canliq = cw[0]; canice = cw[1]; tv = cw[2];
    float cmc = cw[3], ecan = cw[4], etran = cw[5], fwet = cw[6];
    float qsubc = cw[7], qfroc = cw[8], qfrzc = cw[9];
    float qmeltc = cw[10], qevac = cw[11], qdewc = cw[12];

    // sublimation, frost, evaporation, and dew
    float qsnsub = K_ZERO;                                           // :6126
    if (c->sneqv > K_ZERO)                                           // :6127
        qsnsub = f_min(qvap, FDIV(c->sneqv, dt));                    // :6128
    float qseva = FSUB(qvap, qsnsub);                                // :6130

    float qsnfro = K_ZERO;                                           // :6132
    if (c->sneqv > K_ZERO)                                           // :6133
        qsnfro = qdew;                                               // :6134
    float qsdew = FSUB(qdew, qsnfro);                                // :6136

    float qsnbot, ponding1, ponding2;
    snowwater(c, dt, zsoil, imelt, ficeold, sfctmp, snowhin, qsnow,
              qsnfro, qsnsub, qrain, par[P_SSI], par[P_SRF],
              &qsnbot, &snoflow, &ponding1, &ponding2);              // :6138

    // SNOWWATER restores DZSNSO(1:NSOIL) to a positive thickness before it
    // returns, which is what 6146 indexes.
    float *dz = c->dzsnso + NSNOW;

    if (frozen_ground) {                                             // :6145
        c->sice[0] = FADD(c->sice[0],
                          FDIV(FMUL(FSUB(qsdew, qseva), dt),
                               FMUL(dz[0], W_K_1000)));              // :6146
        qsdew = K_ZERO;                                              // :6147
        qseva = K_ZERO;                                              // :6148
        if (c->sice[0] < K_ZERO) {                                   // :6149
            c->sh2o[0] = FADD(c->sh2o[0], c->sice[0]);               // :6150
            c->sice[0] = K_ZERO;                                     // :6151
        }
        smc[0] = FADD(c->sh2o[0], c->sice[0]);                       // :6153
    }

    // convert units (mm/s -> m/s)
    qinsur = FMUL(FDIV(FADD(FADD(ponding, ponding1), ponding2), dt),
                  W_K_MILLI);                                        // :6159

    if (c->isnow == 0)                                               // :6161
        qinsur = FADD(qinsur,
                      FMUL(FADD(FADD(qsnbot, qsdew), qrain),
                           W_K_MILLI));                              // :6162
    else                                                             // :6163
        qinsur = FADD(qinsur,
                      FMUL(FADD(qsnbot, qsdew), W_K_MILLI));         // :6164

    qseva = FMUL(qseva, W_K_MILLI);                                  // :6167

    for (int iz = 0; iz < nroot; ++iz)                               // :6169
        etrani[iz] = FMUL(FMUL(etran, btrani[iz]), W_K_MILLI);       // :6170

    // added soil timestep capability
    acc_qinsur = FADD(acc_qinsur, qinsur);                           // :6178
    acc_qseva  = FADD(acc_qseva, qseva);                             // :6179
    float acc_e[NSOIL];
    for (int k = 0; k < NSOIL; ++k)
        acc_e[k] = FADD(acc_etrani[k], etrani[k]);                   // :6180

    // `if (calculate_soil)` at 6183 is always true under soiltstep=0.0, and
    // soil_update_steps == 1 makes DT_soil == DT and the three divisions at
    // 6204-6206 the identity.
    float dt_soil = dt;                                              // :6185
    float qseva_avg = acc_qseva;                                     // :6204
    float qinsur_avg = acc_qinsur;                                   // :6205
    const float *etrani_avg = acc_e;                                 // :6206

    if (ist == 2) {                                                  // :6209
        runsrf = K_ZERO;                                             // :6210
        if (wslake >= W_K_WSLMAX)                                    // :6211
            runsrf = FMUL(FMUL(qinsur_avg, W_K_1000), dt_soil);
        wslake = FSUB(FADD(wslake,
                           FMUL(FMUL(FSUB(qinsur_avg, qseva_avg), W_K_1000),
                                dt_soil)),
                      runsrf);                                       // :6212
        // QDRAIN, WCND and FCRMAX stay undefined on this branch and nothing
        // reads them: 6235 and 6252-6254 are inside the ELSE.
    } else {                                                         // :6213
        float qdrain, wcnd[NSOIL], fcrmax;
        d_soilwater(par, urban, dt_soil, zsoil, dz, qinsur_avg, qseva_avg,
                    etrani_avg, c->sice, c->sh2o, smc, &runsub,
                    &runsrf, &qdrain, wcnd, &fcrmax);                // :6214
        soldrain = FMUL(qdrain, dt_soil);  // qdrain mm/s -> mm per LSM step

        // OPT_RUN==1 GROUNDWATER at 6225-6231 is dead.
        runsub = FADD(runsub, qdrain);                               // :6235

        for (int iz = 0; iz < NSOIL; ++iz)                           // :6238
            smc[iz] = FADD(c->sh2o[iz], c->sice[iz]);                // :6239

        // OPT_RUN==5 SHALLOWWATERTABLE at 6242-6250 is dead.

        runsrf = FMUL(runsrf, dt_soil);                              // :6252
        runsub = FMUL(runsub, dt_soil);                              // :6253
        qtldrn = FMUL(qtldrn, dt_soil);                              // :6254
    }

    runsub = FADD(runsub, FMUL(snoflow, dt));                        // :6259

    for (int k = 0; k < NSOIL; ++k) {
        o[O_SMC + k] = smc[k];
        o[O_ACC_ETRANI + k] = acc_e[k];
    }
    o[O_CANLIQ] = canliq; o[O_CANICE] = canice; o[O_TV] = tv;
    o[O_WSLAKE] = wslake;
    o[O_ACC_QINSUR] = acc_qinsur; o[O_ACC_QSEVA] = acc_qseva;
    o[O_CMC] = cmc; o[O_ECAN] = ecan; o[O_ETRAN] = etran; o[O_FWET] = fwet;
    o[O_RUNSRF] = runsrf; o[O_RUNSUB] = runsub; o[O_QTLDRN] = qtldrn;
    o[O_PONDING1] = ponding1; o[O_PONDING2] = ponding2;
    o[O_QSNBOT] = qsnbot; o[O_QSNSUB] = qsnsub; o[O_QSNFRO] = qsnfro;
    o[O_QSUBC] = qsubc; o[O_QFROC] = qfroc; o[O_QFRZC] = qfrzc;
    o[O_QMELTC] = qmeltc; o[O_QEVAC] = qevac; o[O_QDEWC] = qdewc;
    o[O_QSEVA] = qseva; o[O_QSDEW] = qsdew; o[O_QINSUR] = qinsur;
    o[O_SNOFLOW] = snoflow;
    o[O_SOILDRAIN] = soldrain;
}

// ==========================================================================
// Entry points.  One thread per fixture case.
// ==========================================================================

extern "C" __global__ void k_water(const float *par, const float *fin,
                                   const int *iin, float *fout, int *iout,
                                   int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const float *p = par + (size_t) tid * W_P_STRIDE;
    const float *x = fin + (size_t) tid * W_IN_STRIDE;
    const int   *ip = iin + (size_t) tid * W_INT_STRIDE;
    float *q = fout + (size_t) tid * W_OUT_STRIDE;

    Col col;
    col_load(&col, x, ip[0]);

    float smc[NSOIL], acc_etrani[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        smc[k] = x[W_SMC + k];
        acc_etrani[k] = x[W_ACC_ETRANI + k];
    }

    d_water(p, ip[1], ip[2], ip[3], &col, smc, ip + 6, x + W_FICEOLD,
            x + W_ZSOIL, x + W_BTRANI,
            x[W_DT], x[W_FCEV], x[W_FCTR], x[W_ELAI], x[W_ESAI], x[W_FVEG],
            x[W_BDFALL], ip[4], ip[5], x[W_SFCTMP], x[W_QVAP], x[W_QDEW],
            x[W_QSNOW], x[W_QRAIN], x[W_SNOWHIN], x[W_PONDING],
            x[W_CANLIQ], x[W_CANICE], x[W_TV], x[W_WSLAKE],
            x[W_ACC_QINSUR], x[W_ACC_QSEVA], acc_etrani, q);

    col_store(&col, q, iout + tid);
}

// Standalone probes for the two glibc transcriptions the assembly is built
// on, so a mis-folded __constant__ entry cannot hide inside a leaf that never
// selects it.
extern "C" __global__ void k_water_expf(const float *x, float *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    y[tid] = glibc_expf(x[tid]);
}

extern "C" __global__ void k_water_powf(const float *x, const float *e,
                                        float *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    y[tid] = glibc_powf(x[tid], e[tid]);
}
