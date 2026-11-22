// gpuwm/core/kernels/noahmp_bareflux.cu
//
// CUDA half of the Noah-MP BARE_FLUX port.  Bitwise-equal to
// gpuwm/core/noahmp_bareflux.py and to the WRF v4.6.1 oracle
// (tree d66e442fccc04111067e29274c9f9eaccc3cef28,
//  sha256(phys/module_sf_noahmplsm.F) =
//  bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282),
// option identity opt_sfc=1 opt_stc=1, NITERB=5.
//
// Covers BARE_FLUX (4174-4479) and the two module procedures it calls on the
// live path: SFCDIF1 (4583-4743) and ESAT (4952-5001).
//
// Three rules make bitwise agreement possible and none of them is optional:
//
// 1. Every float32 and float64 operation uses an explicit rounding intrinsic
//    (__fadd_rn/__fsub_rn/__fmul_rn/__fdiv_rn/__fsqrt_rn, __dadd_rn/__dsub_rn/
//    __dmul_rn/__fma_rn).  That pins the hardware rounding mode AND makes
//    nvcc's contraction pass a no-op, so -fmad=true cannot fuse a site the
//    Fortran did not fuse.
// 2. Every constant lives in __constant__ memory as a bit pattern.  ptxas
//    12.8's constant folder does not honour round-to-nearest-even on literal
//    arrays, so a table of FP32 literals can have its differences mis-folded
//    at compile time.  __fsub_rn pins the hardware, not the folder.  The
//    ESAT polynomial coefficients are exactly such a table, which is why
//    C_ESAT lives in __constant__ memory rather than as local literals.
// 3. LOG/ATAN/**0.25 are glibc calls in the oracle, so glibc_logf,
//    glibc_atanf and glibc_powf below transcribe glibc 2.39's own
//    algorithms.  CUDA's logf, atanf, powf, __logf and __powf are all
//    different functions and none of them can hold a max_ulp-0 gate.
//    SQRT is the one exception: it is correctly rounded in both, so
//    __fsqrt_rn matches gfortran's sqrtss.
//
// Dead under the pinned identity and deliberately absent: SFCDIF2 (opt_sfc=2)
// with its CH/CM rescale and snow clamp, and the opt_stc==3 snow-melt blend.
// The kernel refuses any other option pair rather than guessing.

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
__constant__ unsigned long long C_LOGF_TAB[32] = {
    0x3FF661EC79F8F3BEULL, 0xBFD57BF7808CAADEULL,
    0x3FF571ED4AAF883DULL, 0xBFD2BEF0A7C06DDBULL,
    0x3FF49539F0F010B0ULL, 0xBFD01EAE7F513A67ULL,
    0x3FF3C995B0B80385ULL, 0xBFCB31D8A68224E9ULL,
    0x3FF30D190C8864A5ULL, 0xBFC6574F0AC07758ULL,
    0x3FF25E227B0B8EA0ULL, 0xBFC1AA2BC79C8100ULL,
    0x3FF1BB4A4A1A343FULL, 0xBFBA4E76CE8C0E5EULL,
    0x3FF12358F08AE5BAULL, 0xBFB1973C5A611CCCULL,
    0x3FF0953F419900A7ULL, 0xBFA252F438E10C1EULL,
    0x3FF0000000000000ULL, 0x0000000000000000ULL,
    0x3FEE608CFD9A47ACULL, 0x3FAAA5AA5DF25984ULL,
    0x3FECA4B31F026AA0ULL, 0x3FBC5E53AA362EB4ULL,
    0x3FEB2036576AFCE6ULL, 0x3FC526E57720DB08ULL,
    0x3FE9C2D163A1AA2DULL, 0x3FCBC2860D224770ULL,
    0x3FE886E6037841EDULL, 0x3FD1058BC8A07EE1ULL,
    0x3FE767DCF5534862ULL, 0x3FD4043057B6EE09ULL,
};
__constant__ unsigned long long C_LOGF_MISC[4] = {
    0x3FE62E42FEFA39EFULL, 0xBFD00EA348B88334ULL, 0x3FD5575B0BE00B6AULL, 0xBFDFFFFEF20A4123ULL,
};
// powf overflow/underflow limits, as glibc writes them (hex float literals).
// [2] is 0x1.fffffffd1d571p+6; [3] is 126.0, the |y*log2(x)| screen.
__constant__ unsigned long long C_LIMITS[4] = {
    0x40562E42E0000000ULL, 0xC059FE3680000000ULL, 0x405FFFFFFFD1D571ULL, 0x405F800000000000ULL,
};
// --------------------------------------------------------------------------
// float32 constants, as bit patterns (rule 2 above).
// --------------------------------------------------------------------------
__constant__ unsigned int C_F32[28] = {
    0x00000000u, 0x3F800000u, 0x40000000u, 0x3F000000u,
    0x40800000u, 0x42C80000u, 0x4388947Bu, 0x3373864Fu,
    0x3ECCCCCDu, 0x411CE608u, 0x447B28F6u, 0x358637BDu,
    0x42480000u, 0xC2480000u, 0x3F1C28F6u, 0x3F1F3B64u,
    0x3EC18937u, 0x41800000u, 0x3E800000u, 0x3F666666u,
    0xC0A00000u, 0xBF800000u, 0x3FC90FDAu, 0x3D4CCCCDu,
    0x3727C5ACu, 0x4B000000u, 0x3DCCCCCDu, 0x3FC00000u,
};
#define K_ZERO     __uint_as_float(C_F32[0])
#define K_ONE      __uint_as_float(C_F32[1])
#define K_TWO      __uint_as_float(C_F32[2])
#define K_HALF     __uint_as_float(C_F32[3])
#define K_FOUR     __uint_as_float(C_F32[4])
#define K_HUND     __uint_as_float(C_F32[5])
#define K_TFRZ     __uint_as_float(C_F32[6])
#define K_SB       __uint_as_float(C_F32[7])
#define K_VKC      __uint_as_float(C_F32[8])
#define K_GRAV     __uint_as_float(C_F32[9])
#define K_CPAIR    __uint_as_float(C_F32[10])
#define K_MPE      __uint_as_float(C_F32[11])
#define K_P50      __uint_as_float(C_F32[12])
#define K_N50      __uint_as_float(C_F32[13])
#define K_P61      __uint_as_float(C_F32[14])
#define K_P622     __uint_as_float(C_F32[15])
#define K_P378     __uint_as_float(C_F32[16])
#define K_SIXTEEN  __uint_as_float(C_F32[17])
#define K_QUARTER  __uint_as_float(C_F32[18])
#define K_P9       __uint_as_float(C_F32[19])
#define K_N5       __uint_as_float(C_F32[20])
#define K_N1       __uint_as_float(C_F32[21])
#define K_PIO2     __uint_as_float(C_F32[22])
#define K_P05      __uint_as_float(C_F32[23])
#define K_E5       __uint_as_float(C_F32[24])
#define K_P2P23    __uint_as_float(C_F32[25])
#define K_P1       __uint_as_float(C_F32[26])
#define K_1P5      __uint_as_float(C_F32[27])

// ESAT polynomial coefficients A0..A6, B0..B6, C0..C6, D0..D6.  This is
// precisely the "literal array of FP32 constants" ptxas 12.8 is known to
// mis-fold, so it lives in __constant__ memory rather than in the function.
__constant__ unsigned int C_ESAT[28] = {
    0x40C37319u, 0x3EE32656u, 0x3C6A1E55u, 0x398AF867u, 0x364B6C50u, 0x32AEB9EAu, 0x2E86F33Bu,
    0x40C37E63u, 0x3F00E367u, 0x3C9A8091u, 0x39DAF453u, 0x36C371F7u, 0x334FD334u, 0x2F4A2E60u,
    0x3EE33B10u, 0x3CEA0BB0u, 0x3A501761u, 0x374BE117u, 0x33DE9991u, 0x2FC2326Bu, 0xAB479299u,
    0x3F00C69Cu, 0x3D1A8D72u, 0x3AA632DDu, 0x37CFD543u, 0x34A15DEFu, 0x3114557Bu, 0x2CFAE738u,
};

// glibc s_atanf.c atanhi[4], atanlo[4], aT[11].  aT[0] is 0x3EAAAAAB -- the
// value glibc's decimal literal rounds to, NOT the 0x3eaaaaaa its own source
// comment claims.  Trusting the comment costs 1 ULP on the |x|<0.4375 arm.
__constant__ unsigned int C_ATAN[19] = {
    0x3EED6338u, 0x3F490FDAu, 0x3F7B985Eu, 0x3FC90FDAu,
    0x31AC3769u, 0x33222168u, 0x33140FB4u, 0x33A22168u,
    0x3EAAAAABu, 0xBE4CCCCDu, 0x3E124925u, 0xBDE38E38u,
    0x3DBA2E6Eu, 0xBD9D8795u, 0x3D886B35u, 0xBD6EF16Bu,
    0x3D4BDA59u, 0xBD15A221u, 0x3C8569D7u,
};

// --------------------------------------------------------------------------
// rounding-pinned primitives
// --------------------------------------------------------------------------
#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))
#define FSQRT(a)   __fsqrt_rn(a)

#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define DFMA(a, b, c) __fma_rn((a), (b), (c))

// gfortran lowers REAL MIN/MAX to minss/maxss, which return the SECOND
// operand whenever the first does not compare strictly less/greater.  The
// operands are only ever equal when they are the same value (or +-0.0, whose
// sign is erased by the subtraction that consumes the result), so this is a
// faithful mirror of gpuwm/core/noahmp_bareflux.py's _fmin/_fmax rather than
// a load-bearing difference.
__device__ __forceinline__ float f_min(float a, float b) { return (a < b) ? a : b; }
__device__ __forceinline__ float f_max(float a, float b) { return (a > b) ? a : b; }

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
// glibc 2.39 logf  --  sysdeps/ieee754/flt-32/e_logf.c, FMA variant.
// Operation order read off the disassembly of __logf_fma, including the
// single FMA that computes logc + k*Ln2.
// --------------------------------------------------------------------------
__device__ float glibc_logf(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3F800000u) return 0.0f;           // log(1) == +0

    if ((ix - 0x00800000u) >= (0x7F800000u - 0x00800000u)) {
        if ((ix * 2u) == 0u) return __uint_as_float(0xFF800000u);   // log(0)
        if (ix == 0x7F800000u) return __uint_as_float(0x7F800000u); // log(inf)
        if ((ix & 0x80000000u) || (ix * 2u) >= 0xFF000000u)
            return __uint_as_float(0x7FC00000u);  // negative or NaN -> NaN
        ix = __float_as_uint(FMUL(x, K_P2P23));   // subnormal: * 0x1p23
        ix -= (23u << 23);
    }

    unsigned int tmp = ix - 0x3F330000u;
    unsigned int i = (tmp >> 19) & 15u;
    int k = ((int) tmp) >> 23;                    // arithmetic shift
    unsigned int iz = ix - (tmp & 0xFF800000u);
    double invc = __longlong_as_double(C_LOGF_TAB[2 * i]);
    double logc = __longlong_as_double(C_LOGF_TAB[2 * i + 1]);
    double z = (double) __uint_as_float(iz);

    double ln2 = __longlong_as_double(C_LOGF_MISC[0]);
    double a0 = __longlong_as_double(C_LOGF_MISC[1]);
    double a1 = __longlong_as_double(C_LOGF_MISC[2]);
    double a2 = __longlong_as_double(C_LOGF_MISC[3]);

    double r = DFMA(z, invc, -1.0);
    double y0 = DFMA((double) k, ln2, logc);
    double r2 = DMUL(r, r);
    double y = DFMA(a1, r, a2);
    y = DFMA(r2, a0, y);
    y = DFMA(r2, y, DADD(y0, r));
    return __double2float_rn(y);
}

// --------------------------------------------------------------------------
// glibc 2.39 powf  --  sysdeps/ieee754/flt-32/e_powf.c, FMA variant.
// Only the domain the radiation leaves reach is transcribed: a strictly
// positive normal base and a finite exponent (GROUNDALB's
// MAX(0.01, COSZ) ** 1.7).  Anything else returns NaN rather than a value
// this kernel cannot vouch for.
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
    if ((ix - 0x00800000u) >= 0x7F000000u || (((2u * iy) - 1u) >= 0xFEFFFFFFu))
        return __uint_as_float(0x7FC00000u);   // outside the ported domain

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
// glibc 2.39 atanf  --  sysdeps/ieee754/flt-32/s_atanf.c (fdlibm kernel).
// Unlike logf/powf, atanf is NOT an ifunc in libm.so.6, so there is no -mfma
// rebuild of it: every operation is a plain float32 multiply or add and none
// of them may be contracted.  Hence FMUL/FADD/FSUB throughout, never FMA.
// --------------------------------------------------------------------------
__device__ float glibc_atanf(float x)
{
    unsigned int hx = __float_as_uint(x);
    unsigned int ix = hx & 0x7FFFFFFFu;
    int signed_hx = (int) hx;
    int id;

    if (ix >= 0x4C000000u) {                 // |x| >= 2^25
        if (ix > 0x7F800000u) return FADD(x, x);              // NaN
        if (signed_hx > 0)
            return FADD(__uint_as_float(C_ATAN[3]), __uint_as_float(C_ATAN[7]));
        return FSUB(FSUB(K_ZERO, __uint_as_float(C_ATAN[3])),
                    __uint_as_float(C_ATAN[7]));
    }

    if (ix < 0x3EE00000u) {                  // |x| < 0.4375
        if (ix < 0x31000000u) return x;      // |x| < 2^-29: atan(x) == x
        id = -1;
    } else {
        x = __uint_as_float(ix);             // fabsf(x)
        if (ix < 0x3F980000u) {              // |x| < 1.1875
            if (ix < 0x3F300000u) {          // 7/16 <= |x| < 11/16
                id = 0;
                x = FDIV(FSUB(FMUL(K_TWO, x), K_ONE), FADD(K_TWO, x));
            } else {                         // 11/16 <= |x| < 19/16
                id = 1;
                x = FDIV(FSUB(x, K_ONE), FADD(x, K_ONE));
            }
        } else {
            if (ix < 0x401C0000u) {          // |x| < 2.4375
                id = 2;
                x = FDIV(FSUB(x, K_1P5), FADD(K_ONE, FMUL(K_1P5, x)));
            } else {                         // 2.4375 <= |x| < 2^25
                id = 3;
                x = FDIV(K_N1, x);
            }
        }
    }

    float z = FMUL(x, x);
    float w = FMUL(z, z);
#define AT(i) __uint_as_float(C_ATAN[8 + (i)])
    float s1 = FMUL(w, AT(10));
    s1 = FMUL(w, FADD(AT(8), s1));
    s1 = FMUL(w, FADD(AT(6), s1));
    s1 = FMUL(w, FADD(AT(4), s1));
    s1 = FMUL(w, FADD(AT(2), s1));
    s1 = FMUL(z, FADD(AT(0), s1));
    float s2 = FMUL(w, AT(9));
    s2 = FMUL(w, FADD(AT(7), s2));
    s2 = FMUL(w, FADD(AT(5), s2));
    s2 = FMUL(w, FADD(AT(3), s2));
    s2 = FMUL(w, FADD(AT(1), s2));
#undef AT
    float s = FADD(s1, s2);
    if (id < 0) return FSUB(x, FMUL(x, s));
    float r = FSUB(__uint_as_float(C_ATAN[id]),
                   FSUB(FSUB(FMUL(x, s), __uint_as_float(C_ATAN[4 + id])), x));
    return (signed_hx < 0) ? FSUB(K_ZERO, r) : r;
}

// --------------------------------------------------------------------------
// gfortran expands REAL**3 and REAL**4 (integer constant exponents) into
// __builtin_powi, which multiplies rather than calling powf.
// --------------------------------------------------------------------------
__device__ __forceinline__ float powi3(float x)
{
    float x2 = FMUL(x, x);
    return FMUL(x2, x);
}

__device__ __forceinline__ float powi4(float x)
{
    float x2 = FMUL(x, x);
    return FMUL(x2, x2);
}

// --------------------------------------------------------------------------
// ESAT -- module_sf_noahmplsm.F:4952-5001
// base 0 -> ESW (water), 7 -> ESI (ice), 14 -> DESW, 21 -> DESI
// --------------------------------------------------------------------------
__device__ __forceinline__ float esat_poly(int base, float t)
{
#define C(i) __uint_as_float(C_ESAT[base + (i)])
    float y = FADD(C(5), FMUL(t, C(6)));
    y = FADD(C(4), FMUL(t, y));
    y = FADD(C(3), FMUL(t, y));
    y = FADD(C(2), FMUL(t, y));
    y = FADD(C(1), FMUL(t, y));
    y = FADD(C(0), FMUL(t, y));
#undef C
    return FMUL(K_HUND, y);
}

// --------------------------------------------------------------------------
// SFCDIF1 -- module_sf_noahmplsm.F:4583-4743, OPT_SFC == 1.
// The struct is the INOUT block BARE_FLUX threads through its five
// stability iterations.
// --------------------------------------------------------------------------
struct Sfcdif1State {
    float moz, fm, fh, fm2, fh2, fv, cm, ch, ch2;
    int mozsgn;
};

__device__ void d_sfcdif1(Sfcdif1State *s, int it, float sfctmp, float rhoair,
                          float h, float qair, float zlvl, float zpd,
                          float z0m, float z0h, float ur, float mpe)
{
    float mozold = s->moz;

    float tmpcm = glibc_logf(FDIV(FSUB(zlvl, zpd), z0m));
    float tmpch = glibc_logf(FDIV(FSUB(zlvl, zpd), z0h));
    float tmpcm2 = glibc_logf(FDIV(FADD(K_TWO, z0m), z0m));
    float tmpch2 = glibc_logf(FDIV(FADD(K_TWO, z0h), z0h));

    float mol, moz2;
    if (it == 1) {
        s->fv = K_ZERO;
        s->moz = K_ZERO;
        mol = K_ZERO;
        moz2 = K_ZERO;
    } else {
        float tvir = FMUL(FADD(K_ONE, FMUL(K_P61, qair)), sfctmp);
        float tmp1 = FDIV(FMUL(FMUL(K_VKC, FDIV(K_GRAV, tvir)), h),
                          FMUL(rhoair, K_CPAIR));
        if (fabsf(tmp1) <= mpe) tmp1 = mpe;
        mol = FDIV(FMUL(K_N1, powi3(s->fv)), tmp1);
        s->moz = f_min(FDIV(FSUB(zlvl, zpd), mol), K_ONE);
        moz2 = f_min(FDIV(FADD(K_TWO, z0h), mol), K_ONE);
    }

    if (FMUL(mozold, s->moz) < K_ZERO) s->mozsgn += 1;
    if (s->mozsgn >= 2) {
        s->moz = K_ZERO;
        s->fm = K_ZERO;
        s->fh = K_ZERO;
        moz2 = K_ZERO;
        s->fm2 = K_ZERO;
        s->fh2 = K_ZERO;
    }

    float fmnew, fhnew, fm2new, fh2new;
    if (s->moz < K_ZERO) {
        float tmp1 = glibc_powf(FSUB(K_ONE, FMUL(K_SIXTEEN, s->moz)), K_QUARTER);
        float tmp2 = glibc_logf(FDIV(FADD(K_ONE, FMUL(tmp1, tmp1)), K_TWO));
        float tmp3 = glibc_logf(FDIV(FADD(K_ONE, tmp1), K_TWO));
        fmnew = FADD(FSUB(FADD(FMUL(K_TWO, tmp3), tmp2),
                          FMUL(K_TWO, glibc_atanf(tmp1))), K_PIO2);
        fhnew = FMUL(K_TWO, tmp2);

        float tmp12 = glibc_powf(FSUB(K_ONE, FMUL(K_SIXTEEN, moz2)), K_QUARTER);
        float tmp22 = glibc_logf(FDIV(FADD(K_ONE, FMUL(tmp12, tmp12)), K_TWO));
        float tmp32 = glibc_logf(FDIV(FADD(K_ONE, tmp12), K_TWO));
        fm2new = FADD(FSUB(FADD(FMUL(K_TWO, tmp32), tmp22),
                           FMUL(K_TWO, glibc_atanf(tmp12))), K_PIO2);
        fh2new = FMUL(K_TWO, tmp22);
    } else {
        fmnew = FMUL(K_N5, s->moz);
        fhnew = fmnew;
        fm2new = FMUL(K_N5, moz2);
        fh2new = fm2new;
    }

    if (it == 1) {
        s->fm = fmnew;
        s->fh = fhnew;
        s->fm2 = fm2new;
        s->fh2 = fh2new;
    } else {
        s->fm = FMUL(K_HALF, FADD(s->fm, fmnew));
        s->fh = FMUL(K_HALF, FADD(s->fh, fhnew));
        s->fm2 = FMUL(K_HALF, FADD(s->fm2, fm2new));
        s->fh2 = FMUL(K_HALF, FADD(s->fh2, fh2new));
    }

    s->fh = f_min(s->fh, FMUL(K_P9, tmpch));
    s->fm = f_min(s->fm, FMUL(K_P9, tmpcm));
    s->fh2 = f_min(s->fh2, FMUL(K_P9, tmpch2));
    s->fm2 = f_min(s->fm2, FMUL(K_P9, tmpcm2));

    float cmfm = FSUB(tmpcm, s->fm);
    float chfh = FSUB(tmpch, s->fh);
    float cm2fm2 = FSUB(tmpcm2, s->fm2);
    float ch2fh2 = FSUB(tmpch2, s->fh2);
    if (fabsf(cmfm) <= mpe) cmfm = mpe;
    if (fabsf(chfh) <= mpe) chfh = mpe;
    if (fabsf(cm2fm2) <= mpe) cm2fm2 = mpe;
    if (fabsf(ch2fh2) <= mpe) ch2fh2 = mpe;

    s->cm = FDIV(FMUL(K_VKC, K_VKC), FMUL(cmfm, cmfm));
    // WRF divides CH by CMFM*CHFH, not CHFH*CHFH.  Not a typo to tidy up.
    s->ch = FDIV(FMUL(K_VKC, K_VKC), FMUL(cmfm, chfh));

    s->fv = FMUL(ur, FSQRT(s->cm));
    s->ch2 = FDIV(FMUL(K_VKC, s->fv), ch2fh2);
    (void) mol;
    (void) cm2fm2;
}

// --------------------------------------------------------------------------
// BARE_FLUX -- module_sf_noahmplsm.F:4174-4479
//
// in[53] per column, in the order
//   dt sag lwdn ur uu vv sfctmp thair qair eair rhoair snowh zlvl zpd z0m
//   fsno emg rsurf lathea gamma rhsur q2 pahb dx dz8w qc psfc sfcprs
//   tgb cm ch qsfc  dzsnso(-2..4) stc(-2..4) df(-2..4)
// ii[5] per column: isnow ivgtyp iloc jloc iurban
// out[13] per column:
//   tgb cm ch qsfc tauxb tauyb irb shb evb ghb t2mb q2b ehb2
//
// dt, thair, q2, dx, dz8w, qc, sfcprs, fsno, the incoming cm/ch/qsfc,
// ivgtyp, iloc and jloc are accepted and never read: under opt_sfc=1 and
// opt_stc=1 no live statement references them.
// --------------------------------------------------------------------------
#define NITERB 5
#define NSNOW 3

__device__ void d_bare_flux(const float *p, const int *q, float *o)
{
    const float sag = p[1], lwdn = p[2], ur = p[3], uu = p[4], vv = p[5];
    const float sfctmp = p[6], qair = p[8], eair = p[9], rhoair = p[10];
    const float snowh = p[11], zlvl = p[12], zpd = p[13], z0m = p[14];
    const float emg = p[16], rsurf = p[17], lathea = p[18], gamma = p[19];
    const float rhsur = p[20], pahb = p[22], psfc = p[26];
    const float *dzsnso = p + 32, *stc = p + 39, *df = p + 46;
    const int isnow = q[0];
    const int urban_flag = q[4];

    // Fortran declares these arrays (-NSNOW+1:NSOIL), so element k lives at
    // index k + NSNOW - 1 of the packed row.
    const int kk = isnow + 1 + NSNOW - 1;

    float tgb = p[28];
    float h = K_ZERO;

    Sfcdif1State s;
    s.moz = K_ZERO;
    s.mozsgn = 0;
    s.fm = K_ZERO;
    s.fh = K_ZERO;
    s.fm2 = K_ZERO;
    s.fh2 = K_ZERO;
    s.fv = K_P1;
    s.cm = K_ZERO;
    s.ch = K_ZERO;
    s.ch2 = K_ZERO;

    const float cir = FMUL(emg, K_SB);
    const float cgh = FDIV(FMUL(K_TWO, df[kk]), dzsnso[kk]);

    float csh = K_ZERO, cev = K_ZERO, estg = K_ZERO, ehb = K_ZERO;
    float irb = K_ZERO, shb = K_ZERO, evb = K_ZERO, ghb = K_ZERO;
    float qsfc = p[31];
    float z0h = z0m;

    for (int it = 1; it <= NITERB; ++it) {
        z0h = z0m;                       // both arms of the ITER==1 test

        d_sfcdif1(&s, it, sfctmp, rhoair, h, qair, zlvl, zpd, z0m, z0h,
                  ur, K_MPE);

        float rahb = f_max(K_ONE, FDIV(K_ONE, FMUL(s.ch, ur)));
        float rawb = rahb;
        ehb = FDIV(K_ONE, rahb);

        float t = f_min(K_P50, f_max(K_N50, FSUB(tgb, K_TFRZ)));
        float destg;
        if (t > K_ZERO) {
            estg = esat_poly(0, t);
            destg = esat_poly(14, t);
        } else {
            estg = esat_poly(7, t);
            destg = esat_poly(21, t);
        }

        csh = FDIV(FMUL(rhoair, K_CPAIR), rahb);
        cev = FDIV(FDIV(FMUL(rhoair, K_CPAIR), gamma), FADD(rsurf, rawb));

        irb = FSUB(FMUL(cir, powi4(tgb)), FMUL(emg, lwdn));
        shb = FMUL(csh, FSUB(tgb, sfctmp));
        evb = FMUL(cev, FSUB(FMUL(estg, rhsur), eair));
        ghb = FMUL(cgh, FSUB(tgb, stc[kk]));

        float b = FADD(FSUB(FSUB(FSUB(FSUB(sag, irb), shb), evb), ghb), pahb);
        float cir4t3 = FMUL(FMUL(K_FOUR, cir), powi3(tgb));
        float a = FADD(FADD(FADD(cir4t3, csh), FMUL(cev, destg)), cgh);
        float dtg = FDIV(b, a);

        irb = FADD(irb, FMUL(cir4t3, dtg));
        shb = FADD(shb, FMUL(csh, dtg));
        evb = FADD(evb, FMUL(FMUL(cev, destg), dtg));
        ghb = FADD(ghb, FMUL(cgh, dtg));

        tgb = FADD(tgb, dtg);
        h = FMUL(csh, FSUB(tgb, sfctmp));

        t = f_min(K_P50, f_max(K_N50, FSUB(tgb, K_TFRZ)));
        estg = (t > K_ZERO) ? esat_poly(0, t) : esat_poly(7, t);
        float er = FMUL(estg, rhsur);
        qsfc = FDIV(FMUL(K_P622, er), FSUB(psfc, FMUL(K_P378, er)));
        // QFX is computed here in WRF and never used again; omitted.
    }

    // opt_stc == 1 snow reset.  The opt_stc == 3 blend is dead and absent.
    if (snowh > K_P05 && tgb > K_TFRZ) {
        tgb = K_TFRZ;
        // Correct-mode SN1 (gpuwm/correct_mode.py): ESTG and QSFC at the
        // reset ground temperature, as WRF's own glacier reset does
        // (:1092-1105); the land reset kept the discarded, warmer TG's.
#ifdef GPUWM_NOAHMP_MELT_RESET_ESAT
        {
            float t0 = f_min(K_P50, f_max(K_N50, FSUB(tgb, K_TFRZ)));
            estg = (t0 > K_ZERO) ? esat_poly(0, t0) : esat_poly(7, t0);
            float er0 = FMUL(estg, rhsur);
            qsfc = FDIV(FMUL(K_P622, er0), FSUB(psfc, FMUL(K_P378, er0)));
        }
#endif
        irb = FSUB(FMUL(cir, powi4(tgb)), FMUL(emg, lwdn));
        shb = FMUL(csh, FSUB(tgb, sfctmp));
        evb = FMUL(cev, FSUB(FMUL(estg, rhsur), eair));
        ghb = FSUB(FADD(sag, pahb), FADD(FADD(irb, shb), evb));
    }

    float tauxb = FSUB(K_ZERO, FMUL(FMUL(FMUL(rhoair, s.cm), ur), uu));
    float tauyb = FSUB(K_ZERO, FMUL(FMUL(FMUL(rhoair, s.cm), ur), vv));

    // opt_sfc == 1 2 m diagnostics.  WRF assigns EHB2 twice; the first
    // assignment is dead and is not reproduced.
    float ehb2 = FDIV(FMUL(s.fv, K_VKC),
                      FSUB(glibc_logf(FDIV(FADD(K_TWO, z0h), z0h)), s.fh2));
    float cq2b = ehb2;
    float t2mb, q2b;
    if (ehb2 < K_E5) {
        t2mb = tgb;
        q2b = qsfc;
    } else {
        t2mb = FSUB(tgb, FDIV(FMUL(FDIV(shb, FMUL(rhoair, K_CPAIR)), K_ONE),
                              ehb2));
        q2b = FSUB(qsfc, FMUL(FDIV(evb, FMUL(lathea, rhoair)),
                              FADD(FDIV(K_ONE, cq2b), rsurf)));
    }
    if (urban_flag) q2b = qsfc;

    o[0] = tgb;
    o[1] = s.cm;
    o[2] = ehb;          // "update CH": BARE_FLUX overwrites CH with EHB
    o[3] = qsfc;
    o[4] = tauxb;
    o[5] = tauyb;
    o[6] = irb;
    o[7] = shb;
    o[8] = evb;
    o[9] = ghb;
    o[10] = t2mb;
    o[11] = q2b;
    o[12] = ehb2;
}

extern "C" __global__ void noahmp_bare_flux(const float *in, const int *ii,
                                            float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_bare_flux(in + 53 * i, ii + 5 * i, out + 13 * i);
}

// Probe kernels.  They exist so the device acceptance test can show that the
// __constant__ tables survived ptxas and that the glibc kernels themselves --
// not merely the leaf built on top of them -- are bitwise right on the
// device.  Without them a table mis-fold could hide inside a leaf that
// happens not to reach the affected table entry.
extern "C" __global__ void noahmp_bareflux_libm_probe(const float *x,
                                                      float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[3 * i + 0] = glibc_logf(x[i]);
        out[3 * i + 1] = glibc_atanf(x[i]);
        out[3 * i + 2] = glibc_powf(x[i], K_QUARTER);
    }
}

extern "C" __global__ void noahmp_bareflux_esat_probe(const float *t,
                                                      float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[4 * i + 0] = esat_poly(0, t[i]);
        out[4 * i + 1] = esat_poly(7, t[i]);
        out[4 * i + 2] = esat_poly(14, t[i]);
        out[4 * i + 3] = esat_poly(21, t[i]);
    }
}
