// Pinned native WRF4.7.1 GNU13/glibc2.39 strict MP52/53 math only.
// Reused from commit c0ffc53b5d751b2a2fc35c747b57b8554795da82^.
// Gamma: Apache-2.0 original project work (docs/gf_gamma_known_delta.md).
// FDLIBM helpers: licenses/LICENSE-FDLIBM-SunPro.txt.
// Arm helpers: licenses/LICENSE-Arm-optimized-routines-MIT.txt.
// This reproduces the named native build, including its rounding errors.
#if defined(P3_NATIVE_GNU239)
// glibc 2.39 float32 transcendentals, shared device code.
//
// PROVENANCE.  Lifted VERBATIM from gpuwm/core/kernels/gf.cu on 2026-08-28,
// where it was lines 235-243 (the FP contraction pins) and 538-1141 (the
// transcriptions).  Not a re-derivation and not a tidy-up: the text is the
// text that gf's 396 parity tests already grade at max_ulp 0 against the
// live glibc 2.39 sweep fixtures gpuwm/data/gf/oracle/gf-libm-*.csv.
//
// WHY IT EXISTS.  p3_native_log/p3_native_exp/p3_native_pow are glibc's own words, not CUDA's.
// CUDA's expf/powf/tgammaf are DIFFERENT functions -- gf.cu's header records
// the measured divergence and keeps a negative control proving it is real --
// so any kernel graded bitwise against a gfortran/glibc oracle must call
// these and not the builtins.  New Tiedtke (cu_physics=16) needs exactly
// p3_native_log, p3_native_exp and p3_native_pow: its whole libm surface is 9 exp, 1 log,
// 10 sqrt and four pow forms (**t13, **0.5777, **0.2, **0.5).  sqrtf is
// correctly rounded on both sides and needs nothing.
//
// This would have been the THIRD transcription in the tree -- gf.cu's
// p3_native_*, and noahmp_leaves.cu's r_log/r_exp/r_pow, which gf.cu:40 records
// it was renamed from.  Two copies is a duplicate; three is a drift hazard.
//
// The contraction pins travel WITH the block because the block uses all
// nine of them 294 times and they are what makes its rounding reproducible
// (the __fmaf_rn/__fmul_rn/__fadd_rn intrinsics are NVIDIA-guaranteed never
// merged, so pinning to the fused form is free).  gf.cu KEEPS its own
// copies at 235-238; identical redefinition of an object-like or
// function-like macro is legal C++ and nothing else in gf.cu moves.
//
// MEASURED, before and after the lift (compile-only, no launch, sm_120 /
// NVRTC 13.0, tools-side probe): the assembled gf source changes by 346
// chars and the cubin by 796 bytes of 4,639,695 -- entirely the ELF
// section-name table (.nv.constant3 <-> .nv.constant4, .nv.global.init
// reordered).  All seven gf entry points keep byte-identical
// local_size_bytes, num_regs and const_size_bytes.

#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))
#define FSQRT(a)   __fsqrt_rn(a)
#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define DDIV(a, b) __ddiv_rn((a), (b))

// ==========================================================================
// glibc 2.39 float32 transcendentals (rule 3)
// ==========================================================================
// e_logf_data.c
__device__ const double P3_NATIVE_LOGF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double P3_NATIVE_LOGF_LOGC[16] = {
    -0x1.57bf7808caadep-2, -0x1.2bef0a7c06ddbp-2, -0x1.01eae7f513a67p-2,
    -0x1.b31d8a68224e9p-3, -0x1.6574f0ac07758p-3, -0x1.1aa2bc79c81p-3,
    -0x1.a4e76ce8c0e5ep-4, -0x1.1973c5a611cccp-4, -0x1.252f438e10c1ep-5,
     0x0p+0,                0x1.aa5aa5df25984p-5,  0x1.c5e53aa362eb4p-4,
     0x1.526e57720db08p-3,  0x1.bc2860d22477p-3,   0x1.1058bc8a07ee1p-2,
     0x1.4043057b6ee09p-2 };
#define P3_NATIVE_LOGF_LN2 0x1.62e42fefa39efp-1
#define P3_NATIVE_LOGF_A0 (-0x1.00ea348b88334p-2)
#define P3_NATIVE_LOGF_A1 (0x1.5575b0be00b6ap-2)
#define P3_NATIVE_LOGF_A2 (-0x1.ffffef20a4123p-2)

// e_exp2f_data.c, shared by expf / exp2f / powf.  EXP2F_TABLE_BITS = 5.
__device__ const unsigned long long P3_NATIVE_EXP2F_TAB[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL,
    0x3fef9301d0125b51ULL, 0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
    0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL, 0x3fef06fe0a31b715ULL,
    0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL,
    0x3feea47eb03a5585ULL, 0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
    0x3feea11473eb0187ULL, 0x3feea589994cce13ULL, 0x3feeace5422aa0dbULL,
    0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL,
    0x3fef3720dcef9069ULL, 0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
    0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL };
#define P3_NATIVE_EXP2F_P0 0x1.c6af84b912394p-5
#define P3_NATIVE_EXP2F_P1 0x1.ebfce50fac4f3p-3
#define P3_NATIVE_EXP2F_P2 0x1.62e42ff0c52d6p-1
#define P3_NATIVE_EXP2F_SHIFT 0x1.8p+52
#define P3_NATIVE_EXP2F_SHIFT_SCALED (0x1.8p+52 / 32.0)

// e_powf_log2_data.c.  POWF_SCALE is 1.0 (TOINT_INTRINSICS = 0 on x86-64).
__device__ const double P3_NATIVE_POWF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double P3_NATIVE_POWF_LOGC[16] = {
    -0x1.efec65b963019p-2, -0x1.b0b6832d4fca4p-2, -0x1.7418b0a1fb77bp-2,
    -0x1.39de91a6dcf7bp-2, -0x1.01d9bf3f2b631p-2, -0x1.97c1d1b3b7afp-3,
    -0x1.2f9e393af3c9fp-3, -0x1.960cbbf788d5cp-4, -0x1.a6f9db6475fcep-5,
     0x0p+0,                0x1.338ca9f24f53dp-4,  0x1.476a9543891bap-3,
     0x1.e840b4ac4e4d2p-3,  0x1.40645f0c6651cp-2,  0x1.88e9c2c1b9ff8p-2,
     0x1.ce0a44eb17bccp-2 };
__device__ const double P3_NATIVE_POWF_A[5] = {
     0x1.27616c9496e0bp-2, -0x1.71969a075c67ap-2,  0x1.ec70a6ca7baddp-2,
    -0x1.7154748bef6c8p-1,  0x1.71547652ab82bp+0 };

// glibc 2.39 sysdeps/ieee754/flt-32/e_logf.c
__device__ float p3_native_log(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
        if (ix * 2u == 0u) return __int_as_float(0xff800000);
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) || ix * 2u >= 0xff000000u)
            return __int_as_float(0x7fc00000);
        ix = __float_as_uint(FMUL(x, 8388608.0f));   /* 0x1p23f */
        ix -= 23u << 23;
    }
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    int k = (int)tmp >> 23;
    unsigned int iz = ix - (tmp & 0xff800000u);
    double z = (double)__uint_as_float(iz);
    double r = DSUB(DMUL(z, P3_NATIVE_LOGF_INVC[i]), 1.0);
    double y0 = DADD(P3_NATIVE_LOGF_LOGC[i], DMUL((double)k, P3_NATIVE_LOGF_LN2));
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(P3_NATIVE_LOGF_A1, r), P3_NATIVE_LOGF_A2);
    y = DADD(DMUL(P3_NATIVE_LOGF_A0, r2), y);
    y = DADD(DMUL(y, r2), DADD(y0, r));
    return __double2float_rn(y);
}

// Round a double to binary32, INCLUDING into the subnormal range.  On this
// toolchain `__double2float_rn` flushes a subnormal result to zero (CuPy
// appends -ftz=true and the compiler emits the flush after the conversion),
// while glibc's expf/powf do produce subnormals.  The correctly rounded
// subnormal is recovered exactly: m * 2^-149 scaling is exact in binary64
// over this band and rint rounds ties to even.  Same function, same
// reasoning, as noahmp_leaves.cu::nmp_d2f_rn -- the sm_120 FP32-DAZ
// countermeasure this repo has already proven.
__device__ float p3_native_d2f_rn(double y)
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

// The 32-entry exp2 core shared by glibc's expf, exp2f and powf.
__device__ double p3_native_exp2_core(double xd, double shift,
                                double p0, double p1, double p2,
                                unsigned long long sign_bias)
{
    double kd = DADD(xd, shift);
    unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = DSUB(kd, shift);
    double r = DSUB(xd, kd);
    unsigned long long t = P3_NATIVE_EXP2F_TAB[ki & 31ULL];
    t += (ki + sign_bias) << (52 - 5);
    double s = __longlong_as_double((long long)t);
    double z = DADD(DMUL(p0, r), p1);
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(p2, r), 1.0);
    y = DADD(DMUL(z, r2), y);
    return DMUL(y, s);
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_expf.c
__device__ float p3_native_exp(float x)
{
    unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return FADD(x, x);
        if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
        if (x < -__int_as_float(0x42cff1b4)) return 0.0f;
    }
    double xd = (double)x;
    double z = DMUL(0x1.71547652b82fep+0 * 32.0, xd);
    return p3_native_d2f_rn(p3_native_exp2_core(
        z, P3_NATIVE_EXP2F_SHIFT,
        P3_NATIVE_EXP2F_P0 / 32.0 / 32.0 / 32.0,
        P3_NATIVE_EXP2F_P1 / 32.0 / 32.0,
        P3_NATIVE_EXP2F_P2 / 32.0, 0ULL));
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_exp2f.c -- the identical core with
// the pre-scaled shift and unscaled polynomial.  tgammaf's Stirling arm is
// the only caller in this kernel and hands it |x| <= ~2.6, but the special
// cases are transcribed anyway so the sweep can grade the whole function.
__device__ float p3_native_exp2(float x)
{
    unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(128.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return FADD(x, x);
        if (x > 0.0f) return __int_as_float(0x7f800000);
        if (x <= -150.0f) return 0.0f;
    }
    double xd = (double)x;
    return p3_native_d2f_rn(p3_native_exp2_core(
        xd, P3_NATIVE_EXP2F_SHIFT_SCALED,
        P3_NATIVE_EXP2F_P0, P3_NATIVE_EXP2F_P1, P3_NATIVE_EXP2F_P2, 0ULL));
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_powf.c log2_inline
__device__ double p3_native_powf_log2(unsigned int ix)
{
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    unsigned int top = tmp & 0xff800000u;
    unsigned int iz = ix - top;
    int k = (int)top >> 23;
    double z = (double)__uint_as_float(iz);
    double r = DSUB(DMUL(z, P3_NATIVE_POWF_INVC[i]), 1.0);
    double y0 = DADD(P3_NATIVE_POWF_LOGC[i], (double)k);
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(P3_NATIVE_POWF_A[0], r), P3_NATIVE_POWF_A[1]);
    double p = DADD(DMUL(P3_NATIVE_POWF_A[2], r), P3_NATIVE_POWF_A[3]);
    double r4 = DMUL(r2, r2);
    double q = DADD(DMUL(P3_NATIVE_POWF_A[4], r), y0);
    q = DADD(DMUL(p, r2), q);
    return DADD(DMUL(y, r4), q);
}

__device__ int p3_native_checkint(unsigned int iy)
{
    int e = (int)(iy >> 23 & 0xffu);
    if (e < 0x7f) return 0;
    if (e > 0x7f + 23) return 2;
    if (iy & ((1u << (0x7f + 23 - e)) - 1u)) return 0;
    if (iy & (1u << (0x7f + 23 - e))) return 1;
    return 2;
}

__device__ bool p3_native_zeroinfnan(unsigned int ix)
{
    return 2u * ix - 1u >= 2u * 0x7f800000u - 1u;
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_powf.c, full special-case surface:
// the beta-shape powers reach kratio == 0 and kratio == 1 on every column
// (powf(0, +y) and powf(+0-adjacent bases), so the zero/int paths are live.
__device__ float p3_native_pow(float x, float y)
{
    unsigned int sign_bias = 0u;
    unsigned int ix = __float_as_uint(x);
    unsigned int iy = __float_as_uint(y);
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || p3_native_zeroinfnan(iy)) {
        if (p3_native_zeroinfnan(iy)) {
            if (2u * iy == 0u) return 1.0f;
            if (ix == 0x3f800000u) return 1.0f;
            if (2u * ix > 2u * 0x7f800000u || 2u * iy > 2u * 0x7f800000u)
                return FADD(x, y);
            if (2u * ix == 2u * 0x3f800000u) return 1.0f;
            if ((2u * ix < 2u * 0x3f800000u) == !(iy & 0x80000000u))
                return 0.0f;
            return FMUL(y, y);
        }
        if (p3_native_zeroinfnan(ix)) {
            float x2 = FMUL(x, x);
            if ((ix & 0x80000000u) && p3_native_checkint(iy) == 1) x2 = -x2;
            return (iy & 0x80000000u) ? FDIV(1.0f, x2) : x2;
        }
        if (ix & 0x80000000u) {
            int yint = p3_native_checkint(iy);
            if (yint == 0) return __int_as_float(0x7fc00000);
            if (yint == 1) sign_bias = 1u << (5 + 11);
            ix &= 0x7fffffffu;
        }
        if (ix < 0x00800000u) {
            ix = __float_as_uint(FMUL(x, 8388608.0f)) & 0x7fffffffu;
            ix -= 23u << 23;
        }
    }
    double logx = p3_native_powf_log2(ix);
    double ylogx = DMUL((double)y, logx);
    unsigned int hi = (unsigned int)
        (((unsigned long long)__double_as_longlong(ylogx) >> 47) & 0xffffULL);
    if (hi >= (unsigned int)
            (((unsigned long long)__double_as_longlong(126.0) >> 47) & 0xffffULL)) {
        if (ylogx > 0x1.fffffffd1d571p+6)
            return sign_bias ? __int_as_float(0xff800000)
                             : __int_as_float(0x7f800000);
        if (ylogx <= -150.0) return sign_bias ? -0.0f : 0.0f;
    }
    return p3_native_d2f_rn(
        p3_native_exp2_core(ylogx, P3_NATIVE_EXP2F_SHIFT_SCALED,
                      P3_NATIVE_EXP2F_P0, P3_NATIVE_EXP2F_P1, P3_NATIVE_EXP2F_P2,
                      (unsigned long long)sign_bias));
}

// --------------------------------------------------------------------------
// glibc 2.39 sysdeps/ieee754/flt-32/s_expm1f.c (SunPro FP32 kernel).  No
// ifunc variant exists on x86-64, so every operation is a plain float32
// op in the written association order -- FMUL/FADD/FSUB/FDIV, never FMA.
// Constant words verified against the decimal literals, not the source
// comments (the C_ATAN precedent: glibc comments have lied before).
// --------------------------------------------------------------------------
#define EM1_HUGE   __uint_as_float(0x7149F2CAu)   /* 1.0e+30 */
#define EM1_OTHR   __uint_as_float(0x42B17180u)   /* o_threshold */
#define EM1_LN2HI  __uint_as_float(0x3F317180u)
#define EM1_LN2LO  __uint_as_float(0x3717F7D1u)
#define EM1_IVLN2  __uint_as_float(0x3FB8AA3Bu)
#define EM1_Q1     __uint_as_float(0xBD088889u)
#define EM1_Q2     __uint_as_float(0x3AD00D01u)
#define EM1_Q3     __uint_as_float(0xB8A670CDu)
#define EM1_Q4     __uint_as_float(0x36867E54u)
#define EM1_Q5     __uint_as_float(0xB457EDBBu)
#define EM1_TINYM1 __uint_as_float(0xBF800000u)   /* tiny - one == -1.0f */

__device__ float p3_native_expm1(float x)
{
    float y, hi, lo, c, t, e, hxs, hfx, r1;
    int k, xsb;
    unsigned int hx = __float_as_uint(x);
    xsb = (int)(hx & 0x80000000u);
    hx &= 0x7fffffffu;
    c = 0.0f;

    if (hx >= 0x4195b844u) {                 /* |x| >= 27*ln2 */
        if (hx >= 0x42b17218u) {             /* |x| >= 88.721... */
            if (hx > 0x7f800000u) return FADD(x, x);            /* NaN */
            if (hx == 0x7f800000u)
                return (xsb == 0) ? x : -1.0f;                  /* +-inf */
            if (x > EM1_OTHR) return FMUL(EM1_HUGE, EM1_HUGE);  /* oflow */
        }
        if (xsb != 0) return EM1_TINYM1;     /* x < -27*ln2: -1 */
    }

    if (hx > 0x3eb17218u) {                  /* |x| > 0.5 ln2 */
        if (hx < 0x3F851592u) {              /* |x| < 1.5 ln2 */
            if (xsb == 0) { hi = FSUB(x, EM1_LN2HI); lo = EM1_LN2LO;  k = 1; }
            else          { hi = FADD(x, EM1_LN2HI); lo = -EM1_LN2LO; k = -1; }
        } else {
            float kf = FADD(FMUL(EM1_IVLN2, x), (xsb == 0) ? 0.5f : -0.5f);
            k  = (int)kf;
            t  = (float)k;
            hi = FSUB(x, FMUL(t, EM1_LN2HI));
            lo = FMUL(t, EM1_LN2LO);
        }
        x = FSUB(hi, lo);
        c = FSUB(FSUB(hi, x), lo);
    } else if (hx < 0x33000000u) {           /* |x| < 2**-25 */
        t = FADD(EM1_HUGE, x);
        return FSUB(x, FSUB(t, FADD(EM1_HUGE, x)));
    } else {
        k = 0;
    }

    hfx = FMUL(0.5f, x);
    hxs = FMUL(x, hfx);
    r1 = FADD(1.0f, FMUL(hxs, FADD(EM1_Q1, FMUL(hxs, FADD(EM1_Q2,
             FMUL(hxs, FADD(EM1_Q3, FMUL(hxs, FADD(EM1_Q4,
             FMUL(hxs, EM1_Q5))))))))));
    t = FSUB(3.0f, FMUL(r1, hfx));
    e = FMUL(hxs, FDIV(FSUB(r1, t), FSUB(6.0f, FMUL(x, t))));
    if (k == 0) return FSUB(x, FSUB(FMUL(x, e), hxs));
    e = FSUB(FMUL(x, FSUB(e, c)), c);
    e = FSUB(e, hxs);
    if (k == -1) return FSUB(FMUL(0.5f, FSUB(x, e)), 0.5f);
    if (k == 1) {
        if (x < -0.25f) return FMUL(-2.0f, FSUB(e, FADD(x, 0.5f)));
        return FADD(1.0f, FMUL(2.0f, FSUB(x, e)));
    }
    if (k <= -2 || k > 56) {
        y = FSUB(1.0f, FSUB(e, x));
        y = __uint_as_float(__float_as_uint(y) + ((unsigned int)k << 23));
        return FSUB(y, 1.0f);
    }
    if (k < 23) {
        t = __uint_as_float(0x3f800000u - (0x1000000u >> k)); /* 1-2^-k */
        y = FSUB(t, FSUB(e, x));
        y = __uint_as_float(__float_as_uint(y) + ((unsigned int)k << 23));
    } else {
        t = __uint_as_float((unsigned int)(0x7f - k) << 23);  /* 2^-k */
        y = FSUB(x, FADD(e, t));
        y = FADD(y, 1.0f);
        y = __uint_as_float(__float_as_uint(y) + ((unsigned int)k << 23));
    }
    return y;
}

// --------------------------------------------------------------------------
// glibc 2.39 sysdeps/ieee754/flt-32/e_lgammaf_r.c, POSITIVE arm only.  The
// negative-x machinery (sin_pif, __lgamma_negf) is deliberately absent:
// tgammaf's callers in this kernel hand it x in (0.5, 2.5) and the sweep
// grades (0.4, 2.6) plus the (2, 8) tail; a negative or non-finite argument
// returns NaN rather than a value this kernel cannot vouch for.  No ifunc
// variant exists on x86-64: plain float32 ops, written association order.
// Every word below was verified against the decimal literal.
// --------------------------------------------------------------------------
#define LG_A0  __uint_as_float(0x3D9E233Fu)
#define LG_A1  __uint_as_float(0x3EA51A66u)
#define LG_A2  __uint_as_float(0x3D89F001u)
#define LG_A3  __uint_as_float(0x3CA89915u)
#define LG_A4  __uint_as_float(0x3BF2027Eu)
#define LG_A5  __uint_as_float(0x3B3D6EC6u)
#define LG_A6  __uint_as_float(0x3A9C54A1u)
#define LG_A7  __uint_as_float(0x3A05B634u)
#define LG_A8  __uint_as_float(0x39679767u)
#define LG_A9  __uint_as_float(0x38E28445u)
#define LG_A10 __uint_as_float(0x37D383A2u)
#define LG_A11 __uint_as_float(0x383C2C75u)
#define LG_TC  __uint_as_float(0x3FBB16C3u)
#define LG_TF  __uint_as_float(0xBDF8CDCDu)
#define LG_TT  __uint_as_float(0x31E61C52u)
// tc - one, folded on the host: FP32 constant-constant subtraction must not
// reach ptxas (rule 2).  Exact: 1.4616321325 - 1 loses no mantissa bits.
#define LG_TCM1 __uint_as_float(0x3EEC5B0Cu)
#define LG_T0  __uint_as_float(0x3EF7B95Eu)
#define LG_T1  __uint_as_float(0xBE17213Cu)
#define LG_T2  __uint_as_float(0x3D845A15u)
#define LG_T3  __uint_as_float(0xBD064D47u)
#define LG_T4  __uint_as_float(0x3C93373Du)
#define LG_T5  __uint_as_float(0xBC28FCFEu)
#define LG_T6  __uint_as_float(0x3BC7E707u)
#define LG_T7  __uint_as_float(0xBB7177FEu)
#define LG_T8  __uint_as_float(0x3B141699u)
#define LG_T9  __uint_as_float(0xBAB7F476u)
#define LG_T10 __uint_as_float(0x3A66F867u)
#define LG_T11 __uint_as_float(0xBA0D3085u)
#define LG_T12 __uint_as_float(0x39A57B6Bu)
#define LG_T13 __uint_as_float(0xB9A3F927u)
#define LG_T14 __uint_as_float(0x39AFE9F7u)
#define LG_U0  __uint_as_float(0xBD9E233Fu)
#define LG_U1  __uint_as_float(0x3F2200F4u)
#define LG_U2  __uint_as_float(0x3FBA3AE7u)
#define LG_U3  __uint_as_float(0x3F7A4BB2u)
#define LG_U4  __uint_as_float(0x3E6A7578u)
#define LG_U5  __uint_as_float(0x3C5B3C5Eu)
#define LG_V1  __uint_as_float(0x401D2EBEu)
#define LG_V2  __uint_as_float(0x4008392Du)
#define LG_V3  __uint_as_float(0x3F44EFDFu)
#define LG_V4  __uint_as_float(0x3DD572AFu)
#define LG_V5  __uint_as_float(0x3B52D5DBu)
#define LG_S0  __uint_as_float(0xBD9E233Fu)
#define LG_S1  __uint_as_float(0x3E5C245Au)
#define LG_S2  __uint_as_float(0x3EA6CC7Au)
#define LG_S3  __uint_as_float(0x3E15DCE6u)
#define LG_S4  __uint_as_float(0x3CDA40E4u)
#define LG_S5  __uint_as_float(0x3AF135B4u)
#define LG_S6  __uint_as_float(0x3805FF67u)
#define LG_R1  __uint_as_float(0x3FB22D3Bu)
#define LG_R2  __uint_as_float(0x3F38D0C5u)
#define LG_R3  __uint_as_float(0x3E300F6Eu)
#define LG_R4  __uint_as_float(0x3C98BF54u)
#define LG_R5  __uint_as_float(0x3A4BEED6u)
#define LG_R6  __uint_as_float(0x36F5D7BDu)
#define LG_W0  __uint_as_float(0x3ED67F1Du)
#define LG_W1  __uint_as_float(0x3DAAAAABu)
#define LG_W2  __uint_as_float(0xBB360B61u)
#define LG_W3  __uint_as_float(0x3A500CFDu)
#define LG_W4  __uint_as_float(0xBA1C065Cu)
#define LG_W5  __uint_as_float(0x3A5B3DD2u)
#define LG_W6  __uint_as_float(0xBAD5C4E8u)

__device__ float p3_native_lgamma_pos(float x)
{
    unsigned int hx = __float_as_uint(x);
    int ix = (int)(hx & 0x7fffffffu);
    if ((int)hx < 0 ) return __int_as_float(0x7fc00000);  /* not ported */
    if (ix >= 0x7f800000) return FMUL(x, x);
    if (ix == 0) return FDIV(1.0f, fabsf(x));
    if (ix < 0x30800000) return -p3_native_log(x);   /* |x| < 2**-30 */

    float t, y, z, p, p1, p2, p3, q, r, w;
    int i;
    y = 0.0f; i = 0;
    if (ix == 0x3f800000 || ix == 0x40000000) {
        r = 0.0f;
    } else if (ix < 0x40000000) {            /* x < 2.0 */
        if (ix <= 0x3f666666) {              /* lgamma(x) = lgamma(x+1)-log(x) */
            r = -p3_native_log(x);
            if (ix >= 0x3f3b4a20)      { y = FSUB(1.0f, x); i = 0; }
            else if (ix >= 0x3e6d3308) { y = FSUB(x, LG_TCM1); i = 1; }
            else                       { y = x; i = 2; }
        } else {
            r = 0.0f;
            if (ix >= 0x3fdda618)      { y = FSUB(2.0f, x); i = 0; }
            else if (ix >= 0x3F9da620) { y = FSUB(x, LG_TC); i = 1; }
            else                       { y = FSUB(x, 1.0f); i = 2; }
        }
        switch (i) {
        case 0:
            z = FMUL(y, y);
            p1 = FADD(LG_A0, FMUL(z, FADD(LG_A2, FMUL(z, FADD(LG_A4,
                 FMUL(z, FADD(LG_A6, FMUL(z, FADD(LG_A8,
                 FMUL(z, LG_A10))))))))));
            p2 = FMUL(z, FADD(LG_A1, FMUL(z, FADD(LG_A3, FMUL(z, FADD(LG_A5,
                 FMUL(z, FADD(LG_A7, FMUL(z, FADD(LG_A9,
                 FMUL(z, LG_A11)))))))))));
            p = FADD(FMUL(y, p1), p2);
            r = FADD(r, FSUB(p, FMUL(0.5f, y)));
            break;
        case 1:
            z = FMUL(y, y);
            w = FMUL(z, y);
            p1 = FADD(LG_T0, FMUL(w, FADD(LG_T3, FMUL(w, FADD(LG_T6,
                 FMUL(w, FADD(LG_T9, FMUL(w, LG_T12))))))));
            p2 = FADD(LG_T1, FMUL(w, FADD(LG_T4, FMUL(w, FADD(LG_T7,
                 FMUL(w, FADD(LG_T10, FMUL(w, LG_T13))))))));
            p3 = FADD(LG_T2, FMUL(w, FADD(LG_T5, FMUL(w, FADD(LG_T8,
                 FMUL(w, FADD(LG_T11, FMUL(w, LG_T14))))))));
            p = FSUB(FMUL(z, p1), FSUB(LG_TT, FMUL(w, FADD(p2, FMUL(y, p3)))));
            r = FADD(r, FADD(LG_TF, p));
            break;
        case 2:
            p1 = FMUL(y, FADD(LG_U0, FMUL(y, FADD(LG_U1, FMUL(y, FADD(LG_U2,
                 FMUL(y, FADD(LG_U3, FMUL(y, FADD(LG_U4,
                 FMUL(y, LG_U5)))))))))));
            p2 = FADD(1.0f, FMUL(y, FADD(LG_V1, FMUL(y, FADD(LG_V2,
                 FMUL(y, FADD(LG_V3, FMUL(y, FADD(LG_V4,
                 FMUL(y, LG_V5))))))))));
            r = FADD(r, FADD(FMUL(-0.5f, y), FDIV(p1, p2)));
            break;
        }
    } else if (ix < 0x41000000) {            /* x < 8.0 */
        i = (int)x;
        y = FSUB(x, (float)i);
        p = FMUL(y, FADD(LG_S0, FMUL(y, FADD(LG_S1, FMUL(y, FADD(LG_S2,
            FMUL(y, FADD(LG_S3, FMUL(y, FADD(LG_S4, FMUL(y, FADD(LG_S5,
            FMUL(y, LG_S6)))))))))))));
        q = FADD(1.0f, FMUL(y, FADD(LG_R1, FMUL(y, FADD(LG_R2, FMUL(y,
            FADD(LG_R3, FMUL(y, FADD(LG_R4, FMUL(y, FADD(LG_R5,
            FMUL(y, LG_R6))))))))))));
        r = FADD(FMUL(0.5f, y), FDIV(p, q));
        z = 1.0f;
        switch (i) {                          /* lgamma(1+s) = log(s)+lgamma(s) */
        case 7: z = FMUL(z, FADD(y, 6.0f));   /* FALLTHRU */
        case 6: z = FMUL(z, FADD(y, 5.0f));   /* FALLTHRU */
        case 5: z = FMUL(z, FADD(y, 4.0f));   /* FALLTHRU */
        case 4: z = FMUL(z, FADD(y, 3.0f));   /* FALLTHRU */
        case 3: z = FMUL(z, FADD(y, 2.0f));
                r = FADD(r, p3_native_log(z));
                break;
        }
    } else if (ix < 0x4c800000) {            /* 8.0 <= x < 2**26 */
        t = p3_native_log(x);
        z = FDIV(1.0f, x);
        y = FMUL(z, z);
        w = FADD(LG_W0, FMUL(z, FADD(LG_W1, FMUL(y, FADD(LG_W2, FMUL(y,
            FADD(LG_W3, FMUL(y, FADD(LG_W4, FMUL(y, FADD(LG_W5,
            FMUL(y, LG_W6))))))))))));
        r = FADD(FMUL(FSUB(x, 0.5f), FSUB(t, 1.0f)), w);
    } else {
        r = FMUL(x, FSUB(p3_native_log(x), 1.0f));
    }
    return r;
}

// --------------------------------------------------------------------------
// glibc 2.39 sysdeps/ieee754/dbl-64/gamma_productf.c: the float
// __gamma_productf computed in double, which is what x86-64 links.
// --------------------------------------------------------------------------
__device__ float p3_native_gamma_product(float x, float x_eps, int n, float *eps)
{
    double x_full = DADD((double)x, (double)x_eps);
    double ret = x_full;
    for (int i = 1; i < n; i++)
        ret = DMUL(ret, DADD(x_full, (double)i));
    float fret = __double2float_rn(ret);
    *eps = __double2float_rn(DDIV(DSUB(ret, (double)fret), (double)fret));
    return fret;
}

// --------------------------------------------------------------------------
// glibc 2.39 sysdeps/ieee754/flt-32/e_gammaf_r.c, positive arm.  x <= 0,
// NaN and inf return NaN (not ported -- the scheme cannot produce them:
// alpha = (tunning*(beta-2)+1)/(1-tunning) with tunning clamped to
// [.2, .9] and beta in {1.3, 2.5, 4.} keeps every argument in
// [1.06, 32.2)).  x >= 36 overflows to +inf exactly as glibc's
// FLT_MAX*FLT_MAX does.
// --------------------------------------------------------------------------
#define GAM_C0     __uint_as_float(0x3DAAAAABu)   /* 0x1.555556p-4 */
#define GAM_C1     __uint_as_float(0xBB360B61u)   /* -0xb.60b61p-12 */
#define GAM_C2     __uint_as_float(0x3A500D01u)   /* 0x3.403404p-12 */
#define GAM_SQRT12 __uint_as_float(0x3F3504F3u)   /* M_SQRT1_2f */
#define GAM_TWOPI  __uint_as_float(0x40C90FDBu)   /* 2*M_PIf, host-folded */

__device__ float p3_native_gammaf_positive(float x, int *exp2_adj)
{
    if (x < 0.5f) {
        *exp2_adj = 0;
        return FDIV(p3_native_exp(p3_native_lgamma_pos(FADD(x, 1.0f))), x);
    } else if (x <= 1.5f) {
        *exp2_adj = 0;
        return p3_native_exp(p3_native_lgamma_pos(x));
    } else if (x < 2.5f) {
        *exp2_adj = 0;
        float x_adj = FSUB(x, 1.0f);
        return FMUL(p3_native_exp(p3_native_lgamma_pos(x_adj)), x_adj);
    } else {
        float eps = 0.0f;
        float x_eps = 0.0f;
        float x_adj = x;
        float prod = 1.0f;
        if (x < 4.0f) {
            float n = ceilf(FSUB(4.0f, x));
            x_adj = FADD(x, n);
            x_eps = FSUB(x, FSUB(x_adj, n));
            prod = p3_native_gamma_product(FSUB(x_adj, n), x_eps, (int)n, &eps);
        }
        float exp_adj = -eps;
        float x_adj_int = roundf(x_adj);
        float x_adj_frac = FSUB(x_adj, x_adj_int);
        int x_adj_log2;
        float x_adj_mant = frexpf(x_adj, &x_adj_log2);
        if (x_adj_mant < GAM_SQRT12) {
            x_adj_log2--;
            x_adj_mant = FMUL(x_adj_mant, 2.0f);
        }
        *exp2_adj = x_adj_log2 * (int)x_adj_int;
        float ret = FDIV(FMUL(FMUL(FMUL(
            p3_native_pow(x_adj_mant, x_adj),
            p3_native_exp2(FMUL((float)x_adj_log2, x_adj_frac))),
            p3_native_exp(-x_adj)),
            FSQRT(FDIV(GAM_TWOPI, x_adj))),
            prod);
        exp_adj = FADD(exp_adj, FMUL(x_eps, p3_native_log(x_adj)));
        float bsum = GAM_C2;
        float x_adj2 = FMUL(x_adj, x_adj);
        bsum = FADD(FDIV(bsum, x_adj2), GAM_C1);
        bsum = FADD(FDIV(bsum, x_adj2), GAM_C0);
        exp_adj = FADD(exp_adj, FDIV(bsum, x_adj));
        return FADD(ret, FMUL(ret, p3_native_expm1(exp_adj)));
    }
}

__device__ float p3_native_tgamma(float x)
{
    unsigned int hx = __float_as_uint(x);
    if ((hx & 0x80000000u) || (hx & 0x7fffffffu) >= 0x7f800000u
        || (hx & 0x7fffffffu) == 0u)
        return __int_as_float(0x7fc00000);   /* outside the ported domain */
    if (x >= 36.0f)
        return __int_as_float(0x7f800000);   /* FLT_MAX*FLT_MAX overflow */
    int exp2_adj;
    float tret = p3_native_gammaf_positive(x, &exp2_adj);
    float ret = scalbnf(tret, exp2_adj);
    // glibc's isinf/iszero fixups return the same +inf / +0 words for a
    // positive argument; nothing further to do.
    return ret;
}

#endif
