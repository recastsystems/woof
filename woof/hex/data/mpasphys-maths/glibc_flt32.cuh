// ======================================================================
// THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
// third-party work.  ArWen distributes the file under the Apache License
// 2.0; the notices below belong to the transcribed parts and are kept
// here because their own licences require it.  Full texts are in the
// repository NOTICE and in the licenses/ directory.
//
// For the two libm grants the text also sits beside the code, in
// gpuwm/core/kernels/LICENSE-third-party.txt.
//
//   Arm optimized-routines -- the logf, expf, exp2f and powf cores and
//   their data tables github.com/ARM-software/optimized-routines:
//   math/logf.c, math/expf.c, math/exp2f.c, math/powf.c and the matching
//   math/*_data.c, published August 2017 and imported into glibc for
//   2.27/2.28 by their own author; transcribed here from glibc 2.39
//   sysdeps/ieee754/flt-32/.
//
//       Copyright (c) 2017-2018, Arm Limited.
//       SPDX-License-Identifier: MIT
//
//   Taken under the MIT branch of Arm's grant.  MIT requires the
//   copyright notice above and its permission notice to travel with every
//   copy; the permission notice is reproduced in full in the files named
//   above.
//
//   NO FDLIBM CODE REMAINS IN THIS FILE.  It carried transcriptions of
//   FDLIBM's expm1f (s_expm1f.c) and lgammaf reduction (e_lgammaf_r.c)
//   through ArWen 2.6.5; both were deleted at 2.6.6 with the gamma block
//   that was their only caller, so the Sun notice that had to travel with
//   them travels no longer -- there is nothing here for it to attach to.
//   The notice is unchanged and still reproduced, for the twelve files
//   that DO carry FDLIBM code -- nine .cu in this directory and three
//   Python modules under gpuwm/core/ -- in the repository NOTICE, in
//   licenses/LICENSE-FDLIBM-SunPro.txt and in
//   gpuwm/core/kernels/LICENSE-third-party.txt.
//
//   Nothing else in this file is a transcription of anything.  gfk_d2f_rn
//   is ArWen's own subnormal-rounding countermeasure and gfk_tgamma is
//   ArWen's own correctly rounded gamma (see its block below and
//   docs/gf_gamma_known_delta.md); both are Apache-2.0 original work.
// ======================================================================
// glibc 2.39 float32 transcendentals, shared device code.
//
// PROVENANCE.  Lifted VERBATIM from gpuwm/core/kernels/gf.cu on 2026-08-28,
// where it was lines 235-243 (the FP contraction pins) and 538-1141 (the
// transcriptions).  Not a re-derivation and not a tidy-up: the text is the
// text that gf's 396 parity tests already grade at max_ulp 0 against the
// live glibc 2.39 sweep fixtures gpuwm/data/gf/oracle/gf-libm-*.csv.
//
// WHY IT EXISTS.  gfk_log/gfk_exp/gfk_pow are glibc's own words, not CUDA's.
// CUDA's expf/powf/tgammaf are DIFFERENT functions -- gf.cu's header records
// the measured divergence and keeps a negative control proving it is real --
// so any kernel graded bitwise against a gfortran/glibc oracle must call
// these and not the builtins.  New Tiedtke (cu_physics=16) needs exactly
// gfk_log, gfk_exp and gfk_pow: its whole libm surface is 9 exp, 1 log,
// 10 sqrt and four pow forms (**t13, **0.5777, **0.2, **0.5).  sqrtf is
// correctly rounded on both sides and needs nothing.
//
// This would have been the THIRD transcription in the tree -- gf.cu's
// gfk_*, and noahmp_leaves.cu's r_log/r_exp/r_pow, which gf.cu:40 records
// it was renamed from.  Two copies is a duplicate; three is a drift hazard.
//
// The contraction pins travel WITH the block because the block uses all
// nine of them 294 times and they are what makes its rounding reproducible
// (project rules: __fmaf_rn/__fmul_rn/__fadd_rn are NVIDIA-guaranteed never
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
__device__ const double GFK_LOGF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double GFK_LOGF_LOGC[16] = {
    -0x1.57bf7808caadep-2, -0x1.2bef0a7c06ddbp-2, -0x1.01eae7f513a67p-2,
    -0x1.b31d8a68224e9p-3, -0x1.6574f0ac07758p-3, -0x1.1aa2bc79c81p-3,
    -0x1.a4e76ce8c0e5ep-4, -0x1.1973c5a611cccp-4, -0x1.252f438e10c1ep-5,
     0x0p+0,                0x1.aa5aa5df25984p-5,  0x1.c5e53aa362eb4p-4,
     0x1.526e57720db08p-3,  0x1.bc2860d22477p-3,   0x1.1058bc8a07ee1p-2,
     0x1.4043057b6ee09p-2 };
#define GFK_LOGF_LN2 0x1.62e42fefa39efp-1
#define GFK_LOGF_A0 (-0x1.00ea348b88334p-2)
#define GFK_LOGF_A1 (0x1.5575b0be00b6ap-2)
#define GFK_LOGF_A2 (-0x1.ffffef20a4123p-2)

// e_exp2f_data.c, shared by expf / exp2f / powf.  EXP2F_TABLE_BITS = 5.
__device__ const unsigned long long GFK_EXP2F_TAB[32] = {
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
#define GFK_EXP2F_P0 0x1.c6af84b912394p-5
#define GFK_EXP2F_P1 0x1.ebfce50fac4f3p-3
#define GFK_EXP2F_P2 0x1.62e42ff0c52d6p-1
#define GFK_EXP2F_SHIFT 0x1.8p+52
#define GFK_EXP2F_SHIFT_SCALED (0x1.8p+52 / 32.0)

// e_powf_log2_data.c.  POWF_SCALE is 1.0 (TOINT_INTRINSICS = 0 on x86-64).
__device__ const double GFK_POWF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double GFK_POWF_LOGC[16] = {
    -0x1.efec65b963019p-2, -0x1.b0b6832d4fca4p-2, -0x1.7418b0a1fb77bp-2,
    -0x1.39de91a6dcf7bp-2, -0x1.01d9bf3f2b631p-2, -0x1.97c1d1b3b7afp-3,
    -0x1.2f9e393af3c9fp-3, -0x1.960cbbf788d5cp-4, -0x1.a6f9db6475fcep-5,
     0x0p+0,                0x1.338ca9f24f53dp-4,  0x1.476a9543891bap-3,
     0x1.e840b4ac4e4d2p-3,  0x1.40645f0c6651cp-2,  0x1.88e9c2c1b9ff8p-2,
     0x1.ce0a44eb17bccp-2 };
__device__ const double GFK_POWF_A[5] = {
     0x1.27616c9496e0bp-2, -0x1.71969a075c67ap-2,  0x1.ec70a6ca7baddp-2,
    -0x1.7154748bef6c8p-1,  0x1.71547652ab82bp+0 };

// glibc 2.39 sysdeps/ieee754/flt-32/e_logf.c
__device__ float gfk_log(float x)
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
    double r = DSUB(DMUL(z, GFK_LOGF_INVC[i]), 1.0);
    double y0 = DADD(GFK_LOGF_LOGC[i], DMUL((double)k, GFK_LOGF_LN2));
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(GFK_LOGF_A1, r), GFK_LOGF_A2);
    y = DADD(DMUL(GFK_LOGF_A0, r2), y);
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
__device__ float gfk_d2f_rn(double y)
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
__device__ double gfk_exp2_core(double xd, double shift,
                                double p0, double p1, double p2,
                                unsigned long long sign_bias)
{
    double kd = DADD(xd, shift);
    unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = DSUB(kd, shift);
    double r = DSUB(xd, kd);
    unsigned long long t = GFK_EXP2F_TAB[ki & 31ULL];
    t += (ki + sign_bias) << (52 - 5);
    double s = __longlong_as_double((long long)t);
    double z = DADD(DMUL(p0, r), p1);
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(p2, r), 1.0);
    y = DADD(DMUL(z, r2), y);
    return DMUL(y, s);
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_expf.c
__device__ float gfk_exp(float x)
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
    return gfk_d2f_rn(gfk_exp2_core(
        z, GFK_EXP2F_SHIFT,
        GFK_EXP2F_P0 / 32.0 / 32.0 / 32.0,
        GFK_EXP2F_P1 / 32.0 / 32.0,
        GFK_EXP2F_P2 / 32.0, 0ULL));
}

// e_exp2f.c (exp2f), s_expm1f.c (expm1f) and e_lgammaf_r.c (lgammaf) were
// transcribed here through 2.6.5 and are DELETED at 2.6.6.  Their only
// caller anywhere in this tree was the earlier gamma this file used to end
// with, which used all three on its way to Gamma; with nothing else calling
// them they had no reason to stay.  That earlier gamma is
// gone, and ArWen's own gamma that replaced it evaluates no logarithm, no
// exponential and no exp2 at all, so all three functions became dead to the
// physics and survived only as slots of the gf_libm_unary_probe TEST
// kernel.  Dead
// transcriptions are not decoration to keep: gfk_lgamma_pos was the ONLY
// transcription of glibc's e_lgammaf_r.c anywhere in the repository, so
// deleting it retires that file from the distribution outright, and exp2f
// and expm1f leave THIS translation unit (mynn_pbl.cu and
// mynn_dmp_sibling.cu still carry their own expm1f for tanhf).  The
// 32-entry exp2 CORE above stays -- gfk_exp and gfk_pow are live physics
// and both call it.
//
// The live float32 surface of this header is now exactly gfk_log, gfk_exp,
// gfk_pow, gfk_d2f_rn and gfk_tgamma.

// glibc 2.39 sysdeps/ieee754/flt-32/e_powf.c log2_inline
__device__ double gfk_powf_log2(unsigned int ix)
{
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    unsigned int top = tmp & 0xff800000u;
    unsigned int iz = ix - top;
    int k = (int)top >> 23;
    double z = (double)__uint_as_float(iz);
    double r = DSUB(DMUL(z, GFK_POWF_INVC[i]), 1.0);
    double y0 = DADD(GFK_POWF_LOGC[i], (double)k);
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(GFK_POWF_A[0], r), GFK_POWF_A[1]);
    double p = DADD(DMUL(GFK_POWF_A[2], r), GFK_POWF_A[3]);
    double r4 = DMUL(r2, r2);
    double q = DADD(DMUL(GFK_POWF_A[4], r), y0);
    q = DADD(DMUL(p, r2), q);
    return DADD(DMUL(y, r4), q);
}

__device__ int gfk_checkint(unsigned int iy)
{
    int e = (int)(iy >> 23 & 0xffu);
    if (e < 0x7f) return 0;
    if (e > 0x7f + 23) return 2;
    if (iy & ((1u << (0x7f + 23 - e)) - 1u)) return 0;
    if (iy & (1u << (0x7f + 23 - e))) return 1;
    return 2;
}

__device__ bool gfk_zeroinfnan(unsigned int ix)
{
    return 2u * ix - 1u >= 2u * 0x7f800000u - 1u;
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_powf.c, full special-case surface:
// the beta-shape powers reach kratio == 0 and kratio == 1 on every column
// (powf(0, +y) and powf(+0-adjacent bases), so the zero/int paths are live.
__device__ float gfk_pow(float x, float y)
{
    unsigned int sign_bias = 0u;
    unsigned int ix = __float_as_uint(x);
    unsigned int iy = __float_as_uint(y);
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || gfk_zeroinfnan(iy)) {
        if (gfk_zeroinfnan(iy)) {
            if (2u * iy == 0u) return 1.0f;
            if (ix == 0x3f800000u) return 1.0f;
            if (2u * ix > 2u * 0x7f800000u || 2u * iy > 2u * 0x7f800000u)
                return FADD(x, y);
            if (2u * ix == 2u * 0x3f800000u) return 1.0f;
            if ((2u * ix < 2u * 0x3f800000u) == !(iy & 0x80000000u))
                return 0.0f;
            return FMUL(y, y);
        }
        if (gfk_zeroinfnan(ix)) {
            float x2 = FMUL(x, x);
            if ((ix & 0x80000000u) && gfk_checkint(iy) == 1) x2 = -x2;
            return (iy & 0x80000000u) ? FDIV(1.0f, x2) : x2;
        }
        if (ix & 0x80000000u) {
            int yint = gfk_checkint(iy);
            if (yint == 0) return __int_as_float(0x7fc00000);
            if (yint == 1) sign_bias = 1u << (5 + 11);
            ix &= 0x7fffffffu;
        }
        if (ix < 0x00800000u) {
            ix = __float_as_uint(FMUL(x, 8388608.0f)) & 0x7fffffffu;
            ix -= 23u << 23;
        }
    }
    double logx = gfk_powf_log2(ix);
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
    return gfk_d2f_rn(
        gfk_exp2_core(ylogx, GFK_EXP2F_SHIFT_SCALED,
                      GFK_EXP2F_P0, GFK_EXP2F_P1, GFK_EXP2F_P2,
                      (unsigned long long)sign_bias));
}

// ==========================================================================
// gfk_tgamma -- ArWen's own float32 gamma.  ORIGINAL WORK, Apache-2.0.
// ==========================================================================
//
// WHAT THIS REPLACED.  Until 2.6.5 this file ended with ArWen's earlier
// gamma, gfk_gamma_product / gfk_gammaf_positive / gfk_tgamma, which
// returned the same float32 words as glibc 2.39's tgammaf and so carried
// that function's rounding errors.  Like the code below, it was this
// project's own work under the project's licence.  2.6.6 replaced it with
// the correctly rounded gamma below; the measurement is further down and
// in docs/gf_gamma_known_delta.md.  Nothing from the earlier gamma
// survives below: no constant, no branch structure, no algorithm.  This
// code evaluates no lgamma, performs no exponentiation, and calls
// nothing.
//
// HOW IT WAS DERIVED.  Gamma on [1,2) is a degree-9 polynomial on each of 16
// equal segments; every other positive argument reduces to [1,2) by the
// functional equation Gamma(z+1) = z*Gamma(z); negative arguments use the
// reflection formula Gamma(x)*Gamma(1-x) = pi/sin(pi x).  The 160
// coefficients in GFK_TG_C were generated from Stirling's asymptotic series
// (DLMF 5.11.1) evaluated in 113-bit arithmetic with the standard Bernoulli
// numbers B2..B24.  GFK_TG_SP holds the Taylor coefficients of sin(pi r),
// c_k = (-1)^k pi^(2k+1)/(2k+1)!, in closed form.  Classical mathematics; no
// implementation of any kind was consulted, and no third-party source was
// read, quoted or fitted against.  The audit trail is lineage-docs/gam-03*.
//
// IT IS CORRECTLY ROUNDED, AND THE REFERENCE IS NOT.  MEASURED against
// libquadmath's 113-bit tgammaq -- an oracle unrelated to glibc -- over all
// 59,768,833 float32 arguments of [0.25, 36], the interval that covers every
// value this scheme can reach:
//
//     this code       0 arguments not correctly rounded
//     glibc 2.39      23,575,230 arguments not correctly rounded (39.44 %),
//                     worst 6 ULP.  tgammaf(4.0f) returns 6.00000048, not 6.
//
// A table fitted to or copied from glibc would carry glibc's own error, ten
// orders of magnitude larger than this code's ~5.6e-17 relative agreement
// with true Gamma.  That is the structural evidence the coefficients are not
// derived from the reference.
//
// >>> DELIBERATE DIVERGENCE FROM WRF.  READ docs/gf_gamma_known_delta.md. <<<
//
// gfortran binds WRF's F2008 gamma() intrinsic to glibc's tgammaf, so this
// kernel no longer reproduces WRF's fzu bit for bit and the deep mass flux
// xmb moves by up to 7.3 per cent on converged columns (median 1.9).  That is
// the amplification recorded at gf.cu:50-56, not an accuracy loss.  MEASURED
// over every reachable (alpha, beta) -- all 53,687,093 float32 tunning values
// of the three drafts -- fzu changes on 68.17 per cent of them: 98.39 per cent
// of the set within 4 ULP, worst 12 (draft 1, beta=2.5), worst relative
// 8.39e-7.  The committed 216-column fixture spans only the first 4 ULP, which
// is why the gate below reads 4 and this bound reads 12.  The scheme's own
// xk = (xaa0-aa1)/mbdt cancellation is what turns any of it into per cent.
//
// AND OURS IS THE CLOSER ONE.  Graded against the exact 113-bit
// Gamma(a+b)/(Gamma(a)Gamma(b)) rounded once to float32, over the same
// 53,687,093: this code is worst 4 ULP from the true fzu and glibc is worst
// 11; ours is strictly closer on 49.95 per cent of the set and strictly
// further on 9.59; mean relative error 3.83e-8 against glibc's 9.67e-8.
// Neither is exact -- rounding three gammas and a multiply and a divide to
// float32 costs up to 4 ULP on its own -- so xmb is not determined to better
// than tens of per cent by ANY float32 build of this scheme, WRF's included.
//
// The divergence is NOT new to ArWen and it is NOT unbounded.  MEASURED, this
// code returns the same word as gpuwm/verify/gf_deep_ref.py::_tgammaf -- the
// float32 CPU authority's own gamma model -- on ALL 59,768,833 arguments of
// [0.25, 36], so the CPU and CUDA paths now agree bitwise where before they
// did not, and the divergence from WRF is exactly the one the CPU suite has
// carried, gated and documented since the port landed
// (tests/test_gf_deep_parity.py::test_fzu_is_the_one_measured_divergence,
// budget 4 ULP; this code lands at 4).
//
// TO GET WRF'S ANSWER BACK, pin fzu.  gfd_get_zu_zd_pdf takes fzu_override
// and gf_deep_stage exposes it in scin; passing the oracle's captured word
// makes the whole chain bitwise against WRF again, which is exactly how the
// CPU suite reaches max_ulp 0 today.  No build flag is needed and none is
// provided -- restoring glibc's bits at run time would mean shipping 22.5 MB
// of measured glibc deviation, which is the thing this change removes.
// ==========================================================================

// Gamma on [1,2): 16 equal segments, degree-9 polynomial in u = r - centre.
__device__ const double GFK_TG_C[16][10] = {
  { 0x1.f73ed01940522p-1, -0x1.092fd20dd784cp-1, 0x1.d1a2ea66d2aefp-1, -0x1.966a1d7f2b9d9p-1, 0x1.af057a1fa6b33p-1, -0x1.a114e80ce35b1p-1, 0x1.99b28f9ddcee8p-1, -0x1.8e73dca5f882p-1, 0x1.843285f62ee7dp-1, -0x1.78e83c533d8b1p-1 },   /* centre 1.031250000 */
  { 0x1.e865a5b755fb9p-1, -0x1.a6b50f60b5c6ep-2, 0x1.8e9a9675e8147p-1, -0x1.39251063e4e66p-1, 0x1.41a834d107f5ep-1, -0x1.237aaf45da8d8p-1, 0x1.0f0e4f41772eap-1, -0x1.f11209307f1cp-2, 0x1.c8cbfdb620ef9p-2, -0x1.a24012d73ac3cp-2 },   /* centre 1.093750000 */
  { 0x1.dcac35f2a7419p-1, -0x1.49cf184c91f8ep-2, 0x1.5ac61acea578ep-1, -0x1.e5d43093961e1p-2, 0x1.e91903e412f2ap-2, -0x1.9eedaece7d029p-2, 0x1.6f07e745bdddep-2, -0x1.3e2e7165ba686p-2, 0x1.14c7245554af4p-2, -0x1.df8a55df66eep-3 },   /* centre 1.156250000 */
  { 0x1.d3aa3cecb6cdp-1, -0x1.f0b8c2384c507p-3, 0x1.327dd5130ef72p-1, -0x1.7a2070b357f21p-2, 0x1.7a7cfacdf879dp-2, -0x1.2c07bb9dae965p-2, 0x1.fb753b1f31c65p-3, -0x1.a0d97e4269587p-3, 0x1.58589453b97bbp-3, -0x1.1b0d3d42d932bp-3 },   /* centre 1.218750000 */
  { 0x1.cd0ebb0c4e488p-1, -0x1.5fa4609a59d2cp-3, 0x1.13236e09cf181p-1, -0x1.2616a66cb748ap-2, 0x1.29fb9224d7007p-2, -0x1.b7c019bda7132p-3, 0x1.658f76dab2a86p-3, -0x1.16bef3649d39bp-3, 0x1.b6aee3baf8284p-4, -0x1.570897c5886ebp-4 },   /* centre 1.281250000 */
  { 0x1.c89aaab6c10fdp-1, -0x1.b8d4972a0d9cp-4, 0x1.f59ee44fdc796p-2, -0x1.c6c163713c6cbp-3, 0x1.dd53c7e7fbca8p-3, -0x1.45d61e648b785p-3, 0x1.00706690f6116p-3, -0x1.7bbc7df8eac4cp-4, 0x1.1d809aa6ec9fp-4, -0x1.a9b3bfeb56d76p-5 },   /* centre 1.343750000 */
  { 0x1.c61d286fe74edp-1, -0x1.8f960eacc3e79p-5, 0x1.d035f977bf7ffp-2, -0x1.5afe653a01c68p-3, 0x1.850c11888a092p-3, -0x1.e6f81933b909dp-4, 0x1.7607930824288p-4, -0x1.06f53c3fa90a8p-4, 0x1.7afbb95743636p-5, -0x1.0de3bdb94b909p-5 },   /* centre 1.406250000 */
  { 0x1.c5709f063f61ep-1, 0x1.8e787a2ac1f9ep-8, 0x1.b3f656a41419p-2, -0x1.026537ebbdf28p-3, 0x1.42e2dcb386c76p-3, -0x1.6dedd3608424ap-4, 0x1.1533b414d61ep-4, -0x1.718dbd05eec3dp-5, 0x1.001e8923ca168p-5, -0x1.5cf9f00355d61p-6 },   /* centre 1.468750000 */
  { 0x1.c678adaa16db2p-1, 0x1.dade0522ce28dp-5, 0x1.9f502b04aa917p-2, -0x1.7061698453edp-4, 0x1.111adaec147d2p-3, -0x1.138a886268f36p-4, 0x1.a17ae1c392bd6p-5, -0x1.0707922c8c43ep-5, 0x1.60077de104a49p-6, -0x1.cb5c7571211ddp-7 },   /* centre 1.531250000 */
  { 0x1.c920953aa5d66p-1, 0x1.b9496a6874861p-4, 0x1.9116c2e5de8d1p-2, -0x1.e346a34c78d6ep-5, 0x1.d74c383999e27p-4, -0x1.9de756ede0cd1p-5, 0x1.3f8131690fd1ap-5, -0x1.7a888b14ec653p-6, 0x1.eb8b5d6cd0026p-7, -0x1.334a276679165p-7 },   /* centre 1.593750000 */
  { 0x1.cd5a098928442p-1, 0x1.3fb8f1d0abf2cp-3, 0x1.8867f0f1d270ap-2, -0x1.0654798b0415fp-5, 0x1.9f3d067f9d585p-4, -0x1.3412ca3fa8323p-5, 0x1.f164f1fbed563p-6, -0x1.12caa9ef9524ep-6, 0x1.5c618b73bc8a7p-7, -0x1.a14084fcb5d75p-8 },   /* centre 1.656250000 */
  { 0x1.d31c4db6ff586p-1, 0x1.a140aba605e6ap-3, 0x1.849a7d111fe3ep-2, -0x1.0665ee5c25becp-7, 0x1.75d82f748db66p-4, -0x1.c1f8dbdb02d95p-6, 0x1.8a422b1844434p-6, -0x1.917ecef6748cdp-7, 0x1.f5201988eff97p-8, -0x1.1f187b1036facp-8 },   /* centre 1.718750000 */
  { 0x1.da6389f09f623p-1, 0x1.0131e5b57cf1ap-2, 0x1.853176b1f3b73p-2, 0x1.c5c9000d6e1cep-7, 0x1.581177702069bp-4, -0x1.3d027a323721fp-6, 0x1.3ec6f9466489fp-6, -0x1.263da5167e69p-7, 0x1.6dbe71bbe090cp-8, -0x1.8fd143df66799p-9 },   /* centre 1.781250000 */
  { 0x1.e3304db941633p-1, 0x1.3217c4a579d7cp-2, 0x1.89d2fbae9b259p-2, 0x1.1807b533d9092p-5, 0x1.43ae1d6087604p-4, -0x1.a0c1119bc32ap-7, 0x1.077745c6a4217p-6, -0x1.aecbc52480399p-8, 0x1.0f00dd6253d47p-8, -0x1.194b848e9be9ep-9 },   /* centre 1.843750000 */
  { 0x1.ed87357995087p-1, 0x1.63cf26c2a3f66p-2, 0x1.924179f06fc8cp-2, 0x1.b66aa0070a6b4p-5, 0x1.370c0b9271813p-4, -0x1.d632bc804cea1p-8, 0x1.be2e3854ef50bp-7, -0x1.391015cd2774dp-8, 0x1.9823259db4371p-9, -0x1.8f35002269e0fp-10 },   /* centre 1.906250000 */
  { 0x1.f970ac84d0a49p-1, 0x1.96cf0f1b4e4dp-2, 0x1.9e56bf5311f1ap-2, 0x1.2815a841a3351p-4, 0x1.30fa25285e4fep-4, -0x1.3d45167c1dcbap-9, 0x1.83e8c0f79bf95p-7, -0x1.bf53a20fad88p-9, 0x1.38d7f76fe48b2p-9, -0x1.1d0e83e2ca312p-10 },   /* centre 1.968750000 */
};

// sin(pi*r) for |r| <= 1/2:  c_k = (-1)^k pi^(2k+1) / (2k+1)!  (Taylor).
// Truncation bound at |r| = 1/2 is 1.81e-23.  Reached only for x < 0.
__device__ const double GFK_TG_SP[13] = {
  0x1.921fb54442d18p+1,   /* 3.1415926535897932385 */
  -0x1.4abbce625be53p+2,   /* -5.1677127800499700292 */
  0x1.466bc6775aae2p+1,   /* 2.5501640398773454439 */
  -0x1.32d2cce62bd86p-1,   /* -0.59926452932079207689 */
  0x1.50783487ee782p-4,   /* 0.082145886611128228799 */
  -0x1.e3074fde8871fp-8,   /* -0.0073704309457143507773 */
  0x1.e8f434d018d63p-12,   /* 0.00046630280576761256442 */
  -0x1.6fadb9f155744p-16,   /* -2.1915353447830215827e-05 */
  0x1.aaec32af93359p-21,   /* 7.9520540014755127848e-07 */
  -0x1.8a404211f9547p-26,   /* -2.294842899726987311e-08 */
  0x1.2877020d52cfp-31,   /* 5.3926646626081284894e-10 */
  -0x1.7215f879e1ac9p-37,   /* -1.0518471716932064455e-11 */
  0x1.859c594ba4573p-43    /* 1.7302192458361107612e-13 */
};

#define GFK_TG_PI   0x1.921fb54442d18p+1
#define GFK_TG_QNAN __uint_as_float(0x7fc00000u)

// Gamma(r) for r in [1,2).  All arithmetic pinned to binary64 add/mul; the
// two subtractions below are exact (Sterbenz), so j and u are exact.
__device__ double gfk_tgamma_poly(double r)
{
    double t = DMUL(DSUB(r, 1.0), 16.0);                 /* [0,16), exact */
    int j = (int)t;
    if (j > 15) j = 15;                                  /* r = nextbelow(2) */
    double u = DSUB(r, DMUL((double)(33 + 2 * j), 0.03125));   /* exact */
    const double *c = GFK_TG_C[j];
    double p = c[9];
    p = DADD(DMUL(p, u), c[8]);
    p = DADD(DMUL(p, u), c[7]);
    p = DADD(DMUL(p, u), c[6]);
    p = DADD(DMUL(p, u), c[5]);
    p = DADD(DMUL(p, u), c[4]);
    p = DADD(DMUL(p, u), c[3]);
    p = DADD(DMUL(p, u), c[2]);
    p = DADD(DMUL(p, u), c[1]);
    p = DADD(DMUL(p, u), c[0]);
    return p;
}

// Gamma(v) in binary64 for 0 < v < 43 (the caller's x < 36 guard bounds the
// reflected argument 1-x below 43, so the recurrence runs at most 41 times).
__device__ double gfk_tgamma_pos(double v)
{
    if (v >= 1.0) {
        int n = (int)v;                                  /* 1 .. 42 */
        double r = DSUB(v, (double)(n - 1));             /* [1,2), exact */
        double p = gfk_tgamma_poly(r);
        for (int i = 1; i < n; i++)
            p = DMUL(p, DSUB(v, (double)i));
        return p;
    }
    return DDIV(gfk_tgamma_poly(DADD(v, 1.0)), v);       /* Gamma(v)=Gamma(v+1)/v */
}

// sin(pi*x) for |x| < 2^23.  k and the subtraction are exact.
__device__ double gfk_tgamma_sinpi(double x)
{
    long long k = (long long)(x < 0.0 ? DSUB(x, 0.5) : DADD(x, 0.5));
    double r  = DSUB(x, (double)k);                      /* [-1/2,1/2], exact */
    double r2 = DMUL(r, r);
    double s  = GFK_TG_SP[12];
    for (int i = 11; i >= 0; i--) s = DADD(DMUL(s, r2), GFK_TG_SP[i]);
    s = DMUL(s, r);
    return (k & 1) ? -s : s;
}

// The correctly-rounded float32 gamma.
//
// The 36.0f overflow guard is arithmetic, not inherited: MEASURED, the last
// float32 with a finite Gamma is 35.0401001, and binary64 holds Gamma out to
// x = 171.6, so ANY threshold in [35.041, 171] gives the same float32 answer
// on every argument.  36 is taken as the smallest integer above the overflow
// point, which is also what bounds the recurrence trip count above at 42.
// glibc picks the same number for the same arithmetic reason; the choice is
// free within a 136-wide interval and carries no expression.
__device__ float gfk_tgamma(float x)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int ax = ix & 0x7fffffffu;

    if (ax >= 0x7f800000u) {                    /* inf / nan */
        if (ix == 0xff800000u) return GFK_TG_QNAN;      /* -inf  -> qNaN */
        return FADD(x, x);                              /* +inf  -> +inf
                                                           NaN   -> same, quieted */
    }
    if (ix < 0x80000000u) {                     /* x >= +0 */
        if (ix == 0u) return FDIV(1.0f, x);             /* +0 -> +inf */
        if (x >= 36.0f)
            return FMUL(FMUL(x, 0x1p127f), 0x1p127f);   /* -> +inf */
        return gfk_d2f_rn(gfk_tgamma_pos((double)x));
    }
    /* x < 0: reflection  Gamma(x) = pi / (sin(pi x) * Gamma(1-x)) */
    if (ix == 0x80000000u) return FDIV(1.0f, x);        /* -0 -> -inf */
    if (ax >= 0x4b000000u) return GFK_TG_QNAN;          /* |x| >= 2^23: every
                                                           float is an integer,
                                                           so every one is a pole */
    {
        double s = gfk_tgamma_sinpi((double)x);
        if (s == 0.0) return GFK_TG_QNAN;               /* negative integer: pole */
        if (x <= -42.0f)                                /* |Gamma| < 2^-150 below
                                                           here; measured, the
                                                           last non-zero float32
                                                           result is at -41.000042 */
            return (s < 0.0) ? -0.0f : 0.0f;
        {
            double g1 = gfk_tgamma_pos(DSUB(1.0, (double)x));   /* 1-x in (1,43) */
            return gfk_d2f_rn(DDIV(GFK_TG_PI, DMUL(s, g1)));
        }
    }
}
