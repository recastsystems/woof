"""RKIND elementary functions with a pinned, defined rounding convention.

The regional byte ladder measured (2026-08-25, evidence
``regional-cpu-l4-20260825``) that instruction-level identity against a
default ``-O3`` ifx reference executable is not source-defined: ifx
fast-math reassociates and contracts at will, and even under the value-safe
reference flags (``-fp-model=source -no-fma -fimf-arch-consistency=true``)
Intel's ``powf`` is not correctly rounded, 4 of 326,810 real regional
exner arguments land 1 ulp off the correctly-rounded value (adjudicated
with 60-digit arithmetic), and ``-fimf-max-error=0.5`` measurably changes
nothing.  numpy/UCRT ``powf`` disagrees with libimf on 33 of the same
arguments.

House law (never bit-exact to a bug): where the native answer is an
implementation accident, the port pins DEFINED behaviour and documents the
divergence.  The defined behaviour here is CORRECT ROUNDING: evaluate in
float64, whose own worst-case libm error is far below half a float32 ulp
on these domains, and round once to float32.  Against the value-safe
reference this leaves a measured residue of at most a few 1-ulp cells per
field per snapshot (the sites where libimf itself is off correct rounding).

For the byte-identity authority proof the hooks below can be rebound to a
ctypes bridge over the exact libimf the reference executable linked
(the proving RTX 5090), driving the residue to zero without changing the shipped
default.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


#: Arrays at least this large are evaluated in contiguous chunks on threads.
#: The functions are elementwise, so every element is computed exactly as in
#: one call; only the wall time changes.  MEASURED (8-card RTX PRO 6000 host, 2026-10-10, 15 km
#: global host preparation): the float64 pow over 2.6 million columns x 55
#: levels took 50 s of a 363 s host preparation on one core.
_PARALLEL_ELEMENTS = 1 << 20
_POOL = None


def _chunked(function: Callable[[np.ndarray], np.ndarray], array: np.ndarray, dtype: Any) -> np.ndarray:
    global _POOL
    flat = np.ascontiguousarray(array).reshape(-1)
    if flat.size < _PARALLEL_ELEMENTS:
        return function(flat).astype(dtype, copy=False).reshape(array.shape)
    import os
    from concurrent.futures import ThreadPoolExecutor
    workers = max(1, min(32, os.cpu_count() or 1))
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rkind-libm")
    out = np.empty(flat.size, dtype=dtype)
    bounds = np.linspace(0, flat.size, workers + 1, dtype=np.int64)

    def run(index: int) -> None:
        lo, hi = int(bounds[index]), int(bounds[index + 1])
        out[lo:hi] = function(flat[lo:hi])

    list(_POOL.map(run, range(workers)))
    return out.reshape(array.shape)


def _powf_correctly_rounded(base: FloatArray, exponent: float) -> FloatArray:
    array = np.asarray(base)
    if array.dtype == np.float64:
        power = np.float64(exponent)
        return _chunked(lambda part: part ** power, array, np.float64)
    power = np.float64(np.float32(exponent))
    return _chunked(lambda part: part.astype(np.float64) ** power, array, array.dtype)


def _sinf_correctly_rounded(phase: FloatArray) -> FloatArray:
    array = np.asarray(phase)
    if array.dtype == np.float64:
        return _chunked(np.sin, array, np.float64)
    return _chunked(lambda part: np.sin(part.astype(np.float64)), array, array.dtype)


#: Rebindable hooks.  Default: correctly rounded via float64 (the pinned
#: defined behaviour).  The proving RTX 5090 authority ladder may rebind these to the
#: reference executable's own libimf via ctypes to prove full byte identity.
powf_rkind: Callable[[FloatArray, float], FloatArray] = _powf_correctly_rounded
sinf_rkind: Callable[[FloatArray], FloatArray] = _sinf_correctly_rounded


__all__ = [
    "powf_rkind",
    "sinf_rkind",
]



def wsm6_device_math_header() -> str:
    """Return the WSM6 composition of existing shared device operators.

    Retained p2-wsm6 qualification
    records whole WSM6 and forecast results separately from broad math ABI
    differences; this composition is not a universal glibc identity claim.
    Build with direct NVRTC and --fmad=false --ftz=false.
    """
    return r"""// ======================================================================
// THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
// third-party work.  WOOF distributes the file under the Apache License
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
//   through WOOF 2.6.5; both were deleted at 2.6.6 with the gamma block
//   that was their only caller, so the Sun notice that had to travel with
//   them travels no longer -- there is nothing here for it to attach to.
//   The notice is unchanged and still reproduced, for the twelve files
//   that DO carry FDLIBM code -- nine .cu in this directory and three
//   Python modules under gpuwm/core/ -- in the repository NOTICE, in
//   licenses/LICENSE-FDLIBM-SunPro.txt and in
//   gpuwm/core/kernels/LICENSE-third-party.txt.
//
//   Nothing else in this file is a transcription of anything.  gfk_d2f_rn
//   is WOOF's own subnormal-rounding countermeasure and gfk_tgamma is
//   WOOF's own correctly rounded gamma (see its block below and
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
// (the project rules: __fmaf_rn/__fmul_rn/__fadd_rn are NVIDIA-guaranteed never
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
// gone, and WOOF's own gamma that replaced it evaluates no logarithm, no
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
// gfk_tgamma -- WOOF's own float32 gamma.  ORIGINAL WORK, Apache-2.0.
// ==========================================================================
//
// WHAT THIS REPLACED.  Until 2.6.5 this file ended with WOOF's earlier
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
// The divergence is NOT new to WOOF and it is NOT unbounded.  MEASURED, this
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

// ======================================================================
// THIRD-PARTY NOTICE. Original CUDA adaptation is distributed under
// Apache-2.0. The MIT grants below cover the transcribed upstream parts.
// Arm optimized-routines exp/log/pow and data are taken
// under MIT from upstream MIT sources, not from glibc source text.
// Copyright (c) 2017-2026, Arm Limited.
// SPDX-License-Identifier: MIT
// CORE-MATH cos/acos retain their MIT notices below.
// uw_cos and uw_acos target correctly rounded binary64 round-to-nearest,
// ties-to-even results. They are NOT glibc implementations.
// PROVENANCE: see tools/uwpbl_wrf471_oracle/libm64/upstream and generate.py.
// Fused sites derive from the disassembly of the oracle host's installed
// libm (Ubuntu glibc 2.43, the x86-64 variant ifunc selects with FMA+AVX2);
// tools/uwpbl_wrf471_oracle/libm64/FMA-SITES.md maps every one.
// MEASURED (tools/uwpbl_wrf471_oracle/libm64/README.md): exp, log and pow
// equal glibc's words on 2^34 CPU and 2^28 GPU arguments each, zero
// mismatches.  cos and acos are CORE-MATH's correctly rounded functions and
// equal them on 2^28 GPU arguments; glibc's own binary64 cos/acos (LGPL IBM
// code, not transcribed) are not correctly rounded and differ from these on
// 0.150 % of cos arguments in [-2.1, 3.15] and 0.070 % of acos arguments in
// [-1, 1], always by 1 ULP.
// ======================================================================
#pragma once
#if !defined(__CUDACC_RTC__)
#include <stdint.h>
#else
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef unsigned int uint32_t;
typedef int int32_t;
#endif
__device__ __forceinline__ double g64_inf(){return __longlong_as_double(0x7ff0000000000000ULL);}
__device__ __forceinline__ int g64_clzll(unsigned long long x){return __clzll(x);}
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
namespace g64_exp {

__device__ __forceinline__ uint64_t asuint64(double x) {union {double f; uint64_t u;} v={x};return v.u;}
__device__ __forceinline__ double asdouble(uint64_t x) {union {uint64_t u; double f;} v={x};return v.f;}
__device__ __forceinline__ int issignaling_inline(double x){uint64_t u=asuint64(x);return ((u&0x7ff8000000000000ULL)==0x7ff0000000000000ULL)&&(u&0xfffffffffffffULL);}
__device__ __forceinline__ double __math_oflow(uint32_t s){return s ? -(g64_inf()):(g64_inf());}
__device__ __forceinline__ double __math_uflow(uint32_t s){return asdouble((uint64_t)(!!s)<<63);}
__device__ __forceinline__ double __math_divzero(uint32_t s){return s ? -(g64_inf()):(g64_inf());}
__device__ __forceinline__ double __math_invalid(double x){uint64_t u=asuint64(x);return asdouble((u&0x7fffffffffffffffULL)>0x7ff0000000000000ULL ? u|0x8000000000000ULL : 0xfff8000000000000ULL);}
struct exp_data {
  double invln2N;
  double negln2hiN;
  double negln2loN;
  double poly[4];
  double shift;
  double exp2_shift;
  double exp2_poly[5];
  double neglog10_2hiN;
  double neglog10_2loN;
  double exp10_poly[5];
  uint64_t tab[2*(1 << 7)];
  double invlog10_2N;
};
struct log_data {
  double ln2hi;
  double ln2lo;
  double poly[6 - 1];
  double poly1[12 - 1];
  struct {double invc, logc;} tab[1 << 7];
};
struct log2_data {
  double invln2hi;
  double invln2lo;
  double poly[7 - 1];
  double poly1[11 - 1];
  struct {double invc, logc;} tab[1 << 6];
};
struct pow_log_data {
  double ln2hi;
  double ln2lo;
  double poly[8 - 1];
  struct {double invc, pad, logc, logctail;} tab[1 << 7];
};
__device__ const struct exp_data __exp_data = {
0x1.71547652b82fep0 * (1 << 7),
-0x1.62e42fefa0000p-8,
-0x1.cf79abc9e3b3ap-47,
{
0x1.ffffffffffdbdp-2,
0x1.555555555543cp-3,
0x1.55555cf172b91p-5,
0x1.1111167a4d017p-7,
},
0x1.8p52,
0x1.8p52 / (1 << 7),
{
0x1.62e42fefa39efp-1,
0x1.ebfbdff82c424p-3,
0x1.c6b08d70cf4b5p-5,
0x1.3b2abd24650ccp-7,
0x1.5d7e09b4e3a84p-10,
},
-0x1.3441350ap-2 / (1 << 7),
0x1.0c0219dc1da99p-39 / (1 << 7),
{
0x1.26bb1bbb55516p1,
0x1.53524c73ce9fep1,
0x1.0470591ce4b26p1,
0x1.2bd76577fe684p0,
0x1.1446eeccd0efbp-1
},
{
0x0, 0x3ff0000000000000,
0x3c9b3b4f1a88bf6e, 0x3feff63da9fb3335,
0xbc7160139cd8dc5d, 0x3fefec9a3e778061,
0xbc905e7a108766d1, 0x3fefe315e86e7f85,
0x3c8cd2523567f613, 0x3fefd9b0d3158574,
0xbc8bce8023f98efa, 0x3fefd06b29ddf6de,
0x3c60f74e61e6c861, 0x3fefc74518759bc8,
0x3c90a3e45b33d399, 0x3fefbe3ecac6f383,
0x3c979aa65d837b6d, 0x3fefb5586cf9890f,
0x3c8eb51a92fdeffc, 0x3fefac922b7247f7,
0x3c3ebe3d702f9cd1, 0x3fefa3ec32d3d1a2,
0xbc6a033489906e0b, 0x3fef9b66affed31b,
0xbc9556522a2fbd0e, 0x3fef9301d0125b51,
0xbc5080ef8c4eea55, 0x3fef8abdc06c31cc,
0xbc91c923b9d5f416, 0x3fef829aaea92de0,
0x3c80d3e3e95c55af, 0x3fef7a98c8a58e51,
0xbc801b15eaa59348, 0x3fef72b83c7d517b,
0xbc8f1ff055de323d, 0x3fef6af9388c8dea,
0x3c8b898c3f1353bf, 0x3fef635beb6fcb75,
0xbc96d99c7611eb26, 0x3fef5be084045cd4,
0x3c9aecf73e3a2f60, 0x3fef54873168b9aa,
0xbc8fe782cb86389d, 0x3fef4d5022fcd91d,
0x3c8a6f4144a6c38d, 0x3fef463b88628cd6,
0x3c807a05b0e4047d, 0x3fef3f49917ddc96,
0x3c968efde3a8a894, 0x3fef387a6e756238,
0x3c875e18f274487d, 0x3fef31ce4fb2a63f,
0x3c80472b981fe7f2, 0x3fef2b4565e27cdd,
0xbc96b87b3f71085e, 0x3fef24dfe1f56381,
0x3c82f7e16d09ab31, 0x3fef1e9df51fdee1,
0xbc3d219b1a6fbffa, 0x3fef187fd0dad990,
0x3c8b3782720c0ab4, 0x3fef1285a6e4030b,
0x3c6e149289cecb8f, 0x3fef0cafa93e2f56,
0x3c834d754db0abb6, 0x3fef06fe0a31b715,
0x3c864201e2ac744c, 0x3fef0170fc4cd831,
0x3c8fdd395dd3f84a, 0x3feefc08b26416ff,
0xbc86a3803b8e5b04, 0x3feef6c55f929ff1,
0xbc924aedcc4b5068, 0x3feef1a7373aa9cb,
0xbc9907f81b512d8e, 0x3feeecae6d05d866,
0xbc71d1e83e9436d2, 0x3feee7db34e59ff7,
0xbc991919b3ce1b15, 0x3feee32dc313a8e5,
0x3c859f48a72a4c6d, 0x3feedea64c123422,
0xbc9312607a28698a, 0x3feeda4504ac801c,
0xbc58a78f4817895b, 0x3feed60a21f72e2a,
0xbc7c2c9b67499a1b, 0x3feed1f5d950a897,
0x3c4363ed60c2ac11, 0x3feece086061892d,
0x3c9666093b0664ef, 0x3feeca41ed1d0057,
0x3c6ecce1daa10379, 0x3feec6a2b5c13cd0,
0x3c93ff8e3f0f1230, 0x3feec32af0d7d3de,
0x3c7690cebb7aafb0, 0x3feebfdad5362a27,
0x3c931dbdeb54e077, 0x3feebcb299fddd0d,
0xbc8f94340071a38e, 0x3feeb9b2769d2ca7,
0xbc87deccdc93a349, 0x3feeb6daa2cf6642,
0xbc78dec6bd0f385f, 0x3feeb42b569d4f82,
0xbc861246ec7b5cf6, 0x3feeb1a4ca5d920f,
0x3c93350518fdd78e, 0x3feeaf4736b527da,
0x3c7b98b72f8a9b05, 0x3feead12d497c7fd,
0x3c9063e1e21c5409, 0x3feeab07dd485429,
0x3c34c7855019c6ea, 0x3feea9268a5946b7,
0x3c9432e62b64c035, 0x3feea76f15ad2148,
0xbc8ce44a6199769f, 0x3feea5e1b976dc09,
0xbc8c33c53bef4da8, 0x3feea47eb03a5585,
0xbc845378892be9ae, 0x3feea34634ccc320,
0xbc93cedd78565858, 0x3feea23882552225,
0x3c5710aa807e1964, 0x3feea155d44ca973,
0xbc93b3efbf5e2228, 0x3feea09e667f3bcd,
0xbc6a12ad8734b982, 0x3feea012750bdabf,
0xbc6367efb86da9ee, 0x3fee9fb23c651a2f,
0xbc80dc3d54e08851, 0x3fee9f7df9519484,
0xbc781f647e5a3ecf, 0x3fee9f75e8ec5f74,
0xbc86ee4ac08b7db0, 0x3fee9f9a48a58174,
0xbc8619321e55e68a, 0x3fee9feb564267c9,
0x3c909ccb5e09d4d3, 0x3feea0694fde5d3f,
0xbc7b32dcb94da51d, 0x3feea11473eb0187,
0x3c94ecfd5467c06b, 0x3feea1ed0130c132,
0x3c65ebe1abd66c55, 0x3feea2f336cf4e62,
0xbc88a1c52fb3cf42, 0x3feea427543e1a12,
0xbc9369b6f13b3734, 0x3feea589994cce13,
0xbc805e843a19ff1e, 0x3feea71a4623c7ad,
0xbc94d450d872576e, 0x3feea8d99b4492ed,
0x3c90ad675b0e8a00, 0x3feeaac7d98a6699,
0x3c8db72fc1f0eab4, 0x3feeace5422aa0db,
0xbc65b6609cc5e7ff, 0x3feeaf3216b5448c,
0x3c7bf68359f35f44, 0x3feeb1ae99157736,
0xbc93091fa71e3d83, 0x3feeb45b0b91ffc6,
0xbc5da9b88b6c1e29, 0x3feeb737b0cdc5e5,
0xbc6c23f97c90b959, 0x3feeba44cbc8520f,
0xbc92434322f4f9aa, 0x3feebd829fde4e50,
0xbc85ca6cd7668e4b, 0x3feec0f170ca07ba,
0x3c71affc2b91ce27, 0x3feec49182a3f090,
0x3c6dd235e10a73bb, 0x3feec86319e32323,
0xbc87c50422622263, 0x3feecc667b5de565,
0x3c8b1c86e3e231d5, 0x3feed09bec4a2d33,
0xbc91bbd1d3bcbb15, 0x3feed503b23e255d,
0x3c90cc319cee31d2, 0x3feed99e1330b358,
0x3c8469846e735ab3, 0x3feede6b5579fdbf,
0xbc82dfcd978e9db4, 0x3feee36bbfd3f37a,
0x3c8c1a7792cb3387, 0x3feee89f995ad3ad,
0xbc907b8f4ad1d9fa, 0x3feeee07298db666,
0xbc55c3d956dcaeba, 0x3feef3a2b84f15fb,
0xbc90a40e3da6f640, 0x3feef9728de5593a,
0xbc68d6f438ad9334, 0x3feeff76f2fb5e47,
0xbc91eee26b588a35, 0x3fef05b030a1064a,
0x3c74ffd70a5fddcd, 0x3fef0c1e904bc1d2,
0xbc91bdfbfa9298ac, 0x3fef12c25bd71e09,
0x3c736eae30af0cb3, 0x3fef199bdd85529c,
0x3c8ee3325c9ffd94, 0x3fef20ab5fffd07a,
0x3c84e08fd10959ac, 0x3fef27f12e57d14b,
0x3c63cdaf384e1a67, 0x3fef2f6d9406e7b5,
0x3c676b2c6c921968, 0x3fef3720dcef9069,
0xbc808a1883ccb5d2, 0x3fef3f0b555dc3fa,
0xbc8fad5d3ffffa6f, 0x3fef472d4a07897c,
0xbc900dae3875a949, 0x3fef4f87080d89f2,
0x3c74a385a63d07a7, 0x3fef5818dcfba487,
0xbc82919e2040220f, 0x3fef60e316c98398,
0x3c8e5a50d5c192ac, 0x3fef69e603db3285,
0x3c843a59ac016b4b, 0x3fef7321f301b460,
0xbc82d52107b43e1f, 0x3fef7c97337b9b5f,
0xbc892ab93b470dc9, 0x3fef864614f5a129,
0x3c74b604603a88d3, 0x3fef902ee78b3ff6,
0x3c83c5ec519d7271, 0x3fef9a51fbc74c83,
0xbc8ff7128fd391f0, 0x3fefa4afa2a490da,
0xbc8dae98e223747d, 0x3fefaf482d8e67f1,
0x3c8ec3bc41aa2008, 0x3fefba1bee615a27,
0x3c842b94c3a9eb32, 0x3fefc52b376bba97,
0x3c8a64a931d185ee, 0x3fefd0765b6e4540,
0xbc8e37bae43be3ed, 0x3fefdbfdad9cbe14,
0x3c77893b4d91cd9d, 0x3fefe7c1819e90d8,
0x3c5305c14160cc89, 0x3feff3c22b8f71f1,
},
0x1.a934f0979a371p1 * (1 << 7)
};
__device__ __forceinline__ double
specialcase (double tmp, uint64_t sbits, uint64_t ki)
{
  double scale, y;
  if ((ki & 0x80000000) == 0)
    {
      sbits -= 1009ull << 52;
      scale = asdouble (sbits);
      y = __dmul_rn(0x1p1009,__fma_rn(scale,tmp,scale));
      return ((y));
    }
  sbits += 1022ull << 52;
  scale = asdouble (sbits);
  y = __dadd_rn(scale,__dmul_rn(scale,tmp));
  if (y < 1.0)
    {
      double hi, lo;
      lo = __dadd_rn(__dsub_rn(scale,y),__dmul_rn(scale,tmp));
      hi = __dadd_rn(1.0,y);
      lo = __dadd_rn(__dadd_rn(__dsub_rn(1.0,hi),y),lo);
      y = __dsub_rn((__dadd_rn(hi,lo)),1.0);
      if (1 && y == 0.0)
 y = 0.0;
      ((void)(__dmul_rn((0x1p-1022),0x1p-1022)));
    }
  y = __dmul_rn(0x1p-1022,y);
  return ((y));
}
__device__ __forceinline__ uint32_t
top12 (double x)
{
  return asuint64 (x) >> 52;
}
__device__ __forceinline__ double
exp_inline (double x, double xtail)
{
  uint32_t abstop;
  uint64_t ki, idx, top, sbits;
  double kd, z, r, r2, scale, tail, tmp;
  abstop = top12 (x) & 0x7ff;
  if ((abstop - top12 (0x1p-54) >= top12 (512.0) - top12 (0x1p-54)))
    {
      if (abstop - top12 (0x1p-54) >= 0x80000000)
 return 1 ? __dadd_rn(1.0,x) : 1.0;
      if (abstop >= top12 (1024.0))
 {
   if (asuint64 (x) == asuint64 (-(g64_inf())))
     return 0.0;
   if (abstop >= top12 ((g64_inf())))
     return __dadd_rn(1.0,x);
   if (asuint64 (x) >> 63)
     return __math_uflow (0);
   else
     return __math_oflow (0);
 }
      abstop = 0;
    }
  z = __dmul_rn(__exp_data.invln2N,x);
  kd = __fma_rn(__exp_data.invln2N, x, __exp_data.shift);
  ki = asuint64 (kd);
  kd = __dsub_rn(kd,__exp_data.shift);
  r = __fma_rn(kd, __exp_data.negln2loN, __fma_rn(kd, __exp_data.negln2hiN, x));
  if (xtail != 0.0)
    r = __dadd_rn(r,xtail);
  idx = 2 * (ki % (1 << 7));
  top = ki << (52 - 7);
  tail = asdouble (__exp_data.tab[idx]);
  sbits = __exp_data.tab[idx + 1] + top;
  r2 = __dmul_rn(r,r);
  tmp = __fma_rn(__dmul_rn(r2,r2), __fma_rn(r,__exp_data.poly[8 - 5],__exp_data.poly[7 - 5]), __fma_rn(r2,__fma_rn(r,__exp_data.poly[6 - 5],__exp_data.poly[5 - 5]),__dadd_rn(tail,r)));
  if ((abstop == 0))
    return specialcase (tmp, sbits, ki);
  scale = asdouble (sbits);
  return __fma_rn(scale,tmp,scale);
}
__device__ __forceinline__ double exp (double x)
{
  return exp_inline (x, 0);
}

}

namespace g64_log {

__device__ __forceinline__ uint64_t asuint64(double x) {union {double f; uint64_t u;} v={x};return v.u;}
__device__ __forceinline__ double asdouble(uint64_t x) {union {uint64_t u; double f;} v={x};return v.f;}
__device__ __forceinline__ int issignaling_inline(double x){uint64_t u=asuint64(x);return ((u&0x7ff8000000000000ULL)==0x7ff0000000000000ULL)&&(u&0xfffffffffffffULL);}
__device__ __forceinline__ double __math_oflow(uint32_t s){return s ? -(g64_inf()):(g64_inf());}
__device__ __forceinline__ double __math_uflow(uint32_t s){return asdouble((uint64_t)(!!s)<<63);}
__device__ __forceinline__ double __math_divzero(uint32_t s){return s ? -(g64_inf()):(g64_inf());}
__device__ __forceinline__ double __math_invalid(double x){uint64_t u=asuint64(x);return asdouble((u&0x7fffffffffffffffULL)>0x7ff0000000000000ULL ? u|0x8000000000000ULL : 0xfff8000000000000ULL);}
struct exp_data {
  double invln2N;
  double negln2hiN;
  double negln2loN;
  double poly[4];
  double shift;
  double exp2_shift;
  double exp2_poly[5];
  double neglog10_2hiN;
  double neglog10_2loN;
  double exp10_poly[5];
  uint64_t tab[2*(1 << 7)];
  double invlog10_2N;
};
struct log_data {
  double ln2hi;
  double ln2lo;
  double poly[6 - 1];
  double poly1[12 - 1];
  struct {double invc, logc;} tab[1 << 7];
};
struct log2_data {
  double invln2hi;
  double invln2lo;
  double poly[7 - 1];
  double poly1[11 - 1];
  struct {double invc, logc;} tab[1 << 6];
};
struct pow_log_data {
  double ln2hi;
  double ln2lo;
  double poly[8 - 1];
  struct {double invc, pad, logc, logctail;} tab[1 << 7];
};
__device__ const struct log_data __log_data = {
0x1.62e42fefa3800p-1,
0x1.ef35793c76730p-45,
{
-0x1.0000000000001p-1,
0x1.555555551305bp-2,
-0x1.fffffffeb459p-3,
0x1.999b324f10111p-3,
-0x1.55575e506c89fp-3,
},
{
-0x1p-1,
0x1.5555555555577p-2,
-0x1.ffffffffffdcbp-3,
0x1.999999995dd0cp-3,
-0x1.55555556745a7p-3,
0x1.24924a344de3p-3,
-0x1.fffffa4423d65p-4,
0x1.c7184282ad6cap-4,
-0x1.999eb43b068ffp-4,
0x1.78182f7afd085p-4,
-0x1.5521375d145cdp-4,
},
{
{0x1.734f0c3e0de9fp+0, -0x1.7cc7f79e69000p-2},
{0x1.713786a2ce91fp+0, -0x1.76feec20d0000p-2},
{0x1.6f26008fab5a0p+0, -0x1.713e31351e000p-2},
{0x1.6d1a61f138c7dp+0, -0x1.6b85b38287800p-2},
{0x1.6b1490bc5b4d1p+0, -0x1.65d5590807800p-2},
{0x1.69147332f0cbap+0, -0x1.602d076180000p-2},
{0x1.6719f18224223p+0, -0x1.5a8ca86909000p-2},
{0x1.6524f99a51ed9p+0, -0x1.54f4356035000p-2},
{0x1.63356aa8f24c4p+0, -0x1.4f637c36b4000p-2},
{0x1.614b36b9ddc14p+0, -0x1.49da7fda85000p-2},
{0x1.5f66452c65c4cp+0, -0x1.445923989a800p-2},
{0x1.5d867b5912c4fp+0, -0x1.3edf439b0b800p-2},
{0x1.5babccb5b90dep+0, -0x1.396ce448f7000p-2},
{0x1.59d61f2d91a78p+0, -0x1.3401e17bda000p-2},
{0x1.5805612465687p+0, -0x1.2e9e2ef468000p-2},
{0x1.56397cee76bd3p+0, -0x1.2941b3830e000p-2},
{0x1.54725e2a77f93p+0, -0x1.23ec58cda8800p-2},
{0x1.52aff42064583p+0, -0x1.1e9e129279000p-2},
{0x1.50f22dbb2bddfp+0, -0x1.1956d2b48f800p-2},
{0x1.4f38f4734ded7p+0, -0x1.141679ab9f800p-2},
{0x1.4d843cfde2840p+0, -0x1.0edd094ef9800p-2},
{0x1.4bd3ec078a3c8p+0, -0x1.09aa518db1000p-2},
{0x1.4a27fc3e0258ap+0, -0x1.047e65263b800p-2},
{0x1.4880524d48434p+0, -0x1.feb224586f000p-3},
{0x1.46dce1b192d0bp+0, -0x1.f474a7517b000p-3},
{0x1.453d9d3391854p+0, -0x1.ea4443d103000p-3},
{0x1.43a2744b4845ap+0, -0x1.e020d44e9b000p-3},
{0x1.420b54115f8fbp+0, -0x1.d60a22977f000p-3},
{0x1.40782da3ef4b1p+0, -0x1.cc00104959000p-3},
{0x1.3ee8f5d57fe8fp+0, -0x1.c202956891000p-3},
{0x1.3d5d9a00b4ce9p+0, -0x1.b81178d811000p-3},
{0x1.3bd60c010c12bp+0, -0x1.ae2c9ccd3d000p-3},
{0x1.3a5242b75dab8p+0, -0x1.a45402e129000p-3},
{0x1.38d22cd9fd002p+0, -0x1.9a877681df000p-3},
{0x1.3755bc5847a1cp+0, -0x1.90c6d69483000p-3},
{0x1.35dce49ad36e2p+0, -0x1.87120a645c000p-3},
{0x1.34679984dd440p+0, -0x1.7d68fb4143000p-3},
{0x1.32f5cceffcb24p+0, -0x1.73cb83c627000p-3},
{0x1.3187775a10d49p+0, -0x1.6a39a9b376000p-3},
{0x1.301c8373e3990p+0, -0x1.60b3154b7a000p-3},
{0x1.2eb4ebb95f841p+0, -0x1.5737d76243000p-3},
{0x1.2d50a0219a9d1p+0, -0x1.4dc7b8fc23000p-3},
{0x1.2bef9a8b7fd2ap+0, -0x1.4462c51d20000p-3},
{0x1.2a91c7a0c1babp+0, -0x1.3b08abc830000p-3},
{0x1.293726014b530p+0, -0x1.31b996b490000p-3},
{0x1.27dfa5757a1f5p+0, -0x1.2875490a44000p-3},
{0x1.268b39b1d3bbfp+0, -0x1.1f3b9f879a000p-3},
{0x1.2539d838ff5bdp+0, -0x1.160c8252ca000p-3},
{0x1.23eb7aac9083bp+0, -0x1.0ce7f57f72000p-3},
{0x1.22a012ba940b6p+0, -0x1.03cdc49fea000p-3},
{0x1.2157996cc4132p+0, -0x1.f57bdbc4b8000p-4},
{0x1.201201dd2fc9bp+0, -0x1.e370896404000p-4},
{0x1.1ecf4494d480bp+0, -0x1.d17983ef94000p-4},
{0x1.1d8f5528f6569p+0, -0x1.bf9674ed8a000p-4},
{0x1.1c52311577e7cp+0, -0x1.adc79202f6000p-4},
{0x1.1b17c74cb26e9p+0, -0x1.9c0c3e7288000p-4},
{0x1.19e010c2c1ab6p+0, -0x1.8a646b372c000p-4},
{0x1.18ab07bb670bdp+0, -0x1.78d01b3ac0000p-4},
{0x1.1778a25efbcb6p+0, -0x1.674f145380000p-4},
{0x1.1648d354c31dap+0, -0x1.55e0e6d878000p-4},
{0x1.151b990275fddp+0, -0x1.4485cdea1e000p-4},
{0x1.13f0ea432d24cp+0, -0x1.333d94d6aa000p-4},
{0x1.12c8b7210f9dap+0, -0x1.22079f8c56000p-4},
{0x1.11a3028ecb531p+0, -0x1.10e4698622000p-4},
{0x1.107fbda8434afp+0, -0x1.ffa6c6ad20000p-5},
{0x1.0f5ee0f4e6bb3p+0, -0x1.dda8d4a774000p-5},
{0x1.0e4065d2a9fcep+0, -0x1.bbcece4850000p-5},
{0x1.0d244632ca521p+0, -0x1.9a1894012c000p-5},
{0x1.0c0a77ce2981ap+0, -0x1.788583302c000p-5},
{0x1.0af2f83c636d1p+0, -0x1.5715e67d68000p-5},
{0x1.09ddb98a01339p+0, -0x1.35c8a49658000p-5},
{0x1.08cabaf52e7dfp+0, -0x1.149e364154000p-5},
{0x1.07b9f2f4e28fbp+0, -0x1.e72c082eb8000p-6},
{0x1.06ab58c358f19p+0, -0x1.a55f152528000p-6},
{0x1.059eea5ecf92cp+0, -0x1.63d62cf818000p-6},
{0x1.04949cdd12c90p+0, -0x1.228fb8caa0000p-6},
{0x1.038c6c6f0ada9p+0, -0x1.c317b20f90000p-7},
{0x1.02865137932a9p+0, -0x1.419355daa0000p-7},
{0x1.0182427ea7348p+0, -0x1.81203c2ec0000p-8},
{0x1.008040614b195p+0, -0x1.0040979240000p-9},
{0x1.fe01ff726fa1ap-1, 0x1.feff384900000p-9},
{0x1.fa11cc261ea74p-1, 0x1.7dc41353d0000p-7},
{0x1.f6310b081992ep-1, 0x1.3cea3c4c28000p-6},
{0x1.f25f63ceeadcdp-1, 0x1.b9fc114890000p-6},
{0x1.ee9c8039113e7p-1, 0x1.1b0d8ce110000p-5},
{0x1.eae8078cbb1abp-1, 0x1.58a5bd001c000p-5},
{0x1.e741aa29d0c9bp-1, 0x1.95c8340d88000p-5},
{0x1.e3a91830a99b5p-1, 0x1.d276aef578000p-5},
{0x1.e01e009609a56p-1, 0x1.07598e598c000p-4},
{0x1.dca01e577bb98p-1, 0x1.253f5e30d2000p-4},
{0x1.d92f20b7c9103p-1, 0x1.42edd8b380000p-4},
{0x1.d5cac66fb5ccep-1, 0x1.606598757c000p-4},
{0x1.d272caa5ede9dp-1, 0x1.7da76356a0000p-4},
{0x1.cf26e3e6b2ccdp-1, 0x1.9ab434e1c6000p-4},
{0x1.cbe6da2a77902p-1, 0x1.b78c7bb0d6000p-4},
{0x1.c8b266d37086dp-1, 0x1.d431332e72000p-4},
{0x1.c5894bd5d5804p-1, 0x1.f0a3171de6000p-4},
{0x1.c26b533bb9f8cp-1, 0x1.067152b914000p-3},
{0x1.bf583eeece73fp-1, 0x1.147858292b000p-3},
{0x1.bc4fd75db96c1p-1, 0x1.2266ecdca3000p-3},
{0x1.b951e0c864a28p-1, 0x1.303d7a6c55000p-3},
{0x1.b65e2c5ef3e2cp-1, 0x1.3dfc33c331000p-3},
{0x1.b374867c9888bp-1, 0x1.4ba366b7a8000p-3},
{0x1.b094b211d304ap-1, 0x1.5933928d1f000p-3},
{0x1.adbe885f2ef7ep-1, 0x1.66acd2418f000p-3},
{0x1.aaf1d31603da2p-1, 0x1.740f8ec669000p-3},
{0x1.a82e63fd358a7p-1, 0x1.815c0f51af000p-3},
{0x1.a5740ef09738bp-1, 0x1.8e92954f68000p-3},
{0x1.a2c2a90ab4b27p-1, 0x1.9bb3602f84000p-3},
{0x1.a01a01393f2d1p-1, 0x1.a8bed1c2c0000p-3},
{0x1.9d79f24db3c1bp-1, 0x1.b5b515c01d000p-3},
{0x1.9ae2505c7b190p-1, 0x1.c2967ccbcc000p-3},
{0x1.9852ef297ce2fp-1, 0x1.cf635d5486000p-3},
{0x1.95cbaeea44b75p-1, 0x1.dc1bd3446c000p-3},
{0x1.934c69de74838p-1, 0x1.e8c01b8cfe000p-3},
{0x1.90d4f2f6752e6p-1, 0x1.f5509c0179000p-3},
{0x1.8e6528effd79dp-1, 0x1.00e6c121fb800p-2},
{0x1.8bfce9fcc007cp-1, 0x1.071b80e93d000p-2},
{0x1.899c0dabec30ep-1, 0x1.0d46b9e867000p-2},
{0x1.87427aa2317fbp-1, 0x1.13687334bd000p-2},
{0x1.84f00acb39a08p-1, 0x1.1980d67234800p-2},
{0x1.82a49e8653e55p-1, 0x1.1f8ffe0cc8000p-2},
{0x1.8060195f40260p-1, 0x1.2595fd7636800p-2},
{0x1.7e22563e0a329p-1, 0x1.2b9300914a800p-2},
{0x1.7beb377dcb5adp-1, 0x1.3187210436000p-2},
{0x1.79baa679725c2p-1, 0x1.377266dec1800p-2},
{0x1.77907f2170657p-1, 0x1.3d54ffbaf3000p-2},
{0x1.756cadbd6130cp-1, 0x1.432eee32fe000p-2},
}
};
__device__ __forceinline__ uint32_t
top16 (double x)
{
  return asuint64 (x) >> 48;
}
__device__ __forceinline__ double log (double x)
{
  double w, z, r, r2, r3, y, invc, logc, kd, hi, lo;
  uint64_t ix, iz, tmp;
  uint32_t top;
  int k, i;
  ix = asuint64 (x);
  top = top16 (x);
  if ((ix - asuint64 (__dsub_rn(1.0,0x1p-4)) < asuint64 (__dadd_rn(1.0,0x1.09p-4)) - asuint64 (__dsub_rn(1.0,0x1p-4))))
    {
      if (1 && (ix == asuint64 (1.0)))
 return 0;
      r = __dsub_rn(x,1.0);
      r2 = __dmul_rn(r,r);
      r3 = __dmul_rn(r,r2);
      y = __fma_rn(r3,__fma_rn(r3,__log_data.poly1[10],__fma_rn(r2,__log_data.poly1[9],__fma_rn(r,__log_data.poly1[8],__log_data.poly1[7]))),__fma_rn(r2,__log_data.poly1[6],__fma_rn(r,__log_data.poly1[5],__log_data.poly1[4]))); y = __fma_rn(r3,y,__fma_rn(r2,__log_data.poly1[3],__fma_rn(r,__log_data.poly1[2],__log_data.poly1[1])));
      w = __dmul_rn(r,0x1p27);
      double rhi = __fma_rn(-r,0x1p27,__fma_rn(r,0x1p27,r));
      double rlo = __dsub_rn(r,rhi);
      w = __dmul_rn(__dmul_rn(rhi,rhi),__log_data.poly1[0]);
      hi = __fma_rn(__dmul_rn(rhi,rhi),__log_data.poly1[0],r);
      lo = __fma_rn(__dmul_rn(rhi,rhi),__log_data.poly1[0],__dsub_rn(r,hi));
      lo = __fma_rn(__dmul_rn(__log_data.poly1[0],rlo),__dadd_rn(rhi,r),lo);
      y = __fma_rn(r3,y,lo);
      y = __dadd_rn(y,hi);
      return (y);
    }
  if ((top - 0x0010 >= 0x7ff0 - 0x0010))
    {
      if (ix * 2 == 0)
 return __math_divzero (1);
      if (ix == asuint64 ((g64_inf())))
 return x;
      if ((top & 0x8000) || (top & 0x7ff0) == 0x7ff0)
 return __math_invalid (x);
      ix = asuint64 (__dmul_rn(x,0x1p52));
      ix -= 52ULL << 52;
    }
  tmp = ix - 0x3fe6000000000000;
  i = (tmp >> (52 - 7)) % (1 << 7);
  k = (int64_t) tmp >> 52;
  iz = ix - (tmp & 0xfffULL << 52);
  invc = __log_data.tab[i].invc;
  logc = __log_data.tab[i].logc;
  z = asdouble (iz);
  r = __fma_rn(z, invc, -1.0);
  kd = (double) k;
  w = __fma_rn(kd,__log_data.ln2hi,logc);
  hi = __dadd_rn(w,r);
  lo = __fma_rn(kd,__log_data.ln2lo,__dadd_rn(__dsub_rn(w,hi),r));
  r2 = __dmul_rn(r,r);
  y = __dadd_rn(__fma_rn(__dmul_rn(r,r2),__fma_rn(r2,__fma_rn(r,__log_data.poly[4],__log_data.poly[3]),__fma_rn(r,__log_data.poly[2],__log_data.poly[1])),__fma_rn(r2,__log_data.poly[0],lo)),hi);
  return (y);
}

}

namespace g64_pow {

__device__ __forceinline__ uint64_t asuint64(double x) {union {double f; uint64_t u;} v={x};return v.u;}
__device__ __forceinline__ double asdouble(uint64_t x) {union {uint64_t u; double f;} v={x};return v.f;}
__device__ __forceinline__ int issignaling_inline(double x){uint64_t u=asuint64(x);return ((u&0x7ff8000000000000ULL)==0x7ff0000000000000ULL)&&(u&0xfffffffffffffULL);}
__device__ __forceinline__ double __math_oflow(uint32_t s){return s ? -(g64_inf()):(g64_inf());}
__device__ __forceinline__ double __math_uflow(uint32_t s){return asdouble((uint64_t)(!!s)<<63);}
__device__ __forceinline__ double __math_divzero(uint32_t s){return s ? -(g64_inf()):(g64_inf());}
__device__ __forceinline__ double __math_invalid(double x){uint64_t u=asuint64(x);return asdouble((u&0x7fffffffffffffffULL)>0x7ff0000000000000ULL ? u|0x8000000000000ULL : 0xfff8000000000000ULL);}
struct exp_data {
  double invln2N;
  double negln2hiN;
  double negln2loN;
  double poly[4];
  double shift;
  double exp2_shift;
  double exp2_poly[5];
  double neglog10_2hiN;
  double neglog10_2loN;
  double exp10_poly[5];
  uint64_t tab[2*(1 << 7)];
  double invlog10_2N;
};
struct log_data {
  double ln2hi;
  double ln2lo;
  double poly[6 - 1];
  double poly1[12 - 1];
  struct {double invc, logc;} tab[1 << 7];
};
struct log2_data {
  double invln2hi;
  double invln2lo;
  double poly[7 - 1];
  double poly1[11 - 1];
  struct {double invc, logc;} tab[1 << 6];
};
struct pow_log_data {
  double ln2hi;
  double ln2lo;
  double poly[8 - 1];
  struct {double invc, pad, logc, logctail;} tab[1 << 7];
};
__device__ const struct exp_data __exp_data = {
0x1.71547652b82fep0 * (1 << 7),
-0x1.62e42fefa0000p-8,
-0x1.cf79abc9e3b3ap-47,
{
0x1.ffffffffffdbdp-2,
0x1.555555555543cp-3,
0x1.55555cf172b91p-5,
0x1.1111167a4d017p-7,
},
0x1.8p52,
0x1.8p52 / (1 << 7),
{
0x1.62e42fefa39efp-1,
0x1.ebfbdff82c424p-3,
0x1.c6b08d70cf4b5p-5,
0x1.3b2abd24650ccp-7,
0x1.5d7e09b4e3a84p-10,
},
-0x1.3441350ap-2 / (1 << 7),
0x1.0c0219dc1da99p-39 / (1 << 7),
{
0x1.26bb1bbb55516p1,
0x1.53524c73ce9fep1,
0x1.0470591ce4b26p1,
0x1.2bd76577fe684p0,
0x1.1446eeccd0efbp-1
},
{
0x0, 0x3ff0000000000000,
0x3c9b3b4f1a88bf6e, 0x3feff63da9fb3335,
0xbc7160139cd8dc5d, 0x3fefec9a3e778061,
0xbc905e7a108766d1, 0x3fefe315e86e7f85,
0x3c8cd2523567f613, 0x3fefd9b0d3158574,
0xbc8bce8023f98efa, 0x3fefd06b29ddf6de,
0x3c60f74e61e6c861, 0x3fefc74518759bc8,
0x3c90a3e45b33d399, 0x3fefbe3ecac6f383,
0x3c979aa65d837b6d, 0x3fefb5586cf9890f,
0x3c8eb51a92fdeffc, 0x3fefac922b7247f7,
0x3c3ebe3d702f9cd1, 0x3fefa3ec32d3d1a2,
0xbc6a033489906e0b, 0x3fef9b66affed31b,
0xbc9556522a2fbd0e, 0x3fef9301d0125b51,
0xbc5080ef8c4eea55, 0x3fef8abdc06c31cc,
0xbc91c923b9d5f416, 0x3fef829aaea92de0,
0x3c80d3e3e95c55af, 0x3fef7a98c8a58e51,
0xbc801b15eaa59348, 0x3fef72b83c7d517b,
0xbc8f1ff055de323d, 0x3fef6af9388c8dea,
0x3c8b898c3f1353bf, 0x3fef635beb6fcb75,
0xbc96d99c7611eb26, 0x3fef5be084045cd4,
0x3c9aecf73e3a2f60, 0x3fef54873168b9aa,
0xbc8fe782cb86389d, 0x3fef4d5022fcd91d,
0x3c8a6f4144a6c38d, 0x3fef463b88628cd6,
0x3c807a05b0e4047d, 0x3fef3f49917ddc96,
0x3c968efde3a8a894, 0x3fef387a6e756238,
0x3c875e18f274487d, 0x3fef31ce4fb2a63f,
0x3c80472b981fe7f2, 0x3fef2b4565e27cdd,
0xbc96b87b3f71085e, 0x3fef24dfe1f56381,
0x3c82f7e16d09ab31, 0x3fef1e9df51fdee1,
0xbc3d219b1a6fbffa, 0x3fef187fd0dad990,
0x3c8b3782720c0ab4, 0x3fef1285a6e4030b,
0x3c6e149289cecb8f, 0x3fef0cafa93e2f56,
0x3c834d754db0abb6, 0x3fef06fe0a31b715,
0x3c864201e2ac744c, 0x3fef0170fc4cd831,
0x3c8fdd395dd3f84a, 0x3feefc08b26416ff,
0xbc86a3803b8e5b04, 0x3feef6c55f929ff1,
0xbc924aedcc4b5068, 0x3feef1a7373aa9cb,
0xbc9907f81b512d8e, 0x3feeecae6d05d866,
0xbc71d1e83e9436d2, 0x3feee7db34e59ff7,
0xbc991919b3ce1b15, 0x3feee32dc313a8e5,
0x3c859f48a72a4c6d, 0x3feedea64c123422,
0xbc9312607a28698a, 0x3feeda4504ac801c,
0xbc58a78f4817895b, 0x3feed60a21f72e2a,
0xbc7c2c9b67499a1b, 0x3feed1f5d950a897,
0x3c4363ed60c2ac11, 0x3feece086061892d,
0x3c9666093b0664ef, 0x3feeca41ed1d0057,
0x3c6ecce1daa10379, 0x3feec6a2b5c13cd0,
0x3c93ff8e3f0f1230, 0x3feec32af0d7d3de,
0x3c7690cebb7aafb0, 0x3feebfdad5362a27,
0x3c931dbdeb54e077, 0x3feebcb299fddd0d,
0xbc8f94340071a38e, 0x3feeb9b2769d2ca7,
0xbc87deccdc93a349, 0x3feeb6daa2cf6642,
0xbc78dec6bd0f385f, 0x3feeb42b569d4f82,
0xbc861246ec7b5cf6, 0x3feeb1a4ca5d920f,
0x3c93350518fdd78e, 0x3feeaf4736b527da,
0x3c7b98b72f8a9b05, 0x3feead12d497c7fd,
0x3c9063e1e21c5409, 0x3feeab07dd485429,
0x3c34c7855019c6ea, 0x3feea9268a5946b7,
0x3c9432e62b64c035, 0x3feea76f15ad2148,
0xbc8ce44a6199769f, 0x3feea5e1b976dc09,
0xbc8c33c53bef4da8, 0x3feea47eb03a5585,
0xbc845378892be9ae, 0x3feea34634ccc320,
0xbc93cedd78565858, 0x3feea23882552225,
0x3c5710aa807e1964, 0x3feea155d44ca973,
0xbc93b3efbf5e2228, 0x3feea09e667f3bcd,
0xbc6a12ad8734b982, 0x3feea012750bdabf,
0xbc6367efb86da9ee, 0x3fee9fb23c651a2f,
0xbc80dc3d54e08851, 0x3fee9f7df9519484,
0xbc781f647e5a3ecf, 0x3fee9f75e8ec5f74,
0xbc86ee4ac08b7db0, 0x3fee9f9a48a58174,
0xbc8619321e55e68a, 0x3fee9feb564267c9,
0x3c909ccb5e09d4d3, 0x3feea0694fde5d3f,
0xbc7b32dcb94da51d, 0x3feea11473eb0187,
0x3c94ecfd5467c06b, 0x3feea1ed0130c132,
0x3c65ebe1abd66c55, 0x3feea2f336cf4e62,
0xbc88a1c52fb3cf42, 0x3feea427543e1a12,
0xbc9369b6f13b3734, 0x3feea589994cce13,
0xbc805e843a19ff1e, 0x3feea71a4623c7ad,
0xbc94d450d872576e, 0x3feea8d99b4492ed,
0x3c90ad675b0e8a00, 0x3feeaac7d98a6699,
0x3c8db72fc1f0eab4, 0x3feeace5422aa0db,
0xbc65b6609cc5e7ff, 0x3feeaf3216b5448c,
0x3c7bf68359f35f44, 0x3feeb1ae99157736,
0xbc93091fa71e3d83, 0x3feeb45b0b91ffc6,
0xbc5da9b88b6c1e29, 0x3feeb737b0cdc5e5,
0xbc6c23f97c90b959, 0x3feeba44cbc8520f,
0xbc92434322f4f9aa, 0x3feebd829fde4e50,
0xbc85ca6cd7668e4b, 0x3feec0f170ca07ba,
0x3c71affc2b91ce27, 0x3feec49182a3f090,
0x3c6dd235e10a73bb, 0x3feec86319e32323,
0xbc87c50422622263, 0x3feecc667b5de565,
0x3c8b1c86e3e231d5, 0x3feed09bec4a2d33,
0xbc91bbd1d3bcbb15, 0x3feed503b23e255d,
0x3c90cc319cee31d2, 0x3feed99e1330b358,
0x3c8469846e735ab3, 0x3feede6b5579fdbf,
0xbc82dfcd978e9db4, 0x3feee36bbfd3f37a,
0x3c8c1a7792cb3387, 0x3feee89f995ad3ad,
0xbc907b8f4ad1d9fa, 0x3feeee07298db666,
0xbc55c3d956dcaeba, 0x3feef3a2b84f15fb,
0xbc90a40e3da6f640, 0x3feef9728de5593a,
0xbc68d6f438ad9334, 0x3feeff76f2fb5e47,
0xbc91eee26b588a35, 0x3fef05b030a1064a,
0x3c74ffd70a5fddcd, 0x3fef0c1e904bc1d2,
0xbc91bdfbfa9298ac, 0x3fef12c25bd71e09,
0x3c736eae30af0cb3, 0x3fef199bdd85529c,
0x3c8ee3325c9ffd94, 0x3fef20ab5fffd07a,
0x3c84e08fd10959ac, 0x3fef27f12e57d14b,
0x3c63cdaf384e1a67, 0x3fef2f6d9406e7b5,
0x3c676b2c6c921968, 0x3fef3720dcef9069,
0xbc808a1883ccb5d2, 0x3fef3f0b555dc3fa,
0xbc8fad5d3ffffa6f, 0x3fef472d4a07897c,
0xbc900dae3875a949, 0x3fef4f87080d89f2,
0x3c74a385a63d07a7, 0x3fef5818dcfba487,
0xbc82919e2040220f, 0x3fef60e316c98398,
0x3c8e5a50d5c192ac, 0x3fef69e603db3285,
0x3c843a59ac016b4b, 0x3fef7321f301b460,
0xbc82d52107b43e1f, 0x3fef7c97337b9b5f,
0xbc892ab93b470dc9, 0x3fef864614f5a129,
0x3c74b604603a88d3, 0x3fef902ee78b3ff6,
0x3c83c5ec519d7271, 0x3fef9a51fbc74c83,
0xbc8ff7128fd391f0, 0x3fefa4afa2a490da,
0xbc8dae98e223747d, 0x3fefaf482d8e67f1,
0x3c8ec3bc41aa2008, 0x3fefba1bee615a27,
0x3c842b94c3a9eb32, 0x3fefc52b376bba97,
0x3c8a64a931d185ee, 0x3fefd0765b6e4540,
0xbc8e37bae43be3ed, 0x3fefdbfdad9cbe14,
0x3c77893b4d91cd9d, 0x3fefe7c1819e90d8,
0x3c5305c14160cc89, 0x3feff3c22b8f71f1,
},
0x1.a934f0979a371p1 * (1 << 7)
};
__device__ const struct pow_log_data __pow_log_data = {
0x1.62e42fefa3800p-1,
0x1.ef35793c76730p-45,
{
-0x1p-1,
0x1.555555555556p-2 * -2,
-0x1.0000000000006p-2 * -2,
0x1.999999959554ep-3 * 4,
-0x1.555555529a47ap-3 * 4,
0x1.2495b9b4845e9p-3 * -8,
-0x1.0002b8b263fc3p-3 * -8,
},
{
{0x1.6a00000000000p+0, 0, -0x1.62c82f2b9c800p-2, 0x1.ab42428375680p-48},
{0x1.6800000000000p+0, 0, -0x1.5d1bdbf580800p-2, -0x1.ca508d8e0f720p-46},
{0x1.6600000000000p+0, 0, -0x1.5767717455800p-2, -0x1.362a4d5b6506dp-45},
{0x1.6400000000000p+0, 0, -0x1.51aad872df800p-2, -0x1.684e49eb067d5p-49},
{0x1.6200000000000p+0, 0, -0x1.4be5f95777800p-2, -0x1.41b6993293ee0p-47},
{0x1.6000000000000p+0, 0, -0x1.4618bc21c6000p-2, 0x1.3d82f484c84ccp-46},
{0x1.5e00000000000p+0, 0, -0x1.404308686a800p-2, 0x1.c42f3ed820b3ap-50},
{0x1.5c00000000000p+0, 0, -0x1.3a64c55694800p-2, 0x1.0b1c686519460p-45},
{0x1.5a00000000000p+0, 0, -0x1.347dd9a988000p-2, 0x1.5594dd4c58092p-45},
{0x1.5800000000000p+0, 0, -0x1.2e8e2bae12000p-2, 0x1.67b1e99b72bd8p-45},
{0x1.5600000000000p+0, 0, -0x1.2895a13de8800p-2, 0x1.5ca14b6cfb03fp-46},
{0x1.5600000000000p+0, 0, -0x1.2895a13de8800p-2, 0x1.5ca14b6cfb03fp-46},
{0x1.5400000000000p+0, 0, -0x1.22941fbcf7800p-2, -0x1.65a242853da76p-46},
{0x1.5200000000000p+0, 0, -0x1.1c898c1699800p-2, -0x1.fafbc68e75404p-46},
{0x1.5000000000000p+0, 0, -0x1.1675cababa800p-2, 0x1.f1fc63382a8f0p-46},
{0x1.4e00000000000p+0, 0, -0x1.1058bf9ae4800p-2, -0x1.6a8c4fd055a66p-45},
{0x1.4c00000000000p+0, 0, -0x1.0a324e2739000p-2, -0x1.c6bee7ef4030ep-47},
{0x1.4a00000000000p+0, 0, -0x1.0402594b4d000p-2, -0x1.036b89ef42d7fp-48},
{0x1.4a00000000000p+0, 0, -0x1.0402594b4d000p-2, -0x1.036b89ef42d7fp-48},
{0x1.4800000000000p+0, 0, -0x1.fb9186d5e4000p-3, 0x1.d572aab993c87p-47},
{0x1.4600000000000p+0, 0, -0x1.ef0adcbdc6000p-3, 0x1.b26b79c86af24p-45},
{0x1.4400000000000p+0, 0, -0x1.e27076e2af000p-3, -0x1.72f4f543fff10p-46},
{0x1.4200000000000p+0, 0, -0x1.d5c216b4fc000p-3, 0x1.1ba91bbca681bp-45},
{0x1.4000000000000p+0, 0, -0x1.c8ff7c79aa000p-3, 0x1.7794f689f8434p-45},
{0x1.4000000000000p+0, 0, -0x1.c8ff7c79aa000p-3, 0x1.7794f689f8434p-45},
{0x1.3e00000000000p+0, 0, -0x1.bc286742d9000p-3, 0x1.94eb0318bb78fp-46},
{0x1.3c00000000000p+0, 0, -0x1.af3c94e80c000p-3, 0x1.a4e633fcd9066p-52},
{0x1.3a00000000000p+0, 0, -0x1.a23bc1fe2b000p-3, -0x1.58c64dc46c1eap-45},
{0x1.3a00000000000p+0, 0, -0x1.a23bc1fe2b000p-3, -0x1.58c64dc46c1eap-45},
{0x1.3800000000000p+0, 0, -0x1.9525a9cf45000p-3, -0x1.ad1d904c1d4e3p-45},
{0x1.3600000000000p+0, 0, -0x1.87fa06520d000p-3, 0x1.bbdbf7fdbfa09p-45},
{0x1.3400000000000p+0, 0, -0x1.7ab890210e000p-3, 0x1.bdb9072534a58p-45},
{0x1.3400000000000p+0, 0, -0x1.7ab890210e000p-3, 0x1.bdb9072534a58p-45},
{0x1.3200000000000p+0, 0, -0x1.6d60fe719d000p-3, -0x1.0e46aa3b2e266p-46},
{0x1.3000000000000p+0, 0, -0x1.5ff3070a79000p-3, -0x1.e9e439f105039p-46},
{0x1.3000000000000p+0, 0, -0x1.5ff3070a79000p-3, -0x1.e9e439f105039p-46},
{0x1.2e00000000000p+0, 0, -0x1.526e5e3a1b000p-3, -0x1.0de8b90075b8fp-45},
{0x1.2c00000000000p+0, 0, -0x1.44d2b6ccb8000p-3, 0x1.70cc16135783cp-46},
{0x1.2c00000000000p+0, 0, -0x1.44d2b6ccb8000p-3, 0x1.70cc16135783cp-46},
{0x1.2a00000000000p+0, 0, -0x1.371fc201e9000p-3, 0x1.178864d27543ap-48},
{0x1.2800000000000p+0, 0, -0x1.29552f81ff000p-3, -0x1.48d301771c408p-45},
{0x1.2600000000000p+0, 0, -0x1.1b72ad52f6000p-3, -0x1.e80a41811a396p-45},
{0x1.2600000000000p+0, 0, -0x1.1b72ad52f6000p-3, -0x1.e80a41811a396p-45},
{0x1.2400000000000p+0, 0, -0x1.0d77e7cd09000p-3, 0x1.a699688e85bf4p-47},
{0x1.2400000000000p+0, 0, -0x1.0d77e7cd09000p-3, 0x1.a699688e85bf4p-47},
{0x1.2200000000000p+0, 0, -0x1.fec9131dbe000p-4, -0x1.575545ca333f2p-45},
{0x1.2000000000000p+0, 0, -0x1.e27076e2b0000p-4, 0x1.a342c2af0003cp-45},
{0x1.2000000000000p+0, 0, -0x1.e27076e2b0000p-4, 0x1.a342c2af0003cp-45},
{0x1.1e00000000000p+0, 0, -0x1.c5e548f5bc000p-4, -0x1.d0c57585fbe06p-46},
{0x1.1c00000000000p+0, 0, -0x1.a926d3a4ae000p-4, 0x1.53935e85baac8p-45},
{0x1.1c00000000000p+0, 0, -0x1.a926d3a4ae000p-4, 0x1.53935e85baac8p-45},
{0x1.1a00000000000p+0, 0, -0x1.8c345d631a000p-4, 0x1.37c294d2f5668p-46},
{0x1.1a00000000000p+0, 0, -0x1.8c345d631a000p-4, 0x1.37c294d2f5668p-46},
{0x1.1800000000000p+0, 0, -0x1.6f0d28ae56000p-4, -0x1.69737c93373dap-45},
{0x1.1600000000000p+0, 0, -0x1.51b073f062000p-4, 0x1.f025b61c65e57p-46},
{0x1.1600000000000p+0, 0, -0x1.51b073f062000p-4, 0x1.f025b61c65e57p-46},
{0x1.1400000000000p+0, 0, -0x1.341d7961be000p-4, 0x1.c5edaccf913dfp-45},
{0x1.1400000000000p+0, 0, -0x1.341d7961be000p-4, 0x1.c5edaccf913dfp-45},
{0x1.1200000000000p+0, 0, -0x1.16536eea38000p-4, 0x1.47c5e768fa309p-46},
{0x1.1000000000000p+0, 0, -0x1.f0a30c0118000p-5, 0x1.d599e83368e91p-45},
{0x1.1000000000000p+0, 0, -0x1.f0a30c0118000p-5, 0x1.d599e83368e91p-45},
{0x1.0e00000000000p+0, 0, -0x1.b42dd71198000p-5, 0x1.c827ae5d6704cp-46},
{0x1.0e00000000000p+0, 0, -0x1.b42dd71198000p-5, 0x1.c827ae5d6704cp-46},
{0x1.0c00000000000p+0, 0, -0x1.77458f632c000p-5, -0x1.cfc4634f2a1eep-45},
{0x1.0c00000000000p+0, 0, -0x1.77458f632c000p-5, -0x1.cfc4634f2a1eep-45},
{0x1.0a00000000000p+0, 0, -0x1.39e87b9fec000p-5, 0x1.502b7f526feaap-48},
{0x1.0a00000000000p+0, 0, -0x1.39e87b9fec000p-5, 0x1.502b7f526feaap-48},
{0x1.0800000000000p+0, 0, -0x1.f829b0e780000p-6, -0x1.980267c7e09e4p-45},
{0x1.0800000000000p+0, 0, -0x1.f829b0e780000p-6, -0x1.980267c7e09e4p-45},
{0x1.0600000000000p+0, 0, -0x1.7b91b07d58000p-6, -0x1.88d5493faa639p-45},
{0x1.0400000000000p+0, 0, -0x1.fc0a8b0fc0000p-7, -0x1.f1e7cf6d3a69cp-50},
{0x1.0400000000000p+0, 0, -0x1.fc0a8b0fc0000p-7, -0x1.f1e7cf6d3a69cp-50},
{0x1.0200000000000p+0, 0, -0x1.fe02a6b100000p-8, -0x1.9e23f0dda40e4p-46},
{0x1.0200000000000p+0, 0, -0x1.fe02a6b100000p-8, -0x1.9e23f0dda40e4p-46},
{0x1.0000000000000p+0, 0, 0x0.0000000000000p+0, 0x0.0000000000000p+0},
{0x1.0000000000000p+0, 0, 0x0.0000000000000p+0, 0x0.0000000000000p+0},
{0x1.fc00000000000p-1, 0, 0x1.0101575890000p-7, -0x1.0c76b999d2be8p-46},
{0x1.f800000000000p-1, 0, 0x1.0205658938000p-6, -0x1.3dc5b06e2f7d2p-45},
{0x1.f400000000000p-1, 0, 0x1.8492528c90000p-6, -0x1.aa0ba325a0c34p-45},
{0x1.f000000000000p-1, 0, 0x1.0415d89e74000p-5, 0x1.111c05cf1d753p-47},
{0x1.ec00000000000p-1, 0, 0x1.466aed42e0000p-5, -0x1.c167375bdfd28p-45},
{0x1.e800000000000p-1, 0, 0x1.894aa149fc000p-5, -0x1.97995d05a267dp-46},
{0x1.e400000000000p-1, 0, 0x1.ccb73cdddc000p-5, -0x1.a68f247d82807p-46},
{0x1.e200000000000p-1, 0, 0x1.eea31c006c000p-5, -0x1.e113e4fc93b7bp-47},
{0x1.de00000000000p-1, 0, 0x1.1973bd1466000p-4, -0x1.5325d560d9e9bp-45},
{0x1.da00000000000p-1, 0, 0x1.3bdf5a7d1e000p-4, 0x1.cc85ea5db4ed7p-45},
{0x1.d600000000000p-1, 0, 0x1.5e95a4d97a000p-4, -0x1.c69063c5d1d1ep-45},
{0x1.d400000000000p-1, 0, 0x1.700d30aeac000p-4, 0x1.c1e8da99ded32p-49},
{0x1.d000000000000p-1, 0, 0x1.9335e5d594000p-4, 0x1.3115c3abd47dap-45},
{0x1.cc00000000000p-1, 0, 0x1.b6ac88dad6000p-4, -0x1.390802bf768e5p-46},
{0x1.ca00000000000p-1, 0, 0x1.c885801bc4000p-4, 0x1.646d1c65aacd3p-45},
{0x1.c600000000000p-1, 0, 0x1.ec739830a2000p-4, -0x1.dc068afe645e0p-45},
{0x1.c400000000000p-1, 0, 0x1.fe89139dbe000p-4, -0x1.534d64fa10afdp-45},
{0x1.c000000000000p-1, 0, 0x1.1178e8227e000p-3, 0x1.1ef78ce2d07f2p-45},
{0x1.be00000000000p-1, 0, 0x1.1aa2b7e23f000p-3, 0x1.ca78e44389934p-45},
{0x1.ba00000000000p-1, 0, 0x1.2d1610c868000p-3, 0x1.39d6ccb81b4a1p-47},
{0x1.b800000000000p-1, 0, 0x1.365fcb0159000p-3, 0x1.62fa8234b7289p-51},
{0x1.b400000000000p-1, 0, 0x1.4913d8333b000p-3, 0x1.5837954fdb678p-45},
{0x1.b200000000000p-1, 0, 0x1.527e5e4a1b000p-3, 0x1.633e8e5697dc7p-45},
{0x1.ae00000000000p-1, 0, 0x1.6574ebe8c1000p-3, 0x1.9cf8b2c3c2e78p-46},
{0x1.ac00000000000p-1, 0, 0x1.6f0128b757000p-3, -0x1.5118de59c21e1p-45},
{0x1.aa00000000000p-1, 0, 0x1.7898d85445000p-3, -0x1.c661070914305p-46},
{0x1.a600000000000p-1, 0, 0x1.8beafeb390000p-3, -0x1.73d54aae92cd1p-47},
{0x1.a400000000000p-1, 0, 0x1.95a5adcf70000p-3, 0x1.7f22858a0ff6fp-47},
{0x1.a000000000000p-1, 0, 0x1.a93ed3c8ae000p-3, -0x1.8724350562169p-45},
{0x1.9e00000000000p-1, 0, 0x1.b31d8575bd000p-3, -0x1.c358d4eace1aap-47},
{0x1.9c00000000000p-1, 0, 0x1.bd087383be000p-3, -0x1.d4bc4595412b6p-45},
{0x1.9a00000000000p-1, 0, 0x1.c6ffbc6f01000p-3, -0x1.1ec72c5962bd2p-48},
{0x1.9600000000000p-1, 0, 0x1.db13db0d49000p-3, -0x1.aff2af715b035p-45},
{0x1.9400000000000p-1, 0, 0x1.e530effe71000p-3, 0x1.212276041f430p-51},
{0x1.9200000000000p-1, 0, 0x1.ef5ade4dd0000p-3, -0x1.a211565bb8e11p-51},
{0x1.9000000000000p-1, 0, 0x1.f991c6cb3b000p-3, 0x1.bcbecca0cdf30p-46},
{0x1.8c00000000000p-1, 0, 0x1.07138604d5800p-2, 0x1.89cdb16ed4e91p-48},
{0x1.8a00000000000p-1, 0, 0x1.0c42d67616000p-2, 0x1.7188b163ceae9p-45},
{0x1.8800000000000p-1, 0, 0x1.1178e8227e800p-2, -0x1.c210e63a5f01cp-45},
{0x1.8600000000000p-1, 0, 0x1.16b5ccbacf800p-2, 0x1.b9acdf7a51681p-45},
{0x1.8400000000000p-1, 0, 0x1.1bf99635a6800p-2, 0x1.ca6ed5147bdb7p-45},
{0x1.8200000000000p-1, 0, 0x1.214456d0eb800p-2, 0x1.a87deba46baeap-47},
{0x1.7e00000000000p-1, 0, 0x1.2bef07cdc9000p-2, 0x1.a9cfa4a5004f4p-45},
{0x1.7c00000000000p-1, 0, 0x1.314f1e1d36000p-2, -0x1.8e27ad3213cb8p-45},
{0x1.7a00000000000p-1, 0, 0x1.36b6776be1000p-2, 0x1.16ecdb0f177c8p-46},
{0x1.7800000000000p-1, 0, 0x1.3c25277333000p-2, 0x1.83b54b606bd5cp-46},
{0x1.7600000000000p-1, 0, 0x1.419b423d5e800p-2, 0x1.8e436ec90e09dp-47},
{0x1.7400000000000p-1, 0, 0x1.4718dc271c800p-2, -0x1.f27ce0967d675p-45},
{0x1.7200000000000p-1, 0, 0x1.4c9e09e173000p-2, -0x1.e20891b0ad8a4p-45},
{0x1.7000000000000p-1, 0, 0x1.522ae0738a000p-2, 0x1.ebe708164c759p-45},
{0x1.6e00000000000p-1, 0, 0x1.57bf753c8d000p-2, 0x1.fadedee5d40efp-46},
{0x1.6c00000000000p-1, 0, 0x1.5d5bddf596000p-2, -0x1.a0b2a08a465dcp-47},
}
};
__device__ __forceinline__ uint32_t
top12 (double x)
{
  return asuint64 (x) >> 52;
}
__device__ __forceinline__ int
checkint (uint64_t iy)
{
  int e = iy >> 52 & 0x7ff;
  if (e < 0x3ff)
    return 0;
  if (e > 0x3ff + 52)
    return 2;
  if (iy & ((1ULL << (0x3ff + 52 - e)) - 1))
    return 0;
  if (iy & (1ULL << (0x3ff + 52 - e)))
    return 1;
  return 2;
}
__device__ __forceinline__ int
zeroinfnan (uint64_t i)
{
  return 2 * i - 1 >= 2 * asuint64 ((g64_inf())) - 1;
}
__device__ __forceinline__ double
log_inline (uint64_t ix, double *tail)
{
  double z, r, y, invc, logc, logctail, kd, hi, t1, t2, lo, lo1, lo2, p;
  uint64_t iz, tmp;
  int k, i;
  tmp = ix - 0x3fe6955500000000;
  i = (tmp >> (52 - 7)) % (1 << 7);
  k = (int64_t) tmp >> 52;
  iz = ix - (tmp & 0xfffULL << 52);
  z = asdouble (iz);
  kd = (double) k;
  invc = __pow_log_data.tab[i].invc;
  logc = __pow_log_data.tab[i].logc;
  logctail = __pow_log_data.tab[i].logctail;
  r = __fma_rn(z, invc, -1.0);
  t1 = __fma_rn(kd,__pow_log_data.ln2hi,logc);
  t2 = __dadd_rn(t1,r);
  lo1 = __fma_rn(kd,__pow_log_data.ln2lo,logctail);
  lo2 = __dadd_rn(__dsub_rn(t1,t2),r);
  double ar, ar2, ar3, lo3, lo4;
  ar = __dmul_rn(__pow_log_data.poly[0],r);
  ar2 = __dmul_rn(r,ar);
  ar3 = __dmul_rn(r,ar2);
  hi = __dadd_rn(t2,ar2);
  lo3 = __fma_rn(ar, r, -ar2);
  lo4 = __dadd_rn(__dsub_rn(t2,hi),ar2);
  p = __fma_rn(ar2,__fma_rn(ar2,__fma_rn(r,__pow_log_data.poly[6],__pow_log_data.poly[5]),__fma_rn(r,__pow_log_data.poly[4],__pow_log_data.poly[3])),__fma_rn(r,__pow_log_data.poly[2],__pow_log_data.poly[1]));
  lo = __fma_rn(ar3,p,__dadd_rn(__dadd_rn(__dadd_rn(lo1,lo2),lo3),lo4));
  y = __dadd_rn(hi,lo);
  *tail = __dadd_rn(__dsub_rn(hi,y),lo);
  return y;
}
__device__ __forceinline__ double
specialcase (double tmp, uint64_t sbits, uint64_t ki)
{
  double scale, y;
  if ((ki & 0x80000000) == 0)
    {
      sbits -= 1009ull << 52;
      scale = asdouble (sbits);
      y = __fma_rn(scale,tmp,scale);
      return ((__dmul_rn(y,0x1p1009)));
   }
  sbits += 1022ull << 52;
  scale = asdouble (sbits);
  y = __dadd_rn(scale,__dmul_rn(scale,tmp));
  if (fabs (y) < 1.0)
    {
      double hi, lo, one = 1.0;
      if (y < 0.0)
 one = -1.0;
      lo = __dadd_rn(__dsub_rn(scale,y),__dmul_rn(scale,tmp));
      hi = __dadd_rn(one,y);
      lo = __dadd_rn(__dadd_rn(__dsub_rn(one,hi),y),lo);
      y = __dsub_rn((__dadd_rn(hi,lo)),one);
      if (y == 0.0)
 y = asdouble (sbits & 0x8000000000000000);
      ((void)(__dmul_rn((0x1p-1022),0x1p-1022)));
    }
  y = __dmul_rn(0x1p-1022,y);
  return ((y));
}
__device__ __forceinline__ double
exp_inline (double x, double xtail, uint32_t sign_bias)
{
  uint32_t abstop;
  uint64_t ki, idx, top, sbits;
  double kd, z, r, r2, scale, tail, tmp;
  abstop = top12 (x) & 0x7ff;
  if ((abstop - top12 (0x1p-54) >= top12 (512.0) - top12 (0x1p-54)))
    {
      if (abstop - top12 (0x1p-54) >= 0x80000000)
 {
   double one = 1 ? __dadd_rn(1.0,x) : 1.0;
   return sign_bias ? -one : one;
 }
      if (abstop >= top12 (1024.0))
 {
   if (asuint64 (x) >> 63)
     return __math_uflow (sign_bias);
   else
     return __math_oflow (sign_bias);
 }
      abstop = 0;
    }
  z = __dmul_rn(__exp_data.invln2N,x);
  kd = __fma_rn(__exp_data.invln2N, x, __exp_data.shift);
  ki = asuint64 (kd);
  kd = __dsub_rn(kd,__exp_data.shift);
  r = __fma_rn(kd, __exp_data.negln2loN, __fma_rn(kd, __exp_data.negln2hiN, x));
  r = __dadd_rn(r,xtail);
  idx = 2 * (ki % (1 << 7));
  top = (ki + sign_bias) << (52 - 7);
  tail = asdouble (__exp_data.tab[idx]);
  sbits = __exp_data.tab[idx + 1] + top;
  r2 = __dmul_rn(r,r);
  tmp = __fma_rn(__dmul_rn(r2,r2), __fma_rn(r,__exp_data.poly[8 - 5],__exp_data.poly[7 - 5]), __fma_rn(r2,__fma_rn(r,__exp_data.poly[6 - 5],__exp_data.poly[5 - 5]),__dadd_rn(tail,r)));
  if ((abstop == 0))
    return specialcase (tmp, sbits, ki);
  scale = asdouble (sbits);
  return __fma_rn(scale,tmp,scale);
}
__device__ __forceinline__ double pow (double x, double y)
{
  uint32_t sign_bias = 0;
  uint64_t ix, iy;
  uint32_t topx, topy;
  ix = asuint64 (x);
  iy = asuint64 (y);
  topx = top12 (x);
  topy = top12 (y);
  if ((topx - 0x001 >= 0x7ff - 0x001 || (topy & 0x7ff) - 0x3be >= 0x43e - 0x3be))
    {
      if ((zeroinfnan (iy)))
 {
   if (2 * iy == 0)
     return issignaling_inline (x) ? __dadd_rn(x,y) : 1.0;
   if (ix == asuint64 (1.0))
     return issignaling_inline (y) ? __dadd_rn(x,y) : 1.0;
   if (2 * ix > 2 * asuint64 ((g64_inf()))
       || 2 * iy > 2 * asuint64 ((g64_inf())))
     return __dadd_rn(x,y);
   if (2 * ix == 2 * asuint64 (1.0))
     return 1.0;
   if ((2 * ix < 2 * asuint64 (1.0)) == !(iy >> 63))
     return 0.0;
   return __dmul_rn(y,y);
 }
      if ((zeroinfnan (ix)))
 {
   double x2 = __dmul_rn(x,x);
   if (ix >> 63 && checkint (iy) == 1)
     {
       x2 = -x2;
       sign_bias = 1;
     }
   if (0 && 2 * ix == 0 && iy >> 63)
     return __math_divzero (sign_bias);
   return iy >> 63 ? (__ddiv_rn(1,x2)) : x2;
 }
      if (ix >> 63)
 {
   int yint = checkint (iy);
   if (yint == 0)
     return __math_invalid (x);
   if (yint == 1)
     sign_bias = (0x800 << 7);
   ix &= 0x7fffffffffffffff;
   topx &= 0x7ff;
 }
      if ((topy & 0x7ff) - 0x3be >= 0x43e - 0x3be)
 {
   if (ix == asuint64 (1.0))
     return 1.0;
   if ((topy & 0x7ff) < 0x3be)
     {
       if (1)
  return ix > asuint64 (1.0) ? __dadd_rn(1.0,y) : __dsub_rn(1.0,y);
       else
  return 1.0;
     }
   return (ix > asuint64 (1.0)) == (topy < 0x800) ? __math_oflow (0)
        : __math_uflow (0);
 }
      if (topx == 0)
 {
   ix = asuint64 (__dmul_rn((x),0x1p52));
   ix &= 0x7fffffffffffffff;
   ix -= 52ULL << 52;
 }
    }
  double lo;
  double hi = log_inline (ix, &lo);
  double ehi, elo;
  ehi = __dmul_rn(y,hi);
  elo = __fma_rn(y,lo,__fma_rn(y,hi,-ehi));
  return exp_inline (ehi, elo, sign_bias);
}

}

/* Correctly-rounded cosine function for binary64 value.

Copyright (c) 2022-2025 Paul Zimmermann and Tom Hubrecht

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
namespace g64_cos {
__device__ const double g64_table_0[2] = {1.0, -1.0};
__device__ const double g64_table_1[][3] = {
        {0x1.8000000000009p-23, 0x1.fffffffffff7p-1, 0x1.b56666666666cp-143},
        {0x1.8000000000024p-22, 0x1.ffffffffffdcp-1, 0x1.b56666666667ep-137},
        {0x1.800000000009p-21, 0x1.ffffffffff7p-1, 0x1.b5666666666c4p-131},
        {0x1.20000000000f3p-20, 0x1.fffffffffebcp-1, 0x1.37642666666fdp-127},
        {0x1.800000000024p-20, 0x1.fffffffffdcp-1, 0x1.b5666666667ddp-125},
      };



// Unsigned arithmetic modulo 2^128 using two limbs. No NVRTC flag needed.
struct u128 {
  uint64_t lo,hi;
  __device__ u128() = default;
  __device__ constexpr u128(uint64_t l):lo(l),hi(0){}
  __device__ constexpr u128(uint64_t l,uint64_t h):lo(l),hi(h){}
  __device__ operator uint64_t() const {return lo;}
  __device__ u128 operator+(u128 b)const{uint64_t l=lo+b.lo;return u128(l,hi+b.hi+(l<lo));}
  __device__ u128 operator-(u128 b)const{return u128(lo-b.lo,hi-b.hi-(lo<b.lo));}
  __device__ u128 operator*(u128 b)const{return u128(lo*b.lo,__umul64hi(lo,b.lo)+lo*b.hi+hi*b.lo);}
  __device__ u128 operator|(u128 b)const{return u128(lo|b.lo,hi|b.hi);}
  template<class K> __device__ u128 operator>>(K k)const{if(!k)return *this;if(k>=128)return u128(0);if(k>=64)return u128(hi>>(k-64));return u128((lo>>k)|(hi<<(64-k)),hi>>k);}
  template<class K> __device__ u128 operator<<(K k)const{if(!k)return *this;if(k>=128)return u128(0);if(k>=64)return u128(0,lo<<(k-64));return u128(lo<<k,(hi<<k)|(lo>>(64-k)));}
  template<class V> __device__ u128 operator+(V v)const{return *this+u128((uint64_t)v);}
  __device__ u128& operator+=(u128 b){*this=*this+b;return *this;}
  __device__ bool operator<(u128 b)const{return hi<b.hi||(hi==b.hi&&lo<b.lo);}
  __device__ bool operator>(u128 b)const{return b<*this;}
  __device__ bool operator==(u128 b)const{return hi==b.hi&&lo==b.lo;}
};

typedef struct {uint64_t lo,hi; int64_t ex; uint64_t sgn;} dint64_t;
__device__ inline u128 get128(const dint64_t *a){return u128(a->lo,a->hi);}
__device__ inline void set128(dint64_t *a,u128 v){a->lo=v.lo;a->hi=v.hi;}

typedef union {
  u128 r;
  struct {
    uint64_t l;
    uint64_t h;
  };
} uint128_t;
typedef union {
  double f;
  uint64_t u;
} f64_u;
__device__ __forceinline__ void fast_extract (int64_t *e, uint64_t *m, double x) {
  f64_u _x; _x.f = x;
  *e = (_x.u >> 52) & 0x7ff;
  *m = (_x.u & (~0ull >> 12)) + (*e ? (1ull << 52) : 0);
  *e = *e - 0x3fe;
}
__device__ __forceinline__ int
dint_zero_p (const dint64_t *a)
{
  return a->hi == 0;
}
__device__ __forceinline__ int cmp(int64_t a, int64_t b) { return (a > b) - (a < b); }
__device__ __forceinline__ int cmpu128 (u128 a, u128 b) { return (a > b) - (a < b); }
__device__ const dint64_t ZERO = {0x0,0x0,-1076,0x0};
__device__ const dint64_t MAGIC = {0x0,0x8000000000000000,-10,0x0};
__device__ __forceinline__ signed char
cmp_dint_abs (const dint64_t *a, const dint64_t *b) {
  if (dint_zero_p (a))
    return dint_zero_p (b) ? 0 : -1;
  if (dint_zero_p (b))
    return +1;
  char c1 = cmp (a->ex, b->ex);
  return c1 ? c1 : cmpu128 (get128(a), get128(b));
}
__device__ __forceinline__ void cp_dint(dint64_t *r, const dint64_t *a) {
  r->ex = a->ex;
  set128(r,get128(a));
  r->sgn = a->sgn;
}
__device__ __forceinline__ void
add_dint (dint64_t *r, const dint64_t *a, const dint64_t *b) {
  if (!(a->hi | a->lo)) {
    cp_dint (r, b);
    return;
  }
  switch (cmp_dint_abs (a, b)) {
  case 0:
    if (a->sgn ^ b->sgn) {
      cp_dint (r, &ZERO);
      return;
    }
    cp_dint (r, a);
    r->ex++;
    return;
  case -1:
    {
      const dint64_t *tmp = a; a = b; b = tmp;
      break;
    }
  }
  u128 A = get128(a), B = get128(b);
  uint64_t k = a->ex - b->ex;
  if (k > 0) {
    B = (k < 128) ? B >> k : u128(0);
  }
  u128 C;
  unsigned char sgn = a->sgn;
  r->ex = a->ex;
  if (a->sgn ^ b->sgn) {
    C = A - B;
    uint64_t ch = C >> 64;
    uint64_t ex = ch ? g64_clzll(ch) : 64 + g64_clzll(C);
    if (ex > 0)
    {
      if (k == 1)
        C = (A << ex) - (get128(b) << (ex - 1));
      else
        C = (A << ex) - (B << ex);
      r->ex -= ex;
      ex = g64_clzll (C >> 64);
    }
    C = C << ex;
    r->ex -= ex;
  } else {
    C = A + B;
    if (C < A)
    {
      C = ((u128) 1 << 127) | (C >> 1);
      r->ex ++;
    }
  }
  r->sgn = sgn;
  set128(r,C);
}
__device__ __forceinline__ void
mul_dint (dint64_t *r, const dint64_t *a, const dint64_t *b) {
  u128 bh = b->hi, bl = b->lo;
  u128 m1 = (u128)(a->hi) * bl;
  u128 m2 = (u128)(a->lo) * bh;
  set128(r,(u128)(a->hi) * bh);
  set128(r,get128(r) + ((m1 >> 64) + (m2 >> 64)));
  uint64_t ex = r->hi >> 63;
  set128(r,get128(r) << (1 - ex));
  r->ex = a->ex + b->ex + ex - 1;
  r->sgn = a->sgn ^ b->sgn;
}
__device__ __forceinline__ void
mul_dint_21 (dint64_t *r, const dint64_t *a, const dint64_t *b) {
  u128 bh = b->hi;
  u128 hi = (u128) (a->hi) * bh;
  u128 lo = (u128) (a->lo) * bh;
  set128(r,hi);
  set128(r,get128(r) + (lo >> 64));
  uint64_t ex = r->hi >> 63;
  set128(r,get128(r) << (1 - ex));
  r->ex = a->ex + b->ex + ex - 1;
  r->sgn = a->sgn ^ b->sgn;
}
__device__ __forceinline__ void dint_fromd (dint64_t *a, double b) {
  fast_extract (&a->ex, &a->hi, b);
  uint32_t t = g64_clzll (a->hi);
  a->sgn = b < 0.0;
  a->hi = a->hi << t;
  a->ex = a->ex - (t > 11 ? t - 12 : 0);
  a->lo = 0;
}
__device__ __forceinline__ void subnormalize_dint(dint64_t *a) {
  if (a->ex > -1023)
    return;
  uint64_t ex = -(1011 + a->ex);
  uint64_t hi = a->hi >> ex;
  uint64_t md = (a->hi >> (ex - 1)) & 0x1;
  uint64_t lo = (a->hi & (~0ull >> ex)) || a->lo;
  switch (0) {
  case 0:
    hi += lo ? md : hi & md;
    break;
  case 0x400:
    hi += a->sgn & (md | lo);
    break;
  case 0x800:
    hi += (!a->sgn) & (md | lo);
    break;
  }
  a->hi = hi << ex;
  a->lo = 0;
  if (!a->hi) {
    a->ex++;
    a->hi = (1ull << 63);
  }
}
__device__ __forceinline__ double dint_tod(dint64_t *a) {
  subnormalize_dint (a);
  f64_u r; r.u = (a->hi >> 11) | (0x3ffll << 52);
  double rd = 0.0;
  if ((a->hi >> 10) & 0x1)
    rd = __dadd_rn(rd,0x1p-53);
  if (a->hi & 0x3ff || a->lo)
    rd = __dadd_rn(rd,0x1p-54);
  if (a->sgn)
    rd = -rd;
  r.u = r.u | a->sgn << 63;
  r.f = __dadd_rn(r.f,rd);
  f64_u e;
  if (a->ex > -1022) {
    if (a->ex > 1024)
      if (a->ex == 1025) {
        r.f = __dmul_rn(r.f,0x1p+1);
        e.f = 0x1p+1023;
      } else {
        r.f = 0x1.fffffffffffffp+1023;
        e.f = 0x1.fffffffffffffp+1023;
      }
    else
      e.u = ((a->ex + 1022) & 0x7ff) << 52;
  } else {
    if (a->ex < -1073) {
      if (a->ex == -1074) {
        r.f = __dmul_rn(r.f,0x1p-1);
        e.f = 0x1p-1074;
      } else {
        r.f = 0x0.0000000000001p-1022;
        e.f = 0x0.0000000000001p-1022;
      }
    } else {
      e.u = 1l << (a->ex + 1073);
    }
  }
  return __dmul_rn(r.f,e.f);
}
typedef union {double f; uint64_t u;} b64u64_u;
__device__ const uint64_t T[20] = {
  0x28be60db9391054a,
   0x7f09d5f47d4d3770,
   0x36d8a5664f10e410,
   0x7f9458eaf7aef158,
   0x6dc91b8e909374b8,
   0x1924bba82746487,
   0x3f877ac72c4a69cf,
   0xba208d7d4baed121,
   0x3a671c09ad17df90,
   0x4e64758e60d4ce7d,
   0x272117e2ef7e4a0e,
   0xc7fe25fff7816603,
   0xfbcbc462d6829b47,
   0xdb4d9fb3c9f2c26d,
   0xd3d18fd9a797fa8b,
   0x5d49eeb1faf97c5e,
   0xcf41ce7de294a4ba,
   0x9afed7ec47e35742,
   0x1580cc11bf1edaea,
   0xfc33ef0826bd0d87,
};
__device__ const dint64_t S[256] = {
  {0x0,0x0,128,0},
  {0x480f7956b6470765,0xc90fc5f66525d257,-8,0},
  {0xcb3ff35bd4d81baa,0xc90f87f3380388d5,-7,0},
  {0xb767005691b9d9d1,0x96cb587284b81770,-6,0},
  {0xf1d7d06db39ea9fc,0xc90e8fe6f63c2330,-6,0},
  {0xd784e031f9af76d6,0xfb514b55ccbe541a,-6,0},
  {0xf91ee371d6467dca,0x96c9b5df1877e9b5,-5,0},
  {0xf56e3c87ae3c56df,0xafea690fd5912ef3,-5,0},
  {0xc539edcbfda0cf2c,0xc90aafbd1b33efc9,-5,0},
  {0x850021e392744a4f,0xe22a7a6729d8e453,-5,0},
  {0xb21ccebc9caac3,0xfb49b98e8e7807f6,-5,0},
  {0xde5b1068d174be9c,0x8a342eda160bf5ae,-4,0},
  {0x37b2dd49d5fca3c0,0x96c32baca2ae68b4,-4,0},
  {0xb56007d16d4ad5a3,0xa351cb7fc30bc889,-4,0},
  {0xcd34d2751c2e1da7,0xafe00694866a1b44,-4,0},
  {0xf10bfca3d6464012,0xbc6dd52c3a342eb5,-4,0},
  {0x6a17954b2b7c5171,0xc8fb2f886ec09f37,-4,0},
  {0x73d1472472f4a390,0xd5880deafc18b534,-4,0},
  {0x438b4a73aecd2541,0xe214689606bf1676,-4,0},
  {0xc4e92d01a2f42935,0xeea037cc04764844,-4,0},
  {0xf0a0e36a000c7350,0xfb2b73cfc106ff68,-4,0},
  {0x60e782313f6161af,0x83db0a7231831d8f,-3,0},
  {0x77724a2b2a669bc4,0x8a2009a6b84d9402,-3,0},
  {0x56e0a8b0d177b55d,0x9064b3a76a22640c,-3,0},
  {0xf77574094d3c35c4,0x96a9049670cfae65,-3,0},
  {0x50ffe4f5caa7f1fa,0x9cecf8962d14c822,-3,0},
  {0xdec1b7f2768bdafa,0xa3308bc93904ad69,-3,0},
  {0x76f8c63986598c79,0xa973ba526a6850d9,-3,0},
  {0xfdd2fc0936594c2d,0xafb68054d520c60b,-3,0},
  {0x924bef13600f9852,0xb5f8d9f3cd8945d6,-3,0},
  {0xeb13e106732687f1,0xbc3ac352ead90abe,-3,0},
  {0xb228a03916371f6f,0xc27c389609850433,-3,0},
  {0xc7396c894bbf7389,0xc8bd35e14da15f0e,-3,0},
  {0x6b47b8c44e5b037e,0xcefdb7592542e1e9,-3,0},
  {0x7337412cf70716cb,0xd53db9224ae01bca,-3,0},
  {0xbb286d23e11c8337,0xdb7d3761c7b263b6,-3,0},
  {0x31883b30137c6e62,0xe1bc2e3cf616a7ac,-3,0},
  {0xeeb8f9c33340a2f2,0xe7fa99d983ee098f,-3,0},
  {0xed16b994af6c18ae,0xee38765d74fe4897,-3,0},
  {0x14e1a5488eaeab96,0xf475bfef2551f5b9,-3,0},
  {0x704729ae56d78a37,0xfab272b54b9871a2,-3,0},
  {0x3eac8308f1113e5e,0x8077456b7dc2d967,-2,0},
  {0xdb1f70118c9c2198,0x8395023dd418e919,-2,0},
  {0xc5a9decdfaad4db5,0x86b26de5933c2e8e,-2,0},
  {0x97965c9860c34e44,0x89cf8676d7abb55b,-2,0},
  {0xdcdca90cc73b116a,0x8cec4a05f12739e8,-2,0},
  {0xa6e3df5975cca9da,0x9008b6a763de75b7,-2,0},
  {0x899c4de737feec22,0x9324ca6fe9a04b4e,-2,0},
  {0xa89a11e07c1fe,0x964083747309d113,-2,0},
  {0x49c4863de522b217,0x995bdfca28b53a54,-2,0},
  {0xe7bc08111d0bfca4,0x9c76dd866c689dcc,-2,0},
  {0xf3ff913a4aadb85e,0x9f917abeda4498df,-2,0},
  {0xa5dbee6084ee1260,0xa2abb58949f2ced7,-2,0},
  {0x69fcb11e19f58619,0xa5c58bfbcfd4436a,-2,0},
  {0xcd12a1f6ab6b095,0xa8defc2cbe2f8fcc,-2,0},
  {0x8c95c4c91179176b,0xabf80432a65ef190,-2,0},
  {0x3feef3bb58b1f10d,0xaf10a22459fe32a6,-2,0},
  {0x16031a34d4fc855d,0xb228d418ec1869ad,-2,0},
  {0xcd73fb5d8d45d302,0xb5409827b25591f0,-2,0},
  {0x187e26d290714d70,0xb857ec684627fa4c,-2,0},
  {0xbddd8a0365d6b1d3,0xbb6ecef285f98a3a,-2,0},
  {0xdfe1b074e22fc666,0xbe853dde9658dc60,-2,0},
  {0xad5a41de48f6b26f,0xc19b3744e3262dcd,-2,0},
  {0xdab4e426409b23a0,0xc4b0b93e20c0213f,-2,0},
  {0x5cc8c00e4fccd850,0xc7c5c1e34d3055b2,-2,0},
  {0xfa6171200ab2efc3,0xcada4f4db157cf77,-2,0},
  {0x65a3132adfb7dfd5,0xcdee5f96e21b332c,-2,0},
  {0xaadb580a1eba209f,0xd101f0d8c18ed1c1,-2,0},
  {0xdf4005ef6a64aa02,0xd415012d802284f0,-2,0},
  {0x1779df36d1cc8912,0xd7278eaf9dcd5b55,-2,0},
  {0xcbabaeb97af8e8aa,0xda399779eb391377,-2,0},
  {0xece7f445cecf1e28,0xdd4b19a78aed6515,-2,0},
  {0xebc61ade6ca83cd,0xe05c1353f27b17e5,-2,0},
  {0x26a0eecdb4f16266,0xe36c829aeba6e720,-2,0},
  {0x82b0aecadf808123,0xe67c659895943123,-2,0},
  {0xb91caf23416e7e80,0xe98bba6965ef725f,-2,0},
  {0x7244ee20f591983b,0xec9a7f2a2a188aeb,-2,0},
  {0x1050cdf22f34182f,0xefa8b1f8084ccdfc,-2,0},
  {0x587f3fa044e2d27d,0xf2b650f080d0da8d,-2,0},
  {0x643720de93ba81bd,0xf5c35a316f1a3c80,-2,0},
  {0x4221dc4ba772598d,0xf8cfcbd90af8d57a,-2,0},
  {0xd24d3023da491920,0xfbdba405e9c00cca,-2,0},
  {0x8b74fe2508ab8fc2,0xfee6e0d6ff6fc5a4,-2,0},
  {0xfd958d68e8b49e6b,0x80f8c035cfee8d76,-1,0},
  {0xfb4c92369f0cf008,0x827dc071bfed6ffa,-1,0},
  {0xcb07b25a7b0372a7,0x8402702f5b30f2a9,-1,0},
  {0x9d3dc689006896f4,0x8586ce7ededc809d,-1,0},
  {0x9d52755ece3f70,0x870ada70ba4e6d49,-1,0},
  {0x984156f553344306,0x888e93158fb3bb04,-1,0},
  {0xa66d1d936c38c329,0x8a11f77e349bc245,-1,0},
  {0x575f33366be0afef,0x8b9506bbb28bb922,-1,0},
  {0xcb590d74f64e77c9,0x8d17bfdf47921ac8,-1,0},
  {0xf2be3ecae62789d4,0x8e9a21fa66d9ee8d,-1,0},
  {0x632b9cff5cfee724,0x901c2c1eb93dee39,-1,0},
  {0x609c464b3dd676ec,0x919ddd5e1ddb8b33,-1,0},
  {0x6a1ff8bfe6396e28,0x931f34caaaa5d23a,-1,0},
  {0xae4ba773da6bf754,0x94a03176acf82d45,-1,0},
  {0xe06a955a5b8e301d,0x9620d274aa290339,-1,0},
  {0xfc8b7184b21f2d50,0x97a116d7601c3515,-1,0},
  {0x9dd1eedf18a2e4df,0x9920fdb1c5d5783d,-1,0},
  {0x9ffa0d23f3c26c62,0x9aa086170c0a8d86,-1,0},
  {0xdab6b478577e7be5,0x9c1faf1a9db554af,-1,0},
  {0xdb895384528d0d60,0x9d9e77d020a5bbe6,-1,0},
  {0x98dbd3555ebcdefe,0x9f1cdf4b76138b02,-1,0},
  {0x2f895f44a303cc0b,0xa09ae4a0bb300a19,-1,0},
  {0xd29d23a624acd00c,0xa21886e449b78316,-1,0},
  {0x2be036401ba87cc2,0xa395c52ab8829dfc,-1,0},
  {0x82d9495ead5be348,0xa5129e88dc17976a,-1,0},
  {0x17218792857f4c5a,0xa68f1213c73b5124,-1,0},
  {0x3269f4702b88324a,0xa80b1ee0cb823c27,-1,0},
  {0x8e3bdf8085321556,0xa986c40579e11c0a,-1,0},
  {0xc1654b64a0081b46,0xab020097a33da341,-1,0},
  {0x811f953984eff83e,0xac7cd3ad58fee7f0,-1,0},
  {0x9a5318ac6fe94e4d,0xadf73c5ced9db0f3,-1,0},
  {0x9fe5f4ea48965e2c,0xaf7139bcf5349ac6,-1,0},
  {0x63c66682bae74898,0xb0eacae4461013ed,-1,0},
  {0x695a5332090bb09b,0xb263eee9f93e3088,-1,0},
  {0x992d96e5021e3c37,0xb3dca4e56b1e54bb,-1,0},
  {0x971f4da709ad4378,0xb554ebee3bf0b58e,-1,0},
  {0x35ebacd79f209137,0xb6ccc31c5065afee,-1,0},
  {0x9cc3ef36746de3b8,0xb8442987d22cf576,-1,0},
  {0xcdb0531c4e58484b,0xb9bb1e4930848ead,-1,0},
  {0x55b92083658bb897,0xbb31a07920c7b256,-1,0},
  {0xa4b0d21fc5036a5,0xbca7af309efd7182,-1,0},
  {0xd1f90f79f46c7e01,0xbe1d4988ee67380c,-1,0},
  {0x91a1b5eb79658c67,0xbf926e9b9a0f2127,-1,0},
  {0x721853f8e528a934,0xc1071d8275561f9b,-1,0},
  {0xcdc2bd470675104d,0xc27b55579c81f96d,-1,0},
  {0x3122c2a59efddc37,0xc3ef1535754b168d,-1,0},
  {0xf4ff2895ab6ebe89,0xc5625c36af6a222f,-1,0},
  {0x14d24739de27e2e9,0xc6d5297645257e8d,-1,0},
  {0x4ce0246ad4fa74,0xc8477c0f7bde8a98,-1,0},
  {0x4319e5ad5b0dcb84,0xc9b9531de49eb968,-1,0},
  {0xfaa3dfe675a65ee2,0xcb2aadbd5ca47af5,-1,0},
  {0x2e663b3c7555a6c3,0xcc9b8b0a0deff5d4,-1,0},
  {0x3c540a9eec47af38,0xce0bea206fcf9192,-1,0},
  {0xa81290bdbaad62e4,0xcf7bca1d476c516d,-1,0},
  {0xb9302788604e88f1,0xd0eb2a1da855fefd,-1,0},
  {0x721fc87ba1d42456,0xd25a093ef50f2482,-1,0},
  {0x87967926fdcecec4,0xd3c8669edf98d680,-1,0},
  {0x1df22346611c6b4b,0xd536415b69fe4c54,-1,0},
  {0x3090d44db12c418c,0xd6a39892e6e04764,-1,0},
  {0xa573f2aa90434ba5,0xd8106b63fa0048a0,-1,0},
  {0x2e349483e3fb2a6a,0xd97cb8ed98cb93f5,-1,0},
  {0x362cb974182e3030,0xdae8804f0ae6015b,-1,0},
  {0x3ccca3982328ed8b,0xdc53c0a7eab49b35,-1,0},
  {0x1a5bd9269d408d7e,0xddbe791825e8099e,-1,0},
  {0xcce2634be2bf54df,0xdf28a8bffe06ca56,-1,0},
  {0x8aa895d5bf3e84ea,0xe0924ec008f734fd,-1,0},
  {0xf7a1f9bd9ba13b6b,0xe1fb6a3931894b38,-1,0},
  {0x7b32c72e31824e51,0xe363fa4cb8005482,-1,0},
  {0xd40e9e6b989f89e5,0xe4cbfe1c329c453a,-1,0},
  {0x2872ce1bfc7ad1cd,0xe63374c98e22f0b4,-1,0},
  {0xf1b65cc5fd780262,0xe79a5d770e6905dc,-1,0},
  {0x431626c10485bdda,0xe900b7474edad637,-1,0},
  {0xcc39cfcc29960b1,0xea66815d4304e6c8,-1,0},
  {0x1d90f780ae951140,0xebcbbadc371c4aaa,-1,0},
  {0xc71debc372b6f9d4,0xed3062e7d086c6f0,-1,0},
  {0x2a24164daec85ccb,0xee9478a40e62bf86,-1,0},
  {0x527233b40d3432bb,0xeff7fb354a0eecb1,-1,0},
  {0x6c48e9e3420b0f1e,0xf15ae9c037b1d8f0,-1,0},
  {0x7f232aee178c6323,0xf2bd4369e6c126d3,-1,0},
  {0x3c7f10db458c337c,0xf41f0757c2889e84,-1,0},
  {0x93fa6107c4327527,0xf58034af92b102a7,-1,0},
  {0xe1079824233fef46,0xf6e0ca977bc6ac45,-1,0},
  {0xa9a56012067c570c,0xf840c835ffbfed66,-1,0},
  {0x8da894471de1a18,0xf9a02cb1fe833a0d,-1,0},
  {0x343fbf4a7d42af3,0xfafef732b66d1742,-1,0},
  {0x27c07c911290b8d1,0xfc5d26dfc4d5cfda,-1,0},
  {0x2377c3799c052fa,0xfdbabae12696eea4,-1,0},
  {0xa9c6ba50490539f,0xff17b25f38907dad,-1,0},
  {0x6f53873e2f1477ff,0x803a06415c170525,0,0},
  {0x5ca183dc973abc22,0x80e7e43a61f5b6cb,0,0},
  {0x9fba97fdf0c4d24c,0x819572af6decac84,0,0},
  {0x6fb2123fedfa6e22,0x8242b1357110d372,0,0},
  {0x91a965931f1a200a,0x82ef9f618dc5b70e,0,0},
  {0xbfd79717f2880abf,0x839c3cc917ff6cb4,0,0},
  {0x246efcff30cb064a,0x8448890195846099,0,0},
  {0x51917cac857fd5f5,0x84f483a0be2f0403,0,0},
  {0x327888fe4b62687b,0x85a02c3c7c2f5ca5,0,0},
  {0x85043222c9bdd18d,0x864b826aec4c74e5,0,0},
  {0x7e0b9b07548471a2,0x86f685c25e25acf5,0,0},
  {0x4e091160e2430712,0x87a135d95473ec89,0,0},
  {0x4f14c8afe4560291,0x884b9246854ab50b,0,0},
  {0xb892ca8361d8c84c,0x88f59aa0da591421,0,0},
  {0xc88302a31afce54a,0x899f4e7f712a765e,0,0},
  {0x660558a02136130a,0x8a48ad799b6759f3,0,0},
  {0x545f7d79ead8fa19,0x8af1b726df15e13c,0,0},
  {0x21a6675f51580bc4,0x8b9a6b1ef6da4502,0,0},
  {0x101a5adbcb9ffb43,0x8c42c8f9d2372644,0,0},
  {0x4d49cbaf15aecd80,0x8cead04f95cdbf66,0,0},
  {0xde2d43c6b67a7cbe,0x8d9280b89b9df49b,0,0},
  {0xbba4cfecbff54867,0x8e39d9cd73464364,0,0},
  {0xaf0e2345f3bd24b4,0x8ee0db26e24390f8,0,0},
  {0x9311a82459aa0f72,0x8f87845de430d777,0,0},
  {0xb144016c7a30b39a,0x902dd50bab06b1b7,0,0},
  {0x9d1072e09b72292,0x90d3ccc99f5ac58b,0,0},
  {0x6714fe6925b78cc4,0x91796b31609f0c54,0,0},
  {0x33d0a284a8c954ad,0x921eafdcc560f9c5,0,0},
  {0x1f8481e704e4a767,0x92c39a65db88809d,0,0},
  {0xb17821911e71c16e,0x93682a66e896f544,0,0},
  {0x1489a97671a42,0x940c5f7a69e5ce1c,0,0},
  {0xd6c7af02d5c16fd9,0x94b0393b14e54156,0,0},
  {0xac0106650f4ef023,0x9553b743d75ac03f,0,0},
  {0xd9f8e1a446e973b9,0x95f6d92fd79f4fba,0,0},
  {0xa7a7556c3b33abc1,0x96999e9a74ddbde3,0,0},
  {0xc0a03934f0cce19b,0x973c071f4750b49c,0,0},
  {0xd243aa0843a2c144,0x97de125a2080a8ed,0,0},
  {0x19cec845ac87a5c6,0x987fbfe70b81a708,0,0},
  {0xc4b992a37fb9b9bd,0x99210f624d30facb,0,0},
  {0x1ab42d43235757b6,0x99c200686472b4a8,0,0},
  {0x7e92c655656e6b85,0x9a6292960a6f0ab0,0,0},
  {0x698b94f50326a043,0x9b02c58832cf95c0,0,0},
  {0x9a5614e8ffbeac6f,0x9ba298dc0bfc6a88,0,0},
  {0xc7fd954194e6d8aa,0x9c420c2eff590e5f,0,0},
  {0x3e93627de8fd5779,0x9ce11f1eb18147b1,0,0},
  {0xe25e39549638ae68,0x9d7fd1490285c9e3,0,0},
  {0x2cad377d5c9c35d8,0x9e1e224c0e28bc94,0,0},
  {0xcc141e10c6460c8b,0x9ebc11c62c1a1dfb,0,0},
  {0xa88d5f46834bbf8d,0x9f599f55f0340061,0,0},
  {0x22cc118a0c118aa0,0x9ff6ca9a2ab6a26d,0,0},
  {0x7cec6df5bea167cf,0xa0939331e8846237,0,0},
  {0x71acea2819360c35,0xa12ff8bc735d8af6,0,0},
  {0x166c36e7bb3c402f,0xa1cbfad9521bfd1b,0,0},
  {0x3b5167ee359a234e,0xa267992848eeb0c0,0,0},
  {0x9443372e20d4377c,0xa302d34959951243,0,0},
  {0xca9a8a720d4c69c,0xa39da8dcc39a38e5,0,0},
  {0xbf623cf5301a2dde,0xa4381983048ff747,0,0},
  {0x23d251cc8d7975cc,0xa4d224dcd849c5b0,0,0},
  {0x189d39ffe11aaa2b,0xa56bca8b391785db,0,0},
  {0x8c33ebf3aa8501fb,0xa6050a2f60002049,0,0},
  {0x9b3ad6e4022183d9,0xa69de36ac4fbfadc,0,0},
  {0x149f6e75993468a3,0xa73655df1f2f489e,0,0},
  {0x6b2a39f856a69781,0xa7ce612e65243291,0,0},
  {0x3463a2c2e6e9cc55,0xa86604facd04d969,0,0},
  {0x6cc14c4f53e2e82d,0xa8fd40e6ccd52ffd,0,0},
  {0xd147625fda929af8,0xa99414951aacae5e,0,0},
  {0xb714ee81b53b4b9d,0xaa2a7fa8acefdd63,0,0},
  {0xe1b3dfc4dbda9bfd,0xaac081c4ba89ba8a,0,0},
  {0xf17cee69b0d2ecde,0xab561a8cbb24f410,0,0},
  {0x1becda8089c1a94c,0xabeb49a46764fd15,0,0},
  {0xf86ba0dde982fb59,0xac800eafb91ef9a9,0,0},
  {0x44bf16268608db96,0xad146952eb9282af,0,0},
  {0x9d30d4cfeb04f1fb,0xada859327ba24151,0,0},
  {0x3d53817865422565,0xae3bddf3280c620d,0,0},
  {0xf74d099042e8f326,0xaecef739f1a2df10,0,0},
  {0xa89a9b8f726b95bf,0xaf61a4ac1b83a1de,0,0},
  {0x8c679e67fc462d51,0xaff3e5ef2b507c06,0,0},
  {0xe4cad00d5c94bcd2,0xb085baa8e966f6da,0,0},
  {0x8d8be132d576e614,0xb117227f6117f9f9,0,0},
  {0x24784f32c3e3e5bd,0xb1a81d18e0df4889,0,0},
  {0x8cc7d4bd05ffd5ae,0xb238aa1bfa9ad507,0,0},
  {0xac9f7ebbc469ef59,0xb2c8c92f83c1eb87,0,0},
  {0x5d6635109164f740,0xb35879fa959c323c,0,0},
  {0xa156468ef6c18c60,0xb3e7bc248d78802e,0,0},
  {0x4a85350f69018c55,0xb4768f550ce389fd,0,0},
};
__device__ const dint64_t C[256] = {
  {0x0,0x8000000000000000,1,0},
  {0x3031437d7eccb9df,0xffffb10b10e80e95,0,0},
  {0x38e310779edfec68,0xfffec42c7454926b,0,0},
  {0x69fff9ae0dedb047,0xfffd3964bc6275ba,0,0},
  {0xb47903f7a19f8ee2,0xfffb10b4dc96dabb,0,0},
  {0x8cc193c5d508e13f,0xfff84a1e29de8571,0,0},
  {0x43366df666fd54ff,0xfff4e5a25a8d095b,0,0},
  {0x5428ed0647c9e5d1,0xfff0e343865bbb13,0,0},
  {0x5657552366961732,0xffec4304266865d9,0,0},
  {0x53aa9423bb0adc21,0xffe704e71533c508,0,0},
  {0x7d209f32d42d864e,0xffe128ef8e9fc17a,0,0},
  {0x4fd8f038449ec436,0xffdaaf212fed72db,0,0},
  {0x664649b4d541b9c5,0xffd3977ff7bae4e9,0,0},
  {0x5595ca3f421ae09c,0xffcbe2104600a0a9,0,0},
  {0x1c676208aa3be545,0xffc38ed6dc0ef98b,0,0},
  {0xccfed60a91097c48,0xffba9dd8dc8b1e83,0,0},
  {0x421e8edaaf59453e,0xffb10f1bcb6bef1d,0,0},
  {0xd2c665c2da3e7844,0xffa6e2a58df6947d,0,0},
  {0x1e1862cca089938b,0xff9c187c6abade6a,0,0},
  {0x2dabd3195a05710f,0xff90b0a7098f6443,0,0},
  {0x519c314973ccae6b,0xff84ab2c738d6a03,0,0},
  {0x3ea4f30adda3016f,0xff780814130c893c,0,0},
  {0x1b9d5851979f28fb,0xff6ac765b39e1e19,0,0},
  {0x50a7bb6a6ee3b0f1,0xff5ce92982087867,0,0},
  {0xf668633f1ab858a,0xff4e6d680c41d0a9,0,0},
  {0xb085c1828f69296a,0xff3f542a416b0134,0,0},
  {0x27e31939e2eec09c,0xff2f9d7971ca0364,0,0},
  {0xf5971326a3540ea9,0xff1f495f4ec430d7,0,0},
  {0x1f1901544271c3f8,0xff0e57e5ead848d1,0,0},
  {0xe0abd3a9b64df725,0xfefcc917b99839a5,0,0},
  {0xec34413e87ef2740,0xfeea9cff8fa2ae54,0,0},
  {0x2f88b949a72ff96c,0xfed7d3a8a29c603b,0,0},
  {0x41390efdc726e9ef,0xfec46d1e89292cf0,0,0},
  {0xb7b6cc53c3abc817,0xfeb0696d3ae4f04d,0,0},
  {0xd3af6ee4f2101c20,0xfe9bc8a1105c22a5,0,0},
  {0xb4f70c910505e10,0xfe868ac6c3043b2e,0,0},
  {0x2907cf2b3f6feac2,0xfe70afeb6d33d6a2,0,0},
  {0xd54faa364b7da8f6,0xfe5a381c8a1aa224,0,0},
  {0x87b8875373a818a4,0xfe432367f5b90a62,0,0},
  {0x8598c2c429caf7,0xfe2b71dbecd7aefc,0,0},
  {0x90cd1d959db674ef,0xfe1323870cfe9a3d,0,0},
  {0x9bfe5c51e91cbdcd,0xfdfa3878546c3d28,0,0},
  {0xe276d247626a23fd,0xfde0b0bf220c2fd4,0,0},
  {0x499ddb331d19539d,0xfdc68c6b356db62f,0,0},
  {0xfac7397cc07a6470,0xfdabcb8caeba091b,0,0},
  {0xd6e270740a186977,0xfd906e340eaa6401,0,0},
  {0x61beb8cd2696fc78,0xfd747472367dd6c5,0,0},
  {0x6c696582f346fd91,0xfd57de5867eedc39,0,0},
  {0xeae6bd951c1dabbe,0xfd3aabf84528b50b,0,0},
  {0x863b87258f11ad7e,0xfd1cdd63d0bc8735,0,0},
  {0xa06fab9f9d106709,0xfcfe72ad6d9641f2,0,0},
  {0xa4e064308f4999f4,0xfcdf6be7def1464c,0,0},
  {0xa3e22b4d38917e73,0xfcbfc926484cd43a,0,0},
  {0x5d582cac7cb4391c,0xfc9f8a7c2d603c60,0,0},
  {0x2880268f2e62955,0xfc7eaffd720ed673,0,0},
  {0x1c0d254b6c8da4bd,0xfc5d39be5a5bbc4b,0,0},
  {0x256778ffcb5c1769,0xfc3b27d38a5d49ab,0,0},
  {0x9433b49289417ea2,0xfc187a52063060c2,0,0},
  {0x25aafd7fdba12c5f,0xfbf5314f31eb7375,0,0},
  {0x7190c94899dff1b8,0xfbd14ce0d191516e,0,0},
  {0xe63ae8632b84473c,0xfbaccd1d0903bb09,0,0},
  {0x75df66f0ec3dd459,0xfb87b21a5bf5b917,0,0},
  {0x61ce9d5ef5a81487,0xfb61fbefadddb985,0,0},
  {0xb4b54683879c9c17,0xfb3baab441e770f7,0,0},
  {0x2172a361fd2a722f,0xfb14be7fbae58156,0,0},
  {0x2079880c450348ac,0xfaed376a1b42e559,0,0},
  {0x4a188aa367f90ab1,0xfac5158bc4f4211f,0,0},
  {0x10655ecd5cc771d8,0xfa9c58fd796837d4,0,0},
  {0x1fe196a53fb5b237,0xfa7301d859796671,0,0},
  {0xd24377c77a591e24,0xfa491035e55da3a3,0,0},
  {0x431c393c7f62da65,0xfa1e842ffc96e4e0,0,0},
  {0xba5dbf4510eddc8f,0xf9f35de0dde328ab,0,0},
  {0x4504ae08d19b2980,0xf9c79d63272c4628,0,0},
  {0x78685d850f80ecdc,0xf99b42d1d57781eb,0,0},
  {0x80e8c17bf80e8f02,0xf96e4e4844d4e82a,0,0},
  {0xc0e2a1352ed7f292,0xf940bfe2304e6c45,0,0},
  {0x68fc6e4d6a920bd2,0xf91297bbb1d6cdbe,0,0},
  {0x9701914c7f8fbcd7,0xf8e3d5f1423842a0,0,0},
  {0xac9f07f54ff5bc14,0xf8b47a9fb902e76c,0,0},
  {0xb36a9dfaadafc1e1,0xf88485e44c7af48a,0,0},
  {0xc7adc6b4988891bb,0xf853f7dc9186b952,0,0},
  {0xa776175bd284fe05,0xf822d0a67b9c5cb5,0,0},
  {0xa76f7efc19aed41c,0xf7f110605caf6390,0,0},
  {0x730785813f78aa1e,0xf7beb728e51dfcb8,0,0},
  {0x214cffcee9dd33ca,0xf78bc51f239e12c6,0,0},
  {0x4becad887680c197,0xf7583a62852a23b2,0,0},
  {0xf99107e50d631330,0xf7241712d4edde49,0,0},
  {0x50ca117eb18beed7,0xf6ef5b503c328589,0,0},
  {0x2c791f59cc1ffc23,0xf6ba073b424b19e8,0,0},
  {0xce8c455197cdf8a7,0xf6841af4cc8048a4,0,0},
  {0x119d358de0493956,0xf64d969e1dfc2119,0,0},
  {0x9dc7e5954c5a8f24,0xf6167a58d7b59026,0,0},
  {0xc8c615e72768d6b5,0xf5dec646f85ba1c6,0,0},
  {0xed0dd4bf62edd13f,0xf5a67a8adc4088ca,0,0},
  {0x275a2bbb2bab6c8a,0xf56d97473d446cda,0,0},
  {0x8da64484aaa0febc,0xf5341c9f32bffeb9,0,0},
  {0x163c5c7f03b718c5,0xf4fa0ab6316ed2ec,0,0},
  {0x890ac4aafa6a37bf,0xf4bf61b00b5982b7,0,0},
  {0xf8f9d3b87d11fd52,0xf48421b0efbf939b,0,0},
  {0x667e06866c07c369,0xf4484add6b01254b,0,0},
  {0x5019794a1f5896e5,0xf40bdd5a6688662f,0,0},
  {0x18ef535a7ffa7a3d,0xf3ced94d28b2ce8a,0,0},
  {0x50f29b4b49f31c37,0xf3913edb54ba2242,0,0},
  {0xd981acdcf6bc3e4,0xf3530e2aea9d3966,0,0},
  {0xa5486bdc455d56a2,0xf314476247088f74,0,0},
  {0x431be53f92ece9e6,0xf2d4eaa8233e997d,0,0},
  {0xebadcdbf915e8f6c,0xf294f82394ffe320,0,0},
  {0xaf0eed81e8c51e55,0xf2546ffc0e72f286,0,0},
  {0xe7112e89103cc0c7,0xf21352595e0bf350,0,0},
  {0x844e6a35ddc2b713,0xf1d19f63ae7428a2,0,0},
  {0x8f6bac72988088b0,0xf18f574386712643,0,0},
  {0x2730081c758fb42b,0xf14c7a21c8cbd0f4,0,0},
  {0x67127db35b287316,0xf1090827b43725fd,0,0},
  {0xc4e557b119ef3185,0xf0c5017ee336ca0f,0,0},
  {0x973ea9903ed5125f,0xf08066514c055f7e,0,0},
  {0x992d39ec5c561d28,0xf03b36c9407aa3e8,0,0},
  {0x62aef7b55319d1d4,0xeff573116df1555d,0,0},
  {0xf03a18a5e16ab641,0xefaf1b54dd2cdf0f,0,0},
  {0x767c0e8ad33bc085,0xef682fbef23ecda6,0,0},
  {0xe2398bf0eeb28cde,0xef20b07b6c6c0b37,0,0},
  {0x86f8c20fb664b01b,0xeed89db66611e307,0,0},
  {0xa1d2c3d018a9279f,0xee8ff79c548acd0f,0,0},
  {0x7872773830d368be,0xee46be5a0813016b,0,0},
  {0xfee6a1eebfa13b4a,0xedfcf21cabacd3b1,0,0},
  {0x11815196b9fbf5df,0xedb29311c504d652,0,0},
  {0x7289102076a125e5,0xed67a1673455c601,0,0},
  {0xddffe98c4f8aa031,0xed1c1d4b344c3d4f,0,0},
  {0xa8392eb238578ab0,0xecd006ec59ea306f,0,0},
  {0x7e610231ac1d6181,0xec835e79946a3145,0,0},
  {0x278047ae3dd0889,0xec3624222d227bd1,0,0},
  {0x1e99ccb9adc62ca6,0xebe85815c767cb00,0,0},
  {0xdae311e656e0661,0xeb99fa84606ff5ff,0,0},
  {0x39e39c6c2ab3655d,0xeb4b0b9e4f345617,0,0},
  {0x3383bbb5156bf1d7,0xeafb8b944453f52f,0,0},
  {0x24db98ad3a0647a1,0xeaab7a9749f584fe,0,0},
  {0x4a0ca5ea449b1c83,0xea5ad8d8c3a91f05,0,0},
  {0x15ad45b4a1b5e823,0xea09a68a6e49cd62,0,0},
  {0xcd24d4bd1056c826,0xe9b7e3de5fdedc8b,0,0},
  {0x89a92b199adfbafa,0xe9659107077cf60f,0,0},
  {0xacb1c26a06e5ae02,0xe912ae372d27045d,0,0},
  {0xf8972affb3d98e1f,0xe8bf3ba1f1aedfbb,0,0},
  {0x9fec1e78c4376186,0xe86b397ace95c46f,0,0},
  {0xbfe8378abfb87b6f,0xe816a7f595ec9232,0,0},
  {0xdbfb0fe56c6f80fe,0xe7c187467233d508,0,0},
  {0x125129529d48a92f,0xe76bd7a1e63b9786,0,0},
  {0xe2ba81b9ce96e02e,0xe715993ccd02fe9c,0,0},
  {0x82fcedb4c6434d76,0xe6becc4c5997af06,0,0},
  {0xdd2a3e32c3859960,0xe667710616f4fc59,0,0},
  {0x7613b68f6ab03130,0xe60f879fe7e2e1e5,0,0},
  {0x9b695cd67c93bd79,0xe5b7105006d4c560,0,0},
  {0x5a7c210a3a15e7ea,0xe55e0b4d05c80388,0,0},
  {0xe1f5a58c80292554,0xe50478cdce2246bc,0,0},
  {0x122785ae67f5515d,0xe4aa5909a08fa7b4,0,0},
  {0x20d63b5b9e3cd6ac,0xe44fac3814e09856,0,0},
  {0x56992551ae074e99,0xe3f4729119e798d9,0,0},
  {0xd1197dc12c63176,0xe398ac4cf556b732,0,0},
  {0x36563e2ffad8351a,0xe33c59a4439cd8ec,0,0},
  {0xd6fe4dd22e60a4a2,0xe2df7acff7c2cf83,0,0},
  {0xfd39138aa2d508ed,0xe28210095b483751,0,0},
  {0xe0521df01a1be6f5,0xe224198a0e002123,0,0},
  {0xf4e8a8372f8c5810,0xe1c5978c05ed8691,0,0},
  {0xe2f9d4600f4d0325,0xe1668a498f1f892c,0,0},
  {0x6ba8a9d9ba877899,0xe106f1fd4b8d7c96,0,0},
  {0x6d6c98fe79817946,0xe0a6cee232f2bb9c,0,0},
  {0x55ff6038a5197367,0xe046213392aa486c,0,0},
  {0x720588ff6547d884,0xdfe4e92d0d8a37f5,0,0},
  {0xab01350f013d78dd,0xdf83270a9bbee890,0,0},
  {0x64a58b2f103485dd,0xdf20db088aa60404,0,0},
  {0x4b19aa71fec3ae6d,0xdebe05637ca94cfb,0,0},
  {0x4248f15548f69ca,0xde5aa65869193805,0,0},
  {0xd597b10a01676659,0xddf6be249c075037,0,0},
  {0x739c45b982193b5e,0xdd924d05b620678a,0,0},
  {0x49c6e0ea76cbcaac,0xdd2d5339ac8692fd,0,0},
  {0xb2069fd0b482b4e8,0xdcc7d0fec8aaf2aa,0,0},
  {0xaca8017e375b64e5,0xdc61c693a82745d5,0,0},
  {0xccb7fd40d543f4a1,0xdbfb34373c974b0e,0,0},
  {0x2c19b63253da43fc,0xdb941a28cb71ec87,0,0},
  {0x5a98479cbef2ecbc,0xdb2c78a7ede238a9,0,0},
  {0x5b267c1bcff0ab62,0xdac44ff490a02710,0,0},
  {0xe257bde73d83dc1a,0xda5ba04ef3c929f4,0,0},
  {0x28e81dcb6dab91ac,0xd9f269f7aab88c29,0,0},
  {0xc4e4dc69fc2fff6f,0xd988ad2f9bdf9bbb,0,0},
  {0x1bb35ad6d2e74b67,0xd91e6a38009da15a,0,0},
  {0x1ed1a8ff78f1b632,0xd8b3a1526517a48b,0,0},
  {0x24b9fe00663574a4,0xd84852c0a80ffcdb,0,0},
  {0xced12d2899b803db,0xd7dc7ec4fabdb011,0,0},
  {0xcb78e80e67ba1b8,0xd77025a1e0a39d8b,0,0},
  {0x6cb3bfd65b38562b,0xd703479a2f6776cc,0,0},
  {0x83f082b570611d7,0xd695e4f10ea88570,0,0},
  {0x7afbefc05e9f7d99,0xd627fde9f7d63e7e,0,0},
  {0x7190b755535d4f18,0xd5b992c8b606a351,0,0},
  {0x7d00ae97abaa4096,0xd54aa3d165cc7018,0,0},
  {0xf630e8b6dac83e69,0xd4db3148750d1819,0,0},
  {0xdc4663a3168698d2,0xd46b3b72a2d68fc9,0,0},
  {0xb77d4f6bd0ee8591,0xd3fac294ff34e4d0,0,0},
  {0xa8faac741a6394dc,0xd389c6f4eb07a41c,0,0},
  {0xeeeaddb72f00e0dd,0xd31848d817d70e16,0,0},
  {0x4300fd1c1ce507e5,0xd2a6488487a91918,0,0},
  {0x981ba7e42537275f,0xd233c6408cd64236,0,0},
  {0xda7485a5aeffeb4c,0xd1c0c252c9de2c86,0,0},
  {0x744fea20e8abef92,0xd14d3d02313c0eed,0,0},
  {0x77a18eb13d2ecde5,0xd0d93696053af098,0,0},
  {0x6b8a685f6cb61c21,0xd064af55d7c9b43e,0,0},
  {0xdaf200dd81212d10,0xcfefa7898a4ef23c,0,0},
  {0xdfcb60445c1bf973,0xcf7a1f794d7ca1b1,0,0},
  {0x4d27090f10c454e,0xcf04176da12390ac,0,0},
  {0xf5babff66def7892,0xce8d8faf5406ab8b,0,0},
  {0x93e391861a034684,0xce16888783ae13b3,0,0},
  {0x23af31db7179a4aa,0xcd9f023f9c3a059e,0,0},
  {0x649474e36b8db9d3,0xcd26fd2158358e7d,0,0},
  {0x83e907fbd7aaf0b0,0xccae7976c0691177,0,0},
  {0xf839ce18e08bfb50,0xcc35778a2bac9ca1,0,0},
  {0x70cbb7f3343451be,0xcbbbf7a63eba0dd5,0,0},
  {0x2293661be51140ab,0xcb41fa15ebff0777,0,0},
  {0xd9944be1631846d8,0xcac77f24736eb553,0,0},
  {0x5328edeb3e6784de,0xca4c871d625361a9,0,0},
  {0x8335241be1693225,0xc9d1124c931fda7a,0,0},
  {0x83b0e96e1249c2b0,0xc95520fe2d40a74b,0,0},
  {0xb562c00b34ee771,0xc8d8b37ea4ed0f62,0,0},
  {0x65862939b83382e0,0xc85bca1abaf7f0a7,0,0},
  {0x2b31bc86877fd2c,0xc7de651f7ca06749,0,0},
  {0xd5c149509e9059f1,0xc76084da43624634,0,0},
  {0xcfe6c1b1a6b4e2a4,0xc6e22998b4c6608e,0,0},
  {0xe993503baf5afb41,0xc66353a8c232a43c,0,0},
  {0x43da25d99267326b,0xc5e40358a8ba05a7,0,0},
  {0xab4906075507e74,0xc56438f6f0ec3cca,0,0},
  {0xdd40950cf1ed92fa,0xc4e3f4d26ea553b6,0,0},
  {0x9dd768f30ca8e85c,0xc463373a40dd06a3,0,0},
  {0xa87e78136665cdb2,0xc3e2007dd175f5a4,0,0},
  {0x8ac9e1386e4cbabb,0xc36050ecd50ca830,0,0},
  {0x74c8f010d986a9e0,0xc2de28d74ac6628b,0,0},
  {0xb7041e9bc8c18b0d,0xc25b888d7c1fcd38,0,0},
  {0xbdf0715cb8b20bd7,0xc1d8705ffcbb6e90,0,0},
  {0x17858573216e0a22,0xc154e09faa2ff69a,0,0},
  {0x2bda5328933c854a,0xc0d0d99dabd65d44,0,0},
  {0x6dd06968e0ed1957,0xc04c5bab7297d322,0,0},
  {0xe4e62d86dd136e78,0xbfc7671ab8bb84c6,0,0},
  {0xd46655d6b012455,0xbf41fc3d81b430db,0,0},
  {0x2715ef03f8543355,0xbebc1b6619ed9116,0,0},
  {0x29d7f7b67d43b177,0xbe35c4e716999630,0,0},
  {0xac85320f528d6d5d,0xbdaef913557d76f0,0,0},
  {0x2ea36923d5d8e213,0xbd27b83dfcbe9279,0,0},
  {0x4a48496734be336d,0xbca002ba7aaf25ea,0,0},
  {0x727c405ffc73af56,0xbc17d8dc859ad583,0,0},
  {0xfce8d84068e825b6,0xbb8f3af81b93095c,0,0},
  {0x5120e35e1c1a250c,0xbb062961823b1ddc,0,0},
  {0x33201477347447d8,0xba7ca46d46946802,0,0},
  {0x39db32d014440024,0xb9f2ac703cca0db3,0,0},
  {0x9de1e3b22b8bf4db,0xb96841bf7ffcb21a,0,0},
  {0xa726f4f0828585c9,0xb8dd64b0720df647,0,0},
  {0x1c041d1ea5fb3fdb,0xb8521598bb6bce26,0,0},
  {0x2e7a35723f3ed035,0xb7c654ce4adba9f2,0,0},
  {0x7f86f63bb23f496a,0xb73a22a755457448,0,0},
  {0xeb2d28ef943dc88c,0xb6ad7f7a557e64f2,0,0},
  {0xea7c015f12b987f7,0xb6206b9e0c13a892,0,0},
  {0x737dd2824b608d13,0xb592e7697f14dd4a,0,0},
};
__device__ const double PSfast[] = {
  0x1.921fb54442d18p+2, 0x1.1a62645446203p-52,
  -0x1.4abbce625be53p5,
  0x1.466bc678d8d63p6,
  -0x1.331554ca19669p6,
};
__device__ const double PCfast[] = {
  0x1p+0, -0x1.923015cp-77,
  -0x1.3bd3cc9be45dep4,
  0x1.03c1f080ad892p6,
  -0x1.55a5c590f9e6ap6,
};
__device__ const dint64_t PS[] = {
  {0xc4c6628b80dc1cd1,0xc90fdaa22168c234,3,0},
  {0x5dc72f712aa57db4,0xa55de7312df295f5,6,1},
  {0x3f33be0021aa54d2,0xa335e33bad570e92,7,0},
  {0xe59d6ab8509a2025,0x9969667315ec2d9d,7,1},
  {0x7d5f8f76fa7d74ed,0xa83c1a43bf1c6485,6,0},
  {0xa7f0339113b8b3c5,0xf16ab2898eae62f9,4,1},
};
__device__ const dint64_t PC[] = {
  {0x0,0x8000000000000000,1,0},
  {0x56e26cd9808c1949,0x9de9e64df22ef2d2,5,1},
  {0x9980f00630cb655e,0x81e0f840dad61d9a,7,0},
  {0xa508509534006249,0xaae9e3f1e5ffcfe2,7,1},
  {0xe0603ce7044eeba,0xf0fa83448dd1e094,6,0},
  {0xec63157807ebffa,0xd368f6f4207cfe49,5,1},
};
__device__ const double SC[256][3] = {
   {0x0p+0, 0x0p+0, 0x1p+0},
   {-0x1.c0f6cp-35, 0x1.921f892b900fep-9, 0x1.ffff621623fap-1},
   {-0x1.9c7935ep-35, 0x1.921f0ea27ce01p-8, 0x1.fffd8858eca2ep-1},
   {-0x1.d14d1acp-34, 0x1.2d96af779b0bbp-7, 0x1.fffa72c986392p-1},
   {-0x1.dba8f6a8p-33, 0x1.921d1ce2d0a1cp-7, 0x1.fff62169dddaap-1},
   {0x1.a6b7cdfp-32, 0x1.f6a29bdb7377p-7, 0x1.fff0943c02419p-1},
   {0x1.b49618dp-33, 0x1.2d936d1506f3dp-6, 0x1.ffe9cb44829cp-1},
   {-0x1.398d6fcp-35, 0x1.5fd4d1e21de6dp-6, 0x1.ffe1c687174b1p-1},
   {-0x1.e9e9a8c8p-31, 0x1.9215597791e0ap-6, 0x1.ffd886097afcfp-1},
   {-0x1.34e844cp-32, 0x1.c454f2e9480c7p-6, 0x1.ffce09ce95933p-1},
   {-0x1.989a8a4p-32, 0x1.f693709b94f92p-6, 0x1.ffc251dfbac0cp-1},
   {0x1.04a9b99p-30, 0x1.146860e69a571p-5, 0x1.ffb55e40a5c43p-1},
   {-0x1.56947cp-36, 0x1.2d865748774adp-5, 0x1.ffa72efff95d1p-1},
   {-0x1.c348768p-35, 0x1.46a396d34121ap-5, 0x1.ff97c420a8451p-1},
   {0x1.9e80552p-32, 0x1.5fc00e6e4c65cp-5, 0x1.ff871dacd8761p-1},
   {0x1.3f11d74p-34, 0x1.78dbaa97099ebp-5, 0x1.ff753bb18af95p-1},
   {0x1.c039af4p-33, 0x1.91f65fc0abc0ap-5, 0x1.ff621e370ca7ap-1},
   {0x1.53e1f8p-35, 0x1.ab101bf74ac2ep-5, 0x1.ff4dc54b00181p-1},
   {0x1.114a649p-29, 0x1.c428d7de920e9p-5, 0x1.ff3830f2e9043p-1},
   {0x1.adf0ef4p-31, 0x1.dd40723a3cdfbp-5, 0x1.ff21614b9d9adp-1},
   {-0x1.d21f5918p-30, 0x1.f656e1e9e59cdp-5, 0x1.ff09565e83d77p-1},
   {-0x1.4f54d708p-30, 0x1.07b612d6be078p-4, 0x1.fef0102c634e3p-1},
   {-0x1.1efec9ap-30, 0x1.1440118ba7bdp-4, 0x1.fed58ecf342dap-1},
   {0x1.cc17ba88p-29, 0x1.20c96cf0a7eedp-4, 0x1.feb9d24646fa6p-1},
   {0x1.121dbe4p-33, 0x1.2d5209628edfp-4, 0x1.fe9cdacf99cffp-1},
   {-0x1.9ecf61p-34, 0x1.39d9f103bf7f7p-4, 0x1.fe7ea854e6b08p-1},
   {-0x1.04ede8ep-31, 0x1.466116c629e5cp-4, 0x1.fe5f3af4ee201p-1},
   {-0x1.1821cecp-31, 0x1.52e773c9920c7p-4, 0x1.fe3e92c0e4108p-1},
   {0x1.cdec726p-31, 0x1.5f6d02131f0b2p-4, 0x1.fe1cafc7f1a24p-1},
   {-0x1.edece4dp-31, 0x1.6bf1b2653648cp-4, 0x1.fdf99233c230cp-1},
   {-0x1.2aa4d1cp-31, 0x1.787585bc45f0fp-4, 0x1.fdd53a01d11d9p-1},
   {0x1.d461592p-32, 0x1.84f871e32cf68p-4, 0x1.fdafa74f16482p-1},
   {0x1.f0cbd728p-29, 0x1.917a71d3d2956p-4, 0x1.fd88da29f302ep-1},
   {-0x1.583247p-30, 0x1.9dfb6c9865b06p-4, 0x1.fd60d2e14a6b1p-1},
   {-0x1.2e81bf4p-30, 0x1.aa7b706bfdbbap-4, 0x1.fd3791484ff5p-1},
   {-0x1.13941418p-28, 0x1.b6fa680a05c27p-4, 0x1.fd0d15a4b8471p-1},
   {0x1.71098ffp-30, 0x1.c3785eba12b42p-4, 0x1.fce15fceddccfp-1},
   {-0x1.c3519e8p-32, 0x1.cff53302f059p-4, 0x1.fcb4703b969e1p-1},
   {0x1.2f522a5p-27, 0x1.dc70fb84af16ep-4, 0x1.fc8646987fc1dp-1},
   {-0x1.ae9bed8p-33, 0x1.e8eb7f8a589e2p-4, 0x1.fc56e3b91ca3ap-1},
   {0x1.f8868b2p-30, 0x1.f564e87d2330fp-4, 0x1.fc264701f9a09p-1},
   {-0x1.b07985f8p-29, 0x1.00ee8835051f4p-3, 0x1.fbf47105f7439p-1},
   {0x1.cbdaa94p-30, 0x1.072a05e1d4d8ep-3, 0x1.fbc16172a9e36p-1},
   {0x1.37c5b908p-28, 0x1.0d64df9619f0dp-3, 0x1.fb8d18b635327p-1},
   {-0x1.068b5fc8p-28, 0x1.139f09bc617f5p-3, 0x1.fb5797351da85p-1},
   {-0x1.8ea66818p-29, 0x1.19d8919fa4ec8p-3, 0x1.fb20dc7da8affp-1},
   {0x1.6278ceb8p-28, 0x1.2011719d50b87p-3, 0x1.fae8e8bd4427fp-1},
   {-0x1.096df84p-29, 0x1.264993433763ap-3, 0x1.faafbcbfca356p-1},
   {0x1.9b2534fp-29, 0x1.2c810967bbf7p-3, 0x1.fa7557d8d987ep-1},
   {0x1.215b4ep-34, 0x1.32b7bfa25c91bp-3, 0x1.fa39bac71954bp-1},
   {-0x1.94db891p-30, 0x1.38edb9d29b39dp-3, 0x1.f9fce56700a6dp-1},
   {0x1.7727f7b8p-29, 0x1.3f22f7c3cce3ap-3, 0x1.f9bed7b8c8d8cp-1},
   {-0x1.0cb33038p-29, 0x1.45576971dd53p-3, 0x1.f97f925d53c83p-1},
   {-0x1.9071106p-31, 0x1.4b8b175c71e22p-3, 0x1.f93f14feb8022p-1},
   {0x1.62741e78p-29, 0x1.51bdfa7ea30d5p-3, 0x1.f8fd5fe3efac8p-1},
   {0x1.f8e16d0cp-28, 0x1.57f00e80e6e12p-3, 0x1.f8ba733a1ceb1p-1},
   {-0x1.76acbcap-31, 0x1.5e2143b7bc1c2p-3, 0x1.f8764fad5e9bfp-1},
   {-0x1.0a0f73ap-30, 0x1.6451a76411746p-3, 0x1.f830f4ad232d8p-1},
   {0x1.ca11d1bcp-28, 0x1.6a8135d7bd143p-3, 0x1.f7ea625eb5af7p-1},
   {-0x1.02f23628p-29, 0x1.70afd74071191p-3, 0x1.f7a299d3f182ap-1},
   {0x1.b34dcb8p-29, 0x1.76dda08544b5cp-3, 0x1.f7599a1ac7ecdp-1},
   {0x1.161ff4p-32, 0x1.7d0a7bf2d4abap-3, 0x1.f70f64322da74p-1},
   {-0x1.c49b8b4p-31, 0x1.83366ddb3de23p-3, 0x1.f6c3f7e7c2707p-1},
   {0x1.21da851p-29, 0x1.8961743b1429p-3, 0x1.f6775552a6ba2p-1},
   {0x1.ac63edap-30, 0x1.8f8b851098588p-3, 0x1.f6297cef0cdd6p-1},
   {0x1.27ef489cp-27, 0x1.95b4a5b9f2cebp-3, 0x1.f5da6e7820551p-1},
   {0x1.ae8937p-30, 0x1.9bdcc07900146p-3, 0x1.f58a2b0689c82p-1},
   {0x1.eb48c7ep-29, 0x1.a203e4a4f950ep-3, 0x1.f538b1d392049p-1},
   {-0x1.bfd282fp-29, 0x1.a829ffaad0d79p-3, 0x1.f4e603d51f1aap-1},
   {0x1.7ccf638p-29, 0x1.ae4f1fa80e1b5p-3, 0x1.f492204c5ef9ep-1},
   {-0x1.2435c578p-28, 0x1.b4732b72ebc86p-3, 0x1.f43d0890e1e72p-1},
   {0x1.0293fecp-30, 0x1.ba9634155f866p-3, 0x1.f3e6bbb6c2ea4p-1},
   {-0x1.7bb1f92p-29, 0x1.c0b82461f65ep-3, 0x1.f38f3ae6f9afcp-1},
   {0x1.27aaebcp-29, 0x1.c6d906faacf65p-3, 0x1.f3368589e17a2p-1},
   {-0x1.2e2bcd5p-27, 0x1.ccf8c3f74a6c9p-3, 0x1.f2dc9cfb5fa74p-1},
   {-0x1.6f070acp-30, 0x1.d31773ba218a8p-3, 0x1.f2817fd4d045bp-1},
   {0x1.469adfcp-29, 0x1.d935004779e57p-3, 0x1.f2252f59c122dp-1},
   {0x1.4f51c18p-32, 0x1.df5164301377ap-3, 0x1.f1c7abdeaa3efp-1},
   {0x1.78e44dap-29, 0x1.e56ca4202807cp-3, 0x1.f168f51c5d5d5p-1},
   {0x1.49bb5f8p-32, 0x1.eb86b4a1b7e9bp-3, 0x1.f1090bc4b68p-1},
   {-0x1.67ba541p-28, 0x1.f19f9369d5e93p-3, 0x1.f0a7effdc937fp-1},
   {0x1.c0cab95p-29, 0x1.f7b74ab7219d2p-3, 0x1.f045a1219e594p-1},
   {-0x1.2b77e32p-30, 0x1.fdcdc0ca3288dp-3, 0x1.efe220cf5c751p-1},
   {-0x1.e0d8cbp-33, 0x1.01f18054c8362p-2, 0x1.ef7d6e54c347dp-1},
   {-0x1.ecd5b9cp-29, 0x1.04fb7f6d35d68p-2, 0x1.ef178a6f9a987p-1},
   {0x1.eb24de5p-29, 0x1.0804e1d369ff2p-2, 0x1.eeb074934fdfp-1},
   {0x1.4a897c4p-30, 0x1.0b0d9d7b0d042p-2, 0x1.ee482e14bcdep-1},
   {0x1.336c376p-30, 0x1.0e15b555e7becp-2, 0x1.eddeb6908ca8cp-1},
   {-0x1.3952d9p-31, 0x1.111d25efd48b8p-2, 0x1.ed740e7eb8dd6p-1},
   {0x1.fc2a5d4p-31, 0x1.1423ef5c7e1bdp-2, 0x1.ed0835dc24e89p-1},
   {0x1.a88ed37p-29, 0x1.172a0eb8361dap-2, 0x1.ec9b2d0ec8288p-1},
   {-0x1.8ca4cb94p-27, 0x1.1a2f7b10b6d7p-2, 0x1.ec2cf55d6117cp-1},
   {0x1.0144524p-27, 0x1.1d3446fd0cd3fp-2, 0x1.ebbd8c1d62f96p-1},
   {-0x1.abf810cp-28, 0x1.203855b85f89ap-2, 0x1.eb4cf57454132p-1},
   {0x1.5d4c5d58p-28, 0x1.233bbcca40561p-2, 0x1.eadb2e40746cap-1},
   {-0x1.a1b0c58p-29, 0x1.263e685b1d714p-2, 0x1.ea68396d87754p-1},
   {-0x1.77c8dacp-29, 0x1.294061d2eb611p-2, 0x1.e9f41597393c8p-1},
   {0x1.915540ep-30, 0x1.2c41a580014cfp-2, 0x1.e97ec348fb87fp-1},
   {-0x1.abb6d9bp-28, 0x1.2f422b2d0990cp-2, 0x1.e90843c55b996p-1},
   {-0x1.b8ee5d58p-28, 0x1.3241f8cea2836p-2, 0x1.e890962268c49p-1},
   {-0x1.1cd29828p-28, 0x1.35410a8396266p-2, 0x1.e817baf85c094p-1},
   {-0x1.e216afp-32, 0x1.383f5e08283e2p-2, 0x1.e79db2a188b0ap-1},
   {-0x1.24afc3p-31, 0x1.3b3cef6993c0bp-2, 0x1.e7227dbf82004p-1},
   {-0x1.aa1657cp-31, 0x1.3e39be4767224p-2, 0x1.e6a61c62d5274p-1},
   {-0x1.c5b65fap-30, 0x1.4135c898485bbp-2, 0x1.e6288ee07fea5p-1},
   {0x1.23e8978p-32, 0x1.44310de3c284bp-2, 0x1.e5a9d54bbd26cp-1},
   {-0x1.2b1d77ap-29, 0x1.472b8976d498dp-2, 0x1.e529f06cb187dp-1},
   {-0x1.daaa348p-31, 0x1.4a253cb97efd1p-2, 0x1.e4a8e007231a2p-1},
   {-0x1.322f5708p-28, 0x1.4d1e2260c3422p-2, 0x1.e426a500f6e33p-1},
   {0x1.64758e8p-29, 0x1.50163eca0b337p-2, 0x1.e3a33e996b722p-1},
   {0x1.12486278p-28, 0x1.530d89a17e007p-2, 0x1.e31eae3fb917bp-1},
   {-0x1.6c3416ccp-27, 0x1.5603fcf8cd8a3p-2, 0x1.e298f502a579bp-1},
   {0x1.ab481ffp-29, 0x1.58f9a896aa209p-2, 0x1.e2121016e14fcp-1},
   {-0x1.6eb838bp-29, 0x1.5bee77aaf890bp-2, 0x1.e18a032eb4df5p-1},
   {-0x1.d159b8p-32, 0x1.5ee2734efeef5p-2, 0x1.e100ccaa6bd78p-1},
   {-0x1.a42e4ap-34, 0x1.61d595bedeabcp-2, 0x1.e0766d944915ep-1},
   {-0x1.43d0dcp-30, 0x1.64c7dd5cc0cd1p-2, 0x1.dfeae63903034p-1},
   {-0x1.8c7bdb7p-27, 0x1.67b9453ca2122p-2, 0x1.df5e378482eaep-1},
   {0x1.1c0ead6p-30, 0x1.6aa9d844c980ap-2, 0x1.ded05f6a23a52p-1},
   {0x1.7d526p-31, 0x1.6d99867e90d92p-2, 0x1.de4160e97b2e2p-1},
   {0x1.924e0368p-28, 0x1.7088555d3c816p-2, 0x1.ddb13afb14e37p-1},
   {-0x1.74b7c3ep-30, 0x1.73763c09fba09p-2, 0x1.dd1fef5335416p-1},
   {-0x1.7943adp-30, 0x1.766340685c982p-2, 0x1.dc8d7ccf2567ap-1},
   {0x1.79dd614p-29, 0x1.794f5f7522b88p-2, 0x1.dbf9e402aa5c3p-1},
   {0x1.7b64f32p-30, 0x1.7c3a939c32d81p-2, 0x1.db652607e0db1p-1},
   {-0x1.2bea5ce8p-28, 0x1.7f24db825141cp-2, 0x1.dacf43268b5bp-1},
   {0x1.733c024p-30, 0x1.820e3b8bf15ap-2, 0x1.da383a7aed887p-1},
   {-0x1.eac0fc94p-27, 0x1.84f6a51d077b3p-2, 0x1.d9a00efd84537p-1},
   {0x1.aca37338p-27, 0x1.87de2f4704f98p-2, 0x1.d906bbf17f4dap-1},
   {-0x1.910c4fp-30, 0x1.8ac4b7dc0d986p-2, 0x1.d86c4862b5d6ep-1},
   {-0x1.33bb86p-31, 0x1.8daa52b4dc041p-2, 0x1.d7d0b0374a559p-1},
   {-0x1.69e1507p-27, 0x1.908ef408ad22p-2, 0x1.d733f5e71c3bcp-1},
   {0x1.cffacf08p-27, 0x1.9372ab7784d36p-2, 0x1.d696161d786c9p-1},
   {-0x1.8629d9fp-26, 0x1.965552b0849abp-2, 0x1.d5f7190eeae23p-1},
   {0x1.415p-30, 0x1.99371687c64f3p-2, 0x1.d556f5155d9ddp-1},
   {-0x1.bd37aad8p-27, 0x1.9c17cf40715cbp-2, 0x1.d4b5b2caf8386p-1},
   {0x1.d02cde7p-26, 0x1.9ef79ea4d995dp-2, 0x1.d4134ac5eb246p-1},
   {-0x1.10547acp-30, 0x1.a1d653d9adf5ep-2, 0x1.d36fc7d291602p-1},
   {-0x1.01a1a228p-27, 0x1.a4b40f9c0120bp-2, 0x1.d2cb22b45236bp-1},
   {0x1.3ce2bacp-29, 0x1.a790ce2056b9ap-2, 0x1.d2255c3ae11a5p-1},
   {-0x1.ccb4a6p-32, 0x1.aa6c828db4ea8p-2, 0x1.d17e774d4e3e2p-1},
   {0x1.5db4bp-29, 0x1.ad47321f29847p-2, 0x1.d0d672bc0b122p-1},
   {0x1.32f6a6ep-29, 0x1.b020d7a285e23p-2, 0x1.d02d4fb84d334p-1},
   {0x1.cf8e39bcp-26, 0x1.b2f97c27f7494p-2, 0x1.cf830c2248c5ep-1},
   {0x1.8927bbp-30, 0x1.b5d10129a750ap-2, 0x1.ced7af22cb105p-1},
   {-0x1.3dec3c1p-28, 0x1.b8a77f8d0bbc5p-2, 0x1.ce2b32e50d6cdp-1},
   {-0x1.26ba536p-28, 0x1.bb7cf08f0290dp-2, 0x1.cd7d98fcf3b1ep-1},
   {0x1.23c568ep-29, 0x1.be51524e3aa53p-2, 0x1.cccee1da3d56ep-1},
   {-0x1.f3b3afp-29, 0x1.c1249c1f5f2f6p-2, 0x1.cc1f0f95e1e24p-1},
   {-0x1.1286a47p-28, 0x1.c3f6d2ef7054bp-2, 0x1.cb6e20ff37e81p-1},
   {0x1.641214ep-29, 0x1.c6c7f594003d9p-2, 0x1.cabc165bf1b6p-1},
   {0x1.0cda7c9p-27, 0x1.c997ff2bffccbp-2, 0x1.ca08f0dee434cp-1},
   {-0x1.5557ac9p-28, 0x1.cc66e7b42e8f1p-2, 0x1.c954b28bca62ep-1},
   {0x1.555eb62p-28, 0x1.cf34bccc567a1p-2, 0x1.c89f57f6e20f3p-1},
   {-0x1.4e0e361p-28, 0x1.d2016cbb5e39ap-2, 0x1.c7e8e59999e1fp-1},
   {0x1.446da1ep-29, 0x1.d4cd039d0ed05p-2, 0x1.c731585f970ebp-1},
   {0x1.103d328p-29, 0x1.d797767638decp-2, 0x1.c678b3174afe1p-1},
   {0x1.5814d6p-28, 0x1.da60c7ae9dc22p-2, 0x1.c5bef522be6fbp-1},
   {-0x1.5e2321ep-29, 0x1.dd28f054cbb3fp-2, 0x1.c5042052c8c42p-1},
   {-0x1.a259ffep-29, 0x1.dfeff54854631p-2, 0x1.c44833611bc7dp-1},
   {-0x1.4f28d8p-31, 0x1.e2b5d34665b35p-2, 0x1.c38b2f278ea7ep-1},
   {-0x1.de571p-36, 0x1.e57a86d137f2p-2, 0x1.c2cd1493d05c2p-1},
   {0x1.e0d8d14p-29, 0x1.e83e0ffb7bfb4p-2, 0x1.c20de3a08ea07p-1},
   {-0x1.12a858ep-28, 0x1.eb0067e48baf4p-2, 0x1.c14d9e2bd511ep-1},
   {0x1.9a17403p-27, 0x1.edc19997a4431p-2, 0x1.c08c413089b2ep-1},
   {0x1.68c8636p-29, 0x1.f0819163d1bcp-2, 0x1.bfc9d21568f32p-1},
   {0x1.4cc5eb8p-29, 0x1.f3405a482e11dp-2, 0x1.bf064dd580fc9p-1},
   {-0x1.fce7cd8p-27, 0x1.f5fde8f3f11d4p-2, 0x1.be41b798f6b97p-1},
   {-0x1.af8169p-29, 0x1.f8ba4c98a9816p-2, 0x1.bd7c0b1a7f14bp-1},
   {0x1.6e39e2p-33, 0x1.fb7575d1ea75p-2, 0x1.bcb54cac5dde5p-1},
   {0x1.30f9256p-28, 0x1.fe2f665dcd168p-2, 0x1.bbed7bd1e17bp-1},
   {0x1.626de2p-31, 0x1.00740ca0d5fbbp-1, 0x1.bb2499f9fe7a3p-1},
   {0x1.5cc703p-30, 0x1.01cfc8afeea0ep-1, 0x1.ba5aa650dd495p-1},
   {-0x1.6191e6p-32, 0x1.032ae54fe4057p-1, 0x1.b98fa2065a5e6p-1},
   {-0x1.6b1485p-31, 0x1.0485624c328c8p-1, 0x1.b8c38d39737bcp-1},
   {-0x1.11fbc3ap-29, 0x1.05df3e66a716dp-1, 0x1.b7f668a580fdp-1},
   {-0x1.0eca7fp-27, 0x1.07387825589ecp-1, 0x1.b728352c44517p-1},
   {-0x1.8073bc9ep-25, 0x1.089109ef1284dp-1, 0x1.b658f630112edp-1},
   {-0x1.9dcf0adp-27, 0x1.09e9051603e29p-1, 0x1.b588a13ab750fp-1},
   {-0x1.06ea9fp-29, 0x1.0b405820e78e7p-1, 0x1.b4b740d3cc07bp-1},
   {-0x1.36a8d0cp-30, 0x1.0c9704a1ea4e5p-1, 0x1.b3e4d40f5524dp-1},
   {0x1.63d1f3p-30, 0x1.0ded0bc01a533p-1, 0x1.b3115a3a628afp-1},
   {0x1.f3181f14p-26, 0x1.0f4270e4787bfp-1, 0x1.b23cd1314c779p-1},
   {-0x1.f269b78p-29, 0x1.109723e75c5cfp-1, 0x1.b167430cfebdbp-1},
   {0x1.1d84dc08p-27, 0x1.11eb36bc9db52p-1, 0x1.b090a4915ee88p-1},
   {-0x1.08e60068p-27, 0x1.133e9ba0061d8p-1, 0x1.afb8fe69a6527p-1},
   {0x1.cda72abp-27, 0x1.14915d557a7c9p-1, 0x1.aee049bc0aeep-1},
   {-0x1.f32f95p-30, 0x1.15e36dfb6bb55p-1, 0x1.ae068f6991699p-1},
   {0x1.138092dp-28, 0x1.1734d6f34d7fp-1, 0x1.ad2bc96c1e1f5p-1},
   {0x1.6b382dd4p-26, 0x1.188595ae376a5p-1, 0x1.ac4ff962bdb6dp-1},
   {-0x1.f12fafap-28, 0x1.19d59f592a587p-1, 0x1.ab7326685eb57p-1},
   {-0x1.2909e5ap-28, 0x1.1b2500aed7ac6p-1, 0x1.aa954823cf815p-1},
   {-0x1.d66a8978p-25, 0x1.1c73aa0150cf9p-1, 0x1.a9b668fb0503fp-1},
   {0x1.311ea86p-27, 0x1.1dc1b7db74db1p-1, 0x1.a8d675d9c6cc8p-1},
   {-0x1.41c02b8p-31, 0x1.1f0f08a1a06a4p-1, 0x1.a7f5853bb4309p-1},
   {-0x1.ca1f4edp-26, 0x1.205ba57211271p-1, 0x1.a71391146958fp-1},
   {-0x1.910ce77p-28, 0x1.21a7988f8326bp-1, 0x1.a63092626202fp-1},
   {0x1.2bfadbeep-25, 0x1.22f2dc71afab6p-1, 0x1.a54c8cd9fd0d9p-1},
   {-0x1.5f1c02a8p-27, 0x1.243d5df4afb93p-1, 0x1.a4678dbbe5e73p-1},
   {-0x1.db12b9p-30, 0x1.2587347f493a4p-1, 0x1.a38184db0df23p-1},
   {-0x1.7b29ep-30, 0x1.26d05490f2f61p-1, 0x1.a29a7a2f40b49p-1},
   {-0x1.b3ddca4p-29, 0x1.2818be6930629p-1, 0x1.a1b26d8f070d7p-1},
   {0x1.e112744p-29, 0x1.2960730ff2bcdp-1, 0x1.a0c95e3df5e0ep-1},
   {-0x1.5269766p-28, 0x1.2aa76dafcbbf4p-1, 0x1.9fdf4fae1df6fp-1},
   {-0x1.09777e1p-28, 0x1.2bedb1b6b4e15p-1, 0x1.9ef43f6cbe162p-1},
   {0x1.ae2051fp-28, 0x1.2d333e4617f25p-1, 0x1.9e082e148680ep-1},
   {-0x1.36f6ced8p-27, 0x1.2e780cb47180ep-1, 0x1.9d1b207f383c3p-1},
   {-0x1.23fdc6bp-28, 0x1.2fbc23fba2f44p-1, 0x1.9c2d1197130a7p-1},
   {0x1.bc540ep-33, 0x1.30ff7fd6d967dp-1, 0x1.9b3e0478b961bp-1},
   {-0x1.cfb4ed7p-28, 0x1.32421da0bf0e9p-1, 0x1.9a4dfb1c89326p-1},
   {0x1.55802aecp-26, 0x1.3384042a92b1dp-1, 0x1.995cf06920d11p-1},
   {0x1.60719e4p-28, 0x1.34c52608e3a92p-1, 0x1.986aee6d6837ep-1},
   {-0x1.cbf2e48p-30, 0x1.36058ac8863b6p-1, 0x1.9777ef832c986p-1},
   {0x1.9061c32p-27, 0x1.374533ab707dp-1, 0x1.9683f2ad7e2ecp-1},
   {-0x1.da84dfep-27, 0x1.3884160f9488fp-1, 0x1.958f000fdd50ap-1},
   {0x1.92e8a74p-29, 0x1.39c23eba6b22ap-1, 0x1.94990dd9cee51p-1},
   {-0x1.bff5d9ap-29, 0x1.3affa20756bddp-1, 0x1.93a225056084ap-1},
   {0x1.4c462p-36, 0x1.3c3c4498e98ebp-1, 0x1.92aa41fbb951cp-1},
   {-0x1.e4613e9p-28, 0x1.3d782261dff62p-1, 0x1.91b167e92d706p-1},
   {0x1.0eb2964p-30, 0x1.3eb33ed579bbep-1, 0x1.90b794146043cp-1},
   {-0x1.60abec2p-29, 0x1.3fed94c834d8ap-1, 0x1.8fbcca9583479p-1},
   {0x1.6954977p-27, 0x1.4127281ddac03p-1, 0x1.8ec1085083553p-1},
   {0x1.a16fec2p-29, 0x1.425ff1f841235p-1, 0x1.8dc452ca328d3p-1},
   {-0x1.27bcdd3p-27, 0x1.4397f44aa44f2p-1, 0x1.8cc6a8771e165p-1},
   {-0x1.60dded4p-28, 0x1.44cf317a563dbp-1, 0x1.8bc8076122736p-1},
   {-0x1.9a8f405cp-26, 0x1.4605a2b02d705p-1, 0x1.8ac875232f3efp-1},
   {0x1.32777dcp-27, 0x1.473b532bc5a67p-1, 0x1.89c7e8713120cp-1},
   {-0x1.1418a7bp-26, 0x1.4870306ca20e2p-1, 0x1.88c670a0ea774p-1},
   {-0x1.fed182ep-28, 0x1.49a44886b534p-1, 0x1.87c401fdf05e5p-1},
   {0x1.86144d8p-27, 0x1.4ad796ea1410cp-1, 0x1.86c0a04dbacc5p-1},
   {0x1.1bc2e6p-33, 0x1.4c0a14640d2afp-1, 0x1.85bc51aa114c2p-1},
   {-0x1.f53d2fep-28, 0x1.4d3bc5aaa8cd5p-1, 0x1.84b7121b30a13p-1},
   {-0x1.2e100ap-30, 0x1.4e6cab91556bep-1, 0x1.83b0e0e6b6cccp-1},
   {-0x1.fa58c62p-29, 0x1.4f9cc1c69fddep-1, 0x1.82a9c1c1ab463p-1},
   {0x1.bb491ep-33, 0x1.50cc09fdcbd92p-1, 0x1.81a1b3342f858p-1},
   {0x1.a11541p-28, 0x1.51fa82c3aa029p-1, 0x1.8098b67ea8509p-1},
   {0x1.ab0a5d3p-27, 0x1.53282b20b96b6p-1, 0x1.7f8ecc791953p-1},
   {-0x1.cba0438p-28, 0x1.5454fe43a7d7cp-1, 0x1.7e83f96af78ap-1},
   {-0x1.0dd83a4p-29, 0x1.5581033a81573p-1, 0x1.7d783712e20ecp-1},
   {-0x1.e9a8299p-28, 0x1.56ac33fbb8253p-1, 0x1.7c6b8acf90fa6p-1},
   {0x1.225c4aap-29, 0x1.57d6939d4b513p-1, 0x1.7b5df1da18065p-1},
   {-0x1.82e66ep-27, 0x1.59001b9e64d79p-1, 0x1.7a4f72157cfdfp-1},
   {0x1.51a6a354p-26, 0x1.5a28d5b36d597p-1, 0x1.794002a7c9023p-1},
   {0x1.13917f4p-26, 0x1.5b50b4e10bec1p-1, 0x1.782faf6dc7ba2p-1},
   {0x1.49310ccp-30, 0x1.5c77bc15ab4efp-1, 0x1.771e75c43942ep-1},
   {0x1.24d493cp-30, 0x1.5d9dee9de49dbp-1, 0x1.760c529bc17bp-1},
   {-0x1.04638f7p-26, 0x1.5ec347044e0f4p-1, 0x1.74f94b0af972p-1},
   {-0x1.3f41b28p-29, 0x1.5fe7cb834600cp-1, 0x1.73e55936a516p-1},
   {-0x1.a5f6f5cp-30, 0x1.610b7515d1562p-1, 0x1.72d083b8214ebp-1},
   {0x1.19fb2ep-28, 0x1.622e459eafbc1p-1, 0x1.71bac8c7b0592p-1},
   {-0x1.56d2c2bp-28, 0x1.6350396fe4e62p-1, 0x1.70a42bec51665p-1},
   {-0x1.3c156c2p-28, 0x1.64715385bed93p-1, 0x1.6f8caa4969708p-1},
   {-0x1.f23e576p-29, 0x1.659191d2fd57fp-1, 0x1.6e7445d74f711p-1},
   {0x1.1e4be38p-30, 0x1.66b0f41d484c4p-1, 0x1.6d5afecd4938dp-1},
   {-0x1.397cc8d8p-27, 0x1.67cf76eac73dfp-1, 0x1.6c40d89625f63p-1},
   {-0x1.202f686p-28, 0x1.68ed1e0990551p-1, 0x1.6b25cf728c35p-1},
};
__device__ __forceinline__ void a_mul(double *hi, double *lo, double a, double b) {
  *hi = __dmul_rn(a,b);
  *lo = __fma_rn (a, b, -*hi);
}
__device__ __forceinline__ void s_mul (double *hi, double *lo, double a, double bh,
                          double bl) {
  a_mul (hi, lo, a, bh);
  *lo = __fma_rn (a, bl, *lo);
}
__device__ __forceinline__ void d_mul(double *hi, double *lo, double ah, double al,
                         double bh, double bl) {
  double s, t;
  a_mul(hi, &s, ah, bh);
  t = __fma_rn(al, bh, s);
  *lo = __fma_rn(ah, bl, t);
}
__device__ __forceinline__ void
fast_two_sum(double *hi, double *lo, double a, double b)
{
  double e;
  *hi = __dadd_rn(a,b);
  e = __dsub_rn(*hi,a);
  *lo = __dsub_rn(b,e);
}
__device__ __forceinline__ void
evalPSfast (double *h, double *l, double xh, double xl, double uh, double ul)
{
  double t;
  *h = PSfast[4];
  *h = __fma_rn (*h, uh, PSfast[3]);
  *h = __fma_rn (*h, uh, PSfast[2]);
  s_mul (h, l, *h, uh, ul);
  fast_two_sum (h, &t, PSfast[0], *h);
  *l = __dadd_rn(*l,__dadd_rn(PSfast[1],t));
  d_mul (h, l, *h, *l, xh, xl);
}
__device__ __forceinline__ void
evalPCfast (double *h, double *l, double uh, double ul)
{
  double t;
  *h = PCfast[4];
  *h = __fma_rn (*h, uh, PCfast[3]);
  *h = __fma_rn (*h, uh, PCfast[2]);
  s_mul (h, l, *h, uh, ul);
  fast_two_sum (h, &t, PCfast[0], *h);
  *l = __dadd_rn(*l,__dadd_rn(PCfast[1],t));
}
__device__ __forceinline__ void
evalPS (dint64_t *Y, dint64_t *X, dint64_t *X2)
{
  mul_dint_21 (Y, X2, PS+5);
  add_dint (Y, Y, PS+4);
  mul_dint (Y, Y, X2);
  add_dint (Y, Y, PS+3);
  mul_dint (Y, Y, X2);
  add_dint (Y, Y, PS+2);
  mul_dint (Y, Y, X2);
  add_dint (Y, Y, PS+1);
  mul_dint (Y, Y, X2);
  add_dint (Y, Y, PS+0);
  mul_dint (Y, Y, X);
}
__device__ __forceinline__ void
evalPC (dint64_t *Y, dint64_t *X2)
{
  mul_dint_21 (Y, X2, PC+5);
  add_dint (Y, Y, PC+4);
  mul_dint (Y, Y, X2);
  add_dint (Y, Y, PC+3);
  mul_dint (Y, Y, X2);
  add_dint (Y, Y, PC+2);
  mul_dint (Y, Y, X2);
  add_dint (Y, Y, PC+1);
  mul_dint (Y, Y, X2);
  add_dint (Y, Y, PC+0);
}
__device__ __forceinline__ void
normalize (dint64_t *X)
{
  int cnt;
  if (X->hi != 0)
  {
    cnt = g64_clzll (X->hi);
    if (cnt)
    {
      X->hi = (X->hi << cnt) | (X->lo >> (64 - cnt));
      X->lo = X->lo << cnt;
    }
    X->ex -= cnt;
  }
  else if (X->lo != 0)
  {
    cnt = g64_clzll (X->lo);
    X->hi = X->lo << cnt;
    X->lo = 0;
    X->ex -= 64 + cnt;
  }
}
__device__ __forceinline__ void
reduce (dint64_t *X)
{
  int e = X->ex;
  u128 u;
  if (e <= 1)
  {
    u = (u128) X->hi * (u128) T[1];
    uint64_t tiny = u;
    X->lo = u >> 64;
    u = (u128) X->hi * (u128) T[0];
    X->lo += u;
    X->hi = (u >> 64) + (X->lo < (uint64_t) u);
    e = X->ex;
    normalize (X);
    e = e - X->ex;
    if (e)
      X->lo |= tiny >> (64 - e);
    return;
  }
  int i = (e < 127) ? 0 : (e - 127 + 64 - 1) / 64;
  uint64_t c[5];
  u = (u128) X->hi * (u128) T[i+3];
  c[0] = u;
  c[1] = u >> 64;
  u = (u128) X->hi * (u128) T[i+2];
  c[1] += u;
  c[2] = (u >> 64) + (c[1] < (uint64_t) u);
  u = (u128) X->hi * (u128) T[i+1];
  c[2] += u;
  c[3] = (u >> 64) + (c[2] < (uint64_t) u);
  u = (u128) X->hi * (u128) T[i];
  c[3] += u;
  c[4] = (u >> 64) + (c[3] < (uint64_t) u);
  int f = e - 64 * i;
  uint64_t tiny;
  if (f < 64)
  {
    X->hi = (c[4] << f) | (c[3] >> (64 - f));
    X->lo = (c[3] << f) | (c[2] >> (64 - f));
    tiny = (c[2] << f) | (c[1] >> (64 - f));
  }
  else if (f == 64)
  {
    X->hi = c[3];
    X->lo = c[2];
    tiny = c[1];
  }
  else
  {
    int g = f - 64;
    u = (u128) X->hi * (u128) T[i+4];
    u = u >> 64;
    c[0] += u;
    c[1] += (u128(c[0]) < u);
    c[2] += (u128(c[0]) < u) && c[1] == 0;
    c[3] += (u128(c[0]) < u) && c[1] == 0 && c[2] == 0;
    c[4] += (u128(c[0]) < u) && c[1] == 0 && c[2] == 0 && c[3] == 0;
    X->hi = (c[3] << g) | (c[2] >> (64 - g));
    X->lo = (c[2] << g) | (c[1] >> (64 - g));
    tiny = (c[1] << g) | (c[0] >> (64 - g));
  }
  X->ex = 0;
  normalize (X);
  if (X->ex < 0)
    X->lo |= tiny >> (64 + X->ex);
}
__device__ __forceinline__ int
reduce2 (dint64_t *X)
{
  if (X->ex <= -11)
    return 0;
  int sh = 64 - 11 - X->ex;
  int i = X->hi >> sh;
  X->hi = X->hi & ((1ull << sh) - 1);
  normalize (X);
  return i;
}
__device__ __forceinline__ void
set_dd (double *h, double *l, uint64_t c1, uint64_t c0)
{
  uint64_t e, f, g;
  b64u64_u t;
  if (c1)
    {
      e = g64_clzll (c1);
      if (e)
        {
          c1 = (c1 << e) | (c0 >> (64 - e));
          c0 = c0 << e;
        }
      f = 0x3fe - e;
      t.u = (f << 52) | ((c1 << 1) >> 12);
      *h = t.f;
      c0 = (c1 << 53) | (c0 >> 11);
      if (c0)
        {
          g = g64_clzll (c0);
          if (g)
            c0 = c0 << g;
          t.u = ((f - 53 - g) << 52) | ((c0 << 1) >> 12);
          *l = t.f;
        }
      else
        *l = 0;
    }
  else if (c0)
    {
      e = g64_clzll (c0);
      f = 0x3fe - 64 - e;
      c0 = c0 << (e+1);
      t.u = (f << 52) | (c0 >> 12);
      *h = t.f;
      c0 = c0 << 52;
      if (c0)
        {
          g = g64_clzll (c0);
          c0 = c0 << (g+1);
          t.u = ((f - 64 - g) << 52) | (c0 >> 12);
          *l = t.f;
        }
      else
        *l = 0;
    }
  else
    *h = *l = 0;
}
__device__ __forceinline__ int
reduce_fast (double *h, double *l, double x, double *err1)
{
  if ((x <= 0x1.921fb54442d17p+2))
    {
      a_mul (h, l, 0x1.45f306dc9c883p-3, x);
      *l = __fma_rn (-0x1.6b01ec5417056p-57, x, *l);
      *err1 = __dmul_rn(0x1.d9p-105,*h);
    }
  else
    {
      b64u64_u t; t.f = x;
      int e = (t.u >> 52) & 0x7ff;
      uint64_t m = (1ull << 52) | (t.u & 0xfffffffffffffull);
      uint64_t c[3];
      u128 u;
      if (e <= 1074)
        {
          u = (u128) m * (u128) T[1];
          c[0] = u;
          c[1] = u >> 64;
          u = (u128) m * (u128) T[0];
          c[1] += u;
          c[2] = (u >> 64) + (c[1] < (uint64_t) u);
          e = 1075 - e;
        }
      else
        {
          int i = (e - 1138 + 63) / 64;
          u = (u128) m * (u128) T[i+2];
          c[0] = u;
          c[1] = u >> 64;
          u = (u128) m * (u128) T[i+1];
          c[1] += u;
          c[2] = (u >> 64) + (c[1] < (uint64_t) u);
          u = (u128) m * (u128) T[i];
          c[2] += u;
          e = 1139 + (i<<6) - e;
        }
      if (e == 64)
        {
          c[0] = c[1];
          c[1] = c[2];
        }
      else
        {
          c[0] = (c[1] << (64 - e)) | c[0] >> e;
          c[1] = (c[2] << (64 - e)) | c[1] >> e;
        }
      set_dd (h, l, c[1], c[0]);
      *err1 = 0x1.01p-76;
    }
  double i = floor (__dmul_rn(*h,0x1p11));
  *h = __fma_rn (i, -0x1p-11, *h);
  return i;
}
__device__ __forceinline__ double
cos_fast (double *h, double *l, double x)
{
  int neg = 0, is_cos = 1;
  double err1;
  int i = reduce_fast (h, l, x, &err1);
  neg = neg ^ (i >> 10);
  i = i & 0x3ff;
  is_cos = is_cos ^ (i >> 9);
  neg = neg ^ (i >> 9);
  i = i & 0x1ff;
  if (i & 0x100)
    {
      is_cos = !is_cos;
      i = 0x1ff - i;
      *h = __dsub_rn(0x1p-11,*h);
      *l = -*l;
    }
  double sh, sl, ch, cl;
  *h = __dsub_rn(*h,SC[i][0]);
  double uh, ul;
  a_mul (&uh, &ul, *h, *h);
  ul = __fma_rn (__dadd_rn(*h,*h), *l, ul);
  evalPSfast (&sh, &sl, *h, *l, uh, ul);
  evalPCfast (&ch, &cl, uh, ul);
  double err;
  if (!is_cos)
    {
      s_mul (&sh, &sl, SC[i][2], sh, sl);
      s_mul (&ch, &cl, SC[i][1], ch, cl);
      fast_two_sum (h, l, ch, sh);
      *l = __dadd_rn(*l,__dadd_rn(sl,cl));
      err = 0x1.55p-69;
    }
  else
    {
      s_mul (&ch, &cl, SC[i][2], ch, cl);
      s_mul (&sh, &sl, SC[i][1], sh, sl);
      fast_two_sum (h, l, ch, -sh);
      *l = __dadd_rn(*l,__dsub_rn(cl,sl));
      err = 0x1.81p-69;
    }
  const auto &sgn = g64_table_0;
  *h = __dmul_rn(*h,sgn[neg]);
  *l = __dmul_rn(*l,sgn[neg]);
  return __dadd_rn(err,err1);
}

__device__ __forceinline__ double
cos_accurate (double x)
{
  dint64_t X[1];
  dint_fromd (X, x);
  reduce (X);
  int neg = 0, is_cos = 1;
  int i = reduce2 (X);
  if (i & 0x400)
  {
    neg = 1;
    i = i & 0x3ff;
  }
  if (i & 0x200)
  {
    neg = !neg;
    is_cos = 0;
    i = i & 0x1ff;
  }
  if (i & 0x100)
  {
    is_cos = !is_cos;
    X->sgn = 1;
    add_dint (X, &MAGIC, X);
    i = 0x1ff - i;
  }
  dint64_t U[1], V[1], X2[1];
  mul_dint (X2, X, X);
  evalPC (U, X2);
  evalPS (V, X, X2);
  if (!is_cos)
  {
    mul_dint (U, S+i, U);
    mul_dint (V, C+i, V);
  }
  else
  {
    mul_dint (U, C+i, U);
    mul_dint (V, S+i, V);
    V->sgn = 1 - V->sgn;
  }
  add_dint (U, U, V);
  uint64_t err = 41;
  uint64_t hi0, hi1, lo0, lo1;
  lo0 = U->lo - err;
  hi0 = U->hi - (lo0 > U->lo);
  lo1 = U->lo + err;
  hi1 = U->hi + (lo1 < U->lo);
  if ((hi0 >> 10) != (hi1 >> 10))
    {
      const auto &exceptions = g64_table_1;
      for (int k = 0; k < 5; k++)
        {
          if (fabs (x) == exceptions[k][0])
            return __dadd_rn(exceptions[k][1],exceptions[k][2]);
        }
    }
  if (neg)
    U->sgn = 1 - U->sgn;
  double y = dint_tod (U);
  return y;
}
__device__ __forceinline__ double cr_cos (double x)
{
  b64u64_u t; t.f = x;
  int e = (t.u >> 52) & 0x7ff;
  if ((e == 0x7ff))
    {
      if ((t.u << 1) == 0x7ffull<<53)
        return __ddiv_rn(0.0,0.0);
      return __dadd_rn(x,x);
    }
  t.u &= 0x7fffffffffffffff;
  if ((t.u <= 0x3e46a09e667f3bcc))
    return __fma_rn (t.f, -0x1p-28, 1.0);
  double h, l, err;
  err = cos_fast (&h, &l, t.f);
  double left = __dadd_rn(h,(__dsub_rn(l,err))), right = __dadd_rn(h,(__dadd_rn(l,err)));
  if ((left == right))
    return left;
  return cos_accurate (t.f);
}

}

/* Correctly-rounded arc cosine of binary64 value.

Copyright (c) 2024-2025 Alexei Sibidanov.

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
namespace g64_acos {
__device__ const double g64_table_0[33][8] = {
    { 1, 0, 0x1.5555555555555p-3, 0x1.33333333333e4p-4,
     0x1.6db6db6d31f82p-5, 0x1.f1c71f6889397p-6, 0x1.6e874b7045b46p-6, 0x1.1f753132271e2p-6},
    {0x1.0055a27e0d033p+0, -0x1.d9ba10494c062p-54, 0x1.57c00cb5d6c4dp-3, 0x1.37881f5649a75p-4,
     0x1.759af49d494ddp-5, 0x1.002e1864dda2ep-5, 0x1.7c2d5d468cdd9p-6, 0x1.292834c025357p-6},
    {0x1.00abe0c129e1ep+0, 0x1.7ceb0ee49d42ap-57, 0x1.5a3385d5c7ba5p-3, 0x1.3bf51056f6637p-4,
     0x1.7dba76b124b37p-5, 0x1.07be4b02e94c4p-5, 0x1.8a6bb92513f01p-6, 0x1.36afd4c615aecp-6},
    {0x1.0102bcffd6acdp+0, -0x1.c2294c65d2e86p-55, 0x1.5caff17351901p-3, 0x1.407abbc04feb3p-4,
     0x1.86179b8005949p-5, 0x1.0f97520bd4e72p-5, 0x1.9950c5c89f3dfp-6, 0x1.44f2344e7b664p-6},
    {0x1.015a397cf0f1cp+0, -0x1.eebd6ccfe3ee3p-55, 0x1.5f3581be7b08bp-3, 0x1.4519ddf1ae531p-4,
     0x1.8eb4b6ee35e92p-5, 0x1.17bc85414cd46p-5, 0x1.a8e5895e3fcf9p-6, 0x1.53fafdc629400p-6},
    {0x1.01b2588811eebp+0, 0x1.7193e5d0a915fp-59, 0x1.61c46a67205d2p-3, 0x1.49d33a6eeae0bp-4,
     0x1.979438563c014p-5, 0x1.20316ae977f05p-5, 0x1.b9339afb53aa4p-6, 0x1.63d6b02c42d0ap-6},
    {0x1.020b1c7df0575p+0, -0x1.dd547e329c1e5p-55, 0x1.645ce0ab901bbp-3, 0x1.4ea79c34fc7a6p-4,
     0x1.a0b8ac08940ecp-5, 0x1.28f9babd0629bp-5, 0x1.ca452cf90a55ep-6, 0x1.7492b016730efp-6},
    {0x1.026487c8c5d71p+0, -0x1.5fd9b68dc3b6ep-54, 0x1.66ff1b67d5d70p-3, 0x1.5397d613373ebp-4,
     0x1.aa24bce3aeb4ap-5, 0x1.3219610b150acp-5, 0x1.dc251825103f1p-6, 0x1.863d5a3932532p-6},
    {0x1.02be9ce0b87cdp+0, 0x1.e5d09da2e0f04p-56, 0x1.69ab5325bc359p-3, 0x1.58a4c3097aab3p-4,
     0x1.b3db3605f46f2p-5, 0x1.3b94821742cabp-5, 0x1.eedee7da72a15p-6, 0x1.98e6179a3e9a0p-6},
    {0x1.03195e4c483f1p+0, -0x1.5db10ad66eacbp-54, 0x1.6c61c22d908f0p-3, 0x1.5dcf46ab9f2cbp-4,
     0x1.bddf049bb1f4dp-5, 0x1.456f7db6ac768p-5, 0x1.013f738bd7bb3p-5, 0x1.ac9d739783d21p-6},
    {0x1.0374cea0c0c9fp+0, -0x1.917bff5241c76p-54, 0x1.6f22a497b2ec0p-3, 0x1.63184d8a79db5p-4,
     0x1.c83339caf946ep-5, 0x1.4faef331019d4p-5, 0x1.0b8917547d678p-5, 0x1.c17533f147e1cp-6},
    {0x1.03d0f082afcc8p+0, -0x1.018bbcddb49ebp-54, 0x1.71ee385efdf06p-3, 0x1.6880cda2d3885p-4,
     0x1.d2db0cbfae54dp-5, 0x1.5a57c56b50c5ep-5, 0x1.16535a40098b2p-5, 0x1.d780730b8ebb8p-6},
    {0x1.042dc6a65ffbfp+0, -0x1.c7ea28dce95d1p-55, 0x1.74c4bd7412f9ep-3, 0x1.6e09c6d2b72bcp-4,
     0x1.ddd9dcda253dep-5, 0x1.656f1f62b5001p-5, 0x1.21a5ae2ac77eep-5, 0x1.eed3bca067f0ep-6},
    {0x1.048b53d05907bp+0, 0x1.634fffed6e2a6p-54, 0x1.77a675d1978bep-3, 0x1.73b4435583415p-4,
     0x1.e9333402ebbf3p-5, 0x1.70fa78fd9f73fp-5, 0x1.2d8804d934fe1p-5, 0x1.03c29691a281cp-5},
    {0x1.04e99ad5e4bcdp+0, -0x1.e97a72fe827e0p-54, 0x1.7a93a5917200cp-3, 0x1.7981584731c05p-4,
     0x1.f4eac9268fae2p-5, 0x1.7cff9c3b19721p-5, 0x1.3a02d9c1e0145p-5, 0x1.10d64a0e56953p-5},
    {0x1.05489e9d99995p+0, 0x1.d177637ec6a2bp-55, 0x1.7d8c930314681p-3, 0x1.7f72262f532e4p-4,
     0x1.0082416e39013p-4, 0x1.8984aac80ddf4p-5, 0x1.471f3caf18eb8p-5, 0x1.1eb1cce6dd570p-5},
    {0x1.05a8621feb16bp+0, -0x1.e5b33b1407c5fp-56, 0x1.809186c2e57ddp-3, 0x1.8587d99442dc8p-4,
     0x1.06c23d1dfcb7fp-4, 0x1.969024036dd22p-5, 0x1.54e6dd4d2af33p-5, 0x1.2d62f439f2a31p-5},
    {0x1.0608e867bff30p+0, 0x1.cbef5d8580027p-55, 0x1.83a2cbd2d8ba2p-3, 0x1.8bc3ab9724c6ep-4,
     0x1.0d377ef1e0c39p-4, 0x1.a428eb7addf84p-5, 0x1.636417bc01ff2p-5, 0x1.3cf8acc7eb2a0p-5},
    {0x1.066a34930ec8dp+0, -0x1.480f445fedad1p-54, 0x1.86c0afb447a74p-3, 0x1.9226e29948d9cp-4,
     0x1.13e44a9b5a3a6p-4, 0x1.b2564fea8b3fep-5, 0x1.72a2023d92458p-5, 0x1.4d8313cec3485p-5},
    {0x1.06cc49d38146cp+0, -0x1.b55394f4fc07bp-55, 0x1.89eb82831feedp-3, 0x1.98b2d2eb9bb23p-4,
     0x1.1acb01e9ab414p-4, 0x1.c12012cbd00c6p-5, 0x1.82ac7c1d15c38p-5, 0x1.5f13925c6edcap-5},
    {0x1.072f2b6f1e601p+0, -0x1.2dcbb05419970p-54, 0x1.8d2397127aebbp-3, 0x1.9f68df88da51dp-4,
     0x1.21ee26a4f62a1p-4, 0x1.d08e707f7ae6fp-5, 0x1.93903dee3feb0p-5, 0x1.71bcfb5c57b59p-5},
    {0x1.0792dcc0fbd20p+0, -0x1.5bf23ee4f9d54p-56, 0x1.9069430ab508ap-3, 0x1.a64a7adb4cd85p-4,
     0x1.29505c8b48349p-4, 0x1.e0aa2921cfa60p-5, 0x1.a55aeb46f4322p-5, 0x1.8593acad3becep-5},
    {0x1.07f76139f761dp+0, 0x1.fa1046481bb82p-54, 0x1.93bcdf091cca6p-3, 0x1.ad59278edc42fp-4,
     0x1.30f46b7261652p-4, 0x1.f17c8a17c843ep-5, 0x1.b81b2619e15b5p-5, 0x1.9aadb395f5ae4p-5},
    {0x1.085cbc61783c1p+0, 0x1.0a6e9efa20176p-54, 0x1.971ec6c1531e4p-3, 0x1.b496797068912p-4,
     0x1.38dd419140184p-4, 0x1.0187bc3357bbbp-4, 0x1.cbe0a3dcafe26p-5, 0x1.b122f4fa499d0p-5},
    {0x1.08c2f1d638e4cp+0, 0x1.b47c159534a3dp-56, 0x1.9a8f592078624p-3, 0x1.bc04165b57ab2p-4,
     0x1.410df5f4bed1dp-4, 0x1.0ab6bdf478c71p-4, 0x1.e0bc44a945c64p-5, 0x1.c90d59bcd5701p-5},
    {0x1.092a054f1a2fcp+0, -0x1.2f657224e9830p-54, 0x1.9e0ef87243a2cp-3, 0x1.c3a3b7366a278p-4,
     0x1.4989cb22e2175p-4, 0x1.1450e5ba7ad39p-4, 0x1.f6c02c8f0ef93p-5, 0x1.e288ffc8d182cp-5},
    {0x1.0991fa9bffbf4p+0, -0x1.ca1140a1abbf4p-58, 0x1.a19e0a8823b80p-3, 0x1.cb772900f9c24p-4,
     0x1.525431f0cbb2ep-4, 0x1.1e5c2d06804e1p-4, 0x1.06ffefa7aa6b8p-4, 0x1.fdb4704dca347p-5},
    {0x1.09fad5a6b68f9p+0, 0x1.aa1f06e92964ep-56, 0x1.a53cf8e28c50ep-3, 0x1.d3804df1de350p-4,
     0x1.5b70cc8fa98dcp-4, 0x1.28def298c979bp-4, 0x1.13482f6347eebp-4, 0x1.0d586de48358cp-4},
    {0x1.0a649a73e61f2p+0, 0x1.74ac0d817e9c7p-55, 0x1.a8ec30dc93891p-3, 0x1.dbc11ea950625p-4,
     0x1.64e371d5616d3p-4, 0x1.33e0023936249p-4, 0x1.204426263066ap-4, 0x1.1cd12e4629723p-4},
    {0x1.0acf4d240ccc4p+0, 0x1.da890f3b40bd3p-54, 0x1.acac23da07797p-3, 0x1.e43bab7741a98p-4,
     0x1.6eb030c631819p-4, 0x1.3f669d2eb516ep-4, 0x1.2e0006ae505aep-4, 0x1.2d58204457c82p-4},
    {0x1.0b3af1f4880bbp+0, 0x1.f450fb78d32bap-56, 0x1.b07d4778263afp-3, 0x1.ecf21db7be0efp-4,
     0x1.78db5465013e4p-4, 0x1.4b7a8376f0996p-4, 0x1.3c88f9f2ef221p-4, 0x1.3f02ad9eb9753p-4},
    {0x1.0ba78d40a9260p+0, -0x1.57b07a441e242p-54, 0x1.b46015c126262p-3, 0x1.f5e6b94713f3dp-4,
     0x1.836967d0afecfp-4, 0x1.5823fdd1707b9p-4, 0x1.4bed355269dc2p-4, 0x1.51e83065121cfp-4},
    {0x1.0c152382d7366p+0, -0x1.ee6913347c2a6p-54, 0x1.b8550d62bfb6ep-3, 0x1.ff1bde0fa3cadp-4,
     0x1.8e5f3ab550989p-4, 0x1.656be8b38ebafp-4, 0x1.5c3c13008a099p-4, 0x1.662225a1b4f77p-4},
  };
__device__ const double g64_table_1[][2] = {{0,0}, {0x1.921fb54442d18p+1, 0x1.1a62633145c07p-53}};
__device__ const double g64_table_2 = -0x1.5555555555555p-3;
__device__ const double g64_table_3 = 0x1.34p-79;
__device__ const double g64_table_4[33][2] = {
    {0x0p+0, 0x0p+0}, {-0x1.912bd0d569a9p-61, 0x1.91f65f10dd814p-5},
    {-0x1.e2718d26ed688p-60, 0x1.917a6bc29b42cp-4}, {0x1.13000a89a11ep-58, 0x1.2c8106e8e613ap-3},
    {-0x1.26d19b9ff8d82p-57, 0x1.8f8b83c69a60bp-3}, {-0x1.42deef11da2c4p-57, 0x1.f19f97b215f1bp-3},
    {-0x1.5d28da2c4612dp-56, 0x1.294062ed59f06p-2}, {-0x1.efdc0d58cf62p-62, 0x1.58f9a75ab1fddp-2},
    {-0x1.72cedd3d5a61p-57, 0x1.87de2a6aea963p-2}, {0x1.5b362cb974183p-57, 0x1.b5d1009e15ccp-2},
    {0x1.e0d891d3c6841p-58, 0x1.e2b5d3806f63bp-2}, {-0x1.a5a014347406cp-55, 0x1.073879922ffeep-1},
    {0x1.b25dd267f66p-55, 0x1.1c73b39ae68c8p-1}, {-0x1.efcc626f74a6fp-57, 0x1.30ff7fce17035p-1},
    {0x1.8076a2cfdc6b3p-57, 0x1.44cf325091dd6p-1}, {-0x1.75720992bfbb2p-55, 0x1.57d69348cecap-1},
    {-0x1.bdd3413b26456p-55, 0x1.6a09e667f3bcdp-1}, {-0x1.0f537acdf0ad7p-56, 0x1.7b5df226aafafp-1},
    {-0x1.2c5e12ed1336dp-55, 0x1.8bc806b151741p-1}, {-0x1.30ee286712474p-55, 0x1.9b3e047f38741p-1},
    {0x1.9f630e8b6dac8p-60, 0x1.a9b66290ea1a3p-1}, {-0x1.bc69f324e6d61p-55, 0x1.b728345196e3ep-1},
    {-0x1.6e0b1757c8d07p-56, 0x1.c38b2f180bdb1p-1}, {-0x1.e7b6bb5ab58aep-58, 0x1.ced7af43cc773p-1},
    {0x1.457e610231ac2p-56, 0x1.d906bcf328d46p-1}, {-0x1.014c76c126527p-55, 0x1.e212104f686e5p-1},
    {0x1.760b1e2e3f81ep-55, 0x1.e9f4156c62ddap-1}, {0x1.52c7adc6b4989p-56, 0x1.f0a7efb9230d7p-1},
    {0x1.562172a361fd3p-56, 0x1.f6297cff75cbp-1}, {-0x1.7a0a8ca13571fp-55, 0x1.fa7557f08a517p-1},
    {-0x1.87df6378811c7p-55, 0x1.fd88da3d12526p-1}, {-0x1.c57bc2e24aa15p-57, 0x1.ff621e3796d7ep-1},
    {0x0p+0, 0x1p+0}
  };
__device__ const double g64_table_5[][2] = {
    {0x1p+0, -0x1.fc2c76456515bp-108}, {0x1.5555555555555p-3, 0x1.5555555623513p-57},
    {0x1.3333333333333p-4, 0x1.9997e3427441bp-59}, {0x1.6db6db6db6db7p-5, -0x1.cb95ff08658e6p-62},
    {0x1.f1c71c71c6d5bp-6, 0x1.b125bccdcc89ep-60}};
__device__ const double g64_table_6[] = {0x1.6e8ba2ec8cb69p-6, 0x1.1c4ea7a15c997p-6, 0x1.ca8355d39bb67p-7};



typedef uint64_t u64;
typedef int64_t i64;
typedef unsigned short ushort;
typedef union {double f; uint64_t u;} b64u64_u;
__device__ __forceinline__ double fasttwosum(double x, double y, double *e){
  double s = __dadd_rn(x,y), z = __dsub_rn(s,x);
  *e = __dsub_rn(y,z);
  return s;
}
__device__ __forceinline__ double twosum(double a, double b, double *t){
  double s = __dadd_rn(a,b);
  double a_prime = __dsub_rn(s,b);
  double b_prime = __dsub_rn(s,a_prime);
  double delta_a = __dsub_rn(a,a_prime);
  double delta_b = __dsub_rn(b,b_prime);
  *t = __dadd_rn(delta_a,delta_b);
  return s;
}
__device__ __forceinline__ double fastsum(double xh, double xl, double yh, double yl, double *e){
  double sl, sh = fasttwosum(xh, yh, &sl);
  *e = __dadd_rn((__dadd_rn(xl,yl)),sl);
  return sh;
}
__device__ __forceinline__ double sum(double xh, double xl, double ch, double cl, double *l){
  double sl, sh = twosum(xh,ch, &sl);
  *l = __dadd_rn((__dadd_rn(xl,cl)),sl);
  return sh;
}
__device__ __forceinline__ double muldd(double xh, double xl, double ch, double cl, double *l){
  double ahhh = __dmul_rn(xh,ch);
  *l = __dadd_rn((__dadd_rn(__dmul_rn(xh,cl),__dmul_rn(xl,ch))),__fma_rn(xh, ch, -ahhh));
  return ahhh;
}
__device__ __forceinline__ double polydd(double xh, double xl, int n, const double c[][2], double *l){
  int i = n-1;
  double ch = fasttwosum(c[i][0], *l, l), cl = __dadd_rn(c[i][1],*l);
  while(--i>=0){
    ch = muldd(xh,xl, ch,cl, &cl);
    ch = fastsum(c[i][0],c[i][1], ch,cl, &cl);
  }
  *l = cl;
  return ch;
}
__device__ __forceinline__ double  as_acos_refine(double, double);
__device__ __forceinline__ double cr_acos (double x){
  const auto &cc = g64_table_0;
  b64u64_u ix; ix.f = x;
  u64 ax = ix.u<<1;
  double t,z,zl,jd,f0h,f0l,eps;
  if(ax>0x7fc0000000000000ull){
    const auto &off = g64_table_1;
    i64 k = ix.u>>63;
    f0h = off[k][0];
    f0l = off[k][1];
    if((ax>=0x7fe0000000000000ull)){
      if(ax==0x7fe0000000000000ull) return __dadd_rn(f0h,f0l);
      if(ax>0xffe0000000000000ull) return __dadd_rn(x,x);
      return __ddiv_rn(0.,0.);
    }
    t = __dsub_rn(2,__dmul_rn(2,fabs(x)));
    jd = rint (__dmul_rn(t,0x1p5));
    z = copysign(__dsqrt_rn(t), x);
    zl = __dmul_rn(__fma_rn(z,z,-t),(__dmul_rn((__ddiv_rn(-0.5,t)),z)));
    t = __dsub_rn(__dmul_rn(0.25,t),__dmul_rn(jd,0x1p-7));
    eps = __dadd_rn(__dmul_rn(fabs(__dmul_rn(z,t)),0x1.8cp-52),0x1p-105);
  } else {
    f0h = 0x1.921fb54442d18p+0;
    f0l = 0x1.1a62633145c07p-54;
    if ((ax <= 0x7e00000000000000ull)) {
      const auto &c = g64_table_2;
      double v = (ax <= 0x791967670d2e8fe8ull) ? 0 : __dmul_rn((__dmul_rn(x,x)),(__dmul_rn(c,x)));
      double h, w;
      h = fasttwosum (f0h, -x, &w);
      double l = __dadd_rn(v,(__dadd_rn(w,f0l)));
      const auto &eps1 = g64_table_3;
      double lb = __dadd_rn(h,(__dsub_rn(l,eps1))), ub = __dadd_rn(h,(__dadd_rn(l,eps1)));
      if((lb!=ub)) return as_acos_refine(x, lb);
      return lb;
    }
    t = __dmul_rn(x,x);
    jd = rint (__dmul_rn(t,0x1p7));
    t = __fma_rn(x,x,__dmul_rn(-0x1p-7,jd));
    z = -x;
    zl = 0;
    eps = __dmul_rn((__dmul_rn(z,t)),0x1.81p-52);
  }
  int64_t j = jd;
  const double *c = cc[j];
  double t2 = __dmul_rn(t,t), d = __dmul_rn(t,(__dadd_rn((__dadd_rn(c[2],__dmul_rn(t,c[3]))),__dmul_rn(t2,(__dadd_rn((__dadd_rn(c[4],__dmul_rn(t,c[5]))),__dmul_rn(t2,(__dadd_rn(c[6],__dmul_rn(t,c[7]))))))))));
  double fh = c[0], fl = __dadd_rn(c[1],d);
  fh = muldd(z,zl, fh,fl, &fl);
  fh = fastsum(f0h,f0l, fh,fl, &fl);
  double lb = __dadd_rn(fh,(__dsub_rn(fl,eps))), ub = __dadd_rn(fh,(__dadd_rn(fl,eps)));
  if((lb!=ub)) return as_acos_refine(x, lb);
  return lb;
}

__device__ __forceinline__ double as_acos_refine(double x, double phi){
  double s2 = __dmul_rn(x,x), dx2 = __fma_rn(x,x,-s2);
  double c2l, c2h = fasttwosum(1.0,-s2,&c2l);
  c2l = __dsub_rn(c2l,dx2);
  c2h = fasttwosum(c2h,c2l,&c2l);
  double ch = __dsqrt_rn(c2h);
  double cl = __dmul_rn((__dsub_rn(c2l,__fma_rn(ch,ch,-c2h))),(__ddiv_rn(0.5,ch)));
  int jf = rint (__dmul_rn(fabs(__dsub_rn(phi,0x1.921fb54442d18p+0)),0x1.45f306dc9c883p+4));
  const auto &s = g64_table_4;
  double Ch = s[32-jf][1], Cl = s[32-jf][0], Sh = s[jf][1], Sl = s[jf][0];
  double ax = fabs(x);
  double dsh = __dsub_rn(ax,Sh), dsl = -Sl;
  double dch = __dsub_rn(ch,Ch), dcl = __dsub_rn(cl,Cl);
  double Sc = __dsub_rn(__fma_rn(Sh, dch, 0x1.8p-4),0x1.8p-4);
  double dSc = __fma_rn(Sh, dch, -Sc);
  double Cs = __dsub_rn(__fma_rn(Ch, dsh, 0x1.8p-4),0x1.8p-4);
  double dCs = __fma_rn(Ch, dsh, -Cs);
  double v = __dsub_rn(Cs,Sc);
  double dv = __dsub_rn(__dsub_rn((__dadd_rn(__dmul_rn(Ch,dsl),__dmul_rn(Cl,dsh))),(__dadd_rn(__dmul_rn(Sh,dcl),__dmul_rn(Sl,dch)))),(__dsub_rn(dSc,dCs)));
  v = fasttwosum(v,dv,&dv);
  double sgn = copysign(1.0, x), jt = __dsub_rn(32,__dmul_rn(jf,sgn));
  const auto &c = g64_table_5;
  const auto &ct = g64_table_6;
  double dv2, v2 = muldd(v,dv, v,dv, &dv2);
  v = __dmul_rn(v,-sgn);
  dv = __dmul_rn(dv,-sgn);
  double fl = __dmul_rn(v2,(__dadd_rn(ct[0],__dmul_rn(v2,(__dadd_rn(ct[1],__dmul_rn(v2,ct[2]))))))), fh = polydd(v2,dv2, 5,c, &fl);
  fh = muldd(v,dv, fh,fl, &fl);
  double ph = __dmul_rn(jt,0x1.921fb54442dp-5), pl = __dmul_rn(0x1.8469898cc518p-53,jt), ps = __dmul_rn(-0x1.fc8f8cbb5bf6cp-102,jt);
  pl = sum(fh,fl, pl,ps, &ps);
  ph = fasttwosum(ph,pl, &pl);
  pl = fasttwosum(pl,ps, &ps);
  ph = fasttwosum(ph,pl, &pl);
  pl = fasttwosum(pl,ps, &ps);
  b64u64_u t; t.f = pl;
  i64 e = ((t.u>>52)&0x7ff) - 1023;
  e = 52-(107+e);
  e = e<0?0:e;
  e = e>52?52:e;
  u64 m = ((u64)1<<52)-((u64)1<<e);
  e = (e == 0) ? 64 : e;
  if((!((t.u+((u64)1<<(e-1)))&m))){
    if(x== 0x1.ffffffffffdc0p-1 ) return __dadd_rn(0x1.8000000000024p-22,0x1p-76);
    if(x== 0x1.53ea6c7255e88p-4 ) return __dadd_rn(0x1.7cdacb6bbe707p+0,0x1p-54);
    if(x== 0x1.fd737be914578p-11) return __dadd_rn(0x1.91e006d41d8d8p+0,0x1.8p-53);
    if(x== 0x1.fffffffffff70p-1 ) return __dadd_rn(0x1.8000000000009p-23,0x1p-77);
    if(x== 0x1.390e6939cd1a6p-5 ) return __dsub_rn(0x1.8856a5d3296a4p+0,0x1p-109);
    b64u64_u w; w.f = ps;
    if((w.u^t.u)>>63)
      t.u--;
    else
      t.u++;
    pl = t.f;
  }
  return __dadd_rn(ph,pl);
}

}

__device__ __forceinline__ double glibc_exp(double x){uint64_t u=g64_exp::asuint64(x);if((u&0x7fffffffffffffffULL)>0x7ff0000000000000ULL)return g64_exp::asdouble(u|0x8000000000000ULL);return g64_exp::exp(x);}
__device__ __forceinline__ double glibc_log(double x){return g64_log::log(x);}
__device__ __forceinline__ double glibc_pow(double x,double y){
 uint64_t ix=g64_pow::asuint64(x),iy=g64_pow::asuint64(y);
 bool nx=(ix&0x7fffffffffffffffULL)>0x7ff0000000000000ULL;
 bool ny=(iy&0x7fffffffffffffffULL)>0x7ff0000000000000ULL;
 if(nx){
  if((iy<<1)==0 && (ix&0x8000000000000ULL))return 1.;
  uint64_t q=ix|0x8000000000000ULL;
  if((iy&0x7fffffffffffffffULL)!=0 && (iy&0x7fffffffffffffffULL)<0x7ff0000000000000ULL && (ix>>63) && g64_pow::checkint(iy)==1)q^=0x8000000000000000ULL;
  return g64_pow::asdouble(q);
 }
 if(ny){if(ix==0x3ff0000000000000ULL && (iy&0x8000000000000ULL))return 1.;return g64_pow::asdouble(iy|0x8000000000000ULL);}
 return g64_pow::pow(x,y);
}
__device__ __forceinline__ double uw_cos(double x){return g64_cos::cr_cos(x);}
__device__ __forceinline__ double uw_acos(double x){return g64_acos::cr_acos(x);}

// thompson_aerosol_libm.cuh -- WOOF's own float32 and binary64 libm words
// for the mp=28 (aerosol-aware Thompson) units.  Prepended to every
// thompson_aerosol_* unit after glibc_flt32.cuh and glibc_flt64.cuh and
// before thompson_aerosol_common.cuh (gpuwm/core/kernels/__init__.py
// _EXTRA_HEADERS).  Self-contained apart from those two headers, so
// tools/thompson_aerosol_column_oracle/libm_check.cpp compiles it for the
// host and compares every word with the oracle host's C library.
#pragma once

// ---------------------------------------------------------------------------
// WOOF's own libm words for WRF's EXP, LOG, LOG10 and ** (the 0 ULP column
// oracle, tools/thompson_aerosol_column_oracle).
// ---------------------------------------------------------------------------
//
// THE BREAKAGE THIS PREVENTS.  WRF's module_mp_thompson.F is compiled by
// gfortran, which lowers a REAL(4) EXP/ALOG/ALOG10/** to the C library's
// expf/logf/log10f/powf and a DOUBLE PRECISION one to exp/log/log10/pow.
// On the x86-64 oracle host (W1: glibc 2.39 on a CPU with FMA, so the
// library takes its FMA code path) those words are NOT correctly rounded:
// expf and powf differ from the correctly rounded result on some
// arguments, and so did this header's earlier "evaluate in double and
// round once" helpers.  CUDA's expf/logf/powf/exp/log/pow/log10 are
// different functions again.  The column oracle measured the result: no
// column of 153 was bit-identical to WRF, under strict or default
// arithmetic.
//
// So every mp=28 unit evaluates these through the words below, in every
// arithmetic mode (a correctness remedy is default-on):
//
//   thompson_aa_expf(x)     float32 exp: the FMA path's operation order on
//                           the 32-entry exp2 table GFK_EXP2F_TAB
//                           (glibc_flt32.cuh, prepended to these units)
//   thompson_aa_logf(x)     gfk_log
//   thompson_aa_powf(x, y)  float32 pow: the FMA path's log2 and exp2 cores
//                           on GFK_POWF_INVC/LOGC/A and GFK_EXP2F_TAB
//   thompson_aa_log10f(x)   k*log10_2lo + ivln10*logf(m), then + k*log10_2hi
//   thompson_aa_exp/log/pow (binary64)  glibc_exp/glibc_log/glibc_pow
//                           (glibc_flt64.cuh, prepended to these units)
//   thompson_aa_log10(x)    the same composition in binary64 on
//                           thompson_aa_log
//
// MEASURED on W1 against the host's own libm, compiled for the host with
// tools/thompson_real_column_parity/cuda_host_shim.h (-ffp-contract=off):
// expf, logf and log10f over all 2^32 float32 inputs; powf, exp, log, pow
// and log10 on 2e8 sampled arguments each; zero mismatches
// (tools/thompson_aerosol_column_oracle/libm_check.cpp).  The device
// evaluates the same operations (every binary64 product-sum below is an
// explicit __fma_rn or a pinned __dmul_rn/__dadd_rn), so it returns the
// same words.  CUDA's own expf/logf/powf/log10f/exp/log/pow/log10 must not
// be called from an mp=28 unit; tests/test_thompson_aerosol_libm.py reads
// the sources for them.
__device__ __forceinline__ float thompson_aa_d2f(double y)
{
    return gfk_d2f_rn(y);
}

__device__ __forceinline__ float thompson_aa_expf(float x)
{
    const unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return __fadd_rn(x, x);
        if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
        if (x < -__int_as_float(0x42cff1b4)) return 0.0f;
    }
    const double xd = (double)x;
    const double inv = 0x1.71547652b82fep+0 * 32.0;
    double kd = __fma_rn(inv, xd, GFK_EXP2F_SHIFT);
    const unsigned long long ki =
        (unsigned long long)__double_as_longlong(kd);
    kd = __dsub_rn(kd, GFK_EXP2F_SHIFT);
    const double r = __fma_rn(inv, xd, -kd);
    unsigned long long t = GFK_EXP2F_TAB[ki & 31ULL];
    t += ki << (52 - 5);
    const double s = __longlong_as_double((long long)t);
    const double z = __fma_rn(GFK_EXP2F_P0 / 32.0 / 32.0 / 32.0, r,
                              GFK_EXP2F_P1 / 32.0 / 32.0);
    const double r2 = __dmul_rn(r, r);
    double y = __fma_rn(GFK_EXP2F_P2 / 32.0, r, 1.0);
    y = __fma_rn(z, r2, y);
    return thompson_aa_d2f(__dmul_rn(y, s));
}

__device__ __forceinline__ float thompson_aa_logf(float x)
{
    return gfk_log(x);
}

__device__ __forceinline__ double thompson_aa_powf_log2(unsigned int ix)
{
    const unsigned int tmp = ix - 0x3f330000u;
    const int i = (int)((tmp >> 19) & 15u);
    const unsigned int top = tmp & 0xff800000u;
    const unsigned int iz = ix - top;
    const int k = (int)top >> 23;
    const double z = (double)__uint_as_float(iz);
    const double r = __fma_rn(z, GFK_POWF_INVC[i], -1.0);
    const double y0 = __dadd_rn(GFK_POWF_LOGC[i], (double)k);
    const double r2 = __dmul_rn(r, r);
    const double y = __fma_rn(GFK_POWF_A[0], r, GFK_POWF_A[1]);
    const double p = __fma_rn(GFK_POWF_A[2], r, GFK_POWF_A[3]);
    const double r4 = __dmul_rn(r2, r2);
    double q = __fma_rn(GFK_POWF_A[4], r, y0);
    q = __fma_rn(p, r2, q);
    return __fma_rn(y, r4, q);
}

__device__ __forceinline__ double thompson_aa_powf_exp2(
        double xd, unsigned long long sign_bias)
{
    double kd = __dadd_rn(xd, GFK_EXP2F_SHIFT_SCALED);
    const unsigned long long ki =
        (unsigned long long)__double_as_longlong(kd);
    kd = __dsub_rn(kd, GFK_EXP2F_SHIFT_SCALED);
    const double r = __dsub_rn(xd, kd);
    unsigned long long t = GFK_EXP2F_TAB[ki & 31ULL];
    t += (ki + sign_bias) << (52 - 5);
    const double s = __longlong_as_double((long long)t);
    const double z = __fma_rn(GFK_EXP2F_P0, r, GFK_EXP2F_P1);
    const double r2 = __dmul_rn(r, r);
    double y = __fma_rn(GFK_EXP2F_P2, r, 1.0);
    y = __fma_rn(z, r2, y);
    return __dmul_rn(y, s);
}

__device__ __forceinline__ float thompson_aa_powf(float x, float y)
{
    unsigned int sign_bias = 0u;
    unsigned int ix = __float_as_uint(x);
    const unsigned int iy = __float_as_uint(y);
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || gfk_zeroinfnan(iy)) {
        if (gfk_zeroinfnan(iy)) {
            if (2u * iy == 0u) return 1.0f;
            if (ix == 0x3f800000u) return 1.0f;
            if (2u * ix > 2u * 0x7f800000u || 2u * iy > 2u * 0x7f800000u)
                return __fadd_rn(x, y);
            if (2u * ix == 2u * 0x3f800000u) return 1.0f;
            if ((2u * ix < 2u * 0x3f800000u) == !(iy & 0x80000000u))
                return 0.0f;
            return __fmul_rn(y, y);
        }
        if (gfk_zeroinfnan(ix)) {
            float x2 = __fmul_rn(x, x);
            if ((ix & 0x80000000u) && gfk_checkint(iy) == 1) x2 = -x2;
            return (iy & 0x80000000u) ? __fdiv_rn(1.0f, x2) : x2;
        }
        if (ix & 0x80000000u) {
            const int yint = gfk_checkint(iy);
            if (yint == 0) return __int_as_float(0x7fc00000);
            if (yint == 1) sign_bias = 1u << (5 + 11);
            ix &= 0x7fffffffu;
        }
        if (ix < 0x00800000u) {
            ix = __float_as_uint(__fmul_rn(x, 8388608.0f)) & 0x7fffffffu;
            ix -= 23u << 23;
        }
    }
    const double logx = thompson_aa_powf_log2(ix);
    const double ylogx = __dmul_rn((double)y, logx);
    const unsigned int hi = (unsigned int)
        (((unsigned long long)__double_as_longlong(ylogx) >> 47) & 0xffffULL);
    if (hi >= (unsigned int)
            (((unsigned long long)__double_as_longlong(126.0) >> 47)
             & 0xffffULL)) {
        if (ylogx > 0x1.fffffffd1d571p+6)
            return sign_bias ? __int_as_float(0xff800000)
                             : __int_as_float(0x7f800000);
        if (ylogx <= -150.0) return sign_bias ? -0.0f : 0.0f;
    }
    return thompson_aa_d2f(
        thompson_aa_powf_exp2(ylogx, (unsigned long long)sign_bias));
}

__device__ __forceinline__ float thompson_aa_log10f(float x)
{
    unsigned int hx = __float_as_uint(x);
    int k = 0;
    if (hx < 0x00800000u || (hx & 0x80000000u)) {
        if ((hx & 0x7fffffffu) == 0u) return __int_as_float(0xff800000);
        if (hx & 0x80000000u) return __int_as_float(0x7fc00000);
        k -= 25;                                  // subnormal: scale by 2^25
        hx = __float_as_uint(__fmul_rn(x, 33554432.0f));
    }
    if (hx > 0x7f7fffffu) return __fadd_rn(x, x);
    k += (int)(hx >> 23) - 127;
    const int i = (int)(((unsigned int)k & 0x80000000u) >> 31);
    hx = (hx & 0x007fffffu) | ((unsigned int)(0x7f - i) << 23);
    const float y = (float)(k + i);
    const float z = __fadd_rn(
        __fmul_rn(__uint_as_float(0x3ede5bd9u),                 // ivln10
                  thompson_aa_logf(__uint_as_float(hx))),
        __fmul_rn(y, __uint_as_float(0x355427dbu)));            // log10_2lo
    return __fadd_rn(z, __fmul_rn(y, __uint_as_float(0x3e9a2080u)));
}

__device__ __forceinline__ double thompson_aa_exp(double x)
{
    return glibc_exp(x);
}

__device__ __forceinline__ double thompson_aa_log(double x)
{
    return glibc_log(x);
}

__device__ __forceinline__ double thompson_aa_pow(double x, double y)
{
    return glibc_pow(x, y);
}

__device__ __forceinline__ double thompson_aa_log10(double x)
{
    unsigned long long hx = (unsigned long long)__double_as_longlong(x);
    long long k;
    if (hx <= 0x000fffffffffffffULL || (hx >> 63)) {
        if ((hx & 0x7fffffffffffffffULL) == 0ULL)
            return __longlong_as_double((long long)0xfff0000000000000ULL);
        if (hx >> 63)
            return __longlong_as_double((long long)0x7ff8000000000000ULL);
        k = -1023 - 54;                           // subnormal: scale by 2^54
        hx = (unsigned long long)__double_as_longlong(__dmul_rn(x, 0x1p54));
    } else {
        k = -1023;
    }
    if (hx > 0x7fefffffffffffffULL) return __dadd_rn(x, x);
    k += (long long)(hx >> 52);
    const long long i = (long long)(((unsigned long long)k) >> 63);
    k += i;
    hx = (hx & 0x000fffffffffffffULL)
        | ((unsigned long long)(0x3ff - i) << 52);
    const double y = (double)k;
    const double z = __dadd_rn(
        __dmul_rn(0x1.bcb7b1526e50ep-2,                          // ivln10
                  thompson_aa_log(__longlong_as_double((long long)hx))),
        __dmul_rn(y, 0x1.9fef311f12b36p-42));                    // log10_2lo
    return __dadd_rn(z, __dmul_rn(y, 0x1.34413509f6000p-2));     // log10_2hi
}

// ---------------------------------------------------------------------------
// hypot, log1p and atan2 for the complex arithmetic of WRF's melting-snow
// reflectivity.  module_mp_radar.F takes ABS, LOG and SQRT of COMPLEX*16
// values, which gfortran lowers to the C library's cabs, clog and csqrt; on
// the oracle host those call hypot, log1p, log and atan2.  Only the mp=28
// reflectivity (thompson_aa_refl10cm in thompson_aerosol_state.cu) reaches
// these three words.
//
//   thompson_aa_hypot   a correction step on sqrt(x*x + y*y) with no fused
//                       multiply-add, the operation order of the oracle
//                       host's hypot: zero mismatches on the sampled
//                       moderate range (libm_check.cpp).  Arguments whose
//                       ratio exceeds 2^54 return the larger plus the
//                       smaller, as the host does; arguments beyond
//                       1e308 are outside the reflectivity's domain.
//   thompson_aa_log1p   the oracle host runs its log1p fused-multiply-add
//                       path (W1's CPU has FMA); every fusion of that path
//                       is an explicit __fma_rn here, the rest pinned
//                       __dmul_rn/__dadd_rn: zero mismatches on the sampled
//                       range (libm_check.cpp).
//   thompson_aa_atan2   a reduction to |t| <= 7/16 on four breakpoints and
//                       an odd minimax polynomial, under 1.5 ulp.  The
//                       host's atan2 is a different and more accurate
//                       routine, so the two agree on most arguments but not
//                       all; libm_check.cpp reports the agreement rate and
//                       the worst error against a 113-bit reference.  It
//                       feeds only the argument of a complex logarithm in a
//                       binary64 chain that WRF rounds to REAL(4) at the end
//                       (ze_snow = SNGL(...) at module_mp_thompson.F:5950),
//                       which a few binary64 ulp do not move; the column
//                       oracle's melting-snow cells are the measurement.
// ---------------------------------------------------------------------------
__device__ __forceinline__ double thompson_aa_hypot(double x, double y)
{
    const double fx = fabs(x), fy = fabs(y);
    const double ax = fx < fy ? fy : fx;
    const double ay = fx < fy ? fx : fy;
    if (!(ax < 1.0e308) || ax == 0.0) return __dadd_rn(ax, ay);
    if (ay <= __dmul_rn(ax, 0x1p-54)) return __dadd_rn(ax, ay);
    const double h = __dsqrt_rn(__dadd_rn(__dmul_rn(ax, ax),
                                          __dmul_rn(ay, ay)));
    double t1, t2;
    if (h <= __dmul_rn(2.0, ay)) {
        const double delta = __dsub_rn(h, ay);
        t1 = __dmul_rn(ax, __dsub_rn(__dmul_rn(2.0, delta), ax));
        t2 = __dmul_rn(__dsub_rn(delta, __dmul_rn(2.0, __dsub_rn(ax, ay))),
                       delta);
    } else {
        const double delta = __dsub_rn(h, ax);
        t1 = __dmul_rn(__dmul_rn(2.0, delta),
                       __dsub_rn(ax, __dmul_rn(2.0, ay)));
        t2 = __dadd_rn(__dmul_rn(__dsub_rn(__dmul_rn(4.0, delta), ay), ay),
                       __dmul_rn(delta, delta));
    }
    return __dsub_rn(h, __ddiv_rn(__dadd_rn(t1, t2), __dmul_rn(2.0, h)));
}

__device__ __forceinline__ double thompson_aa_log1p(double x)
{
    const double ln2_hi = 6.93147180369123816490e-01;
    const double ln2_lo = 1.90821492927058770002e-10;
    const double lp1 = 6.666666666666735130e-01;
    const double lp2 = 3.999999999940941908e-01;
    const double lp3 = 2.857142874366239149e-01;
    const double lp4 = 2.222219843214978396e-01;
    const double lp5 = 1.818357216161805012e-01;
    const double lp6 = 1.531383769920937332e-01;
    const double lp7 = 1.479819860511658591e-01;
    const int hx = (int)((unsigned long long)__double_as_longlong(x) >> 32);
    const int ax = hx & 0x7fffffff;
    int k = 1, hu = 0;
    double f = 0.0, c = 0.0;
    if (hx < 0x3FDA827A) {
        if (ax >= 0x3ff00000) {
            if (x == -1.0)
                return __longlong_as_double((long long)0xfff0000000000000ULL);
            return __longlong_as_double((long long)0x7ff8000000000000ULL);
        }
        if (ax < 0x3e200000) {
            if (ax < 0x3c900000) return x;
            return __fma_rn(-__dmul_rn(x, x), 0.5, x);
        }
        if (hx > 0 || hx <= (int)0xbfd2bec4) { k = 0; f = x; hu = 1; }
    } else if (hx >= 0x7ff00000) {
        return __dadd_rn(x, x);
    }
    if (k != 0) {
        double u;
        if (hx < 0x43400000) {
            u = __dadd_rn(1.0, x);
            hu = (int)((unsigned long long)__double_as_longlong(u) >> 32);
            k = (hu >> 20) - 1023;
            c = k > 0 ? __dsub_rn(1.0, __dsub_rn(u, x))
                      : __dsub_rn(x, __dsub_rn(u, 1.0));
            c = __ddiv_rn(c, u);
        } else {
            u = x;
            hu = (int)((unsigned long long)__double_as_longlong(u) >> 32);
            k = (hu >> 20) - 1023;
            c = 0.0;
        }
        hu &= 0x000fffff;
        const unsigned long long low =
            (unsigned long long)__double_as_longlong(u) & 0xffffffffULL;
        if (hu < 0x6a09e) {
            u = __longlong_as_double((long long)(
                ((unsigned long long)(hu | 0x3ff00000) << 32) | low));
        } else {
            k += 1;
            u = __longlong_as_double((long long)(
                ((unsigned long long)(hu | 0x3fe00000) << 32) | low));
            hu = (0x00100000 - hu) >> 2;
        }
        f = __dsub_rn(u, 1.0);
    }
    const double dk = (double)k;
    const double hfsq = __dmul_rn(__dmul_rn(0.5, f), f);
    if (hu == 0) {
        if (f == 0.0) {
            if (k == 0) return 0.0;
            return __fma_rn(dk, ln2_hi, __fma_rn(dk, ln2_lo, c));
        }
        const double r = __dmul_rn(__fma_rn(-0.66666666666666666, f, 1.0),
                                   hfsq);
        if (k == 0) return __dsub_rn(f, r);
        return __fma_rn(dk, ln2_hi, -__dsub_rn(
            __dsub_rn(r, __fma_rn(dk, ln2_lo, c)), f));
    }
    const double s = __ddiv_rn(f, __dadd_rn(2.0, f));
    const double z = __dmul_rn(s, s);
    const double r2 = __fma_rn(z, lp3, lp2);
    const double r3 = __fma_rn(z, lp5, lp4);
    const double r4 = __fma_rn(z, lp7, lp6);
    const double z2 = __dmul_rn(z, z);
    const double z4 = __dmul_rn(z2, z2);
    const double z6 = __dmul_rn(z4, z2);
    double r = __fma_rn(z, lp1, __dmul_rn(z2, r2));
    r = __fma_rn(z4, r3, r);
    r = __fma_rn(z6, r4, r);
    const double t = __dmul_rn(__dadd_rn(hfsq, r), s);
    if (k == 0) return __dsub_rn(f, __dsub_rn(hfsq, t));
    return __fma_rn(dk, ln2_hi, -__dsub_rn(__dsub_rn(hfsq,
        __dadd_rn(__fma_rn(dk, ln2_lo, c), t)), f));
}

__device__ __forceinline__ double thompson_aa_atan_reduced(double x)
{
    const double at0 = 3.33333333333329318027e-01;
    const double at1 = -1.99999999998764832476e-01;
    const double at2 = 1.42857142725034663711e-01;
    const double at3 = -1.11111104054623557880e-01;
    const double at4 = 9.09088713343650656196e-02;
    const double at5 = -7.69187620504482999495e-02;
    const double at6 = 6.66107313738753120669e-02;
    const double at7 = -5.83357013379057348645e-02;
    const double at8 = 4.97687799461593236017e-02;
    const double at9 = -3.65315727442169155270e-02;
    const double at10 = 1.62858201153657823623e-02;
    // atan(0.5), atan(1), atan(1.5) and pi/2, each as a high and low part.
    const double hi0 = 4.63647609000806093515e-01;
    const double hi1 = 7.85398163397448278999e-01;
    const double hi2 = 9.82793723247329054082e-01;
    const double hi3 = 1.57079632679489655800e+00;
    const double lo0 = 2.26987774529616870924e-17;
    const double lo1 = 3.06161699786838301793e-17;
    const double lo2 = 1.39033110312309984516e-17;
    const double lo3 = 6.12323399573676603587e-17;
    const int hx = (int)((unsigned long long)__double_as_longlong(x) >> 32);
    const int ix = hx & 0x7fffffff;
    double hi = 0.0, lo = 0.0;
    bool reduced = true;
    if (ix >= 0x44100000) {
        const double r = __dadd_rn(hi3, lo3);
        return hx > 0 ? r : -r;
    }
    if (ix < 0x3fdc0000) {
        if (ix < 0x3e400000) return x;
        reduced = false;
    } else {
        x = fabs(x);
        if (ix < 0x3ff30000) {
            if (ix < 0x3fe60000) {
                hi = hi0; lo = lo0;
                x = __ddiv_rn(__dsub_rn(__dmul_rn(2.0, x), 1.0),
                              __dadd_rn(2.0, x));
            } else {
                hi = hi1; lo = lo1;
                x = __ddiv_rn(__dsub_rn(x, 1.0), __dadd_rn(x, 1.0));
            }
        } else if (ix < 0x40038000) {
            hi = hi2; lo = lo2;
            x = __ddiv_rn(__dsub_rn(x, 1.5),
                          __dadd_rn(1.0, __dmul_rn(1.5, x)));
        } else {
            hi = hi3; lo = lo3;
            x = __ddiv_rn(-1.0, x);
        }
    }
    const double z = __dmul_rn(x, x);
    const double w = __dmul_rn(z, z);
    const double s1 = __dmul_rn(z, __dadd_rn(at0, __dmul_rn(w, __dadd_rn(at2,
        __dmul_rn(w, __dadd_rn(at4, __dmul_rn(w, __dadd_rn(at6,
        __dmul_rn(w, __dadd_rn(at8, __dmul_rn(w, at10))))))))))) ;
    const double s2 = __dmul_rn(w, __dadd_rn(at1, __dmul_rn(w, __dadd_rn(at3,
        __dmul_rn(w, __dadd_rn(at5, __dmul_rn(w, __dadd_rn(at7,
        __dmul_rn(w, at9)))))))));
    const double xs = __dmul_rn(x, __dadd_rn(s1, s2));
    if (!reduced) return __dsub_rn(x, xs);
    const double r = __dsub_rn(hi, __dsub_rn(__dsub_rn(xs, lo), x));
    return hx < 0 ? -r : r;
}

__device__ __forceinline__ double thompson_aa_atan2(double y, double x)
{
    const double pi_o_2 = 1.5707963267948965580e+00;
    const double pi = 3.1415926535897931160e+00;
    const double pi_lo = 1.2246467991473531772e-16;
    if (isnan(x) || isnan(y)) return __dadd_rn(x, y);
    const unsigned long long ux = (unsigned long long)__double_as_longlong(x);
    const unsigned long long uy = (unsigned long long)__double_as_longlong(y);
    if (x == 1.0) return thompson_aa_atan_reduced(y);
    // m = 2*sign(x) + sign(y).
    const int m = (int)((uy >> 63) & 1ULL) | (int)(((ux >> 63) & 1ULL) << 1);
    if (y == 0.0) {
        if (m < 2) return y;
        return m == 2 ? pi : -pi;
    }
    if (x == 0.0) return (uy >> 63) ? -pi_o_2 : pi_o_2;
    if (isinf(y)) return (uy >> 63) ? -pi_o_2 : pi_o_2;
    if (isinf(x)) {
        if (m == 0) return 0.0;
        if (m == 1) return -0.0;
        return m == 2 ? pi : -pi;
    }
    const int ix = (int)((ux >> 52) & 0x7ffULL);
    const int iy = (int)((uy >> 52) & 0x7ffULL);
    const int k = iy - ix;
    double z;
    int mm = m;
    if (k > 60) {
        z = __dadd_rn(pi_o_2, __dmul_rn(0.5, pi_lo));
        mm = m & 1;
    } else if ((ux >> 63) && k < -60) {
        z = 0.0;
    } else {
        z = thompson_aa_atan_reduced(fabs(__ddiv_rn(y, x)));
    }
    switch (mm) {
    case 0: return z;
    case 1: return -z;
    case 2: return __dsub_rn(pi, __dsub_rn(z, pi_lo));
    default: return __dsub_rn(__dsub_rn(z, pi_lo), pi);
    }
}

// ======================================================================
// THIRD-PARTY NOTICE. Parts of this file are hand transcriptions of
// third-party work. WOOF distributes the file under the Apache License
// 2.0; the notices below belong to the transcribed parts and are kept
// here because their own licences require it. Full texts are in the
// repository NOTICE, licenses/, and beside the code in
// gpuwm/core/kernels/LICENSE-third-party.txt.
//
// Arm optimized-routines: glibc_sinf / glibc_cosf and their data tables.
// github.com/ARM-software/optimized-routines: math/sinf.c, math/cosf.c,
// math/sincosf.h, math/sincosf_data.c, published in 2018 and imported
// into glibc in August 2018; transcribed here from glibc 2.43
// sysdeps/ieee754/flt-32/s_sinf.c, s_cosf.c, s_sincosf.h,
// sincosf_poly.h and s_sincosf_data.c (x86 FMA IFUNC variant).
//
// Copyright (c) 2018-2024, Arm Limited.
// Copyright (c) 2018-2019, Arm Limited.
// SPDX-License-Identifier: MIT
//
// Taken under the MIT branch of Arm's grant, as glibc_flt32.cuh is.
// The algorithm, constants and tables reproduce Arm's work; glibc's
// integration macros, aliases and IFUNC wrapper are not transcribed.
// Arm 5e8389113b47 (2018), MIT relicensing 11253b0b9d6b,
// current fe09d4e7ed62. The full permission notice is reproduced in
// licenses/LICENSE-Arm-optimized-routines-MIT.txt and the files above.
//
// CORE-MATH MIT transcriptions from sysdeps/ieee754/flt-32/:
// e_asinf.c and s_asincosf_data.c (src/binary32/asin/asinf.c, bc385c2);
// e_acosf.c (src/binary32/acos/acosf.c, 56dd347);
// s_atanf.c (src/binary32/atan/atanf.c, a8066a5);
// s_tanf.c (src/binary32/tan/tanf.c, 59d21d7).
// Copyright (c) 2022-2024 Alexei Sibidanov.
// Copyright (c) 2023-2024 Alexei Sibidanov.
// Full text: licenses/LICENSE-CORE-MATH-MIT.txt.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
// No SunPro FDLIBM code is used.
// ======================================================================
// Target: Ubuntu GLIBC 2.43-2ubuntu2.4, x86-64 AVX2+FMA, FE_TONEAREST.
// sinf/cosf use __sinf_fma/__cosf_fma. The other four scalar functions
// have no CPU IFUNC. Floating exceptions and errno are not device outputs.
// Arithmetic is pinned to the selected libm machine instructions, not CUDA
// trig builtins. Use -std=c++17. No FADD/FMUL macros are defined here.
#ifndef GPUWM_GLIBC_TRIG_FLT32_CUH
#define GPUWM_GLIBC_TRIG_FLT32_CUH
typedef unsigned int gt_u32;
typedef unsigned long long gt_u64;
typedef long long gt_i64;

__device__ inline float gt_d2f(double x) { return __double2float_rn(x); }
__device__ inline double gt_identity(double x) { return x; }
__device__ inline gt_u64 gt_as_u64(double x) { return (gt_u64)__double_as_longlong(x); }
__device__ inline double gt_abs(double x) {
  return __longlong_as_double((long long)(gt_as_u64(x) & 0x7fffffffffffffffULL));
}
__device__ inline float gt_absf(float x) { return __uint_as_float(__float_as_uint(x) & 0x7fffffffU); }
__device__ inline double gt_sign(double x, double y) {
  return __longlong_as_double((long long)((gt_as_u64(x) & 0x7fffffffffffffffULL) | (gt_as_u64(y) & 0x8000000000000000ULL)));
}
__device__ inline float gt_signf(float x, float y) {
  return __uint_as_float((__float_as_uint(x)&0x7fffffffU) | (__float_as_uint(y)&0x80000000U));
}
__device__ inline float gt_invalid(float x) {
  gt_u32 t=__float_as_uint(x);
  return __uint_as_float((t & 0x7fffffffU)>0x7f800000U ? t|0x00400000U : 0xffc00000U);
}
__device__ inline double gt_roundeven(double x) {
  // This helper only receives |x| < 2^28 * 2/pi. Conversion pins RN.
  return (double)__double2ll_rn(x);
}
__device__ inline float gt_tiny_fmaf(float a,float b,float x) {
  // Only called by the three tiny-input paths below, where the fused
  // correction is less than half an ULP in RN. Preserve exponent-zero
  // inputs by bits because CUDA FTZ flushes even an explicit FMA's result.
  // All normal inputs retain glibc's explicit fused operation.
  if ((__float_as_uint(x)&0x7f800000U)==0) return x;
  return __fmaf_rn(a,b,x);
}

// s_sincosf_data.c and sincosf_poly.h.
struct gt_sincos_t { double sign[4], hpi_inv, hpi, c0,c1,c2,c3,c4,s1,s2,s3; };
__device__ const gt_sincos_t gt_sincos_table[2] = {
 {{1,-1,-1,1},0x1.45F306DC9C883p+23,0x1.921FB54442D18p0,
  0x1p0,-0x1.ffffffd0c621cp-2,0x1.55553e1068f19p-5,-0x1.6c087e89a359dp-10,0x1.99343027bf8c3p-16,
  -0x1.555545995a603p-3,0x1.1107605230bc4p-7,-0x1.994eb3774cf24p-13},
 {{1,-1,-1,1},0x1.45F306DC9C883p+23,0x1.921FB54442D18p0,
  -0x1p0,0x1.ffffffd0c621cp-2,-0x1.55553e1068f19p-5,0x1.6c087e89a359dp-10,-0x1.99343027bf8c3p-16,
  -0x1.555545995a603p-3,0x1.1107605230bc4p-7,-0x1.994eb3774cf24p-13}
};
__device__ const gt_u32 gt_inv_pio4[24] = {
 0xa2,0xa2f9,0xa2f983,0xa2f9836e,0xf9836e4e,0x836e4e44,0x6e4e4415,0x4e441529,
 0x441529fc,0x1529fc27,0x29fc2757,0xfc2757d1,0x2757d1f5,0x57d1f534,0xd1f534dd,0xf534ddc0,
 0x34ddc0db,0xddc0db62,0xc0db6295,0xdb629599,0x6295993c,0x95993c43,0x993c4390,0x3c439041
};
__device__ inline float gt_sinf_poly(double x,double x2,const gt_sincos_t* p,int n) {
 if (!(n&1)) {
  double x3=__dmul_rn(x,x2);
  double s1=__fma_rn(x2,p->s3,p->s2);
  double x7=__dmul_rn(x3,x2);
  double s=__fma_rn(x3,p->s1,x);
  return gt_d2f(__fma_rn(x7,s1,s));
 }
 double x4=__dmul_rn(x2,x2);
 double c2=__fma_rn(x2,p->c4,p->c3);
 double c1=__fma_rn(x2,p->c1,p->c0);
 double x6=__dmul_rn(x4,x2);
 double c=__fma_rn(x4,p->c2,c1);
 return gt_d2f(__fma_rn(x6,c2,c));
}
__device__ inline double gt_reduce_large(gt_u32 xi,int* np) {
 const gt_u32* arr=&gt_inv_pio4[(xi>>26)&15];
 int shift=(xi>>23)&7;
 xi=((xi&0xffffffU)|0x800000U)<<shift;
 gt_u64 r0=(gt_u32)(xi*arr[0]);
 gt_u64 r1=(gt_u64)xi*arr[4], r2=(gt_u64)xi*arr[8];
 r0=((r2>>32)|(r0<<32))+r1;
 gt_u64 n=(r0+(1ULL<<61))>>62;
 r0-=n<<62;
 *np=(int)n;
 return __dmul_rn((double)(gt_i64)r0,0x1.921FB54442D18p-62);
}
__device__ inline float gt_sincos(float y,int cosine) {
 gt_u32 yi=__float_as_uint(y), top=(yi>>20)&0x7ffU;
 if (top>=0x7f8U) return gt_invalid(y);
 if (top<0x398U) return cosine ? 1.0f : y;
 double x=(double)y;
 const gt_sincos_t* p=&gt_sincos_table[0];
 int n=0, sign=0;
 if (top>=0x3f4U) {
  if (top<0x42fU) {
   double r=__dmul_rn(x,p->hpi_inv);
   n=((int)r+0x800000)>>24;
   x=__fma_rn(-(double)n,p->hpi,x);
  } else {
   x=gt_reduce_large(yi,&n);
   sign=(int)(yi>>31);
  }
  int q=(n+sign)&3;
  if (q&2) p=&gt_sincos_table[1];
  double x2=__dmul_rn(x,x);
  return gt_sinf_poly(__dmul_rn(x,gt_sincos_table[0].sign[q]),x2,p,n^cosine);
 }
 return gt_sinf_poly(x,__dmul_rn(x,x),p,cosine);
}
__device__ float glibc_sinf(float x) { return gt_sincos(x,0); }
__device__ float glibc_cosf(float x) { return gt_sincos(x,1); }

__device__ const double gt_asincos_c0[12] = {
 0x1.555555555529cp-3,0x1.333333337e0ddp-4,0x1.6db6db3b4465ep-5,
 0x1.f1c72e13ac306p-6,0x1.6e89cebe06bc4p-6,0x1.1c6dcf5289094p-6,
 0x1.c6dbbcc7c6315p-7,0x1.8f8dc2615e996p-7,0x1.a5833b7bf15e8p-8,
 0x1.43f44ace1665cp-6,-0x1.0fb17df881c73p-6,0x1.07520c026b2d6p-5
};
__device__ const double gt_asincos_c1[12] = {
 0x1.6a09e667f3bcbp+0,0x1.e2b7dddff2db9p-4,0x1.b27247ab42dbcp-6,
 0x1.02995cc4e0744p-7,0x1.5ffb0276ec8eap-9,0x1.033885a928decp-10,
 0x1.911f2be23f8c7p-12,0x1.4c3c55d2437fdp-13,0x1.af477e1d7b461p-15,
 0x1.abd6bdff67dcbp-15,-0x1.1717e86d0fa28p-16,0x1.6ff526de46023p-16
};
// s_tanf.c rbig: limb form of glibc's unsigned 128-bit multiplies.
__device__ const gt_u64 gt_tan_ipi[4] = {
 0xfe5163abdebbc562ULL,0xdb6295993c439041ULL,0xfc2757d1f534ddc0ULL,0xa2f9836e4e441529ULL
};
__device__ inline double gt_rbig(gt_u32 u,int* q) {
 gt_u64 m=(u&0x7fffffU)|0x800000U;
 gt_u64 h=__umul64hi(m,gt_tan_ipi[0]);
 gt_u64 p1l=m*gt_tan_ipi[1]+h;
 h=__umul64hi(m,gt_tan_ipi[1])+(p1l<h);
 gt_u64 p2l=m*gt_tan_ipi[2]+h;
 h=__umul64hi(m,gt_tan_ipi[2])+(p2l<h);
 gt_u64 p3l=m*gt_tan_ipi[3]+h;
 gt_u64 p3h=__umul64hi(m,gt_tan_ipi[3])+(p3l<h);
 int s=(int)((u>>23)&255)-127-23;
 gt_u64 a; gt_u32 i;
 if(s<64) { i=(gt_u32)((p3h<<s)|(p3l>>(64-s))); a=(p3l<<s)|(p2l>>(64-s)); }
 else if(s==64) { i=(gt_u32)p3l; a=p2l; }
 else { i=(gt_u32)((p3l<<(s-64))|(p2l>>(128-s))); a=(p2l<<(s-64))|(p1l>>(128-s)); }
 gt_u32 sign=0U-(u>>31);
 i-= (gt_u32)((gt_i64)a>>63);
 i=(i^sign)-sign;
 *q=(int)i;
 gt_i64 az=(gt_i64)(a^(gt_u64)(gt_i64)(int)sign);
 return __dmul_rn((double)az,0x1p-64);
}

// glibc-2.43 sysdeps/ieee754/flt-32/e_asinf.c
__device__ const double gt_asinf_b[] = {0x1.0000000000005p+0, 0x1.55557aeca105dp-3, 0x1.3314ec3db7d12p-4, 0x1.775738a5a6f92p-5, 0x1.5d5f7ce1c8538p-8, 0x1.605c6d58740fp-2, -0x1.5728b732d73c6p+1, 0x1.f152170f151ebp+3, -0x1.f962ea3ca992ep+5, 0x1.71971e17375ap+7, -0x1.860512b4ba23p+8, 0x1.26a3b8d4bdb14p+9, -0x1.36f2ea5698b51p+9, 0x1.b3d722aebfa2ep+8, -0x1.6cf89703b1289p+7, 0x1.1518af6a65e2dp+5};

__device__ static float gt_asinf_as_special(float x)
{
  gt_u32 ax = __float_as_uint(x) << 1;
  if (ax > (0xffu << 24))
    return gt_invalid(x); // Preserve the input NaN sign and payload.
  return gt_invalid(0.0);
}


__device__ static double gt_asinf_poly12(double z, const double *c)
{
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  double c0 = __dadd_rn(c[0], __dmul_rn(z, c[1]));
  double c2 = __dadd_rn(c[2], __dmul_rn(z, c[3]));
  double c4 = __dadd_rn(c[4], __dmul_rn(z, c[5]));
  double c6 = __dadd_rn(c[6], __dmul_rn(z, c[7]));
  double c8 = __dadd_rn(c[8], __dmul_rn(z, c[9]));
  double c10 = __dadd_rn(c[10], __dmul_rn(z, c[11]));
  c0 = __dadd_rn(c0, __dmul_rn(c2, z2));
  c4 = __dadd_rn(c4, __dmul_rn(c6, z2));
  c8 = __dadd_rn(c8, __dmul_rn(z2, c10));
  c0 = __dadd_rn(c0, __dmul_rn(z4, __dadd_rn(c4, __dmul_rn(z4, c8))));
  return c0;
}


__device__ float glibc_asinf(float x)
{
  const double pi2 = 0x1.921fb54442d18p+0;
  double xs = x;
  double r;
  gt_u32 ax = __float_as_uint(x) << 1;
  if (ax > (0x7f << 24))
    return gt_asinf_as_special(x);
  if (ax < 0x7ec29000u)
  {
    if (ax < (115 << 24))
      return gt_tiny_fmaf(x, 0x1p-25f, x);
    /* table gt_asinf_b */;
    double z = xs;
    double z2 = __dmul_rn(z, z);
    double z4 = __dmul_rn(z2, z2);
    double z8 = __dmul_rn(z4, z4);
    double z16 = __dmul_rn(z8, z8);
    r = __dmul_rn(z, __dadd_rn(__dadd_rn(__dadd_rn(__dadd_rn(gt_asinf_b[0], __dmul_rn(z2, gt_asinf_b[1])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[2], __dmul_rn(z2, gt_asinf_b[3])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_asinf_b[4], __dmul_rn(z2, gt_asinf_b[5])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[6], __dmul_rn(z2, gt_asinf_b[7])))))), __dmul_rn(z16, __dadd_rn(__dadd_rn(__dadd_rn(gt_asinf_b[8], __dmul_rn(z2, gt_asinf_b[9])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[10], __dmul_rn(z2, gt_asinf_b[11])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_asinf_b[12], __dmul_rn(z2, gt_asinf_b[13])), __dmul_rn(z4, __dadd_rn(gt_asinf_b[14], __dmul_rn(z2, gt_asinf_b[15])))))))));
    float ub = gt_d2f(r);
    float lb = gt_d2f(__dsub_rn(r, __dmul_rn(z, 0x1.efa8ebp-31)));
    if (ub == lb)
      return ub;
  }
  if (ax < (0x7eu << 24))
  {
    double z = xs;
    double z2 = __dmul_rn(z, z);
    double c0 = gt_asinf_poly12(z2, gt_asincos_c0);
    r = __dadd_rn(z, __dmul_rn(__dmul_rn(z, z2), c0));
  }
  else
  {
    if (ax == 0x7e55688au)
      return __fadd_rn(gt_signf(0x1.75b8a2p-1f, x), gt_signf(0x1p-26f, x));
    if (ax == 0x7e107434u)
      return __fadd_rn(gt_signf(0x1.1f4b64p-1f, x), gt_signf(0x1p-26f, x));
    double bx = gt_abs(xs);
    double z = __dsub_rn(1.0, bx);
    double s = __dsqrt_rn(z);
    r = __dsub_rn(pi2, __dmul_rn(s, gt_asinf_poly12(z, gt_asincos_c1)));
    r = gt_sign(r, xs);
  }
  return gt_d2f(r);
}


// glibc-2.43 sysdeps/ieee754/flt-32/e_acosf.c
__device__ const double gt_acosf_o[] = {0, 0x1.921fb54442d18p+1};
__device__ const double gt_acosf_b[] = {0x1.fffffffd9ccb8p-1, 0x1.5555c94838007p-3, 0x1.32ded4b7c20fap-4, 0x1.8566df703309ep-5, -0x1.980c959bec9a3p-6, 0x1.56fbb04998344p-1, -0x1.403d8e4c49f52p+2, 0x1.b06c3e9f311eap+4, -0x1.9ea97c4e2c21fp+6, 0x1.200b8261cc61bp+8, -0x1.2274c2799a5c7p+9, 0x1.a558a59cc19d3p+9, -0x1.aca4b6a529ffp+9, 0x1.228744703f813p+9, -0x1.d7dbb0b322228p+7, 0x1.5c2018c0c0105p+5};

__device__ static float gt_acosf_as_special(float x)
{
  const float pih = gt_d2f(0x1.921fb6p+1);
  const float pil = -0x1p-24f;
  gt_u32 t = __float_as_uint(x);
  if (t == (0x7fu << 23))
    return 0.0f;
  if (t == (0x17fu << 23))
    return __fadd_rn(pih, pil);
  gt_u32 ax = t << 1;
  if (ax > (0xffu << 24))
    return gt_invalid(x); // Preserve the input NaN sign and payload.
  return gt_invalid(0.0);
}


__device__ inline static double gt_acosf_poly12(double z, const double *c)
{
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  double c0 = __dadd_rn(c[0], __dmul_rn(z, c[1]));
  double c2 = __dadd_rn(c[2], __dmul_rn(z, c[3]));
  double c4 = __dadd_rn(c[4], __dmul_rn(z, c[5]));
  double c6 = __dadd_rn(c[6], __dmul_rn(z, c[7]));
  double c8 = __dadd_rn(c[8], __dmul_rn(z, c[9]));
  double c10 = __dadd_rn(c[10], __dmul_rn(z, c[11]));
  c0 = __dadd_rn(c0, __dmul_rn(c2, z2));
  c4 = __dadd_rn(c4, __dmul_rn(c6, z2));
  c8 = __dadd_rn(c8, __dmul_rn(z2, c10));
  c0 = __dadd_rn(c0, __dmul_rn(z4, __dadd_rn(c4, __dmul_rn(z4, c8))));
  return c0;
}


__device__ float glibc_acosf(float x)
{
  double pi2 = 0x1.921fb54442d18p+0;
  /* table gt_acosf_o */;
  double xs = x;
  double r;
  gt_u32 t = __float_as_uint(x);
  gt_u32 ax = t << 1;
  if (ax >= (0x7f << 24))
    return gt_acosf_as_special(x);
  if (ax < 0x7ec2a1dcu)
  {
    /* table gt_acosf_b */;
    if (ax <= 0x40000000u)
      return gt_d2f(gt_identity(pi2));
    double z = xs;
    double z2 = __dmul_rn(z, z);
    double z4 = __dmul_rn(z2, z2);
    double z8 = __dmul_rn(z4, z4);
    double z16 = __dmul_rn(z8, z8);
    r = __dmul_rn(z, __dadd_rn(__dadd_rn(__dadd_rn(__dadd_rn(gt_acosf_b[0], __dmul_rn(z2, gt_acosf_b[1])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[2], __dmul_rn(z2, gt_acosf_b[3])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_acosf_b[4], __dmul_rn(z2, gt_acosf_b[5])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[6], __dmul_rn(z2, gt_acosf_b[7])))))), __dmul_rn(z16, __dadd_rn(__dadd_rn(__dadd_rn(gt_acosf_b[8], __dmul_rn(z2, gt_acosf_b[9])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[10], __dmul_rn(z2, gt_acosf_b[11])))), __dmul_rn(z8, __dadd_rn(__dadd_rn(gt_acosf_b[12], __dmul_rn(z2, gt_acosf_b[13])), __dmul_rn(z4, __dadd_rn(gt_acosf_b[14], __dmul_rn(z2, gt_acosf_b[15])))))))));
    float ub = gt_d2f(__dsub_rn(0x1.921fb54574191p+0, r));
    float lb = gt_d2f(__dsub_rn(0x1.921fb543118ap+0, r));
    if (ub == lb)
      return ub;
  }
  if (ax < (0x7eu << 24))
  {
    if (t == 0x328885a3u)
      return __fadd_rn(0x1.921fb6p+0f, 0x1p-25f);
    if (t == 0x39826222u)
      return __fadd_rn(0x1.920f6ap+0f, 0x1p-25f);
    double x2 = __dmul_rn(xs, xs);
    r = __dsub_rn(__dsub_rn(pi2, xs), __dmul_rn(__dmul_rn(xs, x2), gt_acosf_poly12(x2, gt_asincos_c0)));
  }
  else
  {
    double bx = gt_abs(xs);
    double z = __dsub_rn(1.0, bx);
    double s = gt_sign(__dsqrt_rn(z), xs);
    r = __dadd_rn(gt_acosf_o[t >> 31], __dmul_rn(s, gt_acosf_poly12(z, gt_asincos_c1)));
  }
  return gt_d2f(r);
}


// glibc-2.43 sysdeps/ieee754/flt-32/s_atanf.c
__device__ const double gt_atanf_cn[] = {0x1.51eccde075d67p-2, 0x1.a76bb5637f2f2p-1, 0x1.81e0eed20de88p-1, 0x1.376c8ca67d11dp-2, 0x1.aec7b69202ac6p-5, 0x1.9561899acc73ep-9, 0x1.bf9fa5b67e6p-16};
__device__ const double gt_atanf_cd[] = {0x1.51eccde075d66p-2, 0x1.dfbdd7b392d28p-1, 0x1p+0, 0x1.fd22bf0e89b54p-2, 0x1.d91ff8b576282p-4, 0x1.653ea99fc9bbp-7, 0x1.1e7fcc202340ap-12};

__device__ float glibc_atanf(float x)
{
  const double pi2 = 0x1.921fb54442d18p+0;
  gt_u32 t = __float_as_uint(x);
  int e = (t >> 23) & 0xff;
  bool gt = e >= 127;
  gt_u32 ta = t & 0x7fffffff;
  if (ta >= 0x4c700518u)
  {
    if (ta > 0x7f800000u)
      return gt_invalid(x); // Preserve the input NaN sign and payload.
    return gt_d2f(gt_sign(pi2, (double) x));
  }
  if (e < (127 - 13))
  {
    if (e < (127 - 25))
    {
      if (!(t << 1))
        return x;
      return gt_tiny_fmaf(-x, gt_absf(x), x);
    }
    return __fmaf_rn(__fmul_rn(-0x1.555556p-2f, x), __fmul_rn(x, x), x);
  }
  double z = x;
  if (gt)
    z = __ddiv_rn(1, z);
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  double z8 = __dmul_rn(z4, z4);
  /* table gt_atanf_cn */;
  /* table gt_atanf_cd */;
  double cn0 = __dadd_rn(gt_atanf_cn[0], __dmul_rn(z2, gt_atanf_cn[1]));
  double cn2 = __dadd_rn(gt_atanf_cn[2], __dmul_rn(z2, gt_atanf_cn[3]));
  double cn4 = __dadd_rn(gt_atanf_cn[4], __dmul_rn(z2, gt_atanf_cn[5]));
  double cn6 = gt_atanf_cn[6];
  cn0 = __dadd_rn(cn0, __dmul_rn(z4, cn2));
  cn4 = __dadd_rn(cn4, __dmul_rn(z4, cn6));
  cn0 = __dadd_rn(cn0, __dmul_rn(z8, cn4));
  cn0 = __dmul_rn(cn0, z);
  double cd0 = __dadd_rn(gt_atanf_cd[0], __dmul_rn(z2, gt_atanf_cd[1]));
  double cd2 = __dadd_rn(gt_atanf_cd[2], __dmul_rn(z2, gt_atanf_cd[3]));
  double cd4 = __dadd_rn(gt_atanf_cd[4], __dmul_rn(z2, gt_atanf_cd[5]));
  double cd6 = gt_atanf_cd[6];
  cd0 = __dadd_rn(cd0, __dmul_rn(z4, cd2));
  cd4 = __dadd_rn(cd4, __dmul_rn(z4, cd6));
  cd0 = __dadd_rn(cd0, __dmul_rn(z8, cd4));
  double r = __ddiv_rn(cn0, cd0);
  if (!gt)
    return gt_d2f(r);
  r = __dadd_rn(__dsub_rn(gt_sign(0x1.0fdaa22168c23p-7, z), r), gt_sign(0x1.9p0, z));
  return gt_d2f(r);
}


// glibc-2.43 sysdeps/ieee754/flt-32/s_tanf.c
__device__ const double gt_tanf_cn[] = {0x1.921fb54442d18p+0, -0x1.fd226e573289fp-2, 0x1.b7a60c8dac9f6p-6, -0x1.725beb40f33e5p-13};
__device__ const double gt_tanf_cd[] = {0x1p+0, -0x1.2395347fb829dp+0, 0x1.2313660f29c36p-3, -0x1.9a707ab98d1c1p-9};
__device__ const double gt_tanf_s[] = {0, 1};
__device__ const struct 
    {
      float arg;
      float rh;
      float rl;
    } gt_tanf_st[] = {{0x1.143ec4p+0f, 0x1.ddf9f6p+0f, -0x1.891d24p-52f}, {0x1.ada6aap+27f, 0x1.e80304p-3f, 0x1.419f46p-58f}, {0x1.af61dap+48f, 0x1.60d1c8p-2f, -0x1.2d6c3ap-55f}, {0x1.0088bcp+52f, 0x1.ca1edp+0f, 0x1.f6053p-53f}, {0x1.f90dfcp+72f, 0x1.597f9cp-1f, 0x1.925978p-53f}, {0x1.cc4e22p+85f, -0x1.f33584p+1f, 0x1.d7254ap-51f}, {0x1.a6ce12p+86f, -0x1.c5612ep-1f, -0x1.26c33ep-53f}, {0x1.6a0b76p+102f, -0x1.e42a1ep+0f, -0x1.1dc906p-52f}};

__device__ inline static double gt_tanf_rltl(float z, int *q)
{
  double x = z;
  double idl = __dmul_rn(-0x1.b1bbead603d8bp-32, x);
  double idh = __dmul_rn(0x1.45f306ep-1, x);
  double id = gt_roundeven(idh);
  *q = (gt_i64) id;
  return __dadd_rn(__dsub_rn(idh, id), idl);
}


__device__ float glibc_tanf(float x)
{
  gt_u32 t = __float_as_uint(x);
  int e = (t >> 23) & 0xff;
  int i;
  double z;
  if (e < (127 + 28))
  {
    if (e < 115)
    {
      if (e < 102)
        return gt_tiny_fmaf(x, gt_absf(x), x);
      float x2 = __fmul_rn(x, x);
      return __fmaf_rn(x, __fmul_rn(0x1.555556p-2f, x2), x);
    }
    z = gt_tanf_rltl(x, &i);
  }
  else
    if (e < 0xff)
    z = gt_rbig(t, &i);
  else
  {
    if (t << 9)
      return gt_invalid(x); // Preserve the input NaN sign and payload.
    return gt_invalid(x);
  }
  double z2 = __dmul_rn(z, z);
  double z4 = __dmul_rn(z2, z2);
  /* table gt_tanf_cn */;
  /* table gt_tanf_cd */;
  /* table gt_tanf_s */;
  double n = __dadd_rn(gt_tanf_cn[0], __dmul_rn(z2, gt_tanf_cn[1]));
  double n2 = __dadd_rn(gt_tanf_cn[2], __dmul_rn(z2, gt_tanf_cn[3]));
  n = __dadd_rn(n, __dmul_rn(z4, n2));
  double d = __dadd_rn(gt_tanf_cd[0], __dmul_rn(z2, gt_tanf_cd[1]));
  double d2 = __dadd_rn(gt_tanf_cd[2], __dmul_rn(z2, gt_tanf_cd[3]));
  d = __dadd_rn(d, __dmul_rn(z4, d2));
  n = __dmul_rn(n, z);
  double s0 = gt_tanf_s[i & 1];
  double s1 = gt_tanf_s[1 - (i & 1)];
  double r1 = __ddiv_rn(__dsub_rn(__dmul_rn(n, s1), __dmul_rn(d, s0)), __dadd_rn(__dmul_rn(n, s0), __dmul_rn(d, s1)));
  gt_u64 tail = (gt_as_u64(r1) + 7) & ((~0ULL) >> 35);
  if (tail <= 14)
  {
    /* table gt_tanf_st */;
    gt_u32 ax = t & ((~0u) >> 1);
    gt_u32 sgn = t >> 31;
    for (int j = 0; j < 8; j++)
    {
      if (__float_as_uint(gt_tanf_st[j].arg) == ax)
      {
        if (sgn)
          return __fsub_rn(-gt_tanf_st[j].rh, gt_tanf_st[j].rl);
        else
          return __fadd_rn(gt_tanf_st[j].rh, gt_tanf_st[j].rl);
      }
    }

  }
  return gt_d2f(r1);
}


#endif

// log10f: CORE-MATH src/binary32/log10/log10f.c (revision bc385c2) as
// glibc 2.43 carries it in sysdeps/ieee754/flt-32/e_log10f.c (no x86-64
// IFUNC variant), transcribed line for line; correctly rounded.
//   Copyright (c) 2022-2023 Alexei Sibidanov.
//   SPDX-License-Identifier: MIT.  Permission is hereby granted, free of
//   charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without
//   restriction, including without limitation the rights to use, copy,
//   modify, merge, publish, distribute, sublicense, and/or sell copies of
//   the Software, and to permit persons to whom the Software is furnished to
//   do so, subject to the following conditions: The above copyright notice
//   and this permission notice shall be included in all copies or
//   substantial portions of the Software.  THE SOFTWARE IS PROVIDED "AS IS",
//   WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
//   Full text: licenses/LICENSE-CORE-MATH-MIT.txt.
__device__ const double ubm_l10_tr[65] = {
  0x1p+0,         0x1.f81f82p-1,  0x1.f07c1fp-1,  0x1.e9131acp-1,
  0x1.e1e1e1ep-1, 0x1.dae6077p-1, 0x1.d41d41dp-1, 0x1.cd85689p-1,
  0x1.c71c71cp-1, 0x1.c0e0704p-1, 0x1.bacf915p-1, 0x1.b4e81b5p-1,
  0x1.af286bdp-1, 0x1.a98ef6p-1,  0x1.a41a41ap-1, 0x1.9ec8e95p-1,
  0x1.999999ap-1, 0x1.948b0fdp-1, 0x1.8f9c19p-1,  0x1.8acb90fp-1,
  0x1.8618618p-1, 0x1.8181818p-1, 0x1.7d05f41p-1, 0x1.78a4c81p-1,
  0x1.745d174p-1, 0x1.702e05cp-1, 0x1.6c16c17p-1, 0x1.6816817p-1,
  0x1.642c859p-1, 0x1.605816p-1,  0x1.5c9882cp-1, 0x1.58ed231p-1,
  0x1.5555555p-1, 0x1.51d07ebp-1, 0x1.4e5e0a7p-1, 0x1.4afd6ap-1,
  0x1.47ae148p-1, 0x1.446f865p-1, 0x1.4141414p-1, 0x1.3e22cbdp-1,
  0x1.3b13b14p-1, 0x1.3813814p-1, 0x1.3521cfbp-1, 0x1.323e34ap-1,
  0x1.2f684bep-1, 0x1.2c9fb4ep-1, 0x1.29e412ap-1, 0x1.27350b9p-1,
  0x1.2492492p-1, 0x1.21fb781p-1, 0x1.1f7047ep-1, 0x1.1cf06aep-1,
  0x1.1a7b961p-1, 0x1.1811812p-1, 0x1.15b1e5fp-1, 0x1.135c811p-1,
  0x1.1111111p-1, 0x1.0ecf56cp-1, 0x1.0c9715p-1,  0x1.0a6810ap-1,
  0x1.0842108p-1, 0x1.0624dd3p-1, 0x1.041041p-1,  0x1.0204081p-1,
  0.5};
__device__ const double ubm_l10_tl[65] = {
  -0x1.d45fd6237ebe3p-47, 0x1.b947689311b6ep-8, 0x1.b5e909c96d7d5p-7,
  0x1.45f4f59ed2165p-6,   0x1.af5f92cbd8f1ep-6, 0x1.0ba01a606de8cp-5,
  0x1.3ed119b9a2b7bp-5,   0x1.714834298eec2p-5, 0x1.a30a9d98357fbp-5,
  0x1.d41d512670813p-5,   0x1.02428c0f65519p-4, 0x1.1a23444eecc3ep-4,
  0x1.31b30543f4cb4p-4,   0x1.48f3ed39bfd04p-4, 0x1.5fe8049a0e423p-4,
  0x1.769140a6aa008p-4,   0x1.8cf1836c98cb3p-4, 0x1.a30a9d55541a1p-4,
  0x1.b8de4d1ee823ep-4,   0x1.ce6e4202ca2e6p-4, 0x1.e3bc1accace07p-4,
  0x1.f8c9683b5abd4p-4,   0x1.06cbd68ca9a6ep-3, 0x1.11142f19df73p-3,
  0x1.1b3e71fa7a97fp-3,   0x1.254b4d37a46e3p-3, 0x1.2f3b6912cbf07p-3,
  0x1.390f683115886p-3,   0x1.42c7e7fffc5a8p-3, 0x1.4c65808c78d3cp-3,
  0x1.55e8c50751c55p-3,   0x1.5f52445dec3d8p-3, 0x1.68a288c3f12p-3,
  0x1.71da17bdf0d19p-3,   0x1.7af973608afd9p-3, 0x1.84011952a2579p-3,
  0x1.8cf1837a7ea6p-3,    0x1.95cb2891e43d6p-3, 0x1.9e8e7b0f869ep-3,
  0x1.a73beaa5db18dp-3,   0x1.afd3e394558d3p-3, 0x1.b856cf060d9f1p-3,
  0x1.c0c5134de1ffcp-3,   0x1.c91f1371bc99fp-3, 0x1.d1652ffcd3f53p-3,
  0x1.d997c6f635e75p-3,   0x1.e1b733ab90f3bp-3, 0x1.e9c3ceadac856p-3,
  0x1.f1bdeec43a305p-3,   0x1.f9a5e7a5fa3fep-3, 0x1.00be05ac02f2bp-2,
  0x1.04a054d81a2d4p-2,   0x1.087a0835957fbp-2, 0x1.0c4b457099517p-2,
  0x1.101431aa1fe51p-2,   0x1.13d4f08b98dd8p-2, 0x1.178da53edb892p-2,
  0x1.1b3e71e9f9d58p-2,   0x1.1ee777defdeedp-2, 0x1.2288d7b48e23bp-2,
  0x1.2622b0f52e49fp-2,   0x1.29b522a4c6314p-2, 0x1.2d404b0e30f8p-2,
  0x1.30c4478f3fbe5p-2,   0x1.34413509f7915p-2};
__device__ const float ubm_l10_st[16] = {
  0x1p+0f,        0x1.4p+3f,      0x1.9p+6f,       0x1.f4p+9f,
  0x1.388p+13f,   0x1.86ap+16f,   0x1.e848p+19f,   0x1.312dp+23f,
  0x1.7d784p+26f, 0x1.dcd65p+29f, 0x1.2a05f2p+33f, 0.0f,
  0.0f,           0.0f,           0.0f,            0.0f};
__device__ const double ubm_l10_b[3] = {
  0x1.bcb7b15c5a2f8p-2, -0x1.bcbb1dbb88ebap-3, 0x1.2871c39d521c6p-3};
__device__ const double ubm_l10_c[7] = {
  0x1.bcb7b1526e50ep-2,  -0x1.bcb7b1526e53dp-3, 0x1.287a7636f3fa2p-3,
  -0x1.bcb7b146a14b3p-4, 0x1.63c627d5219cbp-4,  -0x1.2880736c8762dp-4,
  0x1.fc1ecf913961ap-5};
__device__ float ubm_log10f_special(float x)
{
  unsigned int ux = __float_as_uint(x);
  if (ux == 0x7f800000u)
    return x; /* +inf */
  unsigned int ax = ux << 1;
  if (ax == 0u)
    return __uint_as_float(0xff800000u); /* +-0: __math_divzerof (1) = -inf */
  if (ax > 0xff000000u)
    return x + x; /* nan */
  return __uint_as_float(0x7fc00000u); /* x < 0: __math_invalidf */
}
__device__ float ubm_log10f(float x)
{
  unsigned int ux = __float_as_uint(x);
  if (ux < (1u << 23) || ux >= 0x7f800000u)
    {
      if (ux == 0 || ux >= 0x7f800000u)
        return ubm_log10f_special(x);
      /* subnormal */
      int n = __clz(ux) - 8;
      ux <<= n;
      ux -= n << 23;
    }
  unsigned int m = ux & ((1u << 23) - 1), j = (m + (1u << (23 - 7))) >> (23 - 6);
  double ix = ubm_l10_tr[j], l = ubm_l10_tl[j];
  int e = ((int) ux >> 23) - 127;
  unsigned int je = e + 1;
  je = (je * 0x4d104d4u) >> 28;
  if (ux == __float_as_uint(ubm_l10_st[je]))
    return (float) je;
  double tz = __longlong_as_double(((long long) m | ((long long) 1023 << 23)) << (52 - 23));
  double z = __dsub_rn(__dmul_rn(tz, ix), 1.0), z2 = __dmul_rn(z, z);
  double r = __dadd_rn(
      __dadd_rn(__dadd_rn(__dmul_rn((double) e, 0x1.34413509f79ffp-2), l),
                __dmul_rn(z, ubm_l10_b[0])),
      __dmul_rn(z2, __dadd_rn(ubm_l10_b[1], __dmul_rn(z, ubm_l10_b[2]))));
  float ub = __double2float_rn(r), lb = __double2float_rn(__dadd_rn(r, 0x1.b008p-34));
  if (ub != lb)
    {
      const double *c = ubm_l10_c;
      double t456 = __dadd_rn(__dadd_rn(c[4], __dmul_rn(z, c[5])), __dmul_rn(z2, c[6]));
      double t23 = __dadd_rn(c[2], __dmul_rn(z, c[3]));
      double t01 = __dadd_rn(c[0], __dmul_rn(z, c[1]));
      double f = __dmul_rn(z, __dadd_rn(t01, __dmul_rn(z2, __dadd_rn(t23, __dmul_rn(z2, t456)))));
      f = __dsub_rn(f, __dmul_rn(0x1.0cee0ed4ca7e9p-54, (double) e));
      f = __dadd_rn(f, __dsub_rn(l, ubm_l10_tl[0]));
      double el = __dmul_rn((double) e, 0x1.34413509f7ap-2);
      r = __dadd_rn(el, f);
      ub = __double2float_rn(r);
      tz = r;
      if (!((((unsigned long long) __double_as_longlong(tz)) & ((1 << 28) - 1))))
        {
          double dr = __dadd_rn(__dsub_rn(el, r), f);
          r = __dadd_rn(r, __dmul_rn(dr, 32.0));
          ub = __double2float_rn(r);
        }
    }
  return ub;
}

struct thompson_aa_complex { double re, im; };

__device__ __forceinline__ thompson_aa_complex thompson_aa_cmake(
    double re, double im)
{
    thompson_aa_complex z; z.re = re; z.im = im; return z;
}

__device__ __forceinline__ thompson_aa_complex thompson_aa_cmul(
    thompson_aa_complex a, thompson_aa_complex b)
{
    return thompson_aa_cmake(
        __dsub_rn(__dmul_rn(a.re, b.re), __dmul_rn(a.im, b.im)),
        __dadd_rn(__dmul_rn(a.re, b.im), __dmul_rn(a.im, b.re)));
}

__device__ __forceinline__ thompson_aa_complex thompson_aa_cscale(
    double s, thompson_aa_complex a)
{
    return thompson_aa_cmake(__dmul_rn(s, a.re), __dmul_rn(s, a.im));
}

__device__ __forceinline__ thompson_aa_complex thompson_aa_csub(
    thompson_aa_complex a, thompson_aa_complex b)
{
    return thompson_aa_cmake(__dsub_rn(a.re, b.re), __dsub_rn(a.im, b.im));
}

__device__ __forceinline__ thompson_aa_complex thompson_aa_cadd(
    thompson_aa_complex a, thompson_aa_complex b)
{
    return thompson_aa_cmake(__dadd_rn(a.re, b.re), __dadd_rn(a.im, b.im));
}

__device__ __forceinline__ thompson_aa_complex thompson_aa_cdiv(
    thompson_aa_complex a, thompson_aa_complex b)
{
    if (fabs(b.re) < fabs(b.im)) {
        const double ratio = __ddiv_rn(b.re, b.im);
        const double div = __dadd_rn(__dmul_rn(b.re, ratio), b.im);
        const double tr = __dadd_rn(__dmul_rn(a.re, ratio), a.im);
        const double ti = __dsub_rn(__dmul_rn(a.im, ratio), a.re);
        return thompson_aa_cmake(__ddiv_rn(tr, div), __ddiv_rn(ti, div));
    }
    const double ratio = __ddiv_rn(b.im, b.re);
    const double div = __dadd_rn(__dmul_rn(b.im, ratio), b.re);
    const double tr = __dadd_rn(__dmul_rn(a.im, ratio), a.re);
    const double ti = __dsub_rn(a.im, __dmul_rn(a.re, ratio));
    return thompson_aa_cmake(__ddiv_rn(tr, div), __ddiv_rn(ti, div));
}

__device__ __forceinline__ double thompson_aa_cabs(thompson_aa_complex z)
{
    return thompson_aa_hypot(z.re, z.im);
}

// csqrt on finite arguments away from the overflow and underflow
// thresholds (refractive indices and their mixtures).
__device__ __forceinline__ thompson_aa_complex thompson_aa_csqrt(
    thompson_aa_complex z)
{
    if (z.im == 0.0) {
        if (z.re < 0.0) {
            return thompson_aa_cmake(0.0, copysign(__dsqrt_rn(-z.re), z.im));
        }
        return thompson_aa_cmake(fabs(__dsqrt_rn(z.re)), copysign(0.0, z.im));
    }
    if (z.re == 0.0) {
        const double r = __dsqrt_rn(__dmul_rn(0.5, fabs(z.im)));
        return thompson_aa_cmake(r, copysign(r, z.im));
    }
    const double d = thompson_aa_hypot(z.re, z.im);
    double r, s;
    if (z.re > 0.0) {
        r = __dsqrt_rn(__dmul_rn(0.5, __dadd_rn(d, z.re)));
        s = __dmul_rn(0.5, __ddiv_rn(z.im, r));
    } else {
        s = __dsqrt_rn(__dmul_rn(0.5, __dsub_rn(d, z.re)));
        r = fabs(__dmul_rn(0.5, __ddiv_rn(z.im, s)));
    }
    return thompson_aa_cmake(r, copysign(s, z.im));
}

// x*x + y*y - 1 for 0.5 <= x < 1, summed exactly from the split products
// in increasing magnitude and rounded once, as the C library's clog forms
// it near the unit circle.
__device__ __forceinline__ double thompson_aa_x2y2m1(double x, double y)
{
    double v[5];
    v[1] = __dmul_rn(x, x);
    v[0] = __fma_rn(x, x, -v[1]);
    v[3] = __dmul_rn(y, y);
    v[2] = __fma_rn(y, y, -v[3]);
    v[4] = -1.0;
    // Insertion sort by magnitude, then each partial sum carried exactly.
    for (int a = 1; a < 5; ++a) {
        const double t = v[a];
        int b = a - 1;
        while (b >= 0 && fabs(v[b]) > fabs(t)) { v[b + 1] = v[b]; --b; }
        v[b + 1] = t;
    }
    for (int n = 0; n <= 3; ++n) {
        const double hi = __dadd_rn(v[n + 1], v[n]);
        const double lo = __dadd_rn(__dsub_rn(v[n + 1], hi), v[n]);
        v[n + 1] = hi;
        v[n] = lo;
        for (int a = n + 2; a < 5; ++a) {
            const double t = v[a];
            int b = a - 1;
            while (b >= n + 1 && fabs(v[b]) > fabs(t)) {
                v[b + 1] = v[b];
                --b;
            }
            v[b + 1] = t;
        }
    }
    return __dadd_rn(__dadd_rn(__dadd_rn(__dadd_rn(v[4], v[3]), v[2]), v[1]),
                     v[0]);
}

// clog on finite nonzero arguments away from the overflow and underflow
// thresholds.
__device__ __forceinline__ thompson_aa_complex thompson_aa_clog(
    thompson_aa_complex z)
{
    double absx = fabs(z.re), absy = fabs(z.im);
    if (absx < absy) { const double t = absx; absx = absy; absy = t; }
    const double eps = 0x1p-52;
    double real;
    if (absx == 1.0) {
        real = __dmul_rn(thompson_aa_log1p(__dmul_rn(absy, absy)), 0.5);
    } else if (absx > 1.0 && absx < 2.0 && absy < 1.0) {
        double d2m1 = __dmul_rn(__dsub_rn(absx, 1.0), __dadd_rn(absx, 1.0));
        if (absy >= eps) d2m1 = __dadd_rn(d2m1, __dmul_rn(absy, absy));
        real = __dmul_rn(thompson_aa_log1p(d2m1), 0.5);
    } else if (absx < 1.0 && absx >= 0.5 && absy < __dmul_rn(eps, 0.5)) {
        const double d2m1 = __dmul_rn(__dsub_rn(absx, 1.0),
                                      __dadd_rn(absx, 1.0));
        real = __dmul_rn(thompson_aa_log1p(d2m1), 0.5);
    } else if (absx < 1.0 && absx >= 0.5
               && __dadd_rn(__dmul_rn(absx, absx), __dmul_rn(absy, absy))
                   >= 0.5) {
        real = __dmul_rn(thompson_aa_log1p(thompson_aa_x2y2m1(absx, absy)),
                         0.5);
    } else {
        real = thompson_aa_log(thompson_aa_hypot(absx, absy));
    }
    return thompson_aa_cmake(real, thompson_aa_atan2(z.im, z.re));
}


__device__ float mp_exp(float x) {return thompson_aa_expf(x);}
__device__ double mp_exp(double x) {return thompson_aa_exp(x);}
__device__ float mp_log(float x) {return thompson_aa_logf(x);}
__device__ double mp_log(double x) {return thompson_aa_log(x);}
__device__ float mp_log10(float x) {return ubm_log10f(x);}
__device__ double mp_log10(double x) {return thompson_aa_log10(x);}


__device__ float mpas_powf(float x,float y) {return thompson_aa_powf(x,y);}
__device__ bool tc_nan(float a) {return (__float_as_uint(a)&0x7fffffffU)>0x7f800000U;}

__device__ float tc_quiet(float a) {return __uint_as_float(__float_as_uint(a)|0x00400000U);}

__device__ float tc_pow(float a,float b) {float r=mpas_powf(a,b);if(!tc_nan(r))return r;if(tc_nan(a))return tc_quiet(a);if(tc_nan(b))return tc_quiet(b);return __uint_as_float(0xffc00000U);}

struct cd { double re, im; };

__device__ __forceinline__ cd cd_make(double re, double im)
{
    cd c; c.re = re; c.im = im; return c;
}

__device__ __forceinline__ cd cd_div(cd a, cd b)
{
    double d = b.re * b.re + b.im * b.im;
    return cd_make((a.re * b.re + a.im * b.im) / d,
                   (a.im * b.re - a.re * b.im) / d);
}


__device__ float mp_pow(float x,float y) {return tc_pow(x,y);}
__device__ double mp_pow(double x,double y) {return thompson_aa_pow(x,y);}
__device__ float mp_sqrt(float x) {return __fsqrt_rn(x);}
__device__ double mp_sqrt(double x) {return __dsqrt_rn(x);}
__device__ float mp_sin(float x) {return glibc_sinf(x);}
__device__ float mp_cos(float x) {return glibc_cosf(x);}
__device__ float mp_atan(float x) {return glibc_atanf(x);}
__device__ double mp_cos(double x) {return uw_cos(x);}
// Existing library calls, explicitly uncertified diagnostic operations.
// No new approximation, native repair, or production fallback is supplied.
__device__ double mp_sin(double x) {return sin(x);}
__device__ double mp_atan(double x) {return atan(x);}
__device__ cx mp_log(cx x) {auto r=thompson_aa_clog(thompson_aa_cmake(x.real(),x.imag()));return cx(r.re,r.im);}
__device__ cx mp_sqrt(cx x) {auto r=thompson_aa_csqrt(thompson_aa_cmake(x.real(),x.imag()));return cx(r.re,r.im);}
__device__ cx4 mp_sqrt(cx4 x) {return thrust::sqrt(x);}
__device__ double mp_abs(cx x) {return thompson_aa_cabs(thompson_aa_cmake(x.real(),x.imag()));}
__device__ cx mp_complex_div(cx x,cx y) {auto r=thompson_aa_cdiv(thompson_aa_cmake(x.real(),x.imag()),thompson_aa_cmake(y.real(),y.imag()));return cx(r.re,r.im);}
__device__ cx4 mp_complex_div(cx4 x,cx4 y) {return x/y;}
"""
