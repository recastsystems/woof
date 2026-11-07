// WRF v4.6.1 MYNN PBL core kernels.
//
// The first kernel is a direct FP32 transcription of
// module_bl_mynn.F:mym_level2. One thread computes one adjacent-level pair.

// ===========================================================================
// gfortran emits one rounded SSE instruction per Fortran operator; NVRTC is
// free to contract a*b+c into a single FMA, and that one fused rounding is
// enough to break bitwise parity with the pinned oracle.  Arithmetic written
// through these helpers is one rounded PTX instruction per Fortran operator.
// Do not rewrite them as plain operators.
// ===========================================================================
// Pinning round-to-nearest is necessary but not sufficient.  CuPy appends
// `-ftz=true` to every NVRTC compile unconditionally
// (cupy/cuda/compiler.py:607, cupy-cuda12x 14.1.1), which makes ptxas emit
// the `.ftz` form of every compiler-generated FP32 instruction: subnormal
// operands read as zero and subnormal results are flushed to zero.  gfortran
// on x86-64 SSE2 does neither, so a subnormal anywhere in the column is an
// automatic CPU/GPU divergence that no rounding intrinsic can repair -- the
// `-ftz` flag is appended after the caller's options, so it cannot be
// overridden through `RawModule(options=...)` either.
//
// This is not hypothetical: the mass-flux tendency fixture drives sqc2 to
// 3.6e-42 (a subnormal) at one level of every column, and `__fdiv_rn` there
// returned +0.0 while the pinned WRF oracle carries the subnormal.
//
// Inline PTX is passed through to ptxas verbatim, so writing the instruction
// without the `.ftz` modifier restores IEEE subnormal behaviour regardless of
// the compile flag.  Do not rewrite these as plain operators or as the
// `__f*_rn` intrinsics: both are subject to `-ftz`.
//
// They are declared here, ahead of every kernel, for a third reason: a
// host-side constant expression written with bare C operators is folded by
// the compiler, not by the hardware, and ptxas 12.x rounds FP32 ties the
// wrong way when it folds.  A `const real` built from literals is exactly
// that expression, so it belongs in these helpers too, wherever the value
// feeds arithmetic that has to match the oracle bit for bit.
__device__ __forceinline__ real mynn_add(real a, real b)
{
    real r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ real mynn_sub(real a, real b)
{
    real r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ real mynn_mul(real a, real b)
{
    real r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ real mynn_div(real a, real b)
{
    real r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

#define MYNN_ADD(x, y) mynn_add((x), (y))
#define MYNN_SUB(x, y) mynn_sub((x), (y))
#define MYNN_MUL(x, y) mynn_mul((x), (y))
#define MYNN_DIV(x, y) mynn_div((x), (y))

// FP64 has no `.ftz` form, so the intrinsics are enough there; they are still
// spelled out because their only job is to make NVRTC's contraction pass a
// no-op.  The host reference this block mirrors is a chain of plain
// Python-float operations, so a fused multiply-add anywhere in it is a
// different number.
#define MYNN_DADD(x, y) __dadd_rn((x), (y))
#define MYNN_DSUB(x, y) __dsub_rn((x), (y))
#define MYNN_DMUL(x, y) __dmul_rn((x), (y))

// Fortran MAX/MIN of two finite reals, in the argument order the reference
// transcription uses: the second argument wins only on a strict compare.
// The compare is inline PTX for the same reason the arithmetic is: under
// `-ftz=true` a plain `b > a` reads a subnormal b as zero, so MAX(0, tiny)
// would return 0 where gfortran returns tiny.  `fminf`/`fmaxf` have the same
// defect -- ptxas emits their `.ftz` form -- so they must not be used for a
// Fortran MIN/MAX whose result feeds oracle-gated arithmetic.
__device__ __forceinline__ bool mynn_gt(real a, real b)
{
    unsigned int p;
    asm("{ .reg .pred q; setp.gt.f32 q, %1, %2; selp.u32 %0, 1, 0, q; }"
        : "=r"(p) : "f"(a), "f"(b));
    return p != 0u;
}

__device__ __forceinline__ real mynn_max2(real a, real b)
{
    return mynn_gt(b, a) ? b : a;
}

__device__ __forceinline__ real mynn_min2(real a, real b)
{
    return mynn_gt(a, b) ? b : a;
}

// ===========================================================================
// glibc 2.39 FP32 elementary functions, on the device.
//
// module_bl_mynn.F:7525-7623 phim/phih are two per-column scalars and they
// were the last piece of the MYNN driver still evaluated on the host, at a
// measured 149 us per column, flat in column count -- 37 s per timestep on a
// quarter-million-column nest.  They could not use the `mynn_powf` /
// `mynn_atanf` pair further down this file, which rounds an FP64 evaluation:
// glibc's atanf is faithfully rather than correctly rounded, so FP64-then-
// round is a *third* function, and the (1 - phi_m)/zet cancellation in the
// unstable arm amplifies the one-ULP disagreement.  Measured against
// gpuwm/data/mynn/oracle/stfunc.csv, that pair missed 22 of 406 unstable
// phim rows by up to 80 ULP and 9 phih rows by up to 84 ULP.  So this block
// transcribes glibc's own algorithms: e_logf.c, e_powf.c on the exp2 core of
// e_exp2f_data.c, and the fdlibm s_atanf.c kernel.
//
// The operation order is gpuwm/core/noahmp_libm.py's -- the copy audited
// against the live glibc 2.39 on the oracle host -- and NOT the -mfma rebuild
// that gpuwm/core/kernels/noahmp_bareflux.cu transcribes.  Every FP64
// multiply-add here is therefore a separate rounded multiply and add.  Two
// transcriptions of one function are a liability; this second one exists only
// because an NVRTC module cannot see another .cu file, and both are pinned to
// the same host reference by the probe kernel below plus the sweep gates in
// tests/test_mynn_pbl_gpu.py.
//
// Rule: the tables are __constant__ bit patterns, never literal arrays.
// ptxas 12.x's constant folder does not honour round-to-nearest-even when it
// folds FP32 literals, and a mis-folded table entry stays invisible until an
// argument reaches it.  __constant__ memory is the only remedy that measured
// clean here; volatile, asm volatile and __device__ static const all failed.
// ===========================================================================

// e_logf_data.c: 16 (invc, logc) pairs, then ln2 and the three coefficients.
__constant__ unsigned long long MYNN_LOGF_TAB[32] = {
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
__constant__ unsigned long long MYNN_LOGF_MISC[4] = {
    0x3FE62E42FEFA39EFULL, 0xBFD00EA348B88334ULL,
    0x3FD5575B0BE00B6AULL, 0xBFDFFFFEF20A4123ULL,
};

// e_powf_log2_data.c.  POWF_SCALE is 1.0 because TOINT_INTRINSICS is 0 on
// x86-64, so every "* POWF_SCALE" in glibc's own source is a no-op here.
__constant__ unsigned long long MYNN_POWF_LOG2_TAB[32] = {
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
__constant__ unsigned long long MYNN_POWF_LOG2_POLY[5] = {
    0x3FD27616C9496E0BULL, 0xBFD71969A075C67AULL, 0x3FDEC70A6CA7BADDULL,
    0xBFE7154748BEF6C8ULL, 0x3FF71547652AB82BULL,
};

// e_exp2f_data.c, EXP2F_TABLE_BITS = 5.
__constant__ unsigned long long MYNN_EXP2F_TAB[32] = {
    0x3FF0000000000000ULL, 0x3FEFD9B0D3158574ULL, 0x3FEFB5586CF9890FULL,
    0x3FEF9301D0125B51ULL, 0x3FEF72B83C7D517BULL, 0x3FEF54873168B9AAULL,
    0x3FEF387A6E756238ULL, 0x3FEF1E9DF51FDEE1ULL, 0x3FEF06FE0A31B715ULL,
    0x3FEEF1A7373AA9CBULL, 0x3FEEDEA64C123422ULL, 0x3FEECE086061892DULL,
    0x3FEEBFDAD5362A27ULL, 0x3FEEB42B569D4F82ULL, 0x3FEEAB07DD485429ULL,
    0x3FEEA47EB03A5585ULL, 0x3FEEA09E667F3BCDULL, 0x3FEE9F75E8EC5F74ULL,
    0x3FEEA11473EB0187ULL, 0x3FEEA589994CCE13ULL, 0x3FEEACE5422AA0DBULL,
    0x3FEEB737B0CDC5E5ULL, 0x3FEEC49182A3F090ULL, 0x3FEED503B23E255DULL,
    0x3FEEE89F995AD3ADULL, 0x3FEEFF76F2FB5E47ULL, 0x3FEF199BDD85529CULL,
    0x3FEF3720DCEF9069ULL, 0x3FEF5818DCFBA487ULL, 0x3FEF7C97337B9B5FULL,
    0x3FEFA4AFA2A490DAULL, 0x3FEFD0765B6E4540ULL,
};
__constant__ unsigned long long MYNN_EXP2F_POLY[3] = {
    0x3FAC6AF84B912394ULL, 0x3FCEBFCE50FAC4F3ULL, 0x3FE62E42FF0C52D6ULL,
};
// [0] is 0x1.8p52/32, the shift powf's exp2 uses; [1] is the y*log2(x)
// overflow limit 0x1.fffffffd1d571p+6; [2] is 126.0, the abstop screen; [3]
// is -150.0, the underflow limit.
__constant__ unsigned long long MYNN_POWF_MISC[4] = {
    0x42E8000000000000ULL, 0x405FFFFFFFD1D571ULL,
    0x405F800000000000ULL, 0xC062C00000000000ULL,
};

// s_atanf.c atanhi[4], atanlo[4], aT[11].  aT[0] is 0x3EAAAAAB, the value
// glibc's decimal literal rounds to, NOT the 0x3eaaaaaa its own source
// comment claims; trusting the comment costs 1 ULP on the |x| < 0.4375 arm.
__constant__ unsigned int MYNN_ATANF_TAB[19] = {
    0x3EED6338u, 0x3F490FDAu, 0x3F7B985Eu, 0x3FC90FDAu,
    0x31AC3769u, 0x33222168u, 0x33140FB4u, 0x33A22168u,
    0x3EAAAAABu, 0xBE4CCCCDu, 0x3E124925u, 0xBDE38E38u,
    0x3DBA2E6Eu, 0xBD9D8795u, 0x3D886B35u, 0xBD6EF16Bu,
    0x3D4BDA59u, 0xBD15A221u, 0x3C8569D7u,
};

// 0x1p23, the subnormal rescale both logf and powf use.
__constant__ unsigned int MYNN_LIBM_F32[1] = { 0x4B000000u };

// glibc 2.39 on an FMA host (every oracle host: x86-64 with FMA/AVX2) runs
// the ifunc-selected -mfma builds of e_logf.c, e_powf.c and e_expf.c
// (__logf_fma, __powf_fma, __expf_fma), whose a*b+c sites GCC contracted.
// lane/mynn-exact measured it on the oracle CPU (Xeon 8559C, glibc 2.39):
// over every positive normal float32 the fused forms below match the live
// libm on logf and on powf for every exponent this unit uses, and on every
// |x| < 88 for expf, where the unfused forms miss 0-2 inputs each (expf at
// 0x4202422F and 0xC27C65D9, the two points gpuwm/core/noahmp_libm.py had
// recorded as unexplained: glibc's expf also contracts InvLn2N*x into both
// the shift add and r).  atanf and tanhf are not ifuncs and stay unfused.
#define MYNN_DFMA(a, b, c) __fma_rn((a), (b), (c))

// The binary32 result of a binary64 value, INCLUDING glibc's subnormal
// results, which __double2float_rn flushes under CuPy's -ftz=true.
__device__ __forceinline__ real mynn_d2f_rn(double y)
{
    double a = fabs(y);
    if (a > 0.0 && a < 1.1754943508222875e-38) {
        double scaled = rint(MYNN_DMUL(a, 7.1362384635297994e+44));
        unsigned int s = (__double_as_longlong(y) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(s | (unsigned int) scaled);
    }
    return __double2float_rn(y);
}

// glibc 2.39 logf -- sysdeps/ieee754/flt-32/e_logf.c, FMA variant
__device__ real mynn_glibc_logf(real x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3F800000u) return 0.0f;              // log(1) is +0 exactly
    if ((ix - 0x00800000u) >= (0x7F800000u - 0x00800000u)) {
        if ((ix * 2u) == 0u) return __uint_as_float(0xFF800000u);
        if (ix == 0x7F800000u) return x;
        if ((ix & 0x80000000u) || (ix * 2u) >= 0xFF000000u)
            return __uint_as_float(0x7FC00000u);
        ix = __float_as_uint(
            MYNN_MUL(x, __uint_as_float(MYNN_LIBM_F32[0])));
        ix -= (23u << 23);
    }
    unsigned int tmp = ix - 0x3F330000u;
    unsigned int i = (tmp >> 19) & 15u;
    int k = ((int) tmp) >> 23;
    unsigned int iz = ix - (tmp & 0xFF800000u);
    double invc = __longlong_as_double(MYNN_LOGF_TAB[2 * i]);
    double logc = __longlong_as_double(MYNN_LOGF_TAB[2 * i + 1]);
    double ln2 = __longlong_as_double(MYNN_LOGF_MISC[0]);
    double a0 = __longlong_as_double(MYNN_LOGF_MISC[1]);
    double a1 = __longlong_as_double(MYNN_LOGF_MISC[2]);
    double a2 = __longlong_as_double(MYNN_LOGF_MISC[3]);
    double z = (double) __uint_as_float(iz);
    double r = MYNN_DFMA(z, invc, -1.0);
    double y0 = MYNN_DFMA((double) k, ln2, logc);
    double r2 = MYNN_DMUL(r, r);
    double y = MYNN_DFMA(a1, r, a2);
    y = MYNN_DFMA(r2, a0, y);
    y = MYNN_DFMA(r2, y, MYNN_DADD(y0, r));
    return mynn_d2f_rn(y);
}

// e_powf.c log2_inline
__device__ double mynn_glibc_powf_log2(unsigned int ix)
{
    unsigned int tmp = ix - 0x3F330000u;
    unsigned int i = (tmp >> 19) & 15u;
    unsigned int top = tmp & 0xFF800000u;
    unsigned int iz = ix - top;
    int k = ((int) top) >> 23;
    double invc = __longlong_as_double(MYNN_POWF_LOG2_TAB[2 * i]);
    double logc = __longlong_as_double(MYNN_POWF_LOG2_TAB[2 * i + 1]);
    double a0 = __longlong_as_double(MYNN_POWF_LOG2_POLY[0]);
    double a1 = __longlong_as_double(MYNN_POWF_LOG2_POLY[1]);
    double a2 = __longlong_as_double(MYNN_POWF_LOG2_POLY[2]);
    double a3 = __longlong_as_double(MYNN_POWF_LOG2_POLY[3]);
    double a4 = __longlong_as_double(MYNN_POWF_LOG2_POLY[4]);
    double z = (double) __uint_as_float(iz);
    double r = MYNN_DFMA(z, invc, -1.0);
    double y0 = MYNN_DADD(logc, (double) k);
    double r2 = MYNN_DMUL(r, r);
    double y = MYNN_DFMA(a0, r, a1);
    double p = MYNN_DFMA(a2, r, a3);
    double r4 = MYNN_DMUL(r2, r2);
    double q = MYNN_DFMA(a4, r, y0);
    q = MYNN_DFMA(p, r2, q);
    return MYNN_DFMA(y, r4, q);
}

// e_exp2f_data.c's 32-entry exp2 core, as e_powf.c calls it.
__device__ double mynn_glibc_powf_exp2(double xd, unsigned long long bias)
{
    double shift = __longlong_as_double(MYNN_POWF_MISC[0]);
    double kd = MYNN_DADD(xd, shift);
    unsigned long long ki = (unsigned long long) __double_as_longlong(kd);
    kd = MYNN_DSUB(kd, shift);
    double r = MYNN_DSUB(xd, kd);
    unsigned long long t = MYNN_EXP2F_TAB[ki & 31ULL];
    t += ((ki + bias) << (52 - 5));
    double s = __longlong_as_double((long long) t);
    double c0 = __longlong_as_double(MYNN_EXP2F_POLY[0]);
    double c1 = __longlong_as_double(MYNN_EXP2F_POLY[1]);
    double c2 = __longlong_as_double(MYNN_EXP2F_POLY[2]);
    double z = MYNN_DFMA(c0, r, c1);
    double r2 = MYNN_DMUL(r, r);
    double y = MYNN_DFMA(c2, r, 1.0);
    y = MYNN_DFMA(z, r2, y);
    return MYNN_DMUL(y, s);
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_expf.c on the same 32-entry exp2 table
// (lane/mynn-exact).  CUDA's expf is a different function; a real-argument
// EXP in module_bl_mynn.F is this one.  Kept in the unit, beside its logf/
// powf/atanf, for the reason the block above gives: an NVRTC module cannot
// see another .cu file, and the tests that compile this source themselves
// must get the same function.  Constants: InvLn2N = 0x1.71547652b82fep+0*32,
// SHIFT = 0x1.8p+52, poly_scaled = C/N^3, C/N^2, C/N (e_exp2f_data.c); the
// FMA variant (see the note at logf) and mynn_d2f_rn's subnormal results.
__constant__ unsigned long long MYNN_EXPF_MISC[5] = {
    0x40471547652B82FEULL, 0x4338000000000000ULL,
    0x3EBC6AF84B912394ULL, 0x3F2EBFCE50FAC4F3ULL, 0x3F962E42FF0C52D6ULL,
};

__device__ real mynn_glibc_expf(real x)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int abstop = (ix >> 20) & 0x7FFu;
    if (abstop >= 0x42Bu) {                       // |x| >= 88 or NaN/inf
        if (ix == 0xFF800000u) return 0.0f;
        if (abstop >= 0x7F8u) return MYNN_ADD(x, x);
        if (mynn_gt(x, __uint_as_float(0x42B17218u)))
            return __uint_as_float(0x7F800000u);
        if (mynn_gt(__uint_as_float(0xC2CFF1B4u), x)) return 0.0f;
    }
    double inv = __longlong_as_double(MYNN_EXPF_MISC[0]);
    double xd = (double) x;
    double shift = __longlong_as_double(MYNN_EXPF_MISC[1]);
    double kd = MYNN_DFMA(inv, xd, shift);
    unsigned long long ki = (unsigned long long) __double_as_longlong(kd);
    kd = MYNN_DSUB(kd, shift);
    double r = MYNN_DFMA(inv, xd, -kd);
    unsigned long long t = MYNN_EXP2F_TAB[ki & 31ULL];
    t += (ki << (52 - 5));
    double s = __longlong_as_double((long long) t);
    double c0 = __longlong_as_double(MYNN_EXPF_MISC[2]);
    double c1 = __longlong_as_double(MYNN_EXPF_MISC[3]);
    double c2 = __longlong_as_double(MYNN_EXPF_MISC[4]);
    double p = MYNN_DFMA(c0, r, c1);
    double r2 = MYNN_DMUL(r, r);
    double y = MYNN_DFMA(c2, r, 1.0);
    y = MYNN_DMUL(MYNN_DFMA(p, r2, y), s);
    return mynn_d2f_rn(y);
}

// e_powf.c checkint: 0 = not an integer, 1 = odd, 2 = even.
__device__ __forceinline__ int mynn_glibc_powf_checkint(unsigned int iy)
{
    int e = (int) ((iy >> 23) & 0xFFu);
    if (e < 0x7F) return 0;
    if (e > 0x7F + 23) return 2;
    if (iy & ((1u << (0x7F + 23 - e)) - 1u)) return 0;
    if (iy & (1u << (0x7F + 23 - e))) return 1;
    return 2;
}

__device__ __forceinline__ bool mynn_glibc_zeroinfnan(unsigned int ix)
{
    return (2u * ix - 1u) >= (2u * 0x7F800000u - 1u);
}

// glibc 2.39 powf -- sysdeps/ieee754/flt-32/e_powf.c, whole domain.  The
// stable arm reaches base +0 (zet == 0 exactly), which is why the special
// cases are transcribed rather than refused.
__device__ real mynn_glibc_powf(real x, real y)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int iy = __float_as_uint(y);
    unsigned long long bias = 0ULL;
    if ((ix - 0x00800000u) >= (0x7F800000u - 0x00800000u)
            || mynn_glibc_zeroinfnan(iy)) {
        if (mynn_glibc_zeroinfnan(iy)) {
            if ((2u * iy) == 0u) return 1.0f;
            if (ix == 0x3F800000u) return 1.0f;
            if ((2u * ix) > (2u * 0x7F800000u)
                    || (2u * iy) > (2u * 0x7F800000u))
                return MYNN_ADD(x, y);
            if ((2u * ix) == (2u * 0x3F800000u)) return 1.0f;
            if (((2u * ix) < (2u * 0x3F800000u))
                    == ((iy & 0x80000000u) == 0u))
                return 0.0f;
            return MYNN_MUL(y, y);
        }
        if (mynn_glibc_zeroinfnan(ix)) {
            real squared = MYNN_MUL(x, x);
            if ((ix & 0x80000000u) && mynn_glibc_powf_checkint(iy) == 1)
                squared = -squared;
            return (iy & 0x80000000u) ? MYNN_DIV(1.0f, squared) : squared;
        }
        if (ix & 0x80000000u) {
            int yint = mynn_glibc_powf_checkint(iy);
            if (yint == 0) return __uint_as_float(0x7FC00000u);
            if (yint == 1) bias = (unsigned long long) (1u << (5 + 11));
            ix &= 0x7FFFFFFFu;
        }
        if (ix < 0x00800000u) {
            ix = __float_as_uint(MYNN_MUL(
                x, __uint_as_float(MYNN_LIBM_F32[0]))) & 0x7FFFFFFFu;
            ix -= (23u << 23);
        }
    }
    double logx = mynn_glibc_powf_log2(ix);
    double ylogx = MYNN_DMUL((double) y, logx);
    unsigned long long ab = (unsigned long long) __double_as_longlong(ylogx);
    unsigned long long lim =
        ((unsigned long long) MYNN_POWF_MISC[2]) >> 47;
    if (((ab >> 47) & 0xFFFFULL) >= lim) {
        if (ylogx > __longlong_as_double(MYNN_POWF_MISC[1]))
            return bias ? __uint_as_float(0xFF800000u)
                        : __uint_as_float(0x7F800000u);
        if (ylogx <= __longlong_as_double(MYNN_POWF_MISC[3]))
            return bias ? -0.0f : 0.0f;
    }
    return mynn_d2f_rn(mynn_glibc_powf_exp2(ylogx, bias));
}

// glibc 2.39 atanf -- sysdeps/ieee754/flt-32/s_atanf.c, the fdlibm kernel.
// atanf is not an ifunc in libm.so.6, so there is no -mfma rebuild of it:
// every operation is a plain FP32 multiply or add and none may be contracted.
__device__ real mynn_glibc_atanf(real x)
{
    unsigned int hx = __float_as_uint(x);
    unsigned int ix = hx & 0x7FFFFFFFu;
    int signed_hx = (int) hx;
    int id;
#define MYNN_ATAN_HI(i) __uint_as_float(MYNN_ATANF_TAB[(i)])
#define MYNN_ATAN_LO(i) __uint_as_float(MYNN_ATANF_TAB[4 + (i)])
#define MYNN_ATAN_T(i)  __uint_as_float(MYNN_ATANF_TAB[8 + (i)])
    if (ix >= 0x4C000000u) {                     // |x| >= 2**25
        if (ix > 0x7F800000u) return MYNN_ADD(x, x);
        if (signed_hx > 0)
            return MYNN_ADD(MYNN_ATAN_HI(3), MYNN_ATAN_LO(3));
        return MYNN_SUB(-MYNN_ATAN_HI(3), MYNN_ATAN_LO(3));
    }
    if (ix < 0x3EE00000u) {                      // |x| < 0.4375
        if (ix < 0x31000000u) return x;          // |x| < 2**-29
        id = -1;
    } else {
        x = __uint_as_float(ix);                 // fabsf
        if (ix < 0x3F980000u) {                  // |x| < 1.1875
            if (ix < 0x3F300000u) {              // 7/16 <= |x| < 11/16
                id = 0;
                x = MYNN_DIV(MYNN_SUB(MYNN_MUL(2.0f, x), 1.0f),
                             MYNN_ADD(2.0f, x));
            } else {                             // 11/16 <= |x| < 19/16
                id = 1;
                x = MYNN_DIV(MYNN_SUB(x, 1.0f), MYNN_ADD(x, 1.0f));
            }
        } else if (ix < 0x401C0000u) {           // |x| < 2.4375
            id = 2;
            x = MYNN_DIV(MYNN_SUB(x, 1.5f),
                         MYNN_ADD(1.0f, MYNN_MUL(1.5f, x)));
        } else {                                 // 2.4375 <= |x| < 2**25
            id = 3;
            x = MYNN_DIV(-1.0f, x);
        }
    }
    real z = MYNN_MUL(x, x);
    real w = MYNN_MUL(z, z);
    real s1 = MYNN_MUL(w, MYNN_ATAN_T(10));
    s1 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(8), s1));
    s1 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(6), s1));
    s1 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(4), s1));
    s1 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(2), s1));
    s1 = MYNN_MUL(z, MYNN_ADD(MYNN_ATAN_T(0), s1));
    real s2 = MYNN_MUL(w, MYNN_ATAN_T(9));
    s2 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(7), s2));
    s2 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(5), s2));
    s2 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(3), s2));
    s2 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(1), s2));
    real s = MYNN_ADD(s1, s2);
    if (id < 0) return MYNN_SUB(x, MYNN_MUL(x, s));
    real r = MYNN_SUB(MYNN_ATAN_HI(id),
                      MYNN_SUB(MYNN_SUB(MYNN_MUL(x, s), MYNN_ATAN_LO(id)), x));
    return (signed_hx < 0) ? -r : r;
#undef MYNN_ATAN_HI
#undef MYNN_ATAN_LO
#undef MYNN_ATAN_T
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_log10f.c, the function a real-argument
// LOG10 links to: FP32 arithmetic on top of glibc's own logf (above).  The
// same algorithm as noahmp_leaves.cu's r_log10, which the Noah-MP lane
// verified against the live glibc 2.39 (gpuwm/core/noahmp_libm.py log10f);
// spelled with the non-FTZ MYNN operators.  CUDA's log10f is a different
// function (up to 2 ULP), and mym_condensation's rh_hack reads it.
__device__ real mynn_glibc_log10f(real x)
{
    const real ivln10 = __int_as_float(0x3ede5bd9);
    const real log10_2hi = __int_as_float(0x3e9a2080);
    const real log10_2lo = __int_as_float(0x355427db);
    const real two25 = __int_as_float(0x4c000000);
    unsigned int hx = __float_as_uint(x);
    int k = 0;
    if ((int)hx < 0x00800000) {
        if ((hx & 0x7fffffffu) == 0u) return __int_as_float(0xff800000);
        if ((int)hx < 0) return __int_as_float(0x7fc00000);
        k -= 25;
        x = MYNN_MUL(x, two25);
        hx = __float_as_uint(x);
    }
    if (hx >= 0x7f800000u) return MYNN_ADD(x, x);
    k += (int)(hx >> 23) - 127;
    int i = (k < 0) ? 1 : 0;
    hx = (hx & 0x007fffffu) | ((unsigned int)(0x7f - i) << 23);
    real y = (real)(k + i);
    x = __uint_as_float(hx);
    real z = MYNN_ADD(MYNN_MUL(y, log10_2lo),
                      MYNN_MUL(ivln10, mynn_glibc_logf(x)));
    return MYNN_ADD(z, MYNN_MUL(y, log10_2hi));
}

// glibc tanhf (fdlibm s_tanhf.c on expm1f), transcribed further down.
__device__ real mynn_tanhf(real x);

// ===========================================================================
// module_bl_mynn.F:7525-7623 phim / phih, bl_mynn_stfunc = 1.
//
// The Fortran returns phi_m, not phi_m - zet; the subtraction that builds pmz
// happens at the call site (:1095).  Constants are :7537-7539, :7590-7592 and
// :272-273.  1.0/2.5 and 1.0/1.1 go through MYNN_DIV rather than being
// written as literals: div.rn.f32 is correctly rounded, so it reproduces the
// host's F(F(1.0)/F(1.1)) exactly, and inline PTX is opaque to the folder.
// ===========================================================================
// `zet >= 0.0f` picks the arm, and it may NOT be written as a bare FP32
// comparison.  CuPy appends `-ftz=true` to every NVRTC compile, so ptxas emits
// `setp.ge.ftz.f32`, which reads a negative subnormal zet as -0.0 -- and -0.0
// compares >= 0, so the thread takes the STABLE arm, where glibc's powf sees a
// negative base under the non-integer exponent 2.5 and correctly returns NaN.
// gfortran on x86-64 SSE2 does not flush, so WRF and the CPU reference take
// the unstable arm there and return 1.0.  Measured: over a strided sweep of
// the whole clamped [-20, 20] domain the bare comparison put 16,383 of
// 4,300,801 points -- every |zet| below FLT_MIN with a negative sign -- on NaN
// for both phim and phih, and a NaN in pmz/phh is a NaN in the entire PBL.
// This is exact IEEE `x >= 0.0` for every input including NaN, and it cannot
// be flushed because it never compares a float.
__device__ __forceinline__ bool mynn_stfunc_stable_arm(real zet)
{
    unsigned int bits = __float_as_uint(zet);
    if ((bits << 1) > 0xFF000000u) return false;   // NaN: `zet >= 0` is false
    if ((bits << 1) == 0u) return true;            // both zeros compare >= 0
    return (bits & 0x80000000u) == 0u;
}

__device__ real mynn_phi_stable(real zet, real a_st, real b_st, real rb_st)
{
    real dummy_0 = MYNN_ADD(1.0f, mynn_glibc_powf(zet, b_st));
    real dummy_1 = MYNN_ADD(zet, mynn_glibc_powf(dummy_0, rb_st));
    real dummy_11 = MYNN_ADD(1.0f, MYNN_MUL(
        mynn_glibc_powf(dummy_0, MYNN_SUB(rb_st, 1.0f)),
        mynn_glibc_powf(zet, MYNN_SUB(b_st, 1.0f))));
    real dummy_2 = MYNN_MUL(MYNN_DIV(-a_st, dummy_1), dummy_11);
    return MYNN_SUB(1.0f, MYNN_MUL(zet, dummy_2));
}

__device__ real mynn_phi_unstable(real zet, real phi, real dummy_psi,
                                  real a_unst)
{
    real dummy_0 = MYNN_SUB(1.0f, MYNN_MUL(a_unst, zet));
    real dummy_1 = mynn_glibc_powf(dummy_0, 0.333333f);
    real dummy_11 = MYNN_MUL(MYNN_MUL(-0.33333f, a_unst),
                             mynn_glibc_powf(dummy_0, -0.6666667f));
    real dummy_2 = MYNN_MUL(0.33333f, MYNN_ADD(MYNN_ADD(
        mynn_glibc_powf(dummy_1, 2.0f), dummy_1), 1.0f));
    real dummy_22 = MYNN_MUL(MYNN_MUL(0.3333f, dummy_11),
                             MYNN_ADD(MYNN_MUL(2.0f, dummy_1), 1.0f));
    real dummy_3 = MYNN_MUL(0.57735f, MYNN_ADD(MYNN_MUL(2.0f, dummy_1), 1.0f));
    real dummy_33 = MYNN_MUL(1.1547f, dummy_11);
    real dummy_4 = MYNN_ADD(MYNN_SUB(
        MYNN_MUL(1.5f, mynn_glibc_logf(dummy_2)),
        MYNN_MUL(1.73205f, mynn_glibc_atanf(dummy_3))), 1.813799364f);
    real dummy_44 = MYNN_SUB(
        MYNN_MUL(MYNN_DIV(1.5f, dummy_2), dummy_22),
        MYNN_DIV(MYNN_MUL(1.73205f, dummy_33),
                 MYNN_ADD(1.0f, MYNN_MUL(dummy_3, dummy_3))));
    dummy_0 = MYNN_MUL(zet, zet);
    dummy_1 = MYNN_DIV(1.0f, MYNN_ADD(1.0f, dummy_0));
    dummy_11 = MYNN_MUL(2.0f, zet);
    dummy_2 = MYNN_ADD(MYNN_ADD(MYNN_DIV(MYNN_SUB(1.0f, phi), zet),
                                MYNN_MUL(dummy_11, dummy_4)),
                       MYNN_MUL(dummy_0, dummy_44));
    dummy_2 = MYNN_MUL(dummy_2, dummy_1);
    dummy_22 = MYNN_MUL(
        -MYNN_MUL(dummy_11, MYNN_ADD(dummy_psi, MYNN_MUL(dummy_0, dummy_4))),
        MYNN_MUL(dummy_1, dummy_1));
    return MYNN_SUB(1.0f, MYNN_MUL(zet, MYNN_ADD(dummy_2, dummy_22)));
}

__device__ real mynn_phim(real zet)
{
    if (mynn_stfunc_stable_arm(zet))
        return mynn_phi_stable(zet, 6.1f, 2.5f, MYNN_DIV(1.0f, 2.5f));
    real dummy_0 = mynn_glibc_powf(
        MYNN_SUB(1.0f, MYNN_MUL(16.0f, zet)), 0.25f);
    real phi_m = MYNN_DIV(1.0f, dummy_0);
    real dummy_psi = MYNN_ADD(MYNN_SUB(MYNN_ADD(
        MYNN_MUL(2.0f, mynn_glibc_logf(
            MYNN_MUL(0.5f, MYNN_ADD(1.0f, dummy_0)))),
        mynn_glibc_logf(MYNN_MUL(0.5f, MYNN_ADD(
            1.0f, MYNN_MUL(dummy_0, dummy_0))))),
        MYNN_MUL(2.0f, mynn_glibc_atanf(dummy_0))), 1.570796f);
    return mynn_phi_unstable(zet, phi_m, dummy_psi, 10.0f);
}

__device__ real mynn_phih(real zet)
{
    if (mynn_stfunc_stable_arm(zet))
        return mynn_phi_stable(zet, 5.3f, 1.1f, MYNN_DIV(1.0f, 1.1f));
    real dummy_0 = mynn_glibc_powf(
        MYNN_SUB(1.0f, MYNN_MUL(16.0f, zet)), 0.5f);
    real phh = MYNN_DIV(1.0f, dummy_0);
    real dummy_psi = MYNN_MUL(2.0f, mynn_glibc_logf(
        MYNN_MUL(0.5f, MYNN_ADD(1.0f, dummy_0))));
    return mynn_phi_unstable(zet, phh, dummy_psi, 34.0f);
}

// Probe kernels.  They exist so a device gate can show that the __constant__
// tables survived ptxas and that the three libm kernels themselves -- not
// merely phim/phih on top of them -- are bitwise right on the device.
// Without them a mis-folded table entry could hide inside an arm no sweep
// argument reaches.
// Column views address level-major storage without changing arithmetic.
template<class T> struct MynnColumn {
    T* p;
    size_t stride;
    __device__ operator MynnColumn<const T>() const { return {p, stride}; }
    __device__ T& operator[](size_t i) const { return p[i * stride]; }
    __device__ MynnColumn operator+(size_t i) const {
        return {p + i * stride, stride};
    }
};

extern "C" __global__
void mynn_glibc_libm_probe(const real* __restrict__ x,
                           const real* __restrict__ y,
                           real* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[3 * i + 0] = mynn_glibc_logf(x[i]);
    out[3 * i + 1] = mynn_glibc_atanf(x[i]);
    out[3 * i + 2] = mynn_glibc_powf(x[i], y[i]);
}

// The other three glibc words the unit calls (lane/mynn-exact): tanhf
// (fdlibm s_tanhf.c on expm1f), log10f (e_log10f.c on logf) and expf
// (mynn_glibc_expf above).  Graded against the live glibc 2.39 over every
// float32 input by tools/mynn_pbl_wrf461_oracle/libm_sweep.py.
extern "C" __global__
void mynn_glibc_libm_probe2(const real* __restrict__ x,
                            real* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[3 * i + 0] = mynn_tanhf(x[i]);
    out[3 * i + 1] = mynn_glibc_log10f(x[i]);
    out[3 * i + 2] = mynn_glibc_expf(x[i]);
}

extern "C" __global__
void mynn_stfunc_probe(const real* __restrict__ zet, real* __restrict__ out,
                       int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[2 * i + 0] = mynn_phim(zet[i]);
    out[2 * i + 1] = mynn_phih(zet[i]);
}

extern "C" __global__
void mynn_level2_pairs(
    const real* __restrict__ dz_a, const real* __restrict__ u_a,
    const real* __restrict__ v_a, const real* __restrict__ thl_a,
    const real* __restrict__ thetav_a, const real* __restrict__ qw_a,
    const real* __restrict__ ql_a, const real* __restrict__ vt_a,
    const real* __restrict__ vq_a, const real* __restrict__ dz_prev_a,
    const real* __restrict__ u_prev_a, const real* __restrict__ v_prev_a,
    const real* __restrict__ thl_prev_a,
    const real* __restrict__ thetav_prev_a,
    const real* __restrict__ qw_prev_a, const real* __restrict__ ql_prev_a,
    const real* __restrict__ vt_prev_a, const real* __restrict__ vq_prev_a,
    real* __restrict__ dtl_o, real* __restrict__ dqw_o,
    real* __restrict__ dtv_o, real* __restrict__ gm_o,
    real* __restrict__ gh_o, real* __restrict__ sm_o,
    real* __restrict__ sh_o, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // These arguments are part of WRF's routine contract but are not used by
    // the active mym_level2 expression (thetav is retained in a comment there).
    (void)thetav_a; (void)ql_a; (void)thetav_prev_a; (void)ql_prev_a;

    // These constants are compile-time expressions when they are written with
    // bare C operators, and ptxas 12.x's folder does not honour
    // round-to-nearest-even on an FP32 tie.  Three steps of g2 are exact ties:
    // `1.0f - 0.340f` (halfway between 0x3F28F5C2 and 0x3F28F5C3),
    // `(15.0f/24.0f) * that`, and the final sum.  mynn_mym_level2_5 has used
    // the helpers for the same quantities since :873; this site and
    // mynn_mym_level2_column below were the last two on the bare spelling,
    // which handed those three ties back to the folder.
    // Measured on CUDA 12.8/ptxas 12.8:
    // the folder happens to round all three the way the SM does, so converting
    // moved no bit of any oracle -- but tests/test_fp32_tie_folding_gpu.py
    // exhibits ties this same toolchain folds the WRONG way, so which ties are
    // safe is luck, and the spelling must not depend on it.
    const real pr = 0.74f, g1 = 0.235f, b1 = 24.0f, b2 = 15.0f;
    const real c2 = 0.729f, c3 = 0.340f, c5 = 0.2f;
    const real a1 = MYNN_DIV(
        MYNN_MUL(b1, MYNN_SUB(1.0f, MYNN_MUL(3.0f, g1))), 6.0f);
    const real c1 = MYNN_SUB(g1, MYNN_DIV(1.0f,
        MYNN_MUL(MYNN_MUL(3.0f, a1), 2.88449914061481660f)));
    const real a2 = MYNN_DIV(
        MYNN_MUL(a1, MYNN_SUB(g1, c1)), MYNN_MUL(g1, pr));
    const real g2 = MYNN_ADD(
        MYNN_MUL(MYNN_DIV(b2, b1), MYNN_SUB(1.0f, c3)),
        MYNN_MUL(MYNN_DIV(MYNN_MUL(2.0f, a1), b1),
                 MYNN_SUB(3.0f, MYNN_MUL(2.0f, c2))));
    const real tv0 = MYNN_MUL(
        MYNN_SUB(MYNN_DIV(461.6f, 287.0f), 1.0f), 300.0f);
    const real gtr = MYNN_DIV(9.81f, 300.0f);

    // Every operator below is a round-to-nearest PTX instruction, matching
    // mynn_mym_level2_column expression for expression.  Written with bare
    // `*`/`+` this body measured sm 9 ULP / sh 11 ULP against
    // gpuwm/data/mynn/oracle/pbl-level2.csv while the same source built with
    // `-fmad=false` measured 0 on every output, and an instrumented copy put
    // the first divergence on `dtq` -- the one `vtt*dtz + vqq*dqz` NVRTC is
    // free to fuse.  From there gh/ri/a2fac inherit 1 ULP, ri4 reaches 5, and
    // the `rf` cancellation against rfc/rf1/rf2 amplifies it to 9 and 11.
    // `fmaxf`/`fminf` are replaced by mynn_max2/mynn_min2 in the same pass:
    // `fminf(x, rfc)` under `-ftz=true` flushes a subnormal x to zero and
    // returns zero where gfortran returns x.
    real dz = dz_a[idx], dz_prev = dz_prev_a[idx];
    real dz_sum = MYNN_ADD(dz, dz_prev);
    real dzk = MYNN_MUL(0.5f, dz_sum);
    real afk = MYNN_DIV(dz, dz_sum);
    real abk = MYNN_SUB(1.0f, afk);
    real du = MYNN_SUB(u_a[idx], u_prev_a[idx]);
    real dv = MYNN_SUB(v_a[idx], v_prev_a[idx]);
    real duz = MYNN_DIV(MYNN_ADD(MYNN_MUL(du, du), MYNN_MUL(dv, dv)),
                        MYNN_MUL(dzk, dzk));
    real dtz = MYNN_DIV(MYNN_SUB(thl_a[idx], thl_prev_a[idx]), dzk);
    real dqz = MYNN_DIV(MYNN_SUB(qw_a[idx], qw_prev_a[idx]), dzk);
    real vtt = MYNN_ADD(MYNN_ADD(1.0f, MYNN_MUL(vt_a[idx], abk)),
                        MYNN_MUL(vt_prev_a[idx], afk));
    real vqq = MYNN_ADD(MYNN_ADD(tv0, MYNN_MUL(vq_a[idx], abk)),
                        MYNN_MUL(vq_prev_a[idx], afk));
    real dtq = MYNN_ADD(MYNN_MUL(vtt, dtz), MYNN_MUL(vqq, dqz));
    real gh = -MYNN_MUL(dtq, gtr);
    real ri = MYNN_DIV(-gh, mynn_max2(duz, 1.0e-10f));
    real a2fac = MYNN_DIV(1.0f, MYNN_ADD(1.0f, mynn_max2(ri, 0.0f)));
#if defined(MYNN_GSD41)
    // GSD MYNN v4.1 mym_level2 divides by a2den = 1 + max(ri, 0):
    // a2/a2den rounds in FP32 where v4.6.1 multiplies by a2fac.
    real a2r = MYNN_DIV(a2, MYNN_ADD(1.0f, mynn_max2(ri, 0.0f)));
    (void)a2fac;
#define MYNN_L2_F1_A2 MYNN_MUL(3.0f, a2r)
#define MYNN_L2_A2 a2r
#else
#define MYNN_L2_F1_A2 MYNN_MUL(MYNN_MUL(3.0f, a2), a2fac)
#define MYNN_L2_A2 MYNN_MUL(a2, a2fac)
#endif

    real rfc = MYNN_DIV(g1, MYNN_ADD(g1, g2));
    real f1 = MYNN_ADD(
        MYNN_ADD(MYNN_MUL(b1, MYNN_SUB(g1, c1)),
                 MYNN_MUL(MYNN_MUL(MYNN_L2_F1_A2,
                                   MYNN_SUB(1.0f, c2)),
                          MYNN_SUB(1.0f, c5))),
        MYNN_MUL(MYNN_MUL(2.0f, a1), MYNN_SUB(3.0f, MYNN_MUL(2.0f, c2))));
    real f2 = MYNN_SUB(MYNN_MUL(b1, MYNN_ADD(g1, g2)),
                       MYNN_MUL(MYNN_MUL(3.0f, a1), MYNN_SUB(1.0f, c2)));
    real rf1 = MYNN_DIV(MYNN_MUL(b1, MYNN_SUB(g1, c1)), f1);
    real rf2 = MYNN_DIV(MYNN_MUL(b1, g1), f2);
    real smc = MYNN_DIV(
        MYNN_MUL(MYNN_DIV(a1, MYNN_L2_A2), f1), f2);
    real shc = MYNN_MUL(MYNN_MUL(3.0f, MYNN_L2_A2),
                        MYNN_ADD(g1, g2));
#undef MYNN_L2_F1_A2
#undef MYNN_L2_A2
    real ri1 = MYNN_DIV(0.5f, smc);
    real ri2 = MYNN_MUL(rf1, smc);
    real ri3 = MYNN_SUB(MYNN_MUL(MYNN_MUL(4.0f, rf2), smc),
                        MYNN_MUL(2.0f, ri2));
    real ri4 = MYNN_MUL(ri2, ri2);
    real radical = MYNN_ADD(
        MYNN_SUB(MYNN_MUL(ri, ri), MYNN_MUL(ri3, ri)), ri4);
    real rf = mynn_min2(MYNN_MUL(ri1,
        MYNN_SUB(MYNN_ADD(ri, ri2), sqrtf(radical))), rfc);
    real sh = MYNN_DIV(MYNN_MUL(shc, MYNN_SUB(rfc, rf)),
                       MYNN_SUB(1.0f, rf));
    real sm = MYNN_MUL(MYNN_DIV(MYNN_MUL(smc, MYNN_SUB(rf1, rf)),
                                MYNN_SUB(rf2, rf)), sh);

    dtl_o[idx] = dtz; dqw_o[idx] = dqz; dtv_o[idx] = dtq;
    gm_o[idx] = duz; gh_o[idx] = gh; sm_o[idx] = sm; sh_o[idx] = sh;
}

extern "C" __global__
void mynn_pblh_scale_columns(
    const real* __restrict__ thetav_raw, const real* __restrict__ qke_raw,
    const real* __restrict__ zw_raw, const real* __restrict__ dz_raw,
    const real* __restrict__ landsea, const real* __restrict__ dx,
    real* __restrict__ zi_o, int* __restrict__ kzi_o,
    real* __restrict__ psig_bl_o, real* __restrict__ psig_shcu_o,
    int nz, int ncol)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    MynnColumn<const real> thetav = {thetav_raw + column, (size_t)ncol};
    MynnColumn<const real> qke = {qke_raw + column, (size_t)ncol};
    MynnColumn<const real> zw = {zw_raw + column, (size_t)ncol};
    MynnColumn<const real> dz = {dz_raw + column, (size_t)ncol};
    int base = 0;
    int zbase = 0;

    real minthv = 9.0e9f;
    int k = 1;  // zero-based WRF kts+1.
    while (zw[zbase + k] <= 200.0f && k < nz) {
        minthv = fminf(minthv, thetav[base + k]);
        ++k;
    }

    real delta = landsea[column] >= 1.5f ? 1.0f : 1.25f;
    real zi = 0.0f;
    for (k = 1; k < nz - 1; ++k) {
        real current = thetav[base + k];
        if (current >= minthv + delta) {
            real fraction = (current - (minthv + delta))
                / fmaxf(current - thetav[base + k - 1], 1.0e-6f);
            zi = zw[zbase + k] - dz[base + k - 1] * fminf(fraction, 1.0f);
        }
        if (k == nz - 2) zi = zw[zbase + 1];
        if (zi != 0.0f) break;
    }

    real maxqke = fmaxf(qke[base], 0.0f);
    real tkeeps = fmaxf(__fdiv_rn(maxqke, 40.0f), 0.02f);
    real pblh_tke = 0.0f;
    for (k = 1; k < nz - 1; ++k) {
        real qtke = fmaxf(qke[base + k] / 2.0f, 0.0f);
        real qtkem1 = fmaxf(qke[base + k - 1] / 2.0f, 0.0f);
        if (qtke <= tkeeps) {
            real fraction = (tkeeps - qtke)
                / fmaxf(qtkem1 - qtke, 1.0e-6f);
            pblh_tke = zw[zbase + k]
                - dz[base + k - 1] * fminf(fraction, 1.0f);
            pblh_tke = fmaxf(pblh_tke, zw[zbase + 1]);
        }
        if (k == nz - 2) pblh_tke = zw[zbase + 1];
        if (pblh_tke != 0.0f) break;
    }
    pblh_tke = fminf(pblh_tke, zi + 350.0f);
    pblh_tke = fmaxf(pblh_tke, fmaxf(zi - 350.0f, 10.0f));
    real weight = 0.5f * mynn_tanhf(__fdiv_rn((zi - 200.0f), 400.0f)) + 0.5f;
    if (maxqke > 0.05f) zi = pblh_tke * (1.0f - weight) + zi * weight;

#if !defined(MYNN_GSD41)
    int kzi = 2;
    for (k = 1; k < nz - 1; ++k) {
        if (zw[zbase + k] >= zi) { kzi = k; break; }
    }
#else
    // GSD MYNN v4.1 GET_PBLH (NOAA-EMC WRF 3.9 module_bl_mynn.F:4948,
    // :4980, :5006): KPBL blends the level of the theta-v height and the
    // level of the TKE height with the same weight as the heights, each
    // the interface below plus NINT of the fraction of that layer.
    int kzi = 2, kzi2 = 2;
    real zi_th = 0.0f;
    for (k = 1; k < nz - 1; ++k) {
        real current = thetav[base + k];
        if (current >= minthv + delta) {
            real fraction = (current - (minthv + delta))
                / fmaxf(current - thetav[base + k - 1], 1.0e-6f);
            zi_th = zw[zbase + k] - dz[base + k - 1] * fminf(fraction, 1.0f);
            kzi = max(k, 1) + (int)roundf(
                __fdiv_rn(zi_th - zw[zbase + k - 1], dz[base + k - 1]));
        }
        if (k == nz - 2) zi_th = zw[zbase + 1];
        if (zi_th != 0.0f) break;
    }
    real tke_h = 0.0f;
    for (k = 1; k < nz - 1; ++k) {
        real qtke = fmaxf(__fdiv_rn(qke[base + k], 2.0f), 0.0f);
        real qtkem1 = fmaxf(__fdiv_rn(qke[base + k - 1], 2.0f), 0.0f);
        if (qtke <= tkeeps) {
            real fraction = (tkeeps - qtke)
                / fmaxf(qtkem1 - qtke, 1.0e-6f);
            tke_h = zw[zbase + k] - dz[base + k - 1] * fminf(fraction, 1.0f);
            tke_h = fmaxf(tke_h, zw[zbase + 1]);
            kzi2 = max(k, 1) + (int)roundf(
                __fdiv_rn(tke_h - zw[zbase + k - 1], dz[base + k - 1]));
        }
        if (k == nz - 2) tke_h = zw[zbase + 1];
        if (tke_h != 0.0f) break;
    }
    // weight is the theta-v height's blend weight computed above (:5003).
    kzi = max((int)((real)kzi2 * (1.0f - weight) + (real)kzi * weight), 1);
#endif

#if defined(MYNN_GSD41)
    // Fork scale awareness, module_bl_mynn.F:6011-6049.
    real dxdh = __fdiv_rn(fmaxf(dx[column], 10.0f), fminf(zi, 3000.0f));
#else
    real dxdh = fmaxf(2.5f * dx[column], 10.0f) / fminf(zi, 3000.0f);
#endif
    real power = mynn_glibc_powf(dxdh, 0.667f), square = dxdh * dxdh;
    real psig_bl = (square + 0.106f * power)
        / (square + 0.066f * power + 0.071f);
#if defined(MYNN_GSD41)
    dxdh = __fdiv_rn(fmaxf(dx[column], 10.0f), fminf(zi + 500.0f, 3500.0f));
#else
    dxdh = fmaxf(2.5f * dx[column], 10.0f)
        / fminf(zi + 500.0f, 3500.0f);
#endif
    power = mynn_glibc_powf(dxdh, 0.667f); square = dxdh * dxdh;
    real psig_shcu = (square + 0.145f * power)
        / (square + 0.172f * power + 0.170f);

    zi_o[column] = zi; kzi_o[column] = kzi;
    psig_bl_o[column] = fminf(fmaxf(psig_bl, 0.0f), 1.0f);
    psig_shcu_o[column] = fminf(fmaxf(psig_shcu, 0.0f), 1.0f);
}

// Default WRF mym_turbulence adjustment after level-2 stability and mixing
// length have been evaluated. Every vertical interface is independent here.
extern "C" __global__
void mynn_turbulence_default_interfaces(
    const real* __restrict__ dz, const real* __restrict__ u,
    const real* __restrict__ v, const real* __restrict__ cldfra,
    const real* __restrict__ edmf_w, const real* __restrict__ edmf_a,
    const real* __restrict__ tkeprodtd, const real* __restrict__ el,
    const real* __restrict__ qkw, const real* __restrict__ dtl,
    const real* __restrict__ dqw, const real* __restrict__ gm,
    const real* __restrict__ gh, real* __restrict__ sm,
    real* __restrict__ sh, real* __restrict__ dfm,
    real* __restrict__ dfh, real* __restrict__ dfq,
    real* __restrict__ pdk, real* __restrict__ pdt,
    real* __restrict__ pdq, real* __restrict__ pdc,
    real* __restrict__ tcd, real* __restrict__ qcd,
    int nz, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int ld = count / nz;
    int k = idx / ld;
    if (k == 0) return;

    // module_bl_mynn.F:279-298.  gfortran folds this parameter chain with
    // MPFR at FP32, one correctly rounded operation at a time.  Written as
    // host constant expressions it is ptxas that folds it instead, and ptxas
    // 12.x resolves some FP32 ties by truncation: on the box these budgets
    // were measured (12.8.93) the `literal` arm of
    // tests/test_fp32_tie_folding_gpu.py mis-rounds 23 of 148 subtraction
    // ties, while the same subtractions on the SM are all correct.
    //
    // `cc3 = 1.0f - 0.340f` IS one of those ties -- the exact difference of
    // the two FP32 operands falls exactly halfway between 0x3F28F5C2 and
    // 0x3F28F5C3 -- and cc3 multiplies straight into e1c, hence into e1..e4,
    // hence into sm and sh.  This toolchain happens to fold it to even, so
    // all ten constants match the CPU reference word for word today; that is
    // the ptxas release schedule holding the port up, not the port.  Pinning
    // the rounding hands every one of them to the SM instead.
    const real pr = 0.74f, g1 = 0.235f, b1 = 24.0f, b2 = 15.0f;
    const real c2 = 0.729f, c3 = 0.340f, c5 = 0.2f;
    const real a1 = MYNN_DIV(
        MYNN_MUL(b1, MYNN_SUB(1.0f, MYNN_MUL(3.0f, g1))), 6.0f);
    const real c1 = MYNN_SUB(g1, MYNN_DIV(1.0f,
        MYNN_MUL(MYNN_MUL(3.0f, a1), 2.88449914061481660f)));
    const real a2 = MYNN_DIV(
        MYNN_MUL(a1, MYNN_SUB(g1, c1)), MYNN_MUL(g1, pr));
    const real cc2 = MYNN_SUB(1.0f, c2), cc3 = MYNN_SUB(1.0f, c3);
    const real e1c = MYNN_MUL(MYNN_MUL(MYNN_MUL(3.0f, a2), b2), cc3);
    const real e2c = MYNN_MUL(MYNN_MUL(MYNN_MUL(9.0f, a1), a2), cc2);
    const real e3c = MYNN_MUL(
        MYNN_MUL(MYNN_MUL(MYNN_MUL(9.0f, a2), a2), cc2),
        MYNN_SUB(1.0f, c5));
    const real e4c = MYNN_MUL(MYNN_MUL(MYNN_MUL(12.0f, a1), a2), cc2);
    const real e5c = MYNN_MUL(MYNN_MUL(6.0f, a1), a1);

    real dzk = 0.5f * (dz[idx] + dz[idx - ld]);
    double elsq = (double)(el[idx] * el[idx]);
    double q3sq = (double)(qkw[idx] * qkw[idx]);
    real source = sm[idx] * gm[idx] + sh[idx] * gh[idx];
    double q2sq = (double)b1 * elsq * (double)source;

    // module_bl_mynn.F:2734.  sh is floored after q2sq has been formed from
    // the unfloored value, and it survives only down the Helfand-Labraga
    // path, where sh is scaled rather than recomputed.
#if !defined(MYNN_GSD41)
    sh[idx] = fmaxf(sh[idx], 1.0e-5f);
#else
    // GSD MYNN v4.1 mym_turbulence (NOAA-EMC WRF 3.9 module_bl_mynn.F
    // :1599-1635) has no floor here, no clip of sh and no cap of sm.
#endif

    real du = u[idx] - u[idx - ld];
    real dv = v[idx] - v[idx - ld];
    real duz = (du * du + dv * dv) / (dzk * dzk);
    real ri = -gh[idx] / fmaxf(duz, 1.0e-10f);
    real a2fac = 1.0f / (1.0f + fmaxf(ri, 0.0f));
    // `a2fac**2` and `3.0*c1*e5c` are real(kind_phys) in the Fortran, and
    // kind_phys is kind(1.0) -- they round to FP32 before the DOUBLE
    // PRECISION operand widens the product.  Squaring or folding them in FP64
    // instead is a different number and was the whole CPU-side residue.  The
    // rounding is pinned rather than left to the compiler because a `const`
    // built from literals is a host constant expression, and ptxas 12.x
    // mis-folds FP32 ties.
    double a2fac_sq = (double)MYNN_MUL(a2fac, a2fac);
    const real three_c1 = MYNN_MUL(3.0f, c1);
    const real three_c1_e5c = MYNN_MUL(three_c1, e5c);
    double gmel = (double)gm[idx] * elsq;
    double ghel = (double)gh[idx] * elsq;
    if (q3sq / elsq < -(double)gh[idx]) q3sq = -elsq * (double)gh[idx];

    double qdiv;
    double e1, e2, e3, e4, eden;
#if defined(MYNN_GSD41)
    // GSD MYNN v4.1 mym_turbulence (NOAA-EMC WRF 3.9 module_bl_mynn.F
    // :1560-1609): the Canuto/Kitamura factor is a REAL denominator,
    // a2den = 1 + max(ri, 0), divided rather than multiplied by its
    // reciprocal; a2den**2 and a2/a2den round in FP32 before they meet the
    // DOUBLE PRECISION operands.  Written with v4.6.1's a2fac this branch
    // landed 1-3 ULP off the fork in sh and every diffusivity downstream.
    (void)a2fac_sq;
    real a2den = MYNN_ADD(1.0f, mynn_max2(ri, 0.0f));
    double a2den_d = (double)a2den;
    double a2den_sq = (double)MYNN_MUL(a2den, a2den);
    if (q3sq < q2sq) {
        qdiv = sqrt(q3sq / q2sq);
        sm[idx] = (real)((double)sm[idx] * qdiv);
        sh[idx] = (real)((double)sh[idx] * qdiv);
        double qdiv2 = qdiv * qdiv;
        e1 = q3sq - (double)e1c * ghel / a2den_d * qdiv2;
        e2 = q3sq - (double)e2c * ghel / a2den_d * qdiv2;
        e3 = e1 + (double)e3c * ghel / a2den_sq * qdiv2;
        e4 = e1 - (double)e4c * ghel / a2den_d * qdiv2;
        eden = e2 * e4 + e3 * (double)e5c * gmel * qdiv2;
        if (eden < 1.0e-20) eden = 1.0e-20;
    } else {
        qdiv = 1.0;
        e1 = q3sq - (double)e1c * ghel / a2den_d;
        e2 = q3sq - (double)e2c * ghel / a2den_d;
        e3 = e1 + (double)e3c * ghel / a2den_sq;
        e4 = e1 - (double)e4c * ghel / a2den_d;
        eden = e2 * e4 + e3 * (double)e5c * gmel;
        if (eden < 1.0e-20) eden = 1.0e-20;
        sm[idx] = (real)(q3sq * (double)a1
            * (e3 - (double)three_c1 * e4) / eden);
        real a2_over_a2den = MYNN_DIV(a2, a2den);
        sh[idx] = (real)(q3sq * (double)a2_over_a2den
            * (e2 + (double)three_c1_e5c * gmel) / eden);
    }
#else
    if (q3sq < q2sq) {
        qdiv = sqrt(q3sq / q2sq);
        sh[idx] = (real)((double)sh[idx] * qdiv);
        sm[idx] = (real)((double)sm[idx] * qdiv);
        double qdiv2 = qdiv * qdiv;
        e1 = q3sq - (double)e1c * ghel * (double)a2fac * qdiv2;
        e2 = q3sq - (double)e2c * ghel * (double)a2fac * qdiv2;
        e3 = e1 + (double)e3c * ghel * a2fac_sq * qdiv2;
        e4 = e1 - (double)e4c * ghel * (double)a2fac * qdiv2;
        eden = e2 * e4 + e3 * (double)e5c * gmel * qdiv2;
        if (eden < 1.0e-20) eden = 1.0e-20;
    } else {
        qdiv = 1.0;
        e1 = q3sq - (double)e1c * ghel * (double)a2fac;
        e2 = q3sq - (double)e2c * ghel * (double)a2fac;
        e3 = e1 + (double)e3c * ghel * a2fac_sq;
        e4 = e1 - (double)e4c * ghel * (double)a2fac;
        eden = e2 * e4 + e3 * (double)e5c * gmel;
        if (eden < 1.0e-20) eden = 1.0e-20;
        sm[idx] = (real)(q3sq * (double)a1
            * (e3 - (double)three_c1 * e4) / eden);
        real a2fac_product = a2 * a2fac;
        sh[idx] = (real)(q3sq * (double)a2fac_product
            * (e2 + (double)three_c1_e5c * gmel) / eden);
    }
#endif

#if !defined(MYNN_GSD41)
    sh[idx] = fminf(fmaxf(sh[idx], 0.0f), 4.0f);
    sm[idx] = fminf(sm[idx], 5.0f * fmaxf(sh[idx], 0.02f));
#endif
    real cldavg = 0.5f * (cldfra[idx - ld] + cldfra[idx]);
    if (edmf_a[idx] > 0.001f || cldavg > 0.02f) {
        real plume_floor = 0.03f
            * fminf(10.0f * edmf_a[idx] * edmf_w[idx], 1.0f);
#if !defined(MYNN_GSD41)
        real cloud_floor = 0.05f * fminf(cldavg, 1.0f);
#else
        // GSD MYNN v4.1 :1820-1821: the cloud floor is 0.03 cldavg.
        real cloud_floor = 0.03f * fminf(cldavg, 1.0f);
#endif
        sm[idx] = fmaxf(sm[idx], fmaxf(plume_floor, cloud_floor));
        sh[idx] = fmaxf(sh[idx], fmaxf(plume_floor, cloud_floor));
    }

    real elq = el[idx] * qkw[idx];
    real elh = (real)((double)elq * qdiv);
    pdk[idx] = elq * (sm[idx] * gm[idx] + sh[idx] * gh[idx])
        + 0.5f * tkeprodtd[idx];
    pdt[idx] = elh * (sh[idx] * dtl[idx]) * dtl[idx];
#if defined(MYNN_GSD41)
    // Preserve the tiny intermediate before QSQ diagnosis, including when
    // this source is composed by a caller that enables FTZ.
    pdq[idx] = MYNN_MUL(MYNN_MUL(elh, MYNN_MUL(sh[idx], dqw[idx])), dqw[idx]);
#else
    pdq[idx] = elh * (sh[idx] * dqw[idx]) * dqw[idx];
#endif
    pdc[idx] = elh * (sh[idx] * dtl[idx]) * dqw[idx] * 0.5f
        + elh * (sh[idx] * dqw[idx]) * dtl[idx] * 0.5f;
    tcd[idx] = 0.0f;
    qcd[idx] = 0.0f;
    dfm[idx] = elq * sm[idx] / dzk;
    dfh[idx] = elq * sh[idx] / dzk;
    dfq[idx] = dfm[idx];
}

// Default closure-2.6 WRF mym_predict. The two tridiagonal systems are
// sequential within a column; independent columns run in parallel.
extern "C" __global__
void mynn_predict_default_columns(
    const real* __restrict__ dz_raw, const real* __restrict__ rho_raw,
    const real* __restrict__ dfq_raw, const real* __restrict__ pdk_raw,
    const real* __restrict__ pdt_raw, const real* __restrict__ pdq_raw,
    const real* __restrict__ pdc_raw, const real* __restrict__ el_raw,
    const real* __restrict__ s_aw_raw, const real* __restrict__ ust,
    const real* __restrict__ pmz, const real* __restrict__ phh,
    const real* __restrict__ delt, const real* __restrict__ qke_in_raw,
    const real* __restrict__ qsq_in_raw, real* __restrict__ qke_out_raw,
    real* __restrict__ tsq_out_raw, real* __restrict__ qsq_out_raw,
    real* __restrict__ cov_out_raw, real* __restrict__ qkw_raw,
    real* __restrict__ rhoinv_raw, real* __restrict__ kqdz_raw,
    real* __restrict__ kmdz_raw, real* __restrict__ a_raw,
    real* __restrict__ b_raw, real* __restrict__ c_raw,
    real* __restrict__ d_raw, real* __restrict__ cp_raw,
    real* __restrict__ dp_raw, int nz, int ncol)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    MynnColumn<const real> dz = {dz_raw + column, (size_t)ncol};
    MynnColumn<const real> rho = {rho_raw + column, (size_t)ncol};
    MynnColumn<const real> dfq = {dfq_raw + column, (size_t)ncol};
    MynnColumn<const real> pdk = {pdk_raw + column, (size_t)ncol};
    MynnColumn<const real> pdt = {pdt_raw + column, (size_t)ncol};
    MynnColumn<const real> pdq = {pdq_raw + column, (size_t)ncol};
    MynnColumn<const real> pdc = {pdc_raw + column, (size_t)ncol};
    MynnColumn<const real> el = {el_raw + column, (size_t)ncol};
    MynnColumn<const real> s_aw = {s_aw_raw + column, (size_t)ncol};
    MynnColumn<const real> qke_in = {qke_in_raw + column, (size_t)ncol};
    MynnColumn<const real> qsq_in = {qsq_in_raw + column, (size_t)ncol};
    MynnColumn<real> qke_out = {qke_out_raw + column, (size_t)ncol};
    MynnColumn<real> tsq_out = {tsq_out_raw + column, (size_t)ncol};
    MynnColumn<real> qsq_out = {qsq_out_raw + column, (size_t)ncol};
    MynnColumn<real> cov_out = {cov_out_raw + column, (size_t)ncol};
    MynnColumn<real> qkw = {qkw_raw + column, (size_t)ncol};
    MynnColumn<real> rhoinv = {rhoinv_raw + column, (size_t)ncol};
    MynnColumn<real> kqdz = {kqdz_raw + column, (size_t)ncol};
    MynnColumn<real> kmdz = {kmdz_raw + column, (size_t)ncol};
    MynnColumn<real> a = {a_raw + column, (size_t)ncol};
    MynnColumn<real> b = {b_raw + column, (size_t)ncol};
    MynnColumn<real> c = {c_raw + column, (size_t)ncol};
    MynnColumn<real> d = {d_raw + column, (size_t)ncol};
    MynnColumn<real> cp = {cp_raw + column, (size_t)ncol};
    MynnColumn<real> dp = {dp_raw + column, (size_t)ncol};
    int base = 0;
    int zbase = 0;
    real step = delt[column];

#if defined(MYNN_GSD41)
    // Fork mym_predict :2044-2114: diffusion is not density weighted and
    // has no EDMF minimum diffusivity.
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        qkw[idx] = sqrtf(fmaxf(qke_in[idx], 0.0f));
        rhoinv[idx] = 1.0f;
        kqdz[idx] = 3.0f * dfq[idx];
        kmdz[idx] = dfq[idx];
    }
    real kqdz_top = 3.0f * dfq[base + nz - 1];
    real kmdz_top = dfq[base + nz - 1];
#else
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        qkw[idx] = sqrtf(fmaxf(qke_in[idx], 0.0f));
        rhoinv[idx] = 1.0f / fmaxf(rho[idx], 1.0e-4f);
    }
    rhoinv[base] = 1.0f / rho[base];
    real rhoz = rho[base];
    kqdz[base] = rhoz * (3.0f * dfq[base]);
    kmdz[base] = rhoz * dfq[base];
    for (int k = 1; k < nz; ++k) {
        int idx = base + k;
        rhoz = (rho[idx] * dz[idx - 1] + rho[idx - 1] * dz[idx])
            / (dz[idx - 1] + dz[idx]);
        rhoz = fmaxf(rhoz, 1.0e-4f);
        kqdz[idx] = rhoz * (3.0f * dfq[idx]);
        kmdz[idx] = rhoz * dfq[idx];
    }
    // The top interface aliases the last mass-level storage slot. Its value
    // is consumed only as k+1 while solving the penultimate row.
    real kqdz_top = rhoz * (3.0f * dfq[base + nz - 1]);
    real kmdz_top = rhoz * dfq[base + nz - 1];
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k;
        kqdz[idx] = fmaxf(kqdz[idx], 0.5f * s_aw[zbase + k]);
        kqdz[idx] = fmaxf(
            kqdz[idx], -0.5f * (s_aw[zbase + k] - s_aw[zbase + k + 1]));
        kmdz[idx] = fmaxf(kmdz[idx], 0.5f * s_aw[zbase + k]);
        kmdz[idx] = fmaxf(
            kmdz[idx], -0.5f * (s_aw[zbase + k] - s_aw[zbase + k + 1]));
    }

#endif
#if defined(MYNN_GSD41)
    // Fork mym_predict :2030-2094. The upper row is x(top)-x(top-1)=0,
    // the floor is 1e-4, and there is no upper TKE limit. Use the same
    // rounded operator sequence as the unmodified Fortran CPU oracle.
    real vkz = MYNN_MUL(MYNN_MUL(0.4f, 0.5f), dz[base]);
    real ustar3 = MYNN_MUL(MYNN_MUL(ust[column], ust[column]), ust[column]);
    real pdk1 = MYNN_DIV(MYNN_MUL(MYNN_MUL(2.0f, ustar3), pmz[column]), vkz);
    real pdk_bottom = MYNN_SUB(pdk1, pdk[base + 1]);
    for (int k = 0; k < nz - 1; ++k) {
        int idx = base + k;
        real next_kqdz = k == nz - 2 ? kqdz_top : kqdz[idx + 1];
        real b1l = MYNN_MUL(MYNN_MUL(24.0f, 0.5f), MYNN_ADD(el[idx + 1], el[idx]));
        real bp = MYNN_DIV(MYNN_MUL(2.0f, qkw[idx]), b1l);
        real pdk_here = k == 0 ? pdk_bottom : pdk[idx];
        real rp = MYNN_ADD(pdk[idx + 1], pdk_here);
        real dtz = MYNN_DIV(step, dz[idx]);
        a[idx] = -MYNN_MUL(dtz, kqdz[idx]);
        b[idx] = MYNN_ADD(MYNN_ADD(1.0f,
            MYNN_MUL(dtz, MYNN_ADD(kqdz[idx], next_kqdz))), MYNN_MUL(bp, step));
        c[idx] = -MYNN_MUL(dtz, next_kqdz);
        d[idx] = MYNN_ADD(MYNN_MUL(rp, step), qke_in[idx]);
    }
    int top = base + nz - 1;
    a[top] = -1.0f; b[top] = 1.0f; c[top] = 0.0f; d[top] = 0.0f;
    cp[base] = MYNN_DIV(c[base], b[base]);
    dp[base] = MYNN_DIV(d[base], b[base]);
    for (int k = 1; k < nz; ++k) {
        int idx = base + k;
        real m = MYNN_SUB(b[idx], MYNN_MUL(cp[idx - 1], a[idx]));
        cp[idx] = MYNN_DIV(c[idx], m);
        dp[idx] = MYNN_DIV(MYNN_SUB(d[idx], MYNN_MUL(dp[idx - 1], a[idx])), m);
    }
    d[top] = dp[top];
    for (int k = nz - 2; k >= 0; --k) {
        int idx = base + k;
        d[idx] = MYNN_SUB(dp[idx], MYNN_MUL(cp[idx], d[idx + 1]));
    }
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        qke_out[idx] = mynn_max2(d[idx], 1.0e-4f);
    }
#else
    real vkz = 0.4f * 0.5f * dz[base];
    real pdk1 = 2.0f * (ust[column] * ust[column] * ust[column])
        * pmz[column] / vkz;
    real pdk_bottom = pdk1 - pdk[base + 1];
    for (int k = 0; k < nz - 1; ++k) {
        int idx = base + k;
        real next_kqdz = k == nz - 2 ? kqdz_top : kqdz[idx + 1];
        real b1l = 24.0f * 0.5f * (el[idx + 1] + el[idx]);
        real bp = 2.0f * qkw[idx] / b1l;
        real pdk_here = k == 0 ? pdk_bottom : pdk[idx];
        real rp = pdk[idx + 1] + pdk_here;
        real dtz = step / dz[idx];
        a[idx] = -dtz * kqdz[idx] * rhoinv[idx];
        b[idx] = 1.0f + dtz * (kqdz[idx] + next_kqdz) * rhoinv[idx]
            + bp * step;
        c[idx] = -dtz * next_kqdz * rhoinv[idx];
        d[idx] = rp * step + qke_in[idx];
    }
    int top = base + nz - 1;
    a[top] = 0.0f; b[top] = 1.0f; c[top] = 0.0f; d[top] = qke_in[top];

    cp[base] = c[base] / b[base];
    dp[base] = d[base] / b[base];
    for (int k = 1; k < nz; ++k) {
        int idx = base + k;
        real m = b[idx] - cp[idx - 1] * a[idx];
        cp[idx] = c[idx] / m;
        dp[idx] = (d[idx] - dp[idx - 1] * a[idx]) / m;
    }
    d[top] = dp[top];
    for (int k = nz - 2; k >= 0; --k) {
        int idx = base + k;
        d[idx] = dp[idx] - cp[idx] * d[idx + 1];
    }
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        qke_out[idx] = fminf(fmaxf(d[idx], 1.0e-3f), 150.0f);
    }

#endif
#if defined(MYNN_GSD41)
    // Fork closure level 2.5 (levflag 2, module_bl_mynn.F:2232-2253, audit
    // P23): no variance is predicted; tsq, qsq and cov are diagnosed from
    // their production terms with the start-of-call qkw, pdt(kts) etc.
    // taken from the next level (:2042-2044), the top row repeating the one
    // below.  Rounded in the Fortran's operation order.
    (void)kmdz; (void)qsq_in; (void)kmdz_top;
    for (int k = 0; k < nz - 1; ++k) {
        int idx = base + k;
        real b2l = qkw[idx] <= 0.0f ? 0.0f
            : MYNN_DIV(MYNN_MUL(MYNN_MUL(15.0f, 0.25f),
                                MYNN_ADD(el[idx + 1], el[idx])), qkw[idx]);
        real pdt_here = k == 0 ? pdt[base + 1] : pdt[idx];
        real pdq_here = k == 0 ? pdq[base + 1] : pdq[idx];
        real pdc_here = k == 0 ? pdc[base + 1] : pdc[idx];
        tsq_out[idx] = MYNN_MUL(b2l, MYNN_ADD(pdt[idx + 1], pdt_here));
        qsq_out[idx] = MYNN_MUL(b2l, MYNN_ADD(pdq[idx + 1], pdq_here));
        cov_out[idx] = MYNN_MUL(b2l, MYNN_ADD(pdc[idx + 1], pdc_here));
    }
    tsq_out[top] = tsq_out[top - 1];
    qsq_out[top] = qsq_out[top - 1];
    cov_out[top] = cov_out[top - 1];
#else
    for (int k = 0; k < nz - 1; ++k) {
        int idx = base + k;
        real next_kmdz = k == nz - 2 ? kmdz_top : kmdz[idx + 1];
        real b2l = 15.0f * 0.5f * (el[idx + 1] + el[idx]);
        real bp = 2.0f * qkw[idx] / b2l;
        real pdq_here = k == 0 ? pdq[base + 1] : pdq[idx];
        real rp = pdq[idx + 1] + pdq_here;
        real dtz = step / dz[idx];
        a[idx] = -dtz * kmdz[idx] * rhoinv[idx];
        b[idx] = 1.0f + dtz * (kmdz[idx] + next_kmdz) * rhoinv[idx]
            + bp * step;
        c[idx] = -dtz * next_kmdz * rhoinv[idx];
        d[idx] = rp * step + qsq_in[idx];
    }
    a[top] = -1.0f; b[top] = 1.0f; c[top] = 0.0f; d[top] = 0.0f;
    cp[base] = c[base] / b[base];
    dp[base] = d[base] / b[base];
    for (int k = 1; k < nz; ++k) {
        int idx = base + k;
        real m = b[idx] - cp[idx - 1] * a[idx];
        cp[idx] = c[idx] / m;
        dp[idx] = (d[idx] - dp[idx - 1] * a[idx]) / m;
    }
    d[top] = dp[top];
    for (int k = nz - 2; k >= 0; --k) {
        int idx = base + k;
        d[idx] = dp[idx] - cp[idx] * d[idx + 1];
    }
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        qsq_out[idx] = fmaxf(d[idx], 1.0e-17f);
    }

    for (int k = 0; k < nz - 1; ++k) {
        int idx = base + k;
        real b2l = qkw[idx] <= 0.0f ? 0.0f
            : 15.0f * 0.25f * (el[idx + 1] + el[idx]) / qkw[idx];
        real pdt_here = k == 0 ? pdt[base + 1] : pdt[idx];
        real pdc_here = k == 0 ? pdc[base + 1] : pdc[idx];
        tsq_out[idx] = b2l * (pdt[idx + 1] + pdt_here);
        cov_out[idx] = b2l * (pdc[idx + 1] + pdc_here);
    }
    tsq_out[top] = tsq_out[top - 1];
    cov_out[top] = cov_out[top - 1];
#endif
    (void)phh;  // WRF computes phm, but default closure overwrites its terms.
}

// module_bl_mynn.F saturation-vapour-pressure polynomials (Pa), evaluated in
// the source's Horner order.
__device__ __forceinline__ real mynn_esat_liquid(real xc)
{
    return 0.611583699e3f + xc * (0.444606896e2f + xc * (0.143177157e1f
        + xc * (0.264224321e-1f + xc * (0.299291081e-3f
        + xc * (0.203154182e-5f + xc * (0.702620698e-8f
        + xc * (0.379534310e-11f + xc * (-0.321582393e-13f))))))));
}

__device__ __forceinline__ real mynn_esat_ice(real xc)
{
    return 0.609868993e3f + xc * (0.499320233e2f + xc * (0.184672631e1f
        + xc * (0.402737184e-1f + xc * (0.565392987e-3f
        + xc * (0.521693933e-5f + xc * (0.307839583e-7f
        + xc * (0.105785160e-9f + xc * (0.161444444e-12f))))))));
}

// module_bl_mynn.F:qsat_blend, phase-blended saturation mixing ratio (kg/kg).
__device__ __forceinline__ real mynn_qsat_blend(real t, real p)
{
    const real t0c = 273.15f, tice = 240.0f, t0cm6 = 273.15f - 6.0f;
    real xc = fmaxf(-80.0f, t - t0c);
    real ceiling = p * 0.15f;
    if (t >= t0cm6) {
        real esl = fminf(mynn_esat_liquid(xc), ceiling);
        return 0.622f * esl / fmaxf(p - esl, 1.0e-5f);
    }
    if (t <= tice) {
        real esi = fminf(mynn_esat_ice(xc), ceiling);
        return 0.622f * esi / fmaxf(p - esi, 1.0e-5f);
    }
    real esl = fminf(mynn_esat_liquid(xc), ceiling);
    real esi = fminf(mynn_esat_ice(xc), ceiling);
    real rslf = 0.622f * esl / fmaxf(p - esl, 1.0e-5f);
    real rsif = 0.622f * esi / fmaxf(p - esi, 1.0e-5f);
    real chi = __fdiv_rn((t0cm6 - t), (t0cm6 - tice));
    return (1.0f - chi) * rslf + chi * rsif;
}

// module_bl_mynn.F:xl_blend, phase-blended latent heat (J/kg).
//
// cpv, cpv_cliq and cpv_cice are the module_bl_mynn_common.F parameter chain,
// and gfortran folds it with MPFR at FP32.  Written as `4.0f * 461.6f` these
// are host constant expressions instead, which ptxas 12.x folds itself -- and
// it rounds FP32 ties the wrong way when it does, as the compile-time tie
// probe in tests/test_fp32_tie_folding_gpu.py records.  All four installed
// ptxas versions happen to land on the right words for these three, so the
// bare spelling has never shown a divergence here; that is the ptxas release
// schedule holding the port up, not the port.  The rounding-pinned form is
// the same three values by construction, on any toolchain.  The identical
// constants are already built this way in mynn_xl_blend_rn.
__device__ __forceinline__ real mynn_xl_blend(real t)
{
    const real t0c = 273.15f, tice = 240.0f;
    const real cpv = MYNN_MUL(4.0f, 461.6f);
    const real cpv_cliq = MYNN_SUB(cpv, 4190.0f);
    const real cpv_cice = MYNN_SUB(cpv, 2106.0f);
    if (t >= t0c) return 2.5e6f + cpv_cliq * (t - t0c);
    if (t <= tice) return 2.85e6f + cpv_cice * (t - t0c);
    real xlvt = 2.5e6f + cpv_cliq * (t - t0c);
    real xlst = 2.85e6f + cpv_cice * (t - t0c);
    real chi = __fdiv_rn((t0c - t), (t0c - tice));
    return (1.0f - chi) * xlvt + chi * xlst;
}

// Default WRF mym_condensation with bl_mynn_cloudpdf=2 (Chaboureau-Bechtold)
// and stochastic perturbations off. One thread owns one column because the
// tropopause search and the zagl accumulation are sequential in the vertical.
// WRF also passes zw, qv, thl, sh, el, dx, hfx, and rmo; the CASE(2) branch
// never reads them, so they are not plumbed into this kernel.
extern "C" __global__
void mynn_condensation_default_columns(
    const real* __restrict__ dz_raw, const real* __restrict__ th_raw,
    const real* __restrict__ qw_raw, const real* __restrict__ qc_raw,
    const real* __restrict__ qi_raw, const real* __restrict__ qs_raw,
    const real* __restrict__ p_raw, const real* __restrict__ exner_raw,
    const real* __restrict__ qsq_raw, const real* __restrict__ rstoch_raw,
    const real* __restrict__ xland, const real* __restrict__ pblh,
    const real* __restrict__ sgm_in_raw, real* __restrict__ qc_bl_raw,
    real* __restrict__ qi_bl_raw, real* __restrict__ cldfra_raw,
    real* __restrict__ vt_raw, real* __restrict__ vq_raw,
    real* __restrict__ sgm_raw, int nz, int ncol)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    MynnColumn<const real> dz = {dz_raw + column, (size_t)ncol};
    MynnColumn<const real> th = {th_raw + column, (size_t)ncol};
    MynnColumn<const real> qw = {qw_raw + column, (size_t)ncol};
    MynnColumn<const real> qc = {qc_raw + column, (size_t)ncol};
    MynnColumn<const real> qi = {qi_raw + column, (size_t)ncol};
    MynnColumn<const real> qs = {qs_raw + column, (size_t)ncol};
    MynnColumn<const real> p = {p_raw + column, (size_t)ncol};
    MynnColumn<const real> exner = {exner_raw + column, (size_t)ncol};
    MynnColumn<const real> qsq = {qsq_raw + column, (size_t)ncol};
    MynnColumn<const real> rstoch = {rstoch_raw + column, (size_t)ncol};
    MynnColumn<const real> sgm_in = {sgm_in_raw + column, (size_t)ncol};
    MynnColumn<real> qc_bl = {qc_bl_raw + column, (size_t)ncol};
    MynnColumn<real> qi_bl = {qi_bl_raw + column, (size_t)ncol};
    MynnColumn<real> cldfra = {cldfra_raw + column, (size_t)ncol};
    MynnColumn<real> vt = {vt_raw + column, (size_t)ncol};
    MynnColumn<real> vq = {vq_raw + column, (size_t)ncol};
    MynnColumn<real> sgm = {sgm_raw + column, (size_t)ncol};
    int base = 0;

    const real rd = 287.0f, rv = 461.6f;
    const real cpd = 7.0f * rd / 2.0f, cpv = 4.0f * rv;
    const real tice = 240.0f, tliq = 269.0f;
    const real tv0 = (461.6f / 287.0f - 1.0f) * 300.0f;
    const real rhcrit = 0.83f, rhmax = 1.02f;
    const real qpct_sfc = 0.025f, qpct_pbl = 0.030f, qpct_trp = 0.040f;
    const real exp_m1 = 0.36787944117144233f;

    // Thompson-style tropopause search. The Fortran DO runs downward from
    // kte-3 with a GOTO exit; falling through leaves the loop variable one
    // below kts, which collapses k_tropo onto kts+2.
    int found = 0;  // one-based WRF level, zero when the loop falls through.
    for (int level = nz - 3; level >= 1; --level) {
        real theta1 = th[base + level - 1];
        real theta2 = th[base + level + 1];
        real ht1 = 44307.692f
            * (1.0f - mynn_glibc_powf(__fdiv_rn(p[base + level - 1], 101325.0f), 0.190f));
        real ht2 = 44307.692f
            * (1.0f - mynn_glibc_powf(__fdiv_rn(p[base + level + 1], 101325.0f), 0.190f));
        real slope = (theta2 - theta1) / (ht2 - ht1);
        if (slope < 10.0f / 1500.0f && ht1 < 19000.0f && ht1 > 4000.0f) {
            found = level;
            break;
        }
    }
    int k_tropo = max(3, found + 2);

    real pblh2 = fmaxf(10.0f, pblh[column]);
    real zagl = 0.0f, dzm1 = 0.0f;
    for (int k = 0; k < nz - 1; ++k) {
        int idx = base + k;
        zagl += 0.5f * (dz[idx] + dzm1);
        dzm1 = dz[idx];

        real t = th[idx] * exner[idx];
        real xl = mynn_xl_blend(t);
        real qsat_tk = mynn_qsat_blend(t, p[idx]);
        real rh = fmaxf(fminf(rhmax, qw[idx] / fmaxf(1.0e-10f, qsat_tk)),
                        0.001f);

        // WRF also stores alp(k) and bet(k) here; CASE(2) never reads them.
        real rsl = xl * qsat_tk / (rv * (t * t));
        real cpm = cpd + qw[idx] * cpv;
        real a = 1.0f / (1.0f + xl * rsl / cpm);
        real b = a * rsl;

        // spp_pbl is zero in this lane, so the perturbation term vanishes.
        real qw_pert = qw[idx] + qw[idx] * 0.5f * rstoch[idx] * 0.0f;
        real qmq = qw_pert - qsat_tk;

        // r3sq is DOUBLE PRECISION in the Fortran, so this is a double
        // square root of an FP32 value rounded back to FP32 on assignment.
        double r3sq = (double)fmaxf(qsq[idx], 0.0f);
        real sgm_k = (real)sqrt(r3sq);
        sgm_k = fminf(sgm_k, qsat_tk * 0.666f);
        real wt = __fdiv_rn(fmaxf(500.0f - fmaxf(dz[idx] - 100.0f, 0.0f), 0.0f), 500.0f);
        sgm_k = sgm_k + sgm_k * 0.2f * (1.0f - wt);
        real qpct = qpct_pbl * wt + qpct_trp * (1.0f - wt);
        qpct = fminf(qpct, fmaxf(qpct_sfc, __fdiv_rn(qpct_pbl * zagl, 500.0f)));
        sgm_k = fmaxf(sgm_k, qsat_tk * qpct);
        sgm[idx] = sgm_k;

        real q1 = qmq / sgm_k;
        real wt2 = fminf(__fdiv_rn(fmaxf(zagl - pblh2, 0.0f), 300.0f), 1.0f);
        real frozen = qi[idx] + qs[idx];
        if (frozen > 1.0e-9f && zagl > pblh2) {
            real rh_hack = fminf(
                rhmax, rhcrit + wt2 * 0.045f * (9.0f + mynn_glibc_log10f(frozen)));
            rh = fmaxf(rh, rh_hack);
            real q1_rh = -3.0f + __fdiv_rn(3.0f * (rh - rhcrit), (1.0f - rhcrit));
            q1 = fmaxf(q1_rh, q1);
        }
        if (qc[idx] > 1.0e-6f && zagl > pblh2) {
            real rh_hack = fminf(
                rhmax, rhcrit + wt2 * 0.08f * (6.0f + mynn_glibc_log10f(qc[idx])));
            rh = fmaxf(rh, rh_hack);
            real q1_rh = -3.0f + __fdiv_rn(3.0f * (rh - rhcrit), (1.0f - rhcrit));
            q1 = fmaxf(q1_rh, q1);
        }

        real q1k = q1;
        real cldfra_k = fmaxf(
            0.0f, fminf(1.0f, 0.5f + 0.36f * mynn_glibc_atanf(1.8f * (q1 + 0.2f))));

        real maxqc = fmaxf(qw[idx] - qsat_tk, 0.0f);
        real ql_water, ql_ice;
        if (q1k < 0.0f) {
            ql_water = sgm_k * mynn_glibc_expf(1.2f * q1k - 1.0f);
            ql_ice = ql_water;
        } else if (q1k > 2.0f) {
            ql_water = fminf(sgm_k * q1k, maxqc);
            ql_ice = sgm_k * q1k;
        } else {
            real shape = sgm_k
                * (exp_m1 + 0.66f * q1k + 0.086f * (q1k * q1k));
            ql_water = fminf(shape, maxqc);
            ql_ice = shape;
        }
        if (cldfra_k < 0.001f) {
            ql_ice = 0.0f;
            ql_water = 0.0f;
            cldfra_k = 0.0f;
        }

        real liq_frac = fminf(1.0f, fmaxf(0.0f, __fdiv_rn((t - tice), (tliq - tice))));
        qc_bl[idx] = liq_frac * ql_water;
        qi_bl[idx] = (1.0f - liq_frac) * ql_ice;
        if (k + 1 >= k_tropo) {
            cldfra_k = 0.0f;
            qc_bl[idx] = 0.0f;
            qi_bl[idx] = 0.0f;
        }

        q1k = xland[column] - 1.5f >= 0.0f ? fmaxf(q1, -2.5f)
                                           : fmaxf(q1, -2.0f);
        real fng;
        if (q1k >= 1.0f) fng = 1.0f;
        else if (q1k >= -1.7f) fng = mynn_glibc_expf(-0.4f * (q1k - 1.0f));
        else if (q1k >= -2.5f) fng = 3.0f + mynn_glibc_expf(-3.8f * (q1k + 1.7f));
        else fng = fminf(23.9f + mynn_glibc_expf(-1.6f * (q1k + 2.5f)), 60.0f);

        real cfmax = fminf(cldfra_k, 0.6f);
        real zsl = fminf(fmaxf(25.0f, 0.1f * pblh2), 100.0f);
        wt = fminf(zagl / zsl, 1.0f);
        cfmax = cfmax * wt;

        real bb = b * t / th[idx];
        real qww = 1.0f + 0.61f * qw[idx];
        real alpha = 0.61f * th[idx];
        real beta = (th[idx] / t) * (__fdiv_rn(xl, cpd)) - 1.61f * th[idx];
        vt[idx] = qww - cfmax * beta * bb * fng - 1.0f;
        vq[idx] = alpha + cfmax * beta * a * fng - tv0;

        real fac_damp = fminf(zagl * 0.0025f, 1.0f);
        real excess = __fdiv_rn(fmaxf(0.0f, rh - 0.92f), 0.145f);
        real cld_factor = 1.0f + fac_damp * fminf(excess * excess, 0.37f);
        cldfra[idx] = fminf(1.0f, cld_factor * cldfra_k);
    }

    int top = base + nz - 1;
    vt[top] = vt[top - 1];
    vq[top] = vq[top - 1];
    sgm[top] = sgm_in[top];
    qc_bl[top] = 0.0f;
    qi_bl[top] = 0.0f;
    cldfra[top] = 0.0f;
}

// ===========================================================================
// module_bl_mynn.F:4027 mynn_tendencies under the mass-flux-free identity,
// together with the module_bl_mynn.F:5137 moisture_check repair it calls at
// module_bl_mynn.F:5020 and the module_bl_mynn.F:5422 tridiag2 solver.
//
// gfortran emits one rounded SSE instruction per Fortran operator; NVRTC is
// free to contract a*b+c into a single FMA, and that one fused rounding is
// enough to break bitwise parity with the pinned oracle.  Every arithmetic
// step below is therefore an explicit round-to-nearest intrinsic.  Do not
// rewrite these as plain operators.
// ===========================================================================
// The round-to-nearest primitives this section is written in are defined at
// the top of this file, because they are the translation unit's arithmetic
// vocabulary and not a property of mynn_tendencies.
// ===========================================================================

// mynn_gt / mynn_max2 / mynn_min2 are defined at the top of this file with the
// arithmetic helpers, for the same reason: they are the translation unit's
// vocabulary, not a property of mynn_tendencies.  mym_level2 needs them too.

// Negative control for the block above, compiled through the same loader and
// therefore the same `-ftz=true`.  Six results on subnormal operands: if the
// helpers ever revert to plain operators or to the `__f*_rn` intrinsics,
// every one of them collapses to zero and tests/test_mynn_pbl_gpu.py says so.
extern "C" __global__
void mynn_denormal_probe(const real* __restrict__ a,
                         const real* __restrict__ b,
                         real* __restrict__ out)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    out[0] = MYNN_ADD(a[0], b[0]);
    out[1] = MYNN_SUB(a[1], b[1]);
    out[2] = MYNN_MUL(a[2], b[2]);
    out[3] = MYNN_DIV(a[3], b[3]);
    out[4] = mynn_max2(0.0f, a[4]);
    out[5] = mynn_min2(a[5], 0.0f);
}

// module_bl_mynn.F:5422 tridiag2.  cpw/dpw are the Fortran cp/dp work
// vectors.  x must not alias cpw or dpw; it may alias d, because the forward
// sweep finishes before the back substitution writes anything.
__device__ void mynn_tridiag2_column(
    MynnColumn<const real> a, MynnColumn<const real> b,
    MynnColumn<const real> c, MynnColumn<const real> d,
    MynnColumn<real> cpw, MynnColumn<real> dpw,
    MynnColumn<real> x, int n)
{
    cpw[0] = MYNN_DIV(c[0], b[0]);
    dpw[0] = MYNN_DIV(d[0], b[0]);
    for (int k = 1; k < n; ++k) {
        real m = MYNN_SUB(b[k], MYNN_MUL(cpw[k - 1], a[k]));
        cpw[k] = MYNN_DIV(c[k], m);
        dpw[k] = MYNN_DIV(MYNN_SUB(d[k], MYNN_MUL(dpw[k - 1], a[k])), m);
    }
    x[n - 1] = dpw[n - 1];
    for (int k = n - 2; k >= 0; --k)
        x[k] = MYNN_SUB(dpw[k], MYNN_MUL(cpw[k], x[k + 1]));
}

// module_bl_mynn.F:5137 moisture_check.  A borrow-from-below repair, not a
// clip: a condensate deficit is paid out of the vapour in the same layer (and
// warms it), a vapour deficit is paid out of the layer below weighted by the
// pressure-thickness ratio, and any residue left in the bottom layer is
// spread over every layer still holding more than 2*qvmin.  The caller hands
// thl to the ``th`` argument, which is what WRF does at line 5020.
__device__ void mynn_moisture_check_column(
    real delt, MynnColumn<const real> dp, MynnColumn<const real> exner,
    MynnColumn<real> qv, MynnColumn<real> qc, MynnColumn<real> qi,
    MynnColumn<real> qs, MynnColumn<real> th, MynnColumn<real> dqv,
    MynnColumn<real> dqc, MynnColumn<real> dqi, MynnColumn<real> dqs,
    MynnColumn<real> dth, int nz)
{
    // module_bl_mynn.F:5162-5164 floors.
    const real qvmin = 1.0e-20f, qcmin = 0.0f, qimin = 0.0f;
    // module_bl_mynn_common.F:85-86; xlscp is derived from (xlv+xlf).
    const real xlvcp = XLV / CP;
    const real xlscp = (XLV + 3.50e5f) / CP;

    real dqv2 = 0.0f;
    for (int k = nz - 1; k >= 0; --k) {
        real dqc2 = mynn_max2(0.0f, MYNN_SUB(qcmin, qc[k]));
        real dqi2 = mynn_max2(0.0f, MYNN_SUB(qimin, qi[k]));
        real dqs2 = mynn_max2(0.0f, MYNN_SUB(qimin, qs[k]));
        real xlvcp_ex = MYNN_DIV(xlvcp, exner[k]);
        real xlscp_ex = MYNN_DIV(xlscp, exner[k]);
        dqc[k] = MYNN_ADD(dqc[k], MYNN_DIV(dqc2, delt));
        dqi[k] = MYNN_ADD(dqi[k], MYNN_DIV(dqi2, delt));
        dqs[k] = MYNN_ADD(dqs[k], MYNN_DIV(dqs2, delt));
        dqv[k] = MYNN_SUB(
            dqv[k], MYNN_DIV(MYNN_ADD(MYNN_ADD(dqc2, dqi2), dqs2), delt));
        dth[k] = MYNN_ADD(
            MYNN_ADD(dth[k], MYNN_MUL(xlvcp_ex, MYNN_DIV(dqc2, delt))),
            MYNN_MUL(xlscp_ex, MYNN_DIV(MYNN_ADD(dqi2, dqs2), delt)));
        qc[k] = MYNN_ADD(qc[k], dqc2);
        qi[k] = MYNN_ADD(qi[k], dqi2);
        qs[k] = MYNN_ADD(qs[k], dqs2);
        qv[k] = MYNN_SUB(MYNN_SUB(MYNN_SUB(qv[k], dqc2), dqi2), dqs2);
        th[k] = MYNN_ADD(
            MYNN_ADD(th[k], MYNN_MUL(xlvcp_ex, dqc2)),
            MYNN_MUL(xlscp_ex, MYNN_ADD(dqi2, dqs2)));
        dqv2 = mynn_max2(0.0f, MYNN_SUB(qvmin, qv[k]));
        dqv[k] = MYNN_ADD(dqv[k], MYNN_DIV(dqv2, delt));
        qv[k] = MYNN_ADD(qv[k], dqv2);
        if (k != 0) {
            real borrow = MYNN_DIV(MYNN_MUL(dqv2, dp[k]), dp[k - 1]);
            qv[k - 1] = MYNN_SUB(qv[k - 1], borrow);
            dqv[k - 1] = MYNN_SUB(dqv[k - 1], MYNN_DIV(borrow, delt));
        }
        qv[k] = mynn_max2(qv[k], qvmin);
        qc[k] = mynn_max2(qc[k], qcmin);
        qi[k] = mynn_max2(qi[k], qimin);
        qs[k] = mynn_max2(qs[k], qimin);
    }
    if (dqv2 > 1.0e-20f) {
        real total = 0.0f;
        for (int k = 0; k < nz; ++k)
            if (qv[k] > MYNN_MUL(2.0f, qvmin))
                total = MYNN_ADD(total, MYNN_MUL(qv[k], dp[k]));
        real aa = MYNN_DIV(MYNN_MUL(dqv2, dp[0]), mynn_max2(1.0e-20f, total));
        if (aa < 0.5f) {
            for (int k = 0; k < nz; ++k) {
                if (qv[k] > MYNN_MUL(2.0f, qvmin)) {
                    real dum = MYNN_MUL(aa, qv[k]);
                    qv[k] = MYNN_SUB(qv[k], dum);
                    dqv[k] = MYNN_SUB(dqv[k], MYNN_DIV(dum, delt));
                }
            }
        }
    }
}

// module_bl_mynn.F:4027 mynn_tendencies with bl_mynn_cloudmix=1,
// bl_mynn_mixqt=0, bl_mynn_mixscalars=0, FLAG_QC/FLAG_QI true and every other
// species flag false.  One thread owns one column: the matrices, the
// tridiagonal sweeps, and the moisture-check borrow chain are all sequential
// in the vertical.
//
// onoff is the Fortran factor built at module_bl_mynn.F:4130-4134 from
// bl_mynn_edmf_mom.  It multiplies the mass flux in the u and v systems only;
// the thl, sqc and sqv systems take s_aw*/sd_aw* unconditionally, so
// onoff=0.0f does not turn the mass flux off, it turns off momentum transport
// by the mass flux.  The mass-flux-free lane is this same kernel launched with
// onoff=0.0f and an all-zero forcing, which the host gate enforces there.
extern "C" __global__
void mynn_tendencies_columns(
    const real* __restrict__ dz_raw, const real* __restrict__ rho_raw,
    const real* __restrict__ u_raw, const real* __restrict__ v_raw,
    const real* __restrict__ th_raw, const real* __restrict__ tk_raw,
    const real* __restrict__ qv_raw, const real* __restrict__ p_raw,
    const real* __restrict__ exner_raw, const real* __restrict__ thl_in_raw,
    const real* __restrict__ sqv_raw, const real* __restrict__ sqc_raw,
    const real* __restrict__ sqi_raw, const real* __restrict__ sqs_raw,
    const real* __restrict__ ozone_raw, const real* __restrict__ tcd_raw,
    const real* __restrict__ qcd_raw, const real* __restrict__ dfm_raw,
    const real* __restrict__ dfh_raw, const real* __restrict__ diss_heat_raw,
    const real* __restrict__ sub_thl_raw, const real* __restrict__ sub_sqv_raw,
    const real* __restrict__ sub_u_raw, const real* __restrict__ sub_v_raw,
    const real* __restrict__ det_thl_raw, const real* __restrict__ det_sqv_raw,
    const real* __restrict__ det_sqc_raw, const real* __restrict__ det_u_raw,
    const real* __restrict__ det_v_raw,
    const real* __restrict__ s_aw_raw, const real* __restrict__ s_awthl_raw,
    const real* __restrict__ s_awqv_raw, const real* __restrict__ s_awqc_raw,
    const real* __restrict__ s_awu_raw, const real* __restrict__ s_awv_raw,
    const real* __restrict__ sd_aw_raw, const real* __restrict__ sd_awthl_raw,
    const real* __restrict__ sd_awqv_raw, const real* __restrict__ sd_awqc_raw,
    const real* __restrict__ sd_awu_raw, const real* __restrict__ sd_awv_raw,
    const real* __restrict__ delt_col, const real* __restrict__ psfc_col,
    const real* __restrict__ ust_col, const real* __restrict__ wspd_col,
    const real* __restrict__ uoce_col, const real* __restrict__ voce_col,
    const real* __restrict__ flt_col, const real* __restrict__ flqv_col,
    const real* __restrict__ flqc_col,
    real* __restrict__ du_raw, real* __restrict__ dv_raw, real* __restrict__ dth_raw,
    real* __restrict__ dqv_raw, real* __restrict__ dqc_raw, real* __restrict__ dqi_raw,
    real* __restrict__ dqs_raw, real* __restrict__ dozone_raw,
    real* __restrict__ thl_raw,
    real* __restrict__ dtz_raw, real* __restrict__ rhoinv_raw,
    real* __restrict__ delp_raw, real* __restrict__ khdz_raw,
    real* __restrict__ kmdz_raw, real* __restrict__ a_raw, real* __restrict__ b_raw,
    real* __restrict__ c_raw, real* __restrict__ d_raw, real* __restrict__ cpw_raw,
    real* __restrict__ dpw_raw, real* __restrict__ sqv2_raw,
    real* __restrict__ sqc2_raw, real* __restrict__ sqi2_raw,
    real* __restrict__ sqs2_raw, real onoff, int nz, int ncol
#if defined(MYNN_GSD41)
    , const real* __restrict__ qc_original_raw,
    const real* __restrict__ qi_original_raw, int old_cloud_tendencies
#endif
    )
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    MynnColumn<const real> dz = {dz_raw + column, (size_t)ncol};
    MynnColumn<const real> rho = {rho_raw + column, (size_t)ncol};
    MynnColumn<const real> u = {u_raw + column, (size_t)ncol};
    MynnColumn<const real> v = {v_raw + column, (size_t)ncol};
    MynnColumn<const real> th = {th_raw + column, (size_t)ncol};
    MynnColumn<const real> tk = {tk_raw + column, (size_t)ncol};
    MynnColumn<const real> qv = {qv_raw + column, (size_t)ncol};
    MynnColumn<const real> p = {p_raw + column, (size_t)ncol};
    MynnColumn<const real> exner = {exner_raw + column, (size_t)ncol};
    MynnColumn<const real> thl_in = {thl_in_raw + column, (size_t)ncol};
    MynnColumn<const real> sqv = {sqv_raw + column, (size_t)ncol};
    MynnColumn<const real> sqc = {sqc_raw + column, (size_t)ncol};
    MynnColumn<const real> sqi = {sqi_raw + column, (size_t)ncol};
    MynnColumn<const real> sqs = {sqs_raw + column, (size_t)ncol};
    MynnColumn<const real> ozone = {ozone_raw + column, (size_t)ncol};
    MynnColumn<const real> tcd = {tcd_raw + column, (size_t)ncol};
    MynnColumn<const real> qcd = {qcd_raw + column, (size_t)ncol};
    MynnColumn<const real> dfm = {dfm_raw + column, (size_t)ncol};
    MynnColumn<const real> dfh = {dfh_raw + column, (size_t)ncol};
    MynnColumn<const real> diss_heat = {diss_heat_raw + column, (size_t)ncol};
    MynnColumn<const real> sub_thl = {sub_thl_raw + column, (size_t)ncol};
    MynnColumn<const real> sub_sqv = {sub_sqv_raw + column, (size_t)ncol};
    MynnColumn<const real> sub_u = {sub_u_raw + column, (size_t)ncol};
    MynnColumn<const real> sub_v = {sub_v_raw + column, (size_t)ncol};
    MynnColumn<const real> det_thl = {det_thl_raw + column, (size_t)ncol};
    MynnColumn<const real> det_sqv = {det_sqv_raw + column, (size_t)ncol};
    MynnColumn<const real> det_sqc = {det_sqc_raw + column, (size_t)ncol};
    MynnColumn<const real> det_u = {det_u_raw + column, (size_t)ncol};
    MynnColumn<const real> det_v = {det_v_raw + column, (size_t)ncol};
    MynnColumn<const real> s_aw = {s_aw_raw + column, (size_t)ncol};
    MynnColumn<const real> s_awthl = {s_awthl_raw + column, (size_t)ncol};
    MynnColumn<const real> s_awqv = {s_awqv_raw + column, (size_t)ncol};
    MynnColumn<const real> s_awqc = {s_awqc_raw + column, (size_t)ncol};
    MynnColumn<const real> s_awu = {s_awu_raw + column, (size_t)ncol};
    MynnColumn<const real> s_awv = {s_awv_raw + column, (size_t)ncol};
    MynnColumn<const real> sd_aw = {sd_aw_raw + column, (size_t)ncol};
    MynnColumn<const real> sd_awthl = {sd_awthl_raw + column, (size_t)ncol};
    MynnColumn<const real> sd_awqv = {sd_awqv_raw + column, (size_t)ncol};
    MynnColumn<const real> sd_awqc = {sd_awqc_raw + column, (size_t)ncol};
    MynnColumn<const real> sd_awu = {sd_awu_raw + column, (size_t)ncol};
    MynnColumn<const real> sd_awv = {sd_awv_raw + column, (size_t)ncol};
    MynnColumn<real> du = {du_raw + column, (size_t)ncol};
    MynnColumn<real> dv = {dv_raw + column, (size_t)ncol};
    MynnColumn<real> dth = {dth_raw + column, (size_t)ncol};
    MynnColumn<real> dqv = {dqv_raw + column, (size_t)ncol};
    MynnColumn<real> dqc = {dqc_raw + column, (size_t)ncol};
    MynnColumn<real> dqi = {dqi_raw + column, (size_t)ncol};
    MynnColumn<real> dqs = {dqs_raw + column, (size_t)ncol};
    MynnColumn<real> dozone = {dozone_raw + column, (size_t)ncol};
    MynnColumn<real> thl = {thl_raw + column, (size_t)ncol};
    MynnColumn<real> dtz = {dtz_raw + column, (size_t)ncol};
    MynnColumn<real> rhoinv = {rhoinv_raw + column, (size_t)ncol};
    MynnColumn<real> delp = {delp_raw + column, (size_t)ncol};
    MynnColumn<real> khdz = {khdz_raw + column, (size_t)ncol};
    MynnColumn<real> kmdz = {kmdz_raw + column, (size_t)ncol};
    MynnColumn<real> a = {a_raw + column, (size_t)ncol};
    MynnColumn<real> b = {b_raw + column, (size_t)ncol};
    MynnColumn<real> c = {c_raw + column, (size_t)ncol};
    MynnColumn<real> d = {d_raw + column, (size_t)ncol};
    MynnColumn<real> cpw = {cpw_raw + column, (size_t)ncol};
    MynnColumn<real> dpw = {dpw_raw + column, (size_t)ncol};
    MynnColumn<real> sqv2 = {sqv2_raw + column, (size_t)ncol};
    MynnColumn<real> sqc2 = {sqc2_raw + column, (size_t)ncol};
    MynnColumn<real> sqi2 = {sqi2_raw + column, (size_t)ncol};
    MynnColumn<real> sqs2 = {sqs2_raw + column, (size_t)ncol};
#if defined(MYNN_GSD41)
    MynnColumn<const real> qc_original = {qc_original_raw + column, (size_t)ncol};
    MynnColumn<const real> qi_original = {qi_original_raw + column, (size_t)ncol};
#endif
    const int base = 0;
    const int zbase = 0;
    const int top = base + nz - 1;

    const real p608 = RV / RD - 1.0f;
    const real xlvcp = XLV / CP;
    const real xlscp = (XLV + 3.50e5f) / CP;

    const real delt = delt_col[column];
    const real psfc = psfc_col[column];
    const real ust = ust_col[column];
    const real wspd = wspd_col[column];
    const real uoce = uoce_col[column];
    const real voce = voce_col[column];
    const real flt = flt_col[column];
    const real flqv = flqv_col[column];
    const real flqc = flqc_col[column];

    // ---- module_bl_mynn.F:4136-4163 diffusion "constants" ----------------
#if defined(MYNN_GSD41)
    // Fork mynn_tendencies :2835-3108 uses DFm/DFh directly, including
    // surface forcing. Density still enters the separate moisture repair.
    real rhosfc = 1.0f;
    dtz[base] = MYNN_DIV(delt, dz[base]);
    real rhoz = rho[base];
    rhoinv[base] = 1.0f;
    // The fork's bottom row uses only the upper interface (:2968, :3112).
    khdz[zbase] = 0.0f;
    kmdz[zbase] = 0.0f;
#else
    real rhosfc = MYNN_DIV(
        psfc,
        MYNN_MUL(RD, MYNN_ADD(tk[base], MYNN_MUL(p608, qv[base]))));
    dtz[base] = MYNN_DIV(delt, dz[base]);
    real rhoz = rho[base];
    rhoinv[base] = MYNN_DIV(1.0f, rho[base]);
    khdz[zbase] = MYNN_MUL(rhoz, dfh[base]);
    kmdz[zbase] = MYNN_MUL(rhoz, dfm[base]);
#endif
    delp[base] = MYNN_SUB(
        psfc,
        MYNN_DIV(MYNN_ADD(MYNN_MUL(p[base + 1], dz[base]),
                          MYNN_MUL(p[base], dz[base + 1])),
                 MYNN_ADD(dz[base], dz[base + 1])));
    for (int k = 1; k < nz; ++k) {
        int idx = base + k;
        dtz[idx] = MYNN_DIV(delt, dz[idx]);
#if defined(MYNN_GSD41)
        rhoinv[idx] = 1.0f;
        khdz[zbase + k] = dfh[idx];
        kmdz[zbase + k] = dfm[idx];
#else
        rhoz = MYNN_DIV(
            MYNN_ADD(MYNN_MUL(rho[idx], dz[idx - 1]),
                     MYNN_MUL(rho[idx - 1], dz[idx])),
            MYNN_ADD(dz[idx - 1], dz[idx]));
        rhoz = mynn_max2(rhoz, 1.0e-4f);
        rhoinv[idx] = MYNN_DIV(1.0f, mynn_max2(rho[idx], 1.0e-4f));
        khdz[zbase + k] = MYNN_MUL(rhoz, dfh[idx]);
        kmdz[zbase + k] = MYNN_MUL(rhoz, dfm[idx]);
#endif
    }
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k;
        delp[idx] = MYNN_SUB(
            MYNN_DIV(MYNN_ADD(MYNN_MUL(p[idx], dz[idx - 1]),
                              MYNN_MUL(p[idx - 1], dz[idx])),
                     MYNN_ADD(dz[idx], dz[idx - 1])),
            MYNN_DIV(MYNN_ADD(MYNN_MUL(p[idx + 1], dz[idx]),
                              MYNN_MUL(p[idx], dz[idx + 1])),
                     MYNN_ADD(dz[idx], dz[idx + 1])));
    }
    delp[top] = delp[top - 1];
    // rhoz(kte+1)=rhoz(kte): rhoz still holds the top mass-level value.
#if defined(MYNN_GSD41)
    khdz[zbase + nz] = dfh[top];
    kmdz[zbase + nz] = dfm[top];
#else
    khdz[zbase + nz] = MYNN_MUL(rhoz, dfh[top]);
    kmdz[zbase + nz] = MYNN_MUL(rhoz, dfm[top]);
#endif

    // module_bl_mynn.F:4165-4171 mass-flux stability floors.  Inert while
    // s_aw is zero, but they are why khdz/kmdz cannot simply be reused from
    // mym_predict once DMP_mf is live.
#if !defined(MYNN_GSD41)
    for (int k = 1; k < nz - 1; ++k) {
        int zk = zbase + k;
        khdz[zk] = mynn_max2(khdz[zk], MYNN_MUL(0.5f, s_aw[zk]));
        khdz[zk] = mynn_max2(
            khdz[zk], -MYNN_MUL(0.5f, MYNN_SUB(s_aw[zk], s_aw[zk + 1])));
        kmdz[zk] = mynn_max2(kmdz[zk], MYNN_MUL(0.5f, s_aw[zk]));
        kmdz[zk] = mynn_max2(
            kmdz[zk], -MYNN_MUL(0.5f, MYNN_SUB(s_aw[zk], s_aw[zk + 1])));
    }
#endif

    // 0.5*dtz(k)*rhoinv(k) and dtz(k)*rhoinv(k) are the shared prefactors of
    // every mass-flux term; the surface pair is reused by all six systems.
    const real hdz0 = MYNN_MUL(MYNN_MUL(0.5f, dtz[base]), rhoinv[base]);
    const real dzinv0 = MYNN_MUL(dtz[base], rhoinv[base]);

    // ---- module_bl_mynn.F:4176-4296 momentum -----------------------------
    real drag = MYNN_DIV(MYNN_MUL(rhosfc, MYNN_MUL(ust, ust)), wspd);
    a[base] = -MYNN_MUL(MYNN_MUL(dtz[base], kmdz[zbase]), rhoinv[base]);
    b[base] = MYNN_SUB(
        MYNN_ADD(1.0f,
                 MYNN_MUL(MYNN_MUL(dtz[base],
                                   MYNN_ADD(kmdz[zbase + 1], drag)),
                          rhoinv[base])),
        MYNN_MUL(MYNN_MUL(hdz0, s_aw[zbase + 1]), onoff));
    b[base] = MYNN_SUB(b[base],
                       MYNN_MUL(MYNN_MUL(hdz0, sd_aw[zbase + 1]), onoff));
    c[base] = MYNN_SUB(
        -MYNN_MUL(MYNN_MUL(dtz[base], kmdz[zbase + 1]), rhoinv[base]),
        MYNN_MUL(MYNN_MUL(hdz0, s_aw[zbase + 1]), onoff));
    c[base] = MYNN_SUB(c[base],
                       MYNN_MUL(MYNN_MUL(hdz0, sd_aw[zbase + 1]), onoff));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        real hdz = MYNN_MUL(MYNN_MUL(0.5f, dtz[idx]), rhoinv[idx]);
        a[idx] = MYNN_ADD(
            -MYNN_MUL(MYNN_MUL(dtz[idx], kmdz[zk]), rhoinv[idx]),
            MYNN_MUL(MYNN_MUL(hdz, s_aw[zk]), onoff));
        a[idx] = MYNN_ADD(a[idx],
                          MYNN_MUL(MYNN_MUL(hdz, sd_aw[zk]), onoff));
        b[idx] = MYNN_ADD(
            MYNN_ADD(1.0f,
                     MYNN_MUL(MYNN_MUL(dtz[idx],
                                       MYNN_ADD(kmdz[zk], kmdz[zk + 1])),
                              rhoinv[idx])),
            MYNN_MUL(MYNN_MUL(hdz, MYNN_SUB(s_aw[zk], s_aw[zk + 1])), onoff));
        b[idx] = MYNN_ADD(
            b[idx],
            MYNN_MUL(MYNN_MUL(hdz, MYNN_SUB(sd_aw[zk], sd_aw[zk + 1])),
                     onoff));
        c[idx] = MYNN_SUB(
            -MYNN_MUL(MYNN_MUL(dtz[idx], kmdz[zk + 1]), rhoinv[idx]),
            MYNN_MUL(MYNN_MUL(hdz, s_aw[zk + 1]), onoff));
        c[idx] = MYNN_SUB(c[idx],
                          MYNN_MUL(MYNN_MUL(hdz, sd_aw[zk + 1]), onoff));
    }
    a[top] = 0.0f;
    b[top] = 1.0f;
    c[top] = 0.0f;

    d[base] = MYNN_SUB(
        MYNN_ADD(u[base],
                 MYNN_DIV(MYNN_MUL(MYNN_MUL(dtz[base], uoce),
                                   MYNN_MUL(ust, ust)),
                          wspd)),
        MYNN_MUL(MYNN_MUL(dzinv0, s_awu[zbase + 1]), onoff));
    d[base] = MYNN_ADD(d[base],
                       MYNN_MUL(MYNN_MUL(dzinv0, sd_awu[zbase + 1]), onoff));
    d[base] = MYNN_ADD(MYNN_ADD(d[base], MYNN_MUL(sub_u[base], delt)),
                       MYNN_MUL(det_u[base], delt));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        real dzinv = MYNN_MUL(dtz[idx], rhoinv[idx]);
        d[idx] = MYNN_ADD(
            u[idx],
            MYNN_MUL(MYNN_MUL(dzinv, MYNN_SUB(s_awu[zk], s_awu[zk + 1])),
                     onoff));
        d[idx] = MYNN_SUB(
            d[idx],
            MYNN_MUL(MYNN_MUL(dzinv, MYNN_SUB(sd_awu[zk], sd_awu[zk + 1])),
                     onoff));
        d[idx] = MYNN_ADD(MYNN_ADD(d[idx], MYNN_MUL(sub_u[idx], delt)),
                          MYNN_MUL(det_u[idx], delt));
    }
    d[top] = u[top];
    mynn_tridiag2_column(a + base, b + base, c + base, d + base,
                         cpw + base, dpw + base, du + base, nz);
    for (int k = 0; k < nz; ++k)
        du[base + k] = MYNN_DIV(MYNN_SUB(du[base + k], u[base + k]), delt);

    d[base] = MYNN_SUB(
        MYNN_ADD(v[base],
                 MYNN_DIV(MYNN_MUL(MYNN_MUL(dtz[base], voce),
                                   MYNN_MUL(ust, ust)),
                          wspd)),
        MYNN_MUL(MYNN_MUL(dzinv0, s_awv[zbase + 1]), onoff));
    d[base] = MYNN_ADD(d[base],
                       MYNN_MUL(MYNN_MUL(dzinv0, sd_awv[zbase + 1]), onoff));
    d[base] = MYNN_ADD(MYNN_ADD(d[base], MYNN_MUL(sub_v[base], delt)),
                       MYNN_MUL(det_v[base], delt));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        real dzinv = MYNN_MUL(dtz[idx], rhoinv[idx]);
        d[idx] = MYNN_ADD(
            v[idx],
            MYNN_MUL(MYNN_MUL(dzinv, MYNN_SUB(s_awv[zk], s_awv[zk + 1])),
                     onoff));
        d[idx] = MYNN_SUB(
            d[idx],
            MYNN_MUL(MYNN_MUL(dzinv, MYNN_SUB(sd_awv[zk], sd_awv[zk + 1])),
                     onoff));
        d[idx] = MYNN_ADD(MYNN_ADD(d[idx], MYNN_MUL(sub_v[idx], delt)),
                          MYNN_MUL(det_v[idx], delt));
    }
    d[top] = v[top];
    mynn_tridiag2_column(a + base, b + base, c + base, d + base,
                         cpw + base, dpw + base, dv + base, nz);
    for (int k = 0; k < nz; ++k)
        dv[base + k] = MYNN_DIV(MYNN_SUB(dv[base + k], v[base + k]), delt);

    // ---- module_bl_mynn.F:4318-4335 shared heat/moisture matrix ----------
    a[base] = -MYNN_MUL(MYNN_MUL(dtz[base], khdz[zbase]), rhoinv[base]);
    b[base] = MYNN_SUB(
        MYNN_ADD(1.0f,
                 MYNN_MUL(MYNN_MUL(dtz[base],
                                   MYNN_ADD(khdz[zbase + 1], khdz[zbase])),
                          rhoinv[base])),
        MYNN_MUL(hdz0, s_aw[zbase + 1]));
    b[base] = MYNN_SUB(b[base], MYNN_MUL(hdz0, sd_aw[zbase + 1]));
    c[base] = MYNN_SUB(
        -MYNN_MUL(MYNN_MUL(dtz[base], khdz[zbase + 1]), rhoinv[base]),
        MYNN_MUL(hdz0, s_aw[zbase + 1]));
    c[base] = MYNN_SUB(c[base], MYNN_MUL(hdz0, sd_aw[zbase + 1]));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        real hdz = MYNN_MUL(MYNN_MUL(0.5f, dtz[idx]), rhoinv[idx]);
        a[idx] = MYNN_ADD(
            -MYNN_MUL(MYNN_MUL(dtz[idx], khdz[zk]), rhoinv[idx]),
            MYNN_MUL(hdz, s_aw[zk]));
        a[idx] = MYNN_ADD(a[idx], MYNN_MUL(hdz, sd_aw[zk]));
        b[idx] = MYNN_ADD(
            MYNN_ADD(1.0f,
                     MYNN_MUL(MYNN_MUL(dtz[idx],
                                       MYNN_ADD(khdz[zk], khdz[zk + 1])),
                              rhoinv[idx])),
            MYNN_MUL(hdz, MYNN_SUB(s_aw[zk], s_aw[zk + 1])));
        b[idx] = MYNN_ADD(b[idx],
                          MYNN_MUL(hdz, MYNN_SUB(sd_aw[zk], sd_aw[zk + 1])));
        c[idx] = MYNN_SUB(
            -MYNN_MUL(MYNN_MUL(dtz[idx], khdz[zk + 1]), rhoinv[idx]),
            MYNN_MUL(hdz, s_aw[zk + 1]));
        c[idx] = MYNN_SUB(c[idx], MYNN_MUL(hdz, sd_aw[zk + 1]));
    }
    a[top] = 0.0f;
    b[top] = 1.0f;
    c[top] = 0.0f;

    // ---- module_bl_mynn.F:4298-4372 liquid-water potential temperature ---
    d[base] = MYNN_ADD(
        MYNN_ADD(thl_in[base],
                 MYNN_MUL(MYNN_MUL(MYNN_MUL(dtz[base], rhosfc), flt),
                          rhoinv[base])),
        MYNN_MUL(tcd[base], delt));
    d[base] = MYNN_SUB(
        MYNN_SUB(d[base], MYNN_MUL(dzinv0, s_awthl[zbase + 1])),
        MYNN_MUL(dzinv0, sd_awthl[zbase + 1]));
    d[base] = MYNN_ADD(
        MYNN_ADD(MYNN_ADD(d[base], MYNN_MUL(diss_heat[base], delt)),
                 MYNN_MUL(sub_thl[base], delt)),
        MYNN_MUL(det_thl[base], delt));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        real dzinv = MYNN_MUL(dtz[idx], rhoinv[idx]);
        d[idx] = MYNN_ADD(thl_in[idx], MYNN_MUL(tcd[idx], delt));
        d[idx] = MYNN_ADD(
            d[idx],
            MYNN_MUL(dzinv, MYNN_SUB(s_awthl[zk], s_awthl[zk + 1])));
        d[idx] = MYNN_ADD(
            d[idx],
            MYNN_MUL(dzinv, MYNN_SUB(sd_awthl[zk], sd_awthl[zk + 1])));
        d[idx] = MYNN_ADD(
            MYNN_ADD(MYNN_ADD(d[idx], MYNN_MUL(diss_heat[idx], delt)),
                     MYNN_MUL(sub_thl[idx], delt)),
            MYNN_MUL(det_thl[idx], delt));
    }
    d[top] = thl_in[top];
    mynn_tridiag2_column(a + base, b + base, c + base, d + base,
                         cpw + base, dpw + base, thl + base, nz);

    // ---- module_bl_mynn.F:4432-4489 cloud water --------------------------
    d[base] = MYNN_ADD(
        MYNN_ADD(sqc[base],
                 MYNN_MUL(MYNN_MUL(MYNN_MUL(dtz[base], rhosfc), flqc),
                          rhoinv[base])),
        MYNN_MUL(qcd[base], delt));
    d[base] = MYNN_SUB(
        MYNN_SUB(d[base], MYNN_MUL(dzinv0, s_awqc[zbase + 1])),
        MYNN_MUL(dzinv0, sd_awqc[zbase + 1]));
    d[base] = MYNN_ADD(d[base], MYNN_MUL(det_sqc[base], delt));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        real dzinv = MYNN_MUL(dtz[idx], rhoinv[idx]);
        d[idx] = MYNN_ADD(sqc[idx], MYNN_MUL(qcd[idx], delt));
        d[idx] = MYNN_ADD(
            d[idx], MYNN_MUL(dzinv, MYNN_SUB(s_awqc[zk], s_awqc[zk + 1])));
        d[idx] = MYNN_ADD(
            d[idx], MYNN_MUL(dzinv, MYNN_SUB(sd_awqc[zk], sd_awqc[zk + 1])));
        d[idx] = MYNN_ADD(d[idx], MYNN_MUL(det_sqc[idx], delt));
    }
    d[top] = sqc[top];
    mynn_tridiag2_column(a + base, b + base, c + base, d + base,
                         cpw + base, dpw + base, sqc2 + base, nz);

    // ---- module_bl_mynn.F:4491-4553 water vapour -------------------------
    // module_bl_mynn.F:4514-4518 limits an unreasonably large *negative*
    // surface moisture flux.  For any positive sqv(kts) the MIN collapses to
    // 0.0, so a downward flux is not limited but deleted.
    real qvflux = flqv;
#if !defined(MYNN_GSD41)
    if (qvflux < 0.0f) {
        qvflux = mynn_max2(
            qvflux,
            MYNN_DIV(
                mynn_min2(MYNN_SUB(MYNN_MUL(0.9f, sqv[base]), 1.0e-8f), 0.0f),
                dtz[base]));
    }
#else
    // MYNN_GSD41 (bl_mynn_version = "gsd_41"): the GSD MYNN v4.1 vapour
    // equation adds dtz*flqv with whatever sign it has (NOAA-EMC WRF 3.9
    // module_bl_mynn.F:4408 flqv = qfx/rho, :3114), so dew and frost take
    // their water from the lowest layer, as the land model's negative QFX
    // says they do.
#endif
    d[base] = MYNN_ADD(
        MYNN_ADD(sqv[base],
                 MYNN_MUL(MYNN_MUL(MYNN_MUL(dtz[base], rhosfc), qvflux),
                          rhoinv[base])),
        MYNN_MUL(qcd[base], delt));
    d[base] = MYNN_SUB(
        MYNN_SUB(d[base], MYNN_MUL(dzinv0, s_awqv[zbase + 1])),
        MYNN_MUL(dzinv0, sd_awqv[zbase + 1]));
    d[base] = MYNN_ADD(MYNN_ADD(d[base], MYNN_MUL(sub_sqv[base], delt)),
                       MYNN_MUL(det_sqv[base], delt));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        real dzinv = MYNN_MUL(dtz[idx], rhoinv[idx]);
        d[idx] = MYNN_ADD(sqv[idx], MYNN_MUL(qcd[idx], delt));
        d[idx] = MYNN_ADD(
            d[idx], MYNN_MUL(dzinv, MYNN_SUB(s_awqv[zk], s_awqv[zk + 1])));
        d[idx] = MYNN_ADD(
            d[idx], MYNN_MUL(dzinv, MYNN_SUB(sd_awqv[zk], sd_awqv[zk + 1])));
        d[idx] = MYNN_ADD(MYNN_ADD(d[idx], MYNN_MUL(sub_sqv[idx], delt)),
                          MYNN_MUL(det_sqv[idx], delt));
    }
    d[top] = sqv[top];
    mynn_tridiag2_column(a + base, b + base, c + base, d + base,
                         cpw + base, dpw + base, sqv2 + base, nz);

    // ---- module_bl_mynn.F:4566-4611 cloud ice: pure diffusion ------------
    a[base] = -MYNN_MUL(MYNN_MUL(dtz[base], khdz[zbase]), rhoinv[base]);
    b[base] = MYNN_ADD(
        1.0f,
        MYNN_MUL(MYNN_MUL(dtz[base], MYNN_ADD(khdz[zbase + 1], khdz[zbase])),
                 rhoinv[base]));
    c[base] = -MYNN_MUL(MYNN_MUL(dtz[base], khdz[zbase + 1]), rhoinv[base]);
    d[base] = sqi[base];
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        a[idx] = -MYNN_MUL(MYNN_MUL(dtz[idx], khdz[zk]), rhoinv[idx]);
        b[idx] = MYNN_ADD(
            1.0f,
            MYNN_MUL(MYNN_MUL(dtz[idx], MYNN_ADD(khdz[zk], khdz[zk + 1])),
                     rhoinv[idx]));
        c[idx] = -MYNN_MUL(MYNN_MUL(dtz[idx], khdz[zk + 1]), rhoinv[idx]);
        d[idx] = sqi[idx];
    }
    a[top] = 0.0f;
    b[top] = 1.0f;
    c[top] = 0.0f;
    d[top] = sqi[top];
    mynn_tridiag2_column(a + base, b + base, c + base, d + base,
                         cpw + base, dpw + base, sqi2 + base, nz);
    // Snow mixing is hard-disabled at module_bl_mynn.F:4618.
    for (int k = 0; k < nz; ++k) sqs2[base + k] = sqs[base + k];

    // ---- module_bl_mynn.F:4931-4993 species tendencies -------------------
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        dqv[idx] = MYNN_DIV(MYNN_SUB(sqv2[idx], sqv[idx]), delt);
        dqc[idx] = MYNN_DIV(MYNN_SUB(sqc2[idx], sqc[idx]), delt);
        dqi[idx] = MYNN_DIV(MYNN_SUB(sqi2[idx], sqi[idx]), delt);
        dqs[idx] = 0.0f;
        // module_bl_mynn.F:4173 zeroes dth so moisture_check can accumulate
        // into it; the theta block below then overwrites it, but the repair's
        // mutation of thl, sqc2, and sqi2 still reaches the answer.  That
        // asymmetry is WRF's, not a transcription slip.
        dth[idx] = 0.0f;
    }
#if defined(MYNN_GSD41)
    if (!old_cloud_tendencies) {
#endif
    mynn_moisture_check_column(
        delt, delp + base, exner + base, sqv2 + base, sqc2 + base,
        sqi2 + base, sqs2 + base, thl + base, dqv + base, dqc + base,
        dqi + base, dqs + base, dth + base, nz);
#if defined(MYNN_GSD41)
    }
    // Fork :3404, :3417 and :3449: convert the updated specific humidity
    // with updated vapour, then subtract the actual original mixing ratio.
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        real denominator = MYNN_SUB(1.0f, sqv2[idx]);
        if (denominator == 0.0f) {
            // The source divides by zero at all-vapour specific humidity.
            // Defined fallback: keep the original three mixing ratios.
            dqv[idx] = 0.0f; dqc[idx] = 0.0f; dqi[idx] = 0.0f;
        } else {
            dqv[idx] = MYNN_DIV(MYNN_SUB(MYNN_DIV(sqv2[idx], denominator), qv[idx]), delt);
            dqc[idx] = MYNN_DIV(MYNN_SUB(MYNN_DIV(sqc2[idx], denominator), qc_original[idx]), delt);
            dqi[idx] = MYNN_DIV(MYNN_SUB(MYNN_DIV(sqi2[idx], denominator), qi_original[idx]), delt);
            if (old_cloud_tendencies) {
                // Fork :3418-3421 and :3450-3453, a defined water-creating
                // clip. Only the explicit source-defect option takes it.
                if (MYNN_ADD(MYNN_MUL(dqc[idx], delt), qc_original[idx]) < 0.0f)
                    dqc[idx] = MYNN_DIV(-qc_original[idx], delt);
                if (MYNN_ADD(MYNN_MUL(dqi[idx], delt), qi_original[idx]) < 0.0f)
                    dqi[idx] = MYNN_DIV(-qi_original[idx], delt);
            }
        }
    }
#endif

    // ---- module_bl_mynn.F:5024-5031 ozone --------------------------------
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        dozone[idx] = 0.0f;
        if (MYNN_ADD(MYNN_MUL(dozone[idx], delt), ozone[idx]) < 0.0f)
            dozone[idx] = MYNN_DIV(-MYNN_MUL(ozone[idx], 0.99f), delt);
    }

#if defined(MYNN_GSD41)
    // Fork :3487-3497 under the explicit defect form; conserving heat otherwise.
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        dth[idx] = MYNN_DIV(
            MYNN_SUB(
                MYNN_ADD(
                    MYNN_ADD(thl[idx],
                             MYNN_MUL(MYNN_DIV(xlvcp, exner[idx]),
                                      old_cloud_tendencies ? sqc[idx] : sqc2[idx])),
                    MYNN_MUL(MYNN_DIV(xlscp, exner[idx]),
                             old_cloud_tendencies ? sqi[idx] : sqi2[idx])),
                th[idx]),
            delt);
    }
#else
    // ---- module_bl_mynn.F:5033-5046 theta --------------------------------
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        dth[idx] = MYNN_DIV(
            MYNN_SUB(
                MYNN_ADD(
                    MYNN_ADD(thl[idx],
                             MYNN_MUL(MYNN_DIV(xlvcp, exner[idx]),
                                      sqc2[idx])),
                    MYNN_MUL(MYNN_DIV(xlscp, exner[idx]), sqi2[idx])),
                th[idx]),
            delt);
    }
#endif
}

// ===========================================================================
// module_bl_mynn.F:1514-1674 mym_initialize, with bl_mynn_mixlength=1 and
// spp_pbl=0.  One thread owns one column: the five-iteration mym_length fixed
// point and the BouLac parcel walks are all sequential in the vertical.
//
// Every arithmetic step uses the round-to-nearest intrinsics for the same
// reason the tendency kernel does.  The transcendentals are the new hazard
// here: gfortran calls glibc, and glibc tanhf is a FP32 expm1f expression, not
// a correctly rounded function.  Over a four-million-point sweep it disagrees
// with the correctly rounded result on 1.8% of arguments, so it has to be
// reproduced rather than approximated.  mynn_expm1f below is a transcription
// of fdlibm s_expm1f.c, which glibc 2.39 still uses verbatim.
// ===========================================================================
__device__ __forceinline__ real mynn_scale_exponent(real y, int k)
{
    return __uint_as_float(__float_as_uint(y) + ((unsigned)k << 23));
}

__device__ real mynn_expm1f(real x)
{
    const real ln2_hi = __uint_as_float(0x3F317180u);
    const real ln2_lo = __uint_as_float(0x3717F7D1u);
    const real invln2 = __uint_as_float(0x3FB8AA3Bu);
    const real q1 = __uint_as_float(0xBD088889u);
    const real q2 = __uint_as_float(0x3AD00D01u);
    const real q3 = __uint_as_float(0xB8A670CDu);
    const real q4 = __uint_as_float(0x36867E54u);
    const real q5 = __uint_as_float(0xB457EDBBu);
    const real tiny = 1.0e-30f;

    unsigned word = __float_as_uint(x);
    unsigned sign = word & 0x80000000u;
    unsigned magnitude = word & 0x7FFFFFFFu;
    if (magnitude >= 0x4195B844u) {              // |x| >= 27*ln2
        if (magnitude >= 0x42B17218u) {          // |x| >= 88.72
            if (magnitude > 0x7F800000u) return MYNN_ADD(x, x);
            if (magnitude == 0x7F800000u) return sign == 0u ? x : -1.0f;
            if (x > 8.8721679688e01f) return __int_as_float(0x7F800000);
        }
        if (sign != 0u) return MYNN_SUB(tiny, 1.0f);
    }
    int k;
    real correction;
    if (magnitude > 0x3EB17218u) {               // |x| > 0.5*ln2
        real hi, lo;
        if (magnitude < 0x3F851592u) {           // |x| < 1.5*ln2
            if (sign == 0u) {
                hi = MYNN_SUB(x, ln2_hi); lo = ln2_lo; k = 1;
            } else {
                hi = MYNN_ADD(x, ln2_hi); lo = -ln2_lo; k = -1;
            }
        } else {
            k = (int)MYNN_ADD(MYNN_MUL(invln2, x), sign == 0u ? 0.5f : -0.5f);
            real scale = (real)k;
            hi = MYNN_SUB(x, MYNN_MUL(scale, ln2_hi));
            lo = MYNN_MUL(scale, ln2_lo);
        }
        x = MYNN_SUB(hi, lo);
        correction = MYNN_SUB(MYNN_SUB(hi, x), lo);
    } else if (magnitude < 0x33000000u) {        // |x| < 2**-25
        return x;
    } else {
        k = 0;
        correction = 0.0f;
    }

    real hfx = MYNN_MUL(0.5f, x);
    real hxs = MYNN_MUL(x, hfx);
    real r1 = MYNN_ADD(1.0f, MYNN_MUL(hxs, MYNN_ADD(q1, MYNN_MUL(hxs,
        MYNN_ADD(q2, MYNN_MUL(hxs, MYNN_ADD(q3, MYNN_MUL(hxs,
            MYNN_ADD(q4, MYNN_MUL(hxs, q5))))))))));
    real t = MYNN_SUB(3.0f, MYNN_MUL(r1, hfx));
    real e = MYNN_MUL(hxs, MYNN_DIV(MYNN_SUB(r1, t),
                                    MYNN_SUB(6.0f, MYNN_MUL(x, t))));
    if (k == 0) return MYNN_SUB(x, MYNN_SUB(MYNN_MUL(x, e), hxs));
    e = MYNN_SUB(MYNN_MUL(x, MYNN_SUB(e, correction)), correction);
    e = MYNN_SUB(e, hxs);
    if (k == -1) return MYNN_SUB(MYNN_MUL(0.5f, MYNN_SUB(x, e)), 0.5f);
    if (k == 1) {
        if (x < -0.25f)
            return MYNN_MUL(-2.0f, MYNN_SUB(e, MYNN_ADD(x, 0.5f)));
        return MYNN_ADD(1.0f, MYNN_MUL(2.0f, MYNN_SUB(x, e)));
    }
    real y;
    if (k <= -2 || k > 56) {
        y = MYNN_SUB(1.0f, MYNN_SUB(e, x));
        y = mynn_scale_exponent(y, k);
        return MYNN_SUB(y, 1.0f);
    }
    if (k < 23) {
        t = __uint_as_float(0x3F800000u - (0x1000000u >> k));
        y = MYNN_SUB(t, MYNN_SUB(e, x));
    } else {
        t = __uint_as_float((unsigned)(0x7F - k) << 23);
        y = MYNN_SUB(x, MYNN_ADD(e, t));
        y = MYNN_ADD(y, 1.0f);
    }
    return mynn_scale_exponent(y, k);
}

// glibc tanhf: 1 - 2/(expm1f(2|x|)+2) above 1, -t/(t+2) below,
// saturated at 22.
__device__ real mynn_tanhf(real x)
{
    const real tiny = 1.0e-30f;
    unsigned word = __float_as_uint(x);
    unsigned magnitude = word & 0x7FFFFFFFu;
    real z;
    if (magnitude >= 0x7F800000u)                // inf or NaN: one/x +- one
        return (word & 0x80000000u) == 0u ? MYNN_ADD(MYNN_DIV(1.0f, x), 1.0f)
                                          : MYNN_SUB(MYNN_DIV(1.0f, x), 1.0f);
    if (magnitude < 0x41B00000u) {               // |x| < 22
        if (magnitude < 0x24000000u)             // |x| < 2**-55
            return MYNN_MUL(x, MYNN_ADD(1.0f, x));
        real ax = __uint_as_float(magnitude);
        if (magnitude >= 0x3F800000u) {          // |x| >= 1
            real t = mynn_expm1f(MYNN_MUL(2.0f, ax));
            z = MYNN_SUB(1.0f, MYNN_DIV(2.0f, MYNN_ADD(t, 2.0f)));
        } else {
            real t = mynn_expm1f(MYNN_MUL(-2.0f, ax));
            z = MYNN_DIV(-t, MYNN_ADD(t, 2.0f));
        }
    } else {
        z = MYNN_SUB(1.0f, tiny);
    }
    return (word & 0x80000000u) == 0u ? z : -z;
}

// Fortran real**real, which gfortran routes to glibc powf.  This used to
// round an FP64 pow, on the claim that it agreed with glibc on every argument
// these routines reach; glibc's powf is not correctly rounded, so that was a
// third function.  It is glibc's own algorithm now (mynn_glibc_powf above).
__device__ __forceinline__ real mynn_powf(real x, real y)
{
    return mynn_glibc_powf(x, y);
}

// module_bl_mynn.F:2417-2567 boulac_length, keeping only the elBLavg branch
// that mym_length CASE(1) consumes.  dlu/dld are caller-owned scratch.
__device__ void mynn_boulac_elblavg(
    MynnColumn<const real> zw, MynnColumn<const real> dz,
    MynnColumn<const real> qtke, MynnColumn<const real> thetaw,
    MynnColumn<real> dlu, MynnColumn<real> dld,
    MynnColumn<real> elblavg, int nz)
{
#if !defined(MYNN_GSD41)
    const real beta = MYNN_DIV(9.81f, 300.0f);
#endif
    for (int iz = 0; iz < nz; ++iz) {
#if defined(MYNN_GSD41)
        // The fork's BouLac parcel buoyancy uses its interface theta.
        const real beta = MYNN_DIV(9.81f, thetaw[iz]);
#endif
        real zup = 0.0f;
        dlu[iz] = MYNN_SUB(MYNN_SUB(zw[nz], zw[iz]), MYNN_MUL(dz[iz], 0.5f));
        real zzz = 0.0f, zup_inf = 0.0f;
        if (iz < nz - 1) {
            int izz = iz, found = 0;
            while (!found) {
                if (izz < nz - 1) {
                    real dzt = dz[izz];
                    zup = MYNN_SUB(zup,
                                   MYNN_MUL(MYNN_MUL(beta, thetaw[iz]), dzt));
                    zup = MYNN_ADD(zup, MYNN_MUL(MYNN_MUL(MYNN_MUL(beta,
                        MYNN_ADD(thetaw[izz + 1], thetaw[izz])), dzt), 0.5f));
                    zzz = MYNN_ADD(zzz, dzt);
                    if (qtke[iz] < zup && qtke[iz] >= zup_inf) {
                        real bbb = MYNN_DIV(
                            MYNN_SUB(thetaw[izz + 1], thetaw[izz]), dzt);
                        real tl;
                        if (bbb != 0.0f) {
                            real b = MYNN_MUL(
                                beta, MYNN_SUB(thetaw[izz], thetaw[iz]));
                            real radical = mynn_max2(MYNN_ADD(MYNN_MUL(b, b),
                                MYNN_MUL(MYNN_MUL(MYNN_MUL(2.0f, bbb), beta),
                                         MYNN_SUB(qtke[iz], zup_inf))), 0.0f);
                            tl = MYNN_DIV(MYNN_DIV(
                                MYNN_ADD(-b, sqrtf(radical)), bbb), beta);
                        } else if (thetaw[izz] != thetaw[iz]) {
                            tl = MYNN_DIV(MYNN_SUB(qtke[iz], zup_inf),
                                MYNN_MUL(beta,
                                         MYNN_SUB(thetaw[izz], thetaw[iz])));
                        } else {
                            tl = 0.0f;
                        }
                        dlu[iz] = MYNN_ADD(MYNN_SUB(zzz, dzt), tl);
                        found = 1;
                    }
                    zup_inf = zup;
                    ++izz;
                } else {
                    found = 1;
                }
            }
        }

        real zdo = 0.0f, zdo_sup = 0.0f;
        dld[iz] = zw[iz];
        zzz = 0.0f;
        if (iz > 0) {
            int izz = iz, found = 0;
            while (!found) {
                if (izz > 0) {
                    real dzt = dz[izz - 1];
                    zdo = MYNN_ADD(zdo,
                                   MYNN_MUL(MYNN_MUL(beta, thetaw[iz]), dzt));
                    zdo = MYNN_SUB(zdo, MYNN_MUL(MYNN_MUL(MYNN_MUL(beta,
                        MYNN_ADD(thetaw[izz - 1], thetaw[izz])), dzt), 0.5f));
                    zzz = MYNN_ADD(zzz, dzt);
                    if (qtke[iz] < zdo && qtke[iz] >= zdo_sup) {
                        real bbb = MYNN_DIV(
                            MYNN_SUB(thetaw[izz], thetaw[izz - 1]), dzt);
                        real tl;
                        if (bbb != 0.0f) {
                            real b = MYNN_MUL(
                                beta, MYNN_SUB(thetaw[izz], thetaw[iz]));
                            real radical = mynn_max2(MYNN_ADD(MYNN_MUL(b, b),
                                MYNN_MUL(MYNN_MUL(MYNN_MUL(2.0f, bbb), beta),
                                         MYNN_SUB(qtke[iz], zdo_sup))), 0.0f);
                            tl = MYNN_DIV(MYNN_DIV(
                                MYNN_ADD(b, sqrtf(radical)), bbb), beta);
                        } else if (thetaw[izz] != thetaw[iz]) {
                            tl = MYNN_DIV(MYNN_SUB(qtke[iz], zdo_sup),
                                MYNN_MUL(beta,
                                         MYNN_SUB(thetaw[izz], thetaw[iz])));
                        } else {
                            tl = 0.0f;
                        }
                        dld[iz] = MYNN_ADD(MYNN_SUB(zzz, dzt), tl);
                        found = 1;
                    }
                    zdo_sup = zdo;
                    --izz;
                } else {
                    found = 1;
                }
            }
        }
        dld[iz] = mynn_min2(dld[iz], zw[iz + 1]);
#if defined(MYNN_GSD41)
        // The fork's CASE(1) consumes lb1, the minimum before either
        // displacement is clipped for lb2. Its soft limit and lid hold
        // follow module_bl_mynn.F:1389-1405 of the v4.1.21 source.
        elblavg[iz] = mynn_min2(dlu[iz], dld[iz]);
        elblavg[iz] = MYNN_DIV(elblavg[iz],
                               MYNN_ADD(1.0f, MYNN_DIV(elblavg[iz], 2000.0f)));
        if (iz == nz - 1) elblavg[iz] = elblavg[iz - 1];
#else
        real up = mynn_max2(0.1f, mynn_min2(dlu[iz], 1000.0f));
        real down = mynn_max2(0.1f, mynn_min2(dld[iz], 1000.0f));
        elblavg[iz] = sqrtf(MYNN_MUL(up, down));
        elblavg[iz] = MYNN_DIV(elblavg[iz],
                               MYNN_ADD(1.0f, MYNN_DIV(elblavg[iz], 2000.0f)));
        if (iz == nz - 1) elblavg[iz] = elblavg[iz - 1];
#endif
    }
}

// WRF v4.6.1 module_bl_mynn.F:2100-2232 mym_length CASE(2).
// See licenses/LICENSE-WRF-public-domain.txt. Ugrid, Uonset, cldavg and the
// stable-branch elb are source temporaries that never reach an output.
__device__ void mynn_mym_length_local_column(
    MynnColumn<const real> dz, MynnColumn<const real> zw,
    MynnColumn<const real> qke, MynnColumn<const real> dtv,
    MynnColumn<const real> edmf_w, MynnColumn<const real> edmf_a,
    real rmo, real fltv, real zi, real psig_bl,
    MynnColumn<real> el, MynnColumn<real> qkw, MynnColumn<real> qtke, int nz)
{
    const real gtr = MYNN_DIV(9.81f, 300.0f);
    const real onethird = MYNN_DIV(1.0f, 3.0f);
    real zi2 = mynn_max2(zi, 300.0f);
    real h1 = mynn_min2(mynn_max2(MYNN_MUL(0.3f, zi2), 300.0f), 600.0f);
    real h2 = MYNN_MUL(h1, 0.5f);
    qtke[0] = mynn_max2(MYNN_MUL(0.5f, qke[0]), MYNN_MUL(0.5f, 1.0e-3f));
    qkw[0] = sqrtf(mynn_max2(qke[0], 1.0e-3f));
    for (int k = 1; k < nz; ++k) {
        real afk = MYNN_DIV(dz[k], MYNN_ADD(dz[k], dz[k - 1]));
        real abk = MYNN_SUB(1.0f, afk);
        qkw[k] = sqrtf(mynn_max2(MYNN_ADD(
            MYNN_MUL(qke[k], abk), MYNN_MUL(qke[k - 1], afk)), 1.0e-3f));
        qtke[k] = MYNN_MUL(0.5f, MYNN_MUL(qkw[k], qkw[k]));
    }
    real elt = 1.0e-5f, vsc_sum = 1.0e-5f;
    real pblh_plus_ent = mynn_max2(MYNN_ADD(zi, h1), 100.0f);
    for (int k = 1; k < nz && zw[k] <= pblh_plus_ent; ++k) {
        real dzk = MYNN_MUL(0.5f, MYNN_ADD(dz[k], dz[k - 1]));
        real qdz = MYNN_MUL(mynn_min2(mynn_max2(qkw[k], 0.03f), 30.0f), dzk);
        elt = MYNN_ADD(elt, MYNN_MUL(qdz, zw[k]));
        vsc_sum = MYNN_ADD(vsc_sum, qdz);
    }
    elt = mynn_min2(mynn_max2(MYNN_DIV(MYNN_MUL(0.22f, elt), vsc_sum),
                              10.0f), 400.0f);
    real vsc = mynn_powf(MYNN_MUL(MYNN_MUL(gtr, elt), mynn_max2(fltv, 0.0f)),
                         onethird);
    el[0] = 0.0f;
    for (int k = 1; k < nz; ++k) {
        real zwk = zw[k];
        real dzk = MYNN_MUL(0.5f, MYNN_ADD(dz[k], dz[k - 1]));
        real wstar = MYNN_MUL(1.25f, mynn_powf(
            MYNN_MUL(MYNN_MUL(gtr, zi), mynn_max2(fltv, 1.0e-4f)), onethird));
        real weight = MYNN_ADD(MYNN_MUL(0.5f, mynn_tanhf(
            MYNN_DIV(MYNN_SUB(zwk, MYNN_ADD(zi2, h1)), h2))), 0.5f);
        real elb_mf, elf;
        if (dtv[k] > 0.0f) {
            real bv = mynn_max2(sqrtf(MYNN_MUL(gtr, dtv[k])), 0.001f);
            real numerator = mynn_max2(MYNN_MUL(0.30f, qkw[k]),
                MYNN_MUL(MYNN_MUL(50.0f, edmf_a[k - 1]), edmf_w[k - 1]));
            elb_mf = MYNN_MUL(MYNN_DIV(numerator, bv), MYNN_ADD(1.0f,
                MYNN_MUL(2.0f, sqrtf(MYNN_DIV(vsc, MYNN_MUL(bv, elt))))));
            real tau = mynn_min2(mynn_max2(
                MYNN_DIV(MYNN_MUL(1000.0f, wstar), 9.81f), 30.0f), 150.0f);
            tau = MYNN_ADD(MYNN_MUL(tau, MYNN_SUB(1.0f, weight)),
                           MYNN_MUL(50.0f, weight));
            elf = mynn_min2(mynn_max2(
                MYNN_MUL(tau, sqrtf(mynn_min2(qtke[k], 40.0f))),
                MYNN_DIV(MYNN_MUL(MYNN_MUL(50.0f, edmf_a[k]), edmf_w[k]), bv)),
                zwk);
        } else {
            real tau = mynn_min2(mynn_max2(
                MYNN_DIV(MYNN_MUL(1000.0f, wstar), 9.81f), 50.0f), 200.0f);
            tau = MYNN_ADD(MYNN_MUL(tau, MYNN_SUB(1.0f, weight)),
                MYNN_MUL(mynn_max2(100.0f, MYNN_MUL(dzk, 0.25f)), weight));
            elf = mynn_min2(MYNN_MUL(tau, sqrtf(mynn_min2(qtke[k], 40.0f))), zwk);
            elb_mf = elf;
        }
        elf = MYNN_DIV(elf, MYNN_ADD(1.0f, MYNN_DIV(elf, 800.0f)));
        elb_mf = mynn_max2(elb_mf, 0.01f);
        real els;
        if (rmo > 0.0f) {
            els = MYNN_DIV(MYNN_MUL(0.4f, zwk), MYNN_ADD(1.0f,
                MYNN_MUL(3.5f, mynn_min2(MYNN_MUL(zwk, rmo), 1.0f))));
        } else {
            els = MYNN_MUL(MYNN_MUL(0.4f, zwk), mynn_powf(MYNN_SUB(1.0f,
                MYNN_MUL(MYNN_MUL(5.0f, zwk), rmo)), 0.2f));
        }
        real els2 = MYNN_MUL(els, els);
        real value = sqrtf(MYNN_DIV(els2, MYNN_ADD(MYNN_ADD(1.0f,
            MYNN_DIV(els2, MYNN_MUL(elt, elt))),
            MYNN_DIV(els2, MYNN_MUL(elb_mf, elb_mf)))));
        value = MYNN_ADD(MYNN_MUL(value, MYNN_SUB(1.0f, weight)),
                         MYNN_MUL(elf, weight));
        real el_les = mynn_min2(MYNN_DIV(els,
            MYNN_ADD(1.0f, MYNN_DIV(els, 12.0f))), elb_mf);
        el[k] = MYNN_ADD(MYNN_MUL(value, psig_bl),
                         MYNN_MUL(MYNN_SUB(1.0f, psig_bl), el_les));
    }
}

#if defined(MYNN_GSD41)
// GSD MYNN v4.1, NOAA-EMC WRF 3.9 branch (tag v4.1.21)
// phys/module_bl_mynn.F:940-1085 mym_length CASE(2).  Public domain (WRF).
// flt is the surface kinematic heat flux and vflx the buoyancy flux the
// caller built from it, (vt(kts)+1)*flt + (vq(kts)+tv0)*flq (:1015); the
// cold start passes both as the zeros mym_initialize holds (:448).
// cldavg (:1003) is a source temporary that never reaches an output.  The
// height integral stops at the top interface below the model lid, where the
// Fortran DO WHILE would read dz(kte+1); a PBL that deep does not occur.
__device__ void mynn_mym_length_gsd41_column(
    MynnColumn<const real> dz, MynnColumn<const real> zw,
    MynnColumn<const real> qke, MynnColumn<const real> dtv,
    MynnColumn<const real> edmf_w, MynnColumn<const real> edmf_a,
    real rmo, real flt, real vflx, real zi, real psig_bl,
    MynnColumn<real> el, MynnColumn<real> qkw, MynnColumn<real> qtke,
    int nz, int unsquared_qtke)
{
    const real gtr = MYNN_DIV(9.81f, 300.0f);
    const real vk = 0.4f, cns = 3.5f, alp2 = 0.3f, alp3 = 2.0f;
    const real alp4 = 20.0f, alp5 = 0.3f, alp6 = 50.0f;
    // :974 alp1 = 0.25 + 0.02*MIN(MAX(zi-200.,0.),1000.)/1000.
    real alp1 = MYNN_ADD(0.25f, MYNN_DIV(MYNN_MUL(0.02f, mynn_min2(
        mynn_max2(MYNN_SUB(zi, 200.0f), 0.0f), 1000.0f)), 1000.0f));
    // :983-986 zi2 floor 100 m; minzi/maxdz/mindz = 300/750/300.
    real zi2 = mynn_max2(zi, 100.0f);
    real h1 = mynn_min2(mynn_max2(MYNN_MUL(0.3f, zi2), 300.0f), 750.0f);
    real h2 = MYNN_MUL(h1, 0.5f);
    // :988-996.  :995 converts with 0.5*qkw (no square); its option 1 (:880)
    // and every later generation use 0.5*qkw**2.  The squared form is the
    // default; bl_mynn_gsd41_unsquared_qtke takes the line as written.
    qtke[0] = mynn_max2(MYNN_MUL(0.5f, qke[0]), 0.01f);
    qkw[0] = sqrtf(mynn_max2(qke[0], 1.0e-10f));
    for (int k = 1; k < nz; ++k) {
        real afk = MYNN_DIV(dz[k], MYNN_ADD(dz[k], dz[k - 1]));
        real abk = MYNN_SUB(1.0f, afk);
        qkw[k] = sqrtf(mynn_max2(MYNN_ADD(
            MYNN_MUL(qke[k], abk), MYNN_MUL(qke[k - 1], afk)), 1.0e-3f));
        qtke[k] = unsquared_qtke ? MYNN_MUL(0.5f, qkw[k])
                                 : MYNN_MUL(0.5f, MYNN_MUL(qkw[k], qkw[k]));
    }
    // :998-1015.  qmin = 0 (:197), so MAX(qkw-qmin, 0.03) is MAX(qkw, 0.03).
    real elt = 1.0e-5f, vsc = 1.0e-5f;
    real pblh_plus_ent = mynn_max2(MYNN_ADD(zi, h1), 100.0f);
    for (int k = 1; k < nz && zw[k] <= pblh_plus_ent; ++k) {
        real dzk = MYNN_MUL(0.5f, MYNN_ADD(dz[k], dz[k - 1]));
        real qdz = MYNN_MUL(mynn_max2(MYNN_SUB(qkw[k], 0.0f), 0.03f), dzk);
        elt = MYNN_ADD(elt, MYNN_MUL(qdz, zw[k]));
        vsc = MYNN_ADD(vsc, qdz);
    }
    elt = mynn_max2(MYNN_DIV(MYNN_MUL(alp1, elt), vsc), 10.0f);
    vsc = mynn_powf(MYNN_MUL(MYNN_MUL(gtr, elt), mynn_max2(vflx, 0.0f)),
                    0.33333f);
    el[0] = 0.0f;
    for (int k = 1; k < nz; ++k) {
        real zwk = zw[k];
        real elb, elf, elb_mf;
        if (dtv[k] > 0.0f) {
            // :1026-1036
            real bv = sqrtf(MYNN_MUL(gtr, dtv[k]));
            elb_mf = MYNN_MUL(
                MYNN_DIV(mynn_max2(MYNN_MUL(alp2, qkw[k]),
                    MYNN_MUL(MYNN_MUL(alp6, edmf_a[k]), edmf_w[k])), bv),
                MYNN_ADD(1.0f, MYNN_MUL(alp3,
                    sqrtf(MYNN_DIV(vsc, MYNN_MUL(bv, elt))))));
            elb = mynn_min2(MYNN_DIV(MYNN_MUL(alp5, qkw[k]), bv), zwk);
            elf = MYNN_DIV(elb, MYNN_ADD(1.0f, MYNN_DIV(elb, 600.0f)));
        } else {
            // :1054-1061
            real tau_cloud = mynn_min2(mynn_max2(MYNN_DIV(
                MYNN_MUL(0.5f, zi), mynn_powf(MYNN_MUL(MYNN_MUL(gtr, zi),
                    mynn_max2(flt, 1.0e-4f)), 0.3333f)), 50.0f), 150.0f);
            real wt = MYNN_ADD(MYNN_MUL(0.5f, mynn_tanhf(
                MYNN_DIV(MYNN_SUB(zwk, MYNN_ADD(zi2, h1)), h2))), 0.5f);
            tau_cloud = MYNN_ADD(MYNN_MUL(tau_cloud, MYNN_SUB(1.0f, wt)),
                                 MYNN_MUL(50.0f, wt));
            elb = mynn_min2(MYNN_MUL(tau_cloud,
                sqrtf(mynn_min2(qtke[k], 30.0f))), zwk);
            elf = elb;
            elb_mf = elb;
        }
        // :1064-1073 surface lengths at z and at z - 4 m.
        real z_m = mynn_max2(0.0f, MYNN_SUB(zwk, 4.0f));
        real els, els1;
        if (rmo > 0.0f) {
            real den = MYNN_ADD(1.0f, MYNN_MUL(cns,
                mynn_min2(MYNN_MUL(zwk, rmo), 1.0f)));
            els = MYNN_DIV(MYNN_MUL(vk, zwk), den);
            els1 = MYNN_DIV(MYNN_MUL(vk, z_m), den);
        } else {
            real fac = mynn_powf(MYNN_SUB(1.0f,
                MYNN_MUL(MYNN_MUL(alp4, zwk), rmo)), 0.2f);
            els = MYNN_MUL(MYNN_MUL(vk, zwk), fac);
            els1 = MYNN_MUL(MYNN_MUL(vk, z_m), fac);
        }
        // :1076-1085
        real wt = MYNN_ADD(MYNN_MUL(0.5f, mynn_tanhf(
            MYNN_DIV(MYNN_SUB(zwk, MYNN_ADD(zi2, h1)), h2))), 0.5f);
        real el_unstab = MYNN_DIV(els, MYNN_ADD(1.0f, MYNN_DIV(els1, elt)));
        real value = mynn_min2(el_unstab, elb_mf);
        value = MYNN_ADD(MYNN_MUL(value, MYNN_SUB(1.0f, wt)),
                         MYNN_MUL(elf, wt));
        real el_les = mynn_min2(MYNN_DIV(els,
            MYNN_ADD(1.0f, MYNN_DIV(els1, 12.0f))), elb_mf);
        el[k] = MYNN_ADD(MYNN_MUL(value, psig_bl),
                         MYNN_MUL(MYNN_SUB(1.0f, psig_bl), el_les));
    }
}
#define MYNN_GSD41_LENGTH_PARAMS , real flt, int unsquared_qtke
#define MYNN_GSD41_LENGTH_ARGS(flt_, unsq_) , (flt_), (unsq_)
#else
#define MYNN_GSD41_LENGTH_PARAMS
#define MYNN_GSD41_LENGTH_ARGS(flt_, unsq_)
#endif

// module_bl_mynn.F:1999-2098 mym_length CASE(1) for one column.  xland, dx,
// flt, flq, vt, vq, cldfra_bl1D and rstoch_col reach the Fortran but CASE(1)
// never reads them; dtv(kts) is likewise never read.
__device__ void mynn_mym_length_column(
    MynnColumn<const real> dz, MynnColumn<const real> zw,
    MynnColumn<const real> u, MynnColumn<const real> v,
    MynnColumn<const real> qke, MynnColumn<const real> dtv,
    MynnColumn<const real> theta, MynnColumn<const real> edmf_w,
    MynnColumn<const real> edmf_a, real rmo, real fltv, real zi,
    real psig_bl, MynnColumn<real> el, MynnColumn<real> qkw,
    MynnColumn<real> qtke, MynnColumn<real> thetaw,
    MynnColumn<real> elblavg, MynnColumn<real> dlu,
    MynnColumn<real> dld, int nz, int mixlength MYNN_GSD41_LENGTH_PARAMS)
{
    if (mixlength == 2) {
#if defined(MYNN_GSD41)
        // Under gsd_41 the caller hands the buoyancy flux vflx in fltv.
        mynn_mym_length_gsd41_column(dz, zw, qke, dtv, edmf_w, edmf_a,
            rmo, flt, fltv, zi, psig_bl, el, qkw, qtke, nz, unsquared_qtke);
#else
        mynn_mym_length_local_column(dz, zw, qke, dtv, edmf_w, edmf_a,
            rmo, fltv, zi, psig_bl, el, qkw, qtke, nz);
#endif
        return;
    }
#if defined(MYNN_GSD41)
    // The fork's operational CASE(1), module_bl_mynn.F:852-966. It uses
    // harmonic lengths and the minimum BouLac displacement, with no EDMF
    // contribution or WRF v4.6.1 wind-dependent enhancement.
    const real gtr = MYNN_DIV(9.81f, 300.0f);
    real zi2 = mynn_max2(zi, 300.0f);
    real h1 = mynn_min2(mynn_max2(MYNN_MUL(0.3f, zi2), 300.0f), 750.0f);
    real h2 = MYNN_DIV(h1, 2.0f);
    qtke[0] = mynn_max2(MYNN_DIV(qke[0], 2.0f), 0.01f);
    thetaw[0] = theta[0];
    qkw[0] = sqrtf(mynn_max2(qke[0], 1.0e-10f));
    for (int k = 1; k < nz; ++k) {
        real afk = MYNN_DIV(dz[k], MYNN_ADD(dz[k], dz[k - 1]));
        real abk = MYNN_SUB(1.0f, afk);
        qkw[k] = sqrtf(mynn_max2(MYNN_ADD(
            MYNN_MUL(qke[k], abk), MYNN_MUL(qke[k - 1], afk)), 1.0e-3f));
        qtke[k] = MYNN_DIV(MYNN_MUL(qkw[k], qkw[k]), 2.0f);
        thetaw[k] = MYNN_ADD(MYNN_MUL(theta[k], abk),
                             MYNN_MUL(theta[k - 1], afk));
    }
    real elt = 1.0e-5f, vsc_sum = 1.0e-5f;
    for (int k = 1; k < nz && zw[k] <= MYNN_ADD(zi2, h1); ++k) {
        real dzk = MYNN_MUL(0.5f, MYNN_ADD(dz[k], dz[k - 1]));
        real qdz = MYNN_MUL(mynn_max2(MYNN_SUB(qkw[k], 0.0f), 0.03f), dzk);
        elt = MYNN_ADD(elt, MYNN_MUL(qdz, zw[k]));
        vsc_sum = MYNN_ADD(vsc_sum, qdz);
    }
    elt = MYNN_DIV(MYNN_MUL(0.23f, elt), vsc_sum);
    real vsc = mynn_powf(MYNN_MUL(MYNN_MUL(gtr, elt), mynn_max2(fltv, 0.0f)),
                         MYNN_DIV(1.0f, 3.0f));
    mynn_boulac_elblavg(zw, dz, qtke, thetaw, dlu, dld, elblavg, nz);
    el[0] = 0.0f;
    for (int k = 1; k < nz; ++k) {
        real zwk = zw[k], elb, elf;
        if (dtv[k] > 0.0f) {
            real bv = sqrtf(MYNN_MUL(gtr, dtv[k]));
            elb = MYNN_MUL(MYNN_DIV(MYNN_MUL(0.65f, qkw[k]), bv),
                MYNN_ADD(1.0f, MYNN_MUL(MYNN_DIV(3.0f, 0.65f),
                    sqrtf(MYNN_DIV(vsc, MYNN_MUL(bv, elt))))));
            elb = mynn_min2(elb, zwk);
            elf = MYNN_DIV(MYNN_MUL(0.65f, qkw[k]), bv);
        } else {
            elb = 1.0e10f;
            elf = elb;
        }
        real els;
        if (rmo > 0.0f) {
            els = MYNN_DIV(MYNN_MUL(0.4f, zwk), MYNN_ADD(1.0f,
                MYNN_MUL(2.3f, mynn_min2(MYNN_MUL(zwk, rmo), 1.0f))));
        } else {
            els = MYNN_MUL(MYNN_MUL(0.4f, zwk), mynn_powf(MYNN_SUB(1.0f,
                MYNN_MUL(MYNN_MUL(20.0f, zwk), rmo)), 0.2f));
        }
        real weight = MYNN_ADD(MYNN_MUL(0.5f, mynn_tanhf(
            MYNN_DIV(MYNN_SUB(zwk, MYNN_ADD(zi2, h1)), h2))), 0.5f);
        real value = mynn_min2(MYNN_DIV(elb, MYNN_ADD(MYNN_ADD(
            MYNN_DIV(elb, elt), MYNN_DIV(elb, els)), 1.0f)), elf);
        value = MYNN_ADD(MYNN_MUL(value, MYNN_SUB(1.0f, weight)),
            MYNN_MUL(MYNN_MUL(0.4f, elblavg[k]), weight));
        el[k] = MYNN_MUL(value, psig_bl);
    }
#else
    const real gtr = MYNN_DIV(9.81f, 300.0f);
    real ugrid = sqrtf(MYNN_ADD(MYNN_MUL(u[0], u[0]), MYNN_MUL(v[0], v[0])));
    real wt_u = MYNN_SUB(1.0f, mynn_min2(
        MYNN_DIV(mynn_max2(MYNN_SUB(ugrid, 15.0f), 0.0f), 30.0f), 0.5f));
    real alp3 = MYNN_MUL(2.5f, wt_u);
    real zi2 = mynn_max2(zi, 300.0f);
    real h1 = mynn_min2(mynn_max2(MYNN_MUL(0.3f, zi2), 300.0f), 600.0f);
    real h2 = MYNN_DIV(h1, 2.0f);
    qtke[0] = mynn_max2(MYNN_MUL(0.5f, qke[0]), 0.5e-3f);
    thetaw[0] = theta[0];
    qkw[0] = sqrtf(mynn_max2(qke[0], 1.0e-3f));
    for (int k = 1; k < nz; ++k) {
        real afk = MYNN_DIV(dz[k], MYNN_ADD(dz[k], dz[k - 1]));
        real abk = MYNN_SUB(1.0f, afk);
        qkw[k] = sqrtf(mynn_max2(
            MYNN_ADD(MYNN_MUL(qke[k], abk), MYNN_MUL(qke[k - 1], afk)),
            1.0e-3f));
        qtke[k] = mynn_max2(MYNN_MUL(0.5f, MYNN_MUL(qkw[k], qkw[k])), 0.005f);
        thetaw[k] = MYNN_ADD(MYNN_MUL(theta[k], abk),
                             MYNN_MUL(theta[k - 1], afk));
    }

    real elt = 1.0e-5f, vsc_sum = 1.0e-5f;
    int k = 1;
    while (k < nz && zw[k] <= MYNN_ADD(zi2, h1)) {
        real dzk = MYNN_MUL(0.5f, MYNN_ADD(dz[k], dz[k - 1]));
        real qdz = MYNN_MUL(mynn_min2(mynn_max2(qkw[k], 0.01f), 30.0f), dzk);
        elt = MYNN_ADD(elt, MYNN_MUL(qdz, zw[k]));
        vsc_sum = MYNN_ADD(vsc_sum, qdz);
        ++k;
    }
    elt = mynn_min2(mynn_max2(MYNN_DIV(MYNN_MUL(0.23f, elt), vsc_sum), 8.0f),
                    400.0f);
    real vsc = mynn_powf(MYNN_MUL(MYNN_MUL(gtr, elt), mynn_max2(fltv, 0.0f)),
                         MYNN_DIV(1.0f, 3.0f));
    mynn_boulac_elblavg(zw, dz, qtke, thetaw, dlu, dld, elblavg, nz);

    el[0] = 0.0f;
    for (k = 1; k < nz; ++k) {
        real zwk = zw[k], elb, elf;
        if (dtv[k] > 0.0f) {
            real bv = mynn_max2(sqrtf(MYNN_MUL(gtr, dtv[k])), 0.0001f);
            real numerator = mynn_max2(
                MYNN_MUL(0.3f, mynn_max2(qkw[k], 0.018f)),
                MYNN_MUL(MYNN_MUL(50.0f, edmf_a[k - 1]), edmf_w[k - 1]));
            elb = MYNN_MUL(MYNN_DIV(numerator, bv), MYNN_ADD(1.0f,
                MYNN_MUL(alp3, sqrtf(MYNN_DIV(vsc, MYNN_MUL(bv, elt))))));
            elb = mynn_min2(elb, zwk);
            elf = MYNN_DIV(mynn_max2(qkw[k], 0.018f), bv);
            elblavg[k] = mynn_max2(elblavg[k], MYNN_DIV(
                MYNN_MUL(MYNN_MUL(50.0f, edmf_a[k - 1]), edmf_w[k - 1]), bv));
        } else {
            elb = 1.0e10f;
            elf = elb;
        }
        real els;
        if (rmo > 0.0f) {
            els = MYNN_DIV(MYNN_MUL(0.4f, zwk), MYNN_ADD(1.0f,
                MYNN_MUL(3.5f, mynn_min2(MYNN_MUL(zwk, rmo), 1.0f))));
        } else {
            els = MYNN_MUL(MYNN_MUL(0.4f, zwk), mynn_powf(MYNN_SUB(1.0f,
                MYNN_MUL(MYNN_MUL(5.0f, zwk), rmo)), 0.2f));
        }
        real weight = MYNN_ADD(MYNN_MUL(0.5f, mynn_tanhf(
            MYNN_DIV(MYNN_SUB(zwk, MYNN_ADD(zi2, h1)), h2))), 0.5f);
        real value = sqrtf(MYNN_DIV(MYNN_MUL(els, els), MYNN_ADD(1.0f,
            MYNN_DIV(MYNN_MUL(els, els), MYNN_MUL(elt, elt)))));
        value = mynn_min2(value, elb);
        value = mynn_min2(value, elf);
        value = MYNN_ADD(MYNN_MUL(value, MYNN_SUB(1.0f, weight)),
                         MYNN_MUL(MYNN_MUL(0.3f, elblavg[k]), weight));
        el[k] = MYNN_MUL(value, psig_bl);
    }
#endif
}

extern "C" __global__
void mynn_mixlength_default_columns(
    const real* __restrict__ dz_raw, const real* __restrict__ zw_raw,
    const real* __restrict__ u_raw, const real* __restrict__ v_raw,
    const real* __restrict__ qke_raw, const real* __restrict__ dtv_raw,
    const real* __restrict__ theta_raw, const real* __restrict__ edmf_w_raw,
    const real* __restrict__ edmf_a_raw, const real* __restrict__ rmo,
    const real* __restrict__ fltv, const real* __restrict__ zi,
    const real* __restrict__ psig_bl, real* __restrict__ el_raw,
    real* __restrict__ qkw_raw, real* __restrict__ qtke_raw,
    real* __restrict__ thetaw_raw, real* __restrict__ elblavg_raw,
    real* __restrict__ dlu_raw, real* __restrict__ dld_raw,
    int mixlength, int nz, int ncol
#if defined(MYNN_GSD41)
    , const real* __restrict__ flt, const real* __restrict__ flq,
    const real* __restrict__ vt_raw, const real* __restrict__ vq_raw,
    int unsquared_qtke
#endif
    )
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    MynnColumn<const real> dz = {dz_raw + column, (size_t)ncol};
    MynnColumn<const real> zw = {zw_raw + column, (size_t)ncol};
    MynnColumn<const real> u = {u_raw + column, (size_t)ncol};
    MynnColumn<const real> v = {v_raw + column, (size_t)ncol};
    MynnColumn<const real> qke = {qke_raw + column, (size_t)ncol};
    MynnColumn<const real> dtv = {dtv_raw + column, (size_t)ncol};
    MynnColumn<const real> theta = {theta_raw + column, (size_t)ncol};
    MynnColumn<const real> edmf_w = {edmf_w_raw + column, (size_t)ncol};
    MynnColumn<const real> edmf_a = {edmf_a_raw + column, (size_t)ncol};
    MynnColumn<real> el = {el_raw + column, (size_t)ncol};
    MynnColumn<real> qkw = {qkw_raw + column, (size_t)ncol};
    MynnColumn<real> qtke = {qtke_raw + column, (size_t)ncol};
    MynnColumn<real> thetaw = {thetaw_raw + column, (size_t)ncol};
    MynnColumn<real> elblavg = {elblavg_raw + column, (size_t)ncol};
    MynnColumn<real> dlu = {dlu_raw + column, (size_t)ncol};
    MynnColumn<real> dld = {dld_raw + column, (size_t)ncol};
    int base = 0, zbase = 0;
#if defined(MYNN_GSD41)
    real flux = fltv[column];
    // GSD MYNN v4.1 module_bl_mynn.F:920,1015: both lengths take their
    // buoyancy flux from the lowest level's vt and vq (mass-level column
    // element 0 of the (ncol, nz) Fortran-order arrays).
    real flt_c = flt[column];
    const real tv0 = MYNN_MUL(
        MYNN_SUB(MYNN_DIV(461.6f, 287.0f), 1.0f), 300.0f);
    real vt0 = vt_raw[column], vq0 = vq_raw[column];
    flux = MYNN_ADD(MYNN_MUL(MYNN_ADD(vt0, 1.0f), flt_c),
                    MYNN_MUL(MYNN_ADD(vq0, tv0), flq[column]));
#endif
    mynn_mym_length_column(dz + base, zw + zbase, u + base, v + base,
        qke + base, dtv + base, theta + base, edmf_w + base, edmf_a + base,
        rmo[column],
#if defined(MYNN_GSD41)
        flux,
#else
        fltv[column],
#endif
        zi[column], psig_bl[column], el + base,
        qkw + base, qtke + base, thetaw + base, elblavg + base,
        dlu + base, dld + base, nz, mixlength
        MYNN_GSD41_LENGTH_ARGS(flt_c, unsquared_qtke));
}

// module_bl_mynn.F:1766-1820 mym_level2 for one column.  The Fortran loop runs
// kts+1..kte, so element 0 of every output keeps the caller value; sm and sh
// are WRF dummy arguments and mym_initialize hands their surface element back
// untouched.  vt and vq are mym_initialize locals zeroed at :1552-1556, so the
// two interpolation terms are written against a literal zero here.
__device__ void mynn_mym_level2_column(
    MynnColumn<const real> dz, MynnColumn<const real> u,
    MynnColumn<const real> v, MynnColumn<const real> thl,
    MynnColumn<const real> qw, MynnColumn<real> dtl,
    MynnColumn<real> dqw, MynnColumn<real> dtv, MynnColumn<real> gm,
    MynnColumn<real> gh, MynnColumn<real> sm, MynnColumn<real> sh,
    int nz)
{
    // Constants through the helpers for the reason mynn_level2_pairs records:
    // three steps of g2 are compile-time FP32 ties and ptxas 12.x's folder
    // does not honour round-to-nearest-even on a tie.  The rest of this
    // routine was already written this way; only the constants were not.
    const real pr = 0.74f, b1 = 24.0f, b2 = 15.0f, g1 = 0.235f;
    const real c2 = 0.729f, c3 = 0.340f, c5 = 0.2f;
    const real a1 = MYNN_DIV(
        MYNN_MUL(b1, MYNN_SUB(1.0f, MYNN_MUL(3.0f, g1))), 6.0f);
    const real c1 = MYNN_SUB(g1, MYNN_DIV(1.0f,
        MYNN_MUL(MYNN_MUL(3.0f, a1), 2.88449914061481660f)));
    const real a2 = MYNN_DIV(
        MYNN_MUL(a1, MYNN_SUB(g1, c1)), MYNN_MUL(g1, pr));
    const real g2 = MYNN_ADD(
        MYNN_MUL(MYNN_DIV(b2, b1), MYNN_SUB(1.0f, c3)),
        MYNN_MUL(MYNN_DIV(MYNN_MUL(2.0f, a1), b1),
                 MYNN_SUB(3.0f, MYNN_MUL(2.0f, c2))));
    const real tv0 = MYNN_MUL(
        MYNN_SUB(MYNN_DIV(461.6f, 287.0f), 1.0f), 300.0f);
    const real gtr = MYNN_DIV(9.81f, 300.0f);

    for (int k = 1; k < nz; ++k) {
        real dzk = MYNN_MUL(0.5f, MYNN_ADD(dz[k], dz[k - 1]));
        real afk = MYNN_DIV(dz[k], MYNN_ADD(dz[k], dz[k - 1]));
        real abk = MYNN_SUB(1.0f, afk);
        real du = MYNN_SUB(u[k], u[k - 1]);
        real dv = MYNN_SUB(v[k], v[k - 1]);
        real duz = MYNN_ADD(MYNN_MUL(du, du), MYNN_MUL(dv, dv));
        duz = MYNN_DIV(duz, MYNN_MUL(dzk, dzk));
        real dtz = MYNN_DIV(MYNN_SUB(thl[k], thl[k - 1]), dzk);
        real dqz = MYNN_DIV(MYNN_SUB(qw[k], qw[k - 1]), dzk);
        real vtt = MYNN_ADD(MYNN_ADD(1.0f, MYNN_MUL(0.0f, abk)),
                            MYNN_MUL(0.0f, afk));
        real vqq = MYNN_ADD(MYNN_ADD(tv0, MYNN_MUL(0.0f, abk)),
                            MYNN_MUL(0.0f, afk));
        real dtq = MYNN_ADD(MYNN_MUL(vtt, dtz), MYNN_MUL(vqq, dqz));
        real level_gh = -MYNN_MUL(dtq, gtr);
        real ri = MYNN_DIV(-level_gh, mynn_max2(duz, 1.0e-10f));
        real a2fac = MYNN_DIV(1.0f, MYNN_ADD(1.0f, mynn_max2(ri, 0.0f)));
#if defined(MYNN_GSD41)
        // GSD MYNN v4.1 mym_level2 divides by a2den = 1 + max(ri, 0):
        // a2/a2den rounds in FP32 where v4.6.1 multiplies by a2fac.
        real a2r = MYNN_DIV(a2, MYNN_ADD(1.0f, mynn_max2(ri, 0.0f)));
        (void)a2fac;
#define MYNN_L2_F1_A2 MYNN_MUL(3.0f, a2r)
#define MYNN_L2_A2 a2r
#else
#define MYNN_L2_F1_A2 MYNN_MUL(MYNN_MUL(3.0f, a2), a2fac)
#define MYNN_L2_A2 MYNN_MUL(a2, a2fac)
#endif

        // a2fac is a runtime value, so f1, rf1, smc and shc are runtime
        // expressions.  a1, c1, a2 and g2 are `asm` results, so every
        // expression built from them is a runtime expression too -- none of
        // rfc, f2, rf2 or the outer f1 terms folds at compile time, and the
        // only literal-only subexpressions left (1-c2, 1-c5, 3-2*c2) are
        // exactly the compile-time FP32 ties the constants block above
        // records.  All of them go through the intrinsics, so this routine
        // and mynn_level2_pairs are now the same expression tree.
        real rfc = MYNN_DIV(g1, MYNN_ADD(g1, g2));
        real f1 = MYNN_ADD(
            MYNN_ADD(MYNN_MUL(b1, MYNN_SUB(g1, c1)),
                     MYNN_MUL(MYNN_MUL(MYNN_L2_F1_A2,
                                       MYNN_SUB(1.0f, c2)),
                              MYNN_SUB(1.0f, c5))),
            MYNN_MUL(MYNN_MUL(2.0f, a1), MYNN_SUB(3.0f, MYNN_MUL(2.0f, c2))));
        real f2 = MYNN_SUB(MYNN_MUL(b1, MYNN_ADD(g1, g2)),
                           MYNN_MUL(MYNN_MUL(3.0f, a1), MYNN_SUB(1.0f, c2)));
        real rf1 = MYNN_DIV(MYNN_MUL(b1, MYNN_SUB(g1, c1)), f1);
        real rf2 = MYNN_DIV(MYNN_MUL(b1, g1), f2);
        real smc = MYNN_DIV(
            MYNN_MUL(MYNN_DIV(a1, MYNN_L2_A2), f1), f2);
        real shc = MYNN_MUL(MYNN_MUL(3.0f, MYNN_L2_A2),
                            MYNN_ADD(g1, g2));
#undef MYNN_L2_F1_A2
#undef MYNN_L2_A2
        real ri1 = MYNN_DIV(0.5f, smc);
        real ri2 = MYNN_MUL(rf1, smc);
        real ri3 = MYNN_SUB(MYNN_MUL(MYNN_MUL(4.0f, rf2), smc),
                            MYNN_MUL(2.0f, ri2));
        real ri4 = MYNN_MUL(ri2, ri2);
        real radical = MYNN_ADD(
            MYNN_SUB(MYNN_MUL(ri, ri), MYNN_MUL(ri3, ri)), ri4);
        real rf = mynn_min2(MYNN_MUL(ri1,
            MYNN_SUB(MYNN_ADD(ri, ri2), sqrtf(radical))), rfc);

        dtl[k] = dtz;
        dqw[k] = dqz;
        dtv[k] = dtq;
        gm[k] = duz;
        gh[k] = level_gh;
        sh[k] = MYNN_DIV(MYNN_MUL(shc, MYNN_SUB(rfc, rf)),
                         MYNN_SUB(1.0f, rf));
        sm[k] = MYNN_MUL(MYNN_DIV(MYNN_MUL(smc, MYNN_SUB(rf1, rf)),
                                  MYNN_SUB(rf2, rf)), sh[k]);
    }
}

#define MYNN_INITIALIZE_SCRATCH 15

extern "C" __global__
void mynn_initialize_default_columns(
    const real* __restrict__ dz_raw, const real* __restrict__ zw_raw,
    const real* __restrict__ u_raw, const real* __restrict__ v_raw,
    const real* __restrict__ thl_raw, const real* __restrict__ qw_raw,
    const real* __restrict__ theta_raw, const real* __restrict__ edmf_w_raw,
    const real* __restrict__ edmf_a_raw, const real* __restrict__ sm_in_raw,
    const real* __restrict__ sh_in_raw, const real* __restrict__ qke_in_raw,
    const real* __restrict__ rmo, const real* __restrict__ ust,
    const real* __restrict__ zi, const real* __restrict__ psig_bl,
    real* __restrict__ el_o_raw, real* __restrict__ qke_o_raw,
    real* __restrict__ tsq_o_raw, real* __restrict__ qsq_o_raw,
    real* __restrict__ cov_o_raw, real* __restrict__ sm_o_raw,
    real* __restrict__ sh_o_raw, real* __restrict__ scratch_raw,
    int initialize_qke, int mixlength, int nz, int ncol
#if defined(MYNN_GSD41)
    , int unsquared_qtke
#endif
    )
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    MynnColumn<const real> dz = {dz_raw + column, (size_t)ncol};
    MynnColumn<const real> zw = {zw_raw + column, (size_t)ncol};
    MynnColumn<const real> u = {u_raw + column, (size_t)ncol};
    MynnColumn<const real> v = {v_raw + column, (size_t)ncol};
    MynnColumn<const real> thl = {thl_raw + column, (size_t)ncol};
    MynnColumn<const real> qw = {qw_raw + column, (size_t)ncol};
    MynnColumn<const real> theta = {theta_raw + column, (size_t)ncol};
    MynnColumn<const real> edmf_w = {edmf_w_raw + column, (size_t)ncol};
    MynnColumn<const real> edmf_a = {edmf_a_raw + column, (size_t)ncol};
    MynnColumn<const real> sm_in = {sm_in_raw + column, (size_t)ncol};
    MynnColumn<const real> sh_in = {sh_in_raw + column, (size_t)ncol};
    MynnColumn<const real> qke_in = {qke_in_raw + column, (size_t)ncol};
    MynnColumn<real> el_o = {el_o_raw + column, (size_t)ncol};
    MynnColumn<real> qke_o = {qke_o_raw + column, (size_t)ncol};
    MynnColumn<real> tsq_o = {tsq_o_raw + column, (size_t)ncol};
    MynnColumn<real> qsq_o = {qsq_o_raw + column, (size_t)ncol};
    MynnColumn<real> cov_o = {cov_o_raw + column, (size_t)ncol};
    MynnColumn<real> sm_o = {sm_o_raw + column, (size_t)ncol};
    MynnColumn<real> sh_o = {sh_o_raw + column, (size_t)ncol};
    MynnColumn<real> scratch = {scratch_raw + column, (size_t)ncol};
    size_t base = 0;
    size_t zbase = 0;
    MynnColumn<real> work = scratch + 0;
    MynnColumn<real> qkw = work;
    MynnColumn<real> qtke = work + nz;
    MynnColumn<real> thetaw = work + 2 * nz;
    MynnColumn<real> elblavg = work + 3 * nz;
    MynnColumn<real> dlu = work + 4 * nz;
    MynnColumn<real> dld = work + 5 * nz;
    MynnColumn<real> dtl = work + 6 * nz;
    MynnColumn<real> dqw = work + 7 * nz;
    MynnColumn<real> dtv = work + 8 * nz;
    MynnColumn<real> gm = work + 9 * nz;
    MynnColumn<real> gh = work + 10 * nz;
    MynnColumn<real> pdk = work + 11 * nz;
    MynnColumn<real> pdt = work + 12 * nz;
    MynnColumn<real> pdq = work + 13 * nz;
    MynnColumn<real> pdc = work + 14 * nz;

    // module_bl_mynn.F:279-280, :306 and module_bl_mynn_common.F:68-69.
#if defined(MYNN_GSD41)
    // Fork module_bl_mynn.F:203 and :546 initialization production floor.
    const real b1 = 24.0f, b2 = 15.0f, karman = 0.4f, qkemin = 1.0e-12f;
#else
    const real b1 = 24.0f, b2 = 15.0f, karman = 0.4f, qkemin = 1.0e-3f;
#endif
    const real onethird = MYNN_DIV(1.0f, 3.0f);
    const real twothirds = MYNN_DIV(2.0f, 3.0f);
    // pmz and phh are mym_initialize locals initialised to 1 at :1545.
    const real b1_pmz = MYNN_MUL(b1, 1.0f);
    const real phh_b2 = MYNN_MUL(1.0f, b2);

    real ustc = ust[column];
    real rmoc = rmo[column];
    real zic = zi[column];
    real psig = psig_bl[column];

    MynnColumn<real> sm = sm_o + base;
    MynnColumn<real> sh = sh_o + base;
    MynnColumn<real> qke = qke_o + base;
    MynnColumn<real> el = el_o + base;
    MynnColumn<real> tsq = tsq_o + base;
    MynnColumn<real> qsq = qsq_o + base;
    MynnColumn<real> cov = cov_o + base;
    for (int k = 0; k < nz; ++k) {
        sm[k] = sm_in[base + k];
        sh[k] = sh_in[base + k];
        qke[k] = qke_in[base + k];
        el[k] = 0.0f;
        tsq[k] = 0.0f;
        qsq[k] = 0.0f;
        cov[k] = 0.0f;
        dtl[k] = 0.0f; dqw[k] = 0.0f; dtv[k] = 0.0f;
        gm[k] = 0.0f; gh[k] = 0.0f;
        pdk[k] = 0.0f; pdt[k] = 0.0f; pdq[k] = 0.0f; pdc[k] = 0.0f;
    }

    mynn_mym_level2_column(dz + base, u + base, v + base, thl + base,
                           qw + base, dtl, dqw, dtv, gm, gh, sm, sh, nz);

    if (initialize_qke) {
        qke[0] = MYNN_MUL(MYNN_MUL(1.5f, MYNN_MUL(ustc, ustc)),
                          mynn_powf(b1_pmz, twothirds));
        for (int k = 1; k < nz; ++k) {
            real taper = MYNN_DIV(
                MYNN_SUB(MYNN_MUL(ustc, 700.0f), zw[zbase + k]),
                MYNN_MUL(mynn_max2(ustc, 0.01f), 700.0f));
            qke[k] = MYNN_MUL(qke[0], mynn_max2(taper, 0.01f));
        }
    }
    // flt and flq are zero mym_initialize locals (:1546), so the surface seeds
    // of tsq, qsq and cov collapse; the terms are kept as the Fortran wrote
    // them so a nonzero flux lane can be added without a rewrite.
    real phm = MYNN_DIV(phh_b2, mynn_powf(b1_pmz, onethird));
    tsq[0] = MYNN_MUL(phm, MYNN_MUL(MYNN_DIV(0.0f, ustc),
                                    MYNN_DIV(0.0f, ustc)));
    qsq[0] = MYNN_MUL(phm, MYNN_MUL(MYNN_DIV(0.0f, ustc),
                                    MYNN_DIV(0.0f, ustc)));
    cov[0] = MYNN_MUL(MYNN_MUL(phm, MYNN_DIV(0.0f, ustc)),
                      MYNN_DIV(0.0f, ustc));
    for (int k = 1; k < nz; ++k) {
        real vkz = MYNN_MUL(karman, zw[zbase + k]);
        el[k] = MYNN_DIV(vkz, MYNN_ADD(1.0f, MYNN_DIV(vkz, 100.0f)));
    }

    // module_bl_mynn.F:1595 fixes lmax = 5.
    for (int iteration = 0; iteration < 5; ++iteration) {
        mynn_mym_length_column(dz + base, zw + zbase, u + base, v + base,
                               qke, dtv, theta + base, edmf_w + base,
                               edmf_a + base, rmoc, 0.0f, zic, psig,
                               el, qkw, qtke, thetaw, elblavg, dlu, dld, nz,
                               mixlength
                               MYNN_GSD41_LENGTH_ARGS(0.0f, unsquared_qtke));
        for (int k = 1; k < nz; ++k) {
            real elq = MYNN_MUL(el[k], qkw[k]);
            pdk[k] = MYNN_MUL(elq, MYNN_ADD(MYNN_MUL(sm[k], gm[k]),
                                            MYNN_MUL(sh[k], gh[k])));
            pdt[k] = MYNN_MUL(MYNN_MUL(elq, sh[k]), MYNN_MUL(dtl[k], dtl[k]));
            pdq[k] = MYNN_MUL(MYNN_MUL(elq, sh[k]), MYNN_MUL(dqw[k], dqw[k]));
            pdc[k] = MYNN_MUL(MYNN_MUL(MYNN_MUL(elq, sh[k]), dtl[k]), dqw[k]);
        }

        real vkz = MYNN_MUL(MYNN_MUL(karman, 0.5f), dz[base]);
        real elv = MYNN_DIV(MYNN_MUL(0.5f, MYNN_ADD(el[1], el[0])), vkz);
        if (initialize_qke) {
            real ust_floor = mynn_max2(ustc, 0.02f);
            qke[0] = MYNN_MUL(
                MYNN_MUL(1.0f, MYNN_MUL(ust_floor, ust_floor)),
                mynn_powf(MYNN_MUL(b1_pmz, elv), twothirds));
        }
        phm = MYNN_DIV(phh_b2, mynn_powf(
            MYNN_DIV(b1_pmz, MYNN_MUL(elv, elv)), onethird));
        tsq[0] = MYNN_MUL(phm, MYNN_MUL(MYNN_DIV(0.0f, ustc),
                                        MYNN_DIV(0.0f, ustc)));
        qsq[0] = MYNN_MUL(phm, MYNN_MUL(MYNN_DIV(0.0f, ustc),
                                        MYNN_DIV(0.0f, ustc)));
        cov[0] = MYNN_MUL(MYNN_MUL(phm, MYNN_DIV(0.0f, ustc)),
                          MYNN_DIV(0.0f, ustc));

        for (int k = 1; k < nz - 1; ++k) {
            real b1l = MYNN_MUL(MYNN_MUL(b1, 0.25f),
                                MYNN_ADD(el[k + 1], el[k]));
            real tmpq = mynn_min2(mynn_max2(
                MYNN_MUL(b1l, MYNN_ADD(pdk[k + 1], pdk[k])), qkemin), 125.0f);
            if (initialize_qke) qke[k] = mynn_powf(tmpq, twothirds);
            real b2l;
            if (qke[k] <= 0.0f) b2l = 0.0f;
            else b2l = MYNN_DIV(MYNN_MUL(b2, MYNN_DIV(b1l, b1)),
                                sqrtf(qke[k]));
            tsq[k] = MYNN_MUL(b2l, MYNN_ADD(pdt[k + 1], pdt[k]));
            qsq[k] = MYNN_MUL(b2l, MYNN_ADD(pdq[k + 1], pdq[k]));
            cov[k] = MYNN_MUL(b2l, MYNN_ADD(pdc[k + 1], pdc[k]));
        }
    }

    if (initialize_qke) {
        qke[0] = MYNN_MUL(0.5f, MYNN_ADD(qke[0], qke[1]));
        qke[nz - 1] = qke[nz - 2];
    }
    tsq[nz - 1] = tsq[nz - 2];
    qsq[nz - 1] = qsq[nz - 2];
    cov[nz - 1] = cov[nz - 2];
}

// ===========================================================================
// module_bl_mynn.F:5679-6823 DMP_mf with bl_mynn_edmf_mom=1,
// bl_mynn_edmf_tke=0, bl_mynn_mixscalars=0, mix_chem=.false. and spp_pbl=0.
// env_subs=.false. (module_bl_mynn.F:336) makes the whole subsidence
// and dynamic-detrainment block unreachable, so sub_*, det_*, the envm_*
// profiles, the plume-overshoot Froude limiter and the Asai-Kasahara
// detrainment rates are absent here; bl_mynn_edmf_dd=0 (:330) means no
// downdraft contributes.  One thread owns one column: each plume is a
// sequential upward integration and the flux limiter rescales the column.
//
// Round-to-nearest intrinsics everywhere, for the reason the tendency kernel
// documents.  atanf, expf and powf go through FP64 and round once, which is
// what glibc agrees with on every argument these expressions reach; tanhf
// cannot be approximated that way and uses the fdlibm transcription above.
// ===========================================================================
#if defined(MYNN_GSD41)
#define MYNN_DMP_NUP 10
#else
#define MYNN_DMP_NUP 8
#endif

// Contraction-pinned forms of the phase-blend helpers.  The plain-operator
// versions above are Horner polynomials, which is the textbook a*b+c pattern:
// NVRTC contracts them, and inside the plume saturation loop that moved
// s_awqc by 112379 ULP.  These are the same expressions with the roundings
// made explicit; the originals are left alone because the tendency kernel is
// already validated against them.
__device__ __forceinline__ real mynn_esat_liquid_rn(real xc)
{
    const real c[9] = {
        0.611583699e3f, 0.444606896e2f, 0.143177157e1f, 0.264224321e-1f,
        0.299291081e-3f, 0.203154182e-5f, 0.702620698e-8f, 0.379534310e-11f,
        -0.321582393e-13f,
    };
    real value = c[8];
    for (int i = 7; i >= 0; --i)
        value = MYNN_ADD(c[i], MYNN_MUL(xc, value));
    return value;
}

__device__ __forceinline__ real mynn_esat_ice_rn(real xc)
{
    const real c[9] = {
        0.609868993e3f, 0.499320233e2f, 0.184672631e1f, 0.402737184e-1f,
        0.565392987e-3f, 0.521693933e-5f, 0.307839583e-7f, 0.105785160e-9f,
        0.161444444e-12f,
    };
    real value = c[8];
    for (int i = 7; i >= 0; --i)
        value = MYNN_ADD(c[i], MYNN_MUL(xc, value));
    return value;
}

__device__ __forceinline__ real mynn_qsat_blend_rn(real t, real p)
{
    const real t0c = 273.15f, tice = 240.0f, t0cm6 = 273.15f - 6.0f;
    real xc = mynn_max2(-80.0f, MYNN_SUB(t, t0c));
    real ceiling = MYNN_MUL(p, 0.15f);
    if (t >= t0cm6) {
        real esl = mynn_min2(mynn_esat_liquid_rn(xc), ceiling);
        return MYNN_DIV(MYNN_MUL(0.622f, esl),
                        mynn_max2(MYNN_SUB(p, esl), 1.0e-5f));
    }
    if (t <= tice) {
        real esi = mynn_min2(mynn_esat_ice_rn(xc), ceiling);
        return MYNN_DIV(MYNN_MUL(0.622f, esi),
                        mynn_max2(MYNN_SUB(p, esi), 1.0e-5f));
    }
    real esl = mynn_min2(mynn_esat_liquid_rn(xc), ceiling);
    real esi = mynn_min2(mynn_esat_ice_rn(xc), ceiling);
    real rslf = MYNN_DIV(MYNN_MUL(0.622f, esl),
                         mynn_max2(MYNN_SUB(p, esl), 1.0e-5f));
    real rsif = MYNN_DIV(MYNN_MUL(0.622f, esi),
                         mynn_max2(MYNN_SUB(p, esi), 1.0e-5f));
    real chi = MYNN_DIV(MYNN_SUB(t0cm6, t), MYNN_SUB(t0cm6, tice));
    return MYNN_ADD(MYNN_MUL(MYNN_SUB(1.0f, chi), rslf), MYNN_MUL(chi, rsif));
}

__device__ __forceinline__ real mynn_xl_blend_rn(real t)
{
    const real t0c = 273.15f, tice = 240.0f;
    const real cpv = MYNN_MUL(4.0f, 461.6f);
    const real cpv_cliq = MYNN_SUB(cpv, 4190.0f);
    const real cpv_cice = MYNN_SUB(cpv, 2106.0f);
    if (t >= t0c)
        return MYNN_ADD(2.5e6f, MYNN_MUL(cpv_cliq, MYNN_SUB(t, t0c)));
    if (t <= tice)
        return MYNN_ADD(2.85e6f, MYNN_MUL(cpv_cice, MYNN_SUB(t, t0c)));
    real xlvt = MYNN_ADD(2.5e6f, MYNN_MUL(cpv_cliq, MYNN_SUB(t, t0c)));
    real xlst = MYNN_ADD(2.85e6f, MYNN_MUL(cpv_cice, MYNN_SUB(t, t0c)));
    real chi = MYNN_DIV(MYNN_SUB(t0c, t), MYNN_SUB(t0c, tice));
    return MYNN_ADD(MYNN_MUL(MYNN_SUB(1.0f, chi), xlvt), MYNN_MUL(chi, xlst));
}

// glibc 2.39 atanf and expf, the functions a real-argument ATAN and EXP link
// to; the bodies rounded an FP64 evaluation before, which is neither.
__device__ __forceinline__ real mynn_atanf(real x)
{
    return mynn_glibc_atanf(x);
}

__device__ __forceinline__ real mynn_expf_rn(real x)
{
    return mynn_glibc_expf(x);
}

#if defined(MYNN_GSD41)
// ===========================================================================
// GSD MYNN v4.1 (NOAA-EMC WRF 3.9 branch, tag v4.1.21, public domain WRF)
// phys/module_bl_mynn.F.  Phase-blended saturation and latent heat
// (:6064-6162): liquid at and above 273.16 K, ice at and below 253 K,
// linear between over 20.16 K, polynomial argument T - 273.16, no ceiling
// on the vapour pressure.  Every operator is one rounded instruction.
// ===========================================================================
__device__ __forceinline__ real mynn_gsd41_poly(real xc, const real* c)
{
    real v = MYNN_ADD(c[7], MYNN_MUL(xc, c[8]));
    for (int i = 6; i >= 0; --i) v = MYNN_ADD(c[i], MYNN_MUL(xc, v));
    return v;
}

__device__ __forceinline__ real mynn_gsd41_esl(real xc)
{
    const real j[9] = {.611583699E03f, .444606896E02f, .143177157E01f,
        .264224321E-1f, .299291081E-3f, .203154182E-5f, .702620698E-8f,
        .379534310E-11f, -.321582393E-13f};
    return mynn_gsd41_poly(xc, j);
}

__device__ __forceinline__ real mynn_gsd41_esi(real xc)
{
    const real kk[9] = {.609868993E03f, .499320233E02f, .184672631E01f,
        .402737184E-1f, .565392987E-3f, .521693933E-5f, .307839583E-7f,
        .105785160E-9f, .161444444E-12f};
    return mynn_gsd41_poly(xc, kk);
}

// :6064-6083 esat_blend
__device__ real mynn_gsd41_esat_blend(real t)
{
    real xc = mynn_max2(-80.0f, MYNN_SUB(t, 273.16f));
    if (t >= 273.16f) return mynn_gsd41_esl(xc);
    if (t <= 253.0f) return mynn_gsd41_esi(xc);
    real chi = MYNN_DIV(MYNN_SUB(273.16f, t), 20.16f);
    return MYNN_ADD(MYNN_MUL(MYNN_SUB(1.0f, chi), mynn_gsd41_esl(xc)),
                    MYNN_MUL(chi, mynn_gsd41_esi(xc)));
}

// :6086-6118 qsat_blend (the blended branch; waterice is never passed)
__device__ real mynn_gsd41_qsat_blend(real t, real pr)
{
    real xc = mynn_max2(-80.0f, MYNN_SUB(t, 273.16f));
    if (t >= 273.16f) {
        real esl = mynn_gsd41_esl(xc);
        return MYNN_DIV(MYNN_MUL(0.622f, esl), MYNN_SUB(pr, esl));
    }
    if (t <= 253.0f) {
        real esi = mynn_gsd41_esi(xc);
        return MYNN_DIV(MYNN_MUL(0.622f, esi), MYNN_SUB(pr, esi));
    }
    real esl = mynn_gsd41_esl(xc);
    real esi = mynn_gsd41_esi(xc);
    real rslf = MYNN_DIV(MYNN_MUL(0.622f, esl), MYNN_SUB(pr, esl));
    real rsif = MYNN_DIV(MYNN_MUL(0.622f, esi), MYNN_SUB(pr, esi));
    real chi = MYNN_DIV(MYNN_SUB(273.16f, t), 20.16f);
    return MYNN_ADD(MYNN_MUL(MYNN_SUB(1.0f, chi), rslf), MYNN_MUL(chi, rsif));
}

// :6121-6141 xl_blend.  cpv = 4*r_v, cliq 4190, cice 2106, xlv 2.5e6,
// xls 2.85e6 (share/module_model_constants.F of the same branch).
__device__ real mynn_gsd41_xl_blend(real t)
{
    const real cpv = MYNN_MUL(4.0f, 461.6f);
    const real cpv_cliq = MYNN_SUB(cpv, 4190.0f);
    const real cpv_cice = MYNN_SUB(cpv, 2106.0f);
    real dt = MYNN_SUB(t, 273.16f);
    if (t >= 273.16f) return MYNN_ADD(2.5e6f, MYNN_MUL(cpv_cliq, dt));
    if (t <= 253.0f) return MYNN_ADD(2.85e6f, MYNN_MUL(cpv_cice, dt));
    real xlvt = MYNN_ADD(2.5e6f, MYNN_MUL(cpv_cliq, dt));
    real xlst = MYNN_ADD(2.85e6f, MYNN_MUL(cpv_cice, dt));
    real chi = MYNN_DIV(MYNN_SUB(273.16f, t), 20.16f);
    return MYNN_ADD(MYNN_MUL(MYNN_SUB(1.0f, chi), xlvt), MYNN_MUL(chi, xlst));
}

// :2293-2737 mym_condensation, bl_mynn_cloudpdf = 2 (Chaboureau and
// Bechtold 2002) with spp_pbl = 0.  qc_bl is the in-cloud condensate (the
// radiation merge multiplies it by cldfra_bl, module_radiation_driver.F of
// the same branch :1289-1299); the generation has no qi_bl, written zero.
// tsq, qsq, cov, rmo and zw below k_tropo never reach a CASE(2) output.
extern "C" __global__
void mynn_condensation_gsd41_columns(
    const real* __restrict__ dz_raw, const real* __restrict__ zw_raw,
    const real* __restrict__ th_raw, const real* __restrict__ thl_raw,
    const real* __restrict__ qw_raw, const real* __restrict__ p_raw,
    const real* __restrict__ exner_raw, const real* __restrict__ el_raw,
    const real* __restrict__ dx, const real* __restrict__ pblh,
    const real* __restrict__ hfx, const real* __restrict__ sgm_in_raw,
    real* __restrict__ qc_bl_raw, real* __restrict__ qi_bl_raw,
    real* __restrict__ cldfra_raw, real* __restrict__ vt_raw,
    real* __restrict__ vq_raw, real* __restrict__ sgm_raw,
    real* __restrict__ work_raw, int nz, int ncol)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    MynnColumn<const real> dz = {dz_raw + column, (size_t)ncol};
    MynnColumn<const real> zw = {zw_raw + column, (size_t)ncol};
    MynnColumn<const real> th = {th_raw + column, (size_t)ncol};
    MynnColumn<const real> thl = {thl_raw + column, (size_t)ncol};
    MynnColumn<const real> qw = {qw_raw + column, (size_t)ncol};
    MynnColumn<const real> p = {p_raw + column, (size_t)ncol};
    MynnColumn<const real> exner = {exner_raw + column, (size_t)ncol};
    MynnColumn<const real> el = {el_raw + column, (size_t)ncol};
    MynnColumn<const real> sgm_in = {sgm_in_raw + column, (size_t)ncol};
    MynnColumn<real> qc_bl = {qc_bl_raw + column, (size_t)ncol};
    MynnColumn<real> qi_bl = {qi_bl_raw + column, (size_t)ncol};
    MynnColumn<real> cldfra = {cldfra_raw + column, (size_t)ncol};
    MynnColumn<real> vt = {vt_raw + column, (size_t)ncol};
    MynnColumn<real> vq = {vq_raw + column, (size_t)ncol};
    MynnColumn<real> sgm = {sgm_raw + column, (size_t)ncol};
    // Five column work vectors: q1, rh, a, b, cld.
    MynnColumn<real> work = {work_raw + column, (size_t)ncol};
    MynnColumn<real> q1 = work, rh = work + nz, aa = work + 2 * nz;
    MynnColumn<real> bv = work + 3 * nz, cld = work + 4 * nz;

    const real r_d = 287.0f, r_v = 461.6f, g = 9.81f;
    const real cp = MYNN_DIV(MYNN_MUL(7.0f, r_d), 2.0f);
    const real cpv = MYNN_MUL(4.0f, r_v);
    const real rcp = MYNN_DIV(r_d, cp);
    const real ep_2 = MYNN_DIV(r_d, r_v);
    const real ep_3 = MYNN_SUB(1.0f, ep_2);
    const real p608 = MYNN_SUB(MYNN_DIV(r_v, r_d), 1.0f);
    const real tv0 = MYNN_MUL(p608, 300.0f);
    const real xlvcp = MYNN_DIV(2.5e6f, cp);
    (void)xlvcp;

    // :2353-2365 tropopause estimate; falling through gives kts+2.
    int found = 0;
    for (int level = nz - 3; level >= 1; --level) {
        real theta1 = th[level - 1], theta2 = th[level + 1];
        real ht1 = MYNN_MUL(44307.692f, MYNN_SUB(1.0f, mynn_powf(
            MYNN_DIV(p[level - 1], 101325.0f), 0.190f)));
        real ht2 = MYNN_MUL(44307.692f, MYNN_SUB(1.0f, mynn_powf(
            MYNN_DIV(p[level + 1], 101325.0f), 0.190f)));
        if (MYNN_DIV(MYNN_SUB(theta2, theta1), MYNN_SUB(ht2, ht1))
                < MYNN_DIV(10.0f, 1500.0f)
            && ht1 < 19000.0f && ht1 > 4000.0f) {
            found = level;
            break;
        }
    }
    int k_tropo = max(3, found + 2);  // one-based

    // :2455-2545 first CASE(2) loop.
    real zagl = 0.0f;
    real lfac = mynn_min2(MYNN_ADD(4.25f, MYNN_DIV(dx[column], 4000.0f)),
                          6.0f);
    real hfx1 = hfx[column], pblh1 = pblh[column];
    for (int k = 0; k < nz - 1; ++k) {
        int km = k > 0 ? k - 1 : 0;
        real t = MYNN_MUL(th[k], exner[k]);
        real esat = mynn_gsd41_esat_blend(t);
        real qsl = MYNN_DIV(MYNN_MUL(ep_2, esat), mynn_max2(1.0e-4f,
            MYNN_SUB(p[k], MYNN_MUL(ep_3, esat))));
        rh[k] = mynn_max2(mynn_min2(1.0f, MYNN_DIV(qw[k],
            mynn_max2(1.0e-8f, qsl))), 0.001f);
        real xl = mynn_gsd41_xl_blend(t);
        real exn_k = mynn_powf(MYNN_DIV(p[k], 100000.0f), rcp);
        real tlk = MYNN_MUL(thl[k], exn_k);
        real qsat_tl = mynn_gsd41_qsat_blend(tlk, p[k]);
        real rsl = MYNN_DIV(MYNN_MUL(xl, qsat_tl),
                            MYNN_MUL(r_v, MYNN_MUL(tlk, tlk)));
        real cpm = MYNN_ADD(cp, MYNN_MUL(qw[k], cpv));
        real a = MYNN_DIV(1.0f, MYNN_ADD(1.0f,
                                         MYNN_DIV(MYNN_MUL(xl, rsl), cpm)));
        aa[k] = a;
        real qmq = MYNN_MUL(a, MYNN_SUB(qw[k], qsat_tl));
        real b = MYNN_MUL(a, rsl);
        bv[k] = b;
        real tl_up = MYNN_MUL(thl[k + 1], mynn_powf(
            MYNN_DIV(p[k + 1], 100000.0f), rcp));
        real tl_dn = MYNN_MUL(thl[km], mynn_powf(
            MYNN_DIV(p[km], 100000.0f), rcp));
        real dtl = MYNN_SUB(MYNN_MUL(0.5f, MYNN_ADD(tl_up, tlk)),
                            MYNN_MUL(0.5f, MYNN_ADD(tlk, tl_dn)));
        real dqw = MYNN_SUB(MYNN_MUL(0.5f, MYNN_ADD(qw[k + 1], qw[k])),
                            MYNN_MUL(0.5f, MYNN_ADD(qw[k], qw[km])));
        real dzk = k == 0 ? MYNN_MUL(0.5f, dz[k])
                          : MYNN_MUL(0.5f, MYNN_ADD(dz[k], dz[k - 1]));
        real cdhdz = MYNN_ADD(MYNN_DIV(dtl, dzk), MYNN_MUL(MYNN_DIV(g, cpm),
                                                   MYNN_ADD(1.0f, qw[k])));
        zagl = MYNN_ADD(zagl, dz[k]);
        real els = zagl;
        real ls_min = MYNN_ADD(300.0f, mynn_min2(MYNN_MUL(3.0f,
            mynn_max2(hfx1, 0.0f)), 300.0f));
        ls_min = mynn_min2(mynn_max2(els, 25.0f), ls_min);
        if (zagl > MYNN_ADD(pblh1, 2000.0f))
            ls_min = mynn_max2(MYNN_ADD(ls_min, MYNN_MUL(0.5f,
                MYNN_SUB(MYNN_ADD(pblh1, 2000.0f), zagl))), 300.0f);
        real ls = mynn_max2(mynn_min2(MYNN_MUL(lfac, el[k]), 600.0f), ls_min);
        real t1 = MYNN_DIV(MYNN_MUL(a, dqw), dzk);
        real term = MYNN_ADD(MYNN_SUB(MYNN_MUL(t1, t1),
            MYNN_DIV(MYNN_MUL(MYNN_MUL(MYNN_MUL(MYNN_MUL(2.0f, a), b), cdhdz),
                              dqw), dzk)),
            MYNN_MUL(MYNN_MUL(b, b), MYNN_MUL(cdhdz, cdhdz)));
        real sg = mynn_max2(1.0e-10f, MYNN_MUL(MYNN_MUL(0.225f, ls),
                                               sqrtf(mynn_max2(0.0f, term))));
        sgm[k] = sg;
        q1[k] = MYNN_DIV(qmq, sg);
        cld[k] = mynn_max2(0.0f, mynn_min2(1.0f, MYNN_ADD(0.5f,
            MYNN_MUL(0.36f, mynn_atanf(MYNN_MUL(1.55f, q1[k]))))));
    }

    // :2547-2552, :2620-2713 second CASE(2) loop.
    zagl = 0.0f;
    real pblh2 = mynn_max2(10.0f, pblh1);
    const real exp_m1 = 0.36787944117144233f;
    for (int k = 0; k < nz - 1; ++k) {
        real t = MYNN_MUL(th[k], exner[k]);
        real q1k = q1[k];
        zagl = MYNN_ADD(zagl, dz[k]);
        real ql;
        if (q1k < 0.0f)
            ql = MYNN_MUL(sgm[k], mynn_expf_rn(MYNN_SUB(MYNN_MUL(1.2f, q1k),
                                                        1.0f)));
        else if (q1k > 2.0f)
            ql = MYNN_MUL(sgm[k], q1k);
        else
            ql = MYNN_MUL(sgm[k], MYNN_ADD(MYNN_ADD(exp_m1,
                MYNN_MUL(0.66f, q1k)), MYNN_MUL(0.086f, MYNN_MUL(q1k, q1k))));
        real cldk = cld[k];
        if (k + 1 >= k_tropo - 1) {
            cldk = 0.0f;
            ql = 0.0f;
        }
        real q1c = mynn_max2(q1[k], -5.0f);
        real fng;
        if (q1c >= 1.0f) fng = 1.0f;
        else if (q1c >= -1.7f)
            fng = mynn_expf_rn(MYNN_MUL(-0.4f, MYNN_SUB(q1c, 1.0f)));
        else if (q1c >= -2.5f)
            fng = MYNN_ADD(3.0f, mynn_expf_rn(MYNN_MUL(-3.8f,
                                                       MYNN_ADD(q1c, 1.7f))));
        else
            fng = mynn_min2(MYNN_ADD(23.9f, mynn_expf_rn(MYNN_MUL(-1.6f,
                MYNN_ADD(q1c, 2.5f)))), 60.0f);
        fng = mynn_min2(fng, 20.0f);
        real xl = mynn_gsd41_xl_blend(t);
        real bb = MYNN_DIV(MYNN_MUL(bv[k], t), th[k]);
        real qww = MYNN_ADD(1.0f, MYNN_MUL(0.61f, qw[k]));
        real alpha = MYNN_MUL(0.61f, th[k]);
        real beta = MYNN_SUB(MYNN_MUL(MYNN_DIV(th[k], t), MYNN_DIV(xl, cp)),
                             MYNN_MUL(1.61f, th[k]));
        real cw = mynn_min2(cldk, 0.99f);
        vt[k] = MYNN_SUB(MYNN_SUB(qww, MYNN_MUL(MYNN_MUL(MYNN_MUL(cw, beta),
                                                         bb), fng)), 1.0f);
        vq[k] = MYNN_SUB(MYNN_ADD(alpha, MYNN_MUL(MYNN_MUL(MYNN_MUL(cw, beta),
                                                           aa[k]), fng)), tv0);
        // :2703-2707.  zw(k_tropo) is the one-based interface k_tropo.
        real base_h = MYNN_ADD(pblh2, 1000.0f);
        real fac_damp = MYNN_SUB(1.0f, mynn_min2(MYNN_DIV(
            mynn_max2(MYNN_SUB(zagl, base_h), 0.0f),
            mynn_max2(MYNN_SUB(zw[k_tropo - 1], base_h), 500.0f)), 1.0f));
        real cld_factor = MYNN_ADD(1.0f, MYNN_MUL(fac_damp, mynn_powf(
            mynn_max2(0.0f, MYNN_DIV(MYNN_SUB(rh[k], 0.75f), 0.26f)), 1.9f)));
        cldk = mynn_min2(1.0f, MYNN_MUL(cld_factor, cldk));
        cldfra[k] = cldk;
        qc_bl[k] = ql;
        qi_bl[k] = 0.0f;
    }
    // :2723-2728
    int top = nz - 1;
    vt[top] = vt[top - 1];
    vq[top] = vq[top - 1];
    sgm[top] = sgm_in[top];
    qc_bl[top] = 0.0f;
    qi_bl[top] = 0.0f;
    cldfra[top] = 0.0f;
}

#endif
// WRF's layer-to-interface interpolation (a_k*dz_k1 + a_k1*dz_k)/(dz_k1+dz_k).
__device__ __forceinline__ real mynn_up(real a_k, real a_k1,
                                        real dz_k, real dz_k1)
{
    return MYNN_DIV(MYNN_ADD(MYNN_MUL(a_k, dz_k1), MYNN_MUL(a_k1, dz_k)),
                    MYNN_ADD(dz_k1, dz_k));
}

// module_bl_mynn.F:6827-6884 condensation_edmf.  qc is intent(inout): the
// plume carries its condensate up as the first guess.
__device__ void mynn_condensation_edmf(
    real qt, real thl, real p, real zagl, real* qc_io, real* thv_o)
{
    const real xlvcp = XLV / CP;
    const real rcp = RD / CP;
    const real rvovrd = RV / RD;
    real qc = *qc_io;
    real exn = mynn_powf(MYNN_DIV(p, 100000.0f), rcp);
    for (int it = 0; it < 50; ++it) {
        real t = MYNN_ADD(MYNN_MUL(exn, thl), MYNN_MUL(xlvcp, qc));
#if defined(MYNN_GSD41)
        real qs = mynn_gsd41_qsat_blend(t, p);
#else
        real qs = mynn_qsat_blend_rn(t, p);
#endif
        real qcold = qc;
        qc = MYNN_ADD(MYNN_MUL(0.5f, qc),
                      MYNN_MUL(0.5f, mynn_max2(MYNN_SUB(qt, qs), 0.0f)));
#if defined(MYNN_GSD41)
        // NOAA-EMC WRF 3.9 module_bl_mynn.F:5954, :5960-5968.
        if (fabsf(MYNN_SUB(qc, qcold)) < 2.0e-5f) break;
#else
        if (fabsf(MYNN_SUB(qc, qcold)) < 1.0e-6f) break;
#endif
    }
    real t = MYNN_ADD(MYNN_MUL(exn, thl), MYNN_MUL(xlvcp, qc));
#if defined(MYNN_GSD41)
    real qs = mynn_gsd41_qsat_blend(t, p);
#else
    real qs = mynn_qsat_blend_rn(t, p);
#endif
    qc = mynn_max2(MYNN_SUB(qt, qs), 0.0f);
    if (zagl < 100.0f) qc = 0.0f;
    *qc_io = qc;
    *thv_o = MYNN_MUL(
        MYNN_ADD(thl, MYNN_MUL(xlvcp, qc)),
        MYNN_SUB(MYNN_ADD(1.0f, MYNN_MUL(qt, MYNN_SUB(rvovrd, 1.0f))),
                 MYNN_MUL(rvovrd, qc)));
}
#if defined(MYNN_GSD41)
// Leaf probe for the fork's condensation_edmf (:5933-5981).  The plume
// ascent calls this same helper, including its saturation blend and stop.
extern "C" __global__
void mynn_gsd41_condensation_edmf_columns(
    const real* __restrict__ qt, const real* __restrict__ thl,
    const real* __restrict__ p, const real* __restrict__ zagl,
    real* __restrict__ qc, real* __restrict__ thv, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    mynn_condensation_edmf(qt[index], thl[index], p[index], zagl[index],
                           qc + index, thv + index);
}
#endif

// Per-column scratch: eight plume vectors of nz+1 plus ENT of nz, then the
// rhoz / dzi / edmf_th work vectors.
#define MYNN_DMP_PLUME_VECTORS 8
#define MYNN_DMP_WORK_VECTORS 3

extern "C" __global__
void mynn_dmp_mf_columns(
    const real* __restrict__ dz_raw, const real* __restrict__ p_raw,
    const real* __restrict__ rho_raw, const real* __restrict__ u_raw,
    const real* __restrict__ v_raw, const real* __restrict__ w_raw,
    const real* __restrict__ th_raw, const real* __restrict__ thl_raw,
    const real* __restrict__ thv_raw, const real* __restrict__ tk_raw,
    const real* __restrict__ qt_raw, const real* __restrict__ qv_raw,
    const real* __restrict__ qc_raw, const real* __restrict__ exner_raw,
    const real* __restrict__ rstoch_raw, const real* __restrict__ qc_bl_in_raw,
    const real* __restrict__ cldfra_bl_in_raw, const real* __restrict__ vt_in_raw,
    const real* __restrict__ vq_in_raw, const real* __restrict__ zw_raw,
    const real* __restrict__ flt_a, const real* __restrict__ fltv_a,
    const real* __restrict__ flq_a, const real* __restrict__ pblh_a,
    const real* __restrict__ dx_a, const real* __restrict__ landsea_a,
    const real* __restrict__ ts_a, const real* __restrict__ psig_shcu_a,
    real* __restrict__ edmf_a_o_raw, real* __restrict__ edmf_w_o_raw,
    real* __restrict__ edmf_qt_o_raw, real* __restrict__ edmf_thl_o_raw,
    real* __restrict__ edmf_ent_o_raw, real* __restrict__ edmf_qc_o_raw,
    real* __restrict__ qc_bl_o_raw, real* __restrict__ cldfra_bl_o_raw,
    real* __restrict__ vt_o_raw, real* __restrict__ vq_o_raw,
    real* __restrict__ s_aw_o_raw, real* __restrict__ s_awthl_o_raw,
    real* __restrict__ s_awqt_o_raw, real* __restrict__ s_awqv_o_raw,
    real* __restrict__ s_awqc_o_raw, real* __restrict__ s_awu_o_raw,
    real* __restrict__ s_awv_o_raw, real* __restrict__ maxwidth_o,
    int* __restrict__ ktop_o, real* __restrict__ ztop_o,
    real* __restrict__ maxmf_o, real* __restrict__ plume_scratch_raw,
#ifdef MYNN_EXPORT_PLUME_EDGE
    real* __restrict__ up_a_pre_o, real* __restrict__ psig_w_o, // MF-EXPORT
    int* __restrict__ plume_active_o, // MF-EXPORT
    real* __restrict__ limiter_adjustment_o, // MF-EXPORT
#endif
    real* __restrict__ work_scratch_raw, int nz, int ncol
#if defined(MYNN_GSD41)
    , const real* __restrict__ sgm_gsd41_raw
    , real* __restrict__ mf_at_base_o, int* __restrict__ kbot_o,
    int* __restrict__ nupdraft_o
#endif
    )
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
#if defined(MYNN_GSD41)
    MynnColumn<const real> sgm_gsd41 = {sgm_gsd41_raw + column, (size_t)ncol};
    mf_at_base_o[column] = 0.0f;
    kbot_o[column] = 0;
#endif
    MynnColumn<const real> dz = {dz_raw + column, (size_t)ncol};
    MynnColumn<const real> p = {p_raw + column, (size_t)ncol};
    MynnColumn<const real> rho = {rho_raw + column, (size_t)ncol};
    MynnColumn<const real> u = {u_raw + column, (size_t)ncol};
    MynnColumn<const real> v = {v_raw + column, (size_t)ncol};
    MynnColumn<const real> w = {w_raw + column, (size_t)ncol};
    MynnColumn<const real> th = {th_raw + column, (size_t)ncol};
    MynnColumn<const real> thl = {thl_raw + column, (size_t)ncol};
    MynnColumn<const real> thv = {thv_raw + column, (size_t)ncol};
    MynnColumn<const real> tk = {tk_raw + column, (size_t)ncol};
    MynnColumn<const real> qt = {qt_raw + column, (size_t)ncol};
    MynnColumn<const real> qv = {qv_raw + column, (size_t)ncol};
    MynnColumn<const real> qc = {qc_raw + column, (size_t)ncol};
    MynnColumn<const real> exner = {exner_raw + column, (size_t)ncol};
    MynnColumn<const real> rstoch = {rstoch_raw + column, (size_t)ncol};
    MynnColumn<const real> qc_bl_in = {qc_bl_in_raw + column, (size_t)ncol};
    MynnColumn<const real> cldfra_bl_in = {cldfra_bl_in_raw + column, (size_t)ncol};
    MynnColumn<const real> vt_in = {vt_in_raw + column, (size_t)ncol};
    MynnColumn<const real> vq_in = {vq_in_raw + column, (size_t)ncol};
    MynnColumn<const real> zw = {zw_raw + column, (size_t)ncol};
    MynnColumn<real> edmf_a_o = {edmf_a_o_raw + column, (size_t)ncol};
    MynnColumn<real> edmf_w_o = {edmf_w_o_raw + column, (size_t)ncol};
    MynnColumn<real> edmf_qt_o = {edmf_qt_o_raw + column, (size_t)ncol};
    MynnColumn<real> edmf_thl_o = {edmf_thl_o_raw + column, (size_t)ncol};
    MynnColumn<real> edmf_ent_o = {edmf_ent_o_raw + column, (size_t)ncol};
    MynnColumn<real> edmf_qc_o = {edmf_qc_o_raw + column, (size_t)ncol};
    MynnColumn<real> qc_bl_o = {qc_bl_o_raw + column, (size_t)ncol};
    MynnColumn<real> cldfra_bl_o = {cldfra_bl_o_raw + column, (size_t)ncol};
    MynnColumn<real> vt_o = {vt_o_raw + column, (size_t)ncol};
    MynnColumn<real> vq_o = {vq_o_raw + column, (size_t)ncol};
    MynnColumn<real> s_aw_o = {s_aw_o_raw + column, (size_t)ncol};
    MynnColumn<real> s_awthl_o = {s_awthl_o_raw + column, (size_t)ncol};
    MynnColumn<real> s_awqt_o = {s_awqt_o_raw + column, (size_t)ncol};
    MynnColumn<real> s_awqv_o = {s_awqv_o_raw + column, (size_t)ncol};
    MynnColumn<real> s_awqc_o = {s_awqc_o_raw + column, (size_t)ncol};
    MynnColumn<real> s_awu_o = {s_awu_o_raw + column, (size_t)ncol};
    MynnColumn<real> s_awv_o = {s_awv_o_raw + column, (size_t)ncol};
    MynnColumn<real> plume_scratch = {plume_scratch_raw + column, (size_t)ncol};
    MynnColumn<real> work_scratch = {work_scratch_raw + column, (size_t)ncol};
    const int nup = MYNN_DMP_NUP;
    size_t base = 0;
    size_t zbase = 0;
    size_t ibase = 0;
    MynnColumn<real> plume = plume_scratch
        + 0;
    MynnColumn<real> up_w = plume;
    MynnColumn<real> up_thl = plume + (size_t)nup * (nz + 1);
    MynnColumn<real> up_thv = plume + (size_t)2 * nup * (nz + 1);
    MynnColumn<real> up_qt = plume + (size_t)3 * nup * (nz + 1);
    MynnColumn<real> up_qc = plume + (size_t)4 * nup * (nz + 1);
    MynnColumn<real> up_a = plume + (size_t)5 * nup * (nz + 1);
    MynnColumn<real> up_u = plume + (size_t)6 * nup * (nz + 1);
    MynnColumn<real> up_v = plume + (size_t)7 * nup * (nz + 1);
    MynnColumn<real> work = work_scratch
        + 0;
    MynnColumn<real> rhoz = work;
    MynnColumn<real> dzi = work + nz;
    MynnColumn<real> edmf_th = work + 2 * nz;
    MynnColumn<real> ent = work + 3 * nz;

    // module_bl_mynn.F:279 p608 comes from module_model_constants.
    const real p608 = RV / RD - 1.0f;
    const real grav = 9.81f;
    const real gtr = MYNN_DIV(9.81f, 300.0f);
    const real tv0 = (461.6f / 287.0f - 1.0f) * 300.0f;
    const real xlvcp = XLV / CP;
    const real cpv = 4.0f * RV;
    const real onethird = MYNN_DIV(1.0f, 3.0f);
    const real atot = 0.10f, lmax = 1000.0f, lmin = 300.0f, dlmin = 0.0f;
    const real dcut = 1.2f, fluxportion = 0.75f, cf_thresh = 0.5f;
    const real dpow = -1.9f;

    real flt = flt_a[column], fltv = fltv_a[column], flq = flq_a[column];
    real pblh = pblh_a[column], dx = dx_a[column];
    real landsea = landsea_a[column], ts = ts_a[column];
    real psig_shcu = psig_shcu_a[column];

    for (int k = 0; k < nz; ++k) {
        edmf_a_o[base + k] = 0.0f;
        edmf_w_o[base + k] = 0.0f;
        edmf_qt_o[base + k] = 0.0f;
        edmf_thl_o[base + k] = 0.0f;
        edmf_ent_o[base + k] = 0.0f;
        edmf_qc_o[base + k] = 0.0f;
        qc_bl_o[base + k] = qc_bl_in[base + k];
        cldfra_bl_o[base + k] = cldfra_bl_in[base + k];
        vt_o[base + k] = vt_in[base + k];
        vq_o[base + k] = vq_in[base + k];
        rhoz[k] = 0.0f;
        dzi[k] = 0.0f;
        edmf_th[k] = 0.0f;
        for (int i = 0; i < nup; ++i) ent[(size_t)i * nz + k] = 0.001f;
    }
    for (int k = 0; k <= nz; ++k) {
        s_aw_o[ibase + k] = 0.0f;
        s_awthl_o[ibase + k] = 0.0f;
        s_awqt_o[ibase + k] = 0.0f;
        s_awqv_o[ibase + k] = 0.0f;
        s_awqc_o[ibase + k] = 0.0f;
        s_awu_o[ibase + k] = 0.0f;
        s_awv_o[ibase + k] = 0.0f;
        for (int i = 0; i < nup; ++i) {
            size_t s = (size_t)i * (nz + 1) + k;
            up_w[s] = 0.0f; up_thl[s] = 0.0f; up_thv[s] = 0.0f;
            up_qt[s] = 0.0f; up_qc[s] = 0.0f; up_a[s] = 0.0f;
            up_u[s] = 0.0f; up_v[s] = 0.0f;
        }
    }
    real nup2 = (real)nup;
#ifdef MYNN_EXPORT_PLUME_EDGE
    real* up_a_pre = up_a_pre_o + (size_t)column * (size_t)(nz + 1) * nup; // MF-EXPORT
    plume_active_o[column] = 0; // MF-EXPORT
    limiter_adjustment_o[column] = 1.0f; // MF-EXPORT
#endif

    // ---- module_bl_mynn.F:5939-5965 resolved-motion taper -----------------
    real maxw = 0.0f, cloud_base = 9000.0f;
    int k50 = 1;
    for (int k = 0; k < nz - 1; ++k) {
        if (zw[zbase + k] > MYNN_ADD(pblh, 500.0f)) break;
        real wpbl = w[base + k];
        if (w[base + k] < 0.0f) wpbl = MYNN_MUL(2.0f, w[base + k]);
        maxw = mynn_max2(maxw, fabsf(wpbl));
        if (zw[zbase + k] <= 50.0f) k50 = k + 1;
#if defined(MYNN_GSD41)
        if (qc[base + k] > 1.0e-5f && cloud_base == 9000.0f)
            cloud_base = MYNN_MUL(0.5f, MYNN_ADD(zw[zbase + k],
                                                 zw[zbase + k + 1]));
#else
        real qc_sgs = mynn_max2(qc[base + k], qc_bl_o[base + k]);
        if (qc_sgs > 1.0e-5f && cldfra_bl_o[base + k] >= 0.5f
            && cloud_base == 9000.0f)
            cloud_base = MYNN_MUL(0.5f, MYNN_ADD(zw[zbase + k],
                                                 zw[zbase + k + 1]));
#endif
    }
#if defined(MYNN_GSD41)
    // Fork :5284-5293 uses svp1, not the full surface virtual heat flux.
    maxw = mynn_max2(0.0f, MYNN_SUB(maxw, 0.5f));
    real psig_w = mynn_max2(0.0f, MYNN_SUB(1.0f, MYNN_DIV(maxw, 0.5f)));
    fltv = MYNN_ADD(flt, MYNN_MUL(0.6112f, flq));
#else
    maxw = mynn_max2(0.0f, MYNN_SUB(maxw, 1.0f));
    real psig_w = mynn_max2(0.0f, MYNN_SUB(1.0f, maxw));
#endif
    psig_w = mynn_min2(psig_w, psig_shcu);
#ifdef MYNN_EXPORT_PLUME_EDGE
    psig_w_o[column] = psig_w; // MF-EXPORT
#endif
    real fltv2 = fltv;
    if (psig_w == 0.0f && fltv > 0.0f) fltv2 = MYNN_MUL(-1.0f, fltv);

    // ---- module_bl_mynn.F:5969-5992 superadiabatic surface layer ----------
    int superadiabatic = 0;
#if defined(MYNN_GSD41)
    real hux = MYNN_SUB(landsea, 1.5f) >= 0.0f ? -0.002f : -0.005f;
    real tvs = ts;
#else
    real hux = MYNN_SUB(landsea, 1.5f) >= 0.0f ? -0.001f : -0.005f;
    real tvs = MYNN_MUL(ts, MYNN_ADD(1.0f, MYNN_MUL(p608, qv[base])));
#endif
    int nsuper = k50 - 1 > 1 ? k50 - 1 : 1;
    for (int k = 0; k < nsuper; ++k) {
        real gradient;
        if (k == 0)
#if defined(MYNN_GSD41)
            gradient = MYNN_DIV(MYNN_SUB(th[base], tvs),
                                MYNN_MUL(0.5f, dz[base]));
#else
            gradient = MYNN_DIV(MYNN_SUB(thv[base], tvs),
                                MYNN_MUL(0.5f, dz[base]));
#endif
        else
            gradient = MYNN_DIV(
#if defined(MYNN_GSD41)
                MYNN_SUB(th[base + k], th[base + k - 1]),
#else
                MYNN_SUB(thv[base + k], thv[base + k - 1]),
#endif
                MYNN_MUL(0.5f, MYNN_ADD(dz[base + k], dz[base + k - 1])));
        if (gradient < hux) {
            superadiabatic = 1;
        } else {
            superadiabatic = 0;
            break;
        }
    }

    // ---- module_bl_mynn.F:6003-6035 plume-size criteria -------------------
#if defined(MYNN_GSD41)
    // Fork :5320-5336 counts fixed 100 m diameter classes.
    nup2 = (real)max(1, min(nup, (int)MYNN_DIV(MYNN_MUL(dx, dcut), 100.0f)));
    real maxwidth = mynn_min2(MYNN_MUL(1.2f, pblh), cloud_base);
    if (MYNN_SUB(landsea, 1.5f) < 0.0f) {
        real width_flx = mynn_max2(mynn_min2(MYNN_MUL(1000.0f, MYNN_ADD(
            MYNN_MUL(0.6f, mynn_tanhf(MYNN_DIV(MYNN_SUB(flt, 0.050f),
                                               0.03f))), 0.5f)),
            1000.0f), 0.0f);
        maxwidth = mynn_min2(maxwidth, width_flx);
    }
    nup2 = mynn_min2((real)max((int)MYNN_DIV(
        MYNN_SUB(maxwidth, fmodf(maxwidth, 100.0f)), 100.0f), 0), nup2);
    real minwidth = 100.0f;
#else
    real maxwidth = mynn_min2(MYNN_MUL(dx, dcut), lmax);
    maxwidth = mynn_min2(maxwidth, MYNN_MUL(1.1f, pblh));
    if (MYNN_SUB(landsea, 1.5f) < 0.0f)
        maxwidth = mynn_min2(maxwidth, MYNN_MUL(0.5f, cloud_base));
    else
        maxwidth = mynn_min2(maxwidth, MYNN_MUL(0.9f, cloud_base));
    real wspd_pbl = sqrtf(mynn_max2(
        MYNN_ADD(MYNN_MUL(u[base], u[base]), MYNN_MUL(v[base], v[base])),
        0.01f));
    real width_flx;
    if (MYNN_SUB(landsea, 1.5f) < 0.0f)
        width_flx = mynn_max2(mynn_min2(MYNN_MUL(1000.0f, MYNN_ADD(
            MYNN_MUL(0.6f, mynn_tanhf(MYNN_DIV(MYNN_SUB(fltv, 0.040f),
                                               0.04f))), 0.5f)),
            1000.0f), 0.0f);
    else
        width_flx = mynn_max2(mynn_min2(MYNN_MUL(1000.0f, MYNN_ADD(
            MYNN_MUL(0.6f, mynn_tanhf(MYNN_DIV(MYNN_SUB(fltv, 0.007f),
                                               0.02f))), 0.5f)),
            1000.0f), 0.0f);
    maxwidth = mynn_min2(maxwidth, width_flx);
    real minwidth = lmin;
    if (maxwidth >= MYNN_SUB(lmax, 1.0f) && fltv > 0.2f)
        minwidth = MYNN_ADD(lmin, MYNN_MUL(dlmin, mynn_min2(
            MYNN_DIV(MYNN_SUB(fltv, 0.2f), 0.3f), 1.0f)));
    if (maxwidth <= minwidth) {
        nup2 = 0.0f;
        maxwidth = 0.0f;
    }
#endif
    int ktop = 0;
    real ztop = 0.0f, maxmf = 0.0f;

#if defined(MYNN_GSD41)
    if (fltv2 > 0.002f && nup2 >= 1.0f && superadiabatic) {
#else
    if (fltv2 > 0.002f && maxwidth > minwidth && superadiabatic) {
#endif
        // ---- module_bl_mynn.F:6041-6066 number density ----------------
        real cn = 0.0f;
#if defined(MYNN_GSD41)
        real dl = 100.0f;
#else
        real dl = MYNN_DIV(MYNN_SUB(maxwidth, minwidth), (real)(nup - 1));
#endif
        for (int i = 0; i < nup; ++i) {
#if defined(MYNN_GSD41)
            if (i >= (int)nup2) break;
#endif
            real len = MYNN_ADD(minwidth, MYNN_MUL(dl, (real)i));
            cn = MYNN_ADD(cn, MYNN_MUL(MYNN_DIV(
                MYNN_MUL(mynn_powf(len, dpow), MYNN_MUL(len, len)),
                MYNN_MUL(dx, dx)), dl));
        }
        real c_norm = MYNN_DIV(atot, cn);
#if defined(MYNN_GSD41)
        real acfac = MYNN_ADD(MYNN_MUL(0.5f, mynn_tanhf(
            MYNN_DIV(MYNN_SUB(fltv2, 0.02f), 0.09f))), 0.5f);
        real ac_wsp = 1.0f;
#else
        real acfac = MYNN_ADD(MYNN_MUL(0.5f, mynn_tanhf(
            MYNN_DIV(MYNN_SUB(fltv2, 0.02f), 0.05f))), 0.5f);
        real ac_wsp;
        if (wspd_pbl <= 10.0f) ac_wsp = 1.0f;
        else ac_wsp = MYNN_SUB(1.0f, mynn_min2(
            MYNN_DIV(MYNN_SUB(wspd_pbl, 10.0f), 15.0f), 1.0f));
        acfac = MYNN_MUL(acfac, ac_wsp);
#endif
        for (int i = 0; i < nup; ++i) {
#if defined(MYNN_GSD41)
            if (i >= (int)nup2) break;
#endif
            real len = MYNN_ADD(minwidth, MYNN_MUL(dl, (real)i));
            real number = MYNN_MUL(c_norm, mynn_powf(len, dpow));
            real area = MYNN_MUL(MYNN_DIV(
                MYNN_MUL(MYNN_MUL(number, len), len), MYNN_MUL(dx, dx)), dl);
            up_a[(size_t)i * (nz + 1)] = MYNN_MUL(area, acfac);
        }

        // ---- module_bl_mynn.F:6079-6144 surface plume properties ------
        const real z0 = 50.0f, pwmin = 0.1f, pwmax = 0.4f;
#if defined(MYNN_GSD41)
        real wstar = mynn_max2(1.0e-2f, mynn_powf(
            MYNN_MUL(MYNN_MUL(MYNN_DIV(grav, thv[base]), fltv2), pblh), onethird));
#else
        real wstar = mynn_max2(1.0e-2f, mynn_powf(
            MYNN_MUL(MYNN_MUL(gtr, fltv2), pblh), onethird));
#endif
        real qstar = MYNN_DIV(mynn_max2(flq, 1.0e-5f), wstar);
        real thstar = MYNN_DIV(flt, wstar);
        const real csigma = 1.34f;
#if defined(MYNN_GSD41)
        real exc_fac = 0.58f;
#else
        real exc_fac = MYNN_SUB(landsea, 1.5f) >= 0.0f
            ? MYNN_MUL(0.58f, 4.0f) : 0.58f;
        exc_fac = MYNN_MUL(exc_fac, ac_wsp);
#endif
        real zratio = mynn_powf(MYNN_DIV(z0, pblh), onethird);
        real sigma_w = MYNN_MUL(MYNN_MUL(MYNN_MUL(csigma, wstar), zratio),
            MYNN_SUB(1.0f, MYNN_DIV(MYNN_MUL(0.8f, z0), pblh)));
#if defined(MYNN_GSD41)
        real scalar_zratio = mynn_powf(MYNN_DIV(z0, pblh), -onethird);
        real sigma_qt = MYNN_MUL(MYNN_MUL(csigma, qstar), scalar_zratio);
        real sigma_th = MYNN_MUL(MYNN_MUL(csigma, thstar), scalar_zratio);
#else
        real sigma_qt = MYNN_MUL(MYNN_MUL(csigma, qstar), zratio);
        real sigma_th = MYNN_MUL(MYNN_MUL(csigma, thstar), zratio);
#endif
        real wmin = mynn_min2(MYNN_MUL(sigma_w, pwmin), 0.1f);
#if defined(MYNN_GSD41)
        real wmax = mynn_min2(MYNN_MUL(sigma_w, pwmax), 0.3f);
#else
        real wmax = mynn_min2(MYNN_MUL(sigma_w, pwmax), 0.5f);
#endif
        for (int i = 0; i < nup; ++i) {
#if defined(MYNN_GSD41)
            if (i >= (int)nup2) break;
#endif
            size_t s = (size_t)i * (nz + 1);
            up_w[s] = MYNN_ADD(wmin, MYNN_MUL(
                MYNN_DIV((real)(i + 1), (real)nup), MYNN_SUB(wmax, wmin)));
            up_u[s] = mynn_up(u[base], u[base + 1], dz[base], dz[base + 1]);
            up_v[s] = mynn_up(v[base], v[base + 1], dz[base], dz[base + 1]);
            up_qc[s] = 0.0f;
            real exc_heat = MYNN_DIV(
                MYNN_MUL(MYNN_MUL(exc_fac, up_w[s]), sigma_th), sigma_w);
            up_thv[s] = MYNN_ADD(mynn_up(thv[base], thv[base + 1],
                                         dz[base], dz[base + 1]), exc_heat);
            up_thl[s] = MYNN_ADD(mynn_up(thl[base], thl[base + 1],
                                         dz[base], dz[base + 1]), exc_heat);
            real exc_moist = MYNN_DIV(
                MYNN_MUL(MYNN_MUL(exc_fac, up_w[s]), sigma_qt), sigma_w);
            up_qt[s] = MYNN_ADD(mynn_up(qt[base], qt[base + 1],
                                        dz[base], dz[base + 1]), exc_moist);
        }

        for (int k = 0; k < nz - 1; ++k)
            rhoz[k] = mynn_up(rho[base + k], rho[base + k + 1],
                              dz[base + k], dz[base + k + 1]);
        rhoz[nz - 1] = rho[base + nz - 1];
        real dxsa = MYNN_SUB(1.0f, mynn_min2(mynn_max2(MYNN_DIV(
            MYNN_SUB(12000.0f, dx), MYNN_SUB(12000.0f, 3000.0f)), 0.0f),
            1.0f));

        // ---- module_bl_mynn.F:6170-6366 plume integration -------------
        for (int i = 0; i < nup; ++i) {
#if defined(MYNN_GSD41)
            if (i >= (int)nup2) break;
            int overshoot = 0;
#endif
            real plume_qc = 0.0f;
            real len = MYNN_ADD(minwidth, MYNN_MUL(dl, (real)i));
            size_t pbase = (size_t)i * (nz + 1);
            size_t ebase = (size_t)i * nz;
            for (int k = 1; k < nz - 1; ++k) {
#if defined(MYNN_GSD41)
                real e = MYNN_DIV(0.35f, MYNN_MUL(mynn_min2(mynn_max2(
                    up_w[pbase + k - 1], 0.75f), 1.9f), len));
#else
                real ent_wmin = MYNN_ADD(0.3f, MYNN_MUL(len, 0.0005f));
                real e = MYNN_DIV(0.33f, MYNN_MUL(mynn_min2(mynn_max2(
                    up_w[pbase + k - 1], ent_wmin), 0.9f), len));
#endif
                e = mynn_max2(e, 0.0003f);
#if defined(MYNN_GSD41)
                real ramp = mynn_min2(MYNN_ADD(pblh, 1500.0f), 3500.0f);
#else
                real ramp = mynn_min2(MYNN_ADD(pblh, 1500.0f), 4000.0f);
#endif
                if (zw[zbase + k] >= ramp)
                    e = MYNN_ADD(e, MYNN_MUL(
                        MYNN_SUB(zw[zbase + k], ramp), 5.0e-6f));
                e = MYNN_MUL(e, MYNN_SUB(1.0f, rstoch[base + k]));
                e = mynn_min2(e, MYNN_DIV(0.9f, MYNN_SUB(
                    zw[zbase + k + 1], zw[zbase + k])));
                ent[ebase + k] = e;

                // pgfac is 0, so the pressure-gradient term is an exact zero;
                // it is written out so the expression tree stays WRF's.
                real uk = mynn_up(u[base + k], u[base + k + 1],
                                  dz[base + k], dz[base + k + 1]);
                real ukm1 = mynn_up(u[base + k - 1], u[base + k],
                                    dz[base + k - 1], dz[base + k]);
                real vk = mynn_up(v[base + k], v[base + k + 1],
                                  dz[base + k], dz[base + k + 1]);
                real vkm1 = mynn_up(v[base + k - 1], v[base + k],
                                    dz[base + k - 1], dz[base + k]);
                real ent_exp = MYNN_MUL(e, MYNN_SUB(zw[zbase + k + 1],
                                                    zw[zbase + k]));
#if defined(MYNN_GSD41)
                real ent_exm = ent_exp;
#else
                real ent_exm = MYNN_MUL(ent_exp, 0.3333f);
#endif
                real qtn = MYNN_ADD(
                    MYNN_MUL(up_qt[pbase + k - 1], MYNN_SUB(1.0f, ent_exp)),
                    MYNN_MUL(qt[base + k], ent_exp));
                real thln = MYNN_ADD(
                    MYNN_MUL(up_thl[pbase + k - 1], MYNN_SUB(1.0f, ent_exp)),
                    MYNN_MUL(thl[base + k], ent_exp));
                real un = MYNN_ADD(MYNN_ADD(
                    MYNN_MUL(up_u[pbase + k - 1], MYNN_SUB(1.0f, ent_exm)),
                    MYNN_MUL(u[base + k], ent_exm)),
                    MYNN_MUL(MYNN_MUL(dxsa, 0.0f), MYNN_SUB(uk, ukm1)));
                real vn = MYNN_ADD(MYNN_ADD(
                    MYNN_MUL(up_v[pbase + k - 1], MYNN_SUB(1.0f, ent_exm)),
                    MYNN_MUL(v[base + k], ent_exm)),
                    MYNN_MUL(MYNN_MUL(dxsa, 0.0f), MYNN_SUB(vk, vkm1)));

                real pk = mynn_up(p[base + k], p[base + k + 1],
                                  dz[base + k], dz[base + k + 1]);
                real thvn;
                mynn_condensation_edmf(qtn, thln, pk, zw[zbase + k + 1],
                                      &plume_qc, &thvn);
                real thvk = mynn_up(thv[base + k], thv[base + k + 1],
                                    dz[base + k], dz[base + k + 1]);
                real buoyancy = MYNN_MUL(grav,
                                         MYNN_SUB(MYNN_DIV(thvn, thvk), 1.0f));
                real bcoeff = buoyancy > 0.0f ? 0.15f : 0.2f;
                real previous = up_w[pbase + k - 1];
                real step = mynn_min2(MYNN_SUB(zw[zbase + k],
                                              zw[zbase + k - 1]), 250.0f);
                real divisor = previous < 0.2f
                    ? mynn_max2(previous, 0.2f) : previous;
                real wn = MYNN_ADD(previous, MYNN_MUL(MYNN_ADD(
                    MYNN_MUL(MYNN_MUL(-2.0f, e), previous),
                    MYNN_DIV(MYNN_MUL(bcoeff, buoyancy), divisor)), step));
                real limit = mynn_min2(MYNN_DIV(MYNN_MUL(1.25f, MYNN_SUB(
                    zw[zbase + k], zw[zbase + k - 1])), 200.0f), 2.0f);
                if (wn > MYNN_ADD(previous, limit))
                    wn = MYNN_ADD(previous, limit);
                if (wn < MYNN_SUB(previous, limit))
                    wn = MYNN_SUB(previous, limit);
                wn = mynn_min2(mynn_max2(wn, 0.0f), 3.0f);
#if defined(MYNN_GSD41)
                // Fork :5547-5565 permits one layer of energetic overshoot.
                if (fltv2 > 0.05f && wn <= 0.0f && overshoot == 0) {
                    overshoot = 1;
                    real thvkm1 = mynn_up(thv[base + k - 1], thv[base + k],
                                         dz[base + k - 1], dz[base + k]);
                    if (MYNN_SUB(thvk, thvkm1) > 0.0f) {
                        real bvf = sqrtf(MYNN_DIV(
                            MYNN_MUL(gtr, MYNN_SUB(thvk, thvkm1)), dz[base + k]));
                        real frz = MYNN_DIV(previous, MYNN_MUL(bvf, dz[base + k]));
                        if (frz >= 0.5f) wn = MYNN_MUL(mynn_min2(frz, 1.0f), previous);
                    }
                } else if (fltv2 > 0.05f && overshoot == 1) {
                    wn = 0.0f;
                }
                wn = MYNN_MUL(wn, mynn_expf_rn(-MYNN_DIV(mynn_max2(MYNN_SUB(
                    zw[zbase + k + 1], mynn_min2(MYNN_ADD(pblh, 2000.0f),
                    3000.0f)), 0.0f), 1000.0f)));
                if (zw[zbase + k + 1] >= mynn_min2(MYNN_ADD(pblh, 3000.0f),
                                                   4500.0f)) wn = 0.0f;
#else
                if (k == 1 && wn == 0.0f) {
                    nup2 = 0.0f;
                    break;
                }
#endif
                if (wn > 0.0f) {
                    up_w[pbase + k] = wn;
                    up_thv[pbase + k] = thvn;
                    up_thl[pbase + k] = thln;
                    up_qt[pbase + k] = qtn;
                    up_qc[pbase + k] = plume_qc;
                    up_u[pbase + k] = un;
                    up_v[pbase + k] = vn;
                    up_a[pbase + k] = up_a[pbase + k - 1];
                    if (k + 1 > ktop) ktop = k + 1;
                } else {
                    break;
                }
            }
        }
    } else {
        nup2 = 0.0f;
    }

#ifdef MYNN_EXPORT_PLUME_EDGE
    for (int k = 0; k <= nz; ++k) // MF-EXPORT
        for (int i = 0; i < nup; ++i) // MF-EXPORT
            up_a_pre[(size_t)k * nup + i] = up_a[(size_t)i * (nz + 1) + k]; // MF-EXPORT
#endif
    if (ktop > nz - 1) ktop = nz - 1;
    ztop = ktop == 0 ? 0.0f : zw[zbase + ktop - 1];

    if (nup2 > 0.0f) {
        // ---- module_bl_mynn.F:6404-6425 interface fluxes --------------
        for (int i = 0; i < nup; ++i) {
            size_t pbase = (size_t)i * (nz + 1);
            for (int k = 0; k < nz - 1; ++k) {
#if defined(MYNN_GSD41)
                if (k >= ktop) break;
#endif
#if defined(MYNN_GSD41)
                real raw = up_a[pbase + k];
#else
                real raw = MYNN_MUL(rhoz[k], up_a[pbase + k]);
#endif
                real aw = MYNN_MUL(raw, up_w[pbase + k]);
                s_aw_o[ibase + k + 1] = MYNN_ADD(
                    s_aw_o[ibase + k + 1], MYNN_MUL(aw, psig_w));
                s_awthl_o[ibase + k + 1] = MYNN_ADD(
                    s_awthl_o[ibase + k + 1],
                    MYNN_MUL(MYNN_MUL(aw, up_thl[pbase + k]), psig_w));
                s_awqt_o[ibase + k + 1] = MYNN_ADD(
                    s_awqt_o[ibase + k + 1],
                    MYNN_MUL(MYNN_MUL(aw, up_qt[pbase + k]), psig_w));
                s_awqc_o[ibase + k + 1] = MYNN_ADD(
                    s_awqc_o[ibase + k + 1],
                    MYNN_MUL(MYNN_MUL(aw, up_qc[pbase + k]), psig_w));
                s_awqv_o[ibase + k + 1] = MYNN_SUB(
                    s_awqt_o[ibase + k + 1], s_awqc_o[ibase + k + 1]);
            }
        }
        // momentum_opt is 1.
        for (int i = 0; i < nup; ++i) {
            size_t pbase = (size_t)i * (nz + 1);
            for (int k = 0; k < nz - 1; ++k) {
#if defined(MYNN_GSD41)
                if (k >= ktop) break;
#endif
#if defined(MYNN_GSD41)
                real aw = MYNN_MUL(up_a[pbase + k], up_w[pbase + k]);
#else
                real aw = MYNN_MUL(MYNN_MUL(rhoz[k], up_a[pbase + k]),
                                   up_w[pbase + k]);
#endif
                s_awu_o[ibase + k + 1] = MYNN_ADD(
                    s_awu_o[ibase + k + 1],
                    MYNN_MUL(MYNN_MUL(aw, up_u[pbase + k]), psig_w));
                s_awv_o[ibase + k + 1] = MYNN_ADD(
                    s_awv_o[ibase + k + 1],
                    MYNN_MUL(MYNN_MUL(aw, up_v[pbase + k]), psig_w));
            }
        }

        // ---- module_bl_mynn.F:6462-6496 heat-flux limiter -------------
        real flx1;
        if (s_aw_o[ibase + 1] != 0.0f) {
            dzi[0] = MYNN_MUL(0.5f, MYNN_ADD(dz[base], dz[base + 1]));
            flx1 = mynn_max2(MYNN_DIV(MYNN_MUL(s_aw_o[ibase + 1],
                MYNN_SUB(th[base], th[base + 1])), dzi[0]), 1.0e-5f);
        } else {
            flx1 = 0.0f;
        }
        real adjustment = 1.0f;
#if defined(MYNN_GSD41)
        real flt2 = flt;
#else
        real flt2 = mynn_max2(flt, 0.0f);
#endif
        real threshold = MYNN_DIV(MYNN_MUL(fluxportion, flt2), dz[base]);
        if (flx1 > threshold && flx1 > 0.0f) {
            adjustment = MYNN_DIV(threshold, flx1);
            for (int k = 0; k <= nz; ++k) {
                s_aw_o[ibase + k] = MYNN_MUL(s_aw_o[ibase + k], adjustment);
                s_awthl_o[ibase + k] = MYNN_MUL(s_awthl_o[ibase + k],
                                                adjustment);
                s_awqt_o[ibase + k] = MYNN_MUL(s_awqt_o[ibase + k],
                                               adjustment);
                s_awqc_o[ibase + k] = MYNN_MUL(s_awqc_o[ibase + k],
                                               adjustment);
                s_awqv_o[ibase + k] = MYNN_MUL(s_awqv_o[ibase + k],
                                               adjustment);
                s_awu_o[ibase + k] = MYNN_MUL(s_awu_o[ibase + k], adjustment);
                s_awv_o[ibase + k] = MYNN_MUL(s_awv_o[ibase + k], adjustment);
            }
            for (int k = 0; k <= nz; ++k)
                for (int i = 0; i < nup; ++i) {
                    size_t s = (size_t)i * (nz + 1) + k;
                    up_a[s] = MYNN_MUL(up_a[s], adjustment);
                }
        }

#ifdef MYNN_EXPORT_PLUME_EDGE
        limiter_adjustment_o[column] = adjustment; // MF-EXPORT
        plume_active_o[column] = 1; // MF-EXPORT
#endif
        // ---- module_bl_mynn.F:6504-6524 plume means -------------------
        for (int k = 0; k < nz - 1; ++k) {
#if defined(MYNN_GSD41)
            if (k >= ktop) break;
#endif
            for (int i = 0; i < nup; ++i) {
                size_t s = (size_t)i * (nz + 1) + k;
                real a = up_a[s];
                edmf_a_o[base + k] = MYNN_ADD(edmf_a_o[base + k], a);
                edmf_w_o[base + k] = MYNN_ADD(edmf_w_o[base + k],
                                              MYNN_MUL(a, up_w[s]));
                edmf_qt_o[base + k] = MYNN_ADD(edmf_qt_o[base + k],
                                               MYNN_MUL(a, up_qt[s]));
                edmf_thl_o[base + k] = MYNN_ADD(edmf_thl_o[base + k],
                                                MYNN_MUL(a, up_thl[s]));
                edmf_ent_o[base + k] = MYNN_ADD(edmf_ent_o[base + k],
                    MYNN_MUL(a, ent[(size_t)i * nz + k]));
                edmf_qc_o[base + k] = MYNN_ADD(edmf_qc_o[base + k],
                                               MYNN_MUL(a, up_qc[s]));
            }
        }
        for (int k = 0; k < nz - 1; ++k) {
#if defined(MYNN_GSD41)
            if (k >= ktop) break;
#endif
            if (edmf_a_o[base + k] > 0.0f) {
                edmf_w_o[base + k] = MYNN_DIV(edmf_w_o[base + k],
                                              edmf_a_o[base + k]);
                edmf_qt_o[base + k] = MYNN_DIV(edmf_qt_o[base + k],
                                               edmf_a_o[base + k]);
                edmf_thl_o[base + k] = MYNN_DIV(edmf_thl_o[base + k],
                                                edmf_a_o[base + k]);
                edmf_ent_o[base + k] = MYNN_DIV(edmf_ent_o[base + k],
                                                edmf_a_o[base + k]);
                edmf_qc_o[base + k] = MYNN_DIV(edmf_qc_o[base + k],
                                               edmf_a_o[base + k]);
                edmf_a_o[base + k] = MYNN_MUL(edmf_a_o[base + k], psig_w);
                real product = MYNN_MUL(edmf_a_o[base + k],
                                        edmf_w_o[base + k]);
                if (product > maxmf) maxmf = product;
#if defined(MYNN_GSD41)
                // Native GSD DMP_mf:5833-5837; the last cloud onset wins.
                if (k > 0 && edmf_qc_o[base + k] > 0.0f
                        && edmf_qc_o[base + k - 1] == 0.0f) {
                    kbot_o[column] = k + 1;
                    mf_at_base_o[column] = product;
                }
#endif
            }
        }

        // ---- module_bl_mynn.F:6619-6625 interface exner, plume theta --
        for (int k = 0; k < nz - 1; ++k) {
            real exneri = mynn_up(exner[base + k], exner[base + k + 1],
                                  dz[base + k], dz[base + k + 1]);
            edmf_th[k] = MYNN_ADD(edmf_thl_o[base + k],
                MYNN_MUL(MYNN_DIV(xlvcp, exneri), edmf_qc_o[base + k]));
            dzi[k] = MYNN_MUL(0.5f, MYNN_ADD(dz[base + k],
                                             dz[base + k + 1]));
        }

        // ---- module_bl_mynn.F:6633-6764 shallow-cumulus cloud fraction -
        for (int k = 1; k < nz - 2; ++k) {
            if (k + 1 > ktop) break;
#if defined(MYNN_GSD41)
            // GSD MYNN v4.1 (NOAA-EMC WRF 3.9 module_bl_mynn.F:5767-5870):
            // wherever the plume holds condensate, a Chaboureau-Bechtold
            // fraction from the plume's convective sigma combined with the
            // stratus sigma, clipped to 0.01..0.6; it replaces the stratus
            // cloud only where that is under 0.5, with in-cloud water; the
            // buoyancy functions are rebuilt with MIN(0.4, cldfra) wherever
            // the plume holds condensate.
            if (MYNN_MUL(0.5f, MYNN_ADD(edmf_qc_o[base + k],
                                        edmf_qc_o[base + k - 1])) > 0.0f) {
                const real r_v = 461.6f, cp = MYNN_DIV(MYNN_MUL(7.0f, 287.0f),
                                                       2.0f);
                const real rcp = MYNN_DIV(287.0f, cp);
                const real ep_2 = MYNN_DIV(287.0f, r_v);
                const real ep_3 = MYNN_SUB(1.0f, ep_2);
                const real cpv_g = MYNN_MUL(4.0f, r_v);
                const real xlvcp_g = MYNN_DIV(2.5e6f, cp);
                const real tv0_g = MYNN_MUL(MYNN_SUB(MYNN_DIV(r_v, 287.0f),
                                                     1.0f), 300.0f);
                real thp = mynn_up(edmf_th[k], edmf_th[k - 1],
                                   dzi[k], dzi[k - 1]);
                real qtp = mynn_up(edmf_qt_o[base + k],
                                   edmf_qt_o[base + k - 1],
                                   dzi[k], dzi[k - 1]);
                real tp = MYNN_MUL(thp, exner[base + k]);
                real esat = mynn_gsd41_esat_blend(tp);
                real qsl = MYNN_DIV(MYNN_MUL(ep_2, esat), mynn_max2(1.0e-4f,
                    MYNN_SUB(p[base + k], MYNN_MUL(ep_3, esat))));
                real qcp;
                if (edmf_qc_o[base + k] > 0.0f
                        && edmf_qc_o[base + k - 1] > 0.0f)
                    qcp = MYNN_MUL(0.5f, MYNN_ADD(edmf_qc_o[base + k],
                                                  edmf_qc_o[base + k - 1]));
                else
                    qcp = mynn_max2(0.0f, MYNN_SUB(qtp, qsl));
                real xl = mynn_gsd41_xl_blend(tk[base + k]);
                real tlk = MYNN_MUL(thl[base + k], mynn_powf(
                    MYNN_DIV(p[base + k], 100000.0f), rcp));
                real qsat_tl = mynn_gsd41_qsat_blend(tlk, p[base + k]);
                real rsl = MYNN_DIV(MYNN_MUL(xl, qsat_tl),
                                    MYNN_MUL(r_v, MYNN_MUL(tlk, tlk)));
                real cpm = MYNN_ADD(cp, MYNN_MUL(qt[base + k], cpv_g));
                real a_cb = MYNN_DIV(1.0f, MYNN_ADD(1.0f,
                    MYNN_DIV(MYNN_MUL(xl, rsl), cpm)));
                real b9 = MYNN_MUL(a_cb, rsl);
                real asum = MYNN_ADD(edmf_a_o[base + k],
                                     edmf_a_o[base + k - 1]);
                real q2p = MYNN_DIV(xlvcp_g, exner[base + k]);
                real pt = MYNN_ADD(thl[base + k], MYNN_MUL(MYNN_MUL(
                    MYNN_MUL(q2p, qcp), 0.5f), asum));
                real bb = MYNN_DIV(MYNN_MUL(b9, tk[base + k]), pt);
                real qww = MYNN_ADD(1.0f, MYNN_MUL(0.61f, qt[base + k]));
                real alpha = MYNN_MUL(0.61f, pt);
                real tt = MYNN_MUL(th[base + k], exner[base + k]);
                real beta = MYNN_SUB(MYNN_DIV(MYNN_MUL(pt, xl),
                    MYNN_MUL(tt, cp)), MYNN_MUL(1.61f, pt));
                real f = a_cb > 0.0f
                    ? mynn_min2(MYNN_DIV(1.0f, a_cb), 4.0f) : 1.0f;
                real sigq = MYNN_MUL(MYNN_MUL(MYNN_MUL(MYNN_MUL(
                    MYNN_MUL(9.0e-3f, 0.5f), asum), 0.5f),
                    MYNN_ADD(edmf_w_o[base + k], edmf_w_o[base + k - 1])), f);
                sigq = sqrtf(MYNN_ADD(MYNN_MUL(sigq, sigq),
                    MYNN_MUL(sgm_gsd41[base + k], sgm_gsd41[base + k])));
                real qmq = MYNN_MUL(a_cb, MYNN_SUB(qt[base + k], qsat_tl));
                real mf_cf = mynn_min2(mynn_max2(MYNN_ADD(0.5f, MYNN_MUL(
                    0.36f, mynn_atanf(MYNN_MUL(1.55f, MYNN_DIV(qmq, sigq))))),
                    0.01f), 0.6f);
                if (cldfra_bl_o[base + k] < 0.5f) {
                    real ahalf = MYNN_MUL(0.5f, asum);
                    if (mf_cf > ahalf) {
                        cldfra_bl_o[base + k] = mf_cf;
                        qc_bl_o[base + k] = MYNN_DIV(MYNN_MUL(MYNN_MUL(
                            qcp, 0.5f), asum), mf_cf);
                    } else {
                        cldfra_bl_o[base + k] = ahalf;
                        qc_bl_o[base + k] = qcp;
                    }
                }
                real q1 = mynn_max2(MYNN_DIV(qmq, mynn_max2(sigq, 1.0e-10f)),
                                    -5.0f);
                real fng;
                if (q1 >= 1.0f) fng = 1.0f;
                else if (q1 >= -1.7f)
                    fng = mynn_expf_rn(MYNN_MUL(-0.4f, MYNN_SUB(q1, 1.0f)));
                else if (q1 >= -2.5f)
                    fng = MYNN_ADD(3.0f, mynn_expf_rn(
                        MYNN_MUL(-3.8f, MYNN_ADD(q1, 1.7f))));
                else
                    fng = mynn_min2(MYNN_ADD(23.9f, mynn_expf_rn(
                        MYNN_MUL(-1.6f, MYNN_ADD(q1, 2.5f)))), 60.0f);
                real cw = mynn_min2(0.4f, cldfra_bl_o[base + k]);
                vt_o[base + k] = MYNN_SUB(MYNN_SUB(qww, MYNN_MUL(MYNN_MUL(
                    MYNN_MUL(cw, beta), bb), fng)), 1.0f);
                vq_o[base + k] = MYNN_SUB(MYNN_ADD(alpha, MYNN_MUL(MYNN_MUL(
                    MYNN_MUL(cw, beta), a_cb), fng)), tv0_g);
            }
            continue;
#endif
            if (!(MYNN_MUL(0.5f, MYNN_ADD(edmf_qc_o[base + k],
                                          edmf_qc_o[base + k - 1])) > 0.0f
                  && cldfra_bl_o[base + k] < cf_thresh))
                continue;
            real aup = mynn_up(edmf_a_o[base + k], edmf_a_o[base + k - 1],
                               dzi[k], dzi[k - 1]);
            real qtp = mynn_up(edmf_qt_o[base + k], edmf_qt_o[base + k - 1],
                               dzi[k], dzi[k - 1]);
            // esat and qsl are computed at :6641-6643 and never read again.
            real qcp;
            if (edmf_qc_o[base + k] > 0.0f && edmf_qc_o[base + k - 1] > 0.0f)
                qcp = mynn_up(edmf_qc_o[base + k], edmf_qc_o[base + k - 1],
                              dzi[k], dzi[k - 1]);
            else
                qcp = mynn_max2(edmf_qc_o[base + k],
                                edmf_qc_o[base + k - 1]);
            real xl = mynn_xl_blend_rn(tk[base + k]);
            real qsat_tk = mynn_qsat_blend_rn(tk[base + k], p[base + k]);
            real rsl = MYNN_DIV(MYNN_MUL(xl, qsat_tk),
                MYNN_MUL(RV, MYNN_MUL(tk[base + k], tk[base + k])));
            real cpm = MYNN_ADD(CP, MYNN_MUL(qt[base + k], cpv));
            real a_cb = MYNN_DIV(1.0f, MYNN_ADD(1.0f,
                MYNN_DIV(MYNN_MUL(xl, rsl), cpm)));
            real b9 = MYNN_MUL(a_cb, rsl);
            real q2p = MYNN_DIV(xlvcp, exner[base + k]);
            real pt = MYNN_ADD(thl[base + k],
                               MYNN_MUL(MYNN_MUL(q2p, qcp), aup));
            real bb = MYNN_DIV(MYNN_MUL(b9, tk[base + k]), pt);
            real qww = MYNN_ADD(1.0f, MYNN_MUL(0.61f, qt[base + k]));
            real alpha = MYNN_MUL(0.61f, pt);
            real beta = MYNN_SUB(MYNN_DIV(MYNN_MUL(pt, xl),
                MYNN_MUL(tk[base + k], CP)), MYNN_MUL(1.61f, pt));
            real sigq = MYNN_MUL(MYNN_MUL(10.0f, aup),
                                 MYNN_SUB(qtp, qt[base + k]));
            sigq = mynn_max2(sigq, MYNN_MUL(qsat_tk, 0.02f));
            sigq = mynn_min2(sigq, MYNN_MUL(qsat_tk, 0.25f));
            real qmq = MYNN_MUL(a_cb, MYNN_SUB(qt[base + k], qsat_tk));
            real q1 = MYNN_DIV(qmq, sigq);
            real mf_cf = mynn_min2(mynn_max2(MYNN_ADD(0.5f, MYNN_MUL(0.36f,
                mynn_atanf(MYNN_MUL(1.55f, q1)))), 0.01f), 0.6f);
            if (MYNN_SUB(landsea, 1.5f) >= 0.0f)
                mf_cf = mynn_max2(mf_cf, MYNN_MUL(1.2f, aup));
            else
                mf_cf = mynn_max2(mf_cf, MYNN_MUL(1.8f, aup));
            mf_cf = mynn_min2(mf_cf, MYNN_MUL(5.0f, aup));
            if (MYNN_MUL(qcp, aup) > 5.0e-5f)
                qc_bl_o[base + k] = MYNN_SUB(
                    MYNN_MUL(1.86f, MYNN_MUL(qcp, aup)), 2.2e-5f);
            else
                qc_bl_o[base + k] = MYNN_MUL(1.18f, MYNN_MUL(qcp, aup));
            cldfra_bl_o[base + k] = mf_cf;
            q1 = mynn_max2(q1, -2.25f);
            real fng;
            if (q1 >= 1.0f) fng = 1.0f;
            else if (q1 >= -1.7f)
                fng = mynn_expf_rn(MYNN_MUL(-0.4f, MYNN_SUB(q1, 1.0f)));
            else if (q1 >= -2.5f)
                fng = MYNN_ADD(3.0f, mynn_expf_rn(
                    MYNN_MUL(-3.8f, MYNN_ADD(q1, 1.7f))));
            else
                fng = mynn_min2(MYNN_ADD(23.9f, mynn_expf_rn(
                    MYNN_MUL(-1.6f, MYNN_ADD(q1, 2.5f)))), 60.0f);
            vt_o[base + k] = MYNN_SUB(MYNN_SUB(qww, MYNN_MUL(MYNN_MUL(
                MYNN_MUL(MYNN_MUL(1.5f, aup), beta), bb), fng)), 1.0f);
            vq_o[base + k] = MYNN_SUB(MYNN_ADD(alpha, MYNN_MUL(MYNN_MUL(
                MYNN_MUL(MYNN_MUL(1.5f, aup), beta), a_cb), fng)), tv0);
        }
    }

    // ---- module_bl_mynn.F:6771-6773 dry-plume sign convention ------------
    if (ktop > 0) {
#if defined(MYNN_GSD41)
        if (ktop > 0) {
#endif
        real maxqc = edmf_qc_o[base];
        for (int k = 1; k < ktop; ++k)
            if (edmf_qc_o[base + k] > maxqc) maxqc = edmf_qc_o[base + k];
        if (maxqc < 1.0e-8f) maxmf = MYNN_MUL(-1.0f, maxmf);
#if defined(MYNN_GSD41)
        }
#endif
    }

    maxwidth_o[column] = maxwidth;
    ktop_o[column] = ktop;
    ztop_o[column] = ztop;
    maxmf_o[column] = maxmf;
#if defined(MYNN_GSD41)
    nupdraft_o[column] = (int)nup2;
#endif
}

#if defined(MYNN_GSD41)
// Native GSD mynn_bl_driver:4477-4485, GRIMS diagnostics. One device thread
// per column, retaining the native FP32 operator and real-power order.
extern "C" __global__
void mynn_gsd41_grims_diagnostics(
    const real* th, const real* qv, const real* rho, const real* hfx,
    const real* qfx, const real* ust, const real* pblh,
    real* wstar, real* delta, int ncol)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    real cpm = MYNN_MUL(CP, MYNN_ADD(1.0f, MYNN_MUL(0.84f, qv[c])));
    real govrth = MYNN_DIV(9.81f, th[c]);
    real ep1 = MYNN_SUB(MYNN_DIV(461.6f, 287.0f), 1.0f);
    real sflux = MYNN_ADD(MYNN_DIV(MYNN_DIV(hfx[c], rho[c]), cpm),
        MYNN_MUL(MYNN_MUL(MYNN_DIV(qfx[c], rho[c]), ep1), th[c]));
    real w3 = MYNN_MUL(MYNN_MUL(govrth, mynn_max2(sflux, 0.0f)), pblh[c]);
    wstar[c] = mynn_powf(w3, 0.33333335f);
    real wm3 = MYNN_ADD(w3, MYNN_MUL(5.0f, mynn_powf(ust[c], 3.0f)));
    real wm2 = mynn_powf(wm3, 0.6666667f);
    real delb = MYNN_MUL(MYNN_MUL(govrth, 0.001f), pblh[c]);
    delta[c] = mynn_min2(MYNN_ADD(MYNN_MUL(0.02f, pblh[c]),
        MYNN_DIV(MYNN_MUL(0.05f, wm2), delb)), 100.0f);
}
#endif

// ===========================================================================
// module_bl_mynn.F:mynn_bl_driver assembly.
//
// These four kernels are the arithmetic the driver does *between* its calls
// to the routines above.  They exist as device code, rather than as CuPy
// array expressions in gpuwm/core/mynn_pbl_gpu.py, for the two reasons this
// file's header already records and which a CuPy expression cannot avoid:
// NVRTC contracts `a*b+c` into an FMA, and CuPy appends `-ftz=true` to every
// compile so a subnormal is flushed.  Written with array operators the
// assembly measured 205 ULP of drift in RUBLTEN and 32 ULP in CLDFRA_BL
// against the oracle-pinned CPU driver on the four-column fixture; written
// through MYNN_ADD/MYNN_MUL it is bitwise.
// ===========================================================================

// module_bl_mynn.F:866-1017.  zw is the sequential FP32 interface-height
// accumulation at :1001-1017 -- not a prefix scan, which re-associates it.
// qke_seed is the Koracin and Berkowicz taper at :775; it is only read on the
// initflag>0 path, and computing it unconditionally costs one thread-local
// max per level.
extern "C" __global__
void mynn_driver_prep_columns(
    const real* __restrict__ dz_raw, const real* __restrict__ exner_raw,
    const real* __restrict__ sqv_raw, const real* __restrict__ sqc_raw,
    const real* __restrict__ sqi_raw, const real* __restrict__ th_raw,
    const real* __restrict__ ust,
    real* __restrict__ zw_raw, real* __restrict__ qv1_raw,
    real* __restrict__ sqw_raw, real* __restrict__ thl_raw,
    real* __restrict__ thetav_raw, real* __restrict__ qke_seed_raw,
    int nz, int ncol)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    MynnColumn<const real> dz = {dz_raw + column, (size_t)ncol};
    MynnColumn<const real> exner = {exner_raw + column, (size_t)ncol};
    MynnColumn<const real> sqv = {sqv_raw + column, (size_t)ncol};
    MynnColumn<const real> sqc = {sqc_raw + column, (size_t)ncol};
    MynnColumn<const real> sqi = {sqi_raw + column, (size_t)ncol};
    MynnColumn<const real> th = {th_raw + column, (size_t)ncol};
    MynnColumn<real> zw = {zw_raw + column, (size_t)ncol};
    MynnColumn<real> qv1 = {qv1_raw + column, (size_t)ncol};
    MynnColumn<real> sqw = {sqw_raw + column, (size_t)ncol};
    MynnColumn<real> thl = {thl_raw + column, (size_t)ncol};
    MynnColumn<real> thetav = {thetav_raw + column, (size_t)ncol};
    MynnColumn<real> qke_seed = {qke_seed_raw + column, (size_t)ncol};
    const real xlvcp = XLV / CP;
    const real xlscp = (XLV + 3.50e5f) / CP;
    const real p608 = RV / RD - 1.0f;
    const int base = 0;
    const int wbase = 0;

    zw[wbase] = 0.0f;
    for (int k = 0; k < nz; ++k)
        zw[wbase + k + 1] = MYNN_ADD(zw[wbase + k], dz[base + k]);

    real ustar = ust[column];
    real lead = MYNN_MUL(5.0f, ustar);
    real reach = MYNN_MUL(ustar, 700.0f);
    real denom = MYNN_MUL(mynn_max2(ustar, 0.01f), 700.0f);
    for (int k = 0; k < nz; ++k) {
        real e = exner[base + k];
        real qc = sqc[base + k];
        real qi = sqi[base + k];
        real qv = sqv[base + k];
        real theta = th[base + k];
        qv1[base + k] = MYNN_DIV(qv, MYNN_SUB(1.0f, qv));
        sqw[base + k] = MYNN_ADD(MYNN_ADD(qv, qc), qi);
        thl[base + k] = MYNN_SUB(
            MYNN_SUB(theta, MYNN_MUL(MYNN_DIV(xlvcp, e), qc)),
            MYNN_MUL(MYNN_DIV(xlscp, e), qi));
#if defined(MYNN_GSD41)
        // The fork's warm-step thetav (module_bl_mynn.F:4268) is
        // th*(1+0.608*sqv), a literal, not p608 = Rv/Rd - 1; DMP_mf reads it.
        thetav[base + k] = MYNN_MUL(
            theta, MYNN_ADD(1.0f, MYNN_MUL(0.608f, qv)));
#else
        thetav[base + k] = MYNN_MUL(
            theta, MYNN_ADD(1.0f, MYNN_MUL(p608, qv)));
#endif
        qke_seed[base + k] = MYNN_MUL(lead, mynn_max2(
            MYNN_DIV(MYNN_SUB(reach, zw[wbase + k]), denom), 0.01f));
    }
}

// module_bl_mynn.F:1057-1097.  FLQC is identically zero under this identity
// (the driver never fills it), so it is written rather than read: a caller
// that has to hand it to mynn_tendencies must not have to assume it.
// zet is clamped to WRF's [-20,20] and then :1095-1096's pmz/phh are built
// here, in the same thread, from the glibc-transcribed phim/phih at the top of
// this file.  They used to come back to the host for that -- a flat 149 us per
// column, which was the scheme's scaling blocker; the arithmetic is two
// scalars per column and belongs where its input already is.
extern "C" __global__
void mynn_driver_surface_columns(
    const real* __restrict__ rho_raw, const real* __restrict__ exner_raw,
    const real* __restrict__ dz_raw, const real* __restrict__ qv1_raw,
    const real* __restrict__ ust, const real* __restrict__ hfx,
    const real* __restrict__ qfx, const real* __restrict__ ts,
    real* __restrict__ flt, real* __restrict__ fltv,
    real* __restrict__ flq, real* __restrict__ flqv,
    real* __restrict__ flqc, real* __restrict__ th_sfc,
    real* __restrict__ rmol, real* __restrict__ zet,
    real* __restrict__ pmz, real* __restrict__ phh,
    int nz, int ncol
#if defined(MYNN_GSD41)
    , const real* __restrict__ rmol_in
#endif
    )
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    MynnColumn<const real> rho = {rho_raw + column, (size_t)ncol};
    MynnColumn<const real> exner = {exner_raw + column, (size_t)ncol};
    MynnColumn<const real> dz = {dz_raw + column, (size_t)ncol};
    MynnColumn<const real> qv1 = {qv1_raw + column, (size_t)ncol};
    const real xlvcp = XLV / CP;
    const real p608 = RV / RD - 1.0f;
    const real kgtr = MYNN_MUL(0.4f, MYNN_DIV(9.81f, 300.0f));
    const int base = 0;

    real rho0 = rho[base];
    real exner0 = exner[base];
    real ustar = ust[column];
    real cpm = MYNN_MUL(CP, MYNN_ADD(1.0f, MYNN_MUL(0.84f, qv1[base])));
    real qflux = MYNN_DIV(qfx[column], rho0);
    real cflux = 0.0f;
    real surface_theta = MYNN_DIV(ts[column], exner0);
    real tflux = MYNN_SUB(MYNN_DIV(hfx[column], MYNN_MUL(rho0, cpm)),
                          MYNN_DIV(MYNN_MUL(xlvcp, cflux), exner0));
    real ustar3 = MYNN_MUL(MYNN_MUL(ustar, ustar), ustar);
    real inverse_l = -MYNN_DIV(MYNN_MUL(kgtr,
        MYNN_ADD(tflux, MYNN_MUL(MYNN_MUL(qflux, p608), surface_theta))),
        mynn_max2(ustar3, 1.0e-6f));
    real stability = MYNN_MUL(MYNN_MUL(0.5f, dz[base]), inverse_l);

    flqv[column] = qflux;
    flqc[column] = cflux;
    flq[column] = MYNN_ADD(qflux, cflux);
    flt[column] = tflux;
    fltv[column] = MYNN_ADD(tflux,
        MYNN_MUL(MYNN_MUL(qflux, p608), surface_theta));
    th_sfc[column] = surface_theta;
#if !defined(MYNN_GSD41)
    rmol[column] = inverse_l;
    real clamped = mynn_min2(mynn_max2(stability, -20.0f), 20.0f);
    zet[column] = clamped;
    pmz[column] = MYNN_SUB(mynn_phim(clamped), clamped);
    phh[column] = mynn_phih(clamped);
#else
    // GSD MYNN v4.1 (NOAA-EMC WRF 3.9 module_bl_mynn.F:4412-4419): 1/L is
    // the surface layer's, as handed in and handed back; z/L is not
    // clipped; the Kansas-type forms with cphm_st = cphh_st = 5 and
    // cphm_unst = cphh_unst = 16.
    (void)inverse_l;
    (void)stability;
    real rm = rmol_in[column];
    rmol[column] = rm;
    real z = MYNN_MUL(MYNN_MUL(0.5f, dz[base]), rm);
    zet[column] = z;
    if (z >= 0.0f) {
        pmz[column] = MYNN_ADD(1.0f, MYNN_MUL(MYNN_SUB(5.0f, 1.0f), z));
        phh[column] = MYNN_ADD(1.0f, MYNN_MUL(5.0f, z));
    } else {
        pmz[column] = MYNN_SUB(MYNN_DIV(1.0f, mynn_powf(
            MYNN_SUB(1.0f, MYNN_MUL(16.0f, z)), 0.25f)), z);
        phh[column] = MYNN_DIV(1.0f, sqrtf(
            MYNN_SUB(1.0f, MYNN_MUL(16.0f, z))));
    }
#endif
}

// module_bl_mynn.F:1223-1233, dheat_opt=1.  qke**1.5 and EXP go through the
// FP64-then-round pair this file uses for every other real**real; the CPU
// reference routes them onto the glibc transcriptions instead, and
// tests/test_mynn_pbl_driver_gpu.py measures what that costs.  The top level
// is never written by the Fortran loop and stays zero.
extern "C" __global__
void mynn_driver_diss_heat_columns(
    const real* __restrict__ el_raw, const real* __restrict__ qke_raw,
    const real* __restrict__ p_raw, real* __restrict__ diss_heat_raw,
    int nz, int ncol)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncol) return;
    MynnColumn<const real> el = {el_raw + column, (size_t)ncol};
    MynnColumn<const real> qke = {qke_raw + column, (size_t)ncol};
    MynnColumn<const real> p = {p_raw + column, (size_t)ncol};
    MynnColumn<real> diss_heat = {diss_heat_raw + column, (size_t)ncol};
    const real b1 = 24.0f;
    const int base = 0;
    for (int k = 0; k < nz - 1; ++k) {
        real blend = mynn_max2(
            MYNN_MUL(0.5f, MYNN_ADD(el[base + k], el[base + k + 1])), 1.0f);
        real value = MYNN_DIV(
            MYNN_DIV(MYNN_MUL(1.0f, mynn_powf(qke[base + k], 1.5f)),
                     MYNN_MUL(b1, blend)), CP);
#if !defined(MYNN_GSD41)
        value = mynn_min2(mynn_max2(value, 0.0f), 0.002f);
        diss_heat[base + k] = MYNN_MUL(value, mynn_expf_rn(
            -MYNN_DIV(10000.0f, mynn_max2(p[base + k], 1.0f))));
#else
        // GSD MYNN v4.1 module_bl_mynn.F:4600-4604: factor 0.5, held to
        // 0..2e-5 K/s, no pressure taper.
        (void)value;
        real heat = MYNN_DIV(MYNN_DIV(MYNN_MUL(0.5f,
            mynn_powf(qke[base + k], 1.5f)), MYNN_MUL(b1, blend)), CP);
        diss_heat[base + k] = mynn_min2(mynn_max2(heat, 0.0f), 0.00002f);
#endif
    }
    diss_heat[base + nz - 1] = 0.0f;
}

// module_bl_mynn.F:5358 retrieve_exchange_coeffs.  The surface level is set
// to zero, not computed, exactly as the Fortran does.
extern "C" __global__
void mynn_driver_exchange_columns(
    const real* __restrict__ dz, const real* __restrict__ dfm,
    const real* __restrict__ dfh, real* __restrict__ k_m,
    real* __restrict__ k_h, int nz, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int ld = count / nz;
    int k = index / ld;
    if (k == 0) {
        k_m[index] = 0.0f;
        k_h[index] = 0.0f;
        return;
    }
    real dzk = MYNN_MUL(0.5f, MYNN_ADD(dz[index], dz[index - ld]));
    k_m[index] = MYNN_MUL(dfm[index], dzk);
    k_h[index] = MYNN_MUL(dfh[index], dzk);
}

// ===========================================================================
// module_bl_mynn_wrapper.F:mynnedmf_wrapper_run.  This is the coupling layer
// WRF's PBL driver actually calls -- mynn_bl_driver is never invoked directly
// from module_pbl_driver.F -- and it owns two unit conversions that are easy
// to lose.  Getting either wrong is a silent O(qv) bias, not a crash.
// ===========================================================================

// :453-475.  WRF hands the PBL mixing ratios; MYNN wants specific values.
// The same (1 + qv) divides all four, so a caller cannot use a per-species
// denominator by accident.
extern "C" __global__
void mynn_wrapper_to_specific(
    const real* __restrict__ qv, const real* __restrict__ qc,
    const real* __restrict__ qi, const real* __restrict__ qs,
    real* __restrict__ sqv, real* __restrict__ sqc,
    real* __restrict__ sqi, real* __restrict__ sqs, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    real denominator = MYNN_ADD(1.0f, qv[index]);
    sqv[index] = MYNN_DIV(qv[index], denominator);
    sqc[index] = MYNN_DIV(qc[index], denominator);
    sqi[index] = MYNN_DIV(qi[index], denominator);
    sqs[index] = MYNN_DIV(qs[index], denominator);
}

// :587-607.  The moisture tendencies and the subgrid cloud water come back
// in specific units and are divided by (1 - sqv).  RTHBLTEN, RUBLTEN and
// RVBLTEN are NOT converted, and CLDFRA_BL is not either -- it is a fraction.
extern "C" __global__
void mynn_wrapper_from_specific(
    const real* __restrict__ sqv, real* __restrict__ rqvblten,
    real* __restrict__ rqcblten, real* __restrict__ rqiblten,
    real* __restrict__ qc_bl, real* __restrict__ qi_bl, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    real denominator = MYNN_SUB(1.0f, sqv[index]);
#if !defined(MYNN_GSD41)
    rqvblten[index] = MYNN_DIV(rqvblten[index], denominator);
    rqcblten[index] = MYNN_DIV(rqcblten[index], denominator);
    rqiblten[index] = MYNN_DIV(rqiblten[index], denominator);
#endif
#if !defined(MYNN_GSD41)
    qc_bl[index] = MYNN_DIV(qc_bl[index], denominator);
    qi_bl[index] = MYNN_DIV(qi_bl[index], denominator);
#else
    // GSD MYNN v4.1 writes its in-cloud QC_BL as computed
    // (module_bl_mynn.F:4692); the generation has no QI_BL.
    (void)qc_bl;
    (void)qi_bl;
#endif
}
#if defined(MYNN_GSD41)

// GSD MYNN v4.1 module_bl_mynn.F:4690-4713, at the write-back: subgrid cloud
// fraction may fall by at most 0.25*delt/ts_decay per call, ts_decay =
// MIN(1800 s, 3*dx/MAX(|V|, 1 m/s)) with the layer's wind; a decayed
// fraction under 0.005 clears both fields; QC_BL is floored at 1e-8 where
// the fraction exceeds 0.005.  cldfra_old is the field carried into the
// call (:4221).  Every element of the (ncol, nz) Fortran-order arrays.
extern "C" __global__
void mynn_gsd41_cloud_decay(
    real* __restrict__ cldfra_bl, real* __restrict__ qc_bl,
    const real* __restrict__ cldfra_old, const real* __restrict__ u,
    const real* __restrict__ v, const real* __restrict__ dx,
    const real* __restrict__ delt, int nz, int ncol)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ncol * nz) return;
    int column = index % ncol;
    real cf = cldfra_bl[index];
    real old = cldfra_old[index];
    if (cf < old) {
        real speed = sqrtf(MYNN_ADD(MYNN_MUL(u[index], u[index]),
                                    MYNN_MUL(v[index], v[index])));
        real ts_decay = mynn_min2(1800.0f, MYNN_DIV(MYNN_MUL(3.0f, dx[column]),
                                                    mynn_max2(speed, 1.0f)));
        cf = mynn_max2(cf, MYNN_SUB(old, MYNN_DIV(MYNN_MUL(0.25f,
                                                           delt[column]),
                                                  ts_decay)));
        cldfra_bl[index] = cf;
        if (cf < 0.005f) {
            cldfra_bl[index] = 0.0f;
            qc_bl[index] = 0.0f;
        }
    }
    if (qc_bl[index] < 1.0e-8f && cldfra_bl[index] > 0.005f)
        qc_bl[index] = 1.0e-8f;
}

// GSD MYNN v4.1 module_bl_mynn.F:4084-4096, :4236-4264: the PBL height
// reads theta-v of the liquid-water potential temperature, with the
// carried subgrid cloud (in-cloud QC_BL times CLDFRA_BL, split by
// (T-254)/15) standing in for resolved cloud only where none is present.
extern "C" __global__
void mynn_gsd41_thvl_columns(
    const real* __restrict__ th, const real* __restrict__ exner,
    const real* __restrict__ sqv, const real* __restrict__ sqc,
    const real* __restrict__ sqi, const real* __restrict__ tk,
    const real* __restrict__ qc_bl, const real* __restrict__ cldfra_bl,
    real* __restrict__ thvl, int flag_qi, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const real xlvcp = MYNN_DIV(XLV, CP);
    const real xlscp = MYNN_DIV(MYNN_ADD(XLV, 3.50e5f), CP);
    real c = sqc[index];
    real i = flag_qi ? sqi[index] : 0.0f;
    real sqc9 = c, sqi9 = i;
    bool clear = flag_qi ? (c < 1.0e-6f && i < 1.0e-8f) : (c < 1.0e-6f);
    if (clear && cldfra_bl[index] > 0.001f) {
        real liq = MYNN_DIV(MYNN_SUB(tk[index], 254.0f), 15.0f);
        liq = mynn_min2(1.0f, mynn_max2(0.0f, liq));
        sqc9 = MYNN_MUL(MYNN_MUL(qc_bl[index], liq), cldfra_bl[index]);
        sqi9 = MYNN_MUL(MYNN_MUL(qc_bl[index], MYNN_SUB(1.0f, liq)),
                        cldfra_bl[index]);
    }
    real thlsg = MYNN_SUB(MYNN_SUB(th[index],
        MYNN_MUL(MYNN_DIV(xlvcp, exner[index]), sqc9)),
        MYNN_MUL(MYNN_DIV(xlscp, exner[index]), sqi9));
    thvl[index] = MYNN_MUL(thlsg, MYNN_ADD(1.0f, MYNN_MUL(0.61f, sqv[index])));
}
#endif
