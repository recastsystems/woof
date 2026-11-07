// Existing device scalar expm1/tanh arithmetic from ruc.cu.
__device__ __forceinline__ real km5_scale_exponent(real y, int k)
{
    return __uint_as_float(__float_as_uint(y) + ((unsigned)k << 23));
}

__device__ real km5_expm1(real x)
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
            if (magnitude > 0x7F800000u) return FADD(x, x);
            if (magnitude == 0x7F800000u) return sign == 0u ? x : -1.0f;
            if (x > 8.8721679688e01f) return __int_as_float(0x7F800000);
        }
        if (sign != 0u) return FSUB(tiny, 1.0f);
    }
    int k;
    real correction;
    if (magnitude > 0x3EB17218u) {               // |x| > 0.5*ln2
        real hi, lo;
        if (magnitude < 0x3F851592u) {           // |x| < 1.5*ln2
            if (sign == 0u) {
                hi = FSUB(x, ln2_hi); lo = ln2_lo; k = 1;
            } else {
                hi = FADD(x, ln2_hi); lo = -ln2_lo; k = -1;
            }
        } else {
            k = (int)FADD(FMUL(invln2, x), sign == 0u ? 0.5f : -0.5f);
            real scale = (real)k;
            hi = FSUB(x, FMUL(scale, ln2_hi));
            lo = FMUL(scale, ln2_lo);
        }
        x = FSUB(hi, lo);
        correction = FSUB(FSUB(hi, x), lo);
    } else if (magnitude < 0x33000000u) {        // |x| < 2**-25
        return x;
    } else {
        k = 0;
        correction = 0.0f;
    }

    real hfx = FMUL(0.5f, x);
    real hxs = FMUL(x, hfx);
    real r1 = FADD(1.0f, FMUL(hxs, FADD(q1, FMUL(hxs,
        FADD(q2, FMUL(hxs, FADD(q3, FMUL(hxs,
            FADD(q4, FMUL(hxs, q5))))))))));
    real t = FSUB(3.0f, FMUL(r1, hfx));
    real e = FMUL(hxs, FDIV(FSUB(r1, t), FSUB(6.0f, FMUL(x, t))));
    if (k == 0) return FSUB(x, FSUB(FMUL(x, e), hxs));
    e = FSUB(FMUL(x, FSUB(e, correction)), correction);
    e = FSUB(e, hxs);
    if (k == -1) return FSUB(FMUL(0.5f, FSUB(x, e)), 0.5f);
    if (k == 1) {
        if (x < -0.25f)
            return FMUL(-2.0f, FSUB(e, FADD(x, 0.5f)));
        return FADD(1.0f, FMUL(2.0f, FSUB(x, e)));
    }
    real y;
    if (k <= -2 || k > 56) {
        y = FSUB(1.0f, FSUB(e, x));
        y = km5_scale_exponent(y, k);
        return FSUB(y, 1.0f);
    }
    if (k < 23) {
        t = __uint_as_float(0x3F800000u - (0x1000000u >> k));
        y = FSUB(t, FSUB(e, x));
    } else {
        t = __uint_as_float((unsigned)(0x7F - k) << 23);
        y = FSUB(x, FADD(e, t));
        y = FADD(y, 1.0f);
    }
    return km5_scale_exponent(y, k);
}

// tanhf in float32 on km5_expm1: 1 - 2/(expm1(2|x|)+2) at |x| >= 1,
// -t/(t+2) with t = expm1(-2|x|) below, x*(1+x) under 2**-55 and 1 beyond
// 22.  The same routine as MYNN's mynn_tanhf.  The earlier body ran this
// reduction on a float64 expm1 rounded once, which is a different expm1f:
// the RUC oracle measured it 1-2 ULP off the reference TANH in the new-snow
// density (module_sf_ruclsm.F:1520-1521) and so in rhosnf and snowfallac.
// gpuwm.core.ruc._f32_tanh is the host mirror.
__device__ __forceinline__
real km5_tanh_strict(real x)
{
    const real tiny = 1.0e-30f;
    unsigned word = __float_as_uint(x);
    unsigned magnitude = word & 0x7FFFFFFFu;
    real z;
    if (magnitude >= 0x7F800000u)                // inf or NaN: one/x +- one
        return (word & 0x80000000u) == 0u ? FADD(FDIV(1.0f, x), 1.0f)
                                          : FSUB(FDIV(1.0f, x), 1.0f);
    if (magnitude < 0x41B00000u) {               // |x| < 22
        if (magnitude < 0x24000000u)             // |x| < 2**-55
            return FMUL(x, FADD(1.0f, x));
        real ax = __uint_as_float(magnitude);
        if (magnitude >= 0x3F800000u) {          // |x| >= 1
            real t = km5_expm1(FMUL(2.0f, ax));
            z = FSUB(1.0f, FDIV(2.0f, FADD(t, 2.0f)));
        } else {
            real t = km5_expm1(FMUL(-2.0f, ax));
            z = FDIV(-t, FADD(t, 2.0f));
        }
    } else {
        z = FSUB(1.0f, tiny);
    }
    return (word & 0x80000000u) == 0u ? z : -z;
}

__device__ float km5_tanh(float x) {
#if defined(GPUWM_WRF_EXACT) || defined(GPUWM_WRF_EXACT_C_DIFFUSION)
 return km5_tanh_strict(x);
#else
 return tanhf(x);
#endif
}
