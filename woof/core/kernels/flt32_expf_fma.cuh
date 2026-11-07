// WOOF's own float32 exp, contracted form: gfk_exp_fma.
//
// Requires glibc_flt32.cuh ahead of it in the assembled source (the loader's
// _EXTRA_HEADERS lists both, in that order).  It holds no table of its own:
// the scale 2**(k/32) comes from that header's gfk_exp2_core called with a
// zero polynomial (which returns its table scale exactly), and it reuses the
// polynomial constants GFK_EXP2F_P0..P2, GFK_EXP2F_SHIFT and gfk_d2f_rn.
//
// WHY A SECOND FORM.  gfk_exp evaluates every a*b+c of the algorithm as a
// separate multiply and add.  The reference a WRF column oracle calls on an
// x86-64 host with FMA (the reference
// host) is the C library's run-time-selected FMA build of expf, whose
// multiply-adds are fused.  The two forms differ at exactly two float32
// inputs over the whole line -- x = 0x4202422F (32.5646324) and
// x = 0xC27C65D9 (-63.0994606), 1 ULP each -- measured by
// tools/myjpbl_wrf461_oracle/libm_sweep.py over all 2**32 inputs on x86-64.
// gfk_exp_fma fuses the same five sites (the shift add, the reduced
// argument, both polynomial halves and their combination) with explicit
// __fma_rn, so it is unaffected by --fmad and matches the reference at every
// input (same sweep).  The MYJ PBL calls it; kernels/mynn_pbl.cu carries the
// same contracted form privately as mynn_glibc_expf.

__device__ float gfk_exp_fma(float x)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int abstop = (ix >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) {
        if (ix == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return __fadd_rn(x, x);
        if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
        if (x < -__int_as_float(0x42cff1b4)) return 0.0f;
    }
    const double inv_ln2_n = 0x1.71547652b82fep+0 * 32.0;
    double xd = (double)x;
    double kd = __fma_rn(inv_ln2_n, xd, GFK_EXP2F_SHIFT);
    kd = __dsub_rn(kd, GFK_EXP2F_SHIFT);
    double r = __fma_rn(inv_ln2_n, xd, -kd);
    // kd is an integer-valued double, so kd + SHIFT inside the core
    // rebuilds the fused shift sum above bit for bit (the same table index
    // and exponent bits); with a zero
    // polynomial the core returns 1.0 * scale, the scale exactly.
    double s = gfk_exp2_core(kd, GFK_EXP2F_SHIFT, 0.0, 0.0, 0.0, 0ULL);
    double z = __fma_rn(GFK_EXP2F_P0 / 32.0 / 32.0 / 32.0, r,
                        GFK_EXP2F_P1 / 32.0 / 32.0);
    double r2 = __dmul_rn(r, r);
    double y = __fma_rn(GFK_EXP2F_P2 / 32.0, r, 1.0);
    y = __dmul_rn(__fma_rn(z, r2, y), s);
    return gfk_d2f_rn(y);
}
