// WOOF scalar float32 atan. Reuses the existing Noah-MP arithmetic.
// Coefficients are supplied by the hash-validated PX constant asset.
__device__ float woof_atan_scalar(float x)
{
    int hx = (int)__float_as_uint(x);
    int ix = hx & 0x7fffffff;
    if (ix >= 0x4c000000) {
        if (ix > 0x7f800000) return __fadd_rn(x, x);
        if (hx > 0) return __fadd_rn(WOOF_ATAN_HI[3], WOOF_ATAN_LO[3]);
        return __fsub_rn(-WOOF_ATAN_HI[3], WOOF_ATAN_LO[3]);
    }
    int id;
    if (ix < 0x3ee00000) {
        if (ix < 0x31000000) return x;
        id = -1;
    } else {
        x = fabsf(x);
        if (ix < 0x3f980000) {
            if (ix < 0x3f300000) {
                id = 0;
                x = __fdiv_rn(__fsub_rn(__fmul_rn(2.0f, x), 1.0f), __fadd_rn(2.0f, x));
            } else {
                id = 1;
                x = __fdiv_rn(__fsub_rn(x, 1.0f), __fadd_rn(x, 1.0f));
            }
        } else if (ix < 0x401c0000) {
            id = 2;
            x = __fdiv_rn(__fsub_rn(x, 1.5f), __fadd_rn(1.0f, __fmul_rn(1.5f, x)));
        } else {
            id = 3;
            x = __fdiv_rn(-1.0f, x);
        }
    }
    float z = __fmul_rn(x, x);
    float w = __fmul_rn(z, z);
    float s1 = __fmul_rn(z, __fadd_rn(WOOF_ATAN_T[0], __fmul_rn(w, __fadd_rn(WOOF_ATAN_T[2],
               __fmul_rn(w, __fadd_rn(WOOF_ATAN_T[4], __fmul_rn(w, __fadd_rn(WOOF_ATAN_T[6],
               __fmul_rn(w, __fadd_rn(WOOF_ATAN_T[8], __fmul_rn(w, WOOF_ATAN_T[10])))))))))));
    float s2 = __fmul_rn(w, __fadd_rn(WOOF_ATAN_T[1], __fmul_rn(w, __fadd_rn(WOOF_ATAN_T[3],
               __fmul_rn(w, __fadd_rn(WOOF_ATAN_T[5], __fmul_rn(w, __fadd_rn(WOOF_ATAN_T[7],
               __fmul_rn(w, WOOF_ATAN_T[9])))))))));
    if (id < 0) return __fsub_rn(x, __fmul_rn(x, __fadd_rn(s1, s2)));
    z = __fsub_rn(WOOF_ATAN_HI[id], __fsub_rn(__fsub_rn(__fmul_rn(x, __fadd_rn(s1, s2)), WOOF_ATAN_LO[id]), x));
    return (hx < 0) ? -z : z;
}
