// WRF cu6 column storage and rounded arithmetic. The transcendental calls
// use WOOF's own routines supplied by the kernel loader.
#pragma once
#ifndef TK_TRACE
#define TK_TRACE(line, value)
#endif

__device__ inline float tk_add(float a, float b) { return __fadd_rn(a, b); }
__device__ inline float tk_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ inline float tk_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ inline float tk_div(float a, float b) { return __fdiv_rn(a, b); }
template<class T> __device__ inline T tk_min(T a, T b) { return a < b ? a : b; }
template<class T> __device__ inline T tk_max(T a, T b) { return a > b ? a : b; }
__device__ inline float tk_abs(float a) { return fabsf(a); }
__device__ inline int tk_abs(int a) { return a < 0 ? -a : a; }
template<int N> __device__ inline float tk_powi(float a) {
    if constexpr (N == 1) return a;
    else if constexpr (N % 2 == 0) {
        float b = tk_powi<N / 2>(a);
        return tk_mul(b, b);
    } else return tk_mul(a, tk_powi<N - 1>(a));
}

// Each thread is one Fortran column. Array coordinates stay one-based.
// The second dimension is vertical and the storage is level-major, so
// adjacent CUDA lanes use adjacent words at every level.
template<class T> struct TkArray {
    T* data;
    int stride;
    int first;
    int count;
    int* status;
    __device__ T& at(int index) const {
        if (index < 0 || index >= count) {
            // F:2282/2289 can write at jk-1=0 when the original cloud top
            // reaches level 3. Refuse that undefined column before its
            // tendencies can enter the model. Keep every access in bounds.
            *status = -1;
            return data[0];
        }
        return data[index * stride];
    }
    __device__ T& operator()(int i) const { return at(i - 1); }
    __device__ T& operator()(int i, int k) const {
        return at((k - 1) * first + i - 1);
    }
    __device__ TkArray<T> slice(int i) const {
        return {data + (i - 1) * stride, stride, first, count - i + 1, status};
    }
};
template<class A, class B> struct TkSame { static constexpr bool value = false; };
template<class A> struct TkSame<A, A> { static constexpr bool value = true; };

// Passed by value to each routine. A child starts after its parent's live
// arrays; sequential calls reuse the same child region. No local column
// arrays or per-thread fixed vertical ceiling are compiled.
struct TkArena {
    float* f;
    int* i;
    bool* b;
    int nf, ni, nb, stride;
    int* status;
    int* activity;
    bool force_continue;
    // True for a column whose F:588 exp runs in a four-wide libmvec step
    // (tk_lane_exp below). Trailing, so an initializer that stops before it
    // leaves it false: the scalar expf of an -O0 or unvectorized build.
    bool mvec;
    bool old_tan;
    template<class T> __device__ TkArray<T> take(int n, int first = 1) {
        if constexpr (TkSame<T, float>::value) {
            TkArray<T> a = {f + nf * stride, stride, first, n, status};
            nf += n;
            return a;
        } else if constexpr (TkSame<T, int>::value) {
            TkArray<T> a = {i + ni * stride, stride, first, n, status};
            ni += n;
            return a;
        } else {
            TkArray<T> a = {b + nb * stride, stride, first, n, status};
            nb += n;
            return a;
        }
    }
};

// glibc 2.39 k_tanf.c and s_tanf.c, Sun fdlibm, transcribed in source
// order. Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business. Permission to
// use, copy, modify, and distribute this software is freely granted,
// provided that this notice is preserved.
// The Arm MIT reduction and tables are shared with glibc_trig_flt32.cuh.
__device__ inline float tk_tanf239(float input) {
    const float T[13] = {3.3333334327e-1f,1.3333334029e-1f,5.3968254477e-2f,
        2.1869488060e-2f,8.8632395491e-3f,3.5920790397e-3f,1.4562094584e-3f,
        5.8804126456e-4f,2.4646313977e-4f,7.8179444245e-5f,7.1407252108e-5f,
        -1.8558637748e-5f,2.5907305826e-5f};
    float x=input, y=0.0f;
    unsigned ix=__float_as_uint(x)&0x7fffffffU;
    if (ix>=0x7f800000U) return tk_sub(x,x);
    int iy=1;
    if (ix>0x3f490fdaU) {
        double dx=(double)x;
        int n;
        if (((ix>>20)&0x7ffU)<0x42fU) {
            double r=__dmul_rn(dx,gt_sincos_table[0].hpi_inv);
            n=((int)r+0x800000)>>24;
            dx=__dsub_rn(dx,__dmul_rn((double)n,gt_sincos_table[0].hpi));
        } else {
            dx=gt_reduce_large(__float_as_uint(x),&n);
            if (__float_as_uint(x)>>31) dx=-dx;
        }
        x=__double2float_rn(dx);
        y=__double2float_rn(__dsub_rn(dx,(double)x));
        iy=1-((n&1)<<1);
    }
    int hx=(int)__float_as_uint(x);
    ix=(unsigned)hx&0x7fffffffU;
    if (ix<0x39000000U) {
        if ((ix|(iy+1))==0) return tk_div(1.0f,fabsf(x));
        return iy==1 ? x : tk_div(-1.0f,x);
    }
    float z,r,v,w,s;
    const float pio4=7.8539812565e-1f, pio4lo=3.7748947079e-8f;
    if (ix>=0x3f2ca140U) {
        if (hx<0) {x=-x; y=-y;}
        z=tk_sub(pio4,x); w=tk_sub(pio4lo,y);
        x=tk_add(z,w); y=0.0f;
        if (fabsf(x)<0x1p-13f)
            return tk_mul((float)((1-((hx>>30)&2))*iy),
                tk_sub(1.0f,tk_mul((float)(2*iy),x)));
    }
    z=tk_mul(x,x); w=tk_mul(z,z);
    r=tk_add(T[1],tk_mul(w,tk_add(T[3],tk_mul(w,tk_add(T[5],tk_mul(w,
        tk_add(T[7],tk_mul(w,tk_add(T[9],tk_mul(w,T[11]))))))))));
    v=tk_mul(z,tk_add(T[2],tk_mul(w,tk_add(T[4],tk_mul(w,tk_add(T[6],tk_mul(w,
        tk_add(T[8],tk_mul(w,tk_add(T[10],tk_mul(w,T[12])))))))))));
    s=tk_mul(z,x);
    r=tk_add(y,tk_mul(z,tk_add(tk_mul(s,tk_add(r,v)),y)));
    r=tk_add(r,tk_mul(T[0],s)); w=tk_add(x,r);
    if (ix>=0x3f2ca140U) {
        v=(float)iy;
        return tk_mul((float)(1-((hx>>30)&2)),tk_sub(v,tk_mul(2.0f,
            tk_sub(x,tk_sub(tk_div(tk_mul(w,w),tk_add(w,v)),r)))));
    }
    if (iy==1) return w;
    z=__uint_as_float(__float_as_uint(w)&0xfffff000U);
    v=tk_sub(r,tk_sub(z,x));
    float a=tk_div(-1.0f,w), t=__uint_as_float(__float_as_uint(a)&0xfffff000U);
    s=tk_add(1.0f,tk_mul(t,z));
    return tk_add(t,tk_mul(a,tk_add(s,tk_mul(t,v))));
}

__device__ inline float tk_tan(float x, TkArena arena) {
    return arena.old_tan ? tk_tanf239(x) : glibc_tanf(x);
}

// glibc 2.43's libmvec expf, _ZGVbN4v_expf_sse4
// (sysdeps/x86_64/fpu/multiarch/svml_s_expf4_core_sse4.S, constants from
// sysdeps/x86_64/fpu/svml_s_expf_data.S), operation for operation; the lanes
// it flags out of range take the scalar expf, as its own fallback does.
// WRF 4.6.1 built with its stock gfortran flags (configure option 34, -O2
// -ftree-vectorize) by gfortran 15.2 vectorizes tiecnv's conversion loop
// (F:575-593, tlucua inlined at F:588) four columns at a time and binds that
// exp to this routine; the last ite-its+1 mod 4 columns of the row keep the
// scalar expf (gfk_exp). Equal to the host libmvec on every non-NaN float32
// (4,278,190,082 inputs, glibc 2.43, 2026-10-29), and it differs from the
// scalar expf on 3.6 per cent of the arguments F:588 can produce.
__device__ inline float tk_mvec_expf(float x) {
    const unsigned ax = __float_as_uint(x) & 0x7fffffffu;
    if ((int)ax > (int)0x42aeac4fu) return gfk_exp(x);
    const float shifter = __uint_as_float(0x4b400000u);
    const float m = __fadd_rn(__fmul_rn(x, __uint_as_float(0x3fb8aa3bu)), shifter);
    const float n = __fsub_rn(m, shifter);
    float r = __fsub_rn(x, __fmul_rn(n, __uint_as_float(0x3f317200u)));
    r = __fsub_rn(r, __fmul_rn(n, __uint_as_float(0x35bfbe8eu)));
    float p = __fadd_rn(__fmul_rn(__uint_as_float(0x3c07d9feu), r),
                        __uint_as_float(0x3d2b8392u));
    p = __fadd_rn(__fmul_rn(p, r), __uint_as_float(0x3e2aacacu));
    p = __fadd_rn(__fmul_rn(p, r), __uint_as_float(0x3effff34u));
    p = __fadd_rn(__fmul_rn(p, r), __uint_as_float(0x3f7ffffeu));
    const float e = __fadd_rn(__fmul_rn(r, p), __uint_as_float(0x3f800000u));
    return __fmul_rn(__uint_as_float((__float_as_uint(m) + 0x7fu) << 23), e);
}

// The exp of F:588 for this column: libmvec in a vectorized lane, glibc's
// scalar expf otherwise.
__device__ inline float tk_lane_exp(TkArena arena, float x) {
    return arena.mvec ? tk_mvec_expf(x) : gfk_exp(x);
}
