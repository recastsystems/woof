"""Pinned RRFS-MPAS MYNN-EDMF CUDA backend.

Numerical text is generated from c942418, including the complete MPAS wrapper.
Shared device maths is supplied by the caller. Forecast acceptance is separate.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics
from typing import Mapping

# Strict compiles optimised: --fmad=false keeps every operation separately rounded and --prec-div/--prec-sqrt keep
# IEEE quotients and roots, so ptxas optimisation changes no word (module replay of real hrrrv5 calls, strict, old
# -O0 image vs this one: 0 different words, rrfs/mynn-speed 2026-10-28). The -O0 image ran the PBL at 242 ms of a
# 322 ms strict step.
# Split compilation (rrfs/speed2-mynn-edmf, 2026-10-28): the image holds one kernel per lane width (LANE_WIDTHS), and
# NVRTC + ptxas otherwise build them on one core (fast 70-126 s on a loaded RTX 5090 host, strict several minutes). Split
# across the host's cores: fast 35 s. It changes only how the compiler schedules its work; every word is the same
# (module replays, fast and strict).
CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true",
                "--split-compile=0", "--ptxas-options=--split-compile=0")

ARRAY_SUPPORT = r'''
// Stack frame (rrfs/mynn-speed, 2026-10-28): the generator inlines the two once-per-column entry routines
// (f_mynnedmf_driver, f_mynnedmf) into the kernel, so their ~100-array by-value parameter blocks leave the stack.
// The kernel's frame is 2,640 B fast / 2,816 B strict, under the 4 KiB at which device_memory_release gives the
// local-memory store back after every PBL phase (two device syncs a step: hrrrv5 x1.2562 measured 30 resets and
// 29 regrows in 30 steps with the 4,432 B frame the inlined operations alone gave).
// Execution shape (rrfs/mynn-speed, 2026-10-28). These operation wrappers were isolated __noinline__ calls, so
// each of the module's 5,079 binary32 operations was a device function call. They are inline now. Strict numerics
// compiles with --fmad=false, so an inlined operation still rounds on its own. Fast numerics compiles with
// --fmad=true and uses the round-to-nearest intrinsics, which are never contracted into an FMA, so every operation
// keeps the separately rounded word the isolated call gave (fast and strict module replays: 0 different words).
#if defined(HEX_NUMERICS_FAST) && HEX_NUMERICS_FAST
#define MYNN_R6_ADD(a,b) __fadd_rn(a,b)
#define MYNN_R6_SUB(a,b) __fsub_rn(a,b)
#define MYNN_R6_MUL(a,b) __fmul_rn(a,b)
#define MYNN_R6_DIV(a,b) __fdiv_rn(a,b)
#else
#define MYNN_R6_ADD(a,b) mpas_add(a,b)
#define MYNN_R6_SUB(a,b) mpas_sub(a,b)
#define MYNN_R6_MUL(a,b) mpas_mul(a,b)
#define MYNN_R6_DIV(a,b) mpas_div(a,b)
#endif
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve strict GNU operation NaN payload/sign alongside shared round-to-nearest arithmetic.
__device__ __forceinline__ float r6_neg(float a) { return __uint_as_float(__float_as_uint(a)^0x80000000u); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve strict GNU operation NaN payload/sign alongside shared round-to-nearest arithmetic.
__device__ __forceinline__ double r6_neg(double a) { return __longlong_as_double(__double_as_longlong(a)^0x8000000000000000ULL); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve strict GNU operation NaN payload/sign alongside shared round-to-nearest arithmetic.
// Fast numerics returns the operation's own result: the GNU x86 NaN payload and sign rules below are strict-only,
// as on the v8.4.2 MYNN fast route (bl_mynn_v842 mynn_real64/real32). Finite words are unchanged (fast module
// replay of real hrrrv5 calls: 0 different words); only a NaN result's payload bits could differ.
__device__ __forceinline__ float r6_nan(float a,float b,float r) {
#if defined(HEX_NUMERICS_FAST) && HEX_NUMERICS_FAST
 return r;
#endif
 unsigned ua=__float_as_uint(a),ub=__float_as_uint(b),ur=__float_as_uint(r);
 if ((ua&0x7fffffffu)>0x7f800000u) return __uint_as_float(ua|0x00400000u);
 if ((ub&0x7fffffffu)>0x7f800000u) return __uint_as_float(ub|0x00400000u);
 if ((ur&0x7fffffffu)>0x7f800000u) return __uint_as_float(0xffc00000u);
 return r;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve strict GNU operation NaN payload/sign alongside shared round-to-nearest arithmetic.
__device__ __forceinline__ double r6_nan(double a,double b,double r) {
#if defined(HEX_NUMERICS_FAST) && HEX_NUMERICS_FAST
 return r;
#endif
 unsigned long long ua=__double_as_longlong(a),ub=__double_as_longlong(b),ur=__double_as_longlong(r);
 if ((ua&0x7fffffffffffffffULL)>0x7ff0000000000000ULL) return __longlong_as_double(ua|0x0008000000000000ULL);
 if ((ub&0x7fffffffffffffffULL)>0x7ff0000000000000ULL) return __longlong_as_double(ub|0x0008000000000000ULL);
 if ((ur&0x7fffffffffffffffULL)>0x7ff0000000000000ULL) return __longlong_as_double(0xfff8000000000000ULL);
 return r;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve strict GNU operation NaN payload/sign alongside shared round-to-nearest arithmetic.
__device__ __forceinline__ float r6_add(float a,float b) { return r6_nan(a,b,MYNN_R6_ADD(a,b)); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve strict GNU operation NaN payload/sign alongside shared round-to-nearest arithmetic.
__device__ __forceinline__ float r6_sub(float a,float b) { return r6_nan(a,b,MYNN_R6_SUB(a,b)); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve strict GNU operation NaN payload/sign alongside shared round-to-nearest arithmetic.
__device__ __forceinline__ float r6_mul(float a,float b) { return r6_nan(a,b,MYNN_R6_MUL(a,b)); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve strict GNU operation NaN payload/sign alongside shared round-to-nearest arithmetic.
__device__ __forceinline__ float r6_div(float a,float b) { return r6_nan(a,b,MYNN_R6_DIV(a,b)); }

template<int R> struct MynnIndices { int value[R]; };
struct MynnRange { int lo, hi; };
template<class I> struct MynnIsRange { static constexpr int value = 0; };
template<> struct MynnIsRange<MynnRange> { static constexpr int value = 1; };
struct MynnArena {
    unsigned char* data;
    unsigned long long offset;
    // Warp tiles (rrfs/mynn-speed): a warp's columns share one tile, each logical 4-byte element holds one word
    // per column (width = columns per warp, lane = this column's slot), so a warp's access to element e of an
    // automatic array is one 128-byte line instead of 32 lines 110 KB apart. With lane groups
    // (rrfs/speed2-mynn-edmf) a warp holds 32/lane_width columns and a group's lanes on consecutive elements
    // still read one line. The logical offsets, and so every word (undefined reads included), are unchanged.
    int width = 1;
    int lane = 0;
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ void* allocate(unsigned long long bytes, int itemsize) {
        offset = (offset + 7ULL) & ~7ULL;
        void* p = data + offset*width + lane*itemsize;
        offset += bytes;
        return p;
    }
};
struct MynnFrame {
    MynnArena& arena;
    unsigned long long saved;
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ MynnFrame(MynnArena& a) : arena(a), saved(a.offset) {}
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ ~MynnFrame() { arena.offset = saved; }
};
template<class T, int R> struct MynnArray {
    T* data = nullptr;
    int lo[R], hi[R], stride[R];
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ bool present() const { return data != nullptr; }
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ int extent(int d) const { return hi[d] - lo[d] + 1; }
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ int size() const {
        int n = 1;
        for (int d=0; d<R; ++d) n *= extent(d);
        return n;
    }
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ void rebase(MynnIndices<R> lower) {
        for (int d=0; d<R; ++d) {
            int n = extent(d);
            lo[d] = lower.value[d]; hi[d] = lo[d]+n-1;
        }
    }
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ T& linear(int index) const {
        int offset = 0;
        for (int d=0; d<R; ++d) {
            offset += (index % extent(d))*stride[d];
            index /= extent(d);
        }
        return data[offset];
    }
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    template<class... I> __device__ T& operator()(I... indices) const {
        static_assert(sizeof...(I) == R, "Fortran array rank");
        int ix[] = {int(indices)...};
        int offset = 0;
        for (int d=0; d<R; ++d) offset += (ix[d]-lo[d])*stride[d];
        return data[offset];
    }
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    template<int N> __device__ void select(MynnArray<T,N>& out, int& d, int& od, int& offset, MynnRange r) const {
        offset += (r.lo-lo[d])*stride[d];
        out.lo[od] = 1; out.hi[od] = r.hi-r.lo+1;
        out.stride[od++] = stride[d++];
    }
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    template<int N> __device__ void select(MynnArray<T,N>& out, int& d, int& od, int& offset, int i) const {
        offset += (i-lo[d])*stride[d++];
    }
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    template<class... I> __device__ auto slice(I... indices) const {
        constexpr int N = (MynnIsRange<I>::value + ... + 0);
        static_assert(N > 0, "Array section must retain a dimension");
        MynnArray<T,N> out;
        int d=0, od=0, offset=0;
        (select(out, d, od, offset, indices), ...);
        out.data = data + offset;
        return out;
    }
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    template<int N> __device__ MynnArray<T,N> sequence(MynnIndices<N> lower,MynnIndices<N> upper) const {
        MynnArray<T,N> out;
        out.data=data;
        int step=1;
        for (int d=0; d<N; ++d) {
            out.lo[d]=lower.value[d]; out.hi[d]=upper.value[d];
            out.stride[d]=step; step *= out.extent(d);
        }
        return out;
    }
};
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
// TILE (columns per warp tile) is a compile-time constant of the lane width (rrfs/speed2-mynn-edmf): a run-time
// width made every automatic array's stride a run-time value and the kernel frame grew past 4 KiB.
template<class T,int R,int TILE> __device__ MynnArray<T,R> mynn_array(MynnArena& arena, MynnIndices<R> lo, MynnIndices<R> hi) {
    static_assert(sizeof(T) == 4, "MYNN warp tiles hold four-byte elements");
    MynnArray<T,R> result;
    int count=1;
    for (int d=0; d<R; ++d) {
        result.lo[d]=lo.value[d]; result.hi[d]=hi.value[d];
        result.stride[d]=count*TILE;
        count *= result.extent(d);
    }
    result.data = static_cast<T*>(arena.allocate(sizeof(T)*count, sizeof(T)));
    return result;
}
struct MynnText {
    const char* text = "";
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ MynnText() = default;
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ MynnText(const char* s) : text(s) {}
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ MynnText& operator=(const char* s) { text=s; return *this; }
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ bool operator==(const MynnText& b) const {
        int i=0;
        while (text[i] && text[i] == b.text[i]) ++i;
        return text[i] == b.text[i];
    }
};
template<class T> struct MynnOptional {
    T value{};
    bool supplied = false;
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ MynnOptional() = default;
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ MynnOptional(T v) : value(v), supplied(true) {}
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ operator T() const { return value; }
    // RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; represent original array bounds, sections and optional presence.
    __device__ bool present() const { return supplied; }
};
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
template<class T> __device__ T mynn_min(T a,T b) { return a<b ? a:b; }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
template<class T> __device__ T mynn_max(T a,T b) { return a>b ? a:b; }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
template<class T,int R> __device__ T mynn_minval(MynnArray<T,R> a) {
    T value=a.linear(0);
    for (int i=1; i<a.size(); ++i) value=mynn_min(value,a.linear(i));
    return value;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
template<class T,int R> __device__ T mynn_maxval(MynnArray<T,R> a) {
    T value=a.linear(0);
    for (int i=1; i<a.size(); ++i) value=mynn_max(value,a.linear(i));
    return value;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; MINLOC first minimum and Fortran indexing.
template<class F> __device__ int mynn_minloc(int count,F function) {
    int index=0;
    auto value=function(0);
    for (int i=1; i<count; ++i) {
        auto next=function(i);
        if (next<value) { value=next; index=i; }
    }
    return index+1;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ double mynn_dadd(double a,double b) { return r6_nan(a,b,__dadd_rn(a,b)); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ double mynn_dsub(double a,double b) { return r6_nan(a,b,__dsub_rn(a,b)); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ double mynn_dmul(double a,double b) { return r6_nan(a,b,__dmul_rn(a,b)); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ double mynn_ddiv(double a,double b) { return r6_nan(a,b,__ddiv_rn(a,b)); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ float mynn_sqrt(float a) { return r6_nan(a,0.f,mpas_sqrt(a)); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ double mynn_sqrt(double a) { return r6_nan(a,0.,__dsqrt_rn(a)); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ float mynn_abs(float a) { return mpas_abs(a); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ double mynn_abs(double a) { return fabs(a); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ int mynn_abs(int a) { return a<0 ? -a:a; }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ float mynn_sign(float a,float b) { return copysignf(mynn_abs(a),b); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2617-3210; preserve double SIGN kind.
__device__ double mynn_sign(double a,double b) { return copysign(mynn_abs(a),b); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ int mynn_mod(int a,int b) { return a%b; }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ int mynn_ceiling(float a) { return __float2int_ru(a); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ int mynn_floor(float a) { return __float2int_rd(a); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
__device__ int mynn_nint(float a) { return (int)(a<0 ? r6_sub(a,0.5f):r6_add(a,0.5f)); }
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
template<int N> __device__ float mynn_ipow(float x) {
    static_assert(N==2 || N==3, "Pinned MYNN integer powers");
    float xx=r6_mul(x,x);
    if constexpr(N==2) return xx;
    return r6_mul(xx,x);
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-8820; preserve intrinsic/operator kind and operation order.
template<int N> __device__ double mynn_ipow(double x) {
    static_assert(N==2 || N==3, "Pinned MYNN integer powers");
    double xx=r6_nan(x,x,__dmul_rn(x,x));
    if constexpr(N==2) return xx;
    return r6_nan(xx,x,__dmul_rn(xx,x));
}
'''

# BEGIN GENERATED RRFS MYNN-EDMF
MODULE_CUDA = r'''
// Generated from pinned RRFS MYNN-EDMF c942418; see translation receipt.
template<int MYNN_LANES> struct MynnColumn {
MynnArena& arena;
int active_i = -1; int active_j = -1;
// Lane group of this column (lane_plan.py, rrfs/speed2-mynn-edmf): lane_width lanes run the column's scalar code
// together; split loops and array assignments give iteration it to lane it % lane_width. lane_width 1 is one
// thread per column. The width is a template argument: with it (and so the warp tile width and every automatic
// array's stride) known at compile time the kernel frame stays under 4 KiB (a run-time width spilled the strides:
// 4,544 B fast, over the 4 KiB at which device_memory_release gives the local-memory store back every phase).
static constexpr int lane_width = MYNN_LANES; static constexpr int mynn_tile = 32 / MYNN_LANES;
int lane_sub = 0; unsigned lane_mask = 0u;
__device__ __forceinline__ void mynn_lane_sync() const { __syncwarp(lane_mask); }
__device__ __forceinline__ int mynn_lane_bcast(int v, int src) const { return __shfl_sync(lane_mask, v, src, lane_width); }
__device__ __forceinline__ float mynn_lane_bcast(float v, int src) const { return __shfl_sync(lane_mask, v, src, lane_width); }
__device__ __forceinline__ double mynn_lane_bcast(double v, int src) const { return __shfl_sync(lane_mask, v, src, lane_width); }
__device__ __forceinline__ bool mynn_lane_bcast(bool v, int src) const { return __shfl_sync(lane_mask, (int)v, src, lane_width) != 0; }
// Trip count of the generated loop for (v=lo; st > 0 ? v <= hi : v >= hi; v += st), st a nonzero literal.
__device__ __forceinline__ static int mynn_trip(int lo, int hi, int st) {
    return st > 0 ? (hi >= lo ? (hi - lo)/st + 1 : 0) : (hi <= lo ? (lo - hi)/(-st) + 1 : 0);
}
static constexpr float r_d = 287.f;
static constexpr float r_v = 461.6f;
static constexpr float cp = 1004.5f;
static constexpr float cpv = ((4.f) * (r_v));
static constexpr float cliq = 4190.f;
static constexpr float cice = 2106.f;
static constexpr float grav = 9.80616f;
static constexpr float karman = 0.4f;
static constexpr float p1000mb = 100000.f;
static constexpr float rcp = ((r_d) / (cp));
static constexpr float xlv = 2.50e6f;
static constexpr float xlf = 3.50e5f;
static constexpr float xls = ((xlv) + (xlf));
static constexpr float svp1 = 0.6112f;
static constexpr float svp2 = 17.67f;
static constexpr float svp3 = 29.65f;
static constexpr float p608 = ((((r_v) / (r_d))) - (1.f));
static constexpr float ep_2 = ((r_d) / (r_v));
static constexpr float rvovrd = ((r_v) / (r_d));
static constexpr float t0c = 273.15f;
static constexpr float tref = 300.0f;
static constexpr float tkmin = 253.0f;
static constexpr float tice = 238.0f;
static constexpr float ep_3 = ((1.f) - (ep_2));
static constexpr float gtr = ((grav) / (tref));
static constexpr float rk = ((cp) / (r_d));
static constexpr float tv0 = ((p608) * (tref));
static constexpr float tv1 = ((((1.f) + (p608))) * (tref));
static constexpr float xlscp = ((((xlv) + (xlf))) / (cp));
static constexpr float xlvcp = ((xlv) / (cp));
static constexpr float g_inv = ((1.f) / (grav));
static constexpr float wfa_max = 800e12f;
static constexpr float wfa_min = 1e6f;
static constexpr float ifa_max = 500e6f;
static constexpr float ifa_min = 0.0f;
static constexpr float wfa_ht = 2000.f;
static constexpr float ifa_ht = 10000.f;
static constexpr float cphm_st = 5.0f;
static constexpr float cphm_unst = 16.0f;
static constexpr float cphh_st = 5.0f;
static constexpr float cphh_unst = 16.0f;
static constexpr float pr = 0.74f;
static constexpr float g1 = 0.235f;
static constexpr float b1 = 24.0f;
static constexpr float b2 = 15.0f;
static constexpr float c2 = 0.729f;
static constexpr float c3 = 0.340f;
static constexpr float c4 = 0.0f;
static constexpr float c5 = 0.2f;
static constexpr float a1 = ((((b1) * (((1.0f) - (((3.0f) * (g1))))))) / (6.0f));
static constexpr float c1 = ((g1) - (((1.0f) / (((((3.0f) * (a1))) * (2.88449914061481660f))))));
static constexpr float a2 = ((((a1) * (((g1) - (c1))))) / (((g1) * (pr))));
static constexpr float g2 = ((((((b2) / (b1))) * (((1.0f) - (c3))))) + (((((((2.0f) * (a1))) / (b1))) * (((3.0f) - (((2.0f) * (c2))))))));
static constexpr float cc2 = ((1.0f) - (c2));
static constexpr float cc3 = ((1.0f) - (c3));
static constexpr float e1c = ((((((3.0f) * (a2))) * (b2))) * (cc3));
static constexpr float e2c = ((((((9.0f) * (a1))) * (a2))) * (cc2));
static constexpr float e3c = ((((((((9.0f) * (a2))) * (a2))) * (cc2))) * (((1.0f) - (c5))));
static constexpr float e4c = ((((((12.0f) * (a1))) * (a2))) * (cc2));
static constexpr float e5c = ((((6.0f) * (a1))) * (a1));
static constexpr float zero = 0.0f;
static constexpr float one = 1.0f;
static constexpr float two = 2.0f;
static constexpr float three = 3.0f;
static constexpr float four = 4.0f;
static constexpr float five = 5.0f;
static constexpr float six = 6.0f;
static constexpr float seven = 7.0f;
static constexpr float eight = 8.0f;
static constexpr float nine = 9.0f;
static constexpr float ten = 10.0f;
static constexpr float twenty = 20.0f;
static constexpr float thirty = 30.0f;
static constexpr float forty = 40.0f;
static constexpr float fifty = 50.0f;
static constexpr float hundred = 100.0f;
static constexpr float p01 = 0.01f;
static constexpr float p1 = 0.1f;
static constexpr float p2 = 0.2f;
static constexpr float p25 = 0.25f;
static constexpr float p3 = 0.3f;
static constexpr float p333 = ((1.0f) / (3.0f));
static constexpr float p4 = 0.4f;
static constexpr float p5 = 0.5f;
static constexpr float p6 = 0.6f;
static constexpr float p666 = ((2.0f) / (3.0f));
static constexpr float p7 = 0.7f;
static constexpr float p75 = 0.75f;
static constexpr float p8 = 0.8f;
static constexpr float p9 = 0.9f;
static constexpr float qmin = 0.0f;
static constexpr float zmax = 1.0f;
static constexpr float sqfac = 3.0f;
static constexpr float qkemin = 1.e-5f;
static constexpr float tliq = 269.f;
static constexpr float rr2 = 0.7071068f;
static constexpr float rrp = 0.3989423f;
static constexpr float ckmod = 1.f;
static constexpr float scaleaware = 1.f;
static constexpr float taue = 80.f;
static constexpr float upwind = 1.0f;
static constexpr float fadv = 1.f;
static constexpr int dheat_opt = 1;
static constexpr bool env_subs = false;
static constexpr int bl_mynn_stfunc = 0;
static constexpr bool debug_code = false;
static constexpr int idbg = 452;
static constexpr int jdbg = 272;
int mynn_level;
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:422-1317
__device__ __forceinline__ void f_mynnedmf(int i, int j, int initflag, bool restart, bool cycling, float delt, MynnArray<float, 1> dz1, float dx, MynnArray<float, 1> u1, MynnArray<float, 1> v1, MynnArray<float, 1> w1, MynnArray<float, 1> th1, MynnArray<float, 1> sqv1, MynnArray<float, 1> sqc1, MynnArray<float, 1> sqi1, MynnArray<float, 1> sqs1, MynnArray<float, 1> qnc1, MynnArray<float, 1> qni1, MynnArray<float, 1> qnwfa1, MynnArray<float, 1> qnifa1, MynnArray<float, 1> qnbca1, MynnArray<float, 1> ozone1, MynnArray<float, 1> pres1, MynnArray<float, 1> ex1, MynnArray<float, 1> rho1, MynnArray<float, 1> tk1, float xland, float ts, float qsfc, float ps, float ust, float ch, float hfx, float qfx, float znt, float wspd, float uoce, float voce, MynnArray<float, 1> qke1, MynnArray<float, 1> qke_adv1, MynnArray<float, 1> el1, MynnArray<float, 1> sh1, MynnArray<float, 1> sm1, MynnArray<float, 1> kh1, MynnArray<float, 1> km1, int nchem, int ndvel, MynnArray<float, 2> chem1, MynnArray<float, 1> vdep, float frp, float emis_ant_no, bool mix_chem, bool enh_mix, MynnArray<float, 2> settle1, MynnArray<float, 2> scalars, int nscalars, MynnArray<float, 1> tsq1, MynnArray<float, 1> qsq1, MynnArray<float, 1> cov1, MynnArray<float, 1> du1, MynnArray<float, 1> dv1, MynnArray<float, 1> dth1, MynnArray<float, 1> dqv1, MynnArray<float, 1> dqc1, MynnArray<float, 1> dqi1, MynnArray<float, 1> dqnc1, MynnArray<float, 1> dqni1, MynnArray<float, 1> dqs1, MynnArray<float, 1> dqnwfa1, MynnArray<float, 1> dqnifa1, MynnArray<float, 1> dqnbca1, MynnArray<float, 1> dozone1, MynnArray<float, 1> rthraten1, float& pblh, int& kpbl, float& maxwidth, float& maxmf, float& ztop_plume, float& excess_h, float& excess_q, MynnArray<float, 1> dqke1, MynnArray<float, 1> qwt1, MynnArray<float, 1> qshear1, MynnArray<float, 1> qbuoy1, MynnArray<float, 1> qdiss1, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> qi_bl1, MynnArray<float, 1> cldfra_bl1, bool bl_mynn_tkeadvect, int tke_budget, int bl_mynn_cloudpdf, int bl_mynn_mixlength, int icloud_bl, float closure, int bl_mynn_edmf, int bl_mynn_edmf_mom, int bl_mynn_edmf_tke, int bl_mynn_mixscalars, int bl_mynn_mixaerosols, int bl_mynn_mixnumcon, int bl_mynn_output, int bl_mynn_cloudmix, int bl_mynn_mixqt, int bl_mynn_edmf_dd, int bl_mynn_ess, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_qt1, MynnArray<float, 1> edmf_thl1, MynnArray<float, 1> edmf_ent1, MynnArray<float, 1> edmf_qc1, MynnArray<float, 1> sub_thl1, MynnArray<float, 1> sub_sqv1, MynnArray<float, 1> det_thl1, MynnArray<float, 1> det_sqv1, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1, bool flag_qc, bool flag_qi, bool flag_qnc, bool flag_qni, bool flag_qs, bool flag_qnwfa, bool flag_qnifa, bool flag_qnbca, bool flag_ozone, int kts, int kte, MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
dz1.rebase({kts});
u1.rebase({kts});
v1.rebase({kts});
w1.rebase({kts});
th1.rebase({kts});
sqv1.rebase({kts});
sqc1.rebase({kts});
sqi1.rebase({kts});
sqs1.rebase({kts});
qnc1.rebase({kts});
qni1.rebase({kts});
qnwfa1.rebase({kts});
qnifa1.rebase({kts});
qnbca1.rebase({kts});
ozone1.rebase({kts});
pres1.rebase({kts});
ex1.rebase({kts});
rho1.rebase({kts});
tk1.rebase({kts});
qke1.rebase({kts});
qke_adv1.rebase({kts});
el1.rebase({kts});
sh1.rebase({kts});
sm1.rebase({kts});
kh1.rebase({kts});
km1.rebase({kts});
chem1.rebase({kts, 1});
vdep.rebase({1});
settle1.rebase({kts, 1});
scalars.rebase({kts, 1});
tsq1.rebase({kts});
qsq1.rebase({kts});
cov1.rebase({kts});
du1.rebase({kts});
dv1.rebase({kts});
dth1.rebase({kts});
dqv1.rebase({kts});
dqc1.rebase({kts});
dqi1.rebase({kts});
dqnc1.rebase({kts});
dqni1.rebase({kts});
dqs1.rebase({kts});
dqnwfa1.rebase({kts});
dqnifa1.rebase({kts});
dqnbca1.rebase({kts});
dozone1.rebase({kts});
rthraten1.rebase({kts});
dqke1.rebase({kts});
qwt1.rebase({kts});
qshear1.rebase({kts});
qbuoy1.rebase({kts});
qdiss1.rebase({kts});
qc_bl1.rebase({kts});
qi_bl1.rebase({kts});
cldfra_bl1.rebase({kts});
edmf_a1.rebase({kts});
edmf_w1.rebase({kts});
edmf_qt1.rebase({kts});
edmf_thl1.rebase({kts});
edmf_ent1.rebase({kts});
edmf_qc1.rebase({kts});
sub_thl1.rebase({kts});
sub_sqv1.rebase({kts});
det_thl1.rebase({kts});
det_sqv1.rebase({kts});
pattern_spp_pbl1.rebase({kts});
float psig_bl;
float psig_shcu;
float rmol;
float rmolh;
int imd;
int jmd;
int k;
int kproblem;
MynnArray<float, 1> qc_bl1_old = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qi_bl1_old = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> cldfra_bl1_old = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dummy1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dummy2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> diss_heat1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> thl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> thv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> thlv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qi1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qs1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqw1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> thl_tot1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qc_tot1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qi_tot1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dfm1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dfh1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dfq1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> tcd1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qcd1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> pdk1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> pdt1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> pdq1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> pdc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> vt1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> vq1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sgm1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> kzero1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_u1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_v1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_qv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 2> s_awchem1 = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nchem});
int ic;
MynnArray<float, 2> s_awscalars1 = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nscalars});
MynnArray<float, 1> dth1mf = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dqv1mf = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dqc1mf = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> du1mf = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dv1mf = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_a_dd1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_w_dd1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_qt_dd1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_thl_dd1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_ent_dd1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_qc_dd1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sub_u1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sub_v1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> det_sqc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> det_u1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> det_v1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> s_aw1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awthl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqt1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awu1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqke1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqnc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqni1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqnwfa1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqnifa1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqnbca1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_aw1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awthl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqt1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awu1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqke1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqi1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqnc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqni1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqnwfa1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqnifa1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> zw1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
float cpm;
float sqcg;
float flt;
float fltv;
float flq;
float flqv;
float flqc;
float pmz;
float phh;
float exnerg;
float zet;
float phi_m;
float afk;
float abk;
float ts_decay;
float qc_bl2;
float qi_bl2;
float th_sfc;
float wsp;
int ktop_plume;
float maxkhtopdown;
MynnArray<float, 1> khtopdown = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> tkeprod_dn = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> tkeprod_up = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
bool initialize_qke;
bool problem;
int nsub;
float delt2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:621
errmsg = " ";
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:622
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:624
if (debug_code) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:625
problem = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:626
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:627
wsp = mynn_sqrt(r6_add(mynn_ipow<2>(u1(k)), mynn_ipow<2>(v1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:628
if (((((((((((((((((mynn_abs(hfx)) > (1200.f))) || (((mynn_abs(qfx)) > (0.001f))))) || (((wsp) > (270.f))))) || (((tk1(k)) > (380.f))))) || (((tk1(k)) < (160.f))))) || (((sqv1(k)) < (zero))))) || (((sqc1(k)) < (zero))))) || (((r6_sub(ps, pres1(kts))) < (zero))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:632
kproblem = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:633
problem = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:634
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:635
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:636
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:637
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:638
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:639
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:640
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:641
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:642
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:643
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:644
if (problem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:645
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:646
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:647
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:648
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:649
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:650
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:651
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:652
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:653
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:656
imd = 10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:657
jmd = 10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:661
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1).size(); mynn_element += lane_width) { edmf_a1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:662
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1).size(); mynn_element += lane_width) { edmf_w1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:663
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt1).size(); mynn_element += lane_width) { edmf_qt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:664
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1).size(); mynn_element += lane_width) { edmf_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:665
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent1).size(); mynn_element += lane_width) { edmf_ent1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:666
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1).size(); mynn_element += lane_width) { edmf_qc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:667
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_u1).size(); mynn_element += lane_width) { edmf_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:668
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_v1).size(); mynn_element += lane_width) { edmf_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:669
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qv1).size(); mynn_element += lane_width) { edmf_qv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:670
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_thl1).size(); mynn_element += lane_width) { sub_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:671
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_sqv1).size(); mynn_element += lane_width) { sub_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:672
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_thl1).size(); mynn_element += lane_width) { det_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:673
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqv1).size(); mynn_element += lane_width) { det_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:674
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a_dd1).size(); mynn_element += lane_width) { edmf_a_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:675
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w_dd1).size(); mynn_element += lane_width) { edmf_w_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:676
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc_dd1).size(); mynn_element += lane_width) { edmf_qc_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:678
ktop_plume = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:679
ztop_plume = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:680
maxwidth = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:681
maxmf = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:682
excess_h = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:683
excess_q = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:684
maxkhtopdown = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:685
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (kzero1).size(); mynn_element += lane_width) { kzero1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:692
if (((((initflag) > (0))) && ((!(restart))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:695
if (((restart) || (cycling))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:696
if (((mynn_maxval(qke1.slice(MynnRange{qke1.lo[0], qke1.hi[0]}))) < (0.0002f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:697
initialize_qke = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:699
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:700
initialize_qke = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:702
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:703
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:704
initialize_qke = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:706
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:708
if ((((!(restart))) || ((!(cycling))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:709
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sh1).size(); mynn_element += lane_width) { sh1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:710
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sm1).size(); mynn_element += lane_width) { sm1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:711
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (el1).size(); mynn_element += lane_width) { el1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:712
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tsq1).size(); mynn_element += lane_width) { tsq1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:713
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qsq1).size(); mynn_element += lane_width) { qsq1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:714
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cov1).size(); mynn_element += lane_width) { cov1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:715
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1).size(); mynn_element += lane_width) { cldfra_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:716
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1).size(); mynn_element += lane_width) { qc_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:717
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi_bl1).size(); mynn_element += lane_width) { qi_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:718
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke1).size(); mynn_element += lane_width) { qke1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:719
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke_adv1).size(); mynn_element += lane_width) { qke_adv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:720
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:721
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqc1).size(); mynn_element += lane_width) { dqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:722
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqi1).size(); mynn_element += lane_width) { dqi1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:723
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqni1).size(); mynn_element += lane_width) { dqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:724
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnc1).size(); mynn_element += lane_width) { dqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:725
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnwfa1).size(); mynn_element += lane_width) { dqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:726
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnifa1).size(); mynn_element += lane_width) { dqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:727
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnbca1).size(); mynn_element += lane_width) { dqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:728
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dozone1).size(); mynn_element += lane_width) { dozone1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:729
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1_old).size(); mynn_element += lane_width) { qc_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:730
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1_old).size(); mynn_element += lane_width) { cldfra_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:731
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sgm1).size(); mynn_element += lane_width) { sgm1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:732
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (vt1).size(); mynn_element += lane_width) { vt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:733
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (vq1).size(); mynn_element += lane_width) { vq1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:734
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (km1).size(); mynn_element += lane_width) { km1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:735
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (kh1).size(); mynn_element += lane_width) { kh1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:737
if (((tke_budget) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:738
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qwt1).size(); mynn_element += lane_width) { qwt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:739
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qshear1).size(); mynn_element += lane_width) { qshear1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:740
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qbuoy1).size(); mynn_element += lane_width) { qbuoy1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:741
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qdiss1).size(); mynn_element += lane_width) { qdiss1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:742
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqke1).size(); mynn_element += lane_width) { dqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:743
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:745
zw1(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:746
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:748
sqw1(k) = r6_add(r6_add(sqv1(k), sqc1(k)), sqi1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:749
thl1(k) = r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), sqc1(k))), r6_mul(r6_div(xlscp, ex1(k)), sqi1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:755
thlv1(k) = r6_mul(thl1(k), r6_add(one, r6_mul(p608, sqv1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:756
thv1(k) = r6_mul(th1(k), r6_sub(r6_add(one, r6_mul(p608, sqv1(k))), r6_add(sqc1(k), sqi1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:757
zw1(((k) + (1))) = r6_add(zw1(k), dz1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:758
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:760
if (initialize_qke) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:764
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:765
qke1(k) = r6_mul(r6_mul(five, ust), mynn_max((float)(r6_div(r6_sub(r6_mul(ust, 700.f), zw1(k)), r6_mul(mynn_max((float)(ust), (float)(p01)), 700.f))), (float)(p01)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:766
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:767
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:771
f_get_pblh(kts, kte, pblh, thv1, qke1, ust, zw1, dz1, xland, kpbl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:775
if (((scaleaware) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:776
f_scale_aware(dx, pblh, psig_bl, psig_shcu);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:777
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:778
psig_bl = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:779
psig_shcu = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:780
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:787
rmol = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:788
rmolh = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:789
f_mym_initialize(kts, kte, xland, pblh, dz1, dx, zw1, pres1, ex1, u1, v1, thl1, sqv1, pblh, th1, thv1, thlv1, sh1, sm1, ust, rmol, rmolh, el1, qke1, tsq1, qsq1, cov1, psig_bl, cldfra_bl1, bl_mynn_mixlength, bl_mynn_ess, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1, initialize_qke, spp_pbl, pattern_spp_pbl1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:806
if ((!(restart))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:808
if (bl_mynn_tkeadvect) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:809
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:810
qke_adv1(k) = qke1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:811
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:812
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:813
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:825
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:829
if (bl_mynn_tkeadvect) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:830
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qke1.slice(MynnRange{kts, kte}).linear(mynn_element) = qke_adv1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:831
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:834
if (((tke_budget) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:835
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqke1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { dqke1.slice(MynnRange{kts, kte}).linear(mynn_element) = qke1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:836
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:837
if (((icloud_bl) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:838
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1_old.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { cldfra_bl1_old.slice(MynnRange{kts, kte}).linear(mynn_element) = cldfra_bl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:839
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1_old.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qc_bl1_old.slice(MynnRange{kts, kte}).linear(mynn_element) = qc_bl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:840
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi_bl1_old.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qi_bl1_old.slice(MynnRange{kts, kte}).linear(mynn_element) = qi_bl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:841
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:842
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1).size(); mynn_element += lane_width) { cldfra_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:843
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1).size(); mynn_element += lane_width) { qc_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:844
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi_bl1).size(); mynn_element += lane_width) { qi_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:845
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1_old).size(); mynn_element += lane_width) { cldfra_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:846
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1_old).size(); mynn_element += lane_width) { qc_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:847
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi_bl1_old).size(); mynn_element += lane_width) { qi_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:848
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:849
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqc1).size(); mynn_element += lane_width) { dqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:850
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqi1).size(); mynn_element += lane_width) { dqi1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:851
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqs1).size(); mynn_element += lane_width) { dqs1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:852
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqni1).size(); mynn_element += lane_width) { dqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:853
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnc1).size(); mynn_element += lane_width) { dqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:854
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnwfa1).size(); mynn_element += lane_width) { dqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:855
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnifa1).size(); mynn_element += lane_width) { dqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:856
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnbca1).size(); mynn_element += lane_width) { dqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:857
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dozone1).size(); mynn_element += lane_width) { dozone1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:859
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1).size(); mynn_element += lane_width) { edmf_a1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:860
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1).size(); mynn_element += lane_width) { edmf_w1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:861
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1).size(); mynn_element += lane_width) { edmf_qc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:862
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qv1).size(); mynn_element += lane_width) { edmf_qv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:863
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1).size(); mynn_element += lane_width) { edmf_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:864
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_u1).size(); mynn_element += lane_width) { edmf_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:865
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_v1).size(); mynn_element += lane_width) { edmf_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:866
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_aw1).size(); mynn_element += lane_width) { s_aw1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:867
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awthl1).size(); mynn_element += lane_width) { s_awthl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:868
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqt1).size(); mynn_element += lane_width) { s_awqt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:869
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqv1).size(); mynn_element += lane_width) { s_awqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:870
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqc1).size(); mynn_element += lane_width) { s_awqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:871
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awu1).size(); mynn_element += lane_width) { s_awu1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:872
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awv1).size(); mynn_element += lane_width) { s_awv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:873
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqke1).size(); mynn_element += lane_width) { s_awqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:874
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnc1).size(); mynn_element += lane_width) { s_awqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:875
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqni1).size(); mynn_element += lane_width) { s_awqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:876
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnwfa1).size(); mynn_element += lane_width) { s_awqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:877
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnifa1).size(); mynn_element += lane_width) { s_awqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:878
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnbca1).size(); mynn_element += lane_width) { s_awqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:879
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:880
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars})).size(); mynn_element += lane_width) { s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:882
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a_dd1).size(); mynn_element += lane_width) { edmf_a_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:883
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w_dd1).size(); mynn_element += lane_width) { edmf_w_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:884
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc_dd1).size(); mynn_element += lane_width) { edmf_qc_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:885
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_aw1).size(); mynn_element += lane_width) { sd_aw1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:886
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awthl1).size(); mynn_element += lane_width) { sd_awthl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:887
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqt1).size(); mynn_element += lane_width) { sd_awqt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:888
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqv1).size(); mynn_element += lane_width) { sd_awqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:889
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqc1).size(); mynn_element += lane_width) { sd_awqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:890
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqi1).size(); mynn_element += lane_width) { sd_awqi1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:891
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnc1).size(); mynn_element += lane_width) { sd_awqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:892
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqni1).size(); mynn_element += lane_width) { sd_awqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:893
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnwfa1).size(); mynn_element += lane_width) { sd_awqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:894
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnifa1).size(); mynn_element += lane_width) { sd_awqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:895
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awu1).size(); mynn_element += lane_width) { sd_awu1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:896
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awv1).size(); mynn_element += lane_width) { sd_awv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:897
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqke1).size(); mynn_element += lane_width) { sd_awqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:898
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_thl1).size(); mynn_element += lane_width) { sub_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:899
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_sqv1).size(); mynn_element += lane_width) { sub_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:900
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_u1).size(); mynn_element += lane_width) { sub_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:901
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_v1).size(); mynn_element += lane_width) { sub_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:902
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_thl1).size(); mynn_element += lane_width) { det_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:903
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqv1).size(); mynn_element += lane_width) { det_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:904
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqc1).size(); mynn_element += lane_width) { det_sqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:905
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_u1).size(); mynn_element += lane_width) { det_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:906
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_v1).size(); mynn_element += lane_width) { det_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:908
zw1(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:909
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:910
zw1(((k) + (1))) = r6_add(zw1(k), dz1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:911
qv1(k) = r6_div(sqv1(k), r6_sub(one, sqv1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:912
qc1(k) = r6_div(sqc1(k), r6_sub(one, sqv1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:913
qi1(k) = r6_div(sqi1(k), r6_sub(one, sqv1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:914
qs1(k) = r6_div(sqs1(k), r6_sub(one, sqv1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:916
sqw1(k) = r6_add(r6_add(sqv1(k), sqc1(k)), sqi1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:917
thl1(k) = r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), sqc1(k))), r6_mul(r6_div(xlscp, ex1(k)), sqi1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:923
thv1(k) = r6_mul(th1(k), r6_sub(r6_add(one, r6_mul(p608, sqv1(k))), r6_add(sqc1(k), sqi1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:924
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:928
f_get_pblh(kts, kte, pblh, thv1, qke1, ust, zw1, dz1, xland, kpbl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:934
if (((scaleaware) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:935
f_scale_aware(dx, pblh, psig_bl, psig_shcu);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:936
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:937
psig_bl = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:938
psig_shcu = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:939
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:941
sqcg = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:942
cpm = r6_mul(cp, r6_add(one, r6_mul(0.84f, mynn_max((float)(sqv1(kts)), (float)(1e-8f)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:943
exnerg = mynn_pow(r6_div(ps, p1000mb), rcp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:952
flqv = r6_div(qfx, rho1(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:953
flqc = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:954
th_sfc = r6_div(ts, ex1(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:957
flq = r6_add(flqv, flqc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:958
flt = r6_sub(r6_div(hfx, r6_mul(rho1(kts), cpm)), r6_div(r6_mul(xlvcp, flqc), ex1(kts)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:959
fltv = r6_add(flt, r6_mul(r6_mul(flqv, p608), th_sfc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:962
rmol = r6_div(r6_mul(r6_neg(r6_mul(karman, gtr)), fltv), mynn_max((float)(mynn_ipow<3>(ust)), (float)(1.0e-6f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:963
rmolh = r6_div(r6_mul(r6_neg(r6_mul(karman, gtr)), flt), mynn_max((float)(mynn_ipow<3>(ust)), (float)(1.0e-6f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:964
zet = r6_mul(r6_mul(p5, dz1(kts)), rmol);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:965
zet = mynn_max((float)(zet), (float)(r6_neg(10.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:966
zet = mynn_min((float)(zet), (float)(10.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:968
if (((bl_mynn_stfunc) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:970
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:971
if (((zet) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:973
pmz = r6_add(one, r6_mul(3.0f, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:974
phh = r6_add(one, r6_mul(cphh_st, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:975
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:976
pmz = r6_sub(r6_div(one, mynn_pow(r6_sub(one, r6_mul(cphm_unst, zet)), 0.25f)), zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:977
phh = r6_div(one, mynn_sqrt(r6_sub(one, r6_mul(cphh_unst, zet))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:978
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:979
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:980
if (((zet) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:982
pmz = r6_add(0.95f, r6_mul(3.5f, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:983
phh = r6_add(one, r6_mul(cphh_st, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:984
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:986
pmz = r6_sub(r6_div(0.95f, mynn_pow(r6_sub(one, r6_mul(cphm_unst, zet)), 0.25f)), zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:987
phh = r6_div(one, mynn_sqrt(r6_sub(one, r6_mul(cphh_unst, zet))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:988
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:989
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:990
phi_m = f_phim(zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:991
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:993
phi_m = f_phim(zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:994
pmz = r6_sub(phi_m, zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:995
phh = f_phih(zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:996
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1003
f_mym_condensation(kts, kte, dx, dz1, zw1, xland, thl1, sqw1, sqv1, sqc1, sqi1, sqs1, pres1, ex1, tsq1, qsq1, cov1, sh1, el1, bl_mynn_cloudpdf, qc_bl1, qi_bl1, cldfra_bl1, pblh, hfx, vt1, vq1, th1, sgm1, closure, spp_pbl, pattern_spp_pbl1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1015
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprod_up).size(); mynn_element += lane_width) { tkeprod_up.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1016
if (((bl_mynn_edmf) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1018
f_dmp_mf(i, j, kts, kte, delt, zw1, dz1, pres1, rho1, bl_mynn_edmf_mom, bl_mynn_edmf_tke, bl_mynn_mixscalars, bl_mynn_mixaerosols, bl_mynn_mixnumcon, u1, v1, w1, th1, thl1, thv1, tk1, sqw1, sqv1, sqc1, qke1, qnc1, qni1, qnwfa1, qnifa1, qnbca1, ex1, vt1, vq1, sgm1, ust, flt, fltv, flq, flqv, pblh, kpbl, dx, xland, th_sfc, edmf_a1, edmf_w1, edmf_qt1, edmf_thl1, edmf_ent1, edmf_qc1, edmf_qv1, edmf_u1, edmf_v1, s_aw1, s_awthl1, s_awqt1, s_awqv1, s_awqc1, s_awu1, s_awv1, s_awqke1, s_awqnc1, s_awqni1, s_awqnwfa1, s_awqnifa1, s_awqnbca1, sub_thl1, sub_sqv1, sub_u1, sub_v1, det_thl1, det_sqv1, det_sqc1, det_u1, det_v1, nchem, chem1, s_awchem1, mix_chem, nscalars, scalars, s_awscalars1, qc_bl1, cldfra_bl1, qc_bl1_old, cldfra_bl1_old, flag_qc, flag_qi, flag_qnc, flag_qni, flag_qnwfa, flag_qnifa, flag_qnbca, psig_shcu, maxwidth, ktop_plume, maxmf, ztop_plume, excess_h, excess_q, spp_pbl, pattern_spp_pbl1, tkeprod_up, el1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1062
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1068
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprod_dn).size(); mynn_element += lane_width) { tkeprod_dn.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1069
maxkhtopdown = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1070
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (khtopdown).size(); mynn_element += lane_width) { khtopdown.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1071
if (((bl_mynn_edmf_dd) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1072
f_ddmp_mf(kts, kte, delt, dx, zw1, dz1, pres1, u1, v1, th1, thl1, thv1, tk1, sqw1, sqv1, sqc1, sqi1, qnc1, qni1, qnwfa1, qnifa1, qke1, rho1, ex1, qc_bl1, qi_bl1, cldfra_bl1, ust, flt, flq, fltv, pblh, kpbl, edmf_a_dd1, edmf_w_dd1, edmf_qt_dd1, edmf_thl_dd1, edmf_ent_dd1, edmf_qc_dd1, sd_aw1, sd_awthl1, sd_awqt1, sd_awqv1, sd_awqc1, sd_awqi1, sd_awqnc1, sd_awqni1, sd_awqnwfa1, sd_awqnifa1, sd_awu1, sd_awv1, sd_awqke1, tkeprod_dn, el1, rthraten1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1091
} else if (((bl_mynn_edmf_dd) == (2))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1092
f_topdown_cloudrad(kts, kte, dz1, zw1, fltv, xland, kpbl, pblh, sqc1, sqi1, sqw1, thl1, th1, ex1, pres1, rho1, thv1, cldfra_bl1, rthraten1, maxkhtopdown, khtopdown, tkeprod_dn);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1097
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1101
delt2 = delt;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1104
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1105
qc_tot1(k) = mynn_max((float)(qc_bl1(k)), (float)(sqc1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1106
qi_tot1(k) = mynn_max((float)(qi_bl1(k)), (float)(r6_add(sqi1(k), sqs1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1107
thlv1(k) = r6_mul(r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), qc_tot1(k))), r6_mul(r6_div(xlscp, ex1(k)), qi_tot1(k))), r6_add(one, r6_mul(p608, sqv1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1110
thl_tot1(k) = r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), qc_tot1(k))), r6_mul(r6_div(xlscp, ex1(k)), qi_tot1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1112
thv1(k) = r6_mul(th1(k), r6_sub(r6_add(one, r6_mul(p608, sqv1(k))), r6_add(qc_tot1(k), qi_tot1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1113
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1115
f_mym_turbulence(kts, kte, xland, closure, dz1, dx, zw1, pres1, ex1, u1, v1, thl1, thv1, thlv1, sqc1, sqw1, qke1, tsq1, qsq1, cov1, vt1, vq1, rmol, rmolh, flt, fltv, flq, pblh, th1, sh1, sm1, el1, dfm1, dfh1, dfq1, tcd1, qcd1, pdk1, pdt1, pdq1, pdc1, qwt1, qshear1, qbuoy1, qdiss1, tke_budget, psig_bl, psig_shcu, cldfra_bl1, bl_mynn_mixlength, bl_mynn_ess, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1, tkeprod_dn, tkeprod_up, spp_pbl, pattern_spp_pbl1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1141
f_mym_predict(kts, kte, closure, delt2, dz1, ust, flt, flq, pmz, phh, el1, dfq1, rho1, pdk1, pdt1, pdq1, pdc1, qke1, tsq1, qsq1, cov1, s_aw1, s_awqke1, bl_mynn_edmf, bl_mynn_edmf_tke, qwt1, qdiss1, tke_budget);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1150
if (((dheat_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1151
{ const int mynn_lo3 = (kts), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1153
diss_heat1(k) = mynn_min((float)(mynn_max((float)(zero), (float)(r6_div(r6_div(mynn_pow(qke1(k), 1.5f), r6_mul(b1, mynn_max((float)(r6_mul(p5, r6_add(el1(k), el1(((k) + (1)))))), (float)(one)))), cp)))), (float)(0.002f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1155
diss_heat1(k) = r6_mul(diss_heat1(k), mynn_exp(r6_neg(r6_div(10000.f, mynn_max((float)(pres1(k)), (float)(one))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1156
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1157
diss_heat1(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1158
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1159
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (diss_heat1).size(); mynn_element += lane_width) { diss_heat1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1160
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1164
f_mynn_tendencies(kts, kte, i, delt, dz1, zw1, xland, rho1, u1, v1, th1, tk1, qv1, qc1, qi1, kzero1, qnc1, qni1, ps, pres1, ex1, thl1, sqv1, sqc1, sqi1, kzero1, sqw1, thl_tot1, qc_tot1, qi_tot1, qnwfa1, qnifa1, qnbca1, ozone1, ust, flt, flq, flqv, flqc, wspd, uoce, voce, tsq1, qsq1, cov1, tcd1, qcd1, dfm1, dfh1, dfq1, du1, dv1, dth1, dqv1, dqc1, dqi1, dqs1, dqnc1, dqni1, dqnwfa1, dqnifa1, dqnbca1, dozone1, diss_heat1, s_aw1, s_awthl1, s_awqt1, s_awqv1, s_awqc1, s_awu1, s_awv1, s_awqnc1, s_awqni1, s_awqnwfa1, s_awqnifa1, s_awqnbca1, sd_aw1, sd_awthl1, sd_awqt1, sd_awqv1, sd_awqc1, sd_awqi1, sd_awqnc1, sd_awqni1, sd_awqnwfa1, sd_awqnifa1, sd_awu1, sd_awv1, sub_thl1, sub_sqv1, sub_u1, sub_v1, det_thl1, det_sqv1, det_sqc1, det_u1, det_v1, flag_qc, flag_qi, flag_qnc, flag_qni, flag_qs, flag_qnwfa, flag_qnifa, flag_qnbca, flag_ozone, cldfra_bl1, bl_mynn_cloudmix, bl_mynn_mixqt, bl_mynn_edmf, bl_mynn_edmf_mom, bl_mynn_mixscalars, bl_mynn_mixaerosols, bl_mynn_mixnumcon);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1209
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1210
f_mynn_mix_chem(kts, kte, i, delt, dz1, pblh, nchem, ndvel, chem1, vdep, settle1, rho1, flt, tcd1, qcd1, dfh1, s_aw1, s_awchem1, zero, zero, enh_mix, bl_mynn_edmf);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1222
{ const int mynn_lo4 = (1), mynn_hi4 = (nchem), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { ic = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1223
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1224
chem1(k, ic) = mynn_max((float)(chem1(k, ic)), (float)(1.e-12f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1225
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1226
} mynn_lane_sync(); ic = mynn_lo4 + mynn_n4*mynn_st4; if (mynn_n4 > 0) { const int mynn_src4 = (mynn_n4 - 1) % lane_width; k = mynn_lane_bcast(k, mynn_src4); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1227
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1229
if (((bl_mynn_mixscalars) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1230
f_mynn_mix_scalars(kts, kte, i, delt, dz1, nscalars, scalars, rho1, flt, tcd1, qcd1, dfh1, s_aw1, s_awscalars1, bl_mynn_edmf);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1238
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1241
f_retrieve_exchange_coeffs(kts, kte, dfm1, dfh1, dz1, km1, kh1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1245
if (((tke_budget) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1248
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1249
qshear1(k) = r6_sub(r6_mul(4.f, r6_div(r6_mul(mynn_ipow<3>(ust), phi_m), r6_mul(karman, dz1(k)))), qshear1(((k) + (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1250
qbuoy1(k) = r6_sub(r6_mul(4.f, r6_neg(r6_div(r6_mul(mynn_ipow<3>(ust), zet), r6_mul(karman, dz1(k))))), qbuoy1(((k) + (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1252
{ const int mynn_lo5 = (kts), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1253
dummy1(k) = r6_mul(p5, r6_add(qshear1(k), qshear1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1254
dummy2(k) = r6_mul(p5, r6_add(qbuoy1(k), qbuoy1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1255
dqke1(k) = r6_div(r6_mul(p5, r6_sub(qke1(k), dqke1(k))), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1256
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1257
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qshear1).size(); mynn_element += lane_width) { qshear1.linear(mynn_element) = dummy1.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1258
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qbuoy1).size(); mynn_element += lane_width) { qbuoy1.linear(mynn_element) = dummy2.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1260
k = kte;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1261
qshear1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1262
qbuoy1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1263
qwt1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1264
qdiss1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1265
dqke1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1266
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1269
if (((bl_mynn_output) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1271
if (((((bl_mynn_output) == (2))) && (((bl_mynn_edmf_dd) == (1))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1272
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_a1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_a_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1273
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_w1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_w_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1274
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_qt1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_qt_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1275
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_thl1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_thl_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1276
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_ent1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_ent_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1277
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_qc1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_qc_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1278
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1279
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1282
if (((debug_code) && (((i) == (idbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1283
if (((mynn_abs(qfx)) > (.001f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1285
if (((mynn_abs(hfx)) > (1100.f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1287
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1288
if (((((sh1(k)) < (0.f))) || (((sh1(k)) > (200.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1290
if (((mynn_abs(vt1(k))) > (2.0f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1292
if (((mynn_abs(vq1(k))) > (7000.f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1294
if (((((qke1(k)) < (r6_neg(1.f)))) || (((qke1(k)) > (200.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1296
if (((((el1(k)) < (0.f))) || (((el1(k)) > (1500.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1298
if (((((km1(k)) < (0.f))) || (((km1(k)) > (2000.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1300
if (((icloud_bl) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1301
if (((((cldfra_bl1(k)) < (zero))) || (((cldfra_bl1(k)) > (one))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1302
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1304
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1305
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1306
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1307
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1310
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke_adv1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qke_adv1.slice(MynnRange{kts, kte}).linear(mynn_element) = qke1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1378-1548
__device__ __noinline__ void f_mym_initialize(int kts, int kte, float xland, float pblh, MynnArray<float, 1> dz, float dx, MynnArray<float, 1> zw, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> thl, MynnArray<float, 1> qw, float zi, MynnArray<float, 1> theta, MynnArray<float, 1> thv, MynnArray<float, 1> thlv, MynnArray<float, 1> sh, MynnArray<float, 1> sm, float ust, float rmol, float rmolh, MynnArray<float, 1> el, MynnArray<float, 1> qke, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, float psig_bl, MynnArray<float, 1> cldfra_bl1, int bl_mynn_mixlength, int bl_mynn_ess, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> edmf_w_dd1, MynnArray<float, 1> edmf_a_dd1, bool initialize_qke, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
zw.rebase({kts});
p.rebase({kts});
exner.rebase({kts});
u.rebase({kts});
v.rebase({kts});
thl.rebase({kts});
qw.rebase({kts});
theta.rebase({kts});
thv.rebase({kts});
thlv.rebase({kts});
sh.rebase({kts});
sm.rebase({kts});
el.rebase({kts});
qke.rebase({kts});
tsq.rebase({kts});
qsq.rebase({kts});
cov.rebase({kts});
cldfra_bl1.rebase({kts});
edmf_w1.rebase({kts});
edmf_a1.rebase({kts});
edmf_w_dd1.rebase({kts});
edmf_a_dd1.rebase({kts});
pattern_spp_pbl1.rebase({kts});
MynnArray<float, 1> ql = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> pdk = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> pdt = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> pdq = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> pdc = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dtl = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dqw = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dtv = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> gm = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> gh = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qkw = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qtw = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> vt = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> vq = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qpe = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
int k;
int l;
int lmax;
float phm;
float vkz;
float elq;
float elv;
float b1l;
float b2l;
float pmz = 1.f;
float phh = 1.f;
float flt = 0.f;
float fltv = 0.f;
float flq = 0.f;
float tmpq;
float ustm;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1422
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1423
ql(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1424
vt(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1425
vq(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1426
qkw(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1427
qtw(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1428
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1429
ustm = mynn_max((float)(ust), (float)(p01));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1432
f_mym_level2(kts, kte, bl_mynn_ess, zw, dz, xland, pblh, u, v, thl, thv, thlv, theta, p, exner, qke, cldfra_bl1, edmf_a1, edmf_w1, qw, ql, vt, vq, dtl, dqw, dtv, gm, gh, sm, sh);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1444
el(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1445
if (initialize_qke) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1447
qke(kts) = r6_mul(r6_mul(1.5f, mynn_ipow<2>(ustm)), mynn_pow(r6_mul(b1, pmz), p666));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1448
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1451
qke(k) = r6_mul(qke(kts), mynn_max((float)(r6_div(r6_sub(r6_mul(ust, 700.f), zw(k)), r6_mul(ustm, 700.f))), (float)(p01)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1452
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1453
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1455
phm = r6_div(r6_mul(phh, b2), mynn_pow(r6_mul(b1, pmz), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1456
tsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flt)), ustm)))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1457
qsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flq)), ustm)))), (float)(5e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1458
cov(kts) = r6_mul(r6_mul(phm, r6_div(flt, ustm)), r6_div(flq, ustm));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1459
qpe(kts) = mynn_min((float)(mynn_max((float)(r6_mul(r6_mul(r6_mul(two, tsq(kts)), mynn_ipow<2>(gtr)), mynn_ipow<2>(taue))), (float)(zero))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1461
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1462
vkz = r6_mul(karman, zw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1463
el(k) = r6_div(vkz, r6_add(one, r6_div(vkz, hundred)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1466
tsq(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1467
qsq(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1468
cov(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1469
qpe(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1470
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; vkz = mynn_lane_bcast(vkz, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1474
lmax = 5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1476
for (l=1; (1) > 0 ? l <= (lmax) : l >= (lmax); l += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1479
f_mym_length(kts, kte, xland, dz, dx, zw, rmol, rmolh, flt, fltv, flq, vt, vq, u, v, qke, qpe, dtv, el, zi, theta, qkw, qtw, psig_bl, cldfra_bl1, bl_mynn_mixlength, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1495
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1496
elq = r6_mul(el(k), qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1497
pdk(k) = r6_mul(elq, r6_add(r6_mul(sm(k), gm(k)), r6_mul(sh(k), gh(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1499
pdt(k) = r6_mul(r6_mul(elq, sh(k)), mynn_ipow<2>(dtl(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1500
pdq(k) = r6_mul(r6_mul(elq, sh(k)), mynn_ipow<2>(dqw(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1501
pdc(k) = r6_mul(r6_mul(r6_mul(elq, sh(k)), dtl(k)), dqw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1502
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; elq = mynn_lane_bcast(elq, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1505
vkz = r6_mul(r6_mul(karman, p5), dz(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1506
elv = r6_div(r6_mul(p5, r6_add(el(((kts) + (1))), el(kts))), vkz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1507
if (initialize_qke) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1509
qke(kts) = r6_mul(r6_mul(one, mynn_ipow<2>(ustm)), mynn_pow(r6_mul(r6_mul(b1, pmz), elv), p666));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1510
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1512
phm = r6_div(r6_mul(phh, b2), mynn_pow(r6_div(r6_mul(b1, pmz), mynn_ipow<2>(elv)), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1513
tsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flt)), ustm)))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1514
qsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flq)), ustm)))), (float)(5e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1515
cov(kts) = r6_mul(r6_mul(phm, r6_div(flt, ustm)), r6_div(flq, ustm));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1517
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1518
b1l = r6_mul(r6_mul(b1, 0.25f), r6_add(el(((k) + (1))), el(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1521
tmpq = mynn_min((float)(mynn_max((float)(r6_mul(b1l, r6_add(pdk(((k) + (1))), pdk(k)))), (float)(qkemin))), (float)(125.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1523
if (initialize_qke) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1524
qke(k) = mynn_pow(tmpq, p666);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1525
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1527
if (((qke(k)) <= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1528
b2l = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1529
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1530
b2l = r6_div(r6_mul(b2, r6_div(b1l, b1)), mynn_sqrt(qke(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1531
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1533
tsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdt(((k) + (1))), pdt(k)))), (float)(zero))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1534
qsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdq(((k) + (1))), pdq(k)))), (float)(zero))), (float)(5e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1535
cov(k) = r6_mul(b2l, r6_add(pdc(((k) + (1))), pdc(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1536
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; b1l = mynn_lane_bcast(b1l, mynn_src3); b2l = mynn_lane_bcast(b2l, mynn_src3); tmpq = mynn_lane_bcast(tmpq, mynn_src3); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1538
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1540
if (initialize_qke) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1541
qke(kts) = r6_mul(p5, r6_add(qke(kts), qke(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1542
qke(kte) = qke(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1543
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1544
tsq(kte) = tsq(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1545
qsq(kte) = qsq(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1546
cov(kte) = cov(((kte) - (1)));
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1592-1791
__device__ __noinline__ void f_mym_level2(int kts, int kte, int bl_mynn_ess, MynnArray<float, 1> zw, MynnArray<float, 1> dz, float xland, float pblh, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> thl, MynnArray<float, 1> thv, MynnArray<float, 1> thlv, MynnArray<float, 1> th, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> qke, MynnArray<float, 1> cldfra, MynnArray<float, 1> edmf_a, MynnArray<float, 1> edmf_w, MynnArray<float, 1> qw, MynnArray<float, 1> ql, MynnArray<float, 1> vt, MynnArray<float, 1> vq, MynnArray<float, 1> dtl, MynnArray<float, 1> dqw, MynnArray<float, 1> dtv, MynnArray<float, 1> gm, MynnArray<float, 1> gh, MynnArray<float, 1> sm, MynnArray<float, 1> sh) {
MynnFrame mynn_frame(arena);
zw.rebase({kts});
dz.rebase({kts});
u.rebase({kts});
v.rebase({kts});
thl.rebase({kts});
thv.rebase({kts});
thlv.rebase({kts});
th.rebase({kts});
p.rebase({kts});
exner.rebase({kts});
qke.rebase({kts});
cldfra.rebase({kts});
edmf_a.rebase({kts});
edmf_w.rebase({kts});
qw.rebase({kts});
ql.rebase({kts});
vt.rebase({kts});
vq.rebase({kts});
dtl.rebase({kts});
dqw.rebase({kts});
dtv.rebase({kts});
gm.rebase({kts});
gh.rebase({kts});
sm.rebase({kts});
sh.rebase({kts});
MynnArray<float, 1> ri = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
int k;
float rfc;
float f1;
float f2;
float rf1;
float rf2;
float smc;
float shc;
float ri1;
float ri2;
float ri3;
float ri4;
float duz;
float dtz;
float dqz;
float vtt;
float vqq;
float dtq;
float dzk;
float afk;
float abk;
float rf;
float a2fac;
float thv1;
float thv0;
float eth0;
float eth1;
float lambda;
float qsat;
float xl;
float tabs;
float clam0;
float clam;
float wt;
float mfi;
float qkei;
float cldfrai;
float ugrid;
float uonset;
float taper;
constexpr float thvp = 0.0f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1636
dtl(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1637
dqw(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1638
dtv(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1639
gm(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1640
gh(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1641
sm(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1642
sh(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1643
ri(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1645
switch (bl_mynn_ess) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1647
case 0: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1649
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1650
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1651
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1652
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1653
duz = r6_add(mynn_ipow<2>(r6_sub(u(k), u(((k) - (1))))), mynn_ipow<2>(r6_sub(v(k), v(((k) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1654
duz = r6_div(duz, mynn_ipow<2>(dzk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1655
dtz = r6_div(r6_sub(thl(k), thl(((k) - (1)))), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1656
dqz = r6_div(r6_sub(qw(k), qw(((k) - (1)))), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1658
vtt = r6_add(r6_add(one, r6_mul(vt(k), abk)), r6_mul(vt(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1659
vqq = r6_add(r6_add(tv0, r6_mul(vq(k), abk)), r6_mul(vq(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1660
dtq = r6_add(r6_mul(vtt, dtz), r6_mul(vqq, dqz));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1662
dtl(k) = dtz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1663
dqw(k) = dqz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1664
dtv(k) = dtq;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1666
gm(k) = duz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1667
gh(k) = r6_neg(r6_mul(dtq, gtr));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1670
ri(k) = r6_neg(r6_div(gh(k), mynn_max((float)(duz), (float)(1.0e-10f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1671
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src0); afk = mynn_lane_bcast(afk, mynn_src0); dqz = mynn_lane_bcast(dqz, mynn_src0); dtq = mynn_lane_bcast(dtq, mynn_src0); dtz = mynn_lane_bcast(dtz, mynn_src0); duz = mynn_lane_bcast(duz, mynn_src0); dzk = mynn_lane_bcast(dzk, mynn_src0); vqq = mynn_lane_bcast(vqq, mynn_src0); vtt = mynn_lane_bcast(vtt, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1673
break; } case 1: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1682
ugrid = mynn_sqrt(r6_add(mynn_ipow<2>(u(kts)), mynn_ipow<2>(v(kts))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1683
uonset = twenty;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1684
taper = r6_sub(one, r6_mul(p9, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, uonset)), (float)(zero)), thirty)), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1685
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1686
clam0 = r6_mul(1.8f, taper);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1687
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1690
clam0 = 0.08f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1691
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1693
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1694
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1695
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1696
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1697
duz = r6_add(mynn_ipow<2>(r6_sub(u(k), u(((k) - (1))))), mynn_ipow<2>(r6_sub(v(k), v(((k) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1698
duz = r6_div(duz, mynn_ipow<2>(dzk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1699
dtz = r6_div(r6_sub(thl(k), thl(((k) - (1)))), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1700
dqz = r6_div(r6_sub(qw(k), qw(((k) - (1)))), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1703
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zw(k), r6_add(pblh, hundred))), (float)(zero)), mynn_max((float)(r6_mul(p3, pblh)), (float)(400.f)))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1704
clam = r6_add(r6_mul(clam0, r6_sub(one, wt)), r6_mul(p2, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1706
clam = r6_mul(clam, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zw(k), twenty)), (float)(zero)), 200.0f)), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1710
thv1 = r6_add(r6_mul(thvp, thv(k)), r6_mul(r6_sub(one, thvp), thlv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1711
thv0 = r6_add(r6_mul(thvp, thv(((k) - (1)))), r6_mul(r6_sub(one, thvp), thlv(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1715
tabs = r6_mul(th(k), exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1716
xl = f_xl_blend(tabs);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1717
qsat = f_qsat_blend(tabs, p(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1718
eth1 = r6_mul(th(k), mynn_exp(r6_div(r6_mul(xl, qsat), r6_mul(cp, tabs))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1720
tabs = r6_mul(th(((k) - (1))), exner(((k) - (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1721
xl = f_xl_blend(tabs);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1722
qsat = f_qsat_blend(tabs, p(((k) - (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1723
eth0 = r6_mul(th(((k) - (1))), mynn_exp(r6_div(r6_mul(xl, qsat), r6_mul(cp, tabs))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1726
mfi = mynn_max((float)(zero), (float)(mynn_min((float)(r6_mul(r6_mul(p5, r6_add(edmf_a(k), edmf_a(((k) - (1))))), r6_mul(p5, r6_add(edmf_w(k), edmf_w(((k) - (1))))))), (float)(p2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1730
qkei = mynn_max((float)(r6_mul(p5, r6_add(qke(k), qke(((k) - (1)))))), (float)(qkemin));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1731
qkei = r6_mul(mynn_min((float)(mynn_sqrt(r6_mul(p25, qkei))), (float)(p5)), p2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1734
cldfrai = r6_mul(p5, r6_add(cldfra(k), cldfra(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1735
cldfrai = r6_mul(mynn_min((float)(cldfrai), (float)(p1)), ten);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1739
lambda = r6_mul(r6_mul(clam, mynn_max((float)(mfi), (float)(qkei))), cldfrai);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1741
dtq = r6_add(r6_div(r6_sub(thv1, thv0), dzk), r6_mul(lambda, mynn_min((float)(r6_div(r6_sub(eth1, eth0), dzk)), (float)(zero))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1744
dtl(k) = dtz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1745
dqw(k) = dqz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1746
dtv(k) = dtq;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1748
gm(k) = duz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1749
gh(k) = r6_neg(r6_mul(dtq, gtr));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1752
ri(k) = r6_neg(r6_div(gh(k), mynn_max((float)(duz), (float)(1.0e-10f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1753
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src1); afk = mynn_lane_bcast(afk, mynn_src1); clam = mynn_lane_bcast(clam, mynn_src1); cldfrai = mynn_lane_bcast(cldfrai, mynn_src1); dqz = mynn_lane_bcast(dqz, mynn_src1); dtq = mynn_lane_bcast(dtq, mynn_src1); dtz = mynn_lane_bcast(dtz, mynn_src1); duz = mynn_lane_bcast(duz, mynn_src1); dzk = mynn_lane_bcast(dzk, mynn_src1); eth0 = mynn_lane_bcast(eth0, mynn_src1); eth1 = mynn_lane_bcast(eth1, mynn_src1); lambda = mynn_lane_bcast(lambda, mynn_src1); mfi = mynn_lane_bcast(mfi, mynn_src1); qkei = mynn_lane_bcast(qkei, mynn_src1); qsat = mynn_lane_bcast(qsat, mynn_src1); tabs = mynn_lane_bcast(tabs, mynn_src1); thv0 = mynn_lane_bcast(thv0, mynn_src1); thv1 = mynn_lane_bcast(thv1, mynn_src1); wt = mynn_lane_bcast(wt, mynn_src1); xl = mynn_lane_bcast(xl, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1755
break; } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1757
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1759
if (((ckmod) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1760
a2fac = r6_div(one, r6_add(one, mynn_max((float)(zero), (float)(ri(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1761
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1762
a2fac = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1763
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1765
rfc = r6_div(g1, r6_add(g1, g2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1766
f1 = r6_add(r6_add(r6_mul(b1, r6_sub(g1, c1)), r6_mul(r6_mul(r6_mul(r6_mul(three, a2), a2fac), r6_sub(one, c2)), r6_sub(one, c5))), r6_mul(r6_mul(two, a1), r6_sub(three, r6_mul(two, c2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1768
f2 = r6_sub(r6_mul(b1, r6_add(g1, g2)), r6_mul(r6_mul(three, a1), r6_sub(one, c2)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1769
rf1 = r6_div(r6_mul(b1, r6_sub(g1, c1)), f1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1770
rf2 = r6_div(r6_mul(b1, g1), f2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1771
smc = r6_div(r6_mul(r6_div(a1, r6_mul(a2, a2fac)), f1), f2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1772
shc = r6_mul(r6_mul(three, r6_mul(a2, a2fac)), r6_add(g1, g2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1774
ri1 = r6_div(p5, smc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1775
ri2 = r6_mul(rf1, smc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1776
ri3 = r6_sub(r6_mul(r6_mul(four, rf2), smc), r6_mul(two, ri2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1777
ri4 = mynn_ipow<2>(ri2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1780
rf = mynn_min((float)(r6_mul(ri1, r6_sub(r6_add(ri(k), ri2), mynn_sqrt(r6_add(r6_sub(mynn_ipow<2>(ri(k)), r6_mul(ri3, ri(k))), ri4))))), (float)(rfc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1782
sh(k) = r6_div(r6_mul(shc, r6_sub(rfc, rf)), r6_sub(one, rf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1783
sm(k) = r6_mul(r6_div(r6_mul(smc, r6_sub(rf1, rf)), r6_sub(rf2, rf)), sh(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1784
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; a2fac = mynn_lane_bcast(a2fac, mynn_src2); f1 = mynn_lane_bcast(f1, mynn_src2); f2 = mynn_lane_bcast(f2, mynn_src2); rf = mynn_lane_bcast(rf, mynn_src2); rf1 = mynn_lane_bcast(rf1, mynn_src2); rf2 = mynn_lane_bcast(rf2, mynn_src2); rfc = mynn_lane_bcast(rfc, mynn_src2); ri1 = mynn_lane_bcast(ri1, mynn_src2); ri2 = mynn_lane_bcast(ri2, mynn_src2); ri3 = mynn_lane_bcast(ri3, mynn_src2); ri4 = mynn_lane_bcast(ri4, mynn_src2); shc = mynn_lane_bcast(shc, mynn_src2); smc = mynn_lane_bcast(smc, mynn_src2); } }
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1811-2246
__device__ __noinline__ void f_mym_length(int kts, int kte, float xland, MynnArray<float, 1> dz, float dx, MynnArray<float, 1> zw, float rmol, float rmolh, float flt, float fltv, float flq, MynnArray<float, 1> vt, MynnArray<float, 1> vq, MynnArray<float, 1> u1, MynnArray<float, 1> v1, MynnArray<float, 1> qke, MynnArray<float, 1> qpe, MynnArray<float, 1> dtv, MynnArray<float, 1> el, float pblh, MynnArray<float, 1> theta, MynnArray<float, 1> qkw, MynnArray<float, 1> qtw, float psig_bl, MynnArray<float, 1> cldfra_bl1, int bl_mynn_mixlength, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> edmf_w_dd1, MynnArray<float, 1> edmf_a_dd1) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
zw.rebase({kts});
vt.rebase({kts});
vq.rebase({kts});
u1.rebase({kts});
v1.rebase({kts});
qke.rebase({kts});
qpe.rebase({kts});
dtv.rebase({kts});
el.rebase({kts});
theta.rebase({kts});
qkw.rebase({kts});
qtw.rebase({kts});
cldfra_bl1.rebase({kts});
edmf_w1.rebase({kts});
edmf_a1.rebase({kts});
edmf_w_dd1.rebase({kts});
edmf_a_dd1.rebase({kts});
float elt;
float vsc;
MynnArray<float, 1> qtke = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> elblmin = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> elblavg = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> thetaw = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
float wt;
float wt2;
float pblh2;
float h1;
float h2;
float hs;
float elblmin0;
float elblavg0;
float cldavg;
float cns;
float alp1;
float alp2;
float alp3;
float alp4;
float alp5;
float alp6;
constexpr float minpblh = 250.f;
constexpr float maxdz = 750.f;
constexpr float mindz = 250.f;
constexpr float zslh = 100.f;
constexpr float csl = 2.f;
constexpr float qkw_elb_min = 0.05f;
int i;
int j;
int k;
float afk;
float abk;
float zwk;
float zwk1;
float dzk;
float qdz;
float vflx;
float bv;
float tau_cloud;
float wstar;
float elb;
float els;
float elf;
float el_stab;
float el_mf;
float el_stab_mf;
float elb_mf;
float elt_max;
float pblh_plus_ent;
float uonset;
float ugrid;
float wt_u1;
float wt_u2;
float el_les;
float qkw_mf;
float z_m;
float el_unstab;
float els1;
float alp3z;
float cpblh;
float wt_dx;
constexpr float ctau = 1000.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1884
switch (bl_mynn_mixlength) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1886
case 0: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1888
cns = 2.7f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1889
alp1 = 0.23f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1890
alp2 = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1891
alp3 = five;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1892
alp4 = hundred;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1893
alp5 = p3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1896
pblh2 = r6_add(pblh, dz(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1897
h1 = mynn_max((float)(r6_mul(0.3f, pblh2)), (float)(mindz));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1898
h1 = mynn_min((float)(h1), (float)(maxdz));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1899
h2 = r6_div(h1, 2.0f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1901
qkw(kts) = mynn_sqrt(mynn_max((float)(qke(kts)), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1902
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1903
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1904
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1905
qkw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(qke(k), abk), r6_mul(qke(((k) - (1))), afk))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1906
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src0); afk = mynn_lane_bcast(afk, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1907
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qtw).size(); mynn_element += lane_width) { qtw.linear(mynn_element) = qkw.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1909
elt = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1910
vsc = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1913
k = ((kts) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1914
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1915
while (((zwk) <= (r6_add(pblh2, h1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1916
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1917
qdz = r6_mul(mynn_max((float)(r6_sub(qkw(k), qmin)), (float)(0.03f)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1918
elt = r6_add(elt, r6_mul(qdz, zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1919
vsc = r6_add(vsc, qdz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1920
k = ((k) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1921
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1922
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1924
elt = r6_div(r6_mul(alp1, elt), vsc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1925
vflx = r6_add(r6_mul(r6_add(vt(kts), one), flt), r6_mul(r6_add(vq(kts), tv0), flq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1926
vsc = mynn_pow(r6_mul(r6_mul(gtr, elt), mynn_max((float)(zero), (float)(vflx))), p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1929
el(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1930
zwk1 = zw(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1932
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1933
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1936
if (((dtv(k)) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1937
bv = mynn_sqrt(r6_mul(gtr, dtv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1938
elb = r6_mul(r6_div(r6_mul(alp2, qkw(k)), bv), r6_add(one, r6_mul(r6_div(alp3, alp2), mynn_sqrt(r6_div(vsc, r6_mul(bv, elt))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1941
elf = r6_div(r6_mul(alp2, qkw(k)), bv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1943
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1944
elb = 1.0e10f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1945
elf = elb;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1946
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1949
if (((rmol) > (0.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1950
els = r6_div(r6_mul(karman, zwk), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1951
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1952
els = r6_mul(r6_mul(karman, zwk), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1953
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1959
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1961
el(k) = mynn_min((float)(r6_div(elb, r6_add(r6_add(r6_div(elb, elt), r6_div(elb, els)), one))), (float)(elf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1963
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1965
break; } case 1: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1968
ugrid = mynn_sqrt(r6_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1969
uonset = twenty;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1970
wt_u1 = r6_sub(one, r6_mul(p2, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, uonset)), (float)(zero)), fifty)), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1971
wt_u2 = r6_sub(one, r6_mul(p5, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, uonset)), (float)(zero)), fifty)), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1973
wt_dx = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(dx, 4000.f)), (float)(zero)), 12000.f)), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1975
cns = 3.5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1976
alp1 = r6_add(r6_mul(0.23f, wt_dx), r6_mul(r6_sub(one, wt_dx), 0.24f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1977
alp2 = r6_add(r6_mul(p3, wt_dx), r6_mul(r6_sub(one, wt_dx), 0.40f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1978
alp3 = r6_mul(five, wt_u2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1979
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1980
alp4 = twenty;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1981
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1982
alp4 = 15.0f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1983
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1984
alp5 = p3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1985
alp6 = fifty;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1988
pblh2 = mynn_max((float)(pblh), (float)(200.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1989
h1 = mynn_max((float)(r6_mul(p3, pblh2)), (float)(300.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1990
h1 = mynn_min((float)(h1), (float)(600.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1991
h2 = r6_mul(h1, p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1993
qtke(kts) = mynn_max((float)(r6_mul(p5, qke(kts))), (float)(r6_mul(p5, qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1994
thetaw(kts) = theta(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1995
qkw(kts) = mynn_sqrt(mynn_max((float)(qke(kts)), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1996
qtw(kts) = mynn_sqrt(mynn_max((float)(r6_add(qke(kts), qpe(kts))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1997
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1998
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1999
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2000
qkw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(qke(k), abk), r6_mul(qke(((k) - (1))), afk))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2001
qtw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(r6_add(qke(k), qpe(k)), abk), r6_mul(r6_add(qke(((k) - (1))), qpe(((k) - (1)))), afk))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2002
qtke(k) = mynn_max((float)(r6_mul(p5, mynn_ipow<2>(qkw(k)))), (float)(0.005f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2003
thetaw(k) = r6_add(r6_mul(theta(k), abk), r6_mul(theta(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2004
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src1); afk = mynn_lane_bcast(afk, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2006
elt = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2007
vsc = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2010
k = ((kts) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2011
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2012
while (((zwk) <= (r6_add(pblh2, mynn_max((float)(mynn_min((float)(r6_mul(p3, pblh2)), (float)(600.f))), (float)(200.f)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2013
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2014
qdz = r6_mul(mynn_min((float)(mynn_max((float)(qkw(k)), (float)(0.01f))), (float)(30.0f)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2015
elt = r6_add(elt, r6_mul(qdz, zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2016
vsc = r6_add(vsc, qdz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2017
k = ((k) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2018
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2019
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2021
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2022
elt_max = r6_add(350.f, r6_mul(100.f, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, fifty)), (float)(zero)), 25.0f)), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2023
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2024
elt_max = 400.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2025
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2026
elt = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(alp1, elt), vsc)), (float)(8.f))), (float)(elt_max));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2029
vflx = r6_sub(fltv, 0.005f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2030
vsc = mynn_pow(r6_mul(r6_mul(gtr, elt), mynn_max((float)(zero), (float)(vflx))), p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2033
el(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2034
zwk1 = zw(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2037
f_boulac_length(kts, kte, zw, dz, qtke, thetaw, elblmin, elblavg);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2039
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2040
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2041
qkw_mf = mynn_max((float)(r6_mul(r6_mul(p5, r6_add(edmf_a1(k), edmf_a1(((k) - (1))))), r6_mul(p5, r6_add(edmf_w1(k), edmf_w1(((k) - (1))))))), (float)(mynn_abs(r6_mul(edmf_a_dd1(((k) - (1))), edmf_w_dd1(((k) - (1)))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2044
cpblh = mynn_min((float)(r6_div(r6_add(zwk, r6_mul(p25, pblh2)), pblh2)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2045
qkw_mf = r6_mul(cpblh, qkw_mf);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2046
alp3z = r6_mul(cpblh, alp3);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2048
if (((dtv(k)) > (0.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2049
bv = mynn_max((float)(mynn_sqrt(r6_mul(gtr, dtv(k)))), (float)(0.001f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2050
elb_mf = r6_div(r6_mul(alp6, qkw_mf), bv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2051
elb_mf = r6_div(elb_mf, r6_add(one, r6_div(elb_mf, 100.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2052
elb = r6_mul(r6_div(r6_mul(alp2, mynn_max((float)(qkw(k)), (float)(qkw_elb_min))), bv), r6_add(one, r6_mul(alp3z, mynn_sqrt(r6_div(vsc, r6_mul(bv, elt))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2054
elb = mynn_max((float)(elb), (float)(elb_mf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2055
elb = mynn_min((float)(elb), (float)(zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2056
elf = r6_div(r6_mul(one, mynn_max((float)(qkw(k)), (float)(qkw_elb_min))), bv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2057
elf = mynn_max((float)(elf), (float)(elb_mf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2058
elblavg(k) = mynn_max((float)(r6_mul(alp5, elblavg(k))), (float)(elb_mf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2059
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2060
elb = 1.0e10f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2061
elf = elb;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2062
elblavg(k) = r6_mul(p4, elblavg(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2063
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2065
z_m = mynn_max((float)(r6_sub(zwk, four)), (float)(karman));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2067
if (((rmol) > (0.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2068
els = r6_div(r6_mul(karman, zwk), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2069
els1 = r6_div(r6_mul(karman, z_m), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2070
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2071
els = r6_mul(r6_mul(karman, zwk), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2072
els1 = r6_mul(r6_mul(karman, z_m), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2073
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2076
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2084
el_unstab = r6_div(els, r6_add(one, r6_div(els1, elt)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2085
el(k) = mynn_min((float)(el_unstab), (float)(elb));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2086
el(k) = mynn_min((float)(el(k)), (float)(elf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2087
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2088
el(k) = r6_mul(el(k), wt_u1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2089
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2090
el(k) = r6_add(r6_mul(el(k), r6_sub(one, wt)), r6_mul(elblavg(k), wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2098
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2100
break; } case 2: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2102
uonset = r6_add(3.5f, r6_mul(dz(kts), p1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2103
ugrid = mynn_sqrt(r6_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2104
cns = 3.5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2105
alp1 = 0.22f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2106
alp2 = p3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2107
alp3 = 2.5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2108
alp4 = five;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2109
alp5 = alp2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2110
alp6 = fifty;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2114
pblh2 = mynn_max((float)(pblh), (float)(200.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2117
h1 = mynn_max((float)(r6_mul(p3, pblh2)), (float)(300.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2118
h1 = mynn_min((float)(h1), (float)(600.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2119
h2 = r6_mul(h1, p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2121
qtke(kts) = mynn_max((float)(r6_mul(p5, qke(kts))), (float)(r6_mul(p5, qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2122
qkw(kts) = mynn_sqrt(mynn_max((float)(qke(kts)), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2123
qtw(kts) = mynn_sqrt(mynn_max((float)(r6_add(qke(kts), qpe(kts))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2124
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2125
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2126
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2127
qkw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(qke(k), abk), r6_mul(qke(((k) - (1))), afk))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2128
qtw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(r6_add(qke(k), qpe(k)), abk), r6_mul(r6_add(qke(((k) - (1))), qpe(((k) - (1)))), afk))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2129
qtke(k) = r6_mul(p5, mynn_ipow<2>(qkw(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2130
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src2); afk = mynn_lane_bcast(afk, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2132
elt = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2133
vsc = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2136
pblh_plus_ent = mynn_max((float)(r6_add(pblh, h1)), (float)(100.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2137
k = ((kts) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2138
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2139
while (((zwk) <= (pblh_plus_ent))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2140
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2141
qdz = r6_mul(mynn_min((float)(mynn_max((float)(r6_sub(qkw(k), qmin)), (float)(0.03f))), (float)(30.0f)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2142
elt = r6_add(elt, r6_mul(qdz, zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2143
vsc = r6_add(vsc, qdz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2144
k = ((k) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2145
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2146
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2148
elt = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(alp1, elt), vsc)), (float)(ten))), (float)(400.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2151
vflx = fltv;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2152
vsc = mynn_pow(r6_mul(r6_mul(gtr, elt), mynn_max((float)(zero), (float)(vflx))), p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2155
el(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2156
zwk1 = zw(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2158
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2159
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2160
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2161
cldavg = r6_mul(p5, r6_add(cldfra_bl1(((k) - (1))), cldfra_bl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2162
qkw_mf = mynn_max((float)(r6_mul(r6_mul(p5, r6_add(edmf_a1(k), edmf_a1(((k) - (1))))), r6_mul(p5, r6_add(edmf_w1(k), edmf_w1(((k) - (1))))))), (float)(mynn_abs(r6_mul(edmf_a_dd1(((k) - (1))), edmf_w_dd1(((k) - (1)))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2166
if (((dtv(k)) > (0.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2168
bv = mynn_max((float)(mynn_sqrt(r6_mul(gtr, dtv(k)))), (float)(0.001f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2170
elb_mf = r6_mul(r6_div(mynn_max((float)(r6_mul(alp2, qkw(k))), (float)(r6_mul(alp6, qkw_mf))), bv), r6_add(one, r6_mul(alp3, mynn_sqrt(r6_div(vsc, r6_mul(bv, elt))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2173
elb = mynn_min((float)(r6_div(mynn_max((float)(r6_mul(alp5, qkw(k))), (float)(r6_mul(alp6, qkw_mf))), bv)), (float)(zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2176
wstar = r6_mul(1.25f, mynn_pow(r6_mul(r6_mul(gtr, pblh), mynn_max((float)(vflx), (float)(1.0e-4f))), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2177
tau_cloud = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(ctau, wstar), grav)), (float)(30.f))), (float)(150.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2179
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2180
tau_cloud = r6_add(r6_mul(tau_cloud, r6_sub(1.f, wt)), r6_mul(fifty, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2181
elf = mynn_min((float)(mynn_max((float)(r6_mul(tau_cloud, mynn_sqrt(mynn_min((float)(qtke(k)), (float)(forty))))), (float)(r6_div(r6_mul(alp6, qkw_mf), bv)))), (float)(zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2191
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2203
wstar = r6_mul(1.25f, mynn_pow(r6_mul(r6_mul(gtr, pblh), mynn_max((float)(vflx), (float)(1.0e-4f))), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2204
tau_cloud = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(ctau, wstar), grav)), (float)(fifty))), (float)(200.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2206
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2208
tau_cloud = r6_add(r6_mul(tau_cloud, r6_sub(one, wt)), r6_mul(mynn_max((float)(r6_mul(dzk, p25)), (float)(hundred)), wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2210
elb = mynn_min((float)(r6_mul(tau_cloud, mynn_sqrt(mynn_min((float)(qtke(k)), (float)(40.f))))), (float)(zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2212
elf = elb;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2213
elb_mf = elb;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2214
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2215
elf = r6_div(elf, r6_add(one, r6_div(elf, 800.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2216
elb_mf = mynn_max((float)(elb_mf), (float)(0.01f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2219
if (((rmol) > (0.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2220
els = r6_div(r6_mul(karman, zwk), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2221
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2222
els = r6_mul(r6_mul(karman, zwk), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2223
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2226
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2229
el(k) = mynn_sqrt(r6_div(mynn_ipow<2>(els), r6_add(r6_add(one, r6_div(mynn_ipow<2>(els), mynn_ipow<2>(elt))), r6_div(mynn_ipow<2>(els), mynn_ipow<2>(elb_mf)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2230
el(k) = r6_add(r6_mul(el(k), r6_sub(one, wt)), r6_mul(elf, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2231
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2233
break; } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2236
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2237
el_les = r6_mul(r6_mul(p25, p5), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2238
el(k) = r6_add(r6_mul(el(k), psig_bl), r6_mul(r6_sub(one, psig_bl), mynn_min((float)(el_les), (float)(el(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2239
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; el_les = mynn_lane_bcast(el_les, mynn_src3); } }
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2261-2409
__device__ __noinline__ void f_boulac_length0(int k, int kts, int kte, MynnArray<float, 1> zw, MynnArray<float, 1> dz, MynnArray<float, 1> qtke, MynnArray<float, 1> theta, float& lb1, float& lb2) {
MynnFrame mynn_frame(arena);
zw.rebase({kts});
dz.rebase({kts});
qtke.rebase({kts});
theta.rebase({kts});
int izz;
int found;
float dlu;
float dld;
float dzt;
float zup;
float beta;
float zup_inf;
float bbb;
float tl;
float zdo;
float zdo_sup;
float zzz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2293
zup = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2294
dlu = r6_sub(r6_sub(zw(((kte) + (1))), zw(k)), r6_mul(dz(k), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2295
zzz = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2296
zup_inf = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2297
beta = gtr;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2301
if (((k) < (kte))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2302
found = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2303
izz = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2304
while (((found) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2306
if (((izz) < (kte))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2307
dzt = dz(izz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2308
zup = r6_sub(zup, r6_mul(r6_mul(beta, theta(k)), dzt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2310
zup = r6_add(zup, r6_mul(r6_mul(r6_mul(beta, r6_add(theta(((izz) + (1))), theta(izz))), dzt), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2311
zzz = r6_add(zzz, dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2313
if (((((qtke(k)) < (zup))) && (((qtke(k)) >= (zup_inf))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2314
bbb = r6_div(r6_sub(theta(((izz) + (1))), theta(izz)), dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2315
if (((bbb) != (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2317
tl = r6_div(r6_div(r6_add(r6_neg(r6_mul(beta, r6_sub(theta(izz), theta(k)))), mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(r6_mul(beta, r6_sub(theta(izz), theta(k)))), r6_mul(r6_mul(r6_mul(two, bbb), beta), r6_sub(qtke(k), zup_inf)))), (float)(0.f)))), bbb), beta);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2320
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2321
if (((theta(izz)) != (theta(k)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2322
tl = r6_div(r6_sub(qtke(k), zup_inf), r6_mul(beta, r6_sub(theta(izz), theta(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2323
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2324
tl = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2325
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2326
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2327
dlu = r6_add(r6_sub(zzz, dzt), tl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2329
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2330
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2331
zup_inf = zup;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2332
izz = ((izz) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2333
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2334
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2335
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2337
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2339
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2344
zdo = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2345
zdo_sup = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2346
dld = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2347
zzz = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2350
if (((k) > (kts))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2352
found = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2353
izz = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2354
while (((found) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2356
if (((izz) > (kts))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2357
dzt = dz(((izz) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2358
zdo = r6_add(zdo, r6_mul(r6_mul(beta, theta(k)), dzt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2360
zdo = r6_sub(zdo, r6_mul(r6_mul(r6_mul(beta, r6_add(theta(((izz) - (1))), theta(izz))), dzt), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2361
zzz = r6_add(zzz, dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2363
if (((((qtke(k)) < (zdo))) && (((qtke(k)) >= (zdo_sup))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2364
bbb = r6_div(r6_sub(theta(izz), theta(((izz) - (1)))), dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2365
if (((bbb) != (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2366
tl = r6_div(r6_div(r6_add(r6_mul(beta, r6_sub(theta(izz), theta(k))), mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(r6_mul(beta, r6_sub(theta(izz), theta(k)))), r6_mul(r6_mul(r6_mul(two, bbb), beta), r6_sub(qtke(k), zdo_sup)))), (float)(0.f)))), bbb), beta);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2369
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2370
if (((theta(izz)) != (theta(k)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2371
tl = r6_div(r6_sub(qtke(k), zdo_sup), r6_mul(beta, r6_sub(theta(izz), theta(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2372
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2373
tl = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2374
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2375
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2376
dld = r6_add(r6_sub(zzz, dzt), tl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2378
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2379
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2380
zdo_sup = zdo;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2381
izz = ((izz) - (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2382
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2383
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2384
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2385
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2387
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2394
dld = mynn_min((float)(dld), (float)(zw(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2395
lb1 = mynn_min((float)(dlu), (float)(dld));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2397
dlu = mynn_max((float)(mynn_min((float)(dlu), (float)(1000.f))), (float)(0.1f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2398
dld = mynn_max((float)(mynn_min((float)(dld), (float)(1000.f))), (float)(0.1f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2399
lb2 = mynn_sqrt(r6_mul(dlu, dld));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2402
if (((k) == (kte))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2403
lb1 = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2404
lb2 = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2405
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2420-2566
__device__ __noinline__ void f_boulac_length(int kts, int kte, MynnArray<float, 1> zw, MynnArray<float, 1> dz, MynnArray<float, 1> qtke, MynnArray<float, 1> theta, MynnArray<float, 1> lb1, MynnArray<float, 1> lb2) {
MynnFrame mynn_frame(arena);
zw.rebase({kts});
dz.rebase({kts});
qtke.rebase({kts});
theta.rebase({kts});
lb1.rebase({kts});
lb2.rebase({kts});
int iz;
int izz;
int found;
MynnArray<float, 1> dlu = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dld = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
constexpr float lmax = 1500.f;
float dzt;
float zup;
float beta;
float zup_inf;
float bbb;
float tl;
float zdo;
float zdo_sup;
float zzz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2442
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { iz = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2447
zup = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2448
dlu(iz) = r6_sub(r6_sub(zw(((kte) + (1))), zw(iz)), r6_mul(dz(iz), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2449
zzz = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2450
zup_inf = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2451
beta = gtr;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2455
if (((iz) < (kte))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2457
found = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2458
izz = iz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2459
while (((found) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2461
if (((izz) < (kte))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2462
dzt = dz(izz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2463
zup = r6_sub(zup, r6_mul(r6_mul(beta, theta(iz)), dzt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2465
zup = r6_add(zup, r6_mul(r6_mul(r6_mul(beta, r6_add(theta(((izz) + (1))), theta(izz))), dzt), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2466
zzz = r6_add(zzz, dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2468
if (((((qtke(iz)) < (zup))) && (((qtke(iz)) >= (zup_inf))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2469
bbb = r6_div(r6_sub(theta(((izz) + (1))), theta(izz)), dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2470
if (((bbb) != (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2472
tl = r6_div(r6_div(r6_add(r6_neg(r6_mul(beta, r6_sub(theta(izz), theta(iz)))), mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(r6_mul(beta, r6_sub(theta(izz), theta(iz)))), r6_mul(r6_mul(r6_mul(two, bbb), beta), r6_sub(qtke(iz), zup_inf)))), (float)(zero)))), bbb), beta);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2475
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2476
if (((theta(izz)) != (theta(iz)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2477
tl = r6_div(r6_sub(qtke(iz), zup_inf), r6_mul(beta, r6_sub(theta(izz), theta(iz))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2478
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2479
tl = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2480
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2481
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2482
dlu(iz) = r6_add(r6_sub(zzz, dzt), tl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2484
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2485
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2486
zup_inf = zup;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2487
izz = ((izz) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2488
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2489
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2490
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2492
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2494
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2499
zdo = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2500
zdo_sup = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2501
dld(iz) = zw(iz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2502
zzz = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2505
if (((iz) > (kts))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2507
found = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2508
izz = iz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2509
while (((found) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2511
if (((izz) > (kts))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2512
dzt = dz(((izz) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2513
zdo = r6_add(zdo, r6_mul(r6_mul(beta, theta(iz)), dzt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2515
zdo = r6_sub(zdo, r6_mul(r6_mul(r6_mul(beta, r6_add(theta(((izz) - (1))), theta(izz))), dzt), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2516
zzz = r6_add(zzz, dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2518
if (((((qtke(iz)) < (zdo))) && (((qtke(iz)) >= (zdo_sup))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2519
bbb = r6_div(r6_sub(theta(izz), theta(((izz) - (1)))), dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2520
if (((bbb) != (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2521
tl = r6_div(r6_div(r6_add(r6_mul(beta, r6_sub(theta(izz), theta(iz))), mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(r6_mul(beta, r6_sub(theta(izz), theta(iz)))), r6_mul(r6_mul(r6_mul(two, bbb), beta), r6_sub(qtke(iz), zdo_sup)))), (float)(zero)))), bbb), beta);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2524
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2525
if (((theta(izz)) != (theta(iz)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2526
tl = r6_div(r6_sub(qtke(iz), zdo_sup), r6_mul(beta, r6_sub(theta(izz), theta(iz))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2527
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2528
tl = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2529
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2530
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2531
dld(iz) = r6_add(r6_sub(zzz, dzt), tl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2533
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2534
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2535
zdo_sup = zdo;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2536
izz = ((izz) - (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2537
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2538
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2539
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2540
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2542
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2548
dld(iz) = mynn_min((float)(dld(iz)), (float)(zw(((iz) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2550
dlu(iz) = mynn_max((float)(r6_div(dlu(iz), r6_add(one, r6_div(dlu(iz), lmax)))), (float)(p1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2551
dld(iz) = mynn_max((float)(r6_div(dld(iz), r6_add(one, r6_div(dld(iz), lmax)))), (float)(p1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2554
lb1(iz) = mynn_min((float)(dlu(iz)), (float)(dld(iz)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2556
lb2(iz) = mynn_sqrt(r6_mul(dlu(iz), dld(iz)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2561
} mynn_lane_sync(); iz = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; beta = mynn_lane_bcast(beta, mynn_src0); zdo = mynn_lane_bcast(zdo, mynn_src0); zdo_sup = mynn_lane_bcast(zdo_sup, mynn_src0); zup = mynn_lane_bcast(zup, mynn_src0); zup_inf = mynn_lane_bcast(zup_inf, mynn_src0); zzz = mynn_lane_bcast(zzz, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2563
lb1(kte) = lb1(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2564
lb2(kte) = lb2(((kte) - (1)));
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2617-3168
__device__ __noinline__ void f_mym_turbulence(int kts, int kte, float xland, float closure, MynnArray<float, 1> dz, float dx, MynnArray<float, 1> zw, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> thl, MynnArray<float, 1> thv, MynnArray<float, 1> thlv, MynnArray<float, 1> ql, MynnArray<float, 1> qw, MynnArray<float, 1> qke, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> vt, MynnArray<float, 1> vq, float rmol, float rmolh, float flt, float fltv, float flq, float pblh, MynnArray<float, 1> theta, MynnArray<float, 1> sh, MynnArray<float, 1> sm, MynnArray<float, 1> el, MynnArray<float, 1> dfm, MynnArray<float, 1> dfh, MynnArray<float, 1> dfq, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> pdk, MynnArray<float, 1> pdt, MynnArray<float, 1> pdq, MynnArray<float, 1> pdc, MynnArray<float, 1> qwt1, MynnArray<float, 1> qshear1, MynnArray<float, 1> qbuoy1, MynnArray<float, 1> qdiss1, int tke_budget, float psig_bl, float psig_shcu, MynnArray<float, 1> cldfra_bl1, int bl_mynn_mixlength, int bl_mynn_ess, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> edmf_w_dd1, MynnArray<float, 1> edmf_a_dd1, MynnArray<float, 1> tkeprod_dn, MynnArray<float, 1> tkeprod_up, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
zw.rebase({kts});
p.rebase({kts});
exner.rebase({kts});
u.rebase({kts});
v.rebase({kts});
thl.rebase({kts});
thv.rebase({kts});
thlv.rebase({kts});
ql.rebase({kts});
qw.rebase({kts});
qke.rebase({kts});
tsq.rebase({kts});
qsq.rebase({kts});
cov.rebase({kts});
vt.rebase({kts});
vq.rebase({kts});
theta.rebase({kts});
sh.rebase({kts});
sm.rebase({kts});
el.rebase({kts});
dfm.rebase({kts});
dfh.rebase({kts});
dfq.rebase({kts});
tcd.rebase({kts});
qcd.rebase({kts});
pdk.rebase({kts});
pdt.rebase({kts});
pdq.rebase({kts});
pdc.rebase({kts});
qwt1.rebase({kts});
qshear1.rebase({kts});
qbuoy1.rebase({kts});
qdiss1.rebase({kts});
cldfra_bl1.rebase({kts});
edmf_w1.rebase({kts});
edmf_a1.rebase({kts});
edmf_w_dd1.rebase({kts});
edmf_a_dd1.rebase({kts});
tkeprod_dn.rebase({kts});
tkeprod_up.rebase({kts});
pattern_spp_pbl1.rebase({kts});
float q3sq_old;
float qwtp_old;
float qwtp_new;
float dudz;
float dvdz;
float dtdz;
float upwp;
float vpwp;
float tpwp;
MynnArray<float, 1> qkw = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qtw = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dtl = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dqw = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dtv = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> gm = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> gh = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qpe = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
int k;
float e6c;
float dzk;
float afk;
float abk;
float vtt;
float vqq;
float cw25;
float clow;
float cupp;
float gamt;
float gamq;
float smd;
float gamv;
float elq;
float elh;
float elqt;
float cldavg;
float a2fac;
float duz;
float ri;
float auh;
float aum;
float adh;
float adm;
float aeh;
float aem;
float req;
float rsl;
float rsl2;
float gmelq;
float sm20;
float sh20;
float sm25max;
float sh25max;
float sm25min;
float sh25min;
float sm_pbl;
float sh_pbl;
float pblh2;
float wt;
float slht;
float mfmax;
float cpblh;
double q2sq;
double t2sq;
double r2sq;
double c2sq;
double elsq;
double gmel;
double ghel;
double q3sq;
double t3sq;
double r3sq;
double c3sq;
double dlsq;
double qdiv;
double e1;
double e2;
double e3;
double e4;
double mynn_enum;
double eden;
double wden;
float shb;
float prlim;
constexpr float prlimit_fre = 6.0f;
constexpr float prlimit_sfc = 4.0f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2707
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qkw).size(); mynn_element += lane_width) { qkw.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2708
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qtw).size(); mynn_element += lane_width) { qtw.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2709
pblh2 = mynn_max((float)(pblh), (float)(200.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2710
if (((closure) < (2.7f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2711
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qpe).size(); mynn_element += lane_width) { qpe.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2712
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2713
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2716
cpblh = mynn_min((float)(r6_div(r6_add(zw(k), r6_mul(0.1f, pblh2)), r6_mul(0.8f, pblh2))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2718
qpe(k) = mynn_min((float)(mynn_max((float)(r6_mul(r6_mul(r6_mul(r6_mul(cpblh, two), tsq(k)), mynn_ipow<2>(gtr)), mynn_ipow<2>(taue))), (float)(zero))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2719
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; cpblh = mynn_lane_bcast(cpblh, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2720
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2722
f_mym_level2(kts, kte, bl_mynn_ess, zw, dz, xland, pblh, u, v, thl, thv, thlv, theta, p, exner, qke, cldfra_bl1, edmf_a1, edmf_w1, qw, ql, vt, vq, dtl, dqw, dtv, gm, gh, sm, sh);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2732
f_mym_length(kts, kte, xland, dz, dx, zw, rmol, rmolh, flt, fltv, flq, vt, vq, u, v, qke, qpe, dtv, el, pblh, theta, qkw, qtw, psig_bl, cldfra_bl1, bl_mynn_mixlength, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2748
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2749
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2750
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2751
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2752
elsq = mynn_ipow<2>(el(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2753
q3sq = mynn_ipow<2>(qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2754
q2sq = mynn_dmul(mynn_dmul(b1, elsq), r6_add(r6_mul(sm(k), gm(k)), r6_mul(sh(k), gh(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2756
sh20 = mynn_max((float)(sh(k)), (float)(1e-5f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2757
sm20 = mynn_max((float)(sm(k)), (float)(1e-5f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2758
sh(k) = mynn_max((float)(sh(k)), (float)(1e-5f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2761
duz = r6_add(mynn_ipow<2>(r6_sub(u(k), u(((k) - (1))))), mynn_ipow<2>(r6_sub(v(k), v(((k) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2762
duz = r6_div(duz, mynn_ipow<2>(dzk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2764
ri = r6_neg(r6_div(gh(k), mynn_max((float)(duz), (float)(1.0e-10f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2765
if (((ckmod) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2766
a2fac = r6_div(one, r6_add(one, mynn_max((float)(zero), (float)(ri))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2767
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2768
a2fac = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2769
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2778
if (((ri) >= (one))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2780
prlim = mynn_min((float)(r6_mul(seven, ri)), (float)(prlimit_fre));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2781
} else if (((((ri) >= (0.01f))) && (((ri) <= (one))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2783
prlim = mynn_min((float)(r6_add(r6_mul(6.873f, ri), r6_div(one, r6_mul(6.873f, ri)))), (float)(prlimit_fre));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2784
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2786
prlim = prlimit_fre;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2787
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2788
wt = mynn_min((float)(r6_div(zw(k), mynn_max((float)(pblh), (float)(200.f)))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2789
prlim = r6_add(r6_mul(wt, prlim), r6_mul(r6_sub(one, wt), mynn_min((float)(prlim), (float)(prlimit_sfc))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2792
gmel = mynn_dmul(gm(k), elsq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2793
ghel = mynn_dmul(gh(k), elsq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2797
if (debug_code) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2798
if (((((sh(k)) < (0.0f))) || (((sm(k)) < (0.0f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2799
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2800
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2801
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2802
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2803
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2804
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2805
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2811
dlsq = elsq;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2812
if (((mynn_ddiv(q3sq, dlsq)) < (r6_neg(gh(k))))) { q3sq = r6_neg(mynn_dmul(dlsq, gh(k))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2814
if (((q3sq) < (q2sq))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2816
qdiv = mynn_sqrt(mynn_ddiv(q3sq, q2sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2832
sh(k) = mynn_dmul(sh(k), qdiv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2833
sm(k) = mynn_dmul(sm(k), qdiv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2841
e1 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e1c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2842
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e2c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2843
e3 = mynn_dadd(e1, mynn_dmul(mynn_dmul(mynn_dmul(e3c, ghel), mynn_ipow<2>(a2fac)), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2844
e4 = mynn_dsub(e1, mynn_dmul(mynn_dmul(mynn_dmul(e4c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2845
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(mynn_dmul(e3, e5c), gmel), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2846
eden = mynn_max((double)(eden), (double)(1.0e-20));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2850
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2856
e1 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(e1c, ghel), a2fac));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2857
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(e2c, ghel), a2fac));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2858
e3 = mynn_dadd(e1, mynn_dmul(mynn_dmul(e3c, ghel), mynn_ipow<2>(a2fac)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2859
e4 = mynn_dsub(e1, mynn_dmul(mynn_dmul(e4c, ghel), a2fac));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2860
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(e3, e5c), gmel));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2861
eden = mynn_max((double)(eden), (double)(1.0e-20));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2863
qdiv = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2865
sm(k) = mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, a1), mynn_dsub(e3, mynn_dmul(r6_mul(three, c1), e4))), eden);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2869
sh(k) = mynn_ddiv(mynn_dmul((tke_budget == 1 ? mynn_dmul(q3sq, r6_mul(a2, a2fac)) : mynn_dadd(mynn_dmul(r6_mul(r6_mul(three, c1), e5c), gmel), e2)), (tke_budget == 1 ? mynn_dadd(mynn_dmul(r6_mul(r6_mul(three, c1), e5c), gmel), e2) : mynn_dmul(q3sq, r6_mul(a2, a2fac)))), eden);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2870
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2873
gmelq = mynn_max((float)(((float)(mynn_ddiv(gmel, q3sq)))), (float)(1e-8f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2874
sm25max = four;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2875
sh25max = four;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2876
sm25min = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2877
sh25min = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2881
if (debug_code) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2882
if (((((((((sh(k)) < (sh25min))) || (((sm(k)) < (sm25min))))) || (((sh(k)) > (sh25max))))) || (((sm(k)) > (sm25max))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2884
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2885
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2886
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2887
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2888
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2889
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2890
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2891
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2892
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2894
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2895
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2898
if (((sh(k)) > (sh25max))) { sh(k) = sh25max; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2899
if (((sh(k)) < (sh25min))) { sh(k) = sh25min; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2903
shb = mynn_max((float)(0.004f), (float)(sh(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2904
sm(k) = mynn_min((float)(sm(k)), (float)(r6_mul(prlim, shb)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2907
if (((closure) >= (3.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2908
t2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, b2), elsq), sh(k)), mynn_ipow<2>(dtl(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2909
r2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, b2), elsq), sh(k)), mynn_ipow<2>(dqw(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2910
c2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, b2), elsq), sh(k)), dtl(k)), dqw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2911
t3sq = mynn_max((float)(zero), (float)(r6_add(r6_mul(tsq(k), abk), r6_mul(tsq(((k) - (1))), afk))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2912
r3sq = mynn_max((float)(zero), (float)(r6_add(r6_mul(qsq(k), abk), r6_mul(qsq(((k) - (1))), afk))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2913
c3sq = r6_add(r6_mul(cov(k), abk), r6_mul(cov(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2916
c3sq = mynn_sign(mynn_min((double)(mynn_abs(c3sq)), (double)(mynn_sqrt(mynn_dmul(t3sq, r3sq)))), c3sq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2918
vtt = r6_add(r6_add(one, r6_mul(vt(k), abk)), r6_mul(vt(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2919
vqq = r6_add(r6_add(tv0, r6_mul(vq(k), abk)), r6_mul(vq(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2921
t2sq = mynn_dadd(mynn_dmul(vtt, t2sq), mynn_dmul(vqq, c2sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2922
r2sq = mynn_dadd(mynn_dmul(vtt, c2sq), mynn_dmul(vqq, r2sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2923
c2sq = mynn_max((double)(0.0e0), (double)(mynn_dadd(mynn_dmul(vtt, t2sq), mynn_dmul(vqq, r2sq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2924
t3sq = mynn_dadd(mynn_dmul(vtt, t3sq), mynn_dmul(vqq, c3sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2925
r3sq = mynn_dadd(mynn_dmul(vtt, c3sq), mynn_dmul(vqq, r3sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2926
c3sq = mynn_max((double)(0.0e0), (double)(mynn_dadd(mynn_dmul(vtt, t3sq), mynn_dmul(vqq, r3sq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2928
cw25 = mynn_ddiv(mynn_dmul(e1, mynn_dadd(e2, mynn_dmul(mynn_dmul(r6_mul(r6_mul(three, c1), e5c), gmel), mynn_ipow<2>(qdiv)))), mynn_dmul(three, eden));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2931
dlsq = elsq;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2932
if (((mynn_ddiv(q3sq, dlsq)) < (r6_neg(gh(k))))) { q3sq = r6_neg(mynn_dmul(dlsq, gh(k))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2937
auh = r6_mul(r6_mul(r6_mul(r6_mul(27.f, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), b2), mynn_ipow<2>(gtr));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2938
aum = r6_mul(r6_mul(r6_mul(r6_mul(r6_mul(54.f, mynn_ipow<2>(a1)), r6_mul(a2, a2fac)), b2), c1), gtr);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2939
adh = r6_mul(r6_mul(r6_mul(r6_mul(nine, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), r6_add(r6_mul(12.f, a1), r6_mul(three, b2))), mynn_ipow<2>(gtr));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2940
adm = r6_mul(r6_mul(r6_mul(r6_mul(18.f, mynn_ipow<2>(a1)), r6_mul(a2, a2fac)), r6_sub(b2, r6_mul(three, r6_mul(a2, a2fac)))), gtr);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2942
aeh = r6_mul(r6_add(r6_mul(r6_mul(r6_mul(nine, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), b1), r6_mul(r6_mul(r6_mul(nine, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), r6_add(r6_mul(12.f, a1), r6_mul(three, b2)))), gtr);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2944
aem = r6_add(r6_mul(r6_mul(r6_mul(r6_mul(three, a1), r6_mul(a2, a2fac)), b1), r6_add(r6_add(r6_mul(three, r6_mul(a2, a2fac)), r6_mul(r6_mul(three, b2), c1)), r6_sub(r6_mul(r6_mul(18.f, a1), c1), b2))), r6_mul(r6_mul(r6_mul(18.f, mynn_ipow<2>(a1)), r6_mul(a2, a2fac)), r6_sub(b2, r6_mul(three, r6_mul(a2, a2fac)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2948
req = r6_neg(r6_div(aeh, aem));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2949
rsl = r6_div(r6_add(auh, r6_mul(aum, req)), r6_add(r6_mul(three, adh), r6_mul(r6_mul(three, adm), req)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2951
rsl = 0.12f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2952
rsl2 = r6_sub(one, r6_mul(two, rsl));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2964
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e2c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2965
e3 = mynn_dadd(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e3c, ghel), mynn_ipow<2>(a2fac)), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2966
e4 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e4c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2967
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(mynn_dmul(e3, e5c), gmel), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2972
wden = mynn_dmul(mynn_dmul(mynn_ddiv(mynn_dmul(r6_mul(cc3, mynn_ipow<2>(gtr)), mynn_ipow<2>(dlsq)), elsq), mynn_ipow<2>(qdiv)), mynn_dsub(mynn_dmul(mynn_dmul(e2, e4c), a2fac), mynn_dmul(mynn_dmul(mynn_dmul(r6_mul(e3c, e5c), gmel), mynn_ipow<2>(a2fac)), mynn_ipow<2>(qdiv))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2975
if (((wden) != (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2977
clow = mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, r6_sub(0.12f, cw25)), eden), wden);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2978
cupp = mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, r6_sub(0.76f, cw25)), eden), wden);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2982
if (((wden) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2983
c3sq = mynn_min((double)(mynn_max((double)(c3sq), (double)(mynn_dadd(c2sq, clow)))), (double)(mynn_dadd(c2sq, cupp)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2984
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2985
c3sq = mynn_max((double)(mynn_min((double)(c3sq), (double)(mynn_dadd(c2sq, clow)))), (double)(mynn_dadd(c2sq, cupp)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2986
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2987
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2989
e1 = mynn_dadd(e2, mynn_dmul(mynn_dmul(e5c, gmel), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2990
eden = mynn_max((double)(eden), (double)(1.0e-20));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2995
e6c = mynn_ddiv(mynn_dmul(r6_mul(r6_mul(r6_mul(3.0f, r6_mul(a2, a2fac)), cc3), gtr), dlsq), elsq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3000
if (((t2sq) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3001
mynn_enum = mynn_max((double)(0.0e0), (double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(t3sq, t2sq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3002
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3003
mynn_enum = mynn_min((double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(t3sq, t2sq))), (double)(0.0e0));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3004
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3005
gamt = r6_neg((float)mynn_ddiv(mynn_dmul(mynn_enum, e1), eden));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3010
if (((r2sq) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3011
mynn_enum = mynn_max((double)(0.0e0), (double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(r3sq, r2sq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3012
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3013
mynn_enum = mynn_min((double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(r3sq, r2sq))), (double)(0.0e0));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3014
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3015
gamq = r6_neg((float)mynn_ddiv(mynn_dmul(mynn_enum, e1), eden));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3020
mynn_enum = mynn_max((double)(0.0e0), (double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(c3sq, c2sq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3024
smd = mynn_ddiv(mynn_dmul(mynn_dmul(mynn_dmul(mynn_ddiv(mynn_dmul(mynn_dmul(dlsq, mynn_enum), gtr), eden), mynn_ipow<2>(qdiv)), r6_add(r6_mul(e3c, mynn_ipow<2>(a2fac)), r6_mul(e4c, a2fac))), a1), r6_mul(a2, a2fac));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3027
gamv = mynn_ddiv(mynn_dmul(mynn_dmul(e1, mynn_enum), gtr), eden);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3028
sm(k) = r6_add(sm(k), smd);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3032
qdiv = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3035
if (debug_code) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3036
if (((((((((sh(k)) < (r6_neg(0.3f)))) || (((sm(k)) < (r6_neg(0.3f)))))) || (((qke(k)) < (r6_neg(0.1f)))))) || (((mynn_abs(smd)) > (2.0f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3038
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3039
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3040
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3041
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3042
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3043
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3044
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3048
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3050
gamt = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3051
gamq = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3052
gamv = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3053
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3057
cldavg = r6_mul(p5, r6_add(cldfra_bl1(((k) - (1))), cldfra_bl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3058
mfmax = mynn_max((float)(r6_mul(r6_mul(r6_mul(p5, r6_add(edmf_a1(((k) - (1))), edmf_a1(k))), p5), r6_add(edmf_w1(((k) - (1))), edmf_w1(k)))), (float)(mynn_abs(r6_mul(edmf_a_dd1(k), edmf_w_dd1(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3060
sm(k) = mynn_max((float)(sm(k)), (float)(r6_mul(0.04f, mynn_min((float)(r6_mul(10.f, mfmax)), (float)(one)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3061
sh(k) = mynn_max((float)(r6_mul(0.04f, mynn_min((float)(r6_mul(10.f, mfmax)), (float)(one)))), (float)(sh(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3063
sm(k) = mynn_max((float)(sm(k)), (float)(r6_mul(0.04f, mynn_min((float)(cldavg), (float)(p5)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3064
sh(k) = mynn_max((float)(r6_mul(0.04f, mynn_min((float)(cldavg), (float)(p5)))), (float)(sh(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3066
if (((closure) == (2.7f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3067
sm(k) = mynn_max((float)(sm(k)), (float)(mynn_min((float)(mynn_max((float)(r6_mul(three, qpe(k))), (float)(zero))), (float)(p2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3068
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3070
elq = r6_mul(el(k), qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3071
elqt = r6_mul(el(k), qtw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3072
elh = mynn_dmul(elq, qdiv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3076
pdk(k) = r6_add(r6_add(r6_mul(elq, r6_add(r6_add(r6_mul(sm(k), gm(k)), r6_mul(sh(k), gh(k))), gamv)), r6_mul(p5, tkeprod_dn(k))), r6_mul(p5, tkeprod_up(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3080
pdt(k) = r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dtl(k)), gamt)), dtl(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3081
pdq(k) = r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dqw(k)), gamq)), dqw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3082
pdc(k) = r6_add(r6_mul(r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dtl(k)), gamt)), dqw(k)), p5), r6_mul(r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dqw(k)), gamq)), dtl(k)), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3086
tcd(k) = r6_mul(elq, gamt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3087
qcd(k) = r6_mul(elq, gamq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3090
dfm(k) = r6_div(r6_mul(elqt, sm(k)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3091
dfh(k) = r6_div(r6_mul(elq, sh(k)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3095
dfq(k) = dfm(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3098
if (((tke_budget) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3114
qshear1(k) = r6_mul(r6_mul(elq, sm(k)), gm(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3122
qbuoy1(k) = r6_add(r6_add(r6_mul(elq, r6_add(r6_mul(gh(k), sh(k)), gamv)), r6_mul(p5, tkeprod_dn(k))), r6_mul(p5, tkeprod_up(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3130
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3132
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; a2fac = mynn_lane_bcast(a2fac, mynn_src1); abk = mynn_lane_bcast(abk, mynn_src1); afk = mynn_lane_bcast(afk, mynn_src1); cldavg = mynn_lane_bcast(cldavg, mynn_src1); dlsq = mynn_lane_bcast(dlsq, mynn_src1); duz = mynn_lane_bcast(duz, mynn_src1); dzk = mynn_lane_bcast(dzk, mynn_src1); e1 = mynn_lane_bcast(e1, mynn_src1); e2 = mynn_lane_bcast(e2, mynn_src1); e3 = mynn_lane_bcast(e3, mynn_src1); e4 = mynn_lane_bcast(e4, mynn_src1); eden = mynn_lane_bcast(eden, mynn_src1); elh = mynn_lane_bcast(elh, mynn_src1); elq = mynn_lane_bcast(elq, mynn_src1); elqt = mynn_lane_bcast(elqt, mynn_src1); elsq = mynn_lane_bcast(elsq, mynn_src1); gamq = mynn_lane_bcast(gamq, mynn_src1); gamt = mynn_lane_bcast(gamt, mynn_src1); gamv = mynn_lane_bcast(gamv, mynn_src1); ghel = mynn_lane_bcast(ghel, mynn_src1); gmel = mynn_lane_bcast(gmel, mynn_src1); gmelq = mynn_lane_bcast(gmelq, mynn_src1); mfmax = mynn_lane_bcast(mfmax, mynn_src1); prlim = mynn_lane_bcast(prlim, mynn_src1); q2sq = mynn_lane_bcast(q2sq, mynn_src1); q3sq = mynn_lane_bcast(q3sq, mynn_src1); qdiv = mynn_lane_bcast(qdiv, mynn_src1); ri = mynn_lane_bcast(ri, mynn_src1); sh20 = mynn_lane_bcast(sh20, mynn_src1); sh25max = mynn_lane_bcast(sh25max, mynn_src1); sh25min = mynn_lane_bcast(sh25min, mynn_src1); shb = mynn_lane_bcast(shb, mynn_src1); sm20 = mynn_lane_bcast(sm20, mynn_src1); sm25max = mynn_lane_bcast(sm25max, mynn_src1); sm25min = mynn_lane_bcast(sm25min, mynn_src1); wt = mynn_lane_bcast(wt, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3135
dfm(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3136
dfh(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3137
dfq(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3138
pdk(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3139
pdt(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3140
pdq(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3141
pdc(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3142
tcd(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3143
qcd(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3145
tcd(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3146
qcd(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3149
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3150
dzk = dz(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3151
tcd(k) = r6_div(r6_sub(tcd(((k) + (1))), tcd(k)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3152
qcd(k) = r6_div(r6_sub(qcd(((k) + (1))), qcd(k)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3153
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3155
if (((spp_pbl) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3156
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3157
dfm(k) = r6_add(dfm(k), r6_mul(r6_mul(r6_mul(dfm(k), pattern_spp_pbl1(k)), 1.5f), mynn_max((float)(mynn_exp(r6_neg(r6_div(mynn_max((float)(zero), (float)(r6_sub(zw(k), 8000.f))), 2000.f)))), (float)(0.001f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3158
dfh(k) = r6_add(dfh(k), r6_mul(r6_mul(r6_mul(dfh(k), pattern_spp_pbl1(k)), 1.5f), mynn_max((float)(mynn_exp(r6_neg(r6_div(mynn_max((float)(zero), (float)(r6_sub(zw(k), 8000.f))), 2000.f)))), (float)(0.001f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3159
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3160
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3216-3607
__device__ __noinline__ void f_mym_predict(int kts, int kte, float closure, float delt, MynnArray<float, 1> dz, float ust, float flt, float flq, float pmz, float phh, MynnArray<float, 1> el, MynnArray<float, 1> dfq, MynnArray<float, 1> rho, MynnArray<float, 1> pdk, MynnArray<float, 1> pdt, MynnArray<float, 1> pdq, MynnArray<float, 1> pdc, MynnArray<float, 1> qke, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> s_aw1, MynnArray<float, 1> s_awqke1, int bl_mynn_edmf, int bl_mynn_edmf_tke, MynnArray<float, 1> qwt1, MynnArray<float, 1> qdiss1, int tke_budget) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
el.rebase({kts});
dfq.rebase({kts});
rho.rebase({kts});
pdk.rebase({kts});
pdt.rebase({kts});
pdq.rebase({kts});
pdc.rebase({kts});
qke.rebase({kts});
tsq.rebase({kts});
qsq.rebase({kts});
cov.rebase({kts});
s_aw1.rebase({kts});
s_awqke1.rebase({kts});
qwt1.rebase({kts});
qdiss1.rebase({kts});
MynnArray<float, 1> tke_up = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dzinv = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
int k;
MynnArray<float, 1> qkw = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> bp = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rp = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> df3q = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
float vkz;
float pdk1;
float phm;
float pdt1;
float pdq1;
float pdc1;
float b1l;
float b2l;
float onoff;
MynnArray<float, 1> dtz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> upcont = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dncont = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> a = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> b = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> c = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> d = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> x = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rhoinv = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rhoz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> kqdz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> kmdz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3258
if (((bl_mynn_edmf_tke) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3259
onoff = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3260
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3261
onoff = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3262
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3265
vkz = r6_mul(r6_mul(karman, p5), dz(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3269
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3271
qkw(k) = mynn_sqrt(mynn_max((float)(qke(k)), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3272
df3q(k) = r6_mul(sqfac, dfq(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3273
dtz(k) = r6_div(delt, dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3274
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3279
rhoz(kts) = rho(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3280
rhoinv(kts) = r6_div(one, rho(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3281
kqdz(kts) = r6_mul(rhoz(kts), df3q(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3282
kmdz(kts) = r6_mul(rhoz(kts), dfq(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3283
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3284
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3285
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3286
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3287
kqdz(k) = r6_mul(rhoz(k), df3q(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3288
kmdz(k) = r6_mul(rhoz(k), dfq(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3289
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3290
rhoz(((kte) + (1))) = rhoz(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3291
kqdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), df3q(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3292
kmdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfq(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3294
if (((bl_mynn_edmf) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3296
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3297
kqdz(k) = mynn_max((float)(kqdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3298
kqdz(k) = mynn_max((float)(kqdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3299
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3300
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3305
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3306
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3308
pdk1 = r6_div(r6_mul(r6_mul(two, mynn_ipow<3>(ust)), pmz), vkz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3309
phm = r6_div(r6_mul(r6_div(two, ust), phh), vkz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3310
pdt1 = r6_mul(phm, mynn_ipow<2>(flt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3311
pdq1 = r6_mul(phm, mynn_ipow<2>(flq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3312
pdc1 = r6_mul(r6_mul(phm, flt), flq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3315
pdk(kts) = r6_sub(pdk1, pdk(((kts) + (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3320
pdt(kts) = pdt(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3321
pdq(kts) = pdq(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3322
pdc(kts) = pdc(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3326
{ const int mynn_lo3 = (kts), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3327
b1l = r6_mul(r6_mul(b1, p5), r6_add(el(((k) + (1))), el(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3328
bp(k) = r6_div(r6_mul(two, qkw(k)), b1l);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3329
rp(k) = r6_add(pdk(((k) + (1))), pdk(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3330
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; b1l = mynn_lane_bcast(b1l, mynn_src3); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3331
bp(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3334
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3336
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3337
upcont(k) = r6_sub(s_awqke1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qke(k), upwind), r6_mul(qke(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3338
dncont(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3339
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3341
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3342
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3344
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3345
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3346
b(1) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), kqdz(((k) + (1)))), rhoinv(k))), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3347
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), kqdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3348
d(1) = r6_sub(r6_add(mynn_max((float)(qke(k)), (float)(qkemin)), r6_mul(rp(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3351
{ const int mynn_lo5 = (((kts) + (1))), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3352
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kqdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3353
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kqdz(k), kqdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3354
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kqdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3355
d(k) = r6_sub(r6_add(mynn_max((float)(qke(k)), (float)(qkemin)), r6_mul(rp(k), delt)), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3357
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3359
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3360
{ const int mynn_lo6 = (kts), mynn_hi6 = (((kte) - (1))), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3366
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), kqdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), s_aw1(k)), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3368
b(k) = r6_add(r6_add(r6_add(1.f, r6_mul(r6_mul(dtz(k), r6_add(kqdz(k), kqdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), onoff)), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3371
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kqdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3373
d(k) = r6_add(r6_add(r6_mul(rp(k), delt), qke(k)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqke1(k), s_awqke1(((k) + (1))))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3375
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3376
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3384
a(kte) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3385
b(kte) = 1.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3386
c(kte) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3387
d(kte) = qke(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3390
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3392
{ const int mynn_lo7 = (kts), mynn_hi7 = (kte), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3394
qke(k) = mynn_max((float)(x(k)), (float)(qkemin));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3395
qke(k) = mynn_min((float)(qke(k)), (float)(150.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3396
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3400
if (((tke_budget) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3402
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tke_up).size(); mynn_element += lane_width) { tke_up.linear(mynn_element) = r6_mul(p5, qke.linear(mynn_element)); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3403
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dzinv).size(); mynn_element += lane_width) { dzinv.linear(mynn_element) = r6_div(one, dz.linear(mynn_element)); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3404
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3405
qwt1(k) = r6_mul(dzinv(k), r6_add(r6_sub(r6_mul(kqdz(((k) + (1))), r6_sub(tke_up(((k) + (1))), tke_up(k))), r6_mul(kqdz(k), tke_up(k))), r6_mul(r6_mul(r6_mul(p5, rhoinv(k)), r6_add(r6_add(r6_mul(s_aw1(((k) + (1))), tke_up(((k) + (1)))), r6_mul(r6_sub(s_aw1(((k) + (1))), s_aw1(k)), tke_up(k))), r6_sub(s_awqke1(k), s_awqke1(((k) + (1)))))), onoff)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3410
{ const int mynn_lo8 = (((kts) + (1))), mynn_hi8 = (((kte) - (1))), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3411
qwt1(k) = r6_mul(dzinv(k), r6_add(r6_sub(r6_mul(kqdz(((k) + (1))), r6_sub(tke_up(((k) + (1))), tke_up(k))), r6_mul(kqdz(k), r6_sub(tke_up(k), tke_up(((k) - (1)))))), r6_mul(r6_mul(r6_mul(p5, rhoinv(k)), r6_add(r6_sub(r6_add(r6_mul(s_aw1(((k) + (1))), tke_up(((k) + (1)))), r6_mul(r6_sub(s_aw1(((k) + (1))), s_aw1(k)), tke_up(k))), r6_mul(s_aw1(k), tke_up(((k) - (1))))), r6_sub(s_awqke1(k), s_awqke1(((k) + (1)))))), onoff)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3417
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3418
k = kte;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3419
qwt1(k) = r6_mul(dzinv(k), r6_add(r6_neg(r6_mul(kqdz(k), r6_sub(tke_up(k), tke_up(((k) - (1)))))), r6_mul(r6_mul(r6_mul(p5, rhoinv(k)), r6_add(r6_sub(r6_neg(r6_mul(s_aw1(k), tke_up(k))), r6_mul(s_aw1(k), tke_up(((k) - (1))))), s_awqke1(k))), onoff)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3422
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qdiss1).size(); mynn_element += lane_width) { qdiss1.linear(mynn_element) = r6_mul(bp.linear(mynn_element), tke_up.linear(mynn_element)); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3423
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3426
if (((closure) > (2.5f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3429
{ const int mynn_lo9 = (kts), mynn_hi9 = (((kte) - (1))), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3430
b2l = r6_mul(r6_mul(b2, p5), r6_add(el(((k) + (1))), el(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3431
bp(k) = r6_div(r6_mul(two, qkw(k)), b2l);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3432
rp(k) = r6_add(pdq(((k) + (1))), pdq(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3433
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; if (mynn_n9 > 0) { const int mynn_src9 = (mynn_n9 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src9); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3442
{ const int mynn_lo10 = (kts), mynn_hi10 = (((kte) - (1))), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3443
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3444
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3445
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3446
d(k) = r6_add(r6_mul(rp(k), delt), mynn_max((float)(zero), (float)(qsq(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3447
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3449
a(kte) = r6_neg(1.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3450
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3451
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3452
d(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3455
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3457
{ const int mynn_lo11 = (kts), mynn_hi11 = (kte), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { k = mynn_lo11 + mynn_it11*mynn_st11;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3459
qsq(k) = mynn_min((float)(mynn_max((float)(x(k)), (float)(1e-17f))), (float)(5e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3460
} mynn_lane_sync(); k = mynn_lo11 + mynn_n11*mynn_st11; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3461
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3463
{ const int mynn_lo12 = (kts), mynn_hi12 = (((kte) - (1))), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { k = mynn_lo12 + mynn_it12*mynn_st12;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3464
if (((qkw(k)) <= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3465
b2l = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3466
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3467
b2l = r6_div(r6_mul(r6_mul(b2, 0.25f), r6_add(el(((k) + (1))), el(k))), qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3468
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3469
qsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdq(((k) + (1))), pdq(k)))), (float)(1e-17f))), (float)(5e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3470
} mynn_lane_sync(); k = mynn_lo12 + mynn_n12*mynn_st12; if (mynn_n12 > 0) { const int mynn_src12 = (mynn_n12 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src12); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3471
qsq(kte) = qsq(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3472
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3475
if (((closure) >= (2.7f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3481
{ const int mynn_lo13 = (kts), mynn_hi13 = (((kte) - (1))), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { k = mynn_lo13 + mynn_it13*mynn_st13;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3482
b2l = r6_mul(r6_mul(b2, p5), r6_add(el(((k) + (1))), el(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3483
bp(k) = r6_div(r6_mul(two, qkw(k)), b2l);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3484
rp(k) = r6_add(pdt(((k) + (1))), pdt(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3485
} mynn_lane_sync(); k = mynn_lo13 + mynn_n13*mynn_st13; if (mynn_n13 > 0) { const int mynn_src13 = (mynn_n13 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src13); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3495
{ const int mynn_lo14 = (kts), mynn_hi14 = (((kte) - (1))), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { k = mynn_lo14 + mynn_it14*mynn_st14;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3501
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3502
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3503
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3504
d(k) = r6_add(r6_mul(rp(k), delt), mynn_max((float)(zero), (float)(tsq(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3505
} mynn_lane_sync(); k = mynn_lo14 + mynn_n14*mynn_st14; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3514
a(kte) = r6_neg(1.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3515
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3516
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3517
d(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3520
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3522
{ const int mynn_lo15 = (kts), mynn_hi15 = (kte), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { k = mynn_lo15 + mynn_it15*mynn_st15;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3524
tsq(k) = mynn_min((float)(mynn_max((float)(zero), (float)(x(k)))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3525
} mynn_lane_sync(); k = mynn_lo15 + mynn_n15*mynn_st15; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3526
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3528
{ const int mynn_lo16 = (kts), mynn_hi16 = (((kte) - (1))), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { k = mynn_lo16 + mynn_it16*mynn_st16;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3529
if (((qkw(k)) <= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3530
b2l = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3531
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3532
b2l = r6_div(r6_mul(r6_mul(b2, 0.25f), r6_add(el(((k) + (1))), el(k))), qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3533
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3534
tsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdt(((k) + (1))), pdt(k)))), (float)(zero))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3535
} mynn_lane_sync(); k = mynn_lo16 + mynn_n16*mynn_st16; if (mynn_n16 > 0) { const int mynn_src16 = (mynn_n16 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src16); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3536
tsq(kte) = tsq(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3537
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3540
if (((closure) >= (3.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3542
{ const int mynn_lo17 = (kts), mynn_hi17 = (((kte) - (1))), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { k = mynn_lo17 + mynn_it17*mynn_st17;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3543
b2l = r6_mul(r6_mul(b2, p5), r6_add(el(((k) + (1))), el(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3544
bp(k) = r6_div(r6_mul(two, qkw(k)), b2l);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3545
rp(k) = r6_add(pdc(((k) + (1))), pdc(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3546
} mynn_lane_sync(); k = mynn_lo17 + mynn_n17*mynn_st17; if (mynn_n17 > 0) { const int mynn_src17 = (mynn_n17 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src17); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3555
{ const int mynn_lo18 = (kts), mynn_hi18 = (((kte) - (1))), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { k = mynn_lo18 + mynn_it18*mynn_st18;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3561
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3562
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3563
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3564
d(k) = r6_add(r6_mul(rp(k), delt), cov(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3565
} mynn_lane_sync(); k = mynn_lo18 + mynn_n18*mynn_st18; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3574
a(kte) = r6_neg(1.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3575
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3576
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3577
d(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3580
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3582
{ const int mynn_lo19 = (kts), mynn_hi19 = (kte), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { k = mynn_lo19 + mynn_it19*mynn_st19;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3584
cov(k) = x(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3585
} mynn_lane_sync(); k = mynn_lo19 + mynn_n19*mynn_st19; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3587
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3590
{ const int mynn_lo20 = (kts), mynn_hi20 = (((kte) - (1))), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { k = mynn_lo20 + mynn_it20*mynn_st20;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3591
if (((qkw(k)) <= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3592
b2l = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3593
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3594
b2l = r6_div(r6_mul(r6_mul(b2, 0.25f), r6_add(el(((k) + (1))), el(k))), qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3595
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3596
cov(k) = r6_mul(b2l, r6_add(pdc(((k) + (1))), pdc(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3597
} mynn_lane_sync(); k = mynn_lo20 + mynn_n20*mynn_st20; if (mynn_n20 > 0) { const int mynn_src20 = (mynn_n20 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src20); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3598
cov(kte) = cov(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3600
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3643-4103
__device__ __noinline__ void f_mym_condensation(int kts, int kte, float dx, MynnArray<float, 1> dz, MynnArray<float, 1> zw, float xland, MynnArray<float, 1> thl, MynnArray<float, 1> qw, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> sh, MynnArray<float, 1> el, int bl_mynn_cloudpdf, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> qi_bl1, MynnArray<float, 1> cldfra_bl1, float pblh1, float hfx1, MynnArray<float, 1> vt, MynnArray<float, 1> vq, MynnArray<float, 1> th, MynnArray<float, 1> sgm, float closure, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
zw.rebase({kts});
thl.rebase({kts});
qw.rebase({kts});
qv.rebase({kts});
qc.rebase({kts});
qi.rebase({kts});
qs.rebase({kts});
p.rebase({kts});
exner.rebase({kts});
tsq.rebase({kts});
qsq.rebase({kts});
cov.rebase({kts});
sh.rebase({kts});
el.rebase({kts});
qc_bl1.rebase({kts});
qi_bl1.rebase({kts});
cldfra_bl1.rebase({kts});
vt.rebase({kts});
vq.rebase({kts});
th.rebase({kts});
sgm.rebase({kts});
pattern_spp_pbl1.rebase({kts});
MynnArray<float, 1> alp = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> a = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> bet = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> b = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> ql = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> q1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rh = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
double t3sq;
double r3sq;
double c3sq;
float qsl;
float esat;
float qsat;
float dqsl;
float cld0;
float q1k;
float qlk;
float eq1;
float qll;
float q2p;
float pt;
float rac;
float qt;
float t;
float xl;
float rsl;
float cpm;
float fng;
float qww;
float alpha;
float beta;
float bb;
float sgmq;
float sgmc;
float ls;
float wt;
float wt2;
float cld_factor;
float fac_damp;
float liq_frac;
float ql_ice;
float ql_water;
float qmq;
float qsat_tk;
float q1_rh;
float rh_adj;
float zsl;
float maxqc;
float cldfra_rh;
float cldfra_qsq;
float cldfra_rh0;
float cldfra_rh1;
float cldfra_qsq0;
float cldfra_qsq1;
float clim;
float qlim;
constexpr float qlim_sfc = 0.008f;
constexpr float qlim_pbl = 0.020f;
constexpr float qlim_trp = 0.025f;
constexpr float clim_sfc = 0.010f;
constexpr float clim_pbl = 0.020f;
constexpr float clim_trp = 0.030f;
constexpr float rhcrit = 0.83f;
constexpr float rhmax = 1.10f;
int i;
int j;
int k;
float erf;
float dth;
float dtl;
float dqw;
float dzk;
float els;
float zagl;
float damp;
float pblh2;
float cfmax;
float theta1;
float theta2;
float ht1;
float ht2;
int k_tropo;
float qw_pert;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3720
for (k=((kte) - (3)); ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3721
theta1 = th(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3722
theta2 = th(((k) + (2)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3723
ht1 = r6_mul(44307.692f, r6_sub(one, mynn_pow(r6_div(p(k), 101325.f), 0.190f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3724
ht2 = r6_mul(44307.692f, r6_sub(one, mynn_pow(r6_div(p(((k) + (2))), 101325.f), 0.190f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3725
if (((((((r6_div(r6_sub(theta2, theta1), r6_sub(ht2, ht1))) < (r6_div(10.f, 1500.f)))) && (((ht1) < (19000.f))))) && (((ht1) > (4000.f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3727
goto mynn_label_86;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3728
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3729
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3730
mynn_label_86: ;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3731
k_tropo = mynn_max((int)(((kts) + (2))), (int)(((k) + (2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3733
zagl = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3735
switch (bl_mynn_cloudpdf) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3737
case 0: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3739
{ const int mynn_lo0 = (kts), mynn_hi0 = (((kte) - (1))), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3740
t = r6_mul(th(k), exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3753
esat = f_esat_blend(t);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3756
qsl = r6_div(r6_mul(ep_2, esat), mynn_max((float)(r6_sub(p(k), r6_mul(ep_3, esat))), (float)(1.e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3758
dqsl = r6_div(r6_mul(r6_mul(qsl, ep_2), xlv), r6_mul(r_d, mynn_ipow<2>(t)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3760
alp(k) = r6_div(one, r6_add(one, r6_mul(dqsl, xlvcp)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3761
bet(k) = r6_mul(dqsl, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3765
t3sq = mynn_max((float)(zero), (float)(tsq(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3766
r3sq = mynn_max((float)(zero), (float)(qsq(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3767
c3sq = cov(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3768
c3sq = mynn_sign(mynn_min((double)(mynn_abs(c3sq)), (double)(mynn_sqrt(mynn_dmul(t3sq, r3sq)))), c3sq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3769
r3sq = mynn_dsub(mynn_dadd(r3sq, mynn_dmul(mynn_ipow<2>(bet(k)), t3sq)), mynn_dmul(r6_mul(2.0f, bet(k)), c3sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3771
qmq = r6_sub(qw(k), qsl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3773
sgm(k) = mynn_sqrt(mynn_max((double)(r3sq), (double)(1.0e-10)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3775
q1(k) = r6_div(qmq, sgm(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3777
cldfra_bl1(k) = r6_mul(0.5f, r6_add(one, mynn_erf(r6_mul(q1(k), rr2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3779
q1k = q1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3780
eq1 = r6_mul(rrp, mynn_exp(r6_mul(r6_neg(r6_mul(0.5f, q1k)), q1k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3781
qll = mynn_max((float)(zero), (float)(r6_add(r6_mul(cldfra_bl1(k), q1k), eq1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3783
ql(k) = r6_mul(r6_mul(alp(k), sgm(k)), qll);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3785
liq_frac = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(t, 240.0f), 29.0f)), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3786
qc_bl1(k) = r6_mul(liq_frac, ql(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3787
qi_bl1(k) = r6_mul(r6_sub(one, liq_frac), ql(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3790
q2p = r6_div(xlvcp, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3791
pt = r6_add(thl(k), r6_mul(q2p, ql(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3794
qt = r6_sub(r6_add(one, r6_mul(p608, qw(k))), r6_mul(r6_mul(r6_add(1.f, p608), r6_add(qc_bl1(k), qi_bl1(k))), cldfra_bl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3795
rac = r6_mul(r6_mul(alp(k), r6_sub(cldfra_bl1(k), r6_mul(qll, eq1))), r6_sub(r6_mul(q2p, qt), r6_mul(r6_add(1.f, p608), pt)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3800
vt(k) = r6_sub(r6_sub(qt, one), r6_mul(rac, bet(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3801
vq(k) = r6_add(r6_sub(r6_mul(p608, pt), tv0), rac);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3803
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; c3sq = mynn_lane_bcast(c3sq, mynn_src0); dqsl = mynn_lane_bcast(dqsl, mynn_src0); eq1 = mynn_lane_bcast(eq1, mynn_src0); esat = mynn_lane_bcast(esat, mynn_src0); liq_frac = mynn_lane_bcast(liq_frac, mynn_src0); pt = mynn_lane_bcast(pt, mynn_src0); q1k = mynn_lane_bcast(q1k, mynn_src0); q2p = mynn_lane_bcast(q2p, mynn_src0); qll = mynn_lane_bcast(qll, mynn_src0); qmq = mynn_lane_bcast(qmq, mynn_src0); qsl = mynn_lane_bcast(qsl, mynn_src0); qt = mynn_lane_bcast(qt, mynn_src0); r3sq = mynn_lane_bcast(r3sq, mynn_src0); rac = mynn_lane_bcast(rac, mynn_src0); t = mynn_lane_bcast(t, mynn_src0); t3sq = mynn_lane_bcast(t3sq, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3805
break; } case 1: case (-(1)): {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3807
{ const int mynn_lo1 = (kts), mynn_hi1 = (((kte) - (1))), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3808
t = r6_mul(th(k), exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3810
esat = f_esat_blend(t);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3813
qsl = r6_div(r6_mul(ep_2, esat), mynn_max((float)(r6_sub(p(k), r6_mul(ep_3, esat))), (float)(1.e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3815
dqsl = r6_div(r6_mul(r6_mul(qsl, ep_2), xlv), r6_mul(r_d, mynn_ipow<2>(t)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3817
alp(k) = r6_div(one, r6_add(one, r6_mul(dqsl, xlvcp)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3818
bet(k) = r6_mul(dqsl, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3820
if (((k) == (kts))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3821
dzk = r6_mul(0.5f, dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3822
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3823
dzk = dz(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3824
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3825
dth = r6_sub(r6_mul(0.5f, r6_add(thl(((k) + (1))), thl(k))), r6_mul(0.5f, r6_add(thl(k), thl(mynn_max((int)(((k) - (1))), (int)(kts))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3826
dqw = r6_sub(r6_mul(0.5f, r6_add(qw(((k) + (1))), qw(k))), r6_mul(0.5f, r6_add(qw(k), qw(mynn_max((int)(((k) - (1))), (int)(kts))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3827
sgm(k) = mynn_sqrt(mynn_max((float)(r6_mul(r6_div(r6_mul(r6_mul(r6_mul(mynn_ipow<2>(alp(k)), mynn_max((float)(mynn_ipow<2>(el(k))), (float)(0.1f))), b2), mynn_max((float)(sh(k)), (float)(0.03f))), 4.f), mynn_ipow<2>(r6_sub(r6_div(dqw, dzk), r6_mul(bet(k), r6_div(dth, dzk)))))), (float)(1.0e-10f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3830
qmq = r6_sub(qw(k), qsl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3831
q1(k) = r6_div(qmq, sgm(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3832
cldfra_bl1(k) = r6_mul(0.5f, r6_add(one, mynn_erf(r6_mul(q1(k), rr2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3837
q1k = q1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3838
eq1 = r6_mul(rrp, mynn_exp(r6_mul(r6_neg(r6_mul(0.5f, q1k)), q1k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3839
qll = mynn_max((float)(zero), (float)(r6_add(r6_mul(cldfra_bl1(k), q1k), eq1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3841
ql(k) = r6_mul(r6_mul(alp(k), sgm(k)), qll);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3842
liq_frac = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(t, 240.0f), 29.0f)), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3843
qc_bl1(k) = r6_mul(liq_frac, ql(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3844
qi_bl1(k) = r6_mul(r6_sub(one, liq_frac), ql(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3847
q2p = r6_div(xlvcp, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3848
pt = r6_add(thl(k), r6_mul(q2p, ql(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3851
qt = r6_sub(r6_add(one, r6_mul(p608, qw(k))), r6_mul(r6_mul(r6_add(1.f, p608), r6_add(qc_bl1(k), qi_bl1(k))), cldfra_bl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3852
rac = r6_mul(r6_mul(alp(k), r6_sub(cldfra_bl1(k), r6_mul(qll, eq1))), r6_sub(r6_mul(q2p, qt), r6_mul(r6_add(1.f, p608), pt)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3857
vt(k) = r6_sub(r6_sub(qt, one), r6_mul(rac, bet(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3858
vq(k) = r6_add(r6_sub(r6_mul(p608, pt), tv0), rac);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3860
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; dqsl = mynn_lane_bcast(dqsl, mynn_src1); dqw = mynn_lane_bcast(dqw, mynn_src1); dth = mynn_lane_bcast(dth, mynn_src1); dzk = mynn_lane_bcast(dzk, mynn_src1); eq1 = mynn_lane_bcast(eq1, mynn_src1); esat = mynn_lane_bcast(esat, mynn_src1); liq_frac = mynn_lane_bcast(liq_frac, mynn_src1); pt = mynn_lane_bcast(pt, mynn_src1); q1k = mynn_lane_bcast(q1k, mynn_src1); q2p = mynn_lane_bcast(q2p, mynn_src1); qll = mynn_lane_bcast(qll, mynn_src1); qmq = mynn_lane_bcast(qmq, mynn_src1); qsl = mynn_lane_bcast(qsl, mynn_src1); qt = mynn_lane_bcast(qt, mynn_src1); rac = mynn_lane_bcast(rac, mynn_src1); t = mynn_lane_bcast(t, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3862
break; } case 2: case (-(2)): {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3866
pblh2 = mynn_max((float)(pblh1), (float)(ten));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3867
{ const int mynn_lo2 = (kts), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3868
zagl = r6_add(zw(k), r6_mul(p5, dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3870
t = r6_mul(th(k), exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3871
xl = f_xl_blend(t);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3872
qsat_tk = f_qsat_blend(t, p(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3873
rh(k) = mynn_max((float)(mynn_min((float)(r6_div(qw(k), mynn_max((float)(qsat_tk), (float)(1e-10f)))), (float)(rhmax))), (float)(0.001f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3876
dqsl = r6_div(r6_mul(r6_mul(qsat_tk, ep_2), xlv), r6_mul(r_d, mynn_ipow<2>(t)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3877
alp(k) = r6_div(one, r6_add(one, r6_mul(dqsl, xlvcp)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3878
bet(k) = r6_mul(dqsl, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3880
rsl = r6_div(r6_mul(xl, qsat_tk), r6_mul(r_v, mynn_ipow<2>(t)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3882
cpm = r6_add(cp, r6_mul(qw(k), cpv));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3883
a(k) = r6_div(one, r6_add(one, r6_div(r6_mul(xl, rsl), cpm)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3884
b(k) = r6_mul(a(k), rsl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3887
qw_pert = r6_add(qw(k), r6_mul(r6_mul(r6_mul(qw(k), p5), pattern_spp_pbl1(k)), ((float)(spp_pbl))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3890
qmq = r6_sub(qw_pert, qsat_tk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3894
r3sq = mynn_max((float)(zero), (float)(qsq(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3895
t3sq = mynn_max((float)(zero), (float)(tsq(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3896
c3sq = cov(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3897
c3sq = mynn_sign(mynn_min((double)(mynn_abs(c3sq)), (double)(mynn_sqrt(mynn_dmul(t3sq, r3sq)))), c3sq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3899
if (((closure) < (2.7f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3900
r3sq = r3sq;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3901
} else if (((((closure) >= (2.7f))) && (((closure) < (3.0f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3902
r3sq = mynn_dadd(r3sq, mynn_dmul(mynn_ipow<2>(bet(k)), t3sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3903
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3904
r3sq = mynn_dsub(mynn_dadd(r3sq, mynn_dmul(mynn_ipow<2>(bet(k)), t3sq)), mynn_dmul(r6_mul(two, bet(k)), c3sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3905
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3908
sgm(k) = mynn_max((float)(mynn_sqrt(((float)(r3sq)))), (float)(1e-13f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3910
sgm(k) = mynn_min((float)(sgm(k)), (float)(r6_mul(qw(k), p5)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3913
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(dz(k), 100.f)), (float)(zero)), 500.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3914
sgm(k) = r6_add(sgm(k), r6_mul(r6_mul(sgm(k), p2), wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3916
sgmq = sgm(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3917
sgmc = sgm(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3920
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, r6_add(pblh2, 10.f))), (float)(zero)), 250.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3921
clim = r6_add(r6_mul(clim_pbl, r6_sub(one, wt)), r6_mul(clim_trp, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3922
zsl = mynn_min((float)(mynn_max((float)(r6_mul(p1, pblh2)), (float)(50.f))), (float)(150.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3923
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, zsl)), (float)(zero)), 150.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3924
clim = r6_add(r6_mul(clim_sfc, r6_sub(one, wt)), r6_mul(clim, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3925
sgmc = mynn_max((float)(sgmc), (float)(r6_mul(qw(k), clim)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3927
sgmc = mynn_max((float)(sgmc), (float)(1e-13f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3929
if (((qmq) >= (zero))) { sgmc = mynn_max((float)(r6_mul(0.02f, qw(k))), (float)(sgmc)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3931
q1(k) = r6_div(qmq, sgmc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3935
wt2 = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, pblh2)), (float)(zero)), 300.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3937
if (((((r6_add(qi(k), qs(k))) > (1.e-10f))) && (((zagl) > (pblh2))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3938
rh_adj = mynn_min((float)(r6_add(rhcrit, r6_mul(r6_mul(wt2, 0.037f), mynn_max((float)(r6_add(ten, mynn_log10(r6_add(qi(k), qs(k))))), (float)(zero))))), (float)(rhmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3939
rh_adj = mynn_max((float)(rh(k)), (float)(rh_adj));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3941
q1_rh = r6_add(r6_neg(3.f), r6_div(r6_mul(three, r6_sub(rh_adj, rhcrit)), r6_sub(one, rhcrit)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3942
q1(k) = mynn_max((float)(q1_rh), (float)(q1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3943
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3945
if (((((qc(k)) > (1.e-5f))) && (((zagl) > (pblh2))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3946
rh_adj = mynn_min((float)(r6_add(rhcrit, r6_mul(r6_mul(wt2, 0.07f), mynn_max((float)(r6_add(five, mynn_log10(qc(k)))), (float)(zero))))), (float)(rhmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3947
rh_adj = mynn_max((float)(rh(k)), (float)(rh_adj));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3949
q1_rh = r6_add(r6_neg(3.f), r6_div(r6_mul(three, r6_sub(rh_adj, rhcrit)), r6_sub(one, rhcrit)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3950
q1(k) = mynn_max((float)(q1_rh), (float)(q1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3951
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3953
q1k = q1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3967
wt2 = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(zagl, r6_sub(pblh1, 100.f)), 200.f)), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3970
cldfra_qsq0 = mynn_max((float)(zero), (float)(mynn_min((float)(r6_add(p5, r6_mul(0.35f, mynn_atan(r6_mul(3.6f, r6_add(q1k, 0.05f)))))), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3972
cldfra_qsq1 = mynn_max((float)(zero), (float)(mynn_min((float)(r6_add(p5, r6_mul(0.39f, mynn_atan(r6_mul(1.6f, r6_add(q1k, 0.55f)))))), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3973
cldfra_qsq = r6_add(r6_mul(cldfra_qsq0, r6_sub(one, wt2)), r6_mul(cldfra_qsq1, wt2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3976
cldfra_rh0 = mynn_min((float)(mynn_max((float)(r6_add(r6_mul(0.56f, mynn_tanh(r6_div(r6_sub(rh(k), 0.976f), 0.030f))), p5)), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3977
cldfra_rh1 = mynn_min((float)(mynn_max((float)(r6_add(r6_mul(0.55f, mynn_tanh(r6_div(r6_sub(rh(k), 0.955f), 0.055f))), p5)), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3978
cldfra_rh = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3980
cldfra_bl1(k) = mynn_max((float)(cldfra_qsq), (float)(cldfra_rh));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3984
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, r6_add(pblh2, ten))), (float)(zero)), 300.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3985
qlim = r6_add(r6_mul(qlim_pbl, r6_sub(one, wt)), r6_mul(qlim_trp, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3986
zsl = mynn_min((float)(mynn_max((float)(r6_mul(p1, pblh2)), (float)(50.f))), (float)(150.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3987
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, zsl)), (float)(zero)), 200.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3988
qlim = r6_add(r6_mul(qlim_sfc, r6_sub(one, wt)), r6_mul(qlim, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3989
sgmq = mynn_max((float)(sgmq), (float)(r6_mul(qw(k), qlim)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3991
ql_water = r6_mul(mynn_min((float)(sgmq), (float)(r6_mul(0.025f, qw(k)))), cldfra_bl1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3992
ql_ice = r6_mul(mynn_min((float)(sgmq), (float)(r6_mul(0.025f, qw(k)))), cldfra_bl1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4016
if (((cldfra_bl1(k)) < (0.001f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4017
ql_ice = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4018
ql_water = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4019
cldfra_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4020
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4022
liq_frac = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(t, tice), r6_sub(tliq, tice))), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4023
qc_bl1(k) = r6_mul(liq_frac, ql_water);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4024
qi_bl1(k) = r6_mul(r6_sub(one, liq_frac), ql_ice);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4028
if (((k) >= (k_tropo))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4029
cldfra_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4030
qc_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4031
qi_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4032
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4038
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4039
q1k = mynn_max((float)(q1(k)), (float)(r6_neg(2.5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4040
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4041
q1k = mynn_max((float)(q1(k)), (float)(r6_neg(2.0f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4042
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4045
if (((q1k) >= (one))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4046
fng = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4047
} else if (((((q1k) >= (r6_neg(1.7f)))) && (((q1k) < (one))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4048
fng = mynn_exp(r6_neg(r6_mul(0.4f, r6_sub(q1k, one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4049
} else if (((((q1k) >= (r6_neg(2.5f)))) && (((q1k) < (r6_neg(1.7f)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4050
fng = r6_add(three, mynn_exp(r6_neg(r6_mul(3.8f, r6_add(q1k, 1.7f)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4051
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4052
fng = mynn_min((float)(r6_add(23.9f, mynn_exp(r6_neg(r6_mul(1.6f, r6_add(q1k, 2.5f)))))), (float)(60.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4053
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4055
cfmax = mynn_min((float)(cldfra_bl1(k)), (float)(0.6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4057
zsl = mynn_min((float)(mynn_max((float)(r6_mul(p1, pblh2)), (float)(25.f))), (float)(hundred));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4058
wt = mynn_min((float)(r6_div(zagl, zsl)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4059
cfmax = r6_mul(cfmax, wt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4061
bb = r6_div(r6_mul(b(k), t), th(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4067
qww = r6_add(one, r6_mul(p608, qw(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4068
alpha = r6_mul(p608, th(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4069
beta = r6_sub(r6_mul(r6_div(th(k), t), r6_div(xl, cp)), r6_mul(1.61f, th(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4070
vt(k) = r6_sub(r6_sub(qww, r6_mul(r6_mul(r6_mul(cfmax, beta), bb), fng)), one);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4071
vq(k) = r6_sub(r6_add(alpha, r6_mul(r6_mul(r6_mul(cfmax, beta), a(k)), fng)), tv0);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4077
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; alpha = mynn_lane_bcast(alpha, mynn_src2); bb = mynn_lane_bcast(bb, mynn_src2); beta = mynn_lane_bcast(beta, mynn_src2); c3sq = mynn_lane_bcast(c3sq, mynn_src2); cfmax = mynn_lane_bcast(cfmax, mynn_src2); cldfra_qsq = mynn_lane_bcast(cldfra_qsq, mynn_src2); cldfra_qsq0 = mynn_lane_bcast(cldfra_qsq0, mynn_src2); cldfra_qsq1 = mynn_lane_bcast(cldfra_qsq1, mynn_src2); cldfra_rh = mynn_lane_bcast(cldfra_rh, mynn_src2); cldfra_rh0 = mynn_lane_bcast(cldfra_rh0, mynn_src2); cldfra_rh1 = mynn_lane_bcast(cldfra_rh1, mynn_src2); clim = mynn_lane_bcast(clim, mynn_src2); cpm = mynn_lane_bcast(cpm, mynn_src2); dqsl = mynn_lane_bcast(dqsl, mynn_src2); fng = mynn_lane_bcast(fng, mynn_src2); liq_frac = mynn_lane_bcast(liq_frac, mynn_src2); q1k = mynn_lane_bcast(q1k, mynn_src2); ql_ice = mynn_lane_bcast(ql_ice, mynn_src2); ql_water = mynn_lane_bcast(ql_water, mynn_src2); qlim = mynn_lane_bcast(qlim, mynn_src2); qmq = mynn_lane_bcast(qmq, mynn_src2); qsat_tk = mynn_lane_bcast(qsat_tk, mynn_src2); qw_pert = mynn_lane_bcast(qw_pert, mynn_src2); qww = mynn_lane_bcast(qww, mynn_src2); r3sq = mynn_lane_bcast(r3sq, mynn_src2); rsl = mynn_lane_bcast(rsl, mynn_src2); sgmc = mynn_lane_bcast(sgmc, mynn_src2); sgmq = mynn_lane_bcast(sgmq, mynn_src2); t = mynn_lane_bcast(t, mynn_src2); t3sq = mynn_lane_bcast(t3sq, mynn_src2); wt = mynn_lane_bcast(wt, mynn_src2); wt2 = mynn_lane_bcast(wt2, mynn_src2); xl = mynn_lane_bcast(xl, mynn_src2); zagl = mynn_lane_bcast(zagl, mynn_src2); zsl = mynn_lane_bcast(zsl, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4079
break; } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4082
if (((bl_mynn_cloudpdf) < (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4083
{ const int mynn_lo3 = (kts), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4084
cldfra_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4085
qc_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4086
qi_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4087
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4088
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4090
ql(kte) = ql(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4091
vt(kte) = vt(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4092
vq(kte) = vq(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4093
qc_bl1(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4094
qi_bl1(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4095
cldfra_bl1(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4096
return;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4109-5679
__device__ __noinline__ void f_mynn_tendencies(int kts, int kte, int i, float delt, MynnArray<float, 1> dz, MynnArray<float, 1> zw, float xland, MynnArray<float, 1> rho, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> th, MynnArray<float, 1> tk, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> qnc, MynnArray<float, 1> qni, float psfc, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> thl, MynnArray<float, 1> sqv, MynnArray<float, 1> sqc, MynnArray<float, 1> sqi, MynnArray<float, 1> sqs, MynnArray<float, 1> sqw, MynnArray<float, 1> thl_tot1, MynnArray<float, 1> qc_tot1, MynnArray<float, 1> qi_tot1, MynnArray<float, 1> qnwfa, MynnArray<float, 1> qnifa, MynnArray<float, 1> qnbca, MynnArray<float, 1> ozone, float ust, float flt, float flq, float flqv, float flqc, float wspd, float uoce, float voce, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> dfm, MynnArray<float, 1> dfh, MynnArray<float, 1> dfq, MynnArray<float, 1> du, MynnArray<float, 1> dv, MynnArray<float, 1> dth, MynnArray<float, 1> dqv, MynnArray<float, 1> dqc, MynnArray<float, 1> dqi, MynnArray<float, 1> dqs, MynnArray<float, 1> dqnc, MynnArray<float, 1> dqni, MynnArray<float, 1> dqnwfa, MynnArray<float, 1> dqnifa, MynnArray<float, 1> dqnbca, MynnArray<float, 1> dozone, MynnArray<float, 1> diss_heat, MynnArray<float, 1> s_aw1, MynnArray<float, 1> s_awthl1, MynnArray<float, 1> s_awqt1, MynnArray<float, 1> s_awqv1, MynnArray<float, 1> s_awqc1, MynnArray<float, 1> s_awu1, MynnArray<float, 1> s_awv1, MynnArray<float, 1> s_awqnc1, MynnArray<float, 1> s_awqni1, MynnArray<float, 1> s_awqnwfa1, MynnArray<float, 1> s_awqnifa1, MynnArray<float, 1> s_awqnbca1, MynnArray<float, 1> sd_aw1, MynnArray<float, 1> sd_awthl1, MynnArray<float, 1> sd_awqt1, MynnArray<float, 1> sd_awqv1, MynnArray<float, 1> sd_awqc1, MynnArray<float, 1> sd_awqi1, MynnArray<float, 1> sd_awqnc1, MynnArray<float, 1> sd_awqni1, MynnArray<float, 1> sd_awqnwfa1, MynnArray<float, 1> sd_awqnifa1, MynnArray<float, 1> sd_awu1, MynnArray<float, 1> sd_awv1, MynnArray<float, 1> sub_thl, MynnArray<float, 1> sub_sqv, MynnArray<float, 1> sub_u, MynnArray<float, 1> sub_v, MynnArray<float, 1> det_thl, MynnArray<float, 1> det_sqv, MynnArray<float, 1> det_sqc, MynnArray<float, 1> det_u, MynnArray<float, 1> det_v, bool flag_qc, bool flag_qi, bool flag_qnc, bool flag_qni, bool flag_qs, bool flag_qnwfa, bool flag_qnifa, bool flag_qnbca, bool flag_ozone, MynnArray<float, 1> cldfra_bl1, int bl_mynn_cloudmix, int bl_mynn_mixqt, int bl_mynn_edmf, int bl_mynn_edmf_mom, int bl_mynn_mixscalars, int bl_mynn_mixaerosols, int bl_mynn_mixnumcon) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
zw.rebase({kts});
rho.rebase({kts});
u.rebase({kts});
v.rebase({kts});
th.rebase({kts});
tk.rebase({kts});
qv.rebase({kts});
qc.rebase({kts});
qi.rebase({kts});
qs.rebase({kts});
qnc.rebase({kts});
qni.rebase({kts});
p.rebase({kts});
exner.rebase({kts});
thl.rebase({kts});
sqv.rebase({kts});
sqc.rebase({kts});
sqi.rebase({kts});
sqs.rebase({kts});
sqw.rebase({kts});
thl_tot1.rebase({kts});
qc_tot1.rebase({kts});
qi_tot1.rebase({kts});
qnwfa.rebase({kts});
qnifa.rebase({kts});
qnbca.rebase({kts});
ozone.rebase({kts});
tsq.rebase({kts});
qsq.rebase({kts});
cov.rebase({kts});
tcd.rebase({kts});
qcd.rebase({kts});
dfm.rebase({kts});
dfh.rebase({kts});
dfq.rebase({kts});
du.rebase({kts});
dv.rebase({kts});
dth.rebase({kts});
dqv.rebase({kts});
dqc.rebase({kts});
dqi.rebase({kts});
dqs.rebase({kts});
dqnc.rebase({kts});
dqni.rebase({kts});
dqnwfa.rebase({kts});
dqnifa.rebase({kts});
dqnbca.rebase({kts});
dozone.rebase({kts});
diss_heat.rebase({kts});
s_aw1.rebase({kts});
s_awthl1.rebase({kts});
s_awqt1.rebase({kts});
s_awqv1.rebase({kts});
s_awqc1.rebase({kts});
s_awu1.rebase({kts});
s_awv1.rebase({kts});
s_awqnc1.rebase({kts});
s_awqni1.rebase({kts});
s_awqnwfa1.rebase({kts});
s_awqnifa1.rebase({kts});
s_awqnbca1.rebase({kts});
sd_aw1.rebase({kts});
sd_awthl1.rebase({kts});
sd_awqt1.rebase({kts});
sd_awqv1.rebase({kts});
sd_awqc1.rebase({kts});
sd_awqi1.rebase({kts});
sd_awqnc1.rebase({kts});
sd_awqni1.rebase({kts});
sd_awqnwfa1.rebase({kts});
sd_awqnifa1.rebase({kts});
sd_awu1.rebase({kts});
sd_awv1.rebase({kts});
sub_thl.rebase({kts});
sub_sqv.rebase({kts});
sub_u.rebase({kts});
sub_v.rebase({kts});
det_thl.rebase({kts});
det_sqv.rebase({kts});
det_sqc.rebase({kts});
det_u.rebase({kts});
det_v.rebase({kts});
cldfra_bl1.rebase({kts});
float wsp;
float wsp2;
float tk2;
float th2;
bool problem;
int kproblem;
MynnArray<float, 1> dtz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dfhc = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dfmc = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> delp = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqv2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqc2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqi2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqs2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqw2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qni2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qnc2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qnwfa2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qnifa2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qnbca2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> ozone2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> zfac = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> plumekh = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rhoinv = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> upcont = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dncont = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> a = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> b = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> c = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> d = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> x = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rhoz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> khdz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> kmdz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
float rhs;
float gfluxm;
float gfluxp;
float dztop;
float maxdfh;
float mindfh;
float maxcf;
float maxkh;
float t;
float esat;
float qsl;
float onoff;
float kh;
float km;
float dzk;
float rhosfc;
float ustovwsp;
float qvflux;
float aero_min;
float aero_max;
float th_new;
float portion_qc;
float portion_qi;
float condensate;
float qsat;
int k;
int kk;
constexpr float nonloc = 1.0f;
constexpr float nc_min = 100.0f;
constexpr float ni_min = 1e-6f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4226
dztop = r6_mul(p5, r6_add(dz(kte), dz(((kte) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4231
if (((bl_mynn_edmf_mom) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4232
onoff = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4233
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4234
onoff = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4235
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4238
if (((wspd) <= (1e-6f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4239
ustovwsp = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4240
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4241
ustovwsp = r6_div(ust, wspd);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4242
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4246
rhosfc = r6_div(psfc, r6_mul(r_d, r6_add(tk(kts), r6_mul(p608, qv(kts)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4247
dtz(kts) = r6_div(delt, dz(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4248
rhoz(kts) = rho(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4249
rhoinv(kts) = r6_div(one, rho(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4250
khdz(kts) = r6_mul(rhoz(kts), dfh(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4251
kmdz(kts) = r6_mul(rhoz(kts), dfm(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4252
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4253
dtz(k) = r6_div(delt, dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4254
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4255
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4256
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4257
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4258
khdz(k) = r6_mul(rhoz(k), dfh(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4259
kmdz(k) = r6_mul(rhoz(k), dfm(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4260
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4261
rhoz(((kte) + (1))) = rhoz(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4262
khdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfh(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4263
kmdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfm(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4267
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4270
delp(k) = r6_mul(r6_mul(rho(k), grav), dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4271
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4273
if (((delp(kts)) < (r6_mul(p333, delp(((kts) + (1))))))) { delp(kts) = r6_mul(p333, delp(((kts) + (1)))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4276
if (((bl_mynn_edmf) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4277
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4278
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_mul(p5, r6_add(s_aw1(k), sd_aw1(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4279
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_sub(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(p5, r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4281
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_mul(p5, r6_add(s_aw1(k), sd_aw1(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4282
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_sub(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(p5, r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4284
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4285
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4287
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dth.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { dth.slice(MynnRange{kts, kte}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4292
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4297
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4298
upcont(k) = r6_mul(onoff, r6_sub(s_awu1(k), r6_mul(s_aw1(k), r6_add(r6_mul(u(k), upwind), r6_mul(u(((k) - (1))), r6_sub(one, upwind))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4299
dncont(k) = r6_mul(onoff, r6_sub(sd_awu1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(u(k), upwind), r6_mul(u(((k) - (1))), r6_sub(one, upwind))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4300
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4302
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4303
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4306
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4307
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4308
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_mul(r6_mul(rhosfc, ust), ustovwsp))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4309
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4310
d(1) = r6_sub(r6_add(u(k), r6_mul(r6_div(r6_mul(r6_mul(dtz(k), uoce), mynn_ipow<2>(ust)), wspd), rho(k))), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4313
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4314
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4315
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4316
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4317
d(k) = r6_add(r6_add(r6_sub(u(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_u(k), delt)), r6_mul(det_u(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4319
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4321
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4323
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4325
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4326
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_mul(r6_mul(rhosfc, ust), ustovwsp))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4329
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4332
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(u(k), r6_div(r6_mul(r6_mul(dtz(k), uoce), mynn_ipow<2>(ust)), wspd)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awu1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awu1(((k) + (1)))), onoff)), r6_mul(sub_u(k), delt)), r6_mul(det_u(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4337
{ const int mynn_lo5 = (((kts) + (1))), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4338
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4341
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4344
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4347
d(k) = r6_add(r6_add(r6_sub(r6_add(u(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awu1(k), s_awu1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awu1(k), sd_awu1(((k) + (1))))), onoff)), r6_mul(sub_u(k), delt)), r6_mul(det_u(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4350
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4352
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4367
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4368
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4369
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4370
d(kte) = u(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4373
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4376
{ const int mynn_lo6 = (kts), mynn_hi6 = (kte), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4378
du(k) = r6_div(r6_sub(x(k), u(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4379
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4385
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4387
{ const int mynn_lo7 = (((kts) + (1))), mynn_hi7 = (((kte) - (1))), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4388
upcont(k) = r6_mul(onoff, r6_sub(s_awv1(k), r6_mul(s_aw1(k), r6_add(r6_mul(v(k), upwind), r6_mul(v(((k) - (1))), r6_sub(one, upwind))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4389
dncont(k) = r6_mul(onoff, r6_sub(sd_awv1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(v(k), upwind), r6_mul(v(((k) - (1))), r6_sub(one, upwind))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4390
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4392
upcont(kte) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4393
dncont(kte) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4396
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4397
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4398
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_mul(r6_mul(rhosfc, ust), ustovwsp))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4399
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4400
d(1) = r6_sub(r6_add(v(k), r6_mul(r6_div(r6_mul(r6_mul(dtz(k), voce), mynn_ipow<2>(ust)), wspd), rho(k))), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4403
{ const int mynn_lo8 = (((kts) + (1))), mynn_hi8 = (((kte) - (1))), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4404
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4405
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4406
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4407
d(k) = r6_add(r6_add(r6_sub(v(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_v(k), delt)), r6_mul(det_v(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4409
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4411
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4413
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4415
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4416
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_div(r6_mul(rhosfc, mynn_ipow<2>(ust)), wspd))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4419
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4422
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(v(k), r6_div(r6_mul(r6_mul(dtz(k), voce), mynn_ipow<2>(ust)), wspd)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awv1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awv1(((k) + (1)))), onoff)), r6_mul(sub_v(k), delt)), r6_mul(det_v(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4427
{ const int mynn_lo9 = (((kts) + (1))), mynn_hi9 = (((kte) - (1))), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4428
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4431
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4434
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4437
d(k) = r6_add(r6_add(r6_sub(r6_add(v(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awv1(k), s_awv1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awv1(k), sd_awv1(((k) + (1))))), onoff)), r6_mul(sub_v(k), delt)), r6_mul(det_v(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4440
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4441
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4455
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4456
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4457
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4458
d(kte) = v(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4461
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4464
{ const int mynn_lo10 = (kts), mynn_hi10 = (kte), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4466
dv(k) = r6_div(r6_sub(x(k), v(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4467
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4472
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4474
{ const int mynn_lo11 = (((kts) + (1))), mynn_hi11 = (((kte) - (1))), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { k = mynn_lo11 + mynn_it11*mynn_st11;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4475
upcont(k) = r6_sub(s_awthl1(k), r6_mul(s_aw1(k), r6_add(r6_mul(thl(k), upwind), r6_mul(thl(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4476
dncont(k) = r6_sub(sd_awthl1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(thl(k), upwind), r6_mul(thl(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4477
} mynn_lane_sync(); k = mynn_lo11 + mynn_n11*mynn_st11; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4479
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4480
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4483
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4484
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4485
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4486
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4487
d(1) = r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(thl(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flt), rhoinv(k))), r6_mul(tcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4491
{ const int mynn_lo12 = (((kts) + (1))), mynn_hi12 = (((kte) - (1))), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { k = mynn_lo12 + mynn_it12*mynn_st12;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4492
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4493
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4494
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4495
d(k) = r6_add(r6_add(r6_add(r6_sub(thl(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4497
} mynn_lane_sync(); k = mynn_lo12 + mynn_n12*mynn_st12; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4499
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4501
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4503
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4504
b(k) = r6_sub(r6_sub(r6_add(1.f, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4507
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4510
d(k) = r6_add(r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(thl(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flt), rhoinv(k))), r6_mul(tcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awthl1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awthl1(((k) + (1))))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4516
{ const int mynn_lo13 = (((kts) + (1))), mynn_hi13 = (((kte) - (1))), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { k = mynn_lo13 + mynn_it13*mynn_st13;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4517
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4520
b(k) = r6_add(r6_add(r6_add(1.f, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4523
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4526
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(thl(k), r6_mul(tcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awthl1(k), s_awthl1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awthl1(k), sd_awthl1(((k) + (1)))))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4532
} mynn_lane_sync(); k = mynn_lo13 + mynn_n13*mynn_st13; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4533
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4548
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4549
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4550
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4551
d(kte) = thl(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4554
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4557
{ const int mynn_lo14 = (kts), mynn_hi14 = (kte), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { k = mynn_lo14 + mynn_it14*mynn_st14;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4559
thl(k) = x(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4560
} mynn_lane_sync(); k = mynn_lo14 + mynn_n14*mynn_st14; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4562
if (((bl_mynn_mixqt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4569
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4571
{ const int mynn_lo15 = (((kts) + (1))), mynn_hi15 = (((kte) - (1))), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { k = mynn_lo15 + mynn_it15*mynn_st15;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4572
upcont(k) = r6_sub(s_awqt1(k), r6_mul(s_aw1(k), r6_add(r6_mul(sqw(k), upwind), r6_mul(sqw(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4573
dncont(k) = r6_sub(sd_awqt1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqw(k), upwind), r6_mul(sqw(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4574
} mynn_lane_sync(); k = mynn_lo15 + mynn_n15*mynn_st15; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4576
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4577
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4580
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4581
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4582
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4583
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4584
d(1) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqw(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flq), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4588
{ const int mynn_lo16 = (((kts) + (1))), mynn_hi16 = (((kte) - (1))), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { k = mynn_lo16 + mynn_it16*mynn_st16;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4589
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4590
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4591
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4592
d(k) = r6_add(r6_add(r6_sub(sqw(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4594
} mynn_lane_sync(); k = mynn_lo16 + mynn_n16*mynn_st16; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4596
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4598
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4600
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4601
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4604
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4607
d(k) = r6_add(r6_sub(r6_add(r6_add(sqw(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flq), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqt1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqt1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4611
{ const int mynn_lo17 = (((kts) + (1))), mynn_hi17 = (((kte) - (1))), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { k = mynn_lo17 + mynn_it17*mynn_st17;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4612
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4615
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4618
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4621
d(k) = r6_sub(r6_add(r6_add(sqw(k), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqt1(k), s_awqt1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqt1(k), sd_awqt1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4624
} mynn_lane_sync(); k = mynn_lo17 + mynn_n17*mynn_st17; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4625
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4638
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4639
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4640
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4641
d(kte) = sqw(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4644
f_tridiag2(kte, a, b, c, d, sqw2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4650
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4651
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqw2).size(); mynn_element += lane_width) { sqw2.linear(mynn_element) = sqw.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4652
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4654
if (((bl_mynn_mixqt) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4659
if (((((bl_mynn_cloudmix) > (0))) && (flag_qc))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4661
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4663
{ const int mynn_lo18 = (((kts) + (1))), mynn_hi18 = (((kte) - (1))), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { k = mynn_lo18 + mynn_it18*mynn_st18;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4664
upcont(k) = r6_sub(s_awqc1(k), r6_mul(s_aw1(k), r6_add(r6_mul(sqc(k), upwind), r6_mul(sqc(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4665
dncont(k) = r6_sub(sd_awqc1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqc(k), upwind), r6_mul(sqc(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4666
} mynn_lane_sync(); k = mynn_lo18 + mynn_n18*mynn_st18; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4668
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4669
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4672
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4673
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4674
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4675
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4676
d(1) = r6_add(r6_sub(r6_add(r6_add(sqc(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flqc), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(det_sqc(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4680
{ const int mynn_lo19 = (((kts) + (1))), mynn_hi19 = (((kte) - (1))), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { k = mynn_lo19 + mynn_it19*mynn_st19;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4681
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4682
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4683
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4684
d(k) = r6_add(r6_sub(sqc(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(det_sqc(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4686
} mynn_lane_sync(); k = mynn_lo19 + mynn_n19*mynn_st19; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4688
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4690
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4692
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4693
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4696
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4699
d(k) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqc(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flqc), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqc1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqc1(((k) + (1))))), r6_mul(det_sqc(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4704
{ const int mynn_lo20 = (((kts) + (1))), mynn_hi20 = (((kte) - (1))), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { k = mynn_lo20 + mynn_it20*mynn_st20;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4705
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4708
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4711
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4714
d(k) = r6_add(r6_sub(r6_add(r6_add(sqc(k), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqc1(k), s_awqc1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqc1(k), sd_awqc1(((k) + (1)))))), r6_mul(det_sqc(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4718
} mynn_lane_sync(); k = mynn_lo20 + mynn_n20*mynn_st20; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4719
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4722
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4723
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4724
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4725
d(kte) = sqc(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4728
f_tridiag2(kte, a, b, c, d, sqc2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4734
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4736
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqc2).size(); mynn_element += lane_width) { sqc2.linear(mynn_element) = sqc.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4737
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4738
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4740
if (((bl_mynn_mixqt) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4746
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4748
qvflux = flqv;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4749
if (((qvflux) < (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4751
qvflux = mynn_max((float)(qvflux), (float)(r6_div(mynn_min((float)(r6_sub(r6_mul(0.9f, sqv(kts)), 1e-8f)), (float)(zero)), dtz(kts))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4752
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4754
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4756
{ const int mynn_lo21 = (((kts) + (1))), mynn_hi21 = (((kte) - (1))), mynn_st21 = (1); const int mynn_n21 = mynn_trip(mynn_lo21, mynn_hi21, mynn_st21); mynn_lane_sync(); for (int mynn_it21 = lane_sub; mynn_it21 < mynn_n21; mynn_it21 += lane_width) { k = mynn_lo21 + mynn_it21*mynn_st21;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4757
upcont(k) = r6_sub(s_awqv1(k), r6_mul(s_aw1(k), r6_add(r6_mul(sqv(k), upwind), r6_mul(sqv(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4758
dncont(k) = r6_sub(sd_awqv1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqv(k), upwind), r6_mul(sqv(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4759
} mynn_lane_sync(); k = mynn_lo21 + mynn_n21*mynn_st21; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4761
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4762
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4765
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4766
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4767
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4768
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4769
d(1) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqv(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), qvflux), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4773
{ const int mynn_lo22 = (((kts) + (1))), mynn_hi22 = (((kte) - (1))), mynn_st22 = (1); const int mynn_n22 = mynn_trip(mynn_lo22, mynn_hi22, mynn_st22); mynn_lane_sync(); for (int mynn_it22 = lane_sub; mynn_it22 < mynn_n22; mynn_it22 += lane_width) { k = mynn_lo22 + mynn_it22*mynn_st22;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4774
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4775
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4776
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4777
d(k) = r6_add(r6_add(r6_sub(sqv(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4779
} mynn_lane_sync(); k = mynn_lo22 + mynn_n22*mynn_st22; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4781
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4783
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4785
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4786
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4789
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4792
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(sqv(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), qvflux), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqv1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqv1(((k) + (1))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4797
{ const int mynn_lo23 = (((kts) + (1))), mynn_hi23 = (((kte) - (1))), mynn_st23 = (1); const int mynn_n23 = mynn_trip(mynn_lo23, mynn_hi23, mynn_st23); mynn_lane_sync(); for (int mynn_it23 = lane_sub; mynn_it23 < mynn_n23; mynn_it23 += lane_width) { k = mynn_lo23 + mynn_it23*mynn_st23;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4798
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4801
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4804
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4807
d(k) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqv(k), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqv1(k), s_awqv1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqv1(k), sd_awqv1(((k) + (1)))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4811
} mynn_lane_sync(); k = mynn_lo23 + mynn_n23*mynn_st23; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4812
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4828
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4829
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4830
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4831
d(kte) = sqv(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4834
f_tridiag2(kte, a, b, c, d, sqv2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4840
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4841
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqv2).size(); mynn_element += lane_width) { sqv2.linear(mynn_element) = sqv.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4842
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4847
if (((((bl_mynn_cloudmix) > (0))) && (flag_qi))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4849
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4851
{ const int mynn_lo24 = (((kts) + (1))), mynn_hi24 = (((kte) - (1))), mynn_st24 = (1); const int mynn_n24 = mynn_trip(mynn_lo24, mynn_hi24, mynn_st24); mynn_lane_sync(); for (int mynn_it24 = lane_sub; mynn_it24 < mynn_n24; mynn_it24 += lane_width) { k = mynn_lo24 + mynn_it24*mynn_st24;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4852
upcont(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4853
dncont(k) = r6_sub(sd_awqi1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqi(k), upwind), r6_mul(sqi(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4854
} mynn_lane_sync(); k = mynn_lo24 + mynn_n24*mynn_st24; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4856
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4857
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4860
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4861
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4862
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4863
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4864
d(1) = r6_sub(sqi(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4867
{ const int mynn_lo25 = (((kts) + (1))), mynn_hi25 = (((kte) - (1))), mynn_st25 = (1); const int mynn_n25 = mynn_trip(mynn_lo25, mynn_hi25, mynn_st25); mynn_lane_sync(); for (int mynn_it25 = lane_sub; mynn_it25 < mynn_n25; mynn_it25 += lane_width) { k = mynn_lo25 + mynn_it25*mynn_st25;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4868
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4869
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4870
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4871
d(k) = r6_sub(sqi(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4872
} mynn_lane_sync(); k = mynn_lo25 + mynn_n25*mynn_st25; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4874
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4876
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4877
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4878
b(k) = r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4881
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4884
d(k) = r6_add(sqi(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqi1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4888
{ const int mynn_lo26 = (((kts) + (1))), mynn_hi26 = (((kte) - (1))), mynn_st26 = (1); const int mynn_n26 = mynn_trip(mynn_lo26, mynn_hi26, mynn_st26); mynn_lane_sync(); for (int mynn_it26 = lane_sub; mynn_it26 < mynn_n26; mynn_it26 += lane_width) { k = mynn_lo26 + mynn_it26*mynn_st26;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4889
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4892
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4895
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4898
d(k) = r6_sub(sqi(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqi1(k), sd_awqi1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4901
} mynn_lane_sync(); k = mynn_lo26 + mynn_n26*mynn_st26; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4902
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4918
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4919
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4920
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4921
d(kte) = sqi(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4924
f_tridiag2(kte, a, b, c, d, sqi2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4930
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4931
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqi2).size(); mynn_element += lane_width) { sqi2.linear(mynn_element) = sqi.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4932
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4938
if (((((bl_mynn_cloudmix) > (0))) && (false))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4940
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4942
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4943
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4944
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4945
d(k) = sqs(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4947
{ const int mynn_lo27 = (((kts) + (1))), mynn_hi27 = (((kte) - (1))), mynn_st27 = (1); const int mynn_n27 = mynn_trip(mynn_lo27, mynn_hi27, mynn_st27); mynn_lane_sync(); for (int mynn_it27 = lane_sub; mynn_it27 < mynn_n27; mynn_it27 += lane_width) { k = mynn_lo27 + mynn_it27*mynn_st27;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4948
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4949
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4950
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4951
d(k) = sqs(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4952
} mynn_lane_sync(); k = mynn_lo27 + mynn_n27*mynn_st27; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4955
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4956
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4957
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4958
d(kte) = sqs(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4961
f_tridiag2(kte, a, b, c, d, sqs2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4967
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4968
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqs2).size(); mynn_element += lane_width) { sqs2.linear(mynn_element) = sqs.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4969
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4974
if (((((((bl_mynn_cloudmix) > (0))) && (flag_qni))) && (((bl_mynn_mixnumcon) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4977
{ const int mynn_lo28 = (kts), mynn_hi28 = (kte), mynn_st28 = (1); const int mynn_n28 = mynn_trip(mynn_lo28, mynn_hi28, mynn_st28); mynn_lane_sync(); for (int mynn_it28 = lane_sub; mynn_it28 < mynn_n28; mynn_it28 += lane_width) { k = mynn_lo28 + mynn_it28*mynn_st28;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4978
qni2(k) = mynn_max((float)(zero), (float)(qni(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4980
if (((sqi(k)) > (1e-12f))) { qni2(k) = mynn_max((float)(qni2(k)), (float)(ni_min)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4981
} mynn_lane_sync(); k = mynn_lo28 + mynn_n28*mynn_st28; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4983
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4985
{ const int mynn_lo29 = (((kts) + (1))), mynn_hi29 = (((kte) - (1))), mynn_st29 = (1); const int mynn_n29 = mynn_trip(mynn_lo29, mynn_hi29, mynn_st29); mynn_lane_sync(); for (int mynn_it29 = lane_sub; mynn_it29 < mynn_n29; mynn_it29 += lane_width) { k = mynn_lo29 + mynn_it29*mynn_st29;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4986
upcont(k) = r6_sub(s_awqni1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qni2(k), upwind), r6_mul(qni2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4987
dncont(k) = r6_sub(sd_awqni1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qni2(k), upwind), r6_mul(qni2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4988
} mynn_lane_sync(); k = mynn_lo29 + mynn_n29*mynn_st29; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4990
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4991
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4994
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4995
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4996
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4997
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4998
d(1) = r6_sub(qni2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5001
{ const int mynn_lo30 = (((kts) + (1))), mynn_hi30 = (((kte) - (1))), mynn_st30 = (1); const int mynn_n30 = mynn_trip(mynn_lo30, mynn_hi30, mynn_st30); mynn_lane_sync(); for (int mynn_it30 = lane_sub; mynn_it30 < mynn_n30; mynn_it30 += lane_width) { k = mynn_lo30 + mynn_it30*mynn_st30;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5002
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5003
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5004
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5005
d(k) = r6_sub(qni2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5006
} mynn_lane_sync(); k = mynn_lo30 + mynn_n30*mynn_st30; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5008
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5010
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5011
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5012
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5015
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5018
d(k) = r6_add(r6_sub(qni2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqni1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqni1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5022
{ const int mynn_lo31 = (((kts) + (1))), mynn_hi31 = (((kte) - (1))), mynn_st31 = (1); const int mynn_n31 = mynn_trip(mynn_lo31, mynn_hi31, mynn_st31); mynn_lane_sync(); for (int mynn_it31 = lane_sub; mynn_it31 < mynn_n31; mynn_it31 += lane_width) { k = mynn_lo31 + mynn_it31*mynn_st31;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5023
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5026
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5029
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5032
d(k) = r6_sub(r6_add(qni2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqni1(k), s_awqni1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqni1(k), sd_awqni1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5035
} mynn_lane_sync(); k = mynn_lo31 + mynn_n31*mynn_st31; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5036
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5039
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5040
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5041
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5042
d(kte) = qni2(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5045
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5048
{ const int mynn_lo32 = (kts), mynn_hi32 = (kte), mynn_st32 = (1); const int mynn_n32 = mynn_trip(mynn_lo32, mynn_hi32, mynn_st32); mynn_lane_sync(); for (int mynn_it32 = lane_sub; mynn_it32 < mynn_n32; mynn_it32 += lane_width) { k = mynn_lo32 + mynn_it32*mynn_st32;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5050
qni2(k) = x(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5052
if (((sqi2(k)) > (1e-12f))) { qni2(k) = mynn_max((float)(qni2(k)), (float)(ni_min)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5053
} mynn_lane_sync(); k = mynn_lo32 + mynn_n32*mynn_st32; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5055
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5056
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qni2).size(); mynn_element += lane_width) { qni2.linear(mynn_element) = qni.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5057
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5063
if (((((((bl_mynn_cloudmix) > (0))) && (flag_qnc))) && (((bl_mynn_mixnumcon) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5066
{ const int mynn_lo33 = (kts), mynn_hi33 = (kte), mynn_st33 = (1); const int mynn_n33 = mynn_trip(mynn_lo33, mynn_hi33, mynn_st33); mynn_lane_sync(); for (int mynn_it33 = lane_sub; mynn_it33 < mynn_n33; mynn_it33 += lane_width) { k = mynn_lo33 + mynn_it33*mynn_st33;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5067
qnc2(k) = mynn_max((float)(zero), (float)(qnc(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5069
if (((sqc(k)) > (1e-12f))) { qnc2(k) = mynn_max((float)(qnc2(k)), (float)(nc_min)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5070
} mynn_lane_sync(); k = mynn_lo33 + mynn_n33*mynn_st33; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5072
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5074
{ const int mynn_lo34 = (((kts) + (1))), mynn_hi34 = (((kte) - (1))), mynn_st34 = (1); const int mynn_n34 = mynn_trip(mynn_lo34, mynn_hi34, mynn_st34); mynn_lane_sync(); for (int mynn_it34 = lane_sub; mynn_it34 < mynn_n34; mynn_it34 += lane_width) { k = mynn_lo34 + mynn_it34*mynn_st34;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5075
upcont(k) = r6_sub(s_awqnc1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnc2(k), upwind), r6_mul(qnc2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5076
dncont(k) = r6_sub(sd_awqnc1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qnc2(k), upwind), r6_mul(qnc2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5077
} mynn_lane_sync(); k = mynn_lo34 + mynn_n34*mynn_st34; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5079
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5080
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5083
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5084
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5085
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5086
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5087
d(1) = r6_sub(qnc2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5090
{ const int mynn_lo35 = (((kts) + (1))), mynn_hi35 = (((kte) - (1))), mynn_st35 = (1); const int mynn_n35 = mynn_trip(mynn_lo35, mynn_hi35, mynn_st35); mynn_lane_sync(); for (int mynn_it35 = lane_sub; mynn_it35 < mynn_n35; mynn_it35 += lane_width) { k = mynn_lo35 + mynn_it35*mynn_st35;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5091
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5092
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5093
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5094
d(k) = r6_sub(qnc2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5095
} mynn_lane_sync(); k = mynn_lo35 + mynn_n35*mynn_st35; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5097
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5099
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5100
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5101
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5104
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5107
d(k) = r6_add(r6_sub(qnc2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnc1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqnc1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5111
{ const int mynn_lo36 = (((kts) + (1))), mynn_hi36 = (((kte) - (1))), mynn_st36 = (1); const int mynn_n36 = mynn_trip(mynn_lo36, mynn_hi36, mynn_st36); mynn_lane_sync(); for (int mynn_it36 = lane_sub; mynn_it36 < mynn_n36; mynn_it36 += lane_width) { k = mynn_lo36 + mynn_it36*mynn_st36;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5112
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5115
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5118
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5121
d(k) = r6_sub(r6_add(qnc2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnc1(k), s_awqnc1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqnc1(k), sd_awqnc1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5124
} mynn_lane_sync(); k = mynn_lo36 + mynn_n36*mynn_st36; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5125
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5128
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5129
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5130
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5131
d(kte) = qnc2(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5134
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5137
{ const int mynn_lo37 = (kts), mynn_hi37 = (kte), mynn_st37 = (1); const int mynn_n37 = mynn_trip(mynn_lo37, mynn_hi37, mynn_st37); mynn_lane_sync(); for (int mynn_it37 = lane_sub; mynn_it37 < mynn_n37; mynn_it37 += lane_width) { k = mynn_lo37 + mynn_it37*mynn_st37;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5139
qnc2(k) = x(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5141
if (((sqc2(k)) > (1e-12f))) { qnc2(k) = mynn_max((float)(qnc2(k)), (float)(nc_min)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5142
} mynn_lane_sync(); k = mynn_lo37 + mynn_n37*mynn_st37; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5144
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5145
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnc2).size(); mynn_element += lane_width) { qnc2.linear(mynn_element) = qnc.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5146
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5151
if (((flag_qnwfa) && (((bl_mynn_mixaerosols) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5153
{ const int mynn_lo38 = (kts), mynn_hi38 = (kte), mynn_st38 = (1); const int mynn_n38 = mynn_trip(mynn_lo38, mynn_hi38, mynn_st38); mynn_lane_sync(); for (int mynn_it38 = lane_sub; mynn_it38 < mynn_n38; mynn_it38 += lane_width) { k = mynn_lo38 + mynn_it38*mynn_st38;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5154
qnwfa2(k) = mynn_max((float)(zero), (float)(qnwfa(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5155
} mynn_lane_sync(); k = mynn_lo38 + mynn_n38*mynn_st38; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5157
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5159
{ const int mynn_lo39 = (((kts) + (1))), mynn_hi39 = (((kte) - (1))), mynn_st39 = (1); const int mynn_n39 = mynn_trip(mynn_lo39, mynn_hi39, mynn_st39); mynn_lane_sync(); for (int mynn_it39 = lane_sub; mynn_it39 < mynn_n39; mynn_it39 += lane_width) { k = mynn_lo39 + mynn_it39*mynn_st39;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5160
upcont(k) = r6_sub(s_awqnwfa1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnwfa2(k), upwind), r6_mul(qnwfa2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5161
dncont(k) = r6_sub(sd_awqnwfa1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qnwfa2(k), upwind), r6_mul(qnwfa2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5162
} mynn_lane_sync(); k = mynn_lo39 + mynn_n39*mynn_st39; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5164
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5165
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5168
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5169
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5170
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5171
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5172
d(1) = r6_sub(qnwfa2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5175
{ const int mynn_lo40 = (((kts) + (1))), mynn_hi40 = (((kte) - (1))), mynn_st40 = (1); const int mynn_n40 = mynn_trip(mynn_lo40, mynn_hi40, mynn_st40); mynn_lane_sync(); for (int mynn_it40 = lane_sub; mynn_it40 < mynn_n40; mynn_it40 += lane_width) { k = mynn_lo40 + mynn_it40*mynn_st40;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5176
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5177
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5178
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5179
d(k) = r6_sub(qnwfa2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5180
} mynn_lane_sync(); k = mynn_lo40 + mynn_n40*mynn_st40; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5182
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5184
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5185
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5186
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5189
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5192
d(k) = r6_add(r6_sub(qnwfa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnwfa1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqnwfa1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5196
{ const int mynn_lo41 = (((kts) + (1))), mynn_hi41 = (((kte) - (1))), mynn_st41 = (1); const int mynn_n41 = mynn_trip(mynn_lo41, mynn_hi41, mynn_st41); mynn_lane_sync(); for (int mynn_it41 = lane_sub; mynn_it41 < mynn_n41; mynn_it41 += lane_width) { k = mynn_lo41 + mynn_it41*mynn_st41;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5197
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5200
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5203
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5206
d(k) = r6_sub(r6_add(qnwfa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnwfa1(k), s_awqnwfa1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqnwfa1(k), sd_awqnwfa1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5209
} mynn_lane_sync(); k = mynn_lo41 + mynn_n41*mynn_st41; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5210
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5213
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5214
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5215
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5216
d(kte) = qnwfa2(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5219
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5222
{ const int mynn_lo42 = (kts), mynn_hi42 = (kte), mynn_st42 = (1); const int mynn_n42 = mynn_trip(mynn_lo42, mynn_hi42, mynn_st42); mynn_lane_sync(); for (int mynn_it42 = lane_sub; mynn_it42 < mynn_n42; mynn_it42 += lane_width) { k = mynn_lo42 + mynn_it42*mynn_st42;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5224
qnwfa2(k) = mynn_max((float)(zero), (float)(x(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5225
} mynn_lane_sync(); k = mynn_lo42 + mynn_n42*mynn_st42; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5228
{ const int mynn_lo43 = (kts), mynn_hi43 = (kte), mynn_st43 = (1); const int mynn_n43 = mynn_trip(mynn_lo43, mynn_hi43, mynn_st43); mynn_lane_sync(); for (int mynn_it43 = lane_sub; mynn_it43 < mynn_n43; mynn_it43 += lane_width) { k = mynn_lo43 + mynn_it43*mynn_st43;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5229
aero_min = r6_mul(wfa_min, mynn_exp(r6_neg(r6_div(zw(k), wfa_ht))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5230
aero_max = r6_mul(wfa_max, mynn_exp(r6_neg(r6_div(zw(k), wfa_ht))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5231
qnwfa2(k) = mynn_min((float)(mynn_max((float)(aero_min), (float)(qnwfa2(k)))), (float)(aero_max));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5232
} mynn_lane_sync(); k = mynn_lo43 + mynn_n43*mynn_st43; if (mynn_n43 > 0) { const int mynn_src43 = (mynn_n43 - 1) % lane_width; aero_max = mynn_lane_bcast(aero_max, mynn_src43); aero_min = mynn_lane_bcast(aero_min, mynn_src43); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5233
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5235
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnwfa2).size(); mynn_element += lane_width) { qnwfa2.linear(mynn_element) = qnwfa.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5236
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5241
if (((flag_qnifa) && (((bl_mynn_mixaerosols) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5243
{ const int mynn_lo44 = (kts), mynn_hi44 = (kte), mynn_st44 = (1); const int mynn_n44 = mynn_trip(mynn_lo44, mynn_hi44, mynn_st44); mynn_lane_sync(); for (int mynn_it44 = lane_sub; mynn_it44 < mynn_n44; mynn_it44 += lane_width) { k = mynn_lo44 + mynn_it44*mynn_st44;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5244
qnifa2(k) = mynn_max((float)(zero), (float)(qnifa(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5245
} mynn_lane_sync(); k = mynn_lo44 + mynn_n44*mynn_st44; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5247
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5249
{ const int mynn_lo45 = (((kts) + (1))), mynn_hi45 = (((kte) - (1))), mynn_st45 = (1); const int mynn_n45 = mynn_trip(mynn_lo45, mynn_hi45, mynn_st45); mynn_lane_sync(); for (int mynn_it45 = lane_sub; mynn_it45 < mynn_n45; mynn_it45 += lane_width) { k = mynn_lo45 + mynn_it45*mynn_st45;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5250
upcont(k) = r6_sub(s_awqnifa1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnifa2(k), upwind), r6_mul(qnifa2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5251
dncont(k) = r6_sub(sd_awqnifa1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qnifa2(k), upwind), r6_mul(qnifa2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5252
} mynn_lane_sync(); k = mynn_lo45 + mynn_n45*mynn_st45; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5254
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5255
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5257
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5258
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5259
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5260
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5261
d(1) = r6_sub(qnifa2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5264
{ const int mynn_lo46 = (((kts) + (1))), mynn_hi46 = (((kte) - (1))), mynn_st46 = (1); const int mynn_n46 = mynn_trip(mynn_lo46, mynn_hi46, mynn_st46); mynn_lane_sync(); for (int mynn_it46 = lane_sub; mynn_it46 < mynn_n46; mynn_it46 += lane_width) { k = mynn_lo46 + mynn_it46*mynn_st46;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5265
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5266
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5267
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5268
d(k) = r6_sub(qnifa2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5269
} mynn_lane_sync(); k = mynn_lo46 + mynn_n46*mynn_st46; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5271
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5273
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5274
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5275
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5278
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5281
d(k) = r6_add(r6_sub(qnifa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnifa1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqnifa1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5285
{ const int mynn_lo47 = (((kts) + (1))), mynn_hi47 = (((kte) - (1))), mynn_st47 = (1); const int mynn_n47 = mynn_trip(mynn_lo47, mynn_hi47, mynn_st47); mynn_lane_sync(); for (int mynn_it47 = lane_sub; mynn_it47 < mynn_n47; mynn_it47 += lane_width) { k = mynn_lo47 + mynn_it47*mynn_st47;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5286
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5289
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5292
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5295
d(k) = r6_sub(r6_add(qnifa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnifa1(k), s_awqnifa1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqnifa1(k), sd_awqnifa1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5298
} mynn_lane_sync(); k = mynn_lo47 + mynn_n47*mynn_st47; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5299
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5302
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5303
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5304
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5305
d(kte) = qnifa2(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5308
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5311
{ const int mynn_lo48 = (kts), mynn_hi48 = (kte), mynn_st48 = (1); const int mynn_n48 = mynn_trip(mynn_lo48, mynn_hi48, mynn_st48); mynn_lane_sync(); for (int mynn_it48 = lane_sub; mynn_it48 < mynn_n48; mynn_it48 += lane_width) { k = mynn_lo48 + mynn_it48*mynn_st48;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5313
qnifa2(k) = mynn_max((float)(zero), (float)(x(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5314
} mynn_lane_sync(); k = mynn_lo48 + mynn_n48*mynn_st48; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5317
{ const int mynn_lo49 = (kts), mynn_hi49 = (kte), mynn_st49 = (1); const int mynn_n49 = mynn_trip(mynn_lo49, mynn_hi49, mynn_st49); mynn_lane_sync(); for (int mynn_it49 = lane_sub; mynn_it49 < mynn_n49; mynn_it49 += lane_width) { k = mynn_lo49 + mynn_it49*mynn_st49;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5318
aero_min = r6_mul(ifa_min, mynn_exp(r6_neg(r6_div(zw(k), ifa_ht))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5319
aero_max = r6_mul(ifa_max, mynn_exp(r6_neg(r6_div(zw(k), ifa_ht))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5320
qnifa2(k) = mynn_min((float)(mynn_max((float)(aero_min), (float)(qnifa2(k)))), (float)(aero_max));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5321
} mynn_lane_sync(); k = mynn_lo49 + mynn_n49*mynn_st49; if (mynn_n49 > 0) { const int mynn_src49 = (mynn_n49 - 1) % lane_width; aero_max = mynn_lane_bcast(aero_max, mynn_src49); aero_min = mynn_lane_bcast(aero_min, mynn_src49); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5323
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5325
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnifa2).size(); mynn_element += lane_width) { qnifa2.linear(mynn_element) = qnifa.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5326
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5331
if (((flag_qnbca) && (((bl_mynn_mixaerosols) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5333
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5335
{ const int mynn_lo50 = (((kts) + (1))), mynn_hi50 = (((kte) - (1))), mynn_st50 = (1); const int mynn_n50 = mynn_trip(mynn_lo50, mynn_hi50, mynn_st50); mynn_lane_sync(); for (int mynn_it50 = lane_sub; mynn_it50 < mynn_n50; mynn_it50 += lane_width) { k = mynn_lo50 + mynn_it50*mynn_st50;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5336
upcont(k) = r6_sub(s_awqnbca1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnbca(k), upwind), r6_mul(qnbca(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5337
dncont(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5338
} mynn_lane_sync(); k = mynn_lo50 + mynn_n50*mynn_st50; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5340
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5341
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5343
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5344
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5345
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5346
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5347
d(1) = r6_sub(qnbca(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5350
{ const int mynn_lo51 = (((kts) + (1))), mynn_hi51 = (((kte) - (1))), mynn_st51 = (1); const int mynn_n51 = mynn_trip(mynn_lo51, mynn_hi51, mynn_st51); mynn_lane_sync(); for (int mynn_it51 = lane_sub; mynn_it51 < mynn_n51; mynn_it51 += lane_width) { k = mynn_lo51 + mynn_it51*mynn_st51;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5351
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5352
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5353
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5354
d(k) = r6_sub(qnbca(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5355
} mynn_lane_sync(); k = mynn_lo51 + mynn_n51*mynn_st51; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5357
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5359
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5360
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5361
b(k) = r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5363
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5365
d(k) = r6_sub(qnbca(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnbca1(((k) + (1)))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5367
{ const int mynn_lo52 = (((kts) + (1))), mynn_hi52 = (((kte) - (1))), mynn_st52 = (1); const int mynn_n52 = mynn_trip(mynn_lo52, mynn_hi52, mynn_st52); mynn_lane_sync(); for (int mynn_it52 = lane_sub; mynn_it52 < mynn_n52; mynn_it52 += lane_width) { k = mynn_lo52 + mynn_it52*mynn_st52;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5368
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5370
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5372
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5374
d(k) = r6_add(qnbca(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnbca1(k), s_awqnbca1(((k) + (1))))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5375
} mynn_lane_sync(); k = mynn_lo52 + mynn_n52*mynn_st52; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5376
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5379
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5380
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5381
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5382
d(kte) = qnbca(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5385
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5388
{ const int mynn_lo53 = (kts), mynn_hi53 = (kte), mynn_st53 = (1); const int mynn_n53 = mynn_trip(mynn_lo53, mynn_hi53, mynn_st53); mynn_lane_sync(); for (int mynn_it53 = lane_sub; mynn_it53 < mynn_n53; mynn_it53 += lane_width) { k = mynn_lo53 + mynn_it53*mynn_st53;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5390
qnbca2(k) = mynn_max((float)(zero), (float)(x(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5391
} mynn_lane_sync(); k = mynn_lo53 + mynn_n53*mynn_st53; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5393
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5395
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnbca2).size(); mynn_element += lane_width) { qnbca2.linear(mynn_element) = qnbca.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5396
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5401
if (flag_ozone) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5402
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5405
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5406
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5407
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5408
d(k) = ozone(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5410
{ const int mynn_lo54 = (((kts) + (1))), mynn_hi54 = (((kte) - (1))), mynn_st54 = (1); const int mynn_n54 = mynn_trip(mynn_lo54, mynn_hi54, mynn_st54); mynn_lane_sync(); for (int mynn_it54 = lane_sub; mynn_it54 < mynn_n54; mynn_it54 += lane_width) { k = mynn_lo54 + mynn_it54*mynn_st54;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5411
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5412
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5413
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5414
d(k) = ozone(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5415
} mynn_lane_sync(); k = mynn_lo54 + mynn_n54*mynn_st54; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5418
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5419
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5420
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5421
d(kte) = ozone(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5424
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5427
{ const int mynn_lo55 = (kts), mynn_hi55 = (kte), mynn_st55 = (1); const int mynn_n55 = mynn_trip(mynn_lo55, mynn_hi55, mynn_st55); mynn_lane_sync(); for (int mynn_it55 = lane_sub; mynn_it55 < mynn_n55; mynn_it55 += lane_width) { k = mynn_lo55 + mynn_it55*mynn_st55;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5429
dozone(k) = r6_div(r6_sub(mynn_max((float)(zero), (float)(x(k))), ozone(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5430
} mynn_lane_sync(); k = mynn_lo55 + mynn_n55*mynn_st55; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5431
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5432
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dozone.slice(MynnRange{dozone.lo[0], dozone.hi[0]})).size(); mynn_element += lane_width) { dozone.slice(MynnRange{dozone.lo[0], dozone.hi[0]}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5433
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5440
if (((bl_mynn_mixqt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5441
{ const int mynn_lo56 = (kts), mynn_hi56 = (kte), mynn_st56 = (1); const int mynn_n56 = mynn_trip(mynn_lo56, mynn_hi56, mynn_st56); mynn_lane_sync(); for (int mynn_it56 = lane_sub; mynn_it56 < mynn_n56; mynn_it56 += lane_width) { k = mynn_lo56 + mynn_it56*mynn_st56;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5443
th_new = r6_add(r6_add(thl(k), r6_mul(r6_div(xlvcp, exner(k)), sqc(k))), r6_mul(r6_div(xlscp, exner(k)), sqi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5446
t = r6_mul(th_new, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5447
qsat = f_qsat_blend(t, p(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5454
if (((((sqc(k)) > (zero))) || (((sqi(k)) > (zero))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5455
sqv2(k) = mynn_min((float)(sqw2(k)), (float)(qsat));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5456
portion_qc = r6_div(sqc(k), r6_add(sqc(k), sqi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5457
portion_qi = r6_div(sqi(k), r6_add(sqc(k), sqi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5458
condensate = mynn_max((float)(zero), (float)(r6_sub(sqw2(k), qsat)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5459
sqc2(k) = r6_mul(condensate, portion_qc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5460
sqi2(k) = r6_mul(condensate, portion_qi);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5461
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5462
sqv2(k) = sqw2(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5463
sqi2(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5464
sqc2(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5465
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5466
} mynn_lane_sync(); k = mynn_lo56 + mynn_n56*mynn_st56; if (mynn_n56 > 0) { const int mynn_src56 = (mynn_n56 - 1) % lane_width; qsat = mynn_lane_bcast(qsat, mynn_src56); t = mynn_lane_bcast(t, mynn_src56); th_new = mynn_lane_bcast(th_new, mynn_src56); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5467
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5473
{ const int mynn_lo57 = (kts), mynn_hi57 = (kte), mynn_st57 = (1); const int mynn_n57 = mynn_trip(mynn_lo57, mynn_hi57, mynn_st57); mynn_lane_sync(); for (int mynn_it57 = lane_sub; mynn_it57 < mynn_n57; mynn_it57 += lane_width) { k = mynn_lo57 + mynn_it57*mynn_st57;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5474
dqv(k) = r6_div(r6_sub(sqv2(k), sqv(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5476
} mynn_lane_sync(); k = mynn_lo57 + mynn_n57*mynn_st57; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5478
if (((bl_mynn_cloudmix) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5483
if (flag_qc) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5484
{ const int mynn_lo58 = (kts), mynn_hi58 = (kte), mynn_st58 = (1); const int mynn_n58 = mynn_trip(mynn_lo58, mynn_hi58, mynn_st58); mynn_lane_sync(); for (int mynn_it58 = lane_sub; mynn_it58 < mynn_n58; mynn_it58 += lane_width) { k = mynn_lo58 + mynn_it58*mynn_st58;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5485
dqc(k) = r6_div(r6_sub(sqc2(k), sqc(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5487
} mynn_lane_sync(); k = mynn_lo58 + mynn_n58*mynn_st58; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5488
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5489
{ const int mynn_lo59 = (kts), mynn_hi59 = (kte), mynn_st59 = (1); const int mynn_n59 = mynn_trip(mynn_lo59, mynn_hi59, mynn_st59); mynn_lane_sync(); for (int mynn_it59 = lane_sub; mynn_it59 < mynn_n59; mynn_it59 += lane_width) { k = mynn_lo59 + mynn_it59*mynn_st59;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5490
dqc(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5491
} mynn_lane_sync(); k = mynn_lo59 + mynn_n59*mynn_st59; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5492
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5497
if (((flag_qnc) && (((bl_mynn_mixnumcon) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5498
{ const int mynn_lo60 = (kts), mynn_hi60 = (kte), mynn_st60 = (1); const int mynn_n60 = mynn_trip(mynn_lo60, mynn_hi60, mynn_st60); mynn_lane_sync(); for (int mynn_it60 = lane_sub; mynn_it60 < mynn_n60; mynn_it60 += lane_width) { k = mynn_lo60 + mynn_it60*mynn_st60;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5499
dqnc(k) = r6_div(r6_sub(qnc2(k), qnc(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5501
} mynn_lane_sync(); k = mynn_lo60 + mynn_n60*mynn_st60; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5502
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5503
{ const int mynn_lo61 = (kts), mynn_hi61 = (kte), mynn_st61 = (1); const int mynn_n61 = mynn_trip(mynn_lo61, mynn_hi61, mynn_st61); mynn_lane_sync(); for (int mynn_it61 = lane_sub; mynn_it61 < mynn_n61; mynn_it61 += lane_width) { k = mynn_lo61 + mynn_it61*mynn_st61;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5504
dqnc(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5505
} mynn_lane_sync(); k = mynn_lo61 + mynn_n61*mynn_st61; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5506
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5511
if (flag_qi) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5512
{ const int mynn_lo62 = (kts), mynn_hi62 = (kte), mynn_st62 = (1); const int mynn_n62 = mynn_trip(mynn_lo62, mynn_hi62, mynn_st62); mynn_lane_sync(); for (int mynn_it62 = lane_sub; mynn_it62 < mynn_n62; mynn_it62 += lane_width) { k = mynn_lo62 + mynn_it62*mynn_st62;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5513
dqi(k) = r6_div(r6_sub(sqi2(k), sqi(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5515
} mynn_lane_sync(); k = mynn_lo62 + mynn_n62*mynn_st62; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5516
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5517
{ const int mynn_lo63 = (kts), mynn_hi63 = (kte), mynn_st63 = (1); const int mynn_n63 = mynn_trip(mynn_lo63, mynn_hi63, mynn_st63); mynn_lane_sync(); for (int mynn_it63 = lane_sub; mynn_it63 < mynn_n63; mynn_it63 += lane_width) { k = mynn_lo63 + mynn_it63*mynn_st63;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5518
dqi(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5519
} mynn_lane_sync(); k = mynn_lo63 + mynn_n63*mynn_st63; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5520
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5525
if (false) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5526
{ const int mynn_lo64 = (kts), mynn_hi64 = (kte), mynn_st64 = (1); const int mynn_n64 = mynn_trip(mynn_lo64, mynn_hi64, mynn_st64); mynn_lane_sync(); for (int mynn_it64 = lane_sub; mynn_it64 < mynn_n64; mynn_it64 += lane_width) { k = mynn_lo64 + mynn_it64*mynn_st64;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5527
dqs(k) = r6_div(r6_sub(sqs2(k), sqs(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5528
} mynn_lane_sync(); k = mynn_lo64 + mynn_n64*mynn_st64; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5529
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5530
{ const int mynn_lo65 = (kts), mynn_hi65 = (kte), mynn_st65 = (1); const int mynn_n65 = mynn_trip(mynn_lo65, mynn_hi65, mynn_st65); mynn_lane_sync(); for (int mynn_it65 = lane_sub; mynn_it65 < mynn_n65; mynn_it65 += lane_width) { k = mynn_lo65 + mynn_it65*mynn_st65;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5531
dqs(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5532
} mynn_lane_sync(); k = mynn_lo65 + mynn_n65*mynn_st65; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5533
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5538
if (((flag_qni) && (((bl_mynn_mixnumcon) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5539
{ const int mynn_lo66 = (kts), mynn_hi66 = (kte), mynn_st66 = (1); const int mynn_n66 = mynn_trip(mynn_lo66, mynn_hi66, mynn_st66); mynn_lane_sync(); for (int mynn_it66 = lane_sub; mynn_it66 < mynn_n66; mynn_it66 += lane_width) { k = mynn_lo66 + mynn_it66*mynn_st66;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5540
dqni(k) = r6_div(r6_sub(qni2(k), qni(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5542
} mynn_lane_sync(); k = mynn_lo66 + mynn_n66*mynn_st66; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5543
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5544
{ const int mynn_lo67 = (kts), mynn_hi67 = (kte), mynn_st67 = (1); const int mynn_n67 = mynn_trip(mynn_lo67, mynn_hi67, mynn_st67); mynn_lane_sync(); for (int mynn_it67 = lane_sub; mynn_it67 < mynn_n67; mynn_it67 += lane_width) { k = mynn_lo67 + mynn_it67*mynn_st67;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5545
dqni(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5546
} mynn_lane_sync(); k = mynn_lo67 + mynn_n67*mynn_st67; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5547
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5548
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5550
{ const int mynn_lo68 = (kts), mynn_hi68 = (kte), mynn_st68 = (1); const int mynn_n68 = mynn_trip(mynn_lo68, mynn_hi68, mynn_st68); mynn_lane_sync(); for (int mynn_it68 = lane_sub; mynn_it68 < mynn_n68; mynn_it68 += lane_width) { k = mynn_lo68 + mynn_it68*mynn_st68;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5551
dqc(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5552
dqnc(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5553
dqi(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5554
dqni(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5555
dqs(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5556
} mynn_lane_sync(); k = mynn_lo68 + mynn_n68*mynn_st68; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5557
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5560
f_moisture_check(kte, delt, delp, exner, sqv2, sqc2, sqi2, sqs2, thl, dqv, dqc, dqi, dqs, dth);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5567
{ const int mynn_lo69 = (kts), mynn_hi69 = (kte), mynn_st69 = (1); const int mynn_n69 = mynn_trip(mynn_lo69, mynn_hi69, mynn_st69); mynn_lane_sync(); for (int mynn_it69 = lane_sub; mynn_it69 < mynn_n69; mynn_it69 += lane_width) { k = mynn_lo69 + mynn_it69*mynn_st69;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5568
if (((r6_add(r6_mul(dozone(k), delt), ozone(k))) < (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5569
dozone(k) = r6_div(r6_neg(r6_mul(ozone(k), 0.99f)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5570
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5571
} mynn_lane_sync(); k = mynn_lo69 + mynn_n69*mynn_st69; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5576
if (flag_qi) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5577
{ const int mynn_lo70 = (kts), mynn_hi70 = (kte), mynn_st70 = (1); const int mynn_n70 = mynn_trip(mynn_lo70, mynn_hi70, mynn_st70); mynn_lane_sync(); for (int mynn_it70 = lane_sub; mynn_it70 < mynn_n70; mynn_it70 += lane_width) { k = mynn_lo70 + mynn_it70*mynn_st70;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5578
dth(k) = r6_div(r6_sub(r6_add(r6_add(thl(k), r6_mul(r6_div(xlvcp, exner(k)), sqc(k))), r6_mul(r6_div(xlscp, exner(k)), sqi(k))), th(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5589
} mynn_lane_sync(); k = mynn_lo70 + mynn_n70*mynn_st70; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5590
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5591
{ const int mynn_lo71 = (kts), mynn_hi71 = (kte), mynn_st71 = (1); const int mynn_n71 = mynn_trip(mynn_lo71, mynn_hi71, mynn_st71); mynn_lane_sync(); for (int mynn_it71 = lane_sub; mynn_it71 < mynn_n71; mynn_it71 += lane_width) { k = mynn_lo71 + mynn_it71*mynn_st71;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5592
dth(k) = r6_div(r6_sub(r6_add(thl(k), r6_mul(r6_div(xlvcp, exner(k)), sqc2(k))), th(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5597
} mynn_lane_sync(); k = mynn_lo71 + mynn_n71*mynn_st71; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5598
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5603
if (((((flag_qnwfa) && (flag_qnifa))) && (((bl_mynn_mixaerosols) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5605
{ const int mynn_lo72 = (kts), mynn_hi72 = (kte), mynn_st72 = (1); const int mynn_n72 = mynn_trip(mynn_lo72, mynn_hi72, mynn_st72); mynn_lane_sync(); for (int mynn_it72 = lane_sub; mynn_it72 < mynn_n72; mynn_it72 += lane_width) { k = mynn_lo72 + mynn_it72*mynn_st72;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5609
dqnwfa(k) = r6_div(r6_sub(qnwfa2(k), qnwfa(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5613
dqnifa(k) = r6_div(r6_sub(qnifa2(k), qnifa(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5614
} mynn_lane_sync(); k = mynn_lo72 + mynn_n72*mynn_st72; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5615
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5616
{ const int mynn_lo73 = (kts), mynn_hi73 = (kte), mynn_st73 = (1); const int mynn_n73 = mynn_trip(mynn_lo73, mynn_hi73, mynn_st73); mynn_lane_sync(); for (int mynn_it73 = lane_sub; mynn_it73 < mynn_n73; mynn_it73 += lane_width) { k = mynn_lo73 + mynn_it73*mynn_st73;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5617
dqnwfa(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5618
dqnifa(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5619
} mynn_lane_sync(); k = mynn_lo73 + mynn_n73*mynn_st73; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5620
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5625
if (((flag_qnbca) && (((bl_mynn_mixaerosols) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5626
{ const int mynn_lo74 = (kts), mynn_hi74 = (kte), mynn_st74 = (1); const int mynn_n74 = mynn_trip(mynn_lo74, mynn_hi74, mynn_st74); mynn_lane_sync(); for (int mynn_it74 = lane_sub; mynn_it74 < mynn_n74; mynn_it74 += lane_width) { k = mynn_lo74 + mynn_it74*mynn_st74;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5627
dqnbca(k) = r6_div(r6_sub(qnbca2(k), qnbca(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5628
} mynn_lane_sync(); k = mynn_lo74 + mynn_n74*mynn_st74; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5629
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5630
{ const int mynn_lo75 = (kts), mynn_hi75 = (kte), mynn_st75 = (1); const int mynn_n75 = mynn_trip(mynn_lo75, mynn_hi75, mynn_st75); mynn_lane_sync(); for (int mynn_it75 = lane_sub; mynn_it75 < mynn_n75; mynn_it75 += lane_width) { k = mynn_lo75 + mynn_it75*mynn_st75;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5631
dqnbca(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5632
} mynn_lane_sync(); k = mynn_lo75 + mynn_n75*mynn_st75; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5633
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5642
if (debug_code) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5643
problem = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5644
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5645
wsp = mynn_sqrt(r6_add(mynn_ipow<2>(u(k)), mynn_ipow<2>(v(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5646
wsp2 = mynn_sqrt(r6_add(mynn_ipow<2>(r6_add(u(k), r6_mul(du(k), delt))), mynn_ipow<2>(r6_add(v(k), r6_mul(du(k), delt)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5647
th2 = r6_add(th(k), r6_mul(dth(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5648
tk2 = r6_mul(th2, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5649
if (((((((wsp2) > (200.f))) || (((tk2) > (360.f))))) || (((tk2) < (160.f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5650
problem = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5651
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5652
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5653
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5654
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5655
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5656
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5657
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5658
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5659
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5660
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5661
kproblem = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5662
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5663
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5664
if (problem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5665
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5666
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5667
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5668
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5669
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5670
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5671
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5672
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5682-5770
__device__ __noinline__ void f_moisture_check(int kte, float delt, MynnArray<float, 1> dp, MynnArray<float, 1> exner, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> th, MynnArray<float, 1> dqv, MynnArray<float, 1> dqc, MynnArray<float, 1> dqi, MynnArray<float, 1> dqs, MynnArray<float, 1> dth) {
MynnFrame mynn_frame(arena);
dp.rebase({1});
exner.rebase({1});
qv.rebase({1});
qc.rebase({1});
qi.rebase({1});
qs.rebase({1});
th.rebase({1});
dqv.rebase({1});
dqc.rebase({1});
dqi.rebase({1});
dqs.rebase({1});
dth.rebase({1});
int k;
float dqc2;
float dqi2;
float dqs2;
float dqv2;
float sum;
float aa;
float dum;
constexpr float qvmin = 1e-20f;
constexpr float qcmin = zero;
constexpr float qimin = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5711
for (k=kte; ((-(1))) > 0 ? k <= (1) : k >= (1); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5712
dqc2 = mynn_max((float)(r6_sub(qcmin, qc(k))), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5713
dqi2 = mynn_max((float)(r6_sub(qimin, qi(k))), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5714
dqs2 = mynn_max((float)(r6_sub(qimin, qs(k))), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5717
dqc(k) = r6_add(dqc(k), r6_div(dqc2, delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5718
dqi(k) = r6_add(dqi(k), r6_div(dqi2, delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5719
dqs(k) = r6_add(dqs(k), r6_div(dqs2, delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5720
dqv(k) = r6_sub(dqv(k), r6_div(r6_add(r6_add(dqc2, dqi2), dqs2), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5721
dth(k) = r6_add(r6_add(dth(k), r6_mul(r6_div(xlvcp, exner(k)), r6_div(dqc2, delt))), r6_mul(r6_div(xlscp, exner(k)), r6_div(r6_add(dqi2, dqs2), delt)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5724
qc(k) = r6_add(qc(k), dqc2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5725
qi(k) = r6_add(qi(k), dqi2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5726
qs(k) = r6_add(qs(k), dqs2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5727
qv(k) = r6_sub(r6_sub(r6_sub(qv(k), dqc2), dqi2), dqs2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5728
th(k) = r6_add(r6_add(th(k), r6_mul(r6_div(xlvcp, exner(k)), dqc2)), r6_mul(r6_div(xlscp, exner(k)), r6_add(dqi2, dqs2)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5732
dqv2 = mynn_max((float)(r6_sub(qvmin, qv(k))), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5733
dqv(k) = r6_add(dqv(k), r6_div(dqv2, delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5734
qv(k) = r6_add(qv(k), dqv2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5735
if (((k) != (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5737
qv(((k) - (1))) = r6_sub(qv(((k) - (1))), r6_div(r6_mul(dqv2, dp(k)), dp(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5738
dqv(((k) - (1))) = r6_sub(dqv(((k) - (1))), r6_div(r6_div(r6_mul(dqv2, dp(k)), dp(((k) - (1)))), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5739
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5740
qv(k) = mynn_max((float)(qv(k)), (float)(qvmin));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5741
qc(k) = mynn_max((float)(qcmin), (float)(qc(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5742
qi(k) = mynn_max((float)(qimin), (float)(qi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5743
qs(k) = mynn_max((float)(qimin), (float)(qs(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5744
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5748
if (((dqv2) > (1.e-20f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5749
sum = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5750
for (k=1; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5751
if (((qv(k)) > (r6_mul(2.0f, qvmin)))) { sum = r6_add(sum, r6_mul(qv(k), dp(k))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5752
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5753
aa = r6_div(r6_mul(dqv2, dp(1)), mynn_max((float)(sum), (float)(1.e-20f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5754
if (((aa) < (p5))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5755
{ const int mynn_lo0 = (1), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5756
if (((qv(k)) > (r6_mul(2.0f, qvmin)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5757
dum = r6_mul(aa, qv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5758
qv(k) = r6_sub(qv(k), dum);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5759
dqv(k) = r6_sub(dqv(k), r6_div(dum, delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5760
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5761
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5762
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5765
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5766
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5768
return;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5774-5931
__device__ __noinline__ void f_mynn_mix_chem(int kts, int kte, int i, float delt, MynnArray<float, 1> dz, float pblh, int nchem, int ndvel, MynnArray<float, 2> chem1, MynnArray<float, 1> vd1, MynnArray<float, 2> settle1, MynnArray<float, 1> rho, float flt, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> dfh, MynnArray<float, 1> s_aw1, MynnArray<float, 2> s_awchem1, float emis_ant_no, float frp, bool enh_mix, int bl_mynn_edmf) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
chem1.rebase({kts, 1});
vd1.rebase({1});
settle1.rebase({kts, 1});
rho.rebase({kts});
tcd.rebase({kts});
qcd.rebase({kts});
dfh.rebase({kts});
s_aw1.rebase({kts});
s_awchem1.rebase({kts, 1});
MynnArray<float, 1> dtz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> upcont = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dncont = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> a = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> b = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> c = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> d = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> x = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
float rhs;
float dztop;
float t;
float dzk;
float hght;
float khdz_old;
float khdz_back;
int k;
int kk;
int kmaxfire;
int ic;
int icall;
MynnArray<float, 1> rhoinv = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rhoz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> khdz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
constexpr float no_threshold = 10.0f;
constexpr float frp_threshold = 10.0f;
constexpr float pblh_threshold = 100.0f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5818
dztop = r6_mul(.5f, r6_add(dz(kte), dz(((kte) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5820
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5821
dtz(k) = r6_div(delt, dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5822
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5826
rhoz(kts) = rho(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5827
rhoinv(kts) = r6_div(one, rho(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5828
khdz(kts) = r6_mul(rhoz(kts), dfh(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5830
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5831
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5832
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5833
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5834
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5835
khdz(k) = r6_mul(rhoz(k), dfh(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5836
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5837
rhoz(((kte) + (1))) = rhoz(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5838
khdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfh(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5840
if (((bl_mynn_edmf) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5842
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5843
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5844
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5845
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5846
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5849
if (enh_mix) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5850
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5851
khdz_old = khdz(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5852
khdz_back = r6_div(r6_mul(pblh, 0.15f), dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5854
if (((pblh) < (pblh_threshold))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5855
if (((emis_ant_no) > (no_threshold))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5856
khdz(k) = mynn_max((float)(r6_mul(1.1f, khdz(k))), (float)(r6_mul(r6_div(mynn_sqrt(r6_div(emis_ant_no, no_threshold)), dz(k)), rhoz(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5858
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5859
if (((frp) > (frp_threshold))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5860
kmaxfire = mynn_ceiling(mynn_log(frp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5861
khdz(k) = mynn_max((float)(r6_mul(1.1f, khdz(k))), (float)(r6_mul(r6_div(r6_mul(r6_sub(one, r6_div(k, r6_mul(kmaxfire, two))), r6_sub(mynn_ipow<2>(mynn_log(frp)), r6_mul(two, mynn_log(frp)))), dz(k)), rhoz(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5863
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5864
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5865
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; khdz_back = mynn_lane_bcast(khdz_back, mynn_src3); khdz_old = mynn_lane_bcast(khdz_old, mynn_src3); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5866
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5872
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5874
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5876
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5877
upcont(k) = r6_sub(s_awchem1(k, ic), r6_mul(s_aw1(k), r6_add(r6_mul(chem1(k, ic), upwind), r6_mul(chem1(((k) - (1)), ic), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5878
dncont(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5879
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5881
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5882
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5884
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5885
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5886
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5887
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5888
d(1) = r6_sub(chem1(k, ic), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5891
{ const int mynn_lo5 = (((kts) + (1))), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5892
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5893
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5894
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5895
d(k) = r6_sub(chem1(k, ic), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5896
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5898
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5900
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5901
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5902
b(k) = r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5903
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5904
d(k) = r6_sub(r6_sub(chem1(k, ic), r6_mul(r6_mul(dtz(k), vd1(ic)), chem1(k, ic))), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awchem1(((k) + (1)), ic)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5908
{ const int mynn_lo6 = (((kts) + (1))), mynn_hi6 = (((kte) - (1))), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5909
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5910
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5912
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5913
d(k) = r6_add(chem1(k, ic), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awchem1(k, ic), s_awchem1(((k) + (1)), ic))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5914
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5915
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5918
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5919
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5920
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5921
d(kte) = chem1(kte, ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5923
f_tridiag3(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5925
{ const int mynn_lo7 = (kts), mynn_hi7 = (kte), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5926
chem1(k, ic) = r6_add(x(k), settle1(k, ic));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5927
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5929
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5935-6055
__device__ __noinline__ void f_mynn_mix_scalars(int kts, int kte, int i, float delt, MynnArray<float, 1> dz, int nscalars, MynnArray<float, 2> scalars, MynnArray<float, 1> rho, float flt, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> dfh, MynnArray<float, 1> s_aw1, MynnArray<float, 2> s_awscalars1, int bl_mynn_edmf) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
scalars.rebase({kts, 1});
rho.rebase({kts});
tcd.rebase({kts});
qcd.rebase({kts});
dfh.rebase({kts});
s_aw1.rebase({kts});
s_awscalars1.rebase({kts, 1});
MynnArray<float, 1> dtz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> upcont = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dncont = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> a = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> b = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> c = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> d = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> x = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
float rhs;
float dztop;
float dzk;
int k;
int ns;
MynnArray<float, 1> rhoinv = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rhoz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> khdz = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5965
dztop = r6_mul(p5, r6_add(dz(kte), dz(((kte) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5967
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5968
dtz(k) = r6_div(delt, dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5969
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5973
rhoz(kts) = rho(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5974
rhoinv(kts) = r6_div(one, rho(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5975
khdz(kts) = r6_mul(rhoz(kts), dfh(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5977
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5978
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5979
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5980
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5981
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5982
khdz(k) = r6_mul(rhoz(k), dfh(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5983
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5984
rhoz(((kte) + (1))) = rhoz(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5985
khdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfh(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5987
if (((bl_mynn_edmf) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5989
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5990
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5991
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5992
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5993
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5999
for (ns=1; (1) > 0 ? ns <= (nscalars) : ns >= (nscalars); ns += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6001
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6003
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6004
upcont(k) = r6_sub(s_awscalars1(k, ns), r6_mul(s_aw1(k), r6_add(r6_mul(scalars(k, ns), upwind), r6_mul(scalars(((k) - (1)), ns), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6005
dncont(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6006
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6008
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6009
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6011
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6012
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6013
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6014
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6015
d(1) = r6_sub(scalars(k, ns), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6018
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6019
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6020
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6021
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6022
d(k) = r6_sub(scalars(k, ns), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6023
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6025
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6027
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6028
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6029
b(k) = r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6030
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6031
d(k) = r6_sub(scalars(k, ns), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awscalars1(((k) + (1)), ns)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6033
{ const int mynn_lo5 = (((kts) + (1))), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6034
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6035
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6037
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6038
d(k) = r6_add(scalars(k, ns), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awscalars1(k, ns), s_awscalars1(((k) + (1)), ns))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6039
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6040
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6043
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6044
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6045
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6046
d(kte) = scalars(kte, ns);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6048
f_tridiag3(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6050
{ const int mynn_lo6 = (kts), mynn_hi6 = (kte), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6051
scalars(k, ns) = mynn_max((float)(x(k)), (float)(1e-12f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6052
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6053
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6059-6084
__device__ __noinline__ void f_retrieve_exchange_coeffs(int kts, int kte, MynnArray<float, 1> dfm, MynnArray<float, 1> dfh, MynnArray<float, 1> dz, MynnArray<float, 1> km1, MynnArray<float, 1> kh1) {
MynnFrame mynn_frame(arena);
dfm.rebase({kts});
dfh.rebase({kts});
dz.rebase({kts});
km1.rebase({kts});
kh1.rebase({kts});
int k;
float dzk;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6072
km1(kts) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6073
kh1(kts) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6075
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6076
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6077
km1(k) = r6_mul(dfm(k), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6078
kh1(k) = r6_mul(dfh(k), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6079
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src0); } }
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6088-6120
__device__ __noinline__ void f_tridiag(int n, MynnArray<float, 1> a, MynnArray<float, 1> b, MynnArray<float, 1> c, MynnArray<float, 1> d) {
MynnFrame mynn_frame(arena);
a.rebase({1});
b.rebase({1});
c.rebase({1});
d.rebase({1});
int i;
float p;
MynnArray<float, 1> q = mynn_array<float, 1, mynn_tile>(arena, {1}, {n});
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6106
c(n) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6107
q(1) = r6_neg(r6_div(c(1), b(1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6108
d(1) = r6_div(d(1), b(1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6110
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6111
p = r6_div(1.f, r6_add(b(i), r6_mul(a(i), q(((i) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6112
q(i) = r6_neg(r6_mul(c(i), p));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6113
d(i) = r6_mul(r6_sub(d(i), r6_mul(a(i), d(((i) - (1))))), p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6114
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6116
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6117
d(i) = r6_add(d(i), r6_mul(q(i), d(((i) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6118
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6124-6156
__device__ __noinline__ void f_tridiag2(int kte, MynnArray<float, 1> a, MynnArray<float, 1> b, MynnArray<float, 1> c, MynnArray<float, 1> d, MynnArray<float, 1> x) {
MynnFrame mynn_frame(arena);
a.rebase({1});
b.rebase({1});
c.rebase({1});
d.rebase({1});
x.rebase({1});
MynnArray<float, 1> cp = mynn_array<float, 1, mynn_tile>(arena, {1}, {kte});
MynnArray<float, 1> dp = mynn_array<float, 1, mynn_tile>(arena, {1}, {kte});
float m;
int k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6141
cp(1) = r6_div(c(1), b(1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6142
dp(1) = r6_div(d(1), b(1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6144
for (k=2; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6145
m = r6_sub(b(k), r6_mul(cp(((k) - (1))), a(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6146
cp(k) = r6_div(c(k), m);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6147
dp(k) = r6_div(r6_sub(d(k), r6_mul(dp(((k) - (1))), a(k))), m);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6148
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6150
x(kte) = dp(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6152
for (k=((kte) - (1)); ((-(1))) > 0 ? k <= (1) : k >= (1); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6153
x(k) = r6_sub(dp(k), r6_mul(cp(k), x(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6154
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6159-6194
__device__ __noinline__ void f_tridiag3(int kte, MynnArray<float, 1> a, MynnArray<float, 1> b, MynnArray<float, 1> c, MynnArray<float, 1> d, MynnArray<float, 1> x) {
MynnFrame mynn_frame(arena);
a.rebase({1});
b.rebase({1});
c.rebase({1});
d.rebase({1});
x.rebase({1});
constexpr int kts = 1;
int k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6180
for (k=((kte) - (1)); ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6181
d(k) = r6_sub(d(k), r6_div(r6_mul(c(k), d(((k) + (1)))), b(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6182
b(k) = r6_sub(b(k), r6_div(r6_mul(c(k), a(((k) + (1)))), b(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6183
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6185
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6186
d(k) = r6_sub(d(k), r6_div(r6_mul(a(k), d(((k) - (1)))), b(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6187
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6189
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6190
x(k) = r6_div(d(k), b(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6191
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6193
return;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6223-6351
__device__ __noinline__ void f_get_pblh(int kts, int kte, float& pblh, MynnArray<float, 1> thv1, MynnArray<float, 1> qke1, float ust, MynnArray<float, 1> zw1, MynnArray<float, 1> dz1, float landsea, int& kpbl) {
MynnFrame mynn_frame(arena);
thv1.rebase({kts});
qke1.rebase({kts});
zw1.rebase({kts});
dz1.rebase({kts});
float pblh_stable;
float qtke;
float qtkem1;
float wt;
float maxqke;
float tkeeps;
float minthv;
float cpblh;
float delt_thv;
constexpr float sbl_lim = 200.f;
constexpr float sbl_damp = 170.f;
int i;
int j;
int k;
int kthv;
int ktke;
constexpr int stable_method = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6247
kpbl = 2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6250
k = ((kts) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6251
kthv = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6252
minthv = 9.e9f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6253
while (((zw1(k)) <= (200.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6254
if (((minthv) > (thv1(k)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6255
minthv = thv1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6256
kthv = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6257
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6258
k = ((k) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6259
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6262
pblh = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6263
k = ((kthv) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6264
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6266
delt_thv = 0.75f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6267
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6269
delt_thv = 1.25f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6270
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6272
pblh = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6273
k = ((kthv) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6274
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6275
if (((thv1(k)) >= (r6_add(minthv, delt_thv)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6276
pblh = r6_sub(zw1(k), r6_mul(dz1(((k) - (1))), mynn_min((float)(r6_div(r6_sub(thv1(k), r6_add(minthv, delt_thv)), mynn_max((float)(r6_sub(thv1(k), thv1(((k) - (1))))), (float)(1e-6f)))), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6279
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6280
if (((k) == (((kte) - (1))))) { pblh = zw1(((kts) + (1))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6281
if (((pblh) != (zero))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6282
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6285
if (((stable_method) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6290
ktke = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6291
maxqke = mynn_max((float)(zero), (float)(qke1(kts)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6294
tkeeps = r6_div(maxqke, 40.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6295
tkeeps = mynn_max((float)(tkeeps), (float)(0.01f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6296
pblh_stable = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6298
k = ((ktke) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6299
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6301
qtke = mynn_max((float)(zero), (float)(r6_mul(p5, qke1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6302
qtkem1 = mynn_max((float)(zero), (float)(r6_mul(p5, qke1(((k) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6303
if (((qtke) <= (tkeeps))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6304
pblh_stable = r6_sub(zw1(k), r6_mul(dz1(((k) - (1))), mynn_min((float)(r6_div(r6_sub(tkeeps, qtke), mynn_max((float)(r6_sub(qtkem1, qtke)), (float)(1e-6f)))), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6307
pblh_stable = mynn_max((float)(pblh_stable), (float)(zw1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6309
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6311
if (((k) == (((kte) - (1))))) { pblh_stable = zw1(((kts) + (1))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6312
if (((pblh_stable) != (zero))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6313
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6319
pblh_stable = mynn_min((float)(pblh_stable), (float)(r6_add(pblh, 350.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6320
pblh_stable = mynn_max((float)(pblh_stable), (float)(mynn_max((float)(r6_sub(pblh, 350.f)), (float)(ten))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6322
if (((maxqke) <= (tkeeps))) { pblh_stable = pblh; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6323
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6325
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(pblh, hundred), hundred))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6326
cpblh = r6_add(r6_mul(400.f, r6_sub(one, wt)), r6_mul(700.f, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6327
pblh_stable = mynn_max((float)(r6_mul(ust, cpblh)), (float)(ten));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6330
pblh_stable = mynn_max((float)(pblh_stable), (float)(r6_sub(pblh, sbl_lim)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6331
pblh_stable = mynn_min((float)(pblh_stable), (float)(r6_add(pblh, sbl_lim)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6332
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6335
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(pblh, sbl_lim), sbl_damp))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6336
pblh = r6_add(r6_mul(pblh_stable, r6_sub(one, wt)), r6_mul(pblh, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6339
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6340
if (((zw1(k)) >= (pblh))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6341
kpbl = ((k) - (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6342
break;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6343
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6344
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6373-7694
__device__ __noinline__ void f_dmp_mf(int i, int j, int kts, int kte, float dt, MynnArray<float, 1> zw1, MynnArray<float, 1> dz1, MynnArray<float, 1> pres1, MynnArray<float, 1> rho1, int momentum_opt, int tke_opt, int scalar_opt, int aerosol_opt, int numcon_opt, MynnArray<float, 1> u1, MynnArray<float, 1> v1, MynnArray<float, 1> w1, MynnArray<float, 1> th1, MynnArray<float, 1> thl1, MynnArray<float, 1> thv1, MynnArray<float, 1> tk1, MynnArray<float, 1> qt1, MynnArray<float, 1> qv1, MynnArray<float, 1> qc1, MynnArray<float, 1> qke1, MynnArray<float, 1> qnc1, MynnArray<float, 1> qni1, MynnArray<float, 1> qnwfa1, MynnArray<float, 1> qnifa1, MynnArray<float, 1> qnbca1, MynnArray<float, 1> ex1, MynnArray<float, 1> vt1, MynnArray<float, 1> vq1, MynnArray<float, 1> sgm1, float ust, float flt, float fltv, float flq, float flqv, float pblh, int kpbl, float dx, float landsea, float ts, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_qt1, MynnArray<float, 1> edmf_thl1, MynnArray<float, 1> edmf_ent1, MynnArray<float, 1> edmf_qc1, MynnArray<float, 1> edmf_qv1, MynnArray<float, 1> edmf_u1, MynnArray<float, 1> edmf_v1, MynnArray<float, 1> s_aw1, MynnArray<float, 1> s_awthl1, MynnArray<float, 1> s_awqt1, MynnArray<float, 1> s_awqv1, MynnArray<float, 1> s_awqc1, MynnArray<float, 1> s_awu1, MynnArray<float, 1> s_awv1, MynnArray<float, 1> s_awqke1, MynnArray<float, 1> s_awqnc1, MynnArray<float, 1> s_awqni1, MynnArray<float, 1> s_awqnwfa1, MynnArray<float, 1> s_awqnifa1, MynnArray<float, 1> s_awqnbca1, MynnArray<float, 1> sub_thl1, MynnArray<float, 1> sub_sqv1, MynnArray<float, 1> sub_u1, MynnArray<float, 1> sub_v1, MynnArray<float, 1> det_thl1, MynnArray<float, 1> det_sqv1, MynnArray<float, 1> det_sqc1, MynnArray<float, 1> det_u1, MynnArray<float, 1> det_v1, int nchem, MynnArray<float, 2> chem1, MynnArray<float, 2> s_awchem1, bool mix_chem, int nscalars, MynnArray<float, 2> scalars, MynnArray<float, 2> s_awscalars1, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> cldfra_bl1, MynnArray<float, 1> qc_bl1_old, MynnArray<float, 1> cldfra_bl1_old, MynnOptional<bool> f_qc, MynnOptional<bool> f_qi, MynnOptional<bool> f_qnc, MynnOptional<bool> f_qni, MynnOptional<bool> f_qnwfa, MynnOptional<bool> f_qnifa, MynnOptional<bool> f_qnbca, float psig_shcu, float& maxwidth, int& ktop, float& maxmf, float& ztop, float& excess_h, float& excess_q, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1, MynnArray<float, 1> tkeprod_up, MynnArray<float, 1> el1) {
MynnFrame mynn_frame(arena);
zw1.rebase({kts});
dz1.rebase({kts});
pres1.rebase({kts});
rho1.rebase({kts});
u1.rebase({kts});
v1.rebase({kts});
w1.rebase({kts});
th1.rebase({kts});
thl1.rebase({kts});
thv1.rebase({kts});
tk1.rebase({kts});
qt1.rebase({kts});
qv1.rebase({kts});
qc1.rebase({kts});
qke1.rebase({kts});
qnc1.rebase({kts});
qni1.rebase({kts});
qnwfa1.rebase({kts});
qnifa1.rebase({kts});
qnbca1.rebase({kts});
ex1.rebase({kts});
vt1.rebase({kts});
vq1.rebase({kts});
sgm1.rebase({kts});
edmf_a1.rebase({kts});
edmf_w1.rebase({kts});
edmf_qt1.rebase({kts});
edmf_thl1.rebase({kts});
edmf_ent1.rebase({kts});
edmf_qc1.rebase({kts});
edmf_qv1.rebase({kts});
edmf_u1.rebase({kts});
edmf_v1.rebase({kts});
s_aw1.rebase({kts});
s_awthl1.rebase({kts});
s_awqt1.rebase({kts});
s_awqv1.rebase({kts});
s_awqc1.rebase({kts});
s_awu1.rebase({kts});
s_awv1.rebase({kts});
s_awqke1.rebase({kts});
s_awqnc1.rebase({kts});
s_awqni1.rebase({kts});
s_awqnwfa1.rebase({kts});
s_awqnifa1.rebase({kts});
s_awqnbca1.rebase({kts});
sub_thl1.rebase({kts});
sub_sqv1.rebase({kts});
sub_u1.rebase({kts});
sub_v1.rebase({kts});
det_thl1.rebase({kts});
det_sqv1.rebase({kts});
det_sqc1.rebase({kts});
det_u1.rebase({kts});
det_v1.rebase({kts});
chem1.rebase({kts, 1});
s_awchem1.rebase({kts, 1});
scalars.rebase({kts, 1});
s_awscalars1.rebase({kts, 1});
qc_bl1.rebase({kts});
cldfra_bl1.rebase({kts});
qc_bl1_old.rebase({kts});
cldfra_bl1_old.rebase({kts});
pattern_spp_pbl1.rebase({kts});
tkeprod_up.rebase({kts});
el1.rebase({kts});
MynnArray<float, 1> edmf_th1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> s_aw2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
constexpr int nup = 8;
constexpr int debug_mf = 0;
float nup2;
MynnArray<float, 2> upw = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upthl = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqt = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqc = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqv = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upa = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upu = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upv = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upthv = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqke = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqnc = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqni = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqnwfa = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqnifa = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqnbca = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> ent = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {kte, nup});
int k;
int ip;
int k50;
float fltv2;
float wstar;
float qstar;
float thstar;
float sigmaw;
float sigmaqt;
float sigmath;
float z0;
float pwmin;
float pwmax;
float wmin;
float wmax;
float wlv;
float psig_w;
float maxw;
float maxqc;
float wpbl;
float b;
float qtn;
float thln;
float thvn;
float qcn;
float un;
float vn;
float qken;
float qncn;
float qnin;
float qnwfan;
float qnifan;
float qnbcan;
float upak;
float wn2;
float wn;
float entexp;
float entexm;
float entw;
float bcoeff;
float thvkm1;
float thvk;
float pk;
float rho_int;
constexpr float wa = ((2.f) / (3.f));
constexpr float wb = 0.002f;
constexpr float wc = 1.5f;
constexpr float atot = 0.10f;
constexpr float lmax = 1000.f;
constexpr float lmin = 300.f;
constexpr float dlmin = 0.f;
float minwidth;
float dl;
constexpr float dcut = 1.2f;
float d;
float cn;
float c;
float l;
float n;
float an2;
float hux;
float wspd_pbl;
float cloud_base;
float maxwidth_dx;
float maxwidth_pbl;
float maxwidth_cld;
float maxwidth_flx;
MynnArray<float, 1> chemn = mynn_array<float, 1, mynn_tile>(arena, {1}, {nchem});
MynnArray<float, 3> upchem = mynn_array<float, 3, mynn_tile>(arena, {kts, 1, 1}, {((kte) + (1)), nup, nchem});
int ic;
MynnArray<float, 2> edmf_chem = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {kte, nchem});
MynnArray<float, 1> scalarsn = mynn_array<float, 1, mynn_tile>(arena, {1}, {nscalars});
MynnArray<float, 3> upscalars = mynn_array<float, 3, mynn_tile>(arena, {kts, 1, 1}, {((kte) + (1)), nup, nscalars});
MynnArray<int, 1> ktop_plume = mynn_array<int, 1, mynn_tile>(arena, {1}, {nup});
bool superadiabatic;
float sigq;
float xl;
float rsl;
float cpm;
float a;
float qmq;
float aup;
float q1;
float diffqt;
float qsat_tk;
float fng;
float qww;
float alpha;
float beta;
float bb;
float f;
float pt;
float t;
float q2p;
float b9;
float satvp;
float rhgrid;
float entfac;
float cf_strat;
float qc_strat;
float cf_mf;
float qc_mf;
float qc_mf_min;
float pct_mf;
float wt2;
float dqwdz;
float tauc;
constexpr float cf_thresh = 0.5f;
MynnArray<float, 1> exneri = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dzi = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rhoi = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qti = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
float thp;
float qtp;
float qcp;
float qcs;
float esat;
float qsl;
float csigma;
float acfac;
float ac_wsp;
int overshoot;
float bvf;
float frz;
float dzp;
float adjustment;
float flx1;
float flt2;
constexpr float fluxportion = 0.75f;
MynnArray<float, 1> envm_a = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> envm_w = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> envm_thl = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> envm_sqv = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> envm_sqc = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> envm_u = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> envm_v = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> envi_a = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> envi_w = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
float temp;
float sublim;
float qc_ent;
float qv_ent;
float qt_ent;
float thl_ent;
float detrate;
float detrateuv;
float oow;
float exc_fac;
float aratio;
float detturb;
float qc_grid;
float qc_sgs;
float qc_plume;
float exc_heat;
float exc_moist;
float tk_int;
float tvs;
float dthvdz;
float zagl;
constexpr float cdet = ((1.f) / (45.f));
constexpr float dzpmax = 300.f;
constexpr float csub = 0.25f;
constexpr float pgfac = 0.00f;
float uk;
float ukm1;
float vk;
float vkm1;
float dxsa;
float qkebl;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6572
ktop = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6573
ztop = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6574
maxmf = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6575
maxwidth = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6576
excess_h = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6577
excess_q = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6579
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upw).size(); mynn_element += lane_width) { upw.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6580
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upthl).size(); mynn_element += lane_width) { upthl.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6581
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upthv).size(); mynn_element += lane_width) { upthv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6582
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqt).size(); mynn_element += lane_width) { upqt.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6583
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upa).size(); mynn_element += lane_width) { upa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6584
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upu).size(); mynn_element += lane_width) { upu.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6585
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upv).size(); mynn_element += lane_width) { upv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6586
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqc).size(); mynn_element += lane_width) { upqc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6587
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqv).size(); mynn_element += lane_width) { upqv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6588
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqke).size(); mynn_element += lane_width) { upqke.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6589
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnc).size(); mynn_element += lane_width) { upqnc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6590
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqni).size(); mynn_element += lane_width) { upqni.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6591
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnwfa).size(); mynn_element += lane_width) { upqnwfa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6592
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnifa).size(); mynn_element += lane_width) { upqnifa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6593
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnbca).size(); mynn_element += lane_width) { upqnbca.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6594
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6595
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upchem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { upchem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6596
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6597
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6598
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upscalars.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nscalars})).size(); mynn_element += lane_width) { upscalars.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nscalars}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6599
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6600
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (ent).size(); mynn_element += lane_width) { ent.linear(mynn_element) = 0.001f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6602
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1).size(); mynn_element += lane_width) { edmf_a1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6603
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1).size(); mynn_element += lane_width) { edmf_w1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6604
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt1).size(); mynn_element += lane_width) { edmf_qt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6605
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1).size(); mynn_element += lane_width) { edmf_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6606
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent1).size(); mynn_element += lane_width) { edmf_ent1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6607
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1).size(); mynn_element += lane_width) { edmf_qc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6608
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qv1).size(); mynn_element += lane_width) { edmf_qv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6609
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_u1).size(); mynn_element += lane_width) { edmf_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6610
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_v1).size(); mynn_element += lane_width) { edmf_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6611
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6612
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_chem.slice(MynnRange{kts, kte}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { edmf_chem.slice(MynnRange{kts, kte}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6613
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6615
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_aw1).size(); mynn_element += lane_width) { s_aw1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6616
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awthl1).size(); mynn_element += lane_width) { s_awthl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6617
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqt1).size(); mynn_element += lane_width) { s_awqt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6618
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqv1).size(); mynn_element += lane_width) { s_awqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6619
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqc1).size(); mynn_element += lane_width) { s_awqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6620
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awu1).size(); mynn_element += lane_width) { s_awu1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6621
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awv1).size(); mynn_element += lane_width) { s_awv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6622
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqke1).size(); mynn_element += lane_width) { s_awqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6623
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnc1).size(); mynn_element += lane_width) { s_awqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6624
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqni1).size(); mynn_element += lane_width) { s_awqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6625
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnwfa1).size(); mynn_element += lane_width) { s_awqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6626
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnifa1).size(); mynn_element += lane_width) { s_awqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6627
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnbca1).size(); mynn_element += lane_width) { s_awqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6628
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6629
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6630
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6631
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6632
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars})).size(); mynn_element += lane_width) { s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6633
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6635
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_thl1).size(); mynn_element += lane_width) { sub_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6636
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_sqv1).size(); mynn_element += lane_width) { sub_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6637
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_u1).size(); mynn_element += lane_width) { sub_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6638
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_v1).size(); mynn_element += lane_width) { sub_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6639
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_thl1).size(); mynn_element += lane_width) { det_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6640
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqv1).size(); mynn_element += lane_width) { det_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6641
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqc1).size(); mynn_element += lane_width) { det_sqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6642
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_u1).size(); mynn_element += lane_width) { det_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6643
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_v1).size(); mynn_element += lane_width) { det_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6644
nup2 = nup;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6645
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprod_up).size(); mynn_element += lane_width) { tkeprod_up.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6647
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6648
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6649
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6650
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6651
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6655
maxw = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6656
cloud_base = 9000.0f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6657
qkebl = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6658
for (k=1; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6659
zagl = r6_add(zw1(k), r6_mul(p5, dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6660
if (((zagl) > (r6_add(pblh, 500.f)))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6662
wpbl = w1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6663
if (((w1(k)) < (zero))) { wpbl = r6_mul(two, w1(k)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6664
maxw = mynn_max((float)(maxw), (float)(mynn_abs(wpbl)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6667
if (((zagl) <= (50.f))) { k50 = k; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6670
if (((k) <= (kpbl))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6671
qkebl = r6_add(qkebl, qke1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6672
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6675
qc_sgs = mynn_max((float)(qc1(k)), (float)(qc_bl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6676
if (((((((qc_sgs) > (1e-5f))) && (((cldfra_bl1(k)) >= (p5))))) && (((cloud_base) == (9000.0f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6677
cloud_base = zw1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6678
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6679
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6680
qkebl = r6_div(qkebl, ((float)(kpbl)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6683
maxw = mynn_max((float)(r6_sub(maxw, one)), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6684
psig_w = mynn_max((float)(r6_sub(one, maxw)), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6685
psig_w = mynn_min((float)(psig_w), (float)(psig_shcu));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6688
fltv2 = fltv;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6689
if (((((psig_w) == (zero))) && (((fltv) > (zero))))) { fltv2 = r6_neg(r6_mul(1.f, fltv)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6691
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6692
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6693
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6694
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6698
superadiabatic = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6699
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6700
hux = r6_neg(0.0024f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6701
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6702
hux = r6_neg(0.0030f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6703
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6704
tvs = r6_mul(ts, r6_add(one, r6_mul(p608, qv1(kts))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6705
if (((r6_div(r6_sub(thv1(kts), tvs), r6_mul(p5, dz1(kts)))) < (hux))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6707
superadiabatic = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6708
for (k=2; (1) > 0 ? k <= (mynn_max((int)(k50), (int)(2))) : k >= (mynn_max((int)(k50), (int)(2))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6709
hux = r6_neg(0.0018f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6710
dthvdz = r6_div(r6_sub(thv1(k), thv1(((k) - (1)))), r6_mul(p5, r6_add(dz1(k), dz1(((k) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6711
if (((dthvdz) < (hux))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6712
superadiabatic = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6713
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6714
superadiabatic = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6715
break;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6716
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6717
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6718
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6720
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6721
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6722
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6733
maxwidth_dx = mynn_min((float)(r6_mul(dx, dcut)), (float)(lmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6735
maxwidth_pbl = mynn_min((float)(r6_mul(1.1f, pblh)), (float)(lmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6737
if (((r6_sub(landsea, 1.5f)) < (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6738
maxwidth_cld = mynn_min((float)(mynn_max((float)(r6_mul(0.5f, cloud_base)), (float)(300.f))), (float)(lmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6739
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6740
maxwidth_cld = mynn_min((float)(mynn_max((float)(r6_mul(0.8f, cloud_base)), (float)(300.f))), (float)(lmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6741
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6743
wspd_pbl = mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts)))), (float)(0.01f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6746
if (((r6_sub(landsea, 1.5f)) < (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6747
maxwidth_flx = mynn_max((float)(zero), (float)(mynn_min((float)(r6_mul(1000.f, r6_add(r6_mul(0.6f, mynn_tanh(r6_div(r6_sub(fltv, 0.040f), 0.04f))), p5))), (float)(1000.f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6748
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6749
maxwidth_flx = mynn_max((float)(zero), (float)(mynn_min((float)(r6_mul(1000.f, r6_add(r6_mul(0.6f, mynn_tanh(r6_div(r6_sub(fltv, 0.007f), 0.02f))), p5))), (float)(1000.f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6751
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6752
maxwidth = mynn_min((float)(maxwidth_dx), (float)(maxwidth_pbl));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6753
maxwidth = mynn_min((float)(maxwidth), (float)(maxwidth_cld));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6754
maxwidth = mynn_min((float)(maxwidth), (float)(maxwidth_flx));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6755
minwidth = lmin;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6757
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6758
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6759
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6760
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6761
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6762
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6763
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6764
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6765
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6766
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6768
if (((maxwidth) <= (minwidth))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6769
nup2 = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6770
maxwidth = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6771
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6774
if (((((((((flt) > (zero))) && (((fltv2) > (0.008f))))) && (((maxwidth) > (minwidth))))) && (superadiabatic))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6777
cn = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6778
d = r6_neg(1.9f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6779
dl = r6_div(r6_sub(maxwidth, minwidth), ((float)(((nup) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6780
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6782
l = r6_add(minwidth, r6_mul(dl, ((float)(((ip) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6783
cn = r6_add(cn, r6_mul(r6_div(r6_mul(mynn_pow(l, d), r6_mul(l, l)), r6_mul(dx, dx)), dl));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6784
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6785
c = r6_div(atot, cn);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6788
if (((r6_sub(landsea, 1.5f)) < (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6789
acfac = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(fltv2, 0.02f), 0.05f))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6790
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6791
acfac = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(fltv2, 0.012f), 0.03f))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6792
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6798
if (((wspd_pbl) <= (10.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6799
ac_wsp = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6800
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6801
ac_wsp = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(zero), (float)(r6_sub(wspd_pbl, 13.0f))), 10.f)), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6802
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6804
acfac = mynn_min((float)(acfac), (float)(ac_wsp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6807
an2 = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6808
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6810
l = r6_add(minwidth, r6_mul(dl, ((float)(((ip) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6811
n = r6_mul(c, mynn_pow(l, d));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6812
upa(1, ip) = r6_mul(r6_div(r6_mul(r6_mul(n, l), l), r6_mul(dx, dx)), dl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6814
upa(1, ip) = r6_mul(upa(1, ip), acfac);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6815
an2 = r6_add(an2, upa(1, ip));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6818
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6820
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6821
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6822
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6823
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6824
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6825
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6826
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6829
z0 = 50.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6830
pwmin = 0.1f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6831
pwmax = 0.4f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6833
wstar = mynn_max((float)(mynn_pow(r6_mul(r6_mul(gtr, fltv2), pblh), p333)), (float)(1.e-2f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6834
qstar = r6_div(mynn_max((float)(flq), (float)(1.0e-5f)), wstar);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6835
thstar = r6_div(flt, wstar);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6837
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6838
csigma = 1.34f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6839
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6840
csigma = 1.34f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6841
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6843
if (env_subs) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6844
exc_fac = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6845
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6846
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6853
exc_fac = 9.28f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6854
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6856
exc_fac = 0.58f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6857
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6858
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6860
exc_fac = r6_mul(exc_fac, ac_wsp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6863
sigmaw = r6_mul(r6_mul(r6_mul(csigma, wstar), mynn_pow(r6_div(z0, pblh), p333)), r6_sub(one, r6_div(r6_mul(0.8f, z0), pblh)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6864
sigmaqt = r6_mul(r6_mul(csigma, qstar), mynn_pow(r6_div(z0, pblh), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6865
sigmath = r6_mul(r6_mul(csigma, thstar), mynn_pow(r6_div(z0, pblh), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6869
wmin = mynn_min((float)(r6_mul(sigmaw, pwmin)), (float)(0.1f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6870
wmax = mynn_min((float)(r6_mul(sigmaw, pwmax)), (float)(0.5f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6872
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6873
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6874
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6875
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6876
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6877
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6880
{ const int mynn_lo0 = (1), mynn_hi0 = (nup), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { ip = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6881
wlv = r6_add(wmin, r6_mul(r6_div(r6_sub(wmax, wmin), ((float)(nup2))), ((float)(((ip) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6882
upa(2, ip) = upa(1, ip);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6884
upw(2, ip) = r6_add(wmin, r6_mul(r6_div(((float)(ip)), ((float)(nup))), r6_sub(wmax, wmin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6885
upu(2, ip) = r6_div(r6_add(r6_mul(u1(kts), dz1(((kts) + (1)))), r6_mul(u1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6886
upv(2, ip) = r6_div(r6_add(r6_mul(v1(kts), dz1(((kts) + (1)))), r6_mul(v1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6887
upqc(2, ip) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6890
exc_heat = r6_div(r6_mul(r6_mul(exc_fac, upw(2, ip)), sigmath), sigmaw);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6891
upthv(2, ip) = r6_add(r6_div(r6_add(r6_mul(thv1(kts), dz1(((kts) + (1)))), r6_mul(thv1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1))))), exc_heat);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6893
upthl(2, ip) = r6_add(r6_div(r6_add(r6_mul(thl1(kts), dz1(((kts) + (1)))), r6_mul(thl1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1))))), exc_heat);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6896
exc_moist = r6_div(r6_mul(r6_mul(exc_fac, upw(2, ip)), sigmaqt), sigmaw);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6897
upqt(2, ip) = r6_add(r6_div(r6_add(r6_mul(qt1(kts), dz1(((kts) + (1)))), r6_mul(qt1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1))))), exc_moist);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6899
upqke(2, ip) = r6_div(r6_add(r6_mul(qke1(kts), dz1(((kts) + (1)))), r6_mul(qke1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6900
upqnc(2, ip) = r6_div(r6_add(r6_mul(qnc1(kts), dz1(((kts) + (1)))), r6_mul(qnc1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6901
upqni(2, ip) = r6_div(r6_add(r6_mul(qni1(kts), dz1(((kts) + (1)))), r6_mul(qni1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6902
upqnwfa(2, ip) = r6_div(r6_add(r6_mul(qnwfa1(kts), dz1(((kts) + (1)))), r6_mul(qnwfa1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6903
upqnifa(2, ip) = r6_div(r6_add(r6_mul(qnifa1(kts), dz1(((kts) + (1)))), r6_mul(qnifa1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6904
upqnbca(2, ip) = r6_div(r6_add(r6_mul(qnbca1(kts), dz1(((kts) + (1)))), r6_mul(qnbca1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6905
} mynn_lane_sync(); ip = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; exc_heat = mynn_lane_bcast(exc_heat, mynn_src0); exc_moist = mynn_lane_bcast(exc_moist, mynn_src0); wlv = mynn_lane_bcast(wlv, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6907
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6908
{ const int mynn_lo1 = (1), mynn_hi1 = (nup), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { ip = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6909
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6910
upchem(2, ip, ic) = r6_div(r6_add(r6_mul(chem1(kts, ic), dz1(((kts) + (1)))), r6_mul(chem1(((kts) + (1)), ic), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6911
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6912
} mynn_lane_sync(); ip = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; ic = mynn_lane_bcast(ic, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6913
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6914
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6915
{ const int mynn_lo2 = (1), mynn_hi2 = (nup), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { ip = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6916
for (ic=1; (1) > 0 ? ic <= (nscalars) : ic >= (nscalars); ic += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6917
upscalars(2, ip, ic) = r6_div(r6_add(r6_mul(scalars(kts, ic), dz1(((kts) + (1)))), r6_mul(scalars(((kts) + (1)), ic), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6918
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6919
} mynn_lane_sync(); ip = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; ic = mynn_lane_bcast(ic, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6920
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6923
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_thl.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_thl.slice(MynnRange{kts, kte}).linear(mynn_element) = thl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6924
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_sqv.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_sqv.slice(MynnRange{kts, kte}).linear(mynn_element) = qv1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6925
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_sqc.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_sqc.slice(MynnRange{kts, kte}).linear(mynn_element) = qc1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6926
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_u.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_u.slice(MynnRange{kts, kte}).linear(mynn_element) = u1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6927
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_v.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_v.slice(MynnRange{kts, kte}).linear(mynn_element) = v1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6929
rhoi(kts) = rho1(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6930
dzi(kts) = r6_mul(p5, dz1(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6931
exneri(kts) = ex1(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6932
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6933
rhoi(k) = r6_div(r6_add(r6_mul(rho1(((k) - (1))), dz1(k)), r6_mul(rho1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6934
dzi(k) = r6_div(r6_add(r6_mul(dz1(((k) - (1))), dz1(k)), r6_mul(dz1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6935
exneri(k) = r6_div(r6_add(r6_mul(ex1(((k) - (1))), dz1(k)), r6_mul(ex1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6936
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6939
dxsa = r6_sub(one, mynn_min((float)(mynn_max((float)(zero), (float)(r6_div(r6_sub(12000.0f, dx), r6_sub(12000.0f, 3000.0f))))), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6941
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (ktop_plume).size(); mynn_element += lane_width) { ktop_plume.linear(mynn_element) = 0; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6943
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6944
qcn = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6945
overshoot = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6946
l = r6_add(minwidth, r6_mul(dl, ((float)(((ip) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6947
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (2))) : k >= (((kte) - (2))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6948
zagl = r6_add(zw1(k), r6_mul(p5, dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6951
wmin = r6_add(p3, r6_mul(l, 0.0005f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6959
entfac = r6_mul(0.21f, mynn_min((float)(mynn_max((float)(mynn_sqrt(qkebl)), (float)(1.30f))), (float)(1.62f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6962
wt2 = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, pblh)), (float)(zero)), 500.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6963
entfac = r6_add(r6_mul(entfac, r6_sub(one, wt2)), r6_mul(wt2, 0.33f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6964
ent(k, ip) = r6_div(entfac, r6_mul(mynn_min((float)(mynn_max((float)(upw(((k) - (1)), ip)), (float)(wmin))), (float)(one)), l));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6972
ent(k, ip) = mynn_max((float)(ent(k, ip)), (float)(0.0003f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6976
if (((zw1(k)) >= (mynn_min((float)(r6_add(pblh, 1500.f)), (float)(4000.f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6977
ent(k, ip) = r6_add(ent(k, ip), r6_mul(r6_sub(zw1(k), mynn_min((float)(r6_add(pblh, 1500.f)), (float)(4000.f))), 5.0e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6978
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6981
ent(k, ip) = r6_mul(ent(k, ip), r6_sub(one, pattern_spp_pbl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6983
ent(k, ip) = mynn_min((float)(ent(k, ip)), (float)(r6_div(0.9f, dz1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6986
uk = r6_div(r6_add(r6_mul(u1(k), dz1(((k) + (1)))), r6_mul(u1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6987
ukm1 = r6_div(r6_add(r6_mul(u1(((k) - (1))), dz1(k)), r6_mul(u1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6988
vk = r6_div(r6_add(r6_mul(v1(k), dz1(((k) + (1)))), r6_mul(v1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6989
vkm1 = r6_div(r6_add(r6_mul(v1(((k) - (1))), dz1(k)), r6_mul(v1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6992
entexp = r6_mul(ent(k, ip), dz1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6993
entexm = r6_mul(entexp, p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6994
qtn = r6_add(r6_mul(upqt(k, ip), r6_sub(one, entexp)), r6_mul(qt1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6995
thln = r6_add(r6_mul(upthl(k, ip), r6_sub(one, entexp)), r6_mul(thl1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6996
un = r6_add(r6_add(r6_mul(upu(k, ip), r6_sub(one, entexm)), r6_mul(u1(k), entexm)), r6_mul(r6_mul(dxsa, pgfac), r6_sub(uk, ukm1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6997
vn = r6_add(r6_add(r6_mul(upv(k, ip), r6_sub(one, entexm)), r6_mul(v1(k), entexm)), r6_mul(r6_mul(dxsa, pgfac), r6_sub(vk, vkm1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6998
qken = r6_add(r6_mul(upqke(k, ip), r6_sub(one, entexp)), r6_mul(qke1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6999
qncn = r6_add(r6_mul(upqnc(k, ip), r6_sub(one, entexp)), r6_mul(qnc1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7000
qnin = r6_add(r6_mul(upqni(k, ip), r6_sub(one, entexp)), r6_mul(qni1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7001
qnwfan = r6_add(r6_mul(upqnwfa(k, ip), r6_sub(one, entexp)), r6_mul(qnwfa1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7002
qnifan = r6_add(r6_mul(upqnifa(k, ip), r6_sub(one, entexp)), r6_mul(qnifa1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7003
qnbcan = r6_add(r6_mul(upqnbca(k, ip), r6_sub(one, entexp)), r6_mul(qnbca1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7007
qc_ent = qcn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7008
qt_ent = qtn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7009
thl_ent = thln;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7019
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7020
{ const int mynn_lo4 = (1), mynn_hi4 = (nchem), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { ic = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7024
chemn(ic) = r6_add(r6_mul(upchem(k, ip, ic), r6_sub(one, entexp)), r6_mul(chem1(k, ic), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7025
} mynn_lane_sync(); ic = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7026
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7027
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7028
{ const int mynn_lo5 = (1), mynn_hi5 = (nscalars), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { ic = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7029
scalarsn(ic) = r6_add(r6_mul(upscalars(k, ip, ic), r6_sub(one, entexp)), r6_mul(scalars(k, ic), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7030
} mynn_lane_sync(); ic = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7031
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7034
pk = r6_div(r6_add(r6_mul(pres1(k), dz1(((k) + (1)))), r6_mul(pres1(((k) + (1))), dz1(k))), r6_add(dz1(k), dz1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7036
f_condensation_edmf(qtn, thln, pk, zw1(((k) + (1))), thvn, qcn);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7039
thvk = r6_div(r6_add(r6_mul(thv1(k), dz1(((k) + (1)))), r6_mul(thv1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7040
thvkm1 = r6_div(r6_add(r6_mul(thv1(((k) - (1))), dz1(k)), r6_mul(thv1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7043
b = r6_mul(grav, r6_sub(r6_div(thvn, thvk), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7044
if (((b) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7045
bcoeff = 0.15f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7046
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7047
bcoeff = 0.2f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7048
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7058
if (((upw(((k) - (1)), ip)) < (0.2f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7059
wn = r6_add(upw(k, ip), r6_mul(r6_add(r6_mul(r6_neg(r6_mul(2.f, ent(k, ip))), upw(k, ip)), r6_div(r6_mul(bcoeff, b), mynn_max((float)(upw(k, ip)), (float)(0.2f)))), mynn_min((float)(dz1(k)), (float)(250.f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7060
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7061
wn = r6_add(upw(k, ip), r6_mul(r6_add(r6_mul(r6_neg(r6_mul(2.f, ent(k, ip))), upw(k, ip)), r6_div(r6_mul(bcoeff, b), upw(k, ip))), mynn_min((float)(dz1(k)), (float)(250.f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7062
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7065
if (((wn) > (r6_add(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7066
wn = r6_add(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7067
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7069
if (((wn) < (r6_sub(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7070
wn = r6_sub(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7071
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7072
wn = mynn_min((float)(mynn_max((float)(zero), (float)(wn))), (float)(3.0f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7076
if (((((k) == (((kts) + (1))))) && (((wn) == (zero))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7077
nup2 = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7078
break;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7079
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7081
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7082
if (((wn) >= (3.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7084
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7085
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7086
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7087
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7088
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7089
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7092
if (((((wn) <= (zero))) && (((overshoot) == (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7093
overshoot = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7094
if (((r6_sub(thvk, thvkm1)) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7095
bvf = mynn_sqrt(r6_div(r6_mul(gtr, r6_sub(thvk, thvkm1)), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7097
frz = r6_div(upw(k, ip), r6_mul(bvf, dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7099
dzp = r6_mul(dz1(k), mynn_max((float)(zero), (float)(mynn_min((float)(frz), (float)(one)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7100
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7101
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7102
dzp = dz1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7103
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7113
aratio = mynn_min((float)(r6_div(upa(k, ip), r6_sub(one, upa(k, ip)))), (float)(p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7114
detturb = 0.00008f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7115
oow = r6_neg(r6_div(0.060f, mynn_max((float)(r6_mul(0.5f, r6_add(wn, upw(k, ip)))), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7116
detrate = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(oow, r6_sub(wn, upw(k, ip))), dz1(k))), (float)(detturb))), (float)(.0002f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7117
detrateuv = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(oow, r6_sub(wn, upw(k, ip))), dz1(k))), (float)(detturb))), (float)(.0001f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7118
envm_thl(k) = r6_add(envm_thl(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(thl_ent, upthl(k, ip))), thl1(k)), detrate), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7119
qv_ent = r6_mul(p5, r6_add(mynn_max((float)(zero), (float)(r6_sub(qt_ent, qc_ent))), mynn_max((float)(zero), (float)(r6_sub(upqt(k, ip), upqc(k, ip))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7120
envm_sqv(k) = r6_add(envm_sqv(k), r6_mul(r6_mul(r6_mul(r6_sub(qv_ent, qv1(k)), detrate), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7121
if (((upqc(k, ip)) > (1e-8f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7122
if (((qc1(k)) > (1e-6f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7123
qc_grid = qc1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7124
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7125
qc_grid = qc_bl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7126
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7127
envm_sqc(k) = r6_add(envm_sqc(k), r6_mul(r6_mul(r6_mul(mynn_max((float)(zero), (float)(r6_sub(r6_mul(r6_mul(upa(k, ip), p5), r6_add(qcn, upqc(k, ip))), qc_grid))), detrate), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7128
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7129
envm_u(k) = r6_add(envm_u(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(un, upu(k, ip))), u1(k)), detrateuv), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7130
envm_v(k) = r6_add(envm_v(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(vn, upv(k, ip))), v1(k)), detrateuv), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7132
if (((wn) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7134
upw(((k) + (1)), ip) = wn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7135
upthv(((k) + (1)), ip) = thvn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7136
upthl(((k) + (1)), ip) = thln;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7137
upqt(((k) + (1)), ip) = qtn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7138
upqc(((k) + (1)), ip) = qcn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7139
upu(((k) + (1)), ip) = un;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7140
upv(((k) + (1)), ip) = vn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7141
upqke(((k) + (1)), ip) = qken;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7142
upqnc(((k) + (1)), ip) = qncn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7143
upqni(((k) + (1)), ip) = qnin;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7144
upqnwfa(((k) + (1)), ip) = qnwfan;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7145
upqnifa(((k) + (1)), ip) = qnifan;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7146
upqnbca(((k) + (1)), ip) = qnbcan;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7147
upa(((k) + (1)), ip) = upa(k, ip);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7148
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7149
{ const int mynn_lo6 = (1), mynn_hi6 = (nchem), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { ic = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7150
upchem(((k) + (1)), ip, ic) = chemn(ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7151
} mynn_lane_sync(); ic = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7152
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7153
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7154
{ const int mynn_lo7 = (1), mynn_hi7 = (nscalars), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { ic = mynn_lo7 + mynn_it7*mynn_st7;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7155
upscalars(((k) + (1)), ip, ic) = scalarsn(ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7156
} mynn_lane_sync(); ic = mynn_lo7 + mynn_n7*mynn_st7; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7157
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7158
ktop = mynn_max((int)(ktop), (int)(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7159
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7160
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7161
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7162
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7163
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7164
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7165
break;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7166
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7167
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7169
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7170
if (((((((((mynn_maxval(upw.slice(MynnRange{upw.lo[0], upw.hi[0]}, ip))) > (10.0f))) || (((mynn_minval(upa.slice(MynnRange{upa.lo[0], upa.hi[0]}, ip))) < (zero))))) || (((mynn_maxval(upa.slice(MynnRange{upa.lo[0], upa.hi[0]}, ip))) > (atot))))) || (((nup2) > (10))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7173
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7174
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7175
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7177
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7178
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7179
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7180
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7181
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7182
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7183
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7184
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7185
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7186
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7187
ktop_plume(ip) = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7188
excess_h = mynn_max((float)(excess_h), (float)(exc_heat));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7189
excess_q = mynn_max((float)(excess_q), (float)(exc_moist));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7190
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7191
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7193
nup2 = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7194
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7196
ktop = mynn_min((int)(ktop), (int)(((kte) - (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7197
if (((ktop) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7198
ztop = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7199
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7200
ztop = zw1(ktop);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7201
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7203
if (((nup2) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7206
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7207
{ const int mynn_lo8 = (kts), mynn_hi8 = (((kte) - (1))), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7208
s_aw1(k) = r6_add(s_aw1(k), r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7209
s_awthl1(k) = r6_add(s_awthl1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upthl(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7210
s_awqt1(k) = r6_add(s_awqt1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqt(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7215
qc_plume = upqc(k, ip);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7219
s_awqc1(k) = r6_add(s_awqc1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), qc_plume), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7220
s_awqv1(k) = r6_sub(s_awqt1(k), s_awqc1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7221
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; if (mynn_n8 > 0) { const int mynn_src8 = (mynn_n8 - 1) % lane_width; qc_plume = mynn_lane_bcast(qc_plume, mynn_src8); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7222
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7224
if (((momentum_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7225
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7226
{ const int mynn_lo9 = (kts), mynn_hi9 = (((kte) - (1))), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7227
s_awu1(k) = r6_add(s_awu1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upu(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7228
s_awv1(k) = r6_add(s_awv1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upv(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7229
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7230
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7231
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7233
if (((tke_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7234
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7235
{ const int mynn_lo10 = (kts), mynn_hi10 = (((kte) - (1))), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7236
s_awqke1(k) = r6_add(s_awqke1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqke(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7237
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7238
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7239
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7241
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7242
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7243
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7244
{ const int mynn_lo11 = (1), mynn_hi11 = (nchem), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { ic = mynn_lo11 + mynn_it11*mynn_st11;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7245
s_awchem1(k, ic) = r6_add(s_awchem1(k, ic), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upchem(k, ip, ic)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7246
} mynn_lane_sync(); ic = mynn_lo11 + mynn_n11*mynn_st11; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7247
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7248
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7249
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7251
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7252
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7253
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7254
{ const int mynn_lo12 = (1), mynn_hi12 = (nscalars), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { ic = mynn_lo12 + mynn_it12*mynn_st12;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7255
s_awscalars1(k, ic) = r6_add(s_awscalars1(k, ic), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upscalars(k, ip, ic)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7256
} mynn_lane_sync(); ic = mynn_lo12 + mynn_n12*mynn_st12; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7257
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7258
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7259
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7260
if (((numcon_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7261
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7262
{ const int mynn_lo13 = (kts), mynn_hi13 = (((kte) - (1))), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { k = mynn_lo13 + mynn_it13*mynn_st13;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7263
s_awqnc1(k) = r6_add(s_awqnc1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnc(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7264
s_awqni1(k) = r6_add(s_awqni1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqni(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7265
} mynn_lane_sync(); k = mynn_lo13 + mynn_n13*mynn_st13; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7266
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7267
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7268
if (((aerosol_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7269
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7270
{ const int mynn_lo14 = (kts), mynn_hi14 = (((kte) - (1))), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { k = mynn_lo14 + mynn_it14*mynn_st14;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7271
s_awqnwfa1(k) = r6_add(s_awqnwfa1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnwfa(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7272
s_awqnifa1(k) = r6_add(s_awqnifa1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnifa(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7273
s_awqnbca1(k) = r6_add(s_awqnbca1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnbca(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7274
} mynn_lane_sync(); k = mynn_lo14 + mynn_n14*mynn_st14; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7275
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7276
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7281
if (((s_aw1(((kts) + (1)))) != (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7282
flx1 = mynn_max((float)(r6_div(r6_mul(s_aw1(((kts) + (1))), r6_sub(thv1(kts), thv1(((kts) + (1))))), dzi(((kts) + (1))))), (float)(1.0e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7283
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7284
flx1 = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7287
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7288
adjustment = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7289
flt2 = mynn_max((float)(zero), (float)(fltv));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7292
if (((((flx1) > (r6_div(r6_mul(fluxportion, flt2), dz1(kts))))) && (((flx1) > (zero))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7293
adjustment = mynn_max((float)(r6_div(r6_div(r6_mul(fluxportion, flt2), dz1(kts)), flx1)), (float)(0.01f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7294
for (int mynn_element=0; mynn_element < (s_aw1).size(); ++mynn_element) { s_aw1.linear(mynn_element) = r6_mul(s_aw1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7295
for (int mynn_element=0; mynn_element < (s_awthl1).size(); ++mynn_element) { s_awthl1.linear(mynn_element) = r6_mul(s_awthl1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7296
for (int mynn_element=0; mynn_element < (s_awqt1).size(); ++mynn_element) { s_awqt1.linear(mynn_element) = r6_mul(s_awqt1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7297
for (int mynn_element=0; mynn_element < (s_awqc1).size(); ++mynn_element) { s_awqc1.linear(mynn_element) = r6_mul(s_awqc1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7298
for (int mynn_element=0; mynn_element < (s_awqv1).size(); ++mynn_element) { s_awqv1.linear(mynn_element) = r6_mul(s_awqv1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7299
if (((numcon_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7300
for (int mynn_element=0; mynn_element < (s_awqnc1).size(); ++mynn_element) { s_awqnc1.linear(mynn_element) = r6_mul(s_awqnc1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7301
for (int mynn_element=0; mynn_element < (s_awqni1).size(); ++mynn_element) { s_awqni1.linear(mynn_element) = r6_mul(s_awqni1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7302
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7303
if (((aerosol_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7304
for (int mynn_element=0; mynn_element < (s_awqnwfa1).size(); ++mynn_element) { s_awqnwfa1.linear(mynn_element) = r6_mul(s_awqnwfa1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7305
for (int mynn_element=0; mynn_element < (s_awqnifa1).size(); ++mynn_element) { s_awqnifa1.linear(mynn_element) = r6_mul(s_awqnifa1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7306
for (int mynn_element=0; mynn_element < (s_awqnbca1).size(); ++mynn_element) { s_awqnbca1.linear(mynn_element) = r6_mul(s_awqnbca1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7307
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7308
if (((momentum_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7309
for (int mynn_element=0; mynn_element < (s_awu1).size(); ++mynn_element) { s_awu1.linear(mynn_element) = r6_mul(s_awu1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7310
for (int mynn_element=0; mynn_element < (s_awv1).size(); ++mynn_element) { s_awv1.linear(mynn_element) = r6_mul(s_awv1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7311
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7312
if (((tke_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7313
for (int mynn_element=0; mynn_element < (s_awqke1).size(); ++mynn_element) { s_awqke1.linear(mynn_element) = r6_mul(s_awqke1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7314
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7315
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7316
for (int mynn_element=0; mynn_element < (s_awchem1).size(); ++mynn_element) { s_awchem1.linear(mynn_element) = r6_mul(s_awchem1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7317
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7318
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7319
for (int mynn_element=0; mynn_element < (s_awscalars1).size(); ++mynn_element) { s_awscalars1.linear(mynn_element) = r6_mul(s_awscalars1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7320
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7321
for (int mynn_element=0; mynn_element < (upa).size(); ++mynn_element) { upa.linear(mynn_element) = r6_mul(upa.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7322
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7323
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7324
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7325
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7326
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7327
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7331
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7332
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7333
edmf_a1(k) = r6_add(edmf_a1(k), upa(((k) + (1)), ip));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7334
edmf_w1(k) = r6_add(edmf_w1(k), r6_mul(r6_mul(upa(((k) + (1)), ip), p5), upw(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7335
edmf_qt1(k) = r6_add(edmf_qt1(k), r6_mul(upa(((k) + (1)), ip), upqt(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7336
edmf_thl1(k) = r6_add(edmf_thl1(k), r6_mul(upa(((k) + (1)), ip), upthl(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7337
edmf_ent1(k) = r6_add(edmf_ent1(k), r6_mul(upa(((k) + (1)), ip), ent(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7338
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_mul(upa(((k) + (1)), ip), upqc(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7339
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7340
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7341
{ const int mynn_lo15 = (((kts) + (1))), mynn_hi15 = (((ktop_plume(ip)) - (1))), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { k = mynn_lo15 + mynn_it15*mynn_st15;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7342
upak = r6_div(r6_add(r6_mul(upa(((k) + (1)), ip), dzi(k)), r6_mul(upa(k, ip), dzi(((k) + (1))))), r6_add(dzi(((k) + (1))), dzi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7343
edmf_a1(k) = r6_add(edmf_a1(k), upak);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7344
edmf_w1(k) = r6_add(edmf_w1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upw(((k) + (1)), ip), dzi(k)), r6_mul(upw(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7345
edmf_qt1(k) = r6_add(edmf_qt1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upqt(((k) + (1)), ip), dzi(k)), r6_mul(upqt(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7346
edmf_thl1(k) = r6_add(edmf_thl1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upthl(((k) + (1)), ip), dzi(k)), r6_mul(upthl(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7347
edmf_ent1(k) = r6_add(edmf_ent1(k), r6_div(r6_mul(upak, r6_add(r6_mul(ent(((k) + (1)), ip), dzi(k)), r6_mul(ent(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7348
if (((((upqc(k, ip)) == (zero))) && (((upqc(((k) + (1)), ip)) > (zero))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7350
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_mul(upak, upqc(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7351
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7352
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upqc(((k) + (1)), ip), dzi(k)), r6_mul(upqc(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7353
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7354
} mynn_lane_sync(); k = mynn_lo15 + mynn_n15*mynn_st15; if (mynn_n15 > 0) { const int mynn_src15 = (mynn_n15 - 1) % lane_width; upak = mynn_lane_bcast(upak, mynn_src15); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7355
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7357
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7358
k = ktop_plume(ip);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7359
edmf_a1(k) = r6_add(edmf_a1(k), upa(k, ip));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7360
edmf_w1(k) = r6_add(edmf_w1(k), r6_mul(upa(k, ip), upw(k, ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7361
edmf_qt1(k) = r6_add(edmf_qt1(k), r6_mul(upa(k, ip), upqt(k, ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7362
edmf_thl1(k) = r6_add(edmf_thl1(k), r6_mul(upa(k, ip), upthl(k, ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7363
edmf_ent1(k) = r6_add(edmf_ent1(k), r6_mul(upa(k, ip), ent(k, ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7364
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_mul(upa(k, ip), upqc(k, ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7365
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7366
{ const int mynn_lo16 = (kts), mynn_hi16 = (((kte) - (1))), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { k = mynn_lo16 + mynn_it16*mynn_st16;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7368
if (((edmf_a1(k)) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7369
edmf_w1(k) = r6_div(edmf_w1(k), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7370
edmf_qt1(k) = r6_div(edmf_qt1(k), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7371
edmf_thl1(k) = r6_div(edmf_thl1(k), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7372
edmf_ent1(k) = r6_div(edmf_ent1(k), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7373
edmf_qc1(k) = r6_div(edmf_qc1(k), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7374
edmf_a1(k) = r6_mul(edmf_a1(k), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7375
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7376
} mynn_lane_sync(); k = mynn_lo16 + mynn_n16*mynn_st16; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7377
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7379
if (((r6_mul(edmf_a1(k), edmf_w1(k))) > (maxmf))) { maxmf = r6_mul(edmf_a1(k), edmf_w1(k)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7382
tkeprod_up(k) = r6_div(r6_mul(mynn_ipow<3>(mynn_abs(edmf_w1(k))), edmf_a1(k)), r6_mul(b1, mynn_max((float)(r6_mul(p5, r6_add(el1(k), el1(((k) + (1)))))), (float)(0.1f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7383
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7386
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7387
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7388
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7389
{ const int mynn_lo17 = (1), mynn_hi17 = (nchem), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { ic = mynn_lo17 + mynn_it17*mynn_st17;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7390
upak = r6_div(r6_add(r6_mul(upa(((k) + (1)), ip), dzi(k)), r6_mul(upa(k, ip), dzi(((k) + (1))))), r6_add(dzi(((k) + (1))), dzi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7391
edmf_chem(k, ic) = r6_add(edmf_chem(k, ic), r6_div(r6_mul(upak, r6_add(r6_mul(upchem(((k) + (1)), ip, ic), dzi(k)), r6_mul(upchem(k, ip, ic), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7392
} mynn_lane_sync(); ic = mynn_lo17 + mynn_n17*mynn_st17; if (mynn_n17 > 0) { const int mynn_src17 = (mynn_n17 - 1) % lane_width; upak = mynn_lane_bcast(upak, mynn_src17); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7393
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7394
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7395
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7396
if (((edmf_a1(k)) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7397
{ const int mynn_lo18 = (1), mynn_hi18 = (nchem), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { ic = mynn_lo18 + mynn_it18*mynn_st18;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7398
edmf_chem(k, ic) = r6_div(edmf_chem(k, ic), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7399
} mynn_lane_sync(); ic = mynn_lo18 + mynn_n18*mynn_st18; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7400
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7401
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7402
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7406
if (env_subs) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7407
{ const int mynn_lo19 = (((kts) + (1))), mynn_hi19 = (((kte) - (1))), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { k = mynn_lo19 + mynn_it19*mynn_st19;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7412
envi_w(k) = r6_mul(p333, r6_add(r6_add(edmf_w1(((k) - (1))), edmf_w1(k)), edmf_w1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7413
envi_a(k) = r6_mul(r6_mul(p333, r6_add(r6_add(edmf_a1(((k) - (1))), edmf_a1(k)), edmf_a1(((k) + (1))))), adjustment);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7414
} mynn_lane_sync(); k = mynn_lo19 + mynn_n19*mynn_st19; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7416
envi_w(kts) = edmf_w1(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7417
envi_a(kts) = edmf_a1(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7419
envi_w(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7420
envi_a(kte) = edmf_a1(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7422
envi_w(((kte) + (1))) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7423
envi_a(((kte) + (1))) = edmf_a1(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7427
if (((envi_w(kts)) > (r6_div(r6_mul(0.9f, dz1(kts)), dt)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7428
sublim = r6_div(r6_div(r6_mul(0.9f, dz1(kts)), dt), envi_w(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7429
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7430
sublim = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7431
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7433
{ const int mynn_lo20 = (kts), mynn_hi20 = (kte), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { k = mynn_lo20 + mynn_it20*mynn_st20;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7434
temp = envi_a(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7435
envi_a(k) = r6_sub(one, temp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7436
envi_w(k) = r6_div(r6_mul(r6_mul(r6_mul(csub, sublim), envi_w(k)), temp), r6_sub(1.f, temp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7437
} mynn_lane_sync(); k = mynn_lo20 + mynn_n20*mynn_st20; if (mynn_n20 > 0) { const int mynn_src20 = (mynn_n20 - 1) % lane_width; temp = mynn_lane_bcast(temp, mynn_src20); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7440
sub_thl1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), thl1(((kts) + (1)))), r6_mul(rho1(kts), thl1(kts)))), dzi(kts)), rhoi(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7442
sub_sqv1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), qv1(((kts) + (1)))), r6_mul(rho1(kts), qv1(kts)))), dzi(kts)), rhoi(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7444
{ const int mynn_lo21 = (((kts) + (1))), mynn_hi21 = (((kte) - (1))), mynn_st21 = (1); const int mynn_n21 = mynn_trip(mynn_lo21, mynn_hi21, mynn_st21); mynn_lane_sync(); for (int mynn_it21 = lane_sub; mynn_it21 < mynn_n21; mynn_it21 += lane_width) { k = mynn_lo21 + mynn_it21*mynn_st21;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7445
dzi(k) = r6_mul(0.5f, r6_add(dz1(k), dz1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7446
sub_thl1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), thl1(((k) + (1)))), r6_mul(rho1(k), thl1(k)))), dzi(k)), rhoi(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7448
sub_sqv1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), qv1(((k) + (1)))), r6_mul(rho1(k), qv1(k)))), dzi(k)), rhoi(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7450
} mynn_lane_sync(); k = mynn_lo21 + mynn_n21*mynn_st21; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7452
{ const int mynn_lo22 = (kts), mynn_hi22 = (((kte) - (1))), mynn_st22 = (1); const int mynn_n22 = mynn_trip(mynn_lo22, mynn_hi22, mynn_st22); mynn_lane_sync(); for (int mynn_it22 = lane_sub; mynn_it22 < mynn_n22; mynn_it22 += lane_width) { k = mynn_lo22 + mynn_it22*mynn_st22;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7453
det_thl1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_thl(k), thl1(k))), envi_a(k)), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7454
det_sqv1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_sqv(k), qv1(k))), envi_a(k)), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7455
det_sqc1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_sqc(k), qc1(k))), envi_a(k)), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7456
} mynn_lane_sync(); k = mynn_lo22 + mynn_n22*mynn_st22; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7458
if (((momentum_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7459
sub_u1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), u1(((kts) + (1)))), r6_mul(rho1(kts), u1(kts)))), dzi(kts)), rhoi(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7461
sub_v1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), v1(((kts) + (1)))), r6_mul(rho1(kts), v1(kts)))), dzi(kts)), rhoi(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7463
{ const int mynn_lo23 = (((kts) + (1))), mynn_hi23 = (((kte) - (1))), mynn_st23 = (1); const int mynn_n23 = mynn_trip(mynn_lo23, mynn_hi23, mynn_st23); mynn_lane_sync(); for (int mynn_it23 = lane_sub; mynn_it23 < mynn_n23; mynn_it23 += lane_width) { k = mynn_lo23 + mynn_it23*mynn_st23;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7464
sub_u1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), u1(((k) + (1)))), r6_mul(rho1(k), u1(k)))), dzi(k)), rhoi(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7466
sub_v1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), v1(((k) + (1)))), r6_mul(rho1(k), v1(k)))), dzi(k)), rhoi(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7468
} mynn_lane_sync(); k = mynn_lo23 + mynn_n23*mynn_st23; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7470
{ const int mynn_lo24 = (kts), mynn_hi24 = (((kte) - (1))), mynn_st24 = (1); const int mynn_n24 = mynn_trip(mynn_lo24, mynn_hi24, mynn_st24); mynn_lane_sync(); for (int mynn_it24 = lane_sub; mynn_it24 < mynn_n24; mynn_it24 += lane_width) { k = mynn_lo24 + mynn_it24*mynn_st24;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7471
det_u1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_u(k), u1(k))), envi_a(k)), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7472
det_v1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_v(k), v1(k))), envi_a(k)), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7473
} mynn_lane_sync(); k = mynn_lo24 + mynn_n24*mynn_st24; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7474
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7475
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7479
{ const int mynn_lo25 = (kts), mynn_hi25 = (((kte) - (1))), mynn_st25 = (1); const int mynn_n25 = mynn_trip(mynn_lo25, mynn_hi25, mynn_st25); mynn_lane_sync(); for (int mynn_it25 = lane_sub; mynn_it25 < mynn_n25; mynn_it25 += lane_width) { k = mynn_lo25 + mynn_it25*mynn_st25;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7480
edmf_th1(k) = r6_add(edmf_thl1(k), r6_mul(r6_div(xlvcp, ex1(k)), edmf_qc1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7481
qti(((k) + (1))) = r6_div(r6_add(r6_mul(qt1(k), dz1(((k) + (1)))), r6_mul(qt1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7482
} mynn_lane_sync(); k = mynn_lo25 + mynn_n25*mynn_st25; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7483
qti(kts) = qt1(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7488
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (2))) : k >= (((kte) - (2))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7489
if (((k) > (ktop))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7490
if (((edmf_qc1(k)) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7492
aup = edmf_a1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7493
thp = edmf_th1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7494
qtp = edmf_qt1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7495
qcp = edmf_qc1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7496
dqwdz = r6_div(r6_sub(qti(k), qti(((k) + (1)))), dz1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7500
esat = f_esat_blend(tk1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7502
qsl = r6_div(r6_mul(ep_2, esat), mynn_max((float)(r6_sub(pres1(k), r6_mul(ep_3, esat))), (float)(1.e-7f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7505
xl = f_xl_blend(tk1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7506
qsat_tk = f_qsat_blend(tk1(k), pres1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7508
rsl = r6_div(r6_mul(xl, qsat_tk), r6_mul(r_v, mynn_ipow<2>(tk1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7510
cpm = r6_add(cp, r6_mul(qt1(k), cpv));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7511
a = r6_div(one, r6_add(one, r6_div(r6_mul(xl, rsl), cpm)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7512
b9 = r6_mul(a, rsl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7514
q2p = r6_div(xlvcp, ex1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7515
pt = r6_add(thl1(k), r6_mul(r6_mul(q2p, qcp), aup));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7516
bb = r6_div(r6_mul(b9, tk1(k)), pt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7522
qww = r6_add(one, r6_mul(p608, qt1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7523
alpha = r6_mul(p608, pt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7524
beta = r6_sub(r6_div(r6_mul(pt, xl), r6_mul(tk1(k), cp)), r6_mul(1.61f, pt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7528
if (((a) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7529
f = mynn_min((float)(r6_div(one, a)), (float)(four));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7530
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7531
f = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7532
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7542
tauc = 1800.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7543
sigq = r6_mul(r6_mul(r6_mul(five, aup), edmf_w1(k)), mynn_max((float)(r6_mul(tauc, mynn_max((float)(zero), (float)(dqwdz)))), (float)(4e-3f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7548
sigq = mynn_max((float)(sigq), (float)(r6_mul(r6_mul(three, aup), r6_sub(qtp, qt1(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7549
sigq = mynn_min((float)(sigq), (float)(r6_mul(qsat_tk, p666)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7554
qmq = r6_sub(qt1(k), qsat_tk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7555
if (((qmq) > (zero))) { sigq = mynn_min((float)(r6_mul(qsat_tk, 0.015f)), (float)(sigq)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7556
q1 = r6_div(qmq, sigq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7558
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7563
cf_mf = mynn_min((float)(mynn_max((float)(r6_add(p5, r6_mul(0.36f, mynn_atan(r6_mul(1.8f, r6_add(q1, p2)))))), (float)(0.01f))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7564
cf_mf = mynn_max((float)(cf_mf), (float)(r6_mul(1.2f, aup)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7566
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7571
cf_mf = mynn_min((float)(mynn_max((float)(r6_add(p5, r6_mul(0.36f, mynn_atan(r6_mul(1.8f, r6_add(q1, p2)))))), (float)(0.01f))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7572
cf_mf = mynn_max((float)(cf_mf), (float)(r6_mul(1.8f, aup)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7574
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7587
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7588
if (((r6_mul(qcp, aup)) > (5e-5f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7589
qc_mf = r6_sub(r6_mul(1.86f, r6_mul(qcp, aup)), 2.2e-5f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7590
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7591
qc_mf = r6_mul(1.20f, r6_mul(qcp, aup));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7592
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7593
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7594
if (((r6_mul(qcp, aup)) > (5e-5f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7595
qc_mf = r6_sub(r6_mul(1.86f, r6_mul(qcp, aup)), 2.2e-5f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7596
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7597
qc_mf = r6_mul(1.20f, r6_mul(qcp, aup));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7598
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7599
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7604
wt2 = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(cf_mf, thirty)), (float)(zero)), r6_sub(hundred, thirty))), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7605
qc_mf_min = r6_add(r6_mul(r6_mul(r6_mul(wt2, qsat_tk), 0.01f), cf_mf), r6_mul(r6_mul(r6_mul(r6_sub(one, wt2), qsat_tk), 0.025f), cf_mf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7606
qc_mf = mynn_max((float)(qc_mf), (float)(qc_mf_min));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7608
cf_strat = cldfra_bl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7609
qc_strat = qc_bl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7610
cldfra_bl1(k) = mynn_max((float)(cf_mf), (float)(cf_strat));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7611
pct_mf = r6_div(cf_mf, cldfra_bl1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7612
qc_bl1(k) = r6_add(r6_mul(qc_mf, pct_mf), r6_mul(r6_sub(one, pct_mf), qc_strat));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7620
q1 = mynn_max((float)(q1), (float)(r6_neg(2.25f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7625
if (((q1) >= (one))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7626
fng = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7627
} else if (((((q1) >= (r6_neg(1.7f)))) && (((q1) < (one))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7628
fng = mynn_exp(r6_neg(r6_mul(p4, r6_sub(q1, one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7629
} else if (((((q1) >= (r6_neg(2.5f)))) && (((q1) < (r6_neg(1.7f)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7630
fng = r6_add(three, mynn_exp(r6_neg(r6_mul(3.8f, r6_add(q1, 1.7f)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7631
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7632
fng = mynn_min((float)(r6_add(23.9f, mynn_exp(r6_neg(r6_mul(1.6f, r6_add(q1, 2.5f)))))), (float)(60.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7633
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7636
vt1(k) = r6_sub(r6_sub(qww, r6_mul(r6_mul(r6_mul(r6_mul(1.5f, aup), beta), bb), fng)), one);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7637
vq1(k) = r6_sub(r6_add(alpha, r6_mul(r6_mul(r6_mul(r6_mul(1.5f, aup), beta), a), fng)), tv0);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7638
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7639
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7641
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7644
if (((ktop) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7645
maxqc = mynn_maxval(edmf_qc1.slice(MynnRange{1, ktop}));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7646
if (((maxqc) < (1.e-8f))) { maxmf = r6_neg(r6_mul(1.f, maxmf)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7647
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7652
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7680
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7681
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7682
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7683
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7684
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7686
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7698-7751
__device__ __noinline__ void f_condensation_edmf(float qt, float thl, float p, float zagl, float& thv, float& qc) {
MynnFrame mynn_frame(arena);
int niter;
int ni;
float diff;
float exn;
float t;
float th;
float qs;
float qcold;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7717
niter = 50;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7719
diff = 1.e-6f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7721
exn = mynn_pow(r6_div(p, p1000mb), rcp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7722
qc = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7723
for (ni=1; (1) > 0 ? ni <= (niter) : ni >= (niter); ni += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7724
t = r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7725
qs = f_qsat_blend(t, p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7726
qcold = qc;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7727
qc = r6_add(r6_mul(p5, qc), r6_mul(p5, mynn_max((float)(zero), (float)(r6_sub(qt, qs)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7728
if (((mynn_abs(r6_sub(qc, qcold))) < (diff))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7729
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7731
t = r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7732
qs = f_qsat_blend(t, p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7733
qc = mynn_max((float)(zero), (float)(r6_sub(qt, qs)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7736
if (((zagl) < (100.f))) { qc = zero; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7739
thv = r6_mul(r6_add(thl, r6_mul(r6_div(xlvcp, exn), qc)), r6_sub(r6_add(one, r6_mul(qt, r6_sub(rvovrd, one))), r6_mul(rvovrd, qc)));
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7754-7825
__device__ __noinline__ void f_condensation_ddmf(float qt, float thl, float p, float zagl, float& thv, float& qc, float& qi, int debug_dd) {
MynnFrame mynn_frame(arena);
int niter;
int ni;
float diff;
float exn;
float t;
float th;
float qs;
float qx;
float qxold;
float frac_ice;
float frac_liq;
float thvin;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7765
niter = 50;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7767
diff = 1.e-6f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7769
if (((r6_add(qi, qc)) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7770
frac_ice = r6_div(qi, r6_add(qi, qc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7771
frac_liq = r6_sub(one, frac_ice);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7772
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7773
frac_ice = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7774
frac_liq = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7775
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7777
if (((debug_dd) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7778
thvin = thv;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7779
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7780
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7781
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7783
exn = mynn_pow(r6_div(p, p1000mb), rcp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7784
for (ni=1; (1) > 0 ? ni <= (niter) : ni >= (niter); ni += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7785
t = r6_add(r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc)), r6_mul(r6_div(xlscp, exn), qi));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7786
qs = f_qsat_blend(t, p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7787
qxold = r6_add(qc, qi);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7788
qx = r6_add(qc, qi);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7789
qx = r6_add(r6_mul(p5, qx), r6_mul(p5, mynn_max((float)(zero), (float)(r6_sub(qt, qs)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7790
qc = r6_mul(frac_liq, qx);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7791
qi = r6_mul(frac_ice, qx);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7792
if (((mynn_abs(r6_sub(qx, qxold))) < (diff))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7793
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7795
t = r6_add(r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc)), r6_mul(r6_div(xlscp, exn), qi));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7796
qs = f_qsat_blend(t, p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7797
qx = mynn_max((float)(zero), (float)(r6_sub(qt, qs)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7798
qc = r6_mul(frac_liq, qx);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7799
qi = r6_mul(frac_ice, qx);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7804
thv = r6_mul(r6_add(r6_add(thl, r6_mul(r6_div(xlvcp, exn), qc)), r6_mul(r6_div(xlscp, exn), qi)), r6_sub(r6_add(one, r6_mul(p608, r6_sub(qt, qx))), qx));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7819
if (((debug_dd) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7820
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7821
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7822
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7823
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7829-7862
__device__ __noinline__ void f_condensation_edmf_r(float qt, float& thl, float p, float zagl, float thv, float& qc) {
MynnFrame mynn_frame(arena);
int niter;
int ni;
float diff;
float exn;
float t;
float th;
float qs;
float qcold;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7841
niter = 50;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7843
diff = 2.e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7845
exn = mynn_pow(r6_div(p, p1000mb), rcp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7847
t = r6_mul(thv, exn);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7851
qc = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7853
for (ni=1; (1) > 0 ? ni <= (niter) : ni >= (niter); ni += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7854
qcold = qc;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7855
t = r6_div(r6_mul(exn, thv), r6_sub(r6_add(1.f, r6_mul(qt, r6_sub(rvovrd, 1.f))), r6_mul(rvovrd, qc)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7856
qs = f_qsat_blend(t, p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7857
qc = mynn_max((float)(0.f), (float)(r6_sub(qt, qs)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7858
if (((mynn_abs(r6_sub(qc, qcold))) < (diff))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7859
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7860
thl = r6_div(r6_sub(t, r6_mul(r6_div(xlv, cp), qc)), exn);
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7883-8386
__device__ __noinline__ void f_ddmp_mf(int kts, int kte, float dt, float dx, MynnArray<float, 1> zw, MynnArray<float, 1> dz, MynnArray<float, 1> p, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> th, MynnArray<float, 1> thl, MynnArray<float, 1> thv, MynnArray<float, 1> tk, MynnArray<float, 1> qt, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qnc, MynnArray<float, 1> qni, MynnArray<float, 1> qnwfa, MynnArray<float, 1> qnifa, MynnArray<float, 1> qke, MynnArray<float, 1> rho, MynnArray<float, 1> exner, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> qi_bl1, MynnArray<float, 1> cldfra_bl1, float ust, float flt, float flq, float fltv, float pblh, int kpbl, MynnArray<float, 1> edmf_a_dd, MynnArray<float, 1> edmf_w_dd, MynnArray<float, 1> edmf_qt_dd, MynnArray<float, 1> edmf_thl_dd, MynnArray<float, 1> edmf_ent_dd, MynnArray<float, 1> edmf_qc_dd, MynnArray<float, 1> sd_aw, MynnArray<float, 1> sd_awthl, MynnArray<float, 1> sd_awqt, MynnArray<float, 1> sd_awqv, MynnArray<float, 1> sd_awqc, MynnArray<float, 1> sd_awqi, MynnArray<float, 1> sd_awqnc, MynnArray<float, 1> sd_awqni, MynnArray<float, 1> sd_awqnwfa, MynnArray<float, 1> sd_awqnifa, MynnArray<float, 1> sd_awu, MynnArray<float, 1> sd_awv, MynnArray<float, 1> sd_awqke, MynnArray<float, 1> tkeprod_dn, MynnArray<float, 1> el, MynnArray<float, 1> rthraten) {
MynnFrame mynn_frame(arena);
zw.rebase({kts});
dz.rebase({kts});
p.rebase({kts});
u.rebase({kts});
v.rebase({kts});
th.rebase({kts});
thl.rebase({kts});
thv.rebase({kts});
tk.rebase({kts});
qt.rebase({kts});
qv.rebase({kts});
qc.rebase({kts});
qi.rebase({kts});
qnc.rebase({kts});
qni.rebase({kts});
qnwfa.rebase({kts});
qnifa.rebase({kts});
qke.rebase({kts});
rho.rebase({kts});
exner.rebase({kts});
qc_bl1.rebase({kts});
qi_bl1.rebase({kts});
cldfra_bl1.rebase({kts});
edmf_a_dd.rebase({kts});
edmf_w_dd.rebase({kts});
edmf_qt_dd.rebase({kts});
edmf_thl_dd.rebase({kts});
edmf_ent_dd.rebase({kts});
edmf_qc_dd.rebase({kts});
sd_aw.rebase({kts});
sd_awthl.rebase({kts});
sd_awqt.rebase({kts});
sd_awqv.rebase({kts});
sd_awqc.rebase({kts});
sd_awqi.rebase({kts});
sd_awqnc.rebase({kts});
sd_awqni.rebase({kts});
sd_awqnwfa.rebase({kts});
sd_awqnifa.rebase({kts});
sd_awu.rebase({kts});
sd_awv.rebase({kts});
sd_awqke.rebase({kts});
tkeprod_dn.rebase({kts});
el.rebase({kts});
rthraten.rebase({kts});
MynnArray<float, 1> sd_aw2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
constexpr int ndd = 5;
constexpr int kmin = 3;
constexpr float minddd = 50.f;
constexpr float maxddd = 500.f;
constexpr float zmin = 50.f;
constexpr float dz200 = 200.f;
constexpr float ct_cooling = (-(0.00011f));
float maxdd2;
float ddd;
float dl;
float add;
float minexc;
float maxexc;
MynnArray<int, 1> dd_initk = mynn_array<int, 1, mynn_tile>(arena, {1}, {ndd});
MynnArray<float, 2> downw = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downthl = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downqt = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downqc = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downa = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downu = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downv = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downthv = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downqi = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downqnc = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downqni = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downqnwfa = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> downqnifa = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
MynnArray<float, 2> ent = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), ndd});
int k;
int i;
int ki;
int qltop;
int qlbase;
float qstar;
float thstar;
float sigmaw;
float sigmaqt;
float sigmath;
float z0;
float pwmin;
float pwmax;
float wmin;
float wmax;
float went;
float mindownw;
float qtn;
float thln;
float thvn;
float qcn;
float un;
float vn;
float qken;
float wn2;
float wn;
float qin;
float qncn;
float qnin;
float qnwfan;
float qnifan;
float thvk;
float pk;
float entexp;
float entw;
float beta_dm;
float entexm;
float rho_int;
float jump_thv;
float jump_qt;
float jump_thetal;
float refthl;
float refthv;
float refthlv;
float refqt;
float refqc;
float refqi;
float refqnc;
float refqni;
float refqnwfa;
float refqnifa;
float refu;
float refv;
float refqke;
float refqt2;
float reftk;
float refp;
float qx_k;
float qx_km1;
float refthvm1;
float cftop;
float crate;
float ac_wsp;
float wspd_pbl;
float radflux;
float f0;
float wstar_rad;
float dz_ent;
bool cloudflg;
bool singlelayer;
float sigq;
float xl;
float rsl;
float cpm;
float a;
float diffqt;
float fng;
float qww;
float alpha;
float beta;
float bb;
float f;
float pt;
float t;
float q2p;
float b9;
float satvp;
float rhgrid;
float def_th;
float def_qt;
float frac_liq;
float frac_ice;
float qs1;
float qs2;
constexpr float wa = 1.f;
constexpr float wb = 1.5f;
constexpr float z00 = 100.f;
float buoy;
float bcoeff;
float ecoeff;
constexpr int debug_dd = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7974
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downw).size(); mynn_element += lane_width) { downw.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7975
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downthl).size(); mynn_element += lane_width) { downthl.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7976
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downthv).size(); mynn_element += lane_width) { downthv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7977
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqt).size(); mynn_element += lane_width) { downqt.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7978
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downa).size(); mynn_element += lane_width) { downa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7979
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downu).size(); mynn_element += lane_width) { downu.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7980
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downv).size(); mynn_element += lane_width) { downv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7981
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqc).size(); mynn_element += lane_width) { downqc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7982
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqi).size(); mynn_element += lane_width) { downqi.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7983
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqnc).size(); mynn_element += lane_width) { downqnc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7984
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqni).size(); mynn_element += lane_width) { downqni.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7985
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqnwfa).size(); mynn_element += lane_width) { downqnwfa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7986
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqnifa).size(); mynn_element += lane_width) { downqnifa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7987
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (ent).size(); mynn_element += lane_width) { ent.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7988
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprod_dn).size(); mynn_element += lane_width) { tkeprod_dn.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7989
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dd_initk).size(); mynn_element += lane_width) { dd_initk.linear(mynn_element) = 0; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7991
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a_dd).size(); mynn_element += lane_width) { edmf_a_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7992
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w_dd).size(); mynn_element += lane_width) { edmf_w_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7993
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt_dd).size(); mynn_element += lane_width) { edmf_qt_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7994
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl_dd).size(); mynn_element += lane_width) { edmf_thl_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7995
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent_dd).size(); mynn_element += lane_width) { edmf_ent_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7996
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc_dd).size(); mynn_element += lane_width) { edmf_qc_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7998
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_aw).size(); mynn_element += lane_width) { sd_aw.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7999
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awthl).size(); mynn_element += lane_width) { sd_awthl.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8000
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqt).size(); mynn_element += lane_width) { sd_awqt.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8001
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqv).size(); mynn_element += lane_width) { sd_awqv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8002
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqc).size(); mynn_element += lane_width) { sd_awqc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8003
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqi).size(); mynn_element += lane_width) { sd_awqi.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8004
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnc).size(); mynn_element += lane_width) { sd_awqnc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8005
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqni).size(); mynn_element += lane_width) { sd_awqni.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8006
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnwfa).size(); mynn_element += lane_width) { sd_awqnwfa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8007
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnifa).size(); mynn_element += lane_width) { sd_awqnifa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8008
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awu).size(); mynn_element += lane_width) { sd_awu.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8009
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awv).size(); mynn_element += lane_width) { sd_awv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8010
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqke).size(); mynn_element += lane_width) { sd_awqke.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8013
cloudflg = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8014
singlelayer = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8015
qltop = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8016
qlbase = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8017
crate = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8018
for (k=mynn_max((int)(((kmin) + (1))), (int)(((kpbl) - (5)))); (1) > 0 ? k <= (mynn_min((int)(((kpbl) + (20))), (int)(((kte) - (1))))) : k >= (mynn_min((int)(((kpbl) + (20))), (int)(((kte) - (1))))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8020
if (((((cldfra_bl1(((k) - (1)))) > (0.5f))) && (((rthraten(((k) - (1)))) < (ct_cooling))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8023
cloudflg = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8024
qltop = ((k) - (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8025
cftop = cldfra_bl1(((k) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8026
crate = mynn_min((float)(crate), (float)(rthraten(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8027
if (((cldfra_bl1(((k) - (2)))) < (0.1f))) { singlelayer = true; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8028
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8029
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8032
maxdd2 = mynn_min((float)(r6_mul(1.2f, dx)), (float)(maxddd));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8033
maxdd2 = mynn_min((float)(maxdd2), (float)(r6_sub(r6_neg(r6_mul(1212121.f, crate)), 30.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8034
maxdd2 = mynn_min((float)(maxdd2), (float)(r6_mul(p5, zw(((qltop) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8035
dl = r6_div(mynn_max((float)(zero), (float)(r6_sub(maxdd2, minddd))), ((float)(((ndd) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8038
if (((cloudflg) && (((dl) > (zero))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8039
for (k=qltop; ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8040
qx_k = mynn_max((float)(qt(k)), (float)(r6_add(qc_bl1(k), qi_bl1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8041
if (((qx_k) > (1e-6f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8042
qlbase = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8043
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8044
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8046
{ const int mynn_lo0 = (1), mynn_hi0 = (ndd), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { i = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8048
dd_initk(i) = qltop;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8049
} mynn_lane_sync(); i = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8052
f0 = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8053
for (k=kmin; (1) > 0 ? k <= (qltop) : k >= (qltop); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8054
radflux = r6_mul(rthraten(k), exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8055
radflux = r6_mul(r6_div(r6_mul(radflux, cp), grav), r6_sub(p(k), p(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8056
if (((radflux) < (zero))) { f0 = r6_add(mynn_abs(radflux), f0); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8057
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8058
f0 = mynn_min((float)(mynn_max((float)(f0), (float)(one))), (float)(200.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8065
add = mynn_min((float)(r6_mul(f0, 0.002f)), (float)(0.3f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8067
wspd_pbl = mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(u(kts)), mynn_ipow<2>(v(kts)))), (float)(0.01f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8068
if (((wspd_pbl) <= (10.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8069
ac_wsp = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8070
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8071
ac_wsp = r6_sub(one, mynn_min((float)(r6_div(r6_sub(wspd_pbl, 10.0f), 15.f)), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8072
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8073
add = r6_mul(add, ac_wsp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8076
dz_ent = r6_mul(p5, r6_add(dz(((qltop) + (1))), dz(qltop)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8077
jump_thv = r6_mul(r6_div(r6_sub(thv(((qltop) + (1))), thv(qltop)), dz_ent), dz200);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8078
jump_qt = r6_mul(r6_div(r6_sub(qt(((qltop) + (1))), qt(qltop)), dz_ent), dz200);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8079
jump_thetal = r6_mul(r6_div(r6_sub(thl(((qltop) + (1))), thl(qltop)), dz_ent), dz200);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8081
ki = qltop;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8082
if (singlelayer) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8083
refthl = thl(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8084
refthlv = r6_mul(refthl, r6_add(1.f, r6_mul(p608, qv(ki))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8085
refthv = thv(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8086
refthvm1 = thv(((ki) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8087
reftk = tk(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8088
refqt = qt(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8089
refqc = qc(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8090
refqi = qi(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8091
refqnc = qnc(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8092
refqni = qni(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8093
refqnwfa = qnwfa(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8094
refqnifa = qnifa(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8095
refu = u(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8096
refv = v(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8097
refqke = qke(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8098
refp = p(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8099
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8100
refthl = r6_div(r6_add(r6_mul(thl(((ki) - (1))), dz(ki)), r6_mul(thl(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8101
refthlv = r6_mul(refthl, r6_add(1.f, r6_div(r6_mul(p608, r6_add(r6_mul(qv(((ki) - (1))), dz(ki)), r6_mul(qv(ki), dz(((ki) - (1)))))), r6_add(dz(ki), dz(((ki) - (1)))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8102
refthv = r6_div(r6_add(r6_mul(thv(((ki) - (1))), dz(ki)), r6_mul(thv(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8103
refthvm1 = r6_div(r6_add(r6_mul(thv(((ki) - (2))), dz(((ki) - (1)))), r6_mul(thv(((ki) - (1))), dz(((ki) - (2))))), r6_add(dz(((ki) - (1))), dz(((ki) - (2)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8104
reftk = r6_div(r6_add(r6_mul(tk(((ki) - (1))), dz(ki)), r6_mul(tk(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8105
refqt = r6_div(r6_add(r6_mul(qt(((ki) - (1))), dz(ki)), r6_mul(qt(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8106
refqc = r6_div(r6_add(r6_mul(qc(((ki) - (1))), dz(ki)), r6_mul(qc(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8107
refqi = r6_div(r6_add(r6_mul(qi(((ki) - (1))), dz(ki)), r6_mul(qi(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8108
refqnc = r6_div(r6_add(r6_mul(qnc(((ki) - (1))), dz(ki)), r6_mul(qnc(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8109
refqni = r6_div(r6_add(r6_mul(qni(((ki) - (1))), dz(ki)), r6_mul(qni(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8110
refqnwfa = r6_div(r6_add(r6_mul(qnwfa(((ki) - (1))), dz(ki)), r6_mul(qnwfa(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8111
refqnifa = r6_div(r6_add(r6_mul(qnifa(((ki) - (1))), dz(ki)), r6_mul(qnifa(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8112
refu = r6_div(r6_add(r6_mul(u(((ki) - (1))), dz(ki)), r6_mul(u(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8113
refv = r6_div(r6_add(r6_mul(v(((ki) - (1))), dz(ki)), r6_mul(v(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8114
refqke = r6_div(r6_add(r6_mul(qke(((ki) - (1))), dz(ki)), r6_mul(qke(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8115
refp = r6_div(r6_add(r6_mul(p(((ki) - (1))), dz(ki)), r6_mul(p(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8116
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8120
wstar_rad = r6_mul(1.25f, mynn_pow(r6_div(r6_mul(r6_mul(grav, dz200), f0), r6_mul(r6_mul(refthl, rho(ki)), cp)), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8121
wstar_rad = mynn_min((float)(mynn_max((float)(wstar_rad), (float)(p1))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8124
went = r6_mul(r6_div(thv(1), r6_mul(r6_mul(grav, jump_thv), dz200)), r6_mul(p5, mynn_ipow<3>(wstar_rad)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8126
qstar = mynn_abs(r6_div(r6_mul(went, jump_qt), wstar_rad));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8127
thstar = r6_sub(r6_div(r6_div(r6_div(f0, rho(ki)), cp), wstar_rad), r6_div(r6_mul(went, jump_thv), wstar_rad));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8129
sigmaw = r6_mul(0.2f, wstar_rad);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8130
sigmaqt = r6_mul(40.f, qstar);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8131
sigmath = r6_mul(1.0f, thstar);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8133
pwmin = r6_neg(1.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8134
pwmax = r6_neg(3.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8135
wmin = mynn_max((float)(mynn_min((float)(r6_mul(sigmaw, pwmin)), (float)(r6_neg(0.1f)))), (float)(r6_neg(0.2f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8136
wmax = mynn_max((float)(mynn_min((float)(r6_mul(sigmaw, pwmax)), (float)(r6_neg(0.2f)))), (float)(r6_neg(0.5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8139
minexc = mynn_min((float)(r6_neg(0.05f)), (float)(r6_mul(0.5f, r6_sub(refthvm1, refthv))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8140
maxexc = mynn_min((float)(r6_sub(minexc, 0.1f)), (float)(r6_neg(0.5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8141
maxexc = mynn_max((float)(maxexc), (float)(r6_sub(thv(1), refthlv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8142
def_th = mynn_max((float)(mynn_min((float)(r6_div(r6_mul(r6_mul(0.05f, r6_neg(0.3f)), sigmath), sigmaw)), (float)(minexc))), (float)(maxexc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8145
qs1 = f_qsat_blend(reftk, refp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8146
qs2 = f_qsat_blend(r6_add(reftk, def_th), refp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8147
refqt2 = r6_div(r6_mul(refqt, qs2), qs1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8149
def_qt = mynn_min((float)(r6_sub(qs2, qs1)), (float)(r6_neg(r6_mul(0.33f, r6_add(refqc, refqi)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8152
if (((r6_add(refqc, refqi)) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8153
frac_liq = r6_div(refqc, r6_add(refqc, refqi));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8154
frac_ice = r6_sub(one, frac_liq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8155
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8156
frac_liq = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8157
frac_ice = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8158
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8160
if (((debug_dd) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8161
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8162
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8163
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8164
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8165
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8166
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8167
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8168
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8169
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8173
{ const int mynn_lo1 = (1), mynn_hi1 = (ndd), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { i = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8174
ki = dd_initk(i);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8177
downw(ki, i) = r6_sub(wmin, r6_mul(r6_div(r6_sub(mynn_abs(wmax), mynn_abs(wmin)), ((float)(((ndd) - (1))))), ((i) - (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8181
downa(ki, i) = r6_div(add, ((float)(ndd)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8182
downu(ki, i) = refu;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8183
downv(ki, i) = refv;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8184
downqc(ki, i) = mynn_max((float)(zero), (float)(r6_add(refqc, r6_mul(def_qt, frac_liq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8185
downqi(ki, i) = mynn_max((float)(zero), (float)(r6_add(refqi, r6_mul(def_qt, frac_ice))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8186
downqnc(ki, i) = refqnc;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8187
downqni(ki, i) = refqni;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8188
downqnwfa(ki, i) = refqnwfa;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8189
downqnifa(ki, i) = refqnifa;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8190
downqt(ki, i) = r6_add(refqt, def_qt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8191
downthv(ki, i) = r6_add(refthv, r6_div(r6_mul(def_th, ((float)(i))), ((float)(ndd))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8192
downthl(ki, i) = r6_add(refthl, r6_div(r6_mul(def_th, ((float)(i))), ((float)(ndd))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8199
} mynn_lane_sync(); i = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; ki = mynn_lane_bcast(ki, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8201
if (((debug_dd) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8202
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8203
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8204
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8205
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8206
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8207
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8208
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8210
for (i=1; (1) > 0 ? i <= (ndd) : i >= (ndd); i += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8211
ddd = r6_add(minddd, r6_mul(dl, ((float)(((i) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8213
thvn = downthv(ki, i);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8214
qcn = downqc(ki, i);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8215
qin = downqi(ki, i);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8216
for (k=((dd_initk(i)) - (1)); ((-(1))) > 0 ? k <= (((kts) + (1))) : k >= (((kts) + (1))); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8218
wmin = r6_add(p1, r6_mul(ddd, 0.0005f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8221
ent(k, i) = r6_div(0.26f, r6_mul(mynn_min((float)(mynn_max((float)(mynn_abs(downw(((k) + (1)), i))), (float)(wmin))), (float)(0.6f)), ddd));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8224
ent(k, i) = mynn_max((float)(ent(k, i)), (float)(0.0003f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8225
ent(k, i) = mynn_max((float)(ent(k, i)), (float)(r6_div(0.06f, zw(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8226
ent(k, i) = mynn_min((float)(ent(k, i)), (float)(r6_div(0.9f, dz(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8231
entexp = r6_mul(ent(k, i), dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8232
entexm = r6_mul(r6_mul(ent(k, i), dz(k)), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8234
qtn = r6_add(r6_mul(downqt(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qt(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8235
thln = r6_add(r6_mul(downthl(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(thl(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8236
qncn = r6_add(r6_mul(downqnc(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qnc(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8237
qnin = r6_add(r6_mul(downqni(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qni(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8238
qnwfan = r6_add(r6_mul(downqnwfa(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qnwfa(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8239
qnifan = r6_add(r6_mul(downqnifa(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qnifa(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8240
un = r6_add(r6_mul(downu(((k) + (1)), i), r6_sub(one, entexm)), r6_mul(u(k), entexm));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8241
vn = r6_add(r6_mul(downv(((k) + (1)), i), r6_sub(one, entexm)), r6_mul(v(k), entexm));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8249
pk = r6_div(r6_add(r6_mul(p(((k) - (1))), dz(k)), r6_mul(p(k), dz(((k) - (1))))), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8250
f_condensation_ddmf(qtn, thln, pk, zw(k), thvn, qcn, qin, debug_dd);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8253
thvk = r6_div(r6_add(r6_mul(thv(((k) - (1))), dz(k)), r6_mul(thv(k), dz(((k) - (1))))), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8254
buoy = r6_mul(grav, r6_sub(r6_div(thvn, thvk), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8256
if (((buoy) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8257
bcoeff = 0.2f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8258
ecoeff = r6_neg(2.0f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8259
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8260
bcoeff = r6_mul(0.2f, mynn_min((float)(r6_div(zw(k), pblh)), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8261
ecoeff = r6_sub(r6_neg(2.0f), r6_div(mynn_max((float)(zero), (float)(r6_sub(r6_mul(0.5f, pblh), zw(k)))), r6_mul(0.5f, pblh)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8262
buoy = r6_mul(buoy, mynn_min((float)(r6_div(zw(k), pblh)), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8263
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8264
if (((k) == (((kts) + (2))))) { buoy = mynn_max((float)(buoy), (float)(0.01f)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8265
if (((k) == (((kts) + (1))))) { buoy = mynn_max((float)(buoy), (float)(0.05f)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8277
mindownw = mynn_min((float)(downw(((k) + (1)), i)), (float)(r6_neg(0.2f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8278
wn = r6_add(downw(((k) + (1)), i), r6_mul(r6_sub(r6_mul(r6_mul(ecoeff, ent(k, i)), downw(((k) + (1)), i)), r6_div(r6_mul(bcoeff, buoy), mindownw)), mynn_min((float)(dz(k)), (float)(250.f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8283
if (((wn) < (r6_sub(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8284
wn = r6_sub(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8285
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8287
if (((wn) > (r6_add(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8288
wn = r6_add(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8289
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8290
wn = mynn_max((float)(mynn_min((float)(wn), (float)(zero))), (float)(r6_neg(3.0f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8292
if (((debug_dd) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8293
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8294
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8295
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8296
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8297
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8298
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8299
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8300
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8301
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8303
if (((wn) < (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8304
downw(k, i) = wn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8305
downthv(k, i) = thvn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8306
downthl(k, i) = thln;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8307
downqt(k, i) = qtn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8308
downqc(k, i) = qcn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8309
downqi(k, i) = qin;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8310
downqnc(k, i) = qncn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8311
downqni(k, i) = qnin;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8312
downqnwfa(k, i) = qnwfan;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8313
downqnifa(k, i) = qnifan;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8314
downu(k, i) = un;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8315
downv(k, i) = vn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8316
downa(k, i) = downa(((k) + (1)), i);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8317
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8328
break;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8329
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8330
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8331
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8333
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downw.slice(1, MynnRange{downw.lo[1], downw.hi[1]})).size(); mynn_element += lane_width) { downw.slice(1, MynnRange{downw.lo[1], downw.hi[1]}).linear(mynn_element) = 0.f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8334
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downa.slice(1, MynnRange{downa.lo[1], downa.hi[1]})).size(); mynn_element += lane_width) { downa.slice(1, MynnRange{downa.lo[1], downa.hi[1]}).linear(mynn_element) = 0.f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8339
{ const int mynn_lo2 = (qltop), mynn_hi2 = (kts), mynn_st2 = ((-(1))); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8340
for (i=1; (1) > 0 ? i <= (ndd) : i >= (ndd); i += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8341
edmf_a_dd(k) = r6_add(edmf_a_dd(k), downa(k, i));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8342
edmf_w_dd(k) = r6_add(edmf_w_dd(k), r6_mul(downa(k, i), downw(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8343
edmf_qt_dd(k) = r6_add(edmf_qt_dd(k), r6_mul(downa(k, i), downqt(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8344
edmf_thl_dd(k) = r6_add(edmf_thl_dd(k), r6_mul(downa(k, i), downthl(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8345
edmf_ent_dd(k) = r6_add(edmf_ent_dd(k), r6_mul(downa(k, i), ent(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8346
edmf_qc_dd(k) = r6_add(edmf_qc_dd(k), r6_mul(downa(k, i), downqc(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8347
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8349
if (((edmf_a_dd(k)) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8350
edmf_w_dd(k) = r6_div(edmf_w_dd(k), edmf_a_dd(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8351
edmf_qt_dd(k) = r6_div(edmf_qt_dd(k), edmf_a_dd(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8352
edmf_thl_dd(k) = r6_div(edmf_thl_dd(k), edmf_a_dd(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8353
edmf_ent_dd(k) = r6_div(edmf_ent_dd(k), edmf_a_dd(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8354
edmf_qc_dd(k) = r6_div(edmf_qc_dd(k), edmf_a_dd(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8355
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8357
tkeprod_dn(k) = r6_div(r6_mul(mynn_ipow<3>(mynn_abs(edmf_w_dd(k))), edmf_a_dd(k)), r6_mul(b1, mynn_max((float)(el(k)), (float)(0.2f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8358
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; i = mynn_lane_bcast(i, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8361
tkeprod_dn(((qltop) + (1))) = r6_div(r6_mul(went, edmf_a_dd(qltop)), r6_mul(b1, mynn_max((float)(el(((qltop) + (1)))), (float)(0.1f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8366
{ const int mynn_lo3 = (kts), mynn_hi3 = (qltop), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8367
rho_int = r6_div(r6_add(r6_mul(rho(k), dz(((k) + (1)))), r6_mul(rho(((k) + (1))), dz(k))), r6_add(dz(((k) + (1))), dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8368
for (i=1; (1) > 0 ? i <= (ndd) : i >= (ndd); i += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8369
sd_aw(k) = r6_add(sd_aw(k), r6_mul(r6_mul(rho_int, downa(k, i)), mynn_abs(downw(k, i))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8370
sd_awthl(k) = r6_add(sd_awthl(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downthl(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8371
sd_awqt(k) = r6_add(sd_awqt(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqt(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8372
sd_awqc(k) = r6_add(sd_awqc(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqc(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8373
sd_awqi(k) = r6_add(sd_awqi(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqi(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8374
sd_awqnc(k) = r6_add(sd_awqnc(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqnc(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8375
sd_awqni(k) = r6_add(sd_awqni(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqni(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8376
sd_awqnwfa(k) = r6_add(sd_awqnwfa(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqnwfa(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8377
sd_awqnifa(k) = r6_add(sd_awqnifa(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqnifa(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8378
sd_awu(k) = r6_add(sd_awu(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downu(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8379
sd_awv(k) = r6_add(sd_awv(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downv(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8380
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8381
sd_awqv(k) = r6_sub(r6_sub(sd_awqt(k), sd_awqc(k)), sd_awqi(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8382
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; i = mynn_lane_bcast(i, mynn_src3); rho_int = mynn_lane_bcast(rho_int, mynn_src3); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8384
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8389-8460
__device__ __noinline__ void f_scale_aware(float dx, float pblh, float& psig_bl, float& psig_shcu) {
MynnFrame mynn_frame(arena);
float dxdh;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8404
psig_bl = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8405
psig_shcu = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8406
dxdh = r6_div(mynn_max((float)(r6_mul(2.5f, dx)), (float)(10.f)), mynn_min((float)(pblh), (float)(3000.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8414
psig_bl = r6_div(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.106f, mynn_pow(dxdh, 0.667f))), r6_add(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.066f, mynn_pow(dxdh, 0.667f))), 0.071f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8417
dxdh = r6_div(mynn_max((float)(r6_mul(2.5f, dx)), (float)(10.f)), mynn_min((float)(r6_add(pblh, 500.f)), (float)(3500.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8444
psig_shcu = r6_div(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.145f, mynn_pow(dxdh, 0.667f))), r6_add(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.172f, mynn_pow(dxdh, 0.667f))), 0.170f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8454
if (((psig_bl) > (one))) { psig_bl = one; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8455
if (((psig_bl) < (zero))) { psig_bl = zero; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8457
if (((psig_shcu) > (one))) { psig_shcu = one; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8458
if (((psig_shcu) < (zero))) { psig_shcu = zero; }
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8470-8513
__device__ __noinline__ float f_esat_blend(float t) {
MynnFrame mynn_frame(arena);
float esat_blend;
float xc;
float esl;
float esi;
float chi;
constexpr float j0 = .611583699e03f;
constexpr float j1 = .444606896e02f;
constexpr float j2 = .143177157e01f;
constexpr float j3 = .264224321e-1f;
constexpr float j4 = .299291081e-3f;
constexpr float j5 = .203154182e-5f;
constexpr float j6 = .702620698e-8f;
constexpr float j7 = .379534310e-11f;
constexpr float j8 = (-(.321582393e-13f));
constexpr float k0 = .609868993e03f;
constexpr float k1 = .499320233e02f;
constexpr float k2 = .184672631e01f;
constexpr float k3 = .402737184e-1f;
constexpr float k4 = .565392987e-3f;
constexpr float k5 = .521693933e-5f;
constexpr float k6 = .307839583e-7f;
constexpr float k7 = .105785160e-9f;
constexpr float k8 = .161444444e-12f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8497
xc = mynn_max((float)(r6_neg(80.f)), (float)(r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8502
if (((t) >= (r6_sub(t0c, 6.f)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8503
esat_blend = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8504
} else if (((t) <= (tice))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8505
esat_blend = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8506
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8507
esl = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8508
esi = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8509
chi = r6_div(r6_sub(r6_sub(t0c, 6.f), t), r6_sub(r6_sub(t0c, 6.f), tice));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8510
esat_blend = r6_add(r6_mul(r6_sub(1.f, chi), esl), r6_mul(chi, esi));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8511
}
return esat_blend;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8521-8570
__device__ __noinline__ float f_qsat_blend(float t, float p) {
MynnFrame mynn_frame(arena);
float qsat_blend;
float xc;
float esl;
float esi;
float rslf;
float rsif;
float chi;
constexpr float j0 = .611583699e03f;
constexpr float j1 = .444606896e02f;
constexpr float j2 = .143177157e01f;
constexpr float j3 = .264224321e-1f;
constexpr float j4 = .299291081e-3f;
constexpr float j5 = .203154182e-5f;
constexpr float j6 = .702620698e-8f;
constexpr float j7 = .379534310e-11f;
constexpr float j8 = (-(.321582393e-13f));
constexpr float k0 = .609868993e03f;
constexpr float k1 = .499320233e02f;
constexpr float k2 = .184672631e01f;
constexpr float k3 = .402737184e-1f;
constexpr float k4 = .565392987e-3f;
constexpr float k5 = .521693933e-5f;
constexpr float k6 = .307839583e-7f;
constexpr float k7 = .105785160e-9f;
constexpr float k8 = .161444444e-12f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8548
xc = mynn_max((float)(r6_neg(80.f)), (float)(r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8550
if (((t) >= (r6_sub(t0c, 6.f)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8551
esl = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8552
esl = mynn_min((float)(esl), (float)(r6_mul(p, 0.15f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8553
qsat_blend = r6_div(r6_mul(0.622f, esl), mynn_max((float)(r6_sub(p, esl)), (float)(1e-5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8554
} else if (((t) <= (tice))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8555
esi = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8556
esi = mynn_min((float)(esi), (float)(r6_mul(p, 0.15f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8557
qsat_blend = r6_div(r6_mul(0.622f, esi), mynn_max((float)(r6_sub(p, esi)), (float)(1e-5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8558
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8559
esl = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8560
esl = mynn_min((float)(esl), (float)(r6_mul(p, 0.15f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8561
esi = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8562
esi = mynn_min((float)(esi), (float)(r6_mul(p, 0.15f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8563
rslf = r6_div(r6_mul(0.622f, esl), mynn_max((float)(r6_sub(p, esl)), (float)(1e-5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8564
rsif = r6_div(r6_mul(0.622f, esi), mynn_max((float)(r6_sub(p, esi)), (float)(1e-5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8566
chi = r6_div(r6_sub(r6_sub(t0c, 6.f), t), r6_sub(r6_sub(t0c, 6.f), tice));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8567
qsat_blend = r6_add(r6_mul(r6_sub(1.f, chi), rslf), r6_mul(chi, rsif));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8568
}
return qsat_blend;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8579-8599
__device__ __noinline__ float f_xl_blend(float t) {
MynnFrame mynn_frame(arena);
float xl_blend;
float xlvt;
float xlst;
float chi;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8587
if (((t) >= (t0c))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8588
xl_blend = r6_add(xlv, r6_mul(r6_sub(cpv, cliq), r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8589
} else if (((t) <= (tice))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8590
xl_blend = r6_add(xls, r6_mul(r6_sub(cpv, cice), r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8591
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8592
xlvt = r6_add(xlv, r6_mul(r6_sub(cpv, cliq), r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8593
xlst = r6_add(xls, r6_mul(r6_sub(cpv, cice), r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8595
chi = r6_div(r6_sub(t0c, t), r6_sub(t0c, tice));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8596
xl_blend = r6_add(r6_mul(r6_sub(1.f, chi), xlvt), r6_mul(chi, xlst));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8597
}
return xl_blend;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8603-8652
__device__ __noinline__ float f_phim(float zet) {
MynnFrame mynn_frame(arena);
float dummy_0;
float dummy_1;
float dummy_11;
float dummy_2;
float dummy_22;
float dummy_3;
float dummy_33;
float dummy_4;
float dummy_44;
float dummy_psi;
constexpr float am_st = 6.1f;
constexpr float bm_st = 2.5f;
constexpr float rbm_st = ((1.f) / (bm_st));
constexpr float ah_st = 5.3f;
constexpr float bh_st = 1.1f;
constexpr float rbh_st = ((1.f) / (bh_st));
constexpr float am_unst = 10.f;
constexpr float ah_unst = 34.f;
float phi_m;
float phim;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8619
if (((zet) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8620
dummy_0 = r6_add(one, mynn_pow(zet, bm_st));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8621
dummy_1 = r6_add(zet, mynn_pow(dummy_0, rbm_st));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8622
dummy_11 = r6_add(one, r6_mul(mynn_pow(dummy_0, r6_sub(rbm_st, 1)), mynn_pow(zet, r6_sub(bm_st, one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8623
dummy_2 = r6_mul(r6_neg(r6_div(am_st, dummy_1)), dummy_11);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8624
phi_m = r6_sub(one, r6_mul(zet, dummy_2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8625
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8626
dummy_0 = mynn_pow(r6_sub(one, r6_mul(cphm_unst, zet)), 0.25f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8627
phi_m = r6_div(one, dummy_0);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8628
dummy_psi = r6_add(r6_sub(r6_add(r6_mul(two, mynn_log(r6_mul(p5, r6_add(one, dummy_0)))), mynn_log(r6_mul(p5, r6_add(one, mynn_ipow<2>(dummy_0))))), r6_mul(two, mynn_atan(dummy_0))), 1.570796f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8630
dummy_0 = r6_sub(one, r6_mul(am_unst, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8631
dummy_1 = mynn_pow(dummy_0, p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8632
dummy_11 = r6_mul(r6_neg(r6_mul(p333, am_unst)), mynn_pow(dummy_0, r6_neg(p666)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8633
dummy_2 = r6_mul(p333, r6_add(r6_add(mynn_ipow<2>(dummy_1), dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8634
dummy_22 = r6_mul(r6_mul(p333, dummy_11), r6_add(r6_mul(two, dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8635
dummy_3 = r6_mul(0.57735f, r6_add(r6_mul(two, dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8636
dummy_33 = r6_mul(1.1547f, dummy_11);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8637
dummy_4 = r6_add(r6_sub(r6_mul(1.5f, mynn_log(dummy_2)), r6_mul(1.73205f, mynn_atan(dummy_3))), 1.813799364f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8638
dummy_44 = r6_sub(r6_mul(r6_div(1.5f, dummy_2), dummy_22), r6_div(r6_mul(1.73205f, dummy_33), r6_add(one, mynn_ipow<2>(dummy_3))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8640
dummy_0 = mynn_ipow<2>(zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8641
dummy_1 = r6_div(one, r6_add(one, dummy_0));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8642
dummy_11 = r6_mul(two, zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8643
dummy_2 = r6_mul(r6_add(r6_add(r6_div(r6_sub(one, phi_m), zet), r6_mul(dummy_11, dummy_4)), r6_mul(dummy_0, dummy_44)), dummy_1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8644
dummy_22 = r6_mul(r6_neg(r6_mul(dummy_11, r6_add(dummy_psi, r6_mul(dummy_0, dummy_4)))), mynn_ipow<2>(dummy_1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8646
phi_m = r6_sub(one, r6_mul(zet, r6_add(dummy_2, dummy_22)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8647
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8650
phim = phi_m;
return phim;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8655-8701
__device__ __noinline__ float f_phih(float zet) {
MynnFrame mynn_frame(arena);
float dummy_0;
float dummy_1;
float dummy_11;
float dummy_2;
float dummy_22;
float dummy_3;
float dummy_33;
float dummy_4;
float dummy_44;
float dummy_psi;
constexpr float am_st = 6.1f;
constexpr float bm_st = 2.5f;
constexpr float rbm_st = ((1.f) / (bm_st));
constexpr float ah_st = 5.3f;
constexpr float bh_st = 1.1f;
constexpr float rbh_st = ((1.f) / (bh_st));
constexpr float am_unst = 10.f;
constexpr float ah_unst = 34.f;
float phh;
float phih;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8671
if (((zet) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8672
dummy_0 = r6_add(one, mynn_pow(zet, bh_st));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8673
dummy_1 = r6_add(zet, mynn_pow(dummy_0, rbh_st));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8674
dummy_11 = r6_add(one, r6_mul(mynn_pow(dummy_0, r6_sub(rbh_st, one)), mynn_pow(zet, r6_sub(bh_st, one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8675
dummy_2 = r6_mul(r6_neg(r6_div(ah_st, dummy_1)), dummy_11);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8676
phih = r6_sub(one, r6_mul(zet, dummy_2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8677
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8678
dummy_0 = mynn_pow(r6_sub(one, r6_mul(cphh_unst, zet)), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8679
phh = r6_div(one, dummy_0);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8680
dummy_psi = r6_mul(two, mynn_log(r6_mul(p5, r6_add(one, dummy_0))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8682
dummy_0 = r6_sub(one, r6_mul(ah_unst, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8683
dummy_1 = mynn_pow(dummy_0, p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8684
dummy_11 = r6_mul(r6_neg(r6_mul(p333, ah_unst)), mynn_pow(dummy_0, r6_neg(p666)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8685
dummy_2 = r6_mul(p333, r6_add(r6_add(mynn_ipow<2>(dummy_1), dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8686
dummy_22 = r6_mul(r6_mul(p333, dummy_11), r6_add(r6_mul(two, dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8687
dummy_3 = r6_mul(0.57735f, r6_add(r6_mul(two, dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8688
dummy_33 = r6_mul(1.1547f, dummy_11);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8689
dummy_4 = r6_add(r6_sub(r6_mul(1.5f, mynn_log(dummy_2)), r6_mul(1.73205f, mynn_atan(dummy_3))), 1.813799364f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8690
dummy_44 = r6_sub(r6_mul(r6_div(1.5f, dummy_2), dummy_22), r6_div(r6_mul(1.73205f, dummy_33), r6_add(one, mynn_ipow<2>(dummy_3))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8692
dummy_0 = mynn_ipow<2>(zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8693
dummy_1 = r6_div(one, r6_add(one, dummy_0));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8694
dummy_11 = r6_mul(two, zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8695
dummy_2 = r6_mul(r6_add(r6_add(r6_div(r6_sub(one, phh), zet), r6_mul(dummy_11, dummy_4)), r6_mul(dummy_0, dummy_44)), dummy_1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8696
dummy_22 = r6_mul(r6_neg(r6_mul(dummy_11, r6_add(dummy_psi, r6_mul(dummy_0, dummy_4)))), mynn_ipow<2>(dummy_1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8698
phih = r6_sub(one, r6_mul(zet, r6_add(dummy_2, dummy_22)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8699
}
return phih;
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8703-8817
__device__ __noinline__ void f_topdown_cloudrad(int kts, int kte, MynnArray<float, 1> dz1, MynnArray<float, 1> zw, float fltv, float xland, int kpbl, float pblh, MynnArray<float, 1> sqc, MynnArray<float, 1> sqi, MynnArray<float, 1> sqw, MynnArray<float, 1> thl, MynnArray<float, 1> th1, MynnArray<float, 1> ex1, MynnArray<float, 1> pres1, MynnArray<float, 1> rho1, MynnArray<float, 1> thv, MynnArray<float, 1> cldfra_bl1, MynnArray<float, 1> rthraten, float& maxkhtopdown, MynnArray<float, 1> khtopdown, MynnArray<float, 1> tkeprodtd) {
MynnFrame mynn_frame(arena);
dz1.rebase({kts});
zw.rebase({kts});
sqc.rebase({kts});
sqi.rebase({kts});
sqw.rebase({kts});
thl.rebase({kts});
th1.rebase({kts});
ex1.rebase({kts});
pres1.rebase({kts});
rho1.rebase({kts});
thv.rebase({kts});
cldfra_bl1.rebase({kts});
rthraten.rebase({kts});
khtopdown.rebase({kts});
tkeprodtd.rebase({kts});
MynnArray<float, 1> zfac = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> wscalek2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> zfacent = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
float bfx0;
float wm2;
float wm3;
float bfxpbl;
float dthvx;
float tmp1;
float temps;
float templ;
float zl1;
float wstar3_2;
float ent_eff;
float radsum;
float radflux;
float we;
float rcldb;
float rvls;
float minrad;
float zminrad;
constexpr float pfac = 2.0f;
constexpr float zfmin = 0.01f;
constexpr float phifac = 8.0f;
int k;
int kk;
int kminrad;
bool cloudflg;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8730
cloudflg = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8731
minrad = 100.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8732
kminrad = kpbl;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8733
zminrad = pblh;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8734
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (khtopdown.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { khtopdown.slice(MynnRange{kts, kte}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8735
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprodtd.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { tkeprodtd.slice(MynnRange{kts, kte}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8736
maxkhtopdown = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8739
for (kk=mynn_max((int)(((kpbl) - (2))), (int)(1)); (1) > 0 ? kk <= (((kpbl) + (3))) : kk >= (((kpbl) + (3))); kk += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8740
if (((((((sqc(kk)) > (1.e-6f))) || (((sqi(kk)) > (1.e-6f))))) || (((cldfra_bl1(kk)) > (0.5f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8742
cloudflg = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8743
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8744
if (((rthraten(kk)) < (minrad))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8745
minrad = rthraten(kk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8746
kminrad = kk;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8747
zminrad = r6_add(zw(kk), r6_mul(0.5f, dz1(kk)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8748
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8749
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8751
if (((mynn_max((int)(kminrad), (int)(kpbl))) < (2))) { cloudflg = false; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8752
if (cloudflg) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8753
zl1 = dz1(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8754
k = mynn_max((int)(((kpbl) - (1))), (int)(((kminrad) - (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8758
templ = r6_mul(thl(k), ex1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8760
rvls = r6_mul(r6_mul(r6_mul(100.f, 6.112f), mynn_exp(r6_div(r6_mul(17.67f, r6_sub(templ, 273.16f)), r6_sub(templ, 29.65f)))), r6_div(ep_2, pres1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8761
temps = r6_add(templ, r6_div(r6_sub(sqw(k), rvls), r6_add(r6_div(cp, xlv), r6_div(r6_mul(r6_mul(ep_2, xlv), rvls), r6_mul(r_d, mynn_ipow<2>(templ))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8762
rvls = r6_mul(r6_mul(r6_mul(100.f, 6.112f), mynn_exp(r6_div(r6_mul(17.67f, r6_sub(temps, 273.15f)), r6_sub(temps, 29.65f)))), r6_div(ep_2, pres1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8763
rcldb = mynn_max((float)(zero), (float)(r6_sub(sqw(k), rvls)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8766
dthvx = r6_sub(r6_add(thl(((k) + (2))), r6_mul(r6_mul(th1(((k) + (2))), p608), sqw(((k) + (2))))), r6_add(thl(k), r6_mul(r6_mul(th1(k), p608), sqw(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8768
dthvx = mynn_max((float)(dthvx), (float)(p1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8769
tmp1 = r6_div(r6_mul(xlvcp, rcldb), r6_mul(ex1(k), dthvx));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8772
ent_eff = r6_add(p2, r6_mul(r6_mul(p2, eight), tmp1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8774
radsum = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8775
for (kk=mynn_max((int)(((kpbl) - (3))), (int)(1)); (1) > 0 ? kk <= (((kpbl) + (3))) : kk >= (((kpbl) + (3))); kk += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8776
radflux = r6_mul(rthraten(kk), ex1(kk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8777
radflux = r6_mul(r6_div(r6_mul(radflux, cp), grav), r6_sub(pres1(kk), pres1(((kk) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8778
if (((radflux) < (zero))) { radsum = r6_add(mynn_abs(radflux), radsum); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8779
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8782
if (((r6_sub(xland, 1.5f)) >= (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8783
radsum = mynn_min((float)(radsum), (float)(90.0f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8784
bfx0 = mynn_max((float)(zero), (float)(r6_div(r6_div(radsum, rho1(k)), cp)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8785
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8786
radsum = mynn_min((float)(r6_mul(p25, radsum)), (float)(30.0f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8787
bfx0 = mynn_max((float)(zero), (float)(r6_sub(r6_div(r6_div(radsum, rho1(k)), cp), mynn_max((float)(zero), (float)(fltv)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8788
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8791
wm3 = r6_mul(r6_mul(r6_div(grav, thv(k)), bfx0), mynn_min((float)(pblh), (float)(1500.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8792
wm2 = r6_add(wm2, mynn_pow(wm3, p666));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8793
bfxpbl = r6_neg(r6_mul(ent_eff, bfx0));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8794
dthvx = mynn_max((float)(r6_sub(thv(((k) + (1))), thv(k))), (float)(p1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8795
we = mynn_max((float)(r6_div(bfxpbl, dthvx)), (float)(r6_neg(mynn_sqrt(mynn_pow(wm3, p666)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8797
{ const int mynn_lo0 = (kts), mynn_hi0 = (((kpbl) + (3))), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { kk = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8799
zfac(kk) = mynn_min((float)(mynn_max((float)(r6_sub(one, r6_div(r6_sub(zw(((kk) + (1))), zl1), r6_sub(zminrad, zl1)))), (float)(zfmin))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8800
zfacent(kk) = r6_mul(r6_mul(ten, mynn_max((float)(zero), (float)(r6_div(r6_sub(zminrad, zw(((kk) + (1)))), zminrad)))), mynn_ipow<3>(r6_sub(one, zfac(kk))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8803
wscalek2(kk) = mynn_pow(r6_mul(r6_mul(r6_mul(phifac, karman), wm3), zfac(kk)), p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8805
khtopdown(kk) = r6_mul(r6_mul(r6_mul(wscalek2(kk), karman), r6_sub(zminrad, zw(((kk) + (1))))), mynn_ipow<3>(r6_sub(one, zfac(kk))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8806
khtopdown(kk) = mynn_max((float)(zero), (float)(khtopdown(kk)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8811
tkeprodtd(kk) = r6_mul(r6_div(r6_mul(r6_mul(two, ent_eff), wm3), mynn_max((float)(pblh), (float)(hundred))), zfacent(kk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8812
tkeprodtd(kk) = mynn_max((float)(zero), (float)(tkeprodtd(kk)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8813
} mynn_lane_sync(); kk = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8814
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8815
maxkhtopdown = mynn_maxval(khtopdown.slice(MynnRange{khtopdown.lo[0], khtopdown.hi[0]}));
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:17-653
__device__ __forceinline__ void f_mynnedmf_driver(int ids, int ide, int jds, int jde, int kds, int kde, int ims, int ime, int jms, int jme, int kms, int kme, int its, int ite, int jts, int jte, int kts, int kte, bool f_qc, bool f_qi, bool f_qs, bool f_qoz, bool f_nc, bool f_ni, bool f_nifa, bool f_nwfa, bool f_nbca, int initflag, bool do_restart, bool do_dacycling, int icloud_bl, float delt, MynnArray<float, 2> dx, MynnArray<float, 2> xland, MynnArray<float, 2> ps, MynnArray<float, 2> ts, MynnArray<float, 2> qsfc, MynnArray<float, 2> ust, MynnArray<float, 2> ch, MynnArray<float, 2> hfx, MynnArray<float, 2> qfx, MynnArray<float, 2> wspd, MynnArray<float, 2> znt, MynnArray<float, 2> uoce, MynnArray<float, 2> voce, MynnArray<float, 3> dz, MynnArray<float, 3> u, MynnArray<float, 3> v, MynnArray<float, 3> w, MynnArray<float, 3> th, MynnArray<float, 3> tt, MynnArray<float, 3> p, MynnArray<float, 3> exner, MynnArray<float, 3> rho, MynnArray<float, 3> qv, MynnArray<float, 3> qc, MynnArray<float, 3> qi, MynnArray<float, 3> qs, MynnArray<float, 3> nc, MynnArray<float, 3> ni, MynnArray<float, 3> nifa, MynnArray<float, 3> nwfa, MynnArray<float, 3> nbca, MynnArray<float, 3> qoz, MynnArray<float, 3> rthraten, MynnArray<float, 2> pblh, MynnArray<int, 2> kpbl, MynnArray<float, 3> cldfra_bl, MynnArray<float, 3> qc_bl, MynnArray<float, 3> qi_bl, MynnArray<float, 2> maxwidth, MynnArray<float, 2> maxmf, MynnArray<float, 2> ztop_plume, MynnArray<float, 2> excess_h, MynnArray<float, 2> excess_q, MynnArray<float, 3> qke, MynnArray<float, 3> qke_adv, MynnArray<float, 3> tsq, MynnArray<float, 3> qsq, MynnArray<float, 3> cov, MynnArray<float, 3> el_pbl, MynnArray<float, 3> rublten, MynnArray<float, 3> rvblten, MynnArray<float, 3> rthblten, MynnArray<float, 3> rqvblten, MynnArray<float, 3> rqcblten, MynnArray<float, 3> rqiblten, MynnArray<float, 3> rqsblten, MynnArray<float, 3> rncblten, MynnArray<float, 3> rniblten, MynnArray<float, 3> rnifablten, MynnArray<float, 3> rnwfablten, MynnArray<float, 3> rnbcablten, MynnArray<float, 3> rqozblten, MynnArray<float, 3> edmf_a, MynnArray<float, 3> edmf_w, MynnArray<float, 3> edmf_qt, MynnArray<float, 3> edmf_thl, MynnArray<float, 3> edmf_ent, MynnArray<float, 3> edmf_qc, MynnArray<float, 3> sub_thl, MynnArray<float, 3> sub_sqv, MynnArray<float, 3> det_thl, MynnArray<float, 3> det_sqv, MynnArray<float, 3> exch_h, MynnArray<float, 3> exch_m, MynnArray<float, 3> dqke, MynnArray<float, 3> qwt, MynnArray<float, 3> qshear, MynnArray<float, 3> qbuoy, MynnArray<float, 3> qdiss, MynnArray<float, 3> sh3d, MynnArray<float, 3> sm3d, int spp_pbl, MynnArray<float, 3> pattern_spp, bool bl_mynn_tkeadvect, int bl_mynn_tkebudget, int bl_mynn_cloudpdf, int bl_mynn_mixlength, float bl_mynn_closure, int bl_mynn_stfunc, int bl_mynn_topdown, int bl_mynn_scaleaware, int bl_mynn_dheat_opt, int bl_mynn_edmf, int bl_mynn_edmf_dd, int bl_mynn_edmf_mom, int bl_mynn_edmf_tke, int bl_mynn_output, int bl_mynn_mixscalars, int bl_mynn_mixaerosols, int bl_mynn_mixnumcon, int bl_mynn_cloudmix, int bl_mynn_mixqt, int bl_mynn_ess, MynnText& errmsg, int& errflg, bool mix_chem, int nchem, int ndvel, MynnArray<float, 4> chem3d, MynnArray<float, 4> settle3d, MynnArray<float, 3> vd3d, MynnOptional<bool> enh_mix, MynnArray<float, 2> frp_mean, MynnArray<float, 2> emis_ant_no) {
MynnFrame mynn_frame(arena);
dx.rebase({ims, jms});
xland.rebase({ims, jms});
ps.rebase({ims, jms});
ts.rebase({ims, jms});
qsfc.rebase({ims, jms});
ust.rebase({ims, jms});
ch.rebase({ims, jms});
hfx.rebase({ims, jms});
qfx.rebase({ims, jms});
wspd.rebase({ims, jms});
znt.rebase({ims, jms});
uoce.rebase({ims, jms});
voce.rebase({ims, jms});
dz.rebase({ims, kms, jms});
u.rebase({ims, kms, jms});
v.rebase({ims, kms, jms});
w.rebase({ims, kms, jms});
th.rebase({ims, kms, jms});
tt.rebase({ims, kms, jms});
p.rebase({ims, kms, jms});
exner.rebase({ims, kms, jms});
rho.rebase({ims, kms, jms});
qv.rebase({ims, kms, jms});
qc.rebase({ims, kms, jms});
qi.rebase({ims, kms, jms});
qs.rebase({ims, kms, jms});
nc.rebase({ims, kms, jms});
ni.rebase({ims, kms, jms});
nifa.rebase({ims, kms, jms});
nwfa.rebase({ims, kms, jms});
nbca.rebase({ims, kms, jms});
qoz.rebase({ims, kms, jms});
rthraten.rebase({ims, kms, jms});
pblh.rebase({ims, jms});
kpbl.rebase({ims, jms});
cldfra_bl.rebase({ims, kms, jms});
qc_bl.rebase({ims, kms, jms});
qi_bl.rebase({ims, kms, jms});
maxwidth.rebase({ims, jms});
maxmf.rebase({ims, jms});
ztop_plume.rebase({ims, jms});
excess_h.rebase({ims, jms});
excess_q.rebase({ims, jms});
qke.rebase({ims, kms, jms});
qke_adv.rebase({ims, kms, jms});
tsq.rebase({ims, kms, jms});
qsq.rebase({ims, kms, jms});
cov.rebase({ims, kms, jms});
el_pbl.rebase({ims, kms, jms});
rublten.rebase({ims, kms, jms});
rvblten.rebase({ims, kms, jms});
rthblten.rebase({ims, kms, jms});
rqvblten.rebase({ims, kms, jms});
rqcblten.rebase({ims, kms, jms});
rqiblten.rebase({ims, kms, jms});
rqsblten.rebase({ims, kms, jms});
rncblten.rebase({ims, kms, jms});
rniblten.rebase({ims, kms, jms});
rnifablten.rebase({ims, kms, jms});
rnwfablten.rebase({ims, kms, jms});
rnbcablten.rebase({ims, kms, jms});
rqozblten.rebase({ims, kms, jms});
edmf_a.rebase({ims, kms, jms});
edmf_w.rebase({ims, kms, jms});
edmf_qt.rebase({ims, kms, jms});
edmf_thl.rebase({ims, kms, jms});
edmf_ent.rebase({ims, kms, jms});
edmf_qc.rebase({ims, kms, jms});
sub_thl.rebase({ims, kms, jms});
sub_sqv.rebase({ims, kms, jms});
det_thl.rebase({ims, kms, jms});
det_sqv.rebase({ims, kms, jms});
exch_h.rebase({ims, kms, jms});
exch_m.rebase({ims, kms, jms});
dqke.rebase({ims, kms, jms});
qwt.rebase({ims, kms, jms});
qshear.rebase({ims, kms, jms});
qbuoy.rebase({ims, kms, jms});
qdiss.rebase({ims, kms, jms});
sh3d.rebase({ims, kms, jms});
sm3d.rebase({ims, kms, jms});
pattern_spp.rebase({ims, kms, jms});
chem3d.rebase({ims, kms, jms, 1});
settle3d.rebase({ims, kms, jms, 1});
vd3d.rebase({ims, jms, 1});
frp_mean.rebase({ims, jms});
emis_ant_no.rebase({ims, jms});
float frp1;
float emisant_no1;
MynnArray<float, 1> vd1 = mynn_array<float, 1, mynn_tile>(arena, {1}, {ndvel});
MynnArray<float, 2> chem1 = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {kte, nchem});
MynnArray<float, 2> settle1 = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {kte, nchem});
constexpr int nscalars = 1;
MynnArray<float, 2> scalars = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {kte, nscalars});
int i;
int k;
int j;
int ic;
int dheat_opt;
int kpbl1;
float denom;
float dx1;
float xland1;
float ps1;
float ts1;
float qsfc1;
float ust1;
float ch1;
float hfx1;
float qfx1;
float wspd1;
float uoce1;
float voce1;
float znt1;
MynnArray<float, 1> dz1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> u1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> v1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> th1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> tt1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> p1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> exner1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rho1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rthraten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> w1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kme});
MynnArray<float, 1> qc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qi1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qs1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> nc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> ni1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> nifa1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> nwfa1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> nbca1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qoz1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> pattern_spp1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
float pblh1;
MynnArray<float, 1> cldfrabl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qcbl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qibl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> elpbl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qke1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qkeadv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> cov1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qsq1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> tsq1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sh1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sm1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rublten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rvblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rthblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rqvblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rqcblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rqiblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rqsblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rncblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rniblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rnifablten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rnwfablten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rnbcablten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rqozblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmfa1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmfw1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmfqt1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmfthl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmfent1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmfqc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> subthl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> subsqv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> detthl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> detsqv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
float maxwidth1;
float maxmf1;
float ztopplume1;
float excessh1;
float excessq1;
MynnArray<float, 1> exchh1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> exchm1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dqke1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qwt1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qshear1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qbuoy1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qdiss1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqi1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqs1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:300
errmsg = " ";
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:301
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:303
for (j=(active_j >= 0 ? active_j : jts); (1) > 0 ? j <= ((active_j >= 0 ? active_j : jte)) : j >= ((active_j >= 0 ? active_j : jte)); j += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:304
for (i=(active_i >= 0 ? active_i : its); (1) > 0 ? i <= ((active_i >= 0 ? active_i : ite)) : i >= ((active_i >= 0 ? active_i : ite)); i += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:307
dx1 = dx(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:308
xland1 = xland(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:309
ps1 = ps(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:310
ts1 = ts(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:311
qsfc1 = qsfc(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:312
ust1 = ust(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:313
ch1 = ch(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:314
hfx1 = hfx(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:315
qfx1 = qfx(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:316
wspd1 = wspd(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:317
uoce1 = uoce(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:318
voce1 = voce(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:319
znt1 = znt(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:321
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:322
dz1(k) = dz(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:323
u1(k) = u(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:324
v1(k) = v(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:325
w1(k) = w(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:326
th1(k) = th(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:327
tt1(k) = tt(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:328
p1(k) = p(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:329
exner1(k) = exner(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:330
rho1(k) = rho(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:331
qv1(k) = qv(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:332
rthraten1(k) = rthraten(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:333
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:334
w1(((kte) + (1))) = w(i, ((kte) + (1)), j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:339
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:340
qc1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:341
qi1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:342
qs1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:343
qoz1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:344
nc1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:345
ni1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:346
nifa1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:347
nwfa1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:348
nbca1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:349
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:350
if (((f_qc) && ((qc.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:351
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:352
qc1(k) = qc(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:353
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:354
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:355
if (((f_qi) && ((qi.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:356
{ const int mynn_lo3 = (kts), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:357
qi1(k) = qi(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:358
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:359
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:360
if (((f_qs) && ((qs.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:361
{ const int mynn_lo4 = (kts), mynn_hi4 = (kte), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:362
qs1(k) = qs(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:363
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:364
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:365
if (((f_nc) && ((nc.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:366
{ const int mynn_lo5 = (kts), mynn_hi5 = (kte), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:367
nc1(k) = nc(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:368
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:369
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:370
if (((f_ni) && ((ni.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:371
{ const int mynn_lo6 = (kts), mynn_hi6 = (kte), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:372
ni1(k) = ni(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:373
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:374
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:375
if (((f_nifa) && ((nifa.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:376
{ const int mynn_lo7 = (kts), mynn_hi7 = (kte), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:377
nifa1(k) = nifa(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:378
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:379
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:380
if (((f_nwfa) && ((nwfa.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:381
{ const int mynn_lo8 = (kts), mynn_hi8 = (kte), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:382
nwfa1(k) = nwfa(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:383
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:384
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:385
if (((f_nbca) && ((nbca.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:386
{ const int mynn_lo9 = (kts), mynn_hi9 = (kte), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:387
nbca1(k) = nbca(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:388
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:389
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:390
if (((f_qoz) && ((qoz.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:391
{ const int mynn_lo10 = (kts), mynn_hi10 = (kte), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:392
qoz1(k) = qoz(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:393
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:394
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:397
f_mynnedmf_pre_run(kte, f_qc, f_qi, f_qs, qv1, qc1, qi1, qs1, sqv1, sqc1, sqi1, sqs1, errmsg, errflg);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:401
if (((((spp_pbl) > (0))) && ((pattern_spp.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:402
{ const int mynn_lo11 = (kts), mynn_hi11 = (kte), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { k = mynn_lo11 + mynn_it11*mynn_st11;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:403
pattern_spp1(k) = pattern_spp(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:404
} mynn_lane_sync(); k = mynn_lo11 + mynn_n11*mynn_st11; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:405
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:406
{ const int mynn_lo12 = (kts), mynn_hi12 = (kte), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { k = mynn_lo12 + mynn_it12*mynn_st12;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:407
pattern_spp1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:408
} mynn_lane_sync(); k = mynn_lo12 + mynn_n12*mynn_st12; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:409
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:412
pblh1 = pblh(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:413
kpbl1 = kpbl(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:415
{ const int mynn_lo13 = (kts), mynn_hi13 = (kte), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { k = mynn_lo13 + mynn_it13*mynn_st13;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:416
cldfrabl1(k) = cldfra_bl(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:417
qcbl1(k) = qc_bl(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:418
qibl1(k) = qi_bl(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:419
} mynn_lane_sync(); k = mynn_lo13 + mynn_n13*mynn_st13; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:421
{ const int mynn_lo14 = (kts), mynn_hi14 = (kte), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { k = mynn_lo14 + mynn_it14*mynn_st14;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:422
elpbl1(k) = el_pbl(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:423
qke1(k) = qke(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:424
qkeadv1(k) = qke_adv(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:425
cov1(k) = cov(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:426
tsq1(k) = tsq(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:427
qsq1(k) = qsq(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:428
sh1(k) = sh3d(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:429
sm1(k) = sm3d(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:430
} mynn_lane_sync(); k = mynn_lo14 + mynn_n14*mynn_st14; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:432
if ((((((((((chem3d.present())) && ((settle3d.present())))) && ((vd3d.present())))) && ((frp_mean.present())))) && ((emis_ant_no.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:434
{ const int mynn_lo15 = (1), mynn_hi15 = (nchem), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { ic = mynn_lo15 + mynn_it15*mynn_st15;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:435
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:436
chem1(k, ic) = chem3d(i, k, j, ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:437
settle1(k, ic) = settle3d(i, k, j, ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:438
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:439
} mynn_lane_sync(); ic = mynn_lo15 + mynn_n15*mynn_st15; if (mynn_n15 > 0) { const int mynn_src15 = (mynn_n15 - 1) % lane_width; k = mynn_lane_bcast(k, mynn_src15); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:440
{ const int mynn_lo16 = (1), mynn_hi16 = (ndvel), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { ic = mynn_lo16 + mynn_it16*mynn_st16;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:441
vd1(ic) = vd3d(i, j, ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:442
} mynn_lane_sync(); ic = mynn_lo16 + mynn_n16*mynn_st16; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:443
frp1 = frp_mean(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:444
emisant_no1 = emis_ant_no(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:445
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:446
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (chem1).size(); mynn_element += lane_width) { chem1.linear(mynn_element) = 0.f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:447
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (settle1).size(); mynn_element += lane_width) { settle1.linear(mynn_element) = 0.f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:448
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (vd1).size(); mynn_element += lane_width) { vd1.linear(mynn_element) = 0.f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:449
frp1 = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:450
emisant_no1 = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:451
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:452
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (scalars).size(); mynn_element += lane_width) { scalars.linear(mynn_element) = 0.0f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:454
{ const int mynn_lo17 = (kts), mynn_hi17 = (kte), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { k = mynn_lo17 + mynn_it17*mynn_st17;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:455
rqcblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:456
rqiblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:457
rqsblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:458
rqozblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:459
rncblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:460
rniblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:461
rnifablten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:462
rnwfablten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:463
rnbcablten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:464
} mynn_lane_sync(); k = mynn_lo17 + mynn_n17*mynn_st17; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:466
f_mynnedmf(i, j, initflag, do_restart, do_dacycling, delt, dz1, dx1, u1, v1, w1, th1, sqv1, sqc1, sqi1, sqs1, nc1, ni1, nwfa1, nifa1, nbca1, qoz1, p1, exner1, rho1, tt1, xland1, ts1, qsfc1, ps1, ust1, ch1, hfx1, qfx1, znt1, wspd1, uoce1, voce1, qke1, qkeadv1, elpbl1, sh1, sm1, exchh1, exchm1, nchem, ndvel, chem1, vd1, frp1, emisant_no1, mix_chem, enh_mix, settle1, scalars, nscalars, tsq1, qsq1, cov1, rublten1, rvblten1, rthblten1, rqvblten1, rqcblten1, rqiblten1, rncblten1, rniblten1, rqsblten1, rnwfablten1, rnifablten1, rnbcablten1, rqozblten1, rthraten1, pblh1, kpbl1, maxwidth1, maxmf1, ztopplume1, excessh1, excessq1, dqke1, qwt1, qshear1, qbuoy1, qdiss1, qcbl1, qibl1, cldfrabl1, bl_mynn_tkeadvect, bl_mynn_tkebudget, bl_mynn_cloudpdf, bl_mynn_mixlength, icloud_bl, bl_mynn_closure, bl_mynn_edmf, bl_mynn_edmf_mom, bl_mynn_edmf_tke, bl_mynn_mixscalars, bl_mynn_mixaerosols, bl_mynn_mixnumcon, bl_mynn_output, bl_mynn_cloudmix, bl_mynn_mixqt, bl_mynn_edmf_dd, bl_mynn_ess, edmfa1, edmfw1, edmfqt1, edmfthl1, edmfent1, edmfqc1, subthl1, subsqv1, detthl1, detsqv1, spp_pbl, pattern_spp1, f_qc, f_qi, f_nc, f_ni, f_qs, f_nwfa, f_nifa, f_nbca, f_qoz, kts, kte, errmsg, errflg);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:528
f_mynnedmf_post_run(kte, f_qc, f_qi, f_qs, delt, qv1, qc1, qi1, qs1, rqvblten1, rqcblten1, rqiblten1, rqsblten1, errmsg, errflg);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:532
pblh(i, j) = pblh1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:533
kpbl(i, j) = kpbl1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:534
{ const int mynn_lo18 = (kts), mynn_hi18 = (kte), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { k = mynn_lo18 + mynn_it18*mynn_st18;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:535
cldfra_bl(i, k, j) = cldfrabl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:536
qc_bl(i, k, j) = qcbl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:537
qi_bl(i, k, j) = qibl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:538
} mynn_lane_sync(); k = mynn_lo18 + mynn_n18*mynn_st18; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:540
{ const int mynn_lo19 = (kts), mynn_hi19 = (kte), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { k = mynn_lo19 + mynn_it19*mynn_st19;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:541
el_pbl(i, k, j) = elpbl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:542
qke(i, k, j) = qke1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:543
qke_adv(i, k, j) = qkeadv1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:544
cov(i, k, j) = cov1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:545
tsq(i, k, j) = tsq1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:546
qsq(i, k, j) = qsq1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:547
sh3d(i, k, j) = sh1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:548
sm3d(i, k, j) = sm1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:549
} mynn_lane_sync(); k = mynn_lo19 + mynn_n19*mynn_st19; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:552
{ const int mynn_lo20 = (kts), mynn_hi20 = (kte), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { k = mynn_lo20 + mynn_it20*mynn_st20;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:553
rublten(i, k, j) = rublten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:554
rvblten(i, k, j) = rvblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:555
rthblten(i, k, j) = rthblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:556
rqvblten(i, k, j) = rqvblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:557
} mynn_lane_sync(); k = mynn_lo20 + mynn_n20*mynn_st20; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:558
if (((f_qc) && ((rqcblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:559
{ const int mynn_lo21 = (kts), mynn_hi21 = (kte), mynn_st21 = (1); const int mynn_n21 = mynn_trip(mynn_lo21, mynn_hi21, mynn_st21); mynn_lane_sync(); for (int mynn_it21 = lane_sub; mynn_it21 < mynn_n21; mynn_it21 += lane_width) { k = mynn_lo21 + mynn_it21*mynn_st21;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:560
rqcblten(i, k, j) = rqcblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:561
} mynn_lane_sync(); k = mynn_lo21 + mynn_n21*mynn_st21; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:562
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:563
if (((f_qi) && ((rqiblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:564
{ const int mynn_lo22 = (kts), mynn_hi22 = (kte), mynn_st22 = (1); const int mynn_n22 = mynn_trip(mynn_lo22, mynn_hi22, mynn_st22); mynn_lane_sync(); for (int mynn_it22 = lane_sub; mynn_it22 < mynn_n22; mynn_it22 += lane_width) { k = mynn_lo22 + mynn_it22*mynn_st22;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:565
rqiblten(i, k, j) = rqiblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:566
} mynn_lane_sync(); k = mynn_lo22 + mynn_n22*mynn_st22; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:567
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:568
if (((f_qs) && ((rqsblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:569
{ const int mynn_lo23 = (kts), mynn_hi23 = (kte), mynn_st23 = (1); const int mynn_n23 = mynn_trip(mynn_lo23, mynn_hi23, mynn_st23); mynn_lane_sync(); for (int mynn_it23 = lane_sub; mynn_it23 < mynn_n23; mynn_it23 += lane_width) { k = mynn_lo23 + mynn_it23*mynn_st23;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:570
rqsblten(i, k, j) = rqsblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:571
} mynn_lane_sync(); k = mynn_lo23 + mynn_n23*mynn_st23; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:572
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:573
if (((f_qoz) && ((rqozblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:574
{ const int mynn_lo24 = (kts), mynn_hi24 = (kte), mynn_st24 = (1); const int mynn_n24 = mynn_trip(mynn_lo24, mynn_hi24, mynn_st24); mynn_lane_sync(); for (int mynn_it24 = lane_sub; mynn_it24 < mynn_n24; mynn_it24 += lane_width) { k = mynn_lo24 + mynn_it24*mynn_st24;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:575
rqozblten(i, k, j) = rqozblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:576
} mynn_lane_sync(); k = mynn_lo24 + mynn_n24*mynn_st24; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:577
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:578
if (((f_nc) && ((rncblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:579
{ const int mynn_lo25 = (kts), mynn_hi25 = (kte), mynn_st25 = (1); const int mynn_n25 = mynn_trip(mynn_lo25, mynn_hi25, mynn_st25); mynn_lane_sync(); for (int mynn_it25 = lane_sub; mynn_it25 < mynn_n25; mynn_it25 += lane_width) { k = mynn_lo25 + mynn_it25*mynn_st25;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:580
rncblten(i, k, j) = rncblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:581
} mynn_lane_sync(); k = mynn_lo25 + mynn_n25*mynn_st25; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:582
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:583
if (((f_ni) && ((rniblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:584
{ const int mynn_lo26 = (kts), mynn_hi26 = (kte), mynn_st26 = (1); const int mynn_n26 = mynn_trip(mynn_lo26, mynn_hi26, mynn_st26); mynn_lane_sync(); for (int mynn_it26 = lane_sub; mynn_it26 < mynn_n26; mynn_it26 += lane_width) { k = mynn_lo26 + mynn_it26*mynn_st26;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:585
rniblten(i, k, j) = rniblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:586
} mynn_lane_sync(); k = mynn_lo26 + mynn_n26*mynn_st26; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:587
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:588
if (((f_nifa) && ((rnifablten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:589
{ const int mynn_lo27 = (kts), mynn_hi27 = (kte), mynn_st27 = (1); const int mynn_n27 = mynn_trip(mynn_lo27, mynn_hi27, mynn_st27); mynn_lane_sync(); for (int mynn_it27 = lane_sub; mynn_it27 < mynn_n27; mynn_it27 += lane_width) { k = mynn_lo27 + mynn_it27*mynn_st27;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:590
rnifablten(i, k, j) = rnifablten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:591
} mynn_lane_sync(); k = mynn_lo27 + mynn_n27*mynn_st27; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:592
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:593
if (((f_nwfa) && ((rnwfablten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:594
{ const int mynn_lo28 = (kts), mynn_hi28 = (kte), mynn_st28 = (1); const int mynn_n28 = mynn_trip(mynn_lo28, mynn_hi28, mynn_st28); mynn_lane_sync(); for (int mynn_it28 = lane_sub; mynn_it28 < mynn_n28; mynn_it28 += lane_width) { k = mynn_lo28 + mynn_it28*mynn_st28;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:595
rnwfablten(i, k, j) = rnwfablten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:596
} mynn_lane_sync(); k = mynn_lo28 + mynn_n28*mynn_st28; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:597
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:598
if (((f_nbca) && ((rnbcablten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:599
{ const int mynn_lo29 = (kts), mynn_hi29 = (kte), mynn_st29 = (1); const int mynn_n29 = mynn_trip(mynn_lo29, mynn_hi29, mynn_st29); mynn_lane_sync(); for (int mynn_it29 = lane_sub; mynn_it29 < mynn_n29; mynn_it29 += lane_width) { k = mynn_lo29 + mynn_it29*mynn_st29;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:600
rnbcablten(i, k, j) = rnbcablten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:601
} mynn_lane_sync(); k = mynn_lo29 + mynn_n29*mynn_st29; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:602
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:604
{ const int mynn_lo30 = (kts), mynn_hi30 = (kte), mynn_st30 = (1); const int mynn_n30 = mynn_trip(mynn_lo30, mynn_hi30, mynn_st30); mynn_lane_sync(); for (int mynn_it30 = lane_sub; mynn_it30 < mynn_n30; mynn_it30 += lane_width) { k = mynn_lo30 + mynn_it30*mynn_st30;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:605
edmf_a(i, k, j) = edmfa1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:606
edmf_w(i, k, j) = edmfw1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:607
edmf_qt(i, k, j) = edmfqt1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:608
edmf_thl(i, k, j) = edmfthl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:609
edmf_ent(i, k, j) = edmfent1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:610
edmf_qc(i, k, j) = edmfqc1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:611
sub_thl(i, k, j) = subthl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:612
sub_sqv(i, k, j) = subsqv1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:613
det_thl(i, k, j) = detthl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:614
det_sqv(i, k, j) = detsqv1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:615
} mynn_lane_sync(); k = mynn_lo30 + mynn_n30*mynn_st30; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:618
maxwidth(i, j) = maxwidth1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:619
maxmf(i, j) = maxmf1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:620
ztop_plume(i, j) = ztopplume1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:621
excess_h(i, j) = excessh1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:622
excess_q(i, j) = excessq1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:624
{ const int mynn_lo31 = (kts), mynn_hi31 = (kte), mynn_st31 = (1); const int mynn_n31 = mynn_trip(mynn_lo31, mynn_hi31, mynn_st31); mynn_lane_sync(); for (int mynn_it31 = lane_sub; mynn_it31 < mynn_n31; mynn_it31 += lane_width) { k = mynn_lo31 + mynn_it31*mynn_st31;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:625
exch_h(i, k, j) = exchh1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:626
exch_m(i, k, j) = exchm1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:627
} mynn_lane_sync(); k = mynn_lo31 + mynn_n31*mynn_st31; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:629
if ((((((((((qwt.present())) && ((qbuoy.present())))) && ((qshear.present())))) && ((qdiss.present())))) && ((dqke.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:631
{ const int mynn_lo32 = (kts), mynn_hi32 = (kte), mynn_st32 = (1); const int mynn_n32 = mynn_trip(mynn_lo32, mynn_hi32, mynn_st32); mynn_lane_sync(); for (int mynn_it32 = lane_sub; mynn_it32 < mynn_n32; mynn_it32 += lane_width) { k = mynn_lo32 + mynn_it32*mynn_st32;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:632
dqke(i, k, j) = dqke1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:633
qwt(i, k, j) = qwt1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:634
qshear(i, k, j) = qshear1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:635
qbuoy(i, k, j) = qbuoy1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:636
qdiss(i, k, j) = qdiss1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:637
} mynn_lane_sync(); k = mynn_lo32 + mynn_n32*mynn_st32; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:638
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:640
if (((mix_chem) && ((chem3d.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:641
{ const int mynn_lo33 = (1), mynn_hi33 = (nchem), mynn_st33 = (1); const int mynn_n33 = mynn_trip(mynn_lo33, mynn_hi33, mynn_st33); mynn_lane_sync(); for (int mynn_it33 = lane_sub; mynn_it33 < mynn_n33; mynn_it33 += lane_width) { ic = mynn_lo33 + mynn_it33*mynn_st33;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:642
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:643
chem3d(i, k, j, ic) = mynn_max((float)(chem1(k, ic)), (float)(1.e-12f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:644
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:645
} mynn_lane_sync(); ic = mynn_lo33 + mynn_n33*mynn_st33; if (mynn_n33 > 0) { const int mynn_src33 = (mynn_n33 - 1) % lane_width; k = mynn_lane_bcast(k, mynn_src33); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:646
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:648
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:649
}
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:659-675
__device__ __noinline__ void f_mynnedmf_pre_init(MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:672
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:673
errmsg = " ";
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:681-697
__device__ __noinline__ void f_mynnedmf_pre_finalize(MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:694
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:695
errmsg = " ";
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:703-772
__device__ __noinline__ void f_mynnedmf_pre_run(int kte, bool f_qc, bool f_qi, bool f_qs, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> sqv, MynnArray<float, 1> sqc, MynnArray<float, 1> sqi, MynnArray<float, 1> sqs, MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
qv.rebase({1});
qc.rebase({1});
qi.rebase({1});
qs.rebase({1});
sqv.rebase({1});
sqc.rebase({1});
sqi.rebase({1});
sqs.rebase({1});
int k;
constexpr int kts = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:741
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:742
sqc(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:743
sqi(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:744
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:747
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:748
sqv(k) = r6_div(qv(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:749
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:752
if (f_qc) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:753
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:754
sqc(k) = r6_div(qc(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:755
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:756
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:757
if (f_qi) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:758
{ const int mynn_lo3 = (kts), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:759
sqi(k) = r6_div(qi(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:760
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:761
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:762
if (f_qs) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:763
{ const int mynn_lo4 = (kts), mynn_hi4 = (kte), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:764
sqs(k) = r6_div(qs(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:765
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:766
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:769
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:770
errmsg = " ";
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:779-795
__device__ __noinline__ void f_mynnedmf_post_init(MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:792
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:793
errmsg = " ";
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:801-817
__device__ __noinline__ void f_mynnedmf_post_finalize(MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:814
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:815
errmsg = " ";
}
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:823-904
__device__ __noinline__ void f_mynnedmf_post_run(int kte, bool f_qc, bool f_qi, bool f_qs, float delt, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> dqv, MynnArray<float, 1> dqc, MynnArray<float, 1> dqi, MynnArray<float, 1> dqs, MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
qv.rebase({1});
qc.rebase({1});
qi.rebase({1});
qs.rebase({1});
dqv.rebase({1});
dqc.rebase({1});
dqi.rebase({1});
dqs.rebase({1});
int k;
constexpr int kts = 1;
float rq;
float sq;
float tem;
MynnArray<float, 1> sqv = mynn_array<float, 1, mynn_tile>(arena, {1}, {kte});
MynnArray<float, 1> sqc = mynn_array<float, 1, mynn_tile>(arena, {1}, {kte});
MynnArray<float, 1> sqi = mynn_array<float, 1, mynn_tile>(arena, {1}, {kte});
MynnArray<float, 1> sqs = mynn_array<float, 1, mynn_tile>(arena, {1}, {kte});
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:866
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:867
sq = r6_div(qv(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:868
sqv(k) = r6_add(sq, r6_mul(dqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:869
rq = r6_div(sqv(k), r6_sub(1.f, sqv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:870
dqv(k) = r6_div(r6_sub(rq, qv(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:871
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src0); sq = mynn_lane_bcast(sq, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:873
if (f_qc) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:874
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:875
sq = r6_div(qc(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:876
sqc(k) = r6_add(sq, r6_mul(dqc(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:877
rq = r6_mul(sqc(k), r6_add(1.f, sqv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:878
dqc(k) = r6_div(r6_sub(rq, qc(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:879
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src1); sq = mynn_lane_bcast(sq, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:880
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:882
if (f_qi) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:883
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:884
sq = r6_div(qi(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:885
sqi(k) = r6_add(sq, r6_mul(dqi(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:886
rq = r6_mul(sqi(k), r6_add(1.f, sqv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:887
dqi(k) = r6_div(r6_sub(rq, qi(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:888
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src2); sq = mynn_lane_bcast(sq, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:889
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:891
if (f_qs) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:892
{ const int mynn_lo3 = (kts), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:893
sq = r6_div(qs(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:894
sqs(k) = r6_add(sq, r6_mul(dqs(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:895
rq = r6_mul(sqs(k), r6_add(1.f, sqv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:896
dqs(k) = r6_div(r6_sub(rq, qs(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:897
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src3); sq = mynn_lane_bcast(sq, mynn_src3); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:898
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:901
errmsg = " ";
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:902
errflg = 0;
}
};
#ifdef HEX_NUMERICS_FAST
#define MYNN_L4_REGS __maxnreg__(160)
#else
#define MYNN_L4_REGS
#endif

// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:17-653
template<int MYNN_LANES> __device__ __forceinline__ void rrfs_mynnedmf_column(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_nc,int f_ni,int f_nifa,int f_nwfa,int f_nbca,int initflag,int do_restart,int do_dacycling,int icloud_bl,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tt,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* nc,float* ni,float* nifa,float* nwfa,float* nbca,float* qoz,float* rthraten,float* pblh,int* kpbl,float* cldfra_bl,float* qc_bl,float* qi_bl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rncblten,float* rniblten,float* rnifablten,float* rnwfablten,float* rnbcablten,float* rqozblten,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int bl_mynn_tkebudget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_stfunc,int bl_mynn_topdown,int bl_mynn_scaleaware,int bl_mynn_dheat_opt,int bl_mynn_edmf,int bl_mynn_edmf_dd,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_output,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_ess,int mix_chem,int nchem,int ndvel,float* chem3d,float* settle3d,float* vd3d,int enh_mix,int present_enh_mix,float* frp_mean,float* emis_ant_no) {
constexpr int lane_width=MYNN_LANES;
// Lane groups (lane_plan.py, rrfs/speed2-mynn-edmf): lane_width lanes (1..32, a power of two) per column, so a
// warp holds 32/lane_width columns; a column's automatic arrays live in a tile shared with the warp's other columns,
// word w of column c at lane c % (32/lane_width), so a group's lanes on consecutive elements read one line.
int thread=blockIdx.x*blockDim.x+threadIdx.x;
int column_index=thread/lane_width;
if(column_index>=batch_count) return;
int index=batch_first+column_index;
constexpr int tile_columns=32/lane_width;
MynnArena arena{workspace+(unsigned long long)(column_index-column_index%tile_columns)*workspace_stride,0,tile_columns,column_index%tile_columns};
MynnColumn<MYNN_LANES> column{arena};
column.lane_sub=thread%lane_width;
column.lane_mask=lane_width==32 ? 0xffffffffu : ((1u<<(lane_width%32))-1u)<<((threadIdx.x&31)/lane_width*lane_width);
column.active_i=its+index%(ite-its+1);
column.active_j=jts+index/(ite-its+1);
MynnText message;int error_flag=0;
MynnArray<float,2> a_dx;a_dx.data=dx;
a_dx.lo[0]=ims;a_dx.hi[0]=ime;a_dx.stride[0]=1;
a_dx.lo[1]=jms;a_dx.hi[1]=jme;a_dx.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_xland;a_xland.data=xland;
a_xland.lo[0]=ims;a_xland.hi[0]=ime;a_xland.stride[0]=1;
a_xland.lo[1]=jms;a_xland.hi[1]=jme;a_xland.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_ps;a_ps.data=ps;
a_ps.lo[0]=ims;a_ps.hi[0]=ime;a_ps.stride[0]=1;
a_ps.lo[1]=jms;a_ps.hi[1]=jme;a_ps.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_ts;a_ts.data=ts;
a_ts.lo[0]=ims;a_ts.hi[0]=ime;a_ts.stride[0]=1;
a_ts.lo[1]=jms;a_ts.hi[1]=jme;a_ts.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_qsfc;a_qsfc.data=qsfc;
a_qsfc.lo[0]=ims;a_qsfc.hi[0]=ime;a_qsfc.stride[0]=1;
a_qsfc.lo[1]=jms;a_qsfc.hi[1]=jme;a_qsfc.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_ust;a_ust.data=ust;
a_ust.lo[0]=ims;a_ust.hi[0]=ime;a_ust.stride[0]=1;
a_ust.lo[1]=jms;a_ust.hi[1]=jme;a_ust.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_ch;a_ch.data=ch;
a_ch.lo[0]=ims;a_ch.hi[0]=ime;a_ch.stride[0]=1;
a_ch.lo[1]=jms;a_ch.hi[1]=jme;a_ch.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_hfx;a_hfx.data=hfx;
a_hfx.lo[0]=ims;a_hfx.hi[0]=ime;a_hfx.stride[0]=1;
a_hfx.lo[1]=jms;a_hfx.hi[1]=jme;a_hfx.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_qfx;a_qfx.data=qfx;
a_qfx.lo[0]=ims;a_qfx.hi[0]=ime;a_qfx.stride[0]=1;
a_qfx.lo[1]=jms;a_qfx.hi[1]=jme;a_qfx.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_wspd;a_wspd.data=wspd;
a_wspd.lo[0]=ims;a_wspd.hi[0]=ime;a_wspd.stride[0]=1;
a_wspd.lo[1]=jms;a_wspd.hi[1]=jme;a_wspd.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_znt;a_znt.data=znt;
a_znt.lo[0]=ims;a_znt.hi[0]=ime;a_znt.stride[0]=1;
a_znt.lo[1]=jms;a_znt.hi[1]=jme;a_znt.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_uoce;a_uoce.data=uoce;
a_uoce.lo[0]=ims;a_uoce.hi[0]=ime;a_uoce.stride[0]=1;
a_uoce.lo[1]=jms;a_uoce.hi[1]=jme;a_uoce.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_voce;a_voce.data=voce;
a_voce.lo[0]=ims;a_voce.hi[0]=ime;a_voce.stride[0]=1;
a_voce.lo[1]=jms;a_voce.hi[1]=jme;a_voce.stride[1]=1*(ime-ims+1);
MynnArray<float,3> a_dz;a_dz.data=dz;
a_dz.lo[0]=ims;a_dz.hi[0]=ime;a_dz.stride[0]=1;
a_dz.lo[1]=kms;a_dz.hi[1]=kme;a_dz.stride[1]=1*(ime-ims+1);
a_dz.lo[2]=jms;a_dz.hi[2]=jme;a_dz.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_u;a_u.data=u;
a_u.lo[0]=ims;a_u.hi[0]=ime;a_u.stride[0]=1;
a_u.lo[1]=kms;a_u.hi[1]=kme;a_u.stride[1]=1*(ime-ims+1);
a_u.lo[2]=jms;a_u.hi[2]=jme;a_u.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_v;a_v.data=v;
a_v.lo[0]=ims;a_v.hi[0]=ime;a_v.stride[0]=1;
a_v.lo[1]=kms;a_v.hi[1]=kme;a_v.stride[1]=1*(ime-ims+1);
a_v.lo[2]=jms;a_v.hi[2]=jme;a_v.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_w;a_w.data=w;
a_w.lo[0]=ims;a_w.hi[0]=ime;a_w.stride[0]=1;
a_w.lo[1]=kms;a_w.hi[1]=kme;a_w.stride[1]=1*(ime-ims+1);
a_w.lo[2]=jms;a_w.hi[2]=jme;a_w.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_th;a_th.data=th;
a_th.lo[0]=ims;a_th.hi[0]=ime;a_th.stride[0]=1;
a_th.lo[1]=kms;a_th.hi[1]=kme;a_th.stride[1]=1*(ime-ims+1);
a_th.lo[2]=jms;a_th.hi[2]=jme;a_th.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_tt;a_tt.data=tt;
a_tt.lo[0]=ims;a_tt.hi[0]=ime;a_tt.stride[0]=1;
a_tt.lo[1]=kms;a_tt.hi[1]=kme;a_tt.stride[1]=1*(ime-ims+1);
a_tt.lo[2]=jms;a_tt.hi[2]=jme;a_tt.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_p;a_p.data=p;
a_p.lo[0]=ims;a_p.hi[0]=ime;a_p.stride[0]=1;
a_p.lo[1]=kms;a_p.hi[1]=kme;a_p.stride[1]=1*(ime-ims+1);
a_p.lo[2]=jms;a_p.hi[2]=jme;a_p.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_exner;a_exner.data=exner;
a_exner.lo[0]=ims;a_exner.hi[0]=ime;a_exner.stride[0]=1;
a_exner.lo[1]=kms;a_exner.hi[1]=kme;a_exner.stride[1]=1*(ime-ims+1);
a_exner.lo[2]=jms;a_exner.hi[2]=jme;a_exner.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rho;a_rho.data=rho;
a_rho.lo[0]=ims;a_rho.hi[0]=ime;a_rho.stride[0]=1;
a_rho.lo[1]=kms;a_rho.hi[1]=kme;a_rho.stride[1]=1*(ime-ims+1);
a_rho.lo[2]=jms;a_rho.hi[2]=jme;a_rho.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qv;a_qv.data=qv;
a_qv.lo[0]=ims;a_qv.hi[0]=ime;a_qv.stride[0]=1;
a_qv.lo[1]=kms;a_qv.hi[1]=kme;a_qv.stride[1]=1*(ime-ims+1);
a_qv.lo[2]=jms;a_qv.hi[2]=jme;a_qv.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qc;a_qc.data=qc;
a_qc.lo[0]=ims;a_qc.hi[0]=ime;a_qc.stride[0]=1;
a_qc.lo[1]=kms;a_qc.hi[1]=kme;a_qc.stride[1]=1*(ime-ims+1);
a_qc.lo[2]=jms;a_qc.hi[2]=jme;a_qc.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qi;a_qi.data=qi;
a_qi.lo[0]=ims;a_qi.hi[0]=ime;a_qi.stride[0]=1;
a_qi.lo[1]=kms;a_qi.hi[1]=kme;a_qi.stride[1]=1*(ime-ims+1);
a_qi.lo[2]=jms;a_qi.hi[2]=jme;a_qi.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qs;a_qs.data=qs;
a_qs.lo[0]=ims;a_qs.hi[0]=ime;a_qs.stride[0]=1;
a_qs.lo[1]=kms;a_qs.hi[1]=kme;a_qs.stride[1]=1*(ime-ims+1);
a_qs.lo[2]=jms;a_qs.hi[2]=jme;a_qs.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_nc;a_nc.data=nc;
a_nc.lo[0]=ims;a_nc.hi[0]=ime;a_nc.stride[0]=1;
a_nc.lo[1]=kms;a_nc.hi[1]=kme;a_nc.stride[1]=1*(ime-ims+1);
a_nc.lo[2]=jms;a_nc.hi[2]=jme;a_nc.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_ni;a_ni.data=ni;
a_ni.lo[0]=ims;a_ni.hi[0]=ime;a_ni.stride[0]=1;
a_ni.lo[1]=kms;a_ni.hi[1]=kme;a_ni.stride[1]=1*(ime-ims+1);
a_ni.lo[2]=jms;a_ni.hi[2]=jme;a_ni.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_nifa;a_nifa.data=nifa;
a_nifa.lo[0]=ims;a_nifa.hi[0]=ime;a_nifa.stride[0]=1;
a_nifa.lo[1]=kms;a_nifa.hi[1]=kme;a_nifa.stride[1]=1*(ime-ims+1);
a_nifa.lo[2]=jms;a_nifa.hi[2]=jme;a_nifa.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_nwfa;a_nwfa.data=nwfa;
a_nwfa.lo[0]=ims;a_nwfa.hi[0]=ime;a_nwfa.stride[0]=1;
a_nwfa.lo[1]=kms;a_nwfa.hi[1]=kme;a_nwfa.stride[1]=1*(ime-ims+1);
a_nwfa.lo[2]=jms;a_nwfa.hi[2]=jme;a_nwfa.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_nbca;a_nbca.data=nbca;
a_nbca.lo[0]=ims;a_nbca.hi[0]=ime;a_nbca.stride[0]=1;
a_nbca.lo[1]=kms;a_nbca.hi[1]=kme;a_nbca.stride[1]=1*(ime-ims+1);
a_nbca.lo[2]=jms;a_nbca.hi[2]=jme;a_nbca.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qoz;a_qoz.data=qoz;
a_qoz.lo[0]=ims;a_qoz.hi[0]=ime;a_qoz.stride[0]=1;
a_qoz.lo[1]=kms;a_qoz.hi[1]=kme;a_qoz.stride[1]=1*(ime-ims+1);
a_qoz.lo[2]=jms;a_qoz.hi[2]=jme;a_qoz.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rthraten;a_rthraten.data=rthraten;
a_rthraten.lo[0]=ims;a_rthraten.hi[0]=ime;a_rthraten.stride[0]=1;
a_rthraten.lo[1]=kms;a_rthraten.hi[1]=kme;a_rthraten.stride[1]=1*(ime-ims+1);
a_rthraten.lo[2]=jms;a_rthraten.hi[2]=jme;a_rthraten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,2> a_pblh;a_pblh.data=pblh;
a_pblh.lo[0]=ims;a_pblh.hi[0]=ime;a_pblh.stride[0]=1;
a_pblh.lo[1]=jms;a_pblh.hi[1]=jme;a_pblh.stride[1]=1*(ime-ims+1);
MynnArray<int,2> a_kpbl;a_kpbl.data=kpbl;
a_kpbl.lo[0]=ims;a_kpbl.hi[0]=ime;a_kpbl.stride[0]=1;
a_kpbl.lo[1]=jms;a_kpbl.hi[1]=jme;a_kpbl.stride[1]=1*(ime-ims+1);
MynnArray<float,3> a_cldfra_bl;a_cldfra_bl.data=cldfra_bl;
a_cldfra_bl.lo[0]=ims;a_cldfra_bl.hi[0]=ime;a_cldfra_bl.stride[0]=1;
a_cldfra_bl.lo[1]=kms;a_cldfra_bl.hi[1]=kme;a_cldfra_bl.stride[1]=1*(ime-ims+1);
a_cldfra_bl.lo[2]=jms;a_cldfra_bl.hi[2]=jme;a_cldfra_bl.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qc_bl;a_qc_bl.data=qc_bl;
a_qc_bl.lo[0]=ims;a_qc_bl.hi[0]=ime;a_qc_bl.stride[0]=1;
a_qc_bl.lo[1]=kms;a_qc_bl.hi[1]=kme;a_qc_bl.stride[1]=1*(ime-ims+1);
a_qc_bl.lo[2]=jms;a_qc_bl.hi[2]=jme;a_qc_bl.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qi_bl;a_qi_bl.data=qi_bl;
a_qi_bl.lo[0]=ims;a_qi_bl.hi[0]=ime;a_qi_bl.stride[0]=1;
a_qi_bl.lo[1]=kms;a_qi_bl.hi[1]=kme;a_qi_bl.stride[1]=1*(ime-ims+1);
a_qi_bl.lo[2]=jms;a_qi_bl.hi[2]=jme;a_qi_bl.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,2> a_maxwidth;a_maxwidth.data=maxwidth;
a_maxwidth.lo[0]=ims;a_maxwidth.hi[0]=ime;a_maxwidth.stride[0]=1;
a_maxwidth.lo[1]=jms;a_maxwidth.hi[1]=jme;a_maxwidth.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_maxmf;a_maxmf.data=maxmf;
a_maxmf.lo[0]=ims;a_maxmf.hi[0]=ime;a_maxmf.stride[0]=1;
a_maxmf.lo[1]=jms;a_maxmf.hi[1]=jme;a_maxmf.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_ztop_plume;a_ztop_plume.data=ztop_plume;
a_ztop_plume.lo[0]=ims;a_ztop_plume.hi[0]=ime;a_ztop_plume.stride[0]=1;
a_ztop_plume.lo[1]=jms;a_ztop_plume.hi[1]=jme;a_ztop_plume.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_excess_h;a_excess_h.data=excess_h;
a_excess_h.lo[0]=ims;a_excess_h.hi[0]=ime;a_excess_h.stride[0]=1;
a_excess_h.lo[1]=jms;a_excess_h.hi[1]=jme;a_excess_h.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_excess_q;a_excess_q.data=excess_q;
a_excess_q.lo[0]=ims;a_excess_q.hi[0]=ime;a_excess_q.stride[0]=1;
a_excess_q.lo[1]=jms;a_excess_q.hi[1]=jme;a_excess_q.stride[1]=1*(ime-ims+1);
MynnArray<float,3> a_qke;a_qke.data=qke;
a_qke.lo[0]=ims;a_qke.hi[0]=ime;a_qke.stride[0]=1;
a_qke.lo[1]=kms;a_qke.hi[1]=kme;a_qke.stride[1]=1*(ime-ims+1);
a_qke.lo[2]=jms;a_qke.hi[2]=jme;a_qke.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qke_adv;a_qke_adv.data=qke_adv;
a_qke_adv.lo[0]=ims;a_qke_adv.hi[0]=ime;a_qke_adv.stride[0]=1;
a_qke_adv.lo[1]=kms;a_qke_adv.hi[1]=kme;a_qke_adv.stride[1]=1*(ime-ims+1);
a_qke_adv.lo[2]=jms;a_qke_adv.hi[2]=jme;a_qke_adv.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_tsq;a_tsq.data=tsq;
a_tsq.lo[0]=ims;a_tsq.hi[0]=ime;a_tsq.stride[0]=1;
a_tsq.lo[1]=kms;a_tsq.hi[1]=kme;a_tsq.stride[1]=1*(ime-ims+1);
a_tsq.lo[2]=jms;a_tsq.hi[2]=jme;a_tsq.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qsq;a_qsq.data=qsq;
a_qsq.lo[0]=ims;a_qsq.hi[0]=ime;a_qsq.stride[0]=1;
a_qsq.lo[1]=kms;a_qsq.hi[1]=kme;a_qsq.stride[1]=1*(ime-ims+1);
a_qsq.lo[2]=jms;a_qsq.hi[2]=jme;a_qsq.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_cov;a_cov.data=cov;
a_cov.lo[0]=ims;a_cov.hi[0]=ime;a_cov.stride[0]=1;
a_cov.lo[1]=kms;a_cov.hi[1]=kme;a_cov.stride[1]=1*(ime-ims+1);
a_cov.lo[2]=jms;a_cov.hi[2]=jme;a_cov.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_el_pbl;a_el_pbl.data=el_pbl;
a_el_pbl.lo[0]=ims;a_el_pbl.hi[0]=ime;a_el_pbl.stride[0]=1;
a_el_pbl.lo[1]=kms;a_el_pbl.hi[1]=kme;a_el_pbl.stride[1]=1*(ime-ims+1);
a_el_pbl.lo[2]=jms;a_el_pbl.hi[2]=jme;a_el_pbl.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rublten;a_rublten.data=rublten;
a_rublten.lo[0]=ims;a_rublten.hi[0]=ime;a_rublten.stride[0]=1;
a_rublten.lo[1]=kms;a_rublten.hi[1]=kme;a_rublten.stride[1]=1*(ime-ims+1);
a_rublten.lo[2]=jms;a_rublten.hi[2]=jme;a_rublten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rvblten;a_rvblten.data=rvblten;
a_rvblten.lo[0]=ims;a_rvblten.hi[0]=ime;a_rvblten.stride[0]=1;
a_rvblten.lo[1]=kms;a_rvblten.hi[1]=kme;a_rvblten.stride[1]=1*(ime-ims+1);
a_rvblten.lo[2]=jms;a_rvblten.hi[2]=jme;a_rvblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rthblten;a_rthblten.data=rthblten;
a_rthblten.lo[0]=ims;a_rthblten.hi[0]=ime;a_rthblten.stride[0]=1;
a_rthblten.lo[1]=kms;a_rthblten.hi[1]=kme;a_rthblten.stride[1]=1*(ime-ims+1);
a_rthblten.lo[2]=jms;a_rthblten.hi[2]=jme;a_rthblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rqvblten;a_rqvblten.data=rqvblten;
a_rqvblten.lo[0]=ims;a_rqvblten.hi[0]=ime;a_rqvblten.stride[0]=1;
a_rqvblten.lo[1]=kms;a_rqvblten.hi[1]=kme;a_rqvblten.stride[1]=1*(ime-ims+1);
a_rqvblten.lo[2]=jms;a_rqvblten.hi[2]=jme;a_rqvblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rqcblten;a_rqcblten.data=rqcblten;
a_rqcblten.lo[0]=ims;a_rqcblten.hi[0]=ime;a_rqcblten.stride[0]=1;
a_rqcblten.lo[1]=kms;a_rqcblten.hi[1]=kme;a_rqcblten.stride[1]=1*(ime-ims+1);
a_rqcblten.lo[2]=jms;a_rqcblten.hi[2]=jme;a_rqcblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rqiblten;a_rqiblten.data=rqiblten;
a_rqiblten.lo[0]=ims;a_rqiblten.hi[0]=ime;a_rqiblten.stride[0]=1;
a_rqiblten.lo[1]=kms;a_rqiblten.hi[1]=kme;a_rqiblten.stride[1]=1*(ime-ims+1);
a_rqiblten.lo[2]=jms;a_rqiblten.hi[2]=jme;a_rqiblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rqsblten;a_rqsblten.data=rqsblten;
a_rqsblten.lo[0]=ims;a_rqsblten.hi[0]=ime;a_rqsblten.stride[0]=1;
a_rqsblten.lo[1]=kms;a_rqsblten.hi[1]=kme;a_rqsblten.stride[1]=1*(ime-ims+1);
a_rqsblten.lo[2]=jms;a_rqsblten.hi[2]=jme;a_rqsblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rncblten;a_rncblten.data=rncblten;
a_rncblten.lo[0]=ims;a_rncblten.hi[0]=ime;a_rncblten.stride[0]=1;
a_rncblten.lo[1]=kms;a_rncblten.hi[1]=kme;a_rncblten.stride[1]=1*(ime-ims+1);
a_rncblten.lo[2]=jms;a_rncblten.hi[2]=jme;a_rncblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rniblten;a_rniblten.data=rniblten;
a_rniblten.lo[0]=ims;a_rniblten.hi[0]=ime;a_rniblten.stride[0]=1;
a_rniblten.lo[1]=kms;a_rniblten.hi[1]=kme;a_rniblten.stride[1]=1*(ime-ims+1);
a_rniblten.lo[2]=jms;a_rniblten.hi[2]=jme;a_rniblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rnifablten;a_rnifablten.data=rnifablten;
a_rnifablten.lo[0]=ims;a_rnifablten.hi[0]=ime;a_rnifablten.stride[0]=1;
a_rnifablten.lo[1]=kms;a_rnifablten.hi[1]=kme;a_rnifablten.stride[1]=1*(ime-ims+1);
a_rnifablten.lo[2]=jms;a_rnifablten.hi[2]=jme;a_rnifablten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rnwfablten;a_rnwfablten.data=rnwfablten;
a_rnwfablten.lo[0]=ims;a_rnwfablten.hi[0]=ime;a_rnwfablten.stride[0]=1;
a_rnwfablten.lo[1]=kms;a_rnwfablten.hi[1]=kme;a_rnwfablten.stride[1]=1*(ime-ims+1);
a_rnwfablten.lo[2]=jms;a_rnwfablten.hi[2]=jme;a_rnwfablten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rnbcablten;a_rnbcablten.data=rnbcablten;
a_rnbcablten.lo[0]=ims;a_rnbcablten.hi[0]=ime;a_rnbcablten.stride[0]=1;
a_rnbcablten.lo[1]=kms;a_rnbcablten.hi[1]=kme;a_rnbcablten.stride[1]=1*(ime-ims+1);
a_rnbcablten.lo[2]=jms;a_rnbcablten.hi[2]=jme;a_rnbcablten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rqozblten;a_rqozblten.data=rqozblten;
a_rqozblten.lo[0]=ims;a_rqozblten.hi[0]=ime;a_rqozblten.stride[0]=1;
a_rqozblten.lo[1]=kms;a_rqozblten.hi[1]=kme;a_rqozblten.stride[1]=1*(ime-ims+1);
a_rqozblten.lo[2]=jms;a_rqozblten.hi[2]=jme;a_rqozblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_edmf_a;a_edmf_a.data=edmf_a;
a_edmf_a.lo[0]=ims;a_edmf_a.hi[0]=ime;a_edmf_a.stride[0]=1;
a_edmf_a.lo[1]=kms;a_edmf_a.hi[1]=kme;a_edmf_a.stride[1]=1*(ime-ims+1);
a_edmf_a.lo[2]=jms;a_edmf_a.hi[2]=jme;a_edmf_a.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_edmf_w;a_edmf_w.data=edmf_w;
a_edmf_w.lo[0]=ims;a_edmf_w.hi[0]=ime;a_edmf_w.stride[0]=1;
a_edmf_w.lo[1]=kms;a_edmf_w.hi[1]=kme;a_edmf_w.stride[1]=1*(ime-ims+1);
a_edmf_w.lo[2]=jms;a_edmf_w.hi[2]=jme;a_edmf_w.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_edmf_qt;a_edmf_qt.data=edmf_qt;
a_edmf_qt.lo[0]=ims;a_edmf_qt.hi[0]=ime;a_edmf_qt.stride[0]=1;
a_edmf_qt.lo[1]=kms;a_edmf_qt.hi[1]=kme;a_edmf_qt.stride[1]=1*(ime-ims+1);
a_edmf_qt.lo[2]=jms;a_edmf_qt.hi[2]=jme;a_edmf_qt.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_edmf_thl;a_edmf_thl.data=edmf_thl;
a_edmf_thl.lo[0]=ims;a_edmf_thl.hi[0]=ime;a_edmf_thl.stride[0]=1;
a_edmf_thl.lo[1]=kms;a_edmf_thl.hi[1]=kme;a_edmf_thl.stride[1]=1*(ime-ims+1);
a_edmf_thl.lo[2]=jms;a_edmf_thl.hi[2]=jme;a_edmf_thl.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_edmf_ent;a_edmf_ent.data=edmf_ent;
a_edmf_ent.lo[0]=ims;a_edmf_ent.hi[0]=ime;a_edmf_ent.stride[0]=1;
a_edmf_ent.lo[1]=kms;a_edmf_ent.hi[1]=kme;a_edmf_ent.stride[1]=1*(ime-ims+1);
a_edmf_ent.lo[2]=jms;a_edmf_ent.hi[2]=jme;a_edmf_ent.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_edmf_qc;a_edmf_qc.data=edmf_qc;
a_edmf_qc.lo[0]=ims;a_edmf_qc.hi[0]=ime;a_edmf_qc.stride[0]=1;
a_edmf_qc.lo[1]=kms;a_edmf_qc.hi[1]=kme;a_edmf_qc.stride[1]=1*(ime-ims+1);
a_edmf_qc.lo[2]=jms;a_edmf_qc.hi[2]=jme;a_edmf_qc.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_sub_thl;a_sub_thl.data=sub_thl;
a_sub_thl.lo[0]=ims;a_sub_thl.hi[0]=ime;a_sub_thl.stride[0]=1;
a_sub_thl.lo[1]=kms;a_sub_thl.hi[1]=kme;a_sub_thl.stride[1]=1*(ime-ims+1);
a_sub_thl.lo[2]=jms;a_sub_thl.hi[2]=jme;a_sub_thl.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_sub_sqv;a_sub_sqv.data=sub_sqv;
a_sub_sqv.lo[0]=ims;a_sub_sqv.hi[0]=ime;a_sub_sqv.stride[0]=1;
a_sub_sqv.lo[1]=kms;a_sub_sqv.hi[1]=kme;a_sub_sqv.stride[1]=1*(ime-ims+1);
a_sub_sqv.lo[2]=jms;a_sub_sqv.hi[2]=jme;a_sub_sqv.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_det_thl;a_det_thl.data=det_thl;
a_det_thl.lo[0]=ims;a_det_thl.hi[0]=ime;a_det_thl.stride[0]=1;
a_det_thl.lo[1]=kms;a_det_thl.hi[1]=kme;a_det_thl.stride[1]=1*(ime-ims+1);
a_det_thl.lo[2]=jms;a_det_thl.hi[2]=jme;a_det_thl.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_det_sqv;a_det_sqv.data=det_sqv;
a_det_sqv.lo[0]=ims;a_det_sqv.hi[0]=ime;a_det_sqv.stride[0]=1;
a_det_sqv.lo[1]=kms;a_det_sqv.hi[1]=kme;a_det_sqv.stride[1]=1*(ime-ims+1);
a_det_sqv.lo[2]=jms;a_det_sqv.hi[2]=jme;a_det_sqv.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_exch_h;a_exch_h.data=exch_h;
a_exch_h.lo[0]=ims;a_exch_h.hi[0]=ime;a_exch_h.stride[0]=1;
a_exch_h.lo[1]=kms;a_exch_h.hi[1]=kme;a_exch_h.stride[1]=1*(ime-ims+1);
a_exch_h.lo[2]=jms;a_exch_h.hi[2]=jme;a_exch_h.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_exch_m;a_exch_m.data=exch_m;
a_exch_m.lo[0]=ims;a_exch_m.hi[0]=ime;a_exch_m.stride[0]=1;
a_exch_m.lo[1]=kms;a_exch_m.hi[1]=kme;a_exch_m.stride[1]=1*(ime-ims+1);
a_exch_m.lo[2]=jms;a_exch_m.hi[2]=jme;a_exch_m.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_dqke;a_dqke.data=dqke;
a_dqke.lo[0]=ims;a_dqke.hi[0]=ime;a_dqke.stride[0]=1;
a_dqke.lo[1]=kms;a_dqke.hi[1]=kme;a_dqke.stride[1]=1*(ime-ims+1);
a_dqke.lo[2]=jms;a_dqke.hi[2]=jme;a_dqke.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qwt;a_qwt.data=qwt;
a_qwt.lo[0]=ims;a_qwt.hi[0]=ime;a_qwt.stride[0]=1;
a_qwt.lo[1]=kms;a_qwt.hi[1]=kme;a_qwt.stride[1]=1*(ime-ims+1);
a_qwt.lo[2]=jms;a_qwt.hi[2]=jme;a_qwt.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qshear;a_qshear.data=qshear;
a_qshear.lo[0]=ims;a_qshear.hi[0]=ime;a_qshear.stride[0]=1;
a_qshear.lo[1]=kms;a_qshear.hi[1]=kme;a_qshear.stride[1]=1*(ime-ims+1);
a_qshear.lo[2]=jms;a_qshear.hi[2]=jme;a_qshear.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qbuoy;a_qbuoy.data=qbuoy;
a_qbuoy.lo[0]=ims;a_qbuoy.hi[0]=ime;a_qbuoy.stride[0]=1;
a_qbuoy.lo[1]=kms;a_qbuoy.hi[1]=kme;a_qbuoy.stride[1]=1*(ime-ims+1);
a_qbuoy.lo[2]=jms;a_qbuoy.hi[2]=jme;a_qbuoy.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qdiss;a_qdiss.data=qdiss;
a_qdiss.lo[0]=ims;a_qdiss.hi[0]=ime;a_qdiss.stride[0]=1;
a_qdiss.lo[1]=kms;a_qdiss.hi[1]=kme;a_qdiss.stride[1]=1*(ime-ims+1);
a_qdiss.lo[2]=jms;a_qdiss.hi[2]=jme;a_qdiss.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_sh3d;a_sh3d.data=sh3d;
a_sh3d.lo[0]=ims;a_sh3d.hi[0]=ime;a_sh3d.stride[0]=1;
a_sh3d.lo[1]=kms;a_sh3d.hi[1]=kme;a_sh3d.stride[1]=1*(ime-ims+1);
a_sh3d.lo[2]=jms;a_sh3d.hi[2]=jme;a_sh3d.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_sm3d;a_sm3d.data=sm3d;
a_sm3d.lo[0]=ims;a_sm3d.hi[0]=ime;a_sm3d.stride[0]=1;
a_sm3d.lo[1]=kms;a_sm3d.hi[1]=kme;a_sm3d.stride[1]=1*(ime-ims+1);
a_sm3d.lo[2]=jms;a_sm3d.hi[2]=jme;a_sm3d.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_pattern_spp;a_pattern_spp.data=pattern_spp;
a_pattern_spp.lo[0]=ims;a_pattern_spp.hi[0]=ime;a_pattern_spp.stride[0]=1;
a_pattern_spp.lo[1]=kms;a_pattern_spp.hi[1]=kme;a_pattern_spp.stride[1]=1*(ime-ims+1);
a_pattern_spp.lo[2]=jms;a_pattern_spp.hi[2]=jme;a_pattern_spp.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,4> a_chem3d;a_chem3d.data=chem3d;
a_chem3d.lo[0]=ims;a_chem3d.hi[0]=ime;a_chem3d.stride[0]=1;
a_chem3d.lo[1]=kms;a_chem3d.hi[1]=kme;a_chem3d.stride[1]=1*(ime-ims+1);
a_chem3d.lo[2]=jms;a_chem3d.hi[2]=jme;a_chem3d.stride[2]=1*(ime-ims+1)*(kme-kms+1);
a_chem3d.lo[3]=1;a_chem3d.hi[3]=nchem;a_chem3d.stride[3]=1*(ime-ims+1)*(kme-kms+1)*(jme-jms+1);
MynnArray<float,4> a_settle3d;a_settle3d.data=settle3d;
a_settle3d.lo[0]=ims;a_settle3d.hi[0]=ime;a_settle3d.stride[0]=1;
a_settle3d.lo[1]=kms;a_settle3d.hi[1]=kme;a_settle3d.stride[1]=1*(ime-ims+1);
a_settle3d.lo[2]=jms;a_settle3d.hi[2]=jme;a_settle3d.stride[2]=1*(ime-ims+1)*(kme-kms+1);
a_settle3d.lo[3]=1;a_settle3d.hi[3]=nchem;a_settle3d.stride[3]=1*(ime-ims+1)*(kme-kms+1)*(jme-jms+1);
MynnArray<float,3> a_vd3d;a_vd3d.data=vd3d;
a_vd3d.lo[0]=ims;a_vd3d.hi[0]=ime;a_vd3d.stride[0]=1;
a_vd3d.lo[1]=jms;a_vd3d.hi[1]=jme;a_vd3d.stride[1]=1*(ime-ims+1);
a_vd3d.lo[2]=1;a_vd3d.hi[2]=ndvel;a_vd3d.stride[2]=1*(ime-ims+1)*(jme-jms+1);
MynnArray<float,2> a_frp_mean;a_frp_mean.data=frp_mean;
a_frp_mean.lo[0]=ims;a_frp_mean.hi[0]=ime;a_frp_mean.stride[0]=1;
a_frp_mean.lo[1]=jms;a_frp_mean.hi[1]=jme;a_frp_mean.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_emis_ant_no;a_emis_ant_no.data=emis_ant_no;
a_emis_ant_no.lo[0]=ims;a_emis_ant_no.hi[0]=ime;a_emis_ant_no.stride[0]=1;
a_emis_ant_no.lo[1]=jms;a_emis_ant_no.hi[1]=jme;a_emis_ant_no.stride[1]=1*(ime-ims+1);
column.f_mynnedmf_driver(ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,(f_qc!=0),(f_qi!=0),(f_qs!=0),(f_qoz!=0),(f_nc!=0),(f_ni!=0),(f_nifa!=0),(f_nwfa!=0),(f_nbca!=0),initflag,(do_restart!=0),(do_dacycling!=0),icloud_bl,delt,a_dx,a_xland,a_ps,a_ts,a_qsfc,a_ust,a_ch,a_hfx,a_qfx,a_wspd,a_znt,a_uoce,a_voce,a_dz,a_u,a_v,a_w,a_th,a_tt,a_p,a_exner,a_rho,a_qv,a_qc,a_qi,a_qs,a_nc,a_ni,a_nifa,a_nwfa,a_nbca,a_qoz,a_rthraten,a_pblh,a_kpbl,a_cldfra_bl,a_qc_bl,a_qi_bl,a_maxwidth,a_maxmf,a_ztop_plume,a_excess_h,a_excess_q,a_qke,a_qke_adv,a_tsq,a_qsq,a_cov,a_el_pbl,a_rublten,a_rvblten,a_rthblten,a_rqvblten,a_rqcblten,a_rqiblten,a_rqsblten,a_rncblten,a_rniblten,a_rnifablten,a_rnwfablten,a_rnbcablten,a_rqozblten,a_edmf_a,a_edmf_w,a_edmf_qt,a_edmf_thl,a_edmf_ent,a_edmf_qc,a_sub_thl,a_sub_sqv,a_det_thl,a_det_sqv,a_exch_h,a_exch_m,a_dqke,a_qwt,a_qshear,a_qbuoy,a_qdiss,a_sh3d,a_sm3d,spp_pbl,a_pattern_spp,(bl_mynn_tkeadvect!=0),bl_mynn_tkebudget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_stfunc,bl_mynn_topdown,bl_mynn_scaleaware,bl_mynn_dheat_opt,bl_mynn_edmf,bl_mynn_edmf_dd,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_output,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_ess,message,error_flag,(mix_chem!=0),nchem,ndvel,a_chem3d,a_settle3d,a_vd3d,(present_enh_mix ? MynnOptional<bool>(enh_mix) : MynnOptional<bool>()),a_frp_mean,a_emis_ant_no);
error_flags[index]=error_flag;
}
extern "C" __global__ void rrfs_mynnedmf_run_l1(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_nc,int f_ni,int f_nifa,int f_nwfa,int f_nbca,int initflag,int do_restart,int do_dacycling,int icloud_bl,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tt,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* nc,float* ni,float* nifa,float* nwfa,float* nbca,float* qoz,float* rthraten,float* pblh,int* kpbl,float* cldfra_bl,float* qc_bl,float* qi_bl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rncblten,float* rniblten,float* rnifablten,float* rnwfablten,float* rnbcablten,float* rqozblten,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int bl_mynn_tkebudget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_stfunc,int bl_mynn_topdown,int bl_mynn_scaleaware,int bl_mynn_dheat_opt,int bl_mynn_edmf,int bl_mynn_edmf_dd,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_output,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_ess,int mix_chem,int nchem,int ndvel,float* chem3d,float* settle3d,float* vd3d,int enh_mix,int present_enh_mix,float* frp_mean,float* emis_ant_no) { rrfs_mynnedmf_column<1>(batch_first,batch_count,workspace,workspace_stride,error_flags,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,f_qc,f_qi,f_qs,f_qoz,f_nc,f_ni,f_nifa,f_nwfa,f_nbca,initflag,do_restart,do_dacycling,icloud_bl,delt,dx,xland,ps,ts,qsfc,ust,ch,hfx,qfx,wspd,znt,uoce,voce,dz,u,v,w,th,tt,p,exner,rho,qv,qc,qi,qs,nc,ni,nifa,nwfa,nbca,qoz,rthraten,pblh,kpbl,cldfra_bl,qc_bl,qi_bl,maxwidth,maxmf,ztop_plume,excess_h,excess_q,qke,qke_adv,tsq,qsq,cov,el_pbl,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqsblten,rncblten,rniblten,rnifablten,rnwfablten,rnbcablten,rqozblten,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc,sub_thl,sub_sqv,det_thl,det_sqv,exch_h,exch_m,dqke,qwt,qshear,qbuoy,qdiss,sh3d,sm3d,spp_pbl,pattern_spp,bl_mynn_tkeadvect,bl_mynn_tkebudget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_stfunc,bl_mynn_topdown,bl_mynn_scaleaware,bl_mynn_dheat_opt,bl_mynn_edmf,bl_mynn_edmf_dd,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_output,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_ess,mix_chem,nchem,ndvel,chem3d,settle3d,vd3d,enh_mix,present_enh_mix,frp_mean,emis_ant_no); }
extern "C" __global__ MYNN_L4_REGS void rrfs_mynnedmf_run_l4(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_nc,int f_ni,int f_nifa,int f_nwfa,int f_nbca,int initflag,int do_restart,int do_dacycling,int icloud_bl,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tt,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* nc,float* ni,float* nifa,float* nwfa,float* nbca,float* qoz,float* rthraten,float* pblh,int* kpbl,float* cldfra_bl,float* qc_bl,float* qi_bl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rncblten,float* rniblten,float* rnifablten,float* rnwfablten,float* rnbcablten,float* rqozblten,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int bl_mynn_tkebudget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_stfunc,int bl_mynn_topdown,int bl_mynn_scaleaware,int bl_mynn_dheat_opt,int bl_mynn_edmf,int bl_mynn_edmf_dd,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_output,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_ess,int mix_chem,int nchem,int ndvel,float* chem3d,float* settle3d,float* vd3d,int enh_mix,int present_enh_mix,float* frp_mean,float* emis_ant_no) { rrfs_mynnedmf_column<4>(batch_first,batch_count,workspace,workspace_stride,error_flags,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,f_qc,f_qi,f_qs,f_qoz,f_nc,f_ni,f_nifa,f_nwfa,f_nbca,initflag,do_restart,do_dacycling,icloud_bl,delt,dx,xland,ps,ts,qsfc,ust,ch,hfx,qfx,wspd,znt,uoce,voce,dz,u,v,w,th,tt,p,exner,rho,qv,qc,qi,qs,nc,ni,nifa,nwfa,nbca,qoz,rthraten,pblh,kpbl,cldfra_bl,qc_bl,qi_bl,maxwidth,maxmf,ztop_plume,excess_h,excess_q,qke,qke_adv,tsq,qsq,cov,el_pbl,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqsblten,rncblten,rniblten,rnifablten,rnwfablten,rnbcablten,rqozblten,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc,sub_thl,sub_sqv,det_thl,det_sqv,exch_h,exch_m,dqke,qwt,qshear,qbuoy,qdiss,sh3d,sm3d,spp_pbl,pattern_spp,bl_mynn_tkeadvect,bl_mynn_tkebudget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_stfunc,bl_mynn_topdown,bl_mynn_scaleaware,bl_mynn_dheat_opt,bl_mynn_edmf,bl_mynn_edmf_dd,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_output,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_ess,mix_chem,nchem,ndvel,chem3d,settle3d,vd3d,enh_mix,present_enh_mix,frp_mean,emis_ant_no); }
extern "C" __global__ void rrfs_mynnedmf_run_l16(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_nc,int f_ni,int f_nifa,int f_nwfa,int f_nbca,int initflag,int do_restart,int do_dacycling,int icloud_bl,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tt,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* nc,float* ni,float* nifa,float* nwfa,float* nbca,float* qoz,float* rthraten,float* pblh,int* kpbl,float* cldfra_bl,float* qc_bl,float* qi_bl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rncblten,float* rniblten,float* rnifablten,float* rnwfablten,float* rnbcablten,float* rqozblten,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int bl_mynn_tkebudget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_stfunc,int bl_mynn_topdown,int bl_mynn_scaleaware,int bl_mynn_dheat_opt,int bl_mynn_edmf,int bl_mynn_edmf_dd,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_output,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_ess,int mix_chem,int nchem,int ndvel,float* chem3d,float* settle3d,float* vd3d,int enh_mix,int present_enh_mix,float* frp_mean,float* emis_ant_no) { rrfs_mynnedmf_column<16>(batch_first,batch_count,workspace,workspace_stride,error_flags,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,f_qc,f_qi,f_qs,f_qoz,f_nc,f_ni,f_nifa,f_nwfa,f_nbca,initflag,do_restart,do_dacycling,icloud_bl,delt,dx,xland,ps,ts,qsfc,ust,ch,hfx,qfx,wspd,znt,uoce,voce,dz,u,v,w,th,tt,p,exner,rho,qv,qc,qi,qs,nc,ni,nifa,nwfa,nbca,qoz,rthraten,pblh,kpbl,cldfra_bl,qc_bl,qi_bl,maxwidth,maxmf,ztop_plume,excess_h,excess_q,qke,qke_adv,tsq,qsq,cov,el_pbl,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqsblten,rncblten,rniblten,rnifablten,rnwfablten,rnbcablten,rqozblten,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc,sub_thl,sub_sqv,det_thl,det_sqv,exch_h,exch_m,dqke,qwt,qshear,qbuoy,qdiss,sh3d,sm3d,spp_pbl,pattern_spp,bl_mynn_tkeadvect,bl_mynn_tkebudget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_stfunc,bl_mynn_topdown,bl_mynn_scaleaware,bl_mynn_dheat_opt,bl_mynn_edmf,bl_mynn_edmf_dd,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_output,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_ess,mix_chem,nchem,ndvel,chem3d,settle3d,vd3d,enh_mix,present_enh_mix,frp_mean,emis_ant_no); }
'''

ARGUMENTS = {'ids': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ide': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jds': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jde': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kds': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kde': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ims': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ime': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jms': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jme': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kms': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kme': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'its': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ite': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jts': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jte': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kts': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kte': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qc': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qi': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qs': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qoz': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_nc': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_ni': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_nifa': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_nwfa': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_nbca': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'initflag': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'do_restart': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'do_dacycling': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'icloud_bl': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'delt': {'kind': 'float', 'dimensions': [], 'intent': 'in', 'optional': False}, 'dx': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'xland': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ps': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ts': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'qsfc': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ust': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ch': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'hfx': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'qfx': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'wspd': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'znt': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'uoce': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'voce': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'dz': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'u': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'v': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'w': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'th': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'tt': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'p': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'exner': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'rho': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'qv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'qc': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qi': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qs': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'nc': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'ni': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'nifa': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'nwfa': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'nbca': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qoz': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'rthraten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'pblh': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'kpbl': {'kind': 'int', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'cldfra_bl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qc_bl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qi_bl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'maxwidth': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'maxmf': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'ztop_plume': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'excess_h': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'excess_q': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'qke': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qke_adv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'tsq': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qsq': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'cov': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'el_pbl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rublten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rvblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rthblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rqvblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rqcblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqiblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqsblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rncblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rniblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rnifablten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rnwfablten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rnbcablten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqozblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_a': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_w': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_qt': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_thl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_ent': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_qc': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'sub_thl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'sub_sqv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'det_thl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'det_sqv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'exch_h': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': False}, 'exch_m': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': False}, 'dqke': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qwt': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qshear': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qbuoy': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qdiss': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'sh3d': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'sm3d': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'spp_pbl': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'pattern_spp': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'bl_mynn_tkeadvect': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_tkebudget': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_cloudpdf': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixlength': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_closure': {'kind': 'float', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_stfunc': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_topdown': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_scaleaware': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_dheat_opt': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_dd': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_mom': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_tke': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_output': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixscalars': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixaerosols': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixnumcon': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_cloudmix': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixqt': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_ess': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'errmsg': {'kind': 'char', 'dimensions': [], 'intent': 'out', 'optional': False}, 'errflg': {'kind': 'int', 'dimensions': [], 'intent': 'out', 'optional': False}, 'mix_chem': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'nchem': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ndvel': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'chem3d': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme', 'nchem'], 'intent': 'inout', 'optional': True}, 'settle3d': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme', 'nchem'], 'intent': 'inout', 'optional': True}, 'vd3d': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme', 'ndvel'], 'intent': 'in', 'optional': True}, 'enh_mix': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': True}, 'frp_mean': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': True}, 'emis_ant_no': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': True}}
WORKSPACE_ARRAYS = [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['n']}, {'kind': 'float', 'dimensions': ['kte']}, {'kind': 'float', 'dimensions': ['kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte', '1:nup']}, {'kind': 'float', 'dimensions': ['nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nscalars']}, {'kind': 'int', 'dimensions': ['1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'int', 'dimensions': ['1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['ndvel']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kme']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}]
WORKSPACE_FRAMES = {'mynnedmf': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['ddmp_mf', 'dmp_mf', 'get_pblh', 'mym_condensation', 'mym_initialize', 'mym_predict', 'mym_turbulence', 'mynn_mix_chem', 'mynn_mix_scalars', 'mynn_tendencies', 'phih', 'phim', 'retrieve_exchange_coeffs', 'scale_aware', 'topdown_cloudrad']}, 'mym_initialize': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['mym_length', 'mym_level2']}, 'mym_level2': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['qsat_blend', 'xl_blend']}, 'mym_length': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['boulac_length']}, 'boulac_length0': {'arrays': [], 'calls': []}, 'boulac_length': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': []}, 'mym_turbulence': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['mym_length', 'mym_level2']}, 'mym_predict': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['tridiag2']}, 'mym_condensation': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['esat_blend', 'qsat_blend', 'xl_blend']}, 'mynn_tendencies': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['moisture_check', 'qsat_blend', 'tridiag2']}, 'moisture_check': {'arrays': [], 'calls': []}, 'mynn_mix_chem': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['tridiag3']}, 'mynn_mix_scalars': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['tridiag3']}, 'retrieve_exchange_coeffs': {'arrays': [], 'calls': []}, 'tridiag': {'arrays': [{'kind': 'float', 'dimensions': ['n']}], 'calls': []}, 'tridiag2': {'arrays': [{'kind': 'float', 'dimensions': ['kte']}, {'kind': 'float', 'dimensions': ['kte']}], 'calls': []}, 'tridiag3': {'arrays': [], 'calls': []}, 'get_pblh': {'arrays': [], 'calls': []}, 'dmp_mf': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte', '1:nup']}, {'kind': 'float', 'dimensions': ['nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nscalars']}, {'kind': 'int', 'dimensions': ['1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['condensation_edmf', 'esat_blend', 'qsat_blend', 'xl_blend']}, 'condensation_edmf': {'arrays': [], 'calls': ['qsat_blend']}, 'condensation_ddmf': {'arrays': [], 'calls': ['qsat_blend']}, 'condensation_edmf_r': {'arrays': [], 'calls': ['qsat_blend']}, 'ddmp_mf': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'int', 'dimensions': ['1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}], 'calls': ['condensation_ddmf', 'qsat_blend']}, 'scale_aware': {'arrays': [], 'calls': []}, 'esat_blend': {'arrays': [], 'calls': []}, 'qsat_blend': {'arrays': [], 'calls': []}, 'xl_blend': {'arrays': [], 'calls': []}, 'phim': {'arrays': [], 'calls': []}, 'phih': {'arrays': [], 'calls': []}, 'topdown_cloudrad': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': []}, 'mynnedmf_driver': {'arrays': [{'kind': 'float', 'dimensions': ['ndvel']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kme']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['mynnedmf', 'mynnedmf_post_run', 'mynnedmf_pre_run']}, 'mynnedmf_pre_init': {'arrays': [], 'calls': []}, 'mynnedmf_pre_finalize': {'arrays': [], 'calls': []}, 'mynnedmf_pre_run': {'arrays': [], 'calls': []}, 'mynnedmf_post_init': {'arrays': [], 'calls': []}, 'mynnedmf_post_finalize': {'arrays': [], 'calls': []}, 'mynnedmf_post_run': {'arrays': [{'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}], 'calls': []}}
# END GENERATED RRFS MYNN-EDMF
# RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1535-1663
# MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:2841-2939; these are Registry defaults, not suite overrides.
OPTION_ARGUMENTS = {
    'config_mynn_tkeadvect': ('bl_mynn_tkeadvect', False),
    'config_mynn_tkebudget': ('bl_mynn_tkebudget', 0),
    'config_mynn_closure': ('bl_mynn_closure', 2.6),
    'config_mynn_cloudpdf': ('bl_mynn_cloudpdf', 2),
    'config_mynn_mixlength': ('bl_mynn_mixlength', 1),
    'config_mynn_stfunc': ('bl_mynn_stfunc', 1),
    'config_mynn_topdown': ('bl_mynn_topdown', 0),
    'config_mynn_scaleaware': ('bl_mynn_scaleaware', 1),
    'config_mynn_dheat_opt': ('bl_mynn_dheat_opt', 1),
    'config_mynn_edmf': ('bl_mynn_edmf', 1),
    'config_mynn_edmf_dd': ('bl_mynn_edmf_dd', 0),
    'config_mynn_edmf_mom': ('bl_mynn_edmf_mom', 1),
    'config_mynn_edmf_tke': ('bl_mynn_edmf_tke', 0),
    'config_mynn_edmf_output': ('bl_mynn_output', 0),
    'config_mynn_mixscalars': ('bl_mynn_mixscalars', 0),
    'config_mynn_mixaerosols': ('bl_mynn_mixaerosols', 1),
    'config_mynn_mixnumcon': ('bl_mynn_mixnumcon', 1),
    'config_mynn_mixclouds': ('bl_mynn_cloudmix', 1),
    'config_mynn_mixqt': ('bl_mynn_mixqt', 0),
    'config_mynn_ess': ('bl_mynn_ess', 1),
    'config_mynn_enh_mix': ('enh_mix', False),
    'config_icloud_bl': ('icloud_bl', 1),
    'config_spp_pbl': ('spp_pbl', 0),
}


def option_arguments(config: Mapping) -> dict:
    """Fork options; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1535-1663."""
    return {arg: config.get(name, default)
            for name, (arg, default) in OPTION_ARGUMENTS.items()}


def argument_shape(metadata: Mapping, arguments: Mapping) -> tuple[int, ...]:
    """Fortran extents; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:118-284."""
    sizes = []
    for dim in metadata['dimensions']:
        edges = dim.split(':')
        lower = int(arguments[edges[0]]) if len(edges) == 2 else 1
        upper = int(arguments[edges[-1]])
        sizes.append(upper - lower + 1)
    return tuple(sizes)


#: Device memory one MYNN-EDMF launch's automatic-storage arena may take (WOOF_HEX_MYNNEDMF_ARENA_MB overrides).
ARENA_BUDGET_BYTES = 2 << 30


#: Lane widths compiled into the image (generate_backend.LANE_WIDTHS): one kernel rrfs_mynnedmf_run_l<W> each.
LANE_WIDTHS = (1, 4, 16)


def lanes_per_column(kernels, ncols: int) -> int:
    """Use W=16 for a single wave, measured W=4 throughput for larger fast tiles.

    Real call24, 102,480 tiled columns on sm_120: W=4 beats W=1 even
    when its warps take several waves (44.78 versus 59.40 ms/100k before
    register tuning). The old all-warps-must-fit rule chose W=1 there.
    W=16 remains the measured winner for 2,562 columns. Strict retains the
    occupancy rule: its uncapped W=1 beats W=4 on the large replay. The environment
    override retains every compiled width for replay and other cards.
    """
    import os
    forced = int(os.environ.get('WOOF_HEX_MYNNEDMF_LANES', '0') or 0)
    if forced:
        if forced not in LANE_WIDTHS:
            raise ValueError(f'WOOF_HEX_MYNNEDMF_LANES must be one of {LANE_WIDTHS}')
        return forced
    import cupy as cp
    from cupy.cuda import driver
    sms = int(cp.cuda.Device().attributes['MultiProcessorCount'])
    fast = _hex_numerics.mode() == 'fast'
    for width in ((16,) if fast else sorted(LANE_WIDTHS, reverse=True)):
        capacity = sms * max(int(driver.occupancyMaxActiveBlocksPerMultiprocessor(kernels[width].ptr, 32, 0)), 1)
        if -(-ncols * width // 32) <= capacity:
            return width
    return 4 if fast else 1


def automatic_batch_columns(ncols: int, stride: int) -> int:
    """Columns per MYNN-EDMF launch: every column at once when its arenas fit the budget, else whole warps.

    THE BREAKAGE THIS PREVENTS: the wrapper launched 32 columns at a time (one warp, one block), so the PBL ran
    as ncols/32 back-to-back single-warp launches: 37.6 s per call on x1.2562 (2,562 columns, RTX 5090, strict
    numerics) against 0.065 s for native bl_mynnedmf on one CPU core (hrrrv5 24 h gnu-strict reference,
    2026-10-28), and it would have grown with every column of a 3 km mesh.
    """
    import os
    budget = int(os.environ.get('WOOF_HEX_MYNNEDMF_ARENA_MB', '0') or 0) << 20 or ARENA_BUDGET_BYTES
    try:
        import cupy as cp
        budget = min(budget, int(cp.cuda.runtime.memGetInfo()[0]) // 4)
    except Exception:
        pass
    fit = max(32, (budget // max(int(stride), 1)) // 32 * 32)
    return int(min(ncols, fit)) if ncols > 0 else 1


def workspace_bytes(arguments: Mapping) -> int:
    """Automatic storage; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:288-323."""
    from math import prod
    bounds = {n: int(arguments[n]) for n in ('kts', 'kte', 'kme', 'nchem', 'ndvel')}
    bounds.update(n=max(bounds['kte'], bounds['kme']), nup=8, ndd=5, nscalars=1)
    def value(text):
        """Pinned extents; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:288-323."""
        # Only these pinned declaration expressions occur; no arbitrary eval.
        parts = text.split('+')
        return sum(int(x) if x.isdecimal() else bounds[x] for x in parts)
    def frame_bytes(arrays):
        total = 0
        for var in arrays:
            extents = []
            for dim in var['dimensions']:
                edges = dim.split(':')
                lo = value(edges[0]) if len(edges) == 2 else 1
                extents.append(max(0, value(edges[-1]) - lo + 1))
            size = {'float': 4, 'int': 4, 'double': 8, 'bool': 1}[var['kind']]
            total = ((total + 7) & ~7) + size * prod(extents)
        return (total + 7) & ~7

    # MynnFrame restores the cursor on every return. Only the largest live
    # call path is needed, not the sum of all sibling routines. For 59 levels
    # this is 71,528 rather than 117,488 bytes/column: 30,016 rather than 18,272
    # columns in the 2 GiB budget. All optional branches stay in the bound.
    peaks = {}
    visiting = set()
    def peak(name):
        if name in visiting:
            raise ValueError('Recursive MYNN arena call graph: ' + name)
        if name not in peaks:
            visiting.add(name)
            frame = WORKSPACE_FRAMES[name]
            peaks[name] = frame_bytes(frame['arrays']) + max(
                (peak(child) for child in frame['calls']), default=0)
            visiting.remove(name)
        return peaks[name]
    return peak('mynnedmf_driver')


def wrapper_allocation_layout(arguments: Mapping) -> dict:
    """Captured entry storage; RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:288-323.

    This describes the exact arena allocations of the translated wrapper.
    Native tap entry records may restore automatic storage before execution;
    expected output records never supply initial storage.
    """
    import re
    from math import prod
    body = MODULE_CUDA.split(' void f_mynnedmf_driver(', 1)[1]
    bounds = {n: int(arguments[n]) for n in ('kts', 'kte', 'kme', 'nchem', 'ndvel')}
    bounds['nscalars'] = 1
    pattern = r'MynnArray<(float|int|double), (\d+)> (\w+) = mynn_array<[^>]+>\(arena, \{([^}]+)\}, \{([^}]+)\}\);'
    layout = {}
    offset = 0
    # The next routine starts after all of the wrapper's declarations/body.
    body = body.split('__device__ __noinline__ void f_', 1)[0]
    def extent_value(expression):
        return sum(int(v) if v.isdecimal() else bounds[v] for v in expression.strip().split('+'))
    for kind, rank, name, lower, upper in re.findall(pattern, body):
        low = [extent_value(v) for v in lower.split(',')]
        high = [extent_value(v) for v in upper.split(',')]
        shape = tuple(max(0, h-l+1) for l,h in zip(low, high))
        size = prod(shape) * {'float':4, 'int':4, 'double':8}[kind]
        offset = (offset+7) & ~7
        layout[name] = (offset, size, shape)
        offset += size
    if not layout or 'dqke1' not in layout:
        raise RuntimeError('Translated wrapper allocation layout is incomplete')
    return layout


class MynnEdmfCuda:
    """Launch the complete original MPAS driver on resident device arrays.

    ``shared_math_source`` must implement the mynn_* libm overloads used by
    MODULE_CUDA. It is supplied by the shared maths seam. No CUDA-libm fallback
    is substituted. Arrays retain the original Fortran (i,k,j[,species]) layout.
    Compilation uses NVRTC directly to preserve --ftz=false.
    """

    def __init__(self, shared_math_source: str | None = None, *, batch_columns: int | None = None,
                 compiled_image: bytes | None = None):
        """Wrapper binding; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:17-653.

        ``batch_columns`` None sizes each launch from the device memory the per-column automatic-storage arena
        may take (:func:`automatic_batch_columns`).  Columns are independent (each thread owns column
        ``batch_first + thread`` and its own arena), so the batch size never changes a word.
        """
        if shared_math_source is None:
            from woof.hex.mynn_glibc_math import mynn_glibc_source
            shared_math_source = mynn_glibc_source()
        if not shared_math_source:
            raise ValueError('Shared glibc device maths is required for the RRFS MYNN module')
        if batch_columns is not None and batch_columns < 1:
            raise ValueError('batch_columns must be positive')
        self.shared_math_source = shared_math_source
        self.batch_columns = None if batch_columns is None else int(batch_columns)
        self._module = self._kernel = None
        self._kernels = {}
        self._lane_widths = {}
        self.compiled_image = compiled_image

    def _compile(self):
        """Compile wrapper; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:17-653."""
        import cupy as cp
        from cupy_backends.cuda.libs import nvrtc
        from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
        source = CUDA_FTZ_HELPERS + self.shared_math_source + ARRAY_SUPPORT + MODULE_CUDA
        if self.compiled_image is not None:
            self._module = cp.cuda.function.Module()
            self._module.load(self.compiled_image)
            self._load_kernels()
            return
        options = _hex_numerics.options(CUDA_OPTIONS) + ('--gpu-architecture=sm_' + cp.cuda.Device().compute_capability,)
        program = nvrtc.createProgram(source, 'rrfs_mynnedmf.cu', (), ())
        try:
            try:
                nvrtc.compileProgram(program, options)
            except Exception as exc:
                raise RuntimeError(nvrtc.getProgramLog(program)) from exc
            ptx = nvrtc.getCUBIN(program)
        finally:
            nvrtc.destroyProgram(program)
        self._module = cp.cuda.function.Module()
        self._module.load(ptx)
        self._load_kernels()

    def _load_kernels(self):
        """One kernel per compiled lane width (LANE_WIDTHS); _kernel is the one-thread-per-column kernel."""
        self._kernels = {width: self._module.get_function(f'rrfs_mynnedmf_run_l{width}') for width in LANE_WIDTHS}
        self._kernel = self._kernels[1]

    def run(self, arguments: Mapping, *, allocation_entries: Mapping | None = None) -> dict:
        """Run wrapper; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:17-653."""
        import cupy as cp
        import numpy as np
        values = dict(arguments)
        actuals = []
        copies = []
        for name, meta in ARGUMENTS.items():
            if name in ('errflg', 'errmsg'):
                continue
            value = values.get(name)
            if meta['dimensions']:
                shape = argument_shape(meta, values)
                dtype = np.float32 if meta['kind'] == 'float' else np.int32
                if value is None:
                    if meta['intent'] == 'out' and not meta['optional']:
                        value = cp.empty(shape, dtype=dtype, order='F')
                        values[name] = value
                    elif meta['optional']:
                        actuals.append(np.uint64(0))
                        continue
                    else:
                        raise ValueError(f'Missing MYNN driver argument {name}')
                if not isinstance(value, cp.ndarray) or value.dtype != dtype or value.shape != shape:
                    raise ValueError(f'{name} requires a resident {np.dtype(dtype)} array with shape {shape}')
                packed = cp.asfortranarray(value)
                actuals.append(packed)
                if packed is not value and meta['intent'] in ('out', 'inout'):
                    copies.append((value, packed))
            else:
                if value is None and not meta['optional']:
                    raise ValueError(f'Missing MYNN driver argument {name}')
                actuals.append(np.float32(0 if value is None else value) if meta['kind'] == 'float' else np.int32(0 if value is None else value))
                if meta['optional']:
                    actuals.append(np.int32(value is not None))
        if self._kernel is None:
            self._compile()
        ncols = (int(values['ite'])-int(values['its'])+1)*(int(values['jte'])-int(values['jts'])+1)
        stride = workspace_bytes(values)
        batch_columns = self.batch_columns or automatic_batch_columns(ncols, stride)
        self.last_batch_columns = batch_columns
        launch_columns = min(ncols, batch_columns)
        # chosen once per launch size (device attribute and occupancy queries stay off the step path)
        lane_width = self._lane_widths.get(launch_columns)
        if lane_width is None:
            lane_width = self._lane_widths[launch_columns] = lanes_per_column(self._kernels, launch_columns)
        self.last_lane_width = lane_width
        tile_columns = 32 // lane_width
        # Zeroed, not empty: with config_mynn_tkebudget=0 the wrapper still copies its never-written automatic
        # TKE-budget arrays (dqke1, qwt1, qshear1, qbuoy1, qdiss1; module_bl_mynnedmf_driver.F90) into dqke, qwt,
        # qshear, qbuoy and qdiss, as native does (mpas_atmphys_driver_pbl.F:1203, undefined words there too).
        # From an uninitialized arena those words changed with the batch size and from run to run; from a
        # zeroed one they are reproducible.  No defined output reads the arena before writing it: 42 of the
        # 47 returned arrays were word-identical at batch 32 and 2,562 on x1.2562 hrrrv5, 2026-10-28.
        # Warp tiles: a launch's columns are padded to whole warps (32/lane_width columns per tile).
        tiles = (min(ncols, batch_columns) + tile_columns - 1)//tile_columns
        workspace = cp.zeros(tiles*tile_columns*stride, dtype=cp.uint8)
        entries = []
        if allocation_entries:
            layout = wrapper_allocation_layout(values)
            for name, value in allocation_entries.items():
                if name not in layout:
                    raise ValueError(f'Unknown wrapper automatic array {name}')
                offset, size, shape = layout[name]
                if not isinstance(value, cp.ndarray) or value.dtype != cp.uint8 or value.size != size:
                    raise ValueError(f'{name} requires {size} captured entry bytes')
                if offset+size > stride:
                    raise RuntimeError('Captured entry exceeds the automatic-storage arena')
                if offset % 4 or size % 4:
                    raise RuntimeError('Captured entry is not whole four-byte words')
                entries.append((offset, size, value.reshape(-1).view(cp.int32)))
        errors = cp.empty(ncols, dtype=cp.int32)
        for first in range(0, ncols, batch_columns):
            count = min(batch_columns, ncols-first)
            words = workspace.view(cp.int32).reshape(-1, stride//4, tile_columns)
            for offset, size, entry in entries:
                # column t's logical word w lives at tile t//tile_columns, row w, slot t%tile_columns
                column = cp.arange(count)
                words[column//tile_columns, offset//4:(offset+size)//4, column % tile_columns] = entry[None, :]
            self._kernels[lane_width](((count*lane_width+31)//32,), (32,),
                                      (np.int32(first), np.int32(count), workspace,
                                       np.uint64(stride), errors, *actuals))
        for original, packed in copies:
            original[...] = packed
        return {**{n: values[n] for n, m in ARGUMENTS.items()
                   if m['intent'] in ('out', 'inout') and m['dimensions'] and values.get(n) is not None},
                'errflg': errors}


# RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:758-849,1134-1247
PERSISTENT_FIELDS = {
    'el_pbl': 'el_pbl', 'cov': 'cov', 'qke': 'qke', 'qke_adv': 'qke_adv',
    'qsq': 'qsq', 'tsq': 'tsq', 'sh3d': 'sh3d', 'sm3d': 'sm3d',
    'cldfra_bl': 'cldfrac_bl', 'qc_bl': 'qc_bl', 'qi_bl': 'qi_bl',
    'edmf_a': 'edmf_a', 'edmf_ent': 'edmf_ent', 'edmf_qc': 'edmf_qc',
    'edmf_qt': 'edmf_qt', 'edmf_thl': 'edmf_thl', 'edmf_w': 'edmf_w',
    'sub_thl': 'sub_thl', 'sub_sqv': 'sub_qv', 'det_thl': 'det_thl',
    'det_sqv': 'det_qv',
}
ZERO_FIELDS = ('dqke', 'qbuoy', 'qdiss', 'qshear', 'qwt', 'rqsblten',
               'rncblten', 'rniblten', 'rnifablten', 'rnwfablten', 'pattern_spp')
SURFACE_OUTPUTS = ('maxwidth', 'maxmf', 'ztop_plume', 'excess_h', 'excess_q')

# RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:787-794
PBL_GEOMETRY_CUDA = r'''
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:787-794
extern "C" __global__ void rrfs_mynnedmf_dx(int n,int its,int ims,int ni,
                                            int jts,int jms,int active_i,float len_disp,
                                            const float* density,float* dx) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<n) {
        int i=its+index%active_i, j=jts+index/active_i;
        dx[(i-ims)+ni*(j-jms)]=mpas_div(len_disp,mynn_pow(density[i-1],0.25f));
    }
}
'''


def prepare_pbl(backend: MynnEdmfCuda, common_arguments: Mapping, fields: Mapping,
                mesh_density, config: Mapping, len_disp: float) -> dict:
    """Pack the active tile; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:758-849.

    Pool indices start at one. Local arrays retain their declared memory bounds.
    Supplied local allocations preserve untouched entry bytes, including halos.
    r4 supplies the common conversions and owns the native allocation lifecycle.
    """
    import cupy as cp
    import numpy as np
    if not isinstance(mesh_density, cp.ndarray) or mesh_density.dtype != np.float32 or mesh_density.ndim != 1:
        raise ValueError('meshDensity requires a resident one-dimensional float32 array')
    mesh_density = cp.ascontiguousarray(mesh_density)
    arguments = {**common_arguments, **option_arguments(config)}
    shape = argument_shape(ARGUMENTS['qke'], arguments)
    its, ite, kts, kte, jts, jte = (int(arguments[n]) for n in ('its','ite','kts','kte','jts','jte'))
    ims, kms, jms = (int(arguments[n]) for n in ('ims','kms','jms'))
    tile = (slice(its-ims, ite-ims+1), slice(kts-kms, kte-kms+1),
            slice(jts-jms, jte-jms+1))
    pool_tile = (slice(kts-1, kte), slice(its-1, ite))
    surface_tile = (tile[0], tile[2])
    def allocation(name, dimensions):
        """Native entry storage; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:758-849."""
        if arguments.get(name) is None:
            arguments[name] = cp.empty(dimensions, dtype=cp.float32, order='F')
        elif (not isinstance(arguments[name], cp.ndarray) or
              arguments[name].dtype != np.float32 or arguments[name].shape != dimensions):
            raise ValueError(f'{name} requires a resident float32 array with shape {dimensions}')
        else:
            arguments[name] = cp.asfortranarray(arguments[name])
        return arguments[name]
    for arg, pool in PERSISTENT_FIELDS.items():
        allocation(arg, shape)[tile] = fields[pool][pool_tile].T[:, :, None]
    surface = argument_shape(ARGUMENTS['dx'], arguments)
    for arg, pool in (('ts', 'skintemp'), ('ch', 'ch'), ('qsfc', 'qsfc')):
        allocation(arg, surface)[surface_tile] = fields[pool][its-1:ite, None]
    for name in ZERO_FIELDS:
        allocation(name, shape)[tile] = np.float32(0)
    for name in SURFACE_OUTPUTS:
        allocation(name, surface)[surface_tile] = np.float32(0)
    allocation('dx', surface)
    # Geometry is an additional device kernel with the same shared maths source.
    if not hasattr(backend, '_geometry'):
        from woof.hex.cuda_backend.aot import raw_image
        from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
        source = CUDA_FTZ_HELPERS + backend.shared_math_source + PBL_GEOMETRY_CUDA
        # The same raw NVRTC compile, through the box's warm store and the pack (woof.hex.cuda_backend.aot).
        module = cp.cuda.function.Module(); module.load(raw_image(source, _hex_numerics.options(CUDA_OPTIONS),
            cp.cuda.Device().compute_capability, filename='rrfs_mynn_geometry.cu'))
        backend._geometry_module = module
        backend._geometry = module.get_function('rrfs_mynnedmf_dx')
    n = (ite-its+1)*(jte-jts+1)
    backend._geometry(((n+127)//128,), (128,),
                      (np.int32(n), np.int32(its), np.int32(ims), np.int32(shape[0]),
                       np.int32(jts), np.int32(jms), np.int32(ite-its+1),
                       np.float32(len_disp), mesh_density, arguments['dx']))
    return arguments


def writeback_pbl(result: Mapping, arguments: Mapping, fields: Mapping) -> None:
    """Copy only the active tile; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1134-1247.

    The native j loop overwrites the same pool cells; its last active j wins.
    """
    its, ite, kts, kte = (int(arguments[n]) for n in ('its','ite','kts','kte'))
    ims, kms, jms = (int(arguments[n]) for n in ('ims','kms','jms'))
    tile = (slice(its-ims, ite-ims+1), slice(kts-kms, kte-kms+1),
            int(arguments['jte'])-jms)
    pool_tile = (slice(kts-1, kte), slice(its-1, ite))
    for arg, pool in PERSISTENT_FIELDS.items():
        fields[pool][pool_tile] = result[arg][tile].T
    for name in ('dqke', 'qbuoy', 'qdiss', 'qshear', 'qwt', 'rqsblten'):
        fields[name][pool_tile] = result[name][tile].T
    if arguments['f_ni']:
        fields['rniblten'][pool_tile] = result['rniblten'][tile].T
    if arguments['f_nc'] and arguments['f_nifa'] and arguments['f_nwfa']:
        for name in ('rncblten', 'rnifablten', 'rnwfablten'):
            fields[name][pool_tile] = result[name][tile].T
    for name in SURFACE_OUTPUTS:
        fields[name][its-1:ite] = result[name][tile[0], tile[2]]


def run_pbl(backend: MynnEdmfCuda, common_arguments: Mapping, fields: Mapping,
            mesh_density, config: Mapping, len_disp: float) -> dict:
    """Run the complete slice; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:758-849,1134-1247.

    r4 supplies the common surface/state conversions and their writeback.
    This slice owns the MYNN persistent state and extra scalar tendencies.
    """
    arguments = prepare_pbl(backend, common_arguments, fields, mesh_density, config, len_disp)
    result = backend.run(arguments)
    writeback_pbl(result, arguments, fields)
    return result


def register_backend(registry, callback) -> None:
    """Register fork branch; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1535-1663."""
    registry.register('pbl', 'bl_mynnedmf', 'hex-cuda', callback)


# RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1593-1663
NATIVE_ARRAY_BINDINGS = {'dx': 'dx_p', 'xland': 'xland_p', 'ps': 'psfc_p', 'ts': 'tsk_p', 'qsfc': 'qsfc_p', 'ust': 'ust_p', 'ch': 'ch_p', 'hfx': 'hfx_p', 'qfx': 'qfx_p', 'wspd': 'wspd_p', 'znt': 'znt_p', 'uoce': 'uoce_p', 'voce': 'voce_p', 'dz': 'dz_p', 'u': 'u_p', 'v': 'v_p', 'w': 'w_p', 'th': 'th_p', 'tt': 't_p', 'p': 'pres_p', 'exner': 'pi_p', 'rho': 'rho_p', 'qv': 'qv_p', 'qc': 'qc_p', 'qi': 'qi_p', 'qs': 'qs_p', 'ni': 'ni_p', 'nc': 'nc_p', 'nifa': 'nifa_p', 'nwfa': 'nwfa_p', 'rthraten': 'rthraten_p', 'pblh': 'hpbl_p', 'kpbl': 'kpbl_p', 'cldfra_bl': 'cldfrabl_p', 'qc_bl': 'qcbl_p', 'qi_bl': 'qibl_p', 'maxwidth': 'maxwidthbl_p', 'maxmf': 'maxmfbl_p', 'ztop_plume': 'zbl_plume_p', 'excess_h': 'excessh_p', 'excess_q': 'excessq_p', 'qke': 'qke_p', 'qke_adv': 'qkeadv_p', 'tsq': 'tsq_p', 'qsq': 'qsq_p', 'cov': 'cov_p', 'el_pbl': 'elpbl_p', 'rublten': 'rublten_p', 'rvblten': 'rvblten_p', 'rthblten': 'rthblten_p', 'rqvblten': 'rqvblten_p', 'rqcblten': 'rqcblten_p', 'rqiblten': 'rqiblten_p', 'rqsblten': 'rqsblten_p', 'rniblten': 'rniblten_p', 'rncblten': 'rncblten_p', 'rnwfablten': 'rnwfablten_p', 'rnifablten': 'rnifablten_p', 'edmf_a': 'edmfa_p', 'edmf_w': 'edmfw_p', 'edmf_qt': 'edmfqt_p', 'edmf_thl': 'edmfthl_p', 'edmf_ent': 'edmfent_p', 'edmf_qc': 'edmfqc_p', 'sub_thl': 'subthl_p', 'sub_sqv': 'subqv_p', 'det_thl': 'detthl_p', 'det_sqv': 'detqv_p', 'exch_h': 'kzh_p', 'exch_m': 'kzm_p', 'dqke': 'dqke_p', 'qwt': 'qwt_p', 'qshear': 'qshear_p', 'qbuoy': 'qbuoy_p', 'qdiss': 'qdiss_p', 'sh3d': 'sh3d_p', 'sm3d': 'sm3d_p', 'pattern_spp': 'pattern_spp_pbl', 'chem3d': 'chem_p', 'vd3d': 'ddvel_p', 'settle3d': 'tend_chem_settle_p', 'frp_mean': 'frp_out_p', 'emis_ant_no': 'aero_emis_for_enhmix_p'}


def native_driver_arguments(prepared, metadata, config, *, itimestep):
    """Actual MPAS call; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1381-1391,1535-1663.

    Preserve full i,k,j allocations and optional absence. Read dt_pbl from the
    native manager rather than substituting the dynamics timestep.
    """
    values = {name: metadata[name] for name in ARGUMENTS
              if name.startswith('f_') or name in (
                  'ids','ide','jds','jde','kds','kde','ims','ime','jms','jme',
                  'kms','kme','its','ite','jts','jte','kts','kte')}
    values.update(option_arguments(config))
    restart = bool(config['config_do_restart'])
    values.update(delt=metadata['dt_pbl'], do_restart=restart,
                  do_dacycling=bool(config['config_do_DAcycling']),
                  initflag=0 if restart or itimestep > 1 else 1,
                  mix_chem=bool(metadata['do_chemistry']),
                  nchem=int(metadata['num_chem']), ndvel=int(metadata['num_chem']))
    for name, spec in ARGUMENTS.items():
        if not spec['dimensions']:
            continue
        native_name = NATIVE_ARRAY_BINDINGS.get(name)
        value = prepared.get(native_name) if native_name else None
        if value is None and not spec['optional']:
            raise RuntimeError('Native MYNN argument is unallocated: ' + name)
        values[name] = value
    return values


def load_cuda_packet(packet, device):
    """Strict precompiled wrapper; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:17-653."""
    import hashlib
    import json
    from pathlib import Path
    from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
    from woof.hex.mynn_glibc_math import mynn_glibc_source
    image = Path(packet['image']).read_bytes()
    metadata = json.loads(Path(packet['metadata']).read_text())
    source = CUDA_FTZ_HELPERS + mynn_glibc_source() + ARRAY_SUPPORT + MODULE_CUDA
    options = list(_hex_numerics.options(CUDA_OPTIONS)) + ['--gpu-architecture=sm_' + device.cuda.Device().compute_capability]
    if (metadata['source_sha256'] != hashlib.sha256(source.encode()).hexdigest()
            or metadata['image_sha256'] != hashlib.sha256(image).hexdigest()
            or metadata['options'] != options):
        raise RuntimeError('MYNN CUDA resource differs from the strict source, image or device architecture')
    return image


def bind_native_pools(provider, call):
    """Hydrate actual native ABI; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:89-271,461-915,1593-1663.

    The generic prepared views discard native tails and species dimensions.
    This scheme keeps complete original allocations and returns changed arrays
    through the same native ABI, preserving original bounds and time levels.
    """
    from . import driver_rrfs as driver
    import sys
    from .resident_pbl_rrfs import resident_slot
    slot = resident_slot(provider, call, module=sys.modules[__name__])
    if slot is not None:
        # Resident facade route: persistent Fortran-order allocations and fused pack/unpack (resident_pbl_rrfs).
        return {'mynnedmf_native': {'resident_slot': slot}}
    native, cp = provider.native, provider.cp
    metadata = {}
    for name in ARGUMENTS:
        if name.startswith('f_') or name in (
                'ids','ide','jds','jde','kds','kde','ims','ime','jms','jme',
                'kms','kme','jts','jte','kts','kte'):
            value = native.get_variable(name)
            if value is None:
                raise RuntimeError('Native MYNN scalar is absent: ' + name)
            metadata[name] = value[0].item()
    for name in ('dt_pbl','do_chemistry','num_chem'):
        value = native.get_variable(name)
        if value is None:
            raise RuntimeError('Native MYNN manager scalar is absent: ' + name)
        metadata[name] = value[0].item()
    metadata.update(its=provider.session.arguments.its, ite=provider.session.arguments.ite)
    # r4's full common boundary addresses MPAS allocations from index one.
    # Reject unsupported origins instead of writing incorrect native addresses.
    if any(metadata[n] != 1 for n in ('ims','kms','jms')):
        raise RuntimeError('r4 common PBL conversion requires native one-based memory origins')
    prepared, lower = {}, {}
    for name in dict.fromkeys((*NATIVE_ARRAY_BINDINGS.values(), 'kzq_p')):
        value = native.get_variable(name)
        if value is not None:
            host, bounds = value
            prepared[name] = cp.asarray(host, order='C')
            lower[name] = bounds
    options = dict(provider.session.layout.configs)
    options.update(call.options)
    if options['config_pbl_scheme'].strip() != 'bl_mynnedmf':
        raise ValueError('MYNN-EDMF callback received a different native selector')
    diag = set(driver._PBL_VOLUME.values()) | set(driver._PBL_DIAG_TEND) | {
        'hfx','hpbl','kpbl','qfx','ust','wspd','znt','ch','qsfc',
        'kzh','kzm','kzq','maxmf','maxwidth','ztop_plume','excess_h','excess_q'}
    tend = set(driver._PBL_COMMON_TEND) | {'rthratenlw','rthratensw','rqsblten'}
    if metadata['f_ni']:
        tend.add('rniblten')
    if metadata['f_nc'] and metadata['f_nifa'] and metadata['f_nwfa']:
        tend.update(('rncblten','rnifablten','rnwfablten'))
    names = {'diag_physics': diag, 'tend_physics': tend,
             'sfc_input': {'xland','skintemp'}, 'mesh': {'meshDensity'}}
    dimensions = {}
    if metadata['do_chemistry']:
        names['state'] = {'scalars'}
        dimensions['chemistry_start'] = provider.session.layout.indices['state', 'chemistry_start']
        if int(options['drydep_opt']) != 0:
            diag.add('ddvel')
            tend.add('tend_chem_settle')
        if options['config_mynn_enh_mix']:
            if options['config_smoke_scheme'].strip() == 'on':
                diag.add('frp_out')
            if (options['config_anthro_scheme'].strip() != 'off'
                    or options['config_rwc_scheme'].strip() != 'off'):
                diag.add('aero_emis_for_enhmix')
    pools = {pool: {name: cp.asarray(native.get_field(pool, name, level=1), order='C')
                    for name in sorted(fields)} for pool, fields in names.items()}
    if not hasattr(provider, '_rrfs_mynnedmf_backend'):
        image = getattr(provider, '_rrfs_mynnedmf_image', None)
        if image is None:
            raise RuntimeError('MYNN native callback requires its strict precompiled CUDA resource')
        provider._rrfs_mynnedmf_backend = MynnEdmfCuda(compiled_image=image)
        provider._rrfs_mynnedmf_common_kernels = driver.StrictPhysicsKernelCache()
    return {'mynnedmf_native': {'native': native, 'prepared': prepared,
            'lower': lower, 'pools': pools, 'metadata': metadata,
            'options': options, 'dimensions': dimensions,
            'backend': provider._rrfs_mynnedmf_backend,
            'kernels': provider._rrfs_mynnedmf_common_kernels}}


def run_slot(call):
    """Full PBL transaction; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:461-1283,1535-1663."""
    import cupy as cp
    from . import driver_rrfs as driver
    context = call.pools['mynnedmf_native']
    if 'resident_slot' in context:
        return context['resident_slot'].run(call)
    metadata, options = context['metadata'], context['options']
    tile = {name: int(metadata[name]) for name in ('its','ite','jts','jte','kts','kte')}
    common = dict(pools=context['pools'], options=options, flags=metadata,
                  dimensions=context['dimensions'], **tile)
    prepared, _ = driver.pbl_from_mpas(
        **common, prepared=context['prepared'], kernel_cache=context['kernels'])
    arguments = native_driver_arguments(prepared, metadata, options, itimestep=call.itimestep)
    result = context['backend'].run(arguments)
    # Read off the step path (deferred_checks_v842): a forecast still stops on the flag, a replay at once.
    # THE COST THIS REMOVES: a host sync per PBL call; the writes below are stream-ordered device copies on the
    # resident route and host copies (which synchronize) on the native-library route.
    from ..deferred_checks_v842 import defer
    defer((result['errflg'] != 0).astype(cp.int32), 'MYNN-EDMF CUDA wrapper returned a native error flag')
    # run() changes resident actuals in place, including Fortran-order copies.
    pools = driver.pbl_to_mpas(**common, prepared=prepared)
    native = context['native']
    for name, value in prepared.items():
        native.put_variable(name, value, lower=context['lower'][name])
    for pool, fields in pools.items():
        for name, value in fields.items():
            native.put_field(pool, name, value, level=1,
                             nlevels=native._fields[pool, name][2])
    return result
