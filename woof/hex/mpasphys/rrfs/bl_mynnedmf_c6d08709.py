"""Pinned RRFS-MPAS MYNN-EDMF CUDA backend.

Numerical text is generated from c6d08709, including the complete MPAS wrapper.
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
// Generated from pinned RRFS MYNN-EDMF c6d08709; see translation receipt.
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
static constexpr float wfa_max = 800e10f;
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
static constexpr float p95 = 0.95f;
static constexpr float p99 = 0.99f;
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
static constexpr int dheat_opt = 1;
static constexpr bool env_subs = false;
static constexpr int bl_mynn_stfunc = 0;
static constexpr bool debug_code = false;
static constexpr int idbg = 452;
static constexpr int jdbg = 272;
int mynn_level;
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:420-1323
__device__ __forceinline__ void f_mynnedmf(int i, int j, int initflag, bool restart, bool cycling, float delt, MynnArray<float, 1> dz1, float dx, MynnArray<float, 1> u1, MynnArray<float, 1> v1, MynnArray<float, 1> w1, MynnArray<float, 1> th1, MynnArray<float, 1> sqv1, MynnArray<float, 1> sqc1, MynnArray<float, 1> sqi1, MynnArray<float, 1> sqs1, MynnArray<float, 1> qnc1, MynnArray<float, 1> qni1, MynnArray<float, 1> qnwfa1, MynnArray<float, 1> qnifa1, MynnArray<float, 1> qnbca1, MynnArray<float, 1> ozone1, MynnArray<float, 1> pres1, MynnArray<float, 1> ex1, MynnArray<float, 1> rho1, MynnArray<float, 1> tk1, float xland, float ts, float qsfc, float ps, float ust, float ch, float hfx, float qfx, float znt, float wspd, float uoce, float voce, MynnArray<float, 1> qke1, MynnArray<float, 1> qke_adv1, MynnArray<float, 1> el1, MynnArray<float, 1> sh1, MynnArray<float, 1> sm1, MynnArray<float, 1> kh1, MynnArray<float, 1> km1, int nchem, int ndvel, MynnArray<float, 2> chem1, MynnArray<float, 1> vdep, float frp, float emis_ant_no, bool mix_chem, bool enh_mix, MynnArray<float, 2> settle1, MynnArray<float, 2> scalars, int nscalars, MynnArray<float, 1> tsq1, MynnArray<float, 1> qsq1, MynnArray<float, 1> cov1, MynnArray<float, 1> du1, MynnArray<float, 1> dv1, MynnArray<float, 1> dth1, MynnArray<float, 1> dqv1, MynnArray<float, 1> dqc1, MynnArray<float, 1> dqi1, MynnArray<float, 1> dqnc1, MynnArray<float, 1> dqni1, MynnArray<float, 1> dqs1, MynnArray<float, 1> dqnwfa1, MynnArray<float, 1> dqnifa1, MynnArray<float, 1> dqnbca1, MynnArray<float, 1> dozone1, MynnArray<float, 1> rthraten1, float& pblh, int& kpbl, float& maxwidth_dd, float& maxwidth, float& maxmf, float& ztop_plume, float& excess_h, float& excess_q, float& maxmf_dd, float& maxtkeprod, float& cldtop_cooling, float& ent_eff, MynnArray<float, 1> dqke1, MynnArray<float, 1> qwt1, MynnArray<float, 1> qshear1, MynnArray<float, 1> qbuoy1, MynnArray<float, 1> qdiss1, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> qi_bl1, MynnArray<float, 1> cldfra_bl1, bool bl_mynn_tkeadvect, int tke_budget, int bl_mynn_cloudpdf, int bl_mynn_mixlength, int icloud_bl, float bl_mynn_closure, int bl_mynn_edmf, int bl_mynn_edmf_mom, int bl_mynn_edmf_tke, int bl_mynn_mixscalars, int bl_mynn_mixaerosols, int bl_mynn_mixnumcon, int bl_mynn_output, int bl_mynn_cloudmix, int bl_mynn_mixqt, int bl_mynn_edmf_dd, int bl_mynn_ess, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_qt1, MynnArray<float, 1> edmf_thl1, MynnArray<float, 1> edmf_ent1, MynnArray<float, 1> edmf_qc1, MynnArray<float, 1> sub_thl1, MynnArray<float, 1> sub_sqv1, MynnArray<float, 1> det_thl1, MynnArray<float, 1> det_sqv1, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1, bool flag_qc, bool flag_qi, bool flag_qnc, bool flag_qni, bool flag_qs, bool flag_qnwfa, bool flag_qnifa, bool flag_qnbca, bool flag_ozone, int kts, int kte, MynnText& errmsg, int& errflg) {
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
MynnArray<float, 1> s_awqsq1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
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
float th_sfc;
float wsp;
int ktop_plume;
MynnArray<float, 1> tkeprod_dn = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> tkeprod_up = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
bool initialize_qke;
bool problem;
int nsub;
float delt2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:617
errmsg = " ";
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:618
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:620
if (debug_code) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:621
problem = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:622
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:623
wsp = mynn_sqrt(r6_add(mynn_ipow<2>(u1(k)), mynn_ipow<2>(v1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:624
if (((((((((((((((((mynn_abs(hfx)) > (1200.f))) || (((mynn_abs(qfx)) > (0.001f))))) || (((wsp) > (270.f))))) || (((tk1(k)) > (380.f))))) || (((tk1(k)) < (160.f))))) || (((sqv1(k)) < (zero))))) || (((sqc1(k)) < (zero))))) || (((r6_sub(ps, pres1(kts))) < (zero))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:628
kproblem = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:629
problem = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:630
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:631
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:632
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:633
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:634
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:635
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:636
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:637
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:638
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:639
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:640
if (problem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:641
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:642
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:643
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:644
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:645
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:646
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:647
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:648
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:649
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:652
imd = 10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:653
jmd = 10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:657
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1).size(); mynn_element += lane_width) { edmf_a1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:658
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1).size(); mynn_element += lane_width) { edmf_w1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:659
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt1).size(); mynn_element += lane_width) { edmf_qt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:660
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1).size(); mynn_element += lane_width) { edmf_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:661
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent1).size(); mynn_element += lane_width) { edmf_ent1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:662
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1).size(); mynn_element += lane_width) { edmf_qc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:663
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_u1).size(); mynn_element += lane_width) { edmf_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:664
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_v1).size(); mynn_element += lane_width) { edmf_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:665
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qv1).size(); mynn_element += lane_width) { edmf_qv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:666
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_thl1).size(); mynn_element += lane_width) { sub_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:667
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_sqv1).size(); mynn_element += lane_width) { sub_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:668
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_thl1).size(); mynn_element += lane_width) { det_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:669
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqv1).size(); mynn_element += lane_width) { det_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:670
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a_dd1).size(); mynn_element += lane_width) { edmf_a_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:671
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w_dd1).size(); mynn_element += lane_width) { edmf_w_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:672
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc_dd1).size(); mynn_element += lane_width) { edmf_qc_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:674
ktop_plume = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:675
ztop_plume = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:676
maxwidth = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:677
maxmf = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:678
excess_h = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:679
excess_q = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:680
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (kzero1).size(); mynn_element += lane_width) { kzero1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:687
if (((((initflag) > (0))) && ((!(restart))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:690
if (((restart) || (cycling))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:691
if (((mynn_maxval(qke1.slice(MynnRange{qke1.lo[0], qke1.hi[0]}))) < (0.0002f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:692
initialize_qke = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:694
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:695
initialize_qke = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:697
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:698
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:699
initialize_qke = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:701
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:703
if ((((!(restart))) || ((!(cycling))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:704
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sh1).size(); mynn_element += lane_width) { sh1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:705
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sm1).size(); mynn_element += lane_width) { sm1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:706
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (el1).size(); mynn_element += lane_width) { el1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:707
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tsq1).size(); mynn_element += lane_width) { tsq1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:708
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qsq1).size(); mynn_element += lane_width) { qsq1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:709
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cov1).size(); mynn_element += lane_width) { cov1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:710
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1).size(); mynn_element += lane_width) { cldfra_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:711
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1).size(); mynn_element += lane_width) { qc_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:712
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi_bl1).size(); mynn_element += lane_width) { qi_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:713
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke1).size(); mynn_element += lane_width) { qke1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:714
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke_adv1).size(); mynn_element += lane_width) { qke_adv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:715
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:716
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqc1).size(); mynn_element += lane_width) { dqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:717
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqi1).size(); mynn_element += lane_width) { dqi1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:718
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqni1).size(); mynn_element += lane_width) { dqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:719
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnc1).size(); mynn_element += lane_width) { dqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:720
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnwfa1).size(); mynn_element += lane_width) { dqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:721
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnifa1).size(); mynn_element += lane_width) { dqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:722
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnbca1).size(); mynn_element += lane_width) { dqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:723
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dozone1).size(); mynn_element += lane_width) { dozone1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:724
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1_old).size(); mynn_element += lane_width) { qc_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:725
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1_old).size(); mynn_element += lane_width) { cldfra_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:726
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sgm1).size(); mynn_element += lane_width) { sgm1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:727
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (vt1).size(); mynn_element += lane_width) { vt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:728
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (vq1).size(); mynn_element += lane_width) { vq1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:729
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (km1).size(); mynn_element += lane_width) { km1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:730
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (kh1).size(); mynn_element += lane_width) { kh1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:732
if (((tke_budget) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:733
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qwt1).size(); mynn_element += lane_width) { qwt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:734
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qshear1).size(); mynn_element += lane_width) { qshear1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:735
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qbuoy1).size(); mynn_element += lane_width) { qbuoy1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:736
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qdiss1).size(); mynn_element += lane_width) { qdiss1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:737
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqke1).size(); mynn_element += lane_width) { dqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:738
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:740
zw1(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:741
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:743
sqw1(k) = r6_add(r6_add(sqv1(k), sqc1(k)), sqi1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:744
thl1(k) = r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), sqc1(k))), r6_mul(r6_div(xlscp, ex1(k)), sqi1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:750
thlv1(k) = r6_mul(thl1(k), r6_add(one, r6_mul(p608, sqv1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:751
thv1(k) = r6_mul(th1(k), r6_sub(r6_add(one, r6_mul(p608, sqv1(k))), r6_add(sqc1(k), sqi1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:752
zw1(((k) + (1))) = r6_add(zw1(k), dz1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:753
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:755
if (initialize_qke) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:759
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:760
qke1(k) = r6_mul(r6_mul(five, ust), mynn_max((float)(r6_div(r6_sub(r6_mul(ust, 700.f), zw1(k)), r6_mul(mynn_max((float)(ust), (float)(p01)), 700.f))), (float)(p01)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:761
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:762
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:766
f_get_pblh(kts, kte, pblh, thv1, qke1, ust, zw1, dz1, xland, kpbl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:770
if (((scaleaware) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:771
f_scale_aware(dx, pblh, psig_bl, psig_shcu);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:772
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:773
psig_bl = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:774
psig_shcu = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:775
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:782
rmol = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:783
rmolh = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:784
f_mym_initialize(kts, kte, xland, pblh, dz1, dx, zw1, pres1, ex1, u1, v1, thl1, sqv1, pblh, th1, thv1, thlv1, sh1, sm1, ust, rmol, rmolh, el1, qke1, tsq1, qsq1, cov1, psig_bl, cldfra_bl1, bl_mynn_mixlength, bl_mynn_ess, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1, initialize_qke, spp_pbl, pattern_spp_pbl1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:801
if ((!(restart))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:803
if (bl_mynn_tkeadvect) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:804
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:805
qke_adv1(k) = qke1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:806
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:807
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:808
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:820
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:824
if (bl_mynn_tkeadvect) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:825
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qke1.slice(MynnRange{kts, kte}).linear(mynn_element) = qke_adv1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:826
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:829
if (((tke_budget) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:830
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqke1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { dqke1.slice(MynnRange{kts, kte}).linear(mynn_element) = qke1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:831
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:832
if (((icloud_bl) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:833
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1_old.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { cldfra_bl1_old.slice(MynnRange{kts, kte}).linear(mynn_element) = cldfra_bl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:834
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1_old.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qc_bl1_old.slice(MynnRange{kts, kte}).linear(mynn_element) = qc_bl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:835
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi_bl1_old.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qi_bl1_old.slice(MynnRange{kts, kte}).linear(mynn_element) = qi_bl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:836
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:837
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1).size(); mynn_element += lane_width) { cldfra_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:838
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1).size(); mynn_element += lane_width) { qc_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:839
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi_bl1).size(); mynn_element += lane_width) { qi_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:840
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1_old).size(); mynn_element += lane_width) { cldfra_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:841
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1_old).size(); mynn_element += lane_width) { qc_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:842
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi_bl1_old).size(); mynn_element += lane_width) { qi_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:843
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:844
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqc1).size(); mynn_element += lane_width) { dqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:845
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqi1).size(); mynn_element += lane_width) { dqi1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:846
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqs1).size(); mynn_element += lane_width) { dqs1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:847
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqni1).size(); mynn_element += lane_width) { dqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:848
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnc1).size(); mynn_element += lane_width) { dqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:849
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnwfa1).size(); mynn_element += lane_width) { dqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:850
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnifa1).size(); mynn_element += lane_width) { dqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:851
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnbca1).size(); mynn_element += lane_width) { dqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:852
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dozone1).size(); mynn_element += lane_width) { dozone1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:854
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1).size(); mynn_element += lane_width) { edmf_a1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:855
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1).size(); mynn_element += lane_width) { edmf_w1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:856
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1).size(); mynn_element += lane_width) { edmf_qc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:857
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qv1).size(); mynn_element += lane_width) { edmf_qv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:858
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1).size(); mynn_element += lane_width) { edmf_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:859
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_u1).size(); mynn_element += lane_width) { edmf_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:860
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_v1).size(); mynn_element += lane_width) { edmf_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:861
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_aw1).size(); mynn_element += lane_width) { s_aw1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:862
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awthl1).size(); mynn_element += lane_width) { s_awthl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:863
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqt1).size(); mynn_element += lane_width) { s_awqt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:864
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqv1).size(); mynn_element += lane_width) { s_awqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:865
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqc1).size(); mynn_element += lane_width) { s_awqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:866
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awu1).size(); mynn_element += lane_width) { s_awu1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:867
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awv1).size(); mynn_element += lane_width) { s_awv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:868
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqke1).size(); mynn_element += lane_width) { s_awqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:869
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqsq1).size(); mynn_element += lane_width) { s_awqsq1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:870
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnc1).size(); mynn_element += lane_width) { s_awqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:871
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqni1).size(); mynn_element += lane_width) { s_awqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:872
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnwfa1).size(); mynn_element += lane_width) { s_awqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:873
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnifa1).size(); mynn_element += lane_width) { s_awqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:874
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnbca1).size(); mynn_element += lane_width) { s_awqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:875
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:876
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars})).size(); mynn_element += lane_width) { s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:878
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a_dd1).size(); mynn_element += lane_width) { edmf_a_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:879
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w_dd1).size(); mynn_element += lane_width) { edmf_w_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:880
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc_dd1).size(); mynn_element += lane_width) { edmf_qc_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:881
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_aw1).size(); mynn_element += lane_width) { sd_aw1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:882
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awthl1).size(); mynn_element += lane_width) { sd_awthl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:883
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqt1).size(); mynn_element += lane_width) { sd_awqt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:884
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqv1).size(); mynn_element += lane_width) { sd_awqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:885
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqc1).size(); mynn_element += lane_width) { sd_awqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:886
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqi1).size(); mynn_element += lane_width) { sd_awqi1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:887
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnc1).size(); mynn_element += lane_width) { sd_awqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:888
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqni1).size(); mynn_element += lane_width) { sd_awqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:889
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnwfa1).size(); mynn_element += lane_width) { sd_awqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:890
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnifa1).size(); mynn_element += lane_width) { sd_awqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:891
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awu1).size(); mynn_element += lane_width) { sd_awu1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:892
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awv1).size(); mynn_element += lane_width) { sd_awv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:893
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqke1).size(); mynn_element += lane_width) { sd_awqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:894
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_thl1).size(); mynn_element += lane_width) { sub_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:895
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_sqv1).size(); mynn_element += lane_width) { sub_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:896
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_u1).size(); mynn_element += lane_width) { sub_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:897
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_v1).size(); mynn_element += lane_width) { sub_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:898
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_thl1).size(); mynn_element += lane_width) { det_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:899
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqv1).size(); mynn_element += lane_width) { det_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:900
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqc1).size(); mynn_element += lane_width) { det_sqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:901
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_u1).size(); mynn_element += lane_width) { det_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:902
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_v1).size(); mynn_element += lane_width) { det_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:904
zw1(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:905
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:906
zw1(((k) + (1))) = r6_add(zw1(k), dz1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:907
qv1(k) = r6_div(sqv1(k), r6_sub(one, sqv1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:908
qc1(k) = r6_div(sqc1(k), r6_sub(one, sqv1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:909
qi1(k) = r6_div(sqi1(k), r6_sub(one, sqv1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:910
qs1(k) = r6_div(sqs1(k), r6_sub(one, sqv1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:912
sqw1(k) = r6_add(r6_add(sqv1(k), sqc1(k)), sqi1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:913
thl1(k) = r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), sqc1(k))), r6_mul(r6_div(xlscp, ex1(k)), sqi1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:919
thv1(k) = r6_mul(th1(k), r6_sub(r6_add(one, r6_mul(p608, sqv1(k))), r6_add(sqc1(k), sqi1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:920
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:924
f_get_pblh(kts, kte, pblh, thv1, qke1, ust, zw1, dz1, xland, kpbl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:930
if (((scaleaware) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:931
f_scale_aware(dx, pblh, psig_bl, psig_shcu);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:932
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:933
psig_bl = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:934
psig_shcu = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:935
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:937
sqcg = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:938
cpm = r6_mul(cp, r6_add(one, r6_mul(0.84f, mynn_max((float)(sqv1(kts)), (float)(1e-8f)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:939
exnerg = mynn_pow(r6_div(ps, p1000mb), rcp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:948
flqv = r6_div(qfx, rho1(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:949
flqc = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:950
th_sfc = r6_div(ts, ex1(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:953
flq = r6_add(flqv, flqc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:954
flt = r6_sub(r6_div(hfx, r6_mul(rho1(kts), cpm)), r6_div(r6_mul(xlvcp, flqc), ex1(kts)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:955
fltv = r6_add(flt, r6_mul(r6_mul(flqv, p608), th_sfc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:958
rmol = r6_div(r6_mul(r6_neg(r6_mul(karman, gtr)), fltv), mynn_max((float)(mynn_ipow<3>(ust)), (float)(1.0e-6f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:959
rmolh = r6_div(r6_mul(r6_neg(r6_mul(karman, gtr)), flt), mynn_max((float)(mynn_ipow<3>(ust)), (float)(1.0e-6f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:960
zet = r6_mul(r6_mul(p5, dz1(kts)), rmol);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:961
zet = mynn_max((float)(zet), (float)(r6_neg(10.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:962
zet = mynn_min((float)(zet), (float)(10.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:964
if (((bl_mynn_stfunc) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:966
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:967
if (((zet) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:969
pmz = r6_add(one, r6_mul(3.0f, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:970
phh = r6_add(one, r6_mul(cphh_st, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:971
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:972
pmz = r6_sub(r6_div(one, mynn_pow(r6_sub(one, r6_mul(cphm_unst, zet)), 0.25f)), zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:973
phh = r6_div(one, mynn_sqrt(r6_sub(one, r6_mul(cphh_unst, zet))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:974
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:975
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:976
if (((zet) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:978
pmz = r6_add(0.95f, r6_mul(3.5f, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:979
phh = r6_add(one, r6_mul(cphh_st, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:980
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:982
pmz = r6_sub(r6_div(0.95f, mynn_pow(r6_sub(one, r6_mul(cphm_unst, zet)), 0.25f)), zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:983
phh = r6_div(one, mynn_sqrt(r6_sub(one, r6_mul(cphh_unst, zet))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:984
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:985
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:986
phi_m = f_phim(zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:987
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:989
phi_m = f_phim(zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:990
pmz = r6_sub(phi_m, zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:991
phh = f_phih(zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:992
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:999
f_mym_condensation(kts, kte, dx, dz1, zw1, xland, thl1, sqw1, sqv1, sqc1, sqi1, sqs1, pres1, ex1, tsq1, qsq1, cov1, sh1, el1, bl_mynn_cloudpdf, qc_bl1, qi_bl1, cldfra_bl1, pblh, hfx, vt1, vq1, th1, sgm1, bl_mynn_closure, spp_pbl, pattern_spp_pbl1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1011
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprod_up).size(); mynn_element += lane_width) { tkeprod_up.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1012
if (((bl_mynn_edmf) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1014
f_dmp_mf(i, j, kts, kte, delt, zw1, dz1, pres1, rho1, bl_mynn_edmf_mom, bl_mynn_edmf_tke, bl_mynn_mixscalars, bl_mynn_mixaerosols, bl_mynn_mixnumcon, bl_mynn_closure, u1, v1, w1, th1, thl1, thv1, tk1, sqw1, sqv1, sqc1, qke1, qsq1, qnc1, qni1, qnwfa1, qnifa1, qnbca1, ex1, vt1, vq1, sgm1, ust, flt, fltv, flq, flqv, pblh, kpbl, dx, xland, th_sfc, edmf_a1, edmf_w1, edmf_qt1, edmf_thl1, edmf_ent1, edmf_qc1, edmf_qv1, edmf_u1, edmf_v1, s_aw1, s_awthl1, s_awqt1, s_awqv1, s_awqc1, s_awu1, s_awv1, s_awqke1, s_awqsq1, s_awqnc1, s_awqni1, s_awqnwfa1, s_awqnifa1, s_awqnbca1, sub_thl1, sub_sqv1, sub_u1, sub_v1, det_thl1, det_sqv1, det_sqc1, det_u1, det_v1, nchem, chem1, s_awchem1, mix_chem, nscalars, scalars, s_awscalars1, qc_bl1, cldfra_bl1, qc_bl1_old, cldfra_bl1_old, flag_qc, flag_qi, flag_qnc, flag_qni, flag_qnwfa, flag_qnifa, flag_qnbca, psig_shcu, maxwidth, ktop_plume, maxmf, ztop_plume, excess_h, excess_q, spp_pbl, pattern_spp_pbl1, tkeprod_up, el1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1059
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1065
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprod_dn).size(); mynn_element += lane_width) { tkeprod_dn.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1066
maxtkeprod = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1067
cldtop_cooling = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1068
ent_eff = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1069
maxmf_dd = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1070
maxwidth_dd = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1071
if (((bl_mynn_edmf_dd) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1072
f_ddmp_mf(kts, kte, delt, dx, zw1, dz1, pres1, u1, v1, th1, thl1, thv1, tk1, sqw1, sqv1, sqc1, sqi1, qnc1, qni1, qnwfa1, qnifa1, qke1, rho1, ex1, qc_bl1, qi_bl1, cldfra_bl1, ust, flt, flq, fltv, pblh, kpbl, edmf_a_dd1, edmf_w_dd1, edmf_qt_dd1, edmf_thl_dd1, edmf_ent_dd1, edmf_qc_dd1, sd_aw1, sd_awthl1, sd_awqt1, sd_awqv1, sd_awqc1, sd_awqi1, sd_awqnc1, sd_awqni1, sd_awqnwfa1, sd_awqnifa1, sd_awu1, sd_awv1, sd_awqke1, tkeprod_dn, el1, rthraten1, psig_bl, maxmf_dd, maxwidth_dd);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1095
f_topdown_cloudrad(kts, kte, dz1, zw1, fltv, u1(kts), v1(kts), xland, kpbl, pblh, sqc1, sqi1, sqw1, thl1, th1, ex1, pres1, rho1, thv1, cldfra_bl1, qc_bl1, qi_bl1, rthraten1, tkeprod_dn, psig_bl, maxtkeprod, cldtop_cooling, ent_eff);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1103
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1107
delt2 = delt;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1110
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1111
qc_tot1(k) = mynn_max((float)(qc_bl1(k)), (float)(sqc1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1112
qi_tot1(k) = mynn_max((float)(qi_bl1(k)), (float)(r6_add(sqi1(k), sqs1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1113
thlv1(k) = r6_mul(r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), qc_tot1(k))), r6_mul(r6_div(xlscp, ex1(k)), qi_tot1(k))), r6_add(one, r6_mul(p608, sqv1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1116
thl_tot1(k) = r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), qc_tot1(k))), r6_mul(r6_div(xlscp, ex1(k)), qi_tot1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1118
thv1(k) = r6_mul(th1(k), r6_sub(r6_add(one, r6_mul(p608, sqv1(k))), r6_add(qc_tot1(k), qi_tot1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1119
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1121
f_mym_turbulence(kts, kte, xland, bl_mynn_closure, dz1, dx, zw1, pres1, ex1, u1, v1, thl1, thv1, thlv1, sqc1, sqw1, qke1, tsq1, qsq1, cov1, vt1, vq1, rmol, rmolh, flt, fltv, flq, pblh, th1, sh1, sm1, el1, dfm1, dfh1, dfq1, tcd1, qcd1, pdk1, pdt1, pdq1, pdc1, qwt1, qshear1, qbuoy1, qdiss1, tke_budget, psig_bl, psig_shcu, cldfra_bl1, bl_mynn_mixlength, bl_mynn_ess, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1, tkeprod_dn, tkeprod_up, spp_pbl, pattern_spp_pbl1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1147
f_mym_predict(kts, kte, bl_mynn_closure, delt2, dz1, ust, flt, flq, pmz, phh, el1, dfq1, rho1, pdk1, pdt1, pdq1, pdc1, qke1, tsq1, qsq1, cov1, s_aw1, s_awqke1, s_awqsq1, bl_mynn_edmf, bl_mynn_edmf_tke, qwt1, qdiss1, tke_budget);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1156
if (((dheat_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1157
{ const int mynn_lo3 = (kts), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1159
diss_heat1(k) = mynn_min((float)(mynn_max((float)(zero), (float)(r6_div(r6_div(mynn_pow(qke1(k), 1.5f), r6_mul(b1, mynn_max((float)(r6_mul(p5, r6_add(el1(k), el1(((k) + (1)))))), (float)(one)))), cp)))), (float)(0.002f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1161
diss_heat1(k) = r6_mul(diss_heat1(k), mynn_exp(r6_neg(r6_div(10000.f, mynn_max((float)(pres1(k)), (float)(one))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1162
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1163
diss_heat1(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1164
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1165
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (diss_heat1).size(); mynn_element += lane_width) { diss_heat1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1166
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1170
f_mynn_tendencies(kts, kte, i, delt, dz1, zw1, xland, pblh, rho1, u1, v1, th1, tk1, qv1, qc1, qi1, kzero1, qnc1, qni1, ps, pres1, ex1, thl1, sqv1, sqc1, sqi1, kzero1, sqw1, thl_tot1, qc_tot1, qi_tot1, qnwfa1, qnifa1, qnbca1, ozone1, ust, flt, flq, flqv, flqc, wspd, uoce, voce, tsq1, qsq1, cov1, tcd1, qcd1, dfm1, dfh1, dfq1, du1, dv1, dth1, dqv1, dqc1, dqi1, dqs1, dqnc1, dqni1, dqnwfa1, dqnifa1, dqnbca1, dozone1, diss_heat1, s_aw1, s_awthl1, s_awqt1, s_awqv1, s_awqc1, s_awu1, s_awv1, s_awqnc1, s_awqni1, s_awqnwfa1, s_awqnifa1, s_awqnbca1, sd_aw1, sd_awthl1, sd_awqt1, sd_awqv1, sd_awqc1, sd_awqi1, sd_awqnc1, sd_awqni1, sd_awqnwfa1, sd_awqnifa1, sd_awu1, sd_awv1, sub_thl1, sub_sqv1, sub_u1, sub_v1, det_thl1, det_sqv1, det_sqc1, det_u1, det_v1, flag_qc, flag_qi, flag_qnc, flag_qni, flag_qs, flag_qnwfa, flag_qnifa, flag_qnbca, flag_ozone, cldfra_bl1, bl_mynn_cloudmix, bl_mynn_mixqt, bl_mynn_edmf, bl_mynn_edmf_mom, bl_mynn_mixscalars, bl_mynn_mixaerosols, bl_mynn_mixnumcon);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1215
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1216
f_mynn_mix_chem(kts, kte, i, delt, dz1, pblh, nchem, ndvel, chem1, vdep, settle1, rho1, flt, tcd1, qcd1, dfh1, s_aw1, s_awchem1, emis_ant_no, frp, enh_mix, bl_mynn_edmf);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1228
{ const int mynn_lo4 = (1), mynn_hi4 = (nchem), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { ic = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1229
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1230
chem1(k, ic) = mynn_max((float)(chem1(k, ic)), (float)(1.e-12f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1231
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1232
} mynn_lane_sync(); ic = mynn_lo4 + mynn_n4*mynn_st4; if (mynn_n4 > 0) { const int mynn_src4 = (mynn_n4 - 1) % lane_width; k = mynn_lane_bcast(k, mynn_src4); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1233
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1235
if (((bl_mynn_mixscalars) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1236
f_mynn_mix_scalars(kts, kte, i, delt, dz1, nscalars, scalars, rho1, flt, tcd1, qcd1, dfh1, s_aw1, s_awscalars1, bl_mynn_edmf);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1244
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1247
f_retrieve_exchange_coeffs(kts, kte, dfm1, dfh1, dz1, km1, kh1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1251
if (((tke_budget) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1254
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1255
qshear1(k) = r6_sub(r6_mul(4.f, r6_div(r6_mul(mynn_ipow<3>(ust), phi_m), r6_mul(karman, dz1(k)))), qshear1(((k) + (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1256
qbuoy1(k) = r6_sub(r6_mul(4.f, r6_neg(r6_div(r6_mul(mynn_ipow<3>(ust), zet), r6_mul(karman, dz1(k))))), qbuoy1(((k) + (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1258
{ const int mynn_lo5 = (kts), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1259
dummy1(k) = r6_mul(p5, r6_add(qshear1(k), qshear1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1260
dummy2(k) = r6_mul(p5, r6_add(qbuoy1(k), qbuoy1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1261
dqke1(k) = r6_div(r6_mul(p5, r6_sub(qke1(k), dqke1(k))), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1262
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1263
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qshear1).size(); mynn_element += lane_width) { qshear1.linear(mynn_element) = dummy1.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1264
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qbuoy1).size(); mynn_element += lane_width) { qbuoy1.linear(mynn_element) = dummy2.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1266
k = kte;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1267
qshear1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1268
qbuoy1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1269
qwt1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1270
qdiss1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1271
dqke1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1272
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1275
if (((bl_mynn_output) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1277
if (((((bl_mynn_output) == (2))) && (((bl_mynn_edmf_dd) == (1))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1278
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_a1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_a_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1279
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_w1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_w_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1280
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_qt1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_qt_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1281
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_thl1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_thl_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1282
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_ent1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_ent_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1283
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_qc1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_qc_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1284
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1285
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1288
if (((debug_code) && (((i) == (idbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1289
if (((mynn_abs(qfx)) > (.001f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1291
if (((mynn_abs(hfx)) > (1100.f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1293
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1294
if (((((sh1(k)) < (0.f))) || (((sh1(k)) > (200.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1296
if (((mynn_abs(vt1(k))) > (2.0f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1298
if (((mynn_abs(vq1(k))) > (7000.f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1300
if (((((qke1(k)) < (r6_neg(1.f)))) || (((qke1(k)) > (200.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1302
if (((((el1(k)) < (0.f))) || (((el1(k)) > (1500.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1304
if (((((km1(k)) < (0.f))) || (((km1(k)) > (2000.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1306
if (((icloud_bl) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1307
if (((((cldfra_bl1(k)) < (zero))) || (((cldfra_bl1(k)) > (one))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1308
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1310
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1311
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1312
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1313
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1316
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke_adv1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qke_adv1.slice(MynnRange{kts, kte}).linear(mynn_element) = qke1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1384-1554
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1428
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1429
ql(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1430
vt(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1431
vq(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1432
qkw(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1433
qtw(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1434
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1435
ustm = mynn_max((float)(ust), (float)(p01));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1438
f_mym_level2(kts, kte, bl_mynn_ess, zw, dz, xland, pblh, u, v, thl, thv, thlv, theta, p, exner, qke, cldfra_bl1, edmf_a1, edmf_w1, qw, ql, vt, vq, dtl, dqw, dtv, gm, gh, sm, sh);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1450
el(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1451
if (initialize_qke) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1453
qke(kts) = r6_mul(r6_mul(1.5f, mynn_ipow<2>(ustm)), mynn_pow(r6_mul(b1, pmz), p666));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1454
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1457
qke(k) = r6_mul(qke(kts), mynn_max((float)(r6_div(r6_sub(r6_mul(ust, 700.f), zw(k)), r6_mul(ustm, 700.f))), (float)(p01)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1458
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1459
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1461
phm = r6_div(r6_mul(phh, b2), mynn_pow(r6_mul(b1, pmz), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1462
tsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flt)), ustm)))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1463
qsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flq)), ustm)))), (float)(5e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1464
cov(kts) = r6_mul(r6_mul(phm, r6_div(flt, ustm)), r6_div(flq, ustm));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1465
qpe(kts) = mynn_min((float)(mynn_max((float)(r6_mul(r6_mul(r6_mul(two, tsq(kts)), mynn_ipow<2>(gtr)), mynn_ipow<2>(taue))), (float)(zero))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1467
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1468
vkz = r6_mul(karman, zw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1469
el(k) = r6_div(vkz, r6_add(one, r6_div(vkz, hundred)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1472
tsq(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1473
qsq(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1474
cov(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1475
qpe(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1476
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; vkz = mynn_lane_bcast(vkz, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1480
lmax = 5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1482
for (l=1; (1) > 0 ? l <= (lmax) : l >= (lmax); l += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1485
f_mym_length(kts, kte, xland, dz, dx, zw, rmol, rmolh, flt, fltv, flq, vt, vq, u, v, qke, qpe, dtv, el, zi, theta, qkw, qtw, psig_bl, cldfra_bl1, bl_mynn_mixlength, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1501
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1502
elq = r6_mul(el(k), qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1503
pdk(k) = r6_mul(elq, r6_add(r6_mul(sm(k), gm(k)), r6_mul(sh(k), gh(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1505
pdt(k) = r6_mul(r6_mul(elq, sh(k)), mynn_ipow<2>(dtl(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1506
pdq(k) = r6_mul(r6_mul(elq, sh(k)), mynn_ipow<2>(dqw(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1507
pdc(k) = r6_mul(r6_mul(r6_mul(elq, sh(k)), dtl(k)), dqw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1508
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; elq = mynn_lane_bcast(elq, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1511
vkz = r6_mul(r6_mul(karman, p5), dz(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1512
elv = r6_div(r6_mul(p5, r6_add(el(((kts) + (1))), el(kts))), vkz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1513
if (initialize_qke) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1515
qke(kts) = r6_mul(r6_mul(one, mynn_ipow<2>(ustm)), mynn_pow(r6_mul(r6_mul(b1, pmz), elv), p666));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1516
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1518
phm = r6_div(r6_mul(phh, b2), mynn_pow(r6_div(r6_mul(b1, pmz), mynn_ipow<2>(elv)), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1519
tsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flt)), ustm)))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1520
qsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flq)), ustm)))), (float)(5e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1521
cov(kts) = r6_mul(r6_mul(phm, r6_div(flt, ustm)), r6_div(flq, ustm));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1523
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1524
b1l = r6_mul(r6_mul(b1, 0.25f), r6_add(el(((k) + (1))), el(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1527
tmpq = mynn_min((float)(mynn_max((float)(r6_mul(b1l, r6_add(pdk(((k) + (1))), pdk(k)))), (float)(qkemin))), (float)(125.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1529
if (initialize_qke) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1530
qke(k) = mynn_pow(tmpq, p666);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1531
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1533
if (((qke(k)) <= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1534
b2l = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1535
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1536
b2l = r6_div(r6_mul(b2, r6_div(b1l, b1)), mynn_sqrt(qke(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1537
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1539
tsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdt(((k) + (1))), pdt(k)))), (float)(zero))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1540
qsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdq(((k) + (1))), pdq(k)))), (float)(zero))), (float)(5e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1541
cov(k) = r6_mul(b2l, r6_add(pdc(((k) + (1))), pdc(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1542
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; b1l = mynn_lane_bcast(b1l, mynn_src3); b2l = mynn_lane_bcast(b2l, mynn_src3); tmpq = mynn_lane_bcast(tmpq, mynn_src3); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1544
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1546
if (initialize_qke) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1547
qke(kts) = r6_mul(p5, r6_add(qke(kts), qke(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1548
qke(kte) = qke(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1549
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1550
tsq(kte) = tsq(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1551
qsq(kte) = qsq(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1552
cov(kte) = cov(((kte) - (1)));
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1598-1800
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
float cfac;
float uonset;
float taper;
constexpr float thvp = 0.0f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1642
dtl(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1643
dqw(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1644
dtv(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1645
gm(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1646
gh(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1647
sm(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1648
sh(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1649
ri(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1651
switch (bl_mynn_ess) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1653
case 0: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1655
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1656
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1657
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1658
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1659
duz = r6_add(mynn_ipow<2>(r6_sub(u(k), u(((k) - (1))))), mynn_ipow<2>(r6_sub(v(k), v(((k) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1660
duz = r6_div(duz, mynn_ipow<2>(dzk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1661
dtz = r6_div(r6_sub(thl(k), thl(((k) - (1)))), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1662
dqz = r6_div(r6_sub(qw(k), qw(((k) - (1)))), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1664
vtt = r6_add(r6_add(one, r6_mul(vt(k), abk)), r6_mul(vt(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1665
vqq = r6_add(r6_add(tv0, r6_mul(vq(k), abk)), r6_mul(vq(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1666
dtq = r6_add(r6_mul(vtt, dtz), r6_mul(vqq, dqz));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1668
dtl(k) = dtz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1669
dqw(k) = dqz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1670
dtv(k) = dtq;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1672
gm(k) = duz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1673
gh(k) = r6_neg(r6_mul(dtq, gtr));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1676
ri(k) = r6_neg(r6_div(gh(k), mynn_max((float)(duz), (float)(1.0e-10f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1677
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src0); afk = mynn_lane_bcast(afk, mynn_src0); dqz = mynn_lane_bcast(dqz, mynn_src0); dtq = mynn_lane_bcast(dtq, mynn_src0); dtz = mynn_lane_bcast(dtz, mynn_src0); duz = mynn_lane_bcast(duz, mynn_src0); dzk = mynn_lane_bcast(dzk, mynn_src0); vqq = mynn_lane_bcast(vqq, mynn_src0); vtt = mynn_lane_bcast(vtt, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1679
break; } case 1: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1688
ugrid = mynn_sqrt(r6_add(mynn_ipow<2>(u(kts)), mynn_ipow<2>(v(kts))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1689
uonset = 15.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1690
taper = r6_sub(one, r6_mul(p95, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, uonset)), (float)(zero)), twenty)), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1691
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1692
clam0 = r6_mul(1.8f, taper);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1693
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1696
clam0 = 0.08f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1697
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1699
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1700
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1701
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1702
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1703
duz = r6_add(mynn_ipow<2>(r6_sub(u(k), u(((k) - (1))))), mynn_ipow<2>(r6_sub(v(k), v(((k) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1704
duz = r6_div(duz, mynn_ipow<2>(dzk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1705
dtz = r6_div(r6_sub(thl(k), thl(((k) - (1)))), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1706
dqz = r6_div(r6_sub(qw(k), qw(((k) - (1)))), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1709
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zw(k), r6_add(pblh, hundred))), (float)(zero)), mynn_max((float)(r6_mul(p3, pblh)), (float)(400.f)))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1710
clam = r6_add(r6_mul(clam0, r6_sub(one, wt)), r6_mul(p2, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1712
clam = r6_mul(clam, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zw(k), twenty)), (float)(zero)), 200.0f)), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1716
thv1 = r6_add(r6_mul(thvp, thv(k)), r6_mul(r6_sub(one, thvp), thlv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1717
thv0 = r6_add(r6_mul(thvp, thv(((k) - (1)))), r6_mul(r6_sub(one, thvp), thlv(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1721
tabs = r6_mul(th(k), exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1722
xl = f_xl_blend(tabs);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1723
qsat = f_qsat_blend(tabs, p(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1724
eth1 = r6_mul(th(k), mynn_exp(r6_div(r6_mul(xl, qsat), r6_mul(cp, tabs))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1726
tabs = r6_mul(th(((k) - (1))), exner(((k) - (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1727
xl = f_xl_blend(tabs);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1728
qsat = f_qsat_blend(tabs, p(((k) - (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1729
eth0 = r6_mul(th(((k) - (1))), mynn_exp(r6_div(r6_mul(xl, qsat), r6_mul(cp, tabs))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1732
mfi = mynn_max((float)(zero), (float)(mynn_min((float)(r6_mul(r6_mul(p5, r6_add(edmf_a(k), edmf_a(((k) - (1))))), r6_mul(p5, r6_add(edmf_w(k), edmf_w(((k) - (1))))))), (float)(p2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1736
qkei = mynn_max((float)(r6_mul(p5, r6_add(qke(k), qke(((k) - (1)))))), (float)(qkemin));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1737
qkei = r6_mul(mynn_min((float)(mynn_sqrt(r6_mul(p25, qkei))), (float)(p5)), p2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1740
cldfrai = r6_mul(p5, r6_add(cldfra(k), cldfra(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1742
cfac = mynn_max((float)(mynn_min((float)(r6_mul(2.7f, mynn_ipow<2>(r6_sub(cldfrai, one)))), (float)(one))), (float)(0.01f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1744
cldfrai = r6_mul(mynn_min((float)(r6_mul(cldfrai, cfac)), (float)(p1)), ten);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1748
lambda = r6_mul(r6_mul(clam, mynn_max((float)(mfi), (float)(qkei))), cldfrai);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1750
dtq = r6_add(r6_div(r6_sub(thv1, thv0), dzk), r6_mul(lambda, mynn_min((float)(r6_div(r6_sub(eth1, eth0), dzk)), (float)(zero))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1753
dtl(k) = dtz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1754
dqw(k) = dqz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1755
dtv(k) = dtq;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1757
gm(k) = duz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1758
gh(k) = r6_neg(r6_mul(dtq, gtr));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1761
ri(k) = r6_neg(r6_div(gh(k), mynn_max((float)(duz), (float)(1.0e-10f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1762
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src1); afk = mynn_lane_bcast(afk, mynn_src1); cfac = mynn_lane_bcast(cfac, mynn_src1); clam = mynn_lane_bcast(clam, mynn_src1); cldfrai = mynn_lane_bcast(cldfrai, mynn_src1); dqz = mynn_lane_bcast(dqz, mynn_src1); dtq = mynn_lane_bcast(dtq, mynn_src1); dtz = mynn_lane_bcast(dtz, mynn_src1); duz = mynn_lane_bcast(duz, mynn_src1); dzk = mynn_lane_bcast(dzk, mynn_src1); eth0 = mynn_lane_bcast(eth0, mynn_src1); eth1 = mynn_lane_bcast(eth1, mynn_src1); lambda = mynn_lane_bcast(lambda, mynn_src1); mfi = mynn_lane_bcast(mfi, mynn_src1); qkei = mynn_lane_bcast(qkei, mynn_src1); qsat = mynn_lane_bcast(qsat, mynn_src1); tabs = mynn_lane_bcast(tabs, mynn_src1); thv0 = mynn_lane_bcast(thv0, mynn_src1); thv1 = mynn_lane_bcast(thv1, mynn_src1); wt = mynn_lane_bcast(wt, mynn_src1); xl = mynn_lane_bcast(xl, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1764
break; } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1766
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1768
if (((ckmod) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1769
a2fac = r6_div(one, r6_add(one, mynn_max((float)(zero), (float)(ri(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1770
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1771
a2fac = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1772
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1774
rfc = r6_div(g1, r6_add(g1, g2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1775
f1 = r6_add(r6_add(r6_mul(b1, r6_sub(g1, c1)), r6_mul(r6_mul(r6_mul(r6_mul(three, a2), a2fac), r6_sub(one, c2)), r6_sub(one, c5))), r6_mul(r6_mul(two, a1), r6_sub(three, r6_mul(two, c2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1777
f2 = r6_sub(r6_mul(b1, r6_add(g1, g2)), r6_mul(r6_mul(three, a1), r6_sub(one, c2)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1778
rf1 = r6_div(r6_mul(b1, r6_sub(g1, c1)), f1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1779
rf2 = r6_div(r6_mul(b1, g1), f2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1780
smc = r6_div(r6_mul(r6_div(a1, r6_mul(a2, a2fac)), f1), f2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1781
shc = r6_mul(r6_mul(three, r6_mul(a2, a2fac)), r6_add(g1, g2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1783
ri1 = r6_div(p5, smc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1784
ri2 = r6_mul(rf1, smc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1785
ri3 = r6_sub(r6_mul(r6_mul(four, rf2), smc), r6_mul(two, ri2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1786
ri4 = mynn_ipow<2>(ri2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1789
rf = mynn_min((float)(r6_mul(ri1, r6_sub(r6_add(ri(k), ri2), mynn_sqrt(r6_add(r6_sub(mynn_ipow<2>(ri(k)), r6_mul(ri3, ri(k))), ri4))))), (float)(rfc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1791
sh(k) = r6_div(r6_mul(shc, r6_sub(rfc, rf)), r6_sub(one, rf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1792
sm(k) = r6_mul(r6_div(r6_mul(smc, r6_sub(rf1, rf)), r6_sub(rf2, rf)), sh(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1793
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; a2fac = mynn_lane_bcast(a2fac, mynn_src2); f1 = mynn_lane_bcast(f1, mynn_src2); f2 = mynn_lane_bcast(f2, mynn_src2); rf = mynn_lane_bcast(rf, mynn_src2); rf1 = mynn_lane_bcast(rf1, mynn_src2); rf2 = mynn_lane_bcast(rf2, mynn_src2); rfc = mynn_lane_bcast(rfc, mynn_src2); ri1 = mynn_lane_bcast(ri1, mynn_src2); ri2 = mynn_lane_bcast(ri2, mynn_src2); ri3 = mynn_lane_bcast(ri3, mynn_src2); ri4 = mynn_lane_bcast(ri4, mynn_src2); shc = mynn_lane_bcast(shc, mynn_src2); smc = mynn_lane_bcast(smc, mynn_src2); } }
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1820-2255
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1893
switch (bl_mynn_mixlength) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1895
case 0: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1897
cns = 2.7f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1898
alp1 = 0.23f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1899
alp2 = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1900
alp3 = five;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1901
alp4 = hundred;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1902
alp5 = p3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1905
pblh2 = r6_add(pblh, dz(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1906
h1 = mynn_max((float)(r6_mul(p3, pblh2)), (float)(mindz));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1907
h1 = mynn_min((float)(h1), (float)(maxdz));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1908
h2 = r6_mul(h1, p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1910
qkw(kts) = mynn_sqrt(mynn_max((float)(qke(kts)), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1911
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1912
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1913
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1914
qkw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(qke(k), abk), r6_mul(qke(((k) - (1))), afk))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1915
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src0); afk = mynn_lane_bcast(afk, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1916
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qtw).size(); mynn_element += lane_width) { qtw.linear(mynn_element) = qkw.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1918
elt = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1919
vsc = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1922
k = ((kts) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1923
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1924
while (((zwk) <= (r6_add(pblh2, h1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1925
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1926
qdz = r6_mul(mynn_max((float)(r6_sub(qkw(k), qmin)), (float)(0.03f)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1927
elt = r6_add(elt, r6_mul(qdz, zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1928
vsc = r6_add(vsc, qdz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1929
k = ((k) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1930
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1931
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1933
elt = r6_div(r6_mul(alp1, elt), vsc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1934
vflx = r6_add(r6_mul(r6_add(vt(kts), one), flt), r6_mul(r6_add(vq(kts), tv0), flq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1935
vsc = mynn_pow(r6_mul(r6_mul(gtr, elt), mynn_max((float)(zero), (float)(vflx))), p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1938
el(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1939
zwk1 = zw(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1941
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1942
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1945
if (((dtv(k)) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1946
bv = mynn_sqrt(r6_mul(gtr, dtv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1947
elb = r6_mul(r6_div(r6_mul(alp2, qkw(k)), bv), r6_add(one, r6_mul(r6_div(alp3, alp2), mynn_sqrt(r6_div(vsc, r6_mul(bv, elt))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1950
elf = r6_div(r6_mul(alp2, qkw(k)), bv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1952
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1953
elb = 1.0e10f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1954
elf = elb;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1955
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1958
if (((rmol) > (0.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1959
els = r6_div(r6_mul(karman, zwk), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1960
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1961
els = r6_mul(r6_mul(karman, zwk), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1962
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1968
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1970
el(k) = mynn_min((float)(r6_div(elb, r6_add(r6_add(r6_div(elb, elt), r6_div(elb, els)), one))), (float)(elf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1972
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1974
break; } case 1: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1977
ugrid = mynn_sqrt(r6_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1978
uonset = 15.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1979
wt_u1 = r6_sub(one, r6_mul(p2, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, uonset)), (float)(zero)), forty)), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1980
wt_u2 = r6_sub(one, r6_mul(p5, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, uonset)), (float)(zero)), forty)), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1982
wt_dx = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(dx, 4000.f)), (float)(zero)), 12000.f)), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1984
cns = 3.5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1985
alp1 = r6_add(r6_mul(0.23f, wt_dx), r6_mul(r6_sub(one, wt_dx), 0.24f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1986
alp2 = r6_add(r6_mul(p3, wt_dx), r6_mul(r6_sub(one, wt_dx), 0.40f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1987
alp3 = r6_mul(five, wt_u2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1988
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1989
alp4 = r6_mul(twenty, wt_u2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1990
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1991
alp4 = 15.0f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1992
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1993
alp5 = p3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1994
alp6 = fifty;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1997
pblh2 = mynn_max((float)(pblh), (float)(200.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1998
h1 = mynn_max((float)(r6_mul(p3, pblh2)), (float)(300.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:1999
h1 = mynn_min((float)(h1), (float)(600.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2000
h2 = r6_mul(h1, p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2002
qtke(kts) = mynn_max((float)(r6_mul(p5, qke(kts))), (float)(r6_mul(p5, qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2003
thetaw(kts) = theta(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2004
qkw(kts) = mynn_sqrt(mynn_max((float)(qke(kts)), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2005
qtw(kts) = mynn_sqrt(mynn_max((float)(r6_add(qke(kts), qpe(kts))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2006
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2007
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2008
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2009
qkw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(qke(k), abk), r6_mul(qke(((k) - (1))), afk))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2010
qtw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(r6_add(qke(k), qpe(k)), abk), r6_mul(r6_add(qke(((k) - (1))), qpe(((k) - (1)))), afk))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2011
qtke(k) = mynn_max((float)(r6_mul(p5, mynn_ipow<2>(qkw(k)))), (float)(0.005f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2012
thetaw(k) = r6_add(r6_mul(theta(k), abk), r6_mul(theta(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2013
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src1); afk = mynn_lane_bcast(afk, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2015
elt = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2016
vsc = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2019
k = ((kts) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2020
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2021
while (((zwk) <= (r6_add(pblh2, mynn_max((float)(mynn_min((float)(r6_mul(p3, pblh2)), (float)(600.f))), (float)(200.f)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2022
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2023
qdz = r6_mul(mynn_min((float)(mynn_max((float)(qkw(k)), (float)(0.01f))), (float)(30.0f)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2024
elt = r6_add(elt, r6_mul(qdz, zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2025
vsc = r6_add(vsc, qdz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2026
k = ((k) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2027
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2028
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2030
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2031
elt_max = r6_add(350.f, r6_mul(100.f, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, twenty)), (float)(zero)), thirty)), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2032
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2033
elt_max = 400.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2034
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2035
elt = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(alp1, elt), vsc)), (float)(8.f))), (float)(elt_max));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2038
vflx = r6_sub(fltv, 0.005f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2039
vsc = mynn_pow(r6_mul(r6_mul(gtr, elt), mynn_max((float)(zero), (float)(vflx))), p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2042
el(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2043
zwk1 = zw(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2046
f_boulac_length(kts, kte, zw, dz, qtke, thetaw, elblmin, elblavg);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2048
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2049
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2050
qkw_mf = mynn_max((float)(r6_mul(r6_mul(p5, r6_add(edmf_a1(k), edmf_a1(((k) - (1))))), r6_mul(p5, r6_add(edmf_w1(k), edmf_w1(((k) - (1))))))), (float)(mynn_abs(r6_mul(edmf_a_dd1(((k) - (1))), edmf_w_dd1(((k) - (1)))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2053
cpblh = mynn_min((float)(r6_div(r6_add(zwk, r6_mul(p25, pblh2)), pblh2)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2054
qkw_mf = r6_mul(cpblh, qkw_mf);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2055
alp3z = r6_mul(cpblh, alp3);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2057
if (((dtv(k)) > (0.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2058
bv = mynn_max((float)(mynn_sqrt(r6_mul(gtr, dtv(k)))), (float)(0.001f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2059
elb_mf = r6_div(r6_mul(alp6, qkw_mf), bv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2060
elb_mf = r6_div(elb_mf, r6_add(one, r6_div(elb_mf, 100.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2061
elb = r6_mul(r6_div(r6_mul(alp2, mynn_max((float)(qkw(k)), (float)(qkw_elb_min))), bv), r6_add(one, r6_mul(alp3z, mynn_sqrt(r6_div(vsc, r6_mul(bv, elt))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2063
elb = mynn_max((float)(elb), (float)(elb_mf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2064
elb = mynn_min((float)(elb), (float)(zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2065
elf = r6_div(r6_mul(one, mynn_max((float)(qkw(k)), (float)(qkw_elb_min))), bv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2066
elf = mynn_max((float)(elf), (float)(elb_mf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2067
elblavg(k) = mynn_max((float)(r6_mul(alp5, elblavg(k))), (float)(elb_mf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2068
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2069
elb = 1.0e10f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2070
elf = elb;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2071
elblavg(k) = r6_mul(p4, elblavg(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2072
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2074
z_m = mynn_max((float)(r6_sub(zwk, four)), (float)(karman));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2076
if (((rmol) > (0.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2077
els = r6_div(r6_mul(karman, zwk), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2078
els1 = r6_div(r6_mul(karman, z_m), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2079
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2080
els = r6_mul(r6_mul(karman, zwk), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2081
els1 = r6_mul(r6_mul(karman, z_m), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2082
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2085
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2093
el_unstab = r6_div(els, r6_add(one, r6_div(els1, elt)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2094
el(k) = mynn_min((float)(el_unstab), (float)(elb));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2095
el(k) = mynn_min((float)(el(k)), (float)(elf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2096
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2097
el(k) = r6_mul(el(k), wt_u1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2098
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2099
el(k) = r6_add(r6_mul(el(k), r6_sub(one, wt)), r6_mul(elblavg(k), wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2107
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2109
break; } case 2: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2111
uonset = r6_add(3.5f, r6_mul(dz(kts), p1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2112
ugrid = mynn_sqrt(r6_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2113
cns = 3.5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2114
alp1 = 0.22f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2115
alp2 = p3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2116
alp3 = 2.5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2117
alp4 = five;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2118
alp5 = alp2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2119
alp6 = fifty;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2123
pblh2 = mynn_max((float)(pblh), (float)(200.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2126
h1 = mynn_max((float)(r6_mul(p3, pblh2)), (float)(300.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2127
h1 = mynn_min((float)(h1), (float)(600.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2128
h2 = r6_mul(h1, p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2130
qtke(kts) = mynn_max((float)(r6_mul(p5, qke(kts))), (float)(r6_mul(p5, qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2131
qkw(kts) = mynn_sqrt(mynn_max((float)(qke(kts)), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2132
qtw(kts) = mynn_sqrt(mynn_max((float)(r6_add(qke(kts), qpe(kts))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2133
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2134
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2135
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2136
qkw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(qke(k), abk), r6_mul(qke(((k) - (1))), afk))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2137
qtw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(r6_add(qke(k), qpe(k)), abk), r6_mul(r6_add(qke(((k) - (1))), qpe(((k) - (1)))), afk))), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2138
qtke(k) = r6_mul(p5, mynn_ipow<2>(qkw(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2139
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src2); afk = mynn_lane_bcast(afk, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2141
elt = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2142
vsc = 1.0e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2145
pblh_plus_ent = mynn_max((float)(r6_add(pblh, h1)), (float)(100.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2146
k = ((kts) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2147
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2148
while (((zwk) <= (pblh_plus_ent))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2149
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2150
qdz = r6_mul(mynn_min((float)(mynn_max((float)(r6_sub(qkw(k), qmin)), (float)(0.03f))), (float)(30.0f)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2151
elt = r6_add(elt, r6_mul(qdz, zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2152
vsc = r6_add(vsc, qdz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2153
k = ((k) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2154
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2155
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2157
elt = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(alp1, elt), vsc)), (float)(ten))), (float)(400.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2160
vflx = fltv;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2161
vsc = mynn_pow(r6_mul(r6_mul(gtr, elt), mynn_max((float)(zero), (float)(vflx))), p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2164
el(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2165
zwk1 = zw(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2167
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2168
zwk = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2169
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2170
cldavg = r6_mul(p5, r6_add(cldfra_bl1(((k) - (1))), cldfra_bl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2171
qkw_mf = mynn_max((float)(r6_mul(r6_mul(p5, r6_add(edmf_a1(k), edmf_a1(((k) - (1))))), r6_mul(p5, r6_add(edmf_w1(k), edmf_w1(((k) - (1))))))), (float)(mynn_abs(r6_mul(edmf_a_dd1(((k) - (1))), edmf_w_dd1(((k) - (1)))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2175
if (((dtv(k)) > (0.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2177
bv = mynn_max((float)(mynn_sqrt(r6_mul(gtr, dtv(k)))), (float)(0.001f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2179
elb_mf = r6_mul(r6_div(mynn_max((float)(r6_mul(alp2, qkw(k))), (float)(r6_mul(alp6, qkw_mf))), bv), r6_add(one, r6_mul(alp3, mynn_sqrt(r6_div(vsc, r6_mul(bv, elt))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2182
elb = mynn_min((float)(r6_div(mynn_max((float)(r6_mul(alp5, qkw(k))), (float)(r6_mul(alp6, qkw_mf))), bv)), (float)(zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2185
wstar = r6_mul(1.25f, mynn_pow(r6_mul(r6_mul(gtr, pblh), mynn_max((float)(vflx), (float)(1.0e-4f))), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2186
tau_cloud = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(ctau, wstar), grav)), (float)(30.f))), (float)(150.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2188
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2189
tau_cloud = r6_add(r6_mul(tau_cloud, r6_sub(1.f, wt)), r6_mul(fifty, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2190
elf = mynn_min((float)(mynn_max((float)(r6_mul(tau_cloud, mynn_sqrt(mynn_min((float)(qtke(k)), (float)(forty))))), (float)(r6_div(r6_mul(alp6, qkw_mf), bv)))), (float)(zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2200
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2212
wstar = r6_mul(1.25f, mynn_pow(r6_mul(r6_mul(gtr, pblh), mynn_max((float)(vflx), (float)(1.0e-4f))), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2213
tau_cloud = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(ctau, wstar), grav)), (float)(fifty))), (float)(200.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2215
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2217
tau_cloud = r6_add(r6_mul(tau_cloud, r6_sub(one, wt)), r6_mul(mynn_max((float)(r6_mul(dzk, p25)), (float)(hundred)), wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2219
elb = mynn_min((float)(r6_mul(tau_cloud, mynn_sqrt(mynn_min((float)(qtke(k)), (float)(40.f))))), (float)(zwk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2221
elf = elb;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2222
elb_mf = elb;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2223
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2224
elf = r6_div(elf, r6_add(one, r6_div(elf, 800.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2225
elb_mf = mynn_max((float)(elb_mf), (float)(0.01f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2228
if (((rmol) > (0.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2229
els = r6_div(r6_mul(karman, zwk), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2230
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2231
els = r6_mul(r6_mul(karman, zwk), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2232
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2235
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2238
el(k) = mynn_sqrt(r6_div(mynn_ipow<2>(els), r6_add(r6_add(one, r6_div(mynn_ipow<2>(els), mynn_ipow<2>(elt))), r6_div(mynn_ipow<2>(els), mynn_ipow<2>(elb_mf)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2239
el(k) = r6_add(r6_mul(el(k), r6_sub(one, wt)), r6_mul(elf, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2240
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2242
break; } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2245
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2246
el_les = r6_mul(r6_mul(p25, p5), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2247
el(k) = r6_add(r6_mul(el(k), psig_bl), r6_mul(r6_sub(one, psig_bl), mynn_min((float)(el_les), (float)(el(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2248
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; el_les = mynn_lane_bcast(el_les, mynn_src3); } }
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2270-2418
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2302
zup = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2303
dlu = r6_sub(r6_sub(zw(((kte) + (1))), zw(k)), r6_mul(dz(k), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2304
zzz = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2305
zup_inf = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2306
beta = gtr;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2310
if (((k) < (kte))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2311
found = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2312
izz = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2313
while (((found) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2315
if (((izz) < (kte))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2316
dzt = dz(izz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2317
zup = r6_sub(zup, r6_mul(r6_mul(beta, theta(k)), dzt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2319
zup = r6_add(zup, r6_mul(r6_mul(r6_mul(beta, r6_add(theta(((izz) + (1))), theta(izz))), dzt), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2320
zzz = r6_add(zzz, dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2322
if (((((qtke(k)) < (zup))) && (((qtke(k)) >= (zup_inf))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2323
bbb = r6_div(r6_sub(theta(((izz) + (1))), theta(izz)), dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2324
if (((bbb) != (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2326
tl = r6_div(r6_div(r6_add(r6_neg(r6_mul(beta, r6_sub(theta(izz), theta(k)))), mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(r6_mul(beta, r6_sub(theta(izz), theta(k)))), r6_mul(r6_mul(r6_mul(two, bbb), beta), r6_sub(qtke(k), zup_inf)))), (float)(0.f)))), bbb), beta);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2329
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2330
if (((theta(izz)) != (theta(k)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2331
tl = r6_div(r6_sub(qtke(k), zup_inf), r6_mul(beta, r6_sub(theta(izz), theta(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2332
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2333
tl = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2334
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2335
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2336
dlu = r6_add(r6_sub(zzz, dzt), tl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2338
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2339
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2340
zup_inf = zup;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2341
izz = ((izz) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2342
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2343
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2344
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2346
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2348
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2353
zdo = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2354
zdo_sup = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2355
dld = zw(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2356
zzz = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2359
if (((k) > (kts))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2361
found = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2362
izz = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2363
while (((found) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2365
if (((izz) > (kts))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2366
dzt = dz(((izz) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2367
zdo = r6_add(zdo, r6_mul(r6_mul(beta, theta(k)), dzt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2369
zdo = r6_sub(zdo, r6_mul(r6_mul(r6_mul(beta, r6_add(theta(((izz) - (1))), theta(izz))), dzt), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2370
zzz = r6_add(zzz, dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2372
if (((((qtke(k)) < (zdo))) && (((qtke(k)) >= (zdo_sup))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2373
bbb = r6_div(r6_sub(theta(izz), theta(((izz) - (1)))), dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2374
if (((bbb) != (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2375
tl = r6_div(r6_div(r6_add(r6_mul(beta, r6_sub(theta(izz), theta(k))), mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(r6_mul(beta, r6_sub(theta(izz), theta(k)))), r6_mul(r6_mul(r6_mul(two, bbb), beta), r6_sub(qtke(k), zdo_sup)))), (float)(0.f)))), bbb), beta);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2378
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2379
if (((theta(izz)) != (theta(k)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2380
tl = r6_div(r6_sub(qtke(k), zdo_sup), r6_mul(beta, r6_sub(theta(izz), theta(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2381
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2382
tl = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2383
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2384
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2385
dld = r6_add(r6_sub(zzz, dzt), tl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2387
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2388
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2389
zdo_sup = zdo;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2390
izz = ((izz) - (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2391
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2392
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2393
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2394
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2396
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2403
dld = mynn_min((float)(dld), (float)(zw(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2404
lb1 = mynn_min((float)(dlu), (float)(dld));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2406
dlu = mynn_max((float)(mynn_min((float)(dlu), (float)(1000.f))), (float)(0.1f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2407
dld = mynn_max((float)(mynn_min((float)(dld), (float)(1000.f))), (float)(0.1f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2408
lb2 = mynn_sqrt(r6_mul(dlu, dld));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2411
if (((k) == (kte))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2412
lb1 = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2413
lb2 = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2414
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2429-2575
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2451
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { iz = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2456
zup = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2457
dlu(iz) = r6_sub(r6_sub(zw(((kte) + (1))), zw(iz)), r6_mul(dz(iz), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2458
zzz = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2459
zup_inf = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2460
beta = gtr;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2464
if (((iz) < (kte))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2466
found = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2467
izz = iz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2468
while (((found) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2470
if (((izz) < (kte))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2471
dzt = dz(izz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2472
zup = r6_sub(zup, r6_mul(r6_mul(beta, theta(iz)), dzt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2474
zup = r6_add(zup, r6_mul(r6_mul(r6_mul(beta, r6_add(theta(((izz) + (1))), theta(izz))), dzt), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2475
zzz = r6_add(zzz, dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2477
if (((((qtke(iz)) < (zup))) && (((qtke(iz)) >= (zup_inf))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2478
bbb = r6_div(r6_sub(theta(((izz) + (1))), theta(izz)), dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2479
if (((bbb) != (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2481
tl = r6_div(r6_div(r6_add(r6_neg(r6_mul(beta, r6_sub(theta(izz), theta(iz)))), mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(r6_mul(beta, r6_sub(theta(izz), theta(iz)))), r6_mul(r6_mul(r6_mul(two, bbb), beta), r6_sub(qtke(iz), zup_inf)))), (float)(zero)))), bbb), beta);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2484
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2485
if (((theta(izz)) != (theta(iz)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2486
tl = r6_div(r6_sub(qtke(iz), zup_inf), r6_mul(beta, r6_sub(theta(izz), theta(iz))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2487
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2488
tl = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2489
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2490
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2491
dlu(iz) = r6_add(r6_sub(zzz, dzt), tl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2493
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2494
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2495
zup_inf = zup;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2496
izz = ((izz) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2497
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2498
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2499
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2501
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2503
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2508
zdo = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2509
zdo_sup = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2510
dld(iz) = zw(iz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2511
zzz = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2514
if (((iz) > (kts))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2516
found = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2517
izz = iz;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2518
while (((found) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2520
if (((izz) > (kts))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2521
dzt = dz(((izz) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2522
zdo = r6_add(zdo, r6_mul(r6_mul(beta, theta(iz)), dzt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2524
zdo = r6_sub(zdo, r6_mul(r6_mul(r6_mul(beta, r6_add(theta(((izz) - (1))), theta(izz))), dzt), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2525
zzz = r6_add(zzz, dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2527
if (((((qtke(iz)) < (zdo))) && (((qtke(iz)) >= (zdo_sup))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2528
bbb = r6_div(r6_sub(theta(izz), theta(((izz) - (1)))), dzt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2529
if (((bbb) != (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2530
tl = r6_div(r6_div(r6_add(r6_mul(beta, r6_sub(theta(izz), theta(iz))), mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(r6_mul(beta, r6_sub(theta(izz), theta(iz)))), r6_mul(r6_mul(r6_mul(two, bbb), beta), r6_sub(qtke(iz), zdo_sup)))), (float)(zero)))), bbb), beta);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2533
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2534
if (((theta(izz)) != (theta(iz)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2535
tl = r6_div(r6_sub(qtke(iz), zdo_sup), r6_mul(beta, r6_sub(theta(izz), theta(iz))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2536
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2537
tl = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2538
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2539
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2540
dld(iz) = r6_add(r6_sub(zzz, dzt), tl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2542
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2543
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2544
zdo_sup = zdo;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2545
izz = ((izz) - (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2546
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2547
found = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2548
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2549
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2551
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2557
dld(iz) = mynn_min((float)(dld(iz)), (float)(zw(((iz) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2559
dlu(iz) = mynn_max((float)(r6_div(dlu(iz), r6_add(one, r6_div(dlu(iz), lmax)))), (float)(p1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2560
dld(iz) = mynn_max((float)(r6_div(dld(iz), r6_add(one, r6_div(dld(iz), lmax)))), (float)(p1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2563
lb1(iz) = mynn_min((float)(dlu(iz)), (float)(dld(iz)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2565
lb2(iz) = mynn_sqrt(r6_mul(dlu(iz), dld(iz)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2570
} mynn_lane_sync(); iz = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; beta = mynn_lane_bcast(beta, mynn_src0); zdo = mynn_lane_bcast(zdo, mynn_src0); zdo_sup = mynn_lane_bcast(zdo_sup, mynn_src0); zup = mynn_lane_bcast(zup, mynn_src0); zup_inf = mynn_lane_bcast(zup_inf, mynn_src0); zzz = mynn_lane_bcast(zzz, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2572
lb1(kte) = lb1(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2573
lb2(kte) = lb2(((kte) - (1)));
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2626-3180
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2716
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qkw).size(); mynn_element += lane_width) { qkw.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2717
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qtw).size(); mynn_element += lane_width) { qtw.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2718
pblh2 = mynn_max((float)(pblh), (float)(200.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2719
if (((closure) < (2.7f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2720
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qpe).size(); mynn_element += lane_width) { qpe.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2721
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2722
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2725
cpblh = mynn_min((float)(r6_div(r6_add(zw(k), r6_mul(p1, pblh2)), r6_mul(0.8f, pblh2))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2727
qpe(k) = mynn_min((float)(mynn_max((float)(r6_mul(r6_mul(r6_mul(r6_mul(cpblh, two), tsq(k)), mynn_ipow<2>(gtr)), mynn_ipow<2>(taue))), (float)(zero))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2728
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; cpblh = mynn_lane_bcast(cpblh, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2729
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2731
f_mym_level2(kts, kte, bl_mynn_ess, zw, dz, xland, pblh, u, v, thl, thv, thlv, theta, p, exner, qke, cldfra_bl1, edmf_a1, edmf_w1, qw, ql, vt, vq, dtl, dqw, dtv, gm, gh, sm, sh);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2741
f_mym_length(kts, kte, xland, dz, dx, zw, rmol, rmolh, flt, fltv, flq, vt, vq, u, v, qke, qpe, dtv, el, pblh, theta, qkw, qtw, psig_bl, cldfra_bl1, bl_mynn_mixlength, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2757
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2758
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2759
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2760
abk = r6_sub(one, afk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2761
elsq = mynn_ipow<2>(el(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2762
q3sq = mynn_ipow<2>(qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2763
q2sq = mynn_dmul(mynn_dmul(b1, elsq), r6_add(r6_mul(sm(k), gm(k)), r6_mul(sh(k), gh(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2765
sh20 = mynn_max((float)(sh(k)), (float)(1e-5f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2766
sm20 = mynn_max((float)(sm(k)), (float)(1e-5f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2767
sh(k) = mynn_max((float)(sh(k)), (float)(1e-5f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2770
duz = r6_add(mynn_ipow<2>(r6_sub(u(k), u(((k) - (1))))), mynn_ipow<2>(r6_sub(v(k), v(((k) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2771
duz = r6_div(duz, mynn_ipow<2>(dzk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2773
ri = r6_neg(r6_div(gh(k), mynn_max((float)(duz), (float)(1.0e-10f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2774
if (((ckmod) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2775
a2fac = r6_div(one, r6_add(one, mynn_max((float)(zero), (float)(ri))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2776
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2777
a2fac = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2778
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2787
if (((ri) >= (one))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2789
prlim = mynn_min((float)(r6_mul(seven, ri)), (float)(prlimit_fre));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2790
} else if (((((ri) >= (0.01f))) && (((ri) <= (one))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2792
prlim = mynn_min((float)(r6_add(r6_mul(6.873f, ri), r6_div(one, r6_mul(6.873f, ri)))), (float)(prlimit_fre));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2793
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2795
prlim = prlimit_fre;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2796
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2797
wt = mynn_min((float)(r6_div(zw(k), mynn_max((float)(pblh), (float)(200.f)))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2798
prlim = r6_add(r6_mul(wt, prlim), r6_mul(r6_sub(one, wt), mynn_min((float)(prlim), (float)(prlimit_sfc))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2801
gmel = mynn_dmul(gm(k), elsq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2802
ghel = mynn_dmul(gh(k), elsq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2806
if (debug_code) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2807
if (((((sh(k)) < (0.0f))) || (((sm(k)) < (0.0f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2808
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2809
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2810
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2811
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2812
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2813
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2814
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2820
dlsq = elsq;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2821
if (((mynn_ddiv(q3sq, dlsq)) < (r6_neg(gh(k))))) { q3sq = r6_neg(mynn_dmul(dlsq, gh(k))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2823
if (((q3sq) < (q2sq))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2825
qdiv = mynn_sqrt(mynn_ddiv(q3sq, q2sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2841
sh(k) = mynn_dmul(sh(k), qdiv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2842
sm(k) = mynn_dmul(sm(k), qdiv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2850
e1 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e1c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2851
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e2c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2852
e3 = mynn_dadd(e1, mynn_dmul(mynn_dmul(mynn_dmul(e3c, ghel), mynn_ipow<2>(a2fac)), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2853
e4 = mynn_dsub(e1, mynn_dmul(mynn_dmul(mynn_dmul(e4c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2854
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(mynn_dmul(e3, e5c), gmel), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2855
eden = mynn_max((double)(eden), (double)(1.0e-20));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2859
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2865
e1 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(e1c, ghel), a2fac));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2866
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(e2c, ghel), a2fac));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2867
e3 = mynn_dadd(e1, mynn_dmul(mynn_dmul(e3c, ghel), mynn_ipow<2>(a2fac)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2868
e4 = mynn_dsub(e1, mynn_dmul(mynn_dmul(e4c, ghel), a2fac));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2869
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(e3, e5c), gmel));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2870
eden = mynn_max((double)(eden), (double)(1.0e-20));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2872
qdiv = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2874
sm(k) = mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, a1), mynn_dsub(e3, mynn_dmul(r6_mul(three, c1), e4))), eden);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2878
sh(k) = mynn_ddiv(mynn_dmul((tke_budget == 1 ? mynn_dmul(q3sq, r6_mul(a2, a2fac)) : mynn_dadd(mynn_dmul(r6_mul(r6_mul(three, c1), e5c), gmel), e2)), (tke_budget == 1 ? mynn_dadd(mynn_dmul(r6_mul(r6_mul(three, c1), e5c), gmel), e2) : mynn_dmul(q3sq, r6_mul(a2, a2fac)))), eden);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2879
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2882
gmelq = mynn_max((float)(((float)(mynn_ddiv(gmel, q3sq)))), (float)(1e-8f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2883
sm25max = four;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2884
sh25max = four;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2885
sm25min = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2886
sh25min = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2890
if (debug_code) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2891
if (((((((((sh(k)) < (sh25min))) || (((sm(k)) < (sm25min))))) || (((sh(k)) > (sh25max))))) || (((sm(k)) > (sm25max))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2893
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2894
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2895
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2896
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2897
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2898
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2899
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2900
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2901
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2903
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2904
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2907
if (((sh(k)) > (sh25max))) { sh(k) = sh25max; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2908
if (((sh(k)) < (sh25min))) { sh(k) = sh25min; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2912
shb = mynn_max((float)(0.004f), (float)(sh(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2913
sm(k) = mynn_min((float)(sm(k)), (float)(r6_mul(prlim, shb)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2916
if (((closure) >= (3.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2917
t2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, b2), elsq), sh(k)), mynn_ipow<2>(dtl(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2918
r2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, b2), elsq), sh(k)), mynn_ipow<2>(dqw(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2919
c2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, b2), elsq), sh(k)), dtl(k)), dqw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2920
t3sq = mynn_max((float)(zero), (float)(r6_add(r6_mul(tsq(k), abk), r6_mul(tsq(((k) - (1))), afk))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2921
r3sq = mynn_max((float)(zero), (float)(r6_add(r6_mul(qsq(k), abk), r6_mul(qsq(((k) - (1))), afk))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2922
c3sq = r6_add(r6_mul(cov(k), abk), r6_mul(cov(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2925
c3sq = mynn_sign(mynn_min((double)(mynn_abs(c3sq)), (double)(mynn_sqrt(mynn_dmul(t3sq, r3sq)))), c3sq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2927
vtt = r6_add(r6_add(one, r6_mul(vt(k), abk)), r6_mul(vt(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2928
vqq = r6_add(r6_add(tv0, r6_mul(vq(k), abk)), r6_mul(vq(((k) - (1))), afk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2930
t2sq = mynn_dadd(mynn_dmul(vtt, t2sq), mynn_dmul(vqq, c2sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2931
r2sq = mynn_dadd(mynn_dmul(vtt, c2sq), mynn_dmul(vqq, r2sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2932
c2sq = mynn_max((double)(0.0e0), (double)(mynn_dadd(mynn_dmul(vtt, t2sq), mynn_dmul(vqq, r2sq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2933
t3sq = mynn_dadd(mynn_dmul(vtt, t3sq), mynn_dmul(vqq, c3sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2934
r3sq = mynn_dadd(mynn_dmul(vtt, c3sq), mynn_dmul(vqq, r3sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2935
c3sq = mynn_max((double)(0.0e0), (double)(mynn_dadd(mynn_dmul(vtt, t3sq), mynn_dmul(vqq, r3sq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2937
cw25 = mynn_ddiv(mynn_dmul(e1, mynn_dadd(e2, mynn_dmul(mynn_dmul(r6_mul(r6_mul(three, c1), e5c), gmel), mynn_ipow<2>(qdiv)))), mynn_dmul(three, eden));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2940
dlsq = elsq;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2941
if (((mynn_ddiv(q3sq, dlsq)) < (r6_neg(gh(k))))) { q3sq = r6_neg(mynn_dmul(dlsq, gh(k))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2946
auh = r6_mul(r6_mul(r6_mul(r6_mul(27.f, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), b2), mynn_ipow<2>(gtr));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2947
aum = r6_mul(r6_mul(r6_mul(r6_mul(r6_mul(54.f, mynn_ipow<2>(a1)), r6_mul(a2, a2fac)), b2), c1), gtr);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2948
adh = r6_mul(r6_mul(r6_mul(r6_mul(nine, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), r6_add(r6_mul(12.f, a1), r6_mul(three, b2))), mynn_ipow<2>(gtr));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2949
adm = r6_mul(r6_mul(r6_mul(r6_mul(18.f, mynn_ipow<2>(a1)), r6_mul(a2, a2fac)), r6_sub(b2, r6_mul(three, r6_mul(a2, a2fac)))), gtr);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2951
aeh = r6_mul(r6_add(r6_mul(r6_mul(r6_mul(nine, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), b1), r6_mul(r6_mul(r6_mul(nine, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), r6_add(r6_mul(12.f, a1), r6_mul(three, b2)))), gtr);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2953
aem = r6_add(r6_mul(r6_mul(r6_mul(r6_mul(three, a1), r6_mul(a2, a2fac)), b1), r6_add(r6_add(r6_mul(three, r6_mul(a2, a2fac)), r6_mul(r6_mul(three, b2), c1)), r6_sub(r6_mul(r6_mul(18.f, a1), c1), b2))), r6_mul(r6_mul(r6_mul(18.f, mynn_ipow<2>(a1)), r6_mul(a2, a2fac)), r6_sub(b2, r6_mul(three, r6_mul(a2, a2fac)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2957
req = r6_neg(r6_div(aeh, aem));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2958
rsl = r6_div(r6_add(auh, r6_mul(aum, req)), r6_add(r6_mul(three, adh), r6_mul(r6_mul(three, adm), req)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2960
rsl = 0.12f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2961
rsl2 = r6_sub(one, r6_mul(two, rsl));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2973
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e2c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2974
e3 = mynn_dadd(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e3c, ghel), mynn_ipow<2>(a2fac)), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2975
e4 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e4c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2976
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(mynn_dmul(e3, e5c), gmel), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2981
wden = mynn_dmul(mynn_dmul(mynn_ddiv(mynn_dmul(r6_mul(cc3, mynn_ipow<2>(gtr)), mynn_ipow<2>(dlsq)), elsq), mynn_ipow<2>(qdiv)), mynn_dsub(mynn_dmul(mynn_dmul(e2, e4c), a2fac), mynn_dmul(mynn_dmul(mynn_dmul(r6_mul(e3c, e5c), gmel), mynn_ipow<2>(a2fac)), mynn_ipow<2>(qdiv))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2984
if (((wden) != (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2986
clow = mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, r6_sub(0.12f, cw25)), eden), wden);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2987
cupp = mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, r6_sub(0.76f, cw25)), eden), wden);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2991
if (((wden) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2992
c3sq = mynn_min((double)(mynn_max((double)(c3sq), (double)(mynn_dadd(c2sq, clow)))), (double)(mynn_dadd(c2sq, cupp)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2993
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2994
c3sq = mynn_max((double)(mynn_min((double)(c3sq), (double)(mynn_dadd(c2sq, clow)))), (double)(mynn_dadd(c2sq, cupp)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2995
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2996
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2998
e1 = mynn_dadd(e2, mynn_dmul(mynn_dmul(e5c, gmel), mynn_ipow<2>(qdiv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:2999
eden = mynn_max((double)(eden), (double)(1.0e-20));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3004
e6c = mynn_ddiv(mynn_dmul(r6_mul(r6_mul(r6_mul(3.0f, r6_mul(a2, a2fac)), cc3), gtr), dlsq), elsq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3009
if (((t2sq) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3010
mynn_enum = mynn_max((double)(0.0e0), (double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(t3sq, t2sq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3011
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3012
mynn_enum = mynn_min((double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(t3sq, t2sq))), (double)(0.0e0));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3013
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3014
gamt = r6_neg((float)mynn_ddiv(mynn_dmul(mynn_enum, e1), eden));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3019
if (((r2sq) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3020
mynn_enum = mynn_max((double)(0.0e0), (double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(r3sq, r2sq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3021
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3022
mynn_enum = mynn_min((double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(r3sq, r2sq))), (double)(0.0e0));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3023
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3024
gamq = r6_neg((float)mynn_ddiv(mynn_dmul(mynn_enum, e1), eden));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3029
mynn_enum = mynn_max((double)(0.0e0), (double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(c3sq, c2sq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3033
smd = mynn_ddiv(mynn_dmul(mynn_dmul(mynn_dmul(mynn_ddiv(mynn_dmul(mynn_dmul(dlsq, mynn_enum), gtr), eden), mynn_ipow<2>(qdiv)), r6_add(r6_mul(e3c, mynn_ipow<2>(a2fac)), r6_mul(e4c, a2fac))), a1), r6_mul(a2, a2fac));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3036
gamv = mynn_ddiv(mynn_dmul(mynn_dmul(e1, mynn_enum), gtr), eden);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3037
sm(k) = r6_add(sm(k), smd);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3041
qdiv = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3044
if (debug_code) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3045
if (((((((((sh(k)) < (r6_neg(0.3f)))) || (((sm(k)) < (r6_neg(0.3f)))))) || (((qke(k)) < (r6_neg(0.1f)))))) || (((mynn_abs(smd)) > (2.0f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3047
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3048
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3049
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3050
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3051
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3052
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3053
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3057
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3059
gamt = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3060
gamq = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3061
gamv = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3062
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3066
cldavg = r6_mul(p5, r6_add(cldfra_bl1(((k) - (1))), cldfra_bl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3067
mfmax = mynn_max((float)(r6_mul(r6_mul(r6_mul(p5, r6_add(edmf_a1(((k) - (1))), edmf_a1(k))), p5), r6_add(edmf_w1(((k) - (1))), edmf_w1(k)))), (float)(mynn_abs(r6_mul(edmf_a_dd1(k), edmf_w_dd1(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3069
sm(k) = mynn_max((float)(sm(k)), (float)(r6_mul(0.04f, mynn_min((float)(r6_mul(10.f, mfmax)), (float)(one)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3070
sh(k) = mynn_max((float)(r6_mul(0.04f, mynn_min((float)(r6_mul(10.f, mfmax)), (float)(one)))), (float)(sh(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3072
sm(k) = mynn_max((float)(sm(k)), (float)(r6_mul(0.04f, mynn_min((float)(cldavg), (float)(p5)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3073
sh(k) = mynn_max((float)(r6_mul(0.04f, mynn_min((float)(cldavg), (float)(p5)))), (float)(sh(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3075
if (((tkeprod_dn(k)) > (1e-6f))) { sm(k) = mynn_max((float)(sm(k)), (float)(0.01f)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3076
if (((tkeprod_dn(k)) > (1e-6f))) { sh(k) = mynn_max((float)(sh(k)), (float)(0.01f)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3078
if (((((closure) >= (2.7f))) && (((closure) < (3.0f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3079
sm(k) = mynn_max((float)(sm(k)), (float)(mynn_min((float)(mynn_max((float)(r6_mul(three, qpe(k))), (float)(zero))), (float)(p2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3080
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3082
elq = r6_mul(el(k), qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3083
elqt = r6_mul(el(k), qtw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3084
elh = mynn_dmul(elq, qdiv);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3088
pdk(k) = r6_add(r6_add(r6_mul(elq, r6_add(r6_add(r6_mul(sm(k), gm(k)), r6_mul(sh(k), gh(k))), gamv)), r6_mul(p5, tkeprod_dn(k))), r6_mul(p5, tkeprod_up(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3092
pdt(k) = r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dtl(k)), gamt)), dtl(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3093
pdq(k) = r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dqw(k)), gamq)), dqw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3094
pdc(k) = r6_add(r6_mul(r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dtl(k)), gamt)), dqw(k)), p5), r6_mul(r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dqw(k)), gamq)), dtl(k)), p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3098
tcd(k) = r6_mul(elq, gamt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3099
qcd(k) = r6_mul(elq, gamq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3102
dfm(k) = r6_div(r6_mul(elqt, sm(k)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3103
dfh(k) = r6_div(r6_mul(elq, sh(k)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3107
dfq(k) = dfm(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3110
if (((tke_budget) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3126
qshear1(k) = r6_mul(r6_mul(elq, sm(k)), gm(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3134
qbuoy1(k) = r6_add(r6_add(r6_mul(elq, r6_add(r6_mul(gh(k), sh(k)), gamv)), r6_mul(p5, tkeprod_dn(k))), r6_mul(p5, tkeprod_up(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3142
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3144
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; a2fac = mynn_lane_bcast(a2fac, mynn_src1); abk = mynn_lane_bcast(abk, mynn_src1); afk = mynn_lane_bcast(afk, mynn_src1); cldavg = mynn_lane_bcast(cldavg, mynn_src1); dlsq = mynn_lane_bcast(dlsq, mynn_src1); duz = mynn_lane_bcast(duz, mynn_src1); dzk = mynn_lane_bcast(dzk, mynn_src1); e1 = mynn_lane_bcast(e1, mynn_src1); e2 = mynn_lane_bcast(e2, mynn_src1); e3 = mynn_lane_bcast(e3, mynn_src1); e4 = mynn_lane_bcast(e4, mynn_src1); eden = mynn_lane_bcast(eden, mynn_src1); elh = mynn_lane_bcast(elh, mynn_src1); elq = mynn_lane_bcast(elq, mynn_src1); elqt = mynn_lane_bcast(elqt, mynn_src1); elsq = mynn_lane_bcast(elsq, mynn_src1); gamq = mynn_lane_bcast(gamq, mynn_src1); gamt = mynn_lane_bcast(gamt, mynn_src1); gamv = mynn_lane_bcast(gamv, mynn_src1); ghel = mynn_lane_bcast(ghel, mynn_src1); gmel = mynn_lane_bcast(gmel, mynn_src1); gmelq = mynn_lane_bcast(gmelq, mynn_src1); mfmax = mynn_lane_bcast(mfmax, mynn_src1); prlim = mynn_lane_bcast(prlim, mynn_src1); q2sq = mynn_lane_bcast(q2sq, mynn_src1); q3sq = mynn_lane_bcast(q3sq, mynn_src1); qdiv = mynn_lane_bcast(qdiv, mynn_src1); ri = mynn_lane_bcast(ri, mynn_src1); sh20 = mynn_lane_bcast(sh20, mynn_src1); sh25max = mynn_lane_bcast(sh25max, mynn_src1); sh25min = mynn_lane_bcast(sh25min, mynn_src1); shb = mynn_lane_bcast(shb, mynn_src1); sm20 = mynn_lane_bcast(sm20, mynn_src1); sm25max = mynn_lane_bcast(sm25max, mynn_src1); sm25min = mynn_lane_bcast(sm25min, mynn_src1); wt = mynn_lane_bcast(wt, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3147
dfm(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3148
dfh(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3149
dfq(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3150
pdk(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3151
pdt(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3152
pdq(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3153
pdc(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3154
tcd(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3155
qcd(kts) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3157
tcd(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3158
qcd(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3161
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3162
dzk = dz(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3163
tcd(k) = r6_div(r6_sub(tcd(((k) + (1))), tcd(k)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3164
qcd(k) = r6_div(r6_sub(qcd(((k) + (1))), qcd(k)), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3165
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3167
if (((spp_pbl) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3168
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3169
dfm(k) = r6_add(dfm(k), r6_mul(r6_mul(r6_mul(dfm(k), pattern_spp_pbl1(k)), 1.5f), mynn_max((float)(mynn_exp(r6_neg(r6_div(mynn_max((float)(zero), (float)(r6_sub(zw(k), 8000.f))), 2000.f)))), (float)(0.001f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3170
dfh(k) = r6_add(dfh(k), r6_mul(r6_mul(r6_mul(dfh(k), pattern_spp_pbl1(k)), 1.5f), mynn_max((float)(mynn_exp(r6_neg(r6_div(mynn_max((float)(zero), (float)(r6_sub(zw(k), 8000.f))), 2000.f)))), (float)(0.001f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3171
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3172
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3228-3656
__device__ __noinline__ void f_mym_predict(int kts, int kte, float closure, float delt, MynnArray<float, 1> dz, float ust, float flt, float flq, float pmz, float phh, MynnArray<float, 1> el, MynnArray<float, 1> dfq, MynnArray<float, 1> rho, MynnArray<float, 1> pdk, MynnArray<float, 1> pdt, MynnArray<float, 1> pdq, MynnArray<float, 1> pdc, MynnArray<float, 1> qke, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> s_aw1, MynnArray<float, 1> s_awqke1, MynnArray<float, 1> s_awqsq1, int bl_mynn_edmf, int bl_mynn_edmf_tke, MynnArray<float, 1> qwt1, MynnArray<float, 1> qdiss1, int tke_budget) {
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
s_awqsq1.rebase({kts});
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3271
if (((bl_mynn_edmf_tke) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3272
onoff = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3273
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3274
onoff = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3275
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3278
vkz = r6_mul(r6_mul(karman, p5), dz(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3282
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3284
qkw(k) = mynn_sqrt(mynn_max((float)(qke(k)), (float)(qkemin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3285
df3q(k) = r6_mul(sqfac, dfq(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3286
dtz(k) = r6_div(delt, dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3287
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3292
rhoz(kts) = rho(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3293
rhoinv(kts) = r6_div(one, rho(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3294
kqdz(kts) = r6_mul(rhoz(kts), df3q(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3295
kmdz(kts) = r6_mul(rhoz(kts), dfq(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3296
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3297
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3298
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3299
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3300
kqdz(k) = r6_mul(rhoz(k), df3q(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3301
kmdz(k) = r6_mul(rhoz(k), dfq(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3302
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3303
rhoz(((kte) + (1))) = rhoz(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3304
kqdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), df3q(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3305
kmdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfq(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3307
if (((bl_mynn_edmf) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3309
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3310
kqdz(k) = mynn_max((float)(kqdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3311
kqdz(k) = mynn_max((float)(kqdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3312
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3313
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3318
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3319
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3324
pdk1 = r6_div(r6_mul(r6_mul(two, mynn_ipow<3>(ust)), pmz), vkz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3325
phm = r6_div(r6_mul(r6_div(two, ust), phh), vkz);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3326
pdt1 = r6_mul(phm, mynn_ipow<2>(flt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3327
pdq1 = r6_mul(phm, mynn_ipow<2>(flq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3328
pdc1 = r6_mul(r6_mul(phm, flt), flq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3331
pdk(kts) = r6_sub(pdk1, pdk(((kts) + (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3336
pdt(kts) = pdt(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3337
pdq(kts) = pdq(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3338
pdc(kts) = pdc(((kts) + (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3344
{ const int mynn_lo3 = (kts), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3345
b1l = r6_mul(r6_mul(b1, p5), r6_add(el(((k) + (1))), el(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3346
bp(k) = r6_div(r6_mul(two, qkw(k)), b1l);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3347
rp(k) = r6_add(pdk(((k) + (1))), pdk(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3348
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; b1l = mynn_lane_bcast(b1l, mynn_src3); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3349
bp(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3352
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3354
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3355
upcont(k) = r6_sub(s_awqke1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qke(k), upwind), r6_mul(qke(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3356
dncont(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3357
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3359
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3360
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3362
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3363
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3364
b(1) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), kqdz(((k) + (1)))), rhoinv(k))), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3365
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), kqdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3366
d(1) = r6_sub(r6_add(mynn_max((float)(qke(k)), (float)(qkemin)), r6_mul(rp(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3369
{ const int mynn_lo5 = (((kts) + (1))), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3370
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kqdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3371
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kqdz(k), kqdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3372
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kqdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3373
d(k) = r6_sub(r6_add(mynn_max((float)(qke(k)), (float)(qkemin)), r6_mul(rp(k), delt)), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3375
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3377
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3378
{ const int mynn_lo6 = (kts), mynn_hi6 = (((kte) - (1))), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3384
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), kqdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), s_aw1(k)), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3386
b(k) = r6_add(r6_add(r6_add(1.f, r6_mul(r6_mul(dtz(k), r6_add(kqdz(k), kqdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), onoff)), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3389
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kqdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3391
d(k) = r6_add(r6_add(r6_mul(rp(k), delt), qke(k)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqke1(k), s_awqke1(((k) + (1))))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3393
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3394
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3402
a(kte) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3403
b(kte) = 1.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3404
c(kte) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3405
d(kte) = qke(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3408
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3410
{ const int mynn_lo7 = (kts), mynn_hi7 = (kte), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3412
qke(k) = mynn_max((float)(x(k)), (float)(qkemin));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3413
qke(k) = mynn_min((float)(qke(k)), (float)(150.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3414
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3418
if (((tke_budget) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3420
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tke_up).size(); mynn_element += lane_width) { tke_up.linear(mynn_element) = r6_mul(p5, qke.linear(mynn_element)); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3421
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dzinv).size(); mynn_element += lane_width) { dzinv.linear(mynn_element) = r6_div(one, dz.linear(mynn_element)); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3422
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3423
qwt1(k) = r6_mul(dzinv(k), r6_add(r6_sub(r6_mul(kqdz(((k) + (1))), r6_sub(tke_up(((k) + (1))), tke_up(k))), r6_mul(kqdz(k), tke_up(k))), r6_mul(r6_mul(r6_mul(p5, rhoinv(k)), r6_add(r6_add(r6_mul(s_aw1(((k) + (1))), tke_up(((k) + (1)))), r6_mul(r6_sub(s_aw1(((k) + (1))), s_aw1(k)), tke_up(k))), r6_sub(s_awqke1(k), s_awqke1(((k) + (1)))))), onoff)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3428
{ const int mynn_lo8 = (((kts) + (1))), mynn_hi8 = (((kte) - (1))), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3429
qwt1(k) = r6_mul(dzinv(k), r6_add(r6_sub(r6_mul(kqdz(((k) + (1))), r6_sub(tke_up(((k) + (1))), tke_up(k))), r6_mul(kqdz(k), r6_sub(tke_up(k), tke_up(((k) - (1)))))), r6_mul(r6_mul(r6_mul(p5, rhoinv(k)), r6_add(r6_sub(r6_add(r6_mul(s_aw1(((k) + (1))), tke_up(((k) + (1)))), r6_mul(r6_sub(s_aw1(((k) + (1))), s_aw1(k)), tke_up(k))), r6_mul(s_aw1(k), tke_up(((k) - (1))))), r6_sub(s_awqke1(k), s_awqke1(((k) + (1)))))), onoff)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3435
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3436
k = kte;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3437
qwt1(k) = r6_mul(dzinv(k), r6_add(r6_neg(r6_mul(kqdz(k), r6_sub(tke_up(k), tke_up(((k) - (1)))))), r6_mul(r6_mul(r6_mul(p5, rhoinv(k)), r6_add(r6_sub(r6_neg(r6_mul(s_aw1(k), tke_up(k))), r6_mul(s_aw1(k), tke_up(((k) - (1))))), s_awqke1(k))), onoff)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3440
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qdiss1).size(); mynn_element += lane_width) { qdiss1.linear(mynn_element) = r6_mul(bp.linear(mynn_element), tke_up.linear(mynn_element)); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3441
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3444
if (((closure) > (2.5f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3448
{ const int mynn_lo9 = (kts), mynn_hi9 = (((kte) - (1))), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3449
b2l = r6_mul(r6_mul(b2, p5), r6_add(el(((k) + (1))), el(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3450
bp(k) = r6_div(r6_mul(two, qkw(k)), b2l);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3451
rp(k) = r6_add(pdq(((k) + (1))), pdq(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3452
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; if (mynn_n9 > 0) { const int mynn_src9 = (mynn_n9 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src9); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3453
bp(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3456
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3457
{ const int mynn_lo10 = (((kts) + (1))), mynn_hi10 = (((kte) - (1))), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3458
upcont(k) = r6_sub(s_awqsq1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qsq(k), upwind), r6_mul(qsq(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3459
dncont(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3460
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3462
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3463
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3465
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3466
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3467
b(1) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), kmdz(((k) + (1)))), rhoinv(k))), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3468
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3469
d(1) = r6_sub(r6_add(mynn_max((float)(zero), (float)(qsq(k))), r6_mul(rp(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3472
{ const int mynn_lo11 = (((kts) + (1))), mynn_hi11 = (((kte) - (1))), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { k = mynn_lo11 + mynn_it11*mynn_st11;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3473
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3474
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3475
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3476
d(k) = r6_sub(r6_add(mynn_max((float)(zero), (float)(qsq(k))), r6_mul(rp(k), delt)), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3478
} mynn_lane_sync(); k = mynn_lo11 + mynn_n11*mynn_st11; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3480
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3482
{ const int mynn_lo12 = (kts), mynn_hi12 = (((kte) - (1))), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { k = mynn_lo12 + mynn_it12*mynn_st12;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3483
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), s_aw1(k)), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3485
b(k) = r6_add(r6_add(r6_add(1.f, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), onoff)), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3488
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3490
d(k) = r6_add(r6_add(r6_mul(rp(k), delt), qsq(k)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqsq1(k), s_awqsq1(((k) + (1))))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3492
} mynn_lane_sync(); k = mynn_lo12 + mynn_n12*mynn_st12; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3493
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3495
a(kte) = r6_neg(one);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3496
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3497
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3498
d(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3501
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3503
{ const int mynn_lo13 = (kts), mynn_hi13 = (kte), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { k = mynn_lo13 + mynn_it13*mynn_st13;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3505
qsq(k) = mynn_max((float)(x(k)), (float)(1e-17f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3506
qsq(k) = mynn_min((float)(qsq(k)), (float)(6e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3507
} mynn_lane_sync(); k = mynn_lo13 + mynn_n13*mynn_st13; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3509
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3511
{ const int mynn_lo14 = (kts), mynn_hi14 = (((kte) - (1))), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { k = mynn_lo14 + mynn_it14*mynn_st14;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3512
if (((qkw(k)) <= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3513
b2l = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3514
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3515
b2l = r6_div(r6_mul(r6_mul(b2, 0.25f), r6_add(el(((k) + (1))), el(k))), qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3516
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3517
qsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdq(((k) + (1))), pdq(k)))), (float)(1e-17f))), (float)(6e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3518
} mynn_lane_sync(); k = mynn_lo14 + mynn_n14*mynn_st14; if (mynn_n14 > 0) { const int mynn_src14 = (mynn_n14 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src14); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3519
qsq(kte) = qsq(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3520
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3524
if (((closure) >= (2.7f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3530
{ const int mynn_lo15 = (kts), mynn_hi15 = (((kte) - (1))), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { k = mynn_lo15 + mynn_it15*mynn_st15;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3531
b2l = r6_mul(r6_mul(b2, p5), r6_add(el(((k) + (1))), el(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3532
bp(k) = r6_div(r6_mul(two, qkw(k)), b2l);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3533
rp(k) = r6_add(pdt(((k) + (1))), pdt(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3534
} mynn_lane_sync(); k = mynn_lo15 + mynn_n15*mynn_st15; if (mynn_n15 > 0) { const int mynn_src15 = (mynn_n15 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src15); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3544
{ const int mynn_lo16 = (kts), mynn_hi16 = (((kte) - (1))), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { k = mynn_lo16 + mynn_it16*mynn_st16;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3550
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3551
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3552
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3553
d(k) = r6_add(r6_mul(rp(k), delt), mynn_max((float)(zero), (float)(tsq(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3554
} mynn_lane_sync(); k = mynn_lo16 + mynn_n16*mynn_st16; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3563
a(kte) = r6_neg(1.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3564
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3565
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3566
d(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3569
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3571
{ const int mynn_lo17 = (kts), mynn_hi17 = (kte), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { k = mynn_lo17 + mynn_it17*mynn_st17;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3573
tsq(k) = mynn_min((float)(mynn_max((float)(zero), (float)(x(k)))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3574
} mynn_lane_sync(); k = mynn_lo17 + mynn_n17*mynn_st17; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3575
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3577
{ const int mynn_lo18 = (kts), mynn_hi18 = (((kte) - (1))), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { k = mynn_lo18 + mynn_it18*mynn_st18;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3578
if (((qkw(k)) <= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3579
b2l = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3580
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3581
b2l = r6_div(r6_mul(r6_mul(b2, 0.25f), r6_add(el(((k) + (1))), el(k))), qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3582
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3583
tsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdt(((k) + (1))), pdt(k)))), (float)(zero))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3584
} mynn_lane_sync(); k = mynn_lo18 + mynn_n18*mynn_st18; if (mynn_n18 > 0) { const int mynn_src18 = (mynn_n18 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src18); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3585
tsq(kte) = tsq(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3586
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3589
if (((closure) >= (3.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3591
{ const int mynn_lo19 = (kts), mynn_hi19 = (((kte) - (1))), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { k = mynn_lo19 + mynn_it19*mynn_st19;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3592
b2l = r6_mul(r6_mul(b2, p5), r6_add(el(((k) + (1))), el(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3593
bp(k) = r6_div(r6_mul(two, qkw(k)), b2l);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3594
rp(k) = r6_add(pdc(((k) + (1))), pdc(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3595
} mynn_lane_sync(); k = mynn_lo19 + mynn_n19*mynn_st19; if (mynn_n19 > 0) { const int mynn_src19 = (mynn_n19 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src19); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3604
{ const int mynn_lo20 = (kts), mynn_hi20 = (((kte) - (1))), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { k = mynn_lo20 + mynn_it20*mynn_st20;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3610
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3611
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3612
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3613
d(k) = r6_add(r6_mul(rp(k), delt), cov(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3614
} mynn_lane_sync(); k = mynn_lo20 + mynn_n20*mynn_st20; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3623
a(kte) = r6_neg(1.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3624
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3625
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3626
d(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3629
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3631
{ const int mynn_lo21 = (kts), mynn_hi21 = (kte), mynn_st21 = (1); const int mynn_n21 = mynn_trip(mynn_lo21, mynn_hi21, mynn_st21); mynn_lane_sync(); for (int mynn_it21 = lane_sub; mynn_it21 < mynn_n21; mynn_it21 += lane_width) { k = mynn_lo21 + mynn_it21*mynn_st21;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3633
cov(k) = x(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3634
} mynn_lane_sync(); k = mynn_lo21 + mynn_n21*mynn_st21; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3636
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3639
{ const int mynn_lo22 = (kts), mynn_hi22 = (((kte) - (1))), mynn_st22 = (1); const int mynn_n22 = mynn_trip(mynn_lo22, mynn_hi22, mynn_st22); mynn_lane_sync(); for (int mynn_it22 = lane_sub; mynn_it22 < mynn_n22; mynn_it22 += lane_width) { k = mynn_lo22 + mynn_it22*mynn_st22;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3640
if (((qkw(k)) <= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3641
b2l = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3642
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3643
b2l = r6_div(r6_mul(r6_mul(b2, 0.25f), r6_add(el(((k) + (1))), el(k))), qkw(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3644
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3645
cov(k) = r6_mul(b2l, r6_add(pdc(((k) + (1))), pdc(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3646
} mynn_lane_sync(); k = mynn_lo22 + mynn_n22*mynn_st22; if (mynn_n22 > 0) { const int mynn_src22 = (mynn_n22 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src22); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3647
cov(kte) = cov(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3649
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3692-4152
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3769
for (k=((kte) - (3)); ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3770
theta1 = th(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3771
theta2 = th(((k) + (2)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3772
ht1 = r6_mul(44307.692f, r6_sub(one, mynn_pow(r6_div(p(k), 101325.f), 0.190f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3773
ht2 = r6_mul(44307.692f, r6_sub(one, mynn_pow(r6_div(p(((k) + (2))), 101325.f), 0.190f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3774
if (((((((r6_div(r6_sub(theta2, theta1), r6_sub(ht2, ht1))) < (r6_div(10.f, 1500.f)))) && (((ht1) < (19000.f))))) && (((ht1) > (4000.f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3776
goto mynn_label_86;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3777
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3778
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3779
mynn_label_86: ;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3780
k_tropo = mynn_max((int)(((kts) + (2))), (int)(((k) + (2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3782
zagl = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3784
switch (bl_mynn_cloudpdf) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3786
case 0: {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3788
{ const int mynn_lo0 = (kts), mynn_hi0 = (((kte) - (1))), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3789
t = r6_mul(th(k), exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3802
esat = f_esat_blend(t);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3805
qsl = r6_div(r6_mul(ep_2, esat), mynn_max((float)(r6_sub(p(k), r6_mul(ep_3, esat))), (float)(1.e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3807
dqsl = r6_div(r6_mul(r6_mul(qsl, ep_2), xlv), r6_mul(r_d, mynn_ipow<2>(t)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3809
alp(k) = r6_div(one, r6_add(one, r6_mul(dqsl, xlvcp)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3810
bet(k) = r6_mul(dqsl, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3814
t3sq = mynn_max((float)(zero), (float)(tsq(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3815
r3sq = mynn_max((float)(zero), (float)(qsq(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3816
c3sq = cov(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3817
c3sq = mynn_sign(mynn_min((double)(mynn_abs(c3sq)), (double)(mynn_sqrt(mynn_dmul(t3sq, r3sq)))), c3sq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3818
r3sq = mynn_dsub(mynn_dadd(r3sq, mynn_dmul(mynn_ipow<2>(bet(k)), t3sq)), mynn_dmul(r6_mul(2.0f, bet(k)), c3sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3820
qmq = r6_sub(qw(k), qsl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3822
sgm(k) = mynn_sqrt(mynn_max((double)(r3sq), (double)(1.0e-10)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3824
q1(k) = r6_div(qmq, sgm(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3826
cldfra_bl1(k) = r6_mul(0.5f, r6_add(one, mynn_erf(r6_mul(q1(k), rr2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3828
q1k = q1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3829
eq1 = r6_mul(rrp, mynn_exp(r6_mul(r6_neg(r6_mul(0.5f, q1k)), q1k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3830
qll = mynn_max((float)(zero), (float)(r6_add(r6_mul(cldfra_bl1(k), q1k), eq1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3832
ql(k) = r6_mul(r6_mul(alp(k), sgm(k)), qll);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3834
liq_frac = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(t, 240.0f), 29.0f)), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3835
qc_bl1(k) = r6_mul(liq_frac, ql(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3836
qi_bl1(k) = r6_mul(r6_sub(one, liq_frac), ql(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3839
q2p = r6_div(xlvcp, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3840
pt = r6_add(thl(k), r6_mul(q2p, ql(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3843
qt = r6_sub(r6_add(one, r6_mul(p608, qw(k))), r6_mul(r6_mul(r6_add(1.f, p608), r6_add(qc_bl1(k), qi_bl1(k))), cldfra_bl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3844
rac = r6_mul(r6_mul(alp(k), r6_sub(cldfra_bl1(k), r6_mul(qll, eq1))), r6_sub(r6_mul(q2p, qt), r6_mul(r6_add(1.f, p608), pt)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3849
vt(k) = r6_sub(r6_sub(qt, one), r6_mul(rac, bet(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3850
vq(k) = r6_add(r6_sub(r6_mul(p608, pt), tv0), rac);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3852
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; c3sq = mynn_lane_bcast(c3sq, mynn_src0); dqsl = mynn_lane_bcast(dqsl, mynn_src0); eq1 = mynn_lane_bcast(eq1, mynn_src0); esat = mynn_lane_bcast(esat, mynn_src0); liq_frac = mynn_lane_bcast(liq_frac, mynn_src0); pt = mynn_lane_bcast(pt, mynn_src0); q1k = mynn_lane_bcast(q1k, mynn_src0); q2p = mynn_lane_bcast(q2p, mynn_src0); qll = mynn_lane_bcast(qll, mynn_src0); qmq = mynn_lane_bcast(qmq, mynn_src0); qsl = mynn_lane_bcast(qsl, mynn_src0); qt = mynn_lane_bcast(qt, mynn_src0); r3sq = mynn_lane_bcast(r3sq, mynn_src0); rac = mynn_lane_bcast(rac, mynn_src0); t = mynn_lane_bcast(t, mynn_src0); t3sq = mynn_lane_bcast(t3sq, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3854
break; } case 1: case (-(1)): {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3856
{ const int mynn_lo1 = (kts), mynn_hi1 = (((kte) - (1))), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3857
t = r6_mul(th(k), exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3859
esat = f_esat_blend(t);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3862
qsl = r6_div(r6_mul(ep_2, esat), mynn_max((float)(r6_sub(p(k), r6_mul(ep_3, esat))), (float)(1.e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3864
dqsl = r6_div(r6_mul(r6_mul(qsl, ep_2), xlv), r6_mul(r_d, mynn_ipow<2>(t)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3866
alp(k) = r6_div(one, r6_add(one, r6_mul(dqsl, xlvcp)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3867
bet(k) = r6_mul(dqsl, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3869
if (((k) == (kts))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3870
dzk = r6_mul(0.5f, dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3871
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3872
dzk = dz(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3873
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3874
dth = r6_sub(r6_mul(0.5f, r6_add(thl(((k) + (1))), thl(k))), r6_mul(0.5f, r6_add(thl(k), thl(mynn_max((int)(((k) - (1))), (int)(kts))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3875
dqw = r6_sub(r6_mul(0.5f, r6_add(qw(((k) + (1))), qw(k))), r6_mul(0.5f, r6_add(qw(k), qw(mynn_max((int)(((k) - (1))), (int)(kts))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3876
sgm(k) = mynn_sqrt(mynn_max((float)(r6_mul(r6_div(r6_mul(r6_mul(r6_mul(mynn_ipow<2>(alp(k)), mynn_max((float)(mynn_ipow<2>(el(k))), (float)(0.1f))), b2), mynn_max((float)(sh(k)), (float)(0.03f))), 4.f), mynn_ipow<2>(r6_sub(r6_div(dqw, dzk), r6_mul(bet(k), r6_div(dth, dzk)))))), (float)(1.0e-10f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3879
qmq = r6_sub(qw(k), qsl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3880
q1(k) = r6_div(qmq, sgm(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3881
cldfra_bl1(k) = r6_mul(0.5f, r6_add(one, mynn_erf(r6_mul(q1(k), rr2))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3886
q1k = q1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3887
eq1 = r6_mul(rrp, mynn_exp(r6_mul(r6_neg(r6_mul(0.5f, q1k)), q1k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3888
qll = mynn_max((float)(zero), (float)(r6_add(r6_mul(cldfra_bl1(k), q1k), eq1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3890
ql(k) = r6_mul(r6_mul(alp(k), sgm(k)), qll);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3891
liq_frac = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(t, 240.0f), 29.0f)), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3892
qc_bl1(k) = r6_mul(liq_frac, ql(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3893
qi_bl1(k) = r6_mul(r6_sub(one, liq_frac), ql(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3896
q2p = r6_div(xlvcp, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3897
pt = r6_add(thl(k), r6_mul(q2p, ql(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3900
qt = r6_sub(r6_add(one, r6_mul(p608, qw(k))), r6_mul(r6_mul(r6_add(1.f, p608), r6_add(qc_bl1(k), qi_bl1(k))), cldfra_bl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3901
rac = r6_mul(r6_mul(alp(k), r6_sub(cldfra_bl1(k), r6_mul(qll, eq1))), r6_sub(r6_mul(q2p, qt), r6_mul(r6_add(1.f, p608), pt)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3906
vt(k) = r6_sub(r6_sub(qt, one), r6_mul(rac, bet(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3907
vq(k) = r6_add(r6_sub(r6_mul(p608, pt), tv0), rac);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3909
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; dqsl = mynn_lane_bcast(dqsl, mynn_src1); dqw = mynn_lane_bcast(dqw, mynn_src1); dth = mynn_lane_bcast(dth, mynn_src1); dzk = mynn_lane_bcast(dzk, mynn_src1); eq1 = mynn_lane_bcast(eq1, mynn_src1); esat = mynn_lane_bcast(esat, mynn_src1); liq_frac = mynn_lane_bcast(liq_frac, mynn_src1); pt = mynn_lane_bcast(pt, mynn_src1); q1k = mynn_lane_bcast(q1k, mynn_src1); q2p = mynn_lane_bcast(q2p, mynn_src1); qll = mynn_lane_bcast(qll, mynn_src1); qmq = mynn_lane_bcast(qmq, mynn_src1); qsl = mynn_lane_bcast(qsl, mynn_src1); qt = mynn_lane_bcast(qt, mynn_src1); rac = mynn_lane_bcast(rac, mynn_src1); t = mynn_lane_bcast(t, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3911
break; } case 2: case (-(2)): {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3915
pblh2 = mynn_max((float)(pblh1), (float)(ten));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3916
{ const int mynn_lo2 = (kts), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3917
zagl = r6_add(zw(k), r6_mul(p5, dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3919
t = r6_mul(th(k), exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3920
xl = f_xl_blend(t);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3921
qsat_tk = f_qsat_blend(t, p(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3922
rh(k) = mynn_max((float)(mynn_min((float)(r6_div(qw(k), mynn_max((float)(qsat_tk), (float)(1e-10f)))), (float)(rhmax))), (float)(0.001f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3925
dqsl = r6_div(r6_mul(r6_mul(qsat_tk, ep_2), xlv), r6_mul(r_d, mynn_ipow<2>(t)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3926
alp(k) = r6_div(one, r6_add(one, r6_mul(dqsl, xlvcp)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3927
bet(k) = r6_mul(dqsl, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3929
rsl = r6_div(r6_mul(xl, qsat_tk), r6_mul(r_v, mynn_ipow<2>(t)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3931
cpm = r6_add(cp, r6_mul(qw(k), cpv));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3932
a(k) = r6_div(one, r6_add(one, r6_div(r6_mul(xl, rsl), cpm)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3933
b(k) = r6_mul(a(k), rsl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3936
qw_pert = r6_add(qw(k), r6_mul(r6_mul(r6_mul(qw(k), p5), pattern_spp_pbl1(k)), ((float)(spp_pbl))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3939
qmq = r6_sub(qw_pert, qsat_tk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3943
r3sq = mynn_max((float)(zero), (float)(qsq(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3944
t3sq = mynn_max((float)(zero), (float)(tsq(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3945
c3sq = cov(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3946
c3sq = mynn_sign(mynn_min((double)(mynn_abs(c3sq)), (double)(mynn_sqrt(mynn_dmul(t3sq, r3sq)))), c3sq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3948
if (((closure) < (2.7f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3949
r3sq = r3sq;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3950
} else if (((((closure) >= (2.7f))) && (((closure) < (3.0f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3951
r3sq = mynn_dadd(r3sq, mynn_dmul(mynn_ipow<2>(bet(k)), t3sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3952
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3953
r3sq = mynn_dsub(mynn_dadd(r3sq, mynn_dmul(mynn_ipow<2>(bet(k)), t3sq)), mynn_dmul(r6_mul(two, bet(k)), c3sq));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3954
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3957
sgm(k) = mynn_max((float)(mynn_sqrt(((float)(r3sq)))), (float)(1e-13f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3959
sgm(k) = mynn_min((float)(sgm(k)), (float)(r6_mul(qw(k), p5)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3962
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(dz(k), 100.f)), (float)(zero)), 500.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3963
sgm(k) = r6_add(sgm(k), r6_mul(r6_mul(sgm(k), p2), wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3965
sgmq = sgm(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3966
sgmc = sgm(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3969
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, r6_add(pblh2, 10.f))), (float)(zero)), 250.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3970
clim = r6_add(r6_mul(clim_pbl, r6_sub(one, wt)), r6_mul(clim_trp, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3971
zsl = mynn_min((float)(mynn_max((float)(r6_mul(p1, pblh2)), (float)(50.f))), (float)(150.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3972
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, zsl)), (float)(zero)), 150.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3973
clim = r6_add(r6_mul(clim_sfc, r6_sub(one, wt)), r6_mul(clim, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3974
sgmc = mynn_max((float)(sgmc), (float)(r6_mul(qw(k), clim)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3976
sgmc = mynn_max((float)(sgmc), (float)(1e-13f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3978
if (((qmq) >= (zero))) { sgmc = mynn_max((float)(r6_mul(0.02f, qw(k))), (float)(sgmc)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3980
q1(k) = r6_div(qmq, sgmc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3984
wt2 = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, pblh2)), (float)(zero)), 300.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3986
if (((((r6_add(qi(k), qs(k))) > (1.e-10f))) && (((zagl) > (pblh2))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3987
rh_adj = mynn_min((float)(r6_add(rhcrit, r6_mul(r6_mul(wt2, 0.037f), mynn_max((float)(r6_add(ten, mynn_log10(r6_add(qi(k), qs(k))))), (float)(zero))))), (float)(rhmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3988
rh_adj = mynn_max((float)(rh(k)), (float)(rh_adj));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3990
q1_rh = r6_add(r6_neg(3.f), r6_div(r6_mul(three, r6_sub(rh_adj, rhcrit)), r6_sub(one, rhcrit)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3991
q1(k) = mynn_max((float)(q1_rh), (float)(q1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3992
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3994
if (((((qc(k)) > (1.e-5f))) && (((zagl) > (pblh2))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3995
rh_adj = mynn_min((float)(r6_add(rhcrit, r6_mul(r6_mul(wt2, 0.07f), mynn_max((float)(r6_add(five, mynn_log10(qc(k)))), (float)(zero))))), (float)(rhmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3996
rh_adj = mynn_max((float)(rh(k)), (float)(rh_adj));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3998
q1_rh = r6_add(r6_neg(3.f), r6_div(r6_mul(three, r6_sub(rh_adj, rhcrit)), r6_sub(one, rhcrit)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:3999
q1(k) = mynn_max((float)(q1_rh), (float)(q1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4000
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4002
q1k = q1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4016
wt2 = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(zagl, r6_sub(pblh1, 100.f)), 200.f)), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4019
cldfra_qsq0 = mynn_max((float)(zero), (float)(mynn_min((float)(r6_add(p5, r6_mul(0.35f, mynn_atan(r6_mul(3.6f, r6_add(q1k, 0.05f)))))), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4021
cldfra_qsq1 = mynn_max((float)(zero), (float)(mynn_min((float)(r6_add(p5, r6_mul(0.39f, mynn_atan(r6_mul(1.6f, r6_add(q1k, 0.55f)))))), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4022
cldfra_qsq = r6_add(r6_mul(cldfra_qsq0, r6_sub(one, wt2)), r6_mul(cldfra_qsq1, wt2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4025
cldfra_rh0 = mynn_min((float)(mynn_max((float)(r6_add(r6_mul(0.56f, mynn_tanh(r6_div(r6_sub(rh(k), 0.976f), 0.030f))), p5)), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4026
cldfra_rh1 = mynn_min((float)(mynn_max((float)(r6_add(r6_mul(0.55f, mynn_tanh(r6_div(r6_sub(rh(k), 0.955f), 0.055f))), p5)), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4027
cldfra_rh = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4029
cldfra_bl1(k) = mynn_max((float)(cldfra_qsq), (float)(cldfra_rh));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4033
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, r6_add(pblh2, ten))), (float)(zero)), 300.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4034
qlim = r6_add(r6_mul(qlim_pbl, r6_sub(one, wt)), r6_mul(qlim_trp, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4035
zsl = mynn_min((float)(mynn_max((float)(r6_mul(p1, pblh2)), (float)(50.f))), (float)(150.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4036
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, zsl)), (float)(zero)), 200.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4037
qlim = r6_add(r6_mul(qlim_sfc, r6_sub(one, wt)), r6_mul(qlim, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4038
sgmq = mynn_max((float)(sgmq), (float)(r6_mul(qw(k), qlim)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4040
ql_water = r6_mul(mynn_min((float)(sgmq), (float)(r6_mul(0.025f, qw(k)))), cldfra_bl1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4041
ql_ice = r6_mul(mynn_min((float)(sgmq), (float)(r6_mul(0.025f, qw(k)))), cldfra_bl1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4065
if (((cldfra_bl1(k)) < (0.001f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4066
ql_ice = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4067
ql_water = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4068
cldfra_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4069
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4071
liq_frac = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(t, tice), r6_sub(tliq, tice))), (float)(zero))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4072
qc_bl1(k) = r6_mul(liq_frac, ql_water);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4073
qi_bl1(k) = r6_mul(r6_sub(one, liq_frac), ql_ice);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4077
if (((k) >= (k_tropo))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4078
cldfra_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4079
qc_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4080
qi_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4081
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4087
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4088
q1k = mynn_max((float)(q1(k)), (float)(r6_neg(2.5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4089
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4090
q1k = mynn_max((float)(q1(k)), (float)(r6_neg(2.0f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4091
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4094
if (((q1k) >= (one))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4095
fng = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4096
} else if (((((q1k) >= (r6_neg(1.7f)))) && (((q1k) < (one))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4097
fng = mynn_exp(r6_neg(r6_mul(0.4f, r6_sub(q1k, one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4098
} else if (((((q1k) >= (r6_neg(2.5f)))) && (((q1k) < (r6_neg(1.7f)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4099
fng = r6_add(three, mynn_exp(r6_neg(r6_mul(3.8f, r6_add(q1k, 1.7f)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4100
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4101
fng = mynn_min((float)(r6_add(23.9f, mynn_exp(r6_neg(r6_mul(1.6f, r6_add(q1k, 2.5f)))))), (float)(60.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4102
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4104
cfmax = mynn_min((float)(cldfra_bl1(k)), (float)(0.6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4106
zsl = mynn_min((float)(mynn_max((float)(r6_mul(p1, pblh2)), (float)(25.f))), (float)(hundred));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4107
wt = mynn_min((float)(r6_div(zagl, zsl)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4108
cfmax = r6_mul(cfmax, wt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4110
bb = r6_div(r6_mul(b(k), t), th(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4116
qww = r6_add(one, r6_mul(p608, qw(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4117
alpha = r6_mul(p608, th(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4118
beta = r6_sub(r6_mul(r6_div(th(k), t), r6_div(xl, cp)), r6_mul(1.61f, th(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4119
vt(k) = r6_sub(r6_sub(qww, r6_mul(r6_mul(r6_mul(cfmax, beta), bb), fng)), one);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4120
vq(k) = r6_sub(r6_add(alpha, r6_mul(r6_mul(r6_mul(cfmax, beta), a(k)), fng)), tv0);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4126
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; alpha = mynn_lane_bcast(alpha, mynn_src2); bb = mynn_lane_bcast(bb, mynn_src2); beta = mynn_lane_bcast(beta, mynn_src2); c3sq = mynn_lane_bcast(c3sq, mynn_src2); cfmax = mynn_lane_bcast(cfmax, mynn_src2); cldfra_qsq = mynn_lane_bcast(cldfra_qsq, mynn_src2); cldfra_qsq0 = mynn_lane_bcast(cldfra_qsq0, mynn_src2); cldfra_qsq1 = mynn_lane_bcast(cldfra_qsq1, mynn_src2); cldfra_rh = mynn_lane_bcast(cldfra_rh, mynn_src2); cldfra_rh0 = mynn_lane_bcast(cldfra_rh0, mynn_src2); cldfra_rh1 = mynn_lane_bcast(cldfra_rh1, mynn_src2); clim = mynn_lane_bcast(clim, mynn_src2); cpm = mynn_lane_bcast(cpm, mynn_src2); dqsl = mynn_lane_bcast(dqsl, mynn_src2); fng = mynn_lane_bcast(fng, mynn_src2); liq_frac = mynn_lane_bcast(liq_frac, mynn_src2); q1k = mynn_lane_bcast(q1k, mynn_src2); ql_ice = mynn_lane_bcast(ql_ice, mynn_src2); ql_water = mynn_lane_bcast(ql_water, mynn_src2); qlim = mynn_lane_bcast(qlim, mynn_src2); qmq = mynn_lane_bcast(qmq, mynn_src2); qsat_tk = mynn_lane_bcast(qsat_tk, mynn_src2); qw_pert = mynn_lane_bcast(qw_pert, mynn_src2); qww = mynn_lane_bcast(qww, mynn_src2); r3sq = mynn_lane_bcast(r3sq, mynn_src2); rsl = mynn_lane_bcast(rsl, mynn_src2); sgmc = mynn_lane_bcast(sgmc, mynn_src2); sgmq = mynn_lane_bcast(sgmq, mynn_src2); t = mynn_lane_bcast(t, mynn_src2); t3sq = mynn_lane_bcast(t3sq, mynn_src2); wt = mynn_lane_bcast(wt, mynn_src2); wt2 = mynn_lane_bcast(wt2, mynn_src2); xl = mynn_lane_bcast(xl, mynn_src2); zagl = mynn_lane_bcast(zagl, mynn_src2); zsl = mynn_lane_bcast(zsl, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4128
break; } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4131
if (((bl_mynn_cloudpdf) < (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4132
{ const int mynn_lo3 = (kts), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4133
cldfra_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4134
qc_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4135
qi_bl1(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4136
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4137
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4139
ql(kte) = ql(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4140
vt(kte) = vt(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4141
vq(kte) = vq(((kte) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4142
qc_bl1(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4143
qi_bl1(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4144
cldfra_bl1(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4145
return;
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4158-5733
__device__ __noinline__ void f_mynn_tendencies(int kts, int kte, int i, float delt, MynnArray<float, 1> dz, MynnArray<float, 1> zw, float xland, float pblh, MynnArray<float, 1> rho, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> th, MynnArray<float, 1> tk, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> qnc, MynnArray<float, 1> qni, float psfc, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> thl, MynnArray<float, 1> sqv, MynnArray<float, 1> sqc, MynnArray<float, 1> sqi, MynnArray<float, 1> sqs, MynnArray<float, 1> sqw, MynnArray<float, 1> thl_tot1, MynnArray<float, 1> qc_tot1, MynnArray<float, 1> qi_tot1, MynnArray<float, 1> qnwfa, MynnArray<float, 1> qnifa, MynnArray<float, 1> qnbca, MynnArray<float, 1> ozone, float ust, float flt, float flq, float flqv, float flqc, float wspd, float uoce, float voce, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> dfm, MynnArray<float, 1> dfh, MynnArray<float, 1> dfq, MynnArray<float, 1> du, MynnArray<float, 1> dv, MynnArray<float, 1> dth, MynnArray<float, 1> dqv, MynnArray<float, 1> dqc, MynnArray<float, 1> dqi, MynnArray<float, 1> dqs, MynnArray<float, 1> dqnc, MynnArray<float, 1> dqni, MynnArray<float, 1> dqnwfa, MynnArray<float, 1> dqnifa, MynnArray<float, 1> dqnbca, MynnArray<float, 1> dozone, MynnArray<float, 1> diss_heat, MynnArray<float, 1> s_aw1, MynnArray<float, 1> s_awthl1, MynnArray<float, 1> s_awqt1, MynnArray<float, 1> s_awqv1, MynnArray<float, 1> s_awqc1, MynnArray<float, 1> s_awu1, MynnArray<float, 1> s_awv1, MynnArray<float, 1> s_awqnc1, MynnArray<float, 1> s_awqni1, MynnArray<float, 1> s_awqnwfa1, MynnArray<float, 1> s_awqnifa1, MynnArray<float, 1> s_awqnbca1, MynnArray<float, 1> sd_aw1, MynnArray<float, 1> sd_awthl1, MynnArray<float, 1> sd_awqt1, MynnArray<float, 1> sd_awqv1, MynnArray<float, 1> sd_awqc1, MynnArray<float, 1> sd_awqi1, MynnArray<float, 1> sd_awqnc1, MynnArray<float, 1> sd_awqni1, MynnArray<float, 1> sd_awqnwfa1, MynnArray<float, 1> sd_awqnifa1, MynnArray<float, 1> sd_awu1, MynnArray<float, 1> sd_awv1, MynnArray<float, 1> sub_thl, MynnArray<float, 1> sub_sqv, MynnArray<float, 1> sub_u, MynnArray<float, 1> sub_v, MynnArray<float, 1> det_thl, MynnArray<float, 1> det_sqv, MynnArray<float, 1> det_sqc, MynnArray<float, 1> det_u, MynnArray<float, 1> det_v, bool flag_qc, bool flag_qi, bool flag_qnc, bool flag_qni, bool flag_qs, bool flag_qnwfa, bool flag_qnifa, bool flag_qnbca, bool flag_ozone, MynnArray<float, 1> cldfra_bl1, int bl_mynn_cloudmix, int bl_mynn_mixqt, int bl_mynn_edmf, int bl_mynn_edmf_mom, int bl_mynn_mixscalars, int bl_mynn_mixaerosols, int bl_mynn_mixnumcon) {
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
float wfa_max2;
float wt;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4276
dztop = r6_mul(p5, r6_add(dz(kte), dz(((kte) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4281
if (((bl_mynn_edmf_mom) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4282
onoff = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4283
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4284
onoff = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4285
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4288
if (((wspd) <= (1e-6f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4289
ustovwsp = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4290
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4291
ustovwsp = r6_div(ust, wspd);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4292
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4296
rhosfc = r6_div(psfc, r6_mul(r_d, r6_add(tk(kts), r6_mul(p608, qv(kts)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4297
dtz(kts) = r6_div(delt, dz(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4298
rhoz(kts) = rho(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4299
rhoinv(kts) = r6_div(one, rho(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4300
khdz(kts) = r6_mul(rhoz(kts), dfh(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4301
kmdz(kts) = r6_mul(rhoz(kts), dfm(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4302
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4303
dtz(k) = r6_div(delt, dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4304
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4305
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4306
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4307
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4308
khdz(k) = r6_mul(rhoz(k), dfh(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4309
kmdz(k) = r6_mul(rhoz(k), dfm(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4310
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4311
rhoz(((kte) + (1))) = rhoz(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4312
khdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfh(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4313
kmdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfm(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4317
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4320
delp(k) = r6_mul(r6_mul(rho(k), grav), dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4321
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4323
if (((delp(kts)) < (r6_mul(p333, delp(((kts) + (1))))))) { delp(kts) = r6_mul(p333, delp(((kts) + (1)))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4326
if (((bl_mynn_edmf) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4327
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4328
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_mul(p5, r6_add(s_aw1(k), sd_aw1(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4329
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_sub(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(p5, r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4331
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_mul(p5, r6_add(s_aw1(k), sd_aw1(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4332
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_sub(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(p5, r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4334
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4335
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4337
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dth.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { dth.slice(MynnRange{kts, kte}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4342
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4344
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4345
upcont(k) = r6_mul(onoff, r6_sub(s_awu1(k), r6_mul(s_aw1(k), r6_add(r6_mul(u(k), upwind), r6_mul(u(((k) - (1))), r6_sub(one, upwind))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4346
dncont(k) = r6_mul(onoff, r6_sub(sd_awu1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(u(k), upwind), r6_mul(u(((k) - (1))), r6_sub(one, upwind))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4347
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4349
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4350
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4353
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4354
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4355
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_mul(r6_mul(rhosfc, ust), ustovwsp))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4356
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4357
d(1) = r6_sub(r6_add(u(k), r6_mul(r6_div(r6_mul(r6_mul(dtz(k), uoce), mynn_ipow<2>(ust)), wspd), rho(k))), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4360
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4361
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4362
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4363
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4364
d(k) = r6_add(r6_add(r6_sub(u(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_u(k), delt)), r6_mul(det_u(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4366
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4368
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4370
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4372
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4373
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_mul(r6_mul(rhosfc, ust), ustovwsp))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4376
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4379
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(u(k), r6_div(r6_mul(r6_mul(dtz(k), uoce), mynn_ipow<2>(ust)), wspd)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awu1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awu1(((k) + (1)))), onoff)), r6_mul(sub_u(k), delt)), r6_mul(det_u(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4384
{ const int mynn_lo5 = (((kts) + (1))), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4385
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4388
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4391
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4394
d(k) = r6_add(r6_add(r6_sub(r6_add(u(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awu1(k), s_awu1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awu1(k), sd_awu1(((k) + (1))))), onoff)), r6_mul(sub_u(k), delt)), r6_mul(det_u(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4397
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4399
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4414
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4415
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4416
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4417
d(kte) = u(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4420
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4423
{ const int mynn_lo6 = (kts), mynn_hi6 = (kte), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4425
du(k) = r6_div(r6_sub(x(k), u(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4426
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4432
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4434
{ const int mynn_lo7 = (((kts) + (1))), mynn_hi7 = (((kte) - (1))), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4435
upcont(k) = r6_mul(onoff, r6_sub(s_awv1(k), r6_mul(s_aw1(k), r6_add(r6_mul(v(k), upwind), r6_mul(v(((k) - (1))), r6_sub(one, upwind))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4436
dncont(k) = r6_mul(onoff, r6_sub(sd_awv1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(v(k), upwind), r6_mul(v(((k) - (1))), r6_sub(one, upwind))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4437
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4439
upcont(kte) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4440
dncont(kte) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4443
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4444
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4445
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_mul(r6_mul(rhosfc, ust), ustovwsp))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4446
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4447
d(1) = r6_sub(r6_add(v(k), r6_mul(r6_div(r6_mul(r6_mul(dtz(k), voce), mynn_ipow<2>(ust)), wspd), rho(k))), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4450
{ const int mynn_lo8 = (((kts) + (1))), mynn_hi8 = (((kte) - (1))), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4451
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4452
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4453
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4454
d(k) = r6_add(r6_add(r6_sub(v(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_v(k), delt)), r6_mul(det_v(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4456
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4458
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4460
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4462
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4463
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_div(r6_mul(rhosfc, mynn_ipow<2>(ust)), wspd))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4466
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4469
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(v(k), r6_div(r6_mul(r6_mul(dtz(k), voce), mynn_ipow<2>(ust)), wspd)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awv1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awv1(((k) + (1)))), onoff)), r6_mul(sub_v(k), delt)), r6_mul(det_v(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4474
{ const int mynn_lo9 = (((kts) + (1))), mynn_hi9 = (((kte) - (1))), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4475
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4478
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4481
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4484
d(k) = r6_add(r6_add(r6_sub(r6_add(v(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awv1(k), s_awv1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awv1(k), sd_awv1(((k) + (1))))), onoff)), r6_mul(sub_v(k), delt)), r6_mul(det_v(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4487
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4488
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4502
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4503
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4504
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4505
d(kte) = v(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4508
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4511
{ const int mynn_lo10 = (kts), mynn_hi10 = (kte), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4513
dv(k) = r6_div(r6_sub(x(k), v(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4514
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4519
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4521
{ const int mynn_lo11 = (((kts) + (1))), mynn_hi11 = (((kte) - (1))), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { k = mynn_lo11 + mynn_it11*mynn_st11;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4522
upcont(k) = r6_sub(s_awthl1(k), r6_mul(s_aw1(k), r6_add(r6_mul(thl(k), upwind), r6_mul(thl(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4523
dncont(k) = r6_sub(sd_awthl1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(thl(k), upwind), r6_mul(thl(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4524
} mynn_lane_sync(); k = mynn_lo11 + mynn_n11*mynn_st11; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4526
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4527
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4530
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4531
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4532
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4533
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4534
d(1) = r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(thl(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flt), rhoinv(k))), r6_mul(tcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4538
{ const int mynn_lo12 = (((kts) + (1))), mynn_hi12 = (((kte) - (1))), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { k = mynn_lo12 + mynn_it12*mynn_st12;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4539
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4540
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4541
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4542
d(k) = r6_add(r6_add(r6_add(r6_sub(thl(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4544
} mynn_lane_sync(); k = mynn_lo12 + mynn_n12*mynn_st12; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4546
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4548
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4550
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4551
b(k) = r6_sub(r6_sub(r6_add(1.f, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4554
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4557
d(k) = r6_add(r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(thl(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flt), rhoinv(k))), r6_mul(tcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awthl1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awthl1(((k) + (1))))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4563
{ const int mynn_lo13 = (((kts) + (1))), mynn_hi13 = (((kte) - (1))), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { k = mynn_lo13 + mynn_it13*mynn_st13;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4564
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4567
b(k) = r6_add(r6_add(r6_add(1.f, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4570
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4573
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(thl(k), r6_mul(tcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awthl1(k), s_awthl1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awthl1(k), sd_awthl1(((k) + (1)))))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4579
} mynn_lane_sync(); k = mynn_lo13 + mynn_n13*mynn_st13; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4580
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4595
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4596
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4597
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4598
d(kte) = thl(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4601
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4604
{ const int mynn_lo14 = (kts), mynn_hi14 = (kte), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { k = mynn_lo14 + mynn_it14*mynn_st14;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4606
thl(k) = x(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4607
} mynn_lane_sync(); k = mynn_lo14 + mynn_n14*mynn_st14; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4609
if (((bl_mynn_mixqt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4616
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4618
{ const int mynn_lo15 = (((kts) + (1))), mynn_hi15 = (((kte) - (1))), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { k = mynn_lo15 + mynn_it15*mynn_st15;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4619
upcont(k) = r6_sub(s_awqt1(k), r6_mul(s_aw1(k), r6_add(r6_mul(sqw(k), upwind), r6_mul(sqw(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4620
dncont(k) = r6_sub(sd_awqt1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqw(k), upwind), r6_mul(sqw(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4621
} mynn_lane_sync(); k = mynn_lo15 + mynn_n15*mynn_st15; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4623
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4624
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4627
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4628
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4629
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4630
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4631
d(1) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqw(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flq), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4635
{ const int mynn_lo16 = (((kts) + (1))), mynn_hi16 = (((kte) - (1))), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { k = mynn_lo16 + mynn_it16*mynn_st16;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4636
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4637
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4638
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4639
d(k) = r6_add(r6_add(r6_sub(sqw(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4641
} mynn_lane_sync(); k = mynn_lo16 + mynn_n16*mynn_st16; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4643
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4645
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4647
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4648
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4651
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4654
d(k) = r6_add(r6_sub(r6_add(r6_add(sqw(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flq), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqt1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqt1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4658
{ const int mynn_lo17 = (((kts) + (1))), mynn_hi17 = (((kte) - (1))), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { k = mynn_lo17 + mynn_it17*mynn_st17;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4659
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4662
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4665
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4668
d(k) = r6_sub(r6_add(r6_add(sqw(k), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqt1(k), s_awqt1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqt1(k), sd_awqt1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4671
} mynn_lane_sync(); k = mynn_lo17 + mynn_n17*mynn_st17; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4672
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4685
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4686
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4687
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4688
d(kte) = sqw(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4691
f_tridiag2(kte, a, b, c, d, sqw2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4697
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4698
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqw2).size(); mynn_element += lane_width) { sqw2.linear(mynn_element) = sqw.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4699
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4701
if (((bl_mynn_mixqt) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4706
if (((((bl_mynn_cloudmix) > (0))) && (flag_qc))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4708
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4710
{ const int mynn_lo18 = (((kts) + (1))), mynn_hi18 = (((kte) - (1))), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { k = mynn_lo18 + mynn_it18*mynn_st18;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4711
upcont(k) = r6_sub(s_awqc1(k), r6_mul(s_aw1(k), r6_add(r6_mul(sqc(k), upwind), r6_mul(sqc(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4712
dncont(k) = r6_sub(sd_awqc1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqc(k), upwind), r6_mul(sqc(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4713
} mynn_lane_sync(); k = mynn_lo18 + mynn_n18*mynn_st18; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4715
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4716
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4719
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4720
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4721
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4722
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4723
d(1) = r6_add(r6_sub(r6_add(r6_add(sqc(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flqc), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(det_sqc(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4727
{ const int mynn_lo19 = (((kts) + (1))), mynn_hi19 = (((kte) - (1))), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { k = mynn_lo19 + mynn_it19*mynn_st19;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4728
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4729
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4730
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4731
d(k) = r6_add(r6_sub(sqc(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(det_sqc(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4733
} mynn_lane_sync(); k = mynn_lo19 + mynn_n19*mynn_st19; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4735
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4737
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4739
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4740
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4743
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4746
d(k) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqc(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flqc), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqc1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqc1(((k) + (1))))), r6_mul(det_sqc(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4751
{ const int mynn_lo20 = (((kts) + (1))), mynn_hi20 = (((kte) - (1))), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { k = mynn_lo20 + mynn_it20*mynn_st20;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4752
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4755
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4758
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4761
d(k) = r6_add(r6_sub(r6_add(r6_add(sqc(k), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqc1(k), s_awqc1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqc1(k), sd_awqc1(((k) + (1)))))), r6_mul(det_sqc(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4765
} mynn_lane_sync(); k = mynn_lo20 + mynn_n20*mynn_st20; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4766
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4769
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4770
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4771
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4772
d(kte) = sqc(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4775
f_tridiag2(kte, a, b, c, d, sqc2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4781
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4783
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqc2).size(); mynn_element += lane_width) { sqc2.linear(mynn_element) = sqc.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4784
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4785
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4787
if (((bl_mynn_mixqt) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4793
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4795
qvflux = flqv;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4796
if (((qvflux) < (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4798
qvflux = mynn_max((float)(qvflux), (float)(r6_div(mynn_min((float)(r6_sub(r6_mul(0.9f, sqv(kts)), 1e-8f)), (float)(zero)), dtz(kts))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4799
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4801
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4803
{ const int mynn_lo21 = (((kts) + (1))), mynn_hi21 = (((kte) - (1))), mynn_st21 = (1); const int mynn_n21 = mynn_trip(mynn_lo21, mynn_hi21, mynn_st21); mynn_lane_sync(); for (int mynn_it21 = lane_sub; mynn_it21 < mynn_n21; mynn_it21 += lane_width) { k = mynn_lo21 + mynn_it21*mynn_st21;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4804
upcont(k) = r6_sub(s_awqv1(k), r6_mul(s_aw1(k), r6_add(r6_mul(sqv(k), upwind), r6_mul(sqv(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4805
dncont(k) = r6_sub(sd_awqv1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqv(k), upwind), r6_mul(sqv(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4806
} mynn_lane_sync(); k = mynn_lo21 + mynn_n21*mynn_st21; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4808
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4809
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4812
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4813
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4814
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4815
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4816
d(1) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqv(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), qvflux), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4820
{ const int mynn_lo22 = (((kts) + (1))), mynn_hi22 = (((kte) - (1))), mynn_st22 = (1); const int mynn_n22 = mynn_trip(mynn_lo22, mynn_hi22, mynn_st22); mynn_lane_sync(); for (int mynn_it22 = lane_sub; mynn_it22 < mynn_n22; mynn_it22 += lane_width) { k = mynn_lo22 + mynn_it22*mynn_st22;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4821
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4822
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4823
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4824
d(k) = r6_add(r6_add(r6_sub(sqv(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4826
} mynn_lane_sync(); k = mynn_lo22 + mynn_n22*mynn_st22; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4828
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4830
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4832
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4833
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4836
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4839
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(sqv(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), qvflux), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqv1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqv1(((k) + (1))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4844
{ const int mynn_lo23 = (((kts) + (1))), mynn_hi23 = (((kte) - (1))), mynn_st23 = (1); const int mynn_n23 = mynn_trip(mynn_lo23, mynn_hi23, mynn_st23); mynn_lane_sync(); for (int mynn_it23 = lane_sub; mynn_it23 < mynn_n23; mynn_it23 += lane_width) { k = mynn_lo23 + mynn_it23*mynn_st23;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4845
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4848
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4851
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4854
d(k) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqv(k), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqv1(k), s_awqv1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqv1(k), sd_awqv1(((k) + (1)))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4858
} mynn_lane_sync(); k = mynn_lo23 + mynn_n23*mynn_st23; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4859
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4875
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4876
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4877
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4878
d(kte) = sqv(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4881
f_tridiag2(kte, a, b, c, d, sqv2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4887
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4888
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqv2).size(); mynn_element += lane_width) { sqv2.linear(mynn_element) = sqv.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4889
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4894
if (((((bl_mynn_cloudmix) > (0))) && (flag_qi))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4896
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4898
{ const int mynn_lo24 = (((kts) + (1))), mynn_hi24 = (((kte) - (1))), mynn_st24 = (1); const int mynn_n24 = mynn_trip(mynn_lo24, mynn_hi24, mynn_st24); mynn_lane_sync(); for (int mynn_it24 = lane_sub; mynn_it24 < mynn_n24; mynn_it24 += lane_width) { k = mynn_lo24 + mynn_it24*mynn_st24;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4899
upcont(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4900
dncont(k) = r6_sub(sd_awqi1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqi(k), upwind), r6_mul(sqi(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4901
} mynn_lane_sync(); k = mynn_lo24 + mynn_n24*mynn_st24; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4903
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4904
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4907
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4908
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4909
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4910
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4911
d(1) = r6_sub(sqi(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4914
{ const int mynn_lo25 = (((kts) + (1))), mynn_hi25 = (((kte) - (1))), mynn_st25 = (1); const int mynn_n25 = mynn_trip(mynn_lo25, mynn_hi25, mynn_st25); mynn_lane_sync(); for (int mynn_it25 = lane_sub; mynn_it25 < mynn_n25; mynn_it25 += lane_width) { k = mynn_lo25 + mynn_it25*mynn_st25;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4915
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4916
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4917
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4918
d(k) = r6_sub(sqi(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4919
} mynn_lane_sync(); k = mynn_lo25 + mynn_n25*mynn_st25; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4921
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4923
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4924
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4925
b(k) = r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4928
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4931
d(k) = r6_add(sqi(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqi1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4935
{ const int mynn_lo26 = (((kts) + (1))), mynn_hi26 = (((kte) - (1))), mynn_st26 = (1); const int mynn_n26 = mynn_trip(mynn_lo26, mynn_hi26, mynn_st26); mynn_lane_sync(); for (int mynn_it26 = lane_sub; mynn_it26 < mynn_n26; mynn_it26 += lane_width) { k = mynn_lo26 + mynn_it26*mynn_st26;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4936
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4939
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4942
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4945
d(k) = r6_sub(sqi(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqi1(k), sd_awqi1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4948
} mynn_lane_sync(); k = mynn_lo26 + mynn_n26*mynn_st26; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4949
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4965
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4966
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4967
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4968
d(kte) = sqi(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4971
f_tridiag2(kte, a, b, c, d, sqi2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4977
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4978
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqi2).size(); mynn_element += lane_width) { sqi2.linear(mynn_element) = sqi.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4979
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4985
if (((((bl_mynn_cloudmix) > (0))) && (false))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4987
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4989
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4990
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4991
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4992
d(k) = sqs(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4994
{ const int mynn_lo27 = (((kts) + (1))), mynn_hi27 = (((kte) - (1))), mynn_st27 = (1); const int mynn_n27 = mynn_trip(mynn_lo27, mynn_hi27, mynn_st27); mynn_lane_sync(); for (int mynn_it27 = lane_sub; mynn_it27 < mynn_n27; mynn_it27 += lane_width) { k = mynn_lo27 + mynn_it27*mynn_st27;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4995
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4996
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4997
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4998
d(k) = sqs(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:4999
} mynn_lane_sync(); k = mynn_lo27 + mynn_n27*mynn_st27; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5002
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5003
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5004
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5005
d(kte) = sqs(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5008
f_tridiag2(kte, a, b, c, d, sqs2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5014
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5015
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqs2).size(); mynn_element += lane_width) { sqs2.linear(mynn_element) = sqs.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5016
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5021
if (((((((bl_mynn_cloudmix) > (0))) && (flag_qni))) && (((bl_mynn_mixnumcon) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5024
{ const int mynn_lo28 = (kts), mynn_hi28 = (kte), mynn_st28 = (1); const int mynn_n28 = mynn_trip(mynn_lo28, mynn_hi28, mynn_st28); mynn_lane_sync(); for (int mynn_it28 = lane_sub; mynn_it28 < mynn_n28; mynn_it28 += lane_width) { k = mynn_lo28 + mynn_it28*mynn_st28;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5025
qni2(k) = mynn_max((float)(zero), (float)(qni(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5027
if (((sqi(k)) > (1e-12f))) { qni2(k) = mynn_max((float)(qni2(k)), (float)(ni_min)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5028
} mynn_lane_sync(); k = mynn_lo28 + mynn_n28*mynn_st28; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5030
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5032
{ const int mynn_lo29 = (((kts) + (1))), mynn_hi29 = (((kte) - (1))), mynn_st29 = (1); const int mynn_n29 = mynn_trip(mynn_lo29, mynn_hi29, mynn_st29); mynn_lane_sync(); for (int mynn_it29 = lane_sub; mynn_it29 < mynn_n29; mynn_it29 += lane_width) { k = mynn_lo29 + mynn_it29*mynn_st29;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5033
upcont(k) = r6_sub(s_awqni1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qni2(k), upwind), r6_mul(qni2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5034
dncont(k) = r6_sub(sd_awqni1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qni2(k), upwind), r6_mul(qni2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5035
} mynn_lane_sync(); k = mynn_lo29 + mynn_n29*mynn_st29; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5037
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5038
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5041
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5042
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5043
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5044
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5045
d(1) = r6_sub(qni2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5048
{ const int mynn_lo30 = (((kts) + (1))), mynn_hi30 = (((kte) - (1))), mynn_st30 = (1); const int mynn_n30 = mynn_trip(mynn_lo30, mynn_hi30, mynn_st30); mynn_lane_sync(); for (int mynn_it30 = lane_sub; mynn_it30 < mynn_n30; mynn_it30 += lane_width) { k = mynn_lo30 + mynn_it30*mynn_st30;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5049
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5050
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5051
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5052
d(k) = r6_sub(qni2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5053
} mynn_lane_sync(); k = mynn_lo30 + mynn_n30*mynn_st30; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5055
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5057
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5058
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5059
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5062
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5065
d(k) = r6_add(r6_sub(qni2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqni1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqni1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5069
{ const int mynn_lo31 = (((kts) + (1))), mynn_hi31 = (((kte) - (1))), mynn_st31 = (1); const int mynn_n31 = mynn_trip(mynn_lo31, mynn_hi31, mynn_st31); mynn_lane_sync(); for (int mynn_it31 = lane_sub; mynn_it31 < mynn_n31; mynn_it31 += lane_width) { k = mynn_lo31 + mynn_it31*mynn_st31;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5070
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5073
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5076
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5079
d(k) = r6_sub(r6_add(qni2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqni1(k), s_awqni1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqni1(k), sd_awqni1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5082
} mynn_lane_sync(); k = mynn_lo31 + mynn_n31*mynn_st31; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5083
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5086
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5087
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5088
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5089
d(kte) = qni2(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5092
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5095
{ const int mynn_lo32 = (kts), mynn_hi32 = (kte), mynn_st32 = (1); const int mynn_n32 = mynn_trip(mynn_lo32, mynn_hi32, mynn_st32); mynn_lane_sync(); for (int mynn_it32 = lane_sub; mynn_it32 < mynn_n32; mynn_it32 += lane_width) { k = mynn_lo32 + mynn_it32*mynn_st32;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5097
qni2(k) = x(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5099
if (((sqi2(k)) > (1e-12f))) { qni2(k) = mynn_max((float)(qni2(k)), (float)(ni_min)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5100
} mynn_lane_sync(); k = mynn_lo32 + mynn_n32*mynn_st32; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5102
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5103
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qni2).size(); mynn_element += lane_width) { qni2.linear(mynn_element) = qni.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5104
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5110
if (((((((bl_mynn_cloudmix) > (0))) && (flag_qnc))) && (((bl_mynn_mixnumcon) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5113
{ const int mynn_lo33 = (kts), mynn_hi33 = (kte), mynn_st33 = (1); const int mynn_n33 = mynn_trip(mynn_lo33, mynn_hi33, mynn_st33); mynn_lane_sync(); for (int mynn_it33 = lane_sub; mynn_it33 < mynn_n33; mynn_it33 += lane_width) { k = mynn_lo33 + mynn_it33*mynn_st33;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5114
qnc2(k) = mynn_max((float)(zero), (float)(qnc(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5116
if (((sqc(k)) > (1e-12f))) { qnc2(k) = mynn_max((float)(qnc2(k)), (float)(nc_min)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5117
} mynn_lane_sync(); k = mynn_lo33 + mynn_n33*mynn_st33; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5119
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5121
{ const int mynn_lo34 = (((kts) + (1))), mynn_hi34 = (((kte) - (1))), mynn_st34 = (1); const int mynn_n34 = mynn_trip(mynn_lo34, mynn_hi34, mynn_st34); mynn_lane_sync(); for (int mynn_it34 = lane_sub; mynn_it34 < mynn_n34; mynn_it34 += lane_width) { k = mynn_lo34 + mynn_it34*mynn_st34;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5122
upcont(k) = r6_sub(s_awqnc1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnc2(k), upwind), r6_mul(qnc2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5123
dncont(k) = r6_sub(sd_awqnc1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qnc2(k), upwind), r6_mul(qnc2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5124
} mynn_lane_sync(); k = mynn_lo34 + mynn_n34*mynn_st34; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5126
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5127
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5130
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5131
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5132
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5133
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5134
d(1) = r6_sub(qnc2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5137
{ const int mynn_lo35 = (((kts) + (1))), mynn_hi35 = (((kte) - (1))), mynn_st35 = (1); const int mynn_n35 = mynn_trip(mynn_lo35, mynn_hi35, mynn_st35); mynn_lane_sync(); for (int mynn_it35 = lane_sub; mynn_it35 < mynn_n35; mynn_it35 += lane_width) { k = mynn_lo35 + mynn_it35*mynn_st35;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5138
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5139
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5140
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5141
d(k) = r6_sub(qnc2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5142
} mynn_lane_sync(); k = mynn_lo35 + mynn_n35*mynn_st35; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5144
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5146
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5147
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5148
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5151
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5154
d(k) = r6_add(r6_sub(qnc2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnc1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqnc1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5158
{ const int mynn_lo36 = (((kts) + (1))), mynn_hi36 = (((kte) - (1))), mynn_st36 = (1); const int mynn_n36 = mynn_trip(mynn_lo36, mynn_hi36, mynn_st36); mynn_lane_sync(); for (int mynn_it36 = lane_sub; mynn_it36 < mynn_n36; mynn_it36 += lane_width) { k = mynn_lo36 + mynn_it36*mynn_st36;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5159
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5162
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5165
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5168
d(k) = r6_sub(r6_add(qnc2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnc1(k), s_awqnc1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqnc1(k), sd_awqnc1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5171
} mynn_lane_sync(); k = mynn_lo36 + mynn_n36*mynn_st36; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5172
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5175
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5176
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5177
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5178
d(kte) = qnc2(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5181
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5184
{ const int mynn_lo37 = (kts), mynn_hi37 = (kte), mynn_st37 = (1); const int mynn_n37 = mynn_trip(mynn_lo37, mynn_hi37, mynn_st37); mynn_lane_sync(); for (int mynn_it37 = lane_sub; mynn_it37 < mynn_n37; mynn_it37 += lane_width) { k = mynn_lo37 + mynn_it37*mynn_st37;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5186
qnc2(k) = x(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5188
if (((sqc2(k)) > (1e-12f))) { qnc2(k) = mynn_max((float)(qnc2(k)), (float)(nc_min)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5189
} mynn_lane_sync(); k = mynn_lo37 + mynn_n37*mynn_st37; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5191
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5192
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnc2).size(); mynn_element += lane_width) { qnc2.linear(mynn_element) = qnc.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5193
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5198
if (((flag_qnwfa) && (((bl_mynn_mixaerosols) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5200
{ const int mynn_lo38 = (kts), mynn_hi38 = (kte), mynn_st38 = (1); const int mynn_n38 = mynn_trip(mynn_lo38, mynn_hi38, mynn_st38); mynn_lane_sync(); for (int mynn_it38 = lane_sub; mynn_it38 < mynn_n38; mynn_it38 += lane_width) { k = mynn_lo38 + mynn_it38*mynn_st38;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5201
qnwfa2(k) = mynn_max((float)(zero), (float)(qnwfa(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5202
} mynn_lane_sync(); k = mynn_lo38 + mynn_n38*mynn_st38; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5204
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5206
{ const int mynn_lo39 = (((kts) + (1))), mynn_hi39 = (((kte) - (1))), mynn_st39 = (1); const int mynn_n39 = mynn_trip(mynn_lo39, mynn_hi39, mynn_st39); mynn_lane_sync(); for (int mynn_it39 = lane_sub; mynn_it39 < mynn_n39; mynn_it39 += lane_width) { k = mynn_lo39 + mynn_it39*mynn_st39;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5207
upcont(k) = r6_sub(s_awqnwfa1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnwfa2(k), upwind), r6_mul(qnwfa2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5208
dncont(k) = r6_sub(sd_awqnwfa1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qnwfa2(k), upwind), r6_mul(qnwfa2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5209
} mynn_lane_sync(); k = mynn_lo39 + mynn_n39*mynn_st39; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5211
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5212
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5215
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5216
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5217
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5218
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5219
d(1) = r6_sub(qnwfa2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5222
{ const int mynn_lo40 = (((kts) + (1))), mynn_hi40 = (((kte) - (1))), mynn_st40 = (1); const int mynn_n40 = mynn_trip(mynn_lo40, mynn_hi40, mynn_st40); mynn_lane_sync(); for (int mynn_it40 = lane_sub; mynn_it40 < mynn_n40; mynn_it40 += lane_width) { k = mynn_lo40 + mynn_it40*mynn_st40;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5223
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5224
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5225
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5226
d(k) = r6_sub(qnwfa2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5227
} mynn_lane_sync(); k = mynn_lo40 + mynn_n40*mynn_st40; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5229
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5231
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5232
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5233
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5236
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5239
d(k) = r6_add(r6_sub(qnwfa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnwfa1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqnwfa1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5243
{ const int mynn_lo41 = (((kts) + (1))), mynn_hi41 = (((kte) - (1))), mynn_st41 = (1); const int mynn_n41 = mynn_trip(mynn_lo41, mynn_hi41, mynn_st41); mynn_lane_sync(); for (int mynn_it41 = lane_sub; mynn_it41 < mynn_n41; mynn_it41 += lane_width) { k = mynn_lo41 + mynn_it41*mynn_st41;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5244
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5247
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5250
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5253
d(k) = r6_sub(r6_add(qnwfa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnwfa1(k), s_awqnwfa1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqnwfa1(k), sd_awqnwfa1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5256
} mynn_lane_sync(); k = mynn_lo41 + mynn_n41*mynn_st41; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5257
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5260
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5261
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5262
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5263
d(kte) = qnwfa2(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5266
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5269
{ const int mynn_lo42 = (kts), mynn_hi42 = (kte), mynn_st42 = (1); const int mynn_n42 = mynn_trip(mynn_lo42, mynn_hi42, mynn_st42); mynn_lane_sync(); for (int mynn_it42 = lane_sub; mynn_it42 < mynn_n42; mynn_it42 += lane_width) { k = mynn_lo42 + mynn_it42*mynn_st42;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5271
qnwfa2(k) = mynn_max((float)(zero), (float)(x(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5272
} mynn_lane_sync(); k = mynn_lo42 + mynn_n42*mynn_st42; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5275
{ const int mynn_lo43 = (kts), mynn_hi43 = (kte), mynn_st43 = (1); const int mynn_n43 = mynn_trip(mynn_lo43, mynn_hi43, mynn_st43); mynn_lane_sync(); for (int mynn_it43 = lane_sub; mynn_it43 < mynn_n43; mynn_it43 += lane_width) { k = mynn_lo43 + mynn_it43*mynn_st43;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5277
if (((r6_sub(xland, 1.5f)) >= (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5278
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zw(k), 100.f)), (float)(zero)), mynn_max((float)(pblh), (float)(200.f)))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5279
wfa_max2 = r6_add(r6_mul(r6_sub(one, wt), 6.e8f), r6_mul(wt, wfa_max));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5280
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5281
wfa_max2 = wfa_max;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5282
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5283
aero_min = r6_mul(wfa_min, mynn_exp(r6_neg(r6_div(zw(k), wfa_ht))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5284
aero_max = r6_mul(wfa_max2, mynn_exp(r6_neg(r6_div(zw(k), wfa_ht))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5285
qnwfa2(k) = mynn_min((float)(mynn_max((float)(aero_min), (float)(qnwfa2(k)))), (float)(aero_max));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5286
} mynn_lane_sync(); k = mynn_lo43 + mynn_n43*mynn_st43; if (mynn_n43 > 0) { const int mynn_src43 = (mynn_n43 - 1) % lane_width; aero_max = mynn_lane_bcast(aero_max, mynn_src43); aero_min = mynn_lane_bcast(aero_min, mynn_src43); wfa_max2 = mynn_lane_bcast(wfa_max2, mynn_src43); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5287
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5289
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnwfa2).size(); mynn_element += lane_width) { qnwfa2.linear(mynn_element) = qnwfa.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5290
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5295
if (((flag_qnifa) && (((bl_mynn_mixaerosols) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5297
{ const int mynn_lo44 = (kts), mynn_hi44 = (kte), mynn_st44 = (1); const int mynn_n44 = mynn_trip(mynn_lo44, mynn_hi44, mynn_st44); mynn_lane_sync(); for (int mynn_it44 = lane_sub; mynn_it44 < mynn_n44; mynn_it44 += lane_width) { k = mynn_lo44 + mynn_it44*mynn_st44;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5298
qnifa2(k) = mynn_max((float)(zero), (float)(qnifa(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5299
} mynn_lane_sync(); k = mynn_lo44 + mynn_n44*mynn_st44; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5301
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5303
{ const int mynn_lo45 = (((kts) + (1))), mynn_hi45 = (((kte) - (1))), mynn_st45 = (1); const int mynn_n45 = mynn_trip(mynn_lo45, mynn_hi45, mynn_st45); mynn_lane_sync(); for (int mynn_it45 = lane_sub; mynn_it45 < mynn_n45; mynn_it45 += lane_width) { k = mynn_lo45 + mynn_it45*mynn_st45;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5304
upcont(k) = r6_sub(s_awqnifa1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnifa2(k), upwind), r6_mul(qnifa2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5305
dncont(k) = r6_sub(sd_awqnifa1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qnifa2(k), upwind), r6_mul(qnifa2(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5306
} mynn_lane_sync(); k = mynn_lo45 + mynn_n45*mynn_st45; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5308
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5309
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5311
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5312
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5313
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5314
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5315
d(1) = r6_sub(qnifa2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5318
{ const int mynn_lo46 = (((kts) + (1))), mynn_hi46 = (((kte) - (1))), mynn_st46 = (1); const int mynn_n46 = mynn_trip(mynn_lo46, mynn_hi46, mynn_st46); mynn_lane_sync(); for (int mynn_it46 = lane_sub; mynn_it46 < mynn_n46; mynn_it46 += lane_width) { k = mynn_lo46 + mynn_it46*mynn_st46;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5319
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5320
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5321
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5322
d(k) = r6_sub(qnifa2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5323
} mynn_lane_sync(); k = mynn_lo46 + mynn_n46*mynn_st46; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5325
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5327
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5328
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5329
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5332
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5335
d(k) = r6_add(r6_sub(qnifa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnifa1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqnifa1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5339
{ const int mynn_lo47 = (((kts) + (1))), mynn_hi47 = (((kte) - (1))), mynn_st47 = (1); const int mynn_n47 = mynn_trip(mynn_lo47, mynn_hi47, mynn_st47); mynn_lane_sync(); for (int mynn_it47 = lane_sub; mynn_it47 < mynn_n47; mynn_it47 += lane_width) { k = mynn_lo47 + mynn_it47*mynn_st47;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5340
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5343
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5346
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5349
d(k) = r6_sub(r6_add(qnifa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnifa1(k), s_awqnifa1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqnifa1(k), sd_awqnifa1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5352
} mynn_lane_sync(); k = mynn_lo47 + mynn_n47*mynn_st47; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5353
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5356
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5357
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5358
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5359
d(kte) = qnifa2(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5362
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5365
{ const int mynn_lo48 = (kts), mynn_hi48 = (kte), mynn_st48 = (1); const int mynn_n48 = mynn_trip(mynn_lo48, mynn_hi48, mynn_st48); mynn_lane_sync(); for (int mynn_it48 = lane_sub; mynn_it48 < mynn_n48; mynn_it48 += lane_width) { k = mynn_lo48 + mynn_it48*mynn_st48;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5367
qnifa2(k) = mynn_max((float)(zero), (float)(x(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5368
} mynn_lane_sync(); k = mynn_lo48 + mynn_n48*mynn_st48; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5371
{ const int mynn_lo49 = (kts), mynn_hi49 = (kte), mynn_st49 = (1); const int mynn_n49 = mynn_trip(mynn_lo49, mynn_hi49, mynn_st49); mynn_lane_sync(); for (int mynn_it49 = lane_sub; mynn_it49 < mynn_n49; mynn_it49 += lane_width) { k = mynn_lo49 + mynn_it49*mynn_st49;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5372
aero_min = r6_mul(ifa_min, mynn_exp(r6_neg(r6_div(zw(k), ifa_ht))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5373
aero_max = r6_mul(ifa_max, mynn_exp(r6_neg(r6_div(zw(k), ifa_ht))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5374
qnifa2(k) = mynn_min((float)(mynn_max((float)(aero_min), (float)(qnifa2(k)))), (float)(aero_max));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5375
} mynn_lane_sync(); k = mynn_lo49 + mynn_n49*mynn_st49; if (mynn_n49 > 0) { const int mynn_src49 = (mynn_n49 - 1) % lane_width; aero_max = mynn_lane_bcast(aero_max, mynn_src49); aero_min = mynn_lane_bcast(aero_min, mynn_src49); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5377
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5379
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnifa2).size(); mynn_element += lane_width) { qnifa2.linear(mynn_element) = qnifa.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5380
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5385
if (((flag_qnbca) && (((bl_mynn_mixaerosols) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5387
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5389
{ const int mynn_lo50 = (((kts) + (1))), mynn_hi50 = (((kte) - (1))), mynn_st50 = (1); const int mynn_n50 = mynn_trip(mynn_lo50, mynn_hi50, mynn_st50); mynn_lane_sync(); for (int mynn_it50 = lane_sub; mynn_it50 < mynn_n50; mynn_it50 += lane_width) { k = mynn_lo50 + mynn_it50*mynn_st50;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5390
upcont(k) = r6_sub(s_awqnbca1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnbca(k), upwind), r6_mul(qnbca(((k) - (1))), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5391
dncont(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5392
} mynn_lane_sync(); k = mynn_lo50 + mynn_n50*mynn_st50; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5394
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5395
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5397
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5398
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5399
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5400
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5401
d(1) = r6_sub(qnbca(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5404
{ const int mynn_lo51 = (((kts) + (1))), mynn_hi51 = (((kte) - (1))), mynn_st51 = (1); const int mynn_n51 = mynn_trip(mynn_lo51, mynn_hi51, mynn_st51); mynn_lane_sync(); for (int mynn_it51 = lane_sub; mynn_it51 < mynn_n51; mynn_it51 += lane_width) { k = mynn_lo51 + mynn_it51*mynn_st51;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5405
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5406
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5407
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5408
d(k) = r6_sub(qnbca(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5409
} mynn_lane_sync(); k = mynn_lo51 + mynn_n51*mynn_st51; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5411
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5413
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5414
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5415
b(k) = r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5417
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5419
d(k) = r6_sub(qnbca(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnbca1(((k) + (1)))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5421
{ const int mynn_lo52 = (((kts) + (1))), mynn_hi52 = (((kte) - (1))), mynn_st52 = (1); const int mynn_n52 = mynn_trip(mynn_lo52, mynn_hi52, mynn_st52); mynn_lane_sync(); for (int mynn_it52 = lane_sub; mynn_it52 < mynn_n52; mynn_it52 += lane_width) { k = mynn_lo52 + mynn_it52*mynn_st52;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5422
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5424
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5426
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5428
d(k) = r6_add(qnbca(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnbca1(k), s_awqnbca1(((k) + (1))))), nonloc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5429
} mynn_lane_sync(); k = mynn_lo52 + mynn_n52*mynn_st52; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5430
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5433
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5434
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5435
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5436
d(kte) = qnbca(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5439
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5442
{ const int mynn_lo53 = (kts), mynn_hi53 = (kte), mynn_st53 = (1); const int mynn_n53 = mynn_trip(mynn_lo53, mynn_hi53, mynn_st53); mynn_lane_sync(); for (int mynn_it53 = lane_sub; mynn_it53 < mynn_n53; mynn_it53 += lane_width) { k = mynn_lo53 + mynn_it53*mynn_st53;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5444
qnbca2(k) = mynn_max((float)(zero), (float)(x(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5445
} mynn_lane_sync(); k = mynn_lo53 + mynn_n53*mynn_st53; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5447
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5449
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnbca2).size(); mynn_element += lane_width) { qnbca2.linear(mynn_element) = qnbca.linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5450
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5455
if (flag_ozone) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5456
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5459
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5460
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5461
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5462
d(k) = ozone(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5464
{ const int mynn_lo54 = (((kts) + (1))), mynn_hi54 = (((kte) - (1))), mynn_st54 = (1); const int mynn_n54 = mynn_trip(mynn_lo54, mynn_hi54, mynn_st54); mynn_lane_sync(); for (int mynn_it54 = lane_sub; mynn_it54 < mynn_n54; mynn_it54 += lane_width) { k = mynn_lo54 + mynn_it54*mynn_st54;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5465
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5466
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5467
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5468
d(k) = ozone(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5469
} mynn_lane_sync(); k = mynn_lo54 + mynn_n54*mynn_st54; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5472
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5473
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5474
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5475
d(kte) = ozone(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5478
f_tridiag2(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5481
{ const int mynn_lo55 = (kts), mynn_hi55 = (kte), mynn_st55 = (1); const int mynn_n55 = mynn_trip(mynn_lo55, mynn_hi55, mynn_st55); mynn_lane_sync(); for (int mynn_it55 = lane_sub; mynn_it55 < mynn_n55; mynn_it55 += lane_width) { k = mynn_lo55 + mynn_it55*mynn_st55;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5483
dozone(k) = r6_div(r6_sub(mynn_max((float)(zero), (float)(x(k))), ozone(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5484
} mynn_lane_sync(); k = mynn_lo55 + mynn_n55*mynn_st55; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5485
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5486
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dozone.slice(MynnRange{dozone.lo[0], dozone.hi[0]})).size(); mynn_element += lane_width) { dozone.slice(MynnRange{dozone.lo[0], dozone.hi[0]}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5487
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5494
if (((bl_mynn_mixqt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5495
{ const int mynn_lo56 = (kts), mynn_hi56 = (kte), mynn_st56 = (1); const int mynn_n56 = mynn_trip(mynn_lo56, mynn_hi56, mynn_st56); mynn_lane_sync(); for (int mynn_it56 = lane_sub; mynn_it56 < mynn_n56; mynn_it56 += lane_width) { k = mynn_lo56 + mynn_it56*mynn_st56;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5497
th_new = r6_add(r6_add(thl(k), r6_mul(r6_div(xlvcp, exner(k)), sqc(k))), r6_mul(r6_div(xlscp, exner(k)), sqi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5500
t = r6_mul(th_new, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5501
qsat = f_qsat_blend(t, p(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5508
if (((((sqc(k)) > (zero))) || (((sqi(k)) > (zero))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5509
sqv2(k) = mynn_min((float)(sqw2(k)), (float)(qsat));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5510
portion_qc = r6_div(sqc(k), r6_add(sqc(k), sqi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5511
portion_qi = r6_div(sqi(k), r6_add(sqc(k), sqi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5512
condensate = mynn_max((float)(zero), (float)(r6_sub(sqw2(k), qsat)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5513
sqc2(k) = r6_mul(condensate, portion_qc);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5514
sqi2(k) = r6_mul(condensate, portion_qi);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5515
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5516
sqv2(k) = sqw2(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5517
sqi2(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5518
sqc2(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5519
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5520
} mynn_lane_sync(); k = mynn_lo56 + mynn_n56*mynn_st56; if (mynn_n56 > 0) { const int mynn_src56 = (mynn_n56 - 1) % lane_width; qsat = mynn_lane_bcast(qsat, mynn_src56); t = mynn_lane_bcast(t, mynn_src56); th_new = mynn_lane_bcast(th_new, mynn_src56); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5521
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5527
{ const int mynn_lo57 = (kts), mynn_hi57 = (kte), mynn_st57 = (1); const int mynn_n57 = mynn_trip(mynn_lo57, mynn_hi57, mynn_st57); mynn_lane_sync(); for (int mynn_it57 = lane_sub; mynn_it57 < mynn_n57; mynn_it57 += lane_width) { k = mynn_lo57 + mynn_it57*mynn_st57;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5528
dqv(k) = r6_div(r6_sub(sqv2(k), sqv(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5530
} mynn_lane_sync(); k = mynn_lo57 + mynn_n57*mynn_st57; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5532
if (((bl_mynn_cloudmix) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5537
if (flag_qc) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5538
{ const int mynn_lo58 = (kts), mynn_hi58 = (kte), mynn_st58 = (1); const int mynn_n58 = mynn_trip(mynn_lo58, mynn_hi58, mynn_st58); mynn_lane_sync(); for (int mynn_it58 = lane_sub; mynn_it58 < mynn_n58; mynn_it58 += lane_width) { k = mynn_lo58 + mynn_it58*mynn_st58;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5539
dqc(k) = r6_div(r6_sub(sqc2(k), sqc(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5541
} mynn_lane_sync(); k = mynn_lo58 + mynn_n58*mynn_st58; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5542
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5543
{ const int mynn_lo59 = (kts), mynn_hi59 = (kte), mynn_st59 = (1); const int mynn_n59 = mynn_trip(mynn_lo59, mynn_hi59, mynn_st59); mynn_lane_sync(); for (int mynn_it59 = lane_sub; mynn_it59 < mynn_n59; mynn_it59 += lane_width) { k = mynn_lo59 + mynn_it59*mynn_st59;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5544
dqc(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5545
} mynn_lane_sync(); k = mynn_lo59 + mynn_n59*mynn_st59; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5546
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5551
if (((flag_qnc) && (((bl_mynn_mixnumcon) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5552
{ const int mynn_lo60 = (kts), mynn_hi60 = (kte), mynn_st60 = (1); const int mynn_n60 = mynn_trip(mynn_lo60, mynn_hi60, mynn_st60); mynn_lane_sync(); for (int mynn_it60 = lane_sub; mynn_it60 < mynn_n60; mynn_it60 += lane_width) { k = mynn_lo60 + mynn_it60*mynn_st60;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5553
dqnc(k) = r6_div(r6_sub(qnc2(k), qnc(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5555
} mynn_lane_sync(); k = mynn_lo60 + mynn_n60*mynn_st60; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5556
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5557
{ const int mynn_lo61 = (kts), mynn_hi61 = (kte), mynn_st61 = (1); const int mynn_n61 = mynn_trip(mynn_lo61, mynn_hi61, mynn_st61); mynn_lane_sync(); for (int mynn_it61 = lane_sub; mynn_it61 < mynn_n61; mynn_it61 += lane_width) { k = mynn_lo61 + mynn_it61*mynn_st61;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5558
dqnc(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5559
} mynn_lane_sync(); k = mynn_lo61 + mynn_n61*mynn_st61; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5560
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5565
if (flag_qi) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5566
{ const int mynn_lo62 = (kts), mynn_hi62 = (kte), mynn_st62 = (1); const int mynn_n62 = mynn_trip(mynn_lo62, mynn_hi62, mynn_st62); mynn_lane_sync(); for (int mynn_it62 = lane_sub; mynn_it62 < mynn_n62; mynn_it62 += lane_width) { k = mynn_lo62 + mynn_it62*mynn_st62;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5567
dqi(k) = r6_div(r6_sub(sqi2(k), sqi(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5569
} mynn_lane_sync(); k = mynn_lo62 + mynn_n62*mynn_st62; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5570
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5571
{ const int mynn_lo63 = (kts), mynn_hi63 = (kte), mynn_st63 = (1); const int mynn_n63 = mynn_trip(mynn_lo63, mynn_hi63, mynn_st63); mynn_lane_sync(); for (int mynn_it63 = lane_sub; mynn_it63 < mynn_n63; mynn_it63 += lane_width) { k = mynn_lo63 + mynn_it63*mynn_st63;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5572
dqi(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5573
} mynn_lane_sync(); k = mynn_lo63 + mynn_n63*mynn_st63; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5574
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5579
if (false) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5580
{ const int mynn_lo64 = (kts), mynn_hi64 = (kte), mynn_st64 = (1); const int mynn_n64 = mynn_trip(mynn_lo64, mynn_hi64, mynn_st64); mynn_lane_sync(); for (int mynn_it64 = lane_sub; mynn_it64 < mynn_n64; mynn_it64 += lane_width) { k = mynn_lo64 + mynn_it64*mynn_st64;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5581
dqs(k) = r6_div(r6_sub(sqs2(k), sqs(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5582
} mynn_lane_sync(); k = mynn_lo64 + mynn_n64*mynn_st64; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5583
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5584
{ const int mynn_lo65 = (kts), mynn_hi65 = (kte), mynn_st65 = (1); const int mynn_n65 = mynn_trip(mynn_lo65, mynn_hi65, mynn_st65); mynn_lane_sync(); for (int mynn_it65 = lane_sub; mynn_it65 < mynn_n65; mynn_it65 += lane_width) { k = mynn_lo65 + mynn_it65*mynn_st65;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5585
dqs(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5586
} mynn_lane_sync(); k = mynn_lo65 + mynn_n65*mynn_st65; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5587
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5592
if (((flag_qni) && (((bl_mynn_mixnumcon) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5593
{ const int mynn_lo66 = (kts), mynn_hi66 = (kte), mynn_st66 = (1); const int mynn_n66 = mynn_trip(mynn_lo66, mynn_hi66, mynn_st66); mynn_lane_sync(); for (int mynn_it66 = lane_sub; mynn_it66 < mynn_n66; mynn_it66 += lane_width) { k = mynn_lo66 + mynn_it66*mynn_st66;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5594
dqni(k) = r6_div(r6_sub(qni2(k), qni(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5596
} mynn_lane_sync(); k = mynn_lo66 + mynn_n66*mynn_st66; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5597
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5598
{ const int mynn_lo67 = (kts), mynn_hi67 = (kte), mynn_st67 = (1); const int mynn_n67 = mynn_trip(mynn_lo67, mynn_hi67, mynn_st67); mynn_lane_sync(); for (int mynn_it67 = lane_sub; mynn_it67 < mynn_n67; mynn_it67 += lane_width) { k = mynn_lo67 + mynn_it67*mynn_st67;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5599
dqni(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5600
} mynn_lane_sync(); k = mynn_lo67 + mynn_n67*mynn_st67; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5601
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5602
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5604
{ const int mynn_lo68 = (kts), mynn_hi68 = (kte), mynn_st68 = (1); const int mynn_n68 = mynn_trip(mynn_lo68, mynn_hi68, mynn_st68); mynn_lane_sync(); for (int mynn_it68 = lane_sub; mynn_it68 < mynn_n68; mynn_it68 += lane_width) { k = mynn_lo68 + mynn_it68*mynn_st68;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5605
dqc(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5606
dqnc(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5607
dqi(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5608
dqni(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5609
dqs(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5610
} mynn_lane_sync(); k = mynn_lo68 + mynn_n68*mynn_st68; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5611
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5614
f_moisture_check(kte, delt, delp, exner, sqv2, sqc2, sqi2, sqs2, thl, dqv, dqc, dqi, dqs, dth);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5621
{ const int mynn_lo69 = (kts), mynn_hi69 = (kte), mynn_st69 = (1); const int mynn_n69 = mynn_trip(mynn_lo69, mynn_hi69, mynn_st69); mynn_lane_sync(); for (int mynn_it69 = lane_sub; mynn_it69 < mynn_n69; mynn_it69 += lane_width) { k = mynn_lo69 + mynn_it69*mynn_st69;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5622
if (((r6_add(r6_mul(dozone(k), delt), ozone(k))) < (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5623
dozone(k) = r6_div(r6_neg(r6_mul(ozone(k), 0.99f)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5624
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5625
} mynn_lane_sync(); k = mynn_lo69 + mynn_n69*mynn_st69; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5630
if (flag_qi) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5631
{ const int mynn_lo70 = (kts), mynn_hi70 = (kte), mynn_st70 = (1); const int mynn_n70 = mynn_trip(mynn_lo70, mynn_hi70, mynn_st70); mynn_lane_sync(); for (int mynn_it70 = lane_sub; mynn_it70 < mynn_n70; mynn_it70 += lane_width) { k = mynn_lo70 + mynn_it70*mynn_st70;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5632
dth(k) = r6_div(r6_sub(r6_add(r6_add(thl(k), r6_mul(r6_div(xlvcp, exner(k)), sqc(k))), r6_mul(r6_div(xlscp, exner(k)), sqi(k))), th(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5643
} mynn_lane_sync(); k = mynn_lo70 + mynn_n70*mynn_st70; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5644
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5645
{ const int mynn_lo71 = (kts), mynn_hi71 = (kte), mynn_st71 = (1); const int mynn_n71 = mynn_trip(mynn_lo71, mynn_hi71, mynn_st71); mynn_lane_sync(); for (int mynn_it71 = lane_sub; mynn_it71 < mynn_n71; mynn_it71 += lane_width) { k = mynn_lo71 + mynn_it71*mynn_st71;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5646
dth(k) = r6_div(r6_sub(r6_add(thl(k), r6_mul(r6_div(xlvcp, exner(k)), sqc2(k))), th(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5651
} mynn_lane_sync(); k = mynn_lo71 + mynn_n71*mynn_st71; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5652
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5657
if (((((flag_qnwfa) && (flag_qnifa))) && (((bl_mynn_mixaerosols) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5659
{ const int mynn_lo72 = (kts), mynn_hi72 = (kte), mynn_st72 = (1); const int mynn_n72 = mynn_trip(mynn_lo72, mynn_hi72, mynn_st72); mynn_lane_sync(); for (int mynn_it72 = lane_sub; mynn_it72 < mynn_n72; mynn_it72 += lane_width) { k = mynn_lo72 + mynn_it72*mynn_st72;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5663
dqnwfa(k) = r6_div(r6_sub(qnwfa2(k), qnwfa(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5667
dqnifa(k) = r6_div(r6_sub(qnifa2(k), qnifa(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5668
} mynn_lane_sync(); k = mynn_lo72 + mynn_n72*mynn_st72; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5669
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5670
{ const int mynn_lo73 = (kts), mynn_hi73 = (kte), mynn_st73 = (1); const int mynn_n73 = mynn_trip(mynn_lo73, mynn_hi73, mynn_st73); mynn_lane_sync(); for (int mynn_it73 = lane_sub; mynn_it73 < mynn_n73; mynn_it73 += lane_width) { k = mynn_lo73 + mynn_it73*mynn_st73;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5671
dqnwfa(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5672
dqnifa(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5673
} mynn_lane_sync(); k = mynn_lo73 + mynn_n73*mynn_st73; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5674
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5679
if (((flag_qnbca) && (((bl_mynn_mixaerosols) > (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5680
{ const int mynn_lo74 = (kts), mynn_hi74 = (kte), mynn_st74 = (1); const int mynn_n74 = mynn_trip(mynn_lo74, mynn_hi74, mynn_st74); mynn_lane_sync(); for (int mynn_it74 = lane_sub; mynn_it74 < mynn_n74; mynn_it74 += lane_width) { k = mynn_lo74 + mynn_it74*mynn_st74;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5681
dqnbca(k) = r6_div(r6_sub(qnbca2(k), qnbca(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5682
} mynn_lane_sync(); k = mynn_lo74 + mynn_n74*mynn_st74; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5683
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5684
{ const int mynn_lo75 = (kts), mynn_hi75 = (kte), mynn_st75 = (1); const int mynn_n75 = mynn_trip(mynn_lo75, mynn_hi75, mynn_st75); mynn_lane_sync(); for (int mynn_it75 = lane_sub; mynn_it75 < mynn_n75; mynn_it75 += lane_width) { k = mynn_lo75 + mynn_it75*mynn_st75;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5685
dqnbca(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5686
} mynn_lane_sync(); k = mynn_lo75 + mynn_n75*mynn_st75; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5687
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5696
if (debug_code) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5697
problem = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5698
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5699
wsp = mynn_sqrt(r6_add(mynn_ipow<2>(u(k)), mynn_ipow<2>(v(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5700
wsp2 = mynn_sqrt(r6_add(mynn_ipow<2>(r6_add(u(k), r6_mul(du(k), delt))), mynn_ipow<2>(r6_add(v(k), r6_mul(du(k), delt)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5701
th2 = r6_add(th(k), r6_mul(dth(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5702
tk2 = r6_mul(th2, exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5703
if (((((((wsp2) > (200.f))) || (((tk2) > (360.f))))) || (((tk2) < (160.f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5704
problem = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5705
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5706
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5707
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5708
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5709
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5710
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5711
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5712
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5713
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5714
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5715
kproblem = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5716
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5717
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5718
if (problem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5719
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5720
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5721
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5722
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5723
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5724
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5725
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5726
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5736-5824
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5765
for (k=kte; ((-(1))) > 0 ? k <= (1) : k >= (1); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5766
dqc2 = mynn_max((float)(r6_sub(qcmin, qc(k))), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5767
dqi2 = mynn_max((float)(r6_sub(qimin, qi(k))), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5768
dqs2 = mynn_max((float)(r6_sub(qimin, qs(k))), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5771
dqc(k) = r6_add(dqc(k), r6_div(dqc2, delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5772
dqi(k) = r6_add(dqi(k), r6_div(dqi2, delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5773
dqs(k) = r6_add(dqs(k), r6_div(dqs2, delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5774
dqv(k) = r6_sub(dqv(k), r6_div(r6_add(r6_add(dqc2, dqi2), dqs2), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5775
dth(k) = r6_add(r6_add(dth(k), r6_mul(r6_div(xlvcp, exner(k)), r6_div(dqc2, delt))), r6_mul(r6_div(xlscp, exner(k)), r6_div(r6_add(dqi2, dqs2), delt)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5778
qc(k) = r6_add(qc(k), dqc2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5779
qi(k) = r6_add(qi(k), dqi2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5780
qs(k) = r6_add(qs(k), dqs2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5781
qv(k) = r6_sub(r6_sub(r6_sub(qv(k), dqc2), dqi2), dqs2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5782
th(k) = r6_add(r6_add(th(k), r6_mul(r6_div(xlvcp, exner(k)), dqc2)), r6_mul(r6_div(xlscp, exner(k)), r6_add(dqi2, dqs2)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5786
dqv2 = mynn_max((float)(r6_sub(qvmin, qv(k))), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5787
dqv(k) = r6_add(dqv(k), r6_div(dqv2, delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5788
qv(k) = r6_add(qv(k), dqv2);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5789
if (((k) != (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5791
qv(((k) - (1))) = r6_sub(qv(((k) - (1))), r6_div(r6_mul(dqv2, dp(k)), dp(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5792
dqv(((k) - (1))) = r6_sub(dqv(((k) - (1))), r6_div(r6_div(r6_mul(dqv2, dp(k)), dp(((k) - (1)))), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5793
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5794
qv(k) = mynn_max((float)(qv(k)), (float)(qvmin));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5795
qc(k) = mynn_max((float)(qcmin), (float)(qc(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5796
qi(k) = mynn_max((float)(qimin), (float)(qi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5797
qs(k) = mynn_max((float)(qimin), (float)(qs(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5798
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5802
if (((dqv2) > (1.e-20f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5803
sum = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5804
for (k=1; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5805
if (((qv(k)) > (r6_mul(2.0f, qvmin)))) { sum = r6_add(sum, r6_mul(qv(k), dp(k))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5806
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5807
aa = r6_div(r6_mul(dqv2, dp(1)), mynn_max((float)(sum), (float)(1.e-20f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5808
if (((aa) < (p5))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5809
{ const int mynn_lo0 = (1), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5810
if (((qv(k)) > (r6_mul(2.0f, qvmin)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5811
dum = r6_mul(aa, qv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5812
qv(k) = r6_sub(qv(k), dum);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5813
dqv(k) = r6_sub(dqv(k), r6_div(dum, delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5814
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5815
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5816
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5819
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5820
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5822
return;
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5828-5985
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
constexpr float no_threshold = 0.1f;
constexpr float frp_threshold = 10.0f;
constexpr float pblh_threshold = 100.0f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5872
dztop = r6_mul(.5f, r6_add(dz(kte), dz(((kte) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5874
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5875
dtz(k) = r6_div(delt, dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5876
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5880
rhoz(kts) = rho(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5881
rhoinv(kts) = r6_div(one, rho(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5882
khdz(kts) = r6_mul(rhoz(kts), dfh(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5884
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5885
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5886
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5887
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5888
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5889
khdz(k) = r6_mul(rhoz(k), dfh(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5890
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5891
rhoz(((kte) + (1))) = rhoz(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5892
khdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfh(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5894
if (((bl_mynn_edmf) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5896
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5897
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5898
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5899
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5900
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5903
if (enh_mix) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5904
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5905
khdz_old = khdz(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5906
khdz_back = r6_div(r6_mul(pblh, 0.15f), dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5908
if (((pblh) < (pblh_threshold))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5909
if (((emis_ant_no) > (no_threshold))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5910
khdz(k) = mynn_max((float)(r6_mul(1.1f, khdz(k))), (float)(r6_mul(r6_div(mynn_sqrt(r6_div(emis_ant_no, no_threshold)), dz(k)), rhoz(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5912
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5913
if (((frp) > (frp_threshold))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5914
kmaxfire = mynn_ceiling(mynn_log(frp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5915
khdz(k) = mynn_max((float)(r6_mul(1.1f, khdz(k))), (float)(r6_mul(r6_div(r6_mul(r6_sub(one, r6_div(k, r6_mul(kmaxfire, two))), r6_sub(mynn_ipow<2>(mynn_log(frp)), r6_mul(two, mynn_log(frp)))), dz(k)), rhoz(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5917
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5918
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5919
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; khdz_back = mynn_lane_bcast(khdz_back, mynn_src3); khdz_old = mynn_lane_bcast(khdz_old, mynn_src3); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5920
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5926
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5928
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5930
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5931
upcont(k) = r6_sub(s_awchem1(k, ic), r6_mul(s_aw1(k), r6_add(r6_mul(chem1(k, ic), upwind), r6_mul(chem1(((k) - (1)), ic), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5932
dncont(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5933
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5935
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5936
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5938
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5939
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5940
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5941
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5942
d(1) = r6_sub(chem1(k, ic), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5945
{ const int mynn_lo5 = (((kts) + (1))), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5946
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5947
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5948
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5949
d(k) = r6_sub(chem1(k, ic), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5950
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5952
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5954
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5955
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5956
b(k) = r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5957
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5958
d(k) = r6_sub(r6_sub(chem1(k, ic), r6_mul(r6_mul(dtz(k), vd1(ic)), chem1(k, ic))), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awchem1(((k) + (1)), ic)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5962
{ const int mynn_lo6 = (((kts) + (1))), mynn_hi6 = (((kte) - (1))), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5963
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5964
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5966
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5967
d(k) = r6_add(chem1(k, ic), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awchem1(k, ic), s_awchem1(((k) + (1)), ic))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5968
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5969
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5972
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5973
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5974
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5975
d(kte) = chem1(kte, ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5977
f_tridiag3(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5979
{ const int mynn_lo7 = (kts), mynn_hi7 = (kte), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5980
chem1(k, ic) = r6_add(x(k), settle1(k, ic));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5981
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5983
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:5989-6109
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6019
dztop = r6_mul(p5, r6_add(dz(kte), dz(((kte) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6021
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6022
dtz(k) = r6_div(delt, dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6023
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6027
rhoz(kts) = rho(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6028
rhoinv(kts) = r6_div(one, rho(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6029
khdz(kts) = r6_mul(rhoz(kts), dfh(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6031
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6032
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6033
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6034
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6035
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6036
khdz(k) = r6_mul(rhoz(k), dfh(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6037
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6038
rhoz(((kte) + (1))) = rhoz(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6039
khdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfh(kte));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6041
if (((bl_mynn_edmf) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6043
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6044
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6045
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6046
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6047
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6053
for (ns=1; (1) > 0 ? ns <= (nscalars) : ns >= (nscalars); ns += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6055
if (((bl_mynn_edmf) > (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6057
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6058
upcont(k) = r6_sub(s_awscalars1(k, ns), r6_mul(s_aw1(k), r6_add(r6_mul(scalars(k, ns), upwind), r6_mul(scalars(((k) - (1)), ns), r6_sub(one, upwind)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6059
dncont(k) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6060
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6062
upcont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6063
dncont(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6065
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6066
a(1) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6067
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6068
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6069
d(1) = r6_sub(scalars(k, ns), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6072
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6073
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6074
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6075
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6076
d(k) = r6_sub(scalars(k, ns), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6077
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6079
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6081
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6082
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6083
b(k) = r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6084
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6085
d(k) = r6_sub(scalars(k, ns), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awscalars1(((k) + (1)), ns)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6087
{ const int mynn_lo5 = (((kts) + (1))), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6088
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6089
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6091
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6092
d(k) = r6_add(scalars(k, ns), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awscalars1(k, ns), s_awscalars1(((k) + (1)), ns))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6093
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6094
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6097
a(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6098
b(kte) = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6099
c(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6100
d(kte) = scalars(kte, ns);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6102
f_tridiag3(kte, a, b, c, d, x);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6104
{ const int mynn_lo6 = (kts), mynn_hi6 = (kte), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6105
scalars(k, ns) = mynn_max((float)(x(k)), (float)(1e-12f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6106
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6107
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6113-6138
__device__ __noinline__ void f_retrieve_exchange_coeffs(int kts, int kte, MynnArray<float, 1> dfm, MynnArray<float, 1> dfh, MynnArray<float, 1> dz, MynnArray<float, 1> km1, MynnArray<float, 1> kh1) {
MynnFrame mynn_frame(arena);
dfm.rebase({kts});
dfh.rebase({kts});
dz.rebase({kts});
km1.rebase({kts});
kh1.rebase({kts});
int k;
float dzk;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6126
km1(kts) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6127
kh1(kts) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6129
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6130
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6131
km1(k) = r6_mul(dfm(k), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6132
kh1(k) = r6_mul(dfh(k), dzk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6133
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src0); } }
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6142-6174
__device__ __noinline__ void f_tridiag(int n, MynnArray<float, 1> a, MynnArray<float, 1> b, MynnArray<float, 1> c, MynnArray<float, 1> d) {
MynnFrame mynn_frame(arena);
a.rebase({1});
b.rebase({1});
c.rebase({1});
d.rebase({1});
int i;
float p;
MynnArray<float, 1> q = mynn_array<float, 1, mynn_tile>(arena, {1}, {n});
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6160
c(n) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6161
q(1) = r6_neg(r6_div(c(1), b(1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6162
d(1) = r6_div(d(1), b(1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6164
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6165
p = r6_div(1.f, r6_add(b(i), r6_mul(a(i), q(((i) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6166
q(i) = r6_neg(r6_mul(c(i), p));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6167
d(i) = r6_mul(r6_sub(d(i), r6_mul(a(i), d(((i) - (1))))), p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6168
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6170
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6171
d(i) = r6_add(d(i), r6_mul(q(i), d(((i) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6172
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6178-6210
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6195
cp(1) = r6_div(c(1), b(1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6196
dp(1) = r6_div(d(1), b(1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6198
for (k=2; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6199
m = r6_sub(b(k), r6_mul(cp(((k) - (1))), a(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6200
cp(k) = r6_div(c(k), m);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6201
dp(k) = r6_div(r6_sub(d(k), r6_mul(dp(((k) - (1))), a(k))), m);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6202
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6204
x(kte) = dp(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6206
for (k=((kte) - (1)); ((-(1))) > 0 ? k <= (1) : k >= (1); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6207
x(k) = r6_sub(dp(k), r6_mul(cp(k), x(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6208
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6213-6248
__device__ __noinline__ void f_tridiag3(int kte, MynnArray<float, 1> a, MynnArray<float, 1> b, MynnArray<float, 1> c, MynnArray<float, 1> d, MynnArray<float, 1> x) {
MynnFrame mynn_frame(arena);
a.rebase({1});
b.rebase({1});
c.rebase({1});
d.rebase({1});
x.rebase({1});
constexpr int kts = 1;
int k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6234
for (k=((kte) - (1)); ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6235
d(k) = r6_sub(d(k), r6_div(r6_mul(c(k), d(((k) + (1)))), b(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6236
b(k) = r6_sub(b(k), r6_div(r6_mul(c(k), a(((k) + (1)))), b(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6237
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6239
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6240
d(k) = r6_sub(d(k), r6_div(r6_mul(a(k), d(((k) - (1)))), b(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6241
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6243
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6244
x(k) = r6_div(d(k), b(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6245
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6247
return;
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6277-6405
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6301
kpbl = 2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6304
k = ((kts) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6305
kthv = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6306
minthv = 9.e9f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6307
while (((zw1(k)) <= (200.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6308
if (((minthv) > (thv1(k)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6309
minthv = thv1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6310
kthv = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6311
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6312
k = ((k) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6313
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6316
pblh = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6317
k = ((kthv) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6318
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6320
delt_thv = 0.75f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6321
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6323
delt_thv = 1.25f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6324
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6326
pblh = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6327
k = ((kthv) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6328
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6329
if (((thv1(k)) >= (r6_add(minthv, delt_thv)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6330
pblh = r6_sub(zw1(k), r6_mul(dz1(((k) - (1))), mynn_min((float)(r6_div(r6_sub(thv1(k), r6_add(minthv, delt_thv)), mynn_max((float)(r6_sub(thv1(k), thv1(((k) - (1))))), (float)(1e-6f)))), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6333
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6334
if (((k) == (((kte) - (1))))) { pblh = zw1(((kts) + (1))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6335
if (((pblh) > (zero))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6336
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6339
if (((stable_method) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6344
ktke = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6345
maxqke = mynn_max((float)(zero), (float)(qke1(kts)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6348
tkeeps = r6_div(maxqke, 40.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6349
tkeeps = mynn_max((float)(tkeeps), (float)(0.01f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6350
pblh_stable = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6352
k = ((ktke) + (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6353
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6355
qtke = mynn_max((float)(zero), (float)(r6_mul(p5, qke1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6356
qtkem1 = mynn_max((float)(zero), (float)(r6_mul(p5, qke1(((k) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6357
if (((qtke) <= (tkeeps))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6358
pblh_stable = r6_sub(zw1(k), r6_mul(dz1(((k) - (1))), mynn_min((float)(r6_div(r6_sub(tkeeps, qtke), mynn_max((float)(r6_sub(qtkem1, qtke)), (float)(1e-6f)))), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6361
pblh_stable = mynn_max((float)(pblh_stable), (float)(zw1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6363
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6365
if (((k) == (((kte) - (1))))) { pblh_stable = zw1(((kts) + (1))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6366
if (((pblh_stable) != (zero))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6367
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6373
pblh_stable = mynn_min((float)(pblh_stable), (float)(r6_add(pblh, 350.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6374
pblh_stable = mynn_max((float)(pblh_stable), (float)(mynn_max((float)(r6_sub(pblh, 350.f)), (float)(ten))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6376
if (((maxqke) <= (tkeeps))) { pblh_stable = pblh; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6377
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6379
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(pblh, hundred), hundred))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6380
cpblh = r6_add(r6_mul(400.f, r6_sub(one, wt)), r6_mul(700.f, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6381
pblh_stable = mynn_max((float)(r6_mul(ust, cpblh)), (float)(ten));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6384
pblh_stable = mynn_max((float)(pblh_stable), (float)(r6_sub(pblh, sbl_lim)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6385
pblh_stable = mynn_min((float)(pblh_stable), (float)(r6_add(pblh, sbl_lim)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6386
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6389
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(pblh, sbl_lim), sbl_damp))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6390
pblh = r6_add(r6_mul(pblh_stable, r6_sub(one, wt)), r6_mul(pblh, wt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6393
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6394
if (((zw1(k)) >= (pblh))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6395
kpbl = ((k) - (1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6396
break;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6397
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6398
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6427-7768
__device__ __noinline__ void f_dmp_mf(int i, int j, int kts, int kte, float dt, MynnArray<float, 1> zw1, MynnArray<float, 1> dz1, MynnArray<float, 1> pres1, MynnArray<float, 1> rho1, int momentum_opt, int tke_opt, int scalar_opt, int aerosol_opt, int numcon_opt, float bl_mynn_closure, MynnArray<float, 1> u1, MynnArray<float, 1> v1, MynnArray<float, 1> w1, MynnArray<float, 1> th1, MynnArray<float, 1> thl1, MynnArray<float, 1> thv1, MynnArray<float, 1> tk1, MynnArray<float, 1> qt1, MynnArray<float, 1> qv1, MynnArray<float, 1> qc1, MynnArray<float, 1> qke1, MynnArray<float, 1> qsq1, MynnArray<float, 1> qnc1, MynnArray<float, 1> qni1, MynnArray<float, 1> qnwfa1, MynnArray<float, 1> qnifa1, MynnArray<float, 1> qnbca1, MynnArray<float, 1> ex1, MynnArray<float, 1> vt1, MynnArray<float, 1> vq1, MynnArray<float, 1> sgm1, float ust, float flt, float fltv, float flq, float flqv, float pblh, int kpbl, float dx, float landsea, float ts, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_qt1, MynnArray<float, 1> edmf_thl1, MynnArray<float, 1> edmf_ent1, MynnArray<float, 1> edmf_qc1, MynnArray<float, 1> edmf_qv1, MynnArray<float, 1> edmf_u1, MynnArray<float, 1> edmf_v1, MynnArray<float, 1> s_aw1, MynnArray<float, 1> s_awthl1, MynnArray<float, 1> s_awqt1, MynnArray<float, 1> s_awqv1, MynnArray<float, 1> s_awqc1, MynnArray<float, 1> s_awu1, MynnArray<float, 1> s_awv1, MynnArray<float, 1> s_awqke1, MynnArray<float, 1> s_awqsq1, MynnArray<float, 1> s_awqnc1, MynnArray<float, 1> s_awqni1, MynnArray<float, 1> s_awqnwfa1, MynnArray<float, 1> s_awqnifa1, MynnArray<float, 1> s_awqnbca1, MynnArray<float, 1> sub_thl1, MynnArray<float, 1> sub_sqv1, MynnArray<float, 1> sub_u1, MynnArray<float, 1> sub_v1, MynnArray<float, 1> det_thl1, MynnArray<float, 1> det_sqv1, MynnArray<float, 1> det_sqc1, MynnArray<float, 1> det_u1, MynnArray<float, 1> det_v1, int nchem, MynnArray<float, 2> chem1, MynnArray<float, 2> s_awchem1, bool mix_chem, int nscalars, MynnArray<float, 2> scalars, MynnArray<float, 2> s_awscalars1, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> cldfra_bl1, MynnArray<float, 1> qc_bl1_old, MynnArray<float, 1> cldfra_bl1_old, MynnOptional<bool> f_qc, MynnOptional<bool> f_qi, MynnOptional<bool> f_qnc, MynnOptional<bool> f_qni, MynnOptional<bool> f_qnwfa, MynnOptional<bool> f_qnifa, MynnOptional<bool> f_qnbca, float psig_shcu, float& maxwidth, int& ktop, float& maxmf, float& ztop, float& excess_h, float& excess_q, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1, MynnArray<float, 1> tkeprod_up, MynnArray<float, 1> el1) {
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
qsq1.rebase({kts});
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
s_awqsq1.rebase({kts});
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
MynnArray<float, 2> upqsq = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
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
float qsqn;
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
int cb_check;
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
float mf_at_cb;
float qcfac;
float sigqfac;
float cffac;
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6630
ktop = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6631
ztop = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6632
maxmf = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6633
maxwidth = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6634
excess_h = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6635
excess_q = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6637
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upw).size(); mynn_element += lane_width) { upw.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6638
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upthl).size(); mynn_element += lane_width) { upthl.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6639
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upthv).size(); mynn_element += lane_width) { upthv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6640
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqt).size(); mynn_element += lane_width) { upqt.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6641
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upa).size(); mynn_element += lane_width) { upa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6642
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upu).size(); mynn_element += lane_width) { upu.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6643
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upv).size(); mynn_element += lane_width) { upv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6644
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqc).size(); mynn_element += lane_width) { upqc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6645
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqv).size(); mynn_element += lane_width) { upqv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6646
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqke).size(); mynn_element += lane_width) { upqke.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6647
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqsq).size(); mynn_element += lane_width) { upqsq.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6648
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnc).size(); mynn_element += lane_width) { upqnc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6649
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqni).size(); mynn_element += lane_width) { upqni.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6650
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnwfa).size(); mynn_element += lane_width) { upqnwfa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6651
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnifa).size(); mynn_element += lane_width) { upqnifa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6652
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnbca).size(); mynn_element += lane_width) { upqnbca.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6653
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6654
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upchem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { upchem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6655
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6656
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6657
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upscalars.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nscalars})).size(); mynn_element += lane_width) { upscalars.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nscalars}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6658
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6659
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (ent).size(); mynn_element += lane_width) { ent.linear(mynn_element) = 0.001f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6661
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1).size(); mynn_element += lane_width) { edmf_a1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6662
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1).size(); mynn_element += lane_width) { edmf_w1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6663
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt1).size(); mynn_element += lane_width) { edmf_qt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6664
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1).size(); mynn_element += lane_width) { edmf_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6665
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent1).size(); mynn_element += lane_width) { edmf_ent1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6666
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1).size(); mynn_element += lane_width) { edmf_qc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6667
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qv1).size(); mynn_element += lane_width) { edmf_qv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6668
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_u1).size(); mynn_element += lane_width) { edmf_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6669
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_v1).size(); mynn_element += lane_width) { edmf_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6670
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6671
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_chem.slice(MynnRange{kts, kte}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { edmf_chem.slice(MynnRange{kts, kte}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6672
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6674
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_aw1).size(); mynn_element += lane_width) { s_aw1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6675
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awthl1).size(); mynn_element += lane_width) { s_awthl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6676
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqt1).size(); mynn_element += lane_width) { s_awqt1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6677
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqv1).size(); mynn_element += lane_width) { s_awqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6678
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqc1).size(); mynn_element += lane_width) { s_awqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6679
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awu1).size(); mynn_element += lane_width) { s_awu1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6680
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awv1).size(); mynn_element += lane_width) { s_awv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6681
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqke1).size(); mynn_element += lane_width) { s_awqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6682
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqsq1).size(); mynn_element += lane_width) { s_awqsq1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6683
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnc1).size(); mynn_element += lane_width) { s_awqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6684
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqni1).size(); mynn_element += lane_width) { s_awqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6685
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnwfa1).size(); mynn_element += lane_width) { s_awqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6686
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnifa1).size(); mynn_element += lane_width) { s_awqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6687
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnbca1).size(); mynn_element += lane_width) { s_awqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6688
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6689
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6690
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6691
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6692
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars})).size(); mynn_element += lane_width) { s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6693
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6695
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_thl1).size(); mynn_element += lane_width) { sub_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6696
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_sqv1).size(); mynn_element += lane_width) { sub_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6697
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_u1).size(); mynn_element += lane_width) { sub_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6698
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_v1).size(); mynn_element += lane_width) { sub_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6699
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_thl1).size(); mynn_element += lane_width) { det_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6700
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqv1).size(); mynn_element += lane_width) { det_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6701
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqc1).size(); mynn_element += lane_width) { det_sqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6702
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_u1).size(); mynn_element += lane_width) { det_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6703
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_v1).size(); mynn_element += lane_width) { det_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6704
nup2 = nup;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6705
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprod_up).size(); mynn_element += lane_width) { tkeprod_up.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6707
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6708
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6709
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6710
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6711
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6715
maxw = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6716
cloud_base = 9000.0f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6717
qkebl = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6718
for (k=1; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6719
zagl = r6_add(zw1(k), r6_mul(p5, dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6720
if (((zagl) > (r6_add(pblh, 500.f)))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6722
wpbl = w1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6723
if (((w1(k)) < (zero))) { wpbl = r6_mul(two, w1(k)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6724
maxw = mynn_max((float)(maxw), (float)(mynn_abs(wpbl)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6727
if (((zagl) <= (50.f))) { k50 = k; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6730
if (((k) <= (kpbl))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6731
qkebl = r6_add(qkebl, qke1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6732
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6735
qc_sgs = mynn_max((float)(qc1(k)), (float)(qc_bl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6736
if (((((((qc_sgs) > (1e-5f))) && (((cldfra_bl1(k)) >= (p5))))) && (((cloud_base) == (9000.0f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6737
cloud_base = zw1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6738
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6739
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6740
qkebl = r6_div(qkebl, ((float)(kpbl)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6743
maxw = mynn_max((float)(r6_sub(maxw, one)), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6744
psig_w = mynn_max((float)(r6_sub(one, maxw)), (float)(zero));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6745
psig_w = mynn_min((float)(psig_w), (float)(psig_shcu));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6748
fltv2 = fltv;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6749
if (((((psig_w) < (1e-2f))) && (((fltv) > (zero))))) { fltv2 = r6_neg(r6_mul(1.f, fltv)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6751
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6752
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6753
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6754
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6758
superadiabatic = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6759
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6760
hux = r6_neg(0.0024f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6761
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6762
hux = r6_neg(0.0030f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6763
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6764
tvs = r6_mul(ts, r6_add(one, r6_mul(p608, qv1(kts))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6765
if (((r6_div(r6_sub(thv1(kts), tvs), r6_mul(p5, dz1(kts)))) < (hux))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6767
superadiabatic = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6768
for (k=2; (1) > 0 ? k <= (mynn_max((int)(k50), (int)(2))) : k >= (mynn_max((int)(k50), (int)(2))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6769
hux = r6_neg(0.0018f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6770
dthvdz = r6_div(r6_sub(thv1(k), thv1(((k) - (1)))), r6_mul(p5, r6_add(dz1(k), dz1(((k) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6771
if (((dthvdz) < (hux))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6772
superadiabatic = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6773
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6774
superadiabatic = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6775
break;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6776
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6777
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6778
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6780
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6781
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6782
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6793
maxwidth_dx = mynn_min((float)(r6_mul(dx, dcut)), (float)(lmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6795
maxwidth_pbl = mynn_min((float)(r6_mul(1.1f, pblh)), (float)(lmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6797
if (((r6_sub(landsea, 1.5f)) < (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6798
maxwidth_cld = mynn_min((float)(mynn_max((float)(r6_mul(0.5f, cloud_base)), (float)(300.f))), (float)(lmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6799
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6800
maxwidth_cld = mynn_min((float)(mynn_max((float)(r6_mul(0.8f, cloud_base)), (float)(300.f))), (float)(lmax));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6801
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6803
wspd_pbl = mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts)))), (float)(0.01f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6806
if (((r6_sub(landsea, 1.5f)) < (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6807
maxwidth_flx = mynn_max((float)(zero), (float)(mynn_min((float)(r6_mul(1000.f, r6_add(r6_mul(0.6f, mynn_tanh(r6_div(r6_sub(fltv, 0.040f), 0.04f))), p5))), (float)(1000.f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6808
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6809
maxwidth_flx = mynn_max((float)(zero), (float)(mynn_min((float)(r6_mul(1000.f, r6_add(r6_mul(0.6f, mynn_tanh(r6_div(r6_sub(fltv, 0.007f), 0.02f))), p5))), (float)(1000.f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6811
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6812
maxwidth = mynn_min((float)(maxwidth_dx), (float)(maxwidth_pbl));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6813
maxwidth = mynn_min((float)(maxwidth), (float)(maxwidth_cld));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6814
maxwidth = mynn_min((float)(maxwidth), (float)(maxwidth_flx));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6815
minwidth = lmin;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6817
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6818
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6819
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6820
; /* Native diagnostic print; no field arithmetic. */
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6828
if (((maxwidth) <= (minwidth))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6829
nup2 = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6830
maxwidth = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6831
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6834
if (((((((((flt) > (zero))) && (((fltv2) > (0.008f))))) && (((maxwidth) > (minwidth))))) && (superadiabatic))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6837
cn = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6838
d = r6_neg(1.9f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6839
dl = r6_div(r6_sub(maxwidth, minwidth), ((float)(((nup) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6840
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6842
l = r6_add(minwidth, r6_mul(dl, ((float)(((ip) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6843
cn = r6_add(cn, r6_mul(r6_div(r6_mul(mynn_pow(l, d), r6_mul(l, l)), r6_mul(dx, dx)), dl));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6844
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6845
c = r6_div(atot, cn);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6848
if (((r6_sub(landsea, 1.5f)) < (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6849
acfac = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(fltv2, 0.02f), 0.05f))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6850
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6851
acfac = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(fltv2, 0.012f), 0.03f))), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6852
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6856
ac_wsp = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(zero), (float)(r6_sub(wspd_pbl, 13.0f))), 12.f)), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6857
acfac = mynn_min((float)(acfac), (float)(ac_wsp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6860
an2 = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6861
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6863
l = r6_add(minwidth, r6_mul(dl, ((float)(((ip) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6864
n = r6_mul(c, mynn_pow(l, d));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6865
upa(1, ip) = r6_mul(r6_div(r6_mul(r6_mul(n, l), l), r6_mul(dx, dx)), dl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6867
upa(1, ip) = r6_mul(upa(1, ip), acfac);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6868
an2 = r6_add(an2, upa(1, ip));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6871
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6873
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6874
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6875
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6876
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6877
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6878
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6879
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6882
z0 = 50.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6883
pwmin = 0.1f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6884
pwmax = 0.4f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6886
wstar = mynn_max((float)(mynn_pow(r6_mul(r6_mul(gtr, fltv2), pblh), p333)), (float)(1.e-2f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6887
qstar = r6_div(mynn_max((float)(flq), (float)(1e-5f)), wstar);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6888
thstar = r6_div(flt, wstar);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6890
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6891
csigma = 1.34f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6892
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6893
csigma = 1.34f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6894
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6896
if (env_subs) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6897
exc_fac = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6898
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6899
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6906
exc_fac = 9.28f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6907
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6909
exc_fac = 0.58f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6910
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6911
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6913
exc_fac = r6_mul(exc_fac, ac_wsp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6916
sigmaw = r6_mul(r6_mul(r6_mul(csigma, wstar), mynn_pow(r6_div(z0, pblh), p333)), r6_sub(one, r6_div(r6_mul(0.8f, z0), pblh)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6917
sigmaqt = r6_mul(r6_mul(csigma, qstar), mynn_pow(r6_div(z0, pblh), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6918
sigmath = r6_mul(r6_mul(csigma, thstar), mynn_pow(r6_div(z0, pblh), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6922
wmin = mynn_min((float)(r6_mul(sigmaw, pwmin)), (float)(0.1f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6923
wmax = mynn_min((float)(r6_mul(sigmaw, pwmax)), (float)(0.5f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6925
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6926
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6927
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6928
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6929
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6930
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6933
{ const int mynn_lo0 = (1), mynn_hi0 = (nup), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { ip = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6934
wlv = r6_add(wmin, r6_mul(r6_div(r6_sub(wmax, wmin), ((float)(nup2))), ((float)(((ip) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6935
upa(2, ip) = upa(1, ip);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6937
upw(2, ip) = r6_add(wmin, r6_mul(r6_div(((float)(ip)), ((float)(nup))), r6_sub(wmax, wmin)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6938
upu(2, ip) = r6_div(r6_add(r6_mul(u1(kts), dz1(((kts) + (1)))), r6_mul(u1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6939
upv(2, ip) = r6_div(r6_add(r6_mul(v1(kts), dz1(((kts) + (1)))), r6_mul(v1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6940
upqc(2, ip) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6943
exc_heat = r6_div(r6_mul(r6_mul(exc_fac, upw(2, ip)), sigmath), sigmaw);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6944
upthv(2, ip) = r6_add(r6_div(r6_add(r6_mul(thv1(kts), dz1(((kts) + (1)))), r6_mul(thv1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1))))), exc_heat);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6946
upthl(2, ip) = r6_add(r6_div(r6_add(r6_mul(thl1(kts), dz1(((kts) + (1)))), r6_mul(thl1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1))))), exc_heat);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6949
exc_moist = r6_div(r6_mul(r6_mul(exc_fac, upw(2, ip)), sigmaqt), sigmaw);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6950
upqt(2, ip) = r6_add(r6_div(r6_add(r6_mul(qt1(kts), dz1(((kts) + (1)))), r6_mul(qt1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1))))), exc_moist);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6952
upqke(2, ip) = r6_div(r6_add(r6_mul(qke1(kts), dz1(((kts) + (1)))), r6_mul(qke1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6953
upqsq(2, ip) = r6_div(r6_add(r6_mul(qsq1(kts), dz1(((kts) + (1)))), r6_mul(qsq1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6954
upqnc(2, ip) = r6_div(r6_add(r6_mul(qnc1(kts), dz1(((kts) + (1)))), r6_mul(qnc1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6955
upqni(2, ip) = r6_div(r6_add(r6_mul(qni1(kts), dz1(((kts) + (1)))), r6_mul(qni1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6956
upqnwfa(2, ip) = r6_div(r6_add(r6_mul(qnwfa1(kts), dz1(((kts) + (1)))), r6_mul(qnwfa1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6957
upqnifa(2, ip) = r6_div(r6_add(r6_mul(qnifa1(kts), dz1(((kts) + (1)))), r6_mul(qnifa1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6958
upqnbca(2, ip) = r6_div(r6_add(r6_mul(qnbca1(kts), dz1(((kts) + (1)))), r6_mul(qnbca1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6959
} mynn_lane_sync(); ip = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; exc_heat = mynn_lane_bcast(exc_heat, mynn_src0); exc_moist = mynn_lane_bcast(exc_moist, mynn_src0); wlv = mynn_lane_bcast(wlv, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6961
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6962
{ const int mynn_lo1 = (1), mynn_hi1 = (nup), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { ip = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6963
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6964
upchem(2, ip, ic) = r6_div(r6_add(r6_mul(chem1(kts, ic), dz1(((kts) + (1)))), r6_mul(chem1(((kts) + (1)), ic), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6965
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6966
} mynn_lane_sync(); ip = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; ic = mynn_lane_bcast(ic, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6967
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6968
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6969
{ const int mynn_lo2 = (1), mynn_hi2 = (nup), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { ip = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6970
for (ic=1; (1) > 0 ? ic <= (nscalars) : ic >= (nscalars); ic += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6971
upscalars(2, ip, ic) = r6_div(r6_add(r6_mul(scalars(kts, ic), dz1(((kts) + (1)))), r6_mul(scalars(((kts) + (1)), ic), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6972
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6973
} mynn_lane_sync(); ip = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; ic = mynn_lane_bcast(ic, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6974
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6977
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_thl.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_thl.slice(MynnRange{kts, kte}).linear(mynn_element) = thl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6978
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_sqv.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_sqv.slice(MynnRange{kts, kte}).linear(mynn_element) = qv1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6979
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_sqc.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_sqc.slice(MynnRange{kts, kte}).linear(mynn_element) = qc1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6980
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_u.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_u.slice(MynnRange{kts, kte}).linear(mynn_element) = u1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6981
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_v.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_v.slice(MynnRange{kts, kte}).linear(mynn_element) = v1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6983
rhoi(kts) = rho1(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6984
dzi(kts) = r6_mul(p5, dz1(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6985
exneri(kts) = ex1(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6986
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6987
rhoi(k) = r6_div(r6_add(r6_mul(rho1(((k) - (1))), dz1(k)), r6_mul(rho1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6988
dzi(k) = r6_div(r6_add(r6_mul(dz1(((k) - (1))), dz1(k)), r6_mul(dz1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6989
exneri(k) = r6_div(r6_add(r6_mul(ex1(((k) - (1))), dz1(k)), r6_mul(ex1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6990
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6993
dxsa = r6_sub(one, mynn_min((float)(mynn_max((float)(zero), (float)(r6_div(r6_sub(12000.0f, dx), r6_sub(12000.0f, 3000.0f))))), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6995
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (ktop_plume).size(); mynn_element += lane_width) { ktop_plume.linear(mynn_element) = 0; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6997
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6998
qcn = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:6999
overshoot = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7000
l = r6_add(minwidth, r6_mul(dl, ((float)(((ip) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7001
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (2))) : k >= (((kte) - (2))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7002
zagl = r6_add(zw1(k), r6_mul(p5, dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7005
wmin = r6_add(p3, r6_mul(l, 0.0005f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7013
entfac = r6_mul(0.21f, mynn_min((float)(mynn_max((float)(mynn_sqrt(qkebl)), (float)(1.30f))), (float)(1.57f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7016
wt2 = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl, pblh)), (float)(zero)), 500.f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7017
entfac = r6_add(r6_mul(entfac, r6_sub(one, wt2)), r6_mul(wt2, 0.33f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7018
ent(k, ip) = r6_div(entfac, r6_mul(mynn_min((float)(mynn_max((float)(upw(((k) - (1)), ip)), (float)(wmin))), (float)(one)), l));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7026
ent(k, ip) = mynn_max((float)(ent(k, ip)), (float)(0.0003f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7030
if (((zw1(k)) >= (mynn_min((float)(r6_add(pblh, 1500.f)), (float)(4000.f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7031
ent(k, ip) = r6_add(ent(k, ip), r6_mul(r6_sub(zw1(k), mynn_min((float)(r6_add(pblh, 1500.f)), (float)(4000.f))), 5.0e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7032
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7035
ent(k, ip) = r6_mul(ent(k, ip), r6_sub(one, pattern_spp_pbl1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7037
ent(k, ip) = mynn_min((float)(ent(k, ip)), (float)(r6_div(0.9f, dz1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7040
uk = r6_div(r6_add(r6_mul(u1(k), dz1(((k) + (1)))), r6_mul(u1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7041
ukm1 = r6_div(r6_add(r6_mul(u1(((k) - (1))), dz1(k)), r6_mul(u1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7042
vk = r6_div(r6_add(r6_mul(v1(k), dz1(((k) + (1)))), r6_mul(v1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7043
vkm1 = r6_div(r6_add(r6_mul(v1(((k) - (1))), dz1(k)), r6_mul(v1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7046
entexp = r6_mul(ent(k, ip), dz1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7047
entexm = r6_mul(entexp, p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7048
qtn = r6_add(r6_mul(upqt(k, ip), r6_sub(one, entexp)), r6_mul(qt1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7049
thln = r6_add(r6_mul(upthl(k, ip), r6_sub(one, entexp)), r6_mul(thl1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7050
un = r6_add(r6_add(r6_mul(upu(k, ip), r6_sub(one, entexm)), r6_mul(u1(k), entexm)), r6_mul(r6_mul(dxsa, pgfac), r6_sub(uk, ukm1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7051
vn = r6_add(r6_add(r6_mul(upv(k, ip), r6_sub(one, entexm)), r6_mul(v1(k), entexm)), r6_mul(r6_mul(dxsa, pgfac), r6_sub(vk, vkm1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7052
qken = r6_add(r6_mul(upqke(k, ip), r6_sub(one, entexp)), r6_mul(qke1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7053
qsqn = r6_add(r6_mul(upqsq(k, ip), r6_sub(one, entexp)), r6_mul(qsq1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7054
qncn = r6_add(r6_mul(upqnc(k, ip), r6_sub(one, entexp)), r6_mul(qnc1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7055
qnin = r6_add(r6_mul(upqni(k, ip), r6_sub(one, entexp)), r6_mul(qni1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7056
qnwfan = r6_add(r6_mul(upqnwfa(k, ip), r6_sub(one, entexp)), r6_mul(qnwfa1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7057
qnifan = r6_add(r6_mul(upqnifa(k, ip), r6_sub(one, entexp)), r6_mul(qnifa1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7058
qnbcan = r6_add(r6_mul(upqnbca(k, ip), r6_sub(one, entexp)), r6_mul(qnbca1(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7062
qc_ent = qcn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7063
qt_ent = qtn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7064
thl_ent = thln;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7074
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7075
{ const int mynn_lo4 = (1), mynn_hi4 = (nchem), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { ic = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7079
chemn(ic) = r6_add(r6_mul(upchem(k, ip, ic), r6_sub(one, entexp)), r6_mul(chem1(k, ic), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7080
} mynn_lane_sync(); ic = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7081
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7082
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7083
{ const int mynn_lo5 = (1), mynn_hi5 = (nscalars), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { ic = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7084
scalarsn(ic) = r6_add(r6_mul(upscalars(k, ip, ic), r6_sub(one, entexp)), r6_mul(scalars(k, ic), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7085
} mynn_lane_sync(); ic = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7086
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7089
pk = r6_div(r6_add(r6_mul(pres1(k), dz1(((k) + (1)))), r6_mul(pres1(((k) + (1))), dz1(k))), r6_add(dz1(k), dz1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7091
f_condensation_edmf(qtn, thln, pk, zw1(((k) + (1))), thvn, qcn);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7094
thvk = r6_div(r6_add(r6_mul(thv1(k), dz1(((k) + (1)))), r6_mul(thv1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7095
thvkm1 = r6_div(r6_add(r6_mul(thv1(((k) - (1))), dz1(k)), r6_mul(thv1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7098
b = r6_mul(grav, r6_sub(r6_div(thvn, thvk), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7099
if (((b) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7100
bcoeff = 0.15f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7101
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7102
bcoeff = 0.2f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7103
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7113
if (((upw(((k) - (1)), ip)) < (0.2f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7114
wn = r6_add(upw(k, ip), r6_mul(r6_add(r6_mul(r6_neg(r6_mul(2.f, ent(k, ip))), upw(k, ip)), r6_div(r6_mul(bcoeff, b), mynn_max((float)(upw(k, ip)), (float)(0.2f)))), mynn_min((float)(dz1(k)), (float)(250.f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7115
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7116
wn = r6_add(upw(k, ip), r6_mul(r6_add(r6_mul(r6_neg(r6_mul(2.f, ent(k, ip))), upw(k, ip)), r6_div(r6_mul(bcoeff, b), upw(k, ip))), mynn_min((float)(dz1(k)), (float)(250.f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7117
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7120
if (((wn) > (r6_add(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7121
wn = r6_add(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7122
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7124
if (((wn) < (r6_sub(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7125
wn = r6_sub(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7126
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7127
wn = mynn_min((float)(mynn_max((float)(zero), (float)(wn))), (float)(3.0f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7131
if (((((k) == (((kts) + (1))))) && (((wn) < (1e-8f))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7132
nup2 = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7133
break;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7134
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7136
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7137
if (((wn) >= (3.0f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7139
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7140
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7141
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7142
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7143
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7144
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7147
if (((((wn) <= (zero))) && (((overshoot) == (0))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7148
overshoot = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7149
if (((r6_sub(thvk, thvkm1)) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7150
bvf = mynn_sqrt(r6_div(r6_mul(gtr, r6_sub(thvk, thvkm1)), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7152
frz = r6_div(upw(k, ip), r6_mul(bvf, dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7154
dzp = r6_mul(dz1(k), mynn_max((float)(zero), (float)(mynn_min((float)(frz), (float)(one)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7155
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7156
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7157
dzp = dz1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7158
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7168
aratio = mynn_min((float)(r6_div(upa(k, ip), r6_sub(one, upa(k, ip)))), (float)(p5));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7169
detturb = 0.00008f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7170
oow = r6_neg(r6_div(0.060f, mynn_max((float)(r6_mul(0.5f, r6_add(wn, upw(k, ip)))), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7171
detrate = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(oow, r6_sub(wn, upw(k, ip))), dz1(k))), (float)(detturb))), (float)(.0002f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7172
detrateuv = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(oow, r6_sub(wn, upw(k, ip))), dz1(k))), (float)(detturb))), (float)(.0001f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7173
envm_thl(k) = r6_add(envm_thl(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(thl_ent, upthl(k, ip))), thl1(k)), detrate), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7174
qv_ent = r6_mul(p5, r6_add(mynn_max((float)(zero), (float)(r6_sub(qt_ent, qc_ent))), mynn_max((float)(zero), (float)(r6_sub(upqt(k, ip), upqc(k, ip))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7175
envm_sqv(k) = r6_add(envm_sqv(k), r6_mul(r6_mul(r6_mul(r6_sub(qv_ent, qv1(k)), detrate), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7176
if (((upqc(k, ip)) > (1e-8f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7177
if (((qc1(k)) > (1e-6f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7178
qc_grid = qc1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7179
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7180
qc_grid = qc_bl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7181
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7182
envm_sqc(k) = r6_add(envm_sqc(k), r6_mul(r6_mul(r6_mul(mynn_max((float)(zero), (float)(r6_sub(r6_mul(r6_mul(upa(k, ip), p5), r6_add(qcn, upqc(k, ip))), qc_grid))), detrate), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7183
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7184
envm_u(k) = r6_add(envm_u(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(un, upu(k, ip))), u1(k)), detrateuv), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7185
envm_v(k) = r6_add(envm_v(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(vn, upv(k, ip))), v1(k)), detrateuv), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7187
if (((wn) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7189
upw(((k) + (1)), ip) = wn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7190
upthv(((k) + (1)), ip) = thvn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7191
upthl(((k) + (1)), ip) = thln;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7192
upqt(((k) + (1)), ip) = qtn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7193
upqc(((k) + (1)), ip) = qcn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7194
upu(((k) + (1)), ip) = un;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7195
upv(((k) + (1)), ip) = vn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7196
upqke(((k) + (1)), ip) = qken;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7197
upqsq(((k) + (1)), ip) = qsqn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7198
upqnc(((k) + (1)), ip) = qncn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7199
upqni(((k) + (1)), ip) = qnin;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7200
upqnwfa(((k) + (1)), ip) = qnwfan;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7201
upqnifa(((k) + (1)), ip) = qnifan;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7202
upqnbca(((k) + (1)), ip) = qnbcan;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7203
upa(((k) + (1)), ip) = upa(k, ip);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7204
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7205
{ const int mynn_lo6 = (1), mynn_hi6 = (nchem), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { ic = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7206
upchem(((k) + (1)), ip, ic) = chemn(ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7207
} mynn_lane_sync(); ic = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7208
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7209
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7210
{ const int mynn_lo7 = (1), mynn_hi7 = (nscalars), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { ic = mynn_lo7 + mynn_it7*mynn_st7;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7211
upscalars(((k) + (1)), ip, ic) = scalarsn(ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7212
} mynn_lane_sync(); ic = mynn_lo7 + mynn_n7*mynn_st7; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7213
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7214
ktop = mynn_max((int)(ktop), (int)(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7215
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7216
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7217
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7218
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7219
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7220
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7221
break;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7222
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7223
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7225
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7226
if (((((((((mynn_maxval(upw.slice(MynnRange{upw.lo[0], upw.hi[0]}, ip))) > (10.0f))) || (((mynn_minval(upa.slice(MynnRange{upa.lo[0], upa.hi[0]}, ip))) < (zero))))) || (((mynn_maxval(upa.slice(MynnRange{upa.lo[0], upa.hi[0]}, ip))) > (atot))))) || (((nup2) > (10))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7229
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7230
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7231
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7233
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7234
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7235
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7236
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7237
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7238
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7239
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7240
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7241
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7242
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7243
ktop_plume(ip) = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7244
excess_h = mynn_max((float)(excess_h), (float)(exc_heat));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7245
excess_q = mynn_max((float)(excess_q), (float)(exc_moist));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7246
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7247
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7249
nup2 = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7250
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7252
ktop = mynn_min((int)(ktop), (int)(((kte) - (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7253
if (((ktop) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7254
ztop = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7255
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7256
ztop = zw1(ktop);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7257
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7259
if (((nup2) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7262
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7263
{ const int mynn_lo8 = (kts), mynn_hi8 = (((kte) - (1))), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7264
s_aw1(k) = r6_add(s_aw1(k), r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7265
s_awthl1(k) = r6_add(s_awthl1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upthl(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7266
s_awqt1(k) = r6_add(s_awqt1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqt(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7271
qc_plume = upqc(k, ip);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7275
s_awqc1(k) = r6_add(s_awqc1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), qc_plume), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7276
s_awqv1(k) = r6_sub(s_awqt1(k), s_awqc1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7277
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; if (mynn_n8 > 0) { const int mynn_src8 = (mynn_n8 - 1) % lane_width; qc_plume = mynn_lane_bcast(qc_plume, mynn_src8); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7278
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7280
if (((momentum_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7281
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7282
{ const int mynn_lo9 = (kts), mynn_hi9 = (((kte) - (1))), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7283
s_awu1(k) = r6_add(s_awu1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upu(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7284
s_awv1(k) = r6_add(s_awv1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upv(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7285
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7286
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7287
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7289
if (((tke_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7290
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7291
{ const int mynn_lo10 = (kts), mynn_hi10 = (((kte) - (1))), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7292
s_awqke1(k) = r6_add(s_awqke1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqke(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7293
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7294
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7296
if (((bl_mynn_closure) > (2.5f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7297
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7298
{ const int mynn_lo11 = (kts), mynn_hi11 = (((kte) - (1))), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { k = mynn_lo11 + mynn_it11*mynn_st11;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7299
s_awqsq1(k) = r6_add(s_awqsq1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqsq(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7300
} mynn_lane_sync(); k = mynn_lo11 + mynn_n11*mynn_st11; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7301
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7302
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7303
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7305
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7306
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7307
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7308
{ const int mynn_lo12 = (1), mynn_hi12 = (nchem), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { ic = mynn_lo12 + mynn_it12*mynn_st12;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7309
s_awchem1(k, ic) = r6_add(s_awchem1(k, ic), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upchem(k, ip, ic)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7310
} mynn_lane_sync(); ic = mynn_lo12 + mynn_n12*mynn_st12; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7311
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7312
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7313
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7315
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7316
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7317
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7318
{ const int mynn_lo13 = (1), mynn_hi13 = (nscalars), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { ic = mynn_lo13 + mynn_it13*mynn_st13;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7319
s_awscalars1(k, ic) = r6_add(s_awscalars1(k, ic), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upscalars(k, ip, ic)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7320
} mynn_lane_sync(); ic = mynn_lo13 + mynn_n13*mynn_st13; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7321
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7322
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7323
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7324
if (((numcon_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7325
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7326
{ const int mynn_lo14 = (kts), mynn_hi14 = (((kte) - (1))), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { k = mynn_lo14 + mynn_it14*mynn_st14;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7327
s_awqnc1(k) = r6_add(s_awqnc1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnc(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7328
s_awqni1(k) = r6_add(s_awqni1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqni(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7329
} mynn_lane_sync(); k = mynn_lo14 + mynn_n14*mynn_st14; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7330
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7331
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7332
if (((aerosol_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7333
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7334
{ const int mynn_lo15 = (kts), mynn_hi15 = (((kte) - (1))), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { k = mynn_lo15 + mynn_it15*mynn_st15;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7335
s_awqnwfa1(k) = r6_add(s_awqnwfa1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnwfa(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7336
s_awqnifa1(k) = r6_add(s_awqnifa1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnifa(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7337
s_awqnbca1(k) = r6_add(s_awqnbca1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnbca(k, ip)), psig_w));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7338
} mynn_lane_sync(); k = mynn_lo15 + mynn_n15*mynn_st15; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7339
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7340
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7345
if (((mynn_abs(s_aw1(((kts) + (1))))) > (1e-8f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7346
flx1 = mynn_max((float)(r6_div(r6_mul(s_aw1(((kts) + (1))), r6_sub(thv1(kts), thv1(((kts) + (1))))), dzi(((kts) + (1))))), (float)(1.0e-6f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7347
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7348
flx1 = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7351
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7352
adjustment = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7353
flt2 = mynn_max((float)(zero), (float)(fltv));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7356
if (((((flx1) > (r6_div(r6_mul(fluxportion, flt2), dz1(kts))))) && (((flx1) > (zero))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7357
adjustment = mynn_max((float)(r6_div(r6_div(r6_mul(fluxportion, flt2), dz1(kts)), flx1)), (float)(0.01f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7358
for (int mynn_element=0; mynn_element < (s_aw1).size(); ++mynn_element) { s_aw1.linear(mynn_element) = r6_mul(s_aw1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7359
for (int mynn_element=0; mynn_element < (s_awthl1).size(); ++mynn_element) { s_awthl1.linear(mynn_element) = r6_mul(s_awthl1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7360
for (int mynn_element=0; mynn_element < (s_awqt1).size(); ++mynn_element) { s_awqt1.linear(mynn_element) = r6_mul(s_awqt1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7361
for (int mynn_element=0; mynn_element < (s_awqc1).size(); ++mynn_element) { s_awqc1.linear(mynn_element) = r6_mul(s_awqc1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7362
for (int mynn_element=0; mynn_element < (s_awqv1).size(); ++mynn_element) { s_awqv1.linear(mynn_element) = r6_mul(s_awqv1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7363
if (((numcon_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7364
for (int mynn_element=0; mynn_element < (s_awqnc1).size(); ++mynn_element) { s_awqnc1.linear(mynn_element) = r6_mul(s_awqnc1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7365
for (int mynn_element=0; mynn_element < (s_awqni1).size(); ++mynn_element) { s_awqni1.linear(mynn_element) = r6_mul(s_awqni1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7366
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7367
if (((aerosol_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7368
for (int mynn_element=0; mynn_element < (s_awqnwfa1).size(); ++mynn_element) { s_awqnwfa1.linear(mynn_element) = r6_mul(s_awqnwfa1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7369
for (int mynn_element=0; mynn_element < (s_awqnifa1).size(); ++mynn_element) { s_awqnifa1.linear(mynn_element) = r6_mul(s_awqnifa1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7370
for (int mynn_element=0; mynn_element < (s_awqnbca1).size(); ++mynn_element) { s_awqnbca1.linear(mynn_element) = r6_mul(s_awqnbca1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7371
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7372
if (((momentum_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7373
for (int mynn_element=0; mynn_element < (s_awu1).size(); ++mynn_element) { s_awu1.linear(mynn_element) = r6_mul(s_awu1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7374
for (int mynn_element=0; mynn_element < (s_awv1).size(); ++mynn_element) { s_awv1.linear(mynn_element) = r6_mul(s_awv1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7375
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7376
if (((tke_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7377
for (int mynn_element=0; mynn_element < (s_awqke1).size(); ++mynn_element) { s_awqke1.linear(mynn_element) = r6_mul(s_awqke1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7378
if (((bl_mynn_closure) > (2.5f))) { for (int mynn_element=0; mynn_element < (s_awqsq1).size(); ++mynn_element) { s_awqsq1.linear(mynn_element) = r6_mul(s_awqsq1.linear(mynn_element), adjustment); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7379
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7380
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7381
for (int mynn_element=0; mynn_element < (s_awchem1).size(); ++mynn_element) { s_awchem1.linear(mynn_element) = r6_mul(s_awchem1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7382
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7383
if (((scalar_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7384
for (int mynn_element=0; mynn_element < (s_awscalars1).size(); ++mynn_element) { s_awscalars1.linear(mynn_element) = r6_mul(s_awscalars1.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7385
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7386
for (int mynn_element=0; mynn_element < (upa).size(); ++mynn_element) { upa.linear(mynn_element) = r6_mul(upa.linear(mynn_element), adjustment); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7387
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7388
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7389
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7390
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7391
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7392
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7396
k = kts;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7397
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7398
edmf_a1(k) = r6_add(edmf_a1(k), upa(((k) + (1)), ip));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7399
edmf_w1(k) = r6_add(edmf_w1(k), r6_mul(r6_mul(upa(((k) + (1)), ip), p5), upw(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7400
edmf_qt1(k) = r6_add(edmf_qt1(k), r6_mul(upa(((k) + (1)), ip), upqt(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7401
edmf_thl1(k) = r6_add(edmf_thl1(k), r6_mul(upa(((k) + (1)), ip), upthl(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7402
edmf_ent1(k) = r6_add(edmf_ent1(k), r6_mul(upa(((k) + (1)), ip), ent(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7403
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_mul(upa(((k) + (1)), ip), upqc(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7404
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7405
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7406
{ const int mynn_lo16 = (((kts) + (1))), mynn_hi16 = (((ktop_plume(ip)) - (1))), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { k = mynn_lo16 + mynn_it16*mynn_st16;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7407
upak = r6_div(r6_add(r6_mul(upa(((k) + (1)), ip), dzi(k)), r6_mul(upa(k, ip), dzi(((k) + (1))))), r6_add(dzi(((k) + (1))), dzi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7408
edmf_a1(k) = r6_add(edmf_a1(k), upak);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7409
edmf_w1(k) = r6_add(edmf_w1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upw(((k) + (1)), ip), dzi(k)), r6_mul(upw(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7410
edmf_qt1(k) = r6_add(edmf_qt1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upqt(((k) + (1)), ip), dzi(k)), r6_mul(upqt(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7411
edmf_thl1(k) = r6_add(edmf_thl1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upthl(((k) + (1)), ip), dzi(k)), r6_mul(upthl(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7412
edmf_ent1(k) = r6_add(edmf_ent1(k), r6_div(r6_mul(upak, r6_add(r6_mul(ent(((k) + (1)), ip), dzi(k)), r6_mul(ent(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7413
if (((((mynn_abs(upqc(k, ip))) < (1e-10f))) && (((upqc(((k) + (1)), ip)) > (zero))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7415
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_mul(upak, upqc(((k) + (1)), ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7416
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7417
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upqc(((k) + (1)), ip), dzi(k)), r6_mul(upqc(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7418
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7419
} mynn_lane_sync(); k = mynn_lo16 + mynn_n16*mynn_st16; if (mynn_n16 > 0) { const int mynn_src16 = (mynn_n16 - 1) % lane_width; upak = mynn_lane_bcast(upak, mynn_src16); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7420
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7422
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7423
k = ktop_plume(ip);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7424
edmf_a1(k) = r6_add(edmf_a1(k), upa(k, ip));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7425
edmf_w1(k) = r6_add(edmf_w1(k), r6_mul(upa(k, ip), upw(k, ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7426
edmf_qt1(k) = r6_add(edmf_qt1(k), r6_mul(upa(k, ip), upqt(k, ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7427
edmf_thl1(k) = r6_add(edmf_thl1(k), r6_mul(upa(k, ip), upthl(k, ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7428
edmf_ent1(k) = r6_add(edmf_ent1(k), r6_mul(upa(k, ip), ent(k, ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7429
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_mul(upa(k, ip), upqc(k, ip)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7430
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7431
{ const int mynn_lo17 = (kts), mynn_hi17 = (((kte) - (1))), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { k = mynn_lo17 + mynn_it17*mynn_st17;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7433
if (((edmf_a1(k)) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7434
edmf_w1(k) = r6_div(edmf_w1(k), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7435
edmf_qt1(k) = r6_div(edmf_qt1(k), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7436
edmf_thl1(k) = r6_div(edmf_thl1(k), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7437
edmf_ent1(k) = r6_div(edmf_ent1(k), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7438
edmf_qc1(k) = r6_div(edmf_qc1(k), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7439
edmf_a1(k) = r6_mul(edmf_a1(k), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7440
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7441
} mynn_lane_sync(); k = mynn_lo17 + mynn_n17*mynn_st17; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7442
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7444
if (((r6_mul(edmf_a1(k), edmf_w1(k))) > (maxmf))) { maxmf = r6_mul(edmf_a1(k), edmf_w1(k)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7447
tkeprod_up(k) = r6_div(r6_mul(mynn_ipow<3>(mynn_abs(edmf_w1(k))), edmf_a1(k)), r6_mul(b1, mynn_max((float)(r6_mul(p5, r6_add(el1(k), el1(((k) + (1)))))), (float)(0.1f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7448
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7451
if (mix_chem) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7452
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7453
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7454
{ const int mynn_lo18 = (1), mynn_hi18 = (nchem), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { ic = mynn_lo18 + mynn_it18*mynn_st18;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7455
upak = r6_div(r6_add(r6_mul(upa(((k) + (1)), ip), dzi(k)), r6_mul(upa(k, ip), dzi(((k) + (1))))), r6_add(dzi(((k) + (1))), dzi(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7456
edmf_chem(k, ic) = r6_add(edmf_chem(k, ic), r6_div(r6_mul(upak, r6_add(r6_mul(upchem(((k) + (1)), ip, ic), dzi(k)), r6_mul(upchem(k, ip, ic), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7457
} mynn_lane_sync(); ic = mynn_lo18 + mynn_n18*mynn_st18; if (mynn_n18 > 0) { const int mynn_src18 = (mynn_n18 - 1) % lane_width; upak = mynn_lane_bcast(upak, mynn_src18); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7458
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7459
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7460
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7461
if (((edmf_a1(k)) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7462
{ const int mynn_lo19 = (1), mynn_hi19 = (nchem), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { ic = mynn_lo19 + mynn_it19*mynn_st19;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7463
edmf_chem(k, ic) = r6_div(edmf_chem(k, ic), edmf_a1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7464
} mynn_lane_sync(); ic = mynn_lo19 + mynn_n19*mynn_st19; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7465
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7466
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7467
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7471
if (env_subs) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7472
{ const int mynn_lo20 = (((kts) + (1))), mynn_hi20 = (((kte) - (1))), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { k = mynn_lo20 + mynn_it20*mynn_st20;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7477
envi_w(k) = r6_mul(p333, r6_add(r6_add(edmf_w1(((k) - (1))), edmf_w1(k)), edmf_w1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7478
envi_a(k) = r6_mul(r6_mul(p333, r6_add(r6_add(edmf_a1(((k) - (1))), edmf_a1(k)), edmf_a1(((k) + (1))))), adjustment);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7479
} mynn_lane_sync(); k = mynn_lo20 + mynn_n20*mynn_st20; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7481
envi_w(kts) = edmf_w1(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7482
envi_a(kts) = edmf_a1(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7484
envi_w(kte) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7485
envi_a(kte) = edmf_a1(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7487
envi_w(((kte) + (1))) = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7488
envi_a(((kte) + (1))) = edmf_a1(kte);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7492
if (((envi_w(kts)) > (r6_div(r6_mul(0.9f, dz1(kts)), dt)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7493
sublim = r6_div(r6_div(r6_mul(0.9f, dz1(kts)), dt), envi_w(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7494
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7495
sublim = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7496
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7498
{ const int mynn_lo21 = (kts), mynn_hi21 = (kte), mynn_st21 = (1); const int mynn_n21 = mynn_trip(mynn_lo21, mynn_hi21, mynn_st21); mynn_lane_sync(); for (int mynn_it21 = lane_sub; mynn_it21 < mynn_n21; mynn_it21 += lane_width) { k = mynn_lo21 + mynn_it21*mynn_st21;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7499
temp = envi_a(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7500
envi_a(k) = r6_sub(one, temp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7501
envi_w(k) = r6_div(r6_mul(r6_mul(r6_mul(csub, sublim), envi_w(k)), temp), r6_sub(1.f, temp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7502
} mynn_lane_sync(); k = mynn_lo21 + mynn_n21*mynn_st21; if (mynn_n21 > 0) { const int mynn_src21 = (mynn_n21 - 1) % lane_width; temp = mynn_lane_bcast(temp, mynn_src21); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7505
sub_thl1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), thl1(((kts) + (1)))), r6_mul(rho1(kts), thl1(kts)))), dzi(kts)), rhoi(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7507
sub_sqv1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), qv1(((kts) + (1)))), r6_mul(rho1(kts), qv1(kts)))), dzi(kts)), rhoi(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7509
{ const int mynn_lo22 = (((kts) + (1))), mynn_hi22 = (((kte) - (1))), mynn_st22 = (1); const int mynn_n22 = mynn_trip(mynn_lo22, mynn_hi22, mynn_st22); mynn_lane_sync(); for (int mynn_it22 = lane_sub; mynn_it22 < mynn_n22; mynn_it22 += lane_width) { k = mynn_lo22 + mynn_it22*mynn_st22;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7510
dzi(k) = r6_mul(0.5f, r6_add(dz1(k), dz1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7511
sub_thl1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), thl1(((k) + (1)))), r6_mul(rho1(k), thl1(k)))), dzi(k)), rhoi(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7513
sub_sqv1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), qv1(((k) + (1)))), r6_mul(rho1(k), qv1(k)))), dzi(k)), rhoi(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7515
} mynn_lane_sync(); k = mynn_lo22 + mynn_n22*mynn_st22; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7517
{ const int mynn_lo23 = (kts), mynn_hi23 = (((kte) - (1))), mynn_st23 = (1); const int mynn_n23 = mynn_trip(mynn_lo23, mynn_hi23, mynn_st23); mynn_lane_sync(); for (int mynn_it23 = lane_sub; mynn_it23 < mynn_n23; mynn_it23 += lane_width) { k = mynn_lo23 + mynn_it23*mynn_st23;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7518
det_thl1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_thl(k), thl1(k))), envi_a(k)), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7519
det_sqv1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_sqv(k), qv1(k))), envi_a(k)), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7520
det_sqc1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_sqc(k), qc1(k))), envi_a(k)), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7521
} mynn_lane_sync(); k = mynn_lo23 + mynn_n23*mynn_st23; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7523
if (((momentum_opt) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7524
sub_u1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), u1(((kts) + (1)))), r6_mul(rho1(kts), u1(kts)))), dzi(kts)), rhoi(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7526
sub_v1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), v1(((kts) + (1)))), r6_mul(rho1(kts), v1(kts)))), dzi(kts)), rhoi(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7528
{ const int mynn_lo24 = (((kts) + (1))), mynn_hi24 = (((kte) - (1))), mynn_st24 = (1); const int mynn_n24 = mynn_trip(mynn_lo24, mynn_hi24, mynn_st24); mynn_lane_sync(); for (int mynn_it24 = lane_sub; mynn_it24 < mynn_n24; mynn_it24 += lane_width) { k = mynn_lo24 + mynn_it24*mynn_st24;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7529
sub_u1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), u1(((k) + (1)))), r6_mul(rho1(k), u1(k)))), dzi(k)), rhoi(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7531
sub_v1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), v1(((k) + (1)))), r6_mul(rho1(k), v1(k)))), dzi(k)), rhoi(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7533
} mynn_lane_sync(); k = mynn_lo24 + mynn_n24*mynn_st24; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7535
{ const int mynn_lo25 = (kts), mynn_hi25 = (((kte) - (1))), mynn_st25 = (1); const int mynn_n25 = mynn_trip(mynn_lo25, mynn_hi25, mynn_st25); mynn_lane_sync(); for (int mynn_it25 = lane_sub; mynn_it25 < mynn_n25; mynn_it25 += lane_width) { k = mynn_lo25 + mynn_it25*mynn_st25;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7536
det_u1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_u(k), u1(k))), envi_a(k)), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7537
det_v1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_v(k), v1(k))), envi_a(k)), psig_w);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7538
} mynn_lane_sync(); k = mynn_lo25 + mynn_n25*mynn_st25; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7539
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7540
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7544
{ const int mynn_lo26 = (kts), mynn_hi26 = (((kte) - (1))), mynn_st26 = (1); const int mynn_n26 = mynn_trip(mynn_lo26, mynn_hi26, mynn_st26); mynn_lane_sync(); for (int mynn_it26 = lane_sub; mynn_it26 < mynn_n26; mynn_it26 += lane_width) { k = mynn_lo26 + mynn_it26*mynn_st26;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7545
edmf_th1(k) = r6_add(edmf_thl1(k), r6_mul(r6_div(xlvcp, ex1(k)), edmf_qc1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7546
qti(((k) + (1))) = r6_div(r6_add(r6_mul(qt1(k), dz1(((k) + (1)))), r6_mul(qt1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7547
} mynn_lane_sync(); k = mynn_lo26 + mynn_n26*mynn_st26; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7548
qti(kts) = qt1(kts);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7553
cb_check = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7554
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (2))) : k >= (((kte) - (2))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7555
if (((k) > (ktop))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7556
if (((edmf_qc1(k)) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7557
if (((cb_check) == (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7558
mf_at_cb = r6_mul(edmf_a1(k), edmf_w1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7559
cb_check = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7560
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7562
aup = edmf_a1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7563
thp = edmf_th1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7564
qtp = edmf_qt1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7565
qcp = edmf_qc1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7566
dqwdz = r6_div(r6_sub(qti(k), qti(((k) + (1)))), dz1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7570
esat = f_esat_blend(tk1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7572
qsl = r6_div(r6_mul(ep_2, esat), mynn_max((float)(r6_sub(pres1(k), r6_mul(ep_3, esat))), (float)(1.e-7f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7575
xl = f_xl_blend(tk1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7576
qsat_tk = f_qsat_blend(tk1(k), pres1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7578
rsl = r6_div(r6_mul(xl, qsat_tk), r6_mul(r_v, mynn_ipow<2>(tk1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7580
cpm = r6_add(cp, r6_mul(qt1(k), cpv));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7581
a = r6_div(one, r6_add(one, r6_div(r6_mul(xl, rsl), cpm)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7582
b9 = r6_mul(a, rsl);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7584
q2p = r6_div(xlvcp, ex1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7585
pt = r6_add(thl1(k), r6_mul(r6_mul(q2p, qcp), aup));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7586
bb = r6_div(r6_mul(b9, tk1(k)), pt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7592
qww = r6_add(one, r6_mul(p608, qt1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7593
alpha = r6_mul(p608, pt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7594
beta = r6_sub(r6_div(r6_mul(pt, xl), r6_mul(tk1(k), cp)), r6_mul(1.61f, pt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7598
if (((a) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7599
f = mynn_min((float)(r6_div(one, a)), (float)(four));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7600
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7601
f = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7602
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7612
tauc = 1800.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7613
sigq = r6_mul(r6_mul(r6_mul(five, aup), edmf_w1(k)), mynn_max((float)(r6_mul(tauc, mynn_max((float)(zero), (float)(dqwdz)))), (float)(4e-3f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7618
sigq = mynn_max((float)(sigq), (float)(r6_mul(r6_mul(three, aup), r6_sub(qtp, qt1(k)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7623
qmq = r6_sub(qt1(k), qsat_tk);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7624
if (((qmq) > (zero))) { sigq = mynn_min((float)(r6_mul(qsat_tk, 0.015f)), (float)(sigq)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7625
q1 = r6_div(qmq, sigq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7627
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7633
cffac = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(mf_at_cb, 0.05f)), (float)(zero)), 0.05f)), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7634
cf_mf = mynn_min((float)(mynn_max((float)(r6_add(p5, r6_mul(0.36f, mynn_atan(r6_mul(1.8f, r6_add(r6_add(q1, p2), cffac)))))), (float)(0.01f))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7635
cf_mf = mynn_max((float)(cf_mf), (float)(r6_mul(1.2f, aup)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7638
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7643
cf_mf = mynn_min((float)(mynn_max((float)(r6_add(p5, r6_mul(0.36f, mynn_atan(r6_mul(1.8f, r6_add(q1, p2)))))), (float)(0.01f))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7644
cf_mf = mynn_max((float)(cf_mf), (float)(r6_mul(1.8f, aup)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7646
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7659
qcfac = r6_add(1.20f, r6_mul(1.0f, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(mf_at_cb, 0.04f)), (float)(zero)), 0.05f)), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7660
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7666
qc_mf = r6_mul(qcfac, r6_mul(qcp, aup));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7667
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7668
if (((r6_mul(qcp, aup)) > (5e-5f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7669
qc_mf = r6_sub(r6_mul(1.86f, r6_mul(qcp, aup)), 2.2e-5f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7670
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7671
qc_mf = r6_mul(1.20f, r6_mul(qcp, aup));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7672
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7673
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7678
wt2 = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(cf_mf, thirty)), (float)(zero)), r6_sub(hundred, thirty))), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7679
qc_mf_min = r6_add(r6_mul(r6_mul(r6_mul(wt2, qsat_tk), 0.01f), cf_mf), r6_mul(r6_mul(r6_mul(r6_sub(one, wt2), qsat_tk), 0.025f), cf_mf));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7680
qc_mf = mynn_max((float)(qc_mf), (float)(qc_mf_min));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7682
cf_strat = cldfra_bl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7683
qc_strat = qc_bl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7684
cldfra_bl1(k) = mynn_max((float)(cf_mf), (float)(cf_strat));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7685
pct_mf = r6_div(cf_mf, cldfra_bl1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7686
qc_bl1(k) = r6_add(r6_mul(qc_mf, pct_mf), r6_mul(r6_sub(one, pct_mf), qc_strat));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7694
q1 = mynn_max((float)(q1), (float)(r6_neg(2.25f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7699
if (((q1) >= (one))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7700
fng = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7701
} else if (((((q1) >= (r6_neg(1.7f)))) && (((q1) < (one))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7702
fng = mynn_exp(r6_neg(r6_mul(p4, r6_sub(q1, one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7703
} else if (((((q1) >= (r6_neg(2.5f)))) && (((q1) < (r6_neg(1.7f)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7704
fng = r6_add(three, mynn_exp(r6_neg(r6_mul(3.8f, r6_add(q1, 1.7f)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7705
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7706
fng = mynn_min((float)(r6_add(23.9f, mynn_exp(r6_neg(r6_mul(1.6f, r6_add(q1, 2.5f)))))), (float)(60.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7707
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7710
vt1(k) = r6_sub(r6_sub(qww, r6_mul(r6_mul(r6_mul(r6_mul(1.5f, aup), beta), bb), fng)), one);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7711
vq1(k) = r6_sub(r6_add(alpha, r6_mul(r6_mul(r6_mul(r6_mul(1.5f, aup), beta), a), fng)), tv0);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7712
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7713
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7715
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7718
if (((ktop) > (0))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7719
maxqc = mynn_maxval(edmf_qc1.slice(MynnRange{1, ktop}));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7720
if (((maxqc) < (1.e-8f))) { maxmf = r6_neg(r6_mul(1.f, maxmf)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7721
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7726
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7754
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7755
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7756
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7757
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7758
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7760
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7772-7825
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7791
niter = 50;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7793
diff = 1.e-6f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7795
exn = mynn_pow(r6_div(p, p1000mb), rcp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7796
qc = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7797
for (ni=1; (1) > 0 ? ni <= (niter) : ni >= (niter); ni += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7798
t = r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7799
qs = f_qsat_blend(t, p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7800
qcold = qc;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7801
qc = r6_add(r6_mul(p5, qc), r6_mul(p5, mynn_max((float)(zero), (float)(r6_sub(qt, qs)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7802
if (((mynn_abs(r6_sub(qc, qcold))) < (diff))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7803
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7805
t = r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7806
qs = f_qsat_blend(t, p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7807
qc = mynn_max((float)(zero), (float)(r6_sub(qt, qs)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7810
if (((zagl) < (100.f))) { qc = zero; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7813
thv = r6_mul(r6_add(thl, r6_mul(r6_div(xlvcp, exn), qc)), r6_sub(r6_add(one, r6_mul(qt, r6_sub(rvovrd, one))), r6_mul(rvovrd, qc)));
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7828-7899
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7839
niter = 50;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7841
diff = 1.e-6f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7843
if (((r6_add(qi, qc)) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7844
frac_ice = r6_div(qi, r6_add(qi, qc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7845
frac_liq = r6_sub(one, frac_ice);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7846
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7847
frac_ice = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7848
frac_liq = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7849
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7851
if (((debug_dd) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7852
thvin = thv;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7853
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7854
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7855
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7857
exn = mynn_pow(r6_div(p, p1000mb), rcp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7858
for (ni=1; (1) > 0 ? ni <= (niter) : ni >= (niter); ni += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7859
t = r6_add(r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc)), r6_mul(r6_div(xlscp, exn), qi));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7860
qs = f_qsat_blend(t, p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7861
qxold = r6_add(qc, qi);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7862
qx = r6_add(qc, qi);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7863
qx = r6_add(r6_mul(p5, qx), r6_mul(p5, mynn_max((float)(zero), (float)(r6_sub(qt, qs)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7864
qc = r6_mul(frac_liq, qx);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7865
qi = r6_mul(frac_ice, qx);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7866
if (((mynn_abs(r6_sub(qx, qxold))) < (diff))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7867
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7869
t = r6_add(r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc)), r6_mul(r6_div(xlscp, exn), qi));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7870
qs = f_qsat_blend(t, p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7871
qx = mynn_max((float)(zero), (float)(r6_sub(qt, qs)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7872
qc = r6_mul(frac_liq, qx);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7873
qi = r6_mul(frac_ice, qx);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7878
thv = r6_mul(r6_add(r6_add(thl, r6_mul(r6_div(xlvcp, exn), qc)), r6_mul(r6_div(xlscp, exn), qi)), r6_sub(r6_add(one, r6_mul(p608, r6_sub(qt, qx))), qx));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7893
if (((debug_dd) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7894
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7895
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7896
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7897
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7903-7936
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7915
niter = 50;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7917
diff = 2.e-5f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7919
exn = mynn_pow(r6_div(p, p1000mb), rcp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7921
t = r6_mul(thv, exn);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7925
qc = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7927
for (ni=1; (1) > 0 ? ni <= (niter) : ni >= (niter); ni += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7928
qcold = qc;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7929
t = r6_div(r6_mul(exn, thv), r6_sub(r6_add(1.f, r6_mul(qt, r6_sub(rvovrd, 1.f))), r6_mul(rvovrd, qc)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7930
qs = f_qsat_blend(t, p);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7931
qc = mynn_max((float)(0.f), (float)(r6_sub(qt, qs)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7932
if (((mynn_abs(r6_sub(qc, qcold))) < (diff))) { break; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7933
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7934
thl = r6_div(r6_sub(t, r6_mul(r6_div(xlv, cp), qc)), exn);
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:7957-8498
__device__ __noinline__ void f_ddmp_mf(int kts, int kte, float dt, float dx, MynnArray<float, 1> zw, MynnArray<float, 1> dz, MynnArray<float, 1> p, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> th, MynnArray<float, 1> thl, MynnArray<float, 1> thv, MynnArray<float, 1> tk, MynnArray<float, 1> qt, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qnc, MynnArray<float, 1> qni, MynnArray<float, 1> qnwfa, MynnArray<float, 1> qnifa, MynnArray<float, 1> qke, MynnArray<float, 1> rho, MynnArray<float, 1> exner, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> qi_bl1, MynnArray<float, 1> cldfra_bl1, float ust, float flt, float flq, float fltv, float pblh, int kpbl, MynnArray<float, 1> edmf_a_dd, MynnArray<float, 1> edmf_w_dd, MynnArray<float, 1> edmf_qt_dd, MynnArray<float, 1> edmf_thl_dd, MynnArray<float, 1> edmf_ent_dd, MynnArray<float, 1> edmf_qc_dd, MynnArray<float, 1> sd_aw, MynnArray<float, 1> sd_awthl, MynnArray<float, 1> sd_awqt, MynnArray<float, 1> sd_awqv, MynnArray<float, 1> sd_awqc, MynnArray<float, 1> sd_awqi, MynnArray<float, 1> sd_awqnc, MynnArray<float, 1> sd_awqni, MynnArray<float, 1> sd_awqnwfa, MynnArray<float, 1> sd_awqnifa, MynnArray<float, 1> sd_awu, MynnArray<float, 1> sd_awv, MynnArray<float, 1> sd_awqke, MynnArray<float, 1> tkeprod_dn, MynnArray<float, 1> el, MynnArray<float, 1> rthraten, float psig, float& maxmf_dd, float& maxwidth_dd) {
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
constexpr float minddd = 100.f;
constexpr float maxddd = 500.f;
constexpr float zmin = 50.f;
constexpr float dz200 = 200.f;
constexpr float cooling_thresh = (-(0.00010f));
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
MynnArray<float, 1> massflux = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
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
float maxcooling;
float dthvx;
float tmp1;
float rcldb;
float ent_eff;
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
float dp;
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8056
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downw).size(); mynn_element += lane_width) { downw.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8057
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downthl).size(); mynn_element += lane_width) { downthl.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8058
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downthv).size(); mynn_element += lane_width) { downthv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8059
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqt).size(); mynn_element += lane_width) { downqt.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8060
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downa).size(); mynn_element += lane_width) { downa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8061
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downu).size(); mynn_element += lane_width) { downu.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8062
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downv).size(); mynn_element += lane_width) { downv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8063
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqc).size(); mynn_element += lane_width) { downqc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8064
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqi).size(); mynn_element += lane_width) { downqi.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8065
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqnc).size(); mynn_element += lane_width) { downqnc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8066
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqni).size(); mynn_element += lane_width) { downqni.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8067
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqnwfa).size(); mynn_element += lane_width) { downqnwfa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8068
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqnifa).size(); mynn_element += lane_width) { downqnifa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8069
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (ent).size(); mynn_element += lane_width) { ent.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8070
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dd_initk).size(); mynn_element += lane_width) { dd_initk.linear(mynn_element) = 0; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8072
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a_dd).size(); mynn_element += lane_width) { edmf_a_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8073
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w_dd).size(); mynn_element += lane_width) { edmf_w_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8074
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt_dd).size(); mynn_element += lane_width) { edmf_qt_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8075
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl_dd).size(); mynn_element += lane_width) { edmf_thl_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8076
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent_dd).size(); mynn_element += lane_width) { edmf_ent_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8077
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc_dd).size(); mynn_element += lane_width) { edmf_qc_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8079
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_aw).size(); mynn_element += lane_width) { sd_aw.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8080
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awthl).size(); mynn_element += lane_width) { sd_awthl.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8081
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqt).size(); mynn_element += lane_width) { sd_awqt.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8082
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqv).size(); mynn_element += lane_width) { sd_awqv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8083
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqc).size(); mynn_element += lane_width) { sd_awqc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8084
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqi).size(); mynn_element += lane_width) { sd_awqi.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8085
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnc).size(); mynn_element += lane_width) { sd_awqnc.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8086
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqni).size(); mynn_element += lane_width) { sd_awqni.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8087
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnwfa).size(); mynn_element += lane_width) { sd_awqnwfa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8088
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnifa).size(); mynn_element += lane_width) { sd_awqnifa.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8089
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awu).size(); mynn_element += lane_width) { sd_awu.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8090
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awv).size(); mynn_element += lane_width) { sd_awv.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8091
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqke).size(); mynn_element += lane_width) { sd_awqke.linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8094
cloudflg = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8095
singlelayer = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8096
qltop = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8097
qlbase = 1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8098
crate = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8099
for (k=mynn_max((int)(((kpbl) - (5))), (int)(kmin)); (1) > 0 ? k <= (mynn_min((int)(((kpbl) + (20))), (int)(((kte) - (1))))) : k >= (mynn_min((int)(((kpbl) + (20))), (int)(((kte) - (1))))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8100
maxcooling = mynn_minval(rthraten.slice(MynnRange{((k) - (1)), ((k) + (1))}));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8102
if (((((((cldfra_bl1(k)) > (p5))) && (((cldfra_bl1(((k) + (1)))) < (p5))))) && (((maxcooling) < (cooling_thresh))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8105
cloudflg = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8106
if (((maxcooling) < (crate))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8107
qltop = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8108
cftop = cldfra_bl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8109
crate = maxcooling;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8110
if (((cldfra_bl1(((k) - (1)))) < (p2))) { singlelayer = true; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8111
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8112
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8113
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8116
maxdd2 = mynn_min((float)(r6_mul(1.2f, dx)), (float)(maxddd));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8117
maxdd2 = mynn_min((float)(maxdd2), (float)(r6_add(r6_mul(r6_neg(r6_mul(100.f, crate)), 3600.f), 200.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8118
maxdd2 = mynn_min((float)(maxdd2), (float)(r6_add(80.f, r6_mul(p666, zw(((qltop) + (1)))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8119
maxwidth_dd = maxdd2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8120
dl = r6_div(mynn_max((float)(zero), (float)(r6_sub(maxdd2, minddd))), ((float)(((ndd) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8123
if (((cloudflg) && (((dl) > (zero))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8124
for (k=qltop; ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8125
qx_k = mynn_max((float)(qt(k)), (float)(r6_add(qc_bl1(k), qi_bl1(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8126
if (((qx_k) > (1e-6f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8127
qlbase = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8128
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8129
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8131
{ const int mynn_lo0 = (1), mynn_hi0 = (ndd), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { i = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8133
dd_initk(i) = qltop;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8134
} mynn_lane_sync(); i = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8137
f0 = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8138
for (k=mynn_max((int)(((qltop) - (2))), (int)(kmin)); (1) > 0 ? k <= (((qltop) + (1))) : k >= (((qltop) + (1))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8139
radflux = r6_mul(rthraten(k), exner(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8140
dp = r6_mul(p5, r6_add(r6_sub(p(k), p(((k) + (1)))), r6_sub(p(((k) - (1))), p(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8141
radflux = r6_mul(r6_div(r6_mul(radflux, cp), grav), dp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8142
if (((radflux) < (zero))) { f0 = r6_add(mynn_abs(radflux), f0); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8143
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8144
f0 = mynn_min((float)(mynn_max((float)(f0), (float)(one))), (float)(200.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8151
add = mynn_min((float)(r6_mul(f0, 0.003f)), (float)(0.3f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8153
wspd_pbl = mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(u(kts)), mynn_ipow<2>(v(kts)))), (float)(0.01f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8154
ac_wsp = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(wspd_pbl, 10.f)), (float)(zero)), 15.f)), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8155
add = r6_mul(add, mynn_min((float)(ac_wsp), (float)(psig)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8158
dz_ent = r6_mul(p5, r6_add(dz(((qltop) + (1))), dz(qltop)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8159
jump_thv = r6_mul(r6_div(r6_sub(thv(((qltop) + (1))), thv(qltop)), dz_ent), dz200);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8160
jump_qt = r6_mul(r6_div(r6_sub(qt(((qltop) + (1))), qt(qltop)), dz_ent), dz200);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8161
jump_thetal = r6_mul(r6_div(r6_sub(thl(((qltop) + (1))), thl(qltop)), dz_ent), dz200);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8163
ki = qltop;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8164
if (singlelayer) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8165
refthl = thl(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8166
refthlv = r6_mul(refthl, r6_add(1.f, r6_mul(p608, qv(ki))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8167
refthv = thv(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8168
refthvm1 = thv(((ki) - (1)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8169
reftk = tk(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8170
refqt = qt(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8171
refqc = qc(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8172
refqi = qi(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8173
refqnc = qnc(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8174
refqni = qni(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8175
refqnwfa = qnwfa(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8176
refqnifa = qnifa(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8177
refu = u(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8178
refv = v(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8179
refqke = qke(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8180
refp = p(ki);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8181
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8182
refthl = r6_div(r6_add(r6_mul(thl(((ki) - (1))), dz(ki)), r6_mul(thl(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8183
refthlv = r6_mul(refthl, r6_add(1.f, r6_div(r6_mul(p608, r6_add(r6_mul(qv(((ki) - (1))), dz(ki)), r6_mul(qv(ki), dz(((ki) - (1)))))), r6_add(dz(ki), dz(((ki) - (1)))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8184
refthv = r6_div(r6_add(r6_mul(thv(((ki) - (1))), dz(ki)), r6_mul(thv(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8185
refthvm1 = r6_div(r6_add(r6_mul(thv(((ki) - (2))), dz(((ki) - (1)))), r6_mul(thv(((ki) - (1))), dz(((ki) - (2))))), r6_add(dz(((ki) - (1))), dz(((ki) - (2)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8186
reftk = r6_div(r6_add(r6_mul(tk(((ki) - (1))), dz(ki)), r6_mul(tk(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8187
refqt = r6_div(r6_add(r6_mul(qt(((ki) - (1))), dz(ki)), r6_mul(qt(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8188
refqc = r6_div(r6_add(r6_mul(qc(((ki) - (1))), dz(ki)), r6_mul(qc(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8189
refqi = r6_div(r6_add(r6_mul(qi(((ki) - (1))), dz(ki)), r6_mul(qi(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8190
refqnc = r6_div(r6_add(r6_mul(qnc(((ki) - (1))), dz(ki)), r6_mul(qnc(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8191
refqni = r6_div(r6_add(r6_mul(qni(((ki) - (1))), dz(ki)), r6_mul(qni(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8192
refqnwfa = r6_div(r6_add(r6_mul(qnwfa(((ki) - (1))), dz(ki)), r6_mul(qnwfa(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8193
refqnifa = r6_div(r6_add(r6_mul(qnifa(((ki) - (1))), dz(ki)), r6_mul(qnifa(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8194
refu = r6_div(r6_add(r6_mul(u(((ki) - (1))), dz(ki)), r6_mul(u(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8195
refv = r6_div(r6_add(r6_mul(v(((ki) - (1))), dz(ki)), r6_mul(v(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8196
refqke = r6_div(r6_add(r6_mul(qke(((ki) - (1))), dz(ki)), r6_mul(qke(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8197
refp = r6_div(r6_add(r6_mul(p(((ki) - (1))), dz(ki)), r6_mul(p(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8198
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8202
wstar_rad = r6_mul(1.25f, mynn_pow(r6_div(r6_mul(r6_mul(grav, dz200), f0), r6_mul(r6_mul(refthl, rho(ki)), cp)), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8203
wstar_rad = mynn_min((float)(mynn_max((float)(wstar_rad), (float)(p1))), (float)(1.5f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8206
dthvx = r6_sub(r6_add(thl(((ki) + (2))), r6_mul(r6_mul(th(((ki) + (2))), p608), qt(((ki) + (2))))), r6_add(thl(ki), r6_mul(r6_mul(th(ki), p608), qt(ki))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8208
dthvx = r6_div(r6_mul(dthvx, dz200), dz_ent);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8209
dthvx = mynn_max((float)(dthvx), (float)(p1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8210
rcldb = mynn_max((float)(r6_add(refqc, refqi)), (float)(r6_add(qc_bl1(ki), qi_bl1(ki))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8211
tmp1 = r6_div(r6_mul(xlvcp, rcldb), r6_mul(exner(ki), dthvx));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8214
ent_eff = mynn_max((float)(zero), (float)(mynn_min((float)(r6_add(p2, r6_mul(r6_mul(p2, eight), tmp1))), (float)(one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8216
went = r6_mul(r6_mul(p333, ent_eff), wstar_rad);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8217
qstar = r6_div(r6_mul(went, jump_qt), wstar_rad);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8218
thstar = r6_sub(r6_div(r6_div(r6_neg(r6_div(f0, rho(ki))), cp), wstar_rad), r6_div(r6_mul(went, jump_thv), wstar_rad));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8220
sigmaw = r6_mul(0.2f, wstar_rad);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8221
sigmaqt = r6_mul(50.f, mynn_abs(qstar));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8222
sigmath = r6_mul(1.5f, mynn_abs(thstar));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8224
pwmin = r6_neg(1.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8225
pwmax = r6_neg(3.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8226
wmin = mynn_max((float)(mynn_min((float)(r6_mul(sigmaw, pwmin)), (float)(r6_neg(0.1f)))), (float)(r6_neg(0.2f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8227
wmax = mynn_max((float)(mynn_min((float)(r6_mul(sigmaw, pwmax)), (float)(r6_neg(0.2f)))), (float)(r6_neg(0.5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8230
minexc = mynn_min((float)(r6_neg(0.06f)), (float)(r6_mul(0.5f, r6_sub(refthvm1, refthv))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8231
maxexc = mynn_min((float)(r6_sub(minexc, 0.1f)), (float)(r6_neg(0.6f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8232
maxexc = mynn_max((float)(maxexc), (float)(r6_sub(thv(1), refthlv)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8233
def_th = mynn_max((float)(mynn_min((float)(r6_div(r6_mul(r6_mul(0.05f, r6_neg(0.3f)), sigmath), sigmaw)), (float)(minexc))), (float)(maxexc));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8236
qs1 = f_qsat_blend(reftk, refp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8237
qs2 = f_qsat_blend(r6_add(reftk, def_th), refp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8238
refqt2 = r6_div(r6_mul(refqt, qs2), qs1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8240
def_qt = mynn_min((float)(r6_sub(qs2, qs1)), (float)(r6_neg(r6_mul(0.33f, r6_add(refqc, refqi)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8243
if (((r6_add(refqc, refqi)) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8244
frac_liq = r6_div(refqc, r6_add(refqc, refqi));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8245
frac_ice = r6_sub(one, frac_liq);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8246
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8247
frac_liq = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8248
frac_ice = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8249
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8251
if (((debug_dd) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8252
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8253
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8254
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8255
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8256
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8257
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8258
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8259
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8260
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8261
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8265
{ const int mynn_lo1 = (1), mynn_hi1 = (ndd), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { i = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8266
ki = dd_initk(i);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8269
downw(ki, i) = r6_sub(wmin, r6_mul(r6_div(r6_sub(mynn_abs(wmax), mynn_abs(wmin)), ((float)(((ndd) - (1))))), ((i) - (1))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8273
downa(ki, i) = r6_div(add, ((float)(ndd)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8274
downu(ki, i) = refu;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8275
downv(ki, i) = refv;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8276
downqc(ki, i) = mynn_max((float)(zero), (float)(r6_add(refqc, r6_mul(def_qt, frac_liq))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8277
downqi(ki, i) = mynn_max((float)(zero), (float)(r6_add(refqi, r6_mul(def_qt, frac_ice))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8278
downqnc(ki, i) = refqnc;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8279
downqni(ki, i) = refqni;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8280
downqnwfa(ki, i) = refqnwfa;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8281
downqnifa(ki, i) = refqnifa;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8282
downqt(ki, i) = r6_add(refqt, def_qt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8283
downthv(ki, i) = r6_add(refthv, r6_div(r6_mul(def_th, ((float)(i))), ((float)(ndd))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8284
downthl(ki, i) = r6_add(refthl, r6_div(r6_mul(def_th, ((float)(i))), ((float)(ndd))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8291
} mynn_lane_sync(); i = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; ki = mynn_lane_bcast(ki, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8293
if (((debug_dd) == (1))) {
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
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8302
for (i=1; (1) > 0 ? i <= (ndd) : i >= (ndd); i += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8303
ddd = r6_add(minddd, r6_mul(dl, ((float)(((i) - (1))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8305
thvn = downthv(ki, i);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8306
qcn = downqc(ki, i);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8307
qin = downqi(ki, i);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8308
for (k=((dd_initk(i)) - (1)); ((-(1))) > 0 ? k <= (((kts) + (1))) : k >= (((kts) + (1))); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8310
wmin = r6_add(p1, r6_mul(ddd, 0.0005f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8314
ent(k, i) = r6_div(0.18f, r6_mul(mynn_min((float)(mynn_max((float)(mynn_abs(downw(((k) + (1)), i))), (float)(wmin))), (float)(0.6f)), ddd));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8317
ent(k, i) = mynn_max((float)(ent(k, i)), (float)(0.0003f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8318
ent(k, i) = mynn_max((float)(ent(k, i)), (float)(r6_div(0.06f, zw(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8319
ent(k, i) = mynn_min((float)(ent(k, i)), (float)(r6_div(0.9f, dz(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8324
entexp = r6_mul(ent(k, i), dz(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8325
entexm = r6_mul(r6_mul(ent(k, i), dz(k)), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8327
qtn = r6_add(r6_mul(downqt(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qt(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8328
thln = r6_add(r6_mul(downthl(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(thl(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8329
qncn = r6_add(r6_mul(downqnc(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qnc(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8330
qnin = r6_add(r6_mul(downqni(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qni(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8331
qnwfan = r6_add(r6_mul(downqnwfa(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qnwfa(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8332
qnifan = r6_add(r6_mul(downqnifa(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qnifa(k), entexp));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8333
un = r6_add(r6_mul(downu(((k) + (1)), i), r6_sub(one, entexm)), r6_mul(u(k), entexm));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8334
vn = r6_add(r6_mul(downv(((k) + (1)), i), r6_sub(one, entexm)), r6_mul(v(k), entexm));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8342
pk = r6_div(r6_add(r6_mul(p(((k) - (1))), dz(k)), r6_mul(p(k), dz(((k) - (1))))), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8343
f_condensation_ddmf(qtn, thln, pk, zw(k), thvn, qcn, qin, debug_dd);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8346
thvk = r6_div(r6_add(r6_mul(thv(((k) - (1))), dz(k)), r6_mul(thv(k), dz(((k) - (1))))), r6_add(dz(k), dz(((k) - (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8347
buoy = r6_mul(grav, r6_sub(r6_div(thvn, thvk), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8349
if (((buoy) > (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8350
bcoeff = 0.2f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8351
ecoeff = r6_neg(2.0f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8352
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8354
bcoeff = r6_mul(0.2f, r6_sub(one, mynn_exp(r6_neg(r6_div(zw(k), r6_mul(0.15f, pblh))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8355
ecoeff = r6_sub(r6_neg(2.0f), mynn_exp(r6_neg(r6_div(zw(k), r6_mul(0.15f, pblh)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8356
buoy = r6_mul(buoy, r6_sub(one, mynn_exp(r6_neg(r6_div(zw(k), r6_mul(0.15f, pblh))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8357
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8358
if (((k) == (((kts) + (2))))) { buoy = mynn_max((float)(buoy), (float)(0.01f)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8359
if (((k) == (((kts) + (1))))) { buoy = mynn_max((float)(buoy), (float)(0.05f)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8371
mindownw = mynn_min((float)(downw(((k) + (1)), i)), (float)(r6_neg(0.2f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8372
wn = r6_add(downw(((k) + (1)), i), r6_mul(r6_sub(r6_mul(r6_mul(ecoeff, ent(k, i)), downw(((k) + (1)), i)), r6_div(r6_mul(bcoeff, buoy), mindownw)), mynn_min((float)(dz(k)), (float)(250.f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8377
if (((wn) < (r6_sub(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8378
wn = r6_sub(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8379
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8381
if (((wn) > (r6_add(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8382
wn = r6_add(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8383
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8384
wn = mynn_max((float)(mynn_min((float)(wn), (float)(zero))), (float)(r6_neg(2.0f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8386
if (((debug_dd) == (1))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8387
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8388
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8389
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8390
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8391
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8392
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8393
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8394
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8395
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8397
if (((wn) < (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8398
downw(k, i) = wn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8399
downthv(k, i) = thvn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8400
downthl(k, i) = thln;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8401
downqt(k, i) = qtn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8402
downqc(k, i) = qcn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8403
downqi(k, i) = qin;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8404
downqnc(k, i) = qncn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8405
downqni(k, i) = qnin;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8406
downqnwfa(k, i) = qnwfan;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8407
downqnifa(k, i) = qnifan;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8408
downu(k, i) = un;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8409
downv(k, i) = vn;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8410
downa(k, i) = downa(((k) + (1)), i);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8411
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8422
break;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8423
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8424
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8425
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8427
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downw.slice(1, MynnRange{downw.lo[1], downw.hi[1]})).size(); mynn_element += lane_width) { downw.slice(1, MynnRange{downw.lo[1], downw.hi[1]}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8428
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downa.slice(1, MynnRange{downa.lo[1], downa.hi[1]})).size(); mynn_element += lane_width) { downa.slice(1, MynnRange{downa.lo[1], downa.hi[1]}).linear(mynn_element) = zero; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8433
{ const int mynn_lo2 = (qltop), mynn_hi2 = (kts), mynn_st2 = ((-(1))); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8434
for (i=1; (1) > 0 ? i <= (ndd) : i >= (ndd); i += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8435
edmf_a_dd(k) = r6_add(edmf_a_dd(k), downa(k, i));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8436
edmf_w_dd(k) = r6_add(edmf_w_dd(k), r6_mul(downa(k, i), downw(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8437
edmf_qt_dd(k) = r6_add(edmf_qt_dd(k), r6_mul(downa(k, i), downqt(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8438
edmf_thl_dd(k) = r6_add(edmf_thl_dd(k), r6_mul(downa(k, i), downthl(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8439
edmf_ent_dd(k) = r6_add(edmf_ent_dd(k), r6_mul(downa(k, i), ent(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8440
edmf_qc_dd(k) = r6_add(edmf_qc_dd(k), r6_mul(downa(k, i), downqc(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8441
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8443
if (((edmf_a_dd(k)) > (0.f))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8444
edmf_w_dd(k) = r6_div(edmf_w_dd(k), edmf_a_dd(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8445
edmf_qt_dd(k) = r6_div(edmf_qt_dd(k), edmf_a_dd(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8446
edmf_thl_dd(k) = r6_div(edmf_thl_dd(k), edmf_a_dd(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8447
edmf_ent_dd(k) = r6_div(edmf_ent_dd(k), edmf_a_dd(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8448
edmf_qc_dd(k) = r6_div(edmf_qc_dd(k), edmf_a_dd(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8449
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8451
tkeprod_dn(k) = r6_div(r6_mul(mynn_ipow<3>(mynn_abs(edmf_w_dd(k))), edmf_a_dd(k)), r6_mul(b1, mynn_max((float)(el(k)), (float)(p5))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8452
tkeprod_dn(k) = mynn_min((float)(tkeprod_dn(k)), (float)(0.0004f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8453
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; i = mynn_lane_bcast(i, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8456
tkeprod_dn(((qltop) + (1))) = r6_div(r6_mul(mynn_ipow<3>(went), edmf_a_dd(qltop)), r6_mul(b1, mynn_max((float)(el(((qltop) + (1)))), (float)(p5))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8457
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (massflux).size(); mynn_element += lane_width) { massflux.linear(mynn_element) = r6_mul(edmf_a_dd.linear(mynn_element), edmf_w_dd.linear(mynn_element)); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8458
maxmf_dd = mynn_abs(mynn_minval(massflux));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8460
if (((debug_dd) == (2))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8461
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8462
if (((((tkeprod_dn(k)) > (0.0004f))) || (((edmf_w_dd(k)) < (r6_neg(two)))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8463
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8464
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8465
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8466
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8467
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8468
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8469
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8470
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8471
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8472
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8473
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8474
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8475
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8478
{ const int mynn_lo3 = (kts), mynn_hi3 = (qltop), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8479
rho_int = r6_div(r6_add(r6_mul(rho(k), dz(((k) + (1)))), r6_mul(rho(((k) + (1))), dz(k))), r6_add(dz(((k) + (1))), dz(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8480
for (i=1; (1) > 0 ? i <= (ndd) : i >= (ndd); i += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8481
sd_aw(k) = r6_add(sd_aw(k), r6_mul(r6_mul(rho_int, downa(k, i)), mynn_abs(downw(k, i))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8482
sd_awthl(k) = r6_add(sd_awthl(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downthl(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8483
sd_awqt(k) = r6_add(sd_awqt(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqt(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8484
sd_awqc(k) = r6_add(sd_awqc(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqc(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8485
sd_awqi(k) = r6_add(sd_awqi(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqi(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8486
sd_awqnc(k) = r6_add(sd_awqnc(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqnc(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8487
sd_awqni(k) = r6_add(sd_awqni(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqni(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8488
sd_awqnwfa(k) = r6_add(sd_awqnwfa(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqnwfa(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8489
sd_awqnifa(k) = r6_add(sd_awqnifa(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqnifa(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8490
sd_awu(k) = r6_add(sd_awu(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downu(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8491
sd_awv(k) = r6_add(sd_awv(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downv(k, i)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8492
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8493
sd_awqv(k) = r6_sub(r6_sub(sd_awqt(k), sd_awqc(k)), sd_awqi(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8494
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; i = mynn_lane_bcast(i, mynn_src3); rho_int = mynn_lane_bcast(rho_int, mynn_src3); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8496
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8501-8572
__device__ __noinline__ void f_scale_aware(float dx, float pblh, float& psig_bl, float& psig_shcu) {
MynnFrame mynn_frame(arena);
float dxdh;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8516
psig_bl = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8517
psig_shcu = one;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8518
dxdh = r6_div(mynn_max((float)(r6_mul(2.5f, dx)), (float)(10.f)), mynn_min((float)(pblh), (float)(3000.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8526
psig_bl = r6_div(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.106f, mynn_pow(dxdh, 0.667f))), r6_add(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.066f, mynn_pow(dxdh, 0.667f))), 0.071f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8529
dxdh = r6_div(mynn_max((float)(r6_mul(2.5f, dx)), (float)(10.f)), mynn_min((float)(r6_add(pblh, 500.f)), (float)(3500.f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8556
psig_shcu = r6_div(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.145f, mynn_pow(dxdh, 0.667f))), r6_add(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.172f, mynn_pow(dxdh, 0.667f))), 0.170f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8566
if (((psig_bl) > (one))) { psig_bl = one; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8567
if (((psig_bl) < (zero))) { psig_bl = zero; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8569
if (((psig_shcu) > (one))) { psig_shcu = one; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8570
if (((psig_shcu) < (zero))) { psig_shcu = zero; }
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8582-8625
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8609
xc = mynn_max((float)(r6_neg(80.f)), (float)(r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8614
if (((t) >= (r6_sub(t0c, 6.f)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8615
esat_blend = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8616
} else if (((t) <= (tice))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8617
esat_blend = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8618
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8619
esl = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8620
esi = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8621
chi = r6_div(r6_sub(r6_sub(t0c, 6.f), t), r6_sub(r6_sub(t0c, 6.f), tice));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8622
esat_blend = r6_add(r6_mul(r6_sub(1.f, chi), esl), r6_mul(chi, esi));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8623
}
return esat_blend;
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8633-8682
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8660
xc = mynn_max((float)(r6_neg(80.f)), (float)(r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8662
if (((t) >= (r6_sub(t0c, 6.f)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8663
esl = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8664
esl = mynn_min((float)(esl), (float)(r6_mul(p, 0.15f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8665
qsat_blend = r6_div(r6_mul(0.622f, esl), mynn_max((float)(r6_sub(p, esl)), (float)(1e-5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8666
} else if (((t) <= (tice))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8667
esi = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8668
esi = mynn_min((float)(esi), (float)(r6_mul(p, 0.15f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8669
qsat_blend = r6_div(r6_mul(0.622f, esi), mynn_max((float)(r6_sub(p, esi)), (float)(1e-5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8670
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8671
esl = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8672
esl = mynn_min((float)(esl), (float)(r6_mul(p, 0.15f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8673
esi = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8674
esi = mynn_min((float)(esi), (float)(r6_mul(p, 0.15f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8675
rslf = r6_div(r6_mul(0.622f, esl), mynn_max((float)(r6_sub(p, esl)), (float)(1e-5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8676
rsif = r6_div(r6_mul(0.622f, esi), mynn_max((float)(r6_sub(p, esi)), (float)(1e-5f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8678
chi = r6_div(r6_sub(r6_sub(t0c, 6.f), t), r6_sub(r6_sub(t0c, 6.f), tice));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8679
qsat_blend = r6_add(r6_mul(r6_sub(1.f, chi), rslf), r6_mul(chi, rsif));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8680
}
return qsat_blend;
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8691-8711
__device__ __noinline__ float f_xl_blend(float t) {
MynnFrame mynn_frame(arena);
float xl_blend;
float xlvt;
float xlst;
float chi;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8699
if (((t) >= (t0c))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8700
xl_blend = r6_add(xlv, r6_mul(r6_sub(cpv, cliq), r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8701
} else if (((t) <= (tice))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8702
xl_blend = r6_add(xls, r6_mul(r6_sub(cpv, cice), r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8703
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8704
xlvt = r6_add(xlv, r6_mul(r6_sub(cpv, cliq), r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8705
xlst = r6_add(xls, r6_mul(r6_sub(cpv, cice), r6_sub(t, t0c)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8707
chi = r6_div(r6_sub(t0c, t), r6_sub(t0c, tice));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8708
xl_blend = r6_add(r6_mul(r6_sub(1.f, chi), xlvt), r6_mul(chi, xlst));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8709
}
return xl_blend;
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8715-8764
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8731
if (((zet) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8732
dummy_0 = r6_add(one, mynn_pow(zet, bm_st));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8733
dummy_1 = r6_add(zet, mynn_pow(dummy_0, rbm_st));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8734
dummy_11 = r6_add(one, r6_mul(mynn_pow(dummy_0, r6_sub(rbm_st, 1)), mynn_pow(zet, r6_sub(bm_st, one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8735
dummy_2 = r6_mul(r6_neg(r6_div(am_st, dummy_1)), dummy_11);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8736
phi_m = r6_sub(one, r6_mul(zet, dummy_2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8737
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8738
dummy_0 = mynn_pow(r6_sub(one, r6_mul(cphm_unst, zet)), 0.25f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8739
phi_m = r6_div(one, dummy_0);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8740
dummy_psi = r6_add(r6_sub(r6_add(r6_mul(two, mynn_log(r6_mul(p5, r6_add(one, dummy_0)))), mynn_log(r6_mul(p5, r6_add(one, mynn_ipow<2>(dummy_0))))), r6_mul(two, mynn_atan(dummy_0))), 1.570796f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8742
dummy_0 = r6_sub(one, r6_mul(am_unst, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8743
dummy_1 = mynn_pow(dummy_0, p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8744
dummy_11 = r6_mul(r6_neg(r6_mul(p333, am_unst)), mynn_pow(dummy_0, r6_neg(p666)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8745
dummy_2 = r6_mul(p333, r6_add(r6_add(mynn_ipow<2>(dummy_1), dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8746
dummy_22 = r6_mul(r6_mul(p333, dummy_11), r6_add(r6_mul(two, dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8747
dummy_3 = r6_mul(0.57735f, r6_add(r6_mul(two, dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8748
dummy_33 = r6_mul(1.1547f, dummy_11);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8749
dummy_4 = r6_add(r6_sub(r6_mul(1.5f, mynn_log(dummy_2)), r6_mul(1.73205f, mynn_atan(dummy_3))), 1.813799364f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8750
dummy_44 = r6_sub(r6_mul(r6_div(1.5f, dummy_2), dummy_22), r6_div(r6_mul(1.73205f, dummy_33), r6_add(one, mynn_ipow<2>(dummy_3))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8752
dummy_0 = mynn_ipow<2>(zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8753
dummy_1 = r6_div(one, r6_add(one, dummy_0));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8754
dummy_11 = r6_mul(two, zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8755
dummy_2 = r6_mul(r6_add(r6_add(r6_div(r6_sub(one, phi_m), zet), r6_mul(dummy_11, dummy_4)), r6_mul(dummy_0, dummy_44)), dummy_1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8756
dummy_22 = r6_mul(r6_neg(r6_mul(dummy_11, r6_add(dummy_psi, r6_mul(dummy_0, dummy_4)))), mynn_ipow<2>(dummy_1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8758
phi_m = r6_sub(one, r6_mul(zet, r6_add(dummy_2, dummy_22)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8759
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8762
phim = phi_m;
return phim;
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8767-8813
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8783
if (((zet) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8784
dummy_0 = r6_add(one, mynn_pow(zet, bh_st));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8785
dummy_1 = r6_add(zet, mynn_pow(dummy_0, rbh_st));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8786
dummy_11 = r6_add(one, r6_mul(mynn_pow(dummy_0, r6_sub(rbh_st, one)), mynn_pow(zet, r6_sub(bh_st, one))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8787
dummy_2 = r6_mul(r6_neg(r6_div(ah_st, dummy_1)), dummy_11);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8788
phih = r6_sub(one, r6_mul(zet, dummy_2));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8789
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8790
dummy_0 = mynn_pow(r6_sub(one, r6_mul(cphh_unst, zet)), p5);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8791
phh = r6_div(one, dummy_0);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8792
dummy_psi = r6_mul(two, mynn_log(r6_mul(p5, r6_add(one, dummy_0))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8794
dummy_0 = r6_sub(one, r6_mul(ah_unst, zet));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8795
dummy_1 = mynn_pow(dummy_0, p333);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8796
dummy_11 = r6_mul(r6_neg(r6_mul(p333, ah_unst)), mynn_pow(dummy_0, r6_neg(p666)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8797
dummy_2 = r6_mul(p333, r6_add(r6_add(mynn_ipow<2>(dummy_1), dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8798
dummy_22 = r6_mul(r6_mul(p333, dummy_11), r6_add(r6_mul(two, dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8799
dummy_3 = r6_mul(0.57735f, r6_add(r6_mul(two, dummy_1), one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8800
dummy_33 = r6_mul(1.1547f, dummy_11);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8801
dummy_4 = r6_add(r6_sub(r6_mul(1.5f, mynn_log(dummy_2)), r6_mul(1.73205f, mynn_atan(dummy_3))), 1.813799364f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8802
dummy_44 = r6_sub(r6_mul(r6_div(1.5f, dummy_2), dummy_22), r6_div(r6_mul(1.73205f, dummy_33), r6_add(one, mynn_ipow<2>(dummy_3))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8804
dummy_0 = mynn_ipow<2>(zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8805
dummy_1 = r6_div(one, r6_add(one, dummy_0));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8806
dummy_11 = r6_mul(two, zet);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8807
dummy_2 = r6_mul(r6_add(r6_add(r6_div(r6_sub(one, phh), zet), r6_mul(dummy_11, dummy_4)), r6_mul(dummy_0, dummy_44)), dummy_1);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8808
dummy_22 = r6_mul(r6_neg(r6_mul(dummy_11, r6_add(dummy_psi, r6_mul(dummy_0, dummy_4)))), mynn_ipow<2>(dummy_1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8810
phih = r6_sub(one, r6_mul(zet, r6_add(dummy_2, dummy_22)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8811
}
return phih;
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8815-8976
__device__ __noinline__ void f_topdown_cloudrad(int kts, int kte, MynnArray<float, 1> dz1, MynnArray<float, 1> zw, float fltv, float u1, float v1, float xland, int kpbl, float pblh, MynnArray<float, 1> sqc, MynnArray<float, 1> sqi, MynnArray<float, 1> sqw, MynnArray<float, 1> thl, MynnArray<float, 1> th1, MynnArray<float, 1> ex1, MynnArray<float, 1> pres1, MynnArray<float, 1> rho1, MynnArray<float, 1> thv, MynnArray<float, 1> cldfra_bl1, MynnArray<float, 1> qc_bl, MynnArray<float, 1> qi_bl, MynnArray<float, 1> rthraten, MynnArray<float, 1> tkeprod_dn, float psig, float& maxtkeprod, float& cldtop_cooling, float& ent_eff) {
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
qc_bl.rebase({kts});
qi_bl.rebase({kts});
rthraten.rebase({kts});
tkeprod_dn.rebase({kts});
constexpr float dz500 = 500.f;
constexpr float dz200 = 200.f;
constexpr float cooling_thresh = (-(0.000069f));
MynnArray<float, 1> zfac = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> zfacent = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> tkeprodorig = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
float dthvx;
float tmp1;
float zfacent_max;
float zagl;
float temps;
float templ;
float zl1;
float zb1;
float dz_ent;
float wstar_rad;
float maxcooling;
float f0;
float radflux;
float went;
float rcldb;
float rvls;
float minrad;
float zminrad;
float dp;
float wspd_pbl;
float ac_wsp;
constexpr float pfac = 2.0f;
constexpr float zfmin = 0.01f;
constexpr float phifac = 8.0f;
int k;
int kk;
int kminrad;
bool cloudflg;
constexpr bool debug = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8853
cloudflg = false;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8854
minrad = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8855
kminrad = kpbl;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8856
zminrad = pblh;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8858
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprodorig.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { tkeprodorig.slice(MynnRange{kts, kte}).linear(mynn_element) = tkeprod_dn.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8861
for (k=1; (1) > 0 ? k <= (mynn_min((int)(((kpbl) + (10))), (int)(((kte) - (1))))) : k >= (mynn_min((int)(((kpbl) + (10))), (int)(((kte) - (1))))); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8862
maxcooling = mynn_minval(rthraten.slice(MynnRange{mynn_max((int)(kts), (int)(((k) - (1)))), ((k) + (1))}));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8864
if (((((((cldfra_bl1(k)) > (p5))) && (((cldfra_bl1(((k) + (1)))) < (p5))))) && (((maxcooling) < (cooling_thresh))))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8867
cloudflg = true;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8868
if (((maxcooling) < (minrad))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8869
minrad = maxcooling;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8870
kminrad = k;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8871
zminrad = r6_add(zw(k), r6_mul(p5, dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8872
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8873
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8874
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8876
if (cloudflg) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8877
zl1 = r6_mul(p25, dz1(kts));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8878
k = kminrad;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8880
templ = r6_mul(thl(k), ex1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8882
rvls = r6_mul(r6_mul(r6_mul(100.f, 6.112f), mynn_exp(r6_div(r6_mul(17.67f, r6_sub(templ, 273.16f)), r6_sub(templ, 29.65f)))), r6_div(ep_2, pres1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8883
temps = r6_add(templ, r6_div(r6_sub(sqw(k), rvls), r6_add(r6_div(cp, xlv), r6_div(r6_mul(r6_mul(ep_2, xlv), rvls), r6_mul(r_d, mynn_ipow<2>(templ))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8884
rvls = r6_mul(r6_mul(r6_mul(100.f, 6.112f), mynn_exp(r6_div(r6_mul(17.67f, r6_sub(temps, 273.15f)), r6_sub(temps, 29.65f)))), r6_div(ep_2, pres1(((k) + (1)))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8885
rcldb = mynn_max((float)(zero), (float)(r6_sub(sqw(k), rvls)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8886
rcldb = mynn_max((float)(rcldb), (float)(r6_add(qc_bl(k), qi_bl(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8889
dthvx = r6_sub(r6_add(thl(((k) + (2))), r6_mul(r6_mul(th1(((k) + (2))), p608), sqw(((k) + (2))))), r6_add(thl(k), r6_mul(r6_mul(th1(k), p608), sqw(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8892
dz_ent = r6_add(r6_div(r6_add(r6_mul(dz1(k), dz1(((k) + (1)))), r6_mul(dz1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k))), r6_div(r6_add(r6_mul(dz1(((k) + (1))), dz1(((k) + (2)))), r6_mul(dz1(((k) + (2))), dz1(((k) + (1))))), r6_add(dz1(((k) + (1))), dz1(((k) + (2))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8894
dthvx = r6_div(r6_mul(dthvx, dz200), dz_ent);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8895
dthvx = mynn_max((float)(dthvx), (float)(p1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8896
tmp1 = r6_div(r6_mul(xlvcp, rcldb), r6_mul(ex1(k), dthvx));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8899
ent_eff = mynn_min((float)(r6_add(p2, r6_mul(r6_mul(p2, eight), tmp1))), (float)(two));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8901
f0 = zero;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8902
for (kk=mynn_max((int)(kts), (int)(((kminrad) - (1)))); (1) > 0 ? kk <= (((kminrad) + (1))) : kk >= (((kminrad) + (1))); kk += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8903
radflux = r6_mul(rthraten(kk), ex1(kk));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8904
dp = r6_mul(r6_mul(rho1(k), grav), dz1(k));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8905
radflux = r6_mul(r6_div(r6_mul(radflux, cp), grav), dp);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8906
if (((radflux) < (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8907
f0 = r6_add(mynn_abs(radflux), f0);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8908
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8909
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8912
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8913
f0 = mynn_min((float)(f0), (float)(150.0f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8914
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8915
f0 = mynn_min((float)(r6_mul(p25, f0)), (float)(30.f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8916
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8917
cldtop_cooling = r6_mul(mynn_abs(minrad), 3600.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8920
wstar_rad = r6_mul(1.25f, mynn_pow(r6_div(r6_mul(r6_mul(r6_div(grav, thv(k)), dz200), f0), r6_mul(rho1(k), cp)), p333));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8921
wstar_rad = mynn_min((float)(mynn_max((float)(wstar_rad), (float)(p1))), (float)(three));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8922
dthvx = mynn_max((float)(r6_sub(thv(((k) + (1))), thv(k))), (float)(p1));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8923
went = r6_mul(r6_mul(p333, ent_eff), wstar_rad);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8927
zb1 = mynn_max((float)(zl1), (float)(r6_sub(zminrad, r6_mul(wstar_rad, 800.f))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8928
{ const int mynn_lo0 = (kts), mynn_hi0 = (((kminrad) + (3))), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8930
zagl = r6_add(zw(k), r6_mul(p5, dz1(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8931
if (((((k) == (kts))) && (((kminrad) == (kts))))) { zagl = r6_mul(0.4f, dz1(kts)); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8933
zfac(k) = mynn_min((float)(mynn_max((float)(r6_sub(one, r6_div(mynn_max((float)(r6_sub(zagl, zb1)), (float)(zero)), r6_sub(zminrad, zl1)))), (float)(zfmin))), (float)(one));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8934
zfacent(k) = r6_mul(mynn_max((float)(zero), (float)(r6_div(r6_sub(zminrad, zagl), zminrad))), mynn_ipow<3>(r6_sub(one, zfac(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8935
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; zagl = mynn_lane_bcast(zagl, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8936
zfacent_max = mynn_maxval(zfacent.slice(MynnRange{kts, ((kminrad) + (3))}));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8937
for (int mynn_element=0; mynn_element < (zfacent).size(); ++mynn_element) { zfacent.linear(mynn_element) = r6_div(zfacent.linear(mynn_element), mynn_max((float)(zfacent_max), (float)(1e-5f))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8940
{ const int mynn_lo1 = (kts), mynn_hi1 = (((kminrad) + (3))), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8941
tkeprod_dn(k) = mynn_max((float)(tkeprod_dn(k)), (float)(r6_mul(r6_div(r6_mul(r6_mul(p3, ent_eff), mynn_ipow<3>(wstar_rad)), dz500), zfacent(k))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8942
tkeprod_dn(k) = mynn_max((float)(zero), (float)(tkeprod_dn(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8943
tkeprod_dn(k) = mynn_min((float)(tkeprod_dn(k)), (float)(0.0004f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8944
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8946
tkeprod_dn(kminrad) = mynn_max((float)(tkeprod_dn(kminrad)), (float)(r6_mul(p333, tkeprod_dn(mynn_max((int)(kts), (int)(((kminrad) - (1))))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8948
tkeprod_dn(((kminrad) + (1))) = mynn_max((float)(tkeprod_dn(((kminrad) + (1)))), (float)(r6_mul(mynn_min((float)(ent_eff), (float)(p5)), tkeprod_dn(mynn_max((int)(kts), (int)(kminrad))))));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8951
wspd_pbl = mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(u1), mynn_ipow<2>(v1))), (float)(0.01f)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8952
ac_wsp = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(wspd_pbl, 10.f)), (float)(zero)), 15.f)), (float)(one)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8953
for (int mynn_element=0; mynn_element < (tkeprod_dn).size(); ++mynn_element) { tkeprod_dn.linear(mynn_element) = r6_mul(tkeprod_dn.linear(mynn_element), mynn_min((float)(ac_wsp), (float)(psig))); }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8954
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8957
maxtkeprod = r6_mul(mynn_maxval(tkeprod_dn), 3600.f);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8959
if (((cloudflg) && (debug))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8961
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8962
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8963
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8964
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8965
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8966
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8967
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8968
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8969
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8970
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8971
for (k=((kminrad) + (3)); ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8972
; /* Native diagnostic print; no field arithmetic. */
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8973
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/module_bl_mynnedmf.F90:8974
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:17-668
__device__ __forceinline__ void f_mynnedmf_driver(int ids, int ide, int jds, int jde, int kds, int kde, int ims, int ime, int jms, int jme, int kms, int kme, int its, int ite, int jts, int jte, int kts, int kte, bool f_qc, bool f_qi, bool f_qs, bool f_qoz, bool f_nc, bool f_ni, bool f_nifa, bool f_nwfa, bool f_nbca, int initflag, bool do_restart, bool do_dacycling, int icloud_bl, float delt, MynnArray<float, 2> dx, MynnArray<float, 2> xland, MynnArray<float, 2> ps, MynnArray<float, 2> ts, MynnArray<float, 2> qsfc, MynnArray<float, 2> ust, MynnArray<float, 2> ch, MynnArray<float, 2> hfx, MynnArray<float, 2> qfx, MynnArray<float, 2> wspd, MynnArray<float, 2> znt, MynnArray<float, 2> uoce, MynnArray<float, 2> voce, MynnArray<float, 3> dz, MynnArray<float, 3> u, MynnArray<float, 3> v, MynnArray<float, 3> w, MynnArray<float, 3> th, MynnArray<float, 3> tt, MynnArray<float, 3> p, MynnArray<float, 3> exner, MynnArray<float, 3> rho, MynnArray<float, 3> qv, MynnArray<float, 3> qc, MynnArray<float, 3> qi, MynnArray<float, 3> qs, MynnArray<float, 3> nc, MynnArray<float, 3> ni, MynnArray<float, 3> nifa, MynnArray<float, 3> nwfa, MynnArray<float, 3> nbca, MynnArray<float, 3> qoz, MynnArray<float, 3> rthraten, MynnArray<float, 2> pblh, MynnArray<int, 2> kpbl, MynnArray<float, 3> cldfra_bl, MynnArray<float, 3> qc_bl, MynnArray<float, 3> qi_bl, MynnArray<float, 2> maxwidth, MynnArray<float, 2> maxmf, MynnArray<float, 2> ztop_plume, MynnArray<float, 2> excess_h, MynnArray<float, 2> excess_q, MynnArray<float, 2> maxwidth_dd, MynnArray<float, 2> maxmf_dd, MynnArray<float, 2> maxtkeprod, MynnArray<float, 2> cldtop_cooling, MynnArray<float, 2> ent_eff, MynnArray<float, 3> qke, MynnArray<float, 3> qke_adv, MynnArray<float, 3> tsq, MynnArray<float, 3> qsq, MynnArray<float, 3> cov, MynnArray<float, 3> el_pbl, MynnArray<float, 3> rublten, MynnArray<float, 3> rvblten, MynnArray<float, 3> rthblten, MynnArray<float, 3> rqvblten, MynnArray<float, 3> rqcblten, MynnArray<float, 3> rqiblten, MynnArray<float, 3> rqsblten, MynnArray<float, 3> rncblten, MynnArray<float, 3> rniblten, MynnArray<float, 3> rnifablten, MynnArray<float, 3> rnwfablten, MynnArray<float, 3> rnbcablten, MynnArray<float, 3> rqozblten, MynnArray<float, 3> edmf_a, MynnArray<float, 3> edmf_w, MynnArray<float, 3> edmf_qt, MynnArray<float, 3> edmf_thl, MynnArray<float, 3> edmf_ent, MynnArray<float, 3> edmf_qc, MynnArray<float, 3> sub_thl, MynnArray<float, 3> sub_sqv, MynnArray<float, 3> det_thl, MynnArray<float, 3> det_sqv, MynnArray<float, 3> exch_h, MynnArray<float, 3> exch_m, MynnArray<float, 3> dqke, MynnArray<float, 3> qwt, MynnArray<float, 3> qshear, MynnArray<float, 3> qbuoy, MynnArray<float, 3> qdiss, MynnArray<float, 3> sh3d, MynnArray<float, 3> sm3d, int spp_pbl, MynnArray<float, 3> pattern_spp, bool bl_mynn_tkeadvect, int bl_mynn_tkebudget, int bl_mynn_cloudpdf, int bl_mynn_mixlength, float bl_mynn_closure, int bl_mynn_stfunc, int bl_mynn_topdown, int bl_mynn_scaleaware, int bl_mynn_dheat_opt, int bl_mynn_edmf, int bl_mynn_edmf_dd, int bl_mynn_edmf_mom, int bl_mynn_edmf_tke, int bl_mynn_output, int bl_mynn_mixscalars, int bl_mynn_mixaerosols, int bl_mynn_mixnumcon, int bl_mynn_cloudmix, int bl_mynn_mixqt, int bl_mynn_ess, MynnText& errmsg, int& errflg, bool mix_chem, int nchem, int ndvel, MynnArray<float, 4> chem3d, MynnArray<float, 4> settle3d, MynnArray<float, 3> vd3d, MynnOptional<bool> enh_mix, MynnArray<float, 2> frp_mean, MynnArray<float, 2> emis_ant_no) {
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
maxwidth_dd.rebase({ims, jms});
maxmf_dd.rebase({ims, jms});
maxtkeprod.rebase({ims, jms});
cldtop_cooling.rebase({ims, jms});
ent_eff.rebase({ims, jms});
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
float maxwidth_dd1;
float maxmf_dd1;
float maxtkeprod1;
float cldtop_cooling1;
float ent_eff1;
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:308
errmsg = " ";
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:309
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:311
for (j=(active_j >= 0 ? active_j : jts); (1) > 0 ? j <= ((active_j >= 0 ? active_j : jte)) : j >= ((active_j >= 0 ? active_j : jte)); j += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:312
for (i=(active_i >= 0 ? active_i : its); (1) > 0 ? i <= ((active_i >= 0 ? active_i : ite)) : i >= ((active_i >= 0 ? active_i : ite)); i += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:315
dx1 = dx(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:316
xland1 = xland(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:317
ps1 = ps(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:318
ts1 = ts(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:319
qsfc1 = qsfc(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:320
ust1 = ust(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:321
ch1 = ch(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:322
hfx1 = hfx(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:323
qfx1 = qfx(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:324
wspd1 = wspd(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:325
uoce1 = uoce(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:326
voce1 = voce(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:327
znt1 = znt(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:329
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:330
dz1(k) = dz(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:331
u1(k) = u(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:332
v1(k) = v(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:333
w1(k) = w(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:334
th1(k) = th(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:335
tt1(k) = tt(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:336
p1(k) = p(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:337
exner1(k) = exner(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:338
rho1(k) = rho(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:339
qv1(k) = qv(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:340
rthraten1(k) = rthraten(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:341
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:342
w1(((kte) + (1))) = w(i, ((kte) + (1)), j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:347
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:348
qc1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:349
qi1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:350
qs1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:351
qoz1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:352
nc1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:353
ni1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:354
nifa1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:355
nwfa1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:356
nbca1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:357
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:358
if (((f_qc) && ((qc.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:359
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:360
qc1(k) = qc(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:361
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:362
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:363
if (((f_qi) && ((qi.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:364
{ const int mynn_lo3 = (kts), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:365
qi1(k) = qi(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:366
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:367
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:368
if (((f_qs) && ((qs.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:369
{ const int mynn_lo4 = (kts), mynn_hi4 = (kte), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:370
qs1(k) = qs(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:371
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:372
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:373
if (((f_nc) && ((nc.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:374
{ const int mynn_lo5 = (kts), mynn_hi5 = (kte), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:375
nc1(k) = nc(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:376
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:377
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:378
if (((f_ni) && ((ni.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:379
{ const int mynn_lo6 = (kts), mynn_hi6 = (kte), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:380
ni1(k) = ni(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:381
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:382
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:383
if (((f_nifa) && ((nifa.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:384
{ const int mynn_lo7 = (kts), mynn_hi7 = (kte), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:385
nifa1(k) = nifa(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:386
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:387
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:388
if (((f_nwfa) && ((nwfa.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:389
{ const int mynn_lo8 = (kts), mynn_hi8 = (kte), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:390
nwfa1(k) = nwfa(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:391
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:392
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:393
if (((f_nbca) && ((nbca.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:394
{ const int mynn_lo9 = (kts), mynn_hi9 = (kte), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:395
nbca1(k) = nbca(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:396
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:397
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:398
if (((f_qoz) && ((qoz.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:399
{ const int mynn_lo10 = (kts), mynn_hi10 = (kte), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:400
qoz1(k) = qoz(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:401
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:402
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:405
f_mynnedmf_pre_run(kte, f_qc, f_qi, f_qs, qv1, qc1, qi1, qs1, sqv1, sqc1, sqi1, sqs1, errmsg, errflg);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:409
if (((((spp_pbl) > (0))) && ((pattern_spp.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:410
{ const int mynn_lo11 = (kts), mynn_hi11 = (kte), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { k = mynn_lo11 + mynn_it11*mynn_st11;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:411
pattern_spp1(k) = pattern_spp(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:412
} mynn_lane_sync(); k = mynn_lo11 + mynn_n11*mynn_st11; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:413
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:414
{ const int mynn_lo12 = (kts), mynn_hi12 = (kte), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { k = mynn_lo12 + mynn_it12*mynn_st12;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:415
pattern_spp1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:416
} mynn_lane_sync(); k = mynn_lo12 + mynn_n12*mynn_st12; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:417
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:420
pblh1 = pblh(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:421
kpbl1 = kpbl(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:423
{ const int mynn_lo13 = (kts), mynn_hi13 = (kte), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { k = mynn_lo13 + mynn_it13*mynn_st13;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:424
cldfrabl1(k) = cldfra_bl(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:425
qcbl1(k) = qc_bl(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:426
qibl1(k) = qi_bl(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:427
} mynn_lane_sync(); k = mynn_lo13 + mynn_n13*mynn_st13; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:429
{ const int mynn_lo14 = (kts), mynn_hi14 = (kte), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { k = mynn_lo14 + mynn_it14*mynn_st14;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:430
elpbl1(k) = el_pbl(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:431
qke1(k) = qke(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:432
qkeadv1(k) = qke_adv(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:433
cov1(k) = cov(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:434
tsq1(k) = tsq(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:435
qsq1(k) = qsq(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:436
sh1(k) = sh3d(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:437
sm1(k) = sm3d(i, k, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:438
} mynn_lane_sync(); k = mynn_lo14 + mynn_n14*mynn_st14; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:440
if ((((((((((chem3d.present())) && ((settle3d.present())))) && ((vd3d.present())))) && ((frp_mean.present())))) && ((emis_ant_no.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:442
{ const int mynn_lo15 = (1), mynn_hi15 = (nchem), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { ic = mynn_lo15 + mynn_it15*mynn_st15;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:443
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:444
chem1(k, ic) = chem3d(i, k, j, ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:445
settle1(k, ic) = settle3d(i, k, j, ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:446
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:447
} mynn_lane_sync(); ic = mynn_lo15 + mynn_n15*mynn_st15; if (mynn_n15 > 0) { const int mynn_src15 = (mynn_n15 - 1) % lane_width; k = mynn_lane_bcast(k, mynn_src15); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:448
{ const int mynn_lo16 = (1), mynn_hi16 = (ndvel), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { ic = mynn_lo16 + mynn_it16*mynn_st16;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:449
vd1(ic) = vd3d(i, j, ic);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:450
} mynn_lane_sync(); ic = mynn_lo16 + mynn_n16*mynn_st16; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:451
frp1 = frp_mean(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:452
emisant_no1 = emis_ant_no(i, j);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:453
} else {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:454
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (chem1).size(); mynn_element += lane_width) { chem1.linear(mynn_element) = 0.f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:455
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (settle1).size(); mynn_element += lane_width) { settle1.linear(mynn_element) = 0.f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:456
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (vd1).size(); mynn_element += lane_width) { vd1.linear(mynn_element) = 0.f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:457
frp1 = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:458
emisant_no1 = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:459
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:460
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (scalars).size(); mynn_element += lane_width) { scalars.linear(mynn_element) = 0.0f; } mynn_lane_sync();
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:462
{ const int mynn_lo17 = (kts), mynn_hi17 = (kte), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { k = mynn_lo17 + mynn_it17*mynn_st17;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:463
rqcblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:464
rqiblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:465
rqsblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:466
rqozblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:467
rncblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:468
rniblten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:469
rnifablten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:470
rnwfablten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:471
rnbcablten1(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:472
} mynn_lane_sync(); k = mynn_lo17 + mynn_n17*mynn_st17; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:474
f_mynnedmf(i, j, initflag, do_restart, do_dacycling, delt, dz1, dx1, u1, v1, w1, th1, sqv1, sqc1, sqi1, sqs1, nc1, ni1, nwfa1, nifa1, nbca1, qoz1, p1, exner1, rho1, tt1, xland1, ts1, qsfc1, ps1, ust1, ch1, hfx1, qfx1, znt1, wspd1, uoce1, voce1, qke1, qkeadv1, elpbl1, sh1, sm1, exchh1, exchm1, nchem, ndvel, chem1, vd1, frp1, emisant_no1, mix_chem, enh_mix, settle1, scalars, nscalars, tsq1, qsq1, cov1, rublten1, rvblten1, rthblten1, rqvblten1, rqcblten1, rqiblten1, rncblten1, rniblten1, rqsblten1, rnwfablten1, rnifablten1, rnbcablten1, rqozblten1, rthraten1, pblh1, kpbl1, maxwidth_dd1, maxwidth1, maxmf1, ztopplume1, excessh1, excessq1, maxmf_dd1, maxtkeprod1, cldtop_cooling1, ent_eff1, dqke1, qwt1, qshear1, qbuoy1, qdiss1, qcbl1, qibl1, cldfrabl1, bl_mynn_tkeadvect, bl_mynn_tkebudget, bl_mynn_cloudpdf, bl_mynn_mixlength, icloud_bl, bl_mynn_closure, bl_mynn_edmf, bl_mynn_edmf_mom, bl_mynn_edmf_tke, bl_mynn_mixscalars, bl_mynn_mixaerosols, bl_mynn_mixnumcon, bl_mynn_output, bl_mynn_cloudmix, bl_mynn_mixqt, bl_mynn_edmf_dd, bl_mynn_ess, edmfa1, edmfw1, edmfqt1, edmfthl1, edmfent1, edmfqc1, subthl1, subsqv1, detthl1, detsqv1, spp_pbl, pattern_spp1, f_qc, f_qi, f_nc, f_ni, f_qs, f_nwfa, f_nifa, f_nbca, f_qoz, kts, kte, errmsg, errflg);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:538
f_mynnedmf_post_run(kte, f_qc, f_qi, f_qs, delt, qv1, qc1, qi1, qs1, rqvblten1, rqcblten1, rqiblten1, rqsblten1, errmsg, errflg);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:542
pblh(i, j) = pblh1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:543
kpbl(i, j) = kpbl1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:544
{ const int mynn_lo18 = (kts), mynn_hi18 = (kte), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { k = mynn_lo18 + mynn_it18*mynn_st18;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:545
cldfra_bl(i, k, j) = cldfrabl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:546
qc_bl(i, k, j) = qcbl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:547
qi_bl(i, k, j) = qibl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:548
} mynn_lane_sync(); k = mynn_lo18 + mynn_n18*mynn_st18; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:550
{ const int mynn_lo19 = (kts), mynn_hi19 = (kte), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { k = mynn_lo19 + mynn_it19*mynn_st19;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:551
el_pbl(i, k, j) = elpbl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:552
qke(i, k, j) = qke1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:553
qke_adv(i, k, j) = qkeadv1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:554
cov(i, k, j) = cov1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:555
tsq(i, k, j) = tsq1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:556
qsq(i, k, j) = qsq1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:557
sh3d(i, k, j) = sh1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:558
sm3d(i, k, j) = sm1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:559
} mynn_lane_sync(); k = mynn_lo19 + mynn_n19*mynn_st19; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:562
{ const int mynn_lo20 = (kts), mynn_hi20 = (kte), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { k = mynn_lo20 + mynn_it20*mynn_st20;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:563
rublten(i, k, j) = rublten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:564
rvblten(i, k, j) = rvblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:565
rthblten(i, k, j) = rthblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:566
rqvblten(i, k, j) = rqvblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:567
} mynn_lane_sync(); k = mynn_lo20 + mynn_n20*mynn_st20; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:568
if (((f_qc) && ((rqcblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:569
{ const int mynn_lo21 = (kts), mynn_hi21 = (kte), mynn_st21 = (1); const int mynn_n21 = mynn_trip(mynn_lo21, mynn_hi21, mynn_st21); mynn_lane_sync(); for (int mynn_it21 = lane_sub; mynn_it21 < mynn_n21; mynn_it21 += lane_width) { k = mynn_lo21 + mynn_it21*mynn_st21;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:570
rqcblten(i, k, j) = rqcblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:571
} mynn_lane_sync(); k = mynn_lo21 + mynn_n21*mynn_st21; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:572
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:573
if (((f_qi) && ((rqiblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:574
{ const int mynn_lo22 = (kts), mynn_hi22 = (kte), mynn_st22 = (1); const int mynn_n22 = mynn_trip(mynn_lo22, mynn_hi22, mynn_st22); mynn_lane_sync(); for (int mynn_it22 = lane_sub; mynn_it22 < mynn_n22; mynn_it22 += lane_width) { k = mynn_lo22 + mynn_it22*mynn_st22;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:575
rqiblten(i, k, j) = rqiblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:576
} mynn_lane_sync(); k = mynn_lo22 + mynn_n22*mynn_st22; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:577
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:578
if (((f_qs) && ((rqsblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:579
{ const int mynn_lo23 = (kts), mynn_hi23 = (kte), mynn_st23 = (1); const int mynn_n23 = mynn_trip(mynn_lo23, mynn_hi23, mynn_st23); mynn_lane_sync(); for (int mynn_it23 = lane_sub; mynn_it23 < mynn_n23; mynn_it23 += lane_width) { k = mynn_lo23 + mynn_it23*mynn_st23;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:580
rqsblten(i, k, j) = rqsblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:581
} mynn_lane_sync(); k = mynn_lo23 + mynn_n23*mynn_st23; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:582
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:583
if (((f_qoz) && ((rqozblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:584
{ const int mynn_lo24 = (kts), mynn_hi24 = (kte), mynn_st24 = (1); const int mynn_n24 = mynn_trip(mynn_lo24, mynn_hi24, mynn_st24); mynn_lane_sync(); for (int mynn_it24 = lane_sub; mynn_it24 < mynn_n24; mynn_it24 += lane_width) { k = mynn_lo24 + mynn_it24*mynn_st24;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:585
rqozblten(i, k, j) = rqozblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:586
} mynn_lane_sync(); k = mynn_lo24 + mynn_n24*mynn_st24; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:587
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:588
if (((f_nc) && ((rncblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:589
{ const int mynn_lo25 = (kts), mynn_hi25 = (kte), mynn_st25 = (1); const int mynn_n25 = mynn_trip(mynn_lo25, mynn_hi25, mynn_st25); mynn_lane_sync(); for (int mynn_it25 = lane_sub; mynn_it25 < mynn_n25; mynn_it25 += lane_width) { k = mynn_lo25 + mynn_it25*mynn_st25;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:590
rncblten(i, k, j) = rncblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:591
} mynn_lane_sync(); k = mynn_lo25 + mynn_n25*mynn_st25; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:592
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:593
if (((f_ni) && ((rniblten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:594
{ const int mynn_lo26 = (kts), mynn_hi26 = (kte), mynn_st26 = (1); const int mynn_n26 = mynn_trip(mynn_lo26, mynn_hi26, mynn_st26); mynn_lane_sync(); for (int mynn_it26 = lane_sub; mynn_it26 < mynn_n26; mynn_it26 += lane_width) { k = mynn_lo26 + mynn_it26*mynn_st26;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:595
rniblten(i, k, j) = rniblten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:596
} mynn_lane_sync(); k = mynn_lo26 + mynn_n26*mynn_st26; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:597
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:598
if (((f_nifa) && ((rnifablten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:599
{ const int mynn_lo27 = (kts), mynn_hi27 = (kte), mynn_st27 = (1); const int mynn_n27 = mynn_trip(mynn_lo27, mynn_hi27, mynn_st27); mynn_lane_sync(); for (int mynn_it27 = lane_sub; mynn_it27 < mynn_n27; mynn_it27 += lane_width) { k = mynn_lo27 + mynn_it27*mynn_st27;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:600
rnifablten(i, k, j) = rnifablten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:601
} mynn_lane_sync(); k = mynn_lo27 + mynn_n27*mynn_st27; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:602
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:603
if (((f_nwfa) && ((rnwfablten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:604
{ const int mynn_lo28 = (kts), mynn_hi28 = (kte), mynn_st28 = (1); const int mynn_n28 = mynn_trip(mynn_lo28, mynn_hi28, mynn_st28); mynn_lane_sync(); for (int mynn_it28 = lane_sub; mynn_it28 < mynn_n28; mynn_it28 += lane_width) { k = mynn_lo28 + mynn_it28*mynn_st28;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:605
rnwfablten(i, k, j) = rnwfablten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:606
} mynn_lane_sync(); k = mynn_lo28 + mynn_n28*mynn_st28; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:607
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:608
if (((f_nbca) && ((rnbcablten.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:609
{ const int mynn_lo29 = (kts), mynn_hi29 = (kte), mynn_st29 = (1); const int mynn_n29 = mynn_trip(mynn_lo29, mynn_hi29, mynn_st29); mynn_lane_sync(); for (int mynn_it29 = lane_sub; mynn_it29 < mynn_n29; mynn_it29 += lane_width) { k = mynn_lo29 + mynn_it29*mynn_st29;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:610
rnbcablten(i, k, j) = rnbcablten1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:611
} mynn_lane_sync(); k = mynn_lo29 + mynn_n29*mynn_st29; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:612
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:614
{ const int mynn_lo30 = (kts), mynn_hi30 = (kte), mynn_st30 = (1); const int mynn_n30 = mynn_trip(mynn_lo30, mynn_hi30, mynn_st30); mynn_lane_sync(); for (int mynn_it30 = lane_sub; mynn_it30 < mynn_n30; mynn_it30 += lane_width) { k = mynn_lo30 + mynn_it30*mynn_st30;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:615
edmf_a(i, k, j) = edmfa1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:616
edmf_w(i, k, j) = edmfw1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:617
edmf_qt(i, k, j) = edmfqt1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:618
edmf_thl(i, k, j) = edmfthl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:619
edmf_ent(i, k, j) = edmfent1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:620
edmf_qc(i, k, j) = edmfqc1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:621
sub_thl(i, k, j) = subthl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:622
sub_sqv(i, k, j) = subsqv1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:623
det_thl(i, k, j) = detthl1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:624
det_sqv(i, k, j) = detsqv1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:625
} mynn_lane_sync(); k = mynn_lo30 + mynn_n30*mynn_st30; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:628
maxwidth(i, j) = maxwidth1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:629
maxmf(i, j) = maxmf1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:630
ztop_plume(i, j) = ztopplume1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:631
excess_h(i, j) = excessh1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:632
excess_q(i, j) = excessq1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:633
maxwidth_dd(i, j) = maxwidth_dd1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:634
maxmf_dd(i, j) = maxmf_dd1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:635
maxtkeprod(i, j) = maxtkeprod1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:636
cldtop_cooling(i, j) = cldtop_cooling1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:637
ent_eff(i, j) = ent_eff1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:639
{ const int mynn_lo31 = (kts), mynn_hi31 = (kte), mynn_st31 = (1); const int mynn_n31 = mynn_trip(mynn_lo31, mynn_hi31, mynn_st31); mynn_lane_sync(); for (int mynn_it31 = lane_sub; mynn_it31 < mynn_n31; mynn_it31 += lane_width) { k = mynn_lo31 + mynn_it31*mynn_st31;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:640
exch_h(i, k, j) = exchh1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:641
exch_m(i, k, j) = exchm1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:642
} mynn_lane_sync(); k = mynn_lo31 + mynn_n31*mynn_st31; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:644
if ((((((((((qwt.present())) && ((qbuoy.present())))) && ((qshear.present())))) && ((qdiss.present())))) && ((dqke.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:646
{ const int mynn_lo32 = (kts), mynn_hi32 = (kte), mynn_st32 = (1); const int mynn_n32 = mynn_trip(mynn_lo32, mynn_hi32, mynn_st32); mynn_lane_sync(); for (int mynn_it32 = lane_sub; mynn_it32 < mynn_n32; mynn_it32 += lane_width) { k = mynn_lo32 + mynn_it32*mynn_st32;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:647
dqke(i, k, j) = dqke1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:648
qwt(i, k, j) = qwt1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:649
qshear(i, k, j) = qshear1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:650
qbuoy(i, k, j) = qbuoy1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:651
qdiss(i, k, j) = qdiss1(k);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:652
} mynn_lane_sync(); k = mynn_lo32 + mynn_n32*mynn_st32; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:653
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:655
if (((mix_chem) && ((chem3d.present())))) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:656
{ const int mynn_lo33 = (1), mynn_hi33 = (nchem), mynn_st33 = (1); const int mynn_n33 = mynn_trip(mynn_lo33, mynn_hi33, mynn_st33); mynn_lane_sync(); for (int mynn_it33 = lane_sub; mynn_it33 < mynn_n33; mynn_it33 += lane_width) { ic = mynn_lo33 + mynn_it33*mynn_st33;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:657
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:658
chem3d(i, k, j, ic) = mynn_max((float)(chem1(k, ic)), (float)(1.e-12f));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:659
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:660
} mynn_lane_sync(); ic = mynn_lo33 + mynn_n33*mynn_st33; if (mynn_n33 > 0) { const int mynn_src33 = (mynn_n33 - 1) % lane_width; k = mynn_lane_bcast(k, mynn_src33); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:661
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:663
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:664
}
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:674-690
__device__ __noinline__ void f_mynnedmf_pre_init(MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:687
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:688
errmsg = " ";
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:696-712
__device__ __noinline__ void f_mynnedmf_pre_finalize(MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:709
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:710
errmsg = " ";
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:718-787
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:756
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:757
sqc(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:758
sqi(k) = 0.f;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:759
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:762
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:763
sqv(k) = r6_div(qv(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:764
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:767
if (f_qc) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:768
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:769
sqc(k) = r6_div(qc(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:770
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:771
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:772
if (f_qi) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:773
{ const int mynn_lo3 = (kts), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:774
sqi(k) = r6_div(qi(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:775
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:776
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:777
if (f_qs) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:778
{ const int mynn_lo4 = (kts), mynn_hi4 = (kte), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:779
sqs(k) = r6_div(qs(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:780
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:781
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:784
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:785
errmsg = " ";
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:794-810
__device__ __noinline__ void f_mynnedmf_post_init(MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:807
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:808
errmsg = " ";
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:816-832
__device__ __noinline__ void f_mynnedmf_post_finalize(MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:829
errflg = 0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:830
errmsg = " ";
}
// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:838-919
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
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:881
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:882
sq = r6_div(qv(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:883
sqv(k) = r6_add(sq, r6_mul(dqv(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:884
rq = r6_div(sqv(k), r6_sub(1.f, sqv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:885
dqv(k) = r6_div(r6_sub(rq, qv(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:886
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src0); sq = mynn_lane_bcast(sq, mynn_src0); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:888
if (f_qc) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:889
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:890
sq = r6_div(qc(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:891
sqc(k) = r6_add(sq, r6_mul(dqc(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:892
rq = r6_mul(sqc(k), r6_add(1.f, sqv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:893
dqc(k) = r6_div(r6_sub(rq, qc(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:894
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src1); sq = mynn_lane_bcast(sq, mynn_src1); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:895
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:897
if (f_qi) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:898
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:899
sq = r6_div(qi(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:900
sqi(k) = r6_add(sq, r6_mul(dqi(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:901
rq = r6_mul(sqi(k), r6_add(1.f, sqv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:902
dqi(k) = r6_div(r6_sub(rq, qi(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:903
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src2); sq = mynn_lane_bcast(sq, mynn_src2); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:904
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:906
if (f_qs) {
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:907
{ const int mynn_lo3 = (kts), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:908
sq = r6_div(qs(k), r6_add(1.f, qv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:909
sqs(k) = r6_add(sq, r6_mul(dqs(k), delt));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:910
rq = r6_mul(sqs(k), r6_add(1.f, sqv(k)));
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:911
dqs(k) = r6_div(r6_sub(rq, qs(k)), delt);
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:912
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src3); sq = mynn_lane_bcast(sq, mynn_src3); } }
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:913
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:916
errmsg = " ";
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:917
errflg = 0;
}
};
#ifdef HEX_NUMERICS_FAST
#define MYNN_L4_REGS __maxnreg__(160)
#else
#define MYNN_L4_REGS
#endif

// RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:17-668
template<int MYNN_LANES> __device__ __forceinline__ void rrfs_mynnedmf_column(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_nc,int f_ni,int f_nifa,int f_nwfa,int f_nbca,int initflag,int do_restart,int do_dacycling,int icloud_bl,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tt,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* nc,float* ni,float* nifa,float* nwfa,float* nbca,float* qoz,float* rthraten,float* pblh,int* kpbl,float* cldfra_bl,float* qc_bl,float* qi_bl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* maxwidth_dd,float* maxmf_dd,float* maxtkeprod,float* cldtop_cooling,float* ent_eff,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rncblten,float* rniblten,float* rnifablten,float* rnwfablten,float* rnbcablten,float* rqozblten,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int bl_mynn_tkebudget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_stfunc,int bl_mynn_topdown,int bl_mynn_scaleaware,int bl_mynn_dheat_opt,int bl_mynn_edmf,int bl_mynn_edmf_dd,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_output,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_ess,int mix_chem,int nchem,int ndvel,float* chem3d,float* settle3d,float* vd3d,int enh_mix,int present_enh_mix,float* frp_mean,float* emis_ant_no) {
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
MynnArray<float,2> a_maxwidth_dd;a_maxwidth_dd.data=maxwidth_dd;
a_maxwidth_dd.lo[0]=ims;a_maxwidth_dd.hi[0]=ime;a_maxwidth_dd.stride[0]=1;
a_maxwidth_dd.lo[1]=jms;a_maxwidth_dd.hi[1]=jme;a_maxwidth_dd.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_maxmf_dd;a_maxmf_dd.data=maxmf_dd;
a_maxmf_dd.lo[0]=ims;a_maxmf_dd.hi[0]=ime;a_maxmf_dd.stride[0]=1;
a_maxmf_dd.lo[1]=jms;a_maxmf_dd.hi[1]=jme;a_maxmf_dd.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_maxtkeprod;a_maxtkeprod.data=maxtkeprod;
a_maxtkeprod.lo[0]=ims;a_maxtkeprod.hi[0]=ime;a_maxtkeprod.stride[0]=1;
a_maxtkeprod.lo[1]=jms;a_maxtkeprod.hi[1]=jme;a_maxtkeprod.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_cldtop_cooling;a_cldtop_cooling.data=cldtop_cooling;
a_cldtop_cooling.lo[0]=ims;a_cldtop_cooling.hi[0]=ime;a_cldtop_cooling.stride[0]=1;
a_cldtop_cooling.lo[1]=jms;a_cldtop_cooling.hi[1]=jme;a_cldtop_cooling.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_ent_eff;a_ent_eff.data=ent_eff;
a_ent_eff.lo[0]=ims;a_ent_eff.hi[0]=ime;a_ent_eff.stride[0]=1;
a_ent_eff.lo[1]=jms;a_ent_eff.hi[1]=jme;a_ent_eff.stride[1]=1*(ime-ims+1);
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
column.f_mynnedmf_driver(ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,(f_qc!=0),(f_qi!=0),(f_qs!=0),(f_qoz!=0),(f_nc!=0),(f_ni!=0),(f_nifa!=0),(f_nwfa!=0),(f_nbca!=0),initflag,(do_restart!=0),(do_dacycling!=0),icloud_bl,delt,a_dx,a_xland,a_ps,a_ts,a_qsfc,a_ust,a_ch,a_hfx,a_qfx,a_wspd,a_znt,a_uoce,a_voce,a_dz,a_u,a_v,a_w,a_th,a_tt,a_p,a_exner,a_rho,a_qv,a_qc,a_qi,a_qs,a_nc,a_ni,a_nifa,a_nwfa,a_nbca,a_qoz,a_rthraten,a_pblh,a_kpbl,a_cldfra_bl,a_qc_bl,a_qi_bl,a_maxwidth,a_maxmf,a_ztop_plume,a_excess_h,a_excess_q,a_maxwidth_dd,a_maxmf_dd,a_maxtkeprod,a_cldtop_cooling,a_ent_eff,a_qke,a_qke_adv,a_tsq,a_qsq,a_cov,a_el_pbl,a_rublten,a_rvblten,a_rthblten,a_rqvblten,a_rqcblten,a_rqiblten,a_rqsblten,a_rncblten,a_rniblten,a_rnifablten,a_rnwfablten,a_rnbcablten,a_rqozblten,a_edmf_a,a_edmf_w,a_edmf_qt,a_edmf_thl,a_edmf_ent,a_edmf_qc,a_sub_thl,a_sub_sqv,a_det_thl,a_det_sqv,a_exch_h,a_exch_m,a_dqke,a_qwt,a_qshear,a_qbuoy,a_qdiss,a_sh3d,a_sm3d,spp_pbl,a_pattern_spp,(bl_mynn_tkeadvect!=0),bl_mynn_tkebudget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_stfunc,bl_mynn_topdown,bl_mynn_scaleaware,bl_mynn_dheat_opt,bl_mynn_edmf,bl_mynn_edmf_dd,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_output,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_ess,message,error_flag,(mix_chem!=0),nchem,ndvel,a_chem3d,a_settle3d,a_vd3d,(present_enh_mix ? MynnOptional<bool>(enh_mix) : MynnOptional<bool>()),a_frp_mean,a_emis_ant_no);
error_flags[index]=error_flag;
}
extern "C" __global__ void rrfs_mynnedmf_run_l1(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_nc,int f_ni,int f_nifa,int f_nwfa,int f_nbca,int initflag,int do_restart,int do_dacycling,int icloud_bl,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tt,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* nc,float* ni,float* nifa,float* nwfa,float* nbca,float* qoz,float* rthraten,float* pblh,int* kpbl,float* cldfra_bl,float* qc_bl,float* qi_bl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* maxwidth_dd,float* maxmf_dd,float* maxtkeprod,float* cldtop_cooling,float* ent_eff,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rncblten,float* rniblten,float* rnifablten,float* rnwfablten,float* rnbcablten,float* rqozblten,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int bl_mynn_tkebudget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_stfunc,int bl_mynn_topdown,int bl_mynn_scaleaware,int bl_mynn_dheat_opt,int bl_mynn_edmf,int bl_mynn_edmf_dd,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_output,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_ess,int mix_chem,int nchem,int ndvel,float* chem3d,float* settle3d,float* vd3d,int enh_mix,int present_enh_mix,float* frp_mean,float* emis_ant_no) { rrfs_mynnedmf_column<1>(batch_first,batch_count,workspace,workspace_stride,error_flags,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,f_qc,f_qi,f_qs,f_qoz,f_nc,f_ni,f_nifa,f_nwfa,f_nbca,initflag,do_restart,do_dacycling,icloud_bl,delt,dx,xland,ps,ts,qsfc,ust,ch,hfx,qfx,wspd,znt,uoce,voce,dz,u,v,w,th,tt,p,exner,rho,qv,qc,qi,qs,nc,ni,nifa,nwfa,nbca,qoz,rthraten,pblh,kpbl,cldfra_bl,qc_bl,qi_bl,maxwidth,maxmf,ztop_plume,excess_h,excess_q,maxwidth_dd,maxmf_dd,maxtkeprod,cldtop_cooling,ent_eff,qke,qke_adv,tsq,qsq,cov,el_pbl,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqsblten,rncblten,rniblten,rnifablten,rnwfablten,rnbcablten,rqozblten,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc,sub_thl,sub_sqv,det_thl,det_sqv,exch_h,exch_m,dqke,qwt,qshear,qbuoy,qdiss,sh3d,sm3d,spp_pbl,pattern_spp,bl_mynn_tkeadvect,bl_mynn_tkebudget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_stfunc,bl_mynn_topdown,bl_mynn_scaleaware,bl_mynn_dheat_opt,bl_mynn_edmf,bl_mynn_edmf_dd,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_output,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_ess,mix_chem,nchem,ndvel,chem3d,settle3d,vd3d,enh_mix,present_enh_mix,frp_mean,emis_ant_no); }
extern "C" __global__ MYNN_L4_REGS void rrfs_mynnedmf_run_l4(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_nc,int f_ni,int f_nifa,int f_nwfa,int f_nbca,int initflag,int do_restart,int do_dacycling,int icloud_bl,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tt,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* nc,float* ni,float* nifa,float* nwfa,float* nbca,float* qoz,float* rthraten,float* pblh,int* kpbl,float* cldfra_bl,float* qc_bl,float* qi_bl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* maxwidth_dd,float* maxmf_dd,float* maxtkeprod,float* cldtop_cooling,float* ent_eff,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rncblten,float* rniblten,float* rnifablten,float* rnwfablten,float* rnbcablten,float* rqozblten,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int bl_mynn_tkebudget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_stfunc,int bl_mynn_topdown,int bl_mynn_scaleaware,int bl_mynn_dheat_opt,int bl_mynn_edmf,int bl_mynn_edmf_dd,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_output,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_ess,int mix_chem,int nchem,int ndvel,float* chem3d,float* settle3d,float* vd3d,int enh_mix,int present_enh_mix,float* frp_mean,float* emis_ant_no) { rrfs_mynnedmf_column<4>(batch_first,batch_count,workspace,workspace_stride,error_flags,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,f_qc,f_qi,f_qs,f_qoz,f_nc,f_ni,f_nifa,f_nwfa,f_nbca,initflag,do_restart,do_dacycling,icloud_bl,delt,dx,xland,ps,ts,qsfc,ust,ch,hfx,qfx,wspd,znt,uoce,voce,dz,u,v,w,th,tt,p,exner,rho,qv,qc,qi,qs,nc,ni,nifa,nwfa,nbca,qoz,rthraten,pblh,kpbl,cldfra_bl,qc_bl,qi_bl,maxwidth,maxmf,ztop_plume,excess_h,excess_q,maxwidth_dd,maxmf_dd,maxtkeprod,cldtop_cooling,ent_eff,qke,qke_adv,tsq,qsq,cov,el_pbl,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqsblten,rncblten,rniblten,rnifablten,rnwfablten,rnbcablten,rqozblten,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc,sub_thl,sub_sqv,det_thl,det_sqv,exch_h,exch_m,dqke,qwt,qshear,qbuoy,qdiss,sh3d,sm3d,spp_pbl,pattern_spp,bl_mynn_tkeadvect,bl_mynn_tkebudget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_stfunc,bl_mynn_topdown,bl_mynn_scaleaware,bl_mynn_dheat_opt,bl_mynn_edmf,bl_mynn_edmf_dd,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_output,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_ess,mix_chem,nchem,ndvel,chem3d,settle3d,vd3d,enh_mix,present_enh_mix,frp_mean,emis_ant_no); }
extern "C" __global__ void rrfs_mynnedmf_run_l16(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_nc,int f_ni,int f_nifa,int f_nwfa,int f_nbca,int initflag,int do_restart,int do_dacycling,int icloud_bl,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tt,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* nc,float* ni,float* nifa,float* nwfa,float* nbca,float* qoz,float* rthraten,float* pblh,int* kpbl,float* cldfra_bl,float* qc_bl,float* qi_bl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* maxwidth_dd,float* maxmf_dd,float* maxtkeprod,float* cldtop_cooling,float* ent_eff,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rncblten,float* rniblten,float* rnifablten,float* rnwfablten,float* rnbcablten,float* rqozblten,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int bl_mynn_tkebudget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_stfunc,int bl_mynn_topdown,int bl_mynn_scaleaware,int bl_mynn_dheat_opt,int bl_mynn_edmf,int bl_mynn_edmf_dd,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_output,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_ess,int mix_chem,int nchem,int ndvel,float* chem3d,float* settle3d,float* vd3d,int enh_mix,int present_enh_mix,float* frp_mean,float* emis_ant_no) { rrfs_mynnedmf_column<16>(batch_first,batch_count,workspace,workspace_stride,error_flags,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,f_qc,f_qi,f_qs,f_qoz,f_nc,f_ni,f_nifa,f_nwfa,f_nbca,initflag,do_restart,do_dacycling,icloud_bl,delt,dx,xland,ps,ts,qsfc,ust,ch,hfx,qfx,wspd,znt,uoce,voce,dz,u,v,w,th,tt,p,exner,rho,qv,qc,qi,qs,nc,ni,nifa,nwfa,nbca,qoz,rthraten,pblh,kpbl,cldfra_bl,qc_bl,qi_bl,maxwidth,maxmf,ztop_plume,excess_h,excess_q,maxwidth_dd,maxmf_dd,maxtkeprod,cldtop_cooling,ent_eff,qke,qke_adv,tsq,qsq,cov,el_pbl,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqsblten,rncblten,rniblten,rnifablten,rnwfablten,rnbcablten,rqozblten,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc,sub_thl,sub_sqv,det_thl,det_sqv,exch_h,exch_m,dqke,qwt,qshear,qbuoy,qdiss,sh3d,sm3d,spp_pbl,pattern_spp,bl_mynn_tkeadvect,bl_mynn_tkebudget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_stfunc,bl_mynn_topdown,bl_mynn_scaleaware,bl_mynn_dheat_opt,bl_mynn_edmf,bl_mynn_edmf_dd,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_output,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_ess,mix_chem,nchem,ndvel,chem3d,settle3d,vd3d,enh_mix,present_enh_mix,frp_mean,emis_ant_no); }
'''

ARGUMENTS = {'ids': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ide': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jds': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jde': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kds': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kde': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ims': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ime': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jms': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jme': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kms': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kme': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'its': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ite': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jts': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jte': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kts': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kte': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qc': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qi': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qs': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qoz': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_nc': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_ni': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_nifa': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_nwfa': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_nbca': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'initflag': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'do_restart': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'do_dacycling': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'icloud_bl': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'delt': {'kind': 'float', 'dimensions': [], 'intent': 'in', 'optional': False}, 'dx': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'xland': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ps': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ts': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'qsfc': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ust': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ch': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'hfx': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'qfx': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'wspd': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'znt': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'uoce': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'voce': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'dz': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'u': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'v': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'w': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'th': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'tt': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'p': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'exner': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'rho': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'qv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'qc': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qi': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qs': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'nc': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'ni': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'nifa': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'nwfa': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'nbca': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qoz': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'rthraten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'pblh': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'kpbl': {'kind': 'int', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'cldfra_bl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qc_bl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qi_bl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'maxwidth': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'maxmf': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'ztop_plume': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'excess_h': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'excess_q': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'maxwidth_dd': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'maxmf_dd': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'maxtkeprod': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'cldtop_cooling': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'ent_eff': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'qke': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qke_adv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'tsq': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qsq': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'cov': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'el_pbl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rublten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rvblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rthblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rqvblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rqcblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqiblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqsblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rncblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rniblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rnifablten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rnwfablten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rnbcablten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqozblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_a': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_w': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_qt': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_thl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_ent': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_qc': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'sub_thl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'sub_sqv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'det_thl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'det_sqv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'exch_h': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': False}, 'exch_m': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': False}, 'dqke': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qwt': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qshear': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qbuoy': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qdiss': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'sh3d': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'sm3d': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'spp_pbl': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'pattern_spp': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'bl_mynn_tkeadvect': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_tkebudget': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_cloudpdf': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixlength': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_closure': {'kind': 'float', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_stfunc': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_topdown': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_scaleaware': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_dheat_opt': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_dd': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_mom': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_tke': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_output': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixscalars': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixaerosols': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixnumcon': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_cloudmix': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixqt': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_ess': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'errmsg': {'kind': 'char', 'dimensions': [], 'intent': 'out', 'optional': False}, 'errflg': {'kind': 'int', 'dimensions': [], 'intent': 'out', 'optional': False}, 'mix_chem': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'nchem': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ndvel': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'chem3d': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme', 'nchem'], 'intent': 'inout', 'optional': True}, 'settle3d': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme', 'nchem'], 'intent': 'inout', 'optional': True}, 'vd3d': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme', 'ndvel'], 'intent': 'in', 'optional': True}, 'enh_mix': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': True}, 'frp_mean': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': True}, 'emis_ant_no': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': True}}
WORKSPACE_ARRAYS = [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['n']}, {'kind': 'float', 'dimensions': ['kte']}, {'kind': 'float', 'dimensions': ['kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte', '1:nup']}, {'kind': 'float', 'dimensions': ['nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nscalars']}, {'kind': 'int', 'dimensions': ['1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'int', 'dimensions': ['1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['ndvel']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kme']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}]
WORKSPACE_FRAMES = {'mynnedmf': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['ddmp_mf', 'dmp_mf', 'get_pblh', 'mym_condensation', 'mym_initialize', 'mym_predict', 'mym_turbulence', 'mynn_mix_chem', 'mynn_mix_scalars', 'mynn_tendencies', 'phih', 'phim', 'retrieve_exchange_coeffs', 'scale_aware', 'topdown_cloudrad']}, 'mym_initialize': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['mym_length', 'mym_level2']}, 'mym_level2': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['qsat_blend', 'xl_blend']}, 'mym_length': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['boulac_length']}, 'boulac_length0': {'arrays': [], 'calls': []}, 'boulac_length': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': []}, 'mym_turbulence': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['mym_length', 'mym_level2']}, 'mym_predict': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['tridiag2']}, 'mym_condensation': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['esat_blend', 'qsat_blend', 'xl_blend']}, 'mynn_tendencies': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['moisture_check', 'qsat_blend', 'tridiag2']}, 'moisture_check': {'arrays': [], 'calls': []}, 'mynn_mix_chem': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['tridiag3']}, 'mynn_mix_scalars': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['tridiag3']}, 'retrieve_exchange_coeffs': {'arrays': [], 'calls': []}, 'tridiag': {'arrays': [{'kind': 'float', 'dimensions': ['n']}], 'calls': []}, 'tridiag2': {'arrays': [{'kind': 'float', 'dimensions': ['kte']}, {'kind': 'float', 'dimensions': ['kte']}], 'calls': []}, 'tridiag3': {'arrays': [], 'calls': []}, 'get_pblh': {'arrays': [], 'calls': []}, 'dmp_mf': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte', '1:nup']}, {'kind': 'float', 'dimensions': ['nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nscalars']}, {'kind': 'int', 'dimensions': ['1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['condensation_edmf', 'esat_blend', 'qsat_blend', 'xl_blend']}, 'condensation_edmf': {'arrays': [], 'calls': ['qsat_blend']}, 'condensation_ddmf': {'arrays': [], 'calls': ['qsat_blend']}, 'condensation_edmf_r': {'arrays': [], 'calls': ['qsat_blend']}, 'ddmp_mf': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'int', 'dimensions': ['1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['condensation_ddmf', 'qsat_blend']}, 'scale_aware': {'arrays': [], 'calls': []}, 'esat_blend': {'arrays': [], 'calls': []}, 'qsat_blend': {'arrays': [], 'calls': []}, 'xl_blend': {'arrays': [], 'calls': []}, 'phim': {'arrays': [], 'calls': []}, 'phih': {'arrays': [], 'calls': []}, 'topdown_cloudrad': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': []}, 'mynnedmf_driver': {'arrays': [{'kind': 'float', 'dimensions': ['ndvel']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kme']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['mynnedmf', 'mynnedmf_post_run', 'mynnedmf_pre_run']}, 'mynnedmf_pre_init': {'arrays': [], 'calls': []}, 'mynnedmf_pre_finalize': {'arrays': [], 'calls': []}, 'mynnedmf_pre_run': {'arrays': [], 'calls': []}, 'mynnedmf_post_init': {'arrays': [], 'calls': []}, 'mynnedmf_post_finalize': {'arrays': [], 'calls': []}, 'mynnedmf_post_run': {'arrays': [{'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}], 'calls': []}}
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
    # call path is needed, not the sum of all sibling routines.
    # All optional branches stay in the bound.
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

# RRFS 44c0849 mpas_atmphys_driver_pbl.F:1637-1639, added downdraft diagnostics.
EXTRA_SURFACE_FIELDS = ('maxwidth_dd', 'maxmf_dd', 'maxtkeprod_dd', 'cldtop_cooling', 'ent_eff')
NATIVE_ARRAY_BINDINGS.update({name: name + '_p' for name in EXTRA_SURFACE_FIELDS if name != 'maxtkeprod_dd'})
NATIVE_ARRAY_BINDINGS['maxtkeprod'] = 'maxtkeprod_dd_p'


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
        'kzh','kzm','kzq','maxmf','maxwidth','ztop_plume','excess_h','excess_q', *EXTRA_SURFACE_FIELDS}
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
                  dimensions=context['dimensions'], extra_surface_fields=EXTRA_SURFACE_FIELDS, **tile)
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
