"""MYNN-EDMF v6.1.1 CUDA backend for the RRFS hrrrv5 route: the selectable suite version "mynn-v6.1".

Numerical text is generated from NCAR/MYNN-EDMF v6.1.1 (2b2da61b05a8ef9f9b06795c816dc01e347e0736), the four files
of its MPAS build (src/MPAS/module_bl_mynnedmf_common.F90, src/module_bl_mynnedmf_diags.F90,
src/module_bl_mynnedmf.F90, src/module_bl_mynnedmf_driver.F90, -Dmpas), including the complete unified driver, by
tests/rrfs_mynnedmf/generate_backend.py --layout v6. The pinned row (bl_mynnedmf_c6d08709, MYNN-EDMF c6d08709 from
RRFS 44c0849) is not changed by this module: it is a separate module, NVRTC unit, packet key and image, selected by
config_mynn_version = "mynn-v6.1" (data/profiles/mpas-physics-suites.v1.json).

The host is RRFS 44c0849; the v6 driver arguments are bound as the RRFS host that already calls the unified driver
does (ufs-community/MPAS-Model noaa/develop de016e0, mpas_atmphys_driver_pbl.F:1748-1828): pint = pres2_hyd_p.

Licence: MYNN-EDMF v6.1.1 is Copyright 2026, NOAA, UCAR/NCAR CU/CIRES, licensed under the Apache License,
Version 2.0 (LICENCE at 2b2da61; licenses/LICENSE-MYNN-EDMF-Apache-2.0.txt). This file is a translation of that work to CUDA.
Shared device maths is supplied by the caller. Forecast acceptance is separate.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target, image_target as _image_target
from woof.hex import numerics as _hex_numerics
from typing import Mapping

# Strict compiles optimised: --fmad=false keeps every operation separately rounded and --prec-div/--prec-sqrt keep
# IEEE quotients and roots, so ptxas optimisation changes no word (module replay of real hrrrv5 calls, strict, old
# -O0 image vs this one: 0 different words, rrfs/mynn-speed 2026-10-28). The -O0 image ran the PBL at 242 ms of a
# 322 ms strict step.
# ptxas splits its work across the host's cores; NVRTC does not.  THE BREAKAGE (woof hex 0.4.14 memory lane,
# 2026-11-06): with NVRTC's own --split-compile=0 (rrfs/speed2-mynn-edmf, 2026-10-28) every compile of this unit gave a
# different machine code -- five compiles of the same source and options (pack builder and runtime route, 8 and 48
# cores, and two on 8 cores) gave five different .text sections -- so a run that compiled it (no matching pack)
# differed from its own rerun on the pack image by 228,559 words at step 1 (8 ranks, fast).  Measured on box-4x with
# NVRTC 13.4.59 (fast): NVRTC split alone, 8 vs 48 cores: two different cubins; ptxas split alone: the same cubin
# byte for byte at 8 and 48 cores (105 s); no split: the same cubin (177 s).  So only ptxas splits.
CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true",
                "--ptxas-options=--split-compile=0")

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

// NCAR/MYNN-EDMF 2b2da61 src/module_bl_mynnedmf_diags.F90:15-100: an OPTIONAL scalar dummy that the routine writes
// (lwp1, iwp1, swp1, cldceil1, wspd*1, maxcldfra*1) refers to the caller's variable, or to nothing when absent.
template<class T> struct MynnOptionalRef {
    T* ptr = nullptr;
    __device__ MynnOptionalRef() = default;
    __device__ explicit MynnOptionalRef(T* p) : ptr(p) {}
    __device__ bool present() const { return ptr != nullptr; }
    __device__ operator T&() const { return *ptr; }
    __device__ MynnOptionalRef& operator=(T v) { *ptr = v; return *this; }
};
// NCAR/MYNN-EDMF 2b2da61 src/module_bl_mynnedmf_diags.F90:61: MAXVAL(ARRAY, MASK=...) as libgfortran gives it:
// the largest masked element ignoring NaNs, NaN when every masked element is NaN, -Infinity when none is masked.
template<class T,int R,class F> __device__ T mynn_maxval_mask(MynnArray<T,R> a, F mask) {
    static_assert(sizeof(T) == 4, "binary32 MAXVAL");
    T result = __int_as_float(0xff800000);
    bool any = false, number = false;
    for (int i=0; i<a.size(); ++i) {
        if (!mask(i)) continue;
        T x = a.linear(i);
        any = true;
        if (!number) { if (x == x) { result = x; number = true; } }
        else if (x > result) result = x;
    }
    return (any && !number) ? __int_as_float(0x7fc00000) : result;
}
'''

# BEGIN GENERATED RRFS MYNN-EDMF
MODULE_CUDA = r'''
// Generated from NCAR/MYNN-EDMF 2b2da61 (MPAS build, -Dmpas); see translation receipt.
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
static constexpr float tliq = 269.0f;
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
static constexpr float qmin = 0.0f;
static constexpr float zmax = 1.0f;
static constexpr float sqfac = 3.0f;
static constexpr float qkemin = 1.e-5f;
static constexpr float rr2 = 0.7071068f;
static constexpr float rrp = 0.3989423f;
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
static constexpr bool debug_code = false;
static constexpr int idbg = 452;
static constexpr int jdbg = 272;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:15-65
__device__ __noinline__ void f_mynnedmf_diags2d(int kts, int kte, MynnArray<float, 1> delp1, MynnArray<float, 1> dz1, MynnArray<float, 1> zw1, MynnArray<float, 1> zagl1, MynnArray<float, 1> u1, MynnArray<float, 1> v1, MynnArray<float, 1> tk1, MynnArray<float, 1> qc1, MynnArray<float, 1> qi1, MynnArray<float, 1> qs1, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> qi_bl1, MynnArray<float, 1> cldfra_bl1, MynnArray<float, 1> rho1, float xland1, float pblh1, MynnOptionalRef<float> lwp1, MynnOptionalRef<float> iwp1, MynnOptionalRef<float> swp1, MynnOptionalRef<float> cldceil1, MynnOptionalRef<float> wspd101, MynnOptionalRef<float> wspd801, MynnOptionalRef<float> wspd1601, MynnOptionalRef<float> maxcldfra1, MynnOptionalRef<float> maxcldfra_pbl1, int bl_mynn_diags2d) {
MynnFrame mynn_frame(arena);
delp1.rebase({kts});
dz1.rebase({kts});
zw1.rebase({kts});
zagl1.rebase({kts});
u1.rebase({kts});
v1.rebase({kts});
tk1.rebase({kts});
qc1.rebase({kts});
qi1.rebase({kts});
qs1.rebase({kts});
qc_bl1.rebase({kts});
qi_bl1.rebase({kts});
cldfra_bl1.rebase({kts});
rho1.rebase({kts});
MynnArray<float, 1> qctotal1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qitotal1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qstotal1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
int k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:36
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:37
if (((((qc1(k)) < (1e-6f))) && (((cldfra_bl1(k)) > (0.01f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:38
qctotal1(k) = qc_bl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:39
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:40
qctotal1(k) = qc1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:41
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:43
if (((((qi1(k)) < (1e-9f))) && (((cldfra_bl1(k)) > (0.01f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:44
qitotal1(k) = qi_bl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:45
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:46
qitotal1(k) = qi1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:47
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:49
qstotal1(k) = qs1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:50
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:52
f_cloud_water_path(kts, kte, delp1, qctotal1, qitotal1, qstotal1, lwp1, iwp1, swp1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:55
f_cloud_ceiling(kts, kte, dz1, zagl1, zw1, tk1, qctotal1, qitotal1, qstotal1, cldfra_bl1, rho1, xland1, cldceil1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:58
if (((bl_mynn_diags2d) >= (2))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:59
f_wspd_at_hgts(kts, kte, zagl1, u1, v1, wspd101, wspd801, wspd1601);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:61
maxcldfra1 = mynn_maxval(cldfra_bl1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:62
maxcldfra_pbl1 = mynn_maxval_mask(cldfra_bl1, [&] (int mynn_element) { return ((zagl1.linear(mynn_element)) < (r6_add(pblh1, 1000.f))); });
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:63
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:70-97
__device__ __noinline__ void f_cloud_water_path(int kts, int kte, MynnArray<float, 1> delp1, MynnArray<float, 1> qctotal1, MynnArray<float, 1> qitotal1, MynnArray<float, 1> qstotal1, MynnOptionalRef<float> lwp1, MynnOptionalRef<float> iwp1, MynnOptionalRef<float> swp1) {
MynnFrame mynn_frame(arena);
delp1.rebase({kts});
qctotal1.rebase({kts});
qitotal1.rebase({kts});
qstotal1.rebase({kts});
float sum1;
float sum2;
float sum3;
int k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:82
sum1 = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:83
sum2 = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:84
sum3 = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:86
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:87
sum1 = r6_add(sum1, mynn_max((float)(zero), (float)(r6_mul(r6_mul(delp1(k), g_inv), qctotal1(k)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:88
sum2 = r6_add(sum2, mynn_max((float)(zero), (float)(r6_mul(r6_mul(delp1(k), g_inv), qitotal1(k)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:89
sum3 = r6_add(sum3, mynn_max((float)(zero), (float)(r6_mul(r6_mul(delp1(k), g_inv), qstotal1(k)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:90
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:92
lwp1 = r6_mul(sum1, 1000.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:93
iwp1 = r6_mul(sum2, 1000.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:94
swp1 = r6_mul(sum3, 1000.f);
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:102-143
__device__ __noinline__ void f_wspd_at_hgts(int kts, int kte, MynnArray<float, 1> zagl1, MynnArray<float, 1> u1, MynnArray<float, 1> v1, float& wspd101, float& wspd801, float& wspd1601) {
MynnFrame mynn_frame(arena);
zagl1.rebase({kts});
u1.rebase({kts});
v1.rebase({kts});
float depth;
int k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:113
wspd101 = r6_neg(99.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:114
wspd801 = r6_neg(99.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:115
wspd1601 = r6_neg(99.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:116
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:117
depth = r6_sub(zagl1(((k) + (1))), zagl1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:121
if (((((zagl1(k)) > (10.f))) && (((wspd101) == (r6_neg(99.f)))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:122
wspd101 = f_interpolate_wind(k, kts, 10.0f, zagl1(k), u1(k), v1(k), u1(((k) - (1))), v1(((k) - (1))), depth);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:125
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:128
if (((((zagl1(k)) > (80.f))) && (((wspd801) == (r6_neg(99.f)))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:129
wspd801 = f_interpolate_wind(k, kts, 80.0f, zagl1(k), u1(k), v1(k), u1(((k) - (1))), v1(((k) - (1))), depth);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:132
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:135
if (((((zagl1(k)) > (160.f))) && (((wspd1601) == (r6_neg(99.f)))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:136
wspd1601 = f_interpolate_wind(k, kts, 160.0f, zagl1(k), u1(k), v1(k), u1(((k) - (1))), v1(((k) - (1))), depth);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:139
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:141
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:148-219
__device__ __noinline__ void f_cloud_ceiling(int kts, int kte, MynnArray<float, 1> dz1, MynnArray<float, 1> zagl1, MynnArray<float, 1> zw1, MynnArray<float, 1> tk1, MynnArray<float, 1> qctotal1, MynnArray<float, 1> qitotal1, MynnArray<float, 1> qstotal1, MynnArray<float, 1> cldfra_bl1, MynnArray<float, 1> rho1, float xland1, float& cldceil1) {
MynnFrame mynn_frame(arena);
dz1.rebase({kts});
zagl1.rebase({kts});
zw1.rebase({kts});
tk1.rebase({kts});
qctotal1.rebase({kts});
qitotal1.rebase({kts});
qstotal1.rebase({kts});
cldfra_bl1.rebase({kts});
rho1.rebase({kts});
float cld_depth;
float cldtau1;
float re_i;
float re_c;
float sum_cldtau;
bool check_fog_layer;
int k;
constexpr float cldtau_thld = 2.0f;
constexpr float min_cloud_dz = 40.0f;
constexpr float rho_w = 1000.0f;
constexpr float rho_i = 917.0f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:168
if (((xland1) == (1.0f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:169
re_c = 7.0e-6f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:170
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:171
re_c = 1.1e-5f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:172
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:174
sum_cldtau = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:175
check_fog_layer = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:176
cldceil1 = r6_neg(99.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:177
cld_depth = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:179
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:181
re_i = r6_add(173.46f, r6_mul(2.14f, r6_sub(tk1(k), 273.15f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:182
re_i = r6_mul(re_i, 1e-6f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:184
cldtau1 = r6_add(r6_div(r6_mul(r6_mul(r6_mul(1.5f, qctotal1(k)), rho1(k)), dz1(k)), r6_mul(rho_w, re_c)), r6_div(r6_mul(r6_mul(r6_mul(1.5f, r6_add(qitotal1(k), qstotal1(k))), rho1(k)), dz1(k)), r6_mul(rho_i, re_i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:188
if (((((cldtau1) > (zero))) && (((cldfra_bl1(k)) >= (0.5f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:189
cld_depth = r6_add(cld_depth, dz1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:190
sum_cldtau = r6_add(sum_cldtau, cldtau1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:192
if (((((k) == (1))) || (((k) == (2))))) { check_fog_layer = true; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:193
if (((cld_depth) >= (min_cloud_dz))) { check_fog_layer = false; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:195
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:196
if (((check_fog_layer) && (((cld_depth) < (min_cloud_dz))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:197
sum_cldtau = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:198
cld_depth = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:199
check_fog_layer = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:200
continue;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:201
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:204
sum_cldtau = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:205
cld_depth = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:206
check_fog_layer = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:207
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:209
if ((!(check_fog_layer))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:210
if (((((sum_cldtau) >= (cldtau_thld))) && (((cldfra_bl1(k)) >= (0.5f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:211
cldceil1 = r6_add(zw1(k), r6_mul(r6_div(r6_sub(sum_cldtau, cldtau_thld), cldtau1), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:213
break;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:214
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:215
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:217
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:221-243
__device__ __noinline__ float f_interpolate_wind(int k_idx, int kts, float target_z, float zagl, float u1_curr, float v1_curr, float u1_prev, float v1_prev, float depth) {
MynnFrame mynn_frame(arena);
float wspd1_hgt;
float wgt;
float wspd1_curr;
float wspd1_prev;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:232
wspd1_curr = mynn_sqrt(r6_add(mynn_ipow<2>(u1_curr), mynn_ipow<2>(v1_curr)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:234
if (((k_idx) == (kts))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:235
wspd1_hgt = wspd1_curr;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:236
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:237
wgt = r6_div(r6_sub(zagl, target_z), depth);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:238
wgt = mynn_min((float)(mynn_max((float)(zero), (float)(wgt))), (float)(1.0f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:239
wspd1_prev = mynn_sqrt(r6_add(mynn_ipow<2>(u1_prev), mynn_ipow<2>(v1_prev)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:240
wspd1_hgt = r6_add(r6_mul(r6_sub(1.0f, wgt), wspd1_curr), r6_mul(wgt, wspd1_prev));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_diags.F90:241
}
return wspd1_hgt;
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:157-1069
__device__ __forceinline__ void f_mynnedmf(int i, int j, int initflag, bool restart, bool cycling, float delt, MynnArray<float, 1> dz1, float dx, MynnArray<float, 1> u1, MynnArray<float, 1> v1, MynnArray<float, 1> w1, MynnArray<float, 1> th1, MynnArray<float, 1> sqv1, MynnArray<float, 1> sqc1, MynnArray<float, 1> sqi1, MynnArray<float, 1> sqs1, MynnArray<float, 1> qv1, MynnArray<float, 1> qc1, MynnArray<float, 1> qi1, MynnArray<float, 1> qs1, MynnArray<float, 1> qnc1, MynnArray<float, 1> qni1, MynnArray<float, 1> qnwfa1, MynnArray<float, 1> qnifa1, MynnArray<float, 1> qnbca1, MynnArray<float, 1> ozone1, MynnArray<float, 1> pres1, MynnArray<float, 1> ex1, MynnArray<float, 1> rho1, MynnArray<float, 1> tk1, MynnArray<float, 1> delp1, MynnArray<float, 1> zw1, MynnArray<float, 1> zagl1, float xland, float ts, float qsfc, float ps, float ust, float ch, float hfx, float qfx, float znt, float wspd, float uoce, float voce, MynnArray<float, 1> qke1, MynnArray<float, 1> qke_adv1, MynnArray<float, 1> el1, MynnArray<float, 1> sh1, MynnArray<float, 1> sm1, MynnArray<float, 1> kh1, MynnArray<float, 1> km1, int nchem, int ndvel, MynnArray<float, 2> chem1, MynnArray<float, 1> vdep, float frp, float emis_ant_no, bool mix_chem, bool enh_mix, MynnArray<float, 2> settle1, MynnArray<float, 2> scalars, int nscalars, MynnArray<float, 1> tsq1, MynnArray<float, 1> qsq1, MynnArray<float, 1> cov1, MynnArray<float, 1> du1, MynnArray<float, 1> dv1, MynnArray<float, 1> dth1, MynnArray<float, 1> dqv1, MynnArray<float, 1> dqc1, MynnArray<float, 1> dqi1, MynnArray<float, 1> dqnc1, MynnArray<float, 1> dqni1, MynnArray<float, 1> dqs1, MynnArray<float, 1> dqnwfa1, MynnArray<float, 1> dqnifa1, MynnArray<float, 1> dqnbca1, MynnArray<float, 1> dozone1, MynnArray<float, 1> rthraten1, float& pblh, int& kpbl, float& maxwidth_dd, float& maxwidth, float& maxmf, float& ztop_plume, float& excess_h, float& excess_q, float& maxmf_dd, float& maxtkeprod, float& cldtop_cooling, float& ent_eff, MynnArray<float, 1> dqke1, MynnArray<float, 1> qwt1, MynnArray<float, 1> qshear1, MynnArray<float, 1> qbuoy1, MynnArray<float, 1> qdiss1, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> qi_bl1, MynnArray<float, 1> cldfra_bl1, bool bl_mynn_tkeadvect, int tke_budget, int bl_mynn_cloudpdf, int bl_mynn_mixlength, float bl_mynn_closure, int bl_mynn_ess, int bl_mynn_edmf, int bl_mynn_edmf_mom, int bl_mynn_edmf_tke, int bl_mynn_mixscalars, int bl_mynn_mixaerosols, int bl_mynn_mixnumcon, int bl_mynn_diags3d, int bl_mynn_cloudmix, int bl_mynn_mixqt, int bl_mynn_edmf_dd, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_qt1, MynnArray<float, 1> edmf_thl1, MynnArray<float, 1> edmf_ent1, MynnArray<float, 1> edmf_qc1, MynnArray<float, 1> sub_thl1, MynnArray<float, 1> sub_sqv1, MynnArray<float, 1> det_thl1, MynnArray<float, 1> det_sqv1, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1, bool flag_qc, bool flag_qi, bool flag_qnc, bool flag_qni, bool flag_qs, bool flag_qnwfa, bool flag_qnifa, bool flag_qnbca, bool flag_ozone, int kts, int kte, MynnText& errmsg, int& errflg) {
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
qv1.rebase({kts});
qc1.rebase({kts});
qi1.rebase({kts});
qs1.rebase({kts});
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
delp1.rebase({kts});
zw1.rebase({kts});
zagl1.rebase({kts});
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
MynnArray<float, 1> s_awqi1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
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
int ktop_plume;
MynnArray<float, 1> tkeprod_dn = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> tkeprod_up = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
constexpr int bl_mynn_stfunc = 0;
constexpr float ckmod = 1.0f;
constexpr float taue = 80.f;
constexpr float upwind = 1.0f;
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
bool initialize_qke;
bool problem;
int nsub;
float delt2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:390
errmsg = " ";
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:391
errflg = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:393
if (debug_code) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:394
problem = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:395
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:396
wsp = mynn_sqrt(r6_add(mynn_ipow<2>(u1(k)), mynn_ipow<2>(v1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:397
if (((((((((((((((((mynn_abs(hfx)) > (1200.f))) || (((mynn_abs(qfx)) > (0.001f))))) || (((wsp) > (270.f))))) || (((tk1(k)) > (380.f))))) || (((tk1(k)) < (160.f))))) || (((sqv1(k)) < (zero))))) || (((sqc1(k)) < (zero))))) || (((r6_sub(ps, pres1(kts))) < (zero))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:401
kproblem = k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:402
problem = true;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:403
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:404
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:405
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:406
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:407
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:408
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:409
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:410
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:411
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:412
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:413
if (problem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:414
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:415
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:416
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:417
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:418
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:419
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:420
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:421
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:422
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:425
imd = 10;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:426
jmd = 10;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:430
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1).size(); mynn_element += lane_width) { edmf_a1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:431
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1).size(); mynn_element += lane_width) { edmf_w1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:432
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt1).size(); mynn_element += lane_width) { edmf_qt1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:433
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1).size(); mynn_element += lane_width) { edmf_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:434
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent1).size(); mynn_element += lane_width) { edmf_ent1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:435
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1).size(); mynn_element += lane_width) { edmf_qc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:436
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_u1).size(); mynn_element += lane_width) { edmf_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:437
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_v1).size(); mynn_element += lane_width) { edmf_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:438
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qv1).size(); mynn_element += lane_width) { edmf_qv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:439
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_thl1).size(); mynn_element += lane_width) { sub_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:440
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_sqv1).size(); mynn_element += lane_width) { sub_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:441
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_thl1).size(); mynn_element += lane_width) { det_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:442
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqv1).size(); mynn_element += lane_width) { det_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:443
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a_dd1).size(); mynn_element += lane_width) { edmf_a_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:444
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w_dd1).size(); mynn_element += lane_width) { edmf_w_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:445
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc_dd1).size(); mynn_element += lane_width) { edmf_qc_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:447
ktop_plume = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:448
ztop_plume = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:449
maxwidth = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:450
maxmf = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:451
excess_h = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:452
excess_q = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:453
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (kzero1).size(); mynn_element += lane_width) { kzero1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:460
if (((((initflag) > (0))) && ((!(restart))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:463
if (((restart) || (cycling))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:464
if (((mynn_maxval(qke1.slice(MynnRange{qke1.lo[0], qke1.hi[0]}))) < (0.0002f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:465
initialize_qke = true;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:467
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:468
initialize_qke = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:470
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:471
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:472
initialize_qke = true;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:474
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:476
if ((((!(restart))) || ((!(cycling))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:477
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sh1).size(); mynn_element += lane_width) { sh1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:478
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sm1).size(); mynn_element += lane_width) { sm1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:479
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (el1).size(); mynn_element += lane_width) { el1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:480
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tsq1).size(); mynn_element += lane_width) { tsq1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:481
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qsq1).size(); mynn_element += lane_width) { qsq1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:482
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cov1).size(); mynn_element += lane_width) { cov1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:483
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1).size(); mynn_element += lane_width) { cldfra_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:484
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1).size(); mynn_element += lane_width) { qc_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:485
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi_bl1).size(); mynn_element += lane_width) { qi_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:486
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke1).size(); mynn_element += lane_width) { qke1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:487
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke_adv1).size(); mynn_element += lane_width) { qke_adv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:488
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:489
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqc1).size(); mynn_element += lane_width) { dqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:490
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqi1).size(); mynn_element += lane_width) { dqi1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:491
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqni1).size(); mynn_element += lane_width) { dqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:492
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnc1).size(); mynn_element += lane_width) { dqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:493
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnwfa1).size(); mynn_element += lane_width) { dqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:494
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnifa1).size(); mynn_element += lane_width) { dqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:495
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnbca1).size(); mynn_element += lane_width) { dqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:496
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dozone1).size(); mynn_element += lane_width) { dozone1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:497
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1_old).size(); mynn_element += lane_width) { qc_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:498
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1_old).size(); mynn_element += lane_width) { cldfra_bl1_old.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:499
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sgm1).size(); mynn_element += lane_width) { sgm1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:500
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (vt1).size(); mynn_element += lane_width) { vt1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:501
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (vq1).size(); mynn_element += lane_width) { vq1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:502
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (km1).size(); mynn_element += lane_width) { km1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:503
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (kh1).size(); mynn_element += lane_width) { kh1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:505
if (((tke_budget) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:506
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qwt1).size(); mynn_element += lane_width) { qwt1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:507
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qshear1).size(); mynn_element += lane_width) { qshear1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:508
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qbuoy1).size(); mynn_element += lane_width) { qbuoy1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:509
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qdiss1).size(); mynn_element += lane_width) { qdiss1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:510
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqke1).size(); mynn_element += lane_width) { dqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:511
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:513
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:515
sqw1(k) = r6_add(r6_add(sqv1(k), sqc1(k)), sqi1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:516
thl1(k) = r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), sqc1(k))), r6_mul(r6_div(xlscp, ex1(k)), sqi1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:522
thlv1(k) = r6_mul(thl1(k), r6_add(one, r6_mul(p608, sqv1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:523
thv1(k) = r6_mul(th1(k), r6_sub(r6_add(one, r6_mul(p608, sqv1(k))), r6_add(sqc1(k), sqi1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:524
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:526
if (initialize_qke) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:530
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:531
qke1(k) = r6_mul(r6_mul(five, ust), mynn_max((float)(r6_div(r6_sub(r6_mul(ust, 700.f), zw1(k)), r6_mul(mynn_max((float)(ust), (float)(p01)), 700.f))), (float)(p01)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:532
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:533
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:537
f_get_pblh(kts, kte, pblh, thv1, qke1, ust, zw1, dz1, xland, kpbl);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:541
f_scale_aware(dx, pblh, psig_bl, psig_shcu);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:548
rmol = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:549
rmolh = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:550
f_mym_initialize(kts, kte, ckmod, taue, xland, pblh, dz1, dx, zw1, pres1, ex1, u1, v1, thl1, sqv1, th1, thv1, thlv1, sh1, sm1, ust, rmol, rmolh, el1, qke1, tsq1, qsq1, cov1, psig_bl, cldfra_bl1, bl_mynn_mixlength, bl_mynn_ess, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1, initialize_qke, spp_pbl, pattern_spp_pbl1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:567
if ((!(restart))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:569
if (bl_mynn_tkeadvect) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:570
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:571
qke_adv1(k) = qke1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:572
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:573
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:574
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:586
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:590
if (bl_mynn_tkeadvect) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:591
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qke1.slice(MynnRange{kts, kte}).linear(mynn_element) = qke_adv1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:592
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:595
if (((tke_budget) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:596
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqke1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { dqke1.slice(MynnRange{kts, kte}).linear(mynn_element) = qke1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:597
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:598
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1_old.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { cldfra_bl1_old.slice(MynnRange{kts, kte}).linear(mynn_element) = cldfra_bl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:599
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1_old.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qc_bl1_old.slice(MynnRange{kts, kte}).linear(mynn_element) = qc_bl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:600
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi_bl1_old.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qi_bl1_old.slice(MynnRange{kts, kte}).linear(mynn_element) = qi_bl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:601
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (cldfra_bl1).size(); mynn_element += lane_width) { cldfra_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:602
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc_bl1).size(); mynn_element += lane_width) { qc_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:603
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi_bl1).size(); mynn_element += lane_width) { qi_bl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:604
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqc1).size(); mynn_element += lane_width) { dqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:605
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqi1).size(); mynn_element += lane_width) { dqi1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:606
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqs1).size(); mynn_element += lane_width) { dqs1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:607
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqni1).size(); mynn_element += lane_width) { dqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:608
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnc1).size(); mynn_element += lane_width) { dqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:609
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnwfa1).size(); mynn_element += lane_width) { dqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:610
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnifa1).size(); mynn_element += lane_width) { dqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:611
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dqnbca1).size(); mynn_element += lane_width) { dqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:612
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dozone1).size(); mynn_element += lane_width) { dozone1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:614
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1).size(); mynn_element += lane_width) { edmf_a1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:615
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1).size(); mynn_element += lane_width) { edmf_w1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:616
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1).size(); mynn_element += lane_width) { edmf_qc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:617
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qv1).size(); mynn_element += lane_width) { edmf_qv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:618
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1).size(); mynn_element += lane_width) { edmf_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:619
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_u1).size(); mynn_element += lane_width) { edmf_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:620
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_v1).size(); mynn_element += lane_width) { edmf_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:621
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_aw1).size(); mynn_element += lane_width) { s_aw1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:622
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awthl1).size(); mynn_element += lane_width) { s_awthl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:623
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqt1).size(); mynn_element += lane_width) { s_awqt1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:624
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqv1).size(); mynn_element += lane_width) { s_awqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:625
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqc1).size(); mynn_element += lane_width) { s_awqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:626
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqi1).size(); mynn_element += lane_width) { s_awqi1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:627
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awu1).size(); mynn_element += lane_width) { s_awu1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:628
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awv1).size(); mynn_element += lane_width) { s_awv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:629
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqke1).size(); mynn_element += lane_width) { s_awqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:630
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqsq1).size(); mynn_element += lane_width) { s_awqsq1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:631
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnc1).size(); mynn_element += lane_width) { s_awqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:632
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqni1).size(); mynn_element += lane_width) { s_awqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:633
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnwfa1).size(); mynn_element += lane_width) { s_awqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:634
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnifa1).size(); mynn_element += lane_width) { s_awqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:635
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnbca1).size(); mynn_element += lane_width) { s_awqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:636
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:637
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars})).size(); mynn_element += lane_width) { s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars}).linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:639
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a_dd1).size(); mynn_element += lane_width) { edmf_a_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:640
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w_dd1).size(); mynn_element += lane_width) { edmf_w_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:641
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc_dd1).size(); mynn_element += lane_width) { edmf_qc_dd1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:642
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_aw1).size(); mynn_element += lane_width) { sd_aw1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:643
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awthl1).size(); mynn_element += lane_width) { sd_awthl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:644
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqt1).size(); mynn_element += lane_width) { sd_awqt1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:645
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqv1).size(); mynn_element += lane_width) { sd_awqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:646
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqc1).size(); mynn_element += lane_width) { sd_awqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:647
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqi1).size(); mynn_element += lane_width) { sd_awqi1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:648
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnc1).size(); mynn_element += lane_width) { sd_awqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:649
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqni1).size(); mynn_element += lane_width) { sd_awqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:650
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnwfa1).size(); mynn_element += lane_width) { sd_awqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:651
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnifa1).size(); mynn_element += lane_width) { sd_awqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:652
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awu1).size(); mynn_element += lane_width) { sd_awu1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:653
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awv1).size(); mynn_element += lane_width) { sd_awv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:654
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqke1).size(); mynn_element += lane_width) { sd_awqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:655
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_thl1).size(); mynn_element += lane_width) { sub_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:656
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_sqv1).size(); mynn_element += lane_width) { sub_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:657
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_u1).size(); mynn_element += lane_width) { sub_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:658
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_v1).size(); mynn_element += lane_width) { sub_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:659
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_thl1).size(); mynn_element += lane_width) { det_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:660
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqv1).size(); mynn_element += lane_width) { det_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:661
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqc1).size(); mynn_element += lane_width) { det_sqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:662
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_u1).size(); mynn_element += lane_width) { det_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:663
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_v1).size(); mynn_element += lane_width) { det_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:665
{ const int mynn_lo3 = (kts), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:667
sqw1(k) = r6_add(r6_add(sqv1(k), sqc1(k)), sqi1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:668
thl1(k) = r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), sqc1(k))), r6_mul(r6_div(xlscp, ex1(k)), sqi1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:674
thv1(k) = r6_mul(th1(k), r6_sub(r6_add(one, r6_mul(p608, sqv1(k))), r6_add(sqc1(k), sqi1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:675
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:679
f_get_pblh(kts, kte, pblh, thv1, qke1, ust, zw1, dz1, xland, kpbl);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:685
f_scale_aware(dx, pblh, psig_bl, psig_shcu);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:687
sqcg = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:688
cpm = r6_mul(cp, r6_add(one, r6_mul(0.84f, mynn_max((float)(sqv1(kts)), (float)(1e-8f)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:689
exnerg = mynn_pow(r6_div(ps, p1000mb), rcp);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:698
flqv = r6_div(qfx, rho1(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:699
flqc = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:700
th_sfc = r6_div(ts, exnerg);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:703
flq = r6_add(flqv, flqc);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:704
flt = r6_sub(r6_div(hfx, r6_mul(rho1(kts), cpm)), r6_div(r6_mul(xlvcp, flqc), ex1(kts)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:705
fltv = r6_add(flt, r6_mul(r6_mul(flqv, p608), th_sfc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:708
rmol = r6_div(r6_mul(r6_neg(r6_mul(karman, gtr)), fltv), mynn_max((float)(mynn_ipow<3>(ust)), (float)(1.0e-6f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:709
rmolh = r6_div(r6_mul(r6_neg(r6_mul(karman, gtr)), flt), mynn_max((float)(mynn_ipow<3>(ust)), (float)(1.0e-6f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:710
zet = r6_mul(r6_mul(p5, dz1(kts)), rmol);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:711
zet = mynn_max((float)(zet), (float)(r6_neg(10.f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:712
zet = mynn_min((float)(zet), (float)(10.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:714
if (((bl_mynn_stfunc) == (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:716
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:717
if (((zet) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:719
pmz = r6_add(one, r6_mul(3.0f, zet));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:720
phh = r6_add(one, r6_mul(cphh_st, zet));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:721
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:722
pmz = r6_sub(r6_div(one, mynn_pow(r6_sub(one, r6_mul(cphm_unst, zet)), 0.25f)), zet);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:723
phh = r6_div(one, mynn_sqrt(r6_sub(one, r6_mul(cphh_unst, zet))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:724
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:725
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:726
if (((zet) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:728
pmz = r6_add(0.95f, r6_mul(3.5f, zet));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:729
phh = r6_add(one, r6_mul(cphh_st, zet));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:730
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:732
pmz = r6_sub(r6_div(0.95f, mynn_pow(r6_sub(one, r6_mul(cphm_unst, zet)), 0.25f)), zet);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:733
phh = r6_div(one, mynn_sqrt(r6_sub(one, r6_mul(cphh_unst, zet))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:734
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:735
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:736
phi_m = f_phim(zet);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:737
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:739
phi_m = f_phim(zet);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:740
pmz = r6_sub(phi_m, zet);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:741
phh = f_phih(zet);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:742
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:749
f_mym_condensation(kts, kte, dx, dz1, zagl1, zw1, xland, thl1, sqw1, sqv1, sqc1, sqi1, sqs1, pres1, ex1, tsq1, qsq1, cov1, sh1, el1, bl_mynn_cloudpdf, qc_bl1, qi_bl1, cldfra_bl1, pblh, hfx, vt1, vq1, th1, sgm1, bl_mynn_closure, spp_pbl, pattern_spp_pbl1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:761
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprod_up).size(); mynn_element += lane_width) { tkeprod_up.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:762
if (((bl_mynn_edmf) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:764
f_dmp_mf(i, j, kts, kte, delt, zw1, dz1, zagl1, pres1, rho1, bl_mynn_edmf_mom, bl_mynn_edmf_tke, bl_mynn_mixscalars, bl_mynn_mixaerosols, bl_mynn_mixnumcon, bl_mynn_closure, u1, v1, w1, th1, thl1, thv1, tk1, sqw1, sqv1, sqc1, sqi1, qke1, qsq1, qnc1, qni1, qnwfa1, qnifa1, qnbca1, ex1, vt1, vq1, sgm1, ust, flt, fltv, flq, flqv, pblh, kpbl, dx, xland, th_sfc, edmf_a1, edmf_w1, edmf_qt1, edmf_thl1, edmf_ent1, edmf_qc1, edmf_qv1, edmf_u1, edmf_v1, s_aw1, s_awthl1, s_awqt1, s_awqv1, s_awqc1, s_awqi1, s_awu1, s_awv1, s_awqke1, s_awqsq1, s_awqnc1, s_awqni1, s_awqnwfa1, s_awqnifa1, s_awqnbca1, sub_thl1, sub_sqv1, sub_u1, sub_v1, det_thl1, det_sqv1, det_sqc1, det_u1, det_v1, nchem, chem1, s_awchem1, mix_chem, nscalars, scalars, s_awscalars1, qc_bl1, cldfra_bl1, qc_bl1_old, cldfra_bl1_old, flag_qc, flag_qi, flag_qnc, flag_qni, flag_qnwfa, flag_qnifa, flag_qnbca, psig_shcu, maxwidth, ktop_plume, maxmf, ztop_plume, excess_h, excess_q, spp_pbl, pattern_spp_pbl1, tkeprod_up, el1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:809
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:815
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprod_dn).size(); mynn_element += lane_width) { tkeprod_dn.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:816
maxtkeprod = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:817
cldtop_cooling = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:818
ent_eff = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:819
maxmf_dd = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:820
maxwidth_dd = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:821
if (((bl_mynn_edmf_dd) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:822
f_ddmp_mf(kts, kte, delt, dx, zw1, dz1, pres1, delp1, u1, v1, th1, thl1, thv1, tk1, sqw1, sqv1, sqc1, sqi1, qnc1, qni1, qnwfa1, qnifa1, qke1, rho1, ex1, qc_bl1, qi_bl1, cldfra_bl1, ust, flt, flq, fltv, pblh, kpbl, edmf_a_dd1, edmf_w_dd1, edmf_qt_dd1, edmf_thl_dd1, edmf_ent_dd1, edmf_qc_dd1, sd_aw1, sd_awthl1, sd_awqt1, sd_awqv1, sd_awqc1, sd_awqi1, sd_awqnc1, sd_awqni1, sd_awqnwfa1, sd_awqnifa1, sd_awu1, sd_awv1, sd_awqke1, tkeprod_dn, el1, rthraten1, psig_bl, maxmf_dd, maxwidth_dd);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:846
f_topdown_cloudrad(kts, kte, dz1, zw1, zagl1, fltv, u1(kts), v1(kts), xland, kpbl, pblh, sqc1, sqi1, sqw1, thl1, th1, ex1, pres1, rho1, thv1, cldfra_bl1, qc_bl1, qi_bl1, rthraten1, tkeprod_dn, psig_bl, maxtkeprod, cldtop_cooling, ent_eff);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:854
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:858
delt2 = delt;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:861
{ const int mynn_lo4 = (kts), mynn_hi4 = (kte), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:862
qc_tot1(k) = mynn_max((float)(qc_bl1(k)), (float)(sqc1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:863
qi_tot1(k) = mynn_max((float)(qi_bl1(k)), (float)(r6_add(sqi1(k), sqs1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:864
thlv1(k) = r6_mul(r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), qc_tot1(k))), r6_mul(r6_div(xlscp, ex1(k)), qi_tot1(k))), r6_add(one, r6_mul(p608, sqv1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:867
thl_tot1(k) = r6_sub(r6_sub(th1(k), r6_mul(r6_div(xlvcp, ex1(k)), qc_tot1(k))), r6_mul(r6_div(xlscp, ex1(k)), qi_tot1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:869
thv1(k) = r6_mul(th1(k), r6_sub(r6_add(one, r6_mul(p608, sqv1(k))), r6_add(qc_tot1(k), qi_tot1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:870
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:872
f_mym_turbulence(kts, kte, xland, bl_mynn_closure, ckmod, taue, dz1, dx, zw1, pres1, ex1, u1, v1, thl1, thv1, thlv1, sqc1, sqw1, qke1, tsq1, qsq1, cov1, vt1, vq1, rmol, rmolh, flt, fltv, flq, pblh, th1, sh1, sm1, el1, dfm1, dfh1, dfq1, tcd1, qcd1, pdk1, pdt1, pdq1, pdc1, qwt1, qshear1, qbuoy1, qdiss1, tke_budget, psig_bl, psig_shcu, cldfra_bl1, bl_mynn_mixlength, bl_mynn_ess, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1, tkeprod_dn, tkeprod_up, spp_pbl, pattern_spp_pbl1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:898
f_mym_predict(kts, kte, bl_mynn_closure, delt2, dz1, ust, flt, flq, pmz, phh, el1, dfq1, rho1, pdk1, pdt1, pdq1, pdc1, qke1, tsq1, qsq1, cov1, s_aw1, s_awqke1, s_awqsq1, bl_mynn_edmf, bl_mynn_edmf_tke, qwt1, qdiss1, tke_budget, upwind);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:909
{ const int mynn_lo5 = (kts), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:910
diss_heat1(k) = mynn_min((float)(mynn_max((float)(zero), (float)(r6_div(r6_div(mynn_pow(qke1(k), 1.5f), r6_mul(b1, mynn_max((float)(r6_mul(p5, r6_add(el1(k), el1(((k) + (1)))))), (float)(one)))), cp)))), (float)(0.002f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:911
diss_heat1(k) = r6_mul(diss_heat1(k), mynn_exp(r6_neg(r6_div(10000.f, mynn_max((float)(pres1(k)), (float)(one))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:912
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:913
diss_heat1(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:917
f_mynn_tendencies(kts, kte, i, delt, dz1, zw1, xland, pblh, rho1, delp1, u1, v1, th1, tk1, qv1, qc1, qi1, kzero1, qnc1, qni1, ps, pres1, ex1, thl1, sqv1, sqc1, sqi1, kzero1, sqw1, thl_tot1, qc_tot1, qi_tot1, qnwfa1, qnifa1, qnbca1, ozone1, ust, flt, flq, flqv, flqc, wspd, uoce, voce, tsq1, qsq1, cov1, tcd1, qcd1, dfm1, dfh1, dfq1, du1, dv1, dth1, dqv1, dqc1, dqi1, dqs1, dqnc1, dqni1, dqnwfa1, dqnifa1, dqnbca1, dozone1, diss_heat1, s_aw1, s_awthl1, s_awqt1, s_awqv1, s_awqc1, s_awqi1, s_awu1, s_awv1, s_awqnc1, s_awqni1, s_awqnwfa1, s_awqnifa1, s_awqnbca1, sd_aw1, sd_awthl1, sd_awqt1, sd_awqv1, sd_awqc1, sd_awqi1, sd_awqnc1, sd_awqni1, sd_awqnwfa1, sd_awqnifa1, sd_awu1, sd_awv1, sub_thl1, sub_sqv1, sub_u1, sub_v1, det_thl1, det_sqv1, det_sqc1, det_u1, det_v1, flag_qc, flag_qi, flag_qnc, flag_qni, flag_qs, flag_qnwfa, flag_qnifa, flag_qnbca, flag_ozone, cldfra_bl1, bl_mynn_cloudmix, bl_mynn_mixqt, bl_mynn_edmf, bl_mynn_edmf_mom, bl_mynn_mixscalars, bl_mynn_mixaerosols, bl_mynn_mixnumcon, upwind);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:963
if (mix_chem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:964
f_mynn_mix_chem(kts, kte, i, delt, dz1, pblh, nchem, ndvel, chem1, vdep, settle1, rho1, flt, tcd1, qcd1, dfh1, s_aw1, s_awchem1, emis_ant_no, frp, enh_mix, bl_mynn_edmf, upwind);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:976
{ const int mynn_lo6 = (1), mynn_hi6 = (nchem), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { ic = mynn_lo6 + mynn_it6*mynn_st6;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:977
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:978
chem1(k, ic) = mynn_max((float)(chem1(k, ic)), (float)(1.e-12f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:979
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:980
} mynn_lane_sync(); ic = mynn_lo6 + mynn_n6*mynn_st6; if (mynn_n6 > 0) { const int mynn_src6 = (mynn_n6 - 1) % lane_width; k = mynn_lane_bcast(k, mynn_src6); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:981
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:983
if (((bl_mynn_mixscalars) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:984
f_mynn_mix_scalars(kts, kte, i, delt, dz1, nscalars, scalars, rho1, flt, tcd1, qcd1, dfh1, s_aw1, s_awscalars1, bl_mynn_edmf, upwind);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:992
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:995
f_retrieve_exchange_coeffs(kts, kte, dfm1, dfh1, dz1, km1, kh1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:999
if (((tke_budget) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1002
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1003
qshear1(k) = r6_sub(r6_mul(4.f, r6_div(r6_mul(mynn_ipow<3>(ust), phi_m), r6_mul(karman, dz1(k)))), qshear1(((k) + (1))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1004
qbuoy1(k) = r6_sub(r6_mul(4.f, r6_neg(r6_div(r6_mul(mynn_ipow<3>(ust), zet), r6_mul(karman, dz1(k))))), qbuoy1(((k) + (1))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1006
{ const int mynn_lo7 = (kts), mynn_hi7 = (((kte) - (1))), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1007
dummy1(k) = r6_mul(p5, r6_add(qshear1(k), qshear1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1008
dummy2(k) = r6_mul(p5, r6_add(qbuoy1(k), qbuoy1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1009
dqke1(k) = r6_div(r6_mul(p5, r6_sub(qke1(k), dqke1(k))), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1010
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1011
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qshear1).size(); mynn_element += lane_width) { qshear1.linear(mynn_element) = dummy1.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1012
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qbuoy1).size(); mynn_element += lane_width) { qbuoy1.linear(mynn_element) = dummy2.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1014
k = kte;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1015
qshear1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1016
qbuoy1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1017
qwt1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1018
qdiss1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1019
dqke1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1020
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1023
if (((bl_mynn_diags3d) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1025
if (((((bl_mynn_diags3d) == (2))) && (((bl_mynn_edmf_dd) == (1))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1026
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_a1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_a_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1027
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_w1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_w_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1028
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_qt1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_qt_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1029
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_thl1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_thl_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1030
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_ent1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_ent_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1031
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { edmf_qc1.slice(MynnRange{kts, kte}).linear(mynn_element) = edmf_qc_dd1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1032
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1033
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1036
if (((debug_code) && (((i) == (idbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1037
if (((mynn_abs(qfx)) > (.001f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1039
if (((mynn_abs(hfx)) > (1100.f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1041
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1042
if (((((sh1(k)) < (0.f))) || (((sh1(k)) > (200.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1044
if (((mynn_abs(vt1(k))) > (2.0f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1046
if (((mynn_abs(vq1(k))) > (7000.f))) { ; /* Native diagnostic print; no field arithmetic. */ }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1048
if (((((qke1(k)) < (r6_neg(1.f)))) || (((qke1(k)) > (200.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1050
if (((((el1(k)) < (0.f))) || (((el1(k)) > (1500.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1052
if (((((km1(k)) < (0.f))) || (((km1(k)) > (2000.f))))) { ; /* Native diagnostic print; no field arithmetic. */ }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1054
if (((((cldfra_bl1(k)) < (zero))) || (((cldfra_bl1(k)) > (one))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1055
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1057
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1058
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1059
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1062
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke_adv1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qke_adv1.slice(MynnRange{kts, kte}).linear(mynn_element) = qke1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1130-1303
__device__ __noinline__ void f_mym_initialize(int kts, int kte, float ckmod, float taue, float xland, float pblh, MynnArray<float, 1> dz, float dx, MynnArray<float, 1> zw, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> thl, MynnArray<float, 1> qw, MynnArray<float, 1> theta, MynnArray<float, 1> thv, MynnArray<float, 1> thlv, MynnArray<float, 1> sh, MynnArray<float, 1> sm, float ust, float rmol, float rmolh, MynnArray<float, 1> el, MynnArray<float, 1> qke, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, float psig_bl, MynnArray<float, 1> cldfra_bl1, int bl_mynn_mixlength, int bl_mynn_ess, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> edmf_w_dd1, MynnArray<float, 1> edmf_a_dd1, bool initialize_qke, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1) {
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1177
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1178
ql(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1179
vt(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1180
vq(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1181
qkw(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1182
qtw(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1183
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1184
ustm = mynn_max((float)(ust), (float)(p01));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1187
f_mym_level2(kts, kte, ckmod, bl_mynn_ess, zw, dz, xland, pblh, u, v, thl, thv, thlv, theta, p, exner, qke, cldfra_bl1, edmf_a1, edmf_w1, qw, ql, vt, vq, dtl, dqw, dtv, gm, gh, sm, sh);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1199
el(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1200
if (initialize_qke) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1202
qke(kts) = r6_mul(r6_mul(1.5f, mynn_ipow<2>(ustm)), mynn_pow(r6_mul(b1, pmz), p666));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1203
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1206
qke(k) = r6_mul(qke(kts), mynn_max((float)(r6_div(r6_sub(r6_mul(ust, 700.f), zw(k)), r6_mul(ustm, 700.f))), (float)(p01)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1207
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1208
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1210
phm = r6_div(r6_mul(phh, b2), mynn_pow(r6_mul(b1, pmz), p333));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1211
tsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flt)), ustm)))), (float)(three));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1212
qsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flq)), ustm)))), (float)(5e-6f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1213
cov(kts) = r6_mul(r6_mul(phm, r6_div(flt, ustm)), r6_div(flq, ustm));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1214
qpe(kts) = mynn_min((float)(mynn_max((float)(r6_mul(r6_mul(r6_mul(two, tsq(kts)), mynn_ipow<2>(gtr)), mynn_ipow<2>(taue))), (float)(zero))), (float)(three));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1216
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1217
vkz = r6_mul(karman, zw(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1218
el(k) = r6_div(vkz, r6_add(one, r6_div(vkz, hundred)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1221
tsq(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1222
qsq(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1223
cov(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1224
qpe(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1225
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; vkz = mynn_lane_bcast(vkz, mynn_src1); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1229
lmax = 5;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1231
for (l=1; (1) > 0 ? l <= (lmax) : l >= (lmax); l += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1234
f_mym_length(kts, kte, xland, dz, dx, zw, rmol, rmolh, flt, fltv, flq, vt, vq, u, v, qke, qpe, dtv, el, pblh, theta, qkw, qtw, psig_bl, cldfra_bl1, bl_mynn_mixlength, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1250
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1251
elq = r6_mul(el(k), qkw(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1252
pdk(k) = r6_mul(elq, r6_add(r6_mul(sm(k), gm(k)), r6_mul(sh(k), gh(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1254
pdt(k) = r6_mul(r6_mul(elq, sh(k)), mynn_ipow<2>(dtl(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1255
pdq(k) = r6_mul(r6_mul(elq, sh(k)), mynn_ipow<2>(dqw(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1256
pdc(k) = r6_mul(r6_mul(r6_mul(elq, sh(k)), dtl(k)), dqw(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1257
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; elq = mynn_lane_bcast(elq, mynn_src2); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1260
vkz = r6_mul(r6_mul(karman, p5), dz(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1261
elv = r6_div(r6_mul(p5, r6_add(el(((kts) + (1))), el(kts))), vkz);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1262
if (initialize_qke) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1264
qke(kts) = r6_mul(r6_mul(one, mynn_ipow<2>(ustm)), mynn_pow(r6_mul(r6_mul(b1, pmz), elv), p666));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1265
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1267
phm = r6_div(r6_mul(phh, b2), mynn_pow(r6_div(r6_mul(b1, pmz), mynn_ipow<2>(elv)), p333));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1268
tsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flt)), ustm)))), (float)(three));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1269
qsq(kts) = mynn_min((float)(r6_mul(phm, mynn_ipow<2>(r6_div(mynn_max((float)(zero), (float)(flq)), ustm)))), (float)(5e-6f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1270
cov(kts) = r6_mul(r6_mul(phm, r6_div(flt, ustm)), r6_div(flq, ustm));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1272
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1273
b1l = r6_mul(r6_mul(b1, 0.25f), r6_add(el(((k) + (1))), el(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1276
tmpq = mynn_min((float)(mynn_max((float)(r6_mul(b1l, r6_add(pdk(((k) + (1))), pdk(k)))), (float)(qkemin))), (float)(125.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1278
if (initialize_qke) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1279
qke(k) = mynn_pow(tmpq, p666);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1280
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1282
if (((qke(k)) <= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1283
b2l = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1284
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1285
b2l = r6_div(r6_mul(b2, r6_div(b1l, b1)), mynn_sqrt(qke(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1286
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1288
tsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdt(((k) + (1))), pdt(k)))), (float)(zero))), (float)(three));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1289
qsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdq(((k) + (1))), pdq(k)))), (float)(zero))), (float)(5e-6f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1290
cov(k) = r6_mul(b2l, r6_add(pdc(((k) + (1))), pdc(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1291
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; b1l = mynn_lane_bcast(b1l, mynn_src3); b2l = mynn_lane_bcast(b2l, mynn_src3); tmpq = mynn_lane_bcast(tmpq, mynn_src3); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1293
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1295
if (initialize_qke) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1296
qke(kts) = r6_mul(p5, r6_add(qke(kts), qke(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1297
qke(kte) = qke(((kte) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1298
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1299
tsq(kte) = tsq(((kte) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1300
qsq(kte) = qsq(((kte) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1301
cov(kte) = cov(((kte) - (1)));
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1347-1549
__device__ __noinline__ void f_mym_level2(int kts, int kte, float ckmod, int bl_mynn_ess, MynnArray<float, 1> zw, MynnArray<float, 1> dz, float xland, float pblh, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> thl, MynnArray<float, 1> thv, MynnArray<float, 1> thlv, MynnArray<float, 1> th, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> qke, MynnArray<float, 1> cldfra, MynnArray<float, 1> edmf_a, MynnArray<float, 1> edmf_w, MynnArray<float, 1> qw, MynnArray<float, 1> ql, MynnArray<float, 1> vt, MynnArray<float, 1> vq, MynnArray<float, 1> dtl, MynnArray<float, 1> dqw, MynnArray<float, 1> dtv, MynnArray<float, 1> gm, MynnArray<float, 1> gh, MynnArray<float, 1> sm, MynnArray<float, 1> sh) {
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1395
dtl(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1396
dqw(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1397
dtv(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1398
gm(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1399
gh(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1400
sm(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1401
sh(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1402
ri(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1404
switch (bl_mynn_ess) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1406
case 0: {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1408
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1409
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1410
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1411
abk = r6_sub(one, afk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1412
duz = r6_add(mynn_ipow<2>(r6_sub(u(k), u(((k) - (1))))), mynn_ipow<2>(r6_sub(v(k), v(((k) - (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1413
duz = r6_div(duz, mynn_ipow<2>(dzk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1414
dtz = r6_div(r6_sub(thl(k), thl(((k) - (1)))), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1415
dqz = r6_div(r6_sub(qw(k), qw(((k) - (1)))), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1417
vtt = r6_add(r6_add(one, r6_mul(vt(k), abk)), r6_mul(vt(((k) - (1))), afk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1418
vqq = r6_add(r6_add(tv0, r6_mul(vq(k), abk)), r6_mul(vq(((k) - (1))), afk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1419
dtq = r6_add(r6_mul(vtt, dtz), r6_mul(vqq, dqz));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1421
dtl(k) = dtz;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1422
dqw(k) = dqz;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1423
dtv(k) = dtq;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1425
gm(k) = duz;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1426
gh(k) = r6_neg(r6_mul(dtq, gtr));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1429
ri(k) = r6_neg(r6_div(gh(k), mynn_max((float)(duz), (float)(1.0e-10f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1430
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src0); afk = mynn_lane_bcast(afk, mynn_src0); dqz = mynn_lane_bcast(dqz, mynn_src0); dtq = mynn_lane_bcast(dtq, mynn_src0); dtz = mynn_lane_bcast(dtz, mynn_src0); duz = mynn_lane_bcast(duz, mynn_src0); dzk = mynn_lane_bcast(dzk, mynn_src0); vqq = mynn_lane_bcast(vqq, mynn_src0); vtt = mynn_lane_bcast(vtt, mynn_src0); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1432
break; } case 1: {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1441
ugrid = mynn_sqrt(r6_add(mynn_ipow<2>(u(kts)), mynn_ipow<2>(v(kts))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1442
uonset = 15.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1443
taper = r6_sub(one, r6_mul(p95, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, uonset)), (float)(zero)), twenty)), (float)(one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1444
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1445
clam0 = r6_mul(1.8f, taper);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1446
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1449
clam0 = 0.08f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1450
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1452
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1453
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1454
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1455
abk = r6_sub(one, afk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1456
duz = r6_add(mynn_ipow<2>(r6_sub(u(k), u(((k) - (1))))), mynn_ipow<2>(r6_sub(v(k), v(((k) - (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1457
duz = r6_div(duz, mynn_ipow<2>(dzk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1458
dtz = r6_div(r6_sub(thl(k), thl(((k) - (1)))), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1459
dqz = r6_div(r6_sub(qw(k), qw(((k) - (1)))), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1462
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zw(k), r6_add(pblh, hundred))), (float)(zero)), mynn_max((float)(r6_mul(p3, pblh)), (float)(400.f)))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1463
clam = r6_add(r6_mul(clam0, r6_sub(one, wt)), r6_mul(p2, wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1465
clam = r6_mul(clam, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zw(k), twenty)), (float)(zero)), 200.0f)), (float)(one)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1469
thv1 = r6_add(r6_mul(thvp, thv(k)), r6_mul(r6_sub(one, thvp), thlv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1470
thv0 = r6_add(r6_mul(thvp, thv(((k) - (1)))), r6_mul(r6_sub(one, thvp), thlv(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1474
tabs = r6_mul(th(k), exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1475
xl = f_xl_blend(tabs);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1476
qsat = f_qsat_blend(tabs, p(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1477
eth1 = r6_mul(th(k), mynn_exp(r6_div(r6_mul(xl, qsat), r6_mul(cp, tabs))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1479
tabs = r6_mul(th(((k) - (1))), exner(((k) - (1))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1480
xl = f_xl_blend(tabs);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1481
qsat = f_qsat_blend(tabs, p(((k) - (1))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1482
eth0 = r6_mul(th(((k) - (1))), mynn_exp(r6_div(r6_mul(xl, qsat), r6_mul(cp, tabs))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1485
mfi = mynn_max((float)(zero), (float)(mynn_min((float)(r6_mul(r6_mul(p5, r6_add(edmf_a(k), edmf_a(((k) - (1))))), r6_mul(p5, r6_add(edmf_w(k), edmf_w(((k) - (1))))))), (float)(p2))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1489
qkei = mynn_max((float)(r6_mul(p5, r6_add(qke(k), qke(((k) - (1)))))), (float)(qkemin));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1490
qkei = r6_mul(mynn_min((float)(mynn_sqrt(r6_mul(p25, qkei))), (float)(p5)), p2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1493
cldfrai = r6_mul(p5, r6_add(cldfra(k), cldfra(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1495
cfac = mynn_max((float)(mynn_min((float)(r6_mul(2.7f, mynn_ipow<2>(r6_sub(cldfrai, one)))), (float)(one))), (float)(0.01f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1497
cldfrai = r6_mul(mynn_min((float)(r6_mul(cldfrai, cfac)), (float)(p1)), ten);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1501
lambda = r6_mul(r6_mul(clam, mynn_max((float)(mfi), (float)(qkei))), cldfrai);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1503
dtq = r6_add(r6_div(r6_sub(thv1, thv0), dzk), r6_mul(lambda, mynn_min((float)(r6_div(r6_sub(eth1, eth0), dzk)), (float)(zero))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1506
dtl(k) = dtz;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1507
dqw(k) = dqz;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1508
dtv(k) = dtq;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1510
gm(k) = duz;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1511
gh(k) = r6_neg(r6_mul(dtq, gtr));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1514
ri(k) = r6_neg(r6_div(gh(k), mynn_max((float)(duz), (float)(1.0e-10f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1515
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src1); afk = mynn_lane_bcast(afk, mynn_src1); cfac = mynn_lane_bcast(cfac, mynn_src1); clam = mynn_lane_bcast(clam, mynn_src1); cldfrai = mynn_lane_bcast(cldfrai, mynn_src1); dqz = mynn_lane_bcast(dqz, mynn_src1); dtq = mynn_lane_bcast(dtq, mynn_src1); dtz = mynn_lane_bcast(dtz, mynn_src1); duz = mynn_lane_bcast(duz, mynn_src1); dzk = mynn_lane_bcast(dzk, mynn_src1); eth0 = mynn_lane_bcast(eth0, mynn_src1); eth1 = mynn_lane_bcast(eth1, mynn_src1); lambda = mynn_lane_bcast(lambda, mynn_src1); mfi = mynn_lane_bcast(mfi, mynn_src1); qkei = mynn_lane_bcast(qkei, mynn_src1); qsat = mynn_lane_bcast(qsat, mynn_src1); tabs = mynn_lane_bcast(tabs, mynn_src1); thv0 = mynn_lane_bcast(thv0, mynn_src1); thv1 = mynn_lane_bcast(thv1, mynn_src1); wt = mynn_lane_bcast(wt, mynn_src1); xl = mynn_lane_bcast(xl, mynn_src1); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1517
break; } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1519
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1521
a2fac = r6_div(one, r6_add(one, r6_mul(ckmod, mynn_max((float)(zero), (float)(ri(k))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1523
rfc = r6_div(g1, r6_add(g1, g2));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1524
f1 = r6_add(r6_add(r6_mul(b1, r6_sub(g1, c1)), r6_mul(r6_mul(r6_mul(r6_mul(three, a2), a2fac), r6_sub(one, c2)), r6_sub(one, c5))), r6_mul(r6_mul(two, a1), r6_sub(three, r6_mul(two, c2))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1526
f2 = r6_sub(r6_mul(b1, r6_add(g1, g2)), r6_mul(r6_mul(three, a1), r6_sub(one, c2)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1527
rf1 = r6_div(r6_mul(b1, r6_sub(g1, c1)), f1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1528
rf2 = r6_div(r6_mul(b1, g1), f2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1529
smc = r6_div(r6_mul(r6_div(a1, r6_mul(a2, a2fac)), f1), f2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1530
shc = r6_mul(r6_mul(three, r6_mul(a2, a2fac)), r6_add(g1, g2));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1532
ri1 = r6_div(p5, smc);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1533
ri2 = r6_mul(rf1, smc);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1534
ri3 = r6_sub(r6_mul(r6_mul(four, rf2), smc), r6_mul(two, ri2));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1535
ri4 = mynn_ipow<2>(ri2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1538
rf = mynn_min((float)(r6_mul(ri1, r6_sub(r6_add(ri(k), ri2), mynn_sqrt(r6_add(r6_sub(mynn_ipow<2>(ri(k)), r6_mul(ri3, ri(k))), ri4))))), (float)(rfc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1540
sh(k) = r6_div(r6_mul(shc, r6_sub(rfc, rf)), r6_sub(one, rf));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1541
sm(k) = r6_mul(r6_div(r6_mul(smc, r6_sub(rf1, rf)), r6_sub(rf2, rf)), sh(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1542
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; a2fac = mynn_lane_bcast(a2fac, mynn_src2); f1 = mynn_lane_bcast(f1, mynn_src2); f2 = mynn_lane_bcast(f2, mynn_src2); rf = mynn_lane_bcast(rf, mynn_src2); rf1 = mynn_lane_bcast(rf1, mynn_src2); rf2 = mynn_lane_bcast(rf2, mynn_src2); rfc = mynn_lane_bcast(rfc, mynn_src2); ri1 = mynn_lane_bcast(ri1, mynn_src2); ri2 = mynn_lane_bcast(ri2, mynn_src2); ri3 = mynn_lane_bcast(ri3, mynn_src2); ri4 = mynn_lane_bcast(ri4, mynn_src2); shc = mynn_lane_bcast(shc, mynn_src2); smc = mynn_lane_bcast(smc, mynn_src2); } }
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1569-2006
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1644
switch (bl_mynn_mixlength) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1646
case 0: {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1648
cns = 2.7f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1649
alp1 = 0.23f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1650
alp2 = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1651
alp3 = five;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1652
alp4 = hundred;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1653
alp5 = p3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1656
pblh2 = r6_add(pblh, dz(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1657
h1 = mynn_max((float)(r6_mul(p3, pblh2)), (float)(mindz));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1658
h1 = mynn_min((float)(h1), (float)(maxdz));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1659
h2 = r6_mul(h1, p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1661
qkw(kts) = mynn_sqrt(mynn_max((float)(qke(kts)), (float)(qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1662
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1663
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1664
abk = r6_sub(one, afk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1665
qkw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(qke(k), abk), r6_mul(qke(((k) - (1))), afk))), (float)(qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1666
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src0); afk = mynn_lane_bcast(afk, mynn_src0); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1667
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qtw).size(); mynn_element += lane_width) { qtw.linear(mynn_element) = qkw.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1669
elt = 1.0e-5f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1670
vsc = 1.0e-5f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1673
k = ((kts) + (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1674
zwk = zw(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1675
while (((zwk) <= (r6_add(pblh2, h1)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1676
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1677
qdz = r6_mul(mynn_max((float)(r6_sub(qkw(k), qmin)), (float)(0.03f)), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1678
elt = r6_add(elt, r6_mul(qdz, zwk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1679
vsc = r6_add(vsc, qdz);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1680
k = ((k) + (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1681
zwk = zw(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1682
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1684
elt = r6_div(r6_mul(alp1, elt), vsc);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1685
vflx = r6_add(r6_mul(r6_add(vt(kts), one), flt), r6_mul(r6_add(vq(kts), tv0), flq));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1686
vsc = mynn_pow(r6_mul(r6_mul(gtr, elt), mynn_max((float)(zero), (float)(vflx))), p333);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1689
el(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1690
zwk1 = zw(((kts) + (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1692
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1693
zwk = zw(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1696
if (((dtv(k)) > (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1697
bv = mynn_sqrt(r6_mul(gtr, dtv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1698
elb = r6_mul(r6_div(r6_mul(alp2, qkw(k)), bv), r6_add(one, r6_mul(r6_div(alp3, alp2), mynn_sqrt(r6_div(vsc, r6_mul(bv, elt))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1701
elf = r6_div(r6_mul(alp2, qkw(k)), bv);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1703
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1704
elb = 1.0e10f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1705
elf = elb;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1706
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1709
if (((rmol) > (0.0f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1710
els = r6_div(r6_mul(karman, zwk), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1711
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1712
els = r6_mul(r6_mul(karman, zwk), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1713
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1719
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1721
el(k) = mynn_min((float)(r6_div(elb, r6_add(r6_add(r6_div(elb, elt), r6_div(elb, els)), one))), (float)(elf));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1723
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1725
break; } case 1: {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1728
ugrid = mynn_sqrt(r6_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1729
uonset = 15.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1730
wt_u1 = r6_sub(one, r6_mul(p2, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, uonset)), (float)(zero)), forty)), (float)(one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1731
wt_u2 = r6_sub(one, r6_mul(p5, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, uonset)), (float)(zero)), forty)), (float)(one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1733
wt_dx = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(dx, 4000.f)), (float)(zero)), 12000.f)), (float)(one)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1735
cns = 3.5f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1736
alp1 = r6_add(r6_mul(0.23f, wt_dx), r6_mul(r6_sub(one, wt_dx), 0.24f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1737
alp2 = r6_add(r6_mul(p3, wt_dx), r6_mul(r6_sub(one, wt_dx), 0.40f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1738
alp3 = r6_mul(five, wt_u2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1739
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1740
alp4 = r6_mul(twenty, wt_u2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1741
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1742
alp4 = 15.0f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1743
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1744
alp5 = p3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1745
alp6 = fifty;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1748
pblh2 = mynn_max((float)(pblh), (float)(200.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1749
h1 = mynn_max((float)(r6_mul(p3, pblh2)), (float)(300.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1750
h1 = mynn_min((float)(h1), (float)(600.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1751
h2 = r6_mul(h1, p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1753
qtke(kts) = mynn_max((float)(r6_mul(p5, qke(kts))), (float)(r6_mul(p5, qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1754
thetaw(kts) = theta(kts);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1755
qkw(kts) = mynn_sqrt(mynn_max((float)(qke(kts)), (float)(qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1756
qtw(kts) = mynn_sqrt(mynn_max((float)(r6_add(qke(kts), qpe(kts))), (float)(qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1757
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1758
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1759
abk = r6_sub(one, afk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1760
qkw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(qke(k), abk), r6_mul(qke(((k) - (1))), afk))), (float)(qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1761
qtw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(r6_add(qke(k), qpe(k)), abk), r6_mul(r6_add(qke(((k) - (1))), qpe(((k) - (1)))), afk))), (float)(qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1762
qtke(k) = mynn_max((float)(r6_mul(p5, mynn_ipow<2>(qkw(k)))), (float)(0.005f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1763
thetaw(k) = r6_add(r6_mul(theta(k), abk), r6_mul(theta(((k) - (1))), afk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1764
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src1); afk = mynn_lane_bcast(afk, mynn_src1); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1766
elt = 1.0e-5f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1767
vsc = 1.0e-5f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1770
k = ((kts) + (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1771
zwk = zw(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1772
while (((zwk) <= (r6_add(pblh2, mynn_max((float)(mynn_min((float)(r6_mul(p3, pblh2)), (float)(600.f))), (float)(200.f)))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1773
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1774
qdz = r6_mul(mynn_min((float)(mynn_max((float)(qkw(k)), (float)(0.01f))), (float)(30.0f)), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1775
elt = r6_add(elt, r6_mul(qdz, zwk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1776
vsc = r6_add(vsc, qdz);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1777
k = ((k) + (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1778
zwk = zw(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1779
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1781
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1782
elt_max = r6_add(350.f, r6_mul(100.f, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(ugrid, twenty)), (float)(zero)), thirty)), (float)(one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1783
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1784
elt_max = 400.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1785
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1786
elt = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(alp1, elt), vsc)), (float)(8.f))), (float)(elt_max));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1789
vflx = r6_sub(fltv, 0.004f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1790
vsc = mynn_pow(r6_mul(r6_mul(gtr, elt), mynn_max((float)(zero), (float)(vflx))), p333);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1793
el(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1794
zwk1 = zw(((kts) + (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1797
f_boulac_length(kts, kte, zw, dz, qtke, thetaw, elblmin, elblavg);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1799
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1800
zwk = zw(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1801
qkw_mf = mynn_max((float)(r6_mul(r6_mul(p5, r6_add(edmf_a1(k), edmf_a1(((k) - (1))))), r6_mul(p5, r6_add(edmf_w1(k), edmf_w1(((k) - (1))))))), (float)(mynn_abs(r6_mul(edmf_a_dd1(((k) - (1))), edmf_w_dd1(((k) - (1)))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1804
cpblh = mynn_min((float)(r6_div(r6_add(zwk, r6_mul(p25, pblh2)), pblh2)), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1805
qkw_mf = r6_mul(cpblh, qkw_mf);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1806
alp3z = r6_mul(cpblh, alp3);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1808
if (((dtv(k)) > (0.0f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1809
bv = mynn_max((float)(mynn_sqrt(r6_mul(gtr, dtv(k)))), (float)(0.001f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1810
elb_mf = r6_div(r6_mul(alp6, qkw_mf), bv);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1811
elb_mf = r6_div(elb_mf, r6_add(one, r6_div(elb_mf, 100.f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1812
elb = r6_mul(r6_div(r6_mul(alp2, mynn_max((float)(qkw(k)), (float)(qkw_elb_min))), bv), r6_add(one, r6_mul(alp3z, mynn_sqrt(r6_div(vsc, r6_mul(bv, elt))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1814
elb = mynn_max((float)(elb), (float)(elb_mf));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1815
elb = mynn_min((float)(elb), (float)(zwk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1816
elf = r6_div(r6_mul(one, mynn_max((float)(qkw(k)), (float)(qkw_elb_min))), bv);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1817
elf = mynn_max((float)(elf), (float)(elb_mf));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1818
elblavg(k) = mynn_max((float)(r6_mul(alp5, elblavg(k))), (float)(elb_mf));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1819
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1820
elb = 1.0e10f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1821
elf = elb;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1822
elblavg(k) = r6_mul(p4, elblavg(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1823
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1825
z_m = mynn_max((float)(r6_sub(zwk, four)), (float)(karman));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1827
if (((rmol) > (0.0f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1828
els = r6_div(r6_mul(karman, zwk), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1829
els1 = r6_div(r6_mul(karman, z_m), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1830
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1831
els = r6_mul(r6_mul(karman, zwk), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1832
els1 = r6_mul(r6_mul(karman, z_m), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1833
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1836
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1844
el_unstab = r6_div(els, r6_add(one, r6_div(els1, elt)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1845
el(k) = mynn_min((float)(el_unstab), (float)(elb));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1846
el(k) = mynn_min((float)(el(k)), (float)(elf));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1847
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1848
el(k) = r6_mul(el(k), wt_u1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1849
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1850
el(k) = r6_add(r6_mul(el(k), r6_sub(one, wt)), r6_mul(elblavg(k), wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1858
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1860
break; } case 2: {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1862
uonset = r6_add(3.5f, r6_mul(dz(kts), p1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1863
ugrid = mynn_sqrt(r6_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1864
cns = 3.5f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1865
alp1 = 0.22f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1866
alp2 = p3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1867
alp3 = 2.5f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1868
alp4 = five;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1869
alp5 = alp2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1870
alp6 = fifty;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1874
pblh2 = mynn_max((float)(pblh), (float)(200.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1877
h1 = mynn_max((float)(r6_mul(p3, pblh2)), (float)(300.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1878
h1 = mynn_min((float)(h1), (float)(600.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1879
h2 = r6_mul(h1, p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1881
qtke(kts) = mynn_max((float)(r6_mul(p5, qke(kts))), (float)(r6_mul(p5, qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1882
qkw(kts) = mynn_sqrt(mynn_max((float)(qke(kts)), (float)(qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1883
qtw(kts) = mynn_sqrt(mynn_max((float)(r6_add(qke(kts), qpe(kts))), (float)(qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1884
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1885
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1886
abk = r6_sub(one, afk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1887
qkw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(qke(k), abk), r6_mul(qke(((k) - (1))), afk))), (float)(qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1888
qtw(k) = mynn_sqrt(mynn_max((float)(r6_add(r6_mul(r6_add(qke(k), qpe(k)), abk), r6_mul(r6_add(qke(((k) - (1))), qpe(((k) - (1)))), afk))), (float)(qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1889
qtke(k) = r6_mul(p5, mynn_ipow<2>(qkw(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1890
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; abk = mynn_lane_bcast(abk, mynn_src2); afk = mynn_lane_bcast(afk, mynn_src2); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1892
elt = 1.0e-5f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1893
vsc = 1.0e-5f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1896
pblh_plus_ent = mynn_max((float)(r6_add(pblh, h1)), (float)(100.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1897
k = ((kts) + (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1898
zwk = zw(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1899
while (((zwk) <= (pblh_plus_ent))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1900
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1901
qdz = r6_mul(mynn_min((float)(mynn_max((float)(r6_sub(qkw(k), qmin)), (float)(0.03f))), (float)(30.0f)), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1902
elt = r6_add(elt, r6_mul(qdz, zwk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1903
vsc = r6_add(vsc, qdz);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1904
k = ((k) + (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1905
zwk = zw(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1906
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1908
elt = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(alp1, elt), vsc)), (float)(ten))), (float)(400.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1911
vflx = r6_sub(fltv, 0.004f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1912
vsc = mynn_pow(r6_mul(r6_mul(gtr, elt), mynn_max((float)(zero), (float)(vflx))), p333);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1915
el(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1916
zwk1 = zw(((kts) + (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1918
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1919
zwk = zw(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1920
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1921
cldavg = r6_mul(p5, r6_add(cldfra_bl1(((k) - (1))), cldfra_bl1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1922
qkw_mf = mynn_max((float)(r6_mul(r6_mul(p5, r6_add(edmf_a1(k), edmf_a1(((k) - (1))))), r6_mul(p5, r6_add(edmf_w1(k), edmf_w1(((k) - (1))))))), (float)(mynn_abs(r6_mul(edmf_a_dd1(((k) - (1))), edmf_w_dd1(((k) - (1)))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1926
if (((dtv(k)) > (0.0f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1928
bv = mynn_max((float)(mynn_sqrt(r6_mul(gtr, dtv(k)))), (float)(0.001f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1930
elb_mf = r6_mul(r6_div(mynn_max((float)(r6_mul(alp2, qkw(k))), (float)(r6_mul(alp6, qkw_mf))), bv), r6_add(one, r6_mul(alp3, mynn_sqrt(r6_div(vsc, r6_mul(bv, elt))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1933
elb = mynn_min((float)(r6_div(mynn_max((float)(r6_mul(alp5, qkw(k))), (float)(r6_mul(alp6, qkw_mf))), bv)), (float)(zwk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1936
wstar = r6_mul(1.25f, mynn_pow(r6_mul(r6_mul(gtr, pblh), mynn_max((float)(vflx), (float)(1.0e-4f))), p333));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1937
tau_cloud = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(ctau, wstar), grav)), (float)(30.f))), (float)(150.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1939
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1940
tau_cloud = r6_add(r6_mul(tau_cloud, r6_sub(1.f, wt)), r6_mul(fifty, wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1941
elf = mynn_min((float)(mynn_max((float)(r6_mul(tau_cloud, mynn_sqrt(mynn_min((float)(qtke(k)), (float)(forty))))), (float)(r6_div(r6_mul(alp6, qkw_mf), bv)))), (float)(zwk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1951
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1963
wstar = r6_mul(1.25f, mynn_pow(r6_mul(r6_mul(gtr, pblh), mynn_max((float)(vflx), (float)(1.0e-4f))), p333));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1964
tau_cloud = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(ctau, wstar), grav)), (float)(fifty))), (float)(200.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1966
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1968
tau_cloud = r6_add(r6_mul(tau_cloud, r6_sub(one, wt)), r6_mul(mynn_max((float)(r6_mul(dzk, p25)), (float)(hundred)), wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1970
elb = mynn_min((float)(r6_mul(tau_cloud, mynn_sqrt(mynn_min((float)(qtke(k)), (float)(40.f))))), (float)(zwk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1972
elf = elb;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1973
elb_mf = elb;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1974
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1975
elf = r6_div(elf, r6_add(one, r6_div(elf, 800.f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1976
elb_mf = mynn_max((float)(elb_mf), (float)(0.01f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1979
if (((rmol) > (0.0f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1980
els = r6_div(r6_mul(karman, zwk), r6_add(one, r6_mul(cns, mynn_min((float)(r6_mul(zwk, rmol)), (float)(zmax)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1981
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1982
els = r6_mul(r6_mul(karman, zwk), mynn_pow(r6_sub(one, r6_mul(r6_mul(alp4, zwk), rmol)), p2));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1983
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1986
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(zwk, r6_add(pblh2, h1)), h2))), p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1989
el(k) = mynn_sqrt(r6_div(mynn_ipow<2>(els), r6_add(r6_add(one, r6_div(mynn_ipow<2>(els), mynn_ipow<2>(elt))), r6_div(mynn_ipow<2>(els), mynn_ipow<2>(elb_mf)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1990
el(k) = r6_add(r6_mul(el(k), r6_sub(one, wt)), r6_mul(elf, wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1991
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1993
break; } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1996
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1997
el_les = r6_mul(r6_mul(p25, p5), r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1998
el(k) = r6_add(r6_mul(el(k), psig_bl), r6_mul(r6_sub(one, psig_bl), mynn_min((float)(el_les), (float)(el(k)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:1999
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; el_les = mynn_lane_bcast(el_les, mynn_src3); } }
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2181-2328
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2204
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { iz = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2209
zup = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2210
dlu(iz) = r6_sub(r6_sub(zw(((kte) + (1))), zw(iz)), r6_mul(dz(iz), p5));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2211
zzz = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2212
zup_inf = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2213
beta = gtr;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2217
if (((iz) < (kte))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2219
found = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2220
izz = iz;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2221
while (((found) == (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2223
if (((izz) < (kte))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2224
dzt = dz(izz);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2225
zup = r6_sub(zup, r6_mul(r6_mul(beta, theta(iz)), dzt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2227
zup = r6_add(zup, r6_mul(r6_mul(r6_mul(beta, r6_add(theta(((izz) + (1))), theta(izz))), dzt), p5));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2228
zzz = r6_add(zzz, dzt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2230
if (((((qtke(iz)) < (zup))) && (((qtke(iz)) >= (zup_inf))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2231
bbb = r6_div(r6_sub(theta(((izz) + (1))), theta(izz)), dzt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2232
if (((bbb) != (0.f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2234
tl = r6_div(r6_div(r6_add(r6_neg(r6_mul(beta, r6_sub(theta(izz), theta(iz)))), mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(r6_mul(beta, r6_sub(theta(izz), theta(iz)))), r6_mul(r6_mul(r6_mul(two, bbb), beta), r6_sub(qtke(iz), zup_inf)))), (float)(zero)))), bbb), beta);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2237
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2238
if (((theta(izz)) != (theta(iz)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2239
tl = r6_div(r6_sub(qtke(iz), zup_inf), r6_mul(beta, r6_sub(theta(izz), theta(iz))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2240
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2241
tl = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2242
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2243
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2244
dlu(iz) = r6_add(r6_sub(zzz, dzt), tl);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2246
found = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2247
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2248
zup_inf = zup;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2249
izz = ((izz) + (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2250
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2251
found = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2252
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2254
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2256
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2261
zdo = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2262
zdo_sup = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2263
dld(iz) = zw(iz);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2264
zzz = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2267
if (((iz) > (kts))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2269
found = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2270
izz = iz;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2271
while (((found) == (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2273
if (((izz) > (kts))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2274
dzt = dz(((izz) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2275
zdo = r6_add(zdo, r6_mul(r6_mul(beta, theta(iz)), dzt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2277
zdo = r6_sub(zdo, r6_mul(r6_mul(r6_mul(beta, r6_add(theta(((izz) - (1))), theta(izz))), dzt), p5));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2278
zzz = r6_add(zzz, dzt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2280
if (((((qtke(iz)) < (zdo))) && (((qtke(iz)) >= (zdo_sup))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2281
bbb = r6_div(r6_sub(theta(izz), theta(((izz) - (1)))), dzt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2282
if (((bbb) != (0.f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2283
tl = r6_div(r6_div(r6_add(r6_mul(beta, r6_sub(theta(izz), theta(iz))), mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(r6_mul(beta, r6_sub(theta(izz), theta(iz)))), r6_mul(r6_mul(r6_mul(two, bbb), beta), r6_sub(qtke(iz), zdo_sup)))), (float)(zero)))), bbb), beta);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2286
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2287
if (((theta(izz)) != (theta(iz)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2288
tl = r6_div(r6_sub(qtke(iz), zdo_sup), r6_mul(beta, r6_sub(theta(iz), theta(izz))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2289
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2290
tl = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2291
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2292
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2293
dld(iz) = r6_add(r6_sub(zzz, dzt), tl);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2295
found = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2296
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2297
zdo_sup = zdo;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2298
izz = ((izz) - (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2299
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2300
found = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2301
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2302
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2304
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2310
dld(iz) = mynn_min((float)(dld(iz)), (float)(zw(((iz) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2312
dlu(iz) = mynn_max((float)(r6_div(dlu(iz), r6_add(one, r6_div(dlu(iz), lmax)))), (float)(p1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2313
dld(iz) = mynn_max((float)(r6_div(dld(iz), r6_add(one, r6_div(dld(iz), lmax)))), (float)(p1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2316
lb1(iz) = mynn_min((float)(dlu(iz)), (float)(dld(iz)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2318
lb2(iz) = mynn_sqrt(r6_mul(dlu(iz), dld(iz)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2323
} mynn_lane_sync(); iz = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; beta = mynn_lane_bcast(beta, mynn_src0); zdo = mynn_lane_bcast(zdo, mynn_src0); zdo_sup = mynn_lane_bcast(zdo_sup, mynn_src0); zup = mynn_lane_bcast(zup, mynn_src0); zup_inf = mynn_lane_bcast(zup_inf, mynn_src0); zzz = mynn_lane_bcast(zzz, mynn_src0); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2325
lb1(kte) = lb1(((kte) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2326
lb2(kte) = lb2(((kte) - (1)));
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2379-2933
__device__ __noinline__ void f_mym_turbulence(int kts, int kte, float xland, float closure, float ckmod, float taue, MynnArray<float, 1> dz, float dx, MynnArray<float, 1> zw, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> thl, MynnArray<float, 1> thv, MynnArray<float, 1> thlv, MynnArray<float, 1> ql, MynnArray<float, 1> qw, MynnArray<float, 1> qke, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> vt, MynnArray<float, 1> vq, float rmol, float rmolh, float flt, float fltv, float flq, float pblh, MynnArray<float, 1> theta, MynnArray<float, 1> sh, MynnArray<float, 1> sm, MynnArray<float, 1> el, MynnArray<float, 1> dfm, MynnArray<float, 1> dfh, MynnArray<float, 1> dfq, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> pdk, MynnArray<float, 1> pdt, MynnArray<float, 1> pdq, MynnArray<float, 1> pdc, MynnArray<float, 1> qwt1, MynnArray<float, 1> qshear1, MynnArray<float, 1> qbuoy1, MynnArray<float, 1> qdiss1, int tke_budget, float psig_bl, float psig_shcu, MynnArray<float, 1> cldfra_bl1, int bl_mynn_mixlength, int bl_mynn_ess, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> edmf_w_dd1, MynnArray<float, 1> edmf_a_dd1, MynnArray<float, 1> tkeprod_dn, MynnArray<float, 1> tkeprod_up, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1) {
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2473
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qkw).size(); mynn_element += lane_width) { qkw.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2474
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qtw).size(); mynn_element += lane_width) { qtw.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2475
pblh2 = mynn_max((float)(pblh), (float)(200.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2476
if (((closure) < (2.7f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2477
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qpe).size(); mynn_element += lane_width) { qpe.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2478
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2479
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2482
cpblh = mynn_min((float)(r6_div(r6_add(zw(k), r6_mul(p1, pblh2)), r6_mul(0.8f, pblh2))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2484
qpe(k) = mynn_min((float)(mynn_max((float)(r6_mul(r6_mul(r6_mul(r6_mul(cpblh, two), tsq(k)), mynn_ipow<2>(gtr)), mynn_ipow<2>(taue))), (float)(zero))), (float)(three));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2485
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; cpblh = mynn_lane_bcast(cpblh, mynn_src0); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2486
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2488
f_mym_level2(kts, kte, ckmod, bl_mynn_ess, zw, dz, xland, pblh, u, v, thl, thv, thlv, theta, p, exner, qke, cldfra_bl1, edmf_a1, edmf_w1, qw, ql, vt, vq, dtl, dqw, dtv, gm, gh, sm, sh);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2498
f_mym_length(kts, kte, xland, dz, dx, zw, rmol, rmolh, flt, fltv, flq, vt, vq, u, v, qke, qpe, dtv, el, pblh, theta, qkw, qtw, psig_bl, cldfra_bl1, bl_mynn_mixlength, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2514
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2515
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2516
afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2517
abk = r6_sub(one, afk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2518
elsq = mynn_ipow<2>(el(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2519
q3sq = mynn_ipow<2>(qkw(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2520
q2sq = mynn_dmul(mynn_dmul(b1, elsq), r6_add(r6_mul(sm(k), gm(k)), r6_mul(sh(k), gh(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2522
sh20 = mynn_max((float)(sh(k)), (float)(1e-5f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2523
sm20 = mynn_max((float)(sm(k)), (float)(1e-5f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2524
sh(k) = mynn_max((float)(sh(k)), (float)(1e-5f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2527
duz = r6_add(mynn_ipow<2>(r6_sub(u(k), u(((k) - (1))))), mynn_ipow<2>(r6_sub(v(k), v(((k) - (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2528
duz = r6_div(duz, mynn_ipow<2>(dzk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2530
ri = r6_neg(r6_div(gh(k), mynn_max((float)(duz), (float)(1.0e-10f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2532
a2fac = r6_div(one, r6_add(one, r6_mul(ckmod, mynn_max((float)(zero), (float)(ri)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2540
if (((ri) >= (one))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2542
prlim = mynn_min((float)(r6_mul(seven, ri)), (float)(prlimit_fre));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2543
} else if (((((ri) >= (0.01f))) && (((ri) <= (one))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2545
prlim = mynn_min((float)(r6_add(r6_mul(6.873f, ri), r6_div(one, r6_mul(6.873f, ri)))), (float)(prlimit_fre));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2546
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2548
prlim = prlimit_fre;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2549
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2550
wt = mynn_min((float)(r6_div(zw(k), mynn_max((float)(pblh), (float)(200.f)))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2551
prlim = r6_add(r6_mul(wt, prlim), r6_mul(r6_sub(one, wt), mynn_min((float)(prlim), (float)(prlimit_sfc))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2554
gmel = mynn_dmul(gm(k), elsq);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2555
ghel = mynn_dmul(gh(k), elsq);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2559
if (debug_code) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2560
if (((((sh(k)) < (0.0f))) || (((sm(k)) < (0.0f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2561
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2562
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2563
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2564
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2565
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2566
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2567
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2573
dlsq = elsq;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2574
if (((mynn_ddiv(q3sq, dlsq)) < (r6_neg(gh(k))))) { q3sq = r6_neg(mynn_dmul(dlsq, gh(k))); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2576
if (((q3sq) < (q2sq))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2578
qdiv = mynn_sqrt(mynn_ddiv(q3sq, q2sq));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2594
sh(k) = mynn_dmul(sh(k), qdiv);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2595
sm(k) = mynn_dmul(sm(k), qdiv);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2603
e1 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e1c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2604
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e2c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2605
e3 = mynn_dadd(e1, mynn_dmul(mynn_dmul(mynn_dmul(e3c, ghel), mynn_ipow<2>(a2fac)), mynn_ipow<2>(qdiv)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2606
e4 = mynn_dsub(e1, mynn_dmul(mynn_dmul(mynn_dmul(e4c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2607
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(mynn_dmul(e3, e5c), gmel), mynn_ipow<2>(qdiv)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2608
eden = mynn_max((double)(eden), (double)(1.0e-20));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2612
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2618
e1 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(e1c, ghel), a2fac));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2619
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(e2c, ghel), a2fac));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2620
e3 = mynn_dadd(e1, mynn_dmul(mynn_dmul(e3c, ghel), mynn_ipow<2>(a2fac)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2621
e4 = mynn_dsub(e1, mynn_dmul(mynn_dmul(e4c, ghel), a2fac));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2622
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(e3, e5c), gmel));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2623
eden = mynn_max((double)(eden), (double)(1.0e-20));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2625
qdiv = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2627
sm(k) = mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, a1), mynn_dsub(e3, mynn_dmul(r6_mul(three, c1), e4))), eden);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2631
sh(k) = mynn_ddiv(mynn_dmul((tke_budget == 1 ? mynn_dmul(q3sq, r6_mul(a2, a2fac)) : mynn_dadd(mynn_dmul(r6_mul(r6_mul(three, c1), e5c), gmel), e2)), (tke_budget == 1 ? mynn_dadd(mynn_dmul(r6_mul(r6_mul(three, c1), e5c), gmel), e2) : mynn_dmul(q3sq, r6_mul(a2, a2fac)))), eden);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2632
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2635
gmelq = mynn_max((float)(((float)(mynn_ddiv(gmel, q3sq)))), (float)(1e-8f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2636
sm25max = four;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2637
sh25max = four;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2638
sm25min = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2639
sh25min = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2643
if (debug_code) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2644
if (((((((((sh(k)) < (sh25min))) || (((sm(k)) < (sm25min))))) || (((sh(k)) > (sh25max))))) || (((sm(k)) > (sm25max))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2646
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2647
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2648
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2649
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2650
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2651
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2652
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2653
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2654
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2656
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2657
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2660
if (((sh(k)) > (sh25max))) { sh(k) = sh25max; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2661
if (((sh(k)) < (sh25min))) { sh(k) = sh25min; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2665
shb = mynn_max((float)(0.004f), (float)(sh(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2666
sm(k) = mynn_min((float)(sm(k)), (float)(r6_mul(prlim, shb)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2669
if (((closure) >= (3.0f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2670
t2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, b2), elsq), sh(k)), mynn_ipow<2>(dtl(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2671
r2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, b2), elsq), sh(k)), mynn_ipow<2>(dqw(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2672
c2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, b2), elsq), sh(k)), dtl(k)), dqw(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2673
t3sq = mynn_max((float)(zero), (float)(r6_add(r6_mul(tsq(k), abk), r6_mul(tsq(((k) - (1))), afk))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2674
r3sq = mynn_max((float)(zero), (float)(r6_add(r6_mul(qsq(k), abk), r6_mul(qsq(((k) - (1))), afk))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2675
c3sq = r6_add(r6_mul(cov(k), abk), r6_mul(cov(((k) - (1))), afk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2678
c3sq = mynn_sign(mynn_min((double)(mynn_abs(c3sq)), (double)(mynn_sqrt(mynn_dmul(t3sq, r3sq)))), c3sq);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2680
vtt = r6_add(r6_add(one, r6_mul(vt(k), abk)), r6_mul(vt(((k) - (1))), afk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2681
vqq = r6_add(r6_add(tv0, r6_mul(vq(k), abk)), r6_mul(vq(((k) - (1))), afk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2683
t2sq = mynn_dadd(mynn_dmul(vtt, t2sq), mynn_dmul(vqq, c2sq));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2684
r2sq = mynn_dadd(mynn_dmul(vtt, c2sq), mynn_dmul(vqq, r2sq));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2685
c2sq = mynn_max((double)(0.0e0), (double)(mynn_dadd(mynn_dmul(vtt, t2sq), mynn_dmul(vqq, r2sq))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2686
t3sq = mynn_dadd(mynn_dmul(vtt, t3sq), mynn_dmul(vqq, c3sq));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2687
r3sq = mynn_dadd(mynn_dmul(vtt, c3sq), mynn_dmul(vqq, r3sq));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2688
c3sq = mynn_max((double)(0.0e0), (double)(mynn_dadd(mynn_dmul(vtt, t3sq), mynn_dmul(vqq, r3sq))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2690
cw25 = mynn_ddiv(mynn_dmul(e1, mynn_dadd(e2, mynn_dmul(mynn_dmul(r6_mul(r6_mul(three, c1), e5c), gmel), mynn_ipow<2>(qdiv)))), mynn_dmul(three, eden));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2693
dlsq = elsq;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2694
if (((mynn_ddiv(q3sq, dlsq)) < (r6_neg(gh(k))))) { q3sq = r6_neg(mynn_dmul(dlsq, gh(k))); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2699
auh = r6_mul(r6_mul(r6_mul(r6_mul(27.f, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), b2), mynn_ipow<2>(gtr));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2700
aum = r6_mul(r6_mul(r6_mul(r6_mul(r6_mul(54.f, mynn_ipow<2>(a1)), r6_mul(a2, a2fac)), b2), c1), gtr);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2701
adh = r6_mul(r6_mul(r6_mul(r6_mul(nine, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), r6_add(r6_mul(12.f, a1), r6_mul(three, b2))), mynn_ipow<2>(gtr));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2702
adm = r6_mul(r6_mul(r6_mul(r6_mul(18.f, mynn_ipow<2>(a1)), r6_mul(a2, a2fac)), r6_sub(b2, r6_mul(three, r6_mul(a2, a2fac)))), gtr);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2704
aeh = r6_mul(r6_add(r6_mul(r6_mul(r6_mul(nine, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), b1), r6_mul(r6_mul(r6_mul(nine, a1), mynn_ipow<2>(r6_mul(a2, a2fac))), r6_add(r6_mul(12.f, a1), r6_mul(three, b2)))), gtr);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2706
aem = r6_add(r6_mul(r6_mul(r6_mul(r6_mul(three, a1), r6_mul(a2, a2fac)), b1), r6_add(r6_add(r6_mul(three, r6_mul(a2, a2fac)), r6_mul(r6_mul(three, b2), c1)), r6_sub(r6_mul(r6_mul(18.f, a1), c1), b2))), r6_mul(r6_mul(r6_mul(18.f, mynn_ipow<2>(a1)), r6_mul(a2, a2fac)), r6_sub(b2, r6_mul(three, r6_mul(a2, a2fac)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2710
req = r6_neg(r6_div(aeh, aem));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2711
rsl = r6_div(r6_add(auh, r6_mul(aum, req)), r6_add(r6_mul(three, adh), r6_mul(r6_mul(three, adm), req)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2713
rsl = 0.12f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2714
rsl2 = r6_sub(one, r6_mul(two, rsl));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2726
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e2c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2727
e3 = mynn_dadd(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e3c, ghel), mynn_ipow<2>(a2fac)), mynn_ipow<2>(qdiv)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2728
e4 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(e4c, ghel), a2fac), mynn_ipow<2>(qdiv)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2729
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(mynn_dmul(e3, e5c), gmel), mynn_ipow<2>(qdiv)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2734
wden = mynn_dmul(mynn_dmul(mynn_ddiv(mynn_dmul(r6_mul(cc3, mynn_ipow<2>(gtr)), mynn_ipow<2>(dlsq)), elsq), mynn_ipow<2>(qdiv)), mynn_dsub(mynn_dmul(mynn_dmul(e2, e4c), a2fac), mynn_dmul(mynn_dmul(mynn_dmul(r6_mul(e3c, e5c), gmel), mynn_ipow<2>(a2fac)), mynn_ipow<2>(qdiv))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2737
if (((wden) != (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2739
clow = mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, r6_sub(0.12f, cw25)), eden), wden);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2740
cupp = mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, r6_sub(0.76f, cw25)), eden), wden);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2744
if (((wden) > (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2745
c3sq = mynn_min((double)(mynn_max((double)(c3sq), (double)(mynn_dadd(c2sq, clow)))), (double)(mynn_dadd(c2sq, cupp)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2746
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2747
c3sq = mynn_max((double)(mynn_min((double)(c3sq), (double)(mynn_dadd(c2sq, clow)))), (double)(mynn_dadd(c2sq, cupp)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2748
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2749
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2751
e1 = mynn_dadd(e2, mynn_dmul(mynn_dmul(e5c, gmel), mynn_ipow<2>(qdiv)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2752
eden = mynn_max((double)(eden), (double)(1.0e-20));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2757
e6c = mynn_ddiv(mynn_dmul(r6_mul(r6_mul(r6_mul(3.0f, r6_mul(a2, a2fac)), cc3), gtr), dlsq), elsq);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2762
if (((t2sq) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2763
mynn_enum = mynn_max((double)(0.0e0), (double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(t3sq, t2sq))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2764
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2765
mynn_enum = mynn_min((double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(t3sq, t2sq))), (double)(0.0e0));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2766
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2767
gamt = r6_neg((float)mynn_ddiv(mynn_dmul(mynn_enum, e1), eden));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2772
if (((r2sq) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2773
mynn_enum = mynn_max((double)(0.0e0), (double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(r3sq, r2sq))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2774
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2775
mynn_enum = mynn_min((double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(r3sq, r2sq))), (double)(0.0e0));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2776
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2777
gamq = r6_neg((float)mynn_ddiv(mynn_dmul(mynn_enum, e1), eden));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2782
mynn_enum = mynn_max((double)(0.0e0), (double)(mynn_dmul(mynn_dmul(qdiv, e6c), mynn_dsub(c3sq, c2sq))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2786
smd = mynn_ddiv(mynn_dmul(mynn_dmul(mynn_dmul(mynn_ddiv(mynn_dmul(mynn_dmul(dlsq, mynn_enum), gtr), eden), mynn_ipow<2>(qdiv)), r6_add(r6_mul(e3c, mynn_ipow<2>(a2fac)), r6_mul(e4c, a2fac))), a1), r6_mul(a2, a2fac));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2789
gamv = mynn_ddiv(mynn_dmul(mynn_dmul(e1, mynn_enum), gtr), eden);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2790
sm(k) = r6_add(sm(k), smd);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2794
qdiv = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2797
if (debug_code) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2798
if (((((((((sh(k)) < (r6_neg(0.3f)))) || (((sm(k)) < (r6_neg(0.3f)))))) || (((qke(k)) < (r6_neg(0.1f)))))) || (((mynn_abs(smd)) > (2.0f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2800
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2801
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2802
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2803
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2804
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2805
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2806
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2810
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2812
gamt = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2813
gamq = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2814
gamv = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2815
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2819
cldavg = r6_mul(p5, r6_add(cldfra_bl1(((k) - (1))), cldfra_bl1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2820
mfmax = mynn_max((float)(r6_mul(r6_mul(r6_mul(p5, r6_add(edmf_a1(((k) - (1))), edmf_a1(k))), p5), r6_add(edmf_w1(((k) - (1))), edmf_w1(k)))), (float)(mynn_abs(r6_mul(edmf_a_dd1(k), edmf_w_dd1(k)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2822
sm(k) = mynn_max((float)(sm(k)), (float)(r6_mul(0.04f, mynn_min((float)(r6_mul(10.f, mfmax)), (float)(one)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2823
sh(k) = mynn_max((float)(r6_mul(0.04f, mynn_min((float)(r6_mul(10.f, mfmax)), (float)(one)))), (float)(sh(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2825
sm(k) = mynn_max((float)(sm(k)), (float)(r6_mul(0.04f, mynn_min((float)(cldavg), (float)(p5)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2826
sh(k) = mynn_max((float)(r6_mul(0.04f, mynn_min((float)(cldavg), (float)(p5)))), (float)(sh(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2828
if (((tkeprod_dn(k)) > (1e-6f))) { sm(k) = mynn_max((float)(sm(k)), (float)(0.01f)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2829
if (((tkeprod_dn(k)) > (1e-6f))) { sh(k) = mynn_max((float)(sh(k)), (float)(0.01f)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2831
if (((((closure) >= (2.7f))) && (((closure) < (3.0f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2832
sm(k) = mynn_max((float)(sm(k)), (float)(mynn_min((float)(mynn_max((float)(r6_mul(three, qpe(k))), (float)(zero))), (float)(p2))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2833
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2835
elq = r6_mul(el(k), qkw(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2836
elqt = r6_mul(el(k), qtw(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2837
elh = mynn_dmul(elq, qdiv);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2841
pdk(k) = r6_add(r6_add(r6_mul(elq, r6_add(r6_add(r6_mul(sm(k), gm(k)), r6_mul(sh(k), gh(k))), gamv)), r6_mul(p5, tkeprod_dn(k))), r6_mul(p5, tkeprod_up(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2845
pdt(k) = r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dtl(k)), gamt)), dtl(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2846
pdq(k) = r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dqw(k)), gamq)), dqw(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2847
pdc(k) = r6_add(r6_mul(r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dtl(k)), gamt)), dqw(k)), p5), r6_mul(r6_mul(r6_mul(elh, r6_add(r6_mul(sh(k), dqw(k)), gamq)), dtl(k)), p5));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2851
tcd(k) = r6_mul(elq, gamt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2852
qcd(k) = r6_mul(elq, gamq);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2855
dfm(k) = r6_div(r6_mul(elqt, sm(k)), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2856
dfh(k) = r6_div(r6_mul(elq, sh(k)), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2860
dfq(k) = dfm(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2863
if (((tke_budget) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2879
qshear1(k) = r6_mul(r6_mul(elq, sm(k)), gm(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2887
qbuoy1(k) = r6_add(r6_add(r6_mul(elq, r6_add(r6_mul(gh(k), sh(k)), gamv)), r6_mul(p5, tkeprod_dn(k))), r6_mul(p5, tkeprod_up(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2895
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2897
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; a2fac = mynn_lane_bcast(a2fac, mynn_src1); abk = mynn_lane_bcast(abk, mynn_src1); afk = mynn_lane_bcast(afk, mynn_src1); cldavg = mynn_lane_bcast(cldavg, mynn_src1); dlsq = mynn_lane_bcast(dlsq, mynn_src1); duz = mynn_lane_bcast(duz, mynn_src1); dzk = mynn_lane_bcast(dzk, mynn_src1); e1 = mynn_lane_bcast(e1, mynn_src1); e2 = mynn_lane_bcast(e2, mynn_src1); e3 = mynn_lane_bcast(e3, mynn_src1); e4 = mynn_lane_bcast(e4, mynn_src1); eden = mynn_lane_bcast(eden, mynn_src1); elh = mynn_lane_bcast(elh, mynn_src1); elq = mynn_lane_bcast(elq, mynn_src1); elqt = mynn_lane_bcast(elqt, mynn_src1); elsq = mynn_lane_bcast(elsq, mynn_src1); gamq = mynn_lane_bcast(gamq, mynn_src1); gamt = mynn_lane_bcast(gamt, mynn_src1); gamv = mynn_lane_bcast(gamv, mynn_src1); ghel = mynn_lane_bcast(ghel, mynn_src1); gmel = mynn_lane_bcast(gmel, mynn_src1); gmelq = mynn_lane_bcast(gmelq, mynn_src1); mfmax = mynn_lane_bcast(mfmax, mynn_src1); prlim = mynn_lane_bcast(prlim, mynn_src1); q2sq = mynn_lane_bcast(q2sq, mynn_src1); q3sq = mynn_lane_bcast(q3sq, mynn_src1); qdiv = mynn_lane_bcast(qdiv, mynn_src1); ri = mynn_lane_bcast(ri, mynn_src1); sh20 = mynn_lane_bcast(sh20, mynn_src1); sh25max = mynn_lane_bcast(sh25max, mynn_src1); sh25min = mynn_lane_bcast(sh25min, mynn_src1); shb = mynn_lane_bcast(shb, mynn_src1); sm20 = mynn_lane_bcast(sm20, mynn_src1); sm25max = mynn_lane_bcast(sm25max, mynn_src1); sm25min = mynn_lane_bcast(sm25min, mynn_src1); wt = mynn_lane_bcast(wt, mynn_src1); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2900
dfm(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2901
dfh(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2902
dfq(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2903
pdk(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2904
pdt(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2905
pdq(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2906
pdc(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2907
tcd(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2908
qcd(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2910
tcd(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2911
qcd(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2914
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2915
dzk = dz(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2916
tcd(k) = r6_div(r6_sub(tcd(((k) + (1))), tcd(k)), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2917
qcd(k) = r6_div(r6_sub(qcd(((k) + (1))), qcd(k)), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2918
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2920
if (((spp_pbl) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2921
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2922
dfm(k) = r6_add(dfm(k), r6_mul(r6_mul(r6_mul(dfm(k), pattern_spp_pbl1(k)), 1.5f), mynn_max((float)(mynn_exp(r6_neg(r6_div(mynn_max((float)(zero), (float)(r6_sub(zw(k), 8000.f))), 2000.f)))), (float)(0.001f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2923
dfh(k) = r6_add(dfh(k), r6_mul(r6_mul(r6_mul(dfh(k), pattern_spp_pbl1(k)), 1.5f), mynn_max((float)(mynn_exp(r6_neg(r6_div(mynn_max((float)(zero), (float)(r6_sub(zw(k), 8000.f))), 2000.f)))), (float)(0.001f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2924
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2925
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:2981-3412
__device__ __noinline__ void f_mym_predict(int kts, int kte, float closure, float delt, MynnArray<float, 1> dz, float ust, float flt, float flq, float pmz, float phh, MynnArray<float, 1> el, MynnArray<float, 1> dfq, MynnArray<float, 1> rho, MynnArray<float, 1> pdk, MynnArray<float, 1> pdt, MynnArray<float, 1> pdq, MynnArray<float, 1> pdc, MynnArray<float, 1> qke, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> s_aw1, MynnArray<float, 1> s_awqke1, MynnArray<float, 1> s_awqsq1, int bl_mynn_edmf, int bl_mynn_edmf_tke, MynnArray<float, 1> qwt1, MynnArray<float, 1> qdiss1, int tke_budget, float upwind) {
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3027
if (((bl_mynn_edmf_tke) == (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3028
onoff = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3029
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3030
onoff = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3031
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3034
vkz = r6_mul(r6_mul(karman, p5), dz(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3038
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3040
qkw(k) = mynn_sqrt(mynn_max((float)(qke(k)), (float)(qkemin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3041
df3q(k) = r6_mul(sqfac, dfq(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3042
dtz(k) = r6_div(delt, dz(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3043
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3048
rhoz(kts) = rho(kts);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3049
rhoinv(kts) = r6_div(one, rho(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3050
kqdz(kts) = r6_mul(rhoz(kts), df3q(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3051
kmdz(kts) = r6_mul(rhoz(kts), dfq(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3052
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3053
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3054
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3055
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3056
kqdz(k) = r6_mul(rhoz(k), df3q(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3057
kmdz(k) = r6_mul(rhoz(k), dfq(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3058
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3059
rhoz(((kte) + (1))) = rhoz(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3060
kqdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), df3q(kte));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3061
kmdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfq(kte));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3063
if (((bl_mynn_edmf) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3065
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3066
kqdz(k) = mynn_max((float)(kqdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3067
kqdz(k) = mynn_max((float)(kqdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3068
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3069
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3074
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3075
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3080
pdk1 = r6_div(r6_mul(r6_mul(two, mynn_ipow<3>(ust)), pmz), vkz);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3081
phm = r6_div(r6_mul(r6_div(two, ust), phh), vkz);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3082
pdt1 = r6_mul(phm, mynn_ipow<2>(flt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3083
pdq1 = r6_mul(phm, mynn_ipow<2>(flq));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3084
pdc1 = r6_mul(r6_mul(phm, flt), flq);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3087
pdk(kts) = r6_sub(pdk1, pdk(((kts) + (1))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3092
pdt(kts) = pdt(((kts) + (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3093
pdq(kts) = pdq(((kts) + (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3094
pdc(kts) = pdc(((kts) + (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3100
{ const int mynn_lo3 = (kts), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3101
b1l = r6_mul(r6_mul(b1, p5), r6_add(el(((k) + (1))), el(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3102
bp(k) = r6_div(r6_mul(two, qkw(k)), b1l);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3103
rp(k) = r6_add(pdk(((k) + (1))), pdk(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3104
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; b1l = mynn_lane_bcast(b1l, mynn_src3); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3105
bp(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3108
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3110
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3111
upcont(k) = r6_sub(s_awqke1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qke(k), upwind), r6_mul(qke(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3112
dncont(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3113
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3115
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3116
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3118
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3119
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3120
b(1) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), kqdz(((k) + (1)))), rhoinv(k))), r6_mul(bp(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3121
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), kqdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3122
d(1) = r6_sub(r6_add(mynn_max((float)(qke(k)), (float)(qkemin)), r6_mul(rp(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3125
{ const int mynn_lo5 = (((kts) + (1))), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3126
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kqdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3127
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kqdz(k), kqdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3128
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kqdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3129
d(k) = r6_sub(r6_add(mynn_max((float)(qke(k)), (float)(qkemin)), r6_mul(rp(k), delt)), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3131
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3133
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3134
{ const int mynn_lo6 = (kts), mynn_hi6 = (((kte) - (1))), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3140
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), kqdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), s_aw1(k)), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3142
b(k) = r6_add(r6_add(r6_add(1.f, r6_mul(r6_mul(dtz(k), r6_add(kqdz(k), kqdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), onoff)), r6_mul(bp(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3145
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kqdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3147
d(k) = r6_add(r6_add(r6_mul(rp(k), delt), qke(k)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqke1(k), s_awqke1(((k) + (1))))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3149
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3150
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3158
a(kte) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3159
b(kte) = 1.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3160
c(kte) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3161
d(kte) = qke(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3164
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3166
{ const int mynn_lo7 = (kts), mynn_hi7 = (kte), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3168
qke(k) = mynn_max((float)(x(k)), (float)(qkemin));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3169
qke(k) = mynn_min((float)(qke(k)), (float)(150.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3170
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3174
if (((tke_budget) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3176
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tke_up).size(); mynn_element += lane_width) { tke_up.linear(mynn_element) = r6_mul(p5, qke.linear(mynn_element)); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3177
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dzinv).size(); mynn_element += lane_width) { dzinv.linear(mynn_element) = r6_div(one, dz.linear(mynn_element)); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3178
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3179
qwt1(k) = r6_mul(dzinv(k), r6_add(r6_sub(r6_mul(kqdz(((k) + (1))), r6_sub(tke_up(((k) + (1))), tke_up(k))), r6_mul(kqdz(k), tke_up(k))), r6_mul(r6_mul(r6_mul(p5, rhoinv(k)), r6_add(r6_add(r6_mul(s_aw1(((k) + (1))), tke_up(((k) + (1)))), r6_mul(r6_sub(s_aw1(((k) + (1))), s_aw1(k)), tke_up(k))), r6_sub(s_awqke1(k), s_awqke1(((k) + (1)))))), onoff)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3184
{ const int mynn_lo8 = (((kts) + (1))), mynn_hi8 = (((kte) - (1))), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3185
qwt1(k) = r6_mul(dzinv(k), r6_add(r6_sub(r6_mul(kqdz(((k) + (1))), r6_sub(tke_up(((k) + (1))), tke_up(k))), r6_mul(kqdz(k), r6_sub(tke_up(k), tke_up(((k) - (1)))))), r6_mul(r6_mul(r6_mul(p5, rhoinv(k)), r6_add(r6_sub(r6_add(r6_mul(s_aw1(((k) + (1))), tke_up(((k) + (1)))), r6_mul(r6_sub(s_aw1(((k) + (1))), s_aw1(k)), tke_up(k))), r6_mul(s_aw1(k), tke_up(((k) - (1))))), r6_sub(s_awqke1(k), s_awqke1(((k) + (1)))))), onoff)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3191
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3192
k = kte;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3193
qwt1(k) = r6_mul(dzinv(k), r6_add(r6_neg(r6_mul(kqdz(k), r6_sub(tke_up(k), tke_up(((k) - (1)))))), r6_mul(r6_mul(r6_mul(p5, rhoinv(k)), r6_add(r6_sub(r6_neg(r6_mul(s_aw1(k), tke_up(k))), r6_mul(s_aw1(k), tke_up(((k) - (1))))), s_awqke1(k))), onoff)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3196
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qdiss1).size(); mynn_element += lane_width) { qdiss1.linear(mynn_element) = r6_mul(bp.linear(mynn_element), tke_up.linear(mynn_element)); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3197
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3200
if (((closure) > (2.5f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3204
{ const int mynn_lo9 = (kts), mynn_hi9 = (((kte) - (1))), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3205
b2l = r6_mul(r6_mul(b2, p5), r6_add(el(((k) + (1))), el(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3206
bp(k) = r6_div(r6_mul(two, qkw(k)), b2l);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3207
rp(k) = r6_add(pdq(((k) + (1))), pdq(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3208
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; if (mynn_n9 > 0) { const int mynn_src9 = (mynn_n9 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src9); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3209
bp(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3212
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3213
{ const int mynn_lo10 = (((kts) + (1))), mynn_hi10 = (((kte) - (1))), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3214
upcont(k) = r6_sub(s_awqsq1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qsq(k), upwind), r6_mul(qsq(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3215
dncont(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3216
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3218
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3219
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3221
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3222
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3223
b(1) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), kmdz(((k) + (1)))), rhoinv(k))), r6_mul(bp(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3224
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3225
d(1) = r6_sub(r6_add(mynn_max((float)(zero), (float)(qsq(k))), r6_mul(rp(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3228
{ const int mynn_lo11 = (((kts) + (1))), mynn_hi11 = (((kte) - (1))), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { k = mynn_lo11 + mynn_it11*mynn_st11;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3229
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3230
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3231
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3232
d(k) = r6_sub(r6_add(mynn_max((float)(zero), (float)(qsq(k))), r6_mul(rp(k), delt)), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3234
} mynn_lane_sync(); k = mynn_lo11 + mynn_n11*mynn_st11; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3236
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3238
{ const int mynn_lo12 = (kts), mynn_hi12 = (((kte) - (1))), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { k = mynn_lo12 + mynn_it12*mynn_st12;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3239
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), s_aw1(k)), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3241
b(k) = r6_add(r6_add(r6_add(1.f, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), onoff)), r6_mul(bp(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3244
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(0.5f, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3246
d(k) = r6_add(r6_add(r6_mul(rp(k), delt), qsq(k)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqsq1(k), s_awqsq1(((k) + (1))))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3248
} mynn_lane_sync(); k = mynn_lo12 + mynn_n12*mynn_st12; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3249
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3251
a(kte) = r6_neg(one);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3252
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3253
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3254
d(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3257
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3259
{ const int mynn_lo13 = (kts), mynn_hi13 = (kte), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { k = mynn_lo13 + mynn_it13*mynn_st13;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3261
qsq(k) = mynn_max((float)(x(k)), (float)(1e-17f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3262
qsq(k) = mynn_min((float)(qsq(k)), (float)(6e-6f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3263
} mynn_lane_sync(); k = mynn_lo13 + mynn_n13*mynn_st13; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3265
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3267
{ const int mynn_lo14 = (kts), mynn_hi14 = (((kte) - (1))), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { k = mynn_lo14 + mynn_it14*mynn_st14;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3268
if (((qkw(k)) <= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3269
b2l = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3270
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3271
b2l = r6_div(r6_mul(r6_mul(b2, 0.25f), r6_add(el(((k) + (1))), el(k))), qkw(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3272
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3273
qsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdq(((k) + (1))), pdq(k)))), (float)(1e-17f))), (float)(6e-6f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3274
} mynn_lane_sync(); k = mynn_lo14 + mynn_n14*mynn_st14; if (mynn_n14 > 0) { const int mynn_src14 = (mynn_n14 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src14); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3275
qsq(kte) = qsq(((kte) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3276
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3280
if (((closure) >= (2.7f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3286
{ const int mynn_lo15 = (kts), mynn_hi15 = (((kte) - (1))), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { k = mynn_lo15 + mynn_it15*mynn_st15;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3287
b2l = r6_mul(r6_mul(b2, p5), r6_add(el(((k) + (1))), el(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3288
bp(k) = r6_div(r6_mul(two, qkw(k)), b2l);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3289
rp(k) = r6_add(pdt(((k) + (1))), pdt(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3290
} mynn_lane_sync(); k = mynn_lo15 + mynn_n15*mynn_st15; if (mynn_n15 > 0) { const int mynn_src15 = (mynn_n15 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src15); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3300
{ const int mynn_lo16 = (kts), mynn_hi16 = (((kte) - (1))), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { k = mynn_lo16 + mynn_it16*mynn_st16;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3306
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3307
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3308
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3309
d(k) = r6_add(r6_mul(rp(k), delt), mynn_max((float)(zero), (float)(tsq(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3310
} mynn_lane_sync(); k = mynn_lo16 + mynn_n16*mynn_st16; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3319
a(kte) = r6_neg(1.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3320
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3321
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3322
d(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3325
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3327
{ const int mynn_lo17 = (kts), mynn_hi17 = (kte), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { k = mynn_lo17 + mynn_it17*mynn_st17;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3329
tsq(k) = mynn_min((float)(mynn_max((float)(zero), (float)(x(k)))), (float)(three));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3330
} mynn_lane_sync(); k = mynn_lo17 + mynn_n17*mynn_st17; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3331
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3333
{ const int mynn_lo18 = (kts), mynn_hi18 = (((kte) - (1))), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { k = mynn_lo18 + mynn_it18*mynn_st18;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3334
if (((qkw(k)) <= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3335
b2l = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3336
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3337
b2l = r6_div(r6_mul(r6_mul(b2, 0.25f), r6_add(el(((k) + (1))), el(k))), qkw(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3338
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3339
tsq(k) = mynn_min((float)(mynn_max((float)(r6_mul(b2l, r6_add(pdt(((k) + (1))), pdt(k)))), (float)(zero))), (float)(three));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3340
} mynn_lane_sync(); k = mynn_lo18 + mynn_n18*mynn_st18; if (mynn_n18 > 0) { const int mynn_src18 = (mynn_n18 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src18); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3341
tsq(kte) = tsq(((kte) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3342
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3345
if (((closure) >= (3.0f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3347
{ const int mynn_lo19 = (kts), mynn_hi19 = (((kte) - (1))), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { k = mynn_lo19 + mynn_it19*mynn_st19;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3348
b2l = r6_mul(r6_mul(b2, p5), r6_add(el(((k) + (1))), el(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3349
bp(k) = r6_div(r6_mul(two, qkw(k)), b2l);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3350
rp(k) = r6_add(pdc(((k) + (1))), pdc(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3351
} mynn_lane_sync(); k = mynn_lo19 + mynn_n19*mynn_st19; if (mynn_n19 > 0) { const int mynn_src19 = (mynn_n19 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src19); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3360
{ const int mynn_lo20 = (kts), mynn_hi20 = (((kte) - (1))), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { k = mynn_lo20 + mynn_it20*mynn_st20;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3366
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3367
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(bp(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3368
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3369
d(k) = r6_add(r6_mul(rp(k), delt), cov(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3370
} mynn_lane_sync(); k = mynn_lo20 + mynn_n20*mynn_st20; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3379
a(kte) = r6_neg(1.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3380
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3381
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3382
d(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3385
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3387
{ const int mynn_lo21 = (kts), mynn_hi21 = (kte), mynn_st21 = (1); const int mynn_n21 = mynn_trip(mynn_lo21, mynn_hi21, mynn_st21); mynn_lane_sync(); for (int mynn_it21 = lane_sub; mynn_it21 < mynn_n21; mynn_it21 += lane_width) { k = mynn_lo21 + mynn_it21*mynn_st21;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3389
cov(k) = x(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3390
} mynn_lane_sync(); k = mynn_lo21 + mynn_n21*mynn_st21; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3392
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3395
{ const int mynn_lo22 = (kts), mynn_hi22 = (((kte) - (1))), mynn_st22 = (1); const int mynn_n22 = mynn_trip(mynn_lo22, mynn_hi22, mynn_st22); mynn_lane_sync(); for (int mynn_it22 = lane_sub; mynn_it22 < mynn_n22; mynn_it22 += lane_width) { k = mynn_lo22 + mynn_it22*mynn_st22;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3396
if (((qkw(k)) <= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3397
b2l = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3398
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3399
b2l = r6_div(r6_mul(r6_mul(b2, 0.25f), r6_add(el(((k) + (1))), el(k))), qkw(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3400
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3401
cov(k) = r6_mul(b2l, r6_add(pdc(((k) + (1))), pdc(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3402
} mynn_lane_sync(); k = mynn_lo22 + mynn_n22*mynn_st22; if (mynn_n22 > 0) { const int mynn_src22 = (mynn_n22 - 1) % lane_width; b2l = mynn_lane_bcast(b2l, mynn_src22); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3403
cov(kte) = cov(((kte) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3405
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3448-3909
__device__ __noinline__ void f_mym_condensation(int kts, int kte, float dx, MynnArray<float, 1> dz, MynnArray<float, 1> zagl1, MynnArray<float, 1> zw, float xland, MynnArray<float, 1> thl, MynnArray<float, 1> qw, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> sh, MynnArray<float, 1> el, int bl_mynn_cloudpdf, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> qi_bl1, MynnArray<float, 1> cldfra_bl1, float pblh1, float hfx1, MynnArray<float, 1> vt, MynnArray<float, 1> vq, MynnArray<float, 1> th, MynnArray<float, 1> sgm, float closure, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
zagl1.rebase({kts});
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
float damp;
float pblh2;
float cfmax;
float theta1;
float theta2;
float ht1;
float ht2;
int k_tropo;
float qw_pert;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3529
for (k=((kte) - (3)); ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3530
theta1 = th(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3531
theta2 = th(((k) + (2)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3532
ht1 = r6_mul(44307.692f, r6_sub(one, mynn_pow(r6_div(p(k), 101325.f), 0.190f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3533
ht2 = r6_mul(44307.692f, r6_sub(one, mynn_pow(r6_div(p(((k) + (2))), 101325.f), 0.190f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3534
if (((((((r6_div(r6_sub(theta2, theta1), r6_sub(ht2, ht1))) < (r6_div(10.f, 1500.f)))) && (((ht1) < (19000.f))))) && (((ht1) > (4000.f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3536
goto mynn_label_86;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3537
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3538
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3539
mynn_label_86: ;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3540
k_tropo = mynn_max((int)(((kts) + (2))), (int)(((k) + (2))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3542
switch (bl_mynn_cloudpdf) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3544
case 0: {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3546
{ const int mynn_lo0 = (kts), mynn_hi0 = (((kte) - (1))), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3547
t = r6_mul(th(k), exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3560
esat = f_esat_blend(t);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3563
qsl = r6_div(r6_mul(ep_2, esat), mynn_max((float)(r6_sub(p(k), r6_mul(ep_3, esat))), (float)(1.e-4f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3565
dqsl = r6_div(r6_mul(r6_mul(qsl, ep_2), xlv), r6_mul(r_d, mynn_ipow<2>(t)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3567
alp(k) = r6_div(one, r6_add(one, r6_mul(dqsl, xlvcp)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3568
bet(k) = r6_mul(dqsl, exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3572
t3sq = mynn_max((float)(zero), (float)(tsq(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3573
r3sq = mynn_max((float)(zero), (float)(qsq(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3574
c3sq = cov(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3575
c3sq = mynn_sign(mynn_min((double)(mynn_abs(c3sq)), (double)(mynn_sqrt(mynn_dmul(t3sq, r3sq)))), c3sq);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3576
r3sq = mynn_dsub(mynn_dadd(r3sq, mynn_dmul(mynn_ipow<2>(bet(k)), t3sq)), mynn_dmul(r6_mul(2.0f, bet(k)), c3sq));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3578
qmq = r6_sub(qw(k), qsl);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3580
sgm(k) = mynn_sqrt(mynn_max((double)(r3sq), (double)(1.0e-10)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3582
q1(k) = r6_div(qmq, sgm(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3584
cldfra_bl1(k) = r6_mul(0.5f, r6_add(one, mynn_erf(r6_mul(q1(k), rr2))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3586
q1k = q1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3587
eq1 = r6_mul(rrp, mynn_exp(r6_mul(r6_neg(r6_mul(0.5f, q1k)), q1k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3588
qll = mynn_max((float)(zero), (float)(r6_add(r6_mul(cldfra_bl1(k), q1k), eq1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3590
ql(k) = r6_mul(r6_mul(alp(k), sgm(k)), qll);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3592
liq_frac = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(t, 240.0f), 29.0f)), (float)(zero))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3593
qc_bl1(k) = r6_mul(liq_frac, ql(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3594
qi_bl1(k) = r6_mul(r6_sub(one, liq_frac), ql(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3597
q2p = r6_div(xlvcp, exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3598
pt = r6_add(thl(k), r6_mul(q2p, ql(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3601
qt = r6_sub(r6_add(one, r6_mul(p608, qw(k))), r6_mul(r6_mul(r6_add(1.f, p608), r6_add(qc_bl1(k), qi_bl1(k))), cldfra_bl1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3602
rac = r6_mul(r6_mul(alp(k), r6_sub(cldfra_bl1(k), r6_mul(qll, eq1))), r6_sub(r6_mul(q2p, qt), r6_mul(r6_add(1.f, p608), pt)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3607
vt(k) = r6_sub(r6_sub(qt, one), r6_mul(rac, bet(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3608
vq(k) = r6_add(r6_sub(r6_mul(p608, pt), tv0), rac);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3610
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; c3sq = mynn_lane_bcast(c3sq, mynn_src0); dqsl = mynn_lane_bcast(dqsl, mynn_src0); eq1 = mynn_lane_bcast(eq1, mynn_src0); esat = mynn_lane_bcast(esat, mynn_src0); liq_frac = mynn_lane_bcast(liq_frac, mynn_src0); pt = mynn_lane_bcast(pt, mynn_src0); q1k = mynn_lane_bcast(q1k, mynn_src0); q2p = mynn_lane_bcast(q2p, mynn_src0); qll = mynn_lane_bcast(qll, mynn_src0); qmq = mynn_lane_bcast(qmq, mynn_src0); qsl = mynn_lane_bcast(qsl, mynn_src0); qt = mynn_lane_bcast(qt, mynn_src0); r3sq = mynn_lane_bcast(r3sq, mynn_src0); rac = mynn_lane_bcast(rac, mynn_src0); t = mynn_lane_bcast(t, mynn_src0); t3sq = mynn_lane_bcast(t3sq, mynn_src0); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3612
break; } case 1: case (-(1)): {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3614
{ const int mynn_lo1 = (kts), mynn_hi1 = (((kte) - (1))), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3615
t = r6_mul(th(k), exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3617
esat = f_esat_blend(t);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3620
qsl = r6_div(r6_mul(ep_2, esat), mynn_max((float)(r6_sub(p(k), r6_mul(ep_3, esat))), (float)(1.e-4f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3622
dqsl = r6_div(r6_mul(r6_mul(qsl, ep_2), xlv), r6_mul(r_d, mynn_ipow<2>(t)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3624
alp(k) = r6_div(one, r6_add(one, r6_mul(dqsl, xlvcp)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3625
bet(k) = r6_mul(dqsl, exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3627
if (((k) == (kts))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3628
dzk = r6_mul(0.5f, dz(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3629
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3630
dzk = dz(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3631
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3632
dth = r6_sub(r6_mul(0.5f, r6_add(thl(((k) + (1))), thl(k))), r6_mul(0.5f, r6_add(thl(k), thl(mynn_max((int)(((k) - (1))), (int)(kts))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3633
dqw = r6_sub(r6_mul(0.5f, r6_add(qw(((k) + (1))), qw(k))), r6_mul(0.5f, r6_add(qw(k), qw(mynn_max((int)(((k) - (1))), (int)(kts))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3634
sgm(k) = mynn_sqrt(mynn_max((float)(r6_mul(r6_div(r6_mul(r6_mul(r6_mul(mynn_ipow<2>(alp(k)), mynn_max((float)(mynn_ipow<2>(el(k))), (float)(p1))), b2), mynn_max((float)(sh(k)), (float)(0.03f))), four), mynn_ipow<2>(r6_sub(r6_div(dqw, dzk), r6_mul(bet(k), r6_div(dth, dzk)))))), (float)(1.0e-10f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3637
qmq = r6_sub(qw(k), qsl);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3638
q1(k) = r6_div(qmq, sgm(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3639
cldfra_bl1(k) = r6_mul(p5, r6_add(one, mynn_erf(r6_mul(q1(k), rr2))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3644
q1k = q1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3645
eq1 = r6_mul(rrp, mynn_exp(r6_mul(r6_neg(r6_mul(0.5f, q1k)), q1k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3646
qll = mynn_max((float)(zero), (float)(r6_add(r6_mul(cldfra_bl1(k), q1k), eq1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3648
ql(k) = r6_mul(r6_mul(alp(k), sgm(k)), qll);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3649
liq_frac = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(t, 240.0f), 29.0f)), (float)(zero))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3650
qc_bl1(k) = r6_mul(liq_frac, ql(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3651
qi_bl1(k) = r6_mul(r6_sub(one, liq_frac), ql(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3654
q2p = r6_div(xlvcp, exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3655
pt = r6_add(thl(k), r6_mul(q2p, ql(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3658
qt = r6_sub(r6_add(one, r6_mul(p608, qw(k))), r6_mul(r6_mul(r6_add(1.f, p608), r6_add(qc_bl1(k), qi_bl1(k))), cldfra_bl1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3659
rac = r6_mul(r6_mul(alp(k), r6_sub(cldfra_bl1(k), r6_mul(qll, eq1))), r6_sub(r6_mul(q2p, qt), r6_mul(r6_add(1.f, p608), pt)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3664
vt(k) = r6_sub(r6_sub(qt, one), r6_mul(rac, bet(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3665
vq(k) = r6_add(r6_sub(r6_mul(p608, pt), tv0), rac);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3667
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; dqsl = mynn_lane_bcast(dqsl, mynn_src1); dqw = mynn_lane_bcast(dqw, mynn_src1); dth = mynn_lane_bcast(dth, mynn_src1); dzk = mynn_lane_bcast(dzk, mynn_src1); eq1 = mynn_lane_bcast(eq1, mynn_src1); esat = mynn_lane_bcast(esat, mynn_src1); liq_frac = mynn_lane_bcast(liq_frac, mynn_src1); pt = mynn_lane_bcast(pt, mynn_src1); q1k = mynn_lane_bcast(q1k, mynn_src1); q2p = mynn_lane_bcast(q2p, mynn_src1); qll = mynn_lane_bcast(qll, mynn_src1); qmq = mynn_lane_bcast(qmq, mynn_src1); qsl = mynn_lane_bcast(qsl, mynn_src1); qt = mynn_lane_bcast(qt, mynn_src1); rac = mynn_lane_bcast(rac, mynn_src1); t = mynn_lane_bcast(t, mynn_src1); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3669
break; } case 2: case (-(2)): {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3673
pblh2 = mynn_max((float)(pblh1), (float)(ten));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3674
{ const int mynn_lo2 = (kts), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3676
t = r6_mul(th(k), exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3677
xl = f_xl_blend(t);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3678
qsat_tk = f_qsat_blend(t, p(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3679
rh(k) = mynn_max((float)(mynn_min((float)(r6_div(qw(k), mynn_max((float)(qsat_tk), (float)(1e-10f)))), (float)(rhmax))), (float)(0.001f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3682
dqsl = r6_div(r6_mul(r6_mul(qsat_tk, ep_2), xlv), r6_mul(r_d, mynn_ipow<2>(t)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3683
alp(k) = r6_div(one, r6_add(one, r6_mul(dqsl, xlvcp)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3684
bet(k) = r6_mul(dqsl, exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3686
rsl = r6_div(r6_mul(xl, qsat_tk), r6_mul(r_v, mynn_ipow<2>(t)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3688
cpm = r6_add(cp, r6_mul(qw(k), cpv));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3689
a(k) = r6_div(one, r6_add(one, r6_div(r6_mul(xl, rsl), cpm)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3690
b(k) = r6_mul(a(k), rsl);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3693
qw_pert = r6_add(qw(k), r6_mul(r6_mul(r6_mul(qw(k), p5), pattern_spp_pbl1(k)), ((float)(spp_pbl))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3696
qmq = r6_sub(qw_pert, qsat_tk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3700
r3sq = mynn_max((float)(zero), (float)(qsq(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3701
t3sq = mynn_max((float)(zero), (float)(tsq(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3702
c3sq = cov(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3703
c3sq = mynn_sign(mynn_min((double)(mynn_abs(c3sq)), (double)(mynn_sqrt(mynn_dmul(t3sq, r3sq)))), c3sq);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3705
if (((closure) < (2.7f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3706
r3sq = r3sq;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3707
} else if (((((closure) >= (2.7f))) && (((closure) < (3.0f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3708
r3sq = mynn_dadd(r3sq, mynn_dmul(mynn_ipow<2>(bet(k)), t3sq));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3709
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3710
r3sq = mynn_dsub(mynn_dadd(r3sq, mynn_dmul(mynn_ipow<2>(bet(k)), t3sq)), mynn_dmul(r6_mul(two, bet(k)), c3sq));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3711
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3714
sgm(k) = mynn_max((float)(mynn_sqrt(((float)(r3sq)))), (float)(1e-13f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3716
sgm(k) = mynn_min((float)(sgm(k)), (float)(r6_mul(qw(k), p5)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3719
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(dz(k), 100.f)), (float)(zero)), 500.f)), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3720
sgm(k) = r6_add(sgm(k), r6_mul(r6_mul(sgm(k), p2), wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3722
sgmq = sgm(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3723
sgmc = sgm(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3726
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl1(k), r6_add(pblh2, 10.f))), (float)(zero)), 250.f)), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3727
clim = r6_add(r6_mul(clim_pbl, r6_sub(one, wt)), r6_mul(clim_trp, wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3728
zsl = mynn_min((float)(mynn_max((float)(r6_mul(p1, pblh2)), (float)(50.f))), (float)(150.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3729
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl1(k), zsl)), (float)(zero)), 150.f)), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3730
clim = r6_add(r6_mul(clim_sfc, r6_sub(one, wt)), r6_mul(clim, wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3731
sgmc = mynn_max((float)(sgmc), (float)(r6_mul(qw(k), clim)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3733
sgmc = mynn_max((float)(sgmc), (float)(1e-13f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3735
if (((qmq) >= (zero))) { sgmc = mynn_max((float)(r6_mul(0.02f, qw(k))), (float)(sgmc)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3737
q1(k) = r6_div(qmq, sgmc);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3741
wt2 = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl1(k), pblh2)), (float)(zero)), 300.f)), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3743
if (((((r6_add(qi(k), qs(k))) > (1.e-10f))) && (((zagl1(k)) > (pblh2))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3744
rh_adj = mynn_min((float)(r6_add(rhcrit, r6_mul(r6_mul(wt2, 0.037f), mynn_max((float)(r6_add(ten, mynn_log10(r6_add(qi(k), qs(k))))), (float)(zero))))), (float)(rhmax));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3745
rh_adj = mynn_max((float)(rh(k)), (float)(rh_adj));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3747
q1_rh = r6_add(r6_neg(3.f), r6_div(r6_mul(three, r6_sub(rh_adj, rhcrit)), r6_sub(one, rhcrit)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3748
q1(k) = mynn_max((float)(q1_rh), (float)(q1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3749
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3751
if (((((qc(k)) > (1.e-5f))) && (((zagl1(k)) > (pblh2))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3752
rh_adj = mynn_min((float)(r6_add(rhcrit, r6_mul(r6_mul(wt2, 0.07f), mynn_max((float)(r6_add(five, mynn_log10(qc(k)))), (float)(zero))))), (float)(rhmax));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3753
rh_adj = mynn_max((float)(rh(k)), (float)(rh_adj));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3755
q1_rh = r6_add(r6_neg(3.f), r6_div(r6_mul(three, r6_sub(rh_adj, rhcrit)), r6_sub(one, rhcrit)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3756
q1(k) = mynn_max((float)(q1_rh), (float)(q1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3757
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3759
q1k = q1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3773
wt2 = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(zagl1(k), r6_sub(pblh1, 100.f)), 200.f)), (float)(zero))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3776
cldfra_qsq0 = mynn_max((float)(zero), (float)(mynn_min((float)(r6_add(p5, r6_mul(0.35f, mynn_atan(r6_mul(3.6f, r6_add(q1k, 0.05f)))))), (float)(one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3778
cldfra_qsq1 = mynn_max((float)(zero), (float)(mynn_min((float)(r6_add(p5, r6_mul(0.39f, mynn_atan(r6_mul(1.6f, r6_add(q1k, 0.55f)))))), (float)(one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3779
cldfra_qsq = r6_add(r6_mul(cldfra_qsq0, r6_sub(one, wt2)), r6_mul(cldfra_qsq1, wt2));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3782
cldfra_rh0 = mynn_min((float)(mynn_max((float)(r6_add(r6_mul(0.56f, mynn_tanh(r6_div(r6_sub(rh(k), 0.976f), 0.030f))), p5)), (float)(zero))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3783
cldfra_rh1 = mynn_min((float)(mynn_max((float)(r6_add(r6_mul(0.55f, mynn_tanh(r6_div(r6_sub(rh(k), 0.955f), 0.055f))), p5)), (float)(zero))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3784
cldfra_rh = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3786
cldfra_bl1(k) = mynn_max((float)(cldfra_qsq), (float)(cldfra_rh));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3790
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl1(k), r6_add(pblh2, ten))), (float)(zero)), 300.f)), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3791
qlim = r6_add(r6_mul(qlim_pbl, r6_sub(one, wt)), r6_mul(qlim_trp, wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3792
zsl = mynn_min((float)(mynn_max((float)(r6_mul(p1, pblh2)), (float)(50.f))), (float)(150.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3793
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl1(k), zsl)), (float)(zero)), 200.f)), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3794
qlim = r6_add(r6_mul(qlim_sfc, r6_sub(one, wt)), r6_mul(qlim, wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3795
sgmq = mynn_max((float)(sgmq), (float)(r6_mul(qw(k), qlim)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3797
ql_water = r6_mul(mynn_min((float)(sgmq), (float)(r6_mul(0.025f, qw(k)))), cldfra_bl1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3798
ql_ice = r6_mul(mynn_min((float)(sgmq), (float)(r6_mul(0.025f, qw(k)))), cldfra_bl1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3822
if (((cldfra_bl1(k)) < (0.001f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3823
ql_ice = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3824
ql_water = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3825
cldfra_bl1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3826
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3828
liq_frac = mynn_min((float)(mynn_max((float)(r6_div(r6_sub(t, tice), r6_sub(tliq, tice))), (float)(zero))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3829
qc_bl1(k) = r6_mul(liq_frac, ql_water);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3830
qi_bl1(k) = r6_mul(r6_sub(one, liq_frac), ql_ice);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3834
if (((k) >= (k_tropo))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3835
cldfra_bl1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3836
qc_bl1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3837
qi_bl1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3838
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3844
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3845
q1k = mynn_max((float)(q1(k)), (float)(r6_neg(2.5f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3846
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3847
q1k = mynn_max((float)(q1(k)), (float)(r6_neg(2.0f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3848
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3851
if (((q1k) >= (one))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3852
fng = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3853
} else if (((((q1k) >= (r6_neg(1.7f)))) && (((q1k) < (one))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3854
fng = mynn_exp(r6_neg(r6_mul(0.4f, r6_sub(q1k, one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3855
} else if (((((q1k) >= (r6_neg(2.5f)))) && (((q1k) < (r6_neg(1.7f)))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3856
fng = r6_add(three, mynn_exp(r6_neg(r6_mul(3.8f, r6_add(q1k, 1.7f)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3857
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3858
fng = mynn_min((float)(r6_add(23.9f, mynn_exp(r6_neg(r6_mul(1.6f, r6_add(q1k, 2.5f)))))), (float)(60.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3859
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3861
cfmax = mynn_min((float)(cldfra_bl1(k)), (float)(0.6f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3863
zsl = mynn_min((float)(mynn_max((float)(r6_mul(p1, pblh2)), (float)(25.f))), (float)(hundred));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3864
wt = mynn_min((float)(r6_div(zagl1(k), zsl)), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3865
cfmax = r6_mul(cfmax, wt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3867
bb = r6_div(r6_mul(b(k), t), th(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3873
qww = r6_add(one, r6_mul(p608, qw(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3874
alpha = r6_mul(p608, th(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3875
beta = r6_sub(r6_mul(r6_div(th(k), t), r6_div(xl, cp)), r6_mul(1.61f, th(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3876
vt(k) = r6_sub(r6_sub(qww, r6_mul(r6_mul(r6_mul(cfmax, beta), bb), fng)), one);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3877
vq(k) = r6_sub(r6_add(alpha, r6_mul(r6_mul(r6_mul(cfmax, beta), a(k)), fng)), tv0);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3883
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; alpha = mynn_lane_bcast(alpha, mynn_src2); bb = mynn_lane_bcast(bb, mynn_src2); beta = mynn_lane_bcast(beta, mynn_src2); c3sq = mynn_lane_bcast(c3sq, mynn_src2); cfmax = mynn_lane_bcast(cfmax, mynn_src2); cldfra_qsq = mynn_lane_bcast(cldfra_qsq, mynn_src2); cldfra_qsq0 = mynn_lane_bcast(cldfra_qsq0, mynn_src2); cldfra_qsq1 = mynn_lane_bcast(cldfra_qsq1, mynn_src2); cldfra_rh = mynn_lane_bcast(cldfra_rh, mynn_src2); cldfra_rh0 = mynn_lane_bcast(cldfra_rh0, mynn_src2); cldfra_rh1 = mynn_lane_bcast(cldfra_rh1, mynn_src2); clim = mynn_lane_bcast(clim, mynn_src2); cpm = mynn_lane_bcast(cpm, mynn_src2); dqsl = mynn_lane_bcast(dqsl, mynn_src2); fng = mynn_lane_bcast(fng, mynn_src2); liq_frac = mynn_lane_bcast(liq_frac, mynn_src2); q1k = mynn_lane_bcast(q1k, mynn_src2); ql_ice = mynn_lane_bcast(ql_ice, mynn_src2); ql_water = mynn_lane_bcast(ql_water, mynn_src2); qlim = mynn_lane_bcast(qlim, mynn_src2); qmq = mynn_lane_bcast(qmq, mynn_src2); qsat_tk = mynn_lane_bcast(qsat_tk, mynn_src2); qw_pert = mynn_lane_bcast(qw_pert, mynn_src2); qww = mynn_lane_bcast(qww, mynn_src2); r3sq = mynn_lane_bcast(r3sq, mynn_src2); rsl = mynn_lane_bcast(rsl, mynn_src2); sgmc = mynn_lane_bcast(sgmc, mynn_src2); sgmq = mynn_lane_bcast(sgmq, mynn_src2); t = mynn_lane_bcast(t, mynn_src2); t3sq = mynn_lane_bcast(t3sq, mynn_src2); wt = mynn_lane_bcast(wt, mynn_src2); wt2 = mynn_lane_bcast(wt2, mynn_src2); xl = mynn_lane_bcast(xl, mynn_src2); zsl = mynn_lane_bcast(zsl, mynn_src2); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3885
break; } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3888
if (((bl_mynn_cloudpdf) < (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3889
{ const int mynn_lo3 = (kts), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3890
cldfra_bl1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3891
qc_bl1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3892
qi_bl1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3893
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3894
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3896
ql(kte) = ql(((kte) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3897
vt(kte) = vt(((kte) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3898
vq(kte) = vq(((kte) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3899
qc_bl1(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3900
qi_bl1(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3901
cldfra_bl1(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3902
return;
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:3915-5480
__device__ __noinline__ void f_mynn_tendencies(int kts, int kte, int i, float delt, MynnArray<float, 1> dz, MynnArray<float, 1> zw, float xland, float pblh, MynnArray<float, 1> rho, MynnArray<float, 1> delp, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> th, MynnArray<float, 1> tk, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> qnc, MynnArray<float, 1> qni, float psfc, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> thl, MynnArray<float, 1> sqv, MynnArray<float, 1> sqc, MynnArray<float, 1> sqi, MynnArray<float, 1> sqs, MynnArray<float, 1> sqw, MynnArray<float, 1> thl_tot1, MynnArray<float, 1> qc_tot1, MynnArray<float, 1> qi_tot1, MynnArray<float, 1> qnwfa, MynnArray<float, 1> qnifa, MynnArray<float, 1> qnbca, MynnArray<float, 1> ozone, float ust, float flt, float flq, float flqv, float flqc, float wspd, float uoce, float voce, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> dfm, MynnArray<float, 1> dfh, MynnArray<float, 1> dfq, MynnArray<float, 1> du, MynnArray<float, 1> dv, MynnArray<float, 1> dth, MynnArray<float, 1> dqv, MynnArray<float, 1> dqc, MynnArray<float, 1> dqi, MynnArray<float, 1> dqs, MynnArray<float, 1> dqnc, MynnArray<float, 1> dqni, MynnArray<float, 1> dqnwfa, MynnArray<float, 1> dqnifa, MynnArray<float, 1> dqnbca, MynnArray<float, 1> dozone, MynnArray<float, 1> diss_heat, MynnArray<float, 1> s_aw1, MynnArray<float, 1> s_awthl1, MynnArray<float, 1> s_awqt1, MynnArray<float, 1> s_awqv1, MynnArray<float, 1> s_awqc1, MynnArray<float, 1> s_awqi1, MynnArray<float, 1> s_awu1, MynnArray<float, 1> s_awv1, MynnArray<float, 1> s_awqnc1, MynnArray<float, 1> s_awqni1, MynnArray<float, 1> s_awqnwfa1, MynnArray<float, 1> s_awqnifa1, MynnArray<float, 1> s_awqnbca1, MynnArray<float, 1> sd_aw1, MynnArray<float, 1> sd_awthl1, MynnArray<float, 1> sd_awqt1, MynnArray<float, 1> sd_awqv1, MynnArray<float, 1> sd_awqc1, MynnArray<float, 1> sd_awqi1, MynnArray<float, 1> sd_awqnc1, MynnArray<float, 1> sd_awqni1, MynnArray<float, 1> sd_awqnwfa1, MynnArray<float, 1> sd_awqnifa1, MynnArray<float, 1> sd_awu1, MynnArray<float, 1> sd_awv1, MynnArray<float, 1> sub_thl, MynnArray<float, 1> sub_sqv, MynnArray<float, 1> sub_u, MynnArray<float, 1> sub_v, MynnArray<float, 1> det_thl, MynnArray<float, 1> det_sqv, MynnArray<float, 1> det_sqc, MynnArray<float, 1> det_u, MynnArray<float, 1> det_v, bool flag_qc, bool flag_qi, bool flag_qnc, bool flag_qni, bool flag_qs, bool flag_qnwfa, bool flag_qnifa, bool flag_qnbca, bool flag_ozone, MynnArray<float, 1> cldfra_bl1, int bl_mynn_cloudmix, int bl_mynn_mixqt, int bl_mynn_edmf, int bl_mynn_edmf_mom, int bl_mynn_mixscalars, int bl_mynn_mixaerosols, int bl_mynn_mixnumcon, float upwind) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
zw.rebase({kts});
rho.rebase({kts});
delp.rebase({kts});
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
s_awqi1.rebase({kts});
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4031
dztop = r6_mul(p5, r6_add(dz(kte), dz(((kte) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4036
if (((bl_mynn_edmf_mom) == (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4037
onoff = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4038
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4039
onoff = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4040
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4043
if (((wspd) <= (1e-6f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4044
ustovwsp = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4045
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4046
ustovwsp = r6_div(ust, wspd);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4047
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4051
rhosfc = r6_div(psfc, r6_mul(r6_mul(r_d, tk(kts)), r6_add(one, r6_mul(p608, qv(kts)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4052
dtz(kts) = r6_div(delt, dz(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4053
rhoz(kts) = rho(kts);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4054
rhoinv(kts) = r6_div(one, rho(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4055
khdz(kts) = r6_mul(rhoz(kts), dfh(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4056
kmdz(kts) = r6_mul(rhoz(kts), dfm(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4057
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4058
dtz(k) = r6_div(delt, dz(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4059
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4060
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4061
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4062
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4063
khdz(k) = r6_mul(rhoz(k), dfh(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4064
kmdz(k) = r6_mul(rhoz(k), dfm(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4065
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src0); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4066
rhoz(((kte) + (1))) = rhoz(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4067
khdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfh(kte));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4068
kmdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfm(kte));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4071
if (((bl_mynn_edmf) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4072
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (((kte) - (1))), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4073
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_mul(p5, r6_add(s_aw1(k), sd_aw1(k)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4074
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_sub(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(p5, r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4076
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_mul(p5, r6_add(s_aw1(k), sd_aw1(k)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4077
kmdz(k) = mynn_max((float)(kmdz(k)), (float)(r6_sub(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(p5, r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4079
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4080
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4082
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dth.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { dth.slice(MynnRange{kts, kte}).linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4087
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4089
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4090
upcont(k) = r6_mul(onoff, r6_sub(s_awu1(k), r6_mul(s_aw1(k), r6_add(r6_mul(u(k), upwind), r6_mul(u(((k) - (1))), r6_sub(one, upwind))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4091
dncont(k) = r6_mul(onoff, r6_sub(sd_awu1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(u(k), upwind), r6_mul(u(((k) - (1))), r6_sub(one, upwind))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4092
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4094
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4095
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4098
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4099
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4100
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_mul(r6_mul(rhosfc, ust), ustovwsp))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4101
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4102
d(1) = r6_sub(r6_add(u(k), r6_mul(r6_mul(r6_mul(r6_mul(r6_mul(dtz(k), uoce), ust), ustovwsp), rhosfc), rhoinv(k))), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4105
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4106
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4107
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4108
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4109
d(k) = r6_add(r6_add(r6_sub(u(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_u(k), delt)), r6_mul(det_u(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4111
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4113
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4115
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4117
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4118
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_mul(r6_mul(rhosfc, ust), ustovwsp))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4121
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4124
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(u(k), r6_mul(r6_mul(r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), rhoinv(k)), uoce), ust), ustovwsp)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awu1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awu1(((k) + (1)))), onoff)), r6_mul(sub_u(k), delt)), r6_mul(det_u(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4129
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4130
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4133
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4136
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4139
d(k) = r6_add(r6_add(r6_sub(r6_add(u(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awu1(k), s_awu1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awu1(k), sd_awu1(((k) + (1))))), onoff)), r6_mul(sub_u(k), delt)), r6_mul(det_u(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4142
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4144
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4159
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4160
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4161
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4162
d(kte) = u(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4165
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4168
{ const int mynn_lo5 = (kts), mynn_hi5 = (kte), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4170
du(k) = r6_div(r6_sub(x(k), u(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4171
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4177
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4179
{ const int mynn_lo6 = (((kts) + (1))), mynn_hi6 = (((kte) - (1))), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4180
upcont(k) = r6_mul(onoff, r6_sub(s_awv1(k), r6_mul(s_aw1(k), r6_add(r6_mul(v(k), upwind), r6_mul(v(((k) - (1))), r6_sub(one, upwind))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4181
dncont(k) = r6_mul(onoff, r6_sub(sd_awv1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(v(k), upwind), r6_mul(v(((k) - (1))), r6_sub(one, upwind))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4182
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4184
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4185
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4188
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4189
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4190
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_mul(r6_mul(rhosfc, ust), ustovwsp))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4191
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4192
d(1) = r6_sub(r6_add(v(k), r6_mul(r6_mul(r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), rhoinv(k)), voce), ust), ustovwsp)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4195
{ const int mynn_lo7 = (((kts) + (1))), mynn_hi7 = (((kte) - (1))), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4196
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4197
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4198
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4199
d(k) = r6_add(r6_add(r6_sub(v(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_v(k), delt)), r6_mul(det_v(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4201
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4203
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4205
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4207
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4208
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(((k) + (1))), r6_mul(r6_mul(rhosfc, ust), ustovwsp))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4211
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4214
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(v(k), r6_mul(r6_mul(r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), rhoinv(k)), voce), ust), ustovwsp)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awv1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awv1(((k) + (1)))), onoff)), r6_mul(sub_v(k), delt)), r6_mul(det_v(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4219
{ const int mynn_lo8 = (((kts) + (1))), mynn_hi8 = (((kte) - (1))), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4220
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4223
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4226
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), kmdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), onoff)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))), onoff));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4229
d(k) = r6_add(r6_add(r6_sub(r6_add(v(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awv1(k), s_awv1(((k) + (1))))), onoff)), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awv1(k), sd_awv1(((k) + (1))))), onoff)), r6_mul(sub_v(k), delt)), r6_mul(det_v(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4232
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4233
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4247
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4248
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4249
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4250
d(kte) = v(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4253
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4256
{ const int mynn_lo9 = (kts), mynn_hi9 = (kte), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4258
dv(k) = r6_div(r6_sub(x(k), v(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4259
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4264
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4266
{ const int mynn_lo10 = (((kts) + (1))), mynn_hi10 = (((kte) - (1))), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4267
upcont(k) = r6_sub(s_awthl1(k), r6_mul(s_aw1(k), r6_add(r6_mul(thl(k), upwind), r6_mul(thl(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4268
dncont(k) = r6_sub(sd_awthl1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(thl(k), upwind), r6_mul(thl(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4269
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4271
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4272
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4275
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4276
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4277
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4278
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4279
d(1) = r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(thl(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flt), rhoinv(k))), r6_mul(tcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4283
{ const int mynn_lo11 = (((kts) + (1))), mynn_hi11 = (((kte) - (1))), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { k = mynn_lo11 + mynn_it11*mynn_st11;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4284
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4285
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4286
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4287
d(k) = r6_add(r6_add(r6_add(r6_sub(thl(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4289
} mynn_lane_sync(); k = mynn_lo11 + mynn_n11*mynn_st11; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4291
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4293
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4295
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4296
b(k) = r6_sub(r6_sub(r6_add(1.f, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4299
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4302
d(k) = r6_add(r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(thl(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flt), rhoinv(k))), r6_mul(tcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awthl1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awthl1(((k) + (1))))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4308
{ const int mynn_lo12 = (((kts) + (1))), mynn_hi12 = (((kte) - (1))), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { k = mynn_lo12 + mynn_it12*mynn_st12;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4309
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4312
b(k) = r6_add(r6_add(r6_add(1.f, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4315
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4318
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(thl(k), r6_mul(tcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awthl1(k), s_awthl1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awthl1(k), sd_awthl1(((k) + (1)))))), r6_mul(diss_heat(k), delt)), r6_mul(sub_thl(k), delt)), r6_mul(det_thl(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4324
} mynn_lane_sync(); k = mynn_lo12 + mynn_n12*mynn_st12; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4325
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4340
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4341
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4342
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4343
d(kte) = thl(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4346
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4349
{ const int mynn_lo13 = (kts), mynn_hi13 = (kte), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { k = mynn_lo13 + mynn_it13*mynn_st13;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4351
thl(k) = x(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4352
} mynn_lane_sync(); k = mynn_lo13 + mynn_n13*mynn_st13; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4354
if (((bl_mynn_mixqt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4361
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4363
{ const int mynn_lo14 = (((kts) + (1))), mynn_hi14 = (((kte) - (1))), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { k = mynn_lo14 + mynn_it14*mynn_st14;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4364
upcont(k) = r6_sub(s_awqt1(k), r6_mul(s_aw1(k), r6_add(r6_mul(sqw(k), upwind), r6_mul(sqw(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4365
dncont(k) = r6_sub(sd_awqt1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqw(k), upwind), r6_mul(sqw(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4366
} mynn_lane_sync(); k = mynn_lo14 + mynn_n14*mynn_st14; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4368
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4369
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4372
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4373
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4374
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4375
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4376
d(1) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqw(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flq), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4380
{ const int mynn_lo15 = (((kts) + (1))), mynn_hi15 = (((kte) - (1))), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { k = mynn_lo15 + mynn_it15*mynn_st15;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4381
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4382
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4383
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4384
d(k) = r6_add(r6_add(r6_sub(sqw(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4386
} mynn_lane_sync(); k = mynn_lo15 + mynn_n15*mynn_st15; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4388
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4390
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4392
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4393
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4396
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4399
d(k) = r6_add(r6_sub(r6_add(r6_add(sqw(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flq), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqt1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqt1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4403
{ const int mynn_lo16 = (((kts) + (1))), mynn_hi16 = (((kte) - (1))), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { k = mynn_lo16 + mynn_it16*mynn_st16;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4404
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4407
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4410
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4413
d(k) = r6_sub(r6_add(r6_add(sqw(k), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqt1(k), s_awqt1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqt1(k), sd_awqt1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4416
} mynn_lane_sync(); k = mynn_lo16 + mynn_n16*mynn_st16; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4417
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4430
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4431
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4432
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4433
d(kte) = sqw(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4436
f_tridiag2(kte, a, b, c, d, sqw2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4442
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4443
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqw2).size(); mynn_element += lane_width) { sqw2.linear(mynn_element) = sqw.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4444
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4446
if (((bl_mynn_mixqt) == (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4451
if (((((bl_mynn_cloudmix) > (0))) && (flag_qc))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4453
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4455
{ const int mynn_lo17 = (((kts) + (1))), mynn_hi17 = (((kte) - (1))), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { k = mynn_lo17 + mynn_it17*mynn_st17;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4456
upcont(k) = r6_sub(s_awqc1(k), r6_mul(s_aw1(k), r6_add(r6_mul(sqc(k), upwind), r6_mul(sqc(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4457
dncont(k) = r6_sub(sd_awqc1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqc(k), upwind), r6_mul(sqc(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4458
} mynn_lane_sync(); k = mynn_lo17 + mynn_n17*mynn_st17; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4460
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4461
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4464
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4465
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4466
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4467
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4468
d(1) = r6_add(r6_sub(r6_add(r6_add(sqc(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flqc), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(det_sqc(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4472
{ const int mynn_lo18 = (((kts) + (1))), mynn_hi18 = (((kte) - (1))), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { k = mynn_lo18 + mynn_it18*mynn_st18;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4473
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4474
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4475
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4476
d(k) = r6_add(r6_sub(sqc(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(det_sqc(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4478
} mynn_lane_sync(); k = mynn_lo18 + mynn_n18*mynn_st18; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4480
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4482
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4484
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4485
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4488
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4491
d(k) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqc(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), flqc), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqc1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqc1(((k) + (1))))), r6_mul(det_sqc(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4496
{ const int mynn_lo19 = (((kts) + (1))), mynn_hi19 = (((kte) - (1))), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { k = mynn_lo19 + mynn_it19*mynn_st19;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4497
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4500
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4503
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4506
d(k) = r6_add(r6_sub(r6_add(r6_add(sqc(k), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqc1(k), s_awqc1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqc1(k), sd_awqc1(((k) + (1)))))), r6_mul(det_sqc(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4510
} mynn_lane_sync(); k = mynn_lo19 + mynn_n19*mynn_st19; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4511
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4514
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4515
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4516
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4517
d(kte) = sqc(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4520
f_tridiag2(kte, a, b, c, d, sqc2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4526
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4528
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqc2).size(); mynn_element += lane_width) { sqc2.linear(mynn_element) = sqc.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4529
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4530
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4532
if (((bl_mynn_mixqt) == (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4538
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4540
qvflux = flqv;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4541
if (((qvflux) < (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4543
qvflux = mynn_max((float)(qvflux), (float)(r6_div(mynn_min((float)(r6_sub(1e-8f, r6_mul(p9, sqv(kts)))), (float)(zero)), r6_mul(r6_mul(dtz(kts), rhosfc), rhoinv(kts)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4544
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4546
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4548
{ const int mynn_lo20 = (((kts) + (1))), mynn_hi20 = (((kte) - (1))), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { k = mynn_lo20 + mynn_it20*mynn_st20;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4549
upcont(k) = r6_sub(s_awqv1(k), r6_mul(s_aw1(k), r6_add(r6_mul(sqv(k), upwind), r6_mul(sqv(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4550
dncont(k) = r6_sub(sd_awqv1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqv(k), upwind), r6_mul(sqv(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4551
} mynn_lane_sync(); k = mynn_lo20 + mynn_n20*mynn_st20; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4553
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4554
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4557
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4558
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4559
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4560
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4561
d(1) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqv(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), qvflux), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1)))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4565
{ const int mynn_lo21 = (((kts) + (1))), mynn_hi21 = (((kte) - (1))), mynn_st21 = (1); const int mynn_n21 = mynn_trip(mynn_lo21, mynn_hi21, mynn_st21); mynn_lane_sync(); for (int mynn_it21 = lane_sub; mynn_it21 < mynn_n21; mynn_it21 += lane_width) { k = mynn_lo21 + mynn_it21*mynn_st21;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4566
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4567
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4568
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4569
d(k) = r6_add(r6_add(r6_sub(sqv(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k)))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4571
} mynn_lane_sync(); k = mynn_lo21 + mynn_n21*mynn_st21; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4573
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4575
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4577
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4578
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4581
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4584
d(k) = r6_add(r6_add(r6_add(r6_sub(r6_add(r6_add(sqv(k), r6_mul(r6_mul(r6_mul(dtz(k), rhosfc), qvflux), rhoinv(k))), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqv1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqv1(((k) + (1))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4589
{ const int mynn_lo22 = (((kts) + (1))), mynn_hi22 = (((kte) - (1))), mynn_st22 = (1); const int mynn_n22 = mynn_trip(mynn_lo22, mynn_hi22, mynn_st22); mynn_lane_sync(); for (int mynn_it22 = lane_sub; mynn_it22 < mynn_n22; mynn_it22 += lane_width) { k = mynn_lo22 + mynn_it22*mynn_st22;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4590
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4593
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4596
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4599
d(k) = r6_add(r6_add(r6_sub(r6_add(r6_add(sqv(k), r6_mul(qcd(k), delt)), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqv1(k), s_awqv1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqv1(k), sd_awqv1(((k) + (1)))))), r6_mul(sub_sqv(k), delt)), r6_mul(det_sqv(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4603
} mynn_lane_sync(); k = mynn_lo22 + mynn_n22*mynn_st22; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4604
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4620
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4621
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4622
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4623
d(kte) = sqv(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4626
f_tridiag2(kte, a, b, c, d, sqv2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4632
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4633
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqv2).size(); mynn_element += lane_width) { sqv2.linear(mynn_element) = sqv.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4634
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4639
if (((((bl_mynn_cloudmix) > (0))) && (flag_qi))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4641
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4643
{ const int mynn_lo23 = (((kts) + (1))), mynn_hi23 = (((kte) - (1))), mynn_st23 = (1); const int mynn_n23 = mynn_trip(mynn_lo23, mynn_hi23, mynn_st23); mynn_lane_sync(); for (int mynn_it23 = lane_sub; mynn_it23 < mynn_n23; mynn_it23 += lane_width) { k = mynn_lo23 + mynn_it23*mynn_st23;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4644
upcont(k) = r6_sub(s_awqi1(k), r6_mul(s_aw1(k), r6_add(r6_mul(sqi(k), upwind), r6_mul(sqi(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4645
dncont(k) = r6_sub(sd_awqi1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(sqi(k), upwind), r6_mul(sqi(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4646
} mynn_lane_sync(); k = mynn_lo23 + mynn_n23*mynn_st23; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4648
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4649
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4652
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4653
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4654
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4655
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4656
d(1) = r6_sub(sqi(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4659
{ const int mynn_lo24 = (((kts) + (1))), mynn_hi24 = (((kte) - (1))), mynn_st24 = (1); const int mynn_n24 = mynn_trip(mynn_lo24, mynn_hi24, mynn_st24); mynn_lane_sync(); for (int mynn_it24 = lane_sub; mynn_it24 < mynn_n24; mynn_it24 += lane_width) { k = mynn_lo24 + mynn_it24*mynn_st24;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4660
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4661
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4662
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4663
d(k) = r6_sub(sqi(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4664
} mynn_lane_sync(); k = mynn_lo24 + mynn_n24*mynn_st24; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4666
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4668
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4669
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4670
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4673
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4676
d(k) = r6_add(r6_sub(sqi(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqi1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqi1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4680
{ const int mynn_lo25 = (((kts) + (1))), mynn_hi25 = (((kte) - (1))), mynn_st25 = (1); const int mynn_n25 = mynn_trip(mynn_lo25, mynn_hi25, mynn_st25); mynn_lane_sync(); for (int mynn_it25 = lane_sub; mynn_it25 < mynn_n25; mynn_it25 += lane_width) { k = mynn_lo25 + mynn_it25*mynn_st25;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4681
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4684
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4687
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4690
d(k) = r6_sub(r6_add(sqi(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqi1(k), s_awqi1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqi1(k), sd_awqi1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4693
} mynn_lane_sync(); k = mynn_lo25 + mynn_n25*mynn_st25; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4694
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4710
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4711
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4712
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4713
d(kte) = sqi(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4716
f_tridiag2(kte, a, b, c, d, sqi2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4722
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4723
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqi2).size(); mynn_element += lane_width) { sqi2.linear(mynn_element) = sqi.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4724
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4730
if (((((bl_mynn_cloudmix) > (0))) && (false))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4732
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4734
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4735
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4736
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4737
d(k) = sqs(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4739
{ const int mynn_lo26 = (((kts) + (1))), mynn_hi26 = (((kte) - (1))), mynn_st26 = (1); const int mynn_n26 = mynn_trip(mynn_lo26, mynn_hi26, mynn_st26); mynn_lane_sync(); for (int mynn_it26 = lane_sub; mynn_it26 < mynn_n26; mynn_it26 += lane_width) { k = mynn_lo26 + mynn_it26*mynn_st26;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4740
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4741
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4742
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4743
d(k) = sqs(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4744
} mynn_lane_sync(); k = mynn_lo26 + mynn_n26*mynn_st26; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4747
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4748
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4749
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4750
d(kte) = sqs(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4753
f_tridiag2(kte, a, b, c, d, sqs2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4759
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4760
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sqs2).size(); mynn_element += lane_width) { sqs2.linear(mynn_element) = sqs.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4761
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4766
if (((((((bl_mynn_cloudmix) > (0))) && (flag_qni))) && (((bl_mynn_mixnumcon) > (0))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4769
{ const int mynn_lo27 = (kts), mynn_hi27 = (kte), mynn_st27 = (1); const int mynn_n27 = mynn_trip(mynn_lo27, mynn_hi27, mynn_st27); mynn_lane_sync(); for (int mynn_it27 = lane_sub; mynn_it27 < mynn_n27; mynn_it27 += lane_width) { k = mynn_lo27 + mynn_it27*mynn_st27;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4770
qni2(k) = mynn_max((float)(zero), (float)(qni(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4772
if (((sqi(k)) > (1e-12f))) { qni2(k) = mynn_max((float)(qni2(k)), (float)(ni_min)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4773
} mynn_lane_sync(); k = mynn_lo27 + mynn_n27*mynn_st27; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4775
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4777
{ const int mynn_lo28 = (((kts) + (1))), mynn_hi28 = (((kte) - (1))), mynn_st28 = (1); const int mynn_n28 = mynn_trip(mynn_lo28, mynn_hi28, mynn_st28); mynn_lane_sync(); for (int mynn_it28 = lane_sub; mynn_it28 < mynn_n28; mynn_it28 += lane_width) { k = mynn_lo28 + mynn_it28*mynn_st28;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4778
upcont(k) = r6_sub(s_awqni1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qni2(k), upwind), r6_mul(qni2(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4779
dncont(k) = r6_sub(sd_awqni1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qni2(k), upwind), r6_mul(qni2(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4780
} mynn_lane_sync(); k = mynn_lo28 + mynn_n28*mynn_st28; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4782
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4783
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4786
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4787
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4788
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4789
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4790
d(1) = r6_sub(qni2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4793
{ const int mynn_lo29 = (((kts) + (1))), mynn_hi29 = (((kte) - (1))), mynn_st29 = (1); const int mynn_n29 = mynn_trip(mynn_lo29, mynn_hi29, mynn_st29); mynn_lane_sync(); for (int mynn_it29 = lane_sub; mynn_it29 < mynn_n29; mynn_it29 += lane_width) { k = mynn_lo29 + mynn_it29*mynn_st29;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4794
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4795
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4796
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4797
d(k) = r6_sub(qni2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4798
} mynn_lane_sync(); k = mynn_lo29 + mynn_n29*mynn_st29; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4800
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4802
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4803
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4804
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4807
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4810
d(k) = r6_add(r6_sub(qni2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqni1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqni1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4814
{ const int mynn_lo30 = (((kts) + (1))), mynn_hi30 = (((kte) - (1))), mynn_st30 = (1); const int mynn_n30 = mynn_trip(mynn_lo30, mynn_hi30, mynn_st30); mynn_lane_sync(); for (int mynn_it30 = lane_sub; mynn_it30 < mynn_n30; mynn_it30 += lane_width) { k = mynn_lo30 + mynn_it30*mynn_st30;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4815
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4818
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4821
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4824
d(k) = r6_sub(r6_add(qni2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqni1(k), s_awqni1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqni1(k), sd_awqni1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4827
} mynn_lane_sync(); k = mynn_lo30 + mynn_n30*mynn_st30; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4828
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4831
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4832
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4833
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4834
d(kte) = qni2(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4837
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4840
{ const int mynn_lo31 = (kts), mynn_hi31 = (kte), mynn_st31 = (1); const int mynn_n31 = mynn_trip(mynn_lo31, mynn_hi31, mynn_st31); mynn_lane_sync(); for (int mynn_it31 = lane_sub; mynn_it31 < mynn_n31; mynn_it31 += lane_width) { k = mynn_lo31 + mynn_it31*mynn_st31;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4842
qni2(k) = x(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4844
if (((sqi2(k)) > (1e-12f))) { qni2(k) = mynn_max((float)(qni2(k)), (float)(ni_min)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4845
} mynn_lane_sync(); k = mynn_lo31 + mynn_n31*mynn_st31; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4847
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4848
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qni2).size(); mynn_element += lane_width) { qni2.linear(mynn_element) = qni.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4849
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4855
if (((((((bl_mynn_cloudmix) > (0))) && (flag_qnc))) && (((bl_mynn_mixnumcon) > (0))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4858
{ const int mynn_lo32 = (kts), mynn_hi32 = (kte), mynn_st32 = (1); const int mynn_n32 = mynn_trip(mynn_lo32, mynn_hi32, mynn_st32); mynn_lane_sync(); for (int mynn_it32 = lane_sub; mynn_it32 < mynn_n32; mynn_it32 += lane_width) { k = mynn_lo32 + mynn_it32*mynn_st32;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4859
qnc2(k) = mynn_max((float)(zero), (float)(qnc(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4861
if (((sqc(k)) > (1e-12f))) { qnc2(k) = mynn_max((float)(qnc2(k)), (float)(nc_min)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4862
} mynn_lane_sync(); k = mynn_lo32 + mynn_n32*mynn_st32; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4864
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4866
{ const int mynn_lo33 = (((kts) + (1))), mynn_hi33 = (((kte) - (1))), mynn_st33 = (1); const int mynn_n33 = mynn_trip(mynn_lo33, mynn_hi33, mynn_st33); mynn_lane_sync(); for (int mynn_it33 = lane_sub; mynn_it33 < mynn_n33; mynn_it33 += lane_width) { k = mynn_lo33 + mynn_it33*mynn_st33;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4867
upcont(k) = r6_sub(s_awqnc1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnc2(k), upwind), r6_mul(qnc2(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4868
dncont(k) = r6_sub(sd_awqnc1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qnc2(k), upwind), r6_mul(qnc2(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4869
} mynn_lane_sync(); k = mynn_lo33 + mynn_n33*mynn_st33; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4871
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4872
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4875
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4876
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4877
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4878
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4879
d(1) = r6_sub(qnc2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4882
{ const int mynn_lo34 = (((kts) + (1))), mynn_hi34 = (((kte) - (1))), mynn_st34 = (1); const int mynn_n34 = mynn_trip(mynn_lo34, mynn_hi34, mynn_st34); mynn_lane_sync(); for (int mynn_it34 = lane_sub; mynn_it34 < mynn_n34; mynn_it34 += lane_width) { k = mynn_lo34 + mynn_it34*mynn_st34;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4883
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4884
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4885
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4886
d(k) = r6_sub(qnc2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4887
} mynn_lane_sync(); k = mynn_lo34 + mynn_n34*mynn_st34; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4889
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4891
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4892
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4893
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4896
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4899
d(k) = r6_add(r6_sub(qnc2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnc1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqnc1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4903
{ const int mynn_lo35 = (((kts) + (1))), mynn_hi35 = (((kte) - (1))), mynn_st35 = (1); const int mynn_n35 = mynn_trip(mynn_lo35, mynn_hi35, mynn_st35); mynn_lane_sync(); for (int mynn_it35 = lane_sub; mynn_it35 < mynn_n35; mynn_it35 += lane_width) { k = mynn_lo35 + mynn_it35*mynn_st35;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4904
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4907
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4910
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4913
d(k) = r6_sub(r6_add(qnc2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnc1(k), s_awqnc1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqnc1(k), sd_awqnc1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4916
} mynn_lane_sync(); k = mynn_lo35 + mynn_n35*mynn_st35; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4917
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4920
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4921
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4922
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4923
d(kte) = qnc2(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4926
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4929
{ const int mynn_lo36 = (kts), mynn_hi36 = (kte), mynn_st36 = (1); const int mynn_n36 = mynn_trip(mynn_lo36, mynn_hi36, mynn_st36); mynn_lane_sync(); for (int mynn_it36 = lane_sub; mynn_it36 < mynn_n36; mynn_it36 += lane_width) { k = mynn_lo36 + mynn_it36*mynn_st36;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4931
qnc2(k) = x(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4933
if (((sqc2(k)) > (1e-12f))) { qnc2(k) = mynn_max((float)(qnc2(k)), (float)(nc_min)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4934
} mynn_lane_sync(); k = mynn_lo36 + mynn_n36*mynn_st36; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4936
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4937
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnc2).size(); mynn_element += lane_width) { qnc2.linear(mynn_element) = qnc.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4938
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4943
if (((flag_qnwfa) && (((bl_mynn_mixaerosols) > (0))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4945
{ const int mynn_lo37 = (kts), mynn_hi37 = (kte), mynn_st37 = (1); const int mynn_n37 = mynn_trip(mynn_lo37, mynn_hi37, mynn_st37); mynn_lane_sync(); for (int mynn_it37 = lane_sub; mynn_it37 < mynn_n37; mynn_it37 += lane_width) { k = mynn_lo37 + mynn_it37*mynn_st37;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4946
qnwfa2(k) = mynn_max((float)(zero), (float)(qnwfa(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4947
} mynn_lane_sync(); k = mynn_lo37 + mynn_n37*mynn_st37; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4949
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4951
{ const int mynn_lo38 = (((kts) + (1))), mynn_hi38 = (((kte) - (1))), mynn_st38 = (1); const int mynn_n38 = mynn_trip(mynn_lo38, mynn_hi38, mynn_st38); mynn_lane_sync(); for (int mynn_it38 = lane_sub; mynn_it38 < mynn_n38; mynn_it38 += lane_width) { k = mynn_lo38 + mynn_it38*mynn_st38;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4952
upcont(k) = r6_sub(s_awqnwfa1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnwfa2(k), upwind), r6_mul(qnwfa2(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4953
dncont(k) = r6_sub(sd_awqnwfa1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qnwfa2(k), upwind), r6_mul(qnwfa2(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4954
} mynn_lane_sync(); k = mynn_lo38 + mynn_n38*mynn_st38; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4956
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4957
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4960
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4961
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4962
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4963
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4964
d(1) = r6_sub(qnwfa2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4967
{ const int mynn_lo39 = (((kts) + (1))), mynn_hi39 = (((kte) - (1))), mynn_st39 = (1); const int mynn_n39 = mynn_trip(mynn_lo39, mynn_hi39, mynn_st39); mynn_lane_sync(); for (int mynn_it39 = lane_sub; mynn_it39 < mynn_n39; mynn_it39 += lane_width) { k = mynn_lo39 + mynn_it39*mynn_st39;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4968
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4969
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4970
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4971
d(k) = r6_sub(qnwfa2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4972
} mynn_lane_sync(); k = mynn_lo39 + mynn_n39*mynn_st39; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4974
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4976
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4977
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4978
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4981
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4984
d(k) = r6_add(r6_sub(qnwfa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnwfa1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqnwfa1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4988
{ const int mynn_lo40 = (((kts) + (1))), mynn_hi40 = (((kte) - (1))), mynn_st40 = (1); const int mynn_n40 = mynn_trip(mynn_lo40, mynn_hi40, mynn_st40); mynn_lane_sync(); for (int mynn_it40 = lane_sub; mynn_it40 < mynn_n40; mynn_it40 += lane_width) { k = mynn_lo40 + mynn_it40*mynn_st40;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4989
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4992
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4995
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:4998
d(k) = r6_sub(r6_add(qnwfa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnwfa1(k), s_awqnwfa1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqnwfa1(k), sd_awqnwfa1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5001
} mynn_lane_sync(); k = mynn_lo40 + mynn_n40*mynn_st40; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5002
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5005
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5006
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5007
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5008
d(kte) = qnwfa2(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5011
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5014
{ const int mynn_lo41 = (kts), mynn_hi41 = (kte), mynn_st41 = (1); const int mynn_n41 = mynn_trip(mynn_lo41, mynn_hi41, mynn_st41); mynn_lane_sync(); for (int mynn_it41 = lane_sub; mynn_it41 < mynn_n41; mynn_it41 += lane_width) { k = mynn_lo41 + mynn_it41*mynn_st41;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5016
qnwfa2(k) = mynn_max((float)(zero), (float)(x(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5017
} mynn_lane_sync(); k = mynn_lo41 + mynn_n41*mynn_st41; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5020
{ const int mynn_lo42 = (kts), mynn_hi42 = (kte), mynn_st42 = (1); const int mynn_n42 = mynn_trip(mynn_lo42, mynn_hi42, mynn_st42); mynn_lane_sync(); for (int mynn_it42 = lane_sub; mynn_it42 < mynn_n42; mynn_it42 += lane_width) { k = mynn_lo42 + mynn_it42*mynn_st42;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5022
if (((r6_sub(xland, 1.5f)) >= (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5023
wt = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zw(k), 100.f)), (float)(zero)), mynn_max((float)(pblh), (float)(200.f)))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5024
wfa_max2 = r6_add(r6_mul(r6_sub(one, wt), 6.e8f), r6_mul(wt, wfa_max));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5025
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5026
wfa_max2 = wfa_max;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5027
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5028
aero_min = r6_mul(wfa_min, mynn_exp(r6_neg(r6_div(zw(k), wfa_ht))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5029
aero_max = r6_mul(wfa_max2, mynn_exp(r6_neg(r6_div(zw(k), wfa_ht))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5030
qnwfa2(k) = mynn_min((float)(mynn_max((float)(aero_min), (float)(qnwfa2(k)))), (float)(aero_max));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5031
} mynn_lane_sync(); k = mynn_lo42 + mynn_n42*mynn_st42; if (mynn_n42 > 0) { const int mynn_src42 = (mynn_n42 - 1) % lane_width; aero_max = mynn_lane_bcast(aero_max, mynn_src42); aero_min = mynn_lane_bcast(aero_min, mynn_src42); wfa_max2 = mynn_lane_bcast(wfa_max2, mynn_src42); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5032
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5034
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnwfa2).size(); mynn_element += lane_width) { qnwfa2.linear(mynn_element) = qnwfa.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5035
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5040
if (((flag_qnifa) && (((bl_mynn_mixaerosols) > (0))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5042
{ const int mynn_lo43 = (kts), mynn_hi43 = (kte), mynn_st43 = (1); const int mynn_n43 = mynn_trip(mynn_lo43, mynn_hi43, mynn_st43); mynn_lane_sync(); for (int mynn_it43 = lane_sub; mynn_it43 < mynn_n43; mynn_it43 += lane_width) { k = mynn_lo43 + mynn_it43*mynn_st43;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5043
qnifa2(k) = mynn_max((float)(zero), (float)(qnifa(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5044
} mynn_lane_sync(); k = mynn_lo43 + mynn_n43*mynn_st43; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5046
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5048
{ const int mynn_lo44 = (((kts) + (1))), mynn_hi44 = (((kte) - (1))), mynn_st44 = (1); const int mynn_n44 = mynn_trip(mynn_lo44, mynn_hi44, mynn_st44); mynn_lane_sync(); for (int mynn_it44 = lane_sub; mynn_it44 < mynn_n44; mynn_it44 += lane_width) { k = mynn_lo44 + mynn_it44*mynn_st44;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5049
upcont(k) = r6_sub(s_awqnifa1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnifa2(k), upwind), r6_mul(qnifa2(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5050
dncont(k) = r6_sub(sd_awqnifa1(k), r6_mul(sd_aw1(k), r6_add(r6_mul(qnifa2(k), upwind), r6_mul(qnifa2(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5051
} mynn_lane_sync(); k = mynn_lo44 + mynn_n44*mynn_st44; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5053
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5054
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5056
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5057
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5058
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5059
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5060
d(1) = r6_sub(qnifa2(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5063
{ const int mynn_lo45 = (((kts) + (1))), mynn_hi45 = (((kte) - (1))), mynn_st45 = (1); const int mynn_n45 = mynn_trip(mynn_lo45, mynn_hi45, mynn_st45); mynn_lane_sync(); for (int mynn_it45 = lane_sub; mynn_it45 < mynn_n45; mynn_it45 += lane_width) { k = mynn_lo45 + mynn_it45*mynn_st45;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5064
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5065
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5066
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5067
d(k) = r6_sub(qnifa2(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5068
} mynn_lane_sync(); k = mynn_lo45 + mynn_n45*mynn_st45; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5070
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5072
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5073
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5074
b(k) = r6_sub(r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5077
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5080
d(k) = r6_add(r6_sub(qnifa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnifa1(((k) + (1))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), sd_awqnifa1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5084
{ const int mynn_lo46 = (((kts) + (1))), mynn_hi46 = (((kte) - (1))), mynn_st46 = (1); const int mynn_n46 = mynn_trip(mynn_lo46, mynn_hi46, mynn_st46); mynn_lane_sync(); for (int mynn_it46 = lane_sub; mynn_it46 < mynn_n46; mynn_it46 += lane_width) { k = mynn_lo46 + mynn_it46*mynn_st46;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5085
a(k) = r6_add(r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5088
b(k) = r6_add(r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1)))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(sd_aw1(k), sd_aw1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5091
c(k) = r6_sub(r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1))))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), sd_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5094
d(k) = r6_sub(r6_add(qnifa2(k), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnifa1(k), s_awqnifa1(((k) + (1)))))), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(sd_awqnifa1(k), sd_awqnifa1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5097
} mynn_lane_sync(); k = mynn_lo46 + mynn_n46*mynn_st46; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5098
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5101
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5102
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5103
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5104
d(kte) = qnifa2(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5107
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5110
{ const int mynn_lo47 = (kts), mynn_hi47 = (kte), mynn_st47 = (1); const int mynn_n47 = mynn_trip(mynn_lo47, mynn_hi47, mynn_st47); mynn_lane_sync(); for (int mynn_it47 = lane_sub; mynn_it47 < mynn_n47; mynn_it47 += lane_width) { k = mynn_lo47 + mynn_it47*mynn_st47;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5112
qnifa2(k) = mynn_max((float)(zero), (float)(x(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5113
} mynn_lane_sync(); k = mynn_lo47 + mynn_n47*mynn_st47; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5116
{ const int mynn_lo48 = (kts), mynn_hi48 = (kte), mynn_st48 = (1); const int mynn_n48 = mynn_trip(mynn_lo48, mynn_hi48, mynn_st48); mynn_lane_sync(); for (int mynn_it48 = lane_sub; mynn_it48 < mynn_n48; mynn_it48 += lane_width) { k = mynn_lo48 + mynn_it48*mynn_st48;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5117
aero_min = r6_mul(ifa_min, mynn_exp(r6_neg(r6_div(zw(k), ifa_ht))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5118
aero_max = r6_mul(ifa_max, mynn_exp(r6_neg(r6_div(zw(k), ifa_ht))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5119
qnifa2(k) = mynn_min((float)(mynn_max((float)(aero_min), (float)(qnifa2(k)))), (float)(aero_max));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5120
} mynn_lane_sync(); k = mynn_lo48 + mynn_n48*mynn_st48; if (mynn_n48 > 0) { const int mynn_src48 = (mynn_n48 - 1) % lane_width; aero_max = mynn_lane_bcast(aero_max, mynn_src48); aero_min = mynn_lane_bcast(aero_min, mynn_src48); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5122
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5124
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnifa2).size(); mynn_element += lane_width) { qnifa2.linear(mynn_element) = qnifa.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5125
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5130
if (((flag_qnbca) && (((bl_mynn_mixaerosols) > (0))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5132
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5134
{ const int mynn_lo49 = (((kts) + (1))), mynn_hi49 = (((kte) - (1))), mynn_st49 = (1); const int mynn_n49 = mynn_trip(mynn_lo49, mynn_hi49, mynn_st49); mynn_lane_sync(); for (int mynn_it49 = lane_sub; mynn_it49 < mynn_n49; mynn_it49 += lane_width) { k = mynn_lo49 + mynn_it49*mynn_st49;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5135
upcont(k) = r6_sub(s_awqnbca1(k), r6_mul(s_aw1(k), r6_add(r6_mul(qnbca(k), upwind), r6_mul(qnbca(((k) - (1))), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5136
dncont(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5137
} mynn_lane_sync(); k = mynn_lo49 + mynn_n49*mynn_st49; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5139
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5140
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5142
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5143
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5144
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5145
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5146
d(1) = r6_sub(qnbca(k), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5149
{ const int mynn_lo50 = (((kts) + (1))), mynn_hi50 = (((kte) - (1))), mynn_st50 = (1); const int mynn_n50 = mynn_trip(mynn_lo50, mynn_hi50, mynn_st50); mynn_lane_sync(); for (int mynn_it50 = lane_sub; mynn_it50 < mynn_n50; mynn_it50 += lane_width) { k = mynn_lo50 + mynn_it50*mynn_st50;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5150
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5151
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5152
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5153
d(k) = r6_sub(qnbca(k), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5154
} mynn_lane_sync(); k = mynn_lo50 + mynn_n50*mynn_st50; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5156
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5158
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5159
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5160
b(k) = r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), nonloc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5162
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), nonloc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5164
d(k) = r6_sub(qnbca(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awqnbca1(((k) + (1)))), nonloc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5166
{ const int mynn_lo51 = (((kts) + (1))), mynn_hi51 = (((kte) - (1))), mynn_st51 = (1); const int mynn_n51 = mynn_trip(mynn_lo51, mynn_hi51, mynn_st51); mynn_lane_sync(); for (int mynn_it51 = lane_sub; mynn_it51 < mynn_n51; mynn_it51 += lane_width) { k = mynn_lo51 + mynn_it51*mynn_st51;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5167
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)), nonloc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5169
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))), nonloc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5171
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))), nonloc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5173
d(k) = r6_add(qnbca(k), r6_mul(r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awqnbca1(k), s_awqnbca1(((k) + (1))))), nonloc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5174
} mynn_lane_sync(); k = mynn_lo51 + mynn_n51*mynn_st51; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5175
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5178
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5179
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5180
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5181
d(kte) = qnbca(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5184
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5187
{ const int mynn_lo52 = (kts), mynn_hi52 = (kte), mynn_st52 = (1); const int mynn_n52 = mynn_trip(mynn_lo52, mynn_hi52, mynn_st52); mynn_lane_sync(); for (int mynn_it52 = lane_sub; mynn_it52 < mynn_n52; mynn_it52 += lane_width) { k = mynn_lo52 + mynn_it52*mynn_st52;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5189
qnbca2(k) = mynn_max((float)(zero), (float)(x(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5190
} mynn_lane_sync(); k = mynn_lo52 + mynn_n52*mynn_st52; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5192
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5194
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qnbca2).size(); mynn_element += lane_width) { qnbca2.linear(mynn_element) = qnbca.linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5195
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5200
if (flag_ozone) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5201
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5204
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5205
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5206
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5207
d(k) = ozone(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5209
{ const int mynn_lo53 = (((kts) + (1))), mynn_hi53 = (((kte) - (1))), mynn_st53 = (1); const int mynn_n53 = mynn_trip(mynn_lo53, mynn_hi53, mynn_st53); mynn_lane_sync(); for (int mynn_it53 = lane_sub; mynn_it53 < mynn_n53; mynn_it53 += lane_width) { k = mynn_lo53 + mynn_it53*mynn_st53;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5210
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5211
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5212
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5213
d(k) = ozone(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5214
} mynn_lane_sync(); k = mynn_lo53 + mynn_n53*mynn_st53; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5217
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5218
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5219
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5220
d(kte) = ozone(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5223
f_tridiag2(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5226
{ const int mynn_lo54 = (kts), mynn_hi54 = (kte), mynn_st54 = (1); const int mynn_n54 = mynn_trip(mynn_lo54, mynn_hi54, mynn_st54); mynn_lane_sync(); for (int mynn_it54 = lane_sub; mynn_it54 < mynn_n54; mynn_it54 += lane_width) { k = mynn_lo54 + mynn_it54*mynn_st54;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5228
dozone(k) = r6_div(r6_sub(mynn_max((float)(zero), (float)(x(k))), ozone(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5229
} mynn_lane_sync(); k = mynn_lo54 + mynn_n54*mynn_st54; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5230
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5231
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dozone.slice(MynnRange{dozone.lo[0], dozone.hi[0]})).size(); mynn_element += lane_width) { dozone.slice(MynnRange{dozone.lo[0], dozone.hi[0]}).linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5232
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5239
if (((bl_mynn_mixqt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5240
{ const int mynn_lo55 = (kts), mynn_hi55 = (kte), mynn_st55 = (1); const int mynn_n55 = mynn_trip(mynn_lo55, mynn_hi55, mynn_st55); mynn_lane_sync(); for (int mynn_it55 = lane_sub; mynn_it55 < mynn_n55; mynn_it55 += lane_width) { k = mynn_lo55 + mynn_it55*mynn_st55;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5242
th_new = r6_add(r6_add(thl(k), r6_mul(r6_div(xlvcp, exner(k)), sqc(k))), r6_mul(r6_div(xlscp, exner(k)), sqi(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5245
t = r6_mul(th_new, exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5246
qsat = f_qsat_blend(t, p(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5253
if (((((sqc(k)) > (zero))) || (((sqi(k)) > (zero))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5254
sqv2(k) = mynn_min((float)(sqw2(k)), (float)(qsat));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5255
portion_qc = r6_div(sqc(k), r6_add(sqc(k), sqi(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5256
portion_qi = r6_div(sqi(k), r6_add(sqc(k), sqi(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5257
condensate = mynn_max((float)(zero), (float)(r6_sub(sqw2(k), qsat)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5258
sqc2(k) = r6_mul(condensate, portion_qc);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5259
sqi2(k) = r6_mul(condensate, portion_qi);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5260
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5261
sqv2(k) = sqw2(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5262
sqi2(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5263
sqc2(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5264
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5265
} mynn_lane_sync(); k = mynn_lo55 + mynn_n55*mynn_st55; if (mynn_n55 > 0) { const int mynn_src55 = (mynn_n55 - 1) % lane_width; qsat = mynn_lane_bcast(qsat, mynn_src55); t = mynn_lane_bcast(t, mynn_src55); th_new = mynn_lane_bcast(th_new, mynn_src55); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5266
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5272
{ const int mynn_lo56 = (kts), mynn_hi56 = (kte), mynn_st56 = (1); const int mynn_n56 = mynn_trip(mynn_lo56, mynn_hi56, mynn_st56); mynn_lane_sync(); for (int mynn_it56 = lane_sub; mynn_it56 < mynn_n56; mynn_it56 += lane_width) { k = mynn_lo56 + mynn_it56*mynn_st56;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5273
dqv(k) = r6_div(r6_sub(sqv2(k), sqv(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5275
} mynn_lane_sync(); k = mynn_lo56 + mynn_n56*mynn_st56; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5277
if (((bl_mynn_cloudmix) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5282
if (flag_qc) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5283
{ const int mynn_lo57 = (kts), mynn_hi57 = (kte), mynn_st57 = (1); const int mynn_n57 = mynn_trip(mynn_lo57, mynn_hi57, mynn_st57); mynn_lane_sync(); for (int mynn_it57 = lane_sub; mynn_it57 < mynn_n57; mynn_it57 += lane_width) { k = mynn_lo57 + mynn_it57*mynn_st57;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5284
dqc(k) = r6_div(r6_sub(sqc2(k), sqc(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5286
} mynn_lane_sync(); k = mynn_lo57 + mynn_n57*mynn_st57; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5287
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5288
{ const int mynn_lo58 = (kts), mynn_hi58 = (kte), mynn_st58 = (1); const int mynn_n58 = mynn_trip(mynn_lo58, mynn_hi58, mynn_st58); mynn_lane_sync(); for (int mynn_it58 = lane_sub; mynn_it58 < mynn_n58; mynn_it58 += lane_width) { k = mynn_lo58 + mynn_it58*mynn_st58;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5289
dqc(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5290
} mynn_lane_sync(); k = mynn_lo58 + mynn_n58*mynn_st58; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5291
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5296
if (((flag_qnc) && (((bl_mynn_mixnumcon) > (0))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5297
{ const int mynn_lo59 = (kts), mynn_hi59 = (kte), mynn_st59 = (1); const int mynn_n59 = mynn_trip(mynn_lo59, mynn_hi59, mynn_st59); mynn_lane_sync(); for (int mynn_it59 = lane_sub; mynn_it59 < mynn_n59; mynn_it59 += lane_width) { k = mynn_lo59 + mynn_it59*mynn_st59;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5298
dqnc(k) = r6_div(r6_sub(qnc2(k), qnc(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5300
} mynn_lane_sync(); k = mynn_lo59 + mynn_n59*mynn_st59; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5301
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5302
{ const int mynn_lo60 = (kts), mynn_hi60 = (kte), mynn_st60 = (1); const int mynn_n60 = mynn_trip(mynn_lo60, mynn_hi60, mynn_st60); mynn_lane_sync(); for (int mynn_it60 = lane_sub; mynn_it60 < mynn_n60; mynn_it60 += lane_width) { k = mynn_lo60 + mynn_it60*mynn_st60;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5303
dqnc(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5304
} mynn_lane_sync(); k = mynn_lo60 + mynn_n60*mynn_st60; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5305
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5310
if (flag_qi) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5311
{ const int mynn_lo61 = (kts), mynn_hi61 = (kte), mynn_st61 = (1); const int mynn_n61 = mynn_trip(mynn_lo61, mynn_hi61, mynn_st61); mynn_lane_sync(); for (int mynn_it61 = lane_sub; mynn_it61 < mynn_n61; mynn_it61 += lane_width) { k = mynn_lo61 + mynn_it61*mynn_st61;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5312
dqi(k) = r6_div(r6_sub(sqi2(k), sqi(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5314
} mynn_lane_sync(); k = mynn_lo61 + mynn_n61*mynn_st61; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5315
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5316
{ const int mynn_lo62 = (kts), mynn_hi62 = (kte), mynn_st62 = (1); const int mynn_n62 = mynn_trip(mynn_lo62, mynn_hi62, mynn_st62); mynn_lane_sync(); for (int mynn_it62 = lane_sub; mynn_it62 < mynn_n62; mynn_it62 += lane_width) { k = mynn_lo62 + mynn_it62*mynn_st62;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5317
dqi(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5318
} mynn_lane_sync(); k = mynn_lo62 + mynn_n62*mynn_st62; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5319
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5324
if (false) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5325
{ const int mynn_lo63 = (kts), mynn_hi63 = (kte), mynn_st63 = (1); const int mynn_n63 = mynn_trip(mynn_lo63, mynn_hi63, mynn_st63); mynn_lane_sync(); for (int mynn_it63 = lane_sub; mynn_it63 < mynn_n63; mynn_it63 += lane_width) { k = mynn_lo63 + mynn_it63*mynn_st63;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5326
dqs(k) = r6_div(r6_sub(sqs2(k), sqs(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5327
} mynn_lane_sync(); k = mynn_lo63 + mynn_n63*mynn_st63; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5328
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5329
{ const int mynn_lo64 = (kts), mynn_hi64 = (kte), mynn_st64 = (1); const int mynn_n64 = mynn_trip(mynn_lo64, mynn_hi64, mynn_st64); mynn_lane_sync(); for (int mynn_it64 = lane_sub; mynn_it64 < mynn_n64; mynn_it64 += lane_width) { k = mynn_lo64 + mynn_it64*mynn_st64;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5330
dqs(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5331
} mynn_lane_sync(); k = mynn_lo64 + mynn_n64*mynn_st64; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5332
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5337
if (((flag_qni) && (((bl_mynn_mixnumcon) > (0))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5338
{ const int mynn_lo65 = (kts), mynn_hi65 = (kte), mynn_st65 = (1); const int mynn_n65 = mynn_trip(mynn_lo65, mynn_hi65, mynn_st65); mynn_lane_sync(); for (int mynn_it65 = lane_sub; mynn_it65 < mynn_n65; mynn_it65 += lane_width) { k = mynn_lo65 + mynn_it65*mynn_st65;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5339
dqni(k) = r6_div(r6_sub(qni2(k), qni(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5341
} mynn_lane_sync(); k = mynn_lo65 + mynn_n65*mynn_st65; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5342
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5343
{ const int mynn_lo66 = (kts), mynn_hi66 = (kte), mynn_st66 = (1); const int mynn_n66 = mynn_trip(mynn_lo66, mynn_hi66, mynn_st66); mynn_lane_sync(); for (int mynn_it66 = lane_sub; mynn_it66 < mynn_n66; mynn_it66 += lane_width) { k = mynn_lo66 + mynn_it66*mynn_st66;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5344
dqni(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5345
} mynn_lane_sync(); k = mynn_lo66 + mynn_n66*mynn_st66; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5346
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5347
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5349
{ const int mynn_lo67 = (kts), mynn_hi67 = (kte), mynn_st67 = (1); const int mynn_n67 = mynn_trip(mynn_lo67, mynn_hi67, mynn_st67); mynn_lane_sync(); for (int mynn_it67 = lane_sub; mynn_it67 < mynn_n67; mynn_it67 += lane_width) { k = mynn_lo67 + mynn_it67*mynn_st67;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5350
dqc(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5351
dqnc(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5352
dqi(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5353
dqni(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5354
dqs(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5355
} mynn_lane_sync(); k = mynn_lo67 + mynn_n67*mynn_st67; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5356
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5361
{ const int mynn_lo68 = (kts), mynn_hi68 = (kte), mynn_st68 = (1); const int mynn_n68 = mynn_trip(mynn_lo68, mynn_hi68, mynn_st68); mynn_lane_sync(); for (int mynn_it68 = lane_sub; mynn_it68 < mynn_n68; mynn_it68 += lane_width) { k = mynn_lo68 + mynn_it68*mynn_st68;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5362
if (((r6_add(r6_mul(dozone(k), delt), ozone(k))) < (0.f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5363
dozone(k) = r6_div(r6_neg(r6_mul(ozone(k), 0.99f)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5364
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5365
} mynn_lane_sync(); k = mynn_lo68 + mynn_n68*mynn_st68; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5372
if (flag_qi) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5373
{ const int mynn_lo69 = (kts), mynn_hi69 = (kte), mynn_st69 = (1); const int mynn_n69 = mynn_trip(mynn_lo69, mynn_hi69, mynn_st69); mynn_lane_sync(); for (int mynn_it69 = lane_sub; mynn_it69 < mynn_n69; mynn_it69 += lane_width) { k = mynn_lo69 + mynn_it69*mynn_st69;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5374
dth(k) = r6_div(r6_sub(r6_add(r6_add(thl(k), r6_mul(r6_div(xlvcp, exner(k)), sqc2(k))), r6_mul(r6_div(xlscp, exner(k)), sqi2(k))), th(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5385
} mynn_lane_sync(); k = mynn_lo69 + mynn_n69*mynn_st69; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5386
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5387
{ const int mynn_lo70 = (kts), mynn_hi70 = (kte), mynn_st70 = (1); const int mynn_n70 = mynn_trip(mynn_lo70, mynn_hi70, mynn_st70); mynn_lane_sync(); for (int mynn_it70 = lane_sub; mynn_it70 < mynn_n70; mynn_it70 += lane_width) { k = mynn_lo70 + mynn_it70*mynn_st70;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5388
dth(k) = r6_div(r6_sub(r6_add(thl(k), r6_mul(r6_div(xlvcp, exner(k)), sqc2(k))), th(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5393
} mynn_lane_sync(); k = mynn_lo70 + mynn_n70*mynn_st70; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5394
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5397
f_moisture_check(kte, delt, delp, exner, sqv2, sqc2, sqi2, sqs2, thl, dqv, dqc, dqi, dqs, dth);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5404
if (((((flag_qnwfa) && (flag_qnifa))) && (((bl_mynn_mixaerosols) > (0))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5406
{ const int mynn_lo71 = (kts), mynn_hi71 = (kte), mynn_st71 = (1); const int mynn_n71 = mynn_trip(mynn_lo71, mynn_hi71, mynn_st71); mynn_lane_sync(); for (int mynn_it71 = lane_sub; mynn_it71 < mynn_n71; mynn_it71 += lane_width) { k = mynn_lo71 + mynn_it71*mynn_st71;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5410
dqnwfa(k) = r6_div(r6_sub(qnwfa2(k), qnwfa(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5414
dqnifa(k) = r6_div(r6_sub(qnifa2(k), qnifa(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5415
} mynn_lane_sync(); k = mynn_lo71 + mynn_n71*mynn_st71; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5416
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5417
{ const int mynn_lo72 = (kts), mynn_hi72 = (kte), mynn_st72 = (1); const int mynn_n72 = mynn_trip(mynn_lo72, mynn_hi72, mynn_st72); mynn_lane_sync(); for (int mynn_it72 = lane_sub; mynn_it72 < mynn_n72; mynn_it72 += lane_width) { k = mynn_lo72 + mynn_it72*mynn_st72;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5418
dqnwfa(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5419
dqnifa(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5420
} mynn_lane_sync(); k = mynn_lo72 + mynn_n72*mynn_st72; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5421
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5426
if (((flag_qnbca) && (((bl_mynn_mixaerosols) > (0))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5427
{ const int mynn_lo73 = (kts), mynn_hi73 = (kte), mynn_st73 = (1); const int mynn_n73 = mynn_trip(mynn_lo73, mynn_hi73, mynn_st73); mynn_lane_sync(); for (int mynn_it73 = lane_sub; mynn_it73 < mynn_n73; mynn_it73 += lane_width) { k = mynn_lo73 + mynn_it73*mynn_st73;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5428
dqnbca(k) = r6_div(r6_sub(qnbca2(k), qnbca(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5429
} mynn_lane_sync(); k = mynn_lo73 + mynn_n73*mynn_st73; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5430
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5431
{ const int mynn_lo74 = (kts), mynn_hi74 = (kte), mynn_st74 = (1); const int mynn_n74 = mynn_trip(mynn_lo74, mynn_hi74, mynn_st74); mynn_lane_sync(); for (int mynn_it74 = lane_sub; mynn_it74 < mynn_n74; mynn_it74 += lane_width) { k = mynn_lo74 + mynn_it74*mynn_st74;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5432
dqnbca(k) = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5433
} mynn_lane_sync(); k = mynn_lo74 + mynn_n74*mynn_st74; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5434
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5443
if (debug_code) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5444
problem = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5445
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5446
wsp = mynn_sqrt(r6_add(mynn_ipow<2>(u(k)), mynn_ipow<2>(v(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5447
wsp2 = mynn_sqrt(r6_add(mynn_ipow<2>(r6_add(u(k), r6_mul(du(k), delt))), mynn_ipow<2>(r6_add(v(k), r6_mul(du(k), delt)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5448
th2 = r6_add(th(k), r6_mul(dth(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5449
tk2 = r6_mul(th2, exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5450
if (((((((wsp2) > (200.f))) || (((tk2) > (360.f))))) || (((tk2) < (160.f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5451
problem = true;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5452
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5453
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5454
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5455
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5456
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5457
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5458
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5459
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5460
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5461
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5462
kproblem = k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5463
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5464
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5465
if (problem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5466
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5467
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5468
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5469
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5470
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5471
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5472
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5473
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5483-5572
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5513
for (k=kte; ((-(1))) > 0 ? k <= (1) : k >= (1); k += ((-(1)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5514
dqc2 = mynn_max((float)(r6_sub(qcmin, qc(k))), (float)(zero));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5515
dqi2 = mynn_max((float)(r6_sub(qimin, qi(k))), (float)(zero));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5516
dqs2 = mynn_max((float)(r6_sub(qimin, qs(k))), (float)(zero));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5519
dqc(k) = r6_add(dqc(k), r6_div(dqc2, delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5520
dqi(k) = r6_add(dqi(k), r6_div(dqi2, delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5521
dqs(k) = r6_add(dqs(k), r6_div(dqs2, delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5522
dqv(k) = r6_sub(dqv(k), r6_div(r6_add(r6_add(dqc2, dqi2), dqs2), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5523
dth(k) = r6_add(r6_add(dth(k), r6_mul(r6_div(xlvcp, exner(k)), r6_div(dqc2, delt))), r6_mul(r6_div(xlscp, exner(k)), r6_div(r6_add(dqi2, dqs2), delt)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5526
qc(k) = r6_add(qc(k), dqc2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5527
qi(k) = r6_add(qi(k), dqi2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5528
qs(k) = r6_add(qs(k), dqs2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5529
qv(k) = r6_sub(r6_sub(r6_sub(qv(k), dqc2), dqi2), dqs2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5530
th(k) = r6_add(r6_add(th(k), r6_mul(r6_div(xlvcp, exner(k)), dqc2)), r6_mul(r6_div(xlscp, exner(k)), r6_add(dqi2, dqs2)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5534
dqv2 = mynn_max((float)(r6_sub(qvmin, qv(k))), (float)(zero));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5535
dqv(k) = r6_add(dqv(k), r6_div(dqv2, delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5536
qv(k) = r6_add(qv(k), dqv2);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5537
if (((k) != (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5539
qv(((k) - (1))) = r6_sub(qv(((k) - (1))), r6_div(r6_mul(dqv2, dp(k)), dp(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5540
dqv(((k) - (1))) = r6_sub(dqv(((k) - (1))), r6_div(r6_div(r6_mul(dqv2, dp(k)), dp(((k) - (1)))), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5541
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5542
qv(k) = mynn_max((float)(qv(k)), (float)(qvmin));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5543
qc(k) = mynn_max((float)(qcmin), (float)(qc(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5544
qi(k) = mynn_max((float)(qimin), (float)(qi(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5545
qs(k) = mynn_max((float)(qimin), (float)(qs(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5546
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5550
if (((dqv2) > (1.e-20f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5551
sum = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5552
for (k=1; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5553
if (((qv(k)) > (r6_mul(two, qvmin)))) { sum = r6_add(sum, r6_mul(qv(k), dp(k))); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5554
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5555
aa = r6_div(r6_mul(dqv2, dp(1)), mynn_max((float)(sum), (float)(1.e-20f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5556
if (((aa) < (p5))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5557
{ const int mynn_lo0 = (1), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5558
if (((qv(k)) > (r6_mul(two, qvmin)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5559
dum = r6_mul(aa, qv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5560
qv(k) = r6_sub(qv(k), dum);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5561
dqv(k) = r6_sub(dqv(k), r6_div(dum, delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5562
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5563
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5564
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5567
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5568
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5570
return;
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5576-5735
__device__ __noinline__ void f_mynn_mix_chem(int kts, int kte, int i, float delt, MynnArray<float, 1> dz, float pblh, int nchem, int ndvel, MynnArray<float, 2> chem1, MynnArray<float, 1> vd1, MynnArray<float, 2> settle1, MynnArray<float, 1> rho, float flt, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> dfh, MynnArray<float, 1> s_aw1, MynnArray<float, 2> s_awchem1, float emis_ant_no, float frp, bool enh_mix, int bl_mynn_edmf, float upwind) {
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5622
dztop = r6_mul(.5f, r6_add(dz(kte), dz(((kte) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5624
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5625
dtz(k) = r6_div(delt, dz(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5626
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5630
rhoz(kts) = rho(kts);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5631
rhoinv(kts) = r6_div(one, rho(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5632
khdz(kts) = r6_mul(rhoz(kts), dfh(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5634
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5635
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5636
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5637
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5638
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5639
khdz(k) = r6_mul(rhoz(k), dfh(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5640
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src1); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5641
rhoz(((kte) + (1))) = rhoz(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5642
khdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfh(kte));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5644
if (((bl_mynn_edmf) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5646
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5647
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5648
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5649
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5650
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5653
if (enh_mix) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5654
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5655
khdz_old = khdz(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5656
khdz_back = r6_div(r6_mul(pblh, 0.15f), dz(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5658
if (((pblh) < (pblh_threshold))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5659
if (((emis_ant_no) > (no_threshold))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5660
khdz(k) = mynn_max((float)(r6_mul(1.1f, khdz(k))), (float)(r6_mul(r6_div(mynn_sqrt(r6_div(emis_ant_no, no_threshold)), dz(k)), rhoz(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5662
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5663
if (((frp) > (frp_threshold))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5664
kmaxfire = mynn_ceiling(mynn_log(frp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5665
khdz(k) = mynn_max((float)(r6_mul(1.1f, khdz(k))), (float)(r6_mul(r6_div(r6_mul(r6_sub(one, r6_div(k, r6_mul(kmaxfire, two))), r6_sub(mynn_ipow<2>(mynn_log(frp)), r6_mul(two, mynn_log(frp)))), dz(k)), rhoz(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5667
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5668
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5669
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; khdz_back = mynn_lane_bcast(khdz_back, mynn_src3); khdz_old = mynn_lane_bcast(khdz_old, mynn_src3); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5670
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5676
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5678
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5680
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5681
upcont(k) = r6_sub(s_awchem1(k, ic), r6_mul(s_aw1(k), r6_add(r6_mul(chem1(k, ic), upwind), r6_mul(chem1(((k) - (1)), ic), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5682
dncont(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5683
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5685
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5686
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5688
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5689
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5690
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5691
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5692
d(1) = r6_sub(chem1(k, ic), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5695
{ const int mynn_lo5 = (((kts) + (1))), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5696
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5697
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5698
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5699
d(k) = r6_sub(chem1(k, ic), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5700
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5702
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5704
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5705
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5706
b(k) = r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5707
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5708
d(k) = r6_sub(r6_sub(chem1(k, ic), r6_mul(r6_mul(dtz(k), vd1(ic)), chem1(k, ic))), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awchem1(((k) + (1)), ic)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5712
{ const int mynn_lo6 = (((kts) + (1))), mynn_hi6 = (((kte) - (1))), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5713
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5714
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5716
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5717
d(k) = r6_add(chem1(k, ic), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awchem1(k, ic), s_awchem1(((k) + (1)), ic))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5718
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5719
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5722
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5723
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5724
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5725
d(kte) = chem1(kte, ic);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5727
f_tridiag3(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5729
{ const int mynn_lo7 = (kts), mynn_hi7 = (kte), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5730
chem1(k, ic) = r6_add(x(k), settle1(k, ic));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5731
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5733
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5739-5861
__device__ __noinline__ void f_mynn_mix_scalars(int kts, int kte, int i, float delt, MynnArray<float, 1> dz, int nscalars, MynnArray<float, 2> scalars, MynnArray<float, 1> rho, float flt, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> dfh, MynnArray<float, 1> s_aw1, MynnArray<float, 2> s_awscalars1, int bl_mynn_edmf, float upwind) {
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5771
dztop = r6_mul(p5, r6_add(dz(kte), dz(((kte) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5773
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5774
dtz(k) = r6_div(delt, dz(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5775
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5779
rhoz(kts) = rho(kts);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5780
rhoinv(kts) = r6_div(one, rho(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5781
khdz(kts) = r6_mul(rhoz(kts), dfh(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5783
{ const int mynn_lo1 = (((kts) + (1))), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5784
rhoz(k) = r6_div(r6_add(r6_mul(rho(k), dz(((k) - (1)))), r6_mul(rho(((k) - (1))), dz(k))), r6_add(dz(((k) - (1))), dz(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5785
rhoz(k) = mynn_max((float)(rhoz(k)), (float)(1e-4f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5786
rhoinv(k) = r6_div(one, mynn_max((float)(rho(k)), (float)(1e-4f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5787
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5788
khdz(k) = r6_mul(rhoz(k), dfh(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5789
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src1); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5790
rhoz(((kte) + (1))) = rhoz(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5791
khdz(((kte) + (1))) = r6_mul(rhoz(((kte) + (1))), dfh(kte));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5793
if (((bl_mynn_edmf) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5795
{ const int mynn_lo2 = (((kts) + (1))), mynn_hi2 = (((kte) - (1))), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5796
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_mul(p5, s_aw1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5797
khdz(k) = mynn_max((float)(khdz(k)), (float)(r6_neg(r6_mul(p5, r6_sub(s_aw1(k), s_aw1(((k) + (1))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5798
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5799
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5805
for (ns=1; (1) > 0 ? ns <= (nscalars) : ns >= (nscalars); ns += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5807
if (((bl_mynn_edmf) > (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5809
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (((kte) - (1))), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5810
upcont(k) = r6_sub(s_awscalars1(k, ns), r6_mul(s_aw1(k), r6_add(r6_mul(scalars(k, ns), upwind), r6_mul(scalars(((k) - (1)), ns), r6_sub(one, upwind)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5811
dncont(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5812
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5814
upcont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5815
dncont(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5817
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5818
a(1) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5819
b(1) = r6_add(one, r6_mul(r6_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5820
c(1) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5821
d(1) = r6_sub(scalars(k, ns), r6_mul(dtz(k), r6_add(upcont(((k) + (1))), dncont(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5824
{ const int mynn_lo4 = (((kts) + (1))), mynn_hi4 = (((kte) - (1))), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5825
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5826
b(k) = r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5827
c(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5828
d(k) = r6_sub(scalars(k, ns), r6_mul(dtz(k), r6_sub(r6_add(r6_sub(upcont(((k) + (1))), upcont(k)), dncont(((k) + (1)))), dncont(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5829
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5831
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5833
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5834
a(k) = r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5835
b(k) = r6_sub(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5836
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5837
d(k) = r6_sub(scalars(k, ns), r6_mul(r6_mul(dtz(k), rhoinv(k)), s_awscalars1(((k) + (1)), ns)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5839
{ const int mynn_lo5 = (((kts) + (1))), mynn_hi5 = (((kte) - (1))), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5840
a(k) = r6_add(r6_mul(r6_neg(r6_mul(dtz(k), khdz(k))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5841
b(k) = r6_add(r6_add(one, r6_mul(r6_mul(dtz(k), r6_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), r6_sub(s_aw1(k), s_aw1(((k) + (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5843
c(k) = r6_sub(r6_mul(r6_neg(r6_mul(dtz(k), khdz(((k) + (1))))), rhoinv(k)), r6_mul(r6_mul(r6_mul(p5, dtz(k)), rhoinv(k)), s_aw1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5844
d(k) = r6_add(scalars(k, ns), r6_mul(r6_mul(dtz(k), rhoinv(k)), r6_sub(s_awscalars1(k, ns), s_awscalars1(((k) + (1)), ns))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5845
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5846
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5849
a(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5850
b(kte) = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5851
c(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5852
d(kte) = scalars(kte, ns);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5854
f_tridiag3(kte, a, b, c, d, x);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5856
{ const int mynn_lo6 = (kts), mynn_hi6 = (kte), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5857
scalars(k, ns) = mynn_max((float)(x(k)), (float)(1e-12f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5858
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5859
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5865-5891
__device__ __noinline__ void f_retrieve_exchange_coeffs(int kts, int kte, MynnArray<float, 1> dfm, MynnArray<float, 1> dfh, MynnArray<float, 1> dz, MynnArray<float, 1> km1, MynnArray<float, 1> kh1) {
MynnFrame mynn_frame(arena);
dfm.rebase({kts});
dfh.rebase({kts});
dz.rebase({kts});
km1.rebase({kts});
kh1.rebase({kts});
int k;
float dzk;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5879
km1(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5880
kh1(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5882
{ const int mynn_lo0 = (((kts) + (1))), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5883
dzk = r6_mul(p5, r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5884
km1(k) = r6_mul(dfm(k), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5885
kh1(k) = r6_mul(dfh(k), dzk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5886
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; dzk = mynn_lane_bcast(dzk, mynn_src0); } }
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5931-5964
__device__ __noinline__ void f_tridiag2(int kte, MynnArray<float, 1> a, MynnArray<float, 1> b, MynnArray<float, 1> c, MynnArray<float, 1> d, MynnArray<float, 1> x) {
MynnFrame mynn_frame(arena);
a.rebase({1});
b.rebase({1});
c.rebase({1});
d.rebase({1});
x.rebase({1});
MynnArray<float, 1> cp1 = mynn_array<float, 1, mynn_tile>(arena, {1}, {kte});
MynnArray<float, 1> dp1 = mynn_array<float, 1, mynn_tile>(arena, {1}, {kte});
float m;
int k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5949
cp1(1) = r6_div(c(1), b(1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5950
dp1(1) = r6_div(d(1), b(1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5952
for (k=2; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5953
m = r6_sub(b(k), r6_mul(cp1(((k) - (1))), a(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5954
cp1(k) = r6_div(c(k), m);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5955
dp1(k) = r6_div(r6_sub(d(k), r6_mul(dp1(((k) - (1))), a(k))), m);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5956
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5958
x(kte) = dp1(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5960
for (k=((kte) - (1)); ((-(1))) > 0 ? k <= (1) : k >= (1); k += ((-(1)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5961
x(k) = r6_sub(dp1(k), r6_mul(cp1(k), x(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5962
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5967-6002
__device__ __noinline__ void f_tridiag3(int kte, MynnArray<float, 1> a, MynnArray<float, 1> b, MynnArray<float, 1> c, MynnArray<float, 1> d, MynnArray<float, 1> x) {
MynnFrame mynn_frame(arena);
a.rebase({1});
b.rebase({1});
c.rebase({1});
d.rebase({1});
x.rebase({1});
constexpr int kts = 1;
int k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5988
for (k=((kte) - (1)); ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5989
d(k) = r6_sub(d(k), r6_div(r6_mul(c(k), d(((k) + (1)))), b(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5990
b(k) = r6_sub(b(k), r6_div(r6_mul(c(k), a(((k) + (1)))), b(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5991
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5993
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5994
d(k) = r6_sub(d(k), r6_div(r6_mul(a(k), d(((k) - (1)))), b(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5995
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5997
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5998
x(k) = r6_div(d(k), b(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:5999
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6001
return;
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6031-6159
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6056
kpbl = 2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6059
k = ((kts) + (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6060
kthv = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6061
minthv = 9.e9f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6062
while (((zw1(k)) <= (200.f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6063
if (((minthv) > (thv1(k)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6064
minthv = thv1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6065
kthv = k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6066
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6067
k = ((k) + (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6068
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6071
pblh = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6072
k = ((kthv) + (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6073
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6075
delt_thv = 0.75f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6076
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6078
delt_thv = 1.25f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6079
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6081
pblh = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6083
for (k=mynn_max((int)(((kthv) + (1))), (int)(((kts) + (1)))); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6084
if (((thv1(k)) >= (r6_add(minthv, delt_thv)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6085
pblh = r6_sub(zw1(k), r6_mul(dz1(((k) - (1))), mynn_min((float)(r6_div(r6_sub(thv1(k), r6_add(minthv, delt_thv)), mynn_max((float)(r6_sub(thv1(k), thv1(((k) - (1))))), (float)(1e-6f)))), (float)(one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6088
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6089
if (((k) == (((kte) - (1))))) { pblh = zw1(((kts) + (1))); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6090
if (((pblh) > (zero))) { break; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6091
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6094
if (((stable_method) == (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6099
ktke = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6100
maxqke = mynn_max((float)(zero), (float)(qke1(kts)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6103
tkeeps = r6_div(maxqke, 40.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6104
tkeeps = mynn_max((float)(tkeeps), (float)(0.01f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6105
pblh_stable = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6107
k = ((ktke) + (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6108
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6109
qtke = mynn_max((float)(zero), (float)(r6_mul(p5, qke1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6110
qtkem1 = mynn_max((float)(zero), (float)(r6_mul(p5, qke1(((k) - (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6111
if (((qtke) <= (tkeeps))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6112
pblh_stable = r6_sub(zw1(k), r6_mul(dz1(((k) - (1))), mynn_min((float)(r6_div(r6_sub(tkeeps, qtke), mynn_max((float)(r6_sub(qtkem1, qtke)), (float)(1e-6f)))), (float)(one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6115
pblh_stable = mynn_max((float)(pblh_stable), (float)(zw1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6117
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6119
if (((k) == (((kte) - (1))))) { pblh_stable = zw1(((kts) + (1))); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6120
if (((pblh_stable) != (zero))) { break; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6121
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6127
pblh_stable = mynn_min((float)(pblh_stable), (float)(r6_add(pblh, 350.f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6128
pblh_stable = mynn_max((float)(pblh_stable), (float)(mynn_max((float)(r6_sub(pblh, 350.f)), (float)(ten))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6130
if (((maxqke) <= (tkeeps))) { pblh_stable = pblh; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6131
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6133
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(pblh, hundred), hundred))), p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6134
cpblh = r6_add(r6_mul(400.f, r6_sub(one, wt)), r6_mul(700.f, wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6135
pblh_stable = mynn_max((float)(r6_mul(ust, cpblh)), (float)(ten));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6138
pblh_stable = mynn_max((float)(pblh_stable), (float)(r6_sub(pblh, sbl_lim)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6139
pblh_stable = mynn_min((float)(pblh_stable), (float)(r6_add(pblh, sbl_lim)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6140
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6143
wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(pblh, sbl_lim), sbl_damp))), p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6144
pblh = r6_add(r6_mul(pblh_stable, r6_sub(one, wt)), r6_mul(pblh, wt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6147
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6148
if (((zw1(k)) >= (pblh))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6149
kpbl = ((k) - (1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6150
break;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6151
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6152
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6181-7572
__device__ __noinline__ void f_dmp_mf(int i, int j, int kts, int kte, float dt, MynnArray<float, 1> zw1, MynnArray<float, 1> dz1, MynnArray<float, 1> zagl1, MynnArray<float, 1> pres1, MynnArray<float, 1> rho1, int momentum_opt, int tke_opt, int scalar_opt, int aerosol_opt, int numcon_opt, float bl_mynn_closure, MynnArray<float, 1> u1, MynnArray<float, 1> v1, MynnArray<float, 1> w1, MynnArray<float, 1> th1, MynnArray<float, 1> thl1, MynnArray<float, 1> thv1, MynnArray<float, 1> tk1, MynnArray<float, 1> qt1, MynnArray<float, 1> qv1, MynnArray<float, 1> qc1, MynnArray<float, 1> qi1, MynnArray<float, 1> qke1, MynnArray<float, 1> qsq1, MynnArray<float, 1> qnc1, MynnArray<float, 1> qni1, MynnArray<float, 1> qnwfa1, MynnArray<float, 1> qnifa1, MynnArray<float, 1> qnbca1, MynnArray<float, 1> ex1, MynnArray<float, 1> vt1, MynnArray<float, 1> vq1, MynnArray<float, 1> sgm1, float ust, float flt, float fltv, float flq, float flqv, float pblh, int kpbl, float dx, float landsea, float ts, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_qt1, MynnArray<float, 1> edmf_thl1, MynnArray<float, 1> edmf_ent1, MynnArray<float, 1> edmf_qc1, MynnArray<float, 1> edmf_qv1, MynnArray<float, 1> edmf_u1, MynnArray<float, 1> edmf_v1, MynnArray<float, 1> s_aw1, MynnArray<float, 1> s_awthl1, MynnArray<float, 1> s_awqt1, MynnArray<float, 1> s_awqv1, MynnArray<float, 1> s_awqc1, MynnArray<float, 1> s_awqi1, MynnArray<float, 1> s_awu1, MynnArray<float, 1> s_awv1, MynnArray<float, 1> s_awqke1, MynnArray<float, 1> s_awqsq1, MynnArray<float, 1> s_awqnc1, MynnArray<float, 1> s_awqni1, MynnArray<float, 1> s_awqnwfa1, MynnArray<float, 1> s_awqnifa1, MynnArray<float, 1> s_awqnbca1, MynnArray<float, 1> sub_thl1, MynnArray<float, 1> sub_sqv1, MynnArray<float, 1> sub_u1, MynnArray<float, 1> sub_v1, MynnArray<float, 1> det_thl1, MynnArray<float, 1> det_sqv1, MynnArray<float, 1> det_sqc1, MynnArray<float, 1> det_u1, MynnArray<float, 1> det_v1, int nchem, MynnArray<float, 2> chem1, MynnArray<float, 2> s_awchem1, bool mix_chem, int nscalars, MynnArray<float, 2> scalars, MynnArray<float, 2> s_awscalars1, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> cldfra_bl1, MynnArray<float, 1> qc_bl1_old, MynnArray<float, 1> cldfra_bl1_old, MynnOptional<bool> f_qc, MynnOptional<bool> f_qi, MynnOptional<bool> f_qnc, MynnOptional<bool> f_qni, MynnOptional<bool> f_qnwfa, MynnOptional<bool> f_qnifa, MynnOptional<bool> f_qnbca, float psig_shcu, float& maxwidth, int& ktop, float& maxmf, float& ztop, float& excess_h, float& excess_q, int spp_pbl, MynnArray<float, 1> pattern_spp_pbl1, MynnArray<float, 1> tkeprod_up, MynnArray<float, 1> el1) {
MynnFrame mynn_frame(arena);
zw1.rebase({kts});
dz1.rebase({kts});
zagl1.rebase({kts});
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
qi1.rebase({kts});
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
s_awqi1.rebase({kts});
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
MynnArray<float, 1> edmf_qi1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> s_aw2 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
constexpr int nup = 8;
constexpr int debug_mf = 0;
float nup2;
MynnArray<float, 2> upw = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upthl = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqt = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqc = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqi = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {((kte) + (1)), nup});
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
float qin;
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
float qip;
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
constexpr bool env_subs = false;
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6421
ktop = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6422
ztop = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6423
maxmf = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6424
maxwidth = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6425
excess_h = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6426
excess_q = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6428
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upw).size(); mynn_element += lane_width) { upw.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6429
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upthl).size(); mynn_element += lane_width) { upthl.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6430
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upthv).size(); mynn_element += lane_width) { upthv.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6431
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqt).size(); mynn_element += lane_width) { upqt.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6432
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upa).size(); mynn_element += lane_width) { upa.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6433
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upu).size(); mynn_element += lane_width) { upu.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6434
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upv).size(); mynn_element += lane_width) { upv.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6435
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqc).size(); mynn_element += lane_width) { upqc.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6436
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqi).size(); mynn_element += lane_width) { upqi.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6437
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqv).size(); mynn_element += lane_width) { upqv.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6438
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqke).size(); mynn_element += lane_width) { upqke.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6439
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqsq).size(); mynn_element += lane_width) { upqsq.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6440
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnc).size(); mynn_element += lane_width) { upqnc.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6441
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqni).size(); mynn_element += lane_width) { upqni.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6442
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnwfa).size(); mynn_element += lane_width) { upqnwfa.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6443
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnifa).size(); mynn_element += lane_width) { upqnifa.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6444
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upqnbca).size(); mynn_element += lane_width) { upqnbca.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6445
if (mix_chem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6446
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upchem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { upchem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6447
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6448
if (((scalar_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6449
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (upscalars.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nscalars})).size(); mynn_element += lane_width) { upscalars.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nscalars}).linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6450
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6451
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (ent).size(); mynn_element += lane_width) { ent.linear(mynn_element) = 0.001f; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6453
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a1).size(); mynn_element += lane_width) { edmf_a1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6454
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w1).size(); mynn_element += lane_width) { edmf_w1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6455
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt1).size(); mynn_element += lane_width) { edmf_qt1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6456
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl1).size(); mynn_element += lane_width) { edmf_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6457
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent1).size(); mynn_element += lane_width) { edmf_ent1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6458
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc1).size(); mynn_element += lane_width) { edmf_qc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6459
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qi1).size(); mynn_element += lane_width) { edmf_qi1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6460
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qv1).size(); mynn_element += lane_width) { edmf_qv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6461
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_u1).size(); mynn_element += lane_width) { edmf_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6462
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_v1).size(); mynn_element += lane_width) { edmf_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6463
if (mix_chem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6464
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_chem.slice(MynnRange{kts, kte}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { edmf_chem.slice(MynnRange{kts, kte}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6465
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6467
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_aw1).size(); mynn_element += lane_width) { s_aw1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6468
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awthl1).size(); mynn_element += lane_width) { s_awthl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6469
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqt1).size(); mynn_element += lane_width) { s_awqt1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6470
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqv1).size(); mynn_element += lane_width) { s_awqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6471
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqc1).size(); mynn_element += lane_width) { s_awqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6472
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqi1).size(); mynn_element += lane_width) { s_awqi1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6473
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awu1).size(); mynn_element += lane_width) { s_awu1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6474
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awv1).size(); mynn_element += lane_width) { s_awv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6475
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqke1).size(); mynn_element += lane_width) { s_awqke1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6476
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqsq1).size(); mynn_element += lane_width) { s_awqsq1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6477
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnc1).size(); mynn_element += lane_width) { s_awqnc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6478
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqni1).size(); mynn_element += lane_width) { s_awqni1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6479
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnwfa1).size(); mynn_element += lane_width) { s_awqnwfa1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6480
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnifa1).size(); mynn_element += lane_width) { s_awqnifa1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6481
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awqnbca1).size(); mynn_element += lane_width) { s_awqnbca1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6482
if (mix_chem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6483
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem})).size(); mynn_element += lane_width) { s_awchem1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem}).linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6484
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6485
if (((scalar_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6486
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars})).size(); mynn_element += lane_width) { s_awscalars1.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nscalars}).linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6487
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6489
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_thl1).size(); mynn_element += lane_width) { sub_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6490
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_sqv1).size(); mynn_element += lane_width) { sub_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6491
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_u1).size(); mynn_element += lane_width) { sub_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6492
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sub_v1).size(); mynn_element += lane_width) { sub_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6493
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_thl1).size(); mynn_element += lane_width) { det_thl1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6494
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqv1).size(); mynn_element += lane_width) { det_sqv1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6495
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_sqc1).size(); mynn_element += lane_width) { det_sqc1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6496
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_u1).size(); mynn_element += lane_width) { det_u1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6497
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (det_v1).size(); mynn_element += lane_width) { det_v1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6498
nup2 = nup;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6499
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprod_up).size(); mynn_element += lane_width) { tkeprod_up.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6501
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6502
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6503
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6504
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6505
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6509
maxw = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6510
cloud_base = 9000.0f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6511
qkebl = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6512
for (k=1; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6513
if (((zagl1(k)) > (r6_add(pblh, 500.f)))) { break; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6515
wpbl = w1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6516
if (((w1(k)) < (zero))) { wpbl = r6_mul(two, w1(k)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6517
maxw = mynn_max((float)(maxw), (float)(mynn_abs(wpbl)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6520
if (((zagl1(k)) <= (50.f))) { k50 = k; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6523
if (((k) <= (kpbl))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6524
qkebl = r6_add(qkebl, qke1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6525
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6528
qc_sgs = mynn_max((float)(qc1(k)), (float)(qc_bl1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6529
if (((((((qc_sgs) > (1e-5f))) && (((cldfra_bl1(k)) >= (p5))))) && (((cloud_base) == (9000.0f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6530
cloud_base = zw1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6531
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6532
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6533
qkebl = r6_div(qkebl, ((float)(kpbl)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6536
maxw = mynn_max((float)(r6_sub(maxw, one)), (float)(zero));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6537
psig_w = mynn_max((float)(r6_sub(one, maxw)), (float)(zero));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6538
psig_w = mynn_min((float)(psig_w), (float)(psig_shcu));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6541
fltv2 = fltv;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6542
if (((((psig_w) < (1e-2f))) && (((fltv) > (zero))))) { fltv2 = r6_neg(r6_mul(1.f, fltv)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6544
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6545
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6546
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6547
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6551
superadiabatic = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6552
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6553
hux = r6_neg(0.0024f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6554
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6555
hux = r6_neg(0.0030f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6556
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6557
tvs = r6_mul(ts, r6_add(one, r6_mul(p608, qv1(kts))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6558
if (((r6_div(r6_sub(thv1(kts), tvs), r6_mul(p5, dz1(kts)))) < (hux))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6560
superadiabatic = true;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6561
for (k=2; (1) > 0 ? k <= (mynn_max((int)(k50), (int)(2))) : k >= (mynn_max((int)(k50), (int)(2))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6562
hux = r6_neg(0.0018f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6563
dthvdz = r6_div(r6_sub(thv1(k), thv1(((k) - (1)))), r6_mul(p5, r6_add(dz1(k), dz1(((k) - (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6564
if (((dthvdz) < (hux))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6565
superadiabatic = true;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6566
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6567
superadiabatic = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6568
break;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6569
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6570
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6571
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6573
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6574
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6575
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6586
maxwidth_dx = mynn_min((float)(r6_mul(dx, dcut)), (float)(lmax));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6588
maxwidth_pbl = mynn_min((float)(r6_mul(1.1f, pblh)), (float)(lmax));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6590
if (((r6_sub(landsea, 1.5f)) < (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6591
maxwidth_cld = mynn_min((float)(mynn_max((float)(r6_mul(0.5f, cloud_base)), (float)(300.f))), (float)(lmax));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6592
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6593
maxwidth_cld = mynn_min((float)(mynn_max((float)(r6_mul(0.8f, cloud_base)), (float)(300.f))), (float)(lmax));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6594
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6596
wspd_pbl = mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts)))), (float)(0.01f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6599
if (((r6_sub(landsea, 1.5f)) < (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6600
maxwidth_flx = mynn_max((float)(zero), (float)(mynn_min((float)(r6_mul(1000.f, r6_add(r6_mul(0.6f, mynn_tanh(r6_div(r6_sub(fltv, 0.040f), 0.04f))), p5))), (float)(1000.f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6601
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6602
maxwidth_flx = mynn_max((float)(zero), (float)(mynn_min((float)(r6_mul(1000.f, r6_add(r6_mul(0.6f, mynn_tanh(r6_div(r6_sub(fltv, 0.007f), 0.02f))), p5))), (float)(1000.f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6604
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6605
maxwidth = mynn_min((float)(maxwidth_dx), (float)(maxwidth_pbl));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6606
maxwidth = mynn_min((float)(maxwidth), (float)(maxwidth_cld));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6607
maxwidth = mynn_min((float)(maxwidth), (float)(maxwidth_flx));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6608
minwidth = lmin;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6610
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6611
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6612
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6613
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6614
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6615
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6616
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6617
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6618
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6619
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6621
if (((maxwidth) <= (minwidth))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6622
nup2 = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6623
maxwidth = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6624
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6627
if (((((((((flt) > (zero))) && (((fltv2) > (0.008f))))) && (((maxwidth) > (minwidth))))) && (superadiabatic))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6630
cn = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6631
d = r6_neg(1.9f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6632
dl = r6_div(r6_sub(maxwidth, minwidth), ((float)(((nup) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6633
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6635
l = r6_add(minwidth, r6_mul(dl, ((float)(((ip) - (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6636
cn = r6_add(cn, r6_mul(r6_div(r6_mul(mynn_pow(l, d), r6_mul(l, l)), r6_mul(dx, dx)), dl));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6637
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6638
c = r6_div(atot, cn);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6641
if (((r6_sub(landsea, 1.5f)) < (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6642
acfac = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(fltv2, 0.02f), 0.05f))), p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6643
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6644
acfac = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(fltv2, 0.012f), 0.03f))), p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6645
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6649
ac_wsp = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(zero), (float)(r6_sub(wspd_pbl, 13.0f))), 12.f)), (float)(one)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6650
acfac = mynn_min((float)(acfac), (float)(ac_wsp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6653
an2 = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6654
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6656
l = r6_add(minwidth, r6_mul(dl, ((float)(((ip) - (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6657
n = r6_mul(c, mynn_pow(l, d));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6658
upa(1, ip) = r6_mul(r6_div(r6_mul(r6_mul(n, l), l), r6_mul(dx, dx)), dl);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6660
upa(1, ip) = r6_mul(upa(1, ip), acfac);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6661
an2 = r6_add(an2, upa(1, ip));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6664
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6666
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6667
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6668
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6669
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6670
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6671
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6672
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6675
z0 = 50.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6676
pwmin = 0.1f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6677
pwmax = 0.4f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6679
wstar = mynn_max((float)(mynn_pow(r6_mul(r6_mul(gtr, fltv2), pblh), p333)), (float)(1.e-2f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6680
qstar = r6_div(mynn_max((float)(flq), (float)(1e-5f)), wstar);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6681
thstar = r6_div(flt, wstar);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6683
csigma = 1.34f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6685
if (env_subs) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6686
exc_fac = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6687
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6688
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6691
exc_fac = 1.3f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6692
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6695
exc_fac = 0.12f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6696
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6697
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6699
exc_fac = r6_mul(exc_fac, ac_wsp);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6702
sigmaw = r6_mul(r6_mul(r6_mul(csigma, wstar), mynn_pow(r6_div(z0, pblh), p333)), r6_sub(one, r6_div(r6_mul(0.8f, z0), pblh)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6703
sigmaqt = r6_mul(r6_mul(csigma, qstar), mynn_pow(r6_div(z0, pblh), r6_neg(p333)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6704
sigmath = r6_mul(r6_mul(csigma, thstar), mynn_pow(r6_div(z0, pblh), r6_neg(p333)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6708
wmin = mynn_min((float)(r6_mul(sigmaw, pwmin)), (float)(0.1f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6709
wmax = mynn_min((float)(r6_mul(sigmaw, pwmax)), (float)(0.5f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6711
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6712
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6713
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6714
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6715
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6716
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6719
{ const int mynn_lo0 = (1), mynn_hi0 = (nup), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { ip = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6720
wlv = r6_add(wmin, r6_mul(r6_div(r6_sub(wmax, wmin), ((float)(nup2))), ((float)(((ip) - (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6721
upa(2, ip) = upa(1, ip);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6723
upw(2, ip) = r6_add(wmin, r6_mul(r6_div(((float)(ip)), ((float)(nup))), r6_sub(wmax, wmin)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6724
upu(2, ip) = r6_div(r6_add(r6_mul(u1(kts), dz1(((kts) + (1)))), r6_mul(u1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6725
upv(2, ip) = r6_div(r6_add(r6_mul(v1(kts), dz1(((kts) + (1)))), r6_mul(v1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6726
upqc(2, ip) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6728
upqi(2, ip) = r6_div(r6_add(r6_mul(qi1(kts), dz1(((kts) + (1)))), r6_mul(qi1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6730
exc_heat = r6_div(r6_mul(r6_mul(exc_fac, upw(2, ip)), sigmath), sigmaw);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6731
upthv(2, ip) = r6_add(r6_div(r6_add(r6_mul(thv1(kts), dz1(((kts) + (1)))), r6_mul(thv1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1))))), exc_heat);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6733
upthl(2, ip) = r6_add(r6_div(r6_add(r6_mul(thl1(kts), dz1(((kts) + (1)))), r6_mul(thl1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1))))), exc_heat);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6736
exc_moist = r6_div(r6_mul(r6_mul(exc_fac, upw(2, ip)), sigmaqt), sigmaw);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6737
upqt(2, ip) = r6_add(r6_div(r6_add(r6_mul(qt1(kts), dz1(((kts) + (1)))), r6_mul(qt1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1))))), exc_moist);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6739
upqke(2, ip) = r6_div(r6_add(r6_mul(qke1(kts), dz1(((kts) + (1)))), r6_mul(qke1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6740
upqsq(2, ip) = r6_div(r6_add(r6_mul(qsq1(kts), dz1(((kts) + (1)))), r6_mul(qsq1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6741
upqnc(2, ip) = r6_div(r6_add(r6_mul(qnc1(kts), dz1(((kts) + (1)))), r6_mul(qnc1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6742
upqni(2, ip) = r6_div(r6_add(r6_mul(qni1(kts), dz1(((kts) + (1)))), r6_mul(qni1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6743
upqnwfa(2, ip) = r6_div(r6_add(r6_mul(qnwfa1(kts), dz1(((kts) + (1)))), r6_mul(qnwfa1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6744
upqnifa(2, ip) = r6_div(r6_add(r6_mul(qnifa1(kts), dz1(((kts) + (1)))), r6_mul(qnifa1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6745
upqnbca(2, ip) = r6_div(r6_add(r6_mul(qnbca1(kts), dz1(((kts) + (1)))), r6_mul(qnbca1(((kts) + (1))), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6746
} mynn_lane_sync(); ip = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; exc_heat = mynn_lane_bcast(exc_heat, mynn_src0); exc_moist = mynn_lane_bcast(exc_moist, mynn_src0); wlv = mynn_lane_bcast(wlv, mynn_src0); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6748
if (mix_chem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6749
{ const int mynn_lo1 = (1), mynn_hi1 = (nup), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { ip = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6750
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6751
upchem(2, ip, ic) = r6_div(r6_add(r6_mul(chem1(kts, ic), dz1(((kts) + (1)))), r6_mul(chem1(((kts) + (1)), ic), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6752
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6753
} mynn_lane_sync(); ip = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; ic = mynn_lane_bcast(ic, mynn_src1); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6754
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6755
if (((scalar_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6756
{ const int mynn_lo2 = (1), mynn_hi2 = (nup), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { ip = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6757
for (ic=1; (1) > 0 ? ic <= (nscalars) : ic >= (nscalars); ic += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6758
upscalars(2, ip, ic) = r6_div(r6_add(r6_mul(scalars(kts, ic), dz1(((kts) + (1)))), r6_mul(scalars(((kts) + (1)), ic), dz1(kts))), r6_add(dz1(kts), dz1(((kts) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6759
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6760
} mynn_lane_sync(); ip = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; ic = mynn_lane_bcast(ic, mynn_src2); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6761
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6764
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_thl.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_thl.slice(MynnRange{kts, kte}).linear(mynn_element) = thl1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6765
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_sqv.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_sqv.slice(MynnRange{kts, kte}).linear(mynn_element) = qv1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6766
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_sqc.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_sqc.slice(MynnRange{kts, kte}).linear(mynn_element) = qc1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6767
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_u.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_u.slice(MynnRange{kts, kte}).linear(mynn_element) = u1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6768
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (envm_v.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { envm_v.slice(MynnRange{kts, kte}).linear(mynn_element) = v1.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6770
rhoi(kts) = rho1(kts);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6771
dzi(kts) = r6_mul(p5, dz1(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6772
exneri(kts) = ex1(kts);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6773
{ const int mynn_lo3 = (((kts) + (1))), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6774
rhoi(k) = r6_div(r6_add(r6_mul(rho1(((k) - (1))), dz1(k)), r6_mul(rho1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6775
dzi(k) = r6_div(r6_add(r6_mul(dz1(((k) - (1))), dz1(k)), r6_mul(dz1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6776
exneri(k) = r6_div(r6_add(r6_mul(ex1(((k) - (1))), dz1(k)), r6_mul(ex1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6777
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6780
dxsa = r6_sub(one, mynn_min((float)(mynn_max((float)(zero), (float)(r6_div(r6_sub(12000.0f, dx), r6_sub(12000.0f, 3000.0f))))), (float)(one)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6782
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (ktop_plume).size(); mynn_element += lane_width) { ktop_plume.linear(mynn_element) = 0; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6784
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6785
qcn = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6786
overshoot = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6787
l = r6_add(minwidth, r6_mul(dl, ((float)(((ip) - (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6788
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (2))) : k >= (((kte) - (2))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6791
wmin = r6_add(p3, r6_mul(l, 0.0005f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6799
entfac = r6_mul(0.21f, mynn_min((float)(mynn_max((float)(mynn_sqrt(qkebl)), (float)(1.30f))), (float)(1.57f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6802
wt2 = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(zagl1(k), pblh)), (float)(zero)), 500.f)), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6803
entfac = r6_add(r6_mul(entfac, r6_sub(one, wt2)), r6_mul(wt2, 0.33f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6804
ent(k, ip) = r6_div(entfac, r6_mul(mynn_min((float)(mynn_max((float)(upw(((k) - (1)), ip)), (float)(wmin))), (float)(one)), l));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6812
ent(k, ip) = mynn_max((float)(ent(k, ip)), (float)(0.0003f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6816
if (((zw1(k)) >= (mynn_min((float)(r6_add(pblh, 1500.f)), (float)(4000.f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6817
ent(k, ip) = r6_add(ent(k, ip), r6_mul(r6_sub(zw1(k), mynn_min((float)(r6_add(pblh, 1500.f)), (float)(4000.f))), 5.0e-6f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6818
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6821
ent(k, ip) = r6_mul(ent(k, ip), r6_sub(one, pattern_spp_pbl1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6823
ent(k, ip) = mynn_min((float)(ent(k, ip)), (float)(r6_div(0.9f, dz1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6826
uk = r6_div(r6_add(r6_mul(u1(k), dz1(((k) + (1)))), r6_mul(u1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6827
ukm1 = r6_div(r6_add(r6_mul(u1(((k) - (1))), dz1(k)), r6_mul(u1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6828
vk = r6_div(r6_add(r6_mul(v1(k), dz1(((k) + (1)))), r6_mul(v1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6829
vkm1 = r6_div(r6_add(r6_mul(v1(((k) - (1))), dz1(k)), r6_mul(v1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6832
entexp = r6_mul(ent(k, ip), dz1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6833
entexm = r6_mul(entexp, p333);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6834
qtn = r6_add(r6_mul(upqt(k, ip), r6_sub(one, entexp)), r6_mul(qt1(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6835
qin = r6_add(r6_mul(upqi(k, ip), r6_sub(one, entexp)), r6_mul(qi1(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6836
thln = r6_add(r6_mul(upthl(k, ip), r6_sub(one, entexp)), r6_mul(thl1(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6837
un = r6_add(r6_add(r6_mul(upu(k, ip), r6_sub(one, entexm)), r6_mul(u1(k), entexm)), r6_mul(r6_mul(dxsa, pgfac), r6_sub(uk, ukm1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6838
vn = r6_add(r6_add(r6_mul(upv(k, ip), r6_sub(one, entexm)), r6_mul(v1(k), entexm)), r6_mul(r6_mul(dxsa, pgfac), r6_sub(vk, vkm1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6839
qken = r6_add(r6_mul(upqke(k, ip), r6_sub(one, entexp)), r6_mul(qke1(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6840
qsqn = r6_add(r6_mul(upqsq(k, ip), r6_sub(one, entexp)), r6_mul(qsq1(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6841
qncn = r6_add(r6_mul(upqnc(k, ip), r6_sub(one, entexp)), r6_mul(qnc1(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6842
qnin = r6_add(r6_mul(upqni(k, ip), r6_sub(one, entexp)), r6_mul(qni1(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6843
qnwfan = r6_add(r6_mul(upqnwfa(k, ip), r6_sub(one, entexp)), r6_mul(qnwfa1(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6844
qnifan = r6_add(r6_mul(upqnifa(k, ip), r6_sub(one, entexp)), r6_mul(qnifa1(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6845
qnbcan = r6_add(r6_mul(upqnbca(k, ip), r6_sub(one, entexp)), r6_mul(qnbca1(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6849
qc_ent = qcn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6850
qt_ent = qtn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6851
thl_ent = thln;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6861
if (mix_chem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6862
{ const int mynn_lo4 = (1), mynn_hi4 = (nchem), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { ic = mynn_lo4 + mynn_it4*mynn_st4;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6866
chemn(ic) = r6_add(r6_mul(upchem(k, ip, ic), r6_sub(one, entexp)), r6_mul(chem1(k, ic), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6867
} mynn_lane_sync(); ic = mynn_lo4 + mynn_n4*mynn_st4; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6868
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6869
if (((scalar_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6870
{ const int mynn_lo5 = (1), mynn_hi5 = (nscalars), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { ic = mynn_lo5 + mynn_it5*mynn_st5;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6871
scalarsn(ic) = r6_add(r6_mul(upscalars(k, ip, ic), r6_sub(one, entexp)), r6_mul(scalars(k, ic), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6872
} mynn_lane_sync(); ic = mynn_lo5 + mynn_n5*mynn_st5; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6873
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6876
pk = r6_div(r6_add(r6_mul(pres1(k), dz1(((k) + (1)))), r6_mul(pres1(((k) + (1))), dz1(k))), r6_add(dz1(k), dz1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6878
f_condensation_edmf(qtn, thln, pk, zw1(((k) + (1))), thvn, qcn);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6881
thvk = r6_div(r6_add(r6_mul(thv1(k), dz1(((k) + (1)))), r6_mul(thv1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6882
thvkm1 = r6_div(r6_add(r6_mul(thv1(((k) - (1))), dz1(k)), r6_mul(thv1(k), dz1(((k) - (1))))), r6_add(dz1(((k) - (1))), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6885
b = r6_mul(grav, r6_sub(r6_div(thvn, thvk), one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6886
if (((b) > (0.f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6887
bcoeff = 0.15f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6888
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6889
bcoeff = 0.2f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6890
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6900
if (((upw(((k) - (1)), ip)) < (0.2f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6901
wn = r6_add(upw(k, ip), r6_mul(r6_add(r6_mul(r6_neg(r6_mul(2.f, ent(k, ip))), upw(k, ip)), r6_div(r6_mul(bcoeff, b), mynn_max((float)(upw(k, ip)), (float)(0.2f)))), mynn_min((float)(dz1(k)), (float)(250.f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6902
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6903
wn = r6_add(upw(k, ip), r6_mul(r6_add(r6_mul(r6_neg(r6_mul(2.f, ent(k, ip))), upw(k, ip)), r6_div(r6_mul(bcoeff, b), upw(k, ip))), mynn_min((float)(dz1(k)), (float)(250.f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6904
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6907
if (((wn) > (r6_add(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6908
wn = r6_add(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6909
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6911
if (((wn) < (r6_sub(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6912
wn = r6_sub(upw(k, ip), mynn_min((float)(r6_div(r6_mul(1.25f, dz1(k)), 200.f)), (float)(two)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6913
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6914
wn = mynn_min((float)(mynn_max((float)(zero), (float)(wn))), (float)(3.0f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6918
if (((((k) == (((kts) + (1))))) && (((wn) < (1e-8f))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6919
nup2 = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6920
break;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6921
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6923
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6924
if (((wn) >= (3.0f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6926
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6927
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6928
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6929
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6930
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6931
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6934
if (((((wn) <= (zero))) && (((overshoot) == (0))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6935
overshoot = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6936
if (((r6_sub(thvk, thvkm1)) > (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6937
bvf = mynn_sqrt(r6_div(r6_mul(gtr, r6_sub(thvk, thvkm1)), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6939
frz = r6_div(upw(k, ip), r6_mul(bvf, dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6941
dzp = r6_mul(dz1(k), mynn_max((float)(zero), (float)(mynn_min((float)(frz), (float)(one)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6942
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6943
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6944
dzp = dz1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6945
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6955
aratio = mynn_min((float)(r6_div(upa(k, ip), r6_sub(one, upa(k, ip)))), (float)(p5));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6956
detturb = 0.00008f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6957
oow = r6_neg(r6_div(0.060f, mynn_max((float)(r6_mul(0.5f, r6_add(wn, upw(k, ip)))), (float)(one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6958
detrate = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(oow, r6_sub(wn, upw(k, ip))), dz1(k))), (float)(detturb))), (float)(.0002f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6959
detrateuv = mynn_min((float)(mynn_max((float)(r6_div(r6_mul(oow, r6_sub(wn, upw(k, ip))), dz1(k))), (float)(detturb))), (float)(.0001f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6960
envm_thl(k) = r6_add(envm_thl(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(thl_ent, upthl(k, ip))), thl1(k)), detrate), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6961
qv_ent = r6_mul(p5, r6_add(mynn_max((float)(zero), (float)(r6_sub(qt_ent, qc_ent))), mynn_max((float)(zero), (float)(r6_sub(upqt(k, ip), upqc(k, ip))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6962
envm_sqv(k) = r6_add(envm_sqv(k), r6_mul(r6_mul(r6_mul(r6_sub(qv_ent, qv1(k)), detrate), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6963
if (((upqc(k, ip)) > (1e-8f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6964
if (((qc1(k)) > (1e-6f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6965
qc_grid = qc1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6966
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6967
qc_grid = qc_bl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6968
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6969
envm_sqc(k) = r6_add(envm_sqc(k), r6_mul(r6_mul(r6_mul(mynn_max((float)(zero), (float)(r6_sub(r6_mul(r6_mul(upa(k, ip), p5), r6_add(qcn, upqc(k, ip))), qc_grid))), detrate), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6970
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6971
envm_u(k) = r6_add(envm_u(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(un, upu(k, ip))), u1(k)), detrateuv), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6972
envm_v(k) = r6_add(envm_v(k), r6_mul(r6_mul(r6_mul(r6_sub(r6_mul(p5, r6_add(vn, upv(k, ip))), v1(k)), detrateuv), aratio), mynn_min((float)(dzp), (float)(dzpmax))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6974
if (((wn) > (0.f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6976
upw(((k) + (1)), ip) = wn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6977
upthv(((k) + (1)), ip) = thvn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6978
upthl(((k) + (1)), ip) = thln;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6979
upqt(((k) + (1)), ip) = qtn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6982
upqc(((k) + (1)), ip) = mynn_max((float)(r6_sub(qcn, qin)), (float)(zero));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6983
upqi(((k) + (1)), ip) = qin;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6984
upu(((k) + (1)), ip) = un;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6985
upv(((k) + (1)), ip) = vn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6986
upqke(((k) + (1)), ip) = qken;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6987
upqsq(((k) + (1)), ip) = qsqn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6988
upqnc(((k) + (1)), ip) = qncn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6989
upqni(((k) + (1)), ip) = qnin;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6990
upqnwfa(((k) + (1)), ip) = qnwfan;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6991
upqnifa(((k) + (1)), ip) = qnifan;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6992
upqnbca(((k) + (1)), ip) = qnbcan;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6993
upa(((k) + (1)), ip) = upa(k, ip);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6994
if (mix_chem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6995
{ const int mynn_lo6 = (1), mynn_hi6 = (nchem), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { ic = mynn_lo6 + mynn_it6*mynn_st6;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6996
upchem(((k) + (1)), ip, ic) = chemn(ic);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6997
} mynn_lane_sync(); ic = mynn_lo6 + mynn_n6*mynn_st6; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6998
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:6999
if (((scalar_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7000
{ const int mynn_lo7 = (1), mynn_hi7 = (nscalars), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { ic = mynn_lo7 + mynn_it7*mynn_st7;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7001
upscalars(((k) + (1)), ip, ic) = scalarsn(ic);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7002
} mynn_lane_sync(); ic = mynn_lo7 + mynn_n7*mynn_st7; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7003
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7004
ktop = mynn_max((int)(ktop), (int)(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7005
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7006
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7007
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7008
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7009
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7010
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7011
break;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7012
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7013
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7015
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7016
if (((((((((mynn_maxval(upw.slice(MynnRange{upw.lo[0], upw.hi[0]}, ip))) > (10.0f))) || (((mynn_minval(upa.slice(MynnRange{upa.lo[0], upa.hi[0]}, ip))) < (zero))))) || (((mynn_maxval(upa.slice(MynnRange{upa.lo[0], upa.hi[0]}, ip))) > (atot))))) || (((nup2) > (10))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7019
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7020
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7021
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7023
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7024
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7025
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7026
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7027
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7028
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7029
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7030
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7031
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7032
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7033
ktop_plume(ip) = k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7034
excess_h = mynn_max((float)(excess_h), (float)(exc_heat));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7035
excess_q = mynn_max((float)(excess_q), (float)(exc_moist));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7036
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7037
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7039
nup2 = 0.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7040
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7042
ktop = mynn_min((int)(ktop), (int)(((kte) - (1))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7043
if (((ktop) == (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7044
ztop = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7045
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7046
ztop = zw1(ktop);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7047
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7049
if (((nup2) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7052
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7053
{ const int mynn_lo8 = (kts), mynn_hi8 = (((kte) - (1))), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7054
s_aw1(k) = r6_add(s_aw1(k), r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7055
s_awthl1(k) = r6_add(s_awthl1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upthl(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7056
s_awqt1(k) = r6_add(s_awqt1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqt(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7057
s_awqi1(k) = r6_add(s_awqi1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqi(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7062
qc_plume = upqc(k, ip);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7066
s_awqc1(k) = r6_add(s_awqc1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), qc_plume), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7067
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; if (mynn_n8 > 0) { const int mynn_src8 = (mynn_n8 - 1) % lane_width; qc_plume = mynn_lane_bcast(qc_plume, mynn_src8); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7068
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7069
{ const int mynn_lo9 = (kts), mynn_hi9 = (((kte) - (1))), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7070
s_awqv1(k) = mynn_max((float)(r6_sub(r6_sub(s_awqt1(k), s_awqc1(k)), s_awqi1(k))), (float)(zero));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7071
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7073
if (((momentum_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7074
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7075
{ const int mynn_lo10 = (kts), mynn_hi10 = (((kte) - (1))), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7076
s_awu1(k) = r6_add(s_awu1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upu(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7077
s_awv1(k) = r6_add(s_awv1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upv(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7078
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7079
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7080
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7082
if (((tke_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7083
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7084
{ const int mynn_lo11 = (kts), mynn_hi11 = (((kte) - (1))), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { k = mynn_lo11 + mynn_it11*mynn_st11;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7085
s_awqke1(k) = r6_add(s_awqke1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqke(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7086
} mynn_lane_sync(); k = mynn_lo11 + mynn_n11*mynn_st11; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7087
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7089
if (((bl_mynn_closure) > (2.5f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7090
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7091
{ const int mynn_lo12 = (kts), mynn_hi12 = (((kte) - (1))), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { k = mynn_lo12 + mynn_it12*mynn_st12;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7092
s_awqsq1(k) = r6_add(s_awqsq1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqsq(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7093
} mynn_lane_sync(); k = mynn_lo12 + mynn_n12*mynn_st12; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7094
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7095
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7096
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7098
if (mix_chem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7099
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7100
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7101
{ const int mynn_lo13 = (1), mynn_hi13 = (nchem), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { ic = mynn_lo13 + mynn_it13*mynn_st13;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7102
s_awchem1(k, ic) = r6_add(s_awchem1(k, ic), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upchem(k, ip, ic)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7103
} mynn_lane_sync(); ic = mynn_lo13 + mynn_n13*mynn_st13; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7104
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7105
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7106
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7108
if (((scalar_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7109
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7110
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7111
{ const int mynn_lo14 = (1), mynn_hi14 = (nscalars), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { ic = mynn_lo14 + mynn_it14*mynn_st14;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7112
s_awscalars1(k, ic) = r6_add(s_awscalars1(k, ic), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upscalars(k, ip, ic)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7113
} mynn_lane_sync(); ic = mynn_lo14 + mynn_n14*mynn_st14; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7114
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7115
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7116
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7117
if (((numcon_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7118
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7119
{ const int mynn_lo15 = (kts), mynn_hi15 = (((kte) - (1))), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { k = mynn_lo15 + mynn_it15*mynn_st15;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7120
s_awqnc1(k) = r6_add(s_awqnc1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnc(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7121
s_awqni1(k) = r6_add(s_awqni1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqni(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7122
} mynn_lane_sync(); k = mynn_lo15 + mynn_n15*mynn_st15; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7123
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7124
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7125
if (((aerosol_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7126
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7127
{ const int mynn_lo16 = (kts), mynn_hi16 = (((kte) - (1))), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { k = mynn_lo16 + mynn_it16*mynn_st16;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7128
s_awqnwfa1(k) = r6_add(s_awqnwfa1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnwfa(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7129
s_awqnifa1(k) = r6_add(s_awqnifa1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnifa(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7130
s_awqnbca1(k) = r6_add(s_awqnbca1(k), r6_mul(r6_mul(r6_mul(r6_mul(rhoi(k), upa(k, ip)), upw(k, ip)), upqnbca(k, ip)), psig_w));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7131
} mynn_lane_sync(); k = mynn_lo16 + mynn_n16*mynn_st16; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7132
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7133
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7138
if (((mynn_abs(s_aw1(((kts) + (1))))) > (1e-8f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7139
flx1 = mynn_max((float)(r6_div(r6_mul(s_aw1(((kts) + (1))), r6_sub(thv1(kts), thv1(((kts) + (1))))), dzi(((kts) + (1))))), (float)(1.0e-6f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7140
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7141
flx1 = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7144
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7145
adjustment = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7146
flt2 = mynn_max((float)(zero), (float)(fltv));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7149
if (((((flx1) > (r6_div(r6_mul(fluxportion, flt2), dz1(kts))))) && (((flx1) > (zero))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7150
adjustment = mynn_max((float)(r6_div(r6_div(r6_mul(fluxportion, flt2), dz1(kts)), flx1)), (float)(0.01f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7151
for (int mynn_element=0; mynn_element < (s_aw1).size(); ++mynn_element) { s_aw1.linear(mynn_element) = r6_mul(s_aw1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7152
for (int mynn_element=0; mynn_element < (s_awthl1).size(); ++mynn_element) { s_awthl1.linear(mynn_element) = r6_mul(s_awthl1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7153
for (int mynn_element=0; mynn_element < (s_awqt1).size(); ++mynn_element) { s_awqt1.linear(mynn_element) = r6_mul(s_awqt1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7154
for (int mynn_element=0; mynn_element < (s_awqc1).size(); ++mynn_element) { s_awqc1.linear(mynn_element) = r6_mul(s_awqc1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7155
for (int mynn_element=0; mynn_element < (s_awqi1).size(); ++mynn_element) { s_awqi1.linear(mynn_element) = r6_mul(s_awqi1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7156
for (int mynn_element=0; mynn_element < (s_awqv1).size(); ++mynn_element) { s_awqv1.linear(mynn_element) = r6_mul(s_awqv1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7157
if (((numcon_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7158
for (int mynn_element=0; mynn_element < (s_awqnc1).size(); ++mynn_element) { s_awqnc1.linear(mynn_element) = r6_mul(s_awqnc1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7159
for (int mynn_element=0; mynn_element < (s_awqni1).size(); ++mynn_element) { s_awqni1.linear(mynn_element) = r6_mul(s_awqni1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7160
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7161
if (((aerosol_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7162
for (int mynn_element=0; mynn_element < (s_awqnwfa1).size(); ++mynn_element) { s_awqnwfa1.linear(mynn_element) = r6_mul(s_awqnwfa1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7163
for (int mynn_element=0; mynn_element < (s_awqnifa1).size(); ++mynn_element) { s_awqnifa1.linear(mynn_element) = r6_mul(s_awqnifa1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7164
for (int mynn_element=0; mynn_element < (s_awqnbca1).size(); ++mynn_element) { s_awqnbca1.linear(mynn_element) = r6_mul(s_awqnbca1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7165
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7166
if (((momentum_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7167
for (int mynn_element=0; mynn_element < (s_awu1).size(); ++mynn_element) { s_awu1.linear(mynn_element) = r6_mul(s_awu1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7168
for (int mynn_element=0; mynn_element < (s_awv1).size(); ++mynn_element) { s_awv1.linear(mynn_element) = r6_mul(s_awv1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7169
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7170
if (((tke_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7171
for (int mynn_element=0; mynn_element < (s_awqke1).size(); ++mynn_element) { s_awqke1.linear(mynn_element) = r6_mul(s_awqke1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7172
if (((bl_mynn_closure) > (2.5f))) { for (int mynn_element=0; mynn_element < (s_awqsq1).size(); ++mynn_element) { s_awqsq1.linear(mynn_element) = r6_mul(s_awqsq1.linear(mynn_element), adjustment); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7173
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7174
if (mix_chem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7175
for (int mynn_element=0; mynn_element < (s_awchem1).size(); ++mynn_element) { s_awchem1.linear(mynn_element) = r6_mul(s_awchem1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7176
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7177
if (((scalar_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7178
for (int mynn_element=0; mynn_element < (s_awscalars1).size(); ++mynn_element) { s_awscalars1.linear(mynn_element) = r6_mul(s_awscalars1.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7179
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7180
for (int mynn_element=0; mynn_element < (upa).size(); ++mynn_element) { upa.linear(mynn_element) = r6_mul(upa.linear(mynn_element), adjustment); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7181
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7182
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7183
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7184
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7185
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7186
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7190
k = kts;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7191
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7192
edmf_a1(k) = r6_add(edmf_a1(k), upa(((k) + (1)), ip));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7193
edmf_w1(k) = r6_add(edmf_w1(k), r6_mul(r6_mul(upa(((k) + (1)), ip), p5), upw(((k) + (1)), ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7194
edmf_qt1(k) = r6_add(edmf_qt1(k), r6_mul(upa(((k) + (1)), ip), upqt(((k) + (1)), ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7195
edmf_thl1(k) = r6_add(edmf_thl1(k), r6_mul(upa(((k) + (1)), ip), upthl(((k) + (1)), ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7196
edmf_ent1(k) = r6_add(edmf_ent1(k), r6_mul(upa(((k) + (1)), ip), ent(((k) + (1)), ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7197
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_mul(upa(((k) + (1)), ip), upqc(((k) + (1)), ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7198
edmf_qi1(k) = r6_add(edmf_qi1(k), r6_mul(upa(((k) + (1)), ip), upqi(((k) + (1)), ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7199
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7200
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7201
{ const int mynn_lo17 = (((kts) + (1))), mynn_hi17 = (((ktop_plume(ip)) - (1))), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { k = mynn_lo17 + mynn_it17*mynn_st17;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7202
upak = r6_div(r6_add(r6_mul(upa(((k) + (1)), ip), dzi(k)), r6_mul(upa(k, ip), dzi(((k) + (1))))), r6_add(dzi(((k) + (1))), dzi(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7203
edmf_a1(k) = r6_add(edmf_a1(k), upak);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7204
edmf_w1(k) = r6_add(edmf_w1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upw(((k) + (1)), ip), dzi(k)), r6_mul(upw(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7205
edmf_qt1(k) = r6_add(edmf_qt1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upqt(((k) + (1)), ip), dzi(k)), r6_mul(upqt(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7206
edmf_thl1(k) = r6_add(edmf_thl1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upthl(((k) + (1)), ip), dzi(k)), r6_mul(upthl(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7207
edmf_ent1(k) = r6_add(edmf_ent1(k), r6_div(r6_mul(upak, r6_add(r6_mul(ent(((k) + (1)), ip), dzi(k)), r6_mul(ent(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7208
if (((((mynn_abs(upqc(k, ip))) < (1e-10f))) && (((upqc(((k) + (1)), ip)) > (zero))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7210
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_mul(upak, upqc(((k) + (1)), ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7211
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7212
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upqc(((k) + (1)), ip), dzi(k)), r6_mul(upqc(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7213
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7214
if (((((mynn_abs(upqi(k, ip))) < (1e-10f))) && (((upqi(((k) + (1)), ip)) > (zero))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7216
edmf_qi1(k) = r6_add(edmf_qi1(k), r6_mul(upak, upqi(((k) + (1)), ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7217
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7218
edmf_qi1(k) = r6_add(edmf_qi1(k), r6_div(r6_mul(upak, r6_add(r6_mul(upqi(((k) + (1)), ip), dzi(k)), r6_mul(upqi(k, ip), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7219
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7220
} mynn_lane_sync(); k = mynn_lo17 + mynn_n17*mynn_st17; if (mynn_n17 > 0) { const int mynn_src17 = (mynn_n17 - 1) % lane_width; upak = mynn_lane_bcast(upak, mynn_src17); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7221
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7223
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7224
k = ktop_plume(ip);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7225
edmf_a1(k) = r6_add(edmf_a1(k), upa(k, ip));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7226
edmf_w1(k) = r6_add(edmf_w1(k), r6_mul(upa(k, ip), upw(k, ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7227
edmf_qt1(k) = r6_add(edmf_qt1(k), r6_mul(upa(k, ip), upqt(k, ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7228
edmf_thl1(k) = r6_add(edmf_thl1(k), r6_mul(upa(k, ip), upthl(k, ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7229
edmf_ent1(k) = r6_add(edmf_ent1(k), r6_mul(upa(k, ip), ent(k, ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7230
edmf_qc1(k) = r6_add(edmf_qc1(k), r6_mul(upa(k, ip), upqc(k, ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7231
edmf_qi1(k) = r6_add(edmf_qi1(k), r6_mul(upa(k, ip), upqi(k, ip)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7232
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7233
{ const int mynn_lo18 = (kts), mynn_hi18 = (((kte) - (1))), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { k = mynn_lo18 + mynn_it18*mynn_st18;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7235
if (((edmf_a1(k)) > (0.f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7236
edmf_w1(k) = r6_div(edmf_w1(k), edmf_a1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7237
edmf_qt1(k) = r6_div(edmf_qt1(k), edmf_a1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7238
edmf_thl1(k) = r6_div(edmf_thl1(k), edmf_a1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7239
edmf_ent1(k) = r6_div(edmf_ent1(k), edmf_a1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7240
edmf_qc1(k) = r6_div(edmf_qc1(k), edmf_a1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7241
edmf_qi1(k) = r6_div(edmf_qi1(k), edmf_a1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7242
edmf_a1(k) = r6_mul(edmf_a1(k), psig_w);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7243
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7244
} mynn_lane_sync(); k = mynn_lo18 + mynn_n18*mynn_st18; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7245
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7247
if (((r6_mul(edmf_a1(k), edmf_w1(k))) > (maxmf))) { maxmf = r6_mul(edmf_a1(k), edmf_w1(k)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7250
tkeprod_up(k) = r6_div(r6_mul(mynn_ipow<3>(mynn_abs(edmf_w1(k))), edmf_a1(k)), r6_mul(b1, mynn_max((float)(r6_mul(p5, r6_add(el1(k), el1(((k) + (1)))))), (float)(0.1f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7251
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7254
if (mix_chem) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7255
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7256
for (ip=1; (1) > 0 ? ip <= (nup) : ip >= (nup); ip += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7257
{ const int mynn_lo19 = (1), mynn_hi19 = (nchem), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { ic = mynn_lo19 + mynn_it19*mynn_st19;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7258
upak = r6_div(r6_add(r6_mul(upa(((k) + (1)), ip), dzi(k)), r6_mul(upa(k, ip), dzi(((k) + (1))))), r6_add(dzi(((k) + (1))), dzi(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7259
edmf_chem(k, ic) = r6_add(edmf_chem(k, ic), r6_div(r6_mul(upak, r6_add(r6_mul(upchem(((k) + (1)), ip, ic), dzi(k)), r6_mul(upchem(k, ip, ic), dzi(((k) + (1)))))), r6_add(dzi(((k) + (1))), dzi(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7260
} mynn_lane_sync(); ic = mynn_lo19 + mynn_n19*mynn_st19; if (mynn_n19 > 0) { const int mynn_src19 = (mynn_n19 - 1) % lane_width; upak = mynn_lane_bcast(upak, mynn_src19); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7261
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7262
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7263
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7264
if (((edmf_a1(k)) > (0.f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7265
{ const int mynn_lo20 = (1), mynn_hi20 = (nchem), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { ic = mynn_lo20 + mynn_it20*mynn_st20;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7266
edmf_chem(k, ic) = r6_div(edmf_chem(k, ic), edmf_a1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7267
} mynn_lane_sync(); ic = mynn_lo20 + mynn_n20*mynn_st20; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7268
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7269
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7270
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7274
if (env_subs) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7275
{ const int mynn_lo21 = (((kts) + (1))), mynn_hi21 = (((kte) - (1))), mynn_st21 = (1); const int mynn_n21 = mynn_trip(mynn_lo21, mynn_hi21, mynn_st21); mynn_lane_sync(); for (int mynn_it21 = lane_sub; mynn_it21 < mynn_n21; mynn_it21 += lane_width) { k = mynn_lo21 + mynn_it21*mynn_st21;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7280
envi_w(k) = r6_mul(p333, r6_add(r6_add(edmf_w1(((k) - (1))), edmf_w1(k)), edmf_w1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7281
envi_a(k) = r6_mul(r6_mul(p333, r6_add(r6_add(edmf_a1(((k) - (1))), edmf_a1(k)), edmf_a1(((k) + (1))))), adjustment);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7282
} mynn_lane_sync(); k = mynn_lo21 + mynn_n21*mynn_st21; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7284
envi_w(kts) = edmf_w1(kts);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7285
envi_a(kts) = edmf_a1(kts);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7287
envi_w(kte) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7288
envi_a(kte) = edmf_a1(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7290
envi_w(((kte) + (1))) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7291
envi_a(((kte) + (1))) = edmf_a1(kte);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7295
if (((envi_w(kts)) > (r6_div(r6_mul(0.9f, dz1(kts)), dt)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7296
sublim = r6_div(r6_div(r6_mul(0.9f, dz1(kts)), dt), envi_w(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7297
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7298
sublim = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7299
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7301
{ const int mynn_lo22 = (kts), mynn_hi22 = (kte), mynn_st22 = (1); const int mynn_n22 = mynn_trip(mynn_lo22, mynn_hi22, mynn_st22); mynn_lane_sync(); for (int mynn_it22 = lane_sub; mynn_it22 < mynn_n22; mynn_it22 += lane_width) { k = mynn_lo22 + mynn_it22*mynn_st22;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7302
temp = envi_a(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7303
envi_a(k) = r6_sub(one, temp);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7304
envi_w(k) = r6_div(r6_mul(r6_mul(r6_mul(csub, sublim), envi_w(k)), temp), r6_sub(1.f, temp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7305
} mynn_lane_sync(); k = mynn_lo22 + mynn_n22*mynn_st22; if (mynn_n22 > 0) { const int mynn_src22 = (mynn_n22 - 1) % lane_width; temp = mynn_lane_bcast(temp, mynn_src22); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7308
sub_thl1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), thl1(((kts) + (1)))), r6_mul(rho1(kts), thl1(kts)))), dzi(kts)), rhoi(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7310
sub_sqv1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), qv1(((kts) + (1)))), r6_mul(rho1(kts), qv1(kts)))), dzi(kts)), rhoi(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7312
{ const int mynn_lo23 = (((kts) + (1))), mynn_hi23 = (((kte) - (1))), mynn_st23 = (1); const int mynn_n23 = mynn_trip(mynn_lo23, mynn_hi23, mynn_st23); mynn_lane_sync(); for (int mynn_it23 = lane_sub; mynn_it23 < mynn_n23; mynn_it23 += lane_width) { k = mynn_lo23 + mynn_it23*mynn_st23;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7313
dzi(k) = r6_mul(0.5f, r6_add(dz1(k), dz1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7314
sub_thl1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), thl1(((k) + (1)))), r6_mul(rho1(k), thl1(k)))), dzi(k)), rhoi(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7316
sub_sqv1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), qv1(((k) + (1)))), r6_mul(rho1(k), qv1(k)))), dzi(k)), rhoi(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7318
} mynn_lane_sync(); k = mynn_lo23 + mynn_n23*mynn_st23; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7320
{ const int mynn_lo24 = (kts), mynn_hi24 = (((kte) - (1))), mynn_st24 = (1); const int mynn_n24 = mynn_trip(mynn_lo24, mynn_hi24, mynn_st24); mynn_lane_sync(); for (int mynn_it24 = lane_sub; mynn_it24 < mynn_n24; mynn_it24 += lane_width) { k = mynn_lo24 + mynn_it24*mynn_st24;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7321
det_thl1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_thl(k), thl1(k))), envi_a(k)), psig_w);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7322
det_sqv1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_sqv(k), qv1(k))), envi_a(k)), psig_w);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7323
det_sqc1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_sqc(k), qc1(k))), envi_a(k)), psig_w);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7324
} mynn_lane_sync(); k = mynn_lo24 + mynn_n24*mynn_st24; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7326
if (((momentum_opt) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7327
sub_u1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), u1(((kts) + (1)))), r6_mul(rho1(kts), u1(kts)))), dzi(kts)), rhoi(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7329
sub_v1(kts) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(0.5f, envi_w(kts)), envi_a(kts)), r6_sub(r6_mul(rho1(((kts) + (1))), v1(((kts) + (1)))), r6_mul(rho1(kts), v1(kts)))), dzi(kts)), rhoi(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7331
{ const int mynn_lo25 = (((kts) + (1))), mynn_hi25 = (((kte) - (1))), mynn_st25 = (1); const int mynn_n25 = mynn_trip(mynn_lo25, mynn_hi25, mynn_st25); mynn_lane_sync(); for (int mynn_it25 = lane_sub; mynn_it25 < mynn_n25; mynn_it25 += lane_width) { k = mynn_lo25 + mynn_it25*mynn_st25;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7332
sub_u1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), u1(((k) + (1)))), r6_mul(rho1(k), u1(k)))), dzi(k)), rhoi(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7334
sub_v1(k) = r6_div(r6_div(r6_mul(r6_mul(r6_mul(r6_mul(0.5f, r6_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), r6_add(envi_a(k), envi_a(((k) - (1))))), r6_sub(r6_mul(rho1(((k) + (1))), v1(((k) + (1)))), r6_mul(rho1(k), v1(k)))), dzi(k)), rhoi(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7336
} mynn_lane_sync(); k = mynn_lo25 + mynn_n25*mynn_st25; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7338
{ const int mynn_lo26 = (kts), mynn_hi26 = (((kte) - (1))), mynn_st26 = (1); const int mynn_n26 = mynn_trip(mynn_lo26, mynn_hi26, mynn_st26); mynn_lane_sync(); for (int mynn_it26 = lane_sub; mynn_it26 < mynn_n26; mynn_it26 += lane_width) { k = mynn_lo26 + mynn_it26*mynn_st26;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7339
det_u1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_u(k), u1(k))), envi_a(k)), psig_w);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7340
det_v1(k) = r6_mul(r6_mul(r6_mul(cdet, r6_sub(envm_v(k), v1(k))), envi_a(k)), psig_w);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7341
} mynn_lane_sync(); k = mynn_lo26 + mynn_n26*mynn_st26; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7342
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7343
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7347
{ const int mynn_lo27 = (kts), mynn_hi27 = (((kte) - (1))), mynn_st27 = (1); const int mynn_n27 = mynn_trip(mynn_lo27, mynn_hi27, mynn_st27); mynn_lane_sync(); for (int mynn_it27 = lane_sub; mynn_it27 < mynn_n27; mynn_it27 += lane_width) { k = mynn_lo27 + mynn_it27*mynn_st27;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7348
edmf_th1(k) = r6_add(r6_add(edmf_thl1(k), r6_mul(r6_div(xlvcp, ex1(k)), edmf_qc1(k))), r6_mul(r6_div(xlscp, ex1(k)), edmf_qi1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7349
qti(((k) + (1))) = r6_div(r6_add(r6_mul(qt1(k), dz1(((k) + (1)))), r6_mul(qt1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7350
} mynn_lane_sync(); k = mynn_lo27 + mynn_n27*mynn_st27; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7351
qti(kts) = qt1(kts);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7356
cb_check = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7357
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (2))) : k >= (((kte) - (2))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7358
if (((k) > (ktop))) { break; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7359
if (((r6_add(edmf_qc1(k), edmf_qi1(k))) > (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7360
if (((cb_check) == (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7361
mf_at_cb = r6_mul(edmf_a1(k), edmf_w1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7362
cb_check = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7363
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7365
aup = edmf_a1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7366
thp = edmf_th1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7367
qtp = edmf_qt1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7368
qcp = edmf_qc1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7369
qip = edmf_qi1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7370
dqwdz = r6_div(r6_sub(qti(k), qti(((k) + (1)))), dz1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7374
esat = f_esat_blend(tk1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7376
qsl = r6_div(r6_mul(ep_2, esat), mynn_max((float)(r6_sub(pres1(k), r6_mul(ep_3, esat))), (float)(1.e-7f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7379
xl = f_xl_blend(tk1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7380
qsat_tk = f_qsat_blend(tk1(k), pres1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7382
rsl = r6_div(r6_mul(xl, qsat_tk), r6_mul(r_v, mynn_ipow<2>(tk1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7384
cpm = r6_add(cp, r6_mul(qt1(k), cpv));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7385
a = r6_div(one, r6_add(one, r6_div(r6_mul(xl, rsl), cpm)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7386
b9 = r6_mul(a, rsl);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7388
q2p = r6_div(xlvcp, ex1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7389
pt = r6_add(thl1(k), r6_mul(r6_add(r6_mul(q2p, qcp), r6_mul(r6_div(xlscp, ex1(k)), qip)), aup));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7390
bb = r6_div(r6_mul(b9, tk1(k)), pt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7396
qww = r6_add(one, r6_mul(p608, qt1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7397
alpha = r6_mul(p608, pt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7398
beta = r6_sub(r6_div(r6_mul(pt, xl), r6_mul(tk1(k), cp)), r6_mul(1.61f, pt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7402
if (((a) > (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7403
f = mynn_min((float)(r6_div(one, a)), (float)(four));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7404
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7405
f = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7406
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7416
tauc = 1800.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7417
sigq = r6_mul(r6_mul(r6_mul(five, aup), edmf_w1(k)), mynn_max((float)(r6_mul(tauc, mynn_max((float)(zero), (float)(dqwdz)))), (float)(4e-3f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7422
sigq = mynn_max((float)(sigq), (float)(r6_mul(r6_mul(three, aup), r6_sub(qtp, qt1(k)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7427
qmq = r6_sub(qt1(k), qsat_tk);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7428
if (((qmq) > (zero))) { sigq = mynn_min((float)(r6_mul(qsat_tk, 0.015f)), (float)(sigq)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7429
q1 = r6_div(qmq, sigq);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7431
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7437
cffac = mynn_min((float)(r6_div(mynn_max((float)(r6_sub(mf_at_cb, 0.05f)), (float)(zero)), 0.05f)), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7438
cf_mf = mynn_min((float)(mynn_max((float)(r6_add(p5, r6_mul(0.36f, mynn_atan(r6_mul(1.8f, r6_add(r6_add(q1, p2), cffac)))))), (float)(0.01f))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7439
cf_mf = mynn_max((float)(cf_mf), (float)(r6_mul(1.2f, aup)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7442
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7447
cf_mf = mynn_min((float)(mynn_max((float)(r6_add(p5, r6_mul(0.36f, mynn_atan(r6_mul(1.8f, r6_add(q1, p2)))))), (float)(0.01f))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7448
cf_mf = mynn_max((float)(cf_mf), (float)(r6_mul(1.8f, aup)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7450
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7463
qcfac = r6_add(1.20f, r6_mul(1.0f, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(mf_at_cb, 0.04f)), (float)(zero)), 0.05f)), (float)(one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7464
if (((r6_sub(landsea, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7470
qc_mf = r6_mul(qcfac, r6_mul(r6_add(qcp, qip), aup));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7471
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7472
if (((r6_mul(r6_add(qcp, qip), aup)) > (5e-5f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7473
qc_mf = r6_sub(r6_mul(1.86f, r6_mul(r6_add(qcp, qip), aup)), 2.2e-5f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7474
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7475
qc_mf = r6_mul(1.20f, r6_mul(r6_add(qcp, qip), aup));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7476
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7477
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7482
wt2 = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(cf_mf, thirty)), (float)(zero)), r6_sub(hundred, thirty))), (float)(one)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7483
qc_mf_min = r6_add(r6_mul(r6_mul(r6_mul(wt2, qsat_tk), 0.01f), cf_mf), r6_mul(r6_mul(r6_mul(r6_sub(one, wt2), qsat_tk), 0.025f), cf_mf));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7484
qc_mf = mynn_max((float)(qc_mf), (float)(qc_mf_min));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7486
cf_strat = cldfra_bl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7487
qc_strat = qc_bl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7488
cldfra_bl1(k) = mynn_max((float)(cf_mf), (float)(cf_strat));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7489
pct_mf = r6_div(cf_mf, cldfra_bl1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7490
qc_bl1(k) = r6_add(r6_mul(qc_mf, pct_mf), r6_mul(r6_sub(one, pct_mf), qc_strat));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7498
q1 = mynn_max((float)(q1), (float)(r6_neg(2.25f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7503
if (((q1) >= (one))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7504
fng = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7505
} else if (((((q1) >= (r6_neg(1.7f)))) && (((q1) < (one))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7506
fng = mynn_exp(r6_neg(r6_mul(p4, r6_sub(q1, one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7507
} else if (((((q1) >= (r6_neg(2.5f)))) && (((q1) < (r6_neg(1.7f)))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7508
fng = r6_add(three, mynn_exp(r6_neg(r6_mul(3.8f, r6_add(q1, 1.7f)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7509
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7510
fng = mynn_min((float)(r6_add(23.9f, mynn_exp(r6_neg(r6_mul(1.6f, r6_add(q1, 2.5f)))))), (float)(60.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7511
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7514
vt1(k) = r6_sub(r6_sub(qww, r6_mul(r6_mul(r6_mul(r6_mul(1.5f, aup), beta), bb), fng)), one);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7515
vq1(k) = r6_sub(r6_add(alpha, r6_mul(r6_mul(r6_mul(r6_mul(1.5f, aup), beta), a), fng)), tv0);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7516
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7517
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7519
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7522
if (((ktop) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7523
maxqc = mynn_maxval(edmf_qc1.slice(MynnRange{1, ktop}));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7524
if (((maxqc) < (1.e-8f))) { maxmf = r6_neg(r6_mul(1.f, maxmf)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7525
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7530
if (((((((debug_mf) == (1))) && (((i) == (idbg))))) && (((j) == (jdbg))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7558
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7559
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7560
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7561
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7562
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7564
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7576-7632
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7598
niter = 50;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7600
diff = 1.e-6f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7602
exn = mynn_pow(r6_div(p, p1000mb), rcp);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7603
qc = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7604
for (ni=1; (1) > 0 ? ni <= (niter) : ni >= (niter); ni += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7605
t = r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7606
qs = f_qsat_blend(t, p);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7607
qcold = qc;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7608
qc = r6_add(r6_mul(p5, qc), r6_mul(p5, mynn_max((float)(zero), (float)(r6_sub(qt, qs)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7609
if (((mynn_abs(r6_sub(qc, qcold))) < (diff))) { break; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7610
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7612
t = r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7613
qs = f_qsat_blend(t, p);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7614
qc = mynn_max((float)(zero), (float)(r6_sub(qt, qs)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7617
if (((zagl) < (100.f))) { qc = zero; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7620
thv = r6_mul(r6_add(thl, r6_mul(r6_div(xlvcp, exn), qc)), r6_sub(r6_add(one, r6_mul(qt, r6_sub(rvovrd, one))), r6_mul(rvovrd, qc)));
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7635-7709
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7649
niter = 50;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7651
diff = 1.e-6f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7653
if (((r6_add(qi, qc)) > (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7654
frac_ice = r6_div(qi, r6_add(qi, qc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7655
frac_liq = r6_sub(one, frac_ice);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7656
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7657
frac_ice = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7658
frac_liq = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7659
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7661
if (((debug_dd) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7662
thvin = thv;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7663
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7664
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7665
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7667
exn = mynn_pow(r6_div(p, p1000mb), rcp);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7668
for (ni=1; (1) > 0 ? ni <= (niter) : ni >= (niter); ni += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7669
t = r6_add(r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc)), r6_mul(r6_div(xlscp, exn), qi));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7670
qs = f_qsat_blend(t, p);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7671
qxold = r6_add(qc, qi);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7672
qx = r6_add(qc, qi);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7673
qx = r6_add(r6_mul(p5, qx), r6_mul(p5, mynn_max((float)(zero), (float)(r6_sub(qt, qs)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7674
qc = r6_mul(frac_liq, qx);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7675
qi = r6_mul(frac_ice, qx);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7676
if (((mynn_abs(r6_sub(qx, qxold))) < (diff))) { break; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7677
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7679
t = r6_add(r6_add(r6_mul(exn, thl), r6_mul(r6_div(xlvcp, exn), qc)), r6_mul(r6_div(xlscp, exn), qi));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7680
qs = f_qsat_blend(t, p);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7681
qx = mynn_max((float)(zero), (float)(r6_sub(qt, qs)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7682
qc = r6_mul(frac_liq, qx);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7683
qi = r6_mul(frac_ice, qx);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7688
thv = r6_mul(r6_add(r6_add(thl, r6_mul(r6_div(xlvcp, exn), qc)), r6_mul(r6_div(xlscp, exn), qi)), r6_sub(r6_add(one, r6_mul(p608, r6_sub(qt, qx))), qx));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7703
if (((debug_dd) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7704
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7705
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7706
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7707
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7770-8316
__device__ __noinline__ void f_ddmp_mf(int kts, int kte, float dt, float dx, MynnArray<float, 1> zw, MynnArray<float, 1> dz, MynnArray<float, 1> p, MynnArray<float, 1> delp, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> th, MynnArray<float, 1> thl, MynnArray<float, 1> thv, MynnArray<float, 1> tk, MynnArray<float, 1> qt, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qnc, MynnArray<float, 1> qni, MynnArray<float, 1> qnwfa, MynnArray<float, 1> qnifa, MynnArray<float, 1> qke, MynnArray<float, 1> rho, MynnArray<float, 1> exner, MynnArray<float, 1> qc_bl1, MynnArray<float, 1> qi_bl1, MynnArray<float, 1> cldfra_bl1, float ust, float flt, float flq, float fltv, float pblh, int kpbl, MynnArray<float, 1> edmf_a_dd, MynnArray<float, 1> edmf_w_dd, MynnArray<float, 1> edmf_qt_dd, MynnArray<float, 1> edmf_thl_dd, MynnArray<float, 1> edmf_ent_dd, MynnArray<float, 1> edmf_qc_dd, MynnArray<float, 1> sd_aw, MynnArray<float, 1> sd_awthl, MynnArray<float, 1> sd_awqt, MynnArray<float, 1> sd_awqv, MynnArray<float, 1> sd_awqc, MynnArray<float, 1> sd_awqi, MynnArray<float, 1> sd_awqnc, MynnArray<float, 1> sd_awqni, MynnArray<float, 1> sd_awqnwfa, MynnArray<float, 1> sd_awqnifa, MynnArray<float, 1> sd_awu, MynnArray<float, 1> sd_awv, MynnArray<float, 1> sd_awqke, MynnArray<float, 1> tkeprod_dn, MynnArray<float, 1> el, MynnArray<float, 1> rthraten, float psig, float& maxmf_dd, float& maxwidth_dd) {
MynnFrame mynn_frame(arena);
zw.rebase({kts});
dz.rebase({kts});
p.rebase({kts});
delp.rebase({kts});
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7873
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downw).size(); mynn_element += lane_width) { downw.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7874
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downthl).size(); mynn_element += lane_width) { downthl.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7875
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downthv).size(); mynn_element += lane_width) { downthv.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7876
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqt).size(); mynn_element += lane_width) { downqt.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7877
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downa).size(); mynn_element += lane_width) { downa.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7878
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downu).size(); mynn_element += lane_width) { downu.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7879
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downv).size(); mynn_element += lane_width) { downv.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7880
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqc).size(); mynn_element += lane_width) { downqc.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7881
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqi).size(); mynn_element += lane_width) { downqi.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7882
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqnc).size(); mynn_element += lane_width) { downqnc.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7883
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqni).size(); mynn_element += lane_width) { downqni.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7884
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqnwfa).size(); mynn_element += lane_width) { downqnwfa.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7885
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downqnifa).size(); mynn_element += lane_width) { downqnifa.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7886
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (ent).size(); mynn_element += lane_width) { ent.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7887
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dd_initk).size(); mynn_element += lane_width) { dd_initk.linear(mynn_element) = 0; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7889
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_a_dd).size(); mynn_element += lane_width) { edmf_a_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7890
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_w_dd).size(); mynn_element += lane_width) { edmf_w_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7891
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qt_dd).size(); mynn_element += lane_width) { edmf_qt_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7892
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_thl_dd).size(); mynn_element += lane_width) { edmf_thl_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7893
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_ent_dd).size(); mynn_element += lane_width) { edmf_ent_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7894
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (edmf_qc_dd).size(); mynn_element += lane_width) { edmf_qc_dd.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7896
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_aw).size(); mynn_element += lane_width) { sd_aw.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7897
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awthl).size(); mynn_element += lane_width) { sd_awthl.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7898
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqt).size(); mynn_element += lane_width) { sd_awqt.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7899
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqv).size(); mynn_element += lane_width) { sd_awqv.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7900
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqc).size(); mynn_element += lane_width) { sd_awqc.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7901
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqi).size(); mynn_element += lane_width) { sd_awqi.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7902
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnc).size(); mynn_element += lane_width) { sd_awqnc.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7903
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqni).size(); mynn_element += lane_width) { sd_awqni.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7904
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnwfa).size(); mynn_element += lane_width) { sd_awqnwfa.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7905
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqnifa).size(); mynn_element += lane_width) { sd_awqnifa.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7906
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awu).size(); mynn_element += lane_width) { sd_awu.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7907
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awv).size(); mynn_element += lane_width) { sd_awv.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7908
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (sd_awqke).size(); mynn_element += lane_width) { sd_awqke.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7911
cloudflg = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7912
singlelayer = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7913
qltop = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7914
qlbase = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7915
crate = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7916
for (k=mynn_max((int)(((kpbl) - (5))), (int)(kmin)); (1) > 0 ? k <= (mynn_min((int)(((kpbl) + (20))), (int)(((kte) - (1))))) : k >= (mynn_min((int)(((kpbl) + (20))), (int)(((kte) - (1))))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7917
maxcooling = mynn_minval(rthraten.slice(MynnRange{((k) - (1)), ((k) + (1))}));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7919
if (((((((cldfra_bl1(k)) > (p5))) && (((cldfra_bl1(((k) + (1)))) < (p5))))) && (((maxcooling) < (cooling_thresh))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7922
cloudflg = true;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7923
if (((maxcooling) < (crate))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7924
qltop = k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7925
cftop = cldfra_bl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7926
crate = maxcooling;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7927
if (((cldfra_bl1(((k) - (1)))) < (p2))) { singlelayer = true; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7928
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7929
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7930
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7933
maxdd2 = mynn_min((float)(r6_mul(1.2f, dx)), (float)(maxddd));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7934
maxdd2 = mynn_min((float)(maxdd2), (float)(r6_add(r6_mul(r6_neg(r6_mul(100.f, crate)), 3600.f), 200.f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7935
maxdd2 = mynn_min((float)(maxdd2), (float)(r6_add(80.f, r6_mul(p666, zw(((qltop) + (1)))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7936
maxwidth_dd = maxdd2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7937
dl = r6_div(mynn_max((float)(zero), (float)(r6_sub(maxdd2, minddd))), ((float)(((ndd) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7940
if (((cloudflg) && (((dl) > (zero))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7941
for (k=qltop; ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7942
qx_k = mynn_max((float)(qt(k)), (float)(r6_add(qc_bl1(k), qi_bl1(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7943
if (((qx_k) > (1e-6f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7944
qlbase = k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7945
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7946
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7948
{ const int mynn_lo0 = (1), mynn_hi0 = (ndd), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { i = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7950
dd_initk(i) = qltop;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7951
} mynn_lane_sync(); i = mynn_lo0 + mynn_n0*mynn_st0; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7954
f0 = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7955
for (k=mynn_max((int)(((qltop) - (2))), (int)(kmin)); (1) > 0 ? k <= (((qltop) + (1))) : k >= (((qltop) + (1))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7956
radflux = r6_mul(rthraten(k), exner(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7958
dp = delp(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7959
radflux = r6_mul(r6_div(r6_mul(radflux, cp), grav), dp);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7960
if (((radflux) < (zero))) { f0 = r6_add(mynn_abs(radflux), f0); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7961
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7962
f0 = mynn_min((float)(mynn_max((float)(f0), (float)(one))), (float)(200.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7969
add = mynn_min((float)(r6_mul(f0, 0.003f)), (float)(0.3f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7971
wspd_pbl = mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(u(kts)), mynn_ipow<2>(v(kts)))), (float)(0.01f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7972
ac_wsp = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(wspd_pbl, 10.f)), (float)(zero)), 15.f)), (float)(one)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7973
add = r6_mul(add, mynn_min((float)(ac_wsp), (float)(psig)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7976
dz_ent = r6_mul(p5, r6_add(dz(((qltop) + (1))), dz(qltop)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7977
jump_thv = r6_mul(r6_div(r6_sub(thv(((qltop) + (1))), thv(qltop)), dz_ent), dz200);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7978
jump_qt = r6_mul(r6_div(r6_sub(qt(((qltop) + (1))), qt(qltop)), dz_ent), dz200);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7979
jump_thetal = r6_mul(r6_div(r6_sub(thl(((qltop) + (1))), thl(qltop)), dz_ent), dz200);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7981
ki = qltop;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7982
if (singlelayer) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7983
refthl = thl(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7984
refthlv = r6_mul(refthl, r6_add(1.f, r6_mul(p608, qv(ki))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7985
refthv = thv(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7986
refthvm1 = thv(((ki) - (1)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7987
reftk = tk(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7988
refqt = qt(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7989
refqc = qc(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7990
refqi = qi(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7991
refqnc = qnc(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7992
refqni = qni(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7993
refqnwfa = qnwfa(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7994
refqnifa = qnifa(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7995
refu = u(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7996
refv = v(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7997
refqke = qke(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7998
refp = p(ki);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:7999
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8000
refthl = r6_div(r6_add(r6_mul(thl(((ki) - (1))), dz(ki)), r6_mul(thl(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8001
refthlv = r6_mul(refthl, r6_add(1.f, r6_div(r6_mul(p608, r6_add(r6_mul(qv(((ki) - (1))), dz(ki)), r6_mul(qv(ki), dz(((ki) - (1)))))), r6_add(dz(ki), dz(((ki) - (1)))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8002
refthv = r6_div(r6_add(r6_mul(thv(((ki) - (1))), dz(ki)), r6_mul(thv(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8003
refthvm1 = r6_div(r6_add(r6_mul(thv(((ki) - (2))), dz(((ki) - (1)))), r6_mul(thv(((ki) - (1))), dz(((ki) - (2))))), r6_add(dz(((ki) - (1))), dz(((ki) - (2)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8004
reftk = r6_div(r6_add(r6_mul(tk(((ki) - (1))), dz(ki)), r6_mul(tk(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8005
refqt = r6_div(r6_add(r6_mul(qt(((ki) - (1))), dz(ki)), r6_mul(qt(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8006
refqc = r6_div(r6_add(r6_mul(qc(((ki) - (1))), dz(ki)), r6_mul(qc(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8007
refqi = r6_div(r6_add(r6_mul(qi(((ki) - (1))), dz(ki)), r6_mul(qi(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8008
refqnc = r6_div(r6_add(r6_mul(qnc(((ki) - (1))), dz(ki)), r6_mul(qnc(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8009
refqni = r6_div(r6_add(r6_mul(qni(((ki) - (1))), dz(ki)), r6_mul(qni(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8010
refqnwfa = r6_div(r6_add(r6_mul(qnwfa(((ki) - (1))), dz(ki)), r6_mul(qnwfa(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8011
refqnifa = r6_div(r6_add(r6_mul(qnifa(((ki) - (1))), dz(ki)), r6_mul(qnifa(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8012
refu = r6_div(r6_add(r6_mul(u(((ki) - (1))), dz(ki)), r6_mul(u(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8013
refv = r6_div(r6_add(r6_mul(v(((ki) - (1))), dz(ki)), r6_mul(v(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8014
refqke = r6_div(r6_add(r6_mul(qke(((ki) - (1))), dz(ki)), r6_mul(qke(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8015
refp = r6_div(r6_add(r6_mul(p(((ki) - (1))), dz(ki)), r6_mul(p(ki), dz(((ki) - (1))))), r6_add(dz(ki), dz(((ki) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8016
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8020
wstar_rad = r6_mul(1.25f, mynn_pow(r6_div(r6_mul(r6_mul(grav, dz200), f0), r6_mul(r6_mul(refthl, rho(ki)), cp)), p333));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8021
wstar_rad = mynn_min((float)(mynn_max((float)(wstar_rad), (float)(p1))), (float)(1.5f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8024
dthvx = r6_sub(r6_add(thl(((ki) + (2))), r6_mul(r6_mul(th(((ki) + (2))), p608), qt(((ki) + (2))))), r6_add(thl(ki), r6_mul(r6_mul(th(ki), p608), qt(ki))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8026
dthvx = r6_div(r6_mul(dthvx, dz200), dz_ent);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8027
dthvx = mynn_max((float)(dthvx), (float)(p1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8028
rcldb = mynn_max((float)(r6_add(refqc, refqi)), (float)(r6_add(qc_bl1(ki), qi_bl1(ki))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8029
tmp1 = r6_div(r6_mul(xlvcp, rcldb), r6_mul(exner(ki), dthvx));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8032
ent_eff = mynn_max((float)(zero), (float)(mynn_min((float)(r6_add(p2, r6_mul(r6_mul(p2, eight), tmp1))), (float)(one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8034
went = r6_mul(r6_mul(p333, ent_eff), wstar_rad);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8035
qstar = r6_div(r6_mul(went, jump_qt), wstar_rad);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8036
thstar = r6_sub(r6_div(r6_div(r6_neg(r6_div(f0, rho(ki))), cp), wstar_rad), r6_div(r6_mul(went, jump_thv), wstar_rad));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8038
sigmaw = r6_mul(0.2f, wstar_rad);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8039
sigmaqt = r6_mul(50.f, mynn_abs(qstar));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8040
sigmath = r6_mul(1.5f, mynn_abs(thstar));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8042
pwmin = r6_neg(1.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8043
pwmax = r6_neg(3.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8044
wmin = mynn_max((float)(mynn_min((float)(r6_mul(sigmaw, pwmin)), (float)(r6_neg(0.1f)))), (float)(r6_neg(0.2f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8045
wmax = mynn_max((float)(mynn_min((float)(r6_mul(sigmaw, pwmax)), (float)(r6_neg(0.2f)))), (float)(r6_neg(0.5f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8048
minexc = mynn_min((float)(r6_neg(0.06f)), (float)(r6_mul(0.5f, r6_sub(refthvm1, refthv))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8049
maxexc = mynn_min((float)(r6_sub(minexc, 0.1f)), (float)(r6_neg(0.6f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8050
maxexc = mynn_max((float)(maxexc), (float)(r6_sub(thv(1), refthlv)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8051
def_th = mynn_max((float)(mynn_min((float)(r6_div(r6_mul(r6_mul(0.05f, r6_neg(0.3f)), sigmath), sigmaw)), (float)(minexc))), (float)(maxexc));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8054
qs1 = f_qsat_blend(reftk, refp);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8055
qs2 = f_qsat_blend(r6_add(reftk, def_th), refp);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8056
refqt2 = r6_div(r6_mul(refqt, qs2), qs1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8058
def_qt = mynn_min((float)(r6_sub(qs2, qs1)), (float)(r6_neg(r6_mul(0.33f, r6_add(refqc, refqi)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8061
if (((r6_add(refqc, refqi)) > (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8062
frac_liq = r6_div(refqc, r6_add(refqc, refqi));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8063
frac_ice = r6_sub(one, frac_liq);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8064
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8065
frac_liq = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8066
frac_ice = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8067
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8069
if (((debug_dd) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8070
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8071
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8072
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8073
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8074
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8075
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8076
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8077
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8078
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8079
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8083
{ const int mynn_lo1 = (1), mynn_hi1 = (ndd), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { i = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8084
ki = dd_initk(i);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8087
downw(ki, i) = r6_sub(wmin, r6_mul(r6_div(r6_sub(mynn_abs(wmax), mynn_abs(wmin)), ((float)(((ndd) - (1))))), ((i) - (1))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8091
downa(ki, i) = r6_div(add, ((float)(ndd)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8092
downu(ki, i) = refu;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8093
downv(ki, i) = refv;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8094
downqc(ki, i) = mynn_max((float)(zero), (float)(r6_add(refqc, r6_mul(def_qt, frac_liq))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8095
downqi(ki, i) = mynn_max((float)(zero), (float)(r6_add(refqi, r6_mul(def_qt, frac_ice))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8096
downqnc(ki, i) = refqnc;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8097
downqni(ki, i) = refqni;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8098
downqnwfa(ki, i) = refqnwfa;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8099
downqnifa(ki, i) = refqnifa;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8100
downqt(ki, i) = r6_add(refqt, def_qt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8101
downthv(ki, i) = r6_add(refthv, r6_div(r6_mul(def_th, ((float)(i))), ((float)(ndd))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8102
downthl(ki, i) = r6_add(refthl, r6_div(r6_mul(def_th, ((float)(i))), ((float)(ndd))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8109
} mynn_lane_sync(); i = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; ki = mynn_lane_bcast(ki, mynn_src1); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8111
if (((debug_dd) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8112
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8113
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8114
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8115
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8116
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8117
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8118
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8120
for (i=1; (1) > 0 ? i <= (ndd) : i >= (ndd); i += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8121
ddd = r6_add(minddd, r6_mul(dl, ((float)(((i) - (1))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8123
thvn = downthv(ki, i);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8124
qcn = downqc(ki, i);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8125
qin = downqi(ki, i);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8126
for (k=((dd_initk(i)) - (1)); ((-(1))) > 0 ? k <= (((kts) + (1))) : k >= (((kts) + (1))); k += ((-(1)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8128
wmin = r6_add(p1, r6_mul(ddd, 0.0005f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8132
ent(k, i) = r6_div(0.18f, r6_mul(mynn_min((float)(mynn_max((float)(mynn_abs(downw(((k) + (1)), i))), (float)(wmin))), (float)(0.6f)), ddd));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8135
ent(k, i) = mynn_max((float)(ent(k, i)), (float)(0.0003f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8136
ent(k, i) = mynn_max((float)(ent(k, i)), (float)(r6_div(0.06f, zw(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8137
ent(k, i) = mynn_min((float)(ent(k, i)), (float)(r6_div(0.9f, dz(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8142
entexp = r6_mul(ent(k, i), dz(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8143
entexm = r6_mul(r6_mul(ent(k, i), dz(k)), p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8145
qtn = r6_add(r6_mul(downqt(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qt(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8146
thln = r6_add(r6_mul(downthl(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(thl(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8147
qncn = r6_add(r6_mul(downqnc(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qnc(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8148
qnin = r6_add(r6_mul(downqni(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qni(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8149
qnwfan = r6_add(r6_mul(downqnwfa(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qnwfa(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8150
qnifan = r6_add(r6_mul(downqnifa(((k) + (1)), i), r6_sub(one, entexp)), r6_mul(qnifa(k), entexp));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8151
un = r6_add(r6_mul(downu(((k) + (1)), i), r6_sub(one, entexm)), r6_mul(u(k), entexm));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8152
vn = r6_add(r6_mul(downv(((k) + (1)), i), r6_sub(one, entexm)), r6_mul(v(k), entexm));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8160
pk = r6_div(r6_add(r6_mul(p(((k) - (1))), dz(k)), r6_mul(p(k), dz(((k) - (1))))), r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8161
f_condensation_ddmf(qtn, thln, pk, zw(k), thvn, qcn, qin, debug_dd);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8164
thvk = r6_div(r6_add(r6_mul(thv(((k) - (1))), dz(k)), r6_mul(thv(k), dz(((k) - (1))))), r6_add(dz(k), dz(((k) - (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8165
buoy = r6_mul(grav, r6_sub(r6_div(thvn, thvk), one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8167
if (((buoy) > (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8168
bcoeff = 0.2f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8169
ecoeff = r6_neg(2.0f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8170
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8172
bcoeff = r6_mul(0.2f, r6_sub(one, mynn_exp(r6_neg(r6_div(zw(k), r6_mul(0.15f, pblh))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8173
ecoeff = r6_sub(r6_neg(2.0f), mynn_exp(r6_neg(r6_div(zw(k), r6_mul(0.15f, pblh)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8174
buoy = r6_mul(buoy, r6_sub(one, mynn_exp(r6_neg(r6_div(zw(k), r6_mul(0.15f, pblh))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8175
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8176
if (((k) == (((kts) + (2))))) { buoy = mynn_max((float)(buoy), (float)(0.01f)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8177
if (((k) == (((kts) + (1))))) { buoy = mynn_max((float)(buoy), (float)(0.05f)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8189
mindownw = mynn_min((float)(downw(((k) + (1)), i)), (float)(r6_neg(0.2f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8190
wn = r6_add(downw(((k) + (1)), i), r6_mul(r6_sub(r6_mul(r6_mul(ecoeff, ent(k, i)), downw(((k) + (1)), i)), r6_div(r6_mul(bcoeff, buoy), mindownw)), mynn_min((float)(dz(k)), (float)(250.f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8195
if (((wn) < (r6_sub(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8196
wn = r6_sub(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8197
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8199
if (((wn) > (r6_add(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8200
wn = r6_add(downw(((k) + (1)), i), mynn_min((float)(r6_div(r6_mul(1.25f, dz(k)), 200.f)), (float)(2.0f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8201
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8202
wn = mynn_max((float)(mynn_min((float)(wn), (float)(zero))), (float)(r6_neg(2.0f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8204
if (((debug_dd) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8205
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8206
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8207
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8208
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8209
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8210
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8211
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8212
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8213
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8215
if (((wn) < (0.f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8216
downw(k, i) = wn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8217
downthv(k, i) = thvn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8218
downthl(k, i) = thln;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8219
downqt(k, i) = qtn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8220
downqc(k, i) = qcn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8221
downqi(k, i) = qin;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8222
downqnc(k, i) = qncn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8223
downqni(k, i) = qnin;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8224
downqnwfa(k, i) = qnwfan;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8225
downqnifa(k, i) = qnifan;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8226
downu(k, i) = un;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8227
downv(k, i) = vn;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8228
downa(k, i) = downa(((k) + (1)), i);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8229
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8240
break;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8241
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8242
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8243
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8245
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downw.slice(1, MynnRange{downw.lo[1], downw.hi[1]})).size(); mynn_element += lane_width) { downw.slice(1, MynnRange{downw.lo[1], downw.hi[1]}).linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8246
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (downa.slice(1, MynnRange{downa.lo[1], downa.hi[1]})).size(); mynn_element += lane_width) { downa.slice(1, MynnRange{downa.lo[1], downa.hi[1]}).linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8251
{ const int mynn_lo2 = (qltop), mynn_hi2 = (kts), mynn_st2 = ((-(1))); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8252
for (i=1; (1) > 0 ? i <= (ndd) : i >= (ndd); i += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8253
edmf_a_dd(k) = r6_add(edmf_a_dd(k), downa(k, i));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8254
edmf_w_dd(k) = r6_add(edmf_w_dd(k), r6_mul(downa(k, i), downw(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8255
edmf_qt_dd(k) = r6_add(edmf_qt_dd(k), r6_mul(downa(k, i), downqt(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8256
edmf_thl_dd(k) = r6_add(edmf_thl_dd(k), r6_mul(downa(k, i), downthl(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8257
edmf_ent_dd(k) = r6_add(edmf_ent_dd(k), r6_mul(downa(k, i), ent(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8258
edmf_qc_dd(k) = r6_add(edmf_qc_dd(k), r6_mul(downa(k, i), downqc(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8259
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8261
if (((edmf_a_dd(k)) > (0.f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8262
edmf_w_dd(k) = r6_div(edmf_w_dd(k), edmf_a_dd(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8263
edmf_qt_dd(k) = r6_div(edmf_qt_dd(k), edmf_a_dd(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8264
edmf_thl_dd(k) = r6_div(edmf_thl_dd(k), edmf_a_dd(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8265
edmf_ent_dd(k) = r6_div(edmf_ent_dd(k), edmf_a_dd(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8266
edmf_qc_dd(k) = r6_div(edmf_qc_dd(k), edmf_a_dd(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8267
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8269
tkeprod_dn(k) = r6_div(r6_mul(mynn_ipow<3>(mynn_abs(edmf_w_dd(k))), edmf_a_dd(k)), r6_mul(b1, mynn_max((float)(el(k)), (float)(p5))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8270
tkeprod_dn(k) = mynn_min((float)(tkeprod_dn(k)), (float)(0.0004f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8271
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; i = mynn_lane_bcast(i, mynn_src2); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8274
tkeprod_dn(((qltop) + (1))) = r6_div(r6_mul(mynn_ipow<3>(went), edmf_a_dd(qltop)), r6_mul(b1, mynn_max((float)(el(((qltop) + (1)))), (float)(p5))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8275
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (massflux).size(); mynn_element += lane_width) { massflux.linear(mynn_element) = r6_mul(edmf_a_dd.linear(mynn_element), edmf_w_dd.linear(mynn_element)); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8276
maxmf_dd = mynn_abs(mynn_minval(massflux));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8278
if (((debug_dd) == (2))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8279
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8280
if (((((tkeprod_dn(k)) > (0.0004f))) || (((edmf_w_dd(k)) < (r6_neg(two)))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8281
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8282
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8283
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8284
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8285
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8286
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8287
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8288
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8289
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8290
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8291
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8292
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8293
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8296
{ const int mynn_lo3 = (kts), mynn_hi3 = (qltop), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8297
rho_int = r6_div(r6_add(r6_mul(rho(k), dz(((k) + (1)))), r6_mul(rho(((k) + (1))), dz(k))), r6_add(dz(((k) + (1))), dz(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8298
for (i=1; (1) > 0 ? i <= (ndd) : i >= (ndd); i += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8299
sd_aw(k) = r6_add(sd_aw(k), r6_mul(r6_mul(rho_int, downa(k, i)), mynn_abs(downw(k, i))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8300
sd_awthl(k) = r6_add(sd_awthl(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downthl(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8301
sd_awqt(k) = r6_add(sd_awqt(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqt(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8302
sd_awqc(k) = r6_add(sd_awqc(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqc(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8303
sd_awqi(k) = r6_add(sd_awqi(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqi(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8304
sd_awqnc(k) = r6_add(sd_awqnc(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqnc(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8305
sd_awqni(k) = r6_add(sd_awqni(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqni(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8306
sd_awqnwfa(k) = r6_add(sd_awqnwfa(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqnwfa(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8307
sd_awqnifa(k) = r6_add(sd_awqnifa(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downqnifa(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8308
sd_awu(k) = r6_add(sd_awu(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downu(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8309
sd_awv(k) = r6_add(sd_awv(k), r6_mul(r6_mul(r6_mul(rho_int, downa(k, i)), downw(k, i)), downv(k, i)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8310
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8311
sd_awqv(k) = r6_sub(r6_sub(sd_awqt(k), sd_awqc(k)), sd_awqi(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8312
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; i = mynn_lane_bcast(i, mynn_src3); rho_int = mynn_lane_bcast(rho_int, mynn_src3); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8314
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8319-8394
__device__ __noinline__ void f_scale_aware(float dx, float pblh, float& psig_bl, float& psig_shcu) {
MynnFrame mynn_frame(arena);
float dxdh;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8336
psig_bl = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8337
psig_shcu = one;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8338
dxdh = r6_div(mynn_max((float)(r6_mul(2.5f, dx)), (float)(ten)), mynn_min((float)(pblh), (float)(3000.f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8346
psig_bl = r6_div(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.106f, mynn_pow(dxdh, 0.667f))), r6_add(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.066f, mynn_pow(dxdh, 0.667f))), 0.071f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8350
dxdh = r6_div(mynn_max((float)(r6_mul(2.5f, dx)), (float)(ten)), mynn_min((float)(r6_add(pblh, 500.f)), (float)(3500.f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8377
psig_shcu = r6_div(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.145f, mynn_pow(dxdh, 0.667f))), r6_add(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.172f, mynn_pow(dxdh, 0.667f))), 0.170f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8388
if (((psig_bl) > (one))) { psig_bl = one; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8389
if (((psig_bl) < (zero))) { psig_bl = zero; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8391
if (((psig_shcu) > (one))) { psig_shcu = one; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8392
if (((psig_shcu) < (zero))) { psig_shcu = zero; }
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8404-8447
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8431
xc = mynn_max((float)(r6_neg(80.f)), (float)(r6_sub(t, t0c)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8436
if (((t) >= (r6_sub(t0c, 6.f)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8437
esat_blend = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8438
} else if (((t) <= (tice))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8439
esat_blend = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8440
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8441
esl = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8442
esi = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8443
chi = r6_div(r6_sub(r6_sub(t0c, 6.f), t), r6_sub(r6_sub(t0c, 6.f), tice));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8444
esat_blend = r6_add(r6_mul(r6_sub(one, chi), esl), r6_mul(chi, esi));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8445
}
return esat_blend;
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8455-8504
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8482
xc = mynn_max((float)(r6_neg(80.f)), (float)(r6_sub(t, t0c)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8484
if (((t) >= (r6_sub(t0c, 6.f)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8485
esl = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8486
esl = mynn_min((float)(esl), (float)(r6_mul(p, 0.15f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8487
qsat_blend = r6_div(r6_mul(0.622f, esl), mynn_max((float)(r6_sub(p, esl)), (float)(1e-5f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8488
} else if (((t) <= (tice))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8489
esi = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8490
esi = mynn_min((float)(esi), (float)(r6_mul(p, 0.15f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8491
qsat_blend = r6_div(r6_mul(0.622f, esi), mynn_max((float)(r6_sub(p, esi)), (float)(1e-5f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8492
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8493
esl = r6_add(j0, r6_mul(xc, r6_add(j1, r6_mul(xc, r6_add(j2, r6_mul(xc, r6_add(j3, r6_mul(xc, r6_add(j4, r6_mul(xc, r6_add(j5, r6_mul(xc, r6_add(j6, r6_mul(xc, r6_add(j7, r6_mul(xc, j8))))))))))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8494
esl = mynn_min((float)(esl), (float)(r6_mul(p, 0.15f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8495
esi = r6_add(k0, r6_mul(xc, r6_add(k1, r6_mul(xc, r6_add(k2, r6_mul(xc, r6_add(k3, r6_mul(xc, r6_add(k4, r6_mul(xc, r6_add(k5, r6_mul(xc, r6_add(k6, r6_mul(xc, r6_add(k7, r6_mul(xc, k8))))))))))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8496
esi = mynn_min((float)(esi), (float)(r6_mul(p, 0.15f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8497
rslf = r6_div(r6_mul(0.622f, esl), mynn_max((float)(r6_sub(p, esl)), (float)(1e-5f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8498
rsif = r6_div(r6_mul(0.622f, esi), mynn_max((float)(r6_sub(p, esi)), (float)(1e-5f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8500
chi = r6_div(r6_sub(r6_sub(t0c, 6.f), t), r6_sub(r6_sub(t0c, 6.f), tice));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8501
qsat_blend = r6_add(r6_mul(r6_sub(one, chi), rslf), r6_mul(chi, rsif));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8502
}
return qsat_blend;
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8513-8534
__device__ __noinline__ float f_xl_blend(float t) {
MynnFrame mynn_frame(arena);
float xl_blend;
float xlvt;
float xlst;
float chi;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8522
if (((t) >= (t0c))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8523
xl_blend = r6_add(xlv, r6_mul(r6_sub(cpv, cliq), r6_sub(t, t0c)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8524
} else if (((t) <= (tice))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8525
xl_blend = r6_add(xls, r6_mul(r6_sub(cpv, cice), r6_sub(t, t0c)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8526
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8527
xlvt = r6_add(xlv, r6_mul(r6_sub(cpv, cliq), r6_sub(t, t0c)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8528
xlst = r6_add(xls, r6_mul(r6_sub(cpv, cice), r6_sub(t, t0c)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8530
chi = r6_div(r6_sub(t0c, t), r6_sub(t0c, tice));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8531
xl_blend = r6_add(r6_mul(r6_sub(one, chi), xlvt), r6_mul(chi, xlst));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8532
}
return xl_blend;
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8538-8587
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8554
if (((zet) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8555
dummy_0 = r6_add(one, mynn_pow(zet, bm_st));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8556
dummy_1 = r6_add(zet, mynn_pow(dummy_0, rbm_st));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8557
dummy_11 = r6_add(one, r6_mul(mynn_pow(dummy_0, r6_sub(rbm_st, 1)), mynn_pow(zet, r6_sub(bm_st, one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8558
dummy_2 = r6_mul(r6_neg(r6_div(am_st, dummy_1)), dummy_11);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8559
phi_m = r6_sub(one, r6_mul(zet, dummy_2));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8560
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8561
dummy_0 = mynn_pow(r6_sub(one, r6_mul(cphm_unst, zet)), 0.25f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8562
phi_m = r6_div(one, dummy_0);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8563
dummy_psi = r6_add(r6_sub(r6_add(r6_mul(two, mynn_log(r6_mul(p5, r6_add(one, dummy_0)))), mynn_log(r6_mul(p5, r6_add(one, mynn_ipow<2>(dummy_0))))), r6_mul(two, mynn_atan(dummy_0))), 1.570796f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8565
dummy_0 = r6_sub(one, r6_mul(am_unst, zet));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8566
dummy_1 = mynn_pow(dummy_0, p333);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8567
dummy_11 = r6_mul(r6_neg(r6_mul(p333, am_unst)), mynn_pow(dummy_0, r6_neg(p666)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8568
dummy_2 = r6_mul(p333, r6_add(r6_add(mynn_ipow<2>(dummy_1), dummy_1), one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8569
dummy_22 = r6_mul(r6_mul(p333, dummy_11), r6_add(r6_mul(two, dummy_1), one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8570
dummy_3 = r6_mul(0.57735f, r6_add(r6_mul(two, dummy_1), one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8571
dummy_33 = r6_mul(1.1547f, dummy_11);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8572
dummy_4 = r6_add(r6_sub(r6_mul(1.5f, mynn_log(dummy_2)), r6_mul(1.73205f, mynn_atan(dummy_3))), 1.813799364f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8573
dummy_44 = r6_sub(r6_mul(r6_div(1.5f, dummy_2), dummy_22), r6_div(r6_mul(1.73205f, dummy_33), r6_add(one, mynn_ipow<2>(dummy_3))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8575
dummy_0 = mynn_ipow<2>(zet);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8576
dummy_1 = r6_div(one, r6_add(one, dummy_0));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8577
dummy_11 = r6_mul(two, zet);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8578
dummy_2 = r6_mul(r6_add(r6_add(r6_div(r6_sub(one, phi_m), zet), r6_mul(dummy_11, dummy_4)), r6_mul(dummy_0, dummy_44)), dummy_1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8579
dummy_22 = r6_mul(r6_neg(r6_mul(dummy_11, r6_add(dummy_psi, r6_mul(dummy_0, dummy_4)))), mynn_ipow<2>(dummy_1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8581
phi_m = r6_sub(one, r6_mul(zet, r6_add(dummy_2, dummy_22)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8582
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8585
phim = phi_m;
return phim;
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8590-8636
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8606
if (((zet) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8607
dummy_0 = r6_add(one, mynn_pow(zet, bh_st));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8608
dummy_1 = r6_add(zet, mynn_pow(dummy_0, rbh_st));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8609
dummy_11 = r6_add(one, r6_mul(mynn_pow(dummy_0, r6_sub(rbh_st, one)), mynn_pow(zet, r6_sub(bh_st, one))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8610
dummy_2 = r6_mul(r6_neg(r6_div(ah_st, dummy_1)), dummy_11);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8611
phih = r6_sub(one, r6_mul(zet, dummy_2));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8612
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8613
dummy_0 = mynn_pow(r6_sub(one, r6_mul(cphh_unst, zet)), p5);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8614
phh = r6_div(one, dummy_0);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8615
dummy_psi = r6_mul(two, mynn_log(r6_mul(p5, r6_add(one, dummy_0))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8617
dummy_0 = r6_sub(one, r6_mul(ah_unst, zet));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8618
dummy_1 = mynn_pow(dummy_0, p333);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8619
dummy_11 = r6_mul(r6_neg(r6_mul(p333, ah_unst)), mynn_pow(dummy_0, r6_neg(p666)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8620
dummy_2 = r6_mul(p333, r6_add(r6_add(mynn_ipow<2>(dummy_1), dummy_1), one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8621
dummy_22 = r6_mul(r6_mul(p333, dummy_11), r6_add(r6_mul(two, dummy_1), one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8622
dummy_3 = r6_mul(0.57735f, r6_add(r6_mul(two, dummy_1), one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8623
dummy_33 = r6_mul(1.1547f, dummy_11);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8624
dummy_4 = r6_add(r6_sub(r6_mul(1.5f, mynn_log(dummy_2)), r6_mul(1.73205f, mynn_atan(dummy_3))), 1.813799364f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8625
dummy_44 = r6_sub(r6_mul(r6_div(1.5f, dummy_2), dummy_22), r6_div(r6_mul(1.73205f, dummy_33), r6_add(one, mynn_ipow<2>(dummy_3))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8627
dummy_0 = mynn_ipow<2>(zet);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8628
dummy_1 = r6_div(one, r6_add(one, dummy_0));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8629
dummy_11 = r6_mul(two, zet);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8630
dummy_2 = r6_mul(r6_add(r6_add(r6_div(r6_sub(one, phh), zet), r6_mul(dummy_11, dummy_4)), r6_mul(dummy_0, dummy_44)), dummy_1);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8631
dummy_22 = r6_mul(r6_neg(r6_mul(dummy_11, r6_add(dummy_psi, r6_mul(dummy_0, dummy_4)))), mynn_ipow<2>(dummy_1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8633
phih = r6_sub(one, r6_mul(zet, r6_add(dummy_2, dummy_22)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8634
}
return phih;
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8638-8803
__device__ __noinline__ void f_topdown_cloudrad(int kts, int kte, MynnArray<float, 1> dz1, MynnArray<float, 1> zw, MynnArray<float, 1> zagl1, float fltv, float u1, float v1, float xland, int kpbl, float pblh, MynnArray<float, 1> sqc, MynnArray<float, 1> sqi, MynnArray<float, 1> sqw, MynnArray<float, 1> thl, MynnArray<float, 1> th1, MynnArray<float, 1> ex1, MynnArray<float, 1> pres1, MynnArray<float, 1> rho1, MynnArray<float, 1> thv, MynnArray<float, 1> cldfra_bl1, MynnArray<float, 1> qc_bl, MynnArray<float, 1> qi_bl, MynnArray<float, 1> rthraten, MynnArray<float, 1> tkeprod_dn, float psig, float& maxtkeprod, float& cldtop_cooling, float& ent_eff) {
MynnFrame mynn_frame(arena);
dz1.rebase({kts});
zw.rebase({kts});
zagl1.rebase({kts});
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8680
cloudflg = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8681
minrad = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8682
kminrad = kpbl;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8683
zminrad = pblh;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8685
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (tkeprodorig.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { tkeprodorig.slice(MynnRange{kts, kte}).linear(mynn_element) = tkeprod_dn.slice(MynnRange{kts, kte}).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8688
for (k=1; (1) > 0 ? k <= (mynn_min((int)(((kpbl) + (10))), (int)(((kte) - (1))))) : k >= (mynn_min((int)(((kpbl) + (10))), (int)(((kte) - (1))))); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8689
maxcooling = mynn_minval(rthraten.slice(MynnRange{mynn_max((int)(kts), (int)(((k) - (1)))), ((k) + (1))}));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8691
if (((((((cldfra_bl1(k)) > (p5))) && (((cldfra_bl1(((k) + (1)))) < (p5))))) && (((maxcooling) < (cooling_thresh))))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8694
cloudflg = true;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8695
if (((maxcooling) < (minrad))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8696
minrad = maxcooling;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8697
kminrad = k;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8698
zminrad = r6_add(zw(k), r6_mul(p5, dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8699
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8700
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8701
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8703
if (cloudflg) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8704
zl1 = r6_mul(p25, dz1(kts));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8705
k = kminrad;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8707
templ = r6_mul(thl(k), ex1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8709
rvls = r6_mul(r6_mul(r6_mul(100.f, 6.112f), mynn_exp(r6_div(r6_mul(17.67f, r6_sub(templ, 273.16f)), r6_sub(templ, 29.65f)))), r6_div(ep_2, pres1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8710
temps = r6_add(templ, r6_div(r6_sub(sqw(k), rvls), r6_add(r6_div(cp, xlv), r6_div(r6_mul(r6_mul(ep_2, xlv), rvls), r6_mul(r_d, mynn_ipow<2>(templ))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8711
rvls = r6_mul(r6_mul(r6_mul(100.f, 6.112f), mynn_exp(r6_div(r6_mul(17.67f, r6_sub(temps, 273.15f)), r6_sub(temps, 29.65f)))), r6_div(ep_2, pres1(((k) + (1)))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8712
rcldb = mynn_max((float)(zero), (float)(r6_sub(sqw(k), rvls)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8713
rcldb = mynn_max((float)(rcldb), (float)(r6_add(qc_bl(k), qi_bl(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8716
dthvx = r6_sub(r6_add(thl(((k) + (2))), r6_mul(r6_mul(th1(((k) + (2))), p608), sqw(((k) + (2))))), r6_add(thl(k), r6_mul(r6_mul(th1(k), p608), sqw(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8719
dz_ent = r6_add(r6_div(r6_add(r6_mul(dz1(k), dz1(((k) + (1)))), r6_mul(dz1(((k) + (1))), dz1(k))), r6_add(dz1(((k) + (1))), dz1(k))), r6_div(r6_add(r6_mul(dz1(((k) + (1))), dz1(((k) + (2)))), r6_mul(dz1(((k) + (2))), dz1(((k) + (1))))), r6_add(dz1(((k) + (1))), dz1(((k) + (2))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8721
dthvx = r6_div(r6_mul(dthvx, dz200), dz_ent);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8722
dthvx = mynn_max((float)(dthvx), (float)(p1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8723
tmp1 = r6_div(r6_mul(xlvcp, rcldb), r6_mul(ex1(k), dthvx));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8726
ent_eff = mynn_min((float)(r6_add(p2, r6_mul(r6_mul(p2, eight), tmp1))), (float)(two));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8728
f0 = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8729
for (kk=mynn_max((int)(kts), (int)(((kminrad) - (1)))); (1) > 0 ? kk <= (((kminrad) + (1))) : kk >= (((kminrad) + (1))); kk += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8730
radflux = r6_mul(rthraten(kk), ex1(kk));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8731
dp = r6_mul(r6_mul(rho1(k), grav), dz1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8732
radflux = r6_mul(r6_div(r6_mul(radflux, cp), grav), dp);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8733
if (((radflux) < (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8734
f0 = r6_add(mynn_abs(radflux), f0);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8735
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8736
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8739
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8740
f0 = mynn_min((float)(f0), (float)(150.0f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8741
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8742
f0 = mynn_min((float)(r6_mul(p25, f0)), (float)(30.f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8743
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8744
cldtop_cooling = r6_mul(mynn_abs(minrad), 3600.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8747
wstar_rad = r6_mul(1.25f, mynn_pow(r6_div(r6_mul(r6_mul(r6_div(grav, thv(k)), dz200), f0), r6_mul(rho1(k), cp)), p333));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8748
wstar_rad = mynn_min((float)(mynn_max((float)(wstar_rad), (float)(p1))), (float)(three));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8749
dthvx = mynn_max((float)(r6_sub(thv(((k) + (1))), thv(k))), (float)(p1));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8750
went = r6_mul(r6_mul(p333, ent_eff), wstar_rad);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8754
zb1 = mynn_max((float)(zl1), (float)(r6_sub(zminrad, r6_mul(wstar_rad, 800.f))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8755
{ const int mynn_lo0 = (kts), mynn_hi0 = (((kminrad) + (3))), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8757
zagl = zagl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8758
if (((((k) == (kts))) && (((kminrad) == (kts))))) { zagl = r6_mul(0.4f, dz1(kts)); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8760
zfac(k) = mynn_min((float)(mynn_max((float)(r6_sub(one, r6_div(mynn_max((float)(r6_sub(zagl, zb1)), (float)(zero)), r6_sub(zminrad, zl1)))), (float)(zfmin))), (float)(one));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8761
zfacent(k) = r6_mul(mynn_max((float)(zero), (float)(r6_div(r6_sub(zminrad, zagl), zminrad))), mynn_ipow<3>(r6_sub(one, zfac(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8762
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; zagl = mynn_lane_bcast(zagl, mynn_src0); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8763
zfacent_max = mynn_maxval(zfacent.slice(MynnRange{kts, ((kminrad) + (3))}));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8764
for (int mynn_element=0; mynn_element < (zfacent).size(); ++mynn_element) { zfacent.linear(mynn_element) = r6_div(zfacent.linear(mynn_element), mynn_max((float)(zfacent_max), (float)(1e-5f))); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8767
{ const int mynn_lo1 = (kts), mynn_hi1 = (((kminrad) + (3))), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8768
tkeprod_dn(k) = mynn_max((float)(tkeprod_dn(k)), (float)(r6_mul(r6_div(r6_mul(r6_mul(p3, ent_eff), mynn_ipow<3>(wstar_rad)), dz500), zfacent(k))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8769
tkeprod_dn(k) = mynn_max((float)(zero), (float)(tkeprod_dn(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8770
tkeprod_dn(k) = mynn_min((float)(tkeprod_dn(k)), (float)(0.0004f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8771
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8773
tkeprod_dn(kminrad) = mynn_max((float)(tkeprod_dn(kminrad)), (float)(r6_mul(p333, tkeprod_dn(mynn_max((int)(kts), (int)(((kminrad) - (1))))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8775
tkeprod_dn(((kminrad) + (1))) = mynn_max((float)(tkeprod_dn(((kminrad) + (1)))), (float)(r6_mul(mynn_min((float)(ent_eff), (float)(p5)), tkeprod_dn(mynn_max((int)(kts), (int)(kminrad))))));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8778
wspd_pbl = mynn_sqrt(mynn_max((float)(r6_add(mynn_ipow<2>(u1), mynn_ipow<2>(v1))), (float)(0.01f)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8779
ac_wsp = r6_sub(one, mynn_min((float)(r6_div(mynn_max((float)(r6_sub(wspd_pbl, 10.f)), (float)(zero)), 15.f)), (float)(one)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8780
for (int mynn_element=0; mynn_element < (tkeprod_dn).size(); ++mynn_element) { tkeprod_dn.linear(mynn_element) = r6_mul(tkeprod_dn.linear(mynn_element), mynn_min((float)(ac_wsp), (float)(psig))); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8781
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8784
maxtkeprod = r6_mul(mynn_maxval(tkeprod_dn), 3600.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8786
if (((cloudflg) && (debug))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8788
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8789
if (((r6_sub(xland, 1.5f)) >= (zero))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8790
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8791
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8792
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8793
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8794
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8795
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8796
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8797
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8798
for (k=((kminrad) + (3)); ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8799
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8800
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf.F90:8801
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:82-974
__device__ __forceinline__ void f_mynnedmf_driver(int ids, int ide, int jds, int jde, int kds, int kde, int ims, int ime, int jms, int jme, int kms, int kme, int its, int ite, int jts, int jte, int kts, int kte, bool f_qc, bool f_qi, bool f_qs, bool f_qoz, bool f_qnc, bool f_qni, bool f_qnifa, bool f_qnwfa, bool f_qnbca, int initflag, bool restart, bool cycling, float delt, MynnArray<float, 2> dx, MynnArray<float, 2> xland, MynnArray<float, 2> ps, MynnArray<float, 2> ts, MynnArray<float, 2> qsfc, MynnArray<float, 2> ust, MynnArray<float, 2> ch, MynnArray<float, 2> hfx, MynnArray<float, 2> qfx, MynnArray<float, 2> wspd, MynnArray<float, 2> znt, MynnArray<float, 2> uoce, MynnArray<float, 2> voce, MynnArray<float, 3> dz, MynnArray<float, 3> u, MynnArray<float, 3> v, MynnArray<float, 3> w, MynnArray<float, 3> th, MynnArray<float, 3> tk, MynnArray<float, 3> p, MynnArray<float, 3> exner, MynnArray<float, 3> rho, MynnArray<float, 3> qv, MynnArray<float, 3> qc, MynnArray<float, 3> qi, MynnArray<float, 3> qs, MynnArray<float, 3> sqv, MynnArray<float, 3> sqc, MynnArray<float, 3> sqi, MynnArray<float, 3> sqs, MynnArray<float, 3> qnc, MynnArray<float, 3> qni, MynnArray<float, 3> qnifa, MynnArray<float, 3> qnwfa, MynnArray<float, 3> qnbca, MynnArray<float, 3> qoz, MynnArray<float, 3> rthraten, MynnArray<float, 3> pint, MynnArray<float, 3> cldfra_bl, MynnArray<float, 3> qc_bl, MynnArray<float, 3> qi_bl, MynnArray<float, 3> qke, MynnArray<float, 3> qke_adv, MynnArray<float, 3> tsq, MynnArray<float, 3> qsq, MynnArray<float, 3> cov, MynnArray<float, 3> el_pbl, MynnArray<float, 3> rublten, MynnArray<float, 3> rvblten, MynnArray<float, 3> rthblten, MynnArray<float, 3> rqvblten, MynnArray<float, 3> rqcblten, MynnArray<float, 3> rqiblten, MynnArray<float, 3> rqsblten, MynnArray<float, 3> rqncblten, MynnArray<float, 3> rqniblten, MynnArray<float, 3> rqnifablten, MynnArray<float, 3> rqnwfablten, MynnArray<float, 3> rqnbcablten, MynnArray<float, 3> rqozblten, MynnArray<float, 2> pblh, MynnArray<int, 2> kpbl, MynnArray<float, 2> maxwidth, MynnArray<float, 2> maxmf, MynnArray<float, 2> ztop_plume, MynnArray<float, 2> excess_h, MynnArray<float, 2> excess_q, MynnArray<float, 2> maxwidth_dd, MynnArray<float, 2> maxmf_dd, MynnArray<float, 2> maxtkeprod, MynnArray<float, 2> cldtop_cooling, MynnArray<float, 2> ent_eff, MynnArray<float, 2> lwp, MynnArray<float, 2> iwp, MynnArray<float, 2> swp, MynnArray<float, 2> wspd10, MynnArray<float, 2> wspd80, MynnArray<float, 2> wspd160, MynnArray<float, 2> cldceil, MynnArray<float, 2> maxcldfra, MynnArray<float, 2> maxcldfra_pbl, MynnArray<float, 3> edmf_a, MynnArray<float, 3> edmf_w, MynnArray<float, 3> edmf_qt, MynnArray<float, 3> edmf_thl, MynnArray<float, 3> edmf_ent, MynnArray<float, 3> edmf_qc, MynnArray<float, 3> sub_thl, MynnArray<float, 3> sub_sqv, MynnArray<float, 3> det_thl, MynnArray<float, 3> det_sqv, MynnArray<float, 3> exch_h, MynnArray<float, 3> exch_m, MynnArray<float, 3> dqke, MynnArray<float, 3> qwt, MynnArray<float, 3> qshear, MynnArray<float, 3> qbuoy, MynnArray<float, 3> qdiss, MynnArray<float, 3> sh3d, MynnArray<float, 3> sm3d, int spp_pbl, MynnArray<float, 3> pattern_spp, bool bl_mynn_tkeadvect, int tke_budget, int bl_mynn_cloudpdf, int bl_mynn_mixlength, float bl_mynn_closure, int bl_mynn_edmf, int bl_mynn_edmf_mom, int bl_mynn_edmf_tke, int bl_mynn_mixscalars, int bl_mynn_mixaerosols, int bl_mynn_mixnumcon, int bl_mynn_cloudmix, int bl_mynn_mixqt, int bl_mynn_edmf_dd, int bl_mynn_ess, int bl_mynn_diags2d, int bl_mynn_diags3d, MynnOptional<bool> mix_chem, MynnOptional<int> nchem, MynnOptional<int> ndvel, MynnOptional<bool> enh_mix, MynnArray<float, 4> chem3d, MynnArray<float, 4> settle3d, MynnArray<float, 3> vd3d, MynnArray<float, 2> frp_mean, MynnArray<float, 2> emis_ant_no, MynnText& errmsg, int& errflg) {
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
tk.rebase({ims, kms, jms});
p.rebase({ims, kms, jms});
exner.rebase({ims, kms, jms});
rho.rebase({ims, kms, jms});
qv.rebase({ims, kms, jms});
qc.rebase({ims, kms, jms});
qi.rebase({ims, kms, jms});
qs.rebase({ims, kms, jms});
sqv.rebase({ims, kms, jms});
sqc.rebase({ims, kms, jms});
sqi.rebase({ims, kms, jms});
sqs.rebase({ims, kms, jms});
qnc.rebase({ims, kms, jms});
qni.rebase({ims, kms, jms});
qnifa.rebase({ims, kms, jms});
qnwfa.rebase({ims, kms, jms});
qnbca.rebase({ims, kms, jms});
qoz.rebase({ims, kms, jms});
rthraten.rebase({ims, kms, jms});
pint.rebase({ims, kms, jms});
cldfra_bl.rebase({ims, kms, jms});
qc_bl.rebase({ims, kms, jms});
qi_bl.rebase({ims, kms, jms});
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
rqncblten.rebase({ims, kms, jms});
rqniblten.rebase({ims, kms, jms});
rqnifablten.rebase({ims, kms, jms});
rqnwfablten.rebase({ims, kms, jms});
rqnbcablten.rebase({ims, kms, jms});
rqozblten.rebase({ims, kms, jms});
pblh.rebase({ims, jms});
kpbl.rebase({ims, jms});
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
lwp.rebase({ims, jms});
iwp.rebase({ims, jms});
swp.rebase({ims, jms});
wspd10.rebase({ims, jms});
wspd80.rebase({ims, jms});
wspd160.rebase({ims, jms});
cldceil.rebase({ims, jms});
maxcldfra.rebase({ims, jms});
maxcldfra_pbl.rebase({ims, jms});
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
bool mix_chem1;
int nchem1;
int ndvel1;
bool enh_mix1;
float frp1;
float emisant_no1;
MynnArray<float, 1> vd1{};
MynnArray<float, 2> chem1{};
MynnArray<float, 2> settle1{};
constexpr int nscalars = 1;
MynnArray<float, 2> scalars = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {kte, nscalars});
int i;
int k;
int j;
int nc;
int nd;
int kpbl1;
float max_hfx;
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
MynnArray<float, 1> tk1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> p1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> exner1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rho1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rthraten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> delp1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> zagl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> zw1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> w1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> qc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qi1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qs1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qnc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qni1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qnifa1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qnwfa1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qnbca1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qoz1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> pattern_spp1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
float pblh1;
float lwp1;
float iwp1;
float swp1;
float wspd101;
float wspd801;
float wspd1601;
float cldceil1;
float maxcldfra1;
float maxcldfra_pbl1;
MynnArray<float, 1> cldfra_bl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qc_bl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qi_bl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> el_pbl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qke1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> qke_adv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
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
MynnArray<float, 1> rqncblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rqniblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rqnifablten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rqnwfablten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rqnbcablten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> rqozblten1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_a1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_w1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_qt1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_thl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_ent1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_qc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sub_thl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sub_sqv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> det_thl1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> det_sqv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
float maxwidth1;
float maxmf1;
float ztop_plume1;
float excess_h1;
float excess_q1;
float maxwidth_dd1;
float maxmf_dd1;
float maxtkeprod1;
float cldtop_cooling1;
float ent_eff1;
MynnArray<float, 1> kh1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> km1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> dqke1{};
MynnArray<float, 1> qwt1{};
MynnArray<float, 1> qshear1{};
MynnArray<float, 1> qbuoy1{};
MynnArray<float, 1> qdiss1{};
MynnArray<float, 1> sqv1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqc1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqi1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
MynnArray<float, 1> sqs1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte});
constexpr bool debug = false;
constexpr bool dry_mixing_ratio = true;
constexpr bool neg_moist_check = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:420
if (debug) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:421
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:422
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:423
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:424
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:425
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:427
errmsg = " ";
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:428
errflg = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:431
if (((tke_budget) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:432
dqke1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte}); for (int mynn_element=0; mynn_element < (dqke1).size(); ++mynn_element) { dqke1.linear(mynn_element) = zero; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:433
qwt1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte}); for (int mynn_element=0; mynn_element < (qwt1).size(); ++mynn_element) { qwt1.linear(mynn_element) = zero; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:434
qshear1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte}); for (int mynn_element=0; mynn_element < (qshear1).size(); ++mynn_element) { qshear1.linear(mynn_element) = zero; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:435
qbuoy1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte}); for (int mynn_element=0; mynn_element < (qbuoy1).size(); ++mynn_element) { qbuoy1.linear(mynn_element) = zero; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:436
qdiss1 = mynn_array<float, 1, mynn_tile>(arena, {kts}, {kte}); for (int mynn_element=0; mynn_element < (qdiss1).size(); ++mynn_element) { qdiss1.linear(mynn_element) = zero; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:437
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:439
if ((enh_mix.present())) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:440
enh_mix1 = enh_mix;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:441
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:442
enh_mix1 = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:443
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:445
if ((((((((((((mix_chem.present())) && ((chem3d.present())))) && ((settle3d.present())))) && ((vd3d.present())))) && ((frp_mean.present())))) && ((emis_ant_no.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:447
mix_chem1 = mix_chem;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:448
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:449
mix_chem1 = false;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:450
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:455
for (j=(active_j >= 0 ? active_j : jts); (1) > 0 ? j <= ((active_j >= 0 ? active_j : jte)) : j >= ((active_j >= 0 ? active_j : jte)); j += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:456
for (i=(active_i >= 0 ? active_i : its); (1) > 0 ? i <= ((active_i >= 0 ? active_i : ite)) : i >= ((active_i >= 0 ? active_i : ite)); i += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:459
dx1 = dx(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:460
xland1 = xland(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:461
ps1 = ps(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:462
ts1 = ts(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:463
qsfc1 = qsfc(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:464
ust1 = ust(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:465
ch1 = ch(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:466
wspd1 = wspd(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:467
uoce1 = uoce(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:468
voce1 = voce(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:469
znt1 = znt(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:475
max_hfx = 1200.f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:476
if ((frp_mean.present())) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:477
if (((frp_mean(i, j)) > (10.f))) { max_hfx = 6000.f; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:478
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:479
hfx1 = hfx(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:480
if (((hfx1) > (max_hfx))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:482
hfx1 = max_hfx;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:483
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:484
if (((hfx1) < (r6_neg(600.f)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:486
hfx1 = r6_neg(600.f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:487
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:488
qfx1 = qfx(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:489
if (((qfx1) > (9e-4f))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:491
qfx1 = 9e-4f;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:492
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:493
if (((qfx1) < (r6_neg(3e-4f)))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:495
qfx1 = r6_neg(3e-4f);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:496
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:498
zw1(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:499
zagl1(kts) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:500
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:501
dz1(k) = dz(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:502
u1(k) = u(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:503
v1(k) = v(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:504
w1(k) = w(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:505
th1(k) = th(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:506
tk1(k) = tk(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:507
p1(k) = p(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:508
exner1(k) = exner(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:509
rho1(k) = rho(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:510
rthraten1(k) = rthraten(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:511
delp1(k) = mynn_max((float)(r6_sub(pint(i, k, j), pint(i, ((k) + (1)), j))), (float)(0.01f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:512
zagl1(k) = r6_add(zw1(k), r6_mul(0.5f, dz1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:513
zw1(((k) + (1))) = r6_add(zw1(k), dz1(k));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:514
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:515
w1(((kte) + (1))) = w(i, ((kte) + (1)), j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:521
if (dry_mixing_ratio) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:522
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:523
qv1(k) = mynn_max((float)(qv(i, k, j)), (float)(1e-10f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:524
qc1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:525
qi1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:526
qs1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:527
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:529
if (((f_qc) && ((qc.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:530
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:531
qc1(k) = mynn_max((float)(zero), (float)(qc(i, k, j)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:532
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:533
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:534
if (((f_qi) && ((qi.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:535
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:536
qi1(k) = mynn_max((float)(zero), (float)(qi(i, k, j)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:537
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:538
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:539
if (((f_qs) && ((qs.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:540
{ const int mynn_lo3 = (kts), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:541
qs1(k) = mynn_max((float)(zero), (float)(qs(i, k, j)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:542
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:543
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:545
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:546
{ const int mynn_lo4 = (kts), mynn_hi4 = (kte), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:547
sqv1(k) = mynn_max((float)(sqv(i, k, j)), (float)(1e-10f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:548
sqc1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:549
sqi1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:550
sqs1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:551
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:553
if (((f_qc) && ((sqc.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:554
{ const int mynn_lo5 = (kts), mynn_hi5 = (kte), mynn_st5 = (1); const int mynn_n5 = mynn_trip(mynn_lo5, mynn_hi5, mynn_st5); mynn_lane_sync(); for (int mynn_it5 = lane_sub; mynn_it5 < mynn_n5; mynn_it5 += lane_width) { k = mynn_lo5 + mynn_it5*mynn_st5;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:555
sqc1(k) = mynn_max((float)(zero), (float)(sqc(i, k, j)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:556
} mynn_lane_sync(); k = mynn_lo5 + mynn_n5*mynn_st5; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:557
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:558
if (((f_qi) && ((sqi.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:559
{ const int mynn_lo6 = (kts), mynn_hi6 = (kte), mynn_st6 = (1); const int mynn_n6 = mynn_trip(mynn_lo6, mynn_hi6, mynn_st6); mynn_lane_sync(); for (int mynn_it6 = lane_sub; mynn_it6 < mynn_n6; mynn_it6 += lane_width) { k = mynn_lo6 + mynn_it6*mynn_st6;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:560
sqi1(k) = mynn_max((float)(zero), (float)(sqi(i, k, j)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:561
} mynn_lane_sync(); k = mynn_lo6 + mynn_n6*mynn_st6; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:562
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:563
if (((f_qs) && ((sqs.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:564
{ const int mynn_lo7 = (kts), mynn_hi7 = (kte), mynn_st7 = (1); const int mynn_n7 = mynn_trip(mynn_lo7, mynn_hi7, mynn_st7); mynn_lane_sync(); for (int mynn_it7 = lane_sub; mynn_it7 < mynn_n7; mynn_it7 += lane_width) { k = mynn_lo7 + mynn_it7*mynn_st7;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:565
sqs1(k) = mynn_max((float)(zero), (float)(sqs(i, k, j)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:566
} mynn_lane_sync(); k = mynn_lo7 + mynn_n7*mynn_st7; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:567
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:568
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:570
{ const int mynn_lo8 = (kts), mynn_hi8 = (kte), mynn_st8 = (1); const int mynn_n8 = mynn_trip(mynn_lo8, mynn_hi8, mynn_st8); mynn_lane_sync(); for (int mynn_it8 = lane_sub; mynn_it8 < mynn_n8; mynn_it8 += lane_width) { k = mynn_lo8 + mynn_it8*mynn_st8;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:571
qoz1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:572
qnc1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:573
qni1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:574
qnifa1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:575
qnwfa1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:576
qnbca1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:577
} mynn_lane_sync(); k = mynn_lo8 + mynn_n8*mynn_st8; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:579
if (((f_qnc) && ((qnc.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:580
{ const int mynn_lo9 = (kts), mynn_hi9 = (kte), mynn_st9 = (1); const int mynn_n9 = mynn_trip(mynn_lo9, mynn_hi9, mynn_st9); mynn_lane_sync(); for (int mynn_it9 = lane_sub; mynn_it9 < mynn_n9; mynn_it9 += lane_width) { k = mynn_lo9 + mynn_it9*mynn_st9;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:581
qnc1(k) = qnc(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:582
} mynn_lane_sync(); k = mynn_lo9 + mynn_n9*mynn_st9; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:583
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:584
if (((f_qni) && ((qni.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:585
{ const int mynn_lo10 = (kts), mynn_hi10 = (kte), mynn_st10 = (1); const int mynn_n10 = mynn_trip(mynn_lo10, mynn_hi10, mynn_st10); mynn_lane_sync(); for (int mynn_it10 = lane_sub; mynn_it10 < mynn_n10; mynn_it10 += lane_width) { k = mynn_lo10 + mynn_it10*mynn_st10;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:586
qni1(k) = qni(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:587
} mynn_lane_sync(); k = mynn_lo10 + mynn_n10*mynn_st10; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:588
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:589
if (((f_qnifa) && ((qnifa.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:590
{ const int mynn_lo11 = (kts), mynn_hi11 = (kte), mynn_st11 = (1); const int mynn_n11 = mynn_trip(mynn_lo11, mynn_hi11, mynn_st11); mynn_lane_sync(); for (int mynn_it11 = lane_sub; mynn_it11 < mynn_n11; mynn_it11 += lane_width) { k = mynn_lo11 + mynn_it11*mynn_st11;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:591
qnifa1(k) = qnifa(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:592
} mynn_lane_sync(); k = mynn_lo11 + mynn_n11*mynn_st11; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:593
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:594
if (((f_qnwfa) && ((qnwfa.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:595
{ const int mynn_lo12 = (kts), mynn_hi12 = (kte), mynn_st12 = (1); const int mynn_n12 = mynn_trip(mynn_lo12, mynn_hi12, mynn_st12); mynn_lane_sync(); for (int mynn_it12 = lane_sub; mynn_it12 < mynn_n12; mynn_it12 += lane_width) { k = mynn_lo12 + mynn_it12*mynn_st12;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:596
qnwfa1(k) = qnwfa(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:597
} mynn_lane_sync(); k = mynn_lo12 + mynn_n12*mynn_st12; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:598
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:599
if (((f_qnbca) && ((qnbca.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:600
{ const int mynn_lo13 = (kts), mynn_hi13 = (kte), mynn_st13 = (1); const int mynn_n13 = mynn_trip(mynn_lo13, mynn_hi13, mynn_st13); mynn_lane_sync(); for (int mynn_it13 = lane_sub; mynn_it13 < mynn_n13; mynn_it13 += lane_width) { k = mynn_lo13 + mynn_it13*mynn_st13;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:601
qnbca1(k) = qnbca(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:602
} mynn_lane_sync(); k = mynn_lo13 + mynn_n13*mynn_st13; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:603
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:604
if (((f_qoz) && ((qoz.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:605
{ const int mynn_lo14 = (kts), mynn_hi14 = (kte), mynn_st14 = (1); const int mynn_n14 = mynn_trip(mynn_lo14, mynn_hi14, mynn_st14); mynn_lane_sync(); for (int mynn_it14 = lane_sub; mynn_it14 < mynn_n14; mynn_it14 += lane_width) { k = mynn_lo14 + mynn_it14*mynn_st14;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:606
qoz1(k) = qoz(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:607
} mynn_lane_sync(); k = mynn_lo14 + mynn_n14*mynn_st14; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:608
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:610
{ const int mynn_lo15 = (kts), mynn_hi15 = (kte), mynn_st15 = (1); const int mynn_n15 = mynn_trip(mynn_lo15, mynn_hi15, mynn_st15); mynn_lane_sync(); for (int mynn_it15 = lane_sub; mynn_it15 < mynn_n15; mynn_it15 += lane_width) { k = mynn_lo15 + mynn_it15*mynn_st15;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:611
rqcblten1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:612
rqiblten1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:613
rqsblten1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:614
rqozblten1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:615
rqncblten1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:616
rqniblten1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:617
rqnifablten1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:618
rqnwfablten1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:619
rqnbcablten1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:620
} mynn_lane_sync(); k = mynn_lo15 + mynn_n15*mynn_st15; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:623
if (dry_mixing_ratio) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:624
f_mynnedmf_qv_to_sqv(kte, f_qc, f_qi, f_qs, qv1, qc1, qi1, qs1, sqv1, sqc1, sqi1, sqs1, errmsg, errflg);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:626
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:627
f_mynnedmf_sqv_to_qv(kte, f_qc, f_qi, f_qs, qv1, qc1, qi1, qs1, sqv1, sqc1, sqi1, sqs1, errmsg, errflg);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:629
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:632
if (((((spp_pbl) > (0))) && ((pattern_spp.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:633
{ const int mynn_lo16 = (kts), mynn_hi16 = (kte), mynn_st16 = (1); const int mynn_n16 = mynn_trip(mynn_lo16, mynn_hi16, mynn_st16); mynn_lane_sync(); for (int mynn_it16 = lane_sub; mynn_it16 < mynn_n16; mynn_it16 += lane_width) { k = mynn_lo16 + mynn_it16*mynn_st16;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:634
pattern_spp1(k) = pattern_spp(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:635
} mynn_lane_sync(); k = mynn_lo16 + mynn_n16*mynn_st16; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:636
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:637
{ const int mynn_lo17 = (kts), mynn_hi17 = (kte), mynn_st17 = (1); const int mynn_n17 = mynn_trip(mynn_lo17, mynn_hi17, mynn_st17); mynn_lane_sync(); for (int mynn_it17 = lane_sub; mynn_it17 < mynn_n17; mynn_it17 += lane_width) { k = mynn_lo17 + mynn_it17*mynn_st17;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:638
pattern_spp1(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:639
} mynn_lane_sync(); k = mynn_lo17 + mynn_n17*mynn_st17; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:640
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:643
pblh1 = pblh(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:644
kpbl1 = kpbl(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:647
if (((((initflag) == (0))) || (restart))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:649
{ const int mynn_lo18 = (kts), mynn_hi18 = (kte), mynn_st18 = (1); const int mynn_n18 = mynn_trip(mynn_lo18, mynn_hi18, mynn_st18); mynn_lane_sync(); for (int mynn_it18 = lane_sub; mynn_it18 < mynn_n18; mynn_it18 += lane_width) { k = mynn_lo18 + mynn_it18*mynn_st18;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:650
cldfra_bl1(k) = cldfra_bl(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:651
qc_bl1(k) = qc_bl(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:652
qi_bl1(k) = qi_bl(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:653
} mynn_lane_sync(); k = mynn_lo18 + mynn_n18*mynn_st18; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:656
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:657
el_pbl1(k) = el_pbl(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:658
qke1(k) = qke(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:660
cov1(k) = cov(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:661
tsq1(k) = tsq(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:662
qsq1(k) = qsq(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:663
sh1(k) = sh3d(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:664
sm1(k) = sm3d(i, k, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:665
if (bl_mynn_tkeadvect) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:666
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke_adv1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qke_adv1.slice(MynnRange{kts, kte}).linear(mynn_element) = qke_adv.slice(i, MynnRange{kts, kte}, j).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:667
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:668
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qke_adv1.slice(MynnRange{kts, kte})).size(); mynn_element += lane_width) { qke_adv1.slice(MynnRange{kts, kte}).linear(mynn_element) = qke.slice(i, MynnRange{kts, kte}, j).linear(mynn_element); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:669
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:670
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:671
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:674
if (mix_chem1) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:675
ndvel1 = ndvel;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:676
nchem1 = nchem;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:677
vd1 = mynn_array<float, 1, mynn_tile>(arena, {1}, {ndvel1});
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:678
chem1 = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {kte, nchem1});
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:679
settle1 = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {kte, nchem1});
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:680
{ const int mynn_lo19 = (1), mynn_hi19 = (nchem), mynn_st19 = (1); const int mynn_n19 = mynn_trip(mynn_lo19, mynn_hi19, mynn_st19); mynn_lane_sync(); for (int mynn_it19 = lane_sub; mynn_it19 < mynn_n19; mynn_it19 += lane_width) { nc = mynn_lo19 + mynn_it19*mynn_st19;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:681
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:682
chem1(k, nc) = chem3d(i, k, j, nc);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:683
settle1(k, nc) = settle3d(i, k, j, nc);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:684
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:685
} mynn_lane_sync(); nc = mynn_lo19 + mynn_n19*mynn_st19; if (mynn_n19 > 0) { const int mynn_src19 = (mynn_n19 - 1) % lane_width; k = mynn_lane_bcast(k, mynn_src19); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:686
{ const int mynn_lo20 = (1), mynn_hi20 = (ndvel), mynn_st20 = (1); const int mynn_n20 = mynn_trip(mynn_lo20, mynn_hi20, mynn_st20); mynn_lane_sync(); for (int mynn_it20 = lane_sub; mynn_it20 < mynn_n20; mynn_it20 += lane_width) { nd = mynn_lo20 + mynn_it20*mynn_st20;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:687
vd1(nd) = vd3d(i, j, nd);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:688
} mynn_lane_sync(); nd = mynn_lo20 + mynn_n20*mynn_st20; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:689
frp1 = frp_mean(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:690
emisant_no1 = emis_ant_no(i, j);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:691
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:692
ndvel1 = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:693
nchem1 = 1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:694
vd1 = mynn_array<float, 1, mynn_tile>(arena, {1}, {ndvel1});
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:695
chem1 = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {kte, nchem1});
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:696
settle1 = mynn_array<float, 2, mynn_tile>(arena, {kts, 1}, {kte, nchem1});
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:697
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (chem1).size(); mynn_element += lane_width) { chem1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:698
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (settle1).size(); mynn_element += lane_width) { settle1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:699
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (vd1).size(); mynn_element += lane_width) { vd1.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:700
frp1 = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:701
emisant_no1 = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:702
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:703
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (scalars).size(); mynn_element += lane_width) { scalars.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:706
if (debug) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:707
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:708
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:710
f_mynnedmf(i, j, initflag, restart, cycling, delt, dz1, dx1, u1, v1, w1, th1, sqv1, sqc1, sqi1, sqs1, qv1, qc1, qi1, qs1, qnc1, qni1, qnwfa1, qnifa1, qnbca1, qoz1, p1, exner1, rho1, tk1, delp1, zw1, zagl1, xland1, ts1, qsfc1, ps1, ust1, ch1, hfx1, qfx1, znt1, wspd1, uoce1, voce1, qke1, qke_adv1, el_pbl1, sh1, sm1, kh1, km1, nchem1, ndvel1, chem1, vd1, frp1, emisant_no1, mix_chem1, enh_mix1, settle1, scalars, nscalars, tsq1, qsq1, cov1, rublten1, rvblten1, rthblten1, rqvblten1, rqcblten1, rqiblten1, rqncblten1, rqniblten1, rqsblten1, rqnwfablten1, rqnifablten1, rqnbcablten1, rqozblten1, rthraten1, pblh1, kpbl1, maxwidth_dd1, maxwidth1, maxmf1, ztop_plume1, excess_h1, excess_q1, maxmf_dd1, maxtkeprod1, cldtop_cooling1, ent_eff1, dqke1, qwt1, qshear1, qbuoy1, qdiss1, qc_bl1, qi_bl1, cldfra_bl1, bl_mynn_tkeadvect, tke_budget, bl_mynn_cloudpdf, bl_mynn_mixlength, bl_mynn_closure, bl_mynn_ess, bl_mynn_edmf, bl_mynn_edmf_mom, bl_mynn_edmf_tke, bl_mynn_mixscalars, bl_mynn_mixaerosols, bl_mynn_mixnumcon, bl_mynn_diags3d, bl_mynn_cloudmix, bl_mynn_mixqt, bl_mynn_edmf_dd, edmf_a1, edmf_w1, edmf_qt1, edmf_thl1, edmf_ent1, edmf_qc1, sub_thl1, sub_sqv1, det_thl1, det_sqv1, spp_pbl, pattern_spp1, f_qc, f_qi, f_qnc, f_qni, f_qs, f_qnwfa, f_qnifa, f_qnbca, f_qoz, kts, kte, errmsg, errflg);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:774
if (debug) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:775
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:776
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:779
if (dry_mixing_ratio) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:780
f_mynnedmf_post_run(kte, f_qc, f_qi, f_qs, delt, qv1, qc1, qi1, qs1, rqvblten1, rqcblten1, rqiblten1, rqsblten1, errmsg, errflg);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:782
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:785
pblh(i, j) = pblh1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:786
kpbl(i, j) = kpbl1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:788
if (dry_mixing_ratio) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:789
{ const int mynn_lo21 = (kts), mynn_hi21 = (kte), mynn_st21 = (1); const int mynn_n21 = mynn_trip(mynn_lo21, mynn_hi21, mynn_st21); mynn_lane_sync(); for (int mynn_it21 = lane_sub; mynn_it21 < mynn_n21; mynn_it21 += lane_width) { k = mynn_lo21 + mynn_it21*mynn_st21;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:790
qc_bl(i, k, j) = r6_div(qc_bl1(k), r6_sub(one, sqv1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:791
qi_bl(i, k, j) = r6_div(qi_bl1(k), r6_sub(one, sqv1(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:792
cldfra_bl(i, k, j) = cldfra_bl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:793
} mynn_lane_sync(); k = mynn_lo21 + mynn_n21*mynn_st21; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:794
} else {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:795
{ const int mynn_lo22 = (kts), mynn_hi22 = (kte), mynn_st22 = (1); const int mynn_n22 = mynn_trip(mynn_lo22, mynn_hi22, mynn_st22); mynn_lane_sync(); for (int mynn_it22 = lane_sub; mynn_it22 < mynn_n22; mynn_it22 += lane_width) { k = mynn_lo22 + mynn_it22*mynn_st22;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:796
qc_bl(i, k, j) = qc_bl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:797
qi_bl(i, k, j) = qi_bl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:798
cldfra_bl(i, k, j) = cldfra_bl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:799
} mynn_lane_sync(); k = mynn_lo22 + mynn_n22*mynn_st22; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:800
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:802
{ const int mynn_lo23 = (kts), mynn_hi23 = (kte), mynn_st23 = (1); const int mynn_n23 = mynn_trip(mynn_lo23, mynn_hi23, mynn_st23); mynn_lane_sync(); for (int mynn_it23 = lane_sub; mynn_it23 < mynn_n23; mynn_it23 += lane_width) { k = mynn_lo23 + mynn_it23*mynn_st23;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:803
el_pbl(i, k, j) = el_pbl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:804
qke(i, k, j) = qke1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:805
qke_adv(i, k, j) = qke_adv1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:806
cov(i, k, j) = cov1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:807
tsq(i, k, j) = tsq1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:808
qsq(i, k, j) = qsq1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:809
sh3d(i, k, j) = sh1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:810
sm3d(i, k, j) = sm1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:811
exch_h(i, k, j) = kh1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:812
exch_m(i, k, j) = km1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:813
} mynn_lane_sync(); k = mynn_lo23 + mynn_n23*mynn_st23; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:816
{ const int mynn_lo24 = (kts), mynn_hi24 = (kte), mynn_st24 = (1); const int mynn_n24 = mynn_trip(mynn_lo24, mynn_hi24, mynn_st24); mynn_lane_sync(); for (int mynn_it24 = lane_sub; mynn_it24 < mynn_n24; mynn_it24 += lane_width) { k = mynn_lo24 + mynn_it24*mynn_st24;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:817
rublten(i, k, j) = rublten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:818
rvblten(i, k, j) = rvblten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:819
rthblten(i, k, j) = rthblten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:820
} mynn_lane_sync(); k = mynn_lo24 + mynn_n24*mynn_st24; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:822
if ((rqvblten.present())) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:823
{ const int mynn_lo25 = (kts), mynn_hi25 = (kte), mynn_st25 = (1); const int mynn_n25 = mynn_trip(mynn_lo25, mynn_hi25, mynn_st25); mynn_lane_sync(); for (int mynn_it25 = lane_sub; mynn_it25 < mynn_n25; mynn_it25 += lane_width) { k = mynn_lo25 + mynn_it25*mynn_st25;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:824
rqvblten(i, k, j) = rqvblten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:825
} mynn_lane_sync(); k = mynn_lo25 + mynn_n25*mynn_st25; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:826
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:827
if (((f_qc) && ((rqcblten.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:828
{ const int mynn_lo26 = (kts), mynn_hi26 = (kte), mynn_st26 = (1); const int mynn_n26 = mynn_trip(mynn_lo26, mynn_hi26, mynn_st26); mynn_lane_sync(); for (int mynn_it26 = lane_sub; mynn_it26 < mynn_n26; mynn_it26 += lane_width) { k = mynn_lo26 + mynn_it26*mynn_st26;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:829
rqcblten(i, k, j) = rqcblten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:830
} mynn_lane_sync(); k = mynn_lo26 + mynn_n26*mynn_st26; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:831
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:832
if (((f_qi) && ((rqiblten.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:833
{ const int mynn_lo27 = (kts), mynn_hi27 = (kte), mynn_st27 = (1); const int mynn_n27 = mynn_trip(mynn_lo27, mynn_hi27, mynn_st27); mynn_lane_sync(); for (int mynn_it27 = lane_sub; mynn_it27 < mynn_n27; mynn_it27 += lane_width) { k = mynn_lo27 + mynn_it27*mynn_st27;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:834
rqiblten(i, k, j) = rqiblten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:835
} mynn_lane_sync(); k = mynn_lo27 + mynn_n27*mynn_st27; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:836
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:837
if (((f_qs) && ((rqsblten.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:838
{ const int mynn_lo28 = (kts), mynn_hi28 = (kte), mynn_st28 = (1); const int mynn_n28 = mynn_trip(mynn_lo28, mynn_hi28, mynn_st28); mynn_lane_sync(); for (int mynn_it28 = lane_sub; mynn_it28 < mynn_n28; mynn_it28 += lane_width) { k = mynn_lo28 + mynn_it28*mynn_st28;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:839
rqsblten(i, k, j) = rqsblten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:840
} mynn_lane_sync(); k = mynn_lo28 + mynn_n28*mynn_st28; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:841
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:842
if (((f_qoz) && ((rqozblten.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:843
{ const int mynn_lo29 = (kts), mynn_hi29 = (kte), mynn_st29 = (1); const int mynn_n29 = mynn_trip(mynn_lo29, mynn_hi29, mynn_st29); mynn_lane_sync(); for (int mynn_it29 = lane_sub; mynn_it29 < mynn_n29; mynn_it29 += lane_width) { k = mynn_lo29 + mynn_it29*mynn_st29;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:844
rqozblten(i, k, j) = rqozblten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:845
} mynn_lane_sync(); k = mynn_lo29 + mynn_n29*mynn_st29; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:846
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:847
if (((f_qnc) && ((rqncblten.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:848
{ const int mynn_lo30 = (kts), mynn_hi30 = (kte), mynn_st30 = (1); const int mynn_n30 = mynn_trip(mynn_lo30, mynn_hi30, mynn_st30); mynn_lane_sync(); for (int mynn_it30 = lane_sub; mynn_it30 < mynn_n30; mynn_it30 += lane_width) { k = mynn_lo30 + mynn_it30*mynn_st30;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:849
rqncblten(i, k, j) = rqncblten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:850
} mynn_lane_sync(); k = mynn_lo30 + mynn_n30*mynn_st30; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:851
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:852
if (((f_qni) && ((rqniblten.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:853
{ const int mynn_lo31 = (kts), mynn_hi31 = (kte), mynn_st31 = (1); const int mynn_n31 = mynn_trip(mynn_lo31, mynn_hi31, mynn_st31); mynn_lane_sync(); for (int mynn_it31 = lane_sub; mynn_it31 < mynn_n31; mynn_it31 += lane_width) { k = mynn_lo31 + mynn_it31*mynn_st31;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:854
rqniblten(i, k, j) = rqniblten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:855
} mynn_lane_sync(); k = mynn_lo31 + mynn_n31*mynn_st31; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:856
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:857
if (((f_qnifa) && ((rqnifablten.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:858
{ const int mynn_lo32 = (kts), mynn_hi32 = (kte), mynn_st32 = (1); const int mynn_n32 = mynn_trip(mynn_lo32, mynn_hi32, mynn_st32); mynn_lane_sync(); for (int mynn_it32 = lane_sub; mynn_it32 < mynn_n32; mynn_it32 += lane_width) { k = mynn_lo32 + mynn_it32*mynn_st32;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:859
rqnifablten(i, k, j) = rqnifablten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:860
} mynn_lane_sync(); k = mynn_lo32 + mynn_n32*mynn_st32; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:861
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:862
if (((f_qnwfa) && ((rqnwfablten.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:863
{ const int mynn_lo33 = (kts), mynn_hi33 = (kte), mynn_st33 = (1); const int mynn_n33 = mynn_trip(mynn_lo33, mynn_hi33, mynn_st33); mynn_lane_sync(); for (int mynn_it33 = lane_sub; mynn_it33 < mynn_n33; mynn_it33 += lane_width) { k = mynn_lo33 + mynn_it33*mynn_st33;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:864
rqnwfablten(i, k, j) = rqnwfablten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:865
} mynn_lane_sync(); k = mynn_lo33 + mynn_n33*mynn_st33; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:866
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:867
if (((f_qnbca) && ((rqnbcablten.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:868
{ const int mynn_lo34 = (kts), mynn_hi34 = (kte), mynn_st34 = (1); const int mynn_n34 = mynn_trip(mynn_lo34, mynn_hi34, mynn_st34); mynn_lane_sync(); for (int mynn_it34 = lane_sub; mynn_it34 < mynn_n34; mynn_it34 += lane_width) { k = mynn_lo34 + mynn_it34*mynn_st34;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:869
rqnbcablten(i, k, j) = rqnbcablten1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:870
} mynn_lane_sync(); k = mynn_lo34 + mynn_n34*mynn_st34; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:871
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:873
if (((bl_mynn_diags3d) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:874
{ const int mynn_lo35 = (kts), mynn_hi35 = (kte), mynn_st35 = (1); const int mynn_n35 = mynn_trip(mynn_lo35, mynn_hi35, mynn_st35); mynn_lane_sync(); for (int mynn_it35 = lane_sub; mynn_it35 < mynn_n35; mynn_it35 += lane_width) { k = mynn_lo35 + mynn_it35*mynn_st35;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:875
edmf_a(i, k, j) = edmf_a1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:876
edmf_w(i, k, j) = edmf_w1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:877
edmf_qt(i, k, j) = edmf_qt1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:878
edmf_thl(i, k, j) = edmf_thl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:879
edmf_ent(i, k, j) = edmf_ent1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:880
edmf_qc(i, k, j) = edmf_qc1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:881
sub_thl(i, k, j) = sub_thl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:882
sub_sqv(i, k, j) = sub_sqv1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:883
det_thl(i, k, j) = det_thl1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:884
det_sqv(i, k, j) = det_sqv1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:885
} mynn_lane_sync(); k = mynn_lo35 + mynn_n35*mynn_st35; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:886
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:889
if (((bl_mynn_edmf) > (0))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:890
maxwidth(i, j) = maxwidth1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:891
maxmf(i, j) = maxmf1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:892
ztop_plume(i, j) = ztop_plume1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:893
excess_h(i, j) = excess_h1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:894
excess_q(i, j) = excess_q1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:895
maxwidth_dd(i, j) = maxwidth_dd1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:896
maxmf_dd(i, j) = maxmf_dd1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:897
maxtkeprod(i, j) = maxtkeprod1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:898
cldtop_cooling(i, j) = cldtop_cooling1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:899
ent_eff(i, j) = ent_eff1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:900
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:902
if (((((((((((((tke_budget) == (1))) && ((qwt.present())))) && ((qbuoy.present())))) && ((qshear.present())))) && ((qdiss.present())))) && ((dqke.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:904
{ const int mynn_lo36 = (kts), mynn_hi36 = (kte), mynn_st36 = (1); const int mynn_n36 = mynn_trip(mynn_lo36, mynn_hi36, mynn_st36); mynn_lane_sync(); for (int mynn_it36 = lane_sub; mynn_it36 < mynn_n36; mynn_it36 += lane_width) { k = mynn_lo36 + mynn_it36*mynn_st36;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:905
dqke(i, k, j) = dqke1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:906
qwt(i, k, j) = qwt1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:907
qshear(i, k, j) = qshear1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:908
qbuoy(i, k, j) = qbuoy1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:909
qdiss(i, k, j) = qdiss1(k);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:910
} mynn_lane_sync(); k = mynn_lo36 + mynn_n36*mynn_st36; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:911
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:913
if (((mix_chem1) && ((chem3d.present())))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:914
{ const int mynn_lo37 = (1), mynn_hi37 = (nchem), mynn_st37 = (1); const int mynn_n37 = mynn_trip(mynn_lo37, mynn_hi37, mynn_st37); mynn_lane_sync(); for (int mynn_it37 = lane_sub; mynn_it37 < mynn_n37; mynn_it37 += lane_width) { nc = mynn_lo37 + mynn_it37*mynn_st37;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:915
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:916
chem3d(i, k, j, nc) = mynn_max((float)(chem1(k, nc)), (float)(1.e-12f));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:917
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:918
} mynn_lane_sync(); nc = mynn_lo37 + mynn_n37*mynn_st37; if (mynn_n37 > 0) { const int mynn_src37 = (mynn_n37 - 1) % lane_width; k = mynn_lane_bcast(k, mynn_src37); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:919
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:920
; /* DEALLOCATE: the storage returns with the routine's arena frame. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:921
; /* DEALLOCATE: the storage returns with the routine's arena frame. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:922
; /* DEALLOCATE: the storage returns with the routine's arena frame. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:925
if (debug) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:926
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:927
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:928
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:930
if (((bl_mynn_diags2d) >= (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:931
f_mynnedmf_diags2d(kts, kte, delp1, dz1, zw1, zagl1, u1, v1, tk1, qc1, qi1, qs1, qc_bl1, qi_bl1, cldfra_bl1, rho1, xland1, pblh1, MynnOptionalRef<float>(&(lwp1)), MynnOptionalRef<float>(&(iwp1)), MynnOptionalRef<float>(&(swp1)), MynnOptionalRef<float>(&(cldceil1)), MynnOptionalRef<float>(&(wspd101)), MynnOptionalRef<float>(&(wspd801)), MynnOptionalRef<float>(&(wspd1601)), MynnOptionalRef<float>(&(maxcldfra1)), MynnOptionalRef<float>(&(maxcldfra_pbl1)), bl_mynn_diags2d);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:943
lwp(i, j) = lwp1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:944
iwp(i, j) = iwp1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:945
swp(i, j) = swp1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:946
cldceil(i, j) = cldceil1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:948
if (((bl_mynn_diags2d) >= (2))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:949
wspd10(i, j) = wspd101;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:950
wspd80(i, j) = wspd801;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:951
wspd160(i, j) = wspd1601;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:953
maxcldfra(i, j) = maxcldfra1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:954
maxcldfra_pbl(i, j) = maxcldfra_pbl1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:955
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:957
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:959
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:960
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:962
if (((tke_budget) == (1))) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:963
; /* DEALLOCATE: the storage returns with the routine's arena frame. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:964
; /* DEALLOCATE: the storage returns with the routine's arena frame. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:965
; /* DEALLOCATE: the storage returns with the routine's arena frame. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:966
; /* DEALLOCATE: the storage returns with the routine's arena frame. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:967
; /* DEALLOCATE: the storage returns with the routine's arena frame. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:968
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:970
if (debug) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:971
; /* Native diagnostic print; no field arithmetic. */
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:972
}
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1025-1096
__device__ __noinline__ void f_mynnedmf_qv_to_sqv(int kte, bool f_qc, bool f_qi, bool f_qs, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> sqv, MynnArray<float, 1> sqc, MynnArray<float, 1> sqi, MynnArray<float, 1> sqs, MynnText& errmsg, int& errflg) {
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1064
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1065
sqc(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1066
sqi(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1067
sqs(k) = zero;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1068
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1071
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1072
sqv(k) = r6_div(qv(k), r6_add(one, qv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1073
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1076
if (f_qc) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1077
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1078
sqc(k) = r6_div(qc(k), r6_add(one, qv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1079
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1080
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1081
if (f_qi) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1082
{ const int mynn_lo3 = (kts), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1083
sqi(k) = r6_div(qi(k), r6_add(one, qv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1084
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1085
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1086
if (f_qs) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1087
{ const int mynn_lo4 = (kts), mynn_hi4 = (kte), mynn_st4 = (1); const int mynn_n4 = mynn_trip(mynn_lo4, mynn_hi4, mynn_st4); mynn_lane_sync(); for (int mynn_it4 = lane_sub; mynn_it4 < mynn_n4; mynn_it4 += lane_width) { k = mynn_lo4 + mynn_it4*mynn_st4;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1088
sqs(k) = r6_div(qs(k), r6_add(one, qv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1089
} mynn_lane_sync(); k = mynn_lo4 + mynn_n4*mynn_st4; }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1090
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1093
errflg = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1094
errmsg = " ";
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1103-1161
__device__ __noinline__ void f_mynnedmf_sqv_to_qv(int kte, bool f_qc, bool f_qi, bool f_qs, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> sqv, MynnArray<float, 1> sqc, MynnArray<float, 1> sqi, MynnArray<float, 1> sqs, MynnText& errmsg, int& errflg) {
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
MynnArray<float, 1> dry_scale = mynn_array<float, 1, mynn_tile>(arena, {1}, {kte});
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1143
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc).size(); mynn_element += lane_width) { qc.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1144
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi).size(); mynn_element += lane_width) { qi.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1145
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qs).size(); mynn_element += lane_width) { qs.linear(mynn_element) = zero; } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1148
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (dry_scale).size(); mynn_element += lane_width) { dry_scale.linear(mynn_element) = r6_div(one, r6_sub(one, sqv.linear(mynn_element))); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1150
mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qv).size(); mynn_element += lane_width) { qv.linear(mynn_element) = mynn_max((float)(r6_mul(sqv.linear(mynn_element), dry_scale.linear(mynn_element))), (float)(1e-10f)); } mynn_lane_sync();
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1153
if (f_qc) { mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qc).size(); mynn_element += lane_width) { qc.linear(mynn_element) = r6_mul(sqc.linear(mynn_element), dry_scale.linear(mynn_element)); } mynn_lane_sync(); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1154
if (f_qi) { mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qi).size(); mynn_element += lane_width) { qi.linear(mynn_element) = r6_mul(sqi.linear(mynn_element), dry_scale.linear(mynn_element)); } mynn_lane_sync(); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1155
if (f_qs) { mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < (qs).size(); mynn_element += lane_width) { qs.linear(mynn_element) = r6_mul(sqs.linear(mynn_element), dry_scale.linear(mynn_element)); } mynn_lane_sync(); }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1158
errflg = 0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1159
errmsg = " ";
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1212-1293
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
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1255
{ const int mynn_lo0 = (kts), mynn_hi0 = (kte), mynn_st0 = (1); const int mynn_n0 = mynn_trip(mynn_lo0, mynn_hi0, mynn_st0); mynn_lane_sync(); for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width) { k = mynn_lo0 + mynn_it0*mynn_st0;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1256
sq = r6_div(qv(k), r6_add(one, qv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1257
sqv(k) = r6_add(sq, r6_mul(dqv(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1258
rq = r6_div(sqv(k), r6_sub(one, sqv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1259
dqv(k) = r6_div(r6_sub(rq, qv(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1260
} mynn_lane_sync(); k = mynn_lo0 + mynn_n0*mynn_st0; if (mynn_n0 > 0) { const int mynn_src0 = (mynn_n0 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src0); sq = mynn_lane_bcast(sq, mynn_src0); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1262
if (f_qc) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1263
{ const int mynn_lo1 = (kts), mynn_hi1 = (kte), mynn_st1 = (1); const int mynn_n1 = mynn_trip(mynn_lo1, mynn_hi1, mynn_st1); mynn_lane_sync(); for (int mynn_it1 = lane_sub; mynn_it1 < mynn_n1; mynn_it1 += lane_width) { k = mynn_lo1 + mynn_it1*mynn_st1;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1264
sq = r6_div(qc(k), r6_add(one, qv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1265
sqc(k) = r6_add(sq, r6_mul(dqc(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1266
rq = r6_div(sqc(k), r6_sub(one, sqv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1267
dqc(k) = r6_div(r6_sub(rq, qc(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1268
} mynn_lane_sync(); k = mynn_lo1 + mynn_n1*mynn_st1; if (mynn_n1 > 0) { const int mynn_src1 = (mynn_n1 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src1); sq = mynn_lane_bcast(sq, mynn_src1); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1269
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1271
if (f_qi) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1272
{ const int mynn_lo2 = (kts), mynn_hi2 = (kte), mynn_st2 = (1); const int mynn_n2 = mynn_trip(mynn_lo2, mynn_hi2, mynn_st2); mynn_lane_sync(); for (int mynn_it2 = lane_sub; mynn_it2 < mynn_n2; mynn_it2 += lane_width) { k = mynn_lo2 + mynn_it2*mynn_st2;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1273
sq = r6_div(qi(k), r6_add(one, qv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1274
sqi(k) = r6_add(sq, r6_mul(dqi(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1275
rq = r6_div(sqi(k), r6_sub(one, sqv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1276
dqi(k) = r6_div(r6_sub(rq, qi(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1277
} mynn_lane_sync(); k = mynn_lo2 + mynn_n2*mynn_st2; if (mynn_n2 > 0) { const int mynn_src2 = (mynn_n2 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src2); sq = mynn_lane_bcast(sq, mynn_src2); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1278
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1280
if (f_qs) {
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1281
{ const int mynn_lo3 = (kts), mynn_hi3 = (kte), mynn_st3 = (1); const int mynn_n3 = mynn_trip(mynn_lo3, mynn_hi3, mynn_st3); mynn_lane_sync(); for (int mynn_it3 = lane_sub; mynn_it3 < mynn_n3; mynn_it3 += lane_width) { k = mynn_lo3 + mynn_it3*mynn_st3;
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1282
sq = r6_div(qs(k), r6_add(one, qv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1283
sqs(k) = r6_add(sq, r6_mul(dqs(k), delt));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1284
rq = r6_div(sqs(k), r6_sub(one, sqv(k)));
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1285
dqs(k) = r6_div(r6_sub(rq, qs(k)), delt);
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1286
} mynn_lane_sync(); k = mynn_lo3 + mynn_n3*mynn_st3; if (mynn_n3 > 0) { const int mynn_src3 = (mynn_n3 - 1) % lane_width; rq = mynn_lane_bcast(rq, mynn_src3); sq = mynn_lane_bcast(sq, mynn_src3); } }
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1287
}
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1290
errmsg = " ";
// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:1291
errflg = 0;
}
};
#if defined(HEX_NUMERICS_FAST) && !defined(__HIP_PLATFORM_AMD__)
#define MYNN_L4_REGS __maxnreg__(160)
#else
#define MYNN_L4_REGS
#endif

// NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90:82-974
template<int MYNN_LANES> __device__ __forceinline__ void rrfs_mynnedmf_column(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_qnc,int f_qni,int f_qnifa,int f_qnwfa,int f_qnbca,int initflag,int restart,int cycling,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tk,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* sqv,float* sqc,float* sqi,float* sqs,float* qnc,float* qni,float* qnifa,float* qnwfa,float* qnbca,float* qoz,float* rthraten,float* pint,float* cldfra_bl,float* qc_bl,float* qi_bl,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rqncblten,float* rqniblten,float* rqnifablten,float* rqnwfablten,float* rqnbcablten,float* rqozblten,float* pblh,int* kpbl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* maxwidth_dd,float* maxmf_dd,float* maxtkeprod,float* cldtop_cooling,float* ent_eff,float* lwp,float* iwp,float* swp,float* wspd10,float* wspd80,float* wspd160,float* cldceil,float* maxcldfra,float* maxcldfra_pbl,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int tke_budget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_edmf,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_edmf_dd,int bl_mynn_ess,int bl_mynn_diags2d,int bl_mynn_diags3d,int mix_chem,int present_mix_chem,int nchem,int present_nchem,int ndvel,int present_ndvel,int enh_mix,int present_enh_mix,float* chem3d,int chem3d_extent3,float* settle3d,int settle3d_extent3,float* vd3d,int vd3d_extent2,float* frp_mean,float* emis_ant_no) {
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
MynnArray<float,3> a_tk;a_tk.data=tk;
a_tk.lo[0]=ims;a_tk.hi[0]=ime;a_tk.stride[0]=1;
a_tk.lo[1]=kms;a_tk.hi[1]=kme;a_tk.stride[1]=1*(ime-ims+1);
a_tk.lo[2]=jms;a_tk.hi[2]=jme;a_tk.stride[2]=1*(ime-ims+1)*(kme-kms+1);
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
MynnArray<float,3> a_sqv;a_sqv.data=sqv;
a_sqv.lo[0]=ims;a_sqv.hi[0]=ime;a_sqv.stride[0]=1;
a_sqv.lo[1]=kms;a_sqv.hi[1]=kme;a_sqv.stride[1]=1*(ime-ims+1);
a_sqv.lo[2]=jms;a_sqv.hi[2]=jme;a_sqv.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_sqc;a_sqc.data=sqc;
a_sqc.lo[0]=ims;a_sqc.hi[0]=ime;a_sqc.stride[0]=1;
a_sqc.lo[1]=kms;a_sqc.hi[1]=kme;a_sqc.stride[1]=1*(ime-ims+1);
a_sqc.lo[2]=jms;a_sqc.hi[2]=jme;a_sqc.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_sqi;a_sqi.data=sqi;
a_sqi.lo[0]=ims;a_sqi.hi[0]=ime;a_sqi.stride[0]=1;
a_sqi.lo[1]=kms;a_sqi.hi[1]=kme;a_sqi.stride[1]=1*(ime-ims+1);
a_sqi.lo[2]=jms;a_sqi.hi[2]=jme;a_sqi.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_sqs;a_sqs.data=sqs;
a_sqs.lo[0]=ims;a_sqs.hi[0]=ime;a_sqs.stride[0]=1;
a_sqs.lo[1]=kms;a_sqs.hi[1]=kme;a_sqs.stride[1]=1*(ime-ims+1);
a_sqs.lo[2]=jms;a_sqs.hi[2]=jme;a_sqs.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qnc;a_qnc.data=qnc;
a_qnc.lo[0]=ims;a_qnc.hi[0]=ime;a_qnc.stride[0]=1;
a_qnc.lo[1]=kms;a_qnc.hi[1]=kme;a_qnc.stride[1]=1*(ime-ims+1);
a_qnc.lo[2]=jms;a_qnc.hi[2]=jme;a_qnc.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qni;a_qni.data=qni;
a_qni.lo[0]=ims;a_qni.hi[0]=ime;a_qni.stride[0]=1;
a_qni.lo[1]=kms;a_qni.hi[1]=kme;a_qni.stride[1]=1*(ime-ims+1);
a_qni.lo[2]=jms;a_qni.hi[2]=jme;a_qni.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qnifa;a_qnifa.data=qnifa;
a_qnifa.lo[0]=ims;a_qnifa.hi[0]=ime;a_qnifa.stride[0]=1;
a_qnifa.lo[1]=kms;a_qnifa.hi[1]=kme;a_qnifa.stride[1]=1*(ime-ims+1);
a_qnifa.lo[2]=jms;a_qnifa.hi[2]=jme;a_qnifa.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qnwfa;a_qnwfa.data=qnwfa;
a_qnwfa.lo[0]=ims;a_qnwfa.hi[0]=ime;a_qnwfa.stride[0]=1;
a_qnwfa.lo[1]=kms;a_qnwfa.hi[1]=kme;a_qnwfa.stride[1]=1*(ime-ims+1);
a_qnwfa.lo[2]=jms;a_qnwfa.hi[2]=jme;a_qnwfa.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qnbca;a_qnbca.data=qnbca;
a_qnbca.lo[0]=ims;a_qnbca.hi[0]=ime;a_qnbca.stride[0]=1;
a_qnbca.lo[1]=kms;a_qnbca.hi[1]=kme;a_qnbca.stride[1]=1*(ime-ims+1);
a_qnbca.lo[2]=jms;a_qnbca.hi[2]=jme;a_qnbca.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_qoz;a_qoz.data=qoz;
a_qoz.lo[0]=ims;a_qoz.hi[0]=ime;a_qoz.stride[0]=1;
a_qoz.lo[1]=kms;a_qoz.hi[1]=kme;a_qoz.stride[1]=1*(ime-ims+1);
a_qoz.lo[2]=jms;a_qoz.hi[2]=jme;a_qoz.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rthraten;a_rthraten.data=rthraten;
a_rthraten.lo[0]=ims;a_rthraten.hi[0]=ime;a_rthraten.stride[0]=1;
a_rthraten.lo[1]=kms;a_rthraten.hi[1]=kme;a_rthraten.stride[1]=1*(ime-ims+1);
a_rthraten.lo[2]=jms;a_rthraten.hi[2]=jme;a_rthraten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_pint;a_pint.data=pint;
a_pint.lo[0]=ims;a_pint.hi[0]=ime;a_pint.stride[0]=1;
a_pint.lo[1]=kms;a_pint.hi[1]=kme;a_pint.stride[1]=1*(ime-ims+1);
a_pint.lo[2]=jms;a_pint.hi[2]=jme;a_pint.stride[2]=1*(ime-ims+1)*(kme-kms+1);
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
MynnArray<float,3> a_rqncblten;a_rqncblten.data=rqncblten;
a_rqncblten.lo[0]=ims;a_rqncblten.hi[0]=ime;a_rqncblten.stride[0]=1;
a_rqncblten.lo[1]=kms;a_rqncblten.hi[1]=kme;a_rqncblten.stride[1]=1*(ime-ims+1);
a_rqncblten.lo[2]=jms;a_rqncblten.hi[2]=jme;a_rqncblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rqniblten;a_rqniblten.data=rqniblten;
a_rqniblten.lo[0]=ims;a_rqniblten.hi[0]=ime;a_rqniblten.stride[0]=1;
a_rqniblten.lo[1]=kms;a_rqniblten.hi[1]=kme;a_rqniblten.stride[1]=1*(ime-ims+1);
a_rqniblten.lo[2]=jms;a_rqniblten.hi[2]=jme;a_rqniblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rqnifablten;a_rqnifablten.data=rqnifablten;
a_rqnifablten.lo[0]=ims;a_rqnifablten.hi[0]=ime;a_rqnifablten.stride[0]=1;
a_rqnifablten.lo[1]=kms;a_rqnifablten.hi[1]=kme;a_rqnifablten.stride[1]=1*(ime-ims+1);
a_rqnifablten.lo[2]=jms;a_rqnifablten.hi[2]=jme;a_rqnifablten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rqnwfablten;a_rqnwfablten.data=rqnwfablten;
a_rqnwfablten.lo[0]=ims;a_rqnwfablten.hi[0]=ime;a_rqnwfablten.stride[0]=1;
a_rqnwfablten.lo[1]=kms;a_rqnwfablten.hi[1]=kme;a_rqnwfablten.stride[1]=1*(ime-ims+1);
a_rqnwfablten.lo[2]=jms;a_rqnwfablten.hi[2]=jme;a_rqnwfablten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rqnbcablten;a_rqnbcablten.data=rqnbcablten;
a_rqnbcablten.lo[0]=ims;a_rqnbcablten.hi[0]=ime;a_rqnbcablten.stride[0]=1;
a_rqnbcablten.lo[1]=kms;a_rqnbcablten.hi[1]=kme;a_rqnbcablten.stride[1]=1*(ime-ims+1);
a_rqnbcablten.lo[2]=jms;a_rqnbcablten.hi[2]=jme;a_rqnbcablten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,3> a_rqozblten;a_rqozblten.data=rqozblten;
a_rqozblten.lo[0]=ims;a_rqozblten.hi[0]=ime;a_rqozblten.stride[0]=1;
a_rqozblten.lo[1]=kms;a_rqozblten.hi[1]=kme;a_rqozblten.stride[1]=1*(ime-ims+1);
a_rqozblten.lo[2]=jms;a_rqozblten.hi[2]=jme;a_rqozblten.stride[2]=1*(ime-ims+1)*(kme-kms+1);
MynnArray<float,2> a_pblh;a_pblh.data=pblh;
a_pblh.lo[0]=ims;a_pblh.hi[0]=ime;a_pblh.stride[0]=1;
a_pblh.lo[1]=jms;a_pblh.hi[1]=jme;a_pblh.stride[1]=1*(ime-ims+1);
MynnArray<int,2> a_kpbl;a_kpbl.data=kpbl;
a_kpbl.lo[0]=ims;a_kpbl.hi[0]=ime;a_kpbl.stride[0]=1;
a_kpbl.lo[1]=jms;a_kpbl.hi[1]=jme;a_kpbl.stride[1]=1*(ime-ims+1);
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
MynnArray<float,2> a_lwp;a_lwp.data=lwp;
a_lwp.lo[0]=ims;a_lwp.hi[0]=ime;a_lwp.stride[0]=1;
a_lwp.lo[1]=jms;a_lwp.hi[1]=jme;a_lwp.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_iwp;a_iwp.data=iwp;
a_iwp.lo[0]=ims;a_iwp.hi[0]=ime;a_iwp.stride[0]=1;
a_iwp.lo[1]=jms;a_iwp.hi[1]=jme;a_iwp.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_swp;a_swp.data=swp;
a_swp.lo[0]=ims;a_swp.hi[0]=ime;a_swp.stride[0]=1;
a_swp.lo[1]=jms;a_swp.hi[1]=jme;a_swp.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_wspd10;a_wspd10.data=wspd10;
a_wspd10.lo[0]=ims;a_wspd10.hi[0]=ime;a_wspd10.stride[0]=1;
a_wspd10.lo[1]=jms;a_wspd10.hi[1]=jme;a_wspd10.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_wspd80;a_wspd80.data=wspd80;
a_wspd80.lo[0]=ims;a_wspd80.hi[0]=ime;a_wspd80.stride[0]=1;
a_wspd80.lo[1]=jms;a_wspd80.hi[1]=jme;a_wspd80.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_wspd160;a_wspd160.data=wspd160;
a_wspd160.lo[0]=ims;a_wspd160.hi[0]=ime;a_wspd160.stride[0]=1;
a_wspd160.lo[1]=jms;a_wspd160.hi[1]=jme;a_wspd160.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_cldceil;a_cldceil.data=cldceil;
a_cldceil.lo[0]=ims;a_cldceil.hi[0]=ime;a_cldceil.stride[0]=1;
a_cldceil.lo[1]=jms;a_cldceil.hi[1]=jme;a_cldceil.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_maxcldfra;a_maxcldfra.data=maxcldfra;
a_maxcldfra.lo[0]=ims;a_maxcldfra.hi[0]=ime;a_maxcldfra.stride[0]=1;
a_maxcldfra.lo[1]=jms;a_maxcldfra.hi[1]=jme;a_maxcldfra.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_maxcldfra_pbl;a_maxcldfra_pbl.data=maxcldfra_pbl;
a_maxcldfra_pbl.lo[0]=ims;a_maxcldfra_pbl.hi[0]=ime;a_maxcldfra_pbl.stride[0]=1;
a_maxcldfra_pbl.lo[1]=jms;a_maxcldfra_pbl.hi[1]=jme;a_maxcldfra_pbl.stride[1]=1*(ime-ims+1);
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
a_chem3d.lo[3]=1;a_chem3d.hi[3]=(1)+chem3d_extent3-1;a_chem3d.stride[3]=1*(ime-ims+1)*(kme-kms+1)*(jme-jms+1);
MynnArray<float,4> a_settle3d;a_settle3d.data=settle3d;
a_settle3d.lo[0]=ims;a_settle3d.hi[0]=ime;a_settle3d.stride[0]=1;
a_settle3d.lo[1]=kms;a_settle3d.hi[1]=kme;a_settle3d.stride[1]=1*(ime-ims+1);
a_settle3d.lo[2]=jms;a_settle3d.hi[2]=jme;a_settle3d.stride[2]=1*(ime-ims+1)*(kme-kms+1);
a_settle3d.lo[3]=1;a_settle3d.hi[3]=(1)+settle3d_extent3-1;a_settle3d.stride[3]=1*(ime-ims+1)*(kme-kms+1)*(jme-jms+1);
MynnArray<float,3> a_vd3d;a_vd3d.data=vd3d;
a_vd3d.lo[0]=ims;a_vd3d.hi[0]=ime;a_vd3d.stride[0]=1;
a_vd3d.lo[1]=jms;a_vd3d.hi[1]=jme;a_vd3d.stride[1]=1*(ime-ims+1);
a_vd3d.lo[2]=1;a_vd3d.hi[2]=(1)+vd3d_extent2-1;a_vd3d.stride[2]=1*(ime-ims+1)*(jme-jms+1);
MynnArray<float,2> a_frp_mean;a_frp_mean.data=frp_mean;
a_frp_mean.lo[0]=ims;a_frp_mean.hi[0]=ime;a_frp_mean.stride[0]=1;
a_frp_mean.lo[1]=jms;a_frp_mean.hi[1]=jme;a_frp_mean.stride[1]=1*(ime-ims+1);
MynnArray<float,2> a_emis_ant_no;a_emis_ant_no.data=emis_ant_no;
a_emis_ant_no.lo[0]=ims;a_emis_ant_no.hi[0]=ime;a_emis_ant_no.stride[0]=1;
a_emis_ant_no.lo[1]=jms;a_emis_ant_no.hi[1]=jme;a_emis_ant_no.stride[1]=1*(ime-ims+1);
column.f_mynnedmf_driver(ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,(f_qc!=0),(f_qi!=0),(f_qs!=0),(f_qoz!=0),(f_qnc!=0),(f_qni!=0),(f_qnifa!=0),(f_qnwfa!=0),(f_qnbca!=0),initflag,(restart!=0),(cycling!=0),delt,a_dx,a_xland,a_ps,a_ts,a_qsfc,a_ust,a_ch,a_hfx,a_qfx,a_wspd,a_znt,a_uoce,a_voce,a_dz,a_u,a_v,a_w,a_th,a_tk,a_p,a_exner,a_rho,a_qv,a_qc,a_qi,a_qs,a_sqv,a_sqc,a_sqi,a_sqs,a_qnc,a_qni,a_qnifa,a_qnwfa,a_qnbca,a_qoz,a_rthraten,a_pint,a_cldfra_bl,a_qc_bl,a_qi_bl,a_qke,a_qke_adv,a_tsq,a_qsq,a_cov,a_el_pbl,a_rublten,a_rvblten,a_rthblten,a_rqvblten,a_rqcblten,a_rqiblten,a_rqsblten,a_rqncblten,a_rqniblten,a_rqnifablten,a_rqnwfablten,a_rqnbcablten,a_rqozblten,a_pblh,a_kpbl,a_maxwidth,a_maxmf,a_ztop_plume,a_excess_h,a_excess_q,a_maxwidth_dd,a_maxmf_dd,a_maxtkeprod,a_cldtop_cooling,a_ent_eff,a_lwp,a_iwp,a_swp,a_wspd10,a_wspd80,a_wspd160,a_cldceil,a_maxcldfra,a_maxcldfra_pbl,a_edmf_a,a_edmf_w,a_edmf_qt,a_edmf_thl,a_edmf_ent,a_edmf_qc,a_sub_thl,a_sub_sqv,a_det_thl,a_det_sqv,a_exch_h,a_exch_m,a_dqke,a_qwt,a_qshear,a_qbuoy,a_qdiss,a_sh3d,a_sm3d,spp_pbl,a_pattern_spp,(bl_mynn_tkeadvect!=0),tke_budget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_edmf,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_edmf_dd,bl_mynn_ess,bl_mynn_diags2d,bl_mynn_diags3d,(present_mix_chem ? MynnOptional<bool>(mix_chem) : MynnOptional<bool>()),(present_nchem ? MynnOptional<int>(nchem) : MynnOptional<int>()),(present_ndvel ? MynnOptional<int>(ndvel) : MynnOptional<int>()),(present_enh_mix ? MynnOptional<bool>(enh_mix) : MynnOptional<bool>()),a_chem3d,a_settle3d,a_vd3d,a_frp_mean,a_emis_ant_no,message,error_flag);
error_flags[index]=error_flag;
}
extern "C" __global__ void rrfs_mynnedmf_run_l1(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_qnc,int f_qni,int f_qnifa,int f_qnwfa,int f_qnbca,int initflag,int restart,int cycling,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tk,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* sqv,float* sqc,float* sqi,float* sqs,float* qnc,float* qni,float* qnifa,float* qnwfa,float* qnbca,float* qoz,float* rthraten,float* pint,float* cldfra_bl,float* qc_bl,float* qi_bl,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rqncblten,float* rqniblten,float* rqnifablten,float* rqnwfablten,float* rqnbcablten,float* rqozblten,float* pblh,int* kpbl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* maxwidth_dd,float* maxmf_dd,float* maxtkeprod,float* cldtop_cooling,float* ent_eff,float* lwp,float* iwp,float* swp,float* wspd10,float* wspd80,float* wspd160,float* cldceil,float* maxcldfra,float* maxcldfra_pbl,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int tke_budget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_edmf,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_edmf_dd,int bl_mynn_ess,int bl_mynn_diags2d,int bl_mynn_diags3d,int mix_chem,int present_mix_chem,int nchem,int present_nchem,int ndvel,int present_ndvel,int enh_mix,int present_enh_mix,float* chem3d,int chem3d_extent3,float* settle3d,int settle3d_extent3,float* vd3d,int vd3d_extent2,float* frp_mean,float* emis_ant_no) { rrfs_mynnedmf_column<1>(batch_first,batch_count,workspace,workspace_stride,error_flags,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,f_qc,f_qi,f_qs,f_qoz,f_qnc,f_qni,f_qnifa,f_qnwfa,f_qnbca,initflag,restart,cycling,delt,dx,xland,ps,ts,qsfc,ust,ch,hfx,qfx,wspd,znt,uoce,voce,dz,u,v,w,th,tk,p,exner,rho,qv,qc,qi,qs,sqv,sqc,sqi,sqs,qnc,qni,qnifa,qnwfa,qnbca,qoz,rthraten,pint,cldfra_bl,qc_bl,qi_bl,qke,qke_adv,tsq,qsq,cov,el_pbl,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqsblten,rqncblten,rqniblten,rqnifablten,rqnwfablten,rqnbcablten,rqozblten,pblh,kpbl,maxwidth,maxmf,ztop_plume,excess_h,excess_q,maxwidth_dd,maxmf_dd,maxtkeprod,cldtop_cooling,ent_eff,lwp,iwp,swp,wspd10,wspd80,wspd160,cldceil,maxcldfra,maxcldfra_pbl,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc,sub_thl,sub_sqv,det_thl,det_sqv,exch_h,exch_m,dqke,qwt,qshear,qbuoy,qdiss,sh3d,sm3d,spp_pbl,pattern_spp,bl_mynn_tkeadvect,tke_budget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_edmf,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_edmf_dd,bl_mynn_ess,bl_mynn_diags2d,bl_mynn_diags3d,mix_chem,present_mix_chem,nchem,present_nchem,ndvel,present_ndvel,enh_mix,present_enh_mix,chem3d,chem3d_extent3,settle3d,settle3d_extent3,vd3d,vd3d_extent2,frp_mean,emis_ant_no); }
extern "C" __global__ MYNN_L4_REGS void rrfs_mynnedmf_run_l4(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_qnc,int f_qni,int f_qnifa,int f_qnwfa,int f_qnbca,int initflag,int restart,int cycling,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tk,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* sqv,float* sqc,float* sqi,float* sqs,float* qnc,float* qni,float* qnifa,float* qnwfa,float* qnbca,float* qoz,float* rthraten,float* pint,float* cldfra_bl,float* qc_bl,float* qi_bl,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rqncblten,float* rqniblten,float* rqnifablten,float* rqnwfablten,float* rqnbcablten,float* rqozblten,float* pblh,int* kpbl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* maxwidth_dd,float* maxmf_dd,float* maxtkeprod,float* cldtop_cooling,float* ent_eff,float* lwp,float* iwp,float* swp,float* wspd10,float* wspd80,float* wspd160,float* cldceil,float* maxcldfra,float* maxcldfra_pbl,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int tke_budget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_edmf,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_edmf_dd,int bl_mynn_ess,int bl_mynn_diags2d,int bl_mynn_diags3d,int mix_chem,int present_mix_chem,int nchem,int present_nchem,int ndvel,int present_ndvel,int enh_mix,int present_enh_mix,float* chem3d,int chem3d_extent3,float* settle3d,int settle3d_extent3,float* vd3d,int vd3d_extent2,float* frp_mean,float* emis_ant_no) { rrfs_mynnedmf_column<4>(batch_first,batch_count,workspace,workspace_stride,error_flags,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,f_qc,f_qi,f_qs,f_qoz,f_qnc,f_qni,f_qnifa,f_qnwfa,f_qnbca,initflag,restart,cycling,delt,dx,xland,ps,ts,qsfc,ust,ch,hfx,qfx,wspd,znt,uoce,voce,dz,u,v,w,th,tk,p,exner,rho,qv,qc,qi,qs,sqv,sqc,sqi,sqs,qnc,qni,qnifa,qnwfa,qnbca,qoz,rthraten,pint,cldfra_bl,qc_bl,qi_bl,qke,qke_adv,tsq,qsq,cov,el_pbl,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqsblten,rqncblten,rqniblten,rqnifablten,rqnwfablten,rqnbcablten,rqozblten,pblh,kpbl,maxwidth,maxmf,ztop_plume,excess_h,excess_q,maxwidth_dd,maxmf_dd,maxtkeprod,cldtop_cooling,ent_eff,lwp,iwp,swp,wspd10,wspd80,wspd160,cldceil,maxcldfra,maxcldfra_pbl,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc,sub_thl,sub_sqv,det_thl,det_sqv,exch_h,exch_m,dqke,qwt,qshear,qbuoy,qdiss,sh3d,sm3d,spp_pbl,pattern_spp,bl_mynn_tkeadvect,tke_budget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_edmf,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_edmf_dd,bl_mynn_ess,bl_mynn_diags2d,bl_mynn_diags3d,mix_chem,present_mix_chem,nchem,present_nchem,ndvel,present_ndvel,enh_mix,present_enh_mix,chem3d,chem3d_extent3,settle3d,settle3d_extent3,vd3d,vd3d_extent2,frp_mean,emis_ant_no); }
extern "C" __global__ void rrfs_mynnedmf_run_l16(int batch_first,int batch_count,unsigned char* workspace,unsigned long long workspace_stride,int* error_flags,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,int f_qc,int f_qi,int f_qs,int f_qoz,int f_qnc,int f_qni,int f_qnifa,int f_qnwfa,int f_qnbca,int initflag,int restart,int cycling,float delt,float* dx,float* xland,float* ps,float* ts,float* qsfc,float* ust,float* ch,float* hfx,float* qfx,float* wspd,float* znt,float* uoce,float* voce,float* dz,float* u,float* v,float* w,float* th,float* tk,float* p,float* exner,float* rho,float* qv,float* qc,float* qi,float* qs,float* sqv,float* sqc,float* sqi,float* sqs,float* qnc,float* qni,float* qnifa,float* qnwfa,float* qnbca,float* qoz,float* rthraten,float* pint,float* cldfra_bl,float* qc_bl,float* qi_bl,float* qke,float* qke_adv,float* tsq,float* qsq,float* cov,float* el_pbl,float* rublten,float* rvblten,float* rthblten,float* rqvblten,float* rqcblten,float* rqiblten,float* rqsblten,float* rqncblten,float* rqniblten,float* rqnifablten,float* rqnwfablten,float* rqnbcablten,float* rqozblten,float* pblh,int* kpbl,float* maxwidth,float* maxmf,float* ztop_plume,float* excess_h,float* excess_q,float* maxwidth_dd,float* maxmf_dd,float* maxtkeprod,float* cldtop_cooling,float* ent_eff,float* lwp,float* iwp,float* swp,float* wspd10,float* wspd80,float* wspd160,float* cldceil,float* maxcldfra,float* maxcldfra_pbl,float* edmf_a,float* edmf_w,float* edmf_qt,float* edmf_thl,float* edmf_ent,float* edmf_qc,float* sub_thl,float* sub_sqv,float* det_thl,float* det_sqv,float* exch_h,float* exch_m,float* dqke,float* qwt,float* qshear,float* qbuoy,float* qdiss,float* sh3d,float* sm3d,int spp_pbl,float* pattern_spp,int bl_mynn_tkeadvect,int tke_budget,int bl_mynn_cloudpdf,int bl_mynn_mixlength,float bl_mynn_closure,int bl_mynn_edmf,int bl_mynn_edmf_mom,int bl_mynn_edmf_tke,int bl_mynn_mixscalars,int bl_mynn_mixaerosols,int bl_mynn_mixnumcon,int bl_mynn_cloudmix,int bl_mynn_mixqt,int bl_mynn_edmf_dd,int bl_mynn_ess,int bl_mynn_diags2d,int bl_mynn_diags3d,int mix_chem,int present_mix_chem,int nchem,int present_nchem,int ndvel,int present_ndvel,int enh_mix,int present_enh_mix,float* chem3d,int chem3d_extent3,float* settle3d,int settle3d_extent3,float* vd3d,int vd3d_extent2,float* frp_mean,float* emis_ant_no) { rrfs_mynnedmf_column<16>(batch_first,batch_count,workspace,workspace_stride,error_flags,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,f_qc,f_qi,f_qs,f_qoz,f_qnc,f_qni,f_qnifa,f_qnwfa,f_qnbca,initflag,restart,cycling,delt,dx,xland,ps,ts,qsfc,ust,ch,hfx,qfx,wspd,znt,uoce,voce,dz,u,v,w,th,tk,p,exner,rho,qv,qc,qi,qs,sqv,sqc,sqi,sqs,qnc,qni,qnifa,qnwfa,qnbca,qoz,rthraten,pint,cldfra_bl,qc_bl,qi_bl,qke,qke_adv,tsq,qsq,cov,el_pbl,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqsblten,rqncblten,rqniblten,rqnifablten,rqnwfablten,rqnbcablten,rqozblten,pblh,kpbl,maxwidth,maxmf,ztop_plume,excess_h,excess_q,maxwidth_dd,maxmf_dd,maxtkeprod,cldtop_cooling,ent_eff,lwp,iwp,swp,wspd10,wspd80,wspd160,cldceil,maxcldfra,maxcldfra_pbl,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc,sub_thl,sub_sqv,det_thl,det_sqv,exch_h,exch_m,dqke,qwt,qshear,qbuoy,qdiss,sh3d,sm3d,spp_pbl,pattern_spp,bl_mynn_tkeadvect,tke_budget,bl_mynn_cloudpdf,bl_mynn_mixlength,bl_mynn_closure,bl_mynn_edmf,bl_mynn_edmf_mom,bl_mynn_edmf_tke,bl_mynn_mixscalars,bl_mynn_mixaerosols,bl_mynn_mixnumcon,bl_mynn_cloudmix,bl_mynn_mixqt,bl_mynn_edmf_dd,bl_mynn_ess,bl_mynn_diags2d,bl_mynn_diags3d,mix_chem,present_mix_chem,nchem,present_nchem,ndvel,present_ndvel,enh_mix,present_enh_mix,chem3d,chem3d_extent3,settle3d,settle3d_extent3,vd3d,vd3d_extent2,frp_mean,emis_ant_no); }
'''

ARGUMENTS = {'ids': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ide': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jds': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jde': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kds': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kde': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ims': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ime': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jms': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jme': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kms': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kme': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'its': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ite': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jts': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'jte': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kts': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kte': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qc': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qi': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qs': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qoz': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qnc': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qni': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qnifa': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qnwfa': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'f_qnbca': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'initflag': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'restart': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'cycling': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'delt': {'kind': 'float', 'dimensions': [], 'intent': 'in', 'optional': False}, 'dx': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'xland': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ps': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ts': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'qsfc': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ust': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'ch': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'hfx': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'qfx': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'wspd': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'znt': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'uoce': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'voce': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False}, 'dz': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'u': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'v': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'w': {'kind': 'float', 'dimensions': ['ims:', 'kms:', 'jms:'], 'intent': 'in', 'optional': False}, 'th': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'tk': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'p': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'exner': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'rho': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'qv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qc': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qi': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qs': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'sqv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'sqc': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'sqi': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'sqs': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qnc': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qni': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qnifa': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qnwfa': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qnbca': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'qoz': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'rthraten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False}, 'pint': {'kind': 'float', 'dimensions': ['ims:', 'kms:', 'jms:'], 'intent': 'in', 'optional': False}, 'cldfra_bl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qc_bl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qi_bl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qke': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qke_adv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'tsq': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'qsq': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'cov': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'el_pbl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rublten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rvblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rthblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'rqvblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqcblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqiblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqsblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqncblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqniblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqnifablten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqnwfablten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqnbcablten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'rqozblten': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'pblh': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'kpbl': {'kind': 'int', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'maxwidth': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'maxmf': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'ztop_plume': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'excess_h': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'excess_q': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'maxwidth_dd': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'maxmf_dd': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'maxtkeprod': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'cldtop_cooling': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'ent_eff': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False}, 'lwp': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'iwp': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'swp': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'wspd10': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': True}, 'wspd80': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': True}, 'wspd160': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': True}, 'cldceil': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'maxcldfra': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': True}, 'maxcldfra_pbl': {'kind': 'float', 'dimensions': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': True}, 'edmf_a': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_w': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_qt': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_thl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_ent': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'edmf_qc': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'sub_thl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'sub_sqv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'det_thl': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'det_sqv': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': True}, 'exch_h': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': False}, 'exch_m': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': False}, 'dqke': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qwt': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qshear': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qbuoy': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'qdiss': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'out', 'optional': True}, 'sh3d': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'sm3d': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'inout', 'optional': False}, 'spp_pbl': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'pattern_spp': {'kind': 'float', 'dimensions': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': True}, 'bl_mynn_tkeadvect': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'tke_budget': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_cloudpdf': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixlength': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_closure': {'kind': 'float', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_mom': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_tke': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixscalars': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixaerosols': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixnumcon': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_cloudmix': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixqt': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_dd': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_ess': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_diags2d': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_diags3d': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'mix_chem': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': True}, 'nchem': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': True}, 'ndvel': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': True}, 'enh_mix': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': True}, 'chem3d': {'kind': 'float', 'dimensions': ['ims:', 'kms:', 'jms:', ':'], 'intent': 'inout', 'optional': True}, 'settle3d': {'kind': 'float', 'dimensions': ['ims:', 'kms:', 'jms:', ':'], 'intent': 'inout', 'optional': True}, 'vd3d': {'kind': 'float', 'dimensions': ['ims:', 'jms:', ':'], 'intent': 'in', 'optional': True}, 'frp_mean': {'kind': 'float', 'dimensions': ['ims:', 'jms:'], 'intent': 'in', 'optional': True}, 'emis_ant_no': {'kind': 'float', 'dimensions': ['ims:', 'jms:'], 'intent': 'in', 'optional': True}, 'errmsg': {'kind': 'char', 'dimensions': [], 'intent': 'out', 'optional': False}, 'errflg': {'kind': 'int', 'dimensions': [], 'intent': 'out', 'optional': False}}
WORKSPACE_ARRAYS = [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kte']}, {'kind': 'float', 'dimensions': ['kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte', '1:nup']}, {'kind': 'float', 'dimensions': ['nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nscalars']}, {'kind': 'int', 'dimensions': ['1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'int', 'dimensions': ['1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['ndvel1']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem1']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem1']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}]
WORKSPACE_FRAMES = {'mynnedmf_diags2d': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['cloud_ceiling', 'cloud_water_path', 'wspd_at_hgts']}, 'cloud_water_path': {'arrays': [], 'calls': []}, 'wspd_at_hgts': {'arrays': [], 'calls': ['interpolate_wind']}, 'cloud_ceiling': {'arrays': [], 'calls': []}, 'interpolate_wind': {'arrays': [], 'calls': []}, 'mynnedmf': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['ddmp_mf', 'dmp_mf', 'get_pblh', 'mym_condensation', 'mym_initialize', 'mym_predict', 'mym_turbulence', 'mynn_mix_chem', 'mynn_mix_scalars', 'mynn_tendencies', 'phih', 'phim', 'retrieve_exchange_coeffs', 'scale_aware', 'topdown_cloudrad']}, 'mym_initialize': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['mym_length', 'mym_level2']}, 'mym_level2': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['qsat_blend', 'xl_blend']}, 'mym_length': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['boulac_length']}, 'boulac_length': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': []}, 'mym_turbulence': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['mym_length', 'mym_level2']}, 'mym_predict': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['tridiag2']}, 'mym_condensation': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['esat_blend', 'qsat_blend', 'xl_blend']}, 'mynn_tendencies': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['moisture_check', 'qsat_blend', 'tridiag2']}, 'moisture_check': {'arrays': [], 'calls': []}, 'mynn_mix_chem': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['tridiag3']}, 'mynn_mix_scalars': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['tridiag3']}, 'retrieve_exchange_coeffs': {'arrays': [], 'calls': []}, 'tridiag2': {'arrays': [{'kind': 'float', 'dimensions': ['kte']}, {'kind': 'float', 'dimensions': ['kte']}], 'calls': []}, 'tridiag3': {'arrays': [], 'calls': []}, 'get_pblh': {'arrays': [], 'calls': []}, 'dmp_mf': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte', '1:nup']}, {'kind': 'float', 'dimensions': ['nchem']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nchem']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem']}, {'kind': 'float', 'dimensions': ['nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:nup', 'nscalars']}, {'kind': 'int', 'dimensions': ['1:nup']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}], 'calls': ['condensation_edmf', 'esat_blend', 'qsat_blend', 'xl_blend']}, 'condensation_edmf': {'arrays': [], 'calls': ['qsat_blend']}, 'condensation_ddmf': {'arrays': [], 'calls': ['qsat_blend']}, 'ddmp_mf': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'int', 'dimensions': ['1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte+1', '1:ndd']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['condensation_ddmf', 'qsat_blend']}, 'scale_aware': {'arrays': [], 'calls': []}, 'esat_blend': {'arrays': [], 'calls': []}, 'qsat_blend': {'arrays': [], 'calls': []}, 'xl_blend': {'arrays': [], 'calls': []}, 'phim': {'arrays': [], 'calls': []}, 'phih': {'arrays': [], 'calls': []}, 'topdown_cloudrad': {'arrays': [{'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': []}, 'mynnedmf_driver': {'arrays': [{'kind': 'float', 'dimensions': ['ndvel1']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem1']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nchem1']}, {'kind': 'float', 'dimensions': ['kts:kte', 'nscalars']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte+1']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}, {'kind': 'float', 'dimensions': ['kts:kte']}], 'calls': ['mynnedmf', 'mynnedmf_diags2d', 'mynnedmf_post_run', 'mynnedmf_qv_to_sqv', 'mynnedmf_sqv_to_qv']}, 'mynnedmf_qv_to_sqv': {'arrays': [], 'calls': []}, 'mynnedmf_sqv_to_qv': {'arrays': [{'kind': 'float', 'dimensions': ['kte']}], 'calls': []}, 'mynnedmf_post_run': {'arrays': [{'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}, {'kind': 'float', 'dimensions': ['1:kte']}], 'calls': []}}
# END GENERATED RRFS MYNN-EDMF
# NCAR/MYNN-EDMF 2b2da61 option names as the unified driver takes them; the host Registry keys are 44c0849's
# (RRFS 44c0849 MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1593-1663, the host this route
# runs). The v6 argument mapping is the RRFS host that already calls the unified driver: ufs-community/MPAS-Model
# noaa/develop de016e0 src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:1748-1828. Options v6.1.1 removed
# (icloud_bl, scaleaware, dheat_opt, stfunc, topdown) are hard-wired upstream to the values hrrrv5 runs.
OPTION_ARGUMENTS = {
    'config_mynn_tkeadvect': ('bl_mynn_tkeadvect', False),
    'config_mynn_tkebudget': ('tke_budget', 0),
    'config_mynn_closure': ('bl_mynn_closure', 2.6),
    'config_mynn_cloudpdf': ('bl_mynn_cloudpdf', 2),
    'config_mynn_mixlength': ('bl_mynn_mixlength', 1),
    'config_mynn_edmf': ('bl_mynn_edmf', 1),
    'config_mynn_edmf_dd': ('bl_mynn_edmf_dd', 0),
    'config_mynn_edmf_mom': ('bl_mynn_edmf_mom', 1),
    'config_mynn_edmf_tke': ('bl_mynn_edmf_tke', 0),
    'config_mynn_edmf_output': ('bl_mynn_diags3d', 0),
    # Not a 44c0849 Registry option: the 2-D diagnostics module (lwp, iwp, swp, cldceil, wspd10/80/160) is off.
    'config_mynn_diags2d': ('bl_mynn_diags2d', 0),
    'config_mynn_mixscalars': ('bl_mynn_mixscalars', 0),
    'config_mynn_mixaerosols': ('bl_mynn_mixaerosols', 1),
    'config_mynn_mixnumcon': ('bl_mynn_mixnumcon', 1),
    'config_mynn_mixclouds': ('bl_mynn_cloudmix', 1),
    'config_mynn_mixqt': ('bl_mynn_mixqt', 0),
    'config_mynn_ess': ('bl_mynn_ess', 1),
    'config_mynn_enh_mix': ('enh_mix', False),
    'config_spp_pbl': ('spp_pbl', 0),
}

#: Memory upper bound of an assumed-shape dimension (generate_backend.ASSUMED_UPPER); a bare ':' is the actual's.
ASSUMED_UPPER = {'ims': 'ime', 'kms': 'kme', 'jms': 'jme'}

#: Host scalars the slot reads once (44c0849 mpas_atmphys_vars names), and the v6 driver argument each feeds.
HOST_FLAGS = {'f_qc': 'f_qc', 'f_qi': 'f_qi', 'f_qs': 'f_qs', 'f_qoz': 'f_qoz', 'f_nc': 'f_qnc', 'f_ni': 'f_qni',
              'f_nifa': 'f_qnifa', 'f_nwfa': 'f_qnwfa', 'f_nbca': 'f_qnbca'}
HOST_SCALARS = (*HOST_FLAGS, 'ids', 'ide', 'jds', 'jde', 'kds', 'kde', 'ims', 'ime', 'jms', 'jme',
                'kms', 'kme', 'jts', 'jte', 'kts', 'kte')


def option_arguments(config: Mapping) -> dict:
    """v6 driver options from the 44c0849 Registry keys (noaa/develop de016e0 driver_pbl.F:1748-1828)."""
    return {arg: config.get(name, default)
            for name, (arg, default) in OPTION_ARGUMENTS.items()}


def assumed_extents(metadata: Mapping) -> list[int]:
    """Dimensions of an argument whose extent the launch passes (a bare ':' in the v6 driver declaration)."""
    out = []
    for d, dim in enumerate(metadata['dimensions']):
        edges = dim.split(':')
        if len(edges) == 2 and not edges[1] and edges[0] not in ASSUMED_UPPER:
            out.append(d)
    return out


def argument_shape(metadata: Mapping, arguments: Mapping, value=None) -> tuple[int, ...]:
    """Fortran extents; NCAR/MYNN-EDMF 2b2da61 src/module_bl_mynnedmf_driver.F90:142-415 (assumed-shape dummies
    span the MPAS memory bounds; a bare ':' dimension takes the actual's extent)."""
    sizes = []
    for d, dim in enumerate(metadata['dimensions']):
        edges = dim.split(':')
        if len(edges) == 2 and not edges[1]:
            if edges[0] in ASSUMED_UPPER:
                edges = [edges[0], ASSUMED_UPPER[edges[0]]]
            else:
                if value is None:
                    raise ValueError('an assumed-shape extent needs the actual array')
                sizes.append(int(value.shape[d]))
                continue
        lower = int(arguments[edges[0]]) if len(edges) == 2 else 1
        upper = int(arguments[edges[-1]])
        sizes.append(upper - lower + 1)
    return tuple(sizes)


#: Device memory one MYNN-EDMF launch's automatic-storage arena may take (WOOF_HEX_MYNNEDMF_ARENA_MB overrides).
ARENA_BUDGET_BYTES = 2 << 30


#: Lane widths compiled into the image (generate_backend.LANE_WIDTHS): one kernel rrfs_mynnedmf_run_l<W> each.
LANE_WIDTHS = (1, 4, 16)


def lanes_per_column(kernels, ncols: int) -> int:
    """Lane width per launch: the pinned module's measured rule (bl_mynnedmf_c6d08709.lanes_per_column)."""
    import os
    forced = int(os.environ.get('WOOF_HEX_MYNNEDMF_LANES', '0') or 0)
    if forced:
        if forced not in LANE_WIDTHS:
            raise ValueError(f'WOOF_HEX_MYNNEDMF_LANES must be one of {LANE_WIDTHS}')
        return forced
    import cupy as cp
    sms = int(cp.cuda.Device().attributes['MultiProcessorCount'])
    fast = _hex_numerics.mode() == 'fast'
    for width in ((16,) if fast else sorted(LANE_WIDTHS, reverse=True)):
        from woof.backend import active_blocks_per_multiprocessor
        capacity = sms * max(int(active_blocks_per_multiprocessor(kernels[width].ptr, 32, 0)), 1)
        if -(-ncols * width // 32) <= capacity:
            return width
    return 4 if fast else 1


def automatic_batch_columns(ncols: int, stride: int) -> int:
    """Columns per launch: every column when the arenas fit the budget (bl_mynnedmf_c6d08709's rule)."""
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
    """Largest live arena path; NCAR/MYNN-EDMF 2b2da61 src/module_bl_mynnedmf_driver.F90:351-436 plus the core.

    The driver's ALLOCATEs (vd1, chem1, settle1 per column; dqke1..qdiss1 when tke_budget=1) are in the bound at
    their largest extent: ndvel1/nchem1 are ndvel/nchem with chemistry mixing on and 1 without it.
    """
    from math import prod
    nchem = int(arguments.get('nchem') or 0)
    ndvel = int(arguments.get('ndvel') or 0)
    bounds = {n: int(arguments[n]) for n in ('kts', 'kte', 'kme')}
    # The core is called with nchem1/ndvel1 (src/module_bl_mynnedmf_driver.F90:675-693), at least 1 even with chemistry off, so its
    # (kts:kte, nchem) automatic arrays take one species. THE BREAKAGE THIS PREVENTS: bounding them with the host's
    # num_chem = 0 (hrrrv5) left the slot short by those arrays; plume columns wrote into the next warp tile's
    # column (x1.2562 hrrrv5 step 1, strict: 20 corrupted columns with NaN tendencies).
    bounds.update(nchem=max(nchem, 1), ndvel=max(ndvel, 1), nchem1=max(nchem, 1), ndvel1=max(ndvel, 1),
                  n=max(bounds['kte'], bounds['kme']), nup=8, ndd=5, nscalars=1)
    def value(text):
        """Pinned extents only (sums of names and integers); no arbitrary eval."""
        parts = text.split('+')
        return sum(int(x) if x.isdecimal() else bounds[x] for x in parts)
    def frame_bytes(arrays):
        # Each array is rounded up to the 8-byte arena alignment on its own, so the bound holds for any order of
        # allocation. THE BREAKAGE THIS PREVENTS: the v6 driver ALLOCATEs vd1/chem1/settle1 after its automatic
        # arrays, not in declaration order, so the declaration-order sum (the pinned module's rule) fell a few
        # padding bytes short; a column on the deepest call path (plumes) wrote past its arena slot into the next
        # warp tile's column (x1.2562 hrrrv5 step 1, strict: 20 corrupted columns with NaN tendencies; a doubled
        # slot removed every difference).
        total = 0
        for var in arrays:
            extents = []
            for dim in var['dimensions']:
                edges = dim.split(':')
                lo = value(edges[0]) if len(edges) == 2 else 1
                extents.append(max(0, value(edges[-1]) - lo + 1))
            size = {'float': 4, 'int': 4, 'double': 8, 'bool': 1}[var['kind']]
            total += (size * prod(extents) + 7) & ~7
        return total
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


class MynnEdmfCuda:
    """Launch the complete NCAR/MYNN-EDMF v6.1.1 MPAS driver on resident device arrays.

    As bl_mynnedmf_c6d08709.MynnEdmfCuda: shared glibc device maths, Fortran (i,k,j[,species]) layout, NVRTC
    directly (--ftz=false), one kernel per lane width. Assumed-shape dummies with a bare ':' dimension (chem3d,
    settle3d, vd3d) take the actual's extent, passed after its pointer.
    """

    def __init__(self, shared_math_source: str | None = None, *, batch_columns: int | None = None,
                 compiled_image: bytes | None = None):
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
        import cupy as cp
        from cupy_backends.cuda.libs import nvrtc
        from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
        source = CUDA_FTZ_HELPERS + self.shared_math_source + ARRAY_SUPPORT + MODULE_CUDA
        if self.compiled_image is not None:
            self._module = cp.cuda.function.Module()
            self._module.load(self.compiled_image)
            self._load_kernels()
            return
        from woof.hip_backend import active
        if active():
            from woof.hex.cuda_backend.aot import raw_image
            ptx = raw_image(source, _hex_numerics.options(CUDA_OPTIONS),
                            _compilation_target(xp=cp), filename='rrfs_mynnedmf_v611.cu')
        else:
            options = _hex_numerics.options(CUDA_OPTIONS) + ('--gpu-architecture=sm_' + cp.cuda.Device().compute_capability,)
            program = nvrtc.createProgram(source, 'rrfs_mynnedmf_v611.cu', (), ())
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
        self._kernels = {width: self._module.get_function(f'rrfs_mynnedmf_run_l{width}') for width in LANE_WIDTHS}
        self._kernel = self._kernels[1]

    def run(self, arguments: Mapping) -> dict:
        """Run the driver; NCAR/MYNN-EDMF 2b2da61 src/module_bl_mynnedmf_driver.F90:80-1012."""
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
                dtype = np.float32 if meta['kind'] == 'float' else np.int32
                passed = assumed_extents(meta)
                if value is None:
                    if meta['intent'] == 'out' and not meta['optional']:
                        value = cp.empty(argument_shape(meta, values), dtype=dtype, order='F')
                        values[name] = value
                    elif meta['optional']:
                        actuals.append(np.uint64(0))
                        actuals.extend(np.int32(0) for _ in passed)
                        continue
                    else:
                        raise ValueError(f'Missing MYNN driver argument {name}')
                shape = argument_shape(meta, values, value)
                if not isinstance(value, cp.ndarray) or value.dtype != dtype or value.shape != shape:
                    raise ValueError(f'{name} requires a resident {np.dtype(dtype)} array with shape {shape}')
                packed = cp.asfortranarray(value)
                actuals.append(packed)
                actuals.extend(np.int32(shape[d]) for d in passed)
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
        lane_width = self._lane_widths.get(launch_columns)
        if lane_width is None:
            lane_width = self._lane_widths[launch_columns] = lanes_per_column(self._kernels, launch_columns)
        self.last_lane_width = lane_width
        tile_columns = 32 // lane_width
        # Zeroed, as the pinned module (reproducible words for any read of never-written automatic storage).
        tiles = (min(ncols, batch_columns) + tile_columns - 1)//tile_columns
        workspace = cp.zeros(tiles*tile_columns*stride, dtype=cp.uint8)
        errors = cp.empty(ncols, dtype=cp.int32)
        for first in range(0, ncols, batch_columns):
            count = min(batch_columns, ncols-first)
            self._kernels[lane_width](((count*lane_width+31)//32,), (32,),
                                      (np.int32(first), np.int32(count), workspace,
                                       np.uint64(stride), errors, *actuals))
        for original, packed in copies:
            original[...] = packed
        return {**{n: values[n] for n, m in ARGUMENTS.items()
                   if m['intent'] in ('out', 'inout') and m['dimensions'] and values.get(n) is not None},
                'errflg': errors}


# RRFS 44c0849 host arrays (mpas_atmphys_vars) for each v6 driver argument, as noaa/develop de016e0
# mpas_atmphys_driver_pbl.F:1748-1828 binds them: tk = t_p, pint = pres2_hyd_p, qn* = n*_p, rqn*blten = rn*blten_p.
# The 2-D diagnostics (lwp ... maxcldfra_pbl) are not 44c0849 arrays; with bl_mynn_diags2d = 0 the driver never
# references them, so they are absent.
NATIVE_ARRAY_BINDINGS = {
    'dx': 'dx_p', 'xland': 'xland_p', 'ps': 'psfc_p', 'ts': 'tsk_p', 'qsfc': 'qsfc_p', 'ust': 'ust_p', 'ch': 'ch_p',
    'hfx': 'hfx_p', 'qfx': 'qfx_p', 'wspd': 'wspd_p', 'znt': 'znt_p', 'uoce': 'uoce_p', 'voce': 'voce_p',
    'dz': 'dz_p', 'u': 'u_p', 'v': 'v_p', 'w': 'w_p', 'th': 'th_p', 'tk': 't_p', 'p': 'pres_p', 'exner': 'pi_p',
    'rho': 'rho_p', 'qv': 'qv_p', 'qc': 'qc_p', 'qi': 'qi_p', 'qs': 'qs_p', 'qni': 'ni_p', 'qnc': 'nc_p',
    'qnifa': 'nifa_p', 'qnwfa': 'nwfa_p', 'rthraten': 'rthraten_p', 'pint': 'pres2_hyd_p',
    'pblh': 'hpbl_p', 'kpbl': 'kpbl_p', 'cldfra_bl': 'cldfrabl_p', 'qc_bl': 'qcbl_p', 'qi_bl': 'qibl_p',
    'maxwidth': 'maxwidthbl_p', 'maxmf': 'maxmfbl_p', 'ztop_plume': 'zbl_plume_p', 'excess_h': 'excessh_p',
    'excess_q': 'excessq_p', 'qke': 'qke_p', 'qke_adv': 'qkeadv_p', 'tsq': 'tsq_p', 'qsq': 'qsq_p', 'cov': 'cov_p',
    'el_pbl': 'elpbl_p', 'rublten': 'rublten_p', 'rvblten': 'rvblten_p', 'rthblten': 'rthblten_p',
    'rqvblten': 'rqvblten_p', 'rqcblten': 'rqcblten_p', 'rqiblten': 'rqiblten_p', 'rqsblten': 'rqsblten_p',
    'rqniblten': 'rniblten_p', 'rqncblten': 'rncblten_p', 'rqnwfablten': 'rnwfablten_p',
    'rqnifablten': 'rnifablten_p', 'edmf_a': 'edmfa_p', 'edmf_w': 'edmfw_p', 'edmf_qt': 'edmfqt_p',
    'edmf_thl': 'edmfthl_p', 'edmf_ent': 'edmfent_p', 'edmf_qc': 'edmfqc_p', 'sub_thl': 'subthl_p',
    'sub_sqv': 'subqv_p', 'det_thl': 'detthl_p', 'det_sqv': 'detqv_p', 'exch_h': 'kzh_p', 'exch_m': 'kzm_p',
    'dqke': 'dqke_p', 'qwt': 'qwt_p', 'qshear': 'qshear_p', 'qbuoy': 'qbuoy_p', 'qdiss': 'qdiss_p',
    'sh3d': 'sh3d_p', 'sm3d': 'sm3d_p', 'pattern_spp': 'pattern_spp_pbl', 'chem3d': 'chem_p', 'vd3d': 'ddvel_p',
    'settle3d': 'tend_chem_settle_p', 'frp_mean': 'frp_out_p', 'emis_ant_no': 'aero_emis_for_enhmix_p',
}

# RRFS 44c0849 mpas_atmphys_driver_pbl.F:1637-1639, added downdraft diagnostics.
EXTRA_SURFACE_FIELDS = ('maxwidth_dd', 'maxmf_dd', 'maxtkeprod_dd', 'cldtop_cooling', 'ent_eff')
NATIVE_ARRAY_BINDINGS.update({name: name + '_p' for name in EXTRA_SURFACE_FIELDS if name != 'maxtkeprod_dd'})
NATIVE_ARRAY_BINDINGS['maxtkeprod'] = 'maxtkeprod_dd_p'


def native_driver_arguments(prepared, metadata, config, *, itimestep):
    """The v6 driver call on the 44c0849 host (noaa/develop de016e0 driver_pbl.F:1748-1828).

    restart = config_do_restart, cycling = config_do_DAcycling, initflag as the 44c0849 host sets it
    (driver_pbl.F:1381-1391). Preserve full i,k,j allocations and optional absence.
    """
    values = {name: metadata[name] for name in (
        'ids','ide','jds','jde','kds','kde','ims','ime','jms','jme','kms','kme','its','ite','jts','jte','kts','kte')}
    values.update({arg: metadata[host] for host, arg in HOST_FLAGS.items()})
    values.update(option_arguments(config))
    restart = bool(config['config_do_restart'])
    values.update(delt=metadata['dt_pbl'], restart=restart,
                  cycling=bool(config['config_do_DAcycling']),
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
    """Strict precompiled wrapper for this module (its own packet key, bl_mynnedmf_v611)."""
    import hashlib
    import json
    from pathlib import Path
    from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
    from woof.hex.mynn_glibc_math import mynn_glibc_source
    image = Path(packet['image']).read_bytes()
    metadata = json.loads(Path(packet['metadata']).read_text())
    from woof.hex.cuda_backend.source_identity import mynn_source_matches
    source = CUDA_FTZ_HELPERS + mynn_glibc_source() + ARRAY_SUPPORT + MODULE_CUDA
    from woof.hip_backend import requested,compile_source
    if requested() == 'hip':
        if (not mynn_source_matches(source, metadata['source_sha256'])
                or metadata['image_sha256'] != hashlib.sha256(image).hexdigest()):
            raise RuntimeError('MYNN resource source or image digest differs')
        return compile_source(source,_hex_numerics.options(CUDA_OPTIONS),filename='rrfs_mynnedmf_v611.cu')[0]
    options = list(_hex_numerics.options(CUDA_OPTIONS)) + ['--gpu-architecture=' + _image_target(xp=device)]
    if (not mynn_source_matches(source, metadata['source_sha256'])
            or metadata['image_sha256'] != hashlib.sha256(image).hexdigest()
            or metadata['options'] != options):
        raise RuntimeError('MYNN CUDA resource differs from the strict source, image or device architecture')
    return image


def bind_native_pools(provider, call):
    """Hydrate the 44c0849 host's PBL arrays for the v6 driver (bl_mynnedmf_c6d08709.bind_native_pools plus pint)."""
    from . import driver_rrfs as driver
    import sys
    from .resident_pbl_rrfs import resident_slot
    slot = resident_slot(provider, call, module=sys.modules[__name__])
    if slot is not None:
        return {'mynnedmf_native': {'resident_slot': slot}}
    native, cp = provider.native, provider.cp
    metadata = {}
    for name in HOST_SCALARS:
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
    """Full PBL transaction (bl_mynnedmf_c6d08709.run_slot) with the v6 driver."""
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
    from ..deferred_checks_v842 import defer
    defer((result['errflg'] != 0).astype(cp.int32), 'MYNN-EDMF CUDA wrapper returned a native error flag')
    pools = driver.pbl_to_mpas(**common, prepared=prepared)
    native = context['native']
    for name, value in prepared.items():
        native.put_variable(name, value, lower=context['lower'][name])
    for pool, fields in pools.items():
        for name, value in fields.items():
            native.put_field(pool, name, value, level=1,
                             nlevels=native._fields[pool, name][2])
    return result


# WOOF_HEX_AD_PARAMS (default unset: nothing changes) runs the forward at chosen values of the carried constants.
from .. import ad_params_override as _ad_params_override
_ad_params_override.patch_module(globals(), "mynnedmf")
