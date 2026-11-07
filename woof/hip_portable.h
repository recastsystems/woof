#ifndef GPUWM_HIP_PORTABLE_H
#define GPUWM_HIP_PORTABLE_H
#ifndef __HIPCC_RTC__
#include <hip/hip_runtime.h>
#endif
// HIP's NVIDIA hip_common.h defines this as an empty token; CuPy tests its
// numeric value. Normalize the compiler marker after including that header.
#if defined(__HIP_PLATFORM_NVIDIA__)
#undef __HIPCC__
#define __HIPCC__ 1
#endif
#define GPUWM_LOGICAL_WARP_SIZE 32
#if defined(__HIP_PLATFORM_AMD__)
#define GPUWM_CUB_NAMESPACE hipcub
// CUDA occupancy minima are not portable across register files and CUs.
// Preserve the required block limit while allowing AMD register allocation.
#define GPUWM_LAUNCH_BOUNDS(max_threads, ...) __launch_bounds__(max_threads)
// Per-operation contract(off) survives inlining without private-memory spills.
__device__ __forceinline__ float gpuwm_fadd_rn(float a, float b) {
#pragma clang fp contract(off)
    return a+b;
}
__device__ __forceinline__ float gpuwm_fsub_rn(float a, float b) {
#pragma clang fp contract(off)
    return a-b;
}
__device__ __forceinline__ float gpuwm_fmul_rn(float a, float b) {
#pragma clang fp contract(off)
    return a*b;
}
__device__ __forceinline__ float gpuwm_fdiv_rn(float a, float b) {
#pragma clang fp contract(off)
    return a/b;
}
__device__ __forceinline__ float gpuwm_fsqrt_rn(float a) { return __builtin_sqrtf(a); }
__device__ __forceinline__ double gpuwm_dadd_rn(double a, double b) {
#pragma clang fp contract(off)
    return a+b;
}
__device__ __forceinline__ double gpuwm_dsub_rn(double a, double b) {
#pragma clang fp contract(off)
    return a-b;
}
__device__ __forceinline__ double gpuwm_dmul_rn(double a, double b) {
#pragma clang fp contract(off)
    return a*b;
}
__device__ __forceinline__ double gpuwm_ddiv_rn(double a, double b) {
#pragma clang fp contract(off)
    return a/b;
}
__device__ __forceinline__ double gpuwm_dsqrt_rn(double a) { return __builtin_sqrt(a); }
__device__ __forceinline__ unsigned gpuwm_lane() {
    return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0));
}
__device__ __forceinline__ unsigned gpuwm_ballot_sync(unsigned mask, int pred) {
    return (unsigned)(__ballot(pred) >> (gpuwm_lane() & 32u)) & mask;
}
__device__ __forceinline__ unsigned gpuwm_activemask() {
    return (unsigned)(__ballot(1) >> (gpuwm_lane() & 32u));
}
__device__ __forceinline__ int gpuwm_any_sync(unsigned mask, int p) { return gpuwm_ballot_sync(mask,p)!=0; }
__device__ __forceinline__ int gpuwm_all_sync(unsigned mask, int p) { return gpuwm_ballot_sync(mask,p)==mask; }
template<class T> __device__ __forceinline__ T gpuwm_ldcg(const T* p) { return *p; }
// HIP supplies the correct clock instruction on CDNA and RDNA. Its rate is
// device dependent: the module loader writes hipDeviceAttributeWallClockRate.
extern "C" { __device__ __constant__ unsigned long long gpuwm_wall_clock_khz=100000ull; }
__device__ __forceinline__ unsigned long long gpuwm_globaltimer_ns() {
    unsigned long long ticks=(unsigned long long)wall_clock64();
    unsigned long long rate=gpuwm_wall_clock_khz;
    return (ticks/rate)*1000000ull+((ticks%rate)*1000000ull)/rate;
}
__device__ __forceinline__ void gpuwm_syncwarp(unsigned mask=0xffffffffu) {
    // Wave instruction execution is converged. The fence orders LDS/global
    // accesses at CUDA's logical warp synchronization points without turning
    // a partial-warp call into an illegal block-wide barrier.
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
    __builtin_amdgcn_wave_barrier();
}
template<class T> __device__ __forceinline__ T gpuwm_shfl_sync(unsigned, T v, int lane, int width=32) { return __shfl(v,lane,width); }
template<class T> __device__ __forceinline__ T gpuwm_shfl_up_sync(unsigned, T v, unsigned d, int width=32) { return __shfl_up(v,d,width); }
template<class T> __device__ __forceinline__ T gpuwm_shfl_down_sync(unsigned, T v, unsigned d, int width=32) { return __shfl_down(v,d,width); }
template<class T> __device__ __forceinline__ T gpuwm_shfl_xor_sync(unsigned, T v, int d, int width=32) { return __shfl_xor(v,d,width); }
#else
#define GPUWM_CUB_NAMESPACE cub
#define GPUWM_LAUNCH_BOUNDS(...) __launch_bounds__(__VA_ARGS__)
#define gpuwm_fadd_rn __fadd_rn
#define gpuwm_fsub_rn __fsub_rn
#define gpuwm_fmul_rn __fmul_rn
#define gpuwm_fdiv_rn __fdiv_rn
#define gpuwm_fsqrt_rn __fsqrt_rn
#define gpuwm_dadd_rn __dadd_rn
#define gpuwm_dsub_rn __dsub_rn
#define gpuwm_dmul_rn __dmul_rn
#define gpuwm_ddiv_rn __ddiv_rn
#define gpuwm_dsqrt_rn __dsqrt_rn
#define gpuwm_ballot_sync __ballot_sync
#define gpuwm_activemask __activemask
#define gpuwm_syncwarp __syncwarp
#define gpuwm_any_sync __any_sync
#define gpuwm_all_sync __all_sync
#define gpuwm_ldcg __ldcg
__device__ __forceinline__ unsigned long long gpuwm_globaltimer_ns() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
template<class T> __device__ __forceinline__ T gpuwm_shfl_sync(unsigned m, T v, int lane, int width=32) { return __shfl_sync(m,v,lane,width); }
template<class T> __device__ __forceinline__ T gpuwm_shfl_up_sync(unsigned m, T v, unsigned d, int width=32) { return __shfl_up_sync(m,v,d,width); }
template<class T> __device__ __forceinline__ T gpuwm_shfl_down_sync(unsigned m, T v, unsigned d, int width=32) { return __shfl_down_sync(m,v,d,width); }
template<class T> __device__ __forceinline__ T gpuwm_shfl_xor_sync(unsigned m, T v, int d, int width=32) { return __shfl_xor_sync(m,v,d,width); }
#endif
#endif
