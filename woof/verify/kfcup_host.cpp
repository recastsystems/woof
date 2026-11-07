// The KF-CuP CPU reference: gpuwm/core/kernels/kfcup.cu compiled as plain
// C++ and run column by column, so the EXACT source NVRTC compiles is graded
// against the WRF capture on a machine with no GPU.
//
// The kernel source is not included from disk here: gpuwm/verify/kfcup_host.py
// writes the string gpuwm.core.kernels.module_source("kfcup") -- the preamble,
// the allow-listed glibc_flt32.cuh header and kfcup.cu, assembled exactly as
// the loader hands them to NVRTC -- to kfcup_module_source.cuh beside this
// file, and the build includes that.  A harness that assembled a different
// string would be grading a different kernel.
//
// Arithmetic: x86-64 SSE binary32/binary64 is IEEE round-to-nearest per
// operation; built with -O2 -ffp-contract=off and without -ffast-math there
// is no contraction and no reassociation, which is the device's strict arm
// (--fmad=false --ftz=false).  Subnormals are preserved on both.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <vector>

#define __device__
#define __global__
#define __forceinline__ inline
#define __constant__ static const
#define __restrict__

struct KcS3 { unsigned x, y, z; };
static KcS3 blockIdx = {0, 0, 0};
static KcS3 blockDim = {1, 1, 1};
static KcS3 threadIdx = {0, 0, 0};

static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline double __fma_rn(double a, double b, double c) { return std::fma(a, b, c); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }
static inline float __uint_as_float(unsigned int u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline unsigned int __float_as_uint(float f) { unsigned int u; std::memcpy(&u, &f, 4); return u; }
static inline float __int_as_float(int i) { float f; std::memcpy(&f, &i, 4); return f; }
static inline int __float_as_int(float f) { int i; std::memcpy(&i, &f, 4); return i; }
static inline long long __double_as_longlong(double d) { long long l; std::memcpy(&l, &d, 8); return l; }
static inline double __longlong_as_double(long long l) { double d; std::memcpy(&d, &l, 8); return d; }
static inline float __double2float_rn(double d) { return (float)d; }
static inline int atomicOr(int* a, int v) { int o = *a; *a = o | v; return o; }
using std::lroundf;

#include "kfcup_module_source.cuh"

static std::vector<float> kc_host_ws;

// One column at a time: threadIdx.x carries the column's lane within a
// one-warp workspace, so the workspace is ONE warp's region and is reused
// column after column -- the device reuses it across tiles the same way.
extern "C" int kfcup_host_run(
    float* lev_in, const float* sca_in, const int* kpbl_in, float* held, int* iheld,
    float* lev_out, const float* ttab, const float* qstab, const float* the0k,
    const float* alu, float plutop, float rdpr, float rdthk, float dt, float dx, int ntst,
    int numbins, float thbinsize, float rbinsize, float mindeepfreq, float minshallowfreq,
    int flags, int nz, int ncol)
{
    const size_t per_thread = (size_t)KFCUP_WS_SLOTS * (size_t)(nz + 1)
        + (size_t)numbins * (size_t)numbins + 2 * (size_t)numbins;
    const size_t need = per_thread * KC_LANES;
    if (kc_host_ws.size() < need) kc_host_ws.assign(need, 0.0f);
    int status = 0;
    blockDim.x = KC_LANES;
    blockIdx.x = 0;
    for (int col = 0; col < ncol; ++col) {
        threadIdx.x = (unsigned)(col % KC_LANES);
        kfcup_column(lev_in, sca_in, kpbl_in, held, iheld, lev_out, ttab, qstab, the0k, alu,
                     plutop, rdpr, rdthk, dt, dx, ntst, numbins, thbinsize, rbinsize,
                     mindeepfreq, minshallowfreq, flags, kc_host_ws.data(), &status,
                     nz, ncol, col - (int)threadIdx.x);
    }
    threadIdx.x = 0;
    blockDim.x = 1;
    return status;
}

extern "C" void kfcup_host_history(float* avg, const float* w, int stepcu, int nz, int ncol) {
    const float tst=(float)(2*stepcu);
    for (int i=0; i<nz*ncol; ++i) avg[i]=kc_history(avg[i],w[i],w[i+ncol],tst);
}
extern "C" void kfcup_host_merge(
    float* cf, float* qc, float* cup, const int* flags, const float* shall,
    const float* cubot, const float* cutop, const int* k, int icloud, int forced, int n) {
    for (int i=0; i<n; ++i)
        kc_radiation_merge(icloud, forced != 0, flags[i], shall[i], cubot[i], cutop[i],
                           k[i], cf[i], qc[i], cup[i]);
}
