// Host harness: compile the EXACT source string nvrtc receives for
// gpuwm/core/kernels/bmj.cu (gpuwm.core.kernels.module_source("bmj"), written
// to bmj_assembled.cu by bmj_host_parity.py) as plain C++, so the kernel can
// be graded against the WRF column oracle on a CPU box.  The GPU run is the
// gate of record (tests/test_bmj_wrf461_parity.py); this is the fast loop.
//
// Fidelity: x86-64 SSE float/double arithmetic is IEEE round-to-nearest per
// operation; built with -ffp-contract=off and no -ffast-math there is no
// fusing and no reassociation, so each FADD/FMUL/FDIV maps to the same
// correctly rounded operation the device intrinsic pins.  The one class of
// difference left is subnormals (the host keeps them), which the GPU run
// measures.  Same scheme as tools/gf_wrf461_oracle/gf_host_harness.cpp.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

#define __device__
#define __global__
#define __constant__ static const
#define __forceinline__ inline
#define __restrict__

struct BmjS3 { unsigned x, y, z; };
static BmjS3 blockIdx = {0, 0, 0};
static BmjS3 blockDim = {1, 1, 1};
static BmjS3 threadIdx = {0, 0, 0};

static inline double __fma_rn(double a, double b, double c) { return std::fma(a, b, c); }
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }
static inline float __uint_as_float(unsigned int u)
{ float f; std::memcpy(&f, &u, 4); return f; }
static inline unsigned int __float_as_uint(float f)
{ unsigned int u; std::memcpy(&u, &f, 4); return u; }
static inline float __int_as_float(int i)
{ float f; std::memcpy(&f, &i, 4); return f; }
static inline long long __double_as_longlong(double d)
{ long long l; std::memcpy(&l, &d, 8); return l; }
static inline double __longlong_as_double(long long l)
{ double d; std::memcpy(&d, &l, 8); return d; }
static inline float __double2float_rn(double d) { return (float)d; }
static inline int __float2int_rn(float f) { return (int)lrintf(f); }
static inline int min(int a, int b) { return a < b ? a : b; }

#include "bmj_assembled.cu"

extern "C" void host_bmj_init_tables(float *tab, float cp, float rd)
{
    static std::vector<float> scratch;
    const int block = 160;
    scratch.assign((size_t)3 * block * 7 * BMJ_JTBQ, 0.0f);
    blockDim.x = block;
    for (unsigned fam = 0; fam < 3; ++fam) {
        blockIdx.x = fam;
        for (unsigned t = 0; t < (unsigned)block; ++t) {
            threadIdx.x = t;
            bmj_init_tables(tab, scratch.data(), cp, rd);
        }
    }
    blockIdx.x = 0; threadIdx.x = 0; blockDim.x = 1;
}

extern "C" void host_bmj_column(
    const float *t, const float *qv, const float *pmid, const float *pi,
    const float *rho, const float *dz8w, const float *pint,
    const float *xland, float *cldefi, const float *tab,
    float *rthcuten, float *rqvcuten, float *ccldfra, float *qcconv,
    float *qiconv, float *raincv, float *pratec, float *cutop, float *cubot,
    float *convcld, int *path, float dt, int stepcu, int rad_feedback,
    int ncol, int nz)
{
    // One column at a time, each with its own lane of a one-wide tile, so
    // the workspace is BMJ_NSLOTS * nz floats and reused (the device reuses
    // it across tiles the same way).
    std::vector<float> ws((size_t)BMJ_NSLOTS * nz, 0.0f);
    blockDim.x = 1; blockIdx.x = 0; threadIdx.x = 0;
    for (int i = 0; i < ncol; ++i) {
        bmj_column(t, qv, pmid, pi, rho, dz8w, pint, xland, cldefi, tab,
                   ws.data(), rthcuten, rqvcuten, ccldfra, qcconv, qiconv,
                   raincv, pratec, cutop, cubot, convcld, path, dt, stepcu,
                   rad_feedback, i, 1, ncol, nz);
    }
}
