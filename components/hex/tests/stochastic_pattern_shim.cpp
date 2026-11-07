// Unit-test ABI for actual CUDA-source AR(1), never a production backend.
#include <math.h>
#define __device__
#define __global__
#define __forceinline__ inline
struct uint4 { unsigned int x, y, z, w; };
inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
    return {x, y, z, w};
}
struct { unsigned int x; } blockDim, blockIdx, threadIdx;
inline unsigned int atomicAdd(unsigned int *a, unsigned int v) { unsigned int o=*a; *a+=v; return o; }
#include "../../../woof/hex/stochastic.cu"

extern "C" void test_ar1(float *coefficients, const float *weights, int count,
                         float phi, float innovation,
                         unsigned long long seed, unsigned long long draw) {
    blockDim.x=128;
    for (blockIdx.x=0; blockIdx.x<unsigned((count+127)/128); ++blockIdx.x)
        for (threadIdx.x=0; threadIdx.x<blockDim.x; ++threadIdx.x)
            ar1(coefficients,weights,count,phi,innovation,seed,draw);
}
