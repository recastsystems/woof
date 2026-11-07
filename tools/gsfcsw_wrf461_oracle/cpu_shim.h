// Test compiler shim. No runtime physics or math implementation is included.
#include <cmath>
#define GSFC_HOST_CHECK 1
#define __device__
#define __global__
#define gfk_log logf
#define gfk_exp expf
#define gfk_pow powf
#define __fsqrt_rn sqrtf
struct gsfc_dim { int x; };
static gsfc_dim blockIdx={0},blockDim={1},threadIdx={0};
