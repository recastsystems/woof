"""Emit test-only CPU compilation of the exact CUDA expressions.

The host bridge replaces device elementary functions with the pinned host
libm. No CPU science backend or production fallback is exposed by the port.
"""
from pathlib import Path
import sys
from woof.hex.mpasphys.ugwp_oro_v842 import CUDA_SOURCE

preamble=r'''
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <algorithm>
using std::min; using std::max;
#define __global__
#define UGWP_LEVELS 55
#define __fsqrt_rn(x) sqrtf(x)
#define __double2float_rn(x) ((float)(x))
#define __fsub_rn(a,b) ((a)-(b))
#define atomicExch(p,v) (*(p)=(v))
#define gfk_pow(x,y) powf(x,y)
#define gfk_exp(x) expf(x)
#define glibc_sinf(x) sinf(x)
// Exact glibc float atan2 in the host-bridged witness, as native calls it.
#define atan2(y,x) ((double)atan2f((float)(y),(float)(x)))
struct dim3 { int x; };
static dim3 blockDim{1},blockIdx{0},threadIdx{0};
'''
wrapper=r'''
extern "C" void host_run(int nc,int nz,const float *a,const float *p,
                         const float *s,const int *kpbl,const float *c,
                         int diags,float *o,float *so,int *err) {
    *err=0;
    for(int i=0;i<nc;++i) {
        blockIdx.x=i;
        ugwp_oro_run(a,p,s,kpbl,o,so,err,c[0],c[1],c[2],c[3],c[4],c[5],c[6],nz,nc,diags);
    }
}
'''
Path(sys.argv[1]).write_text(preamble+CUDA_SOURCE+wrapper)
